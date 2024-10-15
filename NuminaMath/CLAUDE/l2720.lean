import Mathlib

namespace NUMINAMATH_CALUDE_retiree_benefit_theorem_l2720_272045

/-- Represents a bank customer --/
structure Customer where
  repayment_rate : ℝ
  monthly_income_stability : ℝ
  preferred_deposit_term : ℝ

/-- Represents a bank's financial metrics --/
structure BankMetrics where
  loan_default_risk : ℝ
  deposit_stability : ℝ
  long_term_liquidity : ℝ

/-- Calculates the benefit for a bank based on customer characteristics --/
def calculate_bank_benefit (c : Customer) : ℝ :=
  c.repayment_rate + c.monthly_income_stability + c.preferred_deposit_term

/-- Represents a retiree customer --/
def retiree : Customer where
  repayment_rate := 0.95
  monthly_income_stability := 0.9
  preferred_deposit_term := 5

/-- Represents an average customer --/
def average_customer : Customer where
  repayment_rate := 0.8
  monthly_income_stability := 0.7
  preferred_deposit_term := 2

/-- Theorem stating that offering special rates to retirees is beneficial for banks --/
theorem retiree_benefit_theorem :
  calculate_bank_benefit retiree > calculate_bank_benefit average_customer :=
by sorry

end NUMINAMATH_CALUDE_retiree_benefit_theorem_l2720_272045


namespace NUMINAMATH_CALUDE_age_ratio_in_five_years_l2720_272043

/-- Represents the ages of Sam and Dan -/
structure Ages where
  sam : ℕ
  dan : ℕ

/-- The conditions given in the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.sam - 3 = 2 * (a.dan - 3)) ∧ 
  (a.sam - 7 = 3 * (a.dan - 7))

/-- The future condition we want to prove -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.dan + years) = 2 * (a.sam + years)

/-- The main theorem to prove -/
theorem age_ratio_in_five_years (a : Ages) :
  age_conditions a → future_ratio a 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_in_five_years_l2720_272043


namespace NUMINAMATH_CALUDE_box_volume_l2720_272001

theorem box_volume (x : ℕ+) :
  (5 * x) * (5 * (x + 1)) * (5 * (x + 2)) = 25 * x^3 + 50 * x^2 + 125 * x :=
by sorry

end NUMINAMATH_CALUDE_box_volume_l2720_272001


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l2720_272033

theorem imaginary_part_of_complex_division (i : ℂ) : i * i = -1 → 
  Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l2720_272033


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2720_272010

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  3 * X^4 - 8 * X^3 + 20 * X^2 - 7 * X + 13 = 
  (X^2 + 5 * X - 3) * q + (168 * X^2 + 44 * X + 85) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2720_272010


namespace NUMINAMATH_CALUDE_shoe_shirt_cost_difference_is_three_l2720_272087

/-- The cost difference between a pair of shoes and a shirt -/
def shoe_shirt_cost_difference : ℝ :=
  let shirt_cost : ℝ := 7
  let shoe_cost : ℝ := shirt_cost + shoe_shirt_cost_difference
  let bag_cost : ℝ := (2 * shirt_cost + shoe_cost) / 2
  let total_cost : ℝ := 2 * shirt_cost + shoe_cost + bag_cost
  shoe_shirt_cost_difference

/-- Theorem stating the cost difference between a pair of shoes and a shirt -/
theorem shoe_shirt_cost_difference_is_three :
  shoe_shirt_cost_difference = 3 := by
  sorry

#eval shoe_shirt_cost_difference

end NUMINAMATH_CALUDE_shoe_shirt_cost_difference_is_three_l2720_272087


namespace NUMINAMATH_CALUDE_exists_valid_surname_l2720_272031

/-- Represents the positions of letters in a 6-letter Russian surname --/
structure SurnameLetter where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Conditions for the Russian writer's surname --/
def is_valid_surname (s : SurnameLetter) : Prop :=
  s.first = s.third ∧
  s.second = s.fourth ∧
  s.fifth = s.first + 9 ∧
  s.sixth = s.second + s.fourth - 2 ∧
  3 * s.first = s.second - 4 ∧
  s.first + s.second + s.third + s.fourth + s.fifth + s.sixth = 83

/-- The theorem stating the existence of a valid surname --/
theorem exists_valid_surname : ∃ (s : SurnameLetter), is_valid_surname s :=
sorry

end NUMINAMATH_CALUDE_exists_valid_surname_l2720_272031


namespace NUMINAMATH_CALUDE_number_problem_l2720_272032

theorem number_problem : ∃ x : ℝ, 0.65 * x - 25 = 90 ∧ abs (x - 176.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2720_272032


namespace NUMINAMATH_CALUDE_train_length_l2720_272007

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (t1 t2 p1 p2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : p1 > 0) (h4 : p2 > 0)
  (h5 : (L + p1) / t1 = (L + p2) / t2) : L = 100 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2720_272007


namespace NUMINAMATH_CALUDE_points_on_unit_circle_l2720_272044

theorem points_on_unit_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3 * t / (2 + t^2)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_points_on_unit_circle_l2720_272044


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l2720_272011

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
    a = 9 ∧ b = 12 ∧ c = 15 →
    a^2 + b^2 = c^2 →
    (1/2) * a * b = (1/2) * c * h →
    h = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l2720_272011


namespace NUMINAMATH_CALUDE_sum_product_plus_one_positive_l2720_272005

theorem sum_product_plus_one_positive (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : 
  a * b + b * c + c * a + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_plus_one_positive_l2720_272005


namespace NUMINAMATH_CALUDE_factors_of_30_to_4th_l2720_272063

theorem factors_of_30_to_4th (h : 30 = 2 * 3 * 5) :
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 30^4) (Nat.divisors (30^4))).card = 123 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_30_to_4th_l2720_272063


namespace NUMINAMATH_CALUDE_theatre_seating_l2720_272070

theorem theatre_seating (total_seats : ℕ) (row_size : ℕ) (expected_attendance : ℕ) : 
  total_seats = 225 → 
  row_size = 15 → 
  expected_attendance = 160 → 
  (total_seats - (((expected_attendance + row_size - 1) / row_size) * row_size)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_theatre_seating_l2720_272070


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l2720_272067

/-- Given a hyperbola x^2 + my^2 = 1 passing through the point (-√2, 2),
    the length of its imaginary axis is 4. -/
theorem hyperbola_imaginary_axis_length 
  (m : ℝ) 
  (h : (-Real.sqrt 2)^2 + m * 2^2 = 1) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1) ∧
    2*b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l2720_272067


namespace NUMINAMATH_CALUDE_renatas_final_balance_l2720_272006

/-- Represents the balance and transactions of Renata's day --/
def renatas_day (initial_amount : ℚ) (charity_donation : ℚ) (prize_pounds : ℚ) 
  (slot_loss_euros : ℚ) (slot_loss_pounds : ℚ) (slot_loss_dollars : ℚ)
  (sunglasses_euros : ℚ) (water_pounds : ℚ) (lottery_ticket : ℚ) (lottery_prize : ℚ)
  (meal_euros : ℚ) (coffee_euros : ℚ) : ℚ :=
  let pound_to_dollar : ℚ := 1.35
  let euro_to_dollar : ℚ := 1.10
  let sunglasses_discount : ℚ := 0.20
  let meal_discount : ℚ := 0.30
  
  let balance1 := initial_amount - charity_donation
  let balance2 := balance1 + prize_pounds * pound_to_dollar
  let balance3 := balance2 - slot_loss_euros * euro_to_dollar
  let balance4 := balance3 - slot_loss_pounds * pound_to_dollar
  let balance5 := balance4 - slot_loss_dollars
  let balance6 := balance5 - sunglasses_euros * (1 - sunglasses_discount) * euro_to_dollar
  let balance7 := balance6 - water_pounds * pound_to_dollar
  let balance8 := balance7 - lottery_ticket
  let balance9 := balance8 + lottery_prize
  let lunch_cost := (meal_euros * (1 - meal_discount) + coffee_euros) * euro_to_dollar
  balance9 - lunch_cost / 2

/-- Theorem stating that Renata's final balance is $35.95 --/
theorem renatas_final_balance :
  renatas_day 50 10 50 30 20 15 15 1 1 30 10 3 = 35.95 := by sorry

end NUMINAMATH_CALUDE_renatas_final_balance_l2720_272006


namespace NUMINAMATH_CALUDE_inverse_of_A_l2720_272022

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

theorem inverse_of_A :
  A⁻¹ = !![-(3/2), 7/2; 1, -2] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2720_272022


namespace NUMINAMATH_CALUDE_average_age_decrease_l2720_272028

theorem average_age_decrease (initial_average : ℝ) : 
  let original_total := 10 * initial_average
  let new_total := original_total - 44 + 14
  let new_average := new_total / 10
  initial_average - new_average = 3 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l2720_272028


namespace NUMINAMATH_CALUDE_period_length_divides_totient_l2720_272037

-- Define L(m) as the period length of the decimal expansion of 1/m
def L (m : ℕ) : ℕ := sorry

-- State the theorem
theorem period_length_divides_totient (m : ℕ) (h : Nat.gcd m 10 = 1) : 
  L m ∣ Nat.totient m := by sorry

end NUMINAMATH_CALUDE_period_length_divides_totient_l2720_272037


namespace NUMINAMATH_CALUDE_power_division_result_l2720_272029

theorem power_division_result : 8^15 / 64^7 = 8 := by sorry

end NUMINAMATH_CALUDE_power_division_result_l2720_272029


namespace NUMINAMATH_CALUDE_room_width_calculation_l2720_272008

/-- Given a room with the specified dimensions and paving costs, prove that the width is 3.75 meters -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
    (h1 : length = 5.5)
    (h2 : total_cost = 24750)
    (h3 : rate_per_sqm = 1200) : 
  total_cost / rate_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2720_272008


namespace NUMINAMATH_CALUDE_percentage_increase_l2720_272089

theorem percentage_increase (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 925 →
  z = 250 →
  (y - z) / z = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2720_272089


namespace NUMINAMATH_CALUDE_apple_purchase_remainder_l2720_272086

theorem apple_purchase_remainder (mark_money carolyn_money apple_cost : ℚ) : 
  mark_money = 2/3 →
  carolyn_money = 1/5 →
  apple_cost = 1/2 →
  mark_money + carolyn_money - apple_cost = 11/30 := by
sorry

end NUMINAMATH_CALUDE_apple_purchase_remainder_l2720_272086


namespace NUMINAMATH_CALUDE_order_of_roots_l2720_272040

theorem order_of_roots (m n p : ℝ) : 
  m = (1/3)^(1/5) → n = (1/4)^(1/3) → p = (1/5)^(1/4) → n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l2720_272040


namespace NUMINAMATH_CALUDE_largest_number_in_sequence_l2720_272094

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence : Type := { s : Fin 8 → ℝ // ∀ i j, i < j → s i < s j }

/-- Checks if a subsequence of 4 consecutive numbers is an arithmetic progression -/
def IsArithmeticProgression (s : IncreasingSequence) (start : Fin 5) (d : ℝ) : Prop :=
  ∀ i : Fin 3, s.val (start + i + 1) - s.val (start + i) = d

/-- Checks if a subsequence of 4 consecutive numbers is a geometric progression -/
def IsGeometricProgression (s : IncreasingSequence) (start : Fin 5) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 3, s.val (start + i + 1) / s.val (start + i) = r

/-- The main theorem -/
theorem largest_number_in_sequence (s : IncreasingSequence) 
  (h1 : ∃ start1 : Fin 5, IsArithmeticProgression s start1 4)
  (h2 : ∃ start2 : Fin 5, IsArithmeticProgression s start2 36)
  (h3 : ∃ start3 : Fin 5, IsGeometricProgression s start3) :
  s.val 7 = 126 ∨ s.val 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_largest_number_in_sequence_l2720_272094


namespace NUMINAMATH_CALUDE_second_number_is_22_l2720_272074

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_22_l2720_272074


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2720_272099

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧ 
   (r+7)^2 - k*(r+7) + 8 = 0 ∧ (s+7)^2 - k*(s+7) + 8 = 0) → 
  k = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2720_272099


namespace NUMINAMATH_CALUDE_zeros_bound_l2720_272091

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem zeros_bound (a : ℝ) :
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → (f a x ≠ 0 ∨ f a y ≠ 0 ∨ f a z ≠ 0)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_bound_l2720_272091


namespace NUMINAMATH_CALUDE_min_value_of_t_l2720_272027

theorem min_value_of_t (x y t : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 3 * x + y + x * y - 13 = 0) 
  (h2 : ∃ (t : ℝ), t ≥ 2 * y + x) : 
  ∀ t, t ≥ 2 * y + x → t ≥ 8 * Real.sqrt 2 - 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_t_l2720_272027


namespace NUMINAMATH_CALUDE_factor_expression_l2720_272081

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by sorry

end NUMINAMATH_CALUDE_factor_expression_l2720_272081


namespace NUMINAMATH_CALUDE_prob_one_white_correct_prob_red_given_red_correct_l2720_272059

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing exactly one white ball
def prob_one_white : ℚ := 3/5

-- Define the probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
def prob_red_given_red : ℚ := 3/5

-- Theorem 1: Probability of drawing exactly one white ball
theorem prob_one_white_correct :
  (Nat.choose white_balls 1 * Nat.choose red_balls (balls_drawn - 1)) / 
  Nat.choose total_balls balls_drawn = prob_one_white := by sorry

-- Theorem 2: Probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
theorem prob_red_given_red_correct :
  (red_balls - 1) / (total_balls - 1) = prob_red_given_red := by sorry

end NUMINAMATH_CALUDE_prob_one_white_correct_prob_red_given_red_correct_l2720_272059


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2720_272004

/-- Given a point and a line, this theorem proves that the equation
    x - 2y + 7 = 0 represents a line passing through the given point
    and perpendicular to the given line. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let point : ℝ × ℝ := (-1, 3)
  let given_line := {(x, y) : ℝ × ℝ | 2 * x + y + 3 = 0}
  let perpendicular_line := {(x, y) : ℝ × ℝ | x - 2 * y + 7 = 0}
  (point ∈ perpendicular_line) ∧
  (∀ (v w : ℝ × ℝ), v ∈ given_line → w ∈ given_line → v ≠ w →
    let slope_given := (w.2 - v.2) / (w.1 - v.1)
    let slope_perp := (y - 3) / (x - (-1))
    slope_given * slope_perp = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2720_272004


namespace NUMINAMATH_CALUDE_number_of_boys_in_school_l2720_272018

theorem number_of_boys_in_school (total : ℕ) (boys : ℕ) :
  total = 1150 →
  (total - boys : ℚ) = (boys : ℚ) * total / 100 →
  boys = 92 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_school_l2720_272018


namespace NUMINAMATH_CALUDE_fraction_equality_l2720_272080

theorem fraction_equality (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = -1) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2720_272080


namespace NUMINAMATH_CALUDE_stock_price_change_l2720_272051

def down_limit : ℝ := 0.9
def up_limit : ℝ := 1.1
def num_limits : ℕ := 3

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  initial_price * (down_limit ^ num_limits) * (up_limit ^ num_limits) < initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_change_l2720_272051


namespace NUMINAMATH_CALUDE_women_meeting_point_l2720_272077

/-- Represents the distance walked by the woman starting from point B -/
def distance_B (h : ℕ) : ℚ :=
  h * (h + 3) / 2

/-- Represents the total distance walked by both women -/
def total_distance (h : ℕ) : ℚ :=
  3 * h + distance_B h

theorem women_meeting_point :
  ∃ (h : ℕ), h > 0 ∧ total_distance h = 60 ∧ distance_B h - 3 * h = 6 := by
  sorry

end NUMINAMATH_CALUDE_women_meeting_point_l2720_272077


namespace NUMINAMATH_CALUDE_container_capacity_l2720_272024

theorem container_capacity (C : ℝ) : 0.40 * C + 14 = 0.75 * C → C = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2720_272024


namespace NUMINAMATH_CALUDE_binomial_divisibility_sequence_l2720_272065

theorem binomial_divisibility_sequence :
  ∃ n : ℕ, n > 2003 ∧ ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 2003 → (n.choose i ∣ n.choose j) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_sequence_l2720_272065


namespace NUMINAMATH_CALUDE_kate_pen_purchase_l2720_272088

theorem kate_pen_purchase (pen_cost : ℝ) (kate_money : ℝ) : 
  pen_cost = 30 → kate_money = pen_cost / 3 → pen_cost - kate_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_kate_pen_purchase_l2720_272088


namespace NUMINAMATH_CALUDE_product_sum_difference_theorem_l2720_272017

theorem product_sum_difference_theorem (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 2688) 
  (h4 : x = 84) : 
  (x + y) - (x - y) = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_theorem_l2720_272017


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2720_272095

theorem smaller_number_in_ratio (a b d u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : u > 0) (h4 : v > 0)
  (h5 : u / v = b / a) (h6 : u + v = d) : 
  min u v = a * d / (a + b) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2720_272095


namespace NUMINAMATH_CALUDE_four_letter_word_count_l2720_272012

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The length of the word -/
def word_length : ℕ := 4

theorem four_letter_word_count : 
  alphabet_size * vowel_count * alphabet_size = 3380 := by
  sorry

#check four_letter_word_count

end NUMINAMATH_CALUDE_four_letter_word_count_l2720_272012


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2720_272075

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2720_272075


namespace NUMINAMATH_CALUDE_train_crossing_time_l2720_272058

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 431.25)
  (h3 : platform_crossing_time = 39) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 16 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2720_272058


namespace NUMINAMATH_CALUDE_equation_solution_l2720_272093

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 ∧ n = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2720_272093


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2720_272098

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 6 = 0 → 
  x₂^2 - 6*x₂ + 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2720_272098


namespace NUMINAMATH_CALUDE_correct_arrangement_l2720_272092

-- Define the squares
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
def points_to : Square → Square → Prop :=
  fun s1 s2 => match s1, s2 with
    | Square.One, Square.B => True
    | Square.B, Square.E => True
    | Square.E, Square.C => True
    | Square.C, Square.D => True
    | Square.D, Square.A => True
    | Square.A, Square.G => True
    | Square.G, Square.F => True
    | Square.F, Square.Nine => True
    | _, _ => False

-- Define the arrangement
def arrangement : Square → Nat
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7
| Square.One => 1
| Square.Nine => 9

-- Theorem statement
theorem correct_arrangement :
  ∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    ∃ s' : Square, points_to s s' ∧ arrangement s' = arrangement s + 1 :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_l2720_272092


namespace NUMINAMATH_CALUDE_circle_and_line_tangency_l2720_272047

-- Define the line l
def line (x y a : ℝ) : Prop := Real.sqrt 3 * x - y - a = 0

-- Define the circle C in polar form
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the circle C in Cartesian form
def circle_cartesian (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_and_line_tangency :
  -- Part I: Equivalence of polar and Cartesian forms of circle C
  (∀ x y ρ θ : ℝ, circle_polar ρ θ ↔ circle_cartesian x y) ∧
  -- Part II: Tangency condition
  (∀ a : ℝ, (∃ x y : ℝ, line x y a ∧ circle_cartesian x y ∧
    (∀ x' y' : ℝ, line x' y' a ∧ circle_cartesian x' y' → x = x' ∧ y = y'))
    ↔ (a = -3 ∨ a = 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_tangency_l2720_272047


namespace NUMINAMATH_CALUDE_initial_number_problem_l2720_272035

theorem initial_number_problem (x : ℝ) : 8 * x - 4 = 2.625 → x = 0.828125 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_problem_l2720_272035


namespace NUMINAMATH_CALUDE_c_range_l2720_272034

theorem c_range (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l2720_272034


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l2720_272056

/-- The probability of drawing four marbles of the same color from a box containing
    3 orange, 7 purple, and 5 green marbles, without replacement. -/
theorem same_color_marble_probability :
  let total_marbles : ℕ := 3 + 7 + 5
  let orange_marbles : ℕ := 3
  let purple_marbles : ℕ := 7
  let green_marbles : ℕ := 5
  let draw_count : ℕ := 4
  
  (orange_marbles.choose draw_count +
   purple_marbles.choose draw_count +
   green_marbles.choose draw_count : ℚ) /
  (total_marbles.choose draw_count : ℚ) = 210 / 1369 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l2720_272056


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l2720_272062

theorem square_plus_reciprocal_squared (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 7 → x^4 + (1/x^4) = 47 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l2720_272062


namespace NUMINAMATH_CALUDE_twelve_solutions_for_quadratic_diophantine_l2720_272053

theorem twelve_solutions_for_quadratic_diophantine (n : ℕ) (x y : ℕ+) 
  (h1 : x^2 - x*y + y^2 = n)
  (h2 : x ≠ y)
  (h3 : x ≠ 2*y)
  (h4 : y ≠ 2*x) :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → p.1^2 - p.1*p.2 + p.2^2 = n) ∧ S.card ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_twelve_solutions_for_quadratic_diophantine_l2720_272053


namespace NUMINAMATH_CALUDE_johann_mail_delivery_l2720_272002

theorem johann_mail_delivery (total : ℕ) (friend_delivery : ℕ) (num_friends : ℕ) :
  total = 180 →
  friend_delivery = 41 →
  num_friends = 2 →
  total - (friend_delivery * num_friends) = 98 :=
by sorry

end NUMINAMATH_CALUDE_johann_mail_delivery_l2720_272002


namespace NUMINAMATH_CALUDE_yannas_baking_problem_l2720_272015

/-- Yanna's baking problem -/
theorem yannas_baking_problem (morning_butter_cookies morning_biscuits afternoon_butter_cookies afternoon_biscuits : ℕ) 
  (h1 : morning_butter_cookies = 20)
  (h2 : afternoon_butter_cookies = 10)
  (h3 : afternoon_biscuits = 20)
  (h4 : morning_biscuits + afternoon_biscuits = morning_butter_cookies + afternoon_butter_cookies + 30) :
  morning_biscuits = 40 := by
  sorry

end NUMINAMATH_CALUDE_yannas_baking_problem_l2720_272015


namespace NUMINAMATH_CALUDE_disjunction_false_l2720_272097

-- Define proposition p
def prop_p (a b : ℝ) : Prop := (a * b > 0) → (|a| + |b| > |a + b|)

-- Define proposition q
def prop_q (a b c : ℝ) : Prop := (c > a^2 + b^2) → (c > 2*a*b)

-- Theorem statement
theorem disjunction_false :
  ¬(∀ a b : ℝ, prop_p a b ∨ ¬(∀ c : ℝ, prop_q a b c)) :=
sorry

end NUMINAMATH_CALUDE_disjunction_false_l2720_272097


namespace NUMINAMATH_CALUDE_car_price_difference_l2720_272013

/-- Proves the difference in price between the old and new car -/
theorem car_price_difference
  (sale_percentage : ℝ)
  (additional_amount : ℝ)
  (new_car_price : ℝ)
  (h1 : sale_percentage = 0.8)
  (h2 : additional_amount = 4000)
  (h3 : new_car_price = 30000)
  (h4 : sale_percentage * (new_car_price - additional_amount) + additional_amount = new_car_price) :
  (new_car_price - additional_amount) / sale_percentage - new_car_price = 2500 := by
sorry

end NUMINAMATH_CALUDE_car_price_difference_l2720_272013


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l2720_272057

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111011_equals_379 :
  binary_to_decimal [true, true, false, true, true, true, true, false, true] = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l2720_272057


namespace NUMINAMATH_CALUDE_part_one_part_two_l2720_272021

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 - (a + 1) * x + b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc 2 3, f x a (-1) = 0) →
  (1/2 : ℝ) ≤ a ∧ a ≤ 5/3 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (∀ a ∈ Set.Icc 2 3, f x a a < 0) →
  1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2720_272021


namespace NUMINAMATH_CALUDE_map_scale_l2720_272000

theorem map_scale (map_length : ℝ) (real_distance : ℝ) (query_length : ℝ) :
  map_length > 0 →
  real_distance > 0 →
  query_length > 0 →
  (15 : ℝ) * real_distance = 45 * map_length →
  25 * real_distance = 75 * map_length := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l2720_272000


namespace NUMINAMATH_CALUDE_binomial_30_3_l2720_272003

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2720_272003


namespace NUMINAMATH_CALUDE_square_area_from_octagon_l2720_272041

theorem square_area_from_octagon (side_length : ℝ) (octagon_area : ℝ) : 
  side_length > 0 →
  octagon_area = 7 * (side_length / 3)^2 →
  octagon_area = 105 →
  side_length^2 = 135 :=
by
  sorry

#check square_area_from_octagon

end NUMINAMATH_CALUDE_square_area_from_octagon_l2720_272041


namespace NUMINAMATH_CALUDE_modulus_of_z_l2720_272050

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : z_condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2720_272050


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l2720_272090

def purple_eggs : ℕ := 30
def yellow_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket : 
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧ 
  purple_eggs % n = 0 ∧ 
  yellow_eggs % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (m ≥ min_eggs_per_basket ∧ 
     purple_eggs % m = 0 ∧ 
     yellow_eggs % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l2720_272090


namespace NUMINAMATH_CALUDE_unique_solution_ABCD_l2720_272025

/-- Represents a base-5 number with two digits --/
def Base5TwoDigit (a b : Nat) : Nat := 5 * a + b

/-- Represents a base-5 number with one digit --/
def Base5OneDigit (a : Nat) : Nat := a

/-- Represents a base-5 number with two identical digits --/
def Base5TwoSameDigit (a : Nat) : Nat := 5 * a + a

theorem unique_solution_ABCD :
  ∀ A B C D : Nat,
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) →
  (A < 5 ∧ B < 5 ∧ C < 5 ∧ D < 5) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (Base5TwoDigit A B + Base5OneDigit C = Base5TwoDigit D 0) →
  (Base5TwoDigit A B + Base5TwoDigit B A = Base5TwoSameDigit D) →
  A = 4 ∧ B = 1 ∧ C = 4 ∧ D = 4 := by
  sorry

#check unique_solution_ABCD

end NUMINAMATH_CALUDE_unique_solution_ABCD_l2720_272025


namespace NUMINAMATH_CALUDE_inscribed_squares_area_l2720_272030

/-- Given three squares inscribed in right triangles with areas A, M, and N,
    where M = 5 and N = 12, prove that A = 17 + 4√15 -/
theorem inscribed_squares_area (A M N : ℝ) (hM : M = 5) (hN : N = 12) :
  A = (Real.sqrt M + Real.sqrt N) ^ 2 →
  A = 17 + 4 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_l2720_272030


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2720_272023

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2720_272023


namespace NUMINAMATH_CALUDE_ellipse_and_intersection_l2720_272009

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * y - 2 * x - 2 = 0

theorem ellipse_and_intersection :
  -- Given conditions
  (ellipse_C 0 2) ∧ 
  (ellipse_C (1/2) (Real.sqrt 3)) ∧
  -- Prove the following
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 + y^2/4 = 1) ∧ 
  (ellipse_C (-1) 0 ∧ line (-1) 0) ∧
  (ellipse_C (1/2) (Real.sqrt 3) ∧ line (1/2) (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_intersection_l2720_272009


namespace NUMINAMATH_CALUDE_actual_time_greater_than_planned_l2720_272014

/-- Proves that the actual running time is greater than the planned time under given conditions -/
theorem actual_time_greater_than_planned (a V : ℝ) (h1 : a > 0) (h2 : V > 0) : 
  (a / (1.25 * V) / 2 + a / (0.8 * V) / 2) > a / V := by
  sorry

#check actual_time_greater_than_planned

end NUMINAMATH_CALUDE_actual_time_greater_than_planned_l2720_272014


namespace NUMINAMATH_CALUDE_arithmetic_progression_log_range_l2720_272055

theorem arithmetic_progression_log_range (x y : ℝ) : 
  (∃ k : ℝ, Real.log 2 - k = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k = Real.log (1 - y)) →
  (y ≥ 7/9 ∧ ∀ M : ℝ, ∃ y' ≥ M, 
    ∃ k' : ℝ, Real.log 2 - k' = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k' = Real.log (1 - y')) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_log_range_l2720_272055


namespace NUMINAMATH_CALUDE_insurance_calculation_l2720_272085

/-- Insurance calculation parameters --/
structure InsuranceParams where
  baseRate : Float
  noTransitionCoeff : Float
  noMedCertCoeff : Float
  assessedValue : Float
  cadasterValue : Float

/-- Calculate adjusted tariff --/
def calcAdjustedTariff (params : InsuranceParams) : Float :=
  params.baseRate * params.noTransitionCoeff * params.noMedCertCoeff

/-- Determine insurance amount --/
def determineInsuranceAmount (params : InsuranceParams) : Float :=
  max params.assessedValue params.cadasterValue

/-- Calculate insurance premium --/
def calcInsurancePremium (amount : Float) (tariff : Float) : Float :=
  amount * tariff

/-- Main theorem --/
theorem insurance_calculation (params : InsuranceParams) 
  (h1 : params.baseRate = 0.002)
  (h2 : params.noTransitionCoeff = 0.8)
  (h3 : params.noMedCertCoeff = 1.3)
  (h4 : params.assessedValue = 14500000)
  (h5 : params.cadasterValue = 15000000) :
  let adjustedTariff := calcAdjustedTariff params
  let insuranceAmount := determineInsuranceAmount params
  let insurancePremium := calcInsurancePremium insuranceAmount adjustedTariff
  adjustedTariff = 0.00208 ∧ 
  insuranceAmount = 15000000 ∧ 
  insurancePremium = 31200 := by
  sorry

end NUMINAMATH_CALUDE_insurance_calculation_l2720_272085


namespace NUMINAMATH_CALUDE_system_equation_solution_l2720_272054

theorem system_equation_solution (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l2720_272054


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l2720_272026

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity and parallelism
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def perpendicular_plane_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (l1 l2 : Line3D) (p : Plane3D)
  (h1 : perpendicular_line_plane l1 p)
  (h2 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (p1 p2 : Plane3D) (l : Line3D)
  (h1 : perpendicular_plane_line p1 l)
  (h2 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l2720_272026


namespace NUMINAMATH_CALUDE_intersection_equals_one_l2720_272096

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem intersection_equals_one : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_one_l2720_272096


namespace NUMINAMATH_CALUDE_binary_to_base_4_conversion_l2720_272049

-- Define the binary number
def binary_num : ℕ := 11011011

-- Define the base 4 number
def base_4_num : ℕ := 3123

-- Theorem stating the equality of the binary and base 4 representations
theorem binary_to_base_4_conversion :
  (binary_num.digits 2).foldl (λ acc d => 2 * acc + d) 0 =
  (base_4_num.digits 4).foldl (λ acc d => 4 * acc + d) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_base_4_conversion_l2720_272049


namespace NUMINAMATH_CALUDE_A_intersect_B_l2720_272061

def A : Set ℝ := {-2, 0, 1, 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem A_intersect_B : A ∩ B = {-2, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2720_272061


namespace NUMINAMATH_CALUDE_train_speed_theorem_l2720_272019

/-- Theorem: Given two trains moving in opposite directions, with one train's speed being 100 kmph,
    lengths of 500 m and 700 m, and a crossing time of 19.6347928529354 seconds,
    the speed of the faster train is 100 kmph. -/
theorem train_speed_theorem (v_slow v_fast : ℝ) (length_slow length_fast : ℝ) (crossing_time : ℝ) :
  v_fast = 100 ∧
  length_slow = 500 ∧
  length_fast = 700 ∧
  crossing_time = 19.6347928529354 ∧
  (length_slow + length_fast) / 1000 / (crossing_time / 3600) = v_slow + v_fast →
  v_fast = 100 := by
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_train_speed_theorem_l2720_272019


namespace NUMINAMATH_CALUDE_focus_directrix_distance_l2720_272064

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Theorem statement
theorem focus_directrix_distance :
  let focus_y := 1 / 16
  let directrix_y := -1 / 16
  |focus_y - directrix_y| = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_focus_directrix_distance_l2720_272064


namespace NUMINAMATH_CALUDE_cost_45_roses_l2720_272060

/-- The cost of a bouquet is directly proportional to the number of roses it contains -/
axiom price_proportional_to_roses (n : ℕ) (price : ℚ) : n > 0 → price > 0 → ∃ k : ℚ, k > 0 ∧ price = k * n

/-- The cost of a bouquet with 15 roses -/
def cost_15 : ℚ := 25

/-- The number of roses in the first bouquet -/
def roses_15 : ℕ := 15

/-- The number of roses in the second bouquet -/
def roses_45 : ℕ := 45

/-- The theorem to prove -/
theorem cost_45_roses : 
  ∃ (k : ℚ), k > 0 ∧ cost_15 = k * roses_15 → k * roses_45 = 75 :=
sorry

end NUMINAMATH_CALUDE_cost_45_roses_l2720_272060


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_with_mean_70_l2720_272048

theorem max_ratio_two_digit_integers_with_mean_70 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 70 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 70 →
  x / y ≤ 99 / 41 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_with_mean_70_l2720_272048


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_3_and_5_l2720_272036

theorem largest_two_digit_multiple_of_3_and_5 : 
  ∃ n : ℕ, n = 90 ∧ 
  n ≥ 10 ∧ n < 100 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 3 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_3_and_5_l2720_272036


namespace NUMINAMATH_CALUDE_vector_perpendicular_and_parallel_l2720_272084

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

theorem vector_perpendicular_and_parallel (k : ℝ) :
  (∀ i : Fin 2, (k * a i + b i) * (a i - 3 * b i) = 0) → k = 19 ∧
  (∃ t : ℝ, ∀ i : Fin 2, k * a i + b i = t * (a i - 3 * b i)) → k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_and_parallel_l2720_272084


namespace NUMINAMATH_CALUDE_maximize_quadrilateral_area_l2720_272071

/-- Given a rectangle ABCD with length 2 and width 1, and points E on AB and F on AD
    such that AE = 2AF, the area of quadrilateral CDFE is maximized when AF = 3/4,
    and the maximum area is 7/8 square units. -/
theorem maximize_quadrilateral_area (A B C D E F : ℝ × ℝ) :
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1
  let ABCD_is_rectangle := 
    (A.1 = B.1 - rectangle_length) ∧ 
    (A.2 = D.2) ∧ 
    (B.2 = C.2) ∧ 
    (C.1 = D.1 + rectangle_length) ∧ 
    (A.2 = B.2 + rectangle_width)
  let E_on_AB := E.2 = A.2
  let F_on_AD := F.1 = A.1
  let AE_equals_2AF := E.1 - A.1 = 2 * (F.2 - A.2)
  let area_CDFE (x : ℝ) := 2 * x^2 - 3 * x + 2
  ABCD_is_rectangle → E_on_AB → F_on_AD → AE_equals_2AF →
    (∃ (x : ℝ), x = 3/4 ∧ 
      (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 1 → area_CDFE y ≤ area_CDFE x) ∧
      area_CDFE x = 7/8) := by
  sorry

end NUMINAMATH_CALUDE_maximize_quadrilateral_area_l2720_272071


namespace NUMINAMATH_CALUDE_minimum_m_value_l2720_272072

noncomputable def f (x : ℝ) : ℝ := Real.log x + (2*x + 1) / x

theorem minimum_m_value (m : ℤ) :
  (∃ x : ℝ, x > 1 ∧ f x < (m * (x - 1) + 2) / x) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_m_value_l2720_272072


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_one_third_l2720_272073

-- Define the properties of the tiled floor
structure TiledFloor :=
  (section_width : ℕ)
  (section_height : ℕ)
  (dark_tiles_per_section : ℕ)

-- Define the fraction of dark tiles
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  floor.dark_tiles_per_section / (floor.section_width * floor.section_height)

-- Theorem statement
theorem dark_tile_fraction_is_one_third 
  (floor : TiledFloor) 
  (h1 : floor.section_width = 6) 
  (h2 : floor.section_height = 4) 
  (h3 : floor.dark_tiles_per_section = 8) : 
  dark_tile_fraction floor = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_one_third_l2720_272073


namespace NUMINAMATH_CALUDE_parabola_line_intersection_properties_l2720_272068

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about properties of intersections between a line through the focus and a parabola -/
theorem parabola_line_intersection_properties (par : Parabola) 
  (A B : ParabolaPoint) (h_on_parabola : A.y^2 = 2*par.p*A.x ∧ B.y^2 = 2*par.p*B.x) 
  (h_through_focus : ∃ (k : ℝ), A.y = k*(A.x - par.p/2) ∧ B.y = k*(B.x - par.p/2)) :
  A.x * B.x = (par.p^2)/4 ∧ 
  1/(A.x + par.p/2) + 1/(B.x + par.p/2) = 2/par.p := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_properties_l2720_272068


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2720_272079

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  stratumPopulation : ℕ
  stratumSample : ℕ
  totalSample : ℕ

/-- The stratified sampling is proportional if the ratio of the stratum in the population
    equals the ratio of the stratum in the sample -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumPopulation * s.totalSample = s.totalPopulation * s.stratumSample

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.totalPopulation = 12000)
  (h2 : s.stratumPopulation = 3600)
  (h3 : s.stratumSample = 60)
  (h4 : isProportional s) :
  s.totalSample = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2720_272079


namespace NUMINAMATH_CALUDE_car_rental_per_mile_rate_l2720_272042

theorem car_rental_per_mile_rate (daily_rate : ℝ) (daily_budget : ℝ) (distance : ℝ) :
  daily_rate = 30 →
  daily_budget = 76 →
  distance = 200 →
  (daily_budget - daily_rate) / distance * 100 = 23 := by
sorry

end NUMINAMATH_CALUDE_car_rental_per_mile_rate_l2720_272042


namespace NUMINAMATH_CALUDE_divisibility_proof_l2720_272046

theorem divisibility_proof (n : ℕ) : 
  (∃ k : ℤ, 32^(3*n) - 1312^n = 1966 * k) ∧ 
  (∃ m : ℤ, 843^(2*n+1) - 1099^(2*n+1) + 16^(4*n+2) = 1967 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l2720_272046


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2720_272066

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y ∨ ∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2720_272066


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l2720_272016

/-- The number of packs Antonov has after giving one pack to his sister -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (initial_candies - candies_per_pack) / candies_per_pack

/-- Theorem stating that Antonov has 2 packs remaining -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l2720_272016


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l2720_272078

theorem multiplicative_inverse_modulo_million : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (N * ((222222 : ℕ) * 476190)) % 1000000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l2720_272078


namespace NUMINAMATH_CALUDE_probability_doubled_l2720_272082

def total_clips : ℕ := 16
def red_clips : ℕ := 4
def blue_clips : ℕ := 5
def green_clips : ℕ := 7
def removed_clips : ℕ := 12

theorem probability_doubled :
  let initial_prob : ℚ := red_clips / total_clips
  let remaining_clips : ℕ := total_clips - removed_clips
  let new_prob : ℚ := red_clips / remaining_clips
  new_prob = 2 * initial_prob := by sorry

end NUMINAMATH_CALUDE_probability_doubled_l2720_272082


namespace NUMINAMATH_CALUDE_distance_between_points_l2720_272039

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 12)
  let p2 : ℝ × ℝ := (10, 0)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 193 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2720_272039


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2720_272052

theorem quadratic_form_k_value :
  ∃ (a h : ℝ), ∀ x : ℝ, 9 * x^2 - 12 * x = a * (x - h)^2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2720_272052


namespace NUMINAMATH_CALUDE_circle_angle_sum_l2720_272038

/-- Given a circle with points X, Y, and Z, where arc XY = 50°, arc YZ = 45°, arc ZX = 90°,
    angle α = (arc XZ - arc YZ) / 2, and angle β = arc YZ / 2,
    prove that the sum of angles α and β equals 47.5°. -/
theorem circle_angle_sum (arcXY arcYZ arcZX : Real) (α β : Real) :
  arcXY = 50 ∧ arcYZ = 45 ∧ arcZX = 90 ∧
  α = (arcXY + arcYZ - arcYZ) / 2 ∧
  β = arcYZ / 2 →
  α + β = 47.5 := by
sorry


end NUMINAMATH_CALUDE_circle_angle_sum_l2720_272038


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2720_272083

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ quadratic m a = 0 ∧ quadratic m b = 0) ∧
  (∀ a b : ℝ, quadratic m a = 0 → quadratic m b = 0 → root_condition a b → (m = -2 ∨ m = 1)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2720_272083


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2720_272076

theorem multiplication_subtraction_equality : 154 * 1836 - 54 * 1836 = 183600 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2720_272076


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_2sqrt2_exists_min_value_l2720_272069

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by
  sorry

theorem min_value_is_2sqrt2 (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  a*b ≥ 2*Real.sqrt 2 :=
by
  sorry

theorem exists_min_value (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = Real.sqrt (x*y) ∧ x*y = 2*Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_2sqrt2_exists_min_value_l2720_272069


namespace NUMINAMATH_CALUDE_base_seven_5432_equals_1934_l2720_272020

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base_seven_5432_equals_1934 : 
  base_seven_to_ten [2, 3, 4, 5] = 1934 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_5432_equals_1934_l2720_272020
