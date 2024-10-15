import Mathlib

namespace NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l1592_159219

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point (2, 1) -/
def point : ℝ × ℝ := (2, 1)

/-- A line passes through a given point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- A line has equal intercepts on both axes -/
def equal_intercepts (l : Line) : Prop :=
  ∃ a : ℝ, l.intercept = a ∧ (-l.intercept / l.slope) = a

/-- There exists a unique line passing through (2, 1) with equal intercepts -/
theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line, passes_through l point ∧ equal_intercepts l := by
  sorry

end NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l1592_159219


namespace NUMINAMATH_CALUDE_BC_vector_l1592_159230

def complex_vector (a b : ℂ) : ℂ := b - a

theorem BC_vector (OA OC AB : ℂ) 
  (h1 : OA = -2 + I) 
  (h2 : OC = 3 + 2*I) 
  (h3 : AB = 1 + 5*I) : 
  complex_vector (OA + AB) OC = 4 - 4*I := by
  sorry

end NUMINAMATH_CALUDE_BC_vector_l1592_159230


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1592_159299

theorem complex_arithmetic_equality : 10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1592_159299


namespace NUMINAMATH_CALUDE_recycling_points_theorem_l1592_159229

/-- Calculates the points earned for recycling paper -/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the conditions, the total points earned is 5 -/
theorem recycling_points_theorem : 
  points_earned 6 28 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_theorem_l1592_159229


namespace NUMINAMATH_CALUDE_fred_dimes_problem_l1592_159234

/-- Represents the number of dimes Fred's sister borrowed -/
def dimes_borrowed (initial_dimes remaining_dimes : ℕ) : ℕ :=
  initial_dimes - remaining_dimes

theorem fred_dimes_problem (initial_dimes remaining_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : remaining_dimes = 4) :
  dimes_borrowed initial_dimes remaining_dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_problem_l1592_159234


namespace NUMINAMATH_CALUDE_allens_mother_age_l1592_159268

-- Define Allen's age as a function of his mother's age
def allen_age (mother_age : ℕ) : ℕ := mother_age - 25

-- Define the condition that in 3 years, the sum of their ages will be 41
def future_age_sum (mother_age : ℕ) : Prop :=
  (mother_age + 3) + (allen_age mother_age + 3) = 41

-- Theorem stating that Allen's mother's present age is 30
theorem allens_mother_age :
  ∃ (mother_age : ℕ), 
    (allen_age mother_age = mother_age - 25) ∧ 
    (future_age_sum mother_age) ∧ 
    (mother_age = 30) := by
  sorry

end NUMINAMATH_CALUDE_allens_mother_age_l1592_159268


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1592_159289

/-- Given a geometric sequence {a_n} where a_2 and a_3 are the roots of x^2 - x - 2013 = 0,
    prove that a_1 * a_4 = -2013 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2^2 - a 2 - 2013 = 0 →  -- a_2 is a root
  a 3^2 - a 3 - 2013 = 0 →  -- a_3 is a root
  a 1 * a 4 = -2013 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1592_159289


namespace NUMINAMATH_CALUDE_oxen_equivalence_l1592_159242

/-- The amount of fodder a buffalo eats per day -/
def buffalo_fodder : ℝ := sorry

/-- The amount of fodder a cow eats per day -/
def cow_fodder : ℝ := sorry

/-- The amount of fodder an ox eats per day -/
def ox_fodder : ℝ := sorry

/-- The total amount of fodder available -/
def total_fodder : ℝ := sorry

theorem oxen_equivalence :
  (3 * buffalo_fodder = 4 * cow_fodder) →
  (15 * buffalo_fodder + 8 * ox_fodder + 24 * cow_fodder) * 36 = total_fodder →
  (30 * buffalo_fodder + 8 * ox_fodder + 64 * cow_fodder) * 18 = total_fodder →
  (∃ n : ℕ, n * ox_fodder = 3 * buffalo_fodder ∧ n * ox_fodder = 4 * cow_fodder ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_oxen_equivalence_l1592_159242


namespace NUMINAMATH_CALUDE_square_difference_equality_l1592_159276

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1592_159276


namespace NUMINAMATH_CALUDE_equation_solutions_l1592_159220

theorem equation_solutions : 
  (∃ s1 : Set ℝ, s1 = {x : ℝ | x^2 + 2*x - 8 = 0} ∧ s1 = {-4, 2}) ∧ 
  (∃ s2 : Set ℝ, s2 = {x : ℝ | x*(x-2) = x-2} ∧ s2 = {2, 1}) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1592_159220


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1592_159298

theorem infinitely_many_perfect_squares (k : ℕ+) :
  ∃ f : ℕ → ℕ+, Monotone f ∧ ∀ i : ℕ, ∃ m : ℕ+, (f i : ℕ) * 2^(k : ℕ) - 7 = m^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1592_159298


namespace NUMINAMATH_CALUDE_michaels_boxes_l1592_159255

/-- Given that Michael has 16 blocks and each box must contain 2 blocks, 
    prove that the number of boxes Michael has is 8. -/
theorem michaels_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h1 : total_blocks = 16) (h2 : blocks_per_box = 2) :
  total_blocks / blocks_per_box = 8 := by
  sorry


end NUMINAMATH_CALUDE_michaels_boxes_l1592_159255


namespace NUMINAMATH_CALUDE_units_digit_of_8_power_2022_l1592_159266

theorem units_digit_of_8_power_2022 : 8^2022 % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8_power_2022_l1592_159266


namespace NUMINAMATH_CALUDE_alice_profit_l1592_159260

def total_bracelets : ℕ := 52
def design_a_bracelets : ℕ := 30
def design_b_bracelets : ℕ := 22
def cost_a : ℚ := 2
def cost_b : ℚ := 4.5
def given_away_a : ℕ := 5
def given_away_b : ℕ := 3
def sell_price_a : ℚ := 0.25
def sell_price_b : ℚ := 0.5

def total_cost : ℚ := design_a_bracelets * cost_a + design_b_bracelets * cost_b
def remaining_a : ℕ := design_a_bracelets - given_away_a
def remaining_b : ℕ := design_b_bracelets - given_away_b
def total_revenue : ℚ := remaining_a * sell_price_a + remaining_b * sell_price_b
def profit : ℚ := total_revenue - total_cost

theorem alice_profit :
  profit = -143.25 :=
sorry

end NUMINAMATH_CALUDE_alice_profit_l1592_159260


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1592_159275

/-- Given a quadratic inequality x^2 - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a and b is equal to 5. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1592_159275


namespace NUMINAMATH_CALUDE_days_not_played_in_june_l1592_159248

/-- The number of days in June. -/
def june_days : ℕ := 30

/-- The number of songs Vivian plays per day. -/
def vivian_songs : ℕ := 10

/-- The number of songs Clara plays per day. -/
def clara_songs : ℕ := vivian_songs - 2

/-- The total number of songs both Vivian and Clara listened to in June. -/
def total_songs : ℕ := 396

/-- The number of days they played songs in June. -/
def days_played : ℕ := total_songs / (vivian_songs + clara_songs)

theorem days_not_played_in_june : june_days - days_played = 8 := by
  sorry

end NUMINAMATH_CALUDE_days_not_played_in_june_l1592_159248


namespace NUMINAMATH_CALUDE_range_of_a_l1592_159225

-- Define the propositions P and Q
def P (x a : ℝ) : Prop := |x - a| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the negations of P and Q
def not_P (x a : ℝ) : Prop := ¬(P x a)
def not_Q (x : ℝ) : Prop := ¬(Q x)

-- Define the condition that not_P is sufficient but not necessary for not_Q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, not_P x a → not_Q x) ∧ (∃ x, not_Q x ∧ ¬(not_P x a))

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1592_159225


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_difference_l1592_159244

open Complex

theorem real_part_of_reciprocal_difference (w : ℂ) (h1 : w ≠ 0) (h2 : w.im ≠ 0) (h3 : abs w = 2) :
  (1 / (2 - w)).re = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_difference_l1592_159244


namespace NUMINAMATH_CALUDE_complete_work_together_l1592_159236

/-- The number of days it takes for two workers to complete a job together,
    given the number of days it takes each worker to complete the job individually. -/
def days_to_complete_together (days_a days_b : ℚ) : ℚ :=
  1 / (1 / days_a + 1 / days_b)

/-- Theorem stating that if worker A takes 9 days and worker B takes 18 days to complete a job individually,
    then together they will complete the job in 6 days. -/
theorem complete_work_together :
  days_to_complete_together 9 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_complete_work_together_l1592_159236


namespace NUMINAMATH_CALUDE_sum_of_abc_l1592_159273

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (eq1 : a^2 + b*c = 115)
  (eq2 : b^2 + a*c = 127)
  (eq3 : c^2 + a*b = 115) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1592_159273


namespace NUMINAMATH_CALUDE_visible_cubes_12_cube_l1592_159245

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Calculates the number of visible unit cubes from a corner of a cube --/
def visible_unit_cubes (c : Cube n) : ℕ :=
  3 * n^2 - 3 * (n - 1) + 1

/-- Theorem stating that for a 12×12×12 cube, the number of visible unit cubes from a corner is 400 --/
theorem visible_cubes_12_cube :
  ∃ (c : Cube 12), visible_unit_cubes c = 400 :=
sorry

end NUMINAMATH_CALUDE_visible_cubes_12_cube_l1592_159245


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1592_159223

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1592_159223


namespace NUMINAMATH_CALUDE_art_exhibition_problem_l1592_159202

/-- Art exhibition visitor and ticket problem -/
theorem art_exhibition_problem 
  (total_saturday : ℕ)
  (sunday_morning_increase : ℚ)
  (sunday_afternoon_increase : ℚ)
  (total_sunday_increase : ℕ)
  (sunday_morning_revenue : ℕ)
  (sunday_afternoon_revenue : ℕ)
  (sunday_morning_adults : ℕ)
  (sunday_afternoon_adults : ℕ)
  (h1 : total_saturday = 300)
  (h2 : sunday_morning_increase = 40 / 100)
  (h3 : sunday_afternoon_increase = 30 / 100)
  (h4 : total_sunday_increase = 100)
  (h5 : sunday_morning_revenue = 4200)
  (h6 : sunday_afternoon_revenue = 7200)
  (h7 : sunday_morning_adults = 70)
  (h8 : sunday_afternoon_adults = 100) :
  ∃ (sunday_morning sunday_afternoon adult_price student_price : ℕ),
    sunday_morning = 140 ∧
    sunday_afternoon = 260 ∧
    adult_price = 40 ∧
    student_price = 20 := by
  sorry


end NUMINAMATH_CALUDE_art_exhibition_problem_l1592_159202


namespace NUMINAMATH_CALUDE_triangle_side_length_l1592_159246

/-- Given a triangle ABC with side lengths a, b, c and angle B, 
    prove that if b = √3, c = 3, and B = 30°, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = Real.sqrt 3 → c = 3 → B = π / 6 → a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1592_159246


namespace NUMINAMATH_CALUDE_power_function_inequality_l1592_159281

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l1592_159281


namespace NUMINAMATH_CALUDE_arithmetic_average_characterization_l1592_159208

/-- φ(n) is the number of positive integers ≤ n and coprime with n -/
def phi (n : ℕ+) : ℕ := sorry

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- One of n, φ(n), or τ(n) is the arithmetic average of the other two -/
def is_arithmetic_average (n : ℕ+) : Prop :=
  (n : ℕ) = (phi n + tau n) / 2 ∨
  phi n = ((n : ℕ) + tau n) / 2 ∨
  tau n = ((n : ℕ) + phi n) / 2

theorem arithmetic_average_characterization (n : ℕ+) :
  is_arithmetic_average n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end NUMINAMATH_CALUDE_arithmetic_average_characterization_l1592_159208


namespace NUMINAMATH_CALUDE_jade_cal_difference_l1592_159296

/-- The number of transactions handled by different people on Thursday -/
def thursday_transactions : ℕ → ℕ
| 0 => 90  -- Mabel's transactions
| 1 => (110 * thursday_transactions 0) / 100  -- Anthony's transactions
| 2 => (2 * thursday_transactions 1) / 3  -- Cal's transactions
| 3 => 84  -- Jade's transactions
| _ => 0

/-- The theorem stating the difference between Jade's and Cal's transactions -/
theorem jade_cal_difference : 
  thursday_transactions 3 - thursday_transactions 2 = 18 :=
sorry

end NUMINAMATH_CALUDE_jade_cal_difference_l1592_159296


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1592_159203

theorem min_value_trigonometric_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1592_159203


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1592_159271

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1592_159271


namespace NUMINAMATH_CALUDE_gift_card_balance_l1592_159283

/-- Calculates the remaining balance on a gift card after a coffee purchase -/
theorem gift_card_balance 
  (gift_card_amount : ℝ) 
  (coffee_price_per_pound : ℝ) 
  (pounds_purchased : ℝ) 
  (h1 : gift_card_amount = 70) 
  (h2 : coffee_price_per_pound = 8.58) 
  (h3 : pounds_purchased = 4) : 
  gift_card_amount - (coffee_price_per_pound * pounds_purchased) = 35.68 := by
sorry

end NUMINAMATH_CALUDE_gift_card_balance_l1592_159283


namespace NUMINAMATH_CALUDE_visitor_difference_l1592_159295

def visitors_previous_day : ℕ := 100
def visitors_that_day : ℕ := 666

theorem visitor_difference : visitors_that_day - visitors_previous_day = 566 := by
  sorry

end NUMINAMATH_CALUDE_visitor_difference_l1592_159295


namespace NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_for_quadratic_l1592_159238

theorem abs_x_lt_2_sufficient_not_necessary_for_quadratic :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_for_quadratic_l1592_159238


namespace NUMINAMATH_CALUDE_angle_measure_l1592_159254

theorem angle_measure (A : ℝ) : 
  (90 - A = (180 - A) / 3 - 10) → A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1592_159254


namespace NUMINAMATH_CALUDE_S_equals_zero_two_neg_two_l1592_159286

def imaginary_unit : ℂ := Complex.I

def S : Set ℂ := {z | ∃ n : ℤ, z = (imaginary_unit ^ n) + (imaginary_unit ^ (-n))}

theorem S_equals_zero_two_neg_two : S = {0, 2, -2} := by sorry

end NUMINAMATH_CALUDE_S_equals_zero_two_neg_two_l1592_159286


namespace NUMINAMATH_CALUDE_linear_system_solution_l1592_159207

theorem linear_system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ - 2*x₂ + x₄ = -3 ∧
   3*x₁ - x₂ - 2*x₃ = 1 ∧
   2*x₁ + x₂ - 2*x₃ - x₄ = 4 ∧
   x₁ + 3*x₂ - 2*x₃ - 2*x₄ = 7) →
  (∃ t u : ℝ, x₁ = -3 + 2*x₂ - x₄ ∧
              x₂ = 2 + (2/5)*t + (3/5)*u ∧
              x₃ = t ∧
              x₄ = u) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1592_159207


namespace NUMINAMATH_CALUDE_cherry_price_theorem_l1592_159256

/-- The price of a bag of cherries satisfies the given conditions -/
theorem cherry_price_theorem (olive_price : ℝ) (bag_count : ℕ) (discount_rate : ℝ) (final_cost : ℝ) :
  olive_price = 7 →
  bag_count = 50 →
  discount_rate = 0.1 →
  final_cost = 540 →
  ∃ (cherry_price : ℝ),
    cherry_price = 5 ∧
    (1 - discount_rate) * (bag_count * cherry_price + bag_count * olive_price) = final_cost :=
by sorry

end NUMINAMATH_CALUDE_cherry_price_theorem_l1592_159256


namespace NUMINAMATH_CALUDE_positive_real_equalities_l1592_159200

theorem positive_real_equalities (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c) ∧
  ((a + b + c) * (a^2 + b^2 + c^2 - a*b - b*c - a*c) = 0 → a = b ∧ b = c) ∧
  (a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d → a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_equalities_l1592_159200


namespace NUMINAMATH_CALUDE_range_of_m_l1592_159294

theorem range_of_m (P : ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) :
  ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1592_159294


namespace NUMINAMATH_CALUDE_no_one_left_behind_l1592_159278

/-- Represents the Ferris wheel problem -/
structure FerrisWheel where
  seats_per_rotation : ℕ
  total_rotations : ℕ
  initial_queue : ℕ
  impatience_rate : ℚ

/-- Calculates the number of people remaining in the queue after a given number of rotations -/
def people_remaining (fw : FerrisWheel) (rotations : ℕ) : ℕ :=
  sorry

/-- The main theorem: proves that no one is left in the queue after three rotations -/
theorem no_one_left_behind (fw : FerrisWheel) 
  (h1 : fw.seats_per_rotation = 56)
  (h2 : fw.total_rotations = 3)
  (h3 : fw.initial_queue = 92)
  (h4 : fw.impatience_rate = 1/10) :
  people_remaining fw 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_one_left_behind_l1592_159278


namespace NUMINAMATH_CALUDE_handshakes_in_specific_convention_l1592_159201

/-- Represents a convention with companies and representatives -/
structure Convention where
  num_companies : ℕ
  reps_per_company : ℕ
  companies_to_shake : ℕ

/-- Calculates the total number of handshakes in the convention -/
def total_handshakes (conv : Convention) : ℕ :=
  let total_people := conv.num_companies * conv.reps_per_company
  let handshakes_per_person := (conv.companies_to_shake * conv.reps_per_company)
  (total_people * handshakes_per_person) / 2

/-- The specific convention described in the problem -/
def specific_convention : Convention :=
  { num_companies := 5
  , reps_per_company := 4
  , companies_to_shake := 2 }

theorem handshakes_in_specific_convention :
  total_handshakes specific_convention = 80 := by
  sorry


end NUMINAMATH_CALUDE_handshakes_in_specific_convention_l1592_159201


namespace NUMINAMATH_CALUDE_xyz_inequality_l1592_159216

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1592_159216


namespace NUMINAMATH_CALUDE_shift_down_two_units_l1592_159235

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x - 1

def vertical_shift (h : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => h x - shift

theorem shift_down_two_units :
  vertical_shift f 2 = g :=
sorry

end NUMINAMATH_CALUDE_shift_down_two_units_l1592_159235


namespace NUMINAMATH_CALUDE_exist_three_quadratics_with_specific_root_properties_l1592_159253

theorem exist_three_quadratics_with_specific_root_properties :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x - 1)^2) ∧
    (∀ x, p₂ x = x^2) ∧
    (∀ x, p₃ x = (x - 2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_exist_three_quadratics_with_specific_root_properties_l1592_159253


namespace NUMINAMATH_CALUDE_unique_divisible_by_twelve_l1592_159249

/-- A function that constructs a four-digit number in the form x27x -/
def constructNumber (x : Nat) : Nat :=
  1000 * x + 270 + x

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_divisible_by_twelve :
  ∃! x : Nat, isSingleDigit x ∧ (constructNumber x) % 12 = 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_twelve_l1592_159249


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l1592_159280

theorem ceiling_squared_negative_fraction : ⌈((-7/4 : ℚ)^2)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l1592_159280


namespace NUMINAMATH_CALUDE_abc_product_absolute_value_l1592_159221

theorem abc_product_absolute_value (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_eq : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) : 
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_absolute_value_l1592_159221


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1592_159212

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1592_159212


namespace NUMINAMATH_CALUDE_losing_position_characterization_l1592_159287

/-- Represents the state of the table-folding game -/
structure GameState where
  n : ℕ
  m : ℕ

/-- Predicate to determine if a game state is a losing position -/
def is_losing_position (state : GameState) : Prop :=
  ∃ k : ℕ, state.m = (state.n + 1) * 2^k - 1

/-- The main theorem stating the characterization of losing positions -/
theorem losing_position_characterization (state : GameState) :
  is_losing_position state ↔ 
  (∀ fold : ℕ, fold > 0 → fold ≤ state.m → 
    ¬is_losing_position ⟨state.n, state.m - fold⟩) ∧
  (∀ fold : ℕ, fold > 0 → fold ≤ state.n → 
    ¬is_losing_position ⟨state.n - fold, state.m⟩) :=
sorry

end NUMINAMATH_CALUDE_losing_position_characterization_l1592_159287


namespace NUMINAMATH_CALUDE_ryan_commute_time_l1592_159210

/-- Ryan's weekly commute time calculation -/
theorem ryan_commute_time : 
  let bike_days : ℕ := 1
  let bus_days : ℕ := 3
  let friend_days : ℕ := 1
  let bike_time : ℕ := 30
  let bus_time : ℕ := bike_time + 10
  let friend_time : ℕ := bike_time / 3
  bike_days * bike_time + bus_days * bus_time + friend_days * friend_time = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_commute_time_l1592_159210


namespace NUMINAMATH_CALUDE_train_length_l1592_159251

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 250 →
  crossing_time = 36 →
  train_speed * crossing_time - platform_length = 470 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1592_159251


namespace NUMINAMATH_CALUDE_minimum_area_is_14_l1592_159284

-- Define the variation ranges
def normal_variation : ℝ := 0.5
def approximate_variation : ℝ := 1.0

-- Define the reported dimensions
def reported_length : ℝ := 4.0
def reported_width : ℝ := 5.0

-- Define the actual minimum dimensions
def min_length : ℝ := reported_length - normal_variation
def min_width : ℝ := reported_width - approximate_variation

-- Define the minimum area
def min_area : ℝ := min_length * min_width

-- Theorem statement
theorem minimum_area_is_14 : min_area = 14 := by
  sorry

end NUMINAMATH_CALUDE_minimum_area_is_14_l1592_159284


namespace NUMINAMATH_CALUDE_no_solution_exists_l1592_159213

theorem no_solution_exists : ¬ ∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1592_159213


namespace NUMINAMATH_CALUDE_triangle_angle_range_l1592_159237

theorem triangle_angle_range (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_tan : Real.tan B ^ 2 = Real.tan A * Real.tan C) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l1592_159237


namespace NUMINAMATH_CALUDE_keanu_fish_spending_l1592_159209

/-- The number of fish Keanu gave to his dog -/
def dog_fish : ℕ := 40

/-- The number of fish Keanu gave to his cat -/
def cat_fish : ℕ := dog_fish / 2

/-- The cost of each fish in dollars -/
def fish_cost : ℕ := 4

/-- The total number of fish Keanu bought -/
def total_fish : ℕ := dog_fish + cat_fish

/-- The total amount Keanu spent on fish in dollars -/
def total_spent : ℕ := total_fish * fish_cost

theorem keanu_fish_spending :
  total_spent = 240 :=
sorry

end NUMINAMATH_CALUDE_keanu_fish_spending_l1592_159209


namespace NUMINAMATH_CALUDE_parallel_lines_ratio_l1592_159262

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  L1 : ℝ → ℝ → Prop
  L2 : ℝ → ℝ → Prop
  a : ℝ
  c : ℝ
  c_pos : c > 0
  is_parallel : ∀ x y, L1 x y ↔ x - y + 1 = 0
  L2_eq : ∀ x y, L2 x y ↔ 3*x + a*y - c = 0
  distance : ℝ

/-- The theorem stating the value of (a-3)/c for the given parallel lines -/
theorem parallel_lines_ratio (lines : ParallelLines) 
  (h_dist : lines.distance = Real.sqrt 2) : 
  (lines.a - 3) / lines.c = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_ratio_l1592_159262


namespace NUMINAMATH_CALUDE_rectangle_area_invariance_l1592_159214

theorem rectangle_area_invariance (x y : ℝ) :
  (x + 5/2) * (y - 2/3) = (x - 5/2) * (y + 4/3) ∧ 
  (x + 5/2) * (y - 2/3) = x * y →
  x * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_invariance_l1592_159214


namespace NUMINAMATH_CALUDE_cube_sum_equation_l1592_159264

theorem cube_sum_equation (y : ℝ) (h : y^3 + 4 / y^3 = 110) : y + 4 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equation_l1592_159264


namespace NUMINAMATH_CALUDE_workshop_workers_l1592_159231

/-- Proves that the total number of workers is 12 given the conditions in the problem -/
theorem workshop_workers (total_average : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) 
  (num_technicians : ℕ) (h1 : total_average = 9000) (h2 : tech_average = 12000) 
  (h3 : non_tech_average = 6000) (h4 : num_technicians = 6) : 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    total_average * total_workers = 
      num_technicians * tech_average + (total_workers - num_technicians) * non_tech_average :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l1592_159231


namespace NUMINAMATH_CALUDE_expand_a_expand_b_expand_c_expand_d_expand_e_l1592_159288

-- Define variables
variable (x y m n : ℝ)

-- Theorem for expression (a)
theorem expand_a : (x + 3*y)^2 = x^2 + 6*x*y + 9*y^2 := by sorry

-- Theorem for expression (b)
theorem expand_b : (2*x + 3*y)^2 = 4*x^2 + 12*x*y + 9*y^2 := by sorry

-- Theorem for expression (c)
theorem expand_c : (m^3 + n^5)^2 = m^6 + 2*m^3*n^5 + n^10 := by sorry

-- Theorem for expression (d)
theorem expand_d : (5*x - 3*y)^2 = 25*x^2 - 30*x*y + 9*y^2 := by sorry

-- Theorem for expression (e)
theorem expand_e : (3*m^5 - 4*n^2)^2 = 9*m^10 - 24*m^5*n^2 + 16*n^4 := by sorry

end NUMINAMATH_CALUDE_expand_a_expand_b_expand_c_expand_d_expand_e_l1592_159288


namespace NUMINAMATH_CALUDE_zoo_arrangement_count_l1592_159218

def num_lions : Nat := 3
def num_zebras : Nat := 4
def num_monkeys : Nat := 6
def total_animals : Nat := num_lions + num_zebras + num_monkeys

theorem zoo_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_lions) * (Nat.factorial num_zebras) * (Nat.factorial num_monkeys) = 622080 :=
by sorry

end NUMINAMATH_CALUDE_zoo_arrangement_count_l1592_159218


namespace NUMINAMATH_CALUDE_cone_height_l1592_159241

/-- Proves that a cone with lateral area 15π cm² and base radius 3 cm has a height of 4 cm -/
theorem cone_height (lateral_area : ℝ) (base_radius : ℝ) (height : ℝ) : 
  lateral_area = 15 * Real.pi ∧ base_radius = 3 → height = 4 := by
  sorry

#check cone_height

end NUMINAMATH_CALUDE_cone_height_l1592_159241


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1592_159205

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1592_159205


namespace NUMINAMATH_CALUDE_adjacent_i_probability_is_one_fifth_l1592_159227

/-- The probability of forming a 10-letter code with two adjacent i's -/
def adjacent_i_probability : ℚ :=
  let total_arrangements := Nat.factorial 10
  let favorable_arrangements := Nat.factorial 9 * Nat.factorial 2
  favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of forming a 10-letter code
    with two adjacent i's is 1/5 -/
theorem adjacent_i_probability_is_one_fifth :
  adjacent_i_probability = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_i_probability_is_one_fifth_l1592_159227


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1592_159233

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧ 
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → 
      f a x ≠ (f_derivative a x₁) * x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1592_159233


namespace NUMINAMATH_CALUDE_guards_in_team_l1592_159272

theorem guards_in_team (s b n : ℕ) : 
  s > 0 ∧ b > 0 ∧ n > 0 →  -- positive integers
  s * b * n = 1001 →  -- total person-nights
  s < n →  -- guards in team less than nights slept
  n < b →  -- nights slept less than number of teams
  s = 7 :=  -- prove number of guards in a team is 7
by sorry

end NUMINAMATH_CALUDE_guards_in_team_l1592_159272


namespace NUMINAMATH_CALUDE_abs_neg_2023_l1592_159269

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l1592_159269


namespace NUMINAMATH_CALUDE_minimum_distances_to_pond_l1592_159247

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a walk in cardinal directions -/
inductive Walk
  | North : Nat → Walk
  | South : Nat → Walk
  | East : Nat → Walk
  | West : Nat → Walk

/-- Calculates the end point after a series of walks -/
def end_point (start : Point) (walks : List Walk) : Point :=
  walks.foldl
    (fun p w =>
      match w with
      | Walk.North n => { x := p.x, y := p.y + n }
      | Walk.South n => { x := p.x, y := p.y - n }
      | Walk.East n => { x := p.x + n, y := p.y }
      | Walk.West n => { x := p.x - n, y := p.y })
    start

/-- Calculates the Manhattan distance between two points -/
def manhattan_distance (p1 p2 : Point) : Nat :=
  (p1.x - p2.x).natAbs + (p1.y - p2.y).natAbs

/-- Anička's initial walk -/
def anicka_walk : List Walk :=
  [Walk.North 5, Walk.East 2, Walk.South 3, Walk.West 4]

/-- Vojta's initial walk -/
def vojta_walk : List Walk :=
  [Walk.South 3, Walk.West 4, Walk.North 1]

theorem minimum_distances_to_pond :
  let anicka_start : Point := { x := 0, y := 0 }
  let vojta_start : Point := { x := 0, y := 0 }
  let pond := end_point anicka_start anicka_walk
  let vojta_end := end_point vojta_start vojta_walk
  vojta_end.x + 5 = pond.x →
  manhattan_distance anicka_start pond = 4 ∧
  manhattan_distance vojta_start pond = 3 :=
by sorry


end NUMINAMATH_CALUDE_minimum_distances_to_pond_l1592_159247


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_symmetric_points_coordinates_l1592_159277

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The symmetric point about the y-axis -/
def symmetricAboutYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis (p : Point2D) :
  let p' := symmetricAboutYAxis p
  p'.x = -p.x ∧ p'.y = p.y := by sorry

/-- Given points A, B, and C -/
def A : Point2D := { x := -3, y := 2 }
def B : Point2D := { x := -4, y := -3 }
def C : Point2D := { x := -1, y := -1 }

/-- Symmetric points A', B', and C' -/
def A' : Point2D := symmetricAboutYAxis A
def B' : Point2D := symmetricAboutYAxis B
def C' : Point2D := symmetricAboutYAxis C

theorem symmetric_points_coordinates :
  A'.x = 3 ∧ A'.y = 2 ∧
  B'.x = 4 ∧ B'.y = -3 ∧
  C'.x = 1 ∧ C'.y = -1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_symmetric_points_coordinates_l1592_159277


namespace NUMINAMATH_CALUDE_pizza_bill_friends_l1592_159293

theorem pizza_bill_friends (total_price : ℕ) (price_per_person : ℕ) (bob_included : Bool) : 
  total_price = 40 → price_per_person = 8 → bob_included = true → 
  (total_price / price_per_person) - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_bill_friends_l1592_159293


namespace NUMINAMATH_CALUDE_train_length_calculation_l1592_159292

/-- Calculates the length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 144 →
  time_seconds = 0.9999200063994881 →
  ∃ (length_meters : ℝ), abs (length_meters - 39.997) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1592_159292


namespace NUMINAMATH_CALUDE_perimeter_bounds_l1592_159291

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  DA : ℕ+
  DA_eq_2005 : DA = 2005
  right_angles : True  -- Represents ∠ABC = ∠ADC = 90°
  max_side_lt_2005 : max AB BC < 2005 ∧ max (max AB BC) CD < 2005

/-- The perimeter of the quadrilateral -/
def perimeter (q : InscribedQuadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.DA.val

/-- Theorem stating the bounds on the perimeter -/
theorem perimeter_bounds (q : InscribedQuadrilateral) :
  4160 ≤ perimeter q ∧ perimeter q ≤ 7772 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_bounds_l1592_159291


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_7_factorial_plus_8_factorial_l1592_159252

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

theorem largest_prime_factor_of_7_factorial_plus_8_factorial :
  largest_prime_factor (factorial 7 + factorial 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_7_factorial_plus_8_factorial_l1592_159252


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1592_159259

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

def parabola (x y : ℝ) : Prop :=
  y^2 = 24 * x

def directrix (x : ℝ) : Prop :=
  x = -6

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ asymptote x y) →
  (∃ x : ℝ, directrix x ∧ ∃ y : ℝ, hyperbola a b x y) →
  (∀ x y : ℝ, hyperbola a b x y ↔ hyperbola 3 (Real.sqrt 27) x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1592_159259


namespace NUMINAMATH_CALUDE_horse_grazing_area_l1592_159290

/-- The area over which a horse can graze when tethered to one corner of a rectangular field --/
theorem horse_grazing_area (field_length : ℝ) (field_width : ℝ) (rope_length : ℝ) 
    (h1 : field_length = 40)
    (h2 : field_width = 24)
    (h3 : rope_length = 14)
    (h4 : rope_length ≤ field_length / 2)
    (h5 : rope_length ≤ field_width / 2) :
  (1/4 : ℝ) * Real.pi * rope_length^2 = 49 * Real.pi := by
  sorry

#check horse_grazing_area

end NUMINAMATH_CALUDE_horse_grazing_area_l1592_159290


namespace NUMINAMATH_CALUDE_number_divided_by_0_025_equals_40_l1592_159215

theorem number_divided_by_0_025_equals_40 (x : ℝ) : x / 0.025 = 40 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_0_025_equals_40_l1592_159215


namespace NUMINAMATH_CALUDE_product_remainder_remainder_98_102_mod_11_l1592_159263

theorem product_remainder (a b n : ℕ) (h : n > 0) : (a * b) % n = ((a % n) * (b % n)) % n := by sorry

theorem remainder_98_102_mod_11 : (98 * 102) % 11 = 1 := by sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_98_102_mod_11_l1592_159263


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1592_159282

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Finset Nat := {1, 2, 5, 7}

theorem complement_of_A_in_U : 
  (U \ A : Finset Nat) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1592_159282


namespace NUMINAMATH_CALUDE_power_fraction_product_l1592_159297

theorem power_fraction_product : (-4/5)^2022 * (5/4)^2023 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_product_l1592_159297


namespace NUMINAMATH_CALUDE_unique_solution_l1592_159204

theorem unique_solution : ∃! (x y : ℕ+), x^(y:ℕ) + 1 = y^(x:ℕ) ∧ 2*(x^(y:ℕ)) = y^(x:ℕ) + 13 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1592_159204


namespace NUMINAMATH_CALUDE_subway_speed_increase_l1592_159226

/-- The speed equation for the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the time at which the train is moving 55 km/h faster -/
theorem subway_speed_increase (s : ℝ) (h1 : 0 ≤ s) (h2 : s ≤ 7) :
  speed s = speed 2 + 55 ↔ s = 7 := by sorry

end NUMINAMATH_CALUDE_subway_speed_increase_l1592_159226


namespace NUMINAMATH_CALUDE_sphere_volume_diameter_relation_l1592_159232

theorem sphere_volume_diameter_relation :
  ∀ (V₁ V₂ d₁ d₂ : ℝ),
  V₁ > 0 → d₁ > 0 →
  V₁ = (π * d₁^3) / 6 →
  V₂ = 2 * V₁ →
  V₂ = (π * d₂^3) / 6 →
  d₂ / d₁ = (2 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_diameter_relation_l1592_159232


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1592_159240

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n > 2 ∧ exterior_angle = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1592_159240


namespace NUMINAMATH_CALUDE_simplify_expression_l1592_159250

theorem simplify_expression :
  ∀ x y : ℝ, (5 - 6*x) - (9 + 5*x - 2*y) = -4 - 11*x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1592_159250


namespace NUMINAMATH_CALUDE_fraction_simplification_l1592_159261

theorem fraction_simplification (b y θ : ℝ) (h : b^2 + y^2 ≠ 0) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2) * Real.cos θ) / (b^2 + y^2) =
  (b^2 * (Real.sqrt (b^2 + y^2) - Real.cos θ) + y^2 * (Real.sqrt (b^2 + y^2) + Real.cos θ)) /
  (b^2 + y^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1592_159261


namespace NUMINAMATH_CALUDE_equation_roots_theorem_l1592_159206

/-- 
Given an equation (x² - px) / (kx - d) = (n - 2) / (n + 2),
where the roots are numerically equal but opposite in sign and their product is 1,
prove that n = 2(k - p) / (k + p).
-/
theorem equation_roots_theorem (p k d n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - p*x) / (k*x - d) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  (∃ r : ℝ, x r * x (-r) = 1) →
  n = 2*(k - p) / (k + p) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_theorem_l1592_159206


namespace NUMINAMATH_CALUDE_min_value_theorem_l1592_159279

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  (x + 1) * (y + 2) ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1592_159279


namespace NUMINAMATH_CALUDE_gcf_of_90_and_105_l1592_159258

theorem gcf_of_90_and_105 : Nat.gcd 90 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_105_l1592_159258


namespace NUMINAMATH_CALUDE_point_symmetry_l1592_159270

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The main theorem -/
theorem point_symmetry (M N P : Point) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point.mk 1 2 →
  P = Point.mk (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l1592_159270


namespace NUMINAMATH_CALUDE_function_range_l1592_159243

theorem function_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - a*x + a + 3 < 0 ∧ x - a < 0)) → 
  a ∈ Set.Icc (-3) 6 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1592_159243


namespace NUMINAMATH_CALUDE_square_difference_division_l1592_159222

theorem square_difference_division : (175^2 - 155^2) / 20 = 330 := by sorry

end NUMINAMATH_CALUDE_square_difference_division_l1592_159222


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1592_159217

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1592_159217


namespace NUMINAMATH_CALUDE_circle_tangent_implies_m_equals_9_l1592_159267

/-- Circle C with equation x^2 + y^2 - 6x - 8y + m = 0 -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

/-- Unit circle with equation x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

/-- Main theorem: If circle C is externally tangent to the unit circle, then m = 9 -/
theorem circle_tangent_implies_m_equals_9 (m : ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_C m x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    externally_tangent center (0, 0) radius 1) →
  m = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_implies_m_equals_9_l1592_159267


namespace NUMINAMATH_CALUDE_quadratic_roots_mean_l1592_159239

theorem quadratic_roots_mean (b c : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ + r₂) / 2 = 9 →
  (r₁ * r₂).sqrt = 21 →
  r₁ + r₂ = -b →
  r₁ * r₂ = c →
  b = -18 ∧ c = 441 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_mean_l1592_159239


namespace NUMINAMATH_CALUDE_constant_sum_sequence_2013_l1592_159265

/-- A sequence where the sum of any three consecutive terms is constant -/
def ConstantSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = a (n + 1) + a (n + 2) + a (n + 3)

theorem constant_sum_sequence_2013 (a : ℕ → ℝ) (x : ℝ) 
    (h_constant_sum : ConstantSumSequence a)
    (h_a3 : a 3 = x)
    (h_a999 : a 999 = 3 - 2*x) :
    a 2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_sequence_2013_l1592_159265


namespace NUMINAMATH_CALUDE_intercepted_segment_length_l1592_159211

/-- The length of the line segment intercepted by curve C on line l -/
theorem intercepted_segment_length :
  let line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 - 1 = 0}
  let curve_C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}
  let intersection := line_l ∩ curve_C
  ∃ (A B : ℝ × ℝ), A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_intercepted_segment_length_l1592_159211


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1592_159274

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1592_159274


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1592_159228

/-- Theorem: For three parallel lines with y-intercepts 2, 3, and 4, 
    if the sum of their x-intercepts is 36, then their slope is -1/4. -/
theorem parallel_lines_slope (m : ℝ) 
  (h1 : m * (-2/m) + m * (-3/m) + m * (-4/m) = 36) : m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1592_159228


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1592_159224

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

theorem hyperbola_equation :
  ∀ (x y : ℝ), 
    (∃ (t : ℝ), hyperbola t y ∧ asymptotes t y) →  -- Hyperbola exists with given asymptotes
    hyperbola 4 (Real.sqrt 3) →                    -- Hyperbola passes through (4, √3)
    hyperbola x y                                  -- The equation of the hyperbola
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1592_159224


namespace NUMINAMATH_CALUDE_pill_supply_lasts_eight_months_l1592_159285

/-- Calculates the duration in months that a pill supply will last -/
def pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) : ℕ :=
  (total_pills * days_per_pill) / days_per_month

/-- Proves that a supply of 120 pills, taken every two days, lasts 8 months -/
theorem pill_supply_lasts_eight_months :
  pill_supply_duration 120 2 30 = 8 := by
  sorry

#eval pill_supply_duration 120 2 30

end NUMINAMATH_CALUDE_pill_supply_lasts_eight_months_l1592_159285


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1592_159257

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1592_159257
