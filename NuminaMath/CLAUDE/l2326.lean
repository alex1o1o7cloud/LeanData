import Mathlib

namespace NUMINAMATH_CALUDE_hypotenuse_of_6_8_triangle_l2326_232699

/-- The Pythagorean theorem for a right-angled triangle -/
def pythagorean_theorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem: In a right-angled triangle with legs of length 6 and 8, the hypotenuse has a length of 10 -/
theorem hypotenuse_of_6_8_triangle :
  ∃ (c : ℝ), pythagorean_theorem 6 8 c ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_6_8_triangle_l2326_232699


namespace NUMINAMATH_CALUDE_ab_value_l2326_232610

theorem ab_value (a b c d : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 25)
  (h3 : a = 2*c + Real.sqrt d) :
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2326_232610


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2326_232614

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 15) = 12 → x = 129 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2326_232614


namespace NUMINAMATH_CALUDE_distribution_count_4_3_l2326_232646

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribution_count (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is equal to 36 -/
theorem distribution_count_4_3 :
  distribution_count 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribution_count_4_3_l2326_232646


namespace NUMINAMATH_CALUDE_smartphone_price_difference_l2326_232606

/-- Calculates the final price after discount and tax --/
def finalPrice (basePrice : ℝ) (quantity : ℕ) (discount : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := basePrice * quantity * (1 - discount)
  discountedPrice * (1 + taxRate)

/-- Proves that the difference between Jane's and Tom's total costs is $112.68 --/
theorem smartphone_price_difference : 
  let storeAPrice := 125
  let storeBPrice := 130
  let storeADiscount := 0.12
  let storeBDiscount := 0.15
  let storeATaxRate := 0.07
  let storeBTaxRate := 0.05
  let tomQuantity := 2
  let janeQuantity := 3
  abs (finalPrice storeBPrice janeQuantity storeBDiscount storeBTaxRate - 
       finalPrice storeAPrice tomQuantity storeADiscount storeATaxRate - 112.68) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l2326_232606


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l2326_232672

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l2326_232672


namespace NUMINAMATH_CALUDE_runners_speed_l2326_232645

/-- The speeds of two runners on a circular track -/
theorem runners_speed (speed_a speed_b : ℝ) (track_length : ℝ) : 
  speed_a > 0 ∧ 
  speed_b > 0 ∧ 
  track_length > 0 ∧ 
  (speed_a + speed_b) * 48 = track_length ∧ 
  (speed_a - speed_b) * 600 = track_length ∧ 
  speed_a = speed_b + 2/3 → 
  speed_a = 9/2 ∧ speed_b = 23/6 := by sorry

end NUMINAMATH_CALUDE_runners_speed_l2326_232645


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2326_232684

-- Define a monic quartic polynomial
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + (f 0)

-- State the theorem
theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-1) = -1 →
  f 2 = -4 →
  f (-3) = -9 →
  f 4 = -16 →
  f 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2326_232684


namespace NUMINAMATH_CALUDE_mike_practice_hours_l2326_232636

/-- Calculates the total practice hours for a goalkeeper before a game -/
def total_practice_hours (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks_until_game : ℕ) : ℕ :=
  (weekday_hours * 5 + saturday_hours) * weeks_until_game

/-- Theorem: Mike's total practice hours before the next game -/
theorem mike_practice_hours :
  total_practice_hours 3 5 3 = 60 := by
  sorry

#eval total_practice_hours 3 5 3

end NUMINAMATH_CALUDE_mike_practice_hours_l2326_232636


namespace NUMINAMATH_CALUDE_factorization_problems_l2326_232650

theorem factorization_problems :
  (∀ x : ℝ, x^2 - 16 = (x + 4) * (x - 4)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2326_232650


namespace NUMINAMATH_CALUDE_last_two_average_l2326_232661

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 45 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 72.5 := by
sorry

end NUMINAMATH_CALUDE_last_two_average_l2326_232661


namespace NUMINAMATH_CALUDE_no_simple_algebraic_solution_l2326_232601

variable (g V₀ a S V t : ℝ)

def velocity_equation := V = g * t + V₀

def displacement_equation := S = (1/2) * g * t^2 + V₀ * t + (1/3) * a * t^3

theorem no_simple_algebraic_solution :
  ∀ g V₀ a S V t : ℝ,
  velocity_equation g V₀ V t →
  displacement_equation g V₀ a S t →
  ¬∃ f : ℝ → ℝ → ℝ → ℝ → ℝ, t = f S g V₀ a :=
by sorry

end NUMINAMATH_CALUDE_no_simple_algebraic_solution_l2326_232601


namespace NUMINAMATH_CALUDE_problem_statement_l2326_232673

def f (x : ℝ) : ℝ := x^2 - 1

theorem problem_statement :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 →
    (∀ m : ℝ, (4 * m^2 * |f x| + 4 * f m ≤ |f (x - 1)| ↔ -1/2 ≤ m ∧ m ≤ 1/2))) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 →
    ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = |2 * f x₂ - a * x₂|) ↔
      ((0 ≤ a ∧ a ≤ 3/2) ∨ a = 3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2326_232673


namespace NUMINAMATH_CALUDE_weight_difference_l2326_232681

-- Define the weights as real numbers
variable (W_A W_B W_C W_D W_E : ℝ)

-- Define the conditions
def condition1 : Prop := (W_A + W_B + W_C) / 3 = 80
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 82
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 81
def condition4 : Prop := W_A = 95
def condition5 : Prop := W_E > W_D

-- Theorem statement
theorem weight_difference (h1 : condition1 W_A W_B W_C)
                          (h2 : condition2 W_A W_B W_C W_D)
                          (h3 : condition3 W_B W_C W_D W_E)
                          (h4 : condition4 W_A)
                          (h5 : condition5 W_D W_E) : 
  W_E - W_D = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2326_232681


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2326_232656

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 4) (h_diagonal : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2326_232656


namespace NUMINAMATH_CALUDE_line_equivalence_slope_and_intercept_l2326_232619

/-- The vector representation of the line -/
def line_vector (x y : ℝ) : ℝ := 2 * (x - 3) + (-1) * (y - (-4))

/-- The slope-intercept form of the line -/
def line_slope_intercept (x y : ℝ) : Prop := y = 2 * x - 10

theorem line_equivalence :
  ∀ x y : ℝ, line_vector x y = 0 ↔ line_slope_intercept x y :=
sorry

theorem slope_and_intercept :
  ∃ m b : ℝ, (∀ x y : ℝ, line_vector x y = 0 → y = m * x + b) ∧ m = 2 ∧ b = -10 :=
sorry

end NUMINAMATH_CALUDE_line_equivalence_slope_and_intercept_l2326_232619


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2326_232635

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2326_232635


namespace NUMINAMATH_CALUDE_hat_knitting_time_l2326_232654

/-- Represents the time (in hours) to knit various items --/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- Calculates the total time to knit one set of clothes --/
def timeForOneSet (t : KnittingTimes) : ℝ :=
  t.hat + t.scarf + 2 * t.mitten + 2 * t.sock + t.sweater

/-- The main theorem stating that the time to knit a hat is 2 hours --/
theorem hat_knitting_time (t : KnittingTimes) 
  (h_scarf : t.scarf = 3)
  (h_mitten : t.mitten = 1)
  (h_sock : t.sock = 1.5)
  (h_sweater : t.sweater = 6)
  (h_total_time : 3 * timeForOneSet t = 48) : 
  t.hat = 2 := by
  sorry

end NUMINAMATH_CALUDE_hat_knitting_time_l2326_232654


namespace NUMINAMATH_CALUDE_blue_face_probability_is_five_eighths_l2326_232651

/-- An octahedron with blue and red faces -/
structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (total_is_sum : total_faces = blue_faces + red_faces)
  (total_is_eight : total_faces = 8)

/-- The probability of rolling a blue face on an octahedron -/
def blue_face_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Theorem: The probability of rolling a blue face on an octahedron with 5 blue faces out of 8 total faces is 5/8 -/
theorem blue_face_probability_is_five_eighths (o : Octahedron) 
  (h : o.blue_faces = 5) : blue_face_probability o = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_is_five_eighths_l2326_232651


namespace NUMINAMATH_CALUDE_black_and_white_drawing_cost_l2326_232693

/-- The cost of a black and white drawing -/
def black_and_white_cost : ℝ := 160

/-- The cost of a color drawing -/
def color_cost : ℝ := 240

/-- The size of the drawing -/
def drawing_size : ℕ × ℕ := (9, 13)

theorem black_and_white_drawing_cost :
  black_and_white_cost = 160 ∧
  color_cost = black_and_white_cost * 1.5 ∧
  color_cost = 240 := by
  sorry

end NUMINAMATH_CALUDE_black_and_white_drawing_cost_l2326_232693


namespace NUMINAMATH_CALUDE_range_of_f_triangle_properties_l2326_232623

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1/2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle where
  A := Real.pi / 3
  a := 2 * Real.sqrt 3
  b := 2
  c := 4

-- Theorem statements
theorem range_of_f : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1/2) 1 := sorry

theorem triangle_properties (t : Triangle) (h1 : 0 < t.A) (h2 : t.A < Real.pi / 2) 
  (h3 : t.a = 2 * Real.sqrt 3) (h4 : t.c = 4) (h5 : f t.A = 1) : 
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) := sorry

end

end NUMINAMATH_CALUDE_range_of_f_triangle_properties_l2326_232623


namespace NUMINAMATH_CALUDE_solve_for_y_l2326_232696

theorem solve_for_y (x y : ℝ) (h : 3 * x + 5 * y = 10) : y = 2 - (3/5) * x := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2326_232696


namespace NUMINAMATH_CALUDE_toaster_tax_rate_l2326_232659

/-- Calculates the mandatory state tax rate for a toaster purchase. -/
theorem toaster_tax_rate (msrp : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) : 
  msrp = 30 →
  total_paid = 54 →
  insurance_rate = 0.2 →
  (total_paid - msrp * (1 + insurance_rate)) / (msrp * (1 + insurance_rate)) = 0.5 := by
  sorry

#check toaster_tax_rate

end NUMINAMATH_CALUDE_toaster_tax_rate_l2326_232659


namespace NUMINAMATH_CALUDE_apples_used_l2326_232617

def initial_apples : ℕ := 40
def remaining_apples : ℕ := 39

theorem apples_used : initial_apples - remaining_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_used_l2326_232617


namespace NUMINAMATH_CALUDE_remainder_theorem_l2326_232616

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - x^2 + 3*x + 4

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = (x + 2) * q x + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2326_232616


namespace NUMINAMATH_CALUDE_minutes_to_hours_l2326_232698

-- Define the number of minutes Marcia spent
def minutes_spent : ℕ := 300

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem: 300 minutes is equal to 5 hours
theorem minutes_to_hours : 
  (minutes_spent : ℚ) / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_hours_l2326_232698


namespace NUMINAMATH_CALUDE_perimeter_of_eight_squares_l2326_232675

theorem perimeter_of_eight_squares (total_area : ℝ) (num_squares : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * num_squares - 2) * side_length + 2 * side_length
  perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_eight_squares_l2326_232675


namespace NUMINAMATH_CALUDE_distance_after_seven_seconds_l2326_232628

/-- The distance fallen by a freely falling body after t seconds -/
def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

/-- The time difference between the start of the two falling bodies -/
def time_difference : ℝ := 5

/-- The distance between the two falling bodies after t seconds -/
def distance_between (t : ℝ) : ℝ :=
  distance_fallen t - distance_fallen (t - time_difference)

/-- Theorem: The distance between the two falling bodies is 220.5 meters after 7 seconds -/
theorem distance_after_seven_seconds :
  distance_between 7 = 220.5 := by sorry

end NUMINAMATH_CALUDE_distance_after_seven_seconds_l2326_232628


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2326_232602

theorem square_sum_given_sum_and_product (a b : ℝ) : 
  a + b = 6 → a * b = 3 → a^2 + b^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2326_232602


namespace NUMINAMATH_CALUDE_quadratic_opens_downwards_iff_a_negative_l2326_232677

/-- A quadratic function of the form y = ax² - 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2

/-- The property that the graph of a quadratic function opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f ((x + y) / 2) > (f x + f y) / 2

theorem quadratic_opens_downwards_iff_a_negative (a : ℝ) :
  opens_downwards (quadratic_function a) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_opens_downwards_iff_a_negative_l2326_232677


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_equals_three_sqrt_two_l2326_232688

theorem sqrt_six_times_sqrt_three_equals_three_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_equals_three_sqrt_two_l2326_232688


namespace NUMINAMATH_CALUDE_range_of_a_l2326_232618

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2326_232618


namespace NUMINAMATH_CALUDE_arithmetic_sequence_35th_term_l2326_232663

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 35th term of the arithmetic sequence is 99. -/
theorem arithmetic_sequence_35th_term 
  (seq : ArithmeticSequence) 
  (h15 : seq.a 15 = 33) 
  (h25 : seq.a 25 = 66) : 
  seq.a 35 = 99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_35th_term_l2326_232663


namespace NUMINAMATH_CALUDE_trig_identity_l2326_232694

theorem trig_identity :
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) =
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2326_232694


namespace NUMINAMATH_CALUDE_ads_ratio_l2326_232641

def problem (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) : Prop :=
  ads_page1 = 12 ∧
  ads_page2 = 2 * ads_page1 ∧
  ads_page3 = ads_page2 + 24 ∧
  ads_page4 = (3 * ads_page2) / 4 ∧
  ads_clicked = 68

theorem ads_ratio (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) :
  problem ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked →
  (ads_clicked : ℚ) / (ads_page1 + ads_page2 + ads_page3 + ads_page4 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ads_ratio_l2326_232641


namespace NUMINAMATH_CALUDE_gcd_count_for_360_l2326_232670

theorem gcd_count_for_360 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.lcm a b * x = 360) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_360_l2326_232670


namespace NUMINAMATH_CALUDE_triangle_side_length_l2326_232676

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  Real.sin B = Real.sin (2 * A) →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  a * Real.sin C = c * Real.sin B →
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2326_232676


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l2326_232633

theorem recurring_decimal_sum : 
  let x := 1 / 3
  let y := 5 / 999
  let z := 7 / 9999
  x + y + z = 10170 / 29997 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l2326_232633


namespace NUMINAMATH_CALUDE_eight_ampersand_five_l2326_232604

def ampersand (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem eight_ampersand_five : ampersand 8 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_eight_ampersand_five_l2326_232604


namespace NUMINAMATH_CALUDE_task_completion_probability_l2326_232668

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 5/8)
  (h2 : p_task1_not_task2 = 1/4)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 3/5 ∧ p_task1 * (1 - p_task2) = p_task1_not_task2 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l2326_232668


namespace NUMINAMATH_CALUDE_base_conversion_3500_to_base_7_l2326_232634

theorem base_conversion_3500_to_base_7 :
  (1 * 7^4 + 3 * 7^3 + 1 * 7^2 + 3 * 7^1 + 0 * 7^0 : ℕ) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3500_to_base_7_l2326_232634


namespace NUMINAMATH_CALUDE_tangent_perpendicular_point_l2326_232691

def f (x : ℝ) := x^4 - x

theorem tangent_perpendicular_point :
  ∃! p : ℝ × ℝ, 
    p.2 = f p.1 ∧ 
    (4 * p.1^3 - 1) * (-1/3) = -1 ∧
    p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_point_l2326_232691


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2326_232692

/-- Given a geometric sequence {a_n} with a₁ = 2 and q = 2,
    prove that if the sum of the first n terms Sn = 126, then n = 6 -/
theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = (a 1) * (1 - 2^n) / (1 - 2) →
  S n = 126 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2326_232692


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l2326_232639

theorem sum_of_three_consecutive_even_numbers : 
  ∀ (a b c : ℕ), 
    a = 80 → 
    b = a + 2 → 
    c = b + 2 → 
    a + b + c = 246 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l2326_232639


namespace NUMINAMATH_CALUDE_negative_cube_root_of_negative_square_minus_one_l2326_232649

theorem negative_cube_root_of_negative_square_minus_one (a : ℝ) :
  ∃ x : ℝ, x < 0 ∧ x^3 = -a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_root_of_negative_square_minus_one_l2326_232649


namespace NUMINAMATH_CALUDE_square_side_length_l2326_232642

/-- Given a square and a regular hexagon where:
    1) The perimeter of the square equals the perimeter of the hexagon
    2) Each side of the hexagon measures 6 cm
    Prove that the length of one side of the square is 9 cm -/
theorem square_side_length (square_perimeter hexagon_perimeter : ℝ) 
  (hexagon_side : ℝ) (h1 : square_perimeter = hexagon_perimeter) 
  (h2 : hexagon_side = 6) (h3 : hexagon_perimeter = 6 * hexagon_side) :
  square_perimeter / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2326_232642


namespace NUMINAMATH_CALUDE_tank_fill_time_l2326_232640

/-- Given two pipes A and B, where A fills a tank in 56 minutes and B fills the tank 7 times as fast as A,
    the time to fill the tank when both pipes are open is 7 minutes. -/
theorem tank_fill_time (time_A : ℝ) (rate_B_multiplier : ℝ) : 
  time_A = 56 → rate_B_multiplier = 7 → 
  1 / (1 / time_A + rate_B_multiplier / time_A) = 7 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l2326_232640


namespace NUMINAMATH_CALUDE_delivery_pay_difference_l2326_232686

/-- Calculates the difference in pay between two delivery workers given their delivery counts and pay rate. -/
theorem delivery_pay_difference 
  (oula_deliveries : ℕ) 
  (tona_deliveries_ratio : ℚ) 
  (pay_per_delivery : ℕ) 
  (h1 : oula_deliveries = 96)
  (h2 : tona_deliveries_ratio = 3/4)
  (h3 : pay_per_delivery = 100) :
  (oula_deliveries * pay_per_delivery : ℕ) - (((tona_deliveries_ratio * oula_deliveries) : ℚ).floor * pay_per_delivery) = 2400 := by
  sorry

#check delivery_pay_difference

end NUMINAMATH_CALUDE_delivery_pay_difference_l2326_232686


namespace NUMINAMATH_CALUDE_triangle_properties_l2326_232600

noncomputable section

/-- Triangle ABC with internal angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.a * t.c * Real.cos t.B = -3

/-- The area of the triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  TriangleArea t = (3 * Real.sqrt 3) / 2 ∧
  t.b ≥ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2326_232600


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l2326_232620

theorem cosine_sum_identity : 
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l2326_232620


namespace NUMINAMATH_CALUDE_no_quinary_country_46_airlines_l2326_232680

/-- A quinary country is a country where each city is connected by air lines with exactly five other cities. -/
structure QuinaryCountry where
  cities : ℕ
  airLines : ℕ
  isQuinary : airLines = (cities * 5) / 2

/-- Theorem: There cannot exist a quinary country with exactly 46 air lines. -/
theorem no_quinary_country_46_airlines : ¬ ∃ (q : QuinaryCountry), q.airLines = 46 := by
  sorry

end NUMINAMATH_CALUDE_no_quinary_country_46_airlines_l2326_232680


namespace NUMINAMATH_CALUDE_handshake_problem_l2326_232643

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: Given 435 handshakes, there are 30 men -/
theorem handshake_problem :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = 435 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l2326_232643


namespace NUMINAMATH_CALUDE_magic_card_profit_theorem_l2326_232648

/-- Calculates the profit from selling a Magic card that increases in value -/
def magic_card_profit (initial_price : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_price * value_multiplier - initial_price

/-- Theorem: The profit from selling a Magic card that triples in value from $100 is $200 -/
theorem magic_card_profit_theorem :
  magic_card_profit 100 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_magic_card_profit_theorem_l2326_232648


namespace NUMINAMATH_CALUDE_derivative_sin_cos_x_l2326_232607

theorem derivative_sin_cos_x (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_x_l2326_232607


namespace NUMINAMATH_CALUDE_smallest_k_for_periodic_sum_l2326_232678

/-- Represents a rational number with a periodic decimal representation -/
structure PeriodicDecimal where
  numerator : ℤ
  period : ℕ+

/-- Returns true if the given natural number is the minimal period of the decimal representation -/
def is_minimal_period (r : ℚ) (p : ℕ+) : Prop :=
  ∃ (m : ℤ), r = m / (10^p.val - 1) ∧ 
  ∀ (q : ℕ+), q < p → ¬∃ (n : ℤ), r = n / (10^q.val - 1)

theorem smallest_k_for_periodic_sum (a b : PeriodicDecimal) : 
  (is_minimal_period (a.numerator / (10^30 - 1)) 30) →
  (is_minimal_period (b.numerator / (10^30 - 1)) 30) →
  (is_minimal_period ((a.numerator - b.numerator) / (10^30 - 1)) 15) →
  (∀ k : ℕ, k < 6 → ¬is_minimal_period ((a.numerator + k * b.numerator) / (10^30 - 1)) 15) →
  is_minimal_period ((a.numerator + 6 * b.numerator) / (10^30 - 1)) 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_periodic_sum_l2326_232678


namespace NUMINAMATH_CALUDE_ceil_zero_exists_ceil_minus_self_eq_point_two_l2326_232669

-- Define the ceiling function [x)
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1: [0) = 1
theorem ceil_zero : ceil 0 = 1 := by sorry

-- Theorem 2: There exists an x such that [x) - x = 0.2
theorem exists_ceil_minus_self_eq_point_two :
  ∃ x : ℝ, (ceil x : ℝ) - x = 0.2 := by sorry

end NUMINAMATH_CALUDE_ceil_zero_exists_ceil_minus_self_eq_point_two_l2326_232669


namespace NUMINAMATH_CALUDE_angle_E_measure_l2326_232647

-- Define the heptagon and its angles
structure Heptagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the properties of the heptagon
def is_valid_heptagon (h : Heptagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧ h.G > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F + h.G = 900

-- Define the conditions given in the problem
def satisfies_conditions (h : Heptagon) : Prop :=
  h.A = h.B ∧ h.A = h.C ∧ h.A = h.D ∧  -- A, B, C, D are congruent
  h.E = h.F ∧                          -- E and F are congruent
  h.A = h.E - 50 ∧                     -- A is 50° less than E
  h.G = 180 - h.E                      -- G is supplementary to E

-- The theorem to prove
theorem angle_E_measure (h : Heptagon) 
  (hvalid : is_valid_heptagon h) 
  (hcond : satisfies_conditions h) : 
  h.E = 184 := by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_angle_E_measure_l2326_232647


namespace NUMINAMATH_CALUDE_inverse_g_sum_l2326_232615

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_g_sum : 
  ∃ (a b c : ℝ), g a = -2 ∧ g b = 0 ∧ g c = 4 ∧ a + b + c = 6 :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l2326_232615


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2326_232690

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10^11 ∧ n < 10^12) ∧  -- 12-digit number
  (n % 36 = 0) ∧             -- divisible by 36
  (∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d)  -- contains each digit 0-9

theorem smallest_valid_number :
  (is_valid_number 100023457896) ∧
  (∀ m : ℕ, m < 100023457896 → ¬(is_valid_number m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2326_232690


namespace NUMINAMATH_CALUDE_f_inequality_l2326_232682

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the condition that f(x) > f'(x) for all x
axiom f_greater_than_f' : ∀ x, f x > f' x

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2326_232682


namespace NUMINAMATH_CALUDE_events_not_independent_l2326_232666

/- Define the sample space -/
def Ω : Type := Fin 10

/- Define the events A and B -/
def A : Set Ω := {ω : Ω | ω.val < 5}
def B : Set Ω := {ω : Ω | ω.val % 2 = 0}

/- Define the probability measure -/
def P : Set Ω → ℝ := sorry

/- State the theorem -/
theorem events_not_independent : ¬(P (A ∩ B) = P A * P B) := by sorry

end NUMINAMATH_CALUDE_events_not_independent_l2326_232666


namespace NUMINAMATH_CALUDE_inequality_proof_l2326_232689

theorem inequality_proof (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  (1 : ℝ) / ((n + 1 : ℝ) ^ (1 / k : ℝ)) + (1 : ℝ) / ((k + 1 : ℝ) ^ (1 / n : ℝ)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2326_232689


namespace NUMINAMATH_CALUDE_fraction_addition_l2326_232608

theorem fraction_addition : (4 : ℚ) / 510 + 25 / 34 = 379 / 510 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2326_232608


namespace NUMINAMATH_CALUDE_tan_double_angle_solution_l2326_232644

theorem tan_double_angle_solution (α : ℝ) (h : Real.tan (2 * α) = 4 / 3) :
  Real.tan α = -2 ∨ Real.tan α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_solution_l2326_232644


namespace NUMINAMATH_CALUDE_gasoline_price_decrease_l2326_232658

theorem gasoline_price_decrease (a : ℝ) : 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 8.1 ∧ 
    final_price = 7.8 ∧ 
    initial_price * (1 - a/100)^2 = final_price) → 
  8.1 * (1 - a/100)^2 = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_gasoline_price_decrease_l2326_232658


namespace NUMINAMATH_CALUDE_simultaneous_sound_arrival_l2326_232695

/-- Given a shooting range of length d meters, a bullet speed of c m/sec, and a speed of sound of s m/sec,
    the point x where the sound of the gunshot and the sound of the bullet hitting the target 
    arrive simultaneously is (d/2) * (1 + s/c) meters from the shooting position. -/
theorem simultaneous_sound_arrival (d c s : ℝ) (hd : d > 0) (hc : c > 0) (hs : s > 0) :
  let x := (d / 2) * (1 + s / c)
  (x / s) = (d / c + (d - x) / s) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_sound_arrival_l2326_232695


namespace NUMINAMATH_CALUDE_line_m_plus_b_l2326_232671

/-- A line passing through three given points has m + b = -1 -/
theorem line_m_plus_b (m b : ℝ) : 
  (3 = m * 3 + b) →  -- Line passes through (3, 3)
  (-1 = m * 1 + b) →  -- Line passes through (1, -1)
  (1 = m * 2 + b) →  -- Line passes through (2, 1)
  m + b = -1 := by
sorry

end NUMINAMATH_CALUDE_line_m_plus_b_l2326_232671


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2326_232625

theorem quadratic_roots_condition (p q : ℝ) : 
  (q < 0) ↔ (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2326_232625


namespace NUMINAMATH_CALUDE_train_distance_proof_l2326_232679

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 55

-- Define the total distance between stations
def total_distance : ℝ := 495

-- Theorem statement
theorem train_distance_proof :
  ∃ (time : ℝ) (distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed_train1 * time ∧
    distance2 = speed_train2 * time ∧
    distance2 = distance1 + distance_difference ∧
    total_distance = distance1 + distance2 :=
by
  sorry

end NUMINAMATH_CALUDE_train_distance_proof_l2326_232679


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_195195_l2326_232613

theorem sum_of_prime_factors_195195 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (195195 + 1))) id = 39) ∧ 
  (∀ p : ℕ, p ∈ Finset.filter Nat.Prime (Finset.range (195195 + 1)) ↔ p.Prime ∧ 195195 % p = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_195195_l2326_232613


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l2326_232603

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l2326_232603


namespace NUMINAMATH_CALUDE_shaded_area_problem_l2326_232652

/-- The area of the shaded region in a square with side length 40 units, 
    where two congruent triangles with base 20 units and height 20 units 
    are removed, is equal to 1200 square units. -/
theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 20 →
  triangle_height = 20 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l2326_232652


namespace NUMINAMATH_CALUDE_edward_tickets_l2326_232622

/-- The number of tickets Edward won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 3

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 4

/-- The number of candies Edward could buy -/
def candies_bought : ℕ := 2

/-- The number of tickets Edward won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

theorem edward_tickets : skee_ball_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_edward_tickets_l2326_232622


namespace NUMINAMATH_CALUDE_age_of_replaced_man_l2326_232627

/-- Given a group of 8 men where two are replaced by two women, prove the age of one of the replaced men. -/
theorem age_of_replaced_man
  (n : ℕ) -- Total number of people
  (m : ℕ) -- Number of men initially
  (w : ℕ) -- Number of women replacing men
  (A : ℝ) -- Initial average age of men
  (increase : ℝ) -- Increase in average age after replacement
  (known_man_age : ℕ) -- Age of one of the replaced men
  (women_avg_age : ℝ) -- Average age of the women
  (h1 : n = 8)
  (h2 : m = 8)
  (h3 : w = 2)
  (h4 : increase = 2)
  (h5 : known_man_age = 10)
  (h6 : women_avg_age = 23)
  : ∃ (other_man_age : ℕ), other_man_age = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_age_of_replaced_man_l2326_232627


namespace NUMINAMATH_CALUDE_triangle_value_l2326_232629

def base_7_to_10 (a b : ℕ) : ℕ := a * 7 + b

def base_9_to_10 (a b : ℕ) : ℕ := a * 9 + b

theorem triangle_value :
  ∃! t : ℕ, t < 10 ∧ base_7_to_10 5 t = base_9_to_10 t 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_value_l2326_232629


namespace NUMINAMATH_CALUDE_cubic_function_property_l2326_232683

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    prove that if f(-2) = -1, then f(2) = 3. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 1)
    (h2 : f (-2) = -1) : 
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2326_232683


namespace NUMINAMATH_CALUDE_michael_singles_percentage_l2326_232687

/-- Calculates the percentage of singles in a player's hits -/
def percentage_singles (total_hits : ℕ) (home_runs triples doubles : ℕ) : ℚ :=
  let non_singles := home_runs + triples + doubles
  let singles := total_hits - non_singles
  (singles : ℚ) / (total_hits : ℚ) * 100

theorem michael_singles_percentage :
  percentage_singles 50 2 3 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_michael_singles_percentage_l2326_232687


namespace NUMINAMATH_CALUDE_two_digit_product_777_l2326_232621

theorem two_digit_product_777 :
  ∀ a b : ℕ,
    10 ≤ a ∧ a < 100 →
    10 ≤ b ∧ b < 100 →
    a * b = 777 →
    ((a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_777_l2326_232621


namespace NUMINAMATH_CALUDE_reciprocal_of_abs_neg_three_l2326_232653

theorem reciprocal_of_abs_neg_three (x : ℝ) : x = |(-3)| → 1 / x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_abs_neg_three_l2326_232653


namespace NUMINAMATH_CALUDE_soccer_sideline_time_l2326_232660

/-- Given a soccer game duration and a player's playing times, calculate the time spent on the sideline -/
theorem soccer_sideline_time (game_duration playing_time1 playing_time2 : ℕ) 
  (h1 : game_duration = 90)
  (h2 : playing_time1 = 20)
  (h3 : playing_time2 = 35) :
  game_duration - (playing_time1 + playing_time2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_soccer_sideline_time_l2326_232660


namespace NUMINAMATH_CALUDE_rectangle_in_circle_area_l2326_232626

theorem rectangle_in_circle_area (r : ℝ) (w h : ℝ) :
  r = 5 ∧ w = 6 ∧ h = 2 →
  w * h ≤ π * r^2 →
  w * h = 12 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_area_l2326_232626


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2326_232664

theorem remainder_divisibility (x : ℕ) (h1 : x > 1) (h2 : ¬ Nat.Prime x) 
  (h3 : 5000 % x = 25) : 9995 % x = 25 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2326_232664


namespace NUMINAMATH_CALUDE_collinear_vectors_l2326_232630

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (e₁ e₂ : V)
variable (A B C : V)
variable (k : ℝ)

theorem collinear_vectors (h1 : e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ¬ ∃ (r : ℝ), e₁ = r • e₂)
  (h2 : B - A = 2 • e₁ + k • e₂)
  (h3 : C - B = e₁ - 3 • e₂)
  (h4 : ∃ (t : ℝ), C - A = t • (B - A)) :
  k = -6 := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2326_232630


namespace NUMINAMATH_CALUDE_subset_relation_l2326_232662

def set_A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def set_B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem subset_relation (a : ℝ) :
  (∀ x : ℝ, x ∈ set_B → x ∈ set_A 1) ∧
  (∀ x : ℝ, x ∈ set_A a → x ∈ set_B ↔ a < -8 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_subset_relation_l2326_232662


namespace NUMINAMATH_CALUDE_equation_solutions_l2326_232638

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (x + 3)^4 + (x + 1)^4 = 82} ∧ s = {0, -4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2326_232638


namespace NUMINAMATH_CALUDE_prob_both_counterfeit_value_l2326_232631

/-- Represents the total number of banknotes --/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes --/
def counterfeit_notes : ℕ := 5

/-- Represents the number of notes drawn --/
def drawn_notes : ℕ := 2

/-- Calculates the probability that both drawn notes are counterfeit given that at least one is counterfeit --/
def prob_both_counterfeit : ℚ :=
  (Nat.choose counterfeit_notes drawn_notes) / 
  (Nat.choose counterfeit_notes drawn_notes + 
   Nat.choose counterfeit_notes 1 * Nat.choose (total_notes - counterfeit_notes) 1)

theorem prob_both_counterfeit_value : 
  prob_both_counterfeit = 2 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_both_counterfeit_value_l2326_232631


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2326_232605

-- Define the conditions
def condition_p (x : ℝ) : Prop := |x + 1| ≤ 4
def condition_q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, condition_q x → condition_p x) ∧
  (∃ x, condition_p x ∧ ¬condition_q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2326_232605


namespace NUMINAMATH_CALUDE_cubic_divisibility_l2326_232637

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of the cubic polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℤ) : ℤ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Condition that one root is the product of the other two -/
def has_product_root (p : CubicPolynomial) : Prop :=
  ∃ (u v : ℚ), u ≠ 0 ∧ v ≠ 0 ∧ 
    (u + v + u*v = -p.a) ∧
    (u*v*(1 + u + v) = p.b) ∧
    (u^2 * v^2 = -p.c)

/-- Main theorem statement -/
theorem cubic_divisibility (p : CubicPolynomial) (h : has_product_root p) :
  (2 * p.eval (-1)) ∣ (p.eval 1 + p.eval (-1) - 2 * (1 + p.eval 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l2326_232637


namespace NUMINAMATH_CALUDE_father_age_triple_marika_age_2014_l2326_232674

/-- Represents a person with their birth year -/
structure Person where
  birthYear : ℕ

/-- Marika, born in 1996 -/
def marika : Person := ⟨1996⟩

/-- Marika's father, born in 1961 -/
def father : Person := ⟨1961⟩

/-- The year when Marika was 10 years old -/
def baseYear : ℕ := 2006

/-- Calculates a person's age in a given year -/
def age (p : Person) (year : ℕ) : ℕ :=
  year - p.birthYear

/-- Theorem stating that 2014 is the first year when the father's age is exactly three times Marika's age -/
theorem father_age_triple_marika_age_2014 :
  (∀ y : ℕ, y < 2014 → y ≥ baseYear → age father y ≠ 3 * age marika y) ∧
  age father 2014 = 3 * age marika 2014 :=
sorry

end NUMINAMATH_CALUDE_father_age_triple_marika_age_2014_l2326_232674


namespace NUMINAMATH_CALUDE_inequality_solution_l2326_232611

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 2 < y) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2326_232611


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2326_232657

def f (x : ℝ) := -x^2 + 2*x + 8

theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2326_232657


namespace NUMINAMATH_CALUDE_garden_tomatoes_count_l2326_232685

/-- Represents the garden layout and vegetable distribution --/
structure Garden where
  tomato_kinds : Nat
  cucumber_kinds : Nat
  cucumbers_per_kind : Nat
  potatoes : Nat
  rows : Nat
  spaces_per_row : Nat
  additional_capacity : Nat

/-- Calculates the number of tomatoes of each kind in the garden --/
def tomatoes_per_kind (g : Garden) : Nat :=
  let total_spaces := g.rows * g.spaces_per_row
  let occupied_spaces := g.cucumber_kinds * g.cucumbers_per_kind + g.potatoes
  let tomato_spaces := total_spaces - occupied_spaces - g.additional_capacity
  tomato_spaces / g.tomato_kinds

/-- Theorem stating that for the given garden configuration, 
    there are 5 tomatoes of each kind --/
theorem garden_tomatoes_count :
  let g : Garden := {
    tomato_kinds := 3,
    cucumber_kinds := 5,
    cucumbers_per_kind := 4,
    potatoes := 30,
    rows := 10,
    spaces_per_row := 15,
    additional_capacity := 85
  }
  tomatoes_per_kind g = 5 := by sorry

end NUMINAMATH_CALUDE_garden_tomatoes_count_l2326_232685


namespace NUMINAMATH_CALUDE_window_dimensions_correct_l2326_232624

/-- Represents the dimensions of a window -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a rectangular window with panes -/
def calculateWindowDimensions (x : ℝ) : WindowDimensions :=
  { width := 4 * x + 10,
    height := 9 * x + 8 }

theorem window_dimensions_correct (x : ℝ) :
  let numRows : ℕ := 3
  let numCols : ℕ := 4
  let numPanes : ℕ := 12
  let paneHeightToWidthRatio : ℝ := 3
  let borderWidth : ℝ := 2
  let dimensions := calculateWindowDimensions x
  (numRows * numCols = numPanes) ∧
  (numRows * (x * paneHeightToWidthRatio) + (numRows + 1) * borderWidth = dimensions.height) ∧
  (numCols * x + (numCols + 1) * borderWidth = dimensions.width) :=
by sorry

end NUMINAMATH_CALUDE_window_dimensions_correct_l2326_232624


namespace NUMINAMATH_CALUDE_decompose_50900300_l2326_232667

theorem decompose_50900300 :
  ∃ (ten_thousands ones : ℕ),
    50900300 = ten_thousands * 10000 + ones ∧
    ten_thousands = 5090 ∧
    ones = 300 := by
  sorry

end NUMINAMATH_CALUDE_decompose_50900300_l2326_232667


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2326_232665

theorem fraction_to_decimal (h : 343 = 7^3) : 7 / 343 = 0.056 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2326_232665


namespace NUMINAMATH_CALUDE_complex_division_l2326_232655

theorem complex_division : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_l2326_232655


namespace NUMINAMATH_CALUDE_mean_proportional_segments_l2326_232612

/-- Given that segment b is the mean proportional between segments a and c,
    prove that if a = 2 and b = 4, then c = 8. -/
theorem mean_proportional_segments (a b c : ℝ) 
  (h1 : b^2 = a * c) -- b is the mean proportional between a and c
  (h2 : a = 2)       -- a = 2 cm
  (h3 : b = 4)       -- b = 4 cm
  : c = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_segments_l2326_232612


namespace NUMINAMATH_CALUDE_digit1Sequence_1482_to_1484_l2326_232632

/-- A sequence of positive integers starting with digit 1 in increasing order -/
def digit1Sequence : ℕ → ℕ := sorry

/-- The nth digit in the concatenated sequence of digit1Sequence -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1482nd, 1483rd, and 1484th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1482) + 10 * (nthDigit 1483) + (nthDigit 1484)

theorem digit1Sequence_1482_to_1484 : targetNumber = 129 := by sorry

end NUMINAMATH_CALUDE_digit1Sequence_1482_to_1484_l2326_232632


namespace NUMINAMATH_CALUDE_sphere_cone_intersection_l2326_232697

/-- Represents the geometry of a sphere and cone with intersecting plane -/
structure GeometrySetup where
  R : ℝ  -- Radius of sphere and base of cone
  m : ℝ  -- Distance from base plane to intersecting plane
  n : ℝ  -- Ratio of truncated cone volume to spherical segment volume

/-- The areas of the circles cut from the sphere and cone are equal -/
def equal_areas (g : GeometrySetup) : Prop :=
  g.m = 2 * g.R / 5 ∨ g.m = 2 * g.R

/-- The volume ratio condition is satisfied -/
def volume_ratio_condition (g : GeometrySetup) : Prop :=
  g.n ≥ 1 / 2

/-- Main theorem combining both conditions -/
theorem sphere_cone_intersection (g : GeometrySetup) :
  (equal_areas g ↔ (2 * g.R * g.m - g.m^2 = g.R^2 * (1 - g.m / (2 * g.R))^2)) ∧
  (volume_ratio_condition g ↔ 
    (π * g.m / 12 * (12 * g.R^2 - 6 * g.R * g.m + g.m^2) = 
     g.n * (π * g.m^2 / 3 * (3 * g.R - g.m)))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_cone_intersection_l2326_232697


namespace NUMINAMATH_CALUDE_quadratic_sum_l2326_232609

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    there exist constants a, b, and c such that 
    f(x) = a(x+b)^2 + c for all x, and a + b + c = -1 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 4*x^2 - 40*x + 100 = a*(x+b)^2 + c) ∧ (a + b + c = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2326_232609
