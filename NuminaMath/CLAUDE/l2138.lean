import Mathlib

namespace power_of_three_plus_five_mod_eight_l2138_213831

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end power_of_three_plus_five_mod_eight_l2138_213831


namespace triangular_array_theorem_l2138_213845

/-- Represents the elements of a triangular array -/
def a (i j : ℕ) : ℚ :=
  sorry

/-- The common ratio for geometric sequences in rows -/
def common_ratio : ℚ := 1 / 2

/-- The common difference for the arithmetic sequence in the first column -/
def common_diff : ℚ := 1 / 4

theorem triangular_array_theorem (n : ℕ) (h : n > 0) :
  ∀ (i j : ℕ), i ≥ j → i > 0 → j > 0 →
  (∀ k, k > 0 → a k 1 - a (k-1) 1 = common_diff) →
  (∀ k l, k > 2 → l > 0 → a k (l+1) / a k l = common_ratio) →
  a n 3 = n / 16 := by
  sorry

end triangular_array_theorem_l2138_213845


namespace x_value_when_y_is_half_l2138_213864

theorem x_value_when_y_is_half :
  ∀ x y : ℚ, y = 2 / (4 * x + 2) → y = 1 / 2 → x = 1 / 2 := by
  sorry

end x_value_when_y_is_half_l2138_213864


namespace solve_system_l2138_213880

theorem solve_system (a b : ℝ) 
  (eq1 : 2020*a + 2030*b = 2050)
  (eq2 : 2030*a + 2040*b = 2060) : 
  a - b = -5 := by
sorry

end solve_system_l2138_213880


namespace simplify_sqrt_difference_l2138_213860

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = 3 - Real.sqrt 3 := by
sorry

end simplify_sqrt_difference_l2138_213860


namespace quadratic_equiv_abs_value_l2138_213825

theorem quadratic_equiv_abs_value : ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ |x - 8| = 3) ↔ (b = -16 ∧ c = 55) := by
  sorry

end quadratic_equiv_abs_value_l2138_213825


namespace seventeenth_group_number_l2138_213861

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroup : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  firstGroup + (groupNumber - 1) * interval

/-- Theorem: The 17th group number in the given systematic sampling is 264 -/
theorem seventeenth_group_number :
  systematicSample 800 50 8 17 = 264 := by
  sorry

end seventeenth_group_number_l2138_213861


namespace simplify_complex_fraction_l2138_213865

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction :
  (3 - 4 * i) / (5 - 2 * i) = 7 / 29 - (14 / 29) * i :=
by sorry

end simplify_complex_fraction_l2138_213865


namespace line_properties_l2138_213802

/-- Given a line passing through two points and a direction vector format, prove the value of 'a' and the x-intercept. -/
theorem line_properties (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 7) →
  p2 = (2, -2) →
  (∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -1)) →
  a = 5/9 ∧ 
  (∃ x : ℝ, x = 4 ∧ 0 = -x + 4) :=
by sorry

end line_properties_l2138_213802


namespace complex_fraction_simplification_l2138_213890

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I :=
by sorry

end complex_fraction_simplification_l2138_213890


namespace polynomial_value_theorem_l2138_213892

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_theorem (a b c d : ℝ) :
  P a b c d 1 = 7 →
  P a b c d 2 = 52 →
  P a b c d 3 = 97 →
  (P a b c d 9 + P a b c d (-5)) / 4 = 1202 :=
by sorry

end polynomial_value_theorem_l2138_213892


namespace charity_ticket_revenue_l2138_213883

/-- Represents the revenue from ticket sales -/
def TicketRevenue (f h d : ℕ) (p : ℚ) : ℚ :=
  f * p + h * (p / 2) + d * (2 * p)

theorem charity_ticket_revenue :
  ∃ (f h d : ℕ) (p : ℚ),
    f + h + d = 200 ∧
    TicketRevenue f h d p = 5000 ∧
    f * p = 4500 :=
by sorry

end charity_ticket_revenue_l2138_213883


namespace max_cuts_length_30x30_225pieces_l2138_213879

/-- Represents a square board with cuts along grid lines -/
structure Board where
  size : ℕ
  pieces : ℕ
  cuts_length : ℕ

/-- The maximum possible total length of cuts for a given board configuration -/
def max_cuts_length (b : Board) : ℕ :=
  (b.pieces * 10 - 4 * b.size) / 2

/-- Theorem stating the maximum possible total length of cuts for the given board -/
theorem max_cuts_length_30x30_225pieces :
  ∃ (b : Board), b.size = 30 ∧ b.pieces = 225 ∧ max_cuts_length b = 1065 := by
  sorry

end max_cuts_length_30x30_225pieces_l2138_213879


namespace quadratic_equality_l2138_213884

theorem quadratic_equality (p q : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x - 1) = x^2 + p*x + q) → 
  (p = 3 ∧ q = -4) := by
sorry

end quadratic_equality_l2138_213884


namespace billy_carnival_tickets_l2138_213842

theorem billy_carnival_tickets : ∀ (ferris_rides bumper_rides ticket_per_ride : ℕ),
  ferris_rides = 7 →
  bumper_rides = 3 →
  ticket_per_ride = 5 →
  (ferris_rides + bumper_rides) * ticket_per_ride = 50 := by
  sorry

end billy_carnival_tickets_l2138_213842


namespace g_neg_one_eq_neg_one_l2138_213862

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_eq_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
    sorry

end g_neg_one_eq_neg_one_l2138_213862


namespace sqrt_four_minus_one_l2138_213869

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end sqrt_four_minus_one_l2138_213869


namespace sampling_is_systematic_l2138_213826

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with rows and seats --/
structure Auditorium where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  auditorium : Auditorium
  seatNumberSelected : Nat

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.seatNumberSelected > 0 ∧ 
  strategy.seatNumberSelected ≤ strategy.auditorium.seatsPerRow ∧
  strategy.seatNumberSelected = strategy.seatNumberSelected

/-- Theorem stating that the given sampling strategy is systematic --/
theorem sampling_is_systematic (a : Auditorium) (s : SamplingStrategy) :
  a.rows = 25 → 
  a.seatsPerRow = 20 → 
  s.auditorium = a → 
  s.seatNumberSelected = 15 → 
  isSystematicSampling s := by
  sorry

#check sampling_is_systematic

end sampling_is_systematic_l2138_213826


namespace distribute_five_balls_three_boxes_l2138_213891

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes with no empty boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 3 := by
  sorry

end distribute_five_balls_three_boxes_l2138_213891


namespace linear_system_solution_l2138_213822

theorem linear_system_solution (x y : ℝ) : 
  3 * x + 2 * y = 2 → 2 * x + 3 * y = 8 → x + y = 2 := by
  sorry

end linear_system_solution_l2138_213822


namespace fraction_ratio_equality_l2138_213863

theorem fraction_ratio_equality (x : ℚ) : (2 / 3 : ℚ) / x = (3 / 5 : ℚ) / (7 / 15 : ℚ) → x = 14 / 27 := by
  sorry

end fraction_ratio_equality_l2138_213863


namespace not_divisible_by_four_sum_of_digits_l2138_213823

def numbers : List Nat := [3674, 3684, 3694, 3704, 3714, 3722]

theorem not_divisible_by_four_sum_of_digits : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    (n % 10 + (n / 10) % 10 = 11) := by
  sorry

end not_divisible_by_four_sum_of_digits_l2138_213823


namespace certain_number_problem_l2138_213811

theorem certain_number_problem (n x : ℝ) : 
  (n - 4) / x = 7 + (8 / x) → x = 6 → n = 54 := by sorry

end certain_number_problem_l2138_213811


namespace tims_books_l2138_213895

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end tims_books_l2138_213895


namespace expression_evaluation_l2138_213841

theorem expression_evaluation (x c : ℝ) (hx : x = 3) (hc : c = 2) :
  (x^2 + c)^2 - (x^2 - c)^2 = 72 := by sorry

end expression_evaluation_l2138_213841


namespace count_nines_in_range_l2138_213833

/-- The number of occurrences of the digit 9 in all integers from 1 to 1000 (inclusive) -/
def count_nines : ℕ := sorry

/-- The range of integers we're considering -/
def range_start : ℕ := 1
def range_end : ℕ := 1000

theorem count_nines_in_range : count_nines = 300 := by sorry

end count_nines_in_range_l2138_213833


namespace fraction_power_four_l2138_213898

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by sorry

end fraction_power_four_l2138_213898


namespace theater_attendance_l2138_213871

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) : 
  ∃ (num_children : ℕ), 
    num_children ≤ total_people ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue ∧
    num_children = 18 := by
  sorry

#check theater_attendance

end theater_attendance_l2138_213871


namespace at_most_one_negative_l2138_213836

theorem at_most_one_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b - c ≤ 0 → b + c - a > 0 ∧ c + a - b > 0) ∧
  (b + c - a ≤ 0 → a + b - c > 0 ∧ c + a - b > 0) ∧
  (c + a - b ≤ 0 → a + b - c > 0 ∧ b + c - a > 0) :=
sorry

end at_most_one_negative_l2138_213836


namespace sum_of_integers_l2138_213854

theorem sum_of_integers : (47 : ℤ) + (-27 : ℤ) = 20 := by
  sorry

end sum_of_integers_l2138_213854


namespace shoe_selection_theorem_l2138_213882

theorem shoe_selection_theorem (n : ℕ) (m : ℕ) (h : n = 5 ∧ m = 4) :
  (Nat.choose n 1) * (Nat.choose (n - 1) (m - 2)) * (Nat.choose 2 1) * (Nat.choose 2 1) = 120 :=
sorry

end shoe_selection_theorem_l2138_213882


namespace mn_equals_six_l2138_213872

/-- Given that -x³yⁿ and 3xᵐy² are like terms, prove that mn = 6 -/
theorem mn_equals_six (x y : ℝ) (m n : ℕ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → -x^3 * y^n = 3 * x^m * y^2) : 
  m * n = 6 := by
  sorry

end mn_equals_six_l2138_213872


namespace triangle_area_inequality_l2138_213843

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : α > 0) (h5 : β > 0) (h6 : γ > 0)
  (h7 : α = 2 * Real.sqrt (b * c))
  (h8 : β = 2 * Real.sqrt (c * a))
  (h9 : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by sorry

end triangle_area_inequality_l2138_213843


namespace nonagon_diagonal_intersection_probability_l2138_213899

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 4 vertices from 9 vertices -/
def num_four_vertices_choices (n : RegularNonagon) : ℕ := Nat.choose 9 4

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability of two randomly chosen diagonals intersecting inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_four_vertices_choices n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end nonagon_diagonal_intersection_probability_l2138_213899


namespace min_coefficient_value_l2138_213870

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ min_box : ℤ, (min_box = 34 ∧ box ≥ min_box) := by
  sorry

end min_coefficient_value_l2138_213870


namespace concert_attendance_l2138_213815

/-- Proves that given the initial ratio of women to men is 1:2, and after 12 women and 29 men left
    the ratio became 1:3, the original number of people at the concert was 21. -/
theorem concert_attendance (w m : ℕ) : 
  w / m = 1 / 2 →  -- Initial ratio of women to men
  (w - 12) / (m - 29) = 1 / 3 →  -- Ratio after some people left
  w + m = 21  -- Total number of people initially
  := by sorry

end concert_attendance_l2138_213815


namespace beautiful_arrangements_theorem_l2138_213877

/-- A beautiful arrangement of numbers 0 to n is a circular arrangement where 
    for any four distinct numbers a, b, c, d with a + c = b + d, 
    the chord joining a and c does not intersect the chord joining b and d -/
def is_beautiful_arrangement (n : ℕ) (arrangement : List ℕ) : Prop :=
  sorry

/-- M is the number of beautiful arrangements of numbers 0 to n -/
def M (n : ℕ) : ℕ :=
  sorry

/-- N is the number of pairs (x, y) of positive integers such that x + y ≤ n and gcd(x, y) = 1 -/
def N (n : ℕ) : ℕ :=
  sorry

/-- For any integer n ≥ 2, M(n) = N(n) + 1 -/
theorem beautiful_arrangements_theorem (n : ℕ) (h : n ≥ 2) : M n = N n + 1 :=
  sorry

end beautiful_arrangements_theorem_l2138_213877


namespace dalton_has_excess_money_l2138_213818

def jump_rope_cost : ℝ := 7
def board_game_cost : ℝ := 12
def ball_cost : ℝ := 4
def jump_rope_discount : ℝ := 2
def ball_discount : ℝ := 1
def jump_rope_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def ball_quantity : ℕ := 4
def allowance_savings : ℝ := 30
def uncle_money : ℝ := 25
def grandma_money : ℝ := 10
def sales_tax_rate : ℝ := 0.08

def total_cost_before_discounts : ℝ :=
  jump_rope_cost * jump_rope_quantity +
  board_game_cost * board_game_quantity +
  ball_cost * ball_quantity

def total_discounts : ℝ :=
  jump_rope_discount * jump_rope_quantity +
  ball_discount * ball_quantity

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discounts

def sales_tax : ℝ :=
  total_cost_after_discounts * sales_tax_rate

def final_total_cost : ℝ :=
  total_cost_after_discounts + sales_tax

def total_money_dalton_has : ℝ :=
  allowance_savings + uncle_money + grandma_money

theorem dalton_has_excess_money :
  total_money_dalton_has - final_total_cost = 9.92 := by sorry

end dalton_has_excess_money_l2138_213818


namespace triangle_side_length_l2138_213888

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  b = Real.sqrt 7 → 
  a = 3 → 
  Real.tan C = Real.sqrt 3 / 2 → 
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) → 
  c = 2 := by sorry

end triangle_side_length_l2138_213888


namespace regular_polygon_interior_angle_sum_l2138_213819

/-- A regular polygon with exterior angles of 45 degrees has interior angle sum of 1080 degrees -/
theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ), 
  n > 2 →
  (360 : ℝ) / n = 45 →
  (n - 2) * 180 = 1080 := by
sorry

end regular_polygon_interior_angle_sum_l2138_213819


namespace bacterial_growth_result_l2138_213807

/-- Represents the bacterial population growth model -/
structure BacterialGrowth where
  initial_population : ℕ
  triple_rate : ℕ  -- number of 5-minute intervals where population triples
  double_rate : ℕ  -- number of 10-minute intervals where population doubles

/-- Calculates the final population given a BacterialGrowth model -/
def final_population (model : BacterialGrowth) : ℕ :=
  model.initial_population * (3 ^ model.triple_rate) * (2 ^ model.double_rate)

/-- Theorem stating that under the given conditions, the final population is 16200 -/
theorem bacterial_growth_result :
  let model : BacterialGrowth := {
    initial_population := 50,
    triple_rate := 4,
    double_rate := 2
  }
  final_population model = 16200 := by sorry

end bacterial_growth_result_l2138_213807


namespace second_shop_payment_l2138_213878

/-- The amount Rahim paid for the books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) (first_shop_amount : ℕ) (average_price : ℕ) : ℕ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Theorem stating the amount Rahim paid for the books from the second shop -/
theorem second_shop_payment :
  second_shop_amount 40 20 600 14 = 240 := by
  sorry

end second_shop_payment_l2138_213878


namespace angle_bisector_coefficient_sum_l2138_213868

/-- Given a triangle ABC with vertices A = (-3, 2), B = (4, -1), and C = (-1, -5),
    the equation of the angle bisector of ∠A in the form dx + 2y + e = 0
    has coefficients d and e such that d + e equals a specific value. -/
theorem angle_bisector_coefficient_sum (d e : ℝ) : 
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ × ℝ := (-1, -5)
  ∃ (k : ℝ), d * A.1 + 2 * A.2 + e = k ∧
             d * B.1 + 2 * B.2 + e = 0 ∧
             d * C.1 + 2 * C.2 + e = 0 →
  d + e = sorry -- The exact value would be calculated here
:= by sorry


end angle_bisector_coefficient_sum_l2138_213868


namespace wheel_configuration_theorem_l2138_213851

/-- Represents a wheel with spokes -/
structure Wheel :=
  (spokes : ℕ)
  (spokes_le_three : spokes ≤ 3)

/-- Represents a configuration of wheels -/
def WheelConfiguration := List Wheel

/-- The total number of spokes in a configuration -/
def total_spokes (config : WheelConfiguration) : ℕ :=
  config.map Wheel.spokes |>.sum

/-- Theorem stating that 3 wheels are possible and 2 wheels are not possible -/
theorem wheel_configuration_theorem 
  (config : WheelConfiguration) 
  (total_spokes_ge_seven : total_spokes config ≥ 7) : 
  (∃ (three_wheel_config : WheelConfiguration), three_wheel_config.length = 3 ∧ total_spokes three_wheel_config ≥ 7) ∧
  (¬ ∃ (two_wheel_config : WheelConfiguration), two_wheel_config.length = 2 ∧ total_spokes two_wheel_config ≥ 7) :=
by sorry

end wheel_configuration_theorem_l2138_213851


namespace expansion_and_binomial_coeff_l2138_213824

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the sum of binomial coefficients for (a + b)^n
def sumBinomialCoeff (n : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (a + b)^n
def thirdTermCoeff (n : ℕ) : ℕ := sorry

theorem expansion_and_binomial_coeff :
  -- Part I: The term containing 1/x^2 in (2x^2 + 1/x)^5
  (binomial 5 4) * 2 = 10 ∧
  -- Part II: If sum of binomial coefficients in (2x^2 + 1/x)^5 is 28 less than
  -- the coefficient of the third term in (√x + 2/x)^n, then n = 6
  ∃ n : ℕ, sumBinomialCoeff 5 = thirdTermCoeff n - 28 → n = 6 :=
by sorry

end expansion_and_binomial_coeff_l2138_213824


namespace locus_is_conic_locus_degeneration_l2138_213887

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  vertex1 : Point
  vertex2 : Point

/-- The locus equation of point P on square S -/
def locusEquation (a b c x y : ℝ) : Prop :=
  (b^2 + c^2) * x^2 - 4 * a * c * x * y + (4 * a^2 + b^2 + c^2 - 4 * a * b) * y^2 = (b^2 + c^2 - 2 * a * b)^2

/-- The condition for locus degeneration -/
def degenerationCondition (a b c : ℝ) : Prop :=
  (a - b)^2 + c^2 = a^2

/-- Theorem stating that the locus of point P on square S is part of a conic -/
theorem locus_is_conic (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  locusEquation a b c P.x P.y :=
by sorry

/-- Theorem stating the condition for locus degeneration -/
theorem locus_degeneration (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  degenerationCondition a b c →
  ∃ (m k : ℝ), P.y = m * P.x + k :=
by sorry

end locus_is_conic_locus_degeneration_l2138_213887


namespace rocky_fights_l2138_213874

/-- Represents the number of fights Rocky boxed in his career. -/
def total_fights : ℕ := sorry

/-- The fraction of fights that were knockouts. -/
def knockout_fraction : ℚ := 1/2

/-- The fraction of knockouts that were in the first round. -/
def first_round_knockout_fraction : ℚ := 1/5

/-- The number of knockouts in the first round. -/
def first_round_knockouts : ℕ := 19

theorem rocky_fights : 
  total_fights = 190 ∧ 
  (knockout_fraction * first_round_knockout_fraction * total_fights : ℚ) = first_round_knockouts := by
  sorry

end rocky_fights_l2138_213874


namespace largest_sum_simplification_l2138_213840

theorem largest_sum_simplification :
  let sums := [1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/5, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/5) ∧ (1/3 + 1/5 = 8/15) := by
  sorry

end largest_sum_simplification_l2138_213840


namespace triangle_formation_l2138_213806

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬can_form_triangle 2 2 5 ∧
  ¬can_form_triangle 1 (Real.sqrt 3) 3 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end triangle_formation_l2138_213806


namespace tangent_lines_theorem_l2138_213850

noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

def tangent_line_at_2 (x y : ℝ) : Prop := y = x - 4

def tangent_lines_through_A (x y : ℝ) : Prop := y = x - 4 ∨ y = -2

theorem tangent_lines_theorem :
  (∀ x y : ℝ, y = f x → tangent_line_at_2 x y ↔ x = 2) ∧
  (∀ x y : ℝ, y = f x → tangent_lines_through_A x y ↔ (x = 2 ∧ y = -2) ∨ (x = 1 ∧ y = -2)) :=
sorry

end tangent_lines_theorem_l2138_213850


namespace expression_simplification_l2138_213859

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (1 / (x - 1) + 1 / (x + 1)) / (x^2 / (3 * x^2 - 3)) = 3 * Real.sqrt 2 := by
  sorry

end expression_simplification_l2138_213859


namespace alberts_earnings_l2138_213821

-- Define Albert's original earnings
def original_earnings : ℝ := 660

-- Theorem statement
theorem alberts_earnings :
  let scenario1 := original_earnings * 1.14 * 0.9
  let scenario2 := original_earnings * 1.15 * 1.2 * 0.9
  (scenario1 = 678) → (scenario2 = 819.72) := by
  sorry

end alberts_earnings_l2138_213821


namespace cassini_identity_l2138_213893

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Cassini's identity for Fibonacci numbers -/
theorem cassini_identity (n : ℕ) (h : n > 0) : 
  (fib (n + 1) * fib (n - 1) - fib n ^ 2 : ℤ) = (-1) ^ n := by
  sorry

end cassini_identity_l2138_213893


namespace initial_students_on_bus_l2138_213813

theorem initial_students_on_bus (students_left_bus : ℕ) (students_remaining : ℕ) 
  (h1 : students_left_bus = 3) 
  (h2 : students_remaining = 7) : 
  students_left_bus + students_remaining = 10 := by
sorry

end initial_students_on_bus_l2138_213813


namespace max_value_tangent_double_angle_l2138_213873

/-- Given a function f(x) = 3sin(x) + cos(x) that reaches its maximum value at x = α, 
    prove that tan(2α) = -3/4 -/
theorem max_value_tangent_double_angle (f : ℝ → ℝ) (α : ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin x + Real.cos x)
  (h₂ : IsLocalMax f α) : 
  Real.tan (2 * α) = -3/4 := by sorry

end max_value_tangent_double_angle_l2138_213873


namespace gathering_women_count_l2138_213805

/-- Represents a gathering with men and women dancing --/
structure Gathering where
  num_men : ℕ
  num_women : ℕ
  men_dance_count : ℕ
  women_dance_count : ℕ

/-- Theorem: In a gathering where each man dances with 4 women, each woman dances with 3 men, 
    and there are 15 men, the number of women is 20 --/
theorem gathering_women_count (g : Gathering) 
  (h1 : g.num_men = 15)
  (h2 : g.men_dance_count = 4)
  (h3 : g.women_dance_count = 3)
  : g.num_women = 20 := by
  sorry

end gathering_women_count_l2138_213805


namespace circle_area_from_diameter_endpoints_l2138_213804

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π := by sorry

end circle_area_from_diameter_endpoints_l2138_213804


namespace surjective_sum_iff_constant_l2138_213885

/-- A function is surjective if every element in the codomain is mapped to by at least one element in the domain. -/
def Surjective (f : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, f x = y

/-- The sum of two functions -/
def FunctionSum (f g : ℤ → ℤ) : ℤ → ℤ := λ x => f x + g x

/-- A function is constant if it maps all inputs to the same output -/
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

/-- The main theorem: a function f preserves surjectivity of g when added to it
    if and only if f is constant -/
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  (∀ g : ℤ → ℤ, Surjective g → Surjective (FunctionSum f g)) ↔ ConstantFunction f :=
sorry

end surjective_sum_iff_constant_l2138_213885


namespace A_equals_B_l2138_213803

-- Define set A
def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}

-- Define set B
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end A_equals_B_l2138_213803


namespace sum_squared_l2138_213827

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) :
  (x + y)^2 = 96 := by
sorry

end sum_squared_l2138_213827


namespace equilateral_triangle_division_l2138_213858

theorem equilateral_triangle_division : 
  ∃ (k m : ℕ), 2007 = 9 + 3 * k ∧ 2008 = 4 + 3 * m :=
by sorry

end equilateral_triangle_division_l2138_213858


namespace inverse_g_84_l2138_213829

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end inverse_g_84_l2138_213829


namespace circle_area_tripled_l2138_213886

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end circle_area_tripled_l2138_213886


namespace vector_operation_result_l2138_213817

/-- Prove that the vector operation (3, -6) - 5(1, -9) + (-1, 4) results in (-3, 43) -/
theorem vector_operation_result : 
  (⟨3, -6⟩ : ℝ × ℝ) - 5 • ⟨1, -9⟩ + ⟨-1, 4⟩ = ⟨-3, 43⟩ := by
  sorry

end vector_operation_result_l2138_213817


namespace inner_circle_to_triangle_ratio_l2138_213835

/-- The ratio of the area of the innermost circle to the area of the equilateral triangle --/
theorem inner_circle_to_triangle_ratio (s : ℝ) (h : s = 10) :
  let R := s * Real.sqrt 3 / 6
  let a := 2 * R
  let r := a / 2
  let A_triangle := Real.sqrt 3 / 4 * s^2
  let A_circle := Real.pi * r^2
  A_circle / A_triangle = Real.pi * Real.sqrt 3 := by sorry

end inner_circle_to_triangle_ratio_l2138_213835


namespace quadratic_coefficients_unique_l2138_213876

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficients_unique :
  ∀ a b c : ℝ,
    (∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c (-0.75)) ∧
    QuadraticFunction a b c (-0.75) = 3.25 ∧
    QuadraticFunction a b c 0 = 1 →
    a = -4 ∧ b = -6 ∧ c = 1 := by
  sorry

end quadratic_coefficients_unique_l2138_213876


namespace mode_of_data_set_l2138_213849

def data_set : List ℕ := [5, 4, 4, 3, 6, 2]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 4 := by
  sorry

end mode_of_data_set_l2138_213849


namespace john_used_four_quarters_l2138_213838

/-- The number of quarters John used to pay for a candy bar -/
def quarters_used (candy_cost dime_value nickel_value quarter_value : ℕ) 
  (num_dimes : ℕ) (change : ℕ) : ℕ :=
  ((candy_cost + change) - (num_dimes * dime_value + nickel_value)) / quarter_value

/-- Theorem stating that John used 4 quarters to pay for the candy bar -/
theorem john_used_four_quarters :
  quarters_used 131 10 5 25 3 4 = 4 := by
  sorry

end john_used_four_quarters_l2138_213838


namespace quadratic_solution_set_l2138_213875

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + x + b

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 ↔ quadratic_function a b x > 0) →
  a + b = -1 :=
by sorry

end quadratic_solution_set_l2138_213875


namespace union_A_B_intersection_complement_A_B_C_subset_B_iff_l2138_213810

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 2) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

theorem C_subset_B_iff (a : ℝ) : C a ⊆ B ↔ a ≤ 3 := by sorry

end union_A_B_intersection_complement_A_B_C_subset_B_iff_l2138_213810


namespace even_odd_periodic_properties_l2138_213894

-- Define the properties of even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of periodic functions
def IsPeriodic (f : ℝ → ℝ) : Prop := ∃ t : ℝ, t ≠ 0 ∧ ∀ x, f (x + t) = f x

-- State the theorem
theorem even_odd_periodic_properties 
  (f g : ℝ → ℝ) 
  (hf_even : IsEven f) 
  (hg_odd : IsOdd g) 
  (hf_periodic : IsPeriodic f) 
  (hg_periodic : IsPeriodic g) : 
  IsOdd (λ x ↦ g (g x)) ∧ IsPeriodic (λ x ↦ f x * g x) := by
  sorry

end even_odd_periodic_properties_l2138_213894


namespace unique_root_condition_l2138_213844

theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  a = (Real.log 6 + 1) / 4 := by
sorry

end unique_root_condition_l2138_213844


namespace triple_hash_40_l2138_213853

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_40 : hash (hash (hash 40)) = 3.86 := by
  sorry

end triple_hash_40_l2138_213853


namespace least_integer_greater_than_sqrt_450_l2138_213820

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 450 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 450 → m ≥ n :=
  sorry

end least_integer_greater_than_sqrt_450_l2138_213820


namespace kaleb_summer_earnings_l2138_213852

/-- Kaleb's lawn mowing business earnings --/
theorem kaleb_summer_earnings 
  (spring_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (total_amount : ℕ) 
  (h1 : spring_earnings = 4)
  (h2 : supplies_cost = 4)
  (h3 : total_amount = 50)
  : ℕ := by
  sorry

#check kaleb_summer_earnings

end kaleb_summer_earnings_l2138_213852


namespace lcm_gcd_product_12_15_l2138_213830

theorem lcm_gcd_product_12_15 : Nat.lcm 12 15 * Nat.gcd 12 15 = 180 := by
  sorry

end lcm_gcd_product_12_15_l2138_213830


namespace prob_both_red_is_one_ninth_l2138_213856

/-- The probability of drawing a red ball from both bags A and B -/
def prob_both_red (red_a white_a red_b white_b : ℕ) : ℚ :=
  (red_a : ℚ) / (red_a + white_a) * (red_b : ℚ) / (red_b + white_b)

/-- Theorem: The probability of drawing a red ball from both Bag A and Bag B is 1/9 -/
theorem prob_both_red_is_one_ninth :
  prob_both_red 4 2 1 5 = 1 / 9 := by
  sorry

#eval prob_both_red 4 2 1 5

end prob_both_red_is_one_ninth_l2138_213856


namespace root_between_roots_l2138_213867

theorem root_between_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + b = 0)
  (h2 : ∃ y : ℝ, y^2 - a*y + b = 0) :
  ∃ (x_1 y_1 z : ℝ), x_1^2 + a*x_1 + b = 0 ∧
                     y_1^2 - a*y_1 + b = 0 ∧
                     z^2 + 2*a*z + 2*b = 0 ∧
                     ((x_1 < z ∧ z < y_1) ∨ (y_1 < z ∧ z < x_1)) :=
by sorry

end root_between_roots_l2138_213867


namespace problem_solution_l2138_213866

theorem problem_solution (x y z : ℚ) : 
  x = 2/3 → y = 3/2 → z = 1/3 → (1/3) * x^7 * y^5 * z^4 = 11/600 := by
  sorry

end problem_solution_l2138_213866


namespace quadratic_minimum_l2138_213889

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end quadratic_minimum_l2138_213889


namespace fraction_equality_l2138_213846

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (1 / a + 1 / b = 1 / 2) → (a * b / (a + b) = 2) := by
sorry

end fraction_equality_l2138_213846


namespace book_sale_profit_percentage_l2138_213847

/-- Calculates the profit percentage for a book sale with given parameters. -/
theorem book_sale_profit_percentage
  (purchase_price : ℝ)
  (purchase_tax_rate : ℝ)
  (shipping_fee : ℝ)
  (selling_price : ℝ)
  (trading_tax_rate : ℝ)
  (h1 : purchase_price = 32)
  (h2 : purchase_tax_rate = 0.05)
  (h3 : shipping_fee = 2.5)
  (h4 : selling_price = 56)
  (h5 : trading_tax_rate = 0.07)
  : ∃ (profit_percentage : ℝ), abs (profit_percentage - 44.26) < 0.01 := by
  sorry


end book_sale_profit_percentage_l2138_213847


namespace power_evaluation_l2138_213897

theorem power_evaluation : (2 ^ 2) ^ (2 ^ (2 + 1)) = 65536 := by sorry

end power_evaluation_l2138_213897


namespace binomial_coefficient_8_3_l2138_213848

theorem binomial_coefficient_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end binomial_coefficient_8_3_l2138_213848


namespace set_union_problem_l2138_213814

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end set_union_problem_l2138_213814


namespace sum_of_solutions_is_eight_l2138_213828

theorem sum_of_solutions_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 7 ∧ N₂ * (N₂ - 8) = 7 ∧ N₁ + N₂ = 8 := by
  sorry

end sum_of_solutions_is_eight_l2138_213828


namespace rectangular_plot_theorem_l2138_213812

/-- Represents a rectangular plot with given properties --/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Theorem stating the properties of the rectangular plot --/
theorem rectangular_plot_theorem (plot : RectangularPlot) : 
  plot.length = plot.breadth + plot.length_breadth_difference ∧
  plot.total_fencing_cost = plot.fencing_cost_per_meter * (2 * plot.length + 2 * plot.breadth) ∧
  plot.length = 65 ∧
  plot.fencing_cost_per_meter = 26.5 ∧
  plot.total_fencing_cost = 5300 →
  plot.length_breadth_difference = 30 := by
  sorry

end rectangular_plot_theorem_l2138_213812


namespace basketball_scores_l2138_213881

theorem basketball_scores (total_players : ℕ) (less_than_yoongi : ℕ) (h1 : total_players = 21) (h2 : less_than_yoongi = 11) :
  total_players - less_than_yoongi - 1 = 8 := by
  sorry

end basketball_scores_l2138_213881


namespace intersection_empty_range_l2138_213896

theorem intersection_empty_range (a : ℝ) : 
  (∀ x : ℝ, (|x - a| < 1 → ¬(1 < x ∧ x < 5))) ↔ (a ≤ 0 ∨ a ≥ 6) := by
  sorry

end intersection_empty_range_l2138_213896


namespace parallel_lines_plane_count_l2138_213808

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the specifics of a line for this problem

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to count the number of planes determined by three lines -/
def count_planes (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem: The number of planes determined by three mutually parallel lines is either 1 or 3 -/
theorem parallel_lines_plane_count (l1 l2 l3 : Line3D) 
  (h1 : are_parallel l1 l2) 
  (h2 : are_parallel l2 l3) 
  (h3 : are_parallel l1 l3) : 
  count_planes l1 l2 l3 = 1 ∨ count_planes l1 l2 l3 = 3 :=
sorry

end parallel_lines_plane_count_l2138_213808


namespace tan_alpha_plus_pi_fourth_l2138_213857

theorem tan_alpha_plus_pi_fourth (α : Real) (m : Real) (h : m ≠ 0) :
  let P : Real × Real := (m, -2*m)
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end tan_alpha_plus_pi_fourth_l2138_213857


namespace ellipse_and_line_properties_l2138_213832

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b ∧ b > 0
  ecc : c / a = Real.sqrt 2 / 2
  perimeter : ℝ
  h_perimeter : perimeter = 4

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_m : m ≠ 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_line : ∀ x y, y = k * x + m
  h_intersect : A.1^2 / (E.b^2) + A.2^2 / (E.a^2) = 1 ∧
                B.1^2 / (E.b^2) + B.2^2 / (E.a^2) = 1
  h_relation : A.1 + 3 * B.1 = 4 * P.1 ∧ A.2 + 3 * B.2 = 4 * P.2

/-- The main theorem -/
theorem ellipse_and_line_properties (E : Ellipse) (L : IntersectingLine E) :
  (E.a = 1 ∧ E.b = Real.sqrt 2 / 2) ∧
  (L.m ∈ Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioo (1/2 : ℝ) 1) :=
sorry

end ellipse_and_line_properties_l2138_213832


namespace sequence_sum_eq_square_l2138_213839

def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (List.range n).sum

theorem sequence_sum_eq_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_eq_square_l2138_213839


namespace no_function_satisfies_equation_l2138_213809

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
sorry

end no_function_satisfies_equation_l2138_213809


namespace turtle_count_difference_l2138_213800

theorem turtle_count_difference (owen_initial : ℕ) (owen_final : ℕ) : 
  owen_initial = 21 →
  owen_final = 50 →
  ∃ (johanna_initial : ℕ),
    johanna_initial < owen_initial ∧
    owen_final = 2 * owen_initial + johanna_initial / 2 ∧
    owen_initial - johanna_initial = 5 :=
by sorry

end turtle_count_difference_l2138_213800


namespace election_total_votes_l2138_213834

/-- Represents an election with two candidates -/
structure Election where
  totalValidVotes : ℕ
  invalidVotes : ℕ
  losingCandidatePercentage : ℚ
  voteDifference : ℕ

/-- The total number of polled votes in the election -/
def totalPolledVotes (e : Election) : ℕ :=
  e.totalValidVotes + e.invalidVotes

/-- Theorem stating the total polled votes for the given election scenario -/
theorem election_total_votes (e : Election) 
  (h1 : e.losingCandidatePercentage = 1/5) 
  (h2 : e.voteDifference = 500) 
  (h3 : e.invalidVotes = 10) :
  totalPolledVotes e = 843 := by
  sorry

end election_total_votes_l2138_213834


namespace ajay_work_days_l2138_213837

/-- The number of days it takes Vijay to complete the work alone -/
def vijay_days : ℝ := 24

/-- The number of days it takes Ajay and Vijay to complete the work together -/
def together_days : ℝ := 6

/-- The number of days it takes Ajay to complete the work alone -/
noncomputable def ajay_days : ℝ := 
  (vijay_days * together_days) / (vijay_days - together_days)

theorem ajay_work_days : ajay_days = 8 := by
  sorry

end ajay_work_days_l2138_213837


namespace sales_after_three_years_l2138_213855

/-- The number of televisions sold initially -/
def initial_sales : ℕ := 327

/-- The annual increase rate as a percentage -/
def increase_rate : ℚ := 20 / 100

/-- The number of years for which the sales increase -/
def years : ℕ := 3

/-- Function to calculate sales after a given number of years -/
def sales_after_years (initial : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

/-- Theorem stating that the sales after 3 years is approximately 565 -/
theorem sales_after_three_years :
  ∃ ε > 0, |sales_after_years initial_sales increase_rate years - 565| < ε :=
sorry

end sales_after_three_years_l2138_213855


namespace equal_distribution_of_cards_l2138_213816

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 := by
  sorry

end equal_distribution_of_cards_l2138_213816


namespace conference_lefthandedness_l2138_213801

theorem conference_lefthandedness 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 7 * blue / 3) 
  (red_left : ℕ) 
  (blue_left : ℕ) 
  (h3 : red_left = red / 3) 
  (h4 : blue_left = 2 * blue / 3) : 
  (red_left + blue_left : ℚ) / total = 13 / 30 := by
sorry

end conference_lefthandedness_l2138_213801
