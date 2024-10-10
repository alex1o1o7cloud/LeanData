import Mathlib

namespace overall_rate_relation_l2126_212689

/-- Given three deposit amounts and their respective interest rates, 
    this theorem proves the relation for the overall annual percentage rate. -/
theorem overall_rate_relation 
  (P1 P2 P3 : ℝ) 
  (R1 R2 R3 : ℝ) 
  (h1 : P1 * (1 + R1)^2 + P2 * (1 + R2)^2 + P3 * (1 + R3)^2 = 2442)
  (h2 : P1 * (1 + R1)^3 + P2 * (1 + R2)^3 + P3 * (1 + R3)^3 = 2926) :
  ∃ R : ℝ, (1 + R)^3 / (1 + R)^2 = 2926 / 2442 :=
sorry

end overall_rate_relation_l2126_212689


namespace simplify_expression_l2126_212645

theorem simplify_expression : ((4 + 6) * 2) / 4 - 3 / 4 = 17 / 4 := by
  sorry

end simplify_expression_l2126_212645


namespace existence_of_numbers_l2126_212616

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem existence_of_numbers : 
  ∃ (a b c : ℕ), 
    sum_of_digits (a + b) < 5 ∧ 
    sum_of_digits (a + c) < 5 ∧ 
    sum_of_digits (b + c) < 5 ∧ 
    sum_of_digits (a + b + c) > 50 := by
  sorry

end existence_of_numbers_l2126_212616


namespace face_mask_profit_l2126_212649

/-- Calculate the total profit from selling face masks --/
theorem face_mask_profit (num_boxes : ℕ) (masks_per_box : ℕ) (total_cost : ℚ) (selling_price : ℚ) :
  num_boxes = 3 →
  masks_per_box = 20 →
  total_cost = 15 →
  selling_price = 1/2 →
  (num_boxes * masks_per_box : ℚ) * selling_price - total_cost = 15 := by
  sorry

end face_mask_profit_l2126_212649


namespace art_gallery_problem_l2126_212635

theorem art_gallery_problem (total_pieces : ℕ) 
  (h1 : total_pieces / 3 = total_pieces - (total_pieces * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total_pieces / 3) / 6 = (total_pieces / 3) - (5 * total_pieces / 18))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total_pieces * 2 / 3) / 3 = (total_pieces * 2 / 3) - (2 * total_pieces / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : 2 * (total_pieces * 2 / 3) / 3 = 1200)  -- 1200 sculptures are not on display
  : total_pieces = 2700 := by
  sorry

end art_gallery_problem_l2126_212635


namespace polynomial_roots_l2126_212640

theorem polynomial_roots : 
  let p (x : ℚ) := 6*x^5 + 29*x^4 - 71*x^3 - 10*x^2 + 24*x + 8
  (p (-2) = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p 1 = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-2/3) = 0) :=
by sorry

end polynomial_roots_l2126_212640


namespace salary_problem_l2126_212678

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h_total : total_salary = 5000)
  (h_a_spend : a_spend_rate = 0.95)
  (h_b_spend : b_spend_rate = 0.85)
  (h_equal_savings : (1 - a_spend_rate) * a_salary = (1 - b_spend_rate) * b_salary)
  (h_total_sum : a_salary + b_salary = total_salary) :
  a_salary = 3750 :=
by
  sorry

end salary_problem_l2126_212678


namespace geometric_progression_sum_inequality_l2126_212612

/-- An increasing positive geometric progression -/
def IsIncreasingPositiveGP (b : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem geometric_progression_sum_inequality 
  (b : ℕ → ℝ) 
  (h_gp : IsIncreasingPositiveGP b) 
  (h_sum : b 4 + b 3 - b 2 - b 1 = 5) : 
  b 6 + b 5 ≥ 20 := by
sorry

end geometric_progression_sum_inequality_l2126_212612


namespace max_q_minus_r_l2126_212605

theorem max_q_minus_r (q r : ℕ) (h : 1025 = 23 * q + r) (hq : q > 0) (hr : r > 0) :
  q - r ≤ 31 ∧ ∃ (q' r' : ℕ), 1025 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0 ∧ q' - r' = 31 :=
sorry

end max_q_minus_r_l2126_212605


namespace abc_divisible_by_four_l2126_212658

theorem abc_divisible_by_four (a b c d : ℤ) (h : a^2 + b^2 + c^2 = d^2) : 
  4 ∣ (a * b * c) := by
  sorry

end abc_divisible_by_four_l2126_212658


namespace product_of_seven_and_sum_l2126_212665

theorem product_of_seven_and_sum (x : ℝ) : 27 - 7 = x * 5 → 7 * (x + 5) = 63 := by
  sorry

end product_of_seven_and_sum_l2126_212665


namespace streamer_hourly_rate_l2126_212681

/-- A streamer's weekly schedule and earnings --/
structure StreamerSchedule where
  daysOff : ℕ
  hoursPerStreamDay : ℕ
  weeklyEarnings : ℕ

/-- Calculate the hourly rate of a streamer --/
def hourlyRate (s : StreamerSchedule) : ℚ :=
  s.weeklyEarnings / ((7 - s.daysOff) * s.hoursPerStreamDay)

/-- Theorem stating that given the specific conditions, the hourly rate is $10 --/
theorem streamer_hourly_rate :
  let s : StreamerSchedule := {
    daysOff := 3,
    hoursPerStreamDay := 4,
    weeklyEarnings := 160
  }
  hourlyRate s = 10 := by
  sorry

end streamer_hourly_rate_l2126_212681


namespace greatest_prime_divisor_digit_sum_l2126_212683

def n : ℕ := 2^15 - 1

theorem greatest_prime_divisor_digit_sum :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧
    (Nat.digits 10 p).sum = 8) :=
by sorry

end greatest_prime_divisor_digit_sum_l2126_212683


namespace problem_solution_l2126_212638

theorem problem_solution (x y : ℝ) (hx : x = 1/2) (hy : y = 2) :
  (1/3) * x^8 * y^9 = 2/3 := by
sorry

end problem_solution_l2126_212638


namespace tangent_perpendicular_implies_negative_a_l2126_212652

/-- Given a real-valued function f(x) = ax³ + ln x, prove that if there exists a positive real number x
    such that the derivative of f at x is zero, then a is negative. -/
theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 3 * a * x^2 + 1 / x = 0) → a < 0 := by
  sorry

end tangent_perpendicular_implies_negative_a_l2126_212652


namespace log_product_problem_l2126_212608

theorem log_product_problem (c d : ℕ+) : 
  (d.val - c.val - 1 = 435) →  -- Number of terms is 435
  (Real.log d.val / Real.log c.val = 3) →  -- Value of the product is 3
  (c.val + d.val = 130) := by
sorry

end log_product_problem_l2126_212608


namespace goldfish_sales_l2126_212622

theorem goldfish_sales (buy_price sell_price tank_cost shortfall_percent : ℚ) 
  (h1 : buy_price = 25 / 100)
  (h2 : sell_price = 75 / 100)
  (h3 : tank_cost = 100)
  (h4 : shortfall_percent = 45 / 100) :
  (tank_cost * (1 - shortfall_percent)) / (sell_price - buy_price) = 110 := by
sorry

end goldfish_sales_l2126_212622


namespace parabola_angle_theorem_l2126_212614

/-- The parabola y² = 4x with focus F -/
structure Parabola where
  F : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  M : ℝ × ℝ
  on_parabola : p.equation M.1 M.2

/-- The foot of the perpendicular from a point to the directrix -/
def footOfPerpendicular (p : Parabola) (point : PointOnParabola p) : ℝ × ℝ := sorry

/-- The angle between two vectors -/
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_angle_theorem (p : Parabola) (M : PointOnParabola p) :
  p.F = (1, 0) →
  p.equation = fun x y ↦ y^2 = 4*x →
  ‖M.M - p.F‖ = 4/3 →
  angle (footOfPerpendicular p M - M.M) (p.F - M.M) = 2*π/3 := by sorry

end parabola_angle_theorem_l2126_212614


namespace units_digit_of_x_l2126_212687

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_x (p x : ℕ) (h1 : p * x = 32^10)
  (h2 : has_units_digit p 6) (h3 : x % 4 = 0) :
  has_units_digit x 1 :=
sorry

end units_digit_of_x_l2126_212687


namespace smallest_stair_count_l2126_212623

theorem smallest_stair_count : ∃ n : ℕ, n = 71 ∧ n > 15 ∧ 
  n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧
  ∀ m : ℕ, m > 15 → m % 3 = 2 → m % 7 = 1 → m % 4 = 3 → m ≥ n :=
by sorry

end smallest_stair_count_l2126_212623


namespace absolute_value_equation_solution_l2126_212672

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 5| = 3*x + 2) ↔ (x = 3/5) := by
  sorry

end absolute_value_equation_solution_l2126_212672


namespace runners_speed_ratio_l2126_212697

theorem runners_speed_ratio :
  ∀ (C : ℝ) (v_V v_P : ℝ),
  C > 0 → v_V > 0 → v_P > 0 →
  (∃ (t_1 : ℝ), t_1 > 0 ∧ v_V * t_1 + v_P * t_1 = C) →
  (∃ (t_2 : ℝ), t_2 > 0 ∧ v_V * t_2 = C + v_V * (C / (v_V + v_P)) ∧ v_P * t_2 = C + v_P * (C / (v_V + v_P))) →
  v_V / v_P = (1 + Real.sqrt 5) / 2 :=
sorry

end runners_speed_ratio_l2126_212697


namespace three_digit_cubes_divisible_by_27_l2126_212659

theorem three_digit_cubes_divisible_by_27 :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧ 
    s.card = 2) :=
by sorry

end three_digit_cubes_divisible_by_27_l2126_212659


namespace smallest_number_divisible_l2126_212639

theorem smallest_number_divisible (a b c d e f : ℕ) (h1 : a = 35 ∧ b = 66)
  (h2 : c = 28 ∧ d = 165) (h3 : e = 25 ∧ f = 231) :
  ∃ (n : ℚ), n = 700 / 33 ∧
  (∃ (k1 k2 k3 : ℕ), n / (a / b) = k1 ∧ n / (c / d) = k2 ∧ n / (e / f) = k3) ∧
  ∀ (m : ℚ), m < n →
  ¬(∃ (l1 l2 l3 : ℕ), m / (a / b) = l1 ∧ m / (c / d) = l2 ∧ m / (e / f) = l3) :=
sorry

end smallest_number_divisible_l2126_212639


namespace largest_n_is_correct_l2126_212631

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function representing the quadratic expression 3x^2 + nx + 90 -/
def quadratic (n : ℕ) (x : ℚ) : ℚ := 3 * x^2 + n * x + 90

/-- A predicate that checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def has_integer_linear_factors (n : ℕ) : Prop :=
  ∃ (a b c d : ℤ), ∀ (x : ℚ), quadratic n x = (a * x + b) * (c * x + d)

theorem largest_n_is_correct :
  (∀ n : ℕ, n > largest_n → ¬(has_integer_linear_factors n)) ∧
  (has_integer_linear_factors largest_n) :=
by sorry


end largest_n_is_correct_l2126_212631


namespace shaded_area_is_63_l2126_212673

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect1 : Rectangle
  rect2 : Rectangle
  overlap : Rectangle

/-- Calculates the shaded area formed by two intersecting rectangles -/
def IntersectingRectangles.shadedArea (ir : IntersectingRectangles) : ℝ :=
  ir.rect1.area + ir.rect2.area - ir.overlap.area

/-- The main theorem stating that the shaded area is 63 square units -/
theorem shaded_area_is_63 (ir : IntersectingRectangles)
  (h1 : ir.rect1 = { width := 4, height := 12 })
  (h2 : ir.rect2 = { width := 5, height := 7 })
  (h3 : ir.overlap = { width := 4, height := 5 }) :
  ir.shadedArea = 63 := by
  sorry

#check shaded_area_is_63

end shaded_area_is_63_l2126_212673


namespace arithmetic_sequence_problem_l2126_212692

/-- Arithmetic sequence with first term 4 and common difference 2 -/
def a (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (2 * 4 + (n - 1) * 2) / 2

/-- The proposition to be proved -/
theorem arithmetic_sequence_problem :
  ∃ (k : ℕ), k > 0 ∧ S k - a (k + 5) = 44 ∧ k = 7 := by
  sorry


end arithmetic_sequence_problem_l2126_212692


namespace inequality_proof_l2126_212619

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 3) :
  Real.sqrt (3 - ((x + y) / 2)^2) + Real.sqrt (3 - ((y + z) / 2)^2) + Real.sqrt (3 - ((z + x) / 2)^2) ≥ 3 * Real.sqrt 2 := by
  sorry

end inequality_proof_l2126_212619


namespace consecutive_four_product_plus_one_is_square_l2126_212680

theorem consecutive_four_product_plus_one_is_square (x : ℤ) :
  ∃ y : ℤ, x * (x + 1) * (x + 2) * (x + 3) + 1 = y ^ 2 := by
  sorry

end consecutive_four_product_plus_one_is_square_l2126_212680


namespace sum_is_composite_l2126_212602

theorem sum_is_composite (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a : ℕ) + b + c + d = x * y :=
by
  sorry

end sum_is_composite_l2126_212602


namespace decision_box_two_exits_l2126_212669

-- Define the types of program blocks
inductive ProgramBlock
  | TerminationBox
  | InputOutputBox
  | ProcessingBox
  | DecisionBox

-- Define a function that returns the number of exit directions for each program block
def exitDirections (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.TerminationBox => 1
  | ProgramBlock.InputOutputBox => 1
  | ProgramBlock.ProcessingBox => 1
  | ProgramBlock.DecisionBox => 2

-- Theorem statement
theorem decision_box_two_exits :
  ∀ (block : ProgramBlock), exitDirections block = 2 ↔ block = ProgramBlock.DecisionBox :=
by sorry


end decision_box_two_exits_l2126_212669


namespace four_digit_numbers_with_6_or_8_l2126_212690

/-- The number of four-digit numbers -/
def total_four_digit_numbers : ℕ := 9000

/-- The number of digits that are not 6 or 8 for the first digit -/
def first_digit_choices : ℕ := 7

/-- The number of digits that are not 6 or 8 for the other digits -/
def other_digit_choices : ℕ := 8

/-- The number of four-digit numbers without 6 or 8 -/
def numbers_without_6_or_8 : ℕ := first_digit_choices * other_digit_choices * other_digit_choices * other_digit_choices

theorem four_digit_numbers_with_6_or_8 :
  total_four_digit_numbers - numbers_without_6_or_8 = 5416 := by
  sorry

end four_digit_numbers_with_6_or_8_l2126_212690


namespace volunteer_distribution_count_l2126_212615

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution_count : 
  distribute_volunteers 5 4 = 216 :=
sorry

end volunteer_distribution_count_l2126_212615


namespace saturday_sales_proof_l2126_212620

/-- The number of caricatures sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The price of each caricature in dollars -/
def price_per_caricature : ℚ := 20

/-- The number of caricatures sold on Sunday -/
def sunday_sales : ℕ := 16

/-- The total revenue for the weekend in dollars -/
def total_revenue : ℚ := 800

theorem saturday_sales_proof : 
  saturday_sales = 24 ∧ 
  price_per_caricature * (saturday_sales + sunday_sales : ℚ) = total_revenue :=
by sorry

end saturday_sales_proof_l2126_212620


namespace sara_peaches_theorem_l2126_212624

/-- The number of peaches Sara picked initially -/
def initial_peaches : ℝ := 61

/-- The number of peaches Sara picked at the orchard -/
def orchard_peaches : ℝ := 24.0

/-- The total number of peaches Sara picked -/
def total_peaches : ℝ := 85

/-- Theorem stating that the initial number of peaches plus the orchard peaches equals the total peaches -/
theorem sara_peaches_theorem : initial_peaches + orchard_peaches = total_peaches := by
  sorry

end sara_peaches_theorem_l2126_212624


namespace sqrt_two_squared_times_three_l2126_212644

theorem sqrt_two_squared_times_three : 4 - (Real.sqrt 2)^2 * 3 = -2 := by
  sorry

end sqrt_two_squared_times_three_l2126_212644


namespace rectangle_area_l2126_212662

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length rectangle_area : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_area = rectangle_width * rectangle_length →
  rectangle_area = 108 := by
  sorry

end rectangle_area_l2126_212662


namespace number_of_officers_l2126_212611

/-- Prove the number of officers in an office given average salaries and number of non-officers -/
theorem number_of_officers
  (avg_salary : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_non_officers : ℕ)
  (h1 : avg_salary = 120)
  (h2 : avg_salary_officers = 430)
  (h3 : avg_salary_non_officers = 110)
  (h4 : num_non_officers = 465) :
  ∃ (num_officers : ℕ),
    avg_salary * (num_officers + num_non_officers) =
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers ∧
    num_officers = 15 :=
by sorry

end number_of_officers_l2126_212611


namespace quadratic_roots_ratio_l2126_212634

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_roots_ratio_l2126_212634


namespace equation_describes_ellipse_and_hyperbola_l2126_212694

/-- A conic section type -/
inductive ConicSection
  | Ellipse
  | Hyperbola
  | Parabola
  | Circle
  | Point
  | Line
  | CrossedLines

/-- Represents the equation y^4 - 8x^4 = 8y^2 - 4 -/
def equation (x y : ℝ) : Prop :=
  y^4 - 8*x^4 = 8*y^2 - 4

/-- The set of conic sections described by the equation -/
def describedConicSections : Set ConicSection :=
  {ConicSection.Ellipse, ConicSection.Hyperbola}

/-- Theorem stating that the equation describes the union of an ellipse and a hyperbola -/
theorem equation_describes_ellipse_and_hyperbola :
  ∀ x y : ℝ, equation x y → 
  ∃ (c : ConicSection), c ∈ describedConicSections :=
sorry

end equation_describes_ellipse_and_hyperbola_l2126_212694


namespace card_sum_difference_l2126_212625

theorem card_sum_difference (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n > 4)
  (h_a : ∀ m ∈ Finset.range (2*n + 5), ⌊a m⌋ = m) :
  ∃ (i j k l : ℕ), i ∈ Finset.range (2*n + 5) ∧ 
                   j ∈ Finset.range (2*n + 5) ∧ 
                   k ∈ Finset.range (2*n + 5) ∧ 
                   l ∈ Finset.range (2*n + 5) ∧
                   i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                   |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) :=
sorry

end card_sum_difference_l2126_212625


namespace quadratic_inequality_l2126_212671

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by
  sorry

end quadratic_inequality_l2126_212671


namespace jones_family_probability_l2126_212651

theorem jones_family_probability :
  let n : ℕ := 8  -- total number of children
  let k : ℕ := 4  -- number of sons (or daughters)
  let p : ℚ := 1/2  -- probability of a child being a son (or daughter)
  Nat.choose n k * p^k * (1-p)^(n-k) = 35/128 :=
by sorry

end jones_family_probability_l2126_212651


namespace range_of_a_l2126_212609

def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 1 ∨ (-1 < a ∧ a < 1) :=
by sorry

end range_of_a_l2126_212609


namespace total_cookies_sum_l2126_212627

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def brother_got : ℕ := 1

/-- The number of cookies taken by the first friend -/
def first_friend_took : ℕ := 3

/-- The number of cookies taken by the second friend -/
def second_friend_took : ℕ := 5

/-- The number of cookies taken by the third friend -/
def third_friend_took : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies is the sum of all distributed and remaining cookies -/
theorem total_cookies_sum : 
  total_cookies = kristy_ate + brother_got + first_friend_took + 
                  second_friend_took + third_friend_took + cookies_left :=
by sorry

end total_cookies_sum_l2126_212627


namespace exists_number_of_1_and_2_divisible_by_2_pow_l2126_212600

/-- A function that checks if a natural number is composed of only digits 1 and 2 -/
def isComposedOf1And2 (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → d = 1 ∨ d = 2

/-- Theorem stating that for all natural numbers n, there exists a number x
    composed of only digits 1 and 2 such that x is divisible by 2^n -/
theorem exists_number_of_1_and_2_divisible_by_2_pow (n : ℕ) :
  ∃ x : ℕ, isComposedOf1And2 x ∧ (2^n ∣ x) := by
  sorry

end exists_number_of_1_and_2_divisible_by_2_pow_l2126_212600


namespace sweetest_sugar_water_l2126_212688

-- Define the initial sugar water concentration
def initial_concentration : ℚ := 25 / 125

-- Define Student A's final concentration (remains the same)
def concentration_A : ℚ := initial_concentration

-- Define Student B's added solution
def added_solution_B : ℚ := 20 / 50

-- Define Student C's added solution
def added_solution_C : ℚ := 2 / 5

-- Theorem statement
theorem sweetest_sugar_water :
  added_solution_C > concentration_A ∧
  added_solution_C > added_solution_B :=
sorry

end sweetest_sugar_water_l2126_212688


namespace cori_age_relation_l2126_212674

theorem cori_age_relation (cori_age aunt_age : ℕ) (years : ℕ) : 
  cori_age = 3 → aunt_age = 19 → 
  (cori_age + years : ℚ) = (1 / 3) * (aunt_age + years : ℚ) → 
  years = 5 := by sorry

end cori_age_relation_l2126_212674


namespace function_inequality_condition_l2126_212655

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (f = fun x ↦ x^2 + 5*x + 6) →
  (a > 0) →
  (b > 0) →
  (∀ x, |x + 1| < b → |f x + 3| < a) ↔ (a > 11/4 ∧ b > 3/2) :=
by sorry

end function_inequality_condition_l2126_212655


namespace special_sequence_sum_l2126_212626

/-- A sequence where the sum of two terms with a term between them increases by a constant amount -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = a n + a (n + 1) + d

theorem special_sequence_sum (a : ℕ → ℝ) (h : SpecialSequence a)
    (h1 : a 2 + a 3 = 4) (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 10 := by
  sorry

end special_sequence_sum_l2126_212626


namespace ring_binder_price_l2126_212663

/-- Proves that the original price of each ring-binder was $20 given the problem conditions -/
theorem ring_binder_price : 
  ∀ (original_backpack_price backpack_price_increase 
     ring_binder_price_decrease num_ring_binders total_spent : ℕ),
  original_backpack_price = 50 →
  backpack_price_increase = 5 →
  ring_binder_price_decrease = 2 →
  num_ring_binders = 3 →
  total_spent = 109 →
  ∃ (original_ring_binder_price : ℕ),
    original_ring_binder_price = 20 ∧
    (original_backpack_price + backpack_price_increase) + 
    num_ring_binders * (original_ring_binder_price - ring_binder_price_decrease) = total_spent :=
by
  sorry

end ring_binder_price_l2126_212663


namespace largest_non_36multiple_composite_sum_l2126_212606

def is_composite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

def is_sum_of_36multiple_and_composite (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k > 0 ∧ is_composite m ∧ n = 36 * k + m

theorem largest_non_36multiple_composite_sum :
  (∀ n > 304, is_sum_of_36multiple_and_composite n) ∧
  ¬is_sum_of_36multiple_and_composite 304 := by
  sorry

end largest_non_36multiple_composite_sum_l2126_212606


namespace cheryl_material_usage_l2126_212632

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 4/9
  let material2 : ℚ := 2/3
  let leftover : ℚ := 8/18
  material_used material1 material2 leftover = 2/3 :=
by
  sorry

#check cheryl_material_usage

end cheryl_material_usage_l2126_212632


namespace action_figure_value_l2126_212653

theorem action_figure_value (n : ℕ) (known_value : ℕ) (discount : ℕ) (total_earned : ℕ) :
  n = 5 →
  known_value = 20 →
  discount = 5 →
  total_earned = 55 →
  ∃ (other_value : ℕ),
    other_value * (n - 1) + known_value = total_earned + n * discount ∧
    other_value = 15 := by
  sorry

end action_figure_value_l2126_212653


namespace smallest_greater_perfect_square_l2126_212630

theorem smallest_greater_perfect_square (a : ℕ) (h : ∃ k : ℕ, a = k^2) :
  (∀ n : ℕ, n > a ∧ (∃ m : ℕ, n = m^2) → n ≥ a + 2*Int.sqrt a + 1) ∧
  (∃ m : ℕ, a + 2*Int.sqrt a + 1 = m^2) :=
sorry

end smallest_greater_perfect_square_l2126_212630


namespace equal_numbers_product_l2126_212603

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 22 →
  b = 18 →
  c = 32 →
  d = e →
  d * e = 196 := by
sorry

end equal_numbers_product_l2126_212603


namespace baxter_earnings_l2126_212691

structure School where
  name : String
  students : ℕ
  days : ℕ
  bonus : ℚ

def total_student_days (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.students * s.days) 0

def total_bonus (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + s.students * s.bonus) 0

theorem baxter_earnings (schools : List School) 
  (h_schools : schools = [
    ⟨"Ajax", 5, 4, 0⟩, 
    ⟨"Baxter", 3, 6, 5⟩, 
    ⟨"Colton", 6, 8, 0⟩
  ])
  (h_total_paid : 920 = (total_student_days schools) * (daily_wage : ℚ) + total_bonus schools)
  (daily_wage : ℚ) :
  ∃ (baxter_earnings : ℚ), baxter_earnings = 204.42 ∧ 
    baxter_earnings = 3 * 6 * daily_wage + 3 * 5 :=
by sorry

end baxter_earnings_l2126_212691


namespace sum_remainder_mod_13_l2126_212657

theorem sum_remainder_mod_13 : (9001 + 9002 + 9003 + 9004) % 13 = 7 := by
  sorry

end sum_remainder_mod_13_l2126_212657


namespace probability_of_five_consecutive_heads_l2126_212677

/-- Represents a sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- Returns true if the given sequence has at least 5 consecutive heads -/
def hasAtLeastFiveConsecutiveHeads (seq : CoinFlipSequence) : Bool :=
  sorry

/-- The total number of possible outcomes when flipping a coin 8 times -/
def totalOutcomes : Nat := 2^8

/-- The number of outcomes with at least 5 consecutive heads -/
def successfulOutcomes : Nat := 13

theorem probability_of_five_consecutive_heads :
  (Nat.card {seq : CoinFlipSequence | hasAtLeastFiveConsecutiveHeads seq} : ℚ) / totalOutcomes = 13 / 256 := by
  sorry

end probability_of_five_consecutive_heads_l2126_212677


namespace range_of_a_l2126_212648

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a * x^2 - x + 1/(16*a))

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Icc 1 2 :=
by sorry

end range_of_a_l2126_212648


namespace equation_solution_l2126_212685

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25) ∧ (x = -20) := by
  sorry

end equation_solution_l2126_212685


namespace minimum_cans_needed_l2126_212664

/-- The number of ounces in each can -/
def can_capacity : ℕ := 10

/-- The minimum number of ounces required -/
def min_ounces : ℕ := 120

/-- The minimum number of cans needed to provide at least the required ounces -/
def min_cans : ℕ := 12

theorem minimum_cans_needed :
  (min_cans * can_capacity ≥ min_ounces) ∧
  (∀ n : ℕ, n * can_capacity ≥ min_ounces → n ≥ min_cans) :=
by sorry

end minimum_cans_needed_l2126_212664


namespace min_value_2x_plus_y_compare_expressions_l2126_212682

-- Problem 1
theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + 2/(y'+1) = 2 → 2*x + y ≤ 2*x' + y' :=
sorry

-- Problem 2
theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a + b = 1) : 
  8 - 1/a ≤ 1/b + 1/(a*b) :=
sorry

end min_value_2x_plus_y_compare_expressions_l2126_212682


namespace mike_tv_and_games_time_l2126_212646

/-- Given Mike's TV and video game habits, prove the total time spent on both activities in a week. -/
theorem mike_tv_and_games_time (tv_hours_per_day : ℕ) (video_game_days_per_week : ℕ) : 
  tv_hours_per_day = 4 →
  video_game_days_per_week = 3 →
  (tv_hours_per_day * 7 + video_game_days_per_week * (tv_hours_per_day / 2)) = 34 := by
sorry


end mike_tv_and_games_time_l2126_212646


namespace basketball_game_theorem_l2126_212696

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 : ℝ)
  (q2 : ℝ)
  (q3 : ℝ)
  (q4 : ℝ)

/-- The game result -/
def GameResult := TeamScores → TeamScores → Prop

/-- Checks if the scores form a decreasing geometric sequence -/
def is_decreasing_geometric (s : TeamScores) : Prop :=
  ∃ r : ℝ, r > 1 ∧ s.q2 = s.q1 / r ∧ s.q3 = s.q2 / r ∧ s.q4 = s.q3 / r

/-- Checks if the scores form a decreasing arithmetic sequence -/
def is_decreasing_arithmetic (s : TeamScores) : Prop :=
  ∃ d : ℝ, d > 0 ∧ s.q2 = s.q1 - d ∧ s.q3 = s.q2 - d ∧ s.q4 = s.q3 - d

/-- Calculates the total score of a team -/
def total_score (s : TeamScores) : ℝ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the score in the second half -/
def second_half_score (s : TeamScores) : ℝ := s.q3 + s.q4

/-- The main theorem to prove -/
theorem basketball_game_theorem (falcons eagles : TeamScores) : 
  (is_decreasing_geometric falcons) →
  (is_decreasing_arithmetic eagles) →
  (falcons.q1 + falcons.q2 = eagles.q1 + eagles.q2) →
  (total_score eagles = total_score falcons + 2) →
  (total_score falcons ≤ 100 ∧ total_score eagles ≤ 100) →
  (second_half_score falcons + second_half_score eagles = 27) := by
  sorry


end basketball_game_theorem_l2126_212696


namespace peaches_in_knapsack_l2126_212656

/-- Given a total of 60 peaches distributed among two identical bags and a knapsack,
    where the knapsack contains half as many peaches as each bag,
    prove that the number of peaches in the knapsack is 12. -/
theorem peaches_in_knapsack :
  let total_peaches : ℕ := 60
  let knapsack_peaches : ℕ := x
  let bag_peaches : ℕ := 2 * x
  x + bag_peaches + bag_peaches = total_peaches →
  x = 12 :=
by
  sorry

end peaches_in_knapsack_l2126_212656


namespace alexander_buckwheat_investment_l2126_212637

theorem alexander_buckwheat_investment (initial_price : ℝ) (final_price : ℝ)
  (one_year_rate_2020 : ℝ) (two_year_rate : ℝ) (one_year_rate_2021 : ℝ)
  (h1 : initial_price = 70)
  (h2 : final_price = 100)
  (h3 : one_year_rate_2020 = 0.1)
  (h4 : two_year_rate = 0.08)
  (h5 : one_year_rate_2021 = 0.05) :
  (initial_price * (1 + one_year_rate_2020) * (1 + one_year_rate_2021) < final_price) ∧
  (initial_price * (1 + two_year_rate)^2 < final_price) :=
by sorry

end alexander_buckwheat_investment_l2126_212637


namespace quadratic_equal_roots_l2126_212698

/-- If the quadratic equation mx² + 2x + 1 = 0 has two equal real roots, then m = 1 -/
theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2*x + 1 = 0 ∧ 
   (∀ y : ℝ, m * y^2 + 2*y + 1 = 0 → y = x)) → 
  m = 1 := by sorry

end quadratic_equal_roots_l2126_212698


namespace perfect_square_5ab4_l2126_212621

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def ends_with_four (n : ℕ) : Prop := n % 10 = 4

def starts_with_five (n : ℕ) : Prop := 5000 ≤ n ∧ n < 6000

def is_5ab4_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4

theorem perfect_square_5ab4 (n : ℕ) :
  is_four_digit n →
  ends_with_four n →
  starts_with_five n →
  is_5ab4_form n →
  ∃ (m : ℕ), n = m^2 →
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4 ∧ a + b = 9 :=
sorry

end perfect_square_5ab4_l2126_212621


namespace function_satisfies_conditions_l2126_212693

-- Define the function
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 3) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
by
  sorry

end function_satisfies_conditions_l2126_212693


namespace debate_team_group_size_l2126_212684

theorem debate_team_group_size 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (num_groups : ℕ) 
  (h1 : num_boys = 31)
  (h2 : num_girls = 32)
  (h3 : num_groups = 7) :
  (num_boys + num_girls) / num_groups = 9 := by
sorry

end debate_team_group_size_l2126_212684


namespace expression_evaluation_l2126_212618

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end expression_evaluation_l2126_212618


namespace vector_difference_norm_l2126_212613

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_norm (a b : V)
  (ha : ‖a‖ = 6)
  (hb : ‖b‖ = 8)
  (hab : ‖a + b‖ = ‖a - b‖) :
  ‖a - b‖ = 10 := by
  sorry

end vector_difference_norm_l2126_212613


namespace fraction_simplification_l2126_212667

theorem fraction_simplification :
  ((5^1004)^4 - (5^1002)^4) / ((5^1003)^4 - (5^1001)^4) = 25 := by
  sorry

end fraction_simplification_l2126_212667


namespace triangle_area_proof_l2126_212676

theorem triangle_area_proof (a b c : ℝ) (h1 : a + b + c = 10 + 2 * Real.sqrt 7) 
  (h2 : a / 2 = b / 3) (h3 : a / 2 = c / Real.sqrt 7) : 
  Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) = 6 * Real.sqrt 3 := by
  sorry

end triangle_area_proof_l2126_212676


namespace correlation_coefficient_comparison_l2126_212695

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r₁ : ℝ := sorry
def r₂ : ℝ := sorry

theorem correlation_coefficient_comparison : r₁ < 0 ∧ 0 < r₂ := by sorry

end correlation_coefficient_comparison_l2126_212695


namespace cost_750_candies_l2126_212670

/-- The cost of buying a given number of chocolate candies with a possible discount -/
def total_cost (candies_per_box : ℕ) (box_cost : ℚ) (num_candies : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let num_boxes := (num_candies + candies_per_box - 1) / candies_per_box
  let cost_before_discount := num_boxes * box_cost
  let discount := if num_candies > discount_threshold then discount_rate * cost_before_discount else 0
  cost_before_discount - discount

/-- The total cost to buy 750 chocolate candies is $180 -/
theorem cost_750_candies :
  total_cost 30 8 750 (1/10) 500 = 180 := by
  sorry

end cost_750_candies_l2126_212670


namespace train_speed_proof_l2126_212636

/-- The speed of the second train in km/hr -/
def second_train_speed : ℝ := 40

/-- The additional distance traveled by the first train in km -/
def additional_distance : ℝ := 100

/-- The total distance between P and Q in km -/
def total_distance : ℝ := 900

/-- The speed of the first train in km/hr -/
def first_train_speed : ℝ := 50

theorem train_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    first_train_speed * t = second_train_speed * t + additional_distance ∧
    first_train_speed * t + second_train_speed * t = total_distance :=
by sorry

end train_speed_proof_l2126_212636


namespace like_terms_exponent_sum_l2126_212679

theorem like_terms_exponent_sum (a b : ℝ) (x y : ℤ) : 
  (∃ k : ℝ, k ≠ 0 ∧ -3 * a^(x + 2*y) * b^9 = k * (2 * a^3 * b^(2*x + y))) → 
  x + y = 4 := by sorry

end like_terms_exponent_sum_l2126_212679


namespace line_properties_l2126_212601

def line_equation (x y : ℝ) : Prop := y = -x + 5

theorem line_properties :
  let angle_with_ox : ℝ := 135
  let intersection_point : ℝ × ℝ := (0, 5)
  let point_A : ℝ × ℝ := (2, 3)
  let point_B : ℝ × ℝ := (2, -3)
  (∀ x y, line_equation x y → 
    (Real.tan (angle_with_ox * π / 180) = -1 ∧ 
     line_equation (intersection_point.1) (intersection_point.2))) ∧
  line_equation point_A.1 point_A.2 ∧
  ¬line_equation point_B.1 point_B.2 :=
by sorry

end line_properties_l2126_212601


namespace vhs_trade_in_value_proof_l2126_212675

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The cost of each DVD in dollars -/
def dvd_cost : ℚ := 10

/-- The total cost to replace all movies in dollars -/
def total_replacement_cost : ℚ := 800

/-- The trade-in value of each VHS in dollars -/
def vhs_trade_in_value : ℚ := 2

theorem vhs_trade_in_value_proof :
  vhs_trade_in_value * num_movies + total_replacement_cost = dvd_cost * num_movies :=
sorry

end vhs_trade_in_value_proof_l2126_212675


namespace quadratic_inequality_solution_l2126_212668

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
sorry

end quadratic_inequality_solution_l2126_212668


namespace perpendicular_line_equation_l2126_212610

/-- The equation of a line perpendicular to 2x - y + 4 = 0 and passing through (-2, 1) is x + 2y = 0 -/
theorem perpendicular_line_equation :
  let given_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 4 = 0
  let point : ℝ × ℝ := (-2, 1)
  let perpendicular_line : ℝ → ℝ → Prop := λ x y => x + 2 * y = 0
  (∀ x y, perpendicular_line x y ↔ 
    (∃ m b, y = m * x + b ∧ 
            m * 2 = -1 ∧ 
            point.2 = m * point.1 + b)) :=
by sorry

end perpendicular_line_equation_l2126_212610


namespace first_knife_price_is_50_l2126_212661

/-- Represents the daily sales data for a door-to-door salesman --/
structure SalesData where
  houses_visited : ℕ
  purchase_rate : ℚ
  expensive_knife_price : ℕ
  weekly_revenue : ℕ
  work_days : ℕ

/-- Calculates the price of the first set of knives based on the given sales data --/
def calculate_knife_price (data : SalesData) : ℚ :=
  let buyers := data.houses_visited * data.purchase_rate
  let expensive_knife_buyers := buyers / 2
  let weekly_expensive_knife_revenue := expensive_knife_buyers * data.expensive_knife_price * data.work_days
  let weekly_first_knife_revenue := data.weekly_revenue - weekly_expensive_knife_revenue
  let weekly_first_knife_sales := expensive_knife_buyers * data.work_days
  weekly_first_knife_revenue / weekly_first_knife_sales

/-- Theorem stating that the price of the first set of knives is $50 --/
theorem first_knife_price_is_50 (data : SalesData)
  (h1 : data.houses_visited = 50)
  (h2 : data.purchase_rate = 1/5)
  (h3 : data.expensive_knife_price = 150)
  (h4 : data.weekly_revenue = 5000)
  (h5 : data.work_days = 5) :
  calculate_knife_price data = 50 := by
  sorry

end first_knife_price_is_50_l2126_212661


namespace bicycle_stock_decrease_l2126_212647

/-- The monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := sorry

/-- The number of months between January 1 and October 1 -/
def months : ℕ := 9

/-- The total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Theorem stating that the monthly decrease in bicycle stock is 4 -/
theorem bicycle_stock_decrease : monthly_decrease = 4 := by
  sorry

end bicycle_stock_decrease_l2126_212647


namespace S_intersect_T_l2126_212633

noncomputable def S : Set ℝ := {y | ∃ x, y = 2^x}
def T : Set ℝ := {x | Real.log (x - 1) < 0}

theorem S_intersect_T : S ∩ T = {x | 1 < x ∧ x < 2} := by sorry

end S_intersect_T_l2126_212633


namespace initial_marbles_l2126_212604

theorem initial_marbles (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  lost = 5 → remaining = 4 → initial = lost + remaining :=
by
  sorry

end initial_marbles_l2126_212604


namespace a_equals_3_sufficient_not_necessary_for_a_squared_9_l2126_212650

theorem a_equals_3_sufficient_not_necessary_for_a_squared_9 :
  (∀ a : ℝ, a = 3 → a^2 = 9) ∧
  (∃ a : ℝ, a ≠ 3 ∧ a^2 = 9) :=
by sorry

end a_equals_3_sufficient_not_necessary_for_a_squared_9_l2126_212650


namespace cube_configurations_l2126_212643

/-- Represents a rotation in 3D space -/
structure Rotation :=
  (fixedConfigurations : ℕ)

/-- The group of rotations for a cube -/
def rotationGroup : Finset Rotation := sorry

/-- The number of white unit cubes -/
def numWhiteCubes : ℕ := 5

/-- The number of blue unit cubes -/
def numBlueCubes : ℕ := 3

/-- The total number of unit cubes -/
def totalCubes : ℕ := numWhiteCubes + numBlueCubes

/-- Calculates the number of fixed configurations for a given rotation -/
def fixedConfigurations (r : Rotation) : ℕ := r.fixedConfigurations

/-- Applies Burnside's Lemma to calculate the number of distinct configurations -/
def distinctConfigurations : ℕ :=
  (rotationGroup.sum fixedConfigurations) / rotationGroup.card

theorem cube_configurations :
  distinctConfigurations = 3 := by sorry

end cube_configurations_l2126_212643


namespace q_array_sum_formula_l2126_212666

/-- Definition of a 1/q-array sum -/
def qArraySum (q : ℚ) : ℚ :=
  (2 * q^2) / ((2*q - 1) * (q - 1))

/-- Theorem: The sum of all terms in a 1/q-array with the given properties is (2q^2) / ((2q-1)(q-1)) -/
theorem q_array_sum_formula (q : ℚ) (hq : q ≠ 0) (hq1 : q ≠ 1/2) (hq2 : q ≠ 1) : 
  qArraySum q = ∑' (r : ℕ) (c : ℕ), (1 / (2*q)^r) * (1 / q^c) :=
sorry

#eval (qArraySum 1220).num % 1220 + (qArraySum 1220).den % 1220

end q_array_sum_formula_l2126_212666


namespace second_smallest_divisible_by_all_less_than_8_sum_of_digits_l2126_212699

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → k ∣ n

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_8_sum_of_digits :
  ∃ M : ℕ, second_smallest is_divisible_by_all_less_than_8 M ∧ sum_of_digits M = 12 :=
sorry

end second_smallest_divisible_by_all_less_than_8_sum_of_digits_l2126_212699


namespace basketball_match_probabilities_l2126_212607

/-- Represents the probability of a team winning a single game -/
structure GameProbability where
  teamA : ℝ
  teamB : ℝ
  sum_to_one : teamA + teamB = 1

/-- Calculates the probability of team A winning by a score of 2 to 1 -/
def prob_A_wins_2_1 (p : GameProbability) : ℝ :=
  2 * p.teamA * p.teamB * p.teamA

/-- Calculates the probability of team B winning the match -/
def prob_B_wins (p : GameProbability) : ℝ :=
  p.teamB * p.teamB + 2 * p.teamA * p.teamB * p.teamB

/-- The main theorem stating the probabilities for the given scenario -/
theorem basketball_match_probabilities (p : GameProbability) 
  (hA : p.teamA = 0.6) (hB : p.teamB = 0.4) :
  prob_A_wins_2_1 p = 0.288 ∧ prob_B_wins p = 0.352 := by
  sorry


end basketball_match_probabilities_l2126_212607


namespace quadratic_function_properties_l2126_212628

/-- The quadratic function y = x^2 + ax + a - 2 -/
def f (a x : ℝ) : ℝ := x^2 + a*x + a - 2

theorem quadratic_function_properties (a : ℝ) :
  -- The function always has two distinct real roots
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  -- The distance between the roots is minimized when a = 2
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, f b x₁ = 0 ∧ f b x₂ = 0 → 
    |x₁ - x₂| ≥ |(-2 : ℝ) - 2|) ∧
  -- When both roots are in the interval (-2, 2), a is in the interval (-2/3, 2)
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ -2 < x₁ ∧ x₁ < 2 ∧ -2 < x₂ ∧ x₂ < 2 → 
    -2/3 < a ∧ a < 2) :=
sorry

end quadratic_function_properties_l2126_212628


namespace grandma_olga_grandchildren_l2126_212686

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Proves that Grandma Olga has 33 grandchildren given the specified conditions -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end grandma_olga_grandchildren_l2126_212686


namespace m_range_l2126_212642

def p (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0

def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0

theorem m_range (m : ℝ) :
  (m < 0) →
  (∀ x, p x → q x m) →
  m ≤ -3 :=
sorry

end m_range_l2126_212642


namespace quadratic_root_implies_s_value_l2126_212629

theorem quadratic_root_implies_s_value 
  (r s : ℝ) 
  (h : (4 + 3*I : ℂ) = -r/(2*2) + (r^2/(2*2)^2 - s/2).sqrt) : 
  s = 50 := by
sorry

end quadratic_root_implies_s_value_l2126_212629


namespace binary_11011000_equals_quaternary_3120_l2126_212617

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 11011000₂ -/
def binary_11011000 : List Bool := [true, true, false, true, true, false, false, false]

theorem binary_11011000_equals_quaternary_3120 :
  decimal_to_quaternary (binary_to_decimal binary_11011000) = [3, 1, 2, 0] := by
  sorry

#eval decimal_to_quaternary (binary_to_decimal binary_11011000)

end binary_11011000_equals_quaternary_3120_l2126_212617


namespace average_of_5_8_N_l2126_212641

theorem average_of_5_8_N (N : ℝ) (h : 8 < N ∧ N < 20) : 
  let avg := (5 + 8 + N) / 3
  avg = 8 ∨ avg = 10 := by
sorry

end average_of_5_8_N_l2126_212641


namespace eighth_iteration_is_zero_l2126_212660

-- Define the function g based on the graph
def g : ℕ → ℕ
| 0 => 0
| 1 => 8
| 2 => 5
| 3 => 0
| 4 => 7
| 5 => 3
| 6 => 9
| 7 => 2
| 8 => 1
| 9 => 4
| _ => 0  -- Default case for numbers not explicitly shown in the graph

-- Define the iteration of g
def iterate_g (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_g n (g x)

-- Theorem statement
theorem eighth_iteration_is_zero : iterate_g 8 5 = 0 := by
  sorry

end eighth_iteration_is_zero_l2126_212660


namespace watch_cost_price_l2126_212654

/-- Proves that the cost price of a watch is 2000, given specific selling conditions. -/
theorem watch_cost_price : 
  ∀ (cost_price : ℝ),
  (cost_price * 0.8 + 520 = cost_price * 1.06) →
  cost_price = 2000 := by
sorry

end watch_cost_price_l2126_212654
