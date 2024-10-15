import Mathlib

namespace NUMINAMATH_CALUDE_amanda_earnings_l195_19597

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Number of appointments on Monday -/
def monday_appointments : ℕ := 5

/-- Duration of each Monday appointment in hours -/
def monday_appointment_duration : ℝ := 1.5

/-- Duration of Tuesday appointment in hours -/
def tuesday_appointment_duration : ℝ := 3

/-- Number of appointments on Thursday -/
def thursday_appointments : ℕ := 2

/-- Duration of each Thursday appointment in hours -/
def thursday_appointment_duration : ℝ := 2

/-- Duration of Saturday appointment in hours -/
def saturday_appointment_duration : ℝ := 6

/-- Total earnings for the week -/
def total_earnings : ℝ :=
  hourly_rate * (monday_appointments * monday_appointment_duration +
                 tuesday_appointment_duration +
                 thursday_appointments * thursday_appointment_duration +
                 saturday_appointment_duration)

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l195_19597


namespace NUMINAMATH_CALUDE_books_read_l195_19591

theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l195_19591


namespace NUMINAMATH_CALUDE_quiz_competition_outcomes_quiz_competition_proof_l195_19583

def participants : Nat := 6

theorem quiz_competition_outcomes (rita_not_third : Bool) : Nat :=
  participants * (participants - 1) * (participants - 2)

theorem quiz_competition_proof :
  quiz_competition_outcomes true = 120 := by sorry

end NUMINAMATH_CALUDE_quiz_competition_outcomes_quiz_competition_proof_l195_19583


namespace NUMINAMATH_CALUDE_pizza_remainder_l195_19500

theorem pizza_remainder (john_portion emma_fraction : ℚ) : 
  john_portion = 4/5 →
  emma_fraction = 1/4 →
  (1 - john_portion) * (1 - emma_fraction) = 3/20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_remainder_l195_19500


namespace NUMINAMATH_CALUDE_triangle_properties_l195_19588

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following statements under the given conditions. -/
theorem triangle_properties (A B C a b c : Real) :
  -- Given conditions
  (4 * Real.sin (A / 2 - B / 2) ^ 2 + 4 * Real.sin A * Real.sin B = 2 + Real.sqrt 2) →
  (b = 4) →
  (1 / 2 * a * b * Real.sin C = 6) →
  -- Triangle inequality and angle sum
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (A + B + C = π) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  -- Statements to prove
  (C = π / 4 ∧
   c = Real.sqrt 10 ∧
   Real.tan (2 * B - C) = 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l195_19588


namespace NUMINAMATH_CALUDE_equation_solution_l195_19569

theorem equation_solution : ∃ x : ℝ, (23 - 5 = 3 + x) ∧ (x = 15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l195_19569


namespace NUMINAMATH_CALUDE_division_equals_fraction_l195_19527

theorem division_equals_fraction : 200 / (12 + 15 * 2 - 4)^2 = 50 / 361 := by
  sorry

end NUMINAMATH_CALUDE_division_equals_fraction_l195_19527


namespace NUMINAMATH_CALUDE_convention_delegates_l195_19559

theorem convention_delegates (total : ℕ) 
  (h1 : 16 ≤ total) 
  (h2 : (total - 16) % 2 = 0) 
  (h3 : 10 ≤ total - 16 - (total - 16) / 2) : 
  total = 36 := by
  sorry

end NUMINAMATH_CALUDE_convention_delegates_l195_19559


namespace NUMINAMATH_CALUDE_half_sum_abs_diff_squares_l195_19558

theorem half_sum_abs_diff_squares : 
  (1/2 : ℝ) * (|20^2 - 15^2| + |15^2 - 20^2|) = 175 := by
  sorry

end NUMINAMATH_CALUDE_half_sum_abs_diff_squares_l195_19558


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l195_19516

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l195_19516


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_positive_l195_19573

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- State the theorem
theorem intersection_nonempty_implies_a_positive (a : ℝ) :
  (A ∩ B a).Nonempty → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_positive_l195_19573


namespace NUMINAMATH_CALUDE_smaller_exterior_angle_implies_obtuse_l195_19566

-- Define a triangle
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define the property of having an exterior angle smaller than its adjacent interior angle
def has_smaller_exterior_angle (t : Triangle) : Prop :=
  ∃ (exterior_angle interior_angle : ℝ), exterior_angle < interior_angle

-- Define an obtuse triangle
def is_obtuse (t : Triangle) : Prop :=
  ∃ (angle : ℝ), angle > Real.pi / 2

-- State the theorem
theorem smaller_exterior_angle_implies_obtuse (t : Triangle) :
  has_smaller_exterior_angle t → is_obtuse t :=
sorry

end NUMINAMATH_CALUDE_smaller_exterior_angle_implies_obtuse_l195_19566


namespace NUMINAMATH_CALUDE_rearranged_cube_surface_area_l195_19554

def slice_heights : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

def last_slice_height (heights : List ℚ) : ℚ :=
  1 - (heights.sum)

def surface_area (heights : List ℚ) : ℚ :=
  2 + 2 + 2  -- top/bottom + sides + front/back

theorem rearranged_cube_surface_area :
  surface_area slice_heights = 6 := by
  sorry

end NUMINAMATH_CALUDE_rearranged_cube_surface_area_l195_19554


namespace NUMINAMATH_CALUDE_velocity_from_similarity_l195_19529

/-- Given a, T, R, L, and x as real numbers, where x represents a distance,
    and assuming the equation (a * T) / (a * T - R) = (L + x) / x holds,
    prove that the velocity of the point described by x is a * (L / R). -/
theorem velocity_from_similarity (a T R L x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R := by
  sorry

end NUMINAMATH_CALUDE_velocity_from_similarity_l195_19529


namespace NUMINAMATH_CALUDE_expression_evaluation_l195_19582

theorem expression_evaluation : (35 * 100) / (0.07 * 100) = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l195_19582


namespace NUMINAMATH_CALUDE_largeSum_congruence_l195_19502

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that a number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_congruence (n : ℕ) : n ≡ sumOfDigits n [MOD 9]

/-- The sum we want to evaluate -/
def largeSum : ℕ := 2 + 55 + 444 + 3333 + 66666 + 777777 + 8888888 + 99999999

/-- Theorem stating that the large sum is congruent to 2 modulo 9 -/
theorem largeSum_congruence : largeSum ≡ 2 [MOD 9] := by sorry

end NUMINAMATH_CALUDE_largeSum_congruence_l195_19502


namespace NUMINAMATH_CALUDE_denarii_problem_l195_19514

theorem denarii_problem (x y : ℚ) : 
  x + 7 = 5 * (y - 7) ∧ 
  y + 5 = 7 * (x - 5) → 
  x = 121 / 17 ∧ y = 167 / 17 := by
sorry

end NUMINAMATH_CALUDE_denarii_problem_l195_19514


namespace NUMINAMATH_CALUDE_even_function_range_l195_19549

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (0, +∞) if f(x) ≤ f(y) for all 0 < x < y -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) (a : ℝ) 
  (h_even : IsEven f)
  (h_incr : IncreasingOnPositive f)
  (h_cond : f a ≥ f 2) :
  a ∈ Set.Iic (-2) ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l195_19549


namespace NUMINAMATH_CALUDE_pages_written_first_week_pages_written_first_week_proof_l195_19572

/-- Calculates the number of pages written in the first week of a 500-page book -/
theorem pages_written_first_week : ℕ :=
  let total_pages : ℕ := 500
  let second_week_write_ratio : ℚ := 30 / 100
  let coffee_damage_ratio : ℚ := 20 / 100
  let remaining_empty_pages : ℕ := 196
  
  -- Define a function to calculate pages written in first week
  let pages_written (x : ℕ) : Prop :=
    let remaining_after_first := total_pages - x
    let remaining_after_second := remaining_after_first - (second_week_write_ratio * remaining_after_first).floor
    let damaged_pages := (coffee_damage_ratio * remaining_after_second).floor
    remaining_after_second - damaged_pages = remaining_empty_pages

  -- The theorem states that 150 satisfies the conditions
  150

/-- Proof of the theorem -/
theorem pages_written_first_week_proof : pages_written_first_week = 150 := by
  sorry

end NUMINAMATH_CALUDE_pages_written_first_week_pages_written_first_week_proof_l195_19572


namespace NUMINAMATH_CALUDE_november_to_december_ratio_l195_19535

/-- Represents the revenue of a toy store in a given month -/
structure Revenue where
  amount : ℝ
  amount_pos : amount > 0

/-- The toy store's revenues for November, December, and January -/
structure StoreRevenue where
  november : Revenue
  december : Revenue
  january : Revenue
  january_is_third_of_november : january.amount = (1/3) * november.amount
  december_is_average_multiple : december.amount = 2.5 * ((november.amount + january.amount) / 2)

/-- The ratio of November's revenue to December's revenue is 3:5 -/
theorem november_to_december_ratio (s : StoreRevenue) :
  s.november.amount / s.december.amount = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_november_to_december_ratio_l195_19535


namespace NUMINAMATH_CALUDE_smallest_parabola_coefficient_l195_19522

theorem smallest_parabola_coefficient (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 3/4)^2 - 25/16) →  -- vertex condition
  (∀ (x y : ℝ), y = a * x^2 + b * x + c) →      -- parabola equation
  a > 0 →                                       -- a is positive
  ∃ (n : ℚ), a + b + c = n →                    -- sum is rational
  ∀ (a' : ℝ), (∃ (b' c' : ℝ) (n' : ℚ), 
    (∀ (x y : ℝ), y = a' * x^2 + b' * x + c') ∧ 
    a' > 0 ∧ 
    a' + b' + c' = n') → 
  a ≤ a' →
  a = 41 := by
sorry

end NUMINAMATH_CALUDE_smallest_parabola_coefficient_l195_19522


namespace NUMINAMATH_CALUDE_problem_solution_l195_19555

theorem problem_solution (x y : ℝ) 
  (h1 : 5 + x = 3 - y) 
  (h2 : 2 + y = 6 + x) : 
  5 - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l195_19555


namespace NUMINAMATH_CALUDE_inequality_proof_l195_19523

theorem inequality_proof (a b c d x y : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (c*d)^(y/2)) :
  x < y := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l195_19523


namespace NUMINAMATH_CALUDE_complex_power_abs_one_l195_19577

theorem complex_power_abs_one : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_abs_one_l195_19577


namespace NUMINAMATH_CALUDE_rectangle_length_l195_19580

/-- Given a rectangle and a square, prove that the length of the rectangle is 15 cm. -/
theorem rectangle_length (w l : ℝ) (square_side : ℝ) : 
  w = 9 → 
  square_side = 12 → 
  4 * square_side = 2 * w + 2 * l → 
  l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l195_19580


namespace NUMINAMATH_CALUDE_pentagon_area_l195_19541

/-- The area of a pentagon with specific properties -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ),
  s₁ = 16 ∧ s₂ = 25 ∧ s₃ = 30 ∧ s₄ = 26 ∧ s₅ = 25 →
  ∃ (triangle_area trapezoid_area : ℝ),
    triangle_area = (1/2) * s₁ * s₂ ∧
    trapezoid_area = (1/2) * (s₄ + s₅) * s₃ ∧
    triangle_area + trapezoid_area = 965 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_area_l195_19541


namespace NUMINAMATH_CALUDE_vector_properties_l195_19562

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -1)
def c (m n : ℝ) : ℝ × ℝ := (m - 2, n)

variable (m n : ℝ)
variable (hm : m > 0)
variable (hn : n > 0)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 1) ∧
  ((a.1 - b.1) * (c m n).1 + (a.2 - b.2) * (c m n).2 = 0 → m + 2*n = 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l195_19562


namespace NUMINAMATH_CALUDE_manny_purchase_theorem_l195_19593

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount in dollars -/
def initial_amount : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount left after Manny's purchase in dollars -/
def amount_left : ℚ := initial_amount - purchase_cost

theorem manny_purchase_theorem : amount_left = 45 := by
  sorry

end NUMINAMATH_CALUDE_manny_purchase_theorem_l195_19593


namespace NUMINAMATH_CALUDE_tax_calculation_l195_19585

/-- Given gross pay and net pay, calculates the tax amount -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l195_19585


namespace NUMINAMATH_CALUDE_contractor_problem_l195_19557

/-- Proves that the original number of men employed is 12 --/
theorem contractor_problem (initial_days : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
  (h1 : initial_days = 5)
  (h2 : absent_men = 8)
  (h3 : actual_days = 15) :
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * actual_days ∧ 
    original_men = 12 := by
  sorry

end NUMINAMATH_CALUDE_contractor_problem_l195_19557


namespace NUMINAMATH_CALUDE_fraction_ordering_l195_19587

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l195_19587


namespace NUMINAMATH_CALUDE_yellow_balls_count_l195_19507

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def red_balls : ℕ := 15
def purple_balls : ℕ := 6
def prob_not_red_or_purple : ℚ := 65/100

theorem yellow_balls_count :
  ∃ (y : ℕ), y = total_balls - (white_balls + green_balls + red_balls + purple_balls) ∧
  (white_balls + green_balls + y : ℚ) / total_balls = prob_not_red_or_purple :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l195_19507


namespace NUMINAMATH_CALUDE_books_initially_l195_19545

/-- Given that Paul bought some books and ended up with a certain total, 
    this theorem proves how many books he had initially. -/
theorem books_initially (bought : ℕ) (total_after : ℕ) (h : bought = 101) (h' : total_after = 151) :
  total_after - bought = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_initially_l195_19545


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_four_l195_19508

theorem sufficient_condition_implies_m_geq_four (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x < 4 → x < m) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_four_l195_19508


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l195_19563

theorem absolute_value_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - i) / (i + 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l195_19563


namespace NUMINAMATH_CALUDE_prime_quadratic_solution_l195_19526

theorem prime_quadratic_solution : 
  {n : ℕ+ | Nat.Prime (n^4 - 27*n^2 + 121)} = {2, 5} := by sorry

end NUMINAMATH_CALUDE_prime_quadratic_solution_l195_19526


namespace NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l195_19506

noncomputable def nested_fourth_root (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

noncomputable def nested_fourth_root_product (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

def cubic_equation (y : ℝ) : Prop := y^3 - y^2 - 1 = 0

theorem nested_fourth_root_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ nested_fourth_root x = nested_fourth_root_product x ∧
  ∃ (y : ℝ), cubic_equation y ∧ x = y^3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l195_19506


namespace NUMINAMATH_CALUDE_jack_remaining_money_l195_19552

def calculate_remaining_money (initial_amount : ℝ) 
  (sparkling_water_bottles : ℕ) (sparkling_water_cost : ℝ)
  (still_water_multiplier : ℕ) (still_water_cost : ℝ)
  (cheddar_cheese_pounds : ℝ) (cheddar_cheese_cost : ℝ)
  (swiss_cheese_pounds : ℝ) (swiss_cheese_cost : ℝ) : ℝ :=
  let sparkling_water_total := sparkling_water_bottles * sparkling_water_cost
  let still_water_total := (sparkling_water_bottles * still_water_multiplier) * still_water_cost
  let cheddar_cheese_total := cheddar_cheese_pounds * cheddar_cheese_cost
  let swiss_cheese_total := swiss_cheese_pounds * swiss_cheese_cost
  let total_cost := sparkling_water_total + still_water_total + cheddar_cheese_total + swiss_cheese_total
  initial_amount - total_cost

theorem jack_remaining_money :
  calculate_remaining_money 150 4 3 3 2.5 2 8.5 1.5 11 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_money_l195_19552


namespace NUMINAMATH_CALUDE_edward_candy_cost_l195_19528

def edward_problem (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) : Prop :=
  let total_tickets := whack_a_mole_tickets + skee_ball_tickets
  let cost_per_candy := total_tickets / candies
  whack_a_mole_tickets = 3 ∧ 
  skee_ball_tickets = 5 ∧ 
  candies = 2 → 
  cost_per_candy = 4

theorem edward_candy_cost : edward_problem 3 5 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_candy_cost_l195_19528


namespace NUMINAMATH_CALUDE_no_seven_flip_l195_19512

/-- A function that returns the reverse of the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Definition of a k-flip number -/
def isKFlip (k : ℕ) (n : ℕ) : Prop :=
  k * n = reverseDigits n

/-- Theorem: There is no 7-flip integer -/
theorem no_seven_flip : ¬∃ (n : ℕ), n > 0 ∧ isKFlip 7 n := by sorry

end NUMINAMATH_CALUDE_no_seven_flip_l195_19512


namespace NUMINAMATH_CALUDE_square_product_exists_l195_19518

theorem square_product_exists (A : Finset ℕ+) (h1 : A.card = 2016) 
  (h2 : ∀ x ∈ A, ∀ p : ℕ, Nat.Prime p → p ∣ x.val → p < 30) : 
  ∃ a b c d : ℕ+, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ∃ m : ℕ, (a.val * b.val * c.val * d.val : ℕ) = m ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_square_product_exists_l195_19518


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l195_19574

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l195_19574


namespace NUMINAMATH_CALUDE_tan_3x_domain_l195_19546

theorem tan_3x_domain (x : ℝ) : 
  ∃ y, y = Real.tan (3 * x) ↔ ∀ k : ℤ, x ≠ π / 6 + k * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_3x_domain_l195_19546


namespace NUMINAMATH_CALUDE_triple_hash_100_l195_19571

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_100 : hash (hash (hash 100)) = 11.08 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_100_l195_19571


namespace NUMINAMATH_CALUDE_max_consecutive_set_size_l195_19521

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Property: sum of digits is not a multiple of 11 -/
def validNumber (n : ℕ) : Prop :=
  sumOfDigits n % 11 ≠ 0

/-- A set of consecutive positive integers with the given property -/
structure ConsecutiveSet :=
  (start : ℕ)
  (size : ℕ)
  (property : ∀ k, k ∈ Finset.range size → validNumber (start + k))

/-- The theorem to be proved -/
theorem max_consecutive_set_size :
  (∃ S : ConsecutiveSet, S.size = 38) ∧
  (∀ S : ConsecutiveSet, S.size ≤ 38) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_set_size_l195_19521


namespace NUMINAMATH_CALUDE_min_sum_is_negative_442_l195_19586

/-- An arithmetic progression with sum S_n for the first n terms. -/
structure ArithmeticProgression where
  S : ℕ → ℤ
  sum_3 : S 3 = -141
  sum_35 : S 35 = 35

/-- The minimum value of S_n for an arithmetic progression satisfying the given conditions. -/
def min_sum (ap : ArithmeticProgression) : ℤ :=
  sorry

/-- Theorem stating that the minimum value of S_n is -442. -/
theorem min_sum_is_negative_442 (ap : ArithmeticProgression) :
  min_sum ap = -442 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_is_negative_442_l195_19586


namespace NUMINAMATH_CALUDE_octagons_700_sticks_4901_l195_19590

/-- The number of sticks required to construct a series of octagons -/
def sticks_for_octagons (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 8 + 7 * (n - 1)

/-- Theorem stating that 700 octagons require 4901 sticks -/
theorem octagons_700_sticks_4901 : sticks_for_octagons 700 = 4901 := by
  sorry

end NUMINAMATH_CALUDE_octagons_700_sticks_4901_l195_19590


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l195_19570

-- Define the circle with center (8, -3) passing through (5, 1)
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25

-- Define the circle x^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the two tangent lines
def tangentLine1 (x y : ℝ) : Prop := y = (4/3) * x - 25/3
def tangentLine2 (x y : ℝ) : Prop := y = (-3/4) * x - 25/4

theorem circle_and_tangent_lines :
  (∀ x y, circle1 x y ↔ ((x = 8 ∧ y = -3) ∨ (x = 5 ∧ y = 1))) ∧
  (∀ x y, tangentLine1 x y → circle2 x y → x = 1 ∧ y = -7) ∧
  (∀ x y, tangentLine2 x y → circle2 x y → x = 1 ∧ y = -7) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l195_19570


namespace NUMINAMATH_CALUDE_warrior_truth_count_l195_19530

/-- Represents the types of weapons a warrior can have as their favorite. -/
inductive Weapon
| sword
| spear
| axe
| bow

/-- Represents a warrior's truthfulness. -/
inductive Truthfulness
| truthful
| liar

/-- Represents the problem setup. -/
structure WarriorProblem where
  totalWarriors : Nat
  swordYes : Nat
  spearYes : Nat
  axeYes : Nat
  bowYes : Nat

/-- The main theorem to prove. -/
theorem warrior_truth_count (problem : WarriorProblem)
  (h_total : problem.totalWarriors = 33)
  (h_sword : problem.swordYes = 13)
  (h_spear : problem.spearYes = 15)
  (h_axe : problem.axeYes = 20)
  (h_bow : problem.bowYes = 27)
  : { truthfulCount : Nat // 
      truthfulCount = 12 ∧
      truthfulCount + (problem.totalWarriors - truthfulCount) * 3 = 
        problem.swordYes + problem.spearYes + problem.axeYes + problem.bowYes } :=
  sorry


end NUMINAMATH_CALUDE_warrior_truth_count_l195_19530


namespace NUMINAMATH_CALUDE_mark_increase_ratio_l195_19596

/-- Proves the ratio of increase in average marks to original average marks
    when one pupil's mark is increased by 40 in a class of 80 pupils. -/
theorem mark_increase_ratio (T : ℝ) (A : ℝ) (h1 : A = T / 80) :
  let new_average := (T + 40) / 80
  let increase := new_average - A
  increase / A = 1 / (2 * A) :=
by sorry

end NUMINAMATH_CALUDE_mark_increase_ratio_l195_19596


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l195_19544

/-- Calculates the regular hourly rate for a bus driver given their total hours worked,
    total compensation, and overtime policy. -/
def calculate_regular_rate (total_hours : ℕ) (total_compensation : ℚ) : ℚ :=
  let regular_hours := min total_hours 40
  let overtime_hours := total_hours - regular_hours
  let rate := total_compensation / (regular_hours + 1.75 * overtime_hours)
  rate

/-- Theorem stating that given the specific conditions of the bus driver's work week,
    their regular hourly rate is $16. -/
theorem bus_driver_regular_rate :
  calculate_regular_rate 54 1032 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l195_19544


namespace NUMINAMATH_CALUDE_eggplant_pounds_l195_19504

/-- Represents the ingredients and costs for Scott's ratatouille recipe --/
structure Ratatouille where
  zucchini_pounds : ℝ
  zucchini_price : ℝ
  tomato_pounds : ℝ
  tomato_price : ℝ
  onion_pounds : ℝ
  onion_price : ℝ
  basil_pounds : ℝ
  basil_price : ℝ
  quart_yield : ℝ
  quart_price : ℝ
  eggplant_price : ℝ

/-- Calculates the total cost of ingredients excluding eggplants --/
def other_ingredients_cost (r : Ratatouille) : ℝ :=
  r.zucchini_pounds * r.zucchini_price +
  r.tomato_pounds * r.tomato_price +
  r.onion_pounds * r.onion_price +
  r.basil_pounds * r.basil_price

/-- Calculates the total cost of the recipe --/
def total_recipe_cost (r : Ratatouille) : ℝ :=
  r.quart_yield * r.quart_price

/-- Calculates the cost spent on eggplants --/
def eggplant_cost (r : Ratatouille) : ℝ :=
  total_recipe_cost r - other_ingredients_cost r

/-- Theorem stating the amount of eggplants bought --/
theorem eggplant_pounds (r : Ratatouille) 
  (h1 : r.zucchini_pounds = 4)
  (h2 : r.zucchini_price = 2)
  (h3 : r.tomato_pounds = 4)
  (h4 : r.tomato_price = 3.5)
  (h5 : r.onion_pounds = 3)
  (h6 : r.onion_price = 1)
  (h7 : r.basil_pounds = 1)
  (h8 : r.basil_price = 5)
  (h9 : r.quart_yield = 4)
  (h10 : r.quart_price = 10)
  (h11 : r.eggplant_price = 2) :
  eggplant_cost r / r.eggplant_price = 5 := by
  sorry


end NUMINAMATH_CALUDE_eggplant_pounds_l195_19504


namespace NUMINAMATH_CALUDE_cos_sin_thirty_squared_difference_l195_19538

theorem cos_sin_thirty_squared_difference :
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  cos_thirty ^ 2 - sin_thirty ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_thirty_squared_difference_l195_19538


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l195_19531

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
    (a = 2 ∧ b = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l195_19531


namespace NUMINAMATH_CALUDE_circle_center_l195_19509

/-- The equation of a circle in the form (x + h)² + (y + k)² = r², where (h, k) is the center. -/
def CircleEquation (h k r : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (x + h)^2 + (y + k)^2 = r^2

/-- The center of the circle (x + 2)² + y² = 5 is (-2, 0). -/
theorem circle_center :
  ∃ (h k : ℝ), CircleEquation h k (Real.sqrt 5) = CircleEquation 2 0 (Real.sqrt 5) ∧ h = -2 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l195_19509


namespace NUMINAMATH_CALUDE_pen_price_is_14_l195_19537

/-- The price of a pen in yuan -/
def pen_price : ℝ := 14

/-- The price of a ballpoint pen in yuan -/
def ballpoint_price : ℝ := 7

/-- The total cost of the pens in yuan -/
def total_cost : ℝ := 49

theorem pen_price_is_14 :
  (2 * pen_price + 3 * ballpoint_price = total_cost) ∧
  (3 * pen_price + ballpoint_price = total_cost) →
  pen_price = 14 := by
sorry

end NUMINAMATH_CALUDE_pen_price_is_14_l195_19537


namespace NUMINAMATH_CALUDE_f_bound_l195_19576

def f (x : ℝ) : ℝ := x^2 - x + 13

theorem f_bound (a x : ℝ) (h : |x - a| < 1) : |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_bound_l195_19576


namespace NUMINAMATH_CALUDE_third_bed_theorem_l195_19581

/-- Represents the number of carrots harvested from each bed and the total harvest weight -/
structure CarrotHarvest where
  first_bed : ℕ
  second_bed : ℕ
  total_weight : ℕ
  carrots_per_pound : ℕ

/-- Calculates the number of carrots in the third bed given the harvest information -/
def third_bed_carrots (harvest : CarrotHarvest) : ℕ :=
  harvest.total_weight * harvest.carrots_per_pound - (harvest.first_bed + harvest.second_bed)

/-- Theorem stating that given the specific harvest conditions, the third bed contains 78 carrots -/
theorem third_bed_theorem (harvest : CarrotHarvest)
  (h1 : harvest.first_bed = 55)
  (h2 : harvest.second_bed = 101)
  (h3 : harvest.total_weight = 39)
  (h4 : harvest.carrots_per_pound = 6) :
  third_bed_carrots harvest = 78 := by
  sorry

#eval third_bed_carrots { first_bed := 55, second_bed := 101, total_weight := 39, carrots_per_pound := 6 }

end NUMINAMATH_CALUDE_third_bed_theorem_l195_19581


namespace NUMINAMATH_CALUDE_everton_college_calculator_cost_l195_19519

/-- The total cost of calculators purchased by Everton college -/
def total_cost (scientific_count : ℕ) (graphing_count : ℕ) (scientific_price : ℕ) (graphing_price : ℕ) : ℕ :=
  scientific_count * scientific_price + graphing_count * graphing_price

/-- Theorem stating the total cost of calculators for Everton college -/
theorem everton_college_calculator_cost :
  total_cost 20 25 10 57 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_everton_college_calculator_cost_l195_19519


namespace NUMINAMATH_CALUDE_cos_270_degrees_l195_19551

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l195_19551


namespace NUMINAMATH_CALUDE_trig_values_of_α_l195_19532

-- Define the angle α and its properties
def α : Real := sorry

-- Define that the terminal side of α passes through (3, 4)
axiom terminal_point : ∃ (r : Real), r * Real.cos α = 3 ∧ r * Real.sin α = 4

-- Theorem to prove
theorem trig_values_of_α :
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_values_of_α_l195_19532


namespace NUMINAMATH_CALUDE_other_divisor_problem_l195_19575

theorem other_divisor_problem (n : ℕ) (h1 : n = 174) : 
  ∃ (x : ℕ), x ≠ 5 ∧ x < 170 ∧ 
  (∀ y : ℕ, y < 170 → y ≠ 5 → n % y = 4 → y ≤ x) ∧
  n % x = 4 ∧ n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l195_19575


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l195_19510

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  P / 5 = P * 8 * (2.5 / 100) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l195_19510


namespace NUMINAMATH_CALUDE_list_property_l195_19560

theorem list_property (S : ℝ) (n : ℝ) (list_size : ℕ) (h1 : list_size = 21) 
  (h2 : n = 4 * ((S - n) / (list_size - 1))) 
  (h3 : n = S / 6) : 
  list_size - 1 = 20 := by
sorry

end NUMINAMATH_CALUDE_list_property_l195_19560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equality_l195_19513

theorem arithmetic_sequence_equality (N : ℕ) : 
  (3 + 4 + 5 + 6 + 7) / 5 = (1993 + 1994 + 1995 + 1996 + 1997) / N → N = 1995 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equality_l195_19513


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l195_19592

theorem cubic_roots_relation (m n p q : ℝ) : 
  (∃ α β : ℝ, α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
  (∃ γ δ : ℝ, γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
  (∀ α β γ δ : ℝ, 
    (α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
    (γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
    (γ = α^3 ∧ δ = β^3 ∨ γ = β^3 ∧ δ = α^3)) →
  p = m^3 - 3*m*n :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l195_19592


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_problem_solution_l195_19565

/-- Given two sets of parallel lines intersecting each other, 
    calculate the number of lines in the second set based on 
    the number of parallelograms formed -/
theorem parallel_lines_intersection (first_set : ℕ) (parallelograms : ℕ) 
  (h1 : first_set = 5) 
  (h2 : parallelograms = 280) : 
  ∃ (second_set : ℕ), second_set * (first_set - 1) = parallelograms := by
  sorry

/-- The specific case for the given problem -/
theorem problem_solution : 
  ∃ (second_set : ℕ), second_set * 4 = 280 ∧ second_set = 71 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_problem_solution_l195_19565


namespace NUMINAMATH_CALUDE_tempo_original_value_l195_19520

/-- Given a tempo insured to 5/7 of its original value, with a 3% premium rate
resulting in a $300 premium, prove that the original value of the tempo is $14,000. -/
theorem tempo_original_value (insurance_ratio : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) :
  insurance_ratio = 5 / 7 →
  premium_rate = 3 / 100 →
  premium_amount = 300 →
  premium_rate * (insurance_ratio * 14000) = premium_amount :=
by sorry

end NUMINAMATH_CALUDE_tempo_original_value_l195_19520


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l195_19595

theorem min_cut_length_for_non_triangle (a b c : ℕ) (ha : a = 9) (hb : b = 16) (hc : c = 18) :
  ∃ x : ℕ, x = 8 ∧
  (∀ y : ℕ, y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
by sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l195_19595


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l195_19533

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) :
  radius = 14 →
  pencil_length = 0.5 →
  (2 * radius) / pencil_length = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l195_19533


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l195_19524

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - NumberOfSuits)) / 
  (StandardDeck * (StandardDeck - 1))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l195_19524


namespace NUMINAMATH_CALUDE_no_fascinating_function_l195_19594

theorem no_fascinating_function : ¬ ∃ (F : ℤ → ℤ), 
  (∀ c : ℤ, ∃ x : ℤ, F x ≠ c) ∧ 
  (∀ x : ℤ, F x = F (412 - x)) ∧
  (∀ x : ℤ, F x = F (414 - x)) ∧
  (∀ x : ℤ, F x = F (451 - x)) :=
by sorry

end NUMINAMATH_CALUDE_no_fascinating_function_l195_19594


namespace NUMINAMATH_CALUDE_power_comparison_l195_19556

theorem power_comparison (h1 : 2 > 1) (h2 : -1.1 > -1.2) : 2^(-1.1) > 2^(-1.2) := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l195_19556


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l195_19542

theorem square_ratio_theorem : 
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (75 : ℚ) / 27 = (a * (b.sqrt : ℚ) / c) ^ 2 ∧
    a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l195_19542


namespace NUMINAMATH_CALUDE_max_free_squares_l195_19539

/-- Represents a chessboard with bugs -/
structure BugChessboard (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a valid move of bugs on the chessboard -/
def ValidMove (board : BugChessboard n) : Prop :=
  ∀ i j : ℕ, i < n ∧ j < n →
    ∃ (i₁ j₁ i₂ j₂ : ℕ), 
      (i₁ < n ∧ j₁ < n ∧ i₂ < n ∧ j₂ < n) ∧
      ((i₁ = i ∧ (j₁ = j + 1 ∨ j₁ = j - 1)) ∨ (j₁ = j ∧ (i₁ = i + 1 ∨ i₁ = i - 1))) ∧
      ((i₂ = i ∧ (j₂ = j + 1 ∨ j₂ = j - 1)) ∨ (j₂ = j ∧ (i₂ = i + 1 ∨ i₂ = i - 1))) ∧
      (i₁ ≠ i₂ ∨ j₁ ≠ j₂)

/-- The number of free squares after a valid move -/
def FreeSquares (board : BugChessboard n) (move : ValidMove board) : ℕ := sorry

/-- The main theorem: the maximal number of free squares after one move is n^2 -/
theorem max_free_squares (n : ℕ) (board : BugChessboard n) :
  ∃ (move : ValidMove board), FreeSquares board move = n^2 :=
sorry

end NUMINAMATH_CALUDE_max_free_squares_l195_19539


namespace NUMINAMATH_CALUDE_count_perimeters_eq_42_l195_19578

/-- Represents a quadrilateral EFGH with specific properties -/
structure Quadrilateral where
  ef : ℕ+
  fg : ℕ+
  gh : ℕ+
  eh : ℕ+
  perimeter_lt_1200 : ef.val + fg.val + gh.val + eh.val < 1200
  right_angle_f : True
  right_angle_g : True
  ef_eq_3 : ef = 3
  gh_eq_eh : gh = eh

/-- The number of different possible perimeter values -/
def count_perimeters : ℕ := sorry

/-- Main theorem stating the number of different possible perimeter values -/
theorem count_perimeters_eq_42 : count_perimeters = 42 := by sorry

end NUMINAMATH_CALUDE_count_perimeters_eq_42_l195_19578


namespace NUMINAMATH_CALUDE_no_real_solutions_l195_19503

theorem no_real_solutions : ∀ x : ℝ, (2*x - 4*x + 7)^2 + 1 ≠ -|x^2 - 1| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l195_19503


namespace NUMINAMATH_CALUDE_draw_with_min_black_balls_l195_19547

def white_balls : ℕ := 6
def black_balls : ℕ := 4
def total_draw : ℕ := 4
def min_black : ℕ := 2

theorem draw_with_min_black_balls (white_balls black_balls total_draw min_black : ℕ) :
  (white_balls = 6) → (black_balls = 4) → (total_draw = 4) → (min_black = 2) →
  (Finset.sum (Finset.range (black_balls - min_black + 1))
    (λ i => Nat.choose black_balls (min_black + i) * Nat.choose white_balls (total_draw - (min_black + i)))) = 115 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_min_black_balls_l195_19547


namespace NUMINAMATH_CALUDE_morgan_pens_l195_19584

theorem morgan_pens (total red blue : ℕ) (h1 : total = 168) (h2 : red = 65) (h3 : blue = 45) :
  total - red - blue = 58 := by
  sorry

end NUMINAMATH_CALUDE_morgan_pens_l195_19584


namespace NUMINAMATH_CALUDE_johns_annual_epipen_cost_l195_19553

/-- Calculates the annual cost of EpiPens for John given the replacement frequency,
    cost per EpiPen, and insurance coverage percentage. -/
def annual_epipen_cost (replacement_months : ℕ) (cost_per_epipen : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let epipens_per_year : ℕ := 12 / replacement_months
  let insurance_coverage : ℕ := cost_per_epipen * insurance_coverage_percent / 100
  let cost_after_insurance : ℕ := cost_per_epipen - insurance_coverage
  epipens_per_year * cost_after_insurance

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 6 500 75 = 250 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_epipen_cost_l195_19553


namespace NUMINAMATH_CALUDE_boys_camp_science_percentage_l195_19505

theorem boys_camp_science_percentage (total_boys : ℕ) (school_A_boys : ℕ) (non_science_boys : ℕ) :
  total_boys = 550 →
  school_A_boys = (20 : ℕ) * total_boys / 100 →
  non_science_boys = 77 →
  (((school_A_boys - non_science_boys) : ℚ) / school_A_boys) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_science_percentage_l195_19505


namespace NUMINAMATH_CALUDE_divide_twelve_by_repeating_third_l195_19589

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of dividing 12 by the repeating decimal 0.3333... --/
theorem divide_twelve_by_repeating_third : 12 / repeating_third = 36 := by sorry

end NUMINAMATH_CALUDE_divide_twelve_by_repeating_third_l195_19589


namespace NUMINAMATH_CALUDE_x_in_interval_l195_19564

theorem x_in_interval (x : ℝ) (hx : x ≠ 0) : x = 2 * (1 / x) * (-x) → -4 < x ∧ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_in_interval_l195_19564


namespace NUMINAMATH_CALUDE_seventh_day_cans_l195_19511

/-- A sequence where the first term is 4 and each subsequent term increases by 5 -/
def canSequence : ℕ → ℕ
  | 0 => 4
  | n + 1 => canSequence n + 5

/-- The 7th term of the sequence is 34 -/
theorem seventh_day_cans : canSequence 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seventh_day_cans_l195_19511


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l195_19515

/-- Represents a rectangle with integer side lengths where one side is 5 feet longer than the other. -/
structure Rectangle where
  short_side : ℕ
  long_side : ℕ
  constraint : long_side = short_side + 5

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.short_side * r.long_side

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.short_side + r.long_side)

/-- Theorem: The rectangle with minimum perimeter satisfying the given conditions has dimensions 23 and 28 feet. -/
theorem min_perimeter_rectangle :
  ∀ r : Rectangle,
    area r ≥ 600 →
    perimeter r ≥ 102 ∧
    (perimeter r = 102 → r.short_side = 23 ∧ r.long_side = 28) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l195_19515


namespace NUMINAMATH_CALUDE_average_of_rst_l195_19543

theorem average_of_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_rst_l195_19543


namespace NUMINAMATH_CALUDE_variance_of_scores_l195_19599

def scores : List ℝ := [87, 90, 90, 91, 91, 94, 94]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := scores.sum / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 36/7 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_scores_l195_19599


namespace NUMINAMATH_CALUDE_pear_vendor_theorem_l195_19598

/-- Represents the actions of a pear vendor over two days --/
def pear_vendor_problem (initial_pears : ℝ) : Prop :=
  let day1_sold := 0.8 * initial_pears
  let day1_remaining := initial_pears - day1_sold
  let day1_thrown := 0.5 * day1_remaining
  let day2_start := day1_remaining - day1_thrown
  let day2_sold := 0.8 * day2_start
  let day2_thrown := day2_start - day2_sold
  let total_thrown := day1_thrown + day2_thrown
  (total_thrown / initial_pears) * 100 = 12

/-- Theorem stating that the pear vendor throws away 12% of the initial pears --/
theorem pear_vendor_theorem :
  ∀ initial_pears : ℝ, initial_pears > 0 → pear_vendor_problem initial_pears :=
by
  sorry

end NUMINAMATH_CALUDE_pear_vendor_theorem_l195_19598


namespace NUMINAMATH_CALUDE_jonathan_social_media_time_l195_19534

/-- Calculates the total time spent on social media in a week -/
def social_media_time_per_week (daily_time : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_time * days_in_week

/-- Proves that Jonathan spends 21 hours on social media in a week -/
theorem jonathan_social_media_time :
  social_media_time_per_week 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_social_media_time_l195_19534


namespace NUMINAMATH_CALUDE_leila_cake_consumption_l195_19561

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem leila_cake_consumption : 
  monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_leila_cake_consumption_l195_19561


namespace NUMINAMATH_CALUDE_fifteen_point_seven_billion_in_scientific_notation_l195_19567

-- Define the number of billions
def billions : ℝ := 15.7

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.57 * (10 ^ 9)

-- Theorem statement
theorem fifteen_point_seven_billion_in_scientific_notation :
  billions * (10 ^ 9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_fifteen_point_seven_billion_in_scientific_notation_l195_19567


namespace NUMINAMATH_CALUDE_line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l195_19579

-- Define basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Line : Type :=
  (p1 : Point) (p2 : Point)

structure Ray : Type :=
  (start : Point) (through : Point)

-- Define equality for lines
def line_eq (l1 l2 : Line) : Prop :=
  (l1.p1 = l2.p1 ∧ l1.p2 = l2.p2) ∨ (l1.p1 = l2.p2 ∧ l1.p2 = l2.p1)

-- Define inequality for rays
def ray_neq (r1 r2 : Ray) : Prop :=
  r1.start ≠ r2.start ∨ r1.through ≠ r2.through

-- Theorem statements
theorem line_ab_equals_ba (A B : Point) : 
  line_eq (Line.mk A B) (Line.mk B A) :=
sorry

theorem infinite_lines_through_point (P : Point) :
  ∀ n : ℕ, ∃ (lines : Fin n → Line), ∀ i : Fin n, (lines i).p1 = P :=
sorry

theorem ray_ab_not_equal_ba (A B : Point) :
  ray_neq (Ray.mk A B) (Ray.mk B A) :=
sorry

theorem ray_line_length_incomparable :
  ¬∃ (f : Ray → ℝ) (g : Line → ℝ), ∀ (r : Ray) (l : Line), f r < g l :=
sorry

end NUMINAMATH_CALUDE_line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l195_19579


namespace NUMINAMATH_CALUDE_quadratic_solution_l195_19550

theorem quadratic_solution : ∃ x : ℚ, x > 0 ∧ 5 * x^2 + 9 * x - 18 = 0 ∧ x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l195_19550


namespace NUMINAMATH_CALUDE_largest_fourth_number_l195_19536

/-- Represents a two-digit number -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n < 100}

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The problem setup -/
def fourthNumberProblem (a b c d : TwoDigitNumber) : Prop :=
  a.val = 34 ∧ b.val = 21 ∧ c.val = 65 ∧ 
  (∃ (x : ℕ), d.val = 40 + x ∧ x < 10) ∧
  4 * (sumOfDigits a.val + sumOfDigits b.val + sumOfDigits c.val + sumOfDigits d.val) = 
    a.val + b.val + c.val + d.val

/-- The theorem to be proved -/
theorem largest_fourth_number : 
  ∀ (a b c d : TwoDigitNumber), 
    fourthNumberProblem a b c d → d.val ≤ 49 := by sorry

end NUMINAMATH_CALUDE_largest_fourth_number_l195_19536


namespace NUMINAMATH_CALUDE_gcd_repeated_six_digit_l195_19540

def is_repeated_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_six_digit :
  ∃ g : ℕ, ∀ n : ℕ, is_repeated_six_digit n → Nat.gcd n g = g ∧ g = 1001 :=
sorry

end NUMINAMATH_CALUDE_gcd_repeated_six_digit_l195_19540


namespace NUMINAMATH_CALUDE_largest_number_l195_19517

theorem largest_number : ∀ (a b c : ℝ), 
  a = 5 ∧ b = 0 ∧ c = -2 → 
  a > b ∧ a > c ∧ a > -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l195_19517


namespace NUMINAMATH_CALUDE_symmetric_point_on_circle_l195_19548

/-- The circle equation: x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation: x - ay + 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  x - a*y + 2 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem symmetric_point_on_circle (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) →
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_circle_l195_19548


namespace NUMINAMATH_CALUDE_sets_subset_theorem_l195_19501

-- Define the sets P₁, P₂, Q₁, and Q₂
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}
def Q₁ (b : ℝ) : Set ℝ := {x | x^2 + x + b > 0}
def Q₂ (b : ℝ) : Set ℝ := {x | x^2 + 2*x + b > 0}

-- State the theorem
theorem sets_subset_theorem :
  (∀ a : ℝ, P₁ a ⊆ P₂ a) ∧ (∃ b : ℝ, Q₁ b ⊆ Q₂ b) := by
  sorry


end NUMINAMATH_CALUDE_sets_subset_theorem_l195_19501


namespace NUMINAMATH_CALUDE_smaller_number_proof_l195_19568

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l195_19568


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_area_relation_l195_19525

/-- Given a quadrilateral with area Q divided by its diagonals into 4 triangles with areas A, B, C, and D,
    prove that A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 -/
theorem quadrilateral_diagonal_area_relation (Q A B C D : ℝ) 
    (hQ : Q > 0) 
    (hA : A > 0) (hB : B > 0) (hC : C > 0) (hD : D > 0)
    (hSum : A + B + C + D = Q) : 
  A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_area_relation_l195_19525
