import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l885_88542

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a :
  ∀ x : ℝ, f a x + g x ≥ 3 → a ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l885_88542


namespace NUMINAMATH_CALUDE_total_distance_covered_l885_88562

/-- The total distance covered by a fox, rabbit, and deer given their speeds and running times -/
theorem total_distance_covered 
  (fox_speed : ℝ) 
  (rabbit_speed : ℝ) 
  (deer_speed : ℝ) 
  (fox_time : ℝ) 
  (rabbit_time : ℝ) 
  (deer_time : ℝ) 
  (h1 : fox_speed = 50) 
  (h2 : rabbit_speed = 60) 
  (h3 : deer_speed = 80) 
  (h4 : fox_time = 2) 
  (h5 : rabbit_time = 5/3) 
  (h6 : deer_time = 3/2) : 
  fox_speed * fox_time + rabbit_speed * rabbit_time + deer_speed * deer_time = 320 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_covered_l885_88562


namespace NUMINAMATH_CALUDE_sum_in_base5_l885_88510

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 --/
structure Base5 (n : ℕ) where
  value : ℕ
  isBase5 : value < 5^n

theorem sum_in_base5 :
  let a := base4ToBase10 203
  let b := base4ToBase10 112
  let c := base4ToBase10 321
  let sum := a + b + c
  base10ToBase5 sum = 2222 :=
sorry

end NUMINAMATH_CALUDE_sum_in_base5_l885_88510


namespace NUMINAMATH_CALUDE_sin_cos_identity_l885_88585

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l885_88585


namespace NUMINAMATH_CALUDE_min_value_theorem_l885_88584

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  3 * x + 1 / (x - 1) ≥ 2 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l885_88584


namespace NUMINAMATH_CALUDE_johns_goals_l885_88568

theorem johns_goals (total_goals : ℝ) (teammate_count : ℕ) (avg_teammate_goals : ℝ) :
  total_goals = 65 ∧
  teammate_count = 9 ∧
  avg_teammate_goals = 4.5 →
  total_goals - (teammate_count : ℝ) * avg_teammate_goals = 24.5 :=
by sorry

end NUMINAMATH_CALUDE_johns_goals_l885_88568


namespace NUMINAMATH_CALUDE_M₁_on_curve_M₂_not_on_curve_M₃_a_value_l885_88526

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (3 * t, 2 * t^2 + 1)

-- Define the points
def M₁ : ℝ × ℝ := (0, 1)
def M₂ : ℝ × ℝ := (5, 4)
def M₃ (a : ℝ) : ℝ × ℝ := (6, a)

-- Theorem statements
theorem M₁_on_curve : ∃ t : ℝ, curve_C t = M₁ := by sorry

theorem M₂_not_on_curve : ¬ ∃ t : ℝ, curve_C t = M₂ := by sorry

theorem M₃_a_value : ∃ a : ℝ, (∃ t : ℝ, curve_C t = M₃ a) → a = 9 := by sorry

end NUMINAMATH_CALUDE_M₁_on_curve_M₂_not_on_curve_M₃_a_value_l885_88526


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l885_88591

/-- Represents a recurring decimal with a single digit repeating -/
def RecurringDecimal (d : ℕ) : ℚ :=
  d / 9

theorem recurring_decimal_sum :
  let a := RecurringDecimal 5
  let b := RecurringDecimal 1
  let c := RecurringDecimal 3
  let d := RecurringDecimal 6
  a + b - c + d = 1 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l885_88591


namespace NUMINAMATH_CALUDE_child_ticket_cost_l885_88590

/-- Proves that the cost of a child ticket is 25 cents given the specified conditions. -/
theorem child_ticket_cost
  (adult_price : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 60)
  (h2 : total_attendees = 280)
  (h3 : total_revenue = 14000)  -- in cents
  (h4 : num_children = 80) :
  ∃ (child_price : ℕ),
    child_price * num_children + adult_price * (total_attendees - num_children) = total_revenue ∧
    child_price = 25 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l885_88590


namespace NUMINAMATH_CALUDE_diagonal_difference_bound_l885_88501

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d e f : ℝ)
  (cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (ptolemy : a * c + b * d = e * f)

-- State the theorem
theorem diagonal_difference_bound (q : CyclicQuadrilateral) :
  |q.e - q.f| ≤ |q.b - q.d| := by sorry

end NUMINAMATH_CALUDE_diagonal_difference_bound_l885_88501


namespace NUMINAMATH_CALUDE_rectangle_area_l885_88564

/-- Proves that a rectangle with length thrice its breadth and perimeter 64 has area 192 -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l885_88564


namespace NUMINAMATH_CALUDE_divisibility_property_l885_88558

theorem divisibility_property (p : ℕ) (hp : p > 3) (hodd : Odd p) :
  ∃ k : ℤ, (p - 3) ^ ((p - 1) / 2) - 1 = k * (p - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l885_88558


namespace NUMINAMATH_CALUDE_internal_tangent_locus_l885_88543

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are internally tangent -/
def are_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

/-- The locus of points -/
def locus (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem internal_tangent_locus (O1 O2 : Circle) 
  (h1 : O1.radius = 7)
  (h2 : O2.radius = 4)
  (h3 : are_internally_tangent O1 O2) :
  locus { center := O1.center, radius := 3 } O2.center :=
sorry

end NUMINAMATH_CALUDE_internal_tangent_locus_l885_88543


namespace NUMINAMATH_CALUDE_platform_length_l885_88555

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 26 seconds to cross a signal pole, prove that the length of the platform is 150 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 26) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 150 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l885_88555


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l885_88594

theorem square_root_of_sixteen : ∃ (x : ℝ), x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l885_88594


namespace NUMINAMATH_CALUDE_volleyball_preference_percentage_l885_88532

theorem volleyball_preference_percentage
  (north_students : ℕ)
  (south_students : ℕ)
  (north_volleyball_percentage : ℚ)
  (south_volleyball_percentage : ℚ)
  (h1 : north_students = 1800)
  (h2 : south_students = 2700)
  (h3 : north_volleyball_percentage = 25 / 100)
  (h4 : south_volleyball_percentage = 35 / 100)
  : (north_students * north_volleyball_percentage + south_students * south_volleyball_percentage) /
    (north_students + south_students) = 31 / 100 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_preference_percentage_l885_88532


namespace NUMINAMATH_CALUDE_g_of_5_eq_neg_7_l885_88556

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 2 * x^4 - 15 * x^3 + 24 * x^2 - 18 * x - 72

/-- Theorem: g(5) equals -7 -/
theorem g_of_5_eq_neg_7 : g 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_eq_neg_7_l885_88556


namespace NUMINAMATH_CALUDE_unused_streetlights_l885_88517

theorem unused_streetlights (total_streetlights : ℕ) (num_squares : ℕ) (lights_per_square : ℕ) :
  total_streetlights = 200 →
  num_squares = 15 →
  lights_per_square = 12 →
  total_streetlights - (num_squares * lights_per_square) = 20 := by
  sorry

#check unused_streetlights

end NUMINAMATH_CALUDE_unused_streetlights_l885_88517


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l885_88561

theorem negative_abs_negative_three : -|-3| = -3 := by sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l885_88561


namespace NUMINAMATH_CALUDE_mean_equality_problem_l885_88506

theorem mean_equality_problem : ∃ z : ℚ, (7 + 12 + 21) / 3 = (15 + z) / 2 ∧ z = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l885_88506


namespace NUMINAMATH_CALUDE_lastTwoDigitsOf7To2020_l885_88566

-- Define the function that gives the last two digits of 7^n
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

-- State the periodicity of the last two digits
axiom lastTwoDigitsPeriodicity (n : ℕ) (h : n ≥ 2) : 
  lastTwoDigits n = lastTwoDigits (n % 4 + 4)

-- Define the theorem
theorem lastTwoDigitsOf7To2020 : lastTwoDigits 2020 = 01 := by
  sorry

end NUMINAMATH_CALUDE_lastTwoDigitsOf7To2020_l885_88566


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l885_88522

theorem min_sum_of_squares (m n : ℕ) (h1 : n = m + 1) (h2 : n^2 - m^2 > 20) :
  ∃ (k : ℕ), k = n^2 + m^2 ∧ k ≥ 221 ∧ ∀ (j : ℕ), (∃ (p q : ℕ), q = p + 1 ∧ q^2 - p^2 > 20 ∧ j = q^2 + p^2) → j ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l885_88522


namespace NUMINAMATH_CALUDE_percentage_decrease_l885_88516

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.3 * y ∧ x = 0.65 * z → y = 0.5 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l885_88516


namespace NUMINAMATH_CALUDE_student_tickets_sold_l885_88537

theorem student_tickets_sold (total_tickets : ℕ) (student_price non_student_price total_money : ℚ)
  (h1 : total_tickets = 193)
  (h2 : student_price = 1/2)
  (h3 : non_student_price = 3/2)
  (h4 : total_money = 206.5)
  (h5 : ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_tickets * student_price + non_student_tickets * non_student_price = total_money) :
  ∃ (student_tickets : ℕ), student_tickets = 83 :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l885_88537


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l885_88540

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 0}

theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l885_88540


namespace NUMINAMATH_CALUDE_lcm_problem_l885_88574

theorem lcm_problem (a b : ℕ) (h : Nat.gcd a b = 47) (ha : a = 210) (hb : b = 517) :
  Nat.lcm a b = 2310 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l885_88574


namespace NUMINAMATH_CALUDE_spending_difference_l885_88587

/-- The cost of the computer table in dollars -/
def table_cost : ℚ := 140

/-- The cost of the computer chair in dollars -/
def chair_cost : ℚ := 100

/-- The cost of the joystick in dollars -/
def joystick_cost : ℚ := 20

/-- Frank's share of the joystick cost -/
def frank_joystick_share : ℚ := 1/4

/-- Eman's share of the joystick cost -/
def eman_joystick_share : ℚ := 1 - frank_joystick_share

/-- Frank's total spending -/
def frank_total : ℚ := table_cost + frank_joystick_share * joystick_cost

/-- Eman's total spending -/
def eman_total : ℚ := chair_cost + eman_joystick_share * joystick_cost

theorem spending_difference : frank_total - eman_total = 30 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l885_88587


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l885_88550

theorem expression_simplification_and_evaluation :
  let x : ℚ := -1
  let y : ℚ := -1/2
  let original_expression := 4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y)
  let simplified_expression := 3*x*y - y^2
  original_expression = simplified_expression ∧ simplified_expression = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l885_88550


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_product_l885_88599

theorem pure_imaginary_complex_product (a : ℝ) : 
  (Complex.im ((1 + a * Complex.I) * (3 - Complex.I)) ≠ 0 ∧ 
   Complex.re ((1 + a * Complex.I) * (3 - Complex.I)) = 0) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_product_l885_88599


namespace NUMINAMATH_CALUDE_symmetric_line_is_symmetric_l885_88579

/-- The point of symmetry -/
def P : ℝ × ℝ := (2, -1)

/-- The equation of the original line: 3x - y - 4 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - y - 4 = 0

/-- The equation of the symmetric line: 3x - y - 7 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x - y - 7 = 0

/-- Definition of symmetry with respect to a point -/
def is_symmetric (line1 line2 : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ),
    line1 x1 y1 → line2 x2 y2 →
    (x1 + x2) / 2 = p.1 ∧ (y1 + y2) / 2 = p.2

/-- The main theorem: the symmetric_line is symmetric to the original_line with respect to P -/
theorem symmetric_line_is_symmetric :
  is_symmetric original_line symmetric_line P :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_is_symmetric_l885_88579


namespace NUMINAMATH_CALUDE_next_five_even_sum_l885_88571

/-- Given a sum 'a' of 5 consecutive even positive integers, 
    the sum of the next 5 consecutive even integers is a + 50 -/
theorem next_five_even_sum (a : ℕ) (x : ℕ) 
  (h1 : x > 0) 
  (h2 : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end NUMINAMATH_CALUDE_next_five_even_sum_l885_88571


namespace NUMINAMATH_CALUDE_negative_square_times_cube_l885_88559

theorem negative_square_times_cube (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_times_cube_l885_88559


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l885_88503

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t,
    prove that t + a_3 = 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  (a 1 * a 3 = (a 2)^2) →
  t + a 3 = 17 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l885_88503


namespace NUMINAMATH_CALUDE_senior_ticket_price_l885_88520

theorem senior_ticket_price 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_senior_ticket_price_l885_88520


namespace NUMINAMATH_CALUDE_median_squares_sum_l885_88554

/-- Given a triangle with side lengths 13, 14, and 15, the sum of the squares of its median lengths is 442.5 -/
theorem median_squares_sum (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let median_sum_squares := 3/4 * (a^2 + b^2 + c^2)
  median_sum_squares = 442.5 := by
  sorry

end NUMINAMATH_CALUDE_median_squares_sum_l885_88554


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_113_l885_88504

theorem alpha_plus_beta_equals_113 (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90*x + 1981) / (x^2 + 63*x - 3420)) →
  α + β = 113 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_113_l885_88504


namespace NUMINAMATH_CALUDE_product_of_divisors_of_product_of_divisors_of_2005_l885_88541

def divisors (n : ℕ) : Finset ℕ := sorry

def divisor_product (n : ℕ) : ℕ := (divisors n).prod id

theorem product_of_divisors_of_product_of_divisors_of_2005 :
  divisor_product (divisor_product 2005) = 2005^9 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_of_product_of_divisors_of_2005_l885_88541


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l885_88530

theorem pigeonhole_divisibility (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l885_88530


namespace NUMINAMATH_CALUDE_simplify_expression_l885_88598

theorem simplify_expression (a b : ℝ) : (2*a^2 - 3*a*b + 8) - (-a*b - a^2 + 8) = 3*a^2 - 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l885_88598


namespace NUMINAMATH_CALUDE_sum_x₁_x₂_equals_three_l885_88525

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_pos : 0 < p₁ ∧ 0 < p₂

/-- The expected value of a discrete random variable -/
def expected_value (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- The variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expected_value X)^2 + X.p₂ * (X.x₂ - expected_value X)^2

/-- Theorem stating the sum of x₁ and x₂ for the given conditions -/
theorem sum_x₁_x₂_equals_three (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_exp : expected_value X = 4/3)
  (h_var : variance X = 2/9) :
  X.x₁ + X.x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x₁_x₂_equals_three_l885_88525


namespace NUMINAMATH_CALUDE_base7_to_base10_ABC21_l885_88518

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 2401 + b * 343 + c * 49 + 15

/-- Theorem: The base 10 equivalent of ABC21₇ is A · 2401 + B · 343 + C · 49 + 15 --/
theorem base7_to_base10_ABC21 (A B C : Nat) 
  (hA : A ≤ 6) (hB : B ≤ 6) (hC : C ≤ 6) :
  base7ToBase10 A B C = A * 2401 + B * 343 + C * 49 + 15 := by
  sorry

#check base7_to_base10_ABC21

end NUMINAMATH_CALUDE_base7_to_base10_ABC21_l885_88518


namespace NUMINAMATH_CALUDE_hundredth_stationary_is_hundred_l885_88563

/-- A function representing the sorting algorithm that swaps adjacent numbers if the larger number is on the left -/
def sortPass (s : List ℕ) : List ℕ := sorry

/-- A predicate that checks if a number at a given index remains stationary during both passes -/
def isStationary (s : List ℕ) (index : ℕ) : Prop := sorry

theorem hundredth_stationary_is_hundred {s : List ℕ} (h1 : s.length = 1982) 
  (h2 : ∀ n, n ∈ s → 1 ≤ n ∧ n ≤ 1982) 
  (h3 : isStationary s 100) : 
  s[99] = 100 := by sorry

end NUMINAMATH_CALUDE_hundredth_stationary_is_hundred_l885_88563


namespace NUMINAMATH_CALUDE_no_integer_solution_l885_88534

theorem no_integer_solution : ¬∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l885_88534


namespace NUMINAMATH_CALUDE_percentage_of_students_passed_l885_88573

/-- Given an examination where 700 students appeared and 455 failed,
    prove that 35% of students passed the examination. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 700) (h2 : failed = 455) :
  (total - failed : ℚ) / total * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_passed_l885_88573


namespace NUMINAMATH_CALUDE_angle_U_is_90_degrees_l885_88553

-- Define the hexagon FIGURE
structure Hexagon where
  F : ℝ
  I : ℝ
  U : ℝ
  G : ℝ
  R : ℝ
  E : ℝ

-- Define the conditions
def hexagon_conditions (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U ∧ 
  h.G + h.E = 180 ∧ 
  h.R + h.U = 180 ∧
  h.F + h.I + h.U + h.G + h.R + h.E = 720

-- Theorem statement
theorem angle_U_is_90_degrees (h : Hexagon) 
  (hc : hexagon_conditions h) : h.U = 90 := by sorry

end NUMINAMATH_CALUDE_angle_U_is_90_degrees_l885_88553


namespace NUMINAMATH_CALUDE_profit_percentage_l885_88533

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 95 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l885_88533


namespace NUMINAMATH_CALUDE_symmetric_points_values_l885_88548

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_values :
  ∀ m n : ℝ,
  symmetric_about_y_axis (-3) (2*m - 1) (n + 1) 4 →
  m = 2.5 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_values_l885_88548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l885_88508

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 9 = 20 →
  4 * a 5 - a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l885_88508


namespace NUMINAMATH_CALUDE_speed_above_limit_l885_88509

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem speed_above_limit : (distance / time) - speed_limit = 15 := by
  sorry

end NUMINAMATH_CALUDE_speed_above_limit_l885_88509


namespace NUMINAMATH_CALUDE_line_equation_theorem_l885_88595

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if the given equation represents the line -/
def isEquationOfLine (a b c : ℝ) (l : Line) : Prop :=
  a ≠ 0 ∧ 
  l.slope = -a / b ∧
  l.yIntercept = -c / b

/-- The main theorem: the equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_theorem : 
  let l : Line := { slope := 3, yIntercept := 4 }
  isEquationOfLine 3 (-1) 4 l := by
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l885_88595


namespace NUMINAMATH_CALUDE_birthday_money_allocation_l885_88535

theorem birthday_money_allocation (total : ℚ) (books snacks apps games : ℚ) : 
  total = 50 ∧ 
  books = (1 : ℚ) / 4 * total ∧
  snacks = (3 : ℚ) / 10 * total ∧
  apps = (7 : ℚ) / 20 * total ∧
  games = total - (books + snacks + apps) →
  games = 5 := by sorry

end NUMINAMATH_CALUDE_birthday_money_allocation_l885_88535


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l885_88531

theorem integer_solutions_equation (n m : ℤ) : 
  n^6 + 3*n^5 + 3*n^4 + 2*n^3 + 3*n^2 + 3*n + 1 = m^3 ↔ (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l885_88531


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l885_88565

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define points P and Q on the parabola
def P : ℝ × ℝ := (4, 8)
def Q : ℝ × ℝ := (-2, 2)

-- Define the tangent lines at P and Q
def tangent_P (x y : ℝ) : Prop := y = 4*x - 8
def tangent_Q (x y : ℝ) : Prop := y = -2*x - 2

-- Define the intersection point A
def A : ℝ × ℝ := (1, -4)

-- Theorem statement
theorem intersection_y_coordinate :
  parabola P.1 P.2 ∧ 
  parabola Q.1 Q.2 ∧ 
  tangent_P A.1 A.2 ∧ 
  tangent_Q A.1 A.2 →
  A.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l885_88565


namespace NUMINAMATH_CALUDE_existence_of_distinct_integers_l885_88570

theorem existence_of_distinct_integers (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n^2 < a ∧ a < (n+1)^2 ∧
    n^2 < b ∧ b < (n+1)^2 ∧
    n^2 < c ∧ c < (n+1)^2 ∧
    (c ∣ a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_distinct_integers_l885_88570


namespace NUMINAMATH_CALUDE_suit_price_theorem_l885_88512

theorem suit_price_theorem (original_price : ℝ) : 
  (original_price * 1.25 * 0.75 = 150) → original_price = 160 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_theorem_l885_88512


namespace NUMINAMATH_CALUDE_arman_age_problem_l885_88588

/-- Given that Arman is six times older than his sister, his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_problem (sister_age_4_years_ago : ℕ) (arman_age sister_age : ℕ) :
  sister_age_4_years_ago = 2 →
  sister_age = sister_age_4_years_ago + 4 →
  arman_age = 6 * sister_age →
  40 - arman_age = 4 := by
sorry

end NUMINAMATH_CALUDE_arman_age_problem_l885_88588


namespace NUMINAMATH_CALUDE_henry_games_count_l885_88502

theorem henry_games_count :
  ∀ (h n l : ℕ),
    h = 3 * n →                 -- Henry had 3 times as many games as Neil initially
    h = 2 * l →                 -- Henry had 2 times as many games as Linda initially
    n = 7 →                     -- Neil had 7 games initially
    l = 7 →                     -- Linda had 7 games initially
    h - 10 = 4 * (n + 6) →      -- After giving games, Henry has 4 times more games than Neil
    h = 62                      -- Henry originally had 62 games
  := by sorry

end NUMINAMATH_CALUDE_henry_games_count_l885_88502


namespace NUMINAMATH_CALUDE_extreme_value_probability_l885_88529

-- Define the die outcomes
def DieOutcome := Fin 6

-- Define the probability space
def Ω := DieOutcome × DieOutcome

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the condition for extreme value
def hasExtremeValue (a b : ℕ) : Prop := a^2 > 4*b

-- State the theorem
theorem extreme_value_probability : 
  P {ω : Ω | hasExtremeValue ω.1.val.succ ω.2.val.succ} = 17/36 := by sorry

end NUMINAMATH_CALUDE_extreme_value_probability_l885_88529


namespace NUMINAMATH_CALUDE_mandy_shirts_total_l885_88505

theorem mandy_shirts_total (black_packs yellow_packs : ℕ) 
  (black_per_pack yellow_per_pack : ℕ) : 
  black_packs = 3 → 
  yellow_packs = 3 → 
  black_per_pack = 5 → 
  yellow_per_pack = 2 → 
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = 21 := by
  sorry

#check mandy_shirts_total

end NUMINAMATH_CALUDE_mandy_shirts_total_l885_88505


namespace NUMINAMATH_CALUDE_evaluate_expression_l885_88572

theorem evaluate_expression : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l885_88572


namespace NUMINAMATH_CALUDE_mary_money_left_l885_88577

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 2 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the amount of money Mary has left is 50 - 10p -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l885_88577


namespace NUMINAMATH_CALUDE_min_value_of_f_l885_88569

/-- The quadratic function f(x) = (x-2)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l885_88569


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l885_88575

/-- Given two real numbers a and b such that lg a + lg b = 0, a ≠ 1, and b ≠ 1,
    the functions f(x) = a^x and g(x) = b^x are symmetric with respect to the y-axis. -/
theorem symmetry_wrt_y_axis (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) 
    (h : Real.log a + Real.log b = 0) :
  ∀ x : ℝ, a^(-x) = b^x := by
  sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l885_88575


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l885_88507

def original_function (x : ℝ) : ℝ := -2 * x

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

def resulting_function (x : ℝ) : ℝ := -2 * x + 6

theorem shift_direct_proportion :
  shift_right original_function 3 = resulting_function := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l885_88507


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l885_88592

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersection_plane_plane : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_parallel_α : parallel_line_plane m α) 
  (h_m_subset_β : subset_line_plane m β) 
  (h_intersection : intersection_plane_plane α β = n) : 
  parallel_line_line m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l885_88592


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l885_88524

theorem consecutive_integers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l885_88524


namespace NUMINAMATH_CALUDE_shelter_ratio_change_l885_88521

/-- Proves that given an initial ratio of dogs to cats of 15:7, 60 dogs in the shelter,
    and 16 additional cats taken in, the new ratio of dogs to cats is 15:11. -/
theorem shelter_ratio_change (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) :
  initial_dogs = 60 →
  initial_dogs / initial_cats = 15 / 7 →
  additional_cats = 16 →
  initial_dogs / (initial_cats + additional_cats) = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_shelter_ratio_change_l885_88521


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l885_88578

def largest_power_of_two_dividing_factorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing_factorial 32)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l885_88578


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l885_88567

theorem circus_ticket_sales (lower_price upper_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  lower_price = 30 → 
  upper_price = 20 → 
  total_tickets = 80 → 
  total_revenue = 2100 → 
  ∃ (lower_seats upper_seats : ℕ), 
    lower_seats + upper_seats = total_tickets ∧ 
    lower_price * lower_seats + upper_price * upper_seats = total_revenue ∧ 
    lower_seats = 50 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l885_88567


namespace NUMINAMATH_CALUDE_second_group_size_l885_88596

/-- The number of persons in the first group -/
def first_group : ℕ := 78

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the job -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem second_group_size :
  second_group = 130 := by sorry

end NUMINAMATH_CALUDE_second_group_size_l885_88596


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l885_88581

theorem units_digit_of_expression : ∃ n : ℕ, (12 + Real.sqrt 36)^17 + (12 - Real.sqrt 36)^17 = 10 * n + 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l885_88581


namespace NUMINAMATH_CALUDE_largest_x_value_l885_88580

theorem largest_x_value (x : ℝ) : 
  x ≠ 7 → 
  ((x^2 - 5*x - 84) / (x - 7) = 2 / (x + 6)) → 
  x ≤ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l885_88580


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l885_88513

theorem existence_of_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b / a ≥ (b + c) / (a + c) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l885_88513


namespace NUMINAMATH_CALUDE_price_reduction_equation_l885_88536

theorem price_reduction_equation (initial_price final_price : ℝ) 
  (h1 : initial_price = 188) 
  (h2 : final_price = 108) 
  (x : ℝ) -- x represents the percentage of each reduction
  (h3 : x ≥ 0 ∧ x < 1) -- ensure x is a valid percentage
  (h4 : final_price = initial_price * (1 - x)^2) -- two equal reductions
  : initial_price * (1 - x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l885_88536


namespace NUMINAMATH_CALUDE_short_story_booklets_l885_88539

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 441 / 9

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := 441

theorem short_story_booklets :
  num_booklets = 49 ∧
  pages_per_booklet * num_booklets = total_pages :=
sorry

end NUMINAMATH_CALUDE_short_story_booklets_l885_88539


namespace NUMINAMATH_CALUDE_a_range_characterization_l885_88593

/-- Proposition p: The domain of the logarithm function is all real numbers -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 7*a - 6 > 0

/-- Proposition q: There exists a real x satisfying the quadratic inequality -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 4 < 0

/-- The set of real numbers a where either p is true and q is false, or p is false and q is true -/
def a_range : Set ℝ := {a | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)}

theorem a_range_characterization : 
  a_range = {a | a < -4 ∨ (1 < a ∧ a ≤ 4) ∨ 6 ≤ a} :=
sorry

end NUMINAMATH_CALUDE_a_range_characterization_l885_88593


namespace NUMINAMATH_CALUDE_inspection_sample_size_l885_88586

/-- Represents a batch of leather shoes -/
structure ShoeBatch where
  total : ℕ

/-- Represents a quality inspection of shoes -/
structure QualityInspection where
  batch : ShoeBatch
  drawn : ℕ

/-- Definition of sample size for a quality inspection -/
def sampleSize (inspection : QualityInspection) : ℕ :=
  inspection.drawn

theorem inspection_sample_size (batch : ShoeBatch) :
  let inspection := QualityInspection.mk batch 50
  sampleSize inspection = 50 := by
  sorry

end NUMINAMATH_CALUDE_inspection_sample_size_l885_88586


namespace NUMINAMATH_CALUDE_coefficient_of_x_term_l885_88515

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (Real.sqrt x - 1)^4 * (x - 1)^2
  ∃ (a b c d e f : ℝ), expansion = a*x^3 + b*x^(5/2) + c*x^2 + d*x^(3/2) + 4*x + f*x^(1/2) + e
  := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_term_l885_88515


namespace NUMINAMATH_CALUDE_train_departure_time_l885_88560

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiff (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem train_departure_time 
  (arrival : Time)
  (journey_duration : Nat)
  (h_arrival : arrival.hours = 10 ∧ arrival.minutes = 0)
  (h_duration : journey_duration = 15) :
  ∃ (departure : Time), 
    timeDiff arrival departure = journey_duration ∧ 
    departure.hours = 9 ∧ 
    departure.minutes = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_departure_time_l885_88560


namespace NUMINAMATH_CALUDE_ruths_track_length_l885_88552

theorem ruths_track_length (sean_piece_length ruth_piece_length total_length : ℝ) :
  sean_piece_length = 8 →
  total_length = 72 →
  (total_length / sean_piece_length) * sean_piece_length = (total_length / ruth_piece_length) * ruth_piece_length →
  ruth_piece_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_ruths_track_length_l885_88552


namespace NUMINAMATH_CALUDE_function_inequalities_l885_88549

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a^(3*x + 1)
def g (x : ℝ) : ℝ := (1/a)^(5*x - 2)

theorem function_inequalities (h1 : a > 0) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1 → (f a x < 1 ↔ x > -1/3)) ∧
  ((0 < a ∧ a < 1 → (f a x ≥ g a x ↔ x ≤ 1/8)) ∧
   (a > 1 → (f a x ≥ g a x ↔ x ≥ 1/8))) := by
  sorry

end

end NUMINAMATH_CALUDE_function_inequalities_l885_88549


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l885_88557

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.cos t.B = t.b * Real.sin t.A)
  (h2 : (Real.sqrt 3 / 4) * t.b^2 = (1/2) * t.a * t.c * Real.sin t.B) :
  t.a / t.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l885_88557


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l885_88519

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for part (I)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for part (II)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l885_88519


namespace NUMINAMATH_CALUDE_four_painters_work_days_l885_88597

/-- The number of work-days required for a given number of painters to complete a job -/
def work_days (num_painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / num_painters

theorem four_painters_work_days :
  let total_work : ℚ := 6 * (3/2)  -- 6 painters * 1.5 days
  (work_days 4 total_work) = 2 + (1/4) := by sorry

end NUMINAMATH_CALUDE_four_painters_work_days_l885_88597


namespace NUMINAMATH_CALUDE_solution_to_equation_l885_88500

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * b + a + b + 2

-- State the theorem
theorem solution_to_equation :
  ∃ x : ℝ, custom_op x 3 = 1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l885_88500


namespace NUMINAMATH_CALUDE_gabby_makeup_set_l885_88582

/-- The amount of money Gabby's mom gave her -/
def moms_gift (cost savings needed_after : ℕ) : ℕ :=
  cost - savings - needed_after

/-- Proof that Gabby's mom gave her $20 -/
theorem gabby_makeup_set : moms_gift 65 35 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gabby_makeup_set_l885_88582


namespace NUMINAMATH_CALUDE_imaginary_power_2019_l885_88576

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2019 : i^2019 = -i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_2019_l885_88576


namespace NUMINAMATH_CALUDE_john_average_speed_l885_88514

/-- John's average speed in miles per hour -/
def john_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time Carla needs to catch up to John in hours -/
def catch_up_time : ℝ := 3

/-- Time difference between John's and Carla's departure in hours -/
def departure_time_difference : ℝ := 0.5

theorem john_average_speed :
  john_speed = 30 ∧
  carla_speed * catch_up_time = john_speed * (catch_up_time + departure_time_difference) :=
sorry

end NUMINAMATH_CALUDE_john_average_speed_l885_88514


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l885_88544

theorem diophantine_equation_solution : ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l885_88544


namespace NUMINAMATH_CALUDE_quadratic_roots_from_intersections_l885_88527

/-- Given a quadratic function f(x) = ax² + bx + c, if its graph intersects
    the x-axis at (1,0) and (4,0), then the solutions to ax² + bx + c = 0
    are x₁ = 1 and x₂ = 4. -/
theorem quadratic_roots_from_intersections
  (a b c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^2 + b * x + c) :
  f 1 = 0 → f 4 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_from_intersections_l885_88527


namespace NUMINAMATH_CALUDE_sun_division_l885_88546

theorem sun_division (x y z total : ℚ) : 
  (y = (45/100) * x) →  -- y gets 45 paisa for each rupee x gets
  (z = (30/100) * x) →  -- z gets 30 paisa for each rupee x gets
  (y = 63) →            -- y's share is Rs. 63
  (total = x + y + z) → -- total is the sum of all shares
  (total = 245) :=      -- prove that the total is Rs. 245
by
  sorry

end NUMINAMATH_CALUDE_sun_division_l885_88546


namespace NUMINAMATH_CALUDE_total_books_count_l885_88528

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 6

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 54 := by sorry

end NUMINAMATH_CALUDE_total_books_count_l885_88528


namespace NUMINAMATH_CALUDE_infinite_non_prime_generating_numbers_l885_88538

theorem infinite_non_prime_generating_numbers :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, m > 1 → ∀ n : ℕ, ¬ Nat.Prime (n^4 + f m) := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_prime_generating_numbers_l885_88538


namespace NUMINAMATH_CALUDE_find_x_value_l885_88589

theorem find_x_value (x : ℝ) :
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.9365079365079367) →
  x = 1.21 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l885_88589


namespace NUMINAMATH_CALUDE_gas_refill_amount_l885_88547

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : ℕ :=
  tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_amount :
  gas_problem 10 12 6 2 = 10 := by sorry

end NUMINAMATH_CALUDE_gas_refill_amount_l885_88547


namespace NUMINAMATH_CALUDE_curve_equation_relationship_l885_88583

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for curves in 2D space
structure Curve where
  points : Set Point2D

-- Define a function type for equations in 2D space
def Equation2D := Point2D → Prop

-- Define the given condition
def satisfiesEquation (C : Curve) (f : Equation2D) : Prop :=
  ∀ p ∈ C.points, f p

-- Theorem statement
theorem curve_equation_relationship (C : Curve) (f : Equation2D) :
  satisfiesEquation C f →
  ¬ (∀ p : Point2D, f p ↔ p ∈ C.points) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_relationship_l885_88583


namespace NUMINAMATH_CALUDE_problem_solution_l885_88511

theorem problem_solution : ∃ x : ℝ, 4 * x - 4 = 2 * 4 + 20 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l885_88511


namespace NUMINAMATH_CALUDE_simplify_fraction_l885_88545

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l885_88545


namespace NUMINAMATH_CALUDE_function_properties_l885_88551

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem proves properties about its maximum value and zero points. -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ x - a * Real.exp x + b
  -- The maximum value of f occurs at ln(1/a) and equals ln(1/a) - 1 + b
  ∃ (x_max : ℝ), x_max = Real.log (1/a) ∧
    ∀ x, f x ≤ f x_max ∧ f x_max = Real.log (1/a) - 1 + b ∧
  -- If f has two distinct zero points, their sum is less than -2*ln(a)
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l885_88551


namespace NUMINAMATH_CALUDE_product_101_101_l885_88523

theorem product_101_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_product_101_101_l885_88523
