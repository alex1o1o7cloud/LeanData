import Mathlib

namespace NUMINAMATH_CALUDE_A_not_always_in_second_quadrant_l2728_272823

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of being on the negative x-axis -/
def isOnNegativeXAxis (p : Point) : Prop :=
  p.x < 0 ∧ p.y = 0

/-- The point A(-a^2-1,|b|) -/
def A (a b : ℝ) : Point :=
  { x := -a^2 - 1, y := |b| }

/-- Theorem stating that A(-a^2-1,|b|) is not always in the second quadrant -/
theorem A_not_always_in_second_quadrant :
  ∃ a b : ℝ, ¬(isInSecondQuadrant (A a b)) ∧ (isInSecondQuadrant (A a b) ∨ isOnNegativeXAxis (A a b)) :=
sorry

end NUMINAMATH_CALUDE_A_not_always_in_second_quadrant_l2728_272823


namespace NUMINAMATH_CALUDE_complex_square_quadrant_l2728_272890

theorem complex_square_quadrant (z : ℂ) : 
  z = Complex.exp (Complex.I * Real.pi * (5/12)) → 
  (z^2).re < 0 ∧ (z^2).im > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_square_quadrant_l2728_272890


namespace NUMINAMATH_CALUDE_line_in_fourth_quadrant_l2728_272862

/-- A line passes through the fourth quadrant if it intersects both the negative x-axis and the positive y-axis. -/
def passes_through_fourth_quadrant (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (a - 2) * x + a * y + 2 * a - 3 = 0

/-- The theorem stating the condition for the line to pass through the fourth quadrant. -/
theorem line_in_fourth_quadrant (a : ℝ) :
  passes_through_fourth_quadrant a ↔ a ∈ Set.Iio 0 ∪ Set.Ioi (3/2) :=
sorry

end NUMINAMATH_CALUDE_line_in_fourth_quadrant_l2728_272862


namespace NUMINAMATH_CALUDE_oranges_per_box_l2728_272886

/-- Given 24 oranges distributed equally among 3 boxes, prove that each box contains 8 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : total_oranges = 24) 
  (h2 : num_boxes = 3) 
  (h3 : oranges_per_box * num_boxes = total_oranges) : 
  oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2728_272886


namespace NUMINAMATH_CALUDE_multiple_choice_test_choices_l2728_272821

theorem multiple_choice_test_choices (n : ℕ) : 
  (n + 1)^4 = 625 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiple_choice_test_choices_l2728_272821


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l2728_272800

theorem half_radius_circle_y (x y : Real) : 
  (2 * Real.pi * x = 10 * Real.pi) →  -- Circumference of circle x is 10π
  (Real.pi * x^2 = Real.pi * y^2) →   -- Areas of circles x and y are equal
  (1/2) * y = 2.5 := by               -- Half of the radius of circle y is 2.5
sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l2728_272800


namespace NUMINAMATH_CALUDE_tim_younger_than_jenny_l2728_272809

-- Define the ages
def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2

-- Theorem statement
theorem tim_younger_than_jenny : jenny_age - tim_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_tim_younger_than_jenny_l2728_272809


namespace NUMINAMATH_CALUDE_nancy_alyssa_book_ratio_l2728_272824

def alyssa_books : ℕ := 36
def nancy_books : ℕ := 252

theorem nancy_alyssa_book_ratio :
  nancy_books / alyssa_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_alyssa_book_ratio_l2728_272824


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l2728_272868

theorem sum_of_three_squares (K : ℕ) (L : ℤ) (h : L % 8 = 7) :
  ¬ ∃ (a b c : ℤ), 4^K * L = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l2728_272868


namespace NUMINAMATH_CALUDE_line_circle_separation_l2728_272877

theorem line_circle_separation (a b : ℝ) 
  (h_inside : a^2 + b^2 < 1) : 
  ∃ (d : ℝ), d > 1 ∧ d = 1 / Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_separation_l2728_272877


namespace NUMINAMATH_CALUDE_handshakes_in_specific_event_l2728_272804

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- people who know each other
  group2_size : ℕ  -- people who know no one
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_in_specific_event :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_event_l2728_272804


namespace NUMINAMATH_CALUDE_existence_of_prime_divisor_greater_than_ten_l2728_272806

/-- A function that returns the smallest prime divisor of a natural number -/
def smallest_prime_divisor (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem existence_of_prime_divisor_greater_than_ten (start : ℕ) 
  (h_start : is_four_digit start) :
  ∃ k : ℕ, k < 10 ∧ smallest_prime_divisor (start + k) > 10 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_divisor_greater_than_ten_l2728_272806


namespace NUMINAMATH_CALUDE_total_apples_is_100_l2728_272817

-- Define the types of apples
inductive AppleType
| Sweet
| Sour

-- Define the price function for apples
def applePrice : AppleType → ℚ
| AppleType.Sweet => 1/2
| AppleType.Sour => 1/10

-- Define the proportion of sweet apples
def sweetProportion : ℚ := 3/4

-- Define the total earnings
def totalEarnings : ℚ := 40

-- Theorem statement
theorem total_apples_is_100 :
  ∃ (n : ℕ), n = 100 ∧
  n * (sweetProportion * applePrice AppleType.Sweet +
       (1 - sweetProportion) * applePrice AppleType.Sour) = totalEarnings :=
by
  sorry


end NUMINAMATH_CALUDE_total_apples_is_100_l2728_272817


namespace NUMINAMATH_CALUDE_inequality_solution_l2728_272867

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | 1 < x }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ 1 < x }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | a * x^2 - (a + 2) * x + 2 < 0 } = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2728_272867


namespace NUMINAMATH_CALUDE_inequality_proof_l2728_272816

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2728_272816


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l2728_272811

theorem min_value_sum_squared_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l2728_272811


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2728_272898

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2728_272898


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2728_272854

/-- A parallelogram with an area of 450 sq m and an altitude twice the corresponding base has a base length of 15 meters. -/
theorem parallelogram_base_length :
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  base * altitude = 450 →
  altitude = 2 * base →
  base = 15 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2728_272854


namespace NUMINAMATH_CALUDE_correct_answer_is_nothing_l2728_272887

/-- Represents the possible answers to the Question of Questions -/
inductive Answer
| Something
| Nothing

/-- Represents a priest -/
structure Priest where
  knowsAnswer : Bool
  alwaysLies : Bool

/-- The response given by a priest -/
def priestResponse : Answer := Answer.Something

/-- Theorem: If a priest who knows the correct answer responds with "Something exists,"
    then the correct answer is "Nothing exists" -/
theorem correct_answer_is_nothing 
  (priest : Priest) 
  (h1 : priest.knowsAnswer = true) 
  (h2 : priest.alwaysLies = true) 
  (h3 : priestResponse = Answer.Something) : 
  Answer.Nothing = Answer.Nothing := by sorry


end NUMINAMATH_CALUDE_correct_answer_is_nothing_l2728_272887


namespace NUMINAMATH_CALUDE_cubic_sum_powers_l2728_272826

theorem cubic_sum_powers (a : ℝ) (h : a^3 + 3*a^2 + 3*a + 2 = 0) :
  (a + 1)^2008 + (a + 1)^2009 + (a + 1)^2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_powers_l2728_272826


namespace NUMINAMATH_CALUDE_sequence_a_integer_sequence_a_recurrence_l2728_272892

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) * (sequence_a (n + 1) + 1) / sequence_a n

theorem sequence_a_integer (n : ℕ) : ∃ k : ℤ, sequence_a n = k := by
  sorry

theorem sequence_a_recurrence (n : ℕ) : n ≥ 1 →
  sequence_a (n + 1) * sequence_a (n - 1) = sequence_a n * (sequence_a n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_integer_sequence_a_recurrence_l2728_272892


namespace NUMINAMATH_CALUDE_ticket_cost_l2728_272836

/-- Given that 7 tickets were purchased for a total of $308, prove that each ticket costs $44. -/
theorem ticket_cost (num_tickets : ℕ) (total_cost : ℕ) (h1 : num_tickets = 7) (h2 : total_cost = 308) :
  total_cost / num_tickets = 44 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_l2728_272836


namespace NUMINAMATH_CALUDE_weather_forecast_probability_l2728_272850

/-- The probability of success for a single trial -/
def p : ℝ := 0.8

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem weather_forecast_probability :
  binomial_probability n k p = 0.384 := by
  sorry

end NUMINAMATH_CALUDE_weather_forecast_probability_l2728_272850


namespace NUMINAMATH_CALUDE_diamond_three_five_l2728_272896

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 2 * y + x * y

-- Theorem statement
theorem diamond_three_five : diamond 3 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_five_l2728_272896


namespace NUMINAMATH_CALUDE_jovana_shells_problem_l2728_272861

/-- Given that Jovana has an initial amount of shells and needs a total amount to fill her bucket,
    this function calculates the additional amount of shells needed. -/
def additional_shells_needed (initial_amount total_amount : ℕ) : ℕ :=
  total_amount - initial_amount

/-- Theorem stating that Jovana needs to add 12 more pounds of shells to fill her bucket. -/
theorem jovana_shells_problem :
  let initial_amount : ℕ := 5
  let total_amount : ℕ := 17
  additional_shells_needed initial_amount total_amount = 12 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_problem_l2728_272861


namespace NUMINAMATH_CALUDE_log_f_geq_one_f_geq_a_iff_a_leq_one_l2728_272888

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Part 1: Prove that log f(x) ≥ 1 when a = -8
theorem log_f_geq_one (x : ℝ) : Real.log (f (-8) x) ≥ 1 := by
  sorry

-- Part 2: Prove that f(x) ≥ a for all x ∈ ℝ if and only if a ≤ 1
theorem f_geq_a_iff_a_leq_one :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_log_f_geq_one_f_geq_a_iff_a_leq_one_l2728_272888


namespace NUMINAMATH_CALUDE_f_properties_l2728_272866

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

theorem f_properties :
  (∃ (x : ℝ), f x = Real.sqrt 2 ∧ ∀ (y : ℝ), f y ≤ Real.sqrt 2) ∧
  (∀ (θ : ℝ), f θ = 3/5 → Real.cos (2 * (π/4 - 2*θ)) = 16/25) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2728_272866


namespace NUMINAMATH_CALUDE_peanut_ratio_l2728_272830

theorem peanut_ratio (initial : ℕ) (eaten_by_bonita : ℕ) (remaining : ℕ)
  (h1 : initial = 148)
  (h2 : eaten_by_bonita = 29)
  (h3 : remaining = 82) :
  (initial - remaining - eaten_by_bonita) / initial = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_peanut_ratio_l2728_272830


namespace NUMINAMATH_CALUDE_b_97_mod_36_l2728_272863

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_97_mod_36 : b 97 % 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_97_mod_36_l2728_272863


namespace NUMINAMATH_CALUDE_system_solution_existence_l2728_272834

/-- Given a system of equations:
    1. y = b - x²
    2. x² + y² + 2a² = 4 - 2a(x + y)
    This theorem states the condition on b for the existence of at least one solution (x, y)
    for some real number a. -/
theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
sorry


end NUMINAMATH_CALUDE_system_solution_existence_l2728_272834


namespace NUMINAMATH_CALUDE_line_perp_plane_properties_l2728_272878

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  line_perp_line : Line → Line → Prop

-- Define the theorem
theorem line_perp_plane_properties {S : Space3D} (a : S.Line) (M : S.Plane) :
  (S.perpendicular a M → ∀ (l : S.Line), S.line_in_plane l M → S.line_perp_line a l) ∧
  (∃ (b : S.Line) (N : S.Plane), (∀ (l : S.Line), S.line_in_plane l N → S.line_perp_line b l) ∧ ¬S.perpendicular b N) :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_properties_l2728_272878


namespace NUMINAMATH_CALUDE_geometric_seq_property_P_iff_q_range_l2728_272841

/-- Property P for a finite sequence -/
def has_property_P (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 < i ∧ i < j ∧ j ≤ n → |a 1 - a i| ≤ |a 1 - a j|

/-- Geometric sequence with first term 1 and common ratio q -/
def geometric_seq (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

theorem geometric_seq_property_P_iff_q_range :
  ∀ q : ℝ, has_property_P (geometric_seq q) 10 ↔ q ∈ Set.Iic (-2) ∪ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_seq_property_P_iff_q_range_l2728_272841


namespace NUMINAMATH_CALUDE_susans_bread_profit_l2728_272885

/-- Susan's bread selling problem -/
theorem susans_bread_profit :
  let total_loaves : ℕ := 60
  let cost_per_loaf : ℚ := 1
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 2
  let evening_price : ℚ := 3/2
  let morning_fraction : ℚ := 1/3
  let afternoon_fraction : ℚ := 1/2

  let morning_sales : ℚ := morning_fraction * total_loaves * morning_price
  let afternoon_sales : ℚ := afternoon_fraction * (total_loaves - morning_fraction * total_loaves) * afternoon_price
  let evening_sales : ℚ := (total_loaves - morning_fraction * total_loaves - afternoon_fraction * (total_loaves - morning_fraction * total_loaves)) * evening_price

  let total_revenue : ℚ := morning_sales + afternoon_sales + evening_sales
  let total_cost : ℚ := total_loaves * cost_per_loaf
  let profit : ℚ := total_revenue - total_cost

  profit = 70 := by sorry

end NUMINAMATH_CALUDE_susans_bread_profit_l2728_272885


namespace NUMINAMATH_CALUDE_triangle_circle_radii_l2728_272835

theorem triangle_circle_radii (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  let r := area / s
  R = (7 * Real.sqrt 3) / 3 ∧ r = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_circle_radii_l2728_272835


namespace NUMINAMATH_CALUDE_tenth_pattern_stones_l2728_272874

def stone_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => stone_sequence n + 3 * (n + 2) - 2

theorem tenth_pattern_stones : stone_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pattern_stones_l2728_272874


namespace NUMINAMATH_CALUDE_initial_puppies_count_l2728_272848

/-- The number of puppies initially in the shelter -/
def initial_puppies : ℕ := sorry

/-- The number of additional puppies brought in -/
def additional_puppies : ℕ := 3

/-- The number of puppies adopted per day -/
def adoptions_per_day : ℕ := 3

/-- The number of days it takes for all puppies to be adopted -/
def days_to_adopt_all : ℕ := 2

/-- Theorem stating that the initial number of puppies is 3 -/
theorem initial_puppies_count : initial_puppies = 3 := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l2728_272848


namespace NUMINAMATH_CALUDE_point_distance_product_l2728_272849

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-5 - 4)^2 + (y₁ - 5)^2 = 12^2) →
  ((-5 - 4)^2 + (y₂ - 5)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -38 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l2728_272849


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2728_272889

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_a1 : a 1 = 2) (h_sum : a 1 + a 3 + a 5 = 14) :
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2728_272889


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l2728_272859

def camp_cedar (num_boys : ℕ) (num_girls : ℕ) (boy_ratio : ℕ) (girl_ratio : ℕ) : ℕ :=
  (num_boys + boy_ratio - 1) / boy_ratio + (num_girls + girl_ratio - 1) / girl_ratio

theorem camp_cedar_counselors :
  let num_boys : ℕ := 80
  let num_girls : ℕ := 6 * num_boys - 40
  let boy_ratio : ℕ := 5
  let girl_ratio : ℕ := 12
  camp_cedar num_boys num_girls boy_ratio girl_ratio = 53 := by
  sorry

#eval camp_cedar 80 (6 * 80 - 40) 5 12

end NUMINAMATH_CALUDE_camp_cedar_counselors_l2728_272859


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2728_272840

/-- 
Given that the equation (x^2)/(5-m) + (y^2)/(m+3) = 1 represents an ellipse,
prove that the range of values for m is (-3, 1) ∪ (1, 5).
-/
theorem ellipse_m_range (x y m : ℝ) : 
  (∃ x y, x^2 / (5 - m) + y^2 / (m + 3) = 1 ∧ 5 - m ≠ m + 3) → 
  m ∈ Set.Ioo (-3 : ℝ) 1 ∪ Set.Ioo 1 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2728_272840


namespace NUMINAMATH_CALUDE_book_sale_profit_l2728_272815

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l2728_272815


namespace NUMINAMATH_CALUDE_winston_gas_tank_capacity_l2728_272865

/-- Represents the gas tank of Winston's car -/
structure GasTank where
  initialGas : ℕ
  usedToStore : ℕ
  usedToDoctor : ℕ
  neededToRefill : ℕ

/-- Calculates the maximum capacity of the gas tank -/
def maxCapacity (tank : GasTank) : ℕ :=
  tank.initialGas - tank.usedToStore - tank.usedToDoctor + tank.neededToRefill

/-- Theorem stating that the maximum capacity of Winston's gas tank is 12 gallons -/
theorem winston_gas_tank_capacity :
  let tank : GasTank := {
    initialGas := 10,
    usedToStore := 6,
    usedToDoctor := 2,
    neededToRefill := 10
  }
  maxCapacity tank = 12 := by sorry

end NUMINAMATH_CALUDE_winston_gas_tank_capacity_l2728_272865


namespace NUMINAMATH_CALUDE_storage_blocks_count_l2728_272820

/-- Calculates the number of blocks needed for a rectangular storage --/
def blocksNeeded (length width height thickness : ℕ) : ℕ :=
  let totalVolume := length * width * height
  let interiorLength := length - 2 * thickness
  let interiorWidth := width - 2 * thickness
  let interiorHeight := height - thickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating the number of blocks needed for the specific storage --/
theorem storage_blocks_count :
  blocksNeeded 20 15 10 2 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_storage_blocks_count_l2728_272820


namespace NUMINAMATH_CALUDE_n_sticks_ge_n_plus_one_minos_l2728_272883

/-- An n-stick is a connected figure of n matches of length 1, placed horizontally or vertically, no two touching except at ends. -/
def NStick (n : ℕ) : Type := sorry

/-- An n-mino is a shape built by connecting n squares of side length 1 on their sides, with a path between each two squares. -/
def NMino (n : ℕ) : Type := sorry

/-- S_n is the number of n-sticks -/
def S (n : ℕ) : ℕ := sorry

/-- M_n is the number of n-minos -/
def M (n : ℕ) : ℕ := sorry

/-- For any natural number n, the number of n-sticks is greater than or equal to the number of (n+1)-minos. -/
theorem n_sticks_ge_n_plus_one_minos (n : ℕ) : S n ≥ M (n + 1) := by sorry

end NUMINAMATH_CALUDE_n_sticks_ge_n_plus_one_minos_l2728_272883


namespace NUMINAMATH_CALUDE_remainder_theorem_l2728_272839

/-- The polynomial P(z) = 4z^4 - 9z^3 + 3z^2 - 17z + 7 -/
def P (z : ℂ) : ℂ := 4 * z^4 - 9 * z^3 + 3 * z^2 - 17 * z + 7

/-- The theorem stating that the remainder of P(z) divided by (z - 2) is -23 -/
theorem remainder_theorem :
  ∃ Q : ℂ → ℂ, P = (fun z ↦ (z - 2) * Q z + (-23)) := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2728_272839


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2728_272851

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2728_272851


namespace NUMINAMATH_CALUDE_rectangle_area_l2728_272891

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2728_272891


namespace NUMINAMATH_CALUDE_tom_catches_jerry_l2728_272847

/-- Represents the figure-eight track -/
structure Track :=
  (small_loop : ℝ)
  (large_loop : ℝ)
  (h_large_double_small : large_loop = 2 * small_loop)

/-- Represents the runners -/
structure Runner :=
  (speed : ℝ)

theorem tom_catches_jerry (track : Track) (tom jerry : Runner) 
  (h1 : tom.speed = track.small_loop / 10)
  (h2 : jerry.speed = track.small_loop / 20)
  (h3 : tom.speed = 2 * jerry.speed) :
  (2 * track.large_loop) / (tom.speed - jerry.speed) = 40 := by
  sorry

#check tom_catches_jerry

end NUMINAMATH_CALUDE_tom_catches_jerry_l2728_272847


namespace NUMINAMATH_CALUDE_base_comparison_l2728_272819

theorem base_comparison (a b n : ℕ) (A_n B_n A_n_minus_1 B_n_minus_1 : ℕ) 
  (ha : a > 1) (hb : b > 1) (hn : n > 1)
  (hA : A_n > 0) (hB : B_n > 0) (hA_minus_1 : A_n_minus_1 > 0) (hB_minus_1 : B_n_minus_1 > 0)
  (hA_def : A_n = a^n + A_n_minus_1) (hB_def : B_n = b^n + B_n_minus_1) :
  (a > b) ↔ (A_n_minus_1 / A_n : ℚ) < (B_n_minus_1 / B_n : ℚ) := by
sorry

end NUMINAMATH_CALUDE_base_comparison_l2728_272819


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2728_272853

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- Square inscribed in the right triangle with vertex at right angle -/
def inscribed_square_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the right triangle with side on hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ (t.b / t.a) * y + y + (t.a / t.b) * y = t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_vertex t1 x)
  (h2 : inscribed_square_hypotenuse t2 y) :
  x / y = 39 / 51 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2728_272853


namespace NUMINAMATH_CALUDE_sum_of_digits_l2728_272832

def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def product : ℕ := num1 * num2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits : 
  thousands_digit product + units_digit product = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2728_272832


namespace NUMINAMATH_CALUDE_range_of_a_l2728_272857

open Real

theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x, g x = x^3 + a*x^2 - x + 2) →
  (∀ x > 0, 2 * f x ≤ (deriv g) x + 2) →
  a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2728_272857


namespace NUMINAMATH_CALUDE_divisible_count_equality_l2728_272881

theorem divisible_count_equality (n : Nat) : n = 56000 →
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 8 ≠ 0) (Finset.range (n + 1))).card =
  (Finset.filter (fun x => x % 8 = 0) (Finset.range (n + 1))).card := by
  sorry

end NUMINAMATH_CALUDE_divisible_count_equality_l2728_272881


namespace NUMINAMATH_CALUDE_clock_malfunction_theorem_l2728_272831

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Fin 24
  minutes : Fin 60

/-- Represents the possible changes to a digit due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease

/-- Applies the digit change to a number, wrapping around if necessary -/
def applyDigitChange (n : Fin 10) (change : DigitChange) : Fin 10 :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10

/-- Applies changes to both digits of a two-digit number -/
def applyTwoDigitChange (n : Fin 100) (tens_change : DigitChange) (units_change : DigitChange) : Fin 100 :=
  let tens := n / 10
  let units := n % 10
  (applyDigitChange tens tens_change) * 10 + (applyDigitChange units units_change)

theorem clock_malfunction_theorem (malfunctioned_time : Time) 
    (h : malfunctioned_time.hours = 9 ∧ malfunctioned_time.minutes = 9) :
    ∃ (original_time : Time) (hours_tens_change hours_units_change minutes_tens_change minutes_units_change : DigitChange),
      original_time.hours = 18 ∧
      original_time.minutes = 18 ∧
      applyTwoDigitChange original_time.hours hours_tens_change hours_units_change = malfunctioned_time.hours ∧
      applyTwoDigitChange original_time.minutes minutes_tens_change minutes_units_change = malfunctioned_time.minutes :=
by sorry

end NUMINAMATH_CALUDE_clock_malfunction_theorem_l2728_272831


namespace NUMINAMATH_CALUDE_share_of_a_l2728_272813

theorem share_of_a (total : ℚ) (a b c : ℚ) : 
  total = 200 →
  total = a + b + c →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 60 := by
  sorry

end NUMINAMATH_CALUDE_share_of_a_l2728_272813


namespace NUMINAMATH_CALUDE_oranges_picked_total_l2728_272869

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 :=
by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l2728_272869


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l2728_272837

/-- The sample capacity for a population of 36 individuals -/
def sample_capacity : ℕ := 6

theorem sample_capacity_proof :
  let total_population : ℕ := 36
  (sample_capacity ∣ total_population) ∧
  (6 ∣ sample_capacity) ∧
  ((total_population - 1) % (sample_capacity + 1) = 0) →
  sample_capacity = 6 :=
by sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l2728_272837


namespace NUMINAMATH_CALUDE_expression_defined_iff_l2728_272855

theorem expression_defined_iff (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x - 2)) / (Real.sqrt (x - 1))) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l2728_272855


namespace NUMINAMATH_CALUDE_car_distance_l2728_272871

/-- Proves that given the ratio of Amar's speed to the car's speed and the distance Amar covers,
    we can calculate the distance the car covers in kilometers. -/
theorem car_distance (amar_speed : ℝ) (car_speed : ℝ) (amar_distance : ℝ) :
  amar_speed / car_speed = 15 / 40 →
  amar_distance = 712.5 →
  ∃ (car_distance : ℝ), car_distance = 1.9 ∧ car_distance * 1000 * (amar_speed / car_speed) = amar_distance :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_l2728_272871


namespace NUMINAMATH_CALUDE_paige_catfish_l2728_272879

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := 7

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := initial_goldfish + disappeared_fish + remaining_fish - initial_goldfish

theorem paige_catfish : initial_catfish = 12 := by
  sorry

end NUMINAMATH_CALUDE_paige_catfish_l2728_272879


namespace NUMINAMATH_CALUDE_factor_x_pow_10_minus_1296_l2728_272852

theorem factor_x_pow_10_minus_1296 (x : ℝ) : x^10 - 1296 = (x^5 + 36) * (x^5 - 36) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_pow_10_minus_1296_l2728_272852


namespace NUMINAMATH_CALUDE_inequalities_hold_l2728_272833

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ab ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2728_272833


namespace NUMINAMATH_CALUDE_balloon_problem_l2728_272801

-- Define the number of balloons each person has
def allan_initial : ℕ := sorry
def jake_initial : ℕ := 6
def allan_bought : ℕ := 3

-- Define the relationship between Allan's and Jake's balloons
theorem balloon_problem :
  allan_initial = 2 :=
by
  have h1 : jake_initial = (allan_initial + allan_bought) + 1 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l2728_272801


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l2728_272810

theorem harmonic_mean_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l2728_272810


namespace NUMINAMATH_CALUDE_initial_cookies_correct_l2728_272876

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 36

/-- The number of cookies Paco gave to his friend -/
def given_cookies : ℕ := 14

/-- The number of cookies Paco ate -/
def eaten_cookies : ℕ := 10

/-- The number of cookies Paco had left -/
def remaining_cookies : ℕ := 12

/-- Theorem stating that the initial number of cookies is correct -/
theorem initial_cookies_correct : 
  initial_cookies = given_cookies + eaten_cookies + remaining_cookies :=
by sorry

end NUMINAMATH_CALUDE_initial_cookies_correct_l2728_272876


namespace NUMINAMATH_CALUDE_public_library_book_count_l2728_272895

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 7092 - 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

theorem public_library_book_count :
  public_library_books = 1986 ∧
  total_books = public_library_books + school_library_books :=
by sorry

end NUMINAMATH_CALUDE_public_library_book_count_l2728_272895


namespace NUMINAMATH_CALUDE_levi_basketball_score_l2728_272808

theorem levi_basketball_score (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (goal_difference : ℕ) :
  levi_initial = 8 →
  brother_initial = 12 →
  brother_additional = 3 →
  goal_difference = 5 →
  (brother_initial + brother_additional + goal_difference) - levi_initial = 12 :=
by sorry

end NUMINAMATH_CALUDE_levi_basketball_score_l2728_272808


namespace NUMINAMATH_CALUDE_least_number_of_beads_beads_divisibility_least_beads_l2728_272812

theorem least_number_of_beads (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem beads_divisibility : 2 ∣ 840 ∧ 3 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_beads : ∃ (n : ℕ), n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ n = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_beads_beads_divisibility_least_beads_l2728_272812


namespace NUMINAMATH_CALUDE_triangle_proof_l2728_272875

theorem triangle_proof (A B C : Real) (a b c : Real) (m n : Real × Real) :
  -- Given conditions
  (A + B + C = π) →
  (m = (Real.cos B, Real.sin B)) →
  (n = (Real.cos C, -Real.sin C)) →
  (m.1 * n.1 + m.2 * n.2 = 1/2) →
  (a = 2 * Real.sqrt 3) →
  (b + c = 4) →
  -- Conclusions
  (A = 2*π/3) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l2728_272875


namespace NUMINAMATH_CALUDE_total_devices_l2728_272894

theorem total_devices (computers televisions : ℕ) : 
  computers = 32 → televisions = 66 → computers + televisions = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_devices_l2728_272894


namespace NUMINAMATH_CALUDE_range_of_a_l2728_272802

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define the relationship between p and q
def sufficient_not_necessary (p q : Prop) : Prop :=
  (¬p → ¬q) ∧ ¬(q → p)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, sufficient_not_necessary (p x) (q x a)) → a < -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2728_272802


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2728_272897

theorem complex_sum_theorem (x y u v w z : ℂ) : 
  v = 2 → 
  w = -x - u → 
  (x + y * Complex.I) + (u + v * Complex.I) + (w + z * Complex.I) = 2 * Complex.I → 
  z + y = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2728_272897


namespace NUMINAMATH_CALUDE_park_area_l2728_272899

/-- A rectangular park with width one-third of length and perimeter 72 meters has area 243 square meters -/
theorem park_area (w : ℝ) (l : ℝ) : 
  w > 0 → l > 0 → w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l2728_272899


namespace NUMINAMATH_CALUDE_f_inequality_l2728_272843

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.sqrt (x^2 + 1) + x) + x^2 - x * Real.sin x

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2728_272843


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2728_272870

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = Real.log x}
def N : Set ℝ := {x | x > 0}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2728_272870


namespace NUMINAMATH_CALUDE_baking_time_undetermined_l2728_272860

/-- Represents the cookie-making process with given information -/
structure CookieBaking where
  total_cookies : ℕ
  mixing_time : ℕ
  eaten_cookies : ℕ
  remaining_cookies : ℕ

/-- States that the baking time cannot be determined from the given information -/
theorem baking_time_undetermined (cb : CookieBaking) 
  (h1 : cb.total_cookies = 32)
  (h2 : cb.mixing_time = 24)
  (h3 : cb.eaten_cookies = 9)
  (h4 : cb.remaining_cookies = 23)
  (h5 : cb.total_cookies = cb.eaten_cookies + cb.remaining_cookies) :
  ¬ ∃ (baking_time : ℕ), baking_time = cb.mixing_time ∨ baking_time ≠ cb.mixing_time :=
by sorry


end NUMINAMATH_CALUDE_baking_time_undetermined_l2728_272860


namespace NUMINAMATH_CALUDE_max_points_tournament_l2728_272856

-- Define the number of teams
def num_teams : ℕ := 8

-- Define the number of top teams with equal points
def num_top_teams : ℕ := 4

-- Define the points for win, draw, and loss
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the function to calculate the total number of games
def total_games (n : ℕ) : ℕ := n.choose 2 * 2

-- Define the function to calculate the maximum points for top teams
def max_points_top_team (n : ℕ) (k : ℕ) : ℕ :=
  (k - 1) * 3 + (n - k) * 3 * 2

-- Theorem statement
theorem max_points_tournament :
  max_points_top_team num_teams num_top_teams = 33 :=
sorry

end NUMINAMATH_CALUDE_max_points_tournament_l2728_272856


namespace NUMINAMATH_CALUDE_mrs_hilt_travel_distance_l2728_272873

/-- Calculates the total miles traveled given the initial odometer reading and additional miles --/
def total_miles_traveled (initial_reading : ℝ) (additional_miles : ℝ) : ℝ :=
  initial_reading + additional_miles

/-- Theorem stating that the total miles traveled is 2,210.23 given the specific conditions --/
theorem mrs_hilt_travel_distance :
  total_miles_traveled 1498.76 711.47 = 2210.23 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_travel_distance_l2728_272873


namespace NUMINAMATH_CALUDE_angelinas_speed_to_gym_l2728_272828

-- Define the constants
def distance_home_to_grocery : ℝ := 200
def distance_grocery_to_gym : ℝ := 300
def time_difference : ℝ := 50

-- Define the variables
variable (v : ℝ) -- Speed from home to grocery

-- Define the theorem
theorem angelinas_speed_to_gym :
  (distance_home_to_grocery / v) - (distance_grocery_to_gym / (2 * v)) = time_difference →
  2 * v = 2 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_to_gym_l2728_272828


namespace NUMINAMATH_CALUDE_max_train_collection_l2728_272838

/-- The number of trains Max receives each year -/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains -/
def collection_years : ℕ := 5

/-- The factor by which Max's parents increase his collection -/
def parents_gift_factor : ℕ := 2

/-- The total number of trains Max has after the collection period and his parents' gift -/
def total_trains : ℕ := trains_per_year * collection_years * parents_gift_factor

theorem max_train_collection :
  total_trains = 30 := by sorry

end NUMINAMATH_CALUDE_max_train_collection_l2728_272838


namespace NUMINAMATH_CALUDE_betty_oranges_purchase_l2728_272827

/-- Represents the problem of determining how many kg of oranges Betty bought. -/
theorem betty_oranges_purchase :
  ∀ (orange_kg : ℝ) (apple_kg : ℝ) (orange_cost : ℝ) (apple_price_per_kg : ℝ),
    apple_kg = 3 →
    orange_cost = 12 →
    apple_price_per_kg = 2 →
    apple_price_per_kg * 2 = orange_cost / orange_kg →
    orange_kg = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_purchase_l2728_272827


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l2728_272807

/-- The number of widgets Nina can purchase at the original price -/
def widgets_purchased (total_money : ℕ) (original_price : ℕ) : ℕ :=
  total_money / original_price

/-- The condition that if the price is reduced by 1, Nina can buy exactly 8 widgets -/
def price_reduction_condition (original_price : ℕ) (total_money : ℕ) : Prop :=
  8 * (original_price - 1) = total_money

theorem nina_widget_purchase :
  ∀ (original_price : ℕ),
    original_price > 0 →
    price_reduction_condition original_price 24 →
    widgets_purchased 24 original_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l2728_272807


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2728_272880

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) :
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2728_272880


namespace NUMINAMATH_CALUDE_green_team_opponent_score_l2728_272814

/-- The final score of a team's opponent given the team's score and lead -/
def opponent_score (team_score : ℕ) (lead : ℕ) : ℕ :=
  team_score - lead

/-- Theorem: Given Green Team's score of 39 and lead of 29, their opponent's score is 10 -/
theorem green_team_opponent_score :
  opponent_score 39 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_green_team_opponent_score_l2728_272814


namespace NUMINAMATH_CALUDE_a_lower_bound_l2728_272893

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * x + 8 * x^3 + a^2 * Real.exp (2 * x) < 4 * x^2 + a * Real.exp x + a^3 * Real.exp (3 * x)

-- State the theorem
theorem a_lower_bound (a : ℝ) (h : inequality_condition a) : a > 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_a_lower_bound_l2728_272893


namespace NUMINAMATH_CALUDE_function_equality_condition_l2728_272822

theorem function_equality_condition (m n p q : ℝ) : 
  let f := λ x : ℝ => m * x^2 + n
  let g := λ x : ℝ => p * x + q
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p^2) = q * (1 - m) := by sorry

end NUMINAMATH_CALUDE_function_equality_condition_l2728_272822


namespace NUMINAMATH_CALUDE_total_hired_is_35_l2728_272803

/-- Represents the daily pay for heavy equipment operators -/
def heavy_equipment_pay : ℕ := 140

/-- Represents the daily pay for general laborers -/
def general_laborer_pay : ℕ := 90

/-- Represents the total payroll -/
def total_payroll : ℕ := 3950

/-- Represents the number of general laborers employed -/
def num_laborers : ℕ := 19

/-- Calculates the total number of people hired given the conditions -/
def total_hired : ℕ := 
  let num_operators := (total_payroll - general_laborer_pay * num_laborers) / heavy_equipment_pay
  num_operators + num_laborers

/-- Proves that the total number of people hired is 35 -/
theorem total_hired_is_35 : total_hired = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_hired_is_35_l2728_272803


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l2728_272846

theorem polynomial_product_equality (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l2728_272846


namespace NUMINAMATH_CALUDE_pear_apple_difference_l2728_272882

theorem pear_apple_difference :
  let red_apples : ℕ := 15
  let green_apples : ℕ := 8
  let pears : ℕ := 32
  let total_apples : ℕ := red_apples + green_apples
  pears - total_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_pear_apple_difference_l2728_272882


namespace NUMINAMATH_CALUDE_subway_security_comprehensive_l2728_272818

-- Define the type for survey options
inductive SurveyOption
| TouristSatisfaction
| SubwaySecurity
| YellowRiverFish
| LightBulbLifespan

-- Define what it means for a survey to be comprehensive
def is_comprehensive (survey : SurveyOption) : Prop :=
  match survey with
  | SurveyOption.SubwaySecurity => true
  | _ => false

-- Theorem statement
theorem subway_security_comprehensive :
  ∀ (survey : SurveyOption),
    is_comprehensive survey ↔ survey = SurveyOption.SubwaySecurity :=
by sorry

end NUMINAMATH_CALUDE_subway_security_comprehensive_l2728_272818


namespace NUMINAMATH_CALUDE_min_value_expression_l2728_272842

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_prod : x * y * z = 3 / 4) (h_sum : x + y + z = 4) :
  x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 ≥ 21/2 ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 3 / 4 ∧ x + y + z = 4 ∧
    x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2728_272842


namespace NUMINAMATH_CALUDE_total_disks_l2728_272864

/-- Represents the number of disks of each color in the bag -/
structure DiskCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The properties of the disk distribution in the bag -/
def validDiskCount (d : DiskCount) : Prop :=
  ∃ (x : ℕ),
    d.blue = 3 * x ∧
    d.yellow = 7 * x ∧
    d.green = 8 * x ∧
    d.green = d.blue + 15

/-- The theorem stating the total number of disks in the bag -/
theorem total_disks (d : DiskCount) (h : validDiskCount d) : 
  d.blue + d.yellow + d.green = 54 := by
  sorry


end NUMINAMATH_CALUDE_total_disks_l2728_272864


namespace NUMINAMATH_CALUDE_additive_inverse_of_negative_2023_l2728_272844

theorem additive_inverse_of_negative_2023 :
  ∃! x : ℝ, -2023 + x = 0 ∧ x = 2023 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_negative_2023_l2728_272844


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l2728_272884

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the right vertex M
def right_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 2 ∧ M.2 = 0 ∧ ellipse_C M.1 M.2

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2 ∧ P ≠ (2, 0)

-- Define the product of slopes condition
def slope_product (M A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - M.1)) * (B.2 / (B.1 - M.1)) = 1/4

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (hM : right_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hAB : A ≠ B) 
  (hSlope : slope_product M A B) :
  ∃ (k : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ 
             A.2 = k * (A.1 + 4) ∧ 
             B.2 = k * (B.1 + 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l2728_272884


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l2728_272858

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x → Even y → Even (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l2728_272858


namespace NUMINAMATH_CALUDE_gift_original_price_gift_price_calculation_l2728_272829

/-- The original price of a gift, given certain conditions --/
theorem gift_original_price (half_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let full_cost := 2 * half_cost
  let discounted_price := (1 - discount_rate) * full_cost / ((1 - discount_rate) * (1 + tax_rate))
  discounted_price

/-- The original price of the gift is approximately $30.50 --/
theorem gift_price_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |gift_original_price 14 0.15 0.08 - 30.50| < ε :=
sorry

end NUMINAMATH_CALUDE_gift_original_price_gift_price_calculation_l2728_272829


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2728_272845

/-- Theorem: In a triangle ABC where angle A is x degrees, angle B is 2x degrees, 
    and angle C is 45°, the value of x is 45°. -/
theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  x + 2*x + 45 = 180 → 
  x = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2728_272845


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2728_272805

/-- The distance from the center of the circle x^2 + y^2 - 2x = 0 to the line 2x + y - 1 = 0 is √5/5 -/
theorem distance_from_circle_center_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 2*x = 0
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  ∃ (center_x center_y : ℝ), 
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    (abs (2*center_x + center_y - 1) / Real.sqrt 5 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l2728_272805


namespace NUMINAMATH_CALUDE_number_problem_l2728_272872

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 10 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2728_272872


namespace NUMINAMATH_CALUDE_opposite_violet_is_blue_l2728_272825

-- Define the colors
inductive Color
  | Orange
  | Black
  | Yellow
  | Violet
  | Blue
  | Pink

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Yellow ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Pink ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

-- Define the opposite face relation
def oppositeFace (i j : Fin 6) : Prop :=
  (i = 0 ∧ j = 5) ∨ (i = 1 ∧ j = 3) ∨ (i = 2 ∧ j = 4) ∨
  (i = 3 ∧ j = 1) ∨ (i = 4 ∧ j = 2) ∨ (i = 5 ∧ j = 0)

-- Theorem statement
theorem opposite_violet_is_blue (c : Cube) :
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) →
  view1 c → view2 c → view3 c →
  ∃ i j : Fin 6, oppositeFace i j ∧ c.faces i = Color.Violet ∧ c.faces j = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_opposite_violet_is_blue_l2728_272825
