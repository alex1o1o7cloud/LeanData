import Mathlib

namespace NUMINAMATH_CALUDE_octal_12345_equals_5349_decimal_l4075_407558

/-- Conversion function from octal to decimal --/
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ (octal.length - 1 - i))) 0

/-- The octal representation of the number --/
def octal_number : List Nat := [1, 2, 3, 4, 5]

/-- Theorem stating that the octal number 12345₈ is equal to 5349 in decimal --/
theorem octal_12345_equals_5349_decimal :
  octal_to_decimal octal_number = 5349 := by
  sorry

#eval octal_to_decimal octal_number

end NUMINAMATH_CALUDE_octal_12345_equals_5349_decimal_l4075_407558


namespace NUMINAMATH_CALUDE_max_a1_value_l4075_407597

/-- A sequence of non-negative real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧
  (∀ n ≥ 2, a (n + 1) = a n - a (n - 1) + n)

theorem max_a1_value (a : ℕ → ℝ) (h : RecurrenceSequence a) (h2022 : a 2 * a 2022 = 1) :
  ∃ (max_a1 : ℝ), a 1 ≤ max_a1 ∧ max_a1 = 4051 / 2025 :=
sorry

end NUMINAMATH_CALUDE_max_a1_value_l4075_407597


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4075_407515

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 12) : 
  (x / y + y / z + z / x + 3 ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z) ∧
  (x / y + y / z + z / x + 3 = Real.sqrt x + Real.sqrt y + Real.sqrt z ↔ x = 4 ∧ y = 4 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4075_407515


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l4075_407551

/-- The number of students who suggested bacon, given the total number of students
    and the number of students who suggested mashed potatoes. -/
def students_suggested_bacon (total : ℕ) (mashed_potatoes : ℕ) : ℕ :=
  total - mashed_potatoes

/-- Theorem stating that the number of students who suggested bacon is 125,
    given the total number of students and those who suggested mashed potatoes. -/
theorem bacon_suggestion_count :
  students_suggested_bacon 310 185 = 125 := by
  sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l4075_407551


namespace NUMINAMATH_CALUDE_soccer_team_size_l4075_407537

theorem soccer_team_size (total_goals : ℕ) (games_played : ℕ) (goals_other_players : ℕ) :
  total_goals = 150 →
  games_played = 15 →
  goals_other_players = 30 →
  ∃ (team_size : ℕ),
    team_size > 0 ∧
    (team_size / 3 : ℚ) * games_played + goals_other_players = total_goals ∧
    team_size = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_size_l4075_407537


namespace NUMINAMATH_CALUDE_work_completion_time_l4075_407529

theorem work_completion_time (a_days b_days : ℝ) (ha : a_days > 0) (hb : b_days > 0) :
  a_days = 60 → b_days = 20 → (a_days⁻¹ + b_days⁻¹)⁻¹ = 15 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4075_407529


namespace NUMINAMATH_CALUDE_typing_orders_count_l4075_407504

/-- The number of letters to be typed during the day -/
def total_letters : ℕ := 12

/-- The set of letter numbers that have already been typed -/
def typed_letters : Finset ℕ := {10, 12}

/-- The set of letter numbers that could potentially be in the in-box -/
def potential_inbox : Finset ℕ := Finset.range 10 ∪ {11}

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_typing_orders : ℕ :=
  (Finset.powerset potential_inbox).sum (fun s => s.card + 1)

/-- The main theorem stating the number of possible typing orders -/
theorem typing_orders_count : possible_typing_orders = 6144 := by
  sorry

end NUMINAMATH_CALUDE_typing_orders_count_l4075_407504


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l4075_407583

theorem margin_in_terms_of_selling_price (n : ℕ) (C S M : ℝ) 
  (h_n : n > 0)
  (h_margin : M = (1/2) * (S - (1/n) * C))
  (h_cost : C = S - M) :
  M = ((n - 1) / (2 * n - 1)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l4075_407583


namespace NUMINAMATH_CALUDE_power_product_specific_calculation_l4075_407508

-- Define the power function for rational numbers
def rat_pow (a : ℚ) (n : ℕ) : ℚ := a ^ n

-- Theorem 1: For any rational numbers a and b, and positive integer n, (ab)^n = a^n * b^n
theorem power_product (a b : ℚ) (n : ℕ+) : rat_pow (a * b) n = rat_pow a n * rat_pow b n := by
  sorry

-- Theorem 2: (3/2)^2019 * (-2/3)^2019 = -1
theorem specific_calculation : rat_pow (3/2) 2019 * rat_pow (-2/3) 2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_specific_calculation_l4075_407508


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l4075_407598

theorem simplify_sqrt_sum : Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l4075_407598


namespace NUMINAMATH_CALUDE_line_through_points_l4075_407561

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

/-- The statement of the problem -/
theorem line_through_points :
  ∃ (s : Finset ℤ), s.card = 4 ∧
    (∀ m ∈ s, m > 0) ∧
    (∀ m ∈ s, ∃ k : ℤ, k > 0 ∧
      Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) ∧
    (∀ m : ℤ, m > 0 →
      (∃ k : ℤ, k > 0 ∧
        Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) →
      m ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4075_407561


namespace NUMINAMATH_CALUDE_right_to_left_grouping_equivalence_l4075_407568

/-- Represents the right-to-left grouping rule for arithmetic expressions -/
def rightToLeftEval (a b c d : ℝ) : ℝ := a - b * (c + d)

/-- Represents the ordinary algebraic notation -/
def ordinaryAlgebraic (a b c d : ℝ) : ℝ := a - b * (c + d)

theorem right_to_left_grouping_equivalence (a b c d : ℝ) :
  rightToLeftEval a b c d = ordinaryAlgebraic a b c d :=
by sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_equivalence_l4075_407568


namespace NUMINAMATH_CALUDE_bobby_deadlift_difference_l4075_407546

/-- Given Bobby's initial deadlift and yearly increase, prove the difference between his deadlift at 18 and 250% of his deadlift at 13 -/
theorem bobby_deadlift_difference (initial_deadlift : ℕ) (yearly_increase : ℕ) (years : ℕ) : 
  initial_deadlift = 300 →
  yearly_increase = 110 →
  years = 5 →
  (initial_deadlift + years * yearly_increase) - (initial_deadlift * 250 / 100) = 100 := by
sorry

end NUMINAMATH_CALUDE_bobby_deadlift_difference_l4075_407546


namespace NUMINAMATH_CALUDE_inequality_proof_l4075_407589

theorem inequality_proof (x₁ x₂ y₁ y₂ : ℝ) (h : x₁^2 + x₂^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ - 1)^2 ≥ (x₁^2 + x₂^2 - 1)*(y₁^2 + y₂^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4075_407589


namespace NUMINAMATH_CALUDE_expression_evaluation_l4075_407570

theorem expression_evaluation (a b : ℚ) (h1 : a = 1) (h2 : b = 1/2) :
  a * (a - 2*b) + (a + b) * (a - b) + (a - b)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4075_407570


namespace NUMINAMATH_CALUDE_min_x_coeff_for_restricted_poly_with_specific_value_l4075_407506

/-- A polynomial with coefficients from the set {0,1,2,3,4,5} -/
def RestrictedPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, (P.coeff i) ∈ ({0, 1, 2, 3, 4, 5} : Set ℤ)

/-- The theorem stating that if P(6) = 2013 for a restricted polynomial P,
    then the coefficient of x in P is at least 5 -/
theorem min_x_coeff_for_restricted_poly_with_specific_value
  (P : Polynomial ℤ) (h : RestrictedPolynomial P) (h2 : P.eval 6 = 2013) :
  P.coeff 1 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_x_coeff_for_restricted_poly_with_specific_value_l4075_407506


namespace NUMINAMATH_CALUDE_simplify_fraction_l4075_407545

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4075_407545


namespace NUMINAMATH_CALUDE_saturday_visitors_200_l4075_407512

/-- Calculates the number of visitors on Saturday given the ticket price, 
    weekday visitors, Sunday visitors, and total revenue -/
def visitors_on_saturday (ticket_price : ℕ) (weekday_visitors : ℕ) 
  (sunday_visitors : ℕ) (total_revenue : ℕ) : ℕ :=
  (total_revenue / ticket_price) - (5 * weekday_visitors) - sunday_visitors

/-- Proves that the number of visitors on Saturday is 200 given the specified conditions -/
theorem saturday_visitors_200 : 
  visitors_on_saturday 3 100 300 3000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_saturday_visitors_200_l4075_407512


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l4075_407523

/-- The quadratic function f(x) = -2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := -2 * (x - 3)^2 + 1

/-- The axis of symmetry of f(x) -/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The axis of symmetry of f(x) = -2(x-3)^2 + 1 is x = 3 -/
theorem axis_of_symmetry_is_correct :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l4075_407523


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4075_407541

theorem sum_of_coefficients (a b c : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 21*x + 108 = (x - b) * (x - c)) →
  a + b + c = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4075_407541


namespace NUMINAMATH_CALUDE_parabola_c_value_l4075_407547

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_c_value :
  ∀ p : Parabola,
  p.vertex = (3, -5) →
  p.contains 0 6 →
  p.c = 288 / 121 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l4075_407547


namespace NUMINAMATH_CALUDE_product_polynomial_sum_l4075_407591

theorem product_polynomial_sum (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 5) = 
   21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) →
  g + h = -28/9 := by
sorry

end NUMINAMATH_CALUDE_product_polynomial_sum_l4075_407591


namespace NUMINAMATH_CALUDE_tenth_term_is_512_l4075_407525

/-- A sequence where each term is twice the previous term, starting with 1 -/
def doubling_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * doubling_sequence n

/-- The 10th term of the doubling sequence is 512 -/
theorem tenth_term_is_512 : doubling_sequence 9 = 512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_512_l4075_407525


namespace NUMINAMATH_CALUDE_son_age_is_22_l4075_407553

/-- Given a man and his son, where:
    1. The man is 24 years older than his son
    2. In two years, the man's age will be twice the age of his son
    This theorem proves that the present age of the son is 22 years. -/
theorem son_age_is_22 (man_age son_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_22_l4075_407553


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l4075_407534

theorem ratio_sum_difference (a b : ℝ) (h1 : a / b = 3 / 8) (h2 : a + b = 44) : b - a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l4075_407534


namespace NUMINAMATH_CALUDE_and_or_relationship_l4075_407550

theorem and_or_relationship (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_or_relationship_l4075_407550


namespace NUMINAMATH_CALUDE_tan_inequality_l4075_407594

open Real

theorem tan_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π/2) 
  (h₂ : 0 < x₂ ∧ x₂ < π/2) 
  (h₃ : x₁ ≠ x₂) : 
  (1/2) * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l4075_407594


namespace NUMINAMATH_CALUDE_coffee_expense_l4075_407514

theorem coffee_expense (items_per_day : ℕ) (cost_per_item : ℕ) (days : ℕ) :
  items_per_day = 2 →
  cost_per_item = 2 →
  days = 30 →
  items_per_day * cost_per_item * days = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_expense_l4075_407514


namespace NUMINAMATH_CALUDE_sanitizer_sprays_effectiveness_l4075_407516

theorem sanitizer_sprays_effectiveness (spray1_kill_rate spray2_kill_rate overlap_rate remaining_rate : Real) :
  spray1_kill_rate = 0.5 →
  overlap_rate = 0.05 →
  remaining_rate = 0.3 →
  1 - (spray1_kill_rate + spray2_kill_rate - overlap_rate) = remaining_rate →
  spray2_kill_rate = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_sprays_effectiveness_l4075_407516


namespace NUMINAMATH_CALUDE_subtraction_division_result_l4075_407566

theorem subtraction_division_result : 1.85 - 1.85 / 1.85 = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_result_l4075_407566


namespace NUMINAMATH_CALUDE_committee_count_theorem_l4075_407576

/-- The number of ways to choose a committee with at least one female member -/
def committee_count (total_members : ℕ) (committee_size : ℕ) (female_members : ℕ) : ℕ :=
  Nat.choose total_members committee_size - Nat.choose (total_members - female_members) committee_size

theorem committee_count_theorem :
  committee_count 30 5 12 = 133938 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_theorem_l4075_407576


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l4075_407552

/-- 
Given a convex pentagon with interior angles measuring x+1, 2x, 3x, 4x, and 5x-1 degrees,
where x is a positive real number and the sum of these angles is 540 degrees,
prove that the measure of the largest angle is 179 degrees.
-/
theorem largest_angle_convex_pentagon (x : ℝ) 
  (h_positive : x > 0)
  (h_sum : (x + 1) + 2*x + 3*x + 4*x + (5*x - 1) = 540) :
  max (x + 1) (max (2*x) (max (3*x) (max (4*x) (5*x - 1)))) = 179 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l4075_407552


namespace NUMINAMATH_CALUDE_charles_reading_days_l4075_407574

/-- Represents the number of pages Charles reads each day -/
def daily_pages : List Nat := [7, 12, 10, 6]

/-- The total number of pages in the book -/
def total_pages : Nat := 96

/-- Calculates the number of days needed to finish the book -/
def days_to_finish (pages : List Nat) (total : Nat) : Nat :=
  let pages_read := pages.sum
  let remaining := total - pages_read
  let weekdays := pages.length
  let average_daily := (pages_read + remaining - 1) / weekdays
  weekdays + (remaining + average_daily - 1) / average_daily

theorem charles_reading_days :
  days_to_finish daily_pages total_pages = 11 := by
  sorry

#eval days_to_finish daily_pages total_pages

end NUMINAMATH_CALUDE_charles_reading_days_l4075_407574


namespace NUMINAMATH_CALUDE_domain_of_g_l4075_407572

-- Define the domain of f
def DomainF : Set ℝ := Set.Icc (-8) 4

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

-- Theorem statement
theorem domain_of_g (f : ℝ → ℝ) (hf : Set.MapsTo f DomainF (Set.range f)) :
  {x : ℝ | g f x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l4075_407572


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l4075_407503

theorem exactly_one_correct_probability 
  (prob_a : ℝ) 
  (prob_b : ℝ) 
  (h_prob_a : prob_a = 0.7) 
  (h_prob_b : prob_b = 0.8) : 
  prob_a * (1 - prob_b) + (1 - prob_a) * prob_b = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l4075_407503


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l4075_407569

theorem simplify_sqrt_expression : 
  (Real.sqrt 448 / Real.sqrt 32) - (Real.sqrt 245 / Real.sqrt 49) = Real.sqrt 2 * Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l4075_407569


namespace NUMINAMATH_CALUDE_tangent_point_circle_properties_l4075_407578

/-- The equation of a circle with center (x₀, y₀) and radius r -/
def circle_equation (x₀ y₀ r : ℝ) (x y : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The equation of the line 2x + y = 0 -/
def line_center (x y : ℝ) : Prop :=
  2 * x + y = 0

/-- The equation of the line x + y - 1 = 0 -/
def line_tangent (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point (2, -1) lies on the tangent line -/
theorem tangent_point : line_tangent 2 (-1) := by sorry

theorem circle_properties (x₀ y₀ r : ℝ) :
  line_center x₀ y₀ →
  (∀ x y, circle_equation x₀ y₀ r x y ↔ (x - 1)^2 + (y + 2)^2 = 2) →
  (∃ x y, circle_equation x₀ y₀ r x y ∧ line_tangent x y) →
  circle_equation x₀ y₀ r 2 (-1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_circle_properties_l4075_407578


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l4075_407518

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union_A_B : (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l4075_407518


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4075_407513

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y < x → ¬(37 ∣ (157639 + y))) ∧
  (37 ∣ (157639 + x)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4075_407513


namespace NUMINAMATH_CALUDE_f_inequality_l4075_407571

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition for f
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem f_inequality (h : strictly_increasing f) : f (-2) < f 1 ∧ f 1 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l4075_407571


namespace NUMINAMATH_CALUDE_sum_of_squares_l4075_407521

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x * x * y + x * y * y = 1056) :
  x * x + y * y = 458 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4075_407521


namespace NUMINAMATH_CALUDE_area_of_region_l4075_407596

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 20 ∧ 
   A = Real.pi * (Real.sqrt ((x + 5)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 10*x - 4*y + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l4075_407596


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l4075_407563

theorem arithmetic_expression_evaluation : 8 / 2 - (3 - 5 + 7) + 3 * 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l4075_407563


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4075_407582

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (x - 1) / (x + 2) ≤ 0 → -2 ≤ x ∧ x ≤ 1) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ (x - 1) / (x + 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4075_407582


namespace NUMINAMATH_CALUDE_same_name_pair_exists_l4075_407585

theorem same_name_pair_exists (n : ℕ) (h_n : n = 33) :
  ∀ (first_name_groups last_name_groups : Fin n → Fin 11),
    (∀ i : Fin 11, ∃ j : Fin n, first_name_groups j = i) →
    (∀ i : Fin 11, ∃ j : Fin n, last_name_groups j = i) →
    ∃ x y : Fin n, x ≠ y ∧ first_name_groups x = first_name_groups y ∧ last_name_groups x = last_name_groups y :=
by
  sorry

#check same_name_pair_exists

end NUMINAMATH_CALUDE_same_name_pair_exists_l4075_407585


namespace NUMINAMATH_CALUDE_max_value_x_1_minus_3x_l4075_407562

theorem max_value_x_1_minus_3x (x : ℝ) (h : 0 < x ∧ x < 1/3) :
  ∃ (max : ℝ), max = 1/12 ∧ ∀ y, 0 < y ∧ y < 1/3 → x * (1 - 3*x) ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_1_minus_3x_l4075_407562


namespace NUMINAMATH_CALUDE_ben_win_probability_l4075_407584

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3/7) 
  (h2 : ∀ (tie_prob : ℚ), tie_prob = 0) : 
  1 - lose_prob = 4/7 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l4075_407584


namespace NUMINAMATH_CALUDE_max_consecutive_new_numbers_l4075_407555

def is_new (n : Nat) : Prop :=
  n > 5 ∧ ∃ m : Nat, (∀ k < n, m % k = 0) ∧ m % n ≠ 0

theorem max_consecutive_new_numbers :
  ∃ a : Nat, a > 5 ∧
    is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧
    ¬(is_new (a - 1) ∧ is_new a ∧ is_new (a + 1) ∧ is_new (a + 2)) ∧
    ¬(is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧ is_new (a + 3)) :=
  sorry

end NUMINAMATH_CALUDE_max_consecutive_new_numbers_l4075_407555


namespace NUMINAMATH_CALUDE_most_tickets_have_four_hits_l4075_407556

/-- Number of matches in a lottery ticket -/
def num_matches : ℕ := 13

/-- Number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 3

/-- Number of tickets with k correct predictions -/
def tickets_with_k_hits (k : ℕ) : ℕ :=
  (num_matches.choose k) * (outcomes_per_match - 1)^(num_matches - k)

/-- The number of correct predictions that maximizes the number of tickets -/
def max_hits : ℕ := 4

theorem most_tickets_have_four_hits :
  ∀ k : ℕ, k ≤ num_matches → k ≠ max_hits →
    tickets_with_k_hits k ≤ tickets_with_k_hits max_hits :=
by sorry

end NUMINAMATH_CALUDE_most_tickets_have_four_hits_l4075_407556


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l4075_407538

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (monotone_decreasing (f a)) ↔ (1/2 ≤ a ∧ a ≤ 5/8) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l4075_407538


namespace NUMINAMATH_CALUDE_jordan_max_points_l4075_407517

structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  freeThrowAttempts : ℕ
  threePointSuccess : ℚ
  twoPointSuccess : ℚ
  freeThrowSuccess : ℚ

def totalShots (game : BasketballGame) : ℕ :=
  game.threePointAttempts + game.twoPointAttempts + game.freeThrowAttempts

def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccess * game.threePointAttempts +
  2 * game.twoPointSuccess * game.twoPointAttempts +
  game.freeThrowSuccess * game.freeThrowAttempts

theorem jordan_max_points :
  ∀ (game : BasketballGame),
  game.threePointSuccess = 1/4 →
  game.twoPointSuccess = 2/5 →
  game.freeThrowSuccess = 4/5 →
  totalShots game = 50 →
  totalPoints game ≤ 39 :=
by sorry

end NUMINAMATH_CALUDE_jordan_max_points_l4075_407517


namespace NUMINAMATH_CALUDE_negation_of_existence_l4075_407526

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l4075_407526


namespace NUMINAMATH_CALUDE_diameter_angle_property_l4075_407520

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a function to check if a point is inside a circle
def isInside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define a function to check if a point is outside a circle
def isOutside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

-- Define a function to calculate the angle between three points
noncomputable def angle (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem diameter_angle_property (c : Circle) (a b : Point) :
  (∀ (x : ℝ × ℝ), x.1 = a.1 ∧ x.2 = b.2 → isInside c x) →  -- a and b are on opposite sides of the circle
  (∀ (p : Point), isInside c p → angle a p b > Real.pi / 2) ∧
  (∀ (p : Point), isOutside c p → angle a p b < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_diameter_angle_property_l4075_407520


namespace NUMINAMATH_CALUDE_a_b_product_l4075_407577

def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 2 * a n / (1 + (a n)^2)

def b : ℕ → ℚ
  | 0 => 4
  | n + 1 => b n ^ 2 - 2 * b n + 2

def b_product : ℕ → ℚ
  | 0 => b 0
  | n + 1 => b_product n * b (n + 1)

theorem a_b_product (n : ℕ) : a (n + 1) * b (n + 1) = 2 * b_product n := by
  sorry

end NUMINAMATH_CALUDE_a_b_product_l4075_407577


namespace NUMINAMATH_CALUDE_min_b_value_l4075_407575

noncomputable section

variables (a b : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

def g (x : ℝ) : ℝ := x^2 - 2 * b * x + 4 / 3

theorem min_b_value (h1 : a = 1/3) 
  (h2 : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 3, f x₁ ≥ g x₂) :
  b ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_b_value_l4075_407575


namespace NUMINAMATH_CALUDE_rihannas_initial_money_l4075_407559

/-- Represents the shopping scenario and calculates Rihanna's initial money --/
def rihannas_shopping (mango_price apple_juice_price : ℕ) (mango_quantity apple_juice_quantity : ℕ) (money_left : ℕ) : ℕ :=
  let total_cost := mango_price * mango_quantity + apple_juice_price * apple_juice_quantity
  total_cost + money_left

/-- Theorem stating that Rihanna's initial money was $50 --/
theorem rihannas_initial_money :
  rihannas_shopping 3 3 6 6 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rihannas_initial_money_l4075_407559


namespace NUMINAMATH_CALUDE_swimming_difference_l4075_407548

theorem swimming_difference (camden_total : ℕ) (susannah_total : ℕ) (weeks : ℕ) : 
  camden_total = 16 → susannah_total = 24 → weeks = 4 →
  (susannah_total / weeks) - (camden_total / weeks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_difference_l4075_407548


namespace NUMINAMATH_CALUDE_meat_market_sales_l4075_407599

theorem meat_market_sales (thursday_sales : ℝ) : 
  (2 * thursday_sales) + thursday_sales + 130 + (130 / 2) = 500 + 325 → 
  thursday_sales = 210 := by
sorry

end NUMINAMATH_CALUDE_meat_market_sales_l4075_407599


namespace NUMINAMATH_CALUDE_parabola_sum_l4075_407505

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (-1, -2) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (x + 3)^2 = (y - 2) / a) ∧
  a * (-1)^2 + b * (-1) + c = -2

theorem parabola_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = -14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l4075_407505


namespace NUMINAMATH_CALUDE_a_in_closed_unit_interval_l4075_407586

def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

theorem a_in_closed_unit_interval (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_a_in_closed_unit_interval_l4075_407586


namespace NUMINAMATH_CALUDE_clock_angle_at_3_15_l4075_407510

/-- The angle between clock hands at 3:15 -/
theorem clock_angle_at_3_15 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute_for_minute_hand : ℝ := 6
  let degrees_per_minute_for_hour_hand : ℝ := 0.5
  let minute_hand_position : ℝ := 15 * degrees_per_minute_for_minute_hand
  let hour_hand_position : ℝ := 3 * degrees_per_hour + 15 * degrees_per_minute_for_hour_hand
  let angle_between_hands : ℝ := |hour_hand_position - minute_hand_position|
  angle_between_hands = 7.5 := by
sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_15_l4075_407510


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4075_407501

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4075_407501


namespace NUMINAMATH_CALUDE_average_of_five_integers_l4075_407535

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 42 → 
  r ≤ 17 → 
  (k + m + r + s + t : ℚ) / 5 = 266 / 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_integers_l4075_407535


namespace NUMINAMATH_CALUDE_ball_probability_l4075_407592

theorem ball_probability (n m : ℕ) : 
  n > 0 ∧ m ≤ n ∧
  (1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5) ∧
  (6 * (m : ℚ) / (n : ℚ) = 4) →
  ((n - m - 1 : ℚ) / (n - 1 : ℚ) = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_probability_l4075_407592


namespace NUMINAMATH_CALUDE_aristocrat_spending_l4075_407588

theorem aristocrat_spending (total_people : ℕ) (men_amount : ℕ) (women_amount : ℕ)
  (men_fraction : ℚ) (women_fraction : ℚ) :
  total_people = 3552 →
  men_amount = 45 →
  women_amount = 60 →
  men_fraction = 1/9 →
  women_fraction = 1/12 →
  ∃ (men women : ℕ),
    men + women = total_people ∧
    (men_fraction * men * men_amount + women_fraction * women * women_amount : ℚ) = 17760 :=
by sorry

end NUMINAMATH_CALUDE_aristocrat_spending_l4075_407588


namespace NUMINAMATH_CALUDE_intersection_volume_l4075_407502

-- Define the two cubes
def cube1 (x y z : ℝ) : Prop := max (|x|) (max |y| |z|) ≤ 1
def cube2 (x y z : ℝ) : Prop := max (|x-1|) (max |y-1| |z-1|) ≤ 1

-- Define the intersection of the two cubes
def intersection (x y z : ℝ) : Prop := cube1 x y z ∧ cube2 x y z

-- Define the volume of a region
noncomputable def volume (region : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem intersection_volume : volume intersection = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_l4075_407502


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4075_407564

/-- Theorem: Eccentricity of a hyperbola with specific properties --/
theorem hyperbola_eccentricity (a b c : ℝ) (h : c^2 = a^2 + b^2) : 
  let f1 : ℝ × ℝ := (-c, 0)
  let f2 : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (c, b^2 / a)
  let B : ℝ × ℝ := (c, -b^2 / a)
  let G : ℝ × ℝ := (c / 3, 0)
  ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    (G.1 - A.1) * (f1.1 - B.1) + (G.2 - A.2) * (f1.2 - B.2) = 0 →
    c / a = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4075_407564


namespace NUMINAMATH_CALUDE_fraction_equality_l4075_407587

theorem fraction_equality (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4075_407587


namespace NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l4075_407524

def base_five_to_decimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

theorem base_conversion_256_to_base_5 :
  base_five_to_decimal 2 0 1 1 = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l4075_407524


namespace NUMINAMATH_CALUDE_exact_two_fours_probability_l4075_407573

-- Define the number of dice
def num_dice : ℕ := 15

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the target number we're looking for
def target_number : ℕ := 4

-- Define the number of dice we want to show the target number
def target_count : ℕ := 2

-- Define the probability of rolling the target number on a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of not rolling the target number on a single die
def single_prob_complement : ℚ := 1 - single_prob

-- Theorem statement
theorem exact_two_fours_probability :
  (Nat.choose num_dice target_count : ℚ) * single_prob ^ target_count * single_prob_complement ^ (num_dice - target_count) =
  (105 : ℚ) * 5^13 / 6^15 := by
  sorry

end NUMINAMATH_CALUDE_exact_two_fours_probability_l4075_407573


namespace NUMINAMATH_CALUDE_orange_selling_loss_l4075_407540

def total_money : ℚ := 75
def ratio_sum : ℕ := 4 + 5 + 6
def cara_ratio : ℕ := 4
def janet_ratio : ℕ := 5
def selling_percentage : ℚ := 80 / 100

theorem orange_selling_loss :
  let cara_money := (cara_ratio : ℚ) / ratio_sum * total_money
  let janet_money := (janet_ratio : ℚ) / ratio_sum * total_money
  let combined_money := cara_money + janet_money
  let selling_price := selling_percentage * combined_money
  combined_money - selling_price = 9 := by sorry

end NUMINAMATH_CALUDE_orange_selling_loss_l4075_407540


namespace NUMINAMATH_CALUDE_package_volume_calculation_l4075_407544

/-- Proves that the total volume needed to package the collection is 3,060,000 cubic inches -/
theorem package_volume_calculation (box_length box_width box_height : ℕ) 
  (cost_per_box total_cost : ℚ) : 
  box_length = 20 →
  box_width = 20 →
  box_height = 15 →
  cost_per_box = 7/10 →
  total_cost = 357 →
  (box_length * box_width * box_height) * (total_cost / cost_per_box) = 3060000 :=
by sorry

end NUMINAMATH_CALUDE_package_volume_calculation_l4075_407544


namespace NUMINAMATH_CALUDE_fraction_simplification_l4075_407580

theorem fraction_simplification (x : ℝ) (h : x ≠ 2) :
  x / (x - 2) + 2 / (2 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4075_407580


namespace NUMINAMATH_CALUDE_f_sum_eq_two_l4075_407543

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_eq_two :
  let f' := deriv f
  f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_eq_two_l4075_407543


namespace NUMINAMATH_CALUDE_blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l4075_407507

/-- Represents a cell on the infinite grid --/
structure Cell :=
  (x : Int) (y : Int)

/-- Represents the color of a cell --/
inductive Color
  | White
  | Blue
  | Red

/-- Represents the game state --/
structure GameState :=
  (grid : Cell → Color)

/-- Blake's score is the size of the largest blue simple polygon --/
def blakeScore (state : GameState) : Nat :=
  sorry

/-- Blake's strategy to color adjacent cells --/
def blakeStrategy (state : GameState) : Cell :=
  sorry

/-- Ruby's strategy to block Blake --/
def rubyStrategy (state : GameState) : Cell × Cell :=
  sorry

/-- The game play function --/
def playGame (initialState : GameState) : Nat :=
  sorry

theorem blake_guarantee_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ∃ (finalState : GameState),
      blakeScore finalState ≥ 4 :=
sorry

theorem ruby_prevent_more_than_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ¬∃ (finalState : GameState),
      blakeScore finalState > 4 :=
sorry

theorem largest_guaranteed_score :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    (∃ (finalState : GameState), blakeScore finalState = 4) ∧
    (¬∃ (finalState : GameState), blakeScore finalState > 4) :=
sorry

end NUMINAMATH_CALUDE_blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l4075_407507


namespace NUMINAMATH_CALUDE_warehouse_optimization_l4075_407581

/-- Represents the warehouse dimensions and costs -/
structure Warehouse where
  budget : ℝ
  frontCost : ℝ
  sideCost : ℝ
  roofCost : ℝ

/-- Calculates the total cost of the warehouse -/
def totalCost (w : Warehouse) (length width : ℝ) : ℝ :=
  w.frontCost * length + 2 * w.sideCost * width + w.roofCost * length * width

/-- Theorem stating the maximum area and optimal length for the warehouse -/
theorem warehouse_optimization (w : Warehouse) 
    (h1 : w.budget = 3200)
    (h2 : w.frontCost = 40)
    (h3 : w.sideCost = 45)
    (h4 : w.roofCost = 20) :
    ∃ (maxArea optLength : ℝ),
      maxArea = 100 ∧
      optLength = 15 ∧
      totalCost w optLength (maxArea / optLength) ≤ w.budget ∧
      ∀ (length width : ℝ), 
        length > 0 → width > 0 → 
        totalCost w length width ≤ w.budget → 
        length * width ≤ maxArea := by
  sorry

end NUMINAMATH_CALUDE_warehouse_optimization_l4075_407581


namespace NUMINAMATH_CALUDE_attendance_scientific_notation_l4075_407549

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem attendance_scientific_notation :
  toScientificNotation 204000 = ScientificNotation.mk 2.04 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_attendance_scientific_notation_l4075_407549


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l4075_407565

theorem addition_preserves_inequality (a b c d : ℝ) : 
  a < b → c < d → a + c < b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l4075_407565


namespace NUMINAMATH_CALUDE_dollar_squared_diff_zero_l4075_407530

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_squared_diff_zero (x y : ℝ) : dollar ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_squared_diff_zero_l4075_407530


namespace NUMINAMATH_CALUDE_unique_sums_count_l4075_407595

def bagA : Finset ℕ := {2, 3, 4}
def bagB : Finset ℕ := {3, 4, 5}

def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem unique_sums_count : possibleSums.card = 5 := by sorry

end NUMINAMATH_CALUDE_unique_sums_count_l4075_407595


namespace NUMINAMATH_CALUDE_plain_calculations_l4075_407590

/-- Given information about two plains A and B, prove various calculations about their areas, populations, and elevation difference. -/
theorem plain_calculations (area_B area_A : ℝ) (pop_density_A pop_density_B : ℝ) 
  (distance_AB : ℝ) (elevation_gradient : ℝ) :
  area_B = 200 →
  area_A = area_B - 50 →
  pop_density_A = 50 →
  pop_density_B = 75 →
  distance_AB = 25 →
  elevation_gradient = 500 / 10 →
  (area_A = 150 ∧ 
   area_A * pop_density_A + area_B * pop_density_B = 22500 ∧
   elevation_gradient * distance_AB = 125) :=
by sorry

end NUMINAMATH_CALUDE_plain_calculations_l4075_407590


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_negative_one_l4075_407579

theorem product_of_fractions_equals_negative_one 
  (a b c : ℝ) 
  (ha : a ≠ 3) 
  (hb : b ≠ 4) 
  (hc : c ≠ 5) : 
  ((a - 3) / (6 - c)) * ((b - 4) / (3 - a)) * ((c - 5) / (4 - b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_negative_one_l4075_407579


namespace NUMINAMATH_CALUDE_base_b_perfect_square_implies_b_greater_than_two_l4075_407567

/-- Represents a number in base b --/
def base_representation (b : ℕ) : ℕ := b^2 + 2*b + 1

/-- Checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem base_b_perfect_square_implies_b_greater_than_two :
  ∀ b : ℕ, is_perfect_square (base_representation b) → b > 2 :=
by sorry

end NUMINAMATH_CALUDE_base_b_perfect_square_implies_b_greater_than_two_l4075_407567


namespace NUMINAMATH_CALUDE_pet_store_dogs_l4075_407522

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:4 and there are 18 cats, there are 24 dogs -/
theorem pet_store_dogs : calculate_dogs 3 4 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l4075_407522


namespace NUMINAMATH_CALUDE_equivalent_expression_l4075_407554

theorem equivalent_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x / (1 - (x - 2) / x)) = -x / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l4075_407554


namespace NUMINAMATH_CALUDE_system_solution_l4075_407542

theorem system_solution : 
  let x : ℚ := 57 / 31
  let y : ℚ := 97 / 31
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4075_407542


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l4075_407519

/-- The number of yellow lights on a Christmas tree -/
def yellow_lights (total red blue : ℕ) : ℕ := total - (red + blue)

/-- Theorem: There are 37 yellow lights on the Christmas tree -/
theorem christmas_tree_lights : yellow_lights 95 26 32 = 37 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l4075_407519


namespace NUMINAMATH_CALUDE_cards_left_l4075_407509

/-- The number of basketball card boxes Ben has -/
def basketball_boxes : ℕ := 4

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 10

/-- The number of baseball card boxes Ben's mother gave him -/
def baseball_boxes : ℕ := 5

/-- The number of cards in each baseball box -/
def baseball_cards_per_box : ℕ := 8

/-- The number of cards Ben gave to his classmates -/
def cards_given_away : ℕ := 58

/-- Theorem stating the number of cards Ben has left -/
theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end NUMINAMATH_CALUDE_cards_left_l4075_407509


namespace NUMINAMATH_CALUDE_complex_circle_equation_l4075_407532

theorem complex_circle_equation (z : ℂ) (h : Complex.abs (z - 1) = 5) :
  ∃ (x y : ℝ), z = Complex.mk x y ∧ 
  -4 ≤ x ∧ x ≤ 6 ∧ 
  (y = Real.sqrt (25 - (x - 1)^2) ∨ y = -Real.sqrt (25 - (x - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l4075_407532


namespace NUMINAMATH_CALUDE_abc_value_l4075_407560

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hbc : b * c = 54) (hca : c * a = 45) :
  a * b * c = 270 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l4075_407560


namespace NUMINAMATH_CALUDE_fill_water_tank_days_l4075_407536

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity : ℕ) (daily_collection : ℕ) : ℕ :=
  (tank_capacity * 1000 + daily_collection - 1) / daily_collection

/-- Theorem stating that it takes 206 days to fill the water tank -/
theorem fill_water_tank_days : days_to_fill_tank 350 1700 = 206 := by
  sorry

end NUMINAMATH_CALUDE_fill_water_tank_days_l4075_407536


namespace NUMINAMATH_CALUDE_cosine_inequality_l4075_407511

theorem cosine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l4075_407511


namespace NUMINAMATH_CALUDE_projection_equality_l4075_407531

/-- Given two vectors in R^2 that project to the same vector, 
    prove that the projection is (16/5, 8/5) -/
theorem projection_equality (v : ℝ × ℝ) :
  let a : ℝ × ℝ := (5, -2)
  let b : ℝ × ℝ := (2, 4)
  let proj (x : ℝ × ℝ) := 
    let dot_prod := x.1 * v.1 + x.2 * v.2
    let v_norm_sq := v.1 * v.1 + v.2 * v.2
    ((dot_prod / v_norm_sq) * v.1, (dot_prod / v_norm_sq) * v.2)
  proj a = proj b → proj a = (16/5, 8/5) :=
by
  sorry

#check projection_equality

end NUMINAMATH_CALUDE_projection_equality_l4075_407531


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l4075_407539

theorem stock_exchange_problem (h l : ℕ) : 
  h = l + l / 5 →  -- 20% more stocks closed higher
  h = 1080 →      -- 1080 stocks closed higher
  h + l = 1980    -- Total number of stocks
  := by sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l4075_407539


namespace NUMINAMATH_CALUDE_range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l4075_407593

-- Define the propositions
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

def q (m : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 → 
  (∃ c : ℝ, c > 0 ∧ x^2 / (m^2 - c^2) + y^2 / m^2 = 1)

def s (m t : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / (m - t) + y^2 / (m - t - 1) = 1 → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1)

-- Theorem statements
theorem range_of_m_for_p_and_q :
  ∀ m : ℝ, (p m ∧ q m) ↔ ((-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

theorem range_of_t_for_q_necessary_not_sufficient_for_s :
  ∀ t : ℝ, (∀ m : ℝ, s m t → q m) ∧ (∃ m : ℝ, q m ∧ ¬s m t) ↔ 
  ((-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l4075_407593


namespace NUMINAMATH_CALUDE_library_book_redistribution_l4075_407528

theorem library_book_redistribution (total_books : Nat) (initial_stack : Nat) (new_stack : Nat)
    (h1 : total_books = 1452)
    (h2 : initial_stack = 42)
    (h3 : new_stack = 43) :
  total_books % new_stack = 33 := by
  sorry

end NUMINAMATH_CALUDE_library_book_redistribution_l4075_407528


namespace NUMINAMATH_CALUDE_no_two_obtuse_angles_l4075_407500

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_180 : angle1 + angle2 + angle3 = 180
  positive : angle1 > 0 ∧ angle2 > 0 ∧ angle3 > 0

-- Define what an obtuse angle is
def isObtuse (angle : Real) : Prop := angle > 90

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : 
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle2) ∧
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle3) ∧
  ¬(isObtuse t.angle2 ∧ isObtuse t.angle3) := by
  sorry


end NUMINAMATH_CALUDE_no_two_obtuse_angles_l4075_407500


namespace NUMINAMATH_CALUDE_house_rent_fraction_l4075_407527

theorem house_rent_fraction (salary : ℚ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℚ) :
  salary = 180000 →
  food_fraction = 1/5 →
  clothes_fraction = 3/5 →
  remaining = 18000 →
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining = salary ∧
    house_rent_fraction = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l4075_407527


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_P_is_intersection_l4075_407557

-- Define the point of intersection P
def P : ℝ × ℝ := (1, 3)

-- Define the given linear functions
def f (x : ℝ) : ℝ := -x + 4
def g (x : ℝ) : ℝ := x + 2

-- Define the reference line
def ref_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∀ x y : ℝ, (x, y) = P → (2 * x - y + 1 = 0) := by sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_equation :
  ∀ x y : ℝ, (x, y) = P → (x + 2 * y - 7 = 0) := by sorry

-- Verify that P is indeed the intersection point of f and g
theorem P_is_intersection :
  f P.1 = P.2 ∧ g P.1 = P.2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_P_is_intersection_l4075_407557


namespace NUMINAMATH_CALUDE_function_properties_l4075_407533

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * log x

theorem function_properties (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a = 1 ∧
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x₀ = -2 ∧ f a x ≥ f a x₀) → a = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4075_407533
