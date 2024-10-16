import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_3_m_range_when_f_geq_8_l1530_153038

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part I
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} := by sorry

-- Theorem for part II
theorem m_range_when_f_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ m ≤ -9 ∨ m ≥ 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_3_m_range_when_f_geq_8_l1530_153038


namespace NUMINAMATH_CALUDE_committee_selections_of_seven_l1530_153023

/-- The number of ways to select a chairperson and a deputy chairperson from a committee. -/
def committee_selections (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a chairperson and a deputy chairperson 
    from a committee of 7 members is 42. -/
theorem committee_selections_of_seven : committee_selections 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_committee_selections_of_seven_l1530_153023


namespace NUMINAMATH_CALUDE_sangita_flying_hours_l1530_153076

/-- Calculates the required flying hours per month to meet a pilot certification goal -/
def required_hours_per_month (total_required : ℕ) (day_hours : ℕ) (night_hours : ℕ) (cross_country_hours : ℕ) (months : ℕ) : ℕ :=
  (total_required - (day_hours + night_hours + cross_country_hours)) / months

/-- Proves that Sangita needs to fly 220 hours per month to meet her goal -/
theorem sangita_flying_hours : 
  required_hours_per_month 1500 50 9 121 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sangita_flying_hours_l1530_153076


namespace NUMINAMATH_CALUDE_trig_identity_l1530_153018

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1530_153018


namespace NUMINAMATH_CALUDE_grinder_purchase_price_l1530_153022

theorem grinder_purchase_price
  (mobile_cost : ℝ)
  (grinder_loss_percent : ℝ)
  (mobile_profit_percent : ℝ)
  (total_profit : ℝ)
  (h1 : mobile_cost = 8000)
  (h2 : grinder_loss_percent = 0.05)
  (h3 : mobile_profit_percent = 0.10)
  (h4 : total_profit = 50) :
  ∃ (grinder_cost : ℝ),
    grinder_cost * (1 - grinder_loss_percent) +
    mobile_cost * (1 + mobile_profit_percent) -
    (grinder_cost + mobile_cost) = total_profit ∧
    grinder_cost = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_purchase_price_l1530_153022


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1530_153081

theorem complex_number_in_first_quadrant (m : ℝ) (h : m > 1) :
  let z : ℂ := m * (3 + Complex.I) - (2 + Complex.I)
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1530_153081


namespace NUMINAMATH_CALUDE_combination_sum_equality_problem_statement_l1530_153041

theorem combination_sum_equality : ∀ (n k : ℕ), k ≤ n →
  (Nat.choose n k) + (Nat.choose n (k+1)) = Nat.choose (n+1) (k+1) :=
sorry

theorem problem_statement : (Nat.choose 12 5) + (Nat.choose 12 6) = Nat.choose 13 6 :=
sorry

end NUMINAMATH_CALUDE_combination_sum_equality_problem_statement_l1530_153041


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l1530_153085

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (x ≠ 0 ∧ y ≠ 0) →
  (10 ≤ x ∧ x < 100) →
  (10 ≤ y ∧ y < 100) →
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →
  x^2 - y^2 = 44 * n →
  x + y + n = 93 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l1530_153085


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l1530_153049

theorem inequalities_satisfied (a b c x y z : ℤ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x^2*y + y^2*z + z^2*x ≤ a^2*b + b^2*c + c^2*a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x^2*y*z ≤ a^2*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l1530_153049


namespace NUMINAMATH_CALUDE_greatest_difference_of_arithmetic_progression_l1530_153079

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has two distinct real roots -/
def hasTwoRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all six quadratic equations with coefficients a, 2b, 4c in any order -/
def generateEquations (a b c : ℤ) : List QuadraticEquation :=
  [
    ⟨a, 2*b, 4*c⟩,
    ⟨a, 4*c, 2*b⟩,
    ⟨2*b, a, 4*c⟩,
    ⟨2*b, 4*c, a⟩,
    ⟨4*c, a, 2*b⟩,
    ⟨4*c, 2*b, a⟩
  ]

/-- The main theorem to be proved -/
theorem greatest_difference_of_arithmetic_progression
  (a b c : ℤ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_decreasing : a > b ∧ b > c)
  (h_arithmetic : ∃ d : ℤ, b = a + d ∧ c = a + 2*d)
  (h_two_roots : ∀ eq ∈ generateEquations a b c, hasTwoRoots eq) :
  ∃ (d : ℤ), d = -3 ∧ a = 4 ∧ b = 1 ∧ c = -2 ∧
  ∀ (d' : ℤ) (a' b' c' : ℤ),
    a' ≠ 0 → b' ≠ 0 → c' ≠ 0 →
    a' > b' → b' > c' →
    b' = a' + d' → c' = a' + 2*d' →
    (∀ eq ∈ generateEquations a' b' c', hasTwoRoots eq) →
    d' ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_of_arithmetic_progression_l1530_153079


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l1530_153036

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5,9) and
    one x-intercept at (0,0), prove that the x-coordinate of the other x-intercept is 10. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 5 = 9 ∧ (∀ y, f 5 ≤ f y))  -- Vertex at (5,9)
  (h3 : f 0 = 0)  -- x-intercept at (0,0)
  : ∃ x, x ≠ 0 ∧ f x = 0 ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l1530_153036


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1530_153007

theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 1 / 111 = 499 / 1189 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1530_153007


namespace NUMINAMATH_CALUDE_park_area_l1530_153070

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of 125 at 50 ps per meter has an area of 3750 square meters -/
theorem park_area (length width : ℝ) (h1 : length / width = 3 / 2) 
  (h2 : 2 * (length + width) * 0.5 = 125) : length * width = 3750 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l1530_153070


namespace NUMINAMATH_CALUDE_largest_k_value_l1530_153093

/-- A function that splits the whole numbers from 1 to 2k into two groups -/
def split_numbers (k : ℕ) : (Fin (2 * k) → Bool) := sorry

/-- A predicate that checks if two numbers share more than two distinct prime factors -/
def share_more_than_two_prime_factors (a b : ℕ) : Prop := sorry

/-- The main theorem stating that 44 is the largest possible value of k -/
theorem largest_k_value : 
  ∀ k : ℕ, k > 44 → 
  ¬∃ (f : Fin (2 * k) → Bool), 
    (∀ i j : Fin (2 * k), i.val < j.val ∧ f i = f j → 
      ¬share_more_than_two_prime_factors (i.val + 1) (j.val + 1)) ∧
    (Fintype.card {i : Fin (2 * k) | f i = true} = k) :=
sorry

end NUMINAMATH_CALUDE_largest_k_value_l1530_153093


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l1530_153054

theorem percentage_of_juniors (total : ℕ) (seniors : ℕ) :
  total = 800 →
  seniors = 160 →
  let sophomores := (total : ℚ) * (1 / 4)
  let freshmen := sophomores + 16
  let juniors := total - (freshmen + sophomores + seniors)
  (juniors / total) * 100 = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l1530_153054


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l1530_153088

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def SecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The given point P with coordinates (-1, 2). -/
def P : ℝ × ℝ := (-1, 2)

/-- Theorem: The point P lies in the second quadrant. -/
theorem P_in_second_quadrant : SecondQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_second_quadrant_l1530_153088


namespace NUMINAMATH_CALUDE_rotation_theorem_l1530_153021

/-- Triangle in 2D plane -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Rotation of a point around another point by 120° clockwise -/
def rotate120 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Generate the sequence of A points -/
def A (n : ℕ) (t : Triangle) : ℝ × ℝ :=
  match n % 3 with
  | 0 => t.A₃
  | 1 => t.A₁
  | _ => t.A₂

/-- Generate the sequence of P points -/
def P (n : ℕ) (t : Triangle) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  match n with
  | 0 => P₀
  | n + 1 => rotate120 (A (n + 1) t) (P n t P₀)

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

theorem rotation_theorem (t : Triangle) (P₀ : ℝ × ℝ) :
  P 1986 t P₀ = P₀ → isEquilateral t := by sorry

end NUMINAMATH_CALUDE_rotation_theorem_l1530_153021


namespace NUMINAMATH_CALUDE_expression_value_l1530_153059

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = 2) : 3 * a + 4 * b - 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1530_153059


namespace NUMINAMATH_CALUDE_sin_negative_300_degrees_l1530_153019

theorem sin_negative_300_degrees : Real.sin (-(300 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_300_degrees_l1530_153019


namespace NUMINAMATH_CALUDE_reservoir_drainage_l1530_153097

/- Given conditions -/
def initial_drainage_rate : ℝ := 8
def initial_drain_time : ℝ := 6
def max_drainage_capacity : ℝ := 12

/- Theorem statement -/
theorem reservoir_drainage :
  let reservoir_volume : ℝ := initial_drainage_rate * initial_drain_time
  let drainage_relation (Q t : ℝ) : Prop := Q = reservoir_volume / t
  let min_drainage_5hours : ℝ := reservoir_volume / 5
  let min_time_max_capacity : ℝ := reservoir_volume / max_drainage_capacity
  
  (reservoir_volume = 48) ∧
  (∀ Q t, drainage_relation Q t ↔ Q = 48 / t) ∧
  (min_drainage_5hours = 9.6) ∧
  (min_time_max_capacity = 4) :=
by sorry

end NUMINAMATH_CALUDE_reservoir_drainage_l1530_153097


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l1530_153087

/-- Given a point C with coordinates (3, k), its reflection D over the y-axis
    with y-coordinate increased by 4, prove that the sum of all coordinates
    of C and D is 2k + 4. -/
theorem sum_of_coordinates (k : ℝ) : 
  let C : ℝ × ℝ := (3, k)
  let D : ℝ × ℝ := (-3, k + 4)
  (C.1 + C.2 + D.1 + D.2) = 2 * k + 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l1530_153087


namespace NUMINAMATH_CALUDE_centroid_eq_circumcenter_implies_equilateral_l1530_153099

/-- A triangle in a 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the centroid of a triangle coincides with its circumcenter, 
then the triangle is equilateral
-/
theorem centroid_eq_circumcenter_implies_equilateral (t : Triangle) :
  centroid t = circumcenter t → is_equilateral t := by sorry

end NUMINAMATH_CALUDE_centroid_eq_circumcenter_implies_equilateral_l1530_153099


namespace NUMINAMATH_CALUDE_biquadratic_equation_roots_l1530_153086

theorem biquadratic_equation_roots (x : ℝ) :
  x^4 - 8*x^2 + 4 = 0 ↔ x = Real.sqrt 3 - 1 ∨ x = Real.sqrt 3 + 1 ∨ x = -(Real.sqrt 3 - 1) ∨ x = -(Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_biquadratic_equation_roots_l1530_153086


namespace NUMINAMATH_CALUDE_event_A_is_certain_l1530_153090

/-- The set of card labels -/
def card_labels : Finset ℕ := {1, 2, 3, 4, 5}

/-- The event "The label is less than 6" -/
def event_A (n : ℕ) : Prop := n < 6

/-- Theorem: Event A is a certain event -/
theorem event_A_is_certain : ∀ n ∈ card_labels, event_A n := by
  sorry

end NUMINAMATH_CALUDE_event_A_is_certain_l1530_153090


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_conditions_l1530_153001

noncomputable def f (a b x : ℝ) : ℝ := a + (b * x - 1) * Real.exp x

theorem tangent_line_and_inequality_conditions 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 1) 
  (h3 : a < 1) 
  (h4 : b = 2) 
  (h5 : ∃! (n : ℤ), f a b n < a * n) :
  a = 1 ∧ b = 2 ∧ 3 / (2 * Real.exp 1) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_conditions_l1530_153001


namespace NUMINAMATH_CALUDE_odd_integer_dividing_power_plus_one_l1530_153091

theorem odd_integer_dividing_power_plus_one (n : ℕ) : 
  n ≥ 1 → 
  Odd n → 
  (n ∣ 3^n + 1) → 
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_dividing_power_plus_one_l1530_153091


namespace NUMINAMATH_CALUDE_min_value_expression_l1530_153050

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 10 / 5 ∧
  ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (abs (x + 3*y - y*(x + 9*y)) + abs (3*y - x + 3*y*(x - y))) / Real.sqrt (x^2 + 9*y^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1530_153050


namespace NUMINAMATH_CALUDE_alex_hula_hoop_duration_l1530_153058

-- Define the hula hoop durations for each person
def nancy_duration : ℕ := 10

-- Casey's duration is 3 minutes less than Nancy's
def casey_duration : ℕ := nancy_duration - 3

-- Morgan's duration is three times Casey's duration
def morgan_duration : ℕ := casey_duration * 3

-- Alex's duration is the sum of Casey's and Morgan's durations minus 2 minutes
def alex_duration : ℕ := casey_duration + morgan_duration - 2

-- Theorem to prove Alex's hula hoop duration
theorem alex_hula_hoop_duration : alex_duration = 26 := by
  sorry

end NUMINAMATH_CALUDE_alex_hula_hoop_duration_l1530_153058


namespace NUMINAMATH_CALUDE_parallel_planes_transitivity_l1530_153055

structure Plane

/-- Two planes are parallel -/
def parallel (p q : Plane) : Prop := sorry

theorem parallel_planes_transitivity 
  (α β γ : Plane) 
  (h1 : α ≠ β) 
  (h2 : α ≠ γ) 
  (h3 : β ≠ γ) 
  (h4 : parallel α β) 
  (h5 : parallel α γ) : 
  parallel β γ := by sorry

end NUMINAMATH_CALUDE_parallel_planes_transitivity_l1530_153055


namespace NUMINAMATH_CALUDE_part_a_part_b_l1530_153098

-- Define the main equation
def main_equation (x p : ℝ) : Prop := x^2 + p = -x/4

-- Define the condition for part a
def condition_a (x₁ x₂ : ℝ) : Prop := x₁/x₂ + x₂/x₁ = -9/4

-- Define the condition for part b
def condition_b (x₁ x₂ : ℝ) : Prop := x₂ = x₁^2 - 1

-- Theorem for part a
theorem part_a (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_a x₁ x₂ → p = -1/23 := by
  sorry

-- Theorem for part b
theorem part_b (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_b x₁ x₂ →
  p = -3/8 ∨ p = -15/8 := by
  sorry

end NUMINAMATH_CALUDE_part_a_part_b_l1530_153098


namespace NUMINAMATH_CALUDE_reciprocal_sum_property_l1530_153033

theorem reciprocal_sum_property (x y : ℝ) (h : x > 0) (h' : y > 0) (h'' : 1 / x + 1 / y = 1) :
  (x - 1) * (y - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_property_l1530_153033


namespace NUMINAMATH_CALUDE_pipe_fill_time_pipe_B_fill_time_l1530_153075

/-- Given two pipes A and B that can fill a tank, this theorem proves the time it takes for pipe B to fill the tank. -/
theorem pipe_fill_time (fill_time_A : ℝ) (fill_time_both : ℝ) (fill_amount : ℝ) : ℝ :=
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_both := fill_amount / fill_time_both
  let fill_rate_B := fill_rate_both - fill_rate_A
  1 / fill_rate_B

/-- The main theorem that proves the time it takes for pipe B to fill the tank under the given conditions. -/
theorem pipe_B_fill_time : pipe_fill_time 16 12.000000000000002 (5/4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_pipe_B_fill_time_l1530_153075


namespace NUMINAMATH_CALUDE_mary_final_book_count_l1530_153012

def calculate_final_books (initial_books : ℕ) (monthly_club_books : ℕ) (months : ℕ) 
  (bought_books : ℕ) (gift_books : ℕ) (removed_books : ℕ) : ℕ :=
  initial_books + monthly_club_books * months + bought_books + gift_books - removed_books

theorem mary_final_book_count : 
  calculate_final_books 72 1 12 7 5 15 = 81 := by sorry

end NUMINAMATH_CALUDE_mary_final_book_count_l1530_153012


namespace NUMINAMATH_CALUDE_min_square_value_l1530_153069

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 * a + 16 * b : ℕ) = r^2)
  (h2 : ∃ s : ℕ, (16 * a - 15 * b : ℕ) = s^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 231361 := by
sorry

end NUMINAMATH_CALUDE_min_square_value_l1530_153069


namespace NUMINAMATH_CALUDE_circle_divides_sides_l1530_153044

/-- An isosceles trapezoid with bases in ratio 3:2 and a circle on the larger base -/
structure IsoscelesTrapezoidWithCircle where
  /-- Length of the smaller base -/
  b : ℝ
  /-- Length of the larger base -/
  a : ℝ
  /-- The bases are in ratio 3:2 -/
  base_ratio : a = (3/2) * b
  /-- The trapezoid is isosceles -/
  isosceles : True
  /-- Radius of the circle (half of the larger base) -/
  r : ℝ
  circle_diameter : r = a / 2
  /-- Length of the segment cut off on the smaller base by the circle -/
  m : ℝ
  segment_half_base : m = b / 2

/-- The circle divides the non-parallel sides of the trapezoid in the ratio 1:2 -/
theorem circle_divides_sides (t : IsoscelesTrapezoidWithCircle) :
  ∃ (x y : ℝ), x + y = t.a - t.b ∧ x / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_divides_sides_l1530_153044


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l1530_153014

def angle_rotation (initial_angle rotation : ℕ) : ℕ :=
  (rotation - initial_angle) % 360

theorem rotated_angle_measure (initial_angle rotation : ℕ) 
  (h1 : initial_angle = 70) 
  (h2 : rotation = 570) : 
  angle_rotation initial_angle rotation = 140 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l1530_153014


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1530_153030

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  (2 : ℕ) * (2 : ℕ) * (Nat.factorial 4) = 96 :=
sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1530_153030


namespace NUMINAMATH_CALUDE_f_at_two_l1530_153027

/-- Horner's method representation of the polynomial 2x^4 + 3x^3 + 5x - 4 --/
def f (x : ℝ) : ℝ := ((2 * x + 3) * x + 0) * x + 5 * x - 4

/-- Theorem stating that f(2) = 62 --/
theorem f_at_two : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l1530_153027


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1530_153094

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (378461 - k) % 13 = 0 ∧ ∀ (m : ℕ), m < k → (378461 - m) % 13 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1530_153094


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1530_153006

/-- Given a distance of 88 miles and a time of 4 hours, prove that the average speed is 22 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 88) (h2 : time = 4) :
  distance / time = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1530_153006


namespace NUMINAMATH_CALUDE_percentage_relation_l1530_153039

theorem percentage_relation (A B C x y : ℝ) : 
  A > C ∧ C > B ∧ B > 0 →
  C = B * (1 + y / 100) →
  A = C * (1 + x / 100) →
  x = 100 * ((100 * (A - B)) / (100 + y)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l1530_153039


namespace NUMINAMATH_CALUDE_zoo_feeding_arrangements_l1530_153083

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange the animals according to the specified pattern -/
def arrangement_count : ℕ :=
  (num_pairs - 1) * -- choices for the second female
  num_pairs * -- choices for the first male
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) * -- choices for remaining females
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) -- choices for remaining males

/-- The theorem stating that the number of possible arrangements is 432000 -/
theorem zoo_feeding_arrangements : arrangement_count = 432000 := by
  sorry

end NUMINAMATH_CALUDE_zoo_feeding_arrangements_l1530_153083


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1530_153028

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1530_153028


namespace NUMINAMATH_CALUDE_weeklySalesTheorem_l1530_153011

/-- Calculates the total sales for a week given the following conditions:
- Number of houses visited per day
- Percentage of customers who buy something
- Percentages and prices of different products
- Number of working days
- Discount percentage on the last day
-/
def calculateWeeklySales (
  housesPerDay : ℕ)
  (buyPercentage : ℚ)
  (product1Percentage : ℚ) (product1Price : ℚ)
  (product2Percentage : ℚ) (product2Price : ℚ)
  (product3Percentage : ℚ) (product3Price : ℚ)
  (product4Percentage : ℚ) (product4Price : ℚ)
  (workingDays : ℕ)
  (lastDayDiscount : ℚ) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, the total sales for the week
    equal $9624.375 -/
theorem weeklySalesTheorem :
  calculateWeeklySales 50 (30/100)
    (35/100) 50
    (40/100) 150
    (15/100) 75
    (10/100) 200
    6 (10/100) = 9624375/1000 := by sorry

end NUMINAMATH_CALUDE_weeklySalesTheorem_l1530_153011


namespace NUMINAMATH_CALUDE_median_eq_twelve_l1530_153096

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  -- The height of the trapezoid
  height : ℝ
  -- The angle AOD, where O is the intersection of diagonals
  angle_AOD : ℝ
  -- Assumption that the height is 4√3
  height_eq : height = 4 * Real.sqrt 3
  -- Assumption that ∠AOD is 120°
  angle_AOD_eq : angle_AOD = 120

/-- The median of an isosceles trapezoid -/
def median (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The median of the given isosceles trapezoid is 12 -/
theorem median_eq_twelve (t : IsoscelesTrapezoid) : median t = 12 := by sorry

end NUMINAMATH_CALUDE_median_eq_twelve_l1530_153096


namespace NUMINAMATH_CALUDE_unique_center_symmetric_not_axis_symmetric_l1530_153066

-- Define the shapes
inductive Shape
  | Square
  | EquilateralTriangle
  | Circle
  | Parallelogram

-- Define the symmetry properties
def is_center_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle => true
  | Shape.Parallelogram => true

def is_axis_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle => true
  | Shape.Parallelogram => false

-- Theorem statement
theorem unique_center_symmetric_not_axis_symmetric :
  ∀ s : Shape, (is_center_symmetric s ∧ ¬is_axis_symmetric s) ↔ s = Shape.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_unique_center_symmetric_not_axis_symmetric_l1530_153066


namespace NUMINAMATH_CALUDE_triangle_side_length_l1530_153061

noncomputable section

/-- Given a triangle ABC with angles A, B, C and opposite sides a, b, c respectively,
    if A = 30°, B = 45°, and a = √2, then b = 2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = π/4 → a = Real.sqrt 2 → 
  (a / Real.sin A = b / Real.sin B) → 
  b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1530_153061


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1530_153000

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a + 2)x and f'(x) is an even function,
    then the equation of the tangent line to y=f(x) at the origin is y = 2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a + 2)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a + 2))
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x ↦ 2*x) = (λ x ↦ f' 0 * x + f 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1530_153000


namespace NUMINAMATH_CALUDE_expression_values_l1530_153053

theorem expression_values (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  let expr := p / |p| + q / |q| + r / |r| + s / |s| + (p * q * r) / |p * q * r| + (p * r * s) / |p * r * s|
  expr = 6 ∨ expr = 2 ∨ expr = 0 ∨ expr = -2 ∨ expr = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1530_153053


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1530_153025

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1530_153025


namespace NUMINAMATH_CALUDE_unit_vectors_parallel_to_a_l1530_153068

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vectors_parallel_to_a :
  let magnitude := Real.sqrt (vector_a.1^2 + vector_a.2^2)
  let unit_vector := (vector_a.1 / magnitude, vector_a.2 / magnitude)
  (unit_vector = (12/13, 5/13) ∨ unit_vector = (-12/13, -5/13)) ∧
  (∀ v : ℝ × ℝ, (v.1^2 + v.2^2 = 1 ∧ ∃ k : ℝ, v = (k * vector_a.1, k * vector_a.2)) →
    (v = (12/13, 5/13) ∨ v = (-12/13, -5/13))) :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_parallel_to_a_l1530_153068


namespace NUMINAMATH_CALUDE_opposite_of_seven_l1530_153048

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l1530_153048


namespace NUMINAMATH_CALUDE_minimal_shots_to_hit_triangle_l1530_153056

/-- A point on the circle --/
structure Point where
  index : Nat
  h_index : index ≥ 1 ∧ index ≤ 29

/-- A shot is a pair of distinct points --/
structure Shot where
  p1 : Point
  p2 : Point
  h_distinct : p1.index ≠ p2.index

/-- A triangle on the circle --/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point
  h_distinct : v1.index ≠ v2.index ∧ v2.index ≠ v3.index ∧ v3.index ≠ v1.index

/-- A function to determine if a shot hits a triangle --/
def hits (s : Shot) (t : Triangle) : Prop :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem minimal_shots_to_hit_triangle :
  ∀ t : Triangle, ∃ K : Nat, K = 100 ∧
    (∀ shots : Finset Shot, shots.card = K →
      (∀ s ∈ shots, hits s t)) ∧
    (∀ K' : Nat, K' < K →
      ∃ shots : Finset Shot, shots.card = K' ∧
        ∃ s ∈ shots, ¬hits s t) :=
sorry

end NUMINAMATH_CALUDE_minimal_shots_to_hit_triangle_l1530_153056


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1530_153042

/-- The number of handshakes in a basketball game with two teams and referees -/
theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 := by
  sorry

#check basketball_handshakes

end NUMINAMATH_CALUDE_basketball_handshakes_l1530_153042


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_l1530_153064

theorem factorization_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_l1530_153064


namespace NUMINAMATH_CALUDE_divisibility_condition_l1530_153034

theorem divisibility_condition (n : ℤ) : 
  (n + 2) ∣ (n^2 + 3) ↔ n ∈ ({-9, -3, -1, 5} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1530_153034


namespace NUMINAMATH_CALUDE_ball_probability_l1530_153046

theorem ball_probability (m : ℕ) : 
  (3 : ℚ) / (3 + 4 + m) = 1 / 3 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1530_153046


namespace NUMINAMATH_CALUDE_inequality_range_l1530_153035

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1530_153035


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l1530_153060

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 5 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 5 = 0 → y = x) ↔ 
  (m = 2 - 2 * Real.sqrt 15 ∨ m = 2 + 2 * Real.sqrt 15) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l1530_153060


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1530_153071

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1530_153071


namespace NUMINAMATH_CALUDE_total_distance_traveled_l1530_153057

-- Define the speeds and conversion factors
def two_sail_speed : ℝ := 50
def one_sail_speed : ℝ := 25
def nautical_to_land_miles : ℝ := 1.15

-- Define the journey segments
def segment1_hours : ℝ := 2
def segment2_hours : ℝ := 3
def segment3_hours : ℝ := 1
def segment4_hours : ℝ := 2
def segment4_speed_reduction : ℝ := 0.3

-- Define the theorem
theorem total_distance_traveled :
  let segment1_distance := one_sail_speed * segment1_hours
  let segment2_distance := two_sail_speed * segment2_hours
  let segment3_distance := one_sail_speed * segment3_hours
  let segment4_distance := (one_sail_speed * (1 - segment4_speed_reduction)) * segment4_hours
  let total_nautical_miles := segment1_distance + segment2_distance + segment3_distance + segment4_distance
  let total_land_miles := total_nautical_miles * nautical_to_land_miles
  total_land_miles = 299 := by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l1530_153057


namespace NUMINAMATH_CALUDE_circle_radius_l1530_153004

/-- Given a circle with area M cm² and circumference N cm,
    where M/N = 15 and the area is 60π cm²,
    prove that the radius of the circle is 2√15 cm. -/
theorem circle_radius (M N : ℝ) (h1 : M / N = 15) (h2 : M = 60 * Real.pi) :
  ∃ (r : ℝ), r = 2 * Real.sqrt 15 ∧ M = Real.pi * r^2 ∧ N = 2 * Real.pi * r :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l1530_153004


namespace NUMINAMATH_CALUDE_one_tree_baskets_l1530_153003

/-- The number of apples that can fit in one basket -/
def apples_per_basket : ℕ := 15

/-- The number of apples produced by 10 trees -/
def apples_from_ten_trees : ℕ := 3000

/-- The number of trees -/
def number_of_trees : ℕ := 10

/-- Theorem: One apple tree can fill 20 baskets -/
theorem one_tree_baskets : 
  (apples_from_ten_trees / number_of_trees) / apples_per_basket = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_tree_baskets_l1530_153003


namespace NUMINAMATH_CALUDE_power_of_power_l1530_153040

theorem power_of_power (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1530_153040


namespace NUMINAMATH_CALUDE_investment_ratio_l1530_153095

theorem investment_ratio (a b c : ℝ) (profit total_profit : ℝ) : 
  a = 3 * b →                           -- A invests 3 times as much as B
  profit = 15000.000000000002 →         -- C's share
  total_profit = 55000 →                -- Total profit
  profit / total_profit = c / (a + b + c) → -- Profit distribution ratio
  a / c = 2                             -- Ratio of A's investment to C's investment
:= by sorry

end NUMINAMATH_CALUDE_investment_ratio_l1530_153095


namespace NUMINAMATH_CALUDE_m_range_theorem_l1530_153067

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2)

/-- Line and parabola have no intersections -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- Equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (5 - 2*m) + y^2 / m = 1 → m * (5 - 2*m) < 0

/-- Main theorem -/
theorem m_range_theorem (m : ℝ) :
  (no_intersection m ∨ is_hyperbola m) ∧ ¬(no_intersection m ∧ is_hyperbola m) →
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1530_153067


namespace NUMINAMATH_CALUDE_inequality_and_equalities_l1530_153080

theorem inequality_and_equalities : 
  ((-3)^2 ≠ -3^2) ∧ 
  (|-5| = -(-5)) ∧ 
  (-Real.sqrt 4 = -2) ∧ 
  ((-1)^3 = -1^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equalities_l1530_153080


namespace NUMINAMATH_CALUDE_circle_area_increase_l1530_153013

theorem circle_area_increase (r : ℝ) (hr : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1530_153013


namespace NUMINAMATH_CALUDE_max_product_sum_200_l1530_153024

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_200_l1530_153024


namespace NUMINAMATH_CALUDE_computer_contract_probability_l1530_153051

theorem computer_contract_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_at_least_one = 9/10) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 7/10 := by
sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l1530_153051


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1530_153089

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1530_153089


namespace NUMINAMATH_CALUDE_left_handed_fiction_readers_count_l1530_153045

/-- Represents a book club with members and their preferences. -/
structure BookClub where
  total_members : ℕ
  fiction_readers : ℕ
  left_handed : ℕ
  right_handed_non_fiction : ℕ

/-- Calculates the number of left-handed fiction readers in the book club. -/
def left_handed_fiction_readers (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.fiction_readers - club.right_handed_non_fiction)

/-- Theorem stating that in a specific book club configuration, 
    the number of left-handed fiction readers is 5. -/
theorem left_handed_fiction_readers_count :
  let club : BookClub := {
    total_members := 25,
    fiction_readers := 15,
    left_handed := 12,
    right_handed_non_fiction := 3
  }
  left_handed_fiction_readers club = 5 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fiction_readers_count_l1530_153045


namespace NUMINAMATH_CALUDE_women_in_salem_l1530_153062

def leesburg_population : ℕ := 58940
def salem_population_multiplier : ℕ := 15
def people_moved_out : ℕ := 130000

def salem_original_population : ℕ := leesburg_population * salem_population_multiplier
def salem_current_population : ℕ := salem_original_population - people_moved_out

theorem women_in_salem : 
  (salem_current_population / 2 : ℕ) = 377050 := by sorry

end NUMINAMATH_CALUDE_women_in_salem_l1530_153062


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1530_153052

/-- The imaginary part of (1 - √2i) / i is -1 -/
theorem imaginary_part_of_z : Complex.im ((1 - Complex.I * Real.sqrt 2) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1530_153052


namespace NUMINAMATH_CALUDE_not_always_equal_l1530_153078

-- Define the binary operation
def binary_op {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), ∃ (c : S), op a b = c

-- Define the property of the operation
def special_property {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), op a (op b a) = b

theorem not_always_equal {S : Type} [Inhabited S] (op : S → S → S) 
  (h1 : binary_op op) (h2 : special_property op) (h3 : ∃ (x y : S), x ≠ y) :
  ∃ (a b : S), op (op a b) a ≠ a := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_l1530_153078


namespace NUMINAMATH_CALUDE_daniel_elsa_distance_diff_l1530_153072

/-- Calculates the difference in distance traveled between two cyclists given their speeds and times on different tracks. -/
def distance_difference (daniel_plain_speed elsa_plain_speed : ℝ)
                        (plain_time : ℝ)
                        (daniel_hilly_speed elsa_hilly_speed : ℝ)
                        (hilly_time : ℝ) : ℝ :=
  let daniel_total := daniel_plain_speed * plain_time + daniel_hilly_speed * hilly_time
  let elsa_total := elsa_plain_speed * plain_time + elsa_hilly_speed * hilly_time
  daniel_total - elsa_total

/-- The difference in distance traveled between Daniel and Elsa is 7 miles. -/
theorem daniel_elsa_distance_diff :
  distance_difference 20 18 3 16 15 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_daniel_elsa_distance_diff_l1530_153072


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1530_153092

theorem complex_modulus_problem (z : ℂ) (h : (1 - 2*I)*z = 5*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1530_153092


namespace NUMINAMATH_CALUDE_lemons_count_l1530_153031

/-- Represents the contents of Tania's fruit baskets -/
structure FruitBaskets where
  total_fruits : ℕ
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  oranges_basket3 : ℕ
  kiwis_basket4 : ℕ
  oranges_basket4 : ℕ

/-- The number of lemons in Tania's baskets -/
def count_lemons (baskets : FruitBaskets) : ℕ :=
  (baskets.total_fruits - (baskets.mangoes + baskets.pears + baskets.pawpaws + 
   baskets.oranges_basket3 + baskets.kiwis_basket4 + baskets.oranges_basket4)) / 3

/-- Theorem stating that the number of lemons in Tania's baskets is 8 -/
theorem lemons_count (baskets : FruitBaskets) 
  (h1 : baskets.total_fruits = 83)
  (h2 : baskets.mangoes = 18)
  (h3 : baskets.pears = 14)
  (h4 : baskets.pawpaws = 10)
  (h5 : baskets.oranges_basket3 = 5)
  (h6 : baskets.kiwis_basket4 = 8)
  (h7 : baskets.oranges_basket4 = 4) :
  count_lemons baskets = 8 := by
  sorry

end NUMINAMATH_CALUDE_lemons_count_l1530_153031


namespace NUMINAMATH_CALUDE_geometry_propositions_l1530_153010

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) : 
  -- Proposition 1 is false
  ¬(∀ m α β, parallel_line_plane m α → parallel_line_plane m β → parallel_plane_plane α β) ∧
  -- Proposition 2 is true
  (∀ m α β, perpendicular_line_plane m α → perpendicular_line_plane m β → parallel_plane_plane α β) ∧
  -- Proposition 3 is false
  ¬(∀ m n α, parallel_line_plane m α → parallel_line_plane n α → parallel_line_line m n) ∧
  -- Proposition 4 is true
  (∀ m n α, perpendicular_line_plane m α → perpendicular_line_plane n α → parallel_line_line m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1530_153010


namespace NUMINAMATH_CALUDE_curve_is_circle_l1530_153005

theorem curve_is_circle (θ : Real) (r : Real → Real) :
  (∀ θ, r θ = 1 / (1 - Real.sin θ)) →
  ∃ (x y : Real → Real), ∀ θ,
    x θ ^ 2 + (y θ - 1) ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_circle_l1530_153005


namespace NUMINAMATH_CALUDE_zero_of_f_l1530_153043

-- Define the function f
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l1530_153043


namespace NUMINAMATH_CALUDE_unique_prime_solution_l1530_153077

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p → Prime q → Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l1530_153077


namespace NUMINAMATH_CALUDE_zeros_of_f_l1530_153084

open Real MeasureTheory Set

noncomputable def f (x : ℝ) := cos x - sin (2 * x)

def I : Set ℝ := Icc 0 (2 * π)

theorem zeros_of_f : 
  (∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ S, x ∈ I ∧ f x = 0) ∧
  (∀ y ∈ I, f y = 0 → y ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1530_153084


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l1530_153026

theorem quadratic_root_existence (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (1/2 * a * x₃^2 + b * x₃ + c = 0) ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l1530_153026


namespace NUMINAMATH_CALUDE_sum_at_two_and_minus_two_l1530_153029

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d

theorem sum_at_two_and_minus_two
  (P : ℝ → ℝ)
  (k : ℝ)
  (h_cubic : cubic_polynomial P)
  (h_zero : P 0 = k)
  (h_one : P 1 = 3 * k)
  (h_neg_one : P (-1) = 4 * k) :
  P 2 + P (-2) = 22 * k :=
sorry

end NUMINAMATH_CALUDE_sum_at_two_and_minus_two_l1530_153029


namespace NUMINAMATH_CALUDE_bank_max_profit_rate_l1530_153037

/-- The bank's profit function --/
def profit (x : ℝ) : ℝ := 480 * x^2 - 10000 * x^3

/-- The derivative of the profit function --/
def profit_derivative (x : ℝ) : ℝ := 960 * x - 30000 * x^2

theorem bank_max_profit_rate :
  ∃ x : ℝ, x ∈ Set.Ioo 0 0.048 ∧
    (∀ y ∈ Set.Ioo 0 0.048, profit y ≤ profit x) ∧
    x = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_bank_max_profit_rate_l1530_153037


namespace NUMINAMATH_CALUDE_quadratic_with_zero_root_l1530_153008

/-- Given a quadratic equation (k-2)x^2 + x + k^2 - 4 = 0 where 0 is one of its roots,
    prove that k = -2 -/
theorem quadratic_with_zero_root (k : ℝ) : 
  (∀ x : ℝ, (k - 2) * x^2 + x + k^2 - 4 = 0 ↔ x = 0 ∨ x = (k^2 - 4) / (2 - k)) →
  ((k - 2) * 0^2 + 0 + k^2 - 4 = 0) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_zero_root_l1530_153008


namespace NUMINAMATH_CALUDE_even_periodic_function_derivative_zero_l1530_153065

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_derivative_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_derivative_zero_l1530_153065


namespace NUMINAMATH_CALUDE_joans_video_game_cost_l1530_153047

/-- Calculates the total cost of video games with discount and tax --/
def totalCost (basketballPrice racingPrice actionPrice : ℝ) 
               (discount tax : ℝ) : ℝ :=
  let discountedTotal := (basketballPrice + racingPrice + actionPrice) * (1 - discount)
  discountedTotal * (1 + tax)

/-- Theorem stating the total cost of Joan's video game purchase --/
theorem joans_video_game_cost :
  let basketballPrice := 5.2
  let racingPrice := 4.23
  let actionPrice := 7.12
  let discount := 0.1
  let tax := 0.06
  ∃ (cost : ℝ), abs (totalCost basketballPrice racingPrice actionPrice discount tax - cost) < 0.005 ∧ cost = 15.79 :=
by
  sorry


end NUMINAMATH_CALUDE_joans_video_game_cost_l1530_153047


namespace NUMINAMATH_CALUDE_new_regression_equation_l1530_153032

-- Define the initial regression line
def initial_regression (x : ℝ) : ℝ := 2 * x - 0.4

-- Define the sample size and mean x
def sample_size : ℕ := 10
def mean_x : ℝ := 2

-- Define the removed points
def removed_point1 : ℝ × ℝ := (-3, 1)
def removed_point2 : ℝ × ℝ := (3, -1)

-- Define the new slope
def new_slope : ℝ := 3

-- Theorem statement
theorem new_regression_equation :
  let new_mean_x := (mean_x * sample_size - (removed_point1.1 + removed_point2.1)) / (sample_size - 2)
  let new_mean_y := (initial_regression mean_x * sample_size - (removed_point1.2 + removed_point2.2)) / (sample_size - 2)
  let new_intercept := new_mean_y - new_slope * new_mean_x
  ∀ x, new_slope * x + new_intercept = 3 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_new_regression_equation_l1530_153032


namespace NUMINAMATH_CALUDE_graph_of_S_l1530_153020

theorem graph_of_S (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (sum_eq : a + b = 2) (prod_eq : a * b = t - 1) (ht : 1 < t ∧ t < 2) :
  (a - b)^2 = 8 - 4*t := by
  sorry

end NUMINAMATH_CALUDE_graph_of_S_l1530_153020


namespace NUMINAMATH_CALUDE_damaged_chair_percentage_is_40_l1530_153009

/-- Represents the number of office chairs initially -/
def initial_chairs : ℕ := 80

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 5

/-- Represents the number of round tables -/
def tables : ℕ := 20

/-- Represents the number of legs each table has -/
def legs_per_table : ℕ := 3

/-- Represents the total number of legs remaining after damage -/
def remaining_legs : ℕ := 300

/-- Calculates the percentage of chairs damaged and disposed of -/
def damaged_chair_percentage : ℚ :=
  let total_initial_legs := initial_chairs * legs_per_chair + tables * legs_per_table
  let disposed_legs := total_initial_legs - remaining_legs
  let disposed_chairs := disposed_legs / legs_per_chair
  (disposed_chairs : ℚ) / initial_chairs * 100

/-- Theorem stating that the percentage of chairs damaged and disposed of is 40% -/
theorem damaged_chair_percentage_is_40 :
  damaged_chair_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_damaged_chair_percentage_is_40_l1530_153009


namespace NUMINAMATH_CALUDE_condition_nature_l1530_153017

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | |2*x - a| < 2}

-- Theorem statement
theorem condition_nature (a : ℝ) :
  (∀ a, 1 ∈ M a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 4 ∧ 1 ∉ M a) := by
  sorry

end NUMINAMATH_CALUDE_condition_nature_l1530_153017


namespace NUMINAMATH_CALUDE_square_filling_theorem_l1530_153082

def is_valid_permutation (p : Fin 5 → Fin 5) : Prop :=
  Function.Injective p ∧ Function.Surjective p

theorem square_filling_theorem :
  ∃ (p : Fin 5 → Fin 5), is_valid_permutation p ∧
    (p 0).val + 1 + (p 1).val + 1 = ((p 2).val + 1) * ((p 3).val + 1 - ((p 4).val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_filling_theorem_l1530_153082


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l1530_153073

/-- Theorem: For an ellipse with equation x²/a² + y²/k = 1 and a focus at (0, √2), k = 2 -/
theorem ellipse_focus_k_value (a : ℝ) (k : ℝ) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / k = 1) →  -- Ellipse equation
  (0^2 / a^2 + (Real.sqrt 2)^2 / k = 1) →  -- Focus (0, √2) is on the ellipse
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l1530_153073


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1530_153016

/-- Given that 2/3 of 10 bananas are worth as much as 8 oranges,
    prove that 1/2 of 5 bananas are worth as much as 3 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
    (2 / 3 : ℚ) * 10 * banana_value = 8 * orange_value →
    (1 / 2 : ℚ) * 5 * banana_value = 3 * orange_value := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1530_153016


namespace NUMINAMATH_CALUDE_new_student_weight_l1530_153074

theorem new_student_weight (n : ℕ) (w_avg_initial w_avg_final w_new : ℝ) :
  n = 29 →
  w_avg_initial = 28 →
  w_avg_final = 27.2 →
  (n : ℝ) * w_avg_initial = ((n : ℝ) + 1) * w_avg_final - w_new →
  w_new = 4 := by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1530_153074


namespace NUMINAMATH_CALUDE_investment_amount_l1530_153063

/-- Calculates the investment amount given the dividend received and share details --/
def calculate_investment (share_value : ℕ) (premium_percentage : ℕ) (dividend_percentage : ℕ) (dividend_received : ℕ) : ℕ :=
  let premium_factor := 1 + premium_percentage / 100
  let share_price := share_value * premium_factor
  let dividend_per_share := share_value * dividend_percentage / 100
  let num_shares := dividend_received / dividend_per_share
  num_shares * share_price

/-- Proves that the investment amount is 14375 given the problem conditions --/
theorem investment_amount : calculate_investment 100 25 5 576 = 14375 := by
  sorry

#eval calculate_investment 100 25 5 576

end NUMINAMATH_CALUDE_investment_amount_l1530_153063


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1530_153002

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1530_153002


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l1530_153015

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l1530_153015
