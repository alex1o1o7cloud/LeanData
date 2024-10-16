import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_proof_l754_75488

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a circle. -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passesThrough.1 - c.center.1)^2 + (c.passesThrough.2 - c.center.2)^2

/-- The specific circle from the problem. -/
def C : Circle :=
  { center := (2, -3)
  , passesThrough := (0, 0) }

theorem circle_equation_proof :
  ∀ x y : ℝ, circleEquation C x y ↔ (x - 2)^2 + (y + 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l754_75488


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l754_75447

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def is_tangent_to_circle (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 ∧
    l.a * p.x + l.b * p.y + l.c = 0

def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem tangent_lines_to_circle (c : Circle) (p : Point) :
  let l1 : Line := { a := 8, b := 15, c := -37 }
  let l2 : Line := { a := 1, b := 0, c := 1 }
  c.center = { x := 2, y := -1 } ∧ c.radius = 3 ∧ p = { x := -1, y := 3 } →
  (is_tangent_to_circle l1 c ∧ point_on_line p l1) ∧
  (is_tangent_to_circle l2 c ∧ point_on_line p l2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l754_75447


namespace NUMINAMATH_CALUDE_bianca_not_recycled_bags_l754_75443

/-- The number of bags Bianca did not recycle -/
def bags_not_recycled (total_bags : ℕ) (points_per_bag : ℕ) (total_points : ℕ) : ℕ :=
  total_bags - (total_points / points_per_bag)

/-- Theorem stating that Bianca did not recycle 8 bags -/
theorem bianca_not_recycled_bags : bags_not_recycled 17 5 45 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bianca_not_recycled_bags_l754_75443


namespace NUMINAMATH_CALUDE_donut_distribution_l754_75438

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group must receive at least one item -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_distribution : distribute 4 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_l754_75438


namespace NUMINAMATH_CALUDE_prob_all_painted_10_beads_l754_75434

/-- A circular necklace with beads -/
structure Necklace :=
  (num_beads : ℕ)

/-- The number of beads selected for painting -/
def num_selected : ℕ := 5

/-- Function to calculate the probability of all beads being painted -/
noncomputable def prob_all_painted (n : Necklace) : ℚ :=
  sorry

/-- Theorem stating the probability of all beads being painted for a 10-bead necklace -/
theorem prob_all_painted_10_beads :
  prob_all_painted { num_beads := 10 } = 17 / 42 :=
sorry

end NUMINAMATH_CALUDE_prob_all_painted_10_beads_l754_75434


namespace NUMINAMATH_CALUDE_calculation_result_l754_75403

theorem calculation_result : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l754_75403


namespace NUMINAMATH_CALUDE_point_on_line_l754_75464

/-- Given a line defined by the equation x = (y / 5) - (2 / 5), 
    if two points (x, y) and (x + 3, y + 15) lie on this line, 
    then x = (y / 5) - (2 / 5) -/
theorem point_on_line (x y : ℝ) : 
  (x = y / 5 - 2 / 5) → 
  (x + 3 = (y + 15) / 5 - 2 / 5) → 
  x = y / 5 - 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l754_75464


namespace NUMINAMATH_CALUDE_find_x_l754_75404

theorem find_x (x y : ℝ) (h1 : x + 2*y = 10) (h2 : y = 4) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l754_75404


namespace NUMINAMATH_CALUDE_lcm_36_90_l754_75497

theorem lcm_36_90 : Nat.lcm 36 90 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_90_l754_75497


namespace NUMINAMATH_CALUDE_inverse_equivalent_is_contrapositive_l754_75432

theorem inverse_equivalent_is_contrapositive (p q : Prop) :
  (q → p) ↔ (¬p → ¬q) :=
sorry

end NUMINAMATH_CALUDE_inverse_equivalent_is_contrapositive_l754_75432


namespace NUMINAMATH_CALUDE_abc_equality_l754_75495

theorem abc_equality (a b c x : ℝ) 
  (h : a * x^2 - b * x - c = b * x^2 - c * x - a ∧ 
       b * x^2 - c * x - a = c * x^2 - a * x - b) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l754_75495


namespace NUMINAMATH_CALUDE_sum_of_odd_function_at_specific_points_l754_75498

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem sum_of_odd_function_at_specific_points (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_odd_function_at_specific_points_l754_75498


namespace NUMINAMATH_CALUDE_range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l754_75479

-- Define propositions p and q
def p (m : ℝ) : Prop := 2^m > Real.sqrt 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m^2 = 0 ∧ x₂^2 - 2*x₂ + m^2 = 0

-- Theorem for the first part
theorem range_when_p_and_q (m : ℝ) :
  p m ∧ q m → m > 1/2 ∧ m < 1 :=
by sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_not_p_and_q (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > -1 ∧ m ≤ 1/2) ∨ m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l754_75479


namespace NUMINAMATH_CALUDE_minimum_value_of_sum_l754_75407

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem minimum_value_of_sum (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  a 4 + a 3 = a 2 + a 1 + 8 →
  ∀ x, a 6 + a 5 ≥ x →
  x ≤ 32 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_sum_l754_75407


namespace NUMINAMATH_CALUDE_inequality_range_l754_75411

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l754_75411


namespace NUMINAMATH_CALUDE_total_nailcutter_sounds_l754_75484

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def number_of_customers : ℕ := 3

/-- The number of sounds produced per nail trimmed -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds produced for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_customer * number_of_customers * sounds_per_nail = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_nailcutter_sounds_l754_75484


namespace NUMINAMATH_CALUDE_original_number_is_ten_l754_75473

theorem original_number_is_ten : ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l754_75473


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l754_75450

/-- Given a quadratic function f(x) = ax^2 + bx + c where b is the geometric mean of a and c,
    prove that f(x) has no real roots. -/
theorem quadratic_no_roots (a b c : ℝ) (h : b^2 = a*c) (h_a : a ≠ 0) (h_c : c ≠ 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l754_75450


namespace NUMINAMATH_CALUDE_digits_used_128_l754_75478

/-- The number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ :=
  (min n 9) +
  (if n ≥ 10 then 2 * (min (n - 9) 90) else 0) +
  (if n ≥ 100 then 3 * (n - 99) else 0)

/-- The theorem stating that the number of digits used to number pages from 1 to 128 is 276 -/
theorem digits_used_128 : digits_used 128 = 276 := by
  sorry

end NUMINAMATH_CALUDE_digits_used_128_l754_75478


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l754_75425

/-- A circle is tangent to the coordinate axes and the hypotenuse of a 45-45-90 triangle -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the x-axis -/
  tangent_x : center.2 = radius
  /-- The circle is tangent to the y-axis -/
  tangent_y : center.1 = radius
  /-- The circle is tangent to the hypotenuse of the 45-45-90 triangle -/
  tangent_hypotenuse : center.1 + center.2 + radius = 2 * Real.sqrt 2

/-- The side length of the 45-45-90 triangle -/
def triangleSide : ℝ := 2

/-- The theorem stating that the radius of the tangent circle is √2 -/
theorem tangent_circle_radius :
  ∀ (c : TangentCircle), c.radius = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l754_75425


namespace NUMINAMATH_CALUDE_average_hours_worked_l754_75426

def april_hours : ℕ := 6
def june_hours : ℕ := 5
def september_hours : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

def total_hours : ℕ := april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month

theorem average_hours_worked (h : total_hours = april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month) : 
  total_hours / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_worked_l754_75426


namespace NUMINAMATH_CALUDE_oranges_per_crate_l754_75449

theorem oranges_per_crate :
  ∀ (num_crates num_boxes nectarines_per_box total_fruit : ℕ),
    num_crates = 12 →
    num_boxes = 16 →
    nectarines_per_box = 30 →
    total_fruit = 2280 →
    total_fruit = num_boxes * nectarines_per_box + num_crates * (total_fruit - num_boxes * nectarines_per_box) / num_crates →
    (total_fruit - num_boxes * nectarines_per_box) / num_crates = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_oranges_per_crate_l754_75449


namespace NUMINAMATH_CALUDE_clock_angle_at_8_30_clock_angle_at_8_30_is_75_l754_75440

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_8_30 : ℝ :=
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let hour_hand_angle : ℝ := hours * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  |hour_hand_angle - minute_hand_angle|

theorem clock_angle_at_8_30_is_75 : clock_angle_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_30_clock_angle_at_8_30_is_75_l754_75440


namespace NUMINAMATH_CALUDE_contractor_average_wage_l754_75415

def average_wage (male_count female_count child_count : ℕ)
                 (male_wage female_wage child_wage : ℚ) : ℚ :=
  let total_workers := male_count + female_count + child_count
  let total_wage := male_count * male_wage + female_count * female_wage + child_count * child_wage
  total_wage / total_workers

theorem contractor_average_wage :
  average_wage 20 15 5 25 20 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_contractor_average_wage_l754_75415


namespace NUMINAMATH_CALUDE_question_one_question_two_l754_75427

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | x^2 + (a - 1) * x - a > 0}
def B (a b : ℝ) : Set ℝ := {x | (x + a) * (x + b) > 0}
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Define the complement of B in ℝ
def C_I_B (a b : ℝ) : Set ℝ := {x | ¬((x + a) * (x + b) > 0)}

-- Theorem for question 1
theorem question_one (a b : ℝ) (h1 : a < b) (h2 : C_I_B a b = M) : 
  a = -1 ∧ b = 3 := by sorry

-- Theorem for question 2
theorem question_two (a b : ℝ) (h : a > b ∧ b > -1) : 
  A a ∩ B a b = {x | x < -a ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_question_one_question_two_l754_75427


namespace NUMINAMATH_CALUDE_no_solutions_for_2500_l754_75463

theorem no_solutions_for_2500 : ¬∃ (a₂ a₀ : ℤ), 
  2500 = a₂ * 10^4 + a₀ ∧ 
  0 ≤ a₂ ∧ a₂ ≤ 9 ∧ 
  0 ≤ a₀ ∧ a₀ ≤ 1000 := by
sorry

end NUMINAMATH_CALUDE_no_solutions_for_2500_l754_75463


namespace NUMINAMATH_CALUDE_largest_colorable_3subsets_correct_l754_75441

/-- The largest number of 3-subsets that can be chosen from a set of n elements
    such that there always exists a 2-coloring with no monochromatic chosen 3-subset -/
def largest_colorable_3subsets (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 4
  else if n = 5 ∨ n = 6 then 9
  else if n ≥ 7 then 6
  else 0

/-- The theorem stating the correct values for the largest number of colorable 3-subsets -/
theorem largest_colorable_3subsets_correct (n : ℕ) (h : n ≥ 3) :
  largest_colorable_3subsets n =
    if n = 3 then 1
    else if n = 4 then 4
    else if n = 5 ∨ n = 6 then 9
    else 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_colorable_3subsets_correct_l754_75441


namespace NUMINAMATH_CALUDE_gcd_13924_32451_l754_75416

theorem gcd_13924_32451 : Nat.gcd 13924 32451 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13924_32451_l754_75416


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l754_75446

theorem exists_subset_with_unique_sum_representation : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l754_75446


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l754_75482

/-- Given that Tyler has 8 aquariums and 512 saltwater animals in total,
    prove that there are 64 animals in each aquarium. -/
theorem animals_per_aquarium (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 8) (h2 : total_animals = 512) :
  total_animals / num_aquariums = 64 := by
  sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l754_75482


namespace NUMINAMATH_CALUDE_inequality_preservation_l754_75445

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l754_75445


namespace NUMINAMATH_CALUDE_fifth_occurrence_of_three_sevenths_l754_75494

/-- Represents a fraction with numerator and denominator -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ+

/-- The sequence of fractions as described in the problem -/
def fractionSequence : ℕ → Fraction := sorry

/-- Two fractions are equivalent if their cross products are equal -/
def areEquivalent (f1 f2 : Fraction) : Prop :=
  f1.numerator * f2.denominator = f2.numerator * f1.denominator

/-- The position of the nth occurrence of a fraction equivalent to the given fraction -/
def positionOfNthOccurrence (f : Fraction) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fifth_occurrence_of_three_sevenths :
  positionOfNthOccurrence ⟨3, 7⟩ 5 = 1211 := by sorry

end NUMINAMATH_CALUDE_fifth_occurrence_of_three_sevenths_l754_75494


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l754_75418

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, 2 * x^2 - 7 * x - 30 < 0 ↔ -5/2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l754_75418


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l754_75457

/-- The x-coordinate of a point P on the x-axis that is equidistant from A(-3, 0) and B(0, 5) is 8/3 -/
theorem equidistant_point_x_coordinate : 
  ∃ x : ℝ, 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ y : ℝ, ((-3 - x)^2 + y^2 = x^2 + (5 - y)^2) → y = 0) ∧
    x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l754_75457


namespace NUMINAMATH_CALUDE_minimum_marbles_to_add_proof_minimum_marbles_l754_75420

theorem minimum_marbles_to_add (initial_marbles : Nat) (people : Nat) : Nat :=
  let additional_marbles := people - initial_marbles % people
  if additional_marbles = people then 0 else additional_marbles

theorem proof_minimum_marbles :
  minimum_marbles_to_add 62 8 = 2 ∧
  (62 + minimum_marbles_to_add 62 8) % 8 = 0 ∧
  ∀ x : Nat, x < minimum_marbles_to_add 62 8 → (62 + x) % 8 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_minimum_marbles_to_add_proof_minimum_marbles_l754_75420


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l754_75475

theorem power_fraction_simplification :
  (1 : ℝ) / ((-5^4)^2) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l754_75475


namespace NUMINAMATH_CALUDE_total_wrappers_eq_49_l754_75413

/-- The number of wrappers gathered by Andy -/
def andy_wrappers : ℕ := 34

/-- The number of wrappers gathered by Max -/
def max_wrappers : ℕ := 15

/-- The total number of wrappers gathered by Andy and Max -/
def total_wrappers : ℕ := andy_wrappers + max_wrappers

theorem total_wrappers_eq_49 : total_wrappers = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_wrappers_eq_49_l754_75413


namespace NUMINAMATH_CALUDE_sales_theorem_l754_75448

def sales_problem (sale1 sale2 sale4 sale5 average : ℕ) : Prop :=
  let total := average * 5
  let known_sales := sale1 + sale2 + sale4 + sale5
  let sale3 := total - known_sales
  sale3 = 9455

theorem sales_theorem :
  sales_problem 5700 8550 3850 14045 7800 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_theorem_l754_75448


namespace NUMINAMATH_CALUDE_complete_square_l754_75471

theorem complete_square (x : ℝ) : (x^2 - 5*x = 31) → ((x - 5/2)^2 = 149/4) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_complete_square_l754_75471


namespace NUMINAMATH_CALUDE_power_equation_solution_l754_75430

theorem power_equation_solution (p : ℕ) : 64^5 = 8^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l754_75430


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l754_75453

/-- Given that the solution set of ax² + 2x + c < 0 is (-∞, -1/3) ∪ (1/2, +∞),
    prove that the solution set of cx² + 2x + a ≤ 0 is [-3, 2]. -/
theorem solution_set_equivalence 
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2))
  (a c : ℝ) :
  ∀ x : ℝ, (c*x^2 + 2*x + a ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l754_75453


namespace NUMINAMATH_CALUDE_minimum_balls_for_16_of_one_color_l754_75405

theorem minimum_balls_for_16_of_one_color : 
  let total_balls : ℕ := 21 + 17 + 24 + 10 + 14 + 14
  let red_balls : ℕ := 21
  let green_balls : ℕ := 17
  let yellow_balls : ℕ := 24
  let blue_balls : ℕ := 10
  let white_balls : ℕ := 14
  let black_balls : ℕ := 14
  ∃ (n : ℕ), n = 84 ∧ 
    (∀ (m : ℕ), m < n → 
      ∃ (r g y b w bl : ℕ), 
        r ≤ red_balls ∧ 
        g ≤ green_balls ∧ 
        y ≤ yellow_balls ∧ 
        b ≤ blue_balls ∧ 
        w ≤ white_balls ∧ 
        bl ≤ black_balls ∧ 
        r + g + y + b + w + bl = m ∧ 
        r < 16 ∧ g < 16 ∧ y < 16 ∧ b < 16 ∧ w < 16 ∧ bl < 16) ∧
    (∀ (k : ℕ), k ≥ n → 
      ∃ (color : ℕ), color ≥ 16 ∧ 
        (color ≤ red_balls ∨ 
         color ≤ green_balls ∨ 
         color ≤ yellow_balls ∨ 
         color ≤ blue_balls ∨ 
         color ≤ white_balls ∨ 
         color ≤ black_balls))
:= by sorry

end NUMINAMATH_CALUDE_minimum_balls_for_16_of_one_color_l754_75405


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l754_75476

theorem arithmetic_mean_difference (p q r : ℝ) : 
  (p + q) / 2 = 10 → (q + r) / 2 = 24 → r - p = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l754_75476


namespace NUMINAMATH_CALUDE_ratio_cubes_equals_twentyseven_l754_75422

theorem ratio_cubes_equals_twentyseven : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_cubes_equals_twentyseven_l754_75422


namespace NUMINAMATH_CALUDE_distance_between_points_l754_75437

/-- The distance between points (4, -6) and (-8, 5) is √265 -/
theorem distance_between_points : Real.sqrt 265 = Real.sqrt ((4 - (-8))^2 + ((-6) - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l754_75437


namespace NUMINAMATH_CALUDE_max_difference_with_broken_calculator_l754_75442

def is_valid_digit (d : ℕ) (valid_digits : List ℕ) : Prop :=
  d ∈ valid_digits

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_difference_with_broken_calculator :
  ∀ (a b c d e f : ℕ),
    is_valid_digit a [3, 5, 9] →
    is_valid_digit b [2, 3, 7] →
    is_valid_digit c [3, 4, 8, 9] →
    is_valid_digit d [2, 3, 7] →
    is_valid_digit e [3, 5, 9] →
    is_valid_digit f [1, 4, 7] →
    is_three_digit_number (100 * a + 10 * b + c) →
    is_three_digit_number (100 * d + 10 * e + f) →
    is_three_digit_number ((100 * a + 10 * b + c) - (100 * d + 10 * e + f)) →
    (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≤ 529 ∧
    (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4 →
      ∀ (x y z u v w : ℕ),
        is_valid_digit x [3, 5, 9] →
        is_valid_digit y [2, 3, 7] →
        is_valid_digit z [3, 4, 8, 9] →
        is_valid_digit u [2, 3, 7] →
        is_valid_digit v [3, 5, 9] →
        is_valid_digit w [1, 4, 7] →
        is_three_digit_number (100 * x + 10 * y + z) →
        is_three_digit_number (100 * u + 10 * v + w) →
        is_three_digit_number ((100 * x + 10 * y + z) - (100 * u + 10 * v + w)) →
        (100 * x + 10 * y + z) - (100 * u + 10 * v + w) ≤ (100 * a + 10 * b + c) - (100 * d + 10 * e + f)) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_with_broken_calculator_l754_75442


namespace NUMINAMATH_CALUDE_circle_properties_l754_75423

/-- Theorem about a circle's properties given a specific sum of circumference, diameter, and radius -/
theorem circle_properties (r : ℝ) (h : 2 * Real.pi * r + 2 * r + r = 27.84) : 
  2 * r = 6 ∧ Real.pi * r^2 = 28.26 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_circle_properties_l754_75423


namespace NUMINAMATH_CALUDE_ship_passengers_l754_75408

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l754_75408


namespace NUMINAMATH_CALUDE_trig_simplification_l754_75444

theorem trig_simplification (x : ℝ) :
  (1 + Real.sin (x + 30 * π / 180) - Real.cos (x + 30 * π / 180)) /
  (1 + Real.sin (x + 30 * π / 180) + Real.cos (x + 30 * π / 180)) =
  Real.tan (x / 2 + 15 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l754_75444


namespace NUMINAMATH_CALUDE_joan_lost_balloons_l754_75468

/-- Given that Joan initially had 8 orange balloons and now has 6,
    prove that she lost 2 balloons. -/
theorem joan_lost_balloons (initial : ℕ) (current : ℕ) (h1 : initial = 8) (h2 : current = 6) :
  initial - current = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_lost_balloons_l754_75468


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l754_75456

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + i^75 = -2*i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l754_75456


namespace NUMINAMATH_CALUDE_negative_inequality_l754_75489

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l754_75489


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l754_75433

theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l754_75433


namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l754_75460

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l754_75460


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l754_75466

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l754_75466


namespace NUMINAMATH_CALUDE_expansion_equals_difference_of_squares_l754_75461

theorem expansion_equals_difference_of_squares (x y : ℝ) : 
  (5*y - x) * (-5*y - x) = x^2 - 25*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_difference_of_squares_l754_75461


namespace NUMINAMATH_CALUDE_homomorphism_characterization_l754_75431

theorem homomorphism_characterization (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y) →
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_homomorphism_characterization_l754_75431


namespace NUMINAMATH_CALUDE_probability_point_not_in_inner_square_l754_75400

/-- The probability that a random point in a larger square is not in a smaller square inside it. -/
theorem probability_point_not_in_inner_square
  (area_A : ℝ) (perimeter_B : ℝ)
  (h_area_A : area_A = 65)
  (h_perimeter_B : perimeter_B = 16)
  (h_positive_A : area_A > 0)
  (h_positive_B : perimeter_B > 0) :
  let side_B := perimeter_B / 4
  let area_B := side_B ^ 2
  (area_A - area_B) / area_A = (65 - 16) / 65 := by
  sorry


end NUMINAMATH_CALUDE_probability_point_not_in_inner_square_l754_75400


namespace NUMINAMATH_CALUDE_decreasing_geometric_sequence_characterization_l754_75458

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem decreasing_geometric_sequence_characterization
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (a 1 > a 2 ∧ a 2 > a 3) ↔ is_decreasing_sequence a :=
by sorry

end NUMINAMATH_CALUDE_decreasing_geometric_sequence_characterization_l754_75458


namespace NUMINAMATH_CALUDE_casper_candy_problem_l754_75491

theorem casper_candy_problem (initial_candies : ℚ) : 
  let day1_remaining := (3/4) * initial_candies - 3
  let day2_remaining := (4/5) * day1_remaining - 5
  let day3_remaining := day2_remaining - 10
  day3_remaining = 10 → initial_candies = 224/3 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l754_75491


namespace NUMINAMATH_CALUDE_team_B_better_image_l754_75459

-- Define the structure for a team
structure Team where
  members : ℕ
  avg_height : ℝ
  height_variance : ℝ

-- Define the two teams
def team_A : Team := { members := 20, avg_height := 160, height_variance := 10.5 }
def team_B : Team := { members := 20, avg_height := 160, height_variance := 1.2 }

-- Define a function to determine which team has a better performance image
def better_performance_image (t1 t2 : Team) : Prop :=
  t1.avg_height = t2.avg_height ∧ t1.height_variance < t2.height_variance

-- Theorem statement
theorem team_B_better_image : 
  better_performance_image team_B team_A :=
sorry

end NUMINAMATH_CALUDE_team_B_better_image_l754_75459


namespace NUMINAMATH_CALUDE_max_sum_nonnegative_l754_75462

theorem max_sum_nonnegative (a b c d : ℝ) (h : a + b + c + d = 0) :
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_nonnegative_l754_75462


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l754_75490

/-- Given a fractional equation and conditions on its solution, 
    this theorem proves the range of values for the parameter a. -/
theorem fractional_equation_solution_range (a x : ℝ) : 
  (a / (x + 2) = 1 - 3 / (x + 2)) →
  (x < 0) →
  (a < -1 ∧ a ≠ -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l754_75490


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l754_75414

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol 
    results in a solution that is 50% alcohol. -/
theorem alcohol_concentration_after_addition 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_alcohol : ℝ) 
  (target_concentration : ℝ) : 
  initial_volume = 6 →
  initial_concentration = 0.3 →
  added_alcohol = 2.4 →
  target_concentration = 0.5 →
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = target_concentration := by
  sorry

#check alcohol_concentration_after_addition

end NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l754_75414


namespace NUMINAMATH_CALUDE_distinct_z_values_l754_75402

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values :
  ∃ (S : Finset ℕ), (∀ x, is_two_digit x → z x ∈ S) ∧ S.card = 10 :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l754_75402


namespace NUMINAMATH_CALUDE_sin_510_degrees_l754_75454

theorem sin_510_degrees : Real.sin (510 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_510_degrees_l754_75454


namespace NUMINAMATH_CALUDE_volunteer_assignment_l754_75485

-- Define the number of volunteers
def num_volunteers : ℕ := 5

-- Define the number of venues
def num_venues : ℕ := 3

-- Define the function to calculate the number of ways to assign volunteers
def ways_to_assign (volunteers : ℕ) (venues : ℕ) : ℕ :=
  venues^volunteers - venues * (venues - 1)^volunteers

-- Theorem statement
theorem volunteer_assignment :
  ways_to_assign num_volunteers num_venues = 147 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_l754_75485


namespace NUMINAMATH_CALUDE_greater_number_problem_l754_75470

theorem greater_number_problem (a b : ℕ+) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l754_75470


namespace NUMINAMATH_CALUDE_esha_lag_behind_anusha_l754_75477

/-- Represents the runners in the race -/
inductive Runner
| Anusha
| Banu
| Esha

/-- The race parameters and conditions -/
structure RaceConditions where
  race_length : ℝ
  speeds : Runner → ℝ
  anusha_fastest : speeds Runner.Anusha > speeds Runner.Banu ∧ speeds Runner.Banu > speeds Runner.Esha
  banu_lag : speeds Runner.Banu / speeds Runner.Anusha = 9 / 10
  esha_lag : speeds Runner.Esha / speeds Runner.Banu = 9 / 10

/-- The theorem to be proved -/
theorem esha_lag_behind_anusha (rc : RaceConditions) (h : rc.race_length = 100) :
  rc.race_length - (rc.speeds Runner.Esha / rc.speeds Runner.Anusha) * rc.race_length = 19 := by
  sorry

end NUMINAMATH_CALUDE_esha_lag_behind_anusha_l754_75477


namespace NUMINAMATH_CALUDE_seventieth_number_is_557_l754_75492

/-- The nth positive integer that leaves a remainder of 5 when divided by 8 -/
def nth_number (n : ℕ) : ℕ := 8 * (n - 1) + 5

/-- Proposition: The 70th positive integer that leaves a remainder of 5 when divided by 8 is 557 -/
theorem seventieth_number_is_557 : nth_number 70 = 557 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_number_is_557_l754_75492


namespace NUMINAMATH_CALUDE_problem_solution_l754_75480

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : x = 9) : 
  y = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l754_75480


namespace NUMINAMATH_CALUDE_area_between_circles_l754_75439

/-- The area of the region between two concentric circles with given radii and a tangent chord --/
theorem area_between_circles (R r c : ℝ) (hR : R = 60) (hr : r = 40) (hc : c = 100)
  (h_concentric : R > r) (h_tangent : c^2 = 4 * (R^2 - r^2)) :
  π * (R^2 - r^2) = 2000 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l754_75439


namespace NUMINAMATH_CALUDE_oatmeal_raisin_percentage_l754_75455

/-- Given a class of students and cookie distribution, calculate the percentage of students who want oatmeal raisin cookies. -/
theorem oatmeal_raisin_percentage 
  (total_students : ℕ) 
  (cookies_per_student : ℕ) 
  (oatmeal_raisin_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : cookies_per_student = 2)
  (h3 : oatmeal_raisin_cookies = 8) : 
  (oatmeal_raisin_cookies / cookies_per_student) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_raisin_percentage_l754_75455


namespace NUMINAMATH_CALUDE_estate_distribution_l754_75435

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) 
  (h1 : ∃ (d s w c : ℕ), d + s + w + c = E) 
  (h2 : ∃ (d s : ℕ), d + s = E / 2) 
  (h3 : ∃ (d s : ℕ), 3 * s = 2 * d) 
  (h4 : ∃ (d w : ℕ), w = 3 * d) 
  (h5 : ∃ (c : ℕ), c = 800) :
  E = 2000 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_l754_75435


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l754_75409

theorem quadratic_equation_real_root (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + Complex.I) * x - 2 - k * Complex.I = 0) → 
  (k = 1 ∨ k = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l754_75409


namespace NUMINAMATH_CALUDE_truck_initial_momentum_l754_75410

/-- Initial momentum of a truck -/
theorem truck_initial_momentum
  (v : ℝ) -- Initial velocity
  (F : ℝ) -- Constant force applied to stop the truck
  (x : ℝ) -- Distance traveled before stopping
  (t : ℝ) -- Time taken to stop
  (h1 : v > 0) -- Assumption: initial velocity is positive
  (h2 : F > 0) -- Assumption: force is positive
  (h3 : x > 0) -- Assumption: distance is positive
  (h4 : t > 0) -- Assumption: time is positive
  (h5 : x = (v * t) / 2) -- Relation between distance, velocity, and time
  (h6 : F * t = v) -- Relation between force, time, and velocity change
  : ∃ (m : ℝ), m * v = (2 * F * x) / v :=
sorry

end NUMINAMATH_CALUDE_truck_initial_momentum_l754_75410


namespace NUMINAMATH_CALUDE_cost_of_treats_treats_cost_is_twelve_l754_75451

/-- Calculates the cost of a bag of treats given the total spent and other expenses --/
theorem cost_of_treats (puppy_cost : ℝ) (dog_food : ℝ) (toys : ℝ) (crate : ℝ) (bed : ℝ) (collar_leash : ℝ) 
  (discount_rate : ℝ) (total_spent : ℝ) : ℝ :=
  let other_items := dog_food + toys + crate + bed + collar_leash
  let discounted_other_items := other_items * (1 - discount_rate)
  let treats_total := total_spent - puppy_cost - discounted_other_items
  treats_total / 2

/-- Proves that the cost of a bag of treats is $12.00 --/
theorem treats_cost_is_twelve : 
  cost_of_treats 20 20 15 20 20 15 0.2 96 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_treats_treats_cost_is_twelve_l754_75451


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l754_75469

/-- The angle of inclination of the tangent line to y = x^3 - 2x + 4 at (1, 3) is 45°. -/
theorem tangent_line_inclination (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 - 2*x + 4 →
  x₀ = 1 →
  y₀ = 3 →
  f x₀ = y₀ →
  HasDerivAt f (3*x₀^2 - 2) x₀ →
  (Real.arctan (3*x₀^2 - 2)) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l754_75469


namespace NUMINAMATH_CALUDE_ellipse_equation_l754_75429

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Half-length of the major axis
  b : ℝ  -- Half-length of the minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem: Given the specified conditions, prove the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.a + e.b = 9)  -- Sum of half-lengths of axes is 18/2 = 9
    (h2 : e.c = 3)        -- One focus is at (3, 0)
    (h3 : e.c^2 = e.a^2 - e.b^2)  -- Relationship between a, b, and c
    : standard_equation e = λ x y ↦ x^2 / 25 + y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l754_75429


namespace NUMINAMATH_CALUDE_fruit_ratio_l754_75424

theorem fruit_ratio (initial_fruits : ℕ) (oranges_left : ℕ) : 
  initial_fruits = 150 → 
  oranges_left = 50 → 
  (oranges_left : ℚ) / (initial_fruits / 2 - oranges_left : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_ratio_l754_75424


namespace NUMINAMATH_CALUDE_sum_357_eq_42_l754_75487

/-- A geometric sequence with first term 3 and the sum of the first, third, and fifth terms equal to 21 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 3
  sum_135 : a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the geometric sequence is 42 -/
theorem sum_357_eq_42 (seq : GeometricSequence) : seq.a 3 + seq.a 5 + seq.a 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_357_eq_42_l754_75487


namespace NUMINAMATH_CALUDE_square_area_increase_l754_75496

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.05 * s
  let original_area := s ^ 2
  let new_area := new_side ^ 2
  (new_area - original_area) / original_area = 0.1025 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l754_75496


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l754_75486

theorem perfect_square_quadratic (c : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 14*x + c = y^2) → c = 49 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l754_75486


namespace NUMINAMATH_CALUDE_fruit_candy_cost_difference_l754_75428

/-- The cost difference between two schools purchasing fruit candy --/
theorem fruit_candy_cost_difference : 
  let school_a_quantity : ℝ := 56
  let school_a_price_per_kg : ℝ := 8.06
  let price_reduction : ℝ := 0.56
  let free_candy_percentage : ℝ := 0.05
  
  let school_b_price_per_kg : ℝ := school_a_price_per_kg - price_reduction
  let school_b_quantity : ℝ := school_a_quantity / (1 + free_candy_percentage)
  
  let school_a_total_cost : ℝ := school_a_quantity * school_a_price_per_kg
  let school_b_total_cost : ℝ := school_b_quantity * school_b_price_per_kg
  
  school_a_total_cost - school_b_total_cost = 51.36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_candy_cost_difference_l754_75428


namespace NUMINAMATH_CALUDE_solution_l754_75465

def problem (f : ℝ → ℝ) : Prop :=
  (∀ x, f x * f (x + 2) = 13) ∧ f 1 = 2

theorem solution (f : ℝ → ℝ) (h : problem f) : f 2015 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_l754_75465


namespace NUMINAMATH_CALUDE_trees_in_yard_l754_75417

/-- The number of trees planted along a yard -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem: There are 24 trees planted along the yard -/
theorem trees_in_yard :
  let yard_length : ℕ := 414
  let tree_distance : ℕ := 18
  num_trees yard_length tree_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l754_75417


namespace NUMINAMATH_CALUDE_oranges_per_group_l754_75421

theorem oranges_per_group (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_group_l754_75421


namespace NUMINAMATH_CALUDE_race_track_radius_l754_75419

/-- Given a circular race track, prove that the radius of the outer circle
    is equal to the radius of the inner circle plus the width of the track. -/
theorem race_track_radius 
  (inner_circumference : ℝ) 
  (track_width : ℝ) 
  (inner_circumference_eq : inner_circumference = 880) 
  (track_width_eq : track_width = 18) :
  ∃ (outer_radius : ℝ),
    outer_radius = inner_circumference / (2 * Real.pi) + track_width :=
by sorry

end NUMINAMATH_CALUDE_race_track_radius_l754_75419


namespace NUMINAMATH_CALUDE_least_value_theorem_l754_75467

theorem least_value_theorem (x y z : ℕ+) 
  (h1 : 5 * y.val = 6 * z.val)
  (h2 : x.val + y.val + z.val = 26) :
  5 * y.val = 30 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l754_75467


namespace NUMINAMATH_CALUDE_profit_margin_ratio_l754_75481

/-- Prove that for an article with selling price S, cost C, and profit margin M = (1/n)S, 
    the ratio of M to C is equal to 1/(n-1) -/
theorem profit_margin_ratio (n : ℝ) (S : ℝ) (C : ℝ) (M : ℝ) 
    (h1 : n ≠ 0) 
    (h2 : n ≠ 1)
    (h3 : M = (1/n) * S) 
    (h4 : C = S - M) : 
  M / C = 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_ratio_l754_75481


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l754_75472

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l754_75472


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l754_75474

theorem complex_fraction_equality : (10 * Complex.I) / (2 - Complex.I) = -2 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l754_75474


namespace NUMINAMATH_CALUDE_cubic_two_intersections_l754_75401

/-- A cubic function that intersects the x-axis at exactly two points -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_two_intersections :
  ∃! a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ∧
  a = -4/27 :=
sorry

end NUMINAMATH_CALUDE_cubic_two_intersections_l754_75401


namespace NUMINAMATH_CALUDE_triangle_third_side_l754_75483

theorem triangle_third_side (a b : ℝ) (h₁ h₂ : ℝ) :
  a = 5 →
  b = 2 * Real.sqrt 6 →
  0 < h₁ →
  0 < h₂ →
  a * h₁ = b * h₂ →
  a + h₁ ≤ b + h₂ →
  ∃ c : ℝ, c * c = a * a + b * b ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l754_75483


namespace NUMINAMATH_CALUDE_present_price_l754_75499

theorem present_price (original_price : ℝ) (discount_rate : ℝ) (num_people : ℕ) 
  (individual_savings : ℝ) :
  original_price > 0 →
  discount_rate = 0.2 →
  num_people = 3 →
  individual_savings = 4 →
  original_price * (1 - discount_rate) = num_people * individual_savings →
  original_price * (1 - discount_rate) = 48 := by
sorry

end NUMINAMATH_CALUDE_present_price_l754_75499


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l754_75493

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2016 - 1) = 2^8 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l754_75493


namespace NUMINAMATH_CALUDE_condition_for_equation_l754_75412

theorem condition_for_equation (x y z : ℤ) : x = y ∧ y = z → x * (x - y) + y * (y - z) + z * (z - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_condition_for_equation_l754_75412


namespace NUMINAMATH_CALUDE_smallest_sticker_count_l754_75406

theorem smallest_sticker_count (S : ℕ) (h1 : S > 3) 
  (h2 : S % 5 = 3) (h3 : S % 11 = 3) (h4 : S % 13 = 3) : 
  S ≥ 718 ∧ ∃ (T : ℕ), T = 718 ∧ T % 5 = 3 ∧ T % 11 = 3 ∧ T % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sticker_count_l754_75406


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l754_75452

theorem fraction_product_simplification :
  (20 : ℚ) / 21 * 35 / 48 * 84 / 55 * 11 / 40 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l754_75452


namespace NUMINAMATH_CALUDE_same_grade_probability_l754_75436

-- Define the number of student volunteers in each grade
def grade_A_volunteers : ℕ := 240
def grade_B_volunteers : ℕ := 160
def grade_C_volunteers : ℕ := 160

-- Define the total number of student volunteers
def total_volunteers : ℕ := grade_A_volunteers + grade_B_volunteers + grade_C_volunteers

-- Define the number of students to be selected using stratified sampling
def selected_students : ℕ := 7

-- Define the number of students to be chosen for sanitation work
def sanitation_workers : ℕ := 2

-- Define the function to calculate the number of students selected from each grade
def students_per_grade (grade_volunteers : ℕ) : ℕ :=
  (grade_volunteers * selected_students) / total_volunteers

-- Theorem: The probability of selecting 2 students from the same grade is 5/21
theorem same_grade_probability :
  (students_per_grade grade_A_volunteers) * (students_per_grade grade_A_volunteers - 1) / 2 +
  (students_per_grade grade_B_volunteers) * (students_per_grade grade_B_volunteers - 1) / 2 +
  (students_per_grade grade_C_volunteers) * (students_per_grade grade_C_volunteers - 1) / 2 =
  5 * (selected_students * (selected_students - 1) / 2) / 21 :=
by sorry

end NUMINAMATH_CALUDE_same_grade_probability_l754_75436
