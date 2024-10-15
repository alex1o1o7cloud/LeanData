import Mathlib

namespace NUMINAMATH_CALUDE_larger_fraction_l4005_400530

theorem larger_fraction (x y : ℚ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/4) :
  max x y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_larger_fraction_l4005_400530


namespace NUMINAMATH_CALUDE_fraction_order_l4005_400557

theorem fraction_order : (6/17)^2 < 8/25 ∧ 8/25 < 10/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l4005_400557


namespace NUMINAMATH_CALUDE_percentage_failed_both_l4005_400587

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 35

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 80

/-- The theorem stating the percentage of students who failed in both subjects -/
theorem percentage_failed_both :
  100 - passed_both = failed_hindi + failed_english - 40 := by sorry

end NUMINAMATH_CALUDE_percentage_failed_both_l4005_400587


namespace NUMINAMATH_CALUDE_b_used_car_for_10_hours_l4005_400548

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  aHours : ℕ
  cHours : ℕ
  bPaid : ℕ

/-- Calculates the number of hours b used the car -/
def calculateBHours (ch : CarHire) : ℕ :=
  let totalHours := ch.aHours + ch.cHours + (ch.bPaid * (ch.aHours + ch.cHours) / (ch.totalCost - ch.bPaid))
  ch.bPaid * totalHours / ch.totalCost

/-- Theorem stating that given the conditions, b used the car for 10 hours -/
theorem b_used_car_for_10_hours (ch : CarHire)
  (h1 : ch.totalCost = 720)
  (h2 : ch.aHours = 9)
  (h3 : ch.cHours = 13)
  (h4 : ch.bPaid = 225) :
  calculateBHours ch = 10 := by
  sorry

#eval calculateBHours ⟨720, 9, 13, 225⟩

end NUMINAMATH_CALUDE_b_used_car_for_10_hours_l4005_400548


namespace NUMINAMATH_CALUDE_resulting_temperature_correct_l4005_400500

/-- The resulting temperature when rising from 5°C to t°C -/
def resulting_temperature (t : ℝ) : ℝ := 5 + t

/-- Theorem stating that the resulting temperature is correct -/
theorem resulting_temperature_correct (t : ℝ) : 
  resulting_temperature t = 5 + t := by sorry

end NUMINAMATH_CALUDE_resulting_temperature_correct_l4005_400500


namespace NUMINAMATH_CALUDE_time_is_one_point_two_hours_l4005_400505

/-- The number of letters in the name -/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_all_rearrangements : ℚ :=
  (name_length.factorial / rearrangements_per_minute : ℚ) / minutes_per_hour

/-- Theorem stating that the time to write all rearrangements is 1.2 hours -/
theorem time_is_one_point_two_hours :
  time_to_write_all_rearrangements = 6/5 := by sorry

end NUMINAMATH_CALUDE_time_is_one_point_two_hours_l4005_400505


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l4005_400564

/-- Represents the points on the circle -/
inductive Point
| one | two | three | four | five | six | seven

/-- Calculates the next point based on the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.four
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.one
  | Point.six => Point.two
  | Point.seven => Point.three

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2023_jumps :
  jumpNTimes Point.seven 2023 = Point.one :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l4005_400564


namespace NUMINAMATH_CALUDE_equation_solution_l4005_400515

theorem equation_solution : ∃ x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4005_400515


namespace NUMINAMATH_CALUDE_average_notebooks_sold_per_day_l4005_400543

theorem average_notebooks_sold_per_day 
  (total_bundles : ℕ) 
  (days : ℕ) 
  (notebooks_per_bundle : ℕ) 
  (h1 : total_bundles = 15) 
  (h2 : days = 5) 
  (h3 : notebooks_per_bundle = 40) : 
  (total_bundles * notebooks_per_bundle) / days = 120 :=
by sorry

end NUMINAMATH_CALUDE_average_notebooks_sold_per_day_l4005_400543


namespace NUMINAMATH_CALUDE_students_in_neither_art_nor_music_l4005_400567

theorem students_in_neither_art_nor_music
  (total : ℕ)
  (art : ℕ)
  (music : ℕ)
  (both : ℕ)
  (h1 : total = 60)
  (h2 : art = 40)
  (h3 : music = 30)
  (h4 : both = 15) :
  total - (art + music - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_in_neither_art_nor_music_l4005_400567


namespace NUMINAMATH_CALUDE_square_area_ratio_l4005_400561

theorem square_area_ratio : 
  let small_side : ℝ := 5
  let large_side : ℝ := small_side + 5
  let small_area : ℝ := small_side ^ 2
  let large_area : ℝ := large_side ^ 2
  large_area / small_area = 4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l4005_400561


namespace NUMINAMATH_CALUDE_three_digit_congruence_count_l4005_400586

theorem three_digit_congruence_count :
  (∃ (S : Finset Nat), 
    (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧
    (∀ x ∈ S, (4897 * x + 603) % 29 = 1427 % 29) ∧
    S.card = 28) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_count_l4005_400586


namespace NUMINAMATH_CALUDE_expo_arrangement_plans_l4005_400599

/-- Represents the number of volunteers --/
def total_volunteers : ℕ := 5

/-- Represents the number of pavilions to be assigned --/
def pavilions_to_assign : ℕ := 3

/-- Represents the number of volunteers who cannot be assigned to a specific pavilion --/
def restricted_volunteers : ℕ := 2

/-- Represents the total number of arrangement plans --/
def total_arrangements : ℕ := 36

/-- Theorem stating the total number of arrangement plans --/
theorem expo_arrangement_plans :
  (total_volunteers = 5) →
  (pavilions_to_assign = 3) →
  (restricted_volunteers = 2) →
  (total_arrangements = 36) :=
by sorry

end NUMINAMATH_CALUDE_expo_arrangement_plans_l4005_400599


namespace NUMINAMATH_CALUDE_equilateral_triangle_point_distances_l4005_400516

theorem equilateral_triangle_point_distances 
  (h x y z : ℝ) 
  (h_pos : h > 0)
  (inside_triangle : x > 0 ∧ y > 0 ∧ z > 0)
  (height_sum : h = x + y + z)
  (triangle_inequality : x + y > z ∧ y + z > x ∧ z + x > y) :
  x < h/2 ∧ y < h/2 ∧ z < h/2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_point_distances_l4005_400516


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l4005_400503

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 + x - 3 < 0
def inequality2 (x : ℝ) : Prop := x * (9 - x) > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | 0 < x ∧ x < 9}

-- Theorem statements
theorem solution_inequality1 : {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : {x : ℝ | inequality2 x} = solution_set2 := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l4005_400503


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l4005_400510

/-- 
Given a scooter purchase, we define the following:
purchase_price: The initial cost of the scooter
selling_price: The price at which the scooter was sold
gain_percent: The percentage gain on the sale
repair_cost: The amount spent on repairs

We prove that the repair cost satisfies the equation relating these variables.
-/
theorem scooter_repair_cost 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (gain_percent : ℝ) 
  (repair_cost : ℝ) 
  (h1 : purchase_price = 4400)
  (h2 : selling_price = 5800)
  (h3 : gain_percent = 0.1154) :
  selling_price = (purchase_price + repair_cost) * (1 + gain_percent) :=
by sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l4005_400510


namespace NUMINAMATH_CALUDE_shower_water_usage_l4005_400594

/-- Calculates the total water usage for showers over a given period --/
def total_water_usage (weeks : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let showers := days / 2
  let total_minutes := showers * shower_duration
  total_minutes * water_per_minute

theorem shower_water_usage : total_water_usage 4 10 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_shower_water_usage_l4005_400594


namespace NUMINAMATH_CALUDE_product_sum_fractions_l4005_400532

theorem product_sum_fractions : (3 * 4 * 5) * (1 / 3 + 1 / 4 - 1 / 5) = 23 := by sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l4005_400532


namespace NUMINAMATH_CALUDE_pie_eating_contest_l4005_400507

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 5/6) (hb : b = 2/3) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c)) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l4005_400507


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l4005_400573

def f (x : ℝ) := 2 * abs (x + 1) + abs (x - 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    b^2 / a + c^2 / b + a^2 / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l4005_400573


namespace NUMINAMATH_CALUDE_target_same_type_as_reference_l4005_400546

/-- Represents a monomial term with variables x and y -/
structure Monomial :=
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- Determines if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  m1.x_exp = m2.x_exp ∧ m1.y_exp = m2.y_exp

/-- The reference monomial 3x²y -/
def reference : Monomial :=
  ⟨2, 1⟩

/-- The monomial -yx² -/
def target : Monomial :=
  ⟨2, 1⟩

theorem target_same_type_as_reference : same_type target reference :=
  sorry

end NUMINAMATH_CALUDE_target_same_type_as_reference_l4005_400546


namespace NUMINAMATH_CALUDE_line_equation_l4005_400572

/-- Given a line l: ax + by + 1 = 0 and a circle x² + y² - 6y + 5 = 0, 
    this theorem proves that the line l is x - y + 3 = 0 
    if it's the axis of symmetry of the circle and perpendicular to x + y + 2 = 0 -/
theorem line_equation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → 
    (x^2 + y^2 - 6*y + 5 = 0 → 
      (∃ c : ℝ, c * (a * x + b * y + 1) = x^2 + y^2 - 6*y + 5))) → 
  (a * 1 + b * 1 = -1) → 
  (a * x + b * y + 1 = 0 ↔ x - y + 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l4005_400572


namespace NUMINAMATH_CALUDE_apps_deleted_l4005_400540

theorem apps_deleted (initial_apps final_apps : ℕ) (h1 : initial_apps = 12) (h2 : final_apps = 4) :
  initial_apps - final_apps = 8 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l4005_400540


namespace NUMINAMATH_CALUDE_inequality_proof_l4005_400514

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + a / b) * (1 + b / c) * (1 + c / a) ≥ 2 * (1 + (a + b + c) / Real.rpow (a * b * c) (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4005_400514


namespace NUMINAMATH_CALUDE_digit_difference_when_reversed_l4005_400536

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem digit_difference_when_reversed (n : ThreeDigitNumber) 
  (h : (n.reversed_value - n.value : ℚ) / 10 = 19.8) : 
  n.hundreds - n.units = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_when_reversed_l4005_400536


namespace NUMINAMATH_CALUDE_rectangle_ratio_l4005_400519

/-- A configuration of squares and a rectangle forming a large square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  largeSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 3 times the small square's side -/
  large_square : largeSide = 3 * s
  /-- The rectangle's length is 3 times the small square's side -/
  rect_length : rectLength = 3 * s
  /-- The rectangle's width is 2 times the small square's side -/
  rect_width : rectWidth = 2 * s

/-- The ratio of the rectangle's length to its width is 3/2 -/
theorem rectangle_ratio (config : SquareConfiguration) :
  config.rectLength / config.rectWidth = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l4005_400519


namespace NUMINAMATH_CALUDE_teddy_material_in_tons_l4005_400556

/-- The amount of fluffy foam material Teddy uses for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The number of pillows Teddy can make -/
def number_of_pillows : ℕ := 3000

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating the amount of fluffy foam material Teddy has in tons -/
theorem teddy_material_in_tons : 
  (material_per_pillow * number_of_pillows) / pounds_per_ton = 3 := by
  sorry

end NUMINAMATH_CALUDE_teddy_material_in_tons_l4005_400556


namespace NUMINAMATH_CALUDE_tangent_points_distance_circle_fixed_point_l4005_400589

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define a point on the directrix
structure PointOnDirectrix where
  x : ℝ
  y : ℝ
  on_directrix : directrix x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the tangent line from a point on the directrix to the parabola
def tangent_line (P : PointOnDirectrix) (Q : PointOnParabola) : Prop :=
  ∃ k : ℝ, Q.y = k * (Q.x - P.x)

-- Theorem 1: Distance between tangent points when P is on x-axis
theorem tangent_points_distance :
  ∀ (P : PointOnDirectrix) (Q R : PointOnParabola),
  P.y = 0 →
  tangent_line P Q →
  tangent_line P R →
  Q ≠ R →
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 16 :=
sorry

-- Theorem 2: Circle with diameter PQ passes through (1, 0)
theorem circle_fixed_point :
  ∀ (P : PointOnDirectrix) (Q : PointOnParabola),
  tangent_line P Q →
  ∃ (r : ℝ),
    (1 - ((P.x + Q.x) / 2))^2 + (0 - ((P.y + Q.y) / 2))^2 = r^2 ∧
    (P.x - ((P.x + Q.x) / 2))^2 + (P.y - ((P.y + Q.y) / 2))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_distance_circle_fixed_point_l4005_400589


namespace NUMINAMATH_CALUDE_investment_solution_l4005_400522

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def investment_problem (principal : ℝ) : Prop :=
  let year1_amount := compound_interest principal 0.05
  let year2_amount := compound_interest year1_amount 0.07
  let year3_amount := compound_interest year2_amount 0.04
  year3_amount = 1232

theorem investment_solution :
  ∃ (principal : ℝ), investment_problem principal ∧ 
    (principal ≥ 1054.75 ∧ principal ≤ 1054.77) :=
by
  sorry

#check investment_solution

end NUMINAMATH_CALUDE_investment_solution_l4005_400522


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l4005_400579

theorem greatest_whole_number_inequality (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(6*y - 5 < 7 - 3*y)) → 
  (6*x - 5 < 7 - 3*x) → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l4005_400579


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_l4005_400596

def is_finite_decimal (n : ℚ) : Prop :=
  ∃ (a b : ℤ), n = a / (2^b * 5^b)

theorem finite_decimal_fraction :
  (is_finite_decimal (9/12)) ∧
  (¬ is_finite_decimal (11/27)) ∧
  (¬ is_finite_decimal (4/7)) ∧
  (¬ is_finite_decimal (8/15)) :=
by sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_l4005_400596


namespace NUMINAMATH_CALUDE_union_of_sets_l4005_400575

def A (a b : ℝ) : Set ℝ := {5, b/a, a-b}
def B (a b : ℝ) : Set ℝ := {b, a+b, -1}

theorem union_of_sets (a b : ℝ) (h : A a b ∩ B a b = {2, -1}) :
  A a b ∪ B a b = {-1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l4005_400575


namespace NUMINAMATH_CALUDE_cheese_problem_l4005_400581

theorem cheese_problem (k : ℕ) (h1 : k > 7) : ∃ (initial : ℕ), initial = 11 ∧ 
  (10 : ℚ) / k + 7 * ((5 : ℚ) / k) = initial ∧ (35 : ℕ) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cheese_problem_l4005_400581


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l4005_400511

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (2 * a^2 - 7 * a + 3 = 0) → 
             (2 * b^2 - 7 * b + 3 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l4005_400511


namespace NUMINAMATH_CALUDE_prob_three_even_dice_l4005_400534

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of even outcomes on each die -/
def num_even_sides : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even_dice : ℕ := 3

/-- The probability of exactly three dice showing even numbers when six fair 12-sided dice are rolled -/
theorem prob_three_even_dice : 
  (num_dice.choose num_even_dice * (num_even_sides / num_sides) ^ num_even_dice * 
  ((num_sides - num_even_sides) / num_sides) ^ (num_dice - num_even_dice) : ℚ) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_dice_l4005_400534


namespace NUMINAMATH_CALUDE_queen_then_club_probability_l4005_400541

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the number of clubs in a standard deck
def numClubs : ℕ := 13

-- Define the probability of drawing a Queen first and a club second
def probQueenThenClub : ℚ := 1 / 52

-- Theorem statement
theorem queen_then_club_probability :
  probQueenThenClub = (numQueens / standardDeck) * (numClubs / (standardDeck - 1)) :=
by sorry

end NUMINAMATH_CALUDE_queen_then_club_probability_l4005_400541


namespace NUMINAMATH_CALUDE_tangent_intersection_of_specific_circles_l4005_400559

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The x-coordinate of the intersection point of a line tangent to two circles -/
def tangentIntersection (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_of_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (18, 0), radius := 8 }
  tangentIntersection c1 c2 = 54 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_of_specific_circles_l4005_400559


namespace NUMINAMATH_CALUDE_sum_even_probability_l4005_400583

/-- Represents a wheel in the game -/
structure Wheel where
  probability_even : ℝ

/-- The game with two wheels -/
structure Game where
  wheel_a : Wheel
  wheel_b : Wheel

/-- A fair wheel has equal probability of landing on even or odd numbers -/
def is_fair (w : Wheel) : Prop := w.probability_even = 1/2

/-- Probability of the sum of two wheels being even -/
def prob_sum_even (g : Game) : ℝ :=
  g.wheel_a.probability_even * g.wheel_b.probability_even +
  (1 - g.wheel_a.probability_even) * (1 - g.wheel_b.probability_even)

theorem sum_even_probability (g : Game) 
  (h1 : is_fair g.wheel_a) 
  (h2 : g.wheel_b.probability_even = 2/3) : 
  prob_sum_even g = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_probability_l4005_400583


namespace NUMINAMATH_CALUDE_train_length_l4005_400598

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 10 → speed * time * (1000 / 3600) = 175 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4005_400598


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_l4005_400542

theorem imaginary_part_of_one_plus_i :
  Complex.im (1 + Complex.I) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_l4005_400542


namespace NUMINAMATH_CALUDE_triangle_angles_calculation_l4005_400549

-- Define the triangle and its properties
def Triangle (A B C : ℝ) (C_ext : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = 180 ∧
  C_ext = A + B

-- Theorem statement
theorem triangle_angles_calculation 
  (A B C C_ext : ℝ) 
  (h : Triangle A B C C_ext) 
  (hA : A = 64) 
  (hB : B = 33) 
  (hC_ext : C_ext = 120) :
  C = 83 ∧ ∃ D, D = 56 ∧ C_ext = A + D :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_calculation_l4005_400549


namespace NUMINAMATH_CALUDE_eighteenth_replacement_november_l4005_400570

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months since January to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth battery replacement, given replacements occur every 7 months starting from January -/
def batteryReplacementMonth (n : ℕ) : Month :=
  monthsToMonth (7 * (n - 1) + 1)

theorem eighteenth_replacement_november :
  batteryReplacementMonth 18 = Month.November := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_november_l4005_400570


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l4005_400560

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : blue_students = 63)
  (h3 : red_students = 81)
  (h4 : total_pairs = 72)
  (h5 : blue_blue_pairs = 29)
  (h6 : total_students = blue_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  (red_students - (total_students - blue_blue_pairs * 2 - blue_students)) / 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l4005_400560


namespace NUMINAMATH_CALUDE_sequence_must_be_finite_l4005_400524

def is_valid_sequence (c : ℕ+) (p : ℕ → ℕ) : Prop :=
  ∀ k, k ≥ 1 →
    Nat.Prime (p k) ∧
    (p (k + 1)) ∣ (p k + c) ∧
    ∀ i, 1 ≤ i ∧ i < k + 1 → p (k + 1) ≠ p i

theorem sequence_must_be_finite (c : ℕ+) :
  ¬∃ p : ℕ → ℕ, is_valid_sequence c p ∧ (∀ n, ∃ k > n, p k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_must_be_finite_l4005_400524


namespace NUMINAMATH_CALUDE_projection_result_l4005_400577

/-- A projection that takes [2, -4] to [3, -3] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The projection satisfies the given condition -/
axiom projection_condition : projection (2, -4) = (3, -3)

/-- Theorem: The projection takes [3, 5] to [-1, 1] -/
theorem projection_result : projection (3, 5) = (-1, 1) := by sorry

end NUMINAMATH_CALUDE_projection_result_l4005_400577


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l4005_400529

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 3 * (180 - x) → (180 - x = 135 ∧ 90 - x = 45) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l4005_400529


namespace NUMINAMATH_CALUDE_parallel_vector_sum_l4005_400535

/-- Given two vectors in ℝ², prove that if one is parallel to their sum, then the first component of the second vector is 1/2. -/
theorem parallel_vector_sum (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.2 = 1) :
  (∃ (k : ℝ), b = k • (a + b)) → b.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_l4005_400535


namespace NUMINAMATH_CALUDE_divisible_by_6_up_to_88_eq_l4005_400502

def divisible_by_6_up_to_88 : Set ℕ :=
  {n | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_eq :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_6_up_to_88_eq_l4005_400502


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l4005_400547

theorem set_inclusion_equivalence (a : ℤ) : 
  let A := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 32}
  (A ⊆ A ∩ B ∧ A.Nonempty) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l4005_400547


namespace NUMINAMATH_CALUDE_equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l4005_400508

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  num_divisions_pos : num_divisions > 0

/-- Theorem: An equilateral triangle can be divided into 25 smaller equilateral triangles -/
theorem equilateral_triangle_25_division (t : EquilateralTriangle) :
  ∃ (d : TriangleDivision), d.original = t ∧ d.num_divisions = 25 :=
sorry

/-- Represents a grouping of the divided triangles -/
structure TriangleGrouping where
  division : TriangleDivision
  num_groups : ℕ
  num_groups_pos : num_groups > 0
  triangles_per_group : ℕ
  triangles_per_group_pos : triangles_per_group > 0
  valid_grouping : division.num_divisions = num_groups * triangles_per_group

/-- Theorem: The 25 smaller triangles can be grouped into 5 equal parts -/
theorem equilateral_triangle_5_equal_parts (t : EquilateralTriangle) :
  ∃ (g : TriangleGrouping), g.division.original = t ∧ g.num_groups = 5 ∧ g.triangles_per_group = 5 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l4005_400508


namespace NUMINAMATH_CALUDE_cards_in_boxes_l4005_400597

/-- The number of ways to distribute n distinct objects into k distinct boxes with no box left empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 cards and 3 boxes -/
def num_cards : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem to prove -/
theorem cards_in_boxes : distribute num_cards num_boxes = 36 := by sorry

end NUMINAMATH_CALUDE_cards_in_boxes_l4005_400597


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l4005_400518

/-- A trinomial ax^2 + bx + c is a perfect square if there exists r such that ax^2 + bx + c = (rx + s)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r s : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (r * x + s)^2

/-- If x^2 + kx + 81 is a perfect square trinomial, then k = 18 or k = -18 -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 k 81 → k = 18 ∨ k = -18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l4005_400518


namespace NUMINAMATH_CALUDE_hexagon_circumference_hexagon_circumference_proof_l4005_400576

/-- The circumference of a regular hexagon with side length 5 centimeters is 30 centimeters. -/
theorem hexagon_circumference : ℝ → Prop :=
  fun side_length =>
    side_length = 5 →
    (6 : ℝ) * side_length = 30

-- The proof is omitted
theorem hexagon_circumference_proof : hexagon_circumference 5 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_circumference_hexagon_circumference_proof_l4005_400576


namespace NUMINAMATH_CALUDE_grade_distribution_l4005_400571

theorem grade_distribution (n : Nat) (k : Nat) : 
  n = 12 ∧ k = 3 → 
  (k^n : Nat) - k * ((k-1)^n : Nat) + (k * (k-2)^n : Nat) = 519156 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l4005_400571


namespace NUMINAMATH_CALUDE_largest_package_size_l4005_400537

theorem largest_package_size (anna beatrice carlos : ℕ) 
  (h1 : anna = 60) (h2 : beatrice = 45) (h3 : carlos = 75) :
  Nat.gcd anna (Nat.gcd beatrice carlos) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l4005_400537


namespace NUMINAMATH_CALUDE_potato_problem_result_l4005_400506

/-- Represents the potato problem --/
structure PotatoProblem where
  totalPotatoes : Nat
  potatoesForWedges : Nat
  wedgesPerPotato : Nat
  chipsPerPotato : Nat

/-- Calculates the difference between potato chips and wedges --/
def chipWedgeDifference (p : PotatoProblem) : Nat :=
  let remainingPotatoes := p.totalPotatoes - p.potatoesForWedges
  let potatoesForChips := remainingPotatoes / 2
  let totalChips := potatoesForChips * p.chipsPerPotato
  let totalWedges := p.potatoesForWedges * p.wedgesPerPotato
  totalChips - totalWedges

/-- Theorem stating the result of the potato problem --/
theorem potato_problem_result :
  let p : PotatoProblem := {
    totalPotatoes := 67,
    potatoesForWedges := 13,
    wedgesPerPotato := 8,
    chipsPerPotato := 20
  }
  chipWedgeDifference p = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_problem_result_l4005_400506


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l4005_400569

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Define the perpendicularity condition
def is_perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_k (k : ℝ) :
  is_perpendicular 
    (fun i => k * (a i) + (b i)) 
    (fun i => (a i) - 3 * (b i)) 
  → k = 19 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l4005_400569


namespace NUMINAMATH_CALUDE_min_value_of_y_l4005_400525

theorem min_value_of_y (x : ℝ) (h1 : x > 3) :
  let y := x + 1 / (x - 3)
  ∀ z, y ≥ z → z ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l4005_400525


namespace NUMINAMATH_CALUDE_quadratic_negative_root_condition_l4005_400553

/-- The quadratic equation ax^2 + 2x + 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop := a * x^2 + 2 * x + 1 = 0

/-- A root of the quadratic equation is negative -/
def has_negative_root (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ quadratic_equation a x

theorem quadratic_negative_root_condition :
  (∀ a : ℝ, a < 0 → has_negative_root a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ has_negative_root a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_condition_l4005_400553


namespace NUMINAMATH_CALUDE_degenerate_ellipse_l4005_400554

/-- An ellipse is degenerate if and only if it consists of a single point -/
theorem degenerate_ellipse (x y c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * (p.1)^2 + (p.2)^2 + 6 * p.1 - 12 * p.2 + c = 0) ↔ c = -39 := by
  sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_l4005_400554


namespace NUMINAMATH_CALUDE_afternoon_rowing_count_l4005_400533

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 35

/-- The total number of campers who went rowing -/
def total_campers : ℕ := 62

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := total_campers - morning_campers

theorem afternoon_rowing_count : afternoon_campers = 27 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowing_count_l4005_400533


namespace NUMINAMATH_CALUDE_C_symmetric_C_area_inequality_C_perimeter_inequality_l4005_400591

-- Define the curve C
def C (a : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 1 ∧ (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) = a^2)

-- Define the fixed points
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Theorem for symmetry
theorem C_symmetric (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P ↔ C a (-P.1, -P.2) := by sorry

-- Theorem for area inequality
theorem C_area_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    (1/2 * Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) * 
      Real.sin (Real.arccos ((P.1 + 1) * (P.1 - 1) + P.2^2) / 
        (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2))))
    ≤ (1/2) * a^2 := by sorry

-- Theorem for perimeter inequality
theorem C_perimeter_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    Real.sqrt ((P.1 + 1)^2 + P.2^2) + Real.sqrt ((P.1 - 1)^2 + P.2^2) + 2 ≥ 2*a + 2 := by sorry

end NUMINAMATH_CALUDE_C_symmetric_C_area_inequality_C_perimeter_inequality_l4005_400591


namespace NUMINAMATH_CALUDE_solution_set_equality_l4005_400520

theorem solution_set_equality : {x : ℤ | (3*x - 1)*(x + 3) = 0} = {-3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l4005_400520


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4005_400531

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4005_400531


namespace NUMINAMATH_CALUDE_ice_cream_ratio_l4005_400545

theorem ice_cream_ratio (sunday pints : ℕ) (k : ℕ) : 
  sunday = 4 →
  let monday := k * sunday
  let tuesday := monday / 3
  let wednesday := tuesday / 2
  18 = sunday + monday + tuesday - wednesday →
  monday / sunday = 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_l4005_400545


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4005_400580

theorem polynomial_simplification (p : ℝ) : 
  (5 * p^4 - 4 * p^3 + 3 * p - 7) + (8 - 9 * p^2 + p^4 + 6 * p) = 
  6 * p^4 - 4 * p^3 - 9 * p^2 + 9 * p + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4005_400580


namespace NUMINAMATH_CALUDE_square_sum_implies_sum_l4005_400565

theorem square_sum_implies_sum (x : ℝ) (h : x > 0) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = 3 → x + x⁻¹ = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_sum_l4005_400565


namespace NUMINAMATH_CALUDE_power_zero_equals_one_l4005_400501

theorem power_zero_equals_one (x : ℝ) (hx : x ≠ 0) : x ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equals_one_l4005_400501


namespace NUMINAMATH_CALUDE_job_completion_time_l4005_400582

/-- Given two workers can finish a job in 15 days and a third worker can finish the job in 30 days,
    prove that all three workers together can finish the job in 10 days. -/
theorem job_completion_time 
  (work_rate_ab : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_ab = 1 / 15) 
  (h2 : work_rate_c = 1 / 30) : 
  1 / (work_rate_ab + work_rate_c) = 10 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l4005_400582


namespace NUMINAMATH_CALUDE_oscar_class_count_l4005_400517

/-- The number of questions per student on the final exam -/
def questions_per_student : ℕ := 10

/-- The number of students per class -/
def students_per_class : ℕ := 35

/-- The total number of questions to review -/
def total_questions : ℕ := 1750

/-- The number of classes Professor Oscar has -/
def number_of_classes : ℕ := total_questions / (questions_per_student * students_per_class)

theorem oscar_class_count :
  number_of_classes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oscar_class_count_l4005_400517


namespace NUMINAMATH_CALUDE_slope_of_line_l4005_400595

/-- The slope of the line 6x + 10y = 30 is -3/5 -/
theorem slope_of_line (x y : ℝ) : 6 * x + 10 * y = 30 → (y - 3 = (-3 / 5) * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l4005_400595


namespace NUMINAMATH_CALUDE_prob_sum_seven_l4005_400574

/-- A type representing the possible outcomes of a single dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- The type of outcomes when throwing a dice twice -/
def TwoThrows := DiceOutcome × DiceOutcome

/-- The sum of points for a pair of dice outcomes -/
def sum_points (throw : TwoThrows) : Nat :=
  match throw with
  | (DiceOutcome.one, b) => 1 + DiceOutcome.toNat b
  | (DiceOutcome.two, b) => 2 + DiceOutcome.toNat b
  | (DiceOutcome.three, b) => 3 + DiceOutcome.toNat b
  | (DiceOutcome.four, b) => 4 + DiceOutcome.toNat b
  | (DiceOutcome.five, b) => 5 + DiceOutcome.toNat b
  | (DiceOutcome.six, b) => 6 + DiceOutcome.toNat b
where
  DiceOutcome.toNat : DiceOutcome → Nat
    | DiceOutcome.one => 1
    | DiceOutcome.two => 2
    | DiceOutcome.three => 3
    | DiceOutcome.four => 4
    | DiceOutcome.five => 5
    | DiceOutcome.six => 6

/-- The set of all possible outcomes when throwing a dice twice -/
def all_outcomes : Finset TwoThrows := sorry

/-- The set of outcomes where the sum of points is 7 -/
def sum_seven_outcomes : Finset TwoThrows := sorry

theorem prob_sum_seven : 
  (Finset.card sum_seven_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_sum_seven_l4005_400574


namespace NUMINAMATH_CALUDE_third_month_sale_l4005_400550

def sale_problem (m1 m2 m4 m5 m6 avg : ℕ) : Prop :=
  let total := avg * 6
  let known_sum := m1 + m2 + m4 + m5 + m6
  total - known_sum = 6200

theorem third_month_sale :
  sale_problem 5420 5660 6350 6500 6470 6100 :=
sorry

end NUMINAMATH_CALUDE_third_month_sale_l4005_400550


namespace NUMINAMATH_CALUDE_evergreen_elementary_grade6_l4005_400512

theorem evergreen_elementary_grade6 (total : ℕ) (grade4 : ℕ) (grade5 : ℕ) 
  (h1 : total = 100)
  (h2 : grade4 = 30)
  (h3 : grade5 = 35) :
  total - grade4 - grade5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_evergreen_elementary_grade6_l4005_400512


namespace NUMINAMATH_CALUDE_number_from_hcf_lcm_and_other_l4005_400593

theorem number_from_hcf_lcm_and_other (a b : ℕ+) : 
  Nat.gcd a b = 12 →
  Nat.lcm a b = 396 →
  b = 99 →
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_number_from_hcf_lcm_and_other_l4005_400593


namespace NUMINAMATH_CALUDE_min_value_of_u_l4005_400509

theorem min_value_of_u (a b : ℝ) (h : 3*a^2 - 10*a*b + 8*b^2 + 5*a - 10*b = 0) :
  ∃ (u_min : ℝ), u_min = -34 ∧ ∀ (u : ℝ), u = 9*a^2 + 72*b + 2 → u ≥ u_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_u_l4005_400509


namespace NUMINAMATH_CALUDE_pet_store_cats_l4005_400538

theorem pet_store_cats (siamese_cats : ℕ) (sold_cats : ℕ) (remaining_cats : ℕ) (house_cats : ℕ) : 
  siamese_cats = 13 → 
  sold_cats = 10 → 
  remaining_cats = 8 → 
  siamese_cats + house_cats - sold_cats = remaining_cats → 
  house_cats = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l4005_400538


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l4005_400555

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1)
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l4005_400555


namespace NUMINAMATH_CALUDE_tom_candy_proof_l4005_400526

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10
def final_candy : ℕ := 19

theorem tom_candy_proof :
  initial_candy + friend_candy + bought_candy = final_candy :=
by sorry

end NUMINAMATH_CALUDE_tom_candy_proof_l4005_400526


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4005_400588

-- Define the sets A and B
def A : Set ℝ := {x | Real.log x ≥ 0}
def B : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4005_400588


namespace NUMINAMATH_CALUDE_smallest_disguisable_triangle_two_sides_perfect_squares_l4005_400578

/-- A triangle with integer side lengths a, b, and c is disguisable if there exists a similar triangle
    with side lengths d, a, b where d ≥ a ≥ b > c -/
def IsDisguisableTriangle (a b c : ℕ) : Prop :=
  ∃ d : ℚ, d ≥ a ∧ a ≥ b ∧ b > c ∧ (d : ℚ) / a = (a : ℚ) / b ∧ (a : ℚ) / b = (b : ℚ) / c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℕ) : ℕ := a + b + c

/-- A number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_disguisable_triangle :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    Perimeter a b c ≥ 19 ∧
    (Perimeter a b c = 19 → (a, b, c) = (9, 6, 4)) :=
sorry

theorem two_sides_perfect_squares :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    (∀ k : ℕ, k < a → ¬IsDisguisableTriangle k (k * b / a) (k * c / a)) →
    (IsPerfectSquare a ∧ IsPerfectSquare c) ∨
    (IsPerfectSquare a ∧ IsPerfectSquare b) ∨
    (IsPerfectSquare b ∧ IsPerfectSquare c) :=
sorry

end NUMINAMATH_CALUDE_smallest_disguisable_triangle_two_sides_perfect_squares_l4005_400578


namespace NUMINAMATH_CALUDE_carol_carrot_count_l4005_400566

/-- The number of carrots Carol picked -/
def carols_carrots : ℝ := 29.0

/-- The number of carrots Carol's mom picked -/
def moms_carrots : ℝ := 16.0

/-- The number of carrots they picked together -/
def joint_carrots : ℝ := 38.0

/-- The total number of carrots picked -/
def total_carrots : ℝ := 83.0

theorem carol_carrot_count : 
  carols_carrots + moms_carrots + joint_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_carol_carrot_count_l4005_400566


namespace NUMINAMATH_CALUDE_katrina_lunch_sales_l4005_400544

/-- The number of cookies sold during the lunch rush -/
def lunch_rush_sales (initial : ℕ) (morning_dozens : ℕ) (afternoon : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12) - afternoon - remaining

/-- Proof that Katrina sold 57 cookies during the lunch rush -/
theorem katrina_lunch_sales :
  lunch_rush_sales 120 3 16 11 = 57 := by
  sorry

end NUMINAMATH_CALUDE_katrina_lunch_sales_l4005_400544


namespace NUMINAMATH_CALUDE_a_4_equals_20_l4005_400584

def sequence_a (n : ℕ) : ℕ := n^2 + n

theorem a_4_equals_20 : sequence_a 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_20_l4005_400584


namespace NUMINAMATH_CALUDE_lcm_of_24_36_40_l4005_400558

theorem lcm_of_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_24_36_40_l4005_400558


namespace NUMINAMATH_CALUDE_largest_fraction_l4005_400513

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l4005_400513


namespace NUMINAMATH_CALUDE_amount_spent_on_sweets_l4005_400585

-- Define the initial amount
def initial_amount : ℚ := 5.10

-- Define the amount given to each friend
def amount_per_friend : ℚ := 1.00

-- Define the number of friends
def number_of_friends : ℕ := 2

-- Define the final amount left
def final_amount : ℚ := 2.05

-- Theorem to prove the amount spent on sweets
theorem amount_spent_on_sweets :
  initial_amount - (amount_per_friend * number_of_friends) - final_amount = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_amount_spent_on_sweets_l4005_400585


namespace NUMINAMATH_CALUDE_triangle_side_length_l4005_400539

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 7)
  (h2 : c = 6)
  (h3 : Real.cos (B - C) = 37/40)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h6 : A + B + C = π) :
  a = Real.sqrt 66.1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4005_400539


namespace NUMINAMATH_CALUDE_december_spending_fraction_l4005_400552

def monthly_savings (month : Nat) : Rat :=
  match month with
  | 1 => 1/10
  | 2 => 3/25
  | 3 => 3/20
  | 4 => 1/5
  | m => if m ≤ 12 then (14 + m)/100 else 0

def total_savings : Rat :=
  (List.range 12).map (λ m => monthly_savings (m + 1)) |> List.sum

theorem december_spending_fraction :
  total_savings = 4 * (1 - monthly_savings 12) →
  1 - monthly_savings 12 = 39/50 := by
  sorry

end NUMINAMATH_CALUDE_december_spending_fraction_l4005_400552


namespace NUMINAMATH_CALUDE_study_time_for_desired_average_l4005_400563

/-- Represents the relationship between study time and test score -/
structure StudyRelationship where
  time : ℝ
  score : ℝ
  k : ℝ
  inverse_prop : score * time = k

/-- Represents two tests with their study times and scores -/
structure TwoTests where
  test1 : StudyRelationship
  test2 : StudyRelationship
  avg_score : ℝ
  avg_constraint : (test1.score + test2.score) / 2 = avg_score

/-- The main theorem to prove -/
theorem study_time_for_desired_average (tests : TwoTests) :
  tests.test1.time = 6 ∧
  tests.test1.score = 80 ∧
  tests.avg_score = 85 →
  tests.test2.time = 16 / 3 :=
by sorry

end NUMINAMATH_CALUDE_study_time_for_desired_average_l4005_400563


namespace NUMINAMATH_CALUDE_popcorn_package_solution_l4005_400504

/-- Represents a package of popcorn buckets -/
structure Package where
  buckets : ℕ
  cost : ℚ

/-- Proves that buying 48 packages of Package B satisfies all conditions -/
theorem popcorn_package_solution :
  let package_b : Package := ⟨9, 8⟩
  let num_packages : ℕ := 48
  let total_buckets : ℕ := num_packages * package_b.buckets
  let total_cost : ℚ := num_packages * package_b.cost
  (total_buckets ≥ 426) ∧ 
  (total_cost ≤ 400) ∧ 
  (num_packages ≤ 60) :=
by
  sorry


end NUMINAMATH_CALUDE_popcorn_package_solution_l4005_400504


namespace NUMINAMATH_CALUDE_total_photos_taken_is_46_l4005_400528

/-- Represents the number of photos on Toby's camera roll at different stages --/
structure PhotoCount where
  initial : Nat
  deletedBadShots : Nat
  catPictures : Nat
  deletedAfterEditing : Nat
  additionalShots : Nat
  secondSession : Nat
  thirdSession : Nat
  deletedFromSecond : Nat
  deletedFromThird : Nat
  final : Nat

/-- Calculates the total number of photos taken in all photo shoots --/
def totalPhotosTaken (p : PhotoCount) : Nat :=
  let firstSessionPhotos := p.final - p.initial + p.deletedBadShots - p.catPictures + 
                            p.deletedAfterEditing - p.additionalShots - 
                            (p.secondSession - p.deletedFromSecond) - 
                            (p.thirdSession - p.deletedFromThird)
  firstSessionPhotos + p.secondSession + p.thirdSession

/-- Theorem stating that the total number of photos taken in all photo shoots is 46 --/
theorem total_photos_taken_is_46 (p : PhotoCount) 
  (h1 : p.initial = 63)
  (h2 : p.deletedBadShots = 7)
  (h3 : p.catPictures = 15)
  (h4 : p.deletedAfterEditing = 3)
  (h5 : p.additionalShots = 5)
  (h6 : p.secondSession = 11)
  (h7 : p.thirdSession = 8)
  (h8 : p.deletedFromSecond = 6)
  (h9 : p.deletedFromThird = 4)
  (h10 : p.final = 112) :
  totalPhotosTaken p = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_taken_is_46_l4005_400528


namespace NUMINAMATH_CALUDE_competition_end_time_l4005_400590

-- Define the start time of the competition
def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight

-- Define the duration of the competition
def duration : Nat := 875  -- in minutes

-- Define the end time of the competition
def end_time : Nat := (start_time + duration) % (24 * 60)

-- Theorem to prove
theorem competition_end_time :
  end_time = 5 * 60 + 35  -- 5:35 a.m. in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_competition_end_time_l4005_400590


namespace NUMINAMATH_CALUDE_medical_staff_composition_l4005_400527

theorem medical_staff_composition :
  ∀ (a b c d : ℕ),
    a + b + c + d = 17 →
    a + b ≥ c + d →
    d > a →
    a > b →
    c ≥ 2 →
    a = 5 ∧ b = 4 ∧ c = 2 ∧ d = 6 :=
by sorry

end NUMINAMATH_CALUDE_medical_staff_composition_l4005_400527


namespace NUMINAMATH_CALUDE_stormi_car_wash_l4005_400523

/-- Proves the number of cars Stormi washed to save for a bicycle --/
theorem stormi_car_wash : 
  ∀ (car_wash_price lawn_mow_income bicycle_price additional_needed : ℕ),
  car_wash_price = 10 →
  lawn_mow_income = 26 →
  bicycle_price = 80 →
  additional_needed = 24 →
  (bicycle_price - additional_needed - lawn_mow_income) / car_wash_price = 3 := by
sorry

end NUMINAMATH_CALUDE_stormi_car_wash_l4005_400523


namespace NUMINAMATH_CALUDE_power_function_through_point_l4005_400551

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 → ∀ x > 0, f x = x ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4005_400551


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l4005_400568

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 8 →
  eliminated_boxes = 14 →
  (high_value_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l4005_400568


namespace NUMINAMATH_CALUDE_power_function_inequality_l4005_400521

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  f ((x₁ + x₂)/2) > (f x₁ + f x₂)/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l4005_400521


namespace NUMINAMATH_CALUDE_total_spent_on_cards_l4005_400592

-- Define the prices and tax rates
def football_card_price : ℝ := 2.73
def football_card_tax_rate : ℝ := 0.05
def football_card_quantity : ℕ := 2

def pokemon_card_price : ℝ := 4.01
def pokemon_card_tax_rate : ℝ := 0.08

def baseball_card_original_price : ℝ := 10
def baseball_card_discount_rate : ℝ := 0.10
def baseball_card_tax_rate : ℝ := 0.06

-- Calculate the total cost
def total_cost : ℝ :=
  -- Football cards
  (football_card_price * football_card_quantity) * (1 + football_card_tax_rate) +
  -- Pokemon cards
  pokemon_card_price * (1 + pokemon_card_tax_rate) +
  -- Baseball cards
  (baseball_card_original_price * (1 - baseball_card_discount_rate)) * (1 + baseball_card_tax_rate)

-- Theorem statement
theorem total_spent_on_cards :
  total_cost = 19.6038 := by sorry

end NUMINAMATH_CALUDE_total_spent_on_cards_l4005_400592


namespace NUMINAMATH_CALUDE_two_numbers_difference_l4005_400562

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23210 →
  b % 5 = 0 →
  a = 2 * (b / 10) →
  b - a = 15480 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l4005_400562
