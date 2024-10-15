import Mathlib

namespace NUMINAMATH_CALUDE_kelly_initial_bracelets_l228_22848

/-- Proves that Kelly initially had 16 bracelets given the problem conditions -/
theorem kelly_initial_bracelets :
  ∀ (k : ℕ), -- k represents Kelly's initial number of bracelets
  let b_initial : ℕ := 5 -- Bingley's initial number of bracelets
  let b_after_kelly : ℕ := b_initial + k / 4 -- Bingley's bracelets after receiving from Kelly
  let b_final : ℕ := b_after_kelly * 2 / 3 -- Bingley's final number of bracelets
  b_final = 6 → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_bracelets_l228_22848


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l228_22831

theorem triangle_angle_measure (P Q R : ℝ) : 
  P = 88 → 
  Q = 2 * R + 18 → 
  P + Q + R = 180 → 
  R = 74 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l228_22831


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l228_22885

/-- Given a school with students taking electives, prove the percentage taking music. -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200)
  : (total_students - dance_students - art_students) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l228_22885


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l228_22886

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 3 / 10) :
  a / c = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l228_22886


namespace NUMINAMATH_CALUDE_village_revenue_comparison_l228_22846

def village_a : List ℝ := [5, 6, 6, 7, 8, 16]
def village_b : List ℝ := [4, 6, 8, 9, 10, 17]

theorem village_revenue_comparison :
  (village_a.sum / village_a.length) < (village_b.sum / village_b.length) := by
  sorry

end NUMINAMATH_CALUDE_village_revenue_comparison_l228_22846


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_l228_22856

theorem quadratic_equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 + 2 * k * x + 1 = 0) ↔ (k = -2 ∨ k = -1 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_l228_22856


namespace NUMINAMATH_CALUDE_zero_det_necessary_not_sufficient_for_parallel_l228_22854

/-- Represents a line in the Cartesian plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The determinant of the coefficients of two lines -/
def coeff_det (l₁ l₂ : Line) : ℝ :=
  l₁.a * l₂.b - l₂.a * l₁.b

/-- Theorem stating that zero determinant is necessary but not sufficient for parallel lines -/
theorem zero_det_necessary_not_sufficient_for_parallel (l₁ l₂ : Line) :
  (are_parallel l₁ l₂ → coeff_det l₁ l₂ = 0) ∧
  ¬(coeff_det l₁ l₂ = 0 → are_parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_zero_det_necessary_not_sufficient_for_parallel_l228_22854


namespace NUMINAMATH_CALUDE_equation_system_ratio_l228_22844

theorem equation_system_ratio (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_system_ratio_l228_22844


namespace NUMINAMATH_CALUDE_representatives_selection_count_l228_22822

def male_students : ℕ := 6
def female_students : ℕ := 3
def total_students : ℕ := male_students + female_students
def representatives : ℕ := 4

theorem representatives_selection_count :
  (Nat.choose total_students representatives) - (Nat.choose male_students representatives) = 111 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_count_l228_22822


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l228_22875

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y^3 + y ≤ x - x^3) : y < x ∧ x < 1 ∧ x^2 + y^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l228_22875


namespace NUMINAMATH_CALUDE_baseball_team_groups_l228_22818

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 5) :
  (new_players + returning_players) / players_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l228_22818


namespace NUMINAMATH_CALUDE_pond_freezes_on_seventh_day_l228_22804

/-- Represents a rectangular pond with given dimensions and freezing properties -/
structure Pond where
  length : ℝ
  width : ℝ
  daily_freeze_distance : ℝ
  first_day_freeze_percent : ℝ
  second_day_freeze_percent : ℝ

/-- Calculates the day when the pond is completely frozen -/
def freezing_day (p : Pond) : ℕ :=
  sorry

/-- Theorem stating that the pond will be completely frozen on the 7th day -/
theorem pond_freezes_on_seventh_day (p : Pond) 
  (h1 : p.length * p.width = 5000)
  (h2 : p.length + p.width = 70.5)
  (h3 : p.daily_freeze_distance = 10)
  (h4 : p.first_day_freeze_percent = 0.202)
  (h5 : p.second_day_freeze_percent = 0.186) : 
  freezing_day p = 7 :=
sorry

end NUMINAMATH_CALUDE_pond_freezes_on_seventh_day_l228_22804


namespace NUMINAMATH_CALUDE_debby_water_bottles_l228_22882

/-- The number of water bottles Debby drinks per day -/
def bottles_per_day : ℕ := 6

/-- The number of days the water bottles would last -/
def days_lasting : ℕ := 2

/-- The number of water bottles Debby bought -/
def bottles_bought : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : bottles_bought = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l228_22882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l228_22847

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_k_value 
  (seq : ArithmeticSequence) 
  (k : ℕ) 
  (h1 : seq.S (k - 2) = -4)
  (h2 : seq.S k = 0)
  (h3 : seq.S (k + 2) = 8) :
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l228_22847


namespace NUMINAMATH_CALUDE_student_group_size_l228_22821

theorem student_group_size (n : ℕ) (h : n > 1) :
  (2 : ℚ) / n = (1 : ℚ) / 5 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_group_size_l228_22821


namespace NUMINAMATH_CALUDE_expression_simplification_l228_22820

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(x + 2) - 5*(3 - 2*x) = 19*x - 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l228_22820


namespace NUMINAMATH_CALUDE_max_garden_area_l228_22887

def garden_area (width : ℝ) : ℝ := 2 * width * width

def garden_perimeter (width : ℝ) : ℝ := 6 * width

theorem max_garden_area :
  ∃ (w : ℝ), w > 0 ∧ garden_perimeter w = 480 ∧
  ∀ (x : ℝ), x > 0 ∧ garden_perimeter x = 480 → garden_area x ≤ garden_area w ∧
  garden_area w = 12800 :=
sorry

end NUMINAMATH_CALUDE_max_garden_area_l228_22887


namespace NUMINAMATH_CALUDE_triangle_area_l228_22843

theorem triangle_area (A B C : Real) (h1 : A > B) (h2 : B > C) 
  (h3 : 2 * Real.cos (2 * B) - 8 * Real.cos B + 5 = 0)
  (h4 : Real.tan A + Real.tan C = 3 + Real.sqrt 3)
  (h5 : 2 * Real.sqrt 3 = Real.sin C * (A - C)) : 
  (1 / 2) * (A - C) * 2 * Real.sqrt 3 = 12 - 4 * Real.sqrt 3 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l228_22843


namespace NUMINAMATH_CALUDE_sum_of_eleventh_powers_l228_22827

/-- Given two real numbers a and b satisfying certain conditions, prove that a^11 + b^11 = 199 -/
theorem sum_of_eleventh_powers (a b : ℝ) : 
  (a + b = 1) →
  (a^2 + b^2 = 3) →
  (a^3 + b^3 = 4) →
  (a^4 + b^4 = 7) →
  (a^5 + b^5 = 11) →
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^11 + b^11 = 199 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eleventh_powers_l228_22827


namespace NUMINAMATH_CALUDE_cylinder_radius_is_18_over_5_l228_22898

/-- A right circular cone with a right circular cylinder inscribed within it. -/
structure ConeWithCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The conditions for our specific cone and cylinder. -/
def cone_cylinder_conditions (c : ConeWithCylinder) : Prop :=
  c.cone_diameter = 12 ∧
  c.cone_altitude = 18 ∧
  c.cylinder_radius * 2 = c.cylinder_radius * 2

theorem cylinder_radius_is_18_over_5 (c : ConeWithCylinder) 
  (h : cone_cylinder_conditions c) : c.cylinder_radius = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_is_18_over_5_l228_22898


namespace NUMINAMATH_CALUDE_initial_average_calculation_l228_22860

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) : 
  n = 10 → 
  correct_avg = 6 → 
  error = 10 → 
  (n * correct_avg - error) / n = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l228_22860


namespace NUMINAMATH_CALUDE_four_digit_count_is_900_l228_22812

/-- The count of four-digit positive integers with thousands digit 3 and non-zero hundreds digit -/
def four_digit_count : ℕ :=
  let thousands_digit := 3
  let hundreds_choices := 9  -- 1 to 9
  let tens_choices := 10     -- 0 to 9
  let ones_choices := 10     -- 0 to 9
  hundreds_choices * tens_choices * ones_choices

theorem four_digit_count_is_900 : four_digit_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_count_is_900_l228_22812


namespace NUMINAMATH_CALUDE_chocolate_probabilities_l228_22892

theorem chocolate_probabilities (w1 n1 w2 n2 : ℕ) 
  (h1 : w1 ≤ n1) (h2 : w2 ≤ n2) (h3 : n1 > 0) (h4 : n2 > 0) :
  ∃ (w1' n1' w2' n2' : ℕ),
    w1' ≤ n1' ∧ w2' ≤ n2' ∧ n1' > 0 ∧ n2' > 0 ∧
    (w1' : ℚ) / n1' = (w1 + w2 : ℚ) / (n1 + n2) ∧
  ∃ (w1'' n1'' w2'' n2'' : ℕ),
    w1'' ≤ n1'' ∧ w2'' ≤ n2'' ∧ n1'' > 0 ∧ n2'' > 0 ∧
    ¬((w1'' : ℚ) / n1'' < (w1'' + w2'' : ℚ) / (n1'' + n2'') ∧
      (w1'' + w2'' : ℚ) / (n1'' + n2'') < (w2'' : ℚ) / n2'') :=
by sorry

end NUMINAMATH_CALUDE_chocolate_probabilities_l228_22892


namespace NUMINAMATH_CALUDE_fraction_sum_l228_22888

theorem fraction_sum (x : ℝ) (h1 : x ≠ 1) (h2 : 2*x ≠ -3) (h3 : 2*x^2 + 5*x - 3 ≠ 0) : 
  (6*x - 8) / (2*x^2 + 5*x - 3) = (-2/5) / (x - 1) + (34/5) / (2*x + 3) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l228_22888


namespace NUMINAMATH_CALUDE_hiking_problem_solution_l228_22864

/-- Represents the hiking problem with given speeds and distances -/
structure HikingProblem where
  total_time : ℚ  -- in hours
  total_distance : ℚ  -- in km
  uphill_speed : ℚ  -- in km/h
  flat_speed : ℚ  -- in km/h
  downhill_speed : ℚ  -- in km/h

/-- Theorem stating the solution to the hiking problem -/
theorem hiking_problem_solution (p : HikingProblem) 
  (h1 : p.total_time = 221 / 60)  -- 3 hours and 41 minutes in decimal form
  (h2 : p.total_distance = 9)
  (h3 : p.uphill_speed = 4)
  (h4 : p.flat_speed = 5)
  (h5 : p.downhill_speed = 6) :
  ∃ (x : ℚ), x = 4 ∧ 
    (2 * x / p.flat_speed + 
     (5 * (p.total_distance - x)) / (12 : ℚ) = p.total_time) := by
  sorry


end NUMINAMATH_CALUDE_hiking_problem_solution_l228_22864


namespace NUMINAMATH_CALUDE_square_one_implies_plus_minus_one_l228_22858

theorem square_one_implies_plus_minus_one (x : ℝ) : x^2 = 1 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_one_implies_plus_minus_one_l228_22858


namespace NUMINAMATH_CALUDE_shirt_to_pants_ratio_l228_22895

/-- Proves that the ratio of the price of the shirt to the price of the pants is 3:4 given the conditions of the problem. -/
theorem shirt_to_pants_ratio (total_cost pants_price shoes_price shirt_price : ℕ) : 
  total_cost = 340 →
  pants_price = 120 →
  shoes_price = pants_price + 10 →
  shirt_price = total_cost - pants_price - shoes_price →
  (shirt_price : ℚ) / pants_price = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shirt_to_pants_ratio_l228_22895


namespace NUMINAMATH_CALUDE_time_saved_two_pipes_l228_22802

/-- Represents the time saved when using two pipes instead of one to fill a reservoir -/
theorem time_saved_two_pipes (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  let time_saved := p - (a * p) / (a + b)
  time_saved = (b * p) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_time_saved_two_pipes_l228_22802


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l228_22872

theorem angle_sum_theorem (θ φ : Real) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 2/5) (h4 : Real.cos φ = 1/2) :
  2 * θ + φ = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l228_22872


namespace NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l228_22800

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a color (either 0 or 1) -/
inductive Color where
  | zero : Color
  | one : Color

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is on the side of a triangle -/
def isOnSide (p : Point) (t : Triangle) : Prop := sorry

/-- Represents a coloring of points on the sides of a triangle -/
def Coloring (t : Triangle) := Point → Color

/-- The main theorem to be proved -/
theorem monochromatic_right_triangle_exists 
  (t : Triangle) 
  (h_equilateral : isEquilateral t) 
  (coloring : Coloring t) : 
  ∃ (p q r : Point), 
    isOnSide p t ∧ isOnSide q t ∧ isOnSide r t ∧
    isRightAngled ⟨p, q, r⟩ ∧
    coloring p = coloring q ∧ coloring q = coloring r :=
sorry

end NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l228_22800


namespace NUMINAMATH_CALUDE_cost_reduction_proof_l228_22899

theorem cost_reduction_proof (x : ℝ) : 
  (x ≥ 0) →  -- Ensure x is non-negative
  (x ≤ 1) →  -- Ensure x is at most 100%
  ((1 - x)^2 = 1 - 0.36) →
  x = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_cost_reduction_proof_l228_22899


namespace NUMINAMATH_CALUDE_bicycle_wheel_radius_l228_22859

theorem bicycle_wheel_radius (diameter : ℝ) (h : diameter = 26) : 
  diameter / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheel_radius_l228_22859


namespace NUMINAMATH_CALUDE_not_prime_fourth_power_minus_four_l228_22862

theorem not_prime_fourth_power_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬∃ q : ℕ, Nat.Prime q ∧ p - 4 = q^4 := by
  sorry

end NUMINAMATH_CALUDE_not_prime_fourth_power_minus_four_l228_22862


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l228_22807

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 9

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ (x y : ℝ), circle_equation x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
                   c.center = (3, -4) ∧
                   c.radius = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l228_22807


namespace NUMINAMATH_CALUDE_probability_of_s_in_statistics_l228_22837

def word : String := "statistics"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· = c) |>.length

theorem probability_of_s_in_statistics :
  (count_letter word 's' : ℚ) / word.length = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_s_in_statistics_l228_22837


namespace NUMINAMATH_CALUDE_job1_rate_is_correct_l228_22832

/-- Represents the hourly rate of job 1 -/
def job1_rate : ℝ := 7

/-- Represents the hourly rate of job 2 -/
def job2_rate : ℝ := 10

/-- Represents the hourly rate of job 3 -/
def job3_rate : ℝ := 12

/-- Represents the number of hours worked on job 1 per day -/
def job1_hours : ℝ := 3

/-- Represents the number of hours worked on job 2 per day -/
def job2_hours : ℝ := 2

/-- Represents the number of hours worked on job 3 per day -/
def job3_hours : ℝ := 4

/-- Represents the number of days worked -/
def days_worked : ℝ := 5

/-- Represents the total earnings for the period -/
def total_earnings : ℝ := 445

theorem job1_rate_is_correct : 
  days_worked * (job1_hours * job1_rate + job2_hours * job2_rate + job3_hours * job3_rate) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_job1_rate_is_correct_l228_22832


namespace NUMINAMATH_CALUDE_colored_paper_distribution_l228_22863

/-- Proves that each female student receives 6 sheets of colored paper given the problem conditions -/
theorem colored_paper_distribution (total_students : ℕ) (total_paper : ℕ) (leftover : ℕ) :
  total_students = 24 →
  total_paper = 50 →
  leftover = 2 →
  ∃ (female_students : ℕ) (male_students : ℕ),
    female_students + male_students = total_students ∧
    male_students = 2 * female_students ∧
    (total_paper - leftover) % female_students = 0 ∧
    (total_paper - leftover) / female_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_distribution_l228_22863


namespace NUMINAMATH_CALUDE_bike_rental_problem_l228_22866

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rentedHours (totalPayment fixedFee hourlyRate : ℚ) : ℚ :=
  (totalPayment - fixedFee) / hourlyRate

theorem bike_rental_problem :
  let totalPayment : ℚ := 80
  let fixedFee : ℚ := 17
  let hourlyRate : ℚ := 7
  rentedHours totalPayment fixedFee hourlyRate = 9 := by
sorry

#eval rentedHours 80 17 7

end NUMINAMATH_CALUDE_bike_rental_problem_l228_22866


namespace NUMINAMATH_CALUDE_trapezoid_area_in_regular_hexagon_l228_22830

/-- The area of a trapezoid formed by connecting midpoints of non-adjacent sides in a regular hexagon -/
theorem trapezoid_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let height := side_length * Real.sqrt 3 / 2
  let trapezoid_base := side_length / 2
  let trapezoid_area := (trapezoid_base + trapezoid_base) * height / 2
  trapezoid_area = 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_regular_hexagon_l228_22830


namespace NUMINAMATH_CALUDE_chalkboard_area_l228_22840

/-- The area of a rectangular chalkboard with width 3 feet and length 2 times its width is 18 square feet. -/
theorem chalkboard_area (width : ℝ) (length : ℝ) : 
  width = 3 → length = 2 * width → width * length = 18 := by
  sorry

end NUMINAMATH_CALUDE_chalkboard_area_l228_22840


namespace NUMINAMATH_CALUDE_volume_bound_l228_22824

/-- 
Given a body in 3D space, its volume does not exceed the square root of the product 
of the areas of its projections onto the coordinate planes.
-/
theorem volume_bound (S₁ S₂ S₃ V : ℝ) 
  (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : V > 0)
  (h_S₁ : S₁ = area_projection_xy)
  (h_S₂ : S₂ = area_projection_yz)
  (h_S₃ : S₃ = area_projection_zx)
  (h_V : V = volume_of_body) : 
  V ≤ Real.sqrt (S₁ * S₂ * S₃) := by
  sorry

end NUMINAMATH_CALUDE_volume_bound_l228_22824


namespace NUMINAMATH_CALUDE_only_D_is_certain_l228_22826

structure Event where
  description : String
  is_certain : Bool

def event_A : Event := { description := "It will definitely rain on a cloudy day", is_certain := false }
def event_B : Event := { description := "When tossing a fair coin, the head side faces up", is_certain := false }
def event_C : Event := { description := "A boy's height is definitely taller than a girl's", is_certain := false }
def event_D : Event := { description := "When oil is dropped into water, the oil will float on the surface of the water", is_certain := true }

def events : List Event := [event_A, event_B, event_C, event_D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by sorry

end NUMINAMATH_CALUDE_only_D_is_certain_l228_22826


namespace NUMINAMATH_CALUDE_parabolas_sum_l228_22879

/-- Given two parabolas that intersect the coordinate axes at four points forming a rhombus -/
structure Parabolas where
  a : ℝ
  b : ℝ
  parabola1 : ℝ → ℝ
  parabola2 : ℝ → ℝ
  h_parabola1 : ∀ x, parabola1 x = a * x^2 - 2
  h_parabola2 : ∀ x, parabola2 x = 6 - b * x^2
  h_rhombus : ∃ x1 x2 y1 y2, 
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola2 0 = y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2
  h_area : (x2 - x1) * (y2 - y1) = 24
  h_b_eq_2a : b = 2 * a

/-- The sum of a and b is 6 -/
theorem parabolas_sum (p : Parabolas) : p.a + p.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_sum_l228_22879


namespace NUMINAMATH_CALUDE_sqrt_5_irrational_l228_22857

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_5_irrational : IsIrrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_irrational_l228_22857


namespace NUMINAMATH_CALUDE_common_tangents_from_guiding_circles_l228_22839

/-- Represents an ellipse with its foci and semi-major axis -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  semiMajorAxis : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Enum representing the possible number of common tangents -/
inductive NumCommonTangents
  | zero
  | one
  | two

/-- Function to determine the number of intersections between two circles -/
def circleIntersections (c1 c2 : Circle) : NumCommonTangents :=
  sorry

/-- Function to get the guiding circle of an ellipse for a given focus -/
def guidingCircle (e : Ellipse) (f : ℝ × ℝ) : Circle :=
  sorry

/-- Theorem stating that the number of common tangents between two ellipses
    sharing a focus is determined by the intersection of their guiding circles -/
theorem common_tangents_from_guiding_circles 
  (e1 e2 : Ellipse) 
  (h : e1.focus1 = e2.focus1) :
  ∃ (f : ℝ × ℝ), 
    let c1 := guidingCircle e1 f
    let c2 := guidingCircle e2 f
    circleIntersections c1 c2 = NumCommonTangents.zero ∨
    circleIntersections c1 c2 = NumCommonTangents.one ∨
    circleIntersections c1 c2 = NumCommonTangents.two :=
  sorry

end NUMINAMATH_CALUDE_common_tangents_from_guiding_circles_l228_22839


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l228_22842

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := x^2 + 2*x*f'₁ - 6

-- State the theorem
theorem f_derivative_at_one :
  ∃ f'₁ : ℝ, (∀ x, deriv (f · f'₁) x = 2*x + 2*f'₁) ∧ f'₁ = -2 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l228_22842


namespace NUMINAMATH_CALUDE_proportion_equation_l228_22811

theorem proportion_equation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x = 3 * y) :
  x / 3 = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equation_l228_22811


namespace NUMINAMATH_CALUDE_polynomial_roots_l228_22870

theorem polynomial_roots : 
  let f : ℂ → ℂ := λ x => 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3
  let r1 := (-1 + Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r2 := (-1 - Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r3 := (-1 + Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  let r4 := (-1 - Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  (f r1 = 0) ∧ (f r2 = 0) ∧ (f r3 = 0) ∧ (f r4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l228_22870


namespace NUMINAMATH_CALUDE_failing_percentage_possible_l228_22868

theorem failing_percentage_possible (n : ℕ) (d a : ℕ) (f_d f_a : ℕ) :
  n = 25 →
  d + a ≥ n →
  (f_d : ℚ) / (d : ℚ) = 3 / 10 →
  (f_a : ℚ) / (a : ℚ) = 3 / 10 →
  ∃ (f_total : ℕ), (f_total : ℚ) / (n : ℚ) > 7 / 20 ∧ f_total ≤ f_d + f_a :=
by sorry


end NUMINAMATH_CALUDE_failing_percentage_possible_l228_22868


namespace NUMINAMATH_CALUDE_complex_equation_solution_l228_22871

theorem complex_equation_solution (z : ℂ) : (1 - 3*I)*z = 2 + 4*I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l228_22871


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l228_22838

-- Define the equation
def equation (x : ℝ) : Prop := (x - 2) * (x + 5) = 0

-- Define sufficient condition
def sufficient (p q : Prop) : Prop := p → q

-- Define necessary condition
def necessary (p q : Prop) : Prop := q → p

-- Theorem statement
theorem x_eq_2_sufficient_not_necessary :
  (sufficient (x = 2) (equation x)) ∧ ¬(necessary (x = 2) (equation x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l228_22838


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l228_22835

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3 * y - 15 → y ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l228_22835


namespace NUMINAMATH_CALUDE_club_officer_selection_l228_22816

/-- The number of ways to select officers in a club with special conditions -/
def select_officers (total_members : ℕ) (officers_needed : ℕ) (special_members : ℕ) : ℕ :=
  let remaining_members := total_members - special_members
  let case1 := remaining_members * (remaining_members - 1) * (remaining_members - 2) * (remaining_members - 3)
  let case2 := officers_needed * (officers_needed - 1) * (officers_needed - 2) * remaining_members
  case1 + case2

/-- Theorem stating the number of ways to select officers under given conditions -/
theorem club_officer_selection :
  select_officers 25 4 3 = 176088 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l228_22816


namespace NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l228_22852

/-- A regular hendecagon is an 11-sided polygon -/
def RegularHendecagon : Nat := 11

/-- The number of diagonals in a regular hendecagon -/
def NumDiagonals : Nat := (RegularHendecagon.choose 2) - RegularHendecagon

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def WaysToChooseTwoDiagonals : Nat := NumDiagonals.choose 2

/-- The number of sets of 4 vertices that determine intersecting diagonals -/
def IntersectingDiagonalSets : Nat := RegularHendecagon.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def IntersectionProbability : Rat := IntersectingDiagonalSets / WaysToChooseTwoDiagonals

theorem hendecagon_diagonal_intersection_probability :
  IntersectionProbability = 165 / 473 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l228_22852


namespace NUMINAMATH_CALUDE_triangle_area_l228_22861

/-- Given a triangle with perimeter 35 cm and inradius 4.5 cm, its area is 78.75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 35 → inradius = 4.5 → area = perimeter / 2 * inradius → area = 78.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l228_22861


namespace NUMINAMATH_CALUDE_cubic_room_floor_perimeter_l228_22876

/-- The perimeter of the floor of a cubic room -/
def floor_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of the floor of a cubic room with side length 5 meters is 20 meters -/
theorem cubic_room_floor_perimeter :
  floor_perimeter 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_room_floor_perimeter_l228_22876


namespace NUMINAMATH_CALUDE_fish_population_estimate_l228_22828

theorem fish_population_estimate (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (initial_tagged : ℚ) →
  (initial_tagged * second_catch : ℚ) / tagged_in_second = 1800 :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l228_22828


namespace NUMINAMATH_CALUDE_correct_order_of_operations_l228_22897

-- Define the expression
def expression : List ℤ := [150, -50, 25, 5]

-- Define the operations
inductive Operation
| Addition
| Subtraction
| Multiplication

-- Define the order of operations
def orderOfOperations : List Operation := [Operation.Multiplication, Operation.Subtraction, Operation.Addition]

-- Function to evaluate the expression
def evaluate (expr : List ℤ) (ops : List Operation) : ℤ :=
  sorry

-- Theorem statement
theorem correct_order_of_operations :
  evaluate expression orderOfOperations = 225 :=
sorry

end NUMINAMATH_CALUDE_correct_order_of_operations_l228_22897


namespace NUMINAMATH_CALUDE_line_slope_l228_22825

theorem line_slope (x y : ℝ) : 
  x - Real.sqrt 3 * y + 3 = 0 → 
  (y - Real.sqrt 3) / (x - (-3)) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l228_22825


namespace NUMINAMATH_CALUDE_propositions_truth_l228_22869

-- Define the necessary geometric objects
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relations
def subset (a : Line) (α : Plane) : Prop := sorry
def perpendicular_line_plane (a : Line) (β : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- Define the propositions
def proposition_p (a : Line) (α β : Plane) : Prop :=
  subset a α → (perpendicular_line_plane a β → perpendicular_planes α β)

-- Define a polyhedron type
def Polyhedron : Type := sorry

-- Define the properties of a polyhedron
def has_two_parallel_faces (p : Polyhedron) : Prop := sorry
def other_faces_are_trapezoids (p : Polyhedron) : Prop := sorry
def is_prism (p : Polyhedron) : Prop := sorry

-- Define proposition q
def proposition_q (p : Polyhedron) : Prop :=
  has_two_parallel_faces p ∧ other_faces_are_trapezoids p → is_prism p

theorem propositions_truth : ∃ (a : Line) (α β : Plane) (p : Polyhedron),
  proposition_p a α β ∧ ¬proposition_q p := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l228_22869


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l228_22851

theorem concentric_circles_radii_difference (r : ℝ) (h : r > 0) :
  let R := (4 * r ^ 2) ^ (1 / 2 : ℝ)
  R - r = r :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l228_22851


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l228_22836

/-- A line passing through two points is parallel to another line -/
def is_parallel_line (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) = -a / b

/-- The value of m for which the line through (-2, m) and (m, 4) is parallel to 2x + y - 1 = 0 -/
theorem parallel_line_m_value :
  ∀ m : ℝ, is_parallel_line (-2) m m 4 2 1 (-1) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l228_22836


namespace NUMINAMATH_CALUDE_min_t_value_l228_22810

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 2]
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement: The minimum value of t that satisfies |f(x₁) - f(x₂)| ≤ t for all x₁, x₂ in the interval is 20
theorem min_t_value : 
  (∃ t : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ t : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) → t ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l228_22810


namespace NUMINAMATH_CALUDE_maggie_income_l228_22867

def office_rate : ℝ := 10
def tractor_rate : ℝ := 12
def tractor_hours : ℝ := 13
def office_hours : ℝ := 2 * tractor_hours

def total_income : ℝ := office_rate * office_hours + tractor_rate * tractor_hours

theorem maggie_income : total_income = 416 := by
  sorry

end NUMINAMATH_CALUDE_maggie_income_l228_22867


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l228_22806

def is_on_circle (x y : ℤ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 16

theorem max_sum_on_circle :
  ∃ (a b : ℤ), is_on_circle a b ∧
  ∀ (x y : ℤ), is_on_circle x y → x + y ≤ a + b ∧
  a + b = 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l228_22806


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l228_22849

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l228_22849


namespace NUMINAMATH_CALUDE_segments_form_triangle_l228_22890

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The line segments 4cm, 5cm, and 6cm can form a triangle. -/
theorem segments_form_triangle : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l228_22890


namespace NUMINAMATH_CALUDE_not_strictly_decreasing_cubic_function_l228_22878

theorem not_strictly_decreasing_cubic_function (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (-x₁^3 + b*x₁^2 - (2*b + 3)*x₁ + 2 - b) ≤ (-x₂^3 + b*x₂^2 - (2*b + 3)*x₂ + 2 - b)) ↔ 
  (b < -1 ∨ b > 3) :=
by sorry

end NUMINAMATH_CALUDE_not_strictly_decreasing_cubic_function_l228_22878


namespace NUMINAMATH_CALUDE_no_real_roots_l228_22817

theorem no_real_roots (a b : ℝ) : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l228_22817


namespace NUMINAMATH_CALUDE_a_range_l228_22814

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3*a else a^x - 2

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (a > 0 ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_a_range_l228_22814


namespace NUMINAMATH_CALUDE_exists_same_color_configuration_l228_22845

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A grid of cells with colors -/
def Grid := Fin 5 → Fin 41 → Color

/-- A configuration of three rows and three columns -/
structure Configuration where
  rows : Fin 3 → Fin 5
  cols : Fin 3 → Fin 41

/-- Check if a configuration has all intersections of the same color -/
def Configuration.allSameColor (grid : Grid) (config : Configuration) : Prop :=
  ∃ c : Color, ∀ i j : Fin 3, grid (config.rows i) (config.cols j) = c

/-- Main theorem: There exists a configuration with all intersections of the same color -/
theorem exists_same_color_configuration (grid : Grid) :
  ∃ config : Configuration, config.allSameColor grid := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_configuration_l228_22845


namespace NUMINAMATH_CALUDE_solve_bowtie_equation_l228_22819

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (g : ℝ) : bowtie 5 g = 14 → g = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_bowtie_equation_l228_22819


namespace NUMINAMATH_CALUDE_colored_rectangle_iff_same_parity_l228_22801

/-- Represents the four colors used to color the squares -/
inductive Color
  | Red
  | Yellow
  | Blue
  | Green

/-- Represents a unit square with colored sides -/
structure ColoredSquare where
  top : Color
  right : Color
  bottom : Color
  left : Color
  different_colors : top ≠ right ∧ top ≠ bottom ∧ top ≠ left ∧ 
                     right ≠ bottom ∧ right ≠ left ∧ 
                     bottom ≠ left

/-- Represents a rectangle formed by colored squares -/
structure ColoredRectangle where
  width : ℕ
  height : ℕ
  top_color : Color
  right_color : Color
  bottom_color : Color
  left_color : Color
  different_colors : top_color ≠ right_color ∧ top_color ≠ bottom_color ∧ top_color ≠ left_color ∧ 
                     right_color ≠ bottom_color ∧ right_color ≠ left_color ∧ 
                     bottom_color ≠ left_color

/-- Theorem stating that a colored rectangle can be formed if and only if its side lengths have the same parity -/
theorem colored_rectangle_iff_same_parity (r : ColoredRectangle) :
  (∃ (squares : List (List ColoredSquare)), 
    squares.length = r.height ∧ 
    (∀ row ∈ squares, row.length = r.width) ∧ 
    -- Additional conditions for correct arrangement of squares
    sorry
  ) ↔ 
  (r.width % 2 = r.height % 2) :=
sorry

end NUMINAMATH_CALUDE_colored_rectangle_iff_same_parity_l228_22801


namespace NUMINAMATH_CALUDE_gcf_36_60_90_l228_22891

theorem gcf_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_60_90_l228_22891


namespace NUMINAMATH_CALUDE_girls_in_class_l228_22809

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) : 
  total_students = 36 → 
  girl_ratio = 4 → 
  boy_ratio = 5 → 
  (girl_ratio + boy_ratio : ℚ) * (total_students / (girl_ratio + boy_ratio : ℕ)) = girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) →
  girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) = 16 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l228_22809


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l228_22884

/-- The symmetric point of P(-2, 1) with respect to the line y = x + 1 is (0, -1) -/
theorem symmetric_point_theorem : 
  let P : ℝ × ℝ := (-2, 1)
  let line (x y : ℝ) : Prop := y = x + 1
  let is_symmetric (P Q : ℝ × ℝ) : Prop := 
    let midpoint := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
    line midpoint.1 midpoint.2 ∧ 
    (Q.2 - P.2) * (Q.1 - P.1) = -1  -- Perpendicular condition
  ∃ Q : ℝ × ℝ, is_symmetric P Q ∧ Q = (0, -1) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l228_22884


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_positive_l228_22823

def S : Set Int := {2, 5, -7, 8, -10}

theorem smallest_sum_of_three_positive : 
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b + c = 15 ∧
   (∀ (x y z : Int), x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y → y ≠ z → x ≠ z → 
    x > 0 → y > 0 → z > 0 → 
    x + y + z ≥ 15)) := by
  sorry

#check smallest_sum_of_three_positive

end NUMINAMATH_CALUDE_smallest_sum_of_three_positive_l228_22823


namespace NUMINAMATH_CALUDE_D_sqrt_sometimes_rational_sometimes_not_l228_22853

def D (x : ℝ) : ℝ := 
  let a := 2*x + 1
  let b := 2*x + 3
  let c := a*b + 5
  a^2 + b^2 + c^2

theorem D_sqrt_sometimes_rational_sometimes_not :
  ∃ x y : ℝ, (∃ q : ℚ, Real.sqrt (D x) = q) ∧ 
             (∀ q : ℚ, Real.sqrt (D y) ≠ q) :=
sorry

end NUMINAMATH_CALUDE_D_sqrt_sometimes_rational_sometimes_not_l228_22853


namespace NUMINAMATH_CALUDE_puppy_food_bags_l228_22855

/-- Calculates the number of bags of special dog food needed for a puppy's first year -/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_weight : ℕ := 5
  let first_period : ℕ := 60
  let first_period_daily_food : ℕ := 2
  let second_period_daily_food : ℕ := 4
  let first_period_total : ℕ := first_period * first_period_daily_food
  let second_period : ℕ := days_in_year - first_period
  let second_period_total : ℕ := second_period * second_period_daily_food
  let total_ounces : ℕ := first_period_total + second_period_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_weight - 1) / bag_weight

theorem puppy_food_bags : bags_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_food_bags_l228_22855


namespace NUMINAMATH_CALUDE_probability_of_three_in_eight_elevenths_l228_22815

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry -- Implementation of decimal representation calculation

theorem probability_of_three_in_eight_elevenths (n d : ℕ) (h : n = 8 ∧ d = 11) :
  let rep := decimal_representation n d
  (rep.count 3) / rep.length = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_of_three_in_eight_elevenths_l228_22815


namespace NUMINAMATH_CALUDE_valid_combinations_count_l228_22850

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of valid four-letter word combinations where the word begins and ends with the same letter, and the second letter is a vowel -/
def valid_combinations : ℕ := alphabet_size * vowel_count * alphabet_size

theorem valid_combinations_count : valid_combinations = 3380 := by
  sorry

end NUMINAMATH_CALUDE_valid_combinations_count_l228_22850


namespace NUMINAMATH_CALUDE_cake_recipe_flour_calculation_l228_22841

/-- Given a ratio of milk to flour and an amount of milk used, calculate the amount of flour needed. -/
def flour_needed (milk_ratio : ℚ) (flour_ratio : ℚ) (milk_used : ℚ) : ℚ :=
  (flour_ratio / milk_ratio) * milk_used

/-- The theorem states that given the specified ratio and milk amount, the flour needed is 1200 mL. -/
theorem cake_recipe_flour_calculation :
  let milk_ratio : ℚ := 60
  let flour_ratio : ℚ := 300
  let milk_used : ℚ := 240
  flour_needed milk_ratio flour_ratio milk_used = 1200 := by
sorry

#eval flour_needed 60 300 240

end NUMINAMATH_CALUDE_cake_recipe_flour_calculation_l228_22841


namespace NUMINAMATH_CALUDE_total_pies_sold_is_29_l228_22873

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_orders : ℕ := 52

/-- Represents the number of customers who ordered slices of chicken pot pie -/
def chicken_pot_pie_orders : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ :=
  shepherds_pie_orders / shepherds_pie_pieces +
  chicken_pot_pie_orders / chicken_pot_pie_pieces

/-- Proves that the total number of pies sold is 29 -/
theorem total_pies_sold_is_29 : total_pies_sold = 29 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_sold_is_29_l228_22873


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l228_22803

-- Define the quadratic function
def quadratic_function (m x : ℝ) : ℝ := m * x^2 - 4 * x + 1

-- State the theorem
theorem quadratic_minimum_value (m : ℝ) :
  (∃ x_min : ℝ, ∀ x : ℝ, quadratic_function m x ≥ quadratic_function m x_min) ∧
  (∃ x_min : ℝ, quadratic_function m x_min = -3) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l228_22803


namespace NUMINAMATH_CALUDE_combination_arrangement_equality_l228_22874

theorem combination_arrangement_equality (m : ℕ) : (Nat.choose m 3) = (m * (m - 1)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_arrangement_equality_l228_22874


namespace NUMINAMATH_CALUDE_cube_edge_increase_l228_22880

theorem cube_edge_increase (e : ℝ) (f : ℝ) (h : e > 0) : (f * e)^3 = 8 * e^3 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l228_22880


namespace NUMINAMATH_CALUDE_sum_of_abc_l228_22883

theorem sum_of_abc (a b c : ℝ) : 
  a * (a - 4) = 5 →
  b * (b - 4) = 5 →
  c * (c - 4) = 5 →
  a^2 + b^2 = c^2 →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  a + b + c = 4 + Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l228_22883


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l228_22833

/-- Calculate the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is approximately 9.23% -/
theorem simple_interest_rate_example :
  let principal := 650
  let amount := 950
  let time := 5
  let rate := simple_interest_rate principal amount time
  (rate ≥ 9.23) ∧ (rate < 9.24) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l228_22833


namespace NUMINAMATH_CALUDE_hash_2_3_4_l228_22893

/-- The # operation defined on real numbers -/
def hash (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem stating that hash(2, 3, 4) = -15 -/
theorem hash_2_3_4 : hash 2 3 4 = -15 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l228_22893


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l228_22829

theorem smallest_integer_satisfying_inequality : 
  ∀ y : ℤ, y < 3 * y - 14 → y ≥ 8 ∧ 8 < 3 * 8 - 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l228_22829


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l228_22865

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l228_22865


namespace NUMINAMATH_CALUDE_boat_journey_time_l228_22881

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time (river_speed : ℝ) (boat_speed : ℝ) (distance : ℝ) : 
  river_speed = 2 →
  boat_speed = 6 →
  distance = 56 →
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 21 := by
  sorry

#check boat_journey_time

end NUMINAMATH_CALUDE_boat_journey_time_l228_22881


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l228_22834

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l228_22834


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l228_22805

-- Define the propositions p and q
def p (x : ℝ) : Prop := x / (x - 2) < 0
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Define the set of x that satisfy p
def set_p : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def set_q (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m > 2 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l228_22805


namespace NUMINAMATH_CALUDE_no_divisible_by_six_l228_22813

theorem no_divisible_by_six : ∀ z : ℕ, z < 10 → ¬(35000 + z * 100 + 45) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_six_l228_22813


namespace NUMINAMATH_CALUDE_solution_range_l228_22877

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + k = 2 * x - 1 ∧ x < 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l228_22877


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l228_22894

theorem quadratic_roots_problem (x y : ℝ) : 
  x + y = 6 → 
  |x - y| = 8 → 
  x^2 - 6*x - 7 = 0 ∧ y^2 - 6*y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l228_22894


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l228_22889

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ γ = α) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The base angle is either 50° or 80°
  (α = 50 ∨ α = 80 ∨ β = 50 ∨ β = 80 ∨ γ = 50 ∨ γ = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l228_22889


namespace NUMINAMATH_CALUDE_max_vector_sum_diff_l228_22808

/-- Given plane vectors a, b, and c satisfying the specified conditions,
    the maximum value of |a + b - c| is 3√2. -/
theorem max_vector_sum_diff (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = ‖b‖ ∧ ‖a‖ ≠ 0)
  (h2 : a.1 * b.1 + a.2 * b.2 = 0)  -- dot product = 0 means perpendicular
  (h3 : ‖c‖ = 2 * Real.sqrt 2)
  (h4 : ‖c - a‖ = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ × ℝ), x = a + b - c → ‖x‖ ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_vector_sum_diff_l228_22808


namespace NUMINAMATH_CALUDE_coefficient_a4b3c2_in_expansion_l228_22896

theorem coefficient_a4b3c2_in_expansion (a b c : ℕ) : 
  (Nat.choose 9 5) * (Nat.choose 5 2) = 1260 := by sorry

end NUMINAMATH_CALUDE_coefficient_a4b3c2_in_expansion_l228_22896
