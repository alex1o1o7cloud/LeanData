import Mathlib

namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l3901_390191

theorem max_students_equal_distribution (pens pencils erasers notebooks : ℕ) 
  (h1 : pens = 1802)
  (h2 : pencils = 1203)
  (h3 : erasers = 1508)
  (h4 : notebooks = 2400) :
  Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l3901_390191


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3901_390152

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let selected_republicans : ℕ := 4
  let selected_democrats : ℕ := 3
  (Nat.choose total_republicans selected_republicans) *
  (Nat.choose total_democrats selected_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3901_390152


namespace NUMINAMATH_CALUDE_line_segment_ratio_l3901_390130

theorem line_segment_ratio (x y z s : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : x / y = y / z)
  (h3 : x + y + z = s)
  (h4 : x + y = z) :
  x / y = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l3901_390130


namespace NUMINAMATH_CALUDE_inequality_proof_l3901_390159

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3901_390159


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3901_390169

theorem museum_ticket_cost : 
  ∀ (num_students num_teachers : ℕ) 
    (student_ticket_cost teacher_ticket_cost : ℕ),
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3901_390169


namespace NUMINAMATH_CALUDE_article_cost_l3901_390170

theorem article_cost (selling_price1 selling_price2 : ℝ) (percentage_diff : ℝ) :
  selling_price1 = 350 →
  selling_price2 = 340 →
  percentage_diff = 0.05 →
  (selling_price1 - selling_price2) / (selling_price2 - (selling_price1 - selling_price2) / percentage_diff) = percentage_diff →
  selling_price1 - (selling_price1 - selling_price2) / percentage_diff = 140 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l3901_390170


namespace NUMINAMATH_CALUDE_find_number_l3901_390174

theorem find_number (G N : ℕ) (h1 : G = 4) (h2 : N % G = 6) (h3 : 1856 % G = 4) : N = 1862 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3901_390174


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3901_390116

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  -- Conditions
  (a > 0) → (b > 0) → (c > 0) →  -- Positive sides
  (a^2 + b^2 = c^2) →            -- Right triangle (Pythagorean theorem)
  (a + b = 7) →                  -- Sum of legs
  (a * b / 2 = 6) →              -- Area
  (c = 5) :=                     -- Conclusion: hypotenuse length
by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3901_390116


namespace NUMINAMATH_CALUDE_not_both_count_l3901_390173

/-- The number of students taking both chemistry and physics -/
def both : ℕ := 15

/-- The total number of students in the chemistry class -/
def chem_total : ℕ := 30

/-- The number of students taking only physics -/
def physics_only : ℕ := 18

/-- The number of students taking chemistry or physics but not both -/
def not_both : ℕ := (chem_total - both) + physics_only

theorem not_both_count : not_both = 33 := by
  sorry

end NUMINAMATH_CALUDE_not_both_count_l3901_390173


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3901_390156

/-- Proves that the average salary of all workers is 750, given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (non_technician_salary : ℕ)
  (h1 : total_workers = 20)
  (h2 : technicians = 5)
  (h3 : technician_salary = 900)
  (h4 : non_technician_salary = 700) :
  (technicians * technician_salary + (total_workers - technicians) * non_technician_salary) / total_workers = 750 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3901_390156


namespace NUMINAMATH_CALUDE_largest_digit_change_l3901_390121

/-- The original incorrect sum -/
def incorrect_sum : ℕ := 1742

/-- The first addend in the original problem -/
def addend1 : ℕ := 789

/-- The second addend in the original problem -/
def addend2 : ℕ := 436

/-- The third addend in the original problem -/
def addend3 : ℕ := 527

/-- The corrected first addend after changing a digit -/
def corrected_addend1 : ℕ := 779

theorem largest_digit_change :
  (∃ (d : ℕ), d ≤ 9 ∧
    corrected_addend1 + addend2 + addend3 = incorrect_sum ∧
    d = (addend1 / 10) % 10 - (corrected_addend1 / 10) % 10 ∧
    ∀ (x y z : ℕ), x ≤ addend1 ∧ y ≤ addend2 ∧ z ≤ addend3 →
      x + y + z = incorrect_sum →
      (∃ (d' : ℕ), d' ≤ 9 ∧ d' = (addend1 / 10) % 10 - (x / 10) % 10) →
      d' ≤ d) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_l3901_390121


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3901_390144

-- Problem 1
theorem problem_1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-2*a^2)^3 * a^2 + a^8 = -7*a^8 := by sorry

-- Problem 3
theorem problem_3 : 2023^2 - 2024 * 2022 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3901_390144


namespace NUMINAMATH_CALUDE_parallel_line_plane_intersection_not_always_parallel_l3901_390129

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between lines and planes
variable (parallelLP : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLL : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersectionPP : Plane → Plane → Line)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane_intersection_not_always_parallel 
  (α β : Plane) (m n : Line) : 
  ∃ (α β : Plane) (m n : Line), 
    α ≠ β ∧ m ≠ n ∧ 
    parallelLP m α ∧ 
    intersectionPP α β = n ∧ 
    ¬(parallelLL m n) := by sorry

end NUMINAMATH_CALUDE_parallel_line_plane_intersection_not_always_parallel_l3901_390129


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l3901_390194

def hamburger_price : ℝ := 5.00
def crackers_price : ℝ := 3.50
def vegetable_price : ℝ := 2.00
def vegetable_bags : ℕ := 4
def cheese_price : ℝ := 3.50
def discount_rate : ℝ := 0.10

def total_before_discount : ℝ :=
  hamburger_price + crackers_price + (vegetable_price * vegetable_bags) + cheese_price

def discount_amount : ℝ := total_before_discount * discount_rate

def total_after_discount : ℝ := total_before_discount - discount_amount

theorem rays_grocery_bill :
  total_after_discount = 18.00 := by
  sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l3901_390194


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l3901_390164

/-- The number of ways to put distinguishable balls into distinguishable boxes -/
def ways_to_distribute (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 81 ways to put 4 distinguishable balls into 3 distinguishable boxes -/
theorem four_balls_three_boxes : ways_to_distribute 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l3901_390164


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3901_390167

theorem complex_magnitude_problem (z : ℂ) (i : ℂ) (h : z = (1 + i) / i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3901_390167


namespace NUMINAMATH_CALUDE_tax_discount_order_invariance_l3901_390126

theorem tax_discount_order_invariance 
  (price : ℝ) 
  (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = 
  price * (1 - discount_rate) * (1 + tax_rate) :=
sorry

end NUMINAMATH_CALUDE_tax_discount_order_invariance_l3901_390126


namespace NUMINAMATH_CALUDE_min_sum_position_max_product_position_l3901_390195

/-- Represents the special number with 1991 nines between two ones -/
def specialNumber : ℕ := 1 * 10^1992 + 1

/-- Calculates the sum when splitting the number at position m -/
def sumAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) + (10^(1992 - m) - 9)

/-- Calculates the product when splitting the number at position m -/
def productAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) * (10^(1992 - m) - 9)

theorem min_sum_position : 
  ∀ m : ℕ, m ≠ 996 → m ≠ 997 → sumAtPosition m > sumAtPosition 996 :=
sorry

theorem max_product_position : 
  ∀ m : ℕ, m ≠ 995 → m ≠ 996 → productAtPosition m < productAtPosition 995 :=
sorry

end NUMINAMATH_CALUDE_min_sum_position_max_product_position_l3901_390195


namespace NUMINAMATH_CALUDE_differences_of_geometric_progression_l3901_390185

/-- Given a geometric progression with first term a₁ and common ratio q,
    the sequence of differences between consecutive terms forms a geometric progression
    with first term a₁(q - 1) and common ratio q. -/
theorem differences_of_geometric_progression
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1) :
  let gp : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let diff : ℕ → ℝ := λ n => gp (n + 1) - gp n
  ∀ n : ℕ, diff (n + 1) = q * diff n :=
by sorry

end NUMINAMATH_CALUDE_differences_of_geometric_progression_l3901_390185


namespace NUMINAMATH_CALUDE_fraction_equality_l3901_390190

theorem fraction_equality : (1721^2 - 1714^2) / (1728^2 - 1707^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3901_390190


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3901_390155

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

-- Define the arithmetic sequence condition
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * a 3 = 2 * ((1/2) * a 5)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence_condition a q →
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3901_390155


namespace NUMINAMATH_CALUDE_adjacent_complementary_implies_complementary_l3901_390104

/-- Two angles are adjacent if they share a common vertex and a common side. -/
def adjacent_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their measures add up to 90 degrees. -/
def complementary_angles (α β : Real) : Prop := α + β = 90

theorem adjacent_complementary_implies_complementary 
  (α β : Real) (h1 : adjacent_angles α β) (h2 : complementary_angles α β) : 
  complementary_angles α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_complementary_implies_complementary_l3901_390104


namespace NUMINAMATH_CALUDE_line_intersection_range_l3901_390197

/-- The line y = e^x + b has at most one common point with both f(x) = e^x and g(x) = ln(x) 
    if and only if b is in the closed interval [-2, 0] -/
theorem line_intersection_range (b : ℝ) : 
  (∀ x : ℝ, (∃! y : ℝ, y = Real.exp x + b ∧ (y = Real.exp x ∨ y = Real.log x)) ∨
            (∀ y : ℝ, y ≠ Real.exp x + b ∨ (y ≠ Real.exp x ∧ y ≠ Real.log x))) ↔ 
  b ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l3901_390197


namespace NUMINAMATH_CALUDE_twins_age_problem_l3901_390139

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 5 → age = 2 := by
sorry

end NUMINAMATH_CALUDE_twins_age_problem_l3901_390139


namespace NUMINAMATH_CALUDE_circle_center_and_tangent_line_l3901_390172

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Define the tangent line l
def tangent_line_equation (x y : ℝ) : Prop := 
  y = (Real.sqrt 3 / 3) * (x + 1) ∨ y = -(Real.sqrt 3 / 3) * (x + 1)

-- Define the point that the line passes through
def point_on_line : ℝ × ℝ := (-1, 0)

theorem circle_center_and_tangent_line :
  (∀ x y, circle_equation x y ↔ (x - 1)^2 + y^2 = 1) ∧
  (tangent_line_equation (point_on_line.1) (point_on_line.2)) ∧
  (∀ x y, tangent_line_equation x y → 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 1 → 
     (x, y) = (x, y) ∨ (x, y) = (x, y))) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_tangent_line_l3901_390172


namespace NUMINAMATH_CALUDE_post_office_distance_l3901_390181

theorem post_office_distance (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) (distance : ℝ) : 
  outbound_speed = 12.5 →
  inbound_speed = 2 →
  total_time = 5.8 →
  distance / outbound_speed + distance / inbound_speed = total_time →
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_post_office_distance_l3901_390181


namespace NUMINAMATH_CALUDE_inequality_solution_f_less_than_one_l3901_390109

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem 1
theorem inequality_solution (x : ℝ) : f x > x + 5 ↔ x > 4 ∨ x < -2 := by sorry

-- Theorem 2
theorem f_less_than_one (x y : ℝ) (h1 : |x - 3*y - 1| < 1/4) (h2 : |2*y + 1| < 1/6) : f x < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_f_less_than_one_l3901_390109


namespace NUMINAMATH_CALUDE_mark_work_hours_l3901_390192

/-- Calculates the number of hours Mark needs to work per week to earn a target amount --/
def hours_per_week (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) : ℚ :=
  let hourly_wage := spring_earnings / (spring_hours_per_week * spring_weeks)
  let total_hours_needed := target_earnings / hourly_wage
  total_hours_needed / target_weeks

theorem mark_work_hours 
  (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) :
  spring_hours_per_week = 35 ∧ 
  spring_weeks = 15 ∧ 
  spring_earnings = 4200 ∧ 
  target_weeks = 50 ∧ 
  target_earnings = 21000 →
  hours_per_week spring_hours_per_week spring_weeks spring_earnings target_weeks target_earnings = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_mark_work_hours_l3901_390192


namespace NUMINAMATH_CALUDE_arun_speed_doubling_l3901_390147

/-- Proves that Arun takes 1 hour less than Anil when he doubles his speed -/
theorem arun_speed_doubling (distance : ℝ) (arun_speed : ℝ) (anil_time : ℝ) :
  distance = 30 →
  arun_speed = 5 →
  distance / arun_speed = anil_time + 2 →
  distance / (2 * arun_speed) = anil_time - 1 := by
  sorry

#check arun_speed_doubling

end NUMINAMATH_CALUDE_arun_speed_doubling_l3901_390147


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3901_390142

theorem simplify_and_rationalize (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  (x / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (z / Real.sqrt 13) =
  15 * Real.sqrt 1001 / 1001 →
  x = Real.sqrt 5 ∧ y = Real.sqrt 9 ∧ z = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3901_390142


namespace NUMINAMATH_CALUDE_prime_ratio_natural_numbers_l3901_390122

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_ratio_natural_numbers :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (is_prime ((x * y^3) / (x + y)) ↔ x = 14 ∧ y = 2) :=
by sorry


end NUMINAMATH_CALUDE_prime_ratio_natural_numbers_l3901_390122


namespace NUMINAMATH_CALUDE_stating_course_selection_schemes_l3901_390117

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of courses with no students -/
def courses_with_no_students : ℕ := 2

/-- Represents the number of courses with students -/
def courses_with_students : ℕ := num_courses - courses_with_no_students

/-- 
  Theorem stating that the number of ways to distribute students among courses
  under the given conditions is 18
-/
theorem course_selection_schemes : 
  (num_courses.choose courses_with_students) * 
  ((num_students.choose courses_with_students) / 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stating_course_selection_schemes_l3901_390117


namespace NUMINAMATH_CALUDE_number_problem_l3901_390100

theorem number_problem : 
  ∃ (number : ℝ), number * 11 = 165 ∧ number = 15 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3901_390100


namespace NUMINAMATH_CALUDE_people_in_line_l3901_390154

theorem people_in_line (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 11 → left = 5 → right = total - left - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l3901_390154


namespace NUMINAMATH_CALUDE_sally_savings_l3901_390198

/-- Represents the trip expenses and savings for Sally's Sea World trip --/
structure SeaWorldTrip where
  parking_cost : ℕ
  entrance_cost : ℕ
  meal_pass_cost : ℕ
  distance_to_sea_world : ℕ
  car_efficiency : ℕ
  gas_cost_per_gallon : ℕ
  additional_savings_needed : ℕ

/-- Calculates the total cost of the trip --/
def total_cost (trip : SeaWorldTrip) : ℕ :=
  trip.parking_cost + trip.entrance_cost + trip.meal_pass_cost +
  (2 * trip.distance_to_sea_world * trip.gas_cost_per_gallon + trip.car_efficiency - 1) / trip.car_efficiency

/-- Theorem stating that Sally has already saved $28 --/
theorem sally_savings (trip : SeaWorldTrip)
  (h1 : trip.parking_cost = 10)
  (h2 : trip.entrance_cost = 55)
  (h3 : trip.meal_pass_cost = 25)
  (h4 : trip.distance_to_sea_world = 165)
  (h5 : trip.car_efficiency = 30)
  (h6 : trip.gas_cost_per_gallon = 3)
  (h7 : trip.additional_savings_needed = 95) :
  total_cost trip - trip.additional_savings_needed = 28 := by
  sorry


end NUMINAMATH_CALUDE_sally_savings_l3901_390198


namespace NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l3901_390199

/-- Given that Joe has 50 toy cars initially and wants to have 62 cars in total,
    prove that the number of additional cars he needs is 12. -/
theorem joe_needs_twelve_more_cars (initial_cars : ℕ) (target_cars : ℕ) 
    (h1 : initial_cars = 50) (h2 : target_cars = 62) : 
    target_cars - initial_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l3901_390199


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3901_390114

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3901_390114


namespace NUMINAMATH_CALUDE_house_painting_time_l3901_390106

/-- Given that 12 women can paint a house in 6 days, prove that 18 women 
    working at the same rate can paint the same house in 4 days. -/
theorem house_painting_time 
  (women_rate : ℝ → ℝ → ℝ) -- Function that takes number of women and days, returns houses painted
  (h1 : women_rate 12 6 = 1) -- 12 women paint 1 house in 6 days
  (h2 : ∀ w d, women_rate w d = w * d * (women_rate 1 1)) -- Linear relationship
  : women_rate 18 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_time_l3901_390106


namespace NUMINAMATH_CALUDE_function_determination_l3901_390149

/-- Given a function f: ℝ → ℝ satisfying f(1/x) = 1/(x+1) for x ≠ 0 and x ≠ -1,
    prove that f(x) = x/(x+1) for x ≠ 0 and x ≠ -1 -/
theorem function_determination (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f (1/x) = 1/(x+1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x/(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l3901_390149


namespace NUMINAMATH_CALUDE_functions_continuous_and_equal_l3901_390115

/-- Darboux property (intermediate value property) -/
def has_darboux_property (f : ℝ → ℝ) : Prop :=
  ∀ a b y, a < b → f a < y → y < f b → ∃ c, a < c ∧ c < b ∧ f c = y

/-- The problem statement -/
theorem functions_continuous_and_equal
  (f g : ℝ → ℝ)
  (h1 : ∀ a, ⨅ (x > a), f x = g a)
  (h2 : ∀ a, ⨆ (x < a), g x = f a)
  (h3 : has_darboux_property f) :
  Continuous f ∧ Continuous g ∧ f = g := by
  sorry

end NUMINAMATH_CALUDE_functions_continuous_and_equal_l3901_390115


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l3901_390111

/-- Represents a batsman's performance --/
structure BatsmanPerformance where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ
  boundaries : ℕ
  strikeRate : ℚ

/-- Calculates the average after the last inning --/
def averageAfterLastInning (b : BatsmanPerformance) : ℚ :=
  (b.innings * b.averageIncrease + b.runsInLastInning) / b.innings

/-- Theorem stating the batsman's average after the 12th inning --/
theorem batsman_average_after_12th_inning (b : BatsmanPerformance)
  (h1 : b.innings = 12)
  (h2 : b.runsInLastInning = 60)
  (h3 : b.averageIncrease = 4)
  (h4 : b.boundaries ≥ 8)
  (h5 : b.strikeRate ≥ 130) :
  averageAfterLastInning b = 16 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l3901_390111


namespace NUMINAMATH_CALUDE_equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l3901_390179

/-- A triangle is cute if the sum of the squares of two sides is equal to twice the square of the third side -/
def IsCuteTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

theorem equilateral_is_cute (a : ℝ) (ha : a > 0) : IsCuteTriangle a a a :=
  sorry

theorem specific_triangle_is_cute : IsCuteTriangle 4 (2 * Real.sqrt 6) (2 * Real.sqrt 5) :=
  sorry

theorem right_angled_cute_triangle_side_length 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_cute : IsCuteTriangle a b c) 
  (h_ac : b = 2 * Real.sqrt 2) : 
  c = 2 * Real.sqrt 6 ∨ c = 2 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l3901_390179


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3901_390124

def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3901_390124


namespace NUMINAMATH_CALUDE_green_hat_cost_l3901_390131

/-- Proves the cost of green hats given the total number of hats, cost of blue hats, 
    total price, and number of green hats. -/
theorem green_hat_cost 
  (total_hats : ℕ) 
  (blue_hat_cost : ℕ) 
  (total_price : ℕ) 
  (green_hats : ℕ) 
  (h1 : total_hats = 85) 
  (h2 : blue_hat_cost = 6) 
  (h3 : total_price = 540) 
  (h4 : green_hats = 30) : 
  (total_price - blue_hat_cost * (total_hats - green_hats)) / green_hats = 7 := by
  sorry

end NUMINAMATH_CALUDE_green_hat_cost_l3901_390131


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l3901_390146

def consecutive_digits_set : List Nat := [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

def digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 10) ((m % 10) :: acc)
    go n []

theorem arithmetic_mean_property :
  let M : Rat := arithmetic_mean consecutive_digits_set
  (M = 137174210) ∧
  (∀ d : Nat, d < 10 → (d ≠ 5 ↔ d ∈ digits M.num.toNat)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l3901_390146


namespace NUMINAMATH_CALUDE_smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l3901_390175

theorem smallest_hypotenuse_right_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 8 →
  c ≥ 4 * Real.sqrt 2 :=
by sorry

theorem isosceles_right_triangle_minimizes_hypotenuse :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    a + b + c = 8 ∧
    c = 4 * Real.sqrt 2 ∧
    a = b :=
by sorry

end NUMINAMATH_CALUDE_smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l3901_390175


namespace NUMINAMATH_CALUDE_exists_distinct_singleton_solutions_l3901_390168

/-- Solution set of x^2 + 4x - 2a ≤ 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 4*x - 2*a ≤ 0}

/-- Solution set of x^2 - ax + a + 3 ≤ 0 -/
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a + 3 ≤ 0}

/-- Theorem stating that there exists an 'a' for which A and B are singleton sets with different elements -/
theorem exists_distinct_singleton_solutions :
  ∃ (a : ℝ), (∃! x, x ∈ A a) ∧ (∃! y, y ∈ B a) ∧ (∀ x y, x ∈ A a → y ∈ B a → x ≠ y) :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_singleton_solutions_l3901_390168


namespace NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l3901_390177

/-- Given a cylinder and a sphere with equal radii, if the ratio of their surface areas is m:n,
    then the ratio of their volumes is (6m - 3n) : 4n. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (H : ℝ) (m n : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ m > 0 ∧ n > 0) :
  (2 * π * R^2 + 2 * π * R * H) / (4 * π * R^2) = m / n →
  (π * R^2 * H) / ((4/3) * π * R^3) = (6 * m - 3 * n) / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l3901_390177


namespace NUMINAMATH_CALUDE_number_divisibility_l3901_390158

theorem number_divisibility (N : ℕ) : 
  N % 7 = 0 ∧ N % 11 = 2 → N / 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l3901_390158


namespace NUMINAMATH_CALUDE_aquarium_illness_percentage_l3901_390123

theorem aquarium_illness_percentage (total_visitors : ℕ) (healthy_visitors : ℕ) : 
  total_visitors = 500 → 
  healthy_visitors = 300 → 
  (total_visitors - healthy_visitors : ℚ) / total_visitors * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_illness_percentage_l3901_390123


namespace NUMINAMATH_CALUDE_pyramid_circumscribed_sphere_area_l3901_390119

theorem pyramid_circumscribed_sphere_area :
  ∀ (a b c : ℝ),
    a = 1 →
    b = Real.sqrt 6 →
    c = 3 →
    (∃ (r : ℝ), r * r = (a * a + b * b + c * c) / 4 ∧
      4 * Real.pi * r * r = 16 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_circumscribed_sphere_area_l3901_390119


namespace NUMINAMATH_CALUDE_paper_distribution_l3901_390113

theorem paper_distribution (num_students : ℕ) (paper_per_student : ℕ) 
  (h1 : num_students = 230) 
  (h2 : paper_per_student = 15) : 
  num_students * paper_per_student = 3450 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l3901_390113


namespace NUMINAMATH_CALUDE_second_meeting_at_six_minutes_l3901_390112

/-- Represents a swimmer in the race -/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the second meeting occurs at 6 minutes -/
theorem second_meeting_at_six_minutes (race : RaceScenario) 
  (h1 : race.poolLength = 120)
  (h2 : race.swimmer1.startPosition = 0)
  (h3 : race.swimmer2.startPosition = 120)
  (h4 : race.firstMeetingTime = 1)
  (h5 : race.firstMeetingPosition = 40)
  (h6 : race.swimmer1.speed = race.firstMeetingPosition / race.firstMeetingTime)
  (h7 : race.swimmer2.speed = (race.poolLength - race.firstMeetingPosition) / race.firstMeetingTime) :
  secondMeetingTime race = 6 :=
sorry

end NUMINAMATH_CALUDE_second_meeting_at_six_minutes_l3901_390112


namespace NUMINAMATH_CALUDE_teacher_budget_shortfall_l3901_390178

def euro_to_usd_rate : ℝ := 1.2
def last_year_budget : ℝ := 6
def this_year_allocation : ℝ := 50
def charity_grant : ℝ := 20
def gift_card : ℝ := 10

def textbooks_price : ℝ := 45
def textbooks_discount : ℝ := 0.15
def textbooks_tax : ℝ := 0.08

def notebooks_price : ℝ := 18
def notebooks_discount : ℝ := 0.10
def notebooks_tax : ℝ := 0.05

def pens_price : ℝ := 27
def pens_discount : ℝ := 0.05
def pens_tax : ℝ := 0.06

def art_supplies_price : ℝ := 35
def art_supplies_tax : ℝ := 0.07

def folders_price : ℝ := 15
def folders_voucher : ℝ := 5
def folders_tax : ℝ := 0.04

theorem teacher_budget_shortfall :
  let converted_budget := last_year_budget * euro_to_usd_rate
  let total_budget := converted_budget + this_year_allocation + charity_grant + gift_card
  
  let textbooks_cost := textbooks_price * (1 - textbooks_discount) * (1 + textbooks_tax)
  let notebooks_cost := notebooks_price * (1 - notebooks_discount) * (1 + notebooks_tax)
  let pens_cost := pens_price * (1 - pens_discount) * (1 + pens_tax)
  let art_supplies_cost := art_supplies_price * (1 + art_supplies_tax)
  let folders_cost := (folders_price - folders_voucher) * (1 + folders_tax)
  
  let total_cost := textbooks_cost + notebooks_cost + pens_cost + art_supplies_cost + folders_cost - gift_card
  
  total_budget - total_cost = -36.16 :=
by sorry

end NUMINAMATH_CALUDE_teacher_budget_shortfall_l3901_390178


namespace NUMINAMATH_CALUDE_max_value_f_neg_one_range_of_a_l3901_390135

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x

-- Define the function h
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2*a - 1) * x + a - 1

-- Theorem for the maximum value of f when a = -1
theorem max_value_f_neg_one :
  ∃ (max : ℝ), max = -1 ∧ ∀ x > 0, f (-1) x ≤ max :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, g a x ≤ h a x) → a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_neg_one_range_of_a_l3901_390135


namespace NUMINAMATH_CALUDE_petya_prize_probability_at_least_one_prize_probability_l3901_390166

-- Define the number of players
def num_players : ℕ := 10

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Theorem for Petya's probability of winning a prize
theorem petya_prize_probability :
  (5 / 6 : ℚ) ^ (num_players - 1) = (5 / 6 : ℚ) ^ 9 := by sorry

-- Theorem for the probability of at least one player winning a prize
theorem at_least_one_prize_probability :
  1 - (1 / die_sides : ℚ) ^ (num_players - 1) = 1 - (1 / 6 : ℚ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_petya_prize_probability_at_least_one_prize_probability_l3901_390166


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_sum_achieved_l3901_390134

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 2*y ≥ 3 + 8*Real.sqrt 2 := by
sorry

theorem min_value_sum_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 2*y₀ = 3 + 8*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_sum_achieved_l3901_390134


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3901_390189

/-- The polynomial to be divided -/
def P (x : ℝ) : ℝ := 9*x^3 - 5*x^2 + 8*x + 15

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def Q (x : ℝ) : ℝ := 9*x^2 + 22*x + 74

/-- The remainder -/
def R : ℝ := 237

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = D x * Q x + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3901_390189


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l3901_390107

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.50) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l3901_390107


namespace NUMINAMATH_CALUDE_opposite_of_negative_2016_l3901_390151

theorem opposite_of_negative_2016 : Int.neg (-2016) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2016_l3901_390151


namespace NUMINAMATH_CALUDE_saturday_zoo_visitors_l3901_390125

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3 * friday_visitors

/-- Theorem stating the number of visitors on Saturday -/
theorem saturday_zoo_visitors : saturday_visitors = 3750 := by sorry

end NUMINAMATH_CALUDE_saturday_zoo_visitors_l3901_390125


namespace NUMINAMATH_CALUDE_price_after_discounts_l3901_390110

-- Define the discount rates
def discount1 : ℚ := 20 / 100
def discount2 : ℚ := 10 / 100
def discount3 : ℚ := 5 / 100

-- Define the original and final prices
def originalPrice : ℚ := 10000
def finalPrice : ℚ := 6800

-- Theorem statement
theorem price_after_discounts :
  originalPrice * (1 - discount1) * (1 - discount2) * (1 - discount3) = finalPrice := by
  sorry

end NUMINAMATH_CALUDE_price_after_discounts_l3901_390110


namespace NUMINAMATH_CALUDE_final_selling_price_l3901_390186

/-- Given an original price and a first discount, calculate the final selling price after an additional 20% discount -/
theorem final_selling_price (m n : ℝ) : 
  let original_price := m
  let first_discount := n
  let price_after_first_discount := original_price - first_discount
  let second_discount_rate := (20 : ℝ) / 100
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = (4/5) * (m - n) := by
sorry

end NUMINAMATH_CALUDE_final_selling_price_l3901_390186


namespace NUMINAMATH_CALUDE_janous_conjecture_l3901_390127

theorem janous_conjecture (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_janous_conjecture_l3901_390127


namespace NUMINAMATH_CALUDE_identify_genuine_coin_l3901_390187

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a weighing operation -/
def weighing (a b : Coin) : WeighingResult :=
  match a, b with
  | Coin.Genuine, Coin.Genuine => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal
  | _, _ => WeighingResult.Unequal

/-- Theorem stating that at least one genuine coin can be identified in at most 2 weighings -/
theorem identify_genuine_coin
  (coins : Fin 5 → Coin)
  (h_genuine : ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ coins i = Coin.Genuine ∧ coins j = Coin.Genuine ∧ coins k = Coin.Genuine)
  (h_counterfeit : ∃ i j, i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit) :
  ∃ (w₁ w₂ : Fin 5 × Fin 5), ∃ (i : Fin 5), coins i = Coin.Genuine :=
sorry

end NUMINAMATH_CALUDE_identify_genuine_coin_l3901_390187


namespace NUMINAMATH_CALUDE_town_population_problem_l3901_390153

theorem town_population_problem (original : ℕ) : 
  (((original + 1500) * 85 / 100) : ℕ) = original - 45 → original = 8800 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3901_390153


namespace NUMINAMATH_CALUDE_right_triangle_median_l3901_390160

/-- Given a right triangle XYZ with ∠XYZ as the right angle, XY = 6, YZ = 8, and N as the midpoint of XZ, prove that YN = 5 -/
theorem right_triangle_median (X Y Z N : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 6^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 8^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = (X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 →
  N = ((X.1 + Z.1) / 2, (X.2 + Z.2) / 2) →
  (Y.1 - N.1)^2 + (Y.2 - N.2)^2 = 5^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_l3901_390160


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3901_390150

theorem complex_equation_solution (a : ℝ) : (Complex.I * a + 1) * (a - Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3901_390150


namespace NUMINAMATH_CALUDE_remaining_payment_prove_remaining_payment_l3901_390184

/-- Given a product with a deposit, sales tax, and processing fee, calculate the remaining amount to be paid -/
theorem remaining_payment (deposit_percentage : ℝ) (deposit_amount : ℝ) (sales_tax_percentage : ℝ) (processing_fee : ℝ) : ℝ :=
  let full_price := deposit_amount / deposit_percentage
  let sales_tax := sales_tax_percentage * full_price
  let total_additional_expenses := sales_tax + processing_fee
  full_price - deposit_amount + total_additional_expenses

/-- Prove that the remaining payment for the given conditions is $1520 -/
theorem prove_remaining_payment :
  remaining_payment 0.1 140 0.15 50 = 1520 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_prove_remaining_payment_l3901_390184


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3901_390136

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3901_390136


namespace NUMINAMATH_CALUDE_driving_time_is_55_minutes_l3901_390163

/-- Calculates the driving time per trip given total moving time, number of trips, and car filling time -/
def driving_time_per_trip (total_time_hours : ℕ) (num_trips : ℕ) (filling_time_minutes : ℕ) : ℕ :=
  let total_time_minutes := total_time_hours * 60
  let total_filling_time := num_trips * filling_time_minutes
  let total_driving_time := total_time_minutes - total_filling_time
  total_driving_time / num_trips

/-- Theorem stating that given the problem conditions, the driving time per trip is 55 minutes -/
theorem driving_time_is_55_minutes :
  driving_time_per_trip 7 6 15 = 55 := by
  sorry

#eval driving_time_per_trip 7 6 15

end NUMINAMATH_CALUDE_driving_time_is_55_minutes_l3901_390163


namespace NUMINAMATH_CALUDE_decimal_168_equals_binary_10101000_l3901_390182

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_168_equals_binary_10101000 :
  toBinary 168 = [false, false, false, true, false, true, false, true] ∧
  fromBinary [false, false, false, true, false, true, false, true] = 168 :=
by sorry

end NUMINAMATH_CALUDE_decimal_168_equals_binary_10101000_l3901_390182


namespace NUMINAMATH_CALUDE_square_tiles_count_l3901_390143

/-- Represents a collection of triangular and square tiles. -/
structure TileCollection where
  triangles : ℕ
  squares : ℕ
  total_tiles : ℕ
  total_edges : ℕ
  tiles_sum : triangles + squares = total_tiles
  edges_sum : 3 * triangles + 4 * squares = total_edges

/-- Theorem stating that in a collection of 32 tiles with 110 edges, there are 14 square tiles. -/
theorem square_tiles_count (tc : TileCollection) 
  (h1 : tc.total_tiles = 32) 
  (h2 : tc.total_edges = 110) : 
  tc.squares = 14 := by
  sorry

#check square_tiles_count

end NUMINAMATH_CALUDE_square_tiles_count_l3901_390143


namespace NUMINAMATH_CALUDE_june_design_purple_tiles_l3901_390108

/-- Represents the number of tiles of each color in June's design -/
structure TileDesign where
  total : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- Theorem stating the number of purple tiles in June's design -/
theorem june_design_purple_tiles (d : TileDesign) : 
  d.total = 20 ∧ 
  d.yellow = 3 ∧ 
  d.blue = d.yellow + 1 ∧ 
  d.white = 7 → 
  d.purple = 6 := by
  sorry

#check june_design_purple_tiles

end NUMINAMATH_CALUDE_june_design_purple_tiles_l3901_390108


namespace NUMINAMATH_CALUDE_room_width_l3901_390132

/-- Given a rectangular room with area 10 square feet and length 5 feet, prove the width is 2 feet -/
theorem room_width (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 10 → length = 5 → area = length * width → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l3901_390132


namespace NUMINAMATH_CALUDE_valid_outfit_count_l3901_390145

/-- The number of colors available for each item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- Calculates the total number of outfit combinations without restrictions -/
def total_combinations : ℕ := num_colors ^ num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of outfits where shoes don't match any other item color -/
def valid_shoe_combinations : ℕ := num_colors * num_colors * num_colors * (num_colors - 1)

/-- Calculates the number of outfits where shirt, pants, and hat are the same color, but shoes are different -/
def same_color_except_shoes : ℕ := num_colors * (num_colors - 1) - num_colors

/-- The main theorem stating the number of valid outfit combinations -/
theorem valid_outfit_count : 
  total_combinations - same_color_outfits - valid_shoe_combinations - same_color_except_shoes = 1104 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l3901_390145


namespace NUMINAMATH_CALUDE_four_number_sequence_l3901_390162

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def is_geometric_sequence (b c d : ℝ) : Prop := c * c = b * d

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : is_arithmetic_sequence a b c)
  (h2 : is_geometric_sequence b c d)
  (h3 : a + d = 16)
  (h4 : b + c = 12) :
  ((a, b, c, d) = (0, 4, 8, 16)) ∨ ((a, b, c, d) = (15, 9, 3, 1)) := by
  sorry

end NUMINAMATH_CALUDE_four_number_sequence_l3901_390162


namespace NUMINAMATH_CALUDE_number_of_tippers_l3901_390148

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def tip_amount : ℕ := 10
def total_earnings : ℕ := 558

theorem number_of_tippers : ℕ :=
  by
    sorry

end NUMINAMATH_CALUDE_number_of_tippers_l3901_390148


namespace NUMINAMATH_CALUDE_angle_sum_equality_l3901_390102

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/7) (h4 : Real.tan β = 3/79) : 5*α + 2*β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l3901_390102


namespace NUMINAMATH_CALUDE_solution_implies_q_value_l3901_390196

theorem solution_implies_q_value (q : ℚ) (h : 2 * q - 3 = 11) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_q_value_l3901_390196


namespace NUMINAMATH_CALUDE_set_union_problem_l3901_390188

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2} →
  B = {3, a} →
  A ∩ B = {1} →
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3901_390188


namespace NUMINAMATH_CALUDE_sum_area_15_disks_on_unit_circle_l3901_390128

/-- The sum of areas of 15 congruent disks covering a unit circle --/
theorem sum_area_15_disks_on_unit_circle : 
  ∃ (r : ℝ), 
    0 < r ∧ 
    (15 : ℝ) * (2 * r) = 2 * π ∧ 
    15 * (π * r^2) = π * (105 - 60 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_area_15_disks_on_unit_circle_l3901_390128


namespace NUMINAMATH_CALUDE_ceiling_cube_fraction_plus_one_l3901_390133

theorem ceiling_cube_fraction_plus_one :
  ⌈(-5/3)^3 + 1⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_cube_fraction_plus_one_l3901_390133


namespace NUMINAMATH_CALUDE_number_percentage_problem_l3901_390165

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l3901_390165


namespace NUMINAMATH_CALUDE_sine_graph_translation_l3901_390118

theorem sine_graph_translation (x : ℝ) :
  5 * Real.sin (2 * (x + π/12) + π/6) = 5 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_translation_l3901_390118


namespace NUMINAMATH_CALUDE_king_hearts_diamonds_probability_l3901_390171

/-- The number of cards in a double deck -/
def total_cards : ℕ := 104

/-- The number of King of Hearts and King of Diamonds cards in a double deck -/
def target_cards : ℕ := 4

/-- The probability of drawing a King of Hearts or King of Diamonds from a shuffled double deck -/
def probability : ℚ := target_cards / total_cards

theorem king_hearts_diamonds_probability :
  probability = 1 / 26 := by sorry

end NUMINAMATH_CALUDE_king_hearts_diamonds_probability_l3901_390171


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3901_390105

theorem imaginary_part_of_complex_division (Z : ℂ) (h : Z = 1 - 2*I) :
  (Complex.im ((1 : ℂ) + 3*I) / Z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3901_390105


namespace NUMINAMATH_CALUDE_bicyclist_scooter_meeting_time_l3901_390120

/-- Represents a vehicle with a constant speed -/
structure Vehicle where
  speed : ℝ

/-- Represents the time when two vehicles meet -/
def MeetingTime (v1 v2 : Vehicle) : ℝ → Prop := sorry

theorem bicyclist_scooter_meeting_time 
  (car motorcycle scooter bicycle : Vehicle)
  (h1 : MeetingTime car scooter 12)
  (h2 : MeetingTime car bicycle 14)
  (h3 : MeetingTime car motorcycle 16)
  (h4 : MeetingTime motorcycle scooter 17)
  (h5 : MeetingTime motorcycle bicycle 18) :
  MeetingTime bicycle scooter (12 + 10/3) :=
sorry

end NUMINAMATH_CALUDE_bicyclist_scooter_meeting_time_l3901_390120


namespace NUMINAMATH_CALUDE_slips_with_two_l3901_390161

theorem slips_with_two (total : ℕ) (expected_value : ℚ) : 
  total = 15 → expected_value = 46/10 → ∃ x y z : ℕ, 
    x + y + z = total ∧ 
    (2 * x + 5 * y + 8 * z : ℚ) / total = expected_value ∧ 
    x = 8 ∧ y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_slips_with_two_l3901_390161


namespace NUMINAMATH_CALUDE_alok_veggie_plates_l3901_390183

/-- Represents the order and payment details of Alok's meal -/
structure MealOrder where
  chapatis : Nat
  rice_plates : Nat
  ice_cream_cups : Nat
  chapati_cost : Nat
  rice_cost : Nat
  veggie_cost : Nat
  total_paid : Nat

/-- Calculates the number of mixed vegetable plates ordered -/
def veggie_plates_ordered (order : MealOrder) : Nat :=
  let known_cost := order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost
  let veggie_total_cost := order.total_paid - known_cost
  veggie_total_cost / order.veggie_cost

/-- Theorem stating that Alok ordered 11 plates of mixed vegetable -/
theorem alok_veggie_plates (order : MealOrder) 
        (h1 : order.chapatis = 16)
        (h2 : order.rice_plates = 5)
        (h3 : order.ice_cream_cups = 6)
        (h4 : order.chapati_cost = 6)
        (h5 : order.rice_cost = 45)
        (h6 : order.veggie_cost = 70)
        (h7 : order.total_paid = 1111) :
        veggie_plates_ordered order = 11 := by
  sorry

end NUMINAMATH_CALUDE_alok_veggie_plates_l3901_390183


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l3901_390176

def charlie_cycle : ℕ := 6
def dana_cycle : ℕ := 7
def total_days : ℕ := 1000

def coinciding_rest_days (c_cycle d_cycle total : ℕ) : ℕ :=
  let lcm := Nat.lcm c_cycle d_cycle
  let full_cycles := total / lcm
  let c_rest_days := 2
  let d_rest_days := 2
  let coinciding_days_per_cycle := 4  -- This should be proven, not assumed
  full_cycles * coinciding_days_per_cycle

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 92 := by
  sorry

#eval coinciding_rest_days charlie_cycle dana_cycle total_days

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l3901_390176


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3901_390137

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleBy1To10 : ℕ := 2520

/-- Checks if a number is divisible by all integers from 1 to 10 -/
def isDivisibleBy1To10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10 :
  isDivisibleBy1To10 smallestDivisibleBy1To10 ∧
  ∀ n : ℕ, 0 < n ∧ n < smallestDivisibleBy1To10 → ¬isDivisibleBy1To10 n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3901_390137


namespace NUMINAMATH_CALUDE_triangle_sin_a_l3901_390157

theorem triangle_sin_a (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  B = π / 4 →
  h = c / 3 →
  (1/2) * a * h = (1/2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  Real.sin A = a * Real.sin B / b →
  Real.sin A = 3 * Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_CALUDE_triangle_sin_a_l3901_390157


namespace NUMINAMATH_CALUDE_inequality_proof_l3901_390193

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3901_390193


namespace NUMINAMATH_CALUDE_field_trip_van_capacity_l3901_390141

theorem field_trip_van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 25 → adults = 5 → vans = 6 →
  (students + adults) / vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_capacity_l3901_390141


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3901_390101

theorem consecutive_integers_product (n : ℕ) : 
  n > 0 ∧ (n + (n + 1) < 150) → n * (n + 1) ≤ 5550 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3901_390101


namespace NUMINAMATH_CALUDE_egg_collection_difference_l3901_390180

/-- Egg collection problem -/
theorem egg_collection_difference :
  ∀ (benjamin carla trisha : ℕ),
  benjamin = 6 →
  carla = 3 * benjamin →
  benjamin + carla + trisha = 26 →
  benjamin - trisha = 4 :=
by sorry

end NUMINAMATH_CALUDE_egg_collection_difference_l3901_390180


namespace NUMINAMATH_CALUDE_part_one_part_two_l3901_390140

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a < 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Part 1
theorem part_one (a : ℝ) (h : a = 3) :
  let U := A a ∪ B
  B ∪ (U \ A a) = {x | x > -2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | A a ∩ B = B} = {a : ℝ | a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3901_390140


namespace NUMINAMATH_CALUDE_exists_permutation_multiple_of_seven_l3901_390103

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a natural number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that there exists a permutation of digits 1, 3, 7, 9 that forms a multiple of 7 -/
theorem exists_permutation_multiple_of_seven :
  ∃ (perm : List ℕ), perm ∈ permutations [1, 3, 7, 9] ∧ (list_to_number perm) % 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_multiple_of_seven_l3901_390103


namespace NUMINAMATH_CALUDE_chocolate_division_l3901_390138

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (num_friends : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  num_friends = 4 →
  (total_chocolate / num_piles) * (num_piles - 1) / num_friends = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l3901_390138
