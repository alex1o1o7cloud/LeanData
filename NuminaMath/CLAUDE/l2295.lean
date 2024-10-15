import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2295_229591

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 7) →                      -- sum condition
  (a < c) →                          -- order condition
  (a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2295_229591


namespace NUMINAMATH_CALUDE_section_area_is_28_sqrt_34_l2295_229582

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  origin : Point3D

/-- Calculates the area of the section cut by a plane in a cube -/
noncomputable def sectionArea (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The area of the section cut by plane α in the given cube is 28√34 -/
theorem section_area_is_28_sqrt_34 :
  let cube : Cube := { edge_length := 12, origin := { x := 0, y := 0, z := 0 } }
  let A : Point3D := cube.origin
  let E : Point3D := { x := 12, y := 0, z := 9 }
  let F : Point3D := { x := 0, y := 12, z := 9 }
  let plane : Plane := { a := 1, b := 1, c := -3/4, d := 0 }
  sectionArea cube plane = 28 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_section_area_is_28_sqrt_34_l2295_229582


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2295_229532

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2295_229532


namespace NUMINAMATH_CALUDE_chord_length_l2295_229507

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - d ^ 2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l2295_229507


namespace NUMINAMATH_CALUDE_no_solution_for_12x4x_divisible_by_99_l2295_229578

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_form (x : ℕ) : ℕ := 12000 + 1000 * x + 40 + x

theorem no_solution_for_12x4x_divisible_by_99 :
  ¬ ∃ x : ℕ, is_single_digit x ∧ (number_form x) % 99 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_12x4x_divisible_by_99_l2295_229578


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2295_229538

theorem product_zero_implies_factor_zero (a b : ℝ) (h : a * b = 0) :
  a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2295_229538


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2295_229557

theorem polynomial_divisibility (a b c : ℕ) :
  ∃ q : Polynomial ℚ, X^(3*a) + X^(3*b+1) + X^(3*c+2) = (X^2 + X + 1) * q :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2295_229557


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2295_229576

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ

/-- Theorem: For an arithmetic sequence with S_5 = 10 and S_10 = 30, S_15 = 60 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 5 = 10) 
  (h2 : a.S 10 = 30) : 
  a.S 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2295_229576


namespace NUMINAMATH_CALUDE_faster_car_speed_l2295_229585

/-- Given two cars traveling in opposite directions for 5 hours, with one car
    traveling 10 mi/h faster than the other, and ending up 500 miles apart,
    prove that the speed of the faster car is 55 mi/h. -/
theorem faster_car_speed (slower_speed faster_speed : ℝ) : 
  faster_speed = slower_speed + 10 →
  5 * slower_speed + 5 * faster_speed = 500 →
  faster_speed = 55 := by sorry

end NUMINAMATH_CALUDE_faster_car_speed_l2295_229585


namespace NUMINAMATH_CALUDE_spider_web_paths_l2295_229529

/-- The number of paths from (0,0) to (m,n) on a grid, moving only up and right -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the target point -/
def target : (ℕ × ℕ) := (4, 3)

theorem spider_web_paths : 
  gridPaths target.1 target.2 = 35 := by sorry

end NUMINAMATH_CALUDE_spider_web_paths_l2295_229529


namespace NUMINAMATH_CALUDE_factorization_equality_l2295_229554

theorem factorization_equality (a b : ℝ) : a * b^2 - 3 * a = a * (b + Real.sqrt 3) * (b - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2295_229554


namespace NUMINAMATH_CALUDE_profit_margin_properties_l2295_229573

/-- Profit margin calculation --/
theorem profit_margin_properties
  (B E : ℝ)  -- Purchase price and selling price
  (hE : E > B)  -- Condition: selling price is greater than purchase price
  (a : ℝ := 100 * (E - B) / B)  -- Profit margin from bottom up
  (f : ℝ := 100 * (E - B) / E)  -- Profit margin from top down
  : 
  (f = 100 * a / (a + 100) ∧ a = 100 * f / (100 - f)) ∧  -- Conversion formulas
  (a - f = a * f / 100)  -- Difference property
  := by sorry

end NUMINAMATH_CALUDE_profit_margin_properties_l2295_229573


namespace NUMINAMATH_CALUDE_workers_required_l2295_229564

/-- Given a craft factory that needs to produce 60 units per day, 
    and each worker can produce x units per day, 
    prove that the number of workers required y is equal to 60/x -/
theorem workers_required (x : ℝ) (h : x > 0) : 
  ∃ y : ℝ, y * x = 60 ∧ y = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_workers_required_l2295_229564


namespace NUMINAMATH_CALUDE_angela_marbles_l2295_229560

theorem angela_marbles :
  ∀ (a : ℕ), 
  (∃ (b c d : ℕ),
    b = 3 * a ∧
    c = 2 * b ∧
    d = 4 * c ∧
    a + b + c + d = 204) →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_angela_marbles_l2295_229560


namespace NUMINAMATH_CALUDE_num_motorcycles_in_parking_lot_l2295_229556

-- Define the number of wheels for each vehicle type
def car_wheels : ℕ := 5
def motorcycle_wheels : ℕ := 2
def tricycle_wheels : ℕ := 3

-- Define the number of cars and tricycles
def num_cars : ℕ := 19
def num_tricycles : ℕ := 11

-- Define the total number of wheels
def total_wheels : ℕ := 184

-- Theorem to prove
theorem num_motorcycles_in_parking_lot :
  ∃ (num_motorcycles : ℕ),
    num_motorcycles = 28 ∧
    num_motorcycles * motorcycle_wheels +
    num_cars * car_wheels +
    num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_num_motorcycles_in_parking_lot_l2295_229556


namespace NUMINAMATH_CALUDE_infinite_sum_theorem_l2295_229509

theorem infinite_sum_theorem (s : ℝ) (hs : s > 0) (heq : s^3 - 3/4 * s + 2 = 0) :
  ∑' n, (n + 1) * s^(2*n + 2) = 16/9 := by
sorry

end NUMINAMATH_CALUDE_infinite_sum_theorem_l2295_229509


namespace NUMINAMATH_CALUDE_pages_to_read_on_third_day_l2295_229567

/-- Given a book with 100 pages and Lance's reading progress over three days,
    prove that he needs to read 35 pages on the third day to finish the book. -/
theorem pages_to_read_on_third_day (pages_day1 pages_day2 : ℕ) 
  (h1 : pages_day1 = 35)
  (h2 : pages_day2 = pages_day1 - 5) :
  100 - (pages_day1 + pages_day2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_on_third_day_l2295_229567


namespace NUMINAMATH_CALUDE_find_unknown_number_l2295_229523

/-- Given two positive integers with known HCF and LCM, find the unknown number -/
theorem find_unknown_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 12) (h3 : Nat.lcm A B = 312) : B = 156 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l2295_229523


namespace NUMINAMATH_CALUDE_roots_shifted_polynomial_l2295_229524

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 4*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 23*x + 7 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_polynomial_l2295_229524


namespace NUMINAMATH_CALUDE_sum_binary_digits_345_l2295_229553

/-- Sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the binary digits of 345 is 5 -/
theorem sum_binary_digits_345 : sum_binary_digits 345 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_345_l2295_229553


namespace NUMINAMATH_CALUDE_marble_distribution_l2295_229522

theorem marble_distribution (n : ℕ) (x : ℕ) :
  (∀ i : ℕ, i ≤ n → i + (n * x - (i * (i + 1)) / 2) / 10 = x) →
  (n * x = n * x - (n * (n + 1)) / 2) →
  (n = 9 ∧ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2295_229522


namespace NUMINAMATH_CALUDE_company_average_salary_l2295_229540

/-- Calculates the average salary for a company given the number of managers,
    number of associates, average salary of managers, and average salary of associates. -/
def average_company_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  total_salary / total_employees

/-- Theorem stating that the average salary for the company is $40,000 -/
theorem company_average_salary :
  average_company_salary 15 75 90000 30000 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_company_average_salary_l2295_229540


namespace NUMINAMATH_CALUDE_least_common_denominator_sum_l2295_229550

theorem least_common_denominator_sum (a b c d e : ℕ) 
  (ha : a = 4) (hb : b = 5) (hc : c = 6) (hd : d = 7) (he : e = 8) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_sum_l2295_229550


namespace NUMINAMATH_CALUDE_jean_money_l2295_229535

theorem jean_money (jane : ℕ) (jean : ℕ) : 
  jean = 3 * jane → 
  jean + jane = 76 → 
  jean = 57 := by
sorry

end NUMINAMATH_CALUDE_jean_money_l2295_229535


namespace NUMINAMATH_CALUDE_largest_quantity_l2295_229506

def D : ℚ := 2007 / 2006 + 2007 / 2008
def E : ℚ := 2008 / 2007 + 2010 / 2007
def F : ℚ := 2009 / 2008 + 2009 / 2010

theorem largest_quantity : E > D ∧ E > F := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2295_229506


namespace NUMINAMATH_CALUDE_incircle_radius_l2295_229588

/-- The ellipse with semi-major axis 4 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

/-- The incircle of the inscribed triangle ABC -/
def incircle (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

/-- A is the left vertex of the ellipse -/
def A : ℝ × ℝ := (-4, 0)

/-- Theorem: The radius of the incircle is 5 -/
theorem incircle_radius : ∃ (r : ℝ), 
  (∀ x y, incircle x y r → ellipse x y) ∧ 
  (incircle A.1 A.2 r) ∧ 
  r = 5 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_l2295_229588


namespace NUMINAMATH_CALUDE_water_cooler_capacity_l2295_229587

/-- Represents the capacity of the water cooler in ounces -/
def cooler_capacity : ℕ := 126

/-- Number of linemen on the team -/
def num_linemen : ℕ := 12

/-- Number of skill position players on the team -/
def num_skill_players : ℕ := 10

/-- Amount of water each lineman drinks in ounces -/
def lineman_water : ℕ := 8

/-- Amount of water each skill position player drinks in ounces -/
def skill_player_water : ℕ := 6

/-- Number of skill position players who can drink before refill -/
def skill_players_before_refill : ℕ := 5

theorem water_cooler_capacity : 
  cooler_capacity = num_linemen * lineman_water + skill_players_before_refill * skill_player_water :=
by sorry

end NUMINAMATH_CALUDE_water_cooler_capacity_l2295_229587


namespace NUMINAMATH_CALUDE_dog_grouping_ways_l2295_229558

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

theorem dog_grouping_ways :
  let remaining_dogs := total_dogs - 2  -- Sparky and Rex are already placed
  let remaining_group1 := group1_size - 1  -- Sparky is already in group 1
  let remaining_group2 := group2_size - 1  -- Rex is already in group 2
  (Nat.choose remaining_dogs remaining_group1) *
  (Nat.choose (remaining_dogs - remaining_group1) remaining_group2) *
  (Nat.choose (remaining_dogs - remaining_group1 - remaining_group2) group3_size) = 4200 := by
sorry

end NUMINAMATH_CALUDE_dog_grouping_ways_l2295_229558


namespace NUMINAMATH_CALUDE_first_week_pushups_l2295_229533

theorem first_week_pushups (initial_pushups : ℕ) (daily_increase : ℕ) (workout_days : ℕ) : 
  initial_pushups = 10 →
  daily_increase = 5 →
  workout_days = 3 →
  (initial_pushups + (initial_pushups + daily_increase) + (initial_pushups + 2 * daily_increase)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_week_pushups_l2295_229533


namespace NUMINAMATH_CALUDE_number_thought_of_l2295_229531

theorem number_thought_of (x : ℚ) : (6 * x) / 2 - 5 = 25 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2295_229531


namespace NUMINAMATH_CALUDE_angle_bisectors_concurrent_l2295_229527

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : Point2D) : Prop := sorry

-- Define that P is an interior point of ABCD
def InteriorPoint (P : Point2D) (A B C D : Point2D) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point2D) : ℝ := sorry

-- Define the angle bisector
def AngleBisector (A B C : Point2D) : Point2D → Point2D → Prop := sorry

-- Define the perpendicular bisector of a line segment
def PerpendicularBisector (A B : Point2D) : Point2D → Point2D → Prop := sorry

-- Define when three lines are concurrent
def Concurrent (L1 L2 L3 : Point2D → Point2D → Prop) : Prop := sorry

theorem angle_bisectors_concurrent 
  (A B C D P : Point2D) 
  (h1 : Quadrilateral A B C D)
  (h2 : InteriorPoint P A B C D)
  (h3 : Angle P A D / Angle P B A / Angle D P A = 1 / 2 / 3)
  (h4 : Angle C B P / Angle B A P / Angle B P C = 1 / 2 / 3) :
  Concurrent 
    (AngleBisector A D P) 
    (AngleBisector P C B) 
    (PerpendicularBisector A B) := by sorry

end NUMINAMATH_CALUDE_angle_bisectors_concurrent_l2295_229527


namespace NUMINAMATH_CALUDE_friends_team_assignment_l2295_229519

theorem friends_team_assignment : 
  let n : ℕ := 8  -- number of friends
  let k : ℕ := 4  -- number of teams
  k ^ n = 65536 := by sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l2295_229519


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2295_229593

theorem diophantine_equation_solution (x y z : ℕ) (h : x^2 + 3*y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2295_229593


namespace NUMINAMATH_CALUDE_baylor_payment_multiple_l2295_229589

theorem baylor_payment_multiple :
  let initial_amount : ℕ := 4000
  let first_client_payment : ℕ := initial_amount / 2
  let second_client_payment : ℕ := first_client_payment + (2 * first_client_payment) / 5
  let combined_payment : ℕ := first_client_payment + second_client_payment
  let final_total : ℕ := 18400
  let third_client_multiple : ℕ := (final_total - initial_amount - combined_payment) / combined_payment
  third_client_multiple = 2 := by sorry

end NUMINAMATH_CALUDE_baylor_payment_multiple_l2295_229589


namespace NUMINAMATH_CALUDE_ashleys_friends_ages_sum_l2295_229584

theorem ashleys_friends_ages_sum :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 36 ∧ c * d = 30) ∨ (a * c = 36 ∧ b * d = 30) ∨ (a * d = 36 ∧ b * c = 30) →
    a + b + c + d = 24 :=
by sorry

end NUMINAMATH_CALUDE_ashleys_friends_ages_sum_l2295_229584


namespace NUMINAMATH_CALUDE_intersection_union_theorem_l2295_229548

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 12 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3*x + 2*b = 0}
def C : Set ℝ := {2, -3}

-- State the theorem
theorem intersection_union_theorem (a b : ℝ) :
  (A a ∩ B b = {2}) →
  (A a = {2, 6}) →
  (B b = {-5, 2}) →
  ((A a ∪ B b) ∩ C = {2}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_theorem_l2295_229548


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l2295_229571

/-- Calculates the savings when using a discount clinic compared to a normal doctor visit -/
theorem discount_clinic_savings
  (normal_cost : ℝ)
  (discount_percentage : ℝ)
  (discount_visits : ℕ)
  (h1 : normal_cost = 200)
  (h2 : discount_percentage = 0.7)
  (h3 : discount_visits = 2) :
  normal_cost - discount_visits * (normal_cost * (1 - discount_percentage)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l2295_229571


namespace NUMINAMATH_CALUDE_peanuts_in_jar_l2295_229511

theorem peanuts_in_jar (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_fraction : ℚ) (carlos_peanuts : ℕ) : 
  initial_peanuts = 220 →
  brock_fraction = 1/4 →
  bonita_fraction = 2/5 →
  carlos_peanuts = 17 →
  initial_peanuts - 
    (initial_peanuts * brock_fraction).floor - 
    ((initial_peanuts - (initial_peanuts * brock_fraction).floor) * bonita_fraction).floor - 
    carlos_peanuts = 82 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_jar_l2295_229511


namespace NUMINAMATH_CALUDE_sum_upper_bound_l2295_229570

theorem sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_upper_bound_l2295_229570


namespace NUMINAMATH_CALUDE_inequality_proof_l2295_229552

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2295_229552


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2295_229581

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  Complex.im z = 3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2295_229581


namespace NUMINAMATH_CALUDE_problem_2003_2001_l2295_229521

theorem problem_2003_2001 : 2003^3 - 2001 * 2003^2 - 2001^2 * 2003 + 2001^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_2003_2001_l2295_229521


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l2295_229520

theorem sqrt_difference_approximation : 
  |Real.sqrt (49 + 121) - Real.sqrt (64 - 36) - 7.75| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l2295_229520


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2295_229559

theorem boys_to_girls_ratio (S : ℚ) (G : ℚ) (B : ℚ) : 
  S > 0 → G > 0 → B > 0 →
  S = G + B →
  (1 / 2) * G = (1 / 5) * S →
  B / G = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2295_229559


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2295_229555

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) ↔ (¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2295_229555


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2295_229541

theorem fraction_subtraction : (8 : ℚ) / 24 - (5 : ℚ) / 40 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2295_229541


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l2295_229583

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_pairs : ℕ) :
  total_students = 148 →
  red_students = 65 →
  green_students = 83 →
  total_pairs = 74 →
  red_pairs = 27 →
  red_students + green_students = total_students →
  2 * total_pairs = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 36 ∧ 
    red_pairs + green_pairs + (total_students - 2 * (red_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l2295_229583


namespace NUMINAMATH_CALUDE_ratio_problem_l2295_229586

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + b) / (b + c) = 4/15 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2295_229586


namespace NUMINAMATH_CALUDE_triangle_inequality_l2295_229502

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 - a*b = c^2) : 
  (a - c) * (b - c) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2295_229502


namespace NUMINAMATH_CALUDE_bakery_problem_proof_l2295_229544

/-- Given the total number of muffins, muffins per box, and available boxes, 
    calculate the number of additional boxes needed --/
def additional_boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : ℕ :=
  (total_muffins / muffins_per_box) - available_boxes

/-- Proof that 9 additional boxes are needed for the given bakery problem --/
theorem bakery_problem_proof : 
  additional_boxes_needed 95 5 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bakery_problem_proof_l2295_229544


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2295_229579

theorem square_plus_reciprocal_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2295_229579


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l2295_229525

/-- The ratio of cylinder volumes formed from a 5x8 rectangle -/
theorem cylinder_volume_ratio : 
  ∀ (h₁ h₂ r₁ r₂ : ℝ), 
    h₁ = 8 ∧ h₂ = 5 ∧ 
    2 * Real.pi * r₁ = 5 ∧ 
    2 * Real.pi * r₂ = 8 →
    max (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) / 
    min (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_volume_ratio_l2295_229525


namespace NUMINAMATH_CALUDE_digit_150_is_2_l2295_229569

/-- The sequence of digits formed by concatenating all integers from 100 down to 50 -/
def digit_sequence : List Nat := sorry

/-- The 150th digit in the sequence -/
def digit_150 : Nat := sorry

/-- Theorem stating that the 150th digit in the sequence is 2 -/
theorem digit_150_is_2 : digit_150 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_2_l2295_229569


namespace NUMINAMATH_CALUDE_number_ratio_l2295_229566

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2295_229566


namespace NUMINAMATH_CALUDE_roses_cost_l2295_229512

theorem roses_cost (dozen : ℕ) (price_per_rose : ℚ) (discount_rate : ℚ) : 
  dozen * 12 * price_per_rose * discount_rate = 288 :=
by
  -- Assuming dozen = 5, price_per_rose = 6, and discount_rate = 0.8
  sorry

#check roses_cost

end NUMINAMATH_CALUDE_roses_cost_l2295_229512


namespace NUMINAMATH_CALUDE_point_D_coordinates_l2295_229562

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℚ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a b : Point) : Prop :=
  distance a p + distance p b = distance a b

theorem point_D_coordinates :
  let X : Point := ⟨-2, 1⟩
  let Y : Point := ⟨4, 9⟩
  ∀ D : Point,
    isOnSegment D X Y →
    distance X D = 2 * distance Y D →
    D.x = 2 ∧ D.y = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l2295_229562


namespace NUMINAMATH_CALUDE_pool_capacity_l2295_229513

/-- The capacity of a swimming pool given specific valve filling rates. -/
theorem pool_capacity 
  (fill_time : ℝ) 
  (valve_a_time : ℝ) 
  (valve_b_time : ℝ) 
  (valve_c_rate_diff : ℝ) 
  (valve_b_rate_diff : ℝ) 
  (h1 : fill_time = 40) 
  (h2 : valve_a_time = 180) 
  (h3 : valve_b_time = 240) 
  (h4 : valve_c_rate_diff = 75) 
  (h5 : valve_b_rate_diff = 60) : 
  ∃ T : ℝ, T = 16200 ∧ 
    T / fill_time = T / valve_a_time + T / valve_b_time + (T / valve_a_time + valve_c_rate_diff) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l2295_229513


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2295_229500

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 94) :
  x^2 + y^2 = 7540 / 81 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2295_229500


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2295_229536

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2295_229536


namespace NUMINAMATH_CALUDE_profitable_iff_price_ge_132_l2295_229542

/-- The transaction fee rate for stock trading in China -/
def fee_rate : ℚ := 75 / 10000

/-- The number of shares traded -/
def num_shares : ℕ := 1000

/-- The price increase per share -/
def price_increase : ℚ := 2

/-- Determines if a stock transaction is profitable given the initial price -/
def is_profitable (x : ℚ) : Prop :=
  (x + price_increase) * (1 - fee_rate) * num_shares ≥ (1 + fee_rate) * num_shares * x

/-- Theorem: The transaction is profitable if and only if the initial share price is at least 132 yuan -/
theorem profitable_iff_price_ge_132 (x : ℚ) : is_profitable x ↔ x ≥ 132 := by
  sorry

end NUMINAMATH_CALUDE_profitable_iff_price_ge_132_l2295_229542


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_30_50_l2295_229549

theorem smallest_divisible_by_18_30_50 : 
  ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 30 ∣ n ∧ 50 ∣ n → n ≥ 450 :=
by
  sorry

#check smallest_divisible_by_18_30_50

end NUMINAMATH_CALUDE_smallest_divisible_by_18_30_50_l2295_229549


namespace NUMINAMATH_CALUDE_min_value_expression_l2295_229510

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 19 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2295_229510


namespace NUMINAMATH_CALUDE_f_composition_proof_l2295_229526

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_proof : f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_proof_l2295_229526


namespace NUMINAMATH_CALUDE_right_triangle_from_medians_l2295_229590

theorem right_triangle_from_medians (m₁ m₂ m₃ : ℝ) 
  (h₁ : m₁ = 5)
  (h₂ : m₂ = Real.sqrt 52)
  (h₃ : m₃ = Real.sqrt 73) :
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
    m₁^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m₂^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m₃^2 = (2*a^2 + 2*b^2 - c^2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_medians_l2295_229590


namespace NUMINAMATH_CALUDE_milk_fraction_after_transfers_l2295_229575

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the problem setup --/
def initial_setup : CupContents × CupContents :=
  ({ tea := 8, milk := 0 }, { tea := 0, milk := 8 })

/-- Transfers a fraction of tea from the first cup to the second --/
def transfer_tea (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let transfer_amount := cup1.tea * fraction
  ({ tea := cup1.tea - transfer_amount, milk := cup1.milk },
   { tea := cup2.tea + transfer_amount, milk := cup2.milk })

/-- Transfers a fraction of the mixture from the second cup to the first --/
def transfer_mixture (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let total2 := cup2.tea + cup2.milk
  let transfer_tea := cup2.tea * fraction
  let transfer_milk := cup2.milk * fraction
  ({ tea := cup1.tea + transfer_tea, milk := cup1.milk + transfer_milk },
   { tea := cup2.tea - transfer_tea, milk := cup2.milk - transfer_milk })

/-- Calculates the fraction of milk in a cup --/
def milk_fraction (cup : CupContents) : ℚ :=
  cup.milk / (cup.tea + cup.milk)

/-- The main theorem to prove --/
theorem milk_fraction_after_transfers :
  let cups1 := transfer_tea initial_setup (1/4)
  let cups2 := transfer_mixture cups1 (1/3)
  milk_fraction cups2.fst = 1/3 := by sorry


end NUMINAMATH_CALUDE_milk_fraction_after_transfers_l2295_229575


namespace NUMINAMATH_CALUDE_store_profit_percentage_l2295_229565

/-- Proves that the profit percentage is 30% given the conditions of the problem -/
theorem store_profit_percentage (cost_price : ℝ) (sale_price : ℝ) :
  cost_price = 20 →
  sale_price = 13 →
  ∃ (selling_price : ℝ),
    selling_price = cost_price * (1 + 30 / 100) ∧
    sale_price = selling_price / 2 :=
by sorry

end NUMINAMATH_CALUDE_store_profit_percentage_l2295_229565


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2295_229577

theorem sum_of_reciprocals_of_roots (m n : ℝ) : 
  m^2 + 2*m - 3 = 0 → n^2 + 2*n - 3 = 0 → m ≠ 0 → n ≠ 0 → 1/m + 1/n = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2295_229577


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2295_229543

/-- Represents a shape formed by unit cubes -/
structure CubeShape where
  cubes : ℕ
  central_cube : Bool
  surrounding_cubes : ℕ

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 5

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { cubes := 8
  , central_cube := true
  , surrounding_cubes := 7 }

theorem volume_to_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2295_229543


namespace NUMINAMATH_CALUDE_max_quarters_problem_l2295_229574

theorem max_quarters_problem :
  ∃! q : ℕ, 8 < q ∧ q < 60 ∧
  q % 4 = 2 ∧
  q % 7 = 3 ∧
  q % 9 = 2 ∧
  q = 38 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_problem_l2295_229574


namespace NUMINAMATH_CALUDE_rectangle_division_perimeter_paradox_l2295_229514

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Theorem stating that there exists a rectangle with non-integer perimeter
    that can be divided into rectangles with integer perimeters -/
theorem rectangle_division_perimeter_paradox :
  ∃ (big : Rectangle) (small1 small2 : Rectangle),
    ¬isInteger big.perimeter ∧
    isInteger small1.perimeter ∧
    isInteger small2.perimeter ∧
    big.width = small1.width ∧
    big.width = small2.width ∧
    big.height = small1.height + small2.height :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_perimeter_paradox_l2295_229514


namespace NUMINAMATH_CALUDE_fruit_box_problem_l2295_229539

theorem fruit_box_problem (total_fruit oranges peaches apples : ℕ) : 
  total_fruit = 56 →
  oranges = total_fruit / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_box_problem_l2295_229539


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l2295_229599

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  missing_squares : ℕ

/-- A domino covers exactly two adjacent squares. -/
def domino_area : ℕ := 2

/-- The total number of squares in a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols - board.missing_squares

/-- A checkerboard can be completely covered by dominoes if and only if
    it has an even number of squares. -/
theorem checkerboard_coverage (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_area ↔ Even (total_squares board) := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l2295_229599


namespace NUMINAMATH_CALUDE_inequality_proof_l2295_229580

theorem inequality_proof (b a : ℝ) : 
  (4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1)) ∧ 
  (a - a * |(-a^2 - 1)| < 1 - a^2 * (a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2295_229580


namespace NUMINAMATH_CALUDE_donut_calculation_l2295_229563

def total_donuts (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts : ℕ) : ℕ :=
  let total_friends := initial_friends + additional_friends
  let donuts_for_friends := total_friends * (donuts_per_friend + extra_donuts)
  let donuts_for_andrew := donuts_per_friend + extra_donuts
  donuts_for_friends + donuts_for_andrew

theorem donut_calculation :
  total_donuts 2 2 3 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_donut_calculation_l2295_229563


namespace NUMINAMATH_CALUDE_problem_solution_l2295_229508

theorem problem_solution : ∃ x : ℝ, 
  ((35 * x)^2 / 100) * x = (23/18) / 100 * 9500 - 175 ∧ 
  abs (x + 0.62857) < 0.00001 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2295_229508


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2295_229501

theorem complex_exponential_to_rectangular : 2 * Real.sqrt 3 * Complex.exp (Complex.I * (13 * Real.pi / 6)) = 3 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2295_229501


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2295_229568

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 3 = 4*y*(y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2295_229568


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2295_229503

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk 1 2 * a + Complex.mk b 0 = Complex.I * 2) → (a = 1 ∧ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2295_229503


namespace NUMINAMATH_CALUDE_log_inequality_l2295_229515

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.log (1/a) / Real.log 0.3 > Real.log (1/b) / Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2295_229515


namespace NUMINAMATH_CALUDE_units_digit_product_l2295_229561

theorem units_digit_product : (17 * 59 * 23) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l2295_229561


namespace NUMINAMATH_CALUDE_choose_formula_l2295_229528

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of ways to choose k items from n items is given by n! / (k!(n-k)!) -/
theorem choose_formula (n k : ℕ) (h : k ≤ n) :
  choose n k = Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_choose_formula_l2295_229528


namespace NUMINAMATH_CALUDE_target_row_sum_equals_2011_squared_l2295_229592

/-- The row number where the sum of all numbers equals 2011² -/
def target_row : ℕ := 1006

/-- The number of elements in the nth row -/
def num_elements (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of elements in the nth row -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1)^2

/-- Theorem stating that the target_row is the row where the sum equals 2011² -/
theorem target_row_sum_equals_2011_squared :
  row_sum target_row = 2011^2 :=
sorry

end NUMINAMATH_CALUDE_target_row_sum_equals_2011_squared_l2295_229592


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l2295_229537

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l2295_229537


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_equivalence_l2295_229596

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def givenLine : ParametricLine where
  x := λ t => 5 + 3 * t
  y := λ t => 10 - 4 * t

/-- The Cartesian form of a line: ax + by + c = 0 -/
structure CartesianLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The Cartesian line we want to prove is equivalent. -/
def targetLine : CartesianLine where
  a := 4
  b := 3
  c := -50

/-- 
Theorem: The given parametric line is equivalent to the target Cartesian line.
-/
theorem parametric_to_cartesian_equivalence :
  ∀ t : ℝ, 
  4 * (givenLine.x t) + 3 * (givenLine.y t) - 50 = 0 :=
by
  sorry

#check parametric_to_cartesian_equivalence

end NUMINAMATH_CALUDE_parametric_to_cartesian_equivalence_l2295_229596


namespace NUMINAMATH_CALUDE_insufficient_album_capacity_l2295_229572

/-- Represents the capacity and quantity of each album type -/
structure AlbumType where
  capacity : ℕ
  quantity : ℕ

/-- Proves that the total capacity of all available albums is less than the total number of pictures -/
theorem insufficient_album_capacity 
  (type_a : AlbumType)
  (type_b : AlbumType)
  (type_c : AlbumType)
  (total_pictures : ℕ)
  (h1 : type_a.capacity = 12)
  (h2 : type_a.quantity = 6)
  (h3 : type_b.capacity = 18)
  (h4 : type_b.quantity = 4)
  (h5 : type_c.capacity = 24)
  (h6 : type_c.quantity = 3)
  (h7 : total_pictures = 480) :
  type_a.capacity * type_a.quantity + 
  type_b.capacity * type_b.quantity + 
  type_c.capacity * type_c.quantity < total_pictures :=
by sorry

end NUMINAMATH_CALUDE_insufficient_album_capacity_l2295_229572


namespace NUMINAMATH_CALUDE_combination_equality_l2295_229545

theorem combination_equality (n : ℕ) : 
  Nat.choose n 14 = Nat.choose n 4 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2295_229545


namespace NUMINAMATH_CALUDE_miniou_circuit_nodes_l2295_229504

/-- Definition of a Miniou circuit -/
structure MiniouCircuit where
  nodes : ℕ
  wires : ℕ
  wire_connects_two_nodes : True
  at_most_one_wire_between_nodes : True
  three_wires_per_node : True

/-- Theorem: A Miniou circuit with 13788 wires has 9192 nodes -/
theorem miniou_circuit_nodes (c : MiniouCircuit) (h : c.wires = 13788) : c.nodes = 9192 := by
  sorry

end NUMINAMATH_CALUDE_miniou_circuit_nodes_l2295_229504


namespace NUMINAMATH_CALUDE_six_digit_number_divisibility_l2295_229547

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by appending double of a three-digit number -/
def makeSixDigitNumber (n : ThreeDigitNumber) : Nat :=
  1000 * n.toNat + 2 * n.toNat

theorem six_digit_number_divisibility (n : ThreeDigitNumber) :
  (∃ k : Nat, makeSixDigitNumber n = 2 * k) ∧
  (∃ m : Nat, makeSixDigitNumber n = 3 * m ↔ ∃ l : Nat, n.toNat = 3 * l) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_divisibility_l2295_229547


namespace NUMINAMATH_CALUDE_flower_bed_perimeter_reduction_l2295_229546

/-- Represents a rectangular flower bed with length and width -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ := 2 * (fb.length + fb.width)

/-- Theorem: The perimeter of a rectangular flower bed decreases by 17.5% 
    after reducing the length by 28% and the width by 28% -/
theorem flower_bed_perimeter_reduction (fb : FlowerBed) :
  let reduced_fb := FlowerBed.mk (fb.length * 0.72) (fb.width * 0.72)
  (perimeter fb - perimeter reduced_fb) / perimeter fb = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_perimeter_reduction_l2295_229546


namespace NUMINAMATH_CALUDE_x_squared_divides_x_plus_y_l2295_229597

theorem x_squared_divides_x_plus_y (x y : ℕ) :
  x^2 ∣ (x^2 + x*y + x + y) → x^2 ∣ (x + y) := by
sorry

end NUMINAMATH_CALUDE_x_squared_divides_x_plus_y_l2295_229597


namespace NUMINAMATH_CALUDE_min_value_problem_l2295_229516

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) :
  (x + 1 / y) * (x + 1 / z) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2295_229516


namespace NUMINAMATH_CALUDE_quadratic_inequality_holds_for_all_x_l2295_229551

theorem quadratic_inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ -2 < k ∧ k < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_holds_for_all_x_l2295_229551


namespace NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l2295_229595

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)

def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → magnitude_squared (vector_a x) = 25) ∧
  (∃ x : ℝ, x ≠ 4 ∧ magnitude_squared (vector_a x) = 25) := by
  sorry

end NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l2295_229595


namespace NUMINAMATH_CALUDE_guppies_count_l2295_229534

-- Define the number of guppies each person has
def haylee_guppies : ℕ := 3 * 12 -- 3 dozen
def jose_guppies : ℕ := haylee_guppies / 2
def charliz_guppies : ℕ := jose_guppies / 3
def nicolai_guppies : ℕ := charliz_guppies * 4

-- Define the total number of guppies
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

-- Theorem to prove
theorem guppies_count : total_guppies = 84 := by
  sorry

end NUMINAMATH_CALUDE_guppies_count_l2295_229534


namespace NUMINAMATH_CALUDE_stratified_sample_older_45_correct_l2295_229517

/-- Calculates the number of employees older than 45 to be drawn in a stratified sample -/
def stratifiedSampleOlder45 (totalEmployees : ℕ) (employeesOlder45 : ℕ) (sampleSize : ℕ) : ℕ :=
  (employeesOlder45 * sampleSize) / totalEmployees

/-- Proves that the stratified sample for employees older than 45 is correct -/
theorem stratified_sample_older_45_correct :
  stratifiedSampleOlder45 400 160 50 = 20 := by
  sorry

#eval stratifiedSampleOlder45 400 160 50

end NUMINAMATH_CALUDE_stratified_sample_older_45_correct_l2295_229517


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_2x_plus_y_eq_7_l2295_229598

def is_solution (x y : ℕ) : Prop := 2 * x + y = 7

theorem positive_integer_solutions_of_2x_plus_y_eq_7 :
  {(x, y) : ℕ × ℕ | is_solution x y ∧ x > 0 ∧ y > 0} = {(1, 5), (2, 3), (3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_2x_plus_y_eq_7_l2295_229598


namespace NUMINAMATH_CALUDE_dinner_cakes_l2295_229518

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_l2295_229518


namespace NUMINAMATH_CALUDE_sunset_colors_l2295_229594

/-- Represents the number of colors in a quick shift -/
def quick_colors : ℕ := 5

/-- Represents the number of colors in a slow shift -/
def slow_colors : ℕ := 2

/-- Represents the duration of each shift in minutes -/
def shift_duration : ℕ := 10

/-- Represents the duration of a complete cycle (quick + slow) in minutes -/
def cycle_duration : ℕ := 2 * shift_duration

/-- Represents the duration of the sunset in minutes -/
def sunset_duration : ℕ := 2 * 60

/-- Represents the number of cycles in the sunset -/
def num_cycles : ℕ := sunset_duration / cycle_duration

/-- Represents the total number of colors in one cycle -/
def colors_per_cycle : ℕ := quick_colors + slow_colors

/-- Theorem stating that the total number of colors seen during the sunset is 42 -/
theorem sunset_colors : num_cycles * colors_per_cycle = 42 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_l2295_229594


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2295_229530

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 4 * Real.cos θ - 3 * Real.sin θ

-- State the theorem
theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (r θ : ℝ), polar_equation r θ ↔ 
      (r * Real.cos θ - center.1)^2 + (r * Real.sin θ - center.2)^2 = radius^2) ∧
    (π * radius^2 = 25 * π / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2295_229530


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2295_229505

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 200)
  (h_time : time = 4) :
  ∃ (rate : ℚ), simple_interest principal time rate = interest ∧ rate = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2295_229505
