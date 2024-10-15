import Mathlib

namespace NUMINAMATH_CALUDE_trip_duration_proof_l940_94055

/-- The battery life in standby mode (in hours) -/
def standby_life : ℝ := 210

/-- The rate at which the battery depletes while talking compared to standby mode -/
def talking_depletion_rate : ℝ := 35

/-- Calculates the total trip duration given the time spent talking -/
def total_trip_duration (talking_time : ℝ) : ℝ := 2 * talking_time

/-- Theorem stating that the total trip duration is 11 hours and 40 minutes -/
theorem trip_duration_proof :
  ∃ (talking_time : ℝ),
    talking_time > 0 ∧
    talking_time ≤ standby_life ∧
    talking_depletion_rate * (standby_life - talking_time) = talking_time ∧
    total_trip_duration talking_time = 11 + 40 / 60 :=
by sorry

end NUMINAMATH_CALUDE_trip_duration_proof_l940_94055


namespace NUMINAMATH_CALUDE_modified_arithmetic_sum_l940_94068

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem modified_arithmetic_sum :
  3 * (arithmetic_sum 110 119 10) = 3435 :=
by sorry

end NUMINAMATH_CALUDE_modified_arithmetic_sum_l940_94068


namespace NUMINAMATH_CALUDE_problem_statement_l940_94091

theorem problem_statement (a b : ℝ) (h : a + b - 1 = 0) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l940_94091


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l940_94057

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (3 * x) + 15 + (3 * x + 6)) / 5 = 26 → x = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l940_94057


namespace NUMINAMATH_CALUDE_remainder_8_pow_2012_mod_10_l940_94056

/-- Definition of exponentiation --/
def pow (a : ℕ) (n : ℕ) : ℕ := (a : ℕ) ^ n

/-- The remainder when 8^2012 is divided by 10 --/
theorem remainder_8_pow_2012_mod_10 : pow 8 2012 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_8_pow_2012_mod_10_l940_94056


namespace NUMINAMATH_CALUDE_keith_initial_cards_l940_94083

/-- Represents the number of cards in Keith's collection --/
structure CardCollection where
  initial : ℕ
  added : ℕ
  remaining : ℕ

/-- Theorem stating the initial number of cards in Keith's collection --/
theorem keith_initial_cards (c : CardCollection) 
  (h1 : c.added = 8)
  (h2 : c.remaining = 46)
  (h3 : c.remaining * 2 = c.initial + c.added) :
  c.initial = 84 := by
  sorry

end NUMINAMATH_CALUDE_keith_initial_cards_l940_94083


namespace NUMINAMATH_CALUDE_future_years_calculation_l940_94005

/-- The number of years in the future when Shekhar will be 26 years old -/
def future_years : ℕ := 6

/-- Shekhar's current age -/
def shekhar_current_age : ℕ := 20

/-- Shobha's current age -/
def shobha_current_age : ℕ := 15

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio : ℚ := 4 / 3

theorem future_years_calculation :
  (shekhar_current_age + future_years = 26) ∧
  (shekhar_current_age : ℚ) / shobha_current_age = age_ratio :=
by sorry

end NUMINAMATH_CALUDE_future_years_calculation_l940_94005


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l940_94025

theorem lcm_gcf_problem (n m : ℕ+) 
  (h1 : Nat.lcm n m = 56)
  (h2 : Nat.gcd n m = 10)
  (h3 : n = 40) :
  m = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l940_94025


namespace NUMINAMATH_CALUDE_min_value_quadratic_l940_94047

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 + 3*x - 5*y ≥ -17/2 ∧ 
  ∃ x y : ℝ, x^2 + 2*x*y + 2*y^2 + 3*x - 5*y = -17/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l940_94047


namespace NUMINAMATH_CALUDE_simplify_power_l940_94035

theorem simplify_power (y : ℝ) : (3 * y^2)^4 = 81 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l940_94035


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l940_94069

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.7 →
  summer_discount = 0.2 →
  (list_price * (1 - max_regular_discount) - list_price * summer_discount) / list_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l940_94069


namespace NUMINAMATH_CALUDE_part_one_part_two_l940_94003

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Part 1: Prove that if A = 30°, then a = 5/3
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 30 * π / 180) :
  t.a = 5/3 := by sorry

-- Part 2: Prove that the maximum area of the triangle is 3
theorem part_two (t : Triangle) (h : triangle_conditions t) :
  (∃ (max_area : ℝ), max_area = 3 ∧ 
    ∀ (t' : Triangle), triangle_conditions t' → 
      1/2 * t'.a * t'.c * Real.sin t'.B ≤ max_area) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l940_94003


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l940_94001

/-- The number of fifth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom

theorem student_guinea_pig_difference :
  total_students - total_guinea_pigs = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l940_94001


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l940_94012

-- Define the lines and points
def l₁ (m : ℝ) := {(x, y) : ℝ × ℝ | (y - m) / (x + 2) = (4 - m) / (m + 2)}
def l₂ := {(x, y) : ℝ × ℝ | 2*x + y - 1 = 0}
def l₃ (n : ℝ) := {(x, y) : ℝ × ℝ | x + n*y + 1 = 0}

def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Define the theorem
theorem parallel_perpendicular_lines (m n : ℝ) : 
  (A m ∈ l₁ m) → 
  (B m ∈ l₁ m) → 
  (∀ (x y : ℝ), (x, y) ∈ l₁ m ↔ (x, y) ∈ l₂) → 
  (∀ (x y : ℝ), (x, y) ∈ l₂ → (x, y) ∈ l₃ n → x = y) → 
  m + n = -10 := by
  sorry

#check parallel_perpendicular_lines

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l940_94012


namespace NUMINAMATH_CALUDE_compute_expression_l940_94042

theorem compute_expression : 9 + 4 * (5 - 2 * 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l940_94042


namespace NUMINAMATH_CALUDE_two_numbers_problem_l940_94060

theorem two_numbers_problem (x y : ℕ+) : 
  x + y = 667 →
  Nat.lcm x y / Nat.gcd x y = 120 →
  ((x = 232 ∧ y = 435) ∨ (x = 552 ∧ y = 115)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l940_94060


namespace NUMINAMATH_CALUDE_xiao_ming_walk_relation_l940_94039

/-- Represents the relationship between remaining distance and time walked
    for a person walking towards a destination. -/
def distance_time_relation (total_distance : ℝ) (speed : ℝ) (x : ℝ) : ℝ :=
  total_distance - speed * x

/-- Theorem stating the relationship between remaining distance and time walked
    for Xiao Ming's walk to school. -/
theorem xiao_ming_walk_relation :
  ∀ x y : ℝ, y = distance_time_relation 1200 70 x ↔ y = -70 * x + 1200 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_walk_relation_l940_94039


namespace NUMINAMATH_CALUDE_inequality_proof_l940_94031

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l940_94031


namespace NUMINAMATH_CALUDE_jenny_bottle_payment_l940_94008

/-- Calculates the payment per bottle for Jenny's recycling --/
def payment_per_bottle (bottle_weight can_weight total_weight can_count can_payment total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - can_count * can_weight
  let bottle_count := remaining_weight / bottle_weight
  let can_total_payment := can_count * can_payment
  let bottle_total_payment := total_payment - can_total_payment
  bottle_total_payment / bottle_count

/-- Theorem stating that Jenny's payment per bottle is 10 cents --/
theorem jenny_bottle_payment :
  payment_per_bottle 6 2 100 20 3 160 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_bottle_payment_l940_94008


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l940_94085

theorem sufficient_condition_range (a : ℝ) : 
  (a > 0) →
  (∀ x : ℝ, (|x - 4| > 6 → x^2 - 2*x + 1 - a^2 > 0)) →
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ |x - 4| ≤ 6) →
  (0 < a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l940_94085


namespace NUMINAMATH_CALUDE_largest_unique_solution_m_l940_94096

theorem largest_unique_solution_m (x y : ℕ) (m : ℕ) : 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = m) → m ≤ 2 * 2005 * 2007 ∧ 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = 2 * 2005 * 2007) :=
sorry

end NUMINAMATH_CALUDE_largest_unique_solution_m_l940_94096


namespace NUMINAMATH_CALUDE_fewer_blue_chairs_than_yellow_l940_94014

/-- Represents the number of chairs of each color in Rodrigo's classroom -/
structure ClassroomChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ

def total_chairs (c : ClassroomChairs) : ℕ := c.red + c.yellow + c.blue

theorem fewer_blue_chairs_than_yellow (c : ClassroomChairs) 
  (h1 : c.red = 4)
  (h2 : c.yellow = 2 * c.red)
  (h3 : total_chairs c - 3 = 15) :
  c.yellow - c.blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_blue_chairs_than_yellow_l940_94014


namespace NUMINAMATH_CALUDE_star_four_three_l940_94024

/-- Definition of the star operation -/
def star (a b : ℤ) : ℤ := a^2 + a*b - b^3

/-- Theorem stating that 4 ⋆ 3 = 1 -/
theorem star_four_three : star 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l940_94024


namespace NUMINAMATH_CALUDE_perfect_square_expression_l940_94089

theorem perfect_square_expression : ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l940_94089


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l940_94098

theorem max_sum_under_constraints (a b : ℝ) :
  (4 * a + 3 * b ≤ 10) →
  (3 * a + 6 * b ≤ 12) →
  a + b ≤ 14 / 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l940_94098


namespace NUMINAMATH_CALUDE_first_group_size_l940_94040

/-- The amount of work done by one person in one day -/
def work_per_person_per_day : ℝ := 1

/-- The number of days to complete the work -/
def days : ℕ := 7

/-- The number of persons in the second group -/
def persons_second_group : ℕ := 9

/-- The amount of work completed by the first group -/
def work_first_group : ℕ := 7

/-- The amount of work completed by the second group -/
def work_second_group : ℕ := 9

/-- The number of persons in the first group -/
def persons_first_group : ℕ := 9

theorem first_group_size :
  persons_first_group * days * work_per_person_per_day = work_first_group ∧
  persons_second_group * days * work_per_person_per_day = work_second_group →
  persons_first_group = 9 := by
sorry

end NUMINAMATH_CALUDE_first_group_size_l940_94040


namespace NUMINAMATH_CALUDE_stratified_sample_middle_school_l940_94075

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a stratified sampling plan -/
structure StratifiedSample :=
  (schoolA : School)
  (schoolB : School)
  (schoolC : School)
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (isArithmeticSequence : schoolA.students + schoolC.students = 2 * schoolB.students)

/-- The theorem statement -/
theorem stratified_sample_middle_school 
  (sample : StratifiedSample)
  (h1 : sample.totalStudents = 1500)
  (h2 : sample.sampleSize = 120) :
  ∃ (d : ℕ), 
    sample.schoolA.students = 40 - d ∧ 
    sample.schoolB.students = 40 ∧ 
    sample.schoolC.students = 40 + d :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_middle_school_l940_94075


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l940_94054

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l940_94054


namespace NUMINAMATH_CALUDE_perfect_square_factors_450_l940_94073

/-- The number of perfect square factors of 450 -/
def num_perfect_square_factors : ℕ := 4

/-- The prime factorization of 450 -/
def factorization_450 : List (ℕ × ℕ) := [(2, 1), (3, 2), (5, 2)]

/-- Theorem stating that the number of perfect square factors of 450 is 4 -/
theorem perfect_square_factors_450 :
  (List.prod (List.map (fun (p : ℕ × ℕ) => p.1 ^ p.2) factorization_450) = 450) →
  (∀ (n : ℕ), n * n ∣ 450 ↔ n ∈ [1, 3, 5, 15]) →
  num_perfect_square_factors = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_450_l940_94073


namespace NUMINAMATH_CALUDE_initial_number_of_kids_l940_94004

theorem initial_number_of_kids (kids_left : ℕ) (kids_gone_home : ℕ) : 
  kids_left = 8 ∧ kids_gone_home = 14 → kids_left + kids_gone_home = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_kids_l940_94004


namespace NUMINAMATH_CALUDE_safe_menu_fraction_l940_94090

theorem safe_menu_fraction (total_dishes : ℕ) (vegetarian_dishes : ℕ) (gluten_free_vegetarian : ℕ) :
  vegetarian_dishes = total_dishes / 3 →
  gluten_free_vegetarian = vegetarian_dishes - 5 →
  (gluten_free_vegetarian : ℚ) / total_dishes = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_safe_menu_fraction_l940_94090


namespace NUMINAMATH_CALUDE_x_plus_four_value_l940_94030

theorem x_plus_four_value (x t : ℝ) 
  (h1 : 6 * x + t = 4 * x - 9) 
  (h2 : t = 7) : 
  x + 4 = -4 := by
sorry

end NUMINAMATH_CALUDE_x_plus_four_value_l940_94030


namespace NUMINAMATH_CALUDE_laundry_wash_time_l940_94023

/-- The time it takes to wash clothes in minutes -/
def clothes_time : ℕ := 30

/-- The time it takes to wash towels in minutes -/
def towels_time : ℕ := 2 * clothes_time

/-- The time it takes to wash sheets in minutes -/
def sheets_time : ℕ := towels_time - 15

/-- The total time it takes to wash all laundry in minutes -/
def total_wash_time : ℕ := clothes_time + towels_time + sheets_time

theorem laundry_wash_time : total_wash_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_laundry_wash_time_l940_94023


namespace NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l940_94046

theorem simplest_fraction_of_decimal (a b : ℕ+) (h : (a : ℚ) / b = 0.478125) :
  (∀ d : ℕ+, d ∣ a → d ∣ b → d = 1) →
  (a : ℕ) = 153 ∧ b = 320 ∧ a + b = 473 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l940_94046


namespace NUMINAMATH_CALUDE_fraction_subtraction_l940_94086

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l940_94086


namespace NUMINAMATH_CALUDE_infinite_pairs_with_difference_one_l940_94006

-- Define the property of being tuanis
def is_tuanis (a b : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (a + b).digits 10 → d = 0 ∨ d = 1

-- Define the sets A and B
def tuanis_set (A B : Set ℕ) : Prop :=
  (∀ a ∈ A, ∃ b ∈ B, is_tuanis a b) ∧
  (∀ b ∈ B, ∃ a ∈ A, is_tuanis a b)

-- The main theorem
theorem infinite_pairs_with_difference_one
  (A B : Set ℕ) (hA : Set.Infinite A) (hB : Set.Infinite B)
  (h_tuanis : tuanis_set A B) :
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 = 1}) ∨
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ B ∧ p.2 ∈ B ∧ p.1 - p.2 = 1}) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_difference_one_l940_94006


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_40_factorial_l940_94050

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_multiples (n k : ℕ) : ℕ := n / k

theorem greatest_power_of_three_in_40_factorial :
  (∀ m : ℕ, m ≤ 18 → is_factor (3^m) (factorial 40)) ∧
  ¬(is_factor (3^19) (factorial 40)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_40_factorial_l940_94050


namespace NUMINAMATH_CALUDE_total_fruits_in_bowl_l940_94062

/-- The total number of fruits in a bowl, given the number of bananas, 
    apples (twice the number of bananas), and oranges. -/
theorem total_fruits_in_bowl (bananas : ℕ) (oranges : ℕ) : 
  bananas = 2 → oranges = 6 → bananas + 2 * bananas + oranges = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_bowl_l940_94062


namespace NUMINAMATH_CALUDE_optimal_rental_plan_minimum_transportation_cost_l940_94080

/-- Represents the rental plan for trucks -/
structure RentalPlan where
  truckA : ℕ
  truckB : ℕ

/-- Checks if a rental plan is valid according to the problem constraints -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.truckA + plan.truckB = 6 ∧
  45 * plan.truckA + 30 * plan.truckB ≥ 240 ∧
  400 * plan.truckA + 300 * plan.truckB ≤ 2300

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.truckA + 300 * plan.truckB

/-- Theorem stating that the optimal plan is 4 Truck A and 2 Truck B -/
theorem optimal_rental_plan :
  ∀ (plan : RentalPlan),
    isValidPlan plan →
    totalCost plan ≥ totalCost { truckA := 4, truckB := 2 } :=
by sorry

/-- Corollary stating the minimum transportation cost -/
theorem minimum_transportation_cost :
  totalCost { truckA := 4, truckB := 2 } = 2200 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_minimum_transportation_cost_l940_94080


namespace NUMINAMATH_CALUDE_number_puzzle_l940_94076

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 6 = x / 4 - 6) ∧ x = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l940_94076


namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_min_value_f_l940_94027

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 3| + |x - 5|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} :=
by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = 7/2 ∧ ∀ (y : ℝ), f y ≥ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_min_value_f_l940_94027


namespace NUMINAMATH_CALUDE_max_cubic_sum_under_constraint_l940_94065

theorem max_cubic_sum_under_constraint (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a + b + c + d = 8) :
  a^3 + b^3 + c^3 + d^3 ≤ 15.625 := by
  sorry

end NUMINAMATH_CALUDE_max_cubic_sum_under_constraint_l940_94065


namespace NUMINAMATH_CALUDE_traffic_light_probability_l940_94013

/-- Represents the duration of traffic light phases in seconds -/
structure TrafficLightCycle where
  greenDuration : ℕ
  redDuration : ℕ

/-- Calculates the probability of waiting at least a given time in a traffic light cycle -/
def waitingProbability (cycle : TrafficLightCycle) (minWaitTime : ℕ) : ℚ :=
  let totalDuration := cycle.greenDuration + cycle.redDuration
  let waitInterval := cycle.redDuration - minWaitTime
  waitInterval / totalDuration

theorem traffic_light_probability (cycle : TrafficLightCycle) 
    (h1 : cycle.greenDuration = 40)
    (h2 : cycle.redDuration = 50) :
    waitingProbability cycle 20 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l940_94013


namespace NUMINAMATH_CALUDE_solve_seashells_problem_l940_94028

def seashells_problem (monday_shells : ℕ) (total_money : ℚ) : Prop :=
  let tuesday_shells : ℕ := monday_shells / 2
  let total_shells : ℕ := monday_shells + tuesday_shells
  let money_per_shell : ℚ := total_money / total_shells
  monday_shells = 30 ∧ total_money = 54 → money_per_shell = 1.20

theorem solve_seashells_problem :
  seashells_problem 30 54 := by sorry

end NUMINAMATH_CALUDE_solve_seashells_problem_l940_94028


namespace NUMINAMATH_CALUDE_truncated_cone_rope_theorem_l940_94099

/-- Represents a truncated cone with given dimensions -/
structure TruncatedCone where
  r₁ : ℝ  -- Upper base radius
  r₂ : ℝ  -- Lower base radius
  h : ℝ   -- Slant height

/-- Calculates the minimum length of the rope for a given truncated cone -/
def min_rope_length (cone : TruncatedCone) : ℝ := sorry

/-- Calculates the minimum distance from the rope to the upper base circumference -/
def min_distance_to_upper_base (cone : TruncatedCone) : ℝ := sorry

theorem truncated_cone_rope_theorem (cone : TruncatedCone) 
  (h₁ : cone.r₁ = 5)
  (h₂ : cone.r₂ = 10)
  (h₃ : cone.h = 20) :
  (min_rope_length cone = 50) ∧ 
  (min_distance_to_upper_base cone = 4) := by sorry

end NUMINAMATH_CALUDE_truncated_cone_rope_theorem_l940_94099


namespace NUMINAMATH_CALUDE_total_spent_equals_1150_l940_94043

-- Define the quantities of toys
def elder_action_figures : ℕ := 60
def younger_action_figures : ℕ := 3 * elder_action_figures
def cars : ℕ := 20
def stuffed_animals : ℕ := 10

-- Define the prices of toys
def elder_action_figure_price : ℕ := 5
def younger_action_figure_price : ℕ := 4
def car_price : ℕ := 3
def stuffed_animal_price : ℕ := 7

-- Define the total cost function
def total_cost : ℕ :=
  elder_action_figures * elder_action_figure_price +
  younger_action_figures * younger_action_figure_price +
  cars * car_price +
  stuffed_animals * stuffed_animal_price

-- Theorem statement
theorem total_spent_equals_1150 : total_cost = 1150 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_1150_l940_94043


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l940_94067

theorem complex_magnitude_problem :
  let z : ℂ := ((1 - 4*I) * (1 + I) + 2 + 4*I) / (3 + 4*I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l940_94067


namespace NUMINAMATH_CALUDE_if_A_then_all_short_answer_correct_l940_94082

/-- Represents the condition for receiving an A grade -/
def receivedA (allShortAnswerCorrect : Bool) (multipleChoicePercentage : ℝ) : Prop :=
  allShortAnswerCorrect ∧ multipleChoicePercentage ≥ 90

/-- Proves that if a student received an A, they must have answered all short-answer questions correctly -/
theorem if_A_then_all_short_answer_correct 
  (student : String) 
  (studentReceivedA : Bool) 
  (studentAllShortAnswerCorrect : Bool) 
  (studentMultipleChoicePercentage : ℝ) : 
  (receivedA studentAllShortAnswerCorrect studentMultipleChoicePercentage → studentReceivedA) →
  (studentReceivedA → studentAllShortAnswerCorrect) :=
by sorry

end NUMINAMATH_CALUDE_if_A_then_all_short_answer_correct_l940_94082


namespace NUMINAMATH_CALUDE_shiny_igneous_fraction_l940_94015

/-- Represents Cliff's rock collection -/
structure RockCollection where
  total : ℕ
  sedimentary : ℕ
  igneous : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- Properties of Cliff's rock collection -/
def isValidCollection (c : RockCollection) : Prop :=
  c.igneous = c.sedimentary / 2 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 30 ∧
  c.total = 270 ∧
  c.total = c.sedimentary + c.igneous

theorem shiny_igneous_fraction (c : RockCollection) 
  (h : isValidCollection c) : 
  (c.shinyIgneous : ℚ) / c.igneous = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shiny_igneous_fraction_l940_94015


namespace NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l940_94066

theorem defeated_candidate_vote_percentage 
  (total_polled_votes : ℕ) 
  (invalid_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_polled_votes = 90083) 
  (h2 : invalid_votes = 83) 
  (h3 : vote_difference = 9000) : 
  let valid_votes := total_polled_votes - invalid_votes
  let defeated_votes := (valid_votes - vote_difference) / 2
  defeated_votes * 100 / valid_votes = 45 := by
sorry

end NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l940_94066


namespace NUMINAMATH_CALUDE_derivative_lg_l940_94087

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem derivative_lg (x : ℝ) (h : x > 0) :
  deriv lg x = 1 / (x * Real.log 10) :=
sorry

end NUMINAMATH_CALUDE_derivative_lg_l940_94087


namespace NUMINAMATH_CALUDE_system_solution_1_l940_94034

theorem system_solution_1 (x y : ℚ) : 
  (3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33) ↔ (x = 6 ∧ y = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_1_l940_94034


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l940_94036

/-- The line equation ax + by = c forming a triangle with coordinate axes -/
structure TriangleLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the sum of altitudes of the triangle formed by the given line and coordinate axes -/
def sumOfAltitudes (line : TriangleLine) : ℝ :=
  sorry

/-- The specific line 8x + 3y = 48 -/
def specificLine : TriangleLine :=
  { a := 8, b := 3, c := 48 }

theorem sum_of_altitudes_for_specific_line :
  sumOfAltitudes specificLine = 370 / Real.sqrt 292 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l940_94036


namespace NUMINAMATH_CALUDE_number_equation_l940_94033

theorem number_equation (x : ℝ) : 2500 - (x / 20.04) = 2450 ↔ x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l940_94033


namespace NUMINAMATH_CALUDE_number_equation_l940_94044

theorem number_equation (x : ℝ) : 3550 - (x / 20.04) = 3500 ↔ x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l940_94044


namespace NUMINAMATH_CALUDE_max_operations_l940_94007

def operation_count (a b : ℕ) : ℕ := sorry

theorem max_operations (a b : ℕ) (ha : a = 2000) (hb : b < 2000) :
  operation_count a b ≤ 10 := by sorry

end NUMINAMATH_CALUDE_max_operations_l940_94007


namespace NUMINAMATH_CALUDE_track_length_is_900_l940_94078

/-- The length of a circular track where two runners meet again -/
def track_length (v1 v2 t : ℝ) : ℝ :=
  (v1 - v2) * t

/-- Theorem stating the length of the track is 900 meters -/
theorem track_length_is_900 :
  let v1 : ℝ := 30  -- Speed of Bruce in m/s
  let v2 : ℝ := 20  -- Speed of Bhishma in m/s
  let t : ℝ := 90   -- Time in seconds
  track_length v1 v2 t = 900 := by
  sorry

#eval track_length 30 20 90  -- Should output 900

end NUMINAMATH_CALUDE_track_length_is_900_l940_94078


namespace NUMINAMATH_CALUDE_max_ratio_squared_max_ratio_squared_achieved_l940_94074

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 ≤ 2 :=
by sorry

theorem max_ratio_squared_achieved (a b : ℝ) :
  ∃ x y : ℝ, 0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_max_ratio_squared_achieved_l940_94074


namespace NUMINAMATH_CALUDE_total_cost_theorem_l940_94052

/-- The cost of items and their relationships -/
structure ItemCosts where
  pencil_cost : ℝ
  pen_cost : ℝ
  notebook_cost : ℝ
  pen_pencil_diff : ℝ
  notebook_pen_ratio : ℝ
  notebook_discount : ℝ
  cad_usd_rate : ℝ

/-- Calculate the total cost in USD -/
def total_cost_usd (costs : ItemCosts) : ℝ :=
  let pen_cost := costs.pencil_cost + costs.pen_pencil_diff
  let notebook_cost := costs.notebook_pen_ratio * pen_cost
  let discounted_notebook_cost := notebook_cost * (1 - costs.notebook_discount)
  let total_cad := costs.pencil_cost + pen_cost + discounted_notebook_cost
  total_cad * costs.cad_usd_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_theorem (costs : ItemCosts) 
  (h1 : costs.pencil_cost = 2)
  (h2 : costs.pen_pencil_diff = 9)
  (h3 : costs.notebook_pen_ratio = 2)
  (h4 : costs.notebook_discount = 0.15)
  (h5 : costs.cad_usd_rate = 1.25) :
  total_cost_usd costs = 39.63 := by
  sorry

#eval total_cost_usd {
  pencil_cost := 2,
  pen_cost := 11,
  notebook_cost := 22,
  pen_pencil_diff := 9,
  notebook_pen_ratio := 2,
  notebook_discount := 0.15,
  cad_usd_rate := 1.25
}

end NUMINAMATH_CALUDE_total_cost_theorem_l940_94052


namespace NUMINAMATH_CALUDE_max_sum_with_divisibility_conditions_l940_94097

theorem max_sum_with_divisibility_conditions (a b c : ℕ) : 
  a > 2022 → b > 2022 → c > 2022 →
  (c - 2022) ∣ (a + b) →
  (b - 2022) ∣ (a + c) →
  (a - 2022) ∣ (b + c) →
  a + b + c ≤ 2022 * 85 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_divisibility_conditions_l940_94097


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l940_94088

/-- The length of the real axis of a hyperbola with equation x²/9 - y² = 1 is 6. -/
theorem hyperbola_real_axis_length :
  ∃ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2/9 - y^2 = 1) →
    (∃ a : ℝ, a > 0 ∧ ∀ x y, f x y ↔ x^2/a^2 - y^2 = 1) →
    2 * Real.sqrt 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l940_94088


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l940_94077

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) :
  let S : ℕ → ℝ
    | 1 => a₁
    | 2 => a₁ + a₁ * q
    | 3 => a₁ + a₁ * q + a₁ * q^2
    | _ => 0  -- We only need S₁, S₂, and S₃ for this problem
  (S 3 - S 2 = S 2 - S 1) → q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l940_94077


namespace NUMINAMATH_CALUDE_base_conversion_l940_94072

theorem base_conversion (b : ℝ) : b > 0 → (3 * 5 + 2 = b^2 + 2) → b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l940_94072


namespace NUMINAMATH_CALUDE_intersection_of_inequalities_l940_94026

theorem intersection_of_inequalities (m n : ℝ) (h : -1 < m ∧ m < 0 ∧ 0 < n) :
  {x : ℝ | m < x ∧ x < n} ∩ {x : ℝ | -1 < x ∧ x < 0} = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_inequalities_l940_94026


namespace NUMINAMATH_CALUDE_symbol_equation_solution_l940_94010

theorem symbol_equation_solution (triangle circle : ℕ) 
  (h1 : triangle + circle + circle = 55)
  (h2 : triangle + circle = 40) :
  circle = 15 ∧ triangle = 25 := by
  sorry

end NUMINAMATH_CALUDE_symbol_equation_solution_l940_94010


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l940_94051

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l940_94051


namespace NUMINAMATH_CALUDE_sum_of_cube_roots_bounded_l940_94032

theorem sum_of_cube_roots_bounded (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  5 < (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) ∧
      (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) < 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_roots_bounded_l940_94032


namespace NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l940_94045

/-- Represents the types of reasoning --/
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

/-- Represents a step in the logical progression --/
structure LogicalStep where
  premise : String
  consequence : String

/-- Represents the characteristics of the reasoning in the Analects passage --/
structure AnalectsReasoning where
  steps : List LogicalStep
  alignsWithCommonSense : Bool
  followsLogicalProgression : Bool

/-- Determines the type of reasoning based on its characteristics --/
def determineReasoningType (reasoning : AnalectsReasoning) : ReasoningType :=
  if reasoning.alignsWithCommonSense && reasoning.followsLogicalProgression then
    ReasoningType.CommonSense
  else
    ReasoningType.Inductive -- Default to another type if conditions are not met

/-- The main theorem stating that the reasoning in the Analects passage is Common Sense reasoning --/
theorem analects_reasoning_is_common_sense (analectsReasoning : AnalectsReasoning) 
    (h1 : analectsReasoning.steps.length > 0)
    (h2 : analectsReasoning.alignsWithCommonSense = true)
    (h3 : analectsReasoning.followsLogicalProgression = true) :
  determineReasoningType analectsReasoning = ReasoningType.CommonSense := by
  sorry

#check analects_reasoning_is_common_sense

end NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l940_94045


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l940_94017

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, a^2 * x^2 + a * x - 1 = 0 ∧ x^2 - a * x - a^2 = 0) →
  (a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
   a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l940_94017


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l940_94009

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x a : ℝ) : Prop := |x - 3| < a ∧ a > 0

-- Define the solution set of p
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | 3 - a < x ∧ x < 3 + a}

-- Theorem statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a > 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l940_94009


namespace NUMINAMATH_CALUDE_two_number_problem_l940_94070

theorem two_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 8.58 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l940_94070


namespace NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_roots_l940_94079

theorem sum_of_squares_of_quadratic_roots : ∀ (s₁ s₂ : ℝ), 
  s₁^2 - 20*s₁ + 32 = 0 → 
  s₂^2 - 20*s₂ + 32 = 0 → 
  s₁^2 + s₂^2 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_roots_l940_94079


namespace NUMINAMATH_CALUDE_root_product_theorem_l940_94037

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a + 1)^2 - p*(b + 1/a + 1) + r = 0) →
  r = 19/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l940_94037


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l940_94048

theorem other_solution_quadratic (h : 40 * (4/5)^2 - 69 * (4/5) + 24 = 0) :
  40 * (3/8)^2 - 69 * (3/8) + 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l940_94048


namespace NUMINAMATH_CALUDE_cos_135_degrees_l940_94018

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l940_94018


namespace NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l940_94095

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_equality_implies_norm_equality 
  (a b : E) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = -2 • b) : 
  ‖a‖ - ‖b‖ = ‖a + b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l940_94095


namespace NUMINAMATH_CALUDE_pool_filling_time_l940_94063

/-- The number of hours it takes for a swimming pool to reach full capacity -/
def full_capacity_hours : ℕ := 8

/-- The factor by which the water volume increases each hour -/
def volume_increase_factor : ℕ := 3

/-- The fraction of the pool's capacity we're interested in -/
def target_fraction : ℚ := 1 / 9

/-- The number of hours it takes to reach the target fraction of capacity -/
def target_hours : ℕ := 6

theorem pool_filling_time :
  (volume_increase_factor ^ (full_capacity_hours - target_hours) : ℚ) = 1 / target_fraction :=
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l940_94063


namespace NUMINAMATH_CALUDE_kyle_weekly_papers_l940_94019

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_only : ℕ) : ℕ :=
  6 * weekday_houses + (weekday_houses - sunday_skip + sunday_only)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_weekly_papers :
  weekly_papers 100 10 30 = 720 := by
  sorry

#eval weekly_papers 100 10 30

end NUMINAMATH_CALUDE_kyle_weekly_papers_l940_94019


namespace NUMINAMATH_CALUDE_last_two_digits_product_l940_94061

/-- Given an integer n that is divisible by 6 and whose last two digits sum to 15,
    the product of its last two digits is 54. -/
theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →  -- Ensure we're dealing with the last two positive digits
  (n % 6 = 0) →    -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  ((n % 100) / 10) * (n % 10) = 54 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l940_94061


namespace NUMINAMATH_CALUDE_last_week_sales_l940_94002

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  week1 = 75 ∧ week2 = 67 ∧ week3 = 75 ∧ week4 = 70

theorem last_week_sales (week5 : ℕ) :
  chocolate_sales 75 67 75 70 week5 →
  (75 + 67 + 75 + 70 + week5) / 5 = 71 →
  week5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_last_week_sales_l940_94002


namespace NUMINAMATH_CALUDE_chair_table_price_percentage_l940_94021

/-- The price of a chair in dollars -/
def chair_price : ℚ := (96 - 84)

/-- The price of a table in dollars -/
def table_price : ℚ := 84

/-- The price of 2 chairs and 1 table -/
def price_2c1t : ℚ := 2 * chair_price + table_price

/-- The price of 1 chair and 2 tables -/
def price_1c2t : ℚ := chair_price + 2 * table_price

/-- The percentage of price_2c1t to price_1c2t -/
def percentage : ℚ := price_2c1t / price_1c2t * 100

theorem chair_table_price_percentage :
  percentage = 60 := by sorry

end NUMINAMATH_CALUDE_chair_table_price_percentage_l940_94021


namespace NUMINAMATH_CALUDE_surface_area_difference_l940_94022

def rectangular_solid_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

def cube_surface_area (s : ℝ) : ℝ :=
  6 * s^2

def new_exposed_area (s : ℝ) : ℝ :=
  3 * s^2

theorem surface_area_difference :
  let original_area := rectangular_solid_surface_area 4 5 6
  let removed_area := cube_surface_area 2
  let exposed_area := new_exposed_area 2
  original_area - removed_area + exposed_area = original_area - 12
  := by sorry

end NUMINAMATH_CALUDE_surface_area_difference_l940_94022


namespace NUMINAMATH_CALUDE_range_of_a_l940_94092

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x y : ℝ, x - y + a = 0 ∧ x^2 + y^2 - 2*x = 1

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp x - a > 1

-- State the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l940_94092


namespace NUMINAMATH_CALUDE_equation_solutions_l940_94041

theorem equation_solutions : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = (2 + Real.sqrt 1121) / 14 ∧ x₃ = (2 - Real.sqrt 1121) / 14) ∧
  (∀ x : ℝ, (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l940_94041


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l940_94020

-- Define set M
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l940_94020


namespace NUMINAMATH_CALUDE_flip_invariant_numbers_l940_94000

/-- A digit that remains unchanged when flipped upside down -/
inductive FlipInvariantDigit : Nat → Prop
  | zero : FlipInvariantDigit 0
  | eight : FlipInvariantDigit 8

/-- A three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- A three-digit number that remains unchanged when flipped upside down -/
def FlipInvariantNumber (n : ThreeDigitNumber) : Prop :=
  FlipInvariantDigit n.hundreds ∧ FlipInvariantDigit n.tens ∧ FlipInvariantDigit n.ones

theorem flip_invariant_numbers :
  ∀ n : ThreeDigitNumber, FlipInvariantNumber n →
    (n.hundreds = 8 ∧ n.tens = 0 ∧ n.ones = 8) ∨ (n.hundreds = 8 ∧ n.tens = 8 ∧ n.ones = 8) :=
by sorry

end NUMINAMATH_CALUDE_flip_invariant_numbers_l940_94000


namespace NUMINAMATH_CALUDE_max_cross_sectional_area_l940_94094

-- Define the prism
def prism_base_side_length : ℝ := 8

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 3 * x - 5 * y + 2 * z = 20

-- Define the cross-sectional area function
noncomputable def cross_sectional_area (h : ℝ) : ℝ := 
  let diagonal := (2 * prism_base_side_length ^ 2 + h ^ 2) ^ (1/2 : ℝ)
  let area := h * diagonal / 2
  area

-- Theorem statement
theorem max_cross_sectional_area :
  ∃ h : ℝ, h > 0 ∧ 
    cross_sectional_area h = 9 * (38 : ℝ).sqrt ∧
    ∀ h' : ℝ, h' > 0 → cross_sectional_area h' ≤ cross_sectional_area h :=
by sorry

end NUMINAMATH_CALUDE_max_cross_sectional_area_l940_94094


namespace NUMINAMATH_CALUDE_grants_apartment_rooms_l940_94049

-- Define the number of rooms in Danielle's apartment
def danielles_rooms : ℕ := 6

-- Define the number of rooms in Heidi's apartment
def heidis_rooms : ℕ := 3 * danielles_rooms

-- Define the number of rooms in Grant's apartment
def grants_rooms : ℕ := heidis_rooms / 9

-- Theorem stating that Grant's apartment has 2 rooms
theorem grants_apartment_rooms : grants_rooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_grants_apartment_rooms_l940_94049


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l940_94093

/-- Represents a workshop with its production quantity -/
structure Workshop where
  production : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleSizes : List ℕ

def StratifiedSampling.totalSampleSize (s : StratifiedSampling) : ℕ :=
  s.sampleSizes.sum

def StratifiedSampling.isValid (s : StratifiedSampling) : Prop :=
  s.workshops.length = s.sampleSizes.length ∧ 
  s.sampleSizes.all (· > 0)

theorem stratified_sampling_theorem (s : StratifiedSampling) 
  (h1 : s.workshops = [⟨120⟩, ⟨90⟩, ⟨60⟩])
  (h2 : s.sampleSizes.length = 3)
  (h3 : s.sampleSizes[2] = 2)
  (h4 : s.isValid)
  (h5 : ∀ s' : StratifiedSampling, s'.workshops = s.workshops → 
        s'.isValid → s'.totalSampleSize ≥ s.totalSampleSize) :
  s.totalSampleSize = 9 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l940_94093


namespace NUMINAMATH_CALUDE_smallest_number_last_three_digits_l940_94084

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def consists_of_2_and_7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_at_least_one_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem smallest_number_last_three_digits :
  ∃ m : ℕ, 
    (∀ k : ℕ, k < m → 
      ¬(is_divisible_by k 6 ∧ 
        is_divisible_by k 8 ∧ 
        consists_of_2_and_7 k ∧ 
        has_at_least_one_2_and_7 k)) ∧
    is_divisible_by m 6 ∧
    is_divisible_by m 8 ∧
    consists_of_2_and_7 m ∧
    has_at_least_one_2_and_7 m ∧
    last_three_digits m = 722 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_last_three_digits_l940_94084


namespace NUMINAMATH_CALUDE_odd_numbers_sum_product_equality_l940_94059

/-- For a positive integer n, there exist n positive odd numbers whose sum equals 
    their product if and only if n is of the form 4k + 1, where k is a non-negative integer. -/
theorem odd_numbers_sum_product_equality (n : ℕ+) : 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, Odd x ∧ x > 0) ∧ 
    (S.sum id = S.prod id)) ↔ 
  ∃ k : ℕ, n = 4 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_sum_product_equality_l940_94059


namespace NUMINAMATH_CALUDE_cashew_nut_purchase_l940_94053

/-- Prove that given the conditions of the nut purchase problem, the number of kilos of cashew nuts bought is 3. -/
theorem cashew_nut_purchase (cashew_price peanut_price peanut_amount total_weight avg_price : ℝ) 
  (h1 : cashew_price = 210)
  (h2 : peanut_price = 130)
  (h3 : peanut_amount = 2)
  (h4 : total_weight = 5)
  (h5 : avg_price = 178) :
  (total_weight - peanut_amount) = 3 := by
  sorry


end NUMINAMATH_CALUDE_cashew_nut_purchase_l940_94053


namespace NUMINAMATH_CALUDE_prob_three_sixes_is_one_over_216_l940_94029

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll (n : ℕ) : ℚ := 1 / standard_die_faces

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling the target sum with the given number of dice -/
def prob_target_sum : ℚ := (prob_single_roll target_sum) ^ num_dice

theorem prob_three_sixes_is_one_over_216 : prob_target_sum = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_sixes_is_one_over_216_l940_94029


namespace NUMINAMATH_CALUDE_task_force_combinations_l940_94016

theorem task_force_combinations (independents greens : ℕ) 
  (h1 : independents = 10) (h2 : greens = 7) : 
  (Nat.choose independents 4) * (Nat.choose greens 3) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_task_force_combinations_l940_94016


namespace NUMINAMATH_CALUDE_parabolas_imply_right_triangle_l940_94064

/-- Two parabolas intersecting the x-axis at the same non-origin point -/
def intersecting_parabolas (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0

/-- The triangle formed by sides a, b, and c is right-angled -/
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

theorem parabolas_imply_right_triangle (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_neq : a ≠ c) 
  (h_intersect : intersecting_parabolas a b c) : 
  right_angled_triangle a b c := by
  sorry

end NUMINAMATH_CALUDE_parabolas_imply_right_triangle_l940_94064


namespace NUMINAMATH_CALUDE_simple_random_sampling_prob_std_dev_transformation_l940_94071

/-- Simple random sampling probability -/
theorem simple_random_sampling_prob (population_size : ℕ) (sample_size : ℕ) :
  population_size = 50 → sample_size = 10 →
  (sample_size : ℝ) / (population_size : ℝ) = 0.2 := by sorry

/-- Standard deviation transformation -/
theorem std_dev_transformation (x : Fin 10 → ℝ) (σ : ℝ) :
  Real.sqrt (Finset.univ.sum (λ i => (x i - Finset.univ.sum x / 10) ^ 2) / 10) = σ →
  Real.sqrt (Finset.univ.sum (λ i => ((2 * x i - 1) - Finset.univ.sum (λ j => 2 * x j - 1) / 10) ^ 2) / 10) = 2 * σ := by sorry

end NUMINAMATH_CALUDE_simple_random_sampling_prob_std_dev_transformation_l940_94071


namespace NUMINAMATH_CALUDE_faster_train_speed_l940_94081

-- Define the lengths of the trains in meters
def train1_length : ℝ := 200
def train2_length : ℝ := 160

-- Define the time taken to cross in seconds
def crossing_time : ℝ := 11.999040076793857

-- Define the speed of the slower train in km/h
def slower_train_speed : ℝ := 40

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem statement
theorem faster_train_speed : 
  ∃ (faster_speed : ℝ),
    faster_speed = 68 ∧ 
    (train1_length + train2_length) / crossing_time * ms_to_kmh = faster_speed + slower_train_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l940_94081


namespace NUMINAMATH_CALUDE_complement_union_theorem_l940_94038

universe u

def U : Set ℕ := {0, 1, 3, 4, 5, 6, 8}
def A : Set ℕ := {1, 4, 5, 8}
def B : Set ℕ := {2, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l940_94038


namespace NUMINAMATH_CALUDE_distinct_collections_is_125_l940_94058

/-- Represents the word "COMPUTATIONS" -/
def word : String := "COMPUTATIONS"

/-- The number of vowels in the word -/
def num_vowels : Nat := 5

/-- The number of consonants in the word, excluding T's -/
def num_consonants_without_t : Nat := 5

/-- The number of T's in the word -/
def num_t : Nat := 2

/-- The number of vowels to select -/
def vowels_to_select : Nat := 4

/-- The number of consonants to select -/
def consonants_to_select : Nat := 4

/-- Calculates the number of distinct collections of letters -/
def distinct_collections : Nat :=
  (Nat.choose num_vowels vowels_to_select) * 
  ((Nat.choose num_consonants_without_t consonants_to_select) + 
   (Nat.choose num_consonants_without_t (consonants_to_select - 1)) +
   (Nat.choose num_consonants_without_t (consonants_to_select - 2)))

/-- Theorem stating that the number of distinct collections is 125 -/
theorem distinct_collections_is_125 : distinct_collections = 125 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_is_125_l940_94058


namespace NUMINAMATH_CALUDE_greatest_common_factor_72_180_270_l940_94011

theorem greatest_common_factor_72_180_270 : Nat.gcd 72 (Nat.gcd 180 270) = 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_72_180_270_l940_94011
