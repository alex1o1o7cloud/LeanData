import Mathlib

namespace NUMINAMATH_CALUDE_square_equation_solution_l3244_324483

theorem square_equation_solution : ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3244_324483


namespace NUMINAMATH_CALUDE_weekly_earnings_l3244_324433

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def tablet_repair_cost : ℕ := 12
def smartwatch_repair_cost : ℕ := 8

def phone_repairs : ℕ := 9
def laptop_repairs : ℕ := 5
def computer_repairs : ℕ := 4
def tablet_repairs : ℕ := 6
def smartwatch_repairs : ℕ := 8

def total_earnings : ℕ := 
  phone_repair_cost * phone_repairs +
  laptop_repair_cost * laptop_repairs +
  computer_repair_cost * computer_repairs +
  tablet_repair_cost * tablet_repairs +
  smartwatch_repair_cost * smartwatch_repairs

theorem weekly_earnings : total_earnings = 382 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_l3244_324433


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3244_324494

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3244_324494


namespace NUMINAMATH_CALUDE_power_inequality_l3244_324420

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3244_324420


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l3244_324434

theorem smallest_value_theorem (a b : ℕ+) (h : a.val^2 - b.val^2 = 16) :
  (∀ (c d : ℕ+), c.val^2 - d.val^2 = 16 →
    (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) ≤
    (c.val + d.val : ℚ) / (c.val - d.val : ℚ) + (c.val - d.val : ℚ) / (c.val + d.val : ℚ)) ∧
  (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l3244_324434


namespace NUMINAMATH_CALUDE_largest_factorial_as_product_of_four_consecutive_l3244_324404

/-- Predicate that checks if a number is expressible as the product of 4 consecutive integers -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) * (x + 2) * (x + 3)

/-- Theorem stating that 6 is the largest integer n such that n! can be expressed as the product of 4 consecutive integers -/
theorem largest_factorial_as_product_of_four_consecutive :
  (6 : ℕ).factorial = 6 * 7 * 8 * 9 ∧
  ∀ n : ℕ, n > 6 → ¬(is_product_of_four_consecutive n.factorial) :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_as_product_of_four_consecutive_l3244_324404


namespace NUMINAMATH_CALUDE_calculation_proof_l3244_324493

theorem calculation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) :
  (c * (a^3 + b^3)) / (a^2 - a*b + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3244_324493


namespace NUMINAMATH_CALUDE_case_one_solutions_case_two_no_solution_l3244_324462

-- Case 1
theorem case_one_solutions (a b : ℝ) (A : ℝ) (ha : a = 14) (hb : b = 16) (hA : A = 45 * π / 180) :
  ∃! (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

-- Case 2
theorem case_two_no_solution (a b : ℝ) (B : ℝ) (ha : a = 60) (hb : b = 48) (hB : B = 60 * π / 180) :
  ¬ ∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

end NUMINAMATH_CALUDE_case_one_solutions_case_two_no_solution_l3244_324462


namespace NUMINAMATH_CALUDE_total_legs_in_javiers_household_l3244_324447

/-- The number of legs in Javier's household -/
def total_legs : ℕ :=
  let num_humans := 5 -- Javier, his wife, and 3 children
  let num_dogs := 2
  let num_cats := 1
  let legs_per_human := 2
  let legs_per_dog := 4
  let legs_per_cat := 4
  num_humans * legs_per_human + num_dogs * legs_per_dog + num_cats * legs_per_cat

theorem total_legs_in_javiers_household :
  total_legs = 22 := by sorry

end NUMINAMATH_CALUDE_total_legs_in_javiers_household_l3244_324447


namespace NUMINAMATH_CALUDE_locus_of_Q_l3244_324456

-- Define the polar coordinate system
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define the circle C
def circle_C (p : PolarCoord) : Prop :=
  p.ρ = 2

-- Define the line l
def line_l (p : PolarCoord) : Prop :=
  p.ρ * (Real.cos p.θ + Real.sin p.θ) = 2

-- Define the relationship between points O, P, Q, and R
def point_relationship (P Q R : PolarCoord) : Prop :=
  Q.ρ * P.ρ = R.ρ^2

-- Theorem statement
theorem locus_of_Q (P Q R : PolarCoord) :
  circle_C R →
  line_l P →
  point_relationship P Q R →
  Q.ρ = 2 * (Real.cos Q.θ + Real.sin Q.θ) ∧ Q.ρ ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_Q_l3244_324456


namespace NUMINAMATH_CALUDE_fourth_part_diminished_l3244_324474

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 160) (h2 : (x / 5) + 4 = (x / 4) - y) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_part_diminished_l3244_324474


namespace NUMINAMATH_CALUDE_bacterial_growth_l3244_324487

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total duration of the culturing process in minutes -/
def total_time : ℕ := 3 * 60

/-- The number of divisions that occur during the culturing process -/
def num_divisions : ℕ := total_time / division_interval

/-- The final number of bacteria after the culturing process -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacterial_growth :
  final_bacteria_count = 512 :=
sorry

end NUMINAMATH_CALUDE_bacterial_growth_l3244_324487


namespace NUMINAMATH_CALUDE_rectangle_area_l3244_324401

/-- Proves that a rectangle with perimeter 126 and difference between sides 37 has an area of 650 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * (l + w) = 126) → 
  (l - w = 37) → 
  (l * w = 650) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3244_324401


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3244_324452

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added,
    perimeter := initial.perimeter + added }

/-- Theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration)
  (h1 : initial.num_tiles = 8)
  (h2 : initial.perimeter = 16)
  (added : ℕ)
  (h3 : added = 3) :
  ∃ (final : TileConfiguration),
    final = add_tiles initial added ∧ 
    final.perimeter = 19 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3244_324452


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3244_324407

/-- Calculates the marks in Biology given the marks in other subjects and the average -/
def marks_in_biology (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

/-- Theorem stating that David's marks in Biology are 85 -/
theorem davids_biology_marks :
  marks_in_biology 81 65 82 67 76 = 85 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3244_324407


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3244_324448

theorem sum_reciprocals_bound (a b c : ℕ) (h : 1 / a + 1 / b + 1 / c < 1) :
  1 / a + 1 / b + 1 / c ≤ 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3244_324448


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3244_324499

/-- Properties of the hyperbola x^2 - y^2 = 2 -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 2
  ∃ (a b c : ℝ),
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (2 * a = 2 * Real.sqrt 2) ∧
    (c^2 = a^2 + b^2) ∧
    (c / a = Real.sqrt 2) ∧
    (∀ x y, (y = x ∨ y = -x) → h x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3244_324499


namespace NUMINAMATH_CALUDE_other_number_proof_l3244_324424

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 2520 →
  Nat.gcd a b = 12 →
  a = 240 →
  b = 126 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l3244_324424


namespace NUMINAMATH_CALUDE_g_of_3_equals_6_l3244_324472

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

theorem g_of_3_equals_6 : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_6_l3244_324472


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3244_324490

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 3 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 90 →
  hypotenuse^2 = short_leg^2 + long_leg^2 →
  hypotenuse = Real.sqrt 593 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3244_324490


namespace NUMINAMATH_CALUDE_quadratic_sum_of_squares_l3244_324484

theorem quadratic_sum_of_squares (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) →
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) →
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_squares_l3244_324484


namespace NUMINAMATH_CALUDE_problem_solution_l3244_324464

theorem problem_solution : (150 * (150 - 4)) / (150 * 150 * 2 - 4) = 21900 / 44996 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3244_324464


namespace NUMINAMATH_CALUDE_carters_additional_cakes_l3244_324429

/-- The number of additional cakes Carter bakes in a week when tripling his usual production. -/
theorem carters_additional_cakes 
  (cheesecakes muffins red_velvet : ℕ) 
  (h1 : cheesecakes = 6)
  (h2 : muffins = 5)
  (h3 : red_velvet = 8) :
  3 * (cheesecakes + muffins + red_velvet) - (cheesecakes + muffins + red_velvet) = 38 :=
by sorry


end NUMINAMATH_CALUDE_carters_additional_cakes_l3244_324429


namespace NUMINAMATH_CALUDE_prime_dates_february_2024_l3244_324444

/-- A natural number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The number of days in February during a leap year. -/
def februaryDaysInLeapYear : ℕ := 29

/-- The month number for February. -/
def februaryMonth : ℕ := 2

/-- A prime date occurs when both the month and day are prime numbers. -/
def isPrimeDate (month day : ℕ) : Prop := isPrime month ∧ isPrime day

/-- The number of prime dates in February of a leap year. -/
def primeDatesInFebruaryLeapYear : ℕ := 10

/-- Theorem stating that the number of prime dates in February 2024 is 10. -/
theorem prime_dates_february_2024 :
  isPrime februaryMonth →
  (∀ d : ℕ, d ≤ februaryDaysInLeapYear → isPrimeDate februaryMonth d ↔ isPrime d) →
  (∃ dates : Finset ℕ, dates.card = primeDatesInFebruaryLeapYear ∧
    ∀ d ∈ dates, d ≤ februaryDaysInLeapYear ∧ isPrime d) :=
by sorry

end NUMINAMATH_CALUDE_prime_dates_february_2024_l3244_324444


namespace NUMINAMATH_CALUDE_village_population_l3244_324469

theorem village_population (P : ℝ) : 
  P > 0 → 
  (P * 0.9 * 0.8 = 3240) → 
  P = 4500 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3244_324469


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l3244_324465

theorem abs_inequality_solution (x : ℝ) : |x + 3| > x + 3 ↔ x < -3 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l3244_324465


namespace NUMINAMATH_CALUDE_no_polynomial_transform_l3244_324441

theorem no_polynomial_transform : ¬∃ (P : ℝ → ℝ), 
  (∀ x : ℝ, ∃ (a b c d : ℝ), P x = a * x^3 + b * x^2 + c * x + d) ∧
  P (-3) = -3 ∧ P (-1) = -1 ∧ P 1 = -3 ∧ P 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_polynomial_transform_l3244_324441


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3244_324416

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {y | y > 2}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M) ∩ N = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3244_324416


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3244_324458

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 * a 4 * a 6 * a 8 = 120 →
  1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7 / 60 →
  S a 9 = 63 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3244_324458


namespace NUMINAMATH_CALUDE_find_other_number_l3244_324427

theorem find_other_number (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3244_324427


namespace NUMINAMATH_CALUDE_acute_angle_trig_equation_l3244_324488

theorem acute_angle_trig_equation (x : Real) (h1 : 0 < x) (h2 : x < π / 2) 
  (h3 : Real.sin x ^ 3 + Real.cos x ^ 3 = Real.sqrt 2 / 2) : x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_trig_equation_l3244_324488


namespace NUMINAMATH_CALUDE_sophia_saves_two_dimes_l3244_324482

/-- Represents the number of pennies in a dime -/
def pennies_per_dime : ℕ := 10

/-- Represents the number of days Sophia saves -/
def saving_days : ℕ := 20

/-- Represents the number of pennies Sophia saves per day -/
def pennies_per_day : ℕ := 1

/-- Calculates the total number of pennies saved -/
def total_pennies : ℕ := saving_days * pennies_per_day

/-- Theorem: Sophia saves 2 dimes in total -/
theorem sophia_saves_two_dimes : 
  total_pennies / pennies_per_dime = 2 := by sorry

end NUMINAMATH_CALUDE_sophia_saves_two_dimes_l3244_324482


namespace NUMINAMATH_CALUDE_divisors_of_sum_for_K_6_l3244_324495

def number_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n.succ)).card

theorem divisors_of_sum_for_K_6 :
  let K : ℕ := 6
  let L : ℕ := number_of_divisors K
  number_of_divisors (K + 2 * L) = 4 := by sorry

end NUMINAMATH_CALUDE_divisors_of_sum_for_K_6_l3244_324495


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l3244_324463

/-- Given a circle with center (3.5, -2) and one endpoint of a diameter at (1, -6),
    prove that the other endpoint of the diameter is at (6, 2). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (3.5, -2) →
  endpoint1 = (1, -6) →
  endpoint2 = (6, 2) →
  (center.1 - endpoint1.1 = endpoint2.1 - center.1) ∧
  (center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
  sorry


end NUMINAMATH_CALUDE_circle_diameter_endpoint_l3244_324463


namespace NUMINAMATH_CALUDE_no_solution_equation_l3244_324405

theorem no_solution_equation (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / (2 * x) ≠ y / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l3244_324405


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l3244_324457

theorem sin_two_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : 2 * Real.cos (2*α) = Real.sin (π/4 - α)) : 
  Real.sin (2*α) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l3244_324457


namespace NUMINAMATH_CALUDE_new_barbell_total_cost_l3244_324435

def old_barbell_cost : ℝ := 250

def new_barbell_cost_increase_percentage : ℝ := 0.3

def sales_tax_percentage : ℝ := 0.1

def new_barbell_cost_before_tax : ℝ := old_barbell_cost * (1 + new_barbell_cost_increase_percentage)

def sales_tax : ℝ := new_barbell_cost_before_tax * sales_tax_percentage

def total_cost : ℝ := new_barbell_cost_before_tax + sales_tax

theorem new_barbell_total_cost : total_cost = 357.50 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_total_cost_l3244_324435


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l3244_324437

/-- The area of an isosceles triangle with two sides of length 5 units and a base of 6 units -/
def isosceles_triangle_area : ℝ := 12

/-- The length of the two equal sides of the isosceles triangle -/
def side_length : ℝ := 5

/-- The length of the base of the isosceles triangle -/
def base_length : ℝ := 6

theorem isosceles_triangle_area_theorem :
  let a := side_length
  let b := base_length
  let height := Real.sqrt (a^2 - (b/2)^2)
  (1/2) * b * height = isosceles_triangle_area :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l3244_324437


namespace NUMINAMATH_CALUDE_specific_group_size_l3244_324454

/-- Represents a group of people with language skills -/
structure LanguageGroup where
  latin : ℕ     -- Number of people who can speak Latin
  french : ℕ    -- Number of people who can speak French
  neither : ℕ   -- Number of people who can't speak either Latin or French
  both : ℕ      -- Number of people who can speak both Latin and French

/-- Calculates the total number of people in the group -/
def totalPeople (group : LanguageGroup) : ℕ :=
  (group.latin + group.french - group.both) + group.neither

/-- Theorem: The specific group has 25 people -/
theorem specific_group_size :
  let group : LanguageGroup := {
    latin := 13,
    french := 15,
    neither := 6,
    both := 9
  }
  totalPeople group = 25 := by sorry

end NUMINAMATH_CALUDE_specific_group_size_l3244_324454


namespace NUMINAMATH_CALUDE_part1_selection_count_part2_selection_count_l3244_324402

def num_male : ℕ := 4
def num_female : ℕ := 5
def total_selected : ℕ := 4

def combinations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem part1_selection_count : 
  combinations num_male 2 * combinations num_female 2 = 60 := by sorry

theorem part2_selection_count :
  let total_selections := combinations num_male 1 * combinations num_female 3 +
                          combinations num_male 2 * combinations num_female 2 +
                          combinations num_male 3 * combinations num_female 1
  let invalid_selections := combinations (num_male - 1) 2 +
                            combinations (num_female - 1) 1 * combinations (num_male - 1) 1 +
                            combinations (num_female - 1) 2
  total_selections - invalid_selections = 99 := by sorry

end NUMINAMATH_CALUDE_part1_selection_count_part2_selection_count_l3244_324402


namespace NUMINAMATH_CALUDE_line_equation_correct_l3244_324467

-- Define the line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the equation of the line we want to prove
def line_equation (x y : ℝ) : Prop :=
  -x + y - 2 = 0

-- Theorem statement
theorem line_equation_correct :
  ∀ (x y : ℝ), line_through_points 3 2 1 4 x y ↔ line_equation x y :=
by sorry

end NUMINAMATH_CALUDE_line_equation_correct_l3244_324467


namespace NUMINAMATH_CALUDE_work_completion_time_l3244_324492

theorem work_completion_time (x : ℝ) : 
  x > 0 ∧ 
  5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3244_324492


namespace NUMINAMATH_CALUDE_range_of_f_l3244_324431

-- Define the function f
def f (x : ℝ) : ℝ := 3 - x

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≤ 1, f x = y} = {y : ℝ | y ≥ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3244_324431


namespace NUMINAMATH_CALUDE_zero_mxn_table_l3244_324412

/-- Represents a move on the table -/
inductive Move
  | Row (i : Nat)
  | Column (j : Nat)
  | Diagonal (d : Int)

/-- Represents the state of the table -/
def Table (m n : Nat) := Fin m → Fin n → Int

/-- Applies a move to the table -/
def applyMove (t : Table m n) (move : Move) (delta : Int) : Table m n :=
  sorry

/-- Checks if all elements in the table are zero -/
def allZero (t : Table m n) : Prop :=
  sorry

/-- Checks if we can change all numbers to zero in a 3x3 table -/
def canZero3x3 : Prop :=
  ∀ (t : Table 3 3), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t)

/-- Main theorem: If we can zero any 3x3 table, we can zero any mxn table -/
theorem zero_mxn_table (m n : Nat) (h : canZero3x3) : 
  ∀ (t : Table m n), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t) :=
  sorry

end NUMINAMATH_CALUDE_zero_mxn_table_l3244_324412


namespace NUMINAMATH_CALUDE_basketball_store_problem_l3244_324479

/- Define the basketball types -/
inductive BasketballType
| A
| B

/- Define the purchase and selling prices -/
def purchase_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 80
  | BasketballType.B => 60

def selling_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 120
  | BasketballType.B => 90

/- Define the conditions -/
def condition1 : Prop :=
  20 * purchase_price BasketballType.A + 30 * purchase_price BasketballType.B = 3400

def condition2 : Prop :=
  30 * purchase_price BasketballType.A + 40 * purchase_price BasketballType.B = 4800

def jump_rope_cost : ℕ := 10

/- Define the theorem -/
theorem basketball_store_problem 
  (m n : ℕ) 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : m * selling_price BasketballType.A + n * selling_price BasketballType.B = 5400) :
  (∃ (a b : ℕ), 
    (a * (selling_price BasketballType.A - purchase_price BasketballType.A - jump_rope_cost) + 
     b * (3 * (selling_price BasketballType.B - purchase_price BasketballType.B) - jump_rope_cost) = 600) ∧
    ((a = 12 ∧ b = 3) ∨ (a = 4 ∧ b = 6))) ∧
  (m * (selling_price BasketballType.A - purchase_price BasketballType.A) + 
   n * (selling_price BasketballType.B - purchase_price BasketballType.B) = 1800) :=
by sorry

end NUMINAMATH_CALUDE_basketball_store_problem_l3244_324479


namespace NUMINAMATH_CALUDE_cake_mix_buyers_cake_mix_buyers_is_50_l3244_324461

theorem cake_mix_buyers (total_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (neither_prob : ℚ) (h1 : total_buyers = 100) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 15) (h4 : neither_prob = 1/4) : ℕ :=
by
  -- The number of buyers who purchase cake mix
  sorry

#check cake_mix_buyers

-- The theorem statement proves that given the conditions,
-- the number of buyers who purchase cake mix is 50
theorem cake_mix_buyers_is_50 : 
  cake_mix_buyers 100 40 15 (1/4) rfl rfl rfl rfl = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_cake_mix_buyers_is_50_l3244_324461


namespace NUMINAMATH_CALUDE_factorization_proof_l3244_324408

theorem factorization_proof (x : ℝ) : -8*x^2 + 8*x - 2 = -2*(2*x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3244_324408


namespace NUMINAMATH_CALUDE_solve_for_a_l3244_324477

theorem solve_for_a : ∀ (x a : ℝ), (3 * x - 5 = x + a) ∧ (x = 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3244_324477


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3244_324417

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3244_324417


namespace NUMINAMATH_CALUDE_union_determines_a_l3244_324496

theorem union_determines_a (A B : Set ℝ) (a : ℝ) : 
  A = {1, 2} → 
  B = {a, a^2 + 1} → 
  A ∪ B = {0, 1, 2} → 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_union_determines_a_l3244_324496


namespace NUMINAMATH_CALUDE_sin_390_l3244_324481

-- Define the period of the sine function
def sine_period : ℝ := 360

-- Define the periodicity property of sine
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the known value of sin 30°
axiom sin_30 : Real.sin 30 = 1 / 2

-- Theorem to prove
theorem sin_390 : Real.sin 390 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_l3244_324481


namespace NUMINAMATH_CALUDE_quadratic_max_l3244_324411

theorem quadratic_max (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, -y^2 - 8*y + 16 ≤ m) ∧ (-x^2 - 8*x + 16 = m) → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l3244_324411


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l3244_324446

/-- Calculates Bhanu's expenditure on house rent given his spending pattern and petrol expenditure -/
theorem bhanu_house_rent_expenditure (income : ℝ) (petrol_expenditure : ℝ) :
  petrol_expenditure = 0.3 * income →
  0.3 * (0.7 * income) = 210 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l3244_324446


namespace NUMINAMATH_CALUDE_minimum_guests_l3244_324491

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ min_guests : ℕ, min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3244_324491


namespace NUMINAMATH_CALUDE_integral_equation_solution_l3244_324449

theorem integral_equation_solution (k : ℝ) : (∫ x in (0:ℝ)..1, (3 * x^2 + k)) = 10 ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l3244_324449


namespace NUMINAMATH_CALUDE_equation_solution_l3244_324440

theorem equation_solution : ∃ x : ℝ, 10.0003 * x = 10000.3 ∧ x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3244_324440


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l3244_324460

def distribute_teachers (n : ℕ) (k : ℕ) (min_a : ℕ) (min_others : ℕ) : ℕ :=
  -- n: total number of teachers
  -- k: number of schools
  -- min_a: minimum number of teachers for school A
  -- min_others: minimum number of teachers for other schools
  sorry

theorem teacher_distribution_count :
  distribute_teachers 6 4 2 1 = 660 := by sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l3244_324460


namespace NUMINAMATH_CALUDE_huron_michigan_fishes_l3244_324475

def total_fishes : ℕ := 97
def ontario_erie_fishes : ℕ := 23
def superior_fishes : ℕ := 44

theorem huron_michigan_fishes :
  total_fishes - (ontario_erie_fishes + superior_fishes) = 30 := by
  sorry

end NUMINAMATH_CALUDE_huron_michigan_fishes_l3244_324475


namespace NUMINAMATH_CALUDE_three_eighths_divided_by_one_fourth_l3244_324466

theorem three_eighths_divided_by_one_fourth : (3 : ℚ) / 8 / ((1 : ℚ) / 4) = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_divided_by_one_fourth_l3244_324466


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3244_324442

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 35)
  (eq2 : 3 * u + 5 * v = -10) :
  u + v = -40 / 43 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3244_324442


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3244_324409

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  (f 3 = 0 ∧ f (-1) = 0) ∧
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3244_324409


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourth_l3244_324428

theorem sin_thirteen_pi_fourth : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourth_l3244_324428


namespace NUMINAMATH_CALUDE_lcm_problem_l3244_324439

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) :
  Nat.lcm b c = 540 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3244_324439


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3244_324400

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3244_324400


namespace NUMINAMATH_CALUDE_doctor_appointment_distance_l3244_324497

/-- Represents the distances Tony needs to drive for his errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the total distance for all errands -/
def totalDistance (d : ErrandDistances) : ℕ :=
  d.groceries + d.haircut + d.doctor

theorem doctor_appointment_distance :
  ∀ (d : ErrandDistances),
    d.groceries = 10 →
    d.haircut = 15 →
    totalDistance d / 2 = 15 →
    d.doctor = 5 := by
  sorry

end NUMINAMATH_CALUDE_doctor_appointment_distance_l3244_324497


namespace NUMINAMATH_CALUDE_car_distance_proof_l3244_324406

/-- Proves that the distance covered by a car is 450 km given the specified conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 50 →
  (3/2 : ℝ) * initial_time * speed = 450 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3244_324406


namespace NUMINAMATH_CALUDE_range_of_m_l3244_324445

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4/x + 1/y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 4/x + 1/y = 1 → x + y ≥ m^2 + m + 3) :
  -3 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3244_324445


namespace NUMINAMATH_CALUDE_probability_theorem_l3244_324430

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The probability of drawing two balls of different colors without replacement -/
def prob_different_without_replacement : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2 : ℚ)

/-- The probability of drawing two balls of different colors with replacement -/
def prob_different_with_replacement : ℚ :=
  2 * (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls

theorem probability_theorem :
  prob_different_without_replacement = 3/5 ∧
  prob_different_with_replacement = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3244_324430


namespace NUMINAMATH_CALUDE_abc_divisibility_theorem_l3244_324415

theorem abc_divisibility_theorem (a b c : ℕ+) 
  (h1 : a * b ∣ c * (c^2 - c + 1)) 
  (h2 : (c^2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c^2 - c + 1) ∨ (a = c^2 - c + 1 ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_abc_divisibility_theorem_l3244_324415


namespace NUMINAMATH_CALUDE_beads_per_earring_is_five_l3244_324418

/-- The number of beads needed to make one earring given Kylie's jewelry-making activities --/
def beads_per_earring : ℕ :=
  let necklaces_monday : ℕ := 10
  let necklaces_tuesday : ℕ := 2
  let bracelets : ℕ := 5
  let earrings : ℕ := 7
  let beads_per_necklace : ℕ := 20
  let beads_per_bracelet : ℕ := 10
  let total_beads : ℕ := 325
  let beads_for_necklaces : ℕ := (necklaces_monday + necklaces_tuesday) * beads_per_necklace
  let beads_for_bracelets : ℕ := bracelets * beads_per_bracelet
  let beads_for_earrings : ℕ := total_beads - beads_for_necklaces - beads_for_bracelets
  beads_for_earrings / earrings

theorem beads_per_earring_is_five : beads_per_earring = 5 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_earring_is_five_l3244_324418


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l3244_324432

theorem complex_roots_theorem (a b c : ℂ) : 
  a + b + c = 1 ∧ 
  a * b + a * c + b * c = 1 ∧ 
  a * b * c = -1 → 
  ({a, b, c} : Set ℂ) = {1, Complex.I, -Complex.I} :=
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l3244_324432


namespace NUMINAMATH_CALUDE_complex_number_location_l3244_324476

theorem complex_number_location (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (m - 1) (3*m - 2)) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3244_324476


namespace NUMINAMATH_CALUDE_pizza_toppings_l3244_324450

theorem pizza_toppings (total_slices cheese_slices olive_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : cheese_slices = 16)
  (h3 : olive_slices = 18)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ cheese_slices ∨ slice ≤ olive_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = 10 ∧ 
    cheese_slices + olive_slices - both_toppings = total_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3244_324450


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l3244_324436

-- Theorem 1
theorem square_sum_ge_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

-- Theorem 2
theorem abs_diff_product_gt_abs_diff (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |1 - a*b| > |a - b| := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l3244_324436


namespace NUMINAMATH_CALUDE_ratio_yz_l3244_324489

theorem ratio_yz (x y z : ℝ) 
  (h1 : (x + 53/18 * y - 143/9 * z) / z = 1)
  (h2 : (3/8 * x - 17/4 * y + z) / y = 1) :
  y / z = 352 / 305 := by
sorry

end NUMINAMATH_CALUDE_ratio_yz_l3244_324489


namespace NUMINAMATH_CALUDE_biff_break_even_l3244_324426

/-- The number of hours required for Biff to break even on his bus trip -/
def break_even_hours (ticket_cost drinks_snacks_cost headphones_cost online_earnings wifi_cost : ℚ) : ℚ :=
  (ticket_cost + drinks_snacks_cost + headphones_cost) / (online_earnings - wifi_cost)

/-- Theorem stating that Biff needs 3 hours to break even on his bus trip -/
theorem biff_break_even :
  break_even_hours 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_l3244_324426


namespace NUMINAMATH_CALUDE_concert_songs_count_l3244_324453

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- Calculates the total number of songs sung by the trios -/
def total_songs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The theorem to be proved -/
theorem concert_songs_count :
  ∀ (sc : SongCount),
    sc.mary = 3 →
    sc.alina = 5 →
    sc.hanna = 6 →
    sc.mary < sc.tina →
    sc.tina < sc.hanna →
    total_songs sc = 6 := by
  sorry


end NUMINAMATH_CALUDE_concert_songs_count_l3244_324453


namespace NUMINAMATH_CALUDE_total_drive_distance_l3244_324455

/-- The total distance of a drive given two drivers with different speed limits and driving times -/
theorem total_drive_distance (christina_speed : ℝ) (friend_speed : ℝ) (christina_time_min : ℝ) (friend_time_hr : ℝ) : 
  christina_speed = 30 →
  friend_speed = 40 →
  christina_time_min = 180 →
  friend_time_hr = 3 →
  christina_speed * (christina_time_min / 60) + friend_speed * friend_time_hr = 210 := by
  sorry

#check total_drive_distance

end NUMINAMATH_CALUDE_total_drive_distance_l3244_324455


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3244_324451

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 ≤ (5-x)/2 ∧ (5-x)/2 < 1/3) ↔ (13/3 < x ∧ x ≤ 17/3) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3244_324451


namespace NUMINAMATH_CALUDE_problem_solution_l3244_324403

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 * b + a * b^2 = 2) :
  (a^3 + b^3 ≥ 2) ∧ ((a + b) * (a^5 + b^5) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3244_324403


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3244_324419

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 121 - y^2 / 49 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) :=
  {(11, 0), (-11, 0)}

-- Theorem statement
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3244_324419


namespace NUMINAMATH_CALUDE_winning_number_correct_l3244_324421

/-- The number of callers needed to win all three prizes -/
def winning_number : ℕ := 1125

/-- The maximum allowed number of callers -/
def max_callers : ℕ := 2000

/-- Checks if a number is divisible by another number -/
def is_divisible (a b : ℕ) : Prop := b ∣ a

/-- Checks if a number is not divisible by 10 -/
def not_multiple_of_ten (n : ℕ) : Prop := ¬(is_divisible n 10)

/-- Theorem stating the winning number is correct -/
theorem winning_number_correct :
  (is_divisible winning_number 100) ∧ 
  (is_divisible winning_number 40) ∧ 
  (is_divisible winning_number 250) ∧
  (not_multiple_of_ten winning_number) ∧
  (∀ n : ℕ, n < winning_number → 
    ¬(is_divisible n 100 ∧ is_divisible n 40 ∧ is_divisible n 250 ∧ not_multiple_of_ten n)) ∧
  (winning_number ≤ max_callers) :=
sorry

end NUMINAMATH_CALUDE_winning_number_correct_l3244_324421


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3244_324413

theorem possible_values_of_a (a b x : ℤ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27*x^3) (h3 : a - b = 2*x) :
  a = (7*x + 5*(6: ℤ).sqrt*x) / 6 ∨ a = (7*x - 5*(6: ℤ).sqrt*x) / 6 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3244_324413


namespace NUMINAMATH_CALUDE_graduating_class_size_l3244_324443

theorem graduating_class_size (boys : ℕ) (girls : ℕ) (h1 : boys = 127) (h2 : girls = boys + 212) :
  boys + girls = 466 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_size_l3244_324443


namespace NUMINAMATH_CALUDE_fourth_root_plus_cube_root_equation_solutions_l3244_324473

theorem fourth_root_plus_cube_root_equation_solutions :
  ∀ x : ℝ, (((3 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/3) : ℝ) = 1) ↔ (x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_plus_cube_root_equation_solutions_l3244_324473


namespace NUMINAMATH_CALUDE_max_sides_in_subdivision_l3244_324478

/-- 
Given a convex polygon with n sides and all its diagonals drawn,
the maximum number of sides a polygon in the subdivision can have is n.
-/
theorem max_sides_in_subdivision (n : ℕ) (h : n ≥ 3) :
  ∃ (max_sides : ℕ), max_sides = n ∧ 
  ∀ (subdivided_polygon_sides : ℕ), 
    subdivided_polygon_sides ≤ max_sides :=
by sorry

end NUMINAMATH_CALUDE_max_sides_in_subdivision_l3244_324478


namespace NUMINAMATH_CALUDE_inequality_no_solution_l3244_324414

theorem inequality_no_solution : {x : ℝ | x * (2 - x) > 3} = ∅ := by sorry

end NUMINAMATH_CALUDE_inequality_no_solution_l3244_324414


namespace NUMINAMATH_CALUDE_wolf_hunting_problem_l3244_324410

theorem wolf_hunting_problem (hunting_wolves : ℕ) (pack_wolves : ℕ) (meat_per_wolf : ℕ) 
  (hunting_days : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 → 
  pack_wolves = 16 → 
  meat_per_wolf = 8 → 
  hunting_days = 5 → 
  meat_per_deer = 200 → 
  (hunting_wolves + pack_wolves) * meat_per_wolf * hunting_days / meat_per_deer / hunting_wolves = 1 := by
  sorry

#check wolf_hunting_problem

end NUMINAMATH_CALUDE_wolf_hunting_problem_l3244_324410


namespace NUMINAMATH_CALUDE_peru_tst_imo_2006_q1_l3244_324468

theorem peru_tst_imo_2006_q1 : 
  {(x, y, z) : ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∃ k : ℕ, (Real.sqrt (2006 / (x + y : ℝ)) + 
               Real.sqrt (2006 / (y + z : ℝ)) + 
               Real.sqrt (2006 / (z + x : ℝ))) = k}
  = {(2006, 2006, 2006), (1003, 1003, 7021), (9027, 9027, 9027)} := by
  sorry


end NUMINAMATH_CALUDE_peru_tst_imo_2006_q1_l3244_324468


namespace NUMINAMATH_CALUDE_business_partnership_timing_l3244_324485

/-- Proves that B joined the business 8 months after A started, given the conditions of the problem -/
theorem business_partnership_timing (a_initial_capital b_capital : ℕ) (x : ℕ) : 
  a_initial_capital = 3500 →
  b_capital = 15750 →
  (a_initial_capital * 12) / (b_capital * (12 - x)) = 2 / 3 →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_timing_l3244_324485


namespace NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l3244_324470

/-- Represents the composition of trail mix -/
structure TrailMix where
  peanuts : ℝ
  chocolate_chips : ℝ
  raisins : ℝ

/-- The total weight of the trail mix -/
def total_weight (mix : TrailMix) : ℝ :=
  mix.peanuts + mix.chocolate_chips + mix.raisins

theorem brandy_trail_mix_peanuts :
  ∀ (mix : TrailMix),
    mix.chocolate_chips = 0.17 →
    mix.raisins = 0.08 →
    total_weight mix = 0.42 →
    mix.peanuts = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l3244_324470


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_base_areas_l3244_324498

theorem pentagonal_pyramid_base_areas (total_surface_area lateral_surface_area : ℝ) :
  total_surface_area = 30 →
  lateral_surface_area = 25 →
  total_surface_area - lateral_surface_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_base_areas_l3244_324498


namespace NUMINAMATH_CALUDE_translate_linear_function_l3244_324438

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (dy : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + dy }

/-- The theorem stating that translating y = -2x + 1 up 4 units results in y = -2x + 5 -/
theorem translate_linear_function :
  let f : LinearFunction := { m := -2, b := 1 }
  let g : LinearFunction := translateVertically f 4
  g.m = -2 ∧ g.b = 5 := by sorry

end NUMINAMATH_CALUDE_translate_linear_function_l3244_324438


namespace NUMINAMATH_CALUDE_equation_solution_l3244_324471

theorem equation_solution : ∃ x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3244_324471


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3244_324480

theorem ratio_of_numbers (sum : ℚ) (bigger : ℚ) (h1 : sum = 143) (h2 : bigger = 104) :
  (sum - bigger) / bigger = 39 / 104 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3244_324480


namespace NUMINAMATH_CALUDE_maddie_thursday_viewing_l3244_324459

/-- Represents the viewing schedule for a TV show --/
structure ViewingSchedule where
  totalEpisodes : ℕ
  episodeLength : ℕ
  mondayMinutes : ℕ
  fridayEpisodes : ℕ
  weekendMinutes : ℕ

/-- Calculates the number of minutes watched on Thursday --/
def thursdayMinutes (schedule : ViewingSchedule) : ℕ :=
  schedule.totalEpisodes * schedule.episodeLength -
  (schedule.mondayMinutes + schedule.fridayEpisodes * schedule.episodeLength + schedule.weekendMinutes)

/-- Theorem stating that Maddie watched 21 minutes on Thursday --/
theorem maddie_thursday_viewing : 
  let schedule : ViewingSchedule := {
    totalEpisodes := 8,
    episodeLength := 44,
    mondayMinutes := 138,
    fridayEpisodes := 2,
    weekendMinutes := 105
  }
  thursdayMinutes schedule = 21 := by
  sorry

end NUMINAMATH_CALUDE_maddie_thursday_viewing_l3244_324459


namespace NUMINAMATH_CALUDE_dvd_price_proof_l3244_324422

/-- The price Mike paid for the DVD at the store -/
def mike_price : ℝ := 5

/-- The price Steve paid for the DVD online -/
def steve_online_price (p : ℝ) : ℝ := 2 * p

/-- The shipping cost Steve paid -/
def steve_shipping_cost (p : ℝ) : ℝ := 0.8 * steve_online_price p

/-- The total amount Steve paid -/
def steve_total_cost (p : ℝ) : ℝ := steve_online_price p + steve_shipping_cost p

theorem dvd_price_proof :
  steve_total_cost mike_price = 18 :=
sorry

end NUMINAMATH_CALUDE_dvd_price_proof_l3244_324422


namespace NUMINAMATH_CALUDE_jason_music_store_expense_l3244_324423

/-- The amount Jason spent at the music store -/
def jason_total_spent (flute_cost music_stand_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_stand_cost + song_book_cost

/-- Theorem: Jason spent $158.35 at the music store -/
theorem jason_music_store_expense :
  jason_total_spent 142.46 8.89 7 = 158.35 := by
  sorry

end NUMINAMATH_CALUDE_jason_music_store_expense_l3244_324423


namespace NUMINAMATH_CALUDE_triangle_angle_value_l3244_324486

theorem triangle_angle_value (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * c * Real.sin A = a * Real.cos C →
  C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l3244_324486


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3244_324425

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = -1 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -3)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3244_324425
