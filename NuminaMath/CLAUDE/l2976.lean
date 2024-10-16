import Mathlib

namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2976_297644

theorem fraction_to_decimal : (73 : ℚ) / 160 = (45625 : ℚ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2976_297644


namespace NUMINAMATH_CALUDE_max_intersections_three_lines_circle_l2976_297665

/-- A line in a 2D plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The number of intersection points between a line and a circle -/
def line_circle_intersections (l : Line) (c : Circle) : ℕ := sorry

/-- The number of intersection points between two lines -/
def line_line_intersections (l1 l2 : Line) : ℕ := sorry

/-- Three distinct lines -/
def three_distinct_lines : Prop :=
  ∃ (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3

theorem max_intersections_three_lines_circle :
  ∀ (l1 l2 l3 : Line) (c : Circle),
  three_distinct_lines →
  (line_circle_intersections l1 c +
   line_circle_intersections l2 c +
   line_circle_intersections l3 c +
   line_line_intersections l1 l2 +
   line_line_intersections l1 l3 +
   line_line_intersections l2 l3) ≤ 9 ∧
  ∃ (l1' l2' l3' : Line) (c' : Circle),
    three_distinct_lines →
    (line_circle_intersections l1' c' +
     line_circle_intersections l2' c' +
     line_circle_intersections l3' c' +
     line_line_intersections l1' l2' +
     line_line_intersections l1' l3' +
     line_line_intersections l2' l3') = 9 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_three_lines_circle_l2976_297665


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l2976_297659

/-- The length of the path traveled by a point on a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (Real.pi * r / 2)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l2976_297659


namespace NUMINAMATH_CALUDE_equation_solvability_l2976_297629

theorem equation_solvability (n : ℕ) (hn : Odd n) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 4 / n = 1 / x + 1 / y) ↔
  (∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∃ k : ℕ, d = 4 * k + 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solvability_l2976_297629


namespace NUMINAMATH_CALUDE_percentage_less_than_l2976_297671

theorem percentage_less_than (p t j : ℝ) 
  (ht : t = p * (1 - 0.0625))
  (hj : j = t * (1 - 0.20)) : 
  j = p * (1 - 0.25) := by
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l2976_297671


namespace NUMINAMATH_CALUDE_indefinite_integral_sin_3x_l2976_297690

theorem indefinite_integral_sin_3x (x : ℝ) :
  (deriv (fun x => -1/3 * (x + 5) * Real.cos (3 * x) + 1/9 * Real.sin (3 * x))) x
  = (x + 5) * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_sin_3x_l2976_297690


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l2976_297682

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 →  -- Ensures it's a three-digit number
  a = 2*c - 3 →  -- Hundreds digit is 3 less than twice the units digit
  ((100*a + 10*b + c) - ((100*c + 10*b + a) + 50)) % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l2976_297682


namespace NUMINAMATH_CALUDE_range_of_valid_m_l2976_297625

/-- The set A as defined in the problem -/
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2) 2, y = x^2 - (3/2)*x + 1}

/-- The set B as defined in the problem -/
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

/-- The range of values for m that satisfies the condition A ⊆ B -/
def valid_m : Set ℝ := {m | A ⊆ B m}

/-- Theorem stating that the range of valid m is (-∞, -9/16] ∪ [3, +∞) -/
theorem range_of_valid_m : valid_m = Set.Iic (-9/16) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_range_of_valid_m_l2976_297625


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2976_297664

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2976_297664


namespace NUMINAMATH_CALUDE_five_hundred_billion_scientific_notation_l2976_297652

/-- Express 500 billion in scientific notation -/
theorem five_hundred_billion_scientific_notation :
  (500000000000 : ℝ) = 5 * 10^11 := by
  sorry

end NUMINAMATH_CALUDE_five_hundred_billion_scientific_notation_l2976_297652


namespace NUMINAMATH_CALUDE_pascal_triangle_50th_row_third_number_l2976_297647

theorem pascal_triangle_50th_row_third_number :
  Nat.choose 50 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_50th_row_third_number_l2976_297647


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l2976_297617

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 15 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l2976_297617


namespace NUMINAMATH_CALUDE_planar_cube_area_is_600_l2976_297645

/-- The side length of each square in centimeters -/
def side_length : ℝ := 10

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The area of the planar figure of a cube in square centimeters -/
def planar_cube_area : ℝ := side_length^2 * cube_faces

/-- Theorem: The area of a planar figure representing a cube, 
    made up of squares with side length 10 cm, is 600 cm² -/
theorem planar_cube_area_is_600 : planar_cube_area = 600 := by
  sorry

end NUMINAMATH_CALUDE_planar_cube_area_is_600_l2976_297645


namespace NUMINAMATH_CALUDE_square_area_6cm_l2976_297610

/-- The area of a square with side length 6 cm is 36 square centimeters. -/
theorem square_area_6cm : 
  let side_length : ℝ := 6
  let area : ℝ := side_length * side_length
  area = 36 := by sorry

end NUMINAMATH_CALUDE_square_area_6cm_l2976_297610


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2976_297606

theorem inequality_equivalence (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2976_297606


namespace NUMINAMATH_CALUDE_smallest_n_dividing_2016_l2976_297673

theorem smallest_n_dividing_2016 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (2016 ∣ (20^m - 16^m)) → m ≥ n) ∧ 
  (2016 ∣ (20^n - 16^n)) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_dividing_2016_l2976_297673


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2976_297633

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def l (x y : ℝ) : Prop := x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := (x-2)^2 + (y-4)^2 = 20

-- Define the ray
def ray (x y : ℝ) : Prop := 2*x - y = 0 ∧ x ≥ 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y, C₁ x y → l x y → (x = 1 ∧ y = 1)) →  -- l is tangent to C₁ at (1,1)
  (∃ a b, ray a b ∧ ∀ x y, C₂ x y → (x - a)^2 + (y - b)^2 = (x^2 + y^2)) →  -- Center of C₂ is on the ray and C₂ passes through origin
  (∃ x₁ y₁ x₂ y₂, C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 48) →  -- Chord length is 4√3
  -- Conclusion
  (∀ x y, l x y ↔ x + y - 2 = 0) ∧
  (∀ x y, C₂ x y ↔ (x-2)^2 + (y-4)^2 = 20) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l2976_297633


namespace NUMINAMATH_CALUDE_john_completion_time_l2976_297669

/-- The number of days it takes Jane to complete the task alone -/
def jane_days : ℝ := 12

/-- The total number of days it took to complete the task -/
def total_days : ℝ := 10.8

/-- The number of days Jane was indisposed before the work was completed -/
def jane_indisposed : ℝ := 6

/-- The number of days it takes John to complete the task alone -/
def john_days : ℝ := 18

theorem john_completion_time :
  (jane_indisposed / john_days) + 
  ((total_days - jane_indisposed) * (1 / john_days + 1 / jane_days)) = 1 :=
sorry

#check john_completion_time

end NUMINAMATH_CALUDE_john_completion_time_l2976_297669


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2976_297618

/-- The quadratic function f(x) = x^2 - ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a

/-- The discriminant of f(x) -/
def discriminant (a : ℝ) : ℝ := a^2 - 4*a

/-- f(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop := discriminant a > 0

/-- Condition "a > 4" is sufficient for f(x) to have two distinct zeros -/
theorem sufficient_condition (a : ℝ) (h : a > 4) : has_two_distinct_zeros a := by
  sorry

/-- Condition "a > 4" is not necessary for f(x) to have two distinct zeros -/
theorem not_necessary_condition : ∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a := by
  sorry

/-- "a > 4" is a sufficient but not necessary condition for f(x) to have two distinct zeros -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a > 4 → has_two_distinct_zeros a) ∧
  (∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2976_297618


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2976_297674

theorem smallest_part_of_proportional_division (total : ℝ) (prop1 prop2 prop3 : ℝ) (additional : ℝ) :
  total = 120 ∧ prop1 = 3 ∧ prop2 = 5 ∧ prop3 = 7 ∧ additional = 4 →
  let x := (total - 3 * additional) / (prop1 + prop2 + prop3)
  let part1 := prop1 * x + additional
  let part2 := prop2 * x + additional
  let part3 := prop3 * x + additional
  min part1 (min part2 part3) = 25.6 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2976_297674


namespace NUMINAMATH_CALUDE_f_properties_l2976_297640

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 0 else 1 - 1 / x

theorem f_properties :
  (f 1 = 0) ∧
  (∀ x > 1, f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x + y > 0 → f (x * f y) * f y = f (x * y / (x + y))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2976_297640


namespace NUMINAMATH_CALUDE_inequality_solution_l2976_297611

theorem inequality_solution (x : ℝ) : (x - 3) / (x^2 + 4*x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2976_297611


namespace NUMINAMATH_CALUDE_percentage_calculation_l2976_297614

theorem percentage_calculation (x : ℝ) : 
  (x / 100) * (25 / 100 * 1600) = 20 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2976_297614


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2976_297627

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2976_297627


namespace NUMINAMATH_CALUDE_linda_coloring_books_l2976_297635

/-- Represents Linda's purchase --/
structure Purchase where
  coloringBookPrice : ℝ
  coloringBookCount : ℕ
  peanutPackPrice : ℝ
  peanutPackCount : ℕ
  stuffedAnimalPrice : ℝ
  totalPaid : ℝ

/-- Theorem stating the number of coloring books Linda bought --/
theorem linda_coloring_books (p : Purchase) 
  (h1 : p.coloringBookPrice = 4)
  (h2 : p.peanutPackPrice = 1.5)
  (h3 : p.peanutPackCount = 4)
  (h4 : p.stuffedAnimalPrice = 11)
  (h5 : p.totalPaid = 25)
  (h6 : p.coloringBookPrice * p.coloringBookCount + 
        p.peanutPackPrice * p.peanutPackCount + 
        p.stuffedAnimalPrice = p.totalPaid) :
  p.coloringBookCount = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_coloring_books_l2976_297635


namespace NUMINAMATH_CALUDE_two_dressers_capacity_l2976_297636

/-- The total number of pieces of clothing that can be held by two dressers -/
def total_clothing_capacity (first_dresser_drawers : ℕ) (first_dresser_capacity : ℕ) 
  (second_dresser_drawers : ℕ) (second_dresser_capacity : ℕ) : ℕ :=
  first_dresser_drawers * first_dresser_capacity + second_dresser_drawers * second_dresser_capacity

/-- Theorem stating the total clothing capacity of two specific dressers -/
theorem two_dressers_capacity : 
  total_clothing_capacity 12 8 6 10 = 156 := by
  sorry

end NUMINAMATH_CALUDE_two_dressers_capacity_l2976_297636


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2976_297651

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2976_297651


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_angles_l2976_297680

theorem isosceles_right_triangle_angles (α : ℝ) :
  α > 0 ∧ α < 90 →
  (α + α + 90 = 180) →
  α = 45 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_angles_l2976_297680


namespace NUMINAMATH_CALUDE_inequality_proof_l2976_297620

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2976_297620


namespace NUMINAMATH_CALUDE_circle_equation_constant_l2976_297648

theorem circle_equation_constant (F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 8*y + F = 0 → 
    ∃ h k : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - k)^2 = 4^2 → 
      x'^2 + y'^2 - 4*x' + 8*y' + F = 0) → 
  F = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_constant_l2976_297648


namespace NUMINAMATH_CALUDE_hall_width_is_15_l2976_297676

/-- Represents the dimensions and cost information of a rectangular hall -/
structure Hall where
  length : ℝ
  height : ℝ
  width : ℝ
  cost_per_sqm : ℝ
  total_expenditure : ℝ

/-- Calculates the total area to be covered with mat in the hall -/
def total_area (h : Hall) : ℝ :=
  2 * (h.length * h.width) + 2 * (h.length * h.height) + 2 * (h.width * h.height)

/-- Theorem stating that given the hall's dimensions and cost information, the width is 15 meters -/
theorem hall_width_is_15 (h : Hall) 
  (h_length : h.length = 20)
  (h_height : h.height = 5)
  (h_cost : h.cost_per_sqm = 50)
  (h_expenditure : h.total_expenditure = 47500)
  (h_area_eq : h.total_expenditure = (total_area h) * h.cost_per_sqm) :
  h.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_is_15_l2976_297676


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2976_297657

theorem cube_root_simplification : Real.rpow (4^6 * 5^3 * 7^3) (1/3) = 560 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2976_297657


namespace NUMINAMATH_CALUDE_solve_roses_problem_l2976_297602

def roses_problem (initial_roses : ℕ) (price_per_rose : ℕ) (total_earnings : ℕ) : Prop :=
  let roses_sold : ℕ := total_earnings / price_per_rose
  let roses_left : ℕ := initial_roses - roses_sold
  roses_left = 4

theorem solve_roses_problem :
  roses_problem 13 4 36 := by
  sorry

end NUMINAMATH_CALUDE_solve_roses_problem_l2976_297602


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l2976_297654

theorem polygon_interior_angle_sum (n : ℕ) (h : n * 40 = 360) : 
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l2976_297654


namespace NUMINAMATH_CALUDE_tom_annual_lease_cost_l2976_297677

/-- Represents Tom's car lease scenario -/
structure CarLease where
  short_drive_days : Nat
  short_drive_miles : Nat
  long_drive_miles : Nat
  cost_per_mile : Rat
  weekly_fee : Nat

/-- Calculates the total annual cost for Tom's car lease -/
def annual_cost (lease : CarLease) : Rat :=
  let days_in_week : Nat := 7
  let weeks_in_year : Nat := 52
  let long_drive_days : Nat := days_in_week - lease.short_drive_days
  let weekly_mileage : Nat := lease.short_drive_days * lease.short_drive_miles + long_drive_days * lease.long_drive_miles
  let weekly_mileage_cost : Rat := (weekly_mileage : Rat) * lease.cost_per_mile
  let total_weekly_cost : Rat := weekly_mileage_cost + (lease.weekly_fee : Rat)
  total_weekly_cost * (weeks_in_year : Rat)

/-- Theorem stating that Tom's annual car lease cost is $7800 -/
theorem tom_annual_lease_cost :
  let tom_lease : CarLease := {
    short_drive_days := 4
    short_drive_miles := 50
    long_drive_miles := 100
    cost_per_mile := 1/10
    weekly_fee := 100
  }
  annual_cost tom_lease = 7800 := by sorry

end NUMINAMATH_CALUDE_tom_annual_lease_cost_l2976_297677


namespace NUMINAMATH_CALUDE_loaded_cartons_l2976_297624

/-- Given information about cartons of canned juice, prove the number of loaded cartons. -/
theorem loaded_cartons (total_cartons : ℕ) (cans_per_carton : ℕ) (cans_left : ℕ) : 
  total_cartons = 50 →
  cans_per_carton = 20 →
  cans_left = 200 →
  total_cartons - (cans_left / cans_per_carton) = 40 :=
by sorry

end NUMINAMATH_CALUDE_loaded_cartons_l2976_297624


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2976_297607

theorem amount_after_two_years 
  (initial_amount : ℝ) 
  (annual_increase_rate : ℝ) 
  (years : ℕ) :
  initial_amount = 32000 →
  annual_increase_rate = 1 / 8 →
  years = 2 →
  initial_amount * (1 + annual_increase_rate) ^ years = 40500 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2976_297607


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2976_297623

theorem original_denominator_problem (d : ℝ) : 
  (3 : ℝ) / d ≠ 0 →
  (3 + 3) / (d + 3) = (1 : ℝ) / 3 →
  d = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2976_297623


namespace NUMINAMATH_CALUDE_johns_total_pay_johns_total_pay_this_year_l2976_297650

/-- Calculates the total pay (salary + bonus) given a salary and bonus percentage -/
def totalPay (salary : ℝ) (bonusPercentage : ℝ) : ℝ :=
  salary * (1 + bonusPercentage)

/-- Theorem: John's total pay is equal to his salary plus his bonus -/
theorem johns_total_pay (salary : ℝ) (bonusPercentage : ℝ) :
  totalPay salary bonusPercentage = salary + (salary * bonusPercentage) :=
by sorry

/-- Theorem: John's total pay this year is $220,000 -/
theorem johns_total_pay_this_year 
  (lastYearSalary lastYearBonus thisYearSalary : ℝ)
  (h1 : lastYearSalary = 100000)
  (h2 : lastYearBonus = 10000)
  (h3 : thisYearSalary = 200000)
  (h4 : lastYearBonus / lastYearSalary = thisYearSalary * bonusPercentage / thisYearSalary) :
  totalPay thisYearSalary (lastYearBonus / lastYearSalary) = 220000 :=
by sorry

end NUMINAMATH_CALUDE_johns_total_pay_johns_total_pay_this_year_l2976_297650


namespace NUMINAMATH_CALUDE_cubic_factorization_l2976_297603

theorem cubic_factorization (x y : ℝ) : x^3 - x*y^2 = x*(x-y)*(x+y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2976_297603


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2976_297613

/-- A 2x2 matrix is a projection matrix if and only if Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The specific form of our matrix Q -/
def Q (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/49; c, 29/49]

theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (Q a c) → a = 20/49 ∧ c = 29/49 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2976_297613


namespace NUMINAMATH_CALUDE_expression_evaluation_l2976_297660

theorem expression_evaluation : (28 / (8 - 3 + 2)) * (4 - 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2976_297660


namespace NUMINAMATH_CALUDE_min_value_problem_l2976_297695

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_eq : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*(y'^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2976_297695


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2976_297600

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4/9 →
  ((4/3) * Real.pi * r₁^3) / ((4/3) * Real.pi * r₂^3) = 8/27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2976_297600


namespace NUMINAMATH_CALUDE_x_range_theorem_l2976_297619

theorem x_range_theorem (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ x : ℝ, (1/a) + (4/b) ≥ |2*x - 1| - |x + 1|) :
  ∀ x : ℝ, -7 ≤ x ∧ x ≤ 11 := by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l2976_297619


namespace NUMINAMATH_CALUDE_braden_winnings_l2976_297615

/-- The amount of money Braden has after winning two bets, given his initial amount --/
def final_amount (initial_amount : ℕ) : ℕ :=
  initial_amount + 2 * initial_amount + 2 * initial_amount

/-- Theorem stating that Braden's final amount is $2000 given an initial amount of $400 --/
theorem braden_winnings :
  final_amount 400 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_braden_winnings_l2976_297615


namespace NUMINAMATH_CALUDE_gross_profit_calculation_l2976_297604

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 44 ∧ gross_profit_percentage = 1.2 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    sales_price = cost + gross_profit ∧
    gross_profit = gross_profit_percentage * cost ∧
    gross_profit = 24 := by
  sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l2976_297604


namespace NUMINAMATH_CALUDE_area_of_triangle_KBC_l2976_297621

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Represents a square -/
structure Square :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a hexagon is equiangular -/
def isEquiangular (h : Hexagon) : Prop := sorry

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- The length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_triangle_KBC 
  (ABCDEF : Hexagon) 
  (ABJI FEHG : Square) 
  (JBK : Triangle) :
  isEquiangular ABCDEF →
  squareArea ABJI = 25 →
  squareArea FEHG = 49 →
  isIsosceles JBK →
  distance ABCDEF.F ABCDEF.E = distance ABCDEF.B ABCDEF.C →
  triangleArea ⟨JBK.B, ABCDEF.B, ABCDEF.C⟩ = 49 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_KBC_l2976_297621


namespace NUMINAMATH_CALUDE_cubic_root_cubes_l2976_297667

/-- Given a cubic equation x^3 + ax^2 + bx + c = 0 with roots α, β, and γ,
    the cubic equation with roots α^3, β^3, and γ^3 is
    x^3 + (a^3 - 3ab + 3c)x^2 + (b^3 + 3c^2 - 3abc)x + c^3 -/
theorem cubic_root_cubes (a b c : ℝ) (α β γ : ℝ) :
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℝ, x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3 = 0
           ↔ x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_cubes_l2976_297667


namespace NUMINAMATH_CALUDE_jellybean_problem_l2976_297699

theorem jellybean_problem (initial_quantity : ℝ) : 
  (0.75^3 * initial_quantity = 27) → initial_quantity = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2976_297699


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l2976_297675

theorem fixed_point_parabola :
  ∀ (t : ℝ), 36 = 4 * (3 : ℝ)^2 + t * 3 - 3 * t := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l2976_297675


namespace NUMINAMATH_CALUDE_effective_price_for_8kg_l2976_297672

/-- Represents the shopkeeper's pricing scheme -/
structure PricingScheme where
  false_weight : Real
  discount_rate : Real
  tax_rate : Real

/-- Calculates the effective price for a given purchase -/
def effective_price (scheme : PricingScheme) (purchase_weight : Real) (cost_price : Real) : Real :=
  let actual_weight := purchase_weight * (scheme.false_weight / 1000)
  let discounted_price := purchase_weight * cost_price * (1 - scheme.discount_rate)
  discounted_price * (1 + scheme.tax_rate)

/-- Theorem stating the effective price for the given scenario -/
theorem effective_price_for_8kg (scheme : PricingScheme) (cost_price : Real) :
  scheme.false_weight = 980 →
  scheme.discount_rate = 0.1 →
  scheme.tax_rate = 0.03 →
  effective_price scheme 8 cost_price = 7.416 * cost_price :=
by sorry

end NUMINAMATH_CALUDE_effective_price_for_8kg_l2976_297672


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l2976_297670

/-- Represents the number of flies eaten per day in a swamp ecosystem -/
def flies_eaten_per_day (
  frog_flies : ℕ)  -- flies eaten by one frog per day
  (fish_frogs : ℕ)  -- frogs eaten by one fish per day
  (gharial_fish : ℕ)  -- fish eaten by one gharial per day
  (heron_frogs : ℕ)  -- frogs eaten by one heron per day
  (heron_fish : ℕ)  -- fish eaten by one heron per day
  (caiman_gharials : ℕ)  -- gharials eaten by one caiman per day
  (caiman_herons : ℕ)  -- herons eaten by one caiman per day
  (num_gharials : ℕ)  -- number of gharials in the swamp
  (num_herons : ℕ)  -- number of herons in the swamp
  (num_caimans : ℕ)  -- number of caimans in the swamp
  : ℕ :=
  sorry

/-- Theorem stating the number of flies eaten per day in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_per_day 30 8 15 5 3 2 2 9 12 7 = 42840 :=
by sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l2976_297670


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l2976_297634

theorem sqrt_sum_comparison : Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l2976_297634


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l2976_297628

/-- Given three numbers x, y, and z satisfying certain conditions, prove that y equals 688/9 -/
theorem sum_and_equal_numbers (x y z : ℚ) 
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) : 
  y = 688 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l2976_297628


namespace NUMINAMATH_CALUDE_function_properties_l2976_297691

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∃ p, p > 0 ∧ ∀ x, f (x + p) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l2976_297691


namespace NUMINAMATH_CALUDE_gas_cost_equation_l2976_297649

/-- The total cost of gas for a trip satisfies the given equation based on the change in cost per person when additional friends join. -/
theorem gas_cost_equation (x : ℝ) : x > 0 → (x / 5) - (x / 8) = 15.50 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_equation_l2976_297649


namespace NUMINAMATH_CALUDE_peach_price_is_40_cents_l2976_297638

/-- Represents the store's discount policy -/
def discount_rate : ℚ := 2 / 10

/-- Represents the number of peaches bought -/
def num_peaches : ℕ := 400

/-- Represents the total amount paid after discount -/
def total_paid : ℚ := 128

/-- Calculates the price of each peach -/
def price_per_peach : ℚ :=
  let total_before_discount := total_paid / (1 - discount_rate)
  total_before_discount / num_peaches

/-- Proves that the price of each peach is $0.40 -/
theorem peach_price_is_40_cents : price_per_peach = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_peach_price_is_40_cents_l2976_297638


namespace NUMINAMATH_CALUDE_fifth_term_is_correct_l2976_297683

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
| 0 => 2*x + y
| 1 => 2*x - y
| 2 => 2*x*y
| 3 => 2*x / y
| n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the sequence is -77/10 -/
theorem fifth_term_is_correct (x y : ℚ) :
  arithmetic_sequence x y 0 = 2*x + y →
  arithmetic_sequence x y 1 = 2*x - y →
  arithmetic_sequence x y 2 = 2*x*y →
  arithmetic_sequence x y 3 = 2*x / y →
  arithmetic_sequence x y 4 = -77/10 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_is_correct_l2976_297683


namespace NUMINAMATH_CALUDE_sports_club_problem_l2976_297655

theorem sports_club_problem (total_members badminton_players tennis_players both : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 17)
  (h4 : both = 6) :
  total_members - (badminton_players + tennis_players - both) = 2 := by
sorry

end NUMINAMATH_CALUDE_sports_club_problem_l2976_297655


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2976_297692

/-- Represents a cube in 3D space -/
structure Cube :=
  (side_length : ℝ)

/-- Represents the shape described in the problem -/
structure Shape :=
  (central_cube : Cube)
  (attached_cubes : Finset Cube)

/-- The shape has 8 unit cubes -/
def shape_has_eight_cubes (s : Shape) : Prop :=
  s.attached_cubes.card = 7 ∧ s.central_cube.side_length = 1 ∧ ∀ c ∈ s.attached_cubes, c.side_length = 1

/-- The shape has cubes attached to all faces of the central cube except the bottom -/
def shape_structure (s : Shape) : Prop :=
  s.attached_cubes.card = 7

/-- Calculates the volume of the shape -/
def volume (s : Shape) : ℝ := 8

/-- Calculates the surface area of the shape -/
def surface_area (s : Shape) : ℝ := 36

/-- The main theorem: the ratio of volume to surface area is 2/9 -/
theorem volume_to_surface_area_ratio (s : Shape) 
  (h1 : shape_has_eight_cubes s) 
  (h2 : shape_structure s) : 
  (volume s) / (surface_area s) = 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2976_297692


namespace NUMINAMATH_CALUDE_brownie_pieces_l2976_297622

/-- Proves that a 24-inch by 30-inch pan can be divided into exactly 60 pieces of 3-inch by 4-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  pan_length = 24 →
  pan_width = 30 →
  piece_length = 3 →
  piece_width = 4 →
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by sorry

end NUMINAMATH_CALUDE_brownie_pieces_l2976_297622


namespace NUMINAMATH_CALUDE_shopping_spree_theorem_l2976_297666

def shopping_spree (initial_amount : ℝ) (book_price : ℝ) (num_books : ℕ) 
  (game_price : ℝ) (water_bottle_price : ℝ) (snack_price : ℝ) (num_snacks : ℕ)
  (bundle_price : ℝ) (book_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let book_total := book_price * num_books
  let discounted_book_total := book_total * (1 - book_discount)
  let subtotal := discounted_book_total + game_price + water_bottle_price + 
                  (snack_price * num_snacks) + bundle_price
  let total_with_tax := subtotal * (1 + tax_rate)
  initial_amount - total_with_tax

theorem shopping_spree_theorem :
  shopping_spree 200 12 5 45 10 3 3 20 0.1 0.12 = 45.44 := by sorry

end NUMINAMATH_CALUDE_shopping_spree_theorem_l2976_297666


namespace NUMINAMATH_CALUDE_triangle_midpoints_x_sum_l2976_297605

theorem triangle_midpoints_x_sum (p q r : ℝ) : 
  p + q + r = 15 → 
  (p + q) / 2 + (q + r) / 2 + (r + p) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoints_x_sum_l2976_297605


namespace NUMINAMATH_CALUDE_complementary_angle_of_25_41_l2976_297656

-- Define a type for angles in degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees
  , minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60
  , minutes := totalMinutes % 60 }

-- Define the given angle
def givenAngle : Angle := { degrees := 25, minutes := 41 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Theorem statement
theorem complementary_angle_of_25_41 :
  Angle.sub rightAngle givenAngle = { degrees := 64, minutes := 19 } := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_of_25_41_l2976_297656


namespace NUMINAMATH_CALUDE_arc_length_radius_l2976_297662

/-- Given an arc length and central angle, calculate the radius of the circle -/
theorem arc_length_radius (arc_length : ℝ) (central_angle : ℝ) : 
  arc_length = 4 → central_angle = 2 → arc_length = central_angle * 2 := by sorry

end NUMINAMATH_CALUDE_arc_length_radius_l2976_297662


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l2976_297653

theorem quadratic_function_m_value :
  ∃! m : ℝ, (abs (m - 1) = 2) ∧ (m - 3 ≠ 0) ∧ (m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l2976_297653


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l2976_297694

/-- The number of times Terrell lifts the weights -/
def usual_lifts : ℕ := 10

/-- The weight of each heavy weight in pounds -/
def heavy_weight : ℕ := 25

/-- The weight of each light weight in pounds -/
def light_weight : ℕ := 20

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The total weight lifted with heavy weights -/
def total_heavy_weight : ℕ := usual_lifts * heavy_weight * num_weights

/-- The number of times Terrell must lift the light weights to equal the total heavy weight -/
noncomputable def light_lifts : ℚ := total_heavy_weight / (light_weight * num_weights)

theorem terrell_weight_lifting :
  light_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l2976_297694


namespace NUMINAMATH_CALUDE_percent_relation_l2976_297663

theorem percent_relation (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2976_297663


namespace NUMINAMATH_CALUDE_second_class_sample_size_l2976_297696

/-- Calculates the number of items to be sampled from a specific class in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (classPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (classPopulation * sampleSize) / totalPopulation

theorem second_class_sample_size :
  let totalPopulation : ℕ := 200
  let secondClassPopulation : ℕ := 60
  let sampleSize : ℕ := 40
  stratifiedSampleSize totalPopulation secondClassPopulation sampleSize = 12 := by
sorry

end NUMINAMATH_CALUDE_second_class_sample_size_l2976_297696


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l2976_297686

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define the function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List ℕ) : ℚ :=
  let reciprocals := numbers.map (fun n => (1 : ℚ) / n)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l2976_297686


namespace NUMINAMATH_CALUDE_rectangle_parallelogram_relationship_l2976_297616

-- Define the types
def Parallelogram : Type := sorry
def Rectangle : Type := sorry

-- Define the relationship between Rectangle and Parallelogram
axiom rectangle_is_parallelogram : Rectangle → Parallelogram

-- State the theorem
theorem rectangle_parallelogram_relationship :
  (∀ r : Rectangle, ∃ p : Parallelogram, p = rectangle_is_parallelogram r) ∧
  ¬(∀ p : Parallelogram, ∃ r : Rectangle, p = rectangle_is_parallelogram r) :=
sorry

end NUMINAMATH_CALUDE_rectangle_parallelogram_relationship_l2976_297616


namespace NUMINAMATH_CALUDE_dividend_calculation_l2976_297626

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2976_297626


namespace NUMINAMATH_CALUDE_max_points_is_36_l2976_297687

/-- Represents a tournament with 8 teams where each team plays every other team twice -/
structure Tournament where
  num_teams : Nat
  games_per_pair : Nat
  win_points : Nat
  draw_points : Nat
  loss_points : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1) / 2) * t.games_per_pair

/-- Calculate the maximum possible points for each of the top three teams -/
def max_points_top_three (t : Tournament) : Nat :=
  let games_against_others := (t.num_teams - 3) * t.games_per_pair
  let points_against_others := games_against_others * t.win_points
  let games_among_top_three := 2 * t.games_per_pair
  let points_among_top_three := games_among_top_three * t.draw_points
  points_against_others + points_among_top_three

/-- The theorem to be proved -/
theorem max_points_is_36 (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 3)
  (h4 : t.draw_points = 1)
  (h5 : t.loss_points = 0) :
  max_points_top_three t = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_points_is_36_l2976_297687


namespace NUMINAMATH_CALUDE_power_product_equals_l2976_297643

theorem power_product_equals : 2^4 * 3^2 * 5^2 * 11 = 39600 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_l2976_297643


namespace NUMINAMATH_CALUDE_william_wins_l2976_297684

theorem william_wins (total_rounds : ℕ) (williams_advantage : ℕ) (williams_wins : ℕ) : 
  total_rounds = 15 → williams_advantage = 5 → williams_wins = 10 → 
  williams_wins = (total_rounds + williams_advantage) / 2 := by
  sorry

end NUMINAMATH_CALUDE_william_wins_l2976_297684


namespace NUMINAMATH_CALUDE_bulb_arrangement_theorem_l2976_297698

def blue_bulbs : ℕ := 7
def red_bulbs : ℕ := 7
def white_bulbs : ℕ := 12

def total_non_white_bulbs : ℕ := blue_bulbs + red_bulbs
def total_slots : ℕ := total_non_white_bulbs + 1

def arrangement_count : ℕ := Nat.choose total_non_white_bulbs blue_bulbs * Nat.choose total_slots white_bulbs

theorem bulb_arrangement_theorem :
  arrangement_count = 1561560 :=
by sorry

end NUMINAMATH_CALUDE_bulb_arrangement_theorem_l2976_297698


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2976_297661

/-- Represents the flour factory problem --/
structure FlourFactory where
  totalWorkers : ℕ
  flourPerWorker : ℕ
  noodlesPerWorker : ℕ
  flourPricePerKg : ℚ
  noodlesPricePerKg : ℚ

/-- Calculates the daily profit based on the number of workers processing noodles --/
def dailyProfit (factory : FlourFactory) (noodleWorkers : ℕ) : ℚ :=
  let flourWorkers := factory.totalWorkers - noodleWorkers
  let flourProfit := (factory.flourPerWorker * flourWorkers : ℕ) * factory.flourPricePerKg
  let noodleProfit := (factory.noodlesPerWorker * noodleWorkers : ℕ) * factory.noodlesPricePerKg
  flourProfit + noodleProfit

/-- Theorem stating the maximum profit and optimal worker allocation --/
theorem max_profit_theorem (factory : FlourFactory) 
    (h1 : factory.totalWorkers = 20)
    (h2 : factory.flourPerWorker = 600)
    (h3 : factory.noodlesPerWorker = 400)
    (h4 : factory.flourPricePerKg = 1/5)
    (h5 : factory.noodlesPricePerKg = 3/5) :
    ∃ (optimalNoodleWorkers : ℕ),
      optimalNoodleWorkers = 12 ∧ 
      dailyProfit factory optimalNoodleWorkers = 384/5 ∧
      ∀ (n : ℕ), n ≤ factory.totalWorkers → 
        dailyProfit factory n ≤ dailyProfit factory optimalNoodleWorkers :=
  sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l2976_297661


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2976_297688

/-- Prove that the vector operation (3, -8) - 3(2, -5) + (-1, 4) equals (-4, 11) -/
theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![2, -5]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  v1 - 3 • v2 + v3 = ![-4, 11] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2976_297688


namespace NUMINAMATH_CALUDE_roses_cut_l2976_297689

def initial_roses : ℕ := 6
def final_roses : ℕ := 16

theorem roses_cut (cut_roses : ℕ) : cut_roses = final_roses - initial_roses := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l2976_297689


namespace NUMINAMATH_CALUDE_garden_border_rocks_l2976_297685

theorem garden_border_rocks (rocks_placed : Float) (additional_rocks : Float) : 
  rocks_placed = 125.0 → additional_rocks = 64.0 → rocks_placed + additional_rocks = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_garden_border_rocks_l2976_297685


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2976_297641

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 - 4 * i) / i
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2976_297641


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_one_fifth_l2976_297658

theorem sqrt_meaningful_iff_x_geq_one_fifth (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 5 * x - 1) ↔ x ≥ 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_one_fifth_l2976_297658


namespace NUMINAMATH_CALUDE_log_sum_equality_fraction_sum_equality_l2976_297632

-- Part 1
theorem log_sum_equality : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 2^(Real.log 3 / Real.log 2) = 5 := by sorry

-- Part 2
theorem fraction_sum_equality : (5 + 1/16)^(1/2) + (-1)^(-1) / 0.75^(-2) + (2 + 10/27)^(-2/3) = 9/4 := by sorry

end NUMINAMATH_CALUDE_log_sum_equality_fraction_sum_equality_l2976_297632


namespace NUMINAMATH_CALUDE_problem_solution_l2976_297642

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(x / c^2) + 1
  else 0

theorem problem_solution (c : ℝ) :
  (0 < c ∧ c < 1) →
  (f c (c^2) = 9/8) →
  (c = 1/2) ∧
  (∀ x : ℝ, f (1/2) x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2976_297642


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l2976_297639

/-- Represents the daily sales and profit of mooncakes -/
structure MooncakeSales where
  initialSales : ℕ
  initialProfit : ℕ
  priceReduction : ℕ
  salesIncrease : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (s : MooncakeSales) (x : ℕ) : ℕ :=
  (s.initialProfit - x) * (s.initialSales + (s.salesIncrease * x) / s.priceReduction)

/-- Theorem stating that a 6 yuan price reduction achieves the target profit -/
theorem optimal_price_reduction (s : MooncakeSales) 
    (h1 : s.initialSales = 80)
    (h2 : s.initialProfit = 30)
    (h3 : s.priceReduction = 5)
    (h4 : s.salesIncrease = 20)
    (h5 : s.targetProfit = 2496) :
    dailyProfit s 6 = s.targetProfit := by
  sorry

#check optimal_price_reduction

end NUMINAMATH_CALUDE_optimal_price_reduction_l2976_297639


namespace NUMINAMATH_CALUDE_cookies_for_lunch_is_five_l2976_297668

/-- Calculates the number of cookies needed to reach the target calorie count for lunch -/
def cookiesForLunch (totalCalories burgerCalories carrotCalories cookieCalories : ℕ) 
                    (numCarrots : ℕ) : ℕ :=
  let remainingCalories := totalCalories - burgerCalories - (carrotCalories * numCarrots)
  remainingCalories / cookieCalories

/-- Proves that the number of cookies each kid gets is 5 -/
theorem cookies_for_lunch_is_five :
  cookiesForLunch 750 400 20 50 5 = 5 := by
  sorry

#eval cookiesForLunch 750 400 20 50 5

end NUMINAMATH_CALUDE_cookies_for_lunch_is_five_l2976_297668


namespace NUMINAMATH_CALUDE_contests_paths_l2976_297637

/-- Represents the number of choices for each step in the path, except the last --/
def choices : ℕ := 2

/-- Represents the number of starting points (number of "C"s at the base) --/
def starting_points : ℕ := 2

/-- Represents the number of steps in the path (length of "CONTESTS" - 1) --/
def path_length : ℕ := 7

theorem contests_paths :
  starting_points * (choices ^ path_length) = 256 := by
  sorry

end NUMINAMATH_CALUDE_contests_paths_l2976_297637


namespace NUMINAMATH_CALUDE_oilseed_germination_theorem_l2976_297609

/-- The average germination rate of oilseeds -/
def average_germination_rate : ℝ := 0.96

/-- The total number of oilseeds -/
def total_oilseeds : ℕ := 2000

/-- The number of oilseeds that cannot germinate -/
def non_germinating_oilseeds : ℕ := 80

/-- Theorem stating that given the average germination rate,
    approximately 80 out of 2000 oilseeds cannot germinate -/
theorem oilseed_germination_theorem :
  ⌊(1 - average_germination_rate) * total_oilseeds⌋ = non_germinating_oilseeds :=
sorry

end NUMINAMATH_CALUDE_oilseed_germination_theorem_l2976_297609


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l2976_297681

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) :
  (a > 0 ∧ a ≤ 2) ↔
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l2976_297681


namespace NUMINAMATH_CALUDE_chromosome_set_variation_l2976_297612

/-- Represents the types of chromosome number variations -/
inductive ChromosomeVariationType
| IndividualChange
| SetChange

/-- Represents the form of chromosome changes -/
inductive ChromosomeChangeForm
| Individual
| Set

/-- Definition of chromosome number variation -/
structure ChromosomeVariation where
  type : ChromosomeVariationType
  form : ChromosomeChangeForm

/-- Theorem stating that one type of chromosome number variation involves
    doubling or halving of chromosomes in the form of chromosome sets -/
theorem chromosome_set_variation :
  ∃ (cv : ChromosomeVariation),
    cv.type = ChromosomeVariationType.SetChange ∧
    cv.form = ChromosomeChangeForm.Set :=
sorry

end NUMINAMATH_CALUDE_chromosome_set_variation_l2976_297612


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2976_297697

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ (4 - Real.sqrt 19) / 3 < x ∧ x < (4 + Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2976_297697


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l2976_297679

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 20)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), a * b = side_area ∧ b * c = front_area ∧ c * a = bottom_area ∧ a * b * c = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l2976_297679


namespace NUMINAMATH_CALUDE_quiz_show_win_probability_l2976_297601

def num_questions : ℕ := 4
def num_options : ℕ := 3
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_options

/-- The probability of winning the quiz show by answering at least 3 out of 4 questions correctly,
    where each question has 3 options and guesses are random. -/
theorem quiz_show_win_probability :
  (Finset.sum (Finset.range (num_questions - min_correct + 1))
    (fun k => (Nat.choose num_questions (num_questions - k)) *
              (probability_correct_guess ^ (num_questions - k)) *
              ((1 - probability_correct_guess) ^ k))) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quiz_show_win_probability_l2976_297601


namespace NUMINAMATH_CALUDE_no_integer_solution_l2976_297693

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 + 1974 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2976_297693


namespace NUMINAMATH_CALUDE_blue_part_length_l2976_297678

/-- Proves that the blue part of a pencil is 3.5 cm long given specific conditions -/
theorem blue_part_length (total_length : ℝ) (black_ratio : ℝ) (white_ratio : ℝ)
  (h1 : total_length = 8)
  (h2 : black_ratio = 1 / 8)
  (h3 : white_ratio = 1 / 2)
  (h4 : black_ratio * total_length + white_ratio * (total_length - black_ratio * total_length) +
    (total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length)) = total_length) :
  total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_part_length_l2976_297678


namespace NUMINAMATH_CALUDE_batsman_average_l2976_297608

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  inningScore : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: Given the conditions, the batsman's new average is 55 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.inningScore = 95)
  (h2 : b.averageIncrease = 2.5)
  : newAverage b = 55 := by
  sorry

#eval newAverage { initialAverage := 52.5, inningScore := 95, averageIncrease := 2.5 }

end NUMINAMATH_CALUDE_batsman_average_l2976_297608


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2976_297631

theorem simplify_square_roots : 16^(1/2) - 625^(1/2) = -21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2976_297631


namespace NUMINAMATH_CALUDE_inverse_variation_doubling_inverse_variation_example_l2976_297646

/-- Given two quantities that vary inversely, if one quantity doubles, the other halves -/
theorem inverse_variation_doubling (a b c d : ℝ) (h1 : a * b = c * d) (h2 : c = 2 * a) :
  d = b / 2 := by
  sorry

/-- When a and b vary inversely, if b = 0.5 when a = 800, then b = 0.25 when a = 1600 -/
theorem inverse_variation_example :
  ∃ (k : ℝ), (800 * 0.5 = k) ∧ (1600 * 0.25 = k) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_doubling_inverse_variation_example_l2976_297646


namespace NUMINAMATH_CALUDE_alan_wings_increase_l2976_297630

/-- Proves that Alan needs to increase his rate by 4 wings per minute to beat Kevin's record -/
theorem alan_wings_increase (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) : 
  kevin_wings = 64 → 
  kevin_time = 8 → 
  alan_rate = 5 → 
  (kevin_wings / kevin_time : ℚ) - alan_rate = 4 := by
  sorry

#check alan_wings_increase

end NUMINAMATH_CALUDE_alan_wings_increase_l2976_297630
