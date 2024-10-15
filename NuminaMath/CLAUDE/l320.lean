import Mathlib

namespace NUMINAMATH_CALUDE_festival_end_day_l320_32011

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that 45 days after a 5-day festival starting on Tuesday is Wednesday -/
theorem festival_end_day (startDay : DayOfWeek) 
  (h : startDay = DayOfWeek.Tuesday) : 
  advanceDays startDay (5 + 45) = DayOfWeek.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_festival_end_day_l320_32011


namespace NUMINAMATH_CALUDE_reimbursement_calculation_l320_32035

/-- Calculates the total reimbursement for a sales rep based on daily mileage -/
def total_reimbursement (rate : ℚ) (miles : List ℚ) : ℚ :=
  (miles.map (· * rate)).sum

/-- Proves that the total reimbursement for the given mileage and rate is $36.00 -/
theorem reimbursement_calculation : 
  let rate : ℚ := 36 / 100
  let daily_miles : List ℚ := [18, 26, 20, 20, 16]
  total_reimbursement rate daily_miles = 36 := by
  sorry

#eval total_reimbursement (36 / 100) [18, 26, 20, 20, 16]

end NUMINAMATH_CALUDE_reimbursement_calculation_l320_32035


namespace NUMINAMATH_CALUDE_percentage_relation_l320_32056

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) : 
  x = 0.65 * z := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l320_32056


namespace NUMINAMATH_CALUDE_modified_geometric_structure_pieces_l320_32016

/-- Calculates the sum of an arithmetic progression -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of pieces in the modified geometric structure -/
theorem modified_geometric_structure_pieces :
  let num_rows : ℕ := 10
  let first_rod_count : ℕ := 3
  let rod_difference : ℕ := 3
  let connector_rows : ℕ := num_rows + 1
  let rod_count := arithmetic_sum first_rod_count rod_difference num_rows
  let connector_count := triangular_number connector_rows
  rod_count + connector_count = 231 := by
  sorry

end NUMINAMATH_CALUDE_modified_geometric_structure_pieces_l320_32016


namespace NUMINAMATH_CALUDE_smallest_number_of_points_l320_32078

/-- The length of the circle -/
def circleLength : ℕ := 1956

/-- The distance between adjacent points in the sequence -/
def distanceStep : ℕ := 3

/-- The number of points required -/
def numPoints : ℕ := 2 * (circleLength / distanceStep)

/-- Theorem stating the smallest number of points satisfying the conditions -/
theorem smallest_number_of_points :
  numPoints = 1304 ∧
  ∀ n : ℕ, n < numPoints →
    ¬(∀ i : Fin n,
      ∃! j : Fin n, i ≠ j ∧ (circleLength * (i.val - j.val : ℤ) / n).natAbs % circleLength = 1 ∧
      ∃! k : Fin n, i ≠ k ∧ (circleLength * (i.val - k.val : ℤ) / n).natAbs % circleLength = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_points_l320_32078


namespace NUMINAMATH_CALUDE_cube_root_property_l320_32050

theorem cube_root_property (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_property_l320_32050


namespace NUMINAMATH_CALUDE_common_difference_is_two_l320_32038

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l320_32038


namespace NUMINAMATH_CALUDE_machines_for_hundred_books_l320_32080

/-- The number of printing machines required to print a given number of books in a given number of days. -/
def machines_required (initial_machines : ℕ) (initial_books : ℕ) (initial_days : ℕ) 
                      (target_books : ℕ) (target_days : ℕ) : ℕ :=
  (target_books * initial_machines * initial_days) / (initial_books * target_days)

/-- Theorem stating that 5 machines are required to print 100 books in 100 days,
    given that 5 machines can print 5 books in 5 days. -/
theorem machines_for_hundred_books : 
  machines_required 5 5 5 100 100 = 5 := by
  sorry

#eval machines_required 5 5 5 100 100

end NUMINAMATH_CALUDE_machines_for_hundred_books_l320_32080


namespace NUMINAMATH_CALUDE_close_numbers_properties_l320_32032

/-- A set of close numbers -/
structure CloseNumbers where
  n : ℕ
  numbers : Fin n → ℝ
  sum : ℝ
  n_gt_one : n > 1
  close : ∀ i, numbers i < sum / (n - 1)

/-- Theorems about close numbers -/
theorem close_numbers_properties (cn : CloseNumbers) :
  (∀ i, cn.numbers i > 0) ∧
  (∀ i j k, cn.numbers i + cn.numbers j > cn.numbers k) ∧
  (∀ i j, cn.numbers i + cn.numbers j > cn.sum / (cn.n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_close_numbers_properties_l320_32032


namespace NUMINAMATH_CALUDE_locus_equation_l320_32023

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₂ with equation (x - 3)² + y² = 81 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 81}

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3)^2 + center.2^2 = (9 - radius)^2

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₂ -/
def locus : Set (ℝ × ℝ) :=
  {p | ∃ r : ℝ, externally_tangent_C₁ p r ∧ internally_tangent_C₂ p r}

theorem locus_equation : locus = {p : ℝ × ℝ | 12 * p.1^2 + 169 * p.2^2 - 36 * p.1 - 1584 = 0} := by
  sorry

end NUMINAMATH_CALUDE_locus_equation_l320_32023


namespace NUMINAMATH_CALUDE_salary_comparison_l320_32053

theorem salary_comparison (raja_salary : ℝ) (ram_salary : ℝ) 
  (h : ram_salary = raja_salary * 1.25) : 
  (raja_salary / ram_salary) = 0.8 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l320_32053


namespace NUMINAMATH_CALUDE_min_tablets_extraction_l320_32072

/-- Represents the number of tablets for each medicine type -/
structure MedicineCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Represents the minimum number of tablets required for each medicine type -/
structure RequiredCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the minimum number of tablets to be extracted -/
def minTablets (total : MedicineCount) (required : RequiredCount) : Nat :=
  sorry

theorem min_tablets_extraction (total : MedicineCount) (required : RequiredCount) :
  total.A = 10 →
  total.B = 14 →
  total.C = 18 →
  total.D = 20 →
  required.A = 3 →
  required.B = 4 →
  required.C = 3 →
  required.D = 2 →
  minTablets total required = 55 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extraction_l320_32072


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l320_32066

variable (x : ℝ)

theorem average_of_five_numbers (x : ℝ) :
  let numbers := [-4*x, 0, 4*x, 12*x, 20*x]
  (numbers.sum / numbers.length : ℝ) = 6.4 * x :=
by sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l320_32066


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l320_32005

/-- Given three points on the graph of y = -4/x, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -4 / x₁)
  (h2 : y₂ = -4 / x₂)
  (h3 : y₃ = -4 / x₃)
  (hx : x₁ < 0 ∧ 0 < x₂ ∧ x₂ < x₃) :
  y₁ > y₃ ∧ y₃ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l320_32005


namespace NUMINAMATH_CALUDE_silva_family_zoo_cost_l320_32092

/-- Calculates the total cost of zoo tickets for a family group -/
def total_zoo_cost (senior_ticket_cost : ℚ) (child_discount : ℚ) (senior_discount : ℚ) : ℚ :=
  let full_price := senior_ticket_cost / (1 - senior_discount)
  let child_price := full_price * (1 - child_discount)
  3 * senior_ticket_cost + 3 * full_price + 3 * child_price

/-- Theorem stating the total cost for the Silva family zoo trip -/
theorem silva_family_zoo_cost :
  total_zoo_cost 7 (4/10) (3/10) = 69 := by
  sorry

#eval total_zoo_cost 7 (4/10) (3/10)

end NUMINAMATH_CALUDE_silva_family_zoo_cost_l320_32092


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l320_32009

theorem binomial_coefficient_ratio (n : ℕ) (k : ℕ) : 
  n = 14 ∧ k = 4 →
  (Nat.choose n k = 1001 ∧ 
   Nat.choose n (k+1) = 2002 ∧ 
   Nat.choose n (k+2) = 3003) ∧
  ∀ m : ℕ, m > 3 → 
    ¬(∃ j : ℕ, ∀ i : ℕ, i < m → 
      Nat.choose n (j+i+1) = (i+1) * Nat.choose n j) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l320_32009


namespace NUMINAMATH_CALUDE_probability_even_distinct_digits_l320_32086

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

def count_favorable_outcomes : ℕ := 7 * 8 * 7 * 5

theorem probability_even_distinct_digits :
  (count_favorable_outcomes : ℚ) / (9999 - 2000 + 1 : ℚ) = 49 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_distinct_digits_l320_32086


namespace NUMINAMATH_CALUDE_complex_additive_inverse_l320_32019

theorem complex_additive_inverse (b : ℝ) : 
  let z : ℂ := (4 + b * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_additive_inverse_l320_32019


namespace NUMINAMATH_CALUDE_lab_budget_theorem_l320_32012

def lab_budget_problem (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
  (min_instruments : ℕ) : Prop :=
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost + chemical_cost + min_instrument_cost
  total_budget = 750 ∧
  flask_cost = 200 ∧
  test_tube_cost = 2/3 * flask_cost ∧
  safety_gear_cost = 1/2 * test_tube_cost ∧
  chemical_cost = 3/4 * flask_cost ∧
  min_instrument_cost ≥ 50 ∧
  min_instruments ≥ 10 ∧
  total_budget - total_spent = 150

theorem lab_budget_theorem :
  ∃ (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
    (min_instruments : ℕ),
  lab_budget_problem total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost min_instruments :=
by
  sorry

end NUMINAMATH_CALUDE_lab_budget_theorem_l320_32012


namespace NUMINAMATH_CALUDE_symmetric_point_about_line_l320_32095

/-- The symmetric point of (x₁, y₁) about the line ax + by + c = 0 is (x₂, y₂) -/
def is_symmetric_point (x₁ y₁ x₂ y₂ a b c : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) * a = -(x₂ - x₁) * b ∧
  -- The midpoint of the two points lies on the line of symmetry
  (a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0)

theorem symmetric_point_about_line :
  is_symmetric_point (-1) 2 (-6) (-3) 1 1 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_about_line_l320_32095


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l320_32063

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) 
  (h2 : x - y = 1 / 101) : 
  x^2 - y^2 = 5 / 1111 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l320_32063


namespace NUMINAMATH_CALUDE_width_length_ratio_l320_32071

/-- A rectangle with given length and perimeter -/
structure Rectangle where
  length : ℝ
  perimeter : ℝ
  width : ℝ
  length_pos : length > 0
  perimeter_pos : perimeter > 0
  width_pos : width > 0
  perimeter_eq : perimeter = 2 * (length + width)

/-- The ratio of width to length for a rectangle with length 10 and perimeter 30 is 1:2 -/
theorem width_length_ratio (rect : Rectangle) 
    (h1 : rect.length = 10) 
    (h2 : rect.perimeter = 30) : 
    rect.width / rect.length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_width_length_ratio_l320_32071


namespace NUMINAMATH_CALUDE_three_values_of_sum_l320_32089

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the property that both domain and range are [a, b]
def domain_range_equal (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → a ≤ f x ∧ f x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem stating that there are exactly 3 different values of a+b
theorem three_values_of_sum :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x, x ∈ s ↔ ∃ a b, domain_range_equal a b ∧ a + b = x) :=
sorry

end NUMINAMATH_CALUDE_three_values_of_sum_l320_32089


namespace NUMINAMATH_CALUDE_function_characterization_l320_32061

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, f (a * b) = f a + f b - f (Nat.gcd a b)) ∧
  (∀ p a : ℕ, Nat.Prime p → (f a ≥ f (a * p) → f a + f p ≥ f a * f p + 1))

theorem function_characterization (f : ℕ → ℕ) (h : is_valid_function f) :
  (∀ n : ℕ, f n = n) ∨ (∀ n : ℕ, f n = 1) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l320_32061


namespace NUMINAMATH_CALUDE_max_amount_C_is_correct_l320_32040

/-- Represents the maximum amount of 11% saline solution (C) that can be used
    to prepare 100 kg of 7% saline solution, given 3% (A) and 8% (B) solutions
    are also available. -/
def maxAmountC : ℝ := 50

/-- The concentration of saline solution A -/
def concentrationA : ℝ := 0.03

/-- The concentration of saline solution B -/
def concentrationB : ℝ := 0.08

/-- The concentration of saline solution C -/
def concentrationC : ℝ := 0.11

/-- The target concentration of the final solution -/
def targetConcentration : ℝ := 0.07

/-- The total amount of the final solution -/
def totalAmount : ℝ := 100

theorem max_amount_C_is_correct :
  ∃ (y : ℝ),
    0 ≤ y ∧
    0 ≤ (totalAmount - maxAmountC - y) ∧
    concentrationC * maxAmountC + concentrationB * y +
      concentrationA * (totalAmount - maxAmountC - y) =
    targetConcentration * totalAmount ∧
    ∀ (x : ℝ),
      x > maxAmountC →
      ¬∃ (z : ℝ),
        0 ≤ z ∧
        0 ≤ (totalAmount - x - z) ∧
        concentrationC * x + concentrationB * z +
          concentrationA * (totalAmount - x - z) =
        targetConcentration * totalAmount :=
by sorry

end NUMINAMATH_CALUDE_max_amount_C_is_correct_l320_32040


namespace NUMINAMATH_CALUDE_divisible_by_six_l320_32047

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n^3 + 5*n = 6*k := by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l320_32047


namespace NUMINAMATH_CALUDE_dandelion_survival_l320_32057

/-- The number of seeds produced by each dandelion -/
def seeds_per_dandelion : ℕ := 300

/-- The fraction of seeds that land in water and die -/
def water_death_fraction : ℚ := 1/3

/-- The fraction of starting seeds eaten by insects -/
def insect_eaten_fraction : ℚ := 1/6

/-- The fraction of remaining seeds that sprout and are immediately eaten -/
def sprout_eaten_fraction : ℚ := 1/2

/-- The number of dandelions that survive long enough to flower -/
def surviving_dandelions : ℕ := 75

theorem dandelion_survival :
  (seeds_per_dandelion : ℚ) * (1 - water_death_fraction) * (1 - insect_eaten_fraction) * (1 - sprout_eaten_fraction) = surviving_dandelions := by
  sorry

end NUMINAMATH_CALUDE_dandelion_survival_l320_32057


namespace NUMINAMATH_CALUDE_number_ordering_eight_ten_equals_four_fifteen_l320_32082

theorem number_ordering : 8^10 < 3^20 ∧ 3^20 < 4^15 := by
  sorry

-- Additional theorem to establish the given condition
theorem eight_ten_equals_four_fifteen : 8^10 = 4^15 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_eight_ten_equals_four_fifteen_l320_32082


namespace NUMINAMATH_CALUDE_unique_prime_generating_number_l320_32029

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_generating_number :
  ∃! n : ℕ, n > 0 ∧ is_prime (n^n + 1) ∧ is_prime ((2*n)^(2*n) + 1) ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_generating_number_l320_32029


namespace NUMINAMATH_CALUDE_age_problem_l320_32052

theorem age_problem (a b : ℕ) : 
  (a : ℚ) / b = 5 / 3 →
  ((a + 2) : ℚ) / (b + 2) = 3 / 2 →
  b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l320_32052


namespace NUMINAMATH_CALUDE_brick_surface_area_l320_32045

theorem brick_surface_area :
  let length : ℝ := 8
  let width : ℝ := 4
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 112 :=
by sorry

end NUMINAMATH_CALUDE_brick_surface_area_l320_32045


namespace NUMINAMATH_CALUDE_sequence_non_positive_l320_32004

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2 * a k + a (k+1) ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l320_32004


namespace NUMINAMATH_CALUDE_paper_cranes_problem_l320_32074

/-- The number of paper cranes folded by student A -/
def cranes_A (x : ℤ) : ℤ := 3 * x - 100

/-- The number of paper cranes folded by student C -/
def cranes_C (x : ℤ) : ℤ := cranes_A x - 67

theorem paper_cranes_problem (x : ℤ) 
  (h1 : cranes_A x + x + cranes_C x = 1000) : 
  cranes_A x = 443 := by
  sorry

end NUMINAMATH_CALUDE_paper_cranes_problem_l320_32074


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l320_32064

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ x = 1 + I) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l320_32064


namespace NUMINAMATH_CALUDE_distance_difference_l320_32043

def sprint_distance : ℝ := 0.88
def jog_distance : ℝ := 0.75

theorem distance_difference : sprint_distance - jog_distance = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l320_32043


namespace NUMINAMATH_CALUDE_problem_solution_l320_32027

def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

theorem problem_solution :
  (∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, f x ≥ m) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = m →
      1 / (a + b) + 1 / (b + c) + 1 / (a + c) ≥ 9 / (2 * m))) ∧
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l320_32027


namespace NUMINAMATH_CALUDE_star_operation_result_l320_32001

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l320_32001


namespace NUMINAMATH_CALUDE_roses_in_vase_l320_32083

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 6

/-- The number of roses Mary added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses in the vase after Mary added more -/
def total_roses : ℕ := initial_roses + added_roses

theorem roses_in_vase : total_roses = 22 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l320_32083


namespace NUMINAMATH_CALUDE_different_color_picks_count_l320_32051

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Colorless

/-- Represents the deck composition -/
structure Deck :=
  (red_cards : Nat)
  (black_cards : Nat)
  (jokers : Nat)

/-- The number of ways to pick two different cards of different colors -/
def different_color_picks (d : Deck) : Nat :=
  -- Red-Black or Black-Red
  2 * d.red_cards * d.black_cards +
  -- Colorless-Red or Colorless-Black
  2 * d.jokers * (d.red_cards + d.black_cards) +
  -- Red-Colorless or Black-Colorless
  2 * (d.red_cards + d.black_cards) * d.jokers

/-- The theorem to be proved -/
theorem different_color_picks_count :
  let d : Deck := { red_cards := 26, black_cards := 26, jokers := 2 }
  different_color_picks d = 1508 := by
  sorry

end NUMINAMATH_CALUDE_different_color_picks_count_l320_32051


namespace NUMINAMATH_CALUDE_james_oranges_l320_32048

theorem james_oranges :
  ∀ (o : ℕ),
    o ≤ 7 →
    (∃ (a : ℕ), a + o = 7 ∧ (65 * o + 40 * a) % 100 = 0) →
    o = 4 :=
by sorry

end NUMINAMATH_CALUDE_james_oranges_l320_32048


namespace NUMINAMATH_CALUDE_average_youtube_viewer_videos_l320_32094

theorem average_youtube_viewer_videos (video_length : ℕ) (ad_time : ℕ) (total_time : ℕ) :
  video_length = 7 →
  ad_time = 3 →
  total_time = 17 →
  ∃ (num_videos : ℕ), num_videos * video_length + ad_time = total_time ∧ num_videos = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_youtube_viewer_videos_l320_32094


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l320_32026

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 → 
  x = 3 * (y + z) → 
  y = 6 * z → 
  x * y * z = 7762.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l320_32026


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l320_32077

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal and not undefined -/
def parallel (l1 l2 : Line) : Prop :=
  l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b

theorem parallel_lines_a_value :
  ∃ (a : ℝ), parallel (Line.mk 2 a (-2)) (Line.mk a (a + 4) (-4)) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l320_32077


namespace NUMINAMATH_CALUDE_second_player_prevents_complete_2x2_l320_32091

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of a square (colored by first player, second player, or uncolored) -/
inductive SquareState
  | FirstPlayer
  | SecondPlayer
  | Uncolored

/-- Represents the game state -/
def GameState := Square → SquareState

/-- Represents a 2x2 square on the board -/
structure Square2x2 where
  topLeft : Square

/-- The strategy function for the second player -/
def secondPlayerStrategy (gs : GameState) (lastMove : Square) : Square := sorry

/-- Checks if a 2x2 square is completely colored by the first player -/
def isComplete2x2FirstPlayer (gs : GameState) (s : Square2x2) : Bool := sorry

/-- The main theorem stating that the second player can always prevent
    the first player from coloring any 2x2 square completely -/
theorem second_player_prevents_complete_2x2 :
  ∀ (numMoves : Nat) (gs : GameState),
    (∀ (s : Square), gs s = SquareState.Uncolored) →
    ∀ (moves : Fin numMoves → Square),
      let finalState := sorry  -- Final game state after all moves
      ∀ (s : Square2x2), ¬(isComplete2x2FirstPlayer finalState s) :=
sorry

end NUMINAMATH_CALUDE_second_player_prevents_complete_2x2_l320_32091


namespace NUMINAMATH_CALUDE_min_value_of_sum_l320_32042

theorem min_value_of_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l320_32042


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l320_32081

/-- Represents a domino tile with two numbers -/
structure Domino :=
  (upper : Nat)
  (lower : Nat)
  (upper_bound : upper ≤ 6)
  (lower_bound : lower ≤ 6)

/-- The set of all possible domino tiles -/
def dominoSet : Finset Domino := sorry

/-- The game state, including the numbers written on the blackboard and remaining tiles -/
structure GameState :=
  (written : Finset Nat)
  (remaining : Finset Domino)

/-- A player's strategy for selecting a domino -/
def Strategy := GameState → Option Domino

/-- Determines if a strategy is winning for the second player -/
def isWinningStrategy (s : Strategy) : Prop := sorry

/-- The main theorem stating that there exists a winning strategy for the second player -/
theorem second_player_winning_strategy :
  ∃ (s : Strategy), isWinningStrategy s := by sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l320_32081


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l320_32010

theorem surrounding_circles_radius (r : ℝ) : r = 4 := by
  -- Given a central circle of radius 2
  -- Surrounded by 4 circles of radius r
  -- The surrounding circles touch the central circle and each other
  -- We need to prove that r = 4
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l320_32010


namespace NUMINAMATH_CALUDE_urn_probability_l320_32046

theorem urn_probability (N : ℚ) : 
  let urn1_green : ℚ := 5
  let urn1_blue : ℚ := 7
  let urn2_green : ℚ := 20
  let urn2_blue : ℚ := N
  let total_probability : ℚ := 65/100
  (urn1_green / (urn1_green + urn1_blue)) * (urn2_green / (urn2_green + urn2_blue)) +
  (urn1_blue / (urn1_green + urn1_blue)) * (urn2_blue / (urn2_green + urn2_blue)) = total_probability →
  N = 280/311 := by
sorry

end NUMINAMATH_CALUDE_urn_probability_l320_32046


namespace NUMINAMATH_CALUDE_triangle_area_l320_32022

/-- Given a triangle ABC where BC = 12 cm, AC = 5 cm, and the angle between BC and AC is 30°,
    prove that the area of the triangle is 15 square centimeters. -/
theorem triangle_area (BC AC : ℝ) (angle : Real) (h : BC = 12 ∧ AC = 5 ∧ angle = 30 * Real.pi / 180) :
  (1 / 2 : ℝ) * BC * (AC * Real.sin angle) = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l320_32022


namespace NUMINAMATH_CALUDE_archibald_win_percentage_l320_32037

theorem archibald_win_percentage (archibald_wins brother_wins : ℕ) : 
  archibald_wins = 12 → brother_wins = 18 → 
  (archibald_wins : ℚ) / (archibald_wins + brother_wins : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_archibald_win_percentage_l320_32037


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l320_32006

theorem salary_increase_percentage (S : ℝ) : 
  S * 1.1 = 770.0000000000001 → 
  S * (1 + 16 / 100) = 812 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l320_32006


namespace NUMINAMATH_CALUDE_solve_equation_l320_32073

theorem solve_equation (y : ℝ) : (7 - y = 4) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l320_32073


namespace NUMINAMATH_CALUDE_apple_pie_calculation_l320_32003

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : apples_per_pie = 4) :
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_calculation_l320_32003


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l320_32025

/-- 
An arithmetic sequence is a sequence where the difference between 
consecutive terms is constant.
-/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- 
Theorem: In an arithmetic sequence where the sum of the first and fifth terms is 10, 
the third term is equal to 5.
-/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 5 = 10) : 
  a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l320_32025


namespace NUMINAMATH_CALUDE_sports_club_membership_l320_32036

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l320_32036


namespace NUMINAMATH_CALUDE_find_m_value_l320_32087

theorem find_m_value (x y m : ℝ) 
  (eq1 : 3 * x + 7 * y = 5 * m - 3)
  (eq2 : 2 * x + 3 * y = 8)
  (eq3 : x + 2 * y = 5) : 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l320_32087


namespace NUMINAMATH_CALUDE_yoongi_flowers_l320_32041

def flowers_problem (initial : ℕ) (to_eunji : ℕ) (to_yuna : ℕ) : Prop :=
  initial - (to_eunji + to_yuna) = 12

theorem yoongi_flowers : flowers_problem 28 7 9 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_flowers_l320_32041


namespace NUMINAMATH_CALUDE_square_root_sum_l320_32096

theorem square_root_sum (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l320_32096


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l320_32085

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l320_32085


namespace NUMINAMATH_CALUDE_speed_in_still_water_l320_32044

/-- The speed of a man rowing a boat in still water, given his downstream speed and the speed of the current. -/
theorem speed_in_still_water 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_speed = 17.9997120913593) 
  (h2 : current_speed = 3) : 
  downstream_speed - current_speed = 14.9997120913593 := by
  sorry

#eval (17.9997120913593 : Float) - 3

end NUMINAMATH_CALUDE_speed_in_still_water_l320_32044


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l320_32076

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := π * r₁^2  -- area of smaller circle
  let A₂ : ℝ := π * r₂^2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8
  := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l320_32076


namespace NUMINAMATH_CALUDE_jenny_money_problem_l320_32062

theorem jenny_money_problem (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
  sorry

end NUMINAMATH_CALUDE_jenny_money_problem_l320_32062


namespace NUMINAMATH_CALUDE_divisor_property_l320_32054

theorem divisor_property (x y : ℕ) (h1 : x % 63 = 11) (h2 : x % y = 2) :
  ∃ (k : ℕ), y ∣ (63 * k + 9) := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l320_32054


namespace NUMINAMATH_CALUDE_rectangle_area_with_perimeter_and_breadth_l320_32075

/-- Theorem: Area of a rectangle with given perimeter and breadth -/
theorem rectangle_area_with_perimeter_and_breadth
  (perimeter : ℝ) (breadth : ℝ) (h_perimeter : perimeter = 900)
  (h_breadth : breadth = 190) :
  let length : ℝ := perimeter / 2 - breadth
  let area : ℝ := length * breadth
  area = 49400 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_perimeter_and_breadth_l320_32075


namespace NUMINAMATH_CALUDE_square_difference_equals_double_product_problem_instance_l320_32007

theorem square_difference_equals_double_product (a b : ℕ) :
  (a + b)^2 - (a^2 + b^2) = 2 * a * b :=
by sorry

-- Specific instance for the given problem
theorem problem_instance : (25 + 15)^2 - (25^2 + 15^2) = 750 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_equals_double_product_problem_instance_l320_32007


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_4sqrt2_l320_32099

theorem sqrt_difference_equals_4sqrt2 :
  Real.sqrt (5 + 6 * Real.sqrt 2) - Real.sqrt (5 - 6 * Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_4sqrt2_l320_32099


namespace NUMINAMATH_CALUDE_solution_set_characterization_l320_32098

-- Define the set M
def M (a : ℝ) := {x : ℝ | x^2 + (a-4)*x - (a+1)*(2*a-3) < 0}

-- State the theorem
theorem solution_set_characterization (a : ℝ) :
  (0 ∈ M a) →
  ((a < -1 ∨ a > 3/2) ∧
   (a < -1 → M a = Set.Ioo (a+1) (3-2*a)) ∧
   (a > 3/2 → M a = Set.Ioo (3-2*a) (a+1))) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l320_32098


namespace NUMINAMATH_CALUDE_intersection_P_Q_l320_32093

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | ∃ y, Real.log (2 - x) = y}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l320_32093


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l320_32058

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: For n pairs of distinct socks arranged randomly, 
    the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : 
  expected_socks n = 2 * n := by sorry

end NUMINAMATH_CALUDE_expected_socks_theorem_l320_32058


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l320_32067

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l320_32067


namespace NUMINAMATH_CALUDE_max_candies_eaten_l320_32069

theorem max_candies_eaten (n : ℕ) (h : n = 28) : 
  (n * (n - 1)) / 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l320_32069


namespace NUMINAMATH_CALUDE_total_students_in_halls_l320_32028

theorem total_students_in_halls (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 →
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 := by
sorry

end NUMINAMATH_CALUDE_total_students_in_halls_l320_32028


namespace NUMINAMATH_CALUDE_expression_factorization_l320_32090

theorem expression_factorization (x y z : ℝ) :
  29.52 * x^2 * y - y^2 * z + z^2 * x - x^2 * z + y^2 * x + z^2 * y - 2 * x * y * z =
  (y - z) * (x + y) * (x - z) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l320_32090


namespace NUMINAMATH_CALUDE_binomial_30_3_l320_32049

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l320_32049


namespace NUMINAMATH_CALUDE_gcd_inequality_l320_32000

theorem gcd_inequality (n d₁ d₂ : ℕ+) : 
  (Nat.gcd n (d₁ + d₂) : ℚ) / (Nat.gcd n d₁ * Nat.gcd n d₂) ≥ 1 / n.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_inequality_l320_32000


namespace NUMINAMATH_CALUDE_intersection_sum_l320_32065

theorem intersection_sum (p q : ℝ) : 
  let M := {x : ℝ | x^2 - 5*x < 0}
  let N := {x : ℝ | p < x ∧ x < 6}
  ({x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 2 < x ∧ x < q}) → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l320_32065


namespace NUMINAMATH_CALUDE_even_function_domain_l320_32033

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def Domain (a : ℝ) : Set ℝ := {x : ℝ | |x + 2 - a| < a}

-- Theorem statement
theorem even_function_domain (f : ℝ → ℝ) (a : ℝ) 
  (h_even : EvenFunction f) 
  (h_domain : Set.range f = Domain a) 
  (h_positive : a > 0) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_domain_l320_32033


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l320_32060

/-- Given the prices of fruits satisfying certain conditions, prove the cost of a specific combination. -/
theorem fruit_cost_theorem (x y z : ℝ) 
  (h1 : 2 * x + y + 4 * z = 6) 
  (h2 : 4 * x + 2 * y + 2 * z = 4) : 
  4 * x + 2 * y + 5 * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l320_32060


namespace NUMINAMATH_CALUDE_candy_container_volume_l320_32014

theorem candy_container_volume (a b c : ℕ) (h : a * b * c = 216) :
  (3 * a) * (2 * b) * (4 * c) = 5184 := by
  sorry

end NUMINAMATH_CALUDE_candy_container_volume_l320_32014


namespace NUMINAMATH_CALUDE_functions_equal_at_three_l320_32055

open Set

-- Define the open interval (2, 4)
def OpenInterval : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the properties of functions f and g
def SatisfiesConditions (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ OpenInterval,
    (2 < f x ∧ f x < 4) ∧
    (2 < g x ∧ g x < 4) ∧
    (f (g x) = x) ∧
    (g (f x) = x) ∧
    (f x * g x = x^2)

-- Theorem statement
theorem functions_equal_at_three
  (f g : ℝ → ℝ)
  (h : SatisfiesConditions f g) :
  f 3 = g 3 := by
  sorry

end NUMINAMATH_CALUDE_functions_equal_at_three_l320_32055


namespace NUMINAMATH_CALUDE_bounded_expression_l320_32084

theorem bounded_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_bounded_expression_l320_32084


namespace NUMINAMATH_CALUDE_perpendicular_slope_l320_32021

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let m₁ := a / b
  let m₂ := -1 / m₁
  (a * x - b * y = c) → (m₂ = -b / a) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l320_32021


namespace NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l320_32015

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - (2*n-21)^2 + (2*n-23)^2 = 
  4*n^2 + 1 :=
by sorry

theorem specific_square_sum_difference : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l320_32015


namespace NUMINAMATH_CALUDE_miriam_flower_care_l320_32024

/-- Calculates the number of flowers Miriam can take care of in a given number of days -/
def flowers_cared_for (hours_per_day : ℕ) (flowers_per_day : ℕ) (num_days : ℕ) : ℕ :=
  (hours_per_day * num_days) * (flowers_per_day / hours_per_day)

/-- Proves that Miriam can take care of 360 flowers in 6 days -/
theorem miriam_flower_care :
  flowers_cared_for 5 60 6 = 360 := by
  sorry

#eval flowers_cared_for 5 60 6

end NUMINAMATH_CALUDE_miriam_flower_care_l320_32024


namespace NUMINAMATH_CALUDE_textbook_packing_probability_l320_32097

/-- Represents the problem of packing textbooks into boxes -/
structure TextbookPacking where
  total_books : Nat
  math_books : Nat
  box_sizes : Finset Nat

/-- The probability of all math books ending up in the same box -/
def probability_all_math_in_same_box (p : TextbookPacking) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given problem -/
theorem textbook_packing_probability :
  let p := TextbookPacking.mk 15 4 {4, 5, 6}
  probability_all_math_in_same_box p = 27 / 1759 :=
sorry

end NUMINAMATH_CALUDE_textbook_packing_probability_l320_32097


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l320_32070

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l320_32070


namespace NUMINAMATH_CALUDE_age_difference_is_zero_l320_32002

/-- Given that Carlos and David were born on the same day in different years,
    prove that the age difference between them is 0 years. -/
theorem age_difference_is_zero (C D m : ℕ) : 
  C = D + m →
  C - 1 = 6 * (D - 1) →
  C = D^3 →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_zero_l320_32002


namespace NUMINAMATH_CALUDE_average_speed_calculation_l320_32039

/-- Given a distance of 10000 meters and a time of 28 minutes, 
    prove that the average speed is approximately 595.24 cm/s. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) : 
  distance = 10000 ∧ time = 28 → 
  ∃ (speed : ℝ), abs (speed - 595.24) < 0.01 ∧ 
  speed = (distance * 100) / (time * 60) := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l320_32039


namespace NUMINAMATH_CALUDE_sum_edge_lengths_truncated_octahedron_l320_32018

/-- A polyhedron with 24 vertices and all edges of length 5 cm -/
structure Polyhedron where
  vertices : ℕ
  edge_length : ℝ
  h_vertices : vertices = 24
  h_edge_length : edge_length = 5

/-- A truncated octahedron is a polyhedron with 36 edges -/
def is_truncated_octahedron (p : Polyhedron) : Prop :=
  ∃ (edges : ℕ), edges = 36

/-- The sum of edge lengths for a polyhedron -/
def sum_edge_lengths (p : Polyhedron) (edges : ℕ) : ℝ :=
  p.edge_length * edges

/-- Theorem: If the polyhedron is a truncated octahedron, 
    then the sum of edge lengths is 180 cm -/
theorem sum_edge_lengths_truncated_octahedron (p : Polyhedron) 
  (h : is_truncated_octahedron p) : 
  ∃ (edges : ℕ), sum_edge_lengths p edges = 180 := by
  sorry


end NUMINAMATH_CALUDE_sum_edge_lengths_truncated_octahedron_l320_32018


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l320_32059

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (f : QuadraticPolynomial) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The discriminant of a quadratic polynomial -/
def QuadraticPolynomial.discriminant (f : QuadraticPolynomial) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- A function has exactly one solution when equal to a linear function -/
def has_exactly_one_solution (f : QuadraticPolynomial) (m : ℝ) (k : ℝ) : Prop :=
  ∃! x : ℝ, f.eval x = m * x + k

theorem quadratic_no_roots (f : QuadraticPolynomial) 
    (h1 : has_exactly_one_solution f 1 (-1))
    (h2 : has_exactly_one_solution f (-2) 2) :
    f.discriminant < 0 := by
  sorry

#check quadratic_no_roots

end NUMINAMATH_CALUDE_quadratic_no_roots_l320_32059


namespace NUMINAMATH_CALUDE_perfect_square_sum_l320_32030

theorem perfect_square_sum (x : ℕ) : x = 12 → ∃ y : ℕ, 2^x + 2^8 + 2^11 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l320_32030


namespace NUMINAMATH_CALUDE_dianes_gambling_problem_l320_32088

/-- Diane's gambling problem -/
theorem dianes_gambling_problem 
  (x y a b : ℝ) 
  (h1 : x * a = 65)
  (h2 : y * b = 150)
  (h3 : x * a - y * b = -50) :
  y * b - x * a = 50 := by
  sorry

end NUMINAMATH_CALUDE_dianes_gambling_problem_l320_32088


namespace NUMINAMATH_CALUDE_angle_bisector_length_formulas_l320_32034

theorem angle_bisector_length_formulas (a b c : ℝ) (α β γ : ℝ) (p R : ℝ) (l_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  α > 0 ∧ β > 0 ∧ γ > 0 ∧
  α + β + γ = π ∧
  p = (a + b + c) / 2 ∧
  R > 0 →
  (l_a = Real.sqrt (4 * p * (p - a) * b * c / ((b + c)^2))) ∧
  (l_a = 2 * b * c * Real.cos (α / 2) / (b + c)) ∧
  (l_a = 2 * R * Real.sin β * Real.sin γ / Real.cos ((β - γ) / 2)) ∧
  (l_a = 4 * p * Real.sin (β / 2) * Real.sin (γ / 2) / (Real.sin β + Real.sin γ)) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_formulas_l320_32034


namespace NUMINAMATH_CALUDE_twin_prime_divisibility_l320_32020

theorem twin_prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end NUMINAMATH_CALUDE_twin_prime_divisibility_l320_32020


namespace NUMINAMATH_CALUDE_gcd_143_98_l320_32068

theorem gcd_143_98 : Nat.gcd 143 98 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_98_l320_32068


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l320_32031

theorem coconut_grove_problem (x : ℕ) : 
  (3 * 60 + 2 * 120 + x * 180 = 100 * (3 + 2 + x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l320_32031


namespace NUMINAMATH_CALUDE_income_percentage_difference_l320_32008

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's -/
theorem income_percentage_difference :
  ∀ (a b c : ℝ),
  -- A's and B's monthly incomes are in the ratio 5:2
  a / b = 5 / 2 →
  -- C's monthly income is 12000
  c = 12000 →
  -- A's annual income is 403200.0000000001
  12 * a = 403200.0000000001 →
  -- B's monthly income is 12% more than C's
  b = 1.12 * c := by
sorry


end NUMINAMATH_CALUDE_income_percentage_difference_l320_32008


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l320_32079

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (2, x)
  parallel a b → x = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l320_32079


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l320_32017

/-- The number of saltwater aquariums Tyler has -/
def num_saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := num_saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l320_32017


namespace NUMINAMATH_CALUDE_julia_bill_ratio_l320_32013

/-- Proves the ratio of Julia's Sunday miles to Bill's Sunday miles -/
theorem julia_bill_ratio (bill_sunday : ℕ) (bill_saturday : ℕ) (julia_sunday : ℕ) :
  bill_sunday = 10 →
  bill_sunday = bill_saturday + 4 →
  bill_sunday + bill_saturday + julia_sunday = 36 →
  julia_sunday = 2 * bill_sunday :=
by sorry

end NUMINAMATH_CALUDE_julia_bill_ratio_l320_32013
