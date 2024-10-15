import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1297_129745

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1297_129745


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l1297_129748

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  fun x => -2 * x

theorem shift_direct_proportion :
  ∃ (f : LinearFunction),
    f.m = -2 ∧
    f.b = 6 ∧
    (∀ x, (horizontalShift originalFunction 3) x = f.m * x + f.b) := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l1297_129748


namespace NUMINAMATH_CALUDE_snow_probability_l1297_129755

-- Define the probability of snow on Friday
def prob_snow_friday : ℝ := 0.4

-- Define the probability of snow on Saturday
def prob_snow_saturday : ℝ := 0.3

-- Define the probability of snow on both days
def prob_snow_both_days : ℝ := prob_snow_friday * prob_snow_saturday

-- Theorem to prove
theorem snow_probability : prob_snow_both_days = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1297_129755


namespace NUMINAMATH_CALUDE_infinite_sets_with_special_divisibility_l1297_129728

theorem infinite_sets_with_special_divisibility :
  ∃ f : ℕ → Fin 1983 → ℕ,
    (∀ k : ℕ, ∀ i j : Fin 1983, i < j → f k i < f k j) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, ∃ a : ℕ, a > 1 ∧ (a ^ 1983 ∣ f k i)) ∧
    (∀ k : ℕ, ∀ i : Fin 1983, i.val < 1982 → f k i.succ = f k i + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sets_with_special_divisibility_l1297_129728


namespace NUMINAMATH_CALUDE_rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l1297_129720

/-- Represents a four-digit positive integer ABCD where A and D are non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : a > 0 ∧ a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : d > 0 ∧ d < 10

/-- Calculates the reverse of a four-digit number -/
def reverse (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Calculates the RD sum of a four-digit number -/
def rdSum (n : FourDigitNumber) : Nat :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) + reverse n

/-- Theorem: The RD sum of ABCD is equal to 1001(A + D) + 110(B + C) -/
theorem rd_sum_formula (n : FourDigitNumber) :
  rdSum n = 1001 * (n.a + n.d) + 110 * (n.b + n.c) := by
  sorry

/-- The number of four-digit integers whose RD sum is 3883 -/
def count_rd_sum_3883 : Nat := 18

/-- Theorem: The number of four-digit integers whose RD sum is 3883 is 18 -/
theorem count_rd_sum_3883_is_18 :
  count_rd_sum_3883 = 18 := by
  sorry

/-- The number of four-digit integers that are equal to the RD sum of a four-digit integer -/
def count_rd_sum_equal : Nat := 143

/-- Theorem: The number of four-digit integers that are equal to the RD sum of a four-digit integer is 143 -/
theorem count_rd_sum_equal_is_143 :
  count_rd_sum_equal = 143 := by
  sorry

end NUMINAMATH_CALUDE_rd_sum_formula_count_rd_sum_3883_is_18_count_rd_sum_equal_is_143_l1297_129720


namespace NUMINAMATH_CALUDE_percentage_of_returned_books_l1297_129722

def initial_books : ℕ := 75
def final_books : ℕ := 57
def loaned_books : ℕ := 60

theorem percentage_of_returned_books :
  (initial_books - final_books) * 100 / loaned_books = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_returned_books_l1297_129722


namespace NUMINAMATH_CALUDE_surfers_problem_l1297_129741

/-- The number of surfers on the beach with fewer surfers -/
def x : ℕ := sorry

/-- The number of surfers on Malibu beach -/
def y : ℕ := sorry

/-- The total number of surfers on both beaches -/
def total : ℕ := 60

theorem surfers_problem :
  (y = 2 * x) ∧ (x + y = total) → x = 20 := by sorry

end NUMINAMATH_CALUDE_surfers_problem_l1297_129741


namespace NUMINAMATH_CALUDE_modified_geometric_progression_sum_of_squares_l1297_129710

/-- The sum of squares of a modified geometric progression -/
theorem modified_geometric_progression_sum_of_squares
  (b c s : ℝ) (h : abs s < 1) :
  let modifiedSum := (c^2 * b^2 * s^4) / (1 - s)
  let modifiedSequence := fun n => if n < 3 then b * s^(n-1) else c * b * s^(n-1)
  ∑' n, (modifiedSequence n)^2 = modifiedSum :=
sorry

end NUMINAMATH_CALUDE_modified_geometric_progression_sum_of_squares_l1297_129710


namespace NUMINAMATH_CALUDE_min_ferries_required_l1297_129736

def ferry_capacity : ℕ := 45
def people_to_transport : ℕ := 523

theorem min_ferries_required : 
  ∃ (n : ℕ), n * ferry_capacity ≥ people_to_transport ∧ 
  ∀ (m : ℕ), m * ferry_capacity ≥ people_to_transport → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_min_ferries_required_l1297_129736


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1297_129783

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A * Real.sin (A/2) + Real.sin B * Real.sin (B/2) + Real.sin C * Real.sin (C/2) ≤ 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1297_129783


namespace NUMINAMATH_CALUDE_exists_hexagonal_2016_l1297_129701

/-- The n-th hexagonal number -/
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

/-- 2016 is a hexagonal number -/
theorem exists_hexagonal_2016 : ∃ n : ℕ, n > 0 ∧ hexagonal n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_exists_hexagonal_2016_l1297_129701


namespace NUMINAMATH_CALUDE_equation_equality_l1297_129770

theorem equation_equality (a b : ℝ) : -a*b + 3*b*a = 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1297_129770


namespace NUMINAMATH_CALUDE_first_number_a10_l1297_129787

def first_number (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1) / 2)

theorem first_number_a10 : first_number 10 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_number_a10_l1297_129787


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1297_129744

/-- A rectangle ABCD with specific properties -/
structure SpecialRectangle where
  -- AB, BC, CD are sides of the rectangle
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- E is the midpoint of BC
  BE : ℝ
  -- Conditions
  rectangle_condition : AB = CD
  perimeter_condition : AB + BC + CD = 20
  midpoint_condition : BE = BC / 2
  diagonal_condition : AB^2 + BE^2 = 9^2

/-- The area of a SpecialRectangle is 19 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.AB * r.BC = 19 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l1297_129744


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_verify_bus_capacities_l1297_129723

/-- Represents the capacity and cost of buses -/
structure BusInfo where
  small_capacity : ℕ
  large_capacity : ℕ
  small_cost : ℕ
  large_cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  small_buses : ℕ
  large_buses : ℕ

/-- Checks if a rental plan is valid for the given number of students -/
def is_valid_plan (info : BusInfo) (students : ℕ) (plan : RentalPlan) : Prop :=
  plan.small_buses * info.small_capacity + plan.large_buses * info.large_capacity = students

/-- Calculates the cost of a rental plan -/
def plan_cost (info : BusInfo) (plan : RentalPlan) : ℕ :=
  plan.small_buses * info.small_cost + plan.large_buses * info.large_cost

/-- Theorem stating the most cost-effective plan -/
theorem most_cost_effective_plan (info : BusInfo) (students : ℕ) : 
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  info.small_cost = 200 →
  info.large_cost = 400 →
  students = 400 →
  ∃ (optimal_plan : RentalPlan),
    is_valid_plan info students optimal_plan ∧
    optimal_plan.small_buses = 2 ∧
    optimal_plan.large_buses = 8 ∧
    plan_cost info optimal_plan = 3600 ∧
    ∀ (plan : RentalPlan), 
      is_valid_plan info students plan → 
      plan_cost info optimal_plan ≤ plan_cost info plan :=
by
  sorry

/-- Verifies the given bus capacities -/
theorem verify_bus_capacities (info : BusInfo) :
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  3 * info.small_capacity + info.large_capacity = 105 ∧
  info.small_capacity + 2 * info.large_capacity = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_verify_bus_capacities_l1297_129723


namespace NUMINAMATH_CALUDE_min_a_value_l1297_129756

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x
def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

-- State the theorem
theorem min_a_value (a : ℝ) :
  (a < 0) →
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 / (g x₁ * g x₂)) →
  a ≥ 3 - 2 / 3 * Real.exp 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_a_value_l1297_129756


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1297_129796

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1297_129796


namespace NUMINAMATH_CALUDE_number_and_square_difference_l1297_129771

theorem number_and_square_difference (N : ℝ) : N^2 - N = 12 ↔ N = 4 ∨ N = -3 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_difference_l1297_129771


namespace NUMINAMATH_CALUDE_pentagon_diagonals_from_vertex_l1297_129777

/-- The number of diagonals that can be drawn from a vertex of an n-sided polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

theorem pentagon_diagonals_from_vertex :
  diagonals_from_vertex pentagon_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_from_vertex_l1297_129777


namespace NUMINAMATH_CALUDE_transformed_function_eq_g_l1297_129706

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 4

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The horizontal shift transformation -/
def shift_left (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

/-- The vertical shift transformation -/
def shift_down (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - k

/-- Theorem stating that the transformed function is equivalent to g -/
theorem transformed_function_eq_g :
  ∀ x, shift_down 3 (shift_left 2 f) x = g x := by sorry

end NUMINAMATH_CALUDE_transformed_function_eq_g_l1297_129706


namespace NUMINAMATH_CALUDE_max_pairs_sum_l1297_129711

theorem max_pairs_sum (k : ℕ) (a b : ℕ → ℕ) : 
  (∀ i : ℕ, i < k → a i < b i) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i : ℕ, i < k → a i ∈ Finset.range 4019 ∧ b i ∈ Finset.range 4019) →
  (∀ i : ℕ, i < k → a i + b i ≤ 4019) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  k ≤ 1607 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l1297_129711


namespace NUMINAMATH_CALUDE_smallest_a_value_l1297_129788

/-- Given a polynomial x^3 - ax^2 + bx - 3003 with three positive integer roots,
    the smallest possible value of a is 45 -/
theorem smallest_a_value (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 3003 = (x - r₁)*(x - r₂)*(x - r₃)) →
  a ≥ 45 ∧ ∃ a₀ b₀ r₁₀ r₂₀ r₃₀, 
    a₀ = 45 ∧ 
    (∀ x, x^3 - a₀*x^2 + b₀*x - 3003 = (x - r₁₀)*(x - r₂₀)*(x - r₃₀)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_a_value_l1297_129788


namespace NUMINAMATH_CALUDE_value_of_x_l1297_129731

theorem value_of_x : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1297_129731


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1297_129765

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1297_129765


namespace NUMINAMATH_CALUDE_compute_expression_l1297_129719

theorem compute_expression : (3 + 7)^2 + Real.sqrt (3^2 + 7^2) = 100 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1297_129719


namespace NUMINAMATH_CALUDE_greenfield_high_school_teachers_l1297_129799

/-- The number of students at Greenfield High School -/
def num_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at Greenfield High School -/
def num_teachers : ℕ := 44

theorem greenfield_high_school_teachers :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end NUMINAMATH_CALUDE_greenfield_high_school_teachers_l1297_129799


namespace NUMINAMATH_CALUDE_preston_order_calculation_l1297_129735

/-- The total amount Preston received from Abra Company's order -/
def total_received (sandwich_price : ℚ) (delivery_fee : ℚ) (num_sandwiches : ℕ) (tip_percentage : ℚ) : ℚ :=
  let subtotal := sandwich_price * num_sandwiches + delivery_fee
  subtotal + subtotal * tip_percentage

/-- Preston's sandwich shop order calculation -/
theorem preston_order_calculation :
  total_received 5 20 18 (1/10) = 121 := by
  sorry

end NUMINAMATH_CALUDE_preston_order_calculation_l1297_129735


namespace NUMINAMATH_CALUDE_det_scalar_mult_l1297_129784

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 4, 3]
def k : ℝ := 3

theorem det_scalar_mult :
  Matrix.det (k • A) = 207 := by sorry

end NUMINAMATH_CALUDE_det_scalar_mult_l1297_129784


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1297_129751

theorem quadratic_inequality_solution (m n : ℝ) : 
  (∀ x, x^2 - m*x + n ≤ 0 ↔ -5 ≤ x ∧ x ≤ 1) → m = -4 ∧ n = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1297_129751


namespace NUMINAMATH_CALUDE_power_equation_solution_l1297_129764

theorem power_equation_solution (n : ℕ) : 5^n = 5 * 25^3 * 125^2 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1297_129764


namespace NUMINAMATH_CALUDE_inequality_proof_l1297_129743

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1297_129743


namespace NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l1297_129739

/-- 
Given positive real numbers x and y such that x + y = 50,
x^3 * y^4 is maximized when x = 150/7 and y = 200/7.
-/
theorem maximize_x_cube_y_fourth (x y : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (sum_xy : x + y = 50) :
  x^3 * y^4 ≤ (150/7)^3 * (200/7)^4 ∧ 
  x^3 * y^4 = (150/7)^3 * (200/7)^4 ↔ x = 150/7 ∧ y = 200/7 := by
  sorry

#check maximize_x_cube_y_fourth

end NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l1297_129739


namespace NUMINAMATH_CALUDE_triangle_area_and_perimeter_l1297_129703

theorem triangle_area_and_perimeter 
  (DE FD : ℝ) 
  (h_DE : DE = 12) 
  (h_FD : FD = 20) 
  (h_right_angle : DE * FD = 2 * (1/2 * DE * FD)) : 
  let EF := Real.sqrt (DE^2 + FD^2)
  (1/2 * DE * FD = 120) ∧ (DE + FD + EF = 32 + 2 * Real.sqrt 136) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_perimeter_l1297_129703


namespace NUMINAMATH_CALUDE_average_of_xyz_l1297_129737

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1297_129737


namespace NUMINAMATH_CALUDE_new_person_weight_l1297_129730

/-- Given 8 persons, if replacing one person weighing 50 kg with a new person 
    increases the average weight by 2.5 kg, then the weight of the new person is 70 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 50 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1297_129730


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l1297_129792

def total_amount : ℝ := 3000
def first_part : ℝ := 299.99999999999994
def second_part : ℝ := total_amount - first_part
def interest_rate1 : ℝ := 0.03
def interest_rate2 : ℝ := 0.05

theorem annual_interest_calculation :
  let interest1 := first_part * interest_rate1
  let interest2 := second_part * interest_rate2
  interest1 + interest2 = 144 := by sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l1297_129792


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l1297_129758

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l1297_129758


namespace NUMINAMATH_CALUDE_percentage_difference_l1297_129768

theorem percentage_difference : (65 / 100 * 40) - (4 / 5 * 25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1297_129768


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_m_eq_neg_one_l1297_129785

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- A complex number z defined as i(5-i) + m -/
def z (m : ℝ) : ℂ := i * (5 - i) + m

/-- Theorem stating that z is purely imaginary if and only if m = -1 -/
theorem z_purely_imaginary_iff_m_eq_neg_one (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = -1 := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_m_eq_neg_one_l1297_129785


namespace NUMINAMATH_CALUDE_students_not_enrolled_l1297_129790

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l1297_129790


namespace NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l1297_129724

/-- Represents the outcome of a die roll -/
inductive DieOutcome
| Prime (n : Nat)
| NonPrimeSquare (n : Nat)
| Other (n : Nat)

/-- The set of possible outcomes for an 8-sided die -/
def dieOutcomes : Finset DieOutcome := sorry

/-- The probability of each outcome, assuming a fair die -/
def prob (outcome : DieOutcome) : ℚ := sorry

/-- The winnings for each outcome -/
def winnings (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime n => n
  | DieOutcome.NonPrimeSquare _ => 2
  | DieOutcome.Other _ => -4

/-- The expected value of winnings for one die toss -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value of winnings is 13/8 -/
theorem expected_value_is_thirteen_eighths :
  expectedValue = 13 / 8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l1297_129724


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_inequality_l1297_129773

theorem rectangle_area_perimeter_inequality (a b : ℕ+) : (a + 2) * (b + 2) - 8 ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_inequality_l1297_129773


namespace NUMINAMATH_CALUDE_print_shop_charges_l1297_129754

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 70

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 35

theorem print_shop_charges :
  charge_y * num_copies = charge_x * num_copies + additional_charge := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charges_l1297_129754


namespace NUMINAMATH_CALUDE_sam_bought_cards_l1297_129786

/-- The number of baseball cards Sam bought from Mike -/
def cards_bought (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

/-- Theorem stating that the number of cards Sam bought is the difference between Mike's initial and current number of cards -/
theorem sam_bought_cards (mike_initial mike_current : ℕ) 
  (h1 : mike_initial = 87) 
  (h2 : mike_current = 74) : 
  cards_bought mike_initial mike_current = 13 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_cards_l1297_129786


namespace NUMINAMATH_CALUDE_buffer_saline_volume_l1297_129727

theorem buffer_saline_volume 
  (total_buffer : ℚ) 
  (solution_b_volume : ℚ) 
  (saline_volume : ℚ) 
  (initial_mixture_volume : ℚ) :
  total_buffer = 3/2 →
  solution_b_volume = 1/4 →
  saline_volume = 1/6 →
  initial_mixture_volume = 5/12 →
  solution_b_volume + saline_volume = initial_mixture_volume →
  (total_buffer * (saline_volume / initial_mixture_volume) : ℚ) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_buffer_saline_volume_l1297_129727


namespace NUMINAMATH_CALUDE_root_product_l1297_129774

theorem root_product (f g : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (∀ x, f x = x^5 - x^3 + 1) →
  (∀ x, g x = x^2 - 2) →
  f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 →
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = -7 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l1297_129774


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1297_129717

def initial_puppies (sold : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  sold + (puppies_per_cage * cages_used)

theorem pet_store_puppies :
  initial_puppies 7 2 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1297_129717


namespace NUMINAMATH_CALUDE_equation_solution_l1297_129714

theorem equation_solution :
  ∃ x : ℝ, (8 : ℝ) ^ (2 * x - 9) = 2 ^ (-2 * x - 3) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1297_129714


namespace NUMINAMATH_CALUDE_difference_of_squares_75_45_l1297_129793

theorem difference_of_squares_75_45 : 75^2 - 45^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_45_l1297_129793


namespace NUMINAMATH_CALUDE_pentagon_probability_l1297_129702

/-- A type representing the points on the pentagon --/
inductive PentagonPoint
| Vertex : Fin 5 → PentagonPoint
| Midpoint : Fin 5 → PentagonPoint

/-- The total number of points on the pentagon --/
def total_points : ℕ := 10

/-- A function to determine if two points are exactly one side apart --/
def one_side_apart (p q : PentagonPoint) : Prop :=
  match p, q with
  | PentagonPoint.Vertex i, PentagonPoint.Vertex j => (j - i) % 5 = 2 ∨ (i - j) % 5 = 2
  | _, _ => False

/-- The number of ways to choose 2 points from the total points --/
def total_choices : ℕ := (total_points.choose 2)

/-- The number of point pairs that are one side apart --/
def favorable_choices : ℕ := 10

theorem pentagon_probability :
  (favorable_choices : ℚ) / total_choices = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_pentagon_probability_l1297_129702


namespace NUMINAMATH_CALUDE_team_average_typing_speed_l1297_129763

def team_size : ℕ := 5

def typing_speeds : List ℕ := [64, 76, 91, 80, 89]

def average_typing_speed : ℚ := (typing_speeds.sum : ℚ) / team_size

theorem team_average_typing_speed :
  average_typing_speed = 80 := by sorry

end NUMINAMATH_CALUDE_team_average_typing_speed_l1297_129763


namespace NUMINAMATH_CALUDE_johns_toy_store_spending_l1297_129779

/-- Proves that the fraction of John's remaining allowance spent at the toy store is 1/3 -/
theorem johns_toy_store_spending (
  total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (candy_store_amount : ℚ)
  (h1 : total_allowance = 33/10)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_store_amount = 88/100) :
  let remaining_after_arcade := total_allowance - arcade_fraction * total_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry


end NUMINAMATH_CALUDE_johns_toy_store_spending_l1297_129779


namespace NUMINAMATH_CALUDE_test_questions_l1297_129721

theorem test_questions (Q : ℝ) : 
  (0.9 * (Q / 2) + 0.95 * (Q / 2) = 74) → Q = 80 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l1297_129721


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1297_129778

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1297_129778


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1297_129712

/-- 
Given a regular triangular pyramid with angle α between a lateral edge and a side of the base,
and a cross-section of area S made through the midpoint of a lateral edge parallel to the lateral face,
the volume V of the pyramid is (8√3 S cos²α) / (3 sin(2α)), where π/6 < α < π/2.
-/
theorem regular_triangular_pyramid_volume 
  (α : Real) 
  (S : Real) 
  (h1 : π/6 < α) 
  (h2 : α < π/2) 
  (h3 : S > 0) : 
  ∃ V : Real, V = (8 * Real.sqrt 3 * S * (Real.cos α)^2) / (3 * Real.sin (2 * α)) := by
  sorry

#check regular_triangular_pyramid_volume

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1297_129712


namespace NUMINAMATH_CALUDE_f_2007_equals_neg_two_l1297_129738

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem f_2007_equals_neg_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_sym : symmetric_around_two f)
  (h_neg_three : f (-3) = -2) :
  f 2007 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_neg_two_l1297_129738


namespace NUMINAMATH_CALUDE_inequality_on_unit_circle_l1297_129795

theorem inequality_on_unit_circle (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 ≤ 
  2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
sorry

end NUMINAMATH_CALUDE_inequality_on_unit_circle_l1297_129795


namespace NUMINAMATH_CALUDE_rogers_final_amount_l1297_129709

def rogers_money (initial : ℕ) (gift : ℕ) (spent : ℕ) : ℕ :=
  initial + gift - spent

theorem rogers_final_amount :
  rogers_money 16 28 25 = 19 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_amount_l1297_129709


namespace NUMINAMATH_CALUDE_correct_initial_lives_l1297_129761

/-- The number of lives a player starts with in a game -/
def initial_lives : ℕ := 2

/-- The number of extra lives gained in the first level -/
def extra_lives_level1 : ℕ := 6

/-- The number of extra lives gained in the second level -/
def extra_lives_level2 : ℕ := 11

/-- The total number of lives after two levels -/
def total_lives : ℕ := 19

theorem correct_initial_lives :
  initial_lives + extra_lives_level1 + extra_lives_level2 = total_lives :=
by sorry

end NUMINAMATH_CALUDE_correct_initial_lives_l1297_129761


namespace NUMINAMATH_CALUDE_count_divisors_with_specific_remainder_l1297_129769

theorem count_divisors_with_specific_remainder :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 17 ∧ 2017 % n = 17) ∧
    (∀ n : ℕ, n > 17 ∧ 2017 % n = 17 → n ∈ S) ∧
    S.card = 13 :=
by sorry

end NUMINAMATH_CALUDE_count_divisors_with_specific_remainder_l1297_129769


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1297_129729

theorem arithmetic_sequence_sum (n : ℕ) (a₁ : ℤ) : n > 1 → (∃ k : ℕ, n * k = 2000) →
  (n * (2 * a₁ + (n - 1) * 2)) / 2 = 2000 ↔ n ∣ 2000 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1297_129729


namespace NUMINAMATH_CALUDE_plate_count_l1297_129734

theorem plate_count (n : ℕ) 
  (h1 : 500 < n ∧ n < 600)
  (h2 : n % 10 = 7)
  (h3 : n % 12 = 7) : 
  n = 547 := by
sorry

end NUMINAMATH_CALUDE_plate_count_l1297_129734


namespace NUMINAMATH_CALUDE_book_exchange_count_l1297_129733

/-- Represents a book exchange in a book club --/
structure BookExchange where
  friends : Finset (Fin 6)
  give : Fin 6 → Fin 6
  receive : Fin 6 → Fin 6

/-- Conditions for a valid book exchange --/
def ValidExchange (e : BookExchange) : Prop :=
  (∀ i, i ∈ e.friends) ∧
  (∀ i, e.give i ≠ i) ∧
  (∀ i, e.receive i ≠ i) ∧
  (∀ i, e.give i ≠ e.receive i) ∧
  (∀ i j, e.give i = e.give j → i = j) ∧
  (∀ i j, e.receive i = e.receive j → i = j)

/-- The number of valid book exchanges --/
def NumberOfExchanges : ℕ := sorry

/-- Theorem stating that the number of valid book exchanges is 160 --/
theorem book_exchange_count : NumberOfExchanges = 160 := by sorry

end NUMINAMATH_CALUDE_book_exchange_count_l1297_129733


namespace NUMINAMATH_CALUDE_group_division_ways_l1297_129726

theorem group_division_ways (n : ℕ) (g₁ g₂ g₃ : ℕ) (h₁ : n = 8) (h₂ : g₁ = 2) (h₃ : g₂ = 3) (h₄ : g₃ = 3) :
  (Nat.choose n g₂ * Nat.choose (n - g₂) g₃) / 2 = 280 :=
by sorry

end NUMINAMATH_CALUDE_group_division_ways_l1297_129726


namespace NUMINAMATH_CALUDE_intersection_difference_l1297_129732

theorem intersection_difference (A B : Set ℕ) (m n : ℕ) :
  A = {1, 2, m} →
  B = {2, 3, 4, n} →
  A ∩ B = {1, 2, 3} →
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_difference_l1297_129732


namespace NUMINAMATH_CALUDE_fraction_equality_l1297_129742

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + 2*b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 10/57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1297_129742


namespace NUMINAMATH_CALUDE_burger_cost_l1297_129725

/-- Given Alice's and Charlie's purchases, prove the cost of a burger -/
theorem burger_cost :
  ∀ (burger_cost soda_cost : ℕ),
  5 * burger_cost + 3 * soda_cost = 500 →
  3 * burger_cost + 2 * soda_cost = 310 →
  burger_cost = 70 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l1297_129725


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1297_129757

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 20 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1297_129757


namespace NUMINAMATH_CALUDE_cos_135_degrees_l1297_129775

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l1297_129775


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1297_129713

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1297_129713


namespace NUMINAMATH_CALUDE_nineteenth_term_is_zero_l1297_129715

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℚ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℚ, ∀ n : ℕ, (1 / (a (n + 1) + 1) - 1 / (a n + 1)) = d

/-- The 19th term of the special sequence is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℚ) (h : special_sequence a) : 
  a 19 = 0 := by
sorry

end NUMINAMATH_CALUDE_nineteenth_term_is_zero_l1297_129715


namespace NUMINAMATH_CALUDE_total_germs_count_l1297_129789

/-- The number of petri dishes in the biology lab. -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish. -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab. -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000. -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end NUMINAMATH_CALUDE_total_germs_count_l1297_129789


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1297_129760

theorem square_sum_zero_implies_both_zero (a b : ℝ) : 
  a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1297_129760


namespace NUMINAMATH_CALUDE_stock_percentage_example_l1297_129780

/-- The percentage of stock that yields a given income from a given investment --/
def stock_percentage (income : ℚ) (investment : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem: The stock percentage for an income of 15000 and investment of 37500 is 40% --/
theorem stock_percentage_example : stock_percentage 15000 37500 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_example_l1297_129780


namespace NUMINAMATH_CALUDE_meat_remaining_l1297_129700

theorem meat_remaining (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ) :
  initial_meat = 20 →
  meatball_fraction = 1/4 →
  spring_roll_meat = 3 →
  initial_meat - (meatball_fraction * initial_meat + spring_roll_meat) = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_remaining_l1297_129700


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1297_129707

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.cos x}

-- Define set B
def B : Set ℝ := {x | x^2 + x ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 ∪ {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1297_129707


namespace NUMINAMATH_CALUDE_logical_conclusion_l1297_129740

/-- Represents whether a student submitted all required essays -/
def submitted_all_essays (student : ℕ) : Prop := sorry

/-- Represents whether a student failed the course -/
def failed_course (student : ℕ) : Prop := sorry

/-- Ms. Thompson's statement -/
axiom thompson_statement : ∀ (student : ℕ), ¬(submitted_all_essays student) → failed_course student

/-- The statement to be proved -/
theorem logical_conclusion : ∀ (student : ℕ), ¬(failed_course student) → submitted_all_essays student :=
sorry

end NUMINAMATH_CALUDE_logical_conclusion_l1297_129740


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1297_129705

theorem quadratic_equation_distinct_roots (p q : ℚ) : 
  (∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = 2*p ∨ x = p + q) ∧ 
  (2*p ≠ p + q) → 
  p = 2/3 ∧ q = -8/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1297_129705


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_equality_l1297_129766

-- Define the box office revenue
def box_office_revenue : ℝ := 1824000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.824 * (10 ^ 9)

-- Theorem stating that the box office revenue is equal to its scientific notation representation
theorem box_office_scientific_notation_equality : 
  box_office_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_box_office_scientific_notation_equality_l1297_129766


namespace NUMINAMATH_CALUDE_restaurant_weekday_earnings_l1297_129762

/-- Represents the daily earnings of a restaurant on weekdays -/
def weekday_earnings : ℝ := sorry

/-- Represents the daily earnings of a restaurant on weekend days -/
def weekend_earnings : ℝ := 2 * weekday_earnings

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The number of weekdays in a week -/
def weekdays_in_week : ℕ := 5

/-- The total monthly earnings of the restaurant -/
def total_monthly_earnings : ℝ := 21600

/-- Theorem stating that the daily weekday earnings of the restaurant are $600 -/
theorem restaurant_weekday_earnings :
  weekday_earnings = 600 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_weekday_earnings_l1297_129762


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1297_129718

theorem two_numbers_problem :
  ∃ (x y : ℝ), y = 33 ∧ x + y = 51 ∧ y = 2 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1297_129718


namespace NUMINAMATH_CALUDE_existence_of_sequence_l1297_129752

theorem existence_of_sequence : ∃ (s : List ℕ), 
  (s.length > 10) ∧ 
  (s.sum = 20) ∧ 
  (∀ (i j : ℕ), i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l1297_129752


namespace NUMINAMATH_CALUDE_locus_of_point_P_l1297_129716

-- Define the 2D plane
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 2)]

-- Define points A, B, and P
variable (A B P : V)

-- Define the distance function
def dist (x y : V) : ℝ := ‖x - y‖

-- Theorem statement
theorem locus_of_point_P (h1 : dist A B = 3) (h2 : dist A P + dist B P = 3) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l1297_129716


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l1297_129753

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 25) (h2 : sum = 3125) :
  (sum : ℚ) / n = 125 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l1297_129753


namespace NUMINAMATH_CALUDE_geometric_progression_values_l1297_129776

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (9 * p + 10) * r = 3 * p ∧ (3 * p) * r = |p - 8|) ↔ 
  (p = -1 ∨ p = 40 / 9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l1297_129776


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1297_129759

theorem geometric_sequence_common_ratio (q : ℝ) : 
  (1 + q + q^2 = 13) ↔ (q = 3 ∨ q = -4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1297_129759


namespace NUMINAMATH_CALUDE_exam_time_allocation_l1297_129772

theorem exam_time_allocation :
  ∀ (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) (type_b_questions : ℕ),
    total_time = 3 * 60 →
    total_questions = 200 →
    type_a_questions = 50 →
    type_b_questions = total_questions - type_a_questions →
    2 * (total_time / total_questions) * type_b_questions = 
      (total_time / total_questions) * type_a_questions →
    (total_time / total_questions) * type_a_questions = 72 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l1297_129772


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l1297_129798

/-- The number of ways to arrange the letters of BANANA with indistinguishable A's and N's -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = (Nat.factorial total_letters) / 
    ((Nat.factorial num_a) * (Nat.factorial num_n) * (Nat.factorial num_b)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l1297_129798


namespace NUMINAMATH_CALUDE_task_fraction_by_B_l1297_129747

theorem task_fraction_by_B (a b : ℚ) : 
  (a = (2/5) * b) → (b = (5/7) * (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_task_fraction_by_B_l1297_129747


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_is_28_7_l1297_129767

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_light_rain : ℝ
  prob_heavy_rain : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate the expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_light_rain * f.light_rain_amount + f.prob_heavy_rain * f.heavy_rain_amount

/-- Calculate the expected total rainfall for a week -/
def expected_weekly_rainfall (f : DailyForecast) : ℝ :=
  7 * expected_daily_rainfall f

/-- The weather forecast for each day of the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
  , prob_light_rain := 0.3
  , prob_heavy_rain := 0.4
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

theorem expected_weekly_rainfall_is_28_7 :
  expected_weekly_rainfall weekly_forecast = 28.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_is_28_7_l1297_129767


namespace NUMINAMATH_CALUDE_johns_donation_is_100_l1297_129797

/-- The size of John's donation to a charity fund --/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) (new_average : ℝ) : ℝ :=
  (num_initial_contributions + 1) * new_average - num_initial_contributions * initial_average

/-- Theorem stating that John's donation is $100 given the problem conditions --/
theorem johns_donation_is_100 :
  johns_donation 50 1 75 = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_donation_is_100_l1297_129797


namespace NUMINAMATH_CALUDE_rolls_combination_count_l1297_129781

theorem rolls_combination_count :
  let total_rolls : ℕ := 8
  let min_per_kind : ℕ := 2
  let num_kinds : ℕ := 3
  let remaining_rolls : ℕ := total_rolls - (min_per_kind * num_kinds)
  Nat.choose (remaining_rolls + num_kinds - 1) (num_kinds - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_rolls_combination_count_l1297_129781


namespace NUMINAMATH_CALUDE_min_value_sum_and_reciprocals_l1297_129708

theorem min_value_sum_and_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 ∧ (a + b + 1/a + 1/b = 4 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_and_reciprocals_l1297_129708


namespace NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l1297_129750

/-- A circle in a 2D plane. -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The maximum number of intersection points between a line segment and a circle. -/
def max_intersections_line_circle : ℕ := 2

/-- The number of sides in a quadrilateral. -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8. -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (max_intersections_line_circle * quadrilateral_sides : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l1297_129750


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1297_129794

theorem inverse_variation_problem (x y : ℝ) :
  (∀ (x y : ℝ), x > 0 ∧ y > 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) →
  (2^3 * 5 = k) →
  (x^3 * 2000 = k) →
  x = 1 / Real.rpow 50 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1297_129794


namespace NUMINAMATH_CALUDE_problem_solution_l1297_129704

noncomputable def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem problem_solution (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1297_129704


namespace NUMINAMATH_CALUDE_tank_capacity_is_640_verify_capacity_l1297_129749

/-- Represents the capacity of a tank in litres. -/
def tank_capacity : ℝ := 640

/-- Represents the time in hours it takes to empty the tank with only the outlet pipe open. -/
def outlet_time : ℝ := 10

/-- Represents the rate at which the inlet pipe adds water, in litres per minute. -/
def inlet_rate : ℝ := 4

/-- Represents the time in hours it takes to empty the tank with both inlet and outlet pipes open. -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 litres given the conditions. -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by
  sorry

/-- Verifies that the calculated capacity matches the given value of 640 litres. -/
theorem verify_capacity :
  tank_capacity = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_640_verify_capacity_l1297_129749


namespace NUMINAMATH_CALUDE_hector_gumballs_l1297_129791

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := 4 * alisha_gumballs - 5

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The total number of gumballs Hector purchased -/
def total_gumballs : ℕ := todd_gumballs + alisha_gumballs + bobby_gumballs + remaining_gumballs

theorem hector_gumballs : total_gumballs = 45 := by
  sorry

end NUMINAMATH_CALUDE_hector_gumballs_l1297_129791


namespace NUMINAMATH_CALUDE_angle4_measure_l1297_129746

-- Define the angles
def angle1 : ℝ := 85
def angle2 : ℝ := 34
def angle3 : ℝ := 20

-- Define the theorem
theorem angle4_measure : 
  ∀ (angle4 angle5 angle6 : ℝ),
  -- Conditions
  (angle1 + angle2 + angle3 + angle5 + angle6 = 180) →
  (angle4 + angle5 + angle6 = 180) →
  -- Conclusion
  angle4 = 139 := by
sorry

end NUMINAMATH_CALUDE_angle4_measure_l1297_129746


namespace NUMINAMATH_CALUDE_marble_158_is_gray_l1297_129782

/-- Represents the color of a marble -/
inductive Color
  | Gray
  | White
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : Color :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => Color.Gray
  | 5 | 6 | 7 | 8 => Color.White
  | _ => Color.Black

theorem marble_158_is_gray : marbleColor 158 = Color.Gray := by
  sorry

end NUMINAMATH_CALUDE_marble_158_is_gray_l1297_129782
