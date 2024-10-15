import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l3637_363753

theorem expression_value (x y : ℝ) (hx : x = 8) (hy : y = 3) :
  (x - 2*y) * (x + 2*y) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3637_363753


namespace NUMINAMATH_CALUDE_binary_linear_equation_sum_l3637_363782

/-- A binary linear equation is an equation where the exponents of all variables are 1. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y, f x y = a * x + b * y + c

/-- Given that x^(3m-3) - 2y^(n-1) = 5 is a binary linear equation, prove that m + n = 10/3 -/
theorem binary_linear_equation_sum (m n : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(3*m-3) - 2*y^(n-1) - 5) →
  m + n = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_binary_linear_equation_sum_l3637_363782


namespace NUMINAMATH_CALUDE_participation_plans_eq_48_l3637_363752

/-- The number of different participation plans for selecting 3 out of 5 students
    for math, physics, and chemistry competitions, where each student competes
    in one subject and student A cannot participate in the physics competition. -/
def participation_plans : ℕ :=
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let competitions : ℕ := 3
  let student_a_options : ℕ := 2  -- math or chemistry

  let scenario1 : ℕ := (total_students - 1).factorial / (total_students - 1 - selected_students).factorial
  let scenario2 : ℕ := student_a_options * ((total_students - 1).factorial / (total_students - 1 - (selected_students - 1)).factorial)

  scenario1 + scenario2

theorem participation_plans_eq_48 :
  participation_plans = 48 := by
  sorry

end NUMINAMATH_CALUDE_participation_plans_eq_48_l3637_363752


namespace NUMINAMATH_CALUDE_ellipse_equation_l3637_363746

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  c : ℝ  -- Distance from center to focus
  h : c < a  -- Ensure c is less than a

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / (e.a^2 - e.c^2) = 1

/-- Theorem: Given the conditions, prove the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.a + e.c = 3)  -- Distance to one focus is 3
  (h2 : e.a - e.c = 1)  -- Distance to the other focus is 1
  (x y : ℝ) :
  standard_equation e x y ↔ x^2/4 + y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3637_363746


namespace NUMINAMATH_CALUDE_exam_pass_count_l3637_363799

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed failed : ℕ), 
    passed + failed = total ∧
    passed * passed_avg + failed * failed_avg = total * overall_avg ∧
    passed = 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_count_l3637_363799


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3637_363724

theorem imaginary_part_of_one_minus_i_squared : Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3637_363724


namespace NUMINAMATH_CALUDE_total_cost_is_985_l3637_363759

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℚ := 3.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_extra_cost : ℚ := 2.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℚ := bus_cost + (bus_cost + train_extra_cost)

/-- Theorem stating that the total cost of one train ride and one bus ride is $9.85 -/
theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_985_l3637_363759


namespace NUMINAMATH_CALUDE_income_percentage_l3637_363755

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mart = tim * 1.6) : 
  mart = juan * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_income_percentage_l3637_363755


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3637_363769

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ y => |y - 8| + 3 * y
  ∃ (y₁ y₂ : ℝ), y₁ = 23/4 ∧ y₂ = 7/2 ∧ f y₁ = 15 ∧ f y₂ = 15 ∧
    (∀ y : ℝ, f y = 15 → y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3637_363769


namespace NUMINAMATH_CALUDE_fraction_power_multiplication_l3637_363710

theorem fraction_power_multiplication :
  (3 / 5 : ℝ)^4 * (2 / 9 : ℝ)^(1/2) = 81 * Real.sqrt 2 / 1875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_multiplication_l3637_363710


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3637_363700

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Intersection points of a line with a hyperbola -/
def intersection (h : Hyperbola) (l : Line) : Set (ℝ × ℝ) :=
  sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (p q r : ℝ × ℝ) : Prop :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) :
  l.point = h.F₂ →
  ∃ (A B : ℝ × ℝ), A ∈ intersection h l ∧ B ∈ intersection h l ∧
  is_equilateral h.F₁ A B →
  eccentricity h = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3637_363700


namespace NUMINAMATH_CALUDE_discount_calculation_l3637_363773

-- Define the number of pens bought and the equivalent marked price
def pens_bought : ℕ := 50
def marked_price_equivalent : ℕ := 46

-- Define the profit percentage
def profit_percent : ℚ := 7608695652173914 / 100000000000000000

-- Define the discount percentage (to be proven)
def discount_percent : ℚ := 1 / 100

theorem discount_calculation :
  let cost_price := marked_price_equivalent
  let selling_price := cost_price * (1 + profit_percent)
  let discount := pens_bought - selling_price
  discount / pens_bought = discount_percent := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3637_363773


namespace NUMINAMATH_CALUDE_cab_driver_income_l3637_363780

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 600)
  (h3 : day3 = 450)
  (h4 : day4 = 400)
  (h5 : day5 = 800)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 500) :
  day2 = 250 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3637_363780


namespace NUMINAMATH_CALUDE_complex_number_location_l3637_363775

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l3637_363775


namespace NUMINAMATH_CALUDE_eight_reader_permutations_l3637_363737

theorem eight_reader_permutations : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_reader_permutations_l3637_363737


namespace NUMINAMATH_CALUDE_rational_function_equality_l3637_363764

theorem rational_function_equality (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 120*x + 3480) / (x^2 + 54*x - 2835)) →
  α + β = 123 := by
sorry

end NUMINAMATH_CALUDE_rational_function_equality_l3637_363764


namespace NUMINAMATH_CALUDE_polynomial_value_impossibility_l3637_363716

theorem polynomial_value_impossibility
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (h_poly : ∃ (Q : ℤ → ℤ), ∀ x, P x = Q x)  -- P is a polynomial
  (a b c d : ℤ)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_values : P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5)  -- P(a) = P(b) = P(c) = P(d) = 5
  : ¬ ∃ (k : ℤ), P k = 8 :=  -- There is no integer k such that P(k) = 8
by sorry

end NUMINAMATH_CALUDE_polynomial_value_impossibility_l3637_363716


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l3637_363732

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard with m rows and n columns -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_3 : m ≥ 3
  n_ge_3 : n ≥ 3

/-- Count of blue squares on the edges (excluding corners) of the chessboard -/
def count_edge_blue (board : Chessboard m n) : ℕ := sorry

/-- Count of standard pairs (adjacent squares with different colors) on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Main theorem: The number of standard pairs is odd iff the number of blue edge squares is odd -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (count_standard_pairs board) ↔ Odd (count_edge_blue board) := by sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l3637_363732


namespace NUMINAMATH_CALUDE_sum_of_basic_terms_divisible_by_four_l3637_363701

/-- A type representing a grid cell that can be either +1 or -1 -/
inductive GridCell
  | pos : GridCell
  | neg : GridCell

/-- A type representing an n × n grid filled with +1 or -1 -/
def Grid (n : ℕ) := Fin n → Fin n → GridCell

/-- A basic term is a product of n cells, no two of which share the same row or column -/
def BasicTerm (n : ℕ) (grid : Grid n) (perm : Equiv.Perm (Fin n)) : ℤ :=
  (Finset.univ.prod fun i => match grid i (perm i) with
    | GridCell.pos => 1
    | GridCell.neg => -1)

/-- The sum of all basic terms for a given grid -/
def SumOfBasicTerms (n : ℕ) (grid : Grid n) : ℤ :=
  (Finset.univ : Finset (Equiv.Perm (Fin n))).sum fun perm => BasicTerm n grid perm

/-- The main theorem: for any n × n grid (n ≥ 4), the sum of all basic terms is divisible by 4 -/
theorem sum_of_basic_terms_divisible_by_four {n : ℕ} (h : n ≥ 4) (grid : Grid n) :
  4 ∣ SumOfBasicTerms n grid := by
  sorry

end NUMINAMATH_CALUDE_sum_of_basic_terms_divisible_by_four_l3637_363701


namespace NUMINAMATH_CALUDE_sequence_problem_l3637_363783

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_sum : a 1 + a 5 + a 9 = 9)
  (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
  (a 2 + a 8) / (1 + b 2 * b 8) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3637_363783


namespace NUMINAMATH_CALUDE_triangle_translation_l3637_363740

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a translation vector
structure Translation :=
  (dx : ℝ)
  (dy : ℝ)

-- Define a function to apply a translation to a point
def translate (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

-- The main theorem
theorem triangle_translation
  (A B C A' : Point)
  (h_A : A = { x := -1, y := -4 })
  (h_B : B = { x := 1, y := 1 })
  (h_C : C = { x := -1, y := 4 })
  (h_A' : A' = { x := 1, y := -1 })
  (h_translation : ∃ t : Translation, translate A t = A') :
  ∃ (B' C' : Point),
    B' = { x := 3, y := 4 } ∧
    C' = { x := 1, y := 7 } ∧
    translate B h_translation.choose = B' ∧
    translate C h_translation.choose = C' :=
sorry

end NUMINAMATH_CALUDE_triangle_translation_l3637_363740


namespace NUMINAMATH_CALUDE_school_population_theorem_l3637_363796

theorem school_population_theorem (b g t : ℕ) : 
  b = 4 * g → g = 10 * t → b + g + t = 51 * t := by sorry

end NUMINAMATH_CALUDE_school_population_theorem_l3637_363796


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3637_363748

/-- 
Theorem: The largest value of n for which 6x^2 + nx + 72 can be factored 
as (6x + A)(x + B), where A and B are integers, is 433.
-/
theorem largest_n_for_factorization : 
  (∃ n : ℤ, ∀ m : ℤ, 
    (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + n * x + 72 = (6 * x + A) * (x + B)) ∧
    (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + m * x + 72 = (6 * x + A) * (x + B)) →
    m ≤ n) ∧
  (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + 433 * x + 72 = (6 * x + A) * (x + B)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3637_363748


namespace NUMINAMATH_CALUDE_christel_gave_five_dolls_l3637_363730

/-- The number of dolls Christel gave to Andrena -/
def dolls_given_by_christel : ℕ := sorry

theorem christel_gave_five_dolls :
  let debelyn_initial := 20
  let debelyn_gave := 2
  let christel_initial := 24
  let andrena_more_than_christel := 2
  let andrena_more_than_debelyn := 3
  dolls_given_by_christel = 5 := by sorry

end NUMINAMATH_CALUDE_christel_gave_five_dolls_l3637_363730


namespace NUMINAMATH_CALUDE_june_initial_stickers_l3637_363750

/-- The number of stickers June had initially -/
def june_initial : ℕ := 76

/-- The number of stickers Bonnie had initially -/
def bonnie_initial : ℕ := 63

/-- The number of stickers their grandparents gave to each of them -/
def gift : ℕ := 25

/-- The combined total of stickers after receiving the gifts -/
def total : ℕ := 189

theorem june_initial_stickers : 
  june_initial + gift + bonnie_initial + gift = total := by sorry

end NUMINAMATH_CALUDE_june_initial_stickers_l3637_363750


namespace NUMINAMATH_CALUDE_no_two_unique_digit_cubes_l3637_363712

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

def no_common_digits (n m : ℕ) : Prop :=
  (n.digits 10).toFinset ∩ (m.digits 10).toFinset = ∅

theorem no_two_unique_digit_cubes (kub : ℕ) 
  (h1 : is_three_digit_number kub)
  (h2 : has_unique_digits kub)
  (h3 : is_cube kub) :
  ¬ ∃ shar : ℕ, 
    is_three_digit_number shar ∧ 
    has_unique_digits shar ∧ 
    is_cube shar ∧ 
    no_common_digits kub shar :=
by sorry

end NUMINAMATH_CALUDE_no_two_unique_digit_cubes_l3637_363712


namespace NUMINAMATH_CALUDE_alice_burger_expense_l3637_363738

/-- The amount Alice spent on burgers in June -/
def aliceSpentOnBurgers (daysInJune : ℕ) (burgersPerDay : ℕ) (costPerBurger : ℕ) : ℕ :=
  daysInJune * burgersPerDay * costPerBurger

/-- Proof that Alice spent $1560 on burgers in June -/
theorem alice_burger_expense :
  aliceSpentOnBurgers 30 4 13 = 1560 := by
  sorry

end NUMINAMATH_CALUDE_alice_burger_expense_l3637_363738


namespace NUMINAMATH_CALUDE_yoojung_notebooks_l3637_363733

theorem yoojung_notebooks (initial : ℕ) : 
  (initial ≥ 5) →                        -- Ensure initial is at least 5
  (((initial - 5) / 2 : ℚ) = 4) →        -- Half of remaining after giving 5 equals 4
  (initial = 13) :=                      -- Prove initial is 13
sorry

end NUMINAMATH_CALUDE_yoojung_notebooks_l3637_363733


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3637_363723

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3637_363723


namespace NUMINAMATH_CALUDE_no_solution_exists_l3637_363761

theorem no_solution_exists (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * (y + z) + y * (z + x) = y * (z + x) + z * (x + y) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3637_363761


namespace NUMINAMATH_CALUDE_special_circle_properties_special_circle_unique_l3637_363725

/-- The circle passing through points A(1,-1) and B(-1,1) with its center on the line x+y-2=0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

theorem special_circle_properties :
  ∀ x y : ℝ,
    special_circle x y →
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∧
    ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                   (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2 :=
by
  sorry

theorem special_circle_unique :
  ∀ f : ℝ → ℝ → Prop,
    (∀ x y : ℝ, f x y → ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1))) →
    (∀ x y : ℝ, f x y → ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                                      (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2) →
    ∀ x y : ℝ, f x y ↔ special_circle x y :=
by
  sorry

end NUMINAMATH_CALUDE_special_circle_properties_special_circle_unique_l3637_363725


namespace NUMINAMATH_CALUDE_modulus_of_z_l3637_363751

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define z
noncomputable def z : ℂ := 4 / (1 + i)^4 - 3 * i

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3637_363751


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_right_triangle_l3637_363736

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if vectors m = (a+c, b) and n = (b, a-c) are parallel, then ABC is a right triangle -/
theorem parallel_vectors_imply_right_triangle 
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_parallel : (a + c) * (a - c) = b^2) :
  a^2 = b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_right_triangle_l3637_363736


namespace NUMINAMATH_CALUDE_tangent_line_at_M_l3637_363741

/-- The circle with equation x^2 + y^2 = 5 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5}

/-- The point M on the circle -/
def M : ℝ × ℝ := (2, -1)

/-- The proposed tangent line equation -/
def TangentLine (x y : ℝ) : Prop := 2*x - y - 5 = 0

/-- Theorem stating that the proposed line is tangent to the circle at M -/
theorem tangent_line_at_M :
  M ∈ Circle ∧
  TangentLine M.1 M.2 ∧
  ∀ p ∈ Circle, p ≠ M → ¬TangentLine p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_M_l3637_363741


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l3637_363704

theorem x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one :
  (∀ x : ℝ, x > 1 → 1 / x < 1) ∧
  (∃ x : ℝ, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l3637_363704


namespace NUMINAMATH_CALUDE_unique_composite_with_bounded_divisors_l3637_363794

def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def isProperDivisor (d n : ℕ) : Prop :=
  1 < d ∧ d < n ∧ n % d = 0

theorem unique_composite_with_bounded_divisors :
  ∃! n : ℕ, isComposite n ∧
    (∀ d : ℕ, isProperDivisor d n → n - 12 ≥ d ∧ d ≥ n - 20) ∧
    n = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_bounded_divisors_l3637_363794


namespace NUMINAMATH_CALUDE_max_value_expression_l3637_363747

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 3 + 9 * y * z ≤ Real.sqrt ((29 * 54) / 5) ∧
  ∃ (x_max y_max z_max : ℝ),
    x_max ≥ 0 ∧ y_max ≥ 0 ∧ z_max ≥ 0 ∧
    x_max^2 + y_max^2 + z_max^2 = 1 ∧
    3 * x_max * z_max * Real.sqrt 3 + 9 * y_max * z_max = Real.sqrt ((29 * 54) / 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3637_363747


namespace NUMINAMATH_CALUDE_construction_contract_l3637_363742

theorem construction_contract (H : ℕ) 
  (first_half : H * 3 / 5 = H - (300 + 500))
  (remaining : 500 = H - (H * 3 / 5 + 300)) : H = 2000 := by
  sorry

end NUMINAMATH_CALUDE_construction_contract_l3637_363742


namespace NUMINAMATH_CALUDE_ending_number_proof_l3637_363790

/-- The ending number of the range [100, n] where the average of integers
    in [100, n] is 100 greater than the average of integers in [50, 250]. -/
def ending_number : ℕ :=
  400

theorem ending_number_proof :
  ∃ (n : ℕ),
    n ≥ 100 ∧
    (n + 100) / 2 = (250 + 50) / 2 + 100 ∧
    n = ending_number :=
by sorry

end NUMINAMATH_CALUDE_ending_number_proof_l3637_363790


namespace NUMINAMATH_CALUDE_bouquet_cost_proportional_cost_of_25_lilies_l3637_363739

/-- The cost of a bouquet of lilies -/
def bouquet_cost (lilies : ℕ) : ℝ :=
  sorry

/-- The number of lilies in the first bouquet -/
def lilies₁ : ℕ := 15

/-- The cost of the first bouquet -/
def cost₁ : ℝ := 30

/-- The number of lilies in the second bouquet -/
def lilies₂ : ℕ := 25

theorem bouquet_cost_proportional :
  ∀ (n m : ℕ), n ≠ 0 → m ≠ 0 →
  bouquet_cost n / n = bouquet_cost m / m :=
  sorry

theorem cost_of_25_lilies :
  bouquet_cost lilies₂ = 50 :=
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_proportional_cost_of_25_lilies_l3637_363739


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3637_363714

theorem polynomial_factorization (a : ℝ) : a^3 + a^2 - a - 1 = (a - 1) * (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3637_363714


namespace NUMINAMATH_CALUDE_square_side_bounds_l3637_363722

/-- A triangle with an inscribed square and circle -/
structure TriangleWithInscriptions where
  /-- The side length of the inscribed square -/
  s : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The square is inscribed such that two vertices are on the base and two on the sides -/
  square_inscribed : True
  /-- The circle is inscribed in the triangle -/
  circle_inscribed : True
  /-- Both s and r are positive -/
  s_pos : 0 < s
  r_pos : 0 < r

/-- The side of the inscribed square is bounded by √2r and 2r -/
theorem square_side_bounds (t : TriangleWithInscriptions) : Real.sqrt 2 * t.r < t.s ∧ t.s < 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_square_side_bounds_l3637_363722


namespace NUMINAMATH_CALUDE_coin_distribution_formula_l3637_363744

/-- An arithmetic sequence representing the distribution of coins among people. -/
def CoinDistribution (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem coin_distribution_formula 
  (a₁ d : ℚ) 
  (h1 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) = 
        (CoinDistribution a₁ d 3) + (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5))
  (h2 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) + (CoinDistribution a₁ d 3) + 
        (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5) = 5) :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → CoinDistribution a₁ d n = -1/6 * n + 3/2 :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_formula_l3637_363744


namespace NUMINAMATH_CALUDE_odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l3637_363745

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Statement 1: f(x) is an odd function if and only if q = 0
theorem odd_function_iff (p q : ℝ) :
  (∀ x : ℝ, f p q (-x) = -(f p q x)) ↔ q = 0 := by sorry

-- Statement 2: The graph of f(x) is symmetric about the point (0, q)
theorem graph_symmetry (p q : ℝ) :
  ∀ x : ℝ, f p q (x) - q = -(f p q (-x) - q) := by sorry

-- Statement 3: When p = 0, the equation f(x) = 0 always has at least one solution
theorem solution_exists_when_p_zero (q : ℝ) :
  ∃ x : ℝ, f 0 q x = 0 := by sorry

-- Statement 4: There exists a combination of p and q such that f(x) = 0 has more than two solutions
theorem more_than_two_solutions :
  ∃ p q : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (f p q x₁ = 0 ∧ f p q x₂ = 0 ∧ f p q x₃ = 0) := by sorry

end NUMINAMATH_CALUDE_odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l3637_363745


namespace NUMINAMATH_CALUDE_distribute_5_3_l3637_363743

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

theorem distribute_5_3 : distribute num_balls num_boxes = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3637_363743


namespace NUMINAMATH_CALUDE_x_fifth_minus_ten_x_l3637_363760

theorem x_fifth_minus_ten_x (x : ℝ) : x = 5 → x^5 - 10*x = 3075 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_ten_x_l3637_363760


namespace NUMINAMATH_CALUDE_max_value_theorem_l3637_363774

theorem max_value_theorem (t x1 x2 : ℝ) : 
  t > 2 → x2 > x1 → x1 > 0 → 
  (Real.exp x1 - x1 = t) → (x2 - Real.log x2 = t) →
  (∃ (c : ℝ), c = Real.log t / (x2 - x1) ∧ c ≤ 1 / Real.exp 1 ∧ 
   ∀ (y1 y2 : ℝ), y2 > y1 → y1 > 0 → 
   (Real.exp y1 - y1 = t) → (y2 - Real.log y2 = t) →
   Real.log t / (y2 - y1) ≤ c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3637_363774


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3637_363709

/-- Given a principal amount that yields 202.50 interest at 4.5% rate, 
    prove that the rate yielding 225 interest on the same principal is 5% -/
theorem interest_rate_calculation (P : ℝ) : 
  P * 0.045 = 202.50 → P * (5 / 100) = 225 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3637_363709


namespace NUMINAMATH_CALUDE_reciprocal_of_hcf_24_182_l3637_363720

theorem reciprocal_of_hcf_24_182 : 
  let a : ℕ := 24
  let b : ℕ := 182
  let hcf := Nat.gcd a b
  1 / (hcf : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_hcf_24_182_l3637_363720


namespace NUMINAMATH_CALUDE_same_solution_implies_m_half_l3637_363735

theorem same_solution_implies_m_half 
  (h1 : ∃ x, 4*x + 2*m = 3*x + 1)
  (h2 : ∃ x, 3*x + 2*m = 6*x + 1)
  (h3 : ∃ x, (4*x + 2*m = 3*x + 1) ∧ (3*x + 2*m = 6*x + 1)) :
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_half_l3637_363735


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l3637_363763

/-- Represents a sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : Fin 5 → ℕ+) : Prop :=
  a 0 = 1 ∧
  (99 : ℚ) / 100 = (a 0 : ℚ) / a 1 + (a 1 : ℚ) / a 2 + (a 2 : ℚ) / a 3 + (a 3 : ℚ) / a 4 ∧
  ∀ k : Fin 3, ((a (k + 1) : ℕ) - 1) * (a (k - 1) : ℕ) ≥ (a k : ℕ)^2 * ((a k : ℕ) - 1)

/-- The theorem stating that there is only one valid sequence -/
theorem unique_valid_sequence :
  ∃! a : Fin 5 → ℕ+, ValidSequence a ∧
    a 0 = 1 ∧ a 1 = 2 ∧ a 2 = 5 ∧ a 3 = 56 ∧ a 4 = 25^2 * 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l3637_363763


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3637_363729

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 84) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3637_363729


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l3637_363715

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  p : ℕ
  o : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (alWeight pWeight oWeight : ℕ) : ℕ :=
  c.al * alWeight + c.p * pWeight + c.o * oWeight

/-- Theorem stating the relationship between the compound composition and its molecular weight -/
theorem compound_oxygen_atoms (alWeight pWeight oWeight : ℕ) (c : Compound) :
  alWeight = 27 ∧ pWeight = 31 ∧ oWeight = 16 ∧ c.al = 1 ∧ c.p = 1 →
  (molecularWeight c alWeight pWeight oWeight = 122 ↔ c.o = 4) :=
by sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l3637_363715


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3637_363707

theorem necessary_but_not_sufficient (x y : ℝ) :
  (¬ ((x > 3) ∨ (y > 2)) → ¬ (x + y > 5)) ∧
  ¬ ((x > 3) ∨ (y > 2) → (x + y > 5)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3637_363707


namespace NUMINAMATH_CALUDE_orange_bows_count_l3637_363793

theorem orange_bows_count (total : ℕ) (black : ℕ) : 
  black = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 12 + (black : ℚ) / total = 1 →
  (1 : ℚ) / 12 * total = 10 :=
by sorry

end NUMINAMATH_CALUDE_orange_bows_count_l3637_363793


namespace NUMINAMATH_CALUDE_triangle_BC_length_l3637_363765

/-- Triangle ABC with given properties --/
structure TriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BX : ℕ
  CX : ℕ
  h_AB : AB = 75
  h_AC : AC = 85
  h_BC : BC = BX + CX
  h_circle : BX^2 + CX^2 = AB^2

/-- Theorem: BC = 89 in the given triangle --/
theorem triangle_BC_length (t : TriangleABC) : t.BC = 89 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l3637_363765


namespace NUMINAMATH_CALUDE_watch_cost_price_l3637_363711

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℚ),
  (C * 88 / 100 : ℚ) + 140 = C * 104 / 100 ∧ C = 875 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3637_363711


namespace NUMINAMATH_CALUDE_battery_change_month_l3637_363770

/-- Given a 7-month interval between battery changes, starting in January,
    prove that the 15th change will occur in March. -/
theorem battery_change_month :
  let interval := 7  -- months between changes
  let start_month := 1  -- January
  let change_number := 15
  let total_months := interval * (change_number - 1)
  let years_passed := total_months / 12
  let extra_months := total_months % 12
  (start_month + extra_months - 1) % 12 + 1 = 3  -- 3 represents March
  := by sorry

end NUMINAMATH_CALUDE_battery_change_month_l3637_363770


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3637_363788

theorem min_value_quadratic (x y : ℝ) : 
  3 * x^2 + 2 * x * y + y^2 - 6 * x + 2 * y + 8 ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3637_363788


namespace NUMINAMATH_CALUDE_evening_painting_l3637_363731

/-- A dodecahedron is a polyhedron with 12 faces -/
def dodecahedron_faces : ℕ := 12

/-- The number of faces Samuel painted in the morning -/
def painted_faces : ℕ := 5

/-- The number of faces Samuel needs to paint in the evening -/
def remaining_faces : ℕ := dodecahedron_faces - painted_faces

theorem evening_painting : remaining_faces = 7 := by
  sorry

end NUMINAMATH_CALUDE_evening_painting_l3637_363731


namespace NUMINAMATH_CALUDE_largest_n_value_l3637_363727

/-- Represents a digit in base 8 or 9 -/
def Digit := Fin 9

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (a b c : Digit) : ℕ :=
  64 * a.val + 8 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (c b a : Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem largest_n_value (a b c : Digit) 
    (h1 : base8ToBase10 a b c = base9ToBase10 c b a)
    (h2 : isEven c.val)
    (h3 : ∀ x y z : Digit, 
      base8ToBase10 x y z = base9ToBase10 z y x → 
      isEven z.val → 
      base8ToBase10 x y z ≤ base8ToBase10 a b c) :
  base8ToBase10 a b c = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_l3637_363727


namespace NUMINAMATH_CALUDE_houses_before_boom_correct_l3637_363787

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County. -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built_during_boom : ℕ := 574

/-- Theorem stating that the number of houses before the boom
    plus the number of houses built during the boom
    equals the current number of houses. -/
theorem houses_before_boom_correct :
  houses_before_boom + houses_built_during_boom = current_houses :=
by sorry

end NUMINAMATH_CALUDE_houses_before_boom_correct_l3637_363787


namespace NUMINAMATH_CALUDE_cos_54_degrees_l3637_363706

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l3637_363706


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l3637_363719

def seven_digit_number (n : ℕ) : ℕ := 854 * 10000 + n * 1000 + 526

theorem divisibility_by_eleven (n : ℕ) : 
  (seven_digit_number n) % 11 = 0 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l3637_363719


namespace NUMINAMATH_CALUDE_lucas_payment_l3637_363789

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_3_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_earned := total_windows * payment_per_window
  let deductions := (days_taken / 3) * deduction_per_3_days
  total_earned - deductions

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 6 = 16 := by
  sorry

#eval calculate_payment 3 3 2 1 6

end NUMINAMATH_CALUDE_lucas_payment_l3637_363789


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l3637_363749

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l3637_363749


namespace NUMINAMATH_CALUDE_common_tangents_count_l3637_363772

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ := sorry

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- Theorem stating that the number of common tangents between C₁ and C₂ is 4 -/
theorem common_tangents_count : num_common_tangents C₁ C₂ = 4 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l3637_363772


namespace NUMINAMATH_CALUDE_ball_drawing_exclusivity_l3637_363766

structure Ball :=
  (color : String)

def Bag := Multiset Ball

def draw (bag : Bag) (n : ℕ) := Multiset Ball

def atLeastOneWhite (draw : Multiset Ball) : Prop := sorry
def bothWhite (draw : Multiset Ball) : Prop := sorry
def atLeastOneRed (draw : Multiset Ball) : Prop := sorry
def exactlyOneWhite (draw : Multiset Ball) : Prop := sorry
def exactlyTwoWhite (draw : Multiset Ball) : Prop := sorry
def bothRed (draw : Multiset Ball) : Prop := sorry

def mutuallyExclusive (e1 e2 : Multiset Ball → Prop) : Prop := sorry

def initialBag : Bag := sorry

theorem ball_drawing_exclusivity :
  let result := draw initialBag 2
  (mutuallyExclusive (exactlyOneWhite) (exactlyTwoWhite)) ∧
  (mutuallyExclusive (atLeastOneWhite) (bothRed)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (bothWhite)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (atLeastOneRed)) := by sorry

end NUMINAMATH_CALUDE_ball_drawing_exclusivity_l3637_363766


namespace NUMINAMATH_CALUDE_technicians_schedule_lcm_l3637_363703

theorem technicians_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_technicians_schedule_lcm_l3637_363703


namespace NUMINAMATH_CALUDE_valid_triangle_l3637_363721

/-- A triangle with side lengths a, b, and c satisfies the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of numbers (2, 3, 4) forms a valid triangle -/
theorem valid_triangle : is_triangle 2 3 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_triangle_l3637_363721


namespace NUMINAMATH_CALUDE_lottery_probability_l3637_363795

theorem lottery_probability (total_tickets : ℕ) (cash_prizes : ℕ) (merch_prizes : ℕ) :
  total_tickets = 1000 →
  cash_prizes = 5 →
  merch_prizes = 20 →
  (cash_prizes + merch_prizes : ℚ) / total_tickets = 25 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l3637_363795


namespace NUMINAMATH_CALUDE_fish_feeding_cost_l3637_363734

/-- Calculates the total cost to feed fish for 30 days given the specified conditions --/
theorem fish_feeding_cost :
  let goldfish_count : ℕ := 50
  let koi_count : ℕ := 30
  let guppies_count : ℕ := 20
  let goldfish_food : ℚ := 1.5
  let koi_food : ℚ := 2.5
  let guppies_food : ℚ := 0.75
  let goldfish_special_food_ratio : ℚ := 0.25
  let koi_special_food_ratio : ℚ := 0.4
  let guppies_special_food_ratio : ℚ := 0.1
  let special_food_cost_goldfish : ℚ := 3
  let special_food_cost_others : ℚ := 4
  let regular_food_cost : ℚ := 2
  let days : ℕ := 30

  (goldfish_count * goldfish_food * (goldfish_special_food_ratio * special_food_cost_goldfish +
    (1 - goldfish_special_food_ratio) * regular_food_cost) +
   koi_count * koi_food * (koi_special_food_ratio * special_food_cost_others +
    (1 - koi_special_food_ratio) * regular_food_cost) +
   guppies_count * guppies_food * (guppies_special_food_ratio * special_food_cost_others +
    (1 - guppies_special_food_ratio) * regular_food_cost)) * days = 12375 :=
by sorry


end NUMINAMATH_CALUDE_fish_feeding_cost_l3637_363734


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l3637_363705

def point_guard_time : ℕ := 130
def shooting_guard_time : ℕ := 145
def small_forward_time : ℕ := 85
def power_forward_time : ℕ := 60
def center_time : ℕ := 180
def number_of_players : ℕ := 5

def total_time : ℕ := point_guard_time + shooting_guard_time + small_forward_time + power_forward_time + center_time

theorem highlight_film_average_time :
  (total_time / number_of_players : ℚ) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l3637_363705


namespace NUMINAMATH_CALUDE_evaluate_expression_l3637_363767

theorem evaluate_expression : (-1 : ℤ) ^ (3^3) + 1 ^ (3^3) = 0 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3637_363767


namespace NUMINAMATH_CALUDE_sabrina_cookies_left_l3637_363754

/-- Calculates the number of cookies Sabrina has left after a series of transactions -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (fathers_cookies : ℕ) : ℕ :=
  let after_brother := initial - to_brother
  let from_mother := 3 * to_brother
  let after_mother := after_brother + from_mother
  let to_sister := after_mother / 3
  let after_sister := after_mother - to_sister
  let from_father := fathers_cookies / 4
  let after_father := after_sister + from_father
  let to_cousin := after_father / 2
  after_father - to_cousin

/-- Theorem stating that Sabrina is left with 18 cookies -/
theorem sabrina_cookies_left :
  cookies_left 28 10 16 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_cookies_left_l3637_363754


namespace NUMINAMATH_CALUDE_wang_house_number_l3637_363708

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 40 ∧ a > 0 ∧ b > 0 ∧ c > 0

def house_number (a b c : ℕ) : ℕ := a + b + c

def is_ambiguous (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂, 
    is_valid_triplet a₁ b₁ c₁ ∧ 
    is_valid_triplet a₂ b₂ c₂ ∧ 
    house_number a₁ b₁ c₁ = n ∧ 
    house_number a₂ b₂ c₂ = n ∧ 
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem wang_house_number : 
  ∃! n, is_ambiguous n ∧ ∀ m, is_ambiguous m → m = n :=
by
  sorry

end NUMINAMATH_CALUDE_wang_house_number_l3637_363708


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_1_over_3020_l3637_363728

def Q (n : ℕ+) : ℚ := (Nat.factorial (3*n-1)) / (Nat.factorial (3*n+1))

theorem smallest_n_for_Q_less_than_1_over_3020 :
  ∀ k : ℕ+, k < 19 → Q k ≥ 1/3020 ∧ Q 19 < 1/3020 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_1_over_3020_l3637_363728


namespace NUMINAMATH_CALUDE_constant_term_product_l3637_363785

-- Define polynomials p, q, r, and s
variable (p q r s : ℝ[X])

-- Define the relationship between s, p, q, and r
axiom h1 : s = p * q * r

-- Define the constant term of p as 2
axiom h2 : p.coeff 0 = 2

-- Define the constant term of s as 6
axiom h3 : s.coeff 0 = 6

-- Theorem to prove
theorem constant_term_product : q.coeff 0 * r.coeff 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l3637_363785


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3637_363784

-- Define sets A and B
def A : Set ℝ := {x | x * (x - 2) < 3}
def B : Set ℝ := {x | 5 / (x + 1) ≥ 1}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3637_363784


namespace NUMINAMATH_CALUDE_composition_ratio_l3637_363771

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 3 * x - 2

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 41 / 31 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3637_363771


namespace NUMINAMATH_CALUDE_vector_operation_l3637_363762

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (2, -2)) :
  2 • a - b = (4, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3637_363762


namespace NUMINAMATH_CALUDE_sandy_lemonade_sales_l3637_363702

theorem sandy_lemonade_sales (sunday_half_dollars : ℕ) (total_amount : ℚ) (half_dollar_value : ℚ) :
  sunday_half_dollars = 6 →
  total_amount = 11.5 →
  half_dollar_value = 0.5 →
  (total_amount - sunday_half_dollars * half_dollar_value) / half_dollar_value = 17 := by
sorry

end NUMINAMATH_CALUDE_sandy_lemonade_sales_l3637_363702


namespace NUMINAMATH_CALUDE_solution_difference_l3637_363757

theorem solution_difference (p q : ℝ) : 
  (p - 3) * (p + 3) = 21 * p - 63 →
  (q - 3) * (q + 3) = 21 * q - 63 →
  p ≠ q →
  p > q →
  p - q = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3637_363757


namespace NUMINAMATH_CALUDE_expression_simplification_l3637_363777

theorem expression_simplification 
  (a c d x y : ℝ) 
  (h : c * x + d * y ≠ 0) : 
  (c * x * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * y^2) + 
   d * y * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / 
  (c * x + d * y) = 
  a^2 * x^2 + 3 * a * c * x * y + c^2 * y^2 := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l3637_363777


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l3637_363798

/-- Prove that the length of the platform is equal to the length of the train -/
theorem train_platform_length_equality 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (train_length : ℝ) 
  (h1 : train_speed = 90 * 1000 / 60) -- 90 km/hr converted to m/min
  (h2 : crossing_time = 1) -- 1 minute
  (h3 : train_length = 750) -- 750 meters
  : train_length = train_speed * crossing_time - train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_equality_l3637_363798


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3637_363756

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (1 : ℝ) / 4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l3637_363756


namespace NUMINAMATH_CALUDE_total_distance_right_triangle_l3637_363786

/-- The total distance traveled in a right-angled triangle XYZ -/
theorem total_distance_right_triangle (XZ YZ XY : ℝ) : 
  XZ = 4000 →
  XY = 5000 →
  XZ^2 + YZ^2 = XY^2 →
  XZ + YZ + XY = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_right_triangle_l3637_363786


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3637_363779

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3637_363779


namespace NUMINAMATH_CALUDE_line_through_points_l3637_363718

/-- Theorem: For a line y = ax + b passing through points (3, 4) and (10, 22), a - b = 6 2/7 -/
theorem line_through_points (a b : ℚ) : 
  (4 : ℚ) = a * 3 + b ∧ (22 : ℚ) = a * 10 + b → a - b = (44 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3637_363718


namespace NUMINAMATH_CALUDE_range_of_m_for_positive_functions_l3637_363713

theorem range_of_m_for_positive_functions (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * m * x - 8 * x + 9 > 0) ∨ (m * x - m > 0)) →
  (0 < m ∧ m < 8) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_positive_functions_l3637_363713


namespace NUMINAMATH_CALUDE_prob_b_leads_2to1_expected_score_b_l3637_363778

/-- Represents a table tennis game between player A and player B -/
structure TableTennisGame where
  /-- Probability of the server scoring a point -/
  serverWinProb : ℝ
  /-- Player A serves first -/
  aServesFirst : Bool

/-- Calculates the probability of player B leading 2-1 at the start of the fourth serve -/
def probBLeads2to1 (game : TableTennisGame) : ℝ := sorry

/-- Calculates the expected score of player B at the start of the fourth serve -/
def expectedScoreB (game : TableTennisGame) : ℝ := sorry

/-- Theorem stating the probability of player B leading 2-1 at the start of the fourth serve -/
theorem prob_b_leads_2to1 (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  probBLeads2to1 game = 0.352 := by sorry

/-- Theorem stating the expected score of player B at the start of the fourth serve -/
theorem expected_score_b (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  expectedScoreB game = 1.400 := by sorry

end NUMINAMATH_CALUDE_prob_b_leads_2to1_expected_score_b_l3637_363778


namespace NUMINAMATH_CALUDE_weight_ratio_after_loss_l3637_363792

def jakes_current_weight : ℝ := 152
def combined_weight : ℝ := 212
def weight_loss : ℝ := 32

def sisters_weight : ℝ := combined_weight - jakes_current_weight
def jakes_new_weight : ℝ := jakes_current_weight - weight_loss

theorem weight_ratio_after_loss : 
  jakes_new_weight / sisters_weight = 2 := by sorry

end NUMINAMATH_CALUDE_weight_ratio_after_loss_l3637_363792


namespace NUMINAMATH_CALUDE_calories_burned_per_mile_l3637_363726

/-- Represents the calories burned per mile walked -/
def calories_per_mile : ℝ := sorry

/-- The total distance walked in miles -/
def total_distance : ℝ := 3

/-- The calories in the candy bar -/
def candy_bar_calories : ℝ := 200

/-- The net calorie deficit -/
def net_deficit : ℝ := 250

theorem calories_burned_per_mile :
  calories_per_mile * total_distance - candy_bar_calories = net_deficit ∧
  calories_per_mile = 150 := by sorry

end NUMINAMATH_CALUDE_calories_burned_per_mile_l3637_363726


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3637_363768

def num_apples : ℕ := 7
def num_oranges : ℕ := 12
def min_fruits_per_basket : ℕ := 2

-- Function to calculate the number of valid fruit baskets
def count_valid_baskets (apples oranges min_fruits : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - (1 + apples + oranges)

-- Theorem stating that the number of valid fruit baskets is 101
theorem fruit_basket_count :
  count_valid_baskets num_apples num_oranges min_fruits_per_basket = 101 := by
  sorry

#eval count_valid_baskets num_apples num_oranges min_fruits_per_basket

end NUMINAMATH_CALUDE_fruit_basket_count_l3637_363768


namespace NUMINAMATH_CALUDE_roberto_outfits_l3637_363717

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets : ℕ) (incompatible_combinations : ℕ) : ℕ :=
  trousers * shirts * jackets - incompatible_combinations * shirts

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let incompatible_combinations : ℕ := 1
  number_of_outfits trousers shirts jackets incompatible_combinations = 133 := by
sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3637_363717


namespace NUMINAMATH_CALUDE_certain_number_value_l3637_363776

theorem certain_number_value (t b c : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 → 
  ∃ x : ℝ, (t + b + c + x) / 4 = 15 ∧ x = 29 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l3637_363776


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3637_363797

theorem quadratic_equation_solution (t s : ℝ) : t = 15 * s^2 + 5 → t = 20 → s = 1 ∨ s = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3637_363797


namespace NUMINAMATH_CALUDE_quadratic_function_conditions_l3637_363758

/-- A quadratic function passing through (1,-4) with vertex at (-1,0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f satisfies the given conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, f x = a * (x + 1)^2) := by
  sorry

#check quadratic_function_conditions

end NUMINAMATH_CALUDE_quadratic_function_conditions_l3637_363758


namespace NUMINAMATH_CALUDE_inequality_proof_l3637_363781

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b * 2^a + a * 2^(-b) ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3637_363781


namespace NUMINAMATH_CALUDE_train_acceleration_time_l3637_363791

/-- Proves that a train starting from rest, accelerating uniformly at 3 m/s², 
    and traveling a distance of 27 m takes sqrt(18) seconds. -/
theorem train_acceleration_time : ∀ (s a : ℝ),
  s = 27 →  -- distance traveled
  a = 3 →   -- acceleration rate
  ∃ t : ℝ,
    s = (1/2) * a * t^2 ∧  -- kinematic equation for uniform acceleration from rest
    t = Real.sqrt 18 := by
  sorry

end NUMINAMATH_CALUDE_train_acceleration_time_l3637_363791
