import Mathlib

namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1342_134247

theorem condition_sufficient_not_necessary :
  (∀ k : ℤ, Real.sin (2 * k * Real.pi + Real.pi / 4) = Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, Real.sin x = Real.sqrt 2 / 2 ∧ ∀ k : ℤ, x ≠ 2 * k * Real.pi + Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1342_134247


namespace NUMINAMATH_CALUDE_boats_geometric_sum_l1342_134221

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boats_geometric_sum :
  geometric_sum 5 3 5 = 605 := by
  sorry

end NUMINAMATH_CALUDE_boats_geometric_sum_l1342_134221


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1342_134273

/-- The smallest natural number that is divisible by 55 and has exactly 117 distinct divisors -/
def smallest_number : ℕ := 12390400

/-- Count the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  smallest_number % 55 = 0 ∧
  count_divisors smallest_number = 117 ∧
  ∀ m : ℕ, m < smallest_number → (m % 55 = 0 ∧ count_divisors m = 117) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1342_134273


namespace NUMINAMATH_CALUDE_division_problem_l1342_134292

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : remainder = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 43 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1342_134292


namespace NUMINAMATH_CALUDE_square_root_equation_l1342_134241

theorem square_root_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1342_134241


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1342_134216

-- Define the sets A and B
def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1342_134216


namespace NUMINAMATH_CALUDE_girl_boy_ratio_l1342_134202

/-- Represents the number of students in the class -/
def total_students : ℕ := 28

/-- Represents the difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 4

/-- Theorem stating that the ratio of girls to boys is 4:3 -/
theorem girl_boy_ratio :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    girls = boys + girl_boy_difference ∧
    girls * 3 = boys * 4 :=
by sorry

end NUMINAMATH_CALUDE_girl_boy_ratio_l1342_134202


namespace NUMINAMATH_CALUDE_polynomial_intersection_l1342_134200

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_intersection (a b c d : ℝ) :
  (∃ (x : ℝ), f a b x = g c d x ∧ x = 50 ∧ f a b x = -200) →
  (g c d (-a/2) = 0) →
  (f a b (-c/2) = 0) →
  (∃ (m : ℝ), (∀ (x : ℝ), f a b x ≥ m) ∧ (∀ (x : ℝ), g c d x ≥ m) ∧
               (∃ (x₁ x₂ : ℝ), f a b x₁ = m ∧ g c d x₂ = m)) →
  a + c = -200 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l1342_134200


namespace NUMINAMATH_CALUDE_range_of_x_l1342_134240

theorem range_of_x (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) :
  x > 1 / 3 ∨ x < -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1342_134240


namespace NUMINAMATH_CALUDE_ordering_abc_l1342_134208

theorem ordering_abc : 
  let a : ℝ := 1/11
  let b : ℝ := Real.sqrt (1/10)
  let c : ℝ := Real.log (11/10)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1342_134208


namespace NUMINAMATH_CALUDE_remainder_problem_l1342_134248

theorem remainder_problem (k : ℕ+) (h : 60 % k.val ^ 2 = 6) : 100 % k.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1342_134248


namespace NUMINAMATH_CALUDE_union_nonempty_iff_in_range_l1342_134265

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | x^2 + (2*a - 3)*x + 2*a^2 - a - 3 = 0}

-- Define the set A (inferred from the problem)
def A (a : ℝ) : Set ℝ := {x | x^2 - (a - 2)*x - 2*a + 4 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a ≤ -6 ∨ (-7/2 ≤ a ∧ a ≤ 3/2) ∨ a ≥ 2}

-- Theorem statement
theorem union_nonempty_iff_in_range (a : ℝ) :
  (A a ∪ B a).Nonempty ↔ a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_union_nonempty_iff_in_range_l1342_134265


namespace NUMINAMATH_CALUDE_quadrilateral_area_implies_k_l1342_134253

/-- A quadrilateral with vertices A(0,3), B(0,k), C(5,10), and D(5,0) -/
structure Quadrilateral (k : ℝ) :=
  (A : ℝ × ℝ := (0, 3))
  (B : ℝ × ℝ := (0, k))
  (C : ℝ × ℝ := (5, 10))
  (D : ℝ × ℝ := (5, 0))

/-- The area of a quadrilateral -/
def area (q : Quadrilateral k) : ℝ :=
  sorry

/-- Theorem stating that if k > 3 and the area of the quadrilateral is 50, then k = 13 -/
theorem quadrilateral_area_implies_k (k : ℝ) (q : Quadrilateral k)
    (h1 : k > 3)
    (h2 : area q = 50) :
  k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_implies_k_l1342_134253


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l1342_134233

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The roots of the quadratic equation x^2 - 73x + k = 0 -/
def roots (k : ℕ) : Set ℝ :=
  {x : ℝ | x^2 - 73*x + k = 0}

/-- The statement that both roots of x^2 - 73x + k = 0 are prime numbers -/
def both_roots_prime (k : ℕ) : Prop :=
  ∀ x ∈ roots k, ∃ n : ℕ, (x : ℝ) = n ∧ is_prime n

/-- There is exactly one value of k such that both roots of x^2 - 73x + k = 0 are prime numbers -/
theorem unique_k_for_prime_roots : ∃! k : ℕ, both_roots_prime k :=
  sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l1342_134233


namespace NUMINAMATH_CALUDE_max_value_of_f_l1342_134220

-- Define the function f(x) = -3x^2 + 9
def f (x : ℝ) : ℝ := -3 * x^2 + 9

-- Theorem stating that the maximum value of f(x) is 9
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l1342_134220


namespace NUMINAMATH_CALUDE_table_height_proof_l1342_134222

/-- Given two configurations of stacked blocks on a table, prove that the table height is 34 inches -/
theorem table_height_proof (r s b : ℝ) (hr : r = 40) (hs : s = 34) (hb : b = 6) :
  ∃ (h l w : ℝ), h = 34 ∧ l + h - w = r ∧ w + h - l + b = s := by
  sorry

end NUMINAMATH_CALUDE_table_height_proof_l1342_134222


namespace NUMINAMATH_CALUDE_problem_statement_l1342_134296

theorem problem_statement (x y : ℝ) (h : (x + 2*y)^3 + x^3 + 2*x + 2*y = 0) : 
  x + y - 1 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1342_134296


namespace NUMINAMATH_CALUDE_assignment_schemes_with_girl_count_l1342_134252

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The number of tasks to be assigned -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of assignment schemes -/
def assignment_schemes (b g s t : ℕ) : ℕ :=
  Nat.descFactorial (b + g) s - Nat.descFactorial b s

/-- Theorem stating that the number of assignment schemes with at least one girl is 186 -/
theorem assignment_schemes_with_girl_count :
  assignment_schemes num_boys num_girls num_selected num_tasks = 186 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_with_girl_count_l1342_134252


namespace NUMINAMATH_CALUDE_total_income_calculation_l1342_134264

/-- Calculates the total income for a clothing store sale --/
def calculate_total_income (tshirt_price : ℚ) (pants_price : ℚ) (skirt_price : ℚ) 
                           (refurbished_tshirt_price : ℚ) (skirt_discount_rate : ℚ) 
                           (tshirt_discount_rate : ℚ) (sales_tax_rate : ℚ) 
                           (tshirts_sold : ℕ) (refurbished_tshirts_sold : ℕ) 
                           (pants_sold : ℕ) (skirts_sold : ℕ) : ℚ :=
  sorry

theorem total_income_calculation :
  let tshirt_price : ℚ := 5
  let pants_price : ℚ := 4
  let skirt_price : ℚ := 6
  let refurbished_tshirt_price : ℚ := tshirt_price / 2
  let skirt_discount_rate : ℚ := 1 / 10
  let tshirt_discount_rate : ℚ := 1 / 5
  let sales_tax_rate : ℚ := 2 / 25
  let tshirts_sold : ℕ := 15
  let refurbished_tshirts_sold : ℕ := 7
  let pants_sold : ℕ := 6
  let skirts_sold : ℕ := 12
  calculate_total_income tshirt_price pants_price skirt_price refurbished_tshirt_price 
                         skirt_discount_rate tshirt_discount_rate sales_tax_rate
                         tshirts_sold refurbished_tshirts_sold pants_sold skirts_sold = 1418 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_total_income_calculation_l1342_134264


namespace NUMINAMATH_CALUDE_book_selection_problem_l1342_134299

theorem book_selection_problem (total_books math_books physics_books selected_books selected_math selected_physics : ℕ) :
  total_books = 20 →
  math_books = 6 →
  physics_books = 4 →
  selected_books = 8 →
  selected_math = 4 →
  selected_physics = 2 →
  (Nat.choose math_books selected_math) * (Nat.choose physics_books selected_physics) *
  (Nat.choose (total_books - math_books - physics_books) (selected_books - selected_math - selected_physics)) = 4050 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_problem_l1342_134299


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1342_134255

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x = 0}

def N : Set ℝ := {x | x - 1 > 0}

theorem intersection_M_complement_N : M ∩ (U \ N) = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1342_134255


namespace NUMINAMATH_CALUDE_lollipop_distribution_theorem_l1342_134295

/-- Represents the lollipop distribution rules and class attendance --/
structure LollipopDistribution where
  mainTeacherRatio : ℕ  -- Students per lollipop for main teacher
  assistantRatio : ℕ    -- Students per lollipop for assistant
  assistantThreshold : ℕ -- Threshold for assistant to start giving lollipops
  initialStudents : ℕ   -- Initial number of students
  lateStudents : List ℕ  -- List of additional students joining later

/-- Calculates the total number of lollipops given away --/
def totalLollipops (d : LollipopDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, 21 lollipops will be given away --/
theorem lollipop_distribution_theorem :
  let d : LollipopDistribution := {
    mainTeacherRatio := 5,
    assistantRatio := 7,
    assistantThreshold := 30,
    initialStudents := 45,
    lateStudents := [10, 5, 5]
  }
  totalLollipops d = 21 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_distribution_theorem_l1342_134295


namespace NUMINAMATH_CALUDE_trivia_team_group_size_l1342_134224

theorem trivia_team_group_size 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 35) 
  (h2 : unpicked_students = 11) 
  (h3 : num_groups = 4) :
  (total_students - unpicked_students) / num_groups = 6 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_group_size_l1342_134224


namespace NUMINAMATH_CALUDE_smallest_value_of_x_plus_yz_l1342_134235

theorem smallest_value_of_x_plus_yz (x y z : ℕ+) (h : x * y + z = 160) :
  ∃ (a b c : ℕ+), a * b + c = 160 ∧ a + b * c = 64 ∧ ∀ (p q r : ℕ+), p * q + r = 160 → p + q * r ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_x_plus_yz_l1342_134235


namespace NUMINAMATH_CALUDE_cos_theta_plus_5pi_6_l1342_134219

theorem cos_theta_plus_5pi_6 (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ / 2 + π / 6) = 2 / 3) : 
  Real.cos (θ + 5 * π / 6) = -4 * Real.sqrt 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_5pi_6_l1342_134219


namespace NUMINAMATH_CALUDE_candy_remaining_l1342_134275

theorem candy_remaining (initial : ℝ) (talitha_took : ℝ) (solomon_took : ℝ) (maya_took : ℝ)
  (h1 : initial = 1012.5)
  (h2 : talitha_took = 283.7)
  (h3 : solomon_took = 398.2)
  (h4 : maya_took = 197.6) :
  initial - (talitha_took + solomon_took + maya_took) = 133 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l1342_134275


namespace NUMINAMATH_CALUDE_nested_radical_solution_l1342_134288

theorem nested_radical_solution :
  ∃! x : ℝ, x > 0 ∧ x = Real.sqrt (3 + x) :=
by
  use (1 + Real.sqrt 13) / 2
  sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l1342_134288


namespace NUMINAMATH_CALUDE_circle_circumference_ratio_l1342_134218

/-- The ratio of the new circumference to the original circumference when the radius is increased by 2 units -/
theorem circle_circumference_ratio (r : ℝ) (h : r > 0) :
  (2 * Real.pi * (r + 2)) / (2 * Real.pi * r) = 1 + 2 / r := by
  sorry

#check circle_circumference_ratio

end NUMINAMATH_CALUDE_circle_circumference_ratio_l1342_134218


namespace NUMINAMATH_CALUDE_inscribed_triangle_condition_l1342_134210

/-- A rectangle with side lengths a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- An equilateral triangle inscribed in a rectangle such that one vertex is at A
    and the other two vertices lie on sides BC and CD respectively. -/
structure InscribedTriangle (rect : Rectangle) where
  vertex_on_BC : ℝ
  vertex_on_CD : ℝ
  vertex_on_BC_in_range : 0 ≤ vertex_on_BC ∧ vertex_on_BC ≤ rect.b
  vertex_on_CD_in_range : 0 ≤ vertex_on_CD ∧ vertex_on_CD ≤ rect.a
  is_equilateral : True  -- We assume this condition is met

/-- The theorem stating the condition for inscribing an equilateral triangle in a rectangle. -/
theorem inscribed_triangle_condition (rect : Rectangle) :
  (∃ t : InscribedTriangle rect, True) ↔ 
  (Real.sqrt 3 / 2 ≤ rect.a / rect.b ∧ rect.a / rect.b ≤ 2 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_condition_l1342_134210


namespace NUMINAMATH_CALUDE_number_equals_twenty_l1342_134263

theorem number_equals_twenty : ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_twenty_l1342_134263


namespace NUMINAMATH_CALUDE_min_sum_squares_l1342_134206

theorem min_sum_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  ∃ (m : ℝ), (∀ (x y z w : ℝ), x + 3*y + 5*z + 7*w = 14 → x^2 + y^2 + z^2 + w^2 ≥ m) ∧
             (a^2 + b^2 + c^2 + d^2 = m) ∧
             (m = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1342_134206


namespace NUMINAMATH_CALUDE_john_yearly_expenses_l1342_134272

/-- Calculates the total amount John needs to pay for his EpiPens and additional medical expenses for a year. -/
def total_yearly_expenses (epipen_cost : ℚ) (first_epipen_coverage : ℚ) (second_epipen_coverage : ℚ) (yearly_medical_expenses : ℚ) (medical_expenses_coverage : ℚ) : ℚ :=
  let first_epipen_payment := epipen_cost * (1 - first_epipen_coverage)
  let second_epipen_payment := epipen_cost * (1 - second_epipen_coverage)
  let total_epipen_cost := first_epipen_payment + second_epipen_payment
  let medical_expenses_payment := yearly_medical_expenses * (1 - medical_expenses_coverage)
  total_epipen_cost + medical_expenses_payment

/-- Theorem stating that John's total yearly expenses are $725 given the problem conditions. -/
theorem john_yearly_expenses :
  total_yearly_expenses 500 0.75 0.6 2000 0.8 = 725 := by
  sorry

end NUMINAMATH_CALUDE_john_yearly_expenses_l1342_134272


namespace NUMINAMATH_CALUDE_square_of_integer_l1342_134285

theorem square_of_integer (x y z : ℤ) (A : ℤ) 
  (h1 : A = x * y + y * z + z * x)
  (h2 : 4 * x + y + z = 0) : 
  ∃ (k : ℤ), (-1) * A = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l1342_134285


namespace NUMINAMATH_CALUDE_circles_tangent_sum_l1342_134278

-- Define the circles and line
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 1
def circle_C2 (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the external tangency condition
def externally_tangent (a b : ℝ) : Prop := (a - 1)^2 + (b + 3)^2 > 4

-- Define the equal tangent length condition
def equal_tangent_length (a b : ℝ) : Prop := 
  ∃ m : ℝ, (4 + 2*a + 2*b)*m + 5 - a^2 - (1 + b)^2 = 0

-- State the theorem
theorem circles_tangent_sum (a b : ℝ) :
  externally_tangent a b →
  equal_tangent_length a b →
  a + b = -2 := by sorry

end NUMINAMATH_CALUDE_circles_tangent_sum_l1342_134278


namespace NUMINAMATH_CALUDE_third_measurement_is_integer_meters_l1342_134291

def tape_length : ℕ := 100
def length1 : ℕ := 600
def length2 : ℕ := 500

theorem third_measurement_is_integer_meters :
  ∃ (k : ℕ), ∀ (third_length : ℕ),
    (tape_length ∣ length1) ∧
    (tape_length ∣ length2) ∧
    (tape_length ∣ third_length) →
    ∃ (n : ℕ), third_length = n * 100 := by
  sorry

end NUMINAMATH_CALUDE_third_measurement_is_integer_meters_l1342_134291


namespace NUMINAMATH_CALUDE_middle_card_is_five_l1342_134242

/-- Represents a set of three cards with distinct positive integers. -/
structure CardSet where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  distinct : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  ascending : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Predicate for Ada's statement about the leftmost card -/
def ada_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.left = cs.left ∧ cs' ≠ cs

/-- Predicate for Bella's statement about the rightmost card -/
def bella_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.right = cs.right ∧ cs' ≠ cs

/-- The main theorem stating that the middle card must be 5 -/
theorem middle_card_is_five :
  ∀ cs : CardSet,
    ada_statement cs →
    bella_statement cs →
    cs.middle = 5 :=
sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l1342_134242


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1342_134234

/-- Given a 25% reduction in oil price, prove the reduced price per kg is 30 Rs. --/
theorem oil_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let reduced_price := 0.75 * original_price
  (600 / reduced_price) = (600 / original_price) + 5 →
  reduced_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1342_134234


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1342_134257

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,3}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1342_134257


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1342_134270

theorem fixed_point_of_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1342_134270


namespace NUMINAMATH_CALUDE_white_bread_loaves_l1342_134267

/-- Given that a restaurant served 0.2 loaf of wheat bread and 0.6 loaves in total,
    prove that the number of loaves of white bread served is 0.4. -/
theorem white_bread_loaves (wheat_bread : Real) (total_bread : Real)
    (h1 : wheat_bread = 0.2)
    (h2 : total_bread = 0.6) :
    total_bread - wheat_bread = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_white_bread_loaves_l1342_134267


namespace NUMINAMATH_CALUDE_f_property_l1342_134217

def property_P (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k + 1 < n ∧
  2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)

theorem f_property :
  (property_P 7) ∧
  (∀ n : ℕ, n ≤ 2016 → property_P n → n ≤ 1934) ∧
  (property_P 1934) :=
sorry

end NUMINAMATH_CALUDE_f_property_l1342_134217


namespace NUMINAMATH_CALUDE_expression_evaluation_l1342_134245

/-- Evaluates the given expression for x = 1.5 and y = -2 -/
theorem expression_evaluation :
  let x : ℝ := 1.5
  let y : ℝ := -2
  let expr := (1.2 * x^3 + 4 * y) * (0.86)^3 - (0.1)^3 / (0.86)^2 + 0.086 + (0.1)^2 * (2 * x^2 - 3 * y^2)
  ∃ ε > 0, |expr + 2.5027737774| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1342_134245


namespace NUMINAMATH_CALUDE_quadratic_solution_l1342_134239

theorem quadratic_solution (x a : ℝ) : x = 3 ∧ x^2 = a → a = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1342_134239


namespace NUMINAMATH_CALUDE_line_opposite_sides_range_l1342_134237

/-- The range of 'a' for a line x + y - a = 0 with (0, 0) and (1, 1) on opposite sides -/
theorem line_opposite_sides_range (a : ℝ) : 
  (∀ x y : ℝ, x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_line_opposite_sides_range_l1342_134237


namespace NUMINAMATH_CALUDE_triangle_side_indeterminate_l1342_134261

/-- Given a triangle ABC with AB = 3 and AC = 2, the length of BC cannot be uniquely determined. -/
theorem triangle_side_indeterminate (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A B = 3) → (d A C = 2) → 
  ¬∃! x : ℝ, d B C = x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_indeterminate_l1342_134261


namespace NUMINAMATH_CALUDE_max_weeks_correct_l1342_134271

/-- Represents a weekly ranking of 10 songs -/
def Ranking := Fin 10 → Fin 10

/-- The maximum number of weeks the same 10 songs can remain in the ranking -/
def max_weeks : ℕ := 46

/-- A function that represents the ranking change from one week to the next -/
def next_week (r : Ranking) : Ranking := sorry

/-- Predicate to check if a song's ranking has dropped -/
def has_dropped (r1 r2 : Ranking) (song : Fin 10) : Prop :=
  r2 song > r1 song

theorem max_weeks_correct (initial : Ranking) :
  ∀ (sequence : ℕ → Ranking),
    (∀ n, sequence (n + 1) = next_week (sequence n)) →
    (∀ n, sequence n ≠ sequence (n + 1)) →
    (∀ n m song, n < m → has_dropped (sequence n) (sequence m) song →
      ∀ k > m, has_dropped (sequence m) (sequence k) song ∨ sequence m song = sequence k song) →
    (∃ n ≤ max_weeks, ∃ song, sequence 0 song ≠ sequence n song) ∧
    (∀ n > max_weeks, ∃ song, sequence 0 song ≠ sequence n song) :=
  sorry


end NUMINAMATH_CALUDE_max_weeks_correct_l1342_134271


namespace NUMINAMATH_CALUDE_line_proof_l1342_134294

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y + 10 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 2*y + 4 = 0
def result_line (x y : ℝ) : Prop := 2*x + 3*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular (3/2) (-2/3) :=
by sorry

end NUMINAMATH_CALUDE_line_proof_l1342_134294


namespace NUMINAMATH_CALUDE_f_symmetric_about_x_eq_2_l1342_134289

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if -2 ≤ x ∧ x ≤ 0 then 2^x - 2^(-x) + x else 0  -- We define f on [-2,0] as given, and 0 elsewhere

-- State the theorem
theorem f_symmetric_about_x_eq_2 :
  (∀ x, x * f x = -x * f (-x)) →  -- y = xf(x) is even
  (∀ x, f (x - 1) + f (x + 3) = 0) →  -- given condition
  (∀ x, f (x - 2) = f (-x + 2)) :=  -- symmetry about x = 2
by sorry

end NUMINAMATH_CALUDE_f_symmetric_about_x_eq_2_l1342_134289


namespace NUMINAMATH_CALUDE_prob_last_is_one_l1342_134232

/-- Represents the set of possible digits Andrea can write. -/
def Digits : Finset ℕ := {1, 2, 3, 4}

/-- Represents whether a number is prime. -/
def isPrime (n : ℕ) : Prop := sorry

/-- Represents the process of writing digits until the sum of the last two is prime. -/
def StoppingProcess : Type := sorry

/-- The probability of the last digit being 1 given the first digit. -/
def probLastIsOne (first : ℕ) : ℚ := sorry

/-- The probability of the last digit being 1 for the entire process. -/
def totalProbLastIsOne : ℚ := sorry

/-- Theorem stating the probability of the last digit being 1 is 17/44. -/
theorem prob_last_is_one :
  totalProbLastIsOne = 17 / 44 := by sorry

end NUMINAMATH_CALUDE_prob_last_is_one_l1342_134232


namespace NUMINAMATH_CALUDE_larger_number_proof_l1342_134280

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 4186) →
  (a = 23 * 13) →
  (b = 23 * 14) →
  max a b = 322 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1342_134280


namespace NUMINAMATH_CALUDE_equation_with_added_constant_l1342_134284

theorem equation_with_added_constant (y : ℝ) (n : ℝ) :
  y^4 - 20*y + 1 = 22 ∧ n = 3 →
  y^4 - 20*y + (1 + n) = 22 + n :=
by sorry

end NUMINAMATH_CALUDE_equation_with_added_constant_l1342_134284


namespace NUMINAMATH_CALUDE_income_change_approx_23_86_percent_l1342_134277

def job_a_initial_weekly : ℚ := 60
def job_a_final_weekly : ℚ := 78
def job_a_quarterly_bonus : ℚ := 50

def job_b_initial_weekly : ℚ := 100
def job_b_final_weekly : ℚ := 115
def job_b_initial_biannual_bonus : ℚ := 200
def job_b_bonus_increase_rate : ℚ := 0.1

def weekly_expenses : ℚ := 30
def weeks_per_quarter : ℕ := 13

def initial_quarterly_income : ℚ :=
  job_a_initial_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_initial_weekly * weeks_per_quarter + job_b_initial_biannual_bonus / 2

def final_quarterly_income : ℚ :=
  job_a_final_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_final_weekly * weeks_per_quarter + 
  (job_b_initial_biannual_bonus * (1 + job_b_bonus_increase_rate)) / 2

def quarterly_expenses : ℚ := weekly_expenses * weeks_per_quarter

def initial_effective_income : ℚ := initial_quarterly_income - quarterly_expenses
def final_effective_income : ℚ := final_quarterly_income - quarterly_expenses

def income_change_percentage : ℚ :=
  (final_effective_income - initial_effective_income) / initial_effective_income * 100

theorem income_change_approx_23_86_percent : 
  ∃ ε > 0, abs (income_change_percentage - 23.86) < ε :=
sorry

end NUMINAMATH_CALUDE_income_change_approx_23_86_percent_l1342_134277


namespace NUMINAMATH_CALUDE_trip_theorem_l1342_134225

/-- Represents the ticket prices and group sizes for a school trip -/
structure TripData where
  adultPrice : ℕ
  studentDiscount : ℚ
  groupDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the number of adults and students in the group -/
def calculateGroup (data : TripData) : ℕ × ℕ :=
  sorry

/-- Calculates the cost of tickets for different purchasing strategies -/
def calculateCosts (data : TripData) (adults : ℕ) (students : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of adults and students, and the most cost-effective purchasing strategy -/
theorem trip_theorem (data : TripData) 
  (h1 : data.adultPrice = 120)
  (h2 : data.studentDiscount = 1/2)
  (h3 : data.groupDiscount = 3/5)
  (h4 : data.totalPeople = 130)
  (h5 : data.totalCost = 9600) :
  let (adults, students) := calculateGroup data
  let (regularCost, allGroupCost, mixedCost) := calculateCosts data adults students
  adults = 30 ∧ 
  students = 100 ∧ 
  mixedCost < allGroupCost ∧
  mixedCost < regularCost :=
sorry

end NUMINAMATH_CALUDE_trip_theorem_l1342_134225


namespace NUMINAMATH_CALUDE_solve_equation_l1342_134207

theorem solve_equation (x : ℝ) : 5 + 7 / x = 6 - 5 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1342_134207


namespace NUMINAMATH_CALUDE_number_fraction_problem_l1342_134212

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l1342_134212


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1342_134215

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1342_134215


namespace NUMINAMATH_CALUDE_hyperbola_foci_on_x_axis_l1342_134223

/-- A curve C defined by mx^2 + (2-m)y^2 = 1 is a hyperbola with foci on the x-axis if and only if m ∈ (2, +∞) -/
theorem hyperbola_foci_on_x_axis (m : ℝ) :
  (∀ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (a^2 + c^2) + y^2 / a^2 = 1) →
  m > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_on_x_axis_l1342_134223


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1342_134281

theorem sum_of_coefficients (P a b c d e f : ℕ) : 
  20112011 = a * P^5 + b * P^4 + c * P^3 + d * P^2 + e * P + f →
  a < P ∧ b < P ∧ c < P ∧ d < P ∧ e < P ∧ f < P →
  a + b + c + d + e + f = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1342_134281


namespace NUMINAMATH_CALUDE_expression_value_l1342_134286

theorem expression_value (x : ℝ) (h : x^2 - 4*x - 1 = 0) : 
  (x - 3) / (x - 4) - 1 / x = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1342_134286


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_of_three_l1342_134250

theorem largest_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * (n + 1) ∧ c = 3 * (n + 2)) → 
  a + b + c = 72 → 
  max a (max b c) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_of_three_l1342_134250


namespace NUMINAMATH_CALUDE_worm_domino_division_l1342_134260

/-- A worm is represented by a list of directions (Up or Right) -/
inductive Direction
| Up
| Right

def Worm := List Direction

/-- Count the number of cells in a worm -/
def cellCount (w : Worm) : Nat :=
  w.length + 1

/-- Predicate to check if a worm can be divided into n dominoes -/
def canDivideIntoDominoes (w : Worm) (n : Nat) : Prop :=
  ∃ (division : List (Worm × Worm)), 
    division.length = n ∧
    (division.map (λ (p : Worm × Worm) => cellCount p.1 + cellCount p.2)).sum = cellCount w

/-- The main theorem -/
theorem worm_domino_division (w : Worm) (n : Nat) :
  n > 2 → (canDivideIntoDominoes w n ↔ cellCount w = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_worm_domino_division_l1342_134260


namespace NUMINAMATH_CALUDE_cat_food_bags_l1342_134243

theorem cat_food_bags (cat_food_weight : ℕ) (dog_food_bags : ℕ) (weight_difference : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) : 
  cat_food_weight = 3 →
  dog_food_bags = 2 →
  weight_difference = 2 →
  ounces_per_pound = 16 →
  total_ounces = 256 →
  ∃ (x : ℕ), x * cat_food_weight * ounces_per_pound + 
    dog_food_bags * (cat_food_weight + weight_difference) * ounces_per_pound = total_ounces ∧ 
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_cat_food_bags_l1342_134243


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l1342_134258

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def q : ℚ := 31 / 63

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1 / 2)

/-- The event of getting 4 consecutive heads -/
def four_heads : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 4 → n + i = 1

/-- The event of getting 3 consecutive tails -/
def three_tails : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 3 → n + i = 0

/-- The probability of an event occurring before another event
    when repeatedly performing an experiment -/
def prob_before (p : ℚ) (event1 event2 : ℕ → Prop) : Prop :=
  ∃ n : ℕ, (∀ k < n, ¬event1 k ∧ ¬event2 k) ∧ event1 n ∧ (∀ k ≤ n, ¬event2 k)

theorem four_heads_before_three_tails :
  fair_coin (λ p => prob_before q four_heads three_tails) :=
sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l1342_134258


namespace NUMINAMATH_CALUDE_exists_touching_arrangement_l1342_134266

/-- Represents a coin as a circle in a 2D plane -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two coins are touching -/
def are_touching (c1 c2 : Coin) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents an arrangement of five coins -/
structure CoinArrangement where
  coins : Fin 5 → Coin
  all_same_size : ∀ i j, (coins i).radius = (coins j).radius

/-- Theorem stating that there exists an arrangement where each coin touches exactly four others -/
theorem exists_touching_arrangement :
  ∃ (arr : CoinArrangement), ∀ i : Fin 5, (∃! j : Fin 5, ¬(are_touching (arr.coins i) (arr.coins j))) :=
sorry

end NUMINAMATH_CALUDE_exists_touching_arrangement_l1342_134266


namespace NUMINAMATH_CALUDE_sum_of_evaluations_is_32_l1342_134274

/-- The expression to be evaluated -/
def expression : List ℕ := [1, 2, 3, 4, 5, 6]

/-- A sign assignment is a list of booleans, where true represents + and false represents - -/
def SignAssignment := List Bool

/-- Evaluate the expression given a sign assignment -/
def evaluate (signs : SignAssignment) : ℤ :=
  sorry

/-- Generate all possible sign assignments -/
def allSignAssignments : List SignAssignment :=
  sorry

/-- Calculate the sum of all evaluations -/
def sumOfEvaluations : ℤ :=
  sorry

/-- The main theorem: The sum of all evaluations is 32 -/
theorem sum_of_evaluations_is_32 : sumOfEvaluations = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evaluations_is_32_l1342_134274


namespace NUMINAMATH_CALUDE_automobile_distance_l1342_134231

/-- Proves the total distance traveled by an automobile given specific conditions -/
theorem automobile_distance (a r : ℝ) : 
  let first_half_distance : ℝ := a / 4
  let first_half_time : ℝ := 2 * r
  let first_half_speed : ℝ := first_half_distance / first_half_time
  let second_half_speed : ℝ := 2 * first_half_speed
  let second_half_time : ℝ := 2 * 60 -- 2 minutes in seconds
  let second_half_distance : ℝ := second_half_speed * second_half_time
  let total_distance_feet : ℝ := first_half_distance + second_half_distance
  let total_distance_yards : ℝ := total_distance_feet / 3
  total_distance_yards = 121 * a / 12 :=
by sorry

end NUMINAMATH_CALUDE_automobile_distance_l1342_134231


namespace NUMINAMATH_CALUDE_fraction_comparison_l1342_134201

theorem fraction_comparison (a b c d : ℝ) (h1 : a/b < c/d) (h2 : b > d) (h3 : d > 0) :
  (a+c)/(b+d) < (1/2) * (a/b + c/d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1342_134201


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l1342_134276

-- Define the valid range for hours and minutes
def valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23
def valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem max_sum_of_digits_24hour_watch : 
  ∀ h m, valid_hour h → valid_minute m →
  sum_of_digits h + sum_of_digits m ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l1342_134276


namespace NUMINAMATH_CALUDE_inequality_proof_l1342_134227

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1342_134227


namespace NUMINAMATH_CALUDE_part_one_part_two_l1342_134226

-- Define the solution set M
def M (a : ℝ) := {x : ℝ | a * x^2 + 5 * x - 2 > 0}

-- Part 1
theorem part_one (a : ℝ) : 2 ∈ M a → a > -2 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  M a = {x : ℝ | 1/2 < x ∧ x < 2} →
  {x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1342_134226


namespace NUMINAMATH_CALUDE_only_A_is_impossible_l1342_134203

-- Define the set of possible ball colors in the bag
inductive BallColor
| Red
| White

-- Define the set of possible outcomes for a dice roll
inductive DiceOutcome
| One | Two | Three | Four | Five | Six

-- Define the set of possible last digits for a license plate
inductive LicensePlateLastDigit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define the events
def event_A : Prop := ∃ (ball : BallColor), ball = BallColor.Red ∨ ball = BallColor.White

def event_B : Prop := True  -- We can't model weather prediction precisely, so we assume it's always possible

def event_C : Prop := ∃ (outcome : DiceOutcome), outcome = DiceOutcome.Six

def event_D : Prop := ∃ (digit : LicensePlateLastDigit), 
  digit = LicensePlateLastDigit.Zero ∨ 
  digit = LicensePlateLastDigit.Two ∨ 
  digit = LicensePlateLastDigit.Four ∨ 
  digit = LicensePlateLastDigit.Six ∨ 
  digit = LicensePlateLastDigit.Eight

-- Theorem stating that only event A is impossible
theorem only_A_is_impossible :
  (¬ event_A) ∧ event_B ∧ event_C ∧ event_D :=
sorry

end NUMINAMATH_CALUDE_only_A_is_impossible_l1342_134203


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1342_134214

/-- The line x - ky + 1 = 0 (k ∈ ℝ) always intersects the circle x^2 + y^2 + 4x - 2y + 2 = 0 -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ (x y : ℝ), 
    (x - k*y + 1 = 0) ∧ 
    (x^2 + y^2 + 4*x - 2*y + 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_line_intersects_circle_l1342_134214


namespace NUMINAMATH_CALUDE_jack_jill_equal_payment_l1342_134229

/-- Represents the pizza order and consumption details --/
structure PizzaOrder where
  totalSlices : ℕ
  baseCost : ℚ
  pepperoniCost : ℚ
  jackPepperoniSlices : ℕ
  jackCheeseSlices : ℕ

/-- Calculates the total cost of the pizza --/
def totalCost (order : PizzaOrder) : ℚ :=
  order.baseCost + order.pepperoniCost

/-- Calculates the cost per slice --/
def costPerSlice (order : PizzaOrder) : ℚ :=
  totalCost order / order.totalSlices

/-- Calculates Jack's payment --/
def jackPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.jackPepperoniSlices + order.jackCheeseSlices)

/-- Calculates Jill's payment --/
def jillPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.totalSlices - order.jackPepperoniSlices - order.jackCheeseSlices)

/-- Theorem: Jack and Jill pay the same amount for their share of the pizza --/
theorem jack_jill_equal_payment (order : PizzaOrder)
  (h1 : order.totalSlices = 12)
  (h2 : order.baseCost = 12)
  (h3 : order.pepperoniCost = 3)
  (h4 : order.jackPepperoniSlices = 4)
  (h5 : order.jackCheeseSlices = 2) :
  jackPayment order = jillPayment order := by
  sorry

end NUMINAMATH_CALUDE_jack_jill_equal_payment_l1342_134229


namespace NUMINAMATH_CALUDE_parallelogram_count_formula_l1342_134259

/-- An equilateral triangle with side length n, tiled with n^2 smaller equilateral triangles -/
structure TiledTriangle (n : ℕ) where
  side_length : ℕ := n
  num_small_triangles : ℕ := n^2

/-- The number of parallelograms in a tiled equilateral triangle -/
def count_parallelograms (t : TiledTriangle n) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating that the number of parallelograms in a tiled equilateral triangle
    is equal to 3 * (n+2 choose 4) -/
theorem parallelogram_count_formula (n : ℕ) (t : TiledTriangle n) :
  count_parallelograms t = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_formula_l1342_134259


namespace NUMINAMATH_CALUDE_unique_pricing_l1342_134268

/-- Represents the price of a sewage treatment equipment model in thousand dollars. -/
structure ModelPrice where
  price : ℝ

/-- Represents the pricing of two sewage treatment equipment models A and B. -/
structure EquipmentPricing where
  modelA : ModelPrice
  modelB : ModelPrice

/-- Checks if the given equipment pricing satisfies the problem conditions. -/
def satisfiesConditions (pricing : EquipmentPricing) : Prop :=
  pricing.modelA.price = pricing.modelB.price + 5 ∧
  2 * pricing.modelA.price + 3 * pricing.modelB.price = 45

/-- Theorem stating that the only pricing satisfying the conditions is A at 12 and B at 7. -/
theorem unique_pricing :
  ∀ (pricing : EquipmentPricing),
    satisfiesConditions pricing →
    pricing.modelA.price = 12 ∧ pricing.modelB.price = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_pricing_l1342_134268


namespace NUMINAMATH_CALUDE_floor_sum_example_l1342_134262

theorem floor_sum_example : ⌊(-13.7 : ℝ)⌋ + ⌊(13.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1342_134262


namespace NUMINAMATH_CALUDE_probability_at_least_nine_correct_l1342_134213

theorem probability_at_least_nine_correct (n : ℕ) (p : ℝ) : 
  n = 10 → 
  p = 1/4 → 
  let P := (n.choose 9) * p^9 * (1-p)^1 + (n.choose 10) * p^10
  ∃ ε > 0, abs (P - 3e-5) < ε := by sorry

end NUMINAMATH_CALUDE_probability_at_least_nine_correct_l1342_134213


namespace NUMINAMATH_CALUDE_ratio_equality_l1342_134246

theorem ratio_equality (a b : ℝ) (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a^2 / 5 = b^3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1342_134246


namespace NUMINAMATH_CALUDE_baker_cakes_l1342_134298

theorem baker_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : pastries_made = 169)
  (h2 : cakes_sold = pastries_sold + 11)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  pastries_made + 11 = 180 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l1342_134298


namespace NUMINAMATH_CALUDE_largest_number_l1342_134230

theorem largest_number (a b c d e : ℕ) :
  a = 30^20 ∧
  b = 10^30 ∧
  c = 30^10 + 20^20 ∧
  d = (30 + 10)^20 ∧
  e = (30 * 20)^10 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1342_134230


namespace NUMINAMATH_CALUDE_largest_alternating_geometric_sequence_l1342_134249

def is_valid_sequence (a b c d : ℕ) : Prop :=
  a > b ∧ b < c ∧ c > d ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  ∃ (r : ℚ), b = a / r ∧ c = a / (r^2) ∧ d = a / (r^3)

theorem largest_alternating_geometric_sequence :
  ∀ (n : ℕ), n ≤ 9999 →
    is_valid_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
    n ≤ 9632 :=
sorry

end NUMINAMATH_CALUDE_largest_alternating_geometric_sequence_l1342_134249


namespace NUMINAMATH_CALUDE_sum_place_values_equals_350077055735_l1342_134279

def numeral : ℕ := 95378637153370261

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_place_values : ℕ :=
  -- Two 3's
  place_value 3 11 + place_value 3 1 +
  -- Three 7's
  place_value 7 10 + place_value 7 6 + place_value 7 2 +
  -- Four 5's
  place_value 5 13 + place_value 5 4 + place_value 5 3 + place_value 5 0

theorem sum_place_values_equals_350077055735 :
  sum_place_values = 350077055735 := by sorry

end NUMINAMATH_CALUDE_sum_place_values_equals_350077055735_l1342_134279


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l1342_134251

/-- Given a rectangle ABCD with dimensions as specified, prove that the volume of the
    triangular prism formed by folding is 594. -/
theorem triangular_prism_volume (A B C D P : ℝ × ℝ) : 
  let AB := 13 * Real.sqrt 3
  let BC := 12 * Real.sqrt 3
  -- A, B, C, D form a rectangle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = AB^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = BC^2 ∧
  -- P is the intersection of diagonals
  (A.1 - C.1) * (B.2 - D.2) = (A.2 - C.2) * (B.1 - D.1) ∧
  P.1 = (A.1 + C.1) / 2 ∧ P.2 = (A.2 + C.2) / 2 →
  -- Volume of the triangular prism after folding
  (1/6) * AB * BC * Real.sqrt (AB^2 + BC^2 - (AB^2 * BC^2) / (AB^2 + BC^2)) = 594 := by
  sorry


end NUMINAMATH_CALUDE_triangular_prism_volume_l1342_134251


namespace NUMINAMATH_CALUDE_formula_correctness_l1342_134297

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 2

theorem formula_correctness : 
  (f 2 = 10) ∧ 
  (f 3 = 21) ∧ 
  (f 4 = 38) ∧ 
  (f 5 = 61) ∧ 
  (f 6 = 90) := by
  sorry

end NUMINAMATH_CALUDE_formula_correctness_l1342_134297


namespace NUMINAMATH_CALUDE_intersection_theorem_l1342_134228

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x < 1}

theorem intersection_theorem : M ∩ (Set.univ \ N) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1342_134228


namespace NUMINAMATH_CALUDE_m_less_than_two_l1342_134204

open Real

/-- Proposition p -/
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + 1 > 0

/-- Proposition q -/
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 ≤ 0

/-- The main theorem -/
theorem m_less_than_two (m : ℝ) (h : ¬(p m) ∨ ¬(q m)) : m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_two_l1342_134204


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1342_134236

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/b = 2 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 2 ∧ x₀ + 2*y₀ = (3 + 2*Real.sqrt 2)/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1342_134236


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1342_134244

/-- Given a rectangle with dimensions 12 inches by 10 inches and an overlapping
    rectangle of 4 inches by 3 inches, if the shaded area is 130 square inches,
    then the perimeter of the non-shaded region is 7 1/3 inches. -/
theorem non_shaded_perimeter (shaded_area : ℝ) : 
  shaded_area = 130 → 
  (12 * 10 + 4 * 3 - shaded_area) / (12 - 4) * 2 + (12 - 4) * 2 = 22 / 3 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1342_134244


namespace NUMINAMATH_CALUDE_polynomial_evaluation_at_negative_two_l1342_134287

theorem polynomial_evaluation_at_negative_two :
  let f : ℝ → ℝ := λ x ↦ x^3 + x^2 + 2*x + 2
  f (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_at_negative_two_l1342_134287


namespace NUMINAMATH_CALUDE_at_least_one_passes_l1342_134256

theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l1342_134256


namespace NUMINAMATH_CALUDE_blue_cards_count_l1342_134254

theorem blue_cards_count (total_cards : ℕ) (blue_cards : ℕ) : 
  (10 : ℕ) + blue_cards = total_cards →
  (blue_cards : ℚ) / total_cards = 4/5 →
  blue_cards = 40 := by
  sorry

end NUMINAMATH_CALUDE_blue_cards_count_l1342_134254


namespace NUMINAMATH_CALUDE_acid_solution_mixing_l1342_134211

theorem acid_solution_mixing (y z : ℝ) (hy : y > 25) :
  (y * y / 100 + z * 40 / 100) / (y + z) * 100 = y + 10 →
  z = 10 * y / (y - 30) := by
sorry

end NUMINAMATH_CALUDE_acid_solution_mixing_l1342_134211


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1342_134282

theorem chess_tournament_participants (n : ℕ) : 
  n > 3 → 
  (n * (n - 1)) / 2 = 26 → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1342_134282


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l1342_134238

def monthly_salary : ℝ := 6250
def initial_savings_rate : ℝ := 0.20
def final_savings : ℝ := 250

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - final_savings
  let percentage_increase := (expense_increase / initial_expenses) * 100
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l1342_134238


namespace NUMINAMATH_CALUDE_girls_in_chemistry_class_l1342_134205

theorem girls_in_chemistry_class (total : ℕ) (girls boys : ℕ) : 
  total = 70 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 40 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_chemistry_class_l1342_134205


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l1342_134269

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ+, n < 101 → ¬(Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.05) ∧ 
  (Real.sqrt 101 - Real.sqrt 100 < 0.05) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l1342_134269


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1342_134290

theorem consecutive_integers_average (a : ℕ) (c : ℕ) (h1 : c = 3 * a + 3) : 
  (c + (c + 1) + (c + 2)) / 3 = 3 * a + 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1342_134290


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1342_134209

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 18 / Real.sqrt (4 * x - 3) = 9} = {3, 9.75} := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1342_134209


namespace NUMINAMATH_CALUDE_max_value_of_a_l1342_134293

theorem max_value_of_a : 
  (∀ x : ℝ, (x + a)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → 
  (∀ b : ℝ, (∀ x : ℝ, (x + b)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → b ≤ 2) ∧
  (∀ x : ℝ, (x + 2)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1342_134293


namespace NUMINAMATH_CALUDE_sock_pairs_theorem_l1342_134283

theorem sock_pairs_theorem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 42 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_theorem_l1342_134283
