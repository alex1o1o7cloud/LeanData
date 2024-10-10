import Mathlib

namespace product_and_quotient_of_geometric_sequences_l260_26093

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem product_and_quotient_of_geometric_sequences
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  is_geometric_sequence (λ n => a n * b n) ∧
  is_geometric_sequence (λ n => a n / b n) :=
sorry

end product_and_quotient_of_geometric_sequences_l260_26093


namespace arithmetic_sequence_sum_l260_26037

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_sum_l260_26037


namespace vector_parallel_implies_x_eq_two_l260_26048

-- Define vectors in R²
def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

-- Define vector addition and subtraction
def add_vectors (u v : Fin 2 → ℝ) : Fin 2 → ℝ := ![u 0 + v 0, u 1 + v 1]
def sub_vectors (u v : Fin 2 → ℝ) : Fin 2 → ℝ := ![u 0 - v 0, u 1 - v 1]

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 1 = u 1 * v 0

-- Theorem statement
theorem vector_parallel_implies_x_eq_two (x : ℝ) :
  parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = 2 := by
  sorry

end vector_parallel_implies_x_eq_two_l260_26048


namespace product_remainder_l260_26031

theorem product_remainder (a b c d : ℕ) (ha : a = 1492) (hb : b = 1776) (hc : c = 1812) (hd : d = 1996) :
  (a * b * c * d) % 5 = 4 := by
  sorry

end product_remainder_l260_26031


namespace max_value_of_p_l260_26018

theorem max_value_of_p (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1)) ∧ 
  p ≤ 10/3 ∧ 
  (∃ (a' b' c' : ℝ) (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') 
    (h' : a' * b' * c' + a' + c' = b'), 
    (2 / (a'^2 + 1)) - (2 / (b'^2 + 1)) + (3 / (c'^2 + 1)) = 10/3) :=
by sorry

end max_value_of_p_l260_26018


namespace complex_number_modulus_l260_26090

theorem complex_number_modulus : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_modulus_l260_26090


namespace max_value_of_z_l260_26000

theorem max_value_of_z (x y k : ℝ) (h1 : x + 2*y - 1 ≥ 0) (h2 : x - y ≥ 0) 
  (h3 : 0 ≤ x) (h4 : x ≤ k) (h5 : ∃ (x_min y_min : ℝ), x_min + k*y_min = -2 ∧ 
  x_min + 2*y_min - 1 ≥ 0 ∧ x_min - y_min ≥ 0 ∧ 0 ≤ x_min ∧ x_min ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≥ -2) :
  ∃ (x_max y_max : ℝ), x_max + k*y_max = 20 ∧ 
  x_max + 2*y_max - 1 ≥ 0 ∧ x_max - y_max ≥ 0 ∧ 0 ≤ x_max ∧ x_max ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≤ 20 :=
sorry

end max_value_of_z_l260_26000


namespace truck_toll_theorem_l260_26069

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  3.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels := 18
  let frontAxleWheels := 2
  let otherAxleWheels := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll axles = 5.00 := by
  sorry

end truck_toll_theorem_l260_26069


namespace cube_root_of_square_l260_26007

theorem cube_root_of_square (x : ℝ) : (x^2)^(1/3) = x^(2/3) := by
  sorry

end cube_root_of_square_l260_26007


namespace expand_product_l260_26066

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7*x + 10 := by
  sorry

end expand_product_l260_26066


namespace fifth_number_21st_row_l260_26056

/-- The number of odd numbers in the nth row of the triangular arrangement -/
def odd_numbers_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  (odd_numbers_in_row n + 1) * n / 2

/-- The nth positive odd number -/
def nth_odd_number (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row :
  let total_before := sum_odd_numbers 20
  let position := total_before + 5
  nth_odd_number position = 809 := by sorry

end fifth_number_21st_row_l260_26056


namespace unique_solution_quadratic_system_l260_26097

theorem unique_solution_quadratic_system (y : ℚ) 
  (eq1 : 10 * y^2 + 9 * y - 2 = 0)
  (eq2 : 30 * y^2 + 77 * y - 14 = 0) : 
  y = 1/5 := by sorry

end unique_solution_quadratic_system_l260_26097


namespace quadratic_general_form_l260_26033

/-- Given a quadratic equation 3x² + 1 = 7x, its general form is 3x² - 7x + 1 = 0 -/
theorem quadratic_general_form : 
  ∀ x : ℝ, 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 := by
sorry

end quadratic_general_form_l260_26033


namespace closest_product_l260_26015

def options : List ℝ := [1600, 1800, 2000, 2200, 2400]

theorem closest_product : 
  let product := 0.000625 * 3142857
  ∀ x ∈ options, x ≠ 1800 → |product - 1800| < |product - x| :=
by sorry

end closest_product_l260_26015


namespace train_crossing_time_l260_26055

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 900 →
  train_speed_kmh = 108 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end train_crossing_time_l260_26055


namespace units_digit_17_35_l260_26058

theorem units_digit_17_35 : 17^35 % 10 = 3 := by
  sorry

end units_digit_17_35_l260_26058


namespace smallest_dual_base_representation_l260_26087

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 2 ∧ d > 2 ∧
      m = 2 * c + 1 ∧
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end smallest_dual_base_representation_l260_26087


namespace sin_equation_solution_l260_26039

theorem sin_equation_solution (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  Real.sin (n * π / 180) = Real.sin (680 * π / 180) →
  n = 40 ∨ n = 140 := by
sorry

end sin_equation_solution_l260_26039


namespace positive_integer_pairs_eq_enumerated_set_l260_26011

def positive_integer_pairs : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = 4}

def enumerated_set : Set (ℕ × ℕ) :=
  {(1, 3), (2, 2), (3, 1)}

theorem positive_integer_pairs_eq_enumerated_set :
  positive_integer_pairs = enumerated_set := by
  sorry

end positive_integer_pairs_eq_enumerated_set_l260_26011


namespace sandis_initial_amount_l260_26063

/-- Proves that Sandi's initial amount was $300 given the conditions of the problem -/
theorem sandis_initial_amount (sandi_initial : ℝ) : 
  (3 * sandi_initial + 150 = 1050) → sandi_initial = 300 := by
  sorry

end sandis_initial_amount_l260_26063


namespace prob_even_first_odd_second_l260_26088

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of even outcomes on a standard die -/
def evenOutcomes : ℕ := 3

/-- The number of odd outcomes on a standard die -/
def oddOutcomes : ℕ := 3

/-- The probability of rolling an even number on one die -/
def probEven : ℚ := evenOutcomes / sides

/-- The probability of rolling an odd number on one die -/
def probOdd : ℚ := oddOutcomes / sides

theorem prob_even_first_odd_second : probEven * probOdd = 1 / 4 := by
  sorry

end prob_even_first_odd_second_l260_26088


namespace largest_number_in_set_l260_26086

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem largest_number_in_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_smallest : a = 2) : 
  c = 10 := by
sorry

end largest_number_in_set_l260_26086


namespace coin_problem_l260_26020

theorem coin_problem (total : ℕ) (difference : ℕ) (heads : ℕ) : 
  total = 128 → difference = 12 → heads = (total + difference) / 2 → heads = 70 := by
  sorry

end coin_problem_l260_26020


namespace cafe_location_l260_26072

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Checks if a point divides a line segment in a given ratio -/
def divides_segment (p1 p2 p : Point) (m n : ℚ) : Prop :=
  p.x = (n * p1.x + m * p2.x) / (m + n) ∧
  p.y = (n * p1.y + m * p2.y) / (m + n)

theorem cafe_location :
  let mark := Point.mk 1 8
  let sandy := Point.mk (-5) 0
  let cafe := Point.mk (-3) (8/3)
  divides_segment mark sandy cafe 1 2 := by
  sorry

end cafe_location_l260_26072


namespace good_student_count_l260_26042

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students making the first claim -/
def firstClaimCount : Nat := 5

/-- The number of students making the second claim -/
def secondClaimCount : Nat := 20

/-- Represents the statements made by students -/
structure Statements where
  firstClaim : Bool  -- True if the statement is true
  secondClaim : Bool -- True if the statement is true

/-- Checks if the first claim is consistent with the given number of good students -/
def checkFirstClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second claim is consistent with the given number of good students -/
def checkSecondClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Checks if the given number of good students is consistent with all statements -/
def isConsistent (goodCount : Nat) (statements : Statements) : Bool :=
  (statements.firstClaim = checkFirstClaim goodCount) &&
  (statements.secondClaim = checkSecondClaim goodCount)

/-- Theorem: The number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (statements : Statements),
    (isConsistent 5 statements ∨ isConsistent 7 statements) ∧
    ∀ (n : Nat), n ≠ 5 ∧ n ≠ 7 → ¬ isConsistent n statements :=
by sorry

end good_student_count_l260_26042


namespace symmetric_line_equation_l260_26021

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to x = 1,
    prove that the equation of l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end symmetric_line_equation_l260_26021


namespace quadratic_inequality_range_l260_26059

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end quadratic_inequality_range_l260_26059


namespace air_quality_probability_l260_26046

theorem air_quality_probability (p_one_day : ℝ) (p_two_days : ℝ) 
  (h1 : p_one_day = 0.8) 
  (h2 : p_two_days = 0.6) : 
  p_two_days / p_one_day = 0.75 := by
  sorry

end air_quality_probability_l260_26046


namespace solution_difference_l260_26008

theorem solution_difference (r s : ℝ) : 
  ((5 * r - 20) / (r^2 + 3*r - 18) = r + 3) →
  ((5 * s - 20) / (s^2 + 3*s - 18) = s + 3) →
  (r ≠ s) →
  (r > s) →
  (r - s = Real.sqrt 29) :=
by sorry

end solution_difference_l260_26008


namespace tan_value_from_sin_plus_cos_l260_26082

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end tan_value_from_sin_plus_cos_l260_26082


namespace toy_cost_price_l260_26073

def toy_problem (num_toys : ℕ) (selling_price : ℚ) (gain_ratio : ℕ) :=
  (num_toys : ℚ) * (selling_price / num_toys) / (num_toys + gain_ratio : ℚ)

theorem toy_cost_price :
  toy_problem 25 62500 5 = 2083 + 1/3 :=
by sorry

end toy_cost_price_l260_26073


namespace lucky_larry_challenge_l260_26075

theorem lucky_larry_challenge (a b c d e f : ℤ) :
  a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8 ∧ f = 5 →
  (a + b - c + d - e + f = a + (b - (c + (d - (e + f))))) ↔ e = 8 := by
sorry

end lucky_larry_challenge_l260_26075


namespace helen_cookies_l260_26095

/-- The number of cookies Helen baked in total -/
def total_cookies : ℕ := 574

/-- The number of cookies Helen baked this morning -/
def morning_cookies : ℕ := 139

/-- The number of cookies Helen baked yesterday -/
def yesterday_cookies : ℕ := total_cookies - morning_cookies

theorem helen_cookies : yesterday_cookies = 435 := by
  sorry

end helen_cookies_l260_26095


namespace coin_value_calculation_l260_26006

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The number of quarters -/
def num_quarters : ℕ := 7

/-- The number of half-dollars -/
def num_half_dollars : ℕ := 5

/-- The total value of the coins in dollars -/
def total_value : ℚ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_quarters * quarter_value +
  num_half_dollars * half_dollar_value

theorem coin_value_calculation :
  total_value = 4.84 := by sorry

end coin_value_calculation_l260_26006


namespace polynomial_B_value_l260_26024

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 9*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 81

theorem polynomial_B_value (A B C D : ℤ) :
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  B = -46 := by
  sorry

end polynomial_B_value_l260_26024


namespace right_triangle_perimeter_l260_26014

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 → 
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end right_triangle_perimeter_l260_26014


namespace sum_of_products_l260_26071

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 4)
  (h3 : z^2 + x*z + x^2 = 79) :
  x*y + y*z + x*z = 20 := by
sorry

end sum_of_products_l260_26071


namespace other_number_proof_l260_26065

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by sorry

end other_number_proof_l260_26065


namespace inverse_proportion_inequality_l260_26094

theorem inverse_proportion_inequality (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  k < 0 →
  y₁ = k / (-3) →
  y₂ = k / (-2) →
  y₃ = k / 3 →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end inverse_proportion_inequality_l260_26094


namespace f_property_f_expression_l260_26099

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2 - 4

-- State the theorem
theorem f_property : ∀ x : ℝ, f (1 + x) = x^2 + 2*x - 1 := by
  sorry

-- Prove that f(x) = x^2 - 4
theorem f_expression : ∀ x : ℝ, f x = x^2 - 4 := by
  sorry

end f_property_f_expression_l260_26099


namespace negation_of_universal_proposition_l260_26057

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ (∃ x : ℝ, x^2 ≤ x - 1) := by
  sorry

end negation_of_universal_proposition_l260_26057


namespace domain_of_f_l260_26013

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (2 * sin x - 1) + sqrt (1 - 2 * cos x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = ⋃ (k : ℤ), Ico (2 * k * π + π / 3) (2 * k * π + 5 * π / 6) :=
by sorry

end domain_of_f_l260_26013


namespace range_of_expression_l260_26051

theorem range_of_expression (x y : ℝ) (h : 4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) :
  10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3 :=
by sorry

end range_of_expression_l260_26051


namespace min_xy_value_l260_26084

theorem min_xy_value (x y : ℝ) :
  (∃ (n : ℕ), n = 12) →
  1 + Real.cos (2 * x + 3 * y - 1) ^ 2 = (x^2 + y^2 + 2*(x+1)*(1-y)) / (x-y+1) →
  ∀ (z : ℝ), x * y ≥ 1/25 ∧ (∃ (a b : ℝ), a * b = 1/25) :=
by sorry

end min_xy_value_l260_26084


namespace ellipse_parabola_properties_l260_26096

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the line y = k(x - 4)
def line_k (x y k : ℝ) : Prop := y = k * (x - 4)

-- Define the line x = 1
def line_x1 (x : ℝ) : Prop := x = 1

theorem ellipse_parabola_properties 
  (a b c : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : c / a = Real.sqrt 3 / 2) 
  (h_focus : ∃ x y, ellipse_C x y a b ∧ parabola_E x y ∧ (x = a ∨ x = -a ∨ y = b ∨ y = -b)) :
  -- 1. Equation of C
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 = 1) ∧
  -- 2. Collinearity of A, P, and N
  (∀ k x_M y_M x_N y_N x_P, 
    k ≠ 0 →
    ellipse_C x_M y_M a b →
    ellipse_C x_N y_N a b →
    line_k x_M y_M k →
    line_k x_N y_N k →
    line_x1 x_P →
    ∃ y_P, line_k x_P y_P k →
    ∃ t, t * (x_P + 2) = 3 ∧ t * y_P = k * (x_N + 2)) ∧
  -- 3. Maximum area of triangle OMN
  (∃ S : ℝ, 
    (∀ k x_M y_M x_N y_N, 
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) ≤ S) ∧
    (∃ k x_M y_M x_N y_N,
      k ≠ 0 →
      ellipse_C x_M y_M a b →
      ellipse_C x_N y_N a b →
      line_k x_M y_M k →
      line_k x_N y_N k →
      (1/2) * abs (x_M * y_N - x_N * y_M) = S) ∧
    S = 1) := by
  sorry

end ellipse_parabola_properties_l260_26096


namespace second_sum_calculation_l260_26085

/-- Proves that given the conditions, the second sum is 1704 --/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) 
  (h1 : total = 2769)
  (h2 : total = first_part + second_part)
  (h3 : (first_part * 3 * 8 / 100) = (second_part * 5 * 3 / 100)) :
  second_part = 1704 := by
  sorry

end second_sum_calculation_l260_26085


namespace system_solution_l260_26053

theorem system_solution (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : z^2 = x^2 + y^2) 
  (h3 : (z + c)^2 = (x + a)^2 + (y + b)^2) : 
  y = (b/a) * x ∧ z = (c/a) * x :=
sorry

end system_solution_l260_26053


namespace star_sqrt_11_l260_26003

/-- Custom binary operation ¤ -/
def star (x y z : ℝ) : ℝ := (x + y)^2 - z^2

theorem star_sqrt_11 (z : ℝ) :
  star (Real.sqrt 11) (Real.sqrt 11) z = 44 → z = 0 := by
  sorry

end star_sqrt_11_l260_26003


namespace last_s_replacement_l260_26026

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the function to calculate the shift for the nth occurrence
def shift (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to apply the shift modulo alphabet size
def apply_shift (shift : ℕ) : ℕ := shift % alphabet_size

-- Theorem statement
theorem last_s_replacement (occurrences : ℕ) (h : occurrences = 12) :
  apply_shift (shift occurrences) = 0 := by sorry

end last_s_replacement_l260_26026


namespace polygon_interior_angles_l260_26036

theorem polygon_interior_angles (n : ℕ) : (n - 2) * 180 = 360 → n = 4 := by
  sorry

end polygon_interior_angles_l260_26036


namespace wallpaper_overlap_l260_26040

theorem wallpaper_overlap (total_area : ℝ) (large_wall_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_area = 500)
  (h2 : large_wall_area = 280)
  (h3 : two_layer_area = 54)
  (h4 : three_layer_area = 28)
  (h5 : four_layer_area = 14) :
  ∃ (six_layer_area : ℝ), 
    six_layer_area = 9 ∧ 
    total_area = (large_wall_area - two_layer_area - three_layer_area) + 
                 2 * two_layer_area + 
                 3 * three_layer_area + 
                 4 * four_layer_area + 
                 6 * six_layer_area :=
by sorry

end wallpaper_overlap_l260_26040


namespace monday_attendance_l260_26030

theorem monday_attendance (tuesday : ℕ) (wed_to_fri : ℕ) (average : ℕ) (days : ℕ)
  (h1 : tuesday = 15)
  (h2 : wed_to_fri = 10)
  (h3 : average = 11)
  (h4 : days = 5) :
  ∃ (monday : ℕ), monday + tuesday + 3 * wed_to_fri = average * days ∧ monday = 10 := by
  sorry

end monday_attendance_l260_26030


namespace square_of_ten_n_plus_five_l260_26061

theorem square_of_ten_n_plus_five (n : ℕ) : (10 * n + 5)^2 = 100 * n * (n + 1) + 25 := by
  sorry

#eval (10 * 199 + 5)^2  -- Should output 3980025

end square_of_ten_n_plus_five_l260_26061


namespace stating_n_gon_triangulation_l260_26028

/-- 
A polygon with n sides (n-gon) can be divided into triangles by non-intersecting diagonals. 
This function represents the number of such triangles.
-/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- 
Theorem stating that the number of triangles into which non-intersecting diagonals 
divide an n-gon is equal to n-2, for any n ≥ 3.
-/
theorem n_gon_triangulation (n : ℕ) (h : n ≥ 3) : 
  num_triangles n = n - 2 := by
  sorry

end stating_n_gon_triangulation_l260_26028


namespace subtract_base6_l260_26049

/-- Convert a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Theorem: Subtracting 35₆ from 131₆ in base 6 is equal to 52₆ --/
theorem subtract_base6 :
  let a := [1, 3, 1]  -- 131₆
  let b := [3, 5]     -- 35₆
  let result := [5, 2] -- 52₆
  base10ToBase6 (base6ToBase10 a - base6ToBase10 b) = result := by
  sorry

end subtract_base6_l260_26049


namespace polynomial_remainder_l260_26079

theorem polynomial_remainder (s : ℤ) : (s^11 + 1) % (s + 1) = 0 := by
  sorry

end polynomial_remainder_l260_26079


namespace farmers_market_sales_l260_26025

theorem farmers_market_sales (total_earnings broccoli_sales cauliflower_sales : ℕ) 
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : cauliflower_sales = 136) :
  ∃ (spinach_sales : ℕ), 
    spinach_sales = 73 ∧ 
    spinach_sales > (2 * broccoli_sales) / 2 ∧
    total_earnings = broccoli_sales + (2 * broccoli_sales) + cauliflower_sales + spinach_sales :=
by
  sorry


end farmers_market_sales_l260_26025


namespace quadratic_root_in_arithmetic_sequence_l260_26050

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ c ≥ b ≥ 0,
    if the quadratic ax^2 + cx + b has exactly one root, then this root is -2 + √3. -/
theorem quadratic_root_in_arithmetic_sequence (a b c : ℝ) 
    (seq : ∃ (d : ℝ), c = a - d ∧ b = a - 2*d) 
    (order : a ≥ c ∧ c ≥ b ∧ b ≥ 0) 
    (one_root : ∃! x : ℝ, a*x^2 + c*x + b = 0) :
  ∃ (x : ℝ), a*x^2 + c*x + b = 0 ∧ x = -2 + Real.sqrt 3 := by
  sorry

end quadratic_root_in_arithmetic_sequence_l260_26050


namespace rope_cut_theorem_l260_26010

/-- Given a rope of 60 meters cut into two pieces, where the longer piece is twice
    the length of the shorter piece, prove that the length of the shorter piece is 20 meters. -/
theorem rope_cut_theorem (total_length : ℝ) (short_piece : ℝ) (long_piece : ℝ) : 
  total_length = 60 →
  long_piece = 2 * short_piece →
  total_length = short_piece + long_piece →
  short_piece = 20 := by
  sorry

end rope_cut_theorem_l260_26010


namespace train_passing_platform_l260_26092

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / time_to_cross_point) = 190 :=
sorry

end train_passing_platform_l260_26092


namespace number_times_one_sixth_squared_l260_26034

theorem number_times_one_sixth_squared (x : ℝ) : x * (1/6)^2 = 6^3 ↔ x = 7776 := by
  sorry

end number_times_one_sixth_squared_l260_26034


namespace evaluate_expression_l260_26004

theorem evaluate_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end evaluate_expression_l260_26004


namespace no_two_digit_primes_with_digit_sum_nine_l260_26060

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ¬∃ n : ℕ, is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 9 := by
  sorry

end no_two_digit_primes_with_digit_sum_nine_l260_26060


namespace mutually_exclusive_head_l260_26078

-- Define the set of people
variable (People : Type)

-- Define the property of standing at the head of the line
variable (stands_at_head : People → Prop)

-- Define A and B as specific people
variable (A B : People)

-- Axiom: A and B are distinct people
axiom A_neq_B : A ≠ B

-- Axiom: Only one person can stand at the head of the line
axiom one_at_head : ∀ (x y : People), stands_at_head x ∧ stands_at_head y → x = y

-- Theorem: The events "A stands at the head of the line" and "B stands at the head of the line" are mutually exclusive
theorem mutually_exclusive_head : 
  ¬(stands_at_head A ∧ stands_at_head B) :=
sorry

end mutually_exclusive_head_l260_26078


namespace quadrilateral_inequalities_l260_26044

-- Define a structure for a convex quadrilateral
structure ConvexQuadrilateral :=
  (a b c d t : ℝ)
  (area_positive : t > 0)
  (sides_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : ConvexQuadrilateral) : Prop := sorry

-- State the theorem
theorem quadrilateral_inequalities (q : ConvexQuadrilateral) :
  (2 * q.t ≤ q.a * q.b + q.c * q.d) ∧
  (2 * q.t ≤ q.a * q.c + q.b * q.d) ∧
  ((2 * q.t = q.a * q.b + q.c * q.d) ∨ (2 * q.t = q.a * q.c + q.b * q.d) → is_cyclic q) :=
by sorry

end quadrilateral_inequalities_l260_26044


namespace calculation_result_l260_26001

theorem calculation_result : (101 * 2012 * 121) / 1111 / 503 = 44 := by
  sorry

end calculation_result_l260_26001


namespace not_divisible_by_five_l260_26019

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^3 + 2*n - 1)) := by
  sorry

end not_divisible_by_five_l260_26019


namespace one_fourth_of_7_2_l260_26080

theorem one_fourth_of_7_2 : (7.2 : ℚ) / 4 = 9 / 5 := by
  sorry

end one_fourth_of_7_2_l260_26080


namespace birthday_crayons_l260_26074

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul had lost or given away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end birthday_crayons_l260_26074


namespace sheet_area_difference_l260_26041

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem sheet_area_difference : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 9
  let sheet2_length : ℝ := 4.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 99 := by
  sorry


end sheet_area_difference_l260_26041


namespace x_twelfth_power_l260_26045

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := by
  sorry

end x_twelfth_power_l260_26045


namespace arithmetic_sequence_difference_l260_26029

/-- The arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The absolute difference between two integers -/
def absDiff (a b : ℤ) : ℕ := (a - b).natAbs

theorem arithmetic_sequence_difference :
  let a := -10  -- First term of the sequence
  let d := 11   -- Common difference of the sequence
  absDiff (arithmeticSequence a d 2025) (arithmeticSequence a d 2010) = 165 := by
sorry

end arithmetic_sequence_difference_l260_26029


namespace side_length_eq_twice_radius_l260_26077

/-- A square with a circle inscribed such that the circle is tangent to two adjacent sides
    and passes through one vertex of the square. -/
structure InscribedCircleSquare where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- The circle is tangent to two adjacent sides of the square -/
  tangent_to_sides : True
  /-- The circle passes through one vertex of the square -/
  passes_through_vertex : True

/-- The side length of a square with an inscribed circle tangent to two adjacent sides
    and passing through one vertex is equal to twice the radius of the circle. -/
theorem side_length_eq_twice_radius (square : InscribedCircleSquare) :
  square.s = 2 * square.r := by
  sorry

end side_length_eq_twice_radius_l260_26077


namespace amanda_hourly_rate_l260_26064

/-- Amanda's cleaning service hourly rate calculation -/
theorem amanda_hourly_rate :
  let monday_hours : ℝ := 7.5
  let tuesday_hours : ℝ := 3
  let thursday_hours : ℝ := 4
  let saturday_hours : ℝ := 6
  let total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours
  let total_earnings : ℝ := 410
  total_earnings / total_hours = 20 := by
sorry

end amanda_hourly_rate_l260_26064


namespace two_even_dice_probability_l260_26089

/-- The probability of rolling an even number on a fair 8-sided die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 2 dice out of 3 -/
def ways_to_choose : ℕ := 3

/-- The probability of exactly two dice showing even numbers when rolling three fair 8-sided dice -/
def prob_two_even : ℚ := ways_to_choose * (prob_even^2 * (1 - prob_even))

theorem two_even_dice_probability : prob_two_even = 3/8 := by
  sorry

end two_even_dice_probability_l260_26089


namespace bottles_per_case_l260_26002

theorem bottles_per_case (april_cases : ℕ) (may_cases : ℕ) (total_bottles : ℕ) : 
  april_cases = 20 → may_cases = 30 → total_bottles = 1000 →
  ∃ (bottles_per_case : ℕ), bottles_per_case * (april_cases + may_cases) = total_bottles ∧ bottles_per_case = 20 :=
by sorry

end bottles_per_case_l260_26002


namespace optimal_renovation_solution_l260_26032

/-- Represents a renovation team -/
structure Team where
  dailyRate : ℕ
  daysAlone : ℕ

/-- The renovation scenario -/
structure RenovationScenario where
  teamA : Team
  teamB : Team
  jointDays : ℕ
  jointCost : ℕ
  mixedDaysA : ℕ
  mixedDaysB : ℕ
  mixedCost : ℕ

/-- Theorem stating the optimal solution for the renovation scenario -/
theorem optimal_renovation_solution (scenario : RenovationScenario) 
  (h1 : scenario.jointDays * (scenario.teamA.dailyRate + scenario.teamB.dailyRate) = scenario.jointCost)
  (h2 : scenario.mixedDaysA * scenario.teamA.dailyRate + scenario.mixedDaysB * scenario.teamB.dailyRate = scenario.mixedCost)
  (h3 : scenario.teamA.daysAlone = 12)
  (h4 : scenario.teamB.daysAlone = 24)
  (h5 : scenario.jointDays = 8)
  (h6 : scenario.jointCost = 3520)
  (h7 : scenario.mixedDaysA = 6)
  (h8 : scenario.mixedDaysB = 12)
  (h9 : scenario.mixedCost = 3480) :
  scenario.teamA.dailyRate = 300 ∧ 
  scenario.teamB.dailyRate = 140 ∧ 
  scenario.teamB.daysAlone * scenario.teamB.dailyRate < scenario.teamA.daysAlone * scenario.teamA.dailyRate :=
by sorry

end optimal_renovation_solution_l260_26032


namespace range_of_a_l260_26043

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → a > 1 := by
  sorry

end range_of_a_l260_26043


namespace quadratic_inequality_range_l260_26023

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 3 > 0) ↔ a > 1/3 :=
by sorry

end quadratic_inequality_range_l260_26023


namespace greatest_divisor_four_consecutive_integers_l260_26047

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 
    12 ∣ (k * (k + 1) * (k + 2) * (k + 3)) ∧
    ∀ (m : ℕ), m > 12 → 
      ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (j * (j + 1) * (j + 2) * (j + 3)))) :=
by sorry

end greatest_divisor_four_consecutive_integers_l260_26047


namespace train_length_l260_26054

/-- The length of a train given relative speeds and passing time -/
theorem train_length (v1 v2 t : ℝ) (h1 : v1 = 36) (h2 : v2 = 45) (h3 : t = 4) :
  (v1 + v2) * (5 / 18) * t = 90 := by
  sorry

#check train_length

end train_length_l260_26054


namespace six_couples_handshakes_l260_26067

/-- The number of handshakes in a gathering of couples where each person shakes hands
    with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, the total number of handshakes is 60 -/
theorem six_couples_handshakes :
  handshakes 6 = 60 := by
  sorry


end six_couples_handshakes_l260_26067


namespace age_problem_l260_26083

theorem age_problem (oleg serezha misha : ℕ) : 
  serezha = oleg + 1 →
  misha = serezha + 1 →
  40 < oleg + serezha + misha →
  oleg + serezha + misha < 45 →
  oleg = 13 ∧ serezha = 14 ∧ misha = 15 := by
sorry

end age_problem_l260_26083


namespace smallest_x_for_digit_sum_50_l260_26070

def sequence_sum (x : ℕ) : ℕ := 100 * x + 4950

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_x_for_digit_sum_50 :
  ∀ x : ℕ, x < 99950 → digit_sum (sequence_sum x) ≠ 50 ∧
  digit_sum (sequence_sum 99950) = 50 :=
sorry

end smallest_x_for_digit_sum_50_l260_26070


namespace arithmetic_calculation_l260_26062

theorem arithmetic_calculation : 8 / 4 - 3^2 + 4 * 5 = 13 := by
  sorry

end arithmetic_calculation_l260_26062


namespace factorization_existence_l260_26098

theorem factorization_existence : ∃ (a b c : ℤ), 
  (∀ x, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ∧ (a = 8 ∨ a = 12) := by
  sorry

end factorization_existence_l260_26098


namespace tv_cost_l260_26017

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 840 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 210 :=
by
  sorry

end tv_cost_l260_26017


namespace rectangular_field_diagonal_shortcut_l260_26009

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y →
  x / y = 5/12 := by
sorry

end rectangular_field_diagonal_shortcut_l260_26009


namespace line_moved_down_l260_26038

/-- The equation of a line obtained by moving y = 2x down 3 units -/
def moved_line (x y : ℝ) : Prop := y = 2 * x - 3

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := y = 2 * x

/-- Moving a line down by a certain number of units subtracts that number from the y-coordinate -/
axiom move_down (a b : ℝ) : ∀ x y, original_line x y → moved_line x (y - b) → b = 3

theorem line_moved_down : 
  ∀ x y, original_line x y → moved_line x (y - 3) :=
sorry

end line_moved_down_l260_26038


namespace doctor_team_count_l260_26091

/-- The number of ways to choose a team of doctors under specific conditions -/
def choose_doctor_team (total_doctors : ℕ) (pediatricians surgeons general_practitioners : ℕ) 
  (team_size : ℕ) : ℕ :=
  (pediatricians.choose 1) * (surgeons.choose 1) * (general_practitioners.choose 1) * 
  ((total_doctors - 3).choose (team_size - 3))

/-- Theorem stating the number of ways to choose a team of 5 doctors from 25 doctors, 
    with specific specialty requirements -/
theorem doctor_team_count : 
  choose_doctor_team 25 5 10 10 5 = 115500 := by
  sorry

end doctor_team_count_l260_26091


namespace charcoal_drawings_l260_26052

theorem charcoal_drawings (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ)
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7)
  (h4 : total = colored_pencil + blending_marker + (total - colored_pencil - blending_marker)) :
  total - colored_pencil - blending_marker = 4 := by
sorry

end charcoal_drawings_l260_26052


namespace five_student_committees_with_two_fixed_l260_26012

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of different five-student committees that can be chosen from a group of 8 students,
    where two specific students must always be included -/
theorem five_student_committees_with_two_fixed (total_students : ℕ) (committee_size : ℕ) (fixed_students : ℕ) :
  total_students = 8 →
  committee_size = 5 →
  fixed_students = 2 →
  choose (total_students - fixed_students) (committee_size - fixed_students) = 20 := by
  sorry


end five_student_committees_with_two_fixed_l260_26012


namespace vendor_profit_l260_26081

/-- Vendor's profit calculation --/
theorem vendor_profit : 
  let apple_buy_price : ℚ := 3 / 2
  let apple_sell_price : ℚ := 2
  let orange_buy_price : ℚ := 2.7 / 3
  let orange_sell_price : ℚ := 1
  let apple_discount_rate : ℚ := 1 / 10
  let orange_discount_rate : ℚ := 3 / 20
  let num_apples : ℕ := 5
  let num_oranges : ℕ := 5

  let discounted_apple_price := apple_sell_price * (1 - apple_discount_rate)
  let discounted_orange_price := orange_sell_price * (1 - orange_discount_rate)

  let total_cost := num_apples * apple_buy_price + num_oranges * orange_buy_price
  let total_revenue := num_apples * discounted_apple_price + num_oranges * discounted_orange_price

  total_revenue - total_cost = 1.25 := by sorry

end vendor_profit_l260_26081


namespace absolute_sum_zero_implies_sum_l260_26005

theorem absolute_sum_zero_implies_sum (a b : ℝ) : 
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end absolute_sum_zero_implies_sum_l260_26005


namespace fewer_bees_than_flowers_l260_26016

theorem fewer_bees_than_flowers : 
  let flowers : ℕ := 5
  let bees : ℕ := 3
  flowers - bees = 2 := by sorry

end fewer_bees_than_flowers_l260_26016


namespace same_color_probability_is_59_225_l260_26076

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (blue : Nat)
  (yellow : Nat)
  (green : Nat)
  (purple : Nat)
  (total : Nat)
  (side_sum : blue + yellow + green + purple = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : Rat :=
  let blue_prob := (d.blue * d.blue : Rat) / (d.total * d.total)
  let yellow_prob := (d.yellow * d.yellow : Rat) / (d.total * d.total)
  let green_prob := (d.green * d.green : Rat) / (d.total * d.total)
  let purple_prob := (d.purple * d.purple : Rat) / (d.total * d.total)
  blue_prob + yellow_prob + green_prob + purple_prob

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { blue := 6
    yellow := 8
    green := 10
    purple := 6
    total := 30
    side_sum := by norm_num }

/-- Theorem stating the probability of two problem dice showing the same color -/
theorem same_color_probability_is_59_225 :
  same_color_probability problem_die = 59 / 225 := by
  sorry


end same_color_probability_is_59_225_l260_26076


namespace equation_implies_fraction_value_l260_26027

theorem equation_implies_fraction_value
  (a x y : ℝ)
  (h : x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (abs (Real.log (x - a) - Real.log (a - y)))) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
sorry

end equation_implies_fraction_value_l260_26027


namespace stephanie_oranges_l260_26035

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := store_visits * oranges_per_visit

theorem stephanie_oranges : total_oranges = 16 := by
  sorry

end stephanie_oranges_l260_26035


namespace blue_string_length_l260_26068

/-- Given the lengths of three strings (red, white, and blue) with specific relationships,
    prove that the blue string is 5 metres long. -/
theorem blue_string_length (red white blue : ℝ) : 
  red = 8 →
  white = 5 * red →
  white = 8 * blue →
  blue = 5 := by
  sorry

end blue_string_length_l260_26068


namespace parking_space_savings_l260_26022

/-- Proves the yearly savings when renting a parking space monthly instead of weekly -/
theorem parking_space_savings (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  monthly_rate = 42 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  weekly_rate * weeks_per_year - monthly_rate * months_per_year = 16 := by
sorry

end parking_space_savings_l260_26022
