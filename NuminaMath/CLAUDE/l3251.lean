import Mathlib

namespace NUMINAMATH_CALUDE_bagel_store_spending_l3251_325167

/-- The total amount spent by Ben and David in the bagel store -/
def total_spent (b d : ℝ) : ℝ := b + d

/-- Ben's spending is $15 more than David's spending -/
def ben_spent_more (b d : ℝ) : Prop := b = d + 15

/-- David's spending is half of Ben's spending -/
def david_spent_half (b d : ℝ) : Prop := d = b / 2

theorem bagel_store_spending (b d : ℝ) 
  (h1 : david_spent_half b d) 
  (h2 : ben_spent_more b d) : 
  total_spent b d = 45 := by
  sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l3251_325167


namespace NUMINAMATH_CALUDE_class_average_weight_l3251_325109

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 30 →
  students_B = 20 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  let total_students := students_A + students_B
  let total_weight := students_A * avg_weight_A + students_B * avg_weight_B
  (total_weight / total_students : ℝ) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l3251_325109


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3251_325155

/-- A rectangular prism with dimensions 3, 4, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: The total number of diagonals in a rectangular prism with dimensions 3, 4, and 5 is 16. -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3251_325155


namespace NUMINAMATH_CALUDE_ratio_of_partial_fractions_l3251_325117

theorem ratio_of_partial_fractions (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / (P : ℚ) = 15 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_partial_fractions_l3251_325117


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3251_325130

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^2 / (2*i - 1)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3251_325130


namespace NUMINAMATH_CALUDE_tank_capacity_l3251_325136

theorem tank_capacity (C : ℝ) 
  (h1 : (3/4 : ℝ) * C + 8 = (7/8 : ℝ) * C) : C = 64 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3251_325136


namespace NUMINAMATH_CALUDE_picnic_class_size_l3251_325168

theorem picnic_class_size : ∃ (x : ℕ), 
  x > 0 ∧ 
  (x / 2 + x / 3 + x / 4 : ℚ) = 65 ∧ 
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_picnic_class_size_l3251_325168


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3251_325127

theorem polynomial_evaluation :
  let f (x : ℤ) := x^3 + x^2 + x + 1
  f (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3251_325127


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_a_values_l3251_325162

theorem ellipse_eccentricity_a_values (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 6 = 1) →
  (let e := Real.sqrt 6 / 6
   ∃ b : ℝ, e^2 = 1 - (min a (Real.sqrt 6))^2 / (max a (Real.sqrt 6))^2) →
  a = 6 * Real.sqrt 5 / 5 ∨ a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_a_values_l3251_325162


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3251_325183

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4)
  (h_ineq : ∀ x : ℝ, -x^2 + 6*x - 8 > 0 ↔ 2 < x ∧ x < 4) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3251_325183


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3251_325181

theorem fractional_equation_solution :
  ∃ x : ℝ, (x / (x + 2) + 4 / (x^2 - 4) = 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3251_325181


namespace NUMINAMATH_CALUDE_tens_digit_of_expression_l3251_325113

theorem tens_digit_of_expression : ∃ n : ℕ, (2023^2024 - 2025 + 6) % 100 = 10 + 100 * n := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_expression_l3251_325113


namespace NUMINAMATH_CALUDE_triangular_grid_edges_l3251_325166

theorem triangular_grid_edges (n : ℕ) (h : n = 1001) : 
  let total_squares := n * (n + 1) / 2
  let total_edges_without_sharing := 4 * total_squares
  let shared_edges := (n - 1) * n / 2 - 1
  total_edges_without_sharing - 2 * shared_edges = 1006004 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_edges_l3251_325166


namespace NUMINAMATH_CALUDE_smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l3251_325191

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(factorial n % 2016 = 0) ∧ (factorial 8 % 2016 = 0) := by sorry

theorem smallest_n_factorial_divisible_by_2016_pow_10 :
  ∀ n : ℕ, n < 63 → ¬(factorial n % (2016^10) = 0) ∧ (factorial 63 % (2016^10) = 0) := by sorry

end NUMINAMATH_CALUDE_smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l3251_325191


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_780_l3251_325149

theorem largest_gcd_of_sum_780 :
  ∃ (x y : ℕ+), x + y = 780 ∧ 
  ∀ (a b : ℕ+), a + b = 780 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 390 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_780_l3251_325149


namespace NUMINAMATH_CALUDE_min_red_chips_l3251_325182

theorem min_red_chips (r w b : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ r :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l3251_325182


namespace NUMINAMATH_CALUDE_intersection_M_N_l3251_325171

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3251_325171


namespace NUMINAMATH_CALUDE_abc_def_ratio_l3251_325110

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l3251_325110


namespace NUMINAMATH_CALUDE_y_is_odd_square_l3251_325142

def x : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_y_is_odd_square_l3251_325142


namespace NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3251_325164

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3251_325164


namespace NUMINAMATH_CALUDE_percy_christmas_money_l3251_325185

/-- The amount of money Percy received at Christmas -/
def christmas_money : ℝ :=
  let playstation_cost : ℝ := 500
  let birthday_money : ℝ := 200
  let game_price : ℝ := 7.5
  let games_sold : ℕ := 20
  playstation_cost - birthday_money - (game_price * games_sold)

theorem percy_christmas_money :
  christmas_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_percy_christmas_money_l3251_325185


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_length_and_angle_l3251_325116

/-- Given a circle with a sector having an arc length of 25 cm and a central angle of 45 degrees,
    the radius of the circle is equal to 100/π cm. -/
theorem circle_radius_from_arc_length_and_angle (L : ℝ) (θ : ℝ) (r : ℝ) :
  L = 25 →
  θ = 45 →
  L = (θ / 360) * (2 * π * r) →
  r = 100 / π :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_length_and_angle_l3251_325116


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3251_325140

/-- Given a = -1 and ab = 2, prove that 3(2a²b + ab²) - (3ab² - a²b) evaluates to -14 -/
theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3251_325140


namespace NUMINAMATH_CALUDE_function_equation_solution_l3251_325186

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0) 
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3251_325186


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l3251_325100

theorem exam_pass_percentage 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (failed_both : Real) 
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l3251_325100


namespace NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l3251_325172

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l3251_325172


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l3251_325184

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l3251_325184


namespace NUMINAMATH_CALUDE_linear_congruence_solution_l3251_325104

theorem linear_congruence_solution (x : ℤ) : 
  (9 * x + 2) % 15 = 7 → x % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_congruence_solution_l3251_325104


namespace NUMINAMATH_CALUDE_elevator_scenarios_count_l3251_325101

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 7

/-- Represents the number of people entering the elevator -/
def num_people : ℕ := 3

/-- Calculates the number of scenarios where exactly one person goes to the top floor
    and person A does not get off on the second floor -/
def elevator_scenarios : ℕ :=
  let a_to_top := (num_floors - 2)^(num_people - 1)
  let others_to_top := (num_people - 1) * (num_floors - 3) * (num_floors - 2)
  a_to_top + others_to_top

/-- The main theorem stating that the number of scenarios is 65 -/
theorem elevator_scenarios_count :
  elevator_scenarios = 65 := by sorry

end NUMINAMATH_CALUDE_elevator_scenarios_count_l3251_325101


namespace NUMINAMATH_CALUDE_circumcenter_coordinates_l3251_325197

/-- A quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The circumcenter of a quadrilateral -/
def circumcenter (q : Quadrilateral) : ℝ × ℝ := sorry

/-- A quadrilateral is inscribed in a circle if its circumcenter exists -/
def isInscribed (q : Quadrilateral) : Prop :=
  ∃ c : ℝ × ℝ, c = circumcenter q

theorem circumcenter_coordinates (q : Quadrilateral) (h : isInscribed q) :
  circumcenter q = (6, 1) := by sorry

end NUMINAMATH_CALUDE_circumcenter_coordinates_l3251_325197


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l3251_325159

/-- The set of vertices of a family of parabolas forms a parabola -/
theorem vertices_form_parabola (a c d : ℝ) (ha : a > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (b : ℝ),
    let (x, y) := f b
    (∀ t, y = a * t^2 + b * t + c * t + d → (x - t) * (2 * a * t + b + c) = 0) ∧
    y = -a * x^2 + d :=
  sorry

end NUMINAMATH_CALUDE_vertices_form_parabola_l3251_325159


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3251_325189

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) 
  (h2 : hyperbola a (-2) 1) :
  ∃ (k : ℝ), k = 1 ∨ k = -1 ∧ 
  ∀ (x y : ℝ), (x + k*y = 0) ↔ (∀ ε > 0, ∃ t > 0, ∀ t' ≥ t, 
    ∃ x' y', hyperbola a x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < ε^2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3251_325189


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3251_325118

/-- The base 6 number represented as a list of digits -/
def base_6_number : List Nat := [1, 0, 2, 1, 1, 1, 0, 1, 1]

/-- Convert a list of digits in base 6 to a natural number -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number we're working with -/
def n : Nat := to_base_10 base_6_number

/-- A number is prime if it has exactly two distinct divisors -/
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ m : Nat, m > 0 → m < p → (p % m = 0 → m = 1)

/-- p divides n -/
def divides (p n : Nat) : Prop := n % p = 0

theorem largest_prime_divisor :
  ∃ (p : Nat), is_prime p ∧ divides p n ∧
  ∀ (q : Nat), is_prime q → divides q n → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3251_325118


namespace NUMINAMATH_CALUDE_empirical_regression_equation_l3251_325123

/-- Data for 10 years of resident income and goods sales -/
def income : List Float := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
def sales : List Float := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

/-- Given statistics -/
def sum_x : Float := 379.6
def sum_y : Float := 391.0
def sum_x_squared : Float := 246.904
def sum_y_squared : Float := 568.9
def correlation_coefficient : Float := 0.95

/-- Mean values -/
def mean_x : Float := sum_x / 10
def mean_y : Float := sum_y / 10

/-- Regression coefficients -/
def b_hat : Float := correlation_coefficient * (sum_y_squared.sqrt / sum_x_squared.sqrt)
def a_hat : Float := mean_y - b_hat * mean_x

theorem empirical_regression_equation :
  (b_hat * 100).round / 100 = 1.44 ∧ 
  (a_hat * 100).round / 100 = -15.56 := by
  sorry

#check empirical_regression_equation

end NUMINAMATH_CALUDE_empirical_regression_equation_l3251_325123


namespace NUMINAMATH_CALUDE_third_term_in_hundredth_group_l3251_325148

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of terms before the nth group -/
def termsBeforeGroup (n : ℕ) : ℕ := triangularNumber (n - 1)

/-- The last term in the nth group -/
def lastTermInGroup (n : ℕ) : ℕ := sequenceTerm (termsBeforeGroup (n + 1))

/-- The kth term in the nth group -/
def termInGroup (n k : ℕ) : ℕ := lastTermInGroup n - 2 * (n - k)

theorem third_term_in_hundredth_group :
  termInGroup 100 3 = 9905 := by sorry

end NUMINAMATH_CALUDE_third_term_in_hundredth_group_l3251_325148


namespace NUMINAMATH_CALUDE_edge_sum_is_144_l3251_325198

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 432 cm³
  volume_eq : a * b * c = 432
  -- Surface area is 432 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 432
  -- Dimensions are in geometric progression
  geometric_progression : b * b = a * c

/-- The sum of the lengths of all edges of the rectangular solid is 144 cm -/
theorem edge_sum_is_144 (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 144 := by
  sorry

end NUMINAMATH_CALUDE_edge_sum_is_144_l3251_325198


namespace NUMINAMATH_CALUDE_megan_eggs_count_l3251_325134

theorem megan_eggs_count :
  ∀ (broken cracked perfect : ℕ),
  broken = 3 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken + cracked + perfect = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_eggs_count_l3251_325134


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l3251_325124

/-- Proves that the cost of gas per gallon is $4, given the conditions of Dan's car fuel efficiency and travel distance. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : 
  miles_per_gallon = 32 → total_miles = 464 → total_cost = 58 → 
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
  sorry

/-- The actual cost of gas per gallon based on the given conditions. -/
def actual_gas_cost : ℝ := 4

#check gas_cost_per_gallon
#check actual_gas_cost

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l3251_325124


namespace NUMINAMATH_CALUDE_panda_arrangement_count_l3251_325128

/-- Represents the number of pandas -/
def num_pandas : ℕ := 9

/-- Represents the number of shortest pandas that must be at the ends -/
def num_shortest : ℕ := 3

/-- Calculates the number of ways to arrange the pandas -/
def panda_arrangements : ℕ :=
  2 * (num_pandas - num_shortest).factorial

/-- Theorem stating that the number of panda arrangements is 1440 -/
theorem panda_arrangement_count :
  panda_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_panda_arrangement_count_l3251_325128


namespace NUMINAMATH_CALUDE_no_valid_prime_pairs_l3251_325152

theorem no_valid_prime_pairs : 
  ∀ a b : ℕ, 
    Prime a → 
    Prime b → 
    b > a → 
    (a - 8) * (b - 8) = 64 → 
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_valid_prime_pairs_l3251_325152


namespace NUMINAMATH_CALUDE_circle_C_properties_l3251_325114

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def is_tangent (circle : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 2

-- State the theorem
theorem circle_C_properties :
  is_tangent circle_C line1 ∧
  is_tangent circle_C line2 ∧
  ∃ (x y : ℝ), circle_C x y ∧ line3 x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l3251_325114


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3251_325139

theorem simple_interest_problem (P R T : ℝ) : 
  P = 300 →
  P * (R + 6) / 100 * T = P * R / 100 * T + 90 →
  T = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3251_325139


namespace NUMINAMATH_CALUDE_smallest_with_2023_divisors_l3251_325192

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible (a b : ℕ) : Prop := sorry

theorem smallest_with_2023_divisors :
  ∃ (n m k : ℕ),
    n > 0 ∧
    num_divisors n = 2023 ∧
    n = m * 6^k ∧
    ¬ is_divisible m 6 ∧
    (∀ (n' m' k' : ℕ),
      n' > 0 →
      num_divisors n' = 2023 →
      n' = m' * 6^k' →
      ¬ is_divisible m' 6 →
      n ≤ n') ∧
    m + k = 745 :=
  sorry

end NUMINAMATH_CALUDE_smallest_with_2023_divisors_l3251_325192


namespace NUMINAMATH_CALUDE_total_pictures_sum_l3251_325131

/-- Represents the number of pictures Zoe has taken -/
structure PictureCount where
  initial : ℕ
  dolphin_show : ℕ
  total : ℕ

/-- Theorem stating that the total number of pictures is the sum of initial and dolphin show pictures -/
theorem total_pictures_sum (zoe : PictureCount)
  (h1 : zoe.initial = 28)
  (h2 : zoe.dolphin_show = 16)
  (h3 : zoe.total = 44) :
  zoe.total = zoe.initial + zoe.dolphin_show := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_sum_l3251_325131


namespace NUMINAMATH_CALUDE_five_integers_exist_l3251_325194

theorem five_integers_exist : ∃ (a b c d e : ℤ),
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  a * b * c = 8 ∧
  c * d * e = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_five_integers_exist_l3251_325194


namespace NUMINAMATH_CALUDE_combinatorial_identity_l3251_325102

theorem combinatorial_identity : Nat.choose 98 97 + 2 * Nat.choose 98 96 + Nat.choose 98 95 = Nat.choose 100 97 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l3251_325102


namespace NUMINAMATH_CALUDE_specific_figure_area_l3251_325178

/-- A fifteen-sided figure drawn on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  /-- The number of full unit squares within the figure -/
  full_squares : ℕ
  /-- The number of rectangles within the figure -/
  rectangles : ℕ
  /-- The width of each rectangle in cm -/
  rectangle_width : ℝ
  /-- The height of each rectangle in cm -/
  rectangle_height : ℝ
  /-- The figure has fifteen sides -/
  sides : ℕ
  sides_eq : sides = 15

/-- The area of the fifteen-sided figure in cm² -/
def figure_area (f : FifteenSidedFigure) : ℝ :=
  f.full_squares + f.rectangles * f.rectangle_width * f.rectangle_height

/-- Theorem stating that the area of the specific fifteen-sided figure is 15 cm² -/
theorem specific_figure_area :
  ∃ f : FifteenSidedFigure, figure_area f = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_area_l3251_325178


namespace NUMINAMATH_CALUDE_absolute_value_square_l3251_325105

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l3251_325105


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3251_325143

theorem alpha_beta_sum (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 75*x + 1236) / (x^2 + 60*x - 3120)) →
  α + β = 139 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3251_325143


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l3251_325119

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l3251_325119


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l3251_325153

noncomputable def f (x : ℝ) := x + 3^(x + 2)

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo (-2 : ℝ) (-1) ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l3251_325153


namespace NUMINAMATH_CALUDE_triangle_yz_length_l3251_325177

/-- Given a triangle XYZ where cos(2X-Z) + sin(X+Z) = 2 and XY = 6, prove that YZ = 3 -/
theorem triangle_yz_length (X Y Z : ℝ) (h1 : 0 < X ∧ 0 < Y ∧ 0 < Z)
  (h2 : X + Y + Z = π) (h3 : Real.cos (2*X - Z) + Real.sin (X + Z) = 2) (h4 : 6 = 6) : 
  3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_yz_length_l3251_325177


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3251_325126

theorem exponent_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3251_325126


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3251_325137

theorem polynomial_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3251_325137


namespace NUMINAMATH_CALUDE_composition_value_l3251_325179

/-- Given two functions f and g, and a composition condition, prove that d equals 18 -/
theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 18 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l3251_325179


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3251_325160

theorem arithmetic_sequence_problem (n : ℕ) : 
  let a₁ : ℤ := 1
  let d : ℤ := 3
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 298 → n = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3251_325160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3251_325115

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (a 1 + a n) * n / 2

-- Theorem statement
theorem arithmetic_sequence_ratio 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 9) 
  (h_a5 : a 5 = 5) : 
  S a 9 / S a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3251_325115


namespace NUMINAMATH_CALUDE_zhang_hong_weight_estimate_l3251_325161

/-- Regression equation for weight based on height -/
def weight_estimate (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Age range for which the regression equation is valid -/
def valid_age_range : Set ℝ := Set.Icc 18 38

theorem zhang_hong_weight_estimate :
  20 ∈ valid_age_range →
  weight_estimate 178 = 69.96 := by
  sorry

end NUMINAMATH_CALUDE_zhang_hong_weight_estimate_l3251_325161


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3251_325173

theorem or_necessary_not_sufficient_for_and (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3251_325173


namespace NUMINAMATH_CALUDE_square_product_sum_inequality_l3251_325112

theorem square_product_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_sum_inequality_l3251_325112


namespace NUMINAMATH_CALUDE_parabola_parameter_l3251_325174

/-- For a parabola with equation y^2 = 4ax and directrix x = -2, the value of a is 2. -/
theorem parabola_parameter (y x a : ℝ) : 
  (∀ y x, y^2 = 4*a*x) →  -- Equation of the parabola
  (∀ x, x = -2 → x = x) →  -- Equation of the directrix (x = -2 represented as a predicate)
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_parameter_l3251_325174


namespace NUMINAMATH_CALUDE_bamboo_pole_problem_l3251_325122

theorem bamboo_pole_problem (pole_length : ℝ) (point_distance : ℝ) 
  (h_pole_length : pole_length = 24)
  (h_point_distance : point_distance = 7) :
  ∃ (height : ℝ), height = 16 + 4 * Real.sqrt 2 ∨ height = 16 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_pole_problem_l3251_325122


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3251_325195

/-- A square with an inscribed circle -/
structure InscribedSquare :=
  (side : ℝ)
  (radius : ℝ)
  (h_radius : radius = side / 2)

/-- A point on the inscribed circle -/
structure CirclePoint (s : InscribedSquare) :=
  (x : ℝ)
  (y : ℝ)
  (h_on_circle : x^2 + y^2 = s.radius^2)

/-- Theorem: If a point on the inscribed circle is 1 unit from one side
    and 2 units from another side, then the area of the square is 100 -/
theorem inscribed_square_area
  (s : InscribedSquare)
  (p : CirclePoint s)
  (h_dist1 : p.x = 1 ∨ p.y = 1)
  (h_dist2 : p.x = 2 ∨ p.y = 2) :
  s.side^2 = 100 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3251_325195


namespace NUMINAMATH_CALUDE_consecutive_product_111222_l3251_325196

theorem consecutive_product_111222 (b : ℕ) :
  b * (b + 1) = 111222 → b = 333 := by sorry

end NUMINAMATH_CALUDE_consecutive_product_111222_l3251_325196


namespace NUMINAMATH_CALUDE_f_minus_one_eq_neg_two_l3251_325190

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_one_eq_neg_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = 2^x) :
  f (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_minus_one_eq_neg_two_l3251_325190


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l3251_325120

theorem triangle_similarity_problem (DC CB AD : ℝ) (h1 : DC = 13) (h2 : CB = 9) 
  (h3 : AD > 0) (h4 : (1/3) * AD + DC + CB = AD) : 
  ∃ FC : ℝ, FC = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l3251_325120


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3251_325170

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle to a fixed point -/
theorem midpoint_trajectory (a b x y : ℝ) : 
  a^2 + b^2 = 1 →  -- P(a,b) is on the unit circle
  x = (a + 3) / 2 ∧ y = b / 2 →  -- M(x,y) is the midpoint of PQ
  (2*x - 3)^2 + 4*y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3251_325170


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3251_325180

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 3/5) 
  (h2 : Real.cos (α - β) = 4/5) : 
  Real.tan α * Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3251_325180


namespace NUMINAMATH_CALUDE_square_sum_equality_l3251_325193

theorem square_sum_equality : 108 * 108 + 92 * 92 = 20128 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3251_325193


namespace NUMINAMATH_CALUDE_problem_solution_l3251_325176

theorem problem_solution : ((-4)^2) * (((-1)^2023) + (3/4) + ((-1/2)^3)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3251_325176


namespace NUMINAMATH_CALUDE_pizza_cost_l3251_325150

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_l3251_325150


namespace NUMINAMATH_CALUDE_work_completion_time_l3251_325175

theorem work_completion_time (q p : ℝ) (h1 : q = 20) 
  (h2 : 4 * (1/p + 1/q) = 1 - 0.5333333333333333) : p = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3251_325175


namespace NUMINAMATH_CALUDE_stall_owner_earnings_l3251_325144

/-- Represents the stall owner's game with ping-pong balls -/
structure BallGame where
  yellow_balls : ℕ := 3
  white_balls : ℕ := 2
  balls_drawn : ℕ := 3
  same_color_reward : ℕ := 5
  diff_color_cost : ℕ := 1
  daily_players : ℕ := 100
  days_in_month : ℕ := 30

/-- Calculates the expected monthly earnings of the stall owner -/
def expected_monthly_earnings (game : BallGame) : ℚ :=
  let total_balls := game.yellow_balls + game.white_balls
  let prob_same_color := (game.yellow_balls.choose game.balls_drawn) / (total_balls.choose game.balls_drawn)
  let daily_earnings := game.daily_players * (game.diff_color_cost * (1 - prob_same_color) - game.same_color_reward * prob_same_color)
  daily_earnings * game.days_in_month

/-- Theorem stating the expected monthly earnings of the stall owner -/
theorem stall_owner_earnings (game : BallGame) : 
  expected_monthly_earnings game = 1200 := by
  sorry

end NUMINAMATH_CALUDE_stall_owner_earnings_l3251_325144


namespace NUMINAMATH_CALUDE_binomial_1300_2_l3251_325141

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l3251_325141


namespace NUMINAMATH_CALUDE_rachelle_pennies_l3251_325147

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ

/-- The conditions of the penny-throwing problem -/
def PennyConditions (p : PennyThrowers) : Prop :=
  p.gretchen = p.rachelle / 2 ∧
  p.rocky = p.gretchen / 3 ∧
  p.rachelle + p.gretchen + p.rocky = 300

/-- Theorem stating that under the given conditions, Rachelle threw 180 pennies -/
theorem rachelle_pennies (p : PennyThrowers) (h : PennyConditions p) : p.rachelle = 180 := by
  sorry

end NUMINAMATH_CALUDE_rachelle_pennies_l3251_325147


namespace NUMINAMATH_CALUDE_phone_bill_percentage_abigail_phone_bill_l3251_325103

theorem phone_bill_percentage (initial_amount : ℝ) (food_percentage : ℝ) 
  (entertainment_cost : ℝ) (final_amount : ℝ) : ℝ :=
  let food_cost := initial_amount * food_percentage
  let after_food := initial_amount - food_cost
  let before_phone_bill := after_food - entertainment_cost
  let phone_bill_cost := before_phone_bill - final_amount
  let phone_bill_percentage := (phone_bill_cost / after_food) * 100
  phone_bill_percentage

theorem abigail_phone_bill : 
  phone_bill_percentage 200 0.6 20 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_percentage_abigail_phone_bill_l3251_325103


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3251_325188

theorem tenth_term_of_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, a n = n * (n + 1) / 2) : 
  a 10 = 55 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3251_325188


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l3251_325165

def kitchen_upgrade_cost (num_knobs : ℕ) (knob_price : ℚ) (num_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (num_knobs * knob_price) + (num_pulls * pull_price)

theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l3251_325165


namespace NUMINAMATH_CALUDE_tank_emptying_time_l3251_325138

/-- The time it takes for a tank to empty with a leak and inlet pipe. -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : inlet_rate_per_minute = 6) : 
  (tank_capacity / (tank_capacity / leak_empty_time - inlet_rate_per_minute * 60)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l3251_325138


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l3251_325157

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

/-- Point P with coordinates (3, -5) -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem stating that the distance from P to the x-axis is 5 -/
theorem distance_P_to_x_axis :
  distance_to_x_axis P = 5 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l3251_325157


namespace NUMINAMATH_CALUDE_yue_bao_scientific_notation_l3251_325129

theorem yue_bao_scientific_notation : 5853 = 5.853 * (10 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_yue_bao_scientific_notation_l3251_325129


namespace NUMINAMATH_CALUDE_mark_buys_extra_large_bags_l3251_325135

/-- Represents the types of balloon bags available --/
inductive BagType
  | Small
  | Medium
  | ExtraLarge

/-- Represents a bag of balloons with its price and quantity --/
structure BalloonBag where
  bagType : BagType
  price : ℕ
  quantity : ℕ

def mark_budget : ℕ := 24
def small_bag : BalloonBag := ⟨BagType.Small, 4, 50⟩
def extra_large_bag : BalloonBag := ⟨BagType.ExtraLarge, 12, 200⟩
def total_balloons : ℕ := 400

/-- Calculates the number of bags that can be bought with a given budget --/
def bags_bought (bag : BalloonBag) (budget : ℕ) : ℕ :=
  budget / bag.price

/-- Calculates the total number of balloons from a given number of bags --/
def total_balloons_from_bags (bag : BalloonBag) (num_bags : ℕ) : ℕ :=
  num_bags * bag.quantity

theorem mark_buys_extra_large_bags :
  bags_bought extra_large_bag mark_budget = 2 ∧
  total_balloons_from_bags extra_large_bag (bags_bought extra_large_bag mark_budget) = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_mark_buys_extra_large_bags_l3251_325135


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l3251_325145

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 54 ∣ m^2) : 
  ∃ (d : ℕ), d ∣ m ∧ d = 9 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l3251_325145


namespace NUMINAMATH_CALUDE_increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l3251_325125

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x - a)

-- Theorem 1
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem function_below_one_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1) ↔ 3/2 < a ∧ a < 2 := by sorry

-- Theorem 3 (partial, due to multiple conditions)
theorem function_range_when_a_geq_two (a : ℝ) (h : a ≥ 2) :
  ∃ l u : ℝ, ∀ x : ℝ, x ∈ Set.Icc 2 4 → l ≤ f a x ∧ f a x ≤ u := by sorry

end NUMINAMATH_CALUDE_increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l3251_325125


namespace NUMINAMATH_CALUDE_emily_cards_l3251_325154

theorem emily_cards (x : ℕ) : x + 7 = 70 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_emily_cards_l3251_325154


namespace NUMINAMATH_CALUDE_max_value_constraint_l3251_325132

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 - 3*x*y + 4*y^2 = 15) : 
  3*x^2 + 2*x*y + y^2 ≤ 50*Real.sqrt 3 + 65 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3251_325132


namespace NUMINAMATH_CALUDE_beatrice_tv_shopping_l3251_325187

theorem beatrice_tv_shopping (x : ℕ) 
  (h1 : x > 0)  -- Beatrice looked at some TVs in the first store
  (h2 : 42 = x + 3*x + 10) : -- Total TVs = First store + Online store + Auction site
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_shopping_l3251_325187


namespace NUMINAMATH_CALUDE_cubic_function_unique_form_l3251_325158

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b

theorem cubic_function_unique_form 
  (a b : ℝ) 
  (h_a : a > 0)
  (h_max : ∃ x₁, ∀ x, f x a b ≤ f x₁ a b ∧ f x₁ a b = 5)
  (h_min : ∃ x₂, ∀ x, f x a b ≥ f x₂ a b ∧ f x₂ a b = 1) :
  ∀ x, f x a b = x^3 + 3*x^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_unique_form_l3251_325158


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l3251_325156

theorem eight_digit_divisibility (A B : ℕ) : 
  A < 10 → B < 10 → (757 * 10^5 + A * 10^4 + B * 10^3 + 384) % 357 = 0 → A = 5 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l3251_325156


namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_mod_5_l3251_325199

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square_mod_5 (n : ℕ) : Prop :=
  ∃ m : ℕ, n ≡ m^2 [ZMOD 5]

theorem sum_of_factorials_perfect_square_mod_5 (n : ℕ+) :
  is_perfect_square_mod_5 (sum_of_factorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_mod_5_l3251_325199


namespace NUMINAMATH_CALUDE_plane_perpendicular_criterion_l3251_325107

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_criterion 
  (m : Line) (α β : Plane) :
  contains β m → perp m α → perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_criterion_l3251_325107


namespace NUMINAMATH_CALUDE_problem_statement_l3251_325106

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 1) : 
  (abs a + abs b ≤ Real.sqrt 2) ∧ (abs (a^3 / b) + abs (b^3 / a) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3251_325106


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3251_325108

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3251_325108


namespace NUMINAMATH_CALUDE_fishing_catches_proof_l3251_325151

theorem fishing_catches_proof (a b c d : ℕ) : 
  a + b = 7 ∧ 
  a + c = 9 ∧ 
  a + d = 14 ∧ 
  b + c = 14 ∧ 
  b + d = 19 ∧ 
  c + d = 21 →
  (a = 1 ∧ b = 6 ∧ c = 8 ∧ d = 13) ∨
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 13) ∨
  (a = 6 ∧ b = 1 ∧ c = 8 ∧ d = 13) ∨
  (a = 6 ∧ b = 8 ∧ c = 1 ∧ d = 13) ∨
  (a = 8 ∧ b = 1 ∧ c = 6 ∧ d = 13) ∨
  (a = 8 ∧ b = 6 ∧ c = 1 ∧ d = 13) :=
by sorry

end NUMINAMATH_CALUDE_fishing_catches_proof_l3251_325151


namespace NUMINAMATH_CALUDE_return_speed_calculation_l3251_325121

/-- Proves that given a round trip between two cities 150 miles apart,
    where the outbound speed is 50 mph and the average round trip speed is 60 mph,
    the return speed is 75 mph. -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 50 →
  average_speed = 60 →
  (2 * distance) / (distance / outbound_speed + distance / (2 * distance / average_speed - distance / outbound_speed)) = average_speed →
  2 * distance / average_speed - distance / outbound_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l3251_325121


namespace NUMINAMATH_CALUDE_square_measurement_error_l3251_325146

theorem square_measurement_error (area_error : Real) (side_error : Real) : 
  area_error = 8.16 → side_error = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l3251_325146


namespace NUMINAMATH_CALUDE_problem_statement_l3251_325133

theorem problem_statement (a b c d : ℕ+) (r : ℚ) 
  (h1 : r = 1 - (a : ℚ) / b - (c : ℚ) / d)
  (h2 : a + c ≤ 1993)
  (h3 : r > 0) :
  r > 1 / (1993 : ℚ)^3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3251_325133


namespace NUMINAMATH_CALUDE_linear_function_constraint_l3251_325163

/-- Given a linear function y = x - k, if for all x < 3, y < 2k, then k ≥ 1 -/
theorem linear_function_constraint (k : ℝ) : 
  (∀ x : ℝ, x < 3 → x - k < 2 * k) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_constraint_l3251_325163


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3251_325169

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define perpendicularity of PA to directrix
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  A.1 = -2 ∧ P.2 = A.2

-- Define the slope of AF
def slope_AF (A : ℝ × ℝ) : Prop :=
  (A.2 - 0) / (A.1 - 2) = -Real.sqrt 3

-- Theorem statement
theorem parabola_focus_distance 
  (P : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : point_on_parabola P) 
  (h2 : perpendicular_to_directrix P A) 
  (h3 : slope_AF A) : 
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3251_325169


namespace NUMINAMATH_CALUDE_donut_distribution_l3251_325111

/-- Given the conditions of the donut distribution problem, prove the number of extra donuts and the distribution in each box. -/
theorem donut_distribution
  (total_donuts : ℕ)
  (num_flavors : ℕ)
  (num_boxes : ℕ)
  (h_total : total_donuts = 125)
  (h_flavors : num_flavors = 5)
  (h_boxes : num_boxes = 12)
  (h_even_split : total_donuts % num_boxes = 0)
  (h_all_flavors : ∀ box, box ≤ num_boxes → ∀ flavor, flavor ≤ num_flavors → 
    ∃ (n : ℕ), n > 0 ∧ n = (total_donuts / num_flavors / num_boxes)) :
  (total_donuts % (num_boxes * (total_donuts / num_flavors / num_boxes))) = 5 ∧
  ∀ box, box ≤ num_boxes → ∀ flavor, flavor ≤ num_flavors →
    (total_donuts / num_flavors / num_boxes) = 2 :=
by sorry

end NUMINAMATH_CALUDE_donut_distribution_l3251_325111
