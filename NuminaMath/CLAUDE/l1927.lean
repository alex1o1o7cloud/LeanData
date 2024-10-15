import Mathlib

namespace NUMINAMATH_CALUDE_third_number_in_nth_row_l1927_192774

/-- Represents the function that gives the third number from the left in the nth row
    of a triangular array of positive odd numbers. -/
def thirdNumber (n : ℕ) : ℕ := n^2 - n + 5

/-- Theorem stating that for n ≥ 3, the third number from the left in the nth row
    of a triangular array of positive odd numbers is n^2 - n + 5. -/
theorem third_number_in_nth_row (n : ℕ) (h : n ≥ 3) :
  thirdNumber n = n^2 - n + 5 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_nth_row_l1927_192774


namespace NUMINAMATH_CALUDE_line_slope_from_y_intercept_l1927_192756

/-- Given a line with equation x + ay + 1 = 0 where a is a real number,
    and y-intercept -2, prove that the slope of the line is -2. -/
theorem line_slope_from_y_intercept (a : ℝ) :
  (∀ x y, x + a * y + 1 = 0 → (x = 0 → y = -2)) →
  ∃ m b, ∀ x y, y = m * x + b ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_from_y_intercept_l1927_192756


namespace NUMINAMATH_CALUDE_system_solution_l1927_192776

theorem system_solution (x y z : ℝ) : 
  (x + y + x * y = 19 ∧ 
   y + z + y * z = 11 ∧ 
   z + x + z * x = 14) ↔ 
  ((x = 4 ∧ y = 3 ∧ z = 2) ∨ 
   (x = -6 ∧ y = -5 ∧ z = -4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1927_192776


namespace NUMINAMATH_CALUDE_complex_fourth_power_l1927_192735

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l1927_192735


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1927_192799

theorem inequality_equivalence (x : ℝ) : 
  (x / 2 ≤ 5 - x ∧ 5 - x < -3 * (2 + x)) ↔ x < -11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1927_192799


namespace NUMINAMATH_CALUDE_airplane_passengers_l1927_192719

theorem airplane_passengers (total : ℕ) (children : ℕ) (h1 : total = 80) (h2 : children = 20) :
  let adults := total - children
  let men := adults / 2
  men = 30 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l1927_192719


namespace NUMINAMATH_CALUDE_water_leaked_l1927_192730

/-- Calculates the amount of water leaked from a bucket given the initial and remaining amounts. -/
theorem water_leaked (initial : ℚ) (remaining : ℚ) (h1 : initial = 0.75) (h2 : remaining = 0.5) :
  initial - remaining = 0.25 := by
  sorry

#check water_leaked

end NUMINAMATH_CALUDE_water_leaked_l1927_192730


namespace NUMINAMATH_CALUDE_units_digit_of_147_25_50_l1927_192772

-- Define a function to calculate the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a power
def unitsDigitOfPower (base : ℕ) (exponent : ℕ) : ℕ :=
  unitsDigit ((unitsDigit base)^exponent)

-- Theorem to prove
theorem units_digit_of_147_25_50 :
  unitsDigitOfPower (unitsDigitOfPower 147 25) 50 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_147_25_50_l1927_192772


namespace NUMINAMATH_CALUDE_students_taking_history_or_statistics_l1927_192753

theorem students_taking_history_or_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_not_statistics : ℕ) : 
  total = 90 → history = 36 → statistics = 32 → history_not_statistics = 25 →
  ∃ (both : ℕ), history - both = history_not_statistics ∧ history + statistics - both = 57 := by
sorry

end NUMINAMATH_CALUDE_students_taking_history_or_statistics_l1927_192753


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1927_192723

-- Define an isosceles triangle with two known side lengths
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b ∨ a = 3 ∨ b = 3

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + 3

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) (h1 : t.a = 3 ∨ t.a = 4) (h2 : t.b = 3 ∨ t.b = 4) :
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1927_192723


namespace NUMINAMATH_CALUDE_certain_number_exists_and_is_one_l1927_192705

theorem certain_number_exists_and_is_one : 
  ∃ (x : ℕ), x > 0 ∧ (57 * x) % 8 = 7 ∧ ∀ (y : ℕ), y > 0 ∧ (57 * y) % 8 = 7 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_is_one_l1927_192705


namespace NUMINAMATH_CALUDE_tangent_and_locus_l1927_192757

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (-1, -4)

-- Define point N
def point_N : ℝ × ℝ := (2, 0)

-- Define the tangent line equations
def tangent_line (x y : ℝ) : Prop := x = -1 ∨ 15*x - 8*y - 17 = 0

-- Define the locus of midpoint T
def locus_T (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 ∧ 0 ≤ x ∧ x < 1/2

theorem tangent_and_locus :
  (∀ x y, circle_O x y → 
    (∃ x' y', tangent_line x' y' ∧ 
      (x' = point_M.1 ∧ y' = point_M.2))) ∧
  (∀ x y, locus_T x y ↔ 
    (∃ p q : ℝ × ℝ, 
      circle_O p.1 p.2 ∧ 
      circle_O q.1 q.2 ∧ 
      (q.2 - point_N.2) * (p.1 - point_N.1) = (q.1 - point_N.1) * (p.2 - point_N.2) ∧
      x = (p.1 + q.1) / 2 ∧ 
      y = (p.2 + q.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_locus_l1927_192757


namespace NUMINAMATH_CALUDE_sqrt_19992000_floor_l1927_192778

theorem sqrt_19992000_floor : ⌊Real.sqrt 19992000⌋ = 4471 := by sorry

end NUMINAMATH_CALUDE_sqrt_19992000_floor_l1927_192778


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_lines_l1927_192717

/-- Given two lines l₁ and l₂ in the form of linear equations,
    prove that the x-intercept of l₁ is 2 when l₁ is perpendicular to l₂ -/
theorem x_intercept_of_perpendicular_lines
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ (a + 3) * x + y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + (a - 1) * y + 4 = 0)
  (h_perp : (a + 3) * 1 + (a - 1) * 1 = 0) :
  ∃ x, l₁ x 0 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_lines_l1927_192717


namespace NUMINAMATH_CALUDE_chord_length_at_specific_angle_shortest_chord_equation_l1927_192718

-- Define the circle
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 8}

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define a chord AB passing through P0
def chord (α : ℝ) : Set (ℝ × ℝ) := {p | p.2 - P0.2 = Real.tan α * (p.1 - P0.1)}

-- Define the length of a chord
def chordLength (α : ℝ) : ℝ := sorry

-- Theorem 1
theorem chord_length_at_specific_angle :
  chordLength (3 * Real.pi / 4) = Real.sqrt 30 := by sorry

-- Define the shortest chord
def shortestChord : Set (ℝ × ℝ) := sorry

-- Theorem 2
theorem shortest_chord_equation :
  shortestChord = {p | p.1 - 2 * p.2 + 5 = 0} := by sorry

end NUMINAMATH_CALUDE_chord_length_at_specific_angle_shortest_chord_equation_l1927_192718


namespace NUMINAMATH_CALUDE_tan_value_for_special_condition_l1927_192752

theorem tan_value_for_special_condition (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_value_for_special_condition_l1927_192752


namespace NUMINAMATH_CALUDE_range_of_a_l1927_192786

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |x - a| > x - 1) ↔ (a < 1 ∨ a > 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1927_192786


namespace NUMINAMATH_CALUDE_combined_mpg_calculation_l1927_192724

/-- Calculates the combined miles per gallon for three cars given their individual efficiencies and a common distance traveled. -/
def combinedMPG (ray_mpg tom_mpg amy_mpg distance : ℚ) : ℚ :=
  let total_distance := 3 * distance
  let total_gas := distance / ray_mpg + distance / tom_mpg + distance / amy_mpg
  total_distance / total_gas

/-- Theorem stating that the combined MPG for the given conditions is 3600/114 -/
theorem combined_mpg_calculation :
  combinedMPG 50 20 40 120 = 3600 / 114 := by
  sorry

#eval combinedMPG 50 20 40 120

end NUMINAMATH_CALUDE_combined_mpg_calculation_l1927_192724


namespace NUMINAMATH_CALUDE_sum_of_factors_l1927_192751

theorem sum_of_factors (d e f : ℤ) : 
  (∀ x : ℝ, x^2 + 21*x + 110 = (x + d)*(x + e)) → 
  (∀ x : ℝ, x^2 - 19*x + 88 = (x - e)*(x - f)) → 
  d + e + f = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1927_192751


namespace NUMINAMATH_CALUDE_emily_caught_four_trout_l1927_192763

def fishing_problem (num_trout : ℕ) : Prop :=
  let num_catfish : ℕ := 3
  let num_bluegill : ℕ := 5
  let weight_trout : ℝ := 2
  let weight_catfish : ℝ := 1.5
  let weight_bluegill : ℝ := 2.5
  let total_weight : ℝ := 25
  (num_trout : ℝ) * weight_trout + 
  (num_catfish : ℝ) * weight_catfish + 
  (num_bluegill : ℝ) * weight_bluegill = total_weight

theorem emily_caught_four_trout : 
  ∃ (n : ℕ), fishing_problem n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_caught_four_trout_l1927_192763


namespace NUMINAMATH_CALUDE_boys_camp_total_l1927_192726

theorem boys_camp_total (total : ℝ) 
  (school_a_percentage : total * 0.2 = total * (20 / 100))
  (science_percentage : (total * 0.2) * 0.3 = (total * 0.2) * (30 / 100))
  (non_science_count : (total * 0.2) * 0.7 = 49) : 
  total = 350 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l1927_192726


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1927_192700

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (8/5) * x^2 - (18/5) * x - 1/5

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-1) = 5 ∧ q 2 = -1 ∧ q 4 = 11 := by
  sorry

#eval q (-1)
#eval q 2
#eval q 4

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1927_192700


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1927_192793

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 0 ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1927_192793


namespace NUMINAMATH_CALUDE_equation_solution_l1927_192773

theorem equation_solution : ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1927_192773


namespace NUMINAMATH_CALUDE_jaxon_toys_count_l1927_192710

/-- The number of toys Jaxon has -/
def jaxon_toys : ℕ := 15

/-- The number of toys Gabriel has -/
def gabriel_toys : ℕ := 2 * jaxon_toys

/-- The number of toys Jerry has -/
def jerry_toys : ℕ := gabriel_toys + 8

theorem jaxon_toys_count :
  jaxon_toys + gabriel_toys + jerry_toys = 83 ∧ jaxon_toys = 15 :=
by sorry

end NUMINAMATH_CALUDE_jaxon_toys_count_l1927_192710


namespace NUMINAMATH_CALUDE_waiter_customers_l1927_192788

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that for the given scenario, the final number of customers is 41. -/
theorem waiter_customers : final_customers 19 14 36 = 41 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l1927_192788


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1927_192707

theorem algebraic_expression_value (p q : ℤ) :
  p * 3^3 + 3 * q + 1 = 2015 →
  p * (-3)^3 - 3 * q + 1 = -2013 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1927_192707


namespace NUMINAMATH_CALUDE_sin_105_degrees_l1927_192798

theorem sin_105_degrees : Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_degrees_l1927_192798


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l1927_192780

theorem polynomial_division_proof (x : ℝ) :
  6 * x^3 + 12 * x^2 - 5 * x + 3 = 
  (3 * x + 4) * (2 * x^2 + (4/3) * x - 31/9) + 235/9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l1927_192780


namespace NUMINAMATH_CALUDE_polynomial_division_identity_l1927_192731

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^6 - 5*x^4 + 3*x^3 - 7*x^2 + 2*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def q (x : ℝ) : ℝ := x^5 + 3*x^4 + 4*x^3 + 15*x^2 + 38*x + 116

/-- The remainder -/
def r : ℝ := 340

/-- Theorem stating the polynomial division identity -/
theorem polynomial_division_identity : 
  ∀ x : ℝ, f x = g x * q x + r := by sorry

end NUMINAMATH_CALUDE_polynomial_division_identity_l1927_192731


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_l1927_192770

theorem imaginary_part_of_pure_imaginary (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (a - Complex.I)
  (z.re = 0) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_l1927_192770


namespace NUMINAMATH_CALUDE_equation_is_linear_one_variable_l1927_192779

/-- Represents a polynomial equation --/
structure PolynomialEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- Checks if a polynomial equation is linear with one variable --/
def is_linear_one_variable (eq : PolynomialEquation) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, eq.lhs x = a * x + b ∧ eq.rhs x = 0

/-- The equation y + 3 = 0 --/
def equation : PolynomialEquation :=
  { lhs := λ y => y + 3
    rhs := λ _ => 0 }

/-- Theorem stating that the equation y + 3 = 0 is a linear equation with one variable --/
theorem equation_is_linear_one_variable : is_linear_one_variable equation := by
  sorry

#check equation_is_linear_one_variable

end NUMINAMATH_CALUDE_equation_is_linear_one_variable_l1927_192779


namespace NUMINAMATH_CALUDE_cups_count_l1927_192708

-- Define the cost of a single paper plate and cup
variable (plate_cost cup_cost : ℝ)

-- Define the number of cups in the second purchase
variable (cups_in_second_purchase : ℕ)

-- First condition: 100 plates and 200 cups cost $7.50
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

-- Second condition: 20 plates and cups_in_second_purchase cups cost $1.50
axiom second_purchase : 20 * plate_cost + cups_in_second_purchase * cup_cost = 1.50

-- Theorem to prove
theorem cups_count : cups_in_second_purchase = 40 := by
  sorry

end NUMINAMATH_CALUDE_cups_count_l1927_192708


namespace NUMINAMATH_CALUDE_janes_skirts_l1927_192745

/-- Proves that Jane bought 2 skirts given the problem conditions -/
theorem janes_skirts :
  let skirt_price : ℕ := 13
  let blouse_price : ℕ := 6
  let num_blouses : ℕ := 3
  let paid : ℕ := 100
  let change : ℕ := 56
  let total_spent : ℕ := paid - change
  ∃ (num_skirts : ℕ), num_skirts * skirt_price + num_blouses * blouse_price = total_spent ∧ num_skirts = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_skirts_l1927_192745


namespace NUMINAMATH_CALUDE_kendras_family_size_l1927_192748

/-- Proves the number of people in Kendra's family given the cookie baking scenario --/
theorem kendras_family_size 
  (cookies_per_batch : ℕ) 
  (num_batches : ℕ) 
  (chips_per_cookie : ℕ) 
  (chips_per_person : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 2)
  (h4 : chips_per_person = 18)
  : (cookies_per_batch * num_batches * chips_per_cookie) / chips_per_person = 4 := by
  sorry

#eval (12 * 3 * 2) / 18  -- Should output 4

end NUMINAMATH_CALUDE_kendras_family_size_l1927_192748


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1927_192782

theorem more_girls_than_boys (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 42 →
  boys_ratio = 3 →
  girls_ratio = 4 →
  (girls_ratio - boys_ratio) * (total_students / (boys_ratio + girls_ratio)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1927_192782


namespace NUMINAMATH_CALUDE_car_average_speed_l1927_192741

/-- Given a car traveling at different speeds for 5 hours, prove that its average speed is 94 km/h -/
theorem car_average_speed (v1 v2 v3 v4 v5 : ℝ) (h1 : v1 = 120) (h2 : v2 = 70) (h3 : v3 = 90) (h4 : v4 = 110) (h5 : v5 = 80) :
  (v1 + v2 + v3 + v4 + v5) / 5 = 94 := by
  sorry

#check car_average_speed

end NUMINAMATH_CALUDE_car_average_speed_l1927_192741


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1927_192701

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (a b : Line) (α : Plane) :
  parallel_line a b → 
  parallel_line_plane b α → 
  ¬ contained_in a α → 
  parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1927_192701


namespace NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1927_192747

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  q_nonzero : q ≠ 0
  is_geometric : ∀ n, as.a (k (n + 1)) = q * as.a (k n)
  k1_not_1 : k 1 ≠ 1
  k2_not_2 : k 2 ≠ 2
  k3_not_6 : k 3 ≠ 6

/-- The main theorem -/
theorem arithmetic_geometric_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1927_192747


namespace NUMINAMATH_CALUDE_dice_probability_l1927_192762

def num_dice : ℕ := 5
def dice_sides : ℕ := 6

def prob_all_same : ℚ := 1 / (dice_sides ^ (num_dice - 1))

def prob_four_same : ℚ := 
  (num_dice * (1 / dice_sides ^ (num_dice - 2)) * ((dice_sides - 1) / dice_sides))

theorem dice_probability : 
  prob_all_same + prob_four_same = 13 / 648 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l1927_192762


namespace NUMINAMATH_CALUDE_contractor_job_problem_l1927_192777

/-- A contractor's job problem -/
theorem contractor_job_problem
  (total_days : ℕ) (initial_workers : ℕ) (first_period : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_workers = 10)
  (h3 : first_period = 20)
  (h4 : remaining_days = 75)
  (h5 : first_period * initial_workers = (total_days * initial_workers) / 4) :
  ∃ (fired : ℕ), 
    fired = 2 ∧
    remaining_days * (initial_workers - fired) = 
      (total_days * initial_workers) - (first_period * initial_workers) :=
by sorry

end NUMINAMATH_CALUDE_contractor_job_problem_l1927_192777


namespace NUMINAMATH_CALUDE_hemisphere_properties_l1927_192712

/-- Properties of a hemisphere with base area 144π -/
theorem hemisphere_properties :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  (2 * π * r^2 + π * r^2 = 432 * π) ∧
  ((2 / 3) * π * r^3 = 1152 * π) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_properties_l1927_192712


namespace NUMINAMATH_CALUDE_probability_not_snow_l1927_192790

theorem probability_not_snow (p : ℚ) (h : p = 2 / 5) : 1 - p = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l1927_192790


namespace NUMINAMATH_CALUDE_factorial_22_representation_l1927_192791

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_22_representation (V R C : ℕ) :
  V < 10 ∧ R < 10 ∧ C < 10 →
  base_ten_representation (factorial 22) =
    [1, 1, 2, 4, 0, V, 4, 6, 1, 7, 4, R, C, 8, 8, 0, 0, 0, 0] →
  V + R + C = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_22_representation_l1927_192791


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1927_192716

theorem rope_cutting_problem :
  let total_length_feet : ℝ := 6
  let number_of_pieces : ℕ := 10
  let inches_per_foot : ℝ := 12
  let piece_length_inches : ℝ := total_length_feet * inches_per_foot / number_of_pieces
  piece_length_inches = 7.2 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1927_192716


namespace NUMINAMATH_CALUDE_toys_between_l1927_192734

theorem toys_between (n : ℕ) (pos_a pos_b : ℕ) (h1 : n = 19) (h2 : pos_a = 9) (h3 : pos_b = 15) :
  pos_b - pos_a - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_between_l1927_192734


namespace NUMINAMATH_CALUDE_maximal_colored_squares_correct_l1927_192742

/-- Given positive integers n and k where n > k^2 > 4, maximal_colored_squares
    returns the maximal number of unit squares that can be colored in an n × n grid,
    such that in any k-group there are two squares with the same color and
    two squares with different colors. -/
def maximal_colored_squares (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) : ℕ :=
  n * (k - 1)^2

/-- Theorem stating that maximal_colored_squares gives the correct result -/
theorem maximal_colored_squares_correct (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) :
  maximal_colored_squares n k h1 h2 = n * (k - 1)^2 := by
  sorry

#check maximal_colored_squares
#check maximal_colored_squares_correct

end NUMINAMATH_CALUDE_maximal_colored_squares_correct_l1927_192742


namespace NUMINAMATH_CALUDE_equation_solution_l1927_192727

theorem equation_solution : ∃ x : ℝ, (3 / (x + 2) = 2 / x) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1927_192727


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l1927_192725

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- The statement that x = 2 is sufficient but not necessary for a ∥ b -/
theorem x_eq_2_sufficient_not_necessary (x : ℝ) :
  (∀ x, x = 2 → are_parallel (1, x - 1) (x + 1, 3)) ∧
  (∃ x, x ≠ 2 ∧ are_parallel (1, x - 1) (x + 1, 3)) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l1927_192725


namespace NUMINAMATH_CALUDE_aria_apple_purchase_l1927_192715

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Aria needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Aria should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem aria_apple_purchase : apples_to_buy = 14 := by
  sorry

end NUMINAMATH_CALUDE_aria_apple_purchase_l1927_192715


namespace NUMINAMATH_CALUDE_inequality_holds_l1927_192794

theorem inequality_holds (x : ℝ) : 
  -1/2 ≤ x ∧ x < 45/8 → (4 * x^2) / ((1 - Real.sqrt (1 + 2*x))^2) < 2*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1927_192794


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1927_192746

/-- 
Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots,
prove that k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1927_192746


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1927_192767

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = -x -/
def symmetryLine (p : Point) : Prop := p.y = -p.x

/-- Definition of symmetry with respect to y = -x -/
def isSymmetric (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

theorem symmetry_of_point :
  let p1 : Point := ⟨1, 4⟩
  let p2 : Point := ⟨-4, -1⟩
  isSymmetric p1 p2 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1927_192767


namespace NUMINAMATH_CALUDE_limit_at_one_equals_five_l1927_192792

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem limit_at_one_equals_five :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (1 + Δx) - f 1) / Δx - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_one_equals_five_l1927_192792


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_1000_l1927_192721

def first_n_odd (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * i + 1)
def first_n_even (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * (i + 1))

theorem sum_difference_even_odd_1000 : 
  (first_n_even 1000).sum - (first_n_odd 1000).sum = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_1000_l1927_192721


namespace NUMINAMATH_CALUDE_multiplication_and_division_results_l1927_192755

theorem multiplication_and_division_results : 
  ((-2) * (-1/8) = 1/4) ∧ ((-5) / (6/5) = -25/6) := by sorry

end NUMINAMATH_CALUDE_multiplication_and_division_results_l1927_192755


namespace NUMINAMATH_CALUDE_f_max_value_l1927_192744

/-- The function f(x) defined as |x+2017| - |x-2016| -/
def f (x : ℝ) := |x + 2017| - |x - 2016|

/-- Theorem stating that the maximum value of f(x) is 4033 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 4033 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1927_192744


namespace NUMINAMATH_CALUDE_max_value_theorem_l1927_192739

theorem max_value_theorem (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  x₁ * x₂^2 * x₃ + x₁ * x₂ * x₃^2 ≤ 27/1024 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1927_192739


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l1927_192789

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (n : ℕ) (initial : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l1927_192789


namespace NUMINAMATH_CALUDE_marble_weight_l1927_192781

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  9 * marble_weight = 5 * car_weight →
  4 * car_weight = 120 →
  marble_weight = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_l1927_192781


namespace NUMINAMATH_CALUDE_binomial_16_12_l1927_192733

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_12_l1927_192733


namespace NUMINAMATH_CALUDE_total_amount_proof_l1927_192713

/-- Proves that the total amount of money divided into two parts is Rs. 2600 -/
theorem total_amount_proof (total : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (income : ℝ) :
  part1 + part2 = total →
  part1 = 1600 →
  rate1 = 0.05 →
  rate2 = 0.06 →
  part1 * rate1 + part2 * rate2 = income →
  income = 140 →
  total = 2600 := by
  sorry

#check total_amount_proof

end NUMINAMATH_CALUDE_total_amount_proof_l1927_192713


namespace NUMINAMATH_CALUDE_smallest_integers_difference_l1927_192732

theorem smallest_integers_difference : ∃ (n₁ n₂ : ℕ), 
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 4620 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_l1927_192732


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l1927_192768

/-- The perimeter of pentagon FGHIJ is 6, given that FG = GH = HI = IJ = 1 -/
theorem pentagon_perimeter (F G H I J : ℝ × ℝ) : 
  (dist F G = 1) → (dist G H = 1) → (dist H I = 1) → (dist I J = 1) →
  dist F G + dist G H + dist H I + dist I J + dist J F = 6 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l1927_192768


namespace NUMINAMATH_CALUDE_elmwood_population_l1927_192769

/-- The number of cities in the County of Elmwood -/
def num_cities : ℕ := 25

/-- The lower bound of the average population per city -/
def avg_pop_lower : ℕ := 3200

/-- The upper bound of the average population per city -/
def avg_pop_upper : ℕ := 3700

/-- The total population of the County of Elmwood -/
def total_population : ℕ := 86250

theorem elmwood_population :
  ∃ (avg_pop : ℚ),
    avg_pop > avg_pop_lower ∧
    avg_pop < avg_pop_upper ∧
    (num_cities : ℚ) * avg_pop = total_population :=
sorry

end NUMINAMATH_CALUDE_elmwood_population_l1927_192769


namespace NUMINAMATH_CALUDE_union_of_sets_l1927_192706

theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | x ≥ 0}) → 
  (B = {x : ℝ | x < 1}) → 
  A ∪ B = Set.univ := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1927_192706


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l1927_192740

/-- Calculates the amount each paying student contributes for lunch -/
def lunch_cost_per_paying_student (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) : ℚ :=
  let paying_students := total_students * (1 - free_lunch_percentage)
  total_cost / paying_students

theorem lunch_cost_theorem (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) 
  (h1 : total_students = 50)
  (h2 : free_lunch_percentage = 2/5)
  (h3 : total_cost = 210) :
  lunch_cost_per_paying_student total_students free_lunch_percentage total_cost = 7 := by
  sorry

#eval lunch_cost_per_paying_student 50 (2/5) 210

end NUMINAMATH_CALUDE_lunch_cost_theorem_l1927_192740


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1927_192797

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^9 + y^8) = 
  15*y^13 - y^12 + 6*y^11 + 5*y^10 - 7*y^9 - 2*y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1927_192797


namespace NUMINAMATH_CALUDE_power_two_2014_mod_7_l1927_192737

theorem power_two_2014_mod_7 :
  ∃ (k : ℤ), 2^2014 = 7 * k + 9 := by sorry

end NUMINAMATH_CALUDE_power_two_2014_mod_7_l1927_192737


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1927_192749

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 145862784)
  (h_hcf : Nat.gcd a b = 792) :
  Nat.lcm a b = 184256 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1927_192749


namespace NUMINAMATH_CALUDE_base8_digit_product_l1927_192784

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8679) = 392 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_l1927_192784


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1927_192796

/-- The volume of a cube given its side length -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

/-- The surface area of a cube given its side length -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area :
  ∀ (side_length1 side_length2 : ℝ),
  side_length1 > 0 →
  side_length2 > 0 →
  cube_volume side_length1 = 8 →
  cube_surface_area side_length2 = 3 * cube_surface_area side_length1 →
  cube_volume side_length2 = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1927_192796


namespace NUMINAMATH_CALUDE_fruit_display_total_l1927_192787

/-- Fruit display problem -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l1927_192787


namespace NUMINAMATH_CALUDE_town_population_division_l1927_192761

/-- Proves that in a town with a population of 480, if the population is divided into three equal parts, each part consists of 160 people. -/
theorem town_population_division (total_population : ℕ) (num_parts : ℕ) (part_size : ℕ) : 
  total_population = 480 → 
  num_parts = 3 → 
  total_population = num_parts * part_size → 
  part_size = 160 := by
  sorry

end NUMINAMATH_CALUDE_town_population_division_l1927_192761


namespace NUMINAMATH_CALUDE_exponent_division_l1927_192785

theorem exponent_division (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1927_192785


namespace NUMINAMATH_CALUDE_mold_radius_l1927_192783

/-- The radius of a circular mold with diameter 4 inches is 2 inches -/
theorem mold_radius (d : ℝ) (h : d = 4) : d / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mold_radius_l1927_192783


namespace NUMINAMATH_CALUDE_min_distance_point_l1927_192765

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The Fermat point of a triangle --/
def fermatPoint (t : Triangle) : ℝ × ℝ := sorry

/-- The sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (t : Triangle) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest angle in a triangle --/
def largestAngle (t : Triangle) : ℝ := sorry

/-- The vertex corresponding to the largest angle in a triangle --/
def largestAngleVertex (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The point that minimizes the sum of distances to the vertices of a triangle --/
theorem min_distance_point (t : Triangle) :
  ∃ (M : ℝ × ℝ), (∀ (p : ℝ × ℝ), sumOfDistances t M ≤ sumOfDistances t p) ∧
  ((largestAngle t < 2 * Real.pi / 3 ∧ M = fermatPoint t) ∨
   (largestAngle t ≥ 2 * Real.pi / 3 ∧ M = largestAngleVertex t)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_point_l1927_192765


namespace NUMINAMATH_CALUDE_fran_required_speed_l1927_192703

/-- Calculates the required average speed for Fran to cover the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) : 
  (joann_speed * joann_time) / fran_time = 120 / 7 := by
  sorry

#check fran_required_speed

end NUMINAMATH_CALUDE_fran_required_speed_l1927_192703


namespace NUMINAMATH_CALUDE_coefficient_a2_l1927_192760

/-- Given z = 1 + i and (z+x)^4 = a_4x^4 + a_3x^3 + a_2x^2 + a_1x + a_0, prove that a_2 = 12i -/
theorem coefficient_a2 (z : ℂ) (a_4 a_3 a_2 a_1 a_0 : ℂ) :
  z = 1 + Complex.I →
  (∀ x : ℂ, (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_2 = 12 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_coefficient_a2_l1927_192760


namespace NUMINAMATH_CALUDE_max_a_equals_min_f_l1927_192743

theorem max_a_equals_min_f : 
  let f (x : ℝ) := x^2 + 2*x - 6
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max)) ∧ 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x) →
  ∃ (a_max x_min : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max) ∧ 
    (∀ (x : ℝ), f x_min ≤ f x) ∧ 
    a_max = f x_min :=
by sorry

end NUMINAMATH_CALUDE_max_a_equals_min_f_l1927_192743


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l1927_192771

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l1927_192771


namespace NUMINAMATH_CALUDE_nesbitt_like_inequality_l1927_192775

theorem nesbitt_like_inequality (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_like_inequality_l1927_192775


namespace NUMINAMATH_CALUDE_face_washing_unit_is_liters_l1927_192754

/-- Represents units of volume measurement -/
inductive VolumeUnit
  | Liters
  | Milliliters
  | Grams

/-- Represents the amount of water used for face washing -/
def face_washing_amount : ℝ := 2

/-- Determines if a given volume unit is appropriate for face washing -/
def is_appropriate_unit (unit : VolumeUnit) : Prop :=
  match unit with
  | VolumeUnit.Liters => true
  | _ => false

theorem face_washing_unit_is_liters :
  is_appropriate_unit VolumeUnit.Liters = true :=
sorry

end NUMINAMATH_CALUDE_face_washing_unit_is_liters_l1927_192754


namespace NUMINAMATH_CALUDE_smallest_number_with_2_and_4_l1927_192728

def smallest_two_digit_number (a b : ℕ) : ℕ := 
  if a ≤ b then 10 * a + b else 10 * b + a

theorem smallest_number_with_2_and_4 : 
  smallest_two_digit_number 2 4 = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_2_and_4_l1927_192728


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l1927_192795

/-- Two triangles are similar -/
def SimilarTriangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- The similarity ratio between two triangles -/
def SimilarityRatio (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- The perimeter of a triangle -/
def Perimeter (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Theorem: If two triangles are similar with a ratio of 1:2, then their perimeters have the same ratio -/
theorem perimeter_ratio_of_similar_triangles (ABC DEF : Set (Fin 3 → ℝ × ℝ)) :
  SimilarTriangles ABC DEF →
  SimilarityRatio ABC DEF = 1 / 2 →
  Perimeter ABC / Perimeter DEF = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l1927_192795


namespace NUMINAMATH_CALUDE_max_balls_count_l1927_192709

/-- Represents the count of balls -/
def n : ℕ := 45

/-- The number of green balls in the first 45 -/
def initial_green : ℕ := 41

/-- The number of green balls in each subsequent batch of 10 -/
def subsequent_green : ℕ := 9

/-- The total number of balls in each subsequent batch -/
def batch_size : ℕ := 10

/-- The minimum percentage of green balls required -/
def min_green_percentage : ℚ := 92 / 100

theorem max_balls_count :
  ∀ m : ℕ, m > n →
    (initial_green : ℚ) / n < min_green_percentage ∨
    (initial_green + (m - n) / batch_size * subsequent_green : ℚ) / m < min_green_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_balls_count_l1927_192709


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1927_192729

theorem divisibility_theorem (n : ℕ) (x : ℤ) (h : n ≥ 1) :
  ∃ k : ℤ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = k * (x-1)^3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1927_192729


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1927_192766

def chicken_soup_quantity : ℕ := 6
def chicken_soup_price : ℚ := 3/2

def tomato_soup_quantity : ℕ := 3
def tomato_soup_price : ℚ := 5/4

def vegetable_soup_quantity : ℕ := 4
def vegetable_soup_price : ℚ := 7/4

def clam_chowder_quantity : ℕ := 2
def clam_chowder_price : ℚ := 2

def french_onion_soup_quantity : ℕ := 1
def french_onion_soup_price : ℚ := 9/5

def minestrone_soup_quantity : ℕ := 5
def minestrone_soup_price : ℚ := 17/10

def total_cost : ℚ := 
  chicken_soup_quantity * chicken_soup_price +
  tomato_soup_quantity * tomato_soup_price +
  vegetable_soup_quantity * vegetable_soup_price +
  clam_chowder_quantity * clam_chowder_price +
  french_onion_soup_quantity * french_onion_soup_price +
  minestrone_soup_quantity * minestrone_soup_price

theorem total_cost_is_correct : total_cost = 3405/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1927_192766


namespace NUMINAMATH_CALUDE_certain_number_proof_l1927_192736

theorem certain_number_proof (x : ℝ) : (((x + 10) * 2) / 2) - 2 = 88 / 2 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1927_192736


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1927_192720

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1927_192720


namespace NUMINAMATH_CALUDE_inequality_proof_l1927_192711

def f (a x : ℝ) : ℝ := |x - a| + 1

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a := 1 / m + 1 / n
  (∀ x, f a x ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2) →
  m + 2 * n ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1927_192711


namespace NUMINAMATH_CALUDE_ab_equals_six_l1927_192714

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l1927_192714


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l1927_192738

/-- Represents the linear regression equation for sales volume and price -/
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

/-- The selling price in yuan -/
def selling_price : ℝ := 10

/-- Theorem stating that the estimated sales volume is approximately 100 pieces when the selling price is 10 yuan -/
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation selling_price - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l1927_192738


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1927_192722

theorem arithmetic_calculations :
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) ∧
  (1 + (-1/2) + 1/3 + (-1/6) = 2/3) ∧
  (3 + 1/4 + (-2 - 3/5) + 5 + 3/4 + (-8 - 2/5) = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1927_192722


namespace NUMINAMATH_CALUDE_convention_handshakes_l1927_192758

theorem convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : 
  num_companies = 5 → 
  reps_per_company = 4 → 
  (num_companies * reps_per_company * (num_companies * reps_per_company - reps_per_company)) / 2 = 160 := by
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l1927_192758


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1927_192750

theorem quadratic_equation_equivalence (m : ℝ) : 
  (∀ x, x^2 - m*x + 6 = 0 ↔ (x - 3)^2 = 3) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1927_192750


namespace NUMINAMATH_CALUDE_oliver_card_collection_l1927_192764

theorem oliver_card_collection (monster_club : ℕ) (alien_baseball : ℕ) (battle_gremlins : ℕ) 
  (h1 : monster_club = 2 * alien_baseball)
  (h2 : battle_gremlins = 48)
  (h3 : battle_gremlins = 3 * alien_baseball) :
  monster_club = 32 := by
  sorry

end NUMINAMATH_CALUDE_oliver_card_collection_l1927_192764


namespace NUMINAMATH_CALUDE_cat_food_finished_l1927_192702

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day after a given number of days -/
def dayAfter (d : Day) (n : ℕ) : Day :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (match d with
    | Day.Monday => Day.Tuesday
    | Day.Tuesday => Day.Wednesday
    | Day.Wednesday => Day.Thursday
    | Day.Thursday => Day.Friday
    | Day.Friday => Day.Saturday
    | Day.Saturday => Day.Sunday
    | Day.Sunday => Day.Monday) n

/-- The amount of food consumed by the cat per day -/
def dailyConsumption : ℚ := 1/5 + 1/6

/-- The total amount of food in the box -/
def totalFood : ℚ := 10

/-- Theorem stating when the cat will finish the food -/
theorem cat_food_finished :
  ∃ (n : ℕ), n * dailyConsumption > totalFood ∧
  (n - 1) * dailyConsumption ≤ totalFood ∧
  dayAfter Day.Monday (n - 1) = Day.Wednesday :=
by sorry


end NUMINAMATH_CALUDE_cat_food_finished_l1927_192702


namespace NUMINAMATH_CALUDE_dividend_calculation_l1927_192759

/-- Proves that given a divisor of -4 2/3, a quotient of -57 1/5, and a remainder of 2 1/9, the dividend is equal to 269 2/45. -/
theorem dividend_calculation (divisor quotient remainder dividend : ℚ) : 
  divisor = -14/3 →
  quotient = -286/5 →
  remainder = 19/9 →
  dividend = divisor * quotient + remainder →
  dividend = 12107/45 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1927_192759


namespace NUMINAMATH_CALUDE_system_solution_existence_l1927_192704

/-- The system of equations has at least one solution if and only if b ≥ -2√2 - 1/4 -/
theorem system_solution_existence (b : ℝ) :
  (∃ a x y : ℝ, y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1927_192704
