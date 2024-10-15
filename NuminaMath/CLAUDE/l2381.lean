import Mathlib

namespace NUMINAMATH_CALUDE_function_value_at_two_l2381_238136

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2

/-- The main theorem stating that g(2) = 5 for any function satisfying the functional equation -/
theorem function_value_at_two (g : ℝ → ℝ) (h : FunctionalEquation g) : g 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2381_238136


namespace NUMINAMATH_CALUDE_modulus_of_complex_difference_l2381_238192

theorem modulus_of_complex_difference (z : ℂ) : z = -1 - Complex.I → Complex.abs (2 - z) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_difference_l2381_238192


namespace NUMINAMATH_CALUDE_simplify_rational_function_l2381_238166

theorem simplify_rational_function (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_rational_function_l2381_238166


namespace NUMINAMATH_CALUDE_ahmed_hassan_apple_ratio_l2381_238194

/-- Ahmed's orchard has 8 orange trees and an unknown number of apple trees. -/
def ahmed_orange_trees : ℕ := 8

/-- Hassan's orchard has 1 apple tree. -/
def hassan_apple_trees : ℕ := 1

/-- Hassan's orchard has 2 orange trees. -/
def hassan_orange_trees : ℕ := 2

/-- The difference in total trees between Ahmed's and Hassan's orchards. -/
def tree_difference : ℕ := 9

/-- Ahmed's apple trees -/
def ahmed_apple_trees : ℕ := ahmed_orange_trees + tree_difference - (hassan_apple_trees + hassan_orange_trees)

theorem ahmed_hassan_apple_ratio :
  ahmed_apple_trees = 4 * hassan_apple_trees := by
  sorry

end NUMINAMATH_CALUDE_ahmed_hassan_apple_ratio_l2381_238194


namespace NUMINAMATH_CALUDE_division_problem_l2381_238167

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2381_238167


namespace NUMINAMATH_CALUDE_probability_of_mathematics_letter_l2381_238172

theorem probability_of_mathematics_letter :
  let total_letters : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / total_letters
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_of_mathematics_letter_l2381_238172


namespace NUMINAMATH_CALUDE_percentage_problem_l2381_238165

theorem percentage_problem (percentage : ℝ) : 
  (percentage * 100 - 20 = 60) → percentage = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2381_238165


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2381_238131

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2381_238131


namespace NUMINAMATH_CALUDE_intramural_teams_l2381_238100

theorem intramural_teams (num_boys : ℕ) (num_girls : ℕ) (max_teams : ℕ) :
  num_boys = 32 →
  max_teams = 8 →
  (∃ (boys_per_team : ℕ), num_boys = max_teams * boys_per_team) →
  (∃ (girls_per_team : ℕ), num_girls = max_teams * girls_per_team) →
  ∃ (k : ℕ), num_girls = 8 * k :=
by sorry

end NUMINAMATH_CALUDE_intramural_teams_l2381_238100


namespace NUMINAMATH_CALUDE_sqrt_ratio_equals_sqrt_five_l2381_238114

theorem sqrt_ratio_equals_sqrt_five : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (4 + 1) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equals_sqrt_five_l2381_238114


namespace NUMINAMATH_CALUDE_folded_paper_height_approx_l2381_238155

/-- The height of a folded sheet of paper -/
def folded_paper_height (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Approximation of 2^10 -/
def approx_2_10 : ℝ := 1000

theorem folded_paper_height_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |folded_paper_height 0.1 20 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_folded_paper_height_approx_l2381_238155


namespace NUMINAMATH_CALUDE_graph_below_line_l2381_238168

noncomputable def f (x : ℝ) := x * Real.log x - x^2 - 1

theorem graph_below_line (x : ℝ) (h : x > 0) : Real.log x - Real.exp x + 1 < 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_below_line_l2381_238168


namespace NUMINAMATH_CALUDE_set_A_representation_l2381_238195

def A : Set (ℝ × ℝ) := {(x, y) | 3 * x + y = 11 ∧ x - y = 1}

theorem set_A_representation : A = {(3, 2)} := by sorry

end NUMINAMATH_CALUDE_set_A_representation_l2381_238195


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2381_238111

def total_people : ℕ := 10
def math_majors : ℕ := 5
def physics_majors : ℕ := 3
def chemistry_majors : ℕ := 2

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.choose total_people math_majors
  let consecutive_arrangements := total_people
  (consecutive_arrangements : ℚ) / total_arrangements = 5 / 126 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2381_238111


namespace NUMINAMATH_CALUDE_abc_sum_and_squares_l2381_238148

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_and_squares_l2381_238148


namespace NUMINAMATH_CALUDE_x_to_neg_y_equals_half_l2381_238183

theorem x_to_neg_y_equals_half (x y : ℝ) (h : Real.sqrt (x + y - 3) = -(x - 2*y)^2) : 
  x^(-y) = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_x_to_neg_y_equals_half_l2381_238183


namespace NUMINAMATH_CALUDE_flour_per_cake_l2381_238127

/-- The amount of flour needed for each cake given the initial conditions -/
theorem flour_per_cake 
  (traci_flour : ℕ) 
  (harris_flour : ℕ) 
  (traci_cakes : ℕ) 
  (harris_cakes : ℕ) 
  (h1 : traci_flour = 500)
  (h2 : harris_flour = 400)
  (h3 : traci_cakes = 9)
  (h4 : harris_cakes = 9) :
  (traci_flour + harris_flour) / (traci_cakes + harris_cakes) = 50 := by
  sorry

end NUMINAMATH_CALUDE_flour_per_cake_l2381_238127


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2381_238143

-- Define the vertices of the trapezoid
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (8, 0)
def v3 : ℝ × ℝ := (6, 10)
def v4 : ℝ × ℝ := (2, 10)

-- Define the trapezoid
def isosceles_trapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  -- Add conditions for isosceles trapezoid here
  True

-- Calculate the area of the trapezoid
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  -- Add area calculation here
  0

-- Theorem statement
theorem isosceles_trapezoid_area :
  isosceles_trapezoid v1 v2 v3 v4 →
  trapezoid_area v1 v2 v3 v4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2381_238143


namespace NUMINAMATH_CALUDE_blanket_collection_l2381_238101

/-- Proves the number of blankets collected on the last day of a three-day collection drive -/
theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (total_blankets : ℕ) : 
  team_size = 15 → 
  first_day_per_person = 2 → 
  total_blankets = 142 → 
  total_blankets - (team_size * first_day_per_person + 3 * (team_size * first_day_per_person)) = 22 := by
  sorry

#check blanket_collection

end NUMINAMATH_CALUDE_blanket_collection_l2381_238101


namespace NUMINAMATH_CALUDE_smallest_multiple_with_digit_sum_l2381_238139

def N : ℕ := 5 * 10^223 - 10^220 - 10^49 - 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_multiple_with_digit_sum :
  (N % 2009 = 0) ∧
  (sum_of_digits N = 2009) ∧
  (∀ m : ℕ, m < N → (m % 2009 = 0 ∧ sum_of_digits m = 2009) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_digit_sum_l2381_238139


namespace NUMINAMATH_CALUDE_quadratic_polynomial_k_value_l2381_238177

/-- A polynomial is quadratic if its degree is exactly 2 -/
def IsQuadratic (p : Polynomial ℝ) : Prop :=
  p.degree = 2

theorem quadratic_polynomial_k_value :
  ∀ k : ℝ,
    IsQuadratic (Polynomial.monomial 3 (k - 2) + Polynomial.monomial 2 k + Polynomial.monomial 1 (-2) + Polynomial.monomial 0 (-6))
    → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_k_value_l2381_238177


namespace NUMINAMATH_CALUDE_sum_31_22_base4_l2381_238124

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_31_22_base4 :
  toBase4 (31 + 22) = [3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_31_22_base4_l2381_238124


namespace NUMINAMATH_CALUDE_second_number_calculation_l2381_238123

theorem second_number_calculation (A B : ℕ) (h1 : A - B = 88) (h2 : A = 110) : B = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l2381_238123


namespace NUMINAMATH_CALUDE_average_of_numbers_sixth_and_seventh_sum_l2381_238171

def numbers : List ℝ := [54, 55, 57, 58, 59, 63, 65, 65]

theorem average_of_numbers : 
  (List.sum numbers) / (List.length numbers : ℝ) = 60 :=
by sorry

theorem sixth_and_seventh_sum : 
  List.sum (List.drop 5 (List.take 7 numbers)) = 54 :=
by sorry

#check average_of_numbers
#check sixth_and_seventh_sum

end NUMINAMATH_CALUDE_average_of_numbers_sixth_and_seventh_sum_l2381_238171


namespace NUMINAMATH_CALUDE_remarkable_number_l2381_238144

theorem remarkable_number : ∃ (x : ℝ), 
  x > 0 ∧ 
  (x - ⌊x⌋) * ⌊x⌋ = (x - ⌊x⌋)^2 ∧ 
  x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_remarkable_number_l2381_238144


namespace NUMINAMATH_CALUDE_exists_valid_solution_l2381_238182

def mother_charge : ℝ := 6.50
def child_charge_per_year : ℝ := 0.65
def total_bill : ℝ := 13.00

def is_valid_solution (twin_age youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem exists_valid_solution :
  ∃ (twin_age youngest_age : ℕ), is_valid_solution twin_age youngest_age ∧ (youngest_age = 2 ∨ youngest_age = 4) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_solution_l2381_238182


namespace NUMINAMATH_CALUDE_fraction_equality_l2381_238137

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x * y) / (x^2 - x * y) = y / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2381_238137


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2381_238142

theorem solution_set_equivalence (x : ℝ) :
  (|(8 - x) / 4| < 3) ↔ (4 < x ∧ x < 20) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2381_238142


namespace NUMINAMATH_CALUDE_flower_bunch_problem_l2381_238156

/-- The number of flowers in each bunch initially -/
def initial_flowers_per_bunch : ℕ := sorry

/-- The number of bunches initially -/
def initial_bunches : ℕ := 8

/-- The number of flowers per bunch in the alternative scenario -/
def alternative_flowers_per_bunch : ℕ := 12

/-- The number of bunches in the alternative scenario -/
def alternative_bunches : ℕ := 6

theorem flower_bunch_problem :
  initial_flowers_per_bunch * initial_bunches = alternative_flowers_per_bunch * alternative_bunches ∧
  initial_flowers_per_bunch = 9 := by sorry

end NUMINAMATH_CALUDE_flower_bunch_problem_l2381_238156


namespace NUMINAMATH_CALUDE_shelby_rain_time_l2381_238176

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  speed_sunny : ℝ  -- Speed when not raining (miles per hour)
  speed_rainy : ℝ  -- Speed when raining (miles per hour)
  total_time : ℝ   -- Total journey time (minutes)
  stop_time : ℝ    -- Total stop time (minutes)
  total_distance : ℝ -- Total distance covered (miles)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given Shelby's driving conditions, she drove 48 minutes in the rain -/
theorem shelby_rain_time (scenario : DrivingScenario) 
  (h1 : scenario.speed_sunny = 40)
  (h2 : scenario.speed_rainy = 25)
  (h3 : scenario.total_time = 75)
  (h4 : scenario.stop_time = 15)
  (h5 : scenario.total_distance = 28) :
  time_in_rain scenario = 48 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l2381_238176


namespace NUMINAMATH_CALUDE_cos_45_degrees_l2381_238170

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l2381_238170


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_empty_l2381_238193

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9 < 0}
def B : Set ℝ := {x | 2 ≤ x + 1 ∧ x + 1 ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_empty (m : ℝ) : 
  A ∩ C m = ∅ ↔ m ≤ -4 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_empty_l2381_238193


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2381_238112

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2381_238112


namespace NUMINAMATH_CALUDE_select_and_assign_volunteers_eq_30_l2381_238115

/-- The number of ways to select and assign volunteers for a two-day event -/
def select_and_assign_volunteers : ℕ :=
  let total_volunteers : ℕ := 5
  let selected_volunteers : ℕ := 4
  let days : ℕ := 2
  let volunteers_per_day : ℕ := 2
  (total_volunteers.choose selected_volunteers) *
  ((selected_volunteers.choose volunteers_per_day) * (days.factorial))

/-- Theorem stating that the number of ways to select and assign volunteers is 30 -/
theorem select_and_assign_volunteers_eq_30 :
  select_and_assign_volunteers = 30 := by
  sorry

end NUMINAMATH_CALUDE_select_and_assign_volunteers_eq_30_l2381_238115


namespace NUMINAMATH_CALUDE_max_value_function_l2381_238197

theorem max_value_function (x : ℝ) (h : x < 5/4) :
  4*x - 2 + 1/(4*x - 5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l2381_238197


namespace NUMINAMATH_CALUDE_painted_cube_problem_l2381_238103

theorem painted_cube_problem (n : ℕ) (h1 : n > 2) :
  (2 * (n - 2)^2 = 2 * (n - 2) * n^2) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l2381_238103


namespace NUMINAMATH_CALUDE_book_sale_loss_l2381_238104

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_l2381_238104


namespace NUMINAMATH_CALUDE_at_op_four_neg_one_l2381_238196

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

/-- Theorem stating that 4 @ (-1) = -4 -/
theorem at_op_four_neg_one : at_op 4 (-1) = -4 := by sorry

end NUMINAMATH_CALUDE_at_op_four_neg_one_l2381_238196


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2381_238126

theorem uncool_parents_count (total_students cool_dads cool_moms both_cool : ℕ) 
  (h1 : total_students = 35)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 11) :
  total_students - (cool_dads + cool_moms - both_cool) = 8 := by
sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l2381_238126


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2381_238173

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_sum : i + i^2 + i^3 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2381_238173


namespace NUMINAMATH_CALUDE_soccer_balls_added_l2381_238118

theorem soccer_balls_added (initial_balls final_balls : ℕ) (h1 : initial_balls = 6) (h2 : final_balls = 24) :
  final_balls - initial_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_added_l2381_238118


namespace NUMINAMATH_CALUDE_complex_modulus_two_thirds_plus_three_i_l2381_238175

theorem complex_modulus_two_thirds_plus_three_i :
  Complex.abs (Complex.ofReal (2/3) + Complex.I * 3) = Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_two_thirds_plus_three_i_l2381_238175


namespace NUMINAMATH_CALUDE_min_value_and_angle_l2381_238119

theorem min_value_and_angle (A : Real) : 
  0 ≤ A ∧ A ≤ 2 * Real.pi →
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    2 * Real.sin (A / 2) + Real.sin A ≤ 2 * Real.sin (θ / 2) + Real.sin θ) →
  2 * Real.sin (A / 2) + Real.sin A = -4 ∧ A = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_angle_l2381_238119


namespace NUMINAMATH_CALUDE_quadratic_derivative_bound_l2381_238141

-- Define a quadratic function
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem quadratic_derivative_bound :
  ∃ A : ℝ,
    (∀ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
      |b| ≤ A) ∧
    (∃ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) ∧
      |b| = A) ∧
    (∀ A' : ℝ,
      (∀ a b c : ℝ,
        (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
        |b| ≤ A') →
      A ≤ A') :=
by sorry

end NUMINAMATH_CALUDE_quadratic_derivative_bound_l2381_238141


namespace NUMINAMATH_CALUDE_b_share_proof_l2381_238132

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 10

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 15

/-- The total wages for the work in Rupees -/
def total_wages : ℕ := 5000

/-- The share of wages B should receive when working together with A -/
def b_share : ℕ := 3000

/-- Theorem stating that B's share of the wages when working with A is 3000 Rupees -/
theorem b_share_proof : 
  b_share = (b_days * total_wages) / (a_days + b_days) := by sorry

end NUMINAMATH_CALUDE_b_share_proof_l2381_238132


namespace NUMINAMATH_CALUDE_equation_solutions_l2381_238133

theorem equation_solutions :
  (∃! x : ℝ, (2 / x = 3 / (x + 2)) ∧ x = 4) ∧
  (¬ ∃ x : ℝ, 1 / (x - 2) = (1 - x) / (2 - x) - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2381_238133


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_lines_l2381_238198

-- Define the circles and point
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 4
def point_P : ℝ × ℝ := (3, 1)

-- Define the intersection of two circles
def circles_intersect (C1 C2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y, C1 x y ∧ C2 x y

-- Define a tangent line to a circle passing through a point
def is_tangent_line (a b c : ℝ) (C : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  (∀ x y, C x y → a*x + b*y + c ≠ 0) ∧
  (∃ x y, C x y ∧ a*x + b*y + c = 0) ∧
  a*(P.1) + b*(P.2) + c = 0

-- Theorem statement
theorem circle_intersection_and_tangent_lines :
  (circles_intersect circle_C circle_C1) ∧
  (is_tangent_line 0 1 (-1) circle_C point_P) ∧
  (is_tangent_line 12 5 (-41) circle_C point_P) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_lines_l2381_238198


namespace NUMINAMATH_CALUDE_total_phone_cost_l2381_238164

def phone_cost : ℝ := 1000
def monthly_contract : ℝ := 200
def case_cost_percentage : ℝ := 0.20
def headphones_cost_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

def total_cost : ℝ :=
  phone_cost +
  (monthly_contract * months_in_year) +
  (phone_cost * case_cost_percentage) +
  (phone_cost * case_cost_percentage * headphones_cost_ratio)

theorem total_phone_cost : total_cost = 3700 := by
  sorry

end NUMINAMATH_CALUDE_total_phone_cost_l2381_238164


namespace NUMINAMATH_CALUDE_car_trip_speed_proof_l2381_238186

/-- Proves that given a trip of 8 hours with an average speed of 34 miles per hour,
    where the first 6 hours are traveled at 30 miles per hour,
    the average speed for the remaining 2 hours is 46 miles per hour. -/
theorem car_trip_speed_proof :
  let total_time : ℝ := 8
  let first_part_time : ℝ := 6
  let first_part_speed : ℝ := 30
  let total_average_speed : ℝ := 34
  let remaining_time : ℝ := total_time - first_part_time
  let total_distance : ℝ := total_time * total_average_speed
  let first_part_distance : ℝ := first_part_time * first_part_speed
  let remaining_distance : ℝ := total_distance - first_part_distance
  let remaining_speed : ℝ := remaining_distance / remaining_time
  remaining_speed = 46 := by sorry

end NUMINAMATH_CALUDE_car_trip_speed_proof_l2381_238186


namespace NUMINAMATH_CALUDE_square_of_sum_l2381_238145

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2381_238145


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l2381_238138

theorem number_puzzle_solution :
  ∃ x : ℚ, x^2 + 100 = (x - 12)^2 ∧ x = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l2381_238138


namespace NUMINAMATH_CALUDE_total_situps_l2381_238105

-- Define the performance profiles for each participant
def adam_performance (round : ℕ) : ℕ :=
  match round with
  | 1 => 40
  | 2 => 35
  | 3 => 30
  | 4 => 20
  | _ => 0

def barney_performance (round : ℕ) : ℕ :=
  max (45 - 3 * (round - 1)) 0

def carrie_performance (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 90
  | 3 | 4 => 80
  | 5 => 70
  | _ => 0

def jerrie_performance (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 95
  | 3 | 4 => 101
  | 5 => 94
  | 6 => 87
  | 7 => 80
  | _ => 0

-- Define the number of rounds for each participant
def adam_rounds : ℕ := 4
def barney_rounds : ℕ := 6
def carrie_rounds : ℕ := 5
def jerrie_rounds : ℕ := 7

-- Define the total sit-ups for each participant
def adam_total : ℕ := (List.range adam_rounds).map adam_performance |>.sum
def barney_total : ℕ := (List.range barney_rounds).map barney_performance |>.sum
def carrie_total : ℕ := (List.range carrie_rounds).map carrie_performance |>.sum
def jerrie_total : ℕ := (List.range jerrie_rounds).map jerrie_performance |>.sum

-- Theorem statement
theorem total_situps :
  adam_total + barney_total + carrie_total + jerrie_total = 1353 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_l2381_238105


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2381_238106

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2381_238106


namespace NUMINAMATH_CALUDE_tangent_at_one_minimum_a_l2381_238185

noncomputable section

def f (x : ℝ) := (1/6) * x^3 + (1/2) * x - x * Real.log x

def domain : Set ℝ := {x | x > 0}

def interval : Set ℝ := {x | 1/Real.exp 1 < x ∧ x < Real.exp 1}

theorem tangent_at_one (x : ℝ) (hx : x ∈ domain) :
  (f x - f 1) = 0 * (x - 1) := by sorry

theorem minimum_a :
  ∃ a : ℝ, (∀ x ∈ interval, f x < a) ∧
  (∀ b : ℝ, (∀ x ∈ interval, f x < b) → a ≤ b) ∧
  a = (1/6) * (Real.exp 1)^3 - (1/2) * (Real.exp 1) := by sorry

end

end NUMINAMATH_CALUDE_tangent_at_one_minimum_a_l2381_238185


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2381_238128

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 3 * 315 * 612 = 36600 := by sorry

-- Problem 2
theorem problem_2 : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2381_238128


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l2381_238199

variable (k : ℕ)

def first_term : ℕ → ℕ := λ k => 3 * k^2 + 2
def common_difference : ℕ := 2
def num_terms : ℕ → ℕ := λ k => 4 * k + 3

theorem arithmetic_series_sum :
  (λ k : ℕ => (num_terms k) * (2 * first_term k + (num_terms k - 1) * common_difference) / 2) =
  (λ k : ℕ => 12 * k^3 + 28 * k^2 + 28 * k + 12) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l2381_238199


namespace NUMINAMATH_CALUDE_movie_date_communication_l2381_238109

theorem movie_date_communication (p : ℝ) (h1 : p = 0.9) :
  p * p + (1 - p) * (1 - p) = 0.82 := by
  sorry

end NUMINAMATH_CALUDE_movie_date_communication_l2381_238109


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2381_238110

/-- Represents the measurement error in a rectangle's dimensions and area -/
structure RectangleMeasurement where
  length_excess : ℝ  -- Percentage excess in length measurement
  width_deficit : ℝ  -- Percentage deficit in width measurement
  area_error : ℝ     -- Percentage error in calculated area

/-- Theorem stating the relationship between measurement errors in a rectangle -/
theorem rectangle_measurement_error 
  (r : RectangleMeasurement) 
  (h1 : r.length_excess = 8)
  (h2 : r.area_error = 2.6) :
  r.width_deficit = 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2381_238110


namespace NUMINAMATH_CALUDE_probability_of_correct_answer_l2381_238134

theorem probability_of_correct_answer (options : Nat) (correct_options : Nat) : 
  options = 4 → correct_options = 1 → (correct_options : ℚ) / options = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_correct_answer_l2381_238134


namespace NUMINAMATH_CALUDE_triangle_inequality_with_additional_segment_l2381_238188

theorem triangle_inequality_with_additional_segment
  (a b c d : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_d_positive : d > 0)
  (a₁ : ℝ) (h_a₁ : a₁ = min a d)
  (b₁ : ℝ) (h_b₁ : b₁ = min b d)
  (c₁ : ℝ) (h_c₁ : c₁ = min c d) :
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_additional_segment_l2381_238188


namespace NUMINAMATH_CALUDE_tournament_theorem_l2381_238130

/-- Represents a team in the tournament -/
inductive Team : Type
| A
| B
| C

/-- Represents the state of a player (active or eliminated) -/
inductive PlayerState : Type
| Active
| Eliminated

/-- Represents the state of the tournament -/
structure TournamentState :=
  (team_players : Team → Fin 9 → PlayerState)
  (matches_played : ℕ)
  (champion_wins : ℕ)

/-- The rules of the tournament -/
def tournament_rules (initial_state : TournamentState) : Prop :=
  ∀ (t : Team), ∃ (i : Fin 9), initial_state.team_players t i = PlayerState.Active

/-- The condition for a team to be eliminated -/
def team_eliminated (state : TournamentState) (t : Team) : Prop :=
  ∀ (i : Fin 9), state.team_players t i = PlayerState.Eliminated

/-- The condition for the tournament to end -/
def tournament_ended (state : TournamentState) : Prop :=
  ∃ (t1 t2 : Team), t1 ≠ t2 ∧ team_eliminated state t1 ∧ team_eliminated state t2

/-- The main theorem to prove -/
theorem tournament_theorem 
  (initial_state : TournamentState) 
  (h_rules : tournament_rules initial_state) :
  (∃ (final_state : TournamentState), 
    tournament_ended final_state ∧ 
    final_state.champion_wins ≥ 9) ∧
  (∀ (final_state : TournamentState),
    tournament_ended final_state → 
    final_state.champion_wins = 11 → 
    final_state.matches_played ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_tournament_theorem_l2381_238130


namespace NUMINAMATH_CALUDE_area_of_triangle_qpo_l2381_238181

/-- Represents a parallelogram ABCD with specific properties -/
structure SpecialParallelogram where
  -- The area of the parallelogram
  area : ℝ
  -- DP bisects BC
  dp_bisects_bc : Bool
  -- CQ bisects AD
  cq_bisects_ad : Bool
  -- DP divides triangle BCD into regions of area k/4 and 3k/4
  dp_divides_bcd : Bool

/-- Theorem stating the area of triangle QPO in the special parallelogram -/
theorem area_of_triangle_qpo (ABCD : SpecialParallelogram) :
  let k := ABCD.area
  let area_qpo := (9 : ℝ) / 8 * k
  ABCD.dp_bisects_bc ∧ ABCD.cq_bisects_ad ∧ ABCD.dp_divides_bcd →
  area_qpo = (9 : ℝ) / 8 * k :=
by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_qpo_l2381_238181


namespace NUMINAMATH_CALUDE_distance_downstream_20min_l2381_238102

/-- Calculates the distance traveled downstream by a boat -/
def distance_traveled (boat_speed wind_speed current_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + current_speed + 0.1 * wind_speed) * time

/-- Theorem: Distance traveled downstream in 20 minutes -/
theorem distance_downstream_20min (c w : ℝ) :
  distance_traveled 26 w c (1/3) = 26/3 + c/3 + 0.1*w/3 := by
  sorry

end NUMINAMATH_CALUDE_distance_downstream_20min_l2381_238102


namespace NUMINAMATH_CALUDE_min_milk_candies_l2381_238108

/-- Represents the number of chocolate candies -/
def chocolate : ℕ := sorry

/-- Represents the number of watermelon candies -/
def watermelon : ℕ := sorry

/-- Represents the number of milk candies -/
def milk : ℕ := sorry

/-- The number of watermelon candies is at most 3 times the number of chocolate candies -/
axiom watermelon_condition : watermelon ≤ 3 * chocolate

/-- The number of milk candies is at least 4 times the number of chocolate candies -/
axiom milk_condition : milk ≥ 4 * chocolate

/-- The total number of chocolate and watermelon candies is no less than 2020 -/
axiom total_condition : chocolate + watermelon ≥ 2020

/-- The minimum number of milk candies required is 2020 -/
theorem min_milk_candies : milk ≥ 2020 := by sorry

end NUMINAMATH_CALUDE_min_milk_candies_l2381_238108


namespace NUMINAMATH_CALUDE_loot_box_loss_l2381_238140

theorem loot_box_loss (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ)
  (h1 : cost_per_box = 5)
  (h2 : avg_value_per_box = 3.5)
  (h3 : total_spent = 40) :
  (total_spent / cost_per_box) * (cost_per_box - avg_value_per_box) = 12 :=
by sorry

end NUMINAMATH_CALUDE_loot_box_loss_l2381_238140


namespace NUMINAMATH_CALUDE_five_T_three_l2381_238179

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem to prove
theorem five_T_three : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_l2381_238179


namespace NUMINAMATH_CALUDE_johnnys_age_l2381_238158

theorem johnnys_age : ∃ (age : ℕ), 
  (age + 2 = 2 * (age - 3)) ∧ age = 8 := by sorry

end NUMINAMATH_CALUDE_johnnys_age_l2381_238158


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2381_238153

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {3, 4, 5} →
  (U \ A) ∩ B = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2381_238153


namespace NUMINAMATH_CALUDE_max_circumference_circle_in_parabola_l2381_238117

/-- A circle located inside the parabola x^2 = 4y and passing through its vertex -/
structure CircleInParabola where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 ≤ radius^2 → x^2 ≤ 4*y
  passes_through_vertex : (0 - center.1)^2 + (0 - center.2)^2 = radius^2

/-- The maximum circumference of a circle located inside the parabola x^2 = 4y 
    and passing through its vertex is 4π -/
theorem max_circumference_circle_in_parabola :
  ∃ (C : CircleInParabola), ∀ (D : CircleInParabola), 2 * Real.pi * C.radius ≥ 2 * Real.pi * D.radius ∧
  2 * Real.pi * C.radius = 4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_max_circumference_circle_in_parabola_l2381_238117


namespace NUMINAMATH_CALUDE_inverse_function_value_l2381_238107

/-- Given that f is the inverse function of g(x) = ax, and f(4) = 2, prove that a = 2 -/
theorem inverse_function_value (a : ℝ) (f g : ℝ → ℝ) :
  (∀ x, g x = a * x) →  -- g is defined as g(x) = ax
  (∀ x, f (g x) = x) →  -- f is the inverse function of g
  (∀ x, g (f x) = x) →  -- f is the inverse function of g (reverse composition)
  f 4 = 2 →             -- f(4) = 2
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_value_l2381_238107


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2381_238122

theorem equal_roots_quadratic (a : ℕ) : 
  (∀ x : ℝ, x^2 - a*x + (a + 3) = 0 → (∃! y : ℝ, y^2 - a*y + (a + 3) = 0)) → 
  a = 6 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2381_238122


namespace NUMINAMATH_CALUDE_parabola_point_coordinate_l2381_238161

/-- Theorem: For a point on the parabola x^2 = 4y with focus (0, 1),
    if its distance from the focus is 5, then its x-coordinate is ±4. -/
theorem parabola_point_coordinate (x y : ℝ) : 
  x^2 = 4*y →  -- Point (x, y) is on the parabola
  (x - 0)^2 + (y - 1)^2 = 5^2 →  -- Distance from (x, y) to focus (0, 1) is 5
  x = 4 ∨ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_coordinate_l2381_238161


namespace NUMINAMATH_CALUDE_greatest_rational_root_quadratic_l2381_238191

theorem greatest_rational_root_quadratic (a b c : ℕ) (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∀ (p q : ℤ), q ≠ 0 → a * (p / q)^2 + b * (p / q) + c = 0 →
  (p : ℚ) / q ≤ (-1 : ℚ) / 99 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_quadratic_l2381_238191


namespace NUMINAMATH_CALUDE_colonization_combinations_l2381_238184

/-- The number of habitable planets --/
def total_planets : ℕ := 18

/-- The number of Earth-like planets --/
def earth_like_planets : ℕ := 9

/-- The number of Mars-like planets --/
def mars_like_planets : ℕ := 9

/-- The resource units required to colonize an Earth-like planet --/
def earth_like_resource : ℕ := 3

/-- The resource units required to colonize a Mars-like planet --/
def mars_like_resource : ℕ := 2

/-- The total resource units available for colonization --/
def total_resources : ℕ := 27

/-- The number of Earth-like planets that can be colonized --/
def colonized_earth_like : ℕ := 7

/-- The number of Mars-like planets that can be colonized --/
def colonized_mars_like : ℕ := 3

theorem colonization_combinations : 
  (Nat.choose earth_like_planets colonized_earth_like) * 
  (Nat.choose mars_like_planets colonized_mars_like) = 3024 :=
by sorry

end NUMINAMATH_CALUDE_colonization_combinations_l2381_238184


namespace NUMINAMATH_CALUDE_correct_dispatch_plans_l2381_238190

/-- The number of teachers available for selection -/
def total_teachers : ℕ := 8

/-- The number of teachers to be selected -/
def selected_teachers : ℕ := 4

/-- The number of remote areas -/
def remote_areas : ℕ := 4

/-- Function to calculate the number of ways to select teachers -/
def select_teachers : ℕ :=
  let with_a_c := Nat.choose (total_teachers - 3) (selected_teachers - 2)
  let without_a_c := Nat.choose (total_teachers - 2) selected_teachers
  with_a_c + without_a_c

/-- Function to calculate the number of ways to arrange teachers in areas -/
def arrange_teachers : ℕ := Nat.factorial selected_teachers

/-- The total number of different dispatch plans -/
def total_dispatch_plans : ℕ := select_teachers * arrange_teachers

theorem correct_dispatch_plans : total_dispatch_plans = 600 := by
  sorry

end NUMINAMATH_CALUDE_correct_dispatch_plans_l2381_238190


namespace NUMINAMATH_CALUDE_total_rainfall_equals_1368_l2381_238189

def average_monthly_rainfall (year : ℕ) : ℝ :=
  35 + 3 * (year - 2010)

def yearly_rainfall (year : ℕ) : ℝ :=
  12 * average_monthly_rainfall year

def total_rainfall_2010_to_2012 : ℝ :=
  yearly_rainfall 2010 + yearly_rainfall 2011 + yearly_rainfall 2012

theorem total_rainfall_equals_1368 :
  total_rainfall_2010_to_2012 = 1368 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_equals_1368_l2381_238189


namespace NUMINAMATH_CALUDE_total_fish_in_lake_l2381_238160

/-- Represents the number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- Represents the number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- Represents the number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- Represents the number of white ducks -/
def white_ducks : ℕ := 3

/-- Represents the number of black ducks -/
def black_ducks : ℕ := 7

/-- Represents the number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- Theorem stating that the total number of fish in the lake is 157 -/
theorem total_fish_in_lake : 
  white_ducks * fish_per_white_duck + 
  black_ducks * fish_per_black_duck + 
  multicolor_ducks * fish_per_multicolor_duck = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_in_lake_l2381_238160


namespace NUMINAMATH_CALUDE_not_perfect_square_with_1234_divisors_l2381_238187

/-- A natural number with exactly 1234 divisors is not a perfect square. -/
theorem not_perfect_square_with_1234_divisors (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x | x ∣ n} ∧ d.card = 1234) → ¬∃ (m : ℕ), n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_with_1234_divisors_l2381_238187


namespace NUMINAMATH_CALUDE_parallel_line_angle_theorem_l2381_238149

-- Define the structure for our geometric configuration
structure ParallelLineConfig where
  -- Angle QTV
  angle_QTV : ℝ
  -- Angle SUV
  angle_SUV : ℝ
  -- Angle TVU
  angle_TVU : ℝ
  -- Assumption that PQ and RS are parallel
  parallel_PQ_RS : True
  -- Assumptions about the given angles
  h_QTV : angle_QTV = 30
  h_SUV : angle_SUV = 40

-- Theorem statement
theorem parallel_line_angle_theorem (config : ParallelLineConfig) :
  config.angle_TVU = 70 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_angle_theorem_l2381_238149


namespace NUMINAMATH_CALUDE_joe_hvac_cost_l2381_238152

/-- The total cost of an HVAC system with given parameters. -/
def hvac_total_cost (num_zones : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  num_zones * vents_per_zone * cost_per_vent

/-- Theorem stating that the total cost of Joe's HVAC system is $20,000. -/
theorem joe_hvac_cost :
  hvac_total_cost 2 5 2000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_cost_l2381_238152


namespace NUMINAMATH_CALUDE_complex_root_of_unity_product_l2381_238120

theorem complex_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_of_unity_product_l2381_238120


namespace NUMINAMATH_CALUDE_simplify_equation_l2381_238157

theorem simplify_equation : ∀ x : ℝ, 
  3 * x + 4.8 * x - 10 * x = 11 * (1 / 5) ↔ -2.2 * x = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_equation_l2381_238157


namespace NUMINAMATH_CALUDE_new_person_weight_is_105_l2381_238135

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that under the given conditions, the weight of the new person is 105 kg -/
theorem new_person_weight_is_105 :
  weight_of_new_person 8 85 2.5 = 105 := by
  sorry

#eval weight_of_new_person 8 85 2.5

end NUMINAMATH_CALUDE_new_person_weight_is_105_l2381_238135


namespace NUMINAMATH_CALUDE_unseen_sum_is_21_l2381_238178

/-- Represents a standard six-sided die -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ (i : Fin 6), faces i + faces (5 - i) = 7)

/-- The sum of the numbers on the unseen faces of two dice -/
def unseen_sum (d1 d2 : Die) (v1 v2 v3 : Fin 6) (w1 w2 w3 : Fin 6) : Nat :=
  (7 - d1.faces v1) + (7 - d1.faces v2) + (7 - d1.faces v3) +
  (7 - d2.faces w1) + (7 - d2.faces w2) + (7 - d2.faces w3)

theorem unseen_sum_is_21 (d1 d2 : Die) :
  d1.faces 0 = 6 → d1.faces 1 = 2 → d1.faces 2 = 3 →
  d2.faces 0 = 1 → d2.faces 1 = 4 → d2.faces 2 = 5 →
  unseen_sum d1 d2 0 1 2 0 1 2 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_unseen_sum_is_21_l2381_238178


namespace NUMINAMATH_CALUDE_kelvin_frog_paths_l2381_238180

/-- Represents a position in the coordinate plane -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Represents a move that Kelvin can make -/
inductive Move
  | Walk : Move
  | Jump : Move

/-- Defines the possible moves Kelvin can make from a given position -/
def possibleMoves (pos : Position) : List Position :=
  [
    {x := pos.x, y := pos.y + 1},     -- Walk up
    {x := pos.x + 1, y := pos.y},     -- Walk right
    {x := pos.x + 1, y := pos.y + 1}, -- Walk diagonally
    {x := pos.x, y := pos.y + 2},     -- Jump up
    {x := pos.x + 2, y := pos.y},     -- Jump right
    {x := pos.x + 1, y := pos.y + 1}  -- Jump diagonally
  ]

/-- Counts the number of ways to reach the target position from the start position -/
def countWays (start : Position) (target : Position) : ℕ :=
  sorry

theorem kelvin_frog_paths : countWays {x := 0, y := 0} {x := 6, y := 8} = 1831830 := by
  sorry

end NUMINAMATH_CALUDE_kelvin_frog_paths_l2381_238180


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_areas_l2381_238146

theorem equilateral_triangle_circle_areas (s : ℝ) (h : s = 12) :
  let r := s / 2
  let sector_area := (π * r^2) / 3
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let shaded_area := 2 * (sector_area - triangle_area)
  ∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 33 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_areas_l2381_238146


namespace NUMINAMATH_CALUDE_negation_of_all_odd_double_even_l2381_238169

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def A : Set ℤ := {n : ℤ | is_odd n}
def B : Set ℤ := {n : ℤ | is_even n}

theorem negation_of_all_odd_double_even :
  (¬ ∀ x ∈ A, (2 * x) ∈ B) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_odd_double_even_l2381_238169


namespace NUMINAMATH_CALUDE_pet_supply_store_dog_food_l2381_238174

/-- Given a pet supply store with cat food and dog food, prove the number of bags of dog food. -/
theorem pet_supply_store_dog_food (cat_food : ℕ) (difference : ℕ) : 
  cat_food = 327 → difference = 273 → cat_food + difference = 600 := by
  sorry

end NUMINAMATH_CALUDE_pet_supply_store_dog_food_l2381_238174


namespace NUMINAMATH_CALUDE_four_wheelers_count_l2381_238121

/-- Given a parking lot with only 2-wheelers and 4-wheelers, and a total of 58 wheels,
    prove that the number of 4-wheelers can be expressed in terms of the number of 2-wheelers. -/
theorem four_wheelers_count (x y : ℕ) (h1 : 2 * x + 4 * y = 58) :
  y = (29 - x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_four_wheelers_count_l2381_238121


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l2381_238163

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  (garden.length * step_length * (garden.width * step_length) : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 15 20
  let step_length := 2
  let yield_per_sqft := 1/2
  expected_potato_yield garden step_length yield_per_sqft = 600 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l2381_238163


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l2381_238151

/-- A furniture shop owner charges 10% more than the cost price. If a customer paid Rs. 8800 for a computer table, then the cost price of the computer table was Rs. 8000. -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8800)
  (h2 : markup_percentage = 0.10) : 
  ∃ (cost_price : ℝ), cost_price = 8000 ∧ selling_price = cost_price * (1 + markup_percentage) := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l2381_238151


namespace NUMINAMATH_CALUDE_quadratic_properties_l2381_238113

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

-- Theorem statement
theorem quadratic_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (f a 0 = -2 → 
    (∃ x y : ℝ, (x = 1/2 ∨ x = -1/2) ∧ y = -9/4 ∧ 
    ∀ t : ℝ, f a t ≥ f a x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2381_238113


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2381_238147

theorem diophantine_equation_solution (x y z : ℤ) :
  2 * x^2 + 3 * y^2 = z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2381_238147


namespace NUMINAMATH_CALUDE_sum_of_ages_in_ten_years_l2381_238125

/-- Theorem: Sum of ages in 10 years -/
theorem sum_of_ages_in_ten_years (my_current_age brother_current_age : ℕ) : 
  my_current_age = 20 →
  my_current_age + 10 = 2 * (brother_current_age + 10) →
  (my_current_age + 10) + (brother_current_age + 10) = 45 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_ten_years_l2381_238125


namespace NUMINAMATH_CALUDE_curve_is_line_l2381_238129

/-- The equation (x^2 + y^2 - 2)√(x-3) = 0 represents a line -/
theorem curve_is_line : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - 2) * Real.sqrt (x - 3) = 0 → y = a * x + b :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_l2381_238129


namespace NUMINAMATH_CALUDE_lucas_L10_units_digit_l2381_238154

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucas_L10_units_digit_l2381_238154


namespace NUMINAMATH_CALUDE_customers_who_tipped_l2381_238159

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l2381_238159


namespace NUMINAMATH_CALUDE_broker_commission_slump_l2381_238150

theorem broker_commission_slump (X : ℝ) (h : X > 0) :
  let Y : ℝ := (4 / 5) * X
  let income_unchanged := 0.04 * X = 0.05 * Y
  let slump_percentage := (1 - Y / X) * 100
  income_unchanged → slump_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_broker_commission_slump_l2381_238150


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2381_238162

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + 29/13 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2381_238162


namespace NUMINAMATH_CALUDE_sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l2381_238116

-- Part 1
theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 = 5 := by sorry

-- Part 2
theorem square_of_sum (a b c : ℝ) :
  (a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c := by sorry

-- Part 3
theorem sum_of_three_squares (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a*b + b*c + a*c = 11) :
  a^2 + b^2 + c^2 = 14 := by sorry

-- Part 4
theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  a^4 + b^4 + c^4 = 18 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l2381_238116
