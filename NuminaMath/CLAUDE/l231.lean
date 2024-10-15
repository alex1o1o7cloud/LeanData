import Mathlib

namespace NUMINAMATH_CALUDE_grade_percentage_calculation_l231_23140

theorem grade_percentage_calculation (total_students : ℕ) 
  (a_both : ℕ) (b_both : ℕ) (c_both : ℕ) (d_c : ℕ) :
  total_students = 40 →
  a_both = 4 →
  b_both = 6 →
  c_both = 3 →
  d_c = 2 →
  (((a_both + b_both + c_both + d_c : ℚ) / total_students) * 100 : ℚ) = 37.5 := by
sorry

end NUMINAMATH_CALUDE_grade_percentage_calculation_l231_23140


namespace NUMINAMATH_CALUDE_library_books_calculation_l231_23182

theorem library_books_calculation (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 60 → 
  return_rate = 7/10 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 57 := by
sorry

end NUMINAMATH_CALUDE_library_books_calculation_l231_23182


namespace NUMINAMATH_CALUDE_smallest_number_with_property_l231_23105

theorem smallest_number_with_property : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    m % 2 ≠ 1 ∨
    m % 3 ≠ 2 ∨
    m % 4 ≠ 3 ∨
    m % 5 ≠ 4 ∨
    m % 6 ≠ 5 ∨
    m % 7 ≠ 6 ∨
    m % 8 ≠ 7 ∨
    m % 9 ≠ 8 ∨
    m % 10 ≠ 9) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_property_l231_23105


namespace NUMINAMATH_CALUDE_simplify_radical_product_l231_23168

theorem simplify_radical_product (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (18 * y) * Real.sqrt (32 * y) = 30 * y * Real.sqrt (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l231_23168


namespace NUMINAMATH_CALUDE_unique_solution_condition_l231_23142

theorem unique_solution_condition (h : ℝ) (h_neq_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l231_23142


namespace NUMINAMATH_CALUDE_complex_conjugate_root_l231_23110

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A complex number is a root of a polynomial if the polynomial evaluates to zero at that number -/
def is_root (f : RealPolynomial) (z : ℂ) : Prop := f z.re = 0 ∧ f z.im = 0

theorem complex_conjugate_root (f : RealPolynomial) (a b : ℝ) :
  is_root f (Complex.mk a b) → is_root f (Complex.mk a (-b)) :=
by sorry

end NUMINAMATH_CALUDE_complex_conjugate_root_l231_23110


namespace NUMINAMATH_CALUDE_chord_length_l231_23133

theorem chord_length (r : ℝ) (h : r = 10) : 
  let chord_length := 2 * (r^2 - (r/2)^2).sqrt
  chord_length = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l231_23133


namespace NUMINAMATH_CALUDE_age_difference_l231_23177

theorem age_difference (older_age younger_age : ℕ) : 
  older_age + younger_age = 74 → older_age = 38 → older_age - younger_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l231_23177


namespace NUMINAMATH_CALUDE_probability_in_painted_cube_l231_23197

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (three_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : ℚ :=
  (cube.three_face_cubes * cube.no_face_cubes : ℚ) / (cube.total_cubes * (cube.total_cubes - 1) / 2)

/-- The theorem to be proved -/
theorem probability_in_painted_cube :
  ∃ (cube : PaintedCube),
    cube.size = 5 ∧
    cube.total_cubes = 125 ∧
    cube.three_face_cubes = 1 ∧
    cube.no_face_cubes = 76 ∧
    probability_three_and_none cube = 2 / 205 :=
sorry

end NUMINAMATH_CALUDE_probability_in_painted_cube_l231_23197


namespace NUMINAMATH_CALUDE_cookies_in_box_l231_23114

theorem cookies_in_box (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) (total_calories : ℕ) :
  cookies_per_bag = 20 →
  calories_per_cookie = 20 →
  total_calories = 1600 →
  total_calories / (cookies_per_bag * calories_per_cookie) = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookies_in_box_l231_23114


namespace NUMINAMATH_CALUDE_find_a_l231_23119

def f (x : ℝ) := 3 * (x - 1) + 2

theorem find_a : ∃ a : ℝ, f a = 5 ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_find_a_l231_23119


namespace NUMINAMATH_CALUDE_salt_percentage_in_water_l231_23192

def salt_mass : ℝ := 10
def water_mass : ℝ := 40

theorem salt_percentage_in_water :
  (salt_mass / water_mass) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_salt_percentage_in_water_l231_23192


namespace NUMINAMATH_CALUDE_marble_division_l231_23131

theorem marble_division (x : ℚ) : 
  (4 * x + 2) + (2 * x - 1) + (3 * x + 3) = 100 → 
  2 * x - 1 = 61 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_division_l231_23131


namespace NUMINAMATH_CALUDE_simplify_expression_l231_23185

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (x + 4)*(3*x - 2) = 4*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l231_23185


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l231_23121

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℕ) : ℕ := width * height

/-- The sum of the areas of four rectangles -/
def totalArea (w1 h1 w2 h2 w3 h3 w4 h4 : ℕ) : ℕ :=
  rectangleArea w1 h1 + rectangleArea w2 h2 + rectangleArea w3 h3 + rectangleArea w4 h4

/-- Theorem stating that the sum of the areas of four specific rectangles is 56 -/
theorem sum_of_rectangle_areas :
  totalArea 7 6 3 2 3 1 5 1 = 56 := by
  sorry

#eval totalArea 7 6 3 2 3 1 5 1

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l231_23121


namespace NUMINAMATH_CALUDE_ivanov_net_worth_is_2300000_l231_23122

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℤ :=
  let apartment_value : ℤ := 3000000
  let car_value : ℤ := 900000
  let bank_deposit : ℤ := 300000
  let securities_value : ℤ := 200000
  let liquid_cash : ℤ := 100000
  let mortgage_balance : ℤ := 1500000
  let car_loan_balance : ℤ := 500000
  let relatives_debt : ℤ := 200000
  let total_assets : ℤ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities : ℤ := mortgage_balance + car_loan_balance + relatives_debt
  total_assets - total_liabilities

theorem ivanov_net_worth_is_2300000 : ivanov_net_worth = 2300000 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_net_worth_is_2300000_l231_23122


namespace NUMINAMATH_CALUDE_sum_of_ages_l231_23126

theorem sum_of_ages (a b c : ℕ) : 
  a = 11 → 
  (a - 3) + (b - 3) + (c - 3) = 6 * (a - 3) → 
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l231_23126


namespace NUMINAMATH_CALUDE_jake_has_eight_peaches_l231_23145

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference in peaches between Steven and Jake -/
def difference : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - difference

theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eight_peaches_l231_23145


namespace NUMINAMATH_CALUDE_algebraic_simplification_l231_23104

theorem algebraic_simplification (a b : ℝ) : 
  14 * a^8 * b^4 / (7 * a^4 * b^4) - a^3 * a - (2 * a^2)^2 = -3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l231_23104


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l231_23151

theorem sqrt_sum_equality : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l231_23151


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_product_l231_23157

theorem geometric_sequence_sum_product (a b c : ℝ) : 
  (∃ q : ℝ, b = a * q ∧ c = b * q) →  -- geometric sequence condition
  a + b + c = 14 →                    -- sum condition
  a * b * c = 64 →                    -- product condition
  ((a = 8 ∧ b = 4 ∧ c = 2) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_product_l231_23157


namespace NUMINAMATH_CALUDE_number_of_students_l231_23123

-- Define the number of 8th-grade students
variable (x : ℕ)

-- Define the conditions
axiom retail_threshold : x < 250
axiom wholesale_threshold : x + 60 ≥ 250
axiom retail_cost : 240 / x * 240 = 240
axiom wholesale_cost : 260 / (x + 60) * (x + 60) = 260
axiom equal_cost : 260 / (x + 60) * 288 = 240 / x * 240

-- Theorem to prove
theorem number_of_students : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l231_23123


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l231_23115

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  let a : ℝ := -3
  let b : ℝ := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 59 ∧ f x_min = -49 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l231_23115


namespace NUMINAMATH_CALUDE_solve_equation_l231_23129

theorem solve_equation : 
  let y : ℚ := 45 / (8 - 3/7)
  y = 315 / 53 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l231_23129


namespace NUMINAMATH_CALUDE_x0_value_l231_23198

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 2015) : x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l231_23198


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l231_23160

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l231_23160


namespace NUMINAMATH_CALUDE_smallest_winning_number_l231_23183

def ian_action (x : ℕ) : ℕ := 3 * x

def marcella_action (x : ℕ) : ℕ := x + 150

def game_sequence (m : ℕ) : ℕ := 
  ian_action (marcella_action (ian_action (marcella_action (ian_action m))))

theorem smallest_winning_number : 
  ∀ m : ℕ, 0 ≤ m ∧ m ≤ 1999 →
    (m < 112 → 
      game_sequence m ≤ 5000 ∧ 
      marcella_action (game_sequence m) ≤ 5000 ∧ 
      ian_action (marcella_action (game_sequence m)) > 5000) →
    (game_sequence 112 ≤ 5000 ∧ 
     marcella_action (game_sequence 112) ≤ 5000 ∧ 
     ian_action (marcella_action (game_sequence 112)) > 5000) :=
by sorry

#check smallest_winning_number

end NUMINAMATH_CALUDE_smallest_winning_number_l231_23183


namespace NUMINAMATH_CALUDE_point_on_line_l231_23153

/-- Given a line passing through points (0, 2) and (-10, 0),
    prove that the point (25, 7) lies on this line. -/
theorem point_on_line : ∀ (x y : ℝ),
  (x = 25 ∧ y = 7) →
  (y - 2) * 10 = (x - 0) * 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l231_23153


namespace NUMINAMATH_CALUDE_point_not_on_line_l231_23128

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def are_collinear (A B C : Point3D) : Prop :=
  ∃ k : ℝ, (C.x - A.x, C.y - A.y, C.z - A.z) = k • (B.x - A.x, B.y - A.y, B.z - A.z)

theorem point_not_on_line : 
  let A : Point3D := ⟨-1, 1, 2⟩
  let B : Point3D := ⟨3, 6, -1⟩
  let C : Point3D := ⟨7, 9, 0⟩
  ¬(are_collinear A B C) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l231_23128


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l231_23132

/-- Proves that the initial volume of a mixture is 45 litres, given the initial and final ratios of milk to water --/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 4 / 1)
  (h2 : initial_milk / (initial_water + 18) = 4 / 3) :
  initial_milk + initial_water = 45 :=
by sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l231_23132


namespace NUMINAMATH_CALUDE_rain_probability_l231_23174

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 4/5) (hn : n = 5) :
  1 - (1 - p)^n = 3124/3125 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l231_23174


namespace NUMINAMATH_CALUDE_total_students_at_concert_l231_23149

/-- The number of buses going to the concert -/
def num_buses : ℕ := 18

/-- The number of students each bus took -/
def students_per_bus : ℕ := 65

/-- Theorem stating the total number of students who went to the concert -/
theorem total_students_at_concert : num_buses * students_per_bus = 1170 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_concert_l231_23149


namespace NUMINAMATH_CALUDE_greatest_b_for_non_range_l231_23188

theorem greatest_b_for_non_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 15 ≠ -9) ↔ b ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_for_non_range_l231_23188


namespace NUMINAMATH_CALUDE_vector_perpendicular_implies_x_value_l231_23141

/-- Given vectors a and b in R^2, if a is perpendicular to 2a + b, then the x-coordinate of b is 10 -/
theorem vector_perpendicular_implies_x_value (a b : ℝ × ℝ) :
  a = (-2, 3) →
  b.2 = -2 →
  (a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0) →
  b.1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_implies_x_value_l231_23141


namespace NUMINAMATH_CALUDE_correct_operation_l231_23193

theorem correct_operation (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l231_23193


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l231_23171

theorem sqrt_inequality_solution_set (x : ℝ) : 
  (Real.sqrt (x + 3) < 2) ↔ (x ∈ Set.Icc (-3) 1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l231_23171


namespace NUMINAMATH_CALUDE_boys_without_calculators_l231_23186

/-- Given a class with boys and girls, and information about calculator possession,
    prove that the number of boys without calculators is 5. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calc : ℕ)
  (girls_with_calc : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calc = 30)
  (h3 : girls_with_calc = 15) :
  total_boys - (total_with_calc - girls_with_calc) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l231_23186


namespace NUMINAMATH_CALUDE_car_speed_l231_23124

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 624 ∧ time = 2 + 2/5 → speed = distance / time → speed = 260 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l231_23124


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l231_23178

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l231_23178


namespace NUMINAMATH_CALUDE_lawn_mowing_difference_l231_23189

theorem lawn_mowing_difference (spring_mowings summer_mowings : ℕ) 
  (h1 : spring_mowings = 8) 
  (h2 : summer_mowings = 5) : 
  spring_mowings - summer_mowings = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_difference_l231_23189


namespace NUMINAMATH_CALUDE_range_of_abc_squared_l231_23136

theorem range_of_abc_squared (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_squared_l231_23136


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l231_23172

theorem arccos_lt_arcsin_iff (x : ℝ) : Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l231_23172


namespace NUMINAMATH_CALUDE_same_color_probability_l231_23118

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 10 ∧ white = 8) :
  let total := green + white
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  (prob_green + prob_white : ℚ) = 73 / 153 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l231_23118


namespace NUMINAMATH_CALUDE_proportion_equality_l231_23169

theorem proportion_equality (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l231_23169


namespace NUMINAMATH_CALUDE_tom_balloons_remaining_l231_23138

/-- Given that Tom has 30 violet balloons initially and gives away 16 balloons,
    prove that he has 14 violet balloons remaining. -/
theorem tom_balloons_remaining (initial : ℕ) (given_away : ℕ) (h1 : initial = 30) (h2 : given_away = 16) :
  initial - given_away = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_remaining_l231_23138


namespace NUMINAMATH_CALUDE_investment_interest_rate_l231_23163

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part : ℝ) 
  (second_rate : ℝ) 
  (total_interest : ℝ) :
  total_investment = 3000 →
  first_part = 300 →
  second_part = total_investment - first_part →
  second_rate = 5 →
  total_interest = 144 →
  total_interest = (first_part * (3 : ℝ) / 100) + (second_part * second_rate / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l231_23163


namespace NUMINAMATH_CALUDE_circle_properties_correct_l231_23199

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, compute its center and radius -/
def computeCircleProperties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_properties_correct (eq : CircleEquation) 
  (h : eq = CircleEquation.mk 4 (-8) 4 (-16) 20) : 
  computeCircleProperties eq = CircleProperties.mk (1, 2) 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_correct_l231_23199


namespace NUMINAMATH_CALUDE_lemons_for_lemonade_l231_23134

/-- Given that 30 lemons make 40 gallons of lemonade, 
    prove that 7.5 lemons are needed for 10 gallons. -/
theorem lemons_for_lemonade :
  let lemons_for_40 : ℚ := 30
  let gallons_40 : ℚ := 40
  let target_gallons : ℚ := 10
  (lemons_for_40 / gallons_40) * target_gallons = 7.5 := by sorry

end NUMINAMATH_CALUDE_lemons_for_lemonade_l231_23134


namespace NUMINAMATH_CALUDE_cone_volume_l231_23139

/-- Given a cone with slant height 2 and base area π, its volume is (√3 * π) / 3 -/
theorem cone_volume (s : ℝ) (a : ℝ) (v : ℝ) 
  (h_slant : s = 2) 
  (h_area : a = π) 
  (h_volume : v = (Real.sqrt 3 * π) / 3) : 
  v = (Real.sqrt 3 * π) / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l231_23139


namespace NUMINAMATH_CALUDE_cube_diff_multiple_implies_sum_squares_multiple_of_sum_l231_23175

theorem cube_diff_multiple_implies_sum_squares_multiple_of_sum (a b c : ℕ) : 
  a < 2017 → b < 2017 → c < 2017 →
  a ≠ b → b ≠ c → a ≠ c →
  (∃ k₁ k₂ k₃ : ℤ, a^3 - b^3 = k₁ * 2017 ∧ b^3 - c^3 = k₂ * 2017 ∧ c^3 - a^3 = k₃ * 2017) →
  ∃ m : ℤ, a^2 + b^2 + c^2 = m * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_cube_diff_multiple_implies_sum_squares_multiple_of_sum_l231_23175


namespace NUMINAMATH_CALUDE_missing_number_proof_l231_23125

theorem missing_number_proof (some_number : ℤ) : 
  some_number = 3 → |9 - 8 * (some_number - 12)| - |5 - 11| = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l231_23125


namespace NUMINAMATH_CALUDE_disease_probabilities_l231_23184

/-- Represents a disease with its incidence rate and probability of showing symptom S -/
structure Disease where
  incidenceRate : ℝ
  probSymptomS : ℝ

/-- Given three diseases and their properties, proves statements about probabilities -/
theorem disease_probabilities (d₁ d₂ d₃ : Disease)
  (h₁ : d₁.incidenceRate = 0.02 ∧ d₁.probSymptomS = 0.4)
  (h₂ : d₂.incidenceRate = 0.05 ∧ d₂.probSymptomS = 0.18)
  (h₃ : d₃.incidenceRate = 0.005 ∧ d₃.probSymptomS = 0.6)
  (h_no_other : ∀ d, d ≠ d₁ ∧ d ≠ d₂ ∧ d ≠ d₃ → d.probSymptomS = 0) :
  let p_s := d₁.incidenceRate * d₁.probSymptomS +
             d₂.incidenceRate * d₂.probSymptomS +
             d₃.incidenceRate * d₃.probSymptomS
  p_s = 0.02 ∧
  (d₁.incidenceRate * d₁.probSymptomS) / p_s = 0.4 ∧
  (d₂.incidenceRate * d₂.probSymptomS) / p_s = 0.45 :=
by sorry


end NUMINAMATH_CALUDE_disease_probabilities_l231_23184


namespace NUMINAMATH_CALUDE_expression_simplification_l231_23102

theorem expression_simplification (x : ℝ) : 2*x + 3*x^2 + 1 - (6 - 2*x - 3*x^2) = 6*x^2 + 4*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l231_23102


namespace NUMINAMATH_CALUDE_prob_higher_2012_l231_23196

-- Define the probability of guessing correctly
def p : ℝ := 0.25

-- Define the complementary probability
def q : ℝ := 1 - p

-- Define the binomial probability function
def binomProb (n : ℕ) (k : ℕ) : ℝ :=
  (n.choose k) * (p ^ k) * (q ^ (n - k))

-- Define the probability of passing in 2011
def prob2011 : ℝ :=
  1 - (binomProb 20 0 + binomProb 20 1 + binomProb 20 2)

-- Define the probability of passing in 2012
def prob2012 : ℝ :=
  1 - (binomProb 40 0 + binomProb 40 1 + binomProb 40 2 + binomProb 40 3 + binomProb 40 4 + binomProb 40 5)

-- Theorem statement
theorem prob_higher_2012 : prob2012 > prob2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_higher_2012_l231_23196


namespace NUMINAMATH_CALUDE_negation_of_difference_l231_23154

theorem negation_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end NUMINAMATH_CALUDE_negation_of_difference_l231_23154


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l231_23108

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 140) (h2 : speed2 = 40) :
  (speed1 + speed2) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l231_23108


namespace NUMINAMATH_CALUDE_floor_plus_square_equals_72_l231_23156

theorem floor_plus_square_equals_72 : 
  ∃! (x : ℝ), x > 0 ∧ ⌊x⌋ + x^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_square_equals_72_l231_23156


namespace NUMINAMATH_CALUDE_zoo_visitors_l231_23101

/-- Given the number of visitors on Friday and the ratio of Saturday visitors to Friday visitors,
    prove that the number of visitors on Saturday is equal to the product of the Friday visitors and the ratio. -/
theorem zoo_visitors (friday_visitors : ℕ) (saturday_ratio : ℕ) :
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = friday_visitors * saturday_ratio :=
by sorry

/-- Example with the given values -/
example : 
  let friday_visitors : ℕ := 1250
  let saturday_ratio : ℕ := 3
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = 3750 :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l231_23101


namespace NUMINAMATH_CALUDE_parabola_point_position_l231_23152

theorem parabola_point_position 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_below : 2 < a + b + c) : 
  2 < c + b + a := by sorry

end NUMINAMATH_CALUDE_parabola_point_position_l231_23152


namespace NUMINAMATH_CALUDE_point_coordinates_l231_23190

def Point := ℝ × ℝ

def x_coordinate (p : Point) : ℝ := p.1

def distance_to_x_axis (p : Point) : ℝ := |p.2|

theorem point_coordinates (P : Point) 
  (h1 : x_coordinate P = -3)
  (h2 : distance_to_x_axis P = 5) :
  P = (-3, 5) ∨ P = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l231_23190


namespace NUMINAMATH_CALUDE_child_ticket_price_is_correct_l231_23195

/-- Calculates the price of a child's ticket given the group composition, adult ticket price, senior discount, and total bill. -/
def childTicketPrice (totalPeople adultCount seniorCount childCount : ℕ) 
                     (adultPrice : ℚ) (seniorDiscount : ℚ) (totalBill : ℚ) : ℚ :=
  let seniorPrice := adultPrice * (1 - seniorDiscount)
  let adultTotal := adultPrice * adultCount
  let seniorTotal := seniorPrice * seniorCount
  let childTotal := totalBill - adultTotal - seniorTotal
  childTotal / childCount

/-- Theorem stating that the child ticket price is $5.63 given the problem conditions. -/
theorem child_ticket_price_is_correct :
  childTicketPrice 50 25 15 10 15 0.25 600 = 5.63 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_price_is_correct_l231_23195


namespace NUMINAMATH_CALUDE_expenditure_estimate_l231_23166

/-- Represents the annual income in billions of yuan -/
def annual_income : ℝ := 15

/-- Represents the relationship between income x and expenditure y -/
def expenditure_function (x : ℝ) : ℝ := 0.8 * x + 0.1

/-- The estimated annual expenditure based on the given income and relationship -/
def estimated_expenditure : ℝ := expenditure_function annual_income

theorem expenditure_estimate : estimated_expenditure = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_estimate_l231_23166


namespace NUMINAMATH_CALUDE_bees_after_five_days_l231_23135

/-- The number of bees in the hive after n days -/
def bees_after_days (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * bees_after_days (n - 1)

/-- The theorem stating that after 5 days, there will be 1024 bees in the hive -/
theorem bees_after_five_days : bees_after_days 5 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bees_after_five_days_l231_23135


namespace NUMINAMATH_CALUDE_unique_y_exists_l231_23164

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

-- Theorem statement
theorem unique_y_exists : ∃! y : ℝ, star 4 y = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_exists_l231_23164


namespace NUMINAMATH_CALUDE_exists_multiple_irreducible_representations_l231_23112

/-- The set V_n for a given n > 2 -/
def V_n (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is irreducible in V_n if it cannot be expressed as a product of two numbers in V_n -/
def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- The main theorem -/
theorem exists_multiple_irreducible_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (irreducibles1 irreducibles2 : List ℕ),
      irreducibles1 ≠ irreducibles2 ∧
      (∀ x ∈ irreducibles1, irreducible_in_V_n n x) ∧
      (∀ x ∈ irreducibles2, irreducible_in_V_n n x) ∧
      (irreducibles1.prod = r) ∧
      (irreducibles2.prod = r) :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_irreducible_representations_l231_23112


namespace NUMINAMATH_CALUDE_faye_country_albums_l231_23137

/-- The number of country albums Faye bought -/
def country_albums : ℕ := sorry

/-- The number of pop albums Faye bought -/
def pop_albums : ℕ := 3

/-- The number of songs per album -/
def songs_per_album : ℕ := 6

/-- The total number of songs Faye bought -/
def total_songs : ℕ := 30

theorem faye_country_albums : 
  country_albums = 2 := by sorry

end NUMINAMATH_CALUDE_faye_country_albums_l231_23137


namespace NUMINAMATH_CALUDE_line_parameterization_l231_23158

/-- Given a line y = 2x + 5 parameterized as (x, y) = (r, -3) + t(5, k),
    prove that r = -4 and k = 10 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t x y : ℝ, x = r + 5*t ∧ y = -3 + k*t → y = 2*x + 5) →
  r = -4 ∧ k = 10 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l231_23158


namespace NUMINAMATH_CALUDE_ruby_count_l231_23117

theorem ruby_count (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 :=
by sorry

end NUMINAMATH_CALUDE_ruby_count_l231_23117


namespace NUMINAMATH_CALUDE_bottle_caps_found_l231_23113

theorem bottle_caps_found (earlier_total current_total : ℕ) 
  (h1 : earlier_total = 25) 
  (h2 : current_total = 32) : 
  current_total - earlier_total = 7 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_found_l231_23113


namespace NUMINAMATH_CALUDE_equation_solutions_l231_23103

noncomputable def fourthRoot (x : ℝ) : ℝ := Real.rpow x (1/4)

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => fourthRoot (53 - 3*x) + fourthRoot (29 + x)
  ∀ x : ℝ, f x = 4 ↔ x = 2 ∨ x = 16 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l231_23103


namespace NUMINAMATH_CALUDE_asteroid_fragments_proof_l231_23173

theorem asteroid_fragments_proof :
  ∃ (X n : ℕ), 
    X > 0 ∧ 
    n > 0 ∧ 
    X / 5 + 26 + n * (X / 7) = X ∧ 
    X = 70 := by
  sorry

end NUMINAMATH_CALUDE_asteroid_fragments_proof_l231_23173


namespace NUMINAMATH_CALUDE_sticker_distribution_l231_23162

theorem sticker_distribution (d : ℕ) (h : d > 0) :
  let total_stickers : ℕ := 72
  let friends : ℕ := d
  let stickers_per_friend : ℚ := total_stickers / friends
  stickers_per_friend = 72 / d :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l231_23162


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_two_l231_23100

theorem min_value_ab (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x⁻¹ + y⁻¹ : ℝ) = Real.sqrt (x * y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  a * b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_two_l231_23100


namespace NUMINAMATH_CALUDE_population_increase_l231_23116

/-- The birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- The death rate in people per two seconds -/
def death_rate : ℚ := 1

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The net population increase in one day -/
def net_increase : ℕ := 259200

theorem population_increase :
  (birth_rate - death_rate) / 2 * seconds_per_day = net_increase := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l231_23116


namespace NUMINAMATH_CALUDE_min_value_problem_l231_23148

-- Define the function f
def f (x a b c : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f x a b c ≥ 4) 
  (hex : ∃ x, f x a b c = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l231_23148


namespace NUMINAMATH_CALUDE_friends_weekly_biking_distance_l231_23165

/-- The total distance biked by two friends in a week -/
def total_weekly_distance (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem: The total distance biked by Onur and Hanil in a week is 2700 kilometers -/
theorem friends_weekly_biking_distance :
  total_weekly_distance 250 40 5 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_friends_weekly_biking_distance_l231_23165


namespace NUMINAMATH_CALUDE_combined_average_marks_l231_23127

theorem combined_average_marks (class1_students class2_students class3_students : ℕ)
  (class1_avg class2_avg class3_avg : ℚ)
  (h1 : class1_students = 35)
  (h2 : class2_students = 45)
  (h3 : class3_students = 25)
  (h4 : class1_avg = 40)
  (h5 : class2_avg = 60)
  (h6 : class3_avg = 75) :
  (class1_students * class1_avg + class2_students * class2_avg + class3_students * class3_avg) /
  (class1_students + class2_students + class3_students) = 5975 / 105 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_l231_23127


namespace NUMINAMATH_CALUDE_T_properties_l231_23187

theorem T_properties (n : ℕ) : 
  let T := (10 * (10^n - 1)) / 81 - n / 9
  ∃ (k : ℕ), T = k ∧ T % 11 = ((n + 1) / 2) % 11 := by
  sorry

end NUMINAMATH_CALUDE_T_properties_l231_23187


namespace NUMINAMATH_CALUDE_fred_limes_picked_l231_23194

theorem fred_limes_picked (total_limes : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) 
  (h1 : total_limes = 103)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35) :
  total_limes - (alyssa_limes + nancy_limes) = 36 := by
sorry

end NUMINAMATH_CALUDE_fred_limes_picked_l231_23194


namespace NUMINAMATH_CALUDE_total_honey_production_total_honey_is_1060_l231_23106

/-- Calculates the total honey production for two bee hives with given characteristics -/
theorem total_honey_production
  (hive1_bees : ℕ)
  (hive1_honey : ℝ)
  (hive2_bee_reduction : ℝ)
  (hive2_honey_increase : ℝ)
  (h1 : hive1_bees = 1000)
  (h2 : hive1_honey = 500)
  (h3 : hive2_bee_reduction = 0.2)
  (h4 : hive2_honey_increase = 0.4)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check total_honey_production

/-- The total honey production is 1060 liters -/
theorem total_honey_is_1060 :
  total_honey_production 1000 500 0.2 0.4 rfl rfl rfl rfl = 1060 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_total_honey_production_total_honey_is_1060_l231_23106


namespace NUMINAMATH_CALUDE_two_students_same_school_probability_l231_23180

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The total number of possible outcomes --/
def total_outcomes : ℕ := num_schools ^ num_students

/-- The number of outcomes where exactly two students choose the same school --/
def favorable_outcomes : ℕ := num_students.choose 2 * num_schools * (num_schools - 1)

/-- The probability of exactly two students choosing the same school --/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem two_students_same_school_probability :
  probability = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_two_students_same_school_probability_l231_23180


namespace NUMINAMATH_CALUDE_cubic_sum_geq_product_sum_l231_23155

theorem cubic_sum_geq_product_sum {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  a^3 + b^3 + c^3 ≥ a*b + b*c + c*a ∧ 
  (a^3 + b^3 + c^3 = a*b + b*c + c*a ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_geq_product_sum_l231_23155


namespace NUMINAMATH_CALUDE_triangle_is_acute_l231_23144

-- Define the triangle and its angles
def Triangle (a1 a2 a3 : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a1 + a2 + a3 = 180

-- Define an acute triangle
def AcuteTriangle (a1 a2 a3 : ℝ) : Prop :=
  Triangle a1 a2 a3 ∧ a1 < 90 ∧ a2 < 90 ∧ a3 < 90

-- Theorem statement
theorem triangle_is_acute (a2 : ℝ) :
  Triangle (2 * a2) a2 (1.5 * a2) → AcuteTriangle (2 * a2) a2 (1.5 * a2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_is_acute_l231_23144


namespace NUMINAMATH_CALUDE_tank_full_time_l231_23147

/-- Represents a water tank with pipes for filling and draining. -/
structure WaterTank where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the time required to fill the tank given the pipe rates and capacity. -/
def time_to_fill (tank : WaterTank) : ℕ :=
  let net_fill_per_cycle := tank.pipeA_rate + tank.pipeB_rate - tank.pipeC_rate
  let cycles := tank.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the given tank will be full after 48 minutes. -/
theorem tank_full_time (tank : WaterTank) 
  (h1 : tank.capacity = 800)
  (h2 : tank.pipeA_rate = 40)
  (h3 : tank.pipeB_rate = 30)
  (h4 : tank.pipeC_rate = 20) :
  time_to_fill tank = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l231_23147


namespace NUMINAMATH_CALUDE_paths_A_to_C_via_B_l231_23191

/-- The number of paths from A to B -/
def paths_A_to_B : ℕ := Nat.choose 6 2

/-- The number of paths from B to C -/
def paths_B_to_C : ℕ := Nat.choose 6 3

/-- The total number of steps from A to C -/
def total_steps : ℕ := 12

/-- The number of steps from A to B -/
def steps_A_to_B : ℕ := 6

/-- The number of steps from B to C -/
def steps_B_to_C : ℕ := 6

theorem paths_A_to_C_via_B : 
  paths_A_to_B * paths_B_to_C = 300 ∧ 
  steps_A_to_B + steps_B_to_C = total_steps :=
by sorry

end NUMINAMATH_CALUDE_paths_A_to_C_via_B_l231_23191


namespace NUMINAMATH_CALUDE_arithmetic_mean_function_is_constant_l231_23107

/-- A function from ℤ × ℤ to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℤ) : Prop :=
  (∀ i j : ℤ, 0 < f (i, j)) ∧ 
  (∀ i j : ℤ, 4 * f (i, j) = f (i-1, j) + f (i+1, j) + f (i, j-1) + f (i, j+1))

/-- Theorem stating that any function satisfying the arithmetic mean property is constant -/
theorem arithmetic_mean_function_is_constant (f : ℤ × ℤ → ℤ) 
  (h : ArithmeticMeanFunction f) : 
  ∃ c : ℤ, ∀ i j : ℤ, f (i, j) = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_function_is_constant_l231_23107


namespace NUMINAMATH_CALUDE_lucy_grocery_problem_l231_23120

/-- Lucy's grocery shopping problem -/
theorem lucy_grocery_problem (total_packs cookies_packs noodles_packs : ℕ) :
  total_packs = 28 →
  cookies_packs = 12 →
  total_packs = cookies_packs + noodles_packs →
  noodles_packs = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_problem_l231_23120


namespace NUMINAMATH_CALUDE_function_properties_l231_23170

-- Define the function f(x) = ax^3 + bx^2
def f (x : ℝ) : ℝ := -6 * x^3 + 9 * x^2

-- State the theorem
theorem function_properties :
  (f 1 = 3) ∧ 
  (deriv f 1 = 0) ∧ 
  (∀ x : ℝ, f x ≥ 0) ∧
  (∃ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l231_23170


namespace NUMINAMATH_CALUDE_quartic_at_one_equals_three_l231_23161

/-- Horner's method for evaluating a quartic polynomial at x = 1 -/
def horner_quartic (a₄ a₃ a₂ a₁ a₀ : ℤ) : ℤ :=
  ((((1 * a₄ + a₃) * 1 + a₂) * 1 + a₁) * 1 + a₀)

/-- The given quartic polynomial evaluated at x = 1 equals 3 -/
theorem quartic_at_one_equals_three :
  horner_quartic 1 (-7) (-9) 11 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quartic_at_one_equals_three_l231_23161


namespace NUMINAMATH_CALUDE_quadratic_factorization_l231_23143

theorem quadratic_factorization (x : ℝ) : x^2 - 3*x - 4 = (x + 1)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l231_23143


namespace NUMINAMATH_CALUDE_infinitely_many_cube_sums_l231_23150

theorem infinitely_many_cube_sums (n : ℕ) : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k : ℕ), ∃ (m : ℕ+), (n^6 + 3 * (f k)) = m^3 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_cube_sums_l231_23150


namespace NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l231_23130

/-- An ellipse with center at the origin and foci on the coordinate axes -/
structure CenteredEllipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The equation of the ellipse -/
def CenteredEllipse.equation (e : CenteredEllipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The ellipse passes through the given points -/
def CenteredEllipse.passes_through (e : CenteredEllipse) : Prop :=
  e.equation 2 (Real.sqrt 2) ∧ e.equation (Real.sqrt 6) 1

/-- The main theorem -/
theorem ellipse_and_circle_theorem (e : CenteredEllipse) 
    (h_passes : e.passes_through) : 
    (e.a^2 = 8 ∧ e.b^2 = 4) ∧
    ∃ (r : ℝ), r^2 = 8/3 ∧
      ∀ (l : ℝ → ℝ → Prop), 
        (∃ (k m : ℝ), ∀ x y, l x y ↔ y = k * x + m) →
        (∃ x y, x^2 + y^2 = r^2 ∧ l x y) →
        ∃ (A B : ℝ × ℝ), 
          A ≠ B ∧
          e.equation A.1 A.2 ∧ 
          e.equation B.1 B.2 ∧
          l A.1 A.2 ∧ 
          l B.1 B.2 ∧
          A.1 * B.1 + A.2 * B.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l231_23130


namespace NUMINAMATH_CALUDE_cookies_leftover_l231_23176

theorem cookies_leftover (naomi oliver penelope : ℕ) 
  (h_naomi : naomi = 53)
  (h_oliver : oliver = 67)
  (h_penelope : penelope = 29)
  (package_size : ℕ) 
  (h_package : package_size = 15) : 
  (naomi + oliver + penelope) % package_size = 14 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_l231_23176


namespace NUMINAMATH_CALUDE_total_flour_in_bowl_l231_23181

-- Define the initial amount of flour in the bowl
def initial_flour : ℚ := 2 + 3/4

-- Define the amount of flour added
def added_flour : ℚ := 45/100

-- Theorem to prove
theorem total_flour_in_bowl :
  initial_flour + added_flour = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_total_flour_in_bowl_l231_23181


namespace NUMINAMATH_CALUDE_andrey_stamps_l231_23109

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andrey_stamps_l231_23109


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l231_23111

theorem consecutive_integers_square_sum (e f g h : ℤ) : 
  (e + 1 = f) → (f + 1 = g) → (g + 1 = h) →
  (e < f) → (f < g) → (g < h) →
  (e^2 + h^2 = 3405) →
  (f^2 * g^2 = 2689600) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l231_23111


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l231_23167

def num_meals : ℕ := 4
def num_fruit_types : ℕ := 3

def prob_same_fruit_all_meals : ℚ := (1 / num_fruit_types) ^ num_meals

theorem joe_fruit_probability :
  1 - (num_fruit_types * prob_same_fruit_all_meals) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l231_23167


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l231_23146

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l231_23146


namespace NUMINAMATH_CALUDE_smallest_a_value_l231_23179

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (45 * x)) :
  a ≥ 45 ∧ ∃ a₀ : ℝ, a₀ ≥ 0 ∧ (∀ x : ℤ, Real.sin (a₀ * x + b) = Real.sin (45 * x)) ∧ a₀ = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l231_23179


namespace NUMINAMATH_CALUDE_remainder_theorem_l231_23159

theorem remainder_theorem (x y z a b c d e : ℕ) : 
  0 < a ∧ a < 211 ∧ 
  0 < b ∧ b < 211 ∧ 
  0 < c ∧ c < 211 ∧ 
  0 < d ∧ d < 251 ∧ 
  0 < e ∧ e < 251 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  x % 211 = a ∧ 
  y % 211 = b ∧ 
  z % 211 = c ∧
  x % 251 = c ∧
  y % 251 = d ∧
  z % 251 = e →
  ∃! R, 0 ≤ R ∧ R < 211 * 251 ∧ (2 * x - y + 3 * z + 47) % (211 * 251) = R :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l231_23159
