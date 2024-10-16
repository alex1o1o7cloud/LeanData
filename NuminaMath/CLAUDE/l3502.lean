import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3502_350206

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧
  (∃ a b, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3502_350206


namespace NUMINAMATH_CALUDE_larger_city_size_proof_l3502_350203

/-- The number of cubic yards in the larger city -/
def larger_city_size : ℕ := 9000

/-- The population density in people per cubic yard -/
def population_density : ℕ := 80

/-- The size of the smaller city in cubic yards -/
def smaller_city_size : ℕ := 6400

/-- The population difference between the larger and smaller city -/
def population_difference : ℕ := 208000

theorem larger_city_size_proof :
  population_density * larger_city_size = 
  population_density * smaller_city_size + population_difference := by
  sorry

end NUMINAMATH_CALUDE_larger_city_size_proof_l3502_350203


namespace NUMINAMATH_CALUDE_square_area_side_3_l3502_350238

/-- The area of a square with side length 3 is 9 square units. -/
theorem square_area_side_3 : 
  ∀ (s : ℝ), s = 3 → s * s = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_side_3_l3502_350238


namespace NUMINAMATH_CALUDE_definite_integral_tan_trig_l3502_350237

theorem definite_integral_tan_trig : 
  ∃ (f : ℝ → ℝ), (∀ x ∈ Set.Icc (π / 4) (Real.arcsin (Real.sqrt (2 / 3))), 
    HasDerivAt f ((8 * Real.tan x) / (3 * (Real.cos x)^2 + 8 * Real.sin (2 * x) - 7)) x) ∧ 
  (f (Real.arcsin (Real.sqrt (2 / 3))) - f (π / 4) = 
    (4 / 21) * Real.log (abs ((7 * Real.sqrt 2 - 2) / 5)) - 
    (4 / 3) * Real.log (abs (2 - Real.sqrt 2))) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tan_trig_l3502_350237


namespace NUMINAMATH_CALUDE_y_influenced_by_other_factors_other_factors_lead_to_random_errors_l3502_350269

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ      -- Random error

/-- Definition of the linear regression model equation -/
def model_equation (m : LinearRegressionModel) : ℝ → ℝ :=
  fun x => m.b * x + m.a + m.e

/-- Theorem stating that y is influenced by factors other than x -/
theorem y_influenced_by_other_factors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.y m.x ≠ m.b * m.x + m.a :=
sorry

/-- Theorem stating that other factors can lead to random errors -/
theorem other_factors_lead_to_random_errors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.e ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_y_influenced_by_other_factors_other_factors_lead_to_random_errors_l3502_350269


namespace NUMINAMATH_CALUDE_tan_585_degrees_l3502_350240

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l3502_350240


namespace NUMINAMATH_CALUDE_acid_mixture_theorem_l3502_350227

/-- Represents an acid solution with a given volume and concentration. -/
structure AcidSolution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure acid in a solution. -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.volume * solution.concentration

/-- Theorem: Mixing 4 liters of 60% acid solution with 16 liters of 75% acid solution
    results in a 72% acid solution with a total volume of 20 liters. -/
theorem acid_mixture_theorem : 
  let solution1 : AcidSolution := ⟨4, 0.6⟩
  let solution2 : AcidSolution := ⟨16, 0.75⟩
  let totalVolume := solution1.volume + solution2.volume
  let totalPureAcid := pureAcid solution1 + pureAcid solution2
  totalVolume = 20 ∧ 
  totalPureAcid / totalVolume = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_theorem_l3502_350227


namespace NUMINAMATH_CALUDE_inequality_theorem_stronger_inequality_best_constant_l3502_350264

theorem inequality_theorem (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| ≥ 2 := by
  sorry

theorem stronger_inequality (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (pa : a ≥ 0) (pb : b ≥ 0) (pc : c ≥ 0) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| > 3 := by
  sorry

theorem best_constant (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| < 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_stronger_inequality_best_constant_l3502_350264


namespace NUMINAMATH_CALUDE_distance_sasha_kolya_l3502_350245

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  length : ℝ

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.sasha.speed > 0 ∧
  r.lyosha.speed > 0 ∧
  r.kolya.speed > 0 ∧
  r.sasha.position = r.length ∧
  r.lyosha.position = r.length - 10 ∧
  r.kolya.position = r.lyosha.position * (r.kolya.speed / r.lyosha.speed)

/-- The theorem to be proved -/
theorem distance_sasha_kolya (r : Race) (h : race_conditions r) :
  r.sasha.position - r.kolya.position = 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_sasha_kolya_l3502_350245


namespace NUMINAMATH_CALUDE_sequence_limit_existence_l3502_350200

theorem sequence_limit_existence (a : ℕ → ℝ) (h : ∀ n, 0 ≤ a n ∧ a n ≤ 1) :
  ∃ (n : ℕ → ℕ) (A : ℝ),
    (∀ i j, i < j → n i < n j) ∧
    (∀ ε > 0, ∃ N, ∀ i j, i ≠ j → i > N → j > N → |a (n i + n j) - A| < ε) := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_existence_l3502_350200


namespace NUMINAMATH_CALUDE_min_room_dimensions_l3502_350256

/-- The minimum dimensions of a rectangular room that can accommodate a 9' × 12' table --/
theorem min_room_dimensions (table_width : ℝ) (table_length : ℝ) 
  (hw : table_width = 9) (hl : table_length = 12) :
  ∃ (S T : ℝ), 
    S > T ∧ 
    S ≥ Real.sqrt (table_width^2 + table_length^2) ∧
    T ≥ max table_width table_length ∧
    ∀ (S' T' : ℝ), (S' > T' ∧ 
                    S' ≥ Real.sqrt (table_width^2 + table_length^2) ∧ 
                    T' ≥ max table_width table_length) → 
                    (S ≤ S' ∧ T ≤ T') ∧
    S = 15 ∧ T = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_room_dimensions_l3502_350256


namespace NUMINAMATH_CALUDE_library_books_percentage_l3502_350209

theorem library_books_percentage (total_books adult_books : ℕ) 
  (h1 : total_books = 160)
  (h2 : adult_books = 104) :
  (total_books - adult_books : ℚ) / total_books * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_library_books_percentage_l3502_350209


namespace NUMINAMATH_CALUDE_problem_solution_l3502_350259

noncomputable section

def f (a : ℝ) (x : ℝ) := a * (3 : ℝ)^x + (3 : ℝ)^(-x)

def g (m : ℝ) (x : ℝ) := (Real.log x / Real.log 2)^2 + 2 * (Real.log x / Real.log 2) + m

theorem problem_solution :
  (∀ x, f a x = f a (-x)) →
  a = 1 ∧
  (∀ x y, 0 < x → x < y → f 1 x < f 1 y) ∧
  (∃ α β, α ≠ β ∧ 1/8 ≤ α ∧ α ≤ 4 ∧ 1/8 ≤ β ∧ β ≤ 4 ∧ g m α = 0 ∧ g m β = 0) →
  -3 ≤ m ∧ m < 1 ∧ α * β = 1/4 :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3502_350259


namespace NUMINAMATH_CALUDE_distance_between_points_l3502_350284

theorem distance_between_points (d : ℝ) : 
  (∃ (x : ℝ), d / 2 + x = d - 5) ∧ 
  (d / 2 + d / 2 - 45 / 8 = 45 / 8) → 
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3502_350284


namespace NUMINAMATH_CALUDE_egyptian_triangle_bisecting_line_exists_l3502_350217

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : Real
  b : Real
  c : Real
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a line that intersects the triangle -/
structure BisectingLine where
  x : Real -- intersection point on shorter leg
  y : Real -- intersection point on hypotenuse
  hx : x = 3 - Real.sqrt 6 / 2
  hy : y = 3 + Real.sqrt 6 / 2

/-- Theorem stating the existence of a bisecting line for an Egyptian triangle -/
theorem egyptian_triangle_bisecting_line_exists (t : EgyptianTriangle) :
  ∃ (l : BisectingLine),
    (l.x + l.y = t.a + t.b) ∧                          -- Bisects perimeter
    (l.x * l.y * (t.b / t.c) / 2 = t.a * t.b / 4) :=   -- Bisects area
by sorry

end NUMINAMATH_CALUDE_egyptian_triangle_bisecting_line_exists_l3502_350217


namespace NUMINAMATH_CALUDE_desks_per_row_l3502_350272

theorem desks_per_row (total_students : ℕ) (restroom_students : ℕ) (rows : ℕ) :
  total_students = 23 →
  restroom_students = 2 →
  rows = 4 →
  let absent_students := 3 * restroom_students - 1
  let present_students := total_students - restroom_students - absent_students
  let total_desks := (3 * present_students) / 2
  total_desks / rows = 6 :=
by sorry

end NUMINAMATH_CALUDE_desks_per_row_l3502_350272


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3502_350218

theorem absolute_value_equation (x : ℝ) : |x + 3| = |x - 5| → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3502_350218


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l3502_350268

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  D = E →  -- Isosceles triangle condition
  F = 2 * D - 40 →  -- Relationship between F and D
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = 70 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l3502_350268


namespace NUMINAMATH_CALUDE_group_size_problem_l3502_350207

theorem group_size_problem (T : ℕ) (L : ℕ) : 
  T > 90 →  -- Total number of people is greater than 90
  L = T - 90 →  -- Number of people under 20 is the total minus 90
  (L : ℚ) / T = 2/5 →  -- Probability of selecting someone under 20 is 0.4
  T = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3502_350207


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_59_l3502_350246

theorem least_positive_integer_multiple_59 : 
  ∃ (x : ℕ+), (∀ (y : ℕ+), y < x → ¬(59 ∣ (2 * y + 51)^2)) ∧ (59 ∣ (2 * x + 51)^2) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_59_l3502_350246


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l3502_350214

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → -- four-digit positive integer
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) → -- divisible by 3, 5, and 7
  1050 ≤ n := by
  sorry

#check least_four_digit_multiple

end NUMINAMATH_CALUDE_least_four_digit_multiple_l3502_350214


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3502_350293

theorem complex_equation_solution (x y : ℝ) :
  (x - 2 * y) * Complex.I = (2 * x + 1 : ℂ) + 3 * Complex.I →
  x = -1/2 ∧ y = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3502_350293


namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l3502_350265

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_factorial_sum :
  sum_factorials 2003 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l3502_350265


namespace NUMINAMATH_CALUDE_max_value_of_squared_differences_l3502_350288

theorem max_value_of_squared_differences (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 = 27) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 9 → (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 27) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_squared_differences_l3502_350288


namespace NUMINAMATH_CALUDE_acute_angle_cosine_difference_l3502_350213

theorem acute_angle_cosine_difference (α : Real) : 
  0 < α → α < π / 2 →  -- acute angle condition
  3 * Real.sin α = Real.tan α →  -- given equation
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_cosine_difference_l3502_350213


namespace NUMINAMATH_CALUDE_right_triangle_area_l3502_350294

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = c^2) (h3 : c = 3) :
  (1/2) * a * b = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3502_350294


namespace NUMINAMATH_CALUDE_integer_solutions_inequality_l3502_350261

theorem integer_solutions_inequality (x : ℤ) : 
  -1 < 2*x + 1 ∧ 2*x + 1 < 5 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_inequality_l3502_350261


namespace NUMINAMATH_CALUDE_good_numbers_exist_exist_good_sum_not_good_l3502_350257

/-- A number is "good" if it can be expressed as a^2 + 161b^2 for some integers a and b -/
def is_good_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 161 * b^2

theorem good_numbers_exist : is_good_number 100 ∧ is_good_number 2010 := by sorry

theorem exist_good_sum_not_good :
  ∃ x y : ℕ+, is_good_number (x^161 + y^161) ∧ ¬is_good_number (x + y) := by sorry

end NUMINAMATH_CALUDE_good_numbers_exist_exist_good_sum_not_good_l3502_350257


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l3502_350248

theorem integral_sqrt_one_minus_x_squared : ∫ x in (-1)..(1), Real.sqrt (1 - x^2) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l3502_350248


namespace NUMINAMATH_CALUDE_existence_of_integers_l3502_350204

theorem existence_of_integers (a b : ℝ) (h : a ≠ b) : 
  ∃ (m n : ℤ), a * (m : ℝ) + b * (n : ℝ) < 0 ∧ b * (m : ℝ) + a * (n : ℝ) > 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_integers_l3502_350204


namespace NUMINAMATH_CALUDE_number_problem_l3502_350220

theorem number_problem : ∃ x : ℝ, x^2 + 75 = (x - 20)^2 ∧ x = 8.125 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3502_350220


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l3502_350287

theorem subset_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l3502_350287


namespace NUMINAMATH_CALUDE_circle_center_point_satisfies_center_circle_center_is_4_2_l3502_350202

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y = 16 → (x - 4)^2 + (y - 2)^2 = 36 := by
  sorry

/-- The point (4, 2) satisfies the center condition of the circle -/
theorem point_satisfies_center : 
  (4 : ℝ)^2 - 8*4 + (2 : ℝ)^2 - 4*2 = 16 := by
  sorry

/-- The center of the circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center_is_4_2 : 
  ∃! (c : ℝ × ℝ), ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = 16 → (x - c.1)^2 + (y - c.2)^2 = 36 ∧ c = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_point_satisfies_center_circle_center_is_4_2_l3502_350202


namespace NUMINAMATH_CALUDE_n2o5_molecular_weight_l3502_350210

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def oxygen_count : ℕ := 5

/-- The molecular weight of N2O5 in g/mol -/
def n2o5_weight : ℝ := nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

theorem n2o5_molecular_weight : n2o5_weight = 108.02 := by
  sorry

end NUMINAMATH_CALUDE_n2o5_molecular_weight_l3502_350210


namespace NUMINAMATH_CALUDE_missing_month_sale_correct_grocer_sale_problem_l3502_350226

/-- Calculates the missing month's sale given sales data for 5 months and the average sale --/
def calculate_missing_month_sale (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing month's sale satisfies the average sale condition --/
theorem missing_month_sale_correct 
  (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) :
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

/-- Applies the theorem to the specific problem values --/
theorem grocer_sale_problem :
  let sale1 : ℕ := 5921
  let sale2 : ℕ := 5468
  let sale4 : ℕ := 6088
  let sale5 : ℕ := 6433
  let sale6 : ℕ := 5922
  let average_sale : ℕ := 5900
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  sale3 = 5568 ∧ (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

end NUMINAMATH_CALUDE_missing_month_sale_correct_grocer_sale_problem_l3502_350226


namespace NUMINAMATH_CALUDE_problem_statement_l3502_350270

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ 4) ∧
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ 2 * x₀ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3502_350270


namespace NUMINAMATH_CALUDE_teachers_survey_result_l3502_350263

def teachers_survey (total : ℕ) (high_bp : ℕ) (stress : ℕ) (both : ℕ) : Prop :=
  let neither := total - (high_bp + stress - both)
  (neither : ℚ) / total * 100 = 20

theorem teachers_survey_result : teachers_survey 150 90 60 30 := by
  sorry

end NUMINAMATH_CALUDE_teachers_survey_result_l3502_350263


namespace NUMINAMATH_CALUDE_line_touches_x_axis_twice_l3502_350212

-- Define the function representing the line
def f (x : ℝ) : ℝ := x^2 - x^3

-- Theorem statement
theorem line_touches_x_axis_twice :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_line_touches_x_axis_twice_l3502_350212


namespace NUMINAMATH_CALUDE_cookies_per_guest_l3502_350251

theorem cookies_per_guest (total_cookies : ℕ) (num_guests : ℕ) (h1 : total_cookies = 38) (h2 : num_guests = 2) :
  total_cookies / num_guests = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_guest_l3502_350251


namespace NUMINAMATH_CALUDE_janet_weekly_income_l3502_350292

/-- Calculates the total income for Janet's exterminator work and sculpture sales --/
def janet_total_income (small_res_rate : ℕ) (large_res_rate : ℕ) (commercial_rate : ℕ)
                       (small_res_hours : ℕ) (large_res_hours : ℕ) (commercial_hours : ℕ)
                       (small_sculpture_price : ℕ) (medium_sculpture_price : ℕ) (large_sculpture_price : ℕ)
                       (small_sculpture_weight : ℕ) (small_sculpture_count : ℕ)
                       (medium_sculpture_weight : ℕ) (medium_sculpture_count : ℕ)
                       (large_sculpture_weight : ℕ) (large_sculpture_count : ℕ) : ℕ :=
  let exterminator_income := small_res_rate * small_res_hours + 
                             large_res_rate * large_res_hours + 
                             commercial_rate * commercial_hours
  let sculpture_income := small_sculpture_price * small_sculpture_weight * small_sculpture_count +
                          medium_sculpture_price * medium_sculpture_weight * medium_sculpture_count +
                          large_sculpture_price * large_sculpture_weight * large_sculpture_count
  exterminator_income + sculpture_income

/-- Theorem stating that Janet's total income for the week is $2320 --/
theorem janet_weekly_income : 
  janet_total_income 70 85 100 10 5 5 20 25 30 4 2 7 1 12 1 = 2320 := by
  sorry

end NUMINAMATH_CALUDE_janet_weekly_income_l3502_350292


namespace NUMINAMATH_CALUDE_number_equals_sixteen_l3502_350291

theorem number_equals_sixteen : ∃ x : ℝ, 0.0025 * x = 0.04 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_sixteen_l3502_350291


namespace NUMINAMATH_CALUDE_smaller_part_area_l3502_350241

/-- The area of the smaller part of a field satisfying given conditions -/
theorem smaller_part_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 1800 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 6 →
  smaller_area = 750 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_area_l3502_350241


namespace NUMINAMATH_CALUDE_fraction_of_a_equal_half_b_l3502_350219

/-- Given two amounts a and b, where their sum is 1210 and b is 484,
    prove that the fraction of a's amount equal to half of b's amount is 1/3 -/
theorem fraction_of_a_equal_half_b (a b : ℕ) : 
  a + b = 1210 → b = 484 → ∃ f : ℚ, f * a = (1 / 2) * b ∧ f = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_a_equal_half_b_l3502_350219


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3502_350228

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3502_350228


namespace NUMINAMATH_CALUDE_min_value_x_l3502_350273

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 - (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l3502_350273


namespace NUMINAMATH_CALUDE_chloes_clothing_boxes_l3502_350223

theorem chloes_clothing_boxes (total_pieces : ℕ) (pieces_per_box : ℕ) (h1 : total_pieces = 32) (h2 : pieces_per_box = 8) :
  total_pieces / pieces_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloes_clothing_boxes_l3502_350223


namespace NUMINAMATH_CALUDE_log_stack_theorem_l3502_350231

/-- Represents a stack of logs -/
structure LogStack where
  bottom_row : ℕ
  top_row : ℕ
  row_difference : ℕ

/-- Calculates the number of rows in the log stack -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_row - stack.top_row) / stack.row_difference + 1

/-- Calculates the total number of logs in the stack -/
def total_logs (stack : LogStack) : ℕ :=
  (num_rows stack * (stack.bottom_row + stack.top_row)) / 2

/-- The main theorem about the log stack -/
theorem log_stack_theorem (stack : LogStack) 
  (h1 : stack.bottom_row = 15)
  (h2 : stack.top_row = 5)
  (h3 : stack.row_difference = 2) :
  total_logs stack = 60 ∧ stack.top_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_theorem_l3502_350231


namespace NUMINAMATH_CALUDE_incorrect_conjunction_falsehood_l3502_350297

theorem incorrect_conjunction_falsehood : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_conjunction_falsehood_l3502_350297


namespace NUMINAMATH_CALUDE_exercise_books_count_l3502_350290

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, prove that there are 30 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books = 30 := by
sorry

end NUMINAMATH_CALUDE_exercise_books_count_l3502_350290


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3502_350205

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ :=
  -l.y_intercept / l.slope

/-- The original line 4x + 5y = 10 -/
def original_line : Line :=
  { slope := -4/5, y_intercept := 2 }

/-- The perpendicular line we're interested in -/
def perpendicular_line : Line :=
  { slope := 5/4, y_intercept := -3 }

theorem x_intercept_of_perpendicular_line :
  perpendicular original_line perpendicular_line ∧
  perpendicular_line.y_intercept = -3 →
  x_intercept perpendicular_line = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3502_350205


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_l3502_350283

/-- A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle (n : ℕ) : 
  (360 : ℝ) / n = 30 → n = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_l3502_350283


namespace NUMINAMATH_CALUDE_decreasing_linear_function_not_in_first_quadrant_l3502_350225

/-- A linear function y = kx + b that decreases as x increases and has a negative y-intercept -/
structure DecreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neg : k < 0
  b_neg : b < 0

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

/-- The theorem stating that a decreasing linear function with negative y-intercept does not pass through the first quadrant -/
theorem decreasing_linear_function_not_in_first_quadrant (f : DecreasingLinearFunction) :
  ∀ x y, y = f.k * x + f.b → (x, y) ∉ FirstQuadrant := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_not_in_first_quadrant_l3502_350225


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_constant_equality_condition_l3502_350208

theorem inequality_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
by sorry

theorem largest_constant :
  ∀ C : ℝ, (∀ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ, (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂))) → C ≤ 3 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  x₁ + x₄ = x₂ + x₅ ∧ x₂ + x₅ = x₃ + x₆ :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_constant_equality_condition_l3502_350208


namespace NUMINAMATH_CALUDE_binomial_10_3_l3502_350262

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3502_350262


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3502_350239

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  2 = Nat.minFac (3^13 + 9^11) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3502_350239


namespace NUMINAMATH_CALUDE_complex_real_roots_relationship_l3502_350230

theorem complex_real_roots_relationship (a : ℝ) : 
  ¬(∀ x : ℂ, x^2 + a*x - a = 0 → (∀ y : ℝ, y^2 - a*y + a ≠ 0)) ∧
  ¬(∀ y : ℝ, y^2 - a*y + a = 0 → (∀ x : ℂ, x^2 + a*x - a ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_complex_real_roots_relationship_l3502_350230


namespace NUMINAMATH_CALUDE_magazines_to_boxes_l3502_350267

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_magazines_to_boxes_l3502_350267


namespace NUMINAMATH_CALUDE_inequality_proof_l3502_350275

theorem inequality_proof (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3*(1 + a^2 + a^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3502_350275


namespace NUMINAMATH_CALUDE_eighth_number_with_digit_sum_13_l3502_350243

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns whether a natural number has a digit sum of 13 -/
def has_digit_sum_13 (n : ℕ) : Prop := digit_sum n = 13

/-- A function that returns the nth positive integer with digit sum 13 -/
def nth_digit_sum_13 (n : ℕ) : ℕ := sorry

theorem eighth_number_with_digit_sum_13 : nth_digit_sum_13 8 = 148 := by sorry

end NUMINAMATH_CALUDE_eighth_number_with_digit_sum_13_l3502_350243


namespace NUMINAMATH_CALUDE_average_price_reduction_l3502_350281

theorem average_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x ≥ 0 ∧ x ≤ 1) : 
  x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_average_price_reduction_l3502_350281


namespace NUMINAMATH_CALUDE_triangle_theorem_l3502_350236

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * sin t.C = t.c * cos t.B + t.c) :
  t.B = π / 3 ∧ 
  (t.b ^ 2 = t.a * t.c → 1 / tan t.A + 1 / tan t.C = 2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3502_350236


namespace NUMINAMATH_CALUDE_gcd_count_l3502_350277

def count_gcd_values (a b : ℕ) : Prop :=
  (Nat.gcd a b * Nat.lcm a b = 360) →
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ (c d : ℕ), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) ∧
                     (∀ y, (∃ (e f : ℕ), Nat.gcd e f = y ∧ Nat.gcd e f * Nat.lcm e f = 360) → y ∈ S) ∧
                     S.card = 14)

theorem gcd_count : ∀ a b : ℕ, count_gcd_values a b :=
sorry

end NUMINAMATH_CALUDE_gcd_count_l3502_350277


namespace NUMINAMATH_CALUDE_logo_shaded_area_l3502_350280

/-- Represents a logo with specific geometric properties. -/
structure Logo where
  /-- Length of vertical straight edges and diameters of small semicircles -/
  edge_length : ℝ
  /-- Rotational symmetry property -/
  has_rotational_symmetry : Prop

/-- Calculates the shaded area of the logo -/
def shaded_area (logo : Logo) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of a logo with specific properties is 4 + π -/
theorem logo_shaded_area (logo : Logo) 
  (h1 : logo.edge_length = 2)
  (h2 : logo.has_rotational_symmetry) : 
  shaded_area logo = 4 + π := by
  sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l3502_350280


namespace NUMINAMATH_CALUDE_sin_EAF_value_l3502_350215

/-- A rectangle ABCD with E and F trisecting CD -/
structure RectangleWithTrisection where
  /-- Point A of the rectangle -/
  A : ℝ × ℝ
  /-- Point B of the rectangle -/
  B : ℝ × ℝ
  /-- Point C of the rectangle -/
  C : ℝ × ℝ
  /-- Point D of the rectangle -/
  D : ℝ × ℝ
  /-- Point E trisecting CD -/
  E : ℝ × ℝ
  /-- Point F trisecting CD -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (C.2 = D.2)
  /-- AB = 8 -/
  AB_length : (B.1 - A.1) = 8
  /-- BC = 6 -/
  BC_length : (B.2 - C.2) = 6
  /-- E and F trisect CD -/
  trisection : (E.1 - C.1) = (2/3) * (D.1 - C.1) ∧ (F.1 - C.1) = (1/3) * (D.1 - C.1)

/-- The sine of angle EAF in the given rectangle with trisection -/
def sin_EAF (r : RectangleWithTrisection) : ℝ :=
  sorry

/-- Theorem stating that sin ∠EAF = 12√13 / 194 -/
theorem sin_EAF_value (r : RectangleWithTrisection) : 
  sin_EAF r = 12 * Real.sqrt 13 / 194 :=
sorry

end NUMINAMATH_CALUDE_sin_EAF_value_l3502_350215


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3502_350271

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d q : ℝ) :
  a 1 = 2 →
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 = a 1 * q →
  a 11 = a 1 * q^2 →
  q = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3502_350271


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_three_digit_numbers_l3502_350285

-- Define a type for 3-digit numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

-- Define a function to check if a number uses given digits
def usesGivenDigits (n : ℕ) (digits : List ℕ) : Prop := sorry

-- Define a function to check if two numbers use all given digits exactly once
def useAllDigitsOnce (a b : ℕ) (digits : List ℕ) : Prop := sorry

-- Theorem statement
theorem smallest_sum_of_two_three_digit_numbers :
  ∃ (a b : ThreeDigitNumber),
    useAllDigitsOnce a.val b.val [1, 2, 3, 7, 8, 9] ∧
    (∀ (x y : ThreeDigitNumber),
      useAllDigitsOnce x.val y.val [1, 2, 3, 7, 8, 9] →
      a.val + b.val ≤ x.val + y.val) ∧
    a.val + b.val = 912 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_three_digit_numbers_l3502_350285


namespace NUMINAMATH_CALUDE_gcd_105_490_l3502_350211

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_490_l3502_350211


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3502_350255

theorem partial_fraction_decomposition :
  ∃ (P Q : ℚ), P = 22/9 ∧ Q = -4/9 ∧
  ∀ (x : ℚ), x ≠ 7 ∧ x ≠ -2 →
    (2*x + 8) / (x^2 - 5*x - 14) = P / (x - 7) + Q / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3502_350255


namespace NUMINAMATH_CALUDE_work_time_relation_l3502_350278

/-- Given three workers A, B, and C, where:
    - A takes m times as long to do a piece of work as B and C together
    - B takes n times as long as C and A together
    - C takes x times as long as A and B together
    This theorem proves that x = (m + n + 2) / (mn - 1) -/
theorem work_time_relation (m n x : ℝ) (hm : m > 0) (hn : n > 0) (hx : x > 0)
  (hA : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a = m * (1/(b+c)))
  (hB : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/b = n * (1/(a+c)))
  (hC : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/c = x * (1/(a+b))) :
  x = (m + n + 2) / (m * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_work_time_relation_l3502_350278


namespace NUMINAMATH_CALUDE_cistern_length_is_8_l3502_350252

/-- Represents a cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern is 8 meters --/
theorem cistern_length_is_8 (c : Cistern) 
    (h_width : c.width = 4)
    (h_depth : c.depth = 1.25)
    (h_area : c.wetSurfaceArea = 62) :
    c.length = 8 := by
  sorry

#check cistern_length_is_8

end NUMINAMATH_CALUDE_cistern_length_is_8_l3502_350252


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l3502_350233

/-- The system of equations -/
def system_equations (x y m : ℝ) : Prop :=
  2 * x - y = m ∧ 3 * x + 2 * y = m + 7

/-- Point in second quadrant with given distances -/
def point_conditions (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0 ∧ y = 3 ∧ x = -2

theorem solution_part1 :
  ∃ x y : ℝ, system_equations x y 0 ∧ x = 1 ∧ y = 2 := by sorry

theorem solution_part2 :
  ∀ x y : ℝ, system_equations x y (-7) ∧ point_conditions x y := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l3502_350233


namespace NUMINAMATH_CALUDE_metallic_sheet_length_proof_l3502_350276

/-- The length of a rectangular metallic sheet that forms a box of volume 24000 m³ when 10 m squares are cut from each corner. -/
def metallic_sheet_length : ℝ := 820

/-- The width of the rectangular metallic sheet. -/
def sheet_width : ℝ := 50

/-- The side length of the square cut from each corner. -/
def corner_cut : ℝ := 10

/-- The volume of the resulting box. -/
def box_volume : ℝ := 24000

theorem metallic_sheet_length_proof :
  (metallic_sheet_length - 2 * corner_cut) * (sheet_width - 2 * corner_cut) * corner_cut = box_volume :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_proof_l3502_350276


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3502_350296

theorem arithmetic_computation : 1 + (6 * 2 - 3 + 5) * 4 / 2 = 29 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3502_350296


namespace NUMINAMATH_CALUDE_alice_bob_race_difference_l3502_350242

/-- The difference in finish times between two runners in a race. -/
def finish_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the difference in finish times for Alice and Bob in a 12-mile race. -/
theorem alice_bob_race_difference :
  finish_time_difference 5 7 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_race_difference_l3502_350242


namespace NUMINAMATH_CALUDE_cost_of_apples_l3502_350298

/-- The cost of apples given the total cost of groceries and the costs of other items -/
theorem cost_of_apples (total cost_bananas cost_bread cost_milk : ℕ) 
  (h1 : total = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7) :
  total - (cost_bananas + cost_bread + cost_milk) = 14 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_apples_l3502_350298


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3502_350235

/-- 
Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
h is equal to -3/2
-/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3502_350235


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l3502_350295

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Represents the total distance run in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def marathon : MarathonDistance := { miles := 30, yards := 520 }
def yards_per_mile : ℕ := 1760
def num_marathons : ℕ := 8

theorem marathon_remainder_yards : 
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧ 
    TotalDistance.yards (
      { miles := m,
        yards := y } : TotalDistance
    ) = 640 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l3502_350295


namespace NUMINAMATH_CALUDE_van_rental_equation_l3502_350289

theorem van_rental_equation (x : ℝ) (h1 : x > 2) : 
  (180 / (x - 2)) - (180 / x) = 3 :=
by sorry

end NUMINAMATH_CALUDE_van_rental_equation_l3502_350289


namespace NUMINAMATH_CALUDE_division_problem_l3502_350254

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 199 →
  divisor = 18 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3502_350254


namespace NUMINAMATH_CALUDE_alligator_count_theorem_l3502_350299

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friends_count : ℕ) (friends_average : ℕ) : ℕ :=
  samara_count + friends_count * friends_average

/-- Theorem stating the total number of alligators seen by Samara and her friends -/
theorem alligator_count_theorem : 
  total_alligators 35 6 15 = 125 := by
  sorry

#eval total_alligators 35 6 15

end NUMINAMATH_CALUDE_alligator_count_theorem_l3502_350299


namespace NUMINAMATH_CALUDE_fraction_inequivalence_l3502_350279

theorem fraction_inequivalence :
  ∃ k : ℝ, k ≠ 0 ∧ k ≠ -1 ∧ (3 * k + 9) / (4 * k + 4) ≠ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequivalence_l3502_350279


namespace NUMINAMATH_CALUDE_cos_210_degrees_l3502_350222

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l3502_350222


namespace NUMINAMATH_CALUDE_outfits_count_l3502_350244

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 4

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of tie options (including not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- The number of jacket options (including not wearing a jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * jacket_options

theorem outfits_count : total_outfits = 576 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3502_350244


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l3502_350232

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 * Nat.factorial (n - 3)) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l3502_350232


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3502_350258

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age : 
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℚ) * A = 
    ((team_size - 2) : ℚ) * (A - 1) + 
    (captain_age : ℚ) + 
    ((captain_age + wicket_keeper_age_diff) : ℚ) →
  A = 31 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3502_350258


namespace NUMINAMATH_CALUDE_circle_theorem_l3502_350260

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A function to check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- A function to check if a point is on a circle -/
def isOn (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- A function to check if four points are concyclic -/
def areConcyclic (p1 p2 p3 p4 : Point) : Prop :=
  ∃ c : Circle, isOn p1 c ∧ isOn p2 c ∧ isOn p3 c ∧ isOn p4 c

theorem circle_theorem (n : ℕ) (points : Fin (2*n+3) → Point) 
  (h : ∀ (a b c d : Fin (2*n+3)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    ¬ areConcyclic (points a) (points b) (points c) (points d)) :
  ∃ (c : Circle) (a b d : Fin (2*n+3)), 
    a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    isOn (points a) c ∧ isOn (points b) c ∧ isOn (points d) c ∧
    (∃ (inside outside : Fin n → Fin (2*n+3)), 
      (∀ i : Fin n, isInside (points (inside i)) c) ∧
      (∀ i : Fin n, ¬isInside (points (outside i)) c)) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l3502_350260


namespace NUMINAMATH_CALUDE_unique_natural_number_with_special_division_property_l3502_350286

theorem unique_natural_number_with_special_division_property :
  ∃! (n : ℕ), ∃ (a b : ℕ),
    n = 12 * b + a ∧
    n = 10 * a + b ∧
    a ≤ 11 ∧
    b ≤ 9 ∧
    n = 119 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_special_division_property_l3502_350286


namespace NUMINAMATH_CALUDE_work_completion_time_l3502_350234

/-- The time (in days) it takes for person a to complete the work alone -/
def time_a : ℝ := 90

/-- The time (in days) it takes for person b to complete the work alone -/
def time_b : ℝ := 45

/-- The time (in days) it takes for persons a, b, and c working together to complete the work -/
def time_together : ℝ := 5

/-- The time (in days) it takes for person c to complete the work alone -/
def time_c : ℝ := 6

/-- The theorem stating that given the work times for a, b, and the group, 
    the time for c to complete the work alone is 6 days -/
theorem work_completion_time :
  (1 / time_a + 1 / time_b + 1 / time_c = 1 / time_together) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3502_350234


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3502_350216

/-- Given a circle centered at (0,b) with radius a, and a hyperbola C: y²/a² - x²/b² = 1 (a > 0, b > 0),
    if the circle and the asymptotes of the hyperbola C are disjoint, 
    then the eccentricity e of C satisfies 1 < e < (√5 + 1)/2. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - b)^2 = a^2}
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * y = a * x ∨ b * y = -a * x}
  let e := Real.sqrt (1 + b^2 / a^2)  -- eccentricity of the hyperbola
  (circle ∩ asymptotes = ∅) → 1 < e ∧ e < (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3502_350216


namespace NUMINAMATH_CALUDE_homework_pages_proof_l3502_350282

theorem homework_pages_proof (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 → 
  math_pages = reading_pages + 3 → 
  total_pages = math_pages + reading_pages → 
  total_pages = 13 := by
sorry

end NUMINAMATH_CALUDE_homework_pages_proof_l3502_350282


namespace NUMINAMATH_CALUDE_smallest_an_correct_l3502_350250

def smallest_an (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 3 then 11
  else 4 * n + 1

theorem smallest_an_correct (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ),
  (∀ i j, 0 ≤ i → i < j → j ≤ n → a i < a j) →
  (∀ i j, 0 ≤ i → i < j → j ≤ n → ¬ Nat.Prime (a j - a i)) →
  a n ≥ smallest_an n :=
sorry

end NUMINAMATH_CALUDE_smallest_an_correct_l3502_350250


namespace NUMINAMATH_CALUDE_difference_of_numbers_l3502_350224

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares : x^2 - y^2 = 24) : 
  |x - y| = 12/5 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l3502_350224


namespace NUMINAMATH_CALUDE_lamp_sales_problem_l3502_350221

/-- Shopping mall lamp sales problem -/
theorem lamp_sales_problem
  (initial_price : ℝ)
  (cost_price : ℝ)
  (initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease_rate : ℝ)
  (h1 : initial_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : 0 < price_increase ∧ price_increase < 20)
  (h5 : sales_decrease_rate = 10) :
  let new_sales := initial_sales - sales_decrease_rate * price_increase
  let new_price := initial_price + price_increase
  let profit := (new_price - cost_price) * new_sales
  ∃ (optimal_increase : ℝ) (max_profit_price : ℝ),
    (new_sales = 600 - 10 * price_increase) ∧
    (profit = 10000 → new_price = 50 ∧ new_sales = 500) ∧
    (max_profit_price = 59 ∧ ∀ x, 0 < x ∧ x < 20 → profit ≤ (59 - cost_price) * (initial_sales - sales_decrease_rate * (59 - initial_price))) :=
by sorry

end NUMINAMATH_CALUDE_lamp_sales_problem_l3502_350221


namespace NUMINAMATH_CALUDE_piglet_gave_two_balloons_l3502_350247

/-- The number of balloons Piglet eventually gave to Eeyore -/
def piglet_balloons : ℕ := 2

/-- The number of balloons Winnie-the-Pooh prepared -/
def pooh_balloons (n : ℕ) : ℕ := 2 * n

/-- The number of balloons Owl prepared -/
def owl_balloons (n : ℕ) : ℕ := 4 * n

/-- The total number of balloons Eeyore received -/
def total_balloons : ℕ := 44

/-- Theorem stating that Piglet gave 2 balloons to Eeyore -/
theorem piglet_gave_two_balloons :
  ∃ (n : ℕ), 
    piglet_balloons + pooh_balloons n + owl_balloons n = total_balloons ∧
    n > piglet_balloons ∧
    piglet_balloons = 2 := by
  sorry


end NUMINAMATH_CALUDE_piglet_gave_two_balloons_l3502_350247


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3502_350229

/-- Given a cube with surface area 150 square units, its volume is 125 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3502_350229


namespace NUMINAMATH_CALUDE_b_visited_city_b_l3502_350201

-- Define the types for students and cities
inductive Student : Type
| A : Student
| B : Student
| C : Student

inductive City : Type
| A : City
| B : City
| C : City

-- Define a function to represent whether a student has visited a city
def hasVisited : Student → City → Prop := sorry

-- State the theorem
theorem b_visited_city_b 
  (h1 : ∀ c : City, hasVisited Student.A c → hasVisited Student.B c)
  (h2 : ¬ hasVisited Student.A City.C)
  (h3 : ¬ hasVisited Student.B City.A)
  (h4 : ∃ c : City, hasVisited Student.A c ∧ hasVisited Student.B c ∧ hasVisited Student.C c)
  : hasVisited Student.B City.B :=
sorry

end NUMINAMATH_CALUDE_b_visited_city_b_l3502_350201


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3502_350249

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3502_350249


namespace NUMINAMATH_CALUDE_residue_mod_14_l3502_350274

theorem residue_mod_14 : (320 * 16 - 28 * 5 + 7) % 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_14_l3502_350274


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3502_350266

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_product_first_four_composites : 
  (first_four_composites.prod % 10 = 8) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3502_350266


namespace NUMINAMATH_CALUDE_drums_per_day_l3502_350253

/-- Given that 266 pickers fill 90 drums in 5 days, prove that the number of drums filled per day is 18. -/
theorem drums_per_day (pickers : ℕ) (total_drums : ℕ) (days : ℕ) 
  (h1 : pickers = 266) 
  (h2 : total_drums = 90) 
  (h3 : days = 5) : 
  total_drums / days = 18 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l3502_350253
