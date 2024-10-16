import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1786_178659

def M : Set ℝ := {-2, 0, 1}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1786_178659


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1786_178660

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1786_178660


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_is_32_l1786_178697

/-- Represents the composition of workers in a factory -/
structure WorkforceComposition where
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  technician_permanent_ratio : ℝ
  non_technician_permanent_ratio : ℝ

/-- Calculates the percentage of temporary employees given a workforce composition -/
def temporary_employee_percentage (wc : WorkforceComposition) : ℝ :=
  100 - (wc.technician_ratio * wc.technician_permanent_ratio + 
         wc.non_technician_ratio * wc.non_technician_permanent_ratio)

/-- The main theorem stating the percentage of temporary employees -/
theorem temporary_employee_percentage_is_32 (wc : WorkforceComposition) 
  (h1 : wc.technician_ratio = 80)
  (h2 : wc.non_technician_ratio = 20)
  (h3 : wc.technician_permanent_ratio = 80)
  (h4 : wc.non_technician_permanent_ratio = 20)
  (h5 : wc.technician_ratio + wc.non_technician_ratio = 100) :
  temporary_employee_percentage wc = 32 := by
  sorry

#eval temporary_employee_percentage ⟨80, 20, 80, 20⟩

end NUMINAMATH_CALUDE_temporary_employee_percentage_is_32_l1786_178697


namespace NUMINAMATH_CALUDE_complex_square_problem_l1786_178690

theorem complex_square_problem (a b : ℝ) (i : ℂ) :
  i^2 = -1 →
  a + i = 2 - b*i →
  (a + b*i)^2 = 3 - 4*i := by
sorry

end NUMINAMATH_CALUDE_complex_square_problem_l1786_178690


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1786_178605

/-- Given a zoo with penguins and polar bears, calculate the total number of animals -/
theorem zoo_animal_count (num_penguins : ℕ) (h1 : num_penguins = 21) 
  (h2 : ∃ (num_polar_bears : ℕ), num_polar_bears = 2 * num_penguins) : 
  ∃ (total_animals : ℕ), total_animals = num_penguins + 2 * num_penguins :=
by sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1786_178605


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l1786_178679

/-- The function f(x) = x ln x - (a/2)x^2 - x is monotonic on (0, +∞) if and only if a ∈ [1/e, +∞) -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x > 0, Monotone (fun x => x * Real.log x - a / 2 * x^2 - x)) ↔ a ≥ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l1786_178679


namespace NUMINAMATH_CALUDE_set_equality_from_union_intersection_equality_l1786_178619

theorem set_equality_from_union_intersection_equality {α : Type*} (A B : Set α) :
  A ∪ B = A ∩ B → A = B := by sorry

end NUMINAMATH_CALUDE_set_equality_from_union_intersection_equality_l1786_178619


namespace NUMINAMATH_CALUDE_bisecting_line_value_l1786_178682

/-- The equation of a line that bisects the circumference of a circle. -/
def bisecting_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- The equation of the circle. -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 2*y + 8 = 0

/-- Theorem stating that if the line y = x + b bisects the circumference of the given circle,
    then b = -5. -/
theorem bisecting_line_value (b : ℝ) :
  (∀ x y : ℝ, bisecting_line b x y ∧ circle_equation x y → 
    ∃ c_x c_y : ℝ, c_x^2 + c_y^2 - 8*c_x + 2*c_y + 8 = 0 ∧ bisecting_line b c_x c_y) →
  b = -5 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_value_l1786_178682


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1786_178687

theorem unique_solution_for_equation : 
  ∃! (m n : ℕ), 3^m - 7^n = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1786_178687


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l1786_178604

/-- The minimum distance between an ellipse and a line -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = (15 : ℝ) / Real.sqrt 41 ∧
  ∀ (x y : ℝ), 
    (x^2 / 25 + y^2 / 9 = 1) →
    (∀ (x' y' : ℝ), (4*x' - 5*y' + 40 = 0) → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l1786_178604


namespace NUMINAMATH_CALUDE_sample_size_is_hundred_l1786_178613

/-- Represents a statistical study on student scores -/
structure ScoreStudy where
  population_size : ℕ
  extracted_size : ℕ

/-- Defines the sample size of a score study -/
def sample_size (study : ScoreStudy) : ℕ := study.extracted_size

/-- Theorem stating that for the given study, the sample size is 100 -/
theorem sample_size_is_hundred (study : ScoreStudy) 
  (h1 : study.population_size = 1000)
  (h2 : study.extracted_size = 100) : 
  sample_size study = 100 := by
  sorry

#check sample_size_is_hundred

end NUMINAMATH_CALUDE_sample_size_is_hundred_l1786_178613


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1786_178601

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1786_178601


namespace NUMINAMATH_CALUDE_car_washing_time_l1786_178641

theorem car_washing_time (x : ℝ) : 
  x > 0 → 
  x + (1/4) * x = 100 → 
  x = 80 :=
by sorry

end NUMINAMATH_CALUDE_car_washing_time_l1786_178641


namespace NUMINAMATH_CALUDE_expression_values_l1786_178662

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1786_178662


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1786_178655

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_a2 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 15) 
  (h_a6 : a 6 = 7) : 
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1786_178655


namespace NUMINAMATH_CALUDE_pancake_milk_calculation_l1786_178600

/-- Given the ratio of pancakes to quarts of milk for 18 pancakes,
    and the conversion rate of quarts to pints,
    prove that the number of pints needed for 9 pancakes is 3. -/
theorem pancake_milk_calculation (pancakes_18 : ℕ) (quarts_18 : ℚ) (pints_per_quart : ℚ) :
  pancakes_18 = 18 →
  quarts_18 = 3 →
  pints_per_quart = 2 →
  (9 : ℚ) * quarts_18 * pints_per_quart / pancakes_18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pancake_milk_calculation_l1786_178600


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1786_178632

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition from the problem
def condition (x y : ℝ) : Prop :=
  x / (1 + i) = 1 - y * i

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : condition x y) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1786_178632


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1786_178693

open Real

theorem trigonometric_expression_evaluation (x : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = fun x ↦ sin x - cos x) 
  (hf' : deriv f = fun x ↦ 2 * f x) : 
  (1 + sin x ^ 2) / (cos x ^ 2 - sin (2 * x)) = -19/5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1786_178693


namespace NUMINAMATH_CALUDE_inequality_proof_l1786_178628

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a * b ≤ 1/4 ∧ Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 ∧ a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1786_178628


namespace NUMINAMATH_CALUDE_pistachio_stairs_l1786_178699

/-- The number of steps between each floor -/
def steps_per_floor : ℕ := 20

/-- The floor where Pistachio lives -/
def target_floor : ℕ := 11

/-- The starting floor -/
def start_floor : ℕ := 1

/-- The total number of steps to reach the target floor -/
def total_steps : ℕ := (target_floor - start_floor) * steps_per_floor

theorem pistachio_stairs : total_steps = 200 := by
  sorry

end NUMINAMATH_CALUDE_pistachio_stairs_l1786_178699


namespace NUMINAMATH_CALUDE_max_integers_above_18_l1786_178670

/-- Given 5 integers that sum to 17, the maximum number of these integers
    that can be larger than 18 is 2. -/
theorem max_integers_above_18 (a b c d e : ℤ) : 
  a + b + c + d + e = 17 → 
  (∀ k : ℕ, k ≤ 5 → 
    (∃ (S : Finset ℤ), S.card = k ∧ S ⊆ {a, b, c, d, e} ∧ (∀ x ∈ S, x > 18)) →
    k ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_max_integers_above_18_l1786_178670


namespace NUMINAMATH_CALUDE_factorial_17_digit_sum_l1786_178694

theorem factorial_17_digit_sum : ∃ (T M H : ℕ),
  (T < 10 ∧ M < 10 ∧ H < 10) ∧
  H = 0 ∧
  (T + M + 35) % 3 = 0 ∧
  (T - M - 2) % 11 = 0 ∧
  T + M + H = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_17_digit_sum_l1786_178694


namespace NUMINAMATH_CALUDE_average_marks_first_class_l1786_178686

theorem average_marks_first_class 
  (students_first_class : ℕ) 
  (students_second_class : ℕ)
  (average_second_class : ℝ)
  (average_all : ℝ) :
  students_first_class = 35 →
  students_second_class = 55 →
  average_second_class = 65 →
  average_all = 57.22222222222222 →
  (students_first_class * (average_all * (students_first_class + students_second_class) - 
   students_second_class * average_second_class)) / 
   (students_first_class * students_first_class) = 45 := by
sorry

end NUMINAMATH_CALUDE_average_marks_first_class_l1786_178686


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1786_178626

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ),
    (1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2)) ∧
    (1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) ∧
    (x = (3^(1/5) + 1) / 2) ∧
    (y = (3^(1/5) - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1786_178626


namespace NUMINAMATH_CALUDE_tangent_lines_count_l1786_178658

/-- The function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Condition for a point (x₀, f(x₀)) to be on a tangent line passing through (0, 0) -/
def is_tangent_point (x₀ : ℝ) : Prop :=
  f x₀ = (f' x₀) * x₀

theorem tangent_lines_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_tangent_point x) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l1786_178658


namespace NUMINAMATH_CALUDE_problem_statement_l1786_178627

theorem problem_statement (x y a : ℝ) 
  (h1 : 2^x = a) 
  (h2 : 3^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1786_178627


namespace NUMINAMATH_CALUDE_max_shipping_cost_l1786_178614

/-- The maximum shipping cost per unit for an electronic component manufacturer --/
theorem max_shipping_cost (production_cost : ℝ) (fixed_costs : ℝ) (units : ℕ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16500)
  (h3 : units = 150)
  (h4 : selling_price = 193.33) :
  ∃ (shipping_cost : ℝ), shipping_cost ≤ 3.33 ∧
    units * (production_cost + shipping_cost) + fixed_costs ≤ units * selling_price :=
by sorry

end NUMINAMATH_CALUDE_max_shipping_cost_l1786_178614


namespace NUMINAMATH_CALUDE_temperature_drop_per_tree_l1786_178643

/-- Proves that the temperature drop per tree is 0.1 degrees -/
theorem temperature_drop_per_tree 
  (cost_per_tree : ℝ) 
  (initial_temp : ℝ) 
  (final_temp : ℝ) 
  (total_cost : ℝ) 
  (h1 : cost_per_tree = 6)
  (h2 : initial_temp = 80)
  (h3 : final_temp = 78.2)
  (h4 : total_cost = 108) :
  (initial_temp - final_temp) / (total_cost / cost_per_tree) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_per_tree_l1786_178643


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l1786_178642

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | ∃ k : ℤ, x = Real.cos (k * Real.pi)}

theorem complement_of_N_in_M : M \ N = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l1786_178642


namespace NUMINAMATH_CALUDE_card_draw_probability_l1786_178651

/-- The number of cards in the set -/
def n : ℕ := 100

/-- The number of draws -/
def k : ℕ := 20

/-- The probability that all drawn numbers are distinct -/
noncomputable def p : ℝ := (n.factorial / (n - k).factorial) / n^k

/-- Main theorem -/
theorem card_draw_probability : p < (9/10)^19 ∧ (9/10)^19 < 1/Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_card_draw_probability_l1786_178651


namespace NUMINAMATH_CALUDE_fifth_place_votes_l1786_178644

theorem fifth_place_votes (total_votes : ℕ) (num_candidates : ℕ) 
  (diff1 diff2 diff3 diff4 : ℕ) :
  total_votes = 3567 →
  num_candidates = 5 →
  diff1 = 143 →
  diff2 = 273 →
  diff3 = 329 →
  diff4 = 503 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - diff1) + (winner_votes - diff2) + 
    (winner_votes - diff3) + (winner_votes - diff4) = total_votes ∧
    winner_votes - diff4 = 700 :=
by sorry

end NUMINAMATH_CALUDE_fifth_place_votes_l1786_178644


namespace NUMINAMATH_CALUDE_product_evaluation_l1786_178608

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1786_178608


namespace NUMINAMATH_CALUDE_max_newspapers_printable_l1786_178611

/-- Represents the number of packages in Box A -/
def box_a_packages : ℕ := 4

/-- Represents the number of sheets per package in Box A -/
def box_a_sheets_per_package : ℕ := 200

/-- Represents the number of packages in Box B -/
def box_b_packages : ℕ := 3

/-- Represents the number of sheets per package in Box B -/
def box_b_sheets_per_package : ℕ := 350

/-- Represents the number of sheets needed for the front page and main articles section -/
def front_page_sheets : ℕ := 10

/-- Represents the number of sheets needed for the sports and clubs section -/
def sports_sheets : ℕ := 7

/-- Represents the number of sheets needed for the arts and entertainment section -/
def arts_sheets : ℕ := 5

/-- Represents the number of sheets needed for the school events and announcements section -/
def events_sheets : ℕ := 3

/-- Theorem stating the maximum number of complete newspapers that can be printed -/
theorem max_newspapers_printable : 
  (box_a_packages * box_a_sheets_per_package + box_b_packages * box_b_sheets_per_package) / 
  (front_page_sheets + sports_sheets + arts_sheets + events_sheets) = 74 := by
  sorry

end NUMINAMATH_CALUDE_max_newspapers_printable_l1786_178611


namespace NUMINAMATH_CALUDE_minimum_orchestra_size_l1786_178665

theorem minimum_orchestra_size : ∃ n : ℕ, n > 0 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 9 = 0 → m % 10 = 0 → m % 11 = 0 → m ≥ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_minimum_orchestra_size_l1786_178665


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1786_178652

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 70,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    prove that the base of the isosceles triangle is 30 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 70)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   isosceles_perimeter = 2 * side + (isosceles_perimeter - 2 * side)) :
  isosceles_perimeter - 2 * (equilateral_perimeter / 3) = 30 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1786_178652


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l1786_178616

theorem sqrt_three_squared : Real.sqrt 3 ^ 2 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l1786_178616


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l1786_178688

theorem triangle_max_side_length (D E F : Real) (side1 side2 : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧
  D + E + F = Real.pi ∧
  -- Given condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 ∧
  -- Two sides have lengths 8 and 15
  side1 = 8 ∧ side2 = 15 →
  -- The maximum length of the third side is 17
  ∃ side3 : Real, side3 ≤ 17 ∧
    ∀ x : Real, (∃ D' E' F' : Real,
      0 < D' ∧ 0 < E' ∧ 0 < F' ∧
      D' + E' + F' = Real.pi ∧
      Real.cos (2 * D') + Real.cos (2 * E') + Real.cos (2 * F') = 1 ∧
      x = ((side1^2 + side2^2 - 2 * side1 * side2 * Real.cos F')^(1/2))) →
    x ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l1786_178688


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1786_178639

-- Define set A
def A : Set ℝ := Set.univ

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x + 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Iic 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1786_178639


namespace NUMINAMATH_CALUDE_tracy_candies_l1786_178636

theorem tracy_candies : ∃ (x : ℕ), 
  (x % 4 = 0) ∧ 
  ((3 * x / 4) % 3 = 0) ∧ 
  (x / 2 - 29 = 10) ∧ 
  x = 78 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_l1786_178636


namespace NUMINAMATH_CALUDE_geometric_sum_2021_l1786_178609

theorem geometric_sum_2021 (x : ℝ) (h1 : x^2021 - 3*x + 1 = 0) (h2 : x ≠ 1) :
  x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + x^2012 + x^2011 + 
  x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + x^2002 + x^2001 + 
  x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + x^1992 + x^1991 + 
  x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + x^1982 + x^1981 + 
  x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + x^1972 + x^1971 + 
  -- ... (continuing the pattern)
  x^50 + x^49 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^42 + x^41 + 
  x^40 + x^39 + x^38 + x^37 + x^36 + x^35 + x^34 + x^33 + x^32 + x^31 + 
  x^30 + x^29 + x^28 + x^27 + x^26 + x^25 + x^24 + x^23 + x^22 + x^21 + 
  x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 + 
  x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_2021_l1786_178609


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1786_178633

def solution_set (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 3) * (x + 2) < 0 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1786_178633


namespace NUMINAMATH_CALUDE_reciprocal_equation_l1786_178624

theorem reciprocal_equation (x : ℝ) : 1 - 1 / (1 - x) = 1 / (1 - x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l1786_178624


namespace NUMINAMATH_CALUDE_unique_bisecting_line_l1786_178696

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  side1_eq : side1 = 6
  side2_eq : side2 = 8
  hypotenuse_eq : hypotenuse = 10
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- A line that potentially bisects the area and perimeter of the triangle -/
structure BisectingLine (t : RightTriangle) where
  x : ℝ  -- distance from a vertex on one side
  y : ℝ  -- distance from the same vertex on another side
  bisects_area : x * y = 30  -- specific to this triangle
  bisects_perimeter : x + y = (t.side1 + t.side2 + t.hypotenuse) / 2

/-- There exists a unique bisecting line for the given right triangle -/
theorem unique_bisecting_line (t : RightTriangle) : 
  ∃! (l : BisectingLine t), True :=
sorry

end NUMINAMATH_CALUDE_unique_bisecting_line_l1786_178696


namespace NUMINAMATH_CALUDE_min_value_of_squares_l1786_178630

theorem min_value_of_squares (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) :
  (∀ x y : ℝ, a > x ∧ x > y ∧ y > c → (a - x)^2 + (x - y)^2 ≥ (a - b)^2 + (b - c)^2) ∧
  (a - b)^2 + (b - c)^2 = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l1786_178630


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1786_178680

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((b * Real.cos C) / Real.cos B + c = (2 * Real.sqrt 3 / 3) * a) →
  B = π / 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_proof_l1786_178680


namespace NUMINAMATH_CALUDE_sum_fractions_and_integer_l1786_178610

theorem sum_fractions_and_integer : 
  (3 / 20 : ℚ) + (5 / 200 : ℚ) + (7 / 2000 : ℚ) + 5 = 5.1785 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_and_integer_l1786_178610


namespace NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l1786_178698

theorem five_consecutive_integers_product_not_square (a : ℕ+) :
  ∃ (n : ℕ), (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) : ℕ) ≠ n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l1786_178698


namespace NUMINAMATH_CALUDE_remaining_milk_average_price_l1786_178656

/-- Calculates the average price of remaining milk packets after returning some packets. -/
theorem remaining_milk_average_price
  (total_packets : ℕ)
  (initial_avg_price : ℚ)
  (returned_packets : ℕ)
  (returned_avg_price : ℚ)
  (h1 : total_packets = 5)
  (h2 : initial_avg_price = 20/100)
  (h3 : returned_packets = 2)
  (h4 : returned_avg_price = 32/100)
  : (total_packets * initial_avg_price - returned_packets * returned_avg_price) / (total_packets - returned_packets) = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_milk_average_price_l1786_178656


namespace NUMINAMATH_CALUDE_fraction_addition_and_multiplication_l1786_178669

theorem fraction_addition_and_multiplication :
  (7 / 12 + 3 / 8) * 2 / 3 = 23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_and_multiplication_l1786_178669


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l1786_178646

-- Define a circle
def Circle : Type := Unit

-- Define a color
inductive Color
| Red
| Blue

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define a coloring of the circle
def Coloring (c : Circle) := Point c → Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  a : Point c
  b : Point c
  c : Point c
  isIsosceles : True  -- Placeholder for the isosceles property

-- Theorem statement
theorem isosceles_triangle_exists (c : Circle) (coloring : Coloring c) :
  ∃ (t : IsoscelesTriangle c), t.a.color = t.b.color ∧ t.b.color = t.c.color :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l1786_178646


namespace NUMINAMATH_CALUDE_convex_shape_volume_is_half_l1786_178678

/-- A cube with midlines of each face divided in a 1:3 ratio -/
structure DividedCube where
  /-- The volume of the original cube -/
  volume : ℝ
  /-- The ratio in which the midlines are divided -/
  divisionRatio : ℝ
  /-- Assumption that the division ratio is 1:3 -/
  ratio_is_one_three : divisionRatio = 1/3

/-- The volume of the convex shape formed by the points dividing the midlines -/
def convexShapeVolume (c : DividedCube) : ℝ := sorry

/-- Theorem stating that the volume of the convex shape is half the volume of the cube -/
theorem convex_shape_volume_is_half (c : DividedCube) : 
  convexShapeVolume c = c.volume / 2 := by sorry

end NUMINAMATH_CALUDE_convex_shape_volume_is_half_l1786_178678


namespace NUMINAMATH_CALUDE_simplify_expression_l1786_178691

theorem simplify_expression (x : ℝ) : 2 * x^5 * (3 * x^9) = 6 * x^14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1786_178691


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1786_178634

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := (n * n) + (n * (n - 1) * n) / 2

/-- Theorem stating that the total number of course selection schemes is 64 -/
theorem course_selection_schemes :
  total_schemes = 64 := by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1786_178634


namespace NUMINAMATH_CALUDE_function_inequality_solution_l1786_178623

/-- Given a function f defined on positive integers and a constant a,
    prove that f(n) = a^(n*(n-1)/2) * f(1) satisfies f(n+1) ≥ a^n * f(n) for all positive integers n. -/
theorem function_inequality_solution (a : ℝ) (f : ℕ+ → ℝ) :
  (∀ n : ℕ+, f n = a^(n.val*(n.val-1)/2) * f 1) →
  (∀ n : ℕ+, f (n + 1) ≥ a^n.val * f n) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l1786_178623


namespace NUMINAMATH_CALUDE_cricket_innings_calculation_l1786_178672

/-- The number of innings played by a cricket player -/
def innings : ℕ := sorry

/-- The current average runs per innings -/
def current_average : ℚ := 22

/-- The increase in average after scoring 92 runs in the next innings -/
def average_increase : ℚ := 5

/-- The runs scored in the next innings -/
def next_innings_runs : ℕ := 92

theorem cricket_innings_calculation :
  (innings * current_average + next_innings_runs) / (innings + 1) = current_average + average_increase →
  innings = 13 := by sorry

end NUMINAMATH_CALUDE_cricket_innings_calculation_l1786_178672


namespace NUMINAMATH_CALUDE_x_eq_1_sufficient_not_necessary_for_quadratic_l1786_178661

theorem x_eq_1_sufficient_not_necessary_for_quadratic : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_1_sufficient_not_necessary_for_quadratic_l1786_178661


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l1786_178684

theorem cubic_equation_properties (k : ℝ) :
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                (x ≠ y ∨ y ≠ z ∨ x ≠ z)) ∧
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                x + y + z = -2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l1786_178684


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_000123_l1786_178692

/-- The scientific notation of a number is represented as a float and an integer exponent -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Converts a float to its scientific notation representation -/
def toScientificNotation (x : Float) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_000123 :
  toScientificNotation 0.000123 = ScientificNotation.mk 1.23 (-4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_000123_l1786_178692


namespace NUMINAMATH_CALUDE_exists_circle_with_n_grid_points_l1786_178664

/-- A grid point is a point with integer coordinates -/
def GridPoint : Type := ℤ × ℤ

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Count the number of grid points within a circle -/
def countGridPointsInCircle (c : Circle) : ℕ :=
  sorry

/-- Main theorem: For any positive integer n, there exists a circle with exactly n grid points -/
theorem exists_circle_with_n_grid_points (n : ℕ) (hn : n > 0) :
  ∃ (c : Circle), countGridPointsInCircle c = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_grid_points_l1786_178664


namespace NUMINAMATH_CALUDE_domain_of_sqrt_tan_minus_sqrt3_l1786_178695

/-- The domain of the function y = √(tan x - √3) -/
theorem domain_of_sqrt_tan_minus_sqrt3 (x : ℝ) :
  x ∈ {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2} ↔
  ∃ y : ℝ, y = Real.sqrt (Real.tan x - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_tan_minus_sqrt3_l1786_178695


namespace NUMINAMATH_CALUDE_inequality_holds_l1786_178649

theorem inequality_holds (x y z : ℝ) : 4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1786_178649


namespace NUMINAMATH_CALUDE_simplest_square_root_l1786_178671

theorem simplest_square_root :
  let options : List ℝ := [Real.sqrt 5, Real.sqrt 4, Real.sqrt 12, Real.sqrt (1/2)]
  ∀ x ∈ options, x ≠ Real.sqrt 5 → ∃ y : ℝ, y * y = x ∧ y ≠ x :=
by sorry

end NUMINAMATH_CALUDE_simplest_square_root_l1786_178671


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l1786_178685

/-- The number of oranges harvested per day (not discarded) -/
def oranges_harvested (sacks_per_day : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_per_day - sacks_discarded) * oranges_per_sack

theorem orange_harvest_theorem :
  oranges_harvested 76 64 50 = 600 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l1786_178685


namespace NUMINAMATH_CALUDE_distinct_values_count_l1786_178647

def parenthesization1 : ℕ := 3^(3^(3^3))
def parenthesization2 : ℕ := 3^((3^3)^3)
def parenthesization3 : ℕ := ((3^3)^3)^3
def parenthesization4 : ℕ := 3^((3^3)^(3^2))

def distinctValues : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4}

theorem distinct_values_count :
  Finset.card distinctValues = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1786_178647


namespace NUMINAMATH_CALUDE_total_sequences_is_288_l1786_178653

/-- Represents a team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  day1_matches : List Match
  no_ties : Bool

/-- Calculates the number of possible outcomes for a given number of matches -/
def possible_outcomes (num_matches : Nat) : Nat :=
  2^num_matches

/-- Calculates the number of possible arrangements for the winners' group on day 2 -/
def winners_arrangements (num_winners : Nat) : Nat :=
  Nat.factorial num_winners

/-- Calculates the number of possible outcomes for the losers' match on day 2 -/
def losers_match_outcomes (num_losers : Nat) : Nat :=
  num_losers * 2

/-- Calculates the total number of possible ranking sequences -/
def total_sequences (t : Tournament) : Nat :=
  possible_outcomes t.day1_matches.length *
  winners_arrangements 3 *
  losers_match_outcomes 3 *
  possible_outcomes 1

/-- The theorem stating that the total number of possible ranking sequences is 288 -/
theorem total_sequences_is_288 (t : Tournament) 
  (h1 : t.day1_matches.length = 3)
  (h2 : t.no_ties = true) :
  total_sequences t = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_sequences_is_288_l1786_178653


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_equals_one_l1786_178657

theorem no_intersection_implies_k_equals_one (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_equals_one_l1786_178657


namespace NUMINAMATH_CALUDE_median_equation_altitude_equation_l1786_178637

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

-- Theorem for the median equation
theorem median_equation : 
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 5*x + y = 20) ∧
    is_line_equation a b c A ∧
    is_line_equation a b c ((B.1 + C.1) / 2, (B.2 + C.2) / 2) :=
sorry

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (a b c : ℝ),
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 3*x + 2*y = 12) ∧
    is_line_equation a b c A ∧
    (∀ (p : ℝ × ℝ), is_line_equation (B.2 - C.2) (C.1 - B.1) 0 p → 
      (p.2 - A.2) * (p.1 - A.1) = -(a * (p.1 - A.1) + b * (p.2 - A.2))^2 / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_median_equation_altitude_equation_l1786_178637


namespace NUMINAMATH_CALUDE_point_on_same_side_l1786_178676

def sameSideOfLine (p1 p2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x1 + y1 - 1) * (x2 + y2 - 1) > 0

def referencePt : ℝ × ℝ := (1, 2)

theorem point_on_same_side : 
  sameSideOfLine (-1, 3) referencePt ∧ 
  ¬sameSideOfLine (0, 0) referencePt ∧ 
  ¬sameSideOfLine (-1, 1) referencePt ∧ 
  ¬sameSideOfLine (2, -3) referencePt :=
by sorry

end NUMINAMATH_CALUDE_point_on_same_side_l1786_178676


namespace NUMINAMATH_CALUDE_problem_statement_l1786_178607

theorem problem_statement (θ : ℝ) 
  (h : Real.sin (π / 4 - θ) + Real.cos (π / 4 - θ) = 1 / 5) :
  Real.cos (2 * θ) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1786_178607


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1786_178674

/-- The number of ways to arrange n unique objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two groups of books, where each group stays together --/
def arrange_book_groups : ℕ := permutations 2

/-- The number of ways to arrange 4 unique math books within their group --/
def arrange_math_books : ℕ := permutations 4

/-- The number of ways to arrange 4 unique English books within their group --/
def arrange_english_books : ℕ := permutations 4

/-- The total number of ways to arrange 4 unique math books and 4 unique English books on a shelf,
    with all math books staying together and all English books staying together --/
def total_arrangements : ℕ := arrange_book_groups * arrange_math_books * arrange_english_books

theorem book_arrangement_theorem : total_arrangements = 1152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1786_178674


namespace NUMINAMATH_CALUDE_john_widget_production_rate_l1786_178667

/-- Represents the number of widgets John can make in an hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the number of days John works per week -/
def days_per_week : ℕ := 5

/-- Represents the total number of widgets John makes in a week -/
def widgets_per_week : ℕ := 800

/-- Proves that the number of widgets John can make in an hour is 20 -/
theorem john_widget_production_rate : 
  widgets_per_hour * (hours_per_day * days_per_week) = widgets_per_week :=
by sorry

end NUMINAMATH_CALUDE_john_widget_production_rate_l1786_178667


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l1786_178612

/-- Represents a game board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Nat × Nat)

/-- Check if a board can be covered by dominoes -/
def can_be_covered (b : Board) (p : DominoPlacement) : Prop :=
  (b.rows * b.cols = 2 * p.length) ∧ 
  (b.black_squares = b.white_squares)

/-- The 5x5 board with specific color pattern -/
def board_5x5 : Board :=
  { rows := 5
  , cols := 5
  , black_squares := 9   -- central 3x3 section
  , white_squares := 16  -- border
  }

/-- Theorem stating that the 5x5 board cannot be covered -/
theorem five_by_five_uncoverable : 
  ∀ p : DominoPlacement, ¬(can_be_covered board_5x5 p) :=
sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l1786_178612


namespace NUMINAMATH_CALUDE_base9_734_equals_base10_598_l1786_178681

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 9^0 + d₁ * 9^1 + d₂ * 9^2

/-- Theorem: The base-9 number 734 is equal to 598 in base-10 --/
theorem base9_734_equals_base10_598 : base9ToBase10 7 3 4 = 598 := by
  sorry

#eval base9ToBase10 7 3 4

end NUMINAMATH_CALUDE_base9_734_equals_base10_598_l1786_178681


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l1786_178673

def total_figures : ℕ := 12
def num_triangles : ℕ := 4
def num_circles : ℕ := 3
def num_squares : ℕ := 5

theorem probability_triangle_or_circle :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l1786_178673


namespace NUMINAMATH_CALUDE_even_sum_difference_l1786_178620

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 62 110 - sum_even_range 42 90 = 500 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l1786_178620


namespace NUMINAMATH_CALUDE_pairwise_ratio_sum_geq_three_halves_l1786_178645

theorem pairwise_ratio_sum_geq_three_halves
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pairwise_ratio_sum_geq_three_halves_l1786_178645


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l1786_178677

theorem min_value_squared_sum (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l1786_178677


namespace NUMINAMATH_CALUDE_puppy_and_food_cost_l1786_178625

/-- Calculates the total cost of a puppy and food for a given number of weeks -/
def totalCost (puppyCost : ℚ) (foodPerDay : ℚ) (daysSupply : ℕ) (cupPerBag : ℚ) (bagCost : ℚ) : ℚ :=
  let totalDays : ℕ := daysSupply
  let totalFood : ℚ := (totalDays : ℚ) * foodPerDay
  let bagsNeeded : ℚ := totalFood / cupPerBag
  let foodCost : ℚ := bagsNeeded * bagCost
  puppyCost + foodCost

/-- Theorem stating that the total cost of a puppy and food for 3 weeks is $14 -/
theorem puppy_and_food_cost :
  totalCost 10 (1/3) 21 (7/2) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_puppy_and_food_cost_l1786_178625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1786_178603

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (m p q : ℕ) 
  (h_arithmetic : is_arithmetic_sequence a) (h_positive : m > 0 ∧ p > 0 ∧ q > 0) :
  (p + q = 2 * m → a p + a q = 2 * a m) ∧
  ∃ b : ℕ → ℝ, is_arithmetic_sequence b ∧ ∃ m' p' q' : ℕ, 
    m' > 0 ∧ p' > 0 ∧ q' > 0 ∧ b p' + b q' = 2 * b m' ∧ p' + q' ≠ 2 * m' :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1786_178603


namespace NUMINAMATH_CALUDE_multiply_b_is_eight_l1786_178668

theorem multiply_b_is_eight (a b x : ℝ) 
  (h1 : 7 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 8) / (b / 7) = 1) : 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_b_is_eight_l1786_178668


namespace NUMINAMATH_CALUDE_distance_to_line_l1786_178618

/-- Represents a line in 2D space using parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the distance from a point to a line given in parametric form --/
def distanceToParametricLine (px py : ℝ) (line : ParametricLine) : ℝ :=
  sorry

/-- The problem statement --/
theorem distance_to_line : 
  let l : ParametricLine := { x := λ t => 1 + t, y := λ t => -1 + t }
  let p : (ℝ × ℝ) := (4, 0)
  distanceToParametricLine p.1 p.2 l = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l1786_178618


namespace NUMINAMATH_CALUDE_uncle_joe_parking_probability_l1786_178663

/-- The number of parking spaces -/
def total_spaces : ℕ := 18

/-- The number of cars that have parked -/
def parked_cars : ℕ := 14

/-- The number of adjacent spaces required for Uncle Joe's truck -/
def required_spaces : ℕ := 2

/-- The probability of finding two adjacent empty spaces -/
def probability_of_parking : ℚ := 113 / 204

theorem uncle_joe_parking_probability :
  probability_of_parking = 1 - (Nat.choose (total_spaces - parked_cars + required_spaces - 1) (required_spaces - 1)) / (Nat.choose total_spaces parked_cars) :=
by sorry

end NUMINAMATH_CALUDE_uncle_joe_parking_probability_l1786_178663


namespace NUMINAMATH_CALUDE_smallest_winning_number_l1786_178683

theorem smallest_winning_number : ∃ N : ℕ,
  N ≥ 0 ∧ N ≤ 999 ∧
  (∀ m : ℕ, m ≥ 0 ∧ m < N →
    (3*m < 1000 ∧
     3*m - 30 < 1000 ∧
     9*m - 90 < 1000 ∧
     9*m - 120 < 1000 ∧
     27*m - 360 < 1000 ∧
     27*m - 390 < 1000 ∧
     81*m - 1170 < 1000 ∧
     81*(m-1) - 1170 ≥ 1000)) ∧
  3*N < 1000 ∧
  3*N - 30 < 1000 ∧
  9*N - 90 < 1000 ∧
  9*N - 120 < 1000 ∧
  27*N - 360 < 1000 ∧
  27*N - 390 < 1000 ∧
  81*N - 1170 < 1000 ∧
  81*(N-1) - 1170 ≥ 1000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l1786_178683


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1786_178622

theorem at_least_one_greater_than_one (a b : ℝ) : a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1786_178622


namespace NUMINAMATH_CALUDE_calculate_net_profit_l1786_178666

/-- Given a purchase price, overhead percentage, and markup, calculate the net profit -/
theorem calculate_net_profit (purchase_price overhead_percentage markup : ℝ) :
  purchase_price = 48 →
  overhead_percentage = 0.20 →
  markup = 45 →
  let overhead := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead
  let selling_price := total_cost + markup
  let net_profit := selling_price - total_cost
  net_profit = 45 := by
  sorry

end NUMINAMATH_CALUDE_calculate_net_profit_l1786_178666


namespace NUMINAMATH_CALUDE_square_cut_diagonal_length_l1786_178654

theorem square_cut_diagonal_length (s x : ℝ) : 
  s > 0 → 
  x > 0 → 
  x^2 = 72 → 
  s^2 = 2 * x^2 → 
  (s - 2*x)^2 + (s - 2*x)^2 = 12^2 := by
sorry

end NUMINAMATH_CALUDE_square_cut_diagonal_length_l1786_178654


namespace NUMINAMATH_CALUDE_shortest_dragon_length_l1786_178631

/-- A function that calculates the sum of digits of a positive integer -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a set of k consecutive positive integers contains a number whose digit sum is divisible by 11 -/
def isDragon (start : ℕ) (k : ℕ) : Prop :=
  ∃ i : ℕ, i < k ∧ (digitSum (start + i) % 11 = 0)

/-- The theorem stating that 39 is the smallest dragon length -/
theorem shortest_dragon_length : 
  (∀ start : ℕ, isDragon start 39) ∧ 
  (∀ k : ℕ, k < 39 → ∃ start : ℕ, ¬isDragon start k) :=
sorry

end NUMINAMATH_CALUDE_shortest_dragon_length_l1786_178631


namespace NUMINAMATH_CALUDE_curve_is_semicircle_l1786_178606

-- Define the curve
def curve (x y : ℝ) : Prop := x - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def semicircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧ x ≥ center.1

theorem curve_is_semicircle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ), curve x y ↔ semicircle center radius x y :=
sorry

end NUMINAMATH_CALUDE_curve_is_semicircle_l1786_178606


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l1786_178640

theorem fifty_billion_scientific_notation :
  (50000000000 : ℝ) = 5.0 * (10 : ℝ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l1786_178640


namespace NUMINAMATH_CALUDE_base7_135_equals_base10_75_l1786_178689

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 135 in base 7 is equal to 75 in base 10 --/
theorem base7_135_equals_base10_75 : base7ToBase10 1 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_base7_135_equals_base10_75_l1786_178689


namespace NUMINAMATH_CALUDE_max_intersection_points_for_arrangement_l1786_178602

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the arrangement of two convex polygons in a plane -/
structure PolygonArrangement where
  A₁ : ConvexPolygon
  A₂ : ConvexPolygon
  same_plane : Bool
  can_intersect : Bool
  no_full_overlap : Bool

/-- Calculates the maximum number of intersection points between two polygons -/
def max_intersection_points (arr : PolygonArrangement) : ℕ :=
  arr.A₁.sides * arr.A₂.sides

/-- Theorem stating the maximum number of intersection points for the given arrangement -/
theorem max_intersection_points_for_arrangement 
  (m : ℕ) 
  (arr : PolygonArrangement) 
  (h1 : arr.A₁.sides = m) 
  (h2 : arr.A₂.sides = m + 2) 
  (h3 : arr.same_plane) 
  (h4 : arr.can_intersect) 
  (h5 : arr.no_full_overlap) 
  (h6 : arr.A₁.convex) 
  (h7 : arr.A₂.convex) : 
  max_intersection_points arr = m^2 + 2*m := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_for_arrangement_l1786_178602


namespace NUMINAMATH_CALUDE_journey_distance_l1786_178648

theorem journey_distance (speed1 speed2 time1 total_time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 70)
  (h3 : time1 = 3.2)
  (h4 : total_time = 8) :
  speed1 * time1 + speed2 * (total_time - time1) = 400 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1786_178648


namespace NUMINAMATH_CALUDE_printer_time_ratio_l1786_178617

/-- Given four printers with their individual completion times, prove the ratio of time taken by printer x alone to the time taken by printers y, z, and w together. -/
theorem printer_time_ratio (x y z w : ℝ) (hx : x = 12) (hy : y = 10) (hz : z = 20) (hw : w = 15) :
  x / (1 / (1/y + 1/z + 1/w)) = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_ratio_l1786_178617


namespace NUMINAMATH_CALUDE_min_value_theorem_l1786_178675

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1786_178675


namespace NUMINAMATH_CALUDE_annulus_area_l1786_178615

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (B C RW : ℝ) (h1 : B > C) (h2 : B^2 - (C+5)^2 = RW^2) :
  (π * B^2) - (π * (C+5)^2) = π * RW^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l1786_178615


namespace NUMINAMATH_CALUDE_perimeter_is_28_inches_l1786_178638

/-- Represents a rectangle with width and height in inches -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of the T-L configuration -/
def perimeter_of_configuration (rect : Rectangle) : ℝ :=
  let horizontal_exposed := 2 * rect.width - 1
  let vertical_exposed := 3 * rect.height + 2 * rect.width
  horizontal_exposed + vertical_exposed

/-- Theorem stating that the perimeter of the T-L configuration is 28 inches -/
theorem perimeter_is_28_inches (rect : Rectangle) 
  (h1 : rect.width = 3)
  (h2 : rect.height = 5) : 
  perimeter_of_configuration rect = 28 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_28_inches_l1786_178638


namespace NUMINAMATH_CALUDE_remaining_fun_is_1050_l1786_178629

/-- Calculates the remaining amount for fun after a series of financial actions --/
def remaining_for_fun (initial_winnings : ℝ) (tax_rate : ℝ) (mortgage_rate : ℝ) 
  (retirement_rate : ℝ) (college_rate : ℝ) (savings : ℝ) : ℝ :=
  let after_tax := initial_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  after_college - savings

/-- Theorem stating that given the specific financial actions, 
    the remaining amount for fun is $1050 --/
theorem remaining_fun_is_1050 : 
  remaining_for_fun 20000 0.55 0.5 (1/3) 0.25 1200 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fun_is_1050_l1786_178629


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1786_178621

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 3 ^ 2 = a 1 * a 9) :
  a 3 / a 6 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1786_178621


namespace NUMINAMATH_CALUDE_product_of_primes_between_n_and_2n_l1786_178635

theorem product_of_primes_between_n_and_2n (n : ℤ) :
  (n > 4 → ∃ p : ℕ, Prime p ∧ n < 2*p ∧ 2*p < 2*n) ∧
  (n > 15 → ∃ p : ℕ, Prime p ∧ n < 6*p ∧ 6*p < 2*n) :=
sorry

end NUMINAMATH_CALUDE_product_of_primes_between_n_and_2n_l1786_178635


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1786_178650

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1786_178650
