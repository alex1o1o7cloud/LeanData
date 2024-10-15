import Mathlib

namespace NUMINAMATH_CALUDE_unique_cube_root_between_9_and_9_2_l4003_400357

theorem unique_cube_root_between_9_and_9_2 :
  ∃! n : ℕ+, 27 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_between_9_and_9_2_l4003_400357


namespace NUMINAMATH_CALUDE_nail_boxes_theorem_l4003_400332

theorem nail_boxes_theorem : ∃ (a b c d : ℕ), 24 * a + 23 * b + 17 * c + 16 * d = 100 := by
  sorry

end NUMINAMATH_CALUDE_nail_boxes_theorem_l4003_400332


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l4003_400367

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, n > 0 ∧ Nat.Prime n ∧ (∃ m : ℕ, n = m^2 - 12) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < n → ¬(Nat.Prime k ∧ ∃ m : ℕ, k = m^2 - 12)) ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l4003_400367


namespace NUMINAMATH_CALUDE_initial_number_proof_l4003_400337

theorem initial_number_proof (x : ℕ) : 7899665 - (3 * 2 * x) = 7899593 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l4003_400337


namespace NUMINAMATH_CALUDE_juice_cans_for_two_dollars_l4003_400341

def anniversary_sale (original_price : ℕ) (discount : ℕ) (total_cost : ℕ) (ice_cream_count : ℕ) (juice_cans : ℕ) : Prop :=
  let sale_price := original_price - discount
  let ice_cream_total := sale_price * ice_cream_count
  let juice_cost := total_cost - ice_cream_total
  ∃ (cans_per_two_dollars : ℕ), 
    cans_per_two_dollars * (juice_cost / 2) = juice_cans ∧
    cans_per_two_dollars = 5

theorem juice_cans_for_two_dollars :
  anniversary_sale 12 2 24 2 10 → ∃ (x : ℕ), x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_juice_cans_for_two_dollars_l4003_400341


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l4003_400320

theorem absolute_value_sum_difference : |(-8)| + (-6) - (-12) = 14 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l4003_400320


namespace NUMINAMATH_CALUDE_coordinates_determine_location_kunming_location_determined_l4003_400396

-- Define a structure for geographical coordinates
structure GeoCoordinates where
  longitude : Real
  latitude : Real

-- Define a function to check if coordinates are valid
def isValidCoordinates (coords : GeoCoordinates) : Prop :=
  -180 ≤ coords.longitude ∧ coords.longitude ≤ 180 ∧
  -90 ≤ coords.latitude ∧ coords.latitude ≤ 90

-- Define a function to determine if coordinates specify a unique location
def specifiesUniqueLocation (coords : GeoCoordinates) : Prop :=
  isValidCoordinates coords

-- Theorem stating that valid coordinates determine a specific location
theorem coordinates_determine_location (coords : GeoCoordinates) :
  isValidCoordinates coords → specifiesUniqueLocation coords :=
by
  sorry

-- Example using the coordinates from the problem
def kunming_coords : GeoCoordinates :=
  { longitude := 102, latitude := 24 }

-- Theorem stating that the Kunming coordinates determine a specific location
theorem kunming_location_determined :
  specifiesUniqueLocation kunming_coords :=
by
  sorry

end NUMINAMATH_CALUDE_coordinates_determine_location_kunming_location_determined_l4003_400396


namespace NUMINAMATH_CALUDE_ones_digit_sum_powers_l4003_400313

theorem ones_digit_sum_powers (n : Nat) : n = 2023 → 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_sum_powers_l4003_400313


namespace NUMINAMATH_CALUDE_base_eight_47_to_base_ten_l4003_400304

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (d1 d2 : Nat) : Nat :=
  d1 * 8 + d2

/-- The base-eight number 47 -/
def base_eight_47 : Nat × Nat := (4, 7)

theorem base_eight_47_to_base_ten :
  base_eight_to_ten base_eight_47.1 base_eight_47.2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_to_base_ten_l4003_400304


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l4003_400310

/-- Calculates the cost of soap for a year given the duration and price of a single bar -/
def soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℚ) : ℚ :=
  (12 / months_per_bar) * price_per_bar

/-- Theorem stating that for soap lasting 2 months and costing $8.00, the yearly cost is $48.00 -/
theorem soap_cost_theorem : soap_cost_for_year 2 8 = 48 := by
  sorry

#eval soap_cost_for_year 2 8

end NUMINAMATH_CALUDE_soap_cost_theorem_l4003_400310


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l4003_400334

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l4003_400334


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l4003_400351

theorem solution_set_abs_inequality (x : ℝ) :
  (|2*x + 3| < 1) ↔ (-2 < x ∧ x < -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l4003_400351


namespace NUMINAMATH_CALUDE_square_root_property_l4003_400300

theorem square_root_property (x : ℝ) :
  Real.sqrt (x + 4) = 3 → (x + 4)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_property_l4003_400300


namespace NUMINAMATH_CALUDE_total_students_l4003_400384

/-- The number of students in each classroom -/
structure ClassroomSizes where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions of the problem -/
def problem_conditions (sizes : ClassroomSizes) : Prop :=
  sizes.tina = sizes.maura ∧
  sizes.zack = (sizes.tina + sizes.maura) / 2 ∧
  sizes.zack = 23

/-- The theorem stating the total number of students -/
theorem total_students (sizes : ClassroomSizes) 
  (h : problem_conditions sizes) : 
  sizes.tina + sizes.maura + sizes.zack = 69 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l4003_400384


namespace NUMINAMATH_CALUDE_lele_can_afford_cars_with_change_l4003_400302

def price_a : ℚ := 46.5
def price_b : ℚ := 54.5
def lele_money : ℚ := 120

theorem lele_can_afford_cars_with_change : 
  price_a + price_b ≤ lele_money ∧ lele_money - (price_a + price_b) = 19 :=
by sorry

end NUMINAMATH_CALUDE_lele_can_afford_cars_with_change_l4003_400302


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l4003_400358

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallelEdgePairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism with different dimensions has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallelEdgePairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l4003_400358


namespace NUMINAMATH_CALUDE_minimum_value_problem_l4003_400386

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.sqrt 2 = Real.sqrt (2^a * 4^b) → 1/x + x/y ≤ 1/a + a/b) ∧
  (1/x + x/y = 2 * Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l4003_400386


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l4003_400382

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l4003_400382


namespace NUMINAMATH_CALUDE_matrix_product_equals_C_l4003_400377

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -2; 0, 2, 4]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 0, 2, -1; 3, 0, 1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, -3; -5, 5, -5; 12, 4, 2]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_equals_C_l4003_400377


namespace NUMINAMATH_CALUDE_equation_solutions_l4003_400340

def equation (x : ℝ) : Prop :=
  (17 * x - x^2) / (x + 2) * (x + (17 - x) / (x + 2)) = 48

theorem equation_solutions :
  {x : ℝ | equation x} = {3, 4, -10 + 4 * Real.sqrt 21, -10 - 4 * Real.sqrt 21} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4003_400340


namespace NUMINAMATH_CALUDE_num_teachers_at_king_middle_school_l4003_400353

/-- The number of students at King Middle School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem: The number of teachers at King Middle School is 72 -/
theorem num_teachers_at_king_middle_school : 
  (num_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by sorry

end NUMINAMATH_CALUDE_num_teachers_at_king_middle_school_l4003_400353


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l4003_400385

-- Define the initial number of dimes and the amounts given by family members
def initial_dimes : ℝ := 19
def dad_gave : ℝ := 39.5
def mom_gave : ℝ := 25.25
def brother_gave : ℝ := 15.75

-- Define the total number of dimes
def total_dimes : ℝ := initial_dimes + dad_gave + mom_gave + brother_gave

-- Theorem to prove
theorem melanie_dimes_count : total_dimes = 99.5 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l4003_400385


namespace NUMINAMATH_CALUDE_expression_simplification_l4003_400349

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4003_400349


namespace NUMINAMATH_CALUDE_difference_of_squares_l4003_400352

theorem difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4003_400352


namespace NUMINAMATH_CALUDE_parabola_equation_l4003_400323

/-- A parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line that passes through the focus of a parabola and intersects it at two points -/
structure IntersectingLine (P : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola_A : A.2^2 = 2 * P.p * A.1
  h_on_parabola_B : B.2^2 = 2 * P.p * B.1
  h_through_focus : True  -- We don't need to specify this condition explicitly for the proof

/-- The theorem stating the conditions and the result to be proved -/
theorem parabola_equation (P : Parabola) (L : IntersectingLine P)
  (h_length : Real.sqrt ((L.A.1 - L.B.1)^2 + (L.A.2 - L.B.2)^2) = 8)
  (h_midpoint : (L.A.1 + L.B.1) / 2 = 2) :
  P.p = 4 ∧ ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*P.p*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l4003_400323


namespace NUMINAMATH_CALUDE_g_five_equals_one_l4003_400308

def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (∀ x : ℝ, g x = g (-x))

theorem g_five_equals_one (g : ℝ → ℝ) (h : g_property g) : g 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l4003_400308


namespace NUMINAMATH_CALUDE_fifth_subject_mark_l4003_400309

/-- Given a student's marks in four subjects and the average across five subjects,
    calculate the mark in the fifth subject. -/
theorem fifth_subject_mark (e m p c : ℕ) (avg : ℚ) (h1 : e = 90) (h2 : m = 92) (h3 : p = 85) (h4 : c = 87) (h5 : avg = 87.8) :
  ∃ (b : ℕ), (e + m + p + c + b : ℚ) / 5 = avg ∧ b = 85 := by
  sorry

#check fifth_subject_mark

end NUMINAMATH_CALUDE_fifth_subject_mark_l4003_400309


namespace NUMINAMATH_CALUDE_characterization_of_functions_l4003_400311

-- Define the property P for a function f
def satisfies_property (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n^2 + 4 * f n = (f (f n))^2

-- Define the three types of functions
def type1 (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = 1 + x

def type2 (f : ℤ → ℤ) : Prop :=
  ∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)

def type3 (f : ℤ → ℤ) : Prop :=
  f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)

-- The main theorem
theorem characterization_of_functions (f : ℤ → ℤ) :
  satisfies_property f ↔ type1 f ∨ type2 f ∨ type3 f :=
sorry

end NUMINAMATH_CALUDE_characterization_of_functions_l4003_400311


namespace NUMINAMATH_CALUDE_ghee_mixture_problem_l4003_400372

theorem ghee_mixture_problem (Q : ℝ) : 
  (0.6 * Q = Q - 0.4 * Q) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * Q = 0.2 * (Q + 10)) →  -- After adding 10 kg, vanaspati is 20%
  Q = 10 := by
sorry

end NUMINAMATH_CALUDE_ghee_mixture_problem_l4003_400372


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l4003_400324

-- Define constants
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem wheel_radius_increase 
  (D d₁ d₂ r : ℝ) 
  (h₁ : D > 0)
  (h₂ : d₁ > 0)
  (h₃ : d₂ > 0)
  (h₄ : r > 0)
  (h₅ : d₁ > d₂)
  (h₆ : D = d₁) :
  ∃ Δr : ℝ, Δr = (D * (30 * π / inches_per_mile) * inches_per_mile) / (2 * π * d₂) - r :=
by
  sorry

#check wheel_radius_increase

end NUMINAMATH_CALUDE_wheel_radius_increase_l4003_400324


namespace NUMINAMATH_CALUDE_odd_square_difference_plus_one_is_perfect_square_l4003_400319

theorem odd_square_difference_plus_one_is_perfect_square 
  (m n : ℤ) 
  (h_m_odd : Odd m) 
  (h_n_odd : Odd n) 
  (h_divides : (m^2 - n^2 + 1) ∣ (n^2 - 1)) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 :=
sorry

end NUMINAMATH_CALUDE_odd_square_difference_plus_one_is_perfect_square_l4003_400319


namespace NUMINAMATH_CALUDE_total_jogging_time_l4003_400339

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the regular jogging time per day in minutes
def regular_time : ℕ := 30

-- Define the extra time jogged on Tuesday in minutes
def extra_tuesday : ℕ := 5

-- Define the extra time jogged on Friday in minutes
def extra_friday : ℕ := 25

-- Define the total jogging time for the week in minutes
def total_time : ℕ := weekdays * regular_time + extra_tuesday + extra_friday

-- Theorem: The total jogging time for the week is equal to 3 hours
theorem total_jogging_time : total_time / 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_jogging_time_l4003_400339


namespace NUMINAMATH_CALUDE_max_belts_is_five_l4003_400389

/-- Represents the shopping problem with hats, ties, and belts. -/
structure ShoppingProblem where
  hatPrice : ℕ
  tiePrice : ℕ
  beltPrice : ℕ
  totalBudget : ℕ

/-- Represents a valid shopping solution. -/
structure ShoppingSolution where
  hats : ℕ
  ties : ℕ
  belts : ℕ

/-- Checks if a solution is valid for a given problem. -/
def isValidSolution (problem : ShoppingProblem) (solution : ShoppingSolution) : Prop :=
  solution.hats ≥ 1 ∧
  solution.ties ≥ 1 ∧
  solution.belts ≥ 1 ∧
  problem.hatPrice * solution.hats +
  problem.tiePrice * solution.ties +
  problem.beltPrice * solution.belts = problem.totalBudget

/-- The main theorem stating that the maximum number of belts is 5. -/
theorem max_belts_is_five (problem : ShoppingProblem)
    (h1 : problem.hatPrice = 3)
    (h2 : problem.tiePrice = 4)
    (h3 : problem.beltPrice = 9)
    (h4 : problem.totalBudget = 60) :
    (∀ s : ShoppingSolution, isValidSolution problem s → s.belts ≤ 5) ∧
    (∃ s : ShoppingSolution, isValidSolution problem s ∧ s.belts = 5) :=
  sorry

end NUMINAMATH_CALUDE_max_belts_is_five_l4003_400389


namespace NUMINAMATH_CALUDE_even_heads_probability_l4003_400398

def coin_flips : ℕ := 8

theorem even_heads_probability : 
  (Finset.filter (fun n => Even n) (Finset.range (coin_flips + 1))).card / 2^coin_flips = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_even_heads_probability_l4003_400398


namespace NUMINAMATH_CALUDE_perpendicular_to_third_not_implies_perpendicular_l4003_400360

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for two lines being perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

/-- Theorem stating that the perpendicularity of two lines to a third line
    does not imply their perpendicularity to each other -/
theorem perpendicular_to_third_not_implies_perpendicular :
  ∃ (l1 l2 l3 : Line3D),
    perpendicular l1 l3 ∧ perpendicular l2 l3 ∧ ¬perpendicular l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_third_not_implies_perpendicular_l4003_400360


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4003_400346

theorem solve_exponential_equation (x : ℝ) :
  3^(x - 1) = (1 : ℝ) / 9 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4003_400346


namespace NUMINAMATH_CALUDE_octagon_all_equal_l4003_400394

/-- Represents an octagon with numbers at each vertex -/
structure Octagon :=
  (vertices : Fin 8 → ℝ)

/-- Condition that each vertex number is the mean of its adjacent vertices -/
def mean_condition (o : Octagon) : Prop :=
  ∀ i : Fin 8, o.vertices i = (o.vertices (i - 1) + o.vertices (i + 1)) / 2

/-- Theorem stating that all vertex numbers must be equal -/
theorem octagon_all_equal (o : Octagon) (h : mean_condition o) : 
  ∀ i j : Fin 8, o.vertices i = o.vertices j :=
sorry

end NUMINAMATH_CALUDE_octagon_all_equal_l4003_400394


namespace NUMINAMATH_CALUDE_num_chords_is_45_num_triangles_is_120_l4003_400397

/- Define the combination function -/
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Define the number of points on the circle -/
def num_points : ℕ := 10

/- Theorem for the number of chords -/
theorem num_chords_is_45 : combination num_points 2 = 45 := by sorry

/- Theorem for the number of triangles -/
theorem num_triangles_is_120 : combination num_points 3 = 120 := by sorry

end NUMINAMATH_CALUDE_num_chords_is_45_num_triangles_is_120_l4003_400397


namespace NUMINAMATH_CALUDE_quotient_change_l4003_400327

theorem quotient_change (A B : ℝ) (h : A / B = 0.514) : 
  (10 * A) / (B / 100) = 514 := by
sorry

end NUMINAMATH_CALUDE_quotient_change_l4003_400327


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l4003_400369

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) 
  (h1 : total = 850)
  (h2 : muslim_percent = 46/100)
  (h3 : sikh_percent = 10/100)
  (h4 : other = 136) :
  (total - (muslim_percent * total).num - (sikh_percent * total).num - other) / total = 28/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l4003_400369


namespace NUMINAMATH_CALUDE_maaza_amount_l4003_400376

/-- The amount of Pepsi in liters -/
def pepsi : ℕ := 144

/-- The amount of Sprite in liters -/
def sprite : ℕ := 368

/-- The number of cans available -/
def num_cans : ℕ := 143

/-- The function to calculate the amount of Maaza given the constraints -/
def calculate_maaza (p s c : ℕ) : ℕ :=
  c * (Nat.gcd p s) - (p + s)

/-- Theorem stating that the amount of Maaza is 1776 liters -/
theorem maaza_amount : calculate_maaza pepsi sprite num_cans = 1776 := by
  sorry

end NUMINAMATH_CALUDE_maaza_amount_l4003_400376


namespace NUMINAMATH_CALUDE_vector_coplanarity_theorem_point_coplanarity_theorem_l4003_400375

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of coplanarity for vectors -/
def coplanar_vectors (a b p : Vector3D) : Prop :=
  ∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)

/-- Definition of coplanarity for points -/
def coplanar_points (M A B P : Point3D) : Prop :=
  ∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)

theorem vector_coplanarity_theorem (a b p : Vector3D) :
  (∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)) →
  coplanar_vectors a b p :=
by sorry

theorem point_coplanarity_theorem (M A B P : Point3D) :
  (∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)) →
  coplanar_points M A B P :=
by sorry

end NUMINAMATH_CALUDE_vector_coplanarity_theorem_point_coplanarity_theorem_l4003_400375


namespace NUMINAMATH_CALUDE_larger_number_proof_l4003_400364

theorem larger_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 3) (h3 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4003_400364


namespace NUMINAMATH_CALUDE_machines_in_first_group_l4003_400345

/-- The number of machines in the first group -/
def num_machines : ℕ := 8

/-- The time taken by the first group to complete a job lot (in hours) -/
def time_first_group : ℕ := 6

/-- The number of machines in the second group -/
def num_machines_second : ℕ := 12

/-- The time taken by the second group to complete a job lot (in hours) -/
def time_second_group : ℕ := 4

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / (num_machines_second * time_second_group)

theorem machines_in_first_group :
  num_machines * work_rate * time_first_group = 1 :=
sorry

end NUMINAMATH_CALUDE_machines_in_first_group_l4003_400345


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l4003_400355

theorem scientific_notation_equality : 3422000 = 3.422 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l4003_400355


namespace NUMINAMATH_CALUDE_cube_preserves_order_l4003_400350

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l4003_400350


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_379_l4003_400328

theorem sqrt_product_plus_one_equals_379 :
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_379_l4003_400328


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l4003_400331

/-- Theorem: Total shaded area of a square carpet with specific shaded squares -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  S = 12 / 4 →              -- S is 1/4 of the carpet side length
  T = S / 4 →               -- T is 1/4 of S
  S^2 + 4 * T^2 = 11.25 :=  -- Total shaded area
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l4003_400331


namespace NUMINAMATH_CALUDE_initial_puppies_count_l4003_400379

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_remaining : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies remaining -/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_remaining := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l4003_400379


namespace NUMINAMATH_CALUDE_lawn_length_is_80_l4003_400317

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  roadWidth : ℝ
  travelCost : ℝ
  totalCost : ℝ

/-- Calculates the area of the roads on the lawn -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specified conditions, the length of the lawn is 80 meters -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.roadWidth = 10)
    (h3 : l.travelCost = 2)
    (h4 : l.totalCost = 2600)
    (h5 : l.totalCost = l.travelCost * roadArea l) :
  l.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_80_l4003_400317


namespace NUMINAMATH_CALUDE_xiao_zhang_four_vcd_probability_l4003_400335

/-- Represents the number of VCD and DVD discs for each person -/
structure DiscCount where
  vcd : ℕ
  dvd : ℕ

/-- The initial disc counts for Xiao Zhang and Xiao Wang -/
def initial_counts : DiscCount × DiscCount :=
  (⟨4, 3⟩, ⟨2, 1⟩)

/-- The total number of discs -/
def total_discs : ℕ :=
  (initial_counts.1.vcd + initial_counts.1.dvd +
   initial_counts.2.vcd + initial_counts.2.dvd)

/-- Theorem stating the probability of Xiao Zhang ending up with exactly 4 VCD discs -/
theorem xiao_zhang_four_vcd_probability :
  let (zhang, wang) := initial_counts
  let p_vcd_exchange := (zhang.vcd * wang.vcd : ℚ) / ((total_discs * (total_discs - 1)) / 2 : ℚ)
  let p_dvd_exchange := (zhang.dvd * wang.dvd : ℚ) / ((total_discs * (total_discs - 1)) / 2 : ℚ)
  p_vcd_exchange + p_dvd_exchange = 11 / 21 := by
  sorry

end NUMINAMATH_CALUDE_xiao_zhang_four_vcd_probability_l4003_400335


namespace NUMINAMATH_CALUDE_line_PB_equation_l4003_400361

-- Define the points A, B, and P
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (5, 0)
def P : ℝ × ℝ := (2, 3)

-- Define the equations of lines PA and PB
def line_PA (x y : ℝ) : Prop := x - y + 1 = 0
def line_PB (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem line_PB_equation :
  (A.1 = -1 ∧ A.2 = 0) →  -- A is on x-axis
  (B.1 = 5 ∧ B.2 = 0) →   -- B is on x-axis
  P.1 = 2 →               -- x-coordinate of P is 2
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- PA = PB
  (∀ x y, line_PA x y ↔ x - y + 1 = 0) →  -- Equation of PA
  (∀ x y, line_PB x y ↔ x + y - 5 = 0) :=  -- Equation of PB
by sorry

end NUMINAMATH_CALUDE_line_PB_equation_l4003_400361


namespace NUMINAMATH_CALUDE_quadratic_integer_value_l4003_400312

theorem quadratic_integer_value (a b c : ℚ) :
  (∀ n : ℤ, ∃ m : ℤ, a * n^2 + b * n + c = m) ↔ 
  (∃ k l m : ℤ, 2 * a = k ∧ a + b = l ∧ c = m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_value_l4003_400312


namespace NUMINAMATH_CALUDE_tan_five_pi_fourth_l4003_400318

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourth_l4003_400318


namespace NUMINAMATH_CALUDE_constant_term_g_l4003_400329

-- Define polynomials f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the condition that h is the product of f and g
variable (h_def : ∀ x, h x = f x * g x)

-- Define the constant terms of f and h
variable (f_const : f 0 = 6)
variable (h_const : h 0 = -18)

-- Theorem statement
theorem constant_term_g : g 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_g_l4003_400329


namespace NUMINAMATH_CALUDE_petya_winning_strategy_exists_l4003_400315

/-- Represents a player in the coin game -/
inductive Player
| Vasya
| Petya

/-- Represents the state of the game -/
structure GameState where
  chests : Nat
  coins : Nat
  currentPlayer : Player

/-- Defines a strategy for Petya -/
def PetyaStrategy := GameState → Nat

/-- Checks if a game state is valid -/
def isValidGameState (state : GameState) : Prop :=
  state.chests > 0 ∧ state.coins ≥ state.chests

/-- Represents the initial game state -/
def initialState : GameState :=
  { chests := 1011, coins := 2022, currentPlayer := Player.Vasya }

/-- Theorem stating Petya's winning strategy exists -/
theorem petya_winning_strategy_exists :
  ∃ (strategy : PetyaStrategy),
    ∀ (game : GameState),
      isValidGameState game →
      game.coins = 2 →
      ∃ (chest : Nat),
        chest < game.chests ∧
        strategy game = chest :=
  sorry

end NUMINAMATH_CALUDE_petya_winning_strategy_exists_l4003_400315


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l4003_400371

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1/x + 1/y + 1/z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (1/a + 1/b + 1/c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l4003_400371


namespace NUMINAMATH_CALUDE_symbiotic_pair_negation_l4003_400314

theorem symbiotic_pair_negation (m n : ℚ) : 
  (m - n = m * n + 1) → (-n - (-m) = (-n) * (-m) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symbiotic_pair_negation_l4003_400314


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4003_400354

theorem simplify_polynomial (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4003_400354


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l4003_400359

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (9, 18) is 17 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 3
  let y1 : ℝ := 4
  let x2 : ℝ := 9
  let y2 : ℝ := 18
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l4003_400359


namespace NUMINAMATH_CALUDE_max_planes_15_points_l4003_400333

/-- The number of points in the space -/
def n : ℕ := 15

/-- A function to calculate the number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The maximum number of unique planes determined by n points in general position -/
def max_planes (n : ℕ) : ℕ := combination n 3

theorem max_planes_15_points :
  max_planes n = 455 :=
sorry

end NUMINAMATH_CALUDE_max_planes_15_points_l4003_400333


namespace NUMINAMATH_CALUDE_min_value_of_sum_l4003_400370

theorem min_value_of_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x) + (x*z/y) + (x*y/z) ≥ Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀^2 + y₀^2 + z₀^2 = 1 ∧
    (y₀*z₀/x₀) + (x₀*z₀/y₀) + (x₀*y₀/z₀) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l4003_400370


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l4003_400365

-- Define the concept of a line in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define perpendicularity relation between lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- The main theorem
theorem perpendicular_lines_parallel (a b c : Line) :
  Perpendicular a b → Perpendicular c b → Parallel a c := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l4003_400365


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l4003_400306

theorem unique_number_with_three_prime_factors (x n : ℕ) :
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 7))) →
  7 ∣ x →
  x = 728 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l4003_400306


namespace NUMINAMATH_CALUDE_second_term_is_half_l4003_400366

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 1 / 4
  property : a 3 * a 5 = 4 * (a 4 - 1)

/-- The second term of the geometric sequence is 1/2 -/
theorem second_term_is_half (seq : GeometricSequence) : seq.a 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_half_l4003_400366


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l4003_400381

theorem sum_of_specific_numbers : 7.52 + 12.23 = 19.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l4003_400381


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l4003_400303

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 6 bananas are worth 2 oranges -/
theorem banana_orange_equivalence (banana orange : ℚ) 
  (h : (3/4 : ℚ) * 12 * banana = 9 * orange) : 
  (1/3 : ℚ) * 6 * banana = 2 * orange := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l4003_400303


namespace NUMINAMATH_CALUDE_daily_profit_calculation_l4003_400342

theorem daily_profit_calculation (num_employees : ℕ) (employee_share : ℚ) (profit_share_percentage : ℚ) :
  num_employees = 9 →
  employee_share = 5 →
  profit_share_percentage = 9/10 →
  profit_share_percentage * ((num_employees : ℚ) * employee_share) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_daily_profit_calculation_l4003_400342


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l4003_400383

theorem cubic_equation_ratio (p q r s : ℝ) : 
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) →
  r / s = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l4003_400383


namespace NUMINAMATH_CALUDE_john_ducks_count_l4003_400374

/-- Proves that John bought 30 ducks given the problem conditions -/
theorem john_ducks_count :
  let cost_per_duck : ℕ := 10
  let weight_per_duck : ℕ := 4
  let price_per_pound : ℕ := 5
  let total_profit : ℕ := 300
  let num_ducks : ℕ := (total_profit / (weight_per_duck * price_per_pound - cost_per_duck))
  num_ducks = 30 := by sorry

end NUMINAMATH_CALUDE_john_ducks_count_l4003_400374


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l4003_400362

theorem unique_solution_power_equation :
  ∀ x y : ℕ+, 3^(x:ℕ) + 7 = 2^(y:ℕ) → x = 2 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l4003_400362


namespace NUMINAMATH_CALUDE_nicky_running_time_l4003_400307

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 400)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  ∃ (t : ℝ), t = 30 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l4003_400307


namespace NUMINAMATH_CALUDE_shipping_cost_shipping_cost_cents_shipping_cost_proof_l4003_400301

/-- Shipping cost calculation for a book -/
theorem shipping_cost (G : ℝ) : ℝ :=
  8 * ⌈G / 100⌉

/-- The shipping cost in cents for a book weighing G grams -/
theorem shipping_cost_cents (G : ℝ) : ℝ :=
  shipping_cost G

/-- Proof that the shipping cost in cents is equal to 8 * ⌈G / 100⌉ -/
theorem shipping_cost_proof (G : ℝ) : shipping_cost_cents G = 8 * ⌈G / 100⌉ := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_shipping_cost_cents_shipping_cost_proof_l4003_400301


namespace NUMINAMATH_CALUDE_correct_calculation_l4003_400336

theorem correct_calculation : -1^4 * (-1)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4003_400336


namespace NUMINAMATH_CALUDE_circle_radius_range_l4003_400378

-- Define the circle C
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define points A and B
def A : ℝ × ℝ := (6, 0)
def B : ℝ × ℝ := (0, 8)

-- Define the line segment AB
def LineSegmentAB := {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • B}

-- Define the condition for points M and N
def ExistsMN (r : ℝ) (P : ℝ × ℝ) :=
  ∃ M N : ℝ × ℝ, M ∈ Circle r ∧ N ∈ Circle r ∧ P.1 - M.1 = N.1 - M.1 ∧ P.2 - M.2 = N.2 - M.2

-- State the theorem
theorem circle_radius_range :
  ∀ r : ℝ, (∀ P ∈ LineSegmentAB, ExistsMN r P) ↔ (8/3 ≤ r ∧ r < 12/5) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_range_l4003_400378


namespace NUMINAMATH_CALUDE_race_runners_count_l4003_400393

theorem race_runners_count : ∃ n : ℕ, 
  n > 5 ∧ 
  5 * 8 + (n - 5) * 10 = 70 ∧ 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_race_runners_count_l4003_400393


namespace NUMINAMATH_CALUDE_real_roots_of_p_l4003_400347

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := x^5 - 3*x^4 - x^2 + 3*x

/-- The set of real roots of the polynomial -/
def root_set : Set ℝ := {0, 1, 3}

/-- Theorem stating that root_set contains exactly the real roots of p -/
theorem real_roots_of_p :
  ∀ x : ℝ, x ∈ root_set ↔ p x = 0 :=
sorry

end NUMINAMATH_CALUDE_real_roots_of_p_l4003_400347


namespace NUMINAMATH_CALUDE_problem_solution_l4003_400388

theorem problem_solution (a b x y : ℝ) 
  (eq1 : 2*a*x + 2*b*y = 6)
  (eq2 : 3*a*x^2 + 3*b*y^2 = 21)
  (eq3 : 4*a*x^3 + 4*b*y^3 = 64)
  (eq4 : 5*a*x^4 + 5*b*y^4 = 210) :
  6*a*x^5 + 6*b*y^5 = 5372 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4003_400388


namespace NUMINAMATH_CALUDE_solution_t_l4003_400348

theorem solution_t : ∃ t : ℝ, (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 1) ∧ t = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_t_l4003_400348


namespace NUMINAMATH_CALUDE_division_problem_l4003_400344

theorem division_problem (n : ℕ) : 
  (n / 15 = 6) ∧ (n % 15 = 5) → n = 95 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l4003_400344


namespace NUMINAMATH_CALUDE_martha_crayons_count_l4003_400356

/-- Calculate the final number of crayons Martha has after losing half and buying new ones. -/
def final_crayons (initial : ℕ) (new_set : ℕ) : ℕ :=
  initial / 2 + new_set

/-- Theorem stating that Martha's final crayon count is correct. -/
theorem martha_crayons_count : final_crayons 18 20 = 29 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayons_count_l4003_400356


namespace NUMINAMATH_CALUDE_money_left_after_game_l4003_400390

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hot_dog_cost : ℕ := 3

theorem money_left_after_game : 
  initial_amount - (ticket_cost + hot_dog_cost) = 9 := by sorry

end NUMINAMATH_CALUDE_money_left_after_game_l4003_400390


namespace NUMINAMATH_CALUDE_larger_ball_radius_l4003_400321

/-- The radius of a larger steel ball formed from the same amount of material as 12 smaller balls -/
theorem larger_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) 
  (h1 : small_radius = 2)
  (h2 : num_small_balls = 12) : 
  ∃ (large_radius : ℝ), large_radius^3 = num_small_balls * small_radius^3 :=
by sorry

end NUMINAMATH_CALUDE_larger_ball_radius_l4003_400321


namespace NUMINAMATH_CALUDE_jackson_hermit_crabs_l4003_400330

/-- Given the conditions of Jackson's souvenir collection, prove that he collected 45 hermit crabs. -/
theorem jackson_hermit_crabs :
  ∀ (hermit_crabs spiral_shells starfish : ℕ),
  spiral_shells = 3 * hermit_crabs →
  starfish = 2 * spiral_shells →
  hermit_crabs + spiral_shells + starfish = 450 →
  hermit_crabs = 45 := by
sorry

end NUMINAMATH_CALUDE_jackson_hermit_crabs_l4003_400330


namespace NUMINAMATH_CALUDE_largest_value_l4003_400395

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (a^2 + b^2) (max (2*a*b) (max a (1/2))) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l4003_400395


namespace NUMINAMATH_CALUDE_delta_airlines_discount_percentage_l4003_400316

theorem delta_airlines_discount_percentage 
  (delta_price : ℝ) 
  (united_price : ℝ) 
  (united_discount : ℝ) 
  (price_difference : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  united_discount = 0.3 →
  price_difference = 90 →
  let united_discounted_price := united_price * (1 - united_discount)
  let delta_discounted_price := united_discounted_price - price_difference
  let delta_discount_amount := delta_price - delta_discounted_price
  let delta_discount_percentage := delta_discount_amount / delta_price
  delta_discount_percentage = 0.2 := by sorry

end NUMINAMATH_CALUDE_delta_airlines_discount_percentage_l4003_400316


namespace NUMINAMATH_CALUDE_valid_numbers_l4003_400380

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a m X : ℕ),
    (a = 1 ∨ a = 2) ∧
    m > 0 ∧
    X < 10^(m-1) ∧
    n = a * 10^(m-1) + X ∧
    3 * n = 10 * X + a

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {142857, 285714, 428571, 571428, 714285, 857142} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l4003_400380


namespace NUMINAMATH_CALUDE_exponential_inequality_l4003_400368

open Real

theorem exponential_inequality (f : ℝ → ℝ) (h : ∀ x, f x = exp x) :
  (∀ a, (∀ x, f x ≥ exp 1 * x + a) ↔ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4003_400368


namespace NUMINAMATH_CALUDE_subtraction_result_l4003_400322

theorem subtraction_result (x : ℝ) (h : 96 / x = 6) : 34 - x = 18 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l4003_400322


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4003_400338

theorem gcd_power_two_minus_one :
  Nat.gcd (2^2024 - 1) (2^2015 - 1) = 2^9 - 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4003_400338


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l4003_400392

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_expression_simplification :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l4003_400392


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l4003_400305

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem: The function f does not pass through the fourth quadrant
theorem not_in_fourth_quadrant :
  ¬ ∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l4003_400305


namespace NUMINAMATH_CALUDE_carmen_fudge_delights_sales_l4003_400325

/-- Represents the number of boxes sold for each cookie type -/
structure CookieSales where
  samoas : Nat
  thin_mints : Nat
  fudge_delights : Nat
  sugar_cookies : Nat

/-- Represents the price of each cookie type -/
structure CookiePrices where
  samoas : Rat
  thin_mints : Rat
  fudge_delights : Rat
  sugar_cookies : Rat

/-- Calculates the total revenue from cookie sales -/
def total_revenue (sales : CookieSales) (prices : CookiePrices) : Rat :=
  sales.samoas * prices.samoas +
  sales.thin_mints * prices.thin_mints +
  sales.fudge_delights * prices.fudge_delights +
  sales.sugar_cookies * prices.sugar_cookies

/-- The main theorem stating that Carmen sold 1 box of fudge delights -/
theorem carmen_fudge_delights_sales
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thin_mints = 2)
  (h3 : sales.sugar_cookies = 9)
  (h4 : prices.samoas = 4)
  (h5 : prices.thin_mints = 7/2)
  (h6 : prices.fudge_delights = 5)
  (h7 : prices.sugar_cookies = 2)
  (h8 : total_revenue sales prices = 42) :
  sales.fudge_delights = 1 := by
  sorry


end NUMINAMATH_CALUDE_carmen_fudge_delights_sales_l4003_400325


namespace NUMINAMATH_CALUDE_ab_value_l4003_400326

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l4003_400326


namespace NUMINAMATH_CALUDE_complex_exponential_conjugate_l4003_400373

theorem complex_exponential_conjugate (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_conjugate_l4003_400373


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4003_400387

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 5 + 17 + 3*x + 11 + 3*x + 6) / 5 = 19 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4003_400387


namespace NUMINAMATH_CALUDE_coeff_x6_q_cubed_is_15_l4003_400391

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^4 + 5*x^2 - 4*x + 1

/-- The coefficient of x^6 in (q(x))^3 -/
def coeff_x6_q_cubed : ℝ := 15

/-- Theorem: The coefficient of x^6 in (q(x))^3 is 15 -/
theorem coeff_x6_q_cubed_is_15 : coeff_x6_q_cubed = 15 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x6_q_cubed_is_15_l4003_400391


namespace NUMINAMATH_CALUDE_factory_production_l4003_400343

theorem factory_production (x : ℝ) 
  (h1 : (2200 / x) - (2400 / (1.2 * x)) = 1) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l4003_400343


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l4003_400363

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l4003_400363


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l4003_400399

theorem triangle_side_lengths 
  (a b c : ℝ) 
  (h1 : a + b + c = 23)
  (h2 : 3 * a + b + c = 43)
  (h3 : a + b + 3 * c = 35)
  (h4 : 2 * (a + b + c) = 46) :
  a = 10 ∧ b = 7 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l4003_400399
