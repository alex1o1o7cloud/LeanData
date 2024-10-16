import Mathlib

namespace NUMINAMATH_CALUDE_salary_calculation_l3717_371752

theorem salary_calculation (salary : ℝ) : 
  salary * (1/5 + 1/10 + 3/5) + 16000 = salary → salary = 160000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l3717_371752


namespace NUMINAMATH_CALUDE_log_division_simplification_l3717_371782

theorem log_division_simplification :
  Real.log 64 / Real.log (1/64) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l3717_371782


namespace NUMINAMATH_CALUDE_third_term_is_four_l3717_371749

/-- A geometric sequence with specific terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sixth_term : a 6 = 6
  ninth_term : a 9 = 9

/-- The third term of the geometric sequence is 4 -/
theorem third_term_is_four (seq : GeometricSequence) : seq.a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_four_l3717_371749


namespace NUMINAMATH_CALUDE_right_triangle_area_l3717_371746

/-- The area of a right triangle with a 30-inch leg and a 34-inch hypotenuse is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3717_371746


namespace NUMINAMATH_CALUDE_not_always_divisible_l3717_371777

theorem not_always_divisible : ¬ ∀ n : ℕ, (5^n - 1) % (4^n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_always_divisible_l3717_371777


namespace NUMINAMATH_CALUDE_multiplier_value_l3717_371705

theorem multiplier_value (n : ℝ) (x : ℝ) (h1 : n = 1) (h2 : 3 * n - 1 = x * n) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l3717_371705


namespace NUMINAMATH_CALUDE_total_distance_l3717_371753

def road_trip (tracy_miles michelle_miles katie_miles : ℕ) : Prop :=
  tracy_miles = 2 * michelle_miles + 20 ∧
  michelle_miles = 3 * katie_miles ∧
  michelle_miles = 294

theorem total_distance (tracy_miles michelle_miles katie_miles : ℕ) 
  (h : road_trip tracy_miles michelle_miles katie_miles) : 
  tracy_miles + michelle_miles + katie_miles = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_l3717_371753


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3717_371709

/-- The solution set of the inequality 3 - 2x - x^2 < 0 -/
def solution_set : Set ℝ := {x | x < -3 ∨ x > 1}

/-- The inequality function -/
def f (x : ℝ) := 3 - 2*x - x^2

theorem inequality_solution_set :
  ∀ x : ℝ, f x < 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3717_371709


namespace NUMINAMATH_CALUDE_line_above_curve_l3717_371766

theorem line_above_curve (k : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 
    (-x - 2*k + 1) > ((1/3)*x^3 - x^2 - 4*x + 1)) → 
  k < -5/6 := by
sorry

end NUMINAMATH_CALUDE_line_above_curve_l3717_371766


namespace NUMINAMATH_CALUDE_professor_seating_count_l3717_371719

/-- The number of chairs in a row -/
def total_chairs : ℕ := 12

/-- The number of professors -/
def num_professors : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of chairs available for professors (excluding first and last) -/
def available_chairs : ℕ := total_chairs - 2

/-- The number of effective chairs after considering spacing requirements -/
def effective_chairs : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to arrange professors' seating -/
def professor_seating_arrangements : ℕ := (effective_chairs.choose num_professors) * num_professors.factorial

theorem professor_seating_count :
  professor_seating_arrangements = 1680 :=
sorry

end NUMINAMATH_CALUDE_professor_seating_count_l3717_371719


namespace NUMINAMATH_CALUDE_dreamy_vacation_probability_l3717_371738

/-- The probability of drawing a dreamy vacation note -/
def p : ℝ := 0.4

/-- The total number of people drawing notes -/
def n : ℕ := 5

/-- The number of people drawing a dreamy vacation note -/
def k : ℕ := 3

/-- The target probability -/
def target_prob : ℝ := 0.2304

/-- Theorem stating that the probability of exactly k people out of n drawing a dreamy vacation note is equal to the target probability -/
theorem dreamy_vacation_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = target_prob := by
  sorry

end NUMINAMATH_CALUDE_dreamy_vacation_probability_l3717_371738


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3717_371786

theorem chocolate_distribution (total_chocolate : ℚ) (num_packages : ℕ) (neighbor_packages : ℕ) :
  total_chocolate = 72 / 7 →
  num_packages = 6 →
  neighbor_packages = 2 →
  (total_chocolate / num_packages) * neighbor_packages = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3717_371786


namespace NUMINAMATH_CALUDE_problem_statement_l3717_371770

theorem problem_statement :
  ∀ x y : ℝ,
  x = 98 * 1.2 →
  y = (x + 35) * 0.9 →
  2 * y - 3 * x = -78.12 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3717_371770


namespace NUMINAMATH_CALUDE_visible_shaded_area_coefficient_sum_l3717_371755

/-- Represents the visible shaded area of a grid with circles on top. -/
def visibleShadedArea (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

/-- The sum of coefficients A and B in the expression A - Bπ for the visible shaded area. -/
def coefficientSum (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

theorem visible_shaded_area_coefficient_sum :
  coefficientSum 6 5 1 1 4 = 41.25 := by sorry

end NUMINAMATH_CALUDE_visible_shaded_area_coefficient_sum_l3717_371755


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3717_371703

theorem smallest_integer_with_given_remainders :
  ∀ x : ℕ,
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) →
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y) →
  x = 167 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3717_371703


namespace NUMINAMATH_CALUDE_circle_angles_theorem_l3717_371781

/-- The number of angles not greater than 120° in a circle with n points -/
def S (n : ℕ) : ℕ := sorry

/-- Binomial coefficient -/
def C (n k : ℕ) : ℕ := sorry

theorem circle_angles_theorem (n k : ℕ) (h1 : n ≥ 3) :
  (2 * C k 2 < S n ∧ S n ≤ C k 2 + C (k+1) 2 → ∃ n_min, n_min = 2*k + 1 ∧ ∀ m, m ≥ 3 ∧ S m = S n → m ≥ n_min) ∧
  (k ≥ 2 → C (k-1) 2 + C k 2 < S n ∧ S n ≤ 2 * C k 2 → ∃ n_min, n_min = 2*k ∧ ∀ m, m ≥ 3 ∧ S m = S n → m ≥ n_min) :=
sorry

end NUMINAMATH_CALUDE_circle_angles_theorem_l3717_371781


namespace NUMINAMATH_CALUDE_root_expression_value_l3717_371750

theorem root_expression_value (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2022 = 0) 
  (h₂ : x₂^2 - x₂ - 2022 = 0) : 
  x₁^3 - 2022*x₁ + x₂^2 = 4045 := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l3717_371750


namespace NUMINAMATH_CALUDE_weekly_syrup_cost_l3717_371733

/-- Calculates the weekly cost of syrup for a convenience store selling soda -/
theorem weekly_syrup_cost
  (weekly_soda_sales : ℕ)
  (gallons_per_box : ℕ)
  (cost_per_box : ℕ)
  (h_weekly_soda_sales : weekly_soda_sales = 180)
  (h_gallons_per_box : gallons_per_box = 30)
  (h_cost_per_box : cost_per_box = 40) :
  (weekly_soda_sales / gallons_per_box) * cost_per_box = 240 :=
by sorry

end NUMINAMATH_CALUDE_weekly_syrup_cost_l3717_371733


namespace NUMINAMATH_CALUDE_meal_cost_theorem_l3717_371778

-- Define variables for item costs
variable (s c p k : ℝ)

-- Define the equations from the given meals
def meal1_equation : Prop := 2 * s + 5 * c + 2 * p + 3 * k = 6.30
def meal2_equation : Prop := 3 * s + 8 * c + 2 * p + 4 * k = 8.40

-- Theorem to prove
theorem meal_cost_theorem 
  (h1 : meal1_equation s c p k)
  (h2 : meal2_equation s c p k) :
  s + c + p + k = 3.15 := by
  sorry


end NUMINAMATH_CALUDE_meal_cost_theorem_l3717_371778


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l3717_371795

/-- Calculates the interval between segments for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 72 students with a sample size of 8 is 9 -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l3717_371795


namespace NUMINAMATH_CALUDE_concatenated_evens_not_divisible_by_24_l3717_371792

def concatenated_evens : ℕ := 121416182022242628303234

theorem concatenated_evens_not_divisible_by_24 : ¬ (concatenated_evens % 24 = 0) := by
  sorry

end NUMINAMATH_CALUDE_concatenated_evens_not_divisible_by_24_l3717_371792


namespace NUMINAMATH_CALUDE_skylar_starting_donation_age_l3717_371796

/-- The age at which Skylar started donating -/
def starting_age (annual_donation : ℕ) (total_donation : ℕ) (current_age : ℕ) : ℕ :=
  current_age - (total_donation / annual_donation)

/-- Theorem stating the age at which Skylar started donating -/
theorem skylar_starting_donation_age :
  starting_age 5000 105000 33 = 12 := by
  sorry

end NUMINAMATH_CALUDE_skylar_starting_donation_age_l3717_371796


namespace NUMINAMATH_CALUDE_integral_equals_ten_implies_k_equals_one_l3717_371704

theorem integral_equals_ten_implies_k_equals_one :
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 10 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ten_implies_k_equals_one_l3717_371704


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3717_371724

theorem pure_imaginary_fraction (m : ℝ) : 
  (∃ (n : ℝ), (Complex.I * n = (2 - m * Complex.I) / (1 + Complex.I))) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3717_371724


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_317_l3717_371707

theorem multiplicative_inverse_203_mod_317 :
  ∃ x : ℕ, x < 317 ∧ (203 * x) % 317 = 1 :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_317_l3717_371707


namespace NUMINAMATH_CALUDE_line_intersects_plane_l3717_371780

theorem line_intersects_plane (α : Subspace ℝ (Fin 3 → ℝ)) 
  (a b u : Fin 3 → ℝ) 
  (ha : a ∈ α) (hb : b ∈ α)
  (ha_def : a = ![1, 1/2, 3])
  (hb_def : b = ![1/2, 1, 1])
  (hu_def : u = ![1/2, 0, 1]) :
  ∃ (t : ℝ), (t • u) ∈ α ∧ t • u ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_plane_l3717_371780


namespace NUMINAMATH_CALUDE_triangle_area_two_solutions_l3717_371743

theorem triangle_area_two_solutions (A B C : ℝ) (AB AC : ℝ) :
  B = π / 6 →  -- 30 degrees in radians
  AB = 2 * Real.sqrt 3 →
  AC = 2 →
  let area := (1 / 2) * AB * AC * Real.sin A
  area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_two_solutions_l3717_371743


namespace NUMINAMATH_CALUDE_sqrt_3x_lt_5x_iff_l3717_371797

theorem sqrt_3x_lt_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt (3 * x) < 5 * x ↔ x > 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_lt_5x_iff_l3717_371797


namespace NUMINAMATH_CALUDE_red_ball_probability_l3717_371712

theorem red_ball_probability (n : ℕ) (r : ℕ) (k : ℕ) (h1 : n = 10) (h2 : r = 3) (h3 : k = 3) :
  let total_balls := n
  let red_balls := r
  let last_children := k
  let prob_one_red := (last_children.choose 1 : ℚ) * (red_balls / total_balls) * ((total_balls - red_balls) / total_balls) ^ 2
  prob_one_red = 441 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_red_ball_probability_l3717_371712


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3717_371715

theorem cubic_equation_natural_roots (p : ℝ) :
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p ∧
    5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p ∧
    5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p) ↔
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3717_371715


namespace NUMINAMATH_CALUDE_library_book_count_l3717_371701

theorem library_book_count : ∃ (initial_books : ℕ), 
  initial_books = 1750 ∧ 
  initial_books + 140 = (27 * initial_books) / 25 := by
sorry

end NUMINAMATH_CALUDE_library_book_count_l3717_371701


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l3717_371759

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * (first_store + second_store + third_store)

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_store + second_store + third_store + fourth_store

theorem helga_shoe_shopping :
  total_shoes = 48 := by sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l3717_371759


namespace NUMINAMATH_CALUDE_cube_side_length_is_three_l3717_371799

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the total number of faces of all unit cubes after slicing -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Calculates the number of blue faces (surface area of the original cube) -/
def blueFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Theorem: If one-third of all faces are blue, then the cube's side length is 3 -/
theorem cube_side_length_is_three (c : Cube) :
  3 * blueFaces c = totalFaces c → c.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_is_three_l3717_371799


namespace NUMINAMATH_CALUDE_min_value_of_f_l3717_371723

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x ∈ Set.Icc 0 3, f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3717_371723


namespace NUMINAMATH_CALUDE_fraction_sum_l3717_371744

theorem fraction_sum : (3 : ℚ) / 8 + 9 / 12 + 5 / 6 = 47 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3717_371744


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3717_371726

theorem arithmetic_calculation : 5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3717_371726


namespace NUMINAMATH_CALUDE_average_difference_l3717_371794

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_enrollments : List ℕ := [60, 30, 20, 5, 3, 2]

def t : ℚ := (class_enrollments.sum : ℚ) / num_teachers

def s : ℚ := (class_enrollments.map (λ n => n * n)).sum / num_students

theorem average_difference : t - s = -21151/1000 := by sorry

end NUMINAMATH_CALUDE_average_difference_l3717_371794


namespace NUMINAMATH_CALUDE_min_cost_to_buy_all_items_l3717_371710

def items : ℕ := 20

-- Define the set of prices
def prices : Finset ℕ := Finset.range items.succ

-- Define the promotion
def promotion_group_size : ℕ := 5
def free_items : ℕ := items / promotion_group_size

-- Define the minimum cost function
def min_cost : ℕ := (Finset.sum prices id) - (Finset.sum (Finset.filter (λ x => x > items - free_items) prices) id)

-- The theorem to prove
theorem min_cost_to_buy_all_items : min_cost = 136 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_to_buy_all_items_l3717_371710


namespace NUMINAMATH_CALUDE_housing_boom_construction_l3717_371717

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction :
  houses_built = 574 :=
by sorry

end NUMINAMATH_CALUDE_housing_boom_construction_l3717_371717


namespace NUMINAMATH_CALUDE_xyz_equals_five_l3717_371718

theorem xyz_equals_five (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l3717_371718


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_excircle_radii_l3717_371761

theorem triangle_perimeter_from_excircle_radii (a b c : ℝ) (ra rb rc : ℝ) :
  ra = 3 ∧ rb = 10 ∧ rc = 15 →
  ra > 0 ∧ rb > 0 ∧ rc > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b + c - a) / 2 = ra ∧ (a + c - b) / 2 = rb ∧ (a + b - c) / 2 = rc →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_excircle_radii_l3717_371761


namespace NUMINAMATH_CALUDE_fraction_equality_l3717_371732

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 10 / 57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3717_371732


namespace NUMINAMATH_CALUDE_baseball_cost_value_l3717_371785

/-- The amount Mike spent on toys -/
def total_spent : ℚ := 20.52

/-- The cost of marbles -/
def marbles_cost : ℚ := 9.05

/-- The cost of the football -/
def football_cost : ℚ := 4.95

/-- The cost of the baseball -/
def baseball_cost : ℚ := total_spent - (marbles_cost + football_cost)

theorem baseball_cost_value : baseball_cost = 6.52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_value_l3717_371785


namespace NUMINAMATH_CALUDE_sandwich_meal_combinations_l3717_371730

theorem sandwich_meal_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (pair : ℕ × ℕ) => 5 * pair.1 + 7 * pair.2 = 90) 
    (Finset.product (Finset.range 19) (Finset.range 13))).card := by
  sorry

end NUMINAMATH_CALUDE_sandwich_meal_combinations_l3717_371730


namespace NUMINAMATH_CALUDE_perfect_power_sequence_exists_l3717_371702

theorem perfect_power_sequence_exists : ∃ a : ℕ+, ∀ k ∈ Set.Icc 2015 2558, 
  ∃ (b : ℕ+) (n : ℕ), n ≥ 2 ∧ (k : ℝ) * a.val = b.val ^ n :=
sorry

end NUMINAMATH_CALUDE_perfect_power_sequence_exists_l3717_371702


namespace NUMINAMATH_CALUDE_solve_for_y_l3717_371767

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3717_371767


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3717_371789

theorem arithmetic_calculation : 1325 + 180 / 60 * 3 - 225 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3717_371789


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3717_371756

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- The problem statement -/
theorem ellipse_intersection_theorem (C : Ellipse) (l₁ : Line) :
  -- The ellipse passes through (2, 1)
  (2 / C.a)^2 + (1 / C.b)^2 = 1 →
  -- The eccentricity is √3/2
  (C.a^2 - C.b^2) / C.a^2 = 3/4 →
  -- There exists a point M on x - y + 2√6 = 0 such that MPQ is equilateral
  ∃ (M : ℝ × ℝ), M.1 - M.2 + 2 * Real.sqrt 6 = 0 ∧
    -- (Condition for equilateral triangle, simplified)
    (M.1^2 + M.2^2) = 3 * ((C.a * C.b * l₁.slope / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2 + 
                           (C.a * C.b / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2) →
  -- Then l₁ is either y = 0 or y = 2x/7
  l₁.slope = 0 ∨ l₁.slope = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3717_371756


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3717_371779

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (D : ℝ) :
  A = 400 * Real.pi →
  A = Real.pi * r^2 →
  D = 2 * r →
  D = 40 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3717_371779


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3717_371758

/-- Given two quadratic equations, where the roots of one are three less than the roots of the other,
    prove that the constant term of the second equation is zero. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
  (∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
  (∀ r s x y : ℝ, 
    (2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
    (x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
    x = r - 3 ∧ y = s - 3) →
  c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3717_371758


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3717_371793

theorem smallest_absolute_value (x : ℝ) : |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3717_371793


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3717_371741

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Theorem for the minimum value of f when a = 2
theorem min_value_of_f (x : ℝ) (h : x ≥ 1) :
  f 2 x ≥ 5 :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x > 0) ↔ a > -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3717_371741


namespace NUMINAMATH_CALUDE_abc_value_l3717_371713

theorem abc_value (a b c k : ℕ+) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 29)
  (h3 : a^2 + b^2 + c^2 = k) :
  a * b * c = 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l3717_371713


namespace NUMINAMATH_CALUDE_diamond_example_l3717_371721

/-- Definition of the diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 * (x - y)^2

/-- Theorem stating that 2 ◇ (3 ◇ 4) = 5745329 -/
theorem diamond_example : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3717_371721


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3717_371748

/-- Given an ellipse with minor axis length 8 and eccentricity 3/5,
    prove that the perimeter of a triangle formed by two points where a line
    through one focus intersects the ellipse and the other focus is 20. -/
theorem ellipse_triangle_perimeter (b : ℝ) (e : ℝ) (a : ℝ) (c : ℝ) 
    (h1 : b = 4)  -- Half of the minor axis length
    (h2 : e = 3/5)  -- Eccentricity
    (h3 : e = c/a)  -- Definition of eccentricity
    (h4 : a^2 = b^2 + c^2)  -- Ellipse equation
    : 4 * a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3717_371748


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l3717_371764

/-- The difference in distance traveled between two cyclists over a given time period -/
def distance_difference (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 * time) - (rate2 * time)

/-- Theorem: The difference in distance traveled between two cyclists, one traveling at 12 miles per hour
    and the other at 10 miles per hour, over a period of 6 hours, is 12 miles. -/
theorem cyclist_distance_difference :
  distance_difference 12 10 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l3717_371764


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3717_371720

def f (a b x : ℝ) := (a * x - 1) * (x + b)

theorem solution_set_theorem (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, f a b (-2 * x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3717_371720


namespace NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l3717_371771

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℝ) (dan : ℝ) (total_after : ℝ) : ℝ :=
  sam_initial - (total_after - dan)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 46.0 16.0 52 = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l3717_371771


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3717_371751

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + k*x₁ + 1 = 3*x₁ + k) ∧ 
  (x₂^2 + k*x₂ + 1 = 3*x₂ + k) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3717_371751


namespace NUMINAMATH_CALUDE_candy_bars_problem_l3717_371736

theorem candy_bars_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℝ) / 100 * jacqueline = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_problem_l3717_371736


namespace NUMINAMATH_CALUDE_vector_BC_l3717_371772

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C.1 - A.1 = -4 → 
  C.2 - A.2 = -3 → 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
sorry


end NUMINAMATH_CALUDE_vector_BC_l3717_371772


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l3717_371708

theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => 6 / x
  f 2 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l3717_371708


namespace NUMINAMATH_CALUDE_prime_divisor_of_binomial_coefficients_l3717_371745

theorem prime_divisor_of_binomial_coefficients (p : ℕ) (n : ℕ) (h_p : Prime p) (h_n : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ Nat.choose n x) ↔ ∃ a : ℕ, a > 0 ∧ n = p^a :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_of_binomial_coefficients_l3717_371745


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3717_371725

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1/3 < x ∧ x < 2) →
  (∀ x, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3717_371725


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l3717_371773

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct ice cream flavors -/
def num_flavors : ℕ := 4

/-- Theorem: The number of permutations of 4 distinct elements is 24 -/
theorem ice_cream_permutations : permutations num_flavors = 24 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l3717_371773


namespace NUMINAMATH_CALUDE_existence_of_additive_approximation_l3717_371762

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists an additive function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 for all x ∈ ℝ. -/
theorem existence_of_additive_approximation (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
    (∀ x : ℝ, |f x - g x| ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_additive_approximation_l3717_371762


namespace NUMINAMATH_CALUDE_coyote_coins_proof_l3717_371711

/-- Represents the number of coins Coyote has after each crossing and payment -/
def coins_after_crossing (initial_coins : ℕ) (num_crossings : ℕ) : ℤ :=
  (3^num_crossings * initial_coins) - (50 * (3^num_crossings - 1) / 2)

/-- Theorem stating that Coyote ends up with 0 coins after 4 crossings if he starts with 25 coins -/
theorem coyote_coins_proof :
  coins_after_crossing 25 4 = 0 := by
  sorry

#eval coins_after_crossing 25 4

end NUMINAMATH_CALUDE_coyote_coins_proof_l3717_371711


namespace NUMINAMATH_CALUDE_five_three_bar_equals_sixteen_thirds_l3717_371763

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / 9

/-- The repeating decimal 5.3̄ -/
def five_three_bar : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 3 }

/-- Theorem: The repeating decimal 5.3̄ is equal to 16/3 -/
theorem five_three_bar_equals_sixteen_thirds :
  repeatingDecimalToRational five_three_bar = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_five_three_bar_equals_sixteen_thirds_l3717_371763


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3717_371760

theorem pasta_preference_ratio (total_students : ℕ) 
  (fettuccine_preference : ℕ) (tortellini_preference : ℕ) 
  (penne_preference : ℕ) (fusilli_preference : ℕ) : 
  total_students = 800 →
  total_students = fettuccine_preference + tortellini_preference + penne_preference + fusilli_preference →
  fettuccine_preference = 2 * tortellini_preference →
  (fettuccine_preference : ℚ) / tortellini_preference = 2 := by
sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3717_371760


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l3717_371742

theorem square_root_of_one_fourth : 
  {x : ℝ | x^2 = (1/4 : ℝ)} = {-(1/2 : ℝ), (1/2 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l3717_371742


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l3717_371757

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 5) :
  large_side^2 - small_side^2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l3717_371757


namespace NUMINAMATH_CALUDE_minimum_time_is_110_l3717_371784

/-- Represents the time taken by each teacher to examine one student -/
structure TeacherTime where
  time : ℕ

/-- Represents the problem of finding the minimum examination time -/
structure ExaminationProblem where
  teacher1 : TeacherTime
  teacher2 : TeacherTime
  totalStudents : ℕ

/-- Calculates the minimum examination time for the given problem -/
def minimumExaminationTime (problem : ExaminationProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum examination time for the given problem is 110 minutes -/
theorem minimum_time_is_110 (problem : ExaminationProblem) 
  (h1 : problem.teacher1.time = 12)
  (h2 : problem.teacher2.time = 7)
  (h3 : problem.totalStudents = 25) :
  minimumExaminationTime problem = 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_is_110_l3717_371784


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3717_371775

-- Define the line equation
def line_equation (a x : ℝ) : ℝ := a * x - 3 * a + 2

-- Theorem stating that the line passes through (3, 2) for any real number a
theorem fixed_point_theorem (a : ℝ) : line_equation a 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l3717_371775


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3717_371783

/-- Time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 190 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 31 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3717_371783


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3717_371791

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The conditions of the problem -/
structure ProblemConditions (a : ℕ → ℝ) : Prop :=
  (geom_seq : geometric_sequence a)
  (sum_cond : a 4 + a 7 = 2)
  (prod_cond : a 2 * a 9 = -8)

/-- The theorem to prove -/
theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h : ProblemConditions a) : 
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3717_371791


namespace NUMINAMATH_CALUDE_number_relations_l3717_371740

theorem number_relations :
  (∃ x : ℤ, x = -2 - 4 ∧ x = -6) ∧
  (∃ y : ℤ, y = -5 + 3 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_number_relations_l3717_371740


namespace NUMINAMATH_CALUDE_triangle_property_l3717_371774

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- S is the area of triangle ABC -/
def area (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem triangle_property (t : Triangle) (h : 4 * Real.sqrt 3 * area t = t.a^2 - (t.b - t.c)^2) :
  t.A = 2 * Real.pi / 3 ∧ 2 / 3 ≤ (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 < 1 :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_property_l3717_371774


namespace NUMINAMATH_CALUDE_gathering_handshakes_l3717_371728

/-- Represents the number of handshakes in a gathering with specific conditions -/
def handshakes_in_gathering (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) 
  (group_b_connected : ℕ) (connections : ℕ) : ℕ :=
  let group_b_isolated := group_b - group_b_connected
  let handshakes_isolated_to_a := group_b_isolated * group_a
  let handshakes_connected_to_a := group_b_connected * (group_a - connections)
  let handshakes_within_b := (group_b * (group_b - 1)) / 2
  handshakes_isolated_to_a + handshakes_connected_to_a + handshakes_within_b

theorem gathering_handshakes : 
  handshakes_in_gathering 40 30 10 3 5 = 330 :=
sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l3717_371728


namespace NUMINAMATH_CALUDE_rahul_work_time_l3717_371787

-- Define the work completion time for Rajesh
def rajesh_time : ℝ := 2

-- Define the total payment
def total_payment : ℝ := 170

-- Define Rahul's share
def rahul_share : ℝ := 68

-- Define Rahul's work completion time (to be proved)
def rahul_time : ℝ := 3

-- Theorem statement
theorem rahul_work_time :
  -- Given conditions
  (rajesh_time = 2) →
  (total_payment = 170) →
  (rahul_share = 68) →
  -- Proof goal
  (rahul_time = 3) := by
    sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_rahul_work_time_l3717_371787


namespace NUMINAMATH_CALUDE_power_simplification_l3717_371706

theorem power_simplification :
  (8^5 / 8^2) * 2^10 - 2^2 = 2^19 - 4 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l3717_371706


namespace NUMINAMATH_CALUDE_nested_cube_root_l3717_371729

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l3717_371729


namespace NUMINAMATH_CALUDE_sum_of_digit_products_2019_l3717_371737

/-- Product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Sum of products of digits for numbers from 1 to n -/
def sumOfDigitProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of products of digits for integers from 1 to 2019 is 184320 -/
theorem sum_of_digit_products_2019 : sumOfDigitProducts 2019 = 184320 := by sorry

end NUMINAMATH_CALUDE_sum_of_digit_products_2019_l3717_371737


namespace NUMINAMATH_CALUDE_f_2008_eq_zero_l3717_371790

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2008_eq_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_f2 : f 2 = 0)
  (h_periodic : ∀ x, f (x + 4) = f x + f 4) :
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2008_eq_zero_l3717_371790


namespace NUMINAMATH_CALUDE_thirty_in_base_6_l3717_371716

/-- Converts a decimal number to its base 6 representation -/
def to_base_6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number -/
def from_base_6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem thirty_in_base_6 :
  to_base_6 30 = [5, 0] ∧ from_base_6 [5, 0] = 30 :=
sorry

end NUMINAMATH_CALUDE_thirty_in_base_6_l3717_371716


namespace NUMINAMATH_CALUDE_sum_triangles_eq_sixteen_l3717_371722

/-- The triangle operation -/
def triangle (a b c : ℕ) : ℕ := a * b - c

/-- The sum of two triangle operations -/
def sum_triangles (a1 b1 c1 a2 b2 c2 : ℕ) : ℕ :=
  triangle a1 b1 c1 + triangle a2 b2 c2

/-- Theorem: The sum of the triangle operations for the given sets of numbers equals 16 -/
theorem sum_triangles_eq_sixteen :
  sum_triangles 2 4 3 3 6 7 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_triangles_eq_sixteen_l3717_371722


namespace NUMINAMATH_CALUDE_population_reaches_max_in_180_years_l3717_371768

-- Define the initial conditions
def initial_year : ℕ := 2023
def island_area : ℕ := 31500
def land_per_person : ℕ := 2
def initial_population : ℕ := 250
def doubling_period : ℕ := 30

-- Define the maximum sustainable population
def max_population : ℕ := island_area / land_per_person

-- Define the population growth function
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / doubling_period))

-- Theorem statement
theorem population_reaches_max_in_180_years :
  ∃ (years : ℕ), years = 180 ∧ 
  population_after_years years ≥ max_population ∧
  population_after_years (years - doubling_period) < max_population :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_in_180_years_l3717_371768


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l3717_371788

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Calculates the result of the operation (AAA + BA) * C -/
def calculate (A B C : Digit) : ℕ :=
  (111 * A.val + 10 * B.val + A.val) * C.val

/-- Checks if three digits are all different -/
def allDifferent (A B C : Digit) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C

theorem largest_four_digit_number :
  ∃ (A B C : Digit), allDifferent A B C ∧ 
    calculate A B C = 8624 ∧
    (∀ (X Y Z : Digit), allDifferent X Y Z → calculate X Y Z ≤ 8624) :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l3717_371788


namespace NUMINAMATH_CALUDE_school_trip_probabilities_l3717_371798

/-- Represents the setup of a school trip with students and a teacher assigned to cities. -/
structure SchoolTrip where
  numStudents : Nat
  numCities : Nat
  studentsPerCity : Nat

/-- Defines the probability of event A: student a and the teacher go to the same city. -/
def probA (trip : SchoolTrip) : ℚ :=
  1 / trip.numCities

/-- Defines the probability of event B: students a and b go to the same city. -/
def probB (trip : SchoolTrip) : ℚ :=
  1 / (trip.numStudents - 1)

/-- Defines the expected value of ξ, the total number of occurrences of events A and B. -/
def expectedXi (trip : SchoolTrip) : ℚ :=
  8 / 15

/-- Theorem stating the probabilities and expected value for the given school trip scenario. -/
theorem school_trip_probabilities (trip : SchoolTrip) :
  trip.numStudents = 6 ∧ trip.numCities = 3 ∧ trip.studentsPerCity = 2 →
  probA trip = 1/3 ∧ probB trip = 1/5 ∧ expectedXi trip = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_school_trip_probabilities_l3717_371798


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3717_371700

-- Define the parabola and hyperbola
def parabola (b : ℝ) (x y : ℝ) : Prop := x^2 = -6*b*y
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the points
def point_O : ℝ × ℝ := (0, 0)
def point_A (a b : ℝ) : ℝ × ℝ := (a, 0)

-- Define the angle equality
def angle_equality (O A B C : ℝ × ℝ) : Prop := 
  (C.2 - O.2) / (C.1 - O.1) = (C.2 - B.2) / (C.1 - B.1)

-- Main theorem
theorem hyperbola_eccentricity (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hB : parabola b (-a*Real.sqrt 13/2) (3*b/2))
  (hC : parabola b (a*Real.sqrt 13/2) (3*b/2))
  (hBC : hyperbola a b (-a*Real.sqrt 13/2) (3*b/2) ∧ 
         hyperbola a b (a*Real.sqrt 13/2) (3*b/2))
  (hAOC : angle_equality point_O (point_A a b) 
    (-a*Real.sqrt 13/2, 3*b/2) (a*Real.sqrt 13/2, 3*b/2)) :
  Real.sqrt (1 + b^2/a^2) = 4*Real.sqrt 3/3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3717_371700


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l3717_371754

/-- The cost price of a bicycle for seller A, given the selling conditions and final price. -/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l3717_371754


namespace NUMINAMATH_CALUDE_cubic_function_property_l3717_371734

/-- Given a cubic function y = ax³ + bx² + cx + d, if (2, y₁) and (-2, y₂) lie on its graph
    and y₁ - y₂ = 12, then c = 3 - 4a. -/
theorem cubic_function_property (a b c d y₁ y₂ : ℝ) :
  y₁ = 8*a + 4*b + 2*c + d →
  y₂ = -8*a + 4*b - 2*c + d →
  y₁ - y₂ = 12 →
  c = 3 - 4*a :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3717_371734


namespace NUMINAMATH_CALUDE_max_sum_constrained_l3717_371739

theorem max_sum_constrained (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_constraint : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l3717_371739


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l3717_371747

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l3717_371747


namespace NUMINAMATH_CALUDE_james_score_l3717_371776

/-- Quiz bowl scoring system at Highridge High -/
structure QuizBowl where
  pointsPerCorrect : ℕ := 2
  bonusPoints : ℕ := 4
  numRounds : ℕ := 5
  questionsPerRound : ℕ := 5

/-- Calculate the total points scored by a student in the quiz bowl -/
def calculatePoints (qb : QuizBowl) (missedQuestions : ℕ) : ℕ :=
  let totalQuestions := qb.numRounds * qb.questionsPerRound
  let correctAnswers := totalQuestions - missedQuestions
  let pointsFromCorrect := correctAnswers * qb.pointsPerCorrect
  let fullRounds := qb.numRounds - (if missedQuestions > 0 then 1 else 0)
  let bonusPointsTotal := fullRounds * qb.bonusPoints
  pointsFromCorrect + bonusPointsTotal

/-- Theorem: James scored 64 points in the quiz bowl -/
theorem james_score (qb : QuizBowl) : calculatePoints qb 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_score_l3717_371776


namespace NUMINAMATH_CALUDE_least_number_of_grapes_l3717_371735

theorem least_number_of_grapes : ∃ n : ℕ, n > 0 ∧ 
  n % 19 = 1 ∧ n % 23 = 1 ∧ n % 29 = 1 ∧ 
  ∀ m : ℕ, m > 0 → m % 19 = 1 → m % 23 = 1 → m % 29 = 1 → n ≤ m :=
by
  use 12209
  sorry

end NUMINAMATH_CALUDE_least_number_of_grapes_l3717_371735


namespace NUMINAMATH_CALUDE_inequality_proof_l3717_371769

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) :
  a^2*b + b^2*c + c^2*a < a*b^2 + b*c^2 + c*a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3717_371769


namespace NUMINAMATH_CALUDE_sector_central_angle_l3717_371714

theorem sector_central_angle (area : ℝ) (arc_length : ℝ) (h1 : area = 3 * Real.pi) (h2 : arc_length = 2 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ area = (1/2) * r^2 * (arc_length / r) ∧ arc_length / r = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3717_371714


namespace NUMINAMATH_CALUDE_four_Y_three_equals_negative_twentythree_l3717_371731

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b * 2 + b^2

-- Theorem statement
theorem four_Y_three_equals_negative_twentythree :
  Y 4 3 = -23 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_negative_twentythree_l3717_371731


namespace NUMINAMATH_CALUDE_mean_temperature_l3717_371765

def temperatures : List ℝ := [82, 84, 86, 88, 90, 92, 84, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 86.375 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3717_371765


namespace NUMINAMATH_CALUDE_solve_equation_l3717_371727

/-- Custom operation for pairs of real numbers -/
def pairOp (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_equation (x : ℝ) (h : pairOp (2 * x) 3 3 (-1) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3717_371727
