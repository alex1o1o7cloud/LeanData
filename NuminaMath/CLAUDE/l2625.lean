import Mathlib

namespace NUMINAMATH_CALUDE_derivative_f_l2625_262530

noncomputable def f (x : ℝ) : ℝ := (Real.sinh x) / (2 * (Real.cosh x)^2) + (1/2) * Real.arctan (Real.sinh x)

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (Real.cosh x)^3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_l2625_262530


namespace NUMINAMATH_CALUDE_abs_negative_eight_l2625_262556

theorem abs_negative_eight : |(-8 : ℤ)| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_eight_l2625_262556


namespace NUMINAMATH_CALUDE_original_number_proof_l2625_262590

theorem original_number_proof : 
  ∃ x : ℝ, (204 / x = 16) ∧ (x = 12.75) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2625_262590


namespace NUMINAMATH_CALUDE_B_power_difference_l2625_262584

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference :
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_difference_l2625_262584


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l2625_262572

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 dots out of 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The number of horizontal lines in the grid -/
def horizontalLines : ℕ := gridSize

/-- The number of vertical lines in the grid -/
def verticalLines : ℕ := gridSize

/-- The number of major diagonals in the grid -/
def majorDiagonals : ℕ := 2

/-- The total number of collinear sets of 4 dots -/
def collinearSets : ℕ := horizontalLines + verticalLines + majorDiagonals

/-- The probability of selecting four collinear dots -/
def collinearProbability : ℚ := collinearSets / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 6 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l2625_262572


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_sum_l2625_262518

theorem rectangular_box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 40)
  (h2 : B * C = 90)
  (h3 : C * A = 100) :
  A + B + C = 83/3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_sum_l2625_262518


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2625_262579

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2*x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2625_262579


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l2625_262561

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x-coordinate on the quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents the zeros of a quadratic function -/
structure QuadraticZeros where
  m : ℝ
  n : ℝ
  h_order : m > n

theorem parabola_zeros_difference (f : QuadraticFunction) (zeros : QuadraticZeros) :
  f.eval 1 = -3 →
  f.eval 3 = 9 →
  f.eval zeros.m = 0 →
  f.eval zeros.n = 0 →
  zeros.m - zeros.n = 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_zeros_difference_l2625_262561


namespace NUMINAMATH_CALUDE_expression_simplification_l2625_262513

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  ((a - 2) / (a^2 + 2*a) - (a - 1) / (a^2 + 4*a + 4)) / ((a - 4) / (a + 2)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2625_262513


namespace NUMINAMATH_CALUDE_mirasol_spending_l2625_262533

/-- Mirasol's spending problem -/
theorem mirasol_spending (initial_amount : ℕ) (coffee_cost : ℕ) (remaining_amount : ℕ) 
  (tumbler_cost : ℕ) :
  initial_amount = 50 →
  coffee_cost = 10 →
  remaining_amount = 10 →
  initial_amount = coffee_cost + tumbler_cost + remaining_amount →
  tumbler_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_spending_l2625_262533


namespace NUMINAMATH_CALUDE_malt_shop_problem_l2625_262511

/-- Represents the number of ounces of chocolate syrup used per shake -/
def syrup_per_shake : ℕ := 4

/-- Represents the number of ounces of chocolate syrup used per cone -/
def syrup_per_cone : ℕ := 6

/-- Represents the number of shakes sold -/
def shakes_sold : ℕ := 2

/-- Represents the total number of ounces of chocolate syrup used -/
def total_syrup_used : ℕ := 14

/-- Represents the number of cones sold -/
def cones_sold : ℕ := 1

theorem malt_shop_problem :
  syrup_per_shake * shakes_sold + syrup_per_cone * cones_sold = total_syrup_used :=
by sorry

end NUMINAMATH_CALUDE_malt_shop_problem_l2625_262511


namespace NUMINAMATH_CALUDE_no_valid_bracelet_arrangement_l2625_262595

/-- The number of bracelets Elizabeth has -/
def n : ℕ := 100

/-- The number of bracelets Elizabeth wears each day -/
def k : ℕ := 3

/-- Represents a valid arrangement of bracelets -/
structure BraceletArrangement where
  days : ℕ
  worn : Fin days → Finset (Fin n)
  size_correct : ∀ d, (worn d).card = k
  all_pairs_once : ∀ i j, i < j → ∃! d, i ∈ worn d ∧ j ∈ worn d

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_bracelet_arrangement : ¬ ∃ arr : BraceletArrangement, True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_bracelet_arrangement_l2625_262595


namespace NUMINAMATH_CALUDE_between_a_and_b_l2625_262520

theorem between_a_and_b (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a < b) :
  a < (|3*a + 2*b| / 5) ∧ (|3*a + 2*b| / 5) < b := by
  sorry

end NUMINAMATH_CALUDE_between_a_and_b_l2625_262520


namespace NUMINAMATH_CALUDE_min_value_of_f_l2625_262514

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2625_262514


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_cuboid_l2625_262578

theorem sphere_surface_area_of_inscribed_cuboid (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 6) (h3 : c = 3) :
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let r := d / 2
  4 * Real.pi * r^2 = 16 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_cuboid_l2625_262578


namespace NUMINAMATH_CALUDE_workers_total_earning_l2625_262555

/-- Calculates the total earning of three workers given their daily wages and work days -/
def total_earning (daily_wage_a daily_wage_b daily_wage_c : ℚ) 
  (days_a days_b days_c : ℕ) : ℚ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- The total earning of three workers with given conditions -/
theorem workers_total_earning : 
  ∃ (daily_wage_a daily_wage_b daily_wage_c : ℚ),
    -- Daily wages ratio is 3:4:5
    daily_wage_a / daily_wage_b = 3 / 4 ∧
    daily_wage_b / daily_wage_c = 4 / 5 ∧
    -- Daily wage of c is Rs. 115
    daily_wage_c = 115 ∧
    -- Total earning calculation
    total_earning daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1702 := by
  sorry

end NUMINAMATH_CALUDE_workers_total_earning_l2625_262555


namespace NUMINAMATH_CALUDE_f_properties_l2625_262550

noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2625_262550


namespace NUMINAMATH_CALUDE_ellipse_properties_l2625_262534

-- Define the ellipse (C)
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line (l) with slope 1 passing through F(1,0)
def Line (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular bisector of MN
def PerpendicularBisector (k : ℝ) (x y : ℝ) : Prop :=
  y + (3*k)/(3 + 4*k^2) = -(1/k)*(x - (4*k^2)/(3 + 4*k^2))

theorem ellipse_properties :
  -- Given conditions
  (Ellipse 2 0) →
  (Ellipse 1 0) →
  -- Prove the following
  (∀ x y, Ellipse x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ 
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2 : ℝ) = 24/7) ∧
  (∀ k y₀, k ≠ 0 →
    PerpendicularBisector k 0 y₀ →
    -Real.sqrt 3 / 12 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2625_262534


namespace NUMINAMATH_CALUDE_total_pupils_count_l2625_262508

/-- The number of girls in the school -/
def num_girls : ℕ := 232

/-- The number of boys in the school -/
def num_boys : ℕ := 253

/-- The total number of pupils in the school -/
def total_pupils : ℕ := num_girls + num_boys

/-- Theorem: The total number of pupils in the school is 485 -/
theorem total_pupils_count : total_pupils = 485 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_count_l2625_262508


namespace NUMINAMATH_CALUDE_sum_of_An_and_Bn_l2625_262551

/-- The sum of numbers in the n-th group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the second and first number in the n-th group of cubes of natural numbers -/
def B (n : ℕ) : ℕ :=
  n^3 - (n - 1)^3

/-- Theorem stating that A_n + B_n = 2n³ for any positive integer n -/
theorem sum_of_An_and_Bn (n : ℕ) : A n + B n = 2 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_An_and_Bn_l2625_262551


namespace NUMINAMATH_CALUDE_new_students_count_l2625_262586

/-- Represents the number of new students who joined the class -/
def new_students : ℕ := sorry

/-- The original average age of the class -/
def original_avg_age : ℕ := 40

/-- The average age of new students -/
def new_students_avg_age : ℕ := 32

/-- The decrease in average age after new students join -/
def avg_age_decrease : ℕ := 4

/-- The original number of students in the class -/
def original_class_size : ℕ := 18

theorem new_students_count :
  (original_class_size * original_avg_age + new_students * new_students_avg_age) / (original_class_size + new_students) = original_avg_age - avg_age_decrease ∧
  new_students = 18 :=
sorry

end NUMINAMATH_CALUDE_new_students_count_l2625_262586


namespace NUMINAMATH_CALUDE_parabola_transformation_theorem_l2625_262559

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola 180 degrees about its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

/-- Shifts a parabola horizontally -/
def shiftHorizontal (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h - shift, k := p.k }

/-- Shifts a parabola vertically -/
def shiftVertical (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + shift }

/-- Calculates the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_theorem :
  let original := Parabola.mk 1 3 4
  let transformed := shiftVertical (shiftHorizontal (rotate180 original) 5) (-4)
  sumOfZeros transformed = 16 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_theorem_l2625_262559


namespace NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l2625_262512

/-- Represents the number of gumdrops of each color in a jar --/
structure GumdropsJar where
  purple : ℕ
  orange : ℕ
  violet : ℕ
  yellow : ℕ
  white : ℕ
  green : ℕ

/-- Calculates the total number of gumdrops in the jar --/
def total_gumdrops (jar : GumdropsJar) : ℕ :=
  jar.purple + jar.orange + jar.violet + jar.yellow + jar.white + jar.green

/-- Theorem stating the number of orange gumdrops after replacement --/
theorem orange_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.white = 40 ∧
  total_gumdrops jar = 160 ∧
  jar.purple = 40 ∧
  jar.orange = 24 ∧
  jar.violet = 32 ∧
  jar.yellow = 24 →
  jar.orange + (jar.purple / 3) = 37 := by
  sorry

#check orange_gumdrops_after_replacement

end NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l2625_262512


namespace NUMINAMATH_CALUDE_fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2625_262544

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem fibonacci_m_digit_count (m : ℕ) (h : m ≥ 2) :
  ∃ k : ℕ, fib k ≥ 10^(m-1) ∧ fib (k+3) < 10^m ∧ fib (k+4) ≥ 10^m :=
sorry

theorem fibonacci_5n_plus_2_digits (n : ℕ) :
  fib (5*n + 2) ≥ 10^n :=
sorry

end NUMINAMATH_CALUDE_fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2625_262544


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2625_262525

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2625_262525


namespace NUMINAMATH_CALUDE_second_tank_fish_length_is_two_l2625_262549

/-- Represents the fish tank system with given conditions -/
structure FishTankSystem where
  first_tank_size : ℝ
  second_tank_size : ℝ
  first_tank_water : ℝ
  first_tank_fish_length : ℝ
  fish_difference_after_eating : ℕ
  (size_relation : first_tank_size = 2 * second_tank_size)
  (first_tank_water_amount : first_tank_water = 48)
  (first_tank_fish_size : first_tank_fish_length = 3)
  (fish_difference : fish_difference_after_eating = 3)

/-- The length of fish in the second tank -/
def second_tank_fish_length (system : FishTankSystem) : ℝ :=
  2

/-- Theorem stating that the length of fish in the second tank is 2 inches -/
theorem second_tank_fish_length_is_two (system : FishTankSystem) :
  second_tank_fish_length system = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_tank_fish_length_is_two_l2625_262549


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2625_262560

/-- Given quadratic polynomials f(x) = x² + bx + c and g(x) = x² + px + q with roots m₁, m₂ and k₁, k₂ respectively,
    prove that f(k₁) + f(k₂) + g(m₁) + g(m₂) ≥ 0. -/
theorem quadratic_roots_inequality (b c p q m₁ m₂ k₁ k₂ : ℝ) :
  let f := fun x => x^2 + b*x + c
  let g := fun x => x^2 + p*x + q
  (f m₁ = 0) ∧ (f m₂ = 0) ∧ (g k₁ = 0) ∧ (g k₂ = 0) →
  f k₁ + f k₂ + g m₁ + g m₂ ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2625_262560


namespace NUMINAMATH_CALUDE_max_triangles_hit_five_times_is_25_l2625_262553

/-- Represents a triangular target divided into smaller equilateral triangles -/
structure Target where
  total_triangles : Nat
  mk_valid : total_triangles = 100

/-- Represents a shot by the sniper -/
structure Shot where
  aimed_triangle : Nat
  hit_triangle : Nat
  mk_valid : hit_triangle = aimed_triangle ∨ 
             hit_triangle = aimed_triangle - 1 ∨ 
             hit_triangle = aimed_triangle + 1

/-- Represents the result of multiple shots -/
def ShotResult := Nat → Nat

/-- The maximum number of triangles that can be hit exactly five times -/
def max_triangles_hit_five_times (t : Target) (shots : List Shot) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles hit exactly five times -/
theorem max_triangles_hit_five_times_is_25 (t : Target) :
  ∃ (shots : List Shot), max_triangles_hit_five_times t shots = 25 ∧
  ∀ (other_shots : List Shot), max_triangles_hit_five_times t other_shots ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_triangles_hit_five_times_is_25_l2625_262553


namespace NUMINAMATH_CALUDE_complete_square_result_l2625_262522

theorem complete_square_result (x : ℝ) : 
  x^2 + 6*x - 4 = 0 ↔ (x + 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_result_l2625_262522


namespace NUMINAMATH_CALUDE_pascal_triangle_specific_element_l2625_262564

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle -/
def row_elements : ℕ := 56

/-- The row number (0-indexed) in Pascal's triangle -/
def row_number : ℕ := row_elements - 1

/-- The position (0-indexed) of the number we're looking for in the row -/
def position : ℕ := 23

theorem pascal_triangle_specific_element : 
  binomial row_number position = 29248649430 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_specific_element_l2625_262564


namespace NUMINAMATH_CALUDE_chuck_puppy_shot_cost_l2625_262576

/-- The total cost of shots for puppies --/
def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

/-- Theorem stating the total cost of shots for Chuck's puppies --/
theorem chuck_puppy_shot_cost :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chuck_puppy_shot_cost_l2625_262576


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2625_262537

theorem vector_sum_proof :
  let v1 : Fin 3 → ℝ := ![(-3), 2, (-1)]
  let v2 : Fin 3 → ℝ := ![1, 5, (-3)]
  v1 + v2 = ![(-2), 7, (-4)] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2625_262537


namespace NUMINAMATH_CALUDE_percentage_problem_l2625_262552

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2625_262552


namespace NUMINAMATH_CALUDE_range_of_a_l2625_262515

-- Define the functions f and g
def f (a x : ℝ) := a - x^2
def g (x : ℝ) := x + 1

-- Define the symmetry condition
def symmetric_about_x_axis (f g : ℝ → ℝ) (a : ℝ) :=
  ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x = -g x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (symmetric_about_x_axis (f a) g a) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2625_262515


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2625_262574

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h_lcm : Nat.lcm X Y = 180) (h_ratio : 5 * X = 2 * Y) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2625_262574


namespace NUMINAMATH_CALUDE_floor_difference_l2625_262516

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := sorry

/-- The number of floors in Building C -/
def floors_C : ℕ := 59

/-- The relationship between floors in Building B and C -/
axiom floors_C_relation : floors_C = 5 * floors_B - 6

/-- The difference in floors between Building A and Building B is 9 -/
theorem floor_difference : floors_B - floors_A = 9 := by sorry

end NUMINAMATH_CALUDE_floor_difference_l2625_262516


namespace NUMINAMATH_CALUDE_average_problem_l2625_262568

theorem average_problem (x : ℝ) : (1 + 3 + x) / 3 = 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2625_262568


namespace NUMINAMATH_CALUDE_expected_faces_six_rolls_l2625_262565

/-- The number of sides on a fair die -/
def n : ℕ := 6

/-- The number of times the die is rolled -/
def k : ℕ := 6

/-- The probability that a specific face does not appear in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces that appear when rolling a fair n-sided die k times -/
def expected_different_faces : ℚ := n * (1 - p^k)

/-- Theorem stating that the expected number of different faces that appear when 
    rolling a fair 6-sided die 6 times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_rolls : 
  expected_different_faces = (n^k - (n-1)^k) / n^(k-1) :=
sorry

end NUMINAMATH_CALUDE_expected_faces_six_rolls_l2625_262565


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2625_262557

theorem product_of_polynomials (g h : ℚ) : 
  (∀ x, (9*x^2 - 5*x + g) * (4*x^2 + h*x - 12) = 36*x^4 - 41*x^3 + 7*x^2 + 13*x - 72) →
  g + h = -11/6 := by sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2625_262557


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2625_262581

theorem simplify_trigonometric_expression (x : ℝ) : 
  2 * Real.sin (2 * x) * Real.sin x + Real.cos (3 * x) = Real.cos x := by
sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2625_262581


namespace NUMINAMATH_CALUDE_find_special_numbers_l2625_262542

theorem find_special_numbers : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x ≥ y ∧ 
  x = 1913 := by
  sorry

end NUMINAMATH_CALUDE_find_special_numbers_l2625_262542


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2625_262575

/-- A quadratic function f(x) = x^2 + px + qx, where p and q are positive constants -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q*x

/-- The theorem stating that the minimum of f occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ 
  ∀ (x : ℝ), f p q x_min ≤ f p q x :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2625_262575


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2625_262591

-- Problem 1
theorem problem_1 (x y : ℝ) : (2*x - 3*y)^2 - (y + 3*x)*(3*x - y) = -5*x^2 - 12*x*y + 10*y^2 := by
  sorry

-- Problem 2
theorem problem_2 : (2+1)*(2^2+1)*(2^4+1)*(2^8+1) - 2^16 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2625_262591


namespace NUMINAMATH_CALUDE_function_inequality_existence_l2625_262566

theorem function_inequality_existence (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + a * f y ≤ y + f (f x)) ↔ (a < 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_existence_l2625_262566


namespace NUMINAMATH_CALUDE_remainder_sum_l2625_262585

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 53) (hb : b % 45 = 17) : (a + b) % 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2625_262585


namespace NUMINAMATH_CALUDE_john_concert_probability_l2625_262509

theorem john_concert_probability
  (p_rain : ℝ)
  (p_john_if_rain : ℝ)
  (p_john_if_sunny : ℝ)
  (h_rain : p_rain = 0.50)
  (h_john_rain : p_john_if_rain = 0.30)
  (h_john_sunny : p_john_if_sunny = 0.90) :
  p_rain * p_john_if_rain + (1 - p_rain) * p_john_if_sunny = 0.60 :=
by sorry

end NUMINAMATH_CALUDE_john_concert_probability_l2625_262509


namespace NUMINAMATH_CALUDE_range_of_m_for_two_zeros_l2625_262594

/-- Given a function f and a real number m, g is defined as their sum -/
def g (f : ℝ → ℝ) (m : ℝ) : ℝ → ℝ := λ x ↦ f x + m

/-- The main theorem -/
theorem range_of_m_for_two_zeros (ω : ℝ) (h_ω_pos : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 2 * (Real.cos (ω * x / 2))^2) 
  (h_period : ∀ x, f (x + 2 * Real.pi / 3) = f x) :
  {m : ℝ | ∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ z₁ ∈ Set.Icc 0 (Real.pi / 3) ∧ z₂ ∈ Set.Icc 0 (Real.pi / 3) ∧ 
    g f m z₁ = 0 ∧ g f m z₂ = 0 ∧ ∀ z ∈ Set.Icc 0 (Real.pi / 3), g f m z = 0 → z = z₁ ∨ z = z₂} = 
  Set.Ioc (-3) (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_two_zeros_l2625_262594


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2625_262554

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2625_262554


namespace NUMINAMATH_CALUDE_cookie_difference_theorem_l2625_262571

def combined_difference (a b c : ℕ) : ℕ :=
  (a.max b - a.min b) + (a.max c - a.min c) + (b.max c - b.min c)

theorem cookie_difference_theorem :
  combined_difference 129 140 167 = 76 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_theorem_l2625_262571


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2625_262531

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 88 →
  E = 4 * F + 20 →
  D + E + F = 180 →
  F = 14.4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2625_262531


namespace NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l2625_262545

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∀ x : ℝ, (24 : ℝ) / x = (80 : ℝ) / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l2625_262545


namespace NUMINAMATH_CALUDE_rearrange_segments_l2625_262587

theorem rearrange_segments (a b : ℕ) : 
  ∃ (f g : Fin 1961 → Fin 1961), 
    ∀ i : Fin 1961, ∃ k : ℕ, 
      (a + f i) + (b + g i) = k + i.val ∧ 
      k + 1960 = (a + f ⟨1960, by norm_num⟩) + (b + g ⟨1960, by norm_num⟩) := by
  sorry

end NUMINAMATH_CALUDE_rearrange_segments_l2625_262587


namespace NUMINAMATH_CALUDE_andrews_sticker_fraction_l2625_262528

theorem andrews_sticker_fraction 
  (total_stickers : ℕ) 
  (andrews_fraction : ℚ) 
  (bills_fraction : ℚ) 
  (total_given : ℕ) :
  total_stickers = 100 →
  bills_fraction = 3/10 →
  total_given = 44 →
  andrews_fraction * total_stickers + 
    bills_fraction * (total_stickers - andrews_fraction * total_stickers) = total_given →
  andrews_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_andrews_sticker_fraction_l2625_262528


namespace NUMINAMATH_CALUDE_wheel_on_semicircle_diameter_l2625_262529

theorem wheel_on_semicircle_diameter (r_wheel r_semicircle : ℝ) 
  (h_wheel : r_wheel = 8)
  (h_semicircle : r_semicircle = 25) :
  let untouched_length := 2 * (r_semicircle - (r_semicircle^2 - r_wheel^2).sqrt)
  untouched_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wheel_on_semicircle_diameter_l2625_262529


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2625_262540

theorem geometric_sequence_middle_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 15 * r = b ∧ b * r = 1) → b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2625_262540


namespace NUMINAMATH_CALUDE_conceived_number_is_seven_l2625_262562

theorem conceived_number_is_seven :
  ∃! (x : ℕ+), (10 * x.val + 7 - x.val ^ 2) / 4 - x.val = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_conceived_number_is_seven_l2625_262562


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2625_262503

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2625_262503


namespace NUMINAMATH_CALUDE_election_votes_l2625_262539

theorem election_votes (votes1 votes3 : ℕ) (winning_percentage : ℚ) 
  (h1 : votes1 = 1136)
  (h2 : votes3 = 11628)
  (h3 : winning_percentage = 55371428571428574 / 100000000000000000)
  (h4 : votes3 > votes1)
  (h5 : ↑votes3 = winning_percentage * ↑(votes1 + votes3 + votes2)) :
  ∃ votes2 : ℕ, votes2 = 8236 := by sorry

end NUMINAMATH_CALUDE_election_votes_l2625_262539


namespace NUMINAMATH_CALUDE_equation_solutions_l2625_262521

/-- The set of real solutions to the equation ∛(3 - x) + √(x - 2) = 1 -/
def solution_set : Set ℝ := {2, 3, 11}

/-- The equation ∛(3 - x) + √(x - 2) = 1 -/
def equation (x : ℝ) : Prop := Real.rpow (3 - x) (1/3) + Real.sqrt (x - 2) = 1

theorem equation_solutions :
  ∀ x : ℝ, x ∈ solution_set ↔ equation x ∧ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2625_262521


namespace NUMINAMATH_CALUDE_average_price_is_86_l2625_262502

def prices : List ℝ := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

theorem average_price_is_86 : 
  (prices.sum / prices.length : ℝ) = 86 := by sorry

end NUMINAMATH_CALUDE_average_price_is_86_l2625_262502


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_neg_reals_l2625_262596

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -abs x

-- State the theorem
theorem f_monotone_increasing_on_neg_reals :
  MonotoneOn f (Set.Iic 0) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_neg_reals_l2625_262596


namespace NUMINAMATH_CALUDE_sticks_remaining_proof_l2625_262535

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := 38

/-- The number of sticks left after Will picked up some -/
def remaining_sticks : ℕ := original_sticks - picked_up_sticks

theorem sticks_remaining_proof : remaining_sticks = 61 := by
  sorry

end NUMINAMATH_CALUDE_sticks_remaining_proof_l2625_262535


namespace NUMINAMATH_CALUDE_lassis_production_l2625_262588

/-- Given a ratio of lassis to fruit units, calculate the number of lassis that can be made from a given number of fruit units -/
def calculate_lassis (ratio_lassis ratio_fruits fruits : ℕ) : ℕ :=
  (ratio_lassis * fruits) / ratio_fruits

/-- Proof that 25 fruit units produce 75 lassis given the initial ratio -/
theorem lassis_production : calculate_lassis 15 5 25 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lassis_production_l2625_262588


namespace NUMINAMATH_CALUDE_remainder_problem_l2625_262505

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2625_262505


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l2625_262517

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l2625_262517


namespace NUMINAMATH_CALUDE_art_class_problem_l2625_262592

theorem art_class_problem (total_students : ℕ) (total_kits : ℕ) (total_artworks : ℕ) 
  (h1 : total_students = 10)
  (h2 : total_kits = 20)
  (h3 : total_artworks = 35)
  (h4 : 2 * total_kits = total_students) -- 1 kit for 2 students
  (h5 : total_students % 2 = 0) -- Ensures even number of students for equal halves
  : ∃ x : ℕ, 
    x * (total_students / 2) + 4 * (total_students / 2) = total_artworks ∧ 
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_art_class_problem_l2625_262592


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l2625_262558

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry


end NUMINAMATH_CALUDE_rotten_bananas_percentage_l2625_262558


namespace NUMINAMATH_CALUDE_train_speed_problem_l2625_262526

/-- Given two trains starting from the same station, traveling along parallel tracks in the same direction,
    with one train traveling at 31 mph, and the distance between them after 8 hours being 160 miles,
    prove that the speed of the first train is 51 mph. -/
theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first train
  (v - 31) * 8 = 160 → 
  v = 51 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2625_262526


namespace NUMINAMATH_CALUDE_grocery_store_order_l2625_262519

theorem grocery_store_order (peas carrots corn : ℕ) 
  (h_peas : peas = 810) 
  (h_carrots : carrots = 954) 
  (h_corn : corn = 675) : 
  ∃ (boxes packs cases : ℕ), 
    boxes * 4 ≥ peas ∧ 
    (boxes - 1) * 4 < peas ∧ 
    packs * 6 = carrots ∧ 
    cases * 5 = corn ∧ 
    boxes = 203 ∧ 
    packs = 159 ∧ 
    cases = 135 := by
  sorry

#check grocery_store_order

end NUMINAMATH_CALUDE_grocery_store_order_l2625_262519


namespace NUMINAMATH_CALUDE_days_worked_together_is_two_l2625_262524

-- Define the efficiencies and time ratios
def efficiency_ratio_A_C : ℚ := 5 / 3
def time_ratio_B_C : ℚ := 2 / 3

-- Define the difference in days between A and C
def days_difference_A_C : ℕ := 6

-- Define the time A took to finish the remaining work
def remaining_work_days_A : ℕ := 6

-- Function to calculate the number of days B and C worked together
def days_worked_together (efficiency_ratio_A_C : ℚ) (time_ratio_B_C : ℚ) 
                         (days_difference_A_C : ℕ) (remaining_work_days_A : ℕ) : ℚ := 
  sorry

-- Theorem statement
theorem days_worked_together_is_two :
  days_worked_together efficiency_ratio_A_C time_ratio_B_C days_difference_A_C remaining_work_days_A = 2 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_together_is_two_l2625_262524


namespace NUMINAMATH_CALUDE_fourth_tree_grows_more_l2625_262563

/-- Represents the daily growth rates of four trees and their total growth over a period --/
structure TreeGrowth where
  first_tree_rate : ℝ
  second_tree_rate : ℝ
  third_tree_rate : ℝ
  fourth_tree_rate : ℝ
  total_days : ℕ
  total_growth : ℝ

/-- The growth rates satisfy the problem conditions --/
def satisfies_conditions (g : TreeGrowth) : Prop :=
  g.first_tree_rate = 1 ∧
  g.second_tree_rate = 2 * g.first_tree_rate ∧
  g.third_tree_rate = 2 ∧
  g.total_days = 4 ∧
  g.total_growth = 32 ∧
  g.first_tree_rate * g.total_days +
  g.second_tree_rate * g.total_days +
  g.third_tree_rate * g.total_days +
  g.fourth_tree_rate * g.total_days = g.total_growth

/-- The theorem stating the difference in growth rates --/
theorem fourth_tree_grows_more (g : TreeGrowth) 
  (h : satisfies_conditions g) : 
  g.fourth_tree_rate - g.third_tree_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_tree_grows_more_l2625_262563


namespace NUMINAMATH_CALUDE_function_periodicity_l2625_262506

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity
  (f : ℝ → ℝ)
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/7) + f (x + 1/6)) :
  is_periodic f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l2625_262506


namespace NUMINAMATH_CALUDE_apple_pile_count_l2625_262507

-- Define the initial number of apples
def initial_apples : ℕ := 8

-- Define the number of apples added
def added_apples : ℕ := 5

-- Theorem to prove
theorem apple_pile_count : initial_apples + added_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_apple_pile_count_l2625_262507


namespace NUMINAMATH_CALUDE_conic_section_focal_distance_l2625_262580

theorem conic_section_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∀ x y : ℝ, x^2 + a * y^2 + a^2 = 0 → 
    ∃ c : ℝ, c = 2 ∧ c^2 = a^2 - a) →
  a = (1 - Real.sqrt 17) / 2 := by
sorry

end NUMINAMATH_CALUDE_conic_section_focal_distance_l2625_262580


namespace NUMINAMATH_CALUDE_scientific_notation_of_35000000_l2625_262548

theorem scientific_notation_of_35000000 :
  (35000000 : ℝ) = 3.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35000000_l2625_262548


namespace NUMINAMATH_CALUDE_base7_to_decimal_correct_l2625_262501

/-- Converts a base 7 digit to its decimal (base 10) value -/
def base7ToDecimal (d : ℕ) : ℕ := d

/-- Represents the number 23456 in base 7 as a list of its digits -/
def base7Number : List ℕ := [2, 3, 4, 5, 6]

/-- Converts a list of base 7 digits to its decimal (base 10) equivalent -/
def convertBase7ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + (base7ToDecimal d) * 7^i) 0

theorem base7_to_decimal_correct :
  convertBase7ToDecimal base7Number = 6068 := by sorry

end NUMINAMATH_CALUDE_base7_to_decimal_correct_l2625_262501


namespace NUMINAMATH_CALUDE_nancy_home_economics_marks_l2625_262541

/-- Represents the marks obtained in different subjects -/
structure Marks where
  american_literature : ℕ
  history : ℕ
  physical_education : ℕ
  art : ℕ
  home_economics : ℕ

/-- Calculates the average marks -/
def average (m : Marks) : ℚ :=
  (m.american_literature + m.history + m.physical_education + m.art + m.home_economics) / 5

theorem nancy_home_economics_marks :
  ∀ m : Marks,
    m.american_literature = 66 →
    m.history = 75 →
    m.physical_education = 68 →
    m.art = 89 →
    average m = 70 →
    m.home_economics = 52 := by
  sorry

end NUMINAMATH_CALUDE_nancy_home_economics_marks_l2625_262541


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2625_262500

theorem base_2_representation_of_123 : 
  ∃ (b : List Bool), 
    (b.length = 7) ∧ 
    (b = [true, true, true, true, false, true, true]) ∧
    (Nat.ofDigits 2 (b.map (fun x => if x then 1 else 0)) = 123) := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2625_262500


namespace NUMINAMATH_CALUDE_janets_sandcastle_height_l2625_262567

/-- Given the heights of two sandcastles, proves that the taller one is the sum of the shorter one's height and the difference between them. -/
theorem janets_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 2.3333333333333335)
  (h2 : height_difference = 1.3333333333333333) : 
  sisters_height + height_difference = 3.6666666666666665 := by
  sorry

#check janets_sandcastle_height

end NUMINAMATH_CALUDE_janets_sandcastle_height_l2625_262567


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_ge_sum_squared_l2625_262536

theorem sum_reciprocal_squared_ge_sum_squared 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_ge_sum_squared_l2625_262536


namespace NUMINAMATH_CALUDE_min_value_on_interval_l2625_262589

def f (x a : ℝ) : ℝ := 3 * x^4 - 8 * x^3 - 18 * x^2 + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f x a = 6) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≤ 6) →
  (∃ x ∈ Set.Icc (-1) 1, f x a = -17) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≥ -17) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l2625_262589


namespace NUMINAMATH_CALUDE_price_quantity_change_cost_difference_l2625_262523

theorem price_quantity_change (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 = P * Q * 0.99 := by
sorry

theorem cost_difference (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 - P * Q = P * Q * (-0.01) := by
sorry

end NUMINAMATH_CALUDE_price_quantity_change_cost_difference_l2625_262523


namespace NUMINAMATH_CALUDE_smallest_number_with_five_primes_including_even_l2625_262597

def is_prime (n : ℕ) : Prop := sorry

def has_five_different_prime_factors (n : ℕ) : Prop := sorry

def has_even_prime_factor (n : ℕ) : Prop := sorry

theorem smallest_number_with_five_primes_including_even :
  ∀ n : ℕ, 
    has_five_different_prime_factors n ∧ 
    has_even_prime_factor n → 
    n ≥ 2310 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_five_primes_including_even_l2625_262597


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2625_262583

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal to 50°
  α = β ∧ α = 50 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2625_262583


namespace NUMINAMATH_CALUDE_car_speed_is_60_l2625_262569

/-- Represents the scenario of two friends traveling to a hunting base -/
structure HuntingTrip where
  walker_distance : ℝ  -- Distance of walker from base
  car_distance : ℝ     -- Distance of car owner from base
  total_time : ℝ       -- Total time to reach the base
  early_start : ℝ      -- Time walker would start earlier in alternative scenario
  early_meet : ℝ       -- Distance from walker's home where they'd meet in alternative scenario

/-- Calculates the speed of the car given the hunting trip scenario -/
def calculate_car_speed (trip : HuntingTrip) : ℝ :=
  60  -- Placeholder for the actual calculation

/-- Theorem stating that the car speed is 60 km/h given the specific scenario -/
theorem car_speed_is_60 (trip : HuntingTrip) 
  (h1 : trip.walker_distance = 46)
  (h2 : trip.car_distance = 30)
  (h3 : trip.total_time = 1)
  (h4 : trip.early_start = 8/3)
  (h5 : trip.early_meet = 11) :
  calculate_car_speed trip = 60 := by
  sorry

#eval calculate_car_speed { 
  walker_distance := 46, 
  car_distance := 30, 
  total_time := 1, 
  early_start := 8/3, 
  early_meet := 11 
}

end NUMINAMATH_CALUDE_car_speed_is_60_l2625_262569


namespace NUMINAMATH_CALUDE_massachusetts_avenue_pairings_l2625_262504

/-- Represents the number of possible pairings for n blocks -/
def pairings : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => pairings (n + 1) + pairings n

/-- The 10th Fibonacci number -/
def fib10 : ℕ := pairings 10

theorem massachusetts_avenue_pairings :
  fib10 = 89 :=
by sorry

end NUMINAMATH_CALUDE_massachusetts_avenue_pairings_l2625_262504


namespace NUMINAMATH_CALUDE_M_characterization_inequality_in_M_l2625_262543

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 4}

-- Theorem 1: Characterization of set M
theorem M_characterization : M = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a^2 + 2*a - 3) * (b^2 + 2*b - 3) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_M_characterization_inequality_in_M_l2625_262543


namespace NUMINAMATH_CALUDE_range_of_a_l2625_262577

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - a*t - a ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2625_262577


namespace NUMINAMATH_CALUDE_equation_solutions_l2625_262582

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, 2*(x+3)^2 = x*(x+3) ↔ x = -3 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2625_262582


namespace NUMINAMATH_CALUDE_subset_union_equality_l2625_262546

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h_nonempty : ∀ i, (A i).Nonempty) : 
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧ 
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end NUMINAMATH_CALUDE_subset_union_equality_l2625_262546


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2625_262593

theorem perfect_square_condition (m : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 4 * x^2 - m * x * y + 9 * y^2 = k^2) →
  m = 12 ∨ m = -12 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2625_262593


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2625_262532

-- Define the two lines
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

-- Define the intersection point
def intersection_point : ℚ × ℚ := (-12/7, 22/7)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2625_262532


namespace NUMINAMATH_CALUDE_cyclist_rate_problem_l2625_262573

/-- Prove that given two cyclists A and B traveling between Newton and Kingston,
    with the given conditions, the rate of cyclist A is 10 mph. -/
theorem cyclist_rate_problem (rate_A rate_B : ℝ) : 
  rate_B = rate_A + 5 →                   -- B travels 5 mph faster than A
  50 / rate_A = (50 + 10) / rate_B →      -- Time for A to travel 40 miles equals time for B to travel 60 miles
  rate_A = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_rate_problem_l2625_262573


namespace NUMINAMATH_CALUDE_condition_A_not_necessary_nor_sufficient_l2625_262510

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop := Real.sqrt (1 + Real.sin θ) = a
def condition_B (θ : Real) (a : Real) : Prop := Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem condition_A_not_necessary_nor_sufficient :
  ¬(∀ θ a, condition_B θ a → condition_A θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) := by
  sorry

end NUMINAMATH_CALUDE_condition_A_not_necessary_nor_sufficient_l2625_262510


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2625_262570

theorem abs_sum_inequality (x b : ℝ) (hb : b > 0) :
  (|x - 2| + |x + 3| < b) ↔ (b > 5) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2625_262570


namespace NUMINAMATH_CALUDE_square_area_ratio_l2625_262527

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂) : s₁^2 / s₂^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2625_262527


namespace NUMINAMATH_CALUDE_investment_ratio_l2625_262538

theorem investment_ratio (a b c : ℝ) (total_profit b_share : ℝ) :
  b = (2/3) * c →
  a = n * b →
  total_profit = 3300 →
  b_share = 600 →
  b_share / total_profit = b / (a + b + c) →
  a / b = 3 :=
sorry

end NUMINAMATH_CALUDE_investment_ratio_l2625_262538


namespace NUMINAMATH_CALUDE_kennedy_gas_consumption_l2625_262547

-- Define the problem parameters
def miles_per_gallon : ℝ := 19
def distance_to_school : ℝ := 15
def distance_to_softball : ℝ := 6
def distance_to_restaurant : ℝ := 2
def distance_to_friend : ℝ := 4
def distance_to_home : ℝ := 11

-- Define the theorem
theorem kennedy_gas_consumption :
  let total_distance := distance_to_school + distance_to_softball + distance_to_restaurant + distance_to_friend + distance_to_home
  total_distance / miles_per_gallon = 2 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_gas_consumption_l2625_262547


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l2625_262599

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l2625_262599


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l2625_262598

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  y = (1/3) * Real.sin (2 * x) →  -- Original curve equation
  x' = 2 * x →                    -- x-scaling
  y' = 3 * y →                    -- y-scaling
  y' = Real.sin x' :=             -- Transformed curve equation
by sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l2625_262598
