import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_inverse_solvable_problems_l2224_222418

/-- A mathematical problem that can be solved by first considering its inverse -/
structure InverseSolvableProblem where
  problem : Type
  inverse_problem : Type
  solve : inverse_problem → problem

/-- Theorem stating that there exist problems solvable by first solving their inverse -/
theorem existence_of_inverse_solvable_problems :
  ∃ (P : InverseSolvableProblem), True :=
  sorry

end NUMINAMATH_CALUDE_existence_of_inverse_solvable_problems_l2224_222418


namespace NUMINAMATH_CALUDE_eye_color_hair_color_proportions_l2224_222457

/-- Represents the population characteristics of a kingdom -/
structure Kingdom where
  total : ℕ
  blondes : ℕ
  blue_eyes : ℕ
  blonde_blue_eyes : ℕ
  blonde_blue_eyes_le_blondes : blonde_blue_eyes ≤ blondes
  blonde_blue_eyes_le_blue_eyes : blonde_blue_eyes ≤ blue_eyes
  blondes_le_total : blondes ≤ total
  blue_eyes_le_total : blue_eyes ≤ total

/-- The main theorem about eye color and hair color proportions in the kingdom -/
theorem eye_color_hair_color_proportions (k : Kingdom) :
  (k.blonde_blue_eyes : ℚ) / k.blue_eyes > (k.blondes : ℚ) / k.total →
  (k.blonde_blue_eyes : ℚ) / k.blondes > (k.blue_eyes : ℚ) / k.total :=
by
  sorry

end NUMINAMATH_CALUDE_eye_color_hair_color_proportions_l2224_222457


namespace NUMINAMATH_CALUDE_area_of_square_on_XY_l2224_222459

-- Define the triangle XYZ
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2

-- Define the theorem
theorem area_of_square_on_XY (t : RightTriangle) 
  (sum_of_squares : t.XY^2 + t.YZ^2 + t.XZ^2 = 500) : 
  t.XY^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_on_XY_l2224_222459


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2224_222426

/-- The complex number (1-i)·i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem point_in_fourth_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) * Complex.I ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2224_222426


namespace NUMINAMATH_CALUDE_transfer_equation_l2224_222484

theorem transfer_equation (x : ℤ) : 
  let initial_A : ℤ := 232
  let initial_B : ℤ := 146
  let final_A : ℤ := initial_A + x
  let final_B : ℤ := initial_B - x
  (final_A = 3 * final_B) ↔ (232 + x = 3 * (146 - x)) :=
by sorry

end NUMINAMATH_CALUDE_transfer_equation_l2224_222484


namespace NUMINAMATH_CALUDE_circle_condition_l2224_222481

/-- A circle in the xy-plane can be represented by the equation (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation of the form x^2 + y^2 + dx + ey + f = 0 -/
def general_quadratic (d e f : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + d*x + e*y + f

theorem circle_condition (m : ℝ) :
  is_circle (general_quadratic (-2) (-4) m) → m < 5 := by
  sorry


end NUMINAMATH_CALUDE_circle_condition_l2224_222481


namespace NUMINAMATH_CALUDE_alice_and_dave_weight_l2224_222491

theorem alice_and_dave_weight
  (alice_bob : ℝ)
  (bob_charlie : ℝ)
  (charlie_dave : ℝ)
  (h1 : alice_bob = 230)
  (h2 : bob_charlie = 220)
  (h3 : charlie_dave = 250) :
  ∃ (alice dave : ℝ), alice + dave = 260 :=
by
  sorry

end NUMINAMATH_CALUDE_alice_and_dave_weight_l2224_222491


namespace NUMINAMATH_CALUDE_point_position_on_line_l2224_222419

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line 
  (O A B C D P : ℝ) 
  (h_order : O ≤ A ∧ A ≤ B ∧ B ≤ C ∧ C ≤ D)
  (h_dist_OA : A - O = a)
  (h_dist_OB : B - O = b)
  (h_dist_OC : C - O = c)
  (h_dist_OD : D - O = d)
  (h_P_between : B ≤ P ∧ P ≤ C)
  (h_ratio : (P - A) / (D - P) = 2 * ((P - B) / (C - P))) :
  P - O = b + c - a :=
sorry

end NUMINAMATH_CALUDE_point_position_on_line_l2224_222419


namespace NUMINAMATH_CALUDE_afternoon_campers_l2224_222474

theorem afternoon_campers (morning_campers : ℕ) (total_campers : ℕ) 
  (h1 : morning_campers = 15) 
  (h2 : total_campers = 32) : 
  total_campers - morning_campers = 17 := by
sorry

end NUMINAMATH_CALUDE_afternoon_campers_l2224_222474


namespace NUMINAMATH_CALUDE_min_sum_squares_l2224_222407

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 7}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2224_222407


namespace NUMINAMATH_CALUDE_inverse_of_17_mod_43_l2224_222475

theorem inverse_of_17_mod_43 :
  ∃ x : ℕ, x < 43 ∧ (17 * x) % 43 = 1 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_inverse_of_17_mod_43_l2224_222475


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2224_222400

def initial_center : ℝ × ℝ := (8, -3)
def reflection_line (x y : ℝ) : Prop := y = x
def translation_vector : ℝ × ℝ := (2, -5)

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translate_point (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem circle_center_transformation :
  let reflected := reflect_point initial_center
  let final := translate_point reflected translation_vector
  final = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2224_222400


namespace NUMINAMATH_CALUDE_peach_difference_l2224_222495

/-- Proves that the difference between green and red peaches is 150 --/
theorem peach_difference : 
  let total_baskets : ℕ := 20
  let odd_red : ℕ := 12
  let odd_green : ℕ := 22
  let even_red : ℕ := 15
  let even_green : ℕ := 20
  let total_odd : ℕ := total_baskets / 2
  let total_even : ℕ := total_baskets / 2
  let total_red : ℕ := odd_red * total_odd + even_red * total_even
  let total_green : ℕ := odd_green * total_odd + even_green * total_even
  total_green - total_red = 150 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2224_222495


namespace NUMINAMATH_CALUDE_grape_price_calculation_l2224_222430

theorem grape_price_calculation (G : ℚ) : 
  (11 * G + 7 * 50 = 1428) → G = 98 := by
  sorry

end NUMINAMATH_CALUDE_grape_price_calculation_l2224_222430


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l2224_222485

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l2224_222485


namespace NUMINAMATH_CALUDE_johns_age_l2224_222471

theorem johns_age (john_age dad_age : ℕ) : 
  dad_age = john_age + 30 →
  john_age + dad_age = 80 →
  john_age = 25 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l2224_222471


namespace NUMINAMATH_CALUDE_largest_product_is_15_l2224_222464

def numbers : List ℤ := [2, -3, 4, -5]

theorem largest_product_is_15 : 
  (List.map (fun x => List.map (fun y => x * y) numbers) numbers).join.maximum? = some 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_is_15_l2224_222464


namespace NUMINAMATH_CALUDE_soda_price_proof_l2224_222402

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 18.36

theorem soda_price_proof :
  (72 * discounted_price = total_price) →
  regular_price = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_proof_l2224_222402


namespace NUMINAMATH_CALUDE_exponents_gp_iff_n_3_6_10_l2224_222453

/-- A function that returns the sequence of exponents in the prime factorization of n! --/
def exponents_of_factorial (n : ℕ) : List ℕ :=
  sorry

/-- Check if a list of natural numbers forms a geometric progression --/
def is_geometric_progression (l : List ℕ) : Prop :=
  sorry

/-- The main theorem stating that the exponents in the prime factorization of n!
    form a geometric progression if and only if n is 3, 6, or 10 --/
theorem exponents_gp_iff_n_3_6_10 (n : ℕ) :
  n ≥ 3 → (is_geometric_progression (exponents_of_factorial n) ↔ n = 3 ∨ n = 6 ∨ n = 10) :=
sorry

end NUMINAMATH_CALUDE_exponents_gp_iff_n_3_6_10_l2224_222453


namespace NUMINAMATH_CALUDE_chenny_friends_l2224_222442

/-- The number of friends Chenny has -/
def num_friends (initial_candies : ℕ) (bought_candies : ℕ) (candies_per_friend : ℕ) : ℕ :=
  (initial_candies + bought_candies) / candies_per_friend

/-- Proof that Chenny has 7 friends -/
theorem chenny_friends : num_friends 10 4 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chenny_friends_l2224_222442


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l2224_222432

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt (x^2 + y^2) + x/2 = x + y) → (min x y)/(max x y) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l2224_222432


namespace NUMINAMATH_CALUDE_equation_solution_l2224_222409

theorem equation_solution (x : ℝ) (h : x ≠ 1/3) :
  (6 * x + 1) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 1) ↔
  x = -1 + (2 * Real.sqrt 3) / 3 ∨ x = -1 - (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2224_222409


namespace NUMINAMATH_CALUDE_num_small_triangles_odd_num_small_triangles_formula_l2224_222446

/-- A triangle with interior points and connections -/
structure TriangleWithPoints where
  n : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points (including vertices) are collinear
  max_connections : Bool  -- points are connected to maximize small triangles
  no_intersections : Bool  -- resulting segments do not intersect

/-- The number of small triangles formed in a TriangleWithPoints -/
def num_small_triangles (t : TriangleWithPoints) : ℕ := 2 * t.n + 1

/-- Theorem stating that the number of small triangles is odd -/
theorem num_small_triangles_odd (t : TriangleWithPoints) : 
  Odd (num_small_triangles t) := by
  sorry

/-- Theorem stating that the number of small triangles is 2n + 1 -/
theorem num_small_triangles_formula (t : TriangleWithPoints) : 
  num_small_triangles t = 2 * t.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_num_small_triangles_odd_num_small_triangles_formula_l2224_222446


namespace NUMINAMATH_CALUDE_fraction_sum_l2224_222449

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2224_222449


namespace NUMINAMATH_CALUDE_parabola_equation_l2224_222488

/-- A parabola with vertex at the origin and focus at (2, 0) has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (p (0, 0)  -- vertex at origin
   ∧ (∀ x y, p (x, y) → (x - 2)^2 + y^2 = 4)  -- focus at (2, 0)
   ∧ (∀ x y, p (x, y) → y^2 = 4 * 2 * x)) :=  -- standard form of parabola with p = 2
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2224_222488


namespace NUMINAMATH_CALUDE_police_officer_arrangements_l2224_222404

def num_officers : ℕ := 5
def num_intersections : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = num_intersections ∧
  d.sum = num_officers ∧
  ∀ x ∈ d, 1 ≤ x ∧ x ≤ 3

def arrangements (d : List ℕ) : ℕ := sorry

def arrangements_with_AB_separate : ℕ := sorry

theorem police_officer_arrangements :
  arrangements_with_AB_separate = 114 := by sorry

end NUMINAMATH_CALUDE_police_officer_arrangements_l2224_222404


namespace NUMINAMATH_CALUDE_carlson_ate_66_candies_l2224_222461

/-- Represents the number of candies Carlson eats on each visit -/
def carlson_candies_per_visit : ℕ := 2

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the total number of candies initially -/
def total_candies : ℕ := 300

/-- Represents the number of candies eaten per week -/
def candies_per_week : ℕ := days_in_week + carlson_candies_per_visit

/-- Calculates the number of candies Carlson ate -/
def carlson_total_candies : ℕ := 
  (total_candies / candies_per_week) * carlson_candies_per_visit

theorem carlson_ate_66_candies : 
  carlson_total_candies = 66 := by
  sorry

end NUMINAMATH_CALUDE_carlson_ate_66_candies_l2224_222461


namespace NUMINAMATH_CALUDE_farm_birds_l2224_222476

theorem farm_birds (chickens ducks turkeys : ℕ) : 
  ducks = 2 * chickens →
  turkeys = 3 * ducks →
  chickens + ducks + turkeys = 1800 →
  chickens = 200 := by
sorry

end NUMINAMATH_CALUDE_farm_birds_l2224_222476


namespace NUMINAMATH_CALUDE_housing_relocation_problem_l2224_222416

/-- Represents the housing relocation problem -/
theorem housing_relocation_problem 
  (household_area : ℝ) 
  (initial_green_space_ratio : ℝ)
  (final_green_space_ratio : ℝ)
  (additional_households : ℕ)
  (min_green_space_ratio : ℝ)
  (h1 : household_area = 150)
  (h2 : initial_green_space_ratio = 0.4)
  (h3 : final_green_space_ratio = 0.15)
  (h4 : additional_households = 20)
  (h5 : min_green_space_ratio = 0.2) :
  ∃ (initial_households : ℕ) (total_area : ℝ) (withdraw_households : ℕ),
    initial_households = 48 ∧ 
    total_area = 12000 ∧ 
    withdraw_households ≥ 4 ∧
    total_area - household_area * initial_households = initial_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households) = final_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households - withdraw_households) ≥ min_green_space_ratio * total_area :=
by sorry

end NUMINAMATH_CALUDE_housing_relocation_problem_l2224_222416


namespace NUMINAMATH_CALUDE_right_triangle_area_l2224_222466

theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 12 →
  α = 30 * π / 180 →
  area = 18 * Real.sqrt 3 →
  area = (1 / 2) * h * h * Real.sin α * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2224_222466


namespace NUMINAMATH_CALUDE_smallest_total_books_l2224_222431

/-- Represents the number of books in each category -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books -/
theorem smallest_total_books :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other ∧ totalBooks other > 3000 →
      totalBooks other ≥ totalBooks books ∧
      totalBooks books = 3003 :=
sorry

end NUMINAMATH_CALUDE_smallest_total_books_l2224_222431


namespace NUMINAMATH_CALUDE_number_problem_l2224_222437

theorem number_problem (x : ℝ) : 
  ((1/5 * 1/4 * x) - (5/100 * x)) + ((1/3 * x) - (1/7 * x)) = (1/10 * x - 12) → 
  x = -132 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2224_222437


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l2224_222479

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l2224_222479


namespace NUMINAMATH_CALUDE_gcd_153_119_l2224_222440

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l2224_222440


namespace NUMINAMATH_CALUDE_sally_fries_proof_l2224_222439

/-- Calculates the final number of fries Sally has after receiving one-third of Mark's fries -/
def sallys_final_fries (sally_initial : ℕ) (mark_initial : ℕ) : ℕ :=
  sally_initial + mark_initial / 3

/-- Proves that Sally's final fry count is 26 given the initial conditions -/
theorem sally_fries_proof :
  sallys_final_fries 14 36 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sally_fries_proof_l2224_222439


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l2224_222423

def a (n : ℕ) : ℚ := (2 * n - 1) / (2 - 3 * n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-2/3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l2224_222423


namespace NUMINAMATH_CALUDE_airport_distance_proof_l2224_222412

/-- Calculates the remaining distance to a destination given the total distance,
    driving speed, and time driven. -/
def remaining_distance (total_distance speed time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem stating that given a total distance of 300 km, a driving speed of 60 km/hour,
    and a driving time of 2 hours, the remaining distance to the destination is 180 km. -/
theorem airport_distance_proof :
  remaining_distance 300 60 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_proof_l2224_222412


namespace NUMINAMATH_CALUDE_arccos_sin_one_l2224_222472

theorem arccos_sin_one : Real.arccos (Real.sin 1) = π / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_l2224_222472


namespace NUMINAMATH_CALUDE_even_function_extension_l2224_222429

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x - x^4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l2224_222429


namespace NUMINAMATH_CALUDE_partition_equality_l2224_222434

/-- The number of partitions of n into non-negative powers of 2 -/
def b (n : ℕ) : ℕ := sorry

/-- The number of partitions of n which include at least one of every power of 2 
    from 1 up to the highest power of 2 in the partition -/
def c (n : ℕ) : ℕ := sorry

/-- For any non-negative integer n, b(n+1) = 2c(n) -/
theorem partition_equality (n : ℕ) : b (n + 1) = 2 * c n := by sorry

end NUMINAMATH_CALUDE_partition_equality_l2224_222434


namespace NUMINAMATH_CALUDE_tangent_line_at_point_2_minus_6_l2224_222438

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem tangent_line_at_point_2_minus_6 :
  let P : ℝ × ℝ := (2, -6)
  let tangent_slope : ℝ := f' P.1
  let tangent_equation (x : ℝ) : ℝ := tangent_slope * (x - P.1) + P.2
  (∀ x, tangent_equation x = -3 * x) ∧ f P.1 = P.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_point_2_minus_6_l2224_222438


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l2224_222460

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/(2*y) = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/(2*w) = 1 → x + 4*y ≤ z + 4*w ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l2224_222460


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2224_222496

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 56 = 1 ∧ (13 * b) % 56 = 1 ∧ 
  (3 * a + 9 * b) % 56 = 29 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2224_222496


namespace NUMINAMATH_CALUDE_division_of_powers_l2224_222405

theorem division_of_powers (x : ℝ) : 2 * x^5 / ((-x)^3) = -2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l2224_222405


namespace NUMINAMATH_CALUDE_profit_sharing_l2224_222486

/-- Profit sharing in a partnership --/
theorem profit_sharing
  (tom_investment jerry_investment : ℝ)
  (total_profit : ℝ)
  (tom_extra : ℝ)
  (h1 : tom_investment = 700)
  (h2 : jerry_investment = 300)
  (h3 : total_profit = 3000)
  (h4 : tom_extra = 800) :
  ∃ (equal_portion : ℝ),
    equal_portion = 1000 ∧
    (equal_portion / 2 + (tom_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion)) =
    (equal_portion / 2 + (jerry_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion) + tom_extra) :=
by sorry

end NUMINAMATH_CALUDE_profit_sharing_l2224_222486


namespace NUMINAMATH_CALUDE_yellow_balloons_total_l2224_222441

/-- The total number of yellow balloons Sam and Mary have together -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (mary : ℝ) : ℝ :=
  (sam_initial - sam_given) + mary

/-- Theorem stating the total number of yellow balloons Sam and Mary have -/
theorem yellow_balloons_total :
  total_balloons 6.0 5.0 7.0 = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balloons_total_l2224_222441


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l2224_222448

theorem concert_ticket_sales : ∀ (adult_tickets : ℕ) (senior_tickets : ℕ),
  -- Total tickets sold is 120
  adult_tickets + senior_tickets + adult_tickets = 120 →
  -- Total revenue is $1100
  12 * adult_tickets + 10 * senior_tickets + 6 * adult_tickets = 1100 →
  -- The number of senior tickets is 20
  senior_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l2224_222448


namespace NUMINAMATH_CALUDE_circle_area_through_point_l2224_222443

/-- The area of a circle with center P(2, 5) passing through point Q(6, -1) is 52π. -/
theorem circle_area_through_point (P Q : ℝ × ℝ) : P = (2, 5) → Q = (6, -1) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 52 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_point_l2224_222443


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l2224_222498

theorem arithmetic_to_geometric_progression 
  (x y z : ℝ) 
  (h1 : y^2 - x*y = z^2 - y^2) : 
  z^2 = y * (2*y - x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l2224_222498


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l2224_222497

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

#check tv_show_average_episodes

end NUMINAMATH_CALUDE_tv_show_average_episodes_l2224_222497


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2224_222468

theorem quadratic_equation_properties (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃! x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  (a^2 - b^2 ≤ 4) ∧ 
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b < c ∧ x₂^2 + a*x₂ + b < c ∧ |x₁ - x₂| = 4) → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2224_222468


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2224_222414

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2224_222414


namespace NUMINAMATH_CALUDE_power_equality_l2224_222422

theorem power_equality (m : ℕ) : 16^6 = 4^m → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2224_222422


namespace NUMINAMATH_CALUDE_legs_sum_is_ten_l2224_222411

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = leg * Real.sqrt 2
  perimeter_eq : leg + leg + hypotenuse = 10 + hypotenuse

-- Theorem statement
theorem legs_sum_is_ten (t : IsoscelesRightTriangle) 
  (h : t.hypotenuse = 7.0710678118654755) : 
  t.leg + t.leg = 10 := by
  sorry

end NUMINAMATH_CALUDE_legs_sum_is_ten_l2224_222411


namespace NUMINAMATH_CALUDE_t_100_gt_t_99_l2224_222428

/-- The number of ways to place n objects with weights 1 to n on a balance with equal weight in each pan. -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T(100) is greater than T(99). -/
theorem t_100_gt_t_99 : T 100 > T 99 := by sorry

end NUMINAMATH_CALUDE_t_100_gt_t_99_l2224_222428


namespace NUMINAMATH_CALUDE_second_project_length_l2224_222480

/-- Represents a digging project with depth, length, and breadth measurements. -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project. -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

theorem second_project_length : 
  ∀ (project1 project2 : DiggingProject),
  project1.depth = 100 →
  project1.length = 25 →
  project1.breadth = 30 →
  project2.depth = 75 →
  project2.breadth = 50 →
  volume project1 = volume project2 →
  project2.length = 20 := by
  sorry

#check second_project_length

end NUMINAMATH_CALUDE_second_project_length_l2224_222480


namespace NUMINAMATH_CALUDE_hash_2_3_2_1_l2224_222451

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c*d

theorem hash_2_3_2_1 : hash 2 3 2 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_2_1_l2224_222451


namespace NUMINAMATH_CALUDE_equation_solution_l2224_222490

theorem equation_solution (x y A : ℝ) : 
  (x + y)^3 - x*y*(x + y) = (x + y) * A → A = x^2 + x*y + y^2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2224_222490


namespace NUMINAMATH_CALUDE_fitness_club_comparison_l2224_222403

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Calculates the yearly cost for a given number of months -/
def yearlyCost (club : FitnessClub) (months : ℕ) : ℕ :=
  club.monthlyCost * months

/-- Calculates the cost per visit given total cost and number of visits -/
def costPerVisit (totalCost : ℕ) (visits : ℕ) : ℚ :=
  totalCost / visits

/-- Represents the two attendance patterns -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .MoodBased => 56

theorem fitness_club_comparison (alpha beta : FitnessClub) 
    (h_alpha : alpha.monthlyCost = 999)
    (h_beta : beta.monthlyCost = 1299) :
    (∀ (pattern : AttendancePattern), 
      costPerVisit (yearlyCost alpha 12) (visitsPerYear pattern) < 
      costPerVisit (yearlyCost beta 12) (visitsPerYear pattern)) ∧
    (costPerVisit (yearlyCost alpha 12) (visitsPerYear .MoodBased) > 
     costPerVisit (yearlyCost beta 8) (visitsPerYear .MoodBased)) := by
  sorry

#check fitness_club_comparison

end NUMINAMATH_CALUDE_fitness_club_comparison_l2224_222403


namespace NUMINAMATH_CALUDE_even_function_sum_l2224_222493

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin if its endpoints are additive inverses -/
def SymmetricDomain (a : ℝ) : Prop :=
  a - 1 = -3 * a

theorem even_function_sum (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : f = fun x ↦ a * x^2 + b * x)
    (h2 : IsEven f)
    (h3 : SymmetricDomain a) : 
  a + b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l2224_222493


namespace NUMINAMATH_CALUDE_fourteen_own_all_pets_l2224_222436

/-- The number of people who own all three types of pets (cats, dogs, and rabbits) -/
def people_with_all_pets (total : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (rabbit_owners : ℕ) (two_pet_owners : ℕ) : ℕ :=
  cat_owners + dog_owners + rabbit_owners - two_pet_owners - total

/-- Theorem stating that given the conditions in the problem, 14 people own all three types of pets -/
theorem fourteen_own_all_pets :
  people_with_all_pets 60 30 40 16 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_own_all_pets_l2224_222436


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l2224_222424

/-- Represents the modified cube structure --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  cornerSize : Nat

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (c : ModifiedCube) : Nat :=
  let remainingCubes := 27 - c.removedCubes
  let visibleCornersPerCube := 4
  let surfaceUnitsPerCorner := 3
  remainingCubes * visibleCornersPerCube * surfaceUnitsPerCorner

/-- The theorem to be proved --/
theorem modified_cube_surface_area :
  ∀ (c : ModifiedCube),
  c.initialSize = 6 ∧
  c.smallCubeSize = 2 ∧
  c.removedCubes = 7 ∧
  c.cornerSize = 1 →
  surfaceArea c = 240 := by
  sorry


end NUMINAMATH_CALUDE_modified_cube_surface_area_l2224_222424


namespace NUMINAMATH_CALUDE_total_cost_per_pineapple_l2224_222421

/-- The cost per pineapple including shipping -/
def cost_per_pineapple (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) : ℚ :=
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples

/-- Theorem: The total cost per pineapple is $3.00 -/
theorem total_cost_per_pineapple :
  cost_per_pineapple (25/20) 12 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_per_pineapple_l2224_222421


namespace NUMINAMATH_CALUDE_sphere_radius_l2224_222489

theorem sphere_radius (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ (r : ℝ), A = 4 * Real.pi * r^2 ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l2224_222489


namespace NUMINAMATH_CALUDE_expected_remainder_mod_64_l2224_222435

/-- The expected value of (a + 2b + 4c + 8d + 16e + 32f) mod 64, where a, b, c, d, e, f 
    are independently and uniformly randomly selected integers from {1,2,...,100} -/
theorem expected_remainder_mod_64 : 
  let S := Finset.range 100
  let M (a b c d e f : ℕ) := a + 2*b + 4*c + 8*d + 16*e + 32*f
  (S.sum (λ a => S.sum (λ b => S.sum (λ c => S.sum (λ d => S.sum (λ e => 
    S.sum (λ f => (M a b c d e f) % 64))))))) / S.card^6 = 63/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_remainder_mod_64_l2224_222435


namespace NUMINAMATH_CALUDE_average_of_multiples_of_four_is_even_l2224_222420

theorem average_of_multiples_of_four_is_even (m n : ℤ) : 
  ∃ k : ℤ, (4*m + 4*n) / 2 = 2*k := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_four_is_even_l2224_222420


namespace NUMINAMATH_CALUDE_mac_loss_is_three_dollars_l2224_222427

-- Define the values of coins in cents
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

-- Define the number of coins in each trade
def dimes_per_trade : ℕ := 3
def nickels_per_trade : ℕ := 7

-- Define the number of trades
def dime_trades : ℕ := 20
def nickel_trades : ℕ := 20

-- Calculate the loss per trade in cents
def dime_trade_loss : ℕ := dimes_per_trade * dime_value - quarter_value
def nickel_trade_loss : ℕ := nickels_per_trade * nickel_value - quarter_value

-- Calculate the total loss in cents
def total_loss_cents : ℕ := dime_trade_loss * dime_trades + nickel_trade_loss * nickel_trades

-- Convert cents to dollars
def cents_to_dollars (cents : ℕ) : ℚ := (cents : ℚ) / 100

-- Theorem: Mac's total loss is $3.00
theorem mac_loss_is_three_dollars :
  cents_to_dollars total_loss_cents = 3 := by
  sorry

end NUMINAMATH_CALUDE_mac_loss_is_three_dollars_l2224_222427


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l2224_222444

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + 2*b + c = 0) :
  ∃ (x : ℝ), -3 < x ∧ x < -1/3 ∧ x = c/a :=
sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l2224_222444


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2224_222483

/-- The binary to decimal conversion for 10010000 -/
def binary_to_decimal_1 : Nat := 144

/-- The binary to decimal conversion for 100100000 -/
def binary_to_decimal_2 : Nat := 288

/-- The function to check if a number is prime -/
def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that 3 is the largest prime factor of both numbers -/
theorem largest_prime_factor :
  ∃ (p : Nat), is_prime p ∧ 
    p ∣ binary_to_decimal_1 ∧ 
    p ∣ binary_to_decimal_2 ∧
    ∀ (q : Nat), is_prime q → 
      q ∣ binary_to_decimal_1 → 
      q ∣ binary_to_decimal_2 → 
      q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2224_222483


namespace NUMINAMATH_CALUDE_fundraising_total_l2224_222467

def total_donations (initial_donors : ℕ) (initial_average : ℕ) (days : ℕ) : ℕ :=
  let donor_counts := List.range days |>.map (fun i => initial_donors * 2^i)
  let daily_averages := List.range days |>.map (fun i => initial_average + 5 * i)
  (List.zip donor_counts daily_averages).map (fun (d, a) => d * a) |>.sum

theorem fundraising_total :
  total_donations 10 10 5 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_total_l2224_222467


namespace NUMINAMATH_CALUDE_well_diameter_proof_l2224_222458

/-- The volume of a circular well -/
def well_volume : ℝ := 175.92918860102841

/-- The depth of the circular well -/
def well_depth : ℝ := 14

/-- The diameter of the circular well -/
def well_diameter : ℝ := 4

theorem well_diameter_proof :
  well_diameter = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_well_diameter_proof_l2224_222458


namespace NUMINAMATH_CALUDE_james_restaurant_revenue_l2224_222455

theorem james_restaurant_revenue :
  -- Define the constants
  let beef_amount : ℝ := 20
  let pork_amount : ℝ := beef_amount / 2
  let meat_per_meal : ℝ := 1.5
  let price_per_meal : ℝ := 20

  -- Calculate total meat
  let total_meat : ℝ := beef_amount + pork_amount

  -- Calculate number of meals
  let number_of_meals : ℝ := total_meat / meat_per_meal

  -- Calculate total revenue
  let total_revenue : ℝ := number_of_meals * price_per_meal

  -- Prove that the total revenue is $400
  total_revenue = 400 := by sorry

end NUMINAMATH_CALUDE_james_restaurant_revenue_l2224_222455


namespace NUMINAMATH_CALUDE_triangle_formation_l2224_222410

/-- Triangle Inequality Theorem: The sum of any two sides of a triangle must be greater than the third side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if a set of three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  can_form_triangle 13 12 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l2224_222410


namespace NUMINAMATH_CALUDE_smallest_number_l2224_222452

theorem smallest_number (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5) :
  min (min a b) c > 0.1 ∧ min (min a b) c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2224_222452


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2224_222456

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = (1 - I)) : 
  Complex.abs z = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2224_222456


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l2224_222413

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l2224_222413


namespace NUMINAMATH_CALUDE_valid_selections_count_l2224_222454

/-- The number of male intern teachers --/
def male_teachers : ℕ := 5

/-- The number of female intern teachers --/
def female_teachers : ℕ := 4

/-- The total number of intern teachers --/
def total_teachers : ℕ := male_teachers + female_teachers

/-- The number of teachers to be selected --/
def selected_teachers : ℕ := 3

/-- The number of ways to select 3 teachers from the total pool --/
def total_selections : ℕ := Nat.descFactorial total_teachers selected_teachers

/-- The number of ways to select 3 male teachers --/
def all_male_selections : ℕ := Nat.descFactorial male_teachers selected_teachers

/-- The number of ways to select 3 female teachers --/
def all_female_selections : ℕ := Nat.descFactorial female_teachers selected_teachers

/-- The number of valid selection schemes --/
def valid_selections : ℕ := total_selections - (all_male_selections + all_female_selections)

theorem valid_selections_count : valid_selections = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l2224_222454


namespace NUMINAMATH_CALUDE_floss_leftover_result_l2224_222463

/-- Calculates the amount of floss left over after distributing to students --/
def floss_leftover (class1_size class2_size class3_size : ℕ) 
                   (floss_per_student1 floss_per_student2 floss_per_student3 : ℚ) 
                   (yards_per_packet : ℚ) : ℚ :=
  let total_floss_needed := class1_size * floss_per_student1 + 
                            class2_size * floss_per_student2 + 
                            class3_size * floss_per_student3
  let packets_needed := (total_floss_needed / yards_per_packet).ceil
  let total_floss_bought := packets_needed * yards_per_packet
  total_floss_bought - total_floss_needed

/-- Theorem stating the amount of floss left over --/
theorem floss_leftover_result : 
  floss_leftover 20 25 30 (3/2) (7/4) 2 35 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_result_l2224_222463


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l2224_222470

theorem distinct_prime_factors_of_30_factorial :
  (∀ p : ℕ, p.Prime → p ≤ 30 → p ∣ Nat.factorial 30) ∧
  (∃ S : Finset ℕ, (∀ p ∈ S, p.Prime ∧ p ≤ 30) ∧ 
                   (∀ p : ℕ, p.Prime → p ≤ 30 → p ∈ S) ∧ 
                   S.card = 10) :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l2224_222470


namespace NUMINAMATH_CALUDE_hospital_staff_count_l2224_222447

theorem hospital_staff_count (total : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (h1 : total = 250) 
  (h2 : ratio_doctors = 2) 
  (h3 : ratio_nurses = 3) : 
  (ratio_nurses * total) / (ratio_doctors + ratio_nurses) = 150 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l2224_222447


namespace NUMINAMATH_CALUDE_f_min_value_l2224_222450

/-- The function f(x) = x^2 + 26x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 26*x + 7

/-- The minimum value of f(x) is -162 -/
theorem f_min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = -162 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l2224_222450


namespace NUMINAMATH_CALUDE_sequence_progression_l2224_222487

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem sequence_progression (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_progression_l2224_222487


namespace NUMINAMATH_CALUDE_octagon_area_is_225_l2224_222478

-- Define the triangle and circle
structure Triangle :=
  (P Q R : ℝ × ℝ)

def circumradius : ℝ := 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := 45

-- Define the points P', Q', R' as intersections of perpendicular bisectors with circumcircle
def P' (t : Triangle) : ℝ × ℝ := sorry
def Q' (t : Triangle) : ℝ × ℝ := sorry
def R' (t : Triangle) : ℝ × ℝ := sorry

-- Define S as reflection of circumcenter over PQ
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of the octagon
def octagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem octagon_area_is_225 (t : Triangle) :
  octagon_area t = 225 := by sorry

end NUMINAMATH_CALUDE_octagon_area_is_225_l2224_222478


namespace NUMINAMATH_CALUDE_circular_path_diameter_increase_l2224_222499

theorem circular_path_diameter_increase 
  (original_rounds : ℕ) 
  (original_time : ℝ) 
  (new_time : ℝ) 
  (original_rounds_pos : original_rounds > 0)
  (original_time_pos : original_time > 0)
  (new_time_pos : new_time > 0)
  (h_original : original_rounds = 8)
  (h_original_time : original_time = 40)
  (h_new_time : new_time = 50) :
  let original_single_round_time := original_time / original_rounds
  let diameter_increase_factor := new_time / original_single_round_time
  diameter_increase_factor = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_path_diameter_increase_l2224_222499


namespace NUMINAMATH_CALUDE_sum_of_integers_l2224_222425

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + 2*c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) : 
  a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2224_222425


namespace NUMINAMATH_CALUDE_bicolored_angles_bound_l2224_222473

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A coloring of segments -/
def Coloring (n : ℕ) := Fin (n + 1) → Fin (n + 1) → Fin n

/-- The number of bicolored angles for a given coloring -/
def bicoloredAngles (n k : ℕ) (points : Fin (n + 1) → Point) (coloring : Coloring k) : ℕ :=
  sorry

/-- Three points are collinear if they lie on the same line -/
def collinear (p q r : Point) : Prop :=
  sorry

theorem bicolored_angles_bound (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 3) 
  (points : Fin (n + 1) → Point) 
  (h3 : ∀ (i j l : Fin (n + 1)), i ≠ j → j ≠ l → i ≠ l → ¬collinear (points i) (points j) (points l)) :
  ∃ (coloring : Coloring k), bicoloredAngles n k points coloring > n * (n / k)^2 * (k.choose 2) :=
sorry

end NUMINAMATH_CALUDE_bicolored_angles_bound_l2224_222473


namespace NUMINAMATH_CALUDE_spherical_coordinate_reflection_l2224_222401

/-- Given a point with rectangular coordinates (3, 8, -6) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ, -φ) has rectangular coordinates (-3, -8, -6). -/
theorem spherical_coordinate_reflection (ρ θ φ : ℝ) : 
  (ρ * Real.sin φ * Real.cos θ = 3 ∧ 
   ρ * Real.sin φ * Real.sin θ = 8 ∧ 
   ρ * Real.cos φ = -6) → 
  (ρ * Real.sin (-φ) * Real.cos θ = -3 ∧ 
   ρ * Real.sin (-φ) * Real.sin θ = -8 ∧ 
   ρ * Real.cos (-φ) = -6) := by
  sorry

end NUMINAMATH_CALUDE_spherical_coordinate_reflection_l2224_222401


namespace NUMINAMATH_CALUDE_pat_final_sticker_count_l2224_222494

/-- Calculates the final number of stickers Pat has at the end of the week -/
def final_sticker_count (initial : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday_out : ℕ) (thursday_in : ℕ) (friday : ℕ) : ℕ :=
  initial + monday - tuesday + wednesday - thursday_out + thursday_in + friday

/-- Theorem stating that Pat ends up with 43 stickers at the end of the week -/
theorem pat_final_sticker_count :
  final_sticker_count 39 15 22 10 12 8 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_pat_final_sticker_count_l2224_222494


namespace NUMINAMATH_CALUDE_equation_solutions_l2224_222445

theorem equation_solutions (x : ℝ) : 
  x ≠ -2 → 
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48) ↔ 
  (x = 1.2 ∨ x = -81.2) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2224_222445


namespace NUMINAMATH_CALUDE_rectangles_combinable_l2224_222477

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a square divided into four rectangles -/
structure DividedSquare where
  side : ℝ
  r1 : Rectangle
  r2 : Rectangle
  r3 : Rectangle
  r4 : Rectangle

/-- Assumption that the sum of areas of two non-adjacent rectangles equals the sum of areas of the other two -/
def equal_area_pairs (s : DividedSquare) : Prop :=
  area s.r1 + area s.r3 = area s.r2 + area s.r4

/-- The theorem to be proved -/
theorem rectangles_combinable (s : DividedSquare) (h : equal_area_pairs s) :
  (s.r1.width = s.r3.width ∨ s.r1.height = s.r3.height) :=
sorry

end NUMINAMATH_CALUDE_rectangles_combinable_l2224_222477


namespace NUMINAMATH_CALUDE_initial_candies_are_52_or_56_l2224_222492

def initial_candies : Set ℕ :=
  {x : ℕ | 
    -- The number of candies after Tracy ate 1/4
    ∃ (a : ℕ), 3 * x = 4 * a ∧
    -- The number of candies after giving 1/3 to Sam
    ∃ (b : ℕ), 2 * a = 3 * b ∧
    -- The number of candies after Tracy and her dad ate 20
    b ≥ 20 ∧
    -- The number of candies after Tracy's sister took 2 to 6
    ∃ (c : ℕ), b - 20 - c = 4 ∧ 2 ≤ c ∧ c ≤ 6
  }

theorem initial_candies_are_52_or_56 : initial_candies = {52, 56} := by sorry

end NUMINAMATH_CALUDE_initial_candies_are_52_or_56_l2224_222492


namespace NUMINAMATH_CALUDE_trust_fund_remaining_zero_l2224_222482

/-- Represents the ratio of distribution for each beneficiary -/
structure DistributionRatio :=
  (dina : Rat)
  (eva : Rat)
  (frank : Rat)

/-- Theorem stating that the remaining fraction of the fund is 0 -/
theorem trust_fund_remaining_zero (ratio : DistributionRatio) 
  (h1 : ratio.dina = 4/8)
  (h2 : ratio.eva = 3/8)
  (h3 : ratio.frank = 1/8)
  (h4 : ratio.dina + ratio.eva + ratio.frank = 1) :
  let remaining : Rat := 1 - (ratio.dina + (1 - ratio.dina) * ratio.eva + (1 - ratio.dina - (1 - ratio.dina) * ratio.eva) * ratio.frank)
  remaining = 0 := by sorry

end NUMINAMATH_CALUDE_trust_fund_remaining_zero_l2224_222482


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l2224_222462

theorem shirt_cost_theorem (cost_first : ℕ) (cost_difference : ℕ) : 
  cost_first = 15 → cost_difference = 6 → cost_first + (cost_first - cost_difference) = 24 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l2224_222462


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2224_222433

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (hyperbola x y) → (asymptote x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2224_222433


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2224_222415

theorem lcm_gcf_problem (n : ℕ+) :
  Nat.lcm n 16 = 52 →
  Nat.gcd n 16 = 8 →
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2224_222415


namespace NUMINAMATH_CALUDE_midpoint_movement_l2224_222406

/-- Given two points A and B in a Cartesian plane, their midpoint, and their new positions after
    movement, prove the new midpoint and its distance from the original midpoint. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let M : ℝ × ℝ := (m, n)
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 6, d - 3)
  let M' : ℝ × ℝ := ((a + 3 + c - 6) / 2, (b + 5 + d - 3) / 2)
  (M = ((a + c) / 2, (b + d) / 2)) →
  (M' = (m - 3 / 2, n + 1) ∧
   Real.sqrt ((m - 3 / 2 - m) ^ 2 + (n + 1 - n) ^ 2) = Real.sqrt 13 / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_movement_l2224_222406


namespace NUMINAMATH_CALUDE_simplify_negative_cube_squared_l2224_222469

theorem simplify_negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_cube_squared_l2224_222469


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2224_222417

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  monotone_increasing : Monotone a
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  condition1 : a 2 * a 5 = 6
  condition2 : a 3 + a 4 = 5

/-- The common ratio of the geometric sequence is 3/2 -/
theorem geometric_sequence_common_ratio (seq : GeometricSequence) :
  seq.a 2 / seq.a 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2224_222417


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2224_222465

theorem pie_eating_contest (first_student_total : ℚ) (second_student_total : ℚ) 
  (third_student_first_pie : ℚ) (third_student_second_pie : ℚ) :
  first_student_total = 7/6 ∧ 
  second_student_total = 4/3 ∧ 
  third_student_first_pie = 1/2 ∧ 
  third_student_second_pie = 1/3 →
  (first_student_total - third_student_first_pie * first_student_total = 2/3) ∧
  (second_student_total - third_student_second_pie * second_student_total = 1) ∧
  (third_student_first_pie * first_student_total + third_student_second_pie * second_student_total = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2224_222465


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l2224_222408

-- Define the line l: mx - y - 3m + 4 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 4 = 0

-- Define the circle O: x^2 + y^2 = 4
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the fixed point M(3,4) on line l
def point_M : ℝ × ℝ := (3, 4)

-- Theorem 1: Maximum distance from circle O to line l
theorem max_distance_circle_to_line :
  ∃ (m : ℝ), ∀ (x y : ℝ), circle_O x y →
    ∃ (max_dist : ℝ), max_dist = 7 ∧
      ∀ (x' y' : ℝ), line_l m x' y' →
        Real.sqrt ((x - x')^2 + (y - y')^2) ≤ max_dist :=
sorry

-- Theorem 2: Tangent line condition
theorem tangent_line_condition :
  ∃ (m : ℝ), m = (12 + 2 * Real.sqrt 21) / 5 ∨ m = (12 - 2 * Real.sqrt 21) / 5 →
    ∀ (x y : ℝ), line_l m x y →
      (∃! (x' y' : ℝ), circle_O x' y' ∧ x = x' ∧ y = y') :=
sorry

-- Theorem 3: Chord length condition
theorem chord_length_condition :
  ∃ (m : ℝ), m = (6 + Real.sqrt 6) / 4 ∨ m = (6 - Real.sqrt 6) / 4 →
    ∃ (x1 y1 x2 y2 : ℝ),
      line_l m x1 y1 ∧ line_l m x2 y2 ∧
      circle_O x1 y1 ∧ circle_O x2 y2 ∧
      Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l2224_222408
