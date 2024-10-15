import Mathlib

namespace NUMINAMATH_CALUDE_min_product_xyz_l337_33785

theorem min_product_xyz (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 3/125 := by
  sorry

end NUMINAMATH_CALUDE_min_product_xyz_l337_33785


namespace NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l337_33745

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 4)
  (sum_products_eq : x*y + x*z + y*z = 3)
  (product_eq : x*y*z = -10) :
  x^3 + y^3 + z^3 = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l337_33745


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l337_33726

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : (U \ (A ∩ B)) = {1, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l337_33726


namespace NUMINAMATH_CALUDE_current_speed_l337_33706

/-- Proves that given a woman swimming downstream 81 km in 9 hours and upstream 36 km in 9 hours, the speed of the current is 2.5 km/h. -/
theorem current_speed (v : ℝ) (c : ℝ) : 
  (v + c) * 9 = 81 → 
  (v - c) * 9 = 36 → 
  c = 2.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l337_33706


namespace NUMINAMATH_CALUDE_tensor_A_B_l337_33768

-- Define the ⊗ operation
def tensor (M N : Set ℝ) : Set ℝ := (M ∪ N) \ (M ∩ N)

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem tensor_A_B : tensor A B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_tensor_A_B_l337_33768


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l337_33769

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 6 < 0) ↔ (∃ x : ℝ, x^2 + x - 6 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l337_33769


namespace NUMINAMATH_CALUDE_quadratic_function_range_l337_33724

/-- A quadratic function with a positive coefficient for the squared term -/
structure PositiveQuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  positive_coeff : ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0

/-- The main theorem -/
theorem quadratic_function_range
  (f : PositiveQuadraticFunction)
  (h_symmetry : ∀ x : ℝ, f.f (2 + x) = f.f (2 - x))
  (h_inequality : ∀ x : ℝ, f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)) :
  {x : ℝ | -2 < x ∧ x < 0} = {x : ℝ | f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l337_33724


namespace NUMINAMATH_CALUDE_young_li_age_is_20_l337_33749

/-- Young Li's age this year -/
def young_li_age : ℕ := 20

/-- Old Li's age this year -/
def old_li_age : ℕ := young_li_age * 5 / 2

theorem young_li_age_is_20 :
  (old_li_age = young_li_age * 5 / 2) ∧
  (old_li_age + 10 = (young_li_age + 10) * 2) →
  young_li_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_young_li_age_is_20_l337_33749


namespace NUMINAMATH_CALUDE_circle_center_transformation_l337_33719

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_x := reflect_x initial_center
  let reflected_y := reflect_y reflected_x
  let final_center := translate reflected_y 5 3
  final_center = (2, 7) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l337_33719


namespace NUMINAMATH_CALUDE_starting_lineup_theorem_l337_33755

def total_players : ℕ := 18
def goalie_count : ℕ := 1
def regular_players_count : ℕ := 10
def captain_count : ℕ := 1

def starting_lineup_count : ℕ :=
  total_players * (Nat.choose (total_players - goalie_count) regular_players_count) * regular_players_count

theorem starting_lineup_theorem :
  starting_lineup_count = 34928640 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_theorem_l337_33755


namespace NUMINAMATH_CALUDE_smallest_w_l337_33704

theorem smallest_w (w : ℕ+) (h1 : (2^5 : ℕ) ∣ (936 * w))
                            (h2 : (3^3 : ℕ) ∣ (936 * w))
                            (h3 : (12^2 : ℕ) ∣ (936 * w)) :
  w ≥ 36 ∧ (∃ (v : ℕ+), v ≥ 36 → 
    (2^5 : ℕ) ∣ (936 * v) ∧ 
    (3^3 : ℕ) ∣ (936 * v) ∧ 
    (12^2 : ℕ) ∣ (936 * v)) :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l337_33704


namespace NUMINAMATH_CALUDE_mustard_at_first_table_l337_33741

-- Define the amount of mustard at each table
def mustard_table1 : ℝ := sorry
def mustard_table2 : ℝ := 0.25
def mustard_table3 : ℝ := 0.38

-- Define the total amount of mustard
def total_mustard : ℝ := 0.88

-- Theorem statement
theorem mustard_at_first_table :
  mustard_table1 + mustard_table2 + mustard_table3 = total_mustard →
  mustard_table1 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_mustard_at_first_table_l337_33741


namespace NUMINAMATH_CALUDE_train_length_l337_33732

/-- Given a train with a speed of 125.99999999999999 km/h that can cross an electric pole in 20 seconds,
    prove that the length of the train is 700 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 125.99999999999999 → 
  time = 20 → 
  length = speed * (1000 / 3600) * time → 
  length = 700 := by sorry

end NUMINAMATH_CALUDE_train_length_l337_33732


namespace NUMINAMATH_CALUDE_paul_lost_crayons_l337_33751

/-- Given information about Paul's crayons --/
def paul_crayons (initial : ℕ) (given_to_friends : ℕ) (difference : ℕ) : Prop :=
  ∃ (lost : ℕ), 
    initial ≥ given_to_friends + lost ∧
    given_to_friends = lost + difference

/-- Theorem stating the number of crayons Paul lost --/
theorem paul_lost_crayons :
  paul_crayons 589 571 410 → ∃ (lost : ℕ), lost = 161 := by
  sorry

end NUMINAMATH_CALUDE_paul_lost_crayons_l337_33751


namespace NUMINAMATH_CALUDE_youngest_child_age_problem_l337_33756

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 50 = 6 := by sorry

end NUMINAMATH_CALUDE_youngest_child_age_problem_l337_33756


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l337_33743

theorem cone_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 4) 
  (hh : h = 3) : 
  r * (Real.sqrt (r^2 + h^2)) * π = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l337_33743


namespace NUMINAMATH_CALUDE_complex_equation_solution_l337_33765

theorem complex_equation_solution (z : ℂ) (h : (1 + 3*Complex.I)*z = Complex.I - 3) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l337_33765


namespace NUMINAMATH_CALUDE_constant_radius_is_cylinder_l337_33715

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

/-- The set of points satisfying r = c -/
def ConstantRadiusSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Theorem: The set of points satisfying r = c forms a cylinder -/
theorem constant_radius_is_cylinder (c : ℝ) :
    IsCylinder (ConstantRadiusSet c) c := by
  sorry


end NUMINAMATH_CALUDE_constant_radius_is_cylinder_l337_33715


namespace NUMINAMATH_CALUDE_geometric_figure_area_l337_33748

theorem geometric_figure_area (x : ℝ) : 
  x > 0 →
  (3*x)^2 + (4*x)^2 + (1/2) * (3*x) * (4*x) = 1200 →
  x = Real.sqrt (1200/31) := by
sorry

end NUMINAMATH_CALUDE_geometric_figure_area_l337_33748


namespace NUMINAMATH_CALUDE_bolzano_weierstrass_unit_interval_l337_33734

/-- Bolzano-Weierstrass theorem for sequences in [0, 1) -/
theorem bolzano_weierstrass_unit_interval (s : ℕ → ℝ) (h : ∀ n, 0 ≤ s n ∧ s n < 1) :
  (∃ (a : Set ℕ), Set.Infinite a ∧ (∀ n ∈ a, s n < 1/2)) ∨
  (∃ (b : Set ℕ), Set.Infinite b ∧ (∀ n ∈ b, 1/2 ≤ s n)) ∧
  ∀ ε > 0, ε < 1/2 → ∃ α : ℝ, 0 ≤ α ∧ α ≤ 1 ∧
    ∃ (c : Set ℕ), Set.Infinite c ∧ ∀ n ∈ c, |s n - α| < ε :=
by sorry

end NUMINAMATH_CALUDE_bolzano_weierstrass_unit_interval_l337_33734


namespace NUMINAMATH_CALUDE_stratified_sample_imported_count_l337_33714

/-- Represents the number of marker lights in a population -/
structure MarkerLightPopulation where
  total : ℕ
  coDeveloped : ℕ
  domestic : ℕ
  h_sum : total = imported + coDeveloped + domestic

/-- Represents a stratified sample of marker lights -/
structure StratifiedSample where
  populationSize : ℕ
  sampleSize : ℕ

/-- Theorem stating that the number of imported marker lights in a stratified sample
    is proportional to their representation in the population -/
theorem stratified_sample_imported_count 
  (population : MarkerLightPopulation)
  (sample : StratifiedSample)
  (h_pop_size : sample.populationSize = population.total)
  (h_imported : sample.importedInPopulation = population.imported)
  (h_sample_size : sample.sampleSize = 20)
  (h_stratified : sample.importedInSample * population.total = 
                  population.imported * sample.sampleSize) :
  sample.importedInSample = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_imported_count_l337_33714


namespace NUMINAMATH_CALUDE_tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l337_33709

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 2) / x + 2 - 2 * a

-- State the theorems to be proved
theorem tangent_line_at_2_when_a_1 :
  ∀ x y : ℝ, f 1 2 = 3/2 ∧ (5 * x - 4 * y - 4 = 0 ↔ y - 3/2 = 5/4 * (x - 2)) :=
sorry

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a ≤ 2 → 
    StrictMono (f a) ∧ 
    StrictMonoOn (f a) (Set.Ioi 0)) ∧
  (a > 2 → 
    (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧
      StrictMonoOn (f a) (Set.Iic x₁) ∧
      StrictAntiOn (f a) (Set.Ioc x₁ 0) ∧
      StrictAntiOn (f a) (Set.Ioc 0 x₂) ∧
      StrictMonoOn (f a) (Set.Ioi x₂))) :=
sorry

theorem f_geq_2ln_x_iff (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 2 * Real.log x) ↔ a ≥ 1 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l337_33709


namespace NUMINAMATH_CALUDE_max_value_ahn_operation_l337_33736

theorem max_value_ahn_operation :
  ∃ (max : ℕ), max = 600 ∧
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 →
  3 * (300 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_ahn_operation_l337_33736


namespace NUMINAMATH_CALUDE_mod_product_equality_l337_33747

theorem mod_product_equality (m : ℕ) : 
  (256 * 738 ≡ m [ZMOD 75]) → 
  (0 ≤ m ∧ m < 75) → 
  m = 53 := by
sorry

end NUMINAMATH_CALUDE_mod_product_equality_l337_33747


namespace NUMINAMATH_CALUDE_gcd_inequality_l337_33771

theorem gcd_inequality (n : ℕ) :
  (∀ k ∈ Finset.range 34, Nat.gcd n (n + k) < Nat.gcd n (n + k + 1)) →
  Nat.gcd n (n + 35) < Nat.gcd n (n + 36) := by
sorry

end NUMINAMATH_CALUDE_gcd_inequality_l337_33771


namespace NUMINAMATH_CALUDE_prism_triangle_areas_sum_l337_33728

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the sum of areas of all triangles with vertices at corners of the prism -/
def sumTriangleAreas (prism : RectangularPrism) : ℝ :=
  sorry

/-- Represents the result of sumTriangleAreas as m + √n + √p -/
structure AreaSum where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts the sum of triangle areas to the AreaSum form -/
def toAreaSum (sum : ℝ) : AreaSum :=
  sorry

theorem prism_triangle_areas_sum (prism : RectangularPrism) 
  (h1 : prism.a = 1) (h2 : prism.b = 1) (h3 : prism.c = 2) : 
  let sum := sumTriangleAreas prism
  let result := toAreaSum sum
  result.m + result.n + result.p = 41 :=
sorry

end NUMINAMATH_CALUDE_prism_triangle_areas_sum_l337_33728


namespace NUMINAMATH_CALUDE_monica_reading_plan_l337_33723

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l337_33723


namespace NUMINAMATH_CALUDE_isosceles_triangle_two_two_one_l337_33795

/-- Checks if three numbers can form a triangle based on the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if three numbers can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem: The set of side lengths (2, 2, 1) forms an isosceles triangle -/
theorem isosceles_triangle_two_two_one :
  is_isosceles_triangle 2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_two_two_one_l337_33795


namespace NUMINAMATH_CALUDE_ab_value_l337_33763

theorem ab_value (a b : ℝ) (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a^b = (1/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ab_value_l337_33763


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l337_33798

theorem sin_three_pi_halves : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l337_33798


namespace NUMINAMATH_CALUDE_initial_amount_correct_l337_33744

/-- The amount of money John initially gave when buying barbells -/
def initial_amount : ℕ := 850

/-- The number of barbells John bought -/
def num_barbells : ℕ := 3

/-- The cost of each barbell in dollars -/
def barbell_cost : ℕ := 270

/-- The amount of change John received in dollars -/
def change_received : ℕ := 40

/-- Theorem stating that the initial amount John gave is correct -/
theorem initial_amount_correct : 
  initial_amount = num_barbells * barbell_cost + change_received :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_correct_l337_33744


namespace NUMINAMATH_CALUDE_contrapositive_proof_l337_33791

theorem contrapositive_proof (a b : ℝ) :
  (∀ a b, a > b → a - 5 > b - 5) ↔ (∀ a b, a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l337_33791


namespace NUMINAMATH_CALUDE_greatest_power_less_than_500_l337_33739

theorem greatest_power_less_than_500 (c d : ℕ+) (h1 : d > 1) 
  (h2 : c^(d:ℕ) < 500) 
  (h3 : ∀ (x y : ℕ+), y > 1 → x^(y:ℕ) < 500 → x^(y:ℕ) ≤ c^(d:ℕ)) : 
  c + d = 24 := by sorry

end NUMINAMATH_CALUDE_greatest_power_less_than_500_l337_33739


namespace NUMINAMATH_CALUDE_select_students_equality_l337_33787

/-- The number of ways to select 5 students from a class of 50, including one president and one 
    vice-president, with at least one of the president or vice-president attending. -/
def select_students (n : ℕ) (k : ℕ) (total : ℕ) (leaders : ℕ) : ℕ :=
  Nat.choose leaders 1 * Nat.choose (total - leaders) (k - 1) +
  Nat.choose leaders 2 * Nat.choose (total - leaders) (k - 2)

theorem select_students_equality :
  select_students 5 5 50 2 = Nat.choose 50 5 - Nat.choose 48 5 :=
sorry

end NUMINAMATH_CALUDE_select_students_equality_l337_33787


namespace NUMINAMATH_CALUDE_marbles_problem_l337_33757

theorem marbles_problem (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 99) (h2 : marbles_per_boy = 9) :
  total_marbles / marbles_per_boy = 11 :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l337_33757


namespace NUMINAMATH_CALUDE_total_pebbles_after_fifteen_days_l337_33740

/-- The number of pebbles collected on the first day -/
def initial_pebbles : ℕ := 3

/-- The daily increase in the number of pebbles collected -/
def daily_increase : ℕ := 2

/-- The number of days Murtha collects pebbles -/
def collection_days : ℕ := 15

/-- The arithmetic sequence of daily pebble collections -/
def pebble_sequence (n : ℕ) : ℕ := initial_pebbles + (n - 1) * daily_increase

/-- The total number of pebbles collected after a given number of days -/
def total_pebbles (n : ℕ) : ℕ := n * (initial_pebbles + pebble_sequence n) / 2

theorem total_pebbles_after_fifteen_days :
  total_pebbles collection_days = 255 := by sorry

end NUMINAMATH_CALUDE_total_pebbles_after_fifteen_days_l337_33740


namespace NUMINAMATH_CALUDE_parabola_shift_l337_33790

def original_function (x : ℝ) : ℝ := -2 * (x + 1)^2 + 5

def shift_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x + shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

theorem parabola_shift :
  ∀ x : ℝ, shift_down (shift_left original_function 2) 4 x = final_function x :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l337_33790


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l337_33725

theorem unique_triplet_solution : 
  ∃! (m n k : ℕ), 3^n + 4^m = 5^k :=
by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l337_33725


namespace NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l337_33746

/-- Two lines in a plane are parallel if they do not intersect. -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l₁ → p ∉ l₂

/-- A line segment is perpendicular to a line if it forms a right angle with the line. -/
def perpendicular (seg : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment. -/
def length (seg : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: All perpendicular line segments between two parallel lines are equal in length. -/
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h_parallel : parallel l₁ l₂) 
  (seg₁ seg₂ : Set (ℝ × ℝ)) 
  (h_perp₁ : perpendicular seg₁ l₁ ∧ perpendicular seg₁ l₂)
  (h_perp₂ : perpendicular seg₂ l₁ ∧ perpendicular seg₂ l₂) :
  length seg₁ = length seg₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l337_33746


namespace NUMINAMATH_CALUDE_function_cuts_x_axis_l337_33797

theorem function_cuts_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x + 2 * x = 0 := by sorry

end NUMINAMATH_CALUDE_function_cuts_x_axis_l337_33797


namespace NUMINAMATH_CALUDE_ratio_equivalence_l337_33783

theorem ratio_equivalence : ∃ (x y : ℚ) (z : ℕ),
  (4 : ℚ) / 5 = 20 / x ∧
  (4 : ℚ) / 5 = y / 20 ∧
  (4 : ℚ) / 5 = (z : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l337_33783


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l337_33789

theorem consecutive_integers_sum (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (Finset.range n).sum (λ i => m - i) = n) ↔ n % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l337_33789


namespace NUMINAMATH_CALUDE_g_nested_3_l337_33731

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_nested_3 : g (g (g (g 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_nested_3_l337_33731


namespace NUMINAMATH_CALUDE_inequality_solution_set_l337_33705

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l337_33705


namespace NUMINAMATH_CALUDE_quadratic_root_value_l337_33750

/-- Given a quadratic equation 6x^2 + 5x + q with roots (-5 ± i√323) / 12, q equals 14.5 -/
theorem quadratic_root_value (q : ℝ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + q = 0 ↔ x = (-5 + Complex.I * Real.sqrt 323) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 323) / 12) →
  q = 14.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l337_33750


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l337_33794

theorem arithmetic_expression_evaluation :
  5 * 7 + 9 * 4 - 30 / 3 + 2^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l337_33794


namespace NUMINAMATH_CALUDE_ten_sentences_per_paragraph_l337_33762

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of pages in the book -/
def pages : ℕ := 50

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Calculates the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ :=
  (reading_speed * total_reading_time) / (pages * paragraphs_per_page)

/-- Theorem stating that there are 10 sentences per paragraph -/
theorem ten_sentences_per_paragraph : sentences_per_paragraph = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_sentences_per_paragraph_l337_33762


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l337_33761

theorem quadratic_root_problem (m : ℝ) : 
  (3 * (1 : ℝ)^2 + m * 1 - 7 = 0) → 
  (∃ x : ℝ, x ≠ 1 ∧ 3 * x^2 + m * x - 7 = 0 ∧ x = -7/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l337_33761


namespace NUMINAMATH_CALUDE_julia_short_amount_l337_33718

/-- Represents the cost and quantity of CDs Julia wants to buy -/
structure CDPurchase where
  rock_price : ℕ
  pop_price : ℕ
  dance_price : ℕ
  country_price : ℕ
  quantity : ℕ

/-- Calculates the amount Julia is short given her CD purchase and available money -/
def amount_short (purchase : CDPurchase) (available_money : ℕ) : ℕ :=
  let total_cost := purchase.quantity * (purchase.rock_price + purchase.pop_price + purchase.dance_price + purchase.country_price)
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Julia is short $25 given the specific CD prices, quantities, and available money -/
theorem julia_short_amount : amount_short ⟨5, 10, 3, 7, 4⟩ 75 = 25 := by
  sorry

end NUMINAMATH_CALUDE_julia_short_amount_l337_33718


namespace NUMINAMATH_CALUDE_candy_left_l337_33764

theorem candy_left (initial : ℝ) (morning : ℝ) (afternoon : ℝ) :
  initial = 38 →
  morning = 7.5 →
  afternoon = 15.25 →
  initial - morning - afternoon = 15.25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_left_l337_33764


namespace NUMINAMATH_CALUDE_tshirt_company_profit_l337_33799

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit (
  num_employees : ℕ
  ) (shirts_per_employee : ℕ
  ) (shift_hours : ℕ
  ) (hourly_wage : ℚ
  ) (per_shirt_bonus : ℚ
  ) (shirt_price : ℚ
  ) (nonemployee_expenses : ℚ
  ) (h1 : num_employees = 20
  ) (h2 : shirts_per_employee = 20
  ) (h3 : shift_hours = 8
  ) (h4 : hourly_wage = 12
  ) (h5 : per_shirt_bonus = 5
  ) (h6 : shirt_price = 35
  ) (h7 : nonemployee_expenses = 1000
  ) : (num_employees * shirts_per_employee * shirt_price) -
      (num_employees * shift_hours * hourly_wage +
       num_employees * shirts_per_employee * per_shirt_bonus +
       nonemployee_expenses) = 9080 := by
  sorry


end NUMINAMATH_CALUDE_tshirt_company_profit_l337_33799


namespace NUMINAMATH_CALUDE_bruce_eggs_l337_33753

theorem bruce_eggs (initial_eggs lost_eggs : ℕ) : 
  initial_eggs ≥ lost_eggs → 
  initial_eggs - lost_eggs = initial_eggs - lost_eggs :=
by
  sorry

#check bruce_eggs 75 70

end NUMINAMATH_CALUDE_bruce_eggs_l337_33753


namespace NUMINAMATH_CALUDE_economic_loss_scientific_notation_l337_33770

-- Define the original number in millions
def original_number : ℝ := 16823

-- Define the scientific notation components
def coefficient : ℝ := 1.6823
def exponent : ℤ := 4

-- Theorem statement
theorem economic_loss_scientific_notation :
  original_number = coefficient * (10 : ℝ) ^ exponent :=
sorry

end NUMINAMATH_CALUDE_economic_loss_scientific_notation_l337_33770


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l337_33711

theorem sphere_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (3 * r)^3) = 3 * ((4 / 3 * Real.pi * r^3) + (4 / 3 * Real.pi * (2 * r)^3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l337_33711


namespace NUMINAMATH_CALUDE_problem_solution_l337_33767

theorem problem_solution (x : ℝ) : 
  (7/11) * (5/13) * x = 48 → (315/100) * x = 617.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l337_33767


namespace NUMINAMATH_CALUDE_prob_red_ball_specific_l337_33727

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  sum_colors : total = red + yellow + green

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem: The probability of drawing a red ball from a bag with 15 balls, 
    of which 8 are red, is 8/15 -/
theorem prob_red_ball_specific : 
  ∃ (bag : ColoredBalls), bag.total = 15 ∧ bag.red = 8 ∧ prob_red_ball bag = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_prob_red_ball_specific_l337_33727


namespace NUMINAMATH_CALUDE_point_inside_circle_l337_33788

theorem point_inside_circle (a : ℝ) : 
  let P : ℝ × ℝ := (5*a + 1, 12*a)
  ((P.1 - 1)^2 + P.2^2 < 1) ↔ (abs a < 1/13) :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l337_33788


namespace NUMINAMATH_CALUDE_cubic_inequality_and_fraction_bound_l337_33775

theorem cubic_inequality_and_fraction_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a/b^2 + b/a^2 ≥ m/2 * (1/a + 1/b)) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_and_fraction_bound_l337_33775


namespace NUMINAMATH_CALUDE_min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l337_33720

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x * |x - 2*a|

-- Part 1: Minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x : ℝ), f (-1) x ≥ m :=
sorry

-- Part 2: Range of x₁/x₂ + x₁ when a = -9
theorem range_of_expression_when_a_neg_nine :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f (-9) x₁ = f (-9) x₂ →
  (x₁/x₂ + x₁ < -16 ∨ x₁/x₂ + x₁ ≥ -4) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l337_33720


namespace NUMINAMATH_CALUDE_grape_rate_proof_l337_33792

theorem grape_rate_proof (grapes_kg mangoes_kg mangoes_rate total_paid : ℕ) 
  (h1 : grapes_kg = 8)
  (h2 : mangoes_kg = 9)
  (h3 : mangoes_rate = 65)
  (h4 : total_paid = 1145)
  : ∃ (grape_rate : ℕ), grape_rate * grapes_kg + mangoes_kg * mangoes_rate = total_paid ∧ grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l337_33792


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l337_33760

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℤ), 5 ≤ n ∧ n ≤ 15 ∧ ∃ (m : ℤ), 2 * n^2 + n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l337_33760


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l337_33730

theorem last_digit_of_sum (n : ℕ) : 
  (5^555 + 6^666 + 7^777) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l337_33730


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l337_33700

def f (x : ℝ) := 10 * abs x

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l337_33700


namespace NUMINAMATH_CALUDE_right_triangle_area_in_circle_l337_33773

/-- The area of a right triangle inscribed in a circle -/
theorem right_triangle_area_in_circle (r : ℝ) (h : r = 5) :
  let a : ℝ := 5 * (10 / 13)
  let b : ℝ := 12 * (10 / 13)
  let c : ℝ := 13 * (10 / 13)
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter is the hypotenuse
  (1/2 * a * b = 6000/169) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_in_circle_l337_33773


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l337_33729

def x : ℕ := 2^2 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 9^9

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem smallest_factor_for_perfect_square :
  (∀ k : ℕ, k < 105 → ¬is_perfect_square (k * x)) ∧
  is_perfect_square (105 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l337_33729


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l337_33710

-- Define the polynomial
def P (x a b : ℝ) : ℝ := 2*x^4 - 3*x^3 + a*x^2 + 7*x + b

-- Define the divisor
def D (x : ℝ) : ℝ := x^2 + x - 2

-- Theorem statement
theorem polynomial_division_theorem (a b : ℝ) :
  (∀ x, ∃ q, P x a b = D x * q) →
  a / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l337_33710


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l337_33774

def day1_income : ℕ := 300
def day2_income : ℕ := 150
def day3_income : ℕ := 750
def day4_income : ℕ := 400
def day5_income : ℕ := 500
def num_days : ℕ := 5

theorem cab_driver_average_income :
  (day1_income + day2_income + day3_income + day4_income + day5_income) / num_days = 420 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l337_33774


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_50_l337_33793

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to find the nth prime number greater than a given value
def nthPrimeGreaterThan (n : ℕ) (start : ℕ) : ℕ :=
  sorry

theorem least_product_of_three_primes_greater_than_50 :
  ∃ p q r : ℕ,
    isPrime p ∧ isPrime q ∧ isPrime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p < q ∧ q < r ∧
    p * q * r = 191557 ∧
    ∀ a b c : ℕ,
      isPrime a ∧ isPrime b ∧ isPrime c ∧
      a > 50 ∧ b > 50 ∧ c > 50 ∧
      a < b ∧ b < c →
      a * b * c ≥ 191557 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_50_l337_33793


namespace NUMINAMATH_CALUDE_solve_for_n_l337_33759

theorem solve_for_n (s m k r P : ℝ) (h : P = (s + m) / ((1 + k)^n + r)) :
  n = Real.log ((s + m - P * r) / P) / Real.log (1 + k) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l337_33759


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l337_33716

/-- Represents the seating position of the k-th person -/
def seat_position (n k : ℕ) : ℕ := (k * (k - 1) / 2) % n

/-- Checks if all seating positions are distinct -/
def all_distinct_positions (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → seat_position n i ≠ seat_position n j

/-- Checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2^m

theorem seating_arrangement_theorem (n : ℕ) :
  n > 0 → (all_distinct_positions n ↔ is_power_of_two n) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l337_33716


namespace NUMINAMATH_CALUDE_repetitions_for_99_cubes_impossible_2016_cubes_l337_33752

/-- The number of cubes after x repetitions of the cutting process -/
def num_cubes (x : ℕ) : ℕ := 7 * x + 1

/-- Theorem stating that 14 repetitions are needed to obtain 99 cubes -/
theorem repetitions_for_99_cubes : ∃ x : ℕ, num_cubes x = 99 ∧ x = 14 := by sorry

/-- Theorem stating that it's impossible to obtain 2016 cubes -/
theorem impossible_2016_cubes : ¬∃ x : ℕ, num_cubes x = 2016 := by sorry

end NUMINAMATH_CALUDE_repetitions_for_99_cubes_impossible_2016_cubes_l337_33752


namespace NUMINAMATH_CALUDE_mall_a_better_deal_l337_33733

/-- Calculates the discount for a given spent amount and promotion rule -/
def calculate_discount (spent : ℕ) (promotion_threshold : ℕ) (promotion_discount : ℕ) : ℕ :=
  (spent / promotion_threshold) * promotion_discount

/-- Calculates the final cost after applying the discount -/
def calculate_final_cost (total : ℕ) (discount : ℕ) : ℕ :=
  total - discount

theorem mall_a_better_deal (shoes_price : ℕ) (sweater_price : ℕ)
    (h_shoes : shoes_price = 699)
    (h_sweater : sweater_price = 910)
    (mall_a_threshold : ℕ) (mall_a_discount : ℕ)
    (mall_b_threshold : ℕ) (mall_b_discount : ℕ)
    (h_mall_a : mall_a_threshold = 200 ∧ mall_a_discount = 101)
    (h_mall_b : mall_b_threshold = 101 ∧ mall_b_discount = 50) :
    let total := shoes_price + sweater_price
    let discount_a := calculate_discount total mall_a_threshold mall_a_discount
    let discount_b := calculate_discount total mall_b_threshold mall_b_discount
    let final_cost_a := calculate_final_cost total discount_a
    let final_cost_b := calculate_final_cost total discount_b
    final_cost_a < final_cost_b ∧ final_cost_a = 801 := by
  sorry

end NUMINAMATH_CALUDE_mall_a_better_deal_l337_33733


namespace NUMINAMATH_CALUDE_f_properties_l337_33702

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (∀ x : ℝ, f (f x) ≤ 0) ∧
  (f 0 ≥ 0 → ∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l337_33702


namespace NUMINAMATH_CALUDE_complex_magnitude_reciprocal_one_plus_i_l337_33758

theorem complex_magnitude_reciprocal_one_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 1 / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_complex_magnitude_reciprocal_one_plus_i_l337_33758


namespace NUMINAMATH_CALUDE_john_total_cost_l337_33782

/-- The total cost for John to raise a child and pay for university tuition -/
def total_cost_for_john : ℕ := 
  let cost_per_year_first_8 := 10000
  let years_first_period := 8
  let years_second_period := 10
  let university_tuition := 250000
  let first_period_cost := cost_per_year_first_8 * years_first_period
  let second_period_cost := 2 * cost_per_year_first_8 * years_second_period
  let total_cost := first_period_cost + second_period_cost + university_tuition
  total_cost / 2

/-- Theorem stating that the total cost for John is $265,000 -/
theorem john_total_cost : total_cost_for_john = 265000 := by
  sorry

end NUMINAMATH_CALUDE_john_total_cost_l337_33782


namespace NUMINAMATH_CALUDE_equation_solution_l337_33708

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l337_33708


namespace NUMINAMATH_CALUDE_sum_of_first_four_powers_of_i_is_zero_l337_33712

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The property that i^2 = -1 -/
axiom i_squared : i^2 = -1

/-- Theorem: The sum of the first four powers of i equals 0 -/
theorem sum_of_first_four_powers_of_i_is_zero : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_powers_of_i_is_zero_l337_33712


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l337_33776

theorem arccos_lt_arcsin_iff (x : ℝ) : 
  Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_l337_33776


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_five_primes_l337_33717

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

theorem arithmetic_mean_of_reciprocals_first_five_primes :
  arithmetic_mean (reciprocals first_five_primes) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_five_primes_l337_33717


namespace NUMINAMATH_CALUDE_iron_cubes_melting_l337_33778

theorem iron_cubes_melting (s1 s2 s3 s_large : ℝ) : 
  s1 = 1 ∧ s2 = 6 ∧ s3 = 8 → 
  s_large^3 = s1^3 + s2^3 + s3^3 →
  s_large = 9 := by
sorry

end NUMINAMATH_CALUDE_iron_cubes_melting_l337_33778


namespace NUMINAMATH_CALUDE_black_white_difference_l337_33742

/-- Represents the number of pieces in a box -/
structure PieceCount where
  black : ℕ
  white : ℕ

/-- The condition of the problem -/
def satisfiesCondition (p : PieceCount) : Prop :=
  (p.black - 1) / p.white = 9 / 7 ∧
  p.black / (p.white - 1) = 7 / 5

/-- The theorem to be proved -/
theorem black_white_difference (p : PieceCount) :
  satisfiesCondition p → p.black - p.white = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_white_difference_l337_33742


namespace NUMINAMATH_CALUDE_complement_of_A_l337_33796

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.univ \ A) ↔ x ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l337_33796


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l337_33707

theorem tangent_point_coordinates (f : ℝ → ℝ) (h : f = λ x ↦ Real.exp x) :
  ∃ (x y : ℝ), x = 1 ∧ y = Real.exp 1 ∧
  (∀ t : ℝ, f t = Real.exp t) ∧
  (∃ m : ℝ, ∀ t : ℝ, y - f x = m * (t - x) ∧ 0 = m * (-x)) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l337_33707


namespace NUMINAMATH_CALUDE_x_value_l337_33772

theorem x_value (w y z x : ℕ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 10) : 
  x = 145 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l337_33772


namespace NUMINAMATH_CALUDE_log_inequality_cube_inequality_l337_33781

theorem log_inequality_cube_inequality (a b : ℝ) :
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) ∧
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_cube_inequality_l337_33781


namespace NUMINAMATH_CALUDE_complex_subtraction_l337_33701

theorem complex_subtraction (a b : ℂ) (ha : a = 6 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l337_33701


namespace NUMINAMATH_CALUDE_california_permutations_count_l337_33780

/-- The number of distinct permutations of a word with repeated letters -/
def wordPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of CALIFORNIA -/
def californiaPermutations : ℕ := wordPermutations 10 [3, 2]

theorem california_permutations_count :
  californiaPermutations = 302400 := by
  sorry

end NUMINAMATH_CALUDE_california_permutations_count_l337_33780


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l337_33777

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l337_33777


namespace NUMINAMATH_CALUDE_smallest_difference_l337_33766

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = 4 ∧ ∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l337_33766


namespace NUMINAMATH_CALUDE_zoo_field_trip_l337_33703

theorem zoo_field_trip (students_class1 students_class2 parent_chaperones : ℕ)
  (students_left chaperones_left remaining : ℕ) :
  students_class1 = 10 →
  students_class2 = 10 →
  parent_chaperones = 5 →
  students_left = 10 →
  chaperones_left = 2 →
  remaining = 15 →
  ∃ (teachers : ℕ),
    teachers = 2 ∧
    (students_class1 + students_class2 + parent_chaperones + teachers) -
    (students_left + chaperones_left) = remaining :=
by sorry

end NUMINAMATH_CALUDE_zoo_field_trip_l337_33703


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l337_33738

/-- Represents a rectangular garden with length and breadth -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 1800)
  (h2 : garden.length = 500) :
  garden.breadth = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l337_33738


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l337_33722

/-- Given a square field with area 3136 sq m and a total cost of 865.80 for barbed wire
    (excluding two 1 m wide gates), the rate per meter of barbed wire is 3.90. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - gate_width * num_gates)) = 3.90 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l337_33722


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_gt_three_l337_33713

theorem negation_of_existence_sqrt_gt_three : 
  (¬ ∃ x : ℝ, Real.sqrt x > 3) ↔ (∀ x : ℝ, Real.sqrt x ≤ 3 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_gt_three_l337_33713


namespace NUMINAMATH_CALUDE_quadratic_max_value_l337_33721

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l337_33721


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l337_33735

theorem common_internal_tangent_length 
  (distance_between_centers : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : distance_between_centers = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt 2211 := by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l337_33735


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l337_33754

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l337_33754


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_binomial_expansion_l337_33737

theorem coefficient_x3y5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  (Nat.choose 8 5 : ℕ) = 56 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_binomial_expansion_l337_33737


namespace NUMINAMATH_CALUDE_price_change_l337_33779

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  original_price * (1 + 0.02) * (1 - 0.02) < original_price :=
by
  sorry

end NUMINAMATH_CALUDE_price_change_l337_33779


namespace NUMINAMATH_CALUDE_cherry_revenue_is_180_l337_33784

/-- Calculates the revenue from cherry pies given the total number of pies,
    the ratio of pie types, and the price of a cherry pie. -/
def cherry_pie_revenue (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) (cherry_price : ℕ) : ℕ :=
  let total_ratio := apple_ratio + blueberry_ratio + cherry_ratio
  let cherry_pies := (total_pies * cherry_ratio) / total_ratio
  cherry_pies * cherry_price

/-- Theorem stating that given 36 total pies with a ratio of 3:2:5 for apple:blueberry:cherry pies,
    and a price of $10 per cherry pie, the total revenue from cherry pies is $180. -/
theorem cherry_revenue_is_180 :
  cherry_pie_revenue 36 3 2 5 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cherry_revenue_is_180_l337_33784


namespace NUMINAMATH_CALUDE_tan_range_proof_l337_33786

theorem tan_range_proof (x : ℝ) (hx : x ∈ Set.Icc (-π/4) (π/4) ∧ x ≠ 0) :
  ∃ y, y = Real.tan (π/2 - x) ↔ y ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_tan_range_proof_l337_33786
