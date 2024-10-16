import Mathlib

namespace NUMINAMATH_CALUDE_new_lines_satisfy_axioms_l2958_295893

-- Define the type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the type for new lines (parabolas and vertical lines)
inductive NewLine
  | Parabola (a b : ℝ)  -- y = (x + a)² + b
  | VerticalLine (c : ℝ)  -- x = c

-- Define when a point lies on a new line
def lies_on (p : Point) (l : NewLine) : Prop :=
  match l with
  | NewLine.Parabola a b => p.y = (p.x + a)^2 + b
  | NewLine.VerticalLine c => p.x = c

-- Axiom 1: For any two distinct points, there exists a unique new line passing through them
axiom exists_unique_newline (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l

-- Axiom 2: Any two distinct new lines intersect in at most one point
axiom at_most_one_intersection (l1 l2 : NewLine) (h : l1 ≠ l2) :
  ∃! p : Point, lies_on p l1 ∧ lies_on p l2

-- Axiom 3: For any new line and a point not on it, there exists a unique new line
--          passing through the point and not intersecting the given line
axiom exists_unique_parallel (l : NewLine) (p : Point) (h : ¬lies_on p l) :
  ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')

-- Theorem: The set of new lines satisfies the three axioms
theorem new_lines_satisfy_axioms :
  (∀ p1 p2 : Point, p1 ≠ p2 → ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l) ∧
  (∀ l1 l2 : NewLine, l1 ≠ l2 → ∃! p : Point, lies_on p l1 ∧ lies_on p l2) ∧
  (∀ l : NewLine, ∀ p : Point, ¬lies_on p l →
    ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')) :=
by sorry

end NUMINAMATH_CALUDE_new_lines_satisfy_axioms_l2958_295893


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2958_295881

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {2,4,5}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1,3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2958_295881


namespace NUMINAMATH_CALUDE_m_range_l2958_295829

/-- The proposition p: The solution set of the inequality |x|+|x-1| > m is R -/
def p (m : ℝ) : Prop :=
  ∀ x, |x| + |x - 1| > m

/-- The proposition q: f(x)=(5-2m)^x is an increasing function -/
def q (m : ℝ) : Prop :=
  ∀ x y, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- The range of m given the conditions -/
theorem m_range :
  ∃ m, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2958_295829


namespace NUMINAMATH_CALUDE_vertical_line_angle_is_90_degrees_l2958_295843

/-- The angle of inclination of a vertical line -/
def angle_of_vertical_line : ℝ := 90

/-- A vertical line is defined by the equation x = 0 -/
def is_vertical_line (f : ℝ → ℝ) : Prop := ∀ y, f y = 0

theorem vertical_line_angle_is_90_degrees (f : ℝ → ℝ) (h : is_vertical_line f) :
  angle_of_vertical_line = 90 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_angle_is_90_degrees_l2958_295843


namespace NUMINAMATH_CALUDE_find_m_value_l2958_295802

theorem find_m_value (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 21) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_find_m_value_l2958_295802


namespace NUMINAMATH_CALUDE_radian_to_degree_conversion_l2958_295860

theorem radian_to_degree_conversion (π : Real) (h : π = 180) :
  (4 / 3 * π : Real) = 240 :=
sorry

end NUMINAMATH_CALUDE_radian_to_degree_conversion_l2958_295860


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_plus_three_l2958_295851

-- Define a function to calculate factorial
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the sum of factorials from 1 to 10
def factorialSum : ℕ := 
  List.sum (List.map factorial (List.range 10))

-- Theorem to prove
theorem units_digit_of_factorial_sum_plus_three : 
  unitsDigit (factorialSum + 3) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_plus_three_l2958_295851


namespace NUMINAMATH_CALUDE_bobby_jump_improvement_l2958_295810

/-- Bobby's jump rope ability as a child and adult -/
def bobby_jumps : ℕ × ℕ := (30, 60)

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
def jump_difference : ℕ := bobby_jumps.2 - bobby_jumps.1

theorem bobby_jump_improvement : jump_difference = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_jump_improvement_l2958_295810


namespace NUMINAMATH_CALUDE_complex_z_value_l2958_295821

-- Define the operation for 2x2 matrices
def matrixOp (a b c d : ℂ) : ℂ := a * d - b * c

-- Theorem statement
theorem complex_z_value (z : ℂ) :
  matrixOp z (1 - Complex.I) (1 + Complex.I) 1 = Complex.I →
  z = 2 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_z_value_l2958_295821


namespace NUMINAMATH_CALUDE_rainfall_second_week_l2958_295809

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 20 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_second_week_l2958_295809


namespace NUMINAMATH_CALUDE_occupancy_is_75_percent_l2958_295878

/-- Represents an apartment complex -/
structure ApartmentComplex where
  buildings : Nat
  studio_per_building : Nat
  two_person_per_building : Nat
  four_person_per_building : Nat
  current_occupancy : Nat

/-- Calculate the maximum occupancy of an apartment complex -/
def max_occupancy (complex : ApartmentComplex) : Nat :=
  complex.buildings * (complex.studio_per_building + 2 * complex.two_person_per_building + 4 * complex.four_person_per_building)

/-- Calculate the occupancy percentage of an apartment complex -/
def occupancy_percentage (complex : ApartmentComplex) : Rat :=
  (complex.current_occupancy : Rat) / (max_occupancy complex)

/-- The main theorem stating that the occupancy percentage is 75% -/
theorem occupancy_is_75_percent (complex : ApartmentComplex) 
  (h1 : complex.buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.current_occupancy = 210) :
  occupancy_percentage complex = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_occupancy_is_75_percent_l2958_295878


namespace NUMINAMATH_CALUDE_vertex_x_coordinate_l2958_295885

def f (x : ℝ) := 3 * x^2 + 9 * x + 5

theorem vertex_x_coordinate (x : ℝ) :
  x = -1.5 ↔ ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_vertex_x_coordinate_l2958_295885


namespace NUMINAMATH_CALUDE_monochromatic_4cycle_exists_l2958_295800

/-- A color for an edge -/
inductive Color
| Red
| Blue

/-- A graph with 6 vertices -/
def Graph6 := Fin 6 → Fin 6 → Color

/-- A 4-cycle in a graph -/
def IsCycle4 (g : Graph6) (v1 v2 v3 v4 : Fin 6) (c : Color) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v4 ∧ v4 ≠ v1 ∧
  g v1 v2 = c ∧ g v2 v3 = c ∧ g v3 v4 = c ∧ g v4 v1 = c

/-- The main theorem: every 6-vertex complete graph with red/blue edges contains a monochromatic 4-cycle -/
theorem monochromatic_4cycle_exists (g : Graph6) 
  (complete : ∀ u v : Fin 6, u ≠ v → (g u v = Color.Red ∨ g u v = Color.Blue)) :
  ∃ (v1 v2 v3 v4 : Fin 6) (c : Color), IsCycle4 g v1 v2 v3 v4 c :=
sorry

end NUMINAMATH_CALUDE_monochromatic_4cycle_exists_l2958_295800


namespace NUMINAMATH_CALUDE_haunted_mansion_entry_exit_l2958_295803

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the haunted mansion through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways to enter and exit the haunted mansion through different windows is 56 -/
theorem haunted_mansion_entry_exit : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_entry_exit_l2958_295803


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l2958_295877

-- Problem 1
theorem factorization_problem1 (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem2 (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l2958_295877


namespace NUMINAMATH_CALUDE_cats_to_dogs_ratio_l2958_295806

theorem cats_to_dogs_ratio (cats : ℕ) (dogs : ℕ) : 
  cats = 16 → dogs = 8 → (cats : ℚ) / dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_cats_to_dogs_ratio_l2958_295806


namespace NUMINAMATH_CALUDE_factorization_equality_l2958_295872

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2958_295872


namespace NUMINAMATH_CALUDE_product_of_roots_l2958_295826

theorem product_of_roots (p q r : ℂ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) →
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) →
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2958_295826


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_proof_l2958_295869

/-- The length of the diagonal of a rectangular prism with dimensions 10, 20, and 10 is 10√6 -/
theorem rectangular_prism_diagonal : ℝ → Prop :=
  fun diagonal =>
    ∃ (a b c : ℝ),
      a = 10 ∧ b = 20 ∧ c = 10 ∧
      diagonal = 10 * Real.sqrt 6 ∧
      diagonal^2 = a^2 + b^2 + c^2

/-- Proof of the theorem -/
theorem rectangular_prism_diagonal_proof : rectangular_prism_diagonal (10 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_proof_l2958_295869


namespace NUMINAMATH_CALUDE_prime_product_not_perfect_square_l2958_295819

/-- The nth prime number -/
def nth_prime (n : ℕ+) : ℕ := sorry

/-- The product of the first n prime numbers -/
def prime_product (n : ℕ+) : ℕ := sorry

/-- A natural number is a perfect square if it's equal to some integer squared -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- The product of the first n prime numbers is not a perfect square -/
theorem prime_product_not_perfect_square (n : ℕ+) : ¬ is_perfect_square (prime_product n) := by
  sorry

end NUMINAMATH_CALUDE_prime_product_not_perfect_square_l2958_295819


namespace NUMINAMATH_CALUDE_gift_box_volume_l2958_295812

/-- The volume of a rectangular box. -/
def boxVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a gift box with dimensions 9 cm wide, 4 cm long, and 7 cm high is 252 cubic centimeters. -/
theorem gift_box_volume :
  boxVolume 4 9 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_volume_l2958_295812


namespace NUMINAMATH_CALUDE_next_perfect_cube_l2958_295875

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧
  y = x * (x : ℝ).sqrt + 3 * x + 3 * (x : ℝ).sqrt + 1 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_cube_l2958_295875


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_25_l2958_295856

theorem largest_four_digit_congruent_to_15_mod_25 : ∃ (n : ℕ), 
  n ≤ 9990 ∧ 
  1000 ≤ n ∧ 
  n < 10000 ∧ 
  n ≡ 15 [MOD 25] ∧
  ∀ (m : ℕ), (1000 ≤ m ∧ m < 10000 ∧ m ≡ 15 [MOD 25]) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_25_l2958_295856


namespace NUMINAMATH_CALUDE_letians_estimate_l2958_295862

/-- Given x and y are positive real numbers with x > y, and z and w are small positive real numbers with z > w,
    prove that (x + z) - (y - w) > x - y. -/
theorem letians_estimate (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
    (hz : z > 0) (hw : w > 0) (hzw : z > w) : 
  (x + z) - (y - w) > x - y := by
  sorry

end NUMINAMATH_CALUDE_letians_estimate_l2958_295862


namespace NUMINAMATH_CALUDE_girls_in_algebra_class_l2958_295847

theorem girls_in_algebra_class (total : ℕ) (girls boys : ℕ) : 
  total = 84 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 48 := by
sorry

end NUMINAMATH_CALUDE_girls_in_algebra_class_l2958_295847


namespace NUMINAMATH_CALUDE_moe_has_least_money_l2958_295832

-- Define the set of people
inductive Person : Type
| Bo : Person
| Coe : Person
| Flo : Person
| Jo : Person
| Moe : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo
axiom bo_coe_more_than_moe : money Person.Bo > money Person.Moe ∧ money Person.Coe > money Person.Moe
axiom jo_between_bo_moe : money Person.Jo > money Person.Moe ∧ money Person.Jo < money Person.Bo

-- Define the theorem
theorem moe_has_least_money :
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l2958_295832


namespace NUMINAMATH_CALUDE_inequality_solution_and_a_range_l2958_295892

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∃ S : Set ℝ, S = {x : ℝ | -5/4 < x ∧ x < 1/2} ∧
    ∀ x, x ∈ S ↔ f x < 4 - |x - 1|) ∧
  ∀ m n : ℝ, m > 0 → n > 0 → m + n = 1 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_a_range_l2958_295892


namespace NUMINAMATH_CALUDE_x_minus_y_range_l2958_295899

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the range of x - y
def range_x_minus_y (x y : ℝ) : Prop :=
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10

-- Theorem statement
theorem x_minus_y_range :
  ∀ (x y ρ θ : ℝ), C ρ θ → x = ρ * Real.cos θ → y = ρ * Real.sin θ → range_x_minus_y x y :=
sorry

end NUMINAMATH_CALUDE_x_minus_y_range_l2958_295899


namespace NUMINAMATH_CALUDE_intersection_point_is_e_e_l2958_295871

theorem intersection_point_is_e_e (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x = Real.exp 1 ∧ y = Real.exp 1) →
  (x^y = y^x ∧ y = x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_e_e_l2958_295871


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2958_295880

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2958_295880


namespace NUMINAMATH_CALUDE_joannas_reading_time_l2958_295837

/-- Joanna's reading problem -/
theorem joannas_reading_time (
  total_pages : ℕ)
  (pages_per_hour : ℕ)
  (monday_hours : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_pages = 248)
  (h2 : pages_per_hour = 16)
  (h3 : monday_hours = 3)
  (h4 : remaining_hours = 6)
  : (total_pages - (monday_hours * pages_per_hour + remaining_hours * pages_per_hour)) / pages_per_hour = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_joannas_reading_time_l2958_295837


namespace NUMINAMATH_CALUDE_sadies_daily_burger_spending_l2958_295817

/-- Sadie's daily burger spending in June -/
def daily_burger_spending (total_spending : ℚ) (days : ℕ) : ℚ :=
  total_spending / days

theorem sadies_daily_burger_spending :
  let total_spending : ℚ := 372
  let days : ℕ := 30
  daily_burger_spending total_spending days = 12.4 := by
  sorry

end NUMINAMATH_CALUDE_sadies_daily_burger_spending_l2958_295817


namespace NUMINAMATH_CALUDE_marble_problem_l2958_295807

theorem marble_problem (a : ℕ) 
  (angela : ℕ) 
  (brian : ℕ) 
  (caden : ℕ) 
  (daryl : ℕ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 2 * brian) 
  (h4 : daryl = 5 * caden) 
  (h5 : angela + brian + caden + daryl = 120) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l2958_295807


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l2958_295835

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l2958_295835


namespace NUMINAMATH_CALUDE_factorization_equality_l2958_295857

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2958_295857


namespace NUMINAMATH_CALUDE_hawks_score_l2958_295854

theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 82 →
  eagles_points - hawks_points = 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 32 := by
sorry

end NUMINAMATH_CALUDE_hawks_score_l2958_295854


namespace NUMINAMATH_CALUDE_sum_floor_series_l2958_295848

theorem sum_floor_series (n : ℕ+) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋) = n := by sorry

end NUMINAMATH_CALUDE_sum_floor_series_l2958_295848


namespace NUMINAMATH_CALUDE_function_roots_imply_a_range_l2958_295870

theorem function_roots_imply_a_range (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * (x + 1) - 2 = x) → (∃ y z : ℝ, y ≠ z ∧ a * y^2 + b * (y + 1) - 2 = y ∧ a * z^2 + b * (z + 1) - 2 = z)) →
  (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_function_roots_imply_a_range_l2958_295870


namespace NUMINAMATH_CALUDE_arrangements_count_l2958_295873

/-- The number of ways to arrange 2 objects out of 2 positions -/
def A_2_2 : ℕ := 2

/-- The number of ways to arrange 2 objects out of 3 positions -/
def A_3_2 : ℕ := 6

/-- The number of ways to bind A and B together -/
def bind_AB : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := 5

/-- Theorem: The number of arrangements of 5 people where A and B must stand next to each other,
    and C and D cannot stand next to each other, is 24. -/
theorem arrangements_count : 
  bind_AB * A_2_2 * A_3_2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2958_295873


namespace NUMINAMATH_CALUDE_passenger_arrangement_l2958_295883

def arrange_passengers (n : ℕ) (r : ℕ) : ℕ :=
  -- Define the function to calculate the number of arrangements
  sorry

theorem passenger_arrangement :
  arrange_passengers 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_passenger_arrangement_l2958_295883


namespace NUMINAMATH_CALUDE_triangle_segment_length_l2958_295859

structure Triangle :=
  (A B C : ℝ × ℝ)

def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem triangle_segment_length 
  (ABC : Triangle) 
  (D : ℝ × ℝ) 
  (h1 : angle ABC.B ABC.A D = 60)
  (h2 : angle ABC.A ABC.B ABC.C = 30)
  (h3 : angle ABC.B ABC.C D = 30)
  (h4 : Real.sqrt ((ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2) = 15)
  (h5 : Real.sqrt ((ABC.C.1 - D.1)^2 + (ABC.C.2 - D.2)^2) = 8) :
  Real.sqrt ((ABC.A.1 - D.1)^2 + (ABC.A.2 - D.2)^2) = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_segment_length_l2958_295859


namespace NUMINAMATH_CALUDE_ratio_of_sums_eleven_l2958_295836

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 0 + seq.a (n - 1)) / 2

theorem ratio_of_sums_eleven (a b : ArithmeticSequence)
    (h : ∀ n, a.a n / b.a n = (2 * n - 1) / (n + 1)) :
  sum_n a 11 / sum_n b 11 = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_eleven_l2958_295836


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2958_295820

/-- Given that x varies inversely as the square of y, prove that x = 2.25 when y = 2,
    given that y = 3 when x = 1. -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as the square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (2.25 = k / (2^2))              -- x = 2.25 when y = 2
  := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2958_295820


namespace NUMINAMATH_CALUDE_trapezoid_circles_problem_l2958_295840

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are concyclic -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop := sorry

/-- Check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculate the ratio of two line segments -/
def segment_ratio (p1 p2 p3 : Point) : ℝ := sorry

theorem trapezoid_circles_problem 
  (A B C D E : Point) 
  (circle1 circle2 : Circle) 
  (line_CD : Line) :
  are_parallel A D B C →
  E.x > B.x ∧ E.x < C.x →
  are_concyclic A C D E →
  circle2.center = circle1.center →
  is_tangent circle2 line_CD →
  distance A B = 12 →
  segment_ratio B E C = 4/5 →
  distance B C = 36 ∧ 
  2/3 < circle1.radius / circle2.radius ∧ 
  circle1.radius / circle2.radius < 4/3 := by sorry

end NUMINAMATH_CALUDE_trapezoid_circles_problem_l2958_295840


namespace NUMINAMATH_CALUDE_pizzas_successfully_served_l2958_295831

theorem pizzas_successfully_served 
  (total_served : ℕ) 
  (returned : ℕ) 
  (h1 : total_served = 9) 
  (h2 : returned = 6) : 
  total_served - returned = 3 :=
by sorry

end NUMINAMATH_CALUDE_pizzas_successfully_served_l2958_295831


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l2958_295844

theorem unique_quadratic_pair : ∃! (b c : ℕ+), 
  (∃! x : ℝ, x^2 + b*x + c = 0) ∧ 
  (∃! x : ℝ, x^2 + c*x + b = 0) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l2958_295844


namespace NUMINAMATH_CALUDE_circumcenters_not_concyclic_l2958_295834

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to get the circumcenter of a triangle
def circumcenter (p1 p2 p3 : Point) : Point := sorry

-- Define a function to check if points are distinct
def areDistinct (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define a function to check if points are concyclic
def areConcyclic (p1 p2 p3 p4 : Point) : Prop := sorry

-- Theorem statement
theorem circumcenters_not_concyclic (q : Quadrilateral) 
  (h_convex : isConvex q)
  (O_A : Point) (O_B : Point) (O_C : Point) (O_D : Point)
  (h_O_A : O_A = circumcenter q.B q.C q.D)
  (h_O_B : O_B = circumcenter q.C q.D q.A)
  (h_O_C : O_C = circumcenter q.D q.A q.B)
  (h_O_D : O_D = circumcenter q.A q.B q.C)
  (h_distinct : areDistinct O_A O_B O_C O_D) :
  ¬(areConcyclic O_A O_B O_C O_D) := by
  sorry

end NUMINAMATH_CALUDE_circumcenters_not_concyclic_l2958_295834


namespace NUMINAMATH_CALUDE_expression_value_l2958_295879

theorem expression_value (a b : ℤ) (ha : a = -3) (hb : b = 2) :
  -a - b^3 + a*b = -11 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2958_295879


namespace NUMINAMATH_CALUDE_third_term_zero_l2958_295818

/-- A sequence where each term is the sum of corresponding terms from two geometric progressions -/
def GeometricSumSequence (u₁ v₁ q p : ℝ) : ℕ → ℝ
  | 0 => u₁ + v₁
  | 1 => u₁ * q + v₁ * p
  | n + 2 => u₁ * q^(n+2) + v₁ * p^(n+2)

/-- Theorem: If the first two terms of a GeometricSumSequence are 0, then the third term is also 0 -/
theorem third_term_zero (u₁ v₁ q p : ℝ) 
  (h1 : GeometricSumSequence u₁ v₁ q p 0 = 0)
  (h2 : GeometricSumSequence u₁ v₁ q p 1 = 0) :
  GeometricSumSequence u₁ v₁ q p 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_third_term_zero_l2958_295818


namespace NUMINAMATH_CALUDE_circle_radius_l2958_295898

/-- Circle with center (3, -5) and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 5)^2 = r^2}

/-- Line 4x - 3y - 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 - 2 = 0}

/-- The shortest distance from a point on the circle to the line is 1 -/
def ShortestDistance (r : ℝ) : Prop :=
  ∃ p ∈ Circle r, ∀ q ∈ Circle r, ∀ l ∈ Line,
    dist p l ≤ dist q l ∧ dist p l = 1

theorem circle_radius (r : ℝ) :
  ShortestDistance r → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2958_295898


namespace NUMINAMATH_CALUDE_cupcakes_leftover_l2958_295863

/-- Proves that given 40 cupcakes, after distributing to two classes and four individuals, 2 cupcakes remain. -/
theorem cupcakes_leftover (total : ℕ) (class1 : ℕ) (class2 : ℕ) (additional : ℕ) : 
  total = 40 → class1 = 18 → class2 = 16 → additional = 4 → 
  total - (class1 + class2 + additional) = 2 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_leftover_l2958_295863


namespace NUMINAMATH_CALUDE_pablo_blocks_sum_l2958_295855

/-- The number of blocks in Pablo's toy block stacks -/
def pablo_blocks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => pablo_blocks 0 + 2  -- Second stack
| 2 => pablo_blocks 1 - 5  -- Third stack
| 3 => pablo_blocks 2 + 5  -- Fourth stack
| _ => 0  -- No more stacks

/-- The total number of blocks used by Pablo -/
def total_blocks : ℕ := pablo_blocks 0 + pablo_blocks 1 + pablo_blocks 2 + pablo_blocks 3

theorem pablo_blocks_sum : total_blocks = 21 := by
  sorry

end NUMINAMATH_CALUDE_pablo_blocks_sum_l2958_295855


namespace NUMINAMATH_CALUDE_harry_bought_apples_l2958_295897

/-- The number of apples Harry initially had -/
def initial_apples : ℕ := 79

/-- The number of apples Harry ended up with -/
def final_apples : ℕ := 84

/-- The number of apples Harry bought -/
def bought_apples : ℕ := final_apples - initial_apples

theorem harry_bought_apples :
  bought_apples = final_apples - initial_apples :=
by sorry

end NUMINAMATH_CALUDE_harry_bought_apples_l2958_295897


namespace NUMINAMATH_CALUDE_simplify_expression_l2958_295895

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 5 * 7 = 5*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2958_295895


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l2958_295886

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -16384 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l2958_295886


namespace NUMINAMATH_CALUDE_c_minus_a_equals_40_l2958_295842

theorem c_minus_a_equals_40
  (a b c d e : ℝ)
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60)
  (h3 : (d + e) / 2 = 80)
  (h4 : (a * b * d) = (b * c * e)) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_c_minus_a_equals_40_l2958_295842


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l2958_295822

/-- A configuration of four mutually tangent spheres on a plane -/
structure SphericalConfiguration where
  radius : ℝ
  mutually_tangent : Bool
  on_plane : Bool

/-- A tetrahedron circumscribed around four spheres -/
structure CircumscribedTetrahedron where
  spheres : SphericalConfiguration
  edge_length : ℝ

/-- The theorem stating that the edge length of a tetrahedron circumscribed around
    four mutually tangent spheres of radius 2 is equal to 4 -/
theorem tetrahedron_edge_length 
  (config : SphericalConfiguration) 
  (tetra : CircumscribedTetrahedron) :
  config.radius = 2 ∧ 
  config.mutually_tangent = true ∧ 
  config.on_plane = true ∧
  tetra.spheres = config →
  tetra.edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l2958_295822


namespace NUMINAMATH_CALUDE_sperm_genotypes_l2958_295846

-- Define the possible alleles
inductive Allele
| A
| a
| Xb
| Y

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the initial spermatogonial cell genotype
def initialGenotype : Genotype := [Allele.A, Allele.a, Allele.Xb, Allele.Y]

-- Define the genotype of the abnormal sperm
def abnormalSperm : Genotype := [Allele.A, Allele.A, Allele.a, Allele.Xb]

-- Define the function to check if a list of genotypes is valid
def isValidResult (sperm1 sperm2 sperm3 : Genotype) : Prop :=
  sperm1 = [Allele.a, Allele.Xb] ∧
  sperm2 = [Allele.Y] ∧
  sperm3 = [Allele.Y]

-- State the theorem
theorem sperm_genotypes (initialCell : Genotype) (abnormalSperm : Genotype) :
  initialCell = initialGenotype →
  abnormalSperm = abnormalSperm →
  ∃ (sperm1 sperm2 sperm3 : Genotype), isValidResult sperm1 sperm2 sperm3 :=
sorry

end NUMINAMATH_CALUDE_sperm_genotypes_l2958_295846


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l2958_295853

theorem stationery_box_sheets (S E : ℕ) : 
  S - (S / 3 + 50) = 50 →
  E = S / 3 + 50 →
  S = 150 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l2958_295853


namespace NUMINAMATH_CALUDE_restaurant_earnings_l2958_295888

theorem restaurant_earnings : 
  let meals_1 := 10
  let price_1 := 8
  let meals_2 := 5
  let price_2 := 10
  let meals_3 := 20
  let price_3 := 4
  meals_1 * price_1 + meals_2 * price_2 + meals_3 * price_3 = 210 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_earnings_l2958_295888


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l2958_295830

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations for perpendicular and parallel
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : perpendicular l α)
  (h3 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l2958_295830


namespace NUMINAMATH_CALUDE_total_time_to_school_l2958_295841

def time_to_gate : ℕ := 15
def time_gate_to_building : ℕ := 6
def time_building_to_room : ℕ := 9

theorem total_time_to_school :
  time_to_gate + time_gate_to_building + time_building_to_room = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_time_to_school_l2958_295841


namespace NUMINAMATH_CALUDE_solve_equation_l2958_295891

theorem solve_equation (x : ℝ) : (3 * x - 7) / 4 = 14 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2958_295891


namespace NUMINAMATH_CALUDE_largest_fraction_l2958_295828

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 7/15, 9/20, 11/25]
  ∀ x ∈ fractions, (7:ℚ)/15 ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2958_295828


namespace NUMINAMATH_CALUDE_hidden_dots_count_l2958_295814

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- The total number of dots on three dice -/
def total_dots : ℕ := 3 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 5 + 6

/-- The number of hidden dots on the dice -/
def hidden_dots : ℕ := total_dots - visible_dots

theorem hidden_dots_count : hidden_dots = 41 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l2958_295814


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l2958_295852

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 13) ∧
  (10 * (n % 10) + (n / 10) = n - 27) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l2958_295852


namespace NUMINAMATH_CALUDE_wilson_family_ages_l2958_295805

theorem wilson_family_ages : ∃ (w e j h t d : ℕ),
  (w > 0) ∧ (e > 0) ∧ (j > 0) ∧ (h > 0) ∧ (t > 0) ∧ (d > 0) ∧
  (w / 2 = e + j + h) ∧
  (w + 5 = (e + 5) + (j + 5) + (h + 5) + 0) ∧
  (e + j + h + t + d = 2 * w) ∧
  (w = e + j) ∧
  (e = t + d) := by
  sorry

end NUMINAMATH_CALUDE_wilson_family_ages_l2958_295805


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2958_295808

/-- The polynomial p(x) = x^3 - 4x^2 + 3x + 2 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 2

/-- The remainder when p(x) is divided by (x - 1) -/
def remainder : ℝ := p 1

theorem polynomial_remainder : remainder = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2958_295808


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2958_295827

/-- Given an infinite geometric series with common ratio 1/4 and sum 48,
    the second term of the sequence is 9. -/
theorem second_term_of_geometric_series :
  ∀ (a : ℝ), -- first term of the series
  let r : ℝ := (1 : ℝ) / 4 -- common ratio
  let S : ℝ := 48 -- sum of the series
  (S = a / (1 - r)) → -- formula for sum of infinite geometric series
  (a * r = 9) -- second term of the sequence
  := by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2958_295827


namespace NUMINAMATH_CALUDE_union_complement_problem_l2958_295839

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 4}
def B : Finset Nat := {2, 5}

theorem union_complement_problem : B ∪ (U \ A) = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l2958_295839


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_l2958_295813

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x² + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

theorem even_function_implies_b_zero (b : ℝ) :
  IsEven (f b) → b = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_l2958_295813


namespace NUMINAMATH_CALUDE_square_difference_equality_l2958_295801

theorem square_difference_equality : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2958_295801


namespace NUMINAMATH_CALUDE_jack_heavier_than_sam_l2958_295866

theorem jack_heavier_than_sam (total_weight jack_weight : ℕ) 
  (h1 : total_weight = 96)
  (h2 : jack_weight = 52) :
  jack_weight - (total_weight - jack_weight) = 8 :=
by sorry

end NUMINAMATH_CALUDE_jack_heavier_than_sam_l2958_295866


namespace NUMINAMATH_CALUDE_no_rational_roots_l2958_295815

def polynomial (x : ℚ) : ℚ :=
  3 * x^5 + 4 * x^4 - 5 * x^3 - 15 * x^2 + 7 * x + 3

theorem no_rational_roots :
  ∀ (x : ℚ), polynomial x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2958_295815


namespace NUMINAMATH_CALUDE_convex_polyhedron_structure_l2958_295838

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- Represents a face of a polyhedron -/
structure Face where
  sides : Nat

/-- Represents a vertex of a polyhedron -/
structure Vertex where
  edges : Nat

/-- Definition of a convex polyhedron with its faces and vertices -/
def ConvexPolyhedronWithFacesAndVertices (p : ConvexPolyhedron) (faces : List Face) (vertices : List Vertex) : Prop :=
  p.convex ∧ faces.length > 0 ∧ vertices.length > 0

/-- Theorem stating that not all faces can have more than 3 sides 
    and not all vertices can have more than 3 edges simultaneously -/
theorem convex_polyhedron_structure 
  (p : ConvexPolyhedron) 
  (faces : List Face) 
  (vertices : List Vertex) 
  (h : ConvexPolyhedronWithFacesAndVertices p faces vertices) :
  ¬(∀ f ∈ faces, f.sides > 3 ∧ ∀ v ∈ vertices, v.edges > 3) :=
by sorry

end NUMINAMATH_CALUDE_convex_polyhedron_structure_l2958_295838


namespace NUMINAMATH_CALUDE_juan_reads_9000_pages_l2958_295894

/-- Calculates the total pages Juan can read from three books given their page counts, reading rates, and lunch time constraints. -/
def total_pages_read (book1_pages book2_pages book3_pages : ℕ) 
                     (book1_rate book2_rate book3_rate : ℕ) 
                     (lunch_time : ℕ) : ℕ :=
  let book1_read_time := book1_pages / book1_rate
  let book2_read_time := book2_pages / book2_rate
  let book3_read_time := book3_pages / book3_rate
  let book1_lunch_time := book1_read_time / 2
  let book2_lunch_time := book2_read_time / 2
  let book3_lunch_time := book3_read_time / 2
  let total_lunch_time := book1_lunch_time + book2_lunch_time + book3_lunch_time
  let remaining_time1 := book1_lunch_time - lunch_time
  let remaining_time2 := book2_lunch_time
  let remaining_time3 := book3_lunch_time
  remaining_time1 * book1_rate + remaining_time2 * book2_rate + remaining_time3 * book3_rate

/-- Theorem stating that given the specific conditions in the problem, Juan can read 9000 pages. -/
theorem juan_reads_9000_pages : 
  total_pages_read 4000 6000 8000 60 40 30 4 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_juan_reads_9000_pages_l2958_295894


namespace NUMINAMATH_CALUDE_minimum_correct_answers_l2958_295858

def test_score (correct : ℕ) : ℤ :=
  4 * correct - (25 - correct)

theorem minimum_correct_answers : 
  ∀ x : ℕ, x ≤ 25 → test_score x > 70 → x ≥ 19 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_correct_answers_l2958_295858


namespace NUMINAMATH_CALUDE_pizza_slices_ordered_l2958_295882

/-- The number of friends Ron ate pizza with -/
def num_friends : ℕ := 2

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating pizza (Ron + his friends) -/
def total_people : ℕ := num_friends + 1

/-- Theorem: The total number of pizza slices ordered is at least 12 -/
theorem pizza_slices_ordered (num_friends : ℕ) (slices_per_person : ℕ) (total_people : ℕ) :
  num_friends = 2 →
  slices_per_person = 4 →
  total_people = num_friends + 1 →
  total_people * slices_per_person ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_ordered_l2958_295882


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l2958_295850

-- Define the cost of toy cars
def toy_cars_cost : ℚ := 14.88

-- Define the cost of toy trucks
def toy_trucks_cost : ℚ := 5.86

-- Define the total cost of toys
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

-- Theorem to prove
theorem total_spent_on_toys :
  total_toys_cost = 20.74 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l2958_295850


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2958_295876

/-- The number of diagonals in a convex n-gon --/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex heptagon has 14 diagonals --/
theorem heptagon_diagonals : numDiagonals 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2958_295876


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l2958_295896

theorem polynomial_coefficient_B (E F G : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+),
    (∀ z : ℂ, z^6 - 15*z^5 + E*z^4 + (-287)*z^3 + F*z^2 + G*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l2958_295896


namespace NUMINAMATH_CALUDE_complement_union_problem_l2958_295845

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem complement_union_problem (A B : Set ℕ) 
  (h1 : (U \ A) ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : (U \ A) ∩ (U \ B) = {2}) :
  U \ (A ∪ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2958_295845


namespace NUMINAMATH_CALUDE_markese_earnings_l2958_295823

theorem markese_earnings (E : ℕ) (h1 : E + (E - 5) = 37) : E - 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_markese_earnings_l2958_295823


namespace NUMINAMATH_CALUDE_black_rhinos_count_l2958_295816

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The weight of each white rhino in pounds -/
def weight_white_rhino : ℕ := 5100

/-- The weight of each black rhino in pounds -/
def weight_black_rhino : ℕ := 2000

/-- The total weight of all rhinos in pounds -/
def total_weight : ℕ := 51700

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := (total_weight - num_white_rhinos * weight_white_rhino) / weight_black_rhino

theorem black_rhinos_count : num_black_rhinos = 8 := by sorry

end NUMINAMATH_CALUDE_black_rhinos_count_l2958_295816


namespace NUMINAMATH_CALUDE_expand_product_l2958_295864

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2958_295864


namespace NUMINAMATH_CALUDE_wages_problem_l2958_295825

/-- Given a sum of money that can pay b's wages for 28 days and both a's and b's wages for 12 days,
    prove that it can pay a's wages for 21 days. -/
theorem wages_problem (S : ℝ) (Wa Wb : ℝ) (S_pays_b_28_days : S = 28 * Wb) 
    (S_pays_both_12_days : S = 12 * (Wa + Wb)) : S = 21 * Wa := by
  sorry

end NUMINAMATH_CALUDE_wages_problem_l2958_295825


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_degrees_l2958_295849

theorem sin_cos_sum_fifteen_seventyfive_degrees : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_degrees_l2958_295849


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2958_295884

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.mk (-2) 1
  is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2958_295884


namespace NUMINAMATH_CALUDE_solution_concentration_change_l2958_295867

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of the solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc replacement_conc fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced)) + (replacement_conc * fraction_replaced)

/-- Theorem stating that replacing 0.7142857142857143 of a 60% solution with a 25% solution
    results in a new concentration of 0.21285714285714285 -/
theorem solution_concentration_change : 
  new_concentration 0.60 0.25 0.7142857142857143 = 0.21285714285714285 := by sorry

end NUMINAMATH_CALUDE_solution_concentration_change_l2958_295867


namespace NUMINAMATH_CALUDE_tournament_probability_l2958_295865

/-- The number of teams in the tournament -/
def num_teams : ℕ := 35

/-- The number of games each team plays -/
def games_per_team : ℕ := num_teams - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := (num_teams * games_per_team) / 2

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 1 / 2

/-- The number of possible outcomes in the tournament -/
def total_outcomes : ℕ := 2^total_games

/-- The number of ways to assign unique victory counts to all teams -/
def unique_victory_assignments : ℕ := num_teams.factorial

theorem tournament_probability : 
  (unique_victory_assignments : ℚ) / total_outcomes = (num_teams.factorial : ℚ) / 2^595 :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l2958_295865


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2958_295868

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π/2) (3*π/4)) :
  Real.sqrt (2 - 2 * Real.sin (2 * α)) - Real.sqrt (1 + Real.cos (2 * α)) = Real.sqrt 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2958_295868


namespace NUMINAMATH_CALUDE_apple_box_weight_l2958_295811

theorem apple_box_weight (n : ℕ) (w : ℝ) (h1 : n = 5) (h2 : w > 30) 
  (h3 : n * (w - 30) = 2 * w) : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l2958_295811


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2958_295804

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2958_295804


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l2958_295874

/-- Represents a number in base n -/
def BaseN (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₁ d₀ : ℕ), x = d₁ * n + d₀ ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ) (a b : ℕ) 
  (h_n : n > 9)
  (h_root : n^2 - a*n + b = 0)
  (h_a : BaseN n 19) :
  BaseN n 90 := by
  sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l2958_295874


namespace NUMINAMATH_CALUDE_train_speed_ratio_l2958_295824

theorem train_speed_ratio : 
  ∀ (c h : ℝ), 
  c > 0 → h > 0 →
  ∃ (x : ℝ), 
  x > 1 ∧
  x = (h / ((x - 1) * c)) / (h / ((1 + x) * c)) ∧
  x = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l2958_295824


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2958_295861

/-- The trajectory of point M(x,y) satisfying the distance condition -/
def trajectory_equation (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2)^(1/2) = |x + 3| + 1

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_is_parabola (x y : ℝ) :
  trajectory_equation x y → y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2958_295861


namespace NUMINAMATH_CALUDE_subtract_and_multiply_l2958_295890

theorem subtract_and_multiply (N V : ℝ) : N = 12 → (4 * N - 3 = 9 * (N - V)) → V = 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_and_multiply_l2958_295890


namespace NUMINAMATH_CALUDE_project_popularity_order_l2958_295833

def park_renovation : ℚ := 9 / 24
def new_library : ℚ := 10 / 30
def street_lighting : ℚ := 7 / 21
def community_garden : ℚ := 8 / 24

theorem project_popularity_order :
  park_renovation > community_garden ∧
  community_garden = new_library ∧
  new_library = street_lighting ∧
  park_renovation > new_library :=
by sorry

end NUMINAMATH_CALUDE_project_popularity_order_l2958_295833


namespace NUMINAMATH_CALUDE_f_100_equals_2_l2958_295887

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else Real.log x

-- Theorem statement
theorem f_100_equals_2 : f 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_100_equals_2_l2958_295887


namespace NUMINAMATH_CALUDE_harry_weekly_earnings_l2958_295889

/-- Represents Harry's dog-walking schedule and earnings --/
structure DogWalker where
  mon_wed_fri_dogs : ℕ
  tuesday_dogs : ℕ
  thursday_dogs : ℕ
  pay_per_dog : ℕ

/-- Calculates the weekly earnings of a dog walker --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  (3 * dw.mon_wed_fri_dogs + dw.tuesday_dogs + dw.thursday_dogs) * dw.pay_per_dog

/-- Harry's specific dog-walking schedule --/
def harry : DogWalker :=
  { mon_wed_fri_dogs := 7
    tuesday_dogs := 12
    thursday_dogs := 9
    pay_per_dog := 5 }

/-- Theorem stating Harry's weekly earnings --/
theorem harry_weekly_earnings :
  weekly_earnings harry = 210 := by
  sorry

end NUMINAMATH_CALUDE_harry_weekly_earnings_l2958_295889
