import Mathlib

namespace NUMINAMATH_CALUDE_average_rate_of_change_cubic_l589_58915

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Theorem statement
theorem average_rate_of_change_cubic (a b : ℝ) (h : a = 1 ∧ b = 2) :
  (f b - f a) / (b - a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_cubic_l589_58915


namespace NUMINAMATH_CALUDE_coeff_x6_q_cubed_is_15_l589_58982

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^4 + 5*x^2 - 4*x + 1

/-- The coefficient of x^6 in (q(x))^3 -/
def coeff_x6_q_cubed : ℝ := 15

/-- Theorem: The coefficient of x^6 in (q(x))^3 is 15 -/
theorem coeff_x6_q_cubed_is_15 : coeff_x6_q_cubed = 15 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x6_q_cubed_is_15_l589_58982


namespace NUMINAMATH_CALUDE_carla_marbles_l589_58949

/-- The number of marbles Carla bought -/
def marbles_bought (initial final : ℕ) : ℕ := final - initial

/-- Proof that Carla bought 134 marbles -/
theorem carla_marbles : marbles_bought 53 187 = 134 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l589_58949


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l589_58999

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular region -/
def area (dim : RectDimensions) : ℝ := dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four
  (rug : RugRegions)
  (inner_width_two : rug.inner.width = 2)
  (middle_wider_by_two : rug.middle.length = rug.inner.length + 4 ∧ rug.middle.width = rug.inner.width + 4)
  (outer_wider_by_two : rug.outer.length = rug.middle.length + 4 ∧ rug.outer.width = rug.middle.width + 4)
  (areas_in_arithmetic_progression : isArithmeticProgression (area rug.inner) (area rug.middle) (area rug.outer)) :
  rug.inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l589_58999


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l589_58920

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (m n : Line) (α β γ : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : α ≠ γ)
  (h4 : β ≠ γ)
  (h5 : perpendicular m β)
  (h6 : parallel m α) :
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l589_58920


namespace NUMINAMATH_CALUDE_vector_addition_proof_l589_58991

def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

theorem vector_addition_proof : 
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l589_58991


namespace NUMINAMATH_CALUDE_num_chords_is_45_num_triangles_is_120_l589_58923

/- Define the combination function -/
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Define the number of points on the circle -/
def num_points : ℕ := 10

/- Theorem for the number of chords -/
theorem num_chords_is_45 : combination num_points 2 = 45 := by sorry

/- Theorem for the number of triangles -/
theorem num_triangles_is_120 : combination num_points 3 = 120 := by sorry

end NUMINAMATH_CALUDE_num_chords_is_45_num_triangles_is_120_l589_58923


namespace NUMINAMATH_CALUDE_molecular_weight_problem_l589_58932

/-- Given that 3 moles of a compound have a molecular weight of 222,
    prove that the molecular weight of 1 mole of the compound is 74. -/
theorem molecular_weight_problem (moles : ℕ) (total_weight : ℝ) :
  moles = 3 →
  total_weight = 222 →
  total_weight / moles = 74 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_problem_l589_58932


namespace NUMINAMATH_CALUDE_inequality_proof_l589_58947

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x < 1) (hy1 : y < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l589_58947


namespace NUMINAMATH_CALUDE_characterization_of_functions_l589_58902

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

end NUMINAMATH_CALUDE_characterization_of_functions_l589_58902


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l589_58935

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + m*y + m = 0 → y = x) ↔ 
  (m = 0 ∨ m = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l589_58935


namespace NUMINAMATH_CALUDE_expression_simplification_l589_58980

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l589_58980


namespace NUMINAMATH_CALUDE_min_sticks_to_remove_8x8_l589_58914

/-- Represents a chessboard with sticks on edges -/
structure Chessboard :=
  (size : Nat)
  (sticks : Nat)

/-- The minimum number of sticks that must be removed to avoid rectangles -/
def min_sticks_to_remove (board : Chessboard) : Nat :=
  Nat.ceil (2 / 3 * (board.size * board.size))

/-- Theorem stating the minimum number of sticks to remove for an 8x8 chessboard -/
theorem min_sticks_to_remove_8x8 :
  let board : Chessboard := ⟨8, 144⟩
  min_sticks_to_remove board = 43 := by
  sorry

#eval min_sticks_to_remove ⟨8, 144⟩

end NUMINAMATH_CALUDE_min_sticks_to_remove_8x8_l589_58914


namespace NUMINAMATH_CALUDE_max_positive_condition_l589_58966

theorem max_positive_condition (a : ℝ) :
  (∀ x : ℝ, max (x^3 + 3*x + a - 9) (a + 2^(5-x) - 3^(x-1)) > 0) ↔ a > -5 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_condition_l589_58966


namespace NUMINAMATH_CALUDE_divisible_by_nine_unique_uphill_divisible_by_nine_l589_58934

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9. -/
theorem divisible_by_nine (n : ℕ) : n % 9 = 0 ↔ (n.digits 10).sum % 9 = 0 :=
sorry

/-- There is exactly one uphill integer divisible by 9. -/
theorem unique_uphill_divisible_by_nine : ∃! n : ℕ, is_uphill n ∧ n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_unique_uphill_divisible_by_nine_l589_58934


namespace NUMINAMATH_CALUDE_total_students_l589_58940

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

end NUMINAMATH_CALUDE_total_students_l589_58940


namespace NUMINAMATH_CALUDE_exists_terrorist_with_eleven_raids_l589_58933

/-- Represents a terrorist in the band -/
structure Terrorist : Type :=
  (id : Nat)

/-- Represents a raid -/
structure Raid : Type :=
  (id : Nat)

/-- Represents the participation of a terrorist in a raid -/
def Participation : Type := Terrorist → Raid → Prop

/-- The total number of terrorists in the band -/
def num_terrorists : Nat := 101

/-- Axiom: Each pair of terrorists has met exactly once in a raid -/
axiom met_once (p : Participation) (t1 t2 : Terrorist) :
  t1 ≠ t2 → ∃! r : Raid, p t1 r ∧ p t2 r

/-- Axiom: No two terrorists have participated in more than one raid together -/
axiom no_multiple_raids (p : Participation) (t1 t2 : Terrorist) (r1 r2 : Raid) :
  t1 ≠ t2 → p t1 r1 → p t2 r1 → p t1 r2 → p t2 r2 → r1 = r2

/-- Theorem: There exists a terrorist who participated in at least 11 different raids -/
theorem exists_terrorist_with_eleven_raids (p : Participation) :
  ∃ t : Terrorist, ∃ (raids : Finset Raid), raids.card ≥ 11 ∧ ∀ r ∈ raids, p t r :=
sorry

end NUMINAMATH_CALUDE_exists_terrorist_with_eleven_raids_l589_58933


namespace NUMINAMATH_CALUDE_green_mandm_probability_l589_58951

structure MandMJar :=
  (green : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) (yellow : ℕ) (purple : ℕ) (brown : ℕ) (pink : ℕ)

def initial_jar : MandMJar :=
  ⟨35, 25, 10, 15, 0, 0, 0, 0⟩

def carter_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green - 20, jar.red - 8, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink⟩

def sister_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red / 2, jar.blue, jar.orange, jar.yellow + 14, jar.purple, jar.brown, jar.pink⟩

def alex_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, 0, jar.yellow - 3, jar.purple + 8, jar.brown, jar.pink⟩

def cousin_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, 0, jar.orange, jar.yellow, jar.purple, jar.brown + 10, jar.pink⟩

def sister_adds_pink (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink + 10⟩

def total_mandms (jar : MandMJar) : ℕ :=
  jar.green + jar.red + jar.blue + jar.orange + jar.yellow + jar.purple + jar.brown + jar.pink

theorem green_mandm_probability :
  let final_jar := sister_adds_pink (cousin_eats (alex_eats (sister_eats (carter_eats initial_jar))))
  (final_jar.green : ℚ) / ((total_mandms final_jar - 1) : ℚ) = 15 / 61 := by sorry

end NUMINAMATH_CALUDE_green_mandm_probability_l589_58951


namespace NUMINAMATH_CALUDE_min_x_plus_3y_min_xy_l589_58969

-- Define the conditions
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1

-- Theorem for the minimum value of x + 3y
theorem min_x_plus_3y (x y : ℝ) (h : condition x y) : 
  x + 3*y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), condition x₀ y₀ ∧ x₀ + 3*y₀ = 16 :=
sorry

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) : 
  x*y ≥ 12 ∧ ∃ (x₀ y₀ : ℝ), condition x₀ y₀ ∧ x₀*y₀ = 12 :=
sorry

end NUMINAMATH_CALUDE_min_x_plus_3y_min_xy_l589_58969


namespace NUMINAMATH_CALUDE_money_left_after_game_l589_58925

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hot_dog_cost : ℕ := 3

theorem money_left_after_game : 
  initial_amount - (ticket_cost + hot_dog_cost) = 9 := by sorry

end NUMINAMATH_CALUDE_money_left_after_game_l589_58925


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l589_58946

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, n > 0 ∧ Nat.Prime n ∧ (∃ m : ℕ, n = m^2 - 12) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < n → ¬(Nat.Prime k ∧ ∃ m : ℕ, k = m^2 - 12)) ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l589_58946


namespace NUMINAMATH_CALUDE_pattern_symmetries_l589_58950

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point

/-- Represents the pattern on the line -/
structure Pattern where
  line : Line
  unit_length : ℝ  -- Length of one repeating unit

/-- Represents a rigid motion transformation -/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (direction : Point) (distance : ℝ)
  | Reflection (line : Line)

/-- Checks if a transformation preserves the pattern -/
def preserves_pattern (p : Pattern) (t : RigidMotion) : Prop :=
  sorry

theorem pattern_symmetries (p : Pattern) :
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (2 * π / 3))) ∧
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (4 * π / 3))) ∧
  (preserves_pattern p (RigidMotion.Translation p.line.direction p.unit_length)) ∧
  (preserves_pattern p (RigidMotion.Reflection p.line)) ∧
  (∃ (perp_line : Line), 
    (perp_line.direction.x * p.line.direction.x + perp_line.direction.y * p.line.direction.y = 0) ∧
    preserves_pattern p (RigidMotion.Reflection perp_line)) :=
by sorry

end NUMINAMATH_CALUDE_pattern_symmetries_l589_58950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triple_sums_l589_58908

/-- Given an arithmetic sequence {a_n}, the sequence formed by sums of consecutive triples
    (a_1 + a_2 + a_3), (a_4 + a_5 + a_6), (a_7 + a_8 + a_9), ... is also an arithmetic sequence. -/
theorem arithmetic_sequence_triple_sums (a : ℕ → ℝ) (d : ℝ) 
    (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d' : ℝ, ∀ n, (a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) + d' = 
    (a (3*(n+1) + 1) + a (3*(n+1) + 2) + a (3*(n+1) + 3)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triple_sums_l589_58908


namespace NUMINAMATH_CALUDE_david_pushups_count_l589_58968

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 19

/-- The difference between Zachary's and David's crunches -/
def crunch_difference : ℕ := 27

/-- David's push-ups -/
def david_pushups : ℕ := zachary_pushups + pushup_difference

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l589_58968


namespace NUMINAMATH_CALUDE_trees_in_garden_l589_58971

/-- 
Given a yard of length 600 meters with trees planted at equal distances, 
one tree at each end, and 24 meters between consecutive trees, 
prove that the total number of trees planted is 26.
-/
theorem trees_in_garden (yard_length : ℕ) (tree_spacing : ℕ) (trees : ℕ) : 
  yard_length = 600 →
  tree_spacing = 24 →
  trees = yard_length / tree_spacing + 1 →
  trees = 26 := by
sorry

end NUMINAMATH_CALUDE_trees_in_garden_l589_58971


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l589_58927

theorem equation_satisfies_condition (x y z : ℤ) :
  x = y ∧ y = z + 1 → x^2 - x*y + y^2 - y*z + z^2 - z*x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l589_58927


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l589_58983

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_expression_simplification :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l589_58983


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l589_58906

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt 2 = Real.sqrt (4^x * 2^y) ∧ 2/x + 1/y = 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l589_58906


namespace NUMINAMATH_CALUDE_equilateral_triangle_point_distances_l589_58978

theorem equilateral_triangle_point_distances 
  (h x y z : ℝ) 
  (h_pos : h > 0)
  (inside_triangle : x > 0 ∧ y > 0 ∧ z > 0)
  (height_sum : h = x + y + z)
  (triangle_inequality : x + y > z ∧ y + z > x ∧ z + x > y) :
  x < h/2 ∧ y < h/2 ∧ z < h/2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_point_distances_l589_58978


namespace NUMINAMATH_CALUDE_five_fold_f_of_three_equals_eight_l589_58972

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem five_fold_f_of_three_equals_eight : f (f (f (f (f 3)))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_fold_f_of_three_equals_eight_l589_58972


namespace NUMINAMATH_CALUDE_candy_bar_difference_l589_58919

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 → 
  lena + 5 = 3 * kevin → 
  nicole = kevin + 4 → 
  lena - nicole = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l589_58919


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l589_58985

/-- The probability of picking two red balls from a bag -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12) (h2 : red_balls = 5) : 
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 5 / 33 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l589_58985


namespace NUMINAMATH_CALUDE_intersection_point_l589_58959

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*Real.cos θ + 3 = 0 ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi

/-- Curve C2 in parametric form -/
def C2 (x y t : ℝ) : Prop := x = t * Real.cos (Real.pi/6) ∧ y = t * Real.sin (Real.pi/6)

/-- The intersection point of C1 and C2 has polar coordinates (√3, π/6) -/
theorem intersection_point : 
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ (∃ (x y t : ℝ), C2 x y t ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
  ρ = Real.sqrt 3 ∧ θ = Real.pi/6 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l589_58959


namespace NUMINAMATH_CALUDE_sum_of_squares_l589_58974

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -21) : 
  x^2 + y^2 + z^2 = 83/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l589_58974


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l589_58941

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

end NUMINAMATH_CALUDE_melanie_dimes_count_l589_58941


namespace NUMINAMATH_CALUDE_time_after_12345_seconds_l589_58956

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  valid : hours < 24 ∧ minutes < 60 ∧ seconds < 60

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem time_after_12345_seconds : 
  addSeconds ⟨18, 15, 0, sorry⟩ 12345 = ⟨21, 40, 45, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_12345_seconds_l589_58956


namespace NUMINAMATH_CALUDE_alpha_tan_beta_gt_beta_tan_alpha_l589_58993

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.tan β > β * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_alpha_tan_beta_gt_beta_tan_alpha_l589_58993


namespace NUMINAMATH_CALUDE_volume_right_triangular_prism_l589_58996

/-- The volume of a right triangular prism given its lateral face areas and lateral edge length -/
theorem volume_right_triangular_prism
  (M N P l : ℝ)
  (hM : M > 0)
  (hN : N > 0)
  (hP : P > 0)
  (hl : l > 0) :
  let V := (1 / (4 * l)) * Real.sqrt ((N + M + P) * (N + P - M) * (N + M - P) * (M + P - N))
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    M = a * l ∧
    N = b * l ∧
    P = c * l ∧
    V = (1 / 2) * l * Real.sqrt ((-a + b + c) * (a - b + c) * (a + b - c) * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_volume_right_triangular_prism_l589_58996


namespace NUMINAMATH_CALUDE_robot_energy_cells_l589_58903

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The robot's reported energy cell count in base 7 -/
def robotReport : List Nat := [1, 2, 3]

theorem robot_energy_cells :
  base7ToBase10 robotReport = 162 := by
  sorry

end NUMINAMATH_CALUDE_robot_energy_cells_l589_58903


namespace NUMINAMATH_CALUDE_solve_euro_equation_l589_58911

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_euro_equation (x : ℝ) :
  (euro 6 (euro x 5) = 480) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l589_58911


namespace NUMINAMATH_CALUDE_right_triangle_area_l589_58938

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l589_58938


namespace NUMINAMATH_CALUDE_sequential_search_element_count_l589_58973

/-- Represents a sequential search in an unordered array -/
structure SequentialSearch where
  n : ℕ  -- number of elements in the array
  avg_comparisons : ℕ  -- average number of comparisons

/-- 
  Theorem: If the average number of comparisons in a sequential search 
  of an unordered array is 100, and the searched element is not in the array, 
  then the number of elements in the array is 200.
-/
theorem sequential_search_element_count 
  (search : SequentialSearch) 
  (h1 : search.avg_comparisons = 100) 
  (h2 : search.avg_comparisons = search.n / 2) : 
  search.n = 200 := by
  sorry

#check sequential_search_element_count

end NUMINAMATH_CALUDE_sequential_search_element_count_l589_58973


namespace NUMINAMATH_CALUDE_average_favorable_draws_l589_58987

def lottery_size : ℕ := 90
def draw_size : ℕ := 5

def favorable_draws : ℕ :=
  (86^2 * 85) / 2 + 87 * 85 + 86

def total_draws : ℕ :=
  lottery_size * (lottery_size - 1) * (lottery_size - 2) * (lottery_size - 3) * (lottery_size - 4) / 120

theorem average_favorable_draws :
  (total_draws : ℚ) / favorable_draws = 5874 / 43 := by sorry

end NUMINAMATH_CALUDE_average_favorable_draws_l589_58987


namespace NUMINAMATH_CALUDE_second_term_is_half_l589_58945

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 1 / 4
  property : a 3 * a 5 = 4 * (a 4 - 1)

/-- The second term of the geometric sequence is 1/2 -/
theorem second_term_is_half (seq : GeometricSequence) : seq.a 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_half_l589_58945


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_negation_l589_58994

theorem negation_of_existence_is_forall_negation :
  (¬ ∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_negation_l589_58994


namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l589_58939

def factorial_sum_last_two_digits : ℕ := 46

theorem last_two_digits_of_factorial_sum :
  let sum := List.sum (List.map Nat.factorial (List.range 25 |>.map (fun i => 4 * i + 3)))
  (sum % 100 = factorial_sum_last_two_digits) :=
by
  sorry

theorem factorial_ends_with_zeros (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℕ, Nat.factorial n = 100 * k :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l589_58939


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l589_58942

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3*i) - 3*(2 - 5*i) = 1 + 12*i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l589_58942


namespace NUMINAMATH_CALUDE_solution_to_inequalities_l589_58962

theorem solution_to_inequalities :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  (11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3) ∧ (x - 4 * y ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_inequalities_l589_58962


namespace NUMINAMATH_CALUDE_largest_value_l589_58921

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (a^2 + b^2) (max (2*a*b) (max a (1/2))) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l589_58921


namespace NUMINAMATH_CALUDE_removal_gives_desired_average_l589_58904

def original_list : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def removed_number : ℕ := 5
def desired_average : ℚ := 10.5

theorem removal_gives_desired_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = desired_average := by
  sorry

end NUMINAMATH_CALUDE_removal_gives_desired_average_l589_58904


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_31_l589_58907

theorem unique_number_with_three_prime_divisors_including_31 (x n : ℕ) :
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧
    x = p * q * 31 ∧ 
    ∀ r : ℕ, Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 31)) →
  x = 32767 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_31_l589_58907


namespace NUMINAMATH_CALUDE_cubic_identity_l589_58988

theorem cubic_identity : ∀ x : ℝ, (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l589_58988


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l589_58963

/-- The sum of six consecutive integers starting from n is equal to 6n + 15 -/
theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l589_58963


namespace NUMINAMATH_CALUDE_number_of_factors_of_N_l589_58929

def N : ℕ := 17^3 + 3 * 17^2 + 3 * 17 + 1

theorem number_of_factors_of_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_N_l589_58929


namespace NUMINAMATH_CALUDE_square_root_of_product_l589_58952

theorem square_root_of_product : Real.sqrt (64 * Real.sqrt 49) = 8 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_l589_58952


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l589_58901

/-- Calculates the cost of soap for a year given the duration and price of a single bar -/
def soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℚ) : ℚ :=
  (12 / months_per_bar) * price_per_bar

/-- Theorem stating that for soap lasting 2 months and costing $8.00, the yearly cost is $48.00 -/
theorem soap_cost_theorem : soap_cost_for_year 2 8 = 48 := by
  sorry

#eval soap_cost_for_year 2 8

end NUMINAMATH_CALUDE_soap_cost_theorem_l589_58901


namespace NUMINAMATH_CALUDE_divisibility_implication_l589_58931

theorem divisibility_implication (m : ℕ+) (h : 39 ∣ m^2) : 39 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l589_58931


namespace NUMINAMATH_CALUDE_peach_difference_l589_58967

def red_peaches : ℕ := 5
def green_peaches : ℕ := 11

theorem peach_difference : green_peaches - red_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l589_58967


namespace NUMINAMATH_CALUDE_complement_of_union_l589_58957

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l589_58957


namespace NUMINAMATH_CALUDE_a_investment_is_800_l589_58992

/-- Represents the investment and profit scenario of three business partners -/
structure BusinessScenario where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  investment_period : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- The business scenario with given conditions -/
def given_scenario : BusinessScenario :=
  { a_investment := 0,  -- Unknown, to be solved
    b_investment := 1000,
    c_investment := 1200,
    investment_period := 2,
    total_profit := 1000,
    c_profit_share := 400 }

/-- Theorem stating that a's investment in the given scenario is 800 -/
theorem a_investment_is_800 (scenario : BusinessScenario) 
  (h1 : scenario = given_scenario) :
  scenario.a_investment = 800 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_is_800_l589_58992


namespace NUMINAMATH_CALUDE_bhaskar_tour_days_l589_58979

def total_budget : ℕ := 360
def extension_days : ℕ := 4
def expense_reduction : ℕ := 3

theorem bhaskar_tour_days :
  ∃ (x : ℕ), x > 0 ∧
  (total_budget / x : ℚ) - expense_reduction = (total_budget / (x + extension_days) : ℚ) ∧
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bhaskar_tour_days_l589_58979


namespace NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l589_58913

/-- The number of children who got on the bus at a stop -/
def ChildrenWhoGotOn (initial final : ℕ) : ℕ := final - initial

theorem children_who_got_on_bus_stop (initial final : ℕ) 
  (h1 : initial = 52) 
  (h2 : final = 76) : 
  ChildrenWhoGotOn initial final = 24 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l589_58913


namespace NUMINAMATH_CALUDE_savings_distribution_child_receives_1680_l589_58909

/-- Calculates the amount each child receives from a couple's savings --/
theorem savings_distribution (husband_weekly : ℕ) (wife_weekly : ℕ) 
  (months : ℕ) (weeks_per_month : ℕ) (num_children : ℕ) : ℕ :=
  let total_savings := (husband_weekly + wife_weekly) * weeks_per_month * months
  let half_savings := total_savings / 2
  half_savings / num_children

/-- Proves that each child receives $1680 given the specific conditions --/
theorem child_receives_1680 : 
  savings_distribution 335 225 6 4 4 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_savings_distribution_child_receives_1680_l589_58909


namespace NUMINAMATH_CALUDE_arithmetic_heptagon_angle_l589_58981

/-- Represents a heptagon with angles in arithmetic progression -/
structure ArithmeticHeptagon where
  -- First angle of the progression
  a : ℝ
  -- Common difference of the progression
  d : ℝ
  -- Constraint: Sum of angles in a heptagon is 900°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) + (a + 6*d) = 900

/-- Theorem: In a heptagon with angles in arithmetic progression, one of the angles can be 128.57° -/
theorem arithmetic_heptagon_angle (h : ArithmeticHeptagon) : 
  ∃ k : Fin 7, h.a + k * h.d = 128.57 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_heptagon_angle_l589_58981


namespace NUMINAMATH_CALUDE_triangle_construction_pieces_l589_58975

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Properties of the triangle construction -/
structure TriangleConstruction where
  rodRows : ℕ
  firstRowRods : ℕ
  rodIncrease : ℕ
  connectorRows : ℕ

/-- Theorem statement for the triangle construction problem -/
theorem triangle_construction_pieces 
  (t : TriangleConstruction) 
  (h1 : t.rodRows = 10)
  (h2 : t.firstRowRods = 4)
  (h3 : t.rodIncrease = 4)
  (h4 : t.connectorRows = t.rodRows + 1) :
  arithmeticSum t.firstRowRods t.rodIncrease t.rodRows + triangularNumber t.connectorRows = 286 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_pieces_l589_58975


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l589_58912

/-- Represents a connection between two circle indices -/
def Connection := Fin 5 × Fin 5

/-- Checks if two circles are connected -/
def is_connected (connections : List Connection) (i j : Fin 5) : Prop :=
  (i, j) ∈ connections ∨ (j, i) ∈ connections

/-- The theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (numbers : Fin 5 → ℕ+) (connections : List Connection),
  (∀ i j : Fin 5, is_connected connections i j →
    (numbers i).val / (numbers j).val = 3 ∨ (numbers i).val / (numbers j).val = 9) ∧
  (∀ i j : Fin 5, ¬is_connected connections i j →
    (numbers i).val / (numbers j).val ≠ 3 ∧ (numbers i).val / (numbers j).val ≠ 9) :=
by sorry


end NUMINAMATH_CALUDE_exists_valid_arrangement_l589_58912


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_379_l589_58910

theorem sqrt_product_plus_one_equals_379 :
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_379_l589_58910


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_a_for_nonempty_solution_l589_58916

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for part I
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part II
theorem range_a_for_nonempty_solution :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_a_for_nonempty_solution_l589_58916


namespace NUMINAMATH_CALUDE_girls_joined_team_l589_58905

/-- Proves that 7 girls joined the track team given the initial and final conditions --/
theorem girls_joined_team (initial_girls : ℕ) (initial_boys : ℕ) (boys_quit : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → initial_boys = 15 → boys_quit = 4 → final_total = 36 →
  final_total - (initial_girls + (initial_boys - boys_quit)) = 7 := by
sorry

end NUMINAMATH_CALUDE_girls_joined_team_l589_58905


namespace NUMINAMATH_CALUDE_inequality_proof_l589_58944

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l589_58944


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l589_58955

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l589_58955


namespace NUMINAMATH_CALUDE_fraction_multiplication_l589_58917

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 7 = (1 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l589_58917


namespace NUMINAMATH_CALUDE_maaza_amount_l589_58989

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

end NUMINAMATH_CALUDE_maaza_amount_l589_58989


namespace NUMINAMATH_CALUDE_vertical_distance_traveled_l589_58970

/-- Calculate the total vertical distance traveled in a week -/
theorem vertical_distance_traveled (story : Nat) (trips_per_day : Nat) (feet_per_story : Nat) (days_in_week : Nat) : 
  story = 5 → trips_per_day = 3 → feet_per_story = 10 → days_in_week = 7 →
  2 * story * feet_per_story * trips_per_day * days_in_week = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_distance_traveled_l589_58970


namespace NUMINAMATH_CALUDE_circle_area_through_points_l589_58958

/-- The area of a circle with center P and passing through Q is 149π -/
theorem circle_area_through_points (P Q : ℝ × ℝ) : 
  P = (-2, 3) → Q = (8, -4) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 149 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l589_58958


namespace NUMINAMATH_CALUDE_no_lower_grade_possible_l589_58976

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  as_earned : ℕ

/-- Theorem stating that Lisa cannot earn a grade lower than A on any remaining quiz --/
theorem no_lower_grade_possible (perf : QuizPerformance) 
  (h1 : perf.total_quizzes = 60)
  (h2 : perf.goal_percentage = 85 / 100)
  (h3 : perf.completed_quizzes = 35)
  (h4 : perf.as_earned = 25) :
  (perf.total_quizzes - perf.completed_quizzes : ℚ) - 
  (↑⌈perf.goal_percentage * perf.total_quizzes⌉ - perf.as_earned) ≤ 0 := by
  sorry

#eval ⌈(85 : ℚ) / 100 * 60⌉ -- Expected output: 51

end NUMINAMATH_CALUDE_no_lower_grade_possible_l589_58976


namespace NUMINAMATH_CALUDE_place_value_difference_l589_58953

def number : ℝ := 135.21

def hundreds_place_value : ℝ := 100
def tenths_place_value : ℝ := 0.1

theorem place_value_difference : 
  hundreds_place_value - tenths_place_value = 99.9 := by
  sorry

end NUMINAMATH_CALUDE_place_value_difference_l589_58953


namespace NUMINAMATH_CALUDE_divisible_by_four_and_six_percentage_l589_58986

theorem divisible_by_four_and_six_percentage (n : ℕ) : 
  (↑(Finset.filter (fun x => x % 4 = 0 ∧ x % 6 = 0) (Finset.range (n + 1))).card / n) * 100 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_and_six_percentage_l589_58986


namespace NUMINAMATH_CALUDE_line_equation_with_triangle_area_l589_58926

/-- The equation of a line passing through two points and forming a triangle -/
theorem line_equation_with_triangle_area 
  (b S : ℝ) (hb : b ≠ 0) (hS : S > 0) :
  let k := 2 * S / b
  let line_eq := fun (x y : ℝ) ↦ 2 * S * x - b^2 * y + 2 * b * S
  (∀ y, line_eq (-b) y = 0) ∧ 
  (∀ x, line_eq x k = 0) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ line_eq x y = 0) ∧
  (S = (1/2) * b * k) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_triangle_area_l589_58926


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l589_58954

theorem sum_of_specific_numbers : 7.52 + 12.23 = 19.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l589_58954


namespace NUMINAMATH_CALUDE_equation_solution_l589_58977

theorem equation_solution : ∃ x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l589_58977


namespace NUMINAMATH_CALUDE_race_runners_count_l589_58984

theorem race_runners_count : ∃ n : ℕ, 
  n > 5 ∧ 
  5 * 8 + (n - 5) * 10 = 70 ∧ 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_race_runners_count_l589_58984


namespace NUMINAMATH_CALUDE_train_journey_time_l589_58936

/-- Proves that the current time taken to cover a distance is 50 minutes 
    given the conditions from the train problem. -/
theorem train_journey_time : 
  ∀ (distance : ℝ) (current_time : ℝ),
    distance > 0 →
    distance = 48 * (current_time / 60) →
    distance = 60 * (40 / 60) →
    current_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l589_58936


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l589_58960

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) 
  (h1 : total = 850)
  (h2 : muslim_percent = 46/100)
  (h3 : sikh_percent = 10/100)
  (h4 : other = 136) :
  (total - (muslim_percent * total).num - (sikh_percent * total).num - other) / total = 28/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l589_58960


namespace NUMINAMATH_CALUDE_unique_friend_groups_l589_58900

theorem unique_friend_groups (n : ℕ) (h : n = 10) : 
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by
  sorry

end NUMINAMATH_CALUDE_unique_friend_groups_l589_58900


namespace NUMINAMATH_CALUDE_johns_pool_depth_l589_58997

theorem johns_pool_depth (sarah_depth john_depth : ℕ) : 
  sarah_depth = 5 →
  john_depth = 2 * sarah_depth + 5 →
  john_depth = 15 := by
  sorry

end NUMINAMATH_CALUDE_johns_pool_depth_l589_58997


namespace NUMINAMATH_CALUDE_larger_ball_radius_l589_58965

/-- The radius of a larger steel ball formed from the same amount of material as 12 smaller balls -/
theorem larger_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) 
  (h1 : small_radius = 2)
  (h2 : num_small_balls = 12) : 
  ∃ (large_radius : ℝ), large_radius^3 = num_small_balls * small_radius^3 :=
by sorry

end NUMINAMATH_CALUDE_larger_ball_radius_l589_58965


namespace NUMINAMATH_CALUDE_child_admission_price_l589_58937

theorem child_admission_price
  (total_people : ℕ)
  (adult_price : ℚ)
  (total_receipts : ℚ)
  (num_children : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : total_receipts = 960)
  (h4 : num_children = 260) :
  (total_receipts - (adult_price * (total_people - num_children))) / num_children = 1 :=
by sorry

end NUMINAMATH_CALUDE_child_admission_price_l589_58937


namespace NUMINAMATH_CALUDE_probability_of_condition_l589_58998

-- Define the bounds for x and y
def x_lower : ℝ := 0
def x_upper : ℝ := 4
def y_lower : ℝ := 0
def y_upper : ℝ := 7

-- Define the condition
def condition (x y : ℝ) : Prop := x + y ≤ 5

-- Define the probability function
def probability : ℝ := sorry

-- Theorem statement
theorem probability_of_condition : probability = 3/7 := by sorry

end NUMINAMATH_CALUDE_probability_of_condition_l589_58998


namespace NUMINAMATH_CALUDE_min_value_of_sum_l589_58961

theorem min_value_of_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x) + (x*z/y) + (x*y/z) ≥ Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀^2 + y₀^2 + z₀^2 = 1 ∧
    (y₀*z₀/x₀) + (x₀*z₀/y₀) + (x₀*y₀/z₀) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l589_58961


namespace NUMINAMATH_CALUDE_xiao_zhang_four_vcd_probability_l589_58928

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

end NUMINAMATH_CALUDE_xiao_zhang_four_vcd_probability_l589_58928


namespace NUMINAMATH_CALUDE_coordinates_determine_location_kunming_location_determined_l589_58922

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

end NUMINAMATH_CALUDE_coordinates_determine_location_kunming_location_determined_l589_58922


namespace NUMINAMATH_CALUDE_stating_dodgeball_tournament_teams_l589_58918

/-- Represents the total points scored in a dodgeball tournament. -/
def total_points : ℕ := 1151

/-- Points awarded for a win in the tournament. -/
def win_points : ℕ := 15

/-- Points awarded for a tie in the tournament. -/
def tie_points : ℕ := 11

/-- Points awarded for a loss in the tournament. -/
def loss_points : ℕ := 0

/-- The number of teams in the tournament. -/
def num_teams : ℕ := 12

/-- 
Theorem stating that given the tournament conditions, 
the number of teams must be 12.
-/
theorem dodgeball_tournament_teams : 
  ∀ n : ℕ, 
    (n * (n - 1) / 2) * win_points ≤ total_points ∧ 
    total_points ≤ (n * (n - 1) / 2) * (win_points + tie_points) / 2 →
    n = num_teams :=
by sorry

end NUMINAMATH_CALUDE_stating_dodgeball_tournament_teams_l589_58918


namespace NUMINAMATH_CALUDE_skittles_distribution_l589_58943

theorem skittles_distribution (total_skittles : ℕ) (skittles_per_person : ℕ) (people : ℕ) :
  total_skittles = 20 →
  skittles_per_person = 2 →
  people * skittles_per_person = total_skittles →
  people = 10 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l589_58943


namespace NUMINAMATH_CALUDE_largest_inscribed_square_area_l589_58995

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 10

/-- The rhombus formed by two identical equilateral triangles -/
structure Rhombus where
  side : ℝ
  is_formed_by_triangles : side = triangle_side

/-- The largest square inscribed in the rhombus -/
def largest_inscribed_square (r : Rhombus) : ℝ := sorry

/-- Theorem stating that the area of the largest inscribed square is 50 -/
theorem largest_inscribed_square_area (r : Rhombus) :
  (largest_inscribed_square r) ^ 2 = 50 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_area_l589_58995


namespace NUMINAMATH_CALUDE_matrix_product_equals_C_l589_58990

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -2; 0, 2, 4]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 0, 2, -1; 3, 0, 1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, -3; -5, 5, -5; 12, 4, 2]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_equals_C_l589_58990


namespace NUMINAMATH_CALUDE_max_circle_sum_l589_58930

/-- Represents the seven regions formed by the intersection of three circles -/
inductive Region
| A  -- shared by all three circles
| B  -- shared by two circles
| C  -- shared by two circles
| D  -- shared by two circles
| E  -- in one circle only
| F  -- in one circle only
| G  -- in one circle only

/-- Assignment of integers to regions -/
def Assignment := Region → Fin 7

/-- A circle is represented by the four regions it contains -/
structure Circle :=
  (r1 r2 r3 r4 : Region)

/-- The three circles in the problem -/
def circles : Fin 3 → Circle := sorry

/-- The sum of values in a circle for a given assignment -/
def circleSum (a : Assignment) (c : Circle) : ℕ :=
  a c.r1 + a c.r2 + a c.r3 + a c.r4

/-- An assignment is valid if all values are distinct -/
def validAssignment (a : Assignment) : Prop :=
  ∀ r1 r2 : Region, r1 ≠ r2 → a r1 ≠ a r2

/-- An assignment satisfies the equal sum condition -/
def satisfiesEqualSum (a : Assignment) : Prop :=
  ∀ c1 c2 : Fin 3, circleSum a (circles c1) = circleSum a (circles c2)

/-- The maximum possible sum for each circle -/
def maxSum : ℕ := 15

theorem max_circle_sum :
  ∃ (a : Assignment), validAssignment a ∧ satisfiesEqualSum a ∧
  (∀ c : Fin 3, circleSum a (circles c) = maxSum) ∧
  (∀ (a' : Assignment), validAssignment a' ∧ satisfiesEqualSum a' →
    ∀ c : Fin 3, circleSum a' (circles c) ≤ maxSum) := by
  sorry

end NUMINAMATH_CALUDE_max_circle_sum_l589_58930


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l589_58964

theorem absolute_value_sum_difference : |(-8)| + (-6) - (-12) = 14 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l589_58964


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l589_58948

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 ∣ 2^2000) ↔ 
  (n = 7 ∨ n = 23) := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l589_58948


namespace NUMINAMATH_CALUDE_max_belts_is_five_l589_58924

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

end NUMINAMATH_CALUDE_max_belts_is_five_l589_58924
