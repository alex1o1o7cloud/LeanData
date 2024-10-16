import Mathlib

namespace NUMINAMATH_CALUDE_river_road_cars_l2385_238542

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →  -- ratio of buses to cars is 1:3
  cars = buses + 40 →           -- 40 fewer buses than cars
  cars = 60 :=                  -- prove that the number of cars is 60
by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l2385_238542


namespace NUMINAMATH_CALUDE_system_solution_condition_l2385_238568

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := Real.arccos ((4 + y) / 4) = Real.arccos (x - a)
def equation2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*x + 8*y = b

-- Define the condition for no more than one solution
def atMostOneSolution (a : ℝ) : Prop :=
  ∀ b : ℝ, ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y b

-- Theorem statement
theorem system_solution_condition (a : ℝ) :
  atMostOneSolution a ↔ a ≤ -15 ∨ a ≥ 19 := by sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2385_238568


namespace NUMINAMATH_CALUDE_oranges_kept_after_51_days_l2385_238526

/-- Calculates the number of sacks of oranges kept after a harvest period -/
def oranges_kept (sacks_harvested_per_day : ℕ) (sacks_discarded_per_day : ℕ) (harvest_days : ℕ) : ℕ :=
  (sacks_harvested_per_day - sacks_discarded_per_day) * harvest_days

/-- Proves that the number of sacks of oranges kept after 51 days of harvest is 153 -/
theorem oranges_kept_after_51_days :
  oranges_kept 74 71 51 = 153 := by
  sorry

end NUMINAMATH_CALUDE_oranges_kept_after_51_days_l2385_238526


namespace NUMINAMATH_CALUDE_factorization_proof_l2385_238539

theorem factorization_proof (a b : ℝ) : 4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2385_238539


namespace NUMINAMATH_CALUDE_highland_baseball_club_members_l2385_238574

/-- The cost of a pair of socks in dollars -/
def sockCost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tShirtAdditionalCost : ℕ := 7

/-- The total expenditure for all members in dollars -/
def totalExpenditure : ℕ := 5112

/-- Calculates the number of members in the Highland Baseball Club -/
def calculateMembers (sockCost tShirtAdditionalCost totalExpenditure : ℕ) : ℕ :=
  let tShirtCost := sockCost + tShirtAdditionalCost
  let capCost := sockCost
  let costPerMember := (sockCost + tShirtCost) + (sockCost + tShirtCost + capCost)
  totalExpenditure / costPerMember

theorem highland_baseball_club_members :
  calculateMembers sockCost tShirtAdditionalCost totalExpenditure = 116 := by
  sorry

end NUMINAMATH_CALUDE_highland_baseball_club_members_l2385_238574


namespace NUMINAMATH_CALUDE_chicken_eggs_per_chicken_l2385_238564

theorem chicken_eggs_per_chicken 
  (num_chickens : ℕ) 
  (num_cartons : ℕ) 
  (eggs_per_carton : ℕ) 
  (h1 : num_chickens = 20)
  (h2 : num_cartons = 10)
  (h3 : eggs_per_carton = 12) :
  (num_cartons * eggs_per_carton) / num_chickens = 6 :=
by sorry

end NUMINAMATH_CALUDE_chicken_eggs_per_chicken_l2385_238564


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2385_238569

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 5)) ↔ x ≠ 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2385_238569


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2385_238512

theorem system_of_equations_solution :
  -- System 1
  (∃ x y : ℚ, y = 2*x ∧ 3*y + 2*x = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℚ, x - 3*y = -2 ∧ 2*x + 3*y = 3 ∧ x = 1/3 ∧ y = 7/9) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2385_238512


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l2385_238565

/-- A line passing through two given points intersects the x-axis at a specific point. -/
theorem line_intersection_with_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 2 ∧ y₁ = 3) 
  (h_point2 : x₂ = -4 ∧ y₂ = 9) : 
  ∃ x : ℝ, x = 5 ∧ (y₂ - y₁) * x + (x₁ * y₂ - x₂ * y₁) = (y₂ - y₁) * x₁ := by
  sorry

#check line_intersection_with_x_axis

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l2385_238565


namespace NUMINAMATH_CALUDE_inequality_and_function_property_l2385_238596

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_function_property_l2385_238596


namespace NUMINAMATH_CALUDE_floor_subtraction_inequality_l2385_238506

theorem floor_subtraction_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end NUMINAMATH_CALUDE_floor_subtraction_inequality_l2385_238506


namespace NUMINAMATH_CALUDE_roots_distribution_l2385_238571

/-- The polynomial p(z) = z^6 + 6z + 10 -/
def p (z : ℂ) : ℂ := z^6 + 6*z + 10

/-- The number of roots of p(z) in the first quadrant -/
def roots_first_quadrant : ℕ := 1

/-- The number of roots of p(z) in the second quadrant -/
def roots_second_quadrant : ℕ := 2

/-- The number of roots of p(z) in the third quadrant -/
def roots_third_quadrant : ℕ := 2

/-- The number of roots of p(z) in the fourth quadrant -/
def roots_fourth_quadrant : ℕ := 1

theorem roots_distribution :
  (∃ (z : ℂ), z.re > 0 ∧ z.im > 0 ∧ p z = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im > 0 ∧ z2.re < 0 ∧ z2.im > 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im < 0 ∧ z2.re < 0 ∧ z2.im < 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z : ℂ), z.re > 0 ∧ z.im < 0 ∧ p z = 0) :=
sorry

end NUMINAMATH_CALUDE_roots_distribution_l2385_238571


namespace NUMINAMATH_CALUDE_only_sqrt_8_not_simplest_l2385_238514

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a radical is in its simplest form
def isSimplestForm (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → ¬isPerfectSquare m

-- Theorem statement
theorem only_sqrt_8_not_simplest : 
  isSimplestForm 10 ∧ 
  isSimplestForm 6 ∧ 
  isSimplestForm 2 ∧ 
  ¬isSimplestForm 8 :=
sorry

end NUMINAMATH_CALUDE_only_sqrt_8_not_simplest_l2385_238514


namespace NUMINAMATH_CALUDE_solution_set_equality_l2385_238581

def solution_set : Set ℝ := {x : ℝ | |x - 1| - |x - 5| < 2}

theorem solution_set_equality : solution_set = Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2385_238581


namespace NUMINAMATH_CALUDE_two_propositions_have_true_converse_l2385_238557

-- Define the propositions
def proposition1 := "Vertical angles are equal"
def proposition2 := "Supplementary angles of the same side are complementary, and two lines are parallel"
def proposition3 := "Corresponding angles of congruent triangles are equal"
def proposition4 := "If the squares of two real numbers are equal, then the two real numbers are equal"

-- Define a function to check if a proposition has a true converse
def hasValidConverse (p : String) : Bool :=
  match p with
  | "Vertical angles are equal" => false
  | "Supplementary angles of the same side are complementary, and two lines are parallel" => true
  | "Corresponding angles of congruent triangles are equal" => false
  | "If the squares of two real numbers are equal, then the two real numbers are equal" => true
  | _ => false

-- Theorem statement
theorem two_propositions_have_true_converse :
  (hasValidConverse proposition1).toNat +
  (hasValidConverse proposition2).toNat +
  (hasValidConverse proposition3).toNat +
  (hasValidConverse proposition4).toNat = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_have_true_converse_l2385_238557


namespace NUMINAMATH_CALUDE_parallel_segments_length_l2385_238573

/-- In a triangle with sides a, b, and c, if three segments parallel to the sides
    pass through one point and have equal length x, then x = (2abc) / (ab + ac + bc). -/
theorem parallel_segments_length (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  x = (2 * a * b * c) / (a * b + a * c + b * c) := by
  sorry

#check parallel_segments_length

end NUMINAMATH_CALUDE_parallel_segments_length_l2385_238573


namespace NUMINAMATH_CALUDE_lucky_lucy_calculation_l2385_238587

theorem lucky_lucy_calculation (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 3) (hr : r = 5) (hs : s = 8) : 
  (p - q - r - s + t = p - (q - (r - (s + t)))) ↔ t = 5 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lucy_calculation_l2385_238587


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l2385_238540

/-- 
Given a parabola y = ax^2 + bx + c with vertex (p, -p) and y-intercept (0, p), 
where p ≠ 0, the value of b is -4.
-/
theorem parabola_coefficient_b (a b c p : ℝ) : 
  p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l2385_238540


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l2385_238548

theorem max_value_sum_sqrt (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 63 ∧
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = Real.sqrt 63 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l2385_238548


namespace NUMINAMATH_CALUDE_stool_height_correct_l2385_238502

/-- The height of the ceiling in meters -/
def ceiling_height : ℝ := 2.4

/-- The height of Alice in meters -/
def alice_height : ℝ := 1.5

/-- The additional reach of Alice above her head in meters -/
def alice_reach : ℝ := 0.5

/-- The distance of the light bulb from the ceiling in meters -/
def bulb_distance : ℝ := 0.2

/-- The height of the stool in meters -/
def stool_height : ℝ := 0.2

theorem stool_height_correct : 
  alice_height + alice_reach + stool_height = ceiling_height - bulb_distance := by
  sorry

end NUMINAMATH_CALUDE_stool_height_correct_l2385_238502


namespace NUMINAMATH_CALUDE_frank_final_position_l2385_238570

def dance_sequence (start : Int) : Int :=
  let step1 := start - 5
  let step2 := step1 + 10
  let step3 := step2 - 2
  let step4 := step3 + (2 * 2)
  step4

theorem frank_final_position :
  dance_sequence 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_final_position_l2385_238570


namespace NUMINAMATH_CALUDE_product_sequence_sum_l2385_238508

theorem product_sequence_sum (c d : ℕ) (h1 : c / 3 = 12) (h2 : d = c - 1) : c + d = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l2385_238508


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l2385_238591

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem factor_divisor_statements : 
  (is_factor 5 25) ∧ 
  (is_divisor 19 209 ∧ ¬is_divisor 19 63) ∧ 
  (is_divisor 20 80) ∧ 
  (is_divisor 14 28 ∧ is_divisor 14 56) ∧ 
  (is_factor 7 140) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l2385_238591


namespace NUMINAMATH_CALUDE_initial_garrison_size_l2385_238553

/-- 
Given a garrison with provisions for a certain number of days, and information about
reinforcements and remaining provisions, this theorem proves the initial number of men.
-/
theorem initial_garrison_size 
  (initial_provision_days : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement_size : ℕ) 
  (remaining_provision_days : ℕ) 
  (h1 : initial_provision_days = 54)
  (h2 : days_before_reinforcement = 15)
  (h3 : reinforcement_size = 600)
  (h4 : remaining_provision_days = 30)
  : ∃ (initial_men : ℕ), 
    initial_men * (initial_provision_days - days_before_reinforcement) = 
    (initial_men + reinforcement_size) * remaining_provision_days ∧ 
    initial_men = 2000 :=
by sorry

end NUMINAMATH_CALUDE_initial_garrison_size_l2385_238553


namespace NUMINAMATH_CALUDE_tiles_on_floor_l2385_238525

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonal_tiles : ℕ

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem: For a rectangular floor with length twice the width and 25 tiles along the diagonal,
    the total number of tiles is 242. -/
theorem tiles_on_floor (floor : TiledFloor) 
    (h1 : floor.length = 2 * floor.width)
    (h2 : floor.diagonal_tiles = 25) :
    total_tiles floor = 242 := by
  sorry

#eval total_tiles { width := 11, length := 22, diagonal_tiles := 25 }

end NUMINAMATH_CALUDE_tiles_on_floor_l2385_238525


namespace NUMINAMATH_CALUDE_sector_radius_for_cone_l2385_238575

/-- 
Given a sector with central angle 120° used to form a cone with base radius 2 cm,
prove that the radius of the sector is 6 cm.
-/
theorem sector_radius_for_cone (θ : Real) (r_base : Real) (r_sector : Real) : 
  θ = 120 → r_base = 2 → (θ / 360) * (2 * Real.pi * r_sector) = 2 * Real.pi * r_base → r_sector = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_for_cone_l2385_238575


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2385_238556

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∃ c, ¬(a * c > b * c)) ∧ 
  (∀ c, a * c + c > b * c + c) ∧ 
  (∀ c, a - c^2 > b - c^2) ∧ 
  (∀ c, a + c^3 > b + c^3) ∧ 
  (∀ c, a * c^3 > b * c^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2385_238556


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2385_238594

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x => a^x) 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2385_238594


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l2385_238567

theorem restaurant_menu_combinations (menu_size : ℕ) (yann_order camille_order : ℕ) : 
  menu_size = 12 →
  yann_order ≠ camille_order →
  yann_order ≤ menu_size ∧ camille_order ≤ menu_size →
  (menu_size * (menu_size - 1) : ℕ) = 132 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l2385_238567


namespace NUMINAMATH_CALUDE_a_pow_b_greater_than_three_pow_n_over_n_l2385_238518

theorem a_pow_b_greater_than_three_pow_n_over_n 
  (a b n : ℕ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : Odd b) 
  (h4 : n ≥ 1) 
  (h5 : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a^b : ℚ) > (3^n : ℚ) / n := by
  sorry

end NUMINAMATH_CALUDE_a_pow_b_greater_than_three_pow_n_over_n_l2385_238518


namespace NUMINAMATH_CALUDE_unicorn_tower_theorem_l2385_238505

/-- Represents the configuration of a unicorn tethered to a cylindrical tower -/
structure UnicornTower where
  rope_length : ℝ
  tower_radius : ℝ
  unicorn_height : ℝ
  rope_end_distance : ℝ

/-- Calculates the length of rope touching the tower -/
def rope_touching_tower (ut : UnicornTower) : ℝ :=
  ut.rope_length - (ut.rope_end_distance + ut.tower_radius)

/-- Theorem stating the properties of the unicorn-tower configuration -/
theorem unicorn_tower_theorem (ut : UnicornTower) 
  (h_rope : ut.rope_length = 20)
  (h_radius : ut.tower_radius = 8)
  (h_height : ut.unicorn_height = 4)
  (h_distance : ut.rope_end_distance = 4) :
  ∃ (a b c : ℕ), 
    c.Prime ∧ 
    rope_touching_tower ut = (a : ℝ) - Real.sqrt b / c ∧
    a = 60 ∧ b = 750 ∧ c = 3 ∧
    a + b + c = 813 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_tower_theorem_l2385_238505


namespace NUMINAMATH_CALUDE_triangle_area_with_ratio_l2385_238532

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C,
    if (b+c):(c+a):(a+b) = 4:5:6 and b+c = 8, then the area of triangle ABC is 15√3/4 -/
theorem triangle_area_with_ratio (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 →
  (c + a) / (a + b) = 5 / 6 →
  b + c = 8 →
  (a + b + c) / 2 > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * a = b * b + c * c - 2 * b * c * Real.cos A →
  b * b = a * a + c * c - 2 * a * c * Real.cos B →
  c * c = a * a + b * b - 2 * a * b * Real.cos C →
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_ratio_l2385_238532


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l2385_238590

/-- Represents a 9x9 checkerboard with numbers from 1 to 81 -/
def Checkerboard : Type := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 +
  number_at board 0 8 +
  number_at board 8 0 +
  number_at board 8 8

/-- The theorem stating that the sum of the numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l2385_238590


namespace NUMINAMATH_CALUDE_power_of_product_l2385_238529

theorem power_of_product (a b : ℝ) : (a * b^3)^3 = a^3 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2385_238529


namespace NUMINAMATH_CALUDE_chord_length_at_135_degrees_chord_equation_when_bisected_l2385_238599

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define a point on the circle
def P : ℝ × ℝ := (-1, 2)

-- Define the chord AB
structure Chord where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  passes_through_P : A.1 ≠ B.1 ∧ (P.2 - A.2) / (P.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)

-- Theorem 1
theorem chord_length_at_135_degrees (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  (AB.B.2 - AB.A.2) / (AB.B.1 - AB.A.1) = -1 →
  Real.sqrt ((AB.A.1 - AB.B.1)^2 + (AB.A.2 - AB.B.2)^2) = Real.sqrt 30 :=
sorry

-- Theorem 2
theorem chord_equation_when_bisected (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  AB.A.1 - P.1 = P.1 - AB.B.1 →
  AB.A.2 - P.2 = P.2 - AB.B.2 →
  ∃ (a b c : ℝ), a * AB.A.1 + b * AB.A.2 + c = 0 ∧
                 a * AB.B.1 + b * AB.B.2 + c = 0 ∧
                 a = 1 ∧ b = -2 ∧ c = 5 :=
sorry

end NUMINAMATH_CALUDE_chord_length_at_135_degrees_chord_equation_when_bisected_l2385_238599


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2385_238579

/-- Given two infinite geometric series:
    Series I with first term a₁ = 12 and common ratio r₁ = 1/3
    Series II with first term a₂ = 12 and common ratio r₂ = (4+n)/12
    If the sum of Series II is five times the sum of Series I, then n = 152 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 12
  let r₁ : ℝ := 1/3
  let a₂ : ℝ := 12
  let r₂ : ℝ := (4+n)/12
  (a₁ / (1 - r₁) = a₂ / (1 - r₂) / 5) → n = 152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l2385_238579


namespace NUMINAMATH_CALUDE_range_of_a_l2385_238582

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically increasing on [0, +∞)
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) 
  (h_even : IsEven f) 
  (h_mono : IsMonoIncreasing f) 
  (h_ineq : f (a - 3) < f 4) : 
  -1 < a ∧ a < 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2385_238582


namespace NUMINAMATH_CALUDE_savings_calculation_l2385_238527

/-- Calculates the total savings over a given period with varying weekly savings rates. -/
def totalSavings (smartphoneCost monthlyGymMembership currentSavings savingsPeriod firstPeriodWeeklySavings secondPeriodWeeklySavings : ℕ) : ℕ :=
  let weeksInMonth := 4
  let firstPeriodMonths := savingsPeriod / 2
  let secondPeriodMonths := savingsPeriod - firstPeriodMonths
  let firstPeriodSavings := firstPeriodWeeklySavings * weeksInMonth * firstPeriodMonths
  let secondPeriodSavings := secondPeriodWeeklySavings * weeksInMonth * secondPeriodMonths
  firstPeriodSavings + secondPeriodSavings

/-- Proves that given the specified conditions, the total savings after four months is $1040. -/
theorem savings_calculation :
  let smartphoneCost := 800
  let monthlyGymMembership := 50
  let currentSavings := 200
  let savingsPeriod := 4
  let firstPeriodWeeklySavings := 50
  let secondPeriodWeeklySavings := 80
  totalSavings smartphoneCost monthlyGymMembership currentSavings savingsPeriod firstPeriodWeeklySavings secondPeriodWeeklySavings = 1040 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2385_238527


namespace NUMINAMATH_CALUDE_orange_count_difference_l2385_238541

/-- Proves that the difference between Marcie's and Brian's orange counts is 0 -/
theorem orange_count_difference (marcie_oranges brian_oranges : ℕ) 
  (h1 : marcie_oranges = 12) (h2 : brian_oranges = 12) : 
  marcie_oranges - brian_oranges = 0 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_difference_l2385_238541


namespace NUMINAMATH_CALUDE_circle_equation_after_translation_l2385_238576

/-- Given a circle C with parametric equations x = 2cos(θ) and y = 2 + 2sin(θ),
    and a translation of the origin to (1, 2), prove that the standard equation
    of the circle in the new coordinate system is (x' - 1)² + (y' - 4)² = 4 -/
theorem circle_equation_after_translation (θ : ℝ) (x y x' y' : ℝ) :
  (x = 2 * Real.cos θ) →
  (y = 2 + 2 * Real.sin θ) →
  (x' = x - 1) →
  (y' = y - 2) →
  (x' - 1)^2 + (y' - 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_after_translation_l2385_238576


namespace NUMINAMATH_CALUDE_parabola_point_inequality_l2385_238511

/-- Prove that for points on a parabola with given conditions, a specific inequality holds -/
theorem parabola_point_inequality (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = (x₁ - 2)^2 + k →
  y₂ = (x₂ - 2)^2 + k →
  x₂ > 2 →
  2 > x₁ →
  x₁ + x₂ < 4 →
  y₁ > y₂ ∧ y₂ > k :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_inequality_l2385_238511


namespace NUMINAMATH_CALUDE_system_solution_l2385_238510

theorem system_solution (x y z : ℝ) : 
  (x + y + x * y = 19 ∧ 
   y + z + y * z = 11 ∧ 
   z + x + z * x = 14) ↔ 
  ((x = 4 ∧ y = 3 ∧ z = 2) ∨ 
   (x = -6 ∧ y = -5 ∧ z = -4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2385_238510


namespace NUMINAMATH_CALUDE_triangle_properties_l2385_238555

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a + t.c)^2 = t.b^2 + 2 * Real.sqrt 3 * t.a * t.c * Real.sin t.C)
  (h2 : t.b = 8)
  (h3 : t.a > t.c)
  (h4 : 1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π/3 ∧ t.a = 5 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2385_238555


namespace NUMINAMATH_CALUDE_inequality_range_l2385_238562

theorem inequality_range (b : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2385_238562


namespace NUMINAMATH_CALUDE_coin_flip_problem_l2385_238536

theorem coin_flip_problem (n : ℕ) 
  (p_tails : ℚ) 
  (p_event : ℚ) : 
  p_tails = 1/2 → 
  p_event = 3125/100000 → 
  p_event = (p_tails^2) * ((1 - p_tails)^3) → 
  n ≥ 5 → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l2385_238536


namespace NUMINAMATH_CALUDE_regular_scoop_cost_l2385_238504

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (regular_scoop : ℚ) : Prop :=
  let kiddie_scoop : ℚ := 3
  let double_scoop : ℚ := 6
  let total_cost : ℚ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Two children
  let num_double : ℕ := 3   -- Three teenagers
  (num_regular * regular_scoop + 
   num_kiddie * kiddie_scoop + 
   num_double * double_scoop) = total_cost

theorem regular_scoop_cost : 
  ∃ (regular_scoop : ℚ), ice_cream_cost regular_scoop ∧ regular_scoop = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_scoop_cost_l2385_238504


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2385_238523

/-- The function y(x) satisfies the given differential equation. -/
theorem function_satisfies_equation (x : ℝ) (hx : x > 0) :
  let y : ℝ → ℝ := λ x => Real.tan (Real.log (3 * x))
  (1 + y x ^ 2) = x * (deriv y x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2385_238523


namespace NUMINAMATH_CALUDE_tommy_wheel_count_l2385_238533

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := truck_wheels * trucks_seen + car_wheels * cars_seen

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheel_count_l2385_238533


namespace NUMINAMATH_CALUDE_log_x_inequality_l2385_238531

theorem log_x_inequality (x : ℝ) (h : 1 < x ∧ x < 2) :
  ((Real.log x) / x)^2 < (Real.log x) / x ∧ (Real.log x) / x < (Real.log (x^2)) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_log_x_inequality_l2385_238531


namespace NUMINAMATH_CALUDE_choose_cooks_l2385_238516

theorem choose_cooks (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_l2385_238516


namespace NUMINAMATH_CALUDE_remaining_amount_l2385_238559

def initial_amount : ℝ := 100.00
def spent_amount : ℝ := 15.00

theorem remaining_amount :
  initial_amount - spent_amount = 85.00 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_l2385_238559


namespace NUMINAMATH_CALUDE_nesbitt_like_inequality_l2385_238509

theorem nesbitt_like_inequality (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_like_inequality_l2385_238509


namespace NUMINAMATH_CALUDE_one_mile_equals_500_rods_l2385_238572

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 10

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 500 rods -/
theorem one_mile_equals_500_rods : rods_in_mile = 500 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_equals_500_rods_l2385_238572


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2385_238580

/-- An isosceles triangle with one side of length 7 and perimeter 17 has other sides of lengths (5, 5) or (7, 3) -/
theorem isosceles_triangle_side_lengths :
  ∀ (a b c : ℝ),
  a = 7 ∧ 
  a + b + c = 17 ∧
  ((b = c) ∨ (a = b) ∨ (a = c)) →
  ((b = 5 ∧ c = 5) ∨ (b = 7 ∧ c = 3) ∨ (b = 3 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2385_238580


namespace NUMINAMATH_CALUDE_four_divisions_for_400_to_25_l2385_238551

/-- The number of divisions needed to reduce a collection of books to a target group size -/
def divisions_needed (total_books : ℕ) (target_group_size : ℕ) : ℕ :=
  if total_books ≤ target_group_size then 0
  else 1 + divisions_needed (total_books / 2) target_group_size

/-- Theorem stating that 4 divisions are needed to reduce 400 books to groups of 25 -/
theorem four_divisions_for_400_to_25 :
  divisions_needed 400 25 = 4 := by
sorry

end NUMINAMATH_CALUDE_four_divisions_for_400_to_25_l2385_238551


namespace NUMINAMATH_CALUDE_quadratic_max_l2385_238550

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 2)^2 - 3

-- State the theorem
theorem quadratic_max (x : ℝ) : 
  (∀ y : ℝ, f y ≤ f 2) ∧ f 2 = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_l2385_238550


namespace NUMINAMATH_CALUDE_integer_roots_of_f_l2385_238521

def f (x : ℤ) : ℤ := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem integer_roots_of_f :
  {x : ℤ | f x = 0} = {1, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_f_l2385_238521


namespace NUMINAMATH_CALUDE_percent_of_number_hundred_fifty_percent_of_eighty_l2385_238544

theorem percent_of_number (percent : ℝ) (number : ℝ) : 
  (percent / 100) * number = (percent * number) / 100 := by sorry

theorem hundred_fifty_percent_of_eighty : 
  (150 : ℝ) / 100 * 80 = 120 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_hundred_fifty_percent_of_eighty_l2385_238544


namespace NUMINAMATH_CALUDE_square_sum_from_means_l2385_238513

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l2385_238513


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l2385_238549

/-- The set of points satisfying (x+y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l2385_238549


namespace NUMINAMATH_CALUDE_emir_book_purchase_l2385_238534

/-- The cost of the dictionary in dollars -/
def dictionary_cost : ℕ := 5

/-- The cost of the dinosaur book in dollars -/
def dinosaur_book_cost : ℕ := 11

/-- The cost of the children's cookbook in dollars -/
def cookbook_cost : ℕ := 5

/-- The amount Emir has saved in dollars -/
def saved_amount : ℕ := 19

/-- The additional money Emir needs to buy all three books -/
def additional_money_needed : ℕ := 2

theorem emir_book_purchase :
  dictionary_cost + dinosaur_book_cost + cookbook_cost - saved_amount = additional_money_needed :=
by sorry

end NUMINAMATH_CALUDE_emir_book_purchase_l2385_238534


namespace NUMINAMATH_CALUDE_original_number_proof_l2385_238585

theorem original_number_proof (x : ℝ) : 1 + 1/x = 11/5 → x = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2385_238585


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2385_238593

/-- Given three consecutive odd integers whose sum is 75 and whose largest and smallest differ by 6, the largest is 27 -/
theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ k : ℤ, c = 2*k + 1) →  -- c is odd
  b = a + 2 →              -- b is the next consecutive odd after a
  c = b + 2 →              -- c is the next consecutive odd after b
  a + b + c = 75 →         -- sum is 75
  c - a = 6 →              -- difference between largest and smallest is 6
  c = 27 :=                -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2385_238593


namespace NUMINAMATH_CALUDE_sarah_wallet_ones_l2385_238583

/-- Represents the contents of Sarah's wallet -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def valid_wallet (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 50 ∧
  w.ones + 2 * w.twos + 5 * w.fives = 146

theorem sarah_wallet_ones :
  ∃ w : Wallet, valid_wallet w ∧ w.ones = 14 := by
  sorry

end NUMINAMATH_CALUDE_sarah_wallet_ones_l2385_238583


namespace NUMINAMATH_CALUDE_weavers_count_proof_l2385_238560

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 6

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 9

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 6

theorem weavers_count_proof :
  (first_group_mats : ℚ) / (first_group_weavers * first_group_days) =
  (second_group_mats : ℚ) / (second_group_weavers * second_group_days) →
  first_group_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_weavers_count_proof_l2385_238560


namespace NUMINAMATH_CALUDE_problem_statement_l2385_238524

theorem problem_statement (m n p q : ℕ) 
  (h : ∀ x : ℝ, x > 0 → (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2*n + p)^(2*q) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2385_238524


namespace NUMINAMATH_CALUDE_initial_oranges_count_l2385_238546

/-- Proves that the initial number of oranges in a bowl is 20, given the specified conditions. -/
theorem initial_oranges_count (apples : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  apples = 14 → removed_oranges = 14 → apple_percentage = 7/10 → 
  ∃ initial_oranges : ℕ, 
    initial_oranges = 20 ∧ 
    (apples : ℚ) / ((apples : ℚ) + (initial_oranges - removed_oranges : ℚ)) = apple_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l2385_238546


namespace NUMINAMATH_CALUDE_circle_angle_theorem_l2385_238530

-- Define the circle and angles
def Circle (F : Point) : Prop := sorry

def angle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem circle_angle_theorem (F A B C D E : Point) :
  Circle F →
  angle B F C = 2 * angle A F B →
  angle C F D = 3 * angle A F B →
  angle D F E = 4 * angle A F B →
  angle E F A = 5 * angle A F B →
  angle B F C = 48 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_theorem_l2385_238530


namespace NUMINAMATH_CALUDE_five_coin_probability_l2385_238519

def num_coins : ℕ := 5

def total_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2

def probability : ℚ := favorable_outcomes / total_outcomes

theorem five_coin_probability :
  probability = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_five_coin_probability_l2385_238519


namespace NUMINAMATH_CALUDE_class_average_l2385_238501

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℕ) 
                      (rest_average : ℕ) : 
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_average = 45 →
  (top_scorers * top_score + 
   (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l2385_238501


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2385_238588

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2385_238588


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2385_238543

/-- Given that θ is an angle in the second quadrant, prove that θ/2 lies in the first or third quadrant. -/
theorem half_angle_quadrant (θ : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < θ ∧ θ < 2 * k * π + π) :
  ∃ k : ℤ, (k * π < θ / 2 ∧ θ / 2 < k * π + π / 2) ∨ 
           (k * π + π < θ / 2 ∧ θ / 2 < k * π + 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2385_238543


namespace NUMINAMATH_CALUDE_filtration_theorem_l2385_238586

/-- The reduction rate of impurities per filtration -/
def reduction_rate : ℝ := 0.2

/-- The target percentage of impurities relative to the original amount -/
def target_percentage : ℝ := 0.05

/-- The logarithm of 2 -/
def log_2 : ℝ := 0.301

/-- The minimum number of filtrations required -/
def min_filtrations : ℕ := 14

theorem filtration_theorem : 
  ∀ n : ℕ, (1 - reduction_rate) ^ n < target_percentage ↔ n ≥ min_filtrations := by
  sorry

end NUMINAMATH_CALUDE_filtration_theorem_l2385_238586


namespace NUMINAMATH_CALUDE_solve_for_A_l2385_238577

/-- Given that 3ab · A = 6a²b - 9ab², prove that A = 2a - 3b -/
theorem solve_for_A (a b A : ℝ) (h : 3 * a * b * A = 6 * a^2 * b - 9 * a * b^2) :
  A = 2 * a - 3 * b := by
sorry

end NUMINAMATH_CALUDE_solve_for_A_l2385_238577


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2385_238578

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_orthogonal : (x - 1) * 1 + 3 * y = 0) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * 1 + 3 * y = 0 ∧ 1 / x + 1 / (3 * y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2385_238578


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2385_238522

/-- Given a point P with coordinates (3, -2) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (3, -2)
  P = (3, -2) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2385_238522


namespace NUMINAMATH_CALUDE_geric_bills_l2385_238537

theorem geric_bills (jessa kyla geric : ℕ) : 
  geric = 2 * kyla →
  kyla = jessa - 2 →
  jessa - 3 = 7 →
  geric = 16 := by
sorry

end NUMINAMATH_CALUDE_geric_bills_l2385_238537


namespace NUMINAMATH_CALUDE_total_bowling_balls_l2385_238520

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l2385_238520


namespace NUMINAMATH_CALUDE_inverse_proposition_l2385_238589

-- Define the original proposition
def corresponding_angles_equal : Prop := sorry

-- Define the inverse proposition
def equal_angles_corresponding : Prop := sorry

-- Theorem stating that equal_angles_corresponding is the inverse of corresponding_angles_equal
theorem inverse_proposition : 
  (corresponding_angles_equal → equal_angles_corresponding) ∧ 
  (equal_angles_corresponding → corresponding_angles_equal) := by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2385_238589


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2385_238592

open Real

theorem trigonometric_inequality : 
  let a := sin (3 * π / 5)
  let b := cos (2 * π / 5)
  let c := tan (2 * π / 5)
  b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2385_238592


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2385_238552

-- Problem 1
theorem problem_1 : -8 - 6 + 24 = 10 := by sorry

-- Problem 2
theorem problem_2 : (-48) / 6 + (-21) * (-1/3) = -1 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1/4) * (-24) = -1 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) * (1 - (-2)^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2385_238552


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2385_238554

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2385_238554


namespace NUMINAMATH_CALUDE_polynomial_property_l2385_238503

-- Define the polynomial Q(x)
def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

-- Define the properties of the polynomial
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 5
  Q d e f 0 = 5 →
  -- The mean of zeros equals the product of zeros
  -d/3 = -f →
  -- The mean of zeros equals the sum of coefficients
  -d/3 = 1 + d + e + f →
  -- Conclusion: e = -26
  e = -26 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l2385_238503


namespace NUMINAMATH_CALUDE_restaurant_production_difference_l2385_238584

/-- Represents the daily production of a restaurant -/
structure DailyProduction where
  pizzas : ℕ
  hotDogs : ℕ
  pizzasMoreThanHotDogs : pizzas > hotDogs

/-- Represents the monthly production of a restaurant -/
def MonthlyProduction (d : DailyProduction) (days : ℕ) : ℕ :=
  days * (d.pizzas + d.hotDogs)

/-- Theorem: The restaurant makes 40 more pizzas than hot dogs every day -/
theorem restaurant_production_difference (d : DailyProduction) 
    (h1 : d.hotDogs = 60)
    (h2 : MonthlyProduction d 30 = 4800) :
  d.pizzas - d.hotDogs = 40 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_difference_l2385_238584


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2385_238535

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2385_238535


namespace NUMINAMATH_CALUDE_function_shift_l2385_238528

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) : f (x + 1) = x^2 - 2*x - 3 → f x = x^2 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l2385_238528


namespace NUMINAMATH_CALUDE_joes_lifts_l2385_238515

theorem joes_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1500)
  (h2 : first_lift = 600) : 
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end NUMINAMATH_CALUDE_joes_lifts_l2385_238515


namespace NUMINAMATH_CALUDE_first_day_sale_is_30_percent_l2385_238566

/-- The percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- The percentage of apples thrown away on the first day -/
def first_day_throwaway_percentage : ℝ := 0.20

/-- The percentage of apples sold on the second day -/
def second_day_sale_percentage : ℝ := 0.50

/-- The total percentage of apples thrown away -/
def total_throwaway_percentage : ℝ := 0.42

/-- Theorem stating that the percentage of apples sold on the first day is 30% -/
theorem first_day_sale_is_30_percent :
  first_day_sale_percentage = 0.30 :=
by
  sorry

end NUMINAMATH_CALUDE_first_day_sale_is_30_percent_l2385_238566


namespace NUMINAMATH_CALUDE_F_and_I_mutually_exclusive_and_complementary_l2385_238500

structure TouristChoice where
  goesToA : Bool
  goesToB : Bool

def E (choice : TouristChoice) : Prop := choice.goesToA ∧ ¬choice.goesToB
def F (choice : TouristChoice) : Prop := choice.goesToA ∨ choice.goesToB
def G (choice : TouristChoice) : Prop := (choice.goesToA ∧ ¬choice.goesToB) ∨ (¬choice.goesToA ∧ choice.goesToB) ∨ (¬choice.goesToA ∧ ¬choice.goesToB)
def H (choice : TouristChoice) : Prop := ¬choice.goesToA
def I (choice : TouristChoice) : Prop := ¬choice.goesToA ∧ ¬choice.goesToB

theorem F_and_I_mutually_exclusive_and_complementary :
  ∀ (choice : TouristChoice),
    (F choice ∧ I choice → False) ∧
    (F choice ∨ I choice) :=
sorry

end NUMINAMATH_CALUDE_F_and_I_mutually_exclusive_and_complementary_l2385_238500


namespace NUMINAMATH_CALUDE_rest_time_calculation_l2385_238507

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (rest_interval : ℝ) (h1 : walking_rate = 10) (h2 : total_distance = 50) 
  (h3 : total_time = 320) (h4 : rest_interval = 10) : 
  (total_time - (total_distance / walking_rate) * 60) / ((total_distance / rest_interval) - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rest_time_calculation_l2385_238507


namespace NUMINAMATH_CALUDE_scavenger_hunt_difference_l2385_238547

theorem scavenger_hunt_difference (lewis_items samantha_items tanya_items : ℕ) : 
  lewis_items = 20 →
  samantha_items = 4 * tanya_items →
  tanya_items = 4 →
  lewis_items - samantha_items = 4 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_difference_l2385_238547


namespace NUMINAMATH_CALUDE_S_4_S_n_l2385_238561

-- Define N(n) as the largest odd factor of n
def N (n : ℕ+) : ℕ := sorry

-- Define S(n) as the sum of N(k) from k=1 to 2^n
def S (n : ℕ) : ℕ := sorry

-- Theorem for S(4)
theorem S_4 : S 4 = 86 := by sorry

-- Theorem for S(n)
theorem S_n (n : ℕ) : S n = (4^n + 2) / 3 := by sorry

end NUMINAMATH_CALUDE_S_4_S_n_l2385_238561


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_84_l2385_238595

theorem gcf_lcm_sum_36_56_84 : 
  let a := 36
  let b := 56
  let c := 84
  Nat.gcd a (Nat.gcd b c) + Nat.lcm a (Nat.lcm b c) = 516 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_84_l2385_238595


namespace NUMINAMATH_CALUDE_original_number_proof_l2385_238558

theorem original_number_proof : ∃! N : ℤ, 
  (N - 8) % 5 = 4 ∧ 
  (N - 8) % 7 = 4 ∧ 
  (N - 8) % 9 = 4 ∧ 
  N = 326 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2385_238558


namespace NUMINAMATH_CALUDE_notebooks_distribution_l2385_238597

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Each child got one-eighth of the number of children in notebooks
  (N / (C / 2) = 16) →  -- If number of children halved, each would get 16 notebooks
  N = 512 := by  -- Total notebooks distributed is 512
sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l2385_238597


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l2385_238598

theorem inverse_proportion_increasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ →
    (1 - 2*m) / x₁ < (1 - 2*m) / x₂) ↔
  m > 1/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l2385_238598


namespace NUMINAMATH_CALUDE_A_subset_B_l2385_238563

def A : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l2385_238563


namespace NUMINAMATH_CALUDE_expression_equals_59_l2385_238517

theorem expression_equals_59 (a b c : ℝ) (ha : a = 17) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_59_l2385_238517


namespace NUMINAMATH_CALUDE_min_skilled_players_exists_tournament_with_three_skilled_l2385_238538

/-- Represents a player in the tournament -/
def Player := Fin 2023

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → Prop

/-- A player is skilled if for every player who defeats them, there exists another player who defeats that player and loses to the skilled player -/
def IsSkilled (result : MatchResult) (p : Player) : Prop :=
  ∀ q, result q p → ∃ r, result p r ∧ result r q

/-- The tournament satisfies the given conditions -/
def ValidTournament (result : MatchResult) : Prop :=
  (∀ p q, p ≠ q → (result p q ∨ result q p)) ∧
  (∀ p, ¬(∀ q, p ≠ q → result p q))

/-- The main theorem: there are at least 3 skilled players in any valid tournament -/
theorem min_skilled_players (result : MatchResult) (h : ValidTournament result) :
  ∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c :=
sorry

/-- There exists a valid tournament with exactly 3 skilled players -/
theorem exists_tournament_with_three_skilled :
  ∃ result : MatchResult, ValidTournament result ∧
  (∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c ∧
    (∀ p, IsSkilled result p → p = a ∨ p = b ∨ p = c)) :=
sorry

end NUMINAMATH_CALUDE_min_skilled_players_exists_tournament_with_three_skilled_l2385_238538


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l2385_238545

/-- Represents the number of pieces cut in each step -/
def CutSequence := List Nat

/-- Calculates the total number of pieces after a sequence of cuts -/
def totalPieces (cuts : CutSequence) : Nat :=
  1 + 4 * (1 + cuts.sum)

theorem paper_cutting_theorem (cuts : CutSequence) :
  ∃ (k : Nat), totalPieces cuts = 4 * k + 1 :=
sorry

theorem smallest_over_2000 :
  ∀ (cuts : CutSequence),
    totalPieces cuts > 2000 →
    totalPieces cuts ≥ 2005 :=
sorry

theorem exactly_2005_exists :
  ∃ (cuts : CutSequence), totalPieces cuts = 2005 :=
sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l2385_238545
