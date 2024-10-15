import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_max_value_l474_47425

theorem triangle_angle_max_value (A B C : ℝ) : 
  A + B + C = π →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_max_value_l474_47425


namespace NUMINAMATH_CALUDE_discount_comparison_l474_47460

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def final_price_option1 : ℝ :=
  apply_successive_discounts initial_amount option1_discounts

def final_price_option2 : ℝ :=
  apply_successive_discounts initial_amount option2_discounts

theorem discount_comparison :
  final_price_option1 - final_price_option2 = 997.50 ∧
  final_price_option2 < final_price_option1 :=
sorry

end NUMINAMATH_CALUDE_discount_comparison_l474_47460


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l474_47412

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1/4

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculate the probability of reaching the target point from the start point
    in at most the maximum number of steps -/
def probabilityToReachTarget : ℚ :=
  45/2048

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget = 45/2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l474_47412


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l474_47498

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), 0 < k' ∧ k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) ∧
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l474_47498


namespace NUMINAMATH_CALUDE_extraction_of_geometric_from_arithmetic_l474_47429

-- Define the arithmetic progression
def arithmeticProgression (a b : ℤ) (k : ℤ) : ℤ := a + b * k

-- Define the geometric progression
def geometricProgression (a b : ℤ) (k : ℕ) : ℤ := a * (b + 1)^k

theorem extraction_of_geometric_from_arithmetic (a b : ℤ) :
  ∃ (f : ℕ → ℤ), (∀ k : ℕ, ∃ l : ℤ, geometricProgression a b k = arithmeticProgression a b l) :=
sorry

end NUMINAMATH_CALUDE_extraction_of_geometric_from_arithmetic_l474_47429


namespace NUMINAMATH_CALUDE_gcf_of_45_135_90_l474_47414

theorem gcf_of_45_135_90 : Nat.gcd 45 (Nat.gcd 135 90) = 45 := by sorry

end NUMINAMATH_CALUDE_gcf_of_45_135_90_l474_47414


namespace NUMINAMATH_CALUDE_jims_initial_reading_speed_l474_47416

/-- Represents Jim's reading habits and speeds -/
structure ReadingHabits where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  initial_hours : ℝ  -- Initial hours read per week
  new_speed : ℝ      -- New reading speed in pages per hour
  new_hours : ℝ      -- New hours read per week

/-- Theorem stating Jim's initial reading speed -/
theorem jims_initial_reading_speed 
  (h : ReadingHabits) 
  (initial_pages : h.initial_speed * h.initial_hours = 600) 
  (speed_increase : h.new_speed = 1.5 * h.initial_speed)
  (time_decrease : h.new_hours = h.initial_hours - 4)
  (new_pages : h.new_speed * h.new_hours = 660) : 
  h.initial_speed = 40 := by
  sorry


end NUMINAMATH_CALUDE_jims_initial_reading_speed_l474_47416


namespace NUMINAMATH_CALUDE_land_area_proof_l474_47495

theorem land_area_proof (original_side : ℝ) (cut_width : ℝ) (remaining_area : ℝ) :
  cut_width = 10 →
  remaining_area = 1575 →
  original_side * (original_side - cut_width) = remaining_area →
  original_side * cut_width = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_land_area_proof_l474_47495


namespace NUMINAMATH_CALUDE_power_sum_and_division_l474_47461

theorem power_sum_and_division (a b c : ℕ) : 3^456 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l474_47461


namespace NUMINAMATH_CALUDE_square_factor_l474_47440

theorem square_factor (a b : ℝ) (square : ℝ) :
  square * (3 * a * b) = 3 * a^2 * b → square = a := by
  sorry

end NUMINAMATH_CALUDE_square_factor_l474_47440


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l474_47494

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, 
  let a : ℝ × ℝ := (1, t)
  let b : ℝ × ℝ := (t, 9)
  parallel a b → t = 3 ∨ t = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l474_47494


namespace NUMINAMATH_CALUDE_siblings_difference_l474_47449

theorem siblings_difference (masud_siblings : ℕ) : 
  masud_siblings = 60 →
  let janet_siblings := 4 * masud_siblings - 60
  let carlos_siblings := (3 * masud_siblings) / 4
  janet_siblings - carlos_siblings = 45 := by
sorry

end NUMINAMATH_CALUDE_siblings_difference_l474_47449


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l474_47437

/-- The number of diagonals from a vertex in a regular decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals from a vertex in a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l474_47437


namespace NUMINAMATH_CALUDE_triangle_configuration_l474_47406

/-- Represents a triangle with side lengths x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hx : x > 0
  hy : y > 0
  hz : z > 0
  hxy : x < y + z
  hyz : y < x + z
  hzx : z < x + y

/-- Theorem about a specific triangle configuration -/
theorem triangle_configuration (a : ℝ) : 
  ∃ (t : Triangle), 
    t.x + t.y = 3 * t.z ∧ 
    t.z + t.y = t.x + a ∧ 
    t.x + t.z = 60 → 
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 42 ∧ t.y = 48 ∧ t.z = 30) := by
  sorry

#check triangle_configuration

end NUMINAMATH_CALUDE_triangle_configuration_l474_47406


namespace NUMINAMATH_CALUDE_art_fair_customers_l474_47417

theorem art_fair_customers (group1 group2 group3 : ℕ) 
  (paintings_per_customer1 paintings_per_customer2 paintings_per_customer3 : ℕ) 
  (total_paintings : ℕ) : 
  group1 = 4 → 
  group2 = 12 → 
  group3 = 4 → 
  paintings_per_customer1 = 2 → 
  paintings_per_customer2 = 1 → 
  paintings_per_customer3 = 4 → 
  total_paintings = 36 → 
  group1 * paintings_per_customer1 + 
  group2 * paintings_per_customer2 + 
  group3 * paintings_per_customer3 = total_paintings → 
  group1 + group2 + group3 = 20 := by
sorry

end NUMINAMATH_CALUDE_art_fair_customers_l474_47417


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l474_47431

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(x-1, x+1) is in the second quadrant and x is an integer, then x must be 0 -/
theorem point_in_second_quadrant (x : ℤ) : in_second_quadrant (x - 1 : ℝ) (x + 1 : ℝ) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l474_47431


namespace NUMINAMATH_CALUDE_difference_of_squares_l474_47456

theorem difference_of_squares (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l474_47456


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l474_47444

/-- A regular polygon with interior angles of 144° has a sum of interior angles equal to 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (h : n ≥ 3) :
  let interior_angle : ℝ := 144
  n * interior_angle = (n - 2) * 180 ∧ n * interior_angle = 1440 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l474_47444


namespace NUMINAMATH_CALUDE_zoo_rhinos_count_zoo_rhinos_count_is_three_l474_47418

/-- Calculates the number of endangered rhinos taken in by a zoo --/
theorem zoo_rhinos_count (initial_animals : ℕ) (gorilla_family : ℕ) (hippo : ℕ) 
  (lion_cubs : ℕ) (final_animals : ℕ) : ℕ :=
  let animals_after_gorillas := initial_animals - gorilla_family
  let animals_after_hippo := animals_after_gorillas + hippo
  let animals_after_cubs := animals_after_hippo + lion_cubs
  let meerkats := 2 * lion_cubs
  let animals_before_rhinos := animals_after_cubs + meerkats
  final_animals - animals_before_rhinos

/-- Proves that the number of endangered rhinos taken in is 3 --/
theorem zoo_rhinos_count_is_three : 
  zoo_rhinos_count 68 6 1 8 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_rhinos_count_zoo_rhinos_count_is_three_l474_47418


namespace NUMINAMATH_CALUDE_open_box_volume_l474_47488

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * 
  (sheet_width - 2 * cut_square_side) * 
  cut_square_side = 5120 := by
sorry

end NUMINAMATH_CALUDE_open_box_volume_l474_47488


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l474_47411

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → 
  s^2 - 5*s + 3 = 0 → 
  r^3 + s^3 = 80 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l474_47411


namespace NUMINAMATH_CALUDE_unique_quadratic_with_real_roots_l474_47423

/-- A geometric progression of length 2016 -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i ∈ Finset.range 2015, a (i + 1) = r * a i

/-- An arithmetic progression of length 2016 -/
def arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i ∈ Finset.range 2015, b (i + 1) = b i + d

/-- The quadratic trinomial P_i(x) = x^2 + a_i * x + b_i -/
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

/-- P_k(x) has real roots iff its discriminant is non-negative -/
def has_real_roots (a b : ℕ → ℝ) (k : ℕ) : Prop :=
  (a k)^2 - 4 * b k ≥ 0

theorem unique_quadratic_with_real_roots
  (a b : ℕ → ℝ)
  (h_geom : geometric_progression a)
  (h_arith : arithmetic_progression b)
  (h_unique : ∃! k : ℕ, k ∈ Finset.range 2016 ∧ has_real_roots a b k) :
  ∃ k : ℕ, (k = 1 ∨ k = 2016) ∧ k ∈ Finset.range 2016 ∧ has_real_roots a b k :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_with_real_roots_l474_47423


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l474_47464

theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * x + 20 = 8 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l474_47464


namespace NUMINAMATH_CALUDE_min_value_xyz_l474_47439

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 27) :
  2 * x + 3 * y + 6 * z ≥ 54 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 27 ∧ 2 * x' + 3 * y' + 6 * z' = 54 := by
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l474_47439


namespace NUMINAMATH_CALUDE_wilson_pays_twelve_l474_47448

/-- The total cost of Wilson's purchase at a fast-food restaurant --/
def total_cost (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ) : ℕ :=
  hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

/-- Theorem stating that Wilson pays $12 in total --/
theorem wilson_pays_twelve :
  ∀ (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ),
    hamburger_price = 5 →
    cola_price = 2 →
    hamburger_quantity = 2 →
    cola_quantity = 3 →
    discount = 4 →
    total_cost hamburger_price cola_price hamburger_quantity cola_quantity discount = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_wilson_pays_twelve_l474_47448


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l474_47420

/-- The number of handshakes at a family gathering -/
def total_handshakes (twin_sets quadruplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let quadruplets := quadruplet_sets * 4
  let twin_handshakes := twins * (twins - 2) / 2
  let quadruplet_handshakes := quadruplets * (quadruplets - 4) / 2
  let cross_handshakes := twins * (quadruplets / 3) + quadruplets * (twins / 4)
  twin_handshakes + quadruplet_handshakes + cross_handshakes

/-- Theorem stating the number of handshakes at the family gathering -/
theorem family_gathering_handshakes :
  total_handshakes 12 8 = 1168 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l474_47420


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_set_l474_47476

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) (h1 : a > 1) :
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x a ≥ m) → a = 7 :=
sorry

-- Theorem for part 2
theorem inequality_solution_set (x : ℝ) :
  f x 7 ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_set_l474_47476


namespace NUMINAMATH_CALUDE_quadrilateral_problem_l474_47401

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_problem (PQRS : Quadrilateral) (T : Point) :
  isConvex PQRS →
  isPerpendicular PQRS.R PQRS.S PQRS.P PQRS.Q →
  isPerpendicular PQRS.P PQRS.Q PQRS.R PQRS.S →
  distance PQRS.R PQRS.S = 52 →
  distance PQRS.P PQRS.Q = 39 →
  isPerpendicular PQRS.Q T PQRS.P PQRS.S →
  T = lineIntersection PQRS.P PQRS.Q PQRS.Q T →
  distance PQRS.P T = 25 →
  distance PQRS.Q T = 14 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_problem_l474_47401


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l474_47499

/-- Given a line l with y-intercept 1 and perpendicular to y = (1/2)x, 
    prove that the equation of l is y = -2x + 1 -/
theorem perpendicular_line_equation (l : Set (ℝ × ℝ)) 
  (y_intercept : (0, 1) ∈ l)
  (perpendicular : ∀ (x y : ℝ), (x, y) ∈ l → (y - 1) = m * x → m * (1/2) = -1) :
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2 * x + 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l474_47499


namespace NUMINAMATH_CALUDE_point_identity_l474_47455

/-- Given a point P(x, y) in the plane, prove that s^2 + c^2 = 1 where
    r is the distance from the origin to P,
    s = y/r,
    c = x/r,
    and c^2 = 4/9 -/
theorem point_identity (x y : ℝ) : 
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  c^2 = 4/9 →
  s^2 + c^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_identity_l474_47455


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l474_47463

/-- 
Given 8 packets of gum with 7 pieces each, and leaving 2 pieces unchewed,
prove that the number of pieces chewed is equal to 54.
-/
theorem mitchell_gum_chewing (packets : Nat) (pieces_per_packet : Nat) (unchewed : Nat) : 
  packets = 8 → pieces_per_packet = 7 → unchewed = 2 →
  packets * pieces_per_packet - unchewed = 54 := by
sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l474_47463


namespace NUMINAMATH_CALUDE_hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l474_47424

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  right_focus : ℝ × ℝ

-- Define the standard form of a hyperbola equation
def standard_form (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 - y^2 / b^2 = 1

-- Theorem for the first part of the problem
theorem hyperbola_equation_with_eccentricity 
  (C : Hyperbola) 
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_eccentricity : ∃ e, e = Real.sqrt 3) :
  ∃ a b, standard_form a b = λ x y => x^2 - y^2 / 2 = 1 :=
sorry

-- Theorem for the second part of the problem
theorem hyperbola_equation_with_asymptote
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_asymptote : ∃ X Y, X + Real.sqrt 2 * Y = 0) :
  ∃ a b, standard_form a b = λ x y => x^2 / 2 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l474_47424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_5_l474_47409

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  a_1 : a 1 = 3
  S_3 : (a 1) + (a 2) + (a 3) = 21
  a_n : ∃ n, a n = 48

/-- The theorem stating that n = 5 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_5 (seq : ArithmeticSequence) :
  ∃ n, seq.a n = 48 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_5_l474_47409


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l474_47491

theorem baker_cakes_problem (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (pastry_cake_difference : ℕ) :
  pastries_made = 131 →
  cakes_sold = 70 →
  pastries_sold = 88 →
  pastry_cake_difference = 112 →
  ∃ cakes_made : ℕ, 
    cakes_made + pastry_cake_difference = pastries_made ∧
    cakes_made = 107 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l474_47491


namespace NUMINAMATH_CALUDE_guilty_pair_is_B_and_C_l474_47469

/-- Represents the guilt status of a defendant -/
inductive GuiltStatus
| Guilty
| Innocent

/-- Represents a defendant -/
inductive Defendant
| A
| B
| C

/-- The guilt status of all defendants -/
def GuiltStatusSet := Defendant → GuiltStatus

/-- At least one of the defendants is guilty -/
def atLeastOneGuilty (gs : GuiltStatusSet) : Prop :=
  ∃ d : Defendant, gs d = GuiltStatus.Guilty

/-- If A is guilty and B is innocent, then C is innocent -/
def conditionalInnocence (gs : GuiltStatusSet) : Prop :=
  (gs Defendant.A = GuiltStatus.Guilty ∧ gs Defendant.B = GuiltStatus.Innocent) →
  gs Defendant.C = GuiltStatus.Innocent

/-- The main theorem stating that B and C are the two defendants such that one of them is definitely guilty -/
theorem guilty_pair_is_B_and_C :
  ∀ gs : GuiltStatusSet,
  atLeastOneGuilty gs →
  conditionalInnocence gs →
  (gs Defendant.B = GuiltStatus.Guilty ∨ gs Defendant.C = GuiltStatus.Guilty) :=
sorry

end NUMINAMATH_CALUDE_guilty_pair_is_B_and_C_l474_47469


namespace NUMINAMATH_CALUDE_car_stop_time_l474_47443

/-- The distance traveled by a car after braking -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time required for the car to stop after braking -/
theorem car_stop_time : ∃ t : ℝ, S t = 0 ∧ t = 6 := by
  sorry

end NUMINAMATH_CALUDE_car_stop_time_l474_47443


namespace NUMINAMATH_CALUDE_rope_division_l474_47454

/-- Given a rope of 1 meter length divided into two parts, where the second part is twice the length of the first part, prove that the length of the first part is 1/3 meter. -/
theorem rope_division (x : ℝ) (h1 : x > 0) (h2 : x + 2*x = 1) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l474_47454


namespace NUMINAMATH_CALUDE_students_liking_neither_l474_47486

theorem students_liking_neither (total : ℕ) (chinese : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 62)
  (h_chinese : chinese = 37)
  (h_math : math = 49)
  (h_both : both = 30) :
  total - (chinese + math - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_neither_l474_47486


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l474_47472

theorem complex_fraction_equality : (2 : ℂ) / (1 - I) = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l474_47472


namespace NUMINAMATH_CALUDE_cute_5digit_integer_count_l474_47489

/-- A function that checks if a list of digits forms a palindrome -/
def isPalindrome (digits : List Nat) : Prop :=
  digits = digits.reverse

/-- A function that checks if the first k digits of a number are divisible by k -/
def firstKDigitsDivisibleByK (digits : List Nat) (k : Nat) : Prop :=
  let firstK := digits.take k
  let num := firstK.foldl (fun acc d => acc * 10 + d) 0
  num % k = 0

/-- A function that checks if a list of digits satisfies all conditions -/
def isCute (digits : List Nat) : Prop :=
  digits.length = 5 ∧
  digits.toFinset = {1, 2, 3, 4, 5} ∧
  isPalindrome digits ∧
  ∀ k, 1 ≤ k ∧ k ≤ 5 → firstKDigitsDivisibleByK digits k

theorem cute_5digit_integer_count :
  ∃! digits : List Nat, isCute digits :=
sorry

end NUMINAMATH_CALUDE_cute_5digit_integer_count_l474_47489


namespace NUMINAMATH_CALUDE_bee_hive_population_l474_47433

/-- The population growth function for bees in a hive -/
def bee_population (initial : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial * growth_factor ^ days

/-- Theorem stating the population of bees after 20 days -/
theorem bee_hive_population :
  bee_population 1 5 20 = 5^20 := by
  sorry

end NUMINAMATH_CALUDE_bee_hive_population_l474_47433


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l474_47430

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l474_47430


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l474_47402

def is_prime_for_all (a : ℕ+) : Prop :=
  ∀ n : ℕ, n < a → Nat.Prime (4 * n^2 + a)

theorem prime_condition_characterization :
  ∀ a : ℕ+, is_prime_for_all a ↔ (a = 3 ∨ a = 7) :=
sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l474_47402


namespace NUMINAMATH_CALUDE_toucan_count_l474_47450

theorem toucan_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 1 → total = 3 → initial = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l474_47450


namespace NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l474_47427

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (3-a)*x + 2*(1-a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1-a}
  else if a = -1 then ∅
  else {x | 1-a < x ∧ x < 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 := by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l474_47427


namespace NUMINAMATH_CALUDE_smallest_possible_d_l474_47465

theorem smallest_possible_d : ∃ (d : ℝ), d ≥ 0 ∧
  (∀ (d' : ℝ), d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l474_47465


namespace NUMINAMATH_CALUDE_rational_equation_solution_l474_47492

theorem rational_equation_solution :
  ∃! x : ℚ, (x ≠ 2/3) ∧ (x ≠ -3) ∧
  ((7*x + 3) / (3*x^2 + 7*x - 6) = (5*x) / (3*x - 2)) ∧
  x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l474_47492


namespace NUMINAMATH_CALUDE_b_age_is_four_l474_47426

-- Define the ages as natural numbers
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem b_age_is_four :
  (a = b + 2) →  -- a is two years older than b
  (b = 2 * c) →  -- b is twice as old as c
  (a + b + c = 12) →  -- The total of the ages is 12
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_four_l474_47426


namespace NUMINAMATH_CALUDE_rectangle_area_properties_l474_47405

-- Define the rectangle's dimensions and measurement errors
def expected_length : Real := 2
def expected_width : Real := 1
def length_std_dev : Real := 0.003
def width_std_dev : Real := 0.002

-- Define the theorem
theorem rectangle_area_properties :
  let expected_area := expected_length * expected_width
  let area_variance := (expected_length^2 * width_std_dev^2) + (expected_width^2 * length_std_dev^2) + (length_std_dev^2 * width_std_dev^2)
  let area_std_dev := Real.sqrt area_variance
  (expected_area = 2) ∧ (area_std_dev * 100 = 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_properties_l474_47405


namespace NUMINAMATH_CALUDE_student_arrangement_l474_47474

/-- The number of arrangements for n male and m female students -/
def arrangement_count (n m : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := sorry

theorem student_arrangement :
  let total_male : ℕ := 5
  let total_female : ℕ := 5
  let females_between : ℕ := 2
  let males_at_ends : ℕ := 2
  
  arrangement_count total_male total_female = 
    choose total_female females_between * 
    permute (total_male - 2) males_at_ends * 
    permute (total_male + total_female - females_between - males_at_ends - 2) 
            (total_male + total_female - females_between - males_at_ends - 2) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_l474_47474


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l474_47445

theorem fourth_root_equation_solutions :
  let f (x : ℝ) := (Real.sqrt (Real.sqrt (43 - 2*x))) + (Real.sqrt (Real.sqrt (39 + 2*x)))
  ∃ (S : Set ℝ), S = {x | f x = 4} ∧ S = {21, -13.5} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l474_47445


namespace NUMINAMATH_CALUDE_max_height_sphere_hemispheres_tower_l474_47413

/-- The maximum height of a tower consisting of a sphere and three hemispheres -/
theorem max_height_sphere_hemispheres_tower (r₀ : ℝ) (h : r₀ = 2017) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₀ ≥ r₁ ∧ r₁ ≥ r₂ ∧ r₂ ≥ r₃ ∧ r₃ > 0 ∧
    r₀ + Real.sqrt (4 * r₀^2) = 3 * r₀ ∧
    3 * r₀ = 6051 :=
by sorry

end NUMINAMATH_CALUDE_max_height_sphere_hemispheres_tower_l474_47413


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l474_47470

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are parts of hypotenuse
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l474_47470


namespace NUMINAMATH_CALUDE_deck_width_l474_47446

/-- Given a rectangular pool of dimensions 10 feet by 12 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 360 square feet, then the width of the deck is 4 feet. -/
theorem deck_width (w : ℝ) : 
  (10 + 2*w) * (12 + 2*w) = 360 → w = 4 := by sorry

end NUMINAMATH_CALUDE_deck_width_l474_47446


namespace NUMINAMATH_CALUDE_brendans_tax_payment_is_correct_l474_47434

/-- Calculates Brendan's weekly tax payment based on his work schedule and income reporting --/
def brendans_weekly_tax_payment (
  waiter_hourly_wage : ℚ)
  (barista_hourly_wage : ℚ)
  (waiter_shift_hours : List ℚ)
  (barista_shift_hours : List ℚ)
  (waiter_hourly_tips : ℚ)
  (barista_hourly_tips : ℚ)
  (waiter_tax_rate : ℚ)
  (barista_tax_rate : ℚ)
  (waiter_reported_tips_ratio : ℚ)
  (barista_reported_tips_ratio : ℚ) : ℚ :=
  let waiter_total_hours := waiter_shift_hours.sum
  let barista_total_hours := barista_shift_hours.sum
  let waiter_wage_income := waiter_total_hours * waiter_hourly_wage
  let barista_wage_income := barista_total_hours * barista_hourly_wage
  let waiter_total_tips := waiter_total_hours * waiter_hourly_tips
  let barista_total_tips := barista_total_hours * barista_hourly_tips
  let waiter_reported_tips := waiter_total_tips * waiter_reported_tips_ratio
  let barista_reported_tips := barista_total_tips * barista_reported_tips_ratio
  let waiter_reported_income := waiter_wage_income + waiter_reported_tips
  let barista_reported_income := barista_wage_income + barista_reported_tips
  let waiter_tax := waiter_reported_income * waiter_tax_rate
  let barista_tax := barista_reported_income * barista_tax_rate
  waiter_tax + barista_tax

theorem brendans_tax_payment_is_correct :
  brendans_weekly_tax_payment 6 8 [8, 8, 12] [6] 12 5 (1/5) (1/4) (1/3) (1/2) = 71.75 := by
  sorry

end NUMINAMATH_CALUDE_brendans_tax_payment_is_correct_l474_47434


namespace NUMINAMATH_CALUDE_no_solution_equation_l474_47447

theorem no_solution_equation :
  ∀ x : ℝ, x ≠ 4 → x - 9 / (x - 4) ≠ 4 - 9 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l474_47447


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l474_47419

theorem quadratic_root_difference (a b c : ℝ) (h : a > 0) :
  let equation := fun x => (5 + 2 * Real.sqrt 5) * x^2 - (3 + Real.sqrt 5) * x + 1
  let larger_root := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let smaller_root := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  equation larger_root = 0 ∧ equation smaller_root = 0 →
  larger_root - smaller_root = Real.sqrt (-3 + (2 * Real.sqrt 5) / 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l474_47419


namespace NUMINAMATH_CALUDE_sand_pile_base_area_l474_47438

/-- Given a rectangular compartment of sand and a conical pile, this theorem proves
    that the base area of the pile is 81/2 square meters. -/
theorem sand_pile_base_area
  (length width height : ℝ)
  (pile_height : ℝ)
  (h_length : length = 6)
  (h_width : width = 1.5)
  (h_height : height = 3)
  (h_pile_height : pile_height = 2)
  (h_volume_conservation : length * width * height = (1/3) * Real.pi * (pile_base_area / Real.pi) * pile_height)
  : pile_base_area = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_base_area_l474_47438


namespace NUMINAMATH_CALUDE_real_y_condition_l474_47477

theorem real_y_condition (x y : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 5 * x * y + x + 7 = 0) ↔ (x ≤ -6/5 ∨ x ≥ 14/5) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l474_47477


namespace NUMINAMATH_CALUDE_milk_water_mixture_volume_l474_47479

/-- Proves that given a mixture of milk and water with an initial ratio of 3:2,
    if adding 46 liters of water changes the ratio to 3:4,
    then the initial volume of the mixture was 115 liters. -/
theorem milk_water_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 3 / 2)
  (h2 : initial_milk / (initial_water + 46) = 3 / 4) :
  initial_milk + initial_water = 115 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_mixture_volume_l474_47479


namespace NUMINAMATH_CALUDE_altitude_angle_relation_l474_47481

/-- For an acute-angled triangle with circumradius R and altitude h from a vertex,
    the angle α at that vertex satisfies the given conditions. -/
theorem altitude_angle_relation (α : Real) (R h : ℝ) : 
  (α < Real.pi / 3 ↔ h < R) ∧ 
  (α = Real.pi / 3 ↔ h = R) ∧ 
  (α > Real.pi / 3 ↔ h > R) :=
by sorry

end NUMINAMATH_CALUDE_altitude_angle_relation_l474_47481


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l474_47441

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l474_47441


namespace NUMINAMATH_CALUDE_max_non_managers_l474_47403

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l474_47403


namespace NUMINAMATH_CALUDE_cafeteria_pies_l474_47497

/-- Given a cafeteria scenario with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
    (h1 : initial_apples = 62)
    (h2 : handed_out = 8)
    (h3 : apples_per_pie = 9) :
    (initial_apples - handed_out) / apples_per_pie = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l474_47497


namespace NUMINAMATH_CALUDE_board_officer_selection_ways_l474_47410

def board_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_special_members : ℕ := 26 * 25 * 24 * 23
def ways_with_one_pair : ℕ := 4 * 3 * 26 * 25
def ways_with_both_pairs : ℕ := 4 * 3 * 2 * 1

theorem board_officer_selection_ways :
  ways_without_special_members + 2 * ways_with_one_pair + ways_with_both_pairs = 374424 :=
sorry

end NUMINAMATH_CALUDE_board_officer_selection_ways_l474_47410


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l474_47484

theorem unique_remainder_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l474_47484


namespace NUMINAMATH_CALUDE_cone_volume_l474_47415

theorem cone_volume (d h : ℝ) (h1 : d = 16) (h2 : h = 12) :
  (1 / 3 : ℝ) * π * (d / 2) ^ 2 * h = 256 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l474_47415


namespace NUMINAMATH_CALUDE_parallelogram_area_l474_47475

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 6), and (5, 6) is 24 square units. -/
theorem parallelogram_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (1, 6)
  let area := abs ((B.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (B.2 - A.2))
  area = 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l474_47475


namespace NUMINAMATH_CALUDE_evaluate_expression_l474_47435

theorem evaluate_expression (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 10) - 1 / (Real.sqrt 10 - Real.sqrt 9) + 
      1 / (Real.sqrt 9 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
      1 / (Real.sqrt 7 - 3) → 
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l474_47435


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l474_47451

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ (k : ℕ), k = 82 ∧ (N + k) % 7 = 0 ∧ (N + k) % 12 = 0 ∧
  ∀ (m : ℕ), m < k → (N + m) % 7 ≠ 0 ∨ (N + m) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l474_47451


namespace NUMINAMATH_CALUDE_trains_meeting_time_l474_47453

/-- Two trains meeting problem -/
theorem trains_meeting_time 
  (distance : ℝ) 
  (speed1 speed2 : ℝ) 
  (start_time2 meet_time : ℝ) 
  (h1 : distance = 200) 
  (h2 : speed1 = 20) 
  (h3 : speed2 = 25) 
  (h4 : start_time2 = 8) 
  (h5 : meet_time = 12) : 
  ∃ start_time1 : ℝ, 
    start_time1 = 7 ∧ 
    speed1 * (meet_time - start_time1) + speed2 * (meet_time - start_time2) = distance :=
sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l474_47453


namespace NUMINAMATH_CALUDE_lineup_organization_l474_47400

/-- The number of ways to organize a football lineup -/
def organize_lineup (total_members : ℕ) (defensive_linemen : ℕ) : ℕ :=
  defensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem: The number of ways to organize a lineup for a team with 7 members,
    of which 4 can play defensive lineman, is 480 -/
theorem lineup_organization :
  organize_lineup 7 4 = 480 := by
  sorry

end NUMINAMATH_CALUDE_lineup_organization_l474_47400


namespace NUMINAMATH_CALUDE_households_with_only_bike_l474_47485

/-- Given a neighborhood with the following properties:
  * There are 90 total households
  * 11 households have neither a car nor a bike
  * 18 households have both a car and a bike
  * 44 households have a car (including those with both)
  Then the number of households with only a bike is 35. -/
theorem households_with_only_bike
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 18)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l474_47485


namespace NUMINAMATH_CALUDE_sum_after_transformation_l474_47432

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l474_47432


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l474_47421

/-- A geometric figure composed of five identical squares arranged in a 'T' shape -/
structure TShape where
  /-- The total area of the figure in square centimeters -/
  total_area : ℝ
  /-- The figure is composed of five identical squares -/
  num_squares : ℕ
  /-- Assumption that the total area is 125 cm² -/
  area_assumption : total_area = 125
  /-- Assumption that the number of squares is 5 -/
  squares_assumption : num_squares = 5

/-- The perimeter of the 'T' shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem stating that the perimeter of the 'T' shaped figure is 35 cm -/
theorem t_shape_perimeter (t : TShape) : perimeter t = 35 :=
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l474_47421


namespace NUMINAMATH_CALUDE_root_equation_difference_l474_47471

theorem root_equation_difference (a b : ℤ) :
  (∃ x : ℝ, x^2 = 7 - 4 * Real.sqrt 3 ∧ x^2 + a * x + b = 0) →
  b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_difference_l474_47471


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l474_47467

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24) ∧ (a ≠ b) ∧ (a.val + b.val = 96) ∧ (∀ (c d : ℕ+), ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24) → (c ≠ d) → (c.val + d.val ≥ 96)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l474_47467


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l474_47458

theorem x_cubed_coefficient (p q : Polynomial ℤ) (hp : p = 3 * X ^ 4 - 2 * X ^ 3 + X ^ 2 - 3) 
  (hq : q = 2 * X ^ 2 + 5 * X - 4) : 
  (p * q).coeff 3 = 13 := by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l474_47458


namespace NUMINAMATH_CALUDE_circle_radius_property_l474_47478

theorem circle_radius_property (r : ℝ) : 
  r > 0 → r * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → 
  ∃ (radius : ℝ), radius > 0 ∧ radius * (2 * Real.pi * radius) = 2 * (Real.pi * radius^2) := by
sorry

end NUMINAMATH_CALUDE_circle_radius_property_l474_47478


namespace NUMINAMATH_CALUDE_find_A_value_l474_47422

theorem find_A_value : ∃ (A B : ℕ), 
  A < 10 ∧ B < 10 ∧ 
  10 * A + 8 + 30 + B = 99 ∧
  A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l474_47422


namespace NUMINAMATH_CALUDE_hyperbola_fixed_point_l474_47490

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define the condition that a point is on the hyperbola
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- Define the perpendicularity condition
def perpendicular (p q : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (q.1 - A.1) + (p.2 - A.2) * (q.2 - A.2) = 0

-- Define the condition that a line passes through a point
def line_passes_through (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (r.1 - p.1) * (q.2 - p.2)

-- Main theorem
theorem hyperbola_fixed_point :
  ∀ (p q : ℝ × ℝ),
    on_hyperbola p →
    on_hyperbola q →
    perpendicular p q →
    line_passes_through p q (3, 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_fixed_point_l474_47490


namespace NUMINAMATH_CALUDE_dollar_three_neg_one_l474_47436

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 2) + a * (b + 1)

-- Theorem to prove
theorem dollar_three_neg_one : dollar 3 (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_one_l474_47436


namespace NUMINAMATH_CALUDE_factorial_expression_equality_l474_47496

theorem factorial_expression_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 6 - 6 * Nat.factorial 5 = 7920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equality_l474_47496


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l474_47407

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem min_slope_tangent_line :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f' x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l474_47407


namespace NUMINAMATH_CALUDE_rectangle_area_change_l474_47428

/-- Given a rectangle with initial dimensions 3 × 7 inches, if shortening one side by 2 inches 
    results in an area of 15 square inches, then shortening the other side by 2 inches 
    will result in an area of 7 square inches. -/
theorem rectangle_area_change (initial_width initial_length : ℝ) 
  (h1 : initial_width = 3)
  (h2 : initial_length = 7)
  (h3 : initial_width * (initial_length - 2) = 15 ∨ (initial_width - 2) * initial_length = 15) :
  (initial_width - 2) * initial_length = 7 ∨ initial_width * (initial_length - 2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l474_47428


namespace NUMINAMATH_CALUDE_xy_plus_inverse_min_value_l474_47462

theorem xy_plus_inverse_min_value (x y : ℝ) 
  (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17 / 4 :=
sorry

end NUMINAMATH_CALUDE_xy_plus_inverse_min_value_l474_47462


namespace NUMINAMATH_CALUDE_solve_system_l474_47404

theorem solve_system (x y : ℝ) : 
  (5 * x - 3 = 2 * x + 9) → 
  (x + y = 10) → 
  (x = 4 ∧ y = 6) := by
sorry


end NUMINAMATH_CALUDE_solve_system_l474_47404


namespace NUMINAMATH_CALUDE_joshuas_bottle_caps_l474_47480

/-- The total number of bottle caps after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Joshua's total bottle caps -/
theorem joshuas_bottle_caps : total_bottle_caps 40 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_joshuas_bottle_caps_l474_47480


namespace NUMINAMATH_CALUDE_adjacent_probability_four_people_l474_47473

def num_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def favorable_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

def probability_adjacent (n : ℕ) : ℚ :=
  (favorable_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem adjacent_probability_four_people :
  probability_adjacent num_people = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_four_people_l474_47473


namespace NUMINAMATH_CALUDE_al_ben_weight_difference_l474_47493

theorem al_ben_weight_difference :
  ∀ (al_weight ben_weight carl_weight : ℕ),
    ben_weight = carl_weight - 16 →
    al_weight = 146 + 38 →
    carl_weight = 175 →
    al_weight - ben_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_al_ben_weight_difference_l474_47493


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l474_47408

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l474_47408


namespace NUMINAMATH_CALUDE_special_sum_value_l474_47459

theorem special_sum_value (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 104) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 282 :=
by
  sorry

end NUMINAMATH_CALUDE_special_sum_value_l474_47459


namespace NUMINAMATH_CALUDE_eunji_class_size_l474_47468

/-- The number of students in Eunji's class -/
def class_size : ℕ := 24

/-- The number of lines the students stand in -/
def num_lines : ℕ := 3

/-- Eunji's position from the front of her row -/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of her row -/
def position_from_back : ℕ := 6

/-- Theorem stating the number of students in Eunji's class -/
theorem eunji_class_size :
  class_size = num_lines * (position_from_front + position_from_back - 1) :=
by sorry

end NUMINAMATH_CALUDE_eunji_class_size_l474_47468


namespace NUMINAMATH_CALUDE_journey_time_equation_l474_47452

theorem journey_time_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 15
  let cyclist_speed : ℝ := x
  let car_speed : ℝ := 2 * x
  let head_start : ℝ := 1 / 2
  distance / cyclist_speed = distance / car_speed + head_start :=
by sorry

end NUMINAMATH_CALUDE_journey_time_equation_l474_47452


namespace NUMINAMATH_CALUDE_mistake_permutations_four_letter_word_l474_47466

/-- The number of permutations of a word with repeated letters -/
def permutations_with_repetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial r

/-- The number of mistake permutations for a 4-letter word with one letter repeated twice -/
theorem mistake_permutations_four_letter_word : 
  permutations_with_repetition 4 2 - 1 = 11 := by
  sorry

#eval permutations_with_repetition 4 2 - 1

end NUMINAMATH_CALUDE_mistake_permutations_four_letter_word_l474_47466


namespace NUMINAMATH_CALUDE_integral_value_l474_47487

-- Define the inequality
def inequality (x a : ℝ) : Prop := 1 - 3 / (x + a) < 0

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-1) 2

-- Theorem statement
theorem integral_value (a : ℝ) 
  (h1 : ∀ x, x ∈ solution_set ↔ inequality x a) :
  ∫ x in a..3, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l474_47487


namespace NUMINAMATH_CALUDE_final_price_approximation_l474_47442

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  initialPrice : ℝ
  week1Reduction : ℝ := 0.10
  week2Reduction : ℝ := 0.15
  week3Reduction : ℝ := 0.20
  additionalQuantity : ℝ := 5
  fixedCost : ℝ := 800

/-- Calculates the final price after three weeks of reductions --/
def finalPrice (opr : OilPriceReduction) : ℝ :=
  opr.initialPrice * (1 - opr.week1Reduction) * (1 - opr.week2Reduction) * (1 - opr.week3Reduction)

/-- Theorem stating the final reduced price is approximately 62.06 --/
theorem final_price_approximation (opr : OilPriceReduction) : 
  ∃ (initialQuantity : ℝ), 
    opr.fixedCost = initialQuantity * opr.initialPrice ∧
    opr.fixedCost = (initialQuantity + opr.additionalQuantity) * (finalPrice opr) ∧
    abs ((finalPrice opr) - 62.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_price_approximation_l474_47442


namespace NUMINAMATH_CALUDE_sin_angle_A_is_sqrt3_div_2_l474_47457

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  -- Side lengths
  AB : ℝ
  CD : ℝ
  AD : ℝ
  -- Angle A in radians
  angleA : ℝ
  -- Conditions
  isIsosceles : AD = AD  -- AD = BC
  isParallel : AB < CD  -- AB parallel to CD implies AB < CD
  angleValue : angleA = 2 * Real.pi / 3  -- 120° in radians
  sideAB : AB = 160
  sideCD : CD = 240
  perimeter : AB + CD + 2 * AD = 800

/-- The sine of angle A in the isosceles trapezoid is √3/2 -/
theorem sin_angle_A_is_sqrt3_div_2 (t : IsoscelesTrapezoid) : Real.sin t.angleA = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_A_is_sqrt3_div_2_l474_47457


namespace NUMINAMATH_CALUDE_chord_length_l474_47482

/-- The length of the chord cut by the line y = 3x on the circle (x+1)^2 + (y-2)^2 = 25 is 3√10. -/
theorem chord_length (x y : ℝ) : 
  y = 3 * x →
  (x + 1)^2 + (y - 2)^2 = 25 →
  ∃ (x1 y1 x2 y2 : ℝ), 
    y1 = 3 * x1 ∧
    (x1 + 1)^2 + (y1 - 2)^2 = 25 ∧
    y2 = 3 * x2 ∧
    (x2 + 1)^2 + (y2 - 2)^2 = 25 ∧
    ((x2 - x1)^2 + (y2 - y1)^2) = 90 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l474_47482


namespace NUMINAMATH_CALUDE_inequality_solution_set_l474_47483

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 1) * (x - 2) < 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l474_47483
