import Mathlib

namespace NUMINAMATH_CALUDE_cube_equality_solution_l2436_243659

theorem cube_equality_solution : ∃! (N : ℕ), N > 0 ∧ 12^3 * 30^3 = 20^3 * N^3 :=
by
  use 18
  sorry

end NUMINAMATH_CALUDE_cube_equality_solution_l2436_243659


namespace NUMINAMATH_CALUDE_simplify_expression_l2436_243656

theorem simplify_expression (a : ℝ) (h1 : a^2 - 1 ≠ 0) (h2 : a ≠ 0) :
  (1 / (a + 1) + 1 / (a^2 - 1)) / (a / (a^2 - 2*a + 1)) = (a - 1) / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2436_243656


namespace NUMINAMATH_CALUDE_mollys_age_l2436_243688

/-- Given the ratio of Sandy's age to Molly's age and Sandy's future age, 
    prove Molly's current age -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l2436_243688


namespace NUMINAMATH_CALUDE_calculator_sale_result_l2436_243612

def calculator_transaction (price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : Prop :=
  let profit_calculator_cost : ℝ := price / (1 + profit_rate)
  let loss_calculator_cost : ℝ := price / (1 - loss_rate)
  let total_cost : ℝ := profit_calculator_cost + loss_calculator_cost
  let total_revenue : ℝ := 2 * price
  total_revenue - total_cost = -7.5

theorem calculator_sale_result :
  calculator_transaction 90 0.2 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_result_l2436_243612


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_is_25_dollars_l2436_243609

/-- The cost of 500 pencils in dollars -/
def cost_of_500_pencils : ℚ :=
  let cost_per_pencil : ℚ := 5 / 100  -- 5 cents in dollars
  let number_of_pencils : ℕ := 500
  cost_per_pencil * number_of_pencils

theorem cost_of_500_pencils_is_25_dollars :
  cost_of_500_pencils = 25 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_is_25_dollars_l2436_243609


namespace NUMINAMATH_CALUDE_tiling_symmetry_l2436_243661

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Tiling relation between rectangles -/
def CanTile (A B : Rectangle) : Prop :=
  ∃ (n m : ℕ), n * A.width = m * B.width ∧ n * A.height = m * B.height

/-- Similarity relation between rectangles -/
def IsSimilarTo (A B : Rectangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ A.width = k * B.width ∧ A.height = k * B.height

/-- Main theorem: If a rectangle similar to A can be tiled with B, 
    then a rectangle similar to B can be tiled with A -/
theorem tiling_symmetry (A B : Rectangle) :
  (∃ (C : Rectangle), IsSimilarTo C A ∧ CanTile C B) →
  (∃ (D : Rectangle), IsSimilarTo D B ∧ CanTile D A) :=
by
  sorry


end NUMINAMATH_CALUDE_tiling_symmetry_l2436_243661


namespace NUMINAMATH_CALUDE_count_not_divisible_1999_l2436_243616

def count_not_divisible (n : ℕ) : ℕ :=
  n - (n / 4 + n / 6 - n / 12)

theorem count_not_divisible_1999 :
  count_not_divisible 1999 = 1333 := by
  sorry

end NUMINAMATH_CALUDE_count_not_divisible_1999_l2436_243616


namespace NUMINAMATH_CALUDE_range_of_a_l2436_243683

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + a) < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 1 ∉ A a ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2436_243683


namespace NUMINAMATH_CALUDE_negation_of_existence_l2436_243618

theorem negation_of_existence (p : Prop) :
  (¬∃ x₀ : ℝ, x₀ ≥ 1 ∧ x₀^2 - x₀ < 0) ↔ (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2436_243618


namespace NUMINAMATH_CALUDE_divisible_by_225_l2436_243651

theorem divisible_by_225 (n : ℕ) : ∃ k : ℤ, 16^n - 15*n - 1 = 225*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_225_l2436_243651


namespace NUMINAMATH_CALUDE_tangent_line_condition_l2436_243642

/-- The function f(x) = 2x - a ln x has a tangent line y = x + 1 at the point (1, f(1)) if and only if a = 1 -/
theorem tangent_line_condition (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2*x - a * Real.log x) ∧ 
   (∃ g : ℝ → ℝ, (∀ x, g x = x + 1) ∧ 
    (∀ h : ℝ → ℝ, HasDerivAt f (g 1 - f 1) 1 → h = g))) ↔ 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l2436_243642


namespace NUMINAMATH_CALUDE_problem_statement_l2436_243690

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 2) :
  (∀ x : ℝ, (1/a^2 + 4/b^2 ≥ |2*x - 1| - |x - 1|) → (-9/2 ≤ x ∧ x ≤ 9/2)) ∧
  ((1/a + 1/b) * (a^5 + b^5) ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2436_243690


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l2436_243641

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  (2 * n) * (2 * n - 2) / 2

/-- Theorem: In a gathering of 8 couples, if each person shakes hands once
    with everyone except their spouse, the total number of handshakes is 112 -/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

#eval handshakes 8  -- Should output 112

end NUMINAMATH_CALUDE_eight_couples_handshakes_l2436_243641


namespace NUMINAMATH_CALUDE_remainder_of_nested_star_l2436_243667

-- Define the star operation
def star (a b : ℕ) : ℕ := a * b - 2

-- Define a function to represent the nested star operations
def nested_star : ℕ → ℕ
| 0 => 9
| n + 1 => star (579 - 10 * n) (nested_star n)

-- Theorem statement
theorem remainder_of_nested_star :
  nested_star 57 % 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_nested_star_l2436_243667


namespace NUMINAMATH_CALUDE_abc_is_right_triangle_l2436_243689

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Parabola y^2 = 4x -/
def on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Line passes through a point -/
def line_passes_through (l : Line) (p : Point) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  let ab_slope := (t.b.y - t.a.y) / (t.b.x - t.a.x)
  let ac_slope := (t.c.y - t.a.y) / (t.c.x - t.a.x)
  ab_slope * ac_slope = -1

theorem abc_is_right_triangle (a b c : Point) (h1 : a.x = 1 ∧ a.y = 2)
    (h2 : on_parabola b) (h3 : on_parabola c)
    (h4 : line_passes_through (Line.mk b c) (Point.mk 5 (-2))) :
    is_right_triangle (Triangle.mk a b c) := by
  sorry

end NUMINAMATH_CALUDE_abc_is_right_triangle_l2436_243689


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2436_243619

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1011011110₂ -/
def binary_input : List Bool := [true, false, true, true, false, true, true, true, true, false]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_input) = [2, 3, 1, 3, 2] :=
sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2436_243619


namespace NUMINAMATH_CALUDE_bricks_for_room_floor_bricks_needed_is_340_l2436_243602

/-- Calculates the number of bricks needed for a rectangular room floor -/
theorem bricks_for_room_floor 
  (length : ℝ) 
  (breadth : ℝ) 
  (bricks_per_sqm : ℕ) 
  (h1 : length = 4) 
  (h2 : breadth = 5) 
  (h3 : bricks_per_sqm = 17) : 
  ℕ := by
  
  sorry

#check bricks_for_room_floor

/-- Proves that 340 bricks are needed for the given room dimensions -/
theorem bricks_needed_is_340 : 
  bricks_for_room_floor 4 5 17 rfl rfl rfl = 340 := by
  
  sorry

end NUMINAMATH_CALUDE_bricks_for_room_floor_bricks_needed_is_340_l2436_243602


namespace NUMINAMATH_CALUDE_factorization_condition_l2436_243613

def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ),
    ∀ (x y : ℤ),
      x^2 + 3*x*y + x + m*y - m = (a*x + b*y + c) * (d*x + e*y + f)

theorem factorization_condition (m : ℤ) :
  is_factorizable m ↔ (m = 0 ∨ m = 12) :=
sorry

end NUMINAMATH_CALUDE_factorization_condition_l2436_243613


namespace NUMINAMATH_CALUDE_angle_sum_proof_l2436_243654

theorem angle_sum_proof (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan β = 1/3) : 
  α + 2*β = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l2436_243654


namespace NUMINAMATH_CALUDE_right_triangle_leg_identity_l2436_243684

theorem right_triangle_leg_identity (a b : ℝ) : 2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_identity_l2436_243684


namespace NUMINAMATH_CALUDE_delta_value_l2436_243631

theorem delta_value : ∃ Δ : ℂ, (4 * (-3) = Δ^2 + 3) ∧ (Δ = Complex.I * Real.sqrt 15 ∨ Δ = -Complex.I * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2436_243631


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l2436_243668

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m^2 + 2*m - 3)

-- Define the condition for z - m + 2 being purely imaginary
def is_purely_imaginary (m : ℝ) : Prop :=
  (z m - m + 2).re = 0 ∧ (z m - m + 2).im ≠ 0

-- Define the condition for point A being in the third quadrant
def in_third_quadrant (m : ℝ) : Prop :=
  (z m).re < 0 ∧ (z m).im < 0

-- Theorem 1: If z - m + 2 is purely imaginary, then m = 2
theorem purely_imaginary_implies_m_eq_two (m : ℝ) :
  is_purely_imaginary m → m = 2 :=
sorry

-- Theorem 2: If point A is in the third quadrant, then -3 < m < 1
theorem third_quadrant_implies_m_range (m : ℝ) :
  in_third_quadrant m → -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l2436_243668


namespace NUMINAMATH_CALUDE_first_number_calculation_l2436_243611

theorem first_number_calculation (average : ℝ) (num1 num2 added_num : ℝ) :
  average = 13 ∧ num1 = 16 ∧ num2 = 8 ∧ added_num = 22 →
  ∃ x : ℝ, (x + num1 + num2 + added_num) / 4 = average ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_number_calculation_l2436_243611


namespace NUMINAMATH_CALUDE_two_tvs_one_mixer_cost_l2436_243606

/-- The cost of a mixer in rupees -/
def mixer_cost : ℕ := 1400

/-- The cost of a TV in rupees -/
def tv_cost : ℕ := 4200

/-- The cost of two mixers and one TV in rupees -/
def two_mixers_one_tv_cost : ℕ := 7000

theorem two_tvs_one_mixer_cost : 2 * tv_cost + mixer_cost = 9800 := by
  sorry

end NUMINAMATH_CALUDE_two_tvs_one_mixer_cost_l2436_243606


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2436_243620

theorem square_sum_theorem (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2436_243620


namespace NUMINAMATH_CALUDE_animus_tower_spiders_l2436_243660

/-- The number of spiders hired for the Animus Tower project -/
def spiders_hired (total_workers beavers_hired : ℕ) : ℕ :=
  total_workers - beavers_hired

/-- Theorem stating the number of spiders hired for the Animus Tower project -/
theorem animus_tower_spiders :
  spiders_hired 862 318 = 544 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_spiders_l2436_243660


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l2436_243697

theorem quadratic_completion_square (x : ℝ) :
  4 * x^2 - 8 * x - 128 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l2436_243697


namespace NUMINAMATH_CALUDE_circle_fraction_l2436_243662

theorem circle_fraction (n : ℕ) (m : ℕ) (h1 : n > 0) (h2 : m ≤ n) :
  (m : ℚ) / n = m * (1 / n) :=
by sorry

#check circle_fraction

end NUMINAMATH_CALUDE_circle_fraction_l2436_243662


namespace NUMINAMATH_CALUDE_tom_bricks_count_l2436_243639

/-- The number of bricks Tom needs to buy -/
def num_bricks : ℕ := 1000

/-- The cost of a brick at full price -/
def full_price : ℚ := 1/2

/-- The total amount Tom spends -/
def total_spent : ℚ := 375

theorem tom_bricks_count :
  (num_bricks / 2 : ℚ) * (full_price / 2) + (num_bricks / 2 : ℚ) * full_price = total_spent :=
sorry

end NUMINAMATH_CALUDE_tom_bricks_count_l2436_243639


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2436_243674

/-- Given two vectors a and b in ℝ², prove that if |a| = 1, |b| = 2, and a + b = (2√2, 1), then |3a + b| = 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  norm a = 1 →
  norm b = 2 →
  a + b = (2 * Real.sqrt 2, 1) →
  norm (3 • a + b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2436_243674


namespace NUMINAMATH_CALUDE_absolute_value_plus_tan_sixty_degrees_l2436_243645

theorem absolute_value_plus_tan_sixty_degrees : 
  |(-2 + Real.sqrt 3)| + Real.tan (π / 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_tan_sixty_degrees_l2436_243645


namespace NUMINAMATH_CALUDE_correct_city_determination_l2436_243687

/-- Represents the two cities on Mars -/
inductive City
| MarsPolis
| MarsCity

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- A Martian's response to the question "Do you live here?" -/
def martianResponse (city : City) (martianOrigin : City) : Answer :=
  match city, martianOrigin with
  | City.MarsPolis, _ => Answer.Yes
  | City.MarsCity, _ => Answer.No

/-- Determines the city based on the Martian's response -/
def determineCity (response : Answer) : City :=
  match response with
  | Answer.Yes => City.MarsPolis
  | Answer.No => City.MarsCity

/-- Theorem stating that asking "Do you live here?" always determines the correct city -/
theorem correct_city_determination (actualCity : City) (martianOrigin : City) :
  determineCity (martianResponse actualCity martianOrigin) = actualCity :=
by sorry

end NUMINAMATH_CALUDE_correct_city_determination_l2436_243687


namespace NUMINAMATH_CALUDE_sphere_distance_to_plane_l2436_243699

/-- The distance from the center of a sphere to the plane formed by three points on its surface. -/
def distance_center_to_plane (r : ℝ) (a b c : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a sphere of radius 13 with three points on its surface forming
    a triangle with side lengths 6, 8, and 10, the distance from the center to the plane
    containing the triangle is 12. -/
theorem sphere_distance_to_plane :
  distance_center_to_plane 13 6 8 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sphere_distance_to_plane_l2436_243699


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2436_243650

theorem fraction_evaluation (a b c : ℝ) (ha : a = 4) (hb : b = -4) (hc : c = 3) :
  3 / (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2436_243650


namespace NUMINAMATH_CALUDE_loan_problem_l2436_243626

theorem loan_problem (P : ℝ) (h : (8 * 0.06 * P) = P - 572) : P = 1100 := by
  sorry

end NUMINAMATH_CALUDE_loan_problem_l2436_243626


namespace NUMINAMATH_CALUDE_carly_running_schedule_l2436_243669

def running_schedule (week1 : ℚ) (week2_multiplier : ℚ) (week2_extra : ℚ) (week3_multiplier : ℚ) (week4_reduction : ℚ) : ℚ → ℚ
  | 1 => week1
  | 2 => week1 * week2_multiplier + week2_extra
  | 3 => (week1 * week2_multiplier + week2_extra) * week3_multiplier
  | 4 => (week1 * week2_multiplier + week2_extra) * week3_multiplier - week4_reduction
  | _ => 0

theorem carly_running_schedule :
  running_schedule 2 2 3 (9/7) 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_carly_running_schedule_l2436_243669


namespace NUMINAMATH_CALUDE_libby_igloo_bricks_l2436_243630

/-- Calculates the total number of bricks in an igloo -/
def igloo_bricks (total_rows : ℕ) (bottom_bricks_per_row : ℕ) (top_bricks_per_row : ℕ) : ℕ :=
  let bottom_rows := total_rows / 2
  let top_rows := total_rows - bottom_rows
  bottom_rows * bottom_bricks_per_row + top_rows * top_bricks_per_row

/-- Proves that Libby's igloo uses 100 bricks -/
theorem libby_igloo_bricks :
  igloo_bricks 10 12 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_libby_igloo_bricks_l2436_243630


namespace NUMINAMATH_CALUDE_plane_equation_proof_l2436_243648

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 --/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def parallelPlanes (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 2 ∧ given_plane.B = -1 ∧ given_plane.C = 3 ∧ given_plane.D = 5 →
  point.x = 2 ∧ point.y = 3 ∧ point.z = -4 →
  ∃ (result_plane : Plane),
    parallelPlanes result_plane given_plane ∧
    pointOnPlane point result_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = 11 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l2436_243648


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2436_243695

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 25 →
  c = b + 7 →
  a ≤ b ∧ b ≤ c →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2436_243695


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l2436_243622

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) ≥ 2 :=
by sorry

theorem min_value_equality :
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + (1 / ((1 + 0) * (1 + 0) * (1 + 0))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l2436_243622


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2436_243633

theorem condition_sufficient_not_necessary :
  (∀ a b : ℝ, a + b = 1 → 4 * a * b ≤ 1) ∧
  (∃ a b : ℝ, 4 * a * b ≤ 1 ∧ a + b ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2436_243633


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2436_243629

/-- Given two vectors a and b in ℝ², prove that if a = (1, -2) and b = (3, x) are perpendicular, then x = 3/2. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, -2) →
  b = (3, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2436_243629


namespace NUMINAMATH_CALUDE_vector_perpendicular_m_l2436_243623

theorem vector_perpendicular_m (a b : ℝ × ℝ) (m : ℝ) : 
  a = (3, 4) → 
  b = (2, -1) → 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → 
  m = 23 / 3 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_m_l2436_243623


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l2436_243644

theorem inverse_of_proposition : 
  (∀ x : ℝ, x < 0 → x^2 > 0) → 
  (∀ x : ℝ, x^2 > 0 → x < 0) := by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l2436_243644


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l2436_243670

/-- Given a > 0 and f(x) = x³ + ax² - 9x - 1, if the tangent line with the smallest slope
    on the curve y = f(x) is perpendicular to the line x - 12y = 0, then a = 3 -/
theorem tangent_line_perpendicular (a : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 - 9*x - 1
  (∃ x₀ : ℝ, ∀ x : ℝ, (deriv f x₀ ≤ deriv f x) ∧ 
    (deriv f x₀ * (1 / 12) = -1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l2436_243670


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2436_243628

/-- The length of the major axis of the ellipse x²/9 + y²/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  let ellipse := fun (x y : ℝ) => x^2/9 + y^2/4 = 1
  ∃ (a b : ℝ), a > b ∧ a^2 = 9 ∧ b^2 = 4 ∧ 2*a = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2436_243628


namespace NUMINAMATH_CALUDE_tangent_product_l2436_243600

theorem tangent_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7)
  (h2 : 2 * Real.sin (2*x - 2*y) = Real.sin (2*x) * Real.sin (2*y)) :
  Real.tan x * Real.tan y = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l2436_243600


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2436_243605

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 40 and a₃ + a₄ = 60, prove that a₇ + a₈ = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 = 40 →                           -- a₁ + a₂ = 40
  a 3 + a 4 = 60 →                           -- a₃ + a₄ = 60
  a 7 + a 8 = 135 :=                         -- a₇ + a₈ = 135
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2436_243605


namespace NUMINAMATH_CALUDE_factorization_equality_l2436_243682

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2436_243682


namespace NUMINAMATH_CALUDE_basketball_probability_result_l2436_243621

/-- The probability that Beth, Jill, and Sandy make a basket while Adam and Jack miss -/
def basketball_probability (p_adam p_beth p_jack p_jill p_sandy : ℚ) : ℚ :=
  (1 - p_adam) * p_beth * (1 - p_jack) * p_jill * p_sandy

/-- Theorem stating the probability for the given scenario -/
theorem basketball_probability_result : 
  basketball_probability (1/5) (2/9) (1/6) (1/7) (1/8) = 1/378 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_result_l2436_243621


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2436_243607

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  -- Given condition
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0 →
  -- Additional conditions
  a = 2 →
  1/2 * b * c * Real.sin A = Real.sqrt 3 →
  -- Conclusion
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2436_243607


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2436_243680

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5123) % 12 = 2900 % 12 ∧
  ∀ (y : ℕ), y > 0 → (y + 5123) % 12 = 2900 % 12 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2436_243680


namespace NUMINAMATH_CALUDE_ice_cube_freeze_time_l2436_243646

/-- The time in minutes to turn frozen ice cubes into one smoothie -/
def time_per_smoothie : ℕ := 3

/-- The total number of smoothies made -/
def num_smoothies : ℕ := 5

/-- The total time in minutes to make all smoothies, including freezing ice cubes -/
def total_time : ℕ := 55

/-- The time in minutes to freeze ice cubes -/
def freeze_time : ℕ := total_time - (time_per_smoothie * num_smoothies)

theorem ice_cube_freeze_time :
  freeze_time = 40 := by sorry

end NUMINAMATH_CALUDE_ice_cube_freeze_time_l2436_243646


namespace NUMINAMATH_CALUDE_negation_of_existence_log_negation_equivalence_l2436_243665

theorem negation_of_existence (p : Real → Prop) :
  (¬∃ x, x > 1 ∧ p x) ↔ (∀ x, x > 1 → ¬p x) := by sorry

theorem log_negation_equivalence :
  (¬∃ x₀, x₀ > 1 ∧ Real.log x₀ > 1) ↔ (∀ x, x > 1 → Real.log x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_log_negation_equivalence_l2436_243665


namespace NUMINAMATH_CALUDE_sequence_formulas_l2436_243685

/-- Given a geometric sequence {a_n} and another sequence {b_n}, prove the formulas for a_n and b_n -/
theorem sequence_formulas (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence condition
  (a 1 = 4) →  -- initial condition for a_n
  (2 * a 2 + a 3 = 60) →  -- additional condition for a_n
  (∀ n, b (n + 1) = b n + a n) →  -- recurrence relation for b_n
  (b 1 = a 2) →  -- initial condition for b_n
  (b 1 > 0) →  -- positivity condition for b_1
  (∀ n, a n = 4 * 3^(n - 1)) ∧ 
  (∀ n, b n = 2 * 3^n + 10) := by
sorry


end NUMINAMATH_CALUDE_sequence_formulas_l2436_243685


namespace NUMINAMATH_CALUDE_class_size_is_40_l2436_243675

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students given the book borrowing data -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- The given book borrowing data for the class -/
def classBorrowers : BookBorrowers := {
  zero := 2,
  one := 12,
  two := 10,
  threeOrMore := 16  -- This value is not given directly, but can be derived
}

theorem class_size_is_40 :
  totalStudents classBorrowers = 40 ∧
  (minBooksBorrowed classBorrowers : ℚ) / (totalStudents classBorrowers) = 2 := by
  sorry


end NUMINAMATH_CALUDE_class_size_is_40_l2436_243675


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2436_243632

/-- The distance between the vertices of the hyperbola x²/64 - y²/81 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/64 - y^2/81 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2436_243632


namespace NUMINAMATH_CALUDE_ten_workers_needed_l2436_243664

/-- Represents the project details and worker productivity --/
structure Project where
  total_days : ℕ
  days_passed : ℕ
  work_completed : ℚ
  current_workers : ℕ

/-- Calculates the minimum number of workers needed to complete the project on schedule --/
def min_workers_needed (p : Project) : ℕ :=
  p.current_workers

/-- Theorem stating that for the given project conditions, 10 workers are needed --/
theorem ten_workers_needed (p : Project)
  (h1 : p.total_days = 40)
  (h2 : p.days_passed = 10)
  (h3 : p.work_completed = 1/4)
  (h4 : p.current_workers = 10) :
  min_workers_needed p = 10 := by
  sorry

#eval min_workers_needed {
  total_days := 40,
  days_passed := 10,
  work_completed := 1/4,
  current_workers := 10
}

end NUMINAMATH_CALUDE_ten_workers_needed_l2436_243664


namespace NUMINAMATH_CALUDE_holey_iff_presentable_l2436_243636

/-- A function is holey if there exists an interval free of its values -/
def IsHoley (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ ∀ x, a < x ∧ x < b → ∀ y, f y ≠ x

/-- A function is presentable if it can be represented as a composition of linear, inverse, and quadratic functions -/
inductive Presentable : (ℝ → ℝ) → Prop
  | linear (k b : ℝ) : Presentable (fun x ↦ k * x + b)
  | inverse : Presentable (fun x ↦ 1 / x)
  | square : Presentable (fun x ↦ x ^ 2)
  | comp {f g : ℝ → ℝ} (hf : Presentable f) (hg : Presentable g) : Presentable (f ∘ g)

/-- The main theorem statement -/
theorem holey_iff_presentable (a b c d : ℝ) 
    (h : ∀ x, x^2 + a*x + b ≠ 0 ∨ x^2 + c*x + d ≠ 0) : 
    IsHoley (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) ↔ 
    Presentable (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) :=
  sorry

end NUMINAMATH_CALUDE_holey_iff_presentable_l2436_243636


namespace NUMINAMATH_CALUDE_cut_tetrahedron_edge_count_l2436_243663

/-- Represents a regular tetrahedron with its vertices cut off. -/
structure CutTetrahedron where
  /-- The number of vertices in the original tetrahedron -/
  original_vertices : Nat
  /-- The number of edges in the original tetrahedron -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The cutting planes do not intersect on the solid -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting off each vertex -/
def edge_count (t : CutTetrahedron) : Nat :=
  t.original_edges + t.original_vertices * t.new_edges_per_cut

/-- Theorem stating that a regular tetrahedron with its vertices cut off has 18 edges -/
theorem cut_tetrahedron_edge_count :
  ∀ (t : CutTetrahedron),
    t.original_vertices = 4 →
    t.original_edges = 6 →
    t.new_edges_per_cut = 3 →
    t.non_intersecting_cuts →
    edge_count t = 18 :=
  sorry

end NUMINAMATH_CALUDE_cut_tetrahedron_edge_count_l2436_243663


namespace NUMINAMATH_CALUDE_number_problem_l2436_243698

theorem number_problem (x : ℚ) : (x / 6) * 12 = 10 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2436_243698


namespace NUMINAMATH_CALUDE_laptop_selection_problem_l2436_243673

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem laptop_selection_problem :
  let type_a : ℕ := 4
  let type_b : ℕ := 5
  let total_selection : ℕ := 3
  (choose type_a 2 * choose type_b 1) + (choose type_a 1 * choose type_b 2) = 70 :=
by sorry

end NUMINAMATH_CALUDE_laptop_selection_problem_l2436_243673


namespace NUMINAMATH_CALUDE_sugar_left_l2436_243647

theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_l2436_243647


namespace NUMINAMATH_CALUDE_phone_number_a_is_five_l2436_243679

/-- Represents a valid 10-digit telephone number -/
structure PhoneNumber where
  digits : Fin 10 → Fin 10
  all_different : ∀ i j, i ≠ j → digits i ≠ digits j
  decreasing_abc : digits 0 > digits 1 ∧ digits 1 > digits 2
  decreasing_def : digits 3 > digits 4 ∧ digits 4 > digits 5
  decreasing_ghij : digits 6 > digits 7 ∧ digits 7 > digits 8 ∧ digits 8 > digits 9
  consecutive_def : ∃ n : ℕ, digits 3 = n + 2 ∧ digits 4 = n + 1 ∧ digits 5 = n
  consecutive_ghij : ∃ n : ℕ, digits 6 = n + 3 ∧ digits 7 = n + 2 ∧ digits 8 = n + 1 ∧ digits 9 = n
  sum_abc : digits 0 + digits 1 + digits 2 = 10

theorem phone_number_a_is_five (p : PhoneNumber) : p.digits 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_a_is_five_l2436_243679


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2436_243604

theorem basketball_win_rate (total_games : ℕ) (first_segment : ℕ) (won_first : ℕ) (target_rate : ℚ) : 
  total_games = 130 →
  first_segment = 70 →
  won_first = 60 →
  target_rate = 3/4 →
  ∃ (x : ℕ), x = 38 ∧ 
    (won_first + x : ℚ) / total_games = target_rate ∧
    x ≤ total_games - first_segment :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l2436_243604


namespace NUMINAMATH_CALUDE_beaver_count_l2436_243614

theorem beaver_count (initial_beavers : Float) (additional_beavers : Float) : 
  initial_beavers = 2.0 → additional_beavers = 1.0 → initial_beavers + additional_beavers = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_beaver_count_l2436_243614


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2436_243655

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by            
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2436_243655


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2436_243696

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The number of possible license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count * vowel_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2436_243696


namespace NUMINAMATH_CALUDE_victory_points_value_l2436_243649

/-- Represents the number of points awarded for different match outcomes -/
structure PointSystem where
  victory : ℕ
  draw : ℕ
  defeat : ℕ

/-- Represents the state of a team's performance in the tournament -/
structure TeamPerformance where
  totalMatches : ℕ
  playedMatches : ℕ
  currentPoints : ℕ
  pointsNeeded : ℕ
  minWinsNeeded : ℕ

/-- The theorem stating the point value for a victory -/
theorem victory_points_value (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw = 1 ∧ 
  ps.defeat = 0 ∧
  tp.totalMatches = 20 ∧
  tp.playedMatches = 5 ∧
  tp.currentPoints = 12 ∧
  tp.pointsNeeded = 40 ∧
  tp.minWinsNeeded = 7 →
  ps.victory = 4 := by
  sorry

end NUMINAMATH_CALUDE_victory_points_value_l2436_243649


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l2436_243617

/-- Given an inequality a ≤ 3x + 6 ≤ b, if the length of the interval of solutions is 15, then b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ (l : ℝ), l = 15 ∧ l = (b - 6) / 3 - (a - 6) / 3) → b - a = 45 := by
  sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l2436_243617


namespace NUMINAMATH_CALUDE_y_value_at_16_l2436_243624

/-- Given a function y = k * x^(1/4) where y = 3 * √3 when x = 9, 
    prove that y = 6 when x = 16 -/
theorem y_value_at_16 (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/4)) →
  y 9 = 3 * Real.sqrt 3 →
  y 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_16_l2436_243624


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2436_243608

theorem jelly_bean_probability (red green yellow blue : ℕ) 
  (h_red : red = 7)
  (h_green : green = 9)
  (h_yellow : yellow = 4)
  (h_blue : blue = 10) :
  (red : ℚ) / (red + green + yellow + blue) = 7 / 30 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2436_243608


namespace NUMINAMATH_CALUDE_triplet_sum_position_l2436_243603

theorem triplet_sum_position 
  (x : Fin 6 → ℝ) 
  (s : Fin 20 → ℝ) 
  (h_order : ∀ i j, i < j → x i < x j) 
  (h_sums : ∀ i j, i < j → s i < s j) 
  (h_distinct : ∀ i j k l m n, i < j → j < k → l < m → m < n → 
    x i + x j + x k ≠ x l + x m + x n) 
  (h_s11 : x 1 + x 2 + x 3 = s 10) 
  (h_s15 : x 1 + x 2 + x 5 = s 14) : 
  x 0 + x 1 + x 5 = s 6 := by
sorry

end NUMINAMATH_CALUDE_triplet_sum_position_l2436_243603


namespace NUMINAMATH_CALUDE_decagon_triangle_count_l2436_243657

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) where
  (polygon : RegularPolygon n)
  (v1 v2 v3 : Fin n)
  (distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1)

/-- Two triangles in a regular polygon are congruent if they have the same shape -/
def CongruentTriangles (n : ℕ) (t1 t2 : PolygonTriangle n) : Prop :=
  sorry

/-- The number of non-congruent triangles in a regular decagon -/
def NumNonCongruentTriangles (p : RegularPolygon 10) : ℕ :=
  sorry

theorem decagon_triangle_count :
  ∀ (p : RegularPolygon 10), NumNonCongruentTriangles p = 8 :=
sorry

end NUMINAMATH_CALUDE_decagon_triangle_count_l2436_243657


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2436_243601

theorem imaginary_part_of_z : Complex.im (((1 : ℂ) - Complex.I) / (2 * Complex.I)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2436_243601


namespace NUMINAMATH_CALUDE_near_integer_intervals_l2436_243672

-- Definition of "near-integer interval"
def near_integer_interval (T : ℝ) : Set ℝ :=
  {x | ∃ (m n : ℤ), m < T ∧ T < n ∧ x ∈ Set.Ioo (↑m : ℝ) (↑n : ℝ) ∧
    ∀ (k : ℤ), k ≤ m ∨ n ≤ k}

-- Theorem statement
theorem near_integer_intervals :
  (near_integer_interval (Real.sqrt 5) = Set.Ioo 2 3) ∧
  (near_integer_interval (-Real.sqrt 10) = Set.Ioo (-4) (-3)) ∧
  (∀ (x y : ℝ), y = Real.sqrt (x - 2023) + Real.sqrt (2023 - x) →
    near_integer_interval (Real.sqrt (x + y)) = Set.Ioo 44 45) :=
by sorry

end NUMINAMATH_CALUDE_near_integer_intervals_l2436_243672


namespace NUMINAMATH_CALUDE_boat_license_combinations_l2436_243681

/-- The number of possible letters for a boat license -/
def letter_choices : ℕ := 4

/-- The number of possible choices for the first digit of a boat license -/
def first_digit_choices : ℕ := 8

/-- The number of possible choices for each of the remaining digits of a boat license -/
def other_digit_choices : ℕ := 10

/-- The number of digits in a boat license after the letter -/
def num_digits : ℕ := 7

/-- Theorem: The number of possible boat license combinations is 32,000,000 -/
theorem boat_license_combinations :
  letter_choices * first_digit_choices * (other_digit_choices ^ (num_digits - 1)) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l2436_243681


namespace NUMINAMATH_CALUDE_rectangle_area_l2436_243692

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y)
  (h3 : (x + 4) * (y - 2) = x * y) :
  x * y = 36 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2436_243692


namespace NUMINAMATH_CALUDE_sum_of_altitudes_is_23_and_one_seventh_l2436_243694

/-- A triangle formed by the line 18x + 9y = 108 and the coordinate axes -/
structure Triangle where
  -- The line equation
  line_eq : ℝ → ℝ → Prop := fun x y => 18 * x + 9 * y = 108
  -- The triangle is formed with coordinate axes
  forms_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ line_eq a 0 ∧ line_eq 0 b

/-- The sum of the lengths of the altitudes of the triangle -/
def sum_of_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the sum of the altitudes is 23 1/7 -/
theorem sum_of_altitudes_is_23_and_one_seventh (t : Triangle) :
  sum_of_altitudes t = 23 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_is_23_and_one_seventh_l2436_243694


namespace NUMINAMATH_CALUDE_journey_time_equation_l2436_243634

theorem journey_time_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 15
  let cyclist_speed : ℝ := x
  let car_speed : ℝ := 2 * x
  let head_start : ℝ := 1 / 2
  distance / cyclist_speed = distance / car_speed + head_start :=
by sorry

end NUMINAMATH_CALUDE_journey_time_equation_l2436_243634


namespace NUMINAMATH_CALUDE_certain_inning_is_19th_l2436_243652

/-- Represents the statistics of a cricketer before and after a certain inning -/
structure CricketerStats where
  prevInnings : ℕ
  prevAverage : ℚ
  runsScored : ℕ
  newAverage : ℚ

/-- Theorem stating that given the conditions, the certain inning was the 19th inning -/
theorem certain_inning_is_19th (stats : CricketerStats)
  (h1 : stats.runsScored = 97)
  (h2 : stats.newAverage = stats.prevAverage + 4)
  (h3 : stats.newAverage = 25) :
  stats.prevInnings + 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_inning_is_19th_l2436_243652


namespace NUMINAMATH_CALUDE_complex_equation_ratio_l2436_243678

theorem complex_equation_ratio (a b : ℝ) : 
  (a - 2*Complex.I)*Complex.I = b + a*Complex.I → a/b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_ratio_l2436_243678


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l2436_243638

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 336 →
  total_cost = 42 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l2436_243638


namespace NUMINAMATH_CALUDE_trains_meeting_time_l2436_243635

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

end NUMINAMATH_CALUDE_trains_meeting_time_l2436_243635


namespace NUMINAMATH_CALUDE_cubic_quadratic_fraction_inequality_l2436_243676

theorem cubic_quadratic_fraction_inequality (s r : ℝ) (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_fraction_inequality_l2436_243676


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2436_243691

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define one asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Statement: The distance from the focus to the asymptote is √3
theorem focus_to_asymptote_distance :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola x y ∧ asymptote x y ∧
  (Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2436_243691


namespace NUMINAMATH_CALUDE_system_sample_fourth_number_l2436_243686

/-- Represents a system sampling of employees -/
structure SystemSample where
  total : Nat
  sample_size : Nat
  sample : Finset Nat

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset Nat) : Prop :=
  ∃ (a d : Nat), ∀ (x : Nat), x ∈ s → ∃ (k : Nat), x = a + k * d

/-- The main theorem about the system sampling -/
theorem system_sample_fourth_number
  (s : SystemSample)
  (h_total : s.total = 52)
  (h_size : s.sample_size = 4)
  (h_contains : {6, 32, 45} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  19 ∈ s.sample :=
sorry

end NUMINAMATH_CALUDE_system_sample_fourth_number_l2436_243686


namespace NUMINAMATH_CALUDE_x_0_interval_l2436_243625

theorem x_0_interval (x_0 : ℝ) (h1 : x_0 ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x_0 + Real.cos x_0 = 2/3) : 
  x_0 ∈ Set.Ioo (7*π/12) (3*π/4) := by
  sorry

end NUMINAMATH_CALUDE_x_0_interval_l2436_243625


namespace NUMINAMATH_CALUDE_exactly_four_intersections_l2436_243637

-- Define the graphs
def graph1 (B : ℝ) (x y : ℝ) : Prop := y = B * x^2
def graph2 (x y : ℝ) : Prop := y^2 + 2 * x^2 = 5 + 6 * y

-- Define an intersection point
def is_intersection (B : ℝ) (x y : ℝ) : Prop :=
  graph1 B x y ∧ graph2 x y

-- Theorem statement
theorem exactly_four_intersections (B : ℝ) (h : B > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    is_intersection B x₁ y₁ ∧
    is_intersection B x₂ y₂ ∧
    is_intersection B x₃ y₃ ∧
    is_intersection B x₄ y₄ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧
    (x₁ ≠ x₄ ∨ y₁ ≠ y₄) ∧
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₂ ≠ x₄ ∨ y₂ ≠ y₄) ∧
    (x₃ ≠ x₄ ∨ y₃ ≠ y₄) ∧
    ∀ (x y : ℝ), is_intersection B x y →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨
       (x = x₃ ∧ y = y₃) ∨ (x = x₄ ∧ y = y₄)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_four_intersections_l2436_243637


namespace NUMINAMATH_CALUDE_product_not_always_greater_l2436_243640

theorem product_not_always_greater : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end NUMINAMATH_CALUDE_product_not_always_greater_l2436_243640


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l2436_243671

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (a b : ℕ), n = 12 * a + 6 * b

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l2436_243671


namespace NUMINAMATH_CALUDE_m_is_fengli_fengli_condition_l2436_243693

/-- Definition of a Fengli number -/
def is_fengli (n : ℤ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

/-- M is a Fengli number -/
theorem m_is_fengli (x y : ℕ) :
  is_fengli (x^2 + 2*x*y + 2*y^2) :=
sorry

/-- Theorem about the value of m for p to be a Fengli number -/
theorem fengli_condition (x y : ℕ) (m : ℤ) (h : x > y) (h' : y > 0) :
  is_fengli (4*x^2 + m*x*y + 2*y^2 - 10*y + 25) ↔ m = 4 ∨ m = -4 :=
sorry

end NUMINAMATH_CALUDE_m_is_fengli_fengli_condition_l2436_243693


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l2436_243658

theorem positive_integer_solutions_of_inequality :
  ∀ x : ℕ+, 9 - 3 * (x : ℝ) > 0 ↔ x = 1 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l2436_243658


namespace NUMINAMATH_CALUDE_f_divisible_by_8_l2436_243653

def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem f_divisible_by_8 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, f n = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_f_divisible_by_8_l2436_243653


namespace NUMINAMATH_CALUDE_pentagon_percentage_is_fifty_percent_l2436_243610

/-- Represents a tiling of the plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square tile -/
  smallSquaresPerTile : ℕ
  /-- The number of smaller squares that form parts of pentagons -/
  smallSquaresInPentagons : ℕ

/-- Calculates the percentage of the plane enclosed by pentagons -/
def pentagonPercentage (tiling : PlaneTiling) : ℚ :=
  (tiling.smallSquaresInPentagons : ℚ) / (tiling.smallSquaresPerTile : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_percentage_is_fifty_percent (tiling : PlaneTiling) 
  (h1 : tiling.smallSquaresPerTile = 16)
  (h2 : tiling.smallSquaresInPentagons = 8) : 
  pentagonPercentage tiling = 50 := by
  sorry

#eval pentagonPercentage { smallSquaresPerTile := 16, smallSquaresInPentagons := 8 }

end NUMINAMATH_CALUDE_pentagon_percentage_is_fifty_percent_l2436_243610


namespace NUMINAMATH_CALUDE_max_m_value_l2436_243643

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  (∀ m : ℝ, a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b ≥ 0) →
  (∃ m : ℝ, m = 3/2 ∧ a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b = 0 ∧
    ∀ m' : ℝ, m' > m → a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m'*a*b < 0) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l2436_243643


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l2436_243615

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10 ∧ 0 < c ∧ c < 10 ∧ 0 < d ∧ d < 10 →
  (a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30) →
  a + b + c + d = 22 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l2436_243615


namespace NUMINAMATH_CALUDE_point_constraints_l2436_243677

theorem point_constraints (x y : ℝ) :
  x^2 + y^2 ≤ 2 →
  -1 ≤ x / (x + y) →
  x / (x + y) ≤ 1 →
  0 ≤ y ∧ -2*x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_point_constraints_l2436_243677


namespace NUMINAMATH_CALUDE_income_182400_max_income_l2436_243627

/-- Represents the income function for the large grain grower --/
def income_function (original_land : ℝ) (original_income_per_mu : ℝ) (additional_land : ℝ) : ℝ :=
  original_land * original_income_per_mu + additional_land * (original_income_per_mu - 2 * additional_land)

/-- Theorem for the total income of 182,400 yuan --/
theorem income_182400 (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (income_function original_land original_income_per_mu x = 182400 ∧ (x = 100 ∨ x = 120))) :=
sorry

/-- Theorem for the maximum total income --/
theorem max_income (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (∀ y : ℝ, income_function original_land original_income_per_mu x ≥ income_function original_land original_income_per_mu y) ∧
             x = 110 ∧
             income_function original_land original_income_per_mu x = 182600) :=
sorry

end NUMINAMATH_CALUDE_income_182400_max_income_l2436_243627


namespace NUMINAMATH_CALUDE_fractional_unit_problem_l2436_243666

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

theorem fractional_unit_problem (n d : ℕ) (h1 : n = 5) (h2 : d = 11) :
  let u := fractional_unit n d
  (u = 1 / 11) ∧
  (n / d + 6 * u = 2) ∧
  (n / d - 5 * u = 1) :=
sorry

end NUMINAMATH_CALUDE_fractional_unit_problem_l2436_243666
