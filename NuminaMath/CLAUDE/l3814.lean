import Mathlib

namespace NUMINAMATH_CALUDE_tan_difference_l3814_381401

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + Real.pi/4) = -1/3) : 
  Real.tan (β - Real.pi/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l3814_381401


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l3814_381443

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 250) 
  (h2 : initial_tagged = 50) 
  (h3 : second_catch = 50) :
  (initial_tagged : ℚ) / total_fish = (initial_tagged : ℚ) / second_catch → 
  (initial_tagged : ℚ) * second_catch / total_fish = 10 :=
by sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l3814_381443


namespace NUMINAMATH_CALUDE_min_dimes_needed_l3814_381452

def jacket_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickels : ℕ := 15

def min_dimes : ℕ := 23

theorem min_dimes_needed (d : ℕ) : 
  (ten_dollar_bills * 10 + quarters * 0.25 + nickels * 0.05 + d * 0.10 : ℚ) ≥ jacket_cost → 
  d ≥ min_dimes := by
sorry

end NUMINAMATH_CALUDE_min_dimes_needed_l3814_381452


namespace NUMINAMATH_CALUDE_jose_ducks_count_l3814_381412

/-- Given that Jose has 28 chickens and 46 fowls in total, prove that he has 18 ducks. -/
theorem jose_ducks_count (chickens : ℕ) (total_fowls : ℕ) (ducks : ℕ) 
    (h1 : chickens = 28) 
    (h2 : total_fowls = 46) 
    (h3 : total_fowls = chickens + ducks) : 
  ducks = 18 := by
  sorry

end NUMINAMATH_CALUDE_jose_ducks_count_l3814_381412


namespace NUMINAMATH_CALUDE_equation_properties_l3814_381426

-- Define the equation
def equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (p : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ p = 0 ∧ equation x₂ p = 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 3 * x₁ * x₂

-- Theorem statement
theorem equation_properties :
  (∀ p : ℝ, has_two_distinct_real_roots p) ∧
  (∀ p x₁ x₂ : ℝ, equation x₁ p = 0 → equation x₂ p = 0 → 
    roots_condition x₁ x₂ → p = 1 ∨ p = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_properties_l3814_381426


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3814_381403

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 3) : x^2 + (1 / x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3814_381403


namespace NUMINAMATH_CALUDE_least_frood_count_l3814_381453

/-- The function representing points earned by dropping n froods -/
def drop_points (n : ℕ) : ℚ := n * (n + 1) / 2

/-- The function representing points earned by eating n froods -/
def eat_points (n : ℕ) : ℚ := 20 * n

/-- The theorem stating that 40 is the least positive integer for which
    dropping froods earns more points than eating them -/
theorem least_frood_count : ∀ n : ℕ, n > 0 → (drop_points n > eat_points n ↔ n ≥ 40) := by
  sorry

end NUMINAMATH_CALUDE_least_frood_count_l3814_381453


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_ratio_l3814_381432

/-- Given a hyperbola x²/a - y²/a = 1 with a > 0, prove that |FP|/|MN| = √2/2 where:
    F is the right focus
    M and N are intersection points of any line through F with the right branch
    P is the intersection of the perpendicular bisector of MN with the x-axis -/
theorem hyperbola_focus_distance_ratio (a : ℝ) (h : a > 0) :
  ∃ (F M N P : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a - y^2/a = 1 → 
      (∃ (t : ℝ), (x, y) = M ∨ (x, y) = N) → 
      (F.1 > 0 ∧ F.2 = 0) ∧
      (∃ (m : ℝ), (M.2 - F.2) = m * (M.1 - F.1) ∧ (N.2 - F.2) = m * (N.1 - F.1)) ∧
      (P.2 = 0 ∧ P.1 = (M.1 + N.1)/2 + (M.2 + N.2)^2 / (2 * (M.1 + N.1)))) →
    ‖F - P‖ / ‖M - N‖ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_ratio_l3814_381432


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3814_381477

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (k : ℝ), m^2 + m - 2 + (m^2 - 1) * I = k * I) → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3814_381477


namespace NUMINAMATH_CALUDE_inverse_mod_53_l3814_381490

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 38) : (36⁻¹ : ZMod 53) = 15 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l3814_381490


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3814_381429

/-- Given a truck that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas, 
    assuming a constant rate of fuel consumption. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_gas : ℝ) (new_gas : ℝ) : 
  initial_distance = 300 ∧ initial_gas = 10 ∧ new_gas = 15 →
  (new_gas * initial_distance) / initial_gas = 450 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3814_381429


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3814_381470

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define number of atoms in the compound
def num_Ca : ℕ := 1
def num_O : ℕ := 2
def num_H : ℕ := 2

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Ca : ℝ) * atomic_weight_Ca + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem compound_molecular_weight : 
  molecular_weight = 74.10 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3814_381470


namespace NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l3814_381408

/-- The number of fish in Jonah's aquarium after all the changes -/
def final_fish_count (initial_fish : ℕ) (added_fish : ℕ) (exchanged_fish : ℕ) (x : ℕ) : ℤ :=
  (initial_fish + added_fish : ℤ) - 2 * x + exchanged_fish

theorem jonah_aquarium_fish_count :
  final_fish_count 14 2 3 x = 19 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l3814_381408


namespace NUMINAMATH_CALUDE_connect_points_is_valid_l3814_381498

-- Define a type for geometric points
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for geometric drawing operations
inductive DrawingOperation
  | DrawRay (start : Point) (length : ℝ)
  | ConnectPoints (a b : Point)
  | DrawMidpoint (a b : Point)
  | DrawDistance (a b : Point)

-- Define a predicate for valid drawing operations
def IsValidDrawingOperation : DrawingOperation → Prop
  | DrawingOperation.ConnectPoints _ _ => True
  | _ => False

-- Theorem statement
theorem connect_points_is_valid :
  ∀ (a b : Point), IsValidDrawingOperation (DrawingOperation.ConnectPoints a b) :=
by sorry

end NUMINAMATH_CALUDE_connect_points_is_valid_l3814_381498


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3814_381464

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^5 + a^4*b + b^5 = -16674 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3814_381464


namespace NUMINAMATH_CALUDE_martha_cakes_l3814_381433

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : Float) (cakes_per_child : Float) : Float :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3.0 18.0 = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l3814_381433


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3814_381465

theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 24 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 6 * (1200^(2/3)) := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3814_381465


namespace NUMINAMATH_CALUDE_lawn_width_is_60_l3814_381467

/-- Represents a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  road_area : ℝ

/-- Theorem: The width of the lawn is 60 meters -/
theorem lawn_width_is_60 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.road_width = 10)
  (h3 : lawn.road_area = 1300)
  : lawn.width = 60 := by
  sorry

#check lawn_width_is_60

end NUMINAMATH_CALUDE_lawn_width_is_60_l3814_381467


namespace NUMINAMATH_CALUDE_equilateral_triangle_coverage_l3814_381466

/-- An equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The union of a set of equilateral triangles -/
def UnionOfTriangles (triangles : Set EquilateralTriangle) : Set (ℝ × ℝ) := sorry

/-- A triangle is contained in a set of points -/
def TriangleContainedIn (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

theorem equilateral_triangle_coverage 
  (Δ : EquilateralTriangle) 
  (a b : ℝ)
  (h_a : Δ.sideLength = a)
  (h_b : b > 0)
  (h_five : ∃ (five_triangles : Finset EquilateralTriangle), 
    five_triangles.card = 5 ∧ 
    (∀ t ∈ five_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles five_triangles.toSet)) :
  ∃ (four_triangles : Finset EquilateralTriangle),
    four_triangles.card = 4 ∧
    (∀ t ∈ four_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles four_triangles.toSet) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_coverage_l3814_381466


namespace NUMINAMATH_CALUDE_flower_bed_length_l3814_381487

/-- A rectangular flower bed with given area and width has a specific length -/
theorem flower_bed_length (area width : ℝ) (h1 : area = 35) (h2 : width = 5) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_length_l3814_381487


namespace NUMINAMATH_CALUDE_bread_left_l3814_381449

theorem bread_left (total : ℕ) (bomi_ate : ℕ) (yejun_ate : ℕ) 
  (h1 : total = 1000)
  (h2 : bomi_ate = 350)
  (h3 : yejun_ate = 500) :
  total - (bomi_ate + yejun_ate) = 150 := by
  sorry

end NUMINAMATH_CALUDE_bread_left_l3814_381449


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3814_381496

/-- The eccentricity of a hyperbola defined by x²/(1+m) - y²/(1-m) = 1 with m > 0 is between 1 and √2 -/
theorem hyperbola_eccentricity_range (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (1 + m) - y^2 / (1 - m) = 1}
  let e := Real.sqrt 2 / Real.sqrt (1 + m)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3814_381496


namespace NUMINAMATH_CALUDE_palindrome_power_sum_l3814_381463

/-- A function to check if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The main theorem stating the condition for 2^n + 2^m + 1 to be a palindrome -/
theorem palindrome_power_sum (m n : ℕ) : 
  isPalindrome (2^n + 2^m + 1) ↔ m ≤ 9 ∨ n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_palindrome_power_sum_l3814_381463


namespace NUMINAMATH_CALUDE_log_division_simplification_l3814_381420

theorem log_division_simplification :
  (Real.log 256 / Real.log 16) / (Real.log (1/256) / Real.log 16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l3814_381420


namespace NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l3814_381440

theorem max_inscribed_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (rect_area : ℝ),
    (∀ (inscribed_rect_area : ℝ),
      inscribed_rect_area ≤ rect_area) ∧
    rect_area = (a * b) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l3814_381440


namespace NUMINAMATH_CALUDE_polyhedron_diagonals_l3814_381416

/-- A polyhedron with the given properties -/
structure Polyhedron :=
  (num_vertices : ℕ)
  (edges_per_vertex : ℕ)

/-- The number of interior diagonals in a polyhedron -/
def interior_diagonals (p : Polyhedron) : ℕ :=
  (p.num_vertices * (p.num_vertices - 1 - p.edges_per_vertex)) / 2

/-- Theorem: A polyhedron with 15 vertices and 6 edges per vertex has 60 interior diagonals -/
theorem polyhedron_diagonals :
  ∀ (p : Polyhedron), p.num_vertices = 15 ∧ p.edges_per_vertex = 6 →
  interior_diagonals p = 60 :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_diagonals_l3814_381416


namespace NUMINAMATH_CALUDE_binomial_18_10_l3814_381492

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 42328 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l3814_381492


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3814_381456

theorem simplify_fraction_product : 8 * (15 / 9) * (-21 / 35) = -8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3814_381456


namespace NUMINAMATH_CALUDE_number_problem_l3814_381405

theorem number_problem (x : ℝ) : 
  (0.25 * x = 0.20 * 650 + 190) → x = 1280 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3814_381405


namespace NUMINAMATH_CALUDE_henrys_socks_l3814_381486

theorem henrys_socks (a b c : ℕ) : 
  a + b + c = 15 →
  2 * a + 3 * b + 5 * c = 36 →
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  a = 11 :=
by sorry

end NUMINAMATH_CALUDE_henrys_socks_l3814_381486


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3814_381457

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 5 * X^2 + 3 * X - 4
  let p₂ : Polynomial ℝ := 6 * X^3 + 2 * X^2 - X + 7
  p₁ * p₂ = 30 * X^5 + 28 * X^4 - 23 * X^3 + 24 * X^2 + 25 * X - 28 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3814_381457


namespace NUMINAMATH_CALUDE_parabola_equation_l3814_381418

/-- A parabola with vertex at the origin and axis of symmetry along a coordinate axis -/
structure Parabola where
  a : ℝ
  axis : Bool -- true for y-axis, false for x-axis

/-- The point (-4, -2) -/
def P : ℝ × ℝ := (-4, -2)

/-- Check if a point satisfies the parabola equation -/
def satisfiesEquation (p : Parabola) (point : ℝ × ℝ) : Prop :=
  if p.axis then
    point.2^2 = p.a * point.1
  else
    point.1^2 = p.a * point.2

theorem parabola_equation :
  ∃ (p1 p2 : Parabola),
    satisfiesEquation p1 P ∧
    satisfiesEquation p2 P ∧
    p1.axis = true ∧
    p2.axis = false ∧
    p1.a = -1 ∧
    p2.a = -8 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3814_381418


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3814_381475

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3814_381475


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l3814_381491

/-- The outer perimeter of a square fence with evenly spaced posts -/
def fence_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let posts_per_side : ℕ := num_posts / 4
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℚ := (gaps_per_side : ℚ) * gap_width + (posts_per_side : ℚ) * post_width
  4 * side_length

/-- Theorem: The outer perimeter of a square fence with 32 posts, each 5 inches wide,
    and evenly spaced with 4 feet between adjacent posts, is 125 1/3 feet. -/
theorem square_fence_perimeter :
  fence_perimeter 32 (5/12) 4 = 125 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l3814_381491


namespace NUMINAMATH_CALUDE_sequence_sum_l3814_381413

-- Define the sequence type
def Sequence := Fin 8 → ℝ

-- Define the property of the sum of any three consecutive terms being 30
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 6, s i + s (i + 1) + s (i + 2) = 30

-- Main theorem
theorem sequence_sum (s : Sequence) 
  (h1 : s 2 = 5)  -- C = 5 (index 2 in 0-based indexing)
  (h2 : ConsecutiveSum s) : s 0 + s 7 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3814_381413


namespace NUMINAMATH_CALUDE_f_of_3_eq_3_l3814_381407

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The main equation defining f -/
axiom f_eq (x : ℝ) : (x^(3^5 - 1) - 1) * f x = (x + 1) * (x^2 + 1) * (x^3 + 1) * (x^(3^4) + 1) - 1

/-- Theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 : f 3 = 3 := by sorry

end NUMINAMATH_CALUDE_f_of_3_eq_3_l3814_381407


namespace NUMINAMATH_CALUDE_floor_a4_div_a3_l3814_381445

def a (k : ℕ) : ℕ := Nat.choose 100 (k + 1)

theorem floor_a4_div_a3 : ⌊(a 4 : ℚ) / (a 3 : ℚ)⌋ = 19 := by sorry

end NUMINAMATH_CALUDE_floor_a4_div_a3_l3814_381445


namespace NUMINAMATH_CALUDE_total_apples_l3814_381434

/-- Proves that the total number of apples given out is 150, given that Harold gave 25 apples to each of 6 people. -/
theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l3814_381434


namespace NUMINAMATH_CALUDE_magnitude_of_a_l3814_381485

def a (t : ℝ) : ℝ × ℝ := (1, t)
def b (t : ℝ) : ℝ × ℝ := (-1, t)

theorem magnitude_of_a (t : ℝ) :
  (2 * a t - b t) • b t = 0 → ‖a t‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_a_l3814_381485


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l3814_381428

def S : Set Int := {7, 25, -1, 12, -3}

theorem smallest_sum_of_three (s : Set Int) (h : s = S) :
  (∃ (a b c : Int), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a + b + c = 3 ∧
    ∀ (x y z : Int), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
      x + y + z ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l3814_381428


namespace NUMINAMATH_CALUDE_power_equality_l3814_381479

theorem power_equality (n : ℕ) : 4^n = 64^2 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3814_381479


namespace NUMINAMATH_CALUDE_age_difference_l3814_381436

/-- Given that the overall age of x and y is 19 years greater than the overall age of y and z,
    prove that Z is 1.9 decades younger than X. -/
theorem age_difference (x y z : ℕ) (h : x + y = y + z + 19) :
  (x - z : ℚ) / 10 = 1.9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3814_381436


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_min_value_achieved_l3814_381482

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / a₀ + 2 / b₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_min_value_achieved_l3814_381482


namespace NUMINAMATH_CALUDE_unique_solution_l3814_381447

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - 23*y + 66*z + 612 = 0 ∧
  y^2 + 62*x - 20*z + 296 = 0 ∧
  z^2 - 22*x + 67*y + 505 = 0

/-- The theorem stating that (-20, -22, -23) is the unique solution to the system -/
theorem unique_solution :
  ∃! (x y z : ℝ), system x y z ∧ x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3814_381447


namespace NUMINAMATH_CALUDE_system_solution_unique_l3814_381459

theorem system_solution_unique :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
    x^4 + y^4 - x^2*y^2 = 13 ∧
    x^2 - y^2 + 2*x*y = 1 ∧
    x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3814_381459


namespace NUMINAMATH_CALUDE_max_value_x_minus_y_l3814_381499

theorem max_value_x_minus_y :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧
  (∀ x y : ℝ, 3 * (x^2 + y^2) = x + y → x - y ≤ max) ∧
  (∃ x y : ℝ, 3 * (x^2 + y^2) = x + y ∧ x - y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_minus_y_l3814_381499


namespace NUMINAMATH_CALUDE_negation_existence_equivalence_l3814_381437

theorem negation_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_equivalence_l3814_381437


namespace NUMINAMATH_CALUDE_min_value_theorem_l3814_381422

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + 3*z = 3) : 
  (2*(x + y)) / (x*y*z) ≥ 14.2222 := by
sorry

#eval (8 : ℚ) / (9 : ℚ) * 16

end NUMINAMATH_CALUDE_min_value_theorem_l3814_381422


namespace NUMINAMATH_CALUDE_broadway_ticket_price_l3814_381442

theorem broadway_ticket_price (num_adults num_children : ℕ) (total_amount : ℚ) :
  num_adults = 400 →
  num_children = 200 →
  total_amount = 16000 →
  ∃ (adult_price child_price : ℚ),
    adult_price = 2 * child_price ∧
    num_adults * adult_price + num_children * child_price = total_amount ∧
    adult_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_broadway_ticket_price_l3814_381442


namespace NUMINAMATH_CALUDE_domain_of_sqrt_2cos_minus_1_l3814_381476

/-- The domain of f(x) = √(2cos(x) - 1) -/
theorem domain_of_sqrt_2cos_minus_1 (x : ℝ) : 
  (∃ f : ℝ → ℝ, f x = Real.sqrt (2 * Real.cos x - 1)) ↔ 
  (∃ k : ℤ, 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_2cos_minus_1_l3814_381476


namespace NUMINAMATH_CALUDE_sequence_term_value_l3814_381461

/-- Given a finite sequence {a_n} with m terms, where S(n) represents the sum of all terms
    starting from the n-th term, prove that a_n = -2n - 1 when 1 ≤ n < m. -/
theorem sequence_term_value (m : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) 
    (h1 : 1 ≤ n) (h2 : n < m) (h3 : ∀ k, 1 ≤ k → k ≤ m → S k = k^2) :
  a n = -2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_value_l3814_381461


namespace NUMINAMATH_CALUDE_gmat_scores_l3814_381410

theorem gmat_scores (u v : ℝ) (h1 : u > v) (h2 : u - v = (u + v) / 2) : v / u = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l3814_381410


namespace NUMINAMATH_CALUDE_added_value_expression_max_value_m_gt_1_max_value_m_le_1_l3814_381430

noncomputable section

variables {a m : ℝ} (h_a : a > 0) (h_m : m > 0)

def x_range (a m : ℝ) : Set ℝ := Set.Ioo 0 ((2 * a * m) / (2 * m + 1))

def y (a x : ℝ) : ℝ := 8 * (a - x) * x^2

theorem added_value_expression (x : ℝ) (hx : x ∈ x_range a m) :
  y a x = 8 * (a - x) * x^2 := by sorry

theorem max_value_m_gt_1 (h_m_gt_1 : m > 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 / 27) * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

theorem max_value_m_le_1 (h_m_le_1 : 0 < m ∧ m ≤ 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 * m^2) / (2 * m + 1)^3 * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

end

end NUMINAMATH_CALUDE_added_value_expression_max_value_m_gt_1_max_value_m_le_1_l3814_381430


namespace NUMINAMATH_CALUDE_three_squares_decomposition_l3814_381438

theorem three_squares_decomposition (n : ℤ) (h : n > 5) :
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_three_squares_decomposition_l3814_381438


namespace NUMINAMATH_CALUDE_choose_two_from_four_l3814_381415

theorem choose_two_from_four (n : ℕ) (h : n = 4) : Nat.choose n 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_l3814_381415


namespace NUMINAMATH_CALUDE_unique_sum_product_solution_l3814_381424

theorem unique_sum_product_solution (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₂ := S - x₂
  (∀ x y : ℝ, x + y = S ∧ x * y = P ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_product_solution_l3814_381424


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3814_381471

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : geometric_sequence a r)
  (h_roots : a 2 * a 6 = 64) :
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3814_381471


namespace NUMINAMATH_CALUDE_average_cookies_per_package_l3814_381441

def cookie_counts : List Nat := [9, 11, 13, 19, 23, 27]

theorem average_cookies_per_package : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 17 := by
  sorry

end NUMINAMATH_CALUDE_average_cookies_per_package_l3814_381441


namespace NUMINAMATH_CALUDE_election_vote_ratio_l3814_381497

theorem election_vote_ratio :
  let joey_votes : ℕ := 8
  let barry_votes : ℕ := 2 * (joey_votes + 3)
  let marcy_votes : ℕ := 66
  (marcy_votes : ℚ) / barry_votes = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l3814_381497


namespace NUMINAMATH_CALUDE_equation_solution_l3814_381439

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 6*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3*Real.sqrt 3 ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3814_381439


namespace NUMINAMATH_CALUDE_min_value_theorem_l3814_381404

theorem min_value_theorem (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3814_381404


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l3814_381481

def A : Set ℝ := {2, 3}
def B (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 1/2 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l3814_381481


namespace NUMINAMATH_CALUDE_george_movie_cost_l3814_381409

/-- The total cost of George's visit to the movie theater -/
def total_cost (ticket_price : ℝ) (nachos_price : ℝ) : ℝ :=
  ticket_price + nachos_price

/-- Theorem: George's total cost for the movie theater visit is $24 -/
theorem george_movie_cost :
  ∀ (ticket_price : ℝ) (nachos_price : ℝ),
    ticket_price = 16 →
    nachos_price = ticket_price / 2 →
    total_cost ticket_price nachos_price = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_george_movie_cost_l3814_381409


namespace NUMINAMATH_CALUDE_pet_food_difference_l3814_381425

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_difference_l3814_381425


namespace NUMINAMATH_CALUDE_quadratic_polynomials_sum_l3814_381435

/-- Two distinct quadratic polynomials with the given properties have a + c = -600 -/
theorem quadratic_polynomials_sum (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  ∀ x y : ℝ, 
  (f ≠ g) →  -- f and g are distinct
  (g (-a/2) = 0) →  -- x-coordinate of vertex of f is a root of g
  (f (-c/2) = 0) →  -- x-coordinate of vertex of g is a root of f
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ g x ≥ m) →  -- f and g yield the same minimum value
  (f 150 = -200 ∧ g 150 = -200) →  -- f and g intersect at (150, -200)
  a + c = -600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_sum_l3814_381435


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3814_381489

theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →  -- angles are positive
    a + b + c = 180 →        -- sum of angles is 180°
    b = 3 * a →              -- ratio condition
    c = 4 * a →              -- ratio condition
    c = 90 :=                -- largest angle is 90°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3814_381489


namespace NUMINAMATH_CALUDE_angle_trig_sum_l3814_381458

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point P on its terminal side, prove that 2sinα + cosα = -2/5 -/
theorem angle_trig_sum (α : Real) (m : Real) (h1 : m < 0) :
  let P : Prod Real Real := (-4 * m, 3 * m)
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_sum_l3814_381458


namespace NUMINAMATH_CALUDE_shortest_side_of_octagon_l3814_381421

theorem shortest_side_of_octagon (x : ℝ) : 
  x > 0 →                             -- x is positive
  x^2 = 100 →                         -- combined area of cut-off triangles
  20 - x = 10 :=                      -- shortest side of octagon
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_octagon_l3814_381421


namespace NUMINAMATH_CALUDE_local_monotonicity_not_implies_global_l3814_381411

/-- A function that satisfies the local monotonicity condition but is not globally monotonic -/
def exists_locally_monotonic_not_globally : Prop :=
  ∃ (f : ℝ → ℝ), 
    (∀ a : ℝ, ∃ b : ℝ, b > a ∧ (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y ∨ f x ≥ f y)) ∧
    ¬(∀ x y : ℝ, x < y → f x ≤ f y ∨ f x ≥ f y)

theorem local_monotonicity_not_implies_global : exists_locally_monotonic_not_globally :=
sorry

end NUMINAMATH_CALUDE_local_monotonicity_not_implies_global_l3814_381411


namespace NUMINAMATH_CALUDE_palmer_photos_before_trip_l3814_381469

def photos_before_trip (first_week : ℕ) (second_week_multiplier : ℕ) (third_fourth_week : ℕ) (total_after_trip : ℕ) : ℕ :=
  total_after_trip - (first_week + second_week_multiplier * first_week + third_fourth_week)

theorem palmer_photos_before_trip :
  photos_before_trip 50 2 80 380 = 150 :=
by sorry

end NUMINAMATH_CALUDE_palmer_photos_before_trip_l3814_381469


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3814_381480

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3814_381480


namespace NUMINAMATH_CALUDE_jackie_phil_same_heads_l3814_381462

/-- The probability of getting heads for a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting heads for the biased coin -/
def biased_coin_prob : ℚ := 4/7

/-- The probability of getting k heads when flipping the three coins -/
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob)^2 * (1 - biased_coin_prob)
  | 1 => 2 * fair_coin_prob * (1 - fair_coin_prob) * (1 - biased_coin_prob) + 
         (1 - fair_coin_prob)^2 * biased_coin_prob
  | 2 => fair_coin_prob^2 * (1 - biased_coin_prob) + 
         2 * fair_coin_prob * (1 - fair_coin_prob) * biased_coin_prob
  | 3 => fair_coin_prob^2 * biased_coin_prob
  | _ => 0

/-- The probability that Jackie and Phil get the same number of heads -/
def prob_same_heads : ℚ :=
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 + (prob_k_heads 3)^2

theorem jackie_phil_same_heads : prob_same_heads = 123/392 := by
  sorry

end NUMINAMATH_CALUDE_jackie_phil_same_heads_l3814_381462


namespace NUMINAMATH_CALUDE_converse_of_square_inequality_l3814_381484

theorem converse_of_square_inequality :
  (∀ a b : ℝ, a > b → a^2 > b^2) →
  (∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_converse_of_square_inequality_l3814_381484


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3814_381446

theorem fixed_point_parabola (d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + d * x + 3 * d
  f (-3) = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3814_381446


namespace NUMINAMATH_CALUDE_dogSchoolCount_l3814_381468

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  roll : ℕ
  sitStay : ℕ
  stayRoll : ℕ
  sitRoll : ℕ
  allThree : ℕ
  none : ℕ

/-- Calculates the total number of dogs in the school -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.sitRoll - d.allThree) +
  (d.stayRoll - d.allThree) +
  (d.sitStay - d.allThree) +
  (d.sit - d.sitRoll - d.sitStay + d.allThree) +
  (d.stay - d.stayRoll - d.sitStay + d.allThree) +
  (d.roll - d.sitRoll - d.stayRoll + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the school is 84 -/
theorem dogSchoolCount (d : DogTricks)
  (h1 : d.sit = 50)
  (h2 : d.stay = 29)
  (h3 : d.roll = 34)
  (h4 : d.sitStay = 17)
  (h5 : d.stayRoll = 12)
  (h6 : d.sitRoll = 18)
  (h7 : d.allThree = 9)
  (h8 : d.none = 9) :
  totalDogs d = 84 := by
  sorry

end NUMINAMATH_CALUDE_dogSchoolCount_l3814_381468


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3814_381431

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
  (∃ (q1 : ℕ), 1428 = d * q1 + 9) ∧
  (∃ (q2 : ℕ), 2206 = d * q2 + 13) ∧
  (∀ (x : ℕ), x > 0 ∧
    (∃ (r1 : ℕ), 1428 = x * r1 + 9) ∧
    (∃ (r2 : ℕ), 2206 = x * r2 + 13) →
    x ≤ d) ∧
  d = 129 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3814_381431


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l3814_381472

/-- Represents the maximum distance a car can travel with tire swapping -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for the given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 24000 36000 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l3814_381472


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_not_constant_l3814_381414

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  a : Point
  b : Point
  c : Point
  equalSideLength : ℝ
  baseSideLength : ℝ

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : IsoscelesTriangle) : Prop := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Theorem: The sum of perpendiculars is not constant for all points inside the triangle -/
theorem sum_of_perpendiculars_not_constant (t : IsoscelesTriangle)
  (h1 : t.equalSideLength = 10)
  (h2 : t.baseSideLength = 8) :
  ∃ p1 p2 : Point,
    isInside p1 t ∧ isInside p2 t ∧
    perpendicularDistance p1 t.a t.b + perpendicularDistance p1 t.b t.c + perpendicularDistance p1 t.c t.a ≠
    perpendicularDistance p2 t.a t.b + perpendicularDistance p2 t.b t.c + perpendicularDistance p2 t.c t.a :=
by sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_not_constant_l3814_381414


namespace NUMINAMATH_CALUDE_evaluate_expression_l3814_381406

theorem evaluate_expression (x : ℝ) (hx : x ≠ 0) :
  (20 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3814_381406


namespace NUMINAMATH_CALUDE_bottle_sales_revenue_l3814_381483

/-- Calculate the total revenue from bottle sales -/
theorem bottle_sales_revenue : 
  let small_bottles : ℕ := 6000
  let big_bottles : ℕ := 14000
  let medium_bottles : ℕ := 9000
  let small_price : ℚ := 2
  let big_price : ℚ := 4
  let medium_price : ℚ := 3
  let small_sold_percent : ℚ := 20 / 100
  let big_sold_percent : ℚ := 23 / 100
  let medium_sold_percent : ℚ := 15 / 100
  
  let small_revenue := (small_bottles : ℚ) * small_sold_percent * small_price
  let big_revenue := (big_bottles : ℚ) * big_sold_percent * big_price
  let medium_revenue := (medium_bottles : ℚ) * medium_sold_percent * medium_price
  
  let total_revenue := small_revenue + big_revenue + medium_revenue
  
  total_revenue = 19330 := by sorry

end NUMINAMATH_CALUDE_bottle_sales_revenue_l3814_381483


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l3814_381478

/-- Represents a country in Maria's stamp collection. -/
inductive Country
| Brazil
| Peru
| France
| Spain

/-- Represents a decade in which stamps were issued. -/
inductive Decade
| Fifties
| Sixties
| Nineties

/-- The cost of a stamp in cents for a given country. -/
def stampCost (c : Country) : ℕ :=
  match c with
  | Country.Brazil => 7
  | Country.Peru => 5
  | Country.France => 7
  | Country.Spain => 6

/-- Whether a country is in South America. -/
def isSouthAmerican (c : Country) : Bool :=
  match c with
  | Country.Brazil => true
  | Country.Peru => true
  | _ => false

/-- The number of stamps Maria has for a given country and decade. -/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Brazil, Decade.Fifties => 6
  | Country.Brazil, Decade.Sixties => 9
  | Country.Peru, Decade.Fifties => 8
  | Country.Peru, Decade.Sixties => 6
  | _, _ => 0

/-- The total cost of stamps for a given country and decade, in cents. -/
def decadeCost (c : Country) (d : Decade) : ℕ :=
  stampCost c * stampCount c d

/-- The theorem stating the total cost of South American stamps issued before the 90s. -/
theorem south_american_stamps_cost :
  (decadeCost Country.Brazil Decade.Fifties +
   decadeCost Country.Brazil Decade.Sixties +
   decadeCost Country.Peru Decade.Fifties +
   decadeCost Country.Peru Decade.Sixties) = 175 := by
  sorry


end NUMINAMATH_CALUDE_south_american_stamps_cost_l3814_381478


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_2007_to_1024_minus_1_l3814_381494

theorem largest_power_of_two_dividing_2007_to_1024_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (2007^1024 - 1)) ∧
  (∀ (m : ℕ), m > 14 → ¬(2^m ∣ (2007^1024 - 1))) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_2007_to_1024_minus_1_l3814_381494


namespace NUMINAMATH_CALUDE_percentage_of_330_l3814_381417

theorem percentage_of_330 : (33 + 1/3 : ℚ) / 100 * 330 = 110 := by sorry

end NUMINAMATH_CALUDE_percentage_of_330_l3814_381417


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l3814_381495

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x * x + p.b * x + p.c

/-- Theorem stating that if the given inequality chain holds, then the leading coefficient is positive -/
theorem quadratic_coefficient_positive (p : QuadraticPolynomial) (n : ℤ) 
  (h : n < p.eval n ∧ p.eval n < p.eval (p.eval n) ∧ p.eval (p.eval n) < p.eval (p.eval (p.eval n))) :
  p.a > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l3814_381495


namespace NUMINAMATH_CALUDE_cube_sum_equals_one_l3814_381474

theorem cube_sum_equals_one (x y : ℝ) 
  (h1 : x * (x^4 + y^4) = y^5) 
  (h2 : x^2 * (x + y) ≠ y^3) : 
  x^3 + y^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_one_l3814_381474


namespace NUMINAMATH_CALUDE_parabola_directrix_l3814_381402

/-- The equation of the directrix of the parabola y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ p : ℝ, p = 4 ∧ y = p/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3814_381402


namespace NUMINAMATH_CALUDE_range_of_a_l3814_381427

open Real

theorem range_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x₁ : ℝ, x₁ > 0 → ∀ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3814_381427


namespace NUMINAMATH_CALUDE_apple_cost_18_pounds_l3814_381493

/-- The cost of apples given a specific rate and weight -/
def appleCost (rate : ℚ) (rateWeight : ℚ) (weight : ℚ) : ℚ :=
  (rate * weight) / rateWeight

/-- Theorem: The cost of 18 pounds of apples at a rate of $6 for 6 pounds is $18 -/
theorem apple_cost_18_pounds :
  appleCost 6 6 18 = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_18_pounds_l3814_381493


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3814_381419

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3814_381419


namespace NUMINAMATH_CALUDE_knife_percentage_after_trade_l3814_381460

/-- Represents Carolyn's silverware set -/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set -/
def total (s : SilverwareSet) : ℕ := s.knives + s.forks + s.spoons

/-- Represents the trade operation -/
def trade (s : SilverwareSet) (knivesGained spoonsTrade : ℕ) : SilverwareSet :=
  { knives := s.knives + knivesGained,
    forks := s.forks,
    spoons := s.spoons - spoonsTrade }

/-- Calculates the percentage of knives in a silverware set -/
def knifePercentage (s : SilverwareSet) : ℚ :=
  (s.knives : ℚ) / (total s : ℚ) * 100

theorem knife_percentage_after_trade :
  let initialSet : SilverwareSet := { knives := 6, forks := 12, spoons := 6 * 3 }
  let finalSet := trade initialSet 10 6
  knifePercentage finalSet = 40 := by sorry

end NUMINAMATH_CALUDE_knife_percentage_after_trade_l3814_381460


namespace NUMINAMATH_CALUDE_subtraction_problem_l3814_381488

theorem subtraction_problem (n : ℝ) (h : n = 5) : ∃! x : ℝ, 7 * n - x = 2 * n + 10 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3814_381488


namespace NUMINAMATH_CALUDE_parabola_directrix_l3814_381451

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (p : ℝ), p = 1/2 ∧ x = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3814_381451


namespace NUMINAMATH_CALUDE_new_person_weight_l3814_381423

theorem new_person_weight
  (n : ℕ)
  (initial_average : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 4)
  (h3 : replaced_weight = 55)
  : ∃ (new_weight : ℝ),
    n * (initial_average + weight_increase) = (n - 1) * initial_average + new_weight ∧
    new_weight = 87
  := by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3814_381423


namespace NUMINAMATH_CALUDE_two_questions_suffice_l3814_381450

-- Define the possible types of siblings
inductive SiblingType
  | Truthful
  | Unpredictable

-- Define a sibling
structure Sibling :=
  (type : SiblingType)

-- Define the farm setup
structure Farm :=
  (siblings : Fin 3 → Sibling)
  (correct_path : Nat)

-- Define the possible answers to a question
inductive Answer
  | Yes
  | No

-- Define a question as a function from a sibling to an answer
def Question := Sibling → Answer

-- Define the theorem
theorem two_questions_suffice (farm : Farm) :
  ∃ (q1 q2 : Question), ∀ (i j : Fin 3),
    (farm.siblings i).type = SiblingType.Truthful →
    (farm.siblings j).type = SiblingType.Truthful →
    i ≠ j →
    ∃ (f : Answer → Answer → Nat),
      f (q1 (farm.siblings i)) (q2 (farm.siblings j)) = farm.correct_path :=
sorry


end NUMINAMATH_CALUDE_two_questions_suffice_l3814_381450


namespace NUMINAMATH_CALUDE_parabola_distance_property_l3814_381448

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q
def Q_condition (Q P : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (px, py) := P
  (qx - 2, qy) = -4 * (px - 2, py)

-- Theorem statement
theorem parabola_distance_property (Q P : ℝ × ℝ) :
  directrix P.1 →
  parabola Q.1 Q.2 →
  Q_condition Q P →
  Real.sqrt ((Q.1 - 2)^2 + Q.2^2) = 20 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_property_l3814_381448


namespace NUMINAMATH_CALUDE_canvas_cost_l3814_381454

/-- Proves that the cost of canvases is $40.00 given the specified conditions -/
theorem canvas_cost (total_spent easel_cost paintbrush_cost canvas_cost : ℚ) : 
  total_spent = 90 ∧ 
  easel_cost = 15 ∧ 
  paintbrush_cost = 15 ∧ 
  total_spent = canvas_cost + (1/2 * canvas_cost) + easel_cost + paintbrush_cost →
  canvas_cost = 40 := by
sorry

end NUMINAMATH_CALUDE_canvas_cost_l3814_381454


namespace NUMINAMATH_CALUDE_employee_preference_city_y_l3814_381400

/-- Proves that given the conditions of the employee relocation problem,
    the percentage of employees preferring city Y is 40%. -/
theorem employee_preference_city_y (
  total_employees : ℕ)
  (relocated_to_x_percent : ℚ)
  (relocated_to_y_percent : ℚ)
  (max_preferred_relocation : ℕ)
  (h1 : total_employees = 200)
  (h2 : relocated_to_x_percent = 30 / 100)
  (h3 : relocated_to_y_percent = 70 / 100)
  (h4 : relocated_to_x_percent + relocated_to_y_percent = 1)
  (h5 : max_preferred_relocation = 140) :
  ∃ (prefer_y_percent : ℚ),
    prefer_y_percent = 40 / 100 ∧
    prefer_y_percent * total_employees = max_preferred_relocation - relocated_to_x_percent * total_employees :=
by sorry

end NUMINAMATH_CALUDE_employee_preference_city_y_l3814_381400


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l3814_381444

/-- An isosceles right triangle with hypotenuse 6√2 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  is_isosceles_right : hypotenuse = 6 * Real.sqrt 2

theorem isosceles_right_triangle_area_and_perimeter 
  (t : IsoscelesRightTriangle) : 
  ∃ (leg : ℝ), 
    leg^2 + leg^2 = t.hypotenuse^2 ∧ 
    (1/2 * leg * leg = 18) ∧ 
    (leg + leg + t.hypotenuse = 12 + 6 * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_area_and_perimeter

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l3814_381444


namespace NUMINAMATH_CALUDE_circle_tangent_problem_l3814_381473

-- Define the circles
def Circle := ℝ × ℝ → Prop

-- Define the tangent line
def TangentLine (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the property of being square-free
def SquareFree (n : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ n) → False

-- Define the property of being relatively prime
def RelativelyPrime (a c : ℕ) : Prop := Nat.gcd a c = 1

theorem circle_tangent_problem (C₁ C₂ : Circle) (m : ℝ) (a b c : ℕ) :
  (∃ x y : ℝ, C₁ (x, y) ∧ C₂ (x, y)) →  -- Circles intersect
  C₁ (8, 6) ∧ C₂ (8, 6) →  -- Intersection point at (8,6)
  (∃ r₁ r₂ : ℝ, r₁ * r₂ = 75) →  -- Product of radii is 75
  (∀ x : ℝ, C₁ (x, 0) → x = 0) ∧ (∀ x : ℝ, C₂ (x, 0) → x = 0) →  -- x-axis is tangent
  (∀ x y : ℝ, C₁ (x, y) ∧ TangentLine m x y → x = 0) ∧ 
  (∀ x y : ℝ, C₂ (x, y) ∧ TangentLine m x y → x = 0) →  -- y = mx is tangent
  m > 0 →  -- m is positive
  m = (a : ℝ) * Real.sqrt (b : ℝ) / (c : ℝ) →  -- m in the form a√b/c
  a > 0 ∧ b > 0 ∧ c > 0 →  -- a, b, c are positive
  SquareFree b →  -- b is square-free
  RelativelyPrime a c →  -- a and c are relatively prime
  a + b + c = 282 := by  -- Conclusion
sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_circle_tangent_problem_l3814_381473


namespace NUMINAMATH_CALUDE_rhombus_area_l3814_381455

/-- The area of a rhombus with side length 13 cm and one diagonal 24 cm is 120 cm² -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 13 → diagonal1 = 24 → side ^ 2 = (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 → 
  (diagonal1 * diagonal2) / 2 = 120 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l3814_381455
