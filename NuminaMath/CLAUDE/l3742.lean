import Mathlib

namespace NUMINAMATH_CALUDE_m_range_proof_l3742_374273

/-- Definition of p -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- Definition of q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

/-- ¬p is a sufficient but not necessary condition for ¬q -/
def not_p_sufficient_not_necessary_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := 0 < m ∧ m ≤ 3

/-- Main theorem: Given the conditions, prove the range of m -/
theorem m_range_proof :
  ∀ m, not_p_sufficient_not_necessary_for_not_q m → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l3742_374273


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3742_374239

theorem consecutive_integers_sum (x : ℤ) :
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 75 →
  (x - 2) + (x + 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3742_374239


namespace NUMINAMATH_CALUDE_throne_occupant_identity_l3742_374291

-- Define the possible species
inductive Species
| Human
| Monkey

-- Define the possible truth-telling nature
inductive Nature
| Knight
| Liar

-- Define the statement made by A
def statement (s : Species) (n : Nature) : Prop :=
  ¬(s = Species.Monkey ∧ n = Nature.Knight)

-- Theorem to prove
theorem throne_occupant_identity :
  ∃ (s : Species) (n : Nature),
    statement s n ∧
    (n = Nature.Knight → statement s n = True) ∧
    (n = Nature.Liar → statement s n = False) ∧
    s = Species.Human ∧
    n = Nature.Knight := by
  sorry

end NUMINAMATH_CALUDE_throne_occupant_identity_l3742_374291


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3742_374218

theorem min_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3742_374218


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l3742_374217

/-- A square pyramid is a three-dimensional geometric shape with a square base and four triangular faces -/
structure SquarePyramid where
  base : Square
  apex : Point

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l3742_374217


namespace NUMINAMATH_CALUDE_f_composition_sqrt2_l3742_374265

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x + 1 else |x|

theorem f_composition_sqrt2 :
  f (f (-Real.sqrt 2)) = 3 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_sqrt2_l3742_374265


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l3742_374204

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l3742_374204


namespace NUMINAMATH_CALUDE_arccos_cos_gt_arcsin_sin_iff_l3742_374272

theorem arccos_cos_gt_arcsin_sin_iff (x : ℝ) : 
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < x ∧ x < 2 * (k + 1) * Real.pi) ↔ 
  Real.arccos (Real.cos x) > Real.arcsin (Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_arccos_cos_gt_arcsin_sin_iff_l3742_374272


namespace NUMINAMATH_CALUDE_trees_left_l3742_374266

theorem trees_left (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 150) 
  (h2 : dead_trees = 24) : 
  initial_trees - dead_trees = 126 := by
sorry

end NUMINAMATH_CALUDE_trees_left_l3742_374266


namespace NUMINAMATH_CALUDE_can_lids_per_box_l3742_374275

theorem can_lids_per_box 
  (num_boxes : ℕ) 
  (initial_lids : ℕ) 
  (final_total_lids : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : initial_lids = 14) 
  (h3 : final_total_lids = 53) :
  (final_total_lids - initial_lids) / num_boxes = 13 := by
  sorry

#check can_lids_per_box

end NUMINAMATH_CALUDE_can_lids_per_box_l3742_374275


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l3742_374279

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ = 32.5 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (φ * π / 180) → θ ≤ φ) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_least_positive_angle_theorem_l3742_374279


namespace NUMINAMATH_CALUDE_intersection_A_B_l3742_374284

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | x = 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3742_374284


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l3742_374207

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m at a rate of Rs. 400 per square metre is Rs. 8250 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 400 = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l3742_374207


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3742_374269

theorem smallest_number_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 3 ∧
  b % 4 = 2 ∧
  b % 6 = 2 ∧
  (∀ c : ℕ, c > 0 ∧ c % 5 = 3 ∧ c % 4 = 2 ∧ c % 6 = 2 → b ≤ c) ∧
  b = 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3742_374269


namespace NUMINAMATH_CALUDE_function_properties_l3742_374206

/-- Given f(x) = a(x+b)(x+c) and g(x) = xf(x) where a ≠ 0 and a, b, c ∈ ℝ,
    prove the following statements -/
theorem function_properties :
  ∃ (a b c : ℝ), a ≠ 0 ∧
    (∀ x, (a * (1 + x) * (x + b) * (x + c) = 0) ↔ (a * (1 - x) * (x + b) * (x + c) = 0)) ∧
    (∀ x, (2 * a * x = -(2 * a * (-x))) ∧ (a * (3 * x^2 + 2 * (b + c) * x + b * c) = a * (3 * (-x)^2 + 2 * (b + c) * (-x) + b * c))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3742_374206


namespace NUMINAMATH_CALUDE_increasing_sequence_count_l3742_374216

def sequence_count (n m : ℕ) : ℕ := Nat.choose (n + m - 1) m

theorem increasing_sequence_count : 
  let n := 675
  let m := 15
  sequence_count n m = Nat.choose 689 15 ∧ 689 % 1000 = 689 := by sorry

#eval sequence_count 675 15
#eval 689 % 1000

end NUMINAMATH_CALUDE_increasing_sequence_count_l3742_374216


namespace NUMINAMATH_CALUDE_unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l3742_374238

/-- The system of equations has a unique solution when a = 2 -/
theorem unique_solution_at_two (x y a : ℝ) : 
  (y = 1 - Real.sqrt x ∧ 
   a - 2 * (a - y)^2 = Real.sqrt x ∧ 
   ∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) 
  → a = 2 := by
  sorry

/-- For any a > 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_above_two (a : ℝ) :
  a > 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- For any 0 ≤ a < 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_below_two (a : ℝ) :
  0 ≤ a ∧ a < 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- The maximum value of a for which the system has a unique solution is 2 -/
theorem max_a_for_unique_solution :
  ∃ (a : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) ∧
  ∀ (b : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ b - 2 * (b - y)^2 = Real.sqrt x) → b ≤ a := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l3742_374238


namespace NUMINAMATH_CALUDE_mysterious_division_l3742_374240

theorem mysterious_division :
  ∃! (d q : ℕ),
    d ∈ Finset.range 900 ∧ d ≥ 100 ∧
    q ∈ Finset.range 90000 ∧ q ≥ 10000 ∧
    10000000 = d * q + (10000000 % d) ∧
    d = 124 ∧ q = 80809 := by
  sorry

end NUMINAMATH_CALUDE_mysterious_division_l3742_374240


namespace NUMINAMATH_CALUDE_last_segment_speed_l3742_374286

/-- Proves that the average speed for the last segment is 67 mph given the conditions of the problem -/
theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) : ℝ :=
  by
  have h1 : total_distance = 96 := by sorry
  have h2 : total_time = 90 / 60 := by sorry
  have h3 : first_segment_speed = 60 := by sorry
  have h4 : second_segment_speed = 65 := by sorry
  
  let overall_average_speed := total_distance / total_time
  have h5 : overall_average_speed = 64 := by sorry
  
  let last_segment_speed := 3 * overall_average_speed - first_segment_speed - second_segment_speed
  
  exact last_segment_speed

end NUMINAMATH_CALUDE_last_segment_speed_l3742_374286


namespace NUMINAMATH_CALUDE_fundraising_problem_l3742_374209

/-- The fundraising problem -/
theorem fundraising_problem (total_goal : ℕ) (num_people : ℕ) (fee_per_person : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : num_people = 8)
  (h3 : fee_per_person = 20) :
  (total_goal + num_people * fee_per_person) / num_people = 320 := by
  sorry

#check fundraising_problem

end NUMINAMATH_CALUDE_fundraising_problem_l3742_374209


namespace NUMINAMATH_CALUDE_hemisphere_to_sphere_surface_area_l3742_374251

/-- Given a hemisphere with base area 81π, prove that the total surface area
    of the sphere obtained by adding a top circular lid is 324π. -/
theorem hemisphere_to_sphere_surface_area :
  ∀ r : ℝ,
  r > 0 →
  π * r^2 = 81 * π →
  4 * π * r^2 = 324 * π := by
sorry

end NUMINAMATH_CALUDE_hemisphere_to_sphere_surface_area_l3742_374251


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3742_374227

/-- Given a quadratic of the form x^2 + bx + 50 where b is positive,
    if it can be written as (x+n)^2 + 8, then b = 2√42. -/
theorem quadratic_coefficient (b n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 50 = (x+n)^2 + 8) → 
  b = 2 * Real.sqrt 42 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3742_374227


namespace NUMINAMATH_CALUDE_circle_passes_through_point_l3742_374253

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a circle
structure Circle where
  center : PointOnParabola
  radius : ℝ
  tangent_to_line : radius = center.x + 2

-- Theorem to prove
theorem circle_passes_through_point :
  ∀ (c : Circle), (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_point_l3742_374253


namespace NUMINAMATH_CALUDE_circle_radius_l3742_374211

theorem circle_radius (r : ℝ) (h : r > 0) :
  π * r^2 + 2 * π * r = 100 * π → r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l3742_374211


namespace NUMINAMATH_CALUDE_gcf_64_80_l3742_374210

theorem gcf_64_80 : Nat.gcd 64 80 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_64_80_l3742_374210


namespace NUMINAMATH_CALUDE_combined_shape_area_l3742_374288

/-- The area of a shape formed by attaching a square to a rectangle -/
theorem combined_shape_area (rectangle_length rectangle_width square_side : Real) :
  rectangle_length = 0.45 →
  rectangle_width = 0.25 →
  square_side = 0.15 →
  rectangle_length * rectangle_width + square_side * square_side = 0.135 := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_area_l3742_374288


namespace NUMINAMATH_CALUDE_no_real_solutions_implies_a_less_than_one_l3742_374200

theorem no_real_solutions_implies_a_less_than_one :
  (∀ x : ℝ, ¬∃ (y : ℝ), y^2 = x + 4 ∧ y = a - 1) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_implies_a_less_than_one_l3742_374200


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l3742_374293

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 5) (hb : b = 7) :
  (Real.sqrt (a^2 + b^2)) * (Real.sqrt (max a b)^2 - (min a b)^2) = Real.sqrt 1776 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l3742_374293


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3742_374260

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : a + b = 7 * (a - b)) (h5 : a^2 + b^2 = 85) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3742_374260


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3742_374203

theorem cubic_roots_sum_of_squares_reciprocals :
  ∀ a b c : ℝ,
  (a^3 - 8*a^2 + 15*a - 7 = 0) →
  (b^3 - 8*b^2 + 15*b - 7 = 0) →
  (c^3 - 8*c^2 + 15*c - 7 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 113/49) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3742_374203


namespace NUMINAMATH_CALUDE_anya_balloons_l3742_374294

theorem anya_balloons (total : ℕ) (colors : ℕ) (anya_fraction : ℚ) 
  (h1 : total = 672) 
  (h2 : colors = 4) 
  (h3 : anya_fraction = 1/2) : 
  (total / colors) * anya_fraction = 84 := by
  sorry

end NUMINAMATH_CALUDE_anya_balloons_l3742_374294


namespace NUMINAMATH_CALUDE_monster_family_kids_eyes_l3742_374232

/-- Represents a monster family with parents and kids -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  total_eyes : ℕ

/-- Calculates the number of eyes each kid has in a monster family -/
def eyes_per_kid (family : MonsterFamily) : ℕ :=
  (family.total_eyes - family.mom_eyes - family.dad_eyes) / family.num_kids

/-- Theorem: In the specific monster family, each kid has 4 eyes -/
theorem monster_family_kids_eyes :
  let family : MonsterFamily := {
    mom_eyes := 1,
    dad_eyes := 3,
    num_kids := 3,
    total_eyes := 16
  }
  eyes_per_kid family = 4 := by sorry

end NUMINAMATH_CALUDE_monster_family_kids_eyes_l3742_374232


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3742_374208

/-- Given an ellipse C and a line l, if l intersects C at two points with a specific distance, then the y-intercept of l is 0. -/
theorem ellipse_line_intersection (x y m : ℝ) : 
  (4 * x^2 + y^2 = 1) →  -- Ellipse equation
  (y = x + m) →          -- Line equation
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (4 * A.1^2 + A.2^2 = 1) ∧ 
    (4 * B.1^2 + B.2^2 = 1) ∧ 
    (A.2 = A.1 + m) ∧ 
    (B.2 = B.1 + m) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 10 / 5)^2)) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3742_374208


namespace NUMINAMATH_CALUDE_tan_half_angle_l3742_374226

theorem tan_half_angle (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_l3742_374226


namespace NUMINAMATH_CALUDE_inequality_solution_l3742_374214

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - 2*x^4) / (x + x^2 - 2*x^3) ≥ -1 ↔ 
  (x ≥ -1 ∧ x ≠ -1/2 ∧ x ≠ 0 ∧ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3742_374214


namespace NUMINAMATH_CALUDE_find_a_over_b_l3742_374298

theorem find_a_over_b (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 0.1875)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_over_b_l3742_374298


namespace NUMINAMATH_CALUDE_exists_collatz_greater_than_2012x_l3742_374278

-- Define the Collatz function
def collatz (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

-- Define the iterated Collatz function
def collatz_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => collatz_iter n (collatz x)

-- State the theorem
theorem exists_collatz_greater_than_2012x : ∃ x : ℕ, x > 0 ∧ collatz_iter 40 x > 2012 * x := by
  sorry

end NUMINAMATH_CALUDE_exists_collatz_greater_than_2012x_l3742_374278


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3742_374255

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 360) ∧ ((n + 1) * (n + 2) ≥ 360) → 
  n + (n + 1) = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3742_374255


namespace NUMINAMATH_CALUDE_tens_digit_of_1998_pow_2003_minus_1995_l3742_374299

theorem tens_digit_of_1998_pow_2003_minus_1995 :
  (1998^2003 - 1995) % 100 / 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_1998_pow_2003_minus_1995_l3742_374299


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3742_374285

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (4 - 5*y) = 8 → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3742_374285


namespace NUMINAMATH_CALUDE_shorter_tank_radius_l3742_374222

/-- Given two cylindrical tanks with equal volume, where one tank's height is four times the other
    and the taller tank has a radius of 10 units, the radius of the shorter tank is 20 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  π * (10 ^ 2) * (4 * h) = π * (20 ^ 2) * h := by sorry

end NUMINAMATH_CALUDE_shorter_tank_radius_l3742_374222


namespace NUMINAMATH_CALUDE_josh_remaining_money_l3742_374250

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℝ) (hat_cost : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (num_cookies : ℕ) : ℝ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * num_cookies)

/-- Proves that Josh has $3 left after his purchases -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l3742_374250


namespace NUMINAMATH_CALUDE_range_of_P_l3742_374263

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) : 
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_P_l3742_374263


namespace NUMINAMATH_CALUDE_maryann_client_call_time_l3742_374248

theorem maryann_client_call_time (total_time accounting_time client_time : ℕ) : 
  total_time = 560 →
  accounting_time = 7 * client_time →
  total_time = accounting_time + client_time →
  client_time = 70 := by
sorry

end NUMINAMATH_CALUDE_maryann_client_call_time_l3742_374248


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l3742_374268

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 3 ∣ m → 7 ∣ m → 11 ∣ m → n ≤ m) ∧
  n = 1155 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l3742_374268


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3742_374215

theorem interest_rate_calculation (total : ℝ) (part2 : ℝ) (years1 : ℝ) (years2 : ℝ) (rate2 : ℝ) :
  total = 2665 →
  part2 = 1332.5 →
  years1 = 5 →
  years2 = 3 →
  rate2 = 0.05 →
  let part1 := total - part2
  let r := (part2 * rate2 * years2) / (part1 * years1)
  r = 0.03 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3742_374215


namespace NUMINAMATH_CALUDE_g_is_even_l3742_374213

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := x^2 * f x

/-- Theorem stating that g is an even function -/
theorem g_is_even (f : ℝ → ℝ) (h : FunctionalEq f) :
  ∀ x : ℝ, g f (-x) = g f x :=
by sorry

end NUMINAMATH_CALUDE_g_is_even_l3742_374213


namespace NUMINAMATH_CALUDE_factorial_sum_power_two_l3742_374256

theorem factorial_sum_power_two (a b c : ℕ+) : 
  (Nat.factorial a.val + Nat.factorial b.val = 2^(Nat.factorial c.val)) ↔ 
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) := by
sorry

end NUMINAMATH_CALUDE_factorial_sum_power_two_l3742_374256


namespace NUMINAMATH_CALUDE_one_fourth_of_ten_times_twelve_divided_by_two_l3742_374297

theorem one_fourth_of_ten_times_twelve_divided_by_two : (1 / 4 : ℚ) * ((10 * 12) / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_ten_times_twelve_divided_by_two_l3742_374297


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l3742_374228

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that returns true if n is the least positive integer with exactly k factors -/
def is_least_with_factors (n k : ℕ+) : Prop :=
  (num_factors n = k) ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 72 12 := by sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l3742_374228


namespace NUMINAMATH_CALUDE_linda_sales_l3742_374277

/-- Calculates the total amount of money made from selling necklaces and rings -/
def total_money_made (num_necklaces : ℕ) (num_rings : ℕ) (cost_per_necklace : ℕ) (cost_per_ring : ℕ) : ℕ :=
  num_necklaces * cost_per_necklace + num_rings * cost_per_ring

/-- Theorem: The total money made from selling 4 necklaces at $12 each and 8 rings at $4 each is $80 -/
theorem linda_sales : total_money_made 4 8 12 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_linda_sales_l3742_374277


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l3742_374244

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 3 → b = 6 → c = 2 → d = 5 →
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l3742_374244


namespace NUMINAMATH_CALUDE_class_size_calculation_l3742_374235

theorem class_size_calculation (tables : Nat) (students_per_table : Nat)
  (bathroom_girls : Nat) (canteen_multiplier : Nat)
  (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat)
  (germany : Nat) (france : Nat) (norway : Nat) (italy : Nat) (spain : Nat) (australia : Nat) :
  tables = 6 →
  students_per_table = 3 →
  bathroom_girls = 5 →
  canteen_multiplier = 5 →
  group1 = 4 →
  group2 = 5 →
  group3 = 6 →
  group4 = 3 →
  germany = 3 →
  france = 4 →
  norway = 3 →
  italy = 2 →
  spain = 2 →
  australia = 1 →
  (tables * students_per_table + bathroom_girls + bathroom_girls * canteen_multiplier +
   group1 + group2 + group3 + group4 +
   germany + france + norway + italy + spain + australia) = 81 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3742_374235


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3742_374292

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 4 * Real.sqrt (2 + x) + 4 * Real.sqrt (2 - x) = 6 * Real.sqrt 3 ∧ x = (3 * Real.sqrt 15) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3742_374292


namespace NUMINAMATH_CALUDE_battery_current_l3742_374202

theorem battery_current (voltage : ℝ) (resistance : ℝ) (current : ℝ → ℝ) :
  voltage = 48 →
  (∀ R, current R = voltage / R) →
  resistance = 12 →
  current resistance = 4 :=
by sorry

end NUMINAMATH_CALUDE_battery_current_l3742_374202


namespace NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l3742_374230

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_lines_to_plane_are_parallel (m n : Line3D) (β : Plane3D) :
  perpendicular m β → perpendicular n β → parallel_lines m n :=
sorry

theorem perpendicular_line_to_planes_are_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l3742_374230


namespace NUMINAMATH_CALUDE_chess_draw_probability_l3742_374280

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l3742_374280


namespace NUMINAMATH_CALUDE_fourth_root_sqrt_five_squared_l3742_374243

theorem fourth_root_sqrt_five_squared : 
  ((5 ^ (1 / 2)) ^ 5) ^ (1 / 4) ^ 2 = 5 * (5 ^ (1 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sqrt_five_squared_l3742_374243


namespace NUMINAMATH_CALUDE_king_queen_prob_l3742_374223

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Queens in a standard deck -/
def numQueens : ℕ := 4

/-- Calculates the probability of drawing a King followed by a Queen from a standard deck -/
def probKingQueen (deck : StandardDeck) : ℚ :=
  (numKings * numQueens : ℚ) / (deckSize * (deckSize - 1))

/-- Theorem stating that the probability of drawing a King followed by a Queen is 4/663 -/
theorem king_queen_prob : 
  ∀ (deck : StandardDeck), probKingQueen deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_king_queen_prob_l3742_374223


namespace NUMINAMATH_CALUDE_water_volume_calculation_l3742_374245

/-- Represents a cylindrical tank with an internal obstruction --/
structure Tank where
  radius : ℝ
  height : ℝ
  obstruction_radius : ℝ

/-- Calculates the volume of water in the tank --/
def water_volume (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- The specific tank in the problem --/
def problem_tank : Tank :=
  { radius := 5
  , height := 12
  , obstruction_radius := 2 }

theorem water_volume_calculation :
  water_volume problem_tank 3 = 110 * Real.pi - 96 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l3742_374245


namespace NUMINAMATH_CALUDE_garden_dimensions_l3742_374242

/-- Represents a rectangular garden with given perimeter and length-to-breadth ratio -/
structure RectangularGarden where
  perimeter : ℝ
  length_breadth_ratio : ℝ
  length : ℝ
  breadth : ℝ
  diagonal : ℝ
  perimeter_eq : perimeter = 2 * (length + breadth)
  ratio_eq : length = length_breadth_ratio * breadth

/-- Theorem about the dimensions of a specific rectangular garden -/
theorem garden_dimensions (g : RectangularGarden) 
  (h_perimeter : g.perimeter = 500)
  (h_ratio : g.length_breadth_ratio = 3/2) :
  g.length = 150 ∧ g.diagonal = Real.sqrt 32500 := by
  sorry

#check garden_dimensions

end NUMINAMATH_CALUDE_garden_dimensions_l3742_374242


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l3742_374296

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / total_students.choose students_to_select

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l3742_374296


namespace NUMINAMATH_CALUDE_tea_leaf_problem_l3742_374224

theorem tea_leaf_problem (num_plants : ℕ) (remaining_fraction : ℚ) (total_remaining : ℕ) :
  num_plants = 3 →
  remaining_fraction = 2/3 →
  total_remaining = 36 →
  ∃ initial_per_plant : ℕ,
    initial_per_plant * num_plants * remaining_fraction = total_remaining ∧
    initial_per_plant = 18 :=
by sorry

end NUMINAMATH_CALUDE_tea_leaf_problem_l3742_374224


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3742_374229

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 20 > 0) →
  a = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3742_374229


namespace NUMINAMATH_CALUDE_birthday_gift_savings_l3742_374282

/-- Calculates the total amount saved for a mother's birthday gift based on orange sales --/
def total_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ)
  (jake_oranges : ℕ) (jake_bundles : ℕ) (jake_price1 : ℚ) (jake_price2 : ℚ) (jake_discount : ℚ) : ℚ :=
  let liam_earnings := (liam_oranges / 2 : ℚ) * liam_price
  let claire_earnings := (claire_oranges : ℚ) * claire_price
  let jake_earnings1 := (jake_bundles / 2 : ℚ) * jake_price1
  let jake_earnings2 := (jake_bundles / 2 : ℚ) * jake_price2
  let jake_total := jake_earnings1 + jake_earnings2
  let jake_discount_amount := jake_total * jake_discount
  let jake_earnings := jake_total - jake_discount_amount
  liam_earnings + claire_earnings + jake_earnings

/-- Theorem stating that the total savings for the mother's birthday gift is $117.88 --/
theorem birthday_gift_savings :
  total_savings 40 (5/2) 30 (6/5) 50 10 3 (9/2) (3/20) = 11788/100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_savings_l3742_374282


namespace NUMINAMATH_CALUDE_sequence_general_term_l3742_374274

/-- A sequence satisfying the given recurrence relation -/
def Sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n + 2 * a (n + 1) = 7 * 3^(n - 1)) ∧ a 1 = 1

/-- Theorem stating that the sequence has the general term a_n = 3^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (h : Sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3742_374274


namespace NUMINAMATH_CALUDE_bottle_production_rate_l3742_374258

/-- Given that 5 identical machines can produce 900 bottles in 4 minutes at a constant rate,
    prove that 6 such machines can produce 270 bottles per minute. -/
theorem bottle_production_rate (rate : ℕ → ℕ → ℕ) : 
  (rate 5 4 = 900) → (rate 6 1 = 270) :=
by
  sorry


end NUMINAMATH_CALUDE_bottle_production_rate_l3742_374258


namespace NUMINAMATH_CALUDE_cubic_factorization_l3742_374241

theorem cubic_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3742_374241


namespace NUMINAMATH_CALUDE_count_not_divides_g_eq_33_l3742_374257

/-- g(n) is the product of proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- Predicate to check if n does not divide g(n) -/
def not_divides_g (n : ℕ) : Prop := ¬(n ∣ g n)

/-- The number of integers n between 2 and 100 (inclusive) for which n does not divide g(n) -/
def count_not_divides_g : ℕ := sorry

theorem count_not_divides_g_eq_33 : count_not_divides_g = 33 := by sorry

end NUMINAMATH_CALUDE_count_not_divides_g_eq_33_l3742_374257


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3742_374295

/-- A convex polygon with area, perimeter, and inscribed circle radius -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ
  inscribed_radius : ℝ
  area_pos : 0 < area
  perimeter_pos : 0 < perimeter
  inscribed_radius_pos : 0 < inscribed_radius

/-- The theorem stating that for any convex polygon, the ratio of its area to its perimeter
    is less than or equal to the radius of its inscribed circle -/
theorem inscribed_circle_radius_bound (poly : ConvexPolygon) :
  poly.area / poly.perimeter ≤ poly.inscribed_radius :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3742_374295


namespace NUMINAMATH_CALUDE_otimes_four_two_l3742_374252

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l3742_374252


namespace NUMINAMATH_CALUDE_base_6_to_10_54123_l3742_374247

def base_6_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

theorem base_6_to_10_54123 :
  base_6_to_10 [3, 2, 1, 4, 5] = 7395 := by
  sorry

end NUMINAMATH_CALUDE_base_6_to_10_54123_l3742_374247


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l3742_374259

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_at_one (a : ℝ) (h : f_derivative a 1 = 3) :
  ∃ m b : ℝ, m = 3 ∧ b = -2 ∧ ∀ x : ℝ, (f a x - (f a 1)) = m * (x - 1) := by sorry

theorem max_value_on_interval :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 2 → f 0 x ≤ M := by sorry

theorem a_range_for_nonnegative (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l3742_374259


namespace NUMINAMATH_CALUDE_find_divisor_l3742_374267

theorem find_divisor (D : ℕ) (x : ℕ) : 
  (∃ (x : ℕ), x ≤ 11 ∧ (2000 - x) % D = 0) → 
  (2000 - x = 1989) →
  D = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3742_374267


namespace NUMINAMATH_CALUDE_children_got_on_bus_l3742_374225

/-- Proves that the number of children who got on the bus at the bus stop is 14 -/
theorem children_got_on_bus (initial_children : ℕ) (final_children : ℕ) 
  (h1 : initial_children = 64) (h2 : final_children = 78) : 
  final_children - initial_children = 14 := by
  sorry

end NUMINAMATH_CALUDE_children_got_on_bus_l3742_374225


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3742_374201

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3742_374201


namespace NUMINAMATH_CALUDE_quadratic_solution_l3742_374290

theorem quadratic_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 15 = 0) → b = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3742_374290


namespace NUMINAMATH_CALUDE_club_group_size_theorem_l3742_374271

theorem club_group_size_theorem (N : ℕ) (x : ℕ) 
  (h1 : 20 < N ∧ N < 50) 
  (h2 : (N - 5) % 6 = 0 ∧ (N - 5) % 7 = 0) 
  (h3 : N % x = 7) : 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_club_group_size_theorem_l3742_374271


namespace NUMINAMATH_CALUDE_mark_initial_punch_l3742_374270

/-- The amount of punch in gallons that Mark initially added to the bowl -/
def initial_punch : ℝ := 4

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds after his cousin drinks -/
def second_addition : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to fill the bowl completely -/
def final_addition : ℝ := 12

theorem mark_initial_punch :
  initial_punch / 2 + second_addition - sally_drinks + final_addition = bowl_capacity :=
by sorry

end NUMINAMATH_CALUDE_mark_initial_punch_l3742_374270


namespace NUMINAMATH_CALUDE_scientific_notation_of_value_l3742_374249

-- Define the nanometer to meter conversion
def nm_to_m : ℝ := 1e-9

-- Define the value in meters
def value_in_meters : ℝ := 7 * nm_to_m

-- Theorem statement
theorem scientific_notation_of_value :
  ∃ (a : ℝ) (n : ℤ), value_in_meters = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_value_l3742_374249


namespace NUMINAMATH_CALUDE_even_quadratic_sum_l3742_374254

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

/-- The main theorem -/
theorem even_quadratic_sum (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  let interval := Set.Icc (-2 * a - 5) 1
  IsEvenOn f (-2 * a - 5) 1 → a + 2 * b = -2 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_sum_l3742_374254


namespace NUMINAMATH_CALUDE_smallest_number_square_and_cube_l3742_374233

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 3

theorem smallest_number_square_and_cube :
  ∃ n : ℕ, n = 72 ∧
    is_perfect_square (n * 2) ∧
    is_perfect_cube (n * 3) ∧
    ∀ m : ℕ, m < n →
      ¬(is_perfect_square (m * 2) ∧ is_perfect_cube (m * 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_square_and_cube_l3742_374233


namespace NUMINAMATH_CALUDE_building_volume_l3742_374283

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular room -/
def volume (d : RoomDimensions) : ℝ := d.length * d.breadth * d.height

/-- Calculates the surface area of the walls of a rectangular room -/
def wallArea (d : RoomDimensions) : ℝ := 2 * (d.length * d.height + d.breadth * d.height)

/-- Calculates the floor area of a rectangular room -/
def floorArea (d : RoomDimensions) : ℝ := d.length * d.breadth

theorem building_volume (firstFloor secondFloor : RoomDimensions)
  (h1 : firstFloor.length = 15)
  (h2 : firstFloor.breadth = 12)
  (h3 : secondFloor.length = 20)
  (h4 : secondFloor.breadth = 10)
  (h5 : secondFloor.height = firstFloor.height)
  (h6 : 2 * floorArea firstFloor = wallArea firstFloor) :
  volume firstFloor + volume secondFloor = 2534.6 := by
  sorry

end NUMINAMATH_CALUDE_building_volume_l3742_374283


namespace NUMINAMATH_CALUDE_arrangement_count_l3742_374236

/-- Represents the number of boys -/
def num_boys : Nat := 2

/-- Represents the number of girls -/
def num_girls : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_boys + num_girls

/-- Represents that the girls are adjacent -/
def girls_adjacent : Prop := True

/-- Represents that boy A is to the left of boy B -/
def boy_A_left_of_B : Prop := True

/-- The number of different arrangements -/
def num_arrangements : Nat := 18

theorem arrangement_count :
  girls_adjacent →
  boy_A_left_of_B →
  num_arrangements = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3742_374236


namespace NUMINAMATH_CALUDE_percent_equality_l3742_374276

theorem percent_equality (x : ℝ) (h : (0.3 * (0.2 * x)) = 24) :
  (0.2 * (0.3 * x)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3742_374276


namespace NUMINAMATH_CALUDE_shopkeeper_sold_450_meters_l3742_374221

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ  -- Total selling price in Rupees
  lossPerMeter : ℕ       -- Loss per meter in Rupees
  costPricePerMeter : ℕ  -- Cost price per meter in Rupees

/-- Calculates the number of meters of cloth sold -/
def metersOfClothSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMeter - sale.lossPerMeter)

/-- Theorem stating that the shopkeeper sold 450 meters of cloth -/
theorem shopkeeper_sold_450_meters :
  let sale : ClothSale := {
    totalSellingPrice := 18000,
    lossPerMeter := 5,
    costPricePerMeter := 45
  }
  metersOfClothSold sale = 450 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_sold_450_meters_l3742_374221


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3742_374262

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The fifth term of a geometric sequence. -/
def FifthTerm (a : ℕ → ℝ) : ℝ := a 5

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_pos : ∀ n, a n > 0)
    (h_geom : IsGeometricSequence a)
    (h_prod : a 1 * a 3 = 16)
    (h_sum : a 3 + a 4 = 24) :
  FifthTerm a = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3742_374262


namespace NUMINAMATH_CALUDE_daughters_work_time_l3742_374264

/-- Given a man can do a piece of work in 4 days, and together with his daughter
    they can do it in 3 days, prove that the daughter can do the work alone in 12 days. -/
theorem daughters_work_time (man_time : ℕ) (combined_time : ℕ) (daughter_time : ℕ) :
  man_time = 4 →
  combined_time = 3 →
  (1 : ℚ) / man_time + (1 : ℚ) / daughter_time = (1 : ℚ) / combined_time →
  daughter_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_daughters_work_time_l3742_374264


namespace NUMINAMATH_CALUDE_count_multiples_of_30_l3742_374234

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_fourth_power_multiple_of_30 : ℕ := 810000

theorem count_multiples_of_30 : 
  (smallest_fourth_power_multiple_of_30 / 30) - (smallest_square_multiple_of_30 / 30) + 1 = 26971 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_30_l3742_374234


namespace NUMINAMATH_CALUDE_john_zoo_snakes_l3742_374219

/-- The number of snakes John has in his zoo --/
def num_snakes : ℕ := 15

/-- The total number of animals in John's zoo --/
def total_animals : ℕ := 114

/-- Theorem stating that the number of snakes in John's zoo is correct --/
theorem john_zoo_snakes :
  (num_snakes : ℚ) +
  (2 * num_snakes : ℚ) +
  ((2 * num_snakes : ℚ) - 5) +
  ((2 * num_snakes : ℚ) - 5 + 8) +
  (1/3 * ((2 * num_snakes : ℚ) - 5 + 8)) = total_animals := by
  sorry

#check john_zoo_snakes

end NUMINAMATH_CALUDE_john_zoo_snakes_l3742_374219


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3742_374205

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let initial_blue := (4/7) * total
  let initial_red := total - initial_blue
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3742_374205


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l3742_374220

theorem consecutive_integers_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 36 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l3742_374220


namespace NUMINAMATH_CALUDE_second_planner_cheaper_l3742_374287

/-- Represents the cost function for an event planner -/
structure EventPlanner where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for an event planner given the number of people -/
def totalCost (planner : EventPlanner) (people : ℕ) : ℕ :=
  planner.basicFee + planner.perPersonFee * people

/-- The first event planner's pricing structure -/
def planner1 : EventPlanner := ⟨120, 18⟩

/-- The second event planner's pricing structure -/
def planner2 : EventPlanner := ⟨250, 15⟩

/-- Theorem stating the conditions for when the second planner becomes less expensive -/
theorem second_planner_cheaper (n : ℕ) :
  (n < 44 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) :=
sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_l3742_374287


namespace NUMINAMATH_CALUDE_cos_is_even_l3742_374289

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cos_is_even_l3742_374289


namespace NUMINAMATH_CALUDE_function_value_at_seven_l3742_374237

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5 where a, b, c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem function_value_at_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_seven_l3742_374237


namespace NUMINAMATH_CALUDE_opposite_abs_difference_l3742_374231

theorem opposite_abs_difference (a : ℤ) : a = -3 → |a - 2| = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_abs_difference_l3742_374231


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3742_374261

theorem ice_cream_cost (two_cones_cost : ℕ) (h : two_cones_cost = 198) : 
  two_cones_cost / 2 = 99 := by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3742_374261


namespace NUMINAMATH_CALUDE_agricultural_profit_optimization_l3742_374246

/-- Represents the profit optimization problem for an agricultural product company -/
theorem agricultural_profit_optimization
  (retail_profit : ℝ) -- Profit from retailing one box
  (wholesale_profit : ℝ) -- Profit from wholesaling one box
  (total_boxes : ℕ) -- Total number of boxes to be sold
  (retail_limit : ℝ) -- Maximum percentage of boxes that can be sold through retail
  (h1 : retail_profit = 70)
  (h2 : wholesale_profit = 40)
  (h3 : total_boxes = 1000)
  (h4 : retail_limit = 0.3) :
  ∃ (retail_boxes wholesale_boxes : ℕ) (max_profit : ℝ),
    retail_boxes + wholesale_boxes = total_boxes ∧
    retail_boxes ≤ (retail_limit * total_boxes) ∧
    max_profit = retail_profit * retail_boxes + wholesale_profit * wholesale_boxes ∧
    retail_boxes = 300 ∧
    wholesale_boxes = 700 ∧
    max_profit = 49000 ∧
    ∀ (r w : ℕ),
      r + w = total_boxes →
      r ≤ (retail_limit * total_boxes) →
      retail_profit * r + wholesale_profit * w ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_agricultural_profit_optimization_l3742_374246


namespace NUMINAMATH_CALUDE_tournament_result_l3742_374281

-- Define the type for teams
inductive Team : Type
| A | B | C | D

-- Define the type for match results
inductive MatchResult : Type
| Win | Loss

-- Define a function to represent the number of wins for each team
def wins : Team → Nat
| Team.A => 2
| Team.B => 0
| Team.C => 1
| Team.D => 3

-- Define a function to represent the number of losses for each team
def losses : Team → Nat
| Team.A => 1
| Team.B => 3
| Team.C => 2
| Team.D => 0

-- Theorem statement
theorem tournament_result :
  (∀ t : Team, wins t + losses t = 3) ∧
  (wins Team.A + wins Team.B + wins Team.C + wins Team.D = 6) ∧
  (losses Team.A + losses Team.B + losses Team.C + losses Team.D = 6) :=
by sorry

end NUMINAMATH_CALUDE_tournament_result_l3742_374281


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_max_sum_achieved_l3742_374212

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7/3 := by
sorry

theorem max_sum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ (x y : ℝ), 4 * x + 3 * y ≤ 9 ∧ 2 * x + 4 * y ≤ 8 ∧ x + y > 7/3 - ε := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_max_sum_achieved_l3742_374212
