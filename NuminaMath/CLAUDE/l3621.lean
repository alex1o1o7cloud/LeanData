import Mathlib

namespace NUMINAMATH_CALUDE_initial_apples_count_l3621_362122

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := sorry

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 6

/-- Theorem stating that the initial number of apples is 11 -/
theorem initial_apples_count : initial_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l3621_362122


namespace NUMINAMATH_CALUDE_num_cases_hearts_D_l3621_362139

/-- The number of cards in a standard deck without jokers -/
def totalCards : ℕ := 52

/-- The number of people among whom the cards are distributed -/
def numPeople : ℕ := 4

/-- The total number of hearts in the deck -/
def totalHearts : ℕ := 13

/-- The number of hearts A has -/
def heartsA : ℕ := 5

/-- The number of hearts B has -/
def heartsB : ℕ := 4

/-- Theorem stating the number of possible cases for D's hearts -/
theorem num_cases_hearts_D : 
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ k : ℕ, k ≤ totalHearts - heartsA - heartsB → 
    (∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k)) ∧
  (∀ k : ℕ, k > totalHearts - heartsA - heartsB → 
    ¬∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k) :=
by sorry

end NUMINAMATH_CALUDE_num_cases_hearts_D_l3621_362139


namespace NUMINAMATH_CALUDE_min_distance_on_parabola_l3621_362195

/-- The minimum distance between two points on y = 2x² where the line
    connecting them is perpendicular to the tangent at one point -/
theorem min_distance_on_parabola :
  let f (x : ℝ) := 2 * x^2
  let tangent_slope (a : ℝ) := 4 * a
  let perpendicular_slope (a : ℝ) := -1 / (tangent_slope a)
  let distance (a : ℝ) := 
    let t := 4 * a^2
    Real.sqrt ((1 / (64 * t^2)) + (1 / (2 * t)) + t + 9/4)
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), distance a ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_on_parabola_l3621_362195


namespace NUMINAMATH_CALUDE_jennys_coins_value_l3621_362109

/-- Represents the value of Jenny's coins in cents -/
def coin_value (n : ℕ) : ℚ :=
  300 - 5 * n

/-- Represents the value of Jenny's coins in cents if nickels and dimes were swapped -/
def swapped_value (n : ℕ) : ℚ :=
  150 + 5 * n

/-- The number of nickels Jenny has -/
def number_of_nickels : ℕ :=
  27

theorem jennys_coins_value :
  coin_value number_of_nickels = 165 ∧
  swapped_value number_of_nickels = coin_value number_of_nickels + 120 :=
sorry

end NUMINAMATH_CALUDE_jennys_coins_value_l3621_362109


namespace NUMINAMATH_CALUDE_translation_right_2_units_l3621_362146

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the right by a given distance -/
def translateRight (p : Point) (d : ℝ) : Point :=
  { x := p.x + d, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 1, y := 2 }
  let A' : Point := translateRight A 2
  A'.x = 3 ∧ A'.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_2_units_l3621_362146


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3621_362152

/-- Theorem: For a convex polygon where the sum of the interior angles is twice the sum of its exterior angles, the number of sides is 6. -/
theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3621_362152


namespace NUMINAMATH_CALUDE_inequality_proof_l3621_362121

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (1/x) + (4/y) + (9/z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3621_362121


namespace NUMINAMATH_CALUDE_shifts_needed_is_six_l3621_362107

/-- Represents the problem of assigning workers to shifts -/
structure ShiftAssignment where
  total_workers : ℕ
  workers_per_shift : ℕ
  total_assignments : ℕ

/-- Calculates the number of shifts needed -/
def number_of_shifts (assignment : ShiftAssignment) : ℕ :=
  assignment.total_workers / assignment.workers_per_shift

/-- Theorem stating that the number of shifts is 6 for the given conditions -/
theorem shifts_needed_is_six (assignment : ShiftAssignment) 
  (h1 : assignment.total_workers = 12)
  (h2 : assignment.workers_per_shift = 2)
  (h3 : assignment.total_assignments = 23760) :
  number_of_shifts assignment = 6 := by
  sorry

#eval number_of_shifts ⟨12, 2, 23760⟩

end NUMINAMATH_CALUDE_shifts_needed_is_six_l3621_362107


namespace NUMINAMATH_CALUDE_product_abcd_l3621_362142

theorem product_abcd (a b c d : ℚ) : 
  (2*a + 4*b + 6*c + 8*d = 48) →
  (4*(d+c) = b) →
  (4*b + 2*c = a) →
  (c + 1 = d) →
  (a * b * c * d = -319603200 / 10503489) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l3621_362142


namespace NUMINAMATH_CALUDE_triangle_cos_C_l3621_362161

/-- Given a triangle ABC where b = 2a and b sin A = c sin C, prove that cos C = 3/4 -/
theorem triangle_cos_C (a b c : ℝ) (A B C : ℝ) : 
  b = 2 * a → b * Real.sin A = c * Real.sin C → Real.cos C = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_C_l3621_362161


namespace NUMINAMATH_CALUDE_diagonal_smallest_angle_at_midpoints_l3621_362131

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length a -/
structure Cube where
  a : ℝ
  center : Point3D

/-- Calculates the angle at which the diagonal is seen from a point on the cube's surface -/
noncomputable def angleFromPoint (c : Cube) (p : Point3D) : ℝ := sorry

/-- Checks if a point is on the surface of the cube -/
def isOnSurface (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the midpoints of the cube's faces -/
def faceMidpoints (c : Cube) : List Point3D := sorry

/-- Main theorem: The diagonal is seen at the smallest angle from the midpoints of the cube's faces -/
theorem diagonal_smallest_angle_at_midpoints (c : Cube) :
  ∀ p : Point3D, isOnSurface c p →
    (p ∉ faceMidpoints c → 
      ∀ m ∈ faceMidpoints c, angleFromPoint c p > angleFromPoint c m) :=
sorry

end NUMINAMATH_CALUDE_diagonal_smallest_angle_at_midpoints_l3621_362131


namespace NUMINAMATH_CALUDE_smallest_population_with_conditions_l3621_362181

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_population_with_conditions : 
  ∃ n : ℕ, 
    is_perfect_square n ∧ 
    (∃ k : ℕ, n + 150 = k^2 + 1) ∧ 
    is_perfect_square (n + 300) ∧
    n = 144 ∧
    ∀ m : ℕ, m < n → 
      ¬(is_perfect_square m ∧ 
        (∃ k : ℕ, m + 150 = k^2 + 1) ∧ 
        is_perfect_square (m + 300)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_population_with_conditions_l3621_362181


namespace NUMINAMATH_CALUDE_expansion_equality_l3621_362173

theorem expansion_equality (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l3621_362173


namespace NUMINAMATH_CALUDE_vector_midpoint_dot_product_l3621_362129

def problem (a b : ℝ × ℝ) : Prop :=
  let m : ℝ × ℝ := (4, 10)
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) ∧
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 444

theorem vector_midpoint_dot_product :
  ∀ a b : ℝ × ℝ, problem a b :=
by
  sorry

end NUMINAMATH_CALUDE_vector_midpoint_dot_product_l3621_362129


namespace NUMINAMATH_CALUDE_no_super_squarish_numbers_l3621_362159

-- Define a super-squarish number
def is_super_squarish (n : ℕ) : Prop :=
  -- Seven-digit number
  1000000 ≤ n ∧ n < 10000000 ∧
  -- No digit is zero
  ∀ d, (n / 10^d) % 10 ≠ 0 ∧
  -- Perfect square
  ∃ y, n = y^2 ∧
  -- First two digits are a perfect square
  ∃ a, (n / 100000)^2 = a ∧
  -- Next three digits are a perfect square
  ∃ b, ((n / 1000) % 1000)^2 = b ∧
  -- Last two digits are a perfect square
  ∃ c, (n % 100)^2 = c

-- Theorem statement
theorem no_super_squarish_numbers : ¬∃ n : ℕ, is_super_squarish n := by
  sorry

end NUMINAMATH_CALUDE_no_super_squarish_numbers_l3621_362159


namespace NUMINAMATH_CALUDE_bicentric_shapes_l3621_362197

-- Define the property of being bicentric
def IsBicentric (shape : Type) : Prop :=
  ∃ (circumscribed inscribed : Type), 
    (∀ (s : shape), ∃ (c : circumscribed), True) ∧ 
    (∀ (s : shape), ∃ (i : inscribed), True)

-- Define the shapes
def Square : Type := Unit
def Rectangle : Type := Unit
def RegularPentagon : Type := Unit
def Hexagon : Type := Unit

-- State the theorem
theorem bicentric_shapes :
  IsBicentric Square ∧
  IsBicentric RegularPentagon ∧
  ¬(∀ (r : Rectangle), IsBicentric Rectangle) ∧
  ¬(∀ (h : Hexagon), IsBicentric Hexagon) :=
sorry

end NUMINAMATH_CALUDE_bicentric_shapes_l3621_362197


namespace NUMINAMATH_CALUDE_square_side_length_l3621_362191

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 18) : ∃ square_side : ℝ,
  square_side = 12 ∧ 4 * square_side = 2 * (rectangle_width + rectangle_length) := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3621_362191


namespace NUMINAMATH_CALUDE_f_of_5_equals_105_l3621_362192

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x

-- State the theorem
theorem f_of_5_equals_105 : f 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_105_l3621_362192


namespace NUMINAMATH_CALUDE_product_in_M_l3621_362132

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 - b^2}

/-- Theorem: The product of any two elements in M is also in M -/
theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M := by
  sorry

end NUMINAMATH_CALUDE_product_in_M_l3621_362132


namespace NUMINAMATH_CALUDE_total_students_l3621_362133

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 120) : 
  boys + girls = 312 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3621_362133


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l3621_362189

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (3/2, 5/2, 6). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 4, 2)
  let C : ℝ × ℝ × ℝ := (3, 5, 6)
  orthocenter A B C = (3/2, 5/2, 6) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l3621_362189


namespace NUMINAMATH_CALUDE_detergent_calculation_l3621_362126

/-- Calculates the total amount of detergent used for washing clothes -/
theorem detergent_calculation (total_clothes cotton_clothes woolen_clothes : ℝ)
  (cotton_detergent wool_detergent : ℝ) : 
  total_clothes = cotton_clothes + woolen_clothes →
  cotton_clothes = 4 →
  woolen_clothes = 5 →
  cotton_detergent = 2 →
  wool_detergent = 1.5 →
  cotton_clothes * cotton_detergent + woolen_clothes * wool_detergent = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l3621_362126


namespace NUMINAMATH_CALUDE_painting_price_theorem_l3621_362178

theorem painting_price_theorem (total_cost : ℕ) (price : ℕ) (quantity : ℕ) :
  total_cost = 104 →
  price > 0 →
  quantity * price = total_cost →
  10 < quantity →
  quantity < 60 →
  (price = 2 ∨ price = 4 ∨ price = 8) :=
by sorry

end NUMINAMATH_CALUDE_painting_price_theorem_l3621_362178


namespace NUMINAMATH_CALUDE_abs_func_no_opposite_signs_l3621_362125

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem abs_func_no_opposite_signs :
  ∀ (a b : ℝ), (abs_func a) * (abs_func b) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_func_no_opposite_signs_l3621_362125


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_min_sum_achieved_l3621_362185

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := by
  sorry

theorem min_sum_achieved (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  ∃ (a b c : ℝ), a ≥ 4 ∧ b ≥ 5 ∧ c ≥ 6 ∧ a^2 + b^2 + c^2 ≥ 90 ∧ a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_min_sum_achieved_l3621_362185


namespace NUMINAMATH_CALUDE_vector_problem_l3621_362143

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

/-- Theorem statement -/
theorem vector_problem (p q : ℝ × ℝ) 
  (h1 : p = (1, 2)) 
  (h2 : vector_op p q = (-3, -4)) : 
  q = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3621_362143


namespace NUMINAMATH_CALUDE_largest_x_value_l3621_362164

theorem largest_x_value : ∃ x : ℝ,
  (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧
  x = (19 + Real.sqrt 229) / 22 ∧
  ∀ y : ℝ, (15 * y^2 - 30 * y + 9) / (4 * y - 3) + 6 * y = 7 * y - 2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3621_362164


namespace NUMINAMATH_CALUDE_pen_purchase_problem_l3621_362176

theorem pen_purchase_problem :
  ∀ (x y : ℕ),
    1.7 * (x : ℝ) + 1.2 * (y : ℝ) = 15 →
    x = 6 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_pen_purchase_problem_l3621_362176


namespace NUMINAMATH_CALUDE_sequence_integer_count_l3621_362149

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ k : ℕ, ∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) ∧
  (∀ k : ℕ, (∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) → k = 5) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l3621_362149


namespace NUMINAMATH_CALUDE_mileage_difference_l3621_362165

/-- Calculates the difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (total_miles : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  total_miles = 372 →
  advertised_mpg - (total_miles / tank_capacity) = 4 := by
  sorry


end NUMINAMATH_CALUDE_mileage_difference_l3621_362165


namespace NUMINAMATH_CALUDE_total_tiles_l3621_362114

theorem total_tiles (yellow blue purple white : ℕ) : 
  yellow = 3 → 
  blue = yellow + 1 → 
  purple = 6 → 
  white = 7 → 
  yellow + blue + purple + white = 20 := by
sorry

end NUMINAMATH_CALUDE_total_tiles_l3621_362114


namespace NUMINAMATH_CALUDE_picture_book_shelves_l3621_362112

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) : 
  books_per_shelf = 7 → 
  mystery_shelves = 8 → 
  total_books = 70 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l3621_362112


namespace NUMINAMATH_CALUDE_sum_of_common_roots_l3621_362135

theorem sum_of_common_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + (2*a - 5)*x + a^2 + 1 = 0 ∧ 
            x^3 + (2*a - 5)*x^2 + (a^2 + 1)*x + a^2 - 4 = 0) →
  (∃ x y : ℝ, x^2 - 9*x + 5 = 0 ∧ y^2 - 9*y + 5 = 0 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_roots_l3621_362135


namespace NUMINAMATH_CALUDE_min_value_implies_a_f_less_than_x_squared_l3621_362117

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
sorry

theorem f_less_than_x_squared (a : ℝ) :
  (∀ x > 1, f a x < x^2) →
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_f_less_than_x_squared_l3621_362117


namespace NUMINAMATH_CALUDE_bcm_hens_count_l3621_362151

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  ↑(total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
  sorry

end NUMINAMATH_CALUDE_bcm_hens_count_l3621_362151


namespace NUMINAMATH_CALUDE_bobbys_shoe_cost_l3621_362174

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost hourly_rate hours_worked discount_percentage : ℚ) : ℚ :=
  mold_cost + (hourly_rate * hours_worked) * (1 - discount_percentage)

/-- Theorem stating that Bobby's total cost for handmade shoes is $730 -/
theorem bobbys_shoe_cost :
  total_cost 250 75 8 0.2 = 730 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoe_cost_l3621_362174


namespace NUMINAMATH_CALUDE_new_average_age_l3621_362124

theorem new_average_age (n : ℕ) (original_avg : ℚ) (new_person_age : ℕ) : 
  n = 8 → original_avg = 14 → new_person_age = 32 → 
  (n * original_avg + new_person_age : ℚ) / (n + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l3621_362124


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3621_362118

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- Ensuring the polygon has at least 3 sides
  (∀ angle : ℝ, angle = 150 → 180 * (n - 2) = n * angle) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3621_362118


namespace NUMINAMATH_CALUDE_tangent_ratio_l3621_362155

/-- A cube with an inscribed sphere -/
structure CubeWithSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  /- The sphere radius is half the edge length -/
  sphere_radius_eq : sphere_radius = edge_length / 2

/-- A point on the edge of a cube -/
structure EdgePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on an edge -/
  on_edge : (x = 0 ∧ y = 0) ∨ (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0)

/-- A point on the inscribed sphere -/
structure SpherePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on the sphere -/
  on_sphere : x^2 + y^2 + z^2 = c.sphere_radius^2

/-- Theorem: The ratio KE:EF is 4:5 -/
theorem tangent_ratio 
  (c : CubeWithSphere) 
  (K : EdgePoint c) 
  (E : SpherePoint c) 
  (F : EdgePoint c) 
  (h_K_midpoint : K.x = c.edge_length / 2 ∨ K.y = c.edge_length / 2 ∨ K.z = c.edge_length / 2)
  (h_tangent : ∃ t : ℝ, K.x + t * (E.x - K.x) = F.x ∧ 
                        K.y + t * (E.y - K.y) = F.y ∧ 
                        K.z + t * (E.z - K.z) = F.z)
  (h_skew : (F.x ≠ K.x ∨ F.y ≠ K.y) ∧ (F.y ≠ K.y ∨ F.z ≠ K.z) ∧ (F.x ≠ K.x ∨ F.z ≠ K.z)) :
  ∃ (ke ef : ℝ), ke / ef = 4 / 5 ∧ 
    ke^2 = (E.x - K.x)^2 + (E.y - K.y)^2 + (E.z - K.z)^2 ∧
    ef^2 = (F.x - E.x)^2 + (F.y - E.y)^2 + (F.z - E.z)^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ratio_l3621_362155


namespace NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l3621_362110

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- The other endpoint of a diameter given the circle and one endpoint --/
def otherDiameterEndpoint (c : Circle) (p : Point) : Point :=
  (2 * c.center.1 - p.1, 2 * c.center.2 - p.2)

theorem diameter_endpoint_coordinates :
  let c : Circle := { center := (3, 5) }
  let p : Point := (0, 1)
  otherDiameterEndpoint c p = (6, 9) := by
  sorry

#check diameter_endpoint_coordinates

end NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l3621_362110


namespace NUMINAMATH_CALUDE_shower_tiles_count_l3621_362144

/-- Represents a shower with three walls -/
structure Shower :=
  (width : Nat)  -- Number of tiles in width
  (height : Nat) -- Number of tiles in height

/-- Calculates the total number of tiles in a shower -/
def totalTiles (s : Shower) : Nat :=
  3 * s.width * s.height

/-- Theorem stating that a shower with 8 tiles in width and 20 in height has 480 tiles in total -/
theorem shower_tiles_count : 
  ∀ s : Shower, s.width = 8 → s.height = 20 → totalTiles s = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l3621_362144


namespace NUMINAMATH_CALUDE_marbles_from_henry_l3621_362157

theorem marbles_from_henry (initial_marbles end_marbles marbles_from_henry : ℕ) 
  (h1 : initial_marbles = 95)
  (h2 : end_marbles = 104)
  (h3 : end_marbles = initial_marbles + marbles_from_henry) :
  marbles_from_henry = 9 := by
  sorry

end NUMINAMATH_CALUDE_marbles_from_henry_l3621_362157


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3621_362140

open Set

noncomputable def A : Set ℝ := {x | x ≥ -1}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_complement_equality : A ∩ (univ \ B) = Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3621_362140


namespace NUMINAMATH_CALUDE_gcd_problem_l3621_362137

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (12 * x) (18 * y) = 60) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (12 * c) (18 * d) ≥ 60) :=
sorry

end NUMINAMATH_CALUDE_gcd_problem_l3621_362137


namespace NUMINAMATH_CALUDE_league_games_count_l3621_362148

/-- The number of unique games played in a league season --/
def uniqueGamesInSeason (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 30 teams, where each team plays 15 games against every other team,
    the total number of unique games played in the season is 6,525. --/
theorem league_games_count :
  uniqueGamesInSeason 30 15 = 6525 := by
  sorry

#eval uniqueGamesInSeason 30 15

end NUMINAMATH_CALUDE_league_games_count_l3621_362148


namespace NUMINAMATH_CALUDE_grape_price_l3621_362134

/-- The price of each box of grapes given the following conditions:
  * 60 bundles of asparagus at $3.00 each
  * 40 boxes of grapes
  * 700 apples at $0.50 each
  * Total worth of the produce is $630
-/
theorem grape_price (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                    (grape_boxes : ℕ) (apple_count : ℕ) (apple_price : ℚ)
                    (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  apple_count = 700 →
  apple_price = 1/2 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + apple_count * apple_price)) / grape_boxes = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_grape_price_l3621_362134


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3621_362113

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 41) / 4
  let x₂ : ℝ := (3 - Real.sqrt 41) / 4
  2 * x₁^2 - 3 * x₁ - 4 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3621_362113


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3621_362127

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem matrix_equation_solution :
  ∃! (p q r : ℝ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -9 ∧ q = 0 ∧ r = 54 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3621_362127


namespace NUMINAMATH_CALUDE_expression_upper_bound_l3621_362190

theorem expression_upper_bound :
  ∃ (U : ℕ), 
    (∃ (S : Finset ℤ), 
      (Finset.card S = 50) ∧ 
      (∀ n ∈ S, 1 < 4*n + 7 ∧ 4*n + 7 < U) ∧
      (∀ U' < U, ∃ n ∈ S, 4*n + 7 ≥ U')) →
    U = 204 :=
by sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l3621_362190


namespace NUMINAMATH_CALUDE_paper_area_difference_l3621_362115

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem paper_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by sorry

end NUMINAMATH_CALUDE_paper_area_difference_l3621_362115


namespace NUMINAMATH_CALUDE_apple_slices_equality_l3621_362177

/-- Represents the number of slices in an apple -/
structure Apple :=
  (slices : ℕ)

/-- Represents the amount of apple eaten -/
def eaten (a : Apple) (s : ℕ) : ℚ :=
  s / a.slices

theorem apple_slices_equality (yeongchan minhyuk : Apple) 
  (h1 : yeongchan.slices = 3)
  (h2 : minhyuk.slices = 12) :
  eaten yeongchan 1 = eaten minhyuk 4 :=
by sorry

end NUMINAMATH_CALUDE_apple_slices_equality_l3621_362177


namespace NUMINAMATH_CALUDE_solution_equivalence_l3621_362156

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (Real.sqrt 22, Real.sqrt 22), (-Real.sqrt 22, -Real.sqrt 22),
   (Real.sqrt 20, -Real.sqrt 20), (-Real.sqrt 20, Real.sqrt 20),
   (((-3 + Real.sqrt 5) / 2) * (2 * Real.sqrt (3 + Real.sqrt 5)), 2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 + Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 + Real.sqrt 5)), -2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (2 * Real.sqrt (3 - Real.sqrt 5)), 2 * Real.sqrt (3 - Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 - Real.sqrt 5)), -2 * Real.sqrt (3 - Real.sqrt 5))}

theorem solution_equivalence :
  {(x, y) : ℝ × ℝ | x^5 = 21*x^3 + y^3 ∧ y^5 = x^3 + 21*y^3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l3621_362156


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3621_362172

/-- Given 60 feet of fencing for a rectangular pen where the length is exactly twice the width,
    the maximum possible area is 200 square feet. -/
theorem max_area_rectangular_pen (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) :
  perimeter = 60 →
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  area = length * width →
  area ≤ 200 ∧ ∃ w l, width = w ∧ length = l ∧ area = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3621_362172


namespace NUMINAMATH_CALUDE_parabola_directrix_l3621_362120

/-- Given a parabola with equation y = -1/4 * x^2, its directrix has the equation y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/4 * x^2) → (∃ (k : ℝ), k = 1 ∧ k = y + 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3621_362120


namespace NUMINAMATH_CALUDE_church_members_difference_church_members_proof_l3621_362108

theorem church_members_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_members adult_percentage children_difference =>
    total_members = 120 →
    adult_percentage = 40 →
    let adult_count := total_members * adult_percentage / 100
    let children_count := total_members - adult_count
    children_count - adult_count = children_difference

-- The proof of the theorem
theorem church_members_proof : church_members_difference 120 40 24 := by
  sorry

end NUMINAMATH_CALUDE_church_members_difference_church_members_proof_l3621_362108


namespace NUMINAMATH_CALUDE_six_students_like_no_option_l3621_362193

/-- Represents the food preferences in a class --/
structure FoodPreferences where
  total_students : ℕ
  french_fries : ℕ
  burgers : ℕ
  pizza : ℕ
  tacos : ℕ
  fries_burgers : ℕ
  fries_pizza : ℕ
  fries_tacos : ℕ
  burgers_pizza : ℕ
  burgers_tacos : ℕ
  pizza_tacos : ℕ
  fries_burgers_pizza : ℕ
  fries_burgers_tacos : ℕ
  fries_pizza_tacos : ℕ
  burgers_pizza_tacos : ℕ
  all_four : ℕ

/-- Calculates the number of students who don't like any food option --/
def studentsLikingNoOption (prefs : FoodPreferences) : ℕ :=
  prefs.total_students -
  (prefs.french_fries + prefs.burgers + prefs.pizza + prefs.tacos -
   prefs.fries_burgers - prefs.fries_pizza - prefs.fries_tacos -
   prefs.burgers_pizza - prefs.burgers_tacos - prefs.pizza_tacos +
   prefs.fries_burgers_pizza + prefs.fries_burgers_tacos +
   prefs.fries_pizza_tacos + prefs.burgers_pizza_tacos -
   prefs.all_four)

/-- Theorem: Given the food preferences, 6 students don't like any option --/
theorem six_students_like_no_option (prefs : FoodPreferences)
  (h1 : prefs.total_students = 35)
  (h2 : prefs.french_fries = 20)
  (h3 : prefs.burgers = 15)
  (h4 : prefs.pizza = 18)
  (h5 : prefs.tacos = 12)
  (h6 : prefs.fries_burgers = 10)
  (h7 : prefs.fries_pizza = 8)
  (h8 : prefs.fries_tacos = 6)
  (h9 : prefs.burgers_pizza = 7)
  (h10 : prefs.burgers_tacos = 5)
  (h11 : prefs.pizza_tacos = 9)
  (h12 : prefs.fries_burgers_pizza = 4)
  (h13 : prefs.fries_burgers_tacos = 3)
  (h14 : prefs.fries_pizza_tacos = 2)
  (h15 : prefs.burgers_pizza_tacos = 1)
  (h16 : prefs.all_four = 1) :
  studentsLikingNoOption prefs = 6 := by
  sorry


end NUMINAMATH_CALUDE_six_students_like_no_option_l3621_362193


namespace NUMINAMATH_CALUDE_tangent_line_through_origin_l3621_362160

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_through_origin (x₀ : ℝ) :
  (f' x₀ = 13 ∧ f x₀ = -f' x₀ * x₀) →
  (x₀ = -2 ∧ f x₀ = -26 ∧ ∀ x, f' x₀ * x = f' x₀ * x₀ + f x₀) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_through_origin_l3621_362160


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3621_362171

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 1110010110₂ -/
def binary_num : List Bool := [true, true, true, false, false, true, false, true, true, false]

/-- The expected quaternary representation of 32112₄ -/
def expected_quaternary : List (Fin 4) := [3, 2, 1, 1, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3621_362171


namespace NUMINAMATH_CALUDE_translation_result_l3621_362167

/-- Represents a point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point downward by a given number of units -/
def translateDown (p : Point) (units : ℝ) : Point :=
  { x := p.x, y := p.y - units }

/-- Translates a point to the right by a given number of units -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The main theorem stating the result of the translation -/
theorem translation_result : 
  let A : Point := { x := -2, y := 2 }
  let B : Point := translateRight (translateDown A 4) 3
  B.x = 1 ∧ B.y = -2 := by sorry

end NUMINAMATH_CALUDE_translation_result_l3621_362167


namespace NUMINAMATH_CALUDE_odd_number_between_bounds_l3621_362163

theorem odd_number_between_bounds (N : ℕ) : 
  N % 2 = 1 → (9.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10.5) → N = 39 ∨ N = 41 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_between_bounds_l3621_362163


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3621_362196

theorem scientific_notation_of_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3621_362196


namespace NUMINAMATH_CALUDE_orange_box_capacity_l3621_362130

/-- 
Given two boxes for carrying oranges, where:
- The first box has a capacity of 80 and is filled 3/4 full
- The second box has an unknown capacity C and is filled 3/5 full
- The total number of oranges in both boxes is 90

This theorem proves that the capacity C of the second box is 50.
-/
theorem orange_box_capacity 
  (box1_capacity : ℕ) 
  (box1_fill : ℚ) 
  (box2_fill : ℚ) 
  (total_oranges : ℕ) 
  (h1 : box1_capacity = 80)
  (h2 : box1_fill = 3/4)
  (h3 : box2_fill = 3/5)
  (h4 : total_oranges = 90) :
  ∃ (C : ℕ), box1_fill * box1_capacity + box2_fill * C = total_oranges ∧ C = 50 := by
sorry

end NUMINAMATH_CALUDE_orange_box_capacity_l3621_362130


namespace NUMINAMATH_CALUDE_katyas_classmates_l3621_362154

theorem katyas_classmates :
  ∀ (N : ℕ) (K : ℕ),
    (K + 10 - (N + 1)) / (N + 1) = K + 1 →
    N > 0 →
    N = 9 := by
  sorry

end NUMINAMATH_CALUDE_katyas_classmates_l3621_362154


namespace NUMINAMATH_CALUDE_stock_market_value_l3621_362186

/-- Calculates the market value of a stock given its income, interest rate, and brokerage fee. -/
def market_value (income : ℚ) (interest_rate : ℚ) (brokerage_rate : ℚ) : ℚ :=
  let face_value := (income * 100) / interest_rate
  let brokerage_fee := (face_value / 100) * brokerage_rate
  face_value - brokerage_fee

/-- Theorem stating that the market value of the stock is 7182 given the specified conditions. -/
theorem stock_market_value :
  market_value 756 10.5 0.25 = 7182 :=
by sorry

end NUMINAMATH_CALUDE_stock_market_value_l3621_362186


namespace NUMINAMATH_CALUDE_carol_initial_blocks_l3621_362100

/-- The number of blocks Carol started with -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Carol lost -/
def lost_blocks : ℕ := 25

/-- The number of blocks Carol ended with -/
def final_blocks : ℕ := 17

/-- Theorem stating that Carol started with 42 blocks -/
theorem carol_initial_blocks : initial_blocks = 42 := by sorry

end NUMINAMATH_CALUDE_carol_initial_blocks_l3621_362100


namespace NUMINAMATH_CALUDE_special_function_properties_l3621_362138

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y, and f(x) > 0 for x > 0 -/
class SpecialFunction (f : ℝ → ℝ) :=
  (add : ∀ x y : ℝ, f (x + y) = f x + f y)
  (pos : ∀ x : ℝ, x > 0 → f x > 0)

/-- The main theorem stating that a SpecialFunction is odd and monotonically increasing -/
theorem special_function_properties (f : ℝ → ℝ) [SpecialFunction f] :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l3621_362138


namespace NUMINAMATH_CALUDE_evaluate_expression_l3621_362116

theorem evaluate_expression (a : ℝ) :
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3621_362116


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3621_362169

theorem geometric_mean_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_gm : Real.sqrt (a * b) = 2) :
  5 ≤ (b + 1/a) + (a + 1/b) ∧ 
  (∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ Real.sqrt (a₀ * b₀) = 2 ∧ (b₀ + 1/a₀) + (a₀ + 1/b₀) = 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3621_362169


namespace NUMINAMATH_CALUDE_troll_count_l3621_362128

theorem troll_count (P B T : ℕ) : 
  P = 6 → 
  B = 4 * P - 6 → 
  T = B / 2 → 
  P + B + T = 33 := by
sorry

end NUMINAMATH_CALUDE_troll_count_l3621_362128


namespace NUMINAMATH_CALUDE_bisection_next_point_l3621_362111

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Define the initial interval
def a : ℝ := 0
def b : ℝ := 2

-- Define the first midpoint
def m₁ : ℝ := 1

-- Theorem statement
theorem bisection_next_point :
  f a < 0 ∧ f b > 0 ∧ f m₁ < 0 →
  (a + b) / 2 = 1.5 := by sorry

end NUMINAMATH_CALUDE_bisection_next_point_l3621_362111


namespace NUMINAMATH_CALUDE_intersection_point_l3621_362179

/-- The quadratic function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The point (0, -1) -/
def point : ℝ × ℝ := (0, -1)

/-- Theorem: The point (0, -1) is the intersection point of y = x^2 - 1 with the y-axis -/
theorem intersection_point :
  (point.1 = 0) ∧ 
  (point.2 = f point.1) ∧ 
  (∀ x : ℝ, x ≠ point.1 → (x, f x) ≠ point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3621_362179


namespace NUMINAMATH_CALUDE_unique_number_equality_l3621_362198

theorem unique_number_equality : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_equality_l3621_362198


namespace NUMINAMATH_CALUDE_stanley_run_distance_l3621_362101

def distance_walked : ℝ := 0.2
def additional_distance : ℝ := 0.2

theorem stanley_run_distance :
  distance_walked + additional_distance = 0.4 := by sorry

end NUMINAMATH_CALUDE_stanley_run_distance_l3621_362101


namespace NUMINAMATH_CALUDE_chinese_english_difference_l3621_362119

/-- The number of hours Ryan spends learning English daily -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours : ℕ := 7

/-- The difference in hours between Chinese and English learning time -/
def learning_difference : ℕ := chinese_hours - english_hours

theorem chinese_english_difference :
  learning_difference = 1 :=
by sorry

end NUMINAMATH_CALUDE_chinese_english_difference_l3621_362119


namespace NUMINAMATH_CALUDE_sin_value_from_tan_cos_l3621_362162

theorem sin_value_from_tan_cos (θ : Real) 
  (h1 : 6 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : π < θ) (h3 : θ < 2 * π) : 
  Real.sin θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_from_tan_cos_l3621_362162


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3621_362106

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + Real.sin x

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (4 * a) + f (b - 9) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (4 * x) + f (y - 9) = 0 → 1 / x + 1 / y ≥ 1) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f (4 * x) + f (y - 9) = 0 ∧ 1 / x + 1 / y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3621_362106


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3621_362180

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3621_362180


namespace NUMINAMATH_CALUDE_vector_OC_on_angle_bisector_l3621_362194

/-- Given points A and B, and a point C on the angle bisector of ∠AOB with |OC| = 2,
    prove that OC is equal to the specified vector. -/
theorem vector_OC_on_angle_bisector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  C.1^2 + C.2^2 = 4 →  -- |OC| = 2
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    C = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
    t * (A.1^2 + A.2^2) = (1 - t) * (B.1^2 + B.2^2)) →  -- C is on the angle bisector
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_OC_on_angle_bisector_l3621_362194


namespace NUMINAMATH_CALUDE_expression_equality_l3621_362145

theorem expression_equality (v u w : ℝ) 
  (h1 : u = 3 * v) 
  (h2 : w = 5 * u) : 
  2 * v + u + w = 20 * v := by sorry

end NUMINAMATH_CALUDE_expression_equality_l3621_362145


namespace NUMINAMATH_CALUDE_expression_evaluation_l3621_362136

theorem expression_evaluation : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3621_362136


namespace NUMINAMATH_CALUDE_pipe_stack_height_l3621_362123

theorem pipe_stack_height (d : ℝ) (h : ℝ) :
  d = 12 →
  h = 2 * d + d * Real.sqrt 3 →
  h = 24 + 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_stack_height_l3621_362123


namespace NUMINAMATH_CALUDE_susan_age_in_five_years_l3621_362175

/-- Represents the current year -/
def current_year : ℕ := 2023

/-- James' age in a given year -/
def james_age (year : ℕ) : ℕ := sorry

/-- Janet's age in a given year -/
def janet_age (year : ℕ) : ℕ := sorry

/-- Susan's age in a given year -/
def susan_age (year : ℕ) : ℕ := sorry

theorem susan_age_in_five_years :
  (∀ year : ℕ, james_age (year - 8) = 2 * janet_age (year - 8)) →
  james_age (current_year + 15) = 37 →
  (∀ year : ℕ, susan_age year = janet_age year - 3) →
  susan_age (current_year + 5) = 17 := by sorry

end NUMINAMATH_CALUDE_susan_age_in_five_years_l3621_362175


namespace NUMINAMATH_CALUDE_tempo_original_value_l3621_362184

/-- The original value of a tempo given its insured value and insurance extent --/
theorem tempo_original_value 
  (insured_value : ℝ) 
  (insurance_extent : ℝ) 
  (h1 : insured_value = 70000) 
  (h2 : insurance_extent = 4/5) : 
  ∃ (original_value : ℝ), 
    original_value = 87500 ∧ 
    insured_value = insurance_extent * original_value :=
by
  sorry

#check tempo_original_value

end NUMINAMATH_CALUDE_tempo_original_value_l3621_362184


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3621_362105

theorem parallel_vectors_sum_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • b) →
  ‖a + b‖ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l3621_362105


namespace NUMINAMATH_CALUDE_first_graders_count_l3621_362188

/-- The number of Kindergarteners to be checked -/
def kindergarteners : ℕ := 26

/-- The number of second graders to be checked -/
def second_graders : ℕ := 20

/-- The number of third graders to be checked -/
def third_graders : ℕ := 25

/-- The time in minutes it takes to check one student -/
def check_time : ℕ := 2

/-- The total time in hours available for all checks -/
def total_time_hours : ℕ := 3

/-- Calculate the number of first graders that need to be checked -/
def first_graders_to_check : ℕ :=
  (total_time_hours * 60 - (kindergarteners + second_graders + third_graders) * check_time) / check_time

/-- Theorem stating that the number of first graders to be checked is 19 -/
theorem first_graders_count : first_graders_to_check = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_graders_count_l3621_362188


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l3621_362170

theorem arithmetic_mean_sqrt2 (a b : ℝ) : 
  a = 1 / (Real.sqrt 2 + 1) → 
  b = 1 / (Real.sqrt 2 - 1) → 
  (a + b) / 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l3621_362170


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3621_362158

theorem algebraic_expression_evaluation : 
  let x : ℚ := -1
  let y : ℚ := 1/2
  2 * (x^2 - 5*x*y) - 3 * (x^2 - 6*x*y) = 3 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3621_362158


namespace NUMINAMATH_CALUDE_x_range_restriction_l3621_362153

-- Define a monotonically decreasing function on (0, +∞)
def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

-- Define the main theorem
theorem x_range_restriction 
  (f : ℝ → ℝ) 
  (h_monotonic : monotonically_decreasing f)
  (h_condition : ∀ x, 0 < x → f x < f (2*x - 2)) :
  ∀ x, (0 < x ∧ f x < f (2*x - 2)) → 1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_x_range_restriction_l3621_362153


namespace NUMINAMATH_CALUDE_parallel_condition_l3621_362102

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

/-- The lines l₁ and l₂ parameterized by m -/
def l1 (m : ℝ) : Line := ⟨1, 2*m, -1⟩
def l2 (m : ℝ) : Line := ⟨3*m+1, -m, -1⟩

/-- The statement to be proved -/
theorem parallel_condition :
  (∀ m : ℝ, are_parallel (l1 m) (l2 m) → m = -1/2 ∨ m = 0) ∧
  (∃ m : ℝ, m ≠ -1/2 ∧ m ≠ 0 ∧ ¬are_parallel (l1 m) (l2 m)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3621_362102


namespace NUMINAMATH_CALUDE_first_quarter_spending_river_town_l3621_362168

/-- The spending during the first quarter of a year, given the initial and end-of-quarter spending -/
def first_quarter_spending (initial_spending end_of_quarter_spending : ℝ) : ℝ :=
  end_of_quarter_spending - initial_spending

/-- Theorem: The spending during the first quarter is 3.1 million dollars -/
theorem first_quarter_spending_river_town : 
  first_quarter_spending 0 3.1 = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_spending_river_town_l3621_362168


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3621_362104

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * ((-2 : ℂ) + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3621_362104


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3621_362141

/-- A quadratic function passing through three given points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_theorem :
  ∃ (a b c : ℝ),
    (QuadraticFunction a b c (-2) = 9) ∧
    (QuadraticFunction a b c 0 = 3) ∧
    (QuadraticFunction a b c 4 = 3) ∧
    (∀ x, QuadraticFunction a b c x = (1/2) * x^2 - 2 * x + 3) ∧
    (let vertex_x := -b / (2*a);
     let vertex_y := QuadraticFunction a b c vertex_x;
     vertex_x = 2 ∧ vertex_y = 1) ∧
    (∀ m : ℝ,
      let y₁ := QuadraticFunction a b c m;
      let y₂ := QuadraticFunction a b c (m+1);
      (m < 3/2 → y₁ > y₂) ∧
      (m = 3/2 → y₁ = y₂) ∧
      (m > 3/2 → y₁ < y₂)) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_theorem_l3621_362141


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3621_362103

theorem cricket_bat_profit_percentage
  (cost_price_A : ℝ)
  (selling_price_C : ℝ)
  (profit_percentage_B : ℝ)
  (h1 : cost_price_A = 156)
  (h2 : selling_price_C = 234)
  (h3 : profit_percentage_B = 25)
  : (((selling_price_C / (1 + profit_percentage_B / 100)) - cost_price_A) / cost_price_A) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3621_362103


namespace NUMINAMATH_CALUDE_at_least_one_geq_quarter_l3621_362147

theorem at_least_one_geq_quarter (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : x * y * z = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * y ≥ 1/4 ∨ (1 - y) * z ≥ 1/4 ∨ (1 - z) * x ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_quarter_l3621_362147


namespace NUMINAMATH_CALUDE_max_value_after_operations_l3621_362183

def initial_numbers : List ℕ := [1, 2, 3]
def num_operations : ℕ := 9

def operation (numbers : List ℕ) : List ℕ :=
  let sum := numbers.sum
  let max := numbers.maximum?
  match max with
  | none => numbers
  | some m => (sum - m) :: (numbers.filter (· ≠ m))

def iterate_operation (n : ℕ) (numbers : List ℕ) : List ℕ :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_operation n (operation numbers)

theorem max_value_after_operations :
  (iterate_operation num_operations initial_numbers).maximum? = some 233 :=
sorry

end NUMINAMATH_CALUDE_max_value_after_operations_l3621_362183


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_fraction_bound_l3621_362150

theorem sqrt_inequality_and_fraction_bound : 
  (Real.sqrt 5 + Real.sqrt 7 > 1 + Real.sqrt 13) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y > 1 → 
    min ((1 + x) / y) ((1 + y) / x) < 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_fraction_bound_l3621_362150


namespace NUMINAMATH_CALUDE_pencil_rows_l3621_362182

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 154) (h2 : pencils_per_row = 11) :
  total_pencils / pencils_per_row = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_rows_l3621_362182


namespace NUMINAMATH_CALUDE_parabola_vector_max_value_l3621_362166

/-- The parabola C: x^2 = 4y -/
def parabola (p : ℝ × ℝ) : Prop := p.1^2 = 4 * p.2

/-- The line l intersecting the parabola at points A and B -/
def line_intersects (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ l ∧ B ∈ l ∧ parabola A ∧ parabola B

/-- Vector from origin to a point -/
def vec_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Vector between two points -/
def vec_between (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Scalar multiplication of a vector -/
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

/-- Vector equality -/
def vec_eq (v w : ℝ × ℝ) : Prop := v = w

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem parabola_vector_max_value 
  (l : Set (ℝ × ℝ)) (A B G : ℝ × ℝ) :
  line_intersects l A B →
  vec_eq (vec_between A B) (scalar_mul 2 (vec_between A G)) →
  (∃ (max : ℝ), 
    max = 16 ∧ 
    ∀ (X Y : ℝ × ℝ), parabola X → parabola Y → 
      (dot_product (vec_from_origin X) (vec_from_origin X) +
       dot_product (vec_from_origin Y) (vec_from_origin Y) -
       2 * dot_product (vec_from_origin X) (vec_from_origin Y) -
       4 * dot_product (vec_from_origin G) (vec_from_origin G)) ≤ max) :=
sorry

end NUMINAMATH_CALUDE_parabola_vector_max_value_l3621_362166


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l3621_362187

/-- Proves that the number of slightly used crayons is 56 -/
theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
  sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l3621_362187


namespace NUMINAMATH_CALUDE_twelve_digit_numbers_with_consecutive_ones_l3621_362199

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_numbers (n : ℕ) : ℕ := 2^n

theorem twelve_digit_numbers_with_consecutive_ones : 
  (valid_numbers 12) - (fibonacci 11) = 3719 := by sorry

end NUMINAMATH_CALUDE_twelve_digit_numbers_with_consecutive_ones_l3621_362199
