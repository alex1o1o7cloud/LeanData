import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_shaded_areas_II_III_equal_l582_58218

-- Define the squares
def Square : Type := Unit

-- Define the total area of each square
noncomputable def total_area (s : Square) : ℝ := 1

-- Define the shaded area for each square
noncomputable def shaded_area_I (s : Square) : ℝ := 1/4 * total_area s
noncomputable def shaded_area_II (s : Square) : ℝ := 1/2 * total_area s
noncomputable def shaded_area_III (s : Square) : ℝ := 1/2 * total_area s

-- Theorem to prove
theorem shaded_areas_comparison (s : Square) :
  shaded_area_I s = 1/4 * total_area s ∧
  shaded_area_II s = 1/2 * total_area s ∧
  shaded_area_III s = 1/2 * total_area s :=
by
  -- Split the conjunction into three parts
  constructor
  · -- Prove shaded_area_I s = 1/4 * total_area s
    rfl
  constructor
  · -- Prove shaded_area_II s = 1/2 * total_area s
    rfl
  · -- Prove shaded_area_III s = 1/2 * total_area s
    rfl

-- Corollary to explicitly state that II and III are equal
theorem shaded_areas_II_III_equal (s : Square) :
  shaded_area_II s = shaded_area_III s :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_shaded_areas_II_III_equal_l582_58218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_g_monotonicity_condition_l582_58266

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

def g (x m : ℝ) : ℝ := f x - m*x

theorem f_monotonicity_and_extrema :
  (∀ x y, x < 1 → y < 1 → x < y → f x > f y) ∧
  (∀ x y, x > 1 → y > 1 → x < y → f x < f y) ∧
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 3 → f x ≥ 1) ∧
  (∃ x, x ∈ Set.Icc (1/2 : ℝ) 3 ∧ f x = 1) ∧
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 3 → f x ≤ 5) ∧
  (∃ x, x ∈ Set.Icc (1/2 : ℝ) 3 ∧ f x = 5) :=
by sorry

theorem g_monotonicity_condition (m : ℝ) :
  (∀ x y, x ∈ Set.Icc 2 4 → y ∈ Set.Icc 2 4 → x < y → (g x m < g y m ∨ g x m > g y m)) ↔
  (m ≤ 2 ∨ m ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_g_monotonicity_condition_l582_58266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relations_l582_58285

/-- In a triangle ABC, given that angle A is greater than angle B, 
    prove that the side opposite to A is longer than the side opposite to B,
    the sine of A is greater than the sine of B, 
    and the cosine of A is less than the cosine of B. -/
theorem triangle_angle_side_relations (A B C : ℝ) (a b c : ℝ) :
  A > B → 
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a > b ∧ Real.sin A > Real.sin B ∧ Real.cos A < Real.cos B := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relations_l582_58285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_midpoints_l582_58246

/-- Given n ≥ 2 distinct points in a plane, the minimum number of distinct midpoints
    of segments formed by these points is 2n-3. -/
theorem min_distinct_midpoints (n : ℕ) (points : Fin n → ℝ × ℝ) 
    (h1 : n ≥ 2) (h2 : Function.Injective points) : 
    ∃ (midpoints : Set (ℝ × ℝ)), 
      (∀ (i j : Fin n), i < j → (points i + points j) / 2 ∈ midpoints) ∧ 
      Finset.card (Finset.univ.image (λ (p : Fin n × Fin n) => (points p.1 + points p.2) / 2)) = 2 * n - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_midpoints_l582_58246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l582_58256

open Matrix

theorem row_swap_and_triple (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  let P : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 0; 0, 0, 1; 0, 1, 0]
  P * Q = !![3 * Q 0 0, 3 * Q 0 1, 3 * Q 0 2;
             Q 2 0, Q 2 1, Q 2 2;
             Q 1 0, Q 1 1, Q 1 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l582_58256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l582_58232

theorem cube_root_eight_to_twelve : (8 : ℝ) ^ (1/3) ^ 12 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l582_58232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crescent_to_hexagon_ratio_l582_58272

/-- A regular hexagon inscribed in a circle -/
structure InscribedHexagon where
  /-- The radius of the circumscribed circle -/
  r : ℝ
  /-- Assumption that the radius is positive -/
  r_pos : r > 0

/-- The area of a regular hexagon -/
noncomputable def hexagonArea (h : InscribedHexagon) : ℝ := 
  3 * Real.sqrt 3 * h.r^2 / 2

/-- The area of a crescent formed by a semicircle on one side of the hexagon -/
noncomputable def crescentArea (h : InscribedHexagon) : ℝ := 
  h.r^2 * (Real.pi / 6 - Real.sqrt 3 / 4)

/-- The theorem stating the ratio of crescent areas to hexagon area -/
theorem crescent_to_hexagon_ratio (h : InscribedHexagon) :
  4 * crescentArea h / hexagonArea h = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crescent_to_hexagon_ratio_l582_58272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_between_s_and_p_l582_58278

/-- The total amount to be distributed -/
def total : ℕ := 10000

/-- The amount received by p -/
def p : ℕ := sorry

/-- The amount received by q -/
def q : ℕ := sorry

/-- The amount received by r -/
def r : ℕ := sorry

/-- The amount received by s -/
def s : ℕ := sorry

/-- The conditions of the problem -/
axiom condition1 : p = 2 * q
axiom condition2 : s = 4 * r
axiom condition3 : q = r
axiom condition4 : s = p / 2
axiom condition5 : p + q + r + s = total

/-- The theorem to be proved -/
theorem difference_between_s_and_p : p - s = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_between_s_and_p_l582_58278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l582_58254

/-- A pyramid with a square base, two lateral faces perpendicular to the base,
    and two lateral faces inclined at 45° to the base. -/
structure SpecialPyramid where
  l : ℝ  -- Length of the middle-sized lateral edge
  (l_pos : l > 0)

/-- The volume of the special pyramid -/
noncomputable def volume (p : SpecialPyramid) : ℝ := (p.l^3 * Real.sqrt 2) / 12

/-- The total surface area of the special pyramid -/
noncomputable def totalSurfaceArea (p : SpecialPyramid) : ℝ := (p.l^2 * (2 + Real.sqrt 2)) / 2

/-- Theorem stating the volume and surface area of the special pyramid -/
theorem special_pyramid_properties (p : SpecialPyramid) :
  volume p = (p.l^3 * Real.sqrt 2) / 12 ∧
  totalSurfaceArea p = (p.l^2 * (2 + Real.sqrt 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l582_58254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l582_58248

/-- Given a triangle ABC with angle B = π/4, prove the sine of angle A and the area of the triangle. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  B = π / 4 →
  b = Real.sqrt 5 →
  c = Real.sqrt 2 →
  a + c = 3 →
  Real.sin A = (3 * Real.sqrt 10) / 10 ∧
  (1 / 2 : ℝ) * a * c * Real.sin B = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l582_58248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_calls_per_day_l582_58257

theorem average_calls_per_day 
  (calls : Fin 5 → ℕ) 
  (total : ℕ := (Finset.sum (Finset.univ : Finset (Fin 5)) calls))
  (average : ℚ := (total : ℚ) / 5) : 
  average = ((calls 0 + calls 1 + calls 2 + calls 3 + calls 4) : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_calls_per_day_l582_58257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cost_calculation_l582_58216

theorem bottle_cost_calculation (R : ℚ) : 
  (R / 250 - 3 / 100 = 5 / 1000) → R = 875 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cost_calculation_l582_58216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l582_58240

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  sine_law : a / Real.sin A = b / Real.sin B
  sine_law' : b / Real.sin B = c / Real.sin C

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  (t.A = 2 * t.B → t.C = 5 * Real.pi / 8) ∧
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l582_58240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_proof_l582_58277

/-- Proves that given the courtyard and brick dimensions, and the total number of bricks,
    the length of each brick is 20 cm. -/
theorem brick_length_proof (courtyard_length courtyard_width brick_width : ℝ) (total_bricks : ℕ)
  (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16)
  (h3 : brick_width = 0.1)
  (h4 : total_bricks = 20000)
  : ∃ (brick_length : ℝ), brick_length = 0.2 := by
  -- The proof goes here
  sorry

#check brick_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_proof_l582_58277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l582_58214

-- Define the triangle ABC
structure Triangle (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V

-- Define a point in 2D space
abbrev Point := ℝ × ℝ

-- Define the given points D, E, and F
noncomputable def D : Point := (4, 0)
noncomputable def E : Point := (80/17, 20/17)
noncomputable def F : Point := (5/2, 5/2)

-- Define necessary concepts
def IsAcute (t : Triangle Point) : Prop := sorry
def IsAltitude (A B C P : Point) : Prop := sorry

-- Define the theorem
theorem triangle_coordinates 
  (ABC : Triangle Point) 
  (h_acute : IsAcute ABC)
  (h_altitude_D : IsAltitude ABC.A ABC.B ABC.C D)
  (h_altitude_E : IsAltitude ABC.B ABC.C ABC.A E)
  (h_altitude_F : IsAltitude ABC.C ABC.A ABC.B F) :
  ABC.A = (4, 4) ∧ ABC.B = (0, 0) ∧ ABC.C = (5, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l582_58214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_102km_l582_58239

-- Define the ship's speed in still water
def ship_speed : ℝ := sorry

-- Define the river flow rate
def river_flow_rate : ℝ := 2

-- Define the distance between ports A and B
def distance_between_ports : ℝ := sorry

-- Define the total round trip time
def total_trip_time : ℝ := 3.2

-- Define the arithmetic sequence of distances traveled in the first three hours
def distance_sequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => a + d
  | 2 => a + 2*d
  | _ => 0

-- Theorem statement
theorem total_distance_is_102km :
  (∀ n : ℕ, n < 3 → distance_sequence (ship_speed - river_flow_rate) river_flow_rate n > 0) →
  (ship_speed > river_flow_rate) →
  (distance_between_ports = 51) →
  (distance_between_ports * 2 = 102) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_102km_l582_58239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_increase_l582_58294

/-- The percentage increase in the price of a computer -/
noncomputable def percentage_increase (original_price new_price : ℝ) : ℝ :=
  ((new_price - original_price) / original_price) * 100

/-- Proof that the percentage increase in the price of a computer is 30% -/
theorem computer_price_increase (y : ℝ) (h1 : 2 * y = 540) (h2 : y > 0) : percentage_increase y 351 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_increase_l582_58294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_discount_percentage_l582_58296

theorem employee_discount_percentage
  (wholesale_cost : ℚ)
  (markup_percentage : ℚ)
  (employee_payment : ℚ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_payment = 192) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_payment
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 20 := by
  sorry

-- Remove the #eval line as it's not necessary for building
-- and might cause issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_discount_percentage_l582_58296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solutions_l582_58226

theorem problem_solutions :
  -- Part 1
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) ∧
  (∀ y : ℝ, 3*(y-1)^2 - 12 = 0 ↔ y = 3 ∨ y = -1) ∧
  (∀ t : ℝ, t*(t-2) = 3 + 4*t ↔ t = 3 + 2*Real.sqrt 3 ∨ t = 3 - 2*Real.sqrt 3) ∧
  -- Part 2
  (∀ x : ℝ, (6*x + 2 = 2*x^2 + 7*x - 1) ↔ (x = 1 ∨ x = -3/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solutions_l582_58226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l582_58210

/-- Given a train with speed excluding stoppages and stoppage time per hour,
    calculate the speed including stoppages -/
noncomputable def train_speed_with_stoppages (speed_without_stoppages : ℝ) (stoppage_time : ℝ) : ℝ :=
  let running_time := 60 - stoppage_time
  let distance := (speed_without_stoppages * running_time) / 60
  distance

/-- Theorem stating that for a train with 45 kmph speed excluding stoppages
    and 16 minutes stoppage per hour, the speed including stoppages is 33 kmph -/
theorem train_speed_theorem :
  train_speed_with_stoppages 45 16 = 33 := by
  -- Unfold the definition of train_speed_with_stoppages
  unfold train_speed_with_stoppages
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval for noncomputable functions, so we'll comment this out
-- #eval train_speed_with_stoppages 45 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l582_58210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l582_58212

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = -8

-- Define the area of the region
noncomputable def region_area : ℝ := 3 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l582_58212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoqiang_journey_l582_58223

noncomputable def distance_home_school (initial_speed : ℝ) (initial_time : ℝ) (remaining_distance : ℝ) : ℝ :=
  initial_speed * initial_time + remaining_distance

noncomputable def total_time_walked (initial_speed : ℝ) (initial_time : ℝ) (remaining_distance : ℝ) (running_speed : ℝ) : ℝ :=
  initial_time + (2 * remaining_distance) / running_speed

theorem xiaoqiang_journey 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (remaining_distance : ℝ) 
  (running_speed : ℝ) 
  (h1 : initial_speed = 60)
  (h2 : initial_time = 5)
  (h3 : remaining_distance = 700)
  (h4 : running_speed = 100) :
  distance_home_school initial_speed initial_time remaining_distance = 1000 ∧
  total_time_walked initial_speed initial_time remaining_distance running_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoqiang_journey_l582_58223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l582_58228

noncomputable section

/-- The distance between two points where a line intersects a unit sphere -/
def intersection_distance : ℝ := 10 / Real.sqrt 11

/-- Start point of the line -/
def start_point : Fin 3 → ℝ := ![3, 4, 5]

/-- End point of the line -/
def end_point : Fin 3 → ℝ := ![0, -2, -2]

/-- Center of the sphere -/
def sphere_center : Fin 3 → ℝ := ![0, 0, 0]

/-- Radius of the sphere -/
def sphere_radius : ℝ := 1

/-- The line passing through the start and end points -/
def line (t : ℝ) : Fin 3 → ℝ := 
  fun i => start_point i + t * (end_point i - start_point i)

/-- The equation of the sphere -/
def sphere_equation (x : Fin 3 → ℝ) : Prop :=
  (x 0 - sphere_center 0)^2 + (x 1 - sphere_center 1)^2 + (x 2 - sphere_center 2)^2 = sphere_radius^2

/-- The intersection points of the line and the sphere -/
def intersections : Set (Fin 3 → ℝ) :=
  {x | ∃ t, line t = x ∧ sphere_equation x}

theorem intersection_distance_proof :
  ∃ p q : Fin 3 → ℝ, p ∈ intersections ∧ q ∈ intersections ∧ p ≠ q ∧
  Real.sqrt (((p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2) : ℝ) = intersection_distance :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l582_58228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l582_58233

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Models Crystal's run -/
noncomputable def crystal_run : ℝ :=
  let start := Point.mk 0 0
  let north_point := Point.mk 0 2
  let northwest_point := Point.mk (-Real.sqrt 2 / 2) (2 + Real.sqrt 2 / 2)
  let southwest_point := Point.mk (-Real.sqrt 2) 2
  distance start southwest_point

theorem crystal_run_distance : crystal_run = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l582_58233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l582_58299

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x + 3 * a

theorem problem_solution :
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≥ 5 * a) → a > 0 ∧ a ≤ 1) ∧
  (∀ n : ℕ+, (Finset.range n).sum (λ i ↦ 1 / (i + 1 : ℝ)) > Real.log (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l582_58299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_five_frames_l582_58219

/-- Represents a square frame cut from a square sheet of paper -/
structure SquareFrame where
  outer_side : ℝ
  inner_side : ℝ

/-- Calculates the area of a square frame -/
def frame_area (frame : SquareFrame) : ℝ :=
  frame.outer_side ^ 2 - frame.inner_side ^ 2

/-- Theorem: The total area covered by five square frames with overlap -/
theorem total_area_five_frames (frame : SquareFrame)
  (h_outer : frame.outer_side = 10)
  (h_inner : frame.inner_side = 8)
  (overlap_area : ℝ)
  (h_overlap : overlap_area = 8) :
  5 * frame_area frame - overlap_area = 172 := by
  -- Expand the definition of frame_area
  have h1 : frame_area frame = frame.outer_side ^ 2 - frame.inner_side ^ 2 := rfl
  
  -- Substitute the known values
  rw [h_outer, h_inner] at h1
  
  -- Simplify the arithmetic
  have h2 : frame_area frame = 100 - 64 := by
    rw [h1]
    norm_num
  
  -- Calculate the total area of 5 frames without overlap
  have h3 : 5 * frame_area frame = 5 * 36 := by
    rw [h2]
    norm_num
  
  -- Subtract the overlap area
  rw [h3, h_overlap]
  norm_num

-- We don't need to evaluate the theorem, so we can remove the #eval line


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_five_frames_l582_58219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_new_figure_l582_58265

/-- A square with a perimeter of 60 inches -/
noncomputable def square_perimeter : ℝ := 60

/-- The side length of the square -/
noncomputable def square_side : ℝ := square_perimeter / 4

/-- The number of sides in the new figure ABFCDE -/
def new_figure_sides : ℕ := 6

/-- Theorem: The perimeter of ABFCDE is 90 inches -/
theorem perimeter_of_new_figure :
  square_side * (new_figure_sides : ℝ) = 90 := by
  -- Unfold definitions
  unfold square_side square_perimeter new_figure_sides
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_new_figure_l582_58265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l582_58217

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3) / Real.log a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ a/2 → f a x1 - f a x2 > 0) →
  1 < a ∧ a < 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l582_58217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_solutions_l582_58281

theorem quadratic_no_real_solutions (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → |m| < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_solutions_l582_58281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_parity_l582_58201

theorem expression_parity (a b c : ℕ) (ha : Odd a) (hb : Even b) :
  Even (5^a + (b-1)^2*c + b*c) ↔ Odd c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_parity_l582_58201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l582_58236

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.b * Real.sin t.C

/-- Theorem about a specific triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : Real.cos t.A = Real.sqrt 6 / 3)
  (h3 : t.B = t.A + Real.pi / 2) :
  t.b = 3 * Real.sqrt 2 ∧ 
  Triangle.area t = 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l582_58236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l582_58253

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (A^2 + B^2)

/-- Line l₁: 3x - 4y + 2 = 0 -/
def l₁ (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

/-- Line l₂: 3x - 4y - 8 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 3 (-4) 2 (-8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l582_58253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_zero_l582_58247

/-- Recursive sequence definition -/
def sequenceABCD (a₁ b₁ c₁ d₁ : ℤ) : ℕ → (ℤ × ℤ × ℤ × ℤ)
  | 0 => (a₁, b₁, c₁, d₁)
  | n + 1 => let (a, b, c, d) := sequenceABCD a₁ b₁ c₁ d₁ n
             (|a - b|, |b - c|, |c - d|, |d - a|)

/-- Theorem statement -/
theorem sequence_converges_to_zero (a₁ b₁ c₁ d₁ : ℤ) :
  ∃ k : ℕ, sequenceABCD a₁ b₁ c₁ d₁ k = (0, 0, 0, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_zero_l582_58247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_product_l582_58262

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the intersecting line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the point N
def N : ℝ × ℝ := (0, -2)

-- Define the product of slopes
def slope_product (k₁ k₂ : ℝ) : ℝ := k₁ * k₂

-- Theorem statement
theorem ellipse_line_intersection_slope_product :
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    P = (x₁, y₁) ∧ Q = (x₂, y₂) ∧
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    P ≠ Q) →
  let k₁ := (P.2 - N.2) / (P.1 - N.1);
  let k₂ := (Q.2 - N.2) / (Q.1 - N.1);
  slope_product k₁ k₂ > 49/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_product_l582_58262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l582_58249

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R h r : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: Volume of a specific truncated cone -/
theorem specific_truncated_cone_volume :
  truncated_cone_volume 8 6 4 = 224 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l582_58249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l582_58207

/-- A hyperbola centered at the origin passing through (-2, 3), (0, -1), and (s, 1) -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  s : ℝ
  /-- The hyperbola passes through (-2, 3) -/
  point1 : 9 - 4 / b^2 = 1
  /-- The hyperbola passes through (0, -1) -/
  point2 : 1 / a^2 = 1
  /-- The hyperbola passes through (s, 1) -/
  point3 : 1 / a^2 - 2 * s^2 / b^2 = 1
  /-- a and b are positive real numbers -/
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that s² = 0 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l582_58207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_f_one_l582_58204

-- Define the function f based on the graph
noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then -0.5 * x^2 - 1.5 * x + 2
  else if x ≤ 2 then -x + 3
  else 0.5 * x^2 - 1.5 * x + 2

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-4) 4

-- State the theorem
theorem unique_double_f_one :
  ∃! x, x ∈ f_domain ∧ f (f x) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_f_one_l582_58204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swift_travel_theorem_l582_58234

/-- The speed (in mph) that makes Ms. Swift 5 minutes late -/
noncomputable def late_speed : ℝ := 50

/-- The speed (in mph) that makes Ms. Swift 5 minutes early -/
noncomputable def early_speed : ℝ := 70

/-- The time difference (in hours) between arrival and scheduled time when late -/
noncomputable def late_diff : ℝ := 5 / 60

/-- The time difference (in hours) between arrival and scheduled time when early -/
noncomputable def early_diff : ℝ := 5 / 60

/-- The exact speed (in mph) required to arrive on time -/
noncomputable def exact_speed : ℝ := 58

theorem swift_travel_theorem :
  ∃ (d t : ℝ),
    d > 0 ∧ t > 0 ∧
    d = late_speed * (t + late_diff) ∧
    d = early_speed * (t - early_diff) ∧
    d / t = exact_speed := by
  sorry

#check swift_travel_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swift_travel_theorem_l582_58234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_90_l582_58293

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- The main theorem -/
theorem shaded_area_is_90 (square_side : ℝ) (triangle : Triangle) 
    (h1 : square_side = 12)
    (h2 : triangle.a = ⟨10, 0⟩)
    (h3 : triangle.b = ⟨22, 0⟩)
    (h4 : triangle.c = ⟨10, 18⟩)
    (h5 : (triangle.c.y - triangle.a.y)^2 = (triangle.b.x - triangle.a.x)^2 + (triangle.c.y - triangle.a.y)^2) -- right triangle
    (intersection : Point)
    (h6 : intersection.y = 0)
    (h7 : intersection.x = 16) :
  triangleArea (intersection.x - triangle.a.x) triangle.c.y = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_90_l582_58293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l582_58227

/-- Calculates the simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal rate : ℝ) (time : ℕ) : ℝ :=
  (principal * rate * (time : ℝ)) / 100

/-- Proves that for a loan of 350 at 4% per annum for 8 years, 
    the difference between the sum lent and the interest is 238 -/
theorem interest_difference : 
  let principal : ℝ := 350
  let rate : ℝ := 4
  let time : ℕ := 8
  let interest := simple_interest principal rate time
  principal - interest = 238 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l582_58227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_approx_l582_58274

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Prismatic structure with base triangle and height -/
structure Prism where
  base : Triangle
  height : ℝ

/-- Calculate the area of a triangle using Heron's formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Calculate the volume of a pyramid with given base area and height -/
noncomputable def pyramidVolume (baseArea height : ℝ) : ℝ :=
  (1 / 3) * baseArea * height

/-- The main theorem: volume of the pyramid is approximately 16.5 -/
theorem pyramid_volume_approx (p : Prism) (h1 : p.base.a = 6) (h2 : p.base.b = 4) 
    (h3 : p.base.c = 5) (h4 : p.height = 5) : 
    ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    |pyramidVolume (triangleArea p.base) p.height - 16.5| < ε := by
  sorry

#check pyramid_volume_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_approx_l582_58274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l582_58288

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then k * x + 7
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) (a k : ℝ) : ℝ :=
  (sales_volume x a k) * (x - 1)

-- Theorem statement
theorem max_profit_at_two :
  ∀ (a k : ℝ),
  (1 < 3 ∧ 3 ≤ 5) →
  (sales_volume 3 a k = 4) →
  (sales_volume 5 a k = 2) →
  (∀ x, 1 < x ∧ x ≤ 5 → profit x a k ≤ profit 2 a k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_two_l582_58288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_count_l582_58203

noncomputable def α : ℝ := 19.94
noncomputable def β : ℝ := α / 10

theorem light_reflection_count : 
  ∀ (AB BC : ℝ),
  AB = BC →
  (⌊(180 - 2 * α) / β⌋ : ℤ).toNat + 1 = 71 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_count_l582_58203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_needed_l582_58282

/-- Represents the number of bushes -/
structure Bushes where
  value : ℕ

/-- Represents the number of containers -/
structure Containers where
  value : ℕ

/-- Represents the number of zucchinis -/
structure Zucchinis where
  value : ℕ

/-- Represents the number of carrots -/
structure Carrots where
  value : ℕ

/-- Conversion from bushes to containers -/
def bushes_to_containers (b : Bushes) : Containers :=
  ⟨10 * b.value⟩

/-- Conversion from containers to zucchinis -/
def containers_to_zucchinis (c : Containers) : Zucchinis :=
  ⟨(3 * c.value) / 8⟩

/-- Conversion from zucchinis to carrots -/
def zucchinis_to_carrots (z : Zucchinis) : Carrots :=
  ⟨(5 * z.value) / 10⟩

/-- The main theorem stating the minimum number of bushes needed -/
theorem min_bushes_needed (target_zucchinis : Zucchinis) (target_carrots : Carrots) : 
  (target_zucchinis.value = 60 ∧ target_carrots.value = 15) →
  (∃ (b : Bushes), 
    (containers_to_zucchinis (bushes_to_containers b)).value ≥ target_zucchinis.value ∧
    (zucchinis_to_carrots (containers_to_zucchinis (bushes_to_containers b))).value ≥ target_carrots.value ∧
    ∀ (b' : Bushes), b'.value < b.value →
      (containers_to_zucchinis (bushes_to_containers b')).value < target_zucchinis.value ∨
      (zucchinis_to_carrots (containers_to_zucchinis (bushes_to_containers b'))).value < target_carrots.value) →
  (∃ (b : Bushes), b.value = 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_needed_l582_58282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_cube_root_sum_l582_58231

theorem largest_x_cube_root_sum (x : ℝ) :
  (Real.rpow x (1/3) + Real.rpow (10 - x) (1/3) = 1) → x ≤ 5 + 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_cube_root_sum_l582_58231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l582_58245

/-- Proves that given a car traveling for two hours with an average speed of 90 km/h
    and a speed of 120 km/h in the first hour, the speed in the second hour must be 60 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 120)
  (h2 : average_speed = 90) :
  2 * average_speed - speed_first_hour = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l582_58245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_time_correct_l582_58276

/-- Represents the rotation speed of a clock hand in degrees per second -/
structure RotationSpeed where
  degreesPerSecond : ℚ

/-- Represents a clock hand -/
structure ClockHand where
  length : ℚ
  rotationSpeed : RotationSpeed

/-- Represents a circular clock -/
structure CircularClock where
  secondHand : ClockHand
  minuteHand : ClockHand

/-- The time in seconds when the area of the triangle formed by the second and minute hands reaches its maximum -/
noncomputable def maxAreaTime (clock : CircularClock) : ℚ :=
  90 / (clock.secondHand.rotationSpeed.degreesPerSecond - clock.minuteHand.rotationSpeed.degreesPerSecond)

theorem max_area_time_correct (clock : CircularClock) 
  (h1 : clock.secondHand.rotationSpeed.degreesPerSecond = 6)
  (h2 : clock.minuteHand.rotationSpeed.degreesPerSecond = (1 : ℚ) / 10) :
  maxAreaTime clock = 15 + 15 / 59 := by
  sorry

#eval (15 : ℚ) + 15 / 59 -- To verify the fraction representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_time_correct_l582_58276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l582_58263

-- Define a power function as noncomputable
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- State the theorem
theorem power_function_through_point :
  ∀ α : ℝ, powerFunction α 3 = Real.sqrt 3 → powerFunction α = λ x ↦ Real.sqrt x :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l582_58263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_n_is_power_of_two_minus_one_l582_58260

/-- The number of ways to cover an n × 2k board with nk dominoes of size 2 × 1 -/
def f (n k : ℕ+) : ℕ := sorry

/-- Theorem: f(n, 2k) is odd for all positive integers k if and only if n = 2^i - 1 for some positive integer i -/
theorem f_odd_iff_n_is_power_of_two_minus_one (n : ℕ+) :
  (∀ k : ℕ+, Odd (f n (2 * k))) ↔ ∃ i : ℕ+, n = 2^(i : ℕ) - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_n_is_power_of_two_minus_one_l582_58260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_plus_one_l582_58279

-- Define the polynomial p(x)
def p (x : ℂ) : ℂ := 985 * x^2021 + 211 * x^2020 - 211

-- Define the set of roots of p(x)
noncomputable def roots : Finset ℂ := sorry

-- State the theorem
theorem sum_of_reciprocals_squared_plus_one :
  Finset.sum roots (λ x ↦ 1 / (x^2 + 1)) = 2021 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_plus_one_l582_58279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l582_58264

/-- Represents the three tribes --/
inductive Tribe
  | One
  | Two
  | Three

/-- Represents a person in the circle --/
structure Person where
  tribe : Tribe

/-- Represents the circle of people --/
def Circle := Vector Person 8

/-- Checks if two tribes are different --/
def differentTribe (t1 t2 : Tribe) : Prop :=
  t1 ≠ t2

/-- Checks if a person's statement about their left neighbor is correct according to the rules --/
def statementIsValid (circle : Circle) (index : Fin 8) : Prop :=
  let leftNeighbor := circle.get ⟨(index.val - 1 + 8) % 8, by sorry⟩
  let currentPerson := circle.get index
  let rightNeighbor := circle.get ⟨(index.val + 1) % 8, by sorry⟩
  (differentTribe currentPerson.tribe rightNeighbor.tribe →
    differentTribe currentPerson.tribe leftNeighbor.tribe) ∧
  (¬differentTribe currentPerson.tribe rightNeighbor.tribe →
    ¬differentTribe currentPerson.tribe leftNeighbor.tribe)

/-- The main theorem stating that a valid arrangement exists --/
theorem valid_arrangement_exists : ∃ (circle : Circle), ∀ (i : Fin 8), statementIsValid circle i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l582_58264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_S_l582_58292

def S : ℕ := 2^2024 + 2^2023 + 2^2022

def is_factor (n m : ℕ) : Bool := m ≠ 0 && n % m = 0

def count_factors (ns : List ℕ) (m : ℕ) : ℕ :=
  (ns.filter (λ n => is_factor m n)).length

theorem factors_of_S :
  count_factors [6, 7, 8, 9, 10] S = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_S_l582_58292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l582_58229

/-- Represents a number in base 8 (octal) --/
structure Octal where
  value : ℕ

/-- Converts an Octal number to its decimal (ℕ) representation --/
def octal_to_decimal (n : Octal) : ℕ := sorry

/-- Converts a decimal (ℕ) number to its Octal representation --/
def decimal_to_octal (n : ℕ) : Octal := sorry

/-- Subtracts two Octal numbers --/
def octal_sub (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Helper function to create Octal numbers --/
def mk_octal (n : ℕ) : Octal := ⟨n⟩

theorem octal_subtraction_theorem :
  octal_sub (mk_octal 752) (mk_octal 364) = mk_octal 376 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l582_58229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_implies_b_equals_5_l582_58222

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 - 5 else b * x + 7

theorem continuity_implies_b_equals_5 :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f b x - f b 3| < ε) → b = 5 :=
by
  sorry

#check continuity_implies_b_equals_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_implies_b_equals_5_l582_58222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_sevenths_l582_58209

/-- The sum of the nth group in the series -/
def group_sum (n : ℕ) : ℚ := (1 / 8^(n-1)) * (1 - 1/2 - 1/4)

/-- The infinite series in question -/
noncomputable def series_sum : ℚ := ∑' n, group_sum n

/-- Theorem stating that the sum of the series is 2/7 -/
theorem series_sum_equals_two_sevenths : series_sum = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_sevenths_l582_58209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l582_58250

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l582_58250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l582_58202

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: The distance between parallel sides of a trapezium with given dimensions -/
theorem trapezium_height (a b area : ℝ) (ha : a = 26) (hb : b = 18) (harea : area = 330) :
  ∃ h : ℝ, trapezium_area a b h = area ∧ h = 15 := by
  sorry

#check trapezium_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l582_58202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_specific_l582_58252

noncomputable section

/-- The area of a kite given its diagonals and the angle between them -/
def kite_area (d1 d2 θ : ℝ) : ℝ := (1/2) * d1 * d2 * Real.sin θ

/-- Theorem: The area of a kite with specified properties is 12√3 cm² -/
theorem kite_area_specific : 
  let d1 : ℝ := 6  -- smaller diagonal
  let d2 : ℝ := 8  -- larger diagonal (4/3 * 6)
  let θ : ℝ := Real.pi/3  -- 60° in radians
  kite_area d1 d2 θ = 12 * Real.sqrt 3 := by
  sorry

end noncomputable section

-- This evaluation will not work in Lean 4 due to noncomputability
-- #eval kite_area 6 8 (Real.pi/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_specific_l582_58252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l582_58224

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) / x ≥ 2

-- Define the solution set
def solution_set : Set ℝ := { x | x ∈ Set.Ico (-1) 0 }

-- Theorem statement
theorem inequality_solution_set : 
  { x : ℝ | inequality x } = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l582_58224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_triangle_inequality_l582_58220

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_triangle_inequality (a b : E) : 
  abs (‖a‖ - ‖b‖) ≤ ‖a + b‖ ∧ ‖a + b‖ ≤ ‖a‖ + ‖b‖ ∧
  abs (‖a‖ - ‖b‖) ≤ ‖a - b‖ ∧ ‖a - b‖ ≤ ‖a‖ + ‖b‖ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_triangle_inequality_l582_58220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_squared_minus_g_squared_l582_58280

-- Define the interval (-π/2, π/2)
def I : Set ℝ := {x | -Real.pi/2 < x ∧ x < Real.pi/2}

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem f_squared_minus_g_squared 
  (f g : ℝ → ℝ) 
  (h1 : ∀ x ∈ I, f x + g x = Real.sqrt ((1 + Real.cos (2*x)) / (1 - Real.sin x)))
  (h2 : is_odd f)
  (h3 : is_even g) :
  ∀ x ∈ I, (f x)^2 - (g x)^2 = -2 * Real.cos x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_squared_minus_g_squared_l582_58280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_lower_bound_l582_58298

/-- Function representing the volume of a tetrahedron given its skew edge distances. -/
noncomputable def volume_of_tetrahedron (h₁ h₂ h₃ : ℝ) : ℝ :=
  sorry

/-- A tetrahedron with skew edge distances h₁, h₂, and h₃ has a volume not less than 1/3 h₁h₂h₃. -/
theorem tetrahedron_volume_lower_bound 
  (h₁ h₂ h₃ : ℝ) 
  (V : ℝ) 
  (h₁_pos : h₁ > 0) 
  (h₂_pos : h₂ > 0) 
  (h₃_pos : h₃ > 0) 
  (V_pos : V > 0)
  (hV : V = volume_of_tetrahedron h₁ h₂ h₃) :
  V ≥ (1/3 : ℝ) * h₁ * h₂ * h₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_lower_bound_l582_58298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_2C_l582_58238

theorem triangle_cos_2C (A B C : ℝ) (S : ℝ) : 
  let BC := 8
  let AC := 5
  S = 12 →
  S = (1/2) * BC * AC * Real.sin C →
  Real.cos (2 * C) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_2C_l582_58238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_l582_58270

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x-1)
def domain_f_x_minus_1 : Set ℝ := Set.Ioc 3 7

-- State the theorem
theorem domain_f_2x (h : ∀ x, f (x - 1) ∈ domain_f_x_minus_1 ↔ x ∈ domain_f_x_minus_1) :
  ∀ x, f (2 * x) ∈ Set.Ioc 1 3 ↔ x ∈ Set.Ioc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_l582_58270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_l582_58215

/-- A hyperbola with given properties -/
structure Hyperbola where
  -- Asymptotic lines are y = ±2x
  asymptote_slope : ℝ
  asymptote_slope_eq : asymptote_slope = 2
  -- Passes through the point (-3, 4√2)
  point_x : ℝ
  point_y : ℝ
  point_eq : (point_x, point_y) = (-3, 4 * Real.sqrt 2)

/-- The intersecting line -/
structure IntersectingLine where
  -- Line equation: 4x - y - 6 = 0
  slope : ℝ
  intercept : ℝ
  line_eq : (slope, intercept) = (4, 6)

/-- Main theorem about the hyperbola and intersection -/
theorem hyperbola_intersection (h : Hyperbola) (l : IntersectingLine) :
  -- The equation of the hyperbola is x² - y²/4 = 1
  (∀ x y, (x^2 - y^2/4 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 - p.2^2/4 = 1}) ∧
  -- The length |AB| of the intersection is 2√102/3
  (∃ A B : ℝ × ℝ, 
    A ∈ {p : ℝ × ℝ | p.1^2 - p.2^2/4 = 1} ∧ 
    B ∈ {p : ℝ × ℝ | p.1^2 - p.2^2/4 = 1} ∧
    A ∈ {p : ℝ × ℝ | 4 * p.1 - p.2 - 6 = 0} ∧ 
    B ∈ {p : ℝ × ℝ | 4 * p.1 - p.2 - 6 = 0} ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 102 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_l582_58215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l582_58241

/-- Represents a rectangle -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Defines similarity between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop :=
  R1.side1 / R2.side1 = R1.side2 / R2.side2

/-- Given a rectangle R1 with one side 4 inches and area 24 square inches,
    and a similar rectangle R2 with diagonal 17 inches,
    prove that the area of R2 is 433.5/3.25 square inches. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) : 
  R1.side1 = 4 →
  R1.area = 24 →
  R2.diagonal = 17 →
  Similar R1 R2 →
  R2.area = 433.5 / 3.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l582_58241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_fraction_l582_58275

/-- Given a ball dropped from an initial height that bounces with a constant fraction,
    this function calculates the total distance traveled as an infinite series sum. -/
noncomputable def totalDistance (initialHeight : ℝ) (fraction : ℝ) : ℝ :=
  initialHeight / (1 - 2 * fraction)

/-- Theorem stating that if a ball is dropped from 120 meters and travels a total of 1080 meters,
    then the fraction of the height it rebounds to after striking the floor is 4/9. -/
theorem ball_bounce_fraction :
  ∃ (f : ℝ), totalDistance 120 f = 1080 ∧ f = 4/9 := by
  -- We claim that f = 4/9 satisfies the conditions
  use 4/9
  constructor
  · -- Show that totalDistance 120 (4/9) = 1080
    rw [totalDistance]
    norm_num
  · -- Trivially, 4/9 = 4/9
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_fraction_l582_58275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_l582_58269

/-- The number that satisfies the given equation is approximately 0.03 -/
theorem approximate_solution : ∃ x : ℝ, 
  abs ((69.28 * 0.004) / x - 9.237333333333334) < 0.000001 ∧ 
  abs (x - 0.03) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_l582_58269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l582_58289

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

theorem axis_of_symmetry :
  ∃ (k : ℤ), ∀ x, f (x - π/12) = f (-x - π/12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l582_58289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l582_58283

-- Define the curves in Cartesian coordinates
def C₁ (x y : ℝ) : Prop := x + y = 2
def C₂ (x y φ : ℝ) : Prop := x = 3 + 3 * Real.cos φ ∧ y = 3 * Real.sin φ ∧ 0 ≤ φ ∧ φ < 2 * Real.pi

-- Define the ray l
def l (ρ θ α : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ Real.pi / 4

-- Theorem statement
theorem curve_intersections :
  ∃ (polar_C₁ : ℝ → ℝ → Prop) (polar_C₂ : ℝ → ℝ → Prop),
    (∀ ρ θ, polar_C₁ ρ θ ↔ ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2) ∧
    (∀ ρ θ, polar_C₂ ρ θ ↔ ρ = 6 * Real.cos θ) ∧
    (∀ α, α_range α →
      ∃ (OA OB : ℝ),
        (∃ ρ, l ρ α α ∧ polar_C₁ ρ α) →
        (∃ ρ, l ρ α α ∧ polar_C₂ ρ α) →
        3 ≤ OB / OA ∧ OB / OA ≤ 3 / 2 * (Real.sqrt 2 + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l582_58283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_triangle_area_l582_58273

-- Define the ellipse
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the circle (renamed to avoid conflict)
def circleEq (c : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = c^2

-- Define the focal distance
noncomputable def focal_distance (a : ℝ) : ℝ := Real.sqrt (a^2 - 1)

-- Define area of triangle (placeholder function)
noncomputable def area_triangle (P F₁ F₂ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_circle_tangent_triangle_area 
  (a : ℝ) 
  (h_a : a > 1) 
  (P : ℝ × ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_ellipse : ellipse a P.1 P.2) 
  (h_circle : circleEq (focal_distance a / 2) P.1 P.2) 
  (h_tangent : ∀ x y, ellipse a x y → circleEq (focal_distance a / 2) x y → (x, y) = P) 
  (h_foci : F₁.1 = -focal_distance a ∧ F₁.2 = 0 ∧ F₂.1 = focal_distance a ∧ F₂.2 = 0) :
  area_triangle P F₁ F₂ = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_triangle_area_l582_58273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_sunscreen_reapplication_l582_58244

/-- Represents the problem of calculating sunscreen reapplication frequency -/
def SunscreenProblem (beach_hours : ℕ) (ounces_per_application : ℕ) (ounces_per_bottle : ℕ) (cost_per_bottle : ℚ) (total_cost : ℚ) : Prop :=
  let applications_per_bottle : ℕ := ounces_per_bottle / ounces_per_application
  let bottles_bought : ℕ := (total_cost / cost_per_bottle).num.natAbs
  let total_applications : ℕ := applications_per_bottle * bottles_bought
  let reapplication_interval : ℕ := beach_hours / total_applications
  reapplication_interval = 2

/-- Theorem stating that Tiffany needs to reapply sunscreen every 2 hours -/
theorem tiffany_sunscreen_reapplication :
  SunscreenProblem 16 3 12 (7/2) 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_sunscreen_reapplication_l582_58244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_z_eq_55_div_6_l582_58295

/-- A parallelogram with specific side lengths and angle -/
structure SpecificParallelogram where
  -- Side lengths
  side1 : ℝ
  side2 : ℝ → ℝ
  side3 : ℝ → ℝ
  side4 : ℝ
  -- Angle between side1 and side3
  angle : ℝ
  -- Conditions
  side1_eq : side1 = 15
  side2_eq : ∀ z, side2 z = 4 * z + 1
  side3_eq : ∀ y, side3 y = 3 * y - 2
  side4_eq : side4 = 15
  angle_eq : angle = 60 * π / 180  -- 60° in radians

/-- The sum of y and z in the specific parallelogram -/
noncomputable def sum_y_z (p : SpecificParallelogram) : ℝ :=
  let y := (p.side1 + 2) / 3
  let z := (p.side1 - 1) / 4
  y + z

/-- Theorem: The sum of y and z in the specific parallelogram equals 55/6 -/
theorem sum_y_z_eq_55_div_6 (p : SpecificParallelogram) : sum_y_z p = 55 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_z_eq_55_div_6_l582_58295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_type_sequence_problem_l582_58259

/-- Definition of M-type sequence -/
def is_m_type_sequence (a : ℕ → ℝ) : Prop :=
  ∃ p q : ℝ, ∀ n : ℕ, a (n + 1) = p * a n + q

/-- Sequence b_n -/
def b (n : ℕ) : ℝ := 2 * n

/-- Sequence c_n -/
def c : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 1 => c n + 2^n

theorem m_type_sequence_problem :
  (∃ p q : ℝ, ∀ n : ℕ, b (n + 1) = p * b n + q ∧ p = 1 ∧ q = 2) ∧
  (∀ n : ℕ, c n = 2^n - 1) ∧
  is_m_type_sequence c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_type_sequence_problem_l582_58259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l582_58290

-- Define the circle and ellipse
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + (y + 1)^2 = 9

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circle_eq x y ∧ ellipse_eq x y}

-- Define the property of being an equilateral triangle
def is_equilateral_triangle (points : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), points = {a, b, c} ∧
    ∀ (p q : ℝ × ℝ), p ∈ points → q ∈ points → p ≠ q →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = 3

-- Theorem statement
theorem intersection_forms_equilateral_triangle :
  is_equilateral_triangle intersection_points := by
  sorry

#check intersection_forms_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l582_58290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_sin_cos_l582_58286

-- Define the sine and cosine functions over the interval [0, 2π]
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos x

-- Define the domain
def domain : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Define the area function
noncomputable def area : ℝ :=
  ∫ x in domain, |f x - g x|

-- State the theorem
theorem enclosed_area_sin_cos : area = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_sin_cos_l582_58286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_meeting_recipes_l582_58221

/-- Calculates the minimum number of full recipes needed for a Math Club meeting --/
def math_club_cookies (total_students : ℕ) (attendance_rate : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let attending_students := (total_students : ℚ) * attendance_rate
  let total_cookies_needed := (attending_students * cookies_per_student).ceil
  let recipes_needed := (total_cookies_needed / cookies_per_recipe : ℚ).ceil
  recipes_needed.toNat

/-- Proves that 14 recipes are needed for the given Math Club meeting conditions --/
theorem math_club_meeting_recipes : 
  math_club_cookies 150 (3/5) 3 20 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_meeting_recipes_l582_58221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l582_58205

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^3))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l582_58205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_contains_similar_halves_l582_58200

/-- A polygon in 2D space -/
structure Polygon where
  vertices : Set (ℝ × ℝ)

/-- A convex polygon -/
def ConvexPolygon (p : Polygon) : Prop :=
  ∀ x y, x ∈ p.vertices → y ∈ p.vertices → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (t * x.1 + (1 - t) * y.1, t * x.2 + (1 - t) * y.2) ∈ p.vertices

/-- Similarity between two polygons with a given scale factor -/
def Similar (p q : Polygon) (scale : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ v, v ∈ p.vertices → 
    ∃ w, w ∈ q.vertices ∧ w = (center.1 + scale * (v.1 - center.1), center.2 + scale * (v.2 - center.2))

/-- Non-overlapping polygons -/
def NonOverlapping (p q : Polygon) : Prop :=
  p.vertices ∩ q.vertices = ∅

/-- Containment of one polygon within another -/
def Contains (outer inner : Polygon) : Prop :=
  inner.vertices ⊆ outer.vertices

/-- The main theorem -/
theorem convex_polygon_contains_similar_halves (Φ : Polygon) (h : ConvexPolygon Φ) :
  ∃ (Φ₁ Φ₂ : Polygon), 
    Similar Φ Φ₁ (1/2) ∧ 
    Similar Φ Φ₂ (1/2) ∧ 
    NonOverlapping Φ₁ Φ₂ ∧ 
    Contains Φ Φ₁ ∧ 
    Contains Φ Φ₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_contains_similar_halves_l582_58200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_center_on_regression_line_l582_58235

/-- Represents a sample point -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents the coefficients of a linear regression line -/
structure RegressionCoefficients where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the mean of a list of real numbers -/
noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

/-- Theorem: The sample center point lies on the regression line -/
theorem sample_center_on_regression_line 
  (samples : List SamplePoint) 
  (coeffs : RegressionCoefficients) : 
  let x_mean := mean (samples.map (·.x))
  let y_mean := mean (samples.map (·.y))
  y_mean = coeffs.b * x_mean + coeffs.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_center_on_regression_line_l582_58235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l582_58261

theorem trigonometric_inequality (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ ∈ Set.Ioo (-π/2) (π/2))
  (h₂ : θ₂ ∈ Set.Ioo (-π/2) (π/2))
  (h₃ : θ₃ ∈ Set.Ioo (-π/2) (π/2))
  (h₄ : θ₄ ∈ Set.Ioo (-π/2) (π/2)) :
  (∃ x : ℝ, (Real.cos θ₁)^2 * (Real.cos θ₂)^2 - (Real.sin θ₁ * Real.sin θ₂ - x)^2 ≥ 0 ∧
             (Real.cos θ₃)^2 * (Real.cos θ₄)^2 - (Real.sin θ₃ * Real.sin θ₄ - x)^2 ≥ 0) ↔
  (Real.sin θ₁)^2 + (Real.sin θ₂)^2 + (Real.sin θ₃)^2 + (Real.sin θ₄)^2 ≤
    2 * (1 + Real.sin θ₁ * Real.sin θ₂ * Real.sin θ₃ * Real.sin θ₄ + Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ * Real.cos θ₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l582_58261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_l582_58255

-- Define a structure for the trapezoid
structure Trapezoid where
  lateral_side1_angle : ℝ
  lateral_side2_length : ℝ
  lateral_side2_angle : ℝ
  smaller_base : ℝ
  median : ℝ

theorem trapezoid_median (α β a b : ℝ) (h1 : β > α) (h2 : α > 0) (h3 : β < π) : 
  let median := b + (a * Real.sin (β - α)) / (2 * Real.sin α)
  ∀ trapezoid : Trapezoid, 
    (trapezoid.lateral_side1_angle = α) → 
    (trapezoid.lateral_side2_length = a) → 
    (trapezoid.lateral_side2_angle = β) → 
    (trapezoid.smaller_base = b) → 
    (trapezoid.median = median) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_l582_58255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_distance_theorem_l582_58284

/-- First plane equation: 4x - 2y + 2z = 10 -/
def plane1 (x y z : ℝ) : Prop := 4*x - 2*y + 2*z = 10

/-- Second plane equation: 8x - 4y + 4z = 4 -/
def plane2 (x y z : ℝ) : Prop := 8*x - 4*y + 4*z = 4

/-- Normal vector of the planes -/
def normal : Fin 3 → ℝ := ![4, -2, 2]

/-- Distance between two parallel planes -/
noncomputable def distance_between_planes : ℝ := 2 * Real.sqrt 6 / 3

/-- Point (1,0,0) -/
def point : Fin 3 → ℝ := ![1, 0, 0]

/-- Distance from point to second plane -/
noncomputable def distance_point_to_plane : ℝ := Real.sqrt 6 / 3

theorem plane_distance_theorem :
  (∀ x y z, plane1 x y z ↔ 4*x - 2*y + 2*z = 10) ∧
  (∀ x y z, plane2 x y z ↔ 8*x - 4*y + 4*z = 4) ∧
  (normal = ![4, -2, 2]) →
  (distance_between_planes = 2 * Real.sqrt 6 / 3) ∧
  (distance_point_to_plane = Real.sqrt 6 / 3) := by
  sorry

#check plane_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_distance_theorem_l582_58284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_relation_l582_58291

/-- An arithmetic sequence with n terms -/
structure ArithmeticSequence where
  n : ℕ
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Sum of terms in an arithmetic sequence -/
noncomputable def sum_of_terms (seq : ArithmeticSequence) : ℝ :=
  seq.n * seq.a + (seq.n * (seq.n - 1) * seq.d) / 2

/-- Sum of squares of terms in an arithmetic sequence -/
noncomputable def sum_of_squares (seq : ArithmeticSequence) : ℝ :=
  seq.n * seq.a^2 + seq.n * (seq.n - 1) * seq.a * seq.d + 
  (seq.n * (seq.n - 1) * (2 * seq.n - 1) * seq.d^2) / 6

/-- Sum of cubes of terms in an arithmetic sequence -/
noncomputable def sum_of_cubes (seq : ArithmeticSequence) : ℝ :=
  seq.n * seq.a^3 + 
  (3 * seq.n * (seq.n - 1) * seq.a^2 * seq.d) / 2 +
  (seq.n * (seq.n - 1) * (2 * seq.n - 1) * seq.a * seq.d^2) / 2 +
  (seq.n^2 * (seq.n - 1)^2 * seq.d^3) / 4

/-- The main theorem to be proved -/
theorem arithmetic_sequence_relation (seq : ArithmeticSequence) :
  let s₁ := sum_of_terms seq
  let s₂ := sum_of_squares seq
  let s₃ := sum_of_cubes seq
  seq.n^2 * s₃ - 3 * seq.n * s₁ * s₂ + 2 * s₁^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_relation_l582_58291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_square_product_count_l582_58271

theorem max_perfect_square_product_count (n : ℕ) (h : n = 2016) : 
  (∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x ≤ n) ∧ 
    (∀ x y, x ∈ S → y ∈ S → ∃ z : ℕ, x * y = z * z) ∧
    (∀ T : Finset ℕ, (∀ x, x ∈ T → x ≤ n) → 
      (∀ x y, x ∈ T → y ∈ T → ∃ z : ℕ, x * y = z * z) → 
      T.card ≤ S.card)) →
  (∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x ≤ n) ∧ 
    (∀ x y, x ∈ S → y ∈ S → ∃ z : ℕ, x * y = z * z) ∧
    S.card = 44) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_square_product_count_l582_58271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_at_22_3_l582_58243

-- Define the piecewise function g(x)
noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 + 1 else b * x + 6

-- Theorem stating that b = 22/3 makes g continuous
theorem g_continuous_at_22_3 :
  Continuous (g (22/3)) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_at_22_3_l582_58243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_sum_l582_58268

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 5

theorem zero_point_interval_sum (a b : ℕ) (x₀ : ℝ) : 
  a > 0 → b > 0 → b - a = 1 → f x₀ = 0 → x₀ ∈ Set.Icc (a : ℝ) (b : ℝ) → a + b = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_sum_l582_58268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l582_58267

noncomputable def PowerFunction (α : ℝ) : ℝ → ℝ := λ x => x ^ α

theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  (∃ α : ℝ, f = PowerFunction α) →  -- f is a power function
  f 2 = Real.sqrt 2 / 2 →           -- f passes through (2, √2/2)
  f 4 = 1 / 2                       -- f(4) = 1/2
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l582_58267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_points_theorem_l582_58287

def M : Finset Int := {1, -2, 3}
def N : Finset Int := {-4, 5, 6, -7}

def is_in_third_or_fourth_quadrant (p : Int × Int) : Prop :=
  p.2 < 0

def count_points_in_third_and_fourth_quadrants : Nat :=
  (M.card * (N.filter (λ x => x < 0)).card) + (N.card * (M.filter (λ x => x < 0)).card)

theorem count_points_theorem : 
  count_points_in_third_and_fourth_quadrants = 10 := by
  -- Unfold the definition
  unfold count_points_in_third_and_fourth_quadrants
  -- Evaluate the expression
  simp [M, N]
  -- The result should be 10
  rfl

#eval count_points_in_third_and_fourth_quadrants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_points_theorem_l582_58287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_b_zero_l582_58206

noncomputable def f (x : ℝ) : ℝ := (x + 1)^3 + x / (x + 1)

def line (b : ℝ) (x : ℝ) : ℝ := -x + b

theorem intersection_sum_implies_b_zero (b : ℝ) :
  (∃ S : Finset ℝ, (∀ x ∈ S, f x = line b x) ∧ (S.sum id = -2)) →
  b = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_b_zero_l582_58206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l582_58213

theorem equation_solution : ∃ x : ℝ, (4 : ℝ)^x + (2 : ℝ)^x - 2 = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l582_58213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_neg_one_third_l582_58242

noncomputable section

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (Real.sqrt (1/2), 0)

/-- The line passing through a focus at angle π/4 -/
def line_through_focus (x y : ℝ) : Prop :=
  y - focus.2 = Real.tan (Real.pi/4) * (x - focus.1) ∨
  y - focus.2 = Real.tan (Real.pi/4) * (x + focus.1)

/-- Theorem: Dot product of OA and OB is -1/3 -/
theorem dot_product_is_neg_one_third :
  ∀ (A B : ℝ × ℝ),
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  line_through_focus A.1 A.2 →
  line_through_focus B.1 B.2 →
  A ≠ B →
  (A.1 * B.1 + A.2 * B.2) = -1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_neg_one_third_l582_58242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_asymptote_l582_58251

/-- Hyperbola C defined by x²/3 - y²/3 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 3 - p.2^2 / 3 = 1}

/-- One of the foci of the hyperbola C -/
noncomputable def F : ℝ × ℝ := (Real.sqrt 6, 0)

/-- One of the asymptotes of the hyperbola C -/
def asymptote : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1}

/-- The distance from a point to a line in ℝ² -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem distance_from_focus_to_asymptote :
  distancePointToLine F asymptote = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_asymptote_l582_58251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_slope_sum_l582_58237

/-- Parabola C: y = 2x^2 -/
noncomputable def C (x : ℝ) : ℝ := 2 * x^2

/-- Line l: y = kx + 1 -/
noncomputable def l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- Intersection points of C and l -/
noncomputable def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, C x = l k x ∧ p = (x, C x)}

/-- Slope of line from origin to point -/
noncomputable def slope_from_origin (p : ℝ × ℝ) : ℝ := p.2 / p.1

theorem intersection_and_slope_sum (k : ℝ) :
  (∃ A B, A ∈ intersection_points k ∧ B ∈ intersection_points k ∧ A ≠ B) ∧
  (∀ A B, A ∈ intersection_points k → B ∈ intersection_points k → A ≠ B →
    slope_from_origin A + slope_from_origin B = 1) →
  k = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_slope_sum_l582_58237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2016_l582_58225

noncomputable def mySequence (n : ℕ) : ℝ :=
  match n with
  | 0 => Real.sqrt 3
  | n + 1 => ⌊mySequence n⌋ + 1 / (mySequence n - ⌊mySequence n⌋)

theorem mySequence_2016 : mySequence 2016 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2016_l582_58225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_arrangement_theorem_l582_58258

/-- Represents a great circle arc on a unit sphere -/
structure GreatCircleArc where
  length : Real
  -- Other properties of the arc could be added here

/-- Represents an arrangement of great circle arcs on a unit sphere -/
structure ArcArrangement where
  n : Nat
  arcs : Finset GreatCircleArc
  pairwise_nonintersecting : ∀ a b, a ∈ arcs → b ∈ arcs → a ≠ b → True  -- Placeholder for non-intersection property
  equal_length : ∀ a b, a ∈ arcs → b ∈ arcs → a.length = b.length

/-- The main theorem about arranging arcs on a unit sphere -/
theorem arc_arrangement_theorem (n : Nat) (h : n > 2) :
  (∃ α : Real, α < π + 2*π/n ∧ 
    ∃ arr : ArcArrangement, arr.n = n ∧ ∀ a, a ∈ arr.arcs → a.length = α) ∧
  (∀ α : Real, α > π + 2*π/n → 
    ¬∃ arr : ArcArrangement, arr.n = n ∧ ∀ a, a ∈ arr.arcs → a.length = α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_arrangement_theorem_l582_58258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorts_cost_l582_58208

def initial_amount : ℕ := 50
def jersey_cost : ℕ := 2
def jersey_count : ℕ := 5
def basketball_cost : ℕ := 18
def remaining_amount : ℕ := 14

theorem shorts_cost : 
  initial_amount - (jersey_cost * jersey_count + basketball_cost) - remaining_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorts_cost_l582_58208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_sum_l582_58297

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_2, a_4, and a_8 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℚ) : Prop :=
  (a 4) ^ 2 = (a 2) * (a 8)

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_sum (a : ℕ → ℚ) :
  arithmetic_seq a → geometric_subseq a →
  arithmetic_sum a 10 = 110 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_sum_l582_58297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l582_58211

theorem triangle_angle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  3 * a = 2 * b →
  (Real.sin A) / a = (Real.sin B) / b → -- Sine Rule
  (Real.sin A) / a = (Real.sin C) / c → -- Sine Rule
  (2 * (Real.sin B)^2 - (Real.sin A)^2) / (Real.sin A)^2 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l582_58211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_students_not_set_l582_58230

-- Define a predicate for being a student at Xinhua High School
def is_student_at_xinhua_high_school : Type → Prop := sorry

-- Define a predicate for being taller (comparative height)
def is_taller : Type → Type → Prop := sorry

-- Define what it means for a collection to have definite elements
def has_definite_elements : (Type → Prop) → Prop := sorry

-- Theorem stating that "All taller students at Xinhua High School" cannot form a set
theorem taller_students_not_set :
  ¬ ∃ (S : Type → Prop), 
    (∀ (x : Type), S x ↔ (is_student_at_xinhua_high_school x ∧ ∃ (y : Type), is_student_at_xinhua_high_school y ∧ is_taller x y)) ∧
    has_definite_elements S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_students_not_set_l582_58230
