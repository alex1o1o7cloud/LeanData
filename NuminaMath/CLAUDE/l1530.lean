import Mathlib

namespace NUMINAMATH_CALUDE_petya_sum_theorem_l1530_153038

/-- Represents Petya's operation on the board numbers -/
def petyaOperation (x y z : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, y, z - 1, x * y)

/-- Represents the invariant property throughout Petya's operations -/
def invariant (x y z : ℕ) (sum : ℕ) : Prop :=
  x * y * z = sum + x * y * z

/-- The main theorem stating that the sum of products on the paper
    equals the initial product of board numbers when process terminates -/
theorem petya_sum_theorem (x y z : ℕ) :
  ∃ (n : ℕ) (sum : ℕ),
    (∃ (a b : ℕ), a * b * 0 = n) ∧
    invariant x y z sum ∧
    sum = x * y * z := by
  sorry

end NUMINAMATH_CALUDE_petya_sum_theorem_l1530_153038


namespace NUMINAMATH_CALUDE_inequality_proof_l1530_153070

theorem inequality_proof (x b : ℝ) (h1 : x < b) (h2 : b < 0) (h3 : b = -2) :
  x^2 > b*x ∧ b*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1530_153070


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_bounded_f_l1530_153092

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem range_of_a_for_bounded_f :
  ∀ a : ℝ, (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = -1 ∨ a = 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_bounded_f_l1530_153092


namespace NUMINAMATH_CALUDE_inequality_implication_l1530_153097

theorem inequality_implication (x y : ℝ) : 5 * x > -5 * y → x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1530_153097


namespace NUMINAMATH_CALUDE_group_size_l1530_153090

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The total weight of the group before the change -/
def W : ℝ := sorry

/-- The weight increase when the new person joins -/
def weight_increase : ℝ := 2.5

/-- The weight of the person being replaced -/
def old_weight : ℝ := 55

/-- The weight of the new person -/
def new_weight : ℝ := 75

theorem group_size :
  (W + new_weight - old_weight) / n = W / n + weight_increase →
  n = 8 := by sorry

end NUMINAMATH_CALUDE_group_size_l1530_153090


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1530_153035

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  min area1 area2 = 6 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1530_153035


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1530_153064

theorem complex_modulus_problem (z : ℂ) (h : z = (2 + Complex.I) / Complex.I + Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1530_153064


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l1530_153058

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,4,6}
def B : Finset Nat := {2,4,5,6}

theorem intersection_complement_equal : A ∩ (U \ B) = {1,3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l1530_153058


namespace NUMINAMATH_CALUDE_vertical_translation_equation_translated_line_equation_l1530_153093

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + d }

theorem vertical_translation_equation (l : Line) (d : ℝ) :
  (translateVertically l d).slope = l.slope ∧
  (translateVertically l d).intercept = l.intercept + d := by
  sorry

/-- The original line y = -2x + 1 -/
def originalLine : Line :=
  { slope := -2, intercept := 1 }

/-- The translation distance -/
def translationDistance : ℝ := 2

theorem translated_line_equation :
  translateVertically originalLine translationDistance =
  { slope := -2, intercept := 3 } := by
  sorry

end NUMINAMATH_CALUDE_vertical_translation_equation_translated_line_equation_l1530_153093


namespace NUMINAMATH_CALUDE_a_range_theorem_l1530_153001

/-- Sequence a_n defined as n^2 - 2an for n ∈ ℕ+ -/
def a_n (a : ℝ) (n : ℕ+) : ℝ := n.val^2 - 2*a*n.val

/-- Proposition: Given a_n = n^2 - 2an for n ∈ ℕ+, and a_n > a_4 for all n ≠ 4,
    the range of values for a is (7/2, 9/2) -/
theorem a_range_theorem (a : ℝ) : 
  (∀ (n : ℕ+), n ≠ 4 → a_n a n > a_n a 4) ↔ 
  (7/2 < a ∧ a < 9/2) :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l1530_153001


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l1530_153082

/-- Given a circle divided into 16 equal arcs, this theorem proves that 
    the sum of an inscribed angle over 3 arcs and an inscribed angle over 5 arcs is 90°. -/
theorem inscribed_angles_sum (circle : Real) (arcs : ℕ) (x y : Real) :
  arcs = 16 →
  x = 3 * (360 / (2 * arcs)) →
  y = 5 * (360 / (2 * arcs)) →
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l1530_153082


namespace NUMINAMATH_CALUDE_paint_mixture_weight_l1530_153067

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue_percent : ℝ
  red_percent : ℝ
  yellow_percent : ℝ
  weight : ℝ

/-- The problem setup for the paint mixture calculation -/
def paint_problem : Prop :=
  ∃ (sky_blue green brown : PaintMixture),
    -- Sky blue paint composition
    sky_blue.blue_percent = 0.1 ∧
    sky_blue.red_percent = 0.9 ∧
    sky_blue.yellow_percent = 0 ∧
    -- Green paint composition
    green.blue_percent = 0.7 ∧
    green.red_percent = 0 ∧
    green.yellow_percent = 0.3 ∧
    -- Brown paint composition
    brown.blue_percent = 0.4 ∧
    -- Red pigment weight in brown paint
    brown.red_percent * brown.weight = 4.5 ∧
    -- Total weight of brown paint
    brown.weight = sky_blue.weight + green.weight ∧
    -- Blue pigment balance
    sky_blue.blue_percent * sky_blue.weight + green.blue_percent * green.weight = 
      brown.blue_percent * brown.weight ∧
    -- Total weight of brown paint is 10 grams
    brown.weight = 10

/-- The main theorem stating that the paint problem implies a 10-gram brown paint -/
theorem paint_mixture_weight : paint_problem → ∃ (brown : PaintMixture), brown.weight = 10 := by
  sorry


end NUMINAMATH_CALUDE_paint_mixture_weight_l1530_153067


namespace NUMINAMATH_CALUDE_intersection_condition_max_area_rhombus_condition_l1530_153017

-- Define the lines and ellipse
def l₁ (k x : ℝ) : ℝ := k * x + 2
def l₂ (k x : ℝ) : ℝ := k * x - 2
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop := ∃ A B C D : ℝ × ℝ,
  ellipse A.1 A.2 ∧ (A.2 = l₁ k A.1 ∨ A.2 = l₂ k A.1) ∧
  ellipse B.1 B.2 ∧ (B.2 = l₁ k B.1 ∨ B.2 = l₂ k B.1) ∧
  ellipse C.1 C.2 ∧ (C.2 = l₁ k C.1 ∨ C.2 = l₂ k C.1) ∧
  ellipse D.1 D.2 ∧ (D.2 = l₁ k D.1 ∨ D.2 = l₂ k D.1)

-- Define the area of the quadrilateral
def area (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define the rhombus condition
def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

theorem intersection_condition (k : ℝ) :
  intersection_points k ↔ abs k > Real.sqrt 3 / 3 := sorry

theorem max_area {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, area A B C D ≤ 4 * Real.sqrt 3 := sorry

theorem rhombus_condition {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, is_rhombus A B C D → k = Real.sqrt 15 / 3 ∨ k = -Real.sqrt 15 / 3 := sorry

end NUMINAMATH_CALUDE_intersection_condition_max_area_rhombus_condition_l1530_153017


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l1530_153016

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2 + (a-b)^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l1530_153016


namespace NUMINAMATH_CALUDE_parabola_points_x_coordinate_l1530_153013

/-- The x-coordinate of points on the parabola y^2 = 12x with distance 8 from the focus -/
theorem parabola_points_x_coordinate (x y : ℝ) : 
  y^2 = 12*x →                             -- Point (x,y) is on the parabola
  (x - 3)^2 + y^2 = 64 →                   -- Distance from (x,y) to focus (3,0) is 8
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_x_coordinate_l1530_153013


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l1530_153099

theorem inverse_proposition_false (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), Continuous f ∧ 
  (∃ x ∈ Set.Ioo a b, f x = 0) ∧ 
  f a * f b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l1530_153099


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l1530_153072

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the 25-gingerbread trays -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the 20-gingerbread trays -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l1530_153072


namespace NUMINAMATH_CALUDE_triangular_plot_size_l1530_153074

/-- The size of a triangular plot of land in acres, given its dimensions on a map and conversion factors. -/
theorem triangular_plot_size (base height : ℝ) (scale_factor : ℝ) (acres_per_square_mile : ℝ) : 
  base = 8 → height = 12 → scale_factor = 1 → acres_per_square_mile = 320 →
  (1/2 * base * height) * scale_factor^2 * acres_per_square_mile = 15360 := by
  sorry

end NUMINAMATH_CALUDE_triangular_plot_size_l1530_153074


namespace NUMINAMATH_CALUDE_circle_center_l1530_153042

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) :
  (3 * x + 4 * y = 24 ∨ 3 * x + 4 * y = -6) →  -- Circle is tangent to these lines
  (3 * x - y = 0) →                           -- Center lies on this line
  (x = 3/5 ∧ y = 9/5) →                       -- Proposed center coordinates
  ∃ (r : ℝ), r > 0 ∧                          -- There exists a positive radius
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →  -- Points on the circle
      (3 * x' + 4 * y' = 24 ∨ 3 * x' + 4 * y' = -6))  -- Touch the given lines
  := by sorry


end NUMINAMATH_CALUDE_circle_center_l1530_153042


namespace NUMINAMATH_CALUDE_pauls_pencil_stock_l1530_153048

/-- Calculates the number of pencils in stock at the end of the week -/
def pencils_in_stock_end_of_week (
  daily_production : ℕ)
  (working_days : ℕ)
  (initial_stock : ℕ)
  (sold_pencils : ℕ) : ℕ :=
  daily_production * working_days + initial_stock - sold_pencils

/-- Proves that Paul has 230 pencils in stock at the end of the week -/
theorem pauls_pencil_stock : 
  pencils_in_stock_end_of_week 100 5 80 350 = 230 := by
  sorry

end NUMINAMATH_CALUDE_pauls_pencil_stock_l1530_153048


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l1530_153021

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 + 2*x, 5],
    ![4 - 2*x, 9]]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l1530_153021


namespace NUMINAMATH_CALUDE_system_of_equations_result_l1530_153098

theorem system_of_equations_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_result_l1530_153098


namespace NUMINAMATH_CALUDE_det_scale_l1530_153083

theorem det_scale (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 10 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 90 := by
sorry

end NUMINAMATH_CALUDE_det_scale_l1530_153083


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1530_153011

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) : 
  b₁ = 3 → 
  b₂ = b₁ * s → 
  b₃ = b₂ * s → 
  ∀ x : ℝ, 3 * b₂ + 7 * b₃ ≥ -18/7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1530_153011


namespace NUMINAMATH_CALUDE_product_of_roots_l1530_153000

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1530_153000


namespace NUMINAMATH_CALUDE_cube_difference_l1530_153069

theorem cube_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l1530_153069


namespace NUMINAMATH_CALUDE_sum_of_odd_symmetric_function_l1530_153071

-- Define an odd function with symmetry about x = 1/2
def is_odd_and_symmetric (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (1/2 + x) = f (1/2 - x))

-- Theorem statement
theorem sum_of_odd_symmetric_function (f : ℝ → ℝ) 
  (h : is_odd_and_symmetric f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_symmetric_function_l1530_153071


namespace NUMINAMATH_CALUDE_horner_method_properties_l1530_153091

def horner_polynomial (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_v2 (x : ℝ) : ℝ := (4 * x + 2) * x + 3.5

theorem horner_method_properties :
  let x : ℝ := 5
  (∃ (max_multiplications : ℕ), max_multiplications = 5 ∧
    ∀ (other_multiplications : ℕ),
      other_multiplications ≤ max_multiplications) ∧
  horner_v2 x = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_properties_l1530_153091


namespace NUMINAMATH_CALUDE_one_meeting_before_return_l1530_153086

/-- Represents a runner on a rectangular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Represents the rectangular track -/
def track_perimeter : ℝ := 140

/-- Calculates the number of meetings between two runners -/
def meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem one_meeting_before_return (runner1 runner2 : Runner) 
  (h1 : runner1.speed = 6)
  (h2 : runner2.speed = 10)
  (h3 : runner1.direction ≠ runner2.direction) : 
  meetings runner1 runner2 = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_before_return_l1530_153086


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l1530_153018

theorem other_solution_of_quadratic (x : ℚ) : 
  (65 * (6/5)^2 + 18 = 104 * (6/5) - 13) →
  (65 * x^2 + 18 = 104 * x - 13) →
  (x ≠ 6/5) →
  x = 5/13 := by
sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l1530_153018


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l1530_153066

theorem product_of_five_consecutive_not_square (n : ℕ) : 
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l1530_153066


namespace NUMINAMATH_CALUDE_photos_framed_by_jack_or_taken_by_octavia_or_sam_l1530_153027

/-- Represents a photographer in the exhibition -/
inductive Photographer
| Octavia
| Sam
| Alice
| Max

/-- Represents a framer in the exhibition -/
inductive Framer
| Jack
| Jane

/-- The number of photographs framed by each framer for each photographer -/
def framed_photos (f : Framer) (p : Photographer) : ℕ :=
  match f, p with
  | Framer.Jack, Photographer.Octavia => 24
  | Framer.Jack, Photographer.Sam => 12
  | Framer.Jack, Photographer.Alice => 8
  | Framer.Jack, Photographer.Max => 0
  | Framer.Jane, Photographer.Octavia => 0
  | Framer.Jane, Photographer.Sam => 10
  | Framer.Jane, Photographer.Alice => 6
  | Framer.Jane, Photographer.Max => 18

/-- The total number of photographs taken by each photographer -/
def total_photos (p : Photographer) : ℕ :=
  match p with
  | Photographer.Octavia => 36
  | Photographer.Sam => 20
  | Photographer.Alice => 14
  | Photographer.Max => 32

/-- Theorem stating the number of photographs either framed by Jack or taken by Octavia or Sam -/
theorem photos_framed_by_jack_or_taken_by_octavia_or_sam :
  (framed_photos Framer.Jack Photographer.Octavia +
   framed_photos Framer.Jack Photographer.Sam +
   framed_photos Framer.Jack Photographer.Alice) +
  (total_photos Photographer.Octavia +
   total_photos Photographer.Sam) = 100 := by
  sorry

end NUMINAMATH_CALUDE_photos_framed_by_jack_or_taken_by_octavia_or_sam_l1530_153027


namespace NUMINAMATH_CALUDE_problem_statement_l1530_153081

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (m : ℝ) : Set ℝ := {x | 1 - 2*m ≤ x ∧ x ≤ 2 + m}

theorem problem_statement :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ M → x ∈ N m) ↔ m ≥ 3) ∧
  (∀ m : ℝ, (M ⊂ N m ∧ M ≠ N m) ↔ m ≤ 3/2) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1530_153081


namespace NUMINAMATH_CALUDE_marshmallow_ratio_l1530_153039

theorem marshmallow_ratio (joe_marshmallows : ℕ) (dad_marshmallows : ℕ) : 
  dad_marshmallows = 21 →
  (joe_marshmallows / 2 + dad_marshmallows / 3 = 49) →
  (joe_marshmallows : ℚ) / dad_marshmallows = 4 := by
sorry

end NUMINAMATH_CALUDE_marshmallow_ratio_l1530_153039


namespace NUMINAMATH_CALUDE_total_water_poured_l1530_153050

/-- 
Given two bottles with capacities of 4 and 8 cups respectively, 
if they are filled to the same fraction of their capacity and 
5.333333333333333 cups of water are poured into the 8-cup bottle, 
then the total amount of water poured into both bottles is 8 cups.
-/
theorem total_water_poured (bottle1_capacity bottle2_capacity : ℝ) 
  (water_in_bottle2 : ℝ) : 
  bottle1_capacity = 4 →
  bottle2_capacity = 8 →
  water_in_bottle2 = 5.333333333333333 →
  (water_in_bottle2 / bottle2_capacity) * bottle1_capacity + water_in_bottle2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_water_poured_l1530_153050


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1530_153085

/-- Parabola y² = x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

/-- Point lies on x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) (r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem parabola_intersection_theorem 
  (O : ℝ × ℝ)  -- Origin
  (P S T : ℝ × ℝ)  -- Points on x-axis
  (A₁ B₁ A₂ B₂ : ℝ × ℝ)  -- Points on parabola
  (h_O : O = (0, 0))
  (h_P : on_x_axis P)
  (h_S : on_x_axis S)
  (h_T : on_x_axis T)
  (h_A₁ : parabola A₁)
  (h_B₁ : parabola B₁)
  (h_A₂ : parabola A₂)
  (h_B₂ : parabola B₂)
  (h_line₁ : line_through A₁ B₁ P)
  (h_line₂ : line_through A₂ B₂ P)
  (h_line₃ : line_through A₁ B₂ S)
  (h_line₄ : line_through A₂ B₁ T) :
  (S.1 - O.1) * (T.1 - O.1) = (P.1 - O.1)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1530_153085


namespace NUMINAMATH_CALUDE_jeff_donuts_days_l1530_153055

/-- The number of donuts Jeff makes per day -/
def donuts_per_day : ℕ := 10

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled : ℕ := 10

/-- The number of days Jeff makes donuts -/
def days_making_donuts : ℕ := 12

theorem jeff_donuts_days :
  days_making_donuts * (donuts_per_day - jeff_eats_per_day) - chris_eats_total =
  boxes_filled * donuts_per_box :=
by
  sorry

end NUMINAMATH_CALUDE_jeff_donuts_days_l1530_153055


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_1985_l1530_153020

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

def exponentSum (powers : List ℕ) : ℕ := powers.sum

theorem least_exponent_sum_for_1985 :
  ∃ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 1985 powers ∧
    ∀ (other_powers : List ℕ),
      sumOfDistinctPowersOfTwo 1985 other_powers →
      exponentSum powers ≤ exponentSum other_powers ∧
      exponentSum powers = 40 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_1985_l1530_153020


namespace NUMINAMATH_CALUDE_grid_coloring_probability_l1530_153043

/-- The number of squares in a row or column of the grid -/
def gridSize : ℕ := 4

/-- The total number of possible colorings for the grid -/
def totalColorings : ℕ := 2^(gridSize^2)

/-- The number of colorings with at least one 3-by-3 yellow square -/
def coloringsWithYellowSquare : ℕ := 510

/-- The probability of obtaining a grid without a 3-by-3 yellow square -/
def probabilityNoYellowSquare : ℚ := (totalColorings - coloringsWithYellowSquare) / totalColorings

theorem grid_coloring_probability :
  probabilityNoYellowSquare = 65026 / 65536 :=
sorry

end NUMINAMATH_CALUDE_grid_coloring_probability_l1530_153043


namespace NUMINAMATH_CALUDE_jessa_cupcakes_l1530_153025

/-- The number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := sorry

/-- The number of fourth-grade classes -/
def fourth_grade_classes : ℕ := 12

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 45

/-- The number of P.E. classes -/
def pe_classes : ℕ := 2

/-- The number of students in each P.E. class -/
def students_per_pe : ℕ := 90

/-- The number of afterschool clubs -/
def afterschool_clubs : ℕ := 4

/-- The number of students in each afterschool club -/
def students_per_afterschool : ℕ := 60

/-- Theorem stating that the total number of cupcakes Jessa needs to make is 960 -/
theorem jessa_cupcakes : total_cupcakes = 960 := by sorry

end NUMINAMATH_CALUDE_jessa_cupcakes_l1530_153025


namespace NUMINAMATH_CALUDE_f_and_g_properties_l1530_153007

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2
def g (x : ℝ) : ℝ := |x| + 1

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically decreasing on (-∞, 0)
def is_decreasing_neg (h : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → h x > h y

theorem f_and_g_properties :
  (is_even f ∧ is_decreasing_neg f) ∧
  (is_even g ∧ is_decreasing_neg g) := by sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l1530_153007


namespace NUMINAMATH_CALUDE_area_two_sectors_l1530_153065

/-- The area of a figure formed by two 45° sectors of a circle with radius 15 -/
theorem area_two_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45 * π / 180) :
  2 * (θ / (2 * π)) * π * r^2 = 56.25 * π :=
sorry

end NUMINAMATH_CALUDE_area_two_sectors_l1530_153065


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1530_153062

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) (h : ∀ n : ℕ, n > 0 → 
    geometric_sequence a₁ r n * geometric_sequence a₁ r (n + 1) = 16 ^ n) : 
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1530_153062


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1530_153046

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (3 - 4 * x) = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1530_153046


namespace NUMINAMATH_CALUDE_fraction_invariance_l1530_153084

theorem fraction_invariance (x y : ℝ) :
  2 * y^2 / (x - y)^2 = 2 * (3*y)^2 / ((3*x) - (3*y))^2 :=
sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1530_153084


namespace NUMINAMATH_CALUDE_car_speed_problem_l1530_153061

theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_time = 24)
  (h2 : initial_time = 4)
  (h3 : initial_speed = 35)
  (h4 : average_speed = 50) :
  let remaining_time := total_time - initial_time
  let total_distance := average_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 53 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1530_153061


namespace NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l1530_153068

-- Statement 1
theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by sorry

-- Statement 2
theorem ab_plus_one_neq_a_plus_b_iff (a b : ℝ) :
  a * b + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by sorry

-- Statement 3
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → a / b + 1 / (a * b) ≥ m) ∧
  a / b + 1 / (a * b) = 2 * Real.sqrt 2 + 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l1530_153068


namespace NUMINAMATH_CALUDE_certain_number_proof_l1530_153094

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = k * 1 + 10 ∧ 2037 = k * 1 + 7) → n = 2040 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1530_153094


namespace NUMINAMATH_CALUDE_find_f_2_l1530_153005

-- Define the real number a
variable (a : ℝ)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem find_f_2 
  (h_a_pos : a > 0)
  (h_a_neq_1 : a ≠ 1)
  (h_f_odd : ∀ x, f (-x) = -f x)
  (h_g_even : ∀ x, g (-x) = g x)
  (h_sum : ∀ x, f x + g x = a^x - a^(-x) + 2)
  (h_g_2 : g 2 = a) :
  f 2 = 15/4 := by
sorry

end NUMINAMATH_CALUDE_find_f_2_l1530_153005


namespace NUMINAMATH_CALUDE_second_platform_length_l1530_153078

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem second_platform_length 
  (train_length : ℝ) 
  (first_platform_length : ℝ) 
  (first_crossing_time : ℝ) 
  (second_crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_length = 200)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * ((train_length + first_platform_length) / first_crossing_time)) - train_length = 300 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l1530_153078


namespace NUMINAMATH_CALUDE_square_area_adjacent_vertices_l1530_153008

/-- The area of a square with adjacent vertices at (-2,3) and (4,3) is 36. -/
theorem square_area_adjacent_vertices : 
  let p1 : ℝ × ℝ := (-2, 3)
  let p2 : ℝ × ℝ := (4, 3)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_adjacent_vertices_l1530_153008


namespace NUMINAMATH_CALUDE_number_problem_l1530_153031

theorem number_problem (N : ℕ) (h1 : ∃ k : ℕ, N = 5 * k) (h2 : N / 5 = 25) :
  (N - 17) / 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l1530_153031


namespace NUMINAMATH_CALUDE_max_guarding_value_l1530_153024

/-- Represents the four possible directions a guard can look --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the 8x8 board --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a guard on the board --/
structure Guard :=
  (pos : Position)
  (dir : Direction)

/-- The type of a valid board configuration --/
def BoardConfiguration := Fin 8 → Fin 8 → Guard

/-- Checks if a guard at position (row, col) is guarded by another guard --/
def isGuardedBy (board : BoardConfiguration) (row col : Fin 8) (otherRow otherCol : Fin 8) : Prop :=
  sorry

/-- Counts the number of guards watching a specific position --/
def countGuardingGuards (board : BoardConfiguration) (row col : Fin 8) : Nat :=
  sorry

/-- Checks if all guards are guarded by at least k other guards --/
def allGuardsGuardedByAtLeastK (board : BoardConfiguration) (k : Nat) : Prop :=
  ∀ row col, countGuardingGuards board row col ≥ k

/-- The main theorem stating that 5 is the maximum value of k --/
theorem max_guarding_value :
  (∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 5) ∧
  (¬∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 6) :=
sorry

end NUMINAMATH_CALUDE_max_guarding_value_l1530_153024


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1530_153056

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and last term 17 -/
def arithmetic_sum : ℕ := 81

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The last term of the sequence -/
def aₙ : ℕ := 17

/-- The number of terms in the sequence -/
def n : ℕ := (aₙ - a₁) / d + 1

theorem arithmetic_sequence_sum :
  (n : ℕ) * (a₁ + aₙ) / 2 = arithmetic_sum :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1530_153056


namespace NUMINAMATH_CALUDE_tangent_equality_l1530_153006

-- Define the types for circles and points
variable (Circle Point : Type)

-- Define the predicates and functions
variable (outside : Circle → Circle → Prop)
variable (touches : Circle → Circle → Point → Point → Prop)
variable (passes_through : Circle → Point → Point → Prop)
variable (intersects_at : Circle → Circle → Point → Prop)
variable (tangent_at : Circle → Point → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_equality 
  (S₁ S₂ S₃ : Circle) 
  (A B C D K : Point) :
  outside S₁ S₂ →
  touches S₁ S₂ A B →
  passes_through S₃ A B →
  intersects_at S₃ S₁ C →
  intersects_at S₃ S₂ D →
  tangent_at S₁ C K →
  tangent_at S₂ D K →
  distance K C = distance K D :=
sorry

end NUMINAMATH_CALUDE_tangent_equality_l1530_153006


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1530_153079

/-- Given that M(3,8) is the midpoint of line segment AB and A(5,6) is one endpoint,
    prove that the product of the coordinates of point B is 10. -/
theorem midpoint_coordinate_product (A B M : ℝ × ℝ) : 
  A = (5, 6) → M = (3, 8) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 * B.2 = 10 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1530_153079


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1530_153032

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 1, P x) ↔ (∀ x > 1, ¬ P x) :=
by
  sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1530_153032


namespace NUMINAMATH_CALUDE_largest_b_value_l1530_153049

theorem largest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : 39 < b) 
  (h3 : (a : ℚ) / b - (30 : ℚ) / b = 0.4) : b ≤ 75 :=
sorry

end NUMINAMATH_CALUDE_largest_b_value_l1530_153049


namespace NUMINAMATH_CALUDE_circle_through_points_tangent_line_through_D_tangent_touches_circle_l1530_153040

-- Define the points
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)
def D : ℝ × ℝ := (-1, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 5

-- Define the tangent line equation
def tangent_line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Theorem for the circle equation
theorem circle_through_points :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 := by sorry

-- Theorem for the tangent line
theorem tangent_line_through_D :
  tangent_line_equation D.1 D.2 := by sorry

-- Theorem that the tangent line touches the circle at exactly one point
theorem tangent_touches_circle :
  ∃! (x y : ℝ), circle_equation x y ∧ tangent_line_equation x y := by sorry

end NUMINAMATH_CALUDE_circle_through_points_tangent_line_through_D_tangent_touches_circle_l1530_153040


namespace NUMINAMATH_CALUDE_weeklyRentIs1200_l1530_153063

/-- Calculates the weekly rent for a flower shop given the following conditions:
  * Utilities cost is 20% of rent
  * 2 employees per shift
  * Store open 16 hours a day for 5 days a week
  * Employee pay is $12.50 per hour
  * Total weekly expenses are $3440
-/
def calculateWeeklyRent (totalExpenses : ℚ) (employeePay : ℚ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (employeesPerShift : ℕ) : ℚ :=
  let totalHours : ℕ := hoursPerDay * daysPerWeek * employeesPerShift
  let weeklyWages : ℚ := employeePay * totalHours
  (totalExpenses - weeklyWages) / 1.2

/-- Proves that the weekly rent for the flower shop is $1200 -/
theorem weeklyRentIs1200 :
  calculateWeeklyRent 3440 12.5 16 5 2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_weeklyRentIs1200_l1530_153063


namespace NUMINAMATH_CALUDE_zeros_of_f_l1530_153009

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

-- Theorem statement
theorem zeros_of_f :
  (∃ x : ℝ, f x = 0) ↔ (x = -4 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1530_153009


namespace NUMINAMATH_CALUDE_wall_length_calculation_l1530_153014

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

theorem wall_length_calculation
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℕ)
  (h1 : brick.length = 80)
  (h2 : brick.width = 11.25)
  (h3 : brick.height = 6)
  (h4 : wall.height = 600)
  (h5 : wall.width = 22.5)
  (h6 : num_bricks = 2000)
  (h7 : num_bricks * brickVolume brick = wallVolume wall) :
  wall.length = 800 := by
  sorry

#check wall_length_calculation

end NUMINAMATH_CALUDE_wall_length_calculation_l1530_153014


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1530_153019

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ) :
  rate = 12.5 →
  interest = 100 →
  time = 2 →
  principal = (interest * 100) / (rate * time) →
  principal = 400 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1530_153019


namespace NUMINAMATH_CALUDE_tim_interest_rate_l1530_153023

/-- Tim's investment amount -/
def tim_investment : ℝ := 500

/-- Lana's investment amount -/
def lana_investment : ℝ := 1000

/-- Lana's annual interest rate -/
def lana_rate : ℝ := 0.05

/-- Number of years -/
def years : ℕ := 2

/-- Interest difference between Tim and Lana after 2 years -/
def interest_difference : ℝ := 2.5

/-- Calculate the compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Tim's annual interest rate -/
def tim_rate : ℝ := 0.1

theorem tim_interest_rate :
  compound_interest tim_investment tim_rate years =
  compound_interest lana_investment lana_rate years + interest_difference := by
  sorry

#check tim_interest_rate

end NUMINAMATH_CALUDE_tim_interest_rate_l1530_153023


namespace NUMINAMATH_CALUDE_initial_people_count_l1530_153080

/-- The number of people who left the table -/
def people_left : ℕ := 6

/-- The number of people who remained at the table -/
def people_remained : ℕ := 5

/-- The initial number of people at the table -/
def initial_people : ℕ := people_left + people_remained

theorem initial_people_count : initial_people = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_count_l1530_153080


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1530_153004

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x ≥ 1) ∧ (∃ x, x ≥ 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1530_153004


namespace NUMINAMATH_CALUDE_jed_gives_away_two_cards_l1530_153095

/-- Represents the number of cards Jed gives away every two weeks -/
def cards_given_away : ℕ := 2

/-- Represents the initial number of cards Jed has -/
def initial_cards : ℕ := 20

/-- Represents the number of cards Jed gets every week -/
def weekly_cards : ℕ := 6

/-- Represents the number of weeks that have passed -/
def weeks_passed : ℕ := 4

/-- Represents the total number of cards Jed has after 4 weeks -/
def final_cards : ℕ := 40

/-- Theorem stating that Jed gives away 2 cards every two weeks -/
theorem jed_gives_away_two_cards : 
  initial_cards + weekly_cards * weeks_passed - cards_given_away * (weeks_passed / 2) = final_cards :=
sorry

end NUMINAMATH_CALUDE_jed_gives_away_two_cards_l1530_153095


namespace NUMINAMATH_CALUDE_find_P_l1530_153010

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set M
def M (P : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + P = 0}

-- Define the complement of M in U
def complement_M (P : ℝ) : Set ℕ := U \ M P

-- Theorem statement
theorem find_P : ∃ P : ℝ, complement_M P = {2, 3} ∧ P = 4 := by sorry

end NUMINAMATH_CALUDE_find_P_l1530_153010


namespace NUMINAMATH_CALUDE_system_solution_l1530_153034

theorem system_solution (a x y : ℝ) : 
  (x / 2 - (2 * x - 3 * y) / 5 = a - 1) →
  (x + 3 = y / 3) →
  (x < 0 ∧ y > 0) ↔ (7/10 < a ∧ a < 64/10) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1530_153034


namespace NUMINAMATH_CALUDE_expression_simplification_l1530_153088

theorem expression_simplification :
  500 * 997 * 0.4995 * 100 = 997^2 * 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1530_153088


namespace NUMINAMATH_CALUDE_currency_and_unit_comparisons_l1530_153077

-- Define the conversion rates
def yuan_to_jiao : ℚ → ℚ := (· * 10)
def dm_to_cm : ℚ → ℚ := (· * 10)
def hectare_to_m2 : ℚ → ℚ := (· * 10000)
def km2_to_hectare : ℚ → ℚ := (· * 100)

-- Define the theorem
theorem currency_and_unit_comparisons :
  (7 > 5.70) ∧
  (70 > 7) ∧
  (80000 > 70000) ∧
  (1 = 1) ∧
  (34 * 6 * 2 = 34 * 12) ∧
  (3.9 = 3.9) := by
  sorry

end NUMINAMATH_CALUDE_currency_and_unit_comparisons_l1530_153077


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1530_153015

theorem inequality_equivalence (x : ℝ) : (x - 1) / (x - 3) ≥ 2 ↔ x ∈ Set.Ioo 3 5 ∪ {5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1530_153015


namespace NUMINAMATH_CALUDE_maya_has_largest_result_l1530_153075

def start_number : ℕ := 15

def sara_result : ℕ := (start_number ^ 2 - 3) + 4

def liam_result : ℕ := ((start_number - 2) ^ 2) + 4

def maya_result : ℕ := (start_number - 3 + 4) ^ 2

theorem maya_has_largest_result :
  maya_result > sara_result ∧ maya_result > liam_result := by
  sorry

end NUMINAMATH_CALUDE_maya_has_largest_result_l1530_153075


namespace NUMINAMATH_CALUDE_factorization_example_l1530_153029

theorem factorization_example (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l1530_153029


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1530_153089

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → a 3 = 4 → d = -2 → a 2 + a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1530_153089


namespace NUMINAMATH_CALUDE_vector_magnitude_l1530_153087

/-- Given vectors a and b in ℝ², where a = (1,3) and (a + b) ⟂ (a - b), prove that |b| = √10 -/
theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0 →
  Real.sqrt ((b.1)^2 + (b.2)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1530_153087


namespace NUMINAMATH_CALUDE_tangent_circle_center_height_l1530_153051

/-- A parabola with equation y = 2x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2 = 2 * p.1 ^ 2}

/-- A circle in the interior of the parabola -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : center.2 < 2 * center.1 ^ 2
  tangent_points : Set (ℝ × ℝ)
  is_tangent : tangent_points ⊆ Parabola
  on_circle : ∀ p ∈ tangent_points, (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2
  symmetry : ∀ p ∈ tangent_points, (-p.1, p.2) ∈ tangent_points

theorem tangent_circle_center_height (c : TangentCircle) :
  ∃ p ∈ c.tangent_points, c.center.2 - p.2 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_height_l1530_153051


namespace NUMINAMATH_CALUDE_hilt_garden_border_l1530_153033

/-- The number of rocks in Mrs. Hilt's completed garden border -/
def total_rocks : ℕ := 189

/-- The number of additional rocks Mrs. Hilt has yet to place -/
def remaining_rocks : ℕ := 64

/-- The number of rocks Mrs. Hilt has already placed -/
def placed_rocks : ℕ := total_rocks - remaining_rocks

theorem hilt_garden_border : placed_rocks = 125 := by
  sorry

end NUMINAMATH_CALUDE_hilt_garden_border_l1530_153033


namespace NUMINAMATH_CALUDE_total_amount_paid_l1530_153012

-- Define the structure for an item
structure Item where
  originalPrice : ℝ
  saleDiscount : ℝ
  membershipDiscount : Bool
  taxRate : ℝ

-- Define the function to calculate the final price of an item
def calculateFinalPrice (item : Item) : ℝ :=
  let priceAfterSale := item.originalPrice * (1 - item.saleDiscount)
  let priceAfterMembership := if item.membershipDiscount then priceAfterSale * 0.95 else priceAfterSale
  priceAfterMembership * (1 + item.taxRate)

-- Define the items
def vase : Item := { originalPrice := 250, saleDiscount := 0.25, membershipDiscount := true, taxRate := 0.12 }
def teacups : Item := { originalPrice := 350, saleDiscount := 0.30, membershipDiscount := false, taxRate := 0.08 }
def plate : Item := { originalPrice := 450, saleDiscount := 0, membershipDiscount := true, taxRate := 0.10 }
def ornament : Item := { originalPrice := 150, saleDiscount := 0.20, membershipDiscount := false, taxRate := 0.06 }

-- Theorem statement
theorem total_amount_paid : 
  calculateFinalPrice vase + calculateFinalPrice teacups + calculateFinalPrice plate + calculateFinalPrice ornament = 1061.55 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1530_153012


namespace NUMINAMATH_CALUDE_walk_a_thon_earnings_l1530_153076

theorem walk_a_thon_earnings (last_year_rate : ℚ) (last_year_total : ℚ) 
  (extra_miles : ℕ) (this_year_rate : ℚ) : 
  last_year_rate = 4 →
  last_year_total = 44 →
  extra_miles = 5 →
  (last_year_total / last_year_rate + extra_miles) * this_year_rate = last_year_total →
  this_year_rate = 11/4 := by
sorry

#eval (11 : ℚ) / 4  -- To show the decimal representation

end NUMINAMATH_CALUDE_walk_a_thon_earnings_l1530_153076


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1530_153059

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1530_153059


namespace NUMINAMATH_CALUDE_intersection_M_N_l1530_153047

def M : Set ℝ := {x : ℝ | |x + 1| < 3}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1530_153047


namespace NUMINAMATH_CALUDE_max_leftover_candy_l1530_153036

theorem max_leftover_candy (y : ℕ) (h : y > 11) : 
  ∃ (q r : ℕ), y = 11 * q + r ∧ r > 0 ∧ r ≤ 10 := by
sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l1530_153036


namespace NUMINAMATH_CALUDE_unique_solution_for_floor_equation_l1530_153022

theorem unique_solution_for_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_floor_equation_l1530_153022


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1530_153052

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1530_153052


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1530_153053

-- Define the circle and square
def circle_radius : ℝ := 4
def square_side : ℝ := 2

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨square_side, 0⟩
def B : Point := ⟨square_side, square_side⟩
def C : Point := ⟨0, square_side⟩

-- Define the extended points D and E
def D : Point := sorry
def E : Point := sorry

-- Define the shaded area
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area = (16 * π / 3) - 6 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1530_153053


namespace NUMINAMATH_CALUDE_pythagorean_theorem_isosceles_right_l1530_153041

/-- An isosceles right triangle with legs of unit length -/
structure IsoscelesRightTriangle where
  /-- The length of each leg is 1 -/
  leg : ℝ
  leg_eq_one : leg = 1

/-- The Pythagorean theorem for an isosceles right triangle -/
theorem pythagorean_theorem_isosceles_right (t : IsoscelesRightTriangle) :
  t.leg ^ 2 + t.leg ^ 2 = (Real.sqrt 2) ^ 2 := by
  sorry

#check pythagorean_theorem_isosceles_right

end NUMINAMATH_CALUDE_pythagorean_theorem_isosceles_right_l1530_153041


namespace NUMINAMATH_CALUDE_zero_points_property_l1530_153026

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem zero_points_property (a : ℝ) :
  (∃ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a (sqrt e) = 0 ∧ f a x₂ = 0) →
  a = sqrt e / (2 * e) ∧ ∀ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a x₂ = 0 → x₂ > e^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_zero_points_property_l1530_153026


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1530_153073

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1530_153073


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l1530_153037

theorem divisibility_by_eleven (a b : ℤ) : 
  (11 ∣ a^2 + b^2) → (11 ∣ a) ∧ (11 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l1530_153037


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1530_153045

theorem sum_of_two_numbers (a b : ℝ) : 
  a + b = 25 → a * b = 144 → |a - b| = 7 → a + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1530_153045


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l1530_153003

theorem five_sixths_of_twelve_fifths : (5 / 6 : ℚ) * (12 / 5 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l1530_153003


namespace NUMINAMATH_CALUDE_paper_area_proof_l1530_153044

/-- The side length of each square piece of paper in centimeters -/
def side_length : ℝ := 8.5

/-- The number of pieces of square paper -/
def num_pieces : ℝ := 3.2

/-- The total area when gluing the pieces together without any gap -/
def total_area : ℝ := 231.2

/-- Theorem stating that the total area of the glued pieces is 231.2 cm² -/
theorem paper_area_proof : 
  side_length * side_length * num_pieces = total_area := by
  sorry

end NUMINAMATH_CALUDE_paper_area_proof_l1530_153044


namespace NUMINAMATH_CALUDE_units_digit_product_l1530_153057

theorem units_digit_product (a b c : ℕ) : 
  a^2010 * b^1004 * c^1002 ≡ 0 [MOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l1530_153057


namespace NUMINAMATH_CALUDE_eliana_steps_l1530_153060

def day1_steps : ℕ := 200 + 300

def day2_steps (d1 : ℕ) : ℕ := d1 * d1

def day3_steps (d1 d2 : ℕ) : ℕ := d1 + d2 + 100

def total_steps (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem eliana_steps :
  let d1 := day1_steps
  let d2 := day2_steps d1
  let d3 := day3_steps d1 d2
  total_steps d1 d2 d3 = 501100 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l1530_153060


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1530_153054

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Theorem for part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Theorem for part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1530_153054


namespace NUMINAMATH_CALUDE_grape_juice_concentration_l1530_153030

/-- Given an initial mixture and added grape juice, calculate the final grape juice concentration -/
theorem grape_juice_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_juice : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_juice = 10) : 
  (initial_volume * initial_concentration + added_juice) / (initial_volume + added_juice) = 0.28 := by
sorry

end NUMINAMATH_CALUDE_grape_juice_concentration_l1530_153030


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1530_153028

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 15 = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) → 
  a + b^2 = 162 / 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1530_153028


namespace NUMINAMATH_CALUDE_apple_pyramid_count_l1530_153096

/-- Represents the number of apples in a layer of the pyramid -/
def layer_count (length width : ℕ) : ℕ := length * width

/-- Represents the pyramid-like stack of apples -/
def apple_pyramid : ℕ :=
  let base := layer_count 4 6
  let second := layer_count 3 5
  let third := layer_count 2 4
  let top := layer_count 2 3  -- double row on top
  base + second + third + top

/-- Theorem stating that the apple pyramid contains exactly 53 apples -/
theorem apple_pyramid_count : apple_pyramid = 53 := by
  sorry

end NUMINAMATH_CALUDE_apple_pyramid_count_l1530_153096


namespace NUMINAMATH_CALUDE_linear_function_slope_condition_l1530_153002

/-- Given two points on a linear function, if x-coordinate increases while y-coordinate decreases, then the slope is less than 2 -/
theorem linear_function_slope_condition (a x₁ y₁ x₂ y₂ : ℝ) : 
  y₁ = (a - 2) * x₁ + 1 →   -- Point A lies on the graph
  y₂ = (a - 2) * x₂ + 1 →   -- Point B lies on the graph
  (x₁ > x₂ → y₁ < y₂) →     -- When x₁ > x₂, y₁ < y₂
  a < 2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_slope_condition_l1530_153002
