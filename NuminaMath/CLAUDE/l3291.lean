import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_circle_radius_quarter_sector_l3291_329116

/-- The radius of an inscribed circle in a quarter circle sector -/
theorem inscribed_circle_radius_quarter_sector (r : ℝ) (h : r = 5) :
  let inscribed_radius := r * (Real.sqrt 2 - 1)
  inscribed_radius = 5 * Real.sqrt 2 - 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_quarter_sector_l3291_329116


namespace NUMINAMATH_CALUDE_sector_central_angle_l3291_329100

theorem sector_central_angle (s : Real) (r : Real) (θ : Real) 
  (h1 : s = π) 
  (h2 : r = 2) 
  (h3 : s = r * θ) : θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3291_329100


namespace NUMINAMATH_CALUDE_opposite_face_points_are_diametrically_opposite_l3291_329163

-- Define a cube
structure Cube where
  side_length : ℝ
  center : ℝ × ℝ × ℝ

-- Define a point on the surface of a cube
structure CubePoint where
  coordinates : ℝ × ℝ × ℝ
  cube : Cube
  on_surface : Bool

-- Define the concept of diametrically opposite points
def diametrically_opposite (p1 p2 : CubePoint) (c : Cube) : Prop :=
  ∃ (t : ℝ), 
    p1.coordinates = (1 - t) • c.center + t • p2.coordinates ∧ 
    0 ≤ t ∧ t ≤ 1

-- Define opposite faces
def opposite_faces (f1 f2 : CubePoint → Prop) (c : Cube) : Prop :=
  ∀ (p1 p2 : CubePoint), f1 p1 → f2 p2 → diametrically_opposite p1 p2 c

-- Theorem statement
theorem opposite_face_points_are_diametrically_opposite 
  (c : Cube) (p s : CubePoint) (f1 f2 : CubePoint → Prop) :
  opposite_faces f1 f2 c →
  f1 p →
  f2 s →
  diametrically_opposite p s c :=
sorry

end NUMINAMATH_CALUDE_opposite_face_points_are_diametrically_opposite_l3291_329163


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l3291_329120

theorem belt_and_road_population_scientific_notation :
  (4500000000 : ℝ) = 4.5 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l3291_329120


namespace NUMINAMATH_CALUDE_count_prime_differences_l3291_329174

def is_in_set (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k - 3 ∧ k ≥ 1

def is_prime_difference (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p - q

theorem count_prime_differences : 
  (∃! (s : Finset ℕ), (∀ n ∈ s, is_in_set n ∧ is_prime_difference n) ∧ s.card = 2) :=
sorry

end NUMINAMATH_CALUDE_count_prime_differences_l3291_329174


namespace NUMINAMATH_CALUDE_calculation_proof_l3291_329133

theorem calculation_proof : (-0.75) / 3 * (-2/5) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3291_329133


namespace NUMINAMATH_CALUDE_translation_theorem_l3291_329196

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C.x = 4 ∧ C.y = 7 ∧
  C = applyTranslation t A →
  applyTranslation t B = { x := 1, y := 2 } := by
  sorry


end NUMINAMATH_CALUDE_translation_theorem_l3291_329196


namespace NUMINAMATH_CALUDE_first_number_in_expression_l3291_329162

theorem first_number_in_expression (x : ℝ) : x * 0.8 + 0.1 * 0.5 = 0.29 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_expression_l3291_329162


namespace NUMINAMATH_CALUDE_total_flowers_l3291_329157

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) :
  num_pots * flowers_per_pot = 17408 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_l3291_329157


namespace NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l3291_329114

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l3291_329114


namespace NUMINAMATH_CALUDE_heavy_washes_count_l3291_329189

/-- Represents the number of gallons of water used for different wash types and conditions --/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  bleachRinseWater : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Calculates the number of heavy washes given the washing machine parameters --/
def calculateHeavyWashes (wm : WashingMachine) : ℕ :=
  (wm.totalWaterUsage - 
   (wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * wm.lightWashCount + 
    wm.bleachRinseWater * wm.bleachedLoadsCount)) / wm.heavyWashWater

/-- Theorem stating that the number of heavy washes is 2 given the specific conditions --/
theorem heavy_washes_count (wm : WashingMachine) 
  (h1 : wm.heavyWashWater = 20)
  (h2 : wm.regularWashWater = 10)
  (h3 : wm.lightWashWater = 2)
  (h4 : wm.bleachRinseWater = 2)
  (h5 : wm.regularWashCount = 3)
  (h6 : wm.lightWashCount = 1)
  (h7 : wm.bleachedLoadsCount = 2)
  (h8 : wm.totalWaterUsage = 76) :
  calculateHeavyWashes wm = 2 := by
  sorry

#eval calculateHeavyWashes {
  heavyWashWater := 20,
  regularWashWater := 10,
  lightWashWater := 2,
  bleachRinseWater := 2,
  regularWashCount := 3,
  lightWashCount := 1,
  bleachedLoadsCount := 2,
  totalWaterUsage := 76
}

end NUMINAMATH_CALUDE_heavy_washes_count_l3291_329189


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l3291_329112

/-- Represents an ellipse with the equation mx^2 + ny^2 = 1 -/
structure Ellipse (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Determines if an ellipse has foci on the y-axis -/
def hasFociOnYAxis (e : Ellipse m n) : Prop := sorry

/-- The main theorem stating that m > n > 0 is necessary and sufficient for
    the equation mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  (∃ e : Ellipse m n, hasFociOnYAxis e) ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l3291_329112


namespace NUMINAMATH_CALUDE_red_markers_count_l3291_329105

def total_markers : ℕ := 3343
def blue_markers : ℕ := 1028

theorem red_markers_count : total_markers - blue_markers = 2315 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l3291_329105


namespace NUMINAMATH_CALUDE_average_age_of_class_l3291_329195

theorem average_age_of_class (total_students : ℕ) 
  (group1_count group2_count : ℕ) 
  (group1_avg group2_avg last_student_age : ℝ) : 
  total_students = group1_count + group2_count + 1 →
  group1_count = 8 →
  group2_count = 6 →
  group1_avg = 14 →
  group2_avg = 16 →
  last_student_age = 17 →
  (group1_count * group1_avg + group2_count * group2_avg + last_student_age) / total_students = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_class_l3291_329195


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l3291_329186

theorem coin_flip_sequences (n k : ℕ) (hn : n = 10) (hk : k = 6) :
  (Nat.choose n k) = 210 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l3291_329186


namespace NUMINAMATH_CALUDE_extended_hexagon_area_l3291_329198

structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def extend_hexagon (h : Hexagon) : Hexagon := sorry

theorem extended_hexagon_area (h : Hexagon) 
  (side_lengths : Fin 6 → ℝ)
  (h_sides : ∀ i, dist (h.vertices i) (h.vertices ((i + 1) % 6)) = side_lengths i)
  (h_area : h.area = 30)
  (h_side_lengths : side_lengths = ![3, 4, 5, 6, 7, 8]) :
  (extend_hexagon h).area = 90 := by
  sorry

end NUMINAMATH_CALUDE_extended_hexagon_area_l3291_329198


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3291_329132

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 15) :
  ∃ (min : ℝ), min = 4/15 ∧ ∀ (a b : ℝ), 0 < a → 0 < b → a + b = 15 → min ≤ 1/a + 1/b :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3291_329132


namespace NUMINAMATH_CALUDE_johns_age_l3291_329182

/-- Given the ages of John, his dad, and his sister, prove that John is 25 years old. -/
theorem johns_age (john dad sister : ℕ) 
  (h1 : john + 30 = dad)
  (h2 : john + dad = 80)
  (h3 : sister = john - 5) :
  john = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3291_329182


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3291_329154

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 6) (hc : c = 7) :
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3291_329154


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l3291_329175

theorem n_times_n_plus_one_div_by_three (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  3 ∣ (n * (n + 1)) := by
sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l3291_329175


namespace NUMINAMATH_CALUDE_bounded_difference_l3291_329164

theorem bounded_difference (x y z : ℝ) :
  x - z < y ∧ x + z > y → -z < x - y ∧ x - y < z := by
  sorry

end NUMINAMATH_CALUDE_bounded_difference_l3291_329164


namespace NUMINAMATH_CALUDE_mobile_wire_left_l3291_329178

/-- The amount of wire left after making mobiles -/
def wire_left (total_wire : ℚ) (wire_per_mobile : ℚ) : ℚ :=
  total_wire - wire_per_mobile * ⌊total_wire / wire_per_mobile⌋

/-- Converts millimeters to centimeters -/
def mm_to_cm (mm : ℚ) : ℚ :=
  mm / 10

theorem mobile_wire_left : 
  mm_to_cm (wire_left 117.6 4) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_mobile_wire_left_l3291_329178


namespace NUMINAMATH_CALUDE_cut_cube_total_count_l3291_329156

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly two faces -/
  two_face_painted : ℕ

/-- Theorem stating that if a cube is cut such that 12 smaller cubes are painted on 2 faces,
    then the total number of smaller cubes is 27 -/
theorem cut_cube_total_count (c : CutCube) (h : c.two_face_painted = 12) : 
  c.edge_count ^ 3 = 27 := by
  sorry

#check cut_cube_total_count

end NUMINAMATH_CALUDE_cut_cube_total_count_l3291_329156


namespace NUMINAMATH_CALUDE_spike_morning_crickets_l3291_329169

/-- The number of crickets Spike hunts in the morning. -/
def morning_crickets : ℕ := 5

/-- The number of crickets Spike hunts in the afternoon and evening. -/
def afternoon_evening_crickets : ℕ := 3 * morning_crickets

/-- The total number of crickets Spike hunts per day. -/
def total_crickets : ℕ := 20

/-- Theorem stating that the number of crickets Spike hunts in the morning is 5. -/
theorem spike_morning_crickets :
  morning_crickets = 5 ∧
  afternoon_evening_crickets = 3 * morning_crickets ∧
  total_crickets = morning_crickets + afternoon_evening_crickets ∧
  total_crickets = 20 :=
by sorry

end NUMINAMATH_CALUDE_spike_morning_crickets_l3291_329169


namespace NUMINAMATH_CALUDE_tan_four_fifths_alpha_l3291_329129

theorem tan_four_fifths_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sqrt 3 * (Real.cos α) ^ 2 - Real.sin (2 * α) + 2 - Real.sqrt 3 = 0) : 
  Real.tan (4 / 5 * α) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_four_fifths_alpha_l3291_329129


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3291_329146

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5, 6}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3291_329146


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3291_329184

/-- Given a rectangle with length thrice its breadth and area 507 m², 
    prove that its perimeter is 104 m. -/
theorem rectangle_perimeter (breadth length : ℝ) : 
  length = 3 * breadth → 
  breadth * length = 507 → 
  2 * (length + breadth) = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3291_329184


namespace NUMINAMATH_CALUDE_largest_root_equation_l3291_329166

theorem largest_root_equation (a b c d : ℝ) 
  (h1 : a + d = 2022)
  (h2 : b + c = 2022)
  (h3 : a ≠ c) :
  ∃ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ 
    x = 1011 ∧
    ∀ y : ℝ, (y - a) * (y - b) = (y - c) * (y - d) → y ≤ 1011 :=
by sorry

end NUMINAMATH_CALUDE_largest_root_equation_l3291_329166


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l3291_329115

/-- Calculates the total number of penalty kicks in a soccer team drill -/
theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * (goalies - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l3291_329115


namespace NUMINAMATH_CALUDE_factor_expression_l3291_329130

theorem factor_expression (y : ℝ) : 75 * y + 45 = 15 * (5 * y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3291_329130


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l3291_329143

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l3291_329143


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3291_329199

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) + 1
  f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3291_329199


namespace NUMINAMATH_CALUDE_original_profit_margin_exists_l3291_329179

/-- Given a reduction in purchase price and an increase in profit margin,
    there exists a unique original profit margin. -/
theorem original_profit_margin_exists :
  ∃! x : ℝ, 
    0 ≤ x ∧ x ≤ 100 ∧
    (1 + (x + 8) / 100) * (1 - 0.064) = 1 + x / 100 :=
by sorry

end NUMINAMATH_CALUDE_original_profit_margin_exists_l3291_329179


namespace NUMINAMATH_CALUDE_total_marbles_l3291_329197

def marble_collection (jar1 jar2 jar3 : ℕ) : Prop :=
  (jar1 = 80) ∧
  (jar2 = 2 * jar1) ∧
  (jar3 = jar1 / 4) ∧
  (jar1 + jar2 + jar3 = 260)

theorem total_marbles :
  ∃ (jar1 jar2 jar3 : ℕ), marble_collection jar1 jar2 jar3 :=
sorry

end NUMINAMATH_CALUDE_total_marbles_l3291_329197


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3291_329131

/-- A structure representing a 3D geometric space with lines and planes. -/
structure GeometricSpace where
  Line : Type
  Plane : Type
  parallelLinePlane : Line → Plane → Prop
  perpendicularLinePlane : Line → Plane → Prop
  parallelPlanes : Plane → Plane → Prop
  perpendicularLines : Line → Line → Prop

/-- Theorem stating the relationship between parallel planes and perpendicular lines. -/
theorem perpendicular_lines_from_parallel_planes 
  (S : GeometricSpace) 
  (α β : S.Plane) 
  (m n : S.Line) :
  S.parallelPlanes α β →
  S.perpendicularLinePlane m α →
  S.parallelLinePlane n β →
  S.perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3291_329131


namespace NUMINAMATH_CALUDE_min_balls_for_three_colors_l3291_329139

theorem min_balls_for_three_colors (num_colors : Nat) (balls_per_color : Nat) 
  (h1 : num_colors = 4) (h2 : balls_per_color = 13) :
  (2 * balls_per_color + 1) = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_colors_l3291_329139


namespace NUMINAMATH_CALUDE_equation_solution_l3291_329106

theorem equation_solution : 
  ∀ x : ℝ, (3*x + 2)*(x + 3) = x + 3 ↔ x = -3 ∨ x = -1/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3291_329106


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3291_329191

/-- A circle with center (-1, 2) that is tangent to the x-axis has the equation (x + 1)^2 + (y - 2)^2 = 4 -/
theorem circle_tangent_to_x_axis :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 ↔ ((x + 1)^2 + (y - 2)^2 = r^2)) ∧
    (∀ (x : ℝ), ∃ (y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3291_329191


namespace NUMINAMATH_CALUDE_new_ratio_after_removing_clothing_l3291_329102

/-- Represents the ratio of books to clothes to electronics -/
structure Ratio :=
  (books : ℕ)
  (clothes : ℕ)
  (electronics : ℕ)

/-- Calculates the new ratio of books to clothes after removing some clothing -/
def newRatio (initial : Ratio) (electronicsWeight : ℕ) (clothingRemoved : ℕ) : Ratio :=
  sorry

/-- Theorem stating the new ratio after removing clothing -/
theorem new_ratio_after_removing_clothing 
  (initial : Ratio)
  (electronicsWeight : ℕ)
  (clothingRemoved : ℕ)
  (h1 : initial = ⟨7, 4, 3⟩)
  (h2 : electronicsWeight = 9)
  (h3 : clothingRemoved = 6) :
  (newRatio initial electronicsWeight clothingRemoved).books = 7 ∧
  (newRatio initial electronicsWeight clothingRemoved).clothes = 2 :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_removing_clothing_l3291_329102


namespace NUMINAMATH_CALUDE_action_figures_removed_l3291_329128

theorem action_figures_removed (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 15 → added = 2 → final = 10 → initial + added - final = 7 := by
sorry

end NUMINAMATH_CALUDE_action_figures_removed_l3291_329128


namespace NUMINAMATH_CALUDE_alien_abduction_percentage_l3291_329125

/-- The number of people initially abducted by the alien -/
def initial_abducted : ℕ := 200

/-- The number of people taken away after returning some -/
def taken_away : ℕ := 40

/-- The number of people left on Earth after returning some and taking away others -/
def left_on_earth : ℕ := 160

/-- The percentage of people returned by the alien -/
def percentage_returned : ℚ := (left_on_earth : ℚ) / (initial_abducted : ℚ) * 100

theorem alien_abduction_percentage :
  percentage_returned = 80 := by sorry

end NUMINAMATH_CALUDE_alien_abduction_percentage_l3291_329125


namespace NUMINAMATH_CALUDE_towers_count_l3291_329160

/-- Represents the number of cubes of each color -/
structure CubeSet where
  yellow : Nat
  purple : Nat
  orange : Nat

/-- Calculates the number of different towers that can be built -/
def countTowers (cubes : CubeSet) (towerHeight : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem towers_count (cubes : CubeSet) (h : cubes = { yellow := 3, purple := 3, orange := 2 }) :
  countTowers cubes 6 = 350 := by
  sorry

end NUMINAMATH_CALUDE_towers_count_l3291_329160


namespace NUMINAMATH_CALUDE_production_average_problem_l3291_329108

theorem production_average_problem (n : ℕ) : 
  (n * 50 + 105) / (n + 1) = 55 → n = 10 := by sorry

end NUMINAMATH_CALUDE_production_average_problem_l3291_329108


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3291_329144

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3291_329144


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3291_329107

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3291_329107


namespace NUMINAMATH_CALUDE_smallest_num_neighbors_correct_l3291_329194

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for two points to be considered neighbors -/
def max_angle : ℝ := 10

/-- Definition of the smallest number of pairs of neighbors function -/
def smallest_num_neighbors (n : ℕ) (θ : ℝ) : ℕ :=
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2)

/-- Theorem stating that the smallest number of pairs of neighbors for the given conditions is correct -/
theorem smallest_num_neighbors_correct :
  smallest_num_neighbors num_points max_angle =
  25 * (Nat.choose 57 2) + 10 * (Nat.choose 58 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_neighbors_correct_l3291_329194


namespace NUMINAMATH_CALUDE_fly_revolutions_at_midnight_l3291_329145

/-- Represents a clock hand --/
inductive ClockHand
| Second
| Minute
| Hour

/-- Represents the state of the fly on the clock --/
structure FlyState where
  currentHand : ClockHand
  revolutions : ℕ

/-- The number of revolutions each hand makes in 12 hours --/
def handRevolutions (hand : ClockHand) : ℕ :=
  match hand with
  | ClockHand.Second => 720
  | ClockHand.Minute => 12
  | ClockHand.Hour => 1

/-- The total number of revolutions made by all hands in 12 hours --/
def totalRevolutions : ℕ :=
  (handRevolutions ClockHand.Second) +
  (handRevolutions ClockHand.Minute) +
  (handRevolutions ClockHand.Hour)

/-- Theorem stating that the fly makes 245 revolutions by midnight --/
theorem fly_revolutions_at_midnight :
  ∃ (finalState : FlyState),
    finalState.currentHand = ClockHand.Second →
    (∀ t, t ∈ Set.Icc (0 : ℝ) 12 →
      ¬ (∃ (h1 h2 h3 : ClockHand), h1 ≠ h2 ∧ h2 ≠ h3 ∧ h1 ≠ h3 ∧
        handRevolutions h1 * t = handRevolutions h2 * t ∧
        handRevolutions h2 * t = handRevolutions h3 * t)) →
    finalState.revolutions = 245 :=
sorry

end NUMINAMATH_CALUDE_fly_revolutions_at_midnight_l3291_329145


namespace NUMINAMATH_CALUDE_constructible_prism_dimensions_l3291_329181

/-- Represents a brick with dimensions 1 × 2 × 4 -/
structure Brick :=
  (length : ℕ := 1)
  (width : ℕ := 2)
  (height : ℕ := 4)

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Predicate to check if a prism can be constructed from bricks -/
def can_construct (p : RectangularPrism) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c

/-- Theorem stating that any constructible prism has dimensions a × 2b × 4c -/
theorem constructible_prism_dimensions (p : RectangularPrism) :
  can_construct p ↔ ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c :=
sorry

end NUMINAMATH_CALUDE_constructible_prism_dimensions_l3291_329181


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l3291_329159

/-- The degree of the polynomial (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l3291_329159


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3291_329147

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f := fun x : ℝ => (3 * x^2 - 2 / x^3)^5
  ∃ c : ℝ, c = 1080 ∧ (∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3291_329147


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3291_329151

/-- Given two parallel vectors a = (-1, 4) and b = (x, 2), prove that x = -1/2 -/
theorem parallel_vectors_x_value (x : ℚ) :
  let a : ℚ × ℚ := (-1, 4)
  let b : ℚ × ℚ := (x, 2)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3291_329151


namespace NUMINAMATH_CALUDE_curve_and_tangent_properties_l3291_329183

/-- The function f(x) -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + n

/-- The tangent line of f at x = 0 -/
def tangent_line (x : ℝ) : ℝ := 4*x + 3

/-- The inequality condition for x > 1 -/
def inequality_condition (m n : ℝ) (k : ℤ) (x : ℝ) : Prop :=
  x > 1 → f m n (x + x * Real.log x) > f m n (↑k * (x - 1))

theorem curve_and_tangent_properties :
  ∃ (m n : ℝ) (k : ℤ),
    (∀ x, f m n x = tangent_line x) ∧
    (∀ x, inequality_condition m n k x) ∧
    m = 4 ∧ n = 3 ∧ k = 3 ∧
    (∀ k' : ℤ, (∀ x, inequality_condition m n k' x) → k' ≤ k) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_tangent_properties_l3291_329183


namespace NUMINAMATH_CALUDE_smallest_d_value_l3291_329173

theorem smallest_d_value : ∃ (d : ℝ), d ≥ 0 ∧ 
  (3 * Real.sqrt 5)^2 + (d + 3)^2 = (3 * d)^2 ∧
  (∀ (x : ℝ), x ≥ 0 ∧ (3 * Real.sqrt 5)^2 + (x + 3)^2 = (3 * x)^2 → d ≤ x) ∧
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l3291_329173


namespace NUMINAMATH_CALUDE_hannah_leah_study_difference_l3291_329190

theorem hannah_leah_study_difference (daily_differences : List Int) 
  (h1 : daily_differences = [15, -5, 25, -15, 35, 0, 20]) 
  (days_in_week : Nat) (h2 : days_in_week = 7) : 
  Int.floor ((daily_differences.sum : ℚ) / days_in_week) = 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_leah_study_difference_l3291_329190


namespace NUMINAMATH_CALUDE_smallest_cover_count_l3291_329127

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The area of a square -/
def Square.area (s : Square) : ℕ := s.side * s.side

/-- A function that checks if a square can be covered by a given number of rectangles -/
def can_cover (s : Square) (r : Rectangle) (n : ℕ) : Prop :=
  s.area = n * r.area

/-- The main theorem -/
theorem smallest_cover_count (r : Rectangle) (h1 : r.width = 3) (h2 : r.height = 4) :
  ∃ (s : Square), can_cover s r 9 ∧ ∀ (s' : Square) (n : ℕ), can_cover s' r n → n ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_cover_count_l3291_329127


namespace NUMINAMATH_CALUDE_student_count_l3291_329101

theorem student_count : ∃ S : ℕ, 
  (S / 3 : ℚ) + 10 = S - 6 ∧ S = 24 := by sorry

end NUMINAMATH_CALUDE_student_count_l3291_329101


namespace NUMINAMATH_CALUDE_final_board_number_l3291_329161

def board_numbers : List Nat :=
  (List.range 221).filter (fun n => (n + 3) % 4 = 3)

def sum_minus_two (a b : Nat) : Nat :=
  a + b - 2

def final_number (numbers : List Nat) : Nat :=
  numbers.foldl sum_minus_two (numbers.head!)

theorem final_board_number :
  final_number board_numbers = 6218 := by
  sorry

end NUMINAMATH_CALUDE_final_board_number_l3291_329161


namespace NUMINAMATH_CALUDE_exists_x0_f_less_than_g_l3291_329119

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin x ^ 2017

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2017 + 2017 ^ x

theorem exists_x0_f_less_than_g :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, f x < g x := by sorry

end NUMINAMATH_CALUDE_exists_x0_f_less_than_g_l3291_329119


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l3291_329149

theorem positive_solution_of_equation : ∃ x : ℝ, x > 0 ∧ 
  (1/2) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 35*x + 7) ∧
  x = (75 + Real.sqrt 5681) / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l3291_329149


namespace NUMINAMATH_CALUDE_cargo_arrival_time_l3291_329165

/-- Calculates the time between leaving the port and arriving at the warehouse -/
def timeBetweenPortAndWarehouse (navigationTime : ℕ) (customsTime : ℕ) (departureTime : ℕ) (expectedArrival : ℕ) : ℕ :=
  departureTime - (navigationTime + customsTime + expectedArrival)

/-- Theorem stating that the cargo arrives at the warehouse 1 day after leaving the port -/
theorem cargo_arrival_time :
  ∀ (navigationTime customsTime departureTime expectedArrival : ℕ),
    navigationTime = 21 →
    customsTime = 4 →
    departureTime = 30 →
    expectedArrival = 2 →
    timeBetweenPortAndWarehouse navigationTime customsTime departureTime expectedArrival = 1 := by
  sorry

#eval timeBetweenPortAndWarehouse 21 4 30 2

end NUMINAMATH_CALUDE_cargo_arrival_time_l3291_329165


namespace NUMINAMATH_CALUDE_extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l3291_329138

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define differentiability for f
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extreme point
def IsExtremePoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extreme_point_implies_zero_derivative
  (x₀ : ℝ) (h_extreme : IsExtremePoint f x₀) :
  deriv f x₀ = 0 :=
sorry

-- State that the converse is not always true
theorem zero_derivative_not_always_extreme_point :
  ¬ (∀ (g : ℝ → ℝ) (hg : Differentiable ℝ g) (x₀ : ℝ),
    deriv g x₀ = 0 → IsExtremePoint g x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l3291_329138


namespace NUMINAMATH_CALUDE_max_npk_l3291_329142

/-- Represents a single digit integer -/
def SingleDigit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit to a two-digit number with repeated digits -/
def toTwoDigit (m : SingleDigit) : ℕ := 11 * m

/-- Checks if a number is three digits -/
def isThreeDigits (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The result of multiplying a two-digit number by a single digit -/
def result (m k : SingleDigit) : ℕ := toTwoDigit m * k

theorem max_npk :
  ∀ m k : SingleDigit,
    m ≠ k →
    isThreeDigits (result m k) →
    ∀ m' k' : SingleDigit,
      m' ≠ k' →
      isThreeDigits (result m' k') →
      result m' k' ≤ 891 :=
sorry

end NUMINAMATH_CALUDE_max_npk_l3291_329142


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3291_329153

theorem power_mod_eleven : 3^21 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3291_329153


namespace NUMINAMATH_CALUDE_base_subtraction_l3291_329126

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their original bases
def num1 : List Nat := [3, 5, 4]  -- 354 in base 9
def num2 : List Nat := [2, 6, 1]  -- 261 in base 6

-- State the theorem
theorem base_subtraction :
  to_base_10 num1 9 - to_base_10 num2 6 = 183 := by sorry

end NUMINAMATH_CALUDE_base_subtraction_l3291_329126


namespace NUMINAMATH_CALUDE_candy_division_problem_l3291_329172

theorem candy_division_problem :
  ∃! x : ℕ, 120 ≤ x ∧ x ≤ 150 ∧ x % 5 = 2 ∧ x % 6 = 5 ∧ x = 137 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_problem_l3291_329172


namespace NUMINAMATH_CALUDE_cos_double_angle_with_tan_l3291_329110

theorem cos_double_angle_with_tan (α : Real) (h : Real.tan α = 1 / 2) : 
  Real.cos (2 * α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_with_tan_l3291_329110


namespace NUMINAMATH_CALUDE_cost_reduction_proof_l3291_329113

theorem cost_reduction_proof (total_reduction : ℝ) (years : ℕ) (annual_reduction : ℝ) : 
  total_reduction = 0.36 ∧ years = 2 → 
  (1 - annual_reduction) ^ years = 1 - total_reduction →
  annual_reduction = 0.2 := by
sorry

end NUMINAMATH_CALUDE_cost_reduction_proof_l3291_329113


namespace NUMINAMATH_CALUDE_inverse_negation_false_l3291_329148

theorem inverse_negation_false : 
  ¬(∀ x : ℝ, (x^2 = 1 ∧ x ≠ 1) → x^2 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_false_l3291_329148


namespace NUMINAMATH_CALUDE_jessica_rent_last_year_l3291_329158

/-- Calculates Jessica's monthly rent last year given the increase in expenses --/
theorem jessica_rent_last_year (food_cost_last_year car_insurance_last_year : ℕ)
  (rent_increase_percent food_increase_percent : ℚ)
  (car_insurance_multiplier : ℕ)
  (total_yearly_increase : ℕ) :
  food_cost_last_year = 200 →
  car_insurance_last_year = 100 →
  rent_increase_percent = 30 / 100 →
  food_increase_percent = 50 / 100 →
  car_insurance_multiplier = 3 →
  total_yearly_increase = 7200 →
  ∃ (rent_last_year : ℕ),
    rent_last_year = 1000 ∧
    12 * ((1 + rent_increase_percent) * rent_last_year - rent_last_year +
         (1 + food_increase_percent) * food_cost_last_year - food_cost_last_year +
         car_insurance_multiplier * car_insurance_last_year - car_insurance_last_year) =
    total_yearly_increase :=
by sorry

end NUMINAMATH_CALUDE_jessica_rent_last_year_l3291_329158


namespace NUMINAMATH_CALUDE_sum_range_l3291_329150

theorem sum_range : ∃ (x : ℚ), 10.5 < x ∧ x < 11 ∧ x = 2 + 1/8 + 3 + 1/3 + 5 + 1/18 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l3291_329150


namespace NUMINAMATH_CALUDE_rectangular_strip_dimensions_l3291_329180

theorem rectangular_strip_dimensions (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_strip_dimensions_l3291_329180


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3291_329152

/-- 
For a quadratic equation x^2 + bx + 9, the equation has at least one real root 
if and only if b belongs to the set (-∞, -6] ∪ [6, ∞)
-/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3291_329152


namespace NUMINAMATH_CALUDE_albert_joshua_difference_l3291_329170

-- Define the number of rocks each person collected
def joshua_rocks : ℕ := 80
def jose_rocks : ℕ := joshua_rocks - 14
def albert_rocks : ℕ := jose_rocks + 20

-- Theorem statement
theorem albert_joshua_difference :
  albert_rocks - joshua_rocks = 6 := by
sorry

end NUMINAMATH_CALUDE_albert_joshua_difference_l3291_329170


namespace NUMINAMATH_CALUDE_isabel_homework_problems_l3291_329122

/-- The total number of problems Isabel has to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
  (math_problems_per_page reading_problems_per_page : ℕ)
  (science_problems_per_page history_problems : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems

/-- Theorem stating that Isabel has to complete 61 problems in total -/
theorem isabel_homework_problems :
  total_problems 2 4 3 1 5 5 7 10 = 61 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problems_l3291_329122


namespace NUMINAMATH_CALUDE_unique_solution_sum_l3291_329136

theorem unique_solution_sum (x : ℝ) (a b c : ℕ+) : 
  x = Real.sqrt ((Real.sqrt 65) / 2 + 5 / 2) →
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  ∃! (a b c : ℕ+), x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l3291_329136


namespace NUMINAMATH_CALUDE_initial_persimmons_l3291_329117

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_persimmons_l3291_329117


namespace NUMINAMATH_CALUDE_range_of_a_l3291_329121

theorem range_of_a (x a : ℝ) : 
  x > 2 → a ≤ x + 2 / (x - 2) → ∃ s : ℝ, s = 2 + 2 * Real.sqrt 2 ∧ IsLUB {a | ∃ x > 2, a ≤ x + 2 / (x - 2)} s :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3291_329121


namespace NUMINAMATH_CALUDE_N_is_composite_l3291_329185

def N (y : ℕ) : ℚ := (y^125 - 1) / (3^22 - 1)

theorem N_is_composite : ∃ (y : ℕ) (a b : ℕ), a > 1 ∧ b > 1 ∧ N y = (a * b : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l3291_329185


namespace NUMINAMATH_CALUDE_translation_right_3_units_l3291_329109

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 1, y := 2 }
  let B : Point := translateRight A 3
  B = { x := 4, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_translation_right_3_units_l3291_329109


namespace NUMINAMATH_CALUDE_remainder_mod_12_l3291_329168

theorem remainder_mod_12 : (1234^567 + 89^1011) % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_12_l3291_329168


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l3291_329167

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) →  -- u is a two-digit positive integer
  (13 * u) % 100 = 52 →  -- when multiplied by 13, the last two digits are 52
  u = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l3291_329167


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l3291_329104

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (shift_hours : ℕ) (build_time : ℕ) (days : ℕ) : ℕ :=
  let chairs_per_shift := min 1 (shift_hours / build_time)
  chairs_per_shift * days

/-- Theorem stating that a worker who works 8-hour shifts and takes 5 hours to build 1 chair
    can build 10 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l3291_329104


namespace NUMINAMATH_CALUDE_height_difference_l3291_329171

/-- Proves that the difference between Ron's height and Dean's height is 8 feet -/
theorem height_difference (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ)
  (h1 : water_depth = 2 * dean_height)
  (h2 : ron_height = 14)
  (h3 : water_depth = 12) :
  ron_height - dean_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l3291_329171


namespace NUMINAMATH_CALUDE_derivative_at_one_l3291_329192

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x + x^3) :
  deriv f 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3291_329192


namespace NUMINAMATH_CALUDE_jogging_distance_three_weeks_l3291_329111

/-- Calculates the total miles jogged over a given number of weeks -/
def total_miles_jogged (miles_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  miles_per_day * days_per_week * num_weeks

/-- Theorem: A person jogging 5 miles per day on weekdays for three weeks covers 75 miles -/
theorem jogging_distance_three_weeks :
  total_miles_jogged 5 5 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_three_weeks_l3291_329111


namespace NUMINAMATH_CALUDE_chloe_first_round_score_l3291_329135

/-- Chloe's trivia game score calculation -/
theorem chloe_first_round_score (first_round : ℤ) 
  (h1 : first_round + 50 - 4 = 86) : first_round = 40 := by
  sorry

end NUMINAMATH_CALUDE_chloe_first_round_score_l3291_329135


namespace NUMINAMATH_CALUDE_plan_D_most_reasonable_l3291_329141

/-- Represents a survey plan for testing vision of junior high school students -/
inductive SurveyPlan
| A  : SurveyPlan  -- Test students in a certain middle school
| B  : SurveyPlan  -- Test all students in a certain district
| C  : SurveyPlan  -- Test all students in the entire city
| D  : SurveyPlan  -- Select 5 schools from each district and test their students

/-- Represents a city with districts and schools -/
structure City where
  numDistricts : Nat
  numSchoolsPerDistrict : Nat

/-- Determines if a survey plan is reasonable based on representativeness and practicality -/
def isReasonable (plan : SurveyPlan) (city : City) : Prop :=
  match plan with
  | SurveyPlan.D => city.numDistricts = 9 ∧ city.numSchoolsPerDistrict ≥ 5
  | _ => False

/-- Theorem stating that plan D is the most reasonable for a city with 9 districts -/
theorem plan_D_most_reasonable (city : City) :
  city.numDistricts = 9 → city.numSchoolsPerDistrict ≥ 5 → 
  ∀ (plan : SurveyPlan), isReasonable plan city → plan = SurveyPlan.D :=
by sorry

end NUMINAMATH_CALUDE_plan_D_most_reasonable_l3291_329141


namespace NUMINAMATH_CALUDE_largest_of_twenty_consecutive_even_integers_with_sum_3000_l3291_329140

/-- Represents a sequence of consecutive even integers -/
structure ConsecutiveEvenIntegers where
  start : ℤ
  count : ℕ
  is_even : Even start

/-- The sum of the sequence -/
def sum_sequence (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.count * (2 * seq.start + (seq.count - 1) * 2) / 2

/-- The largest integer in the sequence -/
def largest_integer (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.start + 2 * (seq.count - 1)

theorem largest_of_twenty_consecutive_even_integers_with_sum_3000 :
  ∀ seq : ConsecutiveEvenIntegers,
    seq.count = 20 →
    sum_sequence seq = 3000 →
    largest_integer seq = 169 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_twenty_consecutive_even_integers_with_sum_3000_l3291_329140


namespace NUMINAMATH_CALUDE_babysitting_age_ratio_l3291_329188

theorem babysitting_age_ratio : 
  ∀ (jane_start_age jane_current_age jane_stop_years_ago oldest_babysat_current_age : ℕ),
    jane_start_age = 16 →
    jane_current_age = 32 →
    jane_stop_years_ago = 10 →
    oldest_babysat_current_age = 24 →
    ∃ (child_age jane_age : ℕ),
      child_age = oldest_babysat_current_age - jane_stop_years_ago ∧
      jane_age = jane_current_age - jane_stop_years_ago ∧
      child_age * 11 = jane_age * 7 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_age_ratio_l3291_329188


namespace NUMINAMATH_CALUDE_set_A_equals_singleton_l3291_329177

-- Define the set A
def A : Set (ℕ × ℕ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.2 = 6 / (p.1 + 3)}

-- State the theorem
theorem set_A_equals_singleton : A = {(3, 1)} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_singleton_l3291_329177


namespace NUMINAMATH_CALUDE_sabrina_basil_leaves_l3291_329155

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := 6

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := 11

/-- Theorem stating the correct number of basil leaves Sabrina needs -/
theorem sabrina_basil_leaves :
  (basil = 2 * sage) ∧
  (sage = verbena - 5) ∧
  (basil + sage + verbena = 29) ∧
  (basil = 12) := by
  sorry

#check sabrina_basil_leaves

end NUMINAMATH_CALUDE_sabrina_basil_leaves_l3291_329155


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3291_329134

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define the open interval (2, 3)
def openInterval : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3291_329134


namespace NUMINAMATH_CALUDE_area_of_triangle_APQ_l3291_329137

/-- Two perpendicular lines intersecting at A(9,12) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : Bool
  intersect_at_A : Bool
  y_intercept_diff : ℝ

/-- The specific configuration of perpendicular lines for our problem -/
def problem_lines : PerpendicularLines where
  A := (9, 12)
  P := (0, 0)
  Q := (0, 6)
  perpendicular := true
  intersect_at_A := true
  y_intercept_diff := 6

/-- The area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle APQ is 27 -/
theorem area_of_triangle_APQ : 
  triangle_area problem_lines.A problem_lines.P problem_lines.Q = 27 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_APQ_l3291_329137


namespace NUMINAMATH_CALUDE_divisibility_cycle_l3291_329124

theorem divisibility_cycle (x y z : ℕ+) : 
  (∃ a : ℕ+, y + 1 = a * x) ∧ 
  (∃ b : ℕ+, z + 1 = b * y) ∧ 
  (∃ c : ℕ+, x + 1 = c * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 2) ∨
   (x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 3 ∧ y = 5 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_cycle_l3291_329124


namespace NUMINAMATH_CALUDE_amusement_park_payment_l3291_329176

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def change_received : ℕ := 74

def family_ticket_cost : ℕ := 
  (regular_ticket_cost - child_discount) * 2 + regular_ticket_cost * 2

theorem amusement_park_payment : 
  family_ticket_cost + change_received = 500 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_payment_l3291_329176


namespace NUMINAMATH_CALUDE_range_of_a_l3291_329123

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ (a ≤ 0 ∨ a ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3291_329123


namespace NUMINAMATH_CALUDE_greatest_three_digit_odd_non_divisible_l3291_329193

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_even_integers_up_to (n : ℕ) : ℕ :=
  let k := n / 2
  k * (k + 1)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem greatest_three_digit_odd_non_divisible :
  ∀ n : ℕ,
    is_three_digit n →
    n % 2 = 1 →
    ¬(factorial n % sum_of_even_integers_up_to n = 0) →
    n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_odd_non_divisible_l3291_329193


namespace NUMINAMATH_CALUDE_binomial_expansion_cube_problem_solution_l3291_329118

theorem binomial_expansion_cube (x : ℕ) :
  x^3 + 3*(x^2) + 3*x + 1 = (x + 1)^3 :=
by sorry

theorem problem_solution : 
  85^3 + 3*(85^2) + 3*85 + 1 = 636256 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_cube_problem_solution_l3291_329118


namespace NUMINAMATH_CALUDE_song_listens_after_three_months_l3291_329187

/-- Calculates the total listens for a song that doubles in popularity each month -/
def totalListens (initialListens : ℕ) (months : ℕ) : ℕ :=
  let doublingSequence := List.range months |>.map (fun i => initialListens * 2^(i + 1))
  initialListens + doublingSequence.sum

/-- Theorem: The total listens after 3 months of doubling is 900,000 given 60,000 initial listens -/
theorem song_listens_after_three_months :
  totalListens 60000 3 = 900000 := by
  sorry

end NUMINAMATH_CALUDE_song_listens_after_three_months_l3291_329187


namespace NUMINAMATH_CALUDE_seedling_purchase_solution_l3291_329103

/-- Represents the unit prices and maximum purchase of seedlings --/
structure SeedlingPurchase where
  price_a : ℝ  -- Unit price of type A seedlings
  price_b : ℝ  -- Unit price of type B seedlings
  max_a : ℕ    -- Maximum number of type A seedlings that can be purchased

/-- Theorem statement for the seedling purchase problem --/
theorem seedling_purchase_solution :
  ∃ (sp : SeedlingPurchase),
    -- Condition 1: 30 bundles of A and 10 bundles of B cost 380 yuan
    30 * sp.price_a + 10 * sp.price_b = 380 ∧
    -- Condition 2: 50 bundles of A and 30 bundles of B cost 740 yuan
    50 * sp.price_a + 30 * sp.price_b = 740 ∧
    -- Condition 3: Budget constraint with discount
    sp.price_a * 0.9 * sp.max_a + sp.price_b * 0.9 * (100 - sp.max_a) ≤ 828 ∧
    -- Solution 1: Unit prices
    sp.price_a = 10 ∧ sp.price_b = 8 ∧
    -- Solution 2: Maximum number of type A seedlings
    sp.max_a = 60 := by
  sorry

end NUMINAMATH_CALUDE_seedling_purchase_solution_l3291_329103
