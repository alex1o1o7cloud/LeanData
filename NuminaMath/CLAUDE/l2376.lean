import Mathlib

namespace NUMINAMATH_CALUDE_circle_tangency_l2376_237637

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  let c1 : ℝ × ℝ := (0, 0)
  let r1 : ℝ := Real.sqrt m
  let c2 : ℝ × ℝ := (-3, 4)
  let r2 : ℝ := 6
  internally_tangent c1 c2 r1 r2 → m = 1 ∨ m = 121 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_l2376_237637


namespace NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_four_l2376_237671

/-- An ellipse with foci at (5, 1 + √8) and (5, 1 - √8), tangent to y = 1 and x = 1 -/
structure SpecialEllipse where
  /-- First focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- Second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_y : True
  /-- The ellipse is tangent to the line x = 1 -/
  tangent_x : True
  /-- The first focus is at (5, 1 + √8) -/
  focus1_coord : focus1 = (5, 1 + Real.sqrt 8)
  /-- The second focus is at (5, 1 - √8) -/
  focus2_coord : focus2 = (5, 1 - Real.sqrt 8)

/-- The length of the major axis of the special ellipse is 4 -/
theorem major_axis_length (e : SpecialEllipse) : ℝ := 4

/-- The major axis length of the special ellipse is indeed 4 -/
theorem major_axis_length_is_four (e : SpecialEllipse) : 
  major_axis_length e = 4 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_four_l2376_237671


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2376_237629

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 6) (hc : c = 7) :
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2376_237629


namespace NUMINAMATH_CALUDE_exists_alpha_floor_minus_n_even_l2376_237692

theorem exists_alpha_floor_minus_n_even :
  ∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n > 0 → Even (⌊α * n⌋ - n) := by
  sorry

end NUMINAMATH_CALUDE_exists_alpha_floor_minus_n_even_l2376_237692


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2376_237681

theorem roots_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ x = α ∨ x = β) → 
  α^3 + 14*β + 5 = -43 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2376_237681


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l2376_237647

theorem square_sum_equals_five (x y : ℝ) (h1 : (x - y)^2 = 25) (h2 : x * y = -10) :
  x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l2376_237647


namespace NUMINAMATH_CALUDE_min_monochromatic_triangles_l2376_237676

/-- A coloring of edges in a complete graph on 2k vertices. -/
def Coloring (k : ℕ) := Fin (2*k) → Fin (2*k) → Bool

/-- The number of monochromatic triangles in a coloring. -/
def monochromaticTriangles (k : ℕ) (c : Coloring k) : ℕ := sorry

/-- The statement of the problem. -/
theorem min_monochromatic_triangles (k : ℕ) (h : k ≥ 3) :
  ∃ (c : Coloring k), monochromaticTriangles k c = k * (k - 1) * (k - 2) / 3 ∧
  ∀ (c' : Coloring k), monochromaticTriangles k c' ≥ k * (k - 1) * (k - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_monochromatic_triangles_l2376_237676


namespace NUMINAMATH_CALUDE_division_by_self_l2376_237602

theorem division_by_self (a : ℝ) (h : a ≠ 0) : 3 * a / a = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_by_self_l2376_237602


namespace NUMINAMATH_CALUDE_grade_study_sample_size_l2376_237697

/-- Represents a statistical study of student grades -/
structure GradeStudy where
  total_students : ℕ
  selected_cards : ℕ

/-- Defines the sample size of a grade study -/
def sample_size (study : GradeStudy) : ℕ := study.selected_cards

/-- Theorem: The sample size of a study with 2000 total students and 200 selected cards is 200 -/
theorem grade_study_sample_size :
  ∀ (study : GradeStudy),
    study.total_students = 2000 →
    study.selected_cards = 200 →
    sample_size study = 200 := by
  sorry

end NUMINAMATH_CALUDE_grade_study_sample_size_l2376_237697


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l2376_237641

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_sufficient_condition 
  (h_parallel : parallel α β) 
  (h_perp : perpendicular n β) : 
  perpendicular n α :=
sorry

-- State that the condition is not necessary
theorem perpendicular_not_necessary_condition :
  ¬(∀ (n : Line) (α β : Plane), 
    perpendicular n α → 
    (parallel α β ∧ perpendicular n β)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l2376_237641


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2376_237624

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2376_237624


namespace NUMINAMATH_CALUDE_arithmetic_progression_duality_l2376_237635

theorem arithmetic_progression_duality 
  (x y z k p n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hk : k > 0) (hp : p > 0) (hn : n > 0)
  (h_arith : ∃ (a d : ℝ), x = a + d * (k - 1) ∧ 
                          y = a + d * (p - 1) ∧ 
                          z = a + d * (n - 1)) :
  ∃ (a' d' : ℝ), 
    (k = a' + d' * (x - 1) ∧
     p = a' + d' * (y - 1) ∧
     n = a' + d' * (z - 1)) ∧
    (∃ (d : ℝ), d * d' = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_duality_l2376_237635


namespace NUMINAMATH_CALUDE_toms_speed_from_r_to_b_l2376_237668

/-- Represents the speed of a journey between two towns -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Represents Tom's entire trip -/
structure TripData where
  rb : Journey
  bc : Journey
  averageSpeed : ℝ

theorem toms_speed_from_r_to_b (trip : TripData) : trip.rb.speed = 60 :=
  by
  have h1 : trip.rb.distance = 2 * trip.bc.distance := by sorry
  have h2 : trip.averageSpeed = 36 := by sorry
  have h3 : trip.bc.speed = 20 := by sorry
  have h4 : trip.averageSpeed = (trip.rb.distance + trip.bc.distance) / 
    (trip.rb.distance / trip.rb.speed + trip.bc.distance / trip.bc.speed) := by sorry
  sorry


end NUMINAMATH_CALUDE_toms_speed_from_r_to_b_l2376_237668


namespace NUMINAMATH_CALUDE_book_purchase_l2376_237620

theorem book_purchase (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (h1 : paperback_cost = 18)
  (h2 : hardcover_cost = 28)
  (h3 : total_cost = 240) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧ 
    hardcover_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_l2376_237620


namespace NUMINAMATH_CALUDE_no_solutions_inequality_l2376_237638

theorem no_solutions_inequality : ¬∃ (n k : ℕ), n ≤ n! - k^n ∧ n! - k^n ≤ k * n := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_inequality_l2376_237638


namespace NUMINAMATH_CALUDE_second_planner_cheaper_at_34_l2376_237658

/-- Represents the pricing model of an event planner -/
structure PricingModel where
  flatFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (model : PricingModel) (people : ℕ) : ℕ :=
  model.flatFee + model.perPersonFee * people

/-- The pricing model of the first planner -/
def planner1 : PricingModel := { flatFee := 150, perPersonFee := 18 }

/-- The pricing model of the second planner -/
def planner2 : PricingModel := { flatFee := 250, perPersonFee := 15 }

/-- Theorem stating that 34 is the least number of people for which the second planner is cheaper -/
theorem second_planner_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (totalCost planner2 34 < totalCost planner1 34) :=
by sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_at_34_l2376_237658


namespace NUMINAMATH_CALUDE_apps_added_l2376_237617

theorem apps_added (initial_apps final_apps : ℕ) (h1 : initial_apps = 17) (h2 : final_apps = 18) :
  final_apps - initial_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_added_l2376_237617


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l2376_237604

/-- The number of televisions of type A -/
def typeA : ℕ := 3

/-- The number of televisions of type B -/
def typeB : ℕ := 3

/-- The number of televisions of type C -/
def typeC : ℕ := 4

/-- The total number of televisions -/
def totalTVs : ℕ := typeA + typeB + typeC

/-- The number of televisions to be selected -/
def selectCount : ℕ := 3

/-- Calculates the number of ways to select r items from n items -/
def combination (n r : ℕ) : ℕ :=
  Nat.choose n r

/-- The theorem to be proved -/
theorem tv_selection_theorem : 
  combination totalTVs selectCount - 
  (combination typeA selectCount + combination typeB selectCount + combination typeC selectCount) = 114 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l2376_237604


namespace NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l2376_237684

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : Real) (train_speed : Real) 
  (jogger_lead : Real) (train_length : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := jogger_lead + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is 24 seconds -/
theorem train_passes_jogger_time :
  train_passes_jogger 9 45 120 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l2376_237684


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2376_237607

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5, 6}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2376_237607


namespace NUMINAMATH_CALUDE_items_washed_is_500_l2376_237657

/-- Calculates the total number of items washed given the number of loads, towels per load, and shirts per load. -/
def total_items_washed (loads : ℕ) (towels_per_load : ℕ) (shirts_per_load : ℕ) : ℕ :=
  loads * (towels_per_load + shirts_per_load)

/-- Proves that the total number of items washed is 500 given the specific conditions. -/
theorem items_washed_is_500 :
  total_items_washed 20 15 10 = 500 := by
  sorry

end NUMINAMATH_CALUDE_items_washed_is_500_l2376_237657


namespace NUMINAMATH_CALUDE_water_formed_moles_l2376_237619

/-- Represents a chemical compound -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def naoh : Compound := ⟨"NaOH", 2⟩
def h2so4 : Compound := ⟨"H2SO4", 2⟩

def balanced_reaction : Reaction := {
  reactants := [⟨"NaOH", 2⟩, ⟨"H2SO4", 1⟩],
  products := [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]
}

/-- Calculates the moles of a product formed in a reaction -/
def moles_formed (reaction : Reaction) (product : Compound) (limiting_reactant : Compound) : ℚ :=
  sorry

theorem water_formed_moles :
  moles_formed balanced_reaction ⟨"H2O", 0⟩ naoh = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_moles_l2376_237619


namespace NUMINAMATH_CALUDE_square_difference_identity_l2376_237622

theorem square_difference_identity : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2376_237622


namespace NUMINAMATH_CALUDE_point_coordinates_l2376_237627

/-- A point in the two-dimensional plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Coordinates of a point in the fourth quadrant with given distances to axes -/
theorem point_coordinates (p : Point) 
  (h1 : fourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2376_237627


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2376_237609

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 7 * p - 6 = 0) → 
  (3 * q^2 + 7 * q - 6 = 0) → 
  (p - 2) * (q - 2) = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2376_237609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2376_237664

/-- An arithmetic sequence is defined by its first term, common difference, and last term. -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ
  last : ℤ

/-- The number of terms in an arithmetic sequence. -/
def numTerms (seq : ArithmeticSequence) : ℤ :=
  (seq.last - seq.first) / seq.diff + 1

/-- Theorem: The arithmetic sequence with first term 13, common difference 3, and last term 73 has exactly 21 terms. -/
theorem arithmetic_sequence_terms : 
  let seq := ArithmeticSequence.mk 13 3 73
  numTerms seq = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2376_237664


namespace NUMINAMATH_CALUDE_complex_equality_implies_power_l2376_237605

/-- Given complex numbers z₁ and z₂, where z₁ = -1 + 3i and z₂ = a + bi³,
    if z₁ = z₂, then b^a = -1/3 -/
theorem complex_equality_implies_power (a b : ℝ) :
  let z₁ : ℂ := -1 + 3 * Complex.I
  let z₂ : ℂ := a + b * Complex.I^3
  z₁ = z₂ → b^a = -1/3 := by sorry

end NUMINAMATH_CALUDE_complex_equality_implies_power_l2376_237605


namespace NUMINAMATH_CALUDE_school_girls_count_l2376_237659

theorem school_girls_count (total_pupils : ℕ) (girl_boy_difference : ℕ) :
  total_pupils = 1455 →
  girl_boy_difference = 281 →
  ∃ (boys girls : ℕ),
    boys + girls = total_pupils ∧
    girls = boys + girl_boy_difference ∧
    girls = 868 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l2376_237659


namespace NUMINAMATH_CALUDE_even_tower_for_odd_walls_l2376_237688

/-- A standard die has opposite faces summing to 7 -/
structure StandardDie where
  faces : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- A tower of dice -/
def DiceTower (n : ℕ) := Fin n → StandardDie

/-- The sum of visible dots on a vertical wall of the tower -/
def wall_sum (tower : DiceTower n) (wall : Fin 4) : ℕ := sorry

theorem even_tower_for_odd_walls (n : ℕ) (tower : DiceTower n) :
  (∀ wall : Fin 4, Odd (wall_sum tower wall)) → Even n := by sorry

end NUMINAMATH_CALUDE_even_tower_for_odd_walls_l2376_237688


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l2376_237623

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 5) (hm : m = 4) :
  (n.choose m) * (m.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l2376_237623


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2376_237680

theorem quadratic_root_problem (a : ℝ) : 
  ((a - 1) * 1^2 - a * 1 + a^2 = 0) → (a ≠ 1) → (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2376_237680


namespace NUMINAMATH_CALUDE_square_rectangle_triangle_relation_l2376_237650

/-- Square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Rectangle with width and height -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Right triangle with base and height -/
structure RightTriangle :=
  (base : ℝ)
  (height : ℝ)

/-- The main theorem -/
theorem square_rectangle_triangle_relation 
  (ABCD : Square)
  (JKHG : Rectangle)
  (EBC : RightTriangle)
  (h1 : JKHG.width = ABCD.side)
  (h2 : EBC.base = ABCD.side)
  (h3 : JKHG.height = EBC.height)
  (h4 : JKHG.width * JKHG.height = 2 * (EBC.base * EBC.height / 2)) :
  EBC.height = 1 := by
  sorry


end NUMINAMATH_CALUDE_square_rectangle_triangle_relation_l2376_237650


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l2376_237660

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l2376_237660


namespace NUMINAMATH_CALUDE_distance_sum_to_axes_l2376_237645

/-- The sum of distances from point P(-1, -2) to x-axis and y-axis is 3 -/
theorem distance_sum_to_axes : 
  let P : ℝ × ℝ := (-1, -2)
  abs P.2 + abs P.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_to_axes_l2376_237645


namespace NUMINAMATH_CALUDE_chameleon_color_change_l2376_237678

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 ∧ blue_factor = 5 ∧ red_factor = 3 →
  ∃ (initial_blue initial_red changed : ℕ),
    initial_blue + initial_red = total ∧
    changed = initial_blue - (initial_blue / blue_factor) ∧
    initial_red + changed = (initial_red * red_factor) ∧
    changed = 80 := by
  sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l2376_237678


namespace NUMINAMATH_CALUDE_equation_solution_l2376_237687

theorem equation_solution : ∃ x : ℝ, 90 + 5 * x / (180 / 3) = 91 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2376_237687


namespace NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l2376_237603

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  opposite_directions a b → x = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l2376_237603


namespace NUMINAMATH_CALUDE_wendy_bought_four_chairs_l2376_237677

def furniture_problem (chairs : ℕ) : Prop :=
  let tables : ℕ := 4
  let time_per_piece : ℕ := 6
  let total_time : ℕ := 48
  (chairs + tables) * time_per_piece = total_time

theorem wendy_bought_four_chairs :
  ∃ (chairs : ℕ), furniture_problem chairs ∧ chairs = 4 :=
sorry

end NUMINAMATH_CALUDE_wendy_bought_four_chairs_l2376_237677


namespace NUMINAMATH_CALUDE_company_average_salary_l2376_237670

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) /
  (num_managers + num_associates : ℚ) = 40000 :=
by sorry

end NUMINAMATH_CALUDE_company_average_salary_l2376_237670


namespace NUMINAMATH_CALUDE_rand_code_is_1236_l2376_237690

/-- A coding system that assigns numerical codes to words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- The code for a given word in the coding system -/
def word_code (system : CodeSystem) (word : String) : Nat :=
  sorry

/-- Our specific coding system -/
def our_system : CodeSystem :=
  { range_code := 12345, random_code := 123678 }

theorem rand_code_is_1236 :
  word_code our_system "rand" = 1236 := by
  sorry

end NUMINAMATH_CALUDE_rand_code_is_1236_l2376_237690


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2376_237663

theorem smaller_number_proof (x y : ℕ+) : 
  (x * y : ℕ) = 323 → 
  (x : ℕ) = (y : ℕ) + 2 → 
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2376_237663


namespace NUMINAMATH_CALUDE_video_game_lives_l2376_237643

theorem video_game_lives (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) :
  initial_players = 11 →
  quitting_players = 5 →
  total_lives = 30 →
  (total_lives / (initial_players - quitting_players) = 5) :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l2376_237643


namespace NUMINAMATH_CALUDE_percentage_of_women_l2376_237652

theorem percentage_of_women (initial_workers : ℕ) (initial_men_fraction : ℚ) 
  (new_hires : ℕ) : 
  initial_workers = 90 → 
  initial_men_fraction = 2/3 → 
  new_hires = 10 → 
  let total_workers := initial_workers + new_hires
  let initial_women := initial_workers * (1 - initial_men_fraction)
  let total_women := initial_women + new_hires
  (total_women / total_workers : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_l2376_237652


namespace NUMINAMATH_CALUDE_shopping_money_l2376_237636

/-- Proves that if a person spends $20 and is left with $4 more than half of their original amount, then their original amount was $48. -/
theorem shopping_money (original : ℕ) : 
  (original / 2 + 4 = original - 20) → original = 48 :=
by sorry

end NUMINAMATH_CALUDE_shopping_money_l2376_237636


namespace NUMINAMATH_CALUDE_raspberry_carton_is_eight_ounces_l2376_237682

/-- Represents the cost and size of fruit cartons, and the amount needed for muffins --/
structure FruitData where
  blueberry_cost : ℚ
  blueberry_size : ℚ
  raspberry_cost : ℚ
  batches : ℕ
  fruit_per_batch : ℚ
  savings : ℚ

/-- Calculates the size of a raspberry carton based on the given data --/
def raspberry_carton_size (data : FruitData) : ℚ :=
  sorry

/-- Theorem stating that the raspberry carton size is 8 ounces --/
theorem raspberry_carton_is_eight_ounces (data : FruitData)
    (h1 : data.blueberry_cost = 5)
    (h2 : data.blueberry_size = 6)
    (h3 : data.raspberry_cost = 3)
    (h4 : data.batches = 4)
    (h5 : data.fruit_per_batch = 12)
    (h6 : data.savings = 22) :
    raspberry_carton_size data = 8 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_carton_is_eight_ounces_l2376_237682


namespace NUMINAMATH_CALUDE_count_threes_to_1000_l2376_237689

/-- Count of digit 3 appearances when listing integers from 1 to n -/
def count_threes (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the count of digit 3 appearances from 1 to 1000 is 300 -/
theorem count_threes_to_1000 : count_threes 1000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_count_threes_to_1000_l2376_237689


namespace NUMINAMATH_CALUDE_min_lines_for_100_squares_l2376_237631

/-- The number of squares formed by n lines when n is odd -/
def squares_odd (n : ℕ) : ℕ := ((n - 3) * (n - 1) * (n + 1)) / 24

/-- The number of squares formed by n lines when n is even -/
def squares_even (n : ℕ) : ℕ := ((n - 2) * n * (n - 1)) / 24

/-- The maximum number of squares that can be formed by n lines -/
def max_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then squares_even n else squares_odd n

/-- Predicate indicating whether it's possible to form exactly k squares with n lines -/
def can_form_squares (n k : ℕ) : Prop :=
  k ≤ max_squares n ∧ k > max_squares (n - 1)

theorem min_lines_for_100_squares :
  ∃ n : ℕ, can_form_squares n 100 ∧ ∀ m : ℕ, m < n → ¬can_form_squares m 100 :=
sorry

end NUMINAMATH_CALUDE_min_lines_for_100_squares_l2376_237631


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2376_237616

/-- Given a quadratic function f(x) = x^2 - 26x + 129, 
    prove that when written in the form (x+d)^2 + e, d + e = -53 -/
theorem quadratic_form_sum (x : ℝ) : 
  ∃ (d e : ℝ), (∀ x, x^2 - 26*x + 129 = (x+d)^2 + e) ∧ (d + e = -53) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2376_237616


namespace NUMINAMATH_CALUDE_ratio_of_sums_and_differences_l2376_237685

theorem ratio_of_sums_and_differences (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
  (x + 1 / x) / (x - 1 / x) = Real.sqrt 7 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_and_differences_l2376_237685


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2376_237649

theorem cube_plus_reciprocal_cube (r : ℝ) (hr : r ≠ 0) 
  (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2376_237649


namespace NUMINAMATH_CALUDE_unique_zero_of_increasing_cubic_l2376_237644

/-- Given an increasing function f(x) = x^3 + bx + c on [-1, 1] with f(1/2) * f(-1/2) < 0,
    prove that f has exactly one zero in [-1, 1]. -/
theorem unique_zero_of_increasing_cubic {b c : ℝ} :
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x + c
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x < f y) →
  f (1/2) * f (-1/2) < 0 →
  ∃! x, x ∈ [-1, 1] ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_of_increasing_cubic_l2376_237644


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2376_237673

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = -Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2376_237673


namespace NUMINAMATH_CALUDE_age_ratio_in_four_years_l2376_237691

/-- Represents the ages of Paul and Kim -/
structure Ages where
  paul : ℕ
  kim : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.paul - 8 = 2 * (a.kim - 8)) ∧ 
  (a.paul - 14 = 3 * (a.kim - 14))

/-- The theorem to prove -/
theorem age_ratio_in_four_years (a : Ages) :
  age_conditions a →
  ∃ (x : ℕ), x = 4 ∧ 
    (a.paul + x) * 2 = (a.kim + x) * 3 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_four_years_l2376_237691


namespace NUMINAMATH_CALUDE_min_value_product_equality_condition_l2376_237653

theorem min_value_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) 
  (h_eq : x₁ = π/4 ∧ x₂ = π/4 ∧ x₃ = π/4 ∧ x₄ = π/4) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) = 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_equality_condition_l2376_237653


namespace NUMINAMATH_CALUDE_total_cost_is_1100_l2376_237632

def piano_cost : ℝ := 500
def num_lessons : ℕ := 20
def lesson_cost : ℝ := 40
def discount_rate : ℝ := 0.25

def total_cost : ℝ := 
  piano_cost + (1 - discount_rate) * (num_lessons : ℝ) * lesson_cost

theorem total_cost_is_1100 : total_cost = 1100 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_1100_l2376_237632


namespace NUMINAMATH_CALUDE_domino_game_strategy_l2376_237611

/-- Represents the players in the game -/
inductive Player
| Alice
| Bob

/-- Represents the outcome of the game -/
inductive Outcome
| Win
| Lose

/-- Represents a grid in the domino game -/
structure Grid :=
  (n : ℕ)
  (m : ℕ)

/-- Determines if a player has a winning strategy on a given grid -/
def has_winning_strategy (player : Player) (grid : Grid) : Prop :=
  match player with
  | Player.Alice => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 1) ∨
      (grid.n % 2 = 1 ∧ grid.m % 2 = 0)
  | Player.Bob => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 0)

/-- Theorem stating the winning strategies for the domino game -/
theorem domino_game_strategy (grid : Grid) :
  (grid.n % 2 = 0 ∧ grid.m % 2 = 0 → has_winning_strategy Player.Bob grid) ∧
  (grid.n % 2 = 0 ∧ grid.m % 2 = 1 → has_winning_strategy Player.Alice grid) :=
sorry

end NUMINAMATH_CALUDE_domino_game_strategy_l2376_237611


namespace NUMINAMATH_CALUDE_odd_totient_power_of_two_l2376_237662

theorem odd_totient_power_of_two (n : ℕ) 
  (h_odd : Odd n)
  (h_phi_n : ∃ k : ℕ, Nat.totient n = 2^k)
  (h_phi_n_plus_one : ∃ m : ℕ, Nat.totient (n+1) = 2^m) :
  (∃ p : ℕ, n+1 = 2^p) ∨ n = 5 := by
sorry

end NUMINAMATH_CALUDE_odd_totient_power_of_two_l2376_237662


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2376_237686

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2376_237686


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l2376_237675

def is_divisible_by_18 (n : ℕ) : Prop := n % 18 = 0

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 520 + x

theorem unique_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_18 (four_digit_number x) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l2376_237675


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2376_237601

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem sufficient_but_not_necessary
  (a₁ q : ℝ) :
  (∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ↔
  (a₁ > 0 ∧ q > 1 ∨
   ∃ a₁' q', (a₁' ≤ 0 ∨ q' ≤ 1) ∧
   ∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2376_237601


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2376_237608

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f := fun x : ℝ => (3 * x^2 - 2 / x^3)^5
  ∃ c : ℝ, c = 1080 ∧ (∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2376_237608


namespace NUMINAMATH_CALUDE_cos_sin_identity_l2376_237698

theorem cos_sin_identity (a : Real) (h : Real.cos (π/6 - a) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + a) - Real.sin (a - π/6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l2376_237698


namespace NUMINAMATH_CALUDE_sqrt_15_range_l2376_237614

theorem sqrt_15_range : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_range_l2376_237614


namespace NUMINAMATH_CALUDE_total_books_two_months_l2376_237648

def books_last_month : ℕ := 4

def books_this_month (n : ℕ) : ℕ := 2 * n

theorem total_books_two_months : 
  books_last_month + books_this_month books_last_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_books_two_months_l2376_237648


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2376_237695

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 → 
  Nat.gcd A B = 30 → 
  A = 231 → 
  B = 300 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2376_237695


namespace NUMINAMATH_CALUDE_function_shift_l2376_237606

theorem function_shift (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 + 4*x - 5) →
  (∀ x, f (x + 1) = x^2 + 8*x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_function_shift_l2376_237606


namespace NUMINAMATH_CALUDE_cut_cube_total_count_cube_cutting_problem_l2376_237626

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly two faces -/
  two_face_painted : ℕ

/-- Theorem stating that a cube cut into smaller cubes with 12 two-face painted cubes results in 27 total cubes -/
theorem cut_cube_total_count (c : CutCube) (h1 : c.two_face_painted = 12) : 
  c.edge_count ^ 3 = 27 := by
  sorry

/-- Main theorem proving the solution to the original problem -/
theorem cube_cutting_problem : 
  ∃ (c : CutCube), c.two_face_painted = 12 ∧ c.edge_count ^ 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_total_count_cube_cutting_problem_l2376_237626


namespace NUMINAMATH_CALUDE_woods_length_l2376_237610

/-- Given a rectangular area of woods with width 8 miles and total area 24 square miles,
    prove that the length of the woods is 3 miles. -/
theorem woods_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 8 → area = 24 → area = width * length → length = 3 := by sorry

end NUMINAMATH_CALUDE_woods_length_l2376_237610


namespace NUMINAMATH_CALUDE_cargo_arrival_time_l2376_237615

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

end NUMINAMATH_CALUDE_cargo_arrival_time_l2376_237615


namespace NUMINAMATH_CALUDE_profit_calculation_l2376_237642

theorem profit_calculation (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : marked_price > 0)
  (h3 : 0.8 * marked_price = 1.2 * cost_price) :
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l2376_237642


namespace NUMINAMATH_CALUDE_clarissa_manuscript_cost_l2376_237661

/-- Calculate the total cost for printing, binding, and processing multiple copies of a manuscript with specified requirements. -/
def manuscript_cost (total_pages : ℕ) (color_pages : ℕ) (bw_cost : ℚ) (color_cost : ℚ) 
                    (binding_cost : ℚ) (index_cost : ℚ) (copies : ℕ) (rush_copies : ℕ) 
                    (rush_cost : ℚ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost
  let total_per_copy := print_cost + additional_cost
  let total_before_rush := (copies : ℚ) * total_per_copy
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_rush + rush_fee

/-- The total cost for Clarissa's manuscript printing job is $310.00. -/
theorem clarissa_manuscript_cost :
  manuscript_cost 400 50 (5/100) (10/100) 5 2 10 5 3 = 310 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_manuscript_cost_l2376_237661


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2376_237612

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

theorem sufficient_not_necessary_condition :
  (∀ a < 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (∃ a ≥ 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2376_237612


namespace NUMINAMATH_CALUDE_parabola_coeffs_sum_l2376_237656

/-- Parabola coefficients -/
structure ParabolaCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Parabola equation -/
def parabola_equation (coeffs : ParabolaCoeffs) (y : ℝ) : ℝ :=
  coeffs.a * y^2 + coeffs.b * y + coeffs.c

/-- Theorem: Parabola coefficients and their sum -/
theorem parabola_coeffs_sum :
  ∀ (coeffs : ParabolaCoeffs),
  parabola_equation coeffs 5 = 6 ∧
  parabola_equation coeffs 3 = 0 ∧
  (∀ y : ℝ, parabola_equation coeffs y = coeffs.a * (y - 3)^2) →
  coeffs.a = 3/2 ∧ coeffs.b = -9 ∧ coeffs.c = 27/2 ∧
  coeffs.a + coeffs.b + coeffs.c = 6 :=
by sorry


end NUMINAMATH_CALUDE_parabola_coeffs_sum_l2376_237656


namespace NUMINAMATH_CALUDE_i_cubed_plus_i_squared_in_third_quadrant_l2376_237625

def i : ℂ := Complex.I

-- Define the quadrants of the complex plane
def first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0
def fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem i_cubed_plus_i_squared_in_third_quadrant :
  third_quadrant (i^3 + i^2) :=
by
  sorry

end NUMINAMATH_CALUDE_i_cubed_plus_i_squared_in_third_quadrant_l2376_237625


namespace NUMINAMATH_CALUDE_batsman_new_average_l2376_237654

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  runsIn17thInning : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem stating the batsman's new average after the 17th inning -/
theorem batsman_new_average (b : Batsman) 
  (h1 : b.runsIn17thInning = 74)
  (h2 : b.averageIncrease = 3) : 
  newAverage b = 26 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l2376_237654


namespace NUMINAMATH_CALUDE_eighth_term_ratio_l2376_237630

/-- Two arithmetic sequences U and V with their partial sums Un and Vn -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (u f v g : ℚ), ∀ n,
    U n = u + (n - 1) * f ∧
    V n = v + (n - 1) * g

/-- Partial sum of the first n terms of an arithmetic sequence -/
def partial_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem eighth_term_ratio
  (U V : ℕ → ℚ)
  (h_arith : arithmetic_sequences U V)
  (h_ratio : ∀ n, partial_sum U n / partial_sum V n = (5 * n + 5) / (3 * n + 9)) :
  U 8 / V 8 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_ratio_l2376_237630


namespace NUMINAMATH_CALUDE_height_ratio_of_cones_l2376_237696

/-- The ratio of heights of two right circular cones with the same base circumference -/
theorem height_ratio_of_cones (r : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → h₁ > 0 → h₂ > 0 →
  (2 * Real.pi * r = 20 * Real.pi) →
  ((1/3) * Real.pi * r^2 * h₁ = 400 * Real.pi) →
  h₂ = 40 →
  h₁ / h₂ = 3/10 := by
sorry

end NUMINAMATH_CALUDE_height_ratio_of_cones_l2376_237696


namespace NUMINAMATH_CALUDE_dozen_chocolates_cost_l2376_237665

/-- The cost of a magazine in dollars -/
def magazine_cost : ℝ := 1

/-- The cost of a chocolate bar in dollars -/
def chocolate_cost : ℝ := 2

/-- The number of magazines that cost the same as 4 chocolate bars -/
def magazines_equal_to_4_chocolates : ℕ := 8

theorem dozen_chocolates_cost (h : 4 * chocolate_cost = magazines_equal_to_4_chocolates * magazine_cost) :
  12 * chocolate_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_dozen_chocolates_cost_l2376_237665


namespace NUMINAMATH_CALUDE_arccos_neg_half_eq_two_pi_thirds_l2376_237669

theorem arccos_neg_half_eq_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_half_eq_two_pi_thirds_l2376_237669


namespace NUMINAMATH_CALUDE_circle_radius_equals_sphere_surface_area_l2376_237672

theorem circle_radius_equals_sphere_surface_area (r : ℝ) : 
  r > 0 → π * r^2 = 4 * π * (2 : ℝ)^2 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equals_sphere_surface_area_l2376_237672


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2023_l2376_237639

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising 7 to the power of 2023 -/
def power : ℕ := 7^2023

/-- Theorem: The units digit of 7^2023 is 3 -/
theorem units_digit_of_7_pow_2023 : unitsDigit power = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2023_l2376_237639


namespace NUMINAMATH_CALUDE_direction_vector_form_l2376_237655

/-- Given a line passing through two points, prove that its direction vector has a specific form -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 2) → p2 = (2, -3) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_form_l2376_237655


namespace NUMINAMATH_CALUDE_skewReflectionAndShrinkIsCorrectTransformation_l2376_237694

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid transformation in 2D space -/
structure RigidTransformation where
  transform : Point2D → Point2D

/-- Skew-reflection across y=x followed by a vertical shrink by a factor of -1 -/
def skewReflectionAndShrink : RigidTransformation :=
  { transform := λ p => Point2D.mk p.y (-p.x) }

theorem skewReflectionAndShrinkIsCorrectTransformation :
  let C := Point2D.mk 3 (-2)
  let D := Point2D.mk 4 (-3)
  let C' := Point2D.mk 1 2
  let D' := Point2D.mk (-2) 3
  (skewReflectionAndShrink.transform C = C') ∧
  (skewReflectionAndShrink.transform D = D') := by
  sorry


end NUMINAMATH_CALUDE_skewReflectionAndShrinkIsCorrectTransformation_l2376_237694


namespace NUMINAMATH_CALUDE_ball_placement_events_l2376_237674

structure Ball :=
  (number : Nat)

structure Box :=
  (number : Nat)

def Placement := Ball → Box

def event_ball1_in_box1 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨1⟩

def event_ball1_in_box2 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨2⟩

def mutually_exclusive (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, ¬(e1 p ∧ e2 p)

def complementary (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, e1 p ↔ ¬(e2 p)

theorem ball_placement_events :
  (mutually_exclusive event_ball1_in_box1 event_ball1_in_box2) ∧
  ¬(complementary event_ball1_in_box1 event_ball1_in_box2) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_events_l2376_237674


namespace NUMINAMATH_CALUDE_probability_of_double_is_two_ninths_l2376_237633

/-- Represents a domino tile with two squares -/
structure Domino :=
  (first : Nat)
  (second : Nat)

/-- The set of all possible dominos with integers from 0 to 7 -/
def dominoSet : Finset Domino :=
  sorry

/-- Predicate to check if a domino is a double -/
def isDouble (d : Domino) : Bool :=
  d.first = d.second

/-- The probability of selecting a double from the domino set -/
def probabilityOfDouble : ℚ :=
  sorry

/-- Theorem stating that the probability of selecting a double is 2/9 -/
theorem probability_of_double_is_two_ninths :
  probabilityOfDouble = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_double_is_two_ninths_l2376_237633


namespace NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l2376_237666

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics + batman_comics

def comic_groups : ℕ := 4

theorem comic_book_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  comic_groups.factorial = 1244160 :=
by sorry

end NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l2376_237666


namespace NUMINAMATH_CALUDE_quadrilateral_complex_point_l2376_237613

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  z : ℂ

/-- Represents a quadrilateral with vertices A, B, C, D -/
structure Quadrilateral where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- Theorem: In quadrilateral ABCD, if A, B, and C correspond to given complex numbers,
    then D corresponds to 1+3i -/
theorem quadrilateral_complex_point (q : Quadrilateral)
    (hA : q.A.z = 2 + I)
    (hB : q.B.z = 4 + 3*I)
    (hC : q.C.z = 3 + 5*I) :
    q.D.z = 1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_complex_point_l2376_237613


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2376_237679

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) → 
  (∃ p : ℕ, p = 17 ∧ Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2376_237679


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2376_237600

-- Define the ellipse E
def ellipse_E (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the vertex condition
def vertex_condition (a b : ℝ) : Prop :=
  ellipse_E 0 1 a b

-- Define the focal length condition
def focal_length_condition (a b : ℝ) : Prop :=
  2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 - b^2)

-- Define the intersection condition
def intersection_condition (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_E x₁ y₁ a b ∧
    ellipse_E x₂ y₂ a b ∧
    y₁ - 1 = k * (x₁ + 2) ∧
    y₂ - 1 = k * (x₂ + 2) ∧
    x₁ ≠ x₂

-- Define the x-intercept distance condition
def x_intercept_distance (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersection_condition k a b ∧
    |x₁ / (1 - y₁) - x₂ / (1 - y₂)| = 2

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) :
  vertex_condition a b ∧ focal_length_condition a b →
  (a = 2 ∧ b = 1) ∧
  (∀ k : ℝ, x_intercept_distance k a b → k = -4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2376_237600


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2376_237621

theorem ratio_sum_problem (a b : ℕ) : 
  a * 3 = b * 8 →  -- The two numbers are in the ratio 8 to 3
  b = 104 →        -- The bigger number is 104
  a + b = 143      -- The sum of the numbers is 143
  := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2376_237621


namespace NUMINAMATH_CALUDE_min_value_x3_l2376_237667

theorem min_value_x3 (x₁ x₂ x₃ : ℝ) 
  (eq1 : x₁ + (1/2) * x₂ + (1/3) * x₃ = 1)
  (eq2 : x₁^2 + (1/2) * x₂^2 + (1/3) * x₃^2 = 3) :
  x₃ ≥ -21/11 ∧ ∃ (x₁' x₂' x₃' : ℝ), 
    x₁' + (1/2) * x₂' + (1/3) * x₃' = 1 ∧
    x₁'^2 + (1/2) * x₂'^2 + (1/3) * x₃'^2 = 3 ∧
    x₃' = -21/11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x3_l2376_237667


namespace NUMINAMATH_CALUDE_urns_can_be_emptied_l2376_237651

/-- Represents the two types of operations that can be performed on the urns -/
inductive UrnOperation
  | Remove : ℕ → UrnOperation
  | DoubleFirst : UrnOperation
  | DoubleSecond : UrnOperation

/-- Applies a single operation to the pair of urns -/
def applyOperation (a b : ℕ) (op : UrnOperation) : ℕ × ℕ :=
  match op with
  | UrnOperation.Remove n => (a - min a n, b - min b n)
  | UrnOperation.DoubleFirst => (2 * a, b)
  | UrnOperation.DoubleSecond => (a, 2 * b)

/-- Theorem: Both urns can be made empty after a finite number of operations -/
theorem urns_can_be_emptied (a b : ℕ) :
  ∃ (ops : List UrnOperation), (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).1 = 0 ∧
                               (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_urns_can_be_emptied_l2376_237651


namespace NUMINAMATH_CALUDE_integral_f_equals_one_plus_pi_over_four_l2376_237693

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else if -1 ≤ x ∧ x ≤ 0 then x + 1
  else 0  -- This case is added to make the function total

-- State the theorem
theorem integral_f_equals_one_plus_pi_over_four :
  ∫ x in (-1)..1, f x = (1 + Real.pi) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_integral_f_equals_one_plus_pi_over_four_l2376_237693


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2376_237640

-- Define the slopes of the lines
def m1 : ℚ := 3 / 4
def m2 : ℚ := -3 / 4
def m3 : ℚ := -3 / 4
def m4 : ℚ := -4 / 3
def m5 : ℚ := 12 / 5

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m1 m5) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m2 m5) ∧
  (¬ are_perpendicular m3 m4) ∧
  (¬ are_perpendicular m3 m5) ∧
  (¬ are_perpendicular m4 m5) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2376_237640


namespace NUMINAMATH_CALUDE_walking_distance_l2376_237618

/-- 
Given a person who walks for time t hours:
- At 12 km/hr, they cover a distance of 12t km
- At 20 km/hr, they cover a distance of 20t km
- The difference between these distances is 30 km

Prove that the actual distance travelled at 12 km/hr is 45 km
-/
theorem walking_distance (t : ℝ) 
  (h1 : 20 * t = 12 * t + 30) : 12 * t = 45 := by sorry

end NUMINAMATH_CALUDE_walking_distance_l2376_237618


namespace NUMINAMATH_CALUDE_bookstore_shipment_l2376_237699

/-- Proves that the total number of books in a shipment is 240, given that 25% are displayed
    in the front and the remaining 180 books are in the storage room. -/
theorem bookstore_shipment (displayed_percent : ℚ) (storage_count : ℕ) : ℕ :=
  let total_books : ℕ := 240
  have h1 : displayed_percent = 25 / 100 := by sorry
  have h2 : storage_count = 180 := by sorry
  have h3 : (1 - displayed_percent) * total_books = storage_count := by sorry
  total_books

#check bookstore_shipment

end NUMINAMATH_CALUDE_bookstore_shipment_l2376_237699


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2376_237683

/-- Represents a chess tournament --/
structure ChessTournament where
  participants : ℕ
  total_games : ℕ
  games_per_player : ℕ
  h1 : total_games = participants * (participants - 1) / 2
  h2 : games_per_player = participants - 1

/-- Theorem: In a chess tournament with 20 participants and 190 total games, 
    each participant plays 19 games --/
theorem chess_tournament_games (t : ChessTournament) 
  (h_participants : t.participants = 20) 
  (h_total_games : t.total_games = 190) : 
  t.games_per_player = 19 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l2376_237683


namespace NUMINAMATH_CALUDE_hari_contribution_correct_l2376_237646

/-- Calculates Hari's contribution to the capital given the initial conditions of the business partnership --/
def calculate_hari_contribution (praveen_capital : ℕ) (praveen_months : ℕ) (hari_months : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  (3 * praveen_capital * praveen_months) / (2 * hari_months)

theorem hari_contribution_correct :
  let praveen_capital : ℕ := 3780
  let total_months : ℕ := 12
  let hari_join_month : ℕ := 5
  let profit_ratio_praveen : ℕ := 2
  let profit_ratio_hari : ℕ := 3
  let praveen_months : ℕ := total_months
  let hari_months : ℕ := total_months - hari_join_month
  calculate_hari_contribution praveen_capital praveen_months hari_months profit_ratio_praveen profit_ratio_hari = 9720 :=
by
  sorry

#eval calculate_hari_contribution 3780 12 7 2 3

end NUMINAMATH_CALUDE_hari_contribution_correct_l2376_237646


namespace NUMINAMATH_CALUDE_ratio_equality_solutions_l2376_237634

theorem ratio_equality_solutions :
  {x : ℝ | (x + 3) / (3 * x + 3) = (3 * x + 4) / (6 * x + 4)} = {0, 1/3} := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solutions_l2376_237634


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2376_237628

theorem power_mod_eleven : 3^21 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2376_237628
