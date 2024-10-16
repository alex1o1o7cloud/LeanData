import Mathlib

namespace NUMINAMATH_CALUDE_swimming_pool_receipts_l3645_364544

theorem swimming_pool_receipts 
  (total_people : ℕ) 
  (children : ℕ) 
  (adults : ℕ) 
  (child_price : ℚ) 
  (adult_price : ℚ) 
  (h1 : total_people = 754)
  (h2 : children = 388)
  (h3 : adults = total_people - children)
  (h4 : child_price = 3/2)
  (h5 : adult_price = 9/4) :
  children * child_price + adults * adult_price = 2811/2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_receipts_l3645_364544


namespace NUMINAMATH_CALUDE_hexagon_area_from_triangle_l3645_364578

/-- Given an equilateral triangle and a regular hexagon with equal perimeters,
    if the area of the triangle is β, then the area of the hexagon is (3/2) * β. -/
theorem hexagon_area_from_triangle (β : ℝ) :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  (3 * x = 6 * y) →  -- Equal perimeters
  (β = Real.sqrt 3 / 4 * x^2) →  -- Area of equilateral triangle
  ∃ (γ : ℝ), γ = 3 * Real.sqrt 3 / 2 * y^2 ∧ γ = 3/2 * β := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_from_triangle_l3645_364578


namespace NUMINAMATH_CALUDE_expression_bound_l3645_364560

theorem expression_bound (x : ℝ) (h : x^2 - 7*x + 12 ≤ 0) : 
  40 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 54 := by
sorry

end NUMINAMATH_CALUDE_expression_bound_l3645_364560


namespace NUMINAMATH_CALUDE_min_value_of_sequence_sequence_satisfies_conditions_l3645_364589

def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 98
  else 102 + (n - 2) * (2 * n + 2)

theorem min_value_of_sequence (n : ℕ) (h : n > 0) :
  sequence_a n / n ≥ 26 ∧ ∃ m : ℕ, m > 0 ∧ sequence_a m / m = 26 :=
by
  sorry

theorem sequence_satisfies_conditions :
  sequence_a 2 = 102 ∧
  ∀ n : ℕ, n > 0 → sequence_a (n + 1) - sequence_a n = 4 * n :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_sequence_satisfies_conditions_l3645_364589


namespace NUMINAMATH_CALUDE_distance_acaster_beetown_is_315_l3645_364537

/-- The distance from Acaster to Beetown in kilometers. -/
def distance_acaster_beetown : ℝ := 315

/-- Lewis's speed in km/h. -/
def lewis_speed : ℝ := 70

/-- Geraint's speed in km/h. -/
def geraint_speed : ℝ := 30

/-- The distance from the meeting point to Beetown in kilometers. -/
def distance_meeting_beetown : ℝ := 105

/-- The time Lewis spends in Beetown in hours. -/
def lewis_stop_time : ℝ := 1

theorem distance_acaster_beetown_is_315 :
  let total_time := distance_acaster_beetown / geraint_speed
  let lewis_travel_time := total_time - lewis_stop_time
  lewis_travel_time * lewis_speed = distance_acaster_beetown + distance_meeting_beetown ∧
  total_time * geraint_speed = distance_acaster_beetown - distance_meeting_beetown ∧
  distance_acaster_beetown = 315 := by
  sorry

#check distance_acaster_beetown_is_315

end NUMINAMATH_CALUDE_distance_acaster_beetown_is_315_l3645_364537


namespace NUMINAMATH_CALUDE_original_price_calculation_l3645_364531

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.06) * (1 + 0.10) = 6876.1) → P = 6650 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3645_364531


namespace NUMINAMATH_CALUDE_michael_passes_donovan_l3645_364528

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℝ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℝ := 40

/-- The number of laps Michael needs to complete to pass Donovan -/
def laps_to_pass : ℕ := 9

theorem michael_passes_donovan :
  (laps_to_pass : ℝ) * michael_lap_time = (laps_to_pass - 1 : ℝ) * donovan_lap_time :=
sorry

end NUMINAMATH_CALUDE_michael_passes_donovan_l3645_364528


namespace NUMINAMATH_CALUDE_max_trees_in_garden_l3645_364502

def garden_width : ℝ := 27.9
def tree_interval : ℝ := 3.1

theorem max_trees_in_garden : 
  ⌊garden_width / tree_interval⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_max_trees_in_garden_l3645_364502


namespace NUMINAMATH_CALUDE_diagonal_intersects_n_rhombuses_l3645_364587

/-- A regular hexagon with side length n -/
structure RegularHexagon (n : ℕ) where
  side_length : ℕ
  is_positive : 0 < side_length
  eq_n : side_length = n

/-- A rhombus with internal angles 60° and 120° -/
structure Rhombus where
  internal_angles : Fin 2 → ℝ
  angle_sum : internal_angles 0 + internal_angles 1 = 180
  angles_correct : (internal_angles 0 = 60 ∧ internal_angles 1 = 120) ∨ 
                   (internal_angles 0 = 120 ∧ internal_angles 1 = 60)

/-- Theorem: The diagonal of a regular hexagon intersects n rhombuses -/
theorem diagonal_intersects_n_rhombuses (n : ℕ) (h : RegularHexagon n) :
  ∃ (rhombuses : Finset Rhombus),
    (Finset.card rhombuses = 3 * n^2) ∧
    (∃ (intersected : Finset Rhombus),
      Finset.card intersected = n ∧
      intersected ⊆ rhombuses) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersects_n_rhombuses_l3645_364587


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3645_364557

theorem battle_station_staffing (n m : ℕ) (h1 : n = 20) (h2 : m = 5) :
  (n - 1).factorial / (n - m).factorial = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3645_364557


namespace NUMINAMATH_CALUDE_inequality_problem_l3645_364554

theorem inequality_problem (m n : ℝ) (h : ∀ x : ℝ, m * x^2 + n * x - 1/m < 0 ↔ x < -1/2 ∨ x > 2) :
  (m = -1 ∧ n = 3/2) ∧
  (∀ a : ℝ, 
    (a < 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 2*a-1 < x ∧ x < 1) ∧
    (a = 1 → ∀ x : ℝ, ¬((2*a-1-x)*(x+m) > 0)) ∧
    (a > 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 1 < x ∧ x < 2*a-1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3645_364554


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3645_364562

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    if its solution set for f(x) > 0 is (-1, 2), 
    then b + c = -3 -/
theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → 
  b + c = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3645_364562


namespace NUMINAMATH_CALUDE_lattice_triangle_area_bound_l3645_364561

/-- A 3D lattice point is represented as a triple of integers -/
def LatticePoint3D := ℤ × ℤ × ℤ

/-- A triangle in 3D space is represented by its three vertices -/
structure Triangle3D where
  v1 : LatticePoint3D
  v2 : LatticePoint3D
  v3 : LatticePoint3D

/-- The area of a triangle -/
noncomputable def area (t : Triangle3D) : ℝ := sorry

/-- Theorem: The area of a triangle with vertices at 3D lattice points is at least 1/2 -/
theorem lattice_triangle_area_bound (t : Triangle3D) : area t ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_bound_l3645_364561


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l3645_364595

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 360 →
  speed1 = 60 →
  distance2 = 120 →
  speed2 = 40 →
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 :=
by
  sorry

#eval (480 : ℚ) / 9

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l3645_364595


namespace NUMINAMATH_CALUDE_chord_length_is_three_l3645_364552

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passing through the focus and perpendicular to x-axis -/
def line (x : ℝ) : Prop := x = (focus.1)

/-- The chord length -/
def chord_length : ℝ := 3

/-- Theorem stating that the chord length cut by the line passing through
    the focus of the ellipse and perpendicular to the x-axis is equal to 3 -/
theorem chord_length_is_three :
  ∀ y₁ y₂ : ℝ,
  ellipse (focus.1) y₁ ∧ ellipse (focus.1) y₂ ∧ y₁ ≠ y₂ →
  |y₁ - y₂| = chord_length :=
sorry

end NUMINAMATH_CALUDE_chord_length_is_three_l3645_364552


namespace NUMINAMATH_CALUDE_triangle_properties_l3645_364529

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a ≠ t.b ∧
  2 * Real.sin (t.A - t.B) = t.a * Real.sin t.A - t.b * Real.sin t.B ∧
  (1/2) * t.a * t.b * Real.sin t.C = 1 ∧
  Real.tan t.C = 2

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.c = 2 ∧ t.a + t.b = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3645_364529


namespace NUMINAMATH_CALUDE_algebraic_identity_l3645_364568

theorem algebraic_identity (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l3645_364568


namespace NUMINAMATH_CALUDE_total_leaves_calculation_l3645_364583

/-- Calculates the total number of leaves falling from cherry and maple trees -/
def total_leaves (initial_cherry : ℕ) (initial_maple : ℕ) 
                 (cherry_ratio : ℕ) (maple_ratio : ℕ) 
                 (cherry_leaves : ℕ) (maple_leaves : ℕ) : ℕ :=
  (initial_cherry * cherry_ratio * cherry_leaves) + 
  (initial_maple * maple_ratio * maple_leaves)

/-- Theorem stating that the total number of leaves is 3650 -/
theorem total_leaves_calculation : 
  total_leaves 7 5 2 3 100 150 = 3650 := by
  sorry

#eval total_leaves 7 5 2 3 100 150

end NUMINAMATH_CALUDE_total_leaves_calculation_l3645_364583


namespace NUMINAMATH_CALUDE_surface_area_of_specific_cut_tetrahedron_l3645_364540

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents the tetrahedron formed by cutting the prism -/
structure CutTetrahedron where
  prism : RightPrism

/-- Calculate the surface area of the cut tetrahedron -/
noncomputable def surface_area (tetra : CutTetrahedron) : ℝ :=
  sorry

/-- Theorem statement for the surface area of the specific cut tetrahedron -/
theorem surface_area_of_specific_cut_tetrahedron :
  let prism := RightPrism.mk 20 10
  let tetra := CutTetrahedron.mk prism
  surface_area tetra = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_cut_tetrahedron_l3645_364540


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3645_364549

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3645_364549


namespace NUMINAMATH_CALUDE_extreme_value_point_property_l3645_364586

/-- The function f(x) = x³ - x² + ax - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

/-- Theorem: If f(x) has an extreme value point x₀ and f(x₁) = f(x₀) where x₁ ≠ x₀, then x₁ + 2x₀ = 1 -/
theorem extreme_value_point_property (a : ℝ) (x₀ x₁ : ℝ) 
  (h_extreme : ∃ ε > 0, ∀ x, |x - x₀| < ε → f a x ≤ f a x₀ ∨ f a x ≥ f a x₀)
  (h_equal : f a x₁ = f a x₀)
  (h_distinct : x₁ ≠ x₀) : 
  x₁ + 2*x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_point_property_l3645_364586


namespace NUMINAMATH_CALUDE_bat_wings_area_is_four_l3645_364585

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the "bat wings" shape -/
structure BatWings where
  rect : Rectangle
  quarterCircleRadius : ℝ

/-- Calculate the area of the "bat wings" -/
noncomputable def batWingsArea (bw : BatWings) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem bat_wings_area_is_four :
  ∀ (bw : BatWings),
    bw.rect.width = 4 ∧
    bw.rect.height = 5 ∧
    bw.quarterCircleRadius = 2 →
    batWingsArea bw = 4 := by
  sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_four_l3645_364585


namespace NUMINAMATH_CALUDE_kelly_games_left_l3645_364556

/-- Calculates the number of games left after finding more and giving some away -/
def games_left (initial : ℕ) (found : ℕ) (given_away : ℕ) : ℕ :=
  initial + found - given_away

/-- Proves that Kelly will have 6 games left -/
theorem kelly_games_left : games_left 80 31 105 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l3645_364556


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3645_364588

theorem algebraic_simplification (a b : ℝ) : -a^2 * (-2*a*b) + 3*a * (a^2*b - 1) = 5*a^3*b - 3*a := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3645_364588


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3645_364567

def biased_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem three_heads_in_eight_tosses :
  biased_coin_probability 8 3 (1/3) = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3645_364567


namespace NUMINAMATH_CALUDE_ryan_pages_theorem_l3645_364527

/-- The number of books Ryan got from the library -/
def ryan_books : ℕ := 5

/-- The number of pages in each of Ryan's brother's books -/
def brother_book_pages : ℕ := 200

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The additional pages Ryan reads per day compared to his brother -/
def ryan_extra_pages_per_day : ℕ := 100

/-- The total number of pages in Ryan's books -/
def ryan_total_pages : ℕ := days_in_week * (brother_book_pages + ryan_extra_pages_per_day)

theorem ryan_pages_theorem :
  ryan_total_pages = 2100 :=
by sorry

end NUMINAMATH_CALUDE_ryan_pages_theorem_l3645_364527


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l3645_364508

/-- The mass of a man who causes a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sink_depth water_density : Real) : Real :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specific conditions --/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

#eval mass_of_man 3 2 0.01 1000

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l3645_364508


namespace NUMINAMATH_CALUDE_sand_pile_volume_l3645_364581

/-- The volume of a cone with diameter 12 feet and height 60% of the diameter is 86.4π cubic feet -/
theorem sand_pile_volume : 
  let diameter : ℝ := 12
  let height : ℝ := 0.6 * diameter
  let radius : ℝ := diameter / 2
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 86.4 * π := by sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l3645_364581


namespace NUMINAMATH_CALUDE_arithmetic_progression_special_case_l3645_364553

/-- 
Given an arithmetic progression (a_n) where a_k = l and a_l = k (k ≠ l),
prove that the general term a_n is equal to k + l - n.
-/
theorem arithmetic_progression_special_case 
  (a : ℕ → ℤ) (k l : ℕ) (h_neq : k ≠ l) 
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_k : a k = l) (h_l : a l = k) :
  ∀ n : ℕ, a n = k + l - n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_special_case_l3645_364553


namespace NUMINAMATH_CALUDE_blueberries_count_l3645_364573

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def berry_increase : ℕ := 10

/-- The increase in the difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 50

theorem blueberries_count : 
  (strawberries - blueberries = berry_increase) ∧ 
  (strawberries = difference_increase) → 
  blueberries = 40 := by sorry

end NUMINAMATH_CALUDE_blueberries_count_l3645_364573


namespace NUMINAMATH_CALUDE_completing_square_transform_l3645_364509

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_completing_square_transform_l3645_364509


namespace NUMINAMATH_CALUDE_count_fractions_is_36_l3645_364594

/-- A function that counts the number of fractions less than 1 with single-digit numerators and denominators -/
def count_fractions : ℕ := 
  let single_digit (n : ℕ) := n ≥ 1 ∧ n ≤ 9
  let is_valid_fraction (n d : ℕ) := single_digit n ∧ single_digit d ∧ n < d
  (Finset.sum (Finset.range 9) (λ d => 
    (Finset.filter (λ n => is_valid_fraction n (d + 1)) (Finset.range (d + 1))).card
  ))

/-- Theorem stating that the count of fractions less than 1 with single-digit numerators and denominators is 36 -/
theorem count_fractions_is_36 : count_fractions = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_fractions_is_36_l3645_364594


namespace NUMINAMATH_CALUDE_sector_central_angle_l3645_364515

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = π → area = 2 * π → ∃ (r : Real) (α : Real),
    r > 0 ∧ α > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = r * α ∧ α = π/4 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3645_364515


namespace NUMINAMATH_CALUDE_action_figures_total_l3645_364500

theorem action_figures_total (initial : ℕ) (added : ℕ) : 
  initial = 8 → added = 2 → initial + added = 10 := by
sorry

end NUMINAMATH_CALUDE_action_figures_total_l3645_364500


namespace NUMINAMATH_CALUDE_circle_area_not_covered_l3645_364559

theorem circle_area_not_covered (outer_diameter inner_diameter : ℝ) 
  (h1 : outer_diameter = 30) 
  (h2 : inner_diameter = 24) : 
  (outer_diameter^2 - inner_diameter^2) / outer_diameter^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_not_covered_l3645_364559


namespace NUMINAMATH_CALUDE_sum_of_digits_squared_difference_l3645_364575

def x : ℕ := 777777777777777
def y : ℕ := 222222222222223

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_squared_difference : 
  digit_sum ((x^2 : ℕ) - (y^2 : ℕ)) = 74 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_squared_difference_l3645_364575


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_2_range_of_a_for_proper_superset_l3645_364520

-- Define sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3*a - 10 ≤ x ∧ x < 2*a + 1}
def Q : Set ℝ := {x | |2*x - 3| ≤ 7}

-- Part 1
theorem intersection_complement_when_a_2 : 
  P 2 ∩ (Set.univ \ Q) = {x | -4 ≤ x ∧ x < -2} := by sorry

-- Part 2
theorem range_of_a_for_proper_superset : 
  {a : ℝ | P a ⊃ Q ∧ P a ≠ Q} = Set.Ioo 2 (8/3) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_2_range_of_a_for_proper_superset_l3645_364520


namespace NUMINAMATH_CALUDE_birds_on_fence_l3645_364516

theorem birds_on_fence : ∃ (B : ℕ), ∃ (x : ℝ), 
  (Real.sqrt (B : ℝ) = x) ∧ 
  (2 * x^2 + 10 = 50) ∧ 
  (B = 20) := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3645_364516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3645_364504

/-- An arithmetic sequence with general term a_n = 4n - 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

/-- The first term of the sequence -/
def first_term : ℤ := arithmetic_sequence 1

/-- The second term of the sequence -/
def second_term : ℤ := arithmetic_sequence 2

/-- The common difference of the sequence -/
def common_difference : ℤ := second_term - first_term

theorem arithmetic_sequence_properties :
  first_term = 1 ∧ common_difference = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3645_364504


namespace NUMINAMATH_CALUDE_triangle_area_inequalities_l3645_364570

/-- The area of a triangle ABC with sides a and b is less than or equal to both
    (1/2)(a² - ab + b²) and ((a + b)/(2√2))² -/
theorem triangle_area_inequalities (a b : ℝ) (hpos : 0 < a ∧ 0 < b) :
  let area := (1/2) * a * b * Real.sin C
  ∃ C, 0 ≤ C ∧ C ≤ π ∧
    area ≤ (1/2) * (a^2 - a*b + b^2) ∧
    area ≤ ((a + b)/(2 * Real.sqrt 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequalities_l3645_364570


namespace NUMINAMATH_CALUDE_minimum_score_for_eligibility_l3645_364548

def minimum_score (q1 q2 q3 : ℚ) (target_average : ℚ) : ℚ :=
  4 * target_average - (q1 + q2 + q3)

theorem minimum_score_for_eligibility 
  (q1 q2 q3 : ℚ) 
  (target_average : ℚ) 
  (h1 : q1 = 80) 
  (h2 : q2 = 85) 
  (h3 : q3 = 78) 
  (h4 : target_average = 85) :
  minimum_score q1 q2 q3 target_average = 97 := by
sorry

end NUMINAMATH_CALUDE_minimum_score_for_eligibility_l3645_364548


namespace NUMINAMATH_CALUDE_solution_sets_intersection_l3645_364571

/-- The solution set of x^2 - 2x - 3 < 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

/-- The solution set of x^2 + x - 6 < 0 -/
def B : Set ℝ := {x | x^2 + x - 6 < 0}

/-- The solution set of x^2 + ax + b < 0 -/
def C (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

theorem solution_sets_intersection (a b : ℝ) :
  C a b = A ∩ B → a + b = -3 := by sorry

end NUMINAMATH_CALUDE_solution_sets_intersection_l3645_364571


namespace NUMINAMATH_CALUDE_point_on_line_with_sum_distance_l3645_364577

-- Define the line l
def Line : Type := ℝ → Prop

-- Define the concept of a point being on the same side of a line
def SameSide (l : Line) (A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define what it means for a point to be on a line
def OnLine (X : ℝ × ℝ) (l : Line) : Prop := sorry

-- Theorem statement
theorem point_on_line_with_sum_distance 
  (l : Line) (A B : ℝ × ℝ) (a : ℝ) 
  (h1 : SameSide l A B) (h2 : a > 0) : 
  ∃ X : ℝ × ℝ, OnLine X l ∧ distance A X + distance X B = a := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_sum_distance_l3645_364577


namespace NUMINAMATH_CALUDE_remainder_of_98_times_102_mod_9_l3645_364536

theorem remainder_of_98_times_102_mod_9 : (98 * 102) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_98_times_102_mod_9_l3645_364536


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l3645_364584

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at_zero : ℝ × ℝ × ℝ
  point_at_one : ℝ × ℝ × ℝ

/-- The vector on the line at a given parameter value -/
def vector_at_t (line : ParameterizedLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x₀, y₀, z₀) := line.point_at_zero
  let (x₁, y₁, z₁) := line.point_at_one
  (x₀ + t * (x₁ - x₀), y₀ + t * (y₁ - y₀), z₀ + t * (z₁ - z₀))

theorem vector_at_negative_one (line : ParameterizedLine) 
  (h₀ : line.point_at_zero = (2, 6, 16))
  (h₁ : line.point_at_one = (1, 1, 8)) :
  vector_at_t line (-1) = (3, 11, 24) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l3645_364584


namespace NUMINAMATH_CALUDE_inverse_proportion_function_l3645_364550

/-- Given that y is inversely proportional to x and y = 1 when x = 2,
    prove that the function expression of y with respect to x is y = 2/x. -/
theorem inverse_proportion_function (x : ℝ) (y : ℝ → ℝ) (k : ℝ) :
  (∀ x ≠ 0, y x = k / x) →  -- y is inversely proportional to x
  y 2 = 1 →                 -- when x = 2, y = 1
  ∀ x ≠ 0, y x = 2 / x :=   -- the function expression is y = 2/x
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_function_l3645_364550


namespace NUMINAMATH_CALUDE_linear_function_property_l3645_364564

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) (h_diff : g 10 - g 5 = 20) :
  g 20 - g 5 = 60 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l3645_364564


namespace NUMINAMATH_CALUDE_macaroons_remaining_l3645_364592

/-- The number of red macaroons initially baked -/
def initial_red : ℕ := 50

/-- The number of green macaroons initially baked -/
def initial_green : ℕ := 40

/-- The number of green macaroons eaten -/
def green_eaten : ℕ := 15

/-- The number of red macaroons eaten is twice the number of green macaroons eaten -/
def red_eaten : ℕ := 2 * green_eaten

/-- The total number of remaining macaroons -/
def remaining_macaroons : ℕ := (initial_red - red_eaten) + (initial_green - green_eaten)

theorem macaroons_remaining :
  remaining_macaroons = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroons_remaining_l3645_364592


namespace NUMINAMATH_CALUDE_sphere_surface_area_for_given_prism_l3645_364551

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  height : ℝ
  volume : ℝ
  prism_on_sphere : Bool

/-- The surface area of a sphere given a PrismOnSphere -/
def sphere_surface_area (p : PrismOnSphere) : ℝ := sorry

theorem sphere_surface_area_for_given_prism :
  ∀ p : PrismOnSphere,
    p.height = 4 ∧ 
    p.volume = 16 ∧ 
    p.prism_on_sphere = true →
    sphere_surface_area p = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_for_given_prism_l3645_364551


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3645_364518

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3645_364518


namespace NUMINAMATH_CALUDE_symmetry_line_l3645_364546

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7)^2 + (y + 4)^2 = 16
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Theorem statement
theorem symmetry_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  circle1 x₁ y₁ → circle2 x₂ y₂ →
  ∃ (x y : ℝ),
  line_of_symmetry x y ∧
  (x = (x₁ + x₂) / 2) ∧
  (y = (y₁ + y₂) / 2) ∧
  ((x - x₁)^2 + (y - y₁)^2 = (x - x₂)^2 + (y - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_line_l3645_364546


namespace NUMINAMATH_CALUDE_apps_deletion_ways_l3645_364501

-- Define the total number of applications
def total_apps : ℕ := 21

-- Define the number of applications to be deleted
def apps_to_delete : ℕ := 6

-- Define the number of special applications
def special_apps : ℕ := 6

-- Define the number of special apps to be selected
def special_apps_to_select : ℕ := 3

-- Define the number of pairs of special apps
def special_pairs : ℕ := 3

-- Theorem statement
theorem apps_deletion_ways :
  (2^special_pairs) * (Nat.choose (total_apps - special_apps) (apps_to_delete - special_apps_to_select)) = 3640 :=
sorry

end NUMINAMATH_CALUDE_apps_deletion_ways_l3645_364501


namespace NUMINAMATH_CALUDE_circle_equation_l3645_364511

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ  -- (ρ, θ)

-- Define the line l
def line_l (p : PolarCoordinate) : Prop :=
  p.1 * Real.cos p.2 + p.1 * Real.sin p.2 = 2

-- Define the point M where line l intersects the polar axis
def point_M : ℝ × ℝ := (2, 0)  -- Cartesian coordinates

-- Define the circle with OM as diameter
def circle_OM (p : PolarCoordinate) : Prop :=
  p.1 = 2 * Real.cos p.2

-- Theorem statement
theorem circle_equation (p : PolarCoordinate) :
  line_l p → circle_OM p :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3645_364511


namespace NUMINAMATH_CALUDE_grid_division_l3645_364517

theorem grid_division (n : ℕ) : 
  (∃ m : ℕ, n^2 = 7 * m ∧ m > 0) ↔ (n > 7 ∧ ∃ k : ℕ, n = 7 * k) := by
sorry

end NUMINAMATH_CALUDE_grid_division_l3645_364517


namespace NUMINAMATH_CALUDE_parabola_translation_l3645_364579

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  vertical_shift (horizontal_shift original_parabola 2) (-3)

theorem parabola_translation :
  resulting_parabola.f = fun x => (x + 2)^2 - 3 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3645_364579


namespace NUMINAMATH_CALUDE_book_area_l3645_364593

/-- The area of a rectangle with length 2 inches and width 3 inches is 6 square inches. -/
theorem book_area : 
  ∀ (length width area : ℝ), 
    length = 2 → 
    width = 3 → 
    area = length * width → 
    area = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_area_l3645_364593


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3645_364582

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3645_364582


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l3645_364532

theorem min_value_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 ≥ 2021.75 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 = 2021.75 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l3645_364532


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l3645_364590

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides * (nonagon_sides - 3)) / 2

theorem nonagon_diagonals_count : nonagon_diagonals = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l3645_364590


namespace NUMINAMATH_CALUDE_symmetry_of_point_l3645_364545

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point with respect to the origin -/
def symmetrical_to_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let P : Point := ⟨3, 2⟩
  let P' : Point := symmetrical_to_origin P
  P'.x = -3 ∧ P'.y = -2 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l3645_364545


namespace NUMINAMATH_CALUDE_tim_earnings_l3645_364599

/-- Calculates the total money earned by Tim given the number of coins received from various sources. -/
def total_money_earned (shine_pennies shine_nickels shine_dimes shine_quarters : ℕ)
                       (tip_pennies tip_nickels tip_dimes tip_half_dollars : ℕ)
                       (stranger_pennies stranger_quarters : ℕ) : ℚ :=
  let penny_value : ℚ := 1 / 100
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  let quarter_value : ℚ := 25 / 100
  let half_dollar_value : ℚ := 50 / 100

  let shine_total : ℚ := shine_pennies * penny_value + shine_nickels * nickel_value +
                         shine_dimes * dime_value + shine_quarters * quarter_value
  let tip_total : ℚ := tip_pennies * penny_value + tip_nickels * nickel_value +
                       tip_dimes * dime_value + tip_half_dollars * half_dollar_value
  let stranger_total : ℚ := stranger_pennies * penny_value + stranger_quarters * quarter_value

  shine_total + tip_total + stranger_total

/-- Theorem stating that Tim's total earnings equal $9.79 given the specified coin counts. -/
theorem tim_earnings :
  total_money_earned 4 3 13 6 15 12 7 9 10 3 = 979 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tim_earnings_l3645_364599


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3645_364525

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3645_364525


namespace NUMINAMATH_CALUDE_product_primitive_roots_congruent_one_l3645_364521

/-- Given a prime p > 3, the product of all primitive roots modulo p is congruent to 1 modulo p -/
theorem product_primitive_roots_congruent_one (p : Nat) (hp : p.Prime) (hp3 : p > 3) :
  ∃ (S : Finset Nat), 
    (∀ s ∈ S, 1 ≤ s ∧ s < p ∧ IsPrimitiveRoot s p) ∧ 
    (∀ x, 1 ≤ x ∧ x < p ∧ IsPrimitiveRoot x p → x ∈ S) ∧
    (S.prod id) % p = 1 := by
  sorry


end NUMINAMATH_CALUDE_product_primitive_roots_congruent_one_l3645_364521


namespace NUMINAMATH_CALUDE_de_length_theorem_l3645_364535

-- Define the circle Ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points on the circle
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (Ω : Circle) (A B C D E : Point) : Prop :=
  -- A and B lie on circle Ω
  (A.x - Ω.center.1)^2 + (A.y - Ω.center.2)^2 = Ω.radius^2 ∧
  (B.x - Ω.center.1)^2 + (B.y - Ω.center.2)^2 = Ω.radius^2 ∧
  -- C and D are trisection points of major arc AB
  -- (This condition is simplified for the sake of the Lean statement)
  (C.x - Ω.center.1)^2 + (C.y - Ω.center.2)^2 = Ω.radius^2 ∧
  (D.x - Ω.center.1)^2 + (D.y - Ω.center.2)^2 = Ω.radius^2 ∧
  -- E is on line AB (simplified condition)
  (E.y - A.y) * (B.x - A.x) = (E.x - A.x) * (B.y - A.y) ∧
  -- Given distances
  ((D.x - C.x)^2 + (D.y - C.y)^2) = 64 ∧  -- DC = 8
  ((D.x - B.x)^2 + (D.y - B.y)^2) = 121   -- DB = 11

-- Main theorem
theorem de_length_theorem (Ω : Circle) (A B C D E : Point) 
  (h : problem_setup Ω A B C D E) :
  ∃ (a b : ℕ), (((E.x - D.x)^2 + (E.y - D.y)^2) = a^2 * b) ∧ 
  (∀ (p : ℕ), p^2 ∣ b → p = 1) → 
  a + b = 37 := by
  sorry

end NUMINAMATH_CALUDE_de_length_theorem_l3645_364535


namespace NUMINAMATH_CALUDE_calculate_expression_l3645_364526

theorem calculate_expression : 15 * (2/3) * 45 + 15 * (1/3) * 90 = 900 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3645_364526


namespace NUMINAMATH_CALUDE_simplify_expression_l3645_364598

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3645_364598


namespace NUMINAMATH_CALUDE_system_solution_proof_l3645_364566

theorem system_solution_proof (x y : ℝ) : 
  (4 / (x^2 + y^2) + x^2 * y^2 = 5 ∧ x^4 + y^4 + 3 * x^2 * y^2 = 20) ↔ 
  ((x = Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = Real.sqrt 2 ∧ y = -Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l3645_364566


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3645_364576

-- Define the function
def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 - 18*x + 7

-- Define the derivative of the function
def f_derivative (x : ℝ) : ℝ := 6*x^2 - 12*x - 18

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 3) ↔ (f_derivative x < 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3645_364576


namespace NUMINAMATH_CALUDE_revenue_difference_is_400_l3645_364513

/-- Represents the revenue difference between making elephant and giraffe statues -/
def revenue_difference (total_jade : ℕ) (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_price : ℕ) : ℕ :=
  let elephant_jade := 2 * giraffe_jade
  let num_giraffes := total_jade / giraffe_jade
  let num_elephants := total_jade / elephant_jade
  let giraffe_revenue := num_giraffes * giraffe_price
  let elephant_revenue := num_elephants * elephant_price
  elephant_revenue - giraffe_revenue

/-- Proves that the revenue difference is $400 for the given conditions -/
theorem revenue_difference_is_400 :
  revenue_difference 1920 120 150 350 = 400 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_is_400_l3645_364513


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3645_364565

theorem simplify_fraction_product : 
  10 * (15 / 8) * (-28 / 45) * (3 / 5) = -7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3645_364565


namespace NUMINAMATH_CALUDE_eighty_sixth_word_ends_with_E_l3645_364510

-- Define the set of letters
inductive Letter : Type
| A | H | S | M | E

-- Define a permutation as a list of letters
def Permutation := List Letter

-- Define the dictionary order for permutations
def dict_order (p1 p2 : Permutation) : Prop := sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : Permutation := sorry

-- Define a function to get the last letter of a permutation
def last_letter (p : Permutation) : Letter := sorry

-- State the theorem
theorem eighty_sixth_word_ends_with_E : 
  last_letter (nth_permutation 86) = Letter.E := by sorry

end NUMINAMATH_CALUDE_eighty_sixth_word_ends_with_E_l3645_364510


namespace NUMINAMATH_CALUDE_high_school_student_distribution_l3645_364569

theorem high_school_student_distribution :
  ∀ (total juniors not_sophomores seniors freshmen sophomores : ℕ),
    total = 800 →
    juniors = (28 * total) / 100 →
    not_sophomores = (75 * total) / 100 →
    seniors = 160 →
    freshmen + sophomores + juniors + seniors = total →
    freshmen + not_sophomores = total →
    freshmen - sophomores = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_student_distribution_l3645_364569


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l3645_364514

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l3645_364514


namespace NUMINAMATH_CALUDE_frog_final_position_l3645_364530

-- Define the circle points
inductive CirclePoint
| One
| Two
| Three
| Four
| Five

-- Define the jump function
def jump (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.One => CirclePoint.Two
  | CirclePoint.Two => CirclePoint.Four
  | CirclePoint.Three => CirclePoint.Four
  | CirclePoint.Four => CirclePoint.One
  | CirclePoint.Five => CirclePoint.One

-- Define the function to perform multiple jumps
def multiJump (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | Nat.succ m => jump (multiJump start m)

-- Theorem statement
theorem frog_final_position :
  multiJump CirclePoint.Five 1995 = CirclePoint.Four := by
  sorry

end NUMINAMATH_CALUDE_frog_final_position_l3645_364530


namespace NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3645_364541

/-- The size of the multiplication table -/
def table_size : ℕ := 15

/-- The count of odd numbers from 1 to table_size -/
def odd_count : ℕ := (table_size + 1) / 2

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of odd entries in the multiplication table -/
def odd_entries : ℕ := odd_count * odd_count

/-- The fraction of odd numbers in the multiplication table -/
def odd_fraction : ℚ := odd_entries / total_entries

theorem odd_fraction_in_multiplication_table :
  odd_fraction = 64 / 225 := by
  sorry

end NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3645_364541


namespace NUMINAMATH_CALUDE_positive_root_range_l3645_364597

-- Define the function f(x) = mx² - 3x + 1
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 3 * x + 1

-- Theorem statement
theorem positive_root_range (m : ℝ) :
  (∃ x > 0, f m x = 0) ↔ m ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_positive_root_range_l3645_364597


namespace NUMINAMATH_CALUDE_tens_digit_13_2023_l3645_364506

theorem tens_digit_13_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_13_2023_l3645_364506


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3645_364505

theorem discount_percentage_proof (pants_price : ℝ) (socks_price : ℝ) (total_after_discount : ℝ) :
  pants_price = 110 →
  socks_price = 60 →
  total_after_discount = 392 →
  let original_total := 4 * pants_price + 2 * socks_price
  let discount_amount := original_total - total_after_discount
  let discount_percentage := (discount_amount / original_total) * 100
  discount_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3645_364505


namespace NUMINAMATH_CALUDE_calculation_result_l3645_364519

theorem calculation_result : (25 * 8 + 1 / (5/7)) / (2014 - 201.4 * 2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3645_364519


namespace NUMINAMATH_CALUDE_parallelogram_values_l3645_364503

/-- Represents a parallelogram EFGH with given side lengths and area formula -/
structure Parallelogram where
  x : ℝ
  y : ℝ
  ef : ℝ := 5 * x + 7
  fg : ℝ := 4 * y + 1
  gh : ℝ := 27
  he : ℝ := 19
  area : ℝ := 2 * x^2 + y^2 + 5 * x * y + 3

/-- Theorem stating the values of x, y, and area for the given parallelogram -/
theorem parallelogram_values (p : Parallelogram) :
  p.x = 4 ∧ p.y = 4.5 ∧ p.area = 145.25 := by sorry

end NUMINAMATH_CALUDE_parallelogram_values_l3645_364503


namespace NUMINAMATH_CALUDE_complex_multiply_i_l3645_364539

theorem complex_multiply_i (i : ℂ) : i * i = -1 → (1 + i) * i = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiply_i_l3645_364539


namespace NUMINAMATH_CALUDE_probability_both_classes_l3645_364512

-- Define the total number of students
def total_students : ℕ := 40

-- Define the number of students in Mandarin
def mandarin_students : ℕ := 30

-- Define the number of students in German
def german_students : ℕ := 35

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the theorem
theorem probability_both_classes : 
  let students_both := mandarin_students + german_students - total_students
  let students_only_mandarin := mandarin_students - students_both
  let students_only_german := german_students - students_both
  let total_ways := choose total_students 2
  let ways_not_both := choose students_only_mandarin 2 + choose students_only_german 2
  (total_ways - ways_not_both) / total_ways = 145 / 156 := by
sorry

end NUMINAMATH_CALUDE_probability_both_classes_l3645_364512


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3645_364580

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3645_364580


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3645_364563

theorem three_numbers_sum (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2744 → a + b + c = 84 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3645_364563


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l3645_364538

theorem x_positive_sufficient_not_necessary_for_abs_x_positive :
  (∀ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l3645_364538


namespace NUMINAMATH_CALUDE_hcf_36_84_l3645_364507

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_36_84_l3645_364507


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3645_364596

theorem triangle_inequalities (a b c s : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_s : s = a + b + c) : 
  ((13 / 27 : ℝ) * s^2 ≤ a^2 + b^2 + c^2 + 4 / s * a * b * c ∧ 
   a^2 + b^2 + c^2 + 4 / s * a * b * c < s^2 / 2) ∧
  (s^2 / 4 < a * b + b * c + c * a - 2 / s * a * b * c ∧ 
   a * b + b * c + c * a - 2 / s * a * b * c < (7 / 27 : ℝ) * s^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3645_364596


namespace NUMINAMATH_CALUDE_folded_rectangle_ratio_l3645_364572

/-- Given a rectangle that when folded along its diagonal forms a non-convex pentagon
    with an area 7/10 of the original rectangle's area, prove that the ratio of the
    longer side to the shorter side of the rectangle is √5. -/
theorem folded_rectangle_ratio (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  let rectangle_area := a * b
  let pentagon_area := (7 / 10) * rectangle_area
  let longer_side := max a b
  let shorter_side := min a b
  (pentagon_area = (7 / 10) * rectangle_area) →
  (longer_side / shorter_side = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_ratio_l3645_364572


namespace NUMINAMATH_CALUDE_puppies_adopted_per_day_l3645_364558

theorem puppies_adopted_per_day :
  ∀ (initial_puppies additional_puppies total_days : ℕ),
    initial_puppies = 3 →
    additional_puppies = 3 →
    total_days = 2 →
    (initial_puppies + additional_puppies) / total_days = 3 :=
by
  sorry

#check puppies_adopted_per_day

end NUMINAMATH_CALUDE_puppies_adopted_per_day_l3645_364558


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l3645_364591

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → x * y = 0) ∧
  ∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l3645_364591


namespace NUMINAMATH_CALUDE_inequality_proof_l3645_364574

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 3*(a*b + b*c + c*a) - 2 ∧
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 6 - 5*(a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3645_364574


namespace NUMINAMATH_CALUDE_total_money_is_correct_l3645_364547

/-- Calculates the total amount of money in Euros given the specified coins and bills and the conversion rate. -/
def total_money_in_euros : ℝ :=
  let pennies : ℕ := 9
  let nickels : ℕ := 4
  let dimes : ℕ := 3
  let quarters : ℕ := 7
  let half_dollars : ℕ := 5
  let one_dollar_coins : ℕ := 2
  let two_dollar_bills : ℕ := 1
  
  let penny_value : ℝ := 0.01
  let nickel_value : ℝ := 0.05
  let dime_value : ℝ := 0.10
  let quarter_value : ℝ := 0.25
  let half_dollar_value : ℝ := 0.50
  let one_dollar_value : ℝ := 1.00
  let two_dollar_value : ℝ := 2.00
  
  let usd_to_euro_rate : ℝ := 0.85
  
  let total_usd : ℝ := 
    pennies * penny_value +
    nickels * nickel_value +
    dimes * dime_value +
    quarters * quarter_value +
    half_dollars * half_dollar_value +
    one_dollar_coins * one_dollar_value +
    two_dollar_bills * two_dollar_value
  
  total_usd * usd_to_euro_rate

/-- Theorem stating that the total amount of money in Euros is equal to 7.514. -/
theorem total_money_is_correct : total_money_in_euros = 7.514 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_correct_l3645_364547


namespace NUMINAMATH_CALUDE_chess_match_probability_l3645_364523

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
sorry

end NUMINAMATH_CALUDE_chess_match_probability_l3645_364523


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l3645_364524

/-- An isosceles trapezoid with bases a and b, and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Point P on the axis of symmetry of the trapezoid -/
structure PointP (t : IsoscelesTrapezoid) where
  x : ℝ  -- Distance from P to one base
  y : ℝ  -- Distance from P to the other base
  sum_eq_h : x + y = t.h
  product_eq_ab_div_4 : x * y = t.a * t.b / 4

theorem isosceles_trapezoid_right_angle_point 
  (t : IsoscelesTrapezoid) : 
  (∃ p : PointP t, True) ↔ t.h^2 ≥ t.a * t.b :=
sorry

theorem isosceles_trapezoid_point_distances 
  (t : IsoscelesTrapezoid) 
  (h : t.h^2 ≥ t.a * t.b) :
  ∃ p : PointP t, 
    (p.x = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2) ∨
    (p.x = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2) :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l3645_364524


namespace NUMINAMATH_CALUDE_geometric_series_problem_l3645_364522

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 18
  let r₁ : ℝ := 6 / 18
  let S₁ : ℝ := a₁ / (1 - r₁)
  let r₂ : ℝ := (6 + n) / 18
  let S₂ : ℝ := a₁ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l3645_364522


namespace NUMINAMATH_CALUDE_converse_opposites_sum_zero_l3645_364555

theorem converse_opposites_sum_zero :
  ∀ x y : ℝ, (x = -y) → (x + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_converse_opposites_sum_zero_l3645_364555


namespace NUMINAMATH_CALUDE_rectangle_short_side_l3645_364534

/-- Proves that for a rectangle with perimeter 38 cm and long side 12 cm, the short side is 7 cm. -/
theorem rectangle_short_side (perimeter long_side short_side : ℝ) : 
  perimeter = 38 ∧ long_side = 12 ∧ perimeter = 2 * long_side + 2 * short_side → short_side = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_short_side_l3645_364534


namespace NUMINAMATH_CALUDE_power_sum_equals_123_l3645_364543

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_equals_123 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_123_l3645_364543


namespace NUMINAMATH_CALUDE_mike_books_before_sale_l3645_364533

def books_before_sale (books_bought books_after : ℕ) : ℕ :=
  books_after - books_bought

theorem mike_books_before_sale :
  books_before_sale 21 56 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mike_books_before_sale_l3645_364533


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3645_364542

/-- Given three square regions A, B, and C with perimeters 16, 20, and 40 units respectively,
    the ratio of the area of region B to the area of region C is 1/4 -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) (pb : 4 * b = 20) (pc : 4 * c = 40) :
  (b ^ 2) / (c ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3645_364542
