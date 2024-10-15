import Mathlib

namespace NUMINAMATH_CALUDE_shyne_garden_theorem_l646_64646

/-- Represents the number of plants that can be grown from one packet of seeds for each type of plant. -/
structure PlantsPerPacket where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the number of seed packets bought for each type of plant. -/
structure PacketsBought where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the percentage of plants that can be grown in each season. -/
structure PlantingPercentages where
  spring_eggplants_peas : ℚ
  summer_sunflowers_cucumbers : ℚ
  both_seasons_tomatoes : ℚ

/-- Calculates the total number of plants Shyne can potentially grow across spring and summer. -/
def totalPlants (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) : ℕ :=
  sorry

/-- Theorem stating that Shyne can potentially grow 366 plants across spring and summer. -/
theorem shyne_garden_theorem (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) :
  plantsPerPacket.eggplants = 14 ∧
  plantsPerPacket.sunflowers = 10 ∧
  plantsPerPacket.tomatoes = 16 ∧
  plantsPerPacket.peas = 20 ∧
  plantsPerPacket.cucumbers = 18 ∧
  packetsBought.eggplants = 6 ∧
  packetsBought.sunflowers = 8 ∧
  packetsBought.tomatoes = 7 ∧
  packetsBought.peas = 9 ∧
  packetsBought.cucumbers = 5 ∧
  percentages.spring_eggplants_peas = 3/5 ∧
  percentages.summer_sunflowers_cucumbers = 7/10 ∧
  percentages.both_seasons_tomatoes = 4/5 →
  totalPlants plantsPerPacket packetsBought percentages = 366 :=
by
  sorry

end NUMINAMATH_CALUDE_shyne_garden_theorem_l646_64646


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l646_64601

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (2 * a * (-1) - b * 2 + 2 = 0) → -- Line passes through circle center (-1, 2)
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (2 * a' * (-1) - b' * 2 + 2 = 0) → 
    (1/a + 1/b) ≤ (1/a' + 1/b')) →
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l646_64601


namespace NUMINAMATH_CALUDE_unique_quadratic_m_l646_64642

def is_quadratic_coefficient (m : ℝ) : Prop :=
  |m| = 2 ∧ m - 2 ≠ 0

theorem unique_quadratic_m :
  ∃! m : ℝ, is_quadratic_coefficient m ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_m_l646_64642


namespace NUMINAMATH_CALUDE_number_line_real_bijection_l646_64669

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define a point on the number line
def Point : Type := NumberLine

-- State the theorem
theorem number_line_real_bijection : 
  ∃ f : Point → ℝ, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_number_line_real_bijection_l646_64669


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l646_64686

theorem speeding_ticket_percentage
  (exceed_limit_percent : ℝ)
  (no_ticket_percent : ℝ)
  (h1 : exceed_limit_percent = 14.285714285714285)
  (h2 : no_ticket_percent = 30) :
  (1 - no_ticket_percent / 100) * exceed_limit_percent = 10 :=
by sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l646_64686


namespace NUMINAMATH_CALUDE_seniors_in_three_sports_l646_64619

theorem seniors_in_three_sports 
  (total_seniors : ℕ) 
  (football : ℕ) 
  (baseball : ℕ) 
  (football_lacrosse : ℕ) 
  (baseball_football : ℕ) 
  (baseball_lacrosse : ℕ) 
  (h1 : total_seniors = 85)
  (h2 : football = 74)
  (h3 : baseball = 26)
  (h4 : football_lacrosse = 17)
  (h5 : baseball_football = 18)
  (h6 : baseball_lacrosse = 13)
  : ∃ (n : ℕ), n = 11 ∧ 
    total_seniors = football + baseball + 2*n - baseball_football - football_lacrosse - baseball_lacrosse + n :=
by sorry

end NUMINAMATH_CALUDE_seniors_in_three_sports_l646_64619


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l646_64616

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 2

-- State the theorem
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 0, StrictMonoOn f (Set.Ioo (-2 : ℝ) 0) := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l646_64616


namespace NUMINAMATH_CALUDE_equation_solvability_l646_64635

theorem equation_solvability (n : ℕ) (hn : Odd n) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 4 / n = 1 / x + 1 / y) ↔
  (∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∃ k : ℕ, d = 4 * k + 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solvability_l646_64635


namespace NUMINAMATH_CALUDE_tan_two_alpha_l646_64668

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l646_64668


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l646_64623

theorem arctan_sum_special_case (a b : ℝ) :
  a = 2/3 →
  (a + 1) * (b + 1) = 3 →
  Real.arctan a + Real.arctan b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l646_64623


namespace NUMINAMATH_CALUDE_ruler_measurement_l646_64662

/-- Represents a ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Checks if a length can be measured using the given ruler -/
def can_measure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (coeffs : List ℤ), length = (List.zip r.marks coeffs).foldl (λ acc (m, c) => acc + m * c) 0

theorem ruler_measurement (r : Ruler) (h : r.marks = [0, 7, 11]) :
  (can_measure r 8) ∧ (can_measure r 5) := by
  sorry

end NUMINAMATH_CALUDE_ruler_measurement_l646_64662


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l646_64648

/-- The cost of ingredients for Martha's lasagna --/
theorem martha_lasagna_cost : 
  let cheese_weight : ℝ := 1.5
  let meat_weight : ℝ := 0.5
  let cheese_price : ℝ := 6
  let meat_price : ℝ := 8
  cheese_weight * cheese_price + meat_weight * meat_price = 13 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_cost_l646_64648


namespace NUMINAMATH_CALUDE_meeting_selection_ways_l646_64613

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of managers -/
def total_managers : ℕ := 7

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of managers who cannot attend together -/
def incompatible_managers : ℕ := 2

/-- The number of ways to select managers for the meeting -/
def select_managers : ℕ :=
  choose (total_managers - incompatible_managers) meeting_size +
  incompatible_managers * choose (total_managers - 1) (meeting_size - 1)

theorem meeting_selection_ways :
  select_managers = 25 := by sorry

end NUMINAMATH_CALUDE_meeting_selection_ways_l646_64613


namespace NUMINAMATH_CALUDE_divisible_by_two_and_three_l646_64605

theorem divisible_by_two_and_three (n : ℕ) : 
  (∃ (k : ℕ), k = 33 ∧ k = (n.div 6).succ) ↔ n = 204 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_two_and_three_l646_64605


namespace NUMINAMATH_CALUDE_parallelogram_double_reflection_l646_64602

-- Define the parallelogram vertices
def A : ℝ × ℝ := (3, 6)
def B : ℝ × ℝ := (5, 10)
def C : ℝ × ℝ := (7, 6)
def D : ℝ × ℝ := (5, 2)

-- Define the reflection functions
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 2)        -- Translate up by 2

-- Theorem statement
theorem parallelogram_double_reflection :
  reflect_y_eq_x_plus_2 (reflect_x_axis D) = (-4, 7) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_double_reflection_l646_64602


namespace NUMINAMATH_CALUDE_coin_probability_l646_64628

theorem coin_probability (p q : ℝ) : 
  q = 1 - p →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l646_64628


namespace NUMINAMATH_CALUDE_pencils_bought_l646_64639

-- Define the cost of a single pencil and notebook
variable (P N : ℝ)

-- Define the number of pencils in the second case
variable (X : ℝ)

-- Conditions from the problem
axiom cost_condition1 : 96 * P + 24 * N = 520
axiom cost_condition2 : X * P + 4 * N = 60
axiom sum_condition : P + N = 15.512820512820513

-- Theorem to prove
theorem pencils_bought : X = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_bought_l646_64639


namespace NUMINAMATH_CALUDE_range_of_valid_m_l646_64636

/-- The set A as defined in the problem -/
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2) 2, y = x^2 - (3/2)*x + 1}

/-- The set B as defined in the problem -/
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

/-- The range of values for m that satisfies the condition A ⊆ B -/
def valid_m : Set ℝ := {m | A ⊆ B m}

/-- Theorem stating that the range of valid m is (-∞, -9/16] ∪ [3, +∞) -/
theorem range_of_valid_m : valid_m = Set.Iic (-9/16) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_range_of_valid_m_l646_64636


namespace NUMINAMATH_CALUDE_spencer_total_distance_l646_64654

/-- The total distance Spencer walked throughout the day -/
def total_distance (d1 d2 d3 d4 d5 d6 d7 : ℝ) : ℝ :=
  d1 + d2 + d3 + d4 + d5 + d6 + d7

/-- Theorem: Given Spencer's walking distances, the total distance is 8.6 miles -/
theorem spencer_total_distance :
  total_distance 1.2 0.6 0.9 1.7 2.1 1.3 0.8 = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_spencer_total_distance_l646_64654


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l646_64634

/-- The length of the path traveled by a point on a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (Real.pi * r / 2)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l646_64634


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l646_64699

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations for perpendicular and parallel
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : perpendicular l α)
  (h3 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l646_64699


namespace NUMINAMATH_CALUDE_bouquet_cost_45_l646_64622

/-- The cost of a bouquet of lilies, given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  30 * (n : ℚ) / 18

theorem bouquet_cost_45 : bouquet_cost 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_45_l646_64622


namespace NUMINAMATH_CALUDE_jacob_pencils_l646_64631

theorem jacob_pencils (total : ℕ) (zain_monday : ℕ) (zain_tuesday : ℕ) : 
  total = 21 →
  zain_monday + zain_tuesday + (2 * zain_monday + zain_tuesday) / 3 = total →
  (2 * zain_monday + zain_tuesday) / 3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_jacob_pencils_l646_64631


namespace NUMINAMATH_CALUDE_omega_set_classification_l646_64680

-- Define the concept of an Ω set
def is_omega_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1}
def set2 : Set (ℝ × ℝ) := {p | p.2 = (p.1 - 1) / Real.exp p.1}
def set3 : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (1 - p.1^2)}
def set4 : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2*p.1 + 2}
def set5 : Set (ℝ × ℝ) := {p | p.2 = Real.cos p.1 + Real.sin p.1}

-- State the theorem
theorem omega_set_classification :
  (¬ is_omega_set set1) ∧
  (is_omega_set set2) ∧
  (is_omega_set set3) ∧
  (¬ is_omega_set set4) ∧
  (is_omega_set set5) := by
  sorry

end NUMINAMATH_CALUDE_omega_set_classification_l646_64680


namespace NUMINAMATH_CALUDE_tangent_lines_problem_l646_64664

theorem tangent_lines_problem (num_not_enclosed : ℕ) (lines_less_than_30 : ℕ) :
  num_not_enclosed = 68 →
  lines_less_than_30 = 4 →
  ∃ (num_tangent_lines : ℕ),
    num_tangent_lines = 30 - lines_less_than_30 ∧
    num_tangent_lines * 2 = num_not_enclosed :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_problem_l646_64664


namespace NUMINAMATH_CALUDE_cone_surface_area_l646_64660

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : ℝ) (h : ℝ) : 
  (r * r * π + r * h * π = 16 * π / 9) →
  (h * h + r * r = 2 * 2) →
  (2 * π * r = 4 * π / 3) →
  (r * h * π = 4 * π / 3) →
  (r * r * π + r * h * π = 16 * π / 9) :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l646_64660


namespace NUMINAMATH_CALUDE_deepak_age_l646_64678

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l646_64678


namespace NUMINAMATH_CALUDE_max_intersection_points_l646_64609

/-- Represents a line in the plane -/
structure Line :=
  (id : ℕ)

/-- The set of all lines -/
def all_lines : Finset Line := sorry

/-- The set of lines that are parallel to each other -/
def parallel_lines : Finset Line := sorry

/-- The set of lines that pass through point B -/
def point_b_lines : Finset Line := sorry

/-- A point of intersection between two lines -/
structure IntersectionPoint :=
  (l1 : Line)
  (l2 : Line)

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem max_intersection_points :
  (∀ l ∈ all_lines, l.id ≤ 150) →
  (∀ l ∈ all_lines, ∀ m ∈ all_lines, l ≠ m → l.id ≠ m.id) →
  (Finset.card all_lines = 150) →
  (∀ n : ℕ, n > 0 → parallel_lines.card = 100) →
  (∀ n : ℕ, n > 0 → point_b_lines.card = 50) →
  (∀ l ∈ parallel_lines, ∀ m ∈ parallel_lines, l ≠ m → ¬∃ p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ point_b_lines, ∀ m ∈ point_b_lines, l ≠ m → ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ parallel_lines, ∀ m ∈ point_b_lines, ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  Finset.card intersection_points = 5001 :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l646_64609


namespace NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l646_64689

theorem log_50_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < b ∧ a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l646_64689


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l646_64611

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l646_64611


namespace NUMINAMATH_CALUDE_a_range_theorem_l646_64645

theorem a_range_theorem (a : ℝ) : 
  (∀ x : ℝ, a^2 * x - 2*(a - x - 4) < 0) ↔ -2 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l646_64645


namespace NUMINAMATH_CALUDE_complex_z_value_l646_64633

-- Define the operation for 2x2 matrices
def matrixOp (a b c d : ℂ) : ℂ := a * d - b * c

-- Theorem statement
theorem complex_z_value (z : ℂ) :
  matrixOp z (1 - Complex.I) (1 + Complex.I) 1 = Complex.I →
  z = 2 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_z_value_l646_64633


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l646_64682

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l646_64682


namespace NUMINAMATH_CALUDE_circle_configuration_radius_l646_64629

/-- Given a configuration of three circles C, D, and E, prove that the radius of circle D is 4√15 - 14 -/
theorem circle_configuration_radius (C D E : ℝ → ℝ → Prop) (A B F : ℝ × ℝ) :
  (∀ x y, C x y ↔ (x - 0)^2 + (y - 0)^2 = 4) →  -- Circle C with radius 2 centered at origin
  (∃ x y, C x y ∧ D x y) →  -- D is internally tangent to C
  (∃ x y, C x y ∧ E x y) →  -- E is tangent to C
  (∃ x y, D x y ∧ E x y) →  -- E is externally tangent to D
  (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ F = (2*t - 1, 0) ∧ E (2*t - 1) 0) →  -- E is tangent to AB at F
  (∀ x y z w, D x y ∧ E z w → (x - z)^2 + (y - w)^2 = (3*r)^2 - r^2) →  -- Radius of D is 3 times radius of E
  (∃ r_D, ∀ x y, D x y ↔ (x - 0)^2 + (y - 0)^2 = r_D^2 ∧ r_D = 4*Real.sqrt 15 - 14) :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_radius_l646_64629


namespace NUMINAMATH_CALUDE_sticker_collection_value_l646_64671

theorem sticker_collection_value (total_stickers : ℕ) (sample_size : ℕ) (sample_value : ℕ) 
  (h1 : total_stickers = 18)
  (h2 : sample_size = 6)
  (h3 : sample_value = 24) :
  (total_stickers : ℚ) * (sample_value : ℚ) / (sample_size : ℚ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_sticker_collection_value_l646_64671


namespace NUMINAMATH_CALUDE_superhero_speed_in_miles_per_hour_l646_64666

-- Define the superhero's speed in kilometers per minute
def superhero_speed_km_per_min : ℝ := 1000

-- Define the conversion factor from kilometers to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_miles_per_hour :
  superhero_speed_km_per_min * minutes_per_hour * km_to_miles = 36000 := by
  sorry

end NUMINAMATH_CALUDE_superhero_speed_in_miles_per_hour_l646_64666


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l646_64620

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l646_64620


namespace NUMINAMATH_CALUDE_quadrilateral_iff_interior_exterior_sum_equal_l646_64630

/-- A polygon has 4 sides if and only if the sum of its interior angles is equal to the sum of its exterior angles. -/
theorem quadrilateral_iff_interior_exterior_sum_equal (n : ℕ) : n = 4 ↔ (n - 2) * 180 = 360 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_iff_interior_exterior_sum_equal_l646_64630


namespace NUMINAMATH_CALUDE_smallest_x_value_l646_64665

theorem smallest_x_value (x y : ℕ+) (h : (4 : ℚ) / 5 = y / (200 + x)) : 
  ∀ z : ℕ+, (4 : ℚ) / 5 = (y : ℚ) / (200 + z) → x ≤ z :=
by sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_smallest_x_value_l646_64665


namespace NUMINAMATH_CALUDE_parabola_coefficients_l646_64657

/-- A parabola with given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_property : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_property : point_y = a * point_x^2 + b * point_x + c
  symmetry_property : b = -2 * a * vertex_x

/-- The theorem stating the values of a, b, and c for the given parabola -/
theorem parabola_coefficients (p : Parabola)
  (h_vertex : p.vertex_x = 2 ∧ p.vertex_y = 4)
  (h_point : p.point_x = 0 ∧ p.point_y = 5) :
  p.a = 1/4 ∧ p.b = -1 ∧ p.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l646_64657


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l646_64667

/-- Represents the daily sales and profit of mooncakes -/
structure MooncakeSales where
  initialSales : ℕ
  initialProfit : ℕ
  priceReduction : ℕ
  salesIncrease : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (s : MooncakeSales) (x : ℕ) : ℕ :=
  (s.initialProfit - x) * (s.initialSales + (s.salesIncrease * x) / s.priceReduction)

/-- Theorem stating that a 6 yuan price reduction achieves the target profit -/
theorem optimal_price_reduction (s : MooncakeSales) 
    (h1 : s.initialSales = 80)
    (h2 : s.initialProfit = 30)
    (h3 : s.priceReduction = 5)
    (h4 : s.salesIncrease = 20)
    (h5 : s.targetProfit = 2496) :
    dailyProfit s 6 = s.targetProfit := by
  sorry

#check optimal_price_reduction

end NUMINAMATH_CALUDE_optimal_price_reduction_l646_64667


namespace NUMINAMATH_CALUDE_sum_floor_series_l646_64661

theorem sum_floor_series (n : ℕ+) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋) = n := by sorry

end NUMINAMATH_CALUDE_sum_floor_series_l646_64661


namespace NUMINAMATH_CALUDE_joe_original_cans_l646_64614

/-- Represents the number of rooms that can be painted with a given number of paint cans -/
def rooms_paintable (cans : ℕ) : ℕ := sorry

/-- The number of rooms Joe could initially paint -/
def initial_rooms : ℕ := 40

/-- The number of rooms Joe could paint after losing cans -/
def remaining_rooms : ℕ := 32

/-- The number of cans Joe lost -/
def lost_cans : ℕ := 2

theorem joe_original_cans :
  ∃ (original_cans : ℕ),
    rooms_paintable original_cans = initial_rooms ∧
    rooms_paintable (original_cans - lost_cans) = remaining_rooms ∧
    original_cans = 10 := by sorry

end NUMINAMATH_CALUDE_joe_original_cans_l646_64614


namespace NUMINAMATH_CALUDE_occupancy_is_75_percent_l646_64676

/-- Represents an apartment complex -/
structure ApartmentComplex where
  buildings : Nat
  studio_per_building : Nat
  two_person_per_building : Nat
  four_person_per_building : Nat
  current_occupancy : Nat

/-- Calculate the maximum occupancy of an apartment complex -/
def max_occupancy (complex : ApartmentComplex) : Nat :=
  complex.buildings * (complex.studio_per_building + 2 * complex.two_person_per_building + 4 * complex.four_person_per_building)

/-- Calculate the occupancy percentage of an apartment complex -/
def occupancy_percentage (complex : ApartmentComplex) : Rat :=
  (complex.current_occupancy : Rat) / (max_occupancy complex)

/-- The main theorem stating that the occupancy percentage is 75% -/
theorem occupancy_is_75_percent (complex : ApartmentComplex) 
  (h1 : complex.buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.current_occupancy = 210) :
  occupancy_percentage complex = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_occupancy_is_75_percent_l646_64676


namespace NUMINAMATH_CALUDE_total_frog_eyes_l646_64688

/-- The number of frogs in the pond -/
def total_frogs : ℕ := 6

/-- The number of eyes for Species A frogs -/
def eyes_species_a : ℕ := 2

/-- The number of eyes for Species B frogs -/
def eyes_species_b : ℕ := 3

/-- The number of eyes for Species C frogs -/
def eyes_species_c : ℕ := 4

/-- The number of Species A frogs -/
def frogs_species_a : ℕ := 2

/-- The number of Species B frogs -/
def frogs_species_b : ℕ := 1

/-- The number of Species C frogs -/
def frogs_species_c : ℕ := 3

theorem total_frog_eyes : 
  frogs_species_a * eyes_species_a + 
  frogs_species_b * eyes_species_b + 
  frogs_species_c * eyes_species_c = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_frog_eyes_l646_64688


namespace NUMINAMATH_CALUDE_tan_five_pi_four_l646_64672

theorem tan_five_pi_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_four_l646_64672


namespace NUMINAMATH_CALUDE_children_on_airplane_l646_64608

/-- Proves that the number of children on an airplane is 20 given specific conditions --/
theorem children_on_airplane (total_passengers : ℕ) (num_men : ℕ) :
  total_passengers = 80 →
  num_men = 30 →
  ∃ (num_women num_children : ℕ),
    num_women = num_men ∧
    num_children = total_passengers - (num_men + num_women) ∧
    num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_on_airplane_l646_64608


namespace NUMINAMATH_CALUDE_cubic_gt_27_implies_abs_gt_3_but_not_conversely_l646_64604

theorem cubic_gt_27_implies_abs_gt_3_but_not_conversely :
  (∀ x : ℝ, x^3 > 27 → |x| > 3) ∧
  (∃ x : ℝ, |x| > 3 ∧ x^3 ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_gt_27_implies_abs_gt_3_but_not_conversely_l646_64604


namespace NUMINAMATH_CALUDE_unique_congruent_integer_l646_64600

theorem unique_congruent_integer (h : ∃ m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7]) :
  ∃! m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7] ∧ m = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_congruent_integer_l646_64600


namespace NUMINAMATH_CALUDE_projection_onto_orthogonal_vector_l646_64644

/-- Given orthogonal vectors a and b in R^2, and the projection of (4, -2) onto a,
    prove that the projection of (4, -2) onto b is (24/5, -2/5). -/
theorem projection_onto_orthogonal_vector 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_proj_a : (4 : ℝ) * a.1 + (-2 : ℝ) * a.2 = (-4/5 : ℝ) * (a.1^2 + a.2^2)) :
  (4 : ℝ) * b.1 + (-2 : ℝ) * b.2 = (24/5 : ℝ) * (b.1^2 + b.2^2) :=
by sorry

end NUMINAMATH_CALUDE_projection_onto_orthogonal_vector_l646_64644


namespace NUMINAMATH_CALUDE_contrapositive_true_l646_64650

theorem contrapositive_true : 
  (∀ x : ℝ, (x^2 ≤ 0 → x ≥ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_true_l646_64650


namespace NUMINAMATH_CALUDE_restaurant_earnings_l646_64673

theorem restaurant_earnings : 
  let meals_1 := 10
  let price_1 := 8
  let meals_2 := 5
  let price_2 := 10
  let meals_3 := 20
  let price_3 := 4
  meals_1 * price_1 + meals_2 * price_2 + meals_3 * price_3 = 210 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_earnings_l646_64673


namespace NUMINAMATH_CALUDE_new_shipment_bears_l646_64679

theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : 
  initial_stock = 6 → bears_per_shelf = 6 → num_shelves = 4 → 
  num_shelves * bears_per_shelf - initial_stock = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l646_64679


namespace NUMINAMATH_CALUDE_compound_interest_proof_l646_64670

/-- The compound interest rate that turns $1200 into $1348.32 in 2 years with annual compounding -/
def compound_interest_rate : ℝ :=
  0.06

theorem compound_interest_proof (initial_sum final_sum : ℝ) (years : ℕ) :
  initial_sum = 1200 →
  final_sum = 1348.32 →
  years = 2 →
  final_sum = initial_sum * (1 + compound_interest_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l646_64670


namespace NUMINAMATH_CALUDE_min_value_theorem_l646_64621

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 3*x + 2*y + y/x ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l646_64621


namespace NUMINAMATH_CALUDE_dot_only_count_l646_64691

/-- Represents the number of letters in an alphabet with specific characteristics. -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  line_only : ℕ
  dot_only : ℕ

/-- Theorem stating that in an alphabet with given properties, 
    the number of letters containing only a dot is 3. -/
theorem dot_only_count (α : Alphabet) 
  (h_total : α.total = 40)
  (h_dot_and_line : α.dot_and_line = 13)
  (h_line_only : α.line_only = 24)
  (h_all_covered : α.total = α.dot_and_line + α.line_only + α.dot_only) :
  α.dot_only = 3 := by
  sorry

end NUMINAMATH_CALUDE_dot_only_count_l646_64691


namespace NUMINAMATH_CALUDE_not_circle_iff_a_eq_zero_l646_64683

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The condition for the equation to represent a circle -/
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the equation does not represent a circle iff a = 0 -/
theorem not_circle_iff_a_eq_zero (a : ℝ) :
  ¬(is_circle a) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_not_circle_iff_a_eq_zero_l646_64683


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l646_64696

/-- The trajectory of point M(x,y) satisfying the distance condition -/
def trajectory_equation (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2)^(1/2) = |x + 3| + 1

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_is_parabola (x y : ℝ) :
  trajectory_equation x y → y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l646_64696


namespace NUMINAMATH_CALUDE_multiple_with_specific_remainder_l646_64612

theorem multiple_with_specific_remainder (x : ℕ) (hx : x > 0) 
  (hx_rem : x % 9 = 5) : 
  (∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 2) ∧ 
  (∀ k : ℕ, k > 0 → (k * x) % 9 = 2 → k ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_specific_remainder_l646_64612


namespace NUMINAMATH_CALUDE_interest_calculation_l646_64692

/-- Calculates the total interest earned from two investments -/
def total_interest (total_investment : ℚ) (rate1 rate2 : ℚ) (amount1 : ℚ) : ℚ :=
  let amount2 := total_investment - amount1
  amount1 * rate1 + amount2 * rate2

/-- Proves that the total interest earned is $490 given the specified conditions -/
theorem interest_calculation :
  let total_investment : ℚ := 8000
  let rate1 : ℚ := 8 / 100
  let rate2 : ℚ := 5 / 100
  let amount1 : ℚ := 3000
  total_interest total_investment rate1 rate2 amount1 = 490 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l646_64692


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l646_64610

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 15*n + 56 ≤ 0 → n ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 15*7 + 56 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ n : ℤ, n < 7 → n^2 - 15*n + 56 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l646_64610


namespace NUMINAMATH_CALUDE_simplify_expression_l646_64651

theorem simplify_expression (a : ℝ) : (1 + a) * (1 - a) + a * (a - 2) = 1 - 2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l646_64651


namespace NUMINAMATH_CALUDE_sum_after_100_operations_l646_64626

def initial_sequence : List ℕ := [2, 11, 8, 9]

def operation (seq : List ℤ) : List ℤ :=
  seq ++ (seq.zip (seq.tail!)).map (fun (a, b) => b - a)

def sum_after_n_operations (n : ℕ) : ℤ :=
  30 + 7 * n

theorem sum_after_100_operations :
  sum_after_n_operations 100 = 730 :=
sorry

end NUMINAMATH_CALUDE_sum_after_100_operations_l646_64626


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l646_64655

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water
  (current_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : current_speed = 5)
  (h2 : downstream_distance = 5)
  (h3 : downstream_time = 1/5) :
  let downstream_speed := (boat_speed : ℝ) + current_speed
  downstream_distance = downstream_speed * downstream_time →
  boat_speed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l646_64655


namespace NUMINAMATH_CALUDE_m_range_l646_64698

/-- The proposition p: The solution set of the inequality |x|+|x-1| > m is R -/
def p (m : ℝ) : Prop :=
  ∀ x, |x| + |x - 1| > m

/-- The proposition q: f(x)=(5-2m)^x is an increasing function -/
def q (m : ℝ) : Prop :=
  ∀ x y, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- The range of m given the conditions -/
theorem m_range :
  ∃ m, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l646_64698


namespace NUMINAMATH_CALUDE_simplify_exponents_l646_64607

theorem simplify_exponents (t : ℝ) : (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l646_64607


namespace NUMINAMATH_CALUDE_max_square_partitions_l646_64603

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFencing : ℕ := 2100

/-- Calculates the number of square partitions given the side length of each square -/
def numPartitions (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide) * (field.length / squareSide)

/-- Calculates the required internal fencing for given partitions -/
def requiredFencing (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide - 1) * field.length + 
  (field.length / squareSide - 1) * field.width

/-- Theorem stating the maximum number of square partitions -/
theorem max_square_partitions (field : FieldDimensions) 
  (h1 : field.width = 30) 
  (h2 : field.length = 45) : 
  (∃ (squareSide : ℕ), 
    numPartitions field squareSide = 75 ∧ 
    requiredFencing field squareSide ≤ availableFencing ∧
    ∀ (otherSide : ℕ), 
      requiredFencing field otherSide ≤ availableFencing → 
      numPartitions field otherSide ≤ 75) :=
  sorry

#check max_square_partitions

end NUMINAMATH_CALUDE_max_square_partitions_l646_64603


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l646_64653

def polynomial_terms (n : ℕ) : ℕ := Nat.choose (n + 4 - 1) (4 - 1)

theorem simplified_expression_terms :
  polynomial_terms 5 = 56 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l646_64653


namespace NUMINAMATH_CALUDE_sides_divisible_by_three_l646_64658

/-- A convex polygon divided into triangles by non-intersecting diagonals. -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon. -/
  sides : ℕ
  /-- The number of triangles in the triangulation. -/
  triangles : ℕ
  /-- The property that each vertex is a vertex of an odd number of triangles. -/
  odd_vertex_property : Bool

/-- 
Theorem: If a convex polygon is divided into triangles by non-intersecting diagonals,
and each vertex of the polygon is a vertex of an odd number of these triangles,
then the number of sides of the polygon is divisible by 3.
-/
theorem sides_divisible_by_three (p : TriangulatedPolygon) 
  (h : p.odd_vertex_property = true) : 
  ∃ k : ℕ, p.sides = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sides_divisible_by_three_l646_64658


namespace NUMINAMATH_CALUDE_multiplication_commutative_l646_64638

theorem multiplication_commutative (a b : ℝ) : a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_multiplication_commutative_l646_64638


namespace NUMINAMATH_CALUDE_polynomial_remainder_l646_64675

/-- The polynomial p(x) = x^3 - 4x^2 + 3x + 2 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 2

/-- The remainder when p(x) is divided by (x - 1) -/
def remainder : ℝ := p 1

theorem polynomial_remainder : remainder = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l646_64675


namespace NUMINAMATH_CALUDE_runners_meeting_time_l646_64632

/-- Represents a runner with their lap time and start time offset -/
structure Runner where
  lap_time : ℕ
  start_offset : ℕ

/-- Calculates the earliest meeting time for multiple runners -/
def earliest_meeting_time (runners : List Runner) : ℕ :=
  sorry

/-- The main theorem stating the earliest meeting time for the given runners -/
theorem runners_meeting_time :
  let ben := Runner.mk 5 0
  let emily := Runner.mk 8 2
  let nick := Runner.mk 9 4
  earliest_meeting_time [ben, emily, nick] = 360 :=
sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l646_64632


namespace NUMINAMATH_CALUDE_max_popsicles_is_16_l646_64625

/-- Represents the cost and quantity of a popsicle package -/
structure PopsiclePackage where
  cost : ℕ
  quantity : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def maxPopsicles (budget : ℕ) (packages : List PopsiclePackage) : ℕ := sorry

/-- The specific problem setup -/
def problemSetup : List PopsiclePackage := [
  ⟨1, 1⟩,  -- Single popsicle
  ⟨3, 3⟩,  -- 3-popsicle box
  ⟨4, 7⟩   -- 7-popsicle box
]

/-- Theorem stating that the maximum number of popsicles Pablo can buy is 16 -/
theorem max_popsicles_is_16 :
  maxPopsicles 10 problemSetup = 16 := by sorry

end NUMINAMATH_CALUDE_max_popsicles_is_16_l646_64625


namespace NUMINAMATH_CALUDE_complex_power_sum_l646_64649

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^1000 + (1 / z^1000) = 2 * Real.cos (20 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l646_64649


namespace NUMINAMATH_CALUDE_officers_selection_count_l646_64617

/-- Represents the number of ways to choose officers from a club. -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  2 * (boys * (boys - 1) * (boys - 2))

/-- Theorem stating the number of ways to choose officers under given conditions. -/
theorem officers_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 2640 := by
  sorry

#eval choose_officers 24 12 12

end NUMINAMATH_CALUDE_officers_selection_count_l646_64617


namespace NUMINAMATH_CALUDE_M_mod_1000_eq_9_l646_64695

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- The theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000_eq_9 : M % 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_eq_9_l646_64695


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_value_l646_64677

theorem infinite_solutions_imply_d_value (d : ℚ) :
  (∀ (x : ℚ), 3 * (5 + 2 * d * x) = 15 * x + 15) → d = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_value_l646_64677


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l646_64690

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 3 * x + (3/2) ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l646_64690


namespace NUMINAMATH_CALUDE_max_points_is_36_l646_64684

/-- Represents a tournament with 8 teams where each team plays every other team twice -/
structure Tournament where
  num_teams : Nat
  games_per_pair : Nat
  win_points : Nat
  draw_points : Nat
  loss_points : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1) / 2) * t.games_per_pair

/-- Calculate the maximum possible points for each of the top three teams -/
def max_points_top_three (t : Tournament) : Nat :=
  let games_against_others := (t.num_teams - 3) * t.games_per_pair
  let points_against_others := games_against_others * t.win_points
  let games_among_top_three := 2 * t.games_per_pair
  let points_among_top_three := games_among_top_three * t.draw_points
  points_against_others + points_among_top_three

/-- The theorem to be proved -/
theorem max_points_is_36 (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 3)
  (h4 : t.draw_points = 1)
  (h5 : t.loss_points = 0) :
  max_points_top_three t = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_points_is_36_l646_64684


namespace NUMINAMATH_CALUDE_saturday_exclamation_l646_64694

/-- Represents the alien's exclamation as a string of 'A's and 'U's -/
def Exclamation := String

/-- Transforms a single character in the exclamation -/
def transformChar (c : Char) : Char :=
  match c with
  | 'A' => 'U'
  | 'U' => 'A'
  | _ => c

/-- Transforms the second half of the exclamation -/
def transformSecondHalf (s : String) : String :=
  s.map transformChar

/-- Generates the next day's exclamation based on the current day -/
def nextDayExclamation (current : Exclamation) : Exclamation :=
  let n := current.length
  let firstHalf := current.take (n / 2)
  let secondHalf := current.drop (n / 2)
  firstHalf ++ transformSecondHalf secondHalf

/-- Generates the nth day's exclamation -/
def nthDayExclamation (n : Nat) : Exclamation :=
  match n with
  | 0 => "A"
  | n + 1 => nextDayExclamation (nthDayExclamation n)

theorem saturday_exclamation :
  nthDayExclamation 5 = "АУУАУААУУААУАУААУУААУААААУУААУАА" :=
by sorry

end NUMINAMATH_CALUDE_saturday_exclamation_l646_64694


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l646_64627

theorem algebraic_expression_value (x y : ℝ) (h : x + y = 2) :
  (1/2) * x^2 + x * y + (1/2) * y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l646_64627


namespace NUMINAMATH_CALUDE_exam_average_l646_64659

theorem exam_average (successful_count unsuccessful_count : ℕ)
                     (successful_avg unsuccessful_avg : ℚ)
                     (h1 : successful_count = 20)
                     (h2 : unsuccessful_count = 20)
                     (h3 : successful_avg = 42)
                     (h4 : unsuccessful_avg = 38) :
  let total_count := successful_count + unsuccessful_count
  let total_points := successful_count * successful_avg + unsuccessful_count * unsuccessful_avg
  total_points / total_count = 40 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l646_64659


namespace NUMINAMATH_CALUDE_cube_root_simplification_l646_64697

theorem cube_root_simplification : Real.rpow (4^6 * 5^3 * 7^3) (1/3) = 560 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l646_64697


namespace NUMINAMATH_CALUDE_solve_for_a_l646_64606

theorem solve_for_a (m d b a : ℝ) (h1 : m = (d * a * b) / (a - b)) (h2 : m ≠ d * b) :
  a = (m * b) / (m - d * b) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l646_64606


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l646_64640

theorem stationery_box_sheets (S E : ℕ) : 
  S - (S / 3 + 50) = 50 →
  E = S / 3 + 50 →
  S = 150 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l646_64640


namespace NUMINAMATH_CALUDE_f_properties_l646_64656

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≤ 0) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≥ -15) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = 0) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l646_64656


namespace NUMINAMATH_CALUDE_marble_weight_calculation_l646_64647

/-- Given two pieces of marble of equal weight and a third piece,
    if the total weight is 0.75 tons and the third piece weighs 0.08333333333333333 ton,
    then the weight of each of the first two pieces is 0.33333333333333335 ton. -/
theorem marble_weight_calculation (w : ℝ) : 
  2 * w + 0.08333333333333333 = 0.75 → w = 0.33333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_calculation_l646_64647


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_4_l646_64693

theorem opposite_of_sqrt_4 : -(Real.sqrt 4) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_4_l646_64693


namespace NUMINAMATH_CALUDE_arithmetic_triangle_b_range_l646_64615

/-- A triangle with side lengths forming an arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d
  sum_of_squares : a^2 + b^2 + c^2 = 21
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The range of possible values for the middle term of the arithmetic sequence --/
theorem arithmetic_triangle_b_range (t : ArithmeticTriangle) :
  t.b ∈ Set.Ioo (Real.sqrt 6) (Real.sqrt 7) ∪ {Real.sqrt 7} :=
sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_b_range_l646_64615


namespace NUMINAMATH_CALUDE_harry_weekly_earnings_l646_64674

/-- Represents Harry's dog-walking schedule and earnings --/
structure DogWalker where
  mon_wed_fri_dogs : ℕ
  tuesday_dogs : ℕ
  thursday_dogs : ℕ
  pay_per_dog : ℕ

/-- Calculates the weekly earnings of a dog walker --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  (3 * dw.mon_wed_fri_dogs + dw.tuesday_dogs + dw.thursday_dogs) * dw.pay_per_dog

/-- Harry's specific dog-walking schedule --/
def harry : DogWalker :=
  { mon_wed_fri_dogs := 7
    tuesday_dogs := 12
    thursday_dogs := 9
    pay_per_dog := 5 }

/-- Theorem stating Harry's weekly earnings --/
theorem harry_weekly_earnings :
  weekly_earnings harry = 210 := by
  sorry

end NUMINAMATH_CALUDE_harry_weekly_earnings_l646_64674


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l646_64618

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^5 + 4 = (X - 3)^2 * q + r ∧ 
  r = 331 * X - 746 ∧
  r.degree < 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l646_64618


namespace NUMINAMATH_CALUDE_output_value_2003_l646_64643

/-- The annual growth rate of the company's output value -/
def growth_rate : ℝ := 0.10

/-- The initial output value of the company in 2000 (in millions of yuan) -/
def initial_value : ℝ := 10

/-- The number of years between 2000 and 2003 -/
def years : ℕ := 3

/-- The expected output value of the company in 2003 (in millions of yuan) -/
def expected_value : ℝ := 13.31

/-- Theorem stating that the company's output value in 2003 will be 13.31 million yuan -/
theorem output_value_2003 : 
  initial_value * (1 + growth_rate) ^ years = expected_value := by
  sorry

end NUMINAMATH_CALUDE_output_value_2003_l646_64643


namespace NUMINAMATH_CALUDE_circumcenters_not_concyclic_l646_64681

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to get the circumcenter of a triangle
def circumcenter (p1 p2 p3 : Point) : Point := sorry

-- Define a function to check if points are distinct
def areDistinct (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define a function to check if points are concyclic
def areConcyclic (p1 p2 p3 p4 : Point) : Prop := sorry

-- Theorem statement
theorem circumcenters_not_concyclic (q : Quadrilateral) 
  (h_convex : isConvex q)
  (O_A : Point) (O_B : Point) (O_C : Point) (O_D : Point)
  (h_O_A : O_A = circumcenter q.B q.C q.D)
  (h_O_B : O_B = circumcenter q.C q.D q.A)
  (h_O_C : O_C = circumcenter q.D q.A q.B)
  (h_O_D : O_D = circumcenter q.A q.B q.C)
  (h_distinct : areDistinct O_A O_B O_C O_D) :
  ¬(areConcyclic O_A O_B O_C O_D) := by
  sorry

end NUMINAMATH_CALUDE_circumcenters_not_concyclic_l646_64681


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l646_64624

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l646_64624


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l646_64687

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -2 ∧ b₀ = -2 ∧ c₀ = 1 ∧ d₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l646_64687


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l646_64652

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l646_64652


namespace NUMINAMATH_CALUDE_dividend_calculation_l646_64637

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l646_64637


namespace NUMINAMATH_CALUDE_hawks_score_l646_64641

theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 82 →
  eagles_points - hawks_points = 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 32 := by
sorry

end NUMINAMATH_CALUDE_hawks_score_l646_64641


namespace NUMINAMATH_CALUDE_quadratic_factor_difference_l646_64685

/-- Given a quadratic expression that can be factored, prove the difference of its factors' constants -/
theorem quadratic_factor_difference (a b : ℤ) : 
  (∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_difference_l646_64685


namespace NUMINAMATH_CALUDE_range_of_a_l646_64663

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l646_64663
