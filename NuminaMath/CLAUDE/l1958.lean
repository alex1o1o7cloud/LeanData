import Mathlib

namespace NUMINAMATH_CALUDE_sticker_distribution_l1958_195833

theorem sticker_distribution (total_stickers : ℕ) (ratio_sum : ℕ) (sam_ratio : ℕ) (andrew_ratio : ℕ) :
  total_stickers = 1500 →
  ratio_sum = 1 + 1 + sam_ratio →
  sam_ratio = 3 →
  andrew_ratio = 1 →
  (total_stickers / ratio_sum * andrew_ratio) + (total_stickers / ratio_sum * sam_ratio * 2 / 3) = 900 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1958_195833


namespace NUMINAMATH_CALUDE_union_M_N_equals_real_l1958_195825

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | x < 3}

-- Statement to prove
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_real_l1958_195825


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1958_195851

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h1 : f 1 = 1) 
  (h2 : f 2 = 2) : 
  f 23 + f (-14) = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1958_195851


namespace NUMINAMATH_CALUDE_sum_inequality_l1958_195881

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) +
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) +
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1958_195881


namespace NUMINAMATH_CALUDE_fewer_correct_answers_l1958_195805

-- Define the number of correct answers for each person
def cherry_correct : ℕ := 17
def nicole_correct : ℕ := 22
def kim_correct : ℕ := cherry_correct + 8

-- State the theorem
theorem fewer_correct_answers :
  kim_correct - nicole_correct = 3 ∧
  nicole_correct < kim_correct :=
by sorry

end NUMINAMATH_CALUDE_fewer_correct_answers_l1958_195805


namespace NUMINAMATH_CALUDE_symmetric_lines_l1958_195812

/-- Given two lines l and k symmetric with respect to y = x, prove that if l has equation y = ax + b, then k has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let l := {p : ℝ × ℝ | p.2 = a * p.1 + b}
  let k := {p : ℝ × ℝ | p.2 = (1/a) * p.1 - b/a}
  let symmetry := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p, p ∈ l ↔ (p.2, p.1) ∈ k) ∧ (∀ p, p ∈ k ↔ (p.2, p.1) ∈ l) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_lines_l1958_195812


namespace NUMINAMATH_CALUDE_rabbit_can_cross_tracks_l1958_195888

/-- The distance from the rabbit (point A) to the railway track -/
def rabbit_distance : ℝ := 160

/-- The speed of the train -/
def train_speed : ℝ := 30

/-- The initial distance of the train from point T -/
def train_initial_distance : ℝ := 300

/-- The speed of the rabbit -/
def rabbit_speed : ℝ := 15

/-- The lower bound of the safe crossing distance -/
def lower_bound : ℝ := 23.21

/-- The upper bound of the safe crossing distance -/
def upper_bound : ℝ := 176.79

theorem rabbit_can_cross_tracks :
  ∃ x : ℝ, lower_bound < x ∧ x < upper_bound ∧
  (((rabbit_distance ^ 2 + x ^ 2).sqrt / rabbit_speed) < ((train_initial_distance + x) / train_speed)) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_can_cross_tracks_l1958_195888


namespace NUMINAMATH_CALUDE_rhombus_shorter_diagonal_l1958_195826

/-- A rhombus with perimeter 9.6 and adjacent angles in ratio 1:2 has a shorter diagonal of length 2.4 -/
theorem rhombus_shorter_diagonal (p : ℝ) (r : ℚ) (d : ℝ) : 
  p = 9.6 → -- perimeter is 9.6
  r = 1/2 → -- ratio of adjacent angles is 1:2
  d = 2.4 -- shorter diagonal is 2.4
  := by sorry

end NUMINAMATH_CALUDE_rhombus_shorter_diagonal_l1958_195826


namespace NUMINAMATH_CALUDE_point_to_line_distance_l1958_195852

theorem point_to_line_distance (M : ℝ) : 
  (|(3 : ℝ) + Real.sqrt 3 * M - 4| / Real.sqrt (1 + 3) = 1) ↔ 
  (M = Real.sqrt 3 ∨ M = -(Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l1958_195852


namespace NUMINAMATH_CALUDE_tray_height_is_five_l1958_195882

/-- The height of a tray formed by cutting and folding a square piece of paper -/
def trayHeight (sideLength : ℝ) (cutDistance : ℝ) (cutAngle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 5 under given conditions -/
theorem tray_height_is_five :
  trayHeight 120 (Real.sqrt 25) (π / 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tray_height_is_five_l1958_195882


namespace NUMINAMATH_CALUDE_triangle_base_measurement_l1958_195801

/-- Given a triangular shape with height 20 cm, if the total area of three similar such shapes is 1200 cm², then the base of each triangle is 40 cm. -/
theorem triangle_base_measurement (height : ℝ) (total_area : ℝ) : 
  height = 20 → total_area = 1200 → ∃ (base : ℝ), base = 40 ∧ 3 * (base * height / 2) = total_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_measurement_l1958_195801


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_l1958_195827

/-- Given a family where some members eat vegetarian, some eat non-vegetarian, and some eat both,
    this theorem proves that the total number of people who eat vegetarian food is 28. -/
theorem vegetarian_eaters_count (only_veg : ℕ) (both : ℕ) 
    (h1 : only_veg = 16) (h2 : both = 12) : only_veg + both = 28 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_l1958_195827


namespace NUMINAMATH_CALUDE_stationery_purchase_l1958_195895

theorem stationery_purchase (brother_money sister_money : ℕ) : 
  brother_money = 2 * sister_money →
  brother_money - 180 = sister_money - 30 →
  brother_money = 300 ∧ sister_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_stationery_purchase_l1958_195895


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l1958_195835

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (not used in the proof, but included for completeness)
theorem square_is_convex (s : Square) : True := by
  sorry

theorem square_is_equiangular (s : Square) : True := by
  sorry

theorem square_has_equal_sides (s : Square) : True := by
  sorry

theorem all_squares_are_similar : ∀ (s1 s2 : Square), True := by
  sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l1958_195835


namespace NUMINAMATH_CALUDE_sequence_product_l1958_195885

theorem sequence_product : 
  (1/4) * 16 * (1/64) * 256 * (1/1024) * 4096 * (1/16384) * 65536 = 256 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1958_195885


namespace NUMINAMATH_CALUDE_original_number_l1958_195817

theorem original_number (w : ℝ) : (1.125 * w) - (0.75 * w) = 30 → w = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1958_195817


namespace NUMINAMATH_CALUDE_elrond_arwen_tulip_ratio_l1958_195855

/-- Given that Arwen picked 20 tulips and the total number of tulips picked by Arwen and Elrond is 60,
    prove that the ratio of Elrond's tulips to Arwen's tulips is 2:1 -/
theorem elrond_arwen_tulip_ratio :
  let arwen_tulips : ℕ := 20
  let total_tulips : ℕ := 60
  let elrond_tulips : ℕ := total_tulips - arwen_tulips
  (elrond_tulips : ℚ) / (arwen_tulips : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_elrond_arwen_tulip_ratio_l1958_195855


namespace NUMINAMATH_CALUDE_tournament_ranking_sequences_l1958_195869

/-- Represents a team in the tournament -/
inductive Team
| E | F | G | H | I | J | K

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  preliminary_matches : List Match
  semifinal_matches : List Match
  final_match : Match
  third_place_match : Match

/-- Represents a possible ranking sequence of four teams -/
structure RankingSequence where
  first : Team
  second : Team
  third : Team
  fourth : Team

/-- The main theorem to prove -/
theorem tournament_ranking_sequences (t : Tournament) :
  (t.preliminary_matches.length = 3) →
  (t.semifinal_matches.length = 2) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.E ∧ m.team2 = Team.F) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.G ∧ m.team2 = Team.H) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.I ∧ m.team2 = Team.J) →
  (∃ m ∈ t.semifinal_matches, m.team2 = Team.K) →
  (∃ ranking_sequences : List RankingSequence,
    ranking_sequences.length = 16 ∧
    ∀ rs ∈ ranking_sequences,
      (rs.first ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.second ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.third ∈ [t.third_place_match.team1, t.third_place_match.team2]) ∧
      (rs.fourth ∈ [t.third_place_match.team1, t.third_place_match.team2])) :=
by
  sorry

end NUMINAMATH_CALUDE_tournament_ranking_sequences_l1958_195869


namespace NUMINAMATH_CALUDE_fishing_trip_theorem_l1958_195802

def is_small_fish (weight : ℕ) : Bool := 1 ≤ weight ∧ weight ≤ 5
def is_medium_fish (weight : ℕ) : Bool := 6 ≤ weight ∧ weight ≤ 12
def is_large_fish (weight : ℕ) : Bool := weight > 12

def brendan_morning_catch : List ℕ := [1, 3, 4, 7, 7, 13, 15, 17]
def brendan_afternoon_catch : List ℕ := [2, 8, 8, 18, 20]
def emily_catch : List ℕ := [5, 6, 9, 11, 14, 20]
def dad_catch : List ℕ := [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21]

def brendan_morning_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ is_large_fish weight
def brendan_afternoon_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ (is_large_fish weight ∧ weight > 15)
def emily_keep (weight : ℕ) : Bool := is_large_fish weight ∨ weight = 5
def dad_keep (weight : ℕ) : Bool := (is_medium_fish weight ∧ weight ≥ 8 ∧ weight ≤ 11) ∨ (is_large_fish weight ∧ weight > 15 ∧ weight ≠ 21)

theorem fishing_trip_theorem :
  (brendan_morning_catch.filter brendan_morning_keep).length +
  (brendan_afternoon_catch.filter brendan_afternoon_keep).length +
  (emily_catch.filter emily_keep).length +
  (dad_catch.filter dad_keep).length = 18 := by
  sorry

end NUMINAMATH_CALUDE_fishing_trip_theorem_l1958_195802


namespace NUMINAMATH_CALUDE_angle_ABC_less_than_60_degrees_l1958_195859

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) where
  A : V
  B : V
  C : V

/-- The angle at vertex B in a triangle -/
def angle_at_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The altitude from vertex A in a triangle -/
def altitude_from_A {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The median from vertex B in a triangle -/
def median_from_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_angled {V : Type*} (t : Triangle V) : Prop := sorry

/-- Predicate to check if the altitude from A is the longest -/
def altitude_A_is_longest {V : Type*} (t : Triangle V) : Prop := sorry

theorem angle_ABC_less_than_60_degrees {V : Type*} (t : Triangle V) :
  is_acute_angled t →
  altitude_A_is_longest t →
  altitude_from_A t = median_from_B t →
  angle_at_B t < 60 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_less_than_60_degrees_l1958_195859


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l1958_195880

/-- A linear function y = (2m + 2)x + 5 is decreasing if and only if m < -1 -/
theorem linear_function_decreasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*m + 2)*x₁ + 5) > ((2*m + 2)*x₂ + 5)) ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l1958_195880


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1958_195836

theorem complex_fraction_sum (a b : ℝ) : 
  (2 + 3 * Complex.I) / Complex.I = Complex.mk a b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1958_195836


namespace NUMINAMATH_CALUDE_equation_is_linear_l1958_195828

/-- An equation is linear with one variable if it can be written in the form ax + b = 0,
    where a and b are constants and a ≠ 0. --/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 7x + 5 = 6(x - 1) --/
def f (x : ℝ) : ℝ := 7 * x + 5 - (6 * (x - 1))

theorem equation_is_linear : is_linear_equation_one_var f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1958_195828


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1958_195898

theorem larger_integer_problem (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 7 / 3 → 
  a * b = 189 → 
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1958_195898


namespace NUMINAMATH_CALUDE_fencing_cost_per_metre_l1958_195878

-- Define the ratio of the sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 9408

-- Define the total cost of fencing
def total_cost : ℚ := 98

-- Statement to prove
theorem fencing_cost_per_metre :
  let length := (ratio_length * Real.sqrt (area / (ratio_length * ratio_width)))
  let width := (ratio_width * Real.sqrt (area / (ratio_length * ratio_width)))
  let perimeter := 2 * (length + width)
  total_cost / perimeter = 0.25 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_metre_l1958_195878


namespace NUMINAMATH_CALUDE_parabola_y_values_l1958_195813

def f (x : ℝ) := -(x - 2)^2

theorem parabola_y_values :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-1) = y₁ → f 1 = y₂ → f 4 = y₃ →
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_values_l1958_195813


namespace NUMINAMATH_CALUDE_sampling_suitability_l1958_195856

/-- Represents a sampling scenario --/
structure SamplingScenario where
  population : ℕ
  sample_size : ℕ
  (valid : sample_size ≤ population)

/-- Determines if a sampling scenario is suitable for simple random sampling --/
def suitable_for_simple_random_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size ≤ 10 ∧ scenario.population ≤ 100

/-- Determines if a sampling scenario is suitable for systematic sampling --/
def suitable_for_systematic_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size > 10 ∧ scenario.population > 100

/-- The first sampling scenario --/
def scenario1 : SamplingScenario where
  population := 10
  sample_size := 2
  valid := by norm_num

/-- The second sampling scenario --/
def scenario2 : SamplingScenario where
  population := 1000
  sample_size := 50
  valid := by norm_num

/-- Theorem stating that the first scenario is suitable for simple random sampling
    and the second scenario is suitable for systematic sampling --/
theorem sampling_suitability :
  suitable_for_simple_random_sampling scenario1 ∧
  suitable_for_systematic_sampling scenario2 := by
  sorry


end NUMINAMATH_CALUDE_sampling_suitability_l1958_195856


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1958_195892

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℝ
  positive : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : CubeDimensions := ⟨4, by norm_num⟩

/-- Represents the cube to be removed -/
def removedCube : CubeDimensions := ⟨2, by norm_num⟩

/-- The number of faces of the removed cube that were initially exposed -/
def initiallyExposedFaces : ℕ := 3

theorem surface_area_unchanged :
  surfaceArea originalCube = 
  surfaceArea originalCube - 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube + 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube :=
sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l1958_195892


namespace NUMINAMATH_CALUDE_range_of_m_l1958_195816

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < m^2}

def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)

theorem range_of_m : 
  {m : ℝ | necessary_not_sufficient m} = {m : ℝ | -1/2 ≤ m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1958_195816


namespace NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1958_195862

/-- Given a circle with n distinct points, this function calculates the maximum number of chords
    intersecting a line through one point, not passing through any other points. -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- The theorem states that for a circle with 2017 points, the maximum number of
    intersecting chords is 1018080. -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by sorry

end NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1958_195862


namespace NUMINAMATH_CALUDE_smallest_tetrahedron_volume_ellipsoid_l1958_195849

/-- The smallest volume of a tetrahedron bounded by a tangent plane to an ellipsoid and coordinate planes -/
theorem smallest_tetrahedron_volume_ellipsoid (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ (V : ℝ), V ≥ (Real.sqrt 3 * a * b * c) / 2 → 
  ∃ (x y z : ℝ), x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 ∧ 
    V = (1/6) * (a^2/x) * (b^2/y) * (c^2/z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_tetrahedron_volume_ellipsoid_l1958_195849


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1958_195853

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1958_195853


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l1958_195824

/-- Given that 40 cows eat 40 bags of husk in 40 days, prove that one cow will eat one bag of husk in 40 days. -/
theorem one_cow_one_bag_days (cows bags days : ℕ) (h : cows = 40 ∧ bags = 40 ∧ days = 40) : 
  (cows * bags) / (cows * days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l1958_195824


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1958_195866

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ :=
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers :
  total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1958_195866


namespace NUMINAMATH_CALUDE_system_solution_exists_l1958_195850

theorem system_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = (3 * m + 2) * x + 1 ∧ y = (5 * m - 4) * x + 5) ↔ m ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1958_195850


namespace NUMINAMATH_CALUDE_min_value_of_angle_sum_l1958_195800

theorem min_value_of_angle_sum (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 → (4 / α + 1 / β ≥ 18 / π) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_angle_sum_l1958_195800


namespace NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1958_195877

/-- Calculates the total cost of sunscreen for a beach trip -/
def sunscreenCost (reapplyTime hours applicationAmount bottleAmount bottlePrice : ℕ) : ℕ :=
  let applications := hours / reapplyTime
  let totalAmount := applications * applicationAmount
  let bottlesNeeded := (totalAmount + bottleAmount - 1) / bottleAmount  -- Ceiling division
  bottlesNeeded * bottlePrice

/-- Theorem: The total cost of sunscreen for Tiffany's beach trip is $14 -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 16 3 12 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1958_195877


namespace NUMINAMATH_CALUDE_negation_of_existence_equivalence_l1958_195864

theorem negation_of_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_equivalence_l1958_195864


namespace NUMINAMATH_CALUDE_handshake_problem_l1958_195875

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l1958_195875


namespace NUMINAMATH_CALUDE_house_population_total_l1958_195891

/-- Represents the number of people on each floor of a three-story house. -/
structure HousePopulation where
  ground : ℕ
  first : ℕ
  second : ℕ

/-- Proves that given the conditions, the total number of people in the house is 60. -/
theorem house_population_total (h : HousePopulation) :
  (h.ground + h.first + h.second = 60) ∧
  (h.first + h.second = 35) ∧
  (h.ground + h.first = 45) ∧
  (h.first = (h.ground + h.first + h.second) / 3) :=
by sorry

end NUMINAMATH_CALUDE_house_population_total_l1958_195891


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_equals_two_l1958_195863

/-- The function f(x) defined as x(x - c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_deriv (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_minimum_implies_c_equals_two :
  ∀ c : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f c x ≥ f c 2) →
  f_deriv c 2 = 0 →
  c = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_equals_two_l1958_195863


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1958_195889

theorem trigonometric_equation_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
    (Real.sin (3 * x) + a * Real.sin (2 * x) + 2 * Real.sin x = 0) →
    (x = 0 ∨ x = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1958_195889


namespace NUMINAMATH_CALUDE_power_of_three_simplification_l1958_195840

theorem power_of_three_simplification :
  3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_simplification_l1958_195840


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1958_195874

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 3).sqrt / (x - 1)) ↔ x ≥ -3/2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1958_195874


namespace NUMINAMATH_CALUDE_garden_perimeter_l1958_195857

/-- The total perimeter of a rectangular garden with an attached triangular flower bed -/
theorem garden_perimeter (garden_length garden_width triangle_height : ℝ) 
  (hl : garden_length = 15)
  (hw : garden_width = 10)
  (ht : triangle_height = 6) :
  2 * (garden_length + garden_width) + 
  (Real.sqrt (garden_length^2 + triangle_height^2) + triangle_height) - 
  garden_length = 41 + Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1958_195857


namespace NUMINAMATH_CALUDE_bicycle_price_l1958_195890

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 120)
  (h2 : upfront_percentage = 0.2) :
  upfront_payment / upfront_percentage = 600 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_l1958_195890


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1958_195809

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1958_195809


namespace NUMINAMATH_CALUDE_total_watermelons_l1958_195867

def jason_watermelons : ℕ := 37
def sandy_watermelons : ℕ := 11

theorem total_watermelons : jason_watermelons + sandy_watermelons = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelons_l1958_195867


namespace NUMINAMATH_CALUDE_mall_sale_plate_cost_l1958_195879

theorem mall_sale_plate_cost 
  (treadmill_price : ℝ)
  (discount_rate : ℝ)
  (num_plates : ℕ)
  (plate_price : ℝ)
  (total_paid : ℝ)
  (h1 : treadmill_price = 1350)
  (h2 : discount_rate = 0.3)
  (h3 : num_plates = 2)
  (h4 : plate_price = 50)
  (h5 : total_paid = 1045) :
  treadmill_price * (1 - discount_rate) + num_plates * plate_price = total_paid ∧
  num_plates * plate_price = 100 := by
sorry

end NUMINAMATH_CALUDE_mall_sale_plate_cost_l1958_195879


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l1958_195846

theorem sqrt_product_equation (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (25 * y) * Real.sqrt (5 * y) * Real.sqrt (20 * y) = 40) :
  y = (Real.sqrt 30 * Real.rpow 3 (1/4)) / 15 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l1958_195846


namespace NUMINAMATH_CALUDE_combined_solid_sum_l1958_195868

/-- A right rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- A pyramid added to a rectangular prism -/
structure PrismWithPyramid :=
  (prism : RectangularPrism)
  (pyramid_base_face : ℕ)

/-- The combined solid (prism and pyramid) -/
def CombinedSolid (pw : PrismWithPyramid) : ℕ × ℕ × ℕ :=
  let new_faces := pw.prism.faces - pw.pyramid_base_face + 4
  let new_edges := pw.prism.edges + 4
  let new_vertices := pw.prism.vertices + 1
  (new_faces, new_edges, new_vertices)

theorem combined_solid_sum (pw : PrismWithPyramid) 
  (h1 : pw.prism.faces = 6)
  (h2 : pw.prism.edges = 12)
  (h3 : pw.prism.vertices = 8)
  (h4 : pw.pyramid_base_face = 1) :
  let (f, e, v) := CombinedSolid pw
  f + e + v = 34 := by sorry

end NUMINAMATH_CALUDE_combined_solid_sum_l1958_195868


namespace NUMINAMATH_CALUDE_f_composition_half_l1958_195865

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l1958_195865


namespace NUMINAMATH_CALUDE_chord_equation_l1958_195842

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define a chord passing through P with P as its midpoint
def is_chord_midpoint (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    (x1 + x2) / 2 = P.1 ∧ (y1 + y2) / 2 = P.2 ∧
    x = (x2 - x1) ∧ y = (y2 - y1)

-- Theorem statement
theorem chord_equation :
  ∀ (x y : ℝ), is_chord_midpoint x y → 2 * x + 3 * y = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1958_195842


namespace NUMINAMATH_CALUDE_stopped_clock_more_accurate_l1958_195876

/-- Represents the frequency of showing correct time for a clock --/
structure ClockAccuracy where
  correct_times_per_day : ℚ

/-- A clock that is one minute slow --/
def slow_clock : ClockAccuracy where
  correct_times_per_day := 1 / 720

/-- A stopped clock --/
def stopped_clock : ClockAccuracy where
  correct_times_per_day := 2

theorem stopped_clock_more_accurate : 
  stopped_clock.correct_times_per_day > slow_clock.correct_times_per_day := by
  sorry

#check stopped_clock_more_accurate

end NUMINAMATH_CALUDE_stopped_clock_more_accurate_l1958_195876


namespace NUMINAMATH_CALUDE_product_closest_to_2400_l1958_195883

def options : List ℝ := [210, 240, 2100, 2400, 24000]

theorem product_closest_to_2400 : 
  let product := 0.000315 * 7928564
  ∀ x ∈ options, x ≠ 2400 → |product - 2400| < |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_2400_l1958_195883


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l1958_195860

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n^2 + 9*n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n-1)

/-- The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∀ n : ℕ+, a (n+1) - a n = a (n+2) - a (n+1) :=
sorry

/-- The sum of the 4th and 6th terms is zero -/
theorem sum_of_4th_and_6th_is_zero : a 4 + a 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l1958_195860


namespace NUMINAMATH_CALUDE_odot_count_53_l1958_195804

/-- Represents a sequence of four symbols -/
structure SymbolSequence :=
  (symbols : Fin 4 → Char)
  (odot_count : (symbols 2 = '⊙' ∧ symbols 3 = '⊙') ∨ (symbols 1 = '⊙' ∧ symbols 2 = '⊙') ∨ (symbols 0 = '⊙' ∧ symbols 3 = '⊙'))

/-- Counts the occurrences of a symbol in the repeated pattern up to a given position -/
def count_symbol (seq : SymbolSequence) (symbol : Char) (n : Nat) : Nat :=
  (n / 4) * 2 + if n % 4 ≥ 3 then 2 else if n % 4 ≥ 2 then 1 else 0

/-- The main theorem stating that the count of ⊙ in the first 53 positions is 26 -/
theorem odot_count_53 (seq : SymbolSequence) : count_symbol seq '⊙' 53 = 26 := by
  sorry


end NUMINAMATH_CALUDE_odot_count_53_l1958_195804


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_pyramid_l1958_195830

theorem lateral_surface_area_of_pyramid (sin_alpha : ℝ) (diagonal_section_area : ℝ) :
  sin_alpha = 15 / 17 →
  diagonal_section_area = 3 * Real.sqrt 34 →
  (4 * diagonal_section_area) / (2 * Real.sqrt ((1 + (-Real.sqrt (1 - sin_alpha^2))) / 2)) = 68 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_pyramid_l1958_195830


namespace NUMINAMATH_CALUDE_problem_statement_l1958_195818

theorem problem_statement :
  -- Statement 1
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) ∧
  -- Statement 2
  (∀ p q : Prop, (p ∧ q) → (p ∨ q)) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(p ∧ q)) ∧
  -- Statement 4 (negation)
  ¬(∀ A B C D : Set α, (A ∪ B = A ∧ C ∩ D = C) → (A ⊆ B ∧ C ⊆ D)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1958_195818


namespace NUMINAMATH_CALUDE_special_function_inequality_l1958_195841

/-- A function satisfying the given properties in the problem -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  special_property : ∀ x, f (1 + x) + f (1 - x) = f 1
  decreasing_on_unit_interval : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f y < f x

/-- The main theorem to be proved -/
theorem special_function_inequality (sf : SpecialFunction) :
  sf.f (-2 + Real.sqrt 2 / 2) < -sf.f (10 / 3) ∧ -sf.f (10 / 3) < sf.f (9 / 2) := by
  sorry


end NUMINAMATH_CALUDE_special_function_inequality_l1958_195841


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1958_195896

/-- Prove that for y = e^(x + x^2) + 2e^x, the equation y' - y = 2x e^(x + x^2) holds. -/
theorem function_satisfies_equation (x : ℝ) : 
  let y := Real.exp (x + x^2) + 2 * Real.exp x
  let y' := Real.exp (x + x^2) * (1 + 2*x) + 2 * Real.exp x
  y' - y = 2 * x * Real.exp (x + x^2) := by
sorry


end NUMINAMATH_CALUDE_function_satisfies_equation_l1958_195896


namespace NUMINAMATH_CALUDE_triangle_inequality_holds_l1958_195884

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides (x : ℕ) : ℕ × ℕ × ℕ :=
  (6, x + 3, 2 * x - 1)

theorem triangle_inequality_holds (x : ℕ) :
  (∃ (y : ℕ), y ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧ x = y) ↔
  (let (a, b, c) := triangle_sides x
   is_valid_triangle a b c ∧ x > 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_holds_l1958_195884


namespace NUMINAMATH_CALUDE_maria_white_towels_l1958_195861

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 40

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 65

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 19

/-- The number of white towels Maria bought -/
def white_towels : ℕ := green_towels + towels_given - towels_left

theorem maria_white_towels : white_towels = 44 := by
  sorry

end NUMINAMATH_CALUDE_maria_white_towels_l1958_195861


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l1958_195834

theorem mean_equality_implies_y_value :
  let mean1 := (7 + 10 + 15 + 23) / 4
  let mean2 := (18 + y + 30) / 3
  mean1 = mean2 → y = -6.75 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l1958_195834


namespace NUMINAMATH_CALUDE_carnival_days_l1958_195854

theorem carnival_days (daily_income total_income : ℕ) 
  (h1 : daily_income = 144)
  (h2 : total_income = 3168) :
  total_income / daily_income = 22 := by
  sorry

end NUMINAMATH_CALUDE_carnival_days_l1958_195854


namespace NUMINAMATH_CALUDE_min_brownies_is_36_l1958_195838

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  length : ℕ
  width : ℕ

/-- Calculates the total number of brownies in the pan -/
def total_brownies (pan : BrowniePan) : ℕ := pan.length * pan.width

/-- Calculates the number of brownies on the perimeter of the pan -/
def perimeter_brownies (pan : BrowniePan) : ℕ := 2 * (pan.length + pan.width) - 4

/-- Calculates the number of brownies in the interior of the pan -/
def interior_brownies (pan : BrowniePan) : ℕ := (pan.length - 2) * (pan.width - 2)

/-- Checks if the pan satisfies the perimeter-to-interior ratio condition -/
def satisfies_ratio (pan : BrowniePan) : Prop :=
  perimeter_brownies pan = 2 * interior_brownies pan

/-- The main theorem stating that 36 is the smallest number of brownies satisfying all conditions -/
theorem min_brownies_is_36 :
  ∃ (pan : BrowniePan), satisfies_ratio pan ∧
    total_brownies pan = 36 ∧
    (∀ (other_pan : BrowniePan), satisfies_ratio other_pan →
      total_brownies other_pan ≥ 36) :=
  sorry

end NUMINAMATH_CALUDE_min_brownies_is_36_l1958_195838


namespace NUMINAMATH_CALUDE_matthews_cows_l1958_195814

/-- Proves that Matthews has 60 cows given the problem conditions -/
theorem matthews_cows :
  ∀ (matthews aaron marovich : ℕ),
  aaron = 4 * matthews →
  aaron + matthews = marovich + 30 →
  matthews + aaron + marovich = 570 →
  matthews = 60 := by
sorry

end NUMINAMATH_CALUDE_matthews_cows_l1958_195814


namespace NUMINAMATH_CALUDE_question_one_l1958_195803

theorem question_one (a b : ℚ) : |a| = 3 ∧ |b| = 1 ∧ a < b → a + b = -2 ∨ a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_question_one_l1958_195803


namespace NUMINAMATH_CALUDE_monday_messages_l1958_195829

/-- Proves that given the specified message sending pattern and average, 
    the number of messages sent on Monday must be 220. -/
theorem monday_messages (x : ℝ) : 
  (x + x/2 + 50 + 50 + 50) / 5 = 96 → x = 220 := by
  sorry

end NUMINAMATH_CALUDE_monday_messages_l1958_195829


namespace NUMINAMATH_CALUDE_g_equivalence_l1958_195837

theorem g_equivalence (x : Real) : 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) = 
  -Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_g_equivalence_l1958_195837


namespace NUMINAMATH_CALUDE_triangle_function_sign_l1958_195831

/-- Triangle with ordered sides -/
structure OrderedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : OrderedTriangle) : ℝ := sorry

/-- The inradius of a triangle -/
noncomputable def inradius (t : OrderedTriangle) : ℝ := sorry

/-- The angle C of a triangle -/
noncomputable def angle_C (t : OrderedTriangle) : ℝ := sorry

theorem triangle_function_sign (t : OrderedTriangle) :
  let f := t.a + t.b - 2 * circumradius t - 2 * inradius t
  let C := angle_C t
  (π / 3 ≤ C ∧ C < π / 2 → f > 0) ∧
  (C = π / 2 → f = 0) ∧
  (π / 2 < C ∧ C < π → f < 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_function_sign_l1958_195831


namespace NUMINAMATH_CALUDE_mother_notebooks_l1958_195894

/-- The number of notebooks the mother initially had -/
def initial_notebooks : ℕ := sorry

/-- The number of children -/
def num_children : ℕ := sorry

/-- If each child gets 13 notebooks, the mother has 8 notebooks left -/
axiom condition1 : initial_notebooks = 13 * num_children + 8

/-- If each child gets 15 notebooks, all notebooks are distributed -/
axiom condition2 : initial_notebooks = 15 * num_children

theorem mother_notebooks : initial_notebooks = 60 := by sorry

end NUMINAMATH_CALUDE_mother_notebooks_l1958_195894


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1958_195822

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : (a - 2*i) * i = b - i) : 
  a + b*i = -1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1958_195822


namespace NUMINAMATH_CALUDE_bounded_sequence_from_constrained_function_l1958_195844

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ, |a n| ≤ M

theorem bounded_sequence_from_constrained_function
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_cont : Continuous (deriv f))
  (hf_bound : ∀ x : ℝ, 0 ≤ |deriv f x| ∧ |deriv f x| ≤ (1 : ℝ) / 2)
  (a : ℕ → ℝ)
  (ha_init : a 1 = 1)
  (ha_rec : ∀ n : ℕ, a (n + 1) = f (a n)) :
  is_bounded_sequence a :=
by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_from_constrained_function_l1958_195844


namespace NUMINAMATH_CALUDE_triangle_area_l1958_195806

/-- Given a triangle ABC with side a = 2, angle A = 30°, and angle C = 45°, 
    prove that its area S is equal to √3 + 1 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 2 → 
  A = π / 6 → 
  C = π / 4 → 
  A + B + C = π → 
  a / Real.sin A = c / Real.sin C → 
  S = (1 / 2) * a * c * Real.sin B → 
  S = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l1958_195806


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1958_195893

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1958_195893


namespace NUMINAMATH_CALUDE_technician_journey_percentage_l1958_195810

theorem technician_journey_percentage (D : ℝ) (h : D > 0) : 
  let total_distance := 2 * D
  let completed_distance := 0.65 * total_distance
  let outbound_distance := D
  let return_distance_completed := completed_distance - outbound_distance
  (return_distance_completed / D) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_technician_journey_percentage_l1958_195810


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l1958_195807

/-- Proves that the number of meters of cloth sold is 60, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8400)
    (h2 : profit_per_meter = 12)
    (h3 : cost_price_per_meter = 128) :
    total_selling_price / (cost_price_per_meter + profit_per_meter) = 60 := by
  sorry

#check cloth_sale_meters

end NUMINAMATH_CALUDE_cloth_sale_meters_l1958_195807


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1958_195823

theorem sum_of_three_numbers : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1958_195823


namespace NUMINAMATH_CALUDE_cookies_left_l1958_195897

theorem cookies_left (whole_cookies : ℕ) (greg_ate : ℕ) (brad_ate : ℕ) : 
  whole_cookies = 14 → greg_ate = 4 → brad_ate = 6 → 
  whole_cookies * 2 - (greg_ate + brad_ate) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l1958_195897


namespace NUMINAMATH_CALUDE_mock_exam_participants_l1958_195899

/-- The number of students who took a mock exam -/
def total_students : ℕ := 400

/-- The number of girls who took the exam -/
def num_girls : ℕ := 100

/-- The proportion of boys who cleared the cut off -/
def boys_cleared_ratio : ℚ := 3/5

/-- The proportion of girls who cleared the cut off -/
def girls_cleared_ratio : ℚ := 4/5

/-- The total proportion of students who qualified -/
def total_qualified_ratio : ℚ := 13/20

theorem mock_exam_participants :
  ∃ (num_boys : ℕ),
    (boys_cleared_ratio * num_boys + girls_cleared_ratio * num_girls : ℚ) = 
    total_qualified_ratio * (num_boys + num_girls) ∧
    total_students = num_boys + num_girls :=
by sorry

end NUMINAMATH_CALUDE_mock_exam_participants_l1958_195899


namespace NUMINAMATH_CALUDE_inequality_product_l1958_195839

theorem inequality_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_l1958_195839


namespace NUMINAMATH_CALUDE_no_integer_solution_l1958_195873

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x*y*z - 1 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1958_195873


namespace NUMINAMATH_CALUDE_det_A_eq_31_l1958_195819

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 3, 6, -2; 1, -1, 3]

theorem det_A_eq_31 : Matrix.det A = 31 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_31_l1958_195819


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1958_195845

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m : ℝ) : Prop :=
  (-m / (2*m - 1)) * (-3 / m) = -1

/-- The first line equation -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + (2*m - 1)*y + 1 = 0

/-- The second line equation -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  3*x + m*y + 2 = 0

/-- m = -1 is sufficient but not necessary for the lines to be perpendicular -/
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → are_perpendicular m) ∧
  ¬(are_perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1958_195845


namespace NUMINAMATH_CALUDE_expand_product_l1958_195887

theorem expand_product (x : ℝ) : 3 * (x^2 - 5*x + 6) * (x^2 + 8*x - 10) = 3*x^4 + 9*x^3 - 132*x^2 + 294*x - 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1958_195887


namespace NUMINAMATH_CALUDE_donation_ratio_l1958_195843

def charity_raffle_problem (total_prize donation hotdog_cost leftover : ℕ) : Prop :=
  total_prize = donation + hotdog_cost + leftover ∧
  total_prize = 114 ∧
  hotdog_cost = 2 ∧
  leftover = 55

theorem donation_ratio (total_prize donation hotdog_cost leftover : ℕ) :
  charity_raffle_problem total_prize donation hotdog_cost leftover →
  (donation : ℚ) / total_prize = 55 / 114 := by
sorry

end NUMINAMATH_CALUDE_donation_ratio_l1958_195843


namespace NUMINAMATH_CALUDE_cone_volume_l1958_195811

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) : 
  let l : ℝ := 15  -- slant height
  let h : ℝ := 9   -- height
  let r : ℝ := Real.sqrt (l^2 - h^2)  -- radius of the base
  (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1958_195811


namespace NUMINAMATH_CALUDE_body_lotion_cost_is_60_l1958_195871

/-- Represents the cost of items and total spent at Target --/
structure TargetPurchase where
  tanya_face_moisturizer_cost : ℕ
  tanya_face_moisturizer_count : ℕ
  tanya_body_lotion_count : ℕ
  total_spent : ℕ

/-- Calculates the cost of each body lotion based on the given conditions --/
def body_lotion_cost (p : TargetPurchase) : ℕ :=
  let tanya_total := p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count + 
                     p.tanya_body_lotion_count * (p.total_spent / 3)
  (p.total_spent / 3 - p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count) / p.tanya_body_lotion_count

/-- Theorem stating that the cost of each body lotion is $60 --/
theorem body_lotion_cost_is_60 (p : TargetPurchase) 
  (h1 : p.tanya_face_moisturizer_cost = 50)
  (h2 : p.tanya_face_moisturizer_count = 2)
  (h3 : p.tanya_body_lotion_count = 4)
  (h4 : p.total_spent = 1020) :
  body_lotion_cost p = 60 := by
  sorry


end NUMINAMATH_CALUDE_body_lotion_cost_is_60_l1958_195871


namespace NUMINAMATH_CALUDE_soccer_players_count_l1958_195832

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l1958_195832


namespace NUMINAMATH_CALUDE_fraction_equality_l1958_195848

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1958_195848


namespace NUMINAMATH_CALUDE_video_views_equation_l1958_195858

/-- Represents the number of views on the first day -/
def initial_views : ℕ := 4400

/-- Represents the increase in views after 4 days -/
def increase_factor : ℕ := 10

/-- Represents the additional views after 2 more days -/
def additional_views : ℕ := 50000

/-- Represents the total views at the end -/
def total_views : ℕ := 94000

/-- Proves that the initial number of views satisfies the given equation -/
theorem video_views_equation : 
  increase_factor * initial_views + additional_views = total_views := by
  sorry

end NUMINAMATH_CALUDE_video_views_equation_l1958_195858


namespace NUMINAMATH_CALUDE_problem_solution_l1958_195820

theorem problem_solution : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 65 ∧ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1958_195820


namespace NUMINAMATH_CALUDE_refrigeratorSample_is_valid_l1958_195821

/-- Represents a systematic sample -/
structure SystematicSample (N : ℕ) (n : ℕ) where
  start : ℕ
  sequence : Fin n → ℕ
  valid : ∀ i : Fin n, sequence i = start + i.val * (N / n)

/-- The specific systematic sample for the refrigerator problem -/
def refrigeratorSample : SystematicSample 60 6 :=
  { start := 3,
    sequence := λ i => 3 + i.val * 10,
    valid := sorry }

/-- Theorem stating that the refrigeratorSample is valid -/
theorem refrigeratorSample_is_valid :
  ∀ i : Fin 6, refrigeratorSample.sequence i ≤ 60 :=
by sorry

end NUMINAMATH_CALUDE_refrigeratorSample_is_valid_l1958_195821


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1958_195847

theorem product_sum_theorem (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 1656 ∧ 
  (a % 10) * b < 1000 →
  a + b = 110 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1958_195847


namespace NUMINAMATH_CALUDE_min_surface_area_to_volume_ratio_cylinder_in_sphere_l1958_195870

/-- For a right circular cylinder inscribed in a sphere of radius R,
    the minimum ratio of surface area to volume is (((4^(1/3)) + 1)^(3/2)) / R. -/
theorem min_surface_area_to_volume_ratio_cylinder_in_sphere (R : ℝ) (h : R > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ r^2 + (h/2)^2 = R^2 ∧
    ∀ (r' h' : ℝ), r' > 0 → h' > 0 → r'^2 + (h'/2)^2 = R^2 →
      (2 * π * r * (r + h)) / (π * r^2 * h) ≥ (((4^(1/3) : ℝ) + 1)^(3/2)) / R :=
by sorry

end NUMINAMATH_CALUDE_min_surface_area_to_volume_ratio_cylinder_in_sphere_l1958_195870


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_quadratic_perimeter_l1958_195808

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two real roots -/
def has_two_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

theorem isosceles_triangle_from_quadratic_perimeter
  (k : ℝ)
  (eq : QuadraticEquation)
  (t : IsoscelesTriangle)
  (h1 : eq = { a := 1, b := -4, c := 2*k })
  (h2 : has_two_real_roots eq)
  (h3 : t.base = 1)
  (h4 : t.leg = 2) :
  perimeter t = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_quadratic_perimeter_l1958_195808


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1958_195872

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1958_195872


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1958_195886

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, a * x^2 + (a + 1) * x + 6 * a = 0) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ < 1 ∧ 1 < x₂) ∧
  (a * x₁^2 + (a + 1) * x₁ + 6 * a = 0) ∧
  (a * x₂^2 + (a + 1) * x₂ + 6 * a = 0) →
  -1/8 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1958_195886


namespace NUMINAMATH_CALUDE_linda_spent_680_l1958_195815

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℚ :=
  let notebook_price : ℚ := 1.20
  let notebook_quantity : ℕ := 3
  let pencil_box_price : ℚ := 1.50
  let pen_box_price : ℚ := 1.70
  notebook_price * notebook_quantity + pencil_box_price + pen_box_price

theorem linda_spent_680 : linda_total_spent = 6.80 := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_680_l1958_195815
