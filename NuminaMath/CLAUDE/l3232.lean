import Mathlib

namespace NUMINAMATH_CALUDE_negation_existence_cube_plus_one_l3232_323219

theorem negation_existence_cube_plus_one (x : ℝ) :
  (¬ ∃ x, x^3 + 1 = 0) ↔ ∀ x, x^3 + 1 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_cube_plus_one_l3232_323219


namespace NUMINAMATH_CALUDE_perimeter_ratio_square_to_rectangle_l3232_323212

/-- The ratio of the perimeter of a square with side length 700 to the perimeter of a rectangle with length 400 and width 300 is 2:1 -/
theorem perimeter_ratio_square_to_rectangle : 
  let square_side : ℕ := 700
  let rect_length : ℕ := 400
  let rect_width : ℕ := 300
  let square_perimeter : ℕ := 4 * square_side
  let rect_perimeter : ℕ := 2 * (rect_length + rect_width)
  (square_perimeter : ℚ) / rect_perimeter = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_square_to_rectangle_l3232_323212


namespace NUMINAMATH_CALUDE_hall_length_l3232_323283

/-- A rectangular hall with breadth two-thirds of its length and area 2400 sq meters has a length of 60 meters. -/
theorem hall_length (length breadth : ℝ) : 
  breadth = (2 / 3) * length →
  length * breadth = 2400 →
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_hall_length_l3232_323283


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3232_323226

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 8 cm, length 5 cm, and height 10 cm is 340 cm² -/
theorem cuboid_surface_area_example : cuboid_surface_area 8 5 10 = 340 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3232_323226


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9999_l3232_323224

theorem largest_prime_factor_of_9999 : ∃ (p : ℕ), p.Prime ∧ p ∣ 9999 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9999 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9999_l3232_323224


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l3232_323210

theorem clock_equivalent_hours : 
  ∃ n : ℕ, n > 5 ∧ 
           n * n - n ≡ 0 [MOD 12] ∧ 
           ∀ m : ℕ, m > 5 ∧ m < n → ¬(m * m - m ≡ 0 [MOD 12]) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l3232_323210


namespace NUMINAMATH_CALUDE_total_cards_traded_l3232_323229

/-- The number of cards traded between Padma and Robert -/
def cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_trade1 : ℕ) (robert_trade1 : ℕ) 
  (padma_trade2 : ℕ) (robert_trade2 : ℕ) : ℕ :=
  (padma_trade1 + padma_trade2) + (robert_trade1 + robert_trade2)

/-- Theorem stating the total number of cards traded -/
theorem total_cards_traded : 
  cards_traded 75 88 2 10 15 8 = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_traded_l3232_323229


namespace NUMINAMATH_CALUDE_angle_C_measure_l3232_323215

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - 1/2

theorem angle_C_measure (ω : ℝ) (a b : ℝ) (A : ℝ) :
  ω > 0 →
  (∀ x, f ω (x + π) = f ω x) →
  (∀ x, ¬∀ y, y ≠ x → y > 0 → f ω (x + y) = f ω x) →
  a = 1 →
  b = Real.sqrt 2 →
  f ω (A / 2) = Real.sqrt 3 / 2 →
  a < b →
  ∃ C, (C = 7 * π / 12 ∨ C = π / 12) ∧
       C + A + Real.arcsin (b * Real.sin A / a) = π :=
by sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3232_323215


namespace NUMINAMATH_CALUDE_stacked_cubes_volume_l3232_323258

/-- Calculates the total volume of stacked cubes -/
def total_volume (cube_dim : ℝ) (rows cols floors : ℕ) : ℝ :=
  (cube_dim ^ 3) * (rows * cols * floors)

/-- The problem statement -/
theorem stacked_cubes_volume :
  let cube_dim : ℝ := 1
  let rows : ℕ := 7
  let cols : ℕ := 5
  let floors : ℕ := 3
  total_volume cube_dim rows cols floors = 105 := by
  sorry

end NUMINAMATH_CALUDE_stacked_cubes_volume_l3232_323258


namespace NUMINAMATH_CALUDE_golf_problem_l3232_323201

/-- Calculates how far beyond the hole a golf ball lands given the total distance to the hole,
    the distance of the first hit, and that the second hit travels half as far as the first. -/
def beyond_hole (total_distance first_hit : ℕ) : ℕ :=
  let second_hit := first_hit / 2
  let distance_after_first := total_distance - first_hit
  second_hit - distance_after_first

/-- Theorem stating that under the given conditions, the ball lands 20 yards beyond the hole. -/
theorem golf_problem : beyond_hole 250 180 = 20 := by
  sorry

end NUMINAMATH_CALUDE_golf_problem_l3232_323201


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_6_l3232_323223

theorem circle_area_with_diameter_6 (π : ℝ) (h : π > 0) :
  let diameter : ℝ := 6
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_6_l3232_323223


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3232_323216

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0)
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : quadratic_function a b c (-1) = 0) :
  {x : ℝ | quadratic_function a b c x ≥ 0} = Set.Icc (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3232_323216


namespace NUMINAMATH_CALUDE_max_ab_and_min_fraction_l3232_323276

theorem max_ab_and_min_fraction (a b x y : ℝ) : 
  a > 0 → b > 0 → 4 * a + b = 1 → 
  x > 0 → y > 0 → x + y = 1 → 
  (∀ a' b', a' > 0 → b' > 0 → 4 * a' + b' = 1 → a * b ≥ a' * b') ∧ 
  (∀ x' y', x' > 0 → y' > 0 → x' + y' = 1 → 4 / x + 9 / y ≤ 4 / x' + 9 / y') ∧
  a * b = 1 / 16 ∧ 
  4 / x + 9 / y = 25 := by
sorry

end NUMINAMATH_CALUDE_max_ab_and_min_fraction_l3232_323276


namespace NUMINAMATH_CALUDE_honda_red_percentage_l3232_323271

theorem honda_red_percentage (total_cars : ℕ) (honda_cars : ℕ) 
  (total_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  total_red_percentage = 60 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (honda_cars : ℚ) * (90 / 100) + 
    (total_cars - honda_cars : ℚ) * non_honda_red_percentage = 
    (total_cars : ℚ) * total_red_percentage :=
by sorry

end NUMINAMATH_CALUDE_honda_red_percentage_l3232_323271


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l3232_323220

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l3232_323220


namespace NUMINAMATH_CALUDE_correct_calculation_l3232_323265

theorem correct_calculation (x y : ℝ) : -x^2*y + 3*x^2*y = 2*x^2*y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3232_323265


namespace NUMINAMATH_CALUDE_james_monthly_earnings_l3232_323256

/-- Calculates the monthly earnings of a Twitch streamer --/
def monthly_earnings (initial_subscribers : ℕ) (gifted_subscribers : ℕ) (earnings_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber

theorem james_monthly_earnings :
  monthly_earnings 150 50 9 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_james_monthly_earnings_l3232_323256


namespace NUMINAMATH_CALUDE_nines_count_to_thousand_l3232_323238

/-- Count of digit 9 appearances in a single integer -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Count of digit 9 appearances in all integers from 1 to n (inclusive) -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 appearances in all integers from 1 to 1000 (inclusive) is 301 -/
theorem nines_count_to_thousand : total_nines 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_nines_count_to_thousand_l3232_323238


namespace NUMINAMATH_CALUDE_number_of_groups_l3232_323253

theorem number_of_groups (max min interval : ℝ) (h1 : max = 140) (h2 : min = 51) (h3 : interval = 10) :
  ⌈(max - min) / interval⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_groups_l3232_323253


namespace NUMINAMATH_CALUDE_class_average_score_l3232_323279

theorem class_average_score (total_students : ℕ) (score1 score2 score3 : ℕ) (rest_average : ℚ) :
  total_students = 35 →
  score1 = 93 →
  score2 = 83 →
  score3 = 87 →
  rest_average = 76 →
  (score1 + score2 + score3 + (total_students - 3) * rest_average) / total_students = 77 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l3232_323279


namespace NUMINAMATH_CALUDE_board_length_before_final_cut_l3232_323240

/-- The length of a board before the final cut, given the initial length, first cut length, and final cut length. -/
def boardLengthBeforeFinalCut (initialLength firstCut finalCut : ℕ) : ℕ :=
  initialLength - firstCut + finalCut

/-- Theorem stating that the length of the boards before the final 7 cm cut is 125 cm. -/
theorem board_length_before_final_cut :
  boardLengthBeforeFinalCut 143 25 7 = 125 := by
  sorry

end NUMINAMATH_CALUDE_board_length_before_final_cut_l3232_323240


namespace NUMINAMATH_CALUDE_striped_quadrilateral_area_l3232_323272

/-- Represents a quadrilateral cut from striped gift wrapping paper -/
structure StripedQuadrilateral where
  /-- The combined area of the grey stripes in the quadrilateral -/
  greyArea : ℝ
  /-- The stripes are equally wide -/
  equalStripes : Bool

/-- Theorem stating that if the grey stripes have an area of 10 in a quadrilateral
    cut from equally striped paper, then the total area of the quadrilateral is 20 -/
theorem striped_quadrilateral_area
  (quad : StripedQuadrilateral)
  (h1 : quad.greyArea = 10)
  (h2 : quad.equalStripes = true) :
  quad.greyArea * 2 = 20 := by
  sorry

#check striped_quadrilateral_area

end NUMINAMATH_CALUDE_striped_quadrilateral_area_l3232_323272


namespace NUMINAMATH_CALUDE_equation_solutions_l3232_323284

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ 
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 6 ∧ 
    (x₁ + 3)*(x₁ - 3) = 3*(x₁ + 3) ∧ (x₂ + 3)*(x₂ - 3) = 3*(x₂ + 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3232_323284


namespace NUMINAMATH_CALUDE_school_choir_members_l3232_323287

theorem school_choir_members :
  ∃! n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ n = 241 :=
by sorry

end NUMINAMATH_CALUDE_school_choir_members_l3232_323287


namespace NUMINAMATH_CALUDE_ryans_leaf_collection_l3232_323249

/-- Given Ryan's leaf collection scenario, prove the number of remaining leaves. -/
theorem ryans_leaf_collection :
  let initial_leaves : ℕ := 89
  let first_loss : ℕ := 24
  let second_loss : ℕ := 43
  initial_leaves - first_loss - second_loss = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_ryans_leaf_collection_l3232_323249


namespace NUMINAMATH_CALUDE_proper_subsets_count_l3232_323208

def S : Finset Nat := {2, 4, 6, 8}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l3232_323208


namespace NUMINAMATH_CALUDE_square_remainder_mod_nine_l3232_323245

theorem square_remainder_mod_nine (n : ℤ) : 
  (n % 9 = 1 ∨ n % 9 = 8) → (n^2) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_nine_l3232_323245


namespace NUMINAMATH_CALUDE_minimum_employees_science_bureau_hiring_l3232_323244

theorem minimum_employees (water : ℕ) (air : ℕ) (both : ℕ) : ℕ :=
  let total := water + air - both
  total

theorem science_bureau_hiring : 
  minimum_employees 98 89 34 = 153 := by sorry

end NUMINAMATH_CALUDE_minimum_employees_science_bureau_hiring_l3232_323244


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l3232_323250

/-- Calculates the number of high school students selected in a stratified sampling -/
def stratified_sampling (total_students : ℕ) (selected_students : ℕ) (high_school_students : ℕ) : ℕ :=
  (high_school_students * selected_students) / total_students

/-- Theorem: In a stratified sampling of 15 students from 165 students, 
    where 66 are high school students, 6 high school students will be selected -/
theorem stratified_sampling_result : 
  stratified_sampling 165 15 66 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l3232_323250


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l3232_323267

-- Define the cubic function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Theorem 1: Tangent line equation
theorem tangent_line_at_zero (a b c : ℝ) :
  ∃ m k, ∀ x, m*x + k = f a b c x + (f a b c 0 - f a b c x) / x :=
sorry

-- Theorem 2: Range of c when a = b = 4 and f has three distinct zeros
theorem range_of_c_for_three_zeros :
  ∃ c, 0 < c ∧ c < 32/27 ∧
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f 4 4 c x = 0 ∧ f 4 4 c y = 0 ∧ f 4 4 c z = 0) :=
sorry

-- Theorem 3: Necessary but not sufficient condition for three distinct zeros
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →
  a^2 - 3*b > 0 :=
sorry

theorem condition_not_sufficient :
  ∃ a b c, a^2 - 3*b > 0 ∧
  ¬(∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l3232_323267


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l3232_323299

/-- A hexagon with specific properties -/
structure Hexagon :=
  (AB EF BE AF CD DF : ℝ)
  (ab_ef_eq : AB = EF)
  (be_af_eq : BE = AF)
  (ab_length : AB = 3)
  (be_length : BE = 4)
  (cd_length : CD = 5)
  (df_length : DF = 5)

/-- The perimeter of the hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BE + h.CD + h.DF + h.EF + h.AF

/-- Theorem stating that the perimeter of the hexagon is 24 units -/
theorem hexagon_perimeter_is_24 (h : Hexagon) : perimeter h = 24 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l3232_323299


namespace NUMINAMATH_CALUDE_inscribed_equiangular_triangle_exists_l3232_323246

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the concept of being inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Define the concept of two triangles being equiangular
def isEquiangular (t1 t2 : Triangle) : Prop :=
  sorry

theorem inscribed_equiangular_triangle_exists 
  (c : Circle) (reference : Triangle) : 
  ∃ (t : Triangle), isInscribed t c ∧ isEquiangular t reference := by
  sorry

end NUMINAMATH_CALUDE_inscribed_equiangular_triangle_exists_l3232_323246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3232_323206

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 + a 4 = 30) →
  (a 2 + a 3 = 15) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3232_323206


namespace NUMINAMATH_CALUDE_sun_city_population_relation_l3232_323282

/-- The population of Willowdale city -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville city -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 12000

/-- The multiple of Roseville City's population that Sun City has 1000 more than -/
def multiple : ℚ := (sun_city_population - 1000) / roseville_population

theorem sun_city_population_relation :
  sun_city_population = multiple * roseville_population + 1000 ∧ multiple = 2 := by sorry

end NUMINAMATH_CALUDE_sun_city_population_relation_l3232_323282


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l3232_323261

theorem equilateral_triangle_division (side_length : ℕ) (h : side_length = 1536) :
  ∃ (n : ℕ), side_length^2 = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l3232_323261


namespace NUMINAMATH_CALUDE_problem_solution_l3232_323297

theorem problem_solution (a b : ℝ) : 
  |a + 1| + (b - 2)^2 = 0 → (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3232_323297


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3232_323263

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem norm_scalar_multiple (v : E) (h : ‖v‖ = 7) : ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3232_323263


namespace NUMINAMATH_CALUDE_cosine_cutting_plane_angle_l3232_323228

/-- Regular hexagonal pyramid with specific cross-section --/
structure HexagonalPyramid where
  base_side : ℝ
  vertex_to_plane_dist : ℝ

/-- Theorem: Cosine of angle between cutting plane and base plane in specific hexagonal pyramid --/
theorem cosine_cutting_plane_angle (pyramid : HexagonalPyramid) 
  (h1 : pyramid.base_side = 8)
  (h2 : pyramid.vertex_to_plane_dist = 3 * Real.sqrt (13/7))
  : Real.sqrt 3 / 4 = 
    Real.sqrt (1 - (28 * pyramid.vertex_to_plane_dist^2) / (9 * pyramid.base_side^2)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_cutting_plane_angle_l3232_323228


namespace NUMINAMATH_CALUDE_women_to_total_ratio_l3232_323291

theorem women_to_total_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  ∃ (women men standing_men : ℕ),
    women + men = total_passengers ∧
    standing_men + seated_men = men ∧
    standing_men = men / 8 ∧
    women * 3 = total_passengers * 2 := by
  sorry

end NUMINAMATH_CALUDE_women_to_total_ratio_l3232_323291


namespace NUMINAMATH_CALUDE_divisibility_of_314n_l3232_323227

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem divisibility_of_314n : 
  ∀ n : ℕ, n < 10 → (is_divisible_by (3140 + n) 18 ↔ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_314n_l3232_323227


namespace NUMINAMATH_CALUDE_rational_function_positivity_l3232_323221

theorem rational_function_positivity (x : ℝ) :
  (x^2 - 9) / (x^2 - 16) > 0 ↔ x < -4 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_positivity_l3232_323221


namespace NUMINAMATH_CALUDE_total_tanks_needed_l3232_323295

/-- Calculates the minimum number of tanks needed to fill all balloons --/
def minTanksNeeded (smallBalloons mediumBalloons largeBalloons : Nat)
  (smallCapacity mediumCapacity largeCapacity : Nat)
  (heliumTankCapacity hydrogenTankCapacity mixtureTankCapacity : Nat) : Nat :=
  let heliumNeeded := smallBalloons * smallCapacity
  let hydrogenNeeded := mediumBalloons * mediumCapacity
  let mixtureNeeded := largeBalloons * largeCapacity
  let heliumTanks := (heliumNeeded + heliumTankCapacity - 1) / heliumTankCapacity
  let hydrogenTanks := (hydrogenNeeded + hydrogenTankCapacity - 1) / hydrogenTankCapacity
  let mixtureTanks := (mixtureNeeded + mixtureTankCapacity - 1) / mixtureTankCapacity
  heliumTanks + hydrogenTanks + mixtureTanks

theorem total_tanks_needed :
  minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500 = 392 := by
  sorry

#eval minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500

end NUMINAMATH_CALUDE_total_tanks_needed_l3232_323295


namespace NUMINAMATH_CALUDE_complex_multiplication_l3232_323207

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + 2*i) * (1 - 2*i) = 6 - 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3232_323207


namespace NUMINAMATH_CALUDE_abc_product_magnitude_l3232_323290

theorem abc_product_magnitude (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a - 1/b = b - 1/c ∧ b - 1/c = c - 1/a) :
    |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_magnitude_l3232_323290


namespace NUMINAMATH_CALUDE_factorization_proof_l3232_323260

theorem factorization_proof (a : ℝ) : 3 * a^2 - 27 = 3 * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3232_323260


namespace NUMINAMATH_CALUDE_password_probability_l3232_323259

-- Define the set of all possible symbols
def AllSymbols : Finset Char := {'!', '@', '#', '$', '%'}

-- Define the set of allowed symbols
def AllowedSymbols : Finset Char := {'!', '@', '#'}

-- Define the set of all possible single digits
def AllDigits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the set of even single digits
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

-- Define the set of non-zero single digits
def NonZeroDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Theorem statement
theorem password_probability :
  (Finset.card EvenDigits : ℚ) / (Finset.card AllDigits) *
  (Finset.card AllowedSymbols : ℚ) / (Finset.card AllSymbols) *
  (Finset.card NonZeroDigits : ℚ) / (Finset.card AllDigits) =
  27 / 100 := by
sorry

end NUMINAMATH_CALUDE_password_probability_l3232_323259


namespace NUMINAMATH_CALUDE_zCoordinate_when_x_is_seven_l3232_323289

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Calculate the z-coordinate for a given x-coordinate on the line -/
def zCoordinate (l : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given line, when x = 7, z = -4 -/
theorem zCoordinate_when_x_is_seven :
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 4, -1) }
  zCoordinate l 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_zCoordinate_when_x_is_seven_l3232_323289


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l3232_323286

-- Define the concept of a plane in 3D space
class Plane :=
  (normal : ℝ → ℝ → ℝ → ℝ)

-- Define the concept of a line in 3D space
class Line :=
  (direction : ℝ → ℝ → ℝ)

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define what it means for a point to be outside a plane
def PointOutsidePlane (p : Point) (plane : Plane) : Prop := sorry

-- Define what it means for a line to pass through a point
def LinePassesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- Define perpendicularity between a line and a plane
def LinePerpendicular (l : Line) (plane : Plane) : Prop := sorry

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (p : Point) (h : PointOutsidePlane p plane) :
  ∃! l : Line, LinePassesThroughPoint l p ∧ LinePerpendicular l plane :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l3232_323286


namespace NUMINAMATH_CALUDE_odd_function_extension_l3232_323264

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x for x > 0
  (∀ x < 0, f x = -x^2 - 2*x) :=  -- f(x) = -x^2 - 2x for x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3232_323264


namespace NUMINAMATH_CALUDE_dead_to_total_ratio_is_three_to_five_l3232_323235

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the problem setup -/
structure FlowerProblem where
  desired_flowers : ℕ
  seeds_per_pack : ℕ
  price_per_pack : ℕ
  total_spent : ℕ

/-- Calculates the ratio of dead seeds to total seeds -/
def dead_to_total_ratio (p : FlowerProblem) : Ratio :=
  let total_packs := p.total_spent / p.price_per_pack
  let total_seeds := total_packs * p.seeds_per_pack
  let dead_seeds := total_seeds - p.desired_flowers
  { numerator := dead_seeds, denominator := total_seeds }

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- The main theorem to prove -/
theorem dead_to_total_ratio_is_three_to_five (p : FlowerProblem) 
    (h1 : p.desired_flowers = 20)
    (h2 : p.seeds_per_pack = 25)
    (h3 : p.price_per_pack = 5)
    (h4 : p.total_spent = 10) : 
    simplify_ratio (dead_to_total_ratio p) = { numerator := 3, denominator := 5 } := by
  sorry

end NUMINAMATH_CALUDE_dead_to_total_ratio_is_three_to_five_l3232_323235


namespace NUMINAMATH_CALUDE_towel_shrinkage_l3232_323269

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) :
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ new_breadth : ℝ, new_breadth = 0.9 * B ∧ new_length * new_breadth = new_area :=
by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l3232_323269


namespace NUMINAMATH_CALUDE_milkshake_cost_l3232_323230

-- Define the initial amount
def initial_amount : ℚ := 15

-- Define the fraction spent on cupcakes
def cupcake_fraction : ℚ := 1/3

-- Define the fraction spent on sandwich
def sandwich_fraction : ℚ := 1/5

-- Define the final amount
def final_amount : ℚ := 4

-- Theorem to prove the cost of the milkshake
theorem milkshake_cost : 
  initial_amount * (1 - cupcake_fraction) * (1 - sandwich_fraction) - final_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_milkshake_cost_l3232_323230


namespace NUMINAMATH_CALUDE_max_pigs_buyable_l3232_323288

def budget : ℕ := 1300
def pig_cost : ℕ := 21
def duck_cost : ℕ := 23
def min_ducks : ℕ := 20

theorem max_pigs_buyable :
  ∀ p d : ℕ,
    p > 0 →
    d ≥ min_ducks →
    pig_cost * p + duck_cost * d ≤ budget →
    p ≤ 40 ∧
    ∃ p' d' : ℕ, p' = 40 ∧ d' ≥ min_ducks ∧ pig_cost * p' + duck_cost * d' = budget :=
by sorry

end NUMINAMATH_CALUDE_max_pigs_buyable_l3232_323288


namespace NUMINAMATH_CALUDE_kerry_candle_cost_l3232_323236

/-- The cost of candles for Kerry's birthday cakes -/
def candle_cost (num_cakes : ℕ) (age : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let total_candles := num_cakes * age
  let boxes_needed := (total_candles + candles_per_box - 1) / candles_per_box
  boxes_needed * cost_per_box

/-- Proof that the cost of candles for Kerry's birthday is $5 -/
theorem kerry_candle_cost :
  candle_cost 3 8 12 (5/2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_kerry_candle_cost_l3232_323236


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3232_323294

theorem quadratic_rewrite_ratio (j : ℝ) :
  let original := 8 * j^2 - 6 * j + 16
  ∃ (c p q : ℝ), 
    (∀ j, original = c * (j + p)^2 + q) ∧
    q / p = -119 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3232_323294


namespace NUMINAMATH_CALUDE_percent_less_than_l3232_323214

theorem percent_less_than (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_percent_less_than_l3232_323214


namespace NUMINAMATH_CALUDE_power_product_cube_l3232_323292

theorem power_product_cube (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l3232_323292


namespace NUMINAMATH_CALUDE_hexagonal_grid_toothpicks_l3232_323293

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of toothpicks in each side of the hexagon -/
def toothpicks_per_side : ℕ := 6

/-- The total number of toothpicks used to build the hexagonal grid -/
def total_toothpicks : ℕ := hexagon_sides * toothpicks_per_side

/-- Theorem: The total number of toothpicks used to build the hexagonal grid is 36 -/
theorem hexagonal_grid_toothpicks :
  total_toothpicks = 36 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_toothpicks_l3232_323293


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3232_323274

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 5 ∧ 1 / (x₀ + 2) + 1 / (y₀ + 2) = 4 / 9 :=
by sorry


end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3232_323274


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l3232_323262

theorem employee_payment_percentage (total_payment y_payment : ℚ) 
  (h1 : total_payment = 616)
  (h2 : y_payment = 280) : 
  (total_payment - y_payment) / y_payment * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l3232_323262


namespace NUMINAMATH_CALUDE_cistern_emptying_rate_l3232_323280

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 75
def rate_combined : ℚ := 1 / 50

-- Define the theorem
theorem cistern_emptying_rate :
  ∃ (rate_C : ℚ), 
    rate_A + rate_B - rate_C = rate_combined ∧ 
    rate_C = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_rate_l3232_323280


namespace NUMINAMATH_CALUDE_extreme_points_sum_condition_l3232_323270

open Real

noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 + a * log x - (a + 1) * x

noncomputable def F (a x : ℝ) : ℝ := f a x + (a - 1) * x

theorem extreme_points_sum_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x > 0 → F a x ≤ max (F a x₁) (F a x₂)) ∧
    F a x₁ + F a x₂ > -2/exp 1 - 2) →
  0 < a ∧ a < 1/exp 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_sum_condition_l3232_323270


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3232_323255

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3232_323255


namespace NUMINAMATH_CALUDE_group_composition_l3232_323209

/-- Proves that in a group of 300 people, where the number of men is twice the number of women,
    and the number of women is 3 times the number of children, the number of children is 30. -/
theorem group_composition (total : ℕ) (children : ℕ) (women : ℕ) (men : ℕ) 
    (h1 : total = 300)
    (h2 : men = 2 * women)
    (h3 : women = 3 * children)
    (h4 : total = children + women + men) : 
  children = 30 := by
sorry

end NUMINAMATH_CALUDE_group_composition_l3232_323209


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3232_323218

def solution_set : Set (ℤ × ℤ) := {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)}

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3232_323218


namespace NUMINAMATH_CALUDE_A_divisibility_l3232_323231

/-- The number consisting of 3^n ones -/
def A (n : ℕ) : ℕ :=
  (10^(3^n) - 1) / 9

/-- The theorem stating that A(n) is divisible by 3^n but not by 3^(n+1) -/
theorem A_divisibility (n : ℕ) : 
  (∃ k : ℕ, A n = k * (3^n)) ∧ ¬(∃ k : ℕ, A n = k * (3^(n+1))) :=
sorry

end NUMINAMATH_CALUDE_A_divisibility_l3232_323231


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l3232_323298

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = √3 and (2c-a)/b * cos(B) = cos(A), prove that a+c is in the range (√3, 2√3]. -/
theorem triangle_side_sum_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = Real.sqrt 3 →
  (2*c - a)/b * Real.cos B = Real.cos A →
  ∃ (x : ℝ), Real.sqrt 3 < x ∧ x ≤ 2 * Real.sqrt 3 ∧ a + c = x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l3232_323298


namespace NUMINAMATH_CALUDE_cafe_chairs_distribution_l3232_323205

theorem cafe_chairs_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_indoor_table : ℕ) (total_chairs : ℕ) : 
  indoor_tables = 9 → 
  outdoor_tables = 11 → 
  chairs_per_indoor_table = 10 → 
  total_chairs = 123 → 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_cafe_chairs_distribution_l3232_323205


namespace NUMINAMATH_CALUDE_student_group_combinations_l3232_323225

theorem student_group_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 4 + Nat.choose n 5 = 126 := by
  sorry

#check student_group_combinations

end NUMINAMATH_CALUDE_student_group_combinations_l3232_323225


namespace NUMINAMATH_CALUDE_goldbach_2024_l3232_323248

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_2024 :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 2024 :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l3232_323248


namespace NUMINAMATH_CALUDE_base5_412_equals_base7_212_l3232_323266

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : Nat) : Nat :=
  sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : Nat) : Nat :=
  sorry

/-- Theorem stating that 412₅ is equal to 212₇ --/
theorem base5_412_equals_base7_212 :
  decimalToBase7 (base5ToDecimal 412) = 212 :=
sorry

end NUMINAMATH_CALUDE_base5_412_equals_base7_212_l3232_323266


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l3232_323281

theorem roots_sum_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a/b + b/a = 18) := by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l3232_323281


namespace NUMINAMATH_CALUDE_quadratic_maximum_l3232_323247

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

/-- The maximum value of the function -/
def y_max : ℝ := 24

theorem quadratic_maximum :
  (∀ x : ℝ, f x ≤ y_max) ∧ f x_max = y_max :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l3232_323247


namespace NUMINAMATH_CALUDE_sum_of_roots_l3232_323275

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3232_323275


namespace NUMINAMATH_CALUDE_max_absolute_value_complex_l3232_323233

theorem max_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - 8 * Complex.I) = 20) :
  Complex.abs z ≤ Real.sqrt 222 :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_complex_l3232_323233


namespace NUMINAMATH_CALUDE_abs_plus_one_minimum_l3232_323278

theorem abs_plus_one_minimum :
  ∃ (min : ℝ) (x₀ : ℝ), (∀ x : ℝ, min ≤ |x| + 1) ∧ (min = |x₀| + 1) ∧ (min = 1 ∧ x₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_plus_one_minimum_l3232_323278


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3232_323202

theorem solution_satisfies_system :
  let x : ℝ := -8/3
  let y : ℝ := -4/5
  (Real.sqrt (1 - 3*x) - 1 = Real.sqrt (5*y - 3*x)) ∧
  (Real.sqrt (5 - 5*y) + Real.sqrt (5*y - 3*x) = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3232_323202


namespace NUMINAMATH_CALUDE_alabama_theorem_l3232_323251

/-- The number of letters in the word "ALABAMA" -/
def total_letters : ℕ := 7

/-- The number of 'A's in the word "ALABAMA" -/
def number_of_as : ℕ := 4

/-- The number of unique arrangements of the letters in "ALABAMA" -/
def alabama_arrangements : ℕ := total_letters.factorial / number_of_as.factorial

theorem alabama_theorem : alabama_arrangements = 210 := by
  sorry

end NUMINAMATH_CALUDE_alabama_theorem_l3232_323251


namespace NUMINAMATH_CALUDE_composite_function_problem_l3232_323239

-- Definition of composite function for linear functions
def composite_function (k₁ b₁ k₂ b₂ : ℝ) : ℝ → ℝ :=
  λ x => (k₁ + k₂) * x + b₁ * b₂

theorem composite_function_problem :
  -- 1. Composite of y=3x+2 and y=-4x+3
  (∀ x, composite_function 3 2 (-4) 3 x = -x + 6) ∧
  -- 2. If composite of y=ax-2 and y=-x+b is y=3x+2, then a=4 and b=-1
  (∀ a b, (∀ x, composite_function a (-2) (-1) b x = 3 * x + 2) → a = 4 ∧ b = -1) ∧
  -- 3. Conditions for passing through first, second, and fourth quadrants
  (∀ k b, (∀ x, (composite_function (-1) b k (-3) x > 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x < 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x > 0 ∧ x < 0)) →
    k < 1 ∧ b < 0) ∧
  -- 4. Fixed point of composite of y=-2x+m and y=3mx-6
  (∀ m, composite_function (-2) m (3*m) (-6) 2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_composite_function_problem_l3232_323239


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l3232_323211

/-- 
Given a man who swims against a current, this theorem proves his swimming speed in still water.
-/
theorem mans_swimming_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : distance = 40) 
  (h2 : time = 5) 
  (h3 : current_speed = 12) : 
  ∃ (speed : ℝ), speed = 20 ∧ distance = time * (speed - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l3232_323211


namespace NUMINAMATH_CALUDE_quadratic_properties_l3232_323254

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a

-- Theorem statement
theorem quadratic_properties (a : ℝ) :
  (∀ x y, x < 1 ∧ y < 1 ∧ x < y → f a x > f a y) ∧
  (∃ x, f a x = 0 → a ≤ 4) ∧
  (¬(a = 3 → ∀ x, f a x > 0 ↔ 1 < x ∧ x < 3)) ∧
  (∀ b, f a 2013 = b → f a (-2009) = b) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3232_323254


namespace NUMINAMATH_CALUDE_exists_translation_without_interior_points_l3232_323257

-- Define a figure on a grid plane
def Figure := Set ℝ × ℝ

-- Define the area of a figure
def area (F : Figure) : ℝ := sorry

-- Define a translation of a figure
def translate (F : Figure) (v : ℝ × ℝ) : Figure := sorry

-- Define if a point is in the interior of a figure
def isInterior (p : ℤ × ℤ) (F : Figure) : Prop := sorry

-- Main theorem
theorem exists_translation_without_interior_points (F : Figure) 
  (h : area F < 1) :
  ∃ v : ℝ × ℝ, ∀ p : ℤ × ℤ, ¬isInterior p (translate F v) := by
  sorry

end NUMINAMATH_CALUDE_exists_translation_without_interior_points_l3232_323257


namespace NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l3232_323268

/-- Proves the equivalence of rectangular and polar coordinate equations --/
theorem rectangular_to_polar_equivalence 
  (x y ρ θ : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) : 
  y^2 = 12*x ↔ ρ * Real.sin θ^2 = 12 * Real.cos θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l3232_323268


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l3232_323237

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The condition that f(x) intersects y = 3 at only one point -/
def intersects_at_one_point (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The theorem statement -/
theorem intersection_implies_a_range :
  ∀ a : ℝ, intersects_at_one_point a → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l3232_323237


namespace NUMINAMATH_CALUDE_unique_k_value_l3232_323252

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 63x + k = 0 with prime roots -/
def quadratic_equation (k : ℕ) (x : ℝ) : Prop :=
  x^2 - 63*x + k = 0

/-- The roots of the quadratic equation are prime numbers -/
def roots_are_prime (k : ℕ) : Prop :=
  ∃ (a b : ℕ), (is_prime a ∧ is_prime b) ∧
  (∀ x : ℝ, quadratic_equation k x ↔ (x = a ∨ x = b))

theorem unique_k_value : ∃! k : ℕ, roots_are_prime k ∧ k = 122 :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l3232_323252


namespace NUMINAMATH_CALUDE_function_inequality_l3232_323242

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) : 
  f 2 + g 1 > f 1 + g 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3232_323242


namespace NUMINAMATH_CALUDE_beta_value_l3232_323222

open Real

def operation (a b c d : ℝ) : ℝ := a * d - b * c

theorem beta_value (α β : ℝ) : 
  cos α = 1/7 →
  operation (sin α) (sin β) (cos α) (cos β) = 3 * Real.sqrt 3 / 14 →
  0 < β →
  β < α →
  α < π/2 →
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_beta_value_l3232_323222


namespace NUMINAMATH_CALUDE_seashell_difference_l3232_323203

theorem seashell_difference (craig_shells : ℕ) (craig_ratio : ℕ) (brian_ratio : ℕ) : 
  craig_shells = 54 → 
  craig_ratio = 9 → 
  brian_ratio = 7 → 
  craig_shells - (craig_shells / craig_ratio * brian_ratio) = 12 := by
sorry

end NUMINAMATH_CALUDE_seashell_difference_l3232_323203


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l3232_323204

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  rectangleWidth : ℕ
  rectangleHeight : ℕ
  lShapeOutward : ℕ
  lShapeDownward : ℕ

/-- Calculate the perimeter of the SquareFigure -/
def calculatePerimeter (figure : SquareFigure) : ℝ :=
  let bottomLength := figure.rectangleWidth * figure.squareSideLength
  let topLength := (figure.rectangleWidth + figure.lShapeOutward) * figure.squareSideLength
  let leftHeight := figure.rectangleHeight * figure.squareSideLength
  let rightHeight := (figure.rectangleHeight + figure.lShapeDownward) * figure.squareSideLength
  bottomLength + topLength + leftHeight + rightHeight

/-- Theorem stating that the perimeter of the specific figure is 26 units -/
theorem specific_figure_perimeter :
  let figure : SquareFigure := {
    squareSideLength := 2
    rectangleWidth := 3
    rectangleHeight := 2
    lShapeOutward := 2
    lShapeDownward := 1
  }
  calculatePerimeter figure = 26 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l3232_323204


namespace NUMINAMATH_CALUDE_y_value_l3232_323232

theorem y_value (x y z : ℤ) 
  (eq1 : x + y + z = 25) 
  (eq2 : x + y = 19) 
  (eq3 : y + z = 18) : 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_y_value_l3232_323232


namespace NUMINAMATH_CALUDE_balloon_blowup_ratio_l3232_323243

theorem balloon_blowup_ratio (total : ℕ) (intact : ℕ) : 
  total = 200 → 
  intact = 80 → 
  (total - intact) / (total / 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_blowup_ratio_l3232_323243


namespace NUMINAMATH_CALUDE_smallest_angle_representation_l3232_323217

theorem smallest_angle_representation (k : ℤ) (α : ℝ) : 
  (19 * π / 5 = 2 * k * π + α) → 
  (∀ β : ℝ, ∃ m : ℤ, 19 * π / 5 = 2 * m * π + β → |α| ≤ |β|) → 
  α = -π / 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_representation_l3232_323217


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3232_323234

theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 34) = Real.sqrt 6664 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3232_323234


namespace NUMINAMATH_CALUDE_card_arrangement_unique_l3232_323241

def CardArrangement (arrangement : List Nat) : Prop :=
  arrangement.length = 9 ∧
  arrangement.toFinset = Finset.range 9 ∧
  ∀ i, i + 2 < arrangement.length →
    ¬(arrangement[i]! < arrangement[i+1]! ∧ arrangement[i+1]! < arrangement[i+2]!) ∧
    ¬(arrangement[i]! > arrangement[i+1]! ∧ arrangement[i+1]! > arrangement[i+2]!)

theorem card_arrangement_unique :
  ∀ arrangement : List Nat,
    CardArrangement arrangement →
    arrangement[3]! = 5 ∧
    arrangement[5]! = 2 ∧
    arrangement[8]! = 9 :=
by sorry

end NUMINAMATH_CALUDE_card_arrangement_unique_l3232_323241


namespace NUMINAMATH_CALUDE_pat_stickers_l3232_323213

theorem pat_stickers (initial_stickers given_away_stickers : ℝ) 
  (h1 : initial_stickers = 39.0)
  (h2 : given_away_stickers = 22.0) :
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l3232_323213


namespace NUMINAMATH_CALUDE_equation_solutions_l3232_323273

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 5) / 2 ∧ x₂ = (-1 - Real.sqrt 5) / 2 ∧
    x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧
    2*(x₁ - 3) = 3*x₁*(x₁ - 3) ∧ 2*(x₂ - 3) = 3*x₂*(x₂ - 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3232_323273


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3232_323296

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_diameter : ℝ)
  (hw : rectangle_width = 15)
  (hh : rectangle_height = 17)
  (hd : circle_diameter = 7) :
  let inner_width := rectangle_width - circle_diameter
  let inner_height := rectangle_height - circle_diameter
  Real.sqrt (inner_width ^ 2 + inner_height ^ 2) = Real.sqrt 164 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3232_323296


namespace NUMINAMATH_CALUDE_simplify_polynomial_product_l3232_323277

theorem simplify_polynomial_product (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^10 + 4*x^9 + x^8) =
  15*x^13 - x^12 + 15*x^11 - 2*x^10 - 5*x^9 - 2*x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_product_l3232_323277


namespace NUMINAMATH_CALUDE_sequence_product_representation_l3232_323200

theorem sequence_product_representation (n a : ℕ) :
  ∃ u v : ℕ, (n : ℚ) / (n + a) = (u : ℚ) / (u + a) * (v : ℚ) / (v + a) := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_representation_l3232_323200


namespace NUMINAMATH_CALUDE_power_of_power_l3232_323285

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3232_323285
