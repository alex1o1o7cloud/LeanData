import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_for_ten_trees_l2192_219206

/-- Calculates the minimum distance to water trees -/
def min_distance_to_water_trees (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  let max_trees_per_trip := 2  -- Xiao Zhang can water 2 trees per trip
  let num_full_trips := (num_trees - 1) / max_trees_per_trip
  let trees_on_last_trip := (num_trees - 1) % max_trees_per_trip + 1
  let full_trip_distance := num_full_trips * (max_trees_per_trip * tree_spacing * 2)
  let last_trip_distance := trees_on_last_trip * tree_spacing
  full_trip_distance + last_trip_distance

/-- The theorem to be proved -/
theorem min_distance_for_ten_trees :
  min_distance_to_water_trees 10 10 = 410 :=
sorry

end NUMINAMATH_CALUDE_min_distance_for_ten_trees_l2192_219206


namespace NUMINAMATH_CALUDE_total_musicians_is_98_l2192_219294

/-- The total number of musicians in the orchestra, band, and choir -/
def total_musicians (orchestra_males orchestra_females band_multiplier choir_males choir_females : ℕ) : ℕ :=
  let orchestra_total := orchestra_males + orchestra_females
  let band_total := band_multiplier * orchestra_total
  let choir_total := choir_males + choir_females
  orchestra_total + band_total + choir_total

/-- Theorem stating that the total number of musicians is 98 given the specific conditions -/
theorem total_musicians_is_98 :
  total_musicians 11 12 2 12 17 = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_musicians_is_98_l2192_219294


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2192_219200

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2192_219200


namespace NUMINAMATH_CALUDE_jessica_has_two_balloons_l2192_219268

/-- The number of blue balloons Jessica has -/
def jessicas_balloons (joan_initial : ℕ) (popped : ℕ) (total_now : ℕ) : ℕ :=
  total_now - (joan_initial - popped)

/-- Theorem: Jessica has 2 blue balloons -/
theorem jessica_has_two_balloons :
  jessicas_balloons 9 5 6 = 2 := by sorry

end NUMINAMATH_CALUDE_jessica_has_two_balloons_l2192_219268


namespace NUMINAMATH_CALUDE_combined_body_is_pentahedron_l2192_219216

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  edge_length : ℝ

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- Represents the new geometric body formed by combining a regular quadrangular pyramid and a regular tetrahedron -/
structure CombinedBody where
  pyramid : RegularQuadrangularPyramid
  tetrahedron : RegularTetrahedron

/-- Defines the property of being a pentahedron -/
def is_pentahedron (body : CombinedBody) : Prop := sorry

theorem combined_body_is_pentahedron 
  (pyramid : RegularQuadrangularPyramid) 
  (tetrahedron : RegularTetrahedron) 
  (h : pyramid.edge_length = tetrahedron.edge_length) : 
  is_pentahedron (CombinedBody.mk pyramid tetrahedron) :=
sorry

end NUMINAMATH_CALUDE_combined_body_is_pentahedron_l2192_219216


namespace NUMINAMATH_CALUDE_james_weekly_take_home_pay_l2192_219280

/-- Calculates James' weekly take-home pay given his work and tax conditions --/
def jamesTakeHomePay (mainJobRate hourlyRate : ℝ) 
                     (secondJobRatePercentage : ℝ) 
                     (mainJobHours overtimeHours : ℕ) 
                     (secondJobHours : ℕ) 
                     (weekendDays : ℕ)
                     (weekendRate : ℝ)
                     (taxDeductions : ℝ)
                     (federalTaxRate stateTaxRate : ℝ) : ℝ :=
  let secondJobRate := mainJobRate * (1 - secondJobRatePercentage)
  let regularHours := mainJobHours - overtimeHours
  let mainJobEarnings := regularHours * mainJobRate + overtimeHours * mainJobRate * 1.5
  let secondJobEarnings := secondJobHours * secondJobRate
  let weekendEarnings := weekendDays * weekendRate
  let totalEarnings := mainJobEarnings + secondJobEarnings + weekendEarnings
  let taxableIncome := totalEarnings - taxDeductions
  let federalTax := taxableIncome * federalTaxRate
  let stateTax := taxableIncome * stateTaxRate
  let totalTaxes := federalTax + stateTax
  totalEarnings - totalTaxes

/-- Theorem stating that James' weekly take-home pay is $885.30 --/
theorem james_weekly_take_home_pay :
  jamesTakeHomePay 20 20 0.2 30 5 15 2 100 200 0.18 0.05 = 885.30 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_take_home_pay_l2192_219280


namespace NUMINAMATH_CALUDE_inequality_proof_l2192_219292

theorem inequality_proof (x : ℝ) (hx : x ≠ 0) :
  max 0 (Real.log (abs x)) ≥ 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
    x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
    x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2192_219292


namespace NUMINAMATH_CALUDE_total_scoops_needed_l2192_219208

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def scoop_size : ℚ := 1/3

theorem total_scoops_needed : 
  (flour_cups / scoop_size + sugar_cups / scoop_size : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_scoops_needed_l2192_219208


namespace NUMINAMATH_CALUDE_expression_simplification_l2192_219240

theorem expression_simplification : 
  ((3 + 4 + 5 + 6)^2 / 4) + ((3 * 6 + 9)^2 / 3) = 324 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2192_219240


namespace NUMINAMATH_CALUDE_petya_wins_petya_wins_game_l2192_219260

/-- Represents the game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25
  , prob_two_caramels := 0.54 }

/-- Theorem stating that Petya has a higher chance of winning the game. -/
theorem petya_wins (g : CandyGame) : g.prob_two_caramels > 0.5 → 
  (1 - g.prob_two_caramels) < 0.5 := by
  sorry

/-- Corollary proving that Petya wins the specific game instance. -/
theorem petya_wins_game : (1 - game.prob_two_caramels) < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_petya_wins_petya_wins_game_l2192_219260


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2192_219234

theorem smallest_factor_for_perfect_square (n : ℕ) : n = 7 ↔ 
  (n > 0 ∧ 
   ∃ (m : ℕ), 1008 * n = m^2 ∧ 
   ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2192_219234


namespace NUMINAMATH_CALUDE_tangent_perpendicular_range_l2192_219259

/-- The range of a when the tangent lines of two specific curves are perpendicular -/
theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 (3/2) ∧ 
    ((a * x₀ + a - 1) * (x₀ - 2) = -1)) → 
  a ∈ Set.Icc 1 (3/2) := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_range_l2192_219259


namespace NUMINAMATH_CALUDE_ratio_calculation_l2192_219231

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2192_219231


namespace NUMINAMATH_CALUDE_stream_speed_l2192_219254

/-- Proves that the speed of a stream is 3 km/h given specific rowing conditions. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (trip_time : ℝ) 
  (h1 : downstream_distance = 75)
  (h2 : upstream_distance = 45)
  (h3 : trip_time = 5) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = trip_time * (boat_speed + stream_speed) ∧
    upstream_distance = trip_time * (boat_speed - stream_speed) ∧
    stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2192_219254


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2192_219241

theorem cookie_jar_problem (initial_cookies : ℕ) (x : ℕ) 
  (h1 : initial_cookies = 7)
  (h2 : initial_cookies - 1 = (initial_cookies + x) / 2) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2192_219241


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2192_219298

theorem smallest_k_no_real_roots : ∀ k : ℤ,
  (∀ x : ℝ, (1/2 : ℝ) * x^2 + 3*x + (k : ℝ) ≠ 0) ↔ k ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2192_219298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2192_219275

/-- Given that -1, a, b, c, -9 form an arithmetic sequence, prove that b = -5 and ac = 21 -/
theorem arithmetic_sequence_problem (a b c : ℝ) 
  (h1 : ∃ (d : ℝ), a - (-1) = d ∧ b - a = d ∧ c - b = d ∧ (-9) - c = d) : 
  b = -5 ∧ a * c = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2192_219275


namespace NUMINAMATH_CALUDE_gravelling_cost_l2192_219264

/-- The cost of gravelling intersecting roads on a rectangular lawn. -/
theorem gravelling_cost 
  (lawn_length lawn_width road_width gravel_cost_per_sqm : ℝ)
  (h_lawn_length : lawn_length = 70)
  (h_lawn_width : lawn_width = 30)
  (h_road_width : road_width = 5)
  (h_gravel_cost : gravel_cost_per_sqm = 4) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by sorry

end NUMINAMATH_CALUDE_gravelling_cost_l2192_219264


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2192_219201

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2192_219201


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l2192_219247

open Real

theorem triangle_side_ratio_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 3 * B ∧  -- Given condition
  a / sin A = b / sin B ∧  -- Sine rule
  a / sin A = c / sin C →  -- Sine rule
  1 < a / b ∧ a / b < 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_range_l2192_219247


namespace NUMINAMATH_CALUDE_special_function_ratio_l2192_219277

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, 2 * b^2 * f a = a^2 * f b

/-- The main theorem -/
theorem special_function_ratio 
  (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 6 ≠ 0) : 
  (f 7 - f 3) / f 6 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_ratio_l2192_219277


namespace NUMINAMATH_CALUDE_divide_forty_five_by_point_zero_five_l2192_219238

theorem divide_forty_five_by_point_zero_five : 45 / 0.05 = 900 := by
  sorry

end NUMINAMATH_CALUDE_divide_forty_five_by_point_zero_five_l2192_219238


namespace NUMINAMATH_CALUDE_diana_eraser_sharing_l2192_219248

theorem diana_eraser_sharing (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 3840) (h2 : erasers_per_friend = 80) :
  total_erasers / erasers_per_friend = 48 := by
  sorry

end NUMINAMATH_CALUDE_diana_eraser_sharing_l2192_219248


namespace NUMINAMATH_CALUDE_vector_lines_correct_l2192_219266

/-- Vector field in R³ -/
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (0, 9*z, -4*y)

/-- Vector lines of the given vector field -/
def vector_lines (x y z C₁ C₂ : ℝ) : Prop :=
  9 * z^2 + 4 * y^2 = C₁ ∧ x = C₂

/-- Theorem stating that the vector lines are correct for the given vector field -/
theorem vector_lines_correct :
  ∀ (x y z C₁ C₂ : ℝ),
    vector_lines x y z C₁ C₂ ↔
    ∃ (t : ℝ), (x, y, z) = (C₂, 
                            9 * t * (vector_field x y z).2.1, 
                            -4 * t * (vector_field x y z).2.2) :=
sorry

end NUMINAMATH_CALUDE_vector_lines_correct_l2192_219266


namespace NUMINAMATH_CALUDE_pickle_problem_l2192_219230

/-- Pickle problem theorem -/
theorem pickle_problem (jars : ℕ) (cucumbers : ℕ) (initial_vinegar : ℕ) 
  (pickles_per_cucumber : ℕ) (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ) :
  jars = 4 →
  cucumbers = 10 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  (initial_vinegar - remaining_vinegar) / vinegar_per_jar = jars →
  (cucumbers * pickles_per_cucumber) / jars = 15 :=
by sorry

end NUMINAMATH_CALUDE_pickle_problem_l2192_219230


namespace NUMINAMATH_CALUDE_playground_boundary_length_l2192_219236

/-- The total boundary length of a square playground with semi-circle arcs -/
theorem playground_boundary_length :
  let area : ℝ := 256
  let side_length : ℝ := Real.sqrt area
  let segment_length : ℝ := side_length / 4
  let arc_radius : ℝ := segment_length / 2
  let arc_length_per_side : ℝ := 3 * π * arc_radius
  let total_arc_length : ℝ := 4 * arc_length_per_side
  let straight_segment_length_per_side : ℝ := 3 * segment_length
  let total_straight_segment_length : ℝ := 4 * straight_segment_length_per_side
  total_arc_length + total_straight_segment_length = 24 * π + 48 := by
  sorry

end NUMINAMATH_CALUDE_playground_boundary_length_l2192_219236


namespace NUMINAMATH_CALUDE_medium_stores_sampled_l2192_219205

/-- Represents the total number of stores in the city -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores in the city -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores in the city -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores in the city -/
def small_ratio : ℕ := 9

/-- Represents the total number of stores to be sampled -/
def sample_size : ℕ := 30

/-- Theorem stating that the number of medium-sized stores to be sampled is 10 -/
theorem medium_stores_sampled : ℕ := by
  sorry

end NUMINAMATH_CALUDE_medium_stores_sampled_l2192_219205


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2192_219222

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 24*a^2 + 50*a - 14 = 0 →
  b^3 - 24*b^2 + 50*b - 14 = 0 →
  c^3 - 24*c^2 + 50*c - 14 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 476/15 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2192_219222


namespace NUMINAMATH_CALUDE_paths_from_A_to_C_l2192_219250

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- A point in the graph -/
inductive Point
| A
| B
| C

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem paths_from_A_to_C :
  (num_paths Point.A Point.B = 3) →
  (num_paths Point.B Point.C = 1) →
  (num_paths Point.A Point.C = 1) →
  total_paths = 4 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_C_l2192_219250


namespace NUMINAMATH_CALUDE_curve_symmetrical_y_axis_l2192_219263

-- Define a function to represent the left-hand side of the equation
def f (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that the curve is symmetrical with respect to the y-axis
theorem curve_symmetrical_y_axis : ∀ x y : ℝ, f x y = 1 ↔ f (-x) y = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetrical_y_axis_l2192_219263


namespace NUMINAMATH_CALUDE_divisibility_condition_l2192_219232

theorem divisibility_condition (n : ℤ) : 
  (∃ a : ℤ, n - 4 = 6 * a) → 
  (∃ b : ℤ, n - 8 = 10 * b) → 
  (∃ k : ℤ, n = 30 * k - 2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2192_219232


namespace NUMINAMATH_CALUDE_dog_cord_length_l2192_219267

theorem dog_cord_length (diameter : ℝ) (h : diameter = 30) : 
  diameter / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_cord_length_l2192_219267


namespace NUMINAMATH_CALUDE_base_conversion_142_to_7_l2192_219297

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 --/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_142_to_7 :
  toBase7 142 = [2, 6, 2] ∧ fromBase7 [2, 6, 2] = 142 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_142_to_7_l2192_219297


namespace NUMINAMATH_CALUDE_no_valid_subset_exists_l2192_219287

/-- The set M defined as the intersection of (0,1) and ℚ -/
def M : Set ℚ := Set.Ioo 0 1 ∩ Set.range Rat.cast

/-- Definition of a valid subset A -/
def is_valid_subset (A : Set ℚ) : Prop :=
  A ⊆ M ∧
  ∀ x ∈ M, ∃! (S : Finset ℚ), (S : Set ℚ) ⊆ A ∧ x = S.sum id

/-- Theorem stating that no valid subset A exists -/
theorem no_valid_subset_exists : ¬∃ A : Set ℚ, is_valid_subset A := by
  sorry

end NUMINAMATH_CALUDE_no_valid_subset_exists_l2192_219287


namespace NUMINAMATH_CALUDE_three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2192_219242

/-- The number of solutions to the system of equations
    x^2 - y^2 = 0 and (x-a)^2 + y^2 = 1 -/
def num_solutions (a : ℝ) : ℕ :=
  sorry

/-- The system has three solutions when a = ±1 -/
theorem three_solutions_at_plus_minus_one :
  (num_solutions 1 = 3) ∧ (num_solutions (-1) = 3) :=
sorry

/-- The system has two solutions when a = ±√2 -/
theorem two_solutions_at_plus_minus_sqrt_two :
  (num_solutions (Real.sqrt 2) = 2) ∧ (num_solutions (-(Real.sqrt 2)) = 2) :=
sorry

/-- The system has four solutions for all other values of a in (-√2, √2) except ±1 -/
theorem four_solutions_between_neg_sqrt_two_and_sqrt_two (a : ℝ) :
  a ∈ Set.Ioo (-(Real.sqrt 2)) (Real.sqrt 2) ∧ a ≠ 1 ∧ a ≠ -1 →
  num_solutions a = 4 :=
sorry

/-- The system has no solutions for |a| > √2 -/
theorem no_solutions_outside_sqrt_two (a : ℝ) :
  |a| > Real.sqrt 2 → num_solutions a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2192_219242


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2192_219293

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def phi_condition (φ : ℝ) : Prop :=
  ∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2

theorem sufficient_not_necessary :
  (∀ φ : ℝ, phi_condition φ → is_even_function (λ x => Real.sin (x + φ))) ∧
  (∃ φ : ℝ, is_even_function (λ x => Real.sin (x + φ)) ∧ ¬phi_condition φ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2192_219293


namespace NUMINAMATH_CALUDE_age_problem_l2192_219257

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2192_219257


namespace NUMINAMATH_CALUDE_enrique_commission_l2192_219251

/-- Calculates the commission earned by Enrique based on his sales --/
theorem enrique_commission :
  let commission_rate : ℚ := 15 / 100
  let suit_price : ℚ := 700
  let suit_quantity : ℕ := 2
  let shirt_price : ℚ := 50
  let shirt_quantity : ℕ := 6
  let loafer_price : ℚ := 150
  let loafer_quantity : ℕ := 2
  let total_sales : ℚ := suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity
  let commission : ℚ := commission_rate * total_sales
  commission = 300
  := by sorry

end NUMINAMATH_CALUDE_enrique_commission_l2192_219251


namespace NUMINAMATH_CALUDE_spencer_jump_rope_session_length_l2192_219246

/-- Proves that Spencer's jump rope session length is 10 minutes -/
theorem spencer_jump_rope_session_length :
  ∀ (jumps_per_minute : ℕ) 
    (sessions_per_day : ℕ) 
    (total_jumps : ℕ) 
    (total_days : ℕ),
  jumps_per_minute = 4 →
  sessions_per_day = 2 →
  total_jumps = 400 →
  total_days = 5 →
  (total_jumps / total_days / sessions_per_day) / jumps_per_minute = 10 :=
by
  sorry

#check spencer_jump_rope_session_length

end NUMINAMATH_CALUDE_spencer_jump_rope_session_length_l2192_219246


namespace NUMINAMATH_CALUDE_loss_equals_five_balls_l2192_219237

/-- Prove that the number of balls the loss equates to is 5 -/
theorem loss_equals_five_balls 
  (cost_price : ℕ) 
  (num_balls_sold : ℕ) 
  (selling_price : ℕ) 
  (h1 : cost_price = 72)
  (h2 : num_balls_sold = 15)
  (h3 : selling_price = 720) :
  (num_balls_sold * cost_price - selling_price) / cost_price = 5 := by
  sorry

#check loss_equals_five_balls

end NUMINAMATH_CALUDE_loss_equals_five_balls_l2192_219237


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2192_219244

def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

def cube_volume (s : ℝ) : ℝ := s^3

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 150) :
  ∃ s : ℝ, cube_surface_area s = surface_area ∧ cube_volume s = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2192_219244


namespace NUMINAMATH_CALUDE_total_legos_after_winning_l2192_219217

def initial_legos : ℕ := 2080
def won_legos : ℕ := 17

theorem total_legos_after_winning :
  initial_legos + won_legos = 2097 := by sorry

end NUMINAMATH_CALUDE_total_legos_after_winning_l2192_219217


namespace NUMINAMATH_CALUDE_min_floor_sum_l2192_219252

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l2192_219252


namespace NUMINAMATH_CALUDE_lcm_of_5_6_8_21_l2192_219279

theorem lcm_of_5_6_8_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 21)) = 840 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_8_21_l2192_219279


namespace NUMINAMATH_CALUDE_carriage_equation_correct_l2192_219213

/-- Represents the scenario of people and carriages as described in the ancient Chinese problem --/
def carriage_problem (x : ℕ) : Prop :=
  -- Three people sharing a carriage leaves two carriages empty
  (3 * (x - 2) : ℤ) = (3 * x - 6 : ℤ) ∧
  -- Two people sharing a carriage leaves nine people walking
  (2 * x + 9 : ℤ) = (3 * x - 6 : ℤ)

/-- The equation 3(x-2) = 2x + 9 correctly represents the carriage problem --/
theorem carriage_equation_correct (x : ℕ) :
  carriage_problem x ↔ (3 * (x - 2) : ℤ) = (2 * x + 9 : ℤ) :=
sorry

end NUMINAMATH_CALUDE_carriage_equation_correct_l2192_219213


namespace NUMINAMATH_CALUDE_faster_car_distance_l2192_219211

/-- Two cars driving towards each other, with one twice as fast as the other and initial distance of 4 miles -/
structure TwoCars where
  slow_speed : ℝ
  fast_speed : ℝ
  initial_distance : ℝ
  slow_distance : ℝ
  fast_distance : ℝ
  meeting_condition : slow_distance + fast_distance = initial_distance
  speed_relation : fast_speed = 2 * slow_speed
  distance_relation : fast_distance = 2 * slow_distance

/-- The theorem stating that the faster car travels 8/3 miles when they meet -/
theorem faster_car_distance (cars : TwoCars) (h : cars.initial_distance = 4) :
  cars.fast_distance = 8/3 := by
  sorry

#check faster_car_distance

end NUMINAMATH_CALUDE_faster_car_distance_l2192_219211


namespace NUMINAMATH_CALUDE_unique_solution_l2192_219281

def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / 999

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_solution (a b c : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c →
  repeating_decimal_2 a b + repeating_decimal_3 a b c = 35 / 37 →
  a = 5 ∧ b = 3 ∧ c = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2192_219281


namespace NUMINAMATH_CALUDE_first_jump_over_2km_l2192_219226

def jump_sequence (n : ℕ) : ℕ :=
  2 * 3^(n - 1)

theorem first_jump_over_2km :
  (∀ k < 8, jump_sequence k ≤ 2000) ∧ jump_sequence 8 > 2000 :=
sorry

end NUMINAMATH_CALUDE_first_jump_over_2km_l2192_219226


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2192_219296

/-- Triangle ABC with vertices A(-1,5), B(-2,-1), and C(4,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (-1, 5)
  , B := (-2, -1)
  , C := (4, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Definition of an altitude in a triangle -/
def isAltitude (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties :
  let t := triangleABC
  let altitude := Line.mk 3 2 (-7)
  isAltitude t altitude ∧ triangleArea t = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2192_219296


namespace NUMINAMATH_CALUDE_tangent_to_both_circumcircles_l2192_219256

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b : Point)

structure Circle := (center : Point) (radius : ℝ)

-- Define the parallelogram
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define a point between two other points
def PointBetween (E B F : Point) : Prop := sorry

-- Define the intersection of two lines
def Intersect (l₁ l₂ : Line) (O : Point) : Prop := sorry

-- Define a line tangent to a circle
def Tangent (l : Line) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def Circumcircle (A B C : Point) : Circle := sorry

-- Main theorem
theorem tangent_to_both_circumcircles 
  (A B C D E F O : Point) 
  (h1 : Parallelogram A B C D)
  (h2 : PointBetween E B F)
  (h3 : Intersect (Line.mk A C) (Line.mk B D) O)
  (h4 : Tangent (Line.mk A E) (Circumcircle A O D))
  (h5 : Tangent (Line.mk D F) (Circumcircle A O D)) :
  Tangent (Line.mk A E) (Circumcircle E O F) ∧ 
  Tangent (Line.mk D F) (Circumcircle E O F) := by sorry

end NUMINAMATH_CALUDE_tangent_to_both_circumcircles_l2192_219256


namespace NUMINAMATH_CALUDE_house_amenities_l2192_219255

theorem house_amenities (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) :
  total = 65 → garage = 50 → pool = 40 → neither = 10 →
  ∃ both : ℕ, both = 35 ∧ garage + pool - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_house_amenities_l2192_219255


namespace NUMINAMATH_CALUDE_circle_radius_satisfies_condition_l2192_219262

/-- The radius of a circle satisfying the given condition -/
def circle_radius : ℝ := 8

/-- The condition that the product of four inches and the circumference equals the area -/
def circle_condition (r : ℝ) : Prop := 4 * (2 * Real.pi * r) = Real.pi * r^2

/-- Theorem stating that the radius satisfies the condition -/
theorem circle_radius_satisfies_condition : 
  circle_condition circle_radius := by sorry

end NUMINAMATH_CALUDE_circle_radius_satisfies_condition_l2192_219262


namespace NUMINAMATH_CALUDE_student_calculation_l2192_219253

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 124 → 
  (2 * chosen_number) - 138 = 110 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_l2192_219253


namespace NUMINAMATH_CALUDE_santa_mandarins_l2192_219214

/-- Represents the exchange game with Santa Claus --/
structure ExchangeGame where
  /-- Number of first type exchanges (5 mandarins for 3 firecrackers and 1 candy) --/
  first_exchanges : ℕ
  /-- Number of second type exchanges (2 firecrackers for 3 mandarins and 1 candy) --/
  second_exchanges : ℕ
  /-- Total number of candies received --/
  total_candies : ℕ
  /-- Constraint: Total exchanges equal total candies --/
  exchanges_eq_candies : first_exchanges + second_exchanges = total_candies
  /-- Constraint: Firecrackers balance out --/
  firecrackers_balance : 3 * first_exchanges = 2 * second_exchanges

/-- The main theorem to prove --/
theorem santa_mandarins (game : ExchangeGame) (h : game.total_candies = 50) :
  5 * game.first_exchanges - 3 * game.second_exchanges = 10 := by
  sorry

end NUMINAMATH_CALUDE_santa_mandarins_l2192_219214


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l2192_219273

theorem cylinder_height_comparison (r₁ r₂ h₁ h₂ : ℝ) :
  r₁ > 0 ∧ r₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 →
  r₂ = 1.1 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.21 * h₂ :=
by sorry

#check cylinder_height_comparison

end NUMINAMATH_CALUDE_cylinder_height_comparison_l2192_219273


namespace NUMINAMATH_CALUDE_min_value_bound_l2192_219271

noncomputable section

variable (a : ℝ) (x₀ : ℝ)

def f (x : ℝ) := Real.exp x - (a * x) / (x + 1)

theorem min_value_bound (h1 : a > 0) (h2 : x₀ > -1) 
  (h3 : ∀ x > -1, f a x ≥ f a x₀) : f a x₀ ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_bound_l2192_219271


namespace NUMINAMATH_CALUDE_largest_decimal_l2192_219243

theorem largest_decimal : ∀ (a b c d e : ℚ),
  a = 0.936 ∧ b = 0.9358 ∧ c = 0.9361 ∧ d = 0.935 ∧ e = 0.921 →
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2192_219243


namespace NUMINAMATH_CALUDE_parking_lot_problem_l2192_219289

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bicycle : Nat
  motorcycle : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bicycles : Nat
  motorcycles : Nat

/-- The theorem stating the relationship between the number of cars and motorcycles -/
theorem parking_lot_problem (wheels : VehicleWheels) (count : VehicleCount) :
  wheels.car = 4 →
  wheels.bicycle = 2 →
  wheels.motorcycle = 2 →
  count.bicycles = 2 * count.motorcycles →
  wheels.car * count.cars + wheels.bicycle * count.bicycles + wheels.motorcycle * count.motorcycles = 196 →
  count.cars = (98 - 3 * count.motorcycles) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l2192_219289


namespace NUMINAMATH_CALUDE_parabola_line_single_intersection_l2192_219215

/-- The value of a that makes the parabola y = ax^2 + 3x + 1 intersect
    the line y = -2x - 3 at only one point is 25/16 -/
theorem parabola_line_single_intersection :
  ∃! a : ℚ, ∀ x : ℚ,
    (a * x^2 + 3 * x + 1 = -2 * x - 3) →
    (∀ y : ℚ, y ≠ x → a * y^2 + 3 * y + 1 ≠ -2 * y - 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_single_intersection_l2192_219215


namespace NUMINAMATH_CALUDE_propositions_truth_l2192_219239

theorem propositions_truth : 
  (∀ a b : ℝ, a > 1 → b > 1 → a * b > 1) ∧ 
  (∃ a b c : ℝ, b = Real.sqrt (a * c) ∧ ¬(∃ r : ℝ, b = a * r ∧ c = b * r)) ∧
  (∃ a b c : ℝ, (∃ r : ℝ, b = a * r ∧ c = b * r) ∧ b ≠ Real.sqrt (a * c)) :=
by sorry


end NUMINAMATH_CALUDE_propositions_truth_l2192_219239


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l2192_219207

theorem largest_n_binomial_equality : ∃ n : ℕ, n = 6 ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l2192_219207


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l2192_219284

theorem cubic_root_sum_product (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let p : ℝ → ℝ := λ x => α * x^3 - α * x^2 + β * x + β
  ∀ x₁ x₂ x₃ : ℝ, (p x₁ = 0 ∧ p x₂ = 0 ∧ p x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l2192_219284


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2192_219274

/-- A type representing a 17-digit number -/
def Digit17 := Fin 10 → Fin 10

/-- Reverses a 17-digit number -/
def reverse (n : Digit17) : Digit17 :=
  fun i => n (16 - i)

/-- Adds two 17-digit numbers -/
def add (a b : Digit17) : Digit17 :=
  sorry

/-- Checks if a number has at least one even digit -/
def hasEvenDigit (n : Digit17) : Prop :=
  ∃ i, (n i).val % 2 = 0

/-- Main theorem: For any 17-digit number, when added to its reverse, 
    the resulting sum contains at least one even digit -/
theorem sum_with_reverse_has_even_digit (n : Digit17) : 
  hasEvenDigit (add n (reverse n)) := by
  sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2192_219274


namespace NUMINAMATH_CALUDE_roots_imply_f_value_l2192_219220

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 75*x + c

-- State the theorem
theorem roots_imply_f_value (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = -2773 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_f_value_l2192_219220


namespace NUMINAMATH_CALUDE_radiator_problem_l2192_219283

/-- Represents the fraction of water remaining after a number of replacements -/
def water_fraction (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / initial_volume) ^ num_replacements

/-- The problem statement -/
theorem radiator_problem (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ)
    (h1 : initial_volume = 20)
    (h2 : replacement_volume = 5)
    (h3 : num_replacements = 4) :
  water_fraction initial_volume replacement_volume num_replacements = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end NUMINAMATH_CALUDE_radiator_problem_l2192_219283


namespace NUMINAMATH_CALUDE_factor_calculation_l2192_219276

theorem factor_calculation (n : ℝ) (f : ℝ) (h1 : n = 155) (h2 : n * f - 200 = 110) : f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2192_219276


namespace NUMINAMATH_CALUDE_strip_to_upper_half_plane_l2192_219212

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the mapping function
noncomputable def w (z : ℂ) (h : ℝ) : ℂ := complex_exp ((Real.pi * z) / h)

-- State the theorem
theorem strip_to_upper_half_plane (z : ℂ) (h : ℝ) (h_pos : h > 0) (z_in_strip : 0 < z.im ∧ z.im < h) :
  (w z h).im > 0 := by sorry

end NUMINAMATH_CALUDE_strip_to_upper_half_plane_l2192_219212


namespace NUMINAMATH_CALUDE_largest_three_digit_arithmetic_sequence_l2192_219285

/-- Checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- Checks if the digits of a three-digit number form an arithmetic sequence -/
def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) - ((n / 10) % 10) = ((n / 10) % 10) - (n % 10)

/-- The main theorem stating that 789 is the largest three-digit number
    with distinct digits forming an arithmetic sequence -/
theorem largest_three_digit_arithmetic_sequence :
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 →
    has_distinct_digits m ∧ digits_form_arithmetic_sequence m →
    m ≤ 789) ∧
  has_distinct_digits 789 ∧ digits_form_arithmetic_sequence 789 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_arithmetic_sequence_l2192_219285


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l2192_219210

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (Real.rpow (17 * x - 1) (1/3) + Real.rpow (11 * x + 1) (1/3) = 2 * Real.rpow x (1/3)) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l2192_219210


namespace NUMINAMATH_CALUDE_draining_cylinder_height_change_rate_l2192_219265

/-- The rate of change of liquid level height in a draining cylindrical container -/
theorem draining_cylinder_height_change_rate 
  (d : ℝ) -- diameter of the base
  (dV_dt : ℝ) -- rate of volume change (negative for draining)
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (h_diff : Differentiable ℝ h) -- h is differentiable
  (cylinder_volume : ∀ t, π * (d/2)^2 * h t = -dV_dt * t + C) -- volume equation
  (h_positive : ∀ t, h t > 0) -- height is always positive
  (dV_dt_negative : dV_dt < 0) -- volume is decreasing
  (d_positive : d > 0) -- diameter is positive
  (h_init : h 0 > 0) -- initial height is positive
  : d = 2 → dV_dt = -0.01 → deriv h t = -0.01 / π := by
  sorry

end NUMINAMATH_CALUDE_draining_cylinder_height_change_rate_l2192_219265


namespace NUMINAMATH_CALUDE_alternating_series_sum_l2192_219221

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then n else -n

def series_sum (n : ℕ) : ℤ := 
  (List.range n).map (λ i => alternating_series (i + 1)) |>.sum

theorem alternating_series_sum : series_sum 11001 = 16501 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l2192_219221


namespace NUMINAMATH_CALUDE_intersected_cells_count_l2192_219203

/-- Represents a grid cell -/
structure Cell where
  x : Int
  y : Int

/-- Represents a grid -/
structure Grid where
  width : Nat
  height : Nat

/-- Checks if a point (x, y) is inside the grid -/
def Grid.contains (g : Grid) (x y : Int) : Prop :=
  -g.width / 2 ≤ x ∧ x < g.width / 2 ∧ -g.height / 2 ≤ y ∧ y < g.height / 2

/-- Counts the number of cells intersected by the line y = mx -/
def countIntersectedCells (g : Grid) (m : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of cells intersected by y = 0.83x on a 60x70 grid is 108 -/
theorem intersected_cells_count :
  let g : Grid := { width := 60, height := 70 }
  let m : ℚ := 83 / 100
  countIntersectedCells g m = 108 := by
  sorry

end NUMINAMATH_CALUDE_intersected_cells_count_l2192_219203


namespace NUMINAMATH_CALUDE_fraction_expression_value_l2192_219270

theorem fraction_expression_value (m n p : ℝ) (h : m + n - p = 0) :
  m * (1 / n - 1 / p) + n * (1 / m - 1 / p) - p * (1 / m + 1 / n) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_value_l2192_219270


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l2192_219282

theorem cos_negative_seventy_nine_pi_sixths : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l2192_219282


namespace NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_achievable_l2192_219228

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 ≥ 7.25 :=
by sorry

theorem minimum_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 = 7.25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_achievable_l2192_219228


namespace NUMINAMATH_CALUDE_mina_driving_problem_l2192_219225

theorem mina_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ) 
  (h : initial_distance = 20 ∧ initial_speed = 40 ∧ second_speed = 60 ∧ target_average_speed = 55) :
  ∃ additional_distance : ℝ,
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_mina_driving_problem_l2192_219225


namespace NUMINAMATH_CALUDE_max_product_sum_l2192_219218

theorem max_product_sum (A M C : ℕ+) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ+, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' + A' + M' + C' ≤
    A * M * C + A * M + M * C + C * A + A + M + C) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l2192_219218


namespace NUMINAMATH_CALUDE_central_park_excess_cans_l2192_219219

def trash_can_problem (central_park : ℕ) (veterans_park : ℕ) : Prop :=
  -- Central Park had some more than half of the number of trash cans as in Veteran's Park
  central_park > veterans_park / 2 ∧
  -- Originally, there were 24 trash cans in Veteran's Park
  veterans_park = 24 ∧
  -- Half of the trash cans from Central Park were moved to Veteran's Park
  -- Now, there are 34 trash cans in Veteran's Park
  central_park / 2 + veterans_park = 34

theorem central_park_excess_cans :
  ∀ central_park veterans_park,
    trash_can_problem central_park veterans_park →
    central_park - veterans_park / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_central_park_excess_cans_l2192_219219


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2192_219229

/-- Given m > 0, n > 0, and the line y = (1/e)x + m + 1 is tangent to the curve y = ln x - n + 2,
    the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_tangent : ∃ x : ℝ, (1 / Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧
                        (1 / Real.exp 1) = 1 / x) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 1 / n₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2192_219229


namespace NUMINAMATH_CALUDE_solve_for_S_l2192_219261

theorem solve_for_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 180 ∧ S = 180 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_S_l2192_219261


namespace NUMINAMATH_CALUDE_intersection_sum_l2192_219299

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) → 
  (6 = (1/3) * 3 + b) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2192_219299


namespace NUMINAMATH_CALUDE_zhaos_estimate_l2192_219204

theorem zhaos_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2*ε) > x - y :=
sorry

end NUMINAMATH_CALUDE_zhaos_estimate_l2192_219204


namespace NUMINAMATH_CALUDE_ed_lost_21_marbles_l2192_219278

-- Define the initial number of marbles for Ed and Doug
def ed_initial (doug_initial : ℕ) : ℕ := doug_initial + 30

-- Define Ed's current number of marbles
def ed_current : ℕ := 91

-- Define Doug's current number of marbles
def doug_current : ℕ := ed_current - 9

-- Define the number of marbles Ed lost
def marbles_lost : ℕ := ed_initial doug_current - ed_current

-- Theorem statement
theorem ed_lost_21_marbles : marbles_lost = 21 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_21_marbles_l2192_219278


namespace NUMINAMATH_CALUDE_f_properties_l2192_219223

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 15

-- Define the theorem
theorem f_properties (a : ℝ) (x : ℝ) (h : |x - a| < 1) :
  (∃ (y : ℝ), |f y| > 5 ↔ (y < -4 ∨ y > 5 ∨ ((1 - Real.sqrt 41) / 2 < y ∧ y < (1 + Real.sqrt 41) / 2))) ∧
  |f x - f a| < 2 * (|a| + 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2192_219223


namespace NUMINAMATH_CALUDE_tank_capacity_comparison_l2192_219245

theorem tank_capacity_comparison :
  let tank_a_height : ℝ := 10
  let tank_a_circumference : ℝ := 7
  let tank_b_height : ℝ := 7
  let tank_b_circumference : ℝ := 10
  let tank_a_volume := π * (tank_a_circumference / (2 * π))^2 * tank_a_height
  let tank_b_volume := π * (tank_b_circumference / (2 * π))^2 * tank_b_height
  (tank_a_volume / tank_b_volume) * 100 = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_comparison_l2192_219245


namespace NUMINAMATH_CALUDE_non_foreign_male_students_l2192_219295

theorem non_foreign_male_students 
  (total_students : ℕ) 
  (female_ratio : ℚ) 
  (foreign_male_ratio : ℚ) :
  total_students = 300 →
  female_ratio = 2/3 →
  foreign_male_ratio = 1/10 →
  (total_students : ℚ) * (1 - female_ratio) * (1 - foreign_male_ratio) = 90 := by
  sorry

end NUMINAMATH_CALUDE_non_foreign_male_students_l2192_219295


namespace NUMINAMATH_CALUDE_right_triangle_area_l2192_219288

theorem right_triangle_area (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (1/2) * a * b = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2192_219288


namespace NUMINAMATH_CALUDE_team_total_score_l2192_219233

def team_score (connor_score amy_score jason_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score

theorem team_total_score : 
  ∀ (connor_score amy_score jason_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    team_score connor_score amy_score jason_score = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l2192_219233


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2192_219202

theorem sufficient_not_necessary_condition (A B : Set α) 
  (h1 : A ∩ B = A) (h2 : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2192_219202


namespace NUMINAMATH_CALUDE_closest_points_on_hyperbola_l2192_219224

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

/-- The distance squared function from a point (x, y) to A(0, -3) -/
def distance_squared (x y : ℝ) : ℝ := x^2 + (y + 3)^2

/-- The point A -/
def A : ℝ × ℝ := (0, -3)

/-- Theorem stating that the given points are the closest to A on the hyperbola -/
theorem closest_points_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    (x₁ = -3 * Real.sqrt 5 / 2 ∧ y₁ = -3 / 2) ∧
    (x₂ = 3 * Real.sqrt 5 / 2 ∧ y₂ = -3 / 2) ∧
    (∀ (x y : ℝ), hyperbola x y → 
      distance_squared x y ≥ distance_squared x₁ y₁ ∧
      distance_squared x y ≥ distance_squared x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_closest_points_on_hyperbola_l2192_219224


namespace NUMINAMATH_CALUDE_simplest_form_sqrt_l2192_219286

/-- A square root is in its simplest form if the number under the root has no perfect square factors other than 1. -/
def is_simplest_form (x : ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → (n^2 : ℝ) ∣ x → False

/-- Given four square roots, prove that √15 is in its simplest form while the others are not. -/
theorem simplest_form_sqrt : 
  is_simplest_form 15 ∧ 
  ¬is_simplest_form 24 ∧ 
  ¬is_simplest_form (7/3) ∧ 
  ¬is_simplest_form 0.9 :=
sorry

end NUMINAMATH_CALUDE_simplest_form_sqrt_l2192_219286


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2192_219209

/-- The quadratic equation (a+1)x^2 - 4x + 1 = 0 has two distinct real roots if and only if a < 3 and a ≠ -1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a + 1) * x^2 - 4 * x + 1 = 0 ∧ (a + 1) * y^2 - 4 * y + 1 = 0) ↔ 
  (a < 3 ∧ a ≠ -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2192_219209


namespace NUMINAMATH_CALUDE_theta_value_l2192_219227

theorem theta_value : ∃! (Θ : ℕ), Θ ∈ Finset.range 10 ∧ (312 : ℚ) / Θ = 40 + 2 * Θ := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l2192_219227


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2192_219235

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2192_219235


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l2192_219269

theorem new_ratio_after_addition (x : ℤ) : 
  (4 * x = 16) →  -- The larger integer is 16
  ((x + 12) / (4 * x) = 1) -- The new ratio is 1 to 1
  := by sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l2192_219269


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2192_219290

/-- Given that x³ + x + m = 7 when x = 1, prove that x³ + x + m = 3 when x = -1. -/
theorem algebraic_expression_value (m : ℝ) 
  (h : 1^3 + 1 + m = 7) : 
  (-1)^3 + (-1) + m = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2192_219290


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2192_219291

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2192_219291


namespace NUMINAMATH_CALUDE_f_is_even_iff_l2192_219249

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = ax^2 + (2a+1)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (2*a + 1) * x - 1

/-- Theorem: The function f is even if and only if a = -1/2 -/
theorem f_is_even_iff (a : ℝ) : IsEven (f a) ↔ a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_f_is_even_iff_l2192_219249


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2192_219258

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2192_219258


namespace NUMINAMATH_CALUDE_sonya_fell_six_times_l2192_219272

/-- The number of times Steven fell while ice skating -/
def steven_falls : ℕ := 3

/-- The number of times Stephanie fell while ice skating -/
def stephanie_falls : ℕ := steven_falls + 13

/-- The number of times Sonya fell while ice skating -/
def sonya_falls : ℕ := stephanie_falls / 2 - 2

/-- Theorem stating that Sonya fell 6 times -/
theorem sonya_fell_six_times : sonya_falls = 6 := by sorry

end NUMINAMATH_CALUDE_sonya_fell_six_times_l2192_219272
