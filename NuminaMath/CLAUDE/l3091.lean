import Mathlib

namespace NUMINAMATH_CALUDE_rohans_savings_l3091_309178

/-- Rohan's monthly savings calculation -/
theorem rohans_savings (salary : ℝ) (food_percent : ℝ) (rent_percent : ℝ) 
  (entertainment_percent : ℝ) (conveyance_percent : ℝ) : 
  salary = 12500 ∧ 
  food_percent = 40 ∧ 
  rent_percent = 20 ∧ 
  entertainment_percent = 10 ∧ 
  conveyance_percent = 10 → 
  salary * (1 - (food_percent + rent_percent + entertainment_percent + conveyance_percent) / 100) = 2500 :=
by sorry

end NUMINAMATH_CALUDE_rohans_savings_l3091_309178


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_slope_l3091_309176

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 6*y + 14 = 0

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  a*x + 4*y - 6 = 0

-- Define symmetry of circle about line
def circle_symmetrical_about_line (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), circle_equation x₀ y₀ ∧ line_equation a x₀ y₀

-- Theorem statement
theorem circle_symmetry_implies_slope :
  ∀ a : ℝ, circle_symmetrical_about_line a → (a = 6) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_slope_l3091_309176


namespace NUMINAMATH_CALUDE_tensor_range_theorem_l3091_309188

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating the range of k given the condition -/
theorem tensor_range_theorem (k : ℝ) :
  (∀ x : ℝ, tensor k x > 0) → k ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_tensor_range_theorem_l3091_309188


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l3091_309125

/-- A geometric sequence with integer common ratio -/
def GeometricSequence (a : ℕ → ℤ) (q : ℤ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h_geom : GeometricSequence a q)
  (h_prod : a 4 * a 7 = -512)
  (h_sum : a 3 + a 8 = 124) :
  a 10 = 512 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l3091_309125


namespace NUMINAMATH_CALUDE_sum_of_complex_exponentials_l3091_309173

/-- The sum of 16 complex exponentials with angles that are multiples of 2π/17 -/
theorem sum_of_complex_exponentials (ω : ℂ) (h : ω = Complex.exp (2 * Real.pi * Complex.I / 17)) :
  (Finset.range 16).sum (fun k => ω ^ (k + 1)) = ω := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_exponentials_l3091_309173


namespace NUMINAMATH_CALUDE_ellipse_tangent_circle_l3091_309101

/-- Ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 2*y^2 = 4

/-- Circle S -/
def circle_S (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Line L -/
def line_L (y : ℝ) : Prop := y = 2

/-- Point A on ellipse C -/
structure Point_A where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Point B on line L -/
structure Point_B where
  x : ℝ
  y : ℝ
  on_line : line_L y

/-- OA perpendicular to OB -/
def perpendicular (A : Point_A) (B : Point_B) : Prop :=
  A.x * B.x + A.y * B.y = 0

/-- Line AB is tangent to circle S -/
def is_tangent (A : Point_A) (B : Point_B) : Prop :=
  ∃ (t : ℝ), circle_S t ((B.y - A.y) / (B.x - A.x) * (t - A.x) + A.y)

theorem ellipse_tangent_circle (A : Point_A) (B : Point_B) 
  (h : perpendicular A B) : is_tangent A B := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_circle_l3091_309101


namespace NUMINAMATH_CALUDE_vector_equality_l3091_309118

theorem vector_equality (m : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1^2 + a.2^2) + (b.1^2 + b.2^2) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_l3091_309118


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l3091_309132

/-- Represents a pentagon with vertices F, G, H, I, J -/
structure Pentagon :=
  (F G H I J : Point)

/-- The area of the pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Condition that the pentagon is constructed from 10 line segments of length 3 -/
def is_valid_pentagon (p : Pentagon) : Prop := sorry

theorem pentagon_area_sum (p : Pentagon) (a b : ℕ) :
  is_valid_pentagon p →
  area p = Real.sqrt a + Real.sqrt b →
  a + b = 29 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l3091_309132


namespace NUMINAMATH_CALUDE_perpendicular_segments_in_cube_l3091_309124

/-- Represents a cube in 3D space -/
structure Cube where
  -- We don't need to define the specifics of a cube for this statement

/-- Represents a line segment in the cube (edge, face diagonal, or space diagonal) -/
structure LineSegment where
  -- We don't need to define the specifics of a line segment for this statement

/-- Checks if a line segment is perpendicular to a given edge of the cube -/
def is_perpendicular (c : Cube) (l : LineSegment) (edge : LineSegment) : Prop :=
  sorry -- Definition not needed for the statement

/-- Counts the number of line segments perpendicular to a given edge -/
def count_perpendicular_segments (c : Cube) (edge : LineSegment) : Nat :=
  sorry -- Definition not needed for the statement

/-- Theorem: The number of line segments perpendicular to any edge in a cube is 12 -/
theorem perpendicular_segments_in_cube (c : Cube) (edge : LineSegment) :
  count_perpendicular_segments c edge = 12 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_segments_in_cube_l3091_309124


namespace NUMINAMATH_CALUDE_unique_intersection_l3091_309106

/-- The value of a for which the graphs of y = ax² + 5x + 2 and y = -2x - 2 intersect at exactly one point -/
def intersection_value : ℚ := 49 / 16

/-- The first graph equation -/
def graph1 (a x : ℚ) : ℚ := a * x^2 + 5 * x + 2

/-- The second graph equation -/
def graph2 (x : ℚ) : ℚ := -2 * x - 2

/-- Theorem stating that the graphs intersect at exactly one point when a = 49/16 -/
theorem unique_intersection :
  ∃! x : ℚ, graph1 intersection_value x = graph2 x :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l3091_309106


namespace NUMINAMATH_CALUDE_intersection_implies_range_140_l3091_309180

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 6)^2 + (y - 3)^2 = 7^2
def circle2 (x y k : ℝ) : Prop := (x - 2)^2 + (y - 6)^2 = k + 40

-- Define the intersection condition
def intersect (k : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y k

-- Theorem statement
theorem intersection_implies_range_140 (a b : ℝ) :
  (∀ k : ℝ, a ≤ k ∧ k ≤ b → intersect k) → b - a = 140 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_range_140_l3091_309180


namespace NUMINAMATH_CALUDE_second_quadrant_angle_sum_l3091_309121

theorem second_quadrant_angle_sum (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (Real.tan (θ + π / 4) = 1 / 2) →  -- tan(θ + π/4) = 1/2
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_angle_sum_l3091_309121


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l3091_309111

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  12 * small_box_count + 16 * large_box_count

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count large_box_count : ℕ),
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 :=
sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l3091_309111


namespace NUMINAMATH_CALUDE_unique_point_perpendicular_segments_l3091_309131

/-- Given a non-zero real number α, there exists a unique point P in the coordinate plane
    such that for every line through P intersecting the parabola y = αx² in two distinct points A and B,
    the segments OA and OB are perpendicular (where O is the origin). -/
theorem unique_point_perpendicular_segments (α : ℝ) (h : α ≠ 0) :
  ∃! P : ℝ × ℝ, ∀ (A B : ℝ × ℝ),
    (A.2 = α * A.1^2) →
    (B.2 = α * B.1^2) →
    (∃ t : ℝ, A.1 + t * (P.1 - A.1) = B.1 ∧ A.2 + t * (P.2 - A.2) = B.2) →
    (A ≠ B) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    P = (0, 1 / α) :=
sorry

end NUMINAMATH_CALUDE_unique_point_perpendicular_segments_l3091_309131


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l3091_309160

theorem nth_equation_pattern (n : ℕ+) : n^2 - n = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l3091_309160


namespace NUMINAMATH_CALUDE_two_digit_number_with_specific_division_properties_l3091_309137

theorem two_digit_number_with_specific_division_properties :
  ∀ n : ℕ,
  (n ≥ 10 ∧ n ≤ 99) →
  (n % 6 = n / 10) →
  (n / 10 = 3 ∧ n % 10 = n % 10) →
  (n = 33 ∨ n = 39) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_with_specific_division_properties_l3091_309137


namespace NUMINAMATH_CALUDE_remainder_2345_times_1976_mod_300_l3091_309112

theorem remainder_2345_times_1976_mod_300 : (2345 * 1976) % 300 = 220 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345_times_1976_mod_300_l3091_309112


namespace NUMINAMATH_CALUDE_binomial_60_2_l3091_309161

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_2_l3091_309161


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3091_309105

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ k, Nat.Prime (p k)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3091_309105


namespace NUMINAMATH_CALUDE_probability_of_28_l3091_309134

/-- Represents a die with a specific face configuration -/
structure Die :=
  (faces : List ℕ)
  (blank_faces : ℕ)

/-- The first die configuration -/
def die1 : Die :=
  { faces := List.range 18, blank_faces := 1 }

/-- The second die configuration -/
def die2 : Die :=
  { faces := (List.range 7) ++ (List.range' 9 20), blank_faces := 1 }

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (d1 d2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

theorem probability_of_28 :
  probability_of_sum die1 die2 28 = 1 / 40 := by sorry

end NUMINAMATH_CALUDE_probability_of_28_l3091_309134


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_one_l3091_309113

theorem sqrt_nine_minus_one : Real.sqrt 9 - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_one_l3091_309113


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l3091_309198

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → 
  s^2 = s * Real.sqrt 2 → 
  4 * s = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l3091_309198


namespace NUMINAMATH_CALUDE_jims_age_fraction_l3091_309196

theorem jims_age_fraction (tom_age_5_years_ago : ℕ) (jim_age_in_2_years : ℕ) : 
  tom_age_5_years_ago = 32 →
  jim_age_in_2_years = 29 →
  ∃ f : ℚ, 
    (jim_age_in_2_years - 9 : ℚ) = f * (tom_age_5_years_ago + 2 : ℚ) + 5 ∧ 
    f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jims_age_fraction_l3091_309196


namespace NUMINAMATH_CALUDE_factoring_left_to_right_l3091_309107

theorem factoring_left_to_right (m n : ℝ) : m^2 - 2*m*n + n^2 = (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factoring_left_to_right_l3091_309107


namespace NUMINAMATH_CALUDE_equal_cost_at_280_minutes_unique_equal_cost_point_l3091_309146

/-- Represents a phone service plan with a monthly fee and per-minute rate. -/
structure ServicePlan where
  monthlyFee : ℝ
  perMinuteRate : ℝ

/-- Calculates the cost of a service plan for a given number of minutes. -/
def planCost (plan : ServicePlan) (minutes : ℝ) : ℝ :=
  plan.monthlyFee + plan.perMinuteRate * minutes

/-- Theorem stating that the costs of two specific phone service plans are equal at 280 minutes. -/
theorem equal_cost_at_280_minutes : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  planCost plan1 280 = planCost plan2 280 := by
  sorry

/-- Theorem stating that 280 minutes is the unique point where the costs are equal. -/
theorem unique_equal_cost_point : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  ∀ x : ℝ, planCost plan1 x = planCost plan2 x ↔ x = 280 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_280_minutes_unique_equal_cost_point_l3091_309146


namespace NUMINAMATH_CALUDE_hamiltonian_cycle_with_at_most_one_color_change_l3091_309119

/-- A complete graph with n vertices where each edge is colored either red or blue -/
structure ColoredCompleteGraph (n : ℕ) where
  vertices : Fin n → Type
  edge_color : ∀ (i j : Fin n), i ≠ j → Bool

/-- A Hamiltonian cycle in the graph -/
def HamiltonianCycle (n : ℕ) (G : ColoredCompleteGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.Nodup }

/-- The number of color changes in a Hamiltonian cycle -/
def ColorChanges (n : ℕ) (G : ColoredCompleteGraph n) (cycle : HamiltonianCycle n G) : ℕ :=
  sorry

/-- Theorem: There exists a Hamiltonian cycle with at most one color change -/
theorem hamiltonian_cycle_with_at_most_one_color_change (n : ℕ) (G : ColoredCompleteGraph n) :
  ∃ (cycle : HamiltonianCycle n G), ColorChanges n G cycle ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_hamiltonian_cycle_with_at_most_one_color_change_l3091_309119


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3091_309158

/-- The distance Arthur walked in miles -/
def distance_walked (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem: Arthur walks 4.5 miles -/
theorem arthur_walk_distance :
  distance_walked 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l3091_309158


namespace NUMINAMATH_CALUDE_total_chips_is_135_l3091_309136

/-- Calculates the total number of chips for Viviana, Susana, and Manuel --/
def total_chips (viviana_vanilla : ℕ) (susana_chocolate : ℕ) : ℕ :=
  let viviana_chocolate := susana_chocolate + 5
  let susana_vanilla := (3 * viviana_vanilla) / 4
  let manuel_vanilla := 2 * susana_vanilla
  let manuel_chocolate := viviana_chocolate / 2
  (viviana_chocolate + viviana_vanilla) + 
  (susana_chocolate + susana_vanilla) + 
  (manuel_chocolate + manuel_vanilla)

/-- Theorem stating the total number of chips is 135 --/
theorem total_chips_is_135 : total_chips 20 25 = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_is_135_l3091_309136


namespace NUMINAMATH_CALUDE_perpendicular_lines_exist_l3091_309130

/-- Two lines l₁ and l₂ in the plane -/
structure Lines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  h₁ : ∀ x y, l₁ (x, y) ↔ x + a * y = 3
  h₂ : ∀ x y, l₂ (x, y) ↔ 3 * x - (a - 2) * y = 2

/-- Perpendicularity condition for two lines -/
def perpendicular (l : Lines) : Prop :=
  1 * 3 + l.a * -(l.a - 2) = 0

/-- Theorem: If the lines are perpendicular, then there exists a real number a satisfying the condition -/
theorem perpendicular_lines_exist (l : Lines) (h : perpendicular l) : 
  ∃ a : ℝ, 1 * 3 + a * -(a - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_exist_l3091_309130


namespace NUMINAMATH_CALUDE_fraction_simplification_l3091_309143

theorem fraction_simplification :
  1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3091_309143


namespace NUMINAMATH_CALUDE_football_sample_size_l3091_309117

/-- Calculates the number of people to be sampled from a group in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (group_size : ℕ) (total_sample : ℕ) : ℕ :=
  (group_size * total_sample) / total_population

/-- Proves that the stratified sample size for the football group is 8 -/
theorem football_sample_size :
  let total_population : ℕ := 120
  let football_size : ℕ := 40
  let basketball_size : ℕ := 60
  let volleyball_size : ℕ := 20
  let total_sample : ℕ := 24
  stratified_sample_size total_population football_size total_sample = 8 := by
  sorry

#eval stratified_sample_size 120 40 24

end NUMINAMATH_CALUDE_football_sample_size_l3091_309117


namespace NUMINAMATH_CALUDE_algebraic_operation_proof_l3091_309144

theorem algebraic_operation_proof (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_operation_proof_l3091_309144


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_l3091_309120

theorem probability_three_tails_one_head : 
  let n : ℕ := 4 -- number of coins
  let k : ℕ := 3 -- number of tails (or heads, whichever is larger)
  let p : ℚ := 1/2 -- probability of getting tails (or heads) for a single coin
  (n.choose k) * p^k * (1 - p)^(n - k) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_l3091_309120


namespace NUMINAMATH_CALUDE_expression_evaluates_to_one_l3091_309165

theorem expression_evaluates_to_one :
  (100^2 - 7^2) / (70^2 - 11^2) * ((70 - 11) * (70 + 11)) / ((100 - 7) * (100 + 7)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluates_to_one_l3091_309165


namespace NUMINAMATH_CALUDE_apartment_ratio_l3091_309129

theorem apartment_ratio (total_floors : ℕ) (max_residents : ℕ) 
  (h1 : total_floors = 12)
  (h2 : max_residents = 264) :
  ∃ (floors_with_6 : ℕ) (floors_with_5 : ℕ),
    floors_with_6 + floors_with_5 = total_floors ∧
    6 * floors_with_6 + 5 * floors_with_5 = max_residents / 4 ∧
    floors_with_6 * 2 = total_floors := by
  sorry

end NUMINAMATH_CALUDE_apartment_ratio_l3091_309129


namespace NUMINAMATH_CALUDE_water_left_in_bathtub_water_left_is_7800_l3091_309110

/-- Calculates the amount of water left in a bathtub given specific conditions. -/
theorem water_left_in_bathtub 
  (faucet_drip_rate : ℝ) 
  (evaporation_rate : ℝ) 
  (time_running : ℝ) 
  (water_dumped : ℝ) : ℝ :=
  let water_added_per_hour := faucet_drip_rate * 60 - evaporation_rate
  let total_water_added := water_added_per_hour * time_running
  let water_remaining := total_water_added - water_dumped * 1000
  water_remaining

/-- Proves that under the given conditions, 7800 ml of water are left in the bathtub. -/
theorem water_left_is_7800 :
  water_left_in_bathtub 40 200 9 12 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_water_left_in_bathtub_water_left_is_7800_l3091_309110


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3091_309139

theorem sin_thirteen_pi_fourths : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3091_309139


namespace NUMINAMATH_CALUDE_change_in_responses_l3091_309103

/-- Represents the percentage of students giving each response --/
structure ResponsePercentages where
  yes : ℝ
  no : ℝ
  undecided : ℝ

/-- The problem statement --/
theorem change_in_responses
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 30)
  (h_initial_undecided : initial.undecided = 30)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 20)
  (h_final_undecided : final.undecided = 20) :
  ∃ (min_change max_change : ℝ),
    (0 ≤ min_change ∧ min_change ≤ 100) ∧
    (0 ≤ max_change ∧ max_change ≤ 100) ∧
    (max_change - min_change = 30) :=
sorry

end NUMINAMATH_CALUDE_change_in_responses_l3091_309103


namespace NUMINAMATH_CALUDE_cats_in_sacks_l3091_309164

theorem cats_in_sacks (cat_prices sack_prices : Finset ℕ) : 
  cat_prices.card = 20 →
  sack_prices.card = 20 →
  (∀ p ∈ cat_prices, 1200 ≤ p ∧ p ≤ 1500) →
  (∀ p ∈ sack_prices, 10 ≤ p ∧ p ≤ 100) →
  cat_prices.toList.Nodup →
  sack_prices.toList.Nodup →
  ∃ (c1 c2 : ℕ) (s1 s2 : ℕ),
    c1 ∈ cat_prices ∧ 
    c2 ∈ cat_prices ∧ 
    s1 ∈ sack_prices ∧ 
    s2 ∈ sack_prices ∧
    c1 ≠ c2 ∧ 
    s1 ≠ s2 ∧ 
    c1 + s1 = c2 + s2 :=
by sorry

end NUMINAMATH_CALUDE_cats_in_sacks_l3091_309164


namespace NUMINAMATH_CALUDE_least_marbles_thirty_two_satisfies_george_marbles_l3091_309154

theorem least_marbles (n : ℕ) : 
  (n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3) → n ≥ 32 :=
by sorry

theorem thirty_two_satisfies : 
  32 % 7 = 1 ∧ 32 % 4 = 2 ∧ 32 % 6 = 3 :=
by sorry

theorem george_marbles : 
  ∃ (n : ℕ), n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3 ∧ 
  ∀ (m : ℕ), (m % 7 = 1 ∧ m % 4 = 2 ∧ m % 6 = 3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_thirty_two_satisfies_george_marbles_l3091_309154


namespace NUMINAMATH_CALUDE_donuts_per_box_l3091_309191

/-- Proves that the number of donuts per box is 10 given the conditions of Jeff's donut-making and eating scenario. -/
theorem donuts_per_box :
  let total_donuts := 10 * 12
  let jeff_eaten := 1 * 12
  let chris_eaten := 8
  let boxes := 10
  let remaining_donuts := total_donuts - jeff_eaten - chris_eaten
  remaining_donuts / boxes = 10 := by
  sorry

end NUMINAMATH_CALUDE_donuts_per_box_l3091_309191


namespace NUMINAMATH_CALUDE_larger_number_problem_l3091_309150

theorem larger_number_problem (x y : ℕ) : 
  x * y = 30 → x + y = 13 → max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3091_309150


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3091_309187

/-- Proves that a hyperbola with given conditions has the equation x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3 / 3) →
  (Real.sqrt 3 * a / 3 = 1) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3091_309187


namespace NUMINAMATH_CALUDE_middle_integer_is_six_l3091_309123

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧
  b = a + 2 ∧ c = b + 2 ∧
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_six :
  ∀ a b c : ℕ, is_valid_triple a b c → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_six_l3091_309123


namespace NUMINAMATH_CALUDE_brand_A_soap_users_l3091_309152

theorem brand_A_soap_users (total : ℕ) (neither : ℕ) (both : ℕ) (ratio : ℕ) : 
  total = 300 →
  neither = 80 →
  both = 40 →
  ratio = 3 →
  total - neither - (ratio * both) - both = 60 :=
by sorry

end NUMINAMATH_CALUDE_brand_A_soap_users_l3091_309152


namespace NUMINAMATH_CALUDE_unique_solution_l3091_309122

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 6

def is_valid_row (a b c d e : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def sum_constraint (a b c d e : ℕ) : Prop :=
  100 * a + 10 * b + c + 10 * c + d + e = 696

theorem unique_solution (a b c d e : ℕ) :
  is_valid_row a b c d e ∧ sum_constraint a b c d e →
  a = 6 ∧ b = 2 ∧ c = 3 ∧ d = 6 ∧ e = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3091_309122


namespace NUMINAMATH_CALUDE_first_digit_must_be_odd_l3091_309128

/-- Represents a permutation of digits 0 to 9 -/
def Permutation := Fin 10 → Fin 10

/-- Checks if a permutation contains each digit exactly once -/
def is_valid_permutation (p : Permutation) : Prop :=
  ∀ i j : Fin 10, p i = p j → i = j

/-- Calculates the sum A as described in the problem -/
def sum_A (p : Permutation) : ℕ :=
  (10 * p 0 + p 1) + (10 * p 2 + p 3) + (10 * p 4 + p 5) + (10 * p 6 + p 7) + (10 * p 8 + p 9)

/-- Calculates the sum B as described in the problem -/
def sum_B (p : Permutation) : ℕ :=
  (10 * p 1 + p 2) + (10 * p 3 + p 4) + (10 * p 5 + p 6) + (10 * p 7 + p 8)

theorem first_digit_must_be_odd (p : Permutation) 
  (h_valid : is_valid_permutation p) 
  (h_equal : sum_A p = sum_B p) : 
  ¬ Even (p 0) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_must_be_odd_l3091_309128


namespace NUMINAMATH_CALUDE_base_digit_conversion_l3091_309135

theorem base_digit_conversion (N : ℕ+) :
  (9^19 ≤ N ∧ N < 9^20) ∧ (27^12 ≤ N ∧ N < 27^13) →
  3^38 ≤ N ∧ N < 3^39 :=
by sorry

end NUMINAMATH_CALUDE_base_digit_conversion_l3091_309135


namespace NUMINAMATH_CALUDE_logical_equivalences_l3091_309151

theorem logical_equivalences (x y : ℝ) : True := by
  have original : x + y = 5 → x = 3 ∧ y = 2 := sorry
  
  -- Converse
  have converse : x = 3 ∧ y = 2 → x + y = 5 := sorry
  
  -- Inverse
  have inverse : x + y ≠ 5 → x ≠ 3 ∨ y ≠ 2 := sorry
  
  -- Contrapositive
  have contrapositive : x ≠ 3 ∨ y ≠ 2 → x + y ≠ 5 := sorry
  
  -- Truth values
  have converse_true : ∀ x y, (x = 3 ∧ y = 2 → x + y = 5) := sorry
  have inverse_true : ∀ x y, (x + y ≠ 5 → x ≠ 3 ∨ y ≠ 2) := sorry
  have contrapositive_false : ¬(∀ x y, (x ≠ 3 ∨ y ≠ 2 → x + y ≠ 5)) := sorry
  
  sorry

#check logical_equivalences

end NUMINAMATH_CALUDE_logical_equivalences_l3091_309151


namespace NUMINAMATH_CALUDE_ice_cream_truck_expenses_l3091_309193

/-- Proves that for an ice cream truck business where each cone costs $5, 
    if 200 cones are sold and a $200 profit is made, 
    then the expenses are 80% of the total sales. -/
theorem ice_cream_truck_expenses (cone_price : ℝ) (cones_sold : ℕ) (profit : ℝ) :
  cone_price = 5 →
  cones_sold = 200 →
  profit = 200 →
  let total_sales := cone_price * cones_sold
  let expenses := total_sales - profit
  expenses / total_sales = 0.8 := by sorry

end NUMINAMATH_CALUDE_ice_cream_truck_expenses_l3091_309193


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l3091_309162

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  (Real.sin α * Real.sin β * Real.sin γ ≤ 3 * Real.sqrt 3 / 8) ∧
  (Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) ≤ 3 * Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l3091_309162


namespace NUMINAMATH_CALUDE_point_distance_ratio_l3091_309148

theorem point_distance_ratio (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  (P.1)^2 + (P.2)^2 = 10^2 → 
  (abs P.2) / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_point_distance_ratio_l3091_309148


namespace NUMINAMATH_CALUDE_milk_price_increase_day_l3091_309174

/-- The day in June when the milk price increased -/
def price_increase_day : ℕ := 19

/-- The cost of milk before the price increase -/
def initial_price : ℕ := 1500

/-- The cost of milk after the price increase -/
def new_price : ℕ := 1600

/-- The total amount spent on milk in June -/
def total_spent : ℕ := 46200

/-- The number of days in June -/
def days_in_june : ℕ := 30

theorem milk_price_increase_day :
  (price_increase_day - 1) * initial_price +
  (days_in_june - (price_increase_day - 1)) * new_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_milk_price_increase_day_l3091_309174


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l3091_309153

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l3091_309153


namespace NUMINAMATH_CALUDE_show_revenue_l3091_309145

def tickets_first_showing : ℕ := 200
def ticket_price : ℕ := 25

theorem show_revenue : 
  (tickets_first_showing + 3 * tickets_first_showing) * ticket_price = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_l3091_309145


namespace NUMINAMATH_CALUDE_constant_order_magnitude_l3091_309114

theorem constant_order_magnitude (k : ℕ) (h : k > 4) :
  k + 2 < 2 * k ∧ 2 * k < k^2 ∧ k^2 < 2^k := by
  sorry

end NUMINAMATH_CALUDE_constant_order_magnitude_l3091_309114


namespace NUMINAMATH_CALUDE_share_division_l3091_309116

/-- Given a total sum to be divided among three people A, B, and C, where
    3 times A's share equals 4 times B's share equals 7 times C's share,
    prove that C's share is 84 when the total sum is 427. -/
theorem share_division (total : ℕ) (a b c : ℚ)
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_prop : 3 * a = 4 * b ∧ 4 * b = 7 * c) :
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l3091_309116


namespace NUMINAMATH_CALUDE_edge_coloring_without_monochromatic_clique_l3091_309189

open Nat Finset

/-- A type representing the two colors used for edge coloring -/
inductive Color
| Yellow
| Violet

/-- Definition of a complete graph with n vertices -/
def CompleteGraph (n : ℕ) := {e : Finset (Fin n × Fin n) | ∀ (i j : Fin n), i ≠ j → (i, j) ∈ e}

/-- Definition of a k-clique in a graph -/
def Clique (k : ℕ) (n : ℕ) (S : Finset (Fin n)) := 
  S.card = k ∧ ∀ (i j : Fin n), i ∈ S → j ∈ S → i ≠ j

/-- Definition of a coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := (Fin n × Fin n) → Color

/-- Definition of a monochromatic k-clique under a given coloring -/
def MonochromaticClique (k : ℕ) (n : ℕ) (c : Coloring n) (S : Finset (Fin n)) :=
  Clique k n S ∧ ∃ (col : Color), ∀ (i j : Fin n), i ∈ S → j ∈ S → i ≠ j → c (i, j) = col

/-- The main theorem stating that for k ≥ 3 and n > 2^(k/2), there exists a 2-coloring 
    of the edges of the complete graph K_n such that no monochromatic k-clique exists -/
theorem edge_coloring_without_monochromatic_clique 
  (k : ℕ) (n : ℕ) (h1 : k ≥ 3) (h2 : n > 2^(k/2)) :
  ∃ (c : Coloring n), ¬∃ (S : Finset (Fin n)), MonochromaticClique k n c S := by
  sorry

end NUMINAMATH_CALUDE_edge_coloring_without_monochromatic_clique_l3091_309189


namespace NUMINAMATH_CALUDE_abraham_budget_l3091_309190

/-- Abraham's shopping budget problem -/
theorem abraham_budget : 
  ∀ (shower_gel_price shower_gel_quantity toothpaste_price laundry_detergent_price remaining_budget : ℕ),
    shower_gel_price = 4 →
    shower_gel_quantity = 4 →
    toothpaste_price = 3 →
    laundry_detergent_price = 11 →
    remaining_budget = 30 →
    shower_gel_price * shower_gel_quantity + toothpaste_price + laundry_detergent_price + remaining_budget = 60 := by
  sorry

end NUMINAMATH_CALUDE_abraham_budget_l3091_309190


namespace NUMINAMATH_CALUDE_magazine_sale_gain_l3091_309179

/-- Calculates the total gain from selling magazines -/
def total_gain (cost_price selling_price : ℝ) (num_magazines : ℕ) : ℝ :=
  (selling_price - cost_price) * num_magazines

/-- Proves that the total gain from selling 10 magazines at $3.50 each, 
    bought at $3 each, is $5 -/
theorem magazine_sale_gain : 
  total_gain 3 3.5 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_magazine_sale_gain_l3091_309179


namespace NUMINAMATH_CALUDE_negative_slope_decreasing_l3091_309186

/-- A linear function with negative slope -/
structure NegativeSlopeLinearFunction where
  k : ℝ
  b : ℝ
  h : k < 0

/-- The function corresponding to a NegativeSlopeLinearFunction -/
def NegativeSlopeLinearFunction.toFun (f : NegativeSlopeLinearFunction) : ℝ → ℝ := 
  fun x ↦ f.k * x + f.b

theorem negative_slope_decreasing (f : NegativeSlopeLinearFunction) 
    (x₁ x₂ : ℝ) (h : x₁ < x₂) : 
    f.toFun x₁ > f.toFun x₂ := by
  sorry

end NUMINAMATH_CALUDE_negative_slope_decreasing_l3091_309186


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l3091_309108

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  price_after_coupon * (1 + tax)

theorem jacket_price_calculation :
  calculate_final_price 150 0.30 0.10 10 0.05 = 88.725 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l3091_309108


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_and_product_l3091_309141

theorem unique_sum_of_squares_and_product (a b : ℕ+) : 
  a ≤ b → 
  a.val^2 + b.val^2 + 8 * a.val * b.val = 2010 → 
  a.val + b.val = 42 :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_and_product_l3091_309141


namespace NUMINAMATH_CALUDE_relay_race_last_year_distance_l3091_309172

/-- Represents the relay race setup and calculations -/
def RelayRace (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ) : Prop :=
  let segment_length := distance_between_1_and_3 / 2
  let total_segments := tables - 1
  let this_year_distance := segment_length * total_segments
  let last_year_distance := this_year_distance / multiplier
  (tables = 6) ∧
  (distance_between_1_and_3 = 400) ∧
  (multiplier = 4) ∧
  (last_year_distance = 250)

/-- Theorem stating that given the conditions, the race distance last year was 250 meters -/
theorem relay_race_last_year_distance :
  ∀ (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ),
  RelayRace tables distance_between_1_and_3 multiplier :=
by
  sorry

end NUMINAMATH_CALUDE_relay_race_last_year_distance_l3091_309172


namespace NUMINAMATH_CALUDE_max_plumber_earnings_l3091_309197

def toilet_rate : ℕ := 50
def shower_rate : ℕ := 40
def sink_rate : ℕ := 30

def job1_earnings : ℕ := 3 * toilet_rate + 3 * sink_rate
def job2_earnings : ℕ := 2 * toilet_rate + 5 * sink_rate
def job3_earnings : ℕ := 1 * toilet_rate + 2 * shower_rate + 3 * sink_rate

theorem max_plumber_earnings :
  max job1_earnings (max job2_earnings job3_earnings) = 250 := by
  sorry

end NUMINAMATH_CALUDE_max_plumber_earnings_l3091_309197


namespace NUMINAMATH_CALUDE_initial_money_equals_spent_plus_left_l3091_309104

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha spent -/
def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost

/-- The theorem stating that the initial amount of money Trisha had
    is equal to the sum of all her expenses plus the amount left after shopping -/
theorem initial_money_equals_spent_plus_left :
  total_spent + money_left = 167 := by sorry

end NUMINAMATH_CALUDE_initial_money_equals_spent_plus_left_l3091_309104


namespace NUMINAMATH_CALUDE_f_2015_value_l3091_309147

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_shift (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x + f 2

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_shift f)
  (h_f1 : f 1 = 2) :
  f 2015 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_2015_value_l3091_309147


namespace NUMINAMATH_CALUDE_least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l3091_309140

theorem least_number_with_remainder (n : ℕ) : n ≥ 40 → n % 6 = 4 → ∃ k : ℕ, n = 6 * k + 4 :=
sorry

theorem forty_is_least : ∀ n : ℕ, n < 40 → n % 6 ≠ 4 :=
sorry

theorem forty_has_remainder_four : 40 % 6 = 4 :=
sorry

theorem least_number_is_forty : 
  (∃ n : ℕ, n % 6 = 4) ∧ 
  (∀ n : ℕ, n % 6 = 4 → n ≥ 40) ∧
  (40 % 6 = 4) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l3091_309140


namespace NUMINAMATH_CALUDE_a_6_equals_448_l3091_309166

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n * 2^(n+1)

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- The 6th term of the sequence equals 448 -/
theorem a_6_equals_448 : a 6 = 448 := by sorry

end NUMINAMATH_CALUDE_a_6_equals_448_l3091_309166


namespace NUMINAMATH_CALUDE_expand_product_l3091_309170

theorem expand_product (x : ℝ) : (4 * x + 2) * (3 * x - 1) * (x + 6) = 12 * x^3 + 74 * x^2 + 10 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3091_309170


namespace NUMINAMATH_CALUDE_min_odd_integers_l3091_309194

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 65) :
  ∃ (x : Finset ℤ), x ⊆ {a, b, c, d, e, f} ∧ x.card = 1 ∧ ∀ i ∈ x, Odd i ∧
  ∀ (y : Finset ℤ), y ⊆ {a, b, c, d, e, f} ∧ (∀ i ∈ y, Odd i) → y.card ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l3091_309194


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3091_309102

theorem chess_tournament_games (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (17 * 16 * n) / 2 = 272) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3091_309102


namespace NUMINAMATH_CALUDE_complex_roots_problem_l3091_309169

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 2 ∧ 
  p * q * r = 2 ∧ 
  p * q + p * r + q * r = 0 → 
  (p = 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_problem_l3091_309169


namespace NUMINAMATH_CALUDE_tiling_cost_theorem_l3091_309142

/-- Calculates the total cost of tiling a wall -/
def total_tiling_cost (wall_width wall_height tile_length tile_width tile_cost : ℕ) : ℕ :=
  let wall_area := wall_width * wall_height
  let tile_area := tile_length * tile_width
  let num_tiles := (wall_area + tile_area - 1) / tile_area  -- Ceiling division
  num_tiles * tile_cost

/-- Theorem: The total cost of tiling the given wall is 540,000 won -/
theorem tiling_cost_theorem : 
  total_tiling_cost 36 72 3 4 2500 = 540000 := by
  sorry

end NUMINAMATH_CALUDE_tiling_cost_theorem_l3091_309142


namespace NUMINAMATH_CALUDE_find_g_of_x_l3091_309182

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) 
  (h : ∀ x, 4 * x^4 + 2 * x^2 - x + 7 + g x = x^3 - 4 * x^2 + 6) : 
  g = λ x => -4 * x^4 + x^3 - 6 * x^2 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_find_g_of_x_l3091_309182


namespace NUMINAMATH_CALUDE_cream_cheese_price_l3091_309195

-- Define variables for bagel and cream cheese prices
variable (B : ℝ) -- Price of one bag of bagels
variable (C : ℝ) -- Price of one package of cream cheese

-- Define the equations from the problem
def monday_equation : Prop := 2 * B + 3 * C = 12
def friday_equation : Prop := 4 * B + 2 * C = 14

-- Theorem statement
theorem cream_cheese_price 
  (h1 : monday_equation B C) 
  (h2 : friday_equation B C) : 
  C = 2.5 := by sorry

end NUMINAMATH_CALUDE_cream_cheese_price_l3091_309195


namespace NUMINAMATH_CALUDE_coat_price_reduction_l3091_309185

/-- Given a coat with an original price and a reduction amount, calculate the percent reduction. -/
theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 200) :
  (reduction_amount / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l3091_309185


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3091_309171

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
If the points (4, -6), (3b - 1, 5), and (b + 4, 4) are collinear, then b = 50/19.
-/
theorem collinear_points_b_value :
  ∀ b : ℝ, collinear 4 (-6) (3*b - 1) 5 (b + 4) 4 → b = 50/19 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3091_309171


namespace NUMINAMATH_CALUDE_monthly_income_A_l3091_309192

/-- Given the average monthly incomes of pairs of individuals, 
    prove that the monthly income of A is 4000. -/
theorem monthly_income_A (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 5050)
  (avg_bc : (b + c) / 2 = 6250)
  (avg_ac : (a + c) / 2 = 5200) :
  a = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_A_l3091_309192


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3091_309109

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  7 * x + 2 * (x^2 - 2) - 4 * (1/2 * x^2 - x + 3) = 11 * x - 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3091_309109


namespace NUMINAMATH_CALUDE_no_real_solutions_l3091_309167

theorem no_real_solutions : ¬∃ (x : ℝ), 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3091_309167


namespace NUMINAMATH_CALUDE_complex_solutions_count_l3091_309159

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 8) / (z^2 - 3*z + 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 8) / (z^2 - 3*z + 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l3091_309159


namespace NUMINAMATH_CALUDE_chicken_count_l3091_309177

theorem chicken_count (total_chickens : ℕ) (hens : ℕ) (roosters : ℕ) (chicks : ℕ) : 
  total_chickens = 15 → 
  hens = 3 → 
  roosters = total_chickens - hens → 
  chicks = roosters - 4 → 
  chicks = 8 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l3091_309177


namespace NUMINAMATH_CALUDE_unique_number_with_nine_divisors_and_special_property_l3091_309133

def has_exactly_nine_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 9

theorem unique_number_with_nine_divisors_and_special_property :
  ∃! n : ℕ, has_exactly_nine_divisors n ∧
  ∃ (a b c : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧
  a + b + c = 79 ∧ a * a = b * c :=
by
  use 441
  sorry

end NUMINAMATH_CALUDE_unique_number_with_nine_divisors_and_special_property_l3091_309133


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3091_309175

theorem abs_sum_inequality (x : ℝ) : 
  |x - 3| + |x + 4| < 10 ↔ x ∈ Set.Ioo (-5.5) 4.5 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3091_309175


namespace NUMINAMATH_CALUDE_original_price_after_percentage_changes_l3091_309163

theorem original_price_after_percentage_changes (p : ℝ) :
  let initial_price := (10000 : ℝ) / (10000 - p^2)
  let price_after_increase := initial_price * (1 + p / 100)
  let final_price := price_after_increase * (1 - p / 100)
  final_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_original_price_after_percentage_changes_l3091_309163


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3091_309183

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: f is defined on {x ∈ ℝ | x ≠ 0}
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) ∧
  -- Condition 2: f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- Condition 3: f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- Condition 4: For any non-zero real numbers x and y, f(xy) = f(x) + f(y)
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → f (x * y) = f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3091_309183


namespace NUMINAMATH_CALUDE_divisibility_problem_l3091_309138

theorem divisibility_problem (n : ℕ) (h : ∀ a : ℕ, a < 60 → ¬(n ∣ a^3)) : n = 216000 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3091_309138


namespace NUMINAMATH_CALUDE_ten_ways_to_distribute_albums_l3091_309156

/-- Represents the number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 10 ways to distribute 4 albums to 4 friends -/
theorem ten_ways_to_distribute_albums :
  distribute_albums 2 3 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_ways_to_distribute_albums_l3091_309156


namespace NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l3091_309155

theorem prime_divisor_of_fermat_number (p k : ℕ) : 
  Prime p → p ∣ (2^(2^k) + 1) → (2^(k+1) ∣ (p - 1)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l3091_309155


namespace NUMINAMATH_CALUDE_squared_binomial_subtraction_difference_of_squares_l3091_309199

-- Problem 1
theorem squared_binomial_subtraction (a b : ℝ) :
  a^2 * b - (-2 * a * b^2)^2 = a^2 * b - 4 * a^2 * b^4 := by sorry

-- Problem 2
theorem difference_of_squares (x y : ℝ) :
  (3 * x - 2 * y) * (3 * x + 2 * y) = 9 * x^2 - 4 * y^2 := by sorry

end NUMINAMATH_CALUDE_squared_binomial_subtraction_difference_of_squares_l3091_309199


namespace NUMINAMATH_CALUDE_sector_perimeter_l3091_309127

theorem sector_perimeter (θ : Real) (r : Real) (h1 : θ = 54) (h2 : r = 20) : 
  let α := θ * (π / 180)
  let arc_length := α * r
  let perimeter := arc_length + 2 * r
  perimeter = 6 * π + 40 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3091_309127


namespace NUMINAMATH_CALUDE_solve_equation_l3091_309181

theorem solve_equation : ∃ x : ℝ, 
  ((0.66^3 - 0.1^3) / 0.66^2) + x + 0.1^2 = 0.5599999999999999 ∧ 
  x = -0.107504 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3091_309181


namespace NUMINAMATH_CALUDE_duty_schedules_count_l3091_309168

/-- Represents the number of people on duty -/
def num_people : ℕ := 3

/-- Represents the number of days in the duty schedule -/
def num_days : ℕ := 6

/-- Represents the number of duty days per person -/
def duty_days_per_person : ℕ := 2

/-- Calculates the number of valid duty schedules -/
def count_duty_schedules : ℕ :=
  let total_arrangements := (num_days.choose duty_days_per_person) * ((num_days - duty_days_per_person).choose duty_days_per_person)
  let invalid_arrangements := 2 * ((num_days - 1).choose duty_days_per_person) * ((num_days - duty_days_per_person - 1).choose duty_days_per_person)
  let double_counted := ((num_days - 2).choose duty_days_per_person) * ((num_days - duty_days_per_person - 2).choose duty_days_per_person)
  total_arrangements - invalid_arrangements + double_counted

theorem duty_schedules_count :
  count_duty_schedules = 42 :=
sorry

end NUMINAMATH_CALUDE_duty_schedules_count_l3091_309168


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3091_309100

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →  -- Equation has real roots
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →         -- x₁ is a root
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →         -- x₂ is a root
  ((x₁ + 1) * (x₂ + 1) = 3) →               -- Given condition
  (m = -3) :=                               -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3091_309100


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3091_309126

theorem complex_equation_solution (x : ℂ) : x / Complex.I = 1 - Complex.I → x = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3091_309126


namespace NUMINAMATH_CALUDE_solution_existence_l3091_309157

theorem solution_existence (m : ℝ) : 
  (∃ x : ℝ, 3 * Real.sin x + 4 * Real.cos x = 2 * m - 1) ↔ 
  -2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_l3091_309157


namespace NUMINAMATH_CALUDE_diamond_two_five_l3091_309149

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a + 3 * b ^ 2 + b

-- Theorem statement
theorem diamond_two_five : diamond 2 5 = 82 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_five_l3091_309149


namespace NUMINAMATH_CALUDE_square_side_length_average_l3091_309184

theorem square_side_length_average (a b c : Real) (ha : a = 25) (hb : b = 64) (hc : c = 121) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3091_309184


namespace NUMINAMATH_CALUDE_xiao_hong_books_l3091_309115

/-- Given that Xiao Hong originally had 5 books and bought 'a' more books,
    prove that her total number of books now is 5 + a. -/
theorem xiao_hong_books (a : ℕ) : 5 + a = 5 + a := by sorry

end NUMINAMATH_CALUDE_xiao_hong_books_l3091_309115
