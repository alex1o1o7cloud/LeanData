import Mathlib

namespace max_surface_area_of_stacked_solids_l87_8755

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the surface area of two stacked rectangular solids -/
def stacked_surface_area (d : Dimensions) (overlap_dim1 overlap_dim2 : ℝ) : ℝ :=
  2 * (overlap_dim1 * overlap_dim2) + 
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the given rectangular solids -/
def solid_dimensions : Dimensions :=
  { length := 5, width := 4, height := 3 }

theorem max_surface_area_of_stacked_solids :
  let d := solid_dimensions
  let sa1 := stacked_surface_area d d.length d.width
  let sa2 := stacked_surface_area d d.length d.height
  let sa3 := stacked_surface_area d d.width d.height
  max sa1 (max sa2 sa3) = 164 := by sorry

end max_surface_area_of_stacked_solids_l87_8755


namespace apple_box_weight_l87_8774

theorem apple_box_weight : 
  ∀ (x : ℝ), 
  (x > 0) →  -- Ensure positive weight
  (3 * x - 3 * 4 = x) → 
  x = 6 := by
  sorry

end apple_box_weight_l87_8774


namespace probability_even_sum_two_wheels_l87_8780

/-- Represents a wheel with sections labeled as even or odd numbers -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ
  odd_sections : ℕ
  sections_sum : even_sections + odd_sections = total_sections

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def probability_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p_even1 := wheel1.even_sections / wheel1.total_sections
  let p_odd1 := wheel1.odd_sections / wheel1.total_sections
  let p_even2 := wheel2.even_sections / wheel2.total_sections
  let p_odd2 := wheel2.odd_sections / wheel2.total_sections
  (p_even1 * p_even2) + (p_odd1 * p_odd2)

theorem probability_even_sum_two_wheels :
  let wheel1 : Wheel := ⟨3, 2, 1, by simp⟩
  let wheel2 : Wheel := ⟨5, 3, 2, by simp⟩
  probability_even_sum wheel1 wheel2 = 8/15 := by
  sorry

end probability_even_sum_two_wheels_l87_8780


namespace blouse_price_proof_l87_8768

/-- The original price of a blouse before discount -/
def original_price : ℝ := 180

/-- The discount percentage applied to the blouse -/
def discount_percentage : ℝ := 18

/-- The price paid after applying the discount -/
def discounted_price : ℝ := 147.60

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem blouse_price_proof : 
  original_price * (1 - discount_percentage / 100) = discounted_price := by
  sorry

end blouse_price_proof_l87_8768


namespace square_area_increase_l87_8726

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.1 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end square_area_increase_l87_8726


namespace quadratic_equation_solutions_l87_8718

theorem quadratic_equation_solutions (a : ℝ) : a^2 + 10 = a + 10^2 ↔ a = 10 ∨ a = -9 :=
sorry

end quadratic_equation_solutions_l87_8718


namespace smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l87_8779

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 3 = 11 * k) → n ≥ 9 :=
by sorry

theorem n_equals_nine : 
  ∃ k : ℕ, 15 * 9 - 3 = 11 * k :=
by sorry

theorem smallest_n_is_nine : 
  (∀ m : ℕ, m < 9 → ¬∃ k : ℕ, 15 * m - 3 = 11 * k) ∧
  (∃ k : ℕ, 15 * 9 - 3 = 11 * k) :=
by sorry

end smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l87_8779


namespace survey_result_l87_8737

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1800)
  (h_tv_dislike : tv_dislike_percent = 40 / 100)
  (h_both_dislike : both_dislike_percent = 25 / 100) :
  ↑⌊tv_dislike_percent * both_dislike_percent * total⌋ = 180 :=
by sorry

end survey_result_l87_8737


namespace problem_solution_l87_8772

theorem problem_solution : (12 : ℝ)^2 * 6^3 / 432 = 72 := by
  sorry

end problem_solution_l87_8772


namespace village_population_l87_8765

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 45000 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 50000 := by
  sorry

end village_population_l87_8765


namespace soap_packages_per_box_l87_8784

theorem soap_packages_per_box (soaps_per_package : ℕ) (num_boxes : ℕ) (total_soaps : ℕ) :
  soaps_per_package = 192 →
  num_boxes = 2 →
  total_soaps = 2304 →
  ∃ (packages_per_box : ℕ), 
    packages_per_box * soaps_per_package * num_boxes = total_soaps ∧
    packages_per_box = 6 :=
by sorry

end soap_packages_per_box_l87_8784


namespace trapezoid_height_l87_8762

-- Define the trapezoid properties
structure IsoscelesTrapezoid where
  diagonal : ℝ
  area : ℝ

-- Define the theorem
theorem trapezoid_height (t : IsoscelesTrapezoid) (h_diagonal : t.diagonal = 10) (h_area : t.area = 48) :
  ∃ (height : ℝ), (height = 6 ∨ height = 8) ∧ 
  (∃ (base_avg : ℝ), base_avg * height = t.area ∧ base_avg^2 + height^2 = t.diagonal^2) :=
sorry

end trapezoid_height_l87_8762


namespace program_flowchart_unique_start_end_l87_8708

/-- Represents a chart with start and end points -/
structure Chart where
  start_points : ℕ
  end_points : ℕ

/-- Definition of a general flowchart -/
def is_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Definition of a program flowchart -/
def is_program_flowchart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points = 1

/-- Definition of a structure chart (assumed equivalent to process chart) -/
def is_structure_chart (c : Chart) : Prop :=
  c.start_points = 1 ∧ c.end_points ≥ 1

/-- Theorem stating that a program flowchart has exactly one start point and one end point -/
theorem program_flowchart_unique_start_end :
  ∀ c : Chart, is_program_flowchart c → c.start_points = 1 ∧ c.end_points = 1 := by
  sorry


end program_flowchart_unique_start_end_l87_8708


namespace union_complement_problem_l87_8778

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end union_complement_problem_l87_8778


namespace remaining_payment_l87_8709

/-- Given a 10% deposit of $80, prove that the remaining amount to be paid is $720 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 80 →
  deposit_percentage = 0.1 →
  deposit = deposit_percentage * total_price →
  total_price - deposit = 720 := by
sorry

end remaining_payment_l87_8709


namespace partial_fraction_decomposition_l87_8753

theorem partial_fraction_decomposition :
  let C : ℚ := 81 / 16
  let D : ℚ := -49 / 16
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
    (7 * x - 3) / (x^2 - 8*x - 48) = C / (x - 12) + D / (x + 4) :=
by
  sorry

end partial_fraction_decomposition_l87_8753


namespace union_of_A_and_B_l87_8761

open Set

-- Define sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end union_of_A_and_B_l87_8761


namespace nested_root_simplification_l87_8750

theorem nested_root_simplification (b : ℝ) (h : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by sorry

end nested_root_simplification_l87_8750


namespace arc_length_calculation_l87_8731

/-- 
Given an arc with radius π cm and central angle 120°, 
prove that its arc length is (2/3)π² cm.
-/
theorem arc_length_calculation (r : ℝ) (θ_degrees : ℝ) (l : ℝ) : 
  r = π → θ_degrees = 120 → l = (2/3) * π^2 → 
  l = r * (θ_degrees * π / 180) :=
by sorry

end arc_length_calculation_l87_8731


namespace quadratic_coefficient_l87_8783

/-- A quadratic function with vertex (-3, 2) passing through (2, -43) has a = -9/5 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) →  -- vertex form
  (a * 2^2 + b * 2 + c = -43) →                       -- passes through (2, -43)
  a = -9/5 := by
sorry

end quadratic_coefficient_l87_8783


namespace m_range_l87_8719

/-- Two points are on opposite sides of a line if the product of their signed distances from the line is negative -/
def opposite_sides (x₁ y₁ x₂ y₂ : ℝ) (a b c : ℝ) : Prop :=
  (a * x₁ + b * y₁ + c) * (a * x₂ + b * y₂ + c) < 0

/-- The theorem stating the range of m given the conditions -/
theorem m_range (m : ℝ) : 
  opposite_sides m 0 2 m 1 1 (-1) → -1 < m ∧ m < 1 := by
  sorry


end m_range_l87_8719


namespace strawberry_candies_count_candy_problem_l87_8786

theorem strawberry_candies_count : ℕ → ℕ → Prop :=
  fun total grape_diff =>
    ∀ (strawberry grape : ℕ),
      strawberry + grape = total →
      grape = strawberry - grape_diff →
      strawberry = 121

theorem candy_problem : strawberry_candies_count 240 2 := by
  sorry

end strawberry_candies_count_candy_problem_l87_8786


namespace nine_circles_problem_l87_8702

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers from 1 to 9 are used exactly once in the grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), g i j = n

/-- Represents a triangle in the grid by its three vertex coordinates -/
structure Triangle where
  v1 : Fin 3 × Fin 3
  v2 : Fin 3 × Fin 3
  v3 : Fin 3 × Fin 3

/-- List of all 7 triangles in the grid -/
def triangles : List Triangle := sorry

/-- Checks if the sum of numbers at the vertices of a triangle is 15 -/
def triangle_sum_is_15 (g : Grid) (t : Triangle) : Prop :=
  (g t.v1.1 t.v1.2).val + (g t.v2.1 t.v2.2).val + (g t.v3.1 t.v3.2).val = 15

/-- The main theorem: there exists a valid grid where all triangles sum to 15 -/
theorem nine_circles_problem :
  ∃ (g : Grid), is_valid_grid g ∧ ∀ t ∈ triangles, triangle_sum_is_15 g t :=
sorry

end nine_circles_problem_l87_8702


namespace dans_music_store_spending_l87_8707

/-- The amount Dan spent at the music store -/
def amount_spent (clarinet_cost song_book_cost amount_left : ℚ) : ℚ :=
  clarinet_cost + song_book_cost - amount_left

/-- Proof that Dan spent $129.22 at the music store -/
theorem dans_music_store_spending :
  amount_spent 130.30 11.24 12.32 = 129.22 := by
  sorry

end dans_music_store_spending_l87_8707


namespace complex_fraction_equals_i_l87_8743

theorem complex_fraction_equals_i (i : ℂ) (hi : i^2 = -1) :
  (2 + i) / (1 - 2*i) = i := by
  sorry

end complex_fraction_equals_i_l87_8743


namespace inequality_proof_l87_8706

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end inequality_proof_l87_8706


namespace negation_of_negative_square_positive_is_false_l87_8716

theorem negation_of_negative_square_positive_is_false : 
  ¬(∀ x : ℝ, x < 0 → x^2 > 0) = False := by sorry

end negation_of_negative_square_positive_is_false_l87_8716


namespace letter_value_proof_l87_8771

/-- Given random integer values for letters of the alphabet, prove that A = 16 -/
theorem letter_value_proof (M A T E : ℤ) : 
  M + A + T + 8 = 28 →
  T + E + A + M = 34 →
  M + E + E + T = 30 →
  A = 16 := by
  sorry

end letter_value_proof_l87_8771


namespace square_of_divisibility_l87_8711

theorem square_of_divisibility (m n : ℤ) 
  (h1 : m ≠ 0) 
  (h2 : n ≠ 0) 
  (h3 : m % 2 = n % 2) 
  (h4 : (n^2 - 1) % (m^2 - n^2 + 1) = 0) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
sorry

end square_of_divisibility_l87_8711


namespace decimal_to_fraction_l87_8724

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by
  sorry

end decimal_to_fraction_l87_8724


namespace average_height_combined_groups_l87_8763

theorem average_height_combined_groups
  (group1_count : ℕ)
  (group2_count : ℕ)
  (total_count : ℕ)
  (average_height : ℝ)
  (h1 : group1_count = 20)
  (h2 : group2_count = 11)
  (h3 : total_count = group1_count + group2_count)
  (h4 : average_height = 20) :
  (group1_count * average_height + group2_count * average_height) / total_count = average_height :=
by sorry

end average_height_combined_groups_l87_8763


namespace cousins_distribution_l87_8751

-- Define the number of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 3

-- Function to calculate the number of ways to distribute cousins
def distribute_cousins (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem stating the result
theorem cousins_distribution :
  distribute_cousins num_cousins num_rooms = 66 := by sorry

end cousins_distribution_l87_8751


namespace f_odd_f_inequality_iff_a_range_l87_8775

noncomputable section

def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_inequality_iff_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 ∧ x < 2 → f (a * x^2 + 2) + f (2 * x - 1) > 0) ↔ a > -5/4 := by sorry

end f_odd_f_inequality_iff_a_range_l87_8775


namespace problem_one_problem_two_l87_8797

-- Problem 1
theorem problem_one (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.cos α = -3 * a / (5 * (-a))) 
  (h3 : Real.sin α = 4 * a / (5 * (-a))) : 
  Real.sin α + 2 * Real.cos α = 2/5 := by sorry

-- Problem 2
theorem problem_two (β : ℝ) (h : Real.tan β = 2) : 
  Real.sin β ^ 2 + 2 * Real.sin β * Real.cos β = 8/5 := by sorry

end problem_one_problem_two_l87_8797


namespace square_sum_equals_fifty_l87_8790

theorem square_sum_equals_fifty (x y : ℝ) 
  (h1 : x + y = -10) 
  (h2 : x = 25 / y) : 
  x^2 + y^2 = 50 := by
sorry

end square_sum_equals_fifty_l87_8790


namespace expected_total_rain_l87_8717

/-- Represents the possible rain outcomes for a day --/
inductive RainOutcome
  | NoRain
  | ThreeInches
  | EightInches

/-- Probability of each rain outcome --/
def rainProbability (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0.5
  | RainOutcome.ThreeInches => 0.3
  | RainOutcome.EightInches => 0.2

/-- Amount of rain for each outcome in inches --/
def rainAmount (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0
  | RainOutcome.ThreeInches => 3
  | RainOutcome.EightInches => 8

/-- Number of days in the forecast --/
def forecastDays : ℕ := 5

/-- Expected value of rain for a single day --/
def dailyExpectedRain : ℝ :=
  (rainProbability RainOutcome.NoRain * rainAmount RainOutcome.NoRain) +
  (rainProbability RainOutcome.ThreeInches * rainAmount RainOutcome.ThreeInches) +
  (rainProbability RainOutcome.EightInches * rainAmount RainOutcome.EightInches)

/-- Theorem: The expected value of the total amount of rain for the forecast period is 12.5 inches --/
theorem expected_total_rain :
  forecastDays * dailyExpectedRain = 12.5 := by
  sorry


end expected_total_rain_l87_8717


namespace orthogonal_vectors_l87_8738

/-- Two vectors are orthogonal if and only if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The problem statement -/
theorem orthogonal_vectors (x : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, x)
  orthogonal a b ↔ x = -3/2 := by
sorry

end orthogonal_vectors_l87_8738


namespace proposition_implication_l87_8798

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
sorry

end proposition_implication_l87_8798


namespace factor_count_l87_8722

def n : ℕ := 2^2 * 3^2 * 7^2

def is_factor (d : ℕ) : Prop := d ∣ n

def is_even (d : ℕ) : Prop := d % 2 = 0

def is_odd (d : ℕ) : Prop := d % 2 = 1

theorem factor_count :
  (∃ (even_factors : Finset ℕ) (odd_factors : Finset ℕ),
    (∀ d ∈ even_factors, is_factor d ∧ is_even d) ∧
    (∀ d ∈ odd_factors, is_factor d ∧ is_odd d) ∧
    (Finset.card even_factors = 18) ∧
    (Finset.card odd_factors = 9) ∧
    (∀ d : ℕ, is_factor d → (d ∈ even_factors ∨ d ∈ odd_factors))) :=
by sorry

end factor_count_l87_8722


namespace square_sum_given_condition_l87_8748

theorem square_sum_given_condition (x y : ℝ) :
  (x - 3)^2 + |2 * y + 1| = 0 → x^2 + y^2 = 9 + 1/4 := by
  sorry

end square_sum_given_condition_l87_8748


namespace no_solutions_to_inequality_l87_8752

theorem no_solutions_to_inequality :
  ¬∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end no_solutions_to_inequality_l87_8752


namespace min_cans_proof_l87_8715

/-- The capacity of a special edition soda can in ounces -/
def can_capacity : ℕ := 15

/-- Half a gallon in ounces -/
def half_gallon : ℕ := 64

/-- The minimum number of cans needed to provide at least half a gallon of soda -/
def min_cans : ℕ := 5

theorem min_cans_proof :
  (∀ n : ℕ, n * can_capacity ≥ half_gallon → n ≥ min_cans) ∧
  (min_cans * can_capacity ≥ half_gallon) :=
sorry

end min_cans_proof_l87_8715


namespace identity_unique_l87_8705

-- Define a group structure
class MyGroup (G : Type) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c : G, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : G, mul one a = a
  mul_one : ∀ a : G, mul a one = a
  mul_left_inv : ∀ a : G, mul (inv a) a = one

-- State the theorem
theorem identity_unique {G : Type} [MyGroup G] (e e' : G)
    (h1 : ∀ g : G, MyGroup.mul e g = g ∧ MyGroup.mul g e = g)
    (h2 : ∀ g : G, MyGroup.mul e' g = g ∧ MyGroup.mul g e' = g) :
    e = e' := by sorry

end identity_unique_l87_8705


namespace planar_graph_iff_euler_l87_8792

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  v : ℕ  -- number of vertices
  g : ℕ  -- number of edges
  s : ℕ  -- number of faces

/-- Euler's formula for planar graphs states that v - g + s = 2 -/
def satisfiesEulersFormula (graph : PlanarGraph) : Prop :=
  graph.v - graph.g + graph.s = 2

/-- A planar graph can be constructed if and only if it satisfies Euler's formula -/
theorem planar_graph_iff_euler (graph : PlanarGraph) :
  ∃ (G : PlanarGraph), G.v = graph.v ∧ G.g = graph.g ∧ G.s = graph.s ↔ satisfiesEulersFormula graph :=
sorry

end planar_graph_iff_euler_l87_8792


namespace toucan_count_l87_8714

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end toucan_count_l87_8714


namespace marble_problem_l87_8742

theorem marble_problem (a : ℚ) : 
  let brian := 3 * a - 4
  let caden := 2 * brian + 2
  let daryl := 4 * caden
  a + brian + caden + daryl = 122 → a = 78 / 17 := by
sorry

end marble_problem_l87_8742


namespace inscribed_hexagon_area_ratio_l87_8791

/-- The ratio of areas between an inscribed hexagon with side length s/2 
    and an outer hexagon with side length s is 1/4 -/
theorem inscribed_hexagon_area_ratio (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 * (s/2)^2 / 2) / (3 * Real.sqrt 3 * s^2 / 2) = 1/4 := by
  sorry

end inscribed_hexagon_area_ratio_l87_8791


namespace half_day_division_count_l87_8725

/-- The number of seconds in a half-day -/
def half_day_seconds : ℕ := 43200

/-- The number of ways to divide a half-day into periods -/
def num_divisions : ℕ := 60

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    satisfying n * m = half_day_seconds is equal to num_divisions -/
theorem half_day_division_count :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds ∧ 
                                   p.1 > 0 ∧ p.2 > 0) 
                 (Finset.product (Finset.range (half_day_seconds + 1)) 
                                 (Finset.range (half_day_seconds + 1)))).card = num_divisions :=
sorry

end half_day_division_count_l87_8725


namespace number_equation_solution_l87_8757

theorem number_equation_solution : 
  ∃ (number : ℝ), 35 - (23 - (number - 32)) = 12 * 2 / (1 / 2) ∧ number = 68 := by
  sorry

end number_equation_solution_l87_8757


namespace package_cost_l87_8785

/-- The cost to mail each package, given the total amount spent, cost per letter, number of letters, and relationship between letters and packages. -/
theorem package_cost (total_spent : ℚ) (letter_cost : ℚ) (num_letters : ℕ) 
  (h1 : total_spent = 4.49)
  (h2 : letter_cost = 0.37)
  (h3 : num_letters = 5)
  (h4 : num_letters = num_packages + 2) : 
  (total_spent - num_letters * letter_cost) / (num_letters - 2) = 0.88 := by
  sorry

end package_cost_l87_8785


namespace max_triangle_area_l87_8764

theorem max_triangle_area (a b : ℝ) (ha : a = 1984) (hb : b = 2016) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → (1/2) * a * b * Real.sin θ ≤ 1998912) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ (1/2) * a * b * Real.sin θ = 1998912) := by
  sorry

end max_triangle_area_l87_8764


namespace expression_equality_l87_8759

theorem expression_equality (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 := by
  sorry

end expression_equality_l87_8759


namespace max_cubes_from_seven_points_l87_8720

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Determines if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Determines if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (plane : Plane3D), pointOnPlane p1 plane ∧ pointOnPlane p2 plane ∧ 
                       pointOnPlane p3 plane ∧ pointOnPlane p4 plane

/-- Represents a cube determined by 7 points -/
structure Cube where
  a1 : Point3D
  a2 : Point3D
  f1 : Point3D
  f2 : Point3D
  e : Point3D
  h : Point3D
  j : Point3D
  lowerFace : Plane3D
  upperFace : Plane3D
  frontFace : Plane3D
  backFace : Plane3D
  rightFace : Plane3D

/-- The main theorem to prove -/
theorem max_cubes_from_seven_points 
  (a1 a2 f1 f2 e h j : Point3D)
  (h1 : pointOnPlane a1 (Cube.lowerFace cube))
  (h2 : pointOnPlane a2 (Cube.lowerFace cube))
  (h3 : pointOnPlane f1 (Cube.upperFace cube))
  (h4 : pointOnPlane f2 (Cube.upperFace cube))
  (h5 : ¬ areCoplanar a1 a2 f1 f2)
  (h6 : pointOnPlane e (Cube.frontFace cube))
  (h7 : pointOnPlane h (Cube.backFace cube))
  (h8 : pointOnPlane j (Cube.rightFace cube))
  : ∃ (n : ℕ), n ≤ 2 ∧ ∀ (m : ℕ), (∃ (cubes : Fin m → Cube), 
    (∀ (i : Fin m), 
      Cube.a1 (cubes i) = a1 ∧
      Cube.a2 (cubes i) = a2 ∧
      Cube.f1 (cubes i) = f1 ∧
      Cube.f2 (cubes i) = f2 ∧
      Cube.e (cubes i) = e ∧
      Cube.h (cubes i) = h ∧
      Cube.j (cubes i) = j) ∧
    (∀ (i j : Fin m), i ≠ j → cubes i ≠ cubes j)) → m ≤ n :=
sorry

end max_cubes_from_seven_points_l87_8720


namespace polynomial_simplification_l87_8794

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
sorry

end polynomial_simplification_l87_8794


namespace f_max_at_neg_two_l87_8732

def f (x : ℝ) : ℝ := x^3 - 12*x

theorem f_max_at_neg_two :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≤ f m :=
sorry

end f_max_at_neg_two_l87_8732


namespace imaginary_part_of_complex_fraction_l87_8769

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im (10 * i / (1 - 2 * i)) = 2 :=
by
  sorry

end imaginary_part_of_complex_fraction_l87_8769


namespace absolute_value_equation_solutions_l87_8749

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 2 ∨ x = 8/3 := by
sorry

end absolute_value_equation_solutions_l87_8749


namespace equal_abc_l87_8758

theorem equal_abc (a b c x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (x * b + (1 - x) * c) / a = (x * c + (1 - x) * a) / b)
  (h5 : (x * b + (1 - x) * c) / a = (x * a + (1 - x) * b) / c) :
  a = b ∧ b = c := by
  sorry

end equal_abc_l87_8758


namespace tourist_catch_up_l87_8729

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

theorem tourist_catch_up 
  (v_bicycle : ℝ) 
  (v_motorcycle : ℝ) 
  (initial_ride_time : ℝ) 
  (break_time : ℝ) 
  (delay_time : ℝ) :
  v_bicycle = 16 →
  v_motorcycle = 56 →
  initial_ride_time = 1.5 →
  break_time = 1.5 →
  delay_time = 4 →
  ∃ t : ℝ, 
    t > 0 ∧ 
    v_bicycle * (initial_ride_time + t) = 
    v_motorcycle * t + v_bicycle * initial_ride_time ∧
    v_bicycle * (initial_ride_time + t) = catch_up_distance :=
by sorry

end tourist_catch_up_l87_8729


namespace kitchen_width_l87_8799

/-- Calculates the width of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_width (length height : ℝ) (total_painted_area : ℝ) : 
  length = 12 ∧ 
  height = 10 ∧ 
  total_painted_area = 1680 → 
  (total_painted_area / 3) = 2 * (length * height + height * (total_painted_area / (3 * height) / 2)) :=
by sorry

end kitchen_width_l87_8799


namespace sum_of_fractions_l87_8728

theorem sum_of_fractions : 
  (5 : ℚ) / 13 + (9 : ℚ) / 11 = (172 : ℚ) / 143 := by
  sorry

end sum_of_fractions_l87_8728


namespace integral_cube_root_x_squared_plus_sqrt_x_l87_8789

theorem integral_cube_root_x_squared_plus_sqrt_x (x : ℝ) :
  (deriv (fun x => (3/5) * x * (x^2)^(1/3) + (2/3) * x * x^(1/2))) x = x^(2/3) + x^(1/2) :=
by sorry

end integral_cube_root_x_squared_plus_sqrt_x_l87_8789


namespace stratified_sampling_sophomores_l87_8739

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomores : ℕ) (selected : ℕ) 
  (h1 : total_students = 2800) 
  (h2 : sophomores = 930) 
  (h3 : selected = 280) :
  (sophomores * selected) / total_students = 93 := by
  sorry

end stratified_sampling_sophomores_l87_8739


namespace sum_of_angles_in_triangle_l87_8788

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define angles in a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the theorem
theorem sum_of_angles_in_triangle (t : Triangle) : 
  angle t 0 + angle t 1 + angle t 2 = 180 := by
  sorry

end sum_of_angles_in_triangle_l87_8788


namespace min_value_of_f_l87_8721

/-- Given a function f(x) = (a + x^2) / x, where a > 0 and x ∈ (0, b),
    prove that the minimum value of f(x) is 2√a when b > √a. -/
theorem min_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > Real.sqrt a) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt a ∧
    ∀ x ∈ Set.Ioo 0 b, (a + x^2) / x ≥ min_val := by
  sorry

end min_value_of_f_l87_8721


namespace sock_profit_calculation_l87_8703

/-- Calculates the total profit from selling socks given specific conditions. -/
theorem sock_profit_calculation : 
  let total_pairs : ℕ := 9
  let cost_per_pair : ℚ := 2
  let purchase_discount : ℚ := 0.1
  let profit_percentage_4_pairs : ℚ := 0.25
  let profit_per_pair_5_pairs : ℚ := 0.2
  let sales_tax : ℚ := 0.05

  let discounted_cost := total_pairs * cost_per_pair * (1 - purchase_discount)
  let selling_price_4_pairs := 4 * cost_per_pair * (1 + profit_percentage_4_pairs)
  let selling_price_5_pairs := 5 * cost_per_pair + 5 * profit_per_pair_5_pairs
  let total_selling_price := (selling_price_4_pairs + selling_price_5_pairs) * (1 + sales_tax)
  let total_profit := total_selling_price - discounted_cost

  total_profit = 5.85 := by sorry

end sock_profit_calculation_l87_8703


namespace function_value_2024_l87_8704

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_value_2024 (a b c : ℝ) 
  (h2021 : f a b c 2021 = 2021)
  (h2022 : f a b c 2022 = 2022)
  (h2023 : f a b c 2023 = 2023) :
  f a b c 2024 = 2030 := by
  sorry

end function_value_2024_l87_8704


namespace vector_perpendicular_l87_8773

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![0, -2]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_perpendicular :
  perpendicular (λ i => a i + 2 * b i) ![3, 2] := by
  sorry

end vector_perpendicular_l87_8773


namespace smallest_angle_solution_l87_8723

theorem smallest_angle_solution (x : ℝ) : 
  (0 < x) → 
  (∀ y : ℝ, 0 < y → 
    Real.tan (8 * π / 180 * y) = (Real.cos (π / 180 * y) - Real.sin (π / 180 * y)) / (Real.cos (π / 180 * y) + Real.sin (π / 180 * y)) → 
    x ≤ y) → 
  Real.tan (8 * π / 180 * x) = (Real.cos (π / 180 * x) - Real.sin (π / 180 * x)) / (Real.cos (π / 180 * x) + Real.sin (π / 180 * x)) → 
  x = 5 := by
  sorry

end smallest_angle_solution_l87_8723


namespace speed_ratio_is_two_to_one_l87_8740

-- Define the given constants
def distance : ℝ := 30
def original_speed : ℝ := 5

-- Define Sameer's speed as a variable
variable (sameer_speed : ℝ)

-- Define Abhay's new speed as a variable
variable (new_speed : ℝ)

-- Define the conditions
def condition1 (sameer_speed : ℝ) : Prop :=
  distance / original_speed = distance / sameer_speed + 2

def condition2 (sameer_speed new_speed : ℝ) : Prop :=
  distance / new_speed = distance / sameer_speed - 1

-- Theorem to prove
theorem speed_ratio_is_two_to_one 
  (h1 : condition1 sameer_speed)
  (h2 : condition2 sameer_speed new_speed) :
  new_speed / original_speed = 2 := by
  sorry


end speed_ratio_is_two_to_one_l87_8740


namespace vector_magnitude_proof_l87_8727

theorem vector_magnitude_proof (a b : ℝ × ℝ × ℝ) :
  a = (1, 1, 0) ∧ b = (-1, 0, 2) →
  ‖(2 • a) - b‖ = Real.sqrt 17 := by
  sorry

end vector_magnitude_proof_l87_8727


namespace least_number_divisible_l87_8730

theorem least_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((1076 + m) % 23 = 0 ∧ (1076 + m) % 29 = 0 ∧ (1076 + m) % 31 = 0)) ∧ 
  ((1076 + n) % 23 = 0 ∧ (1076 + n) % 29 = 0 ∧ (1076 + n) % 31 = 0) → 
  n = 19601 := by
sorry

end least_number_divisible_l87_8730


namespace cubic_polynomial_integer_root_l87_8795

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 5 - Real.sqrt 11)
  (h2 : ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0 ∧ n = -10 :=
by sorry

end cubic_polynomial_integer_root_l87_8795


namespace proposition_logic_l87_8787

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 ≥ 3)) (hq : q ↔ (3 > 4)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end proposition_logic_l87_8787


namespace stall_problem_l87_8712

theorem stall_problem (area_diff : ℝ) (cost_A cost_B : ℝ) (total_area_A total_area_B : ℝ) (total_stalls : ℕ) :
  area_diff = 2 →
  cost_A = 20 →
  cost_B = 40 →
  total_area_A = 150 →
  total_area_B = 120 →
  total_stalls = 100 →
  ∃ (area_A area_B : ℝ) (num_A num_B : ℕ),
    area_A = area_B + area_diff ∧
    (total_area_A / area_A : ℝ) = (3/4) * (total_area_B / area_B) ∧
    num_A + num_B = total_stalls ∧
    num_B ≥ 3 * num_A ∧
    area_A = 5 ∧
    area_B = 3 ∧
    cost_A * area_A * num_A + cost_B * area_B * num_B = 11500 ∧
    ∀ (other_num_A other_num_B : ℕ),
      other_num_A + other_num_B = total_stalls →
      other_num_B ≥ 3 * other_num_A →
      cost_A * area_A * other_num_A + cost_B * area_B * other_num_B ≥ 11500 :=
by sorry

end stall_problem_l87_8712


namespace xy_value_l87_8735

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end xy_value_l87_8735


namespace solve_exponential_equation_l87_8796

theorem solve_exponential_equation :
  ∃ x : ℝ, 16 = 4 * (4 : ℝ) ^ (x - 1) ∧ x = 2 := by
  sorry

end solve_exponential_equation_l87_8796


namespace parallelogram_side_sum_l87_8734

/-- Given a parallelogram with sides measuring 7, 9, 8y-1, and 2x+3 units consecutively,
    prove that x + y = 4 -/
theorem parallelogram_side_sum (x y : ℝ) : 
  (7 : ℝ) = 8*y - 1 → (9 : ℝ) = 2*x + 3 → x + y = 4 := by
  sorry

end parallelogram_side_sum_l87_8734


namespace square_area_from_perimeter_l87_8776

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 36 →
  area = (perimeter / 4) ^ 2 →
  area = 81 := by
sorry

end square_area_from_perimeter_l87_8776


namespace initial_maple_trees_count_l87_8781

/-- The number of maple trees to be planted -/
def trees_to_plant : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_maple_trees : ℕ := final_maple_trees - trees_to_plant

theorem initial_maple_trees_count : initial_maple_trees = 2 := by
  sorry

end initial_maple_trees_count_l87_8781


namespace range_of_a_satisfying_equation_l87_8782

open Real

theorem range_of_a_satisfying_equation :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 3 * x + a * (2 * y - 4 * ℯ * x) * (log y - log x) = 0) ↔ 
  (a < 0 ∨ a ≥ 3 / (2 * ℯ)) := by
  sorry

end range_of_a_satisfying_equation_l87_8782


namespace grid_column_contains_all_numbers_l87_8760

/-- Represents the state of the grid after a certain number of transformations -/
structure GridState (n : ℕ) :=
  (grid : Fin n → Fin n → Fin n)

/-- Represents the transformation rule for the grid -/
def transform_row (n k m : ℕ) (row : Fin n → Fin n) : Fin n → Fin n :=
  sorry

/-- Fills the grid according to the given rule -/
def fill_grid (n k m : ℕ) : ℕ → GridState n :=
  sorry

theorem grid_column_contains_all_numbers
  (n k m : ℕ) 
  (h_m_gt_k : m > k) 
  (h_coprime : Nat.Coprime m (n - k)) :
  ∀ (col : Fin n), 
    ∃ (rows : Finset (Fin n)), 
      rows.card = n ∧ 
      (∀ i : Fin n, ∃ row ∈ rows, (fill_grid n k m n).grid row col = i) :=
sorry

end grid_column_contains_all_numbers_l87_8760


namespace value_of_expression_l87_8741

theorem value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 := by
  sorry

end value_of_expression_l87_8741


namespace special_square_difference_l87_8770

theorem special_square_difference : 123456789^2 - 123456788 * 123456790 = 1 := by
  sorry

end special_square_difference_l87_8770


namespace negation_of_existential_proposition_l87_8766

theorem negation_of_existential_proposition (l : ℝ) :
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end negation_of_existential_proposition_l87_8766


namespace ratio_y_over_x_is_six_l87_8767

theorem ratio_y_over_x_is_six (x y : ℝ) 
  (h1 : Real.sqrt (3 * x) * (1 + 1 / (x + y)) = 2)
  (h2 : Real.sqrt (7 * y) * (1 - 1 / (x + y)) = 4 * Real.sqrt 2) :
  y / x = 6 := by
sorry

end ratio_y_over_x_is_six_l87_8767


namespace planting_schemes_count_l87_8744

/-- The number of seed types available -/
def total_seeds : ℕ := 5

/-- The number of plots to be planted -/
def plots : ℕ := 4

/-- The number of choices for the first plot -/
def first_plot_choices : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The main theorem stating the total number of planting schemes -/
theorem planting_schemes_count : 
  first_plot_choices * permutations (total_seeds - 1) (plots - 1) = 48 := by
  sorry

end planting_schemes_count_l87_8744


namespace line_AB_slope_and_equation_l87_8733

/-- Given points A(0,-2) and B(√3,1), prove the slope of line AB is √3 and its equation is y = √3x - 2 -/
theorem line_AB_slope_and_equation :
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (Real.sqrt 3, 1)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let equation (x : ℝ) : ℝ := slope * x + (A.2 - slope * A.1)
  slope = Real.sqrt 3 ∧ ∀ x, equation x = Real.sqrt 3 * x - 2 := by
  sorry

end line_AB_slope_and_equation_l87_8733


namespace x_intercept_of_line_l87_8754

/-- The x-intercept of the line 2x + 3y + 6 = 0 is -3 -/
theorem x_intercept_of_line (x y : ℝ) :
  2 * x + 3 * y + 6 = 0 → y = 0 → x = -3 := by
  sorry

end x_intercept_of_line_l87_8754


namespace fourth_side_distance_l87_8736

/-- Given a square and a point inside it, if the distances from the point to three sides are 4, 7, and 12,
    then the distance to the fourth side is either 9 or 15. -/
theorem fourth_side_distance (s : ℝ) (d1 d2 d3 d4 : ℝ) : 
  s > 0 ∧ d1 = 4 ∧ d2 = 7 ∧ d3 = 12 ∧ 
  d1 + d2 + d3 + d4 = s → 
  d4 = 9 ∨ d4 = 15 := by
  sorry

end fourth_side_distance_l87_8736


namespace equation_represents_circle_l87_8710

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 0)^2 + (y - 0)^2 = 25

-- Define what a circle is in terms of its equation
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), f x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem equation_represents_circle : is_circle equation := by
  sorry

end equation_represents_circle_l87_8710


namespace max_spend_amount_l87_8745

/-- Represents the number of coins of each denomination a person has --/
structure CoinCount where
  coin100 : Nat
  coin50  : Nat
  coin10  : Nat

/-- Calculates the total value in won from a given CoinCount --/
def totalValue (coins : CoinCount) : Nat :=
  100 * coins.coin100 + 50 * coins.coin50 + 10 * coins.coin10

/-- Jimin's coin count --/
def jiminCoins : CoinCount := { coin100 := 5, coin50 := 1, coin10 := 0 }

/-- Seok-jin's coin count --/
def seokJinCoins : CoinCount := { coin100 := 2, coin50 := 0, coin10 := 7 }

/-- The theorem stating the maximum amount Jimin and Seok-jin can spend together --/
theorem max_spend_amount :
  totalValue jiminCoins + totalValue seokJinCoins = 820 := by sorry

end max_spend_amount_l87_8745


namespace g_512_minus_g_256_eq_zero_l87_8747

-- Define σ(n) as the sum of all positive divisors of n
def σ (n : ℕ+) : ℕ := sorry

-- Define g(n) = 2σ(n)/n
def g (n : ℕ+) : ℚ := 2 * (σ n) / n

-- Theorem statement
theorem g_512_minus_g_256_eq_zero : g 512 - g 256 = 0 := by sorry

end g_512_minus_g_256_eq_zero_l87_8747


namespace max_a_value_l87_8777

-- Define the quadratic polynomial
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value :
  (∃ (a_max : ℝ), ∀ (a b : ℝ),
    (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y) →
    a ≤ a_max ∧
    (∃ (b : ℝ), ∀ (x : ℝ), ∃ (y : ℝ), f a_max b y = f a_max b x + y)) ∧
  (∀ (a_greater : ℝ),
    (∃ (a b : ℝ), a > a_greater ∧
      (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y)) →
    a_greater < 1/2) :=
sorry

end max_a_value_l87_8777


namespace max_leftover_grapes_l87_8713

theorem max_leftover_grapes (n : ℕ) : ∃ (k : ℕ), n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
by sorry

end max_leftover_grapes_l87_8713


namespace difference_of_squares_l87_8700

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end difference_of_squares_l87_8700


namespace no_solution_condition_l87_8746

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 5 * |x - 4*a| + |x - a^2| + 4*x - 3*a ≠ 0) ↔ (a < -9 ∨ a > 0) := by
  sorry

end no_solution_condition_l87_8746


namespace boys_running_speed_l87_8756

theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 55 →
  time = 88 →
  speed = (4 * side_length / time) * 3.6 →
  speed = 9 := by sorry

end boys_running_speed_l87_8756


namespace joey_studies_five_nights_per_week_l87_8701

/-- Represents Joey's study schedule and calculates the number of weekday study nights per week -/
def joeys_study_schedule (weekday_hours_per_night : ℕ) (weekend_hours_per_day : ℕ) 
  (total_weeks : ℕ) (total_study_hours : ℕ) : ℕ :=
  let weekend_days := 2 * total_weeks
  let weekend_hours := weekend_hours_per_day * weekend_days
  let weekday_hours := total_study_hours - weekend_hours
  let weekday_nights := weekday_hours / weekday_hours_per_night
  weekday_nights / total_weeks

/-- Theorem stating that Joey studies 5 nights per week on weekdays -/
theorem joey_studies_five_nights_per_week :
  joeys_study_schedule 2 3 6 96 = 5 := by
  sorry

end joey_studies_five_nights_per_week_l87_8701


namespace bird_families_count_l87_8793

/-- The number of bird families that flew to Africa -/
def families_to_africa : ℕ := 47

/-- The number of bird families that flew to Asia -/
def families_to_asia : ℕ := 94

/-- The difference between families that flew to Asia and Africa -/
def difference : ℕ := 47

/-- The total number of bird families before migration -/
def total_families : ℕ := families_to_africa + families_to_asia

theorem bird_families_count : 
  (families_to_asia = families_to_africa + difference) → 
  (total_families = 141) := by
  sorry

end bird_families_count_l87_8793
