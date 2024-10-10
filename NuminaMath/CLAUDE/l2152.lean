import Mathlib

namespace interest_calculation_l2152_215292

/-- Calculates the total interest earned on an investment --/
def total_interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the total interest earned is approximately $563.16 --/
theorem interest_calculation : 
  let principal := 1200
  let rate := 0.08
  let time := 5
  abs (total_interest_earned principal rate time - 563.16) < 0.01 := by
  sorry

end interest_calculation_l2152_215292


namespace intersection_of_A_and_B_l2152_215203

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l2152_215203


namespace volume_of_bounded_figure_l2152_215278

-- Define a cube with edge length 1
def cube : Set (Fin 3 → ℝ) := {v | ∀ i, 0 ≤ v i ∧ v i ≤ 1}

-- Define the planes through centers of adjacent sides
def planes : Set (Set (Fin 3 → ℝ)) :=
  {p | ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p = {v | v i + v j + v k = 3/2}}

-- Define the bounded figure
def bounded_figure : Set (Fin 3 → ℝ) :=
  {v ∈ cube | ∀ p ∈ planes, v ∈ p}

-- Theorem statement
theorem volume_of_bounded_figure :
  MeasureTheory.volume bounded_figure = 1/2 := by sorry

end volume_of_bounded_figure_l2152_215278


namespace largest_divisor_consecutive_odd_squares_l2152_215200

/-- Two integers are consecutive odd numbers if their difference is 2 and they are both odd -/
def ConsecutiveOddNumbers (m n : ℤ) : Prop :=
  m - n = 2 ∧ Odd m ∧ Odd n

/-- The largest divisor of m^2 - n^2 for consecutive odd numbers m and n where n < m is 8 -/
theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (h1 : ConsecutiveOddNumbers m n) (h2 : n < m) : 
  (∃ (k : ℤ), m^2 - n^2 = 8 * k) ∧ 
  (∀ (d : ℤ), d > 8 → ¬(∀ (j : ℤ), m^2 - n^2 = d * j)) := by
  sorry

end largest_divisor_consecutive_odd_squares_l2152_215200


namespace function_range_contained_in_unit_interval_l2152_215216

theorem function_range_contained_in_unit_interval 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end function_range_contained_in_unit_interval_l2152_215216


namespace isosceles_triangle_base_length_l2152_215250

theorem isosceles_triangle_base_length 
  (perimeter : ℝ) 
  (one_side : ℝ) 
  (h_perimeter : perimeter = 15) 
  (h_one_side : one_side = 3) 
  (h_isosceles : ∃ (leg : ℝ), 2 * leg + one_side = perimeter) :
  one_side = 3 := by
  sorry

end isosceles_triangle_base_length_l2152_215250


namespace correct_calculation_l2152_215234

theorem correct_calculation (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end correct_calculation_l2152_215234


namespace sample_size_is_80_l2152_215298

/-- Represents the ratio of products A, B, and C in production -/
def productionRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C

/-- The number of products of type B selected in the sample -/
def selectedB : ℕ := 24

/-- The total sample size -/
def n : ℕ := 80

/-- Theorem stating that the given conditions lead to a sample size of 80 -/
theorem sample_size_is_80 : 
  (productionRatio 1 : ℚ) / (productionRatio 0 + productionRatio 1 + productionRatio 2) = selectedB / n :=
sorry

end sample_size_is_80_l2152_215298


namespace f_2020_is_sin_l2152_215223

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2020_is_sin : f 2020 = Real.sin := by
  sorry

end f_2020_is_sin_l2152_215223


namespace triangle_properties_l2152_215220

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / (Real.sin A) = b / (Real.sin B) ∧ 
  b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition: b^2 + c^2 - a^2 = bc
  b^2 + c^2 - a^2 = b * c →
  -- Area of triangle is 3√3/2
  (1/2) * a * b * (Real.sin C) = (3 * Real.sqrt 3) / 2 →
  -- Given condition: sin C + √3 cos C = 2
  Real.sin C + Real.sqrt 3 * Real.cos C = 2 →
  -- Prove: A = π/3 and a = 3
  A = π/3 ∧ a = 3 := by sorry

end triangle_properties_l2152_215220


namespace wooden_block_volume_l2152_215261

/-- A rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.width * block.height

/-- The surface area of a wooden block -/
def surfaceArea (block : WoodenBlock) : ℝ :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- The increase in surface area after sawing -/
def surfaceAreaIncrease (block : WoodenBlock) (sections : ℕ) : ℝ :=
  2 * (sections - 1) * block.width * block.height

theorem wooden_block_volume
  (block : WoodenBlock)
  (h_length : block.length = 10)
  (h_sections : ℕ)
  (h_sections_eq : h_sections = 6)
  (h_area_increase : surfaceAreaIncrease block h_sections = 1) :
  volume block = 10 := by
  sorry

end wooden_block_volume_l2152_215261


namespace reflection_x_axis_l2152_215283

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_x_axis (P : ℝ × ℝ) (h : P = (-2, 1)) :
  reflect_x P = (-2, -1) := by
  sorry

end reflection_x_axis_l2152_215283


namespace s_scale_indeterminate_l2152_215233

/-- Represents a linear relationship between two measurement scales -/
structure ScaleRelation where
  /-- Slope of the linear relationship -/
  a : ℝ
  /-- Y-intercept of the linear relationship -/
  b : ℝ

/-- Converts a p-scale measurement to an s-scale measurement -/
def toSScale (relation : ScaleRelation) (p : ℝ) : ℝ :=
  relation.a * p + relation.b

/-- Theorem stating that the s-scale measurement for p=24 cannot be uniquely determined -/
theorem s_scale_indeterminate (known_p : ℝ) (known_s : ℝ) (target_p : ℝ) 
    (h1 : known_p = 6) (h2 : known_s = 30) (h3 : target_p = 24) :
    ∃ (r1 r2 : ScaleRelation), r1 ≠ r2 ∧ 
    toSScale r1 known_p = known_s ∧
    toSScale r2 known_p = known_s ∧
    toSScale r1 target_p ≠ toSScale r2 target_p :=
  sorry

end s_scale_indeterminate_l2152_215233


namespace G_minimized_at_three_l2152_215274

/-- The number of devices required for a base-n system -/
noncomputable def G (n : ℕ) (M : ℕ) : ℝ :=
  (n : ℝ) / Real.log n * Real.log (M + 1)

/-- The theorem stating that G is minimized when n = 3 -/
theorem G_minimized_at_three (M : ℕ) :
  ∀ n : ℕ, n ≥ 2 → G 3 M ≤ G n M :=
sorry

end G_minimized_at_three_l2152_215274


namespace point_with_specific_rate_of_change_l2152_215260

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem point_with_specific_rate_of_change :
  ∃ (x y : ℝ), f x = y ∧ f' x = 5 ∧ x = 4 ∧ y = 9 := by sorry

end point_with_specific_rate_of_change_l2152_215260


namespace sum_of_exponents_15_factorial_l2152_215213

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def primeFactorExponents (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  (primeFactorExponents (largestPerfectSquareDivisor (factorial 15))).sum = 10 := by
  sorry

end sum_of_exponents_15_factorial_l2152_215213


namespace third_number_proof_l2152_215214

theorem third_number_proof (A B C : ℕ+) : 
  A = 600 → 
  B = 840 → 
  Nat.gcd A (Nat.gcd B C) = 60 →
  Nat.lcm A (Nat.lcm B C) = 50400 →
  C = 6 := by
sorry

end third_number_proof_l2152_215214


namespace base_five_representation_l2152_215247

theorem base_five_representation (b : ℕ) : 
  (b^3 ≤ 329 ∧ 329 < b^4 ∧ 329 % b % 2 = 0) ↔ b = 5 :=
by sorry

end base_five_representation_l2152_215247


namespace arcsin_arccos_equation_solutions_l2152_215290

theorem arcsin_arccos_equation_solutions :
  ∀ x : ℝ, (x = 0 ∨ x = 1/2 ∨ x = -1/2) →
  Real.arcsin (2*x) + Real.arcsin (1 - 2*x) = Real.arccos (2*x) := by
sorry

end arcsin_arccos_equation_solutions_l2152_215290


namespace log_sum_difference_l2152_215211

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 4 / Real.log 10 = 2 + Real.log 2.5 / Real.log 10 := by
  sorry

end log_sum_difference_l2152_215211


namespace crow_eating_quarter_l2152_215295

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def crow_eating_time (fraction_eaten : ℚ) (time : ℚ) : Prop :=
  fraction_eaten * time = (1 : ℚ) / 5 * 6

/-- Proves that it takes 7.5 hours for a crow to eat 1/4 of the nuts, 
    given that it eats 1/5 of the nuts in 6 hours -/
theorem crow_eating_quarter : 
  crow_eating_time (1 / 4) (15 / 2) :=
sorry

end crow_eating_quarter_l2152_215295


namespace central_number_is_ten_l2152_215285

/-- A triangular grid with 10 integers -/
structure TriangularGrid :=
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 x : ℤ)

/-- The sum of all ten numbers is 43 -/
def total_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 + g.b1 + g.b2 + g.b3 + g.c1 + g.c2 + g.c3 + g.x = 43

/-- The sum of any three numbers such that any two of them are close is 11 -/
def close_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 = 11 ∧
  g.b1 + g.b2 + g.b3 = 11 ∧
  g.c1 + g.c2 + g.c3 = 11

/-- Theorem: The central number is 10 -/
theorem central_number_is_ten (g : TriangularGrid) 
  (h1 : total_sum g) (h2 : close_sum g) : g.x = 10 := by
  sorry

end central_number_is_ten_l2152_215285


namespace circumscribed_sphere_surface_area_l2152_215218

/-- Given a triangle with side lengths 8, 10, and 12, the surface area of the circumscribed sphere
    of the triangular prism formed by connecting the midpoints of the triangle's sides
    is equal to 77π/2. -/
theorem circumscribed_sphere_surface_area (A₁ A₂ A₃ B C D : ℝ × ℝ × ℝ) : 
  let side_lengths := [8, 10, 12]
  ∀ (a b c : ℝ), 
    a ∈ side_lengths ∧ 
    b ∈ side_lengths ∧ 
    c ∈ side_lengths ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (‖A₁ - A₂‖ = a ∧ ‖A₂ - A₃‖ = b ∧ ‖A₃ - A₁‖ = c) →
    (B = (A₁ + A₂) / 2 ∧ C = (A₂ + A₃) / 2 ∧ D = (A₃ + A₁) / 2) →
    let R := Real.sqrt (77 / 8)
    4 * π * R^2 = 77 * π / 2 :=
by sorry

end circumscribed_sphere_surface_area_l2152_215218


namespace expand_product_l2152_215257

theorem expand_product (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end expand_product_l2152_215257


namespace ball_hits_ground_time_l2152_215212

theorem ball_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ -8 * t^2 - 12 * t + 72 = 0 ∧ abs (t - 2.34) < 0.01 := by
  sorry

end ball_hits_ground_time_l2152_215212


namespace triangle_properties_l2152_215267

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : Real.cos t.A = 1/8)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 15 * Real.sqrt 7 / 4) :
  Real.sin t.C = Real.sqrt 7 / 4 ∧ t.b + t.c = 9 := by
  sorry

end triangle_properties_l2152_215267


namespace max_value_expression_l2152_215296

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c))) = 3/2 * (b^2 + c) + 3 * a^2 := by
sorry

end max_value_expression_l2152_215296


namespace problem_solution_l2152_215230

theorem problem_solution (x y : ℤ) : 
  x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
  sorry

end problem_solution_l2152_215230


namespace train_crossing_time_l2152_215209

/-- Proves that a train crossing a signal pole takes 18 seconds given specific conditions -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_length = 300 →
  platform_length = 600.0000000000001 →
  platform_crossing_time = 54 →
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end train_crossing_time_l2152_215209


namespace land_area_needed_l2152_215241

def land_cost_per_sqm : ℝ := 50
def brick_cost_per_1000 : ℝ := 100
def roof_tile_cost_per_tile : ℝ := 10
def num_bricks_needed : ℝ := 10000
def num_roof_tiles_needed : ℝ := 500
def total_construction_cost : ℝ := 106000

theorem land_area_needed :
  ∃ (x : ℝ),
    x * land_cost_per_sqm +
    (num_bricks_needed / 1000) * brick_cost_per_1000 +
    num_roof_tiles_needed * roof_tile_cost_per_tile =
    total_construction_cost ∧
    x = 2000 :=
by sorry

end land_area_needed_l2152_215241


namespace remainder_problem_l2152_215281

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 5 * 2 + R ∧ R < 5) → 
  (∃ Q, N = 4 * Q + 2) → 
  (∃ R, N = 5 * 2 + R ∧ R = 4) := by
sorry

end remainder_problem_l2152_215281


namespace berry_package_cost_l2152_215246

/-- The cost of one package of berries given Martin's consumption habits and spending --/
theorem berry_package_cost (daily_consumption : ℚ) (package_size : ℚ) (days : ℕ) (total_spent : ℚ) : 
  daily_consumption = 1/2 →
  package_size = 1 →
  days = 30 →
  total_spent = 30 →
  (total_spent / (days * daily_consumption / package_size) = 2) :=
by
  sorry

end berry_package_cost_l2152_215246


namespace stream_rate_calculation_l2152_215258

/-- Proves that given a boat with speed 16 km/hr in still water, traveling 126 km downstream in 6 hours, the rate of the stream is 5 km/hr. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_rate : ℝ) :
  boat_speed = 16 →
  distance = 126 →
  time = 6 →
  distance = (boat_speed + stream_rate) * time →
  stream_rate = 5 := by
sorry

end stream_rate_calculation_l2152_215258


namespace unique_prime_with_prime_successors_l2152_215282

theorem unique_prime_with_prime_successors :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) ∧ p = 3 := by
  sorry

end unique_prime_with_prime_successors_l2152_215282


namespace boys_in_class_l2152_215254

theorem boys_in_class (initial_girls : ℕ) (initial_boys : ℕ) (final_girls : ℕ) :
  (initial_girls : ℚ) / initial_boys = 5 / 6 →
  (final_girls : ℚ) / initial_boys = 2 / 3 →
  initial_girls - final_girls = 20 →
  initial_boys = 120 := by
sorry

end boys_in_class_l2152_215254


namespace union_of_A_and_B_l2152_215288

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end union_of_A_and_B_l2152_215288


namespace arithmetic_sequence_sum_l2152_215229

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 4 = 3 →
  a 5 = 7 →
  a 6 = 11 →
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) = -5 :=
by
  sorry

end arithmetic_sequence_sum_l2152_215229


namespace regular_octagon_area_1_5_sqrt_2_l2152_215236

/-- The area of a regular octagon with side length s -/
noncomputable def regularOctagonArea (s : ℝ) : ℝ := 2 * s^2 * (1 + Real.sqrt 2)

theorem regular_octagon_area_1_5_sqrt_2 :
  regularOctagonArea (1.5 * Real.sqrt 2) = 9 + 9 * Real.sqrt 2 := by
  sorry

end regular_octagon_area_1_5_sqrt_2_l2152_215236


namespace chess_tournament_games_l2152_215287

theorem chess_tournament_games (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 = 36 := by
  sorry

end chess_tournament_games_l2152_215287


namespace father_son_age_difference_l2152_215228

theorem father_son_age_difference : ∀ (father_age son_age : ℕ),
  son_age = 33 →
  father_age + 2 = 2 * (son_age + 2) →
  father_age - son_age = 35 := by
  sorry

end father_son_age_difference_l2152_215228


namespace blocks_left_l2152_215201

/-- Given that Randy had 59 blocks initially and used 36 blocks to build a tower,
    prove that he has 23 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : used_blocks = 36) :
  initial_blocks - used_blocks = 23 :=
by sorry

end blocks_left_l2152_215201


namespace set_equality_implies_coefficients_l2152_215297

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem set_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end set_equality_implies_coefficients_l2152_215297


namespace prob_different_suits_no_jokers_l2152_215293

/-- Extended deck with 54 cards including 2 jokers -/
def extendedDeck : ℕ := 54

/-- Number of jokers in the extended deck -/
def jokers : ℕ := 2

/-- Number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- Probability of picking two cards of different suits given no jokers are picked -/
theorem prob_different_suits_no_jokers :
  let nonJokerCards := extendedDeck - jokers
  let firstPickOptions := nonJokerCards
  let secondPickOptions := nonJokerCards - 1
  let differentSuitOptions := (numSuits - 1) * cardsPerSuit
  (differentSuitOptions : ℚ) / secondPickOptions = 13 / 17 := by
  sorry

end prob_different_suits_no_jokers_l2152_215293


namespace beths_marbles_l2152_215291

/-- Proves that given the conditions of Beth's marble problem, she initially had 72 marbles. -/
theorem beths_marbles (initial_per_color : ℕ) : 
  (3 * initial_per_color) - (5 + 10 + 15) = 42 → 
  3 * initial_per_color = 72 := by
  sorry

end beths_marbles_l2152_215291


namespace grade_distribution_l2152_215289

theorem grade_distribution (total_students : ℕ) 
  (prob_A : ℝ) (prob_B : ℝ) (prob_C : ℝ) 
  (h1 : prob_A = 0.8 * prob_B) 
  (h2 : prob_C = 1.2 * prob_B) 
  (h3 : prob_A + prob_B + prob_C = 1) 
  (h4 : total_students = 40) :
  ∃ (A B C : ℕ), 
    A + B + C = total_students ∧ 
    A = 10 ∧ 
    B = 14 ∧ 
    C = 16 := by
  sorry

end grade_distribution_l2152_215289


namespace parabola_locus_l2152_215269

/-- The locus of points from which a parabola is seen at a 45° angle -/
theorem parabola_locus (p : ℝ) (u v : ℝ) : 
  (∃ (m₁ m₂ : ℝ), 
    -- Two distinct tangent lines exist
    m₁ ≠ m₂ ∧
    -- The tangent lines touch the parabola
    (∀ (x y : ℝ), y^2 = 2*p*x → (y - v = m₁*(x - u) ∨ y - v = m₂*(x - u))) ∧
    -- The angle between the tangent lines is 45°
    (m₁ - m₂) / (1 + m₁*m₂) = 1) →
  -- The point (u, v) lies on the hyperbola
  (u + 3*p/2)^2 - v^2 = 2*p^2 :=
by sorry

end parabola_locus_l2152_215269


namespace sequence_bound_l2152_215294

theorem sequence_bound (k : ℕ) (h_k : k > 0) : 
  (∃ (a : ℕ → ℚ), a 0 = 1 / k ∧ 
    (∀ n : ℕ, n > 0 → a n = a (n - 1) + (1 : ℚ) / (n ^ 2 : ℚ) * (a (n - 1)) ^ 2) ∧
    (∀ n : ℕ, n > 0 → a n < 1)) ↔ 
  k ≥ 3 :=
sorry

end sequence_bound_l2152_215294


namespace dance_group_average_age_l2152_215205

theorem dance_group_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_female_age : ℝ) 
  (avg_male_age : ℝ) 
  (h1 : num_females = 12)
  (h2 : avg_female_age = 25)
  (h3 : num_males = 18)
  (h4 : avg_male_age = 40)
  (h5 : num_females + num_males = 30) : 
  (num_females * avg_female_age + num_males * avg_male_age) / (num_females + num_males) = 34 := by
  sorry

end dance_group_average_age_l2152_215205


namespace problem_solution_l2152_215240

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = Real.sqrt 6 / 2 := by
  sorry

end problem_solution_l2152_215240


namespace white_patterns_count_l2152_215264

/-- The number of different white figures on an n × n board created by k rectangles -/
def whitePatterns (n k : ℕ) : ℕ :=
  (Nat.choose n k) ^ 2

/-- Theorem stating the number of different white figures -/
theorem white_patterns_count (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : k ≤ n) :
  whitePatterns n k = (Nat.choose n k) ^ 2 := by
  sorry

end white_patterns_count_l2152_215264


namespace discount_difference_l2152_215255

theorem discount_difference : 
  let initial_amount : ℝ := 12000
  let single_discount : ℝ := 0.3
  let first_successive_discount : ℝ := 0.2
  let second_successive_discount : ℝ := 0.1
  let single_discounted_amount : ℝ := initial_amount * (1 - single_discount)
  let successive_discounted_amount : ℝ := initial_amount * (1 - first_successive_discount) * (1 - second_successive_discount)
  successive_discounted_amount - single_discounted_amount = 240 :=
by sorry

end discount_difference_l2152_215255


namespace arithmetic_expression_equality_l2152_215263

theorem arithmetic_expression_equality : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end arithmetic_expression_equality_l2152_215263


namespace adult_tickets_sold_l2152_215248

/-- Proves that the number of adult tickets sold is 122, given the total number of tickets
    and the relationship between student and adult tickets. -/
theorem adult_tickets_sold (total_tickets : ℕ) (adult_tickets : ℕ) (student_tickets : ℕ)
    (h1 : total_tickets = 366)
    (h2 : student_tickets = 2 * adult_tickets)
    (h3 : total_tickets = adult_tickets + student_tickets) :
    adult_tickets = 122 := by
  sorry

end adult_tickets_sold_l2152_215248


namespace faye_crayons_count_l2152_215253

/-- Given that Faye arranges her crayons in 16 rows with 6 crayons per row,
    prove that she has 96 crayons in total. -/
theorem faye_crayons_count : 
  let rows : ℕ := 16
  let crayons_per_row : ℕ := 6
  rows * crayons_per_row = 96 := by
sorry

end faye_crayons_count_l2152_215253


namespace base9_calculation_l2152_215243

/-- Converts a base 10 number to its base 9 representation -/
def toBase9 (n : ℕ) : ℕ := sorry

/-- Converts a base 9 number to its base 10 representation -/
def fromBase9 (n : ℕ) : ℕ := sorry

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a + fromBase9 b)

/-- Subtraction in base 9 -/
def subBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a - fromBase9 b)

theorem base9_calculation :
  subBase9 (addBase9 (addBase9 2365 1484) 782) 671 = 4170 := by sorry

end base9_calculation_l2152_215243


namespace smallest_tablecloth_diameter_l2152_215245

/-- The smallest diameter of a circular tablecloth that can completely cover a square table with sides of 1 meter is √2 meters. -/
theorem smallest_tablecloth_diameter (table_side : ℝ) (h : table_side = 1) :
  let diagonal := Real.sqrt (2 * table_side ^ 2)
  diagonal = Real.sqrt 2 := by
  sorry

end smallest_tablecloth_diameter_l2152_215245


namespace number_percentage_problem_l2152_215215

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 14 → (40/100 : ℝ) * N = 168 := by
  sorry

end number_percentage_problem_l2152_215215


namespace largest_base_for_12_4th_power_l2152_215251

/-- Given a natural number n and a base b, returns the sum of digits of n in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is a valid base (greater than 1) -/
def isValidBase (b : ℕ) : Prop := b > 1

theorem largest_base_for_12_4th_power :
  ∀ b : ℕ, isValidBase b →
    (b ≤ 7 ↔ sumOfDigits (12^4) b ≠ 2^5) ∧
    (b > 7 → sumOfDigits (12^4) b = 2^5) :=
sorry

end largest_base_for_12_4th_power_l2152_215251


namespace periodic_sine_function_l2152_215225

theorem periodic_sine_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  a ∈ Set.Ioo 0 π →
  (∀ x, f (x + a) = f (x + 3 * a)) →
  a = π / 2 := by
sorry

end periodic_sine_function_l2152_215225


namespace company_sales_royalties_l2152_215222

/-- A company's sales and royalties problem -/
theorem company_sales_royalties
  (initial_sales : ℝ)
  (initial_royalties : ℝ)
  (next_royalties : ℝ)
  (royalty_ratio_decrease : ℝ)
  (h1 : initial_sales = 10000000)
  (h2 : initial_royalties = 2000000)
  (h3 : next_royalties = 8000000)
  (h4 : royalty_ratio_decrease = 0.6)
  : ∃ (next_sales : ℝ), next_sales = 100000000 ∧
    next_royalties / next_sales = (initial_royalties / initial_sales) * (1 - royalty_ratio_decrease) :=
by sorry

end company_sales_royalties_l2152_215222


namespace max_min_sum_l2152_215280

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum (M m : ℝ) (hM : ∀ x, f x ≤ M) (hm : ∀ x, m ≤ f x) : M + m = 4 := by
  sorry

end max_min_sum_l2152_215280


namespace arithmetic_expression_equality_l2152_215242

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = -14 := by
  sorry

end arithmetic_expression_equality_l2152_215242


namespace arithmetic_sequence_sum_l2152_215268

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 4 → a 2 + a 6 = 8 := by
  sorry

end arithmetic_sequence_sum_l2152_215268


namespace snack_package_average_l2152_215202

theorem snack_package_average (cookie_counts : List ℕ)
  (candy_counts : List ℕ) (pie_counts : List ℕ) :
  cookie_counts.length = 4 →
  candy_counts.length = 3 →
  pie_counts.length = 2 →
  cookie_counts.sum + candy_counts.sum + pie_counts.sum = 153 →
  cookie_counts.sum / cookie_counts.length = 17 →
  (cookie_counts.sum + candy_counts.sum + pie_counts.sum) /
    (cookie_counts.length + candy_counts.length + pie_counts.length) = 17 :=
by
  sorry

end snack_package_average_l2152_215202


namespace walkway_and_border_area_is_912_l2152_215286

/-- Represents the dimensions and layout of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  border_width : ℕ

/-- Calculates the total area of walkways and decorative border in the garden -/
def walkway_and_border_area (g : Garden) : ℕ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width + 2 * g.border_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width + 2 * g.border_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - beds_area

/-- Theorem stating that the walkway and border area for the given garden specifications is 912 square feet -/
theorem walkway_and_border_area_is_912 :
  walkway_and_border_area ⟨4, 3, 8, 3, 2, 4⟩ = 912 := by
  sorry

end walkway_and_border_area_is_912_l2152_215286


namespace hundred_with_six_digits_l2152_215232

theorem hundred_with_six_digits (x : ℕ) (h : x ≠ 0) (h2 : x < 10) :
  (100 * x + 10 * x + x) - (10 * x + x) = 100 * x :=
by sorry

end hundred_with_six_digits_l2152_215232


namespace nested_fraction_square_l2152_215284

theorem nested_fraction_square (x : ℚ) (h : x = 1/3) :
  let f := (x + 2) / (x - 2)
  ((f + 2) / (f - 2))^2 = 961/1369 := by
  sorry

end nested_fraction_square_l2152_215284


namespace correct_equation_l2152_215237

/-- Represents the problem of sending a letter over a certain distance with two horses of different speeds. -/
def letter_problem (distance : ℝ) (slow_delay : ℝ) (fast_early : ℝ) (speed_ratio : ℝ) :=
  ∀ x : ℝ, x > 3 → (distance / (x + slow_delay)) * speed_ratio = distance / (x - fast_early)

/-- The theorem states that the given equation correctly represents the problem for the specific values mentioned. -/
theorem correct_equation : letter_problem 900 1 3 2 := by sorry

end correct_equation_l2152_215237


namespace quadratic_integer_intersections_l2152_215262

def f (m : ℕ+) (x : ℝ) : ℝ := m * x^2 + (-m - 2) * x + 2

theorem quadratic_integer_intersections (m : ℕ+) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) →
  (f m 1 = 0 ∧ f m 2 = 0 ∧ f m 0 = 2) :=
sorry

end quadratic_integer_intersections_l2152_215262


namespace solutions_to_z_sixth_eq_neg_eight_l2152_215244

theorem solutions_to_z_sixth_eq_neg_eight :
  {z : ℂ | z^6 = -8} = {1 + I, 1 - I, -1 + I, -1 - I} := by
  sorry

end solutions_to_z_sixth_eq_neg_eight_l2152_215244


namespace cyclic_trapezoid_area_l2152_215238

/-- Represents a cyclic trapezoid with parallel sides a and b, where a < b -/
structure CyclicTrapezoid where
  a : ℝ
  b : ℝ
  h : a < b

/-- The area of a cyclic trapezoid given the conditions -/
def area (t : CyclicTrapezoid) : Set ℝ :=
  let t₁ := (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  let t₂ := (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  {t₁, t₂}

/-- Theorem stating that the area of the cyclic trapezoid is either t₁ or t₂ -/
theorem cyclic_trapezoid_area (t : CyclicTrapezoid) :
  ∃ A ∈ area t, A = (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4 ∨
                 A = (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4 := by
  sorry


end cyclic_trapezoid_area_l2152_215238


namespace spinning_class_duration_l2152_215226

/-- Calculates the number of hours worked out in each spinning class. -/
def hours_per_class (classes_per_week : ℕ) (calories_per_minute : ℕ) (total_calories_per_week : ℕ) : ℚ :=
  (total_calories_per_week / classes_per_week) / (calories_per_minute * 60)

/-- Proves that given the specified conditions, James works out for 1.5 hours in each spinning class. -/
theorem spinning_class_duration :
  let classes_per_week : ℕ := 3
  let calories_per_minute : ℕ := 7
  let total_calories_per_week : ℕ := 1890
  hours_per_class classes_per_week calories_per_minute total_calories_per_week = 3/2 := by
  sorry

#eval hours_per_class 3 7 1890

end spinning_class_duration_l2152_215226


namespace solution_set_inequality_l2152_215219

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 50 60 : Set ℝ) = {x | (x - 50) * (60 - x) > 0} :=
by sorry

end solution_set_inequality_l2152_215219


namespace swimmer_problem_l2152_215210

/-- Swimmer problem -/
theorem swimmer_problem (a s r : ℝ) (ha : a > 0) (hs : s > 0) (hr : r > 0) 
  (h_order : s < r ∧ r < (100 * s) / (50 + s)) :
  ∃ (x z : ℝ),
    x = (100 * s - 50 * r - r * s) / ((3 * s - r) * a) ∧
    z = (100 * s - 50 * r - r * s) / ((r - s) * a) ∧
    x > 0 ∧ z > 0 ∧
    ∃ (y t : ℝ),
      y > 0 ∧ t > 0 ∧
      t * z = (t + a) * y ∧
      t * z = (t + 2 * a) * x ∧
      (50 + r) / z = (50 - r) / x - 2 * a ∧
      (50 + s) / z = (50 - s) / y - a :=
by
  sorry

end swimmer_problem_l2152_215210


namespace people_needed_for_two_hours_l2152_215208

/-- Represents the rate at which water enters the boat (units per hour) -/
def water_entry_rate : ℝ := 2

/-- Represents the amount of water one person can bail out per hour -/
def bailing_rate : ℝ := 1

/-- Represents the total amount of water to be bailed out -/
def total_water : ℝ := 30

/-- Given the conditions from the problem, proves that 14 people are needed to bail out the water in 2 hours -/
theorem people_needed_for_two_hours : 
  (∀ (p : ℕ), p = 10 → p * bailing_rate * 3 = total_water + water_entry_rate * 3) →
  (∀ (p : ℕ), p = 5 → p * bailing_rate * 8 = total_water + water_entry_rate * 8) →
  ∃ (p : ℕ), p = 14 ∧ p * bailing_rate * 2 = total_water + water_entry_rate * 2 :=
by sorry

end people_needed_for_two_hours_l2152_215208


namespace projection_of_congruent_vectors_l2152_215299

/-- Definition of vector congruence -/
def is_congruent (a b : ℝ × ℝ) : Prop :=
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.sqrt (a.1^2 + a.2^2) / Real.sqrt (b.1^2 + b.2^2) = Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

/-- Theorem: Projection of a-b on a when b is congruent to a -/
theorem projection_of_congruent_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0))
    (h_congruent : is_congruent b a) :
  let proj := ((a.1 - b.1) * a.1 + (a.2 - b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  proj = (a.1^2 + a.2^2 - (b.1^2 + b.2^2)) / Real.sqrt (a.1^2 + a.2^2) := by
  sorry

end projection_of_congruent_vectors_l2152_215299


namespace sixtieth_pair_is_five_six_l2152_215259

/-- Represents a pair of integers in the sequence -/
structure Pair :=
  (first : ℕ)
  (second : ℕ)

/-- Returns the sum of elements in a pair -/
def pairSum (p : Pair) : ℕ := p.first + p.second

/-- Returns the number of pairs in the first n levels -/
def pairsInFirstNLevels (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Returns the nth pair in the sequence -/
def nthPair (n : ℕ) : Pair :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem sixtieth_pair_is_five_six :
  nthPair 60 = Pair.mk 5 6 := by
  sorry

end sixtieth_pair_is_five_six_l2152_215259


namespace arithmetic_sequence_sum_l2152_215224

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₂ = 9 and S₄ = 22, S₈ = 60 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h₂ : seq.S 2 = 9)
    (h₄ : seq.S 4 = 22) :
    seq.S 8 = 60 := by
  sorry

end arithmetic_sequence_sum_l2152_215224


namespace fraction_equals_zero_l2152_215207

theorem fraction_equals_zero (x : ℝ) :
  (x + 2) / (3 - x) = 0 ∧ 3 - x ≠ 0 → x = -2 := by
  sorry

end fraction_equals_zero_l2152_215207


namespace point_on_h_graph_l2152_215265

theorem point_on_h_graph (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 7 → 
  (∀ x, h x = (g x + 1)^2) → 
  ∃ x y, h x = y ∧ x + y = 68 := by
sorry

end point_on_h_graph_l2152_215265


namespace cubic_equation_result_l2152_215276

theorem cubic_equation_result (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end cubic_equation_result_l2152_215276


namespace initial_water_percentage_is_70_l2152_215270

/-- The percentage of liquid X in solution Y -/
def liquid_x_percentage : ℝ := 30

/-- The initial mass of solution Y in kg -/
def initial_mass : ℝ := 8

/-- The mass of water that evaporates in kg -/
def evaporated_water : ℝ := 3

/-- The mass of solution Y added after evaporation in kg -/
def added_solution : ℝ := 3

/-- The percentage of liquid X in the new solution -/
def new_liquid_x_percentage : ℝ := 41.25

/-- The initial percentage of water in solution Y -/
def initial_water_percentage : ℝ := 100 - liquid_x_percentage

theorem initial_water_percentage_is_70 :
  initial_water_percentage = 70 :=
sorry

end initial_water_percentage_is_70_l2152_215270


namespace tank_capacity_l2152_215279

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (used_gallons : ℕ) : 
  initial_fraction = 3/4 → 
  final_fraction = 1/4 → 
  used_gallons = 24 → 
  ∃ (total_capacity : ℕ), 
    total_capacity = 48 ∧ 
    (initial_fraction - final_fraction) * total_capacity = used_gallons :=
by
  sorry

end tank_capacity_l2152_215279


namespace game_result_l2152_215217

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def chris_rolls : List ℕ := [5, 2, 1, 6]
def dana_rolls : List ℕ := [6, 2, 3, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result :
  let chris_points := total_points chris_rolls
  let dana_points := total_points dana_rolls
  dana_points = 27 ∧ chris_points * dana_points = 297 := by sorry

end game_result_l2152_215217


namespace equation_solution_l2152_215249

theorem equation_solution : 
  {x : ℝ | (x ≠ 0 ∧ x + 2 ≠ 0 ∧ x + 4 ≠ 0 ∧ x + 6 ≠ 0 ∧ x + 8 ≠ 0) ∧ 
           (1/x + 1/(x+2) - 1/(x+4) - 1/(x+6) + 1/(x+8) = 0)} = 
  {-4 - 2 * Real.sqrt 3, 2 - 2 * Real.sqrt 3} := by
sorry

end equation_solution_l2152_215249


namespace seungchan_book_pages_l2152_215239

/-- The number of pages in Seungchan's children's book -/
def total_pages : ℝ := 250

/-- The fraction of the book Seungchan read until yesterday -/
def read_yesterday : ℝ := 0.2

/-- The fraction of the remaining part Seungchan read today -/
def read_today : ℝ := 0.35

/-- The number of pages left after today's reading -/
def pages_left : ℝ := 130

theorem seungchan_book_pages :
  (1 - read_yesterday) * (1 - read_today) * total_pages = pages_left :=
sorry

end seungchan_book_pages_l2152_215239


namespace total_money_value_l2152_215252

def us_100_bills : ℕ := 2
def us_50_bills : ℕ := 5
def us_10_bills : ℕ := 5
def canadian_20_bills : ℕ := 15
def euro_10_notes : ℕ := 20
def us_quarters : ℕ := 50
def us_dimes : ℕ := 120

def cad_to_usd_rate : ℚ := 0.80
def eur_to_usd_rate : ℚ := 1.10

def total_us_currency : ℚ := 
  us_100_bills * 100 + 
  us_50_bills * 50 + 
  us_10_bills * 10 + 
  us_quarters * 0.25 + 
  us_dimes * 0.10

def total_cad_in_usd : ℚ := canadian_20_bills * 20 * cad_to_usd_rate
def total_eur_in_usd : ℚ := euro_10_notes * 10 * eur_to_usd_rate

theorem total_money_value : 
  total_us_currency + total_cad_in_usd + total_eur_in_usd = 984.50 := by
  sorry

end total_money_value_l2152_215252


namespace prime_square_diff_divisible_by_24_l2152_215227

theorem prime_square_diff_divisible_by_24 (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) : 
  24 ∣ (p^2 - q^2) := by
  sorry

end prime_square_diff_divisible_by_24_l2152_215227


namespace square_side_length_l2152_215271

/-- Given a square and a rectangle with specific properties, prove that the side length of the square is 15 cm. -/
theorem square_side_length (s : ℝ) : 
  s > 0 →  -- side length is positive
  4 * s = 2 * (18 + 216 / 18) →  -- perimeters are equal
  s = 15 := by
  sorry

end square_side_length_l2152_215271


namespace bus_stop_walk_time_l2152_215204

theorem bus_stop_walk_time (usual_time : ℝ) (usual_speed : ℝ) : 
  usual_speed > 0 →
  (2 / 3 * usual_speed) * (usual_time + 15) = usual_speed * usual_time →
  usual_time = 30 := by
  sorry

end bus_stop_walk_time_l2152_215204


namespace f_2019_equals_2_l2152_215256

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) + f (x - 2) = 2 * f 2) ∧
  (∀ x, f (x + 1) = -f (-x - 1)) ∧
  (f 1 = 2)

theorem f_2019_equals_2 (f : ℝ → ℝ) (h : f_property f) : f 2019 = 2 := by
  sorry

end f_2019_equals_2_l2152_215256


namespace circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l2152_215206

/-- The fixed point P through which all lines pass -/
def P : ℝ × ℝ := (2, -1)

/-- The radius of the circle -/
def r : ℝ := 2

/-- The line equation that passes through P for all a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop :=
  (1 - a) * x + y + 2 * a - 1 = 0

/-- The circle equation with center P and radius r -/
def circle_equation (x y : ℝ) : Prop :=
  (x - P.1)^2 + (y - P.2)^2 = r^2

theorem circle_equation_simplified :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

theorem fixed_point_satisfies_line :
  ∀ a : ℝ, line_equation a P.1 P.2 :=
by sorry

theorem main_theorem :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l2152_215206


namespace sum_first_last_is_14_l2152_215235

/-- A sequence of seven terms satisfying specific conditions -/
structure SevenTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  R_eq_7 : R = 7
  sum_consecutive_3 : ∀ (x y z : ℝ), (x = P ∧ y = Q ∧ z = R) ∨
                                     (x = Q ∧ y = R ∧ z = S) ∨
                                     (x = R ∧ y = S ∧ z = T) ∨
                                     (x = S ∧ y = T ∧ z = U) ∨
                                     (x = T ∧ y = U ∧ z = V) →
                                     x + y + z = 21

/-- The sum of the first and last terms in a seven-term sequence is 14 -/
theorem sum_first_last_is_14 (seq : SevenTermSequence) : seq.P + seq.V = 14 := by
  sorry

end sum_first_last_is_14_l2152_215235


namespace intersection_sum_l2152_215275

/-- Given two lines y = mx + 5 and y = 2x + b intersecting at (7, 10),
    prove that the sum of constants b and m is equal to -23/7 -/
theorem intersection_sum (m b : ℚ) : 
  (7 * m + 5 = 10) → (2 * 7 + b = 10) → b + m = -23/7 := by
  sorry

end intersection_sum_l2152_215275


namespace price_increase_demand_decrease_l2152_215272

theorem price_increase_demand_decrease (P Q : ℝ) (P_new Q_new : ℝ) : 
  P_new = 1.2 * P →  -- Price increases by 20%
  P_new * Q_new = 1.1 * (P * Q) →  -- Total income increases by 10%
  (Q - Q_new) / Q = 1 / 12 :=  -- Demand decreases by 1/12
by sorry

end price_increase_demand_decrease_l2152_215272


namespace article_cost_price_l2152_215221

theorem article_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_increase : ℝ) 
  (h1 : loss_percentage = 25)
  (h2 : gain_percentage = 15)
  (h3 : price_increase = 500) : 
  ∃ (cost_price : ℝ), 
    cost_price * (1 - loss_percentage / 100) + price_increase = cost_price * (1 + gain_percentage / 100) ∧ 
    cost_price = 1250 := by
  sorry

#check article_cost_price

end article_cost_price_l2152_215221


namespace cubic_roots_from_quadratic_roots_l2152_215273

theorem cubic_roots_from_quadratic_roots (a b c d x₁ x₂ : ℝ) :
  (x₁^2 - (a + d)*x₁ + ad - bc = 0) →
  (x₂^2 - (a + d)*x₂ + ad - bc = 0) →
  ((x₁^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₁^3) + (a*d - b*c)^3 = 0) ∧
  ((x₂^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₂^3) + (a*d - b*c)^3 = 0) :=
by sorry

end cubic_roots_from_quadratic_roots_l2152_215273


namespace binomial_12_11_l2152_215277

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binomial_12_11_l2152_215277


namespace modular_inverse_of_35_mod_37_l2152_215231

theorem modular_inverse_of_35_mod_37 :
  ∃ x : ℤ, (35 * x) % 37 = 1 ∧ x % 37 = 18 := by
  sorry

end modular_inverse_of_35_mod_37_l2152_215231


namespace complex_simplification_l2152_215266

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) - 2*i * (3 - 4*i) = 20 - 20*i :=
by sorry

end complex_simplification_l2152_215266
