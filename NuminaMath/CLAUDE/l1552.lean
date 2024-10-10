import Mathlib

namespace second_item_is_14_l1552_155251

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  initial_selection : ℕ

/-- Calculates the second item in a systematic sample -/
def second_item (s : SystematicSampling) : ℕ :=
  s.initial_selection + (s.population_size / s.sample_size)

/-- Theorem: In the given systematic sampling scenario, the second item is 14 -/
theorem second_item_is_14 :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    initial_selection := 4
  }
  second_item s = 14 := by
  sorry


end second_item_is_14_l1552_155251


namespace wings_temperature_l1552_155294

/-- Given an initial oven temperature and a required temperature increase,
    calculate the final required temperature. -/
def required_temperature (initial_temp increase : ℕ) : ℕ :=
  initial_temp + increase

/-- Theorem: The required temperature for the wings is 546 degrees,
    given an initial temperature of 150 degrees and a needed increase of 396 degrees. -/
theorem wings_temperature : required_temperature 150 396 = 546 := by
  sorry

end wings_temperature_l1552_155294


namespace cone_volume_over_pi_l1552_155235

/-- Given a cone formed from a 240-degree sector of a circle with radius 24,
    the volume of the cone divided by π is equal to 2048√5/3 -/
theorem cone_volume_over_pi (r : ℝ) (θ : ℝ) :
  r = 24 →
  θ = 240 * π / 180 →
  let base_radius := r * θ / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 2048 * Real.sqrt 5 / 3 := by
sorry

end cone_volume_over_pi_l1552_155235


namespace sum_of_preceding_terms_l1552_155239

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_preceding_terms (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a (n - 1) = 39 →
  n ≥ 3 →
  a (n - 2) + a (n - 3) = 60 := by
  sorry

end sum_of_preceding_terms_l1552_155239


namespace rectangle_area_with_inscribed_circle_rectangle_area_is_588_l1552_155270

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let circle_radius : ℝ := 7
  let length_to_width_ratio : ℝ := 3
  let rectangle_width : ℝ := 2 * circle_radius
  let rectangle_length : ℝ := length_to_width_ratio * rectangle_width
  rectangle_length * rectangle_width

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end rectangle_area_with_inscribed_circle_rectangle_area_is_588_l1552_155270


namespace system_solution_l1552_155203

theorem system_solution :
  ∃! (x y : ℝ), (3 * x = 2 * y) ∧ (x - 2 * y = -4) :=
by
  sorry

end system_solution_l1552_155203


namespace three_parallel_lines_theorem_l1552_155231

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary properties for a plane in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a function to check if three lines are coplanar
def are_coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count the number of planes determined by three lines
def count_planes (l1 l2 l3 : Line3D) : Nat :=
  sorry -- Count the number of planes

-- Define a function to count the number of parts space is divided into by these planes
def count_space_parts (planes : List Plane3D) : Nat :=
  sorry -- Count the number of parts

-- Theorem statement
theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  (count_planes a b c = 3) ∧ 
  (count_space_parts (sorry : List Plane3D) = 7) := by
  sorry


end three_parallel_lines_theorem_l1552_155231


namespace cats_total_l1552_155291

theorem cats_total (initial_cats bought_cats : Float) : 
  initial_cats = 11.0 → bought_cats = 43.0 → initial_cats + bought_cats = 54.0 := by
  sorry

end cats_total_l1552_155291


namespace platform_pillar_height_l1552_155256

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular platform with pillars -/
structure Platform where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  slopeAngle : ℝ
  pillarHeightP : ℝ
  pillarHeightQ : ℝ
  pillarHeightR : ℝ

/-- The height of the pillar at point S -/
def pillarHeightS (p : Platform) : ℝ :=
  sorry

theorem platform_pillar_height
  (p : Platform)
  (h_PQ : p.Q.x - p.P.x = 10)
  (h_PR : p.R.y - p.P.y = 15)
  (h_slope : p.slopeAngle = π / 6)
  (h_heightP : p.pillarHeightP = 7)
  (h_heightQ : p.pillarHeightQ = 10)
  (h_heightR : p.pillarHeightR = 12) :
  pillarHeightS p = 7.5 * Real.sqrt 3 :=
sorry

end platform_pillar_height_l1552_155256


namespace bill_face_value_l1552_155265

/-- Face value of a bill given true discount and banker's discount -/
def face_value (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount * bankers_discount / (bankers_discount - true_discount)

/-- Theorem: Given the true discount and banker's discount, prove the face value is 2460 -/
theorem bill_face_value :
  face_value 360 421.7142857142857 = 2460 := by
  sorry

end bill_face_value_l1552_155265


namespace number_of_divisors_36_l1552_155219

theorem number_of_divisors_36 : Nat.card {d : ℕ | d ∣ 36} = 9 := by
  sorry

end number_of_divisors_36_l1552_155219


namespace complex_number_quadrant_l1552_155238

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l1552_155238


namespace total_wood_pieces_l1552_155292

/-- The number of pieces of wood that can be contained in one sack -/
def sack_capacity : ℕ := 20

/-- The number of sacks filled with wood -/
def filled_sacks : ℕ := 4

/-- Theorem stating that the total number of wood pieces gathered is equal to
    the product of sack capacity and the number of filled sacks -/
theorem total_wood_pieces :
  sack_capacity * filled_sacks = 80 := by sorry

end total_wood_pieces_l1552_155292


namespace image_of_A_under_f_l1552_155215

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

theorem image_of_A_under_f : Set.image f A = {1, 4} := by sorry

end image_of_A_under_f_l1552_155215


namespace distance_between_vertices_l1552_155214

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = -(1/12) * x^2 + 3
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ x y : ℝ, equation x y → 
  (parabola1 x y ∨ parabola2 x y) → 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
sorry

end distance_between_vertices_l1552_155214


namespace quadratic_sum_l1552_155259

/-- A quadratic function passing through two given points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  point1 : p + q + r = 5
  point2 : 4*p + 2*q + r = 3

/-- The theorem stating that p+q+2r equals 10 for the given quadratic function -/
theorem quadratic_sum (g : QuadraticFunction) : g.p + g.q + 2*g.r = 10 := by
  sorry

#check quadratic_sum

end quadratic_sum_l1552_155259


namespace percentage_difference_l1552_155290

theorem percentage_difference (x y : ℝ) (h : x = 0.65 * y) : y = (1 + 0.35) * x := by
  sorry

end percentage_difference_l1552_155290


namespace intersection_of_sets_l1552_155264

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - x < 0} →
  B = {x : ℝ | -2 < x ∧ x < 2} →
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_sets_l1552_155264


namespace original_cube_volume_l1552_155236

/-- Given two similar cubes where one has twice the side length of the other,
    if the larger cube has a volume of 216 cubic feet,
    then the smaller cube has a volume of 27 cubic feet. -/
theorem original_cube_volume
  (s : ℝ)  -- side length of the original cube
  (h1 : (2 * s) ^ 3 = 216)  -- volume of the larger cube is 216 cubic feet
  : s ^ 3 = 27 :=
by
  sorry

end original_cube_volume_l1552_155236


namespace sin_cos_function_at_pi_12_l1552_155243

theorem sin_cos_function_at_pi_12 :
  let f : ℝ → ℝ := λ x ↦ Real.sin x ^ 4 - Real.cos x ^ 4
  f (π / 12) = -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_function_at_pi_12_l1552_155243


namespace probability_of_white_ball_l1552_155268

theorem probability_of_white_ball
  (P_red P_black P_yellow P_white : ℝ)
  (h_red : P_red = 1/3)
  (h_black_yellow : P_black + P_yellow = 5/12)
  (h_yellow_white : P_yellow + P_white = 5/12)
  (h_sum : P_red + P_black + P_yellow + P_white = 1)
  (h_nonneg : P_red ≥ 0 ∧ P_black ≥ 0 ∧ P_yellow ≥ 0 ∧ P_white ≥ 0) :
  P_white = 1/4 :=
sorry

end probability_of_white_ball_l1552_155268


namespace picnic_difference_l1552_155213

/-- Proves that the difference between adults and children at a picnic is 20 --/
theorem picnic_difference (total : ℕ) (men : ℕ) (women : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 200 → 
  men = women + 20 → 
  men = 65 → 
  total = adults + children → 
  adults = men + women → 
  adults - children = 20 := by
sorry

end picnic_difference_l1552_155213


namespace equation_holds_for_seven_halves_l1552_155287

theorem equation_holds_for_seven_halves : 
  let x : ℚ := 7/2
  let y : ℚ := (x^2 - 9) / (x - 3)
  y = 3*x - 4 := by sorry

end equation_holds_for_seven_halves_l1552_155287


namespace points_for_tie_l1552_155275

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of points awarded for a win -/
def points_for_win : ℕ := 3

/-- The number of points awarded for a loss -/
def points_for_loss : ℕ := 0

/-- The difference between max and min total points -/
def max_min_difference : ℕ := 15

/-- The number of games played in the tournament -/
def num_games : ℕ := (num_teams * (num_teams - 1)) / 2

theorem points_for_tie (T : ℕ) : 
  (num_games * points_for_win) - (num_games * T) = max_min_difference → T = 2 := by
  sorry

end points_for_tie_l1552_155275


namespace total_silk_dyed_l1552_155202

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end total_silk_dyed_l1552_155202


namespace largest_k_is_correct_l1552_155286

/-- The largest natural number k for which there exists a natural number n 
    satisfying the inequality sin(n + 1) < sin(n + 2) < sin(n + 3) < ... < sin(n + k) -/
def largest_k : ℕ := 3

/-- Predicate that checks if the sine inequality holds for a given n and k -/
def sine_inequality (n k : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → Real.sin (n + i) < Real.sin (n + j)

theorem largest_k_is_correct :
  (∃ n : ℕ, sine_inequality n largest_k) ∧
  (∀ k : ℕ, k > largest_k → ¬∃ n : ℕ, sine_inequality n k) :=
sorry

end largest_k_is_correct_l1552_155286


namespace triangle_inequality_squared_l1552_155249

/-- Given a triangle with sides a, b, and c, prove that (a^2 + b^2 + ab) / c^2 < 1 --/
theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < c) (triangle : c < a + b) :
  (a^2 + b^2 + a*b) / c^2 < 1 := by
  sorry

#check triangle_inequality_squared

end triangle_inequality_squared_l1552_155249


namespace union_of_sets_l1552_155295

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end union_of_sets_l1552_155295


namespace sum_of_multiples_l1552_155257

def smallest_two_digit_multiple_of_7 : ℕ := sorry

def smallest_three_digit_multiple_of_5 : ℕ := sorry

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_7 + smallest_three_digit_multiple_of_5 = 114 := by sorry

end sum_of_multiples_l1552_155257


namespace base_conversion_equality_l1552_155220

-- Define the base-5 number 132₅
def base_5_num : ℕ := 1 * 5^2 + 3 * 5^1 + 2 * 5^0

-- Define the base-b number 221ᵦ as a function of b
def base_b_num (b : ℝ) : ℝ := 2 * b^2 + 2 * b + 1

-- Theorem statement
theorem base_conversion_equality :
  ∃ b : ℝ, b > 0 ∧ base_5_num = base_b_num b ∧ b = (-1 + Real.sqrt 83) / 2 := by
  sorry

end base_conversion_equality_l1552_155220


namespace oil_measurement_l1552_155266

/-- The amount of oil currently in Scarlett's measuring cup -/
def current_oil : ℝ := 0.16666666666666674

/-- The amount of oil Scarlett adds to the measuring cup -/
def added_oil : ℝ := 0.6666666666666666

/-- The total amount of oil after Scarlett adds more -/
def total_oil : ℝ := 0.8333333333333334

/-- Theorem stating that the current amount of oil plus the added amount equals the total amount -/
theorem oil_measurement :
  current_oil + added_oil = total_oil :=
by sorry

end oil_measurement_l1552_155266


namespace equilateral_triangle_properties_l1552_155298

theorem equilateral_triangle_properties (side : ℝ) (h : side = 20) :
  let height := side * (Real.sqrt 3) / 2
  let half_side := side / 2
  height = 10 * Real.sqrt 3 ∧ half_side = 10 := by
  sorry

end equilateral_triangle_properties_l1552_155298


namespace quadratic_factorization_l1552_155208

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 10 * a * x + 25 * a = a * (x - 5)^2 := by
  sorry

end quadratic_factorization_l1552_155208


namespace cost_price_of_ball_l1552_155224

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 15 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 72 := by
  sorry

end cost_price_of_ball_l1552_155224


namespace complex_fraction_equality_l1552_155205

theorem complex_fraction_equality : (5 * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I := by
  sorry

end complex_fraction_equality_l1552_155205


namespace min_sum_dimensions_for_volume_2184_l1552_155278

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a rectangular box -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the sum of dimensions of a rectangular box -/
def sumDimensions (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem: The minimum sum of dimensions for a box with volume 2184 is 36 -/
theorem min_sum_dimensions_for_volume_2184 :
  (∃ (d : BoxDimensions), volume d = 2184) →
  (∀ (d : BoxDimensions), volume d = 2184 → sumDimensions d ≥ 36) ∧
  (∃ (d : BoxDimensions), volume d = 2184 ∧ sumDimensions d = 36) := by
  sorry

end min_sum_dimensions_for_volume_2184_l1552_155278


namespace quadratic_max_value_change_l1552_155277

theorem quadratic_max_value_change (a b c : ℝ) (h_a : a < 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let max_value (a' : ℝ) := -b^2 / (4 * a') + c
  (max_value (a + 1) = max_value a + 27 / 2) →
  (max_value (a - 4) = max_value a - 9) →
  (max_value (a - 2) = max_value a - 27 / 4) :=
by sorry

end quadratic_max_value_change_l1552_155277


namespace min_time_for_eight_people_l1552_155255

/-- Represents a group of people sharing information -/
structure InformationSharingGroup where
  numPeople : Nat
  callDuration : Nat
  initialInfo : Fin numPeople → Nat

/-- Represents the minimum time needed for complete information sharing -/
def minTimeForCompleteSharing (group : InformationSharingGroup) : Nat :=
  sorry

/-- Theorem stating the minimum time for the specific problem -/
theorem min_time_for_eight_people
  (group : InformationSharingGroup)
  (h1 : group.numPeople = 8)
  (h2 : group.callDuration = 3)
  (h3 : ∀ i j : Fin group.numPeople, i ≠ j → group.initialInfo i ≠ group.initialInfo j) :
  minTimeForCompleteSharing group = 9 :=
sorry

end min_time_for_eight_people_l1552_155255


namespace tangent_line_m_value_l1552_155222

-- Define the curve
def curve (x m n : ℝ) : ℝ := x^3 + m*x + n

-- Define the line
def line (x : ℝ) : ℝ := 3*x + 1

-- State the theorem
theorem tangent_line_m_value :
  ∀ (m n : ℝ),
  (curve 1 m n = 4) →  -- The point (1, 4) lies on the curve
  (line 1 = 4) →       -- The point (1, 4) lies on the line
  (∀ x : ℝ, curve x m n ≤ line x) →  -- The line is tangent to the curve (no other intersection)
  (m = 0) := by
sorry

end tangent_line_m_value_l1552_155222


namespace shifted_quadratic_sum_l1552_155211

/-- The sum of coefficients after shifting a quadratic function -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 2)^2 + 2 * (x + 2) + 4 = a * x^2 + b * x + c) → 
  a + b + c = 37 := by
sorry

end shifted_quadratic_sum_l1552_155211


namespace isosceles_triangle_height_l1552_155230

theorem isosceles_triangle_height (s : ℝ) (h : ℝ) : 
  (1/2 : ℝ) * s * h = 2 * s^2 → h = 4 * s := by
  sorry

end isosceles_triangle_height_l1552_155230


namespace complex_sum_squared_l1552_155276

noncomputable def i : ℂ := Complex.I

theorem complex_sum_squared 
  (a b c : ℂ) 
  (eq1 : a^2 + a*b + b^2 = 1 + i)
  (eq2 : b^2 + b*c + c^2 = -2)
  (eq3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*i) / 3 := by
sorry

end complex_sum_squared_l1552_155276


namespace sprint_medal_awards_l1552_155285

/-- The number of ways to award medals in the international sprinting event -/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) 
  (americans_winning : ℕ) : ℕ :=
  -- The actual calculation would go here
  216

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem sprint_medal_awards : 
  medal_award_ways 10 4 3 2 = 216 := by
  sorry

end sprint_medal_awards_l1552_155285


namespace point_in_region_l1552_155284

theorem point_in_region (m : ℝ) :
  (2 * m + 3 < 4) → m < 1/2 := by
  sorry

end point_in_region_l1552_155284


namespace probability_largest_smaller_theorem_l1552_155252

/-- The probability that the largest number in each row is smaller than the largest number in each row with more numbers, given n rows arranged as described. -/
def probability_largest_smaller (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating the probability for the arrangement of numbers in rows. -/
theorem probability_largest_smaller_theorem (n : ℕ) :
  let total_numbers := n * (n + 1) / 2
  let row_sizes := List.range n.succ
  probability_largest_smaller n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by
  sorry

end probability_largest_smaller_theorem_l1552_155252


namespace sin_2a_value_l1552_155246

theorem sin_2a_value (a : Real) (h1 : a ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin (π / 4 - a)) : 
  Real.sin (2 * a) = -8 / 9 := by
  sorry

end sin_2a_value_l1552_155246


namespace minimum_dice_value_l1552_155279

theorem minimum_dice_value (X : ℕ) : (1 + 5 + X > 2 + 4 + 5) ↔ X ≥ 6 :=
by sorry

end minimum_dice_value_l1552_155279


namespace constant_term_expansion_l1552_155283

/-- The constant term in the expansion of (x - 1/(2x))^6 is -5/2 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (x - 1/(2*x))^6
  ∃ (c : ℝ), (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = -5/2 :=
sorry

end constant_term_expansion_l1552_155283


namespace equation_solution_l1552_155299

theorem equation_solution (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / 
  ((x - 4) * (x - 2) * (x - 1)) = 1 →
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 := by
sorry

end equation_solution_l1552_155299


namespace fraction_zero_implies_x_neg_two_l1552_155227

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (abs x - 2) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end fraction_zero_implies_x_neg_two_l1552_155227


namespace base_ten_proof_l1552_155289

/-- Given that in base b, the square of 31_b is 1021_b, prove that b = 10 -/
theorem base_ten_proof (b : ℕ) (h : b > 3) : 
  (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end base_ten_proof_l1552_155289


namespace equivalent_discount_l1552_155296

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  first_discount = 0.2 →
  second_discount = 0.25 →
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * (1 - 0.4) :=
by
  sorry

end equivalent_discount_l1552_155296


namespace min_balls_for_fifteen_in_box_l1552_155248

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least 15 of a single color -/
def minBallsForFifteen (counts : BallCounts) : Nat :=
  sorry

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_for_fifteen_in_box :
  let counts : BallCounts := {
    red := 28, green := 20, yellow := 19,
    blue := 13, white := 11, black := 9
  }
  minBallsForFifteen counts = 76 := by sorry

end min_balls_for_fifteen_in_box_l1552_155248


namespace cubic_roots_sum_of_cubes_reciprocals_l1552_155221

theorem cubic_roots_sum_of_cubes_reciprocals 
  (a b c d : ℂ) 
  (r s t : ℂ) 
  (h₁ : a ≠ 0) 
  (h₂ : d ≠ 0) 
  (h₃ : a * r^3 + b * r^2 + c * r + d = 0) 
  (h₄ : a * s^3 + b * s^2 + c * s + d = 0) 
  (h₅ : a * t^3 + b * t^2 + c * t + d = 0) : 
  1 / r^3 + 1 / s^3 + 1 / t^3 = c^3 / d^3 := by
  sorry

end cubic_roots_sum_of_cubes_reciprocals_l1552_155221


namespace school_rewards_problem_l1552_155282

/-- The price of a practical backpack -/
def backpack_price : ℝ := 60

/-- The price of a multi-functional pencil case -/
def pencil_case_price : ℝ := 40

/-- The total budget for purchases -/
def total_budget : ℝ := 1140

/-- The total number of items to be purchased -/
def total_items : ℕ := 25

/-- The maximum number of backpacks that can be purchased -/
def max_backpacks : ℕ := 7

theorem school_rewards_problem :
  (3 * backpack_price + 2 * pencil_case_price = 260) ∧
  (5 * backpack_price + 4 * pencil_case_price = 460) ∧
  (∀ m : ℕ, m ≤ total_items → 
    backpack_price * m + pencil_case_price * (total_items - m) ≤ total_budget) →
  max_backpacks = 7 := by sorry

end school_rewards_problem_l1552_155282


namespace marked_circle_triangles_l1552_155245

/-- A circle with n equally spaced points on its circumference -/
structure MarkedCircle (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Number of triangles that can be formed with n points -/
def num_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of equilateral triangles that can be formed with n points -/
def num_equilateral_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of right triangles that can be formed with n points -/
def num_right_triangles (c : MarkedCircle n) : ℕ := sorry

theorem marked_circle_triangles 
  (c4 : MarkedCircle 4) 
  (c5 : MarkedCircle 5) 
  (c6 : MarkedCircle 6) : 
  (num_triangles c4 = 4) ∧ 
  (num_equilateral_triangles c5 = 0) ∧ 
  (num_right_triangles c6 = 12) := by sorry

end marked_circle_triangles_l1552_155245


namespace small_plate_diameter_l1552_155204

theorem small_plate_diameter
  (big_plate_diameter : ℝ)
  (uncovered_fraction : ℝ)
  (h1 : big_plate_diameter = 12)
  (h2 : uncovered_fraction = 0.3055555555555555) :
  ∃ (small_plate_diameter : ℝ),
    small_plate_diameter = 10 ∧
    (1 - uncovered_fraction) * (π * big_plate_diameter^2 / 4) = π * small_plate_diameter^2 / 4 :=
by sorry

end small_plate_diameter_l1552_155204


namespace extreme_values_of_f_range_of_a_for_f_greater_than_g_l1552_155242

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - a * x + (a + 3) / x
def g (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 4 * x + 2 * a

-- Theorem for part 1
theorem extreme_values_of_f (x : ℝ) (hx : x > 0) :
  let f_half := f (1/2)
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ y, y > 0 → f_half y ≥ f_half x_min) ∧
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ y, y > 0 → f_half y ≤ f_half x_max) ∧
  (∀ y, y > 0 → f_half y ≥ 3) ∧
  (∀ y, y > 0 → f_half y ≤ 4 * Real.log 7 - 3) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_f_greater_than_g (a : ℝ) (ha : a ≥ 1) :
  (∃ (x₁ x₂ : ℝ), 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 ∧ f a x₁ > g a x₂) ↔
  (1 ≤ a ∧ a < 4) :=
sorry

end extreme_values_of_f_range_of_a_for_f_greater_than_g_l1552_155242


namespace sports_enjoyment_misreporting_l1552_155234

theorem sports_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let not_enjoy := 0.3 * total
  let enjoy_say_enjoy := 0.75 * enjoy
  let enjoy_say_not := 0.25 * enjoy
  let not_enjoy_say_not := 0.85 * not_enjoy
  let not_enjoy_say_enjoy := 0.15 * not_enjoy
  enjoy_say_not / (enjoy_say_not + not_enjoy_say_not) = 7 / 17 := by
sorry

end sports_enjoyment_misreporting_l1552_155234


namespace prob_two_green_balls_l1552_155280

/-- The probability of drawing two green balls from a bag containing two green balls and one red ball when two balls are randomly drawn. -/
theorem prob_two_green_balls (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ) : 
  total_balls = 3 → 
  green_balls = 2 → 
  red_balls = 1 → 
  (green_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1/3 := by
sorry

end prob_two_green_balls_l1552_155280


namespace hash_2_3_1_5_equals_6_l1552_155225

/-- The # operation for real numbers -/
def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that #(2, 3, 1, 5) = 6 -/
theorem hash_2_3_1_5_equals_6 : hash 2 3 1 5 = 6 := by
  sorry

end hash_2_3_1_5_equals_6_l1552_155225


namespace parabola_vertex_l1552_155281

/-- The parabola is defined by the equation y = 2(x-3)^2 + 1 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola has coordinates (3, 1) -/
theorem parabola_vertex : ∃ (x y : ℝ), parabola x y ∧ x = 3 ∧ y = 1 := by sorry

end parabola_vertex_l1552_155281


namespace investment_ratio_l1552_155247

theorem investment_ratio (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  (p * 10) / (q * 20) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end investment_ratio_l1552_155247


namespace angle_sum_bound_l1552_155237

theorem angle_sum_bound (A B : Real) (h_triangle : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_inequality : ∀ x > 0, (Real.sin B / Real.cos A)^x + (Real.sin A / Real.cos B)^x < 2) :
  0 < A + B ∧ A + B < π/2 := by sorry

end angle_sum_bound_l1552_155237


namespace chord_AB_equation_tangent_circle_equation_l1552_155288

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define point M
def M : ℝ × ℝ := (1, 4)

-- Define the chord AB passing through M as its midpoint
def chord_AB (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    (x, y) = ((x1 + x2)/2, (y1 + y2)/2) ∧
    (x, y) = M

-- Define the tangent line l
def tangent_line (x0 y0 b : ℝ) : Prop :=
  E x0 y0 ∧ y0 = x0 + b

-- Theorem for the equation of line AB
theorem chord_AB_equation :
  ∀ x y : ℝ, chord_AB x y → x - 2*y + 7 = 0 := sorry

-- Theorem for the equation of the circle
theorem tangent_circle_equation :
  ∀ x0 y0 b : ℝ,
    tangent_line x0 y0 b →
    ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 4 := sorry

end chord_AB_equation_tangent_circle_equation_l1552_155288


namespace at_least_one_not_less_than_two_l1552_155200

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l1552_155200


namespace min_perimeter_triangle_l1552_155293

theorem min_perimeter_triangle (a b x : ℕ) (h1 : a = 40) (h2 : b = 50) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, y ≠ x → a + b + y > a + b + x) → 
  a + b + x = 101 := by
sorry

end min_perimeter_triangle_l1552_155293


namespace doritos_ratio_l1552_155216

theorem doritos_ratio (total_bags : ℕ) (doritos_piles : ℕ) (bags_per_pile : ℕ) 
  (h1 : total_bags = 80)
  (h2 : doritos_piles = 4)
  (h3 : bags_per_pile = 5) : 
  (doritos_piles * bags_per_pile : ℚ) / total_bags = 1 / 4 := by
  sorry

end doritos_ratio_l1552_155216


namespace tan_alpha_plus_pi_fourth_l1552_155273

/-- Given vectors a and b, where a is parallel to b, prove that tan(α + π/4) = 3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (-2, Real.cos α))
  (hb : b = (-1, Real.sin α))
  (parallel : ∃ (k : ℝ), a = k • b) :
  Real.tan (α + π/4) = 3 := by
  sorry

end tan_alpha_plus_pi_fourth_l1552_155273


namespace decimal_equals_fraction_l1552_155263

/-- The decimal representation of 0.1⁻35 as a real number -/
def decimal_rep : ℚ := 0.1 + (35 / 990)

/-- The fraction 67/495 as a rational number -/
def fraction : ℚ := 67 / 495

/-- Assertion that 67 and 495 are coprime -/
axiom coprime_67_495 : Nat.Coprime 67 495

theorem decimal_equals_fraction : decimal_rep = fraction := by sorry

end decimal_equals_fraction_l1552_155263


namespace cone_volume_from_half_sector_l1552_155228

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := r * π
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
sorry

end cone_volume_from_half_sector_l1552_155228


namespace intersection_area_is_4pi_l1552_155260

-- Define the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(2, 3), (2, 15), (13, 3), (13, 15)]

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 13)^2 + (y - 3)^2 = 16

-- Define the area of intersection
def intersection_area (rect : List (ℝ × ℝ)) (circle : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem intersection_area_is_4pi :
  intersection_area rectangle_vertices circle_equation = 4 * Real.pi := by sorry

end intersection_area_is_4pi_l1552_155260


namespace diego_paycheck_l1552_155210

/-- Diego's monthly paycheck problem -/
theorem diego_paycheck (monthly_expenses : ℝ) (annual_savings : ℝ) (h1 : monthly_expenses = 4600) (h2 : annual_savings = 4800) :
  monthly_expenses + annual_savings / 12 = 5000 := by
  sorry

end diego_paycheck_l1552_155210


namespace vector_operation_result_l1552_155258

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_result :
  vector_operation = (-3, 32) := by sorry

end vector_operation_result_l1552_155258


namespace probability_four_twos_in_five_rolls_l1552_155207

theorem probability_four_twos_in_five_rolls : 
  let n_rolls : ℕ := 5
  let n_sides : ℕ := 6
  let n_twos : ℕ := 4
  let p_two : ℚ := 1 / n_sides
  let p_not_two : ℚ := 1 - p_two
  Nat.choose n_rolls n_twos * p_two ^ n_twos * p_not_two ^ (n_rolls - n_twos) = 3125 / 7776 := by
  sorry

end probability_four_twos_in_five_rolls_l1552_155207


namespace bernardo_always_less_than_silvia_l1552_155272

def bernardo_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def silvia_set : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def bernardo_number (a b c : ℕ) : ℕ := 
  100 * min a (min b c) + 10 * (a + b + c - max a (max b c) - min a (min b c)) + max a (max b c)

def silvia_number (x y z : ℕ) : ℕ := 
  100 * max x (max y z) + 10 * (x + y + z - max x (max y z) - min x (min y z)) + min x (min y z)

theorem bernardo_always_less_than_silvia :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a ∈ bernardo_set → b ∈ bernardo_set → c ∈ bernardo_set →
    x ∈ silvia_set → y ∈ silvia_set → z ∈ silvia_set →
    a ≠ b → b ≠ c → a ≠ c →
    x ≠ y → y ≠ z → x ≠ z →
    bernardo_number a b c < silvia_number x y z := by
  sorry

end bernardo_always_less_than_silvia_l1552_155272


namespace discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l1552_155233

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ := a^2 - 4*(a - 1)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (a : ℝ) : discriminant a ≥ 0 := by
  sorry

-- Theorem 2: When one root is greater than 3, a > 4
theorem root_greater_than_three_implies_a_greater_than_four (a : ℝ) :
  (∃ x, quadratic a x = 0 ∧ x > 3) → a > 4 := by
  sorry

end discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l1552_155233


namespace checkerboard_covering_l1552_155209

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  removed_squares : Fin (4 * size * size) × Fin (4 * size * size)

/-- Represents a 2 × 1 domino -/
structure Domino

/-- Predicate to check if two squares are of opposite colors -/
def opposite_colors (c : Checkerboard) (s1 s2 : Fin (4 * c.size * c.size)) : Prop :=
  (s1.val + s2.val) % 2 = 1

/-- Predicate to check if a checkerboard can be covered by dominoes -/
def can_cover (c : Checkerboard) : Prop :=
  ∃ (covering : List (Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size))),
    (∀ (square : Fin (4 * c.size * c.size)), 
      square ≠ c.removed_squares.1 ∧ square ≠ c.removed_squares.2 → 
      ∃ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
        domino ∈ covering ∧ (square = domino.1 ∨ square = domino.2)) ∧
    (∀ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
      domino ∈ covering → 
      (domino.1 ≠ c.removed_squares.1 ∧ domino.1 ≠ c.removed_squares.2) ∧
      (domino.2 ≠ c.removed_squares.1 ∧ domino.2 ≠ c.removed_squares.2) ∧
      (domino.1.val + 1 = domino.2.val ∨ domino.1.val + 2 * c.size = domino.2.val))

/-- Theorem stating that any 2k × 2k checkerboard with two squares of opposite colors removed can be covered by 2 × 1 dominoes -/
theorem checkerboard_covering (k : ℕ) (c : Checkerboard) 
  (h_size : c.size = 2 * k)
  (h_opposite : opposite_colors c c.removed_squares.1 c.removed_squares.2) :
  can_cover c :=
sorry

end checkerboard_covering_l1552_155209


namespace reflection_xoz_coordinates_l1552_155223

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the xOz plane -/
def reflectXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem reflection_xoz_coordinates :
  let P : Point3D := { x := 3, y := -2, z := 1 }
  reflectXOZ P = { x := 3, y := 2, z := 1 } := by
  sorry

end reflection_xoz_coordinates_l1552_155223


namespace empty_cube_exists_l1552_155212

/-- Represents a 3D coordinate within the cube -/
structure Coord where
  x : Fin 5
  y : Fin 5
  z : Fin 5

/-- Represents the state of a unit cube -/
inductive CubeState
  | Occupied
  | Empty

/-- The type of the cube, mapping coordinates to cube states -/
def Cube := Coord → CubeState

/-- Checks if two coordinates are adjacent -/
def isAdjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- A function representing the movement of objects -/
def moveObjects (initial : Cube) : Cube :=
  sorry

theorem empty_cube_exists (initial : Cube) :
  (∀ c : Coord, initial c = CubeState.Occupied) →
  ∃ c : Coord, (moveObjects initial) c = CubeState.Empty :=
sorry

end empty_cube_exists_l1552_155212


namespace no_perfect_square_131_base_n_l1552_155269

theorem no_perfect_square_131_base_n : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 12 ∧ ∃ (k : ℤ), n^2 + 3*n + 1 = k^2 := by
  sorry

end no_perfect_square_131_base_n_l1552_155269


namespace not_linear_in_M_exp_in_M_sin_in_M_iff_l1552_155229

/-- The set M of functions satisfying f(x+T) = T⋅f(x) for some non-zero T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x}

theorem not_linear_in_M :
    ∀ T : ℝ, T ≠ 0 → ∃ x : ℝ, x + T ≠ T * x := by sorry

theorem exp_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
    (∃ T : ℝ, T > 0 ∧ a^T = T) → (fun x ↦ a^x) ∈ M := by sorry

theorem sin_in_M_iff (k : ℝ) :
    (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end not_linear_in_M_exp_in_M_sin_in_M_iff_l1552_155229


namespace trailing_zeros_of_square_l1552_155253

/-- The number of trailing zeros in (10^10 - 2)^2 is 17 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^10 - 2)^2 = n * 10^17 ∧ n % 10 ≠ 0 := by
  sorry

end trailing_zeros_of_square_l1552_155253


namespace inverse_as_linear_combination_l1552_155241

def N : Matrix (Fin 2) (Fin 2) ℚ := !![4, 0; 2, -6]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/24 ∧ d = 1/12 := by
  sorry

end inverse_as_linear_combination_l1552_155241


namespace six_regions_three_colors_l1552_155218

/-- The number of ways to color n regions using k colors --/
def colorings (n k : ℕ) : ℕ := k^n

/-- The number of ways to color n regions using exactly k colors --/
def exactColorings (n k : ℕ) : ℕ :=
  (Nat.choose k k) * k^n - (Nat.choose k (k-1)) * (k-1)^n + (Nat.choose k (k-2)) * (k-2)^n

theorem six_regions_three_colors :
  exactColorings 6 3 = 540 := by sorry

end six_regions_three_colors_l1552_155218


namespace equation_one_real_root_l1552_155271

theorem equation_one_real_root :
  ∃! x : ℝ, (Real.sqrt (x^2 + 2*x - 63) + Real.sqrt (x + 9) - Real.sqrt (7 - x) + x + 13 = 0) ∧
             (x^2 + 2*x - 63 ≥ 0) ∧
             (x + 9 ≥ 0) ∧
             (7 - x ≥ 0) := by
  sorry

end equation_one_real_root_l1552_155271


namespace positive_difference_theorem_l1552_155244

theorem positive_difference_theorem : ∃ (x : ℝ), x > 0 ∧ x = |((8^2 * 8^2) / 8) - ((8^2 + 8^2) / 8)| ∧ x = 496 := by
  sorry

end positive_difference_theorem_l1552_155244


namespace unique_solution_for_equation_l1552_155297

theorem unique_solution_for_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end unique_solution_for_equation_l1552_155297


namespace parabola_shift_theorem_l1552_155240

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x - h)^2 + k -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally and vertically -/
def Parabola.shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

/-- The original parabola y = -2x^2 -/
def original_parabola : Parabola :=
  { a := -2, h := 0, k := 0 }

/-- Theorem stating that shifting the original parabola down 1 unit and right 3 units
    results in the equation y = -2(x - 3)^2 - 1 -/
theorem parabola_shift_theorem (x : ℝ) :
  (original_parabola.shift 3 (-1)).equation x = -2 * (x - 3)^2 - 1 := by
  sorry

end parabola_shift_theorem_l1552_155240


namespace discount_percentage_l1552_155267

theorem discount_percentage (cupcake_price cookie_price : ℝ) 
  (cupcakes_sold cookies_sold : ℕ) (total_revenue : ℝ) :
  cupcake_price = 3 →
  cookie_price = 2 →
  cupcakes_sold = 16 →
  cookies_sold = 8 →
  total_revenue = 32 →
  ∃ (x : ℝ), 
    (cupcakes_sold : ℝ) * (cupcake_price * (100 - x) / 100) + 
    (cookies_sold : ℝ) * (cookie_price * (100 - x) / 100) = total_revenue ∧
    x = 50 := by
  sorry

end discount_percentage_l1552_155267


namespace first_question_percentage_l1552_155254

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 49)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 32)
  : ∃ first_correct : ℝ, first_correct = 63 := by
  sorry

end first_question_percentage_l1552_155254


namespace train_length_proof_l1552_155232

/-- Given a train that passes a pole in 10 seconds and a 1250m long platform in 60 seconds,
    prove that the length of the train is 250 meters. -/
theorem train_length_proof (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 10)
    (h2 : platform_time = 60)
    (h3 : platform_length = 1250) : 
  ∃ (train_length : ℝ), train_length = 250 ∧ 
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

end train_length_proof_l1552_155232


namespace one_fifths_in_nine_fifths_l1552_155261

theorem one_fifths_in_nine_fifths : (9 : ℚ) / 5 / (1 / 5) = 9 := by
  sorry

end one_fifths_in_nine_fifths_l1552_155261


namespace remainder_scaling_l1552_155262

theorem remainder_scaling (a b c r : ℤ) : 
  (a = b * c + r) → (0 ≤ r) → (r < b) → 
  ∃ (q : ℤ), (3 * a = 3 * b * q + 3 * r) ∧ (0 ≤ 3 * r) ∧ (3 * r < 3 * b) :=
by sorry

end remainder_scaling_l1552_155262


namespace alice_minimum_score_l1552_155250

def minimum_score (scores : List Float) (target_average : Float) (total_terms : Nat) : Float :=
  let sum_scores := scores.sum
  let remaining_terms := total_terms - scores.length
  (target_average * total_terms.toFloat - sum_scores) / remaining_terms.toFloat

theorem alice_minimum_score :
  let alice_scores := [84, 88, 82, 79]
  let target_average := 85
  let total_terms := 5
  minimum_score alice_scores target_average total_terms = 92 := by
  sorry

end alice_minimum_score_l1552_155250


namespace polygon_sides_from_angle_sum_l1552_155206

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 1080 → (n - 2) * 180 = angle_sum → n = 8 := by
  sorry

end polygon_sides_from_angle_sum_l1552_155206


namespace rectangle_division_condition_l1552_155226

/-- A rectangle can be divided into two unequal but similar rectangles if and only if its longer side is more than twice the length of its shorter side. -/
theorem rectangle_division_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b) :
  (∃ x : ℝ, 0 < x ∧ x < a ∧ x * b = (a - x) * x) ↔ a > 2 * b :=
sorry

end rectangle_division_condition_l1552_155226


namespace initial_apples_l1552_155217

theorem initial_apples (minseok_ate jaeyoon_ate apples_left : ℕ) 
  (h1 : minseok_ate = 3)
  (h2 : jaeyoon_ate = 3)
  (h3 : apples_left = 2) : 
  minseok_ate + jaeyoon_ate + apples_left = 8 :=
by sorry

end initial_apples_l1552_155217


namespace total_painting_time_l1552_155201

/-- Time to paint each type of flower in minutes -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def sunflower_time : ℕ := 10
def tulip_time : ℕ := 4
def vine_time : ℕ := 2
def peony_time : ℕ := 8

/-- Number of each type of flower to paint -/
def lily_count : ℕ := 23
def rose_count : ℕ := 15
def orchid_count : ℕ := 9
def sunflower_count : ℕ := 12
def tulip_count : ℕ := 18
def vine_count : ℕ := 30
def peony_count : ℕ := 27

/-- Theorem stating the total time to paint all flowers -/
theorem total_painting_time : 
  lily_time * lily_count + 
  rose_time * rose_count + 
  orchid_time * orchid_count + 
  sunflower_time * sunflower_count + 
  tulip_time * tulip_count + 
  vine_time * vine_count + 
  peony_time * peony_count = 715 := by
  sorry

end total_painting_time_l1552_155201


namespace smallest_n_for_candy_l1552_155274

theorem smallest_n_for_candy (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ 15 * m % 10 = 0 ∧ 15 * m % 18 = 0 ∧ 15 * m % 25 = 0 → m ≥ n) ∧
  n > 0 ∧ 15 * n % 10 = 0 ∧ 15 * n % 18 = 0 ∧ 15 * n % 25 = 0 →
  n = 30 :=
by sorry

end smallest_n_for_candy_l1552_155274
