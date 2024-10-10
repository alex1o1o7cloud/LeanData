import Mathlib

namespace max_abs_sum_on_circle_l101_10129

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) : 
  |x| + |y| ≤ 2 ∧ ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ |a| + |b| = 2 := by
  sorry

end max_abs_sum_on_circle_l101_10129


namespace symmetric_point_correct_l101_10145

-- Define the original point
def original_point : ℝ × ℝ := (-1, 1)

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric point
def symmetric_point : ℝ × ℝ := (2, -2)

-- Theorem statement
theorem symmetric_point_correct : 
  let (x₁, y₁) := original_point
  let (x₂, y₂) := symmetric_point
  (line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧ 
   (x₂ - x₁) = (y₂ - y₁) ∧
   (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4 * ((x₁ - (x₁ + x₂) / 2)^2 + (y₁ - (y₁ + y₂) / 2)^2)) :=
by sorry

end symmetric_point_correct_l101_10145


namespace shortest_distance_correct_l101_10151

/-- Represents the lengths of six lines meeting at a point -/
structure SixLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0

/-- The shortest distance to draw all lines without lifting the pencil -/
def shortestDistance (lines : SixLines) : ℝ :=
  lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f)

/-- Theorem stating that the shortest distance formula is correct -/
theorem shortest_distance_correct (lines : SixLines) :
  shortestDistance lines = lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f) :=
by sorry

end shortest_distance_correct_l101_10151


namespace same_solution_implies_c_equals_four_l101_10112

theorem same_solution_implies_c_equals_four :
  ∀ x c : ℝ,
  (3 * x + 9 = 0) →
  (c * x + 15 = 3) →
  c = 4 :=
by
  sorry

end same_solution_implies_c_equals_four_l101_10112


namespace function_value_at_negative_two_l101_10114

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1 where f(2) = -1, prove that f(-2) = 3 -/
theorem function_value_at_negative_two 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 1)
  (h2 : f 2 = -1) : 
  f (-2) = 3 := by
sorry

end function_value_at_negative_two_l101_10114


namespace point_not_in_second_quadrant_l101_10150

theorem point_not_in_second_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (m + 1, m)
  ¬ (P.1 < 0 ∧ P.2 > 0) :=
by sorry

end point_not_in_second_quadrant_l101_10150


namespace min_pencils_for_ten_correct_l101_10193

/-- Represents the number of pencils of each color in the drawer -/
structure PencilDrawer :=
  (orange : ℕ)
  (purple : ℕ)
  (grey : ℕ)
  (cyan : ℕ)
  (violet : ℕ)

/-- The minimum number of pencils to ensure at least 10 of one color -/
def minPencilsForTen (drawer : PencilDrawer) : ℕ := 43

/-- Theorem stating the minimum number of pencils needed -/
theorem min_pencils_for_ten_correct (drawer : PencilDrawer) 
  (h1 : drawer.orange = 26)
  (h2 : drawer.purple = 22)
  (h3 : drawer.grey = 18)
  (h4 : drawer.cyan = 15)
  (h5 : drawer.violet = 10) :
  minPencilsForTen drawer = 43 ∧
  ∀ n : ℕ, n < 43 → ¬(∃ color : ℕ, color ≥ 10 ∧ 
    (color ≤ drawer.orange ∨ 
     color ≤ drawer.purple ∨ 
     color ≤ drawer.grey ∨ 
     color ≤ drawer.cyan ∨ 
     color ≤ drawer.violet)) := by
  sorry

end min_pencils_for_ten_correct_l101_10193


namespace birds_on_fence_proof_l101_10143

/-- Given an initial number of birds and the number of birds remaining,
    calculate the number of birds that flew away. -/
def birds_flew_away (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem birds_on_fence_proof :
  let initial_birds : ℝ := 12.0
  let remaining_birds : ℕ := 4
  birds_flew_away initial_birds remaining_birds = 8 := by
  sorry

end birds_on_fence_proof_l101_10143


namespace angle_A_measure_perimeter_range_l101_10139

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 - 2*t.b*t.c*(Real.cos t.A) = (t.b + t.c)^2

/-- Theorem stating the measure of angle A -/
theorem angle_A_measure (t : Triangle) (h : satisfiesCondition t) : t.A = 2*π/3 := by
  sorry

/-- Theorem stating the range of the perimeter when a = 3 -/
theorem perimeter_range (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 3) :
  6 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2*Real.sqrt 3 + 3 := by
  sorry

end angle_A_measure_perimeter_range_l101_10139


namespace phillips_apples_l101_10134

theorem phillips_apples (ben phillip tom : ℕ) : 
  ben = phillip + 8 →
  3 * ben = 8 * tom →
  tom = 18 →
  phillip = 40 := by
sorry

end phillips_apples_l101_10134


namespace circle_center_is_two_two_l101_10108

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 10 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 18

theorem circle_center_is_two_two :
  CircleCenter 2 2 := by sorry

end circle_center_is_two_two_l101_10108


namespace logarithmic_equation_solution_l101_10104

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 3 ∧ lg (x - 3) + lg x = 1 :=
by
  -- The proof would go here
  sorry

end logarithmic_equation_solution_l101_10104


namespace basketball_shooting_improvement_l101_10109

theorem basketball_shooting_improvement (initial_shots : ℕ) (initial_made : ℕ) (next_game_shots : ℕ) (new_average : ℚ) : 
  initial_shots = 35 → 
  initial_made = 15 → 
  next_game_shots = 15 → 
  new_average = 11/20 → 
  ∃ (next_game_made : ℕ), 
    next_game_made = 13 ∧ 
    (initial_made + next_game_made : ℚ) / (initial_shots + next_game_shots : ℚ) = new_average :=
by sorry

#check basketball_shooting_improvement

end basketball_shooting_improvement_l101_10109


namespace distance_to_walk_back_l101_10187

/-- Represents the distance traveled by Vintik and Shpuntik -/
def TravelDistance (x y : ℝ) : Prop :=
  -- Vintik's total distance is 12 km
  2 * x + y = 12 ∧
  -- Total fuel consumption is 75 liters
  3 * x + 15 * y = 75 ∧
  -- x represents half of Vintik's forward distance
  x > 0 ∧ y > 0

/-- The theorem stating the distance to walk back home -/
theorem distance_to_walk_back (x y : ℝ) (h : TravelDistance x y) : 
  3 * x - 3 * y = 9 :=
sorry

end distance_to_walk_back_l101_10187


namespace common_difference_is_four_l101_10159

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_correct : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 1 - seq.a 0

theorem common_difference_is_four (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 12) : 
  common_difference seq = 4 := by
  sorry

end common_difference_is_four_l101_10159


namespace arithmetic_to_geometric_sequence_l101_10123

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2*d ∧ 
  a₄ = a₁ + 3*d ∧
  ((a₁*a₃ = a₂^2) ∨ (a₁*a₄ = a₂^2) ∨ (a₁*a₄ = a₃^2) ∨ (a₂*a₄ = a₃^2)) →
  a₁/d = 1 ∨ a₁/d = -4 := by
sorry

end arithmetic_to_geometric_sequence_l101_10123


namespace night_temperature_l101_10167

def noon_temperature : Int := -2
def temperature_drop : Int := 4

theorem night_temperature : 
  noon_temperature - temperature_drop = -6 := by sorry

end night_temperature_l101_10167


namespace expression_equals_one_l101_10157

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ -1) (h2 : x^3 ≠ 1) : 
  ((x+1)^3 * (x^2-x+1)^3 / (x^3+1)^3)^2 * ((x-1)^3 * (x^2+x+1)^3 / (x^3-1)^3)^2 = 1 :=
by sorry

end expression_equals_one_l101_10157


namespace total_triangles_l101_10194

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  small_triangles : ℕ

/-- Counts the total number of triangles in a divided triangle -/
def count_triangles (t : DividedTriangle) : ℕ :=
  t.small_triangles + (t.small_triangles - 1) + 1

/-- The problem setup -/
def triangle_problem : Prop :=
  ∃ (t1 t2 : DividedTriangle),
    t1.small_triangles = 3 ∧
    t2.small_triangles = 3 ∧
    count_triangles t1 + count_triangles t2 = 13

/-- The theorem to prove -/
theorem total_triangles : triangle_problem := by
  sorry

end total_triangles_l101_10194


namespace joe_running_speed_l101_10100

/-- Proves that Joe's running speed is 16 km/h given the problem conditions -/
theorem joe_running_speed (pete_speed : ℝ) : 
  pete_speed > 0 →
  (2 * pete_speed * (40 / 60) + pete_speed * (40 / 60) = 16) →
  2 * pete_speed = 16 := by
  sorry

#check joe_running_speed

end joe_running_speed_l101_10100


namespace new_profit_percentage_after_doubling_price_l101_10175

-- Define the initial profit percentage
def initial_profit_percentage : ℝ := 30

-- Define the price multiplier for the new selling price
def price_multiplier : ℝ := 2

-- Theorem to prove
theorem new_profit_percentage_after_doubling_price :
  let original_selling_price := 100 + initial_profit_percentage
  let new_selling_price := price_multiplier * original_selling_price
  let new_profit := new_selling_price - 100
  let new_profit_percentage := (new_profit / 100) * 100
  new_profit_percentage = 160 := by sorry

end new_profit_percentage_after_doubling_price_l101_10175


namespace angle_A_is_pi_div_6_max_area_when_a_is_2_l101_10171

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem 1: Prove that angle A = π/6
theorem angle_A_is_pi_div_6 (t : Triangle) (h : condition t) : t.A = π / 6 :=
sorry

-- Theorem 2: Prove that when a = 2, the maximum area of triangle ABC is 2 + √3
theorem max_area_when_a_is_2 (t : Triangle) (h1 : condition t) (h2 : t.a = 2) :
  (t.b * t.c * Real.sin t.A / 2) ≤ 2 + Real.sqrt 3 :=
sorry

end angle_A_is_pi_div_6_max_area_when_a_is_2_l101_10171


namespace gcd_5005_11011_l101_10147

theorem gcd_5005_11011 : Nat.gcd 5005 11011 = 1001 := by
  sorry

end gcd_5005_11011_l101_10147


namespace gathering_handshakes_l101_10107

/-- Represents a group of people at a gathering --/
structure Gathering where
  total : ℕ
  groupA : ℕ
  groupB : ℕ
  knownInA : ℕ
  hTotal : total = groupA + groupB
  hKnownInA : knownInA ≤ groupA

/-- Calculates the number of handshakes in the gathering --/
def handshakes (g : Gathering) : ℕ :=
  let handshakesBetweenGroups := g.groupB * (g.groupA - g.knownInA)
  let handshakesWithinB := g.groupB * (g.groupB - 1) / 2 - g.groupB * g.knownInA
  handshakesBetweenGroups + handshakesWithinB

/-- Theorem stating the number of handshakes in the specific gathering --/
theorem gathering_handshakes :
  ∃ (g : Gathering),
    g.total = 40 ∧
    g.groupA = 25 ∧
    g.groupB = 15 ∧
    g.knownInA = 5 ∧
    handshakes g = 330 := by
  sorry

end gathering_handshakes_l101_10107


namespace ratio_of_partial_fractions_l101_10164

theorem ratio_of_partial_fractions (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / (P : ℚ) = 15 / 11 := by
sorry

end ratio_of_partial_fractions_l101_10164


namespace number_problem_l101_10125

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end number_problem_l101_10125


namespace triangle_abc_properties_l101_10168

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 2 * Real.sqrt 6 / 3 →
  Real.sin A * Real.cos B = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  C = 2 * π / 3 ∧
  (1/2 * b * c * Real.sin A : Real) = (6 - 2 * Real.sqrt 3) / 9 :=
by sorry

end triangle_abc_properties_l101_10168


namespace intersection_of_A_and_B_l101_10122

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l101_10122


namespace tomatoes_eaten_by_birds_l101_10166

theorem tomatoes_eaten_by_birds 
  (initial_tomatoes : ℕ) 
  (remaining_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  initial_tomatoes - remaining_tomatoes = 7 := by
sorry

end tomatoes_eaten_by_birds_l101_10166


namespace polynomial_relationship_l101_10155

def f (x : ℝ) : ℝ := x^2 + x

theorem polynomial_relationship : 
  (f 1 = 2) ∧ 
  (f 2 = 6) ∧ 
  (f 3 = 12) ∧ 
  (f 4 = 20) ∧ 
  (f 5 = 30) := by
  sorry

end polynomial_relationship_l101_10155


namespace inequality_solution_l101_10188

theorem inequality_solution (a b : ℝ) :
  (∀ x, b - a * x > 0 ↔ 
    ((a > 0 ∧ x < b / a) ∨ 
     (a < 0 ∧ x > b / a) ∨ 
     (a = 0 ∧ False))) :=
by sorry

end inequality_solution_l101_10188


namespace only_square_relationship_functional_l101_10133

/-- Represents a relationship between two variables -/
structure Relationship where
  is_functional : Bool

/-- The relationship between the side length and the area of a square -/
def square_relationship : Relationship := sorry

/-- The relationship between rice yield and the amount of fertilizer applied -/
def rice_fertilizer_relationship : Relationship := sorry

/-- The relationship between snowfall and the rate of traffic accidents -/
def snowfall_accidents_relationship : Relationship := sorry

/-- The relationship between a person's height and weight -/
def height_weight_relationship : Relationship := sorry

/-- Theorem stating that only the square relationship is functional -/
theorem only_square_relationship_functional :
  square_relationship.is_functional ∧
  ¬rice_fertilizer_relationship.is_functional ∧
  ¬snowfall_accidents_relationship.is_functional ∧
  ¬height_weight_relationship.is_functional :=
by sorry

end only_square_relationship_functional_l101_10133


namespace battle_station_staffing_l101_10121

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.descFactorial k) = 30240 := by
  sorry

end battle_station_staffing_l101_10121


namespace nellie_legos_proof_l101_10192

/-- Calculates the remaining number of legos after losing some and giving some away. -/
def remaining_legos (initial : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial - lost - given

/-- Proves that given 380 initial legos, after losing 57 and giving away 24, 299 legos remain. -/
theorem nellie_legos_proof :
  remaining_legos 380 57 24 = 299 := by
  sorry

end nellie_legos_proof_l101_10192


namespace work_completion_rate_l101_10124

theorem work_completion_rate (a_days : ℕ) (b_days : ℕ) : 
  a_days = 8 → b_days = a_days / 2 → (1 : ℚ) / a_days + (1 : ℚ) / b_days = 3 / 8 := by
  sorry

#check work_completion_rate

end work_completion_rate_l101_10124


namespace max_leftover_pencils_l101_10178

theorem max_leftover_pencils :
  ∀ (n : ℕ), 
  ∃ (q : ℕ), 
  n = 7 * q + (n % 7) ∧ 
  n % 7 ≤ 6 ∧
  ∀ (r : ℕ), r > n % 7 → r > 6 := by
  sorry

end max_leftover_pencils_l101_10178


namespace sqrt_three_irrational_l101_10130

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l101_10130


namespace inner_triangle_perimeter_value_l101_10174

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Lines parallel to the sides of the triangle -/
structure ParallelLines :=
  (ℓD : ℝ)  -- Length of intersection with triangle interior
  (ℓE : ℝ)
  (ℓF : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_value :
  let t : Triangle := { DE := 150, EF := 250, FD := 200 }
  let p : ParallelLines := { ℓD := 65, ℓE := 55, ℓF := 25 }
  inner_triangle_perimeter t p = 990 :=
sorry

end inner_triangle_perimeter_value_l101_10174


namespace linear_function_through_origin_l101_10128

theorem linear_function_through_origin (k : ℝ) : 
  (∀ x y : ℝ, y = (k - 2) * x + (k^2 - 4)) →  -- Definition of the linear function
  ((0 : ℝ) = (k - 2) * (0 : ℝ) + (k^2 - 4)) →  -- The function passes through the origin
  (k - 2 ≠ 0) →  -- Ensure the function remains linear
  k = -2 := by sorry

end linear_function_through_origin_l101_10128


namespace parallel_vectors_sum_l101_10180

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

/-- Scalar multiplication of a vector -/
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem parallel_vectors_sum (y : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → vec_add a (vec_scalar_mul 2 b) = (5, 10) := by
  sorry

end parallel_vectors_sum_l101_10180


namespace bridge_length_bridge_length_specific_l101_10160

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- The bridge length is 195 meters given specific train parameters -/
theorem bridge_length_specific : bridge_length 180 45 30 = 195 := by
  sorry

end bridge_length_bridge_length_specific_l101_10160


namespace percentage_problem_l101_10103

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 := by
  sorry

end percentage_problem_l101_10103


namespace inverse_of_A_l101_10137

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; 5, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/22, 1/11; -5/22, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l101_10137


namespace wheat_field_and_fertilizer_l101_10179

theorem wheat_field_and_fertilizer 
  (field_size : ℕ) 
  (fertilizer_amount : ℕ) 
  (h1 : 6 * field_size = fertilizer_amount + 300)
  (h2 : 5 * field_size + 200 = fertilizer_amount) :
  field_size = 500 ∧ fertilizer_amount = 2700 := by
sorry

end wheat_field_and_fertilizer_l101_10179


namespace beth_class_students_left_l101_10110

/-- The number of students who left Beth's class in the final year -/
def students_left (initial : ℕ) (joined : ℕ) (final : ℕ) : ℕ :=
  initial + joined - final

theorem beth_class_students_left : 
  students_left 150 30 165 = 15 := by sorry

end beth_class_students_left_l101_10110


namespace triangle_problem_l101_10144

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (2 * b + Real.sqrt 3 * c) * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0 →
  A = 5 * π / 6 ∧
  (a = 2 → 
    ∃ (lower upper : ℝ), lower = 2 ∧ upper = 2 * Real.sqrt 3 ∧
    ∀ (x : ℝ), (∃ (b' c' : ℝ), b' + Real.sqrt 3 * c' = x ∧
      b' / (Real.sin B) = c' / (Real.sin C) ∧
      (2 * b' + Real.sqrt 3 * c') * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0) →
    lower < x ∧ x < upper) :=
by sorry

end triangle_problem_l101_10144


namespace butterfly_flight_l101_10195

theorem butterfly_flight (field_length field_width start_distance : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : start_distance = 6)
  (h4 : start_distance < field_length / 2) :
  let end_distance := field_length - 2 * start_distance
  let flight_distance := Real.sqrt (field_width ^ 2 + end_distance ^ 2)
  flight_distance = 17 := by sorry

end butterfly_flight_l101_10195


namespace expression_value_l101_10120

theorem expression_value (x y : ℝ) (h : 2 * x + y = 1) :
  (y + 1)^2 - (y^2 - 4 * x + 4) = -1 := by
  sorry

end expression_value_l101_10120


namespace m_value_l101_10172

/-- Triangle DEF with median DG to side EF -/
structure TriangleDEF where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  DG : ℝ
  is_median : DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DG * DG = 2 * (m * m)

/-- The value of m in the equation DG = m√2 for the given triangle -/
def find_m (t : TriangleDEF) : ℝ := sorry

/-- Theorem stating that m = √266 / 2 for the given triangle -/
theorem m_value (t : TriangleDEF) : find_m t = Real.sqrt 266 / 2 := by sorry

end m_value_l101_10172


namespace problem_proof_l101_10186

theorem problem_proof : (-24) * (1/3 - 5/6 + 3/8) = 3 := by
  sorry

end problem_proof_l101_10186


namespace wire_service_reporters_l101_10182

/-- The percentage of reporters who cover local politics in country x -/
def local_politics_percentage : ℝ := 35

/-- The percentage of reporters who cover politics but not local politics in country x -/
def non_local_politics_percentage : ℝ := 30

/-- The percentage of reporters who do not cover politics -/
def non_politics_percentage : ℝ := 50

theorem wire_service_reporters :
  local_politics_percentage = 35 →
  non_local_politics_percentage = 30 →
  non_politics_percentage = 50 := by
  sorry

end wire_service_reporters_l101_10182


namespace distinct_collections_l101_10127

def word : String := "PHYSICS"

def num_magnets : ℕ := 7

def vowels_fallen : ℕ := 3

def consonants_fallen : ℕ := 3

def s_indistinguishable : Prop := True

theorem distinct_collections : ℕ := by
  sorry

end distinct_collections_l101_10127


namespace correct_calculation_l101_10161

theorem correct_calculation (a : ℝ) : -3*a - 2*a = -5*a := by
  sorry

end correct_calculation_l101_10161


namespace functions_for_12_functions_for_2007_functions_for_2_pow_2007_l101_10162

-- Define the functions
def φ : ℕ → ℕ := sorry
def σ : ℕ → ℕ := sorry
def τ : ℕ → ℕ := sorry

-- Theorem for n = 12
theorem functions_for_12 :
  φ 12 = 4 ∧ σ 12 = 28 ∧ τ 12 = 6 := by sorry

-- Theorem for n = 2007
theorem functions_for_2007 :
  φ 2007 = 1332 ∧ σ 2007 = 2912 ∧ τ 2007 = 6 := by sorry

-- Theorem for n = 2^2007
theorem functions_for_2_pow_2007 :
  φ (2^2007) = 2^2006 ∧ 
  σ (2^2007) = 2^2008 - 1 ∧ 
  τ (2^2007) = 2008 := by sorry

end functions_for_12_functions_for_2007_functions_for_2_pow_2007_l101_10162


namespace new_student_weight_l101_10146

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_avg = 27.5 →
  (initial_count * initial_avg + (initial_count + 1) * new_avg - initial_count * initial_avg) / (initial_count + 1) = 13 := by
  sorry

end new_student_weight_l101_10146


namespace fourteen_sided_figure_area_l101_10132

/-- A fourteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FourteenSidedFigure where
  /-- The number of full unit squares in the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles in the figure -/
  small_triangles : ℕ

/-- Calculate the area of a fourteen-sided figure -/
def calculate_area (figure : FourteenSidedFigure) : ℝ :=
  figure.full_squares + (figure.small_triangles * 0.5)

theorem fourteen_sided_figure_area :
  ∀ (figure : FourteenSidedFigure),
    figure.full_squares = 10 →
    figure.small_triangles = 8 →
    calculate_area figure = 14 := by
  sorry

end fourteen_sided_figure_area_l101_10132


namespace abc_sum_sixteen_l101_10142

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by sorry

end abc_sum_sixteen_l101_10142


namespace alden_nephews_ratio_l101_10190

-- Define the number of nephews Alden has now
def alden_nephews_now : ℕ := 100

-- Define the number of nephews Alden had 10 years ago
def alden_nephews_10_years_ago : ℕ := 50

-- Define the number of nephews Vihaan has now
def vihaan_nephews : ℕ := alden_nephews_now + 60

-- Theorem stating the ratio of Alden's nephews 10 years ago to now
theorem alden_nephews_ratio : 
  (alden_nephews_10_years_ago : ℚ) / (alden_nephews_now : ℚ) = 1 / 2 :=
by
  -- Assume the total number of nephews is 260
  have total_nephews : alden_nephews_now + vihaan_nephews = 260 := by sorry
  
  -- Prove the ratio
  sorry


end alden_nephews_ratio_l101_10190


namespace strawberries_per_jar_solution_l101_10196

/-- The number of strawberries used in one jar of jam -/
def strawberries_per_jar (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_price : ℕ) (total_revenue : ℕ) : ℕ :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars_sold := total_revenue / jar_price
  total_strawberries / jars_sold

theorem strawberries_per_jar_solution :
  strawberries_per_jar 16 20 4 40 = 7 := by
  sorry

end strawberries_per_jar_solution_l101_10196


namespace symmetric_point_about_origin_l101_10149

/-- Given a point P with coordinates (2, -4), this theorem proves that its symmetric point
    about the origin has coordinates (-2, 4). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (2, -4)
  let symmetric_point : ℝ × ℝ := (-P.1, -P.2)
  symmetric_point = (-2, 4) := by sorry

end symmetric_point_about_origin_l101_10149


namespace building_cost_l101_10111

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Proves that the total cost of all units in the given building is 4950 dollars -/
theorem building_cost : total_cost 12 360 450 7 = 4950 := by
  sorry

end building_cost_l101_10111


namespace quadratic_equation_solution_l101_10198

theorem quadratic_equation_solution (x : ℝ) : x^2 - 5 = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end quadratic_equation_solution_l101_10198


namespace stating_cardboard_ratio_l101_10176

/-- Represents the number of dogs on a Type 1 cardboard -/
def dogs_type1 : ℕ := 28

/-- Represents the number of cats on a Type 1 cardboard -/
def cats_type1 : ℕ := 28

/-- Represents the number of cats on a Type 2 cardboard -/
def cats_type2 : ℕ := 42

/-- Represents the required ratio of cats to dogs -/
def required_ratio : ℚ := 5 / 3

/-- 
Theorem stating that the ratio of Type 1 to Type 2 cardboard 
that satisfies the required cat to dog ratio is 9:4
-/
theorem cardboard_ratio : 
  ∀ (x y : ℚ), 
    x > 0 → y > 0 →
    (cats_type1 * x + cats_type2 * y) / (dogs_type1 * x) = required_ratio →
    x / y = 9 / 4 := by
  sorry

end stating_cardboard_ratio_l101_10176


namespace kims_average_round_answers_l101_10136

/-- Represents the number of correct answers in each round of a math contest -/
structure ContestResults where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Calculates the total points earned in the contest -/
def totalPoints (results : ContestResults) : ℕ :=
  2 * results.easy + 3 * results.average + 5 * results.hard

/-- Kim's contest results -/
def kimsResults : ContestResults := {
  easy := 6,
  average := 2,  -- This is what we want to prove
  hard := 4
}

theorem kims_average_round_answers :
  totalPoints kimsResults = 38 :=
by sorry

end kims_average_round_answers_l101_10136


namespace prob_compatible_donor_is_65_percent_l101_10177

/-- Represents the blood types --/
inductive BloodType
  | O
  | A
  | B
  | AB

/-- Distribution of blood types in the population --/
def bloodTypeDistribution : BloodType → ℝ
  | BloodType.O  => 0.50
  | BloodType.A  => 0.15
  | BloodType.B  => 0.30
  | BloodType.AB => 0.05

/-- Predicate for blood types compatible with Type A --/
def compatibleWithA : BloodType → Prop
  | BloodType.O => True
  | BloodType.A => True
  | _ => False

/-- The probability of selecting a compatible donor for a Type A patient --/
def probCompatibleDonor : ℝ :=
  (bloodTypeDistribution BloodType.O) + (bloodTypeDistribution BloodType.A)

theorem prob_compatible_donor_is_65_percent :
  probCompatibleDonor = 0.65 := by
  sorry

end prob_compatible_donor_is_65_percent_l101_10177


namespace increase_decrease_threshold_l101_10185

theorem increase_decrease_threshold (S x y : ℝ) 
  (hS : S > 0) (hxy : x > y) (hy : y > 0) : 
  ((S * (1 + x/100) + 15) * (1 - y/100) > S + 10) ↔ 
  (x > y + (x*y/100) + 500 - (1500*y/S)) :=
sorry

end increase_decrease_threshold_l101_10185


namespace tan_equality_unique_solution_l101_10148

theorem tan_equality_unique_solution : 
  ∃! (n : ℤ), -100 < n ∧ n < 100 ∧ Real.tan (n * π / 180) = Real.tan (216 * π / 180) :=
by
  -- The unique solution is n = 36
  use 36
  sorry

end tan_equality_unique_solution_l101_10148


namespace recipe_total_cups_l101_10189

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalIngredients (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given a recipe with ratio 2:5:3 and 10 cups of flour, the total ingredients is 20 cups -/
theorem recipe_total_cups (ratio : RecipeRatio) (h1 : ratio.butter = 2) (h2 : ratio.flour = 5) (h3 : ratio.sugar = 3) :
  totalIngredients ratio 10 = 20 := by
  sorry

end recipe_total_cups_l101_10189


namespace gcd_5280_12155_l101_10199

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l101_10199


namespace midnight_probability_l101_10140

/-- Represents the words from which letters are selected -/
inductive Word
| ROAD
| LIGHTS
| TIME

/-- Represents the target word MIDNIGHT -/
def targetWord : String := "MIDNIGHT"

/-- Number of letters to select from each word -/
def selectCount (w : Word) : Nat :=
  match w with
  | .ROAD => 2
  | .LIGHTS => 3
  | .TIME => 4

/-- The probability of selecting the required letters from a given word -/
def selectionProbability (w : Word) : Rat :=
  match w with
  | .ROAD => 1 / 3
  | .LIGHTS => 1 / 20
  | .TIME => 1 / 4

/-- The total probability of selecting all required letters -/
def totalProbability : Rat :=
  (selectionProbability .ROAD) * (selectionProbability .LIGHTS) * (selectionProbability .TIME)

theorem midnight_probability : totalProbability = 1 / 240 := by
  sorry

end midnight_probability_l101_10140


namespace angela_height_l101_10117

/-- Given the heights of five people with specific relationships, prove Angela's height. -/
theorem angela_height (carl becky amy helen angela : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = becky * 1.2)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4) :
  angela = 295 := by
  sorry

end angela_height_l101_10117


namespace puzzle_solution_l101_10152

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Calculate the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem puzzle_solution :
  ∀ (row col1 col2 : ArithmeticSequence),
    row.first = 28 →
    nthTerm row 4 = 25 →
    nthTerm row 5 = 32 →
    nthTerm col2 7 = -10 →
    col2.first = nthTerm row 7 →
    col1.first = 28 →
    col1.diff = 7 →
    col2.first = -6 := by
  sorry

end puzzle_solution_l101_10152


namespace problem_statement_l101_10156

theorem problem_statement :
  (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 ≥ 2) ∧
  (¬ ∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∧
  ((∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∨ (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 > 2)) :=
by sorry

end problem_statement_l101_10156


namespace sum_of_ages_l101_10181

/-- Given that Rachel is 19 years old and 4 years older than Leah, 
    prove that the sum of their ages is 34. -/
theorem sum_of_ages (rachel_age : ℕ) (leah_age : ℕ) : 
  rachel_age = 19 → rachel_age = leah_age + 4 → rachel_age + leah_age = 34 := by
  sorry

end sum_of_ages_l101_10181


namespace diagonal_length_of_quadrilateral_l101_10154

theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 11)
  (h2 : offset2 = 9)
  (h3 : area = 400) :
  (2 * area) / (offset1 + offset2) = 40 :=
sorry

end diagonal_length_of_quadrilateral_l101_10154


namespace curve_translation_l101_10101

-- Define a function representing the original curve
variable (f : ℝ → ℝ)

-- Define the translation
def translate (curve : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  fun x ↦ curve (x - h) + k

-- Theorem statement
theorem curve_translation (f : ℝ → ℝ) :
  ∃ h k : ℝ, 
    (translate f h k 2 = 3) ∧ 
    (translate f h k = fun x ↦ f (x - 1) + 2) ∧
    (h = 1) ∧ (k = 2) := by
  sorry


end curve_translation_l101_10101


namespace karlson_candies_theorem_l101_10106

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 33

/-- The number of minutes Karlson has -/
def total_minutes : ℕ := 33

/-- The maximum number of candies Karlson can eat -/
def max_candies : ℕ := initial_ones.choose 2

/-- Theorem stating that the maximum number of candies Karlson can eat
    is equal to the number of unique pairs from the initial ones -/
theorem karlson_candies_theorem :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 :=
by sorry

end karlson_candies_theorem_l101_10106


namespace count_divisible_numbers_l101_10115

theorem count_divisible_numbers : 
  (Finset.filter 
    (fun k : ℕ => k ≤ 267000 ∧ (k^2 - 1) % 267 = 0) 
    (Finset.range 267001)).card = 4000 :=
by sorry

end count_divisible_numbers_l101_10115


namespace curve_property_l101_10191

/-- Given a function f(x) = a*ln(x) + b*x + 1 with specific properties, prove a - b = 10 -/
theorem curve_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * Real.log x + b * x + 1
  (∀ x, HasDerivAt f (a / x + b) x) →
  HasDerivAt f (-2) 1 →
  HasDerivAt f 0 (2/3) →
  a - b = 10 := by
sorry

end curve_property_l101_10191


namespace train_distance_difference_l101_10102

theorem train_distance_difference (v : ℝ) (h1 : v > 0) : 
  let d_ab := 7 * v
  let d_bc := 5 * v
  6 = d_ab + d_bc →
  d_ab - d_bc = 1 := by
sorry

end train_distance_difference_l101_10102


namespace sum_of_differences_mod_1000_l101_10184

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences_mod_1000 : N % 1000 = 304 := by sorry

end sum_of_differences_mod_1000_l101_10184


namespace luke_candy_purchase_l101_10197

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
def candyPieces (whackAMoleTickets skeeBallTickets candyCost : ℕ) : ℕ :=
  (whackAMoleTickets + skeeBallTickets) / candyCost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase :
  candyPieces 2 13 3 = 5 := by
  sorry

end luke_candy_purchase_l101_10197


namespace expression_factorization_l101_10165

theorem expression_factorization (x : ℝ) : 
  (9 * x^4 - 138 * x^3 + 49 * x^2) - (-3 * x^4 + 27 * x^3 - 14 * x^2) = 
  3 * x^2 * (4 * x - 3) * (x - 7) := by
sorry

end expression_factorization_l101_10165


namespace triangle_inequality_l101_10116

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  2 * (a^2 + b^2) > c^2 ∧ 
  ∀ ε > 0, ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (2 - ε) * (a'^2 + b'^2) ≤ c'^2 := by
  sorry

end triangle_inequality_l101_10116


namespace unique_cube_root_property_l101_10141

theorem unique_cube_root_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∃ (m : ℕ), n = m^3 ∧ m = n / 1000) ∧
  n = 32768 := by
  sorry

end unique_cube_root_property_l101_10141


namespace smallest_two_digit_prime_with_composite_reversal_l101_10170

/-- Returns true if n is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with the digit 3 -/
def starts_with_three (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 39

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The theorem stating that 53 is the smallest two-digit prime starting with 3 
    whose digit reversal is composite -/
theorem smallest_two_digit_prime_with_composite_reversal : 
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    starts_with_three n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverse_digits n)) ∧
    (∀ m : ℕ, m < n → 
      is_two_digit m → 
      starts_with_three m → 
      Nat.Prime m → 
      Nat.Prime (reverse_digits m)) ∧
    n = 53 :=
sorry

end smallest_two_digit_prime_with_composite_reversal_l101_10170


namespace complex_number_in_first_quadrant_l101_10126

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I : ℂ) / (2 + Complex.I) = ⟨a, b⟩ := by
  sorry

end complex_number_in_first_quadrant_l101_10126


namespace decimal_to_fraction_l101_10119

theorem decimal_to_fraction : 
  (3.68 : ℚ) = 92 / 25 := by sorry

end decimal_to_fraction_l101_10119


namespace hyperbola_eccentricity_l101_10153

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and a point P on C such that
    PF₁ ⟂ F₁F₂ and PF₁ = F₁F₂, prove that the eccentricity of C is √2 + 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ))
  (hC : C = {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
  (F₁ F₂ P : ℝ × ℝ)
  (hF₁ : F₁ ∈ C) (hF₂ : F₂ ∈ C) (hP : P ∈ C)
  (hLeft : (F₁.1 < F₂.1)) -- F₁ is left of F₂
  (hPerp : (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0) -- PF₁ ⟂ F₁F₂
  (hEqual : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) -- PF₁ = F₁F₂
  : (Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) / (2 * a)) = Real.sqrt 2 + 1 := by
  sorry

end hyperbola_eccentricity_l101_10153


namespace highest_power_of_six_in_twelve_factorial_l101_10173

/-- The highest power of 6 that divides 12! is 6^5 -/
theorem highest_power_of_six_in_twelve_factorial :
  ∃ k : ℕ, (12 : ℕ).factorial = 6^5 * k ∧ ¬(∃ m : ℕ, (12 : ℕ).factorial = 6^6 * m) := by
  sorry

end highest_power_of_six_in_twelve_factorial_l101_10173


namespace rect_to_polar_conversion_l101_10135

/-- Conversion from rectangular coordinates to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = -1 ∧ y = 1 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end rect_to_polar_conversion_l101_10135


namespace joint_club_afternoon_solution_l101_10158

/-- Represents the joint club afternoon scenario with two classes -/
structure JointClubAfternoon where
  a : ℕ  -- number of students in class A
  b : ℕ  -- number of students in class B
  K : ℕ  -- the amount each student would pay if one class covered all costs

/-- Conditions for the joint club afternoon -/
def scenario (j : JointClubAfternoon) : Prop :=
  -- Total contribution for the first event
  5 * j.a + 3 * j.b = j.K * j.a
  ∧
  -- Total contribution for the second event
  4 * j.a + 6 * j.b = j.K * j.b
  ∧
  -- Class B has more students than class A
  j.b > j.a

/-- Theorem stating the solution to the problem -/
theorem joint_club_afternoon_solution (j : JointClubAfternoon) 
  (h : scenario j) : j.K = 9 ∧ j.b > j.a := by
  sorry


end joint_club_afternoon_solution_l101_10158


namespace unequal_gender_probability_l101_10105

theorem unequal_gender_probability :
  let n : ℕ := 12  -- Total number of children
  let p : ℚ := 1/2  -- Probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- Total number of possible outcomes
  let equal_outcomes : ℕ := n.choose (n/2)  -- Number of outcomes with equal boys and girls
  (1 : ℚ) - (equal_outcomes : ℚ) / total_outcomes = 793/1024 :=
by sorry

end unequal_gender_probability_l101_10105


namespace team_formation_count_l101_10118

/-- The number of ways to form a team of 4 students from 6 university students -/
def team_formation_ways : ℕ := 180

/-- The number of university students -/
def total_students : ℕ := 6

/-- The number of students in the team -/
def team_size : ℕ := 4

/-- The number of team leaders -/
def num_leaders : ℕ := 1

/-- The number of deputy team leaders -/
def num_deputies : ℕ := 1

/-- The number of ordinary members -/
def num_ordinary : ℕ := 2

theorem team_formation_count :
  team_formation_ways = 
    (total_students.choose num_leaders) * 
    ((total_students - num_leaders).choose num_deputies) * 
    ((total_students - num_leaders - num_deputies).choose num_ordinary) :=
sorry

end team_formation_count_l101_10118


namespace sum_of_first_four_terms_l101_10138

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 243) :
  (a 1) + (a 2) + (a 3) + (a 4) = 120 :=
sorry

end sum_of_first_four_terms_l101_10138


namespace sum_cubic_over_power_of_three_l101_10131

/-- The sum of the infinite series ∑_{k = 1}^∞ (k^3 / 3^k) is equal to 1.5 -/
theorem sum_cubic_over_power_of_three :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = (3/2 : ℝ) := by
  sorry

end sum_cubic_over_power_of_three_l101_10131


namespace intersection_distance_l101_10113

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The start point of the line -/
def startPoint : Point3D := ⟨1, 2, 3⟩

/-- The end point of the line -/
def endPoint : Point3D := ⟨3, 6, 7⟩

/-- The center of the unit sphere -/
def sphereCenter : Point3D := ⟨0, 0, 0⟩

/-- The radius of the unit sphere -/
def sphereRadius : ℝ := 1

/-- Theorem stating that the distance between the two intersection points of the line and the unit sphere is 12√145/33 -/
theorem intersection_distance : 
  ∃ (p1 p2 : Point3D), 
    (∃ (t1 t2 : ℝ), 
      p1 = ⟨startPoint.x + t1 * (endPoint.x - startPoint.x), 
            startPoint.y + t1 * (endPoint.y - startPoint.y), 
            startPoint.z + t1 * (endPoint.z - startPoint.z)⟩ ∧
      p2 = ⟨startPoint.x + t2 * (endPoint.x - startPoint.x), 
            startPoint.y + t2 * (endPoint.y - startPoint.y), 
            startPoint.z + t2 * (endPoint.z - startPoint.z)⟩ ∧
      (p1.x - sphereCenter.x)^2 + (p1.y - sphereCenter.y)^2 + (p1.z - sphereCenter.z)^2 = sphereRadius^2 ∧
      (p2.x - sphereCenter.x)^2 + (p2.y - sphereCenter.y)^2 + (p2.z - sphereCenter.z)^2 = sphereRadius^2) →
    ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2) = (12 * Real.sqrt 145 / 33)^2 := by
  sorry

end intersection_distance_l101_10113


namespace circle_radius_from_arc_length_and_angle_l101_10163

/-- Given a circle with a sector having an arc length of 25 cm and a central angle of 45 degrees,
    the radius of the circle is equal to 100/π cm. -/
theorem circle_radius_from_arc_length_and_angle (L : ℝ) (θ : ℝ) (r : ℝ) :
  L = 25 →
  θ = 45 →
  L = (θ / 360) * (2 * π * r) →
  r = 100 / π :=
by sorry

end circle_radius_from_arc_length_and_angle_l101_10163


namespace sum_of_rearranged_digits_l101_10183

theorem sum_of_rearranged_digits : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end sum_of_rearranged_digits_l101_10183


namespace ten_player_tournament_matches_l101_10169

/-- Calculates the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tennis tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end ten_player_tournament_matches_l101_10169
