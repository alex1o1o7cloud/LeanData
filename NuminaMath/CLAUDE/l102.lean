import Mathlib

namespace gcf_of_32_and_12_l102_10254

theorem gcf_of_32_and_12 (n : ℕ) (h1 : n = 32) (h2 : Nat.lcm n 12 = 48) :
  Nat.gcd n 12 = 8 := by
  sorry

end gcf_of_32_and_12_l102_10254


namespace solve_inequality_one_solve_inequality_two_l102_10286

namespace InequalitySolver

-- Part 1
theorem solve_inequality_one (x : ℝ) :
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 :=
sorry

-- Part 2
theorem solve_inequality_two (x a : ℝ) :
  (a = 0 → ¬∃x, x^2 - a*x - 2*a^2 < 0) ∧
  (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
  (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a)) :=
sorry

end InequalitySolver

end solve_inequality_one_solve_inequality_two_l102_10286


namespace shooting_probabilities_l102_10293

/-- Probability of shooter A hitting the target -/
def prob_A : ℝ := 0.7

/-- Probability of shooter B hitting the target -/
def prob_B : ℝ := 0.6

/-- Probability of shooter C hitting the target -/
def prob_C : ℝ := 0.5

/-- Probability that at least one person hits the target -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Probability that exactly two people hit the target -/
def prob_exactly_two : ℝ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

theorem shooting_probabilities : 
  prob_at_least_one = 0.94 ∧ prob_exactly_two = 0.44 := by
  sorry

end shooting_probabilities_l102_10293


namespace expression_evaluation_l102_10269

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end expression_evaluation_l102_10269


namespace rectangle_max_regions_l102_10218

/-- The maximum number of regions a rectangle can be divided into with n line segments --/
def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1
  else max_regions (n - 1) + n

/-- Theorem: A rectangle with 5 line segments can be divided into at most 16 regions --/
theorem rectangle_max_regions :
  max_regions 5 = 16 := by
  sorry

end rectangle_max_regions_l102_10218


namespace special_key_102_presses_l102_10253

def f (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem special_key_102_presses :
  iterate_f 102 7 = 7 := by
  sorry

end special_key_102_presses_l102_10253


namespace remainder_3042_div_29_l102_10260

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end remainder_3042_div_29_l102_10260


namespace complex_distance_theorem_l102_10224

theorem complex_distance_theorem : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧ 
  ∀ (z : ℂ), Complex.abs z = 1 → 1 + z + z^2 ≠ 0 → 
    Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c) = d :=
by sorry

end complex_distance_theorem_l102_10224


namespace chairs_per_trip_l102_10272

theorem chairs_per_trip 
  (num_students : ℕ) 
  (trips_per_student : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5) 
  (h2 : trips_per_student = 10) 
  (h3 : total_chairs = 250) : 
  (total_chairs / (num_students * trips_per_student) : ℚ) = 5 := by
sorry

end chairs_per_trip_l102_10272


namespace orange_juice_concentrate_size_l102_10226

/-- The size of a can of orange juice concentrate in ounces -/
def concentrate_size : ℝ := 420

/-- The number of servings to be prepared -/
def num_servings : ℕ := 280

/-- The size of each serving in ounces -/
def serving_size : ℝ := 6

/-- The ratio of water cans to concentrate cans -/
def water_to_concentrate_ratio : ℝ := 3

theorem orange_juice_concentrate_size :
  concentrate_size * (1 + water_to_concentrate_ratio) * num_servings = serving_size * num_servings :=
sorry

end orange_juice_concentrate_size_l102_10226


namespace subset_implies_x_value_l102_10234

theorem subset_implies_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-2, 1} → 
  B = {0, 1, x + 1} → 
  A ⊆ B → 
  x = -3 := by
sorry

end subset_implies_x_value_l102_10234


namespace smallest_number_l102_10246

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end smallest_number_l102_10246


namespace parallel_vectors_sum_l102_10279

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => 4
  | 2 => x

def b (y : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => y
  | 2 => 2

-- Theorem statement
theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, a x i = k * b y i)) →
  x + y = 6 :=
by sorry

end parallel_vectors_sum_l102_10279


namespace product_equals_sum_implies_x_value_l102_10267

theorem product_equals_sum_implies_x_value (x : ℝ) : 
  let S : Set ℝ := {3, 6, 9, x}
  (∃ (a b : ℝ), a ∈ S ∧ b ∈ S ∧ (∀ y ∈ S, a ≤ y ∧ y ≤ b) ∧ a * b = (3 + 6 + 9 + x)) →
  x = 9/4 := by
sorry

end product_equals_sum_implies_x_value_l102_10267


namespace sarah_trucks_left_l102_10225

/-- The number of trucks Sarah has left after giving some away -/
def trucks_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sarah has 38 trucks left after starting with 51 and giving away 13 -/
theorem sarah_trucks_left :
  trucks_left 51 13 = 38 := by
  sorry

end sarah_trucks_left_l102_10225


namespace max_area_rectangle_with_perimeter_30_l102_10221

/-- The maximum area of a rectangle with perimeter 30 meters is 225/4 square meters. -/
theorem max_area_rectangle_with_perimeter_30 :
  let perimeter : ℝ := 30
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = perimeter ∧
    ∀ (a b : ℝ), a > 0 → b > 0 → 2 * (a + b) = perimeter →
      x * y ≥ a * b ∧ x * y = 225 / 4 := by
  sorry

end max_area_rectangle_with_perimeter_30_l102_10221


namespace inequality_proof_l102_10212

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3/2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end inequality_proof_l102_10212


namespace apple_purchase_theorem_l102_10268

/-- The cost of apples with a two-tier pricing system -/
def apple_cost (l q : ℚ) (x : ℚ) : ℚ :=
  if x ≤ 30 then l * x
  else l * 30 + q * (x - 30)

theorem apple_purchase_theorem (l q : ℚ) :
  (∀ x, x ≤ 30 → apple_cost l q x = l * x) ∧
  (∀ x, x > 30 → apple_cost l q x = l * 30 + q * (x - 30)) ∧
  (apple_cost l q 36 = 366) ∧
  (apple_cost l q 15 = 150) ∧
  (∃ x, apple_cost l q x = 333) →
  ∃ x, apple_cost l q x = 333 ∧ x = 33 :=
by sorry

end apple_purchase_theorem_l102_10268


namespace cut_prism_edge_count_l102_10299

/-- A rectangular prism with cut corners -/
structure CutPrism where
  /-- The number of vertices in the original rectangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original rectangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The planes cutting the prism do not intersect within the prism -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting the corners -/
def new_edge_count (p : CutPrism) : Nat :=
  p.original_edges + p.original_vertices * p.new_edges_per_cut

/-- Theorem stating that a rectangular prism with cut corners has 36 edges -/
theorem cut_prism_edge_count :
  ∀ (p : CutPrism),
  p.original_vertices = 8 →
  p.original_edges = 12 →
  p.new_edges_per_cut = 3 →
  p.non_intersecting_cuts →
  new_edge_count p = 36 := by
  sorry

end cut_prism_edge_count_l102_10299


namespace cakes_sold_l102_10202

theorem cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 62 →
  additional_cakes = 149 →
  remaining_cakes = 67 →
  initial_cakes + additional_cakes - remaining_cakes = 144 :=
by sorry

end cakes_sold_l102_10202


namespace sum_remainder_nine_specific_sum_remainder_l102_10242

theorem sum_remainder_nine (n : ℕ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) % 9 = ((n % 9 + (n + 1) % 9 + (n + 2) % 9 + (n + 3) % 9 + (n + 4) % 9) % 9) := by
  sorry

theorem specific_sum_remainder :
  (9150 + 9151 + 9152 + 9153 + 9154) % 9 = 1 := by
  sorry

end sum_remainder_nine_specific_sum_remainder_l102_10242


namespace division_problem_l102_10200

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end division_problem_l102_10200


namespace yoongi_subtraction_l102_10250

theorem yoongi_subtraction (A B C : Nat) (h1 : A ≥ 1) (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) :
  (1000 * A + 100 * B + 10 * C + 6) - 57 = 1819 →
  (1000 * A + 100 * B + 10 * C + 9) - 57 = 1822 := by
sorry

end yoongi_subtraction_l102_10250


namespace area_30_60_90_triangle_l102_10210

/-- The area of a 30-60-90 triangle with hypotenuse 6 is 9√3/2 -/
theorem area_30_60_90_triangle (h : Real) (A : Real) : 
  h = 6 → -- hypotenuse is 6 units
  A = (9 * Real.sqrt 3) / 2 → -- area is 9√3/2 square units
  ∃ (s1 s2 : Real), -- there exist two sides s1 and s2 such that
    s1^2 + s2^2 = h^2 ∧ -- Pythagorean theorem
    s1 = h / 2 ∧ -- shortest side is half the hypotenuse
    s2 = s1 * Real.sqrt 3 ∧ -- longer side is √3 times the shorter side
    A = (1 / 2) * s1 * s2 -- area formula
  := by sorry

end area_30_60_90_triangle_l102_10210


namespace metallic_sheet_width_l102_10249

/-- Given a rectangular metallic sheet with length 48 m, from which squares of side 
    length 6 m are cut from each corner to form an open box, prove that if the 
    volume of the resulting box is 5184 m³, then the width of the original 
    metallic sheet is 36 m. -/
theorem metallic_sheet_width (sheet_length : ℝ) (cut_square_side : ℝ) (box_volume : ℝ) 
  (sheet_width : ℝ) :
  sheet_length = 48 →
  cut_square_side = 6 →
  box_volume = 5184 →
  box_volume = (sheet_length - 2 * cut_square_side) * 
               (sheet_width - 2 * cut_square_side) * 
               cut_square_side →
  sheet_width = 36 :=
by sorry

end metallic_sheet_width_l102_10249


namespace zeros_sum_greater_than_four_l102_10289

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k

theorem zeros_sum_greater_than_four (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : k > Real.exp 2)
  (h₂ : f k x₁ = 0)
  (h₃ : f k x₂ = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := by
  sorry

end zeros_sum_greater_than_four_l102_10289


namespace sin_cos_sum_formula_l102_10207

theorem sin_cos_sum_formula (α β : ℝ) : 
  Real.sin α * Real.sin β - Real.cos α * Real.cos β = - Real.cos (α + β) := by
  sorry

end sin_cos_sum_formula_l102_10207


namespace square_difference_l102_10219

theorem square_difference : (30 : ℕ)^2 - (29 : ℕ)^2 = 59 := by
  sorry

end square_difference_l102_10219


namespace smallest_dual_base_representation_l102_10236

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = 1 * a + 3 ∧
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → 1 * x + 3 = 3 * y + 1 → 1 * x + 3 ≥ 13) :=
by sorry

end smallest_dual_base_representation_l102_10236


namespace largest_A_when_quotient_equals_remainder_l102_10288

theorem largest_A_when_quotient_equals_remainder (A B C : ℕ) : 
  A = 7 * B + C → B = C → A ≤ 48 ∧ ∃ (A₀ B₀ C₀ : ℕ), A₀ = 7 * B₀ + C₀ ∧ B₀ = C₀ ∧ A₀ = 48 := by
  sorry

end largest_A_when_quotient_equals_remainder_l102_10288


namespace arithmetic_sequence_sixth_term_l102_10239

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : (a 2)^2 + 12*(a 2) - 8 = 0 ∧ (a 10)^2 + 12*(a 10) - 8 = 0) :
  a 6 = -6 := by
sorry

end arithmetic_sequence_sixth_term_l102_10239


namespace second_polygon_sides_l102_10211

theorem second_polygon_sides (p1 p2 : ℕ → ℝ) (n2 : ℕ) :
  (∀ k : ℕ, p1 k = p2 k) →  -- Same perimeter
  (p1 45 = 45 * (3 * p2 n2)) →  -- First polygon has 45 sides and 3 times the side length
  n2 * p2 n2 = p2 n2 * 135 →  -- Perimeter of second polygon
  n2 = 135 := by
sorry

end second_polygon_sides_l102_10211


namespace hyperbola_condition_l102_10275

/-- Defines a hyperbola in terms of its equation -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), m * x^2 + n * y^2 = 1 ∧ 
  ∀ (a b : ℝ), a^2 / (1/m) - b^2 / (1/n) = 1 ∨ a^2 / (1/n) - b^2 / (1/m) = 1

/-- Theorem stating that mn < 0 is a necessary and sufficient condition for mx^2 + ny^2 = 1 to represent a hyperbola -/
theorem hyperbola_condition (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end hyperbola_condition_l102_10275


namespace tangent_line_y_intercept_l102_10270

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (6, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 3 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 6 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 6 * Real.sqrt 2 :=
  by sorry

end tangent_line_y_intercept_l102_10270


namespace triangle_perimeter_l102_10213

/-- The perimeter of a triangle with vertices at (1, 4), (-7, 0), and (1, 0) is equal to 4√5 + 12. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 4)
  let B : ℝ × ℝ := (-7, 0)
  let C : ℝ × ℝ := (1, 0)
  let d₁ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d₂ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let d₃ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d₁ + d₂ + d₃ = 4 * Real.sqrt 5 + 12 := by
  sorry


end triangle_perimeter_l102_10213


namespace f_range_l102_10284

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem f_range : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | -9 ≤ y ∧ y ≤ 7} := by
  sorry

end f_range_l102_10284


namespace simplify_expression_l102_10261

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (b * c)) * (a * b / (a^2 - (b + c)^2)) = 1 / (c * (a - b - c)) :=
by sorry

end simplify_expression_l102_10261


namespace extended_quadrilateral_area_l102_10208

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The extended quadrilateral formed by extending the sides of the original quadrilateral -/
noncomputable def extendedQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of the extended quadrilateral is five times the area of the original quadrilateral -/
theorem extended_quadrilateral_area (q : Quadrilateral) :
  area (extendedQuadrilateral q) = 5 * area q := by sorry

end extended_quadrilateral_area_l102_10208


namespace tan_forty_five_degrees_equals_one_l102_10228

theorem tan_forty_five_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end tan_forty_five_degrees_equals_one_l102_10228


namespace existential_vs_universal_quantifier_l102_10217

theorem existential_vs_universal_quantifier :
  ¬(∀ (x₀ : ℝ), x₀^2 > 3 ↔ ∃ (x₀ : ℝ), x₀^2 > 3) :=
by sorry

end existential_vs_universal_quantifier_l102_10217


namespace money_sharing_l102_10222

theorem money_sharing (amanda ben carlos diana total : ℕ) : 
  amanda = 45 →
  amanda + ben + carlos + diana = total →
  3 * ben = 5 * amanda →
  3 * carlos = 6 * amanda →
  3 * diana = 8 * amanda →
  total = 330 := by
sorry

end money_sharing_l102_10222


namespace eldest_boy_age_l102_10264

theorem eldest_boy_age (boys : Fin 3 → ℕ) 
  (avg_age : (boys 0 + boys 1 + boys 2) / 3 = 15)
  (proportion : ∃ (x : ℕ), boys 0 = 3 * x ∧ boys 1 = 5 * x ∧ boys 2 = 7 * x) :
  boys 2 = 21 := by
  sorry

end eldest_boy_age_l102_10264


namespace least_integer_absolute_value_l102_10220

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, y < x → |3 * y + 10| > 25) ∧ |3 * x + 10| ≤ 25 ↔ x = -11 := by
  sorry

end least_integer_absolute_value_l102_10220


namespace triangle_side_length_l102_10235

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  a = Real.sqrt 3 →
  2 * (Real.cos ((A + C) / 2))^2 = (Real.sqrt 2 - 1) * Real.cos B →
  A = π / 3 →
  -- Conclusion
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end triangle_side_length_l102_10235


namespace return_trip_duration_l102_10266

/-- Represents the flight scenario with given conditions -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  time_against_wind : ℝ -- time flying against the wind
  time_diff_still_air : ℝ -- time difference compared to still air for return trip

/-- The possible durations for the return trip -/
def possible_return_times : Set ℝ := {60, 40}

/-- Theorem stating that the return trip duration is either 60 or 40 minutes -/
theorem return_trip_duration (scenario : FlightScenario) 
  (h1 : scenario.time_against_wind = 120)
  (h2 : scenario.time_diff_still_air = 20)
  (h3 : scenario.d > 0)
  (h4 : scenario.p > scenario.w)
  (h5 : scenario.w > 0) :
  ∃ (t : ℝ), t ∈ possible_return_times ∧ 
    scenario.d / (scenario.p + scenario.w) = t := by
  sorry


end return_trip_duration_l102_10266


namespace prime_pairs_perfect_square_l102_10251

theorem prime_pairs_perfect_square :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (∃ a : ℕ, p^2 + p*q + q^2 = a^2) → 
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end prime_pairs_perfect_square_l102_10251


namespace final_book_count_l102_10274

/-- The number of storybooks in a library after borrowing and returning books. -/
def library_books (initial : ℕ) (borrowed : ℕ) (returned : ℕ) : ℕ :=
  initial - borrowed + returned

/-- Theorem stating that given the initial conditions, the library ends up with 72 books. -/
theorem final_book_count :
  library_books 95 58 35 = 72 := by
  sorry

end final_book_count_l102_10274


namespace point_N_coordinates_l102_10243

-- Define the point M and vector a
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

-- Define the relation between MN and a
def MN_relation (N : ℝ × ℝ) : Prop :=
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2)

-- Theorem statement
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, MN_relation N ∧ N = (2, 0) := by sorry

end point_N_coordinates_l102_10243


namespace roots_polynomial_equation_l102_10205

theorem roots_polynomial_equation (p q : ℝ) (α β γ δ : ℂ) :
  (α^2 + p*α + 1 = 0) →
  (β^2 + p*β + 1 = 0) →
  (γ^2 + q*γ + 1 = 0) →
  (δ^2 + q*δ + 1 = 0) →
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end roots_polynomial_equation_l102_10205


namespace min_value_when_k_is_one_l102_10201

/-- The function for which we want to find the minimum value -/
def f (x k : ℝ) : ℝ := x^2 - (2*k + 3)*x + 2*k^2 - k - 3

/-- The theorem stating the minimum value of the function when k = 1 -/
theorem min_value_when_k_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = -33/4 := by
  sorry

end min_value_when_k_is_one_l102_10201


namespace total_paid_is_705_l102_10290

/-- Calculates the total amount paid for fruits given their quantities and rates -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: The total amount paid for the given quantities and rates of grapes and mangoes is 705 -/
theorem total_paid_is_705 :
  total_amount_paid 3 70 9 55 = 705 := by
  sorry

end total_paid_is_705_l102_10290


namespace trigonometric_identities_l102_10229

theorem trigonometric_identities (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  (Real.tan α = -1 / 3) ∧ 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2) ∧ 
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5) := by
  sorry

end trigonometric_identities_l102_10229


namespace quadratic_inequality_l102_10265

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end quadratic_inequality_l102_10265


namespace ternary_121_equals_16_l102_10263

def ternary_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 3^2 + d₁ * 3^1 + d₀ * 3^0

theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end ternary_121_equals_16_l102_10263


namespace greatest_number_less_than_200_with_odd_factors_l102_10291

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_less_than_200_with_odd_factors : 
  (∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196) ∧ 
  has_odd_number_of_factors 196 ∧ 
  196 < 200 :=
sorry

end greatest_number_less_than_200_with_odd_factors_l102_10291


namespace hostel_provisions_l102_10298

/-- Given a hostel with provisions for a certain number of men, 
    calculate the initial number of days the provisions were planned for. -/
theorem hostel_provisions 
  (initial_men : ℕ) 
  (men_left : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : men_left = 50)
  (h3 : days_after_leaving = 60) :
  (initial_men * (initial_men - men_left) * days_after_leaving) / 
  ((initial_men - men_left) * initial_men) = 48 := by
sorry

end hostel_provisions_l102_10298


namespace min_value_expression_l102_10277

theorem min_value_expression (x : ℝ) (h : x > 2) :
  (x^2 + 8) / Real.sqrt (x - 2) ≥ 22 ∧
  ∃ x₀ > 2, (x₀^2 + 8) / Real.sqrt (x₀ - 2) = 22 :=
by sorry

end min_value_expression_l102_10277


namespace sum_1_to_15_mod_11_l102_10209

theorem sum_1_to_15_mod_11 : (List.range 15).sum % 11 = 10 := by sorry

end sum_1_to_15_mod_11_l102_10209


namespace square_land_multiple_l102_10206

theorem square_land_multiple (a p k : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4) ^ 2 → 
  5 * a = k * p + 45 → 
  k = 10 := by
sorry

end square_land_multiple_l102_10206


namespace always_not_three_l102_10238

def is_single_digit (n : ℕ) : Prop := n < 10

def statement_I (n : ℕ) : Prop := n = 2
def statement_II (n : ℕ) : Prop := n ≠ 3
def statement_III (n : ℕ) : Prop := n = 5
def statement_IV (n : ℕ) : Prop := Even n

theorem always_not_three (n : ℕ) (h_single_digit : is_single_digit n) 
  (h_three_true : ∃ (a b c : Prop) (ha : a) (hb : b) (hc : c), 
    (a = statement_I n ∨ a = statement_II n ∨ a = statement_III n ∨ a = statement_IV n) ∧
    (b = statement_I n ∨ b = statement_II n ∨ b = statement_III n ∨ b = statement_IV n) ∧
    (c = statement_I n ∨ c = statement_II n ∨ c = statement_III n ∨ c = statement_IV n) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  statement_II n := by
  sorry

end always_not_three_l102_10238


namespace arithmetic_geometric_sequence_l102_10282

/-- An arithmetic sequence with a common difference of 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
sorry

end arithmetic_geometric_sequence_l102_10282


namespace buckingham_palace_visitors_l102_10287

/-- The difference in visitors between the current day and the sum of the previous two days -/
def visitor_difference (current_day : ℕ) (previous_day : ℕ) (two_days_ago : ℕ) : ℤ :=
  (current_day : ℤ) - (previous_day + two_days_ago : ℤ)

/-- Theorem stating the visitor difference for the given numbers -/
theorem buckingham_palace_visitors :
  visitor_difference 1321 890 765 = -334 := by
  sorry

end buckingham_palace_visitors_l102_10287


namespace negation_of_existential_proposition_l102_10231

theorem negation_of_existential_proposition :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end negation_of_existential_proposition_l102_10231


namespace triple_sharp_100_l102_10227

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.6 * N + 1

-- State the theorem
theorem triple_sharp_100 : sharp (sharp (sharp 100)) = 23.56 := by
  sorry

end triple_sharp_100_l102_10227


namespace rectangle_diagonal_l102_10276

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (length width : ℝ),
    2 * (length + width) = perimeter ∧ 
    length * width_ratio = width * length_ratio ∧
    length^2 + width^2 = 656 := by
  sorry

end rectangle_diagonal_l102_10276


namespace trigonometric_identity_l102_10203

theorem trigonometric_identity (A B C : ℝ) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 + 2 * Real.cos A * Real.cos B * Real.cos C := by
  sorry

end trigonometric_identity_l102_10203


namespace intersection_sum_l102_10258

theorem intersection_sum (c d : ℝ) : 
  (2 * 4 + c = 6) →  -- First line passes through (4, 6)
  (5 * 4 + d = 6) →  -- Second line passes through (4, 6)
  c + d = -16 := by
  sorry

end intersection_sum_l102_10258


namespace thompson_children_ages_l102_10233

/-- Represents the ages of Miss Thompson's children -/
def ChildrenAges : Type := Fin 5 → Nat

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  o_range : ones ≥ 0 ∧ ones ≤ 9
  different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

theorem thompson_children_ages
  (ages : ChildrenAges)
  (number : ThreeDigitNumber)
  (h_oldest : ages 0 = 11)
  (h_middle : ages 2 = 7)
  (h_different : ∀ i j, i ≠ j → ages i ≠ ages j)
  (h_divisible_oldest : (number.hundreds * 100 + number.tens * 10 + number.ones) % 11 = 0)
  (h_divisible_middle : (number.hundreds * 100 + number.tens * 10 + number.ones) % 7 = 0)
  (h_youngest : ∃ i, ages i = number.ones)
  : ¬(∃ i, ages i = 6) :=
by sorry

end thompson_children_ages_l102_10233


namespace max_soap_boxes_in_carton_l102_10295

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem: The maximum number of soap boxes that can fit in the carton is 300 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end max_soap_boxes_in_carton_l102_10295


namespace triangle_area_l102_10245

def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -2 * x + 3

theorem triangle_area : 
  let x_intercept := (3 : ℝ) / 2
  let intersection_x := (1 : ℝ)
  let intersection_y := line1 intersection_x
  let base := x_intercept
  let height := intersection_y
  (1 / 2) * base * height = 3 / 4 := by sorry

end triangle_area_l102_10245


namespace basketball_spectators_l102_10215

theorem basketball_spectators 
  (total_spectators : ℕ) 
  (men : ℕ) 
  (ratio_children : ℕ) 
  (ratio_women : ℕ) : 
  total_spectators = 25000 →
  men = 15320 →
  ratio_children = 7 →
  ratio_women = 3 →
  (total_spectators - men) * ratio_children / (ratio_children + ratio_women) = 6776 :=
by sorry

end basketball_spectators_l102_10215


namespace arrange_five_photos_l102_10232

theorem arrange_five_photos (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end arrange_five_photos_l102_10232


namespace existence_of_m_n_l102_10285

theorem existence_of_m_n (h k : ℕ+) (ε : ℝ) (hε : ε > 0) :
  ∃ (m n : ℕ+), ε < |h * Real.sqrt m - k * Real.sqrt n| ∧ |h * Real.sqrt m - k * Real.sqrt n| < 2 * ε :=
sorry

end existence_of_m_n_l102_10285


namespace shoe_store_sale_l102_10281

theorem shoe_store_sale (sneakers sandals boots : ℕ) 
  (h1 : sneakers = 2) 
  (h2 : sandals = 4) 
  (h3 : boots = 11) : 
  sneakers + sandals + boots = 17 := by
  sorry

end shoe_store_sale_l102_10281


namespace continuous_function_composite_power_l102_10230

theorem continuous_function_composite_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) → k ≥ 0 := by
  sorry

end continuous_function_composite_power_l102_10230


namespace platform_length_l102_10241

/-- The length of a platform given a train's speed, crossing time, and length -/
theorem platform_length 
  (train_speed : Real) 
  (crossing_time : Real) 
  (train_length : Real) : 
  train_speed = 72 * (5/18) → 
  crossing_time = 36 → 
  train_length = 470.06 → 
  (train_speed * crossing_time) - train_length = 249.94 := by
  sorry

end platform_length_l102_10241


namespace sum_bound_l102_10283

theorem sum_bound (k a b c : ℝ) (h1 : k > 1) (h2 : a ≥ 0) (h3 : b ≥ 0) (h4 : c ≥ 0)
  (h5 : a ≤ k * c) (h6 : b ≤ k * c) (h7 : a * b ≤ c^2) :
  a + b ≤ (k + 1/k) * c := by
  sorry

end sum_bound_l102_10283


namespace proposition_analysis_l102_10294

-- Define the propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statement
theorem proposition_analysis :
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  (p ∧ ¬q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬(¬p ∨ q) :=
sorry

end proposition_analysis_l102_10294


namespace sqrt_square_eq_abs_l102_10252

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end sqrt_square_eq_abs_l102_10252


namespace test_total_points_l102_10292

theorem test_total_points (total_questions : ℕ) (two_point_questions : ℕ) : 
  total_questions = 40 → 
  two_point_questions = 30 → 
  (total_questions - two_point_questions) * 4 + two_point_questions * 2 = 100 := by
sorry

end test_total_points_l102_10292


namespace stream_speed_l102_10223

/-- 
Given a boat with a speed of 22 km/hr in still water that travels 108 km downstream in 4 hours,
prove that the speed of the stream is 5 km/hr.
-/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 108 →
  time = 4 →
  boat_speed + stream_speed = distance / time →
  stream_speed = 5 := by
sorry

end stream_speed_l102_10223


namespace prime_square_mod_60_l102_10296

-- Define the set of primes greater than 3
def PrimesGreaterThan3 : Set ℕ := {p : ℕ | Nat.Prime p ∧ p > 3}

-- Theorem statement
theorem prime_square_mod_60 (p : ℕ) (h : p ∈ PrimesGreaterThan3) : 
  p ^ 2 % 60 = 1 ∨ p ^ 2 % 60 = 49 := by
  sorry

end prime_square_mod_60_l102_10296


namespace sin_is_2_type_function_x_plus_cos_is_2_type_function_l102_10204

-- Define what it means for a function to be a t-type function
def is_t_type_function (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv f x₁) + (deriv f x₂) = t

-- State the theorem for sin x
theorem sin_is_2_type_function :
  is_t_type_function Real.sin 2 :=
sorry

-- State the theorem for x + cos x
theorem x_plus_cos_is_2_type_function :
  is_t_type_function (fun x => x + Real.cos x) 2 :=
sorry

end sin_is_2_type_function_x_plus_cos_is_2_type_function_l102_10204


namespace interest_rate_calculation_l102_10248

theorem interest_rate_calculation (P t : ℝ) (diff : ℝ) : 
  P = 10000 → 
  t = 2 → 
  diff = 49 → 
  P * (1 + 7/100)^t - P - (P * 7 * t / 100) = diff :=
by sorry

end interest_rate_calculation_l102_10248


namespace parabola_equation_l102_10257

/-- A parabola is defined by its directrix. -/
structure Parabola where
  directrix : ℝ

/-- The standard equation of a parabola. -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 28 * x

/-- Theorem: If the directrix of a parabola is x = -7, then its standard equation is y² = 28x. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = -7) : standard_equation p := by
  sorry

end parabola_equation_l102_10257


namespace class_grouping_l102_10280

/-- Given a class where students can form 8 pairs when grouped in twos,
    prove that the number of groups formed when students are grouped in fours is 4. -/
theorem class_grouping (num_pairs : ℕ) (h : num_pairs = 8) :
  (2 * num_pairs) / 4 = 4 := by
  sorry

end class_grouping_l102_10280


namespace service_center_location_example_highway_valid_l102_10256

/-- Represents a highway with exits and a service center -/
structure Highway where
  fourth_exit : ℝ
  ninth_exit : ℝ
  service_center : ℝ

/-- The service center is halfway between the fourth and ninth exits -/
def is_halfway (h : Highway) : Prop :=
  h.service_center = (h.fourth_exit + h.ninth_exit) / 2

/-- Theorem: Given the conditions, the service center is at milepost 90 -/
theorem service_center_location (h : Highway)
  (h_fourth : h.fourth_exit = 30)
  (h_ninth : h.ninth_exit = 150)
  (h_halfway : is_halfway h) :
  h.service_center = 90 := by
  sorry

/-- Example highway satisfying the conditions -/
def example_highway : Highway :=
  { fourth_exit := 30
  , ninth_exit := 150
  , service_center := 90 }

/-- The example highway satisfies all conditions -/
theorem example_highway_valid :
  is_halfway example_highway ∧
  example_highway.fourth_exit = 30 ∧
  example_highway.ninth_exit = 150 ∧
  example_highway.service_center = 90 := by
  sorry

end service_center_location_example_highway_valid_l102_10256


namespace smallest_matching_end_digits_correct_l102_10237

/-- The smallest positive integer M such that M and M^2 + 1 end in the same sequence of four digits in base 10, where the first digit of the four is not zero. -/
def smallest_matching_end_digits : ℕ := 3125

/-- Predicate to check if a number ends with the same four digits as its square plus one. -/
def ends_with_same_four_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≡ n^2 + 1 [ZMOD 10000]

theorem smallest_matching_end_digits_correct :
  ends_with_same_four_digits smallest_matching_end_digits ∧
  ∀ m : ℕ, m < smallest_matching_end_digits → ¬ends_with_same_four_digits m := by
  sorry

end smallest_matching_end_digits_correct_l102_10237


namespace ellipse_standard_equation_l102_10255

/-- The standard equation of an ellipse with specific properties -/
theorem ellipse_standard_equation :
  ∀ (a b : ℝ),
  (a = 2 ∧ b = 1) →
  (∀ (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) ↔ 
    (y^2 / a^2 + (x - 2)^2 / b^2 = 1 ∧ 
     a > b ∧ 
     a = 2 * b)) :=
by sorry

end ellipse_standard_equation_l102_10255


namespace other_x_intercept_of_quadratic_l102_10259

/-- Given a quadratic function with vertex (4, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 7. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 4) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                 -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 7 :=
by sorry

end other_x_intercept_of_quadratic_l102_10259


namespace circumscribed_radius_of_specific_trapezoid_l102_10271

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  lateral : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumscribed_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { base1 := 2, base2 := 14, lateral := 10 }
  circumscribedRadius t = 5 * Real.sqrt 2 := by
  sorry

end circumscribed_radius_of_specific_trapezoid_l102_10271


namespace perfect_square_condition_solution_uniqueness_l102_10273

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def repeated_digits (x y : ℕ) (n : ℕ) : ℕ :=
  x * 10^(2*n) + 6 * 10^n + y

theorem perfect_square_condition (x y : ℕ) : Prop :=
  x ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → is_perfect_square (repeated_digits x y n)

theorem solution_uniqueness :
  ∀ x y : ℕ, perfect_square_condition x y →
    ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
by sorry

end perfect_square_condition_solution_uniqueness_l102_10273


namespace divisible_by_six_l102_10247

theorem divisible_by_six (x : ℤ) : 
  (∃ k : ℤ, x^2 + 5*x - 12 = 6*k) ↔ (∃ t : ℤ, x = 3*t ∨ x = 3*t + 1) :=
by sorry

end divisible_by_six_l102_10247


namespace expected_worth_of_coin_flip_l102_10297

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Probability of getting heads on a single flip -/
def probHeads : ℚ := 2/3

/-- Probability of getting tails on a single flip -/
def probTails : ℚ := 1/3

/-- Reward for getting heads -/
def rewardHeads : ℚ := 5

/-- Penalty for getting tails -/
def penaltyTails : ℚ := -9

/-- Additional penalty for three consecutive tails -/
def penaltyThreeTails : ℚ := -10

/-- Expected value of a single coin flip -/
def expectedValueSingleFlip : ℚ := probHeads * rewardHeads + probTails * penaltyTails

/-- Probability of getting three consecutive tails -/
def probThreeTails : ℚ := probTails^3

/-- Additional expected loss from three consecutive tails -/
def expectedAdditionalLoss : ℚ := probThreeTails * penaltyThreeTails

/-- Total expected value of a coin flip -/
def totalExpectedValue : ℚ := expectedValueSingleFlip + expectedAdditionalLoss

theorem expected_worth_of_coin_flip :
  totalExpectedValue = -1/27 := by sorry

end expected_worth_of_coin_flip_l102_10297


namespace fair_coin_five_tosses_l102_10244

/-- The probability of getting heads in a single toss of a fair coin -/
def p_heads : ℚ := 1/2

/-- The number of tosses -/
def n : ℕ := 5

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_exactly (k : ℕ) : ℚ :=
  ↑(n.choose k) * p_heads ^ k * (1 - p_heads) ^ (n - k)

/-- The probability of getting at least 2 heads in 5 tosses of a fair coin -/
def prob_at_least_two : ℚ :=
  1 - prob_exactly 0 - prob_exactly 1

theorem fair_coin_five_tosses :
  prob_at_least_two = 13/16 := by sorry

end fair_coin_five_tosses_l102_10244


namespace four_digit_divisible_by_9_l102_10216

/-- Represents a four-digit number in the form 5BB3 where B is a single digit -/
def fourDigitNumber (B : Nat) : Nat :=
  5000 + 100 * B + 10 * B + 3

/-- Checks if a number is divisible by 9 -/
def isDivisibleBy9 (n : Nat) : Prop :=
  n % 9 = 0

/-- B is a single digit -/
def isSingleDigit (B : Nat) : Prop :=
  B ≥ 0 ∧ B ≤ 9

theorem four_digit_divisible_by_9 :
  ∃ B : Nat, isSingleDigit B ∧ isDivisibleBy9 (fourDigitNumber B) → B = 5 := by
  sorry

end four_digit_divisible_by_9_l102_10216


namespace vacation_cost_sharing_l102_10214

theorem vacation_cost_sharing (john_paid mary_paid lisa_paid : ℝ) (j m : ℝ) : 
  john_paid = 150 →
  mary_paid = 90 →
  lisa_paid = 210 →
  j = 150 - john_paid →
  m = 150 - mary_paid →
  j - m = -60 := by
sorry

end vacation_cost_sharing_l102_10214


namespace rectangular_equation_of_C_chord_length_l102_10240

-- Define the polar curve C
def polar_curve (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 8 * Real.cos θ

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 2 + t ∧ y = Real.sqrt 3 * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation_of_C (x y : ℝ) :
  (∃ ρ θ, polar_curve ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 8*x :=
sorry

-- Theorem for the chord length |AB|
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁, line_l t₁ A.1 A.2 ∧ A.2^2 = 8*A.1) →
  (∃ t₂, line_l t₂ B.1 B.2 ∧ B.2^2 = 8*B.1) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32/3 :=
sorry

end rectangular_equation_of_C_chord_length_l102_10240


namespace sibling_age_sum_l102_10278

/-- Given the ages of three siblings, proves that the sum of two siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = maggie + 3 →
  juliet + 2 = ralph →
  juliet = 10 →
  maggie + ralph = 19 := by
sorry

end sibling_age_sum_l102_10278


namespace questions_per_day_l102_10262

/-- Given a mathematician who needs to write a certain number of questions for two projects in one week,
    this theorem proves the number of questions he should complete each day. -/
theorem questions_per_day
  (project1_questions : ℕ)
  (project2_questions : ℕ)
  (days_in_week : ℕ)
  (h1 : project1_questions = 518)
  (h2 : project2_questions = 476)
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 := by
  sorry

end questions_per_day_l102_10262
