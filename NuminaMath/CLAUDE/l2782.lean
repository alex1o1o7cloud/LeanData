import Mathlib

namespace inequality_solution_l2782_278298

theorem inequality_solution (x : ℝ) : 
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 ≥ 2020*x ↔ x = 1) := by
  sorry

end inequality_solution_l2782_278298


namespace triangle_side_angle_relation_l2782_278265

theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  2 * c^2 - 2 * a^2 = b^2 →
  2 * c * Real.cos A - 2 * a * Real.cos C = b :=
by sorry

end triangle_side_angle_relation_l2782_278265


namespace scheduled_halt_duration_l2782_278287

def average_speed : ℝ := 87
def total_distance : ℝ := 348
def scheduled_start_time : ℝ := 9
def scheduled_end_time : ℝ := 13.75  -- 1:45 PM in decimal hours

theorem scheduled_halt_duration :
  let travel_time_without_halt := total_distance / average_speed
  let scheduled_travel_time := scheduled_end_time - scheduled_start_time
  scheduled_travel_time - travel_time_without_halt = 0.75 := by sorry

end scheduled_halt_duration_l2782_278287


namespace range_of_x_l2782_278235

theorem range_of_x (x : ℝ) : (16 - x^2 ≥ 0) ↔ (-4 ≤ x ∧ x ≤ 4) := by
  sorry

end range_of_x_l2782_278235


namespace square_fraction_count_l2782_278263

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ k : ℤ, n / (25 - n) = k^2 ∧ 25 - n ≠ 0) ∧ 
    S.card = 2 :=
sorry

end square_fraction_count_l2782_278263


namespace perfect_square_property_l2782_278209

theorem perfect_square_property (n : ℕ) (hn : n ≥ 3) 
  (hx : ∃ x : ℕ, 1 + 3 * n = x ^ 2) : 
  ∃ a b c : ℕ, 1 + (3 * n + 3) / (a ^ 2 + b ^ 2 + c ^ 2) = 4 := by
  sorry

end perfect_square_property_l2782_278209


namespace revenue_decrease_l2782_278230

theorem revenue_decrease (last_year_revenue : ℝ) : 
  let projected_revenue := 1.25 * last_year_revenue
  let actual_revenue := 0.6 * projected_revenue
  let decrease := projected_revenue - actual_revenue
  let percentage_decrease := (decrease / projected_revenue) * 100
  percentage_decrease = 40 := by
sorry

end revenue_decrease_l2782_278230


namespace not_divisible_by_169_l2782_278290

theorem not_divisible_by_169 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 7*n - 4 = 169*k := by
  sorry

end not_divisible_by_169_l2782_278290


namespace water_level_decrease_l2782_278201

def water_level_change (change : ℝ) : ℝ := change

theorem water_level_decrease (decrease : ℝ) : 
  water_level_change (-decrease) = -decrease :=
by sorry

end water_level_decrease_l2782_278201


namespace quadratic_root_theorem_l2782_278236

theorem quadratic_root_theorem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + (ab+bc+ca)
  (f 2 = 0) → (∃ x, f x = 0 ∧ x ≠ 2) → (∃ x, f x = 0 ∧ x = a+b+c-2) :=
by sorry

end quadratic_root_theorem_l2782_278236


namespace line_equation_through_ellipse_points_l2782_278204

/-- The equation of a line passing through two points on an ellipse -/
theorem line_equation_through_ellipse_points 
  (A B : ℝ × ℝ) -- Two points on the ellipse
  (h_ellipse_A : (A.1^2 / 16) + (A.2^2 / 12) = 1) -- A is on the ellipse
  (h_ellipse_B : (B.1^2 / 16) + (B.2^2 / 12) = 1) -- B is on the ellipse
  (h_midpoint : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1)) -- Midpoint of AB is (2, 1)
  : ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ 
                    a * B.1 + b * B.2 + c = 0 ∧ 
                    (a, b, c) = (3, 2, -8) :=
by sorry

end line_equation_through_ellipse_points_l2782_278204


namespace binomial_and_power_evaluation_l2782_278267

theorem binomial_and_power_evaluation : 
  (Nat.choose 12 6 = 924) ∧ ((1 + 1 : ℕ)^12 = 4096) := by
  sorry

end binomial_and_power_evaluation_l2782_278267


namespace certain_positive_integer_value_l2782_278255

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem certain_positive_integer_value :
  ∀ (i k m n : Nat),
    factorial 8 = 2^i * 3^k * 5^m * 7^n →
    i + k + m + n = 11 →
    n = 1 := by
  sorry

end certain_positive_integer_value_l2782_278255


namespace equation_solution_l2782_278227

theorem equation_solution : ∃ s : ℚ, 
  (s^2 - 6*s + 8) / (s^2 - 9*s + 14) = (s^2 - 3*s - 18) / (s^2 - 2*s - 24) ∧ 
  s = -5/4 := by
  sorry

end equation_solution_l2782_278227


namespace subset_iff_a_in_range_l2782_278277

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 < 0}
def B : Set ℝ := {x | (x + 1)/(x - 2) < 0}

-- State the theorem
theorem subset_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

end subset_iff_a_in_range_l2782_278277


namespace like_terms_sum_zero_l2782_278226

theorem like_terms_sum_zero (a b : ℝ) (m n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a^(m+1) * b^3 + (n-1) * a^2 * b^3 = 0) → (m = 1 ∧ n = 0) := by
  sorry

end like_terms_sum_zero_l2782_278226


namespace right_triangle_area_right_triangle_area_proof_l2782_278208

/-- The area of a right triangle with legs of length 3 and 5 is 7.5 -/
theorem right_triangle_area : Real → Prop :=
  fun a => 
    ∃ (b h : Real),
      b = 3 ∧
      h = 5 ∧
      a = (1 / 2) * b * h ∧
      a = 7.5

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 7.5 := by
  sorry

end right_triangle_area_right_triangle_area_proof_l2782_278208


namespace quadratic_equation_properties_l2782_278202

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  (∃ x : ℝ, f x = 0) ↔ m ≤ 4 ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁^2 + x₂^2 + (x₁*x₂)^2 = 40 → m = -4) :=
by sorry

end quadratic_equation_properties_l2782_278202


namespace f_is_even_and_decreasing_l2782_278200

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x) :=
sorry

end f_is_even_and_decreasing_l2782_278200


namespace shelter_animals_count_l2782_278293

/-- Calculates the total number of animals in a shelter given the initial conditions --/
def totalAnimals (initialCats : ℕ) : ℕ :=
  let adoptedCats := initialCats / 3
  let remainingCats := initialCats - adoptedCats
  let newCats := adoptedCats * 2
  let totalCats := remainingCats + newCats
  let dogs := totalCats * 2
  totalCats + dogs

/-- Theorem stating that given the initial conditions, the total number of animals is 60 --/
theorem shelter_animals_count : totalAnimals 15 = 60 := by
  sorry

end shelter_animals_count_l2782_278293


namespace pet_center_cats_l2782_278211

theorem pet_center_cats (initial_dogs : ℕ) (adopted_dogs : ℕ) (new_cats : ℕ) (final_total : ℕ) :
  initial_dogs = 36 →
  adopted_dogs = 20 →
  new_cats = 12 →
  final_total = 57 →
  ∃ initial_cats : ℕ,
    initial_cats = 29 ∧
    final_total = (initial_dogs - adopted_dogs) + (initial_cats + new_cats) :=
by sorry

end pet_center_cats_l2782_278211


namespace square_plus_fifteen_perfect_square_l2782_278250

theorem square_plus_fifteen_perfect_square (n : ℤ) : 
  (∃ m : ℤ, n^2 + 15 = m^2) ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 := by
  sorry

end square_plus_fifteen_perfect_square_l2782_278250


namespace fraction_inequality_function_minimum_l2782_278249

-- Problem 1
theorem fraction_inequality (c a b : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Problem 2
theorem function_minimum (x : ℝ) (h : x > 2) :
  x + 16 / (x - 2) ≥ 10 := by sorry

end fraction_inequality_function_minimum_l2782_278249


namespace x_range_l2782_278252

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem x_range (x : ℝ) :
  (f (x - 2) > f 3) → -1 < x ∧ x < 5 := by
  sorry

end x_range_l2782_278252


namespace five_people_seven_chairs_arrangement_l2782_278242

/-- The number of ways to arrange people in chairs with one person fixed -/
def arrangement_count (total_chairs : ℕ) (total_people : ℕ) (fixed_position : ℕ) : ℕ :=
  (total_chairs - 1).factorial / (total_chairs - total_people).factorial

/-- Theorem: Five people can be arranged in seven chairs with one person fixed in the middle in 360 ways -/
theorem five_people_seven_chairs_arrangement : 
  arrangement_count 7 5 4 = 360 := by
sorry

end five_people_seven_chairs_arrangement_l2782_278242


namespace base_nine_subtraction_l2782_278243

/-- Represents a number in base 9 --/
def BaseNine : Type := ℕ

/-- Converts a base 9 number to its decimal (base 10) representation --/
def to_decimal (n : BaseNine) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base 9 representation --/
def from_decimal (n : ℕ) : BaseNine := sorry

/-- Subtracts two base 9 numbers --/
def base_nine_sub (a b : BaseNine) : BaseNine := sorry

/-- The main theorem to prove --/
theorem base_nine_subtraction :
  base_nine_sub (from_decimal 256) (from_decimal 143) = from_decimal 113 := by sorry

end base_nine_subtraction_l2782_278243


namespace latus_rectum_of_parabola_l2782_278273

/-- Given a parabola with equation y = 8x^2, its latus rectum has equation y = 1/32 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y = 8 * x^2 → (∃ (x₀ : ℝ), y = 1/32 ∧ x₀ ≠ 0 ∧ y = 8 * x₀^2) :=
by sorry

end latus_rectum_of_parabola_l2782_278273


namespace rod_cutting_l2782_278283

/-- Given a rod of 17 meters long from which 20 pieces can be cut,
    prove that the length of each piece is 85 centimeters. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length_cm : ℝ) :
  rod_length = 17 →
  num_pieces = 20 →
  piece_length_cm = (rod_length / num_pieces) * 100 →
  piece_length_cm = 85 := by
  sorry

end rod_cutting_l2782_278283


namespace light_source_height_l2782_278270

/-- Given a cube with edge length 3 cm, illuminated by a light source x cm directly
    above and 3 cm horizontally from a top vertex, if the shadow area outside the
    cube's base is 75 square cm, then x = 7 cm. -/
theorem light_source_height (x : ℝ) : 
  let cube_edge : ℝ := 3
  let horizontal_distance : ℝ := 3
  let shadow_area : ℝ := 75
  let total_area : ℝ := cube_edge^2 + shadow_area
  let shadow_side : ℝ := Real.sqrt total_area
  let height_increase : ℝ := shadow_side - cube_edge
  x = (cube_edge * (cube_edge + height_increase)) / height_increase → x = 7 := by
  sorry

end light_source_height_l2782_278270


namespace proposition_relation_l2782_278296

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) →
  (0 < a ∧ a < 1) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → (0 < a ∧ a < 1)) :=
by sorry

end proposition_relation_l2782_278296


namespace interest_rate_calculation_l2782_278299

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 320
    and the compound interest for 2 years is 340, then the interest rate is 12.5% per annum. -/
theorem interest_rate_calculation (P R : ℝ) 
  (h_simple : (P * R * 2) / 100 = 320)
  (h_compound : P * ((1 + R / 100)^2 - 1) = 340) :
  R = 12.5 := by
sorry

end interest_rate_calculation_l2782_278299


namespace equation_solution_l2782_278246

theorem equation_solution : ∃ x : ℝ, (16 : ℝ) ^ (2 * x - 3) = (1 / 2 : ℝ) ^ (x + 8) ↔ x = 4 / 9 := by
  sorry

end equation_solution_l2782_278246


namespace prob_one_tail_theorem_l2782_278257

/-- The probability of getting exactly one tail in 5 flips of a biased coin -/
def prob_one_tail_in_five_flips (p : ℝ) : ℝ :=
  5 * p * (1 - p)^4

/-- Theorem: The probability of getting exactly one tail in 5 flips of a biased coin -/
theorem prob_one_tail_theorem (p q : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_sum : p + q = 1) :
  prob_one_tail_in_five_flips p = 5 * p * q^4 :=
sorry

end prob_one_tail_theorem_l2782_278257


namespace dress_hemming_time_l2782_278241

/-- The time required to hem a dress given its length, stitch size, and stitching rate -/
theorem dress_hemming_time 
  (dress_length : ℝ) 
  (stitch_length : ℝ) 
  (stitches_per_minute : ℝ) 
  (h1 : dress_length = 3) -- dress length in feet
  (h2 : stitch_length = 1/4 / 12) -- stitch length in feet (1/4 inch converted to feet)
  (h3 : stitches_per_minute = 24) :
  dress_length / (stitch_length * stitches_per_minute) = 6 := by
  sorry

end dress_hemming_time_l2782_278241


namespace min_cubes_needed_l2782_278221

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := ℕ → ℕ → ℕ → Bool

/-- Checks if a cube is present at the given coordinates -/
def has_cube (grid : CubeGrid) (x y z : ℕ) : Prop := grid x y z = true

/-- Checks if the grid satisfies the adjacency condition -/
def satisfies_adjacency (grid : CubeGrid) : Prop :=
  ∀ x y z, has_cube grid x y z →
    (has_cube grid (x+1) y z ∨ has_cube grid (x-1) y z ∨
     has_cube grid x (y+1) z ∨ has_cube grid x (y-1) z ∨
     has_cube grid x y (z+1) ∨ has_cube grid x y (z-1))

/-- Checks if the grid matches the given front view -/
def matches_front_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 0 1 0) ∧ (has_cube grid 0 2 0) ∧
  (has_cube grid 1 0 0) ∧ (has_cube grid 1 1 0) ∧
  (has_cube grid 2 0 0) ∧ (has_cube grid 2 1 0)

/-- Checks if the grid matches the given side view -/
def matches_side_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 1 0 0) ∧ (has_cube grid 2 0 0) ∧
  (has_cube grid 2 0 1) ∧
  (has_cube grid 2 0 2)

/-- Counts the number of cubes in the grid -/
def count_cubes (grid : CubeGrid) : ℕ :=
  sorry -- Implementation omitted

/-- The main theorem to be proved -/
theorem min_cubes_needed :
  ∃ (grid : CubeGrid),
    satisfies_adjacency grid ∧
    matches_front_view grid ∧
    matches_side_view grid ∧
    count_cubes grid = 5 ∧
    (∀ (other_grid : CubeGrid),
      satisfies_adjacency other_grid →
      matches_front_view other_grid →
      matches_side_view other_grid →
      count_cubes other_grid ≥ 5) :=
  sorry

end min_cubes_needed_l2782_278221


namespace min_value_x_plus_2y_l2782_278272

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z : ℝ, z = x + 2*y → z ≥ 4 ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' + 2*y' + 2*x'*y' = 8 ∧ x' + 2*y' = 4 :=
sorry

end min_value_x_plus_2y_l2782_278272


namespace T_is_three_rays_with_common_point_l2782_278205

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

-- Define what it means for a set to be three rays with a common point
def is_three_rays_with_common_point (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, ∃ r₁ r₂ r₃ : Set (ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    r₁ ∩ r₂ = {p} ∧ r₁ ∩ r₃ = {p} ∧ r₂ ∩ r₃ = {p} ∧
    (∀ q ∈ r₁, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (0, -1)) ∧
    (∀ q ∈ r₂, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (-1, 0)) ∧
    (∀ q ∈ r₃, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (1, 1))

-- State the theorem
theorem T_is_three_rays_with_common_point : is_three_rays_with_common_point T := by
  sorry

end T_is_three_rays_with_common_point_l2782_278205


namespace complex_simplification_and_multiplication_l2782_278285

/-- The imaginary unit -/
def i : ℂ := Complex.I

theorem complex_simplification_and_multiplication :
  ((6 - 3 * i) - (2 - 5 * i)) * (1 + 2 * i) = 10 * i := by sorry

end complex_simplification_and_multiplication_l2782_278285


namespace max_distance_covered_l2782_278224

/-- The maximum distance a person can cover in 6 hours, 
    given that they travel at 5 km/hr for half the distance 
    and 4 km/hr for the other half. -/
theorem max_distance_covered (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 6 →
  speed1 = 5 →
  speed2 = 4 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 120 / 9 := by
  sorry

end max_distance_covered_l2782_278224


namespace abc_sum_sqrt_l2782_278284

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 20) 
  (h2 : c + a = 22) 
  (h3 : a + b = 24) : 
  Real.sqrt (2 * a * b * c * (a + b + c)) = 1287 := by
  sorry

end abc_sum_sqrt_l2782_278284


namespace race_result_l2782_278258

/-- Represents a participant in the race -/
structure Participant where
  position : ℝ
  speed : ℝ

/-- The race setup -/
structure Race where
  distance : ℝ
  a : Participant
  b : Participant
  c : Participant

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 60 ∧
  r.a.position = r.distance ∧
  r.b.position = r.distance - 10 ∧
  r.c.position = r.distance - 20

/-- Theorem stating the result of the race -/
theorem race_result (r : Race) :
  race_conditions r →
  (r.distance / r.b.speed - r.distance / r.c.speed) * r.c.speed = 12 :=
by
  sorry

end race_result_l2782_278258


namespace min_values_xy_and_x_plus_y_l2782_278238

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : 
  (x * y ≥ 4) ∧ (x + y ≥ 9/2) := by
  sorry

end min_values_xy_and_x_plus_y_l2782_278238


namespace balloon_count_l2782_278217

/-- The number of gold balloons -/
def gold_balloons : ℕ := sorry

/-- The number of silver balloons -/
def silver_balloons : ℕ := sorry

/-- The number of black balloons -/
def black_balloons : ℕ := 150

theorem balloon_count : 
  (silver_balloons = 2 * gold_balloons) ∧ 
  (gold_balloons + silver_balloons + black_balloons = 573) → 
  gold_balloons = 141 :=
by sorry

end balloon_count_l2782_278217


namespace exactly_five_triangles_l2782_278218

/-- A triangle with integral side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  sum_eq_8 : a + b + c = 8
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

/-- Count of distinct triangles with perimeter 8 -/
def count_triangles : ℕ := sorry

/-- The main theorem stating there are exactly 5 such triangles -/
theorem exactly_five_triangles : count_triangles = 5 := by sorry

end exactly_five_triangles_l2782_278218


namespace girls_divisible_by_nine_l2782_278220

theorem girls_divisible_by_nine (N : Nat) (m c d u : Nat) : 
  N < 10000 →
  N = 1000 * m + 100 * c + 10 * d + u →
  let B := m + c + d + u
  let G := N - B
  G % 9 = 0 := by
sorry

end girls_divisible_by_nine_l2782_278220


namespace work_completion_time_l2782_278214

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end work_completion_time_l2782_278214


namespace hip_hop_class_cost_l2782_278274

/-- The cost of one hip-hop class -/
def hip_hop_cost : ℕ := sorry

/-- The cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- The cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- The total number of hip-hop classes per week -/
def hip_hop_classes : ℕ := 2

/-- The total number of ballet classes per week -/
def ballet_classes : ℕ := 2

/-- The total number of jazz classes per week -/
def jazz_classes : ℕ := 1

/-- The total cost of all classes per week -/
def total_cost : ℕ := 52

theorem hip_hop_class_cost :
  hip_hop_cost * hip_hop_classes + ballet_cost * ballet_classes + jazz_cost * jazz_classes = total_cost ∧
  hip_hop_cost = 10 :=
sorry

end hip_hop_class_cost_l2782_278274


namespace rosie_pies_theorem_l2782_278281

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end rosie_pies_theorem_l2782_278281


namespace special_function_value_l2782_278203

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 :=
by
  sorry

end special_function_value_l2782_278203


namespace card_sum_problem_l2782_278294

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end card_sum_problem_l2782_278294


namespace pattern_36_l2782_278251

-- Define a function that represents the pattern
def f (n : ℕ) : ℕ :=
  if n = 1 then 6
  else if n ≤ 5 then 360 + n
  else 3600 + n

-- State the theorem
theorem pattern_36 : f 36 = 3636 := by
  sorry

end pattern_36_l2782_278251


namespace fencing_cost_theorem_l2782_278275

/-- Represents a pentagon with side lengths and angles -/
structure Pentagon where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ae : ℝ
  angle_bac : ℝ
  angle_abc : ℝ
  angle_bcd : ℝ
  angle_cde : ℝ
  angle_dea : ℝ

/-- Calculate the total cost of fencing a pentagon -/
def fencingCost (p : Pentagon) (costPerMeter : ℝ) : ℝ :=
  (p.ab + p.bc + p.cd + p.de + p.ae) * costPerMeter

/-- Theorem: The total cost of fencing the given irregular pentagon is Rs. 300 -/
theorem fencing_cost_theorem (p : Pentagon) (h1 : p.ab = 20)
    (h2 : p.bc = 25) (h3 : p.cd = 30) (h4 : p.de = 35) (h5 : p.ae = 40)
    (h6 : p.angle_bac = 110) (h7 : p.angle_abc = 95) (h8 : p.angle_bcd = 100)
    (h9 : p.angle_cde = 105) (h10 : p.angle_dea = 115) :
    fencingCost p 2 = 300 := by
  sorry

#check fencing_cost_theorem

end fencing_cost_theorem_l2782_278275


namespace both_runners_in_picture_probability_l2782_278253

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Calculates the probability of both runners being in a picture -/
def probability_both_in_picture (rachel : Runner) (robert : Runner) : ℚ :=
  sorry

/-- Main theorem: The probability of both runners being in the picture is 3/16 -/
theorem both_runners_in_picture_probability :
  let rachel : Runner := { lapTime := 90, direction := true }
  let robert : Runner := { lapTime := 80, direction := false }
  probability_both_in_picture rachel robert = 3 / 16 := by sorry

end both_runners_in_picture_probability_l2782_278253


namespace max_andy_cookies_l2782_278256

def total_cookies : ℕ := 30

def valid_distribution (andy_cookies : ℕ) : Prop :=
  andy_cookies + 3 * andy_cookies ≤ total_cookies

theorem max_andy_cookies :
  ∃ (max : ℕ), valid_distribution max ∧
    ∀ (n : ℕ), valid_distribution n → n ≤ max :=
by
  sorry

end max_andy_cookies_l2782_278256


namespace fraction_equals_91_when_x_is_3_l2782_278288

theorem fraction_equals_91_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
sorry

end fraction_equals_91_when_x_is_3_l2782_278288


namespace bananas_per_friend_l2782_278297

/-- Given Virginia has 40 bananas and shares them equally among 40 friends,
    prove that each friend receives 1 banana. -/
theorem bananas_per_friend (total_bananas : ℕ) (num_friends : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_friends = 40) :
  total_bananas / num_friends = 1 := by
  sorry

end bananas_per_friend_l2782_278297


namespace inscribed_square_area_ratio_l2782_278247

/-- The ratio of the area of a square inscribed in an ellipse to the area of a square inscribed in a circle -/
theorem inscribed_square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse_square_area := (4 * a^2 * b^2) / (a^2 + b^2)
  let circle_square_area := 2 * b^2
  ellipse_square_area / circle_square_area = 2 * a^2 / (a^2 + b^2) := by
sorry

end inscribed_square_area_ratio_l2782_278247


namespace club_officer_selection_ways_l2782_278210

def club_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_alice_bob : ℕ := 28 * 27 * 26 * 25
def ways_with_alice_bob : ℕ := 4 * 3 * 28 * 27

theorem club_officer_selection_ways :
  (ways_without_alice_bob + ways_with_alice_bob) = 500472 := by
  sorry

end club_officer_selection_ways_l2782_278210


namespace parcel_weight_sum_l2782_278259

/-- Given three parcels with weights x, y, and z, prove that their total weight is 209 pounds. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 146)
  (h3 : z + x = 140) : 
  x + y + z = 209 := by
  sorry

end parcel_weight_sum_l2782_278259


namespace jeff_matches_won_l2782_278239

/-- Represents the duration of the tennis competition in minutes -/
def total_playtime : ℕ := 225

/-- Represents the time in minutes it takes Jeff to score a point -/
def minutes_per_point : ℕ := 7

/-- Represents the minimum number of points required to win a match -/
def points_to_win : ℕ := 12

/-- Represents the break time in minutes between matches -/
def break_time : ℕ := 5

/-- Calculates the total number of points Jeff scored during the competition -/
def total_points : ℕ := total_playtime / minutes_per_point

/-- Calculates the duration of a single match in minutes, including playtime and break time -/
def match_duration : ℕ := points_to_win * minutes_per_point + break_time

/-- Represents the number of matches Jeff won during the competition -/
def matches_won : ℕ := total_playtime / match_duration

theorem jeff_matches_won : matches_won = 2 := by sorry

end jeff_matches_won_l2782_278239


namespace range_of_a_l2782_278237

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + 5 * y = 6 * a) ∧ (2 * x + 6 * y = 3 * a + 3)

-- Define the constraint
def constraint (x y : ℝ) : Prop :=
  x - y > 0

-- Theorem statement
theorem range_of_a (x y a : ℝ) :
  system x y a → constraint x y → a > 1 :=
by sorry

end range_of_a_l2782_278237


namespace line_y_intercept_l2782_278271

/-- A line with slope -3 and x-intercept (7,0) has y-intercept (0, 21) -/
theorem line_y_intercept (f : ℝ → ℝ) (h1 : ∀ x y, f y - f x = -3 * (y - x)) 
  (h2 : f 7 = 0) : f 0 = 21 := by
  sorry

end line_y_intercept_l2782_278271


namespace weekly_earnings_calculation_l2782_278289

/- Define the basic fees and attendance -/
def kidFee : ℚ := 3
def adultFee : ℚ := 6
def weekdayKids : ℕ := 8
def weekdayAdults : ℕ := 10
def weekendKids : ℕ := 12
def weekendAdults : ℕ := 15

/- Define the discounts and special rates -/
def weekendRate : ℚ := 1.5
def groupDiscountRate : ℚ := 0.8
def membershipDiscountRate : ℚ := 0.9
def weekdayGroupBookings : ℕ := 2
def weekendMemberships : ℕ := 8

/- Calculate earnings -/
def weekdayEarnings : ℚ := 5 * (weekdayKids * kidFee + weekdayAdults * adultFee)
def weekendEarnings : ℚ := 2 * (weekendKids * kidFee * weekendRate + weekendAdults * adultFee * weekendRate)

/- Calculate discounts -/
def weekdayGroupDiscount : ℚ := 5 * weekdayGroupBookings * (kidFee + adultFee) * (1 - groupDiscountRate)
def weekendMembershipDiscount : ℚ := 2 * weekendMemberships * adultFee * weekendRate * (1 - membershipDiscountRate)

/- Define the total weekly earnings -/
def totalWeeklyEarnings : ℚ := weekdayEarnings + weekendEarnings - weekdayGroupDiscount - weekendMembershipDiscount

/- The theorem to prove -/
theorem weekly_earnings_calculation : totalWeeklyEarnings = 738.6 := by
  sorry


end weekly_earnings_calculation_l2782_278289


namespace yellow_balls_count_l2782_278245

theorem yellow_balls_count (red white : ℕ) (a : ℝ) :
  red = 2 →
  white = 4 →
  (a / (red + white + a) = 1 / 4) →
  a = 2 := by
sorry

end yellow_balls_count_l2782_278245


namespace no_even_rectangle_with_sum_120_l2782_278234

/-- Represents a rectangle with positive even integer side lengths -/
structure EvenRectangle where
  length : ℕ
  width : ℕ
  length_positive : length > 0
  width_positive : width > 0
  length_even : Even length
  width_even : Even width

/-- Calculates the area of an EvenRectangle -/
def area (r : EvenRectangle) : ℕ := r.length * r.width

/-- Calculates the modified perimeter of an EvenRectangle -/
def modifiedPerimeter (r : EvenRectangle) : ℕ := 2 * (r.length + r.width) + 6

/-- Theorem stating that there's no EvenRectangle with A + P' = 120 -/
theorem no_even_rectangle_with_sum_120 :
  ∀ r : EvenRectangle, area r + modifiedPerimeter r ≠ 120 := by
  sorry

end no_even_rectangle_with_sum_120_l2782_278234


namespace third_consecutive_odd_integer_l2782_278244

theorem third_consecutive_odd_integer (x : ℤ) : 
  (∀ n : ℤ, (x + 2*n) % 2 ≠ 0) →  -- x is odd
  3*x = 2*(x + 4) + 3 →          -- condition from the problem
  x + 4 = 15 :=                  -- third integer is 15
by sorry

end third_consecutive_odd_integer_l2782_278244


namespace probability_no_shaded_correct_l2782_278295

/-- Represents a 2 by 1001 rectangle with middle squares shaded --/
structure ShadedRectangle where
  width : Nat
  height : Nat
  shaded_column : Nat

/-- The probability of choosing a rectangle without a shaded square --/
def probability_no_shaded (r : ShadedRectangle) : ℚ :=
  500 / 1001

/-- Theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded_correct (r : ShadedRectangle) 
  (h1 : r.width = 1001) 
  (h2 : r.height = 2) 
  (h3 : r.shaded_column = 501) : 
  probability_no_shaded r = 500 / 1001 := by
  sorry

end probability_no_shaded_correct_l2782_278295


namespace pasture_rent_is_175_l2782_278248

/-- Represents the rent share of a person based on their oxen and months of grazing -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the rent shares and one known payment -/
def calculateTotalRent (shares : List RentShare) (knownShare : RentShare) (knownPayment : ℕ) : ℕ :=
  let totalOxenMonths := shares.foldl (fun acc s => acc + s.oxen * s.months) 0
  let knownShareOxenMonths := knownShare.oxen * knownShare.months
  (totalOxenMonths * knownPayment) / knownShareOxenMonths

/-- Theorem: The total rent of the pasture is 175 given the problem conditions -/
theorem pasture_rent_is_175 :
  let shares := [
    RentShare.mk 10 7,  -- A's share
    RentShare.mk 12 5,  -- B's share
    RentShare.mk 15 3   -- C's share
  ]
  let knownShare := RentShare.mk 15 3  -- C's share
  let knownPayment := 45  -- C's payment
  calculateTotalRent shares knownShare knownPayment = 175 := by
  sorry


end pasture_rent_is_175_l2782_278248


namespace two_numbers_sum_product_difference_l2782_278212

theorem two_numbers_sum_product_difference (n : ℕ) (hn : n = 38) :
  ∃ x y : ℕ,
    1 ≤ x ∧ x < y ∧ y ≤ n ∧
    (n * (n + 1)) / 2 - x - y = x * y ∧
    y - x = 39 := by
  sorry

end two_numbers_sum_product_difference_l2782_278212


namespace complex_argument_range_l2782_278260

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
  k * π + π / 2 - Real.arccos (3 / 4) / 2 ≤ Complex.arg z ∧
  Complex.arg z ≤ k * π + π / 2 + Real.arccos (3 / 4) / 2 := by
  sorry

end complex_argument_range_l2782_278260


namespace right_triangle_perimeter_hypotenuse_ratio_l2782_278254

theorem right_triangle_perimeter_hypotenuse_ratio 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 3*x + 3*y
  let b := 4*x
  let c := 4*y
  let perimeter := a + b + c
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) →
  (perimeter / a = 7/3 ∨ perimeter / b = 56/25 ∨ perimeter / c = 56/25) :=
by sorry

end right_triangle_perimeter_hypotenuse_ratio_l2782_278254


namespace ratio_pq_qr_l2782_278228

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the points P, Q, and R on the circle
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distance between two points
def distance : Point → Point → ℝ := sorry

-- Define the length of an arc
def arcLength : Point → Point → ℝ := sorry

-- State the theorem
theorem ratio_pq_qr (h1 : distance P Q = distance P R)
                    (h2 : distance P Q > radius)
                    (h3 : arcLength Q R = 2 * Real.pi) :
  distance P Q / arcLength Q R = 2 / Real.pi := by
  sorry

end ratio_pq_qr_l2782_278228


namespace wire_division_l2782_278282

/-- Given a wire of length 28 cm divided into quarters, prove that each quarter is 7 cm long. -/
theorem wire_division (wire_length : ℝ) (h : wire_length = 28) :
  wire_length / 4 = 7 := by
sorry

end wire_division_l2782_278282


namespace rectangular_parallelepiped_volume_l2782_278240

/-- The volume of a rectangular parallelepiped -/
def volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a rectangular parallelepiped with width 15 cm, length 6 cm, and height 4 cm is 360 cubic centimeters -/
theorem rectangular_parallelepiped_volume :
  volume 15 6 4 = 360 := by
  sorry

end rectangular_parallelepiped_volume_l2782_278240


namespace digit_equation_solution_l2782_278233

theorem digit_equation_solution :
  ∀ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 →
    100 * A + 10 * B + C = 3 * (A + B + C) + 294 →
    (A + B + C) * (100 * A + 10 * B + C) = 2295 →
    A = 3 := by
  sorry

end digit_equation_solution_l2782_278233


namespace quadratic_max_value_l2782_278206

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_max_value (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, f x m ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 3, f x m = 1) → 
  m = -2 := by
sorry

end quadratic_max_value_l2782_278206


namespace rectangle_measurement_error_l2782_278291

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.05 * L) * ((1 - p) * W) = (1 + 0.008) * (L * W) → p = 0.04 := by
  sorry

end rectangle_measurement_error_l2782_278291


namespace certain_number_problem_l2782_278264

theorem certain_number_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = ((x + 40 + 25) / 3 + 5) → x = 10 := by
  sorry

end certain_number_problem_l2782_278264


namespace quadratic_factorization_l2782_278229

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 16*x + 60 = (x - a)*(x - b)) : 
  3*b - a = 8 := by sorry

end quadratic_factorization_l2782_278229


namespace dispatch_plans_eq_180_l2782_278262

/-- Represents the number of male officials -/
def num_males : ℕ := 5

/-- Represents the number of female officials -/
def num_females : ℕ := 3

/-- Represents the total number of officials -/
def total_officials : ℕ := num_males + num_females

/-- Represents the minimum number of officials in each group -/
def min_group_size : ℕ := 3

/-- Calculates the number of ways to divide officials into two groups -/
def dispatch_plans : ℕ := sorry

/-- Theorem stating that the number of dispatch plans is 180 -/
theorem dispatch_plans_eq_180 : dispatch_plans = 180 := by sorry

end dispatch_plans_eq_180_l2782_278262


namespace inequality_proof_l2782_278261

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 :=
by sorry

end inequality_proof_l2782_278261


namespace greatest_number_under_150_with_odd_factors_l2782_278292

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_under_150_with_odd_factors : 
  (∀ n : ℕ, n < 150 ∧ has_odd_number_of_factors n → n ≤ 144) ∧ 
  144 < 150 ∧ 
  has_odd_number_of_factors 144 :=
sorry

end greatest_number_under_150_with_odd_factors_l2782_278292


namespace bottle_production_l2782_278286

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 12 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 270)
  (time : ℕ)
  (h3 : time = 4) :
  (12 * bottles_per_minute * time) / machines = 2160 :=
sorry

end bottle_production_l2782_278286


namespace choose_starters_with_triplet_l2782_278213

/-- The number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose 7 starters from 16 players with at least one triplet -/
def ways_to_choose_starters : ℕ := 9721

/-- Theorem stating that the number of ways to choose 7 starters from 16 players,
    including a set of triplets, such that at least one of the triplets is in the
    starting lineup, is equal to 9721 -/
theorem choose_starters_with_triplet :
  (Nat.choose num_triplets 1 * Nat.choose (total_players - num_triplets) (num_starters - 1) +
   Nat.choose num_triplets 2 * Nat.choose (total_players - num_triplets) (num_starters - 2) +
   Nat.choose num_triplets 3 * Nat.choose (total_players - num_triplets) (num_starters - 3)) =
  ways_to_choose_starters :=
by sorry

end choose_starters_with_triplet_l2782_278213


namespace frequency_20_plus_l2782_278269

-- Define the sample size
def sample_size : ℕ := 35

-- Define the frequencies for each interval
def freq_5_10 : ℕ := 5
def freq_10_15 : ℕ := 12
def freq_15_20 : ℕ := 7
def freq_20_25 : ℕ := 5
def freq_25_30 : ℕ := 4
def freq_30_35 : ℕ := 2

-- Theorem to prove
theorem frequency_20_plus (h : freq_5_10 + freq_10_15 + freq_15_20 + freq_20_25 + freq_25_30 + freq_30_35 = sample_size) :
  (freq_20_25 + freq_25_30 + freq_30_35 : ℚ) / sample_size = 11 / 35 := by
  sorry

end frequency_20_plus_l2782_278269


namespace sum_of_square_areas_l2782_278268

/-- Given three squares with side lengths satisfying certain conditions, 
    prove that the sum of their areas is 189. -/
theorem sum_of_square_areas (x a b : ℝ) 
  (h1 : a + b + x = 23)
  (h2 : 9 ≤ (min a b)^2)
  (h3 : (min a b)^2 ≤ 25)
  (h4 : max a b ≥ 5) :
  x^2 + a^2 + b^2 = 189 := by
  sorry

end sum_of_square_areas_l2782_278268


namespace triangle_angle_c_l2782_278279

theorem triangle_angle_c (a b : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 6 →
  b = 2 * Real.sqrt 3 →
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C = π / 2 := by
  sorry

end triangle_angle_c_l2782_278279


namespace ice_cube_ratio_l2782_278231

def ice_cubes_in_glass : ℕ := 8
def number_of_trays : ℕ := 2
def spaces_per_tray : ℕ := 12

def total_ice_cubes : ℕ := number_of_trays * spaces_per_tray
def ice_cubes_in_pitcher : ℕ := total_ice_cubes - ice_cubes_in_glass

theorem ice_cube_ratio :
  (ice_cubes_in_pitcher : ℚ) / ice_cubes_in_glass = 2 / 1 := by sorry

end ice_cube_ratio_l2782_278231


namespace apples_to_oranges_ratio_l2782_278225

/-- Represents the number of fruits of each type on the display -/
structure FruitDisplay where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit display -/
def validFruitDisplay (d : FruitDisplay) : Prop :=
  d.oranges = 2 * d.bananas ∧
  d.bananas = 5 ∧
  d.apples + d.oranges + d.bananas = 35

/-- Theorem stating that for a valid fruit display, the ratio of apples to oranges is 2:1 -/
theorem apples_to_oranges_ratio (d : FruitDisplay) (h : validFruitDisplay d) :
  d.apples * 1 = d.oranges * 2 := by
  sorry

end apples_to_oranges_ratio_l2782_278225


namespace probability_closer_to_center_l2782_278215

/-- The probability that a randomly chosen point in a circle of radius 3 
    is closer to the center than to the boundary -/
theorem probability_closer_to_center (r : ℝ) (h : r = 3) : 
  (π * (r/2)^2) / (π * r^2) = 1/4 := by
  sorry

end probability_closer_to_center_l2782_278215


namespace work_for_series_springs_l2782_278223

/-- Work required to stretch a system of two springs in series -/
theorem work_for_series_springs (k₁ k₂ : ℝ) (x : ℝ) (h₁ : k₁ = 6000) (h₂ : k₂ = 12000) (h₃ : x = 0.1) :
  (1 / 2) * (1 / (1 / k₁ + 1 / k₂)) * x^2 = 20 := by
  sorry

#check work_for_series_springs

end work_for_series_springs_l2782_278223


namespace contrapositive_of_zero_product_l2782_278266

theorem contrapositive_of_zero_product (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) →
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end contrapositive_of_zero_product_l2782_278266


namespace no_matching_roots_l2782_278280

theorem no_matching_roots : ∀ x : ℝ,
  (x^2 - 4*x + 3 = 0) → 
  ¬(∃ y : ℝ, (y = x - 1 ∧ y = x - 3)) :=
by sorry

end no_matching_roots_l2782_278280


namespace airplane_seats_l2782_278222

theorem airplane_seats : ∀ s : ℕ,
  s ≥ 30 →
  (30 : ℝ) + 0.4 * s + (3/5) * s ≤ s →
  s = 150 :=
by
  sorry

end airplane_seats_l2782_278222


namespace num_arrangements_l2782_278278

/-- Represents the number of volunteers --/
def num_volunteers : ℕ := 4

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Calculates the number of ways to arrange volunteers into communities --/
def arrange_volunteers : ℕ := sorry

/-- Theorem stating that the number of arrangements is 36 --/
theorem num_arrangements : arrange_volunteers = 36 := by sorry

end num_arrangements_l2782_278278


namespace g_is_odd_l2782_278216

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end g_is_odd_l2782_278216


namespace divisor_of_p_l2782_278207

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 100 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 150) :
  7 ∣ p := by
  sorry

end divisor_of_p_l2782_278207


namespace correct_mean_problem_l2782_278276

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 50 41 23 48 = 41.5 := by
  sorry

end correct_mean_problem_l2782_278276


namespace bill_sunday_run_l2782_278219

/-- Represents the miles run by Bill, Julia, and Mark over a weekend -/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  markSaturday : ℝ
  markSunday : ℝ

/-- Conditions for the weekend run -/
def weekendRunConditions (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = 2 * run.billSunday ∧
  run.markSaturday = 5 ∧
  run.markSunday = run.markSaturday + 2 ∧
  run.billSaturday + run.billSunday + run.juliaSunday + run.markSaturday + run.markSunday = 50

/-- Theorem stating that under the given conditions, Bill ran 10.5 miles on Sunday -/
theorem bill_sunday_run (run : WeekendRun) (h : weekendRunConditions run) : 
  run.billSunday = 10.5 := by
  sorry


end bill_sunday_run_l2782_278219


namespace circumcircle_equation_l2782_278232

/-- Given a triangle ABC with sides defined by the equations:
    BC: x cos θ₁ + y sin θ₁ - p₁ = 0
    CA: x cos θ₂ + y sin θ₂ - p₂ = 0
    AB: x cos θ₃ + y sin θ₃ - p₃ = 0
    This theorem states that any point P(x, y) on the circumcircle of ABC
    satisfies the given equation. -/
theorem circumcircle_equation (θ₁ θ₂ θ₃ p₁ p₂ p₃ x y : ℝ) :
  (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * Real.sin (θ₂ - θ₃) +
  (x * Real.cos θ₃ + y * Real.sin θ₃ - p₃) * (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * Real.sin (θ₃ - θ₁) +
  (x * Real.cos θ₁ + y * Real.sin θ₁ - p₁) * (x * Real.cos θ₂ + y * Real.sin θ₂ - p₂) * Real.sin (θ₁ - θ₂) = 0 :=
by sorry

end circumcircle_equation_l2782_278232
