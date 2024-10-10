import Mathlib

namespace value_of_x_l3532_353200

theorem value_of_x : ∀ (w y z x : ℤ), 
  w = 50 → 
  z = w + 25 → 
  y = z + 15 → 
  x = y + 7 → 
  x = 97 := by
  sorry

end value_of_x_l3532_353200


namespace art_class_earnings_l3532_353252

def art_class_problem (price_per_class : ℚ) (saturday_attendance : ℕ) : ℚ :=
  let sunday_attendance := saturday_attendance / 2
  let total_attendance := saturday_attendance + sunday_attendance
  price_per_class * total_attendance

theorem art_class_earnings :
  art_class_problem 10 20 = 300 :=
by
  sorry

end art_class_earnings_l3532_353252


namespace money_distribution_l3532_353286

/-- Given three people A, B, and C with a total of 1000 rupees between them,
    where B and C together have 600 rupees, and C has 300 rupees,
    prove that A and C together have 700 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 1000 →
  B + C = 600 →
  C = 300 →
  A + C = 700 := by
  sorry

end money_distribution_l3532_353286


namespace ball_bounce_distance_l3532_353285

/-- Calculate the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 150
  let reboundFactor : ℝ := 3/4
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 765.703125 := by
  sorry

end ball_bounce_distance_l3532_353285


namespace mark_and_carolyn_money_sum_l3532_353208

theorem mark_and_carolyn_money_sum : 
  (5 : ℚ) / 6 + (2 : ℚ) / 5 = (37 : ℚ) / 30 := by sorry

end mark_and_carolyn_money_sum_l3532_353208


namespace kyles_speed_l3532_353297

theorem kyles_speed (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (distance_difference : ℝ) :
  joseph_speed = 50 →
  joseph_time = 2.5 →
  kyle_time = 2 →
  distance_difference = 1 →
  joseph_speed * joseph_time = kyle_time * (joseph_speed * joseph_time - distance_difference) / kyle_time →
  (joseph_speed * joseph_time - distance_difference) / kyle_time = 62 :=
by
  sorry

#check kyles_speed

end kyles_speed_l3532_353297


namespace expression_evaluation_l3532_353284

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2 * y)^2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end expression_evaluation_l3532_353284


namespace range_of_a_l3532_353212

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * x^2 + a * x - a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a > 2 ∨ a < -2 := by
  sorry

end range_of_a_l3532_353212


namespace arithmetic_sequence_sum_l3532_353293

theorem arithmetic_sequence_sum (a₁ a₄ a₁₀ : ℚ) (n : ℕ) : 
  a₁ = -3 → a₄ = 4 → a₁₀ = 40 → n = 10 → 
  (n : ℚ) / 2 * (a₁ + a₁₀) = 285 := by sorry

end arithmetic_sequence_sum_l3532_353293


namespace maple_taller_than_pine_l3532_353227

-- Define the heights of the trees
def pine_height : ℚ := 15 + 1/4
def maple_height : ℚ := 20 + 2/3

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem maple_taller_than_pine :
  height_difference = 5 + 5/12 := by sorry

end maple_taller_than_pine_l3532_353227


namespace larger_solution_of_quadratic_l3532_353273

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 19*x - 48 = 0 → x ≤ 24 :=
by
  sorry

end larger_solution_of_quadratic_l3532_353273


namespace percentage_reduction_optimal_price_increase_l3532_353283

-- Define the original price
def original_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the initial profit per kilogram
def initial_profit : ℝ := 10

-- Define the initial daily sales
def initial_sales : ℝ := 500

-- Define the sales decrease per yuan of price increase
def sales_decrease_rate : ℝ := 20

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Theorem for the percentage reduction
theorem percentage_reduction :
  ∃ (r : ℝ), r > 0 ∧ r < 1 ∧ original_price * (1 - r)^2 = final_price ∧ r = 0.2 := by sorry

-- Theorem for the optimal price increase
theorem optimal_price_increase :
  ∃ (x : ℝ), x > 0 ∧
    (initial_profit + x) * (initial_sales - sales_decrease_rate * x) = target_profit ∧
    x = 5 := by sorry

end percentage_reduction_optimal_price_increase_l3532_353283


namespace inequality_solution_l3532_353242

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (4 * x + 6) ≤ 5 ↔ x < -3/2 ∨ x > -1/8 := by
  sorry

end inequality_solution_l3532_353242


namespace identity_function_divisibility_l3532_353272

theorem identity_function_divisibility (f : ℕ → ℕ) :
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) →
  ∀ m : ℕ, f m = m :=
by sorry

end identity_function_divisibility_l3532_353272


namespace quadratic_equation_solution_l3532_353224

theorem quadratic_equation_solution (x₁ : ℚ) (h₁ : x₁ = 3/4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) : 
  ∃ x₂ : ℚ, x₂ = -31/6 ∧ 72 * x₂^2 + 39 * x₂ - 18 = 0 :=
by sorry

end quadratic_equation_solution_l3532_353224


namespace stratified_sampling_male_count_l3532_353229

theorem stratified_sampling_male_count :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (sample_size : ℕ),
  total_employees = 120 →
  female_employees = 72 →
  sample_size = 15 →
  (total_employees - female_employees) * sample_size / total_employees = 6 :=
by sorry

end stratified_sampling_male_count_l3532_353229


namespace intersection_of_M_and_N_l3532_353290

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_M_and_N_l3532_353290


namespace final_block_count_l3532_353287

theorem final_block_count :
  let initial_blocks : ℕ := 250
  let added_blocks : ℕ := 13
  let intermediate_blocks : ℕ := initial_blocks + added_blocks
  let doubling_factor : ℕ := 2
  let final_blocks : ℕ := intermediate_blocks * doubling_factor
  final_blocks = 526 := by sorry

end final_block_count_l3532_353287


namespace solution_is_two_l3532_353253

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (y : ℝ) : Prop :=
  lg (y - 1) - lg y = lg (2 * y - 2) - lg (y + 2)

-- Theorem statement
theorem solution_is_two :
  ∃ y : ℝ, y > 1 ∧ y + 2 > 0 ∧ equation y ∧ y = 2 :=
sorry

end solution_is_two_l3532_353253


namespace rectangle_side_lengths_l3532_353276

theorem rectangle_side_lengths :
  ∀ x y : ℝ,
  x > 0 →
  y > 0 →
  y = 2 * x →
  x * y = 2 * (x + y) →
  (x = 3 ∧ y = 6) :=
by
  sorry

end rectangle_side_lengths_l3532_353276


namespace bee_legs_count_l3532_353296

theorem bee_legs_count (legs_per_bee : ℕ) (num_bees : ℕ) (h : legs_per_bee = 6) :
  legs_per_bee * num_bees = 48 ↔ num_bees = 8 :=
by
  sorry

end bee_legs_count_l3532_353296


namespace actual_weight_of_three_bags_l3532_353236

/-- The actual weight of three bags of food given their labeled weight and deviations -/
theorem actual_weight_of_three_bags 
  (labeled_weight : ℕ) 
  (num_bags : ℕ) 
  (deviation1 deviation2 deviation3 : ℤ) : 
  labeled_weight = 200 → 
  num_bags = 3 → 
  deviation1 = 10 → 
  deviation2 = -16 → 
  deviation3 = -11 → 
  (labeled_weight * num_bags : ℤ) + deviation1 + deviation2 + deviation3 = 583 := by
  sorry

end actual_weight_of_three_bags_l3532_353236


namespace sum_factorials_mod_15_l3532_353203

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_15 : sum_factorials 50 % 15 = 3 := by sorry

end sum_factorials_mod_15_l3532_353203


namespace shelbys_journey_l3532_353262

/-- Shelby's scooter journey with varying weather conditions -/
theorem shelbys_journey 
  (speed_sunny : ℝ) 
  (speed_rainy : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (break_time : ℝ)
  (h1 : speed_sunny = 40)
  (h2 : speed_rainy = 15)
  (h3 : total_distance = 20)
  (h4 : total_time = 50)
  (h5 : break_time = 5) :
  ∃ (rainy_time : ℝ),
    rainy_time = 24 ∧
    speed_sunny * (total_time - rainy_time - break_time) / 60 + 
    speed_rainy * rainy_time / 60 = total_distance :=
by sorry

end shelbys_journey_l3532_353262


namespace correct_sampling_methods_l3532_353222

-- Define the types of sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Define a situation
structure Situation where
  description : String
  populationSize : Nat
  sampleSize : Nat

-- Define a function to determine the appropriate sampling method
def appropriateSamplingMethod (s : Situation) : SamplingMethod :=
  sorry

-- Define the three situations
def situation1 : Situation :=
  { description := "Selecting 2 students from each class"
  , populationSize := 0  -- We don't know the exact population size
  , sampleSize := 2 }

def situation2 : Situation :=
  { description := "Selecting 12 students from a class with different score ranges"
  , populationSize := 62  -- 10 + 40 + 12
  , sampleSize := 12 }

def situation3 : Situation :=
  { description := "Arranging tracks for 6 students in a 400m final"
  , populationSize := 6
  , sampleSize := 6 }

-- Theorem stating the correct sampling methods for each situation
theorem correct_sampling_methods :
  (appropriateSamplingMethod situation1 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod situation2 = SamplingMethod.Stratified) ∧
  (appropriateSamplingMethod situation3 = SamplingMethod.SimpleRandom) :=
  sorry

end correct_sampling_methods_l3532_353222


namespace expression_equality_l3532_353279

theorem expression_equality : 4 + 3/10 + 9/1000 = 4.309 := by
  sorry

end expression_equality_l3532_353279


namespace three_planes_intersection_count_l3532_353221

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Represents the intersection of two planes -/
inductive PlanesIntersection
  | Line
  | Empty

/-- Represents the number of intersection lines between three planes -/
inductive IntersectionCount
  | One
  | Three

/-- Function to determine how two planes intersect -/
def planesIntersect (p1 p2 : Plane3D) : PlanesIntersection :=
  sorry

/-- 
Given three planes in 3D space that intersect each other pairwise,
prove that the number of their intersection lines is either 1 or 3.
-/
theorem three_planes_intersection_count 
  (p1 p2 p3 : Plane3D)
  (h12 : planesIntersect p1 p2 = PlanesIntersection.Line)
  (h23 : planesIntersect p2 p3 = PlanesIntersection.Line)
  (h31 : planesIntersect p3 p1 = PlanesIntersection.Line) :
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.One ∨ count = IntersectionCount.Three) :=
by
  sorry

end three_planes_intersection_count_l3532_353221


namespace tetrahedron_volume_not_unique_l3532_353267

/-- Represents a tetrahedron with face areas and circumradius -/
structure Tetrahedron where
  face_areas : Fin 4 → ℝ
  circumradius : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas and circumradius -/
theorem tetrahedron_volume_not_unique : ∃ (t1 t2 : Tetrahedron), 
  (∀ i : Fin 4, t1.face_areas i = t2.face_areas i) ∧ 
  t1.circumradius = t2.circumradius ∧ 
  volume t1 ≠ volume t2 :=
sorry

end tetrahedron_volume_not_unique_l3532_353267


namespace solve_for_a_l3532_353259

theorem solve_for_a (y : ℝ) (h1 : y > 0) (h2 : (a * y) / 20 + (3 * y) / 10 = 0.7 * y) : a = 8 := by
  sorry

end solve_for_a_l3532_353259


namespace section_through_center_l3532_353280

-- Define a cube
def Cube := Set (ℝ × ℝ × ℝ)

-- Define a plane section
def PlaneSection := Set (ℝ × ℝ × ℝ)

-- Define the center of a cube
def centerOfCube (c : Cube) : ℝ × ℝ × ℝ := sorry

-- Define the volume of a set in 3D space
def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define what it means for a plane to pass through a point
def passesThrough (p : PlaneSection) (point : ℝ × ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem section_through_center (c : Cube) (s : PlaneSection) :
  (∃ (A B : Set (ℝ × ℝ × ℝ)), A ∪ B = c ∧ A ∩ B = s ∧ volume A = volume B) →
  passesThrough s (centerOfCube c) := by sorry

end section_through_center_l3532_353280


namespace max_m_value_l3532_353218

theorem max_m_value (A B C D : ℝ × ℝ) (m : ℝ) : 
  A = (1, 0) →
  B = (0, 1) →
  C = (a, b) →
  D = (c, d) →
  (∀ a b c d : ℝ, 
    (c - a)^2 + (d - b)^2 ≥ 
    (m - 2) * (a * c + b * d) + 
    m * (a * 0 + b * 1) * (c * 1 + d * 0)) →
  ∃ m_max : ℝ, m_max = Real.sqrt 5 - 1 ∧ 
    (∀ m' : ℝ, (∀ a b c d : ℝ, 
      (c - a)^2 + (d - b)^2 ≥ 
      (m' - 2) * (a * c + b * d) + 
      m' * (a * 0 + b * 1) * (c * 1 + d * 0)) → 
    m' ≤ m_max) :=
by sorry

end max_m_value_l3532_353218


namespace vector_parallel_condition_l3532_353226

/-- Given plane vectors a, b, and c, if a + 2b is parallel to c, then the x-coordinate of c is -2. -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) :
  a = (3, 1) →
  b = (-1, 1) →
  c.2 = -6 →
  (∃ (k : ℝ), k • (a + 2 • b) = c) →
  c.1 = -2 := by
  sorry

end vector_parallel_condition_l3532_353226


namespace hypotenuse_length_l3532_353278

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  -- Side lengths
  a : ℝ  -- Length of the side opposite to the 30° angle
  b : ℝ  -- Length of the side opposite to the 60° angle
  c : ℝ  -- Length of the hypotenuse (opposite to the 90° angle)
  -- Properties of a 30-60-90 triangle
  h1 : a = c / 2
  h2 : b = a * Real.sqrt 3

/-- Theorem: In a 30-60-90 triangle with side length opposite to 60° angle equal to 12, 
    the length of the hypotenuse is 8√3 -/
theorem hypotenuse_length (t : Triangle30_60_90) (h : t.b = 12) : t.c = 8 * Real.sqrt 3 := by
  sorry


end hypotenuse_length_l3532_353278


namespace min_period_tan_2x_l3532_353246

/-- The minimum positive period of the function y = tan 2x is π/2 -/
theorem min_period_tan_2x : 
  let f : ℝ → ℝ := λ x => Real.tan (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧ 
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
    p = π / 2 :=
by sorry

end min_period_tan_2x_l3532_353246


namespace ratio_antecedent_l3532_353261

theorem ratio_antecedent (ratio_a ratio_b consequent : ℚ) : 
  ratio_a / ratio_b = 4 / 6 →
  consequent = 45 →
  ratio_a / ratio_b = ratio_a / consequent →
  ratio_a = 30 := by
  sorry

end ratio_antecedent_l3532_353261


namespace fraction_sum_equality_l3532_353255

theorem fraction_sum_equality : (3 : ℚ) / 8 + 9 / 12 - 1 / 6 = 23 / 24 := by sorry

end fraction_sum_equality_l3532_353255


namespace initial_test_count_l3532_353238

theorem initial_test_count (initial_avg : ℝ) (improved_avg : ℝ) (lowest_score : ℝ) :
  initial_avg = 35 →
  improved_avg = 40 →
  lowest_score = 20 →
  ∃ n : ℕ,
    n > 1 ∧
    (n : ℝ) * initial_avg = ((n : ℝ) - 1) * improved_avg + lowest_score ∧
    n = 4 :=
by sorry

end initial_test_count_l3532_353238


namespace two_digit_reverse_divisible_by_11_l3532_353211

theorem two_digit_reverse_divisible_by_11 (a b : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) : 
  (1000 * a + 100 * b + 10 * b + a) % 11 = 0 := by
  sorry

end two_digit_reverse_divisible_by_11_l3532_353211


namespace natural_number_representation_with_distinct_powers_l3532_353232

theorem natural_number_representation_with_distinct_powers : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧ N = b₁ - b₂ ∧ N = c₁ - c₂ ∧ N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry


end natural_number_representation_with_distinct_powers_l3532_353232


namespace only_prime_perfect_square_l3532_353294

theorem only_prime_perfect_square : 
  ∀ p : ℕ, Prime p → (∃ k : ℕ, 5^p + 12^p = k^2) → p = 2 :=
by sorry

end only_prime_perfect_square_l3532_353294


namespace books_added_by_marta_l3532_353240

def initial_books : ℕ := 38
def final_books : ℕ := 48

theorem books_added_by_marta : 
  final_books - initial_books = 10 := by sorry

end books_added_by_marta_l3532_353240


namespace quadratic_inequality_range_l3532_353217

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≥ 1 := by
  sorry

end quadratic_inequality_range_l3532_353217


namespace rectangle_area_l3532_353266

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 246) : L * B = 3650 := by
  sorry

end rectangle_area_l3532_353266


namespace no_real_roots_iff_m_zero_l3532_353275

theorem no_real_roots_iff_m_zero (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) ↔ m = 0 := by
  sorry

end no_real_roots_iff_m_zero_l3532_353275


namespace travel_time_for_a_l3532_353205

/-- Represents the travel time of a person given their relative speed and time difference from a reference traveler -/
def travelTime (relativeSpeed : ℚ) (timeDiff : ℚ) : ℚ :=
  (4 : ℚ) / 3 * ((3 : ℚ) / 2 + timeDiff)

theorem travel_time_for_a (speedRatio : ℚ) (timeDiffHours : ℚ) 
    (h1 : speedRatio = 3 / 4) 
    (h2 : timeDiffHours = 1 / 2) : 
  travelTime speedRatio timeDiffHours = 2 := by
  sorry

#eval travelTime (3/4) (1/2)

end travel_time_for_a_l3532_353205


namespace two_intersecting_lines_determine_plane_l3532_353295

-- Define the basic types
def Point : Type := sorry
def Line : Type := sorry
def Plane : Type := sorry

-- Define the axioms of solid geometry (focusing on Axiom 3)
axiom intersecting_lines (l1 l2 : Line) : Prop
axiom determine_plane (l1 l2 : Line) (p : Plane) : Prop

-- Axiom 3: Two intersecting lines determine a plane
axiom axiom_3 (l1 l2 : Line) (p : Plane) : 
  intersecting_lines l1 l2 → determine_plane l1 l2 p

-- Theorem to prove
theorem two_intersecting_lines_determine_plane (l1 l2 : Line) :
  intersecting_lines l1 l2 → ∃ p : Plane, determine_plane l1 l2 p :=
sorry

end two_intersecting_lines_determine_plane_l3532_353295


namespace arithmetic_series_sum_l3532_353228

/-- 
Given an arithmetic series of consecutive integers with first term (k^2 - k + 1),
prove that the sum of the first (k + 2) terms is equal to k^3 + (3k^2)/2 + k/2 + 2.
-/
theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℚ := k^2 - k + 1
  let n : ℕ := k + 2
  let S := (n : ℚ) / 2 * (a₁ + (a₁ + (n - 1)))
  S = k^3 + (3 * k^2) / 2 + k / 2 + 2 := by
  sorry


end arithmetic_series_sum_l3532_353228


namespace sugar_profit_problem_l3532_353264

theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_18_percent : ℝ) 
  (overall_profit_percent : ℝ) (profit_18_percent : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percent = 14 →
  profit_18_percent = 18 →
  ∃ (remaining_profit_percent : ℝ),
    remaining_profit_percent = 8 ∧
    (sugar_at_18_percent * (1 + profit_18_percent / 100) + 
     (total_sugar - sugar_at_18_percent) * (1 + remaining_profit_percent / 100)) / total_sugar
    = 1 + overall_profit_percent / 100 :=
by sorry

end sugar_profit_problem_l3532_353264


namespace cubic_root_identity_l3532_353281

theorem cubic_root_identity (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 := by
  sorry

end cubic_root_identity_l3532_353281


namespace first_digit_853_base8_l3532_353233

/-- The first digit of the base 8 representation of a natural number -/
def firstDigitBase8 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 8).succ
    (n / 8^(k-1)) % 8

theorem first_digit_853_base8 :
  firstDigitBase8 853 = 1 := by
sorry

end first_digit_853_base8_l3532_353233


namespace combined_selling_price_l3532_353269

/-- Calculate the combined selling price of three articles given their cost prices and profit/loss percentages. -/
theorem combined_selling_price (cost1 cost2 cost3 : ℝ) : 
  cost1 = 70 →
  cost2 = 120 →
  cost3 = 150 →
  ∃ (sell1 sell2 sell3 : ℝ),
    (2/3 * sell1 = 0.85 * cost1) ∧
    (sell2 = cost2 * 1.3) ∧
    (sell3 = cost3 * 0.8) ∧
    (sell1 + sell2 + sell3 = 365.25) := by
  sorry

#check combined_selling_price

end combined_selling_price_l3532_353269


namespace cube_volume_from_surface_area_l3532_353201

-- Define the surface area of the cube
def surface_area : ℝ := 1350

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    side_length > 0 ∧
    side_length^3 = 3375 := by
  sorry


end cube_volume_from_surface_area_l3532_353201


namespace tunneled_cube_surface_area_l3532_353209

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure TunneledCube where
  sideLength : ℝ
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculates the surface area of a tunneled cube -/
def surfaceArea (cube : TunneledCube) : ℝ := sorry

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem tunneled_cube_surface_area :
  ∃ (cube : TunneledCube) (u v w : ℕ),
    cube.sideLength = 10 ∧
    surfaceArea cube = u + v * Real.sqrt w ∧
    isSquareFree w ∧
    u + v + w = 472 := by sorry

end tunneled_cube_surface_area_l3532_353209


namespace animal_ages_sum_l3532_353256

/-- Represents the ages of the animals in the problem -/
structure AnimalAges where
  porcupine : ℕ
  owl : ℕ
  lion : ℕ

/-- Defines the conditions given in the problem -/
def valid_ages (ages : AnimalAges) : Prop :=
  ages.owl = 2 * ages.porcupine ∧
  ages.owl = ages.lion + 2 ∧
  ages.lion = ages.porcupine + 4

/-- The theorem to be proved -/
theorem animal_ages_sum (ages : AnimalAges) :
  valid_ages ages → ages.porcupine + ages.owl + ages.lion = 28 := by
  sorry


end animal_ages_sum_l3532_353256


namespace tims_soda_cans_l3532_353215

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by sorry

end tims_soda_cans_l3532_353215


namespace car_sale_profit_percentage_l3532_353204

/-- Calculate the net profit percentage for a car sale --/
theorem car_sale_profit_percentage 
  (purchase_price : ℝ) 
  (repair_cost_percentage : ℝ) 
  (sales_tax_percentage : ℝ) 
  (registration_fee_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000)
  (h2 : repair_cost_percentage = 0.35)
  (h3 : sales_tax_percentage = 0.08)
  (h4 : registration_fee_percentage = 0.06)
  (h5 : selling_price = 64900) :
  let total_cost := purchase_price * (1 + repair_cost_percentage + sales_tax_percentage + registration_fee_percentage)
  let net_profit := selling_price - total_cost
  let net_profit_percentage := (net_profit / total_cost) * 100
  ∃ ε > 0, |net_profit_percentage - 3.71| < ε :=
by sorry

end car_sale_profit_percentage_l3532_353204


namespace fly_probabilities_l3532_353245

def fly_path (n m : ℕ) : ℕ := Nat.choose (n + m) n

theorem fly_probabilities :
  let p1 := (fly_path 8 10 : ℚ) / 2^18
  let p2 := ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18
  let p3 := (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
             2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
             (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18
  (p1 = (fly_path 8 10 : ℚ) / 2^18) ∧
  (p2 = ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18) ∧
  (p3 = (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
         2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
         (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18) :=
by sorry

end fly_probabilities_l3532_353245


namespace cube_surface_area_l3532_353231

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 600 := by
sorry

end cube_surface_area_l3532_353231


namespace librarian_took_books_l3532_353271

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  total_books = 34 →
  books_per_shelf = 3 →
  shelves_needed = 9 →
  total_books - (books_per_shelf * shelves_needed) = 7 := by
sorry

end librarian_took_books_l3532_353271


namespace impossible_coin_probabilities_l3532_353225

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end impossible_coin_probabilities_l3532_353225


namespace work_completion_time_a_completion_time_l3532_353207

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 6

/-- The time it takes for worker b to complete the remaining work after both workers work for 1 day -/
def b_remaining_time : ℝ := 2.0000000000000004

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 2

theorem work_completion_time :
  (1 / a_time + 1 / b_time) + b_remaining_time / b_time = 1 := by sorry

theorem a_completion_time : a_time = 2 := by sorry

end work_completion_time_a_completion_time_l3532_353207


namespace monday_distance_l3532_353257

/-- Debby's jogging distances over three days -/
structure JoggingDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  total : ℝ

/-- The jogging distances satisfy the given conditions -/
def satisfies_conditions (d : JoggingDistances) : Prop :=
  d.tuesday = 5 ∧ d.wednesday = 9 ∧ d.total = 16 ∧ d.monday + d.tuesday + d.wednesday = d.total

/-- Theorem: Debby jogged 2 kilometers on Monday -/
theorem monday_distance (d : JoggingDistances) (h : satisfies_conditions d) : d.monday = 2 := by
  sorry

end monday_distance_l3532_353257


namespace triangle_two_solutions_l3532_353249

theorem triangle_two_solutions (a b : ℝ) (A B : ℝ) :
  b = 2 →
  B = π / 4 →
  (∃ (C : ℝ), 0 < C ∧ C < π ∧ A + B + C = π ∧ a / Real.sin A = b / Real.sin B) →
  (∃ (C' : ℝ), 0 < C' ∧ C' < π ∧ C' ≠ C ∧ A + B + C' = π ∧ a / Real.sin A = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end triangle_two_solutions_l3532_353249


namespace quadratic_one_root_l3532_353237

/-- A quadratic function with coefficients a = 1, b = 4, and c = n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + n

/-- The discriminant of the quadratic function f -/
def discriminant (n : ℝ) : ℝ := 4^2 - 4*1*n

theorem quadratic_one_root (n : ℝ) :
  (∃! x, f n x = 0) ↔ n = 4 := by sorry

end quadratic_one_root_l3532_353237


namespace isosceles_if_root_one_equilateral_roots_l3532_353250

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 - 2 * t.c * x + (t.a - t.b)

theorem isosceles_if_root_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.c :=
sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t 1 = 0) :=
sorry

end isosceles_if_root_one_equilateral_roots_l3532_353250


namespace lcm_gcf_problem_l3532_353219

theorem lcm_gcf_problem (n m : ℕ) (h1 : Nat.lcm n m = 48) (h2 : Nat.gcd n m = 18) (h3 : m = 16) : n = 54 := by
  sorry

end lcm_gcf_problem_l3532_353219


namespace smallest_multiplier_for_cube_l3532_353288

theorem smallest_multiplier_for_cube (n : ℕ) : 
  (∀ m : ℕ, m < 300 → ¬∃ k : ℕ, 720 * m = k^3) ∧ 
  (∃ k : ℕ, 720 * 300 = k^3) := by
  sorry

end smallest_multiplier_for_cube_l3532_353288


namespace tangent_sum_given_ratio_l3532_353251

theorem tangent_sum_given_ratio (α : Real) :
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3 →
  Real.tan (α + π/4) = -3 := by
  sorry

end tangent_sum_given_ratio_l3532_353251


namespace ellipse_max_min_sum_absolute_values_l3532_353244

theorem ellipse_max_min_sum_absolute_values :
  ∀ x y : ℝ, x^2/4 + y^2/9 = 1 →
  (∃ a b : ℝ, a^2/4 + b^2/9 = 1 ∧ |a| + |b| = 3) ∧
  (∃ c d : ℝ, c^2/4 + d^2/9 = 1 ∧ |c| + |d| = 2) ∧
  (∀ z w : ℝ, z^2/4 + w^2/9 = 1 → |z| + |w| ≤ 3 ∧ |z| + |w| ≥ 2) :=
sorry

end ellipse_max_min_sum_absolute_values_l3532_353244


namespace initial_production_was_200_l3532_353210

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The number of cars cut from production due to metal shortages -/
def cars_cut : ℕ := 50

/-- The fraction of remaining production after pandemic cuts -/
def production_fraction : ℚ := 1/2

/-- The final number of doors produced -/
def final_doors : ℕ := 375

/-- Theorem stating that the initial planned production was 200 cars -/
theorem initial_production_was_200 : 
  ∃ (initial_cars : ℕ), 
    (doors_per_car : ℚ) * production_fraction * (initial_cars - cars_cut) = final_doors ∧ 
    initial_cars = 200 := by
  sorry

end initial_production_was_200_l3532_353210


namespace lucy_groceries_l3532_353220

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end lucy_groceries_l3532_353220


namespace smallest_multiple_of_7_4_5_l3532_353254

theorem smallest_multiple_of_7_4_5 : ∃ n : ℕ+, (∀ m : ℕ+, m.val % 7 = 0 ∧ m.val % 4 = 0 ∧ m.val % 5 = 0 → n ≤ m) ∧ n.val % 7 = 0 ∧ n.val % 4 = 0 ∧ n.val % 5 = 0 := by
  sorry

end smallest_multiple_of_7_4_5_l3532_353254


namespace sufficient_not_necessary_l3532_353289

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
  sorry

end sufficient_not_necessary_l3532_353289


namespace optical_mice_ratio_l3532_353241

theorem optical_mice_ratio (total_mice : ℕ) (trackball_mice : ℕ) : 
  total_mice = 80 →
  trackball_mice = 20 →
  (total_mice / 2 : ℚ) = total_mice / 2 →
  (total_mice - total_mice / 2 - trackball_mice : ℚ) / total_mice = 1 / 4 :=
by sorry

end optical_mice_ratio_l3532_353241


namespace arithmetic_progression_polynomial_j_value_l3532_353234

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j*x^2 + k*x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end arithmetic_progression_polynomial_j_value_l3532_353234


namespace larger_integer_is_fifteen_l3532_353298

theorem larger_integer_is_fifteen (a b : ℤ) : 
  (a : ℚ) / b = 1 / 3 → 
  (a + 10 : ℚ) / b = 1 → 
  b = 15 := by
sorry

end larger_integer_is_fifteen_l3532_353298


namespace polynomial_simplification_l3532_353282

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x^6 + x^5 + 3*x^4 + 2*x^2 + 15
def q (x : ℝ) : ℝ := x^6 + x^5 + 4*x^4 - x^3 + x^2 + 18
def r (x : ℝ) : ℝ := x^6 - x^4 + x^3 + x^2 - 3

-- State the theorem
theorem polynomial_simplification (x : ℝ) : p x - q x = r x := by
  sorry

end polynomial_simplification_l3532_353282


namespace gcd_g_x_l3532_353270

def g (x : ℤ) : ℤ := (3*x+8)*(5*x+1)*(11*x+6)*(2*x+3)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 12096 * k) : 
  Int.gcd (g x) x = 144 := by
  sorry

end gcd_g_x_l3532_353270


namespace equation_solution_unique_l3532_353214

theorem equation_solution_unique :
  ∃! (x y : ℝ), x ≥ 2 ∧ y ≥ 1 ∧
  36 * Real.sqrt (x - 2) + 4 * Real.sqrt (y - 1) = 28 - 4 * Real.sqrt (x - 2) - Real.sqrt (y - 1) ∧
  x = 5 ∧ y = 3 := by
  sorry

end equation_solution_unique_l3532_353214


namespace ticket_cost_difference_l3532_353247

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  (num_adults * adult_ticket_price) - (num_children * child_ticket_price) = 50 := by
sorry

end ticket_cost_difference_l3532_353247


namespace clean_80_cars_per_day_l3532_353274

/-- Represents the Super Clean Car Wash Company's operations -/
structure CarWash where
  price_per_car : ℕ  -- Price per car in dollars
  total_revenue : ℕ  -- Total revenue in dollars
  num_days : ℕ       -- Number of days

/-- Calculates the number of cars cleaned per day -/
def cars_per_day (cw : CarWash) : ℕ :=
  (cw.total_revenue / cw.price_per_car) / cw.num_days

/-- Theorem stating that the number of cars cleaned per day is 80 -/
theorem clean_80_cars_per_day (cw : CarWash)
  (h1 : cw.price_per_car = 5)
  (h2 : cw.total_revenue = 2000)
  (h3 : cw.num_days = 5) :
  cars_per_day cw = 80 := by
  sorry

#eval cars_per_day ⟨5, 2000, 5⟩

end clean_80_cars_per_day_l3532_353274


namespace princes_wish_fulfilled_l3532_353263

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- Represents the state of the round table -/
def RoundTable := List Knight

/-- Checks if two knights from the same city have gold goblets -/
def sameCity_haveGold (table : RoundTable) : Bool := sorry

/-- Rotates the goblets one position to the right -/
def rotateGoblets (table : RoundTable) : RoundTable := sorry

theorem princes_wish_fulfilled 
  (initial_table : RoundTable)
  (h1 : initial_table.length = 13)
  (h2 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.filter Knight.hasGoldGoblet).length = k)
  (h3 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity_haveGold (n.iterate rotateGoblets initial_table) := by
  sorry

end princes_wish_fulfilled_l3532_353263


namespace fraction_equality_l3532_353260

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 := by
sorry

end fraction_equality_l3532_353260


namespace money_distribution_l3532_353265

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem money_distribution (s : Share) : 
  s.b = 0.65 * s.a → 
  s.c = 0.4 * s.a → 
  s.c = 56 → 
  s.a + s.b + s.c = 287 := by
  sorry


end money_distribution_l3532_353265


namespace celia_savings_l3532_353239

def weekly_food_budget : ℕ := 100
def num_weeks : ℕ := 4
def monthly_rent : ℕ := 1500
def monthly_streaming : ℕ := 30
def monthly_cell_phone : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weekly_food_budget * num_weeks + monthly_rent + monthly_streaming + monthly_cell_phone

def savings : ℚ := (total_spending : ℚ) * savings_rate

theorem celia_savings : savings = 198 := by sorry

end celia_savings_l3532_353239


namespace range_of_g_l3532_353248

theorem range_of_g (x : ℝ) : 
  -(1/4) ≤ Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ∧ 
  Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ≤ 3/4 := by
sorry

end range_of_g_l3532_353248


namespace infinite_centers_of_symmetry_l3532_353223

/-- A type representing a geometric figure. -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in the figure. -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a symmetry operation on a figure. -/
def SymmetryOperation : Type := Figure → Figure

/-- Represents a center of symmetry for a figure. -/
def CenterOfSymmetry (f : Figure) : Type := Point

/-- Composition of symmetry operations. -/
def composeSymmetry (s1 s2 : SymmetryOperation) : SymmetryOperation :=
  fun f => s1 (s2 f)

/-- 
  If a figure has more than one center of symmetry, 
  it must have infinitely many centers of symmetry.
-/
theorem infinite_centers_of_symmetry (f : Figure) :
  (∃ (c1 c2 : CenterOfSymmetry f), c1 ≠ c2) →
  ∀ n : ℕ, ∃ (centers : Finset (CenterOfSymmetry f)), centers.card > n :=
sorry

end infinite_centers_of_symmetry_l3532_353223


namespace probability_point_between_C_and_D_l3532_353258

/-- Given points A, B, C, and D on a line segment AB where AB = 4AD and AB = 3BC,
    the probability that a randomly selected point on AB is between C and D is 5/12. -/
theorem probability_point_between_C_and_D 
  (A B C D : ℝ) 
  (h_order : A ≤ C ∧ C ≤ D ∧ D ≤ B) 
  (h_AB_4AD : B - A = 4 * (D - A))
  (h_AB_3BC : B - A = 3 * (B - C)) : 
  (D - C) / (B - A) = 5 / 12 := by
  sorry

end probability_point_between_C_and_D_l3532_353258


namespace mike_fish_per_hour_l3532_353230

/-- Represents the number of fish Mike can catch in one hour -/
def M : ℕ := sorry

/-- The number of fish Jim can catch in one hour -/
def jim_catch : ℕ := 2 * M

/-- The number of fish Bob can catch in one hour -/
def bob_catch : ℕ := 3 * M

/-- The total number of fish caught by all three in 40 minutes -/
def total_40min : ℕ := (2 * M / 3) + (4 * M / 3) + (2 * M)

/-- The number of fish Jim catches in the remaining 20 minutes -/
def jim_20min : ℕ := 2 * M / 3

/-- The total number of fish caught in one hour -/
def total_catch : ℕ := total_40min + jim_20min

theorem mike_fish_per_hour : 
  (total_catch = 140) → (M = 30) := by sorry

end mike_fish_per_hour_l3532_353230


namespace next_meeting_after_105_days_l3532_353268

/-- Represents the number of days between cinema visits for each boy -/
structure VisitIntervals :=
  (kolya : ℕ)
  (seryozha : ℕ)
  (vanya : ℕ)

/-- The least number of days after which all three boys meet at the cinema again -/
def nextMeeting (intervals : VisitIntervals) : ℕ :=
  Nat.lcm intervals.kolya (Nat.lcm intervals.seryozha intervals.vanya)

/-- Theorem stating that the next meeting of all three boys occurs after 105 days -/
theorem next_meeting_after_105_days :
  let intervals : VisitIntervals := { kolya := 3, seryozha := 7, vanya := 5 }
  nextMeeting intervals = 105 := by sorry

end next_meeting_after_105_days_l3532_353268


namespace greatest_number_in_set_l3532_353299

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (n : ℕ) (start : ℕ) : Set ℕ :=
  {x : ℕ | ∃ k : ℕ, k < n ∧ x = start + 2 * k}

theorem greatest_number_in_set (s : Set ℕ) :
  s = ConsecutiveMultiplesOf2 50 56 →
  ∃ m : ℕ, m ∈ s ∧ ∀ x ∈ s, x ≤ m ∧ m = 154 :=
by sorry

end greatest_number_in_set_l3532_353299


namespace machinery_expenditure_l3532_353216

theorem machinery_expenditure (C : ℝ) (h : C > 0) : 
  let raw_material_cost : ℝ := (1 / 4) * C
  let remaining_after_raw : ℝ := C - raw_material_cost
  let final_remaining : ℝ := 0.675 * C
  let machinery_cost : ℝ := remaining_after_raw - final_remaining
  machinery_cost / remaining_after_raw = 1 / 10 := by
sorry

end machinery_expenditure_l3532_353216


namespace rectangle_area_change_l3532_353235

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15% and
    its width is decreased by 15%, the new area will be 527.55 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h : l * w = 540) :
  (1.15 * l) * (0.85 * w) = 527.55 := by sorry

end rectangle_area_change_l3532_353235


namespace projection_matrix_values_l3532_353292

def isProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/36; c, 16/36]

theorem projection_matrix_values :
  ∀ a c : ℚ, isProjectionMatrix (P a c) → a = 1/27 ∧ c = 5/27 := by
  sorry

end projection_matrix_values_l3532_353292


namespace tangency_point_difference_l3532_353202

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  cyclic : True  -- Represents that the quadrilateral is cyclic
  inscribed : True  -- Represents that there's an inscribed circle

/-- The specific quadrilateral from the problem --/
def specificQuad : InscribedQuadrilateral where
  a := 60
  b := 110
  c := 140
  d := 90
  positive := by simp
  cyclic := True.intro
  inscribed := True.intro

/-- The point of tangency divides the side of length 140 into m and n --/
def tangencyPoint (q : InscribedQuadrilateral) : ℝ × ℝ :=
  sorry

/-- The theorem to prove --/
theorem tangency_point_difference (q : InscribedQuadrilateral) 
  (h : q = specificQuad) : 
  let (m, n) := tangencyPoint q
  |m - n| = 120 := by
  sorry

end tangency_point_difference_l3532_353202


namespace polynomial_evaluation_l3532_353206

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 :=
by sorry


end polynomial_evaluation_l3532_353206


namespace john_years_taking_pictures_l3532_353213

/-- Calculates the number of years John has been taking pictures given the following conditions:
  * John takes 10 pictures every day
  * Each memory card can store 50 images
  * Each memory card costs $60
  * John spent $13,140 on memory cards
-/
def years_taking_pictures (
  pictures_per_day : ℕ)
  (images_per_card : ℕ)
  (card_cost : ℕ)
  (total_spent : ℕ)
  : ℕ :=
  let cards_bought := total_spent / card_cost
  let total_images := cards_bought * images_per_card
  let days_taking_pictures := total_images / pictures_per_day
  days_taking_pictures / 365

theorem john_years_taking_pictures :
  years_taking_pictures 10 50 60 13140 = 3 := by
  sorry

end john_years_taking_pictures_l3532_353213


namespace min_rectangles_cover_square_l3532_353291

/-- The smallest number of 2-by-3 non-overlapping rectangles needed to cover a 12-by-12 square exactly -/
def min_rectangles : ℕ := 24

/-- The side length of the square -/
def square_side : ℕ := 12

/-- The width of the rectangle -/
def rect_width : ℕ := 2

/-- The height of the rectangle -/
def rect_height : ℕ := 3

/-- The area of the square -/
def square_area : ℕ := square_side ^ 2

/-- The area of a single rectangle -/
def rect_area : ℕ := rect_width * rect_height

theorem min_rectangles_cover_square :
  min_rectangles * rect_area = square_area ∧
  ∃ (rows columns : ℕ),
    rows * columns = min_rectangles ∧
    rows * rect_height = square_side ∧
    columns * rect_width = square_side :=
by sorry

end min_rectangles_cover_square_l3532_353291


namespace angle_FAH_is_45_degrees_l3532_353277

/-- Given a unit square ABCD with EF parallel to AB, GH parallel to BC, BF = 1/4, and BF + DH = FH, 
    the measure of angle FAH is 45 degrees. -/
theorem angle_FAH_is_45_degrees (A B C D E F G H : ℝ × ℝ) : 
  -- Unit square ABCD
  A = (0, 1) ∧ B = (0, 0) ∧ C = (1, 0) ∧ D = (1, 1) →
  -- EF is parallel to AB
  (E.2 - F.2) / (E.1 - F.1) = (A.2 - B.2) / (A.1 - B.1) →
  -- GH is parallel to BC
  (G.2 - H.2) / (G.1 - H.1) = (B.2 - C.2) / (B.1 - C.1) →
  -- BF = 1/4
  F = (1/4, 0) →
  -- BF + DH = FH
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) + 
  Real.sqrt ((D.1 - H.1)^2 + (D.2 - H.2)^2) = 
  Real.sqrt ((F.1 - H.1)^2 + (F.2 - H.2)^2) →
  -- Angle FAH is 45 degrees
  Real.arctan (((A.2 - F.2) / (A.1 - F.1) - (A.2 - H.2) / (A.1 - H.1)) / 
    (1 + (A.2 - F.2) / (A.1 - F.1) * (A.2 - H.2) / (A.1 - H.1))) * (180 / Real.pi) = 45 := by
  sorry

end angle_FAH_is_45_degrees_l3532_353277


namespace modulo_equivalence_unique_solution_l3532_353243

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [ZMOD 15] ∧ n = 8 := by
  sorry

end modulo_equivalence_unique_solution_l3532_353243
