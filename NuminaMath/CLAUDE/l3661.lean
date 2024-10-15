import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3661_366116

/-- Represents a teacher with their name and number of created questions -/
structure Teacher where
  name : String
  questions : ℕ

/-- Represents the result of stratified sampling -/
structure SamplingResult where
  wu : ℕ
  wang : ℕ
  zhang : ℕ

/-- Calculates the number of questions selected for each teacher in stratified sampling -/
def stratifiedSampling (teachers : List Teacher) (totalSamples : ℕ) : SamplingResult :=
  sorry

/-- Calculates the probability of selecting at least one question from a specific teacher -/
def probabilityAtLeastOne (samplingResult : SamplingResult) (teacherQuestions : ℕ) (selectionSize : ℕ) : ℚ :=
  sorry

theorem stratified_sampling_theorem (wu wang zhang : Teacher) (h1 : wu.questions = 350) (h2 : wang.questions = 700) (h3 : zhang.questions = 1050) :
  let teachers := [wu, wang, zhang]
  let result := stratifiedSampling teachers 6
  result.wu = 1 ∧ result.wang = 2 ∧ result.zhang = 3 ∧
  probabilityAtLeastOne result result.wang 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3661_366116


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3661_366198

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3661_366198


namespace NUMINAMATH_CALUDE_concrete_mixture_percentage_l3661_366149

/-- Proves that mixing 7 tons of 80% cement mixture with 3 tons of 20% cement mixture
    results in a 62% cement mixture when making 10 tons of concrete. -/
theorem concrete_mixture_percentage : 
  let total_concrete : ℝ := 10
  let mixture_80_percent : ℝ := 7
  let mixture_20_percent : ℝ := total_concrete - mixture_80_percent
  let cement_in_80_percent : ℝ := mixture_80_percent * 0.8
  let cement_in_20_percent : ℝ := mixture_20_percent * 0.2
  let total_cement : ℝ := cement_in_80_percent + cement_in_20_percent
  total_cement / total_concrete = 0.62 := by
sorry

end NUMINAMATH_CALUDE_concrete_mixture_percentage_l3661_366149


namespace NUMINAMATH_CALUDE_determine_opposite_resident_l3661_366191

/-- Represents a resident on the hexagonal street -/
inductive Resident
| Knight
| Liar

/-- Represents a vertex of the hexagonal street -/
def Vertex := Fin 6

/-- Represents the street layout -/
structure HexagonalStreet where
  residents : Vertex → Resident

/-- Represents a letter asking about neighbor relationships -/
structure Letter where
  sender : Vertex
  recipient : Vertex
  askedAbout : Vertex

/-- Determines if two vertices are neighbors in a regular hexagon -/
def areNeighbors (v1 v2 : Vertex) : Bool :=
  (v1.val + 1) % 6 = v2.val ∨ (v1.val + 5) % 6 = v2.val

/-- The main theorem stating that it's possible to determine the opposite resident with at most 4 letters -/
theorem determine_opposite_resident (street : HexagonalStreet) (start : Vertex) :
  ∃ (letters : List Letter), letters.length ≤ 4 ∧
    ∃ (opposite : Vertex), (start.val + 3) % 6 = opposite.val ∧
      (∀ (response : Letter → Bool), 
        ∃ (deduced_resident : Resident), street.residents opposite = deduced_resident) :=
  sorry

end NUMINAMATH_CALUDE_determine_opposite_resident_l3661_366191


namespace NUMINAMATH_CALUDE_sum_of_squares_of_conjugates_l3661_366177

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_squares_of_conjugates : (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_conjugates_l3661_366177


namespace NUMINAMATH_CALUDE_proposition_falsity_l3661_366179

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 6) : 
  ¬ P 5 := by
sorry

end NUMINAMATH_CALUDE_proposition_falsity_l3661_366179


namespace NUMINAMATH_CALUDE_platform_length_l3661_366167

/-- Given a train that passes a pole and a platform, calculate the platform length -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 100 →
  pole_time = 15 →
  platform_time = 40 →
  ∃ (platform_length : ℝ),
    platform_length = 500 / 3 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3661_366167


namespace NUMINAMATH_CALUDE_runner_meetings_l3661_366142

/-- The number of meetings between two runners on a circular track -/
def number_of_meetings (speed1 speed2 : ℝ) (laps : ℕ) : ℕ :=
  sorry

/-- The theorem stating the number of meetings for the given problem -/
theorem runner_meetings :
  let speed1 := (4 : ℝ)
  let speed2 := (10 : ℝ)
  let laps := 28
  number_of_meetings speed1 speed2 laps = 77 :=
sorry

end NUMINAMATH_CALUDE_runner_meetings_l3661_366142


namespace NUMINAMATH_CALUDE_touchdown_points_l3661_366146

theorem touchdown_points : ℕ → Prop :=
  fun p =>
    let team_a_touchdowns : ℕ := 7
    let team_b_touchdowns : ℕ := 9
    let point_difference : ℕ := 14
    (team_b_touchdowns * p = team_a_touchdowns * p + point_difference) →
    p = 7

-- Proof
example : touchdown_points 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l3661_366146


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3661_366186

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 295 →
  train_speed_kmh = 75 →
  crossing_time = 45 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 642.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3661_366186


namespace NUMINAMATH_CALUDE_water_consumption_theorem_l3661_366137

/-- Calculates the number of glasses of water drunk per day given the bottle capacity,
    number of refills per week, glass size, and days in a week. -/
def glassesPerDay (bottleCapacity : ℕ) (refillsPerWeek : ℕ) (glassSize : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (bottleCapacity * refillsPerWeek) / (glassSize * daysPerWeek)

/-- Theorem stating that given the specified conditions, the number of glasses of water
    drunk per day is equal to 4. -/
theorem water_consumption_theorem :
  let bottleCapacity : ℕ := 35
  let refillsPerWeek : ℕ := 4
  let glassSize : ℕ := 5
  let daysPerWeek : ℕ := 7
  glassesPerDay bottleCapacity refillsPerWeek glassSize daysPerWeek = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_theorem_l3661_366137


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3661_366157

-- Define the ellipse and parabola equations
def ellipse (x y a : ℝ) : Prop := x^2 + 4*(y-a)^2 = 4
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the theorem
theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, ellipse x y a ∧ parabola x y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3661_366157


namespace NUMINAMATH_CALUDE_max_diff_color_triangles_17gon_l3661_366192

/-- Regular 17-gon with colored edges -/
structure ColoredPolygon where
  n : Nat
  colors : Nat
  no_monochromatic : Bool

/-- The number of edges in a regular 17-gon -/
def num_edges (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1)) / 2

/-- The total number of triangles in a regular 17-gon -/
def total_triangles (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1) * (p.n - 2)) / 6

/-- The minimum number of isosceles triangles (triangles with at least two sides of the same color) -/
def min_isosceles_triangles (p : ColoredPolygon) : Nat :=
  p.n * p.colors

/-- The maximum number of triangles with all edges of different colors -/
def max_diff_color_triangles (p : ColoredPolygon) : Nat :=
  total_triangles p - min_isosceles_triangles p

/-- Theorem: The maximum number of triangles with all edges of different colors in a regular 17-gon
    with 8 colors and no monochromatic triangles is 544 -/
theorem max_diff_color_triangles_17gon :
  ∀ p : ColoredPolygon,
    p.n = 17 →
    p.colors = 8 →
    p.no_monochromatic = true →
    num_edges p = 136 →
    max_diff_color_triangles p = 544 := by
  sorry

end NUMINAMATH_CALUDE_max_diff_color_triangles_17gon_l3661_366192


namespace NUMINAMATH_CALUDE_external_roads_different_colors_l3661_366175

/-- Represents a city with colored streets and intersections -/
structure ColoredCity where
  /-- Number of intersections in the city -/
  n : ℕ
  /-- Number of colors used for streets (assumed to be 3) -/
  num_colors : ℕ
  /-- Number of streets meeting at each intersection (assumed to be 3) -/
  streets_per_intersection : ℕ
  /-- Number of roads leading out of the city (assumed to be 3) -/
  num_external_roads : ℕ
  /-- Condition: Streets are colored using three colors -/
  h_num_colors : num_colors = 3
  /-- Condition: Exactly three streets meet at each intersection -/
  h_streets_per_intersection : streets_per_intersection = 3
  /-- Condition: Three roads lead out of the city -/
  h_num_external_roads : num_external_roads = 3

/-- Theorem: In a ColoredCity, the three roads leading out of the city have different colors -/
theorem external_roads_different_colors (city : ColoredCity) :
  ∃ (c₁ c₂ c₃ : ℕ), c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ ∧
  c₁ ≤ city.num_colors ∧ c₂ ≤ city.num_colors ∧ c₃ ≤ city.num_colors :=
sorry

end NUMINAMATH_CALUDE_external_roads_different_colors_l3661_366175


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3661_366147

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 70 → 
  E = 2 * F + 18 → 
  D + E + F = 180 → 
  F = 92 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3661_366147


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3661_366136

theorem trigonometric_identities (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  ((3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 8/9) ∧ 
  (Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 1 = 13/10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3661_366136


namespace NUMINAMATH_CALUDE_james_payment_l3661_366193

theorem james_payment (adoption_fee : ℝ) (friend_percentage : ℝ) (james_payment : ℝ) : 
  adoption_fee = 200 →
  friend_percentage = 0.25 →
  james_payment = adoption_fee - (adoption_fee * friend_percentage) →
  james_payment = 150 := by
sorry

end NUMINAMATH_CALUDE_james_payment_l3661_366193


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3661_366194

/-- For the quadratic x^2 - 24x + 60, when written as (x+b)^2 + c, b+c equals -96 -/
theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 60 = (x+b)^2 + c) → b + c = -96 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3661_366194


namespace NUMINAMATH_CALUDE_grass_seed_coverage_l3661_366171

/-- Calculates the area covered by one bag of grass seed given the dimensions of a rectangular lawn
and the total area covered by a known number of bags. -/
theorem grass_seed_coverage 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (extra_area : ℝ) 
  (num_bags : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : extra_area = 208)
  (h4 : num_bags = 4) :
  (lawn_length * lawn_width + extra_area) / num_bags = 250 :=
by sorry

end NUMINAMATH_CALUDE_grass_seed_coverage_l3661_366171


namespace NUMINAMATH_CALUDE_triangle_properties_l3661_366131

/-- Triangle ABC with vertices A(-1,4), B(-2,-1), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from BC in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 + p.2 - 3 = 0

/-- The area of triangle ABC -/
def area (t : Triangle) : ℝ := 8

theorem triangle_properties :
  let t : Triangle := { A := (-1, 4), B := (-2, -1), C := (2, 3) }
  (∀ p, altitude t p ↔ p.1 + p.2 - 3 = 0) ∧ area t = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3661_366131


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3661_366199

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x = 3) : 2*x^2 - 4*x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3661_366199


namespace NUMINAMATH_CALUDE_area_range_of_special_triangle_l3661_366100

/-- Given an acute triangle ABC where angles A, B, C form an arithmetic sequence
    and the side opposite to angle B has length √3, prove that the area S of the triangle
    satisfies √3/2 < S ≤ 3√3/4. -/
theorem area_range_of_special_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧  -- sum of angles in a triangle
  2 * B = A + C ∧  -- A, B, C form an arithmetic sequence
  b = Real.sqrt 3 ∧  -- side opposite to B has length √3
  S = (1 / 2) * a * c * Real.sin B ∧  -- area formula
  a * Real.sin B = b * Real.sin A ∧  -- sine law
  c * Real.sin B = b * Real.sin C  -- sine law
  →
  Real.sqrt 3 / 2 < S ∧ S ≤ 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_area_range_of_special_triangle_l3661_366100


namespace NUMINAMATH_CALUDE_no_solution_arccos_arcsin_l3661_366143

theorem no_solution_arccos_arcsin : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arccos_arcsin_l3661_366143


namespace NUMINAMATH_CALUDE_division_remainder_l3661_366101

theorem division_remainder :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 136 →
    divisor = 15 →
    quotient = 9 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3661_366101


namespace NUMINAMATH_CALUDE_smallest_solution_l3661_366174

def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6 ∧
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4)

theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 16 ∧ equation 16 := by sorry

end NUMINAMATH_CALUDE_smallest_solution_l3661_366174


namespace NUMINAMATH_CALUDE_stationery_shop_restocking_l3661_366139

/-- Calculates the total number of pencils and rulers after restocking in a stationery shop. -/
theorem stationery_shop_restocking
  (initial_pencils : ℕ)
  (initial_pens : ℕ)
  (initial_rulers : ℕ)
  (sold_pencils : ℕ)
  (sold_pens : ℕ)
  (given_rulers : ℕ)
  (pencil_restock_factor : ℕ)
  (ruler_restock_factor : ℕ)
  (h1 : initial_pencils = 112)
  (h2 : initial_pens = 78)
  (h3 : initial_rulers = 46)
  (h4 : sold_pencils = 32)
  (h5 : sold_pens = 56)
  (h6 : given_rulers = 12)
  (h7 : pencil_restock_factor = 5)
  (h8 : ruler_restock_factor = 3)
  : (initial_pencils - sold_pencils + pencil_restock_factor * (initial_pencils - sold_pencils)) +
    (initial_rulers - given_rulers + ruler_restock_factor * (initial_rulers - given_rulers)) = 616 := by
  sorry

#check stationery_shop_restocking

end NUMINAMATH_CALUDE_stationery_shop_restocking_l3661_366139


namespace NUMINAMATH_CALUDE_square_sum_fourth_powers_l3661_366150

theorem square_sum_fourth_powers (a b c : ℝ) 
  (h1 : a^2 - b^2 = 5)
  (h2 : a * b = 2)
  (h3 : a^2 + b^2 + c^2 = 8) :
  a^4 + b^4 + c^4 = 38 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fourth_powers_l3661_366150


namespace NUMINAMATH_CALUDE_smallest_number_divisible_when_increased_l3661_366124

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n % d = 0)

theorem smallest_number_divisible_when_increased : ∃! n : ℕ, 
  (is_divisible_by_all (n + 9) [8, 11, 24]) ∧ 
  (∀ m : ℕ, m < n → ¬ is_divisible_by_all (m + 9) [8, 11, 24]) ∧
  n = 255 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_when_increased_l3661_366124


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l3661_366162

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Assumption that AB + CD = 150
  sum_sides : ab + cd = 150
  -- Assumption that AB = 3CD
  ab_triple_cd : ab = 3 * cd
  -- Assumption that area ratio of ABC to ADC is 4:1
  area_ratio_def : area_ratio = 4 / 1

/-- Theorem stating that under given conditions, AB = 120 cm -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 120 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ab_length_l3661_366162


namespace NUMINAMATH_CALUDE_weekend_price_is_105_l3661_366127

def original_price : ℝ := 250
def sale_discount : ℝ := 0.4
def weekend_discount : ℝ := 0.3

def sale_price : ℝ := original_price * (1 - sale_discount)
def weekend_price : ℝ := sale_price * (1 - weekend_discount)

theorem weekend_price_is_105 : weekend_price = 105 := by sorry

end NUMINAMATH_CALUDE_weekend_price_is_105_l3661_366127


namespace NUMINAMATH_CALUDE_sin_two_alpha_on_line_l3661_366120

/-- Given an angle α where its terminal side intersects the line y = 2x, prove that sin(2α) = 4/5 -/
theorem sin_two_alpha_on_line (α : Real) : 
  (∃ (P : Real × Real), P.2 = 2 * P.1 ∧ P ≠ (0, 0) ∧ 
    P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧ 
    P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)) → 
  Real.sin (2 * α) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_two_alpha_on_line_l3661_366120


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_algebraic_expression_value_l3661_366159

-- Problem 1
theorem simplify_and_evaluate :
  let x : ℤ := -3
  (x^2 + 4*x - (2*x^2 - x + x^2) - (3*x - 1)) = -23 := by sorry

-- Problem 2
theorem algebraic_expression_value (m n : ℤ) 
  (h1 : m + n = 2) (h2 : m * n = -3) :
  2*(m*n + (-3*m)) - 3*(2*n - m*n) = -27 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_algebraic_expression_value_l3661_366159


namespace NUMINAMATH_CALUDE_cos_sin_75_deg_l3661_366125

theorem cos_sin_75_deg : Real.cos (75 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_75_deg_l3661_366125


namespace NUMINAMATH_CALUDE_ice_palace_steps_count_l3661_366182

/-- The number of steps in the Ice Palace staircase -/
def ice_palace_steps : ℕ := 30

/-- The time Alice takes to walk 20 steps (in seconds) -/
def time_for_20_steps : ℕ := 120

/-- The time Alice takes to walk all steps (in seconds) -/
def time_for_all_steps : ℕ := 180

/-- Theorem: The number of steps in the Ice Palace staircase is 30 -/
theorem ice_palace_steps_count :
  ice_palace_steps = (time_for_all_steps * 20) / time_for_20_steps :=
sorry

end NUMINAMATH_CALUDE_ice_palace_steps_count_l3661_366182


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3661_366168

theorem quadratic_equivalence (c : ℝ) : 
  ({a : ℝ | ∀ x : ℝ, x^2 + a*x + a/4 + 1/2 > 0} = {x : ℝ | x^2 - x + c < 0}) → 
  c = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3661_366168


namespace NUMINAMATH_CALUDE_maddie_had_15_books_l3661_366188

/-- The number of books Maddie had -/
def maddie_books : ℕ := sorry

/-- The number of books Luisa had -/
def luisa_books : ℕ := 18

/-- The number of books Amy had -/
def amy_books : ℕ := 6

/-- Theorem stating that Maddie had 15 books -/
theorem maddie_had_15_books : maddie_books = 15 := by
  have h1 : amy_books + luisa_books = maddie_books + 9 := sorry
  sorry

end NUMINAMATH_CALUDE_maddie_had_15_books_l3661_366188


namespace NUMINAMATH_CALUDE_sachin_age_l3661_366121

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 12 = rahul_age * 5) :
  sachin_age = 5 := by
sorry

end NUMINAMATH_CALUDE_sachin_age_l3661_366121


namespace NUMINAMATH_CALUDE_train_length_proof_l3661_366184

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 52 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 36 seconds,
    then the length of each train is 80 meters. -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) : 
  faster_speed = 52 →
  slower_speed = 36 →
  passing_time = 36 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 80 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l3661_366184


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l3661_366156

theorem max_value_of_product_sum (a b c : ℝ) (h : a + 3 * b + c = 5) :
  (∀ x y z : ℝ, x + 3 * y + z = 5 → a * b + a * c + b * c ≥ x * y + x * z + y * z) ∧
  a * b + a * c + b * c = 25 / 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l3661_366156


namespace NUMINAMATH_CALUDE_truthful_dwarfs_count_l3661_366107

-- Define the total number of dwarfs
def total_dwarfs : ℕ := 10

-- Define the number of dwarfs who raised hands for each ice cream type
def vanilla_hands : ℕ := total_dwarfs
def chocolate_hands : ℕ := total_dwarfs / 2
def fruit_hands : ℕ := 1

-- Define the total number of hands raised
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

-- Theorem to prove
theorem truthful_dwarfs_count : 
  ∃ (truthful : ℕ) (lying : ℕ), 
    truthful + lying = total_dwarfs ∧ 
    lying = total_hands_raised - total_dwarfs ∧
    truthful = 4 := by
  sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_count_l3661_366107


namespace NUMINAMATH_CALUDE_roberts_reading_l3661_366102

/-- Given Robert's reading rate and book length, calculate the maximum number of complete books he can read in a given time. -/
theorem roberts_reading (reading_rate : ℕ) (book_length : ℕ) (available_time : ℕ) :
  reading_rate > 0 →
  book_length > 0 →
  available_time > 0 →
  reading_rate = 120 →
  book_length = 360 →
  available_time = 8 →
  (available_time * reading_rate) / book_length = 2 :=
by sorry

end NUMINAMATH_CALUDE_roberts_reading_l3661_366102


namespace NUMINAMATH_CALUDE_boris_neighbors_l3661_366190

-- Define the type for people
inductive Person : Type
  | Arkady | Boris | Vera | Galya | Danya | Egor

-- Define the circle as a function from positions to people
def Circle := Fin 6 → Person

-- Define the conditions
def satisfies_conditions (c : Circle) : Prop :=
  -- Danya stands next to Vera, on her right side
  ∃ i, c i = Person.Vera ∧ c (i + 1) = Person.Danya
  -- Galya stands opposite Egor
  ∧ ∃ j, c j = Person.Egor ∧ c (j + 3) = Person.Galya
  -- Egor stands next to Danya
  ∧ ∃ k, c k = Person.Danya ∧ (c (k + 1) = Person.Egor ∨ c (k - 1) = Person.Egor)
  -- Arkady and Galya do not stand next to each other
  ∧ ∀ l, c l = Person.Arkady → c (l + 1) ≠ Person.Galya ∧ c (l - 1) ≠ Person.Galya

-- Theorem statement
theorem boris_neighbors (c : Circle) (h : satisfies_conditions c) :
  ∃ i, c i = Person.Boris ∧ 
    ((c (i - 1) = Person.Arkady ∧ c (i + 1) = Person.Galya) ∨
     (c (i - 1) = Person.Galya ∧ c (i + 1) = Person.Arkady)) :=
by
  sorry

end NUMINAMATH_CALUDE_boris_neighbors_l3661_366190


namespace NUMINAMATH_CALUDE_mikes_shopping_l3661_366148

/-- Mike's shopping problem -/
theorem mikes_shopping (food wallet shirt : ℝ) 
  (h1 : shirt = wallet / 3)
  (h2 : wallet = food + 60)
  (h3 : shirt + wallet + food = 150) :
  food = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shopping_l3661_366148


namespace NUMINAMATH_CALUDE_base3_20202_equals_182_l3661_366189

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The theorem stating that the base-3 number 20202 is equal to 182 in base 10 -/
theorem base3_20202_equals_182 : base3_to_base10 [2, 0, 2, 0, 2] = 182 := by
  sorry

#eval base3_to_base10 [2, 0, 2, 0, 2]

end NUMINAMATH_CALUDE_base3_20202_equals_182_l3661_366189


namespace NUMINAMATH_CALUDE_cubic_equality_solution_l3661_366113

theorem cubic_equality_solution : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equality_solution_l3661_366113


namespace NUMINAMATH_CALUDE_min_trig_fraction_l3661_366196

theorem min_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (17/8) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_trig_fraction_l3661_366196


namespace NUMINAMATH_CALUDE_gum_pack_size_l3661_366106

theorem gum_pack_size : ∃ x : ℕ+, 
  (30 : ℚ) - 2 * x.val = 30 * 40 / (40 + 4 * x.val) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_pack_size_l3661_366106


namespace NUMINAMATH_CALUDE_train_length_l3661_366128

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 36 →  -- speed in km/hr
  time = 9 →    -- time in seconds
  speed * (time / 3600) = 90 / 1000 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3661_366128


namespace NUMINAMATH_CALUDE_sum_13_impossible_l3661_366172

-- Define the type for dice faces
def DieFace := Fin 6

-- Define the function to calculate the sum of two dice
def diceSum (d1 d2 : DieFace) : Nat := d1.val + d2.val + 2

-- Theorem statement
theorem sum_13_impossible :
  ¬ ∃ (d1 d2 : DieFace), diceSum d1 d2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_13_impossible_l3661_366172


namespace NUMINAMATH_CALUDE_always_greater_than_m_l3661_366129

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end NUMINAMATH_CALUDE_always_greater_than_m_l3661_366129


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3661_366112

theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 576 * Real.pi) :
  let r := Real.sqrt (circle_area / Real.pi)
  let hexagon_area := 6 * ((r^2 * Real.sqrt 3) / 4)
  hexagon_area = 864 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3661_366112


namespace NUMINAMATH_CALUDE_tommy_initial_candy_l3661_366145

/-- The amount of candy each person has after sharing equally -/
def shared_amount : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

/-- Hugh's initial amount of candy -/
def hugh_initial : ℕ := 8

/-- Melany's initial amount of candy -/
def melany_initial : ℕ := 7

/-- Tommy's initial amount of candy -/
def tommy_initial : ℕ := shared_amount * num_people - hugh_initial - melany_initial

theorem tommy_initial_candy : tommy_initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_tommy_initial_candy_l3661_366145


namespace NUMINAMATH_CALUDE_equation_implication_l3661_366138

theorem equation_implication (x y : ℝ) : 
  x^2 - 3*x*y + 2*y^2 + x - y = 0 → 
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 → 
  x*y - 12*x + 15*y = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implication_l3661_366138


namespace NUMINAMATH_CALUDE_time_to_destination_l3661_366170

-- Define the walking speeds and distances
def your_speed : ℝ := 2
def harris_speed : ℝ := 1
def harris_time : ℝ := 2
def distance_ratio : ℝ := 3

-- Theorem statement
theorem time_to_destination : 
  your_speed * harris_speed = 2 → 
  (your_speed * (distance_ratio * harris_time)) / your_speed = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_destination_l3661_366170


namespace NUMINAMATH_CALUDE_exponential_inequality_l3661_366110

theorem exponential_inequality (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  Real.exp ((x₁ + x₂) / 2) < (Real.exp x₁ + Real.exp x₂) / (x₁ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3661_366110


namespace NUMINAMATH_CALUDE_adult_admission_price_l3661_366115

theorem adult_admission_price
  (total_people : ℕ)
  (total_receipts : ℕ)
  (num_children : ℕ)
  (child_price : ℕ)
  (h1 : total_people = 610)
  (h2 : total_receipts = 960)
  (h3 : num_children = 260)
  (h4 : child_price = 1) :
  (total_receipts - num_children * child_price) / (total_people - num_children) = 2 :=
by sorry

end NUMINAMATH_CALUDE_adult_admission_price_l3661_366115


namespace NUMINAMATH_CALUDE_heart_properties_l3661_366126

def heart (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem heart_properties :
  (∀ x y : ℝ, heart x y = heart y x) ∧
  (∃ x y : ℝ, 2 * (heart x y) ≠ heart (2*x) (2*y)) ∧
  (∀ x : ℝ, heart x 0 = x^2) ∧
  (∀ x : ℝ, heart x x = 0) ∧
  (∀ x y : ℝ, x ≠ y → heart x y > 0) :=
by sorry

end NUMINAMATH_CALUDE_heart_properties_l3661_366126


namespace NUMINAMATH_CALUDE_journey_mpg_approx_30_3_l3661_366152

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) (gas_fills : List ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The average miles per gallon for the given journey is approximately 30.3 -/
theorem journey_mpg_approx_30_3 :
  let initial_odometer := 34650
  let final_odometer := 35800
  let gas_fills := [8, 10, 15, 5]
  let mpg := average_mpg initial_odometer final_odometer gas_fills
  ∃ ε > 0, abs (mpg - 30.3) < ε ∧ ε < 0.1 := by
  sorry

#eval average_mpg 34650 35800 [8, 10, 15, 5]

end NUMINAMATH_CALUDE_journey_mpg_approx_30_3_l3661_366152


namespace NUMINAMATH_CALUDE_balls_distribution_theorem_l3661_366160

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem balls_distribution_theorem :
  distribute_balls 5 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_balls_distribution_theorem_l3661_366160


namespace NUMINAMATH_CALUDE_lindas_broken_eggs_l3661_366195

/-- The number of eggs Linda broke -/
def broken_eggs (initial_white : ℕ) (initial_brown : ℕ) (total_after : ℕ) : ℕ :=
  initial_white + initial_brown - total_after

theorem lindas_broken_eggs :
  let initial_brown := 5
  let initial_white := 3 * initial_brown
  let total_after := 12
  broken_eggs initial_white initial_brown total_after = 8 := by
  sorry

#eval broken_eggs (3 * 5) 5 12  -- Should output 8

end NUMINAMATH_CALUDE_lindas_broken_eggs_l3661_366195


namespace NUMINAMATH_CALUDE_average_equation_solution_l3661_366132

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (7 * x + 3) + (3 * x + 9)) / 3 = 5 * x - 10 → x = 12.5 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3661_366132


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3661_366108

theorem quadratic_is_perfect_square (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 - 8 * x + 16 = (r * x + s)^2) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3661_366108


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l3661_366187

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (∀ m : ℕ, m < 11 → ¬(isMultipleOf11 m ∧ ¬isLucky m)) ∧
  (isMultipleOf11 11 ∧ ¬isLucky 11) :=
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l3661_366187


namespace NUMINAMATH_CALUDE_sqrt_squared_eq_self_sqrt_784_squared_l3661_366109

theorem sqrt_squared_eq_self (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by sorry

theorem sqrt_784_squared : (Real.sqrt 784) ^ 2 = 784 := by sorry

end NUMINAMATH_CALUDE_sqrt_squared_eq_self_sqrt_784_squared_l3661_366109


namespace NUMINAMATH_CALUDE_sine_inequality_solution_l3661_366164

theorem sine_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) ∧ 
  (|a| + |b| ≥ 2 / Real.sqrt 3) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_l3661_366164


namespace NUMINAMATH_CALUDE_total_wattage_after_increase_l3661_366104

theorem total_wattage_after_increase (light_a light_b light_c light_d : ℝ)
  (increase_a increase_b increase_c increase_d : ℝ) :
  light_a = 60 →
  light_b = 40 →
  light_c = 50 →
  light_d = 80 →
  increase_a = 0.12 →
  increase_b = 0.20 →
  increase_c = 0.15 →
  increase_d = 0.10 →
  (light_a * (1 + increase_a) +
   light_b * (1 + increase_b) +
   light_c * (1 + increase_c) +
   light_d * (1 + increase_d)) = 260.7 := by
  sorry

end NUMINAMATH_CALUDE_total_wattage_after_increase_l3661_366104


namespace NUMINAMATH_CALUDE_intersection_M_N_l3661_366130

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3661_366130


namespace NUMINAMATH_CALUDE_other_communities_count_l3661_366133

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 32 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 119 ∧ 
    (other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l3661_366133


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3661_366105

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  IsParallel : Line → Plane → Prop
  HasCommonPoint : Line → Plane → Prop

-- State the theorem
theorem line_plane_intersection
  (S : Space3D)
  (a : S.Line)
  (α : S.Plane)
  (h : ¬S.IsParallel a α) :
  S.HasCommonPoint a α :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l3661_366105


namespace NUMINAMATH_CALUDE_sin_cos_square_identity_l3661_366119

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_square_identity_l3661_366119


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l3661_366117

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Given point P -/
def P : Point3D := ⟨3, 1, 5⟩

/-- Function to find the symmetric point about the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

/-- Theorem: The point symmetric to P(3,1,5) about the origin is (3,-1,-5) -/
theorem symmetric_point_of_P :
  symmetricPoint P = Point3D.mk 3 (-1) (-5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l3661_366117


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3661_366144

theorem adult_ticket_cost 
  (total_tickets : ℕ) 
  (total_receipts : ℕ) 
  (child_ticket_cost : ℕ) 
  (child_tickets_sold : ℕ) 
  (h1 : total_tickets = 130) 
  (h2 : total_receipts = 840) 
  (h3 : child_ticket_cost = 4) 
  (h4 : child_tickets_sold = 90) : 
  (total_receipts - child_tickets_sold * child_ticket_cost) / (total_tickets - child_tickets_sold) = 12 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3661_366144


namespace NUMINAMATH_CALUDE_geometric_series_sum_and_comparison_l3661_366163

theorem geometric_series_sum_and_comparison :
  let a : ℝ := 2
  let r : ℝ := 1/4
  let S : ℝ := a / (1 - r)
  S = 8/3 ∧ S ≤ 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_and_comparison_l3661_366163


namespace NUMINAMATH_CALUDE_fencing_required_l3661_366155

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 10 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ 2 * width + uncovered_side = 146 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l3661_366155


namespace NUMINAMATH_CALUDE_derivative_of_2ln_derivative_of_exp_div_x_l3661_366153

-- Function 1: f(x) = 2ln(x)
theorem derivative_of_2ln (x : ℝ) (h : x > 0) : 
  deriv (fun x => 2 * Real.log x) x = 2 / x := by sorry

-- Function 2: f(x) = e^x / x
theorem derivative_of_exp_div_x (x : ℝ) (h : x ≠ 0) : 
  deriv (fun x => Real.exp x / x) x = (Real.exp x * x - Real.exp x) / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_2ln_derivative_of_exp_div_x_l3661_366153


namespace NUMINAMATH_CALUDE_product_digit_sum_l3661_366111

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  thousands_digit product + units_digit product = 3 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3661_366111


namespace NUMINAMATH_CALUDE_laura_running_speed_approx_l3661_366154

/-- Laura's workout parameters --/
structure WorkoutParams where
  totalDuration : ℝ  -- Total workout duration in minutes
  bikingDistance : ℝ  -- Biking distance in miles
  transitionTime : ℝ  -- Transition time in minutes
  runningDistance : ℝ  -- Running distance in miles

/-- Calculate Laura's running speed given workout parameters --/
def calculateRunningSpeed (params : WorkoutParams) (x : ℝ) : ℝ :=
  x^2 - 1

/-- Theorem stating that Laura's running speed is approximately 83.33 mph --/
theorem laura_running_speed_approx (params : WorkoutParams) :
  ∃ x : ℝ,
    params.totalDuration = 150 ∧
    params.bikingDistance = 30 ∧
    params.transitionTime = 10 ∧
    params.runningDistance = 5 ∧
    (params.totalDuration - params.transitionTime) / 60 = params.bikingDistance / (3*x + 2) + params.runningDistance / (x^2 - 1) ∧
    abs (calculateRunningSpeed params x - 83.33) < 0.01 :=
  sorry


end NUMINAMATH_CALUDE_laura_running_speed_approx_l3661_366154


namespace NUMINAMATH_CALUDE_fencing_cost_is_1950_l3661_366122

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  length_width_relation : length = width + 10
  perimeter_constraint : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 300
  fencing_rate_value : fencing_rate = 6.5

/-- The cost of fencing the rectangular plot. -/
def fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_rate

/-- Theorem stating the fencing cost for the given rectangular plot. -/
theorem fencing_cost_is_1950 (plot : RectangularPlot) : fencing_cost plot = 1950 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_is_1950_l3661_366122


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3661_366181

/-- A hyperbola with the given properties has eccentricity √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (M N Q E P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (M.1^2 / a^2 - M.2^2 / b^2 = 1) →
  (N.1^2 / a^2 - N.2^2 / b^2 = 1) →
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →
  N = (-M.1, -M.2) →
  Q = (M.1, -M.2) →
  E = (M.1, -3 * M.2) →
  (P.2 - M.2) * (P.1 - M.1) = -(N.2 - M.2) * (N.1 - M.1) →
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3661_366181


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3661_366118

theorem expression_equals_negative_one 
  (x y z : ℝ) 
  (hx : x ≠ 1) 
  (hy : y ≠ 2) 
  (hz : z ≠ 3) : 
  (x - 1) / (3 - z) * (y - 2) / (1 - x) * (z - 3) / (2 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3661_366118


namespace NUMINAMATH_CALUDE_largest_c_value_l3661_366123

theorem largest_c_value (c : ℝ) : 
  (3 * c + 4) * (c - 2) = 7 * c →
  c ≤ (9 + Real.sqrt 177) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_value_l3661_366123


namespace NUMINAMATH_CALUDE_slope_angle_range_l3661_366185

/-- Given two lines and their intersection in the first quadrant, 
    prove the range of the slope angle of one line -/
theorem slope_angle_range (k : ℝ) : 
  let l1 : ℝ → ℝ := λ x => k * x - Real.sqrt 3
  let l2 : ℝ → ℝ := λ x => (6 - 2 * x) / 3
  let x_intersect := (3 * Real.sqrt 3 + 6) / (2 + 3 * k)
  let y_intersect := (6 * k - 2 * Real.sqrt 3) / (2 + 3 * k)
  (x_intersect > 0 ∧ y_intersect > 0) →
  let θ := Real.arctan k
  θ > π / 6 ∧ θ < π / 2 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l3661_366185


namespace NUMINAMATH_CALUDE_complex_modulus_l3661_366151

theorem complex_modulus (z : ℂ) (h : z = (1/2 : ℂ) + (5/2 : ℂ) * Complex.I) : 
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3661_366151


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l3661_366178

theorem larger_divided_by_smaller : 
  let a := 8
  let b := 22
  let larger := max a b
  let smaller := min a b
  larger / smaller = 2.75 := by sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l3661_366178


namespace NUMINAMATH_CALUDE_equation_solution_count_l3661_366135

theorem equation_solution_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (n^2 - 2*n - 2)*n^2 + 47 = (n^2 - 2*n - 2)*16*n - 16) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_count_l3661_366135


namespace NUMINAMATH_CALUDE_item_frequency_proof_l3661_366103

theorem item_frequency_proof (total : ℕ) (second_grade : ℕ) 
  (h1 : total = 400) (h2 : second_grade = 20) : 
  let first_grade := total - second_grade
  (first_grade : ℚ) / total = 95 / 100 ∧ 
  (second_grade : ℚ) / total = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_item_frequency_proof_l3661_366103


namespace NUMINAMATH_CALUDE_homework_problems_left_l3661_366169

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) : 
  math_problems + science_problems - finished_problems = 15 := by
sorry

end NUMINAMATH_CALUDE_homework_problems_left_l3661_366169


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l3661_366141

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l3661_366141


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l3661_366134

theorem quadratic_equation_conversion (x : ℝ) : 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n) → 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n ∧ m + n = 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l3661_366134


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l3661_366158

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_parking_lot : 
  total_cars * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l3661_366158


namespace NUMINAMATH_CALUDE_inequality_proof_l3661_366161

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3661_366161


namespace NUMINAMATH_CALUDE_sams_age_l3661_366173

/-- Given that Sam and Drew have a combined age of 54 and Sam is half of Drew's age,
    prove that Sam is 18 years old. -/
theorem sams_age (total_age : ℕ) (drews_age : ℕ) (sams_age : ℕ) 
    (h1 : total_age = 54)
    (h2 : sams_age + drews_age = total_age)
    (h3 : sams_age = drews_age / 2) : 
  sams_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_age_l3661_366173


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3661_366183

theorem compound_interest_rate : 
  ∀ (P : ℝ) (A : ℝ) (I : ℝ) (t : ℕ) (r : ℝ),
  A = 19828.80 →
  I = 2828.80 →
  t = 2 →
  A = P + I →
  A = P * (1 + r) ^ t →
  r = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3661_366183


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l3661_366165

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 25 → x + y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l3661_366165


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3661_366166

def f (x : ℝ) := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3661_366166


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3661_366176

theorem unique_integer_solution :
  ∃! x : ℤ, (x - 3 : ℚ) ^ (27 - x^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3661_366176


namespace NUMINAMATH_CALUDE_video_game_spend_l3661_366197

/-- Calculates the amount spent on video games given total pocket money and fractions spent on other items --/
def video_game_expenditure (total : ℚ) (books : ℚ) (snacks : ℚ) (toys : ℚ) : ℚ :=
  total - (books * total + snacks * total + toys * total)

/-- Theorem stating that the amount spent on video games is 6 dollars --/
theorem video_game_spend :
  let total : ℚ := 40
  let books : ℚ := 2 / 5
  let snacks : ℚ := 1 / 4
  let toys : ℚ := 1 / 5
  video_game_expenditure total books snacks toys = 6 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spend_l3661_366197


namespace NUMINAMATH_CALUDE_charging_pile_growth_l3661_366114

/-- Represents the growth of smart charging piles over two months -/
theorem charging_pile_growth 
  (initial_count : ℕ) 
  (final_count : ℕ) 
  (growth_rate : ℝ) 
  (h1 : initial_count = 301)
  (h2 : final_count = 500)
  : initial_count * (1 + growth_rate)^2 = final_count := by
  sorry

#check charging_pile_growth

end NUMINAMATH_CALUDE_charging_pile_growth_l3661_366114


namespace NUMINAMATH_CALUDE_f_5_solutions_l3661_366180

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself 5 times -/
def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

/-- Theorem: The solutions to f(f(f(f(f(x))))) = 0 are x = -6 ± 6^(1/32) -/
theorem f_5_solutions :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32) :=
by sorry

end NUMINAMATH_CALUDE_f_5_solutions_l3661_366180


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3661_366140

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3661_366140
