import Mathlib

namespace percentage_of_muslim_boys_l3722_372239

/-- Given a school with 850 boys, where 28% are Hindus, 10% are Sikhs, 
    and 136 boys belong to other communities, prove that 46% of the boys are Muslims. -/
theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 136 →
  (↑(total_boys - (total_boys * hindu_percent).floor - (total_boys * sikh_percent).floor - other_boys) / total_boys : ℚ) = 46 / 100 :=
by
  sorry

#eval (850 : ℕ) - (850 * (28 / 100 : ℚ)).floor - (850 * (10 / 100 : ℚ)).floor - 136

end percentage_of_muslim_boys_l3722_372239


namespace tan_205_in_terms_of_cos_155_l3722_372247

theorem tan_205_in_terms_of_cos_155 (a : ℝ) (h : Real.cos (155 * π / 180) = a) :
  Real.tan (205 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end tan_205_in_terms_of_cos_155_l3722_372247


namespace sum_equals_result_l3722_372250

-- Define the sum
def sum : ℚ := 10/9 + 9/10

-- Define the result as a rational number (2 + 1/10)
def result : ℚ := 2 + 1/10

-- Theorem stating that the sum equals the result
theorem sum_equals_result : sum = result := by sorry

end sum_equals_result_l3722_372250


namespace switches_in_position_a_after_process_l3722_372262

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 4

/-- The set of all switches -/
def switches : Finset Switch :=
  sorry

/-- The process of advancing switches -/
def advance_switches : Finset Switch → Finset Switch :=
  sorry

/-- The final state after 729 steps -/
def final_state : Finset Switch :=
  sorry

/-- Count switches in position A -/
def count_position_a (s : Finset Switch) : Nat :=
  sorry

theorem switches_in_position_a_after_process :
  count_position_a final_state = 409 := by
  sorry

end switches_in_position_a_after_process_l3722_372262


namespace polygon_intersection_theorem_l3722_372200

-- Define a convex polygon
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a point to be inside a circle
def point_inside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define what it means for a polygon to be inside a circle
def polygon_inside_circle {n : ℕ} (p : ConvexPolygon n) (c : Circle) : Prop :=
  ∀ i : Fin n, point_inside_circle (p.vertices i) c

-- Define what it means for two polygons to intersect
def polygons_intersect {n m : ℕ} (p1 : ConvexPolygon n) (p2 : ConvexPolygon m) : Prop :=
  sorry

-- The main theorem
theorem polygon_intersection_theorem {n m : ℕ}
  (p1 : ConvexPolygon n) (p2 : ConvexPolygon m) (c1 c2 : Circle)
  (h1 : polygon_inside_circle p1 c1)
  (h2 : polygon_inside_circle p2 c2)
  (h3 : polygons_intersect p1 p2) :
  (∃ i : Fin n, point_inside_circle (p1.vertices i) c2) ∨
  (∃ j : Fin m, point_inside_circle (p2.vertices j) c1) :=
sorry

end polygon_intersection_theorem_l3722_372200


namespace polynomial_equality_conditions_l3722_372232

theorem polynomial_equality_conditions (A B C p q : ℝ) :
  (∀ x : ℝ, A * x^4 + B * x^2 + C = A * (x^2 + p * x + q) * (x^2 - p * x + q)) →
  (A * (2 * q - p^2) = B ∧ A * q^2 = C) :=
by sorry

end polynomial_equality_conditions_l3722_372232


namespace investment_interest_rate_exists_and_unique_l3722_372212

theorem investment_interest_rate_exists_and_unique :
  ∃! r : ℝ, 
    r > 0 ∧ 
    6000 * (1 + r)^10 = 24000 ∧ 
    6000 * (1 + r)^15 = 48000 := by
  sorry

end investment_interest_rate_exists_and_unique_l3722_372212


namespace sqrt_five_power_calculation_l3722_372259

theorem sqrt_five_power_calculation : 
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * 5 ^ (3/4) := by
  sorry

end sqrt_five_power_calculation_l3722_372259


namespace simon_stamps_l3722_372236

theorem simon_stamps (initial_stamps : ℕ) (friend1_stamps : ℕ) (friend2_stamps : ℕ) (friend3_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : friend1_stamps = 15)
  (h3 : friend2_stamps = 23)
  (h4 : initial_stamps + friend1_stamps + friend2_stamps + friend3_stamps = 61) :
  friend3_stamps = 23 ∧ friend1_stamps + friend2_stamps + friend3_stamps = 61 := by
  sorry

end simon_stamps_l3722_372236


namespace distance_between_points_l3722_372248

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end distance_between_points_l3722_372248


namespace angle_sum_around_point_l3722_372240

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 150 > 0 ∧ 
  x + x + 150 = 360 →
  x = 105 := by sorry

end angle_sum_around_point_l3722_372240


namespace bottle_cap_calculation_l3722_372291

theorem bottle_cap_calculation (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end bottle_cap_calculation_l3722_372291


namespace aquarium_solution_l3722_372284

def aquarium_animals (otters seals sea_lions : ℕ) : Prop :=
  (otters + seals = 7 ∨ otters = 7 ∨ seals = 7) ∧
  (sea_lions + seals = 6 ∨ sea_lions = 6 ∨ seals = 6) ∧
  (otters + sea_lions = 5 ∨ otters = 5 ∨ sea_lions = 5) ∧
  (otters ≤ seals ∨ seals ≤ otters) ∧
  (otters ≤ sea_lions ∧ seals ≤ sea_lions)

theorem aquarium_solution :
  ∃! (otters seals sea_lions : ℕ),
    aquarium_animals otters seals sea_lions ∧
    otters = 5 ∧ seals = 7 ∧ sea_lions = 6 :=
sorry

end aquarium_solution_l3722_372284


namespace dice_probabilities_l3722_372278

-- Define the type for a die
def Die : Type := Fin 6

-- Define the sample space
def SampleSpace : Type := Die × Die

-- Define the probability measure
noncomputable def P : Set SampleSpace → ℝ := sorry

-- Define the event of rolling the same number on both dice
def SameNumber : Set SampleSpace :=
  {p : SampleSpace | p.1 = p.2}

-- Define the event of rolling a sum less than 7
def SumLessThan7 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 < 7}

-- Define the event of rolling a sum equal to or greater than 11
def SumGreaterEqual11 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 ≥ 11}

theorem dice_probabilities :
  P SameNumber = 1/6 ∧
  P SumLessThan7 = 5/12 ∧
  P SumGreaterEqual11 = 1/12 := by sorry

end dice_probabilities_l3722_372278


namespace better_performance_against_teamB_l3722_372202

/-- Represents the statistics for a team --/
structure TeamStats :=
  (points : List Nat)
  (rebounds : List Nat)
  (turnovers : List Nat)

/-- Calculate the average of a list of numbers --/
def average (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

/-- Calculate the comprehensive score for a team --/
def comprehensiveScore (stats : TeamStats) : Rat :=
  average stats.points + 1.2 * average stats.rebounds - average stats.turnovers

/-- Xiao Bin's statistics against Team A --/
def teamA : TeamStats :=
  { points := [21, 29, 24, 26],
    rebounds := [10, 10, 14, 10],
    turnovers := [2, 2, 3, 5] }

/-- Xiao Bin's statistics against Team B --/
def teamB : TeamStats :=
  { points := [25, 31, 16, 22],
    rebounds := [17, 15, 12, 8],
    turnovers := [2, 0, 4, 2] }

/-- Theorem: Xiao Bin's comprehensive score against Team B is higher than against Team A --/
theorem better_performance_against_teamB :
  comprehensiveScore teamB > comprehensiveScore teamA :=
by
  sorry


end better_performance_against_teamB_l3722_372202


namespace corresponding_time_l3722_372296

-- Define the ratio
def ratio : ℚ := 8 / 4

-- Define the conversion factor from seconds to minutes
def seconds_to_minutes : ℚ := 1 / 60

-- State the theorem
theorem corresponding_time (t : ℚ) : 
  ratio = 8 / t → t = 4 * seconds_to_minutes :=
by sorry

end corresponding_time_l3722_372296


namespace factor_polynomial_l3722_372287

theorem factor_polynomial (x : ℝ) : 45 * x^3 - 135 * x^7 = 45 * x^3 * (1 - 3 * x^4) := by
  sorry

end factor_polynomial_l3722_372287


namespace equation_solution_l3722_372297

theorem equation_solution (a b : ℝ) :
  b ≠ 0 →
  (∀ x, (4 * a * x + 1) / b - 5 = 3 * x / b) ↔
  (b = 0 ∧ False) ∨
  (a = 3/4 ∧ b = 1/5) ∨
  (4 * a - 3 ≠ 0 ∧ ∃! x, x = (5 * b - 1) / (4 * a - 3)) :=
by sorry

end equation_solution_l3722_372297


namespace three_digit_sum_theorem_l3722_372215

/-- Given three distinct single-digit numbers, returns the largest three-digit number that can be formed using these digits. -/
def largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the second largest three-digit number that can be formed using these digits. -/
def second_largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the smallest three-digit number that can be formed using these digits. -/
def smallest_three_digit (a b c : Nat) : Nat := sorry

theorem three_digit_sum_theorem :
  let a := 2
  let b := 5
  let c := 8
  largest_three_digit a b c + smallest_three_digit a b c + second_largest_three_digit a b c = 1935 := by
  sorry

end three_digit_sum_theorem_l3722_372215


namespace tangent_equations_not_equivalent_l3722_372270

open Real

theorem tangent_equations_not_equivalent :
  ¬(∀ x : ℝ, (tan (2 * x) - (1 / tan x) = 0) ↔ ((2 * tan x) / (1 - tan x ^ 2) - 1 / tan x = 0)) :=
by sorry

end tangent_equations_not_equivalent_l3722_372270


namespace x_minus_25_is_perfect_square_l3722_372290

theorem x_minus_25_is_perfect_square (n : ℕ) :
  let x := 10^(2*n + 4) + 10^(n + 3) + 50
  ∃ k : ℕ, x - 25 = k^2 := by
  sorry

end x_minus_25_is_perfect_square_l3722_372290


namespace isosceles_triangle_80_vertex_angle_l3722_372211

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of the vertex angle in degrees -/
  vertex_angle : ℝ
  /-- The measure of one of the base angles in degrees -/
  base_angle : ℝ
  /-- The triangle is isosceles -/
  isosceles : base_angle = 180 - vertex_angle - base_angle
  /-- One angle is 80 degrees -/
  has_80_degree : vertex_angle = 80 ∨ base_angle = 80
  /-- The sum of angles is 180 degrees -/
  angle_sum : vertex_angle + 2 * base_angle = 180

/-- The vertex angle in an isosceles triangle with one 80-degree angle is either 80 or 20 degrees -/
theorem isosceles_triangle_80_vertex_angle (t : IsoscelesTriangle80) :
  t.vertex_angle = 80 ∨ t.vertex_angle = 20 := by
  sorry

end isosceles_triangle_80_vertex_angle_l3722_372211


namespace height_C_ceiling_l3722_372201

/-- A right-angled triangle with given heights -/
structure RightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Heights
  hA : ℝ
  hB : ℝ
  hC : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  height_A : hA = 5
  height_B : hB = 15
  -- Area consistency
  area_consistency : a * hA = b * hB

/-- The smallest integer ceiling of the height from C to AB is 5 -/
theorem height_C_ceiling (t : RightTriangle) : ⌈t.hC⌉ = 5 := by
  sorry

end height_C_ceiling_l3722_372201


namespace max_profits_l3722_372234

def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

def average_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

theorem max_profits :
  (∃ (x_max : ℕ+), ∀ (x : ℕ+), total_profit x ≤ total_profit x_max ∧ 
    total_profit x_max = 45) ∧
  (∃ (x_avg_max : ℕ+), ∀ (x : ℕ+), average_annual_profit x ≤ average_annual_profit x_avg_max ∧ 
    average_annual_profit x_avg_max = 6) :=
by sorry

end max_profits_l3722_372234


namespace sampling_methods_are_appropriate_l3722_372245

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with sales points -/
structure Region where
  name : String
  salesPoints : Nat

/-- Represents a company with multiple regions -/
structure Company where
  regions : List Region
  totalSalesPoints : Nat

/-- Represents an investigation -/
structure Investigation where
  sampleSize : Nat
  samplingMethod : SamplingMethod

/-- The company in the problem -/
def problemCompany : Company :=
  { regions := [
      { name := "A", salesPoints := 150 },
      { name := "B", salesPoints := 120 },
      { name := "C", salesPoints := 180 },
      { name := "D", salesPoints := 150 }
    ],
    totalSalesPoints := 600
  }

/-- The first investigation in the problem -/
def investigation1 : Investigation :=
  { sampleSize := 100,
    samplingMethod := SamplingMethod.StratifiedSampling
  }

/-- The second investigation in the problem -/
def investigation2 : Investigation :=
  { sampleSize := 7,
    samplingMethod := SamplingMethod.SimpleRandomSampling
  }

/-- Checks if stratified sampling is appropriate for the given company and investigation -/
def isStratifiedSamplingAppropriate (company : Company) (investigation : Investigation) : Prop :=
  investigation.samplingMethod = SamplingMethod.StratifiedSampling ∧
  company.regions.length > 1 ∧
  investigation.sampleSize < company.totalSalesPoints

/-- Checks if simple random sampling is appropriate for the given sample size and population -/
def isSimpleRandomSamplingAppropriate (sampleSize : Nat) (populationSize : Nat) : Prop :=
  sampleSize < populationSize

/-- Theorem stating that the sampling methods are appropriate for the given investigations -/
theorem sampling_methods_are_appropriate :
  isStratifiedSamplingAppropriate problemCompany investigation1 ∧
  isSimpleRandomSamplingAppropriate investigation2.sampleSize 20 :=
  sorry

end sampling_methods_are_appropriate_l3722_372245


namespace train_speed_l3722_372203

/-- Proves that the speed of a train is 90 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 225) (h2 : time = 9) :
  (length / 1000) / (time / 3600) = 90 := by
  sorry

#check train_speed

end train_speed_l3722_372203


namespace probability_three_heads_twelve_coins_l3722_372252

theorem probability_three_heads_twelve_coins : 
  (Nat.choose 12 3 : ℚ) / (2^12 : ℚ) = 55 / 1024 := by
  sorry

end probability_three_heads_twelve_coins_l3722_372252


namespace number_and_square_sum_l3722_372246

theorem number_and_square_sum (x : ℝ) : x + x^2 = 342 → x = 18 ∨ x = -19 := by
  sorry

end number_and_square_sum_l3722_372246


namespace adults_on_bicycles_l3722_372221

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem: The number of adults riding bicycles is 6 -/
theorem adults_on_bicycles : 
  ∃ (a : ℕ), a * bicycle_wheels + children_on_tricycles * tricycle_wheels = total_wheels ∧ a = 6 :=
sorry

end adults_on_bicycles_l3722_372221


namespace omega_value_l3722_372266

-- Define the complex numbers z and ω
variable (z ω : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the conditions
axiom pure_imaginary : ∃ (y : ℝ), (1 + 3 * i) * z = i * y
axiom omega_def : ω = z / (2 + i)
axiom omega_abs : Complex.abs ω = 5 * Real.sqrt 2

-- State the theorem to be proved
theorem omega_value : ω = 7 - i ∨ ω = -(7 - i) := by sorry

end omega_value_l3722_372266


namespace rectangle_area_theorem_l3722_372231

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 35 square units and y > 0, then y = 7. -/
theorem rectangle_area_theorem (y : ℝ) : y > 0 → 5 * y = 35 → y = 7 := by
  sorry

end rectangle_area_theorem_l3722_372231


namespace eighth_term_of_specific_sequence_l3722_372293

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_specific_sequence :
  let a₁ : ℚ := 27
  let r : ℚ := 2/3
  geometric_sequence a₁ r 8 = 128/81 := by
sorry

end eighth_term_of_specific_sequence_l3722_372293


namespace binomial_square_example_l3722_372258

theorem binomial_square_example : 16^2 + 2*(16*5) + 5^2 = 441 := by
  sorry

end binomial_square_example_l3722_372258


namespace floor_T_equals_120_l3722_372264

-- Define positive real numbers p, q, r, s
variable (p q r s : ℝ)

-- Define the conditions
axiom p_pos : p > 0
axiom q_pos : q > 0
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom sum_squares_pq : p^2 + q^2 = 2500
axiom sum_squares_rs : r^2 + s^2 = 2500
axiom product_pr : p * r = 1200
axiom product_qs : q * s = 1200

-- Define T
def T : ℝ := p + q + r + s

-- Theorem to prove
theorem floor_T_equals_120 : ⌊T p q r s⌋ = 120 := by sorry

end floor_T_equals_120_l3722_372264


namespace hunter_can_kill_wolf_l3722_372292

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  center : Point

/-- Checks if a point is within an equilateral triangle -/
def isWithinTriangle (p : Point) (t : EquilateralTriangle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Theorem: Hunter can always kill the wolf -/
theorem hunter_can_kill_wolf (t : EquilateralTriangle) 
  (h_side : t.sideLength = 100) :
  ∃ (hunter : Point), 
    ∀ (wolf : Point), 
      isWithinTriangle wolf t → 
        distance hunter wolf ≤ 30 := by
  sorry

end hunter_can_kill_wolf_l3722_372292


namespace no_natural_solutions_l3722_372299

theorem no_natural_solutions :
  (∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z) ∧
  (∀ x y z u : ℕ, x^2 + y^2 + z^2 + u^2 ≠ 2*x*y*z*u) := by
  sorry

end no_natural_solutions_l3722_372299


namespace cone_section_properties_l3722_372277

/-- Given a right circular cone with base radius 25 cm and slant height 42 cm,
    when cut by a plane parallel to the base such that the volumes of the two resulting parts are equal,
    the radius of the circular intersection is 25 * (1/2)^(1/3) cm
    and the height of the smaller cone is sqrt(1139) * (1/2)^(1/3) cm. -/
theorem cone_section_properties :
  let base_radius : ℝ := 25
  let slant_height : ℝ := 42
  let cone_height : ℝ := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  let section_radius : ℝ := base_radius * (1/2) ^ (1/3)
  let small_cone_height : ℝ := cone_height * (1/2) ^ (1/3)
  (1/3) * Real.pi * base_radius ^ 2 * cone_height = 2 * ((1/3) * Real.pi * section_radius ^ 2 * small_cone_height) →
  section_radius = 25 * (1/2) ^ (1/3) ∧ small_cone_height = Real.sqrt 1139 * (1/2) ^ (1/3) := by
  sorry


end cone_section_properties_l3722_372277


namespace tetrahedron_surface_area_l3722_372226

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the projection of a tetrahedron onto a plane -/
structure TetrahedronProjection where
  area : ℝ
  has_60_degree_angle : Bool

/-- 
Given a regular tetrahedron and its projection onto a plane parallel to the line segment 
connecting the midpoints of two opposite edges, prove that the surface area of the tetrahedron 
is 2x^2 √2/3, where x is the edge length of the tetrahedron.
-/
theorem tetrahedron_surface_area 
  (t : RegularTetrahedron) 
  (p : TetrahedronProjection) 
  (h : p.has_60_degree_angle = true) : 
  ℝ :=
by
  sorry

#check tetrahedron_surface_area

end tetrahedron_surface_area_l3722_372226


namespace rice_dumpling_costs_l3722_372295

theorem rice_dumpling_costs (total_cost_honey : ℝ) (total_cost_date : ℝ) 
  (cost_diff : ℝ) (h1 : total_cost_honey = 1300) (h2 : total_cost_date = 1000) 
  (h3 : cost_diff = 0.6) :
  ∃ (cost_date cost_honey : ℝ),
    cost_date = 2 ∧ 
    cost_honey = 2.6 ∧
    cost_honey = cost_date + cost_diff ∧
    total_cost_honey / cost_honey = total_cost_date / cost_date :=
by
  sorry

end rice_dumpling_costs_l3722_372295


namespace four_Z_three_equals_one_l3722_372267

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

-- Theorem to prove
theorem four_Z_three_equals_one : Z 4 3 = 1 := by
  sorry

end four_Z_three_equals_one_l3722_372267


namespace quadratic_inequality_properties_l3722_372230

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -1/2 and 2 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x ∈ Set.Ioo (-1/2 : ℝ) 2, QuadraticFunction a b c x > 0) →
  (QuadraticFunction a b c (-1/2) = 0) →
  (QuadraticFunction a b c 2 = 0) →
  (b > 0 ∧ c > 0 ∧ a + b + c > 0) := by
  sorry

end quadratic_inequality_properties_l3722_372230


namespace largest_k_value_l3722_372265

/-- Triangle side lengths are positive real numbers that satisfy the triangle inequality --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

/-- The inequality holds for all triangles --/
def inequality_holds (k : ℝ) : Prop :=
  ∀ t : Triangle, (t.a + t.b + t.c)^3 ≥ (5/2) * (t.a^3 + t.b^3 + t.c^3) + k * t.a * t.b * t.c

/-- 39/2 is the largest real number satisfying the inequality --/
theorem largest_k_value : 
  (∀ k : ℝ, k > 39/2 → ¬(inequality_holds k)) ∧ 
  inequality_holds (39/2) := by
  sorry

end largest_k_value_l3722_372265


namespace quadratic_inequality_empty_solution_l3722_372229

theorem quadratic_inequality_empty_solution : 
  {x : ℝ | -x^2 + 2*x - 3 > 0} = ∅ := by sorry

end quadratic_inequality_empty_solution_l3722_372229


namespace simon_treasures_l3722_372268

/-- The number of sand dollars Simon collected -/
def sand_dollars : ℕ := sorry

/-- The number of sea glass pieces Simon collected -/
def sea_glass : ℕ := sorry

/-- The number of seashells Simon collected -/
def seashells : ℕ := sorry

/-- The total number of treasures Simon collected -/
def total_treasures : ℕ := 190

theorem simon_treasures : 
  sea_glass = 3 * sand_dollars ∧ 
  seashells = 5 * sea_glass ∧
  total_treasures = sand_dollars + sea_glass + seashells →
  sand_dollars = 10 := by sorry

end simon_treasures_l3722_372268


namespace robins_hair_length_l3722_372276

/-- Given that Robin's hair is currently 13 inches long after cutting off 4 inches,
    prove that his initial hair length was 17 inches. -/
theorem robins_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 13)
  (h2 : cut_length = 4) : 
  current_length + cut_length = 17 := by
sorry

end robins_hair_length_l3722_372276


namespace goldbach_negation_equiv_l3722_372274

-- Define Goldbach's Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define the negation of Goldbach's Conjecture
def not_goldbach : Prop :=
  ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Theorem stating the equivalence
theorem goldbach_negation_equiv :
  ¬goldbach_conjecture ↔ not_goldbach := by sorry

end goldbach_negation_equiv_l3722_372274


namespace ship_ratio_l3722_372233

/-- Given the conditions of ships in a busy port, prove the ratio of sailboats to fishing boats -/
theorem ship_ratio : 
  ∀ (cruise cargo sailboats fishing : ℕ),
  cruise = 4 →
  cargo = 2 * cruise →
  sailboats = cargo + 6 →
  cruise + cargo + sailboats + fishing = 28 →
  sailboats / fishing = 7 ∧ fishing ≠ 0 :=
by
  sorry

end ship_ratio_l3722_372233


namespace dance_cost_theorem_l3722_372261

/-- Represents the cost calculation for dance shoes and fans. -/
structure DanceCost where
  x : ℝ  -- Number of fans per pair of shoes
  yA : ℝ -- Cost at supermarket A
  yB : ℝ -- Cost at supermarket B

/-- Calculates the cost for dance shoes and fans given the conditions. -/
def calculate_cost (x : ℝ) : DanceCost :=
  { x := x
  , yA := 27 * x + 270
  , yB := 30 * x + 240 }

/-- Theorem stating the relationship between costs and number of fans. -/
theorem dance_cost_theorem (x : ℝ) (h : x ≥ 2) :
  let cost := calculate_cost x
  cost.yA = 27 * x + 270 ∧
  cost.yB = 30 * x + 240 ∧
  (x < 10 → cost.yB < cost.yA) ∧
  (x = 10 → cost.yB = cost.yA) ∧
  (x > 10 → cost.yA < cost.yB) := by
  sorry

#check dance_cost_theorem

end dance_cost_theorem_l3722_372261


namespace jack_payback_l3722_372281

/-- The amount borrowed by Jack -/
def principal : ℝ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def totalAmount : ℝ := principal * (1 + interestRate)

/-- Theorem stating that the total amount Jack will pay back is $1320 -/
theorem jack_payback : totalAmount = 1320 := by
  sorry

end jack_payback_l3722_372281


namespace decimal_33_is_quaternary_201_l3722_372244

-- Define a function to convert decimal to quaternary
def decimalToQuaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec convert (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else convert (m / 4) ((m % 4) :: acc)
    convert n []

-- Theorem statement
theorem decimal_33_is_quaternary_201 :
  decimalToQuaternary 33 = [2, 0, 1] := by
  sorry


end decimal_33_is_quaternary_201_l3722_372244


namespace max_positive_integer_solution_of_inequality_system_l3722_372251

theorem max_positive_integer_solution_of_inequality_system :
  ∃ (x : ℝ), (3 * x - 1 > x + 1) ∧ ((4 * x - 5) / 3 ≤ x) ∧
  (∀ (y : ℤ), (3 * y - 1 > y + 1) ∧ ((4 * y - 5) / 3 ≤ y) → y ≤ 5) ∧
  (3 * 5 - 1 > 5 + 1) ∧ ((4 * 5 - 5) / 3 ≤ 5) :=
by sorry

end max_positive_integer_solution_of_inequality_system_l3722_372251


namespace labor_cost_increase_l3722_372280

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- The initial cost ratio --/
def initial_ratio : CarCost := ⟨4, 3, 2⟩

/-- The percentage changes in costs --/
structure CostChanges where
  raw_material : ℝ := 0.10  -- 10% increase
  labor : ℝ                 -- Unknown, to be calculated
  overhead : ℝ := -0.05     -- 5% decrease
  total : ℝ := 0.06         -- 6% increase

/-- Calculates the new cost based on the initial cost and percentage change --/
def new_cost (initial : ℝ) (change : ℝ) : ℝ :=
  initial * (1 + change)

/-- Theorem stating that the labor cost increased by 8% --/
theorem labor_cost_increase (c : CostChanges) :
  c.labor = 0.08 := by sorry

end labor_cost_increase_l3722_372280


namespace pie_distribution_l3722_372228

theorem pie_distribution (T R B S : ℕ) : 
  R = T / 2 →
  B = R - 14 →
  S = (R + B) / 2 →
  T = R + B + S →
  (T = 42 ∧ R = 21 ∧ B = 7 ∧ S = 14) :=
by sorry

end pie_distribution_l3722_372228


namespace max_distance_complex_l3722_372272

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 81 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((1 + 2*I)*z^4 - z^6) ≤ max_dist :=
sorry

end max_distance_complex_l3722_372272


namespace weaving_factory_profit_maximization_l3722_372207

/-- Represents the profit maximization problem in a weaving factory --/
theorem weaving_factory_profit_maximization 
  (total_workers : ℕ) 
  (fabric_per_worker : ℕ) 
  (clothing_per_worker : ℕ) 
  (fabric_per_clothing : ℚ) 
  (fabric_profit : ℚ) 
  (clothing_profit : ℕ) 
  (h_total : total_workers = 150)
  (h_fabric : fabric_per_worker = 30)
  (h_clothing : clothing_per_worker = 4)
  (h_fabric_clothing : fabric_per_clothing = 3/2)
  (h_fabric_profit : fabric_profit = 2)
  (h_clothing_profit : clothing_profit = 25) :
  ∃ (x : ℕ), 
    x ≤ total_workers ∧ 
    (clothing_profit * clothing_per_worker * x : ℚ) + 
    (fabric_profit * (fabric_per_worker * (total_workers - x) - 
    fabric_per_clothing * clothing_per_worker * x)) = 11800 ∧
    x = 100 := by
  sorry

end weaving_factory_profit_maximization_l3722_372207


namespace profit_percentage_calculation_l3722_372219

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price = 560) :
  (selling_price - cost_price) / cost_price * 100 = 40 := by
  sorry

end profit_percentage_calculation_l3722_372219


namespace simplify_fraction_product_l3722_372271

theorem simplify_fraction_product : (225 : ℚ) / 10125 * 45 = 1 := by
  sorry

end simplify_fraction_product_l3722_372271


namespace equation_solution_l3722_372242

theorem equation_solution (x : ℝ) : x ≠ 3 →
  (x - 7 = (4 * |x - 3|) / (x - 3)) ↔ x = 11 :=
by sorry

end equation_solution_l3722_372242


namespace pizza_delivery_solution_l3722_372238

/-- Represents the pizza delivery problem -/
def PizzaDelivery (total_pizzas : ℕ) (total_time : ℕ) (avg_time_per_stop : ℕ) : Prop :=
  ∃ (two_pizza_stops : ℕ),
    two_pizza_stops * 2 + (total_pizzas - two_pizza_stops * 2) = total_pizzas ∧
    (two_pizza_stops + (total_pizzas - two_pizza_stops * 2)) * avg_time_per_stop = total_time

/-- Theorem stating the solution to the pizza delivery problem -/
theorem pizza_delivery_solution :
  PizzaDelivery 12 40 4 → ∃ (two_pizza_stops : ℕ), two_pizza_stops = 2 :=
by
  sorry


end pizza_delivery_solution_l3722_372238


namespace rotated_point_x_coordinate_l3722_372257

/-- Given a point P(1,2) in the Cartesian plane, prove that when the vector OP
    is rotated counterclockwise by 5π/6 around the origin O to obtain vector OQ,
    the x-coordinate of Q is -√3/2 - 2√5. -/
theorem rotated_point_x_coordinate (P Q : ℝ × ℝ) (h1 : P = (1, 2)) :
  (∃ θ : ℝ, θ = 5 * π / 6 ∧
   Q.1 = P.1 * Real.cos θ - P.2 * Real.sin θ ∧
   Q.2 = P.1 * Real.sin θ + P.2 * Real.cos θ) →
  Q.1 = -Real.sqrt 3 / 2 - 2 * Real.sqrt 5 := by
  sorry

end rotated_point_x_coordinate_l3722_372257


namespace min_value_cos_sum_l3722_372256

theorem min_value_cos_sum (x : ℝ) : 
  ∃ (m : ℝ), m = -Real.sqrt 2 ∧ ∀ y : ℝ, 
    Real.cos (3*y + π/6) + Real.cos (3*y - π/3) ≥ m :=
by sorry

end min_value_cos_sum_l3722_372256


namespace quadratic_polynomial_roots_l3722_372273

theorem quadratic_polynomial_roots (x y : ℝ) (t : ℝ → ℝ) : 
  x + y = 12 → x * (3 * y) = 108 → 
  (∀ r, t r = 0 ↔ r = x ∨ r = y) → 
  t = fun r ↦ r^2 - 12*r + 36 :=
by sorry

end quadratic_polynomial_roots_l3722_372273


namespace trig_identity_l3722_372255

theorem trig_identity (α : ℝ) : 
  Real.sin (π + α)^2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end trig_identity_l3722_372255


namespace bride_groom_age_difference_l3722_372283

theorem bride_groom_age_difference :
  ∀ (bride_age groom_age : ℕ),
    bride_age = 102 →
    bride_age + groom_age = 185 →
    bride_age - groom_age = 19 :=
by
  sorry

end bride_groom_age_difference_l3722_372283


namespace peanut_difference_l3722_372206

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end peanut_difference_l3722_372206


namespace area_ratio_theorem_l3722_372285

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  u : ℝ
  v : ℝ
  w : ℝ

/-- The theorem statement -/
theorem area_ratio_theorem (t : TriangleWithPoints) 
  (h_PQ : t.PQ = 12)
  (h_QR : t.QR = 16)
  (h_PR : t.PR = 20)
  (h_positive : t.u > 0 ∧ t.v > 0 ∧ t.w > 0)
  (h_sum : t.u + t.v + t.w = 3/4)
  (h_sum_squares : t.u^2 + t.v^2 + t.w^2 = 1/2) :
  let area_PQR := (1/2) * t.PQ * t.QR
  let area_XYZ := area_PQR * (1 - (t.u * (1 - t.w) + t.v * (1 - t.u) + t.w * (1 - t.v)))
  area_XYZ / area_PQR = 9/32 := by
  sorry

end area_ratio_theorem_l3722_372285


namespace inverse_of_B_squared_l3722_372243

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![2, 3; 0, -1]) : 
  (B^2)⁻¹ = !![4, 3; 0, 1] := by
sorry

end inverse_of_B_squared_l3722_372243


namespace fraction_value_l3722_372218

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
  sorry

end fraction_value_l3722_372218


namespace arithmetic_sequence_sum_ratio_l3722_372260

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- The theorem states that for an arithmetic sequence where a₄ = 2a₃, 
    the ratio of S₇ to S₅ is equal to 14/5 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (h : a 4 = 2 * a 3) : 
  S 7 a / S 5 a = 14 / 5 := by
  sorry

end arithmetic_sequence_sum_ratio_l3722_372260


namespace min_sum_of_product_l3722_372241

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end min_sum_of_product_l3722_372241


namespace cookie_sharing_proof_l3722_372289

/-- Given a total number of cookies and the number of cookies each person gets,
    calculate the number of people sharing the cookies. -/
def number_of_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

/-- Prove that when sharing 24 cookies equally among people,
    with each person getting 4 cookies, the number of people is 6. -/
theorem cookie_sharing_proof :
  number_of_people 24 4 = 6 := by
  sorry

end cookie_sharing_proof_l3722_372289


namespace tan_60_minus_sin_60_l3722_372224

theorem tan_60_minus_sin_60 : Real.tan (π / 3) - Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end tan_60_minus_sin_60_l3722_372224


namespace quadratic_equation_properties_l3722_372227

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + (2*m - 1)*x + m^2 = 0

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x

def range_of_m : Set ℝ :=
  {m : ℝ | m ≤ 1/4}

def roots_relation (m : ℝ) (α β : ℝ) : Prop :=
  quadratic_equation m α ∧ quadratic_equation m β ∧ α ≠ β

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m → m ∈ range_of_m) ∧
  (∃ m : ℝ, m = -1 ∧ 
    ∃ α β : ℝ, roots_relation m α β ∧ α^2 + β^2 - α*β = 6) :=
sorry

end quadratic_equation_properties_l3722_372227


namespace equation_proof_l3722_372263

theorem equation_proof (x : ℚ) : x = 5 → 65 + (x * 12) / 60 = 66 := by
  sorry

end equation_proof_l3722_372263


namespace ellipse_hyperbola_eccentricity_l3722_372214

/-- Given an ellipse and a hyperbola with the same a and b parameters,
    prove that if the ellipse has eccentricity 1/2,
    then the hyperbola has eccentricity √7/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (∃ c : ℝ, c/a = 1/2)) :
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ c' : ℝ, c'/a = Real.sqrt 7 / 2) :=
by sorry

end ellipse_hyperbola_eccentricity_l3722_372214


namespace coronavirus_recoveries_day2_l3722_372209

/-- Proves that the number of recoveries on day 2 is 50, given the conditions of the Coronavirus case problem. -/
theorem coronavirus_recoveries_day2 
  (initial_cases : ℕ) 
  (day2_increase : ℕ) 
  (day3_new_cases : ℕ) 
  (day3_recoveries : ℕ) 
  (total_cases_day3 : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : day2_increase = 500)
  (h3 : day3_new_cases = 1500)
  (h4 : day3_recoveries = 200)
  (h5 : total_cases_day3 = 3750) :
  ∃ (day2_recoveries : ℕ), 
    initial_cases + day2_increase - day2_recoveries + day3_new_cases - day3_recoveries = total_cases_day3 ∧ 
    day2_recoveries = 50 := by
  sorry

end coronavirus_recoveries_day2_l3722_372209


namespace trigonometric_sequence_solution_l3722_372217

theorem trigonometric_sequence_solution (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, 2 * (Real.cos (a n))^2 = Real.cos (a (n + 1))) →
  (∀ n, Real.cos (a (n + 1)) ≥ 0) →
  (∀ n, |Real.cos (a n)| ≤ 1 / Real.sqrt 2) →
  (∀ n, a (n + 1) = a n + d) →
  (∃ k : ℤ, d = 2 * Real.pi * ↑k ∧ k ≠ 0) →
  (∃ m : ℤ, a 1 = Real.pi / 2 + Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = Real.pi / 3 + 2 * Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = -Real.pi / 3 + 2 * Real.pi * ↑m) :=
by sorry

end trigonometric_sequence_solution_l3722_372217


namespace function_composition_identity_l3722_372220

/-- Piecewise function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 10 - 4 * x

/-- Theorem stating that if f(f(x)) = x for all x, then a + b = 21/4 -/
theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 21/4 := by
  sorry

end function_composition_identity_l3722_372220


namespace binary_1101001101_equals_base4_12021_l3722_372279

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_12021 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  let decimal := binary_to_decimal binary
  let base4 := decimal_to_base4 decimal
  base4 = [1, 2, 0, 2, 1] := by sorry

end binary_1101001101_equals_base4_12021_l3722_372279


namespace arithmetic_evaluation_l3722_372282

theorem arithmetic_evaluation : 4 + 10 / 2 - 2 * 3 = 3 := by
  sorry

end arithmetic_evaluation_l3722_372282


namespace min_white_points_l3722_372269

theorem min_white_points (total_points : ℕ) (h_total : total_points = 100) :
  ∃ (n : ℕ), n = 10 ∧ 
  (∀ (k : ℕ), k < n → k + (k.choose 3) < total_points) ∧
  (n + (n.choose 3) ≥ total_points) := by
  sorry

end min_white_points_l3722_372269


namespace product_of_roots_quadratic_l3722_372275

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ * x₂ = -3 := by
  sorry

end product_of_roots_quadratic_l3722_372275


namespace percent_calculation_l3722_372253

theorem percent_calculation (a b c d e : ℝ) 
  (h1 : c = 0.25 * a)
  (h2 : c = 0.1 * b)
  (h3 : d = 0.5 * b)
  (h4 : d = 0.2 * e)
  (h5 : e = 0.15 * a)
  (h6 : e = 0.05 * c)
  (h7 : a ≠ 0)
  (h8 : c ≠ 0) :
  (d * b + c * e) / (a * c) = 12.65 := by
  sorry

#check percent_calculation

end percent_calculation_l3722_372253


namespace absolute_value_equation_solution_l3722_372286

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |x^2 - 4*x + 4| - (3 - x)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 5) / 2 ∧
              x₂ = (3 - Real.sqrt 5) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end absolute_value_equation_solution_l3722_372286


namespace fraction_equality_l3722_372294

theorem fraction_equality (p q : ℝ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end fraction_equality_l3722_372294


namespace stratified_sample_grade12_l3722_372210

/-- Represents the number of students in a grade -/
structure GradePopulation where
  total : ℕ
  sampled : ℕ

/-- Represents the school population -/
structure SchoolPopulation where
  grade11 : GradePopulation
  grade12 : GradePopulation

/-- Checks if the sampling is stratified (same ratio across grades) -/
def isStratifiedSample (school : SchoolPopulation) : Prop :=
  school.grade11.sampled * school.grade12.total = school.grade12.sampled * school.grade11.total

/-- The main theorem -/
theorem stratified_sample_grade12 (school : SchoolPopulation) 
    (h1 : school.grade11.total = 500)
    (h2 : school.grade12.total = 450)
    (h3 : school.grade11.sampled = 20)
    (h4 : isStratifiedSample school) :
  school.grade12.sampled = 18 := by
  sorry

#check stratified_sample_grade12

end stratified_sample_grade12_l3722_372210


namespace card_arrangements_sum_14_l3722_372249

-- Define the card suits
inductive Suit
| Hearts
| Clubs

-- Define the card values
def CardValue := Fin 4

-- Define a card as a pair of suit and value
def Card := Suit × CardValue

-- Define the deck of 8 cards
def deck : Finset Card := sorry

-- Function to calculate the sum of card values
def sumCardValues (hand : Finset Card) : Nat := sorry

-- Function to count different arrangements
def countArrangements (hand : Finset Card) : Nat := sorry

theorem card_arrangements_sum_14 :
  (Finset.filter (fun hand => hand.card = 4 ∧ sumCardValues hand = 14)
    (Finset.powerset deck)).sum countArrangements = 396 := by
  sorry

end card_arrangements_sum_14_l3722_372249


namespace k_range_when_proposition_p_false_l3722_372223

theorem k_range_when_proposition_p_false (k : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, k * 4^x - k * 2^(x + 1) + 6 * (k - 5) ≠ 0) →
  k ∈ Set.Iio 5 ∪ Set.Ioi 6 :=
sorry

end k_range_when_proposition_p_false_l3722_372223


namespace income_expenditure_ratio_l3722_372208

def income : ℕ := 10000
def savings : ℕ := 2000

def expenditure : ℕ := income - savings

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem income_expenditure_ratio :
  ratio_simplify income expenditure = (5, 4) := by
  sorry

end income_expenditure_ratio_l3722_372208


namespace even_function_property_l3722_372298

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-6) 6, HasDerivAt f (f x) x) →
  IsEven f →
  MonoDecreasing f (-6) 0 →
  f 4 - f 1 > 0 :=
sorry

end even_function_property_l3722_372298


namespace strawberry_picking_problem_l3722_372205

/-- Strawberry picking problem -/
theorem strawberry_picking_problem 
  (brother_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) 
  (equal_share : ℕ) 
  (family_members : ℕ) 
  (h1 : brother_baskets = 3)
  (h2 : strawberries_per_basket = 15)
  (h3 : kimberly_multiplier = 8)
  (h4 : equal_share = 168)
  (h5 : family_members = 4) :
  kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
  (family_members * equal_share - 
   kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
   (brother_baskets * strawberries_per_basket)) = 93 := by
  sorry

end strawberry_picking_problem_l3722_372205


namespace min_value_of_a_l3722_372235

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := 
  fun x => if x > 0 then Real.exp x + a else -(Real.exp (-x) + a)

-- State the theorem
theorem min_value_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) →  -- f is odd
  (∀ x y : ℝ, x < y → f a x < f a y) →  -- f is strictly increasing (monotonic)
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x < f b y) → 
            b ≥ -1 :=
by sorry

end min_value_of_a_l3722_372235


namespace complement_intersection_theorem_l3722_372204

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end complement_intersection_theorem_l3722_372204


namespace extra_apples_l3722_372213

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 25)
  (h2 : green_apples = 17)
  (h3 : students = 10) :
  red_apples + green_apples - students = 32 := by
  sorry

end extra_apples_l3722_372213


namespace coffee_stock_calculation_l3722_372237

/-- Represents the initial amount of coffee in stock -/
def initial_stock : ℝ := sorry

/-- The fraction of initial stock that is decaffeinated -/
def initial_decaf_fraction : ℝ := 0.4

/-- The amount of new coffee purchased -/
def new_purchase : ℝ := 100

/-- The fraction of new purchase that is decaffeinated -/
def new_decaf_fraction : ℝ := 0.6

/-- The fraction of total stock that is decaffeinated after the purchase -/
def final_decaf_fraction : ℝ := 0.44

theorem coffee_stock_calculation :
  initial_stock = 400 ∧
  final_decaf_fraction * (initial_stock + new_purchase) =
    initial_decaf_fraction * initial_stock + new_decaf_fraction * new_purchase :=
sorry

end coffee_stock_calculation_l3722_372237


namespace divisibility_by_nine_implies_divisibility_by_three_l3722_372222

theorem divisibility_by_nine_implies_divisibility_by_three (u v : ℤ) :
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end divisibility_by_nine_implies_divisibility_by_three_l3722_372222


namespace remainder_proof_l3722_372288

theorem remainder_proof : 123456789012 % 252 = 84 := by
  sorry

end remainder_proof_l3722_372288


namespace rectangle_ratio_is_two_l3722_372254

/-- Represents the configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : ℝ
  h_area_ratio : area_ratio = 9
  h_outer_side : outer_side = inner_side + 2 * rect_short
  h_rect_long : rect_long + rect_short = outer_side

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio_is_two (config : SquareRectConfig) :
  config.rect_long / config.rect_short = 2 := by
  sorry

end rectangle_ratio_is_two_l3722_372254


namespace fishing_tomorrow_l3722_372216

/-- Represents the fishing schedule in a coastal village --/
structure FishingVillage where
  daily : ℕ        -- Number of people fishing every day
  everyOther : ℕ   -- Number of people fishing every other day
  everyThree : ℕ   -- Number of people fishing every three days
  yesterday : ℕ    -- Number of people who fished yesterday
  today : ℕ        -- Number of people fishing today

/-- Calculates the number of people fishing tomorrow --/
def tomorrowFishers (v : FishingVillage) : ℕ :=
  v.daily + v.everyThree + (v.everyOther - (v.yesterday - v.daily))

/-- Theorem stating that given the village's fishing pattern, 
    15 people will fish tomorrow --/
theorem fishing_tomorrow (v : FishingVillage) 
  (h1 : v.daily = 7)
  (h2 : v.everyOther = 8)
  (h3 : v.everyThree = 3)
  (h4 : v.yesterday = 12)
  (h5 : v.today = 10) :
  tomorrowFishers v = 15 := by
  sorry

end fishing_tomorrow_l3722_372216


namespace solution_set_inequality_l3722_372225

/-- The solution set of the inequality x(9-x) > 0 is the open interval (0,9) -/
theorem solution_set_inequality (x : ℝ) : x * (9 - x) > 0 ↔ x ∈ Set.Ioo 0 9 := by
  sorry

end solution_set_inequality_l3722_372225
