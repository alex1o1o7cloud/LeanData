import Mathlib

namespace books_sold_is_24_l1112_111298

/-- Calculates the number of books sold to buy a clarinet -/
def books_sold_for_clarinet (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let additional_needed := clarinet_cost - initial_savings
  let halfway_savings := additional_needed / 2
  let total_to_save := halfway_savings + additional_needed
  total_to_save / book_price

theorem books_sold_is_24 :
  books_sold_for_clarinet 10 90 5 = 24 := by
  sorry

#eval books_sold_for_clarinet 10 90 5

end books_sold_is_24_l1112_111298


namespace cross_section_area_less_than_half_face_l1112_111254

/-- A cube with an inscribed sphere and a triangular cross-section touching the sphere -/
structure CubeWithSphereAndCrossSection where
  /-- Side length of the cube -/
  a : ℝ
  /-- Assumption that the cube has positive side length -/
  a_pos : 0 < a
  /-- The triangular cross-section touches the inscribed sphere -/
  touches_sphere : Bool

/-- The area of the triangular cross-section is less than half the area of the cube face -/
theorem cross_section_area_less_than_half_face (cube : CubeWithSphereAndCrossSection) :
  ∃ (area : ℝ), area < (1/2) * cube.a^2 ∧ 
  (∀ (cross_section_area : ℝ), cross_section_area ≤ area) :=
sorry

end cross_section_area_less_than_half_face_l1112_111254


namespace cube_midpoint_planes_l1112_111294

-- Define a cube type
structure Cube where
  -- Add necessary properties of a cube

-- Define a plane type
structure Plane where
  -- Add necessary properties of a plane

-- Define a function to check if a plane contains a midpoint of a cube's edge
def containsMidpoint (p : Plane) (c : Cube) : Prop :=
  sorry

-- Define a function to count the number of midpoints a plane contains
def countMidpoints (p : Plane) (c : Cube) : ℕ :=
  sorry

-- Define a function to check if a plane contains at least 3 midpoints
def containsAtLeastThreeMidpoints (p : Plane) (c : Cube) : Prop :=
  countMidpoints p c ≥ 3

-- Define a function to count the number of planes containing at least 3 midpoints
def countPlanesWithAtLeastThreeMidpoints (c : Cube) : ℕ :=
  sorry

-- Theorem statement
theorem cube_midpoint_planes (c : Cube) :
  countPlanesWithAtLeastThreeMidpoints c = 81 :=
sorry

end cube_midpoint_planes_l1112_111294


namespace tire_circumference_l1112_111201

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference
  (revolutions_per_minute : ℝ)
  (car_speed_kmh : ℝ)
  (h1 : revolutions_per_minute = 400)
  (h2 : car_speed_kmh = 48)
  : ∃ (circumference : ℝ), circumference = 2 := by
  sorry

end tire_circumference_l1112_111201


namespace regression_relationships_l1112_111292

/-- Represents the possibility that x is not related to y -/
def notRelatedPossibility : ℝ → ℝ := sorry

/-- Represents the fitting effect of the regression line -/
def fittingEffect : ℝ → ℝ := sorry

/-- Represents the degree of fit -/
def degreeOfFit : ℝ → ℝ := sorry

theorem regression_relationships :
  (∀ k₁ k₂ : ℝ, k₁ < k₂ → notRelatedPossibility k₁ > notRelatedPossibility k₂) ∧
  (∀ s₁ s₂ : ℝ, s₁ < s₂ → fittingEffect s₁ > fittingEffect s₂) ∧
  (∀ r₁ r₂ : ℝ, r₁ < r₂ → degreeOfFit r₁ < degreeOfFit r₂) :=
by sorry

end regression_relationships_l1112_111292


namespace fraction_product_l1112_111283

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by sorry

end fraction_product_l1112_111283


namespace sector_central_angle_l1112_111246

theorem sector_central_angle (area : Real) (radius : Real) (central_angle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  area = 1 / 2 * central_angle * radius ^ 2 →
  central_angle = 3 * Real.pi / 4 := by
  sorry

end sector_central_angle_l1112_111246


namespace fraction_equality_l1112_111239

theorem fraction_equality (a : ℕ+) : (a : ℚ) / ((a : ℚ) + 36) = 775 / 1000 → a = 124 := by
  sorry

end fraction_equality_l1112_111239


namespace symmetric_point_of_P_l1112_111232

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point P
def P : Point := (-1, 2)

-- Define symmetry with respect to the origin
def symmetricPoint (p : Point) : Point :=
  (-p.1, -p.2)

-- Theorem statement
theorem symmetric_point_of_P :
  symmetricPoint P = (1, -2) := by
  sorry

end symmetric_point_of_P_l1112_111232


namespace arithmetic_sequence_intersection_l1112_111241

/-- Given two arithmetic sequences {a_n} and {b_n}, prove that they intersect at n = 5 -/
theorem arithmetic_sequence_intersection :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  let b : ℕ → ℤ := λ n => -2 + 4 * (n - 1)
  ∃! n : ℕ, a n = b n ∧ n = 5 := by
  sorry

end arithmetic_sequence_intersection_l1112_111241


namespace max_sum_on_circle_l1112_111245

def circle_equation (x y : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 36

theorem max_sum_on_circle :
  ∃ (max : ℤ),
    (∀ x y : ℤ, circle_equation x y → x + y ≤ max) ∧
    (∃ x y : ℤ, circle_equation x y ∧ x + y = max) ∧
    max = 8 := by
  sorry

end max_sum_on_circle_l1112_111245


namespace paint_usage_l1112_111238

theorem paint_usage (mary_paint mike_paint sun_paint total_paint : ℝ) 
  (h1 : mike_paint = mary_paint + 2)
  (h2 : sun_paint = 5)
  (h3 : total_paint = 13)
  (h4 : mary_paint + mike_paint + sun_paint = total_paint) :
  mary_paint = 3 := by
sorry

end paint_usage_l1112_111238


namespace hyperbola_equation_l1112_111277

/-- A hyperbola is defined by its equation in the form ax^2 + by^2 = c,
    where a, b, and c are real numbers and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_opposite_signs : a * b < 0

/-- The point (x, y) in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  h.a * p.x^2 + h.b * p.y^2 = h.c

/-- Two hyperbolas have the same asymptotes if their equations are proportional -/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ h1.a = k * h2.a ∧ h1.b = k * h2.b

theorem hyperbola_equation (h1 : Hyperbola) (h2 : Hyperbola) (p : Point) :
  same_asymptotes h1 { a := 1, b := -1/4, c := 1, h_opposite_signs := sorry } →
  point_on_hyperbola h2 { x := 2, y := 0 } →
  h2 = { a := 1/4, b := -1/16, c := 1, h_opposite_signs := sorry } :=
sorry

end hyperbola_equation_l1112_111277


namespace triangle_perimeter_bounds_l1112_111235

theorem triangle_perimeter_bounds (a b c : ℝ) (h : a * b + b * c + c * a = 12) :
  let k := a + b + c
  6 ≤ k ∧ k ≤ 4 * Real.sqrt 3 := by
  sorry

end triangle_perimeter_bounds_l1112_111235


namespace pentagon_cannot_tile_l1112_111252

-- Define a type for regular polygons
inductive RegularPolygon
  | Hexagon
  | Pentagon
  | Square
  | Triangle

-- Function to calculate the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℝ :=
  match p with
  | .Hexagon => 120
  | .Pentagon => 108
  | .Square => 90
  | .Triangle => 60

-- Function to check if a polygon can tile the plane
def canTilePlane (p : RegularPolygon) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ n * interiorAngle p = 360

-- Theorem stating that only the pentagon cannot tile the plane
theorem pentagon_cannot_tile :
  ∀ (p : RegularPolygon), ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
sorry

end pentagon_cannot_tile_l1112_111252


namespace at_least_one_zero_negation_l1112_111230

theorem at_least_one_zero_negation (a b : ℝ) :
  ¬(a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0) := by sorry

end at_least_one_zero_negation_l1112_111230


namespace eighth_group_selection_l1112_111291

/-- Represents the systematic sampling method for a population --/
def systematicSampling (populationSize : Nat) (groupCount : Nat) (t : Nat) : Nat → Nat :=
  fun k => (t + k - 1) % 10 + (k - 1) * 10

/-- Theorem stating the correct number selected from the 8th group --/
theorem eighth_group_selection
  (populationSize : Nat)
  (groupCount : Nat)
  (t : Nat)
  (h1 : populationSize = 100)
  (h2 : groupCount = 10)
  (h3 : t = 7) :
  systematicSampling populationSize groupCount t 8 = 75 := by
  sorry

#check eighth_group_selection

end eighth_group_selection_l1112_111291


namespace problem_solution_l1112_111282

theorem problem_solution (x : ℚ) : x - 2/5 = 7/15 - 1/3 - 1/6 → x = 11/30 := by
  sorry

end problem_solution_l1112_111282


namespace equal_distance_travel_l1112_111215

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 4) (h3 : v3 = 5) (h4 : t = 47/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 3 :=
by sorry

end equal_distance_travel_l1112_111215


namespace news_spread_theorem_l1112_111224

/-- Represents the spread of news in a village -/
structure NewsSpread where
  residents : ℕ
  start_date : ℕ
  current_date : ℕ
  informed_residents : Finset ℕ

/-- The number of days since the news started spreading -/
def days_passed (ns : NewsSpread) : ℕ :=
  ns.current_date - ns.start_date

/-- Predicate to check if all residents are informed -/
def all_informed (ns : NewsSpread) : Prop :=
  ns.informed_residents.card = ns.residents

theorem news_spread_theorem (ns : NewsSpread) 
  (h_residents : ns.residents = 20)
  (h_start : ns.start_date = 1) :
  (∃ d₁ d₂, d₁ ≤ 15 ∧ d₂ ≥ 18 ∧ days_passed {ns with current_date := ns.start_date + d₁} < ns.residents ∧
            all_informed {ns with current_date := ns.start_date + d₂}) ∧
  (∀ d, d > 20 → all_informed {ns with current_date := ns.start_date + d}) :=
by sorry

end news_spread_theorem_l1112_111224


namespace star_three_neg_two_thirds_l1112_111222

-- Define the ☆ operation
def star (x y : ℚ) : ℚ := x^2 + x*y

-- State the theorem
theorem star_three_neg_two_thirds : star 3 (-2/3) = 7 := by
  sorry

end star_three_neg_two_thirds_l1112_111222


namespace combinations_equal_twelve_l1112_111240

/-- The number of wall color choices -/
def wall_colors : Nat := 4

/-- The number of flooring type choices -/
def flooring_types : Nat := 3

/-- The total number of combinations of wall color and flooring type -/
def total_combinations : Nat := wall_colors * flooring_types

/-- Theorem: The total number of combinations is 12 -/
theorem combinations_equal_twelve : total_combinations = 12 := by
  sorry

end combinations_equal_twelve_l1112_111240


namespace complex_division_result_l1112_111244

theorem complex_division_result (z : ℂ) (h : z = 1 + I) : z / (1 - I) = I := by
  sorry

end complex_division_result_l1112_111244


namespace prism_volume_l1112_111285

/-- 
A right rectangular prism with one side length of 4 inches, 
and two faces with areas of 24 and 16 square inches respectively, 
has a volume of 64 cubic inches.
-/
theorem prism_volume : 
  ∀ (x y z : ℝ), 
  x = 4 → 
  x * y = 24 → 
  y * z = 16 → 
  x * y * z = 64 := by
sorry

end prism_volume_l1112_111285


namespace vector_operation_l1112_111272

theorem vector_operation (a b c : ℝ × ℝ × ℝ) :
  a = (2, 0, 1) →
  b = (-3, 1, -1) →
  c = (1, 1, 0) →
  a + 2 • b - 3 • c = (-7, -1, -1) := by
sorry

end vector_operation_l1112_111272


namespace initial_type_x_plants_l1112_111258

def initial_total : ℕ := 50
def final_total : ℕ := 1042
def days : ℕ := 12
def x_growth_factor : ℕ := 2^4  -- Type X doubles 4 times in 12 days
def y_growth_factor : ℕ := 3^3  -- Type Y triples 3 times in 12 days

theorem initial_type_x_plants : 
  ∃ (x y : ℕ), 
    x + y = initial_total ∧ 
    x_growth_factor * x + y_growth_factor * y = final_total ∧ 
    x = 28 := by
  sorry

end initial_type_x_plants_l1112_111258


namespace triangle_cosine_values_l1112_111247

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a - c = (√6/6)b and sin B = √6 sin C, then cos A = √6/4 and cos(2A - π/6) = (√15 - √3)/8 -/
theorem triangle_cosine_values (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b)
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) :
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.cos (2 * A - π / 6) = (Real.sqrt 15 - Real.sqrt 3) / 8 := by
  sorry

end triangle_cosine_values_l1112_111247


namespace percentage_problem_l1112_111214

theorem percentage_problem (x : ℝ) (h : 75 = 0.6 * x) : x = 125 := by
  sorry

end percentage_problem_l1112_111214


namespace youtube_video_length_l1112_111299

theorem youtube_video_length (total_time : ℕ) (video1_length : ℕ) (video2_length : ℕ) :
  total_time = 510 ∧
  video1_length = 120 ∧
  video2_length = 270 →
  ∃ (last_video_length : ℕ),
    last_video_length * 2 = total_time - (video1_length + video2_length) ∧
    last_video_length = 60 := by
  sorry

end youtube_video_length_l1112_111299


namespace ellipse_properties_l1112_111220

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the line l -/
def line_l (x : ℝ) : Prop := x = -3

theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 ∧
  2 * Real.sqrt 3 = 2 * b ∧
  Real.sqrt 2 / 2 = Real.sqrt (a^2 - b^2) / a →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 6 + y^2 / 3 = 1) ∧
  (∃ min_value : ℝ,
    min_value = 0 ∧
    ∀ x y : ℝ,
    ellipse_C x y a b ∧ y > 0 →
    (x + 3)^2 - y^2 ≥ min_value) ∧
  (∃ m : ℝ,
    m = 9/8 ∧
    ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ a b ∧
    ellipse_C x₂ y₂ a b ∧
    y₁ = 1/4 * x₁ + m ∧
    y₂ = 1/4 * x₂ + m ∧
    ∃ xg yg : ℝ,
    ellipse_C xg yg a b ∧
    xg - (-3) = x₂ - x₁ ∧
    yg - 3 = y₂ - y₁) :=
by sorry

end ellipse_properties_l1112_111220


namespace reciprocal_of_negative_half_l1112_111281

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by
  sorry

end reciprocal_of_negative_half_l1112_111281


namespace division_equality_l1112_111248

theorem division_equality : 204 / 12.75 = 16 := by
  -- Given condition
  have h1 : 2.04 / 1.275 = 1.6 := by sorry
  
  -- Define the scaling factor
  let scale : ℝ := 100 / 10
  
  -- Prove that 204 / 12.75 = 16
  sorry

end division_equality_l1112_111248


namespace toms_profit_l1112_111226

def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_price : ℕ := 20
def salt_needed : ℕ := 10
def salt_price : ℚ := 1/5
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

def total_cost : ℚ := 
  (flour_needed / flour_bag_size * flour_bag_price : ℚ) + 
  (salt_needed * salt_price) + 
  promotion_cost

def total_revenue : ℕ := ticket_price * tickets_sold

theorem toms_profit : 
  total_revenue - total_cost = 8798 := by sorry

end toms_profit_l1112_111226


namespace unique_intersection_l1112_111273

-- Define the line equation
def line (x b : ℝ) : ℝ := 2 * x + b

-- Define the parabola equation
def parabola (x b : ℝ) : ℝ := x^2 + b * x + 1

-- Define the y-intercept of the parabola
def y_intercept (b : ℝ) : ℝ := parabola 0 b

-- Theorem statement
theorem unique_intersection :
  ∃! b : ℝ, line 0 b = y_intercept b := by sorry

end unique_intersection_l1112_111273


namespace modular_equation_solution_l1112_111289

theorem modular_equation_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n + 3) % 151 = 45 % 151 ∧ n = 109 := by
  sorry

end modular_equation_solution_l1112_111289


namespace silver_dollar_problem_l1112_111229

/-- The problem of calculating the total value of silver dollars -/
theorem silver_dollar_problem (x y : ℕ) : 
  -- Mr. Ha owns x silver dollars, which is 2/3 of Mr. Phung's amount
  x = (2 * y) / 3 →
  -- Mr. Phung has y silver dollars, which is 16 more than Mr. Chiu's amount
  y = 56 + 16 →
  -- The total value of all silver dollars is $483.75
  (x + y + 56 + (((x + y + 56) * 120) / 100)) * (5 / 4) = 96750 / 200 := by
  sorry

end silver_dollar_problem_l1112_111229


namespace intersection_point_correct_l1112_111274

/-- The intersection point of two lines y = -2x and y = x -/
def intersection_point : ℝ × ℝ := (0, 0)

/-- Function representing y = -2x -/
def f (x : ℝ) : ℝ := -2 * x

/-- Function representing y = x -/
def g (x : ℝ) : ℝ := x

/-- Theorem stating that (0, 0) is the unique intersection point of y = -2x and y = x -/
theorem intersection_point_correct :
  (∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2) ∧
  (∀ p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 → p = intersection_point) :=
sorry

end intersection_point_correct_l1112_111274


namespace casey_stay_is_three_months_l1112_111263

/-- Calculates the number of months Casey stays at the motel --/
def casey_stay_duration (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_month : ℕ) (total_savings : ℕ) : ℕ :=
  let monthly_cost_weekly := weekly_rate * weeks_per_month
  let savings_per_month := monthly_cost_weekly - monthly_rate
  total_savings / savings_per_month

/-- Proves that Casey stays for 3 months given the specified rates and savings --/
theorem casey_stay_is_three_months :
  casey_stay_duration 280 1000 4 360 = 3 := by
  sorry

end casey_stay_is_three_months_l1112_111263


namespace steven_shirt_count_l1112_111295

def brian_shirts : ℕ := 3
def andrew_shirts : ℕ := 6 * brian_shirts
def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end steven_shirt_count_l1112_111295


namespace jeans_extra_trips_l1112_111267

theorem jeans_extra_trips (total_trips : ℕ) (jeans_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : jeans_trips = 23) : 
  jeans_trips - (total_trips - jeans_trips) = 6 := by
  sorry

end jeans_extra_trips_l1112_111267


namespace b_min_at_3_l1112_111216

def a (n : ℕ+) : ℕ := n

def S (n : ℕ+) : ℕ := n * (n + 1) / 2

def b (n : ℕ+) : ℚ := (2 * S n + 7) / n

theorem b_min_at_3 :
  ∀ n : ℕ+, n ≠ 3 → b n > b 3 :=
sorry

end b_min_at_3_l1112_111216


namespace largest_product_of_three_primes_digit_sum_l1112_111260

/-- A function that returns true if a number is a single-digit prime -/
def isSingleDigitPrime (p : ℕ) : Prop :=
  p < 10 ∧ Nat.Prime p

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    isSingleDigitPrime d ∧
    isSingleDigitPrime e ∧
    Nat.Prime (d^2 + e^2) ∧
    n = d * e * (d^2 + e^2) ∧
    (∀ (m : ℕ), m > n →
      ¬(∃ (p q r : ℕ), isSingleDigitPrime p ∧
                        isSingleDigitPrime q ∧
                        Nat.Prime r ∧
                        r = p^2 + q^2 ∧
                        m = p * q * r)) ∧
    sumOfDigits n = 11 :=
by sorry

end largest_product_of_three_primes_digit_sum_l1112_111260


namespace intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l1112_111269

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ 5 ≤ x}

-- Theorems to prove
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

theorem complement_B_union_P : (U \ B) ∪ P = {x | x ≤ 0 ∨ 3 < x} := by sorry

theorem intersection_AB_complement_P : (A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2} := by sorry

end intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l1112_111269


namespace cubic_root_sum_l1112_111206

theorem cubic_root_sum (d e f : ℕ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  let x : ℝ := (Real.rpow d (1/3) + Real.rpow e (1/3) + 3) / f
  (27 * x^3 - 15 * x^2 - 9 * x - 3 = 0) →
  d + e + f = 126 := by
  sorry

end cubic_root_sum_l1112_111206


namespace absolute_value_inequality_l1112_111218

theorem absolute_value_inequality (x : ℝ) : ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end absolute_value_inequality_l1112_111218


namespace quadrilateral_diagonal_length_l1112_111268

theorem quadrilateral_diagonal_length 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) (diagonal : ℝ) :
  offset1 = 10 →
  offset2 = 6 →
  area = 240 →
  area = (1 / 2) * diagonal * (offset1 + offset2) →
  diagonal = 30 := by
sorry

end quadrilateral_diagonal_length_l1112_111268


namespace range_of_m_for_B_subset_A_l1112_111255

/-- The set B defined as {x | -m < x < 2} -/
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < 2}

/-- Theorem stating the range of m for which B is a subset of A -/
theorem range_of_m_for_B_subset_A (A : Set ℝ) :
  (∀ m : ℝ, B m ⊆ A) ↔ (∀ m : ℝ, m ≤ (1/2)) :=
sorry

end range_of_m_for_B_subset_A_l1112_111255


namespace sequence_property_l1112_111236

/-- A sequence where all terms are distinct starting from index 2 -/
def DistinctSequence (x : ℕ → ℝ) : Prop :=
  ∀ i j, i ≥ 2 → j ≥ 2 → i ≠ j → x i ≠ x j

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n = (x (n - 1) + 98 * x n + x (n + 1)) / 100

theorem sequence_property (x : ℕ → ℝ) 
    (h1 : DistinctSequence x) 
    (h2 : SatisfiesRecurrence x) : 
  Real.sqrt ((x 2023 - x 1) / 2022 * (2021 / (x 2023 - x 2))) + 2021 = 2022 := by
  sorry

end sequence_property_l1112_111236


namespace cans_per_person_day2_is_2_5_l1112_111271

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_restock : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans each person took on the second day --/
def cans_per_person_day2 (fb : FoodBank) : ℚ :=
  let day1_remaining := fb.initial_stock - fb.day1_people * fb.day1_cans_per_person
  let after_day1_restock := day1_remaining + fb.day1_restock
  let day2_given := fb.total_cans_given - fb.day1_people * fb.day1_cans_per_person
  day2_given / fb.day2_people

/-- Theorem stating that given the conditions, each person took 2.5 cans on the second day --/
theorem cans_per_person_day2_is_2_5 (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_restock = 3000)
  (h7 : fb.total_cans_given = 2500) :
  cans_per_person_day2 fb = 5/2 := by
  sorry

end cans_per_person_day2_is_2_5_l1112_111271


namespace multiply_powers_of_a_l1112_111223

theorem multiply_powers_of_a (a : ℝ) : 5 * a^3 * (3 * a^3) = 15 * a^6 := by
  sorry

end multiply_powers_of_a_l1112_111223


namespace x_fourth_plus_inverse_fourth_l1112_111212

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end x_fourth_plus_inverse_fourth_l1112_111212


namespace abc_equation_solutions_l1112_111259

theorem abc_equation_solutions (a b c : ℕ+) :
  a * b * c + a * b + c = a ^ 3 →
  ((b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1))) :=
sorry

end abc_equation_solutions_l1112_111259


namespace optimal_order_l1112_111231

variable (p1 p2 p3 : ℝ)

-- Probabilities are between 0 and 1
axiom prob_range1 : 0 ≤ p1 ∧ p1 ≤ 1
axiom prob_range2 : 0 ≤ p2 ∧ p2 ≤ 1
axiom prob_range3 : 0 ≤ p3 ∧ p3 ≤ 1

-- Ordering of probabilities
axiom prob_order : p3 < p1 ∧ p1 < p2

-- Function to calculate probability of winning two games in a row
def win_probability (p_first p_second p_third : ℝ) : ℝ :=
  p_first * p_second + (1 - p_first) * p_second * p_third

-- Theorem stating that playing against p2 (highest probability) second is optimal
theorem optimal_order :
  win_probability p1 p2 p3 > win_probability p2 p1 p3 ∧
  win_probability p3 p2 p1 > win_probability p2 p3 p1 :=
sorry

end optimal_order_l1112_111231


namespace division_scaling_certain_number_proof_l1112_111225

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (100 * a) / (100 * b) = c := by
  sorry

theorem certain_number_proof :
  29.94 / 1.45 = 17.7 → 2994 / 14.5 = 17.7 := by
  sorry

end division_scaling_certain_number_proof_l1112_111225


namespace acute_angle_solution_l1112_111249

theorem acute_angle_solution : ∃ x : Real, 
  0 < x ∧ 
  x < π / 2 ∧ 
  2 * (Real.sin x)^2 + Real.sin x - Real.sin (2 * x) = 3 * Real.cos x ∧ 
  x = π / 4 := by
  sorry

end acute_angle_solution_l1112_111249


namespace nominal_rate_for_given_ear_l1112_111270

/-- Given an effective annual rate and compounding frequency, 
    calculate the nominal rate of interest per annum. -/
def nominal_rate (ear : ℝ) (n : ℕ) : ℝ :=
  n * ((1 + ear) ^ (1 / n) - 1)

/-- Theorem stating that for an effective annual rate of 12.36% 
    with half-yearly compounding, the nominal rate is approximately 11.66% -/
theorem nominal_rate_for_given_ear :
  let ear := 0.1236
  let n := 2
  abs (nominal_rate ear n - 0.1166) < 0.0001 := by sorry

end nominal_rate_for_given_ear_l1112_111270


namespace negation_of_for_all_positive_negation_of_specific_quadratic_l1112_111296

theorem negation_of_for_all_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by sorry

theorem negation_of_specific_quadratic :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 3 * x + 4 ≤ 0) := by
  apply negation_of_for_all_positive (fun x ↦ 2 * x^2 - 3 * x + 4)

end negation_of_for_all_positive_negation_of_specific_quadratic_l1112_111296


namespace cubic_equation_ratio_l1112_111256

theorem cubic_equation_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) →
  c / d = -1 / 12 := by
  sorry

end cubic_equation_ratio_l1112_111256


namespace money_relation_l1112_111261

theorem money_relation (a b : ℝ) 
  (h1 : 8 * a - b = 98) 
  (h2 : 2 * a + b > 36) : 
  a > 13.4 ∧ b > 9.2 := by
sorry

end money_relation_l1112_111261


namespace interval_covering_theorem_l1112_111279

/-- Definition of the interval I_k -/
def I (a : ℝ → ℝ) (k : ℕ) : Set ℝ := {x | a k ≤ x ∧ x ≤ a k + 1}

/-- The main theorem stating the minimum and maximum values of N -/
theorem interval_covering_theorem (N : ℕ) (a : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 100, ∃ k ∈ Finset.range N, x ∈ I a k) →
  (∀ k ∈ Finset.range N, ∃ x ∈ Set.Icc 0 100, ∀ i ∈ Finset.range N, i ≠ k → x ∉ I a i) →
  100 ≤ N ∧ N ≤ 200 := by
  sorry

end interval_covering_theorem_l1112_111279


namespace max_divisors_1_to_20_l1112_111266

def divisorCount (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

#eval maxDivisorCount 20
#eval divisorCount 12
#eval divisorCount 18
#eval divisorCount 20

end max_divisors_1_to_20_l1112_111266


namespace longest_tape_measure_l1112_111227

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 2400) 
  (hb : b = 3600) 
  (hc : c = 5400) : 
  Nat.gcd a (Nat.gcd b c) = 300 := by
  sorry

end longest_tape_measure_l1112_111227


namespace rectangle_perimeter_l1112_111288

theorem rectangle_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 192) :
  2 * (b + l) = 64 := by
sorry

end rectangle_perimeter_l1112_111288


namespace arithmetic_sequence_constant_l1112_111284

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_constant
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_const : ∃ k : ℝ, a 2 + a 4 + a 15 = k) :
  ∃ c : ℝ, a 7 = c :=
sorry

end arithmetic_sequence_constant_l1112_111284


namespace board_game_impossibility_l1112_111290

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing two numbers with their difference -/
def replace_with_diff (s : ℤ) (a b : ℤ) : ℤ := s - 2 * min a b

/-- Theorem: It's impossible to reduce the sum of numbers from 1 to 1989 to zero
    by repeatedly replacing any two numbers with their difference -/
theorem board_game_impossibility :
  ∀ (ops : ℕ),
  ∃ (result : ℤ),
  result ≠ 0 ∧
  (∃ (numbers : List ℤ),
    numbers.sum = result ∧
    numbers.length + ops = 1989 ∧
    (∀ (x : ℤ), x ∈ numbers → x ≥ 0)) :=
by sorry


end board_game_impossibility_l1112_111290


namespace equivalent_expression_l1112_111242

theorem equivalent_expression (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x^2 - 1) / x)) = -x / Real.sqrt (x^2 - x + 1) := by
  sorry

end equivalent_expression_l1112_111242


namespace smallest_number_divisible_by_primes_l1112_111237

def primes : List Nat := [11, 17, 19, 23, 29, 37, 41]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ p ∈ lst, (n % p = 0)

theorem smallest_number_divisible_by_primes :
  ∀ n : Nat,
    (n < 3075837206 →
      ¬(is_divisible_by_all (n - 27) primes)) ∧
    (is_divisible_by_all (3075837206 - 27) primes) :=
by sorry

end smallest_number_divisible_by_primes_l1112_111237


namespace arithmetic_sequence_sum_l1112_111275

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ = 5 and a₅ = 33,
    prove that a₃ + a₄ = 38. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 33) :
  a 3 + a 4 = 38 := by
sorry

end arithmetic_sequence_sum_l1112_111275


namespace units_digit_of_fraction_main_theorem_l1112_111204

def numerator : ℕ := 22 * 23 * 24 * 25 * 26 * 27
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) : 
  (n / d) % 10 = ((n % (d * 10)) / d) % 10 :=
sorry

theorem main_theorem : (numerator / denominator) % 10 = 8 :=
sorry

end units_digit_of_fraction_main_theorem_l1112_111204


namespace water_requirement_l1112_111221

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 6000

/-- The number of months the water lasts -/
def num_months : ℕ := 4

/-- The amount of water required per household per month -/
def water_per_household_per_month : ℕ := total_water / (num_households * num_months)

/-- Theorem stating that the water required per household per month is 150 litres -/
theorem water_requirement : water_per_household_per_month = 150 := by
  sorry

end water_requirement_l1112_111221


namespace smallest_maximizer_of_g_l1112_111211

/-- Sum of all positive divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Function g(n) = σ(n) / n -/
def g (n : ℕ) : ℚ := (σ n : ℚ) / n

/-- Theorem stating that 6 is the smallest N maximizing g(n) for 1 ≤ n ≤ 100 -/
theorem smallest_maximizer_of_g :
  ∃ (N : ℕ), N = 6 ∧ 
  (∀ n : ℕ, 1 ≤ n → n ≤ 100 → n ≠ N → g n < g N) ∧
  (∀ m : ℕ, 1 ≤ m → m < N → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ k ≠ m ∧ g m ≤ g k) :=
sorry

end smallest_maximizer_of_g_l1112_111211


namespace airplane_seats_multiple_l1112_111287

theorem airplane_seats_multiple (total_seats first_class_seats : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class_seats = 77)
  (h3 : ∃ m : ℕ, total_seats = first_class_seats + (m * first_class_seats + 2)) :
  ∃ m : ℕ, m = 4 ∧ total_seats = first_class_seats + (m * first_class_seats + 2) :=
by sorry

end airplane_seats_multiple_l1112_111287


namespace remaining_distance_to_hotel_l1112_111251

def totalDistance : ℝ := 1200

def drivingSegments : List (ℝ × ℝ) := [
  (60, 2),   -- 60 miles/hour for 2 hours
  (40, 1),   -- 40 miles/hour for 1 hour
  (70, 2.5), -- 70 miles/hour for 2.5 hours
  (50, 4),   -- 50 miles/hour for 4 hours
  (80, 1),   -- 80 miles/hour for 1 hour
  (60, 3)    -- 60 miles/hour for 3 hours
]

def distanceTraveled : ℝ := (drivingSegments.map (fun (speed, time) => speed * time)).sum

theorem remaining_distance_to_hotel : 
  totalDistance - distanceTraveled = 405 := by sorry

end remaining_distance_to_hotel_l1112_111251


namespace book_purchase_ratio_l1112_111203

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

/-- The number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased only book B -/
def only_B : ℕ := both / 2

/-- The total number of people who purchased book A -/
def total_A : ℕ := only_A + both

/-- The total number of people who purchased book B -/
def total_B : ℕ := only_B + both

/-- The ratio of people who purchased book A to those who purchased book B is 2:1 -/
theorem book_purchase_ratio : total_A / total_B = 2 := by
  sorry

end book_purchase_ratio_l1112_111203


namespace number_of_employees_l1112_111243

/-- Proves the number of employees in an organization given salary information --/
theorem number_of_employees
  (avg_salary : ℝ)
  (new_avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : new_avg_salary = 1650)
  (h3 : manager_salary = 4650) :
  ∃ (num_employees : ℕ),
    (num_employees : ℝ) * avg_salary + manager_salary = (num_employees + 1) * new_avg_salary ∧
    num_employees = 20 := by
  sorry


end number_of_employees_l1112_111243


namespace insertion_methods_eq_336_l1112_111276

/- Given 5 books originally and 3 books to insert -/
def original_books : ℕ := 5
def books_to_insert : ℕ := 3

/- The number of gaps increases after each insertion -/
def gaps (n : ℕ) : ℕ := n + 1

/- The total number of insertion methods -/
def insertion_methods : ℕ :=
  (gaps original_books) * (gaps (original_books + 1)) * (gaps (original_books + 2))

/- Theorem stating that the number of insertion methods is 336 -/
theorem insertion_methods_eq_336 : insertion_methods = 336 := by
  sorry

end insertion_methods_eq_336_l1112_111276


namespace smallest_binary_divisible_by_225_l1112_111264

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_divisible_by_225 :
  ∃ (n : ℕ), is_binary_number n ∧ 225 ∣ n ∧
  ∀ (m : ℕ), is_binary_number m → 225 ∣ m → n ≤ m :=
by
  -- The proof would go here
  sorry

#eval (11111111100 : ℕ).digits 10  -- To verify the number in base 10
#eval 11111111100 % 225  -- To verify divisibility by 225

end smallest_binary_divisible_by_225_l1112_111264


namespace linear_function_quadrants_l1112_111205

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

/-- Predicate for a point (x, y) being in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the graph of y = 2x + 1 passes through quadrants I, II, and IV -/
theorem linear_function_quadrants :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (y₁ = LinearFunction 2 1 x₁) ∧ InQuadrantI x₁ y₁ ∧
    (y₂ = LinearFunction 2 1 x₂) ∧ InQuadrantII x₂ y₂ ∧
    (y₃ = LinearFunction 2 1 x₃) ∧ InQuadrantIV x₃ y₃ :=
by
  sorry


end linear_function_quadrants_l1112_111205


namespace geometric_sequence_a1_l1112_111200

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 2) * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 4 = 8) : 
  a 1 = 64 := by
  sorry

end geometric_sequence_a1_l1112_111200


namespace sixth_number_tenth_row_l1112_111278

/-- Represents a triangular number array with specific properties -/
structure TriangularArray where
  -- The first number of each row forms an arithmetic sequence
  first_term : ℚ
  common_difference : ℚ
  -- The numbers in each row form a geometric sequence
  common_ratio : ℚ

/-- Get the nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Get the nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The main theorem -/
theorem sixth_number_tenth_row (arr : TriangularArray) 
  (h1 : arr.first_term = 1/4)
  (h2 : arr.common_difference = 1/4)
  (h3 : arr.common_ratio = 1/2) :
  let first_number_tenth_row := arithmeticSequenceTerm arr.first_term arr.common_difference 10
  geometricSequenceTerm first_number_tenth_row arr.common_ratio 6 = 5/64 := by
  sorry

end sixth_number_tenth_row_l1112_111278


namespace previous_day_visitors_count_l1112_111207

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 666

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 566

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitors_count : previous_day_visitors = 100 := by
  sorry

end previous_day_visitors_count_l1112_111207


namespace complement_of_union_l1112_111209

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_of_union :
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end complement_of_union_l1112_111209


namespace remainder_problem_l1112_111213

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 186 = 19)
  (hy : y % 248 = 23)
  (hz : z % 372 = 29) :
  ((x * y * z) + 47) % 93 = 6 := by
  sorry

end remainder_problem_l1112_111213


namespace red_balls_count_l1112_111257

theorem red_balls_count (total_balls : ℕ) (red_frequency : ℚ) (h1 : total_balls = 40) (h2 : red_frequency = 15 / 100) : 
  ⌊total_balls * red_frequency⌋ = 6 := by
sorry

end red_balls_count_l1112_111257


namespace ratio_equality_l1112_111202

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- Theorem stating the equality of the ratio of specific terms in sequences a and b -/
theorem ratio_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end ratio_equality_l1112_111202


namespace complement_intersection_theorem_l1112_111233

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_theorem :
  ((U \ M) ∩ N) = {-3, -4} := by
  sorry

end complement_intersection_theorem_l1112_111233


namespace marian_needs_31_trays_l1112_111217

/-- The number of trays Marian needs to prepare cookies for classmates and teachers -/
def trays_needed (cookies_for_classmates : ℕ) (cookies_for_teachers : ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (cookies_for_classmates + cookies_for_teachers + cookies_per_tray - 1) / cookies_per_tray

/-- Proof that Marian needs 31 trays to prepare cookies for classmates and teachers -/
theorem marian_needs_31_trays :
  trays_needed 276 92 12 = 31 := by
  sorry

end marian_needs_31_trays_l1112_111217


namespace problem_statement_l1112_111262

def n : ℕ := 2^2015 - 1

def s_q (q k : ℕ) : ℕ := sorry

def f_n (x : ℕ) : ℕ := sorry

def N : ℕ := sorry

theorem problem_statement : 
  N ≡ 382 [MOD 1000] := by sorry

end problem_statement_l1112_111262


namespace r_nonzero_l1112_111297

/-- A polynomial of degree 5 with specific properties -/
def Q (p q r s t : ℝ) (x : ℝ) : ℝ :=
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- The property that Q has five distinct x-intercepts including (0,0) -/
def has_five_distinct_intercepts (p q r s t : ℝ) : Prop :=
  ∃ (α β : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ α ≠ β ∧
    ∀ x, Q p q r s t x = 0 ↔ x = 0 ∨ x = α ∨ x = -α ∨ x = β ∨ x = -β

/-- The theorem stating that r must be non-zero given the conditions -/
theorem r_nonzero (p q r s t : ℝ) 
  (h : has_five_distinct_intercepts p q r s t) : r ≠ 0 := by
  sorry

end r_nonzero_l1112_111297


namespace min_pizzas_for_johns_car_l1112_111219

/-- Calculates the minimum number of pizzas needed to recover car cost -/
def min_pizzas_to_recover_cost (car_cost : ℕ) (earnings_per_pizza : ℕ) (expenses_per_pizza : ℕ) : ℕ :=
  ((car_cost + (earnings_per_pizza - expenses_per_pizza - 1)) / (earnings_per_pizza - expenses_per_pizza))

/-- Theorem: Given the specified conditions, the minimum number of pizzas to recover car cost is 1667 -/
theorem min_pizzas_for_johns_car : 
  min_pizzas_to_recover_cost 5000 10 7 = 1667 := by
  sorry

#eval min_pizzas_to_recover_cost 5000 10 7

end min_pizzas_for_johns_car_l1112_111219


namespace max_value_of_a_l1112_111234

theorem max_value_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_a_l1112_111234


namespace quadratic_equations_solutions_l1112_111210

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 4 * x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 + 6 * x = 5
  let sol1_1 : ℝ := -1 + Real.sqrt 2 / 2
  let sol1_2 : ℝ := -1 - Real.sqrt 2 / 2
  let sol2_1 : ℝ := -3 + Real.sqrt 14
  let sol2_2 : ℝ := -3 - Real.sqrt 14
  (eq1 sol1_1 ∧ eq1 sol1_2) ∧ (eq2 sol2_1 ∧ eq2 sol2_2) := by sorry

end quadratic_equations_solutions_l1112_111210


namespace sum_of_cubes_l1112_111286

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 12) :
  x^3 + y^3 = 91 := by sorry

end sum_of_cubes_l1112_111286


namespace complex_fraction_equality_l1112_111208

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end complex_fraction_equality_l1112_111208


namespace triangle_area_triangle_area_proof_l1112_111250

/-- The area of a triangle with base 18 and height 6 is 54 -/
theorem triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 18 ∧ height = 6 → area = (base * height) / 2 → area = 54

-- The proof is omitted
theorem triangle_area_proof : triangle_area 18 6 54 := by sorry

end triangle_area_triangle_area_proof_l1112_111250


namespace calculate_expression_l1112_111228

theorem calculate_expression : (2 * Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 6 = 7 * Real.sqrt 2 / 2 := by
  sorry

end calculate_expression_l1112_111228


namespace division_problem_l1112_111293

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1575) (h3 : a = b * q + 15) : q = 7 := by
  sorry

end division_problem_l1112_111293


namespace polynomial_identity_l1112_111280

theorem polynomial_identity (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 2 * (a - b) * (b - c) * (c - a) := by
  sorry

end polynomial_identity_l1112_111280


namespace special_square_area_special_square_area_is_64_l1112_111253

/-- A square in the coordinate plane with specific properties -/
structure SpecialSquare where
  verticesOnY2 : ℝ × ℝ → Prop
  verticesOnY10 : ℝ × ℝ → Prop
  sidesParallelOrPerpendicular : Prop

/-- The area of the special square is 64 -/
theorem special_square_area (s : SpecialSquare) : ℝ :=
  64

/-- The main theorem stating that the area of the special square is 64 -/
theorem special_square_area_is_64 (s : SpecialSquare) : special_square_area s = 64 := by
  sorry

end special_square_area_special_square_area_is_64_l1112_111253


namespace percentage_of_4_to_50_percentage_of_4_to_50_proof_l1112_111265

theorem percentage_of_4_to_50 : ℝ → Prop :=
  fun x => (4 / 50 * 100 = x) → x = 8

-- The proof goes here
theorem percentage_of_4_to_50_proof : percentage_of_4_to_50 8 := by
  sorry

end percentage_of_4_to_50_percentage_of_4_to_50_proof_l1112_111265
