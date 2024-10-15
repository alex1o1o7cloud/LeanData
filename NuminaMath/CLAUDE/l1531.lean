import Mathlib

namespace NUMINAMATH_CALUDE_two_part_trip_first_part_length_l1531_153139

/-- Proves that in a two-part trip with given conditions, the first part is 30 km long -/
theorem two_part_trip_first_part_length 
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_first_part = 60)
  (h3 : speed_second_part = 30)
  (h4 : average_speed = 40) :
  ∃ (first_part_distance : ℝ),
    first_part_distance = 30 ∧
    first_part_distance / speed_first_part + (total_distance - first_part_distance) / speed_second_part = total_distance / average_speed :=
by sorry

end NUMINAMATH_CALUDE_two_part_trip_first_part_length_l1531_153139


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1531_153195

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let angle_BAC := 2 * Real.pi / 3
  let AB := 3
  true  -- We don't need to specify all properties of the triangle

-- Define point D on BC
def point_D_on_BC (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • B + t • C

-- Define the condition BD = 2DC
def BD_equals_2DC (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = 2/3 ∧ D = (1 - t) • B + t • C

-- Main theorem
theorem triangle_ABC_properties 
  (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_D_on_BC : point_D_on_BC B C D)
  (h_BD_2DC : BD_equals_2DC B C D) :
  (∀ (area_ABC : ℝ), area_ABC = 3 * Real.sqrt 3 → 
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = Real.sqrt 37) ∧
  (∀ (AD : ℝ), AD = 1 → 
    (let area_ABD := Real.sqrt 3 / 4 * 3; true)) :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1531_153195


namespace NUMINAMATH_CALUDE_computer_sales_total_l1531_153108

theorem computer_sales_total (total : ℕ) : 
  (total / 2 : ℕ) + (total / 3 : ℕ) + 12 = total → total = 72 := by
  sorry

end NUMINAMATH_CALUDE_computer_sales_total_l1531_153108


namespace NUMINAMATH_CALUDE_range_of_a_l1531_153172

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 3 ∧ y > 3 ∧ x ≠ y ∧
  x^2 - 3*a*x + 2*a^2 + 1 = 0 ∧ y^2 - 3*a*y + 2*a^2 + 1 = 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 7/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1531_153172


namespace NUMINAMATH_CALUDE_felicity_gasoline_usage_l1531_153148

/-- Represents the fuel consumption and distance data for a road trip. -/
structure RoadTripData where
  felicity_mpg : ℝ
  adhira_mpg : ℝ
  benjamin_ethanol_mpg : ℝ
  benjamin_biodiesel_mpg : ℝ
  total_distance : ℝ
  adhira_felicity_diff : ℝ
  felicity_benjamin_diff : ℝ
  ethanol_ratio : ℝ
  biodiesel_ratio : ℝ
  felicity_adhira_fuel_ratio : ℝ
  benjamin_adhira_fuel_diff : ℝ

/-- Calculates the amount of gasoline used by Felicity given the road trip data. -/
def calculate_felicity_gasoline (data : RoadTripData) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Felicity used 56 gallons of gasoline on her trip. -/
theorem felicity_gasoline_usage (data : RoadTripData) 
    (h1 : data.felicity_mpg = 35)
    (h2 : data.adhira_mpg = 25)
    (h3 : data.benjamin_ethanol_mpg = 30)
    (h4 : data.benjamin_biodiesel_mpg = 20)
    (h5 : data.total_distance = 1750)
    (h6 : data.adhira_felicity_diff = 150)
    (h7 : data.felicity_benjamin_diff = 50)
    (h8 : data.ethanol_ratio = 0.35)
    (h9 : data.biodiesel_ratio = 0.65)
    (h10 : data.felicity_adhira_fuel_ratio = 2)
    (h11 : data.benjamin_adhira_fuel_diff = 5) :
  calculate_felicity_gasoline data = 56 := by
  sorry

end NUMINAMATH_CALUDE_felicity_gasoline_usage_l1531_153148


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1531_153153

theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 4*x + 1 = 0 ∧ (m - 1) * y^2 - 4*y + 1 = 0) ↔ 
  (m ≤ 5 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1531_153153


namespace NUMINAMATH_CALUDE_zero_point_location_l1531_153120

-- Define the function f
variable {f : ℝ → ℝ}

-- Define the property of having exactly one zero point in an interval
def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_point_location (h1 : has_unique_zero f 0 16)
                            (h2 : has_unique_zero f 0 8)
                            (h3 : has_unique_zero f 0 4)
                            (h4 : has_unique_zero f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_point_location_l1531_153120


namespace NUMINAMATH_CALUDE_gcd_299_621_l1531_153102

theorem gcd_299_621 : Nat.gcd 299 621 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_299_621_l1531_153102


namespace NUMINAMATH_CALUDE_platform_length_l1531_153115

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) :
  train_length = 360 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 43.2 →
  (train_speed_kmh * (1000 / 3600) * time_to_pass) - train_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1531_153115


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l1531_153154

/-- The probability of a point randomly chosen within a right-angled triangle
    with legs 8 and 15 lying inside its inscribed circle is 3π/20. -/
theorem inscribed_circle_probability : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let r : ℝ := (a * b) / (a + b + c)
  let triangle_area : ℝ := (1/2) * a * b
  let circle_area : ℝ := Real.pi * r^2
  (circle_area / triangle_area) = (3 * Real.pi) / 20 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l1531_153154


namespace NUMINAMATH_CALUDE_fourth_student_id_l1531_153164

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_id : ℕ

/-- Checks if a given ID is in the systematic sample. -/
def SystematicSample.contains (s : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.first_id + k * s.interval

/-- The theorem to be proved. -/
theorem fourth_student_id
  (s : SystematicSample)
  (h_class_size : s.class_size = 52)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_29 : s.contains 29)
  (h_contains_42 : s.contains 42) :
  s.contains 16 :=
sorry

end NUMINAMATH_CALUDE_fourth_student_id_l1531_153164


namespace NUMINAMATH_CALUDE_min_value_theorem_l1531_153162

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_f1 : f a b 1 = 2) :
  (1 / a + 4 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1531_153162


namespace NUMINAMATH_CALUDE_inequality_system_product_l1531_153189

theorem inequality_system_product (x y : ℤ) : 
  (x^3 + y^2 - 3*y + 1 < 0 ∧ 3*x^3 - y^2 + 3*y > 0) → 
  (∃ (y1 y2 : ℤ), y1 ≠ y2 ∧ 
    (x^3 + y1^2 - 3*y1 + 1 < 0 ∧ 3*x^3 - y1^2 + 3*y1 > 0) ∧
    (x^3 + y2^2 - 3*y2 + 1 < 0 ∧ 3*x^3 - y2^2 + 3*y2 > 0) ∧
    y1 * y2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_product_l1531_153189


namespace NUMINAMATH_CALUDE_chessboard_coverage_l1531_153104

/-- Represents a subregion of a chessboard -/
structure Subregion where
  rows : Finset Nat
  cols : Finset Nat

/-- The chessboard and its subregions -/
structure Chessboard where
  n : Nat
  subregions : Finset Subregion

/-- The semi-perimeter of a subregion -/
def semiPerimeter (s : Subregion) : Nat :=
  s.rows.card + s.cols.card

/-- Whether a subregion covers a cell -/
def covers (s : Subregion) (i j : Nat) : Prop :=
  i ∈ s.rows ∧ j ∈ s.cols

/-- The main diagonal of the chessboard -/
def mainDiagonal (n : Nat) : Set (Nat × Nat) :=
  {p | p.1 = p.2 ∧ p.1 < n}

/-- The theorem to be proved -/
theorem chessboard_coverage (cb : Chessboard) : 
  (∀ s ∈ cb.subregions, semiPerimeter s ≥ cb.n) →
  (∀ p ∈ mainDiagonal cb.n, ∃ s ∈ cb.subregions, covers s p.1 p.2) →
  (cb.subregions.sum (λ s => (s.rows.card * s.cols.card)) ≥ cb.n^2 / 2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l1531_153104


namespace NUMINAMATH_CALUDE_marathon_speed_ratio_l1531_153193

/-- The ratio of average speeds of two marathon runners -/
theorem marathon_speed_ratio (distance : ℝ) (jack_time jill_time : ℝ) 
  (h1 : distance = 41)
  (h2 : jack_time = 4.5)
  (h3 : jill_time = 4.1) : 
  (distance / jack_time) / (distance / jill_time) = 82 / 90 := by
  sorry

end NUMINAMATH_CALUDE_marathon_speed_ratio_l1531_153193


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1531_153199

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 4 < 0} = {x : ℝ | -1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1531_153199


namespace NUMINAMATH_CALUDE_solution_difference_l1531_153114

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

/-- Definition of p and q as solutions to the equation -/
def p_and_q_are_solutions (p q : ℝ) : Prop :=
  equation p ∧ equation q ∧ p ≠ q

theorem solution_difference (p q : ℝ) :
  p_and_q_are_solutions p q → p > q → p - q = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l1531_153114


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1531_153133

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 2/3 ∧ Q = 8/9 ∧ R = -5/9) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1531_153133


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l1531_153151

/-- The market value of a machine after two years of depreciation -/
theorem machine_value_after_two_years
  (purchase_price : ℝ)
  (yearly_depreciation_rate : ℝ)
  (h1 : purchase_price = 8000)
  (h2 : yearly_depreciation_rate = 0.1) :
  purchase_price * (1 - yearly_depreciation_rate)^2 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l1531_153151


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_edge_sum_l1531_153116

/-- Represents a hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edges_equal : edge_length > 0
  total_length_constraint : total_edge_length = 18 * edge_length

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid where
  edge_length : ℝ
  edges_equal : edge_length > 0

/-- Theorem stating the relationship between hexagonal prism and quadrangular pyramid edge lengths -/
theorem quadrangular_pyramid_edge_sum 
  (h : HexagonalPrism) 
  (q : QuadrangularPyramid) 
  (edge_equality : q.edge_length = h.edge_length) :
  8 * q.edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_quadrangular_pyramid_edge_sum_l1531_153116


namespace NUMINAMATH_CALUDE_sqrt_difference_squared_l1531_153121

theorem sqrt_difference_squared : (Real.sqrt 25 - Real.sqrt 9)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_squared_l1531_153121


namespace NUMINAMATH_CALUDE_complex_coordinates_l1531_153100

theorem complex_coordinates (z : ℂ) (h : z = Complex.I * (2 + 4 * Complex.I)) : 
  z.re = -4 ∧ z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l1531_153100


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1531_153160

theorem complex_magnitude_problem (z : ℂ) (h : (3 + 4*Complex.I)*z = 2 + Complex.I) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1531_153160


namespace NUMINAMATH_CALUDE_incorrect_statement_l1531_153157

theorem incorrect_statement : ¬ (∀ a b c : ℝ, a > b → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1531_153157


namespace NUMINAMATH_CALUDE_chess_program_ratio_l1531_153144

theorem chess_program_ratio (total_students : ℕ) (chess_students : ℕ) (tournament_students : ℕ) 
  (h1 : total_students = 24)
  (h2 : tournament_students = 4)
  (h3 : chess_students = 2 * tournament_students)
  : (chess_students : ℚ) / total_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chess_program_ratio_l1531_153144


namespace NUMINAMATH_CALUDE_min_cost_rose_garden_l1531_153187

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : Float
  tulip : Float
  orchid : Float
  rose : Float
  peony : Float

/-- Represents the dimensions of each region in the flower bed -/
structure FlowerBedRegions where
  bottom_left : Nat × Nat
  top_left : Nat × Nat
  bottom_right : Nat × Nat
  middle_right : Nat × Nat
  top_right : Nat × Nat

/-- Calculates the minimum cost for Rose's garden -/
def calculateMinCost (costs : FlowerCost) (regions : FlowerBedRegions) : Float :=
  sorry

/-- Theorem stating that the minimum cost for Rose's garden is $173.75 -/
theorem min_cost_rose_garden (costs : FlowerCost) (regions : FlowerBedRegions) :
  costs.sunflower = 0.75 ∧
  costs.tulip = 1.25 ∧
  costs.orchid = 1.75 ∧
  costs.rose = 2 ∧
  costs.peony = 2.5 ∧
  regions.bottom_left = (7, 2) ∧
  regions.top_left = (5, 5) ∧
  regions.bottom_right = (6, 4) ∧
  regions.middle_right = (8, 3) ∧
  regions.top_right = (8, 3) →
  calculateMinCost costs regions = 173.75 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_rose_garden_l1531_153187


namespace NUMINAMATH_CALUDE_harish_paint_time_l1531_153182

/-- The time it takes Harish to paint the wall alone -/
def harish_time : ℝ := 3

/-- The time it takes Ganpat to paint the wall alone -/
def ganpat_time : ℝ := 6

/-- The time it takes Harish and Ganpat to paint the wall together -/
def combined_time : ℝ := 2

theorem harish_paint_time :
  (1 / harish_time + 1 / ganpat_time = 1 / combined_time) →
  harish_time = 3 := by
sorry

end NUMINAMATH_CALUDE_harish_paint_time_l1531_153182


namespace NUMINAMATH_CALUDE_purchase_cost_l1531_153118

/-- The cost of a single can of soda in dollars -/
def soda_cost : ℝ := 1

/-- The number of soda cans purchased -/
def num_sodas : ℕ := 3

/-- The number of soups purchased -/
def num_soups : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 1

/-- The cost of a single soup in dollars -/
def soup_cost : ℝ := num_sodas * soda_cost

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := num_sodas * soda_cost + num_soups * soup_cost + num_sandwiches * sandwich_cost

theorem purchase_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l1531_153118


namespace NUMINAMATH_CALUDE_chloe_apples_l1531_153166

theorem chloe_apples (chloe_apples dylan_apples : ℕ) : 
  chloe_apples = dylan_apples + 8 →
  dylan_apples = chloe_apples / 3 →
  chloe_apples = 12 := by
sorry

end NUMINAMATH_CALUDE_chloe_apples_l1531_153166


namespace NUMINAMATH_CALUDE_test_score_difference_l1531_153138

theorem test_score_difference (score_60 score_75 score_85 score_95 : ℝ)
  (percent_60 percent_75 percent_85 percent_95 : ℝ) :
  score_60 = 60 ∧ 
  score_75 = 75 ∧ 
  score_85 = 85 ∧ 
  score_95 = 95 ∧
  percent_60 = 0.2 ∧
  percent_75 = 0.4 ∧
  percent_85 = 0.25 ∧
  percent_95 = 0.15 ∧
  percent_60 + percent_75 + percent_85 + percent_95 = 1 →
  let mean := percent_60 * score_60 + percent_75 * score_75 + 
              percent_85 * score_85 + percent_95 * score_95
  let median := score_75
  abs (mean - median) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_test_score_difference_l1531_153138


namespace NUMINAMATH_CALUDE_prob_at_least_one_l1531_153198

theorem prob_at_least_one (P₁ P₂ : ℝ) 
  (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) 
  (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) : 
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_l1531_153198


namespace NUMINAMATH_CALUDE_divisibility_problem_l1531_153181

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) (hy : y ≠ -1) 
  (h : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1531_153181


namespace NUMINAMATH_CALUDE_positive_integer_inequality_l1531_153101

theorem positive_integer_inequality (a b c : ℕ+) 
  (h : (a : ℤ) + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_inequality_l1531_153101


namespace NUMINAMATH_CALUDE_city_graph_property_l1531_153175

/-- A graph representing cities and flights -/
structure CityGraph where
  V : Type*
  E : V → V → Prop
  N : Nat
  vertex_count : Fintype V
  city_count : Fintype.card V = N

/-- Path of length at most 2 between two vertices -/
def PathOfLength2 (G : CityGraph) (u v : G.V) : Prop :=
  G.E u v ∨ ∃ w, G.E u w ∧ G.E w v

/-- The main theorem -/
theorem city_graph_property (G : CityGraph) 
  (not_fully_connected : ∀ v : G.V, ∃ u : G.V, ¬G.E v u)
  (unique_path : ∀ u v : G.V, ∃! p : PathOfLength2 G u v, True) :
  ∃ k : Nat, G.N - 1 = k * k :=
sorry

end NUMINAMATH_CALUDE_city_graph_property_l1531_153175


namespace NUMINAMATH_CALUDE_like_terms_imply_exponents_l1531_153165

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, m1 x y ≠ 0 ∧ m2 x y ≠ 0 → (x = x ∧ y = y)

/-- The first monomial 2x^3y^4 -/
def m1 (x y : ℕ) : ℚ := 2 * (x^3 * y^4)

/-- The second monomial -2x^ay^(2b) -/
def m2 (a b x y : ℕ) : ℚ := -2 * (x^a * y^(2*b))

theorem like_terms_imply_exponents (a b : ℕ) :
  are_like_terms m1 (m2 a b) → a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponents_l1531_153165


namespace NUMINAMATH_CALUDE_system_solution_l1531_153112

theorem system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1531_153112


namespace NUMINAMATH_CALUDE_expression_evaluation_l1531_153174

theorem expression_evaluation : 
  (-1/2)⁻¹ + (π - 3)^0 + |1 - Real.sqrt 2| + Real.sin (45 * π / 180) * Real.sin (30 * π / 180) = 
  5 * Real.sqrt 2 / 4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1531_153174


namespace NUMINAMATH_CALUDE_fathers_age_l1531_153163

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 10 = (1 / 2) * (father_age + 10) → 
  father_age = 50 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1531_153163


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1531_153168

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_three :
  reciprocal (-3) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1531_153168


namespace NUMINAMATH_CALUDE_discount_calculation_l1531_153192

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) : 
  marked_price = 17.5 →
  discount_rate = 0.3 →
  2 * marked_price * (1 - discount_rate) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l1531_153192


namespace NUMINAMATH_CALUDE_ernie_makes_four_circles_l1531_153191

/-- The number of circles Ernie can make -/
def ernies_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) : ℕ :=
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle

/-- Theorem: Given the conditions from the problem, Ernie can make 4 circles -/
theorem ernie_makes_four_circles :
  ernies_circles 80 8 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_makes_four_circles_l1531_153191


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1531_153158

theorem log_sum_equals_two :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1531_153158


namespace NUMINAMATH_CALUDE_cookie_recipe_l1531_153142

-- Define the conversion rates
def quart_to_pint : ℚ := 2
def pint_to_cup : ℚ := 1/4

-- Define the recipe for 24 cookies
def milk_for_24 : ℚ := 4  -- in quarts
def sugar_for_24 : ℚ := 6  -- in cups

-- Define the number of cookies we want to bake
def cookies_to_bake : ℚ := 6

-- Define the scaling factor
def scaling_factor : ℚ := cookies_to_bake / 24

-- Theorem to prove
theorem cookie_recipe :
  (milk_for_24 * quart_to_pint * scaling_factor = 2) ∧
  (sugar_for_24 * scaling_factor = 1.5) := by
  sorry


end NUMINAMATH_CALUDE_cookie_recipe_l1531_153142


namespace NUMINAMATH_CALUDE_fox_kolobok_meeting_l1531_153141

theorem fox_kolobok_meeting (n : ℕ) (m : ℕ) (h1 : n = 14) (h2 : m = 92) :
  ∃ (i j : ℕ) (f : ℕ → ℕ), i ≠ j ∧ i < n ∧ j < n ∧ f i = f j ∧ (∀ k < n, f k ≤ m) :=
by
  sorry

end NUMINAMATH_CALUDE_fox_kolobok_meeting_l1531_153141


namespace NUMINAMATH_CALUDE_equation_solution_l1531_153125

theorem equation_solution :
  ∃ x : ℝ, (3 / (x + 2) - 1 / x = 0) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1531_153125


namespace NUMINAMATH_CALUDE_peter_completes_work_in_35_days_l1531_153156

/-- The number of days Matt and Peter take to complete the work together -/
def total_days_together : ℚ := 20

/-- The number of days Matt and Peter work together before Matt stops -/
def days_worked_together : ℚ := 12

/-- The number of days Peter takes to complete the remaining work after Matt stops -/
def peter_remaining_days : ℚ := 14

/-- The fraction of work completed when Matt and Peter work together for 12 days -/
def work_completed_together : ℚ := days_worked_together / total_days_together

/-- The fraction of work Peter completes after Matt stops -/
def peter_remaining_work : ℚ := 1 - work_completed_together

/-- Peter's work rate (fraction of work completed per day) -/
def peter_work_rate : ℚ := peter_remaining_work / peter_remaining_days

/-- The number of days Peter takes to complete the work separately -/
def peter_total_days : ℚ := 1 / peter_work_rate

theorem peter_completes_work_in_35_days :
  peter_total_days = 35 := by sorry

end NUMINAMATH_CALUDE_peter_completes_work_in_35_days_l1531_153156


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l1531_153179

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let evens := S.filter (λ x => x % 2 = 0)
  let odds := S.filter (λ x => x % 2 ≠ 0)
  odds.card + 1 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l1531_153179


namespace NUMINAMATH_CALUDE_line_condition_perpendicular_condition_equal_intercepts_condition_l1531_153119

/-- The equation of a line with parameter m -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 5 - 2*m = 0

/-- The condition for the equation to represent a line -/
theorem line_condition (m : ℝ) : 
  (∃ x y, line_equation m x y) ↔ m ≠ -1 :=
sorry

/-- The condition for the line to be perpendicular to the x-axis -/
theorem perpendicular_condition (m : ℝ) :
  (m^2 - 2*m - 3 = 0 ∧ 2*m^2 + m - 1 ≠ 0) ↔ m = 1/2 :=
sorry

/-- The condition for the line to have equal intercepts on both axes -/
theorem equal_intercepts_condition (m : ℝ) :
  (∃ a ≠ 0, line_equation m a 0 ∧ line_equation m 0 (-a)) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_line_condition_perpendicular_condition_equal_intercepts_condition_l1531_153119


namespace NUMINAMATH_CALUDE_job_completion_multiple_l1531_153140

/-- Given workers A and B, and their work rates, calculate the multiple of the original job they complete when working together for a given number of days. -/
theorem job_completion_multiple 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days : ℝ) 
  (h1 : days_A > 0) 
  (h2 : days_B > 0) 
  (h3 : work_days > 0) : 
  work_days * (1 / days_A + 1 / days_B) = 4 := by
  sorry

#check job_completion_multiple 45 30 72

end NUMINAMATH_CALUDE_job_completion_multiple_l1531_153140


namespace NUMINAMATH_CALUDE_base_seven_addition_l1531_153135

/-- Given an addition problem in base 7: 5XY₇ + 62₇ = 64X₇, prove that X + Y = 8 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (6 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_base_seven_addition_l1531_153135


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l1531_153194

/-- Ice cream order details and prices -/
structure IceCreamOrder where
  kiddie_price : ℚ
  regular_price : ℚ
  double_price : ℚ
  sprinkles_price : ℚ
  nuts_price : ℚ
  discount_rate : ℚ
  regular_with_nuts : ℕ
  kiddie_with_sprinkles : ℕ
  double_with_both : ℕ
  regular_with_sprinkles : ℕ
  regular_with_nuts_only : ℕ

/-- Calculate the total cost of an ice cream order after applying the discount -/
def total_cost_after_discount (order : IceCreamOrder) : ℚ :=
  let subtotal := 
    order.regular_with_nuts * (order.regular_price + order.nuts_price) +
    order.kiddie_with_sprinkles * (order.kiddie_price + order.sprinkles_price) +
    order.double_with_both * (order.double_price + order.nuts_price + order.sprinkles_price) +
    order.regular_with_sprinkles * (order.regular_price + order.sprinkles_price) +
    order.regular_with_nuts_only * (order.regular_price + order.nuts_price)
  subtotal * (1 - order.discount_rate)

/-- Theorem stating that the given ice cream order costs $49.50 after discount -/
theorem ice_cream_cost_theorem (order : IceCreamOrder) 
  (h1 : order.kiddie_price = 3)
  (h2 : order.regular_price = 4)
  (h3 : order.double_price = 6)
  (h4 : order.sprinkles_price = 1)
  (h5 : order.nuts_price = 3/2)
  (h6 : order.discount_rate = 1/10)
  (h7 : order.regular_with_nuts = 2)
  (h8 : order.kiddie_with_sprinkles = 2)
  (h9 : order.double_with_both = 3)
  (h10 : order.regular_with_sprinkles = 1)
  (h11 : order.regular_with_nuts_only = 1) :
  total_cost_after_discount order = 99/2 :=
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l1531_153194


namespace NUMINAMATH_CALUDE_correct_calculation_l1531_153188

theorem correct_calculation (x : ℤ) (h : x - 59 = 43) : x - 46 = 56 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1531_153188


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1531_153176

-- Define the function f(x) = (2x + 1)³
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1531_153176


namespace NUMINAMATH_CALUDE_prob_different_topics_correct_l1531_153180

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5 / 6

/-- Theorem stating that the probability of two students selecting different topics
    from num_topics options is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end NUMINAMATH_CALUDE_prob_different_topics_correct_l1531_153180


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l1531_153167

/-- Represents the number of handshakes in a gathering of couples -/
def handshakes_in_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands with
    everyone except their spouse and one other person, there are 54 handshakes -/
theorem six_couples_handshakes :
  handshakes_in_gathering 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l1531_153167


namespace NUMINAMATH_CALUDE_unit_conversions_l1531_153146

theorem unit_conversions :
  (∀ (cm : ℝ), cm * (1 / 100) = cm / 100) ∧
  (∀ (hectares : ℝ), hectares * 10000 = hectares * 10000) ∧
  (∀ (kg g : ℝ), kg + g / 1000 = kg + g * (1 / 1000)) ∧
  (∀ (m cm : ℝ), m + cm / 100 = m + cm * (1 / 100)) →
  (120 : ℝ) * (1 / 100) = 1.2 ∧
  (0.3 : ℝ) * 10000 = 3000 ∧
  10 + 10 / 1000 = 10.01 ∧
  1 + 3 / 100 = 1.03 := by
  sorry

end NUMINAMATH_CALUDE_unit_conversions_l1531_153146


namespace NUMINAMATH_CALUDE_system_solution_l1531_153177

theorem system_solution : 
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℚ),
    (x₁ = 0 ∧ y₁ = -1 ∧ z₁ = 1) ∧
    (x₂ = 3 ∧ y₂ = 2 ∧ z₂ = 4) ∧
    (x₁ = (y₁ + 1) / (3 * y₁ - 5) ∧ 
     y₁ = (3 * z₁ - 2) / (2 * z₁ - 3) ∧ 
     z₁ = (3 * x₁ - 1) / (x₁ - 1)) ∧
    (x₂ = (y₂ + 1) / (3 * y₂ - 5) ∧ 
     y₂ = (3 * z₂ - 2) / (2 * z₂ - 3) ∧ 
     z₂ = (3 * x₂ - 1) / (x₂ - 1)) := by
  sorry


end NUMINAMATH_CALUDE_system_solution_l1531_153177


namespace NUMINAMATH_CALUDE_division_equality_l1531_153130

theorem division_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (3 * a * b)) / (b / (3 * a)) = 1 / (b ^ 2) := by
sorry

end NUMINAMATH_CALUDE_division_equality_l1531_153130


namespace NUMINAMATH_CALUDE_grandma_contribution_correct_l1531_153137

/-- Calculates the amount Zoe's grandma gave her for the trip -/
def grandma_contribution (total_cost : ℚ) (candy_bars : ℕ) (profit_per_bar : ℚ) : ℚ :=
  total_cost - (candy_bars : ℚ) * profit_per_bar

/-- Proves that the grandma's contribution is correct -/
theorem grandma_contribution_correct (total_cost : ℚ) (candy_bars : ℕ) (profit_per_bar : ℚ) :
  grandma_contribution total_cost candy_bars profit_per_bar =
  total_cost - (candy_bars : ℚ) * profit_per_bar :=
by
  sorry

#eval grandma_contribution 485 188 (5/4)

end NUMINAMATH_CALUDE_grandma_contribution_correct_l1531_153137


namespace NUMINAMATH_CALUDE_post_office_mail_theorem_l1531_153184

/-- Calculates the total number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) : ℕ :=
  let total_per_day := letters_per_day + packages_per_day
  let total_per_month := total_per_day * days_per_month
  total_per_month * 6

/-- Theorem stating that a post office receiving 60 letters and 20 packages per day
    handles 14400 pieces of mail in six months, assuming 30-day months -/
theorem post_office_mail_theorem :
  mail_in_six_months 60 20 30 = 14400 := by
  sorry

#eval mail_in_six_months 60 20 30

end NUMINAMATH_CALUDE_post_office_mail_theorem_l1531_153184


namespace NUMINAMATH_CALUDE_lateral_surface_area_l1531_153152

-- Define the frustum
structure Frustum where
  r₁ : ℝ  -- upper base radius
  r₂ : ℝ  -- lower base radius
  h : ℝ   -- height
  l : ℝ   -- slant height

-- Define the conditions
def frustum_conditions (f : Frustum) : Prop :=
  f.r₂ = 4 * f.r₁ ∧ f.h = 4 * f.r₁ ∧ f.l = 10

-- Theorem to prove
theorem lateral_surface_area (f : Frustum) 
  (hf : frustum_conditions f) : 
  π * (f.r₁ + f.r₂) * f.l = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_l1531_153152


namespace NUMINAMATH_CALUDE_root_product_theorem_l1531_153134

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1531_153134


namespace NUMINAMATH_CALUDE_vector_problems_l1531_153155

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, -3)

theorem vector_problems :
  -- Part I: Dot product
  (a.1 * b.1 + a.2 * b.2 = -2) ∧
  -- Part II: Parallel vector with given magnitude
  (∃ (c : ℝ × ℝ), (c.2 / c.1 = a.2 / a.1) ∧ 
                  (c.1^2 + c.2^2 = 20) ∧ 
                  ((c = (-2, -4)) ∨ (c = (2, 4)))) ∧
  -- Part III: Perpendicular vectors condition
  (∃ (k : ℝ), ((b.1 + k * a.1)^2 + (b.2 + k * a.2)^2 = 
               (b.1 - k * a.1)^2 + (b.2 - k * a.2)^2) ∧
              (k^2 = 5)) :=
by sorry


end NUMINAMATH_CALUDE_vector_problems_l1531_153155


namespace NUMINAMATH_CALUDE_max_rounds_le_three_l1531_153196

/-- The number of students for a given n -/
def num_students (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The game described in the problem -/
structure ClassroomGame (n : ℕ) where
  n_gt_one : n > 1
  students : ℕ := num_students n
  classrooms : ℕ := n
  capacities : List ℕ := List.range n

/-- The maximum number of rounds possible in the game -/
def max_rounds (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of rounds is at most 3 -/
theorem max_rounds_le_three (n : ℕ) (game : ClassroomGame n) : max_rounds n ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_rounds_le_three_l1531_153196


namespace NUMINAMATH_CALUDE_expression_evaluation_l1531_153126

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1531_153126


namespace NUMINAMATH_CALUDE_system_equations_properties_l1531_153107

theorem system_equations_properties (a : ℝ) (x y : ℝ) 
  (h1 : x + 3*y = 4 - a) 
  (h2 : x - y = 3*a) 
  (h3 : -3 ≤ a ∧ a ≤ 1) :
  (a = -2 → x = -y) ∧ 
  (a = 1 → x + y = 3) ∧ 
  (x ≤ 1 → 1 ≤ y ∧ y ≤ 4) := by
sorry

end NUMINAMATH_CALUDE_system_equations_properties_l1531_153107


namespace NUMINAMATH_CALUDE_cousins_distribution_eq_52_l1531_153111

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_52 : cousins_distribution = 52 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_eq_52_l1531_153111


namespace NUMINAMATH_CALUDE_page_lines_increase_l1531_153113

theorem page_lines_increase (L : ℕ) (h : L + 60 = 240) : 
  (60 : ℝ) / L = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l1531_153113


namespace NUMINAMATH_CALUDE_three_number_difference_l1531_153122

theorem three_number_difference (x y z : ℝ) : 
  x = 2 * y ∧ x = 3 * z ∧ (x + y + z) / 3 = 88 → x - z = 96 := by
  sorry

end NUMINAMATH_CALUDE_three_number_difference_l1531_153122


namespace NUMINAMATH_CALUDE_log_ratio_theorem_l1531_153145

theorem log_ratio_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Real.log (a^3)) / (Real.log (a^2)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_theorem_l1531_153145


namespace NUMINAMATH_CALUDE_gcd_437_323_l1531_153161

theorem gcd_437_323 : Nat.gcd 437 323 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_437_323_l1531_153161


namespace NUMINAMATH_CALUDE_score_statistics_l1531_153169

def scores : List ℕ := [42, 43, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 47, 47]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem score_statistics :
  mode scores = 46 ∧ median scores = 45 := by sorry

end NUMINAMATH_CALUDE_score_statistics_l1531_153169


namespace NUMINAMATH_CALUDE_fruit_purchase_change_l1531_153171

/-- The change received when purchasing fruit -/
def change (a : ℝ) : ℝ := 100 - 3 * a

/-- Theorem stating the change received when purchasing fruit -/
theorem fruit_purchase_change (a : ℝ) (h : a ≤ 30) :
  change a = 100 - 3 * a := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_change_l1531_153171


namespace NUMINAMATH_CALUDE_coin_arrangements_l1531_153128

/-- Represents the number of gold coins -/
def num_gold_coins : Nat := 4

/-- Represents the number of silver coins -/
def num_silver_coins : Nat := 4

/-- Represents the total number of coins -/
def total_coins : Nat := num_gold_coins + num_silver_coins

/-- Calculates the number of ways to arrange gold and silver coins -/
def color_arrangements : Nat := Nat.choose total_coins num_gold_coins

/-- Calculates the number of valid orientations (face up or down) -/
def orientation_arrangements : Nat := total_coins + 1

/-- Theorem: The number of distinguishable arrangements of 8 coins (4 gold and 4 silver)
    stacked so that no two adjacent coins are face to face is 630 -/
theorem coin_arrangements :
  color_arrangements * orientation_arrangements = 630 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangements_l1531_153128


namespace NUMINAMATH_CALUDE_equidistant_point_x_coord_l1531_153159

/-- A point in the coordinate plane equally distant from the x-axis, y-axis, and the line x + y = 4 -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  dist_x_axis : |y| = |x|
  dist_y_axis : |x| = |y|
  dist_line : |x + y - 4| / Real.sqrt 2 = |x|

/-- The x-coordinate of an equidistant point is 2 -/
theorem equidistant_point_x_coord (p : EquidistantPoint) : p.x = 2 := by
  sorry

#check equidistant_point_x_coord

end NUMINAMATH_CALUDE_equidistant_point_x_coord_l1531_153159


namespace NUMINAMATH_CALUDE_f_negative_a_equals_negative_three_l1531_153185

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) - 2 / (2^x + 1)

theorem f_negative_a_equals_negative_three (a : ℝ) (h : f a = 1) : f (-a) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_equals_negative_three_l1531_153185


namespace NUMINAMATH_CALUDE_sum_of_doubles_l1531_153109

theorem sum_of_doubles (a b c d e f : ℚ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_doubles_l1531_153109


namespace NUMINAMATH_CALUDE_cactus_path_problem_l1531_153183

theorem cactus_path_problem (num_plants : ℕ) (camel_steps : ℕ) (kangaroo_jumps : ℕ) (total_distance : ℝ) :
  num_plants = 51 →
  camel_steps = 56 →
  kangaroo_jumps = 14 →
  total_distance = 7920 →
  let num_gaps := num_plants - 1
  let total_camel_steps := num_gaps * camel_steps
  let total_kangaroo_jumps := num_gaps * kangaroo_jumps
  let camel_step_length := total_distance / total_camel_steps
  let kangaroo_jump_length := total_distance / total_kangaroo_jumps
  kangaroo_jump_length - camel_step_length = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_cactus_path_problem_l1531_153183


namespace NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l1531_153149

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M with coordinates (1, 2) -/
def M : ℝ × ℝ := (1, 2)

theorem reflection_of_M_across_x_axis :
  reflect_x M = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l1531_153149


namespace NUMINAMATH_CALUDE_product_mod_23_is_zero_l1531_153143

theorem product_mod_23_is_zero :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_is_zero_l1531_153143


namespace NUMINAMATH_CALUDE_remaining_cakes_l1531_153105

/-- The number of cakes Baker initially had -/
def initial_cakes : ℕ := 167

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 108

/-- Theorem: The number of remaining cakes is 59 -/
theorem remaining_cakes : initial_cakes - sold_cakes = 59 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cakes_l1531_153105


namespace NUMINAMATH_CALUDE_angle_B_magnitude_triangle_area_l1531_153132

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a + 2 * t.c = 2 * t.b * Real.cos t.A

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3

def satisfiesCondition3 (t : Triangle) : Prop :=
  t.a + t.c = 4

-- Theorem 1
theorem angle_B_magnitude (t : Triangle) (h : satisfiesCondition1 t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfiesCondition1 t) 
  (h2 : satisfiesCondition2 t) 
  (h3 : satisfiesCondition3 t) : 
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_magnitude_triangle_area_l1531_153132


namespace NUMINAMATH_CALUDE_natalia_crates_count_l1531_153136

/-- The number of novels Natalia has --/
def novels : ℕ := 145

/-- The number of comics Natalia has --/
def comics : ℕ := 271

/-- The number of documentaries Natalia has --/
def documentaries : ℕ := 419

/-- The number of albums Natalia has --/
def albums : ℕ := 209

/-- The number of items each crate can hold --/
def crate_capacity : ℕ := 9

/-- The total number of items Natalia has --/
def total_items : ℕ := novels + comics + documentaries + albums

/-- The number of crates Natalia needs --/
def crates_needed : ℕ := (total_items + crate_capacity - 1) / crate_capacity

theorem natalia_crates_count : crates_needed = 116 := by
  sorry

end NUMINAMATH_CALUDE_natalia_crates_count_l1531_153136


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l1531_153190

theorem simplified_fraction_ratio (m : ℝ) (c d : ℤ) :
  (∃ (k : ℝ), (5 * m + 15) / 5 = k ∧ k = c * m + d) →
  d / c = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l1531_153190


namespace NUMINAMATH_CALUDE_negation_of_existence_circle_negation_l1531_153127

theorem negation_of_existence (P : ℝ × ℝ → Prop) :
  (¬ ∃ p, P p) ↔ (∀ p, ¬ P p) := by sorry

theorem circle_negation :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_circle_negation_l1531_153127


namespace NUMINAMATH_CALUDE_work_completion_time_l1531_153117

theorem work_completion_time 
  (total_time : ℝ) 
  (joint_work_time : ℝ) 
  (remaining_work_time : ℝ) 
  (h1 : total_time = 24) 
  (h2 : joint_work_time = 16) 
  (h3 : remaining_work_time = 16) : 
  (total_time * remaining_work_time) / (total_time - joint_work_time) = 48 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1531_153117


namespace NUMINAMATH_CALUDE_giraffe_zebra_ratio_is_three_to_one_l1531_153178

/-- Represents the zoo layout and animal distribution --/
structure Zoo where
  tiger_enclosures : ℕ
  zebra_enclosures_per_tiger : ℕ
  tigers_per_enclosure : ℕ
  zebras_per_enclosure : ℕ
  giraffes_per_enclosure : ℕ
  total_animals : ℕ

/-- Calculates the ratio of giraffe enclosures to zebra enclosures --/
def giraffe_zebra_enclosure_ratio (zoo : Zoo) : ℚ :=
  let total_zebra_enclosures := zoo.tiger_enclosures * zoo.zebra_enclosures_per_tiger
  let total_tigers := zoo.tiger_enclosures * zoo.tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zoo.zebras_per_enclosure
  let total_giraffes := zoo.total_animals - total_tigers - total_zebras
  let giraffe_enclosures := total_giraffes / zoo.giraffes_per_enclosure
  giraffe_enclosures / total_zebra_enclosures

/-- The main theorem stating the ratio of giraffe enclosures to zebra enclosures --/
theorem giraffe_zebra_ratio_is_three_to_one (zoo : Zoo)
  (h1 : zoo.tiger_enclosures = 4)
  (h2 : zoo.zebra_enclosures_per_tiger = 2)
  (h3 : zoo.tigers_per_enclosure = 4)
  (h4 : zoo.zebras_per_enclosure = 10)
  (h5 : zoo.giraffes_per_enclosure = 2)
  (h6 : zoo.total_animals = 144) :
  giraffe_zebra_enclosure_ratio zoo = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_zebra_ratio_is_three_to_one_l1531_153178


namespace NUMINAMATH_CALUDE_football_game_attendance_l1531_153173

theorem football_game_attendance (saturday : ℕ) (expected_total : ℕ) : 
  saturday = 80 →
  expected_total = 350 →
  (saturday + (saturday - 20) + (saturday - 20 + 50) + (saturday + (saturday - 20))) - expected_total = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1531_153173


namespace NUMINAMATH_CALUDE_total_cost_is_53_l1531_153103

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 10
def discount_threshold : ℕ := 15
def discount_amount : ℕ := 5

def total_items : ℕ := sandwich_quantity + soda_quantity

def total_cost : ℕ :=
  sandwich_cost * sandwich_quantity + soda_cost * soda_quantity - 
  if total_items > discount_threshold then discount_amount else 0

theorem total_cost_is_53 : total_cost = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_53_l1531_153103


namespace NUMINAMATH_CALUDE_vowels_on_board_l1531_153170

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l1531_153170


namespace NUMINAMATH_CALUDE_m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l1531_153110

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m : ℝ) : Prop :=
  (m - 1) / m = (m - 1) / 2

/-- m = 2 is a sufficient condition for the lines to be parallel -/
theorem m_2_sufficient : are_parallel 2 := by sorry

/-- m = 2 is not a necessary condition for the lines to be parallel -/
theorem m_2_not_necessary : ∃ m : ℝ, m ≠ 2 ∧ are_parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem m_2_sufficient_but_not_necessary : 
  (are_parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ are_parallel m) := by sorry

end NUMINAMATH_CALUDE_m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l1531_153110


namespace NUMINAMATH_CALUDE_train_crossing_time_l1531_153197

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 225 →
  train_speed_kmh = 90 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1531_153197


namespace NUMINAMATH_CALUDE_circle_area_with_constraints_fountain_base_area_l1531_153123

/-- The area of a circle with specific constraints -/
theorem circle_area_with_constraints (d : ℝ) (r : ℝ) :
  d = 20 →  -- diameter is 20 feet
  r ^ 2 = 10 ^ 2 + 15 ^ 2 →  -- radius squared equals 10^2 + 15^2 (from Pythagorean theorem)
  π * r ^ 2 = 325 * π := by
  sorry

/-- The main theorem proving the area of the circular base -/
theorem fountain_base_area : ∃ (A : ℝ), A = 325 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_constraints_fountain_base_area_l1531_153123


namespace NUMINAMATH_CALUDE_expression_simplification_and_value_l1531_153129

theorem expression_simplification_and_value (a : ℤ) 
  (h1 : 0 < a) (h2 : a < Int.floor (Real.sqrt 5)) : 
  (((a^2 - 1) / (a^2 + 2*a)) / ((a - 1) / a) - a / (a + 2) : ℚ) = 1 / (a + 2) ∧
  (((1^2 - 1) / (1^2 + 2*1)) / ((1 - 1) / 1) - 1 / (1 + 2) : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_value_l1531_153129


namespace NUMINAMATH_CALUDE_branches_per_tree_is_100_l1531_153147

/-- Represents the farm with trees and their branches -/
structure Farm where
  num_trees : ℕ
  leaves_per_subbranch : ℕ
  subbranches_per_branch : ℕ
  total_leaves : ℕ

/-- Calculates the number of branches per tree on the farm -/
def branches_per_tree (f : Farm) : ℕ :=
  f.total_leaves / (f.num_trees * f.subbranches_per_branch * f.leaves_per_subbranch)

/-- Theorem stating that the number of branches per tree is 100 -/
theorem branches_per_tree_is_100 (f : Farm) 
  (h1 : f.num_trees = 4)
  (h2 : f.leaves_per_subbranch = 60)
  (h3 : f.subbranches_per_branch = 40)
  (h4 : f.total_leaves = 96000) : 
  branches_per_tree f = 100 := by
  sorry

#eval branches_per_tree { num_trees := 4, leaves_per_subbranch := 60, subbranches_per_branch := 40, total_leaves := 96000 }

end NUMINAMATH_CALUDE_branches_per_tree_is_100_l1531_153147


namespace NUMINAMATH_CALUDE_remainder_theorem_l1531_153150

theorem remainder_theorem :
  (7 * 10^20 + 3^20) % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1531_153150


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1531_153186

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ + x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1531_153186


namespace NUMINAMATH_CALUDE_dollar_square_sum_l1531_153106

/-- The dollar operation -/
def dollar (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x + y)²$(y + x)² -/
theorem dollar_square_sum (x y : ℝ) : 
  dollar ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_square_sum_l1531_153106


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1531_153124

/-- The total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 112.5

/-- Theorem stating that the tank capacity is correct given the problem conditions. -/
theorem tank_capacity_proof :
  tank_capacity = 112.5 ∧
  (0.5 * tank_capacity = 0.9 * tank_capacity - 45) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1531_153124


namespace NUMINAMATH_CALUDE_smallest_n_for_geometric_sums_l1531_153131

def is_geometric_sum (x : ℕ) : Prop :=
  ∃ (a r : ℕ), r > 1 ∧ x = a + a*r + a*r^2

theorem smallest_n_for_geometric_sums : 
  (∀ n : ℕ, n < 6 → ¬(is_geometric_sum (7*n + 1) ∧ is_geometric_sum (8*n + 1))) ∧
  (is_geometric_sum (7*6 + 1) ∧ is_geometric_sum (8*6 + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_geometric_sums_l1531_153131
