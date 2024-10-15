import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l393_39318

theorem arithmetic_geometric_sequence_solution :
  ∀ a b c : ℝ,
  (b - a = c - b) →                      -- arithmetic sequence
  (a + b + c = 12) →                     -- sum is 12
  ((b + 2)^2 = (a + 2) * (c + 5)) →      -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l393_39318


namespace NUMINAMATH_CALUDE_collinear_probability_value_l393_39377

/-- A 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := 25

/-- The number of sets of 5 collinear dots in the grid -/
def collinear_sets : ℕ := 12

/-- The number of ways to choose 5 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots 5

/-- The probability of selecting 5 collinear dots from the grid -/
def collinear_probability : ℚ := collinear_sets / total_choices

theorem collinear_probability_value :
  collinear_probability = 12 / 53130 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_value_l393_39377


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l393_39337

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l393_39337


namespace NUMINAMATH_CALUDE_jeans_pricing_l393_39329

theorem jeans_pricing (C : ℝ) (h1 : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.96 * C
  (customer_price - retailer_price) / retailer_price = 0.40 := by sorry

end NUMINAMATH_CALUDE_jeans_pricing_l393_39329


namespace NUMINAMATH_CALUDE_firm_ratio_l393_39361

theorem firm_ratio (partners associates : ℕ) : 
  partners = 14 ∧ 
  14 * 34 = associates + 35 → 
  (partners : ℚ) / associates = 2 / 63 := by
sorry

end NUMINAMATH_CALUDE_firm_ratio_l393_39361


namespace NUMINAMATH_CALUDE_pyramid_face_area_l393_39384

/-- The total area of triangular faces of a right square-based pyramid -/
theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : 
  base_edge = 8 → lateral_edge = 9 → 
  (4 * (1/2 * base_edge * Real.sqrt ((lateral_edge^2) - (base_edge/2)^2))) = 16 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_face_area_l393_39384


namespace NUMINAMATH_CALUDE_optimal_solution_l393_39342

-- Define the problem parameters
def days_together : ℝ := 12
def cost_A_per_day : ℝ := 40000
def cost_B_per_day : ℝ := 30000
def max_days : ℝ := 30 -- one month

-- Define the relationship between Team A and Team B's completion times
def team_B_multiplier : ℝ := 1.5

-- Define the function to calculate days needed for Team A
def days_A (x : ℝ) : Prop := (1 / x) + (1 / (team_B_multiplier * x)) = (1 / days_together)

-- Define the function to calculate days needed for Team B
def days_B (x : ℝ) : ℝ := team_B_multiplier * x

-- Define the cost function for Team A working alone
def cost_A (x : ℝ) : ℝ := cost_A_per_day * x

-- Define the cost function for Team B working alone
def cost_B (x : ℝ) : ℝ := cost_B_per_day * days_B x

-- Define the cost function for both teams working together
def cost_together : ℝ := (cost_A_per_day + cost_B_per_day) * days_together

-- Theorem: Team A working alone for 20 days is the optimal solution
theorem optimal_solution (x : ℝ) :
  days_A x →
  x ≤ max_days →
  days_B x ≤ max_days →
  cost_A x ≤ cost_B x ∧
  cost_A x ≤ cost_together ∧
  x = 20 ∧
  cost_A x = 800000 :=
sorry

end NUMINAMATH_CALUDE_optimal_solution_l393_39342


namespace NUMINAMATH_CALUDE_markup_calculation_l393_39340

/-- Calculates the required markup given the purchase price, overhead percentage, and desired net profit. -/
def calculate_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the markup for the given conditions is $53.75 -/
theorem markup_calculation :
  let purchase_price : ℝ := 75
  let overhead_percent : ℝ := 0.45
  let net_profit : ℝ := 20
  calculate_markup purchase_price overhead_percent net_profit = 53.75 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l393_39340


namespace NUMINAMATH_CALUDE_negation_of_fraction_inequality_l393_39321

theorem negation_of_fraction_inequality :
  (¬ ∀ x : ℝ, 1 / (x - 2) < 0) ↔ (∃ x : ℝ, 1 / (x - 2) > 0 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_fraction_inequality_l393_39321


namespace NUMINAMATH_CALUDE_cube_sum_property_l393_39371

/-- A cube is a three-dimensional geometric shape -/
structure Cube where

/-- The number of edges in a cube -/
def Cube.num_edges (c : Cube) : ℕ := 12

/-- The number of corners in a cube -/
def Cube.num_corners (c : Cube) : ℕ := 8

/-- The number of faces in a cube -/
def Cube.num_faces (c : Cube) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of a cube is 26 -/
theorem cube_sum_property (c : Cube) : 
  c.num_edges + c.num_corners + c.num_faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_property_l393_39371


namespace NUMINAMATH_CALUDE_longest_to_shortest_l393_39389

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hyp_short : shorterLeg = hypotenuse / 2
  hyp_long : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
structure FourTriangles where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  hyp_relation1 : t1.longerLeg = t2.hypotenuse
  hyp_relation2 : t2.longerLeg = t3.hypotenuse
  hyp_relation3 : t3.longerLeg = t4.hypotenuse

theorem longest_to_shortest (triangles : FourTriangles) 
    (h : triangles.t1.hypotenuse = 16) : 
    triangles.t4.longerLeg = 9 := by
  sorry

end NUMINAMATH_CALUDE_longest_to_shortest_l393_39389


namespace NUMINAMATH_CALUDE_monotone_increasing_k_range_l393_39330

/-- A function f(x) = kx^2 - ln x is monotonically increasing in the interval (1, +∞) -/
def is_monotone_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x ≤ f y

/-- The range of k for which f(x) = kx^2 - ln x is monotonically increasing in (1, +∞) -/
theorem monotone_increasing_k_range (k : ℝ) :
  (is_monotone_increasing (fun x => k * x^2 - Real.log x) k) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_k_range_l393_39330


namespace NUMINAMATH_CALUDE_proportion_proof_1_proportion_proof_2_l393_39349

theorem proportion_proof_1 : 
  let x : ℚ := 1/12
  (x : ℚ) / (5/9 : ℚ) = (1/20 : ℚ) / (1/3 : ℚ) := by sorry

theorem proportion_proof_2 : 
  let x : ℚ := 5/4
  (x : ℚ) / (1/4 : ℚ) = (1/2 : ℚ) / (1/10 : ℚ) := by sorry

end NUMINAMATH_CALUDE_proportion_proof_1_proportion_proof_2_l393_39349


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l393_39324

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 13
  let total_entries : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_entries : ℕ := odd_numbers * odd_numbers
  (odd_entries : ℚ) / total_entries = 36 / 169 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l393_39324


namespace NUMINAMATH_CALUDE_infinitely_many_unlucky_numbers_l393_39355

/-- A natural number is unlucky if it cannot be represented as x^2 - 1 or y^2 - 1
    for any natural numbers x, y > 1. -/
def isUnlucky (n : ℕ) : Prop :=
  ∀ x y : ℕ, x > 1 ∧ y > 1 → n ≠ x^2 - 1 ∧ n ≠ y^2 - 1

/-- There are infinitely many unlucky numbers. -/
theorem infinitely_many_unlucky_numbers :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ isUnlucky n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_unlucky_numbers_l393_39355


namespace NUMINAMATH_CALUDE_valid_selections_count_l393_39335

/-- The number of male athletes -/
def num_males : ℕ := 4

/-- The number of female athletes -/
def num_females : ℕ := 5

/-- The total number of athletes to be chosen -/
def num_chosen : ℕ := 3

/-- The function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of valid selections -/
def total_selections : ℕ := 
  choose num_males 1 * choose num_females 2 + 
  choose num_males 2 * choose num_females 1

theorem valid_selections_count : total_selections = 70 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l393_39335


namespace NUMINAMATH_CALUDE_root_sum_quotient_l393_39313

/-- Given a quadratic equation m(x^2 - 2x) + 3x + 4 = 0 with roots p and q,
    and m₁ and m₂ are values of m for which p/q + q/p = 2,
    prove that m₁/m₂ + m₂/m₁ = 178/9 -/
theorem root_sum_quotient (m₁ m₂ : ℝ) (p q : ℝ) :
  (m₁ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₁ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (m₂ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₂ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (p / q + q / p = 2) →
  m₁ / m₂ + m₂ / m₁ = 178 / 9 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l393_39313


namespace NUMINAMATH_CALUDE_height_radius_ratio_is_2pi_l393_39302

/-- A cylinder with a square lateral surface -/
structure SquareLateralCylinder where
  radius : ℝ
  height : ℝ
  lateral_surface_is_square : height = 2 * Real.pi * radius

/-- The ratio of height to radius for a cylinder with a square lateral surface is 2π -/
theorem height_radius_ratio_is_2pi (c : SquareLateralCylinder) :
  c.height / c.radius = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_height_radius_ratio_is_2pi_l393_39302


namespace NUMINAMATH_CALUDE_number_of_tourists_l393_39322

theorem number_of_tourists (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end NUMINAMATH_CALUDE_number_of_tourists_l393_39322


namespace NUMINAMATH_CALUDE_new_year_markup_l393_39311

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.2 →
  february_discount = 0.06 →
  final_profit = 0.41 →
  ∃ (new_year_markup : ℝ),
    (1 - february_discount) * (1 + new_year_markup) * (1 + initial_markup) = 1 + final_profit ∧
    new_year_markup = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_new_year_markup_l393_39311


namespace NUMINAMATH_CALUDE_eraser_cost_l393_39381

theorem eraser_cost (total_students : Nat) (buyers : Nat) (erasers_per_student : Nat) (total_cost : Nat) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  erasers_per_student > 2 →
  total_cost = 3978 →
  ∃ (cost : Nat), cost > erasers_per_student ∧
                  buyers * erasers_per_student * cost = total_cost ∧
                  cost = 17 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_l393_39381


namespace NUMINAMATH_CALUDE_inequality_proof_l393_39358

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l393_39358


namespace NUMINAMATH_CALUDE_longest_side_length_l393_39303

/-- The polygonal region defined by the given system of inequalities -/
def PolygonalRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 5 ∧ 3 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The vertices of the polygonal region -/
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (1, 0), (0, 5)}

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The length of the longest side of the polygonal region is √26 -/
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
  ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices →
  squaredDistance p q ≥ squaredDistance r s ∧
  squaredDistance p q = 26 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_length_l393_39303


namespace NUMINAMATH_CALUDE_fourth_side_length_l393_39363

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  radius_positive : radius > 0
  sides_positive : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0
  inscribed : side1 ≤ 2 * radius ∧ side2 ≤ 2 * radius ∧ side3 ≤ 2 * radius ∧ side4 ≤ 2 * radius

/-- The theorem stating the length of the fourth side -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
    (h1 : q.radius = 250)
    (h2 : q.side1 = 250)
    (h3 : q.side2 = 250)
    (h4 : q.side3 = 100) :
    q.side4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l393_39363


namespace NUMINAMATH_CALUDE_jeans_price_increase_l393_39346

theorem jeans_price_increase (C : ℝ) (C_pos : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.54 * C
  (customer_price - retailer_price) / retailer_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l393_39346


namespace NUMINAMATH_CALUDE_inequality_condition_l393_39375

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 2 → x^2 - 2*x + a < 0) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l393_39375


namespace NUMINAMATH_CALUDE_fence_repair_problem_l393_39354

theorem fence_repair_problem : ∃ n : ℕ+, 
  (∃ x y : ℕ, x + y = n ∧ 2 * x + 3 * y = 87) ∧
  (∃ a b : ℕ, a + b = n ∧ 3 * a + 5 * b = 94) :=
by sorry

end NUMINAMATH_CALUDE_fence_repair_problem_l393_39354


namespace NUMINAMATH_CALUDE_water_tank_capacity_l393_39385

theorem water_tank_capacity (x : ℝ) : 
  (2/3 : ℝ) * x - (1/3 : ℝ) * x = 15 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l393_39385


namespace NUMINAMATH_CALUDE_power_of_125_two_thirds_l393_39393

theorem power_of_125_two_thirds : (125 : ℝ) ^ (2/3) = 25 := by sorry

end NUMINAMATH_CALUDE_power_of_125_two_thirds_l393_39393


namespace NUMINAMATH_CALUDE_cars_per_row_section_h_l393_39366

/-- Prove that the number of cars in each row of Section H is 9 --/
theorem cars_per_row_section_h (
  section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (cars_per_minute : ℕ)
  (search_time : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars_per_row : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_cars_per_minute : cars_per_minute = 11)
  (h_search_time : search_time = 30)
  : (cars_per_minute * search_time - section_g_rows * section_g_cars_per_row) / section_h_rows = 9 := by
  sorry

end NUMINAMATH_CALUDE_cars_per_row_section_h_l393_39366


namespace NUMINAMATH_CALUDE_common_terms_arithmetic_progression_l393_39365

/-- Definition of the first arithmetic progression -/
def a (n : ℕ) : ℤ := 4*n - 3

/-- Definition of the second arithmetic progression -/
def b (n : ℕ) : ℤ := 3*n - 1

/-- Function to generate the sequence of common terms -/
def common_terms (m : ℕ) : ℤ := 12*m + 5

/-- Theorem stating that the sequence of common terms forms an arithmetic progression with common difference 12 -/
theorem common_terms_arithmetic_progression :
  ∀ m : ℕ, ∃ n k : ℕ, 
    a n = b k ∧ 
    a n = common_terms m ∧ 
    common_terms (m + 1) - common_terms m = 12 :=
sorry

end NUMINAMATH_CALUDE_common_terms_arithmetic_progression_l393_39365


namespace NUMINAMATH_CALUDE_ana_win_probability_l393_39314

/-- Represents the probability of winning for a player in the coin flipping game -/
def winProbability (playerPosition : ℕ) : ℚ :=
  (1 / 2) ^ playerPosition / (1 - (1 / 2) ^ 4)

/-- The coin flipping game with four players -/
theorem ana_win_probability :
  winProbability 4 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ana_win_probability_l393_39314


namespace NUMINAMATH_CALUDE_train_speed_problem_l393_39309

def train_journey (x : ℝ) (v : ℝ) : Prop :=
  let first_part_distance := x
  let first_part_speed := 40
  let second_part_distance := 2 * x
  let second_part_speed := v
  let total_distance := 3 * x
  let average_speed := 24
  (first_part_distance / first_part_speed + second_part_distance / second_part_speed) * average_speed = total_distance

theorem train_speed_problem (x : ℝ) (hx : x > 0) :
  ∃ v : ℝ, train_journey x v ∧ v = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l393_39309


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l393_39379

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (cost_per_deck : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  cost_per_deck * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent 64 dollars on trick decks -/
theorem trick_decks_total_spent :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l393_39379


namespace NUMINAMATH_CALUDE_complex_multiplication_l393_39301

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l393_39301


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l393_39315

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l393_39315


namespace NUMINAMATH_CALUDE_calculation_proof_l393_39343

theorem calculation_proof : (2468 * 629) / (1234 * 37) = 34 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l393_39343


namespace NUMINAMATH_CALUDE_book_ratio_is_two_to_one_l393_39391

/-- Represents the number of books Thabo owns in each category -/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def satisfiesConditions (books : BookCounts) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 180 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 30

/-- The ratio of paperback fiction to paperback nonfiction books is 2:1 -/
def hasRatioTwoToOne (books : BookCounts) : Prop :=
  2 * books.paperbackNonfiction = books.paperbackFiction

theorem book_ratio_is_two_to_one (books : BookCounts) 
  (h : satisfiesConditions books) : hasRatioTwoToOne books := by
  sorry

#check book_ratio_is_two_to_one

end NUMINAMATH_CALUDE_book_ratio_is_two_to_one_l393_39391


namespace NUMINAMATH_CALUDE_apartment_doors_count_l393_39319

/-- Calculates the total number of doors needed for apartment buildings -/
def total_doors (num_buildings : ℕ) (floors_per_building : ℕ) (apartments_per_floor : ℕ) (doors_per_apartment : ℕ) : ℕ :=
  num_buildings * floors_per_building * apartments_per_floor * doors_per_apartment

/-- Proves that the total number of doors needed for the given apartment buildings is 1008 -/
theorem apartment_doors_count :
  total_doors 2 12 6 7 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_apartment_doors_count_l393_39319


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_3_l393_39373

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 4) - (1 - x) / (4 - x) = 0

-- Define what it means for x to be a positive root
def is_positive_root (m x : ℝ) : Prop :=
  equation m x ∧ x > 0

-- Theorem statement
theorem equation_positive_root_implies_m_equals_3 :
  ∀ m : ℝ, (∃ x : ℝ, is_positive_root m x) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_3_l393_39373


namespace NUMINAMATH_CALUDE_mixture_ratio_l393_39388

/-- Proves that combining 5 liters of Mixture A (2/3 alcohol, 1/3 water) with 14 liters of Mixture B (4/7 alcohol, 3/7 water) results in a mixture with an alcohol to water volume ratio of 34:23 -/
theorem mixture_ratio (mixture_a_volume : ℚ) (mixture_b_volume : ℚ)
  (mixture_a_alcohol_ratio : ℚ) (mixture_a_water_ratio : ℚ)
  (mixture_b_alcohol_ratio : ℚ) (mixture_b_water_ratio : ℚ)
  (h1 : mixture_a_volume = 5)
  (h2 : mixture_b_volume = 14)
  (h3 : mixture_a_alcohol_ratio = 2/3)
  (h4 : mixture_a_water_ratio = 1/3)
  (h5 : mixture_b_alcohol_ratio = 4/7)
  (h6 : mixture_b_water_ratio = 3/7) :
  (mixture_a_volume * mixture_a_alcohol_ratio + mixture_b_volume * mixture_b_alcohol_ratio) /
  (mixture_a_volume * mixture_a_water_ratio + mixture_b_volume * mixture_b_water_ratio) = 34/23 :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l393_39388


namespace NUMINAMATH_CALUDE_function_value_at_two_l393_39325

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 3 * x - 2) :
  f 2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l393_39325


namespace NUMINAMATH_CALUDE_room_dimension_l393_39345

theorem room_dimension (b h d : ℝ) (hb : b = 8) (hh : h = 9) (hd : d = 17) :
  ∃ l : ℝ, l = 12 ∧ d^2 = l^2 + b^2 + h^2 := by sorry

end NUMINAMATH_CALUDE_room_dimension_l393_39345


namespace NUMINAMATH_CALUDE_rectangle_y_value_l393_39326

/-- Given a rectangle with vertices at (2, y), (10, y), (2, -1), and (10, -1),
    where y is negative and the area is 96 square units, prove that y = -13. -/
theorem rectangle_y_value (y : ℝ) : 
  y < 0 → -- y is negative
  (10 - 2) * |(-1) - y| = 96 → -- area of the rectangle is 96
  y = -13 := by sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l393_39326


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l393_39356

theorem quadratic_root_conditions (k : ℤ) : 
  (∃ x y : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 ∧
              (k^2 + 1) * y^2 - (4 - k) * y + 1 = 0 ∧
              x > 1 ∧ y < 1) →
  k = -1 ∨ k = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l393_39356


namespace NUMINAMATH_CALUDE_fraction_meaningful_l393_39323

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x + 5)) ↔ x ≠ -5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l393_39323


namespace NUMINAMATH_CALUDE_probability_of_red_bean_l393_39386

/-- The probability of choosing a red bean from a bag -/
theorem probability_of_red_bean 
  (initial_red : ℕ) 
  (initial_black : ℕ) 
  (added_red : ℕ) 
  (added_black : ℕ) 
  (h1 : initial_red = 5)
  (h2 : initial_black = 9)
  (h3 : added_red = 3)
  (h4 : added_black = 3) : 
  (initial_red + added_red : ℚ) / (initial_red + initial_black + added_red + added_black) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_bean_l393_39386


namespace NUMINAMATH_CALUDE_sea_lion_penguin_ratio_l393_39316

theorem sea_lion_penguin_ratio :
  let sea_lions : ℕ := 48
  let penguins : ℕ := sea_lions + 84
  (sea_lions : ℚ) / penguins = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_sea_lion_penguin_ratio_l393_39316


namespace NUMINAMATH_CALUDE_berry_expense_l393_39308

-- Define the daily consumption of berries
def daily_consumption : ℚ := 1/2

-- Define the package size
def package_size : ℚ := 1

-- Define the cost per package
def cost_per_package : ℚ := 2

-- Define the number of days
def days : ℕ := 30

-- Theorem to prove
theorem berry_expense : 
  (days : ℚ) * cost_per_package * (daily_consumption / package_size) = 30 := by
  sorry

end NUMINAMATH_CALUDE_berry_expense_l393_39308


namespace NUMINAMATH_CALUDE_two_red_marbles_probability_l393_39305

/-- The probability of drawing two red marbles without replacement from a bag containing 5 red marbles and 7 white marbles is 5/33. -/
theorem two_red_marbles_probability :
  let total_marbles : ℕ := 5 + 7
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let prob_first_red : ℚ := red_marbles / total_marbles
  let prob_second_red : ℚ := (red_marbles - 1) / (total_marbles - 1)
  prob_first_red * prob_second_red = 5 / 33 :=
by sorry

end NUMINAMATH_CALUDE_two_red_marbles_probability_l393_39305


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l393_39397

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 + a_5 + a_6 + a_8 = 25, prove that a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l393_39397


namespace NUMINAMATH_CALUDE_inscribed_isosceles_tangent_circle_radius_l393_39380

/-- Given an isosceles triangle inscribed in a circle, with a second circle
    tangent to both legs of the triangle and the first circle, this theorem
    states the radius of the second circle in terms of the base and base angle
    of the isosceles triangle. -/
theorem inscribed_isosceles_tangent_circle_radius
  (a : ℝ) (α : ℝ) (h_a_pos : a > 0) (h_α_pos : α > 0) (h_α_lt_pi_2 : α < π / 2) :
  ∃ (r : ℝ),
    r > 0 ∧
    r = a / (2 * Real.sin α * (1 + Real.cos α)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_isosceles_tangent_circle_radius_l393_39380


namespace NUMINAMATH_CALUDE_range_of_m_l393_39357

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 12 * x - x^3 else -2 * x

theorem range_of_m (m : ℝ) :
  (∀ y ∈ Set.Iic m, f y ∈ Set.Ici (-16)) ∧
  (∀ z : ℝ, z ≥ -16 → ∃ x ∈ Set.Iic m, f x = z) →
  m ∈ Set.Icc (-2) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l393_39357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l393_39398

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a 0 = 3, a 1 = 7, a n = x,
    a (n+1) = y, a (n+2) = t, a (n+3) = 35, and t = 31,
    prove that x + y = 50 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 0 = 3)
  (h3 : a 1 = 7)
  (h4 : a n = x)
  (h5 : a (n+1) = y)
  (h6 : a (n+2) = t)
  (h7 : a (n+3) = 35)
  (h8 : t = 31) :
  x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l393_39398


namespace NUMINAMATH_CALUDE_fast_food_order_cost_l393_39353

/-- Calculates the total cost of a fast-food order with discount and tax --/
theorem fast_food_order_cost
  (burger_cost : ℝ)
  (sandwich_cost : ℝ)
  (smoothie_cost : ℝ)
  (num_smoothies : ℕ)
  (discount_rate : ℝ)
  (discount_threshold : ℝ)
  (tax_rate : ℝ)
  (h1 : burger_cost = 5)
  (h2 : sandwich_cost = 4)
  (h3 : smoothie_cost = 4)
  (h4 : num_smoothies = 2)
  (h5 : discount_rate = 0.15)
  (h6 : discount_threshold = 10)
  (h7 : tax_rate = 0.1) :
  let total_before_discount := burger_cost + sandwich_cost + (smoothie_cost * num_smoothies)
  let discount := if total_before_discount > discount_threshold then total_before_discount * discount_rate else 0
  let total_after_discount := total_before_discount - discount
  let tax := total_after_discount * tax_rate
  let total_cost := total_after_discount + tax
  ∃ (n : ℕ), (n : ℝ) / 100 = total_cost ∧ n = 1590 :=
by sorry


end NUMINAMATH_CALUDE_fast_food_order_cost_l393_39353


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l393_39370

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 1) * (y^2 + 5*y + 1) * (z^2 + 5*z + 1) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 1) * (b^2 + 5*b + 1) * (c^2 + 5*c + 1) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l393_39370


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_l393_39396

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel (α β : Plane) (a : Line) :
  plane_parallel α β → line_subset_plane a β → line_parallel_plane a α :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_l393_39396


namespace NUMINAMATH_CALUDE_proportion_sum_l393_39369

theorem proportion_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end NUMINAMATH_CALUDE_proportion_sum_l393_39369


namespace NUMINAMATH_CALUDE_digit_sum_problem_l393_39350

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 
  100 * P + 10 * Q + Q + 
  100 * P + 10 * P + Q + 
  100 * Q + 10 * Q + Q = 876 → 
  P + Q = 5 := by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l393_39350


namespace NUMINAMATH_CALUDE_sum_not_prime_l393_39332

theorem sum_not_prime (a b c x y z : ℕ+) (h1 : a * x * y = b * y * z) (h2 : b * y * z = c * z * x) :
  ∃ (k m : ℕ+), a + b + c + x + y + z = k * m ∧ k ≠ 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l393_39332


namespace NUMINAMATH_CALUDE_product_inequality_l393_39339

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  min (a * (1 - b)) (min (b * (1 - c)) (c * (1 - a))) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l393_39339


namespace NUMINAMATH_CALUDE_f_properties_l393_39387

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_properties :
  (f 0 = 1) ∧
  (f 1 = 1/2) ∧
  (∀ x : ℝ, 0 < f x ∧ f x ≤ 1) ∧
  (∀ y : ℝ, 0 < y ∧ y ≤ 1 → ∃ x : ℝ, f x = y) := by sorry

end NUMINAMATH_CALUDE_f_properties_l393_39387


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l393_39382

/-- An arithmetic sequence where each term is not 0 and satisfies a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 6 - (a 7)^2 + a 8 = 0

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : GeometricSequence b)
  (h_equal : b 7 = a 7) :
  b 4 * b 7 * b 10 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l393_39382


namespace NUMINAMATH_CALUDE_problem_solution_l393_39306

-- Define propositions P and Q
def P (x a : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := 2 * Real.sin x > 1 ∧ x^2 - x - 2 < 0

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, a = 2 ∧ P x a ∧ Q x → π / 6 < x ∧ x < 2) ∧
  ((∀ x : ℝ, ¬(P x a) → ¬(Q x)) ∧ (∃ x : ℝ, Q x ∧ P x a) → 2 / 3 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l393_39306


namespace NUMINAMATH_CALUDE_partition_exists_l393_39368

/-- The set of weights from 1 to 101 grams -/
def weights : Finset ℕ := Finset.range 101

/-- The sum of all weights from 1 to 101 grams -/
def total_sum : ℕ := weights.sum id

/-- The remaining weights after removing the 19-gram weight -/
def remaining_weights : Finset ℕ := weights.erase 19

/-- The sum of remaining weights -/
def remaining_sum : ℕ := remaining_weights.sum id

/-- A partition of the remaining weights into two subsets -/
structure Partition :=
  (subset1 subset2 : Finset ℕ)
  (partition_complete : subset1 ∪ subset2 = remaining_weights)
  (partition_disjoint : subset1 ∩ subset2 = ∅)
  (equal_size : subset1.card = subset2.card)
  (size_fifty : subset1.card = 50)

/-- The theorem stating that a valid partition exists -/
theorem partition_exists : ∃ (p : Partition), p.subset1.sum id = p.subset2.sum id :=
sorry

end NUMINAMATH_CALUDE_partition_exists_l393_39368


namespace NUMINAMATH_CALUDE_least_apples_count_l393_39394

theorem least_apples_count (b : ℕ) : 
  (b > 0) →
  (b % 3 = 2) → 
  (b % 4 = 3) → 
  (b % 5 = 1) → 
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 1 → n ≥ b) →
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_least_apples_count_l393_39394


namespace NUMINAMATH_CALUDE_alex_academic_year_hours_l393_39395

/-- Calculates the number of hours Alex needs to work per week during the academic year --/
def academic_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (academic_weeks : ℕ) (academic_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_weeks * summer_hours_per_week
  let hourly_rate := summer_earnings / summer_total_hours
  let academic_total_hours := academic_earnings / hourly_rate
  academic_total_hours / academic_weeks

/-- Theorem stating that Alex needs to work 20 hours per week during the academic year --/
theorem alex_academic_year_hours : 
  academic_year_hours_per_week 8 40 4000 32 8000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_academic_year_hours_l393_39395


namespace NUMINAMATH_CALUDE_zero_properties_l393_39390

theorem zero_properties : 
  (0 : ℕ) = 0 ∧ (0 : ℤ) = 0 ∧ (0 : ℝ) = 0 ∧ ¬(0 > 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_properties_l393_39390


namespace NUMINAMATH_CALUDE_right_triangle_sin_x_l393_39336

theorem right_triangle_sin_x (X Y Z : Real) (sinX cosX tanX : Real) :
  -- Right triangle XYZ with ∠Y = 90°
  (X^2 + Y^2 = Z^2) →
  -- 4sinX = 5cosX
  (4 * sinX = 5 * cosX) →
  -- tanX = XY/YZ
  (tanX = X / Y) →
  -- sinX = 5√41 / 41
  sinX = 5 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_x_l393_39336


namespace NUMINAMATH_CALUDE_b_91_mod_49_l393_39364

/-- Definition of the sequence bₙ -/
def b (n : ℕ) : ℕ := 12^n + 14^n

/-- Theorem stating that b₉₁ mod 49 = 38 -/
theorem b_91_mod_49 : b 91 % 49 = 38 := by
  sorry

end NUMINAMATH_CALUDE_b_91_mod_49_l393_39364


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l393_39338

noncomputable def f (a x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem f_monotonicity_and_extrema :
  ∀ a : ℝ, a > 0 →
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1/a ∧ 1/a < x₂ ∧ x₂ < 1 → f a x₁ < f a (1/a) ∧ f a (1/a) < f a x₂) ∧
    (f a (1/a) = -(a + 1/a) * Real.log a + a - 1/a) ∧
    (f a a = (a + 1/a) * Real.log a + 1/a - a) ∧
    (∀ x : ℝ, x > 0 → f a x ≥ f a (1/a) ∧ f a x ≤ f a a) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l393_39338


namespace NUMINAMATH_CALUDE_speed_A_calculation_l393_39372

-- Define the speeds of A and B
def speed_A : ℝ := 5.913043478260869
def speed_B : ℝ := 7.555555555555555

-- Define the time when B overtakes A (in hours)
def overtake_time : ℝ := 1.8

-- Define the head start time of A (in hours)
def head_start : ℝ := 0.5

-- Theorem statement
theorem speed_A_calculation :
  speed_A * (overtake_time + head_start) = speed_B * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_speed_A_calculation_l393_39372


namespace NUMINAMATH_CALUDE_last_year_honey_harvest_l393_39352

/-- 
Given Diane's honey harvest information:
- This year's harvest: 8564 pounds
- Increase from last year: 6085 pounds

Prove that last year's harvest was 2479 pounds.
-/
theorem last_year_honey_harvest 
  (this_year : ℕ) 
  (increase : ℕ) 
  (h1 : this_year = 8564)
  (h2 : increase = 6085) :
  this_year - increase = 2479 := by
sorry

end NUMINAMATH_CALUDE_last_year_honey_harvest_l393_39352


namespace NUMINAMATH_CALUDE_actual_height_of_boy_l393_39392

/-- Calculates the actual height of a boy in a class given the following conditions:
  * There are 35 boys in the class
  * The initially calculated average height was 182 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 180 cm
-/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 182 →
  wrong_height = 166 →
  actual_avg = 180 →
  ∃ (x : ℝ), x = 236 ∧ n * actual_avg = (n * initial_avg - wrong_height + x) :=
by sorry

end NUMINAMATH_CALUDE_actual_height_of_boy_l393_39392


namespace NUMINAMATH_CALUDE_solution_difference_l393_39362

theorem solution_difference (p q : ℝ) : 
  p ≠ q → 
  ((6 * p - 18) / (p^2 + 4*p - 21) = p + 3) →
  ((6 * q - 18) / (q^2 + 4*q - 21) = q + 3) →
  p > q →
  p - q = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l393_39362


namespace NUMINAMATH_CALUDE_danny_apples_danny_bought_73_apples_l393_39378

def pinky_apples : ℕ := 36
def total_apples : ℕ := 109

theorem danny_apples : ℕ → Prop :=
  fun x => x = total_apples - pinky_apples

theorem danny_bought_73_apples : danny_apples 73 := by
  sorry

end NUMINAMATH_CALUDE_danny_apples_danny_bought_73_apples_l393_39378


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l393_39344

theorem simplify_sqrt_fraction : 
  (Real.sqrt 462 / Real.sqrt 330) + (Real.sqrt 245 / Real.sqrt 175) = 12 * Real.sqrt 35 / 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l393_39344


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l393_39320

/-- The sum of the series Σ(n=0 to ∞) of 3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1)) is equal to 1 -/
theorem series_sum_equals_one :
  ∑' n : ℕ, (3 : ℝ) ^ n / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l393_39320


namespace NUMINAMATH_CALUDE_power_function_property_l393_39312

theorem power_function_property (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ a) → f 2 = Real.sqrt 2 → f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l393_39312


namespace NUMINAMATH_CALUDE_songs_added_l393_39307

/-- Calculates the number of new songs added to an mp3 player. -/
theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 9 → final = 10 → final - (initial - deleted) = 8 := by
  sorry

end NUMINAMATH_CALUDE_songs_added_l393_39307


namespace NUMINAMATH_CALUDE_is_min_point_l393_39333

/-- The function representing the translated graph -/
def f (x : ℝ) : ℝ := |x - 4| - 3

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -3)

/-- Theorem stating that min_point is the minimum of the function f -/
theorem is_min_point :
  ∀ x : ℝ, f x ≥ f min_point.fst ∧ f min_point.fst = min_point.snd := by
  sorry

end NUMINAMATH_CALUDE_is_min_point_l393_39333


namespace NUMINAMATH_CALUDE_smallest_b_for_non_range_l393_39351

theorem smallest_b_for_non_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 10 ≠ -6) ↔ b ≤ -7 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_non_range_l393_39351


namespace NUMINAMATH_CALUDE_parallel_postulate_l393_39328

-- Define a Point type
def Point : Type := ℝ × ℝ

-- Define a Line type
def Line : Type := Point → Point → Prop

-- Define a parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def OnLine (p : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem parallel_postulate (l : Line) (p : Point) : 
  ¬(OnLine p l) → ∃! (m : Line), Parallel m l ∧ OnLine p m := by sorry

end NUMINAMATH_CALUDE_parallel_postulate_l393_39328


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l393_39304

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence where the 5th term is 80 and the 7th term is 320, the 9th term is 1280. -/
theorem geometric_sequence_ninth_term (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_5th : a 5 = 80) 
    (h_7th : a 7 = 320) : 
  a 9 = 1280 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l393_39304


namespace NUMINAMATH_CALUDE_parallel_transitivity_false_l393_39327

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the relations
variable (belongs_to : Point → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_false :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel_line_plane l α →
    parallel_line_plane m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_false_l393_39327


namespace NUMINAMATH_CALUDE_estimated_weight_not_exact_weight_estimated_weight_is_approximation_l393_39383

/-- Represents the linear regression model for height and weight --/
structure HeightWeightModel where
  slope : ℝ
  intercept : ℝ

/-- The estimated weight based on the linear regression model --/
def estimated_weight (model : HeightWeightModel) (height : ℝ) : ℝ :=
  model.slope * height + model.intercept

/-- The given linear regression model for the problem --/
def given_model : HeightWeightModel :=
  { slope := 0.85, intercept := -85.71 }

/-- Theorem stating that the estimated weight for a 160cm tall girl is not necessarily her exact weight --/
theorem estimated_weight_not_exact_weight :
  ∃ (actual_weight : ℝ), 
    estimated_weight given_model 160 ≠ actual_weight ∧ 
    actual_weight > 0 := by
  sorry

/-- Theorem stating that the estimated weight is just an approximation --/
theorem estimated_weight_is_approximation (height : ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ 
    ∀ (actual_weight : ℝ), 
      actual_weight > 0 →
      |estimated_weight given_model height - actual_weight| < ε := by
  sorry

end NUMINAMATH_CALUDE_estimated_weight_not_exact_weight_estimated_weight_is_approximation_l393_39383


namespace NUMINAMATH_CALUDE_inequality_conditions_l393_39341

theorem inequality_conditions (a b : ℝ) :
  ((b > 0 ∧ 0 > a) → (1 / a < 1 / b)) ∧
  ((0 > a ∧ a > b) → (1 / a < 1 / b)) ∧
  ((a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b)) ∧
  ((a > b ∧ b > 0) → (1 / a < 1 / b)) := by
sorry

end NUMINAMATH_CALUDE_inequality_conditions_l393_39341


namespace NUMINAMATH_CALUDE_project_hours_proof_l393_39317

theorem project_hours_proof (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 105) :
  kate + mark + pat = 189 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_proof_l393_39317


namespace NUMINAMATH_CALUDE_ones_divisible_by_power_of_three_l393_39359

/-- Given a natural number n ≥ 1, the function returns the number formed by 3^n consecutive ones. -/
def number_of_ones (n : ℕ) : ℕ :=
  (10^(3^n) - 1) / 9

/-- Theorem stating that for any natural number n ≥ 1, the number formed by 3^n consecutive ones
    is divisible by 3^n. -/
theorem ones_divisible_by_power_of_three (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, number_of_ones n = 3^n * k :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_power_of_three_l393_39359


namespace NUMINAMATH_CALUDE_f_difference_l393_39348

def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 4*x

theorem f_difference : f 3 - f (-3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l393_39348


namespace NUMINAMATH_CALUDE_point_motion_time_l393_39367

/-- 
Given two points A and B initially separated by distance a, moving along different sides of a right angle 
towards its vertex with constant speed v, where B reaches the vertex t units of time before A, 
this theorem states the time x that A takes to reach the vertex.
-/
theorem point_motion_time (a v t : ℝ) (h : a > v * t) : 
  ∃ x : ℝ, x = (t * v + Real.sqrt (2 * a^2 - v^2 * t^2)) / (2 * v) ∧ 
  x * v = Real.sqrt ((x * v)^2 + ((x - t) * v)^2) :=
sorry

end NUMINAMATH_CALUDE_point_motion_time_l393_39367


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l393_39376

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l393_39376


namespace NUMINAMATH_CALUDE_k_value_l393_39310

theorem k_value (k : ℝ) : (5 + k) * (5 - k) = 5^2 - 2^3 → k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l393_39310


namespace NUMINAMATH_CALUDE_number_equals_scientific_rep_l393_39399

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 1300000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.3
  , exponent := 6
  , h_coeff := by sorry }

theorem number_equals_scientific_rep :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent :=
by sorry

end NUMINAMATH_CALUDE_number_equals_scientific_rep_l393_39399


namespace NUMINAMATH_CALUDE_circle_area_difference_l393_39347

theorem circle_area_difference (r₁ r₂ : ℝ) (h₁ : r₁ = 14) (h₂ : r₂ = 10) :
  π * r₁^2 - π * r₂^2 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l393_39347


namespace NUMINAMATH_CALUDE_sqrt_four_twos_to_fourth_l393_39360

theorem sqrt_four_twos_to_fourth : Real.sqrt (2^4 + 2^4 + 2^4 + 2^4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_twos_to_fourth_l393_39360


namespace NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l393_39300

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * n

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 0) + (hotdog_sequence 1) + (hotdog_sequence 2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l393_39300


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l393_39331

/-- Represents the number of students -/
def total_students : ℕ := 7

/-- Represents the number of students on each side of the tallest student -/
def students_per_side : ℕ := 3

/-- The number of possible arrangements of students for the photo -/
def num_arrangements : ℕ := Nat.choose (total_students - 1) students_per_side

/-- Theorem stating that the number of arrangements is correct -/
theorem photo_arrangements_count :
  num_arrangements = 20 :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l393_39331


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l393_39374

theorem irrational_among_given_numbers :
  let a : ℝ := -1/7
  let b : ℝ := Real.sqrt 11
  let c : ℝ := 0.3
  let d : ℝ := Real.sqrt 25
  Irrational b ∧ ¬(Irrational a ∨ Irrational c ∨ Irrational d) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l393_39374


namespace NUMINAMATH_CALUDE_tower_has_four_levels_l393_39334

/-- Calculates the number of levels in a tower given the number of steps per level,
    blocks per step, and total blocks climbed. -/
def tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / steps_per_level

/-- Theorem stating that a tower with 8 steps per level, 3 blocks per step,
    and 96 total blocks climbed has 4 levels. -/
theorem tower_has_four_levels :
  tower_levels 8 3 96 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tower_has_four_levels_l393_39334
