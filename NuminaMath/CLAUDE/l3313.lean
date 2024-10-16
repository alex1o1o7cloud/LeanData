import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3313_331368

theorem min_value_expression (x : ℝ) (h : x > 10) :
  (x^2 + 36) / (x - 10) ≥ 4 * Real.sqrt 34 + 20 :=
by sorry

theorem min_value_achieved (x : ℝ) (h : x > 10) :
  ∃ y > 10, (y^2 + 36) / (y - 10) = 4 * Real.sqrt 34 + 20 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3313_331368


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3313_331329

theorem contrapositive_equivalence (a b : ℝ) :
  ((a^2 + b^2 = 0) → (a = 0 ∧ b = 0)) ↔ ((a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3313_331329


namespace NUMINAMATH_CALUDE_crayon_selection_proof_l3313_331341

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_proof : choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_proof_l3313_331341


namespace NUMINAMATH_CALUDE_height_of_cylinder_A_l3313_331349

/-- Theorem: Height of Cylinder A given volume ratio with Cylinder B -/
theorem height_of_cylinder_A (r_A r_B h_B : ℝ) 
  (h_circum_A : 2 * Real.pi * r_A = 8)
  (h_circum_B : 2 * Real.pi * r_B = 10)
  (h_height_B : h_B = 8)
  (h_volume_ratio : Real.pi * r_A^2 * (7 : ℝ) = 0.5600000000000001 * Real.pi * r_B^2 * h_B) :
  ∃ h_A : ℝ, h_A = 7 ∧ Real.pi * r_A^2 * h_A = 0.5600000000000001 * Real.pi * r_B^2 * h_B := by
  sorry

end NUMINAMATH_CALUDE_height_of_cylinder_A_l3313_331349


namespace NUMINAMATH_CALUDE_consistent_production_rate_l3313_331331

/-- Represents the rate of paint drum production -/
structure PaintProduction where
  days : ℕ
  drums : ℕ

/-- Calculates the daily production rate -/
def dailyRate (p : PaintProduction) : ℚ :=
  p.drums / p.days

theorem consistent_production_rate : 
  let scenario1 : PaintProduction := ⟨3, 18⟩
  let scenario2 : PaintProduction := ⟨60, 360⟩
  dailyRate scenario1 = dailyRate scenario2 ∧ dailyRate scenario1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_consistent_production_rate_l3313_331331


namespace NUMINAMATH_CALUDE_limes_picked_equals_total_l3313_331352

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 25

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- Theorem: The sum of limes picked by Alyssa and Mike equals the total number of limes picked -/
theorem limes_picked_equals_total : alyssas_limes + mikes_limes = total_limes := by
  sorry

end NUMINAMATH_CALUDE_limes_picked_equals_total_l3313_331352


namespace NUMINAMATH_CALUDE_semicircles_in_rectangle_l3313_331315

theorem semicircles_in_rectangle (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₁ > r₂) :
  let height := 2 * Real.sqrt (r₁ * r₂)
  let rectangle_area := height * (r₁ + r₂)
  let semicircles_area := π / 2 * (r₁^2 + r₂^2)
  semicircles_area / rectangle_area = (π / 2 * (r₁^2 + r₂^2)) / (2 * Real.sqrt (r₁ * r₂) * (r₁ + r₂)) :=
by sorry

end NUMINAMATH_CALUDE_semicircles_in_rectangle_l3313_331315


namespace NUMINAMATH_CALUDE_exists_k_in_interval_l3313_331396

theorem exists_k_in_interval (x : ℝ) (hx_pos : 0 < x) (hx_le_one : x ≤ 1) :
  ∃ k : ℕ+, (4/3 : ℝ) < (k : ℝ) * x ∧ (k : ℝ) * x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_in_interval_l3313_331396


namespace NUMINAMATH_CALUDE_total_chestnuts_weight_l3313_331376

/-- The weight of chestnuts Eun-soo picked in kilograms -/
def eun_soo_kg : ℝ := 2

/-- The weight of chestnuts Eun-soo picked in grams (in addition to the kilograms) -/
def eun_soo_g : ℝ := 600

/-- The weight of chestnuts Min-gi picked in grams -/
def min_gi_g : ℝ := 3700

/-- The conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem total_chestnuts_weight : 
  eun_soo_kg * kg_to_g + eun_soo_g + min_gi_g = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_chestnuts_weight_l3313_331376


namespace NUMINAMATH_CALUDE_product_of_numbers_l3313_331300

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x / y = 5 / 4) :
  x * y = 320 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3313_331300


namespace NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l3313_331340

theorem difference_of_sum_and_difference_of_squares 
  (x y : ℝ) 
  (h1 : x + y = 6) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l3313_331340


namespace NUMINAMATH_CALUDE_special_integers_proof_l3313_331360

theorem special_integers_proof (k : ℕ) (h : k ≥ 2) :
  (∀ m n : ℕ, 1 ≤ m ∧ m < n ∧ n ≤ k → ¬(k ∣ (n^(n-1) - m^(m-1)))) ↔ (k = 2 ∨ k = 3) :=
sorry

end NUMINAMATH_CALUDE_special_integers_proof_l3313_331360


namespace NUMINAMATH_CALUDE_chess_and_go_pricing_and_max_purchase_l3313_331369

/-- The unit price of a Chinese chess set -/
def chinese_chess_price : ℝ := 25

/-- The unit price of a Go set -/
def go_price : ℝ := 30

/-- The total number of sets to be purchased -/
def total_sets : ℕ := 120

/-- The maximum total cost -/
def max_total_cost : ℝ := 3500

theorem chess_and_go_pricing_and_max_purchase :
  (2 * chinese_chess_price + go_price = 80) ∧
  (4 * chinese_chess_price + 3 * go_price = 190) ∧
  (∀ m : ℕ, m ≤ total_sets → 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost →
    m ≤ 100) ∧
  (∃ m : ℕ, m = 100 ∧ 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost) :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_pricing_and_max_purchase_l3313_331369


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3313_331326

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 4)
  (h_sum : a 1 + a 2 + a 3 = 14) :
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ m n p : ℕ, m < n → n < p → a m + a p ≠ 2 * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3313_331326


namespace NUMINAMATH_CALUDE_initial_crayons_l3313_331386

theorem initial_crayons (initial : ℕ) : initial + 3 = 12 → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l3313_331386


namespace NUMINAMATH_CALUDE_optimal_optimism_coefficient_l3313_331338

theorem optimal_optimism_coefficient 
  (a b c x : ℝ) 
  (h1 : b > a) 
  (h2 : 0 < x ∧ x < 1) 
  (h3 : c = a + x * (b - a)) 
  (h4 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_optimal_optimism_coefficient_l3313_331338


namespace NUMINAMATH_CALUDE_light_beam_reflection_l3313_331314

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two points, returns the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p1.y - p2.y
    b := p2.x - p1.x
    c := p1.x * p2.y - p2.x * p1.y }

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The x-axis -/
def x_axis : Line :=
  { a := 0, b := 1, c := 0 }

/-- Reflects a point across the x-axis -/
def reflect_point_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem light_beam_reflection (M N : Point) 
    (h1 : M = { x := 4, y := 5 })
    (h2 : N = { x := 2, y := 0 })
    (h3 : point_on_line N x_axis) :
  ∃ (l : Line), 
    l = { a := 5, b := -2, c := -10 } ∧ 
    point_on_line M l ∧ 
    point_on_line N l ∧
    point_on_line (reflect_point_x_axis M) l :=
  sorry

end NUMINAMATH_CALUDE_light_beam_reflection_l3313_331314


namespace NUMINAMATH_CALUDE_grape_juice_amount_l3313_331372

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ
  orange_watermelon_oz : ℝ

/-- The fruit drink satisfies the given conditions -/
def valid_fruit_drink (drink : FruitDrink) : Prop :=
  drink.orange_percent = 0.15 ∧
  drink.watermelon_percent = 0.60 ∧
  drink.orange_watermelon_oz = 120 ∧
  drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 1

/-- Calculate the amount of grape juice in ounces -/
def grape_juice_oz (drink : FruitDrink) : ℝ :=
  drink.grape_percent * drink.total

/-- Theorem stating that the amount of grape juice is 40 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h : valid_fruit_drink drink) : grape_juice_oz drink = 40 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_amount_l3313_331372


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3313_331313

theorem quadratic_root_difference (p : ℝ) : 
  p > 0 → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + 1 = 0 ∧ 
    x₂^2 + p*x₂ + 1 = 0 ∧ 
    |x₁ - x₂| = 1) → 
  p = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3313_331313


namespace NUMINAMATH_CALUDE_ned_gave_away_13_games_l3313_331334

/-- The number of games Ned originally had -/
def original_games : ℕ := 19

/-- The number of games Ned currently has -/
def current_games : ℕ := 6

/-- The number of games Ned gave away -/
def games_given_away : ℕ := original_games - current_games

theorem ned_gave_away_13_games : games_given_away = 13 := by
  sorry

end NUMINAMATH_CALUDE_ned_gave_away_13_games_l3313_331334


namespace NUMINAMATH_CALUDE_transfer_increases_averages_l3313_331336

/-- Represents a group of students with their total score and count -/
structure StudentGroup where
  totalScore : ℚ
  count : ℕ

/-- Calculates the average score of a group -/
def averageScore (group : StudentGroup) : ℚ :=
  group.totalScore / group.count

/-- Theorem: Transferring Lopatin and Filin increases average scores in both groups -/
theorem transfer_increases_averages
  (groupA groupB : StudentGroup)
  (lopatinScore filinScore : ℚ)
  (h1 : groupA.count = 10)
  (h2 : groupB.count = 10)
  (h3 : averageScore groupA = 47.2)
  (h4 : averageScore groupB = 41.8)
  (h5 : 41.8 < lopatinScore) (h6 : lopatinScore < 47.2)
  (h7 : 41.8 < filinScore) (h8 : filinScore < 47.2)
  (h9 : lopatinScore = 47)
  (h10 : filinScore = 44) :
  let newGroupA : StudentGroup := ⟨groupA.totalScore - lopatinScore - filinScore, 8⟩
  let newGroupB : StudentGroup := ⟨groupB.totalScore + lopatinScore + filinScore, 12⟩
  averageScore newGroupA > 47.5 ∧ averageScore newGroupB > 42.2 := by
  sorry

end NUMINAMATH_CALUDE_transfer_increases_averages_l3313_331336


namespace NUMINAMATH_CALUDE_zachary_pushup_count_l3313_331344

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference between Zachary's and David's push-ups -/
def pushup_difference : ℕ := 7

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := david_pushups + pushup_difference

theorem zachary_pushup_count : zachary_pushups = 51 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushup_count_l3313_331344


namespace NUMINAMATH_CALUDE_sum_edge_lengths_truncated_octahedron_l3313_331317

/-- A polyhedron with 24 vertices and all edges of length 5 cm -/
structure Polyhedron where
  vertices : ℕ
  edge_length : ℝ
  h_vertices : vertices = 24
  h_edge_length : edge_length = 5

/-- A truncated octahedron is a polyhedron with 36 edges -/
def is_truncated_octahedron (p : Polyhedron) : Prop :=
  ∃ (edges : ℕ), edges = 36

/-- The sum of edge lengths for a polyhedron -/
def sum_edge_lengths (p : Polyhedron) (edges : ℕ) : ℝ :=
  p.edge_length * edges

/-- Theorem: If the polyhedron is a truncated octahedron, 
    then the sum of edge lengths is 180 cm -/
theorem sum_edge_lengths_truncated_octahedron (p : Polyhedron) 
  (h : is_truncated_octahedron p) : 
  ∃ (edges : ℕ), sum_edge_lengths p edges = 180 := by
  sorry


end NUMINAMATH_CALUDE_sum_edge_lengths_truncated_octahedron_l3313_331317


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_in_range_l3313_331327

theorem inequality_holds_iff_x_in_range :
  ∀ x : ℝ, (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ 
  (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_in_range_l3313_331327


namespace NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l3313_331375

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l3313_331375


namespace NUMINAMATH_CALUDE_function_zero_at_seven_fifths_l3313_331383

theorem function_zero_at_seven_fifths :
  let f : ℝ → ℝ := λ x ↦ 5 * x - 7
  f (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_at_seven_fifths_l3313_331383


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l3313_331316

/-- The number of saltwater aquariums Tyler has -/
def num_saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := num_saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l3313_331316


namespace NUMINAMATH_CALUDE_bar_and_line_charts_represent_amount_l3313_331312

-- Define bar charts and line charts as types that can represent data
def BarChart : Type := Unit
def LineChart : Type := Unit

-- Define a property for charts that can represent amount
def CanRepresentAmount (chart : Type) : Prop := True

-- State the theorem
theorem bar_and_line_charts_represent_amount :
  CanRepresentAmount BarChart ∧ CanRepresentAmount LineChart := by
  sorry

end NUMINAMATH_CALUDE_bar_and_line_charts_represent_amount_l3313_331312


namespace NUMINAMATH_CALUDE_crackers_per_friend_l3313_331361

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l3313_331361


namespace NUMINAMATH_CALUDE_peters_books_difference_l3313_331392

/-- Given that Peter has 20 books, he has read 40% of them, and his brother has read 10% of them,
    prove that Peter has read 6 more books than his brother. -/
theorem peters_books_difference (total_books : ℕ) (peter_percentage : ℚ) (brother_percentage : ℚ) :
  total_books = 20 →
  peter_percentage = 2/5 →
  brother_percentage = 1/10 →
  (total_books : ℚ) * peter_percentage - (total_books : ℚ) * brother_percentage = 6 := by
  sorry

end NUMINAMATH_CALUDE_peters_books_difference_l3313_331392


namespace NUMINAMATH_CALUDE_min_value_of_f_l3313_331325

/-- Given positive real numbers a, b, c, x, y, z satisfying the given conditions,
    the minimum value of the function f is 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  (∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' →
    c * y' + b * z' = a →
    a * z' + c * x' = b →
    b * x' + a * y' = c →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2) ∧
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3313_331325


namespace NUMINAMATH_CALUDE_D_most_suitable_l3313_331304

structure Athlete where
  name : String
  averageScore : Float
  variance : Float

def isMoreSuitable (a b : Athlete) : Prop :=
  (a.averageScore > b.averageScore) ∨
  (a.averageScore = b.averageScore ∧ a.variance < b.variance)

def A : Athlete := ⟨"A", 9.5, 6.6⟩
def B : Athlete := ⟨"B", 9.6, 6.7⟩
def C : Athlete := ⟨"C", 9.5, 6.7⟩
def D : Athlete := ⟨"D", 9.6, 6.6⟩

theorem D_most_suitable :
  isMoreSuitable D A ∧ isMoreSuitable D B ∧ isMoreSuitable D C := by
  sorry

end NUMINAMATH_CALUDE_D_most_suitable_l3313_331304


namespace NUMINAMATH_CALUDE_sum_fraction_inequality_l3313_331381

theorem sum_fraction_inequality (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_inequality_l3313_331381


namespace NUMINAMATH_CALUDE_morning_milk_calculation_l3313_331364

/-- The number of gallons of milk Aunt May got this morning -/
def morning_milk : ℕ := 365

/-- The number of gallons of milk Aunt May got in the evening -/
def evening_milk : ℕ := 380

/-- The number of gallons of milk Aunt May sold -/
def sold_milk : ℕ := 612

/-- The number of gallons of milk left over from yesterday -/
def leftover_milk : ℕ := 15

/-- The number of gallons of milk remaining -/
def remaining_milk : ℕ := 148

/-- Theorem stating that the morning milk calculation is correct -/
theorem morning_milk_calculation :
  morning_milk + evening_milk + leftover_milk - sold_milk = remaining_milk :=
by sorry

end NUMINAMATH_CALUDE_morning_milk_calculation_l3313_331364


namespace NUMINAMATH_CALUDE_two_from_three_permutations_l3313_331393

/-- The number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- There are 3 people to choose from -/
def total_people : ℕ := 3

/-- We are choosing 2 people to line up -/
def people_to_choose : ℕ := 2

/-- The number of ways to line up 2 people from a group of 3 is 6 -/
theorem two_from_three_permutations :
  permutations total_people people_to_choose = 6 := by sorry

end NUMINAMATH_CALUDE_two_from_three_permutations_l3313_331393


namespace NUMINAMATH_CALUDE_composition_of_transformations_l3313_331373

-- Define the transformations
def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

-- State the theorem
theorem composition_of_transformations :
  f (g (-1, 2)) = (1, -3) := by sorry

end NUMINAMATH_CALUDE_composition_of_transformations_l3313_331373


namespace NUMINAMATH_CALUDE_complex_point_location_l3313_331345

theorem complex_point_location (x y : ℝ) 
  (h : (x + y) + (y - 1) * Complex.I = (2 * x + 3 * y) + (2 * y + 1) * Complex.I) : 
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l3313_331345


namespace NUMINAMATH_CALUDE_hiker_first_pack_weight_hiker_first_pack_weight_proof_l3313_331397

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : resupply_ratio = 0.25)
  : Real :=
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let resupply_weight := resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - resupply_weight
  37.5

theorem hiker_first_pack_weight_proof : hiker_first_pack_weight 0.5 2.5 8 5 0.25 rfl rfl rfl rfl rfl = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_first_pack_weight_hiker_first_pack_weight_proof_l3313_331397


namespace NUMINAMATH_CALUDE_printing_presses_l3313_331363

theorem printing_presses (papers : ℕ) (initial_time hours : ℝ) (known_presses : ℕ) :
  papers > 0 →
  initial_time > 0 →
  hours > 0 →
  known_presses > 0 →
  (papers : ℝ) / (initial_time * (papers / (hours * known_presses : ℝ))) = 40 :=
by
  sorry

#check printing_presses 500000 9 12 30

end NUMINAMATH_CALUDE_printing_presses_l3313_331363


namespace NUMINAMATH_CALUDE_part_one_part_two_l3313_331370

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) :
  (p x 1) ∧ (q x) → 2 ≤ x ∧ x < 3 := by sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ¬(q x → ¬(p x a))) →
  ∃ a : ℝ, 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3313_331370


namespace NUMINAMATH_CALUDE_board_game_theorem_l3313_331348

/-- Represents the operation of replacing two numbers with their combination -/
def combine (a b : ℚ) : ℚ := a * b + a + b

/-- The set of initial numbers on the board -/
def initial_numbers (n : ℕ) : List ℚ := List.range n |>.map (λ i => 1 / (i + 1))

/-- The invariant product of all numbers on the board increased by 1 -/
def product_plus_one (numbers : List ℚ) : ℚ := numbers.foldl (λ acc x => acc * (x + 1)) 1

/-- The final number after n-1 operations -/
def final_number (n : ℕ) : ℚ := n

theorem board_game_theorem (n : ℕ) (h : n > 0) :
  ∃ (operations : List (ℕ × ℕ)),
    operations.length = n - 1 ∧
    final_number n = product_plus_one (initial_numbers n) - 1 := by
  sorry

#check board_game_theorem

end NUMINAMATH_CALUDE_board_game_theorem_l3313_331348


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3313_331324

theorem gcd_of_powers_minus_one : Nat.gcd (2^300 - 1) (2^315 - 1) = 2^15 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3313_331324


namespace NUMINAMATH_CALUDE_no_extremum_range_l3313_331354

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has no extremum -/
theorem no_extremum_range (a : ℝ) : 
  (∀ x : ℝ, f_derivative a x ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_no_extremum_range_l3313_331354


namespace NUMINAMATH_CALUDE_die_roll_probability_l3313_331347

theorem die_roll_probability : 
  let p : ℝ := 1 / 2  -- probability of rolling an even number on a single die
  let n : ℕ := 8      -- number of rolls
  1 - (1 - p) ^ n = 255 / 256 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3313_331347


namespace NUMINAMATH_CALUDE_congruence_solution_l3313_331311

theorem congruence_solution (m : ℤ) : 
  (13 * m) % 47 = 8 % 47 ↔ m % 47 = 20 % 47 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3313_331311


namespace NUMINAMATH_CALUDE_unique_c_for_complex_magnitude_l3313_331355

theorem unique_c_for_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - 2 * c * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_complex_magnitude_l3313_331355


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3313_331362

theorem invalid_votes_percentage (total_votes : ℕ) (winning_percentage : ℚ) (losing_votes : ℕ) :
  total_votes = 7500 →
  winning_percentage = 55 / 100 →
  losing_votes = 2700 →
  (total_votes - (losing_votes / (1 - winning_percentage))) / total_votes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3313_331362


namespace NUMINAMATH_CALUDE_expansion_terms_count_l3313_331322

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 is equal to (10 choose 3) -/
theorem expansion_terms_count : dissimilar_terms = 120 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l3313_331322


namespace NUMINAMATH_CALUDE_solid_color_marbles_percentage_l3313_331303

theorem solid_color_marbles_percentage
  (yellow_percentage : ℝ)
  (other_solid_percentage : ℝ)
  (h1 : yellow_percentage = 5)
  (h2 : other_solid_percentage = 85) :
  yellow_percentage + other_solid_percentage = 90 :=
by sorry

end NUMINAMATH_CALUDE_solid_color_marbles_percentage_l3313_331303


namespace NUMINAMATH_CALUDE_tire_circumference_l3313_331391

/-- The circumference of a tire given its rotational speed and the car's velocity -/
theorem tire_circumference (rpm : ℝ) (velocity_kmh : ℝ) : rpm = 400 → velocity_kmh = 72 → 
  ∃ (circumference : ℝ), circumference = 3 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l3313_331391


namespace NUMINAMATH_CALUDE_sum_abcd_is_negative_six_l3313_331366

theorem sum_abcd_is_negative_six 
  (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_is_negative_six_l3313_331366


namespace NUMINAMATH_CALUDE_f_one_equals_neg_log_four_l3313_331384

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then -x * Real.log (3 - x) else -(-x * Real.log (3 + x))

-- State the theorem
theorem f_one_equals_neg_log_four :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≤ 0, f x = -x * Real.log (3 - x)) →  -- definition for x ≤ 0
  f 1 = -Real.log 4 := by
sorry

end NUMINAMATH_CALUDE_f_one_equals_neg_log_four_l3313_331384


namespace NUMINAMATH_CALUDE_find_number_l3313_331342

theorem find_number : ∃ x : ℝ, 11 * x + 1 = 45 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3313_331342


namespace NUMINAMATH_CALUDE_solution_triplets_l3313_331357

theorem solution_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧
  (2 * y^3 + 1 = 3 * x * y) ∧
  (2 * z^3 + 1 = 3 * y * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triplets_l3313_331357


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3313_331337

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 144 is 8√2 -/
theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3313_331337


namespace NUMINAMATH_CALUDE_boat_speed_difference_l3313_331328

/-- Proves that the boat's speed is 1 km/h greater than the stream current speed --/
theorem boat_speed_difference (V : ℝ) : 
  let S := 1 -- distance in km
  let V₁ := 2*V + 1 -- river current speed in km/h
  let T := 1 -- total time in hours
  ∃ (U : ℝ), -- boat's speed
    U > V ∧ -- boat is faster than stream current
    S / (U - V) - S / (U + V) + S / V₁ = T ∧ -- time equation
    U - V = 1 -- difference in speeds
  := by sorry

end NUMINAMATH_CALUDE_boat_speed_difference_l3313_331328


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3313_331388

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧
    (x < 1 ∧ x > m - 1) ∧
    (y < 1 ∧ y > m - 1) ∧
    ∀ (z : ℤ), (z < 1 ∧ z > m - 1) → (z = x ∨ z = y)

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3313_331388


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l3313_331359

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The sun's radius in kilometers -/
def sun_radius_km : ℝ := 696000

/-- The sun's radius in meters -/
def sun_radius_m : ℝ := sun_radius_km * km_to_m

theorem sun_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ sun_radius_m = a * (10 : ℝ) ^ n ∧ a = 6.96 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l3313_331359


namespace NUMINAMATH_CALUDE_probability_green_or_blue_ten_sided_die_l3313_331302

/-- Represents a 10-sided die with colored faces -/
structure ColoredDie :=
  (total_sides : Nat)
  (red_faces : Nat)
  (yellow_faces : Nat)
  (green_faces : Nat)
  (blue_faces : Nat)
  (valid_die : total_sides = red_faces + yellow_faces + green_faces + blue_faces)

/-- Calculates the probability of rolling either a green or blue face -/
def probability_green_or_blue (die : ColoredDie) : Rat :=
  (die.green_faces + die.blue_faces : Rat) / die.total_sides

/-- Theorem stating the probability of rolling either a green or blue face -/
theorem probability_green_or_blue_ten_sided_die :
  ∃ (die : ColoredDie),
    die.total_sides = 10 ∧
    die.red_faces = 4 ∧
    die.yellow_faces = 3 ∧
    die.green_faces = 2 ∧
    die.blue_faces = 1 ∧
    probability_green_or_blue die = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_or_blue_ten_sided_die_l3313_331302


namespace NUMINAMATH_CALUDE_jenny_money_problem_l3313_331319

theorem jenny_money_problem (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
  sorry

end NUMINAMATH_CALUDE_jenny_money_problem_l3313_331319


namespace NUMINAMATH_CALUDE_twenty_eighth_term_of_sequence_l3313_331301

def sequence_term (n : ℕ) : ℚ :=
  1 / (2 ^ (sum_of_repeated_terms n))
where
  sum_of_repeated_terms : ℕ → ℕ
  | 0 => 0
  | k + 1 => if (sum_of_repeated_terms k + k + 1 < n) then k + 1 else k

theorem twenty_eighth_term_of_sequence :
  sequence_term 28 = 1 / (2 ^ 7) :=
sorry

end NUMINAMATH_CALUDE_twenty_eighth_term_of_sequence_l3313_331301


namespace NUMINAMATH_CALUDE_assign_25_to_4_l3313_331339

/-- The number of ways to assign different service providers to children -/
def assignProviders (n m : ℕ) : ℕ :=
  (n - 0) * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: Assigning 25 service providers to 4 children results in 303600 possibilities -/
theorem assign_25_to_4 : assignProviders 25 4 = 303600 := by
  sorry

#eval assignProviders 25 4

end NUMINAMATH_CALUDE_assign_25_to_4_l3313_331339


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3313_331377

theorem complex_number_quadrant : 
  let z : ℂ := (5 * Complex.I) / (1 - 2 * Complex.I)
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3313_331377


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3313_331321

theorem max_imaginary_part_of_roots (z : ℂ) (φ : ℝ) :
  z^6 - z^4 + z^2 - 1 = 0 →
  -π/2 ≤ φ ∧ φ ≤ π/2 →
  z.im = Real.sin φ →
  z.im ≤ Real.sin (π/4) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3313_331321


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3313_331310

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 4 + 2 * a^(x - 1)
  f 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3313_331310


namespace NUMINAMATH_CALUDE_product_of_non_shared_sides_squared_l3313_331374

/-- Represents a right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- Theorem about the product of non-shared sides of two specific right triangles -/
theorem product_of_non_shared_sides_squared
  (T₁ T₂ : RightTriangle)
  (h₁ : T₁.area = 3)
  (h₂ : T₂.area = 4)
  (h₃ : T₁.side1 = T₂.side1)  -- Shared side
  (h₄ : T₁.side2 = T₂.side2)  -- Shared side
  (h₅ : T₁.side1 = T₁.side2)  -- 45°-45°-90° triangle condition
  : (T₁.hypotenuse * T₂.hypotenuse)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_non_shared_sides_squared_l3313_331374


namespace NUMINAMATH_CALUDE_total_dress_designs_l3313_331379

/-- The number of fabric colors available --/
def num_colors : ℕ := 5

/-- The number of patterns available --/
def num_patterns : ℕ := 6

/-- The number of fabric types available --/
def num_fabric_types : ℕ := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_fabric_types = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l3313_331379


namespace NUMINAMATH_CALUDE_count_divisible_by_three_is_334_l3313_331330

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of numbers b_k divisible by 3, where 1 ≤ k ≤ 500 -/
def count_divisible_by_three : ℕ := sorry

theorem count_divisible_by_three_is_334 : count_divisible_by_three = 334 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_three_is_334_l3313_331330


namespace NUMINAMATH_CALUDE_incorrect_number_value_l3313_331382

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg incorrect_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 20)
  (h3 : incorrect_value = 26)
  (h4 : correct_avg = 26) :
  ∃ (actual_value : ℚ),
    n * correct_avg - (n * initial_avg - incorrect_value + actual_value) = 0 ∧
    actual_value = 86 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l3313_331382


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l3313_331398

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_ratio : ℝ
  water_ratio : ℝ
  gravel_weight : ℝ

/-- Properties of the cement mixture -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_ratio = 1/2 ∧
  m.water_ratio = 1/5 ∧
  m.gravel_weight = 15 ∧
  m.sand_ratio + m.water_ratio + m.gravel_weight / m.total_weight = 1

/-- Theorem stating that the total weight of the mixture is 50 pounds -/
theorem cement_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : 
  m.total_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l3313_331398


namespace NUMINAMATH_CALUDE_sum_floor_equals_n_l3313_331389

/-- For any natural number n, the sum of floor((n+2^k)/(2^(k+1))) from k=0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_n_l3313_331389


namespace NUMINAMATH_CALUDE_sampling_method_is_systematic_l3313_331399

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production line -/
structure ProductionLine where
  product : Type
  conveyorBelt : Bool
  inspectionInterval : ℕ
  fixedPosition : Bool

/-- Determines the sampling method based on the production line characteristics -/
def determineSamplingMethod (line : ProductionLine) : SamplingMethod :=
  if line.conveyorBelt && line.inspectionInterval > 0 && line.fixedPosition then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem: The sampling method for the given production line is systematic sampling -/
theorem sampling_method_is_systematic (line : ProductionLine) 
  (h1 : line.conveyorBelt = true)
  (h2 : line.inspectionInterval = 10)
  (h3 : line.fixedPosition = true) :
  determineSamplingMethod line = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_sampling_method_is_systematic_l3313_331399


namespace NUMINAMATH_CALUDE_pie_eating_contest_ratio_l3313_331307

theorem pie_eating_contest_ratio (bill_pies sierra_pies adam_pies : ℕ) :
  adam_pies = bill_pies + 3 →
  sierra_pies = 12 →
  bill_pies + adam_pies + sierra_pies = 27 →
  sierra_pies / bill_pies = 2 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_ratio_l3313_331307


namespace NUMINAMATH_CALUDE_determinant_scaling_l3313_331356

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 3 →
  Matrix.det !![3*x, 3*y; 6*z, 6*w] = 54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3313_331356


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3313_331358

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3313_331358


namespace NUMINAMATH_CALUDE_bookmarked_pages_march_end_l3313_331390

/-- Represents the number of bookmarked pages at the end of a month -/
def bookmarked_pages_at_month_end (
  pages_per_day : ℕ
) (initial_pages : ℕ) (days_in_month : ℕ) : ℕ :=
  initial_pages + pages_per_day * days_in_month

/-- Theorem: Given the conditions, prove that the total bookmarked pages at the end of March is 1330 -/
theorem bookmarked_pages_march_end :
  bookmarked_pages_at_month_end 30 400 31 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bookmarked_pages_march_end_l3313_331390


namespace NUMINAMATH_CALUDE_bowl_game_points_ratio_l3313_331387

theorem bowl_game_points_ratio :
  ∀ (noa_points phillip_points : ℕ) (multiple : ℚ),
    noa_points = 30 →
    phillip_points = noa_points * multiple →
    noa_points + phillip_points = 90 →
    phillip_points / noa_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_points_ratio_l3313_331387


namespace NUMINAMATH_CALUDE_a_range_l3313_331333

-- Define the function f(x) piecewise
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log a
  else (6 - a) * x - 4 * a

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) →
  1 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3313_331333


namespace NUMINAMATH_CALUDE_adjacent_nonadjacent_probability_l3313_331365

def num_students : ℕ := 5

def total_arrangements : ℕ := num_students.factorial

def valid_arrangements : ℕ := 24

theorem adjacent_nonadjacent_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_adjacent_nonadjacent_probability_l3313_331365


namespace NUMINAMATH_CALUDE_bobby_free_throws_l3313_331351

theorem bobby_free_throws (initial_throws : ℕ) (initial_success_rate : ℚ)
  (additional_throws : ℕ) (new_success_rate : ℚ) :
  initial_throws = 30 →
  initial_success_rate = 3/5 →
  additional_throws = 10 →
  new_success_rate = 16/25 →
  ∃ (last_successful_throws : ℕ),
    last_successful_throws = 8 ∧
    (initial_success_rate * initial_throws + last_successful_throws) / 
    (initial_throws + additional_throws) = new_success_rate :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_free_throws_l3313_331351


namespace NUMINAMATH_CALUDE_one_third_in_one_sixth_l3313_331335

theorem one_third_in_one_sixth :
  (1 : ℚ) / 6 / ((1 : ℚ) / 3) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_one_third_in_one_sixth_l3313_331335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3313_331380

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (h1 : seq.S 3 = 9)
  (h2 : seq.S 6 = 36) :
  seq.a 7 + seq.a 8 + seq.a 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3313_331380


namespace NUMINAMATH_CALUDE_fritz_money_l3313_331367

theorem fritz_money (fritz sean rick : ℝ) : 
  sean = fritz / 2 + 4 →
  rick = 3 * sean →
  rick + sean = 96 →
  fritz = 40 := by
sorry

end NUMINAMATH_CALUDE_fritz_money_l3313_331367


namespace NUMINAMATH_CALUDE_student_number_problem_l3313_331378

theorem student_number_problem (x y : ℤ) : 
  x = 121 → 2 * x - y = 102 → y = 140 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3313_331378


namespace NUMINAMATH_CALUDE_min_value_expression_l3313_331306

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  ∃ (min : ℝ), min = 60 ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' * b' * c' = 27 → 
    a'^2 + 6*a'*b' + 9*b'^2 + 3*c'^2 ≥ min) ∧
  (a^2 + 6*a*b + 9*b^2 + 3*c^2 = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3313_331306


namespace NUMINAMATH_CALUDE_function_characterization_l3313_331318

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, f (a * b) = f a + f b - f (Nat.gcd a b)) ∧
  (∀ p a : ℕ, Nat.Prime p → (f a ≥ f (a * p) → f a + f p ≥ f a * f p + 1))

theorem function_characterization (f : ℕ → ℕ) (h : is_valid_function f) :
  (∀ n : ℕ, f n = n) ∨ (∀ n : ℕ, f n = 1) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3313_331318


namespace NUMINAMATH_CALUDE_water_lost_is_eight_gallons_l3313_331385

/-- Represents the water filling and leaking scenario of a pool --/
structure PoolFilling where
  hour1_rate : ℝ
  hour2_3_rate : ℝ
  hour4_rate : ℝ
  final_amount : ℝ

/-- Calculates the amount of water lost due to the leak --/
def water_lost (p : PoolFilling) : ℝ :=
  p.hour1_rate * 1 + p.hour2_3_rate * 2 + p.hour4_rate * 1 - p.final_amount

/-- Theorem stating that for the given scenario, the water lost is 8 gallons --/
theorem water_lost_is_eight_gallons : 
  ∀ (p : PoolFilling), 
  p.hour1_rate = 8 ∧ 
  p.hour2_3_rate = 10 ∧ 
  p.hour4_rate = 14 ∧ 
  p.final_amount = 34 → 
  water_lost p = 8 := by
  sorry


end NUMINAMATH_CALUDE_water_lost_is_eight_gallons_l3313_331385


namespace NUMINAMATH_CALUDE_part_a_part_b_part_c_l3313_331305

-- Define a structure for convex polyhedra
structure ConvexPolyhedron where
  planes_of_symmetry : ℕ
  axes_of_symmetry : ℕ

-- Define a function to calculate the maximum number of planes of symmetry
def max_planes_of_symmetry (A B : ConvexPolyhedron) : ℕ :=
  if A.planes_of_symmetry = B.planes_of_symmetry
  then A.planes_of_symmetry + 1
  else min A.planes_of_symmetry B.planes_of_symmetry

-- Define a function to calculate the maximum number of axes of symmetry
def max_axes_of_symmetry (A B : ConvexPolyhedron) : ℕ :=
  if A.axes_of_symmetry = B.axes_of_symmetry
  then A.axes_of_symmetry
  else 1

-- Theorem for part a
theorem part_a (A B : ConvexPolyhedron) 
  (h1 : A.planes_of_symmetry = 2012) 
  (h2 : B.planes_of_symmetry = 2012) :
  max_planes_of_symmetry A B = 2013 :=
sorry

-- Theorem for part b
theorem part_b (A B : ConvexPolyhedron) 
  (h1 : A.planes_of_symmetry = 2012) 
  (h2 : B.planes_of_symmetry = 2013) :
  max_planes_of_symmetry A B = 2012 :=
sorry

-- Theorem for part c
theorem part_c (A B : ConvexPolyhedron) 
  (h1 : A.axes_of_symmetry = 2012) 
  (h2 : B.axes_of_symmetry = 2013) :
  max_axes_of_symmetry A B = 1 :=
sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_c_l3313_331305


namespace NUMINAMATH_CALUDE_box_surface_area_l3313_331346

/-- Calculates the interior surface area of a box formed by removing square corners from a rectangular sheet -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- The interior surface area of the box is 731 square units -/
theorem box_surface_area : 
  interior_surface_area 25 35 6 = 731 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l3313_331346


namespace NUMINAMATH_CALUDE_decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l3313_331395

theorem decrease_by_percentage (x : ℝ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 100) :
  x - (p / 100) * x = x * (1 - p / 100) :=
sorry

theorem eighty_decreased_by_eightyfive_percent :
  80 - (85 / 100) * 80 = 12 :=
sorry

end NUMINAMATH_CALUDE_decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l3313_331395


namespace NUMINAMATH_CALUDE_savings_account_growth_l3313_331309

/-- Represents the total amount in a savings account after a given number of months -/
def total_amount (initial_deposit : ℝ) (monthly_rate : ℝ) (months : ℝ) : ℝ :=
  initial_deposit * (1 + monthly_rate * months)

theorem savings_account_growth (x : ℝ) :
  let initial_deposit : ℝ := 100
  let monthly_rate : ℝ := 0.006
  let y : ℝ := total_amount initial_deposit monthly_rate x
  y = 100 * (1 + 0.006 * x) ∧
  total_amount initial_deposit monthly_rate 4 = 102.4 := by
  sorry

#check savings_account_growth

end NUMINAMATH_CALUDE_savings_account_growth_l3313_331309


namespace NUMINAMATH_CALUDE_sin_cube_identity_l3313_331343

theorem sin_cube_identity (θ : ℝ) :
  ∃! (c d : ℝ), ∀ θ, Real.sin θ ^ 3 = c * Real.sin (3 * θ) + d * Real.sin θ :=
by
  -- The unique pair (c, d) is (-1/4, 3/4)
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l3313_331343


namespace NUMINAMATH_CALUDE_ball_distribution_l3313_331332

theorem ball_distribution (n : ℕ) (k : ℕ) :
  -- Part (a): No empty boxes
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 19 5 → n = 20 ∧ k = 6) ∧
  -- Part (b): Some boxes can be empty
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 25 5 → n = 20 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l3313_331332


namespace NUMINAMATH_CALUDE_office_paper_shortage_l3313_331371

def paper_shortage (pack1 pack2 mon_wed_fri_usage tue_thu_usage : ℕ) (period : ℕ) : ℤ :=
  (pack1 + pack2 : ℤ) - (3 * mon_wed_fri_usage + 2 * tue_thu_usage) * period

theorem office_paper_shortage :
  paper_shortage 240 320 60 100 2 = -200 :=
by sorry

end NUMINAMATH_CALUDE_office_paper_shortage_l3313_331371


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3313_331394

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 448) 
  (h2 : height = 14) 
  (h3 : area = base * height) : 
  base = 32 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3313_331394


namespace NUMINAMATH_CALUDE_y_fourth_power_zero_l3313_331353

theorem y_fourth_power_zero (y : ℝ) (hy : y > 0) 
  (h : Real.sqrt (1 - y^2) + Real.sqrt (1 + y^2) = 2) : y^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_fourth_power_zero_l3313_331353


namespace NUMINAMATH_CALUDE_probability_divisible_by_three_l3313_331350

/-- The set of positive integers from 1 to 2007 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2007}

/-- The probability that a number in S is divisible by 3 -/
def prob_div_3 : ℚ := 669 / 2007

/-- The probability that a number in S is not divisible by 3 -/
def prob_not_div_3 : ℚ := 1338 / 2007

/-- The probability that b and c satisfy the condition when a is not divisible by 3 -/
def prob_bc_condition : ℚ := 2 / 9

theorem probability_divisible_by_three :
  (prob_div_3 + prob_not_div_3 * prob_bc_condition : ℚ) = 1265 / 2007 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_three_l3313_331350


namespace NUMINAMATH_CALUDE_quadruple_sequence_no_repetition_l3313_331308

/-- Transformation function for quadruples -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- Generates the sequence of quadruples starting from an initial quadruple -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_sequence_no_repetition (a₀ b₀ c₀ d₀ : ℝ) :
  (a₀, b₀, c₀, d₀) ≠ (1, 1, 1, 1) →
  ∀ i j : ℕ, i ≠ j →
    quadruple_sequence (a₀, b₀, c₀, d₀) i ≠ quadruple_sequence (a₀, b₀, c₀, d₀) j :=
by sorry

end NUMINAMATH_CALUDE_quadruple_sequence_no_repetition_l3313_331308


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l3313_331320

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l3313_331320


namespace NUMINAMATH_CALUDE_clara_sticker_ratio_l3313_331323

/-- Given Clara's sticker distribution, prove the ratio of stickers given to best friends
    to stickers left after giving to the boy is 1:2 -/
theorem clara_sticker_ratio :
  ∀ (initial stickers_to_boy stickers_left : ℕ),
  initial = 100 →
  stickers_to_boy = 10 →
  stickers_left = 45 →
  (initial - stickers_to_boy - stickers_left) * 2 = initial - stickers_to_boy :=
by sorry

end NUMINAMATH_CALUDE_clara_sticker_ratio_l3313_331323
