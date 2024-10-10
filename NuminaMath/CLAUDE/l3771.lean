import Mathlib

namespace weight_problem_l3771_377139

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 41 kg. -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 27 →
  (b + c) / 2 = 41 := by
sorry

end weight_problem_l3771_377139


namespace courtyard_length_l3771_377163

/-- Proves that the length of a courtyard is 25 meters given specific conditions -/
theorem courtyard_length : 
  ∀ (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ),
  width = 15 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (width * (total_bricks : ℝ) * brick_length * brick_width) / width = 25 := by
sorry

end courtyard_length_l3771_377163


namespace parabola_proof_l3771_377198

def parabola (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem parabola_proof :
  ∃ (b c : ℝ),
    (parabola 3 b c = 0) ∧
    (parabola 0 b c = -3) ∧
    (∀ x, parabola x b c = x^2 - 2*x - 3) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≤ 5) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≥ -4) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = 5) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = -4) :=
by
  sorry


end parabola_proof_l3771_377198


namespace paint_replacement_theorem_l3771_377115

def paint_replacement_fractions (initial_red initial_blue initial_green : ℚ)
                                (replacement_red replacement_blue replacement_green : ℚ)
                                (final_red final_blue final_green : ℚ) : Prop :=
  let r := (initial_red - final_red) / (initial_red - replacement_red)
  let b := (initial_blue - final_blue) / (initial_blue - replacement_blue)
  let g := (initial_green - final_green) / (initial_green - replacement_green)
  r = 2/3 ∧ b = 3/5 ∧ g = 7/15

theorem paint_replacement_theorem :
  paint_replacement_fractions (60/100) (40/100) (25/100) (30/100) (15/100) (10/100) (40/100) (25/100) (18/100) :=
by
  sorry

end paint_replacement_theorem_l3771_377115


namespace max_positive_cyclic_sequence_l3771_377118

theorem max_positive_cyclic_sequence (x : Fin 2022 → ℝ) 
  (h_nonzero : ∀ i, x i ≠ 0)
  (h_inequality : ∀ i : Fin 2022, x i + 1 / x (Fin.succ i) < 0)
  (h_cyclic : x 0 = x (Fin.last 2021)) : 
  (Finset.filter (fun i => x i > 0) Finset.univ).card ≤ 1010 := by
  sorry

end max_positive_cyclic_sequence_l3771_377118


namespace solution_set_when_a_is_one_a_value_for_given_solution_set_l3771_377193

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (I)
theorem solution_set_when_a_is_one (x : ℝ) :
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

-- Theorem for part (II)
theorem a_value_for_given_solution_set (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  a = 2 :=
sorry

end solution_set_when_a_is_one_a_value_for_given_solution_set_l3771_377193


namespace min_distance_vectors_l3771_377166

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem min_distance_vectors (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = 2 * Real.pi / 3)
  (h2 : a.1 * b.1 + a.2 * b.2 = -1) : 
  ∀ (c d : ℝ × ℝ), angle_between_vectors c d = 2 * Real.pi / 3 → 
  c.1 * d.1 + c.2 * d.2 = -1 → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≤ Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) :=
by sorry

end min_distance_vectors_l3771_377166


namespace systematic_sample_sequence_l3771_377194

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  first_number : Nat

/-- Calculates the next numbers in a systematic sample sequence -/
def next_numbers (s : SystematicSample) : List Nat :=
  let step := s.total_students / s.sample_size
  [1, 2, 3, 4].map (fun i => s.first_number + i * step)

theorem systematic_sample_sequence (s : SystematicSample) 
  (h1 : s.total_students = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.first_number = 4) :
  next_numbers s = [16, 28, 40, 52] := by
  sorry

end systematic_sample_sequence_l3771_377194


namespace new_years_eve_appetizer_cost_l3771_377151

def cost_per_person (chips_cost creme_fraiche_cost caviar_cost : ℚ) (num_people : ℕ) : ℚ :=
  (chips_cost + creme_fraiche_cost + caviar_cost) / num_people

theorem new_years_eve_appetizer_cost :
  cost_per_person 3 5 73 3 = 27 := by
  sorry

end new_years_eve_appetizer_cost_l3771_377151


namespace fraction_problem_l3771_377122

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 35)
  (h2 : (40/100) * N = 420) : F = 2/3 := by
  sorry

end fraction_problem_l3771_377122


namespace S_is_specific_set_l3771_377197

/-- A set of complex numbers satisfying certain conditions -/
def S : Set ℂ :=
  {z : ℂ | ∃ (n : ℕ), 2 < n ∧ n < 6 ∧ Complex.abs z = 1}

/-- The condition that 1 is in S -/
axiom one_in_S : (1 : ℂ) ∈ S

/-- The closure property of S -/
axiom S_closure (z₁ z₂ : ℂ) (h₁ : z₁ ∈ S) (h₂ : z₂ ∈ S) :
  z₁ - 2 * z₂ * Complex.cos (Complex.arg (z₁ / z₂)) ∈ S

/-- The theorem to be proved -/
theorem S_is_specific_set : S = {-1, 1, -Complex.I, Complex.I} := by
  sorry

end S_is_specific_set_l3771_377197


namespace min_distance_between_graphs_l3771_377128

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_between_graphs :
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ ∀ (x : ℝ), distance x ≥ min_dist :=
sorry

end min_distance_between_graphs_l3771_377128


namespace isosceles_triangle_square_equal_area_l3771_377171

/-- 
Given an isosceles triangle with base s and height h, and a square with side length s,
if their areas are equal, then the height of the triangle is twice the side length of the square.
-/
theorem isosceles_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) :
  (1 / 2) * s * h = s^2 → h = 2 * s := by sorry

end isosceles_triangle_square_equal_area_l3771_377171


namespace tile_1x1_position_l3771_377146

/-- Represents a position in the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents a 1x3 tile -/
structure Tile1x3 where
  start : Position
  horizontal : Bool

/-- Represents the placement of tiles in the 7x7 grid -/
structure TilePlacement where
  tiles1x3 : Finset Tile1x3
  tile1x1 : Position

/-- Predicate to check if a position is in the center or adjacent to the edges -/
def isCenterOrEdgeAdjacent (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6 ∨
  p.row = 1 ∨ p.row = 5 ∨ p.col = 1 ∨ p.col = 5

/-- Main theorem: The 1x1 tile must be in the center or adjacent to the edges -/
theorem tile_1x1_position (placement : TilePlacement) 
  (h1 : placement.tiles1x3.card = 16) 
  (h2 : ∀ t ∈ placement.tiles1x3, t.start.row < 7 ∧ t.start.col < 7) 
  (h3 : ∀ t ∈ placement.tiles1x3, 
    if t.horizontal 
    then t.start.col < 5 
    else t.start.row < 5) :
  isCenterOrEdgeAdjacent placement.tile1x1 :=
sorry

end tile_1x1_position_l3771_377146


namespace forty_two_divisible_by_seven_l3771_377100

theorem forty_two_divisible_by_seven : ∃ k : ℤ, 42 = 7 * k := by
  sorry

end forty_two_divisible_by_seven_l3771_377100


namespace unique_solution_condition_l3771_377157

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end unique_solution_condition_l3771_377157


namespace log_equation_solution_l3771_377199

theorem log_equation_solution : 
  ∃! x : ℝ, (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4)) ∧ 
  (x = 11 / 2) := by
  sorry

end log_equation_solution_l3771_377199


namespace arithmetic_sequence_problem_l3771_377125

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_a3 : a 3 = 5) 
  (h_a5 : a 5 = 3) : 
  a 8 = 0 := by
sorry

end arithmetic_sequence_problem_l3771_377125


namespace pool_cleaning_threshold_l3771_377117

/-- Represents the pool maintenance scenario -/
structure PoolMaintenance where
  capacity : ℕ  -- Pool capacity in milliliters
  splash_per_jump : ℕ  -- Amount of water splashed out per jump in milliliters
  num_jumps : ℕ  -- Number of jumps before cleaning

/-- Calculates the percentage of water remaining in the pool after jumps -/
def remaining_water_percentage (p : PoolMaintenance) : ℚ :=
  let remaining_water := p.capacity - p.splash_per_jump * p.num_jumps
  (remaining_water : ℚ) / (p.capacity : ℚ) * 100

/-- Theorem stating that the remaining water percentage is 80% for the given scenario -/
theorem pool_cleaning_threshold (p : PoolMaintenance) 
  (h1 : p.capacity = 2000000)
  (h2 : p.splash_per_jump = 400)
  (h3 : p.num_jumps = 1000) :
  remaining_water_percentage p = 80 := by
  sorry


end pool_cleaning_threshold_l3771_377117


namespace same_terminal_side_l3771_377190

-- Define a function to normalize angles to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem stating that -390° and 330° have the same terminal side
theorem same_terminal_side : normalizeAngle (-390) = normalizeAngle 330 := by
  sorry

end same_terminal_side_l3771_377190


namespace total_tickets_sold_l3771_377143

theorem total_tickets_sold (adult_price child_price : ℕ) 
  (adult_tickets child_tickets total_receipts : ℕ) :
  adult_price = 12 →
  child_price = 4 →
  adult_tickets = 90 →
  child_tickets = 40 →
  total_receipts = 840 →
  adult_tickets * adult_price + child_tickets * child_price = total_receipts →
  adult_tickets + child_tickets = 130 := by
sorry

end total_tickets_sold_l3771_377143


namespace fraction_product_equivalence_l3771_377162

theorem fraction_product_equivalence :
  ∀ x : ℝ, x ≠ 1 → ((x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0) :=
by sorry

end fraction_product_equivalence_l3771_377162


namespace triangle_midpoints_sum_l3771_377137

theorem triangle_midpoints_sum (a b c : ℝ) : 
  a + b + c = 15 → 
  a - b = 3 → 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end triangle_midpoints_sum_l3771_377137


namespace triangle_equilateral_l3771_377173

theorem triangle_equilateral (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_equation : 2 * (a * b^2 + b * c^2 + c * a^2) = a^2 * b + b^2 * c + c^2 * a + 3 * a * b * c) : 
  a = b ∧ b = c := by
  sorry

end triangle_equilateral_l3771_377173


namespace trapezium_side_length_l3771_377196

theorem trapezium_side_length 
  (x : ℝ) 
  (h : x > 0) 
  (area : ℝ) 
  (height : ℝ) 
  (other_side : ℝ) 
  (h_area : area = 228) 
  (h_height : height = 12) 
  (h_other_side : other_side = 18) 
  (h_trapezium_area : area = (1/2) * (x + other_side) * height) : 
  x = 20 := by
sorry

end trapezium_side_length_l3771_377196


namespace no_real_solutions_l3771_377142

theorem no_real_solutions :
  ¬∃ (x : ℝ), (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7 := by
  sorry

end no_real_solutions_l3771_377142


namespace price_difference_chips_pretzels_l3771_377121

/-- The price difference between chips and pretzels -/
theorem price_difference_chips_pretzels :
  ∀ (pretzel_price chip_price : ℕ),
    pretzel_price = 4 →
    2 * chip_price + 2 * pretzel_price = 22 →
    chip_price > pretzel_price →
    chip_price - pretzel_price = 3 := by
  sorry

end price_difference_chips_pretzels_l3771_377121


namespace soccer_substitutions_mod_1000_l3771_377167

/-- Number of ways to make n substitutions -/
def num_substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (13 - k) * num_substitutions k

/-- Total number of ways to make up to 4 substitutions -/
def total_substitutions : ℕ :=
  (List.range 5).map num_substitutions |> List.sum

theorem soccer_substitutions_mod_1000 :
  total_substitutions % 1000 = 25 := by
  sorry

end soccer_substitutions_mod_1000_l3771_377167


namespace range_of_expression_l3771_377158

theorem range_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 ≤ x^2 + y^2 + Real.sqrt (x * y) ∧ x^2 + y^2 + Real.sqrt (x * y) ≤ 9/8 := by
  sorry

end range_of_expression_l3771_377158


namespace simplify_sqrt_a_squared_b_over_two_l3771_377148

theorem simplify_sqrt_a_squared_b_over_two
  (a b : ℝ) (ha : a < 0) :
  Real.sqrt ((a^2 * b) / 2) = -a / 2 * Real.sqrt (2 * b) :=
by sorry

end simplify_sqrt_a_squared_b_over_two_l3771_377148


namespace apples_picked_total_l3771_377145

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total : total_apples = 11 := by
  sorry

end apples_picked_total_l3771_377145


namespace root_difference_implies_k_value_l3771_377126

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) →
  (∃ r s : ℝ, r^2 - k*r + 10 = 0 ∧ s^2 - k*s + 10 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 →
              (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 :=
by sorry

end root_difference_implies_k_value_l3771_377126


namespace probability_rain_given_east_wind_l3771_377136

/-- The probability of an east wind blowing -/
def P_east_wind : ℚ := 3/10

/-- The probability of rain -/
def P_rain : ℚ := 11/30

/-- The probability of both an east wind blowing and rain -/
def P_east_wind_and_rain : ℚ := 4/15

/-- The probability of rain given that there is an east wind blowing -/
def P_rain_given_east_wind : ℚ := P_east_wind_and_rain / P_east_wind

theorem probability_rain_given_east_wind :
  P_rain_given_east_wind = 8/9 := by
  sorry

end probability_rain_given_east_wind_l3771_377136


namespace sum_less_than_sqrt_three_sum_squares_l3771_377140

theorem sum_less_than_sqrt_three_sum_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c < Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end sum_less_than_sqrt_three_sum_squares_l3771_377140


namespace roots_sum_equals_four_l3771_377169

/-- Given that x₁ and x₂ are the roots of ln|x-2| = m for some real m, prove that x₁ + x₂ = 4 -/
theorem roots_sum_equals_four (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (|x₁ - 2|) = m) 
  (h₂ : Real.log (|x₂ - 2|) = m) : 
  x₁ + x₂ = 4 := by
  sorry

end roots_sum_equals_four_l3771_377169


namespace parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l3771_377101

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the product of distances from A and B to F
def distance_product (x1 x2 : ℝ) : ℝ := (x1 + 1) * (x2 + 1)

theorem parabola_minimum_distance_product :
  ∀ k : ℝ, ∀ x1 x2 : ℝ,
  (∃ y1 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ 
   line_through_focus k x1 y1 ∧ line_through_focus k x2 y2) →
  distance_product x1 x2 ≥ 4 :=
sorry

theorem parabola_minimum_distance_product_achieved :
  ∃ k x1 x2 : ℝ, ∃ y1 y2 : ℝ,
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line_through_focus k x1 y1 ∧ line_through_focus k x2 y2 ∧
  distance_product x1 x2 = 4 :=
sorry

end parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l3771_377101


namespace jia_opened_physical_store_l3771_377104

-- Define the possible shop types
inductive ShopType
| Taobao
| WeChat
| Physical

-- Define the graduates
inductive Graduate
| Jia
| Yi
| Bing

-- Define a function that assigns a shop type to each graduate
def shop : Graduate → ShopType := sorry

-- Define the statements made by each graduate
def jia_statement : Prop :=
  shop Graduate.Jia = ShopType.Taobao ∧ shop Graduate.Yi = ShopType.WeChat

def yi_statement : Prop :=
  shop Graduate.Jia = ShopType.WeChat ∧ shop Graduate.Bing = ShopType.Taobao

def bing_statement : Prop :=
  shop Graduate.Jia = ShopType.Physical ∧ shop Graduate.Yi = ShopType.Taobao

-- Define a function to count the number of true parts in a statement
def true_count (statement : Prop) : Nat := sorry

-- Theorem: Given the conditions, Jia must have opened a physical store
theorem jia_opened_physical_store :
  (true_count jia_statement = 1) →
  (true_count yi_statement = 1) →
  (true_count bing_statement = 1) →
  (shop Graduate.Jia = ShopType.Physical) :=
by sorry

end jia_opened_physical_store_l3771_377104


namespace two_digit_number_representation_l3771_377147

/-- Represents a two-digit number -/
def two_digit_number (a b : ℕ) : ℕ := 10 * b + a

/-- Theorem stating that a two-digit number with digit a in the units place
    and digit b in the tens place is represented as 10b + a -/
theorem two_digit_number_representation (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : b ≠ 0) :
  two_digit_number a b = 10 * b + a := by
  sorry

end two_digit_number_representation_l3771_377147


namespace M_in_fourth_quadrant_l3771_377134

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end M_in_fourth_quadrant_l3771_377134


namespace train_crossing_time_l3771_377123

/-- Proves that a train of given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2.5 := by
  sorry

end train_crossing_time_l3771_377123


namespace square_root_divided_by_15_l3771_377191

theorem square_root_divided_by_15 : Real.sqrt 3600 / 15 = 4 := by
  sorry

end square_root_divided_by_15_l3771_377191


namespace inscribed_angles_sum_l3771_377161

/-- Given a circle divided into 15 equal arcs, this theorem proves that
    the sum of two inscribed angles, one subtended by 3 arcs and the other by 5 arcs,
    is equal to 96 degrees. -/
theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle = 360) →
  (x = 3 * (circle / 15) / 2) →
  (y = 5 * (circle / 15) / 2) →
  x + y = 96 := by
  sorry

end inscribed_angles_sum_l3771_377161


namespace f_max_min_on_interval_l3771_377179

def f (x : ℝ) := 6 - 12 * x + x^3

theorem f_max_min_on_interval :
  let a := -1/3
  let b := 1
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    x_max = a ∧
    x_min = b ∧
    f x_max = 269/27 ∧
    f x_min = -5 :=
sorry

end f_max_min_on_interval_l3771_377179


namespace calculate_expression_l3771_377184

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end calculate_expression_l3771_377184


namespace p_necessary_not_sufficient_for_not_q_l3771_377174

-- Define the conditions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4
def q (x : ℝ) : Prop := |x - 2| > 1

-- Define the negation of q
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that p is a necessary but not sufficient condition for ¬q
theorem p_necessary_not_sufficient_for_not_q :
  (∀ x, not_q x → p x) ∧ 
  (∃ x, p x ∧ ¬(not_q x)) :=
sorry

end p_necessary_not_sufficient_for_not_q_l3771_377174


namespace angle_x_is_60_l3771_377113

/-- Given a geometric configuration where:
  1. y + 140° forms a straight angle
  2. There's a triangle with angles 40°, 80°, and z°
  3. x is an angle opposite to z
Prove that x = 60° -/
theorem angle_x_is_60 (y z x : ℝ) : 
  y + 140 = 180 →  -- Straight angle property
  40 + 80 + z = 180 →  -- Triangle angle sum property
  x = z →  -- Opposite angles are equal
  x = 60 := by sorry

end angle_x_is_60_l3771_377113


namespace student_arrangement_l3771_377119

theorem student_arrangement (n m k : ℕ) (hn : n = 5) (hm : m = 4) (hk : k = 2) : 
  (Nat.choose m k / 2) * (Nat.factorial n / Nat.factorial (n - k)) = 60 := by
  sorry

end student_arrangement_l3771_377119


namespace conic_section_type_l3771_377111

theorem conic_section_type (x y : ℝ) : 
  (9 * x^2 - 16 * y^2 = 0) → 
  ∃ (m₁ m₂ : ℝ), (∀ x y, (y = m₁ * x ∨ y = m₂ * x) ↔ 9 * x^2 - 16 * y^2 = 0) :=
by sorry

end conic_section_type_l3771_377111


namespace least_value_f_1998_l3771_377159

/-- A function from positive integers to positive integers satisfying the given condition -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ s t : ℕ+, f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  ∃ (f : ℕ+ → ℕ+), SpecialFunction f ∧
    (∀ g : ℕ+ → ℕ+, SpecialFunction g → f 1998 ≤ g 1998) ∧
    f 1998 = 120 :=
sorry

end least_value_f_1998_l3771_377159


namespace stream_speed_l3771_377165

/-- Proves that the speed of a stream is 3 kmph given specific boat travel times -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) : 
  boat_speed = 15 →
  downstream_time = 1 →
  upstream_time = 1.5 →
  (boat_speed + 3) * downstream_time = (boat_speed - 3) * upstream_time :=
by sorry

end stream_speed_l3771_377165


namespace arithmetic_operations_l3771_377177

theorem arithmetic_operations : 
  (6 + (-8) - (-5) = 3) ∧ (18 / (-3) + (-2) * (-4) = 2) := by
  sorry

end arithmetic_operations_l3771_377177


namespace common_root_condition_l3771_377156

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1001 ∧ 1001 * x = m - 1000 * x) ↔ (m = 2001 ∨ m = -2001) := by
  sorry

end common_root_condition_l3771_377156


namespace ellipse_and_circle_properties_l3771_377183

-- Define the points and shapes
structure Point where
  x : ℝ
  y : ℝ

def F : Point := ⟨0, -1⟩
def A : Point := ⟨0, 2⟩
def O : Point := ⟨0, 0⟩

structure Circle where
  center : Point
  radius : ℝ

structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

def Line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the problem conditions
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

def is_tangent_to_line (c : Circle) (l : ℝ → ℝ) : Prop :=
  ∃ p : Point, is_on_circle p c ∧ p.y = l p.x ∧
  ∀ q : Point, q ≠ p → is_on_circle q c → q.y ≠ l q.x

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.b^2 + (p.y - e.center.y)^2 / e.a^2 = 1

def is_focus_of_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 + (p.y - e.center.y)^2 = e.a^2 - e.b^2

-- Define the theorem
theorem ellipse_and_circle_properties :
  ∀ (Q : Circle) (N : Ellipse) (M : ℝ → ℝ) (m : ℝ → ℝ → ℝ) (Z : ℝ → ℝ),
  (∀ x : ℝ, is_on_circle F Q) →
  (is_tangent_to_line Q (Line 0 1)) →
  (N.center = O) →
  (is_focus_of_ellipse F N) →
  (is_on_ellipse A N) →
  (∀ k : ℝ, ∃ B C D E : Point,
    is_on_ellipse B N ∧ is_on_ellipse C N ∧
    B.y = m k B.x ∧ C.y = m k C.x ∧
    D.x^2 = -4 * D.y ∧ E.x^2 = -4 * E.y ∧
    D.y = m k D.x ∧ E.y = m k E.x) →
  (∀ x : ℝ, M x = -x^2 / 4) →
  (N.a = 2 ∧ N.b = Real.sqrt 3) →
  (∀ k : ℝ, 9 ≤ Z k ∧ Z k < 12) :=
by sorry

end ellipse_and_circle_properties_l3771_377183


namespace simple_interest_rate_for_doubling_l3771_377152

/-- Simple interest rate for a sum that doubles in 10 years -/
theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) :
  let years : ℝ := 10
  let final_amount : ℝ := 2 * principal
  let rate : ℝ := (final_amount - principal) / (principal * years) * 100
  rate = 10 := by
  sorry

end simple_interest_rate_for_doubling_l3771_377152


namespace no_solution_implies_m_geq_two_l3771_377116

theorem no_solution_implies_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end no_solution_implies_m_geq_two_l3771_377116


namespace tiffany_cans_l3771_377127

theorem tiffany_cans (x : ℕ) : x + 4 = 8 → x = 4 := by
  sorry

end tiffany_cans_l3771_377127


namespace fraction_simplification_l3771_377144

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (2*x - 1) / (x - 1) + x / (1 - x) = 1 := by
  sorry

end fraction_simplification_l3771_377144


namespace third_tea_price_l3771_377132

/-- The price of the first variety of tea in Rs per kg -/
def price1 : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price2 : ℝ := 135

/-- The price of the mixture in Rs per kg -/
def mixPrice : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio1 : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio2 : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio3 : ℝ := 2

/-- The theorem stating the price of the third variety of tea -/
theorem third_tea_price : 
  ∃ (price3 : ℝ), 
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixPrice ∧ 
    price3 = 175.5 := by
  sorry

end third_tea_price_l3771_377132


namespace largest_integral_x_l3771_377149

theorem largest_integral_x : ∃ (x : ℤ),
  (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ 
  (x : ℚ) / 6 < (7 / 11 : ℚ) ∧
  (∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (7 / 11 : ℚ) → y ≤ x) ∧
  x = 3 :=
by sorry

end largest_integral_x_l3771_377149


namespace midpoint_coordinate_sum_l3771_377186

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 7)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 9 := by sorry

end midpoint_coordinate_sum_l3771_377186


namespace subset_iff_a_eq_one_l3771_377187

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_iff_a_eq_one (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_iff_a_eq_one_l3771_377187


namespace unique_representation_l3771_377189

theorem unique_representation (A : ℕ) : 
  ∃! (x y : ℕ), A = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end unique_representation_l3771_377189


namespace systematic_sample_interval_count_l3771_377120

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of selected items within a given interval in a systematic sample -/
def selectedInInterval (s : SystematicSample) : ℕ :=
  let stepSize := s.totalPopulation / s.sampleSize
  let intervalSize := s.intervalEnd - s.intervalStart + 1
  intervalSize / stepSize

/-- The main theorem statement -/
theorem systematic_sample_interval_count :
  let s : SystematicSample := {
    totalPopulation := 840,
    sampleSize := 21,
    intervalStart := 481,
    intervalEnd := 720
  }
  selectedInInterval s = 6 := by sorry

end systematic_sample_interval_count_l3771_377120


namespace cosine_of_angle_between_vectors_l3771_377195

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is 2√13/13 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end cosine_of_angle_between_vectors_l3771_377195


namespace irrational_free_iff_zero_and_rational_l3771_377130

def M (a b c d : ℝ) : Set ℝ :=
  {y | ∃ x, y = a * x^3 + b * x^2 + c * x + d}

theorem irrational_free_iff_zero_and_rational (a b c d : ℝ) :
  (∀ y ∈ M a b c d, ∃ (q : ℚ), (y : ℝ) = q) ↔
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ ∃ (q : ℚ), (d : ℝ) = q) :=
sorry

end irrational_free_iff_zero_and_rational_l3771_377130


namespace sufficient_not_necessary_condition_l3771_377155

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x > 2 ∧ (x - 1)^2 > 1) ∧
  (∃ x : ℝ, (x - 1)^2 > 1 ∧ ¬(x > 2)) :=
by sorry

end sufficient_not_necessary_condition_l3771_377155


namespace length_of_lm_l3771_377133

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  area : ℝ
  altitude : ℝ
  base : ℝ

/-- A line segment parallel to the base of the triangle -/
structure ParallelLine where
  length : ℝ

/-- The resulting trapezoid after cutting the triangle -/
structure Trapezoid where
  area : ℝ

/-- Theorem: Length of LM in the given isosceles triangle scenario -/
theorem length_of_lm (triangle : IsoscelesTriangle) (trapezoid : Trapezoid) 
    (h1 : triangle.area = 200)
    (h2 : triangle.altitude = 40)
    (h3 : trapezoid.area = 150)
    (h4 : triangle.base = 2 * triangle.area / triangle.altitude) :
  ∃ (lm : ParallelLine), lm.length = 5 := by
  sorry

end length_of_lm_l3771_377133


namespace van_distance_van_distance_proof_l3771_377188

/-- The distance covered by a van under specific conditions -/
theorem van_distance : ℝ :=
  let initial_time : ℝ := 6
  let new_time_factor : ℝ := 3/2
  let new_speed : ℝ := 28
  let distance := new_speed * (new_time_factor * initial_time)
  252

/-- Proof that the van's distance is 252 km -/
theorem van_distance_proof : van_distance = 252 := by
  sorry

end van_distance_van_distance_proof_l3771_377188


namespace astrophysics_degrees_l3771_377106

def microphotonics : Real := 12
def home_electronics : Real := 24
def food_additives : Real := 15
def genetically_modified_microorganisms : Real := 29
def industrial_lubricants : Real := 8
def total_degrees : Real := 360

def other_sectors_total : Real :=
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def astrophysics_percentage : Real := 100 - other_sectors_total

theorem astrophysics_degrees : 
  (astrophysics_percentage / 100) * total_degrees = 43.2 := by
  sorry

end astrophysics_degrees_l3771_377106


namespace truck_loading_capacity_correct_bag_count_l3771_377109

theorem truck_loading_capacity (truck_capacity : ℕ) 
                                (box_count box_weight : ℕ) 
                                (crate_count crate_weight : ℕ) 
                                (sack_count sack_weight : ℕ) 
                                (bag_weight : ℕ) : ℕ :=
  let total_loaded := box_count * box_weight + crate_count * crate_weight + sack_count * sack_weight
  let remaining_capacity := truck_capacity - total_loaded
  remaining_capacity / bag_weight

theorem correct_bag_count : 
  truck_loading_capacity 13500 100 100 10 60 50 50 40 = 10 := by
  sorry

end truck_loading_capacity_correct_bag_count_l3771_377109


namespace middle_term_of_arithmetic_sequence_l3771_377168

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

theorem middle_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 12) 
  (h_last : a 6 = 54) :
  a 3 = 33 := by
sorry

end middle_term_of_arithmetic_sequence_l3771_377168


namespace digits_of_product_l3771_377108

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ (2^15 * 5^10 * 12 : ℕ) < 10^n ∧ (2^15 * 5^10 * 12 : ℕ) ≥ 10^(n-1) ∧ n = 13 := by
  sorry

end digits_of_product_l3771_377108


namespace equal_positive_numbers_l3771_377182

theorem equal_positive_numbers (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d) : 
  a = b ∧ b = c ∧ c = d := by
sorry

end equal_positive_numbers_l3771_377182


namespace october_visitors_l3771_377112

theorem october_visitors (oct nov dec : ℕ) : 
  nov = oct * 115 / 100 →
  dec = nov + 15 →
  oct + nov + dec = 345 →
  oct = 100 := by
sorry

end october_visitors_l3771_377112


namespace geometric_sequence_general_term_l3771_377110

/-- Geometric sequence with first three terms summing to 14 and common ratio 2 -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 = 14)

/-- The general term of the geometric sequence is 2^n -/
theorem geometric_sequence_general_term (a : ℕ+ → ℝ) 
  (h : GeometricSequence a) : 
  ∀ n : ℕ+, a n = 2^(n : ℝ) := by
sorry

end geometric_sequence_general_term_l3771_377110


namespace subtraction_problem_l3771_377185

theorem subtraction_problem : 
  (7000 / 10) - (7000 * (1 / 10) / 100) = 693 := by
  sorry

end subtraction_problem_l3771_377185


namespace rectangle_perimeter_l3771_377164

theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 363 → 2 * (l + b) = 88 := by
  sorry

end rectangle_perimeter_l3771_377164


namespace function_inequality_l3771_377135

theorem function_inequality (a x : ℝ) (h1 : a ≥ Real.exp (-2)) (h2 : x > 0) :
  a * x * Real.exp x - (x + 1)^2 ≥ Real.log x - x^2 - x - 2 := by
  sorry

end function_inequality_l3771_377135


namespace solution_verification_l3771_377160

theorem solution_verification (x : ℚ) : 
  x = 22 / 5 ↔ 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) := by
  sorry

end solution_verification_l3771_377160


namespace chopped_cube_height_l3771_377154

-- Define the cube
def cube_edge_length : ℝ := 2

-- Define the cut face as an equilateral triangle
def cut_face_is_equilateral_triangle : Prop := sorry

-- Define the remaining height
def remaining_height : ℝ := cube_edge_length - 1

-- Theorem statement
theorem chopped_cube_height :
  cut_face_is_equilateral_triangle →
  remaining_height = 1 := by sorry

end chopped_cube_height_l3771_377154


namespace min_value_reciprocal_sum_l3771_377181

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end min_value_reciprocal_sum_l3771_377181


namespace expression_simplification_l3771_377105

variable (a b : ℝ)

theorem expression_simplification (h : a ≠ -1) :
  b * (a - 1 + 1 / (a + 1)) / ((a^2 + 2*a) / (a + 1)) = a * b / (a + 2) :=
by sorry

end expression_simplification_l3771_377105


namespace max_geometric_mean_of_sequence_l3771_377131

/-- Given a sequence of six numbers where one number is 1, any three consecutive numbers have the same arithmetic mean, and the arithmetic mean of all six numbers is A, the maximum value of the geometric mean of any three consecutive numbers is ∛((3A - 1)² / 4). -/
theorem max_geometric_mean_of_sequence (A : ℝ) (seq : Fin 6 → ℝ) 
  (h1 : ∃ i, seq i = 1)
  (h2 : ∀ i : Fin 4, (seq i + seq (i + 1) + seq (i + 2)) / 3 = 
                     (seq (i + 1) + seq (i + 2) + seq (i + 3)) / 3)
  (h3 : (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5) / 6 = A) :
  ∃ i : Fin 4, (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) ≤ ((3*A - 1)^2 / 4)^(1/3 : ℝ) ∧ 
  ∀ j : Fin 4, (seq j * seq (j + 1) * seq (j + 2))^(1/3 : ℝ) ≤ 
               (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) :=
by sorry

end max_geometric_mean_of_sequence_l3771_377131


namespace box_surface_area_proof_l3771_377129

noncomputable def surface_area_of_box (a b c : ℝ) : ℝ :=
  2 * (a * b + b * c + c * a)

theorem box_surface_area_proof (a b c : ℝ) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c = 2 * a) 
  (h4 : 4 * a + 4 * b + 4 * c = 180) 
  (h5 : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  ∃ ε > 0, |surface_area_of_box a b c - 1051.540| < ε :=
sorry

end box_surface_area_proof_l3771_377129


namespace eliot_account_balance_l3771_377172

/-- Represents the bank account balances of Al and Eliot -/
structure BankAccounts where
  al : ℝ
  eliot : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (accounts : BankAccounts) : Prop :=
  accounts.al > accounts.eliot ∧
  accounts.al - accounts.eliot = (1 / 12) * (accounts.al + accounts.eliot) ∧
  1.10 * accounts.al - 1.15 * accounts.eliot = 22

/-- The theorem stating Eliot's account balance -/
theorem eliot_account_balance :
  ∀ accounts : BankAccounts, satisfiesConditions accounts → accounts.eliot = 146.67 := by
  sorry

end eliot_account_balance_l3771_377172


namespace solve_system_l3771_377180

theorem solve_system (x y : ℝ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 := by
  sorry

end solve_system_l3771_377180


namespace multiple_with_few_digits_l3771_377124

open Nat

theorem multiple_with_few_digits (k : ℕ) (h : k > 1) :
  ∃ p : ℕ, p.gcd k = k ∧ p < k^4 ∧ (∃ (d₁ d₂ d₃ d₄ : ℕ) (h : d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10),
    ∀ d : ℕ, d ∈ p.digits 10 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) :=
by sorry

end multiple_with_few_digits_l3771_377124


namespace max_sum_given_sum_squares_and_product_l3771_377153

theorem max_sum_given_sum_squares_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) (h2 : x * y = 45) : 
  x + y ≤ Real.sqrt 220 := by
  sorry

end max_sum_given_sum_squares_and_product_l3771_377153


namespace solution_set_part1_range_of_m_part2_l3771_377175

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 3| + |x + m|

-- Part 1: Solution set of f(x) ≥ 6 when m = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Part 2: Range of m when solution set of f(x) ≤ 5 is not empty
theorem range_of_m_part2 :
  (∃ x : ℝ, f m x ≤ 5) → m ∈ Set.Icc (-8) (-2) :=
sorry

end solution_set_part1_range_of_m_part2_l3771_377175


namespace four_digit_divisible_by_9_l3771_377114

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem four_digit_divisible_by_9 (B : ℕ) : 
  B ≤ 9 → is_divisible_by_9 (5000 + 100 * B + 10 * B + 3) → B = 5 := by
  sorry

end four_digit_divisible_by_9_l3771_377114


namespace primes_between_30_and_50_l3771_377107

/-- Count of prime numbers in a given range -/
def countPrimes (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

/-- The theorem stating that there are 5 prime numbers between 30 and 50 -/
theorem primes_between_30_and_50 : countPrimes 31 49 = 5 := by
  sorry

end primes_between_30_and_50_l3771_377107


namespace circumscribed_sphere_surface_area_l3771_377192

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius^2 = 3 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_l3771_377192


namespace fraction_meaningful_l3771_377138

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x^2 - 1)) ↔ x ≠ 1 ∧ x ≠ -1 := by
sorry

end fraction_meaningful_l3771_377138


namespace line_parabola_intersection_condition_l3771_377178

/-- Parabola C with equation x² = 1/2 * y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 1/2 * y

/-- Line passing through points (0, -4) and (t, 0) -/
def line_AB (t x y : ℝ) : Prop := 4*x - t*y - 4*t = 0

/-- The line does not intersect the parabola -/
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola_C x y ∧ line_AB t x y → False

/-- The range of t for which the line does not intersect the parabola -/
theorem line_parabola_intersection_condition (t : ℝ) :
  no_intersection t ↔ t < -Real.sqrt 2 ∨ t > Real.sqrt 2 :=
sorry

end line_parabola_intersection_condition_l3771_377178


namespace probability_no_adjacent_same_l3771_377150

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def sides : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := 25 / 108

/-- Theorem stating the probability of no two adjacent people rolling the same number -/
theorem probability_no_adjacent_same :
  prob_no_adjacent_same = 25 / 108 := by sorry

end probability_no_adjacent_same_l3771_377150


namespace range_of_x_l3771_377176

/-- Given a set M containing two elements, x^2 - 5x + 7 and 1, 
    prove that the range of real numbers x is all real numbers except 2 and 3. -/
theorem range_of_x (M : Set ℝ) (h : M = {x^2 - 5*x + 7 | x : ℝ} ∪ {1}) :
  {x : ℝ | x^2 - 5*x + 7 ≠ 1} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} :=
sorry

end range_of_x_l3771_377176


namespace average_headcount_rounded_l3771_377170

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ := (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem average_headcount_rounded : round_to_nearest average_headcount = 11467 := by
  sorry

end average_headcount_rounded_l3771_377170


namespace complex_equation_solution_l3771_377102

def i : ℂ := Complex.I

theorem complex_equation_solution :
  ∀ z : ℂ, (2 - i) * z = i^2021 → z = -1/5 + 2/5*i :=
by
  sorry

end complex_equation_solution_l3771_377102


namespace vector_b_proof_l3771_377141

def vector_a : Fin 2 → ℝ := ![2, -1]

theorem vector_b_proof (b : Fin 2 → ℝ) 
  (collinear : ∃ k : ℝ, k > 0 ∧ b = k • vector_a)
  (magnitude : Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = 2 * Real.sqrt 5) :
  b = ![4, -2] := by
  sorry

end vector_b_proof_l3771_377141


namespace min_value_theorem_l3771_377103

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 6) : 
  (1/x + 4/y + 9/z) ≥ 98/3 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 6 ∧ 1/x₀ + 4/y₀ + 9/z₀ = 98/3 :=
by sorry

end min_value_theorem_l3771_377103
