import Mathlib

namespace NUMINAMATH_CALUDE_square_area_increase_l3961_396190

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.4 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.96 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l3961_396190


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l3961_396160

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l3961_396160


namespace NUMINAMATH_CALUDE_sam_lee_rates_sum_of_squares_l3961_396191

theorem sam_lee_rates_sum_of_squares : 
  ∃ (r c k : ℕ+), 
    (4 * r + 5 * c + 3 * k = 120) ∧ 
    (5 * r + 3 * c + 4 * k = 138) ∧ 
    (r ^ 2 + c ^ 2 + k ^ 2 = 436) := by
  sorry

end NUMINAMATH_CALUDE_sam_lee_rates_sum_of_squares_l3961_396191


namespace NUMINAMATH_CALUDE_fraction_value_l3961_396153

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 2 * d) : 
  (a * c) / (b * d) = 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3961_396153


namespace NUMINAMATH_CALUDE_building_height_is_100_l3961_396155

/-- The height of a building with an elevator --/
def building_height (acceleration : ℝ) (constant_velocity : ℝ) (constant_time : ℝ) (acc_time : ℝ) : ℝ :=
  -- Distance during acceleration and deceleration
  2 * (0.5 * acceleration * acc_time^2) +
  -- Distance during constant velocity
  constant_velocity * constant_time

/-- Theorem stating the height of the building --/
theorem building_height_is_100 :
  building_height 2.5 5 18 2 = 100 := by
  sorry

#eval building_height 2.5 5 18 2

end NUMINAMATH_CALUDE_building_height_is_100_l3961_396155


namespace NUMINAMATH_CALUDE_probability_walking_200_or_less_l3961_396139

/-- Number of gates in the airport --/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet --/
def gate_distance : ℕ := 50

/-- Maximum walking distance in feet --/
def max_distance : ℕ := 200

/-- Calculate the number of favorable outcomes --/
def favorable_outcomes : ℕ := sorry

/-- Calculate the total number of possible outcomes --/
def total_outcomes : ℕ := num_gates * (num_gates - 1)

/-- The probability of walking 200 feet or less --/
theorem probability_walking_200_or_less :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 19 := by sorry

end NUMINAMATH_CALUDE_probability_walking_200_or_less_l3961_396139


namespace NUMINAMATH_CALUDE_unique_solution_l3961_396127

/-- Define the function f as specified in the problem -/
def f (x y z : ℕ+) : ℤ :=
  (((x + y - 2) * (x + y - 1)) / 2) - z

/-- Theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (a b c d : ℕ+), f a b c = 1993 ∧ f c d a = 1993 ∧ a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3961_396127


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3961_396125

/-- The quadratic equation bx^2 - 12x + 9 = 0 has exactly one solution when b = 4 -/
theorem quadratic_one_solution (b : ℝ) : 
  (∃! x, b * x^2 - 12 * x + 9 = 0) ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3961_396125


namespace NUMINAMATH_CALUDE_max_harmonious_t_patterns_l3961_396144

/-- Represents a coloring of an 8x8 grid --/
def Coloring := Fin 8 → Fin 8 → Bool

/-- Represents a T-shaped pattern in the grid --/
structure TPattern where
  row : Fin 8
  col : Fin 8
  orientation : Fin 4

/-- The total number of T-shaped patterns in an 8x8 grid --/
def total_t_patterns : Nat := 168

/-- Checks if a T-pattern is harmonious under a given coloring --/
def is_harmonious (c : Coloring) (t : TPattern) : Bool :=
  sorry

/-- Counts the number of harmonious T-patterns for a given coloring --/
def count_harmonious (c : Coloring) : Nat :=
  sorry

/-- The maximum number of harmonious T-patterns possible --/
def max_harmonious : Nat := 132

theorem max_harmonious_t_patterns :
  ∃ (c : Coloring), count_harmonious c = max_harmonious ∧
  ∀ (c' : Coloring), count_harmonious c' ≤ max_harmonious :=
sorry

end NUMINAMATH_CALUDE_max_harmonious_t_patterns_l3961_396144


namespace NUMINAMATH_CALUDE_belyNaliv_triple_l3961_396138

/-- Represents the number of apples of each variety -/
structure AppleCount where
  antonovka : ℝ
  grushovka : ℝ
  belyNaliv : ℝ

/-- The total number of apples -/
def totalApples (count : AppleCount) : ℝ :=
  count.antonovka + count.grushovka + count.belyNaliv

/-- Condition: Tripling Antonovka apples increases the total by 70% -/
axiom antonovka_triple (count : AppleCount) :
  2 * count.antonovka = 0.7 * totalApples count

/-- Condition: Tripling Grushovka apples increases the total by 50% -/
axiom grushovka_triple (count : AppleCount) :
  2 * count.grushovka = 0.5 * totalApples count

/-- Theorem: Tripling Bely Naliv apples increases the total by 80% -/
theorem belyNaliv_triple (count : AppleCount) :
  2 * count.belyNaliv = 0.8 * totalApples count := by
  sorry

end NUMINAMATH_CALUDE_belyNaliv_triple_l3961_396138


namespace NUMINAMATH_CALUDE_correct_calculation_l3961_396150

theorem correct_calculation (x : ℝ) (h : x + 20 = 180) : x / 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3961_396150


namespace NUMINAMATH_CALUDE_cupboard_cost_price_proof_l3961_396120

/-- The cost price of a cupboard satisfying given conditions -/
def cupboard_cost_price : ℝ := 6250

/-- The selling price of the cupboard -/
def selling_price (cost : ℝ) : ℝ := cost * (1 - 0.12)

/-- The selling price that would result in a 12% profit -/
def profit_selling_price (cost : ℝ) : ℝ := cost * (1 + 0.12)

theorem cupboard_cost_price_proof :
  selling_price cupboard_cost_price + 1500 = profit_selling_price cupboard_cost_price :=
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_proof_l3961_396120


namespace NUMINAMATH_CALUDE_mario_age_l3961_396106

/-- Mario and Maria's ages problem -/
theorem mario_age (mario_age maria_age : ℕ) : 
  mario_age + maria_age = 7 → 
  mario_age = maria_age + 1 → 
  mario_age = 4 := by
sorry

end NUMINAMATH_CALUDE_mario_age_l3961_396106


namespace NUMINAMATH_CALUDE_ad_campaign_cost_l3961_396193

/-- Calculates the total cost of an ad campaign with given parameters and discount rules --/
theorem ad_campaign_cost 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (full_page_rate : ℝ) 
  (half_page_rate : ℝ) 
  (quarter_page_rate : ℝ) 
  (eighth_page_rate : ℝ) 
  (half_page_count : ℕ) 
  (quarter_page_count : ℕ) 
  (eighth_page_count : ℕ) 
  (discount_rate_4_to_5 : ℝ) 
  (discount_rate_6_or_more : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  full_page_rate = 6.5 → 
  half_page_rate = 8 → 
  quarter_page_rate = 10 → 
  eighth_page_rate = 12 → 
  half_page_count = 1 → 
  quarter_page_count = 3 → 
  eighth_page_count = 4 → 
  discount_rate_4_to_5 = 0.1 → 
  discount_rate_6_or_more = 0.15 → 
  ∃ (total_cost : ℝ), total_cost = 1606.5 := by
  sorry


end NUMINAMATH_CALUDE_ad_campaign_cost_l3961_396193


namespace NUMINAMATH_CALUDE_electronic_dogs_distance_l3961_396104

/-- Represents a vertex of a cube --/
inductive Vertex
| A | B | C | D | A1 | B1 | C1 | D1

/-- Represents the position of an electronic dog on the cube --/
structure DogPosition where
  vertex : Vertex
  segments_completed : Nat

/-- The cube with edge length 1 --/
def unitCube : Set Vertex := {Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.A1, Vertex.B1, Vertex.C1, Vertex.D1}

/-- The distance between two vertices of the unit cube --/
def distance (v1 v2 : Vertex) : Real := sorry

/-- The movement rule for the dogs --/
def validMove (v1 v2 v3 : Vertex) : Prop := sorry

/-- The final position of the black dog after 2008 segments --/
def blackDogFinalPosition : DogPosition := ⟨Vertex.A, 2008⟩

/-- The final position of the yellow dog after 2009 segments --/
def yellowDogFinalPosition : DogPosition := ⟨Vertex.A1, 2009⟩

theorem electronic_dogs_distance :
  distance blackDogFinalPosition.vertex yellowDogFinalPosition.vertex = 1 := by sorry

end NUMINAMATH_CALUDE_electronic_dogs_distance_l3961_396104


namespace NUMINAMATH_CALUDE_savings_calculation_l3961_396121

/-- Calculates the savings of a person given their income and the ratio of income to expenditure -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given a person's income of 14000 and income to expenditure ratio of 7:6, their savings are 2000 -/
theorem savings_calculation :
  calculate_savings 14000 7 6 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3961_396121


namespace NUMINAMATH_CALUDE_eleventh_term_ratio_l3961_396168

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.firstTerm + (n - 1) * seq.commonDiff) / 2

/-- nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem eleventh_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, sumOfTerms seq1 n / sumOfTerms seq2 n = (7 * n + 1) / (4 * n + 27)) :
  nthTerm seq1 11 / nthTerm seq2 11 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_ratio_l3961_396168


namespace NUMINAMATH_CALUDE_sine_HAC_specific_prism_l3961_396179

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- Calculate the sine of the angle HAC in a rectangular prism -/
def sineHAC (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The sine of angle HAC in the given rectangular prism is √143 / 13 -/
theorem sine_HAC_specific_prism :
  let prism : RectangularPrism := {
    a := 2,
    b := 2,
    c := 3,
    A := ⟨0, 0, 0⟩,
    B := ⟨2, 0, 0⟩,
    C := ⟨2, 2, 0⟩,
    D := ⟨0, 2, 0⟩,
    E := ⟨0, 0, 3⟩,
    F := ⟨2, 0, 3⟩,
    G := ⟨2, 2, 3⟩,
    H := ⟨0, 2, 3⟩
  }
  sineHAC prism = Real.sqrt 143 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sine_HAC_specific_prism_l3961_396179


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l3961_396151

theorem strawberry_milk_probability :
  let n : ℕ := 6  -- Total number of days
  let k : ℕ := 5  -- Number of successful days
  let p : ℚ := 3/4  -- Probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 729/2048 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l3961_396151


namespace NUMINAMATH_CALUDE_library_items_count_l3961_396163

theorem library_items_count (notebooks : ℕ) (pens : ℕ) : 
  notebooks = 30 →
  pens = notebooks + 50 →
  notebooks + pens = 110 := by
  sorry

end NUMINAMATH_CALUDE_library_items_count_l3961_396163


namespace NUMINAMATH_CALUDE_horizon_fantasy_meetup_handshakes_l3961_396100

/-- Calculates the number of handshakes in a group where everyone shakes hands with everyone else once -/
def handshakesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of handshakes between two groups where everyone in one group shakes hands with everyone in the other group once -/
def handshakesBetweenGroups (n m : ℕ) : ℕ := n * m

theorem horizon_fantasy_meetup_handshakes :
  let gremlins : ℕ := 25
  let imps : ℕ := 20
  let sprites : ℕ := 10
  let gremlinHandshakes := handshakesInGroup gremlins
  let gremlinImpHandshakes := handshakesBetweenGroups gremlins imps
  let spriteHandshakes := handshakesInGroup sprites
  let gremlinSpriteHandshakes := handshakesBetweenGroups gremlins sprites
  gremlinHandshakes + gremlinImpHandshakes + spriteHandshakes + gremlinSpriteHandshakes = 1095 := by
  sorry

#eval handshakesInGroup 25 + handshakesBetweenGroups 25 20 + handshakesInGroup 10 + handshakesBetweenGroups 25 10

end NUMINAMATH_CALUDE_horizon_fantasy_meetup_handshakes_l3961_396100


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_proposition_4_l3961_396112

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Notation
local notation:50 l1:50 " ⊥ " l2:50 => perpendicular l1 l2
local notation:50 l1:50 " ∥ " l2:50 => parallel l1 l2
local notation:50 l:50 " ⊥ " p:50 => perpendicularToPlane l p
local notation:50 l:50 " ∥ " p:50 => parallelToPlane l p
local notation:50 p1:50 " ⊥ " p2:50 => perpendicularPlanes p1 p2
local notation:50 p1:50 " ∥ " p2:50 => parallelPlanes p1 p2

-- Theorem statements
theorem proposition_1 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ⊥ β) → (m ⊥ n) → (α ⊥ β) := by sorry

theorem proposition_2 (m n : Line) (α β : Plane) :
  ¬ ((m ∥ α) → (n ∥ β) → (m ∥ n) → (α ∥ β)) := by sorry

theorem proposition_3 (m n : Line) (α β : Plane) :
  ¬ ((m ⊥ α) → (n ∥ β) → (m ⊥ n) → (α ⊥ β)) := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ∥ β) → (m ∥ n) → (α ⊥ β) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_proposition_4_l3961_396112


namespace NUMINAMATH_CALUDE_total_crayons_l3961_396156

theorem total_crayons (boxes : ℕ) (crayons_per_box : ℕ) (h1 : boxes = 8) (h2 : crayons_per_box = 7) :
  boxes * crayons_per_box = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3961_396156


namespace NUMINAMATH_CALUDE_neighbor_purchase_theorem_l3961_396119

/-- Proves that given the conditions of the problem, the total amount spent is 168 shillings -/
theorem neighbor_purchase_theorem (x : ℝ) 
  (h1 : x > 0)  -- Quantity purchased is positive
  (h2 : 2*x + 1.5*x = 3.5*x)  -- Total cost equation
  (h3 : (3.5*x/2)/2 + (3.5*x/2)/1.5 = 2*x + 2)  -- Equal division condition
  : 3.5*x = 168 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_purchase_theorem_l3961_396119


namespace NUMINAMATH_CALUDE_diamond_value_l3961_396110

/-- The diamond operation for non-zero integers -/
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (2 : ℚ) / b

/-- Theorem stating the value of a ◇ b given the conditions -/
theorem diamond_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  diamond a b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l3961_396110


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l3961_396187

/-- Given a line with slope -3.5 and y-intercept 1.5, 
    the x-coordinate of the point with y-coordinate 1025 is -1023.5 / 3.5 -/
theorem line_point_x_coordinate 
  (slope : ℝ) 
  (y_intercept : ℝ) 
  (y : ℝ) 
  (h1 : slope = -3.5) 
  (h2 : y_intercept = 1.5) 
  (h3 : y = 1025) : 
  (y - y_intercept) / (-slope) = -1023.5 / 3.5 := by
  sorry

#eval -1023.5 / 3.5

end NUMINAMATH_CALUDE_line_point_x_coordinate_l3961_396187


namespace NUMINAMATH_CALUDE_car_speed_graph_comparison_l3961_396157

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The graph representation of a car's travel -/
structure GraphLine where
  height : ℝ
  length : ℝ

/-- Theorem: If Car N travels at three times the speed of Car M for the same distance,
    then on a speed-time graph, Car N's line will be thrice as high and one-third the length of Car M's line. -/
theorem car_speed_graph_comparison (m n : Car) (gm gn : GraphLine) :
  n.speed = 3 * m.speed →
  m.distance = n.distance →
  gm.height = m.speed →
  gm.length = m.time →
  gn.height = n.speed →
  gn.length = n.time →
  gn.height = 3 * gm.height ∧ gn.length = gm.length / 3 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_graph_comparison_l3961_396157


namespace NUMINAMATH_CALUDE_bolt_width_calculation_l3961_396140

/-- The width of a bolt of fabric given specific cuts and remaining area --/
theorem bolt_width_calculation (living_room_length living_room_width bedroom_length bedroom_width bolt_length remaining_fabric : ℝ) 
  (h1 : living_room_length = 4)
  (h2 : living_room_width = 6)
  (h3 : bedroom_length = 2)
  (h4 : bedroom_width = 4)
  (h5 : bolt_length = 12)
  (h6 : remaining_fabric = 160) :
  (remaining_fabric + living_room_length * living_room_width + bedroom_length * bedroom_width) / bolt_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_bolt_width_calculation_l3961_396140


namespace NUMINAMATH_CALUDE_king_arthur_advisors_l3961_396103

theorem king_arthur_advisors (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  let q := 1 - p
  let prob_correct_two_advisors := p^2 + 2*p*q*(1/2)
  prob_correct_two_advisors = p :=
by sorry

end NUMINAMATH_CALUDE_king_arthur_advisors_l3961_396103


namespace NUMINAMATH_CALUDE_wedge_volume_approximation_l3961_396114

/-- The volume of a wedge cut from a cylinder --/
theorem wedge_volume_approximation (r h : ℝ) (h_r : r = 6) (h_h : h = 6) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := cylinder_volume / 2
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |wedge_volume - 339.12| < ε :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_approximation_l3961_396114


namespace NUMINAMATH_CALUDE_points_needed_in_next_game_l3961_396188

def last_home_game_score : ℕ := 62

def first_away_game_score : ℕ := last_home_game_score / 2

def second_away_game_score : ℕ := first_away_game_score + 18

def third_away_game_score : ℕ := second_away_game_score + 2

def cumulative_score_goal : ℕ := 4 * last_home_game_score

def current_cumulative_score : ℕ := 
  last_home_game_score + first_away_game_score + second_away_game_score + third_away_game_score

theorem points_needed_in_next_game : 
  cumulative_score_goal - current_cumulative_score = 55 := by
  sorry

end NUMINAMATH_CALUDE_points_needed_in_next_game_l3961_396188


namespace NUMINAMATH_CALUDE_harry_sea_stars_harry_collected_34_sea_stars_l3961_396192

theorem harry_sea_stars : ℕ → Prop :=
  fun sea_stars =>
    sea_stars + 21 + 29 = 59 + 25 ∧ 
    sea_stars = 34

/-- Proof that Harry collected 34 sea stars initially -/
theorem harry_collected_34_sea_stars : ∃ (sea_stars : ℕ), harry_sea_stars sea_stars :=
by
  sorry

end NUMINAMATH_CALUDE_harry_sea_stars_harry_collected_34_sea_stars_l3961_396192


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l3961_396158

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ,
    x > 0 →
    y = x + 1 →
    x * y = 812 →
    x + y = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l3961_396158


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3961_396142

theorem max_value_trig_expression (α β : Real) (h1 : 0 ≤ α ∧ α ≤ π/4) (h2 : 0 ≤ β ∧ β ≤ π/4) :
  ∃ (M : Real), M = Real.sqrt 5 ∧ ∀ (x y : Real), 0 ≤ x ∧ x ≤ π/4 → 0 ≤ y ∧ y ≤ π/4 →
    Real.sin (x - y) + 2 * Real.sin (x + y) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3961_396142


namespace NUMINAMATH_CALUDE_day_of_week_problem_l3961_396143

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek :=
  sorry

theorem day_of_week_problem (N : Year) :
  dayOfWeek N 250 = DayOfWeek.Sunday →
  dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Sunday →
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_problem_l3961_396143


namespace NUMINAMATH_CALUDE_slope_angle_range_l3961_396124

-- Define the slope k and the angle θ
variable (k : ℝ) (θ : ℝ)

-- Define the condition that the lines intersect in the first quadrant
def intersect_in_first_quadrant (k : ℝ) : Prop :=
  (3 + Real.sqrt 3) / (1 + k) > 0 ∧ (3 * k - Real.sqrt 3) / (1 + k) > 0

-- Define the relationship between k and θ
def slope_angle_relation (k θ : ℝ) : Prop :=
  k = Real.tan θ

-- State the theorem
theorem slope_angle_range (h1 : intersect_in_first_quadrant k) 
  (h2 : slope_angle_relation k θ) : 
  θ > Real.pi / 6 ∧ θ < Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l3961_396124


namespace NUMINAMATH_CALUDE_star_symmetric_set_eq_three_lines_l3961_396164

/-- The star operation -/
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_symmetric_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x + y = 0 -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0}

theorem star_symmetric_set_eq_three_lines :
  star_symmetric_set = three_lines := by sorry

end NUMINAMATH_CALUDE_star_symmetric_set_eq_three_lines_l3961_396164


namespace NUMINAMATH_CALUDE_calories_burned_jogging_l3961_396123

/-- Calculate calories burned by jogging -/
theorem calories_burned_jogging (laps_per_night : ℕ) (feet_per_lap : ℕ) (feet_per_calorie : ℕ) (days : ℕ) : 
  laps_per_night = 5 →
  feet_per_lap = 100 →
  feet_per_calorie = 25 →
  days = 5 →
  (laps_per_night * feet_per_lap * days) / feet_per_calorie = 100 := by
  sorry

end NUMINAMATH_CALUDE_calories_burned_jogging_l3961_396123


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3961_396169

/-- The line l with equation x - y + √3 = 0 intersects the circle C with equation x² + (y - √2)² = 2 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), (x - y + Real.sqrt 3 = 0) ∧ (x^2 + (y - Real.sqrt 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3961_396169


namespace NUMINAMATH_CALUDE_imaginary_unit_power_sum_l3961_396147

theorem imaginary_unit_power_sum : ∀ i : ℂ, i^2 = -1 →
  i^15300 + i^15301 + i^15302 + i^15303 + i^15304 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_sum_l3961_396147


namespace NUMINAMATH_CALUDE_total_distance_mercedes_davonte_l3961_396194

/-- 
Given:
- Jonathan ran 7.5 kilometers
- Mercedes ran twice the distance of Jonathan
- Davonte ran 2 kilometers farther than Mercedes

Prove that the total distance run by Mercedes and Davonte is 32 kilometers
-/
theorem total_distance_mercedes_davonte (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ)
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_mercedes_davonte_l3961_396194


namespace NUMINAMATH_CALUDE_candy_ratio_l3961_396186

theorem candy_ratio (chocolate_bars : ℕ) (m_and_ms : ℕ) (marshmallows : ℕ) :
  chocolate_bars = 5 →
  marshmallows = 6 * m_and_ms →
  chocolate_bars + m_and_ms + marshmallows = 250 →
  m_and_ms / chocolate_bars = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l3961_396186


namespace NUMINAMATH_CALUDE_two_cos_45_equals_sqrt_2_l3961_396133

theorem two_cos_45_equals_sqrt_2 : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_45_equals_sqrt_2_l3961_396133


namespace NUMINAMATH_CALUDE_election_votes_total_l3961_396180

/-- Proves that the total number of votes in an election is 180, given that Emma received 4/15 of the total votes and 48 votes in total. -/
theorem election_votes_total (emma_fraction : Rat) (emma_votes : ℕ) (total_votes : ℕ) 
  (h1 : emma_fraction = 4 / 15)
  (h2 : emma_votes = 48)
  (h3 : emma_fraction * total_votes = emma_votes) :
  total_votes = 180 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_total_l3961_396180


namespace NUMINAMATH_CALUDE_cos_585_degrees_l3961_396165

theorem cos_585_degrees :
  Real.cos (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_585_degrees_l3961_396165


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l3961_396111

theorem smallest_x_abs_equation : 
  (∃ x : ℝ, |x - 8| = 9) ∧ (∀ x : ℝ, |x - 8| = 9 → x ≥ -1) ∧ |-1 - 8| = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l3961_396111


namespace NUMINAMATH_CALUDE_evaluate_expression_l3961_396115

theorem evaluate_expression : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3961_396115


namespace NUMINAMATH_CALUDE_simplify_power_l3961_396149

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by sorry

end NUMINAMATH_CALUDE_simplify_power_l3961_396149


namespace NUMINAMATH_CALUDE_sled_total_distance_l3961_396134

/-- The distance traveled by a sled in n seconds, given initial distance and acceleration -/
def sledDistance (initialDistance : ℕ) (acceleration : ℕ) (n : ℕ) : ℕ :=
  n * (2 * initialDistance + (n - 1) * acceleration) / 2

/-- Theorem stating the total distance traveled by the sled -/
theorem sled_total_distance :
  sledDistance 8 10 40 = 8120 := by
  sorry

end NUMINAMATH_CALUDE_sled_total_distance_l3961_396134


namespace NUMINAMATH_CALUDE_circle_center_range_l3961_396146

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the circle C
def circle_C (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_center_range :
  ∀ center_x center_y : ℝ,
  line_l center_x center_y →
  (∃ x : ℝ, circle_C center_x center_y x 0 ∧ circle_C center_x center_y (-x) 0 ∧ x^2 = 3/4) →
  (∃ mx my : ℝ, circle_C center_x center_y mx my ∧
    (mx - point_A.1)^2 + (my - point_A.2)^2 = 4 * ((mx - center_x)^2 + (my - center_y)^2)) →
  0 ≤ center_x ∧ center_x ≤ 12/5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_range_l3961_396146


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_product_60_l3961_396174

/-- The number of distinct prime factors of the product of divisors of 60 -/
theorem distinct_prime_factors_of_divisor_product_60 : ∃ (B : ℕ), 
  (∀ d : ℕ, d ∣ 60 → d ∣ B) ∧ 
  (∀ n : ℕ, (∀ d : ℕ, d ∣ 60 → d ∣ n) → B ∣ n) ∧
  (Nat.card {p : ℕ | Nat.Prime p ∧ p ∣ B} = 3) :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_product_60_l3961_396174


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3961_396172

theorem polynomial_division_theorem (x : ℝ) :
  (x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32) * (x + 2) + 76 = x^6 + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3961_396172


namespace NUMINAMATH_CALUDE_coin_distribution_proof_l3961_396197

/-- Represents the coin distribution scheme between Charlie and Fred -/
def coin_distribution (x : ℕ) : Prop :=
  -- Charlie's coins are the sum of 1 to x
  let charlie_coins := x * (x + 1) / 2
  -- Fred's coins are x at the end
  let fred_coins := x
  -- Charlie has 5 times as many coins as Fred
  charlie_coins = 5 * fred_coins

/-- The total number of coins after distribution -/
def total_coins (x : ℕ) : ℕ := x * 6

theorem coin_distribution_proof :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_proof_l3961_396197


namespace NUMINAMATH_CALUDE_sum_of_triple_products_of_roots_l3961_396130

theorem sum_of_triple_products_of_roots (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 18 * p^2 - 14 * p + 7 = 0) →
  (4 * q^4 - 8 * q^3 + 18 * q^2 - 14 * q + 7 = 0) →
  (4 * r^4 - 8 * r^3 + 18 * r^2 - 14 * r + 7 = 0) →
  (4 * s^4 - 8 * s^3 + 18 * s^2 - 14 * s + 7 = 0) →
  p * q * r + p * q * s + p * r * s + q * r * s = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_triple_products_of_roots_l3961_396130


namespace NUMINAMATH_CALUDE_base_number_proof_l3961_396113

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : 4 * x^(2*n) = 4^18) (h2 : n = 17) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3961_396113


namespace NUMINAMATH_CALUDE_ap_terms_count_l3961_396184

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  even_n : Even n
  odd_sum : (n / 2) * (a + (a + (n - 2) * d)) = 30
  even_sum : (n / 2) * ((a + d) + (a + (n - 1) * d)) = 45
  last_first_diff : (a + (n - 1) * d) - a = 7.5

/-- The theorem stating that the number of terms in the arithmetic progression is 12 -/
theorem ap_terms_count (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_terms_count_l3961_396184


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l3961_396132

/-- Calculates the profit percentage with a given discount, based on the no-discount profit percentage. -/
def profit_with_discount (no_discount_profit : ℝ) (discount : ℝ) : ℝ :=
  ((1 + no_discount_profit) * (1 - discount) - 1) * 100

/-- Theorem stating that with a 5% discount and a 150% no-discount profit, the profit is 137.5% -/
theorem discount_profit_calculation :
  profit_with_discount 1.5 0.05 = 137.5 := by
  sorry

#eval profit_with_discount 1.5 0.05

end NUMINAMATH_CALUDE_discount_profit_calculation_l3961_396132


namespace NUMINAMATH_CALUDE_coopers_age_l3961_396176

theorem coopers_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (maria_older : maria = dante + 1) :
  cooper = 6 := by
  sorry

end NUMINAMATH_CALUDE_coopers_age_l3961_396176


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3961_396170

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 170 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3961_396170


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l3961_396167

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Represents the line PQ that cuts the triangle -/
structure CuttingLine where
  length : ℝ

/-- Theorem statement for the triangle problem -/
theorem triangle_cut_theorem 
  (triangle : IsoscelesTriangle) 
  (cutting_line : CuttingLine) : 
  triangle.height = 30 ∧ 
  triangle.base * triangle.height / 2 = 180 ∧
  triangle.base * triangle.height / 2 - 135 = 
    (triangle.base * triangle.height / 2) / 4 →
  cutting_line.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l3961_396167


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l3961_396182

theorem smallest_divisible_by_one_to_ten : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l3961_396182


namespace NUMINAMATH_CALUDE_kirills_height_l3961_396148

theorem kirills_height (h_kirill : ℕ) (h_brother : ℕ) 
  (height_difference : h_brother = h_kirill + 14)
  (total_height : h_kirill + h_brother = 112) : 
  h_kirill = 49 := by
  sorry

end NUMINAMATH_CALUDE_kirills_height_l3961_396148


namespace NUMINAMATH_CALUDE_sum_of_medians_is_64_l3961_396136

def median (scores : List ℕ) : ℚ :=
  sorry

theorem sum_of_medians_is_64 (scores_A scores_B : List ℕ) : 
  median scores_A + median scores_B = 64 :=
sorry

end NUMINAMATH_CALUDE_sum_of_medians_is_64_l3961_396136


namespace NUMINAMATH_CALUDE_point_on_parametric_line_l3961_396199

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a point P(2,4) on a line defined by x = 1 + t and y = 3 - at, 
    where t is a parameter, the value of a must be -1 -/
theorem point_on_parametric_line (P : Point) (l : ParametricLine) (a : ℝ) :
  P.x = 2 ∧ P.y = 4 ∧
  (∃ t : ℝ, l.x t = 1 + t ∧ l.y t = 3 - a * t) ∧
  (∃ t : ℝ, P.x = l.x t ∧ P.y = l.y t) →
  a = -1 := by
  sorry

#check point_on_parametric_line

end NUMINAMATH_CALUDE_point_on_parametric_line_l3961_396199


namespace NUMINAMATH_CALUDE_expression_evaluation_l3961_396159

theorem expression_evaluation (x y z : ℝ) (hx : x = -6) (hy : y = -3) (hz : z = 1/2) :
  4 * z * (x - y)^2 - (x * z) / y + 3 * Real.sin (y * z) = 17 + 3 * Real.sin (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3961_396159


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l3961_396122

theorem power_ranger_stickers (box1 box2 total : ℕ) : 
  box1 = 23 →
  box2 = box1 + 12 →
  total = box1 + box2 →
  total = 58 := by sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l3961_396122


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3961_396166

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3961_396166


namespace NUMINAMATH_CALUDE_competition_results_l3961_396118

def scores_8_1 : List ℕ := [70, 70, 75, 75, 75, 75, 80, 80, 80, 85, 90, 90, 90, 90, 90, 95, 95, 95, 100, 100]
def scores_8_2 : List ℕ := [75, 75, 80, 80, 80, 80, 80, 85, 85, 85, 85, 85, 85, 85, 85, 90, 90, 95, 95, 100]

def median (l : List ℕ) : ℚ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (median scores_8_1 = 87.5) ∧
  (mean scores_8_2 = 85) ∧
  (variance scores_8_2 < variance scores_8_1) := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l3961_396118


namespace NUMINAMATH_CALUDE_josh_book_purchase_l3961_396152

/-- The number of books Josh bought -/
def num_books : ℕ := sorry

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def film_cost : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 4

/-- The cost of each CD in dollars -/
def cd_cost : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

theorem josh_book_purchase : 
  num_books * book_cost + num_films * film_cost + num_cds * cd_cost = total_spent ∧ 
  num_books = 4 := by sorry

end NUMINAMATH_CALUDE_josh_book_purchase_l3961_396152


namespace NUMINAMATH_CALUDE_cakes_dinner_today_l3961_396126

def cakes_lunch_today : ℕ := 5
def cakes_yesterday : ℕ := 3
def total_cakes : ℕ := 14

theorem cakes_dinner_today : ∃ x : ℕ, x = total_cakes - cakes_lunch_today - cakes_yesterday ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_dinner_today_l3961_396126


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3961_396141

theorem largest_angle_in_special_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_ratio : (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 4/5 ∧
             (Real.sin C + Real.sin A) / (Real.sin A + Real.sin B) = 5/6) :
  max A (max B C) = 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3961_396141


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3961_396137

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a 5 (3 * n) / arithmetic_sum a 5 n = c) →
  a = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3961_396137


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3961_396198

-- Define the set M
def M (k : ℝ) : Set ℝ := {x : ℝ | |x| > k}

-- Define the statement
theorem sufficient_not_necessary (k : ℝ) :
  (k = 2 → 2 ∈ (M k)ᶜ) ∧ (∃ k', k' ≠ 2 ∧ 2 ∈ (M k')ᶜ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3961_396198


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3961_396109

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 7x - 15
- The third term is 4x + 2
- The nth term is 4018

Prove that n = 803
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (7 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (7 * x - 15 : ℚ) = (4 * x + 2 : ℚ) - (7 * x - 15 : ℚ) ∧
  (8 : ℚ) + (n - 1 : ℚ) * 5 = 4018 →
  n = 803 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3961_396109


namespace NUMINAMATH_CALUDE_kilometer_to_leaps_l3961_396161

/-- Conversion between units of length -/
theorem kilometer_to_leaps 
  (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (h1 : p * (1 : ℝ) = q * (1 : ℝ))  -- p strides = q leaps
  (h2 : r * (1 : ℝ) = s * (1 : ℝ))  -- r bounds = s strides
  (h3 : t * (1 : ℝ) = u * (1 : ℝ))  -- t bounds = u kilometers
  : (1 : ℝ) * (1 : ℝ) = (t * s * q) / (u * r * p) * (1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_kilometer_to_leaps_l3961_396161


namespace NUMINAMATH_CALUDE_pencil_price_decrease_l3961_396189

/-- The original price for a set of pencils -/
def original_set_price : ℚ := 4

/-- The number of pencils in the original set -/
def original_set_count : ℕ := 3

/-- The promotional price for a set of pencils -/
def promo_set_price : ℚ := 3

/-- The number of pencils in the promotional set -/
def promo_set_count : ℕ := 4

/-- Calculate the price per pencil given the set price and count -/
def price_per_pencil (set_price : ℚ) (set_count : ℕ) : ℚ :=
  set_price / set_count

/-- Calculate the percent decrease between two prices -/
def percent_decrease (old_price : ℚ) (new_price : ℚ) : ℚ :=
  (old_price - new_price) / old_price * 100

/-- The theorem stating the percent decrease in pencil price -/
theorem pencil_price_decrease :
  let original_price := price_per_pencil original_set_price original_set_count
  let promo_price := price_per_pencil promo_set_price promo_set_count
  let decrease := percent_decrease original_price promo_price
  ∃ (ε : ℚ), abs (decrease - 43.6) < ε ∧ ε < 0.1 :=
sorry

end NUMINAMATH_CALUDE_pencil_price_decrease_l3961_396189


namespace NUMINAMATH_CALUDE_conference_games_l3961_396102

theorem conference_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (h1 : total_teams = 12)
  (h2 : divisions = 3)
  (h3 : teams_per_division = 4)
  (h4 : total_teams = divisions * teams_per_division) :
  (total_teams * (3 * (teams_per_division - 1) + 2 * (total_teams - teams_per_division))) / 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l3961_396102


namespace NUMINAMATH_CALUDE_equation_solution_l3961_396177

theorem equation_solution (x : ℝ) : 
  (x^2 - 2*x - 8 = -(x + 4)*(x - 1)) ↔ (x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3961_396177


namespace NUMINAMATH_CALUDE_perimeter_of_specific_arrangement_l3961_396171

/-- Represents the arrangement of unit squares in the figure -/
def SquareArrangement : Type := Unit  -- Placeholder for the specific arrangement

/-- Calculates the perimeter of the given square arrangement -/
def perimeter (arrangement : SquareArrangement) : ℕ :=
  26  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the perimeter of the given square arrangement is 26 -/
theorem perimeter_of_specific_arrangement :
  ∀ (arrangement : SquareArrangement), perimeter arrangement = 26 := by
  sorry

#check perimeter_of_specific_arrangement

end NUMINAMATH_CALUDE_perimeter_of_specific_arrangement_l3961_396171


namespace NUMINAMATH_CALUDE_nancy_coffee_spend_l3961_396101

/-- The amount Nancy spends on coffee over a given number of days -/
def coffee_expenditure (days : ℕ) (espresso_price iced_price : ℚ) : ℚ :=
  days * (espresso_price + iced_price)

/-- Theorem: Nancy spends $110.00 on coffee over 20 days -/
theorem nancy_coffee_spend :
  coffee_expenditure 20 3 2.5 = 110 := by
sorry

end NUMINAMATH_CALUDE_nancy_coffee_spend_l3961_396101


namespace NUMINAMATH_CALUDE_s13_is_constant_l3961_396178

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given S_4 + a_25 = 5, S_13 is constant -/
theorem s13_is_constant (seq : ArithmeticSequence) 
    (h : seq.S 4 + seq.a 25 = 5) : 
  ∃ c : ℝ, seq.S 13 = c := by
  sorry

end NUMINAMATH_CALUDE_s13_is_constant_l3961_396178


namespace NUMINAMATH_CALUDE_christmas_day_is_saturday_l3961_396131

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek := sorry

/-- Function to add days to a given date -/
def addDays (date : Date) (days : Nat) : Date := sorry

theorem christmas_day_is_saturday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 25)
  (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
  dayOfWeek (Date.mk 12 25) = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_christmas_day_is_saturday_l3961_396131


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l3961_396195

/-- The number of ways to arrange 2 boys and 3 girls in a row with specific conditions -/
def arrangementCount : ℕ :=
  let totalPeople : ℕ := 5
  let boys : ℕ := 2
  let girls : ℕ := 3
  let boyA : ℕ := 1
  48

/-- Theorem stating that the number of arrangements satisfying the given conditions is 48 -/
theorem correct_arrangement_count :
  arrangementCount = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l3961_396195


namespace NUMINAMATH_CALUDE_polygon_25_diagonals_l3961_396128

/-- The number of diagonals in a convex polygon with n sides,
    where each vertex connects only to vertices at least k places apart. -/
def diagonals (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - (2*k + 1))) / 2

/-- Theorem: A convex 25-sided polygon where each vertex connects only to
    vertices at least 2 places apart in sequence has 250 diagonals. -/
theorem polygon_25_diagonals :
  diagonals 25 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_polygon_25_diagonals_l3961_396128


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l3961_396145

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Theorem: In a geometric sequence where a_7 = 1/4 and a_3 * a_5 = 4(a_4 - 1), a_2 = 8 -/
theorem geometric_sequence_a2 (a : ℕ → ℚ) 
    (h_geom : GeometricSequence a) 
    (h_a7 : a 7 = 1/4) 
    (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l3961_396145


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l3961_396105

theorem smallest_integer_with_remainder_one : ∃ k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  k % 5 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → k ≤ m) ∧
  k = 1366 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l3961_396105


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3961_396181

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3961_396181


namespace NUMINAMATH_CALUDE_sqrt_five_approximation_l3961_396116

theorem sqrt_five_approximation :
  (2^2 < 5 ∧ 5 < 3^2) →
  (2.2^2 < 5 ∧ 5 < 2.3^2) →
  (2.23^2 < 5 ∧ 5 < 2.24^2) →
  (2.236^2 < 5 ∧ 5 < 2.237^2) →
  ∃ (x : ℝ), x^2 = 5 ∧ |x - 2.24| < 0.005 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_five_approximation_l3961_396116


namespace NUMINAMATH_CALUDE_garden_area_l3961_396196

theorem garden_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l3961_396196


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3961_396135

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 2 * (b + c) →
  b = 5 * c →
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3961_396135


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3961_396162

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 18)
  (h_fourth : a 4 = 3) :
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3961_396162


namespace NUMINAMATH_CALUDE_weight_difference_l3961_396117

/-- Given Heather's and Emily's weights, prove the weight difference between them. -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h_heather : heather_weight = 87)
  (h_emily : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3961_396117


namespace NUMINAMATH_CALUDE_bella_roses_from_parents_l3961_396107

/-- The number of dancer friends Bella has -/
def num_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := 44

/-- The number of roses Bella received from her parents -/
def roses_from_parents : ℕ := total_roses - (num_friends * roses_per_friend)

theorem bella_roses_from_parents :
  roses_from_parents = 24 :=
sorry

end NUMINAMATH_CALUDE_bella_roses_from_parents_l3961_396107


namespace NUMINAMATH_CALUDE_race_result_l3961_396175

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- The race setup -/
def Race (sasha lesha kolya : Runner) : Prop :=
  sasha.speed > 0 ∧ lesha.speed > 0 ∧ kolya.speed > 0 ∧
  sasha.speed ≠ lesha.speed ∧ sasha.speed ≠ kolya.speed ∧ lesha.speed ≠ kolya.speed ∧
  sasha.distance = 100 ∧
  lesha.distance = 90 ∧
  kolya.distance = 81

theorem race_result (sasha lesha kolya : Runner) 
  (h : Race sasha lesha kolya) : 
  sasha.distance - kolya.distance = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l3961_396175


namespace NUMINAMATH_CALUDE_bill_division_l3961_396154

/-- The total bill amount when three people divide it evenly -/
def total_bill (individual_payment : ℕ) : ℕ := 3 * individual_payment

/-- Theorem: If three people divide a bill evenly and each pays $33, then the total bill is $99 -/
theorem bill_division (individual_payment : ℕ) 
  (h : individual_payment = 33) : 
  total_bill individual_payment = 99 := by
  sorry

end NUMINAMATH_CALUDE_bill_division_l3961_396154


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3961_396129

theorem perfect_square_binomial (x : ℝ) (k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 + 24*x + k = (x + b)^2) ↔ k = 144 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3961_396129


namespace NUMINAMATH_CALUDE_first_degree_function_determination_l3961_396183

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

-- State the theorem
theorem first_degree_function_determination
  (f : ℝ → ℝ)
  (h1 : FirstDegreeFunction f)
  (h2 : 2 * f 2 - 3 * f 1 = 5)
  (h3 : 2 * f 0 - f (-1) = 1) :
  ∀ x, f x = 3 * x - 2 :=
sorry

end NUMINAMATH_CALUDE_first_degree_function_determination_l3961_396183


namespace NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3961_396185

theorem difference_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁ - r₂ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3961_396185


namespace NUMINAMATH_CALUDE_root_in_interval_l3961_396173

def f (x : ℝ) := 3*x^2 + 3*x - 8

theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (f 1 < 0) →
  (f 1.5 > 0) →
  (f 1.25 < 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3961_396173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3961_396108

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 3 + a 13 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3961_396108
