import Mathlib

namespace symmetry_coordinates_l3042_304272

/-- Given two points are symmetric with respect to the origin, 
    their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetry_coordinates :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
  sorry

end symmetry_coordinates_l3042_304272


namespace book_reading_rate_l3042_304289

/-- Calculates the number of pages read per day given the total number of pages and days spent reading. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℕ :=
  total_pages / total_days

/-- Theorem stating that reading 12518 pages over 569 days results in 22 pages per day. -/
theorem book_reading_rate :
  pages_per_day 12518 569 = 22 := by
  sorry

end book_reading_rate_l3042_304289


namespace digitCubeSequence_1729th_term_l3042_304239

/-- Sum of cubes of digits of a natural number -/
def sumCubesOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the sum of cubes of digits -/
def digitCubeSequence : ℕ → ℕ
  | 0 => 1729
  | n + 1 => sumCubesOfDigits (digitCubeSequence n)

/-- The 1729th term of the digit cube sequence is 370 -/
theorem digitCubeSequence_1729th_term :
  digitCubeSequence 1728 = 370 := by sorry

end digitCubeSequence_1729th_term_l3042_304239


namespace square_area_error_l3042_304221

def error_in_area (excess_error : Real) (deficit_error : Real) : Real :=
  let correct_factor := (1 + excess_error) * (1 - deficit_error)
  (1 - correct_factor) * 100

theorem square_area_error :
  error_in_area 0.03 0.04 = 1.12 := by
  sorry

end square_area_error_l3042_304221


namespace rhombus_area_l3042_304203

/-- A rhombus with perimeter 20cm and diagonals in ratio 4:3 has an area of 24cm². -/
theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 →  -- diagonals are positive
  d₁ / d₂ = 4 / 3 →  -- ratio of diagonals is 4:3
  (d₁^2 + d₂^2) / 2 = 25 →  -- perimeter is 20cm (side length is 5cm)
  d₁ * d₂ / 2 = 24 := by sorry

end rhombus_area_l3042_304203


namespace hyperbola_eccentricity_l3042_304244

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ P : ℝ × ℝ, (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
       (|P.1 - (-c)| + |P.1 - c| = 3*b) ∧
       (|P.1 - (-c)| * |P.1 - c| = 9/4 * a * b))
  (h4 : c^2 = a^2 + b^2) : 
  (c / a : ℝ) = 5/3 := by
  sorry

end hyperbola_eccentricity_l3042_304244


namespace projected_strings_intersection_criterion_l3042_304294

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral2D where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the ratio of two line segments -/
def segmentRatio (P Q R : Point2D) : ℝ := sorry

/-- Determines if two projected strings intersect in 3D space -/
def stringsIntersect (quad : Quadrilateral2D) (P Q R S : Point2D) : Prop :=
  let ratio1 := segmentRatio quad.A P quad.B
  let ratio2 := segmentRatio quad.B Q quad.C
  let ratio3 := segmentRatio quad.C R quad.D
  let ratio4 := segmentRatio quad.D S quad.A
  ratio1 * ratio2 * ratio3 * ratio4 = 1

/-- Theorem: Projected strings intersect in 3D iff their segment ratio product is 1 -/
theorem projected_strings_intersection_criterion 
  (quad : Quadrilateral2D) (P Q R S : Point2D) : 
  stringsIntersect quad P Q R S ↔ 
  segmentRatio quad.A P quad.B * 
  segmentRatio quad.B Q quad.C * 
  segmentRatio quad.C R quad.D * 
  segmentRatio quad.D S quad.A = 1 := by sorry

#check projected_strings_intersection_criterion

end projected_strings_intersection_criterion_l3042_304294


namespace complete_square_with_integer_l3042_304230

theorem complete_square_with_integer : 
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 8*x + 20 = (x + b)^2 + k := by
  sorry

end complete_square_with_integer_l3042_304230


namespace polynomial_equality_l3042_304213

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ m : ℝ, P m - (4 * m^3 + m^2 + 5) = 3 * m^4 - 4 * m^3 - m^2 + m - 8) →
  (∀ m : ℝ, P m = 3 * m^4 + m - 3) :=
by
  sorry

end polynomial_equality_l3042_304213


namespace correct_operation_l3042_304296

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end correct_operation_l3042_304296


namespace negative_two_cubed_equality_l3042_304243

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end negative_two_cubed_equality_l3042_304243


namespace problem_statement_l3042_304209

theorem problem_statement : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  sorry

end problem_statement_l3042_304209


namespace daily_earnings_a_and_c_l3042_304211

/-- Given three workers a, b, and c, with their daily earnings, prove that a and c together earn $400 per day. -/
theorem daily_earnings_a_and_c (a b c : ℕ) : 
  a + b + c = 600 →  -- Total earnings of a, b, and c
  b + c = 300 →      -- Combined earnings of b and c
  c = 100 →          -- Earnings of c
  a + c = 400 :=     -- Combined earnings of a and c
by
  sorry

end daily_earnings_a_and_c_l3042_304211


namespace card_selection_ways_l3042_304256

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of cards per suit in a standard deck
def cards_per_suit : ℕ := 13

-- Define the total number of cards in a standard deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Define the number of cards to choose
def cards_to_choose : ℕ := 4

-- Define the number of cards to keep after discarding
def cards_to_keep : ℕ := 3

-- Theorem statement
theorem card_selection_ways :
  (num_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) * cards_to_choose = 114244 := by
  sorry

end card_selection_ways_l3042_304256


namespace line_perp_parallel_implies_planes_perp_l3042_304283

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_perp_parallel_implies_planes_perp 
  (m : Line3D) (α β : Plane3D) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end line_perp_parallel_implies_planes_perp_l3042_304283


namespace congruence_solution_unique_solution_in_range_l3042_304246

theorem congruence_solution (m : ℤ) : 
  (13 * m ≡ 9 [ZMOD 47]) ↔ (m ≡ 26 [ZMOD 47]) :=
by sorry

theorem unique_solution_in_range : 
  ∃! x : ℕ, x < 47 ∧ (13 * x ≡ 9 [ZMOD 47]) :=
by sorry

end congruence_solution_unique_solution_in_range_l3042_304246


namespace perpendicular_line_equation_l3042_304275

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 1 ∧
  point.x = 2 ∧ point.y = -1 →
  ∃ (l : Line), l.perpendicular given_line ∧ point.liesOn l ∧
  l.a = 2 ∧ l.b = 1 ∧ l.c = -3 := by
  sorry

end perpendicular_line_equation_l3042_304275


namespace almond_distribution_l3042_304206

/-- The number of almonds Elaine received -/
def elaine_almonds : ℕ := 12

/-- The number of almonds Daniel received -/
def daniel_almonds : ℕ := elaine_almonds - 8

theorem almond_distribution :
  (elaine_almonds = daniel_almonds + 8) ∧
  (daniel_almonds = elaine_almonds / 3) →
  elaine_almonds = 12 := by
  sorry

end almond_distribution_l3042_304206


namespace hat_and_glasses_probability_l3042_304269

theorem hat_and_glasses_probability
  (total_hats : ℕ)
  (total_glasses : ℕ)
  (prob_hat_given_glasses : ℚ)
  (h1 : total_hats = 60)
  (h2 : total_glasses = 40)
  (h3 : prob_hat_given_glasses = 1 / 4) :
  (total_glasses : ℚ) * prob_hat_given_glasses / total_hats = 1 / 6 :=
by sorry

end hat_and_glasses_probability_l3042_304269


namespace sum_of_p_and_q_l3042_304273

theorem sum_of_p_and_q (p q : ℤ) : 
  p > 1 → 
  q > 1 → 
  (2 * q - 1) % p = 0 → 
  (2 * p - 1) % q = 0 → 
  p + q = 8 := by
  sorry

end sum_of_p_and_q_l3042_304273


namespace point_c_coordinates_l3042_304245

/-- Given points A and B in ℝ³, if vector AC is half of vector AB, then C has specific coordinates -/
theorem point_c_coordinates (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 2, 7) → 
  B = (-2, 4, 3) → 
  C - A = (1 / 2 : ℝ) • (B - A) → 
  C = (0, 3, 5) := by
  sorry

end point_c_coordinates_l3042_304245


namespace apple_box_weight_l3042_304237

theorem apple_box_weight (n : ℕ) (w : ℝ) (h1 : n = 5) (h2 : w > 30) 
  (h3 : n * (w - 30) = 2 * w) : w = 50 := by
  sorry

end apple_box_weight_l3042_304237


namespace constant_sum_l3042_304263

/-- The number of distinct roots of a rational function -/
noncomputable def distinctRoots (num : ℝ → ℝ) (denom : ℝ → ℝ) : ℕ := sorry

theorem constant_sum (a b : ℝ) : 
  distinctRoots (λ x => (x+a)*(x+b)*(x+10)) (λ x => (x+4)^2) = 3 →
  distinctRoots (λ x => (x+2*a)*(x+4)*(x+5)) (λ x => (x+b)*(x+10)) = 1 →
  100*a + b = 205 := by sorry

end constant_sum_l3042_304263


namespace project_popularity_order_l3042_304228

def park_renovation : ℚ := 9 / 24
def new_library : ℚ := 10 / 30
def street_lighting : ℚ := 7 / 21
def community_garden : ℚ := 8 / 24

theorem project_popularity_order :
  park_renovation > community_garden ∧
  community_garden = new_library ∧
  new_library = street_lighting ∧
  park_renovation > new_library :=
by sorry

end project_popularity_order_l3042_304228


namespace polygon_diagonals_l3042_304298

/-- A polygon with interior angle sum of 1800° has 9 diagonals from any vertex -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end polygon_diagonals_l3042_304298


namespace jelly_cost_for_sandwiches_l3042_304253

/-- The cost of jelly for N sandwiches -/
def jelly_cost (N B J : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of peanut butter and jelly for N sandwiches -/
def total_cost (N B J : ℕ+) : ℚ :=
  (N * (3 * B + 7 * J) : ℚ) / 100

theorem jelly_cost_for_sandwiches
  (N B J : ℕ+)
  (h1 : total_cost N B J = 252 / 100)
  (h2 : N > 1) :
  jelly_cost N B J = 168 / 100 :=
by sorry

end jelly_cost_for_sandwiches_l3042_304253


namespace spherical_coord_transformation_l3042_304217

/-- Given a point with rectangular coordinates (a, b, c) and 
    spherical coordinates (3, 3π/4, π/6), prove that the point 
    with rectangular coordinates (a, -b, c) has spherical 
    coordinates (3, 7π/4, π/6) -/
theorem spherical_coord_transformation 
  (a b c : ℝ) 
  (h1 : a = 3 * Real.sin (π/6) * Real.cos (3*π/4))
  (h2 : b = 3 * Real.sin (π/6) * Real.sin (3*π/4))
  (h3 : c = 3 * Real.cos (π/6)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 3 ∧ 
    θ = 7*π/4 ∧ 
    φ = π/6 ∧
    a = ρ * Real.sin φ * Real.cos θ ∧
    -b = ρ * Real.sin φ * Real.sin θ ∧
    c = ρ * Real.cos φ :=
by sorry

end spherical_coord_transformation_l3042_304217


namespace egg_collection_theorem_l3042_304257

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_theorem : total_eggs = 26 := by
  sorry

end egg_collection_theorem_l3042_304257


namespace bobby_jump_improvement_l3042_304236

/-- Bobby's jump rope ability as a child and adult -/
def bobby_jumps : ℕ × ℕ := (30, 60)

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
def jump_difference : ℕ := bobby_jumps.2 - bobby_jumps.1

theorem bobby_jump_improvement : jump_difference = 30 := by
  sorry

end bobby_jump_improvement_l3042_304236


namespace badge_exchange_l3042_304288

theorem badge_exchange (vasya_initial : ℕ) (tolya_initial : ℕ) : 
  (vasya_initial = tolya_initial + 5) →
  (vasya_initial - (24 * vasya_initial) / 100 + (20 * tolya_initial) / 100 = 
   tolya_initial - (20 * tolya_initial) / 100 + (24 * vasya_initial) / 100 - 1) →
  (vasya_initial = 50 ∧ tolya_initial = 45) :=
by sorry

#check badge_exchange

end badge_exchange_l3042_304288


namespace intersection_A_complement_B_l3042_304271

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | Real.log (x - 2) < 1}

def B : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x < 12} := by sorry

end intersection_A_complement_B_l3042_304271


namespace min_teams_in_championship_l3042_304216

/-- Represents a soccer championship with the given rules --/
structure SoccerChampionship where
  numTeams : ℕ
  /-- Each team plays one match against every other team --/
  totalMatches : ℕ := numTeams * (numTeams - 1) / 2
  /-- Winning team gets 2 points, tie gives 1 point to each team, losing team gets 0 points --/
  pointSystem : List ℕ := [2, 1, 0]

/-- Represents the points of a team --/
structure TeamPoints where
  wins : ℕ
  draws : ℕ
  points : ℕ := 2 * wins + draws

/-- The condition that one team has the most points but fewer wins than any other team --/
def hasUniqueLeader (c : SoccerChampionship) (leader : TeamPoints) (others : List TeamPoints) : Prop :=
  ∀ team ∈ others, leader.points > team.points ∧ leader.wins < team.wins

/-- The main theorem stating the minimum number of teams --/
theorem min_teams_in_championship : 
  ∀ c : SoccerChampionship, 
  ∀ leader : TeamPoints,
  ∀ others : List TeamPoints,
  hasUniqueLeader c leader others →
  c.numTeams ≥ 6 :=
sorry

end min_teams_in_championship_l3042_304216


namespace simplify_expression_l3042_304251

theorem simplify_expression (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 2 * a * b - a^2 ≠ 0) :
  (a^2 - 2*a*b + b^2) / (a*b) - (2*a*b - b^2) / (2*a*b - a^2) = (a^2 - 2*a*b + 2*b^2) / (a*b) := by
  sorry

end simplify_expression_l3042_304251


namespace valid_words_count_l3042_304274

def alphabet_size : ℕ := 15
def max_word_length : ℕ := 5

def total_words (n : ℕ) : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

def words_without_letter (n : ℕ) : ℕ := 
  ((alphabet_size - 1) ^ 1) + ((alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 3) + ((alphabet_size - 1) ^ 4) + 
  ((alphabet_size - 1) ^ 5)

def words_without_two_letters (n : ℕ) : ℕ := 
  ((alphabet_size - 2) ^ 1) + ((alphabet_size - 2) ^ 2) + 
  ((alphabet_size - 2) ^ 3) + ((alphabet_size - 2) ^ 4) + 
  ((alphabet_size - 2) ^ 5)

theorem valid_words_count : 
  total_words alphabet_size - 2 * words_without_letter alphabet_size + 
  words_without_two_letters alphabet_size = 62460 := by
  sorry

end valid_words_count_l3042_304274


namespace f_has_two_zeros_l3042_304290

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - 3*x

-- Theorem statement
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_zeros_l3042_304290


namespace no_perfect_square_in_range_l3042_304279

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 15 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end no_perfect_square_in_range_l3042_304279


namespace infinite_solutions_and_sum_of_exceptions_l3042_304220

theorem infinite_solutions_and_sum_of_exceptions :
  let A : ℚ := 3
  let B : ℚ := 5
  let C : ℚ := 40/3
  let f (x : ℚ) := (x + B) * (A * x + 40) / ((x + C) * (x + 5))
  (∀ x, x ≠ -C → x ≠ -5 → f x = 3) ∧
  (-5 + (-C) = -55/3) := by
  sorry

end infinite_solutions_and_sum_of_exceptions_l3042_304220


namespace precious_stone_cost_l3042_304223

theorem precious_stone_cost (num_stones : ℕ) (total_amount : ℕ) (h1 : num_stones = 8) (h2 : total_amount = 14280) :
  total_amount / num_stones = 1785 := by
sorry

end precious_stone_cost_l3042_304223


namespace money_left_over_l3042_304231

-- Define the given conditions
def video_game_cost : ℝ := 60
def discount_rate : ℝ := 0.15
def candy_cost : ℝ := 5
def sales_tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 3
def babysitting_rate : ℝ := 8
def hours_worked : ℝ := 9

-- Define the theorem
theorem money_left_over :
  let discounted_price := video_game_cost * (1 - discount_rate)
  let video_game_total := discounted_price + shipping_fee
  let video_game_with_tax := video_game_total * (1 + sales_tax_rate)
  let candy_with_tax := candy_cost * (1 + sales_tax_rate)
  let total_cost := video_game_with_tax + candy_with_tax
  let earnings := babysitting_rate * hours_worked
  earnings - total_cost = 7.10 := by
  sorry

end money_left_over_l3042_304231


namespace medicine_supply_duration_l3042_304219

/-- Represents the duration of the medicine supply in months -/
def medicine_duration (pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  let days_per_pill := (days_between_doses : ℚ) / pill_fraction
  let total_days := (pills : ℚ) * days_per_pill
  let days_per_month := 30
  total_days / days_per_month

/-- The theorem stating that the given medicine supply lasts 18 months -/
theorem medicine_supply_duration :
  medicine_duration 60 (1/3) 3 = 18 := by
  sorry

end medicine_supply_duration_l3042_304219


namespace five_digit_number_probability_l3042_304267

/-- The set of digits to choose from -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- The number of digits to select -/
def num_selected : Nat := 3

/-- The length of the number to form -/
def num_length : Nat := 5

/-- The number of digits that should be used twice -/
def num_twice_used : Nat := 2

/-- The probability of forming a number with two digits each used twice -/
def probability : Rat := 3/5

theorem five_digit_number_probability :
  (Finset.card digits = 5) →
  (num_selected = 3) →
  (num_length = 5) →
  (num_twice_used = 2) →
  (probability = 3/5) :=
by sorry

end five_digit_number_probability_l3042_304267


namespace min_a_value_l3042_304284

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) → a ≥ 1/3 :=
by
  sorry

end min_a_value_l3042_304284


namespace range_of_m_l3042_304281

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ -1}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x ≤ 2*m}

theorem range_of_m (m : ℝ) :
  (A ∩ B m = ∅) → (A ∪ B m = A) → m ≤ -1 := by
  sorry

end range_of_m_l3042_304281


namespace paint_remaining_rooms_l3042_304248

def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

theorem paint_remaining_rooms 
  (total_rooms : ℕ) 
  (time_per_room : ℕ) 
  (painted_rooms : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : time_per_room = 7) 
  (h3 : painted_rooms = 2) : 
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 63 := by
sorry

end paint_remaining_rooms_l3042_304248


namespace m_range_l3042_304229

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem statement
theorem m_range (m : ℝ) : 1 ∈ A m ∧ 3 ∉ A m → 0 < m ∧ m < 2 :=
by
  sorry

end m_range_l3042_304229


namespace thursday_to_tuesday_ratio_l3042_304207

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddysCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Thursday --/
def thursdayPurchase (cards : BuddysCards) : ℕ :=
  cards.thursday - cards.wednesday

/-- The theorem stating the ratio of Thursday's purchase to Tuesday's amount --/
theorem thursday_to_tuesday_ratio (cards : BuddysCards) :
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.wednesday = cards.tuesday + 12 →
  cards.thursday = 32 →
  thursdayPurchase cards * 3 = cards.tuesday := by
  sorry

end thursday_to_tuesday_ratio_l3042_304207


namespace function_value_at_m_l3042_304264

/-- Given a function f(x) = x³ + ax + 3 where f(-m) = 1, prove that f(m) = 5 -/
theorem function_value_at_m (a m : ℝ) : 
  (fun x : ℝ ↦ x^3 + a*x + 3) (-m) = 1 → 
  (fun x : ℝ ↦ x^3 + a*x + 3) m = 5 := by
sorry

end function_value_at_m_l3042_304264


namespace A_equals_nine_l3042_304212

/-- A3 is a two-digit number -/
def A3 : ℕ := sorry

/-- A is the tens digit of A3 -/
def A : ℕ := A3 / 10

/-- The ones digit of A3 -/
def B : ℕ := A3 % 10

/-- A3 is a two-digit number -/
axiom A3_two_digit : 10 ≤ A3 ∧ A3 ≤ 99

/-- A3 - 41 = 52 -/
axiom A3_equation : A3 - 41 = 52

theorem A_equals_nine : A = 9 := by
  sorry

end A_equals_nine_l3042_304212


namespace problem_solution_l3042_304259

-- Definition of the relation (x, y) = n
def relation (x y n : ℝ) : Prop := x^n = y

theorem problem_solution :
  -- Part 1
  relation 10 1000 3 ∧
  relation (-5) 25 2 ∧
  -- Part 2
  (∀ x, relation x 16 2 → (x = 4 ∨ x = -4)) ∧
  -- Part 3
  (∀ a b, relation 4 a 2 → relation b 8 3 → relation b a 4) :=
by sorry

end problem_solution_l3042_304259


namespace quadratic_root_power_sums_l3042_304242

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    s_n denotes the sum of the n-th powers of the roots. -/
def s (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

/-- Theorem stating the relations between sums of powers of roots of a quadratic equation -/
theorem quadratic_root_power_sums 
  (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h : a ≠ 0)
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∀ n : ℕ, n ≥ 2 → a * s n x₁ x₂ + b * s (n-1) x₁ x₂ + c * s (n-2) x₁ x₂ = 0) ∧
  (a * s 2 x₁ x₂ + b * s 1 x₁ x₂ + 2 * c = 0) :=
by sorry

end quadratic_root_power_sums_l3042_304242


namespace photo_arrangements_l3042_304260

/-- Represents the number of people in the photo arrangement --/
def total_people : ℕ := 7

/-- Represents the number of students in the photo arrangement --/
def num_students : ℕ := 6

/-- Represents the position of the teacher in the row --/
def teacher_position : ℕ := 4

/-- Represents the number of positions to the left of the teacher --/
def left_positions : ℕ := 3

/-- Represents the number of positions to the right of the teacher --/
def right_positions : ℕ := 3

/-- Represents the number of positions available for Student A --/
def positions_for_A : ℕ := 5

/-- Represents the number of positions available for Student B --/
def positions_for_B : ℕ := 5

/-- Represents the number of remaining students after placing A and B --/
def remaining_students : ℕ := 4

/-- Theorem stating the number of different arrangements --/
theorem photo_arrangements :
  (positions_for_A * (positions_for_B - 1) * (remaining_students!)) * 2 = 960 := by
  sorry

end photo_arrangements_l3042_304260


namespace intersection_A_B_l3042_304222

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_A_B_l3042_304222


namespace unique_solution_diophantine_equation_l3042_304201

theorem unique_solution_diophantine_equation :
  ∀ a b c d : ℕ+,
    4^(a:ℕ) * 5^(b:ℕ) - 3^(c:ℕ) * 11^(d:ℕ) = 1 →
    a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 1 :=
by sorry

end unique_solution_diophantine_equation_l3042_304201


namespace units_digit_period_four_units_digit_2_power_2012_l3042_304247

/-- The units digit of 2^n -/
def unitsDigit (n : ℕ) : ℕ := 2^n % 10

/-- The pattern of units digits for powers of 2 repeats every 4 steps -/
theorem units_digit_period_four (n : ℕ) : 
  unitsDigit n = unitsDigit (n + 4) :=
sorry

/-- The units digit of 2^2012 is 6 -/
theorem units_digit_2_power_2012 : unitsDigit 2012 = 6 :=
sorry

end units_digit_period_four_units_digit_2_power_2012_l3042_304247


namespace common_roots_product_l3042_304249

/-- Given two polynomials that share exactly two roots, prove that the product of these common roots is 1/3 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s t u : ℝ),
    (∀ x : ℝ, x^4 - 3*x^3 + C*x + 24 = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    (∀ x : ℝ, x^4 - D*x^3 + 4*x^2 + 72 = (x - p)*(x - q)*(x - t)*(x - u)) ∧
    p ≠ q ∧ 
    p * q = 1/3 :=
by sorry

end common_roots_product_l3042_304249


namespace prism_pyramid_height_relation_l3042_304261

/-- Given an equilateral triangle with side length a, prove that if a prism and a pyramid
    are constructed on this triangle with height m, and the lateral surface area of the prism
    equals the lateral surface area of the pyramid, then m = a/6 -/
theorem prism_pyramid_height_relation (a : ℝ) (m : ℝ) (h_pos : a > 0) : 
  (3 * a * m = (3 * a / 2) * Real.sqrt (m^2 + a^2 / 12)) → m = a / 6 := by
  sorry

end prism_pyramid_height_relation_l3042_304261


namespace points_on_line_l3042_304224

/-- A line in the 2D plane defined by the equation 7x + 2y = 41 -/
def line (x y : ℝ) : Prop := 7 * x + 2 * y = 41

/-- Point A with coordinates (5, 3) -/
def point_A : ℝ × ℝ := (5, 3)

/-- Point B with coordinates (-5, 38) -/
def point_B : ℝ × ℝ := (-5, 38)

/-- Theorem stating that points A and B lie on the given line -/
theorem points_on_line :
  line point_A.1 point_A.2 ∧ line point_B.1 point_B.2 := by
  sorry

end points_on_line_l3042_304224


namespace maximize_product_l3042_304280

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^3 * y^4 ≤ 30^3 * 20^4 ∧
  (x^3 * y^4 = 30^3 * 20^4 ↔ x = 30 ∧ y = 20) :=
by sorry

end maximize_product_l3042_304280


namespace canteen_distance_l3042_304255

theorem canteen_distance (girls_camp_distance boys_camp_distance : ℝ) 
  (h1 : girls_camp_distance = 600)
  (h2 : boys_camp_distance = 800) :
  let hypotenuse := Real.sqrt (girls_camp_distance ^ 2 + boys_camp_distance ^ 2)
  let canteen_distance := Real.sqrt ((girls_camp_distance ^ 2 + (hypotenuse / 2) ^ 2))
  ⌊canteen_distance⌋ = 781 := by
  sorry

end canteen_distance_l3042_304255


namespace square_split_into_pentagons_or_hexagons_l3042_304285

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- The number of sides of a polygon -/
def Polygon.sides (p : Polygon) : ℕ := p.vertices.length

/-- A concave polygon -/
def ConcavePolygon (p : Polygon) : Prop := sorry

/-- The area of a polygon -/
def Polygon.area (p : Polygon) : ℝ := sorry

/-- A square with side length 1 -/
def UnitSquare : Polygon := sorry

/-- Two polygons are equal in area -/
def EqualArea (p1 p2 : Polygon) : Prop := p1.area = p2.area

/-- A polygon is contained within another polygon -/
def ContainedIn (p1 p2 : Polygon) : Prop := sorry

/-- The union of two polygons -/
def PolygonUnion (p1 p2 : Polygon) : Polygon := sorry

theorem square_split_into_pentagons_or_hexagons :
  ∃ (p1 p2 : Polygon),
    (p1.sides = 5 ∧ p2.sides = 5 ∨ p1.sides = 6 ∧ p2.sides = 6) ∧
    ConcavePolygon p1 ∧
    ConcavePolygon p2 ∧
    EqualArea p1 p2 ∧
    ContainedIn p1 UnitSquare ∧
    ContainedIn p2 UnitSquare ∧
    PolygonUnion p1 p2 = UnitSquare :=
sorry

end square_split_into_pentagons_or_hexagons_l3042_304285


namespace unique_perpendicular_line_l3042_304265

-- Define the concept of a plane in 3D space
class Plane :=
  (normal : ℝ → ℝ → ℝ → ℝ)

-- Define the concept of a line in 3D space
class Line :=
  (direction : ℝ → ℝ → ℝ)

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define what it means for a point to be outside a plane
def PointOutsidePlane (p : Point) (plane : Plane) : Prop := sorry

-- Define what it means for a line to pass through a point
def LinePassesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- Define perpendicularity between a line and a plane
def LinePerpendicular (l : Line) (plane : Plane) : Prop := sorry

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (p : Point) (h : PointOutsidePlane p plane) :
  ∃! l : Line, LinePassesThroughPoint l p ∧ LinePerpendicular l plane :=
sorry

end unique_perpendicular_line_l3042_304265


namespace rectangle_parallelogram_relationship_l3042_304278

-- Define the types
def Parallelogram : Type := sorry
def Rectangle : Type := sorry

-- Define the relationship between Rectangle and Parallelogram
axiom rectangle_is_parallelogram : Rectangle → Parallelogram

-- State the theorem
theorem rectangle_parallelogram_relationship :
  (∀ r : Rectangle, ∃ p : Parallelogram, p = rectangle_is_parallelogram r) ∧
  ¬(∀ p : Parallelogram, ∃ r : Rectangle, p = rectangle_is_parallelogram r) :=
sorry

end rectangle_parallelogram_relationship_l3042_304278


namespace unique_zero_point_condition_l3042_304233

/-- The function f(x) = ax³ - 3x² + 2 has only one zero point if and only if a ∈ (-∞, -√2) ∪ (√2, +∞) -/
theorem unique_zero_point_condition (a : ℝ) :
  (∃! x, a * x^3 - 3 * x^2 + 2 = 0) ↔ a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by
  sorry

end unique_zero_point_condition_l3042_304233


namespace line_intersects_circle_l3042_304205

theorem line_intersects_circle (m : ℝ) (h_m : 0 < m ∧ m < 4/3) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ + m^2 - m = 0) ∧ 
  (x₂^2 + m*x₂ + m^2 - m = 0) ∧
  ∃ (x y : ℝ), 
    (m*x + y + m^2 - m = 0) ∧ 
    ((x - 1)^2 + (y + 1)^2 = 1) := by
  sorry

end line_intersects_circle_l3042_304205


namespace hyperbola_equation_from_parameters_l3042_304270

/-- Represents a hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  focal_width : ℝ
  eccentricity : ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with given parameters has the specified equation -/
theorem hyperbola_equation_from_parameters (h : Hyperbola) 
  (hw : h.focal_width = 8) 
  (he : h.eccentricity = 2) : 
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end hyperbola_equation_from_parameters_l3042_304270


namespace quadratic_roots_in_fourth_quadrant_l3042_304295

/-- A point in the fourth quadrant -/
structure FourthQuadrantPoint where
  x : ℝ
  y : ℝ
  x_pos : 0 < x
  y_neg : y < 0

/-- Quadratic equation coefficients -/
structure QuadraticCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (q : QuadraticCoeffs) : Prop :=
  0 < q.b ^ 2 - 4 * q.a * q.c

theorem quadratic_roots_in_fourth_quadrant 
  (p : FourthQuadrantPoint) (q : QuadraticCoeffs) 
  (h : p.x = q.a ∧ p.y = q.c) : 
  has_two_distinct_real_roots q := by
  sorry

end quadratic_roots_in_fourth_quadrant_l3042_304295


namespace roots_of_quadratic_equation_l3042_304282

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_equation_l3042_304282


namespace carol_final_score_is_negative_nineteen_l3042_304204

-- Define the scores and multipliers for each round
def first_round_score : Int := 17
def second_round_base_score : Int := 6
def second_round_multiplier : Int := 2
def last_round_base_loss : Int := 16
def last_round_multiplier : Int := 3

-- Define Carol's final score
def carol_final_score : Int := 
  first_round_score + 
  (second_round_base_score * second_round_multiplier) - 
  (last_round_base_loss * last_round_multiplier)

-- Theorem to prove Carol's final score
theorem carol_final_score_is_negative_nineteen : 
  carol_final_score = -19 := by
  sorry

end carol_final_score_is_negative_nineteen_l3042_304204


namespace king_ducats_distribution_l3042_304208

theorem king_ducats_distribution (n : ℕ) (total_ducats : ℕ) :
  (∃ (a : ℕ),
    -- The eldest son receives 'a' ducats in the first round
    a + n = 21 ∧
    -- Total ducats in the first round
    n * a - (n - 1) * n / 2 +
    -- Total ducats in the second round
    n * (n + 1) / 2 = total_ducats) →
  n = 7 ∧ total_ducats = 105 := by
sorry

end king_ducats_distribution_l3042_304208


namespace ned_earnings_l3042_304276

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63 -/
theorem ned_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

end ned_earnings_l3042_304276


namespace locus_of_circumcenter_l3042_304258

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the product OA · OB · OC
def product (c : Circle) (t : Triangle) : ℝ := sorry

theorem locus_of_circumcenter 
  (c : Circle) 
  (t : Triangle) 
  (p : ℝ) 
  (h : product c t = p^3) :
  ∃ (P : ℝ × ℝ), 
    P = circumcenter t ∧ 
    distance c.center P = (p / (4 * c.radius^2)) * Real.sqrt (p * (p^3 - 8 * c.radius^3)) :=
sorry

end locus_of_circumcenter_l3042_304258


namespace additional_dividend_calculation_l3042_304202

/-- Calculates the additional dividend per share given expected and actual earnings -/
def additional_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let earnings_difference := actual_earnings - expected_earnings
  let additional_earnings := max earnings_difference 0
  additional_earnings / 2

/-- Proves that the additional dividend is $0.15 per share given the problem conditions -/
theorem additional_dividend_calculation :
  let expected_earnings : ℚ := 80 / 100
  let actual_earnings : ℚ := 110 / 100
  additional_dividend expected_earnings actual_earnings = 15 / 100 := by
  sorry


end additional_dividend_calculation_l3042_304202


namespace school_choir_members_l3042_304266

theorem school_choir_members :
  ∃! n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ n = 241 :=
by sorry

end school_choir_members_l3042_304266


namespace least_abaaba_six_primes_l3042_304214

def is_abaaba_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
  n = a * 100000 + b * 10000 + a * 1000 + a * 100 + b * 10 + a

def is_product_of_six_distinct_primes (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ * p₆

theorem least_abaaba_six_primes :
  (is_abaaba_form 282282 ∧ is_product_of_six_distinct_primes 282282) ∧
  (∀ n : ℕ, n < 282282 → ¬(is_abaaba_form n ∧ is_product_of_six_distinct_primes n)) :=
by sorry

end least_abaaba_six_primes_l3042_304214


namespace smallest_fraction_between_l3042_304252

theorem smallest_fraction_between (p q : ℕ) : 
  p > 0 → q > 0 → (7 : ℚ)/12 < p/q → p/q < 5/8 → 
  (∀ p' q' : ℕ, p' > 0 → q' > 0 → q' < q → (7 : ℚ)/12 < p'/q' → p'/q' < 5/8 → False) →
  q - p = 2 := by
sorry

end smallest_fraction_between_l3042_304252


namespace function_property_l3042_304254

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

theorem function_property (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → |f a x₁ - f a x₂| < 4) →
  0 < a ∧ a < 2 := by
  sorry

end function_property_l3042_304254


namespace quadratic_function_unique_coefficients_l3042_304291

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

theorem quadratic_function_unique_coefficients 
  (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 2 3, f a b x ≤ 4) 
  (h_min : ∀ x ∈ Set.Icc 2 3, f a b x ≥ 1) 
  (h_max_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 4) 
  (h_min_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 1) : 
  a = 1 ∧ b = 0 := by
sorry

end quadratic_function_unique_coefficients_l3042_304291


namespace problem_solution_l3042_304268

noncomputable section

def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 3), Real.cos (x / 3))
def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 3), Real.cos (x / 3))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def a : ℝ := 2

variable (A B C : ℝ)
variable (b c : ℝ)

axiom triangle_condition : (2 * a - b) * Real.cos C = c * Real.cos B
axiom f_of_A : f A = 3 / 2

theorem problem_solution :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ k : ℤ, ∀ x : ℝ, f ((-π/4 + 3*π/2*↑k) + x) = f ((-π/4 + 3*π/2*↑k) - x)) ∧
  c = Real.sqrt 3 := by
  sorry

end problem_solution_l3042_304268


namespace roses_problem_l3042_304293

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 21

/-- The number of roses Jessica cut from her garden -/
def cut_roses : ℕ := 28

/-- The number of roses Jessica threw away -/
def thrown_roses : ℕ := 34

/-- The number of roses currently in the vase -/
def current_roses : ℕ := 15

theorem roses_problem :
  initial_roses = 21 ∧
  thrown_roses = cut_roses + 6 ∧
  current_roses = initial_roses + cut_roses - thrown_roses :=
by sorry

end roses_problem_l3042_304293


namespace inscribed_circle_radius_l3042_304232

/-- Given a sector with a central angle of 60° and an arc length of 2π,
    its inscribed circle has a radius of 2. -/
theorem inscribed_circle_radius (θ : ℝ) (arc_length : ℝ) (R : ℝ) (r : ℝ) :
  θ = π / 3 →
  arc_length = 2 * π →
  arc_length = θ * R →
  3 * r = R →
  r = 2 :=
by sorry

end inscribed_circle_radius_l3042_304232


namespace frog_jump_probability_l3042_304241

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The grid dimensions -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 5

/-- The jump distance -/
def jumpDistance : ℕ := 2

/-- The starting point of the frog -/
def startPoint : Point := ⟨2, 3⟩

/-- Predicate to check if a point is on a horizontal edge -/
def isOnHorizontalEdge (p : Point) : Prop :=
  p.y = 0 ∨ p.y = gridHeight

/-- Predicate to check if a point is on the grid -/
def isOnGrid (p : Point) : Prop :=
  p.x ≤ gridWidth ∧ p.y ≤ gridHeight

/-- The probability of reaching a horizontal edge from a given point -/
noncomputable def probReachHorizontalEdge (p : Point) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  probReachHorizontalEdge startPoint = 3/4 :=
sorry

end frog_jump_probability_l3042_304241


namespace ln_exp_equals_id_l3042_304200

theorem ln_exp_equals_id : ∀ x : ℝ, Real.log (Real.exp x) = x := by sorry

end ln_exp_equals_id_l3042_304200


namespace braden_winnings_l3042_304277

/-- The amount of money Braden has after winning two bets, given his initial amount --/
def final_amount (initial_amount : ℕ) : ℕ :=
  initial_amount + 2 * initial_amount + 2 * initial_amount

/-- Theorem stating that Braden's final amount is $2000 given an initial amount of $400 --/
theorem braden_winnings :
  final_amount 400 = 2000 := by
  sorry

end braden_winnings_l3042_304277


namespace rainfall_second_week_l3042_304235

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 20 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 12 := by
  sorry

end rainfall_second_week_l3042_304235


namespace arithmetic_sequence_problem_l3042_304250

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ + a₈ = 10, prove that a₅ = 5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 5 = 5 := by
sorry

end arithmetic_sequence_problem_l3042_304250


namespace new_female_percentage_new_female_percentage_proof_l3042_304215

theorem new_female_percentage (initial_female_percentage : ℝ) 
                               (additional_male_hires : ℕ) 
                               (total_employees_after : ℕ) : ℝ :=
  let initial_employees := total_employees_after - additional_male_hires
  let initial_female_employees := (initial_female_percentage / 100) * initial_employees
  (initial_female_employees / total_employees_after) * 100

#check 
  @new_female_percentage 60 20 240 = 55

theorem new_female_percentage_proof :
  new_female_percentage 60 20 240 = 55 := by
  sorry

end new_female_percentage_new_female_percentage_proof_l3042_304215


namespace sphere_volume_from_box_diagonal_l3042_304292

theorem sphere_volume_from_box_diagonal (a b c : ℝ) (ha : a = 3 * Real.sqrt 2) (hb : b = 4 * Real.sqrt 2) (hc : c = 5 * Real.sqrt 2) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  (4 / 3) * Real.pi * (diagonal / 2)^3 = 500 * Real.pi / 3 := by sorry

end sphere_volume_from_box_diagonal_l3042_304292


namespace spring_migration_scientific_notation_l3042_304297

theorem spring_migration_scientific_notation :
  (260000000 : ℝ) = 2.6 * (10 ^ 8) := by
  sorry

end spring_migration_scientific_notation_l3042_304297


namespace rational_roots_imply_rational_roots_l3042_304240

theorem rational_roots_imply_rational_roots (c : ℝ) (p q : ℚ) :
  (p^2 - p + c = 0) → (q^2 - q + c = 0) →
  ∃ (r s : ℚ), r^2 + p*r - q = 0 ∧ s^2 + p*s - q = 0 := by
  sorry

end rational_roots_imply_rational_roots_l3042_304240


namespace max_quarters_sasha_l3042_304234

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Sasha has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Sasha has $4.80 in U.S. coins and three times as many nickels as quarters,
prove that the maximum number of quarters she could have is 12.
-/
theorem max_quarters_sasha : 
  ∃ (q : ℕ), q ≤ 12 ∧ 
  q * quarter_value + 3 * q * nickel_value = total_money ∧
  ∀ (n : ℕ), n * quarter_value + 3 * n * nickel_value = total_money → n ≤ q :=
by sorry

end max_quarters_sasha_l3042_304234


namespace number_of_parents_attending_l3042_304238

/-- The number of parents attending a school meeting -/
theorem number_of_parents_attending (S R B N : ℕ) : 
  S = 25 →  -- number of parents volunteering to supervise
  B = 11 →  -- number of parents volunteering for both supervising and bringing refreshments
  R = 42 →  -- number of parents volunteering to bring refreshments
  R = (3 * N) / 2 →  -- R is 1.5 times N
  S + R - B + N = 95 :=  -- total number of parents
by sorry

end number_of_parents_attending_l3042_304238


namespace bob_distance_walked_l3042_304225

theorem bob_distance_walked (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (h1 : total_distance = 80)
  (h2 : yolanda_rate = 8)
  (h3 : bob_rate = 9) : 
  ∃ t : ℝ, t * (yolanda_rate + bob_rate) = total_distance - yolanda_rate ∧ 
  bob_rate * t = 648 / 17 := by
sorry

end bob_distance_walked_l3042_304225


namespace complex_problem_l3042_304299

def complex_equation (z : ℂ) : Prop := (1 + 2*Complex.I) * z = 3 - 4*Complex.I

def third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_problem (z : ℂ) (h : complex_equation z) :
  z = -1 - 2*Complex.I ∧ third_quadrant z := by
  sorry

end complex_problem_l3042_304299


namespace corn_stalks_per_row_l3042_304210

/-- Proves that given 5 rows of corn, 8 corn stalks per bushel, and a total harvest of 50 bushels,
    the number of corn stalks in each row is 80. -/
theorem corn_stalks_per_row 
  (rows : ℕ) 
  (stalks_per_bushel : ℕ) 
  (total_bushels : ℕ) 
  (h1 : rows = 5)
  (h2 : stalks_per_bushel = 8)
  (h3 : total_bushels = 50) :
  (total_bushels * stalks_per_bushel) / rows = 80 := by
  sorry

end corn_stalks_per_row_l3042_304210


namespace reeyas_average_score_l3042_304227

theorem reeyas_average_score : 
  let scores : List ℕ := [65, 67, 76, 82, 85]
  (scores.sum / scores.length : ℚ) = 75 := by
  sorry

end reeyas_average_score_l3042_304227


namespace hexagon_angle_sum_l3042_304218

-- Define the hexagon and its properties
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  x : ℝ
  y : ℝ
  h : A = 34
  i : B = 74
  j : C = 32

-- State the theorem
theorem hexagon_angle_sum (H : Hexagon) : H.x + H.y = 40 := by
  sorry

end hexagon_angle_sum_l3042_304218


namespace sum_of_specific_sequences_l3042_304286

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_of_sequences (seq1 seq2 : List ℕ) : ℕ :=
  (seq1 ++ seq2).sum

theorem sum_of_specific_sequences :
  let seq1 := arithmetic_sequence 3 10 5
  let seq2 := arithmetic_sequence 7 10 5
  sum_of_sequences seq1 seq2 = 250 := by
  sorry

end sum_of_specific_sequences_l3042_304286


namespace quadratic_form_ratio_l3042_304226

theorem quadratic_form_ratio (k : ℝ) : ∃ (d r s : ℝ),
  5 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -22 := by
  sorry

end quadratic_form_ratio_l3042_304226


namespace marble_probability_l3042_304287

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 5

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 7

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := blue_marbles + white_marbles + red_marbles

/-- The number of marbles to be drawn -/
def marbles_drawn : ℕ := total_marbles - 2

/-- The probability of having one white and one blue marble remaining after randomly drawing marbles until only two are left -/
theorem marble_probability : 
  (Nat.choose blue_marbles blue_marbles * Nat.choose white_marbles (white_marbles - 1) * Nat.choose red_marbles red_marbles) / 
  Nat.choose total_marbles marbles_drawn = 7 / 120 := by
  sorry

end marble_probability_l3042_304287


namespace max_profit_at_optimal_price_l3042_304262

/-- Represents the product pricing and sales model -/
structure ProductModel where
  initial_price : ℝ
  initial_sales : ℝ
  price_demand_slope : ℝ
  cost_price : ℝ

/-- Calculates the profit function for a given price decrease -/
def profit_function (model : ProductModel) (x : ℝ) : ℝ :=
  let new_price := model.initial_price - x
  let new_sales := model.initial_sales + model.price_demand_slope * x
  (new_price - model.cost_price) * new_sales

/-- Theorem stating the maximum profit and optimal price decrease -/
theorem max_profit_at_optimal_price (model : ProductModel) 
  (h_initial_price : model.initial_price = 60)
  (h_initial_sales : model.initial_sales = 300)
  (h_price_demand_slope : model.price_demand_slope = 30)
  (h_cost_price : model.cost_price = 40) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    profit_function model x = 6750 ∧ 
    ∀ (y : ℝ), profit_function model y ≤ profit_function model x :=
  sorry

#eval profit_function ⟨60, 300, 30, 40⟩ 5

end max_profit_at_optimal_price_l3042_304262
