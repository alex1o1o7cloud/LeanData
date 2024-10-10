import Mathlib

namespace age_puzzle_l3898_389840

theorem age_puzzle (A N : ℕ) (h1 : A = 30) (h2 : (A + 5) * N - (A - 5) * N = A) : N = 3 := by
  sorry

end age_puzzle_l3898_389840


namespace fliers_remaining_l3898_389843

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_afternoon : afternoon_fraction = 1 / 4) :
  total - (total * morning_fraction).floor - ((total - (total * morning_fraction).floor) * afternoon_fraction).floor = 1350 :=
by sorry

end fliers_remaining_l3898_389843


namespace min_cost_for_20_oranges_l3898_389857

/-- Represents a discount scheme for oranges -/
structure DiscountScheme where
  quantity : ℕ
  price : ℕ

/-- Calculates the cost of oranges given a discount scheme and number of groups -/
def calculateCost (scheme : DiscountScheme) (groups : ℕ) : ℕ :=
  scheme.price * groups

/-- Finds the minimum cost for a given number of oranges using available discount schemes -/
def minCostForOranges (schemes : List DiscountScheme) (targetOranges : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem min_cost_for_20_oranges :
  let schemes := [
    DiscountScheme.mk 4 12,
    DiscountScheme.mk 7 21
  ]
  minCostForOranges schemes 20 = 60 := by
  sorry

end min_cost_for_20_oranges_l3898_389857


namespace parallel_lines_distance_l3898_389802

/-- Given two parallel lines 3x - 2y - 1 = 0 and 6x + ay + c = 0 with a distance of 2√13/13 between them, prove that (c + 2)/a = 1 -/
theorem parallel_lines_distance (a c : ℝ) : 
  (∀ x y : ℝ, 3 * x - 2 * y - 1 = 0 ↔ 6 * x + a * y + c = 0) →  -- lines are equivalent
  (∃ k : ℝ, k ≠ 0 ∧ 3 = k * 6 ∧ -2 = k * a) →  -- lines are parallel
  (|c/2 + 1| / Real.sqrt 13 = 2 * Real.sqrt 13 / 13) →  -- distance between lines
  (c + 2) / a = 1 :=
by sorry

end parallel_lines_distance_l3898_389802


namespace inequality_solution_set_l3898_389897

theorem inequality_solution_set (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end inequality_solution_set_l3898_389897


namespace triangle_area_l3898_389864

/-- Given a triangle with side lengths a, b, c where:
  - a = 13
  - The angle opposite side a is 60°
  - b : c = 4 : 3
  Prove that the area of the triangle is 39√3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (h1 : a = 13) (h2 : A = π / 3)
    (h3 : ∃ (k : ℝ), b = 4 * k ∧ c = 3 * k) :
    (1 / 2) * b * c * Real.sin A = 39 * Real.sqrt 3 := by
  sorry

end triangle_area_l3898_389864


namespace double_dimensions_cylinder_l3898_389807

/-- A cylindrical container with original volume and new volume after doubling dimensions -/
structure Container where
  originalVolume : ℝ
  newVolume : ℝ

/-- The volume of a cylinder doubles when its radius is doubled -/
def volumeDoubledRadius (v : ℝ) : ℝ := 4 * v

/-- The volume of a cylinder doubles when its height is doubled -/
def volumeDoubledHeight (v : ℝ) : ℝ := 2 * v

/-- Theorem: Doubling all dimensions of a 5-gallon cylindrical container results in a 40-gallon container -/
theorem double_dimensions_cylinder (c : Container) 
  (h₁ : c.originalVolume = 5)
  (h₂ : c.newVolume = volumeDoubledHeight (volumeDoubledRadius c.originalVolume)) :
  c.newVolume = 40 := by
  sorry

#check double_dimensions_cylinder

end double_dimensions_cylinder_l3898_389807


namespace remainder_of_1021_pow_1022_mod_1023_l3898_389823

theorem remainder_of_1021_pow_1022_mod_1023 : 
  1021^1022 % 1023 = 16 := by
  sorry

end remainder_of_1021_pow_1022_mod_1023_l3898_389823


namespace sum_of_squares_l3898_389862

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : a^2 + b^2 = 13 := by
  sorry

end sum_of_squares_l3898_389862


namespace extreme_value_implies_a_equals_5_l3898_389890

/-- The function f(x) = x³ + ax² + 3x - 9 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) --/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 :=
sorry

end extreme_value_implies_a_equals_5_l3898_389890


namespace pentagon_square_side_ratio_l3898_389804

theorem pentagon_square_side_ratio :
  let pentagon_perimeter : ℝ := 100
  let square_perimeter : ℝ := 100
  let pentagon_side : ℝ := pentagon_perimeter / 5
  let square_side : ℝ := square_perimeter / 4
  pentagon_side / square_side = 4 / 5 := by
sorry

end pentagon_square_side_ratio_l3898_389804


namespace all_terms_are_perfect_squares_l3898_389835

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 2 * (a (n + 2) + a (n + 1)) - a n

/-- Theorem: All terms in the sequence a_n are perfect squares -/
theorem all_terms_are_perfect_squares :
  ∃ x : ℕ → ℤ, ∀ n : ℕ, a n = (x n)^2 := by
  sorry

end all_terms_are_perfect_squares_l3898_389835


namespace xiaoming_average_is_92_l3898_389809

/-- Calculates the weighted average of Xiao Ming's math scores -/
def xiaoming_weighted_average : ℚ :=
  let regular_score : ℚ := 89
  let midterm_score : ℚ := 91
  let final_score : ℚ := 95
  let regular_weight : ℚ := 3
  let midterm_weight : ℚ := 3
  let final_weight : ℚ := 4
  (regular_score * regular_weight + midterm_score * midterm_weight + final_score * final_weight) /
  (regular_weight + midterm_weight + final_weight)

/-- Theorem stating that Xiao Ming's weighted average math score is 92 -/
theorem xiaoming_average_is_92 : xiaoming_weighted_average = 92 := by
  sorry

end xiaoming_average_is_92_l3898_389809


namespace number_of_roses_roses_count_l3898_389891

theorem number_of_roses (vase_capacity : ℕ) (carnations : ℕ) (vases : ℕ) : ℕ :=
  let total_flowers := vase_capacity * vases
  total_flowers - carnations

theorem roses_count : number_of_roses 6 7 9 = 47 := by
  sorry

end number_of_roses_roses_count_l3898_389891


namespace inequality_solution_l3898_389831

theorem inequality_solution (x : ℝ) : (x - 2) * (6 + 2*x) > 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end inequality_solution_l3898_389831


namespace equation_solution_l3898_389898

theorem equation_solution (p : ℝ) (hp : p > 0) :
  ∃ x : ℝ, Real.sqrt (x^2 + 2*p*x - p^2) - Real.sqrt (x^2 - 2*p*x - p^2) = 1 ↔
  (|p| < 1/2 ∧ (x = Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)) ∨
               x = -Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)))) :=
by sorry

end equation_solution_l3898_389898


namespace b_rent_exceeds_total_cost_l3898_389839

/-- Represents the rent rates for different animals -/
structure RentRates where
  horse : ℕ
  cow : ℕ
  sheep : ℕ
  goat : ℕ

/-- Represents the animals and duration for a renter -/
structure RenterAnimals where
  horses : ℕ
  horseDuration : ℕ
  sheep : ℕ
  sheepDuration : ℕ
  goats : ℕ
  goatDuration : ℕ

/-- Calculates the total rent for a renter given their animals and rent rates -/
def calculateRent (animals : RenterAnimals) (rates : RentRates) : ℕ :=
  animals.horses * animals.horseDuration * rates.horse +
  animals.sheep * animals.sheepDuration * rates.sheep +
  animals.goats * animals.goatDuration * rates.goat

/-- The total cost of the pasture -/
def totalPastureCost : ℕ := 5820

/-- The rent rates for different animals -/
def givenRates : RentRates :=
  { horse := 30
    cow := 40
    sheep := 20
    goat := 25 }

/-- B's animals and their durations -/
def bAnimals : RenterAnimals :=
  { horses := 16
    horseDuration := 9
    sheep := 18
    sheepDuration := 7
    goats := 4
    goatDuration := 6 }

theorem b_rent_exceeds_total_cost :
  calculateRent bAnimals givenRates > totalPastureCost := by
  sorry

end b_rent_exceeds_total_cost_l3898_389839


namespace range_of_m_l3898_389892

-- Define the quadratic function
def f (m x : ℝ) := m * x^2 - m * x - 1

-- Define the solution set
def solution_set (m : ℝ) := {x : ℝ | f m x ≥ 0}

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, solution_set m = ∅) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

end range_of_m_l3898_389892


namespace min_value_cube_sum_plus_inverse_cube_equality_condition_l3898_389867

theorem min_value_cube_sum_plus_inverse_cube (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ 4^(1/4) :=
sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 = 4^(1/4) ↔ a = b ∧ a = (4^(1/4) / 2)^(1/3) :=
sorry

end min_value_cube_sum_plus_inverse_cube_equality_condition_l3898_389867


namespace quadratic_radical_equivalence_l3898_389885

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_radical_equivalence (m : ℕ) :
  (is_prime 2 ∧ is_prime (2023 - m)) → m = 2021 := by
  sorry

end quadratic_radical_equivalence_l3898_389885


namespace boys_circle_distance_l3898_389811

theorem boys_circle_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) : 
  n * (2 * (2 * r)) = 800 := by
  sorry

end boys_circle_distance_l3898_389811


namespace line_equation_through_two_points_l3898_389853

/-- The equation of a line passing through two points is x + y = 1 -/
theorem line_equation_through_two_points :
  ∀ (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ),
  A = (1, -2) →
  B = (-3, 2) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ((x - 1) * (2 - (-2)) = (y - (-2)) * ((-3) - 1))) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y = 1) :=
by sorry

end line_equation_through_two_points_l3898_389853


namespace batsman_average_after_15th_innings_l3898_389884

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1 : ℚ)

/-- Theorem: Batsman's average after 15th innings -/
theorem batsman_average_after_15th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 85 = stats.average + 3)
  : newAverage stats 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end batsman_average_after_15th_innings_l3898_389884


namespace aubrey_garden_yield_l3898_389846

/-- Represents Aubrey's garden layout and plant yields --/
structure Garden where
  total_rows : Nat
  tomato_plants_per_row : Nat
  cucumber_plants_per_row : Nat
  bell_pepper_plants_per_row : Nat
  tomato_yield_first_last : Nat
  tomato_yield_middle : Nat
  cucumber_yield_a : Nat
  cucumber_yield_b : Nat
  bell_pepper_yield : Nat

/-- Calculates the total yield of vegetables in Aubrey's garden --/
def calculate_yield (g : Garden) : Nat × Nat × Nat :=
  let pattern_rows := 4
  let patterns := g.total_rows / pattern_rows
  let tomato_rows := patterns
  let cucumber_rows := 2 * patterns
  let bell_pepper_rows := patterns

  let tomatoes_per_row := 2 * g.tomato_yield_first_last + (g.tomato_plants_per_row - 2) * g.tomato_yield_middle
  let cucumbers_per_row := (g.cucumber_plants_per_row / 2) * (g.cucumber_yield_a + g.cucumber_yield_b)
  let bell_peppers_per_row := g.bell_pepper_plants_per_row * g.bell_pepper_yield

  let total_tomatoes := tomato_rows * tomatoes_per_row
  let total_cucumbers := cucumber_rows * cucumbers_per_row
  let total_bell_peppers := bell_pepper_rows * bell_peppers_per_row

  (total_tomatoes, total_cucumbers, total_bell_peppers)

/-- Theorem stating the total yield of Aubrey's garden --/
theorem aubrey_garden_yield (g : Garden)
  (h1 : g.total_rows = 20)
  (h2 : g.tomato_plants_per_row = 8)
  (h3 : g.cucumber_plants_per_row = 6)
  (h4 : g.bell_pepper_plants_per_row = 12)
  (h5 : g.tomato_yield_first_last = 6)
  (h6 : g.tomato_yield_middle = 4)
  (h7 : g.cucumber_yield_a = 4)
  (h8 : g.cucumber_yield_b = 5)
  (h9 : g.bell_pepper_yield = 2) :
  calculate_yield g = (180, 270, 120) := by
  sorry

#eval calculate_yield {
  total_rows := 20,
  tomato_plants_per_row := 8,
  cucumber_plants_per_row := 6,
  bell_pepper_plants_per_row := 12,
  tomato_yield_first_last := 6,
  tomato_yield_middle := 4,
  cucumber_yield_a := 4,
  cucumber_yield_b := 5,
  bell_pepper_yield := 2
}

end aubrey_garden_yield_l3898_389846


namespace exists_n_pow_half_n_eq_ten_l3898_389812

theorem exists_n_pow_half_n_eq_ten : ∃ n : ℝ, n ^ (n / 2) = 10 := by
  sorry

end exists_n_pow_half_n_eq_ten_l3898_389812


namespace opposite_of_negative_sqrt3_squared_l3898_389825

theorem opposite_of_negative_sqrt3_squared : -((-Real.sqrt 3)^2) = -3 := by
  sorry

end opposite_of_negative_sqrt3_squared_l3898_389825


namespace tank_capacity_l3898_389814

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  currentPercentage : ℝ
  currentVolume : ℝ

/-- Theorem: A cylindrical tank that is 25% full with 60 liters has a total capacity of 240 liters --/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.currentPercentage = 0.25)
  (h2 : tank.currentVolume = 60) : 
  tank.capacity = 240 := by
  sorry

#check tank_capacity

end tank_capacity_l3898_389814


namespace range_of_f_l3898_389841

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Iic 14 :=
sorry

end range_of_f_l3898_389841


namespace parabola_directrix_l3898_389826

/-- Represents a parabola with equation y = -4x^2 + 4 -/
structure Parabola where
  /-- The y-coordinate of the focus -/
  f : ℝ
  /-- The y-coordinate of the directrix -/
  d : ℝ

/-- Theorem: The directrix of the parabola y = -4x^2 + 4 is y = 65/16 -/
theorem parabola_directrix (p : Parabola) : p.d = 65/16 := by
  sorry

end parabola_directrix_l3898_389826


namespace smallest_dual_palindrome_l3898_389886

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 17 :=
sorry

end smallest_dual_palindrome_l3898_389886


namespace duck_travel_east_l3898_389819

def days_to_south : ℕ := 40
def days_to_north : ℕ := 2 * days_to_south
def total_days : ℕ := 180

def days_to_east : ℕ := total_days - days_to_south - days_to_north

theorem duck_travel_east : days_to_east = 60 := by
  sorry

end duck_travel_east_l3898_389819


namespace domain_of_g_l3898_389816

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 := by sorry

end domain_of_g_l3898_389816


namespace inequalities_proof_l3898_389818

theorem inequalities_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : a < b) (h3 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) := by
sorry

end inequalities_proof_l3898_389818


namespace carla_wins_one_l3898_389866

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carla : Player

/-- Represents the result of a game for a player -/
inductive GameResult : Type
| Win : GameResult
| Loss : GameResult

/-- The number of games each player plays against each other player -/
def gamesPerPair : Nat := 2

/-- The total number of games in the tournament -/
def totalGames : Nat := 12

/-- The number of wins for a given player -/
def wins (p : Player) : Nat :=
  match p with
  | Player.Alice => 5
  | Player.Bob => 6
  | Player.Carla => 1  -- This is what we want to prove

/-- The number of losses for a given player -/
def losses (p : Player) : Nat :=
  match p with
  | Player.Alice => 3
  | Player.Bob => 2
  | Player.Carla => 5

theorem carla_wins_one :
  (∀ p : Player, wins p + losses p = totalGames / 2) ∧
  (wins Player.Alice + wins Player.Bob + wins Player.Carla = totalGames) :=
by sorry

end carla_wins_one_l3898_389866


namespace parallel_vector_sum_diff_l3898_389838

/-- Given two vectors a and b in ℝ², where a = (1, -1) and b = (t, 1),
    if a + b is parallel to a - b, then t = -1. -/
theorem parallel_vector_sum_diff (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end parallel_vector_sum_diff_l3898_389838


namespace hide_and_seek_players_l3898_389837

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis) ∧ ¬Andrew ∧ ¬Gena :=
sorry

end hide_and_seek_players_l3898_389837


namespace intersection_range_l3898_389836

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}
def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

-- State the theorem
theorem intersection_range (b : ℝ) : 
  (M ∩ N b).Nonempty → b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end intersection_range_l3898_389836


namespace stone_to_crystal_ratio_is_two_to_one_l3898_389848

/-- A bracelet making scenario with Nancy and Rose -/
structure BraceletScenario where
  beads_per_bracelet : ℕ
  nancy_metal_beads : ℕ
  nancy_pearl_beads : ℕ
  rose_crystal_beads : ℕ
  total_bracelets : ℕ

/-- Calculate the ratio of Rose's stone beads to crystal beads -/
def stone_to_crystal_ratio (scenario : BraceletScenario) : ℚ :=
  let total_beads := scenario.total_bracelets * scenario.beads_per_bracelet
  let nancy_total_beads := scenario.nancy_metal_beads + scenario.nancy_pearl_beads
  let rose_total_beads := total_beads - nancy_total_beads
  let rose_stone_beads := rose_total_beads - scenario.rose_crystal_beads
  (rose_stone_beads : ℚ) / scenario.rose_crystal_beads

/-- The given bracelet scenario -/
def given_scenario : BraceletScenario :=
  { beads_per_bracelet := 8
  , nancy_metal_beads := 40
  , nancy_pearl_beads := 60  -- 40 + 20
  , rose_crystal_beads := 20
  , total_bracelets := 20 }

theorem stone_to_crystal_ratio_is_two_to_one :
  stone_to_crystal_ratio given_scenario = 2 := by
  sorry

end stone_to_crystal_ratio_is_two_to_one_l3898_389848


namespace bag_of_balls_l3898_389873

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 17)
  (h4 : red = 3)
  (h5 : purple = 1)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 95 / 100) :
  white + green + yellow + red + purple = 80 := by
  sorry

end bag_of_balls_l3898_389873


namespace digit_interchange_effect_l3898_389881

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) : 
  n = 9 → 
  p > q → 
  p - q = 1 → 
  (10 * p + q) - (10 * q + p) = (n : ℤ) := by
  sorry

end digit_interchange_effect_l3898_389881


namespace negation_of_existence_negation_of_quadratic_inequality_l3898_389895

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃x < 1, P x) ↔ (∀x < 1, ¬P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬∃x < 1, x^2 + 2*x + 1 ≤ 0) ↔ (∀x < 1, x^2 + 2*x + 1 > 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3898_389895


namespace two_digit_powers_of_three_l3898_389842

theorem two_digit_powers_of_three :
  ∃! (s : Finset ℕ), (∀ n ∈ s, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ s.card = 2 := by
  sorry

end two_digit_powers_of_three_l3898_389842


namespace line_equation_point_slope_l3898_389888

/-- A line passing through point (-1, 1) with slope 2 has the equation y = 2x + 3 -/
theorem line_equation_point_slope : 
  ∀ (x y : ℝ), y = 2*x + 3 ↔ (y - 1 = 2*(x - (-1)) ∧ (x, y) ≠ (-1, 1)) ∨ (x, y) = (-1, 1) :=
by sorry

end line_equation_point_slope_l3898_389888


namespace max_sum_cubes_l3898_389879

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  (∃ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 ∧ 
   x^3 + y^3 + z^3 + w^3 + v^3 ≥ a^3 + b^3 + c^3 + d^3 + e^3) ∧
  (∀ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 → 
   x^3 + y^3 + z^3 + w^3 + v^3 ≤ 5 * Real.sqrt 5) :=
by sorry

end max_sum_cubes_l3898_389879


namespace quadratic_equation_solution_l3898_389845

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 3 ∧ x₂ = -5) ∧ 
  (x₁^2 + 2*x₁ - 15 = 0) ∧ 
  (x₂^2 + 2*x₂ - 15 = 0) := by
  sorry

end quadratic_equation_solution_l3898_389845


namespace social_gathering_attendance_l3898_389865

theorem social_gathering_attendance
  (num_men : ℕ)
  (dances_per_man : ℕ)
  (dances_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_dances_per_man : dances_per_man = 4)
  (h_dances_per_woman : dances_per_woman = 3) :
  (num_men * dances_per_man) / dances_per_woman = 20 := by
sorry

end social_gathering_attendance_l3898_389865


namespace min_ellipse_area_l3898_389875

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- A circle with center (h, 0) and radius 1 -/
structure Circle where
  h : ℝ

/-- The ellipse is tangent to the circle -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧ (x - c.h)^2 + y^2 = 1

/-- The theorem stating the minimum area of the ellipse -/
theorem min_ellipse_area (e : Ellipse) (c1 c2 : Circle) 
  (h1 : is_tangent e c1) (h2 : is_tangent e c2) (h3 : c1.h = 2) (h4 : c2.h = -2) :
  e.a * e.b * π ≥ (10 * Real.sqrt 15 / 3) * π :=
sorry

end min_ellipse_area_l3898_389875


namespace problem_statement_l3898_389899

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : b^3 + b ≤ a - a^3) :
  (b < a ∧ a < 1) ∧ a^2 + b^2 < 1 := by
  sorry

end problem_statement_l3898_389899


namespace phone_price_is_3000_l3898_389805

/-- Represents the payment plan for a phone purchase -/
structure PaymentPlan where
  initialPayment : ℕ
  monthlyPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a payment plan -/
def totalCost (plan : PaymentPlan) : ℕ :=
  plan.initialPayment + plan.monthlyPayment * (plan.duration - 1)

/-- Represents the two-part payment plan -/
structure TwoPartPlan where
  firstHalfPayment : ℕ
  secondHalfPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a two-part payment plan -/
def twoPartTotalCost (plan : TwoPartPlan) : ℕ :=
  (plan.firstHalfPayment * (plan.duration / 2)) + (plan.secondHalfPayment * (plan.duration / 2))

/-- The theorem stating that the phone price is 3000 yuan given the described payment plans -/
theorem phone_price_is_3000 (plan1 : PaymentPlan) (plan2 : TwoPartPlan) 
    (h1 : plan1.initialPayment = 800)
    (h2 : plan1.monthlyPayment = 200)
    (h3 : plan2.firstHalfPayment = 350)
    (h4 : plan2.secondHalfPayment = 150)
    (h5 : plan1.duration = plan2.duration)
    (h6 : totalCost plan1 = twoPartTotalCost plan2) :
    totalCost plan1 = 3000 := by
  sorry

end phone_price_is_3000_l3898_389805


namespace arithmetic_sequence_a6_l3898_389889

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 7)
  (h_a5_a2 : a 5 = a 2 + 6) :
  a 6 = 13 := by
  sorry

end arithmetic_sequence_a6_l3898_389889


namespace intersection_M_N_l3898_389821

-- Define set M
def M : Set ℕ := {y | y < 6}

-- Define set N
def N : Set ℕ := {2, 3, 6}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l3898_389821


namespace fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l3898_389876

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_third_of_one_fourth_of_one_fifth_of_sixty :
  (1 : ℚ) / 3 * (1 : ℚ) / 4 * (1 : ℚ) / 5 * 60 = 1 := by sorry

end fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l3898_389876


namespace log_equality_implies_p_q_equal_three_l3898_389813

theorem log_equality_implies_p_q_equal_three (p q : ℝ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) 
  (h_log : Real.log p + Real.log q = Real.log (2*p + q)) : 
  p = 3 ∧ q = 3 := by
sorry

end log_equality_implies_p_q_equal_three_l3898_389813


namespace fraction_repetend_correct_l3898_389877

/-- The repetend of the decimal representation of 7/19 -/
def repetend : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The fraction we're considering -/
def fraction : Rat := 7 / 19

theorem fraction_repetend_correct :
  ∃ (k : Nat), fraction = (k : Rat) / 10^repetend.length + 
    (List.sum (List.zipWith (λ (d i : Nat) => d * 10^(repetend.length - 1 - i)) repetend (List.range repetend.length)) : Rat) / 
    (10^repetend.length - 1) / 19 :=
sorry

end fraction_repetend_correct_l3898_389877


namespace fraction_of_fraction_of_forty_l3898_389855

theorem fraction_of_fraction_of_forty : (2/3 : ℚ) * ((3/4 : ℚ) * 40) = 20 := by
  sorry

end fraction_of_fraction_of_forty_l3898_389855


namespace bob_weight_is_165_l3898_389893

def jim_weight : ℝ := sorry
def bob_weight : ℝ := sorry

axiom combined_weight : jim_weight + bob_weight = 220
axiom weight_relation : bob_weight - 2 * jim_weight = bob_weight / 3

theorem bob_weight_is_165 : bob_weight = 165 := by sorry

end bob_weight_is_165_l3898_389893


namespace min_white_surface_area_l3898_389849

/-- Represents a cube with side length 4, composed of unit cubes -/
structure LargeCube :=
  (side_length : Nat)
  (total_cubes : Nat)
  (red_cubes : Nat)
  (white_cubes : Nat)

/-- The fraction of the surface area that is white when minimized -/
def min_white_fraction (c : LargeCube) : Rat :=
  5 / 96

/-- Theorem stating the minimum fraction of white surface area -/
theorem min_white_surface_area (c : LargeCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.red_cubes = 58)
  (h4 : c.white_cubes = 6) :
  min_white_fraction c = 5 / 96 := by
  sorry

end min_white_surface_area_l3898_389849


namespace no_real_solutions_l3898_389815

/-- Given a function f(x) = x^2 + 2x + a, where f(bx) = 9x^2 - 6x + 2,
    prove that the equation f(ax + b) = 0 has no real solutions. -/
theorem no_real_solutions (a b : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = x^2 + 2*x + a) ∧
   (∀ x, f (b*x) = 9*x^2 - 6*x + 2)) →
  (∀ x, (x^2 + 2*x + a) ≠ 0) :=
by sorry

end no_real_solutions_l3898_389815


namespace chinese_remainder_theorem_two_three_l3898_389822

theorem chinese_remainder_theorem_two_three :
  (∀ (a b : ℤ), ∃ (x : ℤ), x ≡ a [ZMOD 5] ∧ x ≡ b [ZMOD 6] ∧ x = 6*a + 25*b) ∧
  (∀ (a b c : ℤ), ∃ (y : ℤ), y ≡ a [ZMOD 5] ∧ y ≡ b [ZMOD 6] ∧ y ≡ c [ZMOD 7] ∧ y = 126*a + 175*b + 120*c) :=
by sorry

end chinese_remainder_theorem_two_three_l3898_389822


namespace recruit_line_unique_solution_l3898_389801

/-- Represents the position of a person in the line of recruits -/
structure Position :=
  (front : ℕ)  -- number of people in front
  (behind : ℕ) -- number of people behind

/-- The line of recruits -/
structure RecruitLine :=
  (total : ℕ)
  (peter : Position)
  (nikolai : Position)
  (denis : Position)

/-- Conditions of the problem -/
def problem_conditions (line : RecruitLine) : Prop :=
  line.peter.front = 50 ∧
  line.nikolai.front = 100 ∧
  line.denis.front = 170 ∧
  (line.peter.behind = 4 * line.denis.behind ∨
   line.nikolai.behind = 4 * line.denis.behind ∨
   line.peter.behind = 4 * line.nikolai.behind) ∧
  line.total = line.denis.front + 1 + line.denis.behind

/-- The theorem to be proved -/
theorem recruit_line_unique_solution :
  ∃! line : RecruitLine, problem_conditions line ∧ line.total = 301 :=
sorry

end recruit_line_unique_solution_l3898_389801


namespace dog_food_duration_aunt_gemma_dog_food_duration_l3898_389828

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, number of sacks, and weight of each sack. -/
theorem dog_food_duration (num_dogs : ℕ) (feedings_per_day : ℕ) (food_per_meal : ℕ)
                          (num_sacks : ℕ) (sack_weight_kg : ℕ) : ℕ :=
  let total_food_grams : ℕ := num_sacks * sack_weight_kg * 1000
  let daily_consumption : ℕ := num_dogs * food_per_meal * feedings_per_day
  total_food_grams / daily_consumption

/-- Proves that given Aunt Gemma's specific conditions, the dog food will last for 50 days. -/
theorem aunt_gemma_dog_food_duration : 
  dog_food_duration 4 2 250 2 50 = 50 := by
  sorry

end dog_food_duration_aunt_gemma_dog_food_duration_l3898_389828


namespace fathers_age_l3898_389874

/-- Represents the ages of family members and proves the father's age -/
theorem fathers_age (total_age sister_age kaydence_age : ℕ) 
  (h1 : total_age = 200)
  (h2 : sister_age = 40)
  (h3 : kaydence_age = 12) :
  ∃ (father_age : ℕ),
    father_age = 60 ∧
    ∃ (mother_age brother_age : ℕ),
      mother_age = father_age - 2 ∧
      brother_age = father_age / 2 ∧
      father_age + mother_age + brother_age + sister_age + kaydence_age = total_age :=
by
  sorry


end fathers_age_l3898_389874


namespace cattle_count_farm_cattle_count_l3898_389833

theorem cattle_count (cow_ratio : ℕ) (bull_ratio : ℕ) (bull_count : ℕ) : ℕ :=
  let total_ratio := cow_ratio + bull_ratio
  let parts := bull_count / bull_ratio
  let total_cattle := parts * total_ratio
  total_cattle

/-- Given a ratio of cows to bulls of 10:27 and 405 bulls, the total number of cattle is 675. -/
theorem farm_cattle_count : cattle_count 10 27 405 = 675 := by
  sorry

end cattle_count_farm_cattle_count_l3898_389833


namespace sin_cos_difference_l3898_389858

theorem sin_cos_difference (x : ℝ) : 
  Real.sin (65 * π / 180 - x) * Real.cos (x - 20 * π / 180) - 
  Real.cos (65 * π / 180 - x) * Real.sin (20 * π / 180 - x) = 
  Real.sqrt 2 / 2 := by
sorry

end sin_cos_difference_l3898_389858


namespace castle_doors_problem_l3898_389810

theorem castle_doors_problem (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end castle_doors_problem_l3898_389810


namespace scrabble_multiplier_is_three_l3898_389871

/-- Represents a three-letter word in Scrabble --/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the multiplier for a given Scrabble word and final score --/
def calculate_multiplier (word : ScrabbleWord) (final_score : ℕ) : ℚ :=
  final_score / (word.first_letter_value + word.middle_letter_value + word.last_letter_value)

theorem scrabble_multiplier_is_three :
  let word : ScrabbleWord := {
    first_letter_value := 1,
    middle_letter_value := 8,
    last_letter_value := 1
  }
  let final_score : ℕ := 30
  calculate_multiplier word final_score = 3 := by
    sorry

end scrabble_multiplier_is_three_l3898_389871


namespace percentage_calculation_l3898_389860

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * (30 / 100) * (50 / 100) * 5200 = 117 → P = 15 := by
  sorry

end percentage_calculation_l3898_389860


namespace tyler_double_flips_l3898_389803

/-- Represents the number of flips in a single move for each gymnast -/
def triple_flip : ℕ := 3
def double_flip : ℕ := 2

/-- Represents the number of triple-flips Jen performed -/
def jen_triple_flips : ℕ := 16

/-- Calculates the total number of flips Jen performed -/
def jen_total_flips : ℕ := jen_triple_flips * triple_flip

/-- Calculates the total number of flips Tyler performed -/
def tyler_total_flips : ℕ := jen_total_flips / 2

/-- Theorem: Given the conditions, Tyler performed 12 double-flips -/
theorem tyler_double_flips : tyler_total_flips / double_flip = 12 := by
  sorry

end tyler_double_flips_l3898_389803


namespace quadratic_equation_integer_roots_l3898_389808

theorem quadratic_equation_integer_roots (m : ℕ) (a : ℝ) :
  (1 ≤ m) →
  (m ≤ 50) →
  (∃ x₁ x₂ : ℕ, 
    x₁ ≠ x₂ ∧
    (x₁ - 2)^2 + (a - m)^2 = 2 * m * x₁ + a^2 - 2 * a * m ∧
    (x₂ - 2)^2 + (a - m)^2 = 2 * m * x₂ + a^2 - 2 * a * m) →
  ∃ k : ℕ, m = k^2 ∧ k^2 ≤ 49 :=
by sorry

end quadratic_equation_integer_roots_l3898_389808


namespace A_intersect_B_l3898_389872

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ m : ℕ, x = 2 * m}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end A_intersect_B_l3898_389872


namespace calculate_expression_l3898_389870

theorem calculate_expression : 
  2 * Real.sin (π / 4) + |(-Real.sqrt 2)| - (π - 2023)^0 - Real.sqrt 2 = Real.sqrt 2 - 1 := by
  sorry

end calculate_expression_l3898_389870


namespace negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l3898_389817

-- 1. The square of every natural number is positive.
theorem negation_square_positive : 
  (∀ n : ℕ, n^2 > 0) ↔ ¬(∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

-- 2. Every real number x is a root of the equation 5x-12=0.
theorem negation_root_equation : 
  (∀ x : ℝ, 5*x - 12 = 0) ↔ ¬(∃ x : ℝ, 5*x - 12 ≠ 0) :=
by sorry

-- 3. For every real number x, there exists a real number y such that x+y>0.
theorem negation_sum_positive : 
  (∀ x : ℝ, ∃ y : ℝ, x + y > 0) ↔ ¬(∃ x : ℝ, ∀ y : ℝ, x + y ≤ 0) :=
by sorry

-- 4. Some prime numbers are odd.
theorem negation_prime_odd : 
  (∃ p : ℕ, Prime p ∧ Odd p) ↔ ¬(∀ p : ℕ, Prime p → ¬Odd p) :=
by sorry

end negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l3898_389817


namespace sum_of_81_and_15_l3898_389847

theorem sum_of_81_and_15 : 81 + 15 = 96 := by
  sorry

end sum_of_81_and_15_l3898_389847


namespace translation_of_parabola_l3898_389852

theorem translation_of_parabola (t m : ℝ) : 
  (∀ x : ℝ, (x - 3)^2 = (t - 3)^2 → x = t) →  -- P is on y=(x-3)^2
  (t - m)^2 = (t - 3)^2 →                     -- Q is on y=x^2
  m = 3 := by
sorry

end translation_of_parabola_l3898_389852


namespace gadget_price_proof_l3898_389878

theorem gadget_price_proof (sticker_price : ℝ) : 
  (0.80 * sticker_price - 80) = (0.65 * sticker_price - 20) → sticker_price = 400 := by
  sorry

end gadget_price_proof_l3898_389878


namespace divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l3898_389861

def σ (n : ℕ) : ℕ := sorry

def φ (n : ℕ) : ℕ := sorry

theorem divisor_sum_and_totient_inequality (n : ℕ) :
  n ≠ 0 → (1 : ℝ) / σ n + (1 : ℝ) / φ n ≥ 2 / n :=
sorry

theorem divisor_sum_and_totient_equality (n : ℕ) :
  n ≠ 0 → ((1 : ℝ) / σ n + (1 : ℝ) / φ n = 2 / n ↔ n = 1) :=
sorry

end divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l3898_389861


namespace max_value_implies_a_l3898_389863

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 := by
  sorry


end max_value_implies_a_l3898_389863


namespace no_unique_solution_l3898_389820

theorem no_unique_solution (a : ℝ) : ¬ ∃! p : ℝ × ℝ, 
  p.1^2 + p.2^2 = 2 ∧ |p.2| - p.1 = a :=
by
  sorry

end no_unique_solution_l3898_389820


namespace smallest_three_digit_multiple_of_17_l3898_389856

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_17_l3898_389856


namespace symmetric_line_passes_through_fixed_point_l3898_389829

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line2D where
  slope : ℝ
  point : ℝ × ℝ

/-- The symmetric point of a given point with respect to a center point -/
def symmetricPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * (p.1 - l.point.1) + l.point.2

/-- Two lines are symmetric about a point if the reflection of any point on one line
    through the center point lies on the other line -/
def symmetricLines (l1 l2 : Line2D) (center : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ, pointOnLine l1 p → pointOnLine l2 (symmetricPoint p center)

theorem symmetric_line_passes_through_fixed_point :
  ∀ (k : ℝ) (l1 l2 : Line2D),
    l1.slope = k ∧
    l1.point = (4, 0) ∧
    symmetricLines l1 l2 (2, 1) →
    pointOnLine l2 (0, 2) := by
  sorry

end symmetric_line_passes_through_fixed_point_l3898_389829


namespace problem_solution_l3898_389824

theorem problem_solution : (π - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 := by
  sorry

end problem_solution_l3898_389824


namespace gain_amount_calculation_l3898_389850

/-- Calculates the amount given the gain and gain percent -/
def calculateAmount (gain : ℚ) (gainPercent : ℚ) : ℚ :=
  gain / (gainPercent / 100)

/-- Theorem: Given a gain of 0.70 rupees and a gain percent of 1%, 
    the amount on which the gain is made is 70 rupees -/
theorem gain_amount_calculation (gain : ℚ) (gainPercent : ℚ) 
  (h1 : gain = 70/100) (h2 : gainPercent = 1) : 
  calculateAmount gain gainPercent = 70 := by
  sorry

#eval calculateAmount (70/100) 1

end gain_amount_calculation_l3898_389850


namespace complex_number_quadrant_l3898_389896

theorem complex_number_quadrant : 
  let z : ℂ := Complex.mk (Real.sin 1) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_number_quadrant_l3898_389896


namespace internship_arrangement_l3898_389827

theorem internship_arrangement (n : Nat) (k : Nat) (m : Nat) : 
  n = 5 → k = 4 → m = 2 →
  (Nat.choose k m / 2) * (Nat.factorial n / (Nat.factorial (n - m))) = 60 := by
  sorry

end internship_arrangement_l3898_389827


namespace line_xz_plane_intersection_l3898_389882

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := {
    p1 := { x := 2, y := -1, z := 3 },
    p2 := { x := 6, y := -4, z := 7 }
  }
  let intersectionPoint : Point3D := { x := 2/3, y := 0, z := 5/3 }
  (intersectionPoint ∈ xzPlane) ∧ 
  (pointOnLine intersectionPoint l) := by sorry

end line_xz_plane_intersection_l3898_389882


namespace marks_initial_friends_l3898_389887

/-- Calculates the initial number of friends Mark had -/
def initial_friends (kept_percentage : ℚ) (contacted_percentage : ℚ) (response_rate : ℚ) (final_friends : ℕ) : ℚ :=
  final_friends / (kept_percentage + contacted_percentage * response_rate)

/-- Proves that Mark initially had 100 friends -/
theorem marks_initial_friends :
  let kept_percentage : ℚ := 2/5
  let contacted_percentage : ℚ := 3/5
  let response_rate : ℚ := 1/2
  let final_friends : ℕ := 70
  initial_friends kept_percentage contacted_percentage response_rate final_friends = 100 := by
  sorry

#eval initial_friends (2/5) (3/5) (1/2) 70

end marks_initial_friends_l3898_389887


namespace range_of_sum_l3898_389880

theorem range_of_sum (a b : ℝ) (h : a^2 - a*b + b^2 = a + b) :
  ∃ t : ℝ, t = a + b ∧ 0 ≤ t ∧ t ≤ 4 :=
sorry

end range_of_sum_l3898_389880


namespace inequality_range_l3898_389883

theorem inequality_range : 
  {a : ℝ | ∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
sorry

end inequality_range_l3898_389883


namespace arithmetic_progression_properties_l3898_389832

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric progression condition
def geometric_progression_condition (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, a 4 = a 2 * r ∧ a 8 = a 4 * r

-- Main theorem
theorem arithmetic_progression_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_progression a)
  (h_a1 : a 1 = 1)
  (h_geom : geometric_progression_condition a) :
  (∀ n : ℕ, a n = n) ∧
  (∀ n : ℕ, n ≤ 98 ↔ 100 * (1 - 1 / (n + 1)) < 99) :=
sorry

end arithmetic_progression_properties_l3898_389832


namespace watsonville_marching_band_max_members_l3898_389859

theorem watsonville_marching_band_max_members
  (m : ℕ)
  (band_size : ℕ)
  (h1 : band_size = 30 * m)
  (h2 : band_size % 31 = 7)
  (h3 : band_size < 1500) :
  band_size ≤ 720 ∧ ∃ (k : ℕ), 30 * k = 720 ∧ 720 % 31 = 7 :=
sorry

end watsonville_marching_band_max_members_l3898_389859


namespace pages_per_day_l3898_389894

theorem pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) :
  pages_per_book / days_per_book = 83 := by
  sorry

end pages_per_day_l3898_389894


namespace defective_units_percentage_l3898_389844

theorem defective_units_percentage 
  (shipped_defective_ratio : Real) 
  (total_shipped_defective_ratio : Real) 
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0016) : 
  total_shipped_defective_ratio / shipped_defective_ratio = 0.04 := by
sorry

end defective_units_percentage_l3898_389844


namespace roots_depend_on_k_l3898_389868

theorem roots_depend_on_k : 
  ∀ (k : ℝ), 
  ∃ (δ : ℝ), 
  δ = 1 + 4*k ∧ 
  (δ > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁ - 1)*(x₁ - 2) = k ∧ (x₂ - 1)*(x₂ - 2) = k) ∧
  (δ = 0 → ∃ (x : ℝ), (x - 1)*(x - 2) = k) ∧
  (δ < 0 → ¬∃ (x : ℝ), (x - 1)*(x - 2) = k) :=
by sorry


end roots_depend_on_k_l3898_389868


namespace half_of_number_l3898_389854

theorem half_of_number (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end half_of_number_l3898_389854


namespace geometric_sequence_sum_ratio_l3898_389806

theorem geometric_sequence_sum_ratio 
  (a q : ℝ) 
  (h_q : q ≠ 1) : 
  let S : ℕ → ℝ := λ n => a * (1 - q^n) / (1 - q)
  (S 6 / S 3 = 1 / 2) → (S 9 / S 3 = 3 / 4) :=
by
  sorry

end geometric_sequence_sum_ratio_l3898_389806


namespace least_n_squared_minus_n_divisibility_l3898_389830

theorem least_n_squared_minus_n_divisibility : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0))) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0)) →
    n ≥ 5) :=
by sorry

end least_n_squared_minus_n_divisibility_l3898_389830


namespace expected_value_theorem_l3898_389800

def N : ℕ := 123456789

/-- The expected value of N' when two distinct digits of N are randomly swapped -/
def expected_value_N_prime : ℚ := 555555555

/-- Theorem stating that the expected value of N' is 555555555 -/
theorem expected_value_theorem : expected_value_N_prime = 555555555 := by sorry

end expected_value_theorem_l3898_389800


namespace quadratic_function_property_l3898_389834

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  -- The axis of symmetry is at x = 3.5
  (∀ x : ℝ, QuadraticFunction a b c (3.5 - x) = QuadraticFunction a b c (3.5 + x)) →
  -- The function passes through the point (2, -1)
  QuadraticFunction a b c 2 = -1 →
  -- p(5) is an integer
  ∃ n : ℤ, QuadraticFunction a b c 5 = n →
  -- Then p(5) = -1
  QuadraticFunction a b c 5 = -1 := by
sorry

end quadratic_function_property_l3898_389834


namespace x_squared_minus_y_squared_l3898_389851

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) :
  x^2 - y^2 = 8 := by
  sorry

end x_squared_minus_y_squared_l3898_389851


namespace simplify_expression_l3898_389869

variable (y : ℝ)

theorem simplify_expression :
  3 * y - 5 * y^2 + 7 - (6 - 3 * y + 5 * y^2 - 2 * y^3) = 2 * y^3 - 10 * y^2 + 6 * y + 1 := by
  sorry

end simplify_expression_l3898_389869
