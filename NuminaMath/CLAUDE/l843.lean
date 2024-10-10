import Mathlib

namespace angle_b_in_special_triangle_l843_84343

/-- In a triangle ABC, if angle A is 80° and angle B equals angle C, then angle B is 50°. -/
theorem angle_b_in_special_triangle (A B C : Real) (h1 : A = 80)
  (h2 : B = C) (h3 : A + B + C = 180) : B = 50 := by
  sorry

end angle_b_in_special_triangle_l843_84343


namespace ap_contains_sixth_power_l843_84358

/-- An arithmetic progression containing squares and cubes contains a sixth power -/
theorem ap_contains_sixth_power (a h : ℕ) (p q : ℕ) : 
  0 < a → 0 < h → p ≠ q → p > 0 → q > 0 →
  (∃ k : ℕ, a + k * h = p^2) → 
  (∃ m : ℕ, a + m * h = q^3) → 
  (∃ n x : ℕ, a + n * h = x^6) :=
sorry

end ap_contains_sixth_power_l843_84358


namespace max_distance_between_circle_centers_l843_84382

theorem max_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 15)
  (h3 : circle_diameter = 8)
  (h4 : circle_diameter ≤ rectangle_width ∧ circle_diameter ≤ rectangle_height) :
  let max_distance := Real.sqrt ((rectangle_width - circle_diameter)^2 + (rectangle_height - circle_diameter)^2)
  max_distance = Real.sqrt 193 := by
  sorry

end max_distance_between_circle_centers_l843_84382


namespace triangle_inequality_l843_84354

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi) :
  (x + y + z)^2 ≥ 4 * (y * z * Real.sin A^2 + z * x * Real.sin B^2 + x * y * Real.sin C^2) := by
  sorry

end triangle_inequality_l843_84354


namespace chicken_nuggets_distribution_l843_84341

theorem chicken_nuggets_distribution (total : ℕ) (alyssa : ℕ) : 
  total = 100 → alyssa + 2 * alyssa + 2 * alyssa = total → alyssa = 20 := by
  sorry

end chicken_nuggets_distribution_l843_84341


namespace high_school_ten_games_l843_84384

def league_size : ℕ := 10
def non_league_games_per_team : ℕ := 6

def intra_league_games (n : ℕ) : ℕ :=
  n * (n - 1)

def total_games (n : ℕ) (m : ℕ) : ℕ :=
  (intra_league_games n) + (n * m)

theorem high_school_ten_games :
  total_games league_size non_league_games_per_team = 150 := by
  sorry

end high_school_ten_games_l843_84384


namespace same_terminal_side_angles_l843_84320

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem same_terminal_side_angles :
  (∃ k₁ : ℤ, angle_set k₁ = 240) ∧
  (∃ k₂ : ℤ, angle_set k₂ = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) →
    (α > 0 → α ≥ 240) ∧
    (α < 0 → α ≤ -120)) :=
sorry

end same_terminal_side_angles_l843_84320


namespace quadratic_function_properties_l843_84352

theorem quadratic_function_properties (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-2) = 0) → (f 3 = 0) → (f (-b / (2 * a)) > 0) →
  (a < 0) ∧ 
  ({x : ℝ | a * x + c > 0} = {x : ℝ | x > 6}) ∧
  (a + b + c > 0) ∧
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1/2}) :=
by sorry

end quadratic_function_properties_l843_84352


namespace exponent_division_l843_84324

theorem exponent_division (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end exponent_division_l843_84324


namespace exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l843_84337

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon : ℝ :=
  40

/-- A regular nonagon has 9 sides. -/
def regular_nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- An exterior angle and its corresponding interior angle sum to 180 degrees. -/
axiom exterior_interior_sum : ℝ → ℝ → Prop

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon_proof :
  exterior_angle_regular_nonagon =
    180 - (sum_interior_angles regular_nonagon_sides / regular_nonagon_sides) :=
by
  sorry

#check exterior_angle_regular_nonagon_proof

end exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l843_84337


namespace fraction_sum_equality_l843_84362

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ (i j : ℕ), (1 : ℚ) / n = (1 : ℚ) / i - (1 : ℚ) / (j + 1) :=
by sorry

end fraction_sum_equality_l843_84362


namespace gcd_sum_and_sum_squares_l843_84312

theorem gcd_sum_and_sum_squares (a b : ℕ) (h : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by sorry

end gcd_sum_and_sum_squares_l843_84312


namespace range_of_a_l843_84397

open Real

theorem range_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x₁ : ℝ, x₁ > 0 → ∀ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by sorry

end range_of_a_l843_84397


namespace james_net_income_l843_84307

def rental_rate : ℕ := 20
def monday_wednesday_hours : ℕ := 8
def friday_hours : ℕ := 6
def sunday_hours : ℕ := 5
def maintenance_cost : ℕ := 35
def insurance_fee : ℕ := 15
def rental_days : ℕ := 4

def total_rental_income : ℕ := 
  rental_rate * (2 * monday_wednesday_hours + friday_hours + sunday_hours)

def total_expenses : ℕ := maintenance_cost + insurance_fee * rental_days

def net_income : ℕ := total_rental_income - total_expenses

theorem james_net_income : net_income = 445 := by
  sorry

end james_net_income_l843_84307


namespace min_value_of_f_l843_84351

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -3 :=
sorry

end min_value_of_f_l843_84351


namespace min_sum_squares_l843_84396

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = m :=
by sorry

end min_sum_squares_l843_84396


namespace solution_of_linear_equation_l843_84328

theorem solution_of_linear_equation :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end solution_of_linear_equation_l843_84328


namespace tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l843_84311

/-- The distance a tiger runs to catch a deer under specific conditions -/
theorem tiger_catch_deer_distance (tiger_leaps_behind : ℕ) 
  (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ) (deer_meters_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_meters_per_leap
  let tiger_distance_per_minute := tiger_leaps_per_minute * tiger_meters_per_leap
  let deer_distance_per_minute := deer_leaps_per_minute * deer_meters_per_leap
  let gain_per_minute := tiger_distance_per_minute - deer_distance_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_distance_per_minute

/-- The distance a tiger runs to catch a deer is 800 meters under the given conditions -/
theorem tiger_catch_deer_distance_is_800 : 
  tiger_catch_deer_distance 50 5 4 8 5 = 800 := by
  sorry

end tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l843_84311


namespace pumpkin_multiple_l843_84360

theorem pumpkin_multiple (moonglow sunshine : ℕ) (h1 : moonglow = 14) (h2 : sunshine = 54) :
  ∃ x : ℕ, x * moonglow + 12 = sunshine ∧ x = 3 := by
  sorry

end pumpkin_multiple_l843_84360


namespace basketball_team_starters_l843_84300

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 11220 :=
by sorry

end basketball_team_starters_l843_84300


namespace solution_set_when_a_is_2_range_of_a_l843_84335

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |2*x - a|

theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 2} = {x : ℝ | 1/4 < x ∧ x < 5/4} := by sorry

theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 3*a + 2) ↔ -3/2 ≤ a ∧ a ≤ -1/4 := by sorry

end solution_set_when_a_is_2_range_of_a_l843_84335


namespace sum_of_numbers_l843_84333

theorem sum_of_numbers : ∀ (a b : ℤ), 
  a = 9 → 
  b = -a + 2 → 
  a + b = 2 :=
by
  sorry

end sum_of_numbers_l843_84333


namespace rachel_milk_consumption_l843_84392

/-- The amount of milk Rachel drinks given the initial amount and fractions poured and drunk -/
theorem rachel_milk_consumption (initial_milk : ℚ) 
  (h1 : initial_milk = 3 / 7)
  (poured_fraction : ℚ) 
  (h2 : poured_fraction = 1 / 2)
  (drunk_fraction : ℚ)
  (h3 : drunk_fraction = 3 / 4) : 
  drunk_fraction * (poured_fraction * initial_milk) = 9 / 56 := by
  sorry

#check rachel_milk_consumption

end rachel_milk_consumption_l843_84392


namespace f_960_minus_f_640_l843_84361

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(960) - f(640) = 5/8 -/
theorem f_960_minus_f_640 : f 960 - f 640 = 5/8 := by sorry

end f_960_minus_f_640_l843_84361


namespace triangle_area_l843_84332

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
sorry

end triangle_area_l843_84332


namespace sum_of_differences_correct_l843_84322

def number : ℕ := 84125398

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_of_differences (n : ℕ) : ℕ :=
  let ones_thousands := place_value 1 3
  let ones_tens := place_value 1 1
  let eights_hundred_millions := place_value 8 8
  let eights_tens := place_value 8 1
  (eights_hundred_millions - ones_thousands) + (eights_tens - ones_tens)

theorem sum_of_differences_correct :
  sum_of_differences number = 79999070 := by sorry

end sum_of_differences_correct_l843_84322


namespace unique_prime_with_prime_quadratics_l843_84374

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof goes here
  sorry

end unique_prime_with_prime_quadratics_l843_84374


namespace probability_three_red_jellybeans_l843_84319

/-- Represents the probability of selecting exactly 3 red jellybeans from a bowl -/
def probability_three_red (total : ℕ) (red : ℕ) (blue : ℕ) (white : ℕ) : ℚ :=
  let total_combinations := Nat.choose total 4
  let favorable_outcomes := Nat.choose red 3 * Nat.choose (blue + white) 1
  favorable_outcomes / total_combinations

/-- Theorem stating the probability of selecting exactly 3 red jellybeans -/
theorem probability_three_red_jellybeans :
  probability_three_red 15 6 3 6 = 4 / 91 := by
  sorry

#eval probability_three_red 15 6 3 6

end probability_three_red_jellybeans_l843_84319


namespace triangle_area_range_l843_84355

/-- Given an obtuse-angled triangle ABC with side c = 2 and angle B = π/3,
    the area S of the triangle satisfies: S ∈ (0, √3/2) ∪ (2√3, +∞) -/
theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) ∧  -- Obtuse-angled triangle condition
  S = (1 / 2) * a * c * Real.sin B →  -- Area formula
  S ∈ Set.Ioo 0 (Real.sqrt 3 / 2) ∪ Set.Ioi (2 * Real.sqrt 3) :=
by
  sorry

end triangle_area_range_l843_84355


namespace rationalize_denominator_cube_root_l843_84330

theorem rationalize_denominator_cube_root (x : ℝ) (h : x > 0) :
  x / (x^(1/3)) = x^(2/3) :=
by sorry

end rationalize_denominator_cube_root_l843_84330


namespace largest_n_divisibility_equality_l843_84380

/-- Count of integers less than or equal to n divisible by d -/
def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

/-- Count of integers less than or equal to n divisible by either a or b -/
def count_divisible_either (n : ℕ) (a b : ℕ) : ℕ :=
  count_divisible n a + count_divisible n b - count_divisible n (a * b)

theorem largest_n_divisibility_equality : ∀ m : ℕ, m > 65 →
  (count_divisible m 3 ≠ count_divisible_either m 5 7) ∧
  (count_divisible 65 3 = count_divisible_either 65 5 7) :=
by sorry

end largest_n_divisibility_equality_l843_84380


namespace complement_of_A_l843_84347

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Icc 0 2 := by sorry

end complement_of_A_l843_84347


namespace find_unknown_number_l843_84398

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 2, 42] → 
  average = 223 → 
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 684 :=
by sorry

end find_unknown_number_l843_84398


namespace rhombus_area_l843_84345

/-- The area of a rhombus with side length 13 cm and one diagonal 24 cm is 120 cm² -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 13 → diagonal1 = 24 → side ^ 2 = (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 → 
  (diagonal1 * diagonal2) / 2 = 120 := by
  sorry

#check rhombus_area

end rhombus_area_l843_84345


namespace tetrahedron_analogy_l843_84308

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogy

-- Define a structure for a reasoning example
structure ReasoningExample where
  description : String
  type : ReasoningType

-- Define the specific example we're interested in
def tetrahedronExample : ReasoningExample :=
  { description := "Inferring the properties of a tetrahedron in space from the properties of a plane triangle"
  , type := ReasoningType.Analogy }

-- Theorem statement
theorem tetrahedron_analogy :
  tetrahedronExample.type = ReasoningType.Analogy :=
by sorry

end tetrahedron_analogy_l843_84308


namespace mark_car_repair_cost_l843_84339

/-- Calculates the total cost of car repair for Mark -/
theorem mark_car_repair_cost :
  let labor_hours : ℝ := 2
  let labor_rate : ℝ := 75
  let part_cost : ℝ := 150
  let cleaning_hours : ℝ := 1
  let cleaning_rate : ℝ := 60
  let labor_discount : ℝ := 0.1
  let tax_rate : ℝ := 0.08

  let labor_cost := labor_hours * labor_rate
  let discounted_labor := labor_cost * (1 - labor_discount)
  let cleaning_cost := cleaning_hours * cleaning_rate
  let subtotal := discounted_labor + part_cost + cleaning_cost
  let total_cost := subtotal * (1 + tax_rate)

  total_cost = 372.60 := by sorry

end mark_car_repair_cost_l843_84339


namespace complex_equation_square_sum_l843_84389

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i) * i = b - i → a^2 + b^2 = 2 := by sorry

end complex_equation_square_sum_l843_84389


namespace modular_congruence_problem_l843_84387

theorem modular_congruence_problem : ∃ m : ℕ, 
  (215 * 953 + 100) % 50 = m ∧ 0 ≤ m ∧ m < 50 :=
by
  use 45
  sorry

end modular_congruence_problem_l843_84387


namespace fruit_display_total_l843_84326

/-- Fruit display problem -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end fruit_display_total_l843_84326


namespace max_g_given_max_f_l843_84391

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem max_g_given_max_f (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 1, |f a b c x| ≤ 1) →
  (∃ a' b' c', ∀ x ∈ Set.Icc 0 1, |g a' b' c' x| ≤ 8 ∧ 
    ∃ x' ∈ Set.Icc 0 1, |g a' b' c' x'| = 8) :=
sorry

end max_g_given_max_f_l843_84391


namespace non_intersecting_path_count_l843_84378

/-- A path on a grid from (0,0) to (n,n) that can only move top or right -/
def GridPath (n : ℕ) := List (Bool)

/-- Two paths are non-intersecting if they don't share any point except (0,0) and (n,n) -/
def NonIntersecting (n : ℕ) (p1 p2 : GridPath n) : Prop := sorry

/-- The number of non-intersecting pairs of paths from (0,0) to (n,n) -/
def NonIntersectingPathCount (n : ℕ) : ℕ := sorry

theorem non_intersecting_path_count (n : ℕ) : 
  NonIntersectingPathCount n = (Nat.choose (2*n-2) (n-1))^2 - (Nat.choose (2*n-2) (n-2))^2 := by sorry

end non_intersecting_path_count_l843_84378


namespace second_player_wins_l843_84383

/-- Represents a point on an infinite grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the game state --/
structure GameState where
  marked_points : List GridPoint
  current_player : Bool  -- true for first player, false for second player

/-- Checks if a set of points forms a convex polygon --/
def is_convex (points : List GridPoint) : Prop :=
  sorry  -- Implementation details omitted

/-- Checks if a move is valid given the current game state --/
def is_valid_move (state : GameState) (new_point : GridPoint) : Prop :=
  is_convex (new_point :: state.marked_points)

/-- Represents a game strategy --/
def Strategy := GameState → Option GridPoint

/-- Checks if a strategy is winning for the current player --/
def is_winning_strategy (strategy : Strategy) : Prop :=
  sorry  -- Implementation details omitted

/-- The main theorem stating that the second player has a winning strategy --/
theorem second_player_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy ∧
    ∀ (initial_state : GameState),
      initial_state.current_player = false →
      is_winning_strategy (λ state => strategy state) :=
sorry

end second_player_wins_l843_84383


namespace solution_set_inequality_l843_84305

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 0 ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end solution_set_inequality_l843_84305


namespace triangle_proof_l843_84388

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.c - t.b = 2 * t.a * Real.cos t.B)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3)
  (h3 : t.c = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.B = Real.pi / 2 := by
  sorry

#check triangle_proof

end triangle_proof_l843_84388


namespace geometric_sequence_property_l843_84390

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 4, a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) (h_a4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end geometric_sequence_property_l843_84390


namespace absolute_value_inequality_l843_84346

theorem absolute_value_inequality (y : ℝ) : 
  (2 ≤ |y - 5| ∧ |y - 5| ≤ 8) ↔ ((-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13)) :=
by sorry

end absolute_value_inequality_l843_84346


namespace volunteer_transfer_l843_84371

theorem volunteer_transfer (initial_group1 initial_group2 : ℕ) 
  (h1 : initial_group1 = 20)
  (h2 : initial_group2 = 26) :
  ∃ x : ℚ, x = 32 / 3 ∧ 
    initial_group1 + x = 2 * (initial_group2 - x) := by
  sorry

end volunteer_transfer_l843_84371


namespace basketball_score_proof_l843_84386

theorem basketball_score_proof (joe tim ken : ℕ) 
  (h1 : tim = joe + 20)
  (h2 : tim * 2 = ken)
  (h3 : joe + tim + ken = 100) :
  tim = 30 := by
  sorry

end basketball_score_proof_l843_84386


namespace banana_group_size_l843_84349

theorem banana_group_size (total_bananas : ℕ) (num_groups : ℕ) (h1 : total_bananas = 180) (h2 : num_groups = 10) :
  total_bananas / num_groups = 18 := by
  sorry

end banana_group_size_l843_84349


namespace max_f_value_l843_84317

/-- The function f(n) is the greatest common divisor of all numbers 
    obtained by permuting the digits of n, including permutations 
    with leading zeroes. -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum value of f(n) for positive integers n 
    where f(n) ≠ n is 81. -/
theorem max_f_value : 
  (∃ (n : ℕ+), f n = 81 ∧ f n ≠ n) ∧ 
  (∀ (n : ℕ+), f n ≠ n → f n ≤ 81) :=
sorry

end max_f_value_l843_84317


namespace weight_increase_percentage_l843_84302

/-- Calculates the percentage increase in weight on the lowering portion of an exercise machine. -/
theorem weight_increase_percentage
  (num_plates : ℕ)
  (plate_weight : ℝ)
  (lowered_weight : ℝ)
  (h1 : num_plates = 10)
  (h2 : plate_weight = 30)
  (h3 : lowered_weight = 360) :
  ((lowered_weight - num_plates * plate_weight) / (num_plates * plate_weight)) * 100 = 20 := by
sorry

end weight_increase_percentage_l843_84302


namespace metal_argument_is_deductive_l843_84306

-- Define the structure of a logical argument
structure Argument where
  premises : List String
  conclusion : String

-- Define the property of being deductive
def is_deductive (arg : Argument) : Prop :=
  ∀ (world : Type) (interpretation : String → world → Prop),
    (∀ p ∈ arg.premises, ∀ w, interpretation p w) →
    (∀ w, interpretation arg.conclusion w)

-- Define the argument about metals and uranium
def metal_argument : Argument :=
  { premises := ["All metals can conduct electricity", "Uranium is a metal"],
    conclusion := "Uranium can conduct electricity" }

-- Theorem statement
theorem metal_argument_is_deductive :
  is_deductive metal_argument :=
sorry

end metal_argument_is_deductive_l843_84306


namespace tens_digit_of_13_pow_1987_l843_84340

theorem tens_digit_of_13_pow_1987 : ∃ n : ℕ, 13^1987 ≡ 10 * n + 7 [ZMOD 100] := by
  sorry

end tens_digit_of_13_pow_1987_l843_84340


namespace magnitude_of_vector_sum_l843_84327

/-- Given vectors e₁ and e₂ forming an angle of 2π/3, prove that |e₁ + 2e₂| = √3 -/
theorem magnitude_of_vector_sum (e₁ e₂ : ℝ × ℝ) : 
  e₁ • e₁ = 1 → 
  e₂ • e₂ = 1 → 
  e₁ • e₂ = -1/2 → 
  let a := e₁ + 2 • e₂ 
  ‖a‖ = Real.sqrt 3 := by
  sorry

end magnitude_of_vector_sum_l843_84327


namespace sequence_properties_l843_84375

/-- Given a sequence {aₙ} with sum Sₙ satisfying Sₙ = t(Sₙ - aₙ + 1) where t ≠ 0 and t ≠ 1,
    and a sequence {bₙ} defined as bₙ = aₙ² + Sₙ · aₙ which is geometric,
    prove that {aₙ} is geometric and find the general term of {bₙ}. -/
theorem sequence_properties (t : ℝ) (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : t ≠ 0) (h2 : t ≠ 1)
  (h3 : ∀ n, S n = t * (S n - a n + 1))
  (h4 : ∀ n, b n = a n ^ 2 + S n * a n)
  (h5 : ∃ q, ∀ n, b (n + 1) = q * b n) :
  (∀ n, a (n + 1) = t * a n) ∧
  (∀ n, b n = t^(n + 1) * (2 * t + 1)^(n - 1) / 2^(n - 2)) :=
sorry

end sequence_properties_l843_84375


namespace foreign_language_speakers_l843_84304

theorem foreign_language_speakers (M F : ℕ) : 
  M = F →  -- number of male students equals number of female students
  (3 : ℚ) / 5 * M + (2 : ℚ) / 3 * F = (19 : ℚ) / 30 * (M + F) := by
  sorry

end foreign_language_speakers_l843_84304


namespace cube_side_length_is_one_l843_84331

/-- The side length of a cube -/
def m : ℕ := sorry

/-- The number of blue faces on the unit cubes -/
def blue_faces : ℕ := 2 * m^2

/-- The total number of faces on all unit cubes -/
def total_faces : ℕ := 6 * m^3

/-- The theorem stating that if one-third of the total faces are blue, then m = 1 -/
theorem cube_side_length_is_one : 
  (blue_faces : ℚ) / total_faces = 1 / 3 → m = 1 := by sorry

end cube_side_length_is_one_l843_84331


namespace meatballs_stolen_l843_84393

theorem meatballs_stolen (original_total original_beef original_chicken original_pork remaining_beef remaining_chicken remaining_pork : ℕ) :
  original_total = 30 →
  original_beef = 15 →
  original_chicken = 10 →
  original_pork = 5 →
  remaining_beef = 10 →
  remaining_chicken = 10 →
  remaining_pork = 5 →
  original_beef - remaining_beef = 5 :=
by sorry

end meatballs_stolen_l843_84393


namespace negation_of_positive_square_l843_84372

theorem negation_of_positive_square (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) := by sorry

end negation_of_positive_square_l843_84372


namespace time_period_is_three_years_l843_84364

/-- Calculates the time period for which a sum is due given the banker's gain, banker's discount, and interest rate. -/
def calculate_time_period (bankers_gain : ℚ) (bankers_discount : ℚ) (interest_rate : ℚ) : ℚ :=
  let true_discount := bankers_discount - bankers_gain
  let ratio := bankers_discount / true_discount
  (ratio - 1) / (interest_rate / 100)

/-- Theorem stating that given the specific values in the problem, the time period is 3 years. -/
theorem time_period_is_three_years :
  let bankers_gain : ℚ := 90
  let bankers_discount : ℚ := 340
  let interest_rate : ℚ := 12
  calculate_time_period bankers_gain bankers_discount interest_rate = 3 := by
  sorry

#eval calculate_time_period 90 340 12

end time_period_is_three_years_l843_84364


namespace grain_distance_equation_l843_84303

/-- The distance between the two towers in feet -/
def tower_distance : ℝ := 400

/-- The height of the church tower in feet -/
def church_tower_height : ℝ := 180

/-- The height of the cathedral tower in feet -/
def cathedral_tower_height : ℝ := 240

/-- The speed of the bird from the church tower in ft/s -/
def church_bird_speed : ℝ := 20

/-- The speed of the bird from the cathedral tower in ft/s -/
def cathedral_bird_speed : ℝ := 25

/-- The theorem stating the equation for the distance of the grain from the church tower -/
theorem grain_distance_equation (x : ℝ) :
  x ≥ 0 ∧ x ≤ tower_distance →
  25 * x = 20 * (tower_distance - x) :=
sorry

end grain_distance_equation_l843_84303


namespace sum_of_solutions_quadratic_l843_84385

theorem sum_of_solutions_quadratic (z : ℂ) : 
  (z^2 = 16*z - 10) → (∃ (z1 z2 : ℂ), z1^2 = 16*z1 - 10 ∧ z2^2 = 16*z2 - 10 ∧ z1 + z2 = 16) :=
by
  sorry

end sum_of_solutions_quadratic_l843_84385


namespace water_in_tank_after_rain_l843_84377

/-- Calculates the final amount of water in a tank after rainfall, considering inflow, leakage, and evaporation. -/
def final_water_amount (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_water + (inflow_rate - leakage_rate - evaporation_rate) * duration

/-- Theorem stating that the final amount of water in the tank is 226 L -/
theorem water_in_tank_after_rain (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) :
  initial_water = 100 ∧
  inflow_rate = 2 ∧
  leakage_rate = 0.5 ∧
  evaporation_rate = 0.1 ∧
  duration = 90 →
  final_water_amount initial_water inflow_rate leakage_rate evaporation_rate duration = 226 :=
by sorry

end water_in_tank_after_rain_l843_84377


namespace donut_sharing_l843_84399

def total_donuts (delta_donuts : ℕ) (gamma_donuts : ℕ) (beta_multiplier : ℕ) : ℕ :=
  delta_donuts + gamma_donuts + (beta_multiplier * gamma_donuts)

theorem donut_sharing :
  let delta_donuts : ℕ := 8
  let gamma_donuts : ℕ := 8
  let beta_multiplier : ℕ := 3
  total_donuts delta_donuts gamma_donuts beta_multiplier = 40 := by
  sorry

end donut_sharing_l843_84399


namespace circle_equation_proof_l843_84329

/-- Prove that the given equation represents a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  ((x - center.1)^2 + (y - center.2)^2 = 
   (point.1 - center.1)^2 + (point.2 - center.2)^2) ↔
  ((x - 2)^2 + (y + 1)^2 = 25) :=
by sorry


end circle_equation_proof_l843_84329


namespace work_done_cyclic_process_work_done_equals_665J_l843_84315

/-- Represents a point in the P-T diagram -/
structure Point where
  pressure : ℝ
  temperature : ℝ

/-- Represents the cyclic process abca -/
structure CyclicProcess where
  a : Point
  b : Point
  c : Point

/-- The gas constant -/
def R : ℝ := 8.314

/-- Theorem: Work done in the cyclic process -/
theorem work_done_cyclic_process (process : CyclicProcess) : ℝ :=
  let T₀ : ℝ := 320
  have h1 : process.a.temperature = T₀ := by sorry
  have h2 : process.c.temperature = T₀ := by sorry
  have h3 : process.a.pressure = process.c.pressure / 2 := by sorry
  have h4 : process.b.pressure = process.a.pressure := by sorry
  have h5 : (process.b.temperature - process.a.temperature) * process.a.pressure > 0 := by sorry
  (1/2) * R * T₀

/-- Main theorem: The work done is equal to 665 J -/
theorem work_done_equals_665J (process : CyclicProcess) : 
  work_done_cyclic_process process = 665 := by sorry

end work_done_cyclic_process_work_done_equals_665J_l843_84315


namespace regular_hexagon_perimeter_l843_84363

/-- The perimeter of a regular hexagon with side length 5 meters is 30 meters. -/
theorem regular_hexagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 5 → 
  (6 : ℝ) * side_length = 30 := by
  sorry

end regular_hexagon_perimeter_l843_84363


namespace pear_peach_weight_equivalence_l843_84365

/-- If 9 pears weigh the same as 6 peaches, then 36 pears weigh the same as 24 peaches. -/
theorem pear_peach_weight_equivalence :
  ∀ (pear_weight peach_weight : ℝ),
  9 * pear_weight = 6 * peach_weight →
  36 * pear_weight = 24 * peach_weight :=
by
  sorry


end pear_peach_weight_equivalence_l843_84365


namespace volume_ratio_equals_edge_product_ratio_l843_84342

/-- Represent a tetrahedron with vertex O and edges OA, OB, OC -/
structure Tetrahedron where
  a : ℝ  -- length of OA
  b : ℝ  -- length of OB
  c : ℝ  -- length of OC
  volume : ℝ  -- volume of the tetrahedron

/-- Two tetrahedrons with congruent trihedral angles at O and O' -/
def CongruentTrihedralTetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  -- We don't explicitly define the congruence, as it's given in the problem statement
  True

theorem volume_ratio_equals_edge_product_ratio
  (t1 t2 : Tetrahedron)
  (h : CongruentTrihedralTetrahedrons t1 t2) :
  t2.volume / t1.volume = (t2.a * t2.b * t2.c) / (t1.a * t1.b * t1.c) := by
  sorry

end volume_ratio_equals_edge_product_ratio_l843_84342


namespace binomial_properties_l843_84316

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialRV)

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The probability of getting 0 successes in a binomial distribution -/
def prob_zero (ξ : BinomialRV) : ℝ := (1 - ξ.p) ^ ξ.n

theorem binomial_properties (ξ : BinomialRV) 
  (h2 : 3 * expectation ξ + 2 = 9.2)
  (h3 : 9 * variance ξ = 12.96) :
  ξ.n = 6 ∧ ξ.p = 0.4 ∧ prob_zero ξ = 0.6^6 := by sorry

end binomial_properties_l843_84316


namespace five_digit_four_digit_division_l843_84356

theorem five_digit_four_digit_division (a b : ℕ) : 
  (a * 11111 = 16 * (b * 1111) + (a * 1111 - 16 * (b * 111) + 2000)) →
  (a ≤ 9) →
  (b ≤ 9) →
  (a * 11111 ≥ b * 1111) →
  (a * 1111 ≥ b * 111) →
  (a = 5 ∧ b = 3) := by
sorry

end five_digit_four_digit_division_l843_84356


namespace inequality_reversal_l843_84381

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end inequality_reversal_l843_84381


namespace gdp_scientific_notation_l843_84350

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in yuan -/
def gdp : ℝ := 338.8e9

theorem gdp_scientific_notation :
  toScientificNotation gdp = ScientificNotation.mk 3.388 10 (by norm_num) :=
sorry

end gdp_scientific_notation_l843_84350


namespace four_numbers_solution_l843_84344

/-- A sequence of four real numbers satisfying the given conditions -/
structure FourNumbers where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  arithmetic_seq : b - a = c - b
  geometric_seq : c * c = b * d
  sum_first_last : a + d = 16
  sum_middle : b + c = 12

/-- The theorem stating that there are only two possible sets of four numbers satisfying the conditions -/
theorem four_numbers_solution (x : FourNumbers) :
  (x.a = 0 ∧ x.b = 4 ∧ x.c = 8 ∧ x.d = 16) ∨
  (x.a = 15 ∧ x.b = 9 ∧ x.c = 3 ∧ x.d = 1) :=
by sorry

end four_numbers_solution_l843_84344


namespace inequality_holds_l843_84301

theorem inequality_holds (x : ℝ) : 
  -1/2 ≤ x ∧ x < 45/8 → (4 * x^2) / ((1 - Real.sqrt (1 + 2*x))^2) < 2*x + 9 := by
  sorry

end inequality_holds_l843_84301


namespace optimal_garden_length_l843_84370

/-- Represents the length of the side perpendicular to the greenhouse -/
def x : ℝ := sorry

/-- The total amount of fencing available -/
def total_fence : ℝ := 280

/-- The maximum allowed length of the side parallel to the greenhouse -/
def max_parallel_length : ℝ := 300

/-- The length of the side parallel to the greenhouse -/
def parallel_length (x : ℝ) : ℝ := total_fence - 2 * x

/-- The area of the garden as a function of x -/
def garden_area (x : ℝ) : ℝ := x * (parallel_length x)

/-- Theorem stating that the optimal length of the side parallel to the greenhouse is 140 feet -/
theorem optimal_garden_length :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    parallel_length x ≤ max_parallel_length ∧ 
    parallel_length x = 140 ∧ 
    ∀ (y : ℝ), y > 0 ∧ parallel_length y ≤ max_parallel_length → 
      garden_area x ≥ garden_area y :=
by sorry

end optimal_garden_length_l843_84370


namespace quadratic_roots_condition_l843_84323

theorem quadratic_roots_condition (c : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + c = 0 ∧ 
  x₂^2 - 2*x₂ + c = 0 ∧ 
  7*x₂ - 4*x₁ = 47 →
  c = -15 := by
sorry

end quadratic_roots_condition_l843_84323


namespace equation_roots_l843_84367

theorem equation_roots : 
  ∃ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁ = -1/12 ∧ x₂ = 1/2 ∧ x₃ = (5 + Complex.I * Real.sqrt 39) / 24 ∧ x₄ = (5 - Complex.I * Real.sqrt 39) / 24) ∧
    (∀ x : ℂ, (12*x - 1)*(6*x - 1)*(4*x - 1)*(3*x - 1) = 5 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end equation_roots_l843_84367


namespace initial_student_count_l843_84395

/-- Given the initial average weight, new average weight after admitting a new student,
    and the weight of the new student, prove that the initial number of students is 19. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.8)
  (h3 : new_student_weight = 11) :
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 := by
  sorry

#check initial_student_count

end initial_student_count_l843_84395


namespace probability_one_from_a_is_11_21_l843_84366

/-- Represents the number of factories in each area -/
structure FactoryCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Represents the number of factories selected from each area -/
structure SelectedCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Calculates the probability of selecting at least one factory from area A
    when choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (counts : FactoryCounts) (selected : SelectedCounts) : Rat :=
  sorry

/-- The main theorem stating the probability is 11/21 given the specific conditions -/
theorem probability_one_from_a_is_11_21 :
  let counts : FactoryCounts := ⟨18, 27, 18⟩
  let selected : SelectedCounts := ⟨2, 3, 2⟩
  probabilityAtLeastOneFromA counts selected = 11 / 21 := by sorry

end probability_one_from_a_is_11_21_l843_84366


namespace team_a_more_uniform_l843_84348

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  mean_height : ℝ
  height_variance : ℝ

/-- Define the concept of height uniformity -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.height_variance < team2.height_variance

theorem team_a_more_uniform : 
  ∀ (team_a team_b : DanceTeam),
    team_a.name = "A" →
    team_b.name = "B" →
    team_a.mean_height = 1.65 →
    team_b.mean_height = 1.65 →
    team_a.height_variance = 1.5 →
    team_b.height_variance = 2.4 →
    more_uniform_heights team_a team_b :=
by sorry

end team_a_more_uniform_l843_84348


namespace triangle_area_after_10_seconds_l843_84353

/-- Represents the position of a runner at time t -/
def RunnerPosition (t : ℝ) := ℝ × ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Represents the position of a runner over time -/
structure Runner where
  initialPos : ℝ × ℝ
  velocity : ℝ

/-- Calculates the position of a runner at time t -/
def runnerPosition (r : Runner) (t : ℝ) : RunnerPosition t := sorry

theorem triangle_area_after_10_seconds
  (a b c : Runner)
  (h1 : triangleArea (runnerPosition a 0) (runnerPosition b 0) (runnerPosition c 0) = 2)
  (h2 : triangleArea (runnerPosition a 5) (runnerPosition b 5) (runnerPosition c 5) = 3) :
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 4) ∨
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 8) := by
  sorry

end triangle_area_after_10_seconds_l843_84353


namespace water_bottle_cost_l843_84373

def initial_amount : ℕ := 50
def final_amount : ℕ := 44
def num_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def num_water_bottles : ℕ := 2

theorem water_bottle_cost :
  (initial_amount - final_amount - num_baguettes * cost_per_baguette) / num_water_bottles = 1 :=
by sorry

end water_bottle_cost_l843_84373


namespace range_of_a_l843_84325

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |x - a| > x - 1) ↔ (a < 1 ∨ a > 3) := by
sorry

end range_of_a_l843_84325


namespace triangle_area_is_nine_l843_84338

-- Define the slopes and intersection point
def slope1 : ℚ := 1/3
def slope2 : ℚ := 3
def intersection : ℚ × ℚ := (1, 1)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the triangle area function
def triangle_area (A B C : ℚ × ℚ) : ℚ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement
theorem triangle_area_is_nine :
  ∃ A B C : ℚ × ℚ,
    A = intersection ∧
    line3 B.1 B.2 ∧
    line3 C.1 C.2 ∧
    B.2 = line1 B.1 ∧
    C.2 = line2 C.1 ∧
    triangle_area A B C = 9 :=
by sorry

end triangle_area_is_nine_l843_84338


namespace vector_magnitude_l843_84357

def a : ℝ × ℝ := (1, 2)
def b : ℝ → ℝ × ℝ := λ t ↦ (2, t)

theorem vector_magnitude (t : ℝ) (h : a.1 * (b t).1 + a.2 * (b t).2 = 0) :
  Real.sqrt ((b t).1^2 + (b t).2^2) = Real.sqrt 5 := by
  sorry

end vector_magnitude_l843_84357


namespace x_equals_four_l843_84359

theorem x_equals_four : ∃! x : ℤ, 2^4 + x = 3^3 - 7 :=
by
  sorry

end x_equals_four_l843_84359


namespace contrapositive_equivalence_l843_84313

theorem contrapositive_equivalence (p q : Prop) : (p → q) → (¬q → ¬p) := by
  sorry

end contrapositive_equivalence_l843_84313


namespace sum_of_perpendiculars_not_constant_l843_84321

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  a : Point
  b : Point
  c : Point
  equalSideLength : ℝ
  baseSideLength : ℝ

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : IsoscelesTriangle) : Prop := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Theorem: The sum of perpendiculars is not constant for all points inside the triangle -/
theorem sum_of_perpendiculars_not_constant (t : IsoscelesTriangle)
  (h1 : t.equalSideLength = 10)
  (h2 : t.baseSideLength = 8) :
  ∃ p1 p2 : Point,
    isInside p1 t ∧ isInside p2 t ∧
    perpendicularDistance p1 t.a t.b + perpendicularDistance p1 t.b t.c + perpendicularDistance p1 t.c t.a ≠
    perpendicularDistance p2 t.a t.b + perpendicularDistance p2 t.b t.c + perpendicularDistance p2 t.c t.a :=
by sorry

end sum_of_perpendiculars_not_constant_l843_84321


namespace fraction_inequality_l843_84369

theorem fraction_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hca : c > a) (hab : a > b) : 
  a / (c - a) > b / (c - b) := by
sorry

end fraction_inequality_l843_84369


namespace absolute_value_of_h_l843_84376

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 80) → 
  |h| = 2 := by
sorry

end absolute_value_of_h_l843_84376


namespace sqrt_p_div_sqrt_q_l843_84379

theorem sqrt_p_div_sqrt_q (p q : ℝ) (h : (1/3)^2 + (1/4)^2 = ((25*p)/(61*q)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt p / Real.sqrt q = 5/2 := by
  sorry

end sqrt_p_div_sqrt_q_l843_84379


namespace bobby_blocks_l843_84334

theorem bobby_blocks (initial_blocks final_blocks given_blocks : ℕ) 
  (h1 : final_blocks = initial_blocks + given_blocks)
  (h2 : final_blocks = 8)
  (h3 : given_blocks = 6) : 
  initial_blocks = 2 := by sorry

end bobby_blocks_l843_84334


namespace square_area_perimeter_ratio_l843_84309

theorem square_area_perimeter_ratio :
  ∀ s₁ s₂ : ℝ,
  s₁ > 0 → s₂ > 0 →
  s₁^2 / s₂^2 = 16 / 81 →
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
sorry

end square_area_perimeter_ratio_l843_84309


namespace ticket_price_increase_l843_84314

theorem ticket_price_increase (last_year_income : ℝ) (club_share_last_year : ℝ) 
  (club_share_this_year : ℝ) (rental_cost : ℝ) : 
  club_share_last_year = 0.1 * last_year_income →
  rental_cost = 0.9 * last_year_income →
  club_share_this_year = 0.2 →
  (((rental_cost / (1 - club_share_this_year)) / last_year_income) - 1) * 100 = 12.5 := by
  sorry

end ticket_price_increase_l843_84314


namespace mr_grey_polo_shirts_l843_84336

/-- Represents the purchase of gifts by Mr. Grey -/
structure GiftPurchase where
  polo_shirt_price : ℕ
  necklace_price : ℕ
  computer_game_price : ℕ
  necklace_count : ℕ
  rebate : ℕ
  total_cost : ℕ

/-- Calculates the number of polo shirts bought given the gift purchase details -/
def calculate_polo_shirts (purchase : GiftPurchase) : ℕ :=
  (purchase.total_cost + purchase.rebate - purchase.necklace_price * purchase.necklace_count - purchase.computer_game_price) / purchase.polo_shirt_price

/-- Theorem stating that Mr. Grey bought 3 polo shirts -/
theorem mr_grey_polo_shirts :
  let purchase : GiftPurchase := {
    polo_shirt_price := 26,
    necklace_price := 83,
    computer_game_price := 90,
    necklace_count := 2,
    rebate := 12,
    total_cost := 322
  }
  calculate_polo_shirts purchase = 3 := by
  sorry

end mr_grey_polo_shirts_l843_84336


namespace remaining_macaroons_weight_l843_84368

theorem remaining_macaroons_weight
  (coconut_count : ℕ) (coconut_weight : ℕ) (almond_count : ℕ) (almond_weight : ℕ)
  (coconut_bags : ℕ) (almond_bags : ℕ) :
  coconut_count = 12 →
  coconut_weight = 5 →
  almond_count = 8 →
  almond_weight = 8 →
  coconut_bags = 4 →
  almond_bags = 2 →
  (coconut_count * coconut_weight - (coconut_count / coconut_bags) * coconut_weight) +
  (almond_count * almond_weight - (almond_count / almond_bags) * almond_weight / 2) = 93 :=
by sorry

end remaining_macaroons_weight_l843_84368


namespace position_of_three_fifths_l843_84310

def sequence_sum (n : ℕ) : ℕ := n - 1

def position_in_group (n m : ℕ) : ℕ := 
  (sequence_sum n * (sequence_sum n + 1)) / 2 + m

theorem position_of_three_fifths : 
  position_in_group 8 3 = 24 := by sorry

end position_of_three_fifths_l843_84310


namespace vasya_multiplication_error_l843_84394

-- Define a structure for a two-digit number
structure TwoDigitNumber where
  tens : Fin 10
  ones : Fin 10
  different : tens ≠ ones

-- Define a structure for the result DDEE
structure ResultDDEE where
  d : Fin 10
  e : Fin 10
  different : d ≠ e

-- Define the main theorem
theorem vasya_multiplication_error 
  (ab vg : TwoDigitNumber) 
  (result : ResultDDEE) 
  (h1 : ab.tens ≠ vg.tens)
  (h2 : ab.tens ≠ vg.ones)
  (h3 : ab.ones ≠ vg.tens)
  (h4 : ab.ones ≠ vg.ones)
  (h5 : (ab.tens * 10 + ab.ones) * (vg.tens * 10 + vg.ones) = result.d * 1000 + result.d * 100 + result.e * 10 + result.e) :
  False :=
sorry

end vasya_multiplication_error_l843_84394


namespace remainder_777_444_mod_13_l843_84318

theorem remainder_777_444_mod_13 : 777^444 % 13 = 1 := by
  sorry

end remainder_777_444_mod_13_l843_84318
