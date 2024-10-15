import Mathlib

namespace NUMINAMATH_CALUDE_match_total_weight_l1975_197591

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 20

/-- The number of weights used in each setup -/
def num_weights : ℕ := 2

/-- The total weight lifted with the original setup in pounds -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of lifts required with the new weights to match the total weight -/
def required_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem match_total_weight : required_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_match_total_weight_l1975_197591


namespace NUMINAMATH_CALUDE_pet_store_parrot_count_l1975_197576

theorem pet_store_parrot_count (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 48)
  (h2 : num_cages = 6)
  (h3 : parakeets_per_cage = 2) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 6 := by
  sorry

#check pet_store_parrot_count

end NUMINAMATH_CALUDE_pet_store_parrot_count_l1975_197576


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1975_197589

theorem inequality_holds_iff_k_in_range (k : ℝ) : 
  (k > 0 ∧ 
   ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ + x₂ = k → 
   (1/x₁ - x₁) * (1/x₂ - x₂) ≥ (k/2 - 2/k)^2) 
  ↔ 
  (0 < k ∧ k ≤ 2 * Real.sqrt (Real.sqrt 5 - 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1975_197589


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1975_197517

theorem absolute_value_simplification : |(-4^2 + 7)| = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1975_197517


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1975_197592

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (3 - 2 * i * x = 5 + 4 * i * x) ∧ (x = i / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1975_197592


namespace NUMINAMATH_CALUDE_runners_meet_time_l1975_197503

def carla_lap_time : ℕ := 5
def jose_lap_time : ℕ := 8
def mary_lap_time : ℕ := 10

theorem runners_meet_time :
  Nat.lcm (Nat.lcm carla_lap_time jose_lap_time) mary_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1975_197503


namespace NUMINAMATH_CALUDE_max_value_product_l1975_197553

theorem max_value_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/1296 ∧
  ∃ a b c, (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/1296 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1975_197553


namespace NUMINAMATH_CALUDE_degree_of_P_l1975_197563

/-- The polynomial in question -/
def P (a b : ℚ) : ℚ := 2/3 * a * b^2 + 4/3 * a^3 * b + 1/3

/-- The degree of a polynomial -/
def polynomial_degree (p : ℚ → ℚ → ℚ) : ℕ :=
  sorry  -- Definition of polynomial degree

theorem degree_of_P : polynomial_degree P = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_P_l1975_197563


namespace NUMINAMATH_CALUDE_total_games_in_our_league_l1975_197581

/-- Represents a sports league with sub-leagues and playoffs -/
structure SportsLeague where
  total_teams : Nat
  num_sub_leagues : Nat
  teams_per_sub_league : Nat
  games_against_each_team : Nat
  teams_advancing : Nat

/-- Calculates the total number of games in the entire season -/
def total_games (league : SportsLeague) : Nat :=
  let sub_league_games := league.num_sub_leagues * (league.teams_per_sub_league * (league.teams_per_sub_league - 1) / 2 * league.games_against_each_team)
  let playoff_teams := league.num_sub_leagues * league.teams_advancing
  let playoff_games := playoff_teams * (playoff_teams - 1) / 2
  sub_league_games + playoff_games

/-- The specific league configuration -/
def our_league : SportsLeague :=
  { total_teams := 100
  , num_sub_leagues := 5
  , teams_per_sub_league := 20
  , games_against_each_team := 6
  , teams_advancing := 4 }

/-- Theorem stating that the total number of games in our league is 5890 -/
theorem total_games_in_our_league : total_games our_league = 5890 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_our_league_l1975_197581


namespace NUMINAMATH_CALUDE_graduation_ceremony_chairs_l1975_197526

/-- Calculates the number of chairs needed for a graduation ceremony --/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  let parent_chairs := graduates * parents_per_graduate
  let graduate_and_parent_chairs := graduates + parent_chairs
  let administrator_chairs := teachers / 2
  graduate_and_parent_chairs + teachers + administrator_chairs

theorem graduation_ceremony_chairs :
  chairs_needed 50 2 20 = 180 :=
by sorry

end NUMINAMATH_CALUDE_graduation_ceremony_chairs_l1975_197526


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_l1975_197594

/-- Given a plane passing through (d,e,f) and intersecting the coordinate axes at D, E, F,
    with (u,v,w) as the center of the sphere through D, E, F, and the origin,
    prove that d/u + e/v + f/w = 2 -/
theorem plane_sphere_intersection (d e f u v w : ℝ) : 
  (∃ (δ ε ϕ : ℝ), 
    δ ≠ 0 ∧ ε ≠ 0 ∧ ϕ ≠ 0 ∧
    u^2 + v^2 + w^2 = (u - δ)^2 + v^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + (v - ε)^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + v^2 + (w - ϕ)^2 ∧
    d / δ + e / ε + f / ϕ = 1) →
  d / u + e / v + f / w = 2 :=
by sorry

end NUMINAMATH_CALUDE_plane_sphere_intersection_l1975_197594


namespace NUMINAMATH_CALUDE_students_with_glasses_and_watches_l1975_197516

theorem students_with_glasses_and_watches (n : ℕ) 
  (glasses : ℚ) (watches : ℚ) (neither : ℚ) (both : ℕ) :
  glasses = 3/5 →
  watches = 5/6 →
  neither = 1/10 →
  (n : ℚ) * glasses + (n : ℚ) * watches - (n : ℚ) + (n : ℚ) * neither = (n : ℚ) →
  both = 16 :=
by
  sorry

#check students_with_glasses_and_watches

end NUMINAMATH_CALUDE_students_with_glasses_and_watches_l1975_197516


namespace NUMINAMATH_CALUDE_function_non_negative_implies_a_value_l1975_197515

/-- Given a function f and a real number a, proves that if f satisfies certain conditions, then a = 2/3 -/
theorem function_non_negative_implies_a_value (a : ℝ) :
  (∀ x > 1 - 2*a, (Real.exp (x - a) - 1) * Real.log (x + 2*a - 1) ≥ 0) →
  a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_a_value_l1975_197515


namespace NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_7_mod_12_l1975_197504

theorem prime_between_30_and_40_with_remainder_7_mod_12 (n : ℕ) : 
  Prime n → 
  30 < n → 
  n < 40 → 
  n % 12 = 7 → 
  n = 31 := by
sorry

end NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_7_mod_12_l1975_197504


namespace NUMINAMATH_CALUDE_point_M_properties_segment_MN_length_l1975_197528

def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m + 3)

theorem point_M_properties (m : ℝ) :
  (M m).1 > 0 ∧ (M m).2 > 0 ∧  -- M is in the first quadrant
  (M m).2 = 2 * (M m).1  -- distance to x-axis is twice distance to y-axis
  → m = 1/3 := by sorry

def N : ℝ × ℝ := (2, 1)

theorem segment_MN_length (m : ℝ) :
  (M m).2 = N.2  -- MN is parallel to x-axis
  → |N.1 - (M m).1| = 5 := by sorry

end NUMINAMATH_CALUDE_point_M_properties_segment_MN_length_l1975_197528


namespace NUMINAMATH_CALUDE_range_of_m_l1975_197549

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2)
  (h_ineq : ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m) :
  ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m → -9 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1975_197549


namespace NUMINAMATH_CALUDE_fraction_equality_l1975_197525

theorem fraction_equality (x m : ℝ) (h : x ≠ 0) :
  x / (x^2 - m*x + 1) = 1 →
  x^3 / (x^6 - m^3*x^3 + 1) = 1 / (3*m^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1975_197525


namespace NUMINAMATH_CALUDE_profit_after_five_days_days_for_ten_thousand_profit_l1975_197556

/-- Profit calculation function -/
def profit (x : ℝ) : ℝ :=
  (50 + 2*x) * (700 - 15*x) - 700 * 40 - 50 * x

/-- Theorem for profit after 5 days -/
theorem profit_after_five_days : profit 5 = 9250 := by sorry

/-- Theorem for days to store for 10,000 yuan profit -/
theorem days_for_ten_thousand_profit :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ profit x = 10000 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_profit_after_five_days_days_for_ten_thousand_profit_l1975_197556


namespace NUMINAMATH_CALUDE_sin_sum_identity_l1975_197560

theorem sin_sum_identity : 
  Real.sin (π/4) * Real.sin (7*π/12) + Real.sin (π/4) * Real.sin (π/12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l1975_197560


namespace NUMINAMATH_CALUDE_a_plus_b_and_abs_a_minus_b_l1975_197580

theorem a_plus_b_and_abs_a_minus_b (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : b^2 = 25) 
  (h3 : a * b < 0) : 
  ((a + b = 3) ∨ (a + b = -3)) ∧ |a - b| = 7 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_and_abs_a_minus_b_l1975_197580


namespace NUMINAMATH_CALUDE_fourth_grade_final_count_l1975_197507

/-- Calculates the final number of students in a class given the initial count and changes throughout the year. -/
def final_student_count (initial : ℕ) (left_first : ℕ) (joined_first : ℕ) (left_second : ℕ) (joined_second : ℕ) : ℕ :=
  initial - left_first + joined_first - left_second + joined_second

/-- Theorem stating that the final number of students in the fourth grade class is 37. -/
theorem fourth_grade_final_count : 
  final_student_count 35 6 4 3 7 = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_final_count_l1975_197507


namespace NUMINAMATH_CALUDE_sandwich_bread_slices_l1975_197529

theorem sandwich_bread_slices 
  (total_sandwiches : ℕ) 
  (bread_packs : ℕ) 
  (slices_per_pack : ℕ) 
  (h1 : total_sandwiches = 8)
  (h2 : bread_packs = 4)
  (h3 : slices_per_pack = 4) :
  (bread_packs * slices_per_pack) / total_sandwiches = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_slices_l1975_197529


namespace NUMINAMATH_CALUDE_total_plants_grown_l1975_197508

def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10
def tomatoes_per_packet : ℕ := 16
def peas_per_packet : ℕ := 20

def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6
def tomato_packets : ℕ := 5
def pea_packets : ℕ := 7

def spring_growth_rate : ℚ := 7/10
def summer_growth_rate : ℚ := 4/5

theorem total_plants_grown (
  eggplants_per_packet sunflowers_per_packet tomatoes_per_packet peas_per_packet : ℕ)
  (eggplant_packets sunflower_packets tomato_packets pea_packets : ℕ)
  (spring_growth_rate summer_growth_rate : ℚ) :
  ⌊(eggplants_per_packet * eggplant_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(peas_per_packet * pea_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(sunflowers_per_packet * sunflower_packets : ℚ) * summer_growth_rate⌋ +
  ⌊(tomatoes_per_packet * tomato_packets : ℚ) * summer_growth_rate⌋ = 249 :=
by sorry

end NUMINAMATH_CALUDE_total_plants_grown_l1975_197508


namespace NUMINAMATH_CALUDE_min_value_reciprocal_squares_l1975_197544

theorem min_value_reciprocal_squares (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_constraint : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ 3 ∧ 
  (1/a^2 + 1/b^2 + 1/c^2 = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

#check min_value_reciprocal_squares

end NUMINAMATH_CALUDE_min_value_reciprocal_squares_l1975_197544


namespace NUMINAMATH_CALUDE_balloon_min_volume_l1975_197500

/-- Represents the relationship between pressure and volume of a gas in a balloon -/
noncomputable def pressure (k : ℝ) (V : ℝ) : ℝ := k / V

theorem balloon_min_volume (k : ℝ) :
  (pressure k 3 = 8000) →
  (∀ V, V ≥ 0.6 → pressure k V ≤ 40000) ∧
  (∀ ε > 0, ∃ V, 0.6 - ε < V ∧ V < 0.6 ∧ pressure k V > 40000) :=
by sorry

end NUMINAMATH_CALUDE_balloon_min_volume_l1975_197500


namespace NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1975_197521

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_length : ℕ
  num_colors : ℕ

/-- Calculates the number of small cubes with only one face colored -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  (c.edge_length - 2)^2 * c.num_colors

/-- Theorem: A cube cut into 125 smaller cubes with 6 different colored faces has 54 small cubes with only one face colored -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube, c.edge_length = 5 → c.num_colors = 6 → one_face_colored_cubes c = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1975_197521


namespace NUMINAMATH_CALUDE_optimal_profit_l1975_197593

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  (500 - 10 * x) * (50 + x) - (500 - 10 * x) * 40

-- Define the optimal price increase
def optimal_price_increase : ℕ := 20

-- Define the optimal selling price
def optimal_selling_price : ℕ := 50 + optimal_price_increase

-- Define the maximum profit
def max_profit : ℝ := 9000

-- Theorem statement
theorem optimal_profit :
  (∀ x : ℕ, profit x ≤ profit optimal_price_increase) ∧
  (profit optimal_price_increase = max_profit) ∧
  (optimal_selling_price = 70) :=
sorry

end NUMINAMATH_CALUDE_optimal_profit_l1975_197593


namespace NUMINAMATH_CALUDE_tom_marble_groups_l1975_197585

def red_marble : ℕ := 1
def green_marble : ℕ := 1
def blue_marble : ℕ := 1
def black_marble : ℕ := 1
def yellow_marbles : ℕ := 4

def total_marbles : ℕ := red_marble + green_marble + blue_marble + black_marble + yellow_marbles

def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tom_marble_groups :
  let non_yellow_choices := choose_two (red_marble + green_marble + blue_marble + black_marble) - 1
  let yellow_combinations := choose_two yellow_marbles
  let color_with_yellow := red_marble + green_marble + blue_marble + black_marble
  non_yellow_choices + yellow_combinations + color_with_yellow = 10 :=
by sorry

end NUMINAMATH_CALUDE_tom_marble_groups_l1975_197585


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l1975_197568

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : ∃ k : ℕ, a^(a:ℕ) = k * b^(b:ℕ))
  (h3 : ¬∃ m : ℕ, a = m * b) :
  ∀ (x y : ℕ+), 
    (Nat.gcd (x + y) 330 = 1) → 
    (∃ k : ℕ, x^(x:ℕ) = k * y^(y:ℕ)) → 
    (¬∃ m : ℕ, x = m * y) → 
    (a + b : ℕ) ≤ (x + y : ℕ) ∧ 
    (a + b : ℕ) = 507 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l1975_197568


namespace NUMINAMATH_CALUDE_lcm_150_414_l1975_197565

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_414_l1975_197565


namespace NUMINAMATH_CALUDE_pool_depths_l1975_197559

/-- Pool depths problem -/
theorem pool_depths (john sarah susan mike : ℕ) : 
  john = 2 * sarah + 5 →  -- John's pool is 5 feet deeper than 2 times Sarah's pool
  john = 15 →  -- John's pool is 15 feet deep
  susan = john + sarah - 3 →  -- Susan's pool is 3 feet shallower than the sum of John's and Sarah's pool depths
  mike = john + sarah + susan + 4 →  -- Mike's pool is 4 feet deeper than the combined depth of John's, Sarah's, and Susan's pools
  sarah = 5 ∧ susan = 17 ∧ mike = 41 := by
  sorry

end NUMINAMATH_CALUDE_pool_depths_l1975_197559


namespace NUMINAMATH_CALUDE_tunnel_length_tunnel_length_proof_l1975_197554

/-- Calculates the length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (exit_time : ℝ) : ℝ :=
  let tunnel_length := train_speed * exit_time / 60 - train_length
  2

theorem tunnel_length_proof :
  tunnel_length 1 60 3 = 2 := by sorry

end NUMINAMATH_CALUDE_tunnel_length_tunnel_length_proof_l1975_197554


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1975_197505

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1975_197505


namespace NUMINAMATH_CALUDE_dog_fruit_problem_l1975_197582

/-- The number of bonnies eaten by the third dog -/
def B : ℕ := sorry

/-- The number of blueberries eaten by the second dog -/
def blueberries : ℕ := sorry

/-- The number of apples eaten by the first dog -/
def apples : ℕ := sorry

/-- The total number of fruits eaten by all three dogs -/
def total_fruits : ℕ := 240

theorem dog_fruit_problem :
  (blueberries = (3 * B) / 4) →
  (apples = 3 * blueberries) →
  (B + blueberries + apples = total_fruits) →
  B = 60 := by sorry

end NUMINAMATH_CALUDE_dog_fruit_problem_l1975_197582


namespace NUMINAMATH_CALUDE_jungkook_final_ball_count_l1975_197570

-- Define the initial state
def jungkook_red_balls : ℕ := 3
def yoongi_blue_balls : ℕ := 2

-- Define the transfer
def transferred_balls : ℕ := 1

-- Theorem to prove
theorem jungkook_final_ball_count :
  jungkook_red_balls + transferred_balls = 4 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_final_ball_count_l1975_197570


namespace NUMINAMATH_CALUDE_smallest_missing_number_is_22_l1975_197548

/-- Represents a problem in HMMT November 2023 -/
structure HMMTProblem where
  round : String
  number : Nat

/-- The set of all problems in HMMT November 2023 -/
def HMMTProblems : Set HMMTProblem := sorry

/-- A number appears in HMMT November 2023 if it's used in at least one problem -/
def appears_in_HMMT (n : Nat) : Prop :=
  ∃ (p : HMMTProblem), p ∈ HMMTProblems ∧ p.number = n

theorem smallest_missing_number_is_22 :
  (∀ n : Nat, n > 0 ∧ n ≤ 21 → appears_in_HMMT n) →
  (¬ appears_in_HMMT 22) →
  ∀ m : Nat, m > 0 ∧ ¬ appears_in_HMMT m → m ≥ 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_missing_number_is_22_l1975_197548


namespace NUMINAMATH_CALUDE_tom_toy_cost_proof_l1975_197520

def tom_toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℕ :=
  (initial_money - game_cost) / num_toys

theorem tom_toy_cost_proof (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) 
  (h1 : initial_money = 57)
  (h2 : game_cost = 49)
  (h3 : num_toys = 2)
  (h4 : initial_money > game_cost) :
  tom_toy_cost initial_money game_cost num_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_toy_cost_proof_l1975_197520


namespace NUMINAMATH_CALUDE_probability_red_then_white_l1975_197532

theorem probability_red_then_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_white : white_balls = 2)
  : (red_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_white_l1975_197532


namespace NUMINAMATH_CALUDE_product_zero_l1975_197509

theorem product_zero (a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ) 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) * 
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) * 
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l1975_197509


namespace NUMINAMATH_CALUDE_odd_function_property_positive_x_property_negative_x_property_l1975_197557

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x + Real.log x else x - Real.log (-x)

-- State the theorem
theorem odd_function_property (x : ℝ) : f (-x) = -f x := by sorry

-- State the positive x property
theorem positive_x_property (x : ℝ) (h : x > 0) : f x = x + Real.log x := by sorry

-- State the negative x property
theorem negative_x_property (x : ℝ) (h : x < 0) : f x = x - Real.log (-x) := by sorry

end NUMINAMATH_CALUDE_odd_function_property_positive_x_property_negative_x_property_l1975_197557


namespace NUMINAMATH_CALUDE_g_of_2_eq_11_l1975_197551

/-- Given a function g(x) = 3x^2 + 2x - 5, prove that g(2) = 11 -/
theorem g_of_2_eq_11 : let g : ℝ → ℝ := fun x ↦ 3 * x^2 + 2 * x - 5; g 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_11_l1975_197551


namespace NUMINAMATH_CALUDE_equation_solution_l1975_197552

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.12 * (30 + x) + 0.02 * (50 + 2 * x) = 20 ∧ x = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1975_197552


namespace NUMINAMATH_CALUDE_cube_sum_zero_or_abc_function_l1975_197574

theorem cube_sum_zero_or_abc_function (a b c : ℝ) 
  (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0) (nonzero_c : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (fourth_sixth_power_eq : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  (a^3 + b^3 + c^3 = 0) ∨ (∃ f : ℝ → ℝ → ℝ → ℝ, a^3 + b^3 + c^3 = f a b c) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_zero_or_abc_function_l1975_197574


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1975_197555

theorem sum_of_decimals : (1 : ℚ) + 0.101 + 0.011 + 0.001 = 1.113 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1975_197555


namespace NUMINAMATH_CALUDE_one_plus_sqrt3i_in_M_l1975_197518

/-- The set M of complex numbers with magnitude 2 -/
def M : Set ℂ := {z : ℂ | Complex.abs z = 2}

/-- Proof that 1 + √3i belongs to M -/
theorem one_plus_sqrt3i_in_M : (1 : ℂ) + Complex.I * Real.sqrt 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_plus_sqrt3i_in_M_l1975_197518


namespace NUMINAMATH_CALUDE_cargo_per_truck_l1975_197578

/-- Represents the problem of determining the cargo per truck given certain conditions --/
theorem cargo_per_truck (x : ℝ) (n : ℕ) (h1 : 55 ≤ x ∧ x ≤ 64) 
  (h2 : x = (x / n - 0.5) * (n + 4)) : 
  x / (n + 4) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_cargo_per_truck_l1975_197578


namespace NUMINAMATH_CALUDE_ylona_initial_count_l1975_197512

/-- The number of rubber bands each person has initially and after Bailey gives some away. -/
structure RubberBands :=
  (bailey_initial : ℕ)
  (justine_initial : ℕ)
  (ylona_initial : ℕ)
  (bailey_final : ℕ)

/-- The conditions of the rubber band problem. -/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.ylona_initial = rb.justine_initial + 2 ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8

/-- Theorem stating that Ylona initially had 24 rubber bands. -/
theorem ylona_initial_count (rb : RubberBands) 
  (h : rubber_band_problem rb) : rb.ylona_initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_ylona_initial_count_l1975_197512


namespace NUMINAMATH_CALUDE_interior_triangles_count_l1975_197543

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of triangles formed inside the circle -/
def num_triangles : ℕ := Nat.choose n 6

/-- Theorem stating the number of triangles formed inside the circle -/
theorem interior_triangles_count : num_triangles = 84 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangles_count_l1975_197543


namespace NUMINAMATH_CALUDE_symmetry_line_theorem_l1975_197539

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Line represented by its equation -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Define Circle O -/
def circle_O : Circle :=
  { equation := λ x y => x^2 + y^2 = 4 }

/-- Define Circle C -/
def circle_C : Circle :=
  { equation := λ x y => x^2 + y^2 + 4*x - 4*y + 4 = 0 }

/-- Define the line of symmetry -/
def line_of_symmetry : Line :=
  { equation := λ x y => x - y + 2 = 0 }

/-- Function to check if a line is the line of symmetry between two circles -/
def is_line_of_symmetry (l : Line) (c1 c2 : Circle) : Prop :=
  sorry -- Definition of symmetry between circles with respect to a line

/-- Theorem stating that the given line is the line of symmetry between Circle O and Circle C -/
theorem symmetry_line_theorem :
  is_line_of_symmetry line_of_symmetry circle_O circle_C := by
  sorry

end NUMINAMATH_CALUDE_symmetry_line_theorem_l1975_197539


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1975_197536

/-- A line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point lies on a line if it satisfies the line's equation --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 3, b := 4, c := 1 }
  let l2 : Line := { a := 3, b := 4, c := -11 }
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1975_197536


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1975_197541

theorem max_value_of_expression (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 ∧
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 9) > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1975_197541


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l1975_197531

theorem combined_tennis_percentage
  (north_total : ℕ)
  (south_total : ℕ)
  (north_tennis_percent : ℚ)
  (south_tennis_percent : ℚ)
  (h1 : north_total = 1800)
  (h2 : south_total = 2700)
  (h3 : north_tennis_percent = 25 / 100)
  (h4 : south_tennis_percent = 35 / 100)
  : (north_total * north_tennis_percent + south_total * south_tennis_percent) / (north_total + south_total) = 31 / 100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l1975_197531


namespace NUMINAMATH_CALUDE_total_share_calculation_l1975_197567

/-- Given three shares x, y, and z, where x is 25% more than y, y is 20% more than z,
    and z is 100, prove that the total amount shared is 370. -/
theorem total_share_calculation (x y z : ℚ) : 
  x = 1.25 * y ∧ y = 1.2 * z ∧ z = 100 → x + y + z = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_share_calculation_l1975_197567


namespace NUMINAMATH_CALUDE_t_cube_surface_area_l1975_197537

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  vertical_cubes : ℕ
  horizontal_cubes : ℕ
  intersection_position : ℕ

/-- Calculates the surface area of a T-shaped structure -/
def surface_area (t : TCube) : ℕ :=
  sorry

/-- The specific T-shaped structure described in the problem -/
def problem_t_cube : TCube :=
  { vertical_cubes := 5
  , horizontal_cubes := 5
  , intersection_position := 3 }

theorem t_cube_surface_area :
  surface_area problem_t_cube = 33 :=
sorry

end NUMINAMATH_CALUDE_t_cube_surface_area_l1975_197537


namespace NUMINAMATH_CALUDE_binomial_150_150_l1975_197502

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1975_197502


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l1975_197569

theorem largest_x_abs_equation : ∃ (x : ℝ), x = 7 ∧ |x + 3| = 10 ∧ ∀ y : ℝ, |y + 3| = 10 → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l1975_197569


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l1975_197579

theorem sum_of_squares_inequality (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l1975_197579


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1975_197534

/-- Proves that in a group of ducks and cows, if the total number of legs is 28 more than twice the number of heads, then the number of cows is 14. -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 28) → cows = 14 := by
  sorry


end NUMINAMATH_CALUDE_duck_cow_problem_l1975_197534


namespace NUMINAMATH_CALUDE_product_remainder_mod_three_l1975_197522

theorem product_remainder_mod_three (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_three_l1975_197522


namespace NUMINAMATH_CALUDE_modified_system_solution_l1975_197506

theorem modified_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : a₁ * 8 + b₁ * 3 = c₁ ∧ a₂ * 8 + b₂ * 3 = c₂) :
  4 * a₁ * 10 + 3 * b₁ * 5 = 5 * c₁ ∧ 
  4 * a₂ * 10 + 3 * b₂ * 5 = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_modified_system_solution_l1975_197506


namespace NUMINAMATH_CALUDE_gum_pack_size_l1975_197511

theorem gum_pack_size (y : ℝ) : 
  (25 - 2 * y) / 40 = 25 / (40 + 4 * y) → y = 2.5 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l1975_197511


namespace NUMINAMATH_CALUDE_smallest_p_for_integer_sqrt_l1975_197597

theorem smallest_p_for_integer_sqrt : ∃ (p : ℕ), p > 0 ∧ 
  (∀ (q : ℕ), q > 0 → q < p → ¬ (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * q)) ∧
  (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * p) ∧
  p = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_p_for_integer_sqrt_l1975_197597


namespace NUMINAMATH_CALUDE_range_of_a_l1975_197527

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) / (x - a) < 0}

theorem range_of_a : ∀ a : ℝ, 
  (2 ∈ A a ∧ 3 ∉ A a) ↔ 
  (a ∈ Set.Icc (1/3 : ℝ) (1/2 : ℝ) ∪ Set.Ioc (2 : ℝ) (3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1975_197527


namespace NUMINAMATH_CALUDE_locus_of_points_l1975_197501

/-- The locus of points with a 3:1 distance ratio to a fixed point and line -/
theorem locus_of_points (x y : ℝ) : 
  let F : ℝ × ℝ := (4.5, 0)
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_line := |x - 0.5|
  dist_to_F = 3 * dist_to_line → x^2 / 2.25 - y^2 / 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l1975_197501


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1975_197545

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.exp 1)⁻¹
  let b : ℝ := (Real.log 2) / 2
  let c : ℝ := (Real.log 3) / 3
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1975_197545


namespace NUMINAMATH_CALUDE_ring_area_equals_three_circles_l1975_197542

theorem ring_area_equals_three_circles 
  (r₁ r₂ r₃ d R r : ℝ) (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ d > 0) :
  (R^2 - r^2 = r₁^2 + r₂^2 + r₃^2) ∧ (R - r = d) →
  (R = ((r₁^2 + r₂^2 + r₃^2) + d^2) / (2*d)) ∧ (r = R - d) := by
sorry

end NUMINAMATH_CALUDE_ring_area_equals_three_circles_l1975_197542


namespace NUMINAMATH_CALUDE_range_x_when_a_zero_range_a_for_p_sufficient_q_l1975_197550

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_x_when_a_zero :
  ∀ x : ℝ, p x ∧ ¬(q 0 x) ↔ -7/2 ≤ x ∧ x < -3 :=
sorry

-- Theorem for the second question
theorem range_a_for_p_sufficient_q :
  (∀ x : ℝ, p x → ∀ a : ℝ, q a x) ↔ ∀ a : ℝ, -5/2 ≤ a ∧ a ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_zero_range_a_for_p_sufficient_q_l1975_197550


namespace NUMINAMATH_CALUDE_even_number_selection_l1975_197547

theorem even_number_selection (p : ℝ) (n : ℕ) 
  (h_p : p = 0.5) 
  (h_n : n = 4) : 
  1 - p^n ≥ 0.9 := by
sorry

end NUMINAMATH_CALUDE_even_number_selection_l1975_197547


namespace NUMINAMATH_CALUDE_not_all_primes_in_arithmetic_progression_l1975_197561

def arithmetic_progression (a d : ℤ) (n : ℕ) : ℤ := a + d * n

theorem not_all_primes_in_arithmetic_progression (a d : ℤ) (h : d ≥ 1) :
  ∃ n : ℕ, ¬ Prime (arithmetic_progression a d n) :=
sorry

end NUMINAMATH_CALUDE_not_all_primes_in_arithmetic_progression_l1975_197561


namespace NUMINAMATH_CALUDE_gcd_2183_1947_l1975_197598

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by sorry

end NUMINAMATH_CALUDE_gcd_2183_1947_l1975_197598


namespace NUMINAMATH_CALUDE_factor_cubic_l1975_197562

theorem factor_cubic (a b c : ℝ) : 
  (∀ x, x^3 - 12*x + 16 = (x + 4)*(a*x^2 + b*x + c)) → 
  a*x^2 + b*x + c = (x - 2)^2 := by
sorry

end NUMINAMATH_CALUDE_factor_cubic_l1975_197562


namespace NUMINAMATH_CALUDE_ball_sampling_theorem_l1975_197558

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the bag with balls -/
structure Bag :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the sampling method -/
inductive SamplingMethod
  | WithReplacement
  | WithoutReplacement

/-- The probability of drawing two balls of different colors with replacement -/
def prob_diff_colors (bag : Bag) (method : SamplingMethod) : ℚ :=
  sorry

/-- The expectation of the number of white balls drawn without replacement -/
def expectation_white (bag : Bag) : ℚ :=
  sorry

/-- The variance of the number of white balls drawn without replacement -/
def variance_white (bag : Bag) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem ball_sampling_theorem (bag : Bag) :
  bag.white = 2 ∧ bag.black = 3 →
  prob_diff_colors bag SamplingMethod.WithReplacement = 12/25 ∧
  expectation_white bag = 4/5 ∧
  variance_white bag = 9/25 :=
sorry

end NUMINAMATH_CALUDE_ball_sampling_theorem_l1975_197558


namespace NUMINAMATH_CALUDE_gcd_of_180_210_588_l1975_197513

theorem gcd_of_180_210_588 : Nat.gcd 180 (Nat.gcd 210 588) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_210_588_l1975_197513


namespace NUMINAMATH_CALUDE_discount_percentage_l1975_197577

theorem discount_percentage 
  (profit_with_discount : ℝ) 
  (profit_without_discount : ℝ) 
  (h1 : profit_with_discount = 0.235) 
  (h2 : profit_without_discount = 0.30) : 
  (profit_without_discount - profit_with_discount) / (1 + profit_without_discount) = 0.05 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l1975_197577


namespace NUMINAMATH_CALUDE_badminton_team_lineup_count_l1975_197533

theorem badminton_team_lineup_count :
  let total_players : ℕ := 18
  let quadruplets : ℕ := 4
  let starters : ℕ := 8
  let non_quadruplets : ℕ := total_players - quadruplets
  let lineups_without_quadruplets : ℕ := Nat.choose non_quadruplets starters
  let lineups_with_one_quadruplet : ℕ := quadruplets * Nat.choose non_quadruplets (starters - 1)
  lineups_without_quadruplets + lineups_with_one_quadruplet = 16731 :=
by sorry

end NUMINAMATH_CALUDE_badminton_team_lineup_count_l1975_197533


namespace NUMINAMATH_CALUDE_largest_package_size_l1975_197586

theorem largest_package_size (alex bella carlos : ℕ) 
  (h_alex : alex = 36)
  (h_bella : bella = 48)
  (h_carlos : carlos = 60) :
  Nat.gcd alex (Nat.gcd bella carlos) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1975_197586


namespace NUMINAMATH_CALUDE_g_negative_three_l1975_197599

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

-- State the theorem
theorem g_negative_three (d e f : ℝ) : g d e f 3 = -9 → g d e f (-3) = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_l1975_197599


namespace NUMINAMATH_CALUDE_smallest_g_is_correct_l1975_197538

/-- The smallest positive integer g such that 3150 * g is a perfect square -/
def smallest_g : ℕ := 14

/-- 3150 * g is a perfect square -/
def is_perfect_square (g : ℕ) : Prop :=
  ∃ n : ℕ, 3150 * g = n^2

theorem smallest_g_is_correct :
  (is_perfect_square smallest_g) ∧
  (∀ g : ℕ, 0 < g ∧ g < smallest_g → ¬(is_perfect_square g)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_g_is_correct_l1975_197538


namespace NUMINAMATH_CALUDE_water_remaining_l1975_197510

/-- Given 3 gallons of water and using 5/4 gallons, prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l1975_197510


namespace NUMINAMATH_CALUDE_workout_solution_correct_l1975_197588

/-- Laura's workout parameters -/
structure WorkoutParams where
  bike_distance : ℝ
  bike_rate : ℝ → ℝ
  transition_time : ℝ
  run_distance : ℝ
  total_time : ℝ

/-- The solution to Laura's workout problem -/
def workout_solution (p : WorkoutParams) : ℝ :=
  8

/-- Theorem stating that the workout_solution is correct -/
theorem workout_solution_correct (p : WorkoutParams) 
  (h1 : p.bike_distance = 25)
  (h2 : p.bike_rate = fun x => 3 * x + 1)
  (h3 : p.transition_time = 1/6)  -- 10 minutes in hours
  (h4 : p.run_distance = 8)
  (h5 : p.total_time = 13/6)  -- 130 minutes in hours
  : ∃ (x : ℝ), 
    x = workout_solution p ∧ 
    p.bike_distance / (p.bike_rate x) + p.transition_time + p.run_distance / x = p.total_time :=
  sorry

#check workout_solution_correct

end NUMINAMATH_CALUDE_workout_solution_correct_l1975_197588


namespace NUMINAMATH_CALUDE_race_distances_main_theorem_l1975_197540

/-- Represents a race between three racers over a certain distance -/
structure Race where
  distance : ℝ
  a_beats_b : ℝ
  b_beats_c : ℝ
  a_beats_c : ℝ

/-- The theorem stating the distances of the two races -/
theorem race_distances (race1 race2 : Race) : 
  race1.distance = 150 ∧ race2.distance = 120 :=
  by
    have h1 : race1 = { distance := 150, a_beats_b := 30, b_beats_c := 15, a_beats_c := 42 } := by sorry
    have h2 : race2 = { distance := 120, a_beats_b := 25, b_beats_c := 20, a_beats_c := 40 } := by sorry
    sorry

/-- The main theorem proving the distances of both races -/
theorem main_theorem : ∃ (race1 race2 : Race), 
  race1.a_beats_b = 30 ∧ 
  race1.b_beats_c = 15 ∧ 
  race1.a_beats_c = 42 ∧
  race2.a_beats_b = 25 ∧ 
  race2.b_beats_c = 20 ∧ 
  race2.a_beats_c = 40 ∧
  race1.distance = 150 ∧ 
  race2.distance = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_race_distances_main_theorem_l1975_197540


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1975_197566

/-- Proves that given the described investment scenario, the initial interest rate is approximately 0.2 -/
theorem investment_rate_proof (initial_investment : ℝ) (years : ℕ) (final_amount : ℝ) : 
  initial_investment = 10000 →
  years = 3 →
  final_amount = 59616 →
  ∃ (r : ℝ), 
    (r ≥ 0) ∧ 
    (r ≤ 1) ∧
    (abs (r - 0.2) < 0.001) ∧
    (final_amount = 3 * initial_investment * (1 + r)^years * 1.15) :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1975_197566


namespace NUMINAMATH_CALUDE_new_rectangle_area_l1975_197514

/-- Given a rectangle with sides a and b (a < b), prove that the area of a new rectangle
    with base (b + 2a) and height (b - a) is b^2 + ab - 2a^2 -/
theorem new_rectangle_area (a b : ℝ) (h : a < b) :
  (b + 2*a) * (b - a) = b^2 + a*b - 2*a^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l1975_197514


namespace NUMINAMATH_CALUDE_range_of_a_l1975_197595

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) ↔ 
  (a ≥ 1 ∨ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1975_197595


namespace NUMINAMATH_CALUDE_angela_is_157_cm_tall_l1975_197584

def amy_height : ℕ := 150

def helen_height (amy : ℕ) : ℕ := amy + 3

def angela_height (helen : ℕ) : ℕ := helen + 4

theorem angela_is_157_cm_tall :
  angela_height (helen_height amy_height) = 157 :=
by sorry

end NUMINAMATH_CALUDE_angela_is_157_cm_tall_l1975_197584


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1975_197519

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_sale_price : ℝ)
  (h1 : original_price = 495)
  (h2 : first_discount_percent = 15)
  (h3 : final_sale_price = 378.675) :
  ∃ (second_discount_percent : ℝ),
    second_discount_percent = 10 ∧
    final_sale_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1975_197519


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_less_than_one_l1975_197546

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If M(-1, a-1) is in the third quadrant, then a < 1 -/
theorem point_in_third_quadrant_implies_a_less_than_one (a : ℝ) :
  in_third_quadrant (Point.mk (-1) (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_less_than_one_l1975_197546


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1975_197590

/-- The capacity of a fuel tank given specific conditions -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 82 + 0.16 * (C - 82) = 30) ∧ 
  (C = 208) := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1975_197590


namespace NUMINAMATH_CALUDE_prime_squared_plus_41_composite_l1975_197583

theorem prime_squared_plus_41_composite (p : ℕ) (hp : Prime p) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p^2 + 41 = a * b :=
sorry

end NUMINAMATH_CALUDE_prime_squared_plus_41_composite_l1975_197583


namespace NUMINAMATH_CALUDE_divisors_of_square_of_four_divisor_number_l1975_197524

/-- A natural number has exactly 4 divisors -/
def has_four_divisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of the square of a number with 4 divisors -/
theorem divisors_of_square_of_four_divisor_number (m : ℕ) :
  has_four_divisors m →
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 7 ∨
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_divisors_of_square_of_four_divisor_number_l1975_197524


namespace NUMINAMATH_CALUDE_pet_store_cages_theorem_l1975_197596

/-- Given a number of initial puppies, sold puppies, and puppies per cage,
    calculate the number of cages needed. -/
def cagesNeeded (initialPuppies soldPuppies puppiesPerCage : ℕ) : ℕ :=
  ((initialPuppies - soldPuppies) + puppiesPerCage - 1) / puppiesPerCage

theorem pet_store_cages_theorem :
  cagesNeeded 36 7 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_theorem_l1975_197596


namespace NUMINAMATH_CALUDE_current_speed_l1975_197564

/-- Proves that the speed of the current is 20 kmph, given the boat's speed in still water and upstream. -/
theorem current_speed (boat_still_speed upstream_speed : ℝ) 
  (h1 : boat_still_speed = 50)
  (h2 : upstream_speed = 30) :
  boat_still_speed - upstream_speed = 20 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l1975_197564


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l1975_197523

theorem max_value_theorem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 :=
by sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (x' y' z' : ℝ), 9 * x'^2 + 4 * y'^2 + 25 * z'^2 = 1 ∧ 3 * x' + 4 * y' + 6 * z' = Real.sqrt 53 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l1975_197523


namespace NUMINAMATH_CALUDE_min_value_theorem_l1975_197572

/-- Given positive real numbers a and b, and a function f with minimum value 4 -/
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : ∀ x, f x a b ≥ 4) (hf_min : ∃ x, f x a b = 4) :
  (a + b = 4) ∧ (∀ a b, a > 0 → b > 0 → a + b = 4 → (1/4) * a^2 + (1/4) * b^2 ≥ 3/16) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ (1/4) * a^2 + (1/4) * b^2 = 3/16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1975_197572


namespace NUMINAMATH_CALUDE_orange_juice_serving_volume_l1975_197571

/-- Proves that the volume of each serving of orange juice is 6 ounces given the specified conditions. -/
theorem orange_juice_serving_volume
  (concentrate_cans : ℕ)
  (concentrate_oz_per_can : ℕ)
  (water_cans_per_concentrate : ℕ)
  (total_servings : ℕ)
  (h1 : concentrate_cans = 60)
  (h2 : concentrate_oz_per_can = 5)
  (h3 : water_cans_per_concentrate = 3)
  (h4 : total_servings = 200) :
  (concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)) / total_servings = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_serving_volume_l1975_197571


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1975_197587

theorem smallest_five_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 → 
      (2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m) → m ≥ n) ∧
    (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) ∧
    n = 11550 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1975_197587


namespace NUMINAMATH_CALUDE_diamond_two_three_l1975_197535

def diamond (a b : ℝ) : ℝ := a^3 * b^2 - b + 2

theorem diamond_two_three : diamond 2 3 = 71 := by sorry

end NUMINAMATH_CALUDE_diamond_two_three_l1975_197535


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l1975_197530

theorem spencer_walk_distance (total : ℝ) (house_to_library : ℝ) (library_to_post : ℝ)
  (h1 : total = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : library_to_post = 0.1) :
  total - (house_to_library + library_to_post) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l1975_197530


namespace NUMINAMATH_CALUDE_probability_less_than_three_l1975_197573

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0,0), (0,2), (2,2), and (2,0) --/
def unitSquare : Square :=
  { bottomLeft := (0, 0), topRight := (2, 2) }

/-- The condition x + y < 3 --/
def lessThanThree (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 3

theorem probability_less_than_three :
  probability unitSquare lessThanThree = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_three_l1975_197573


namespace NUMINAMATH_CALUDE_zelda_success_probability_l1975_197575

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/3) 
  (h2 : p_yvonne = 1/2) 
  (h3 : p_xy_not_z = 0.0625) : 
  ∃ p_zelda : ℝ, p_zelda = 0.625 ∧ p_xavier * p_yvonne * (1 - p_zelda) = p_xy_not_z :=
by sorry

end NUMINAMATH_CALUDE_zelda_success_probability_l1975_197575
