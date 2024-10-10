import Mathlib

namespace coal_division_l992_99202

/-- Given 3 tons of coal divided equally into 5 parts, prove the fraction and amount of each part -/
theorem coal_division (total_coal : ℝ) (num_parts : ℕ) 
  (h1 : total_coal = 3)
  (h2 : num_parts = 5) :
  (1 : ℝ) / num_parts = 1 / 5 ∧ 
  total_coal / num_parts = 3 / 5 := by
  sorry

end coal_division_l992_99202


namespace company_sampling_methods_l992_99212

/-- Enumeration of regions --/
inductive Region
| A
| B
| C
| D

/-- Enumeration of sampling methods --/
inductive SamplingMethod
| StratifiedSampling
| SimpleRandomSampling

/-- Structure representing the sales points distribution --/
structure SalesDistribution where
  total_points : ℕ
  region_points : Region → ℕ
  large_points_C : ℕ

/-- Structure representing an investigation --/
structure Investigation where
  sample_size : ℕ
  population_size : ℕ

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (dist : SalesDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given scenario --/
theorem company_sampling_methods 
  (dist : SalesDistribution)
  (inv1 inv2 : Investigation)
  (h1 : dist.total_points = 600)
  (h2 : dist.region_points Region.A = 150)
  (h3 : dist.region_points Region.B = 120)
  (h4 : dist.region_points Region.C = 180)
  (h5 : dist.region_points Region.D = 150)
  (h6 : dist.large_points_C = 20)
  (h7 : inv1.sample_size = 100)
  (h8 : inv1.population_size = 600)
  (h9 : inv2.sample_size = 7)
  (h10 : inv2.population_size = 20) :
  (appropriate_sampling_method dist inv1 = SamplingMethod.StratifiedSampling) ∧
  (appropriate_sampling_method dist inv2 = SamplingMethod.SimpleRandomSampling) :=
sorry

end company_sampling_methods_l992_99212


namespace three_player_rotation_l992_99269

/-- Represents the number of games played by each player in a three-player table tennis rotation. -/
structure GameCount where
  player1 : ℕ
  player2 : ℕ
  player3 : ℕ

/-- 
Theorem: In a three-player table tennis rotation where the losing player is replaced by the non-participating player,
if Player 1 played 10 games and Player 2 played 21 games, then Player 3 must have played 11 games.
-/
theorem three_player_rotation (gc : GameCount) 
  (h1 : gc.player1 = 10)
  (h2 : gc.player2 = 21)
  (h_total : gc.player1 + gc.player2 + gc.player3 = 2 * gc.player2) :
  gc.player3 = 11 := by
  sorry


end three_player_rotation_l992_99269


namespace floor_times_x_eq_152_l992_99284

theorem floor_times_x_eq_152 : ∃ x : ℝ, (⌊x⌋ : ℝ) * x = 152 ∧ x = 38 / 3 := by
  sorry

end floor_times_x_eq_152_l992_99284


namespace tens_digit_of_8_pow_2023_l992_99222

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 ≡ 10 * k + 1 [ZMOD 100] := by
  sorry

end tens_digit_of_8_pow_2023_l992_99222


namespace games_given_solution_l992_99260

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := sorry

/-- Henry's initial number of games -/
def henry_initial : ℕ := 58

/-- Neil's initial number of games -/
def neil_initial : ℕ := 7

theorem games_given_solution :
  (henry_initial - games_given = 4 * (neil_initial + games_given)) ∧
  games_given = 6 := by sorry

end games_given_solution_l992_99260


namespace extreme_value_conditions_l992_99223

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_conditions (a b : ℝ) : 
  f a b (-1) = 8 ∧ f_derivative a b (-1) = 0 → a = 2 ∧ b = -7 := by sorry

end extreme_value_conditions_l992_99223


namespace ball_ratio_proof_l992_99273

theorem ball_ratio_proof (a b x : ℕ) : 
  (a / (a + b + x) = 1/4) →
  ((a + x) / (b + x) = 2/3) →
  (3*a - b = x) →
  (2*b - 3*a = x) →
  (a / b = 1/2) := by
  sorry

end ball_ratio_proof_l992_99273


namespace cylinder_volume_l992_99213

/-- Given a cylinder whose lateral surface unfolds into a rectangle with length 2a and width a, 
    its volume is either a³/π or a³/(2π) -/
theorem cylinder_volume (a : ℝ) (h : a > 0) :
  ∃ (V : ℝ), (V = a^3 / π ∨ V = a^3 / (2*π)) ∧
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  ((2*π*r = 2*a ∧ h = a) ∨ (2*π*r = a ∧ h = 2*a)) ∧
  V = π * r^2 * h :=
sorry

end cylinder_volume_l992_99213


namespace violets_to_carnations_ratio_l992_99277

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  violets : ℕ
  carnations : ℕ
  tulips : ℕ
  roses : ℕ

/-- The conditions of the flower shop -/
def FlowerShopConditions (shop : FlowerShop) : Prop :=
  shop.tulips = shop.violets / 4 ∧
  shop.roses = shop.tulips ∧
  shop.carnations = (2 * (shop.violets + shop.carnations + shop.tulips + shop.roses)) / 3

/-- The theorem stating the ratio of violets to carnations -/
theorem violets_to_carnations_ratio (shop : FlowerShop) 
  (h : FlowerShopConditions shop) : 
  shop.violets = shop.carnations / 3 := by
  sorry

#check violets_to_carnations_ratio

end violets_to_carnations_ratio_l992_99277


namespace shaded_perimeter_value_l992_99282

/-- The perimeter of the shaded region formed by four quarter-circle arcs in a unit square --/
def shadedPerimeter : ℝ := sorry

/-- The square PQRS with side length 1 --/
def unitSquare : Set (ℝ × ℝ) := sorry

/-- Arc TRU with center P --/
def arcTRU : Set (ℝ × ℝ) := sorry

/-- Arc VPW with center R --/
def arcVPW : Set (ℝ × ℝ) := sorry

/-- Arc UV with center S --/
def arcUV : Set (ℝ × ℝ) := sorry

/-- Arc WT with center Q --/
def arcWT : Set (ℝ × ℝ) := sorry

/-- The theorem stating that the perimeter of the shaded region is (2√2 - 1)π --/
theorem shaded_perimeter_value : shadedPerimeter = (2 * Real.sqrt 2 - 1) * Real.pi := by sorry

end shaded_perimeter_value_l992_99282


namespace cookies_with_four_cups_l992_99231

/-- Represents the number of cookies that can be made with a given amount of flour,
    maintaining a constant ratio of flour to sugar. -/
def cookies_made (flour : ℚ) : ℚ :=
  24 * flour / 3

/-- The ratio of flour to sugar remains constant. -/
axiom constant_ratio : ∀ (f : ℚ), cookies_made f / f = 24 / 3

theorem cookies_with_four_cups :
  cookies_made 4 = 128 :=
sorry

end cookies_with_four_cups_l992_99231


namespace gwens_birthday_money_l992_99290

/-- The amount of money Gwen spent -/
def amount_spent : ℕ := 8

/-- The amount of money Gwen has left -/
def amount_left : ℕ := 6

/-- The total amount of money Gwen received for her birthday -/
def total_amount : ℕ := amount_spent + amount_left

theorem gwens_birthday_money : total_amount = 14 := by
  sorry

end gwens_birthday_money_l992_99290


namespace square_eq_necessary_condition_l992_99281

theorem square_eq_necessary_condition (x h k : ℝ) :
  (x + h)^2 = k → k ≥ 0 := by
  sorry

end square_eq_necessary_condition_l992_99281


namespace expression_evaluation_l992_99274

theorem expression_evaluation : 10 - 9 + 8 * 7^2 + 6 - 5 * 4 + 3 - 2 = 380 := by
  sorry

end expression_evaluation_l992_99274


namespace articles_produced_is_y_l992_99251

/-- Given that x men working x hours a day for x days produce x articles,
    this function calculates the number of articles produced by x men
    working x hours a day for y days. -/
def articles_produced (x y : ℝ) : ℝ :=
  y

/-- Theorem stating that the number of articles produced is y -/
theorem articles_produced_is_y (x y : ℝ) (h : x > 0) :
  articles_produced x y = y :=
by sorry

end articles_produced_is_y_l992_99251


namespace product_of_repeating_decimal_666_and_8_mixed_number_representation_l992_99204

def repeating_decimal_666 : ℚ := 2/3

theorem product_of_repeating_decimal_666_and_8 :
  repeating_decimal_666 * 8 = 16/3 :=
sorry

theorem mixed_number_representation :
  16/3 = 5 + 1/3 :=
sorry

end product_of_repeating_decimal_666_and_8_mixed_number_representation_l992_99204


namespace later_arrival_l992_99230

/-- A man's journey to his office -/
structure JourneyToOffice where
  usual_rate : ℝ
  usual_time : ℝ
  slower_rate : ℝ
  slower_time : ℝ

/-- The conditions of the problem -/
def journey_conditions (j : JourneyToOffice) : Prop :=
  j.usual_time = 1 ∧ j.slower_rate = 3/4 * j.usual_rate

/-- The theorem to be proved -/
theorem later_arrival (j : JourneyToOffice) 
  (h : journey_conditions j) : 
  j.slower_time - j.usual_time = 1/3 := by
  sorry

end later_arrival_l992_99230


namespace intersection_points_distance_squared_l992_99261

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles is 16 -/
theorem intersection_points_distance_squared
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1.center = (1, 3))
  (h2 : c1.radius = 3)
  (h3 : c2.center = (1, -4))
  (h4 : c2.radius = 6)
  : ∃ p1 p2 : ℝ × ℝ,
    squaredDistance p1 p2 = 16 ∧
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 := by
  sorry

end intersection_points_distance_squared_l992_99261


namespace sin_315_degrees_l992_99278

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l992_99278


namespace stone_length_is_four_dm_l992_99293

/-- Represents the dimensions of a rectangular hall in meters -/
structure HallDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular stone in decimeters -/
structure StoneDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : HallDimensions) : ℝ := d.length * d.width

/-- Converts meters to decimeters -/
def meterToDecimeter (m : ℝ) : ℝ := m * 10

/-- Theorem: Given a hall of 36m x 15m, 2700 stones required, and stone width of 5dm,
    prove that the length of each stone is 4 dm -/
theorem stone_length_is_four_dm (hall : HallDimensions) (stone : StoneDimensions) 
    (num_stones : ℕ) : 
    hall.length = 36 → 
    hall.width = 15 → 
    num_stones = 2700 → 
    stone.width = 5 → 
    stone.length = 4 := by
  sorry

end stone_length_is_four_dm_l992_99293


namespace probability_at_least_three_speak_l992_99263

def probability_of_success : ℚ := 1 / 3

def number_of_trials : ℕ := 7

def minimum_successes : ℕ := 3

theorem probability_at_least_three_speak :
  (1 : ℚ) - (Finset.sum (Finset.range minimum_successes) (λ k =>
    (Nat.choose number_of_trials k : ℚ) *
    probability_of_success ^ k *
    (1 - probability_of_success) ^ (number_of_trials - k)))
  = 939 / 2187 := by
  sorry

end probability_at_least_three_speak_l992_99263


namespace cubic_sum_theorem_l992_99233

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -6)
  (product_condition : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 := by
sorry

end cubic_sum_theorem_l992_99233


namespace formula_holds_for_given_pairs_l992_99242

def formula (x : ℕ) : ℕ := x^2 + 4*x + 3

theorem formula_holds_for_given_pairs : 
  (formula 1 = 3) ∧ 
  (formula 2 = 8) ∧ 
  (formula 3 = 15) ∧ 
  (formula 4 = 24) ∧ 
  (formula 5 = 35) := by
  sorry

end formula_holds_for_given_pairs_l992_99242


namespace equation_roots_l992_99258

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 - 2*x^3 - 7*x^2 - 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 + Real.sqrt 5) / 2) ∧
    (b = (1 - Real.sqrt 5) / 2) ∧
    (c = (-1 + Real.sqrt 37) / 6) ∧
    (d = (-1 - Real.sqrt 37) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end equation_roots_l992_99258


namespace a_range_l992_99215

theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) → 
  3 < a ∧ a < 5 := by
sorry

end a_range_l992_99215


namespace little_john_remaining_money_l992_99229

/-- Calculates the remaining money after Little John's expenditures -/
def remaining_money (initial_amount spent_on_sweets toy_cost friend_gift number_of_friends : ℚ) : ℚ :=
  initial_amount - (spent_on_sweets + toy_cost + friend_gift * number_of_friends)

/-- Theorem: Given Little John's initial amount and expenditures, the remaining money is $11.55 -/
theorem little_john_remaining_money :
  remaining_money 20.10 1.05 2.50 1.00 5 = 11.55 := by
  sorry

#eval remaining_money 20.10 1.05 2.50 1.00 5

end little_john_remaining_money_l992_99229


namespace direct_proportional_function_inequality_l992_99291

/-- A direct proportional function satisfying f[f(x)] ≥ x - 3 for all real x
    must be either f(x) = -x or f(x) = x -/
theorem direct_proportional_function_inequality 
  (f : ℝ → ℝ) 
  (h_prop : ∃ (a : ℝ), ∀ x, f x = a * x) 
  (h_ineq : ∀ x, f (f x) ≥ x - 3) :
  (∀ x, f x = -x) ∨ (∀ x, f x = x) := by
sorry

end direct_proportional_function_inequality_l992_99291


namespace sum_geq_three_l992_99265

theorem sum_geq_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end sum_geq_three_l992_99265


namespace line_through_P_with_equal_intercepts_l992_99216

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space by its equation ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point P(2,3)
def P : Point2D := ⟨2, 3⟩

-- The two possible lines
def line1 : Line2D := ⟨3, -2, 0⟩
def line2 : Line2D := ⟨1, 1, -5⟩

-- The theorem to prove
theorem line_through_P_with_equal_intercepts :
  (pointOnLine P line1 ∧ equalIntercepts line1) ∨
  (pointOnLine P line2 ∧ equalIntercepts line2) := by
  sorry

end line_through_P_with_equal_intercepts_l992_99216


namespace max_basketballs_l992_99217

/-- The cost of footballs and basketballs -/
structure BallCosts where
  football : ℕ
  basketball : ℕ

/-- The problem setup -/
structure ProblemSetup where
  costs : BallCosts
  total_balls : ℕ
  max_cost : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  3 * setup.costs.football + 2 * setup.costs.basketball = 310 ∧
  2 * setup.costs.football + 5 * setup.costs.basketball = 500 ∧
  setup.total_balls = 96 ∧
  setup.max_cost = 5800

/-- The theorem to prove -/
theorem max_basketballs (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  ∃ (x : ℕ), x ≤ setup.total_balls ∧ 
    x * setup.costs.basketball + (setup.total_balls - x) * setup.costs.football ≤ setup.max_cost ∧
    ∀ (y : ℕ), y > x → 
      y * setup.costs.basketball + (setup.total_balls - y) * setup.costs.football > setup.max_cost :=
by
  sorry

end max_basketballs_l992_99217


namespace modular_inverse_28_mod_29_l992_99286

theorem modular_inverse_28_mod_29 : ∃ x : ℤ, (28 * x) % 29 = 1 :=
by
  use 28
  sorry

end modular_inverse_28_mod_29_l992_99286


namespace male_puppies_count_l992_99244

/-- Proves that the number of male puppies is 10 given the specified conditions -/
theorem male_puppies_count (total_puppies : ℕ) (female_puppies : ℕ) (ratio : ℚ) :
  total_puppies = 12 →
  female_puppies = 2 →
  ratio = 1/5 →
  total_puppies = female_puppies + (female_puppies / ratio) :=
by
  sorry

end male_puppies_count_l992_99244


namespace min_value_of_function_l992_99203

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a^5 + a^5 - 2) * (1/b^5 + b^5 - 2) ≥ 31^4 / 32^2 := by
  sorry

end min_value_of_function_l992_99203


namespace ellen_painted_twenty_vines_l992_99266

/-- Represents the time in minutes required to paint different types of flowers and vines. -/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- Represents the number of each type of flower and vine painted. -/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting given the painting times and flower counts. -/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies + times.rose * counts.roses + 
  times.orchid * counts.orchids + times.vine * counts.vines

/-- Theorem stating that Ellen painted 20 vines given the problem conditions. -/
theorem ellen_painted_twenty_vines 
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.roses = 10)
  (h7 : counts.orchids = 6)
  (h8 : totalPaintingTime times counts = 213) :
  counts.vines = 20 := by
  sorry

end ellen_painted_twenty_vines_l992_99266


namespace lego_storage_time_l992_99208

/-- The time needed to store all Lego pieces -/
def storage_time (total_pieces : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  (total_pieces - 1) / net_increase_per_minute + 1

/-- Theorem: It takes 43 minutes to store 45 Lego pieces with a net increase of 1 piece per minute -/
theorem lego_storage_time :
  storage_time 45 1 = 43 := by
  sorry

end lego_storage_time_l992_99208


namespace inequality_proof_l992_99246

open Real

noncomputable def f (a x : ℝ) : ℝ := (a/2) * x^2 - (a-2) * x - 2 * x * log x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : 0 < a ∧ a < 2)
  (h_x : x₁ < x₂)
  (h_zeros : ∃ (x : ℝ), x = x₁ ∨ x = x₂ ∧ (deriv (f a)) x = 0) :
  x₂ - x₁ > 4/a - 2 := by
  sorry

end inequality_proof_l992_99246


namespace stack_height_l992_99205

/-- Calculates the vertical distance of a stack of linked rings -/
def verticalDistance (topDiameter : ℕ) (bottomDiameter : ℕ) (thickness : ℕ) : ℕ :=
  let numberOfRings := (topDiameter - bottomDiameter) / 2 + 1
  let innerDiameterSum := (numberOfRings * (topDiameter - thickness * 2 + bottomDiameter - thickness * 2)) / 2
  innerDiameterSum + thickness * 2

/-- The vertical distance of the stack of rings is 76 cm -/
theorem stack_height : verticalDistance 20 4 2 = 76 := by
  sorry

end stack_height_l992_99205


namespace equidistant_sum_constant_sum_of_terms_l992_99201

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of equidistant terms in an arithmetic sequence is constant -/
theorem equidistant_sum_constant {a : ℕ → ℝ} (h : arithmetic_sequence a) :
  ∀ n k : ℕ, a n + a (n + k) = a (n - 1) + a (n + k + 1) :=
sorry

theorem sum_of_terms (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end equidistant_sum_constant_sum_of_terms_l992_99201


namespace inscribed_sphere_in_cone_l992_99262

theorem inscribed_sphere_in_cone (a b c : ℝ) : 
  let cone_base_radius : ℝ := 20
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := (120 * (Real.sqrt 13 - 10)) / 27
  sphere_radius = a * Real.sqrt c - b →
  a + b + c = 253 := by
  sorry

end inscribed_sphere_in_cone_l992_99262


namespace sfl_entrances_l992_99236

/-- Given that there are 283 people waiting at each entrance and 1415 people in total,
    prove that the number of entrances is 5. -/
theorem sfl_entrances (people_per_entrance : ℕ) (total_people : ℕ) 
  (h1 : people_per_entrance = 283) 
  (h2 : total_people = 1415) :
  total_people / people_per_entrance = 5 := by
  sorry

end sfl_entrances_l992_99236


namespace factory_output_increase_l992_99226

theorem factory_output_increase (planned_output actual_output : ℝ) 
  (h1 : planned_output = 20)
  (h2 : actual_output = 24) : 
  (actual_output - planned_output) / planned_output = 0.2 := by
  sorry

end factory_output_increase_l992_99226


namespace quadratic_completion_of_square_l992_99271

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 6*x

-- Define the general form of a quadratic expression
def general_form (a h k x : ℝ) : ℝ := a*(x - h)^2 + k

-- Theorem statement
theorem quadratic_completion_of_square :
  ∃ (a h k : ℝ), ∀ x, quadratic_expr x = general_form a h k x → k = -9 := by
  sorry

end quadratic_completion_of_square_l992_99271


namespace power_function_properties_l992_99287

def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧
  (∀ x, f m (-x) = f m x) →
  m = 1 := by sorry

end power_function_properties_l992_99287


namespace selection_ways_10_people_l992_99299

/-- The number of ways to choose a president, vice-president, and 2-person committee from n people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating that there are 2520 ways to make the selection from 10 people -/
theorem selection_ways_10_people :
  selection_ways 10 = 2520 := by
  sorry

end selection_ways_10_people_l992_99299


namespace M_simplification_M_specific_value_l992_99272

/-- Given expressions for A and B -/
def A (x y : ℝ) : ℝ := x^2 - 3*x*y - y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y - 3*y^2

/-- The expression M defined as 2A - B -/
def M (x y : ℝ) : ℝ := 2 * A x y - B x y

/-- Theorem stating that M simplifies to x^2 - 3xy + y^2 -/
theorem M_simplification (x y : ℝ) : M x y = x^2 - 3*x*y + y^2 := by
  sorry

/-- Theorem stating that M equals 11 when x = -2 and y = 1 -/
theorem M_specific_value : M (-2) 1 = 11 := by
  sorry

end M_simplification_M_specific_value_l992_99272


namespace pineapple_profit_l992_99289

/-- Calculates Jonah's profit from selling pineapples --/
theorem pineapple_profit : 
  let num_pineapples : ℕ := 6
  let price_per_pineapple : ℚ := 3
  let discount_rate : ℚ := 0.2
  let discount_threshold : ℕ := 4
  let rings_per_pineapple : ℕ := 10
  let price_per_two_rings : ℚ := 5
  let price_per_four_ring_set : ℚ := 16

  let total_cost : ℚ := if num_pineapples > discount_threshold
    then num_pineapples * price_per_pineapple * (1 - discount_rate)
    else num_pineapples * price_per_pineapple

  let total_rings : ℕ := num_pineapples * rings_per_pineapple
  let revenue_from_two_rings : ℚ := price_per_two_rings
  let remaining_rings : ℕ := total_rings - 2
  let full_sets : ℕ := remaining_rings / 4
  let revenue_from_sets : ℚ := full_sets * price_per_four_ring_set

  let total_revenue : ℚ := revenue_from_two_rings + revenue_from_sets
  let profit : ℚ := total_revenue - total_cost

  profit = 219.6 := by sorry

end pineapple_profit_l992_99289


namespace largest_number_l992_99276

theorem largest_number (S : Set ℤ) (h : S = {0, 2, -1, -2}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 2 := by
  sorry

end largest_number_l992_99276


namespace max_sum_squared_integers_l992_99238

theorem max_sum_squared_integers (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  i + j + k ≤ 77 := by
sorry

end max_sum_squared_integers_l992_99238


namespace partition_contains_perfect_square_sum_l992_99232

theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Set ℕ), (A ∪ B = Finset.range n.succ) → (A ∩ B = ∅) →
  (∃ (x y : ℕ), x ≠ y ∧ ((x ∈ A ∧ y ∈ A) ∨ (x ∈ B ∧ y ∈ B)) ∧ ∃ (z : ℕ), x + y = z^2) :=
by sorry

end partition_contains_perfect_square_sum_l992_99232


namespace businessmen_neither_coffee_nor_tea_l992_99225

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 6) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end businessmen_neither_coffee_nor_tea_l992_99225


namespace factorization_of_2m_squared_minus_18_l992_99294

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end factorization_of_2m_squared_minus_18_l992_99294


namespace keith_picked_six_apples_l992_99279

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The total number of apples picked -/
def total_apples : ℕ := 16

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_picked_six_apples : keith_apples = 6 := by
  sorry

end keith_picked_six_apples_l992_99279


namespace exists_polynomial_for_E_l992_99227

/-- Definition of E(m) as described in the problem -/
def E (m : ℕ) : ℕ :=
  (Finset.univ.filter (fun s : Finset (Fin 6) => s.card = 6)).card

/-- The main theorem to be proved -/
theorem exists_polynomial_for_E :
  ∃ (c₄ c₃ c₂ c₁ c₀ : ℚ),
    ∀ (m : ℕ), m ≥ 6 → m % 2 = 0 →
      E m = c₄ * m^4 + c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀ := by
  sorry

end exists_polynomial_for_E_l992_99227


namespace average_difference_l992_99255

/-- Given that the average of a and b is 50, and the average of b and c is 70, prove that c - a = 40 -/
theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
  sorry

end average_difference_l992_99255


namespace line_through_tangent_intersections_l992_99292

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- A point inside the ellipse -/
def M : ℝ × ℝ := (3, 2)

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  a : ℝ × ℝ
  b : ℝ × ℝ
  ha : is_ellipse a.1 a.2
  hb : is_ellipse b.1 b.2

/-- The intersection point of tangent lines -/
structure TangentIntersection where
  p : ℝ × ℝ
  line : IntersectingLine
  -- Additional properties for tangent intersection could be added here

/-- The theorem statement -/
theorem line_through_tangent_intersections 
  (ab cd : IntersectingLine) 
  (p q : TangentIntersection) 
  (hp : p.line = ab) 
  (hq : q.line = cd) 
  (hab : ab.a.1 * M.1 / 25 + ab.a.2 * M.2 / 9 = 1)
  (hcd : cd.a.1 * M.1 / 25 + cd.a.2 * M.2 / 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + (1 - 3 * k / 25) * 9 / 2 ↔ 3 * x / 25 + 2 * y / 9 = 1 :=
sorry

end line_through_tangent_intersections_l992_99292


namespace lcm_hcf_problem_l992_99240

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem lcm_hcf_problem (A B : ℕ) (h1 : Nat.lcm A B = 7700) (h2 : Nat.gcd A B = 11) (h3 : B = 275) :
  A = 308 := by
  sorry

end lcm_hcf_problem_l992_99240


namespace complex_arithmetic_equality_l992_99297

theorem complex_arithmetic_equality : (18 * 23 - 24 * 17) / 3 + 5 = 7 := by
  sorry

end complex_arithmetic_equality_l992_99297


namespace barbaras_candy_purchase_l992_99243

/-- Theorem: Barbara's Candy Purchase
Given:
- initial_candies: The number of candies Barbara had initially
- final_candies: The number of candies Barbara has after buying more
- bought_candies: The number of candies Barbara bought

Prove that bought_candies = 18, given initial_candies = 9 and final_candies = 27
-/
theorem barbaras_candy_purchase 
  (initial_candies : ℕ) 
  (final_candies : ℕ) 
  (bought_candies : ℕ) 
  (h1 : initial_candies = 9)
  (h2 : final_candies = 27)
  (h3 : final_candies = initial_candies + bought_candies) :
  bought_candies = 18 := by
  sorry

end barbaras_candy_purchase_l992_99243


namespace tens_digit_of_3_to_2023_l992_99270

theorem tens_digit_of_3_to_2023 : ∃ n : ℕ, 3^2023 ≡ 20 + n [ZMOD 100] :=
by
  sorry

end tens_digit_of_3_to_2023_l992_99270


namespace exists_monochromatic_parallelepiped_l992_99296

-- Define the set A as points in ℤ³
def A : Set (ℤ × ℤ × ℤ) := Set.univ

-- Define a color assignment function
def colorAssignment (p : ℕ) : (ℤ × ℤ × ℤ) → Fin p := sorry

-- Define a rectangular parallelepiped
def isRectangularParallelepiped (vertices : Finset (ℤ × ℤ × ℤ)) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_parallelepiped (p : ℕ) (hp : p > 0) :
  ∃ (vertices : Finset (ℤ × ℤ × ℤ)),
    vertices.card = 8 ∧
    isRectangularParallelepiped vertices ∧
    ∃ (c : Fin p), ∀ v ∈ vertices, colorAssignment p v = c :=
  sorry

end exists_monochromatic_parallelepiped_l992_99296


namespace roots_of_Q_are_fifth_powers_of_roots_of_P_l992_99275

-- Define the polynomial P
def P (x : ℂ) : ℂ := x^3 - 3*x + 1

-- Define the polynomial Q
def Q (y : ℂ) : ℂ := y^3 + 15*y^2 - 198*y + 1

-- Theorem statement
theorem roots_of_Q_are_fifth_powers_of_roots_of_P :
  ∀ (α : ℂ), P α = 0 → ∃ (β : ℂ), Q (β^5) = 0 ∧ P β = 0 :=
by sorry

end roots_of_Q_are_fifth_powers_of_roots_of_P_l992_99275


namespace impossible_grid_arrangement_l992_99264

/-- A type representing a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Predicate to check if a grid contains all numbers from 1 to 42 exactly once -/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 42 → ∃! (i : Fin 6) (j : Fin 7), g i j = n

/-- Predicate to check if all vertical 1x2 rectangles in a grid have even sum -/
def all_vertical_sums_even (g : Grid) : Prop :=
  ∀ (i : Fin 5) (j : Fin 7), Even (g i j + g (i.succ) j)

/-- Theorem stating the impossibility of the desired grid arrangement -/
theorem impossible_grid_arrangement : 
  ¬ ∃ (g : Grid), contains_all_numbers g ∧ all_vertical_sums_even g :=
sorry

end impossible_grid_arrangement_l992_99264


namespace farm_animals_l992_99268

theorem farm_animals (horses cows : ℕ) : 
  horses = 5 * cows →                           -- Initial ratio of horses to cows is 5:1
  (horses - 15) = 17 * (cows + 15) / 7 →        -- New ratio after transaction is 17:7
  horses - 15 - (cows + 15) = 50 := by          -- Difference after transaction is 50
sorry

end farm_animals_l992_99268


namespace a_10_has_many_nines_l992_99257

def a : ℕ → ℕ
  | 0 => 9
  | n + 1 => 3 * (a n)^4 + 4 * (a n)^3

theorem a_10_has_many_nines : ∃ k : ℕ, k ≥ 1024 ∧ a 10 ≡ 10^k - 1 [ZMOD 10^k] :=
sorry

end a_10_has_many_nines_l992_99257


namespace pure_imaginary_square_l992_99248

def complex (a b : ℝ) : ℂ := ⟨a, b⟩

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_square (x : ℝ) :
  is_pure_imaginary ((complex x 1)^2) → x = 1 ∨ x = -1 :=
by sorry

end pure_imaginary_square_l992_99248


namespace least_subtraction_for_divisibility_l992_99250

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧
  20 ∣ (50248 - n) ∧
  ∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m)) := by
  sorry

end least_subtraction_for_divisibility_l992_99250


namespace food_bank_donation_l992_99200

theorem food_bank_donation (first_week_donation : ℝ) : first_week_donation = 40 :=
  let second_week_donation := 2 * first_week_donation
  let total_donation := first_week_donation + second_week_donation
  let remaining_food := 36
  have h1 : remaining_food = 0.3 * total_donation := by sorry
  have h2 : 36 = 0.3 * (3 * first_week_donation) := by sorry
  have h3 : first_week_donation = 36 / 0.9 := by sorry
  sorry

#check food_bank_donation

end food_bank_donation_l992_99200


namespace window_washing_time_l992_99235

/-- The time it takes your friend to wash a window (in minutes) -/
def friend_time : ℝ := 3

/-- The total time it takes both of you to wash 25 windows (in minutes) -/
def total_time : ℝ := 30

/-- The number of windows you wash together -/
def num_windows : ℝ := 25

/-- Your time to wash a window (in minutes) -/
def your_time : ℝ := 2

theorem window_washing_time :
  (1 / friend_time + 1 / your_time) * total_time = num_windows :=
sorry

end window_washing_time_l992_99235


namespace inequality_system_solution_l992_99206

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 ≥ 1 ∧ x ≥ a) ↔ x ≥ 2) → a = 2 := by
  sorry

end inequality_system_solution_l992_99206


namespace equilateral_triangle_count_l992_99259

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Counts distinct equilateral triangles with at least two vertices from the polygon -/
def count_equilateral_triangles (p : RegularPolygon 11) : ℕ :=
  sorry

/-- The main theorem stating the count of distinct equilateral triangles -/
theorem equilateral_triangle_count (p : RegularPolygon 11) :
  count_equilateral_triangles p = 92 :=
sorry

end equilateral_triangle_count_l992_99259


namespace complex_exp_13pi_over_2_l992_99241

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_13pi_over_2_l992_99241


namespace smallest_four_digit_divisible_by_44_l992_99280

theorem smallest_four_digit_divisible_by_44 : 
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 44 = 0 → 1023 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_44_l992_99280


namespace quadratic_polynomial_conditions_l992_99256

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) : 
  (∀ x, p x = (7/8) * x^2 - (13/4) * x + 3) →
  p (-2) = 13 ∧ p 0 = 3 ∧ p 2 = 0 := by
  sorry

end quadratic_polynomial_conditions_l992_99256


namespace no_square_divisible_by_six_between_50_and_120_l992_99219

theorem no_square_divisible_by_six_between_50_and_120 : 
  ¬ ∃ x : ℕ, x^2 = x ∧ x % 6 = 0 ∧ 50 < x ∧ x < 120 := by
sorry

end no_square_divisible_by_six_between_50_and_120_l992_99219


namespace expected_black_pairs_in_deck_l992_99221

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

theorem expected_black_pairs_in_deck : 
  expected_black_pairs 52 30 = 870 / 51 := by
  sorry

end expected_black_pairs_in_deck_l992_99221


namespace sin_10_50_70_product_l992_99210

theorem sin_10_50_70_product : Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end sin_10_50_70_product_l992_99210


namespace product_divisible_by_sum_implies_inequality_l992_99209

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : m + n ≤ n^2 := by
  sorry

end product_divisible_by_sum_implies_inequality_l992_99209


namespace cos_equality_theorem_l992_99245

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (310 * π / 180)) ↔ (n = 50 ∨ n = 310) :=
by sorry

end cos_equality_theorem_l992_99245


namespace book_arrangement_proof_l992_99288

def arrange_books (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

theorem book_arrangement_proof :
  arrange_books 5 = 252 := by
  sorry

end book_arrangement_proof_l992_99288


namespace helens_hotdogs_count_l992_99211

/-- The number of hotdogs Dylan's mother brought -/
def dylans_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := 480

/-- The number of hotdogs Helen's mother brought -/
def helens_hotdogs : ℕ := total_hotdogs - dylans_hotdogs

theorem helens_hotdogs_count : helens_hotdogs = 101 := by
  sorry

end helens_hotdogs_count_l992_99211


namespace smallest_coin_arrangement_l992_99252

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The number of proper divisors of a positive integer greater than 2 -/
def num_proper_divisors_gt_2 (n : ℕ+) : ℕ := sorry

/-- Checks if all divisors d of n where 2 < d < n, n/d is an integer -/
def all_divisors_divide (n : ℕ+) : Prop := sorry

theorem smallest_coin_arrangement :
  ∃ (n : ℕ+), num_divisors n = 19 ∧ 
              num_proper_divisors_gt_2 n = 17 ∧ 
              all_divisors_divide n ∧
              (∀ m : ℕ+, m < n → 
                (num_divisors m ≠ 19 ∨ 
                 num_proper_divisors_gt_2 m ≠ 17 ∨ 
                 ¬all_divisors_divide m)) ∧
              n = 2700 := by sorry

end smallest_coin_arrangement_l992_99252


namespace cubic_projection_equality_l992_99220

/-- A cubic function -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem: For a cubic function and two horizontal lines intersecting it, 
    the difference between the middle x-coordinates equals the sum of the 
    differences between the outer x-coordinates. -/
theorem cubic_projection_equality 
  (a b c d : ℝ) 
  (x₁ x₂ x₃ X₁ X₂ X₃ : ℝ) 
  (y₁ y₂ Y₁ Y₂ : ℝ) 
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃) 
  (h₃ : X₁ < X₂) (h₄ : X₂ < X₃) 
  (h₅ : cubic_function a b c d x₁ = y₁) 
  (h₆ : cubic_function a b c d x₂ = y₁) 
  (h₇ : cubic_function a b c d x₃ = y₁) 
  (h₈ : cubic_function a b c d X₁ = Y₁) 
  (h₉ : cubic_function a b c d X₂ = Y₁) 
  (h₁₀ : cubic_function a b c d X₃ = Y₁) :
  x₂ - X₂ = (X₁ - x₁) + (X₃ - x₃) := by sorry

end cubic_projection_equality_l992_99220


namespace blue_lipstick_count_l992_99224

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h3 : ∃ red_lipstick_wearers : ℕ, red_lipstick_wearers = lipstick_wearers / 4)
  (h4 : ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = red_lipstick_wearers / 5) :
  ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = 5 := by
sorry

end blue_lipstick_count_l992_99224


namespace root_ratio_equality_l992_99234

/-- 
Given a complex polynomial z^4 + az^3 + bz^2 + cz + d with roots p, q, r, s,
if a^2d = c^2 and c ≠ 0, then p/r = s/q.
-/
theorem root_ratio_equality (a b c d p q r s : ℂ) : 
  p * q * r * s = d → 
  p + q + r + s = -a → 
  a^2 * d = c^2 → 
  c ≠ 0 → 
  p / r = s / q := by
sorry

end root_ratio_equality_l992_99234


namespace smallest_square_side_exists_valid_division_5_l992_99298

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a division of a square into smaller squares -/
structure SquareDivision where
  original : Square
  parts : List Square
  sum_areas : (parts.map (λ s => s.side ^ 2)).sum = original.side ^ 2

/-- The property we want to prove -/
def is_valid_division (d : SquareDivision) : Prop :=
  d.parts.length = 15 ∧
  (d.parts.filter (λ s => s.side = 1)).length ≥ 12

/-- The main theorem -/
theorem smallest_square_side :
  ∀ d : SquareDivision, is_valid_division d → d.original.side ≥ 5 :=
by sorry

/-- The existence of a valid division with side 5 -/
theorem exists_valid_division_5 :
  ∃ d : SquareDivision, d.original.side = 5 ∧ is_valid_division d :=
by sorry

end smallest_square_side_exists_valid_division_5_l992_99298


namespace abc_product_l992_99267

theorem abc_product (a b c : ℝ) (h1 : b + c = 3) (h2 : c + a = 6) (h3 : a + b = 7) : a * b * c = 10 := by
  sorry

end abc_product_l992_99267


namespace circle_radius_problem_l992_99249

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

theorem circle_radius_problem (A B C D : Circle) :
  are_externally_tangent A B ∧
  are_externally_tangent A C ∧
  are_externally_tangent B C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 1 ∧
  (let (x, y) := D.center; (x - A.center.1)^2 + (y - A.center.2)^2 = A.radius^2) →
  B.radius = 8/9 := by
sorry

end circle_radius_problem_l992_99249


namespace students_with_d_grade_l992_99237

theorem students_with_d_grade (total_students : ℕ) 
  (a_fraction b_fraction c_fraction : ℚ) : 
  total_students = 800 →
  a_fraction = 1/5 →
  b_fraction = 1/4 →
  c_fraction = 1/2 →
  total_students - (total_students * a_fraction + total_students * b_fraction + total_students * c_fraction) = 40 := by
  sorry

end students_with_d_grade_l992_99237


namespace distance_between_points_l992_99228

theorem distance_between_points (x : ℝ) : 
  let A := 3 + x
  let B := 3 - x
  |A - B| = 8 → |x| = 4 := by
sorry

end distance_between_points_l992_99228


namespace fraction_value_l992_99214

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end fraction_value_l992_99214


namespace arithmetic_sequence_third_term_l992_99239

theorem arithmetic_sequence_third_term (a x : ℝ) : 
  a + (a + 2*x) = 6 → a + x = 3 := by
  sorry

end arithmetic_sequence_third_term_l992_99239


namespace green_brunette_percentage_is_54_l992_99253

/-- Represents the hair and eye color distribution of an island's population -/
structure IslandPopulation where
  blueBrunettes : ℕ
  blueBlondes : ℕ
  greenBlondes : ℕ
  greenBrunettes : ℕ

/-- The proportion of brunettes among blue-eyed inhabitants is 65% -/
def blueBrunettesProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes) = 13 / 20

/-- The proportion of blue-eyed among blondes is 70% -/
def blueBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBlondes : ℚ) / (pop.blueBlondes + pop.greenBlondes) = 7 / 10

/-- The proportion of blondes among green-eyed inhabitants is 10% -/
def greenBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.greenBlondes : ℚ) / (pop.greenBlondes + pop.greenBrunettes) = 1 / 10

/-- The percentage of green-eyed brunettes in the total population -/
def greenBrunettePercentage (pop : IslandPopulation) : ℚ :=
  (pop.greenBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes + pop.greenBlondes + pop.greenBrunettes) * 100

/-- Theorem stating that the percentage of green-eyed brunettes is 54% -/
theorem green_brunette_percentage_is_54 (pop : IslandPopulation) :
  blueBrunettesProportion pop → blueBlondeProportion pop → greenBlondeProportion pop →
  greenBrunettePercentage pop = 54 := by
  sorry

end green_brunette_percentage_is_54_l992_99253


namespace range_of_a_l992_99218

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (¬p x a → ¬q x)) →
  -1 ≤ a ∧ a ≤ 6 := by
  sorry

end range_of_a_l992_99218


namespace parabola_equation_l992_99295

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line x = 2
def Line (x : ℝ) : Prop := x = 2

-- Define the intersection points D and E
def Intersect (p : ℝ) (D E : ℝ × ℝ) : Prop :=
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1

-- Define the orthogonality condition
def Orthogonal (O D E : ℝ × ℝ) : Prop :=
  (D.1 - O.1) * (E.1 - O.1) + (D.2 - O.2) * (E.2 - O.2) = 0

-- The main theorem
theorem parabola_equation (p : ℝ) (D E : ℝ × ℝ) :
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1 ∧ 
  Orthogonal (0, 0) D E →
  ∀ x y : ℝ, C p x y ↔ y^2 = 2*x :=
sorry

end parabola_equation_l992_99295


namespace set_equality_l992_99283

theorem set_equality : 
  let M : Set ℝ := {3, 2}
  let N : Set ℝ := {x | x^2 - 5*x + 6 = 0}
  M = N := by sorry

end set_equality_l992_99283


namespace circle_center_sum_l992_99254

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = 7 := by
sorry

end circle_center_sum_l992_99254


namespace complex_power_difference_l992_99285

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := by
  sorry

end complex_power_difference_l992_99285


namespace age_difference_l992_99247

theorem age_difference (tyson_age frederick_age julian_age kyle_age : ℕ) : 
  tyson_age = 20 →
  frederick_age = 2 * tyson_age →
  julian_age = frederick_age - 20 →
  kyle_age = 25 →
  kyle_age - julian_age = 5 := by
sorry

end age_difference_l992_99247


namespace permutations_of_47722_l992_99207

def digits : List ℕ := [4, 7, 7, 2, 2]

theorem permutations_of_47722 : Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2) = 30 := by
  sorry

end permutations_of_47722_l992_99207
