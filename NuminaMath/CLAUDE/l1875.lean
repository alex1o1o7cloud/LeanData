import Mathlib

namespace solve_equation_l1875_187587

theorem solve_equation : 45 / (7 - 3/4) = 36/5 := by sorry

end solve_equation_l1875_187587


namespace simplify_fraction_product_l1875_187570

theorem simplify_fraction_product : 
  (360 : ℚ) / 24 * (10 : ℚ) / 240 * (6 : ℚ) / 3 * (9 : ℚ) / 18 = 5 / 8 := by
  sorry

end simplify_fraction_product_l1875_187570


namespace paving_cost_calculation_l1875_187501

-- Define the room dimensions and paving rate
def room_length : Real := 5.5
def room_width : Real := 3.75
def paving_rate : Real := 400

-- Define the theorem
theorem paving_cost_calculation :
  let area : Real := room_length * room_width
  let cost : Real := area * paving_rate
  cost = 8250 := by
  sorry

end paving_cost_calculation_l1875_187501


namespace simplify_expression_l1875_187597

theorem simplify_expression : 18 * (8 / 15) * (1 / 12)^2 = 1 / 15 := by
  sorry

end simplify_expression_l1875_187597


namespace stream_speed_equation_l1875_187557

/-- The speed of the stream for a boat trip -/
theorem stream_speed_equation (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 9)
  (h2 : distance = 210)
  (h3 : total_time = 84) :
  ∃ x : ℝ, x^2 = 39 ∧ 
    (distance / (boat_speed + x) + distance / (boat_speed - x) = total_time) := by
  sorry

end stream_speed_equation_l1875_187557


namespace parabola_b_value_l1875_187508

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -2p), where p ≠ 0, 
    the value of b is 6/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) →
  (a * 0^2 + b * 0 + c = -2 * p) →
  b = 6 / p := by
  sorry

end parabola_b_value_l1875_187508


namespace de_length_l1875_187559

/-- Triangle ABC with sides AB = 24, AC = 26, and BC = 22 -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Points D and E on sides AB and AC respectively -/
structure PointsDE (T : Triangle) :=
  (D : ℝ) (E : ℝ)
  (hD : D ≥ 0 ∧ D ≤ T.AB)
  (hE : E ≥ 0 ∧ E ≤ T.AC)

/-- DE is parallel to BC and contains the center of the inscribed circle -/
def contains_incenter (T : Triangle) (P : PointsDE T) : Prop :=
  ∃ k : ℝ, P.D / T.AB = P.E / T.AC ∧ k > 0 ∧ k < 1 ∧
    P.D = k * T.AB ∧ P.E = k * T.AC

/-- The main theorem -/
theorem de_length (T : Triangle) (P : PointsDE T) 
    (h1 : T.AB = 24) (h2 : T.AC = 26) (h3 : T.BC = 22)
    (h4 : contains_incenter T P) : 
  P.E - P.D = 275 / 18 := by sorry

end de_length_l1875_187559


namespace players_who_quit_correct_players_who_quit_l1875_187586

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  initial_players - remaining_players

theorem correct_players_who_quit :
  players_who_quit 10 8 24 = 7 := by
  sorry

end players_who_quit_correct_players_who_quit_l1875_187586


namespace rain_probability_three_days_l1875_187516

theorem rain_probability_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.40) 
  (h2 : prob_saturday = 0.60) 
  (h3 : prob_sunday = 0.35) 
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1) 
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1) 
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.084 := by
sorry

end rain_probability_three_days_l1875_187516


namespace egg_order_problem_l1875_187556

theorem egg_order_problem (total : ℚ) : 
  (total > 0) →
  (total * (1 - 1/4) * (1 - 2/3) = 9) →
  total = 18 := by
sorry

end egg_order_problem_l1875_187556


namespace labeling_existence_condition_l1875_187590

/-- A labeling function for lattice points -/
def LabelingFunction := ℤ × ℤ → ℕ+

/-- The property that a labeling satisfies the distance condition for a given c -/
def SatisfiesDistanceCondition (f : LabelingFunction) (c : ℝ) : Prop :=
  ∀ i : ℕ+, ∀ p q : ℤ × ℤ, f p = i ∧ f q = i → dist p q ≥ c ^ (i : ℝ)

/-- The property that a labeling uses only finitely many labels -/
def UsesFiniteLabels (f : LabelingFunction) : Prop :=
  ∃ n : ℕ, ∀ p : ℤ × ℤ, (f p : ℕ) ≤ n

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧ c < Real.sqrt 2) ↔
  (∃ f : LabelingFunction, SatisfiesDistanceCondition f c ∧ UsesFiniteLabels f) :=
sorry

end labeling_existence_condition_l1875_187590


namespace sheila_attends_probability_l1875_187515

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_sheila_goes_rain : ℝ := 0.3
def prob_sheila_goes_sunny : ℝ := 0.7
def prob_friend_drives : ℝ := 0.5

-- Define the probability of Sheila attending the picnic
def prob_sheila_attends : ℝ :=
  (prob_rain * prob_sheila_goes_rain + prob_sunny * prob_sheila_goes_sunny) * prob_friend_drives

-- Theorem statement
theorem sheila_attends_probability :
  prob_sheila_attends = 0.25 := by
  sorry

end sheila_attends_probability_l1875_187515


namespace even_decreasing_inequality_l1875_187543

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) : 
  f 1 > f (-2) ∧ f (-2) > f 3 :=
sorry

end even_decreasing_inequality_l1875_187543


namespace shortest_distance_point_to_parabola_l1875_187548

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (7, 15)
  let parabola := λ x : ℝ => (x, x^2)
  ∃ d : ℝ, d = 2 * Real.sqrt 13 ∧
    ∀ x : ℝ, d ≤ Real.sqrt ((7 - x)^2 + (15 - x^2)^2) :=
by sorry

end shortest_distance_point_to_parabola_l1875_187548


namespace square_value_l1875_187585

theorem square_value (square q : ℤ) 
  (eq1 : square + q = 74)
  (eq2 : square + 2 * q ^ 2 = 180) : 
  square = 66 := by sorry

end square_value_l1875_187585


namespace square_roots_and_cube_root_problem_l1875_187551

theorem square_roots_and_cube_root_problem (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3 * a - 14)^2 = k ∧ (a - 2)^2 = k) → 
  ((b - 15)^(1/3) = -3) → 
  (a = 4 ∧ b = -12 ∧ (∀ x : ℝ, x^2 = 4*a + b ↔ x = 2 ∨ x = -2)) :=
by sorry

end square_roots_and_cube_root_problem_l1875_187551


namespace percentage_relation_l1875_187568

theorem percentage_relation (x a b : ℝ) (ha : a = 0.06 * x) (hb : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end percentage_relation_l1875_187568


namespace stock_purchase_probabilities_l1875_187535

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 6

/-- The number of individuals making purchases -/
def num_individuals : ℕ := 4

/-- The probability that all individuals purchase the same stock -/
def prob_all_same : ℚ := 1 / 216

/-- The probability that at most two individuals purchase the same stock -/
def prob_at_most_two_same : ℚ := 65 / 72

/-- Given 6 stocks and 4 individuals randomly selecting one stock each,
    prove the probabilities of certain outcomes -/
theorem stock_purchase_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_individuals - 1)) ∧
  (prob_at_most_two_same = 
    (num_stocks * (num_stocks - 1) * Nat.choose num_individuals 2 + 
     num_stocks * Nat.factorial num_individuals) / 
    (num_stocks ^ num_individuals)) := by
  sorry

end stock_purchase_probabilities_l1875_187535


namespace range_of_a_l1875_187569

theorem range_of_a (a : ℝ) : Real.sqrt (a^2) = -a → a ≤ 0 := by
  sorry

end range_of_a_l1875_187569


namespace abs_value_inequality_l1875_187502

theorem abs_value_inequality (x : ℝ) : 
  (|x - 2| + |x - 4| ≤ 3) ↔ (3/2 ≤ x ∧ x < 4) := by
  sorry

end abs_value_inequality_l1875_187502


namespace min_value_sum_squares_l1875_187572

theorem min_value_sum_squares (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ (min : ℝ), min = 12 ∧ a^2 + 4*b^2 + 9*c^2 ≥ min :=
by sorry

end min_value_sum_squares_l1875_187572


namespace extra_grass_seed_coverage_l1875_187596

/-- Calculates the extra coverage of grass seed after reseeding a lawn -/
theorem extra_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) : 
  lawn_length = 35 → 
  lawn_width = 48 → 
  seed_bags = 6 → 
  coverage_per_bag = 500 → 
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 1320 :=
by
  sorry

#check extra_grass_seed_coverage

end extra_grass_seed_coverage_l1875_187596


namespace inverse_variation_l1875_187524

/-- Given quantities a and b that vary inversely, if b = 0.5 when a = 800, 
    then b = 0.25 when a = 1600 -/
theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) 
  (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 := by
  sorry

end inverse_variation_l1875_187524


namespace sports_package_channels_l1875_187595

/-- The number of channels in Larry's cable package at different stages --/
structure CablePackage where
  initial : Nat
  after_replacement : Nat
  after_reduction : Nat
  after_sports : Nat
  after_supreme : Nat
  final : Nat

/-- The number of channels in the sports package --/
def sports_package (cp : CablePackage) : Nat :=
  cp.final - cp.after_supreme

theorem sports_package_channels : ∀ cp : CablePackage,
  cp.initial = 150 →
  cp.after_replacement = cp.initial - 20 + 12 →
  cp.after_reduction = cp.after_replacement - 10 →
  cp.after_supreme = cp.after_sports + 7 →
  cp.final = 147 →
  sports_package cp = 8 := by
  sorry

#eval sports_package { 
  initial := 150,
  after_replacement := 142,
  after_reduction := 132,
  after_sports := 140,
  after_supreme := 147,
  final := 147
}

end sports_package_channels_l1875_187595


namespace cookie_jar_spending_ratio_l1875_187528

/-- Proves that the ratio of Martha's spending to Doris' spending is 1:2 --/
theorem cookie_jar_spending_ratio 
  (initial_amount : ℕ) 
  (doris_spent : ℕ) 
  (final_amount : ℕ) 
  (h1 : initial_amount = 24)
  (h2 : doris_spent = 6)
  (h3 : final_amount = 15) :
  ∃ (martha_spent : ℕ), 
    martha_spent = initial_amount - doris_spent - final_amount ∧
    martha_spent * 2 = doris_spent := by
  sorry

#check cookie_jar_spending_ratio

end cookie_jar_spending_ratio_l1875_187528


namespace flower_bed_fraction_is_correct_l1875_187507

/-- Represents the dimensions and areas of a yard with a pool and flower beds. -/
structure YardLayout where
  yard_length : ℝ
  yard_width : ℝ
  pool_length : ℝ
  pool_width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ

/-- Calculates the fraction of usable yard area occupied by flower beds. -/
def flower_bed_fraction (layout : YardLayout) : ℚ :=
  sorry

/-- Theorem stating that the fraction of usable yard occupied by flower beds is 9/260. -/
theorem flower_bed_fraction_is_correct (layout : YardLayout) : 
  layout.yard_length = 30 ∧ 
  layout.yard_width = 10 ∧ 
  layout.pool_length = 10 ∧ 
  layout.pool_width = 4 ∧
  layout.trapezoid_side1 = 16 ∧ 
  layout.trapezoid_side2 = 22 →
  flower_bed_fraction layout = 9 / 260 :=
  sorry

end flower_bed_fraction_is_correct_l1875_187507


namespace min_segments_11x11_grid_l1875_187564

/-- Represents a grid of lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the number of internal nodes in a grid -/
def internal_nodes (g : Grid) : ℕ :=
  (g.horizontal_lines - 2) * (g.vertical_lines - 2)

/-- Calculates the minimum number of segments to erase -/
def min_segments_to_erase (g : Grid) : ℕ :=
  (internal_nodes g + 1) / 2

/-- The theorem stating the minimum number of segments to erase in an 11x11 grid -/
theorem min_segments_11x11_grid :
  ∃ (g : Grid), g.horizontal_lines = 11 ∧ g.vertical_lines = 11 ∧
  min_segments_to_erase g = 41 :=
sorry

end min_segments_11x11_grid_l1875_187564


namespace bowling_team_size_l1875_187576

/-- The number of players in a bowling team -/
def num_players : ℕ := sorry

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of rounds in a season -/
def num_rounds : ℕ := 10

/-- The team's current total score after 9 rounds -/
def current_score : ℕ := 10440

/-- The difference between the league record and the minimum average needed in the final round -/
def final_round_diff : ℕ := 27

theorem bowling_team_size :
  (num_players * league_record * num_rounds - current_score) / num_players = 
  league_record - final_round_diff ∧
  num_players = 4 := by sorry

end bowling_team_size_l1875_187576


namespace stephanie_oranges_l1875_187558

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := total_oranges / store_visits

theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end stephanie_oranges_l1875_187558


namespace geometric_sequence_formula_l1875_187538

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_q : ∃ q : ℝ, q ∈ Set.Ioo 0 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25)
  (h_mean : Real.sqrt (a 3 * a 5) = 2) :
  ∀ n : ℕ, a n = 2^(5 - n) :=
sorry

end geometric_sequence_formula_l1875_187538


namespace money_distribution_l1875_187567

/-- Given three people A, B, and C with money, prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 450 →
  B + C = 350 →
  C = 100 →
  A + C = 200 := by
  sorry

end money_distribution_l1875_187567


namespace pool_length_calculation_l1875_187571

/-- Calculates the length of a rectangular pool given its draining rate, width, depth, initial capacity, and time to drain. -/
theorem pool_length_calculation (drain_rate : ℝ) (width depth : ℝ) (initial_capacity : ℝ) (drain_time : ℝ) :
  drain_rate = 60 →
  width = 40 →
  depth = 10 →
  initial_capacity = 0.8 →
  drain_time = 800 →
  (drain_rate * drain_time) / initial_capacity / (width * depth) = 150 :=
by
  sorry

end pool_length_calculation_l1875_187571


namespace cone_slant_height_l1875_187578

/-- Given a cone with base radius 5 cm and unfolded side area 60π cm², 
    prove that its slant height is 12 cm -/
theorem cone_slant_height (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 5 → A = 60 * Real.pi → A = (Real.pi * r * l) → l = 12 := by
  sorry

end cone_slant_height_l1875_187578


namespace functional_equation_solution_l1875_187582

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4) :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := by
  sorry

end functional_equation_solution_l1875_187582


namespace fourth_group_number_l1875_187530

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (start : ℕ) (interval : ℕ) (group : ℕ) : ℕ :=
  start + (group - 1) * interval

/-- Theorem: In a systematic sampling of 90 students, with adjacent group numbers 14 and 23,
    the student number from the fourth group is 32. -/
theorem fourth_group_number :
  let total := 90
  let start := 14
  let interval := 23 - 14
  let group := 4
  systematic_sample total start interval group = 32 := by
  sorry

end fourth_group_number_l1875_187530


namespace bottles_drunk_per_day_l1875_187594

theorem bottles_drunk_per_day (initial_bottles : ℕ) (remaining_bottles : ℕ) (days : ℕ) : 
  initial_bottles = 301 → remaining_bottles = 157 → days = 1 →
  initial_bottles - remaining_bottles = 144 := by
sorry

end bottles_drunk_per_day_l1875_187594


namespace xyz_value_l1875_187517

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 16 * Real.rpow 4 (1/3))
  (h2 : x * z = 28 * Real.rpow 4 (1/3))
  (h3 : y * z = 112 / Real.rpow 4 (1/3)) :
  x * y * z = 112 * Real.sqrt 7 := by
sorry

end xyz_value_l1875_187517


namespace ava_distance_covered_l1875_187526

/-- Represents the race scenario where Aubrey and Ava are running --/
structure RaceScenario where
  race_length : ℝ  -- Length of the race in kilometers
  ava_remaining : ℝ  -- Distance Ava has left to finish in meters

/-- Calculates the distance Ava has covered in meters --/
def distance_covered (scenario : RaceScenario) : ℝ :=
  scenario.race_length * 1000 - scenario.ava_remaining

/-- Theorem stating that Ava covered 833 meters in the given scenario --/
theorem ava_distance_covered (scenario : RaceScenario)
  (h1 : scenario.race_length = 1)
  (h2 : scenario.ava_remaining = 167) :
  distance_covered scenario = 833 := by
  sorry

end ava_distance_covered_l1875_187526


namespace megan_museum_pictures_l1875_187520

/-- Represents the number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- Represents the number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- Represents the number of pictures Megan had left after deleting -/
def remaining_pictures : ℕ := 2

theorem megan_museum_pictures :
  zoo_pictures + museum_pictures = remaining_pictures + deleted_pictures :=
by sorry

end megan_museum_pictures_l1875_187520


namespace ball_purchase_solution_l1875_187509

/-- Represents the cost and quantity of soccer balls and basketballs -/
structure BallPurchase where
  soccer_cost : ℝ
  basketball_cost : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- Conditions for the ball purchase problem -/
def BallPurchaseConditions (bp : BallPurchase) : Prop :=
  7 * bp.soccer_cost = 5 * bp.basketball_cost ∧
  40 * bp.soccer_cost + 20 * bp.basketball_cost = 3400 ∧
  bp.soccer_quantity + bp.basketball_quantity = 100 ∧
  bp.soccer_cost * bp.soccer_quantity + bp.basketball_cost * bp.basketball_quantity ≤ 6300

/-- Theorem stating the solution to the ball purchase problem -/
theorem ball_purchase_solution (bp : BallPurchase) 
  (h : BallPurchaseConditions bp) : 
  bp.soccer_cost = 50 ∧ 
  bp.basketball_cost = 70 ∧ 
  bp.basketball_quantity ≤ 65 :=
sorry

end ball_purchase_solution_l1875_187509


namespace sample_size_accuracy_l1875_187574

theorem sample_size_accuracy (population : Type) (sample : Set population) (estimate : Set population → ℝ) (accuracy : Set population → ℝ) :
  ∀ s₁ s₂ : Set population, s₁ ⊆ s₂ → accuracy s₁ ≤ accuracy s₂ := by
  sorry

end sample_size_accuracy_l1875_187574


namespace even_sum_probability_l1875_187536

/-- Probability of obtaining an even sum when spinning two wheels -/
theorem even_sum_probability (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ)
  (h1 : wheel1_total = 6)
  (h2 : wheel1_even = 2)
  (h3 : wheel2_total = 5)
  (h4 : wheel2_even = 3) :
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total +
  ((wheel1_total - wheel1_even) : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total =
  7 / 15 :=
by sorry

end even_sum_probability_l1875_187536


namespace math_books_count_l1875_187579

theorem math_books_count (total_books : ℕ) (math_price history_price total_price : ℕ) :
  total_books = 80 →
  math_price = 4 →
  history_price = 5 →
  total_price = 373 →
  ∃ (math_books : ℕ), 
    math_books * math_price + (total_books - math_books) * history_price = total_price ∧
    math_books = 27 := by
  sorry

end math_books_count_l1875_187579


namespace x_cubed_coef_sum_l1875_187553

def binomial_coef (n k : ℕ) : ℤ := (-1)^k * (n.choose k)

def expansion_coef (n : ℕ) : ℤ := binomial_coef n 3

theorem x_cubed_coef_sum :
  expansion_coef 5 + expansion_coef 6 + expansion_coef 7 + expansion_coef 8 = -121 :=
by sorry

end x_cubed_coef_sum_l1875_187553


namespace monotonic_increasing_range_a_l1875_187514

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 12*x - 1

-- State the theorem
theorem monotonic_increasing_range_a :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc (-6) 6 := by
  sorry

end monotonic_increasing_range_a_l1875_187514


namespace sequence_constant_iff_perfect_square_l1875_187546

/-- S(n) is defined as n minus the largest perfect square less than or equal to n -/
def S (n : ℕ) : ℕ := n - (Nat.sqrt n) ^ 2

/-- The sequence a_n is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | n + 1 => a A n + S (a A n)

/-- A non-negative integer is a perfect square if it's equal to some integer squared -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2

/-- The main theorem: the sequence becomes constant iff A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  (∃ N : ℕ, ∀ n ≥ N, a A n = a A N) ↔ is_perfect_square A := by
  sorry

end sequence_constant_iff_perfect_square_l1875_187546


namespace total_colored_pencils_l1875_187500

/-- The number of colored pencils each person has -/
structure ColoredPencils where
  cheryl : ℕ
  cyrus : ℕ
  madeline : ℕ
  daniel : ℕ

/-- The conditions of the colored pencils problem -/
def colored_pencils_conditions (p : ColoredPencils) : Prop :=
  p.cheryl = 3 * p.cyrus ∧
  p.madeline = 63 ∧
  p.madeline * 2 = p.cheryl ∧
  p.daniel = ((p.cheryl + p.cyrus + p.madeline) * 25 + 99) / 100

/-- The theorem stating the total number of colored pencils -/
theorem total_colored_pencils (p : ColoredPencils) 
  (h : colored_pencils_conditions p) : 
  p.cheryl + p.cyrus + p.madeline + p.daniel = 289 := by
  sorry

end total_colored_pencils_l1875_187500


namespace common_factor_is_gcf_l1875_187542

-- Define the polynomial terms
def term1 (x y : ℤ) : ℤ := 7 * x^2 * y
def term2 (x y : ℤ) : ℤ := 21 * x * y^2

-- Define the common factor
def common_factor (x y : ℤ) : ℤ := 7 * x * y

-- Theorem statement
theorem common_factor_is_gcf :
  ∀ (x y : ℤ), 
    (∃ (a b : ℤ), term1 x y = common_factor x y * a ∧ term2 x y = common_factor x y * b) ∧
    (∀ (z : ℤ), (∃ (c d : ℤ), term1 x y = z * c ∧ term2 x y = z * d) → z ∣ common_factor x y) :=
sorry

end common_factor_is_gcf_l1875_187542


namespace emily_earnings_theorem_l1875_187506

/-- The amount of money Emily makes by selling chocolate bars -/
def emily_earnings (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem: Emily makes $77 by selling all but 4 bars from a box of 15 bars costing $7 each -/
theorem emily_earnings_theorem : emily_earnings 15 7 4 = 77 := by
  sorry

end emily_earnings_theorem_l1875_187506


namespace athlete_distance_difference_l1875_187539

theorem athlete_distance_difference : 
  let field_length : ℚ := 24
  let mary_fraction : ℚ := 3/8
  let edna_fraction : ℚ := 2/3
  let lucy_fraction : ℚ := 5/6
  let mary_distance : ℚ := field_length * mary_fraction
  let edna_distance : ℚ := mary_distance * edna_fraction
  let lucy_distance : ℚ := edna_distance * lucy_fraction
  mary_distance - lucy_distance = 4 := by
sorry

end athlete_distance_difference_l1875_187539


namespace least_sum_pqr_l1875_187527

theorem least_sum_pqr (p q r : ℕ) : 
  p > 1 → q > 1 → r > 1 → 
  17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1) →
  ∀ p' q' r' : ℕ, 
    p' > 1 → q' > 1 → r' > 1 → 
    17 * (p' + 1) = 28 * (q' + 1) ∧ 28 * (q' + 1) = 35 * (r' + 1) →
    p + q + r ≤ p' + q' + r' ∧ p + q + r = 290 :=
by sorry

end least_sum_pqr_l1875_187527


namespace negation_of_existence_l1875_187522

theorem negation_of_existence (T S : Type → Prop) : 
  (¬ ∃ x, T x ∧ S x) ↔ (∀ x, T x → ¬ S x) := by sorry

end negation_of_existence_l1875_187522


namespace two_fifths_percent_of_450_l1875_187529

theorem two_fifths_percent_of_450 : (2 / 5) / 100 * 450 = 1.8 := by
  sorry

end two_fifths_percent_of_450_l1875_187529


namespace smallest_B_for_divisibility_by_three_l1875_187523

def seven_digit_number (B : Nat) : Nat :=
  4000000 + B * 100000 + 803942

theorem smallest_B_for_divisibility_by_three :
  ∃ (B : Nat), B < 10 ∧ 
    seven_digit_number B % 3 = 0 ∧
    ∀ (C : Nat), C < B → seven_digit_number C % 3 ≠ 0 :=
by sorry

end smallest_B_for_divisibility_by_three_l1875_187523


namespace yuri_roll_less_than_yuko_l1875_187561

/-- Represents a player's dice roll in the board game -/
structure DiceRoll :=
  (d1 d2 d3 : Nat)

/-- The game state after both players have rolled -/
structure GameState :=
  (yuri_roll : DiceRoll)
  (yuko_roll : DiceRoll)
  (yuko_ahead : Bool)

/-- Calculate the sum of a dice roll -/
def roll_sum (roll : DiceRoll) : Nat :=
  roll.d1 + roll.d2 + roll.d3

/-- Theorem stating that if Yuko is ahead, Yuri's roll sum must be less than Yuko's -/
theorem yuri_roll_less_than_yuko (state : GameState) 
  (h1 : state.yuko_roll = DiceRoll.mk 1 5 6)
  (h2 : state.yuko_ahead = true) : 
  roll_sum state.yuri_roll < roll_sum state.yuko_roll :=
by
  sorry

end yuri_roll_less_than_yuko_l1875_187561


namespace ellipse_equation_form_l1875_187589

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  center_x : ℝ
  center_y : ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (e : Ellipse) : Prop :=
  e.center_x = 0 ∧
  e.center_y = 0 ∧
  e.foci_on_axes ∧
  e.eccentricity = Real.sqrt 3 / 2 ∧
  e.passes_through = (2, 0)

-- Define the equation of the ellipse
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.center_x)^2 / e.a^2 + (y - e.center_y)^2 / e.b^2 = 1

-- Theorem statement
theorem ellipse_equation_form (e : Ellipse) :
  satisfies_conditions e →
  (∀ x y, ellipse_equation e x y ↔ (x^2 / 4 + y^2 = 1 ∨ x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end ellipse_equation_form_l1875_187589


namespace circle_chord_triangles_l1875_187510

-- Define the number of points on the circle
def n : ℕ := 9

-- Define the number of chords
def num_chords : ℕ := n.choose 2

-- Define the number of intersections inside the circle
def num_intersections : ℕ := n.choose 4

-- Define the number of triangles formed by intersections
def num_triangles : ℕ := num_intersections.choose 3

-- Theorem statement
theorem circle_chord_triangles :
  num_triangles = 315750 :=
sorry

end circle_chord_triangles_l1875_187510


namespace solve_equation_l1875_187563

theorem solve_equation (r : ℚ) : (r - 45) / 2 = (3 - 2 * r) / 5 → r = 77 / 3 := by
  sorry

end solve_equation_l1875_187563


namespace lavender_bouquet_cost_l1875_187555

/-- The cost of a bouquet is directly proportional to the number of lavenders it contains. -/
def is_proportional (cost : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → cost n / n = cost m / m

/-- Given that a bouquet of 15 lavenders costs $25 and the price is directly proportional
    to the number of lavenders, prove that a bouquet of 50 lavenders costs $250/3. -/
theorem lavender_bouquet_cost (cost : ℕ → ℚ)
    (h_prop : is_proportional cost)
    (h_15 : cost 15 = 25) :
    cost 50 = 250 / 3 := by
  sorry

end lavender_bouquet_cost_l1875_187555


namespace two_cos_forty_five_equals_sqrt_two_l1875_187552

theorem two_cos_forty_five_equals_sqrt_two : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end two_cos_forty_five_equals_sqrt_two_l1875_187552


namespace circle_radius_from_polar_l1875_187549

/-- The radius of a circle defined by the polar equation ρ = 6cosθ is 3 -/
theorem circle_radius_from_polar (θ : ℝ) :
  let ρ := 6 * Real.cos θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    radius = 3 ∧
    ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
      x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end circle_radius_from_polar_l1875_187549


namespace inequality_solution_equivalence_l1875_187541

def satisfies_inequality (x : ℝ) : Prop :=
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4

def solution_set : Set ℝ :=
  {x | x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x}

theorem inequality_solution_equivalence :
  ∀ x : ℝ, satisfies_inequality x ↔ x ∈ solution_set :=
sorry

end inequality_solution_equivalence_l1875_187541


namespace blue_beads_count_l1875_187577

theorem blue_beads_count (total : ℕ) (blue_neighbors : ℕ) (green_neighbors : ℕ) :
  total = 30 →
  blue_neighbors = 26 →
  green_neighbors = 20 →
  ∃ blue_count : ℕ,
    blue_count = 18 ∧
    blue_count ≤ total ∧
    blue_count * 2 ≥ blue_neighbors ∧
    (total - blue_count) * 2 ≥ green_neighbors :=
by
  sorry


end blue_beads_count_l1875_187577


namespace sam_dimes_count_l1875_187503

def final_dimes (initial : ℕ) (dad_gave : ℕ) (mom_took : ℕ) (sister_sets : ℕ) (dimes_per_set : ℕ) : ℕ :=
  initial + dad_gave - mom_took + sister_sets * dimes_per_set

theorem sam_dimes_count :
  final_dimes 9 7 3 4 2 = 21 := by
  sorry

end sam_dimes_count_l1875_187503


namespace salad_ratio_l1875_187593

theorem salad_ratio (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : 
  mushrooms = 3 →
  cherry_tomatoes = 2 * mushrooms →
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  red_bacon_bits = 32 →
  (red_bacon_bits : ℚ) / bacon_bits = 1 / 3 := by
  sorry

end salad_ratio_l1875_187593


namespace m_range_theorem_l1875_187573

/-- Proposition p: The solution set of the inequality |x-1| > m-1 is ℝ -/
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1

/-- Proposition q: f(x) = -(5-2m)x is a decreasing function -/
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5 - 2*m)*x > -(5 - 2*m)*y

/-- Either p or q is true -/
def either_p_or_q (m : ℝ) : Prop := p m ∨ q m

/-- Both p and q are false propositions -/
def both_p_and_q_false (m : ℝ) : Prop := ¬(p m) ∧ ¬(q m)

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 2

theorem m_range_theorem :
  ∀ m : ℝ, (either_p_or_q m ∧ ¬(both_p_and_q_false m)) ↔ m_range m :=
by sorry

end m_range_theorem_l1875_187573


namespace power_mod_eleven_l1875_187562

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end power_mod_eleven_l1875_187562


namespace second_book_has_32_pictures_l1875_187534

/-- The number of pictures in the second coloring book -/
def second_book_pictures (first_book_pictures colored_pictures remaining_pictures : ℕ) : ℕ :=
  (colored_pictures + remaining_pictures) - first_book_pictures

/-- Theorem stating that the second coloring book has 32 pictures -/
theorem second_book_has_32_pictures :
  second_book_pictures 23 44 11 = 32 := by
  sorry

end second_book_has_32_pictures_l1875_187534


namespace orange_stack_problem_l1875_187544

/-- Calculates the number of oranges in a pyramid-like stack --/
def orangeStackSum (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let layers := min base_width base_length
  let layerSum (n : ℕ) : ℕ := (base_width - n + 1) * (base_length - n + 1)
  (List.range layers).map layerSum |>.sum

/-- The pyramid-like stack of oranges problem --/
theorem orange_stack_problem :
  orangeStackSum 5 8 = 100 := by
  sorry

end orange_stack_problem_l1875_187544


namespace height_difference_l1875_187537

theorem height_difference (height_a height_b : ℝ) :
  height_b = height_a * (1 + 66.67 / 100) →
  (height_b - height_a) / height_b * 100 = 40 := by
sorry

end height_difference_l1875_187537


namespace min_value_sum_fractions_l1875_187513

theorem min_value_sum_fractions (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  (∃ (x : ℝ), 0 < x → (x + x + k) / x + (x + x + k) / x + (x + x + k) / x = 9) :=
sorry

end min_value_sum_fractions_l1875_187513


namespace egg_problem_solution_l1875_187580

/-- Calculates the difference between perfect and cracked eggs given the initial conditions --/
def egg_difference (total_dozens : ℕ) (broken : ℕ) : ℕ :=
  let total := total_dozens * 12
  let cracked := 2 * broken
  let perfect := total - broken - cracked
  perfect - cracked

theorem egg_problem_solution :
  egg_difference 2 3 = 9 := by
  sorry

end egg_problem_solution_l1875_187580


namespace fraction_division_l1875_187592

theorem fraction_division : (4 : ℚ) / 5 / ((8 : ℚ) / 15) = 3 / 2 := by
  sorry

end fraction_division_l1875_187592


namespace solve_exponential_equation_l1875_187525

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 625 ∧ x = 2/3 := by
  sorry

end solve_exponential_equation_l1875_187525


namespace largest_multiple_of_8_with_negation_greater_than_neg_200_l1875_187547

theorem largest_multiple_of_8_with_negation_greater_than_neg_200 :
  ∃ (n : ℤ), n = 192 ∧ 
  (∀ (m : ℤ), m % 8 = 0 ∧ -m > -200 → m ≤ n) ∧
  192 % 8 = 0 ∧
  -192 > -200 :=
by sorry

end largest_multiple_of_8_with_negation_greater_than_neg_200_l1875_187547


namespace quadratic_factorization_l1875_187581

theorem quadratic_factorization (a b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end quadratic_factorization_l1875_187581


namespace probability_at_least_two_correct_l1875_187598

-- Define the number of questions and choices
def total_questions : ℕ := 30
def choices_per_question : ℕ := 6
def guessed_questions : ℕ := 5

-- Define the probability of a correct answer
def p_correct : ℚ := 1 / choices_per_question

-- Define the binomial probability function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem probability_at_least_two_correct :
  1 - binomial_prob guessed_questions 0 p_correct
    - binomial_prob guessed_questions 1 p_correct = 763 / 3888 := by
  sorry

end probability_at_least_two_correct_l1875_187598


namespace sam_has_sixteen_dimes_l1875_187540

/-- The number of dimes Sam has after receiving some from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Sam has 16 dimes after receiving some from his dad -/
theorem sam_has_sixteen_dimes : total_dimes 9 7 = 16 := by
  sorry

end sam_has_sixteen_dimes_l1875_187540


namespace real_root_range_l1875_187599

theorem real_root_range (a : ℝ) : 
  (∃ x : ℝ, (2 : ℝ)^(2*x) + (2 : ℝ)^x * a + a + 1 = 0) → 
  a ≤ 2 - 2 * Real.sqrt 2 := by
sorry

end real_root_range_l1875_187599


namespace rachel_book_count_l1875_187521

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end rachel_book_count_l1875_187521


namespace inequality_solution_l1875_187531

theorem inequality_solution (x : ℝ) :
  x > 2 →
  (((x - 2) ^ (x^2 - 6*x + 8)) > 1) ↔ (x > 2 ∧ x < 3) ∨ x > 4 :=
by sorry

end inequality_solution_l1875_187531


namespace water_tank_evaporation_l1875_187512

/-- Calculates the remaining water in a tank after evaporation --/
def remaining_water (initial_amount : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_amount - evaporation_rate * days

/-- Proves that 450 gallons remain after 50 days of evaporation --/
theorem water_tank_evaporation :
  remaining_water 500 1 50 = 450 := by
  sorry

end water_tank_evaporation_l1875_187512


namespace scientific_notation_of_505000_l1875_187591

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number to be represented in scientific notation -/
def number : ℝ := 505000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 5.05
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation of 505,000 is 5.05 × 10^5 -/
theorem scientific_notation_of_505000 :
  toScientificNotation number = expected := by sorry

end scientific_notation_of_505000_l1875_187591


namespace special_number_divisibility_l1875_187584

/-- Represents a 4-digit number with the given properties -/
structure SpecialNumber where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  has_three_unique_digits : ∃ (a b c : Nat), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ((value / 1000 = a ∧ (value / 100) % 10 = a) ∨
     (value / 1000 = a ∧ (value / 10) % 10 = a) ∨
     (value / 1000 = a ∧ value % 10 = a) ∨
     ((value / 100) % 10 = a ∧ (value / 10) % 10 = a) ∨
     ((value / 100) % 10 = a ∧ value % 10 = a) ∨
     ((value / 10) % 10 = a ∧ value % 10 = a)) ∧
    value = a * 1000 + b * 100 + c * 10 + (if value / 1000 = a then b else a)

/-- Mrs. Smith's age is the last two digits of the special number -/
def mrs_smith_age (n : SpecialNumber) : Nat := n.value % 100

/-- The ages of Mrs. Smith's children -/
def children_ages : Finset Nat := Finset.range 12 \ {0}

theorem special_number_divisibility (n : SpecialNumber) :
  ∃ (x : Nat), x ∈ children_ages ∧ ¬(n.value % x = 0) ∧
  ∀ (y : Nat), y ∈ children_ages ∧ y ≠ x → n.value % y = 0 →
  x = 3 := by sorry

#check special_number_divisibility

end special_number_divisibility_l1875_187584


namespace marble_jar_problem_l1875_187504

theorem marble_jar_problem (num_marbles : ℕ) : 
  (∀ (x : ℚ), num_marbles / 24 = x → num_marbles / 26 = x - 1) →
  num_marbles = 312 := by
sorry

end marble_jar_problem_l1875_187504


namespace faye_remaining_money_l1875_187566

/-- Calculates the remaining money for Faye after her purchases -/
def remaining_money (initial_money : ℚ) (cupcake_price : ℚ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℚ) (cookie_box_quantity : ℕ) : ℚ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Theorem stating that Faye's remaining money is $30 -/
theorem faye_remaining_money :
  remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end faye_remaining_money_l1875_187566


namespace judy_caught_one_fish_l1875_187560

/-- Represents the number of fish caught by each family member and other fishing details -/
structure FishingTrip where
  ben_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Judy caught based on the fishing trip details -/
def judy_fish (trip : FishingTrip) : ℕ :=
  (trip.total_filets / trip.filets_per_fish) -
  (trip.ben_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back)

/-- Theorem stating that Judy caught 1 fish given the specific conditions of the fishing trip -/
theorem judy_caught_one_fish :
  let trip : FishingTrip := {
    ben_fish := 4,
    billy_fish := 3,
    jim_fish := 2,
    susie_fish := 5,
    thrown_back := 3,
    total_filets := 24,
    filets_per_fish := 2
  }
  judy_fish trip = 1 := by sorry

end judy_caught_one_fish_l1875_187560


namespace expression_simplification_l1875_187588

/-- Given real numbers x and y, prove that the expression
    ((x² + y²)(x² - y²)) / ((x² + y²) + (x² - y²)) + ((x² + y²) + (x² - y²)) / ((x² + y²)(x² - y²))
    simplifies to (x⁴ + y⁴)² / (2x²(x⁴ - y⁴)) -/
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  (P * Q) / (P + Q) + (P + Q) / (P * Q) = (x^4 + y^4)^2 / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end expression_simplification_l1875_187588


namespace sqrt_two_subset_P_l1875_187545

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by
  sorry

end sqrt_two_subset_P_l1875_187545


namespace sin_pi_sixth_minus_2alpha_l1875_187550

theorem sin_pi_sixth_minus_2alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end sin_pi_sixth_minus_2alpha_l1875_187550


namespace equal_segments_iff_proportion_l1875_187519

/-- A triangle with side lengths a, b, and c where a ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_c : a ≤ c

/-- The internal bisector of the incenter divides the median from point B into three equal segments -/
def has_equal_segments (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ 
    let m := (t.a^2 + t.c^2 - t.b^2/2) / 2
    (3*x)^2 = m ∧
    ((t.a + t.c - t.b)/2)^2 = 2*x^2 ∧
    ((t.c - t.a)/2)^2 = 2*x^2

/-- The side lengths satisfy the proportion a/5 = b/10 = c/13 -/
def satisfies_proportion (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 5*k ∧ t.b = 10*k ∧ t.c = 13*k

theorem equal_segments_iff_proportion (t : Triangle) :
  has_equal_segments t ↔ satisfies_proportion t := by
  sorry

end equal_segments_iff_proportion_l1875_187519


namespace ramesh_refrigerator_price_l1875_187533

/-- Represents the price Ramesh paid for a refrigerator given certain conditions --/
def ramesh_paid_price (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let transport_cost : ℝ := 125
  let installation_cost : ℝ := 250
  let profit_rate : ℝ := 0.10
  let selling_price : ℝ := 20350
  (1 + profit_rate) * P = selling_price ∧
  (1 - discount_rate) * P + transport_cost + installation_cost = 15175

theorem ramesh_refrigerator_price :
  ∃ P : ℝ, ramesh_paid_price P :=
sorry

end ramesh_refrigerator_price_l1875_187533


namespace sum_of_integers_l1875_187575

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end sum_of_integers_l1875_187575


namespace harry_fish_count_l1875_187583

/-- The number of fish Sam has -/
def sam_fish : ℕ := 7

/-- The number of fish Joe has relative to Sam -/
def joe_multiplier : ℕ := 8

/-- The number of fish Harry has relative to Joe -/
def harry_multiplier : ℕ := 4

/-- The number of fish Joe has -/
def joe_fish : ℕ := joe_multiplier * sam_fish

/-- The number of fish Harry has -/
def harry_fish : ℕ := harry_multiplier * joe_fish

theorem harry_fish_count : harry_fish = 224 := by
  sorry

end harry_fish_count_l1875_187583


namespace radius_C₁_is_sqrt_30_l1875_187532

/-- Two circles C₁ and C₂ with the following properties:
    1. The center O of C₁ lies on C₂
    2. C₁ and C₂ intersect at points X and Y
    3. There exists a point Z on C₂ exterior to C₁
    4. XZ = 13, OZ = 11, YZ = 7 -/
structure TwoCircles where
  O : ℝ × ℝ  -- Center of C₁
  X : ℝ × ℝ  -- Intersection point
  Y : ℝ × ℝ  -- Intersection point
  Z : ℝ × ℝ  -- Point on C₂ exterior to C₁
  C₁ : Set (ℝ × ℝ)  -- Circle C₁
  C₂ : Set (ℝ × ℝ)  -- Circle C₂
  O_on_C₂ : O ∈ C₂
  X_on_both : X ∈ C₁ ∧ X ∈ C₂
  Y_on_both : Y ∈ C₁ ∧ Y ∈ C₂
  Z_on_C₂ : Z ∈ C₂
  Z_exterior_C₁ : Z ∉ C₁
  XZ_length : dist X Z = 13
  OZ_length : dist O Z = 11
  YZ_length : dist Y Z = 7

/-- The radius of C₁ is √30 -/
theorem radius_C₁_is_sqrt_30 (tc : TwoCircles) : 
  ∃ (center : ℝ × ℝ) (r : ℝ), tc.C₁ = {p : ℝ × ℝ | dist p center = r} ∧ r = Real.sqrt 30 :=
sorry

end radius_C₁_is_sqrt_30_l1875_187532


namespace arithmetic_sequence_common_difference_l1875_187554

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ = -n² + 4n,
    prove that the common difference d is -2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- The given sum formula
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum function
  (h3 : ∀ n, a (n+1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -2 := by
  sorry

end arithmetic_sequence_common_difference_l1875_187554


namespace choose_two_from_eleven_l1875_187518

theorem choose_two_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 2 → Nat.choose n k = 55 := by
  sorry

end choose_two_from_eleven_l1875_187518


namespace smallest_three_digit_perfect_square_append_l1875_187511

theorem smallest_three_digit_perfect_square_append : 
  ∃ (a : ℕ), 
    (100 ≤ a ∧ a ≤ 999) ∧ 
    (∃ (n : ℕ), 1001 * a + 1 = n^2) ∧
    (∀ (b : ℕ), 100 ≤ b ∧ b < a → ¬∃ (m : ℕ), 1001 * b + 1 = m^2) ∧
    a = 183 :=
by sorry

end smallest_three_digit_perfect_square_append_l1875_187511


namespace cube_volume_from_surface_area_l1875_187565

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end cube_volume_from_surface_area_l1875_187565


namespace road_building_divisibility_l1875_187505

/-- Represents the number of ways to build roads between n cities with the given constraints -/
def T (n : ℕ) : ℕ :=
  sorry  -- Definition of T_n based on the problem constraints

/-- The main theorem to be proved -/
theorem road_building_divisibility (n : ℕ) (h : n > 1) :
  (n % 2 = 1 → n ∣ T n) ∧ (n % 2 = 0 → (n / 2) ∣ T n) :=
by sorry

end road_building_divisibility_l1875_187505
