import Mathlib

namespace NUMINAMATH_CALUDE_reseating_twelve_women_l2719_271992

def reseating_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- Consider the empty case as 1
  | 1 => 1
  | 2 => 3
  | n + 3 => reseating_ways (n + 2) + reseating_ways (n + 1) + reseating_ways n

theorem reseating_twelve_women :
  reseating_ways 12 = 1201 := by
  sorry

end NUMINAMATH_CALUDE_reseating_twelve_women_l2719_271992


namespace NUMINAMATH_CALUDE_proportion_solution_l2719_271934

theorem proportion_solution : ∃ x : ℚ, (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2719_271934


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2719_271907

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x - 4| > 6 ↔ x < 0 ∨ x > 12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2719_271907


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2719_271981

theorem quadratic_factorization_sum (a w c d : ℝ) : 
  (∀ x, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2719_271981


namespace NUMINAMATH_CALUDE_prob_ace_ten_jack_standard_deck_l2719_271938

/-- Represents a standard deck of 52 playing cards. -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (tens : Nat)
  (jacks : Nat)

/-- The probability of drawing an Ace, then a 10, then a Jack from a standard 52-card deck without replacement. -/
def prob_ace_ten_jack (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.cards *
  (d.tens : ℚ) / (d.cards - 1) *
  (d.jacks : ℚ) / (d.cards - 2)

/-- Theorem stating that the probability of drawing an Ace, then a 10, then a Jack
    from a standard 52-card deck without replacement is 8/16575. -/
theorem prob_ace_ten_jack_standard_deck :
  prob_ace_ten_jack ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_ten_jack_standard_deck_l2719_271938


namespace NUMINAMATH_CALUDE_union_equals_M_l2719_271976

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≠ 0}

theorem union_equals_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_equals_M_l2719_271976


namespace NUMINAMATH_CALUDE_f_below_tangent_and_inequality_l2719_271989

-- Define the function f(x) = (2-x)e^x
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the tangent line l(x) = x + 2
def l (x : ℝ) : ℝ := x + 2

theorem f_below_tangent_and_inequality (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ l x) ∧
  (f (1 / n - 1 / (n + 1)) + (1 / Real.exp 2) * f (2 - 1 / n) ≤ 2 + 1 / n) := by
  sorry

end NUMINAMATH_CALUDE_f_below_tangent_and_inequality_l2719_271989


namespace NUMINAMATH_CALUDE_train_length_calculation_l2719_271972

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (overtake_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  overtake_time = 45 →
  ∃ (train_length : ℝ), train_length = 62.5 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l2719_271972


namespace NUMINAMATH_CALUDE_triangle_count_l2719_271923

theorem triangle_count : ∃ (n : ℕ), n = 36 ∧ 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ p.2 ∧ p.1 + p.2 > 11) 
    (Finset.product (Finset.range 12) (Finset.range 12))).card :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_l2719_271923


namespace NUMINAMATH_CALUDE_lawn_area_20_l2719_271969

/-- The area of a rectangular lawn with given width and length -/
def lawn_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular lawn with width 5 feet and length 4 feet is 20 square feet -/
theorem lawn_area_20 : lawn_area 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lawn_area_20_l2719_271969


namespace NUMINAMATH_CALUDE_jane_earnings_l2719_271959

/-- Calculates the earnings from selling eggs over a given number of weeks -/
def egg_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let eggs_per_week := num_chickens * eggs_per_chicken
  let dozens_per_week := eggs_per_week / 12
  let earnings_per_week := dozens_per_week * price_per_dozen
  earnings_per_week * num_weeks

/-- Proves that Jane's earnings from selling eggs over two weeks is $20 -/
theorem jane_earnings :
  egg_earnings 10 6 2 2 = 20 := by
  sorry

#eval egg_earnings 10 6 2 2

end NUMINAMATH_CALUDE_jane_earnings_l2719_271959


namespace NUMINAMATH_CALUDE_marie_erasers_l2719_271971

/-- Given Marie's eraser situation, prove that she ends up with 755 erasers. -/
theorem marie_erasers (initial : ℕ) (lost : ℕ) (packs_bought : ℕ) (erasers_per_pack : ℕ) 
  (h1 : initial = 950)
  (h2 : lost = 420)
  (h3 : packs_bought = 3)
  (h4 : erasers_per_pack = 75) : 
  initial - lost + packs_bought * erasers_per_pack = 755 := by
  sorry

#check marie_erasers

end NUMINAMATH_CALUDE_marie_erasers_l2719_271971


namespace NUMINAMATH_CALUDE_alice_nike_sales_alice_nike_sales_proof_l2719_271954

/-- Proves that Alice sold 8 Nike shoes given the problem conditions -/
theorem alice_nike_sales : Int → Prop :=
  fun x =>
    let quota : Int := 1000
    let adidas_price : Int := 45
    let nike_price : Int := 60
    let reebok_price : Int := 35
    let adidas_sold : Int := 6
    let reebok_sold : Int := 9
    let over_goal : Int := 65
    (adidas_price * adidas_sold + nike_price * x + reebok_price * reebok_sold = quota + over_goal) →
    x = 8

/-- Proof of the theorem -/
theorem alice_nike_sales_proof : ∃ x, alice_nike_sales x :=
  sorry

end NUMINAMATH_CALUDE_alice_nike_sales_alice_nike_sales_proof_l2719_271954


namespace NUMINAMATH_CALUDE_point_division_l2719_271916

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB
def on_line_segment (P A B : V) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the ratio condition
def ratio_condition (P A B : V) : Prop := ∃ (k : ℝ), k > 0 ∧ 2 • (P - A) = k • (B - P) ∧ 7 • (P - A) = k • (B - P)

-- Theorem statement
theorem point_division (h1 : on_line_segment P A B) (h2 : ratio_condition P A B) :
  P = (7/9 : ℝ) • A + (2/9 : ℝ) • B :=
sorry

end NUMINAMATH_CALUDE_point_division_l2719_271916


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_existence_l2719_271941

theorem geometric_arithmetic_progression_existence :
  ∃ (q : ℝ) (i j k : ℕ), 
    1 < q ∧ 
    i < j ∧ j < k ∧ 
    q^j - q^i = q^k - q^j ∧
    1.9 < q :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_existence_l2719_271941


namespace NUMINAMATH_CALUDE_wendys_recycling_points_l2719_271973

/-- Given that Wendy earns 5 points per recycled bag, had 11 bags, and didn't recycle 2 bags,
    prove that she earned 45 points. -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_wendys_recycling_points_l2719_271973


namespace NUMINAMATH_CALUDE_product_coefficient_sum_l2719_271977

theorem product_coefficient_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  12 * a + 6 * b + 3 * c + d = -27 := by
sorry

end NUMINAMATH_CALUDE_product_coefficient_sum_l2719_271977


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2719_271987

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_even_number_of_factors (n : ℕ) : Prop :=
  Even (Nat.card (Nat.divisors n))

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    ((has_even_number_of_factors n ∨ n > 50) ∧
     ¬(has_even_number_of_factors n ∧ n > 50)) ∧
    ((Odd n ∨ n > 60) ∧ ¬(Odd n ∧ n > 60)) ∧
    ((Even n ∨ n > 70) ∧ ¬(Even n ∧ n > 70)) ∧
    n = 64 :=
by
  sorry

#check unique_number_satisfying_conditions

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2719_271987


namespace NUMINAMATH_CALUDE_even_function_value_l2719_271932

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_value (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : ∀ x, f (x + 2) * f x = 4)
  (h_positive : ∀ x, f x > 0) :
  f 2017 = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_value_l2719_271932


namespace NUMINAMATH_CALUDE_rocky_day1_miles_l2719_271993

def rocky_training (day1 : ℝ) : Prop :=
  let day2 := 2 * day1
  let day3 := 3 * day2
  day1 + day2 + day3 = 36

theorem rocky_day1_miles : ∃ (x : ℝ), rocky_training x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_rocky_day1_miles_l2719_271993


namespace NUMINAMATH_CALUDE_certain_number_problem_l2719_271927

theorem certain_number_problem (x : ℤ) (h : x + 14 = 56) : 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2719_271927


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l2719_271975

-- Equation 1
theorem solution_equation_one : 
  ∀ x : ℝ, 2 * x^2 - 4 * x - 1 = 0 ↔ x = 1 + Real.sqrt 6 / 2 ∨ x = 1 - Real.sqrt 6 / 2 := by
sorry

-- Equation 2
theorem solution_equation_two :
  ∀ x : ℝ, (x - 1) * (x + 2) = 28 ↔ x = -6 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l2719_271975


namespace NUMINAMATH_CALUDE_root_existence_l2719_271998

theorem root_existence (h1 : Real.log 1.5 < 4/11) (h2 : Real.log 2 > 2/7) :
  ∃ x : ℝ, 1/4 < x ∧ x < 1/2 ∧ Real.log (2*x + 1) = 1 / (3*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l2719_271998


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l2719_271931

theorem baker_cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) : 
  initial_cakes = 110 →
  additional_cakes = 76 →
  remaining_cakes = 111 →
  initial_cakes + additional_cakes - remaining_cakes = 75 := by
sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l2719_271931


namespace NUMINAMATH_CALUDE_family_ages_solution_l2719_271908

/-- Represents the ages of the family members -/
structure FamilyAges where
  father : ℕ
  person : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ∃ u : ℕ,
    ages.father + 6 = 3 * (ages.person - u) ∧
    ages.father = ages.person + ages.sister - u ∧
    ages.person = ages.father - u ∧
    ages.father + 19 = 2 * ages.sister

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ ages : FamilyAges,
    satisfiesConditions ages ∧
    ages.father = 69 ∧
    ages.person = 47 ∧
    ages.sister = 44 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l2719_271908


namespace NUMINAMATH_CALUDE_percentage_difference_l2719_271974

theorem percentage_difference (x y z : ℝ) : 
  x = 1.2 * y ∧ x = 0.36 * z → y = 0.3 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2719_271974


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2719_271996

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 9) = 12 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2719_271996


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l2719_271901

theorem arctan_sum_equation (x : ℝ) : 
  3 * Real.arctan (1/4) + Real.arctan (1/20) + Real.arctan (1/x) = π/4 → x = 1985 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l2719_271901


namespace NUMINAMATH_CALUDE_smallest_n_with_hcf_condition_l2719_271910

theorem smallest_n_with_hcf_condition : 
  ∃ (n : ℕ), n > 0 ∧ n ≠ 11 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n ∧ m ≠ 11 → Nat.gcd (m - 11) (3 * m + 20) = 1) ∧
  Nat.gcd (n - 11) (3 * n + 20) > 1 ∧
  n = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_hcf_condition_l2719_271910


namespace NUMINAMATH_CALUDE_exponential_graph_condition_l2719_271909

/-- A function f : ℝ → ℝ does not pass through the first quadrant if
    for all x > 0, f(x) ≤ 0 or for all x ≥ 0, f(x) < 0 -/
def not_pass_first_quadrant (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x ≤ 0) ∨ (∀ x ≥ 0, f x < 0)

theorem exponential_graph_condition
  (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (h : not_pass_first_quadrant (fun x ↦ a^x + b - 1)) :
  0 < a ∧ a < 1 ∧ b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exponential_graph_condition_l2719_271909


namespace NUMINAMATH_CALUDE_green_balloons_l2719_271952

theorem green_balloons (total : Nat) (red : Nat) (green : Nat) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_l2719_271952


namespace NUMINAMATH_CALUDE_running_time_ratio_l2719_271986

theorem running_time_ratio (danny_time steve_time : ℝ) 
  (h1 : danny_time = 25)
  (h2 : steve_time / 2 + 12.5 = danny_time) : 
  danny_time / steve_time = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_running_time_ratio_l2719_271986


namespace NUMINAMATH_CALUDE_naoh_combined_is_54_l2719_271968

/-- Represents the balanced chemical equation coefficients -/
structure BalancedEquation :=
  (naoh_coeff : ℕ)
  (h2so4_coeff : ℕ)
  (h2o_coeff : ℕ)

/-- Represents the given information about the reaction -/
structure ReactionInfo :=
  (h2so4_available : ℕ)
  (h2o_formed : ℕ)
  (equation : BalancedEquation)

/-- Calculates the number of moles of NaOH combined in the reaction -/
def naoh_combined (info : ReactionInfo) : ℕ :=
  info.h2o_formed * info.equation.naoh_coeff / info.equation.h2o_coeff

/-- Theorem stating that given the reaction information, 54 moles of NaOH were combined -/
theorem naoh_combined_is_54 (info : ReactionInfo) 
  (h_h2so4 : info.h2so4_available = 3)
  (h_h2o : info.h2o_formed = 54)
  (h_eq : info.equation = {naoh_coeff := 2, h2so4_coeff := 1, h2o_coeff := 2}) :
  naoh_combined info = 54 := by
  sorry

end NUMINAMATH_CALUDE_naoh_combined_is_54_l2719_271968


namespace NUMINAMATH_CALUDE_last_element_proof_l2719_271925

def first_row (n : ℕ) : ℕ := 2*n - 1

def third_row (n : ℕ) : ℕ := (first_row n) * (first_row n)^2 - (first_row n)

theorem last_element_proof : third_row 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_last_element_proof_l2719_271925


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2719_271913

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Creates a playlist with 12 songs, where each song is 20 seconds longer than the previous one -/
def createPlaylist : Playlist :=
  List.range 12 |>.map (fun i => 20 * (i + 1))

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total listening time in seconds -/
def totalListeningTime : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song within the first 5 minutes -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) : ℚ :=
  let totalArrangements := Nat.factorial playlist.length
  let favorableArrangements := 3 * Nat.factorial (playlist.length - 2)
  1 - (favorableArrangements : ℚ) / totalArrangements

theorem probability_not_hearing_favorite_song :
  probabilityNotHearingFavoriteSong createPlaylist = 43 / 44 := by
  sorry

#eval probabilityNotHearingFavoriteSong createPlaylist

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2719_271913


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2719_271915

/-- Given a point and a line, find the equation of a parallel line passing through the point. -/
theorem parallel_line_equation (x₀ y₀ : ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, (x = x₀ ∧ y = y₀) ∨ (a * x + b * y + k = 0) ↔ 
  (x = -3 ∧ y = -1) ∨ (x - 3 * y = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2719_271915


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2719_271966

theorem diophantine_equation_solution :
  ∀ x y : ℕ+,
  let d := Nat.gcd x.val y.val
  x.val * y.val * d = x.val + y.val + d^2 →
  (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2719_271966


namespace NUMINAMATH_CALUDE_minimum_race_distance_l2719_271943

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (a_to_wall : ℝ) (wall_to_b : ℝ) :
  wall_length = 1500 ∧ a_to_wall = 400 ∧ wall_to_b = 600 →
  ⌊Real.sqrt (wall_length ^ 2 + (a_to_wall + wall_to_b) ^ 2) + 0.5⌋ = 1803 := by
  sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l2719_271943


namespace NUMINAMATH_CALUDE_opposite_player_no_aces_l2719_271942

/-- The number of cards in the deck -/
def deck_size : ℕ := 32

/-- The number of players -/
def num_players : ℕ := 4

/-- The number of cards each player receives -/
def cards_per_player : ℕ := deck_size / num_players

/-- The number of aces in the deck -/
def num_aces : ℕ := 4

/-- The probability that the opposite player has no aces given that one player has no aces -/
def opposite_player_no_aces_prob : ℚ := 130 / 759

theorem opposite_player_no_aces (h1 : deck_size = 32) 
                                (h2 : num_players = 4) 
                                (h3 : cards_per_player = deck_size / num_players) 
                                (h4 : num_aces = 4) : 
  opposite_player_no_aces_prob = 130 / 759 := by
  sorry

end NUMINAMATH_CALUDE_opposite_player_no_aces_l2719_271942


namespace NUMINAMATH_CALUDE_interview_pass_probability_l2719_271921

-- Define the number of questions
def num_questions : ℕ := 3

-- Define the probability of answering a question correctly
def prob_correct : ℝ := 0.7

-- Define the number of attempts per question
def num_attempts : ℕ := 3

-- Theorem statement
theorem interview_pass_probability :
  1 - (1 - prob_correct) ^ num_attempts = 0.973 := by
  sorry

end NUMINAMATH_CALUDE_interview_pass_probability_l2719_271921


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2719_271970

theorem smallest_perfect_square_divisible_by_3_and_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ 3 ∣ n ∧ 5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, m = j^2) → 3 ∣ m → 5 ∣ m → m ≥ n :=
by
  -- Proof goes here
  sorry

#eval (15 : ℕ)^2  -- Expected output: 225

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2719_271970


namespace NUMINAMATH_CALUDE_abs_rational_nonnegative_l2719_271951

theorem abs_rational_nonnegative (x : ℚ) : 0 ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_abs_rational_nonnegative_l2719_271951


namespace NUMINAMATH_CALUDE_intersection_M_N_l2719_271965

def M : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≤ 0}

def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2719_271965


namespace NUMINAMATH_CALUDE_sum_product_bounds_l2719_271928

theorem sum_product_bounds (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -(1/2) ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l2719_271928


namespace NUMINAMATH_CALUDE_ratio_equality_l2719_271984

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2719_271984


namespace NUMINAMATH_CALUDE_log_simplification_l2719_271926

-- Define variables
variable (p q r s t z : ℝ)
variable (h₁ : p > 0)
variable (h₂ : q > 0)
variable (h₃ : r > 0)
variable (h₄ : s > 0)
variable (h₅ : t > 0)
variable (h₆ : z > 0)

-- State the theorem
theorem log_simplification :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l2719_271926


namespace NUMINAMATH_CALUDE_product_of_binary_numbers_l2719_271920

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem product_of_binary_numbers :
  let a := [true, true, false, false, true, true]
  let b := [true, true, false, true]
  let result := [true, false, false, true, true, false, false, false, true, false, true]
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_numbers_l2719_271920


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l2719_271911

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l2719_271911


namespace NUMINAMATH_CALUDE_parabola_vertex_l2719_271949

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex coordinates of the parabola y = 2(x-3)^2 + 1 are (3, 1) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2719_271949


namespace NUMINAMATH_CALUDE_quadratic_roots_coefficients_l2719_271936

theorem quadratic_roots_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b = -3 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_coefficients_l2719_271936


namespace NUMINAMATH_CALUDE_one_right_angled_triangle_l2719_271982

/-- A triangle with side lengths 15, 20, and x has exactly one right angle -/
def has_one_right_angle (x : ℤ) : Prop :=
  (x ^ 2 = 15 ^ 2 + 20 ^ 2) ∨ 
  (15 ^ 2 = x ^ 2 + 20 ^ 2) ∨ 
  (20 ^ 2 = 15 ^ 2 + x ^ 2)

/-- The triangle inequality is satisfied -/
def satisfies_triangle_inequality (x : ℤ) : Prop :=
  x > 0 ∧ 15 + 20 > x ∧ 15 + x > 20 ∧ 20 + x > 15

/-- There exists exactly one integer x that satisfies the conditions -/
theorem one_right_angled_triangle : 
  ∃! x : ℤ, satisfies_triangle_inequality x ∧ has_one_right_angle x :=
sorry

end NUMINAMATH_CALUDE_one_right_angled_triangle_l2719_271982


namespace NUMINAMATH_CALUDE_work_earnings_equation_l2719_271955

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 2) * (t + 3) + 3 → t = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l2719_271955


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2719_271963

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a * Real.cos t.C = 2 * t.c * Real.cos t.A ∧ 
  Real.tan t.C = 1/2 ∧
  t.b = 5

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = 3 * Real.pi / 4 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 5/2) := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l2719_271963


namespace NUMINAMATH_CALUDE_triangle_forming_sets_l2719_271947

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of numbers we're checking --/
def sets : List (ℝ × ℝ × ℝ) := [
  (1, 2, 3),
  (2, 3, 4),
  (3, 4, 5),
  (3, 6, 9)
]

/-- The theorem stating which sets can form triangles --/
theorem triangle_forming_sets :
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → can_form_triangle a b c) ↔
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (2, 3, 4)) ∧
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (3, 4, 5)) ∧
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → (a, b, c) ≠ (2, 3, 4) → (a, b, c) ≠ (3, 4, 5) → ¬can_form_triangle a b c) :=
sorry

end NUMINAMATH_CALUDE_triangle_forming_sets_l2719_271947


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l2719_271978

/-- Number of valid sequences without two consecutive heads for n coin tosses -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => f (n + 1) + f n

/-- Probability of no two consecutive heads in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (f n : ℚ) / (2^n : ℚ)

/-- Theorem: The probability of no two heads appearing consecutively in 10 coin tosses is 9/64 -/
theorem prob_no_consecutive_heads_10 : prob_no_consecutive_heads 10 = 9/64 := by
  sorry

#eval prob_no_consecutive_heads 10

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l2719_271978


namespace NUMINAMATH_CALUDE_min_production_avoids_loss_less_than_min_production_incurs_loss_l2719_271914

/-- The minimum production quantity to avoid a loss -/
def min_production : ℝ := 150

/-- The unit selling price in million yuan -/
def unit_price : ℝ := 0.25

/-- The total cost function in million yuan for x units -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The total revenue function in million yuan for x units -/
def total_revenue (x : ℝ) : ℝ := unit_price * x

/-- Theorem stating that the minimum production quantity to avoid a loss is 150 units -/
theorem min_production_avoids_loss :
  ∀ x : ℝ, x ≥ min_production → total_revenue x ≥ total_cost x :=
by
  sorry

/-- Theorem stating that any production quantity less than 150 units results in a loss -/
theorem less_than_min_production_incurs_loss :
  ∀ x : ℝ, 0 ≤ x ∧ x < min_production → total_revenue x < total_cost x :=
by
  sorry

end NUMINAMATH_CALUDE_min_production_avoids_loss_less_than_min_production_incurs_loss_l2719_271914


namespace NUMINAMATH_CALUDE_rectangle_length_l2719_271940

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 9 →
  rect_width = 3 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 27 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l2719_271940


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l2719_271964

theorem systematic_sampling_probability
  (total_parts : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (sample_size : Nat)
  (h1 : total_parts = 120)
  (h2 : first_grade = 24)
  (h3 : second_grade = 36)
  (h4 : third_grade = 60)
  (h5 : sample_size = 20)
  (h6 : total_parts = first_grade + second_grade + third_grade) :
  (sample_size : ℚ) / (total_parts : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l2719_271964


namespace NUMINAMATH_CALUDE_expression_evaluation_l2719_271924

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x + y) * (2*x - y) - 3*(2*x^2 - x*y) + y^2 = -14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2719_271924


namespace NUMINAMATH_CALUDE_square_fraction_below_line_l2719_271967

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line passing through two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Represents a square defined by its corners -/
structure Square :=
  (bottomLeft : Point)
  (topRight : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  sorry

/-- Finds the intersection of a line with the right edge of a square -/
def rightEdgeIntersection (l : Line) (s : Square) : Point :=
  sorry

/-- Main theorem: The fraction of the square's area below the line is 1/18 -/
theorem square_fraction_below_line :
  let s := Square.mk (Point.mk 2 0) (Point.mk 5 3)
  let l := Line.mk (Point.mk 2 3) (Point.mk 5 1)
  let intersection := rightEdgeIntersection l s
  let belowArea := triangleArea (Point.mk 2 0) (Point.mk 5 0) intersection
  let totalArea := squareArea s
  belowArea / totalArea = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_below_line_l2719_271967


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_equals_3_l2719_271912

theorem sqrt_81_div_3_equals_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_equals_3_l2719_271912


namespace NUMINAMATH_CALUDE_discount_profit_theorem_l2719_271939

theorem discount_profit_theorem (cost : ℝ) (h_cost_pos : cost > 0) : 
  let discount_rate : ℝ := 0.1
  let profit_rate_with_discount : ℝ := 0.2
  let selling_price_with_discount : ℝ := (1 - discount_rate) * ((1 + profit_rate_with_discount) * cost)
  let selling_price_without_discount : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount : ℝ := selling_price_without_discount - cost
  let profit_rate_without_discount : ℝ := profit_without_discount / cost
  profit_rate_without_discount = 1/3 := by sorry

end NUMINAMATH_CALUDE_discount_profit_theorem_l2719_271939


namespace NUMINAMATH_CALUDE_line_contains_point_l2719_271953

/-- Given a line represented by the equation -3/4 - 3kx = 7y that contains the point (1/3, -8),
    prove that k = 55.25 -/
theorem line_contains_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3 : ℝ) = 7 * (-8 : ℝ) → k = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l2719_271953


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l2719_271904

theorem vector_angle_cosine (α β : Real) (a b : ℝ × ℝ) :
  -π/2 < α ∧ α < 0 ∧ 0 < β ∧ β < π/2 →
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  ‖a - b‖ = Real.sqrt 10 / 5 →
  Real.cos α = 12/13 →
  Real.cos (α - β) = 4/5 ∧ Real.cos β = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l2719_271904


namespace NUMINAMATH_CALUDE_katie_candy_count_l2719_271944

theorem katie_candy_count (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : sister_candy = 23)
  (h2 : eaten_candy = 8)
  (h3 : remaining_candy = 23) :
  ∃ (katie_candy : ℕ), katie_candy = 8 ∧ katie_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_katie_candy_count_l2719_271944


namespace NUMINAMATH_CALUDE_triangle_side_length_l2719_271994

theorem triangle_side_length (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + c = 2 * b →          -- Given condition
  a * c = 6 →              -- Given condition
  Real.cos (60 * π / 180) = (a^2 + c^2 - b^2) / (2 * a * c) →  -- Cosine theorem for 60°
  b = Real.sqrt 6 := by
sorry

-- Note: We use Real.cos and Real.sqrt to represent cosine and square root functions

end NUMINAMATH_CALUDE_triangle_side_length_l2719_271994


namespace NUMINAMATH_CALUDE_trouser_original_price_l2719_271948

/-- 
Given a trouser with a sale price of $70 after a 30% decrease,
prove that its original price was $100.
-/
theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 70 → 
  discount_percentage = 30 → 
  sale_price = (1 - discount_percentage / 100) * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l2719_271948


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2719_271956

/-- Definition of an H function -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: A function is an H function if and only if it is strictly increasing -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2719_271956


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2719_271937

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 40 →
  arithmetic_sequence a₁ d 22 = 44 →
  arithmetic_sequence a₁ d 5 = -24 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2719_271937


namespace NUMINAMATH_CALUDE_smallest_n_for_coprime_subset_l2719_271997

def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 100}

theorem smallest_n_for_coprime_subset : 
  ∃ (n : Nat), n = 75 ∧ 
  (∀ (A : Set Nat), A ⊆ S → A.Finite → A.ncard ≥ n → 
    ∃ (a b c d : Nat), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
    Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d) ∧
  (∀ (m : Nat), m < 75 → 
    ∃ (B : Set Nat), B ⊆ S ∧ B.Finite ∧ B.ncard = m ∧
    ¬(∃ (a b c d : Nat), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ d ∈ B ∧ 
      Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
      Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_coprime_subset_l2719_271997


namespace NUMINAMATH_CALUDE_number_of_nests_l2719_271988

theorem number_of_nests (birds : ℕ) (nests : ℕ) : 
  birds = 6 → birds = nests + 3 → nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nests_l2719_271988


namespace NUMINAMATH_CALUDE_christine_wandering_time_l2719_271903

/-- Given a distance of 20 miles and a speed of 4 miles per hour, 
    the time taken is 5 hours. -/
theorem christine_wandering_time :
  let distance : ℝ := 20
  let speed : ℝ := 4
  let time := distance / speed
  time = 5 := by sorry

end NUMINAMATH_CALUDE_christine_wandering_time_l2719_271903


namespace NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l2719_271961

theorem residue_of_negative_1235_mod_29 : Int.mod (-1235) 29 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l2719_271961


namespace NUMINAMATH_CALUDE_jill_and_emily_total_l2719_271900

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  maria : ℕ
  emily : ℕ

/-- The conditions of the peach distribution problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 14 ∧
  p.jake = p.steven - 6 ∧
  p.jake = p.jill + 3 ∧
  p.maria = 2 * p.jake ∧
  p.emily = p.maria - 9

/-- The theorem stating that Jill and Emily have 12 peaches in total -/
theorem jill_and_emily_total (p : Peaches) (h : peach_conditions p) : 
  p.jill + p.emily = 12 := by
  sorry

end NUMINAMATH_CALUDE_jill_and_emily_total_l2719_271900


namespace NUMINAMATH_CALUDE_min_value_of_f_l2719_271945

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 4*x*y + 5*y^2 - 8*x + 6*y + 2

/-- Theorem stating that the minimum value of f is -7 -/
theorem min_value_of_f :
  ∃ (x y : ℝ), ∀ (a b : ℝ), f x y ≤ f a b ∧ f x y = -7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2719_271945


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l2719_271930

/-- The radius of the quarter circle -/
def R : ℝ := 12

/-- The radius of the largest inscribed circle -/
def r : ℝ := 3

/-- Theorem stating that r is the radius of the largest inscribed circle -/
theorem largest_inscribed_circle_radius : 
  (R - r)^2 - r^2 = (R/2 + r)^2 - (R/2 - r)^2 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l2719_271930


namespace NUMINAMATH_CALUDE_gdp_growth_problem_l2719_271917

/-- Calculates the GDP after compound growth -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ years

/-- The GDP growth problem -/
theorem gdp_growth_problem :
  let initial_gdp : ℝ := 9593.3
  let growth_rate : ℝ := 0.073
  let years : ℕ := 4
  let final_gdp : ℝ := gdp_growth initial_gdp growth_rate years
  ∃ ε > 0, |final_gdp - 127254| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_gdp_growth_problem_l2719_271917


namespace NUMINAMATH_CALUDE_division_result_l2719_271905

theorem division_result : 
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 := by sorry

end NUMINAMATH_CALUDE_division_result_l2719_271905


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l2719_271958

-- Define the tetrahedron with four positive integers on its faces
def Tetrahedron (a b c d : ℕ+) : Prop :=
  -- The sum of the products of each combination of three numbers is 770
  a.val * b.val * c.val + a.val * b.val * d.val + a.val * c.val * d.val + b.val * c.val * d.val = 770

-- Theorem statement
theorem tetrahedron_sum (a b c d : ℕ+) (h : Tetrahedron a b c d) :
  a.val + b.val + c.val + d.val = 57 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l2719_271958


namespace NUMINAMATH_CALUDE_theater_revenue_l2719_271979

theorem theater_revenue (total_seats : ℕ) (adult_price child_price : ℕ) (child_tickets : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_price = 5 →
  child_tickets = 63 →
  (total_seats = child_tickets + (total_seats - child_tickets)) →
  child_tickets * child_price + (total_seats - child_tickets) * adult_price = 519 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_l2719_271979


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2719_271983

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) 
  (m n : Fin 2 → Real) :
  m 0 = Real.cos A ∧ m 1 = Real.sin A ∧
  n 0 = Real.sqrt 2 - Real.sin A ∧ n 1 = Real.cos A ∧
  (m 0 * n 0 + m 1 * n 1 = 1) ∧
  b = 4 * Real.sqrt 2 ∧
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ 
  (1/2 : Real) * b * c * Real.sin A = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2719_271983


namespace NUMINAMATH_CALUDE_sum_sequence_square_l2719_271918

theorem sum_sequence_square (n : ℕ) : 
  (List.range n).sum + n + (List.range n).reverse.sum = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sequence_square_l2719_271918


namespace NUMINAMATH_CALUDE_function_property_l2719_271922

def is_valid_function (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_property (f : ℕ+ → ℝ) (h : is_valid_function f) (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2719_271922


namespace NUMINAMATH_CALUDE_regression_change_l2719_271919

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in y when x increases by one unit -/
def change_in_y (regression : LinearRegression) : ℝ := -regression.b

/-- Theorem: For the given regression equation ŷ = 2 - 1.5x, 
    when x increases by one unit, y decreases by 1.5 units -/
theorem regression_change (regression : LinearRegression) 
  (h1 : regression.a = 2) 
  (h2 : regression.b = -1.5) : 
  change_in_y regression = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_change_l2719_271919


namespace NUMINAMATH_CALUDE_max_excellent_courses_l2719_271906

/-- A course video with two attributes: number of views and expert score -/
structure CourseVideo where
  views : ℕ
  expertScore : ℕ

/-- Defines when one course video is not inferior to another -/
def notInferior (a b : CourseVideo) : Prop :=
  a.views ≥ b.views ∨ a.expertScore ≥ b.expertScore

/-- Defines an excellent course video -/
def isExcellent (a : CourseVideo) (courses : Finset CourseVideo) : Prop :=
  ∀ b ∈ courses, b ≠ a → notInferior a b

/-- Theorem: It's possible to have 5 excellent course videos among 5 courses -/
theorem max_excellent_courses (courses : Finset CourseVideo) (h : courses.card = 5) :
  ∃ excellentCourses : Finset CourseVideo,
    excellentCourses ⊆ courses ∧
    excellentCourses.card = 5 ∧
    ∀ a ∈ excellentCourses, isExcellent a courses := by
  sorry

end NUMINAMATH_CALUDE_max_excellent_courses_l2719_271906


namespace NUMINAMATH_CALUDE_united_additional_charge_is_correct_l2719_271935

/-- The additional charge per minute for United Telephone -/
def united_additional_charge : ℚ := 1/4

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 6

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Atlantic Call's additional charge per minute -/
def atlantic_additional_charge : ℚ := 1/5

/-- The number of minutes at which both companies' bills are equal -/
def equal_minutes : ℕ := 120

theorem united_additional_charge_is_correct : 
  united_base_rate + equal_minutes * united_additional_charge = 
  atlantic_base_rate + equal_minutes * atlantic_additional_charge :=
by sorry

end NUMINAMATH_CALUDE_united_additional_charge_is_correct_l2719_271935


namespace NUMINAMATH_CALUDE_brothers_to_madelines_money_ratio_l2719_271991

theorem brothers_to_madelines_money_ratio (madelines_money : ℕ) (total_money : ℕ) : 
  madelines_money = 48 →
  total_money = 72 →
  (total_money - madelines_money) * 2 = madelines_money :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_to_madelines_money_ratio_l2719_271991


namespace NUMINAMATH_CALUDE_cube_sphere_surface_area_ratio_l2719_271990

-- Define a cube with an inscribed sphere
structure CubeWithInscribedSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  h_diameter : sphere_radius * 2 = edge_length

-- Theorem statement
theorem cube_sphere_surface_area_ratio 
  (c : CubeWithInscribedSphere) : 
  (6 * c.edge_length^2) / (4 * Real.pi * c.sphere_radius^2) = 6 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_surface_area_ratio_l2719_271990


namespace NUMINAMATH_CALUDE_chord_length_l2719_271902

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ - ρ * Real.cos (2 * θ) - 12 * Real.cos θ = 0

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = -4/5 * t + 2 ∧ y = 3/5 * t

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop := y^2 = 6 * x

-- Define the line l in normal form
def line_l_normal (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

-- Theorem statement
theorem chord_length :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  curve_C_rect x₁ y₁ ∧ curve_C_rect x₂ y₂ ∧
  line_l_normal x₁ y₁ ∧ line_l_normal x₂ y₂ ∧
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (20 * Real.sqrt 7) / 3 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2719_271902


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2719_271933

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 6) 
  (h2 : t.base2 = 12) 
  (h3 : t.area = 36) : 
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2719_271933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2719_271950

/-- An arithmetic sequence with given first term and 17th term -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 17 = 66 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : 
  (∀ n : ℕ, a n = 4 * n - 2) ∧ 
  (¬ ∃ n : ℕ, a n = 88) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2719_271950


namespace NUMINAMATH_CALUDE_sarah_interview_count_l2719_271960

theorem sarah_interview_count (oranges pears apples strawberries : ℕ) 
  (h_oranges : oranges = 70)
  (h_pears : pears = 120)
  (h_apples : apples = 147)
  (h_strawberries : strawberries = 113) :
  oranges + pears + apples + strawberries = 450 := by
  sorry

end NUMINAMATH_CALUDE_sarah_interview_count_l2719_271960


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2719_271985

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 6)
  (h3 : lending_rate = 0.09)
  (h4 : gain_per_year = 200) :
  let interest_received := principal * lending_rate * time
  let total_gain := gain_per_year * time
  let interest_paid := interest_received - total_gain
  let borrowing_rate := interest_paid / (principal * time)
  borrowing_rate = 0.08 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2719_271985


namespace NUMINAMATH_CALUDE_min_value_expression_l2719_271980

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2719_271980


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2719_271999

theorem modulus_of_complex : Complex.abs (7/4 - 3*I) = (Real.sqrt 193)/4 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2719_271999


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l2719_271946

/-- Given angles α, β, γ of a triangle, the determinant of the matrix
    | tan α   sin α cos α   1 |
    | tan β   sin β cos β   1 |
    | tan γ   sin γ cos γ   1 |
    is equal to 0. -/
theorem triangle_angle_determinant (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l2719_271946


namespace NUMINAMATH_CALUDE_farm_animals_problem_l2719_271929

/-- Represents the farm animals problem --/
theorem farm_animals_problem (cows ducks pigs : ℕ) : 
  cows = 20 →
  ducks = (3 : ℕ) * cows / 2 →
  cows + ducks + pigs = 60 →
  pigs = (cows + ducks) / 5 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_problem_l2719_271929


namespace NUMINAMATH_CALUDE_west_movement_notation_l2719_271962

/-- Represents the direction of movement -/
inductive Direction
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

theorem west_movement_notation :
  let eastMovement : Movement := ⟨5, Direction.East⟩
  let westMovement : Movement := ⟨3, Direction.West⟩
  toNumber eastMovement = 5 →
  toNumber westMovement = -3 := by
  sorry

end NUMINAMATH_CALUDE_west_movement_notation_l2719_271962


namespace NUMINAMATH_CALUDE_min_negations_for_zero_sum_l2719_271957

def clock_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def sum_list (l : List ℤ) : ℤ := l.foldl (· + ·) 0

def negate_elements (l : List ℕ) (indices : List ℕ) : List ℤ :=
  l.enum.map (fun (i, x) => if i + 1 ∈ indices then -x else x)

theorem min_negations_for_zero_sum :
  ∃ (indices : List ℕ),
    (indices.length = 4) ∧
    (sum_list (negate_elements clock_numbers indices) = 0) ∧
    (∀ (other_indices : List ℕ),
      sum_list (negate_elements clock_numbers other_indices) = 0 →
      other_indices.length ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_negations_for_zero_sum_l2719_271957


namespace NUMINAMATH_CALUDE_quadratic_sum_l2719_271995

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 40 * x + 160 = a * (x + b)^2 + c) ∧ (a + b + c = 120.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2719_271995
