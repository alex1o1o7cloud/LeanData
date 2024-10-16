import Mathlib

namespace NUMINAMATH_CALUDE_roses_in_vase_l3653_365372

/-- The number of roses in the vase initially -/
def initial_roses : ℕ := 9

/-- The number of orchids in the vase initially -/
def initial_orchids : ℕ := 6

/-- The number of orchids in the vase now -/
def current_orchids : ℕ := 13

/-- The difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 10

/-- The number of roses in the vase now -/
def current_roses : ℕ := 3

theorem roses_in_vase :
  current_roses = current_orchids - orchid_rose_difference := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3653_365372


namespace NUMINAMATH_CALUDE_independent_of_b_implies_k_equals_two_l3653_365367

/-- If the algebraic expression ab(5ka-3b)-(ka-b)(3ab-4a²) is independent of b, then k = 2 -/
theorem independent_of_b_implies_k_equals_two (a b k : ℝ) :
  (∀ b, ∃ C, a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2) = C) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_independent_of_b_implies_k_equals_two_l3653_365367


namespace NUMINAMATH_CALUDE_shaded_probability_l3653_365320

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  total_count : Nat

/-- The probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded_count / d.total_count

theorem shaded_probability (d : Diagram) 
  (h1 : d.total_count > 4)
  (h2 : d.shaded_count = d.total_count / 2)
  (h3 : d.shaded_count = (d.triangles.filter Triangle.shaded).length)
  (h4 : d.total_count = d.triangles.length) :
  probability_shaded d = 1 / 2 := by
  sorry

#check shaded_probability

end NUMINAMATH_CALUDE_shaded_probability_l3653_365320


namespace NUMINAMATH_CALUDE_triangle_side_difference_l3653_365329

theorem triangle_side_difference (x : ℕ) : 
  (x > 0) →
  (x + 10 > 8) →
  (x + 8 > 10) →
  (10 + 8 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y > 0 ∧ y + 10 > 8 ∧ y + 8 > 10 ∧ 10 + 8 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l3653_365329


namespace NUMINAMATH_CALUDE_min_value_sum_l3653_365335

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = (1 : ℝ) / 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = (1 : ℝ) / 4 → 
  a + 3 * b ≤ x + 3 * y ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3653_365335


namespace NUMINAMATH_CALUDE_evaluate_expression_l3653_365398

theorem evaluate_expression : 5^4 + 5^4 + 5^4 - 5^4 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3653_365398


namespace NUMINAMATH_CALUDE_fraction_integrality_l3653_365324

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b : ℚ) / c + (a * c : ℚ) / b + (b * c : ℚ) / a = n) :
  (∃ (n1 : ℤ), (a * b : ℚ) / c = n1) ∧ 
  (∃ (n2 : ℤ), (a * c : ℚ) / b = n2) ∧ 
  (∃ (n3 : ℤ), (b * c : ℚ) / a = n3) := by
sorry

end NUMINAMATH_CALUDE_fraction_integrality_l3653_365324


namespace NUMINAMATH_CALUDE_congruence_product_l3653_365378

theorem congruence_product (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a * c) ≡ (b * d) [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_product_l3653_365378


namespace NUMINAMATH_CALUDE_solution_set_for_neg_eight_solution_range_for_a_l3653_365359

-- Define the inequality function
def inequality (x a : ℝ) : Prop :=
  |x - 3| + |x + 2| ≤ |a + 1|

-- Theorem 1: Solution set when a = -8
theorem solution_set_for_neg_eight :
  Set.Icc (-3 : ℝ) 4 = {x : ℝ | inequality x (-8)} :=
sorry

-- Theorem 2: Range of a for which the inequality has solutions
theorem solution_range_for_a :
  {a : ℝ | ∃ x, inequality x a} = Set.Iic (-6) ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_neg_eight_solution_range_for_a_l3653_365359


namespace NUMINAMATH_CALUDE_number_puzzle_l3653_365379

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ x / 5 + 7 = x / 4 - 7 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3653_365379


namespace NUMINAMATH_CALUDE_pencils_lost_l3653_365360

theorem pencils_lost (initial_pencils : ℕ) (current_pencils : ℕ) (lost_pencils : ℕ) :
  initial_pencils = 30 →
  current_pencils = 16 →
  current_pencils = initial_pencils - lost_pencils - (initial_pencils - lost_pencils) / 3 →
  lost_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_lost_l3653_365360


namespace NUMINAMATH_CALUDE_perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l3653_365338

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define non-coincidence
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perp_to_same_plane_implies_parallel 
  (a b : Line) (α : Plane) 
  (h1 : non_coincident_lines a b) 
  (h2 : perp a α) (h3 : perp b α) : 
  parallel a b := by sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perp_to_two_planes_implies_parallel 
  (a : Line) (α β : Plane) 
  (h1 : non_coincident_planes α β) 
  (h2 : perp a α) (h3 : perp a β) : 
  plane_parallel α β := by sorry

end NUMINAMATH_CALUDE_perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l3653_365338


namespace NUMINAMATH_CALUDE_pie_apples_ratio_l3653_365326

def total_apples : ℕ := 62
def refrigerator_apples : ℕ := 25
def muffin_apples : ℕ := 6

def pie_apples : ℕ := total_apples - refrigerator_apples - muffin_apples

theorem pie_apples_ratio :
  (pie_apples : ℚ) / total_apples = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_pie_apples_ratio_l3653_365326


namespace NUMINAMATH_CALUDE_equal_sum_blocks_iff_three_l3653_365365

/-- A function that checks if a prime number p allows the sequence of natural numbers
    from 1 to p to be divided into several consecutive blocks with identical sums -/
def has_equal_sum_blocks (p : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), 
    Prime p ∧
    k > 1 ∧
    m < p ∧
    (m * (m + 1)) / 2 = k * p ∧
    p * (p + 1) / 2 = k * p

/-- Theorem stating that the only prime number p that satisfies the condition is 3 -/
theorem equal_sum_blocks_iff_three :
  ∀ p : ℕ, Prime p → (has_equal_sum_blocks p ↔ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_blocks_iff_three_l3653_365365


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3653_365309

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → 
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3653_365309


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l3653_365368

theorem largest_positive_root_bound (a₂ a₁ a₀ : ℝ) (h1 : |a₂| ≤ 3) (h2 : |a₁| ≤ 3) (h3 : |a₀| ≤ 3) (h4 : a₂ + a₁ + a₀ = -6) :
  ∃ r : ℝ, 2 < r ∧ r < 3 ∧ r^3 + a₂*r^2 + a₁*r + a₀ = 0 ∧
  ∀ x : ℝ, x > 0 ∧ x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l3653_365368


namespace NUMINAMATH_CALUDE_no_xyz_solution_l3653_365301

theorem no_xyz_solution : ¬∃ (x y z : ℕ), 
  0 ≤ x ∧ x ≤ 9 ∧
  0 ≤ y ∧ y ≤ 9 ∧
  0 ≤ z ∧ z ≤ 9 ∧
  100 * x + 10 * y + z = y * (10 * x + z) := by
  sorry

end NUMINAMATH_CALUDE_no_xyz_solution_l3653_365301


namespace NUMINAMATH_CALUDE_teamA_win_probability_l3653_365382

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  numTeams : Nat
  noTies : Bool
  equalWinChance : Bool
  teamAWonFirst : Bool

/-- Calculates the probability that Team A finishes with more points than Team B -/
def probabilityTeamAWins (tournament : SoccerTournament) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem teamA_win_probability 
  (tournament : SoccerTournament) 
  (h1 : tournament.numTeams = 9)
  (h2 : tournament.noTies = true)
  (h3 : tournament.equalWinChance = true)
  (h4 : tournament.teamAWonFirst = true) :
  probabilityTeamAWins tournament = 9714 / 8192 :=
sorry

end NUMINAMATH_CALUDE_teamA_win_probability_l3653_365382


namespace NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l3653_365357

theorem lcm_ratio_implies_gcd (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  Nat.gcd A B = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l3653_365357


namespace NUMINAMATH_CALUDE_min_fence_posts_is_22_l3653_365328

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := (length / post_spacing) + 1
  let short_side_posts := (width / post_spacing) + 1
  (2 * long_side_posts) + short_side_posts - 2

/-- The minimum number of fence posts for the given dimensions is 22 -/
theorem min_fence_posts_is_22 :
  fence_posts 80 50 10 = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_posts_is_22_l3653_365328


namespace NUMINAMATH_CALUDE_garrison_size_l3653_365349

-- Define the parameters
def initial_days : ℕ := 31
def days_passed : ℕ := 27
def people_left : ℕ := 200
def remaining_days : ℕ := 8

-- Theorem statement
theorem garrison_size :
  ∀ (M : ℕ),
  (M * (initial_days - days_passed) = (M - people_left) * remaining_days) →
  M = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_garrison_size_l3653_365349


namespace NUMINAMATH_CALUDE_minAreaLine_minProductLine_l3653_365332

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the type for lines in 2D space
def Line := ℝ → ℝ → Prop

-- Function to calculate the area of triangle AOB given a line
def areaAOB (l : Line) : ℝ := sorry

-- Function to calculate the product of |PA| and |PB| given a line
def productPAPB (l : Line) : ℝ := sorry

-- Predicate to check if a line passes through point P
def passesThroughP (l : Line) : Prop :=
  l P.1 P.2

-- Predicate to check if a line intersects positive x and y axes
def intersectsPositiveAxes (l : Line) : Prop := sorry

-- Theorem for part 1
theorem minAreaLine :
  ∃ (l : Line),
    passesThroughP l ∧
    intersectsPositiveAxes l ∧
    (∀ (l' : Line), passesThroughP l' → intersectsPositiveAxes l' → areaAOB l ≤ areaAOB l') ∧
    (∀ x y, l x y ↔ x + 2*y - 4 = 0) :=
  sorry

-- Theorem for part 2
theorem minProductLine :
  ∃ (l : Line),
    passesThroughP l ∧
    intersectsPositiveAxes l ∧
    (∀ (l' : Line), passesThroughP l' → intersectsPositiveAxes l' → productPAPB l ≤ productPAPB l') ∧
    (∀ x y, l x y ↔ x + y - 3 = 0) :=
  sorry

end NUMINAMATH_CALUDE_minAreaLine_minProductLine_l3653_365332


namespace NUMINAMATH_CALUDE_distance_between_five_and_six_l3653_365347

/-- The distance to the nearest town in miles -/
def d : ℝ := sorry

/-- Alice's statement is false -/
axiom alice_false : ¬(d ≥ 6)

/-- Bob's statement is false -/
axiom bob_false : ¬(d ≤ 5)

/-- Charlie's statement is false -/
axiom charlie_false : ¬(d ≤ 4)

/-- Theorem: The distance to the nearest town is between 5 and 6 miles -/
theorem distance_between_five_and_six : 5 < d ∧ d < 6 := by sorry

end NUMINAMATH_CALUDE_distance_between_five_and_six_l3653_365347


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3653_365351

theorem train_speed_calculation (rail_length : Real) (time_period : Real) : 
  rail_length = 40 ∧ time_period = 30 / 60 →
  ∃ (ε : Real), ε > 0 ∧ ∀ (train_speed : Real),
    train_speed > 0 →
    |train_speed - (train_speed * 5280 / 60 / rail_length * time_period)| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3653_365351


namespace NUMINAMATH_CALUDE_sqrt_7_to_6th_power_l3653_365377

theorem sqrt_7_to_6th_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7_to_6th_power_l3653_365377


namespace NUMINAMATH_CALUDE_second_solution_concentration_l3653_365342

/-- Represents an alcohol solution --/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions --/
structure AlcoholMixture where
  solution1 : AlcoholSolution
  solution2 : AlcoholSolution
  final : AlcoholSolution

/-- The alcohol mixture satisfies the given conditions --/
def satisfies_conditions (mixture : AlcoholMixture) : Prop :=
  mixture.final.volume = 200 ∧
  mixture.final.concentration = 0.15 ∧
  mixture.solution1.volume = 75 ∧
  mixture.solution1.concentration = 0.20 ∧
  mixture.solution2.volume = mixture.final.volume - mixture.solution1.volume

theorem second_solution_concentration
  (mixture : AlcoholMixture)
  (h : satisfies_conditions mixture) :
  mixture.solution2.concentration = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_second_solution_concentration_l3653_365342


namespace NUMINAMATH_CALUDE_vowel_count_l3653_365397

theorem vowel_count (num_vowels : ℕ) (total_written : ℕ) : 
  num_vowels = 5 → total_written = 15 → (total_written / num_vowels : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_vowel_count_l3653_365397


namespace NUMINAMATH_CALUDE_sentence_B_is_correct_l3653_365384

/-- Represents a sentence in English --/
structure Sentence where
  text : String

/-- Checks if a sentence is grammatically correct --/
def is_grammatically_correct (s : Sentence) : Prop := sorry

/-- The four sentences given in the problem --/
def sentence_A : Sentence := { text := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams." }

def sentence_B : Sentence := { text := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." }

def sentence_C : Sentence := { text := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high." }

def sentence_D : Sentence := { text := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves." }

/-- Theorem stating that sentence B is grammatically correct --/
theorem sentence_B_is_correct :
  is_grammatically_correct sentence_B ∧
  ¬is_grammatically_correct sentence_A ∧
  ¬is_grammatically_correct sentence_C ∧
  ¬is_grammatically_correct sentence_D :=
by
  sorry

end NUMINAMATH_CALUDE_sentence_B_is_correct_l3653_365384


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3653_365374

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) (h : subset n α) :
  (perp_line m n → perp_plane m α) ∧ 
  ¬(perp_line m n ↔ perp_plane m α) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3653_365374


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l3653_365303

theorem initial_children_on_bus (children_got_on : ℕ) (total_children : ℕ) 
  (h1 : children_got_on = 38)
  (h2 : total_children = 64) :
  total_children - children_got_on = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l3653_365303


namespace NUMINAMATH_CALUDE_collection_distribution_l3653_365306

def karl_stickers : ℕ := 25
def karl_cards : ℕ := 15
def karl_keychains : ℕ := 5
def karl_stamps : ℕ := 10

def ryan_stickers : ℕ := karl_stickers + 20
def ryan_cards : ℕ := karl_cards - 10
def ryan_keychains : ℕ := karl_keychains + 2
def ryan_stamps : ℕ := karl_stamps

def ben_stickers : ℕ := ryan_stickers - 10
def ben_cards : ℕ := ryan_cards / 2
def ben_keychains : ℕ := karl_keychains * 2
def ben_stamps : ℕ := karl_stamps + 5

def total_items : ℕ := karl_stickers + karl_cards + karl_keychains + karl_stamps +
                       ryan_stickers + ryan_cards + ryan_keychains + ryan_stamps +
                       ben_stickers + ben_cards + ben_keychains + ben_stamps

def num_collectors : ℕ := 4

theorem collection_distribution :
  total_items = 184 ∧ total_items % num_collectors = 0 ∧ total_items / num_collectors = 46 := by
  sorry

end NUMINAMATH_CALUDE_collection_distribution_l3653_365306


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3653_365380

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 2 < 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3653_365380


namespace NUMINAMATH_CALUDE_castle_tour_limit_l3653_365366

structure Castle where
  side_length : ℝ
  num_halls : ℕ
  hall_side_length : ℝ
  has_doors : Bool

def max_visitable_halls (c : Castle) : ℕ :=
  sorry

theorem castle_tour_limit (c : Castle) 
  (h1 : c.side_length = 100)
  (h2 : c.num_halls = 100)
  (h3 : c.hall_side_length = 10)
  (h4 : c.has_doors = true) :
  max_visitable_halls c ≤ 91 :=
sorry

end NUMINAMATH_CALUDE_castle_tour_limit_l3653_365366


namespace NUMINAMATH_CALUDE_bills_rats_l3653_365327

theorem bills_rats (total : ℕ) (ratio : ℕ) (h1 : total = 70) (h2 : ratio = 6) : 
  (ratio * total) / (ratio + 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_bills_rats_l3653_365327


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3653_365333

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) (h1 : total_students = 128) 
  (h2 : red_students = 70) (h3 : green_students = 58) (h4 : total_pairs = 64) 
  (h5 : red_red_pairs = 34) (h6 : total_students = red_students + green_students) :
  (green_students - (total_students - 2 * red_red_pairs)) / 2 = 28 := by
sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3653_365333


namespace NUMINAMATH_CALUDE_sixth_group_number_l3653_365336

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  total_groups : ℕ
  first_group_number : ℕ
  eighth_group_number : ℕ

/-- Theorem stating the number drawn in the sixth group. -/
theorem sixth_group_number (s : SystematicSampling)
  (h1 : s.total_students = 800)
  (h2 : s.total_groups = 50)
  (h3 : s.eighth_group_number = 9 * s.first_group_number)
  : (s.first_group_number + 5 * (s.total_students / s.total_groups)) = 94 := by
  sorry


end NUMINAMATH_CALUDE_sixth_group_number_l3653_365336


namespace NUMINAMATH_CALUDE_football_team_progress_l3653_365396

/-- 
Given a football team's yard changes, calculate their net progress.
-/
theorem football_team_progress 
  (loss : ℤ) 
  (gain : ℤ) 
  (h1 : loss = -5)
  (h2 : gain = 9) : 
  loss + gain = 4 := by
sorry

end NUMINAMATH_CALUDE_football_team_progress_l3653_365396


namespace NUMINAMATH_CALUDE_sqrt_of_sixteen_l3653_365331

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sixteen_l3653_365331


namespace NUMINAMATH_CALUDE_power_product_evaluation_l3653_365383

theorem power_product_evaluation : 
  let a : ℕ := 2
  a^3 * a^4 = 128 := by sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l3653_365383


namespace NUMINAMATH_CALUDE_integer_solution_exists_l3653_365362

theorem integer_solution_exists (n : ℤ) : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a = b + c ∨ b = a + c ∨ c = a + b) ∧
  a * n + b = c :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l3653_365362


namespace NUMINAMATH_CALUDE_equation_solutions_l3653_365350

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 + x^2*y + x*y^2 + y^3 = 8*(x^2 + x*y + y^2 + 1)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(8, -2), (-2, 8), (4 + Real.sqrt 15, 4 - Real.sqrt 15), (4 - Real.sqrt 15, 4 + Real.sqrt 15)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3653_365350


namespace NUMINAMATH_CALUDE_problem_solution_l3653_365345

theorem problem_solution :
  ∀ (x a b c : ℤ),
    x ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
    ((a * x^4) / b * c)^3 = x^3 →
    a + b + c = 9 →
    ((x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3653_365345


namespace NUMINAMATH_CALUDE_chord_distance_l3653_365390

/-- Given a circle intersected by three equally spaced parallel lines resulting in chords of lengths 38, 38, and 34, the distance between two adjacent parallel chords is 6. -/
theorem chord_distance (r : ℝ) (d : ℝ) : 
  d > 0 ∧ 
  r^2 = d^2 + 19^2 ∧ 
  r^2 = (3*d)^2 + 17^2 →
  2*d = 6 :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_l3653_365390


namespace NUMINAMATH_CALUDE_no_real_roots_l3653_365353

theorem no_real_roots : ∀ x : ℝ, x^2 - 4*x + 8 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3653_365353


namespace NUMINAMATH_CALUDE_orthocentre_constructible_l3653_365393

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A triangle in the plane -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Definition of a circumcircle of a triangle -/
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  sorry

/-- Definition of a circumcentre of a triangle -/
def isCircumcentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of an orthocentre of a triangle -/
def isOrthocentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of constructible using only a straightedge -/
def isStraightedgeConstructible (p : Point) (given : Set Point) : Prop :=
  sorry

/-- The main theorem -/
theorem orthocentre_constructible (t : Triangle) (c : Circle) (o : Point) :
  isCircumcircle c t → isCircumcentre o t →
  ∃ h : Point, isOrthocentre h t ∧ 
    isStraightedgeConstructible h {t.A, t.B, t.C, o} :=
  sorry

end NUMINAMATH_CALUDE_orthocentre_constructible_l3653_365393


namespace NUMINAMATH_CALUDE_alligator_count_l3653_365305

theorem alligator_count (crocodiles vipers total : ℕ) 
  (h1 : crocodiles = 22)
  (h2 : vipers = 5)
  (h3 : total = 50)
  (h4 : ∃ alligators : ℕ, crocodiles + alligators + vipers = total) :
  ∃ alligators : ℕ, alligators = 23 ∧ crocodiles + alligators + vipers = total :=
by sorry

end NUMINAMATH_CALUDE_alligator_count_l3653_365305


namespace NUMINAMATH_CALUDE_puzzle_solution_l3653_365341

theorem puzzle_solution (p q r s t : ℕ+) 
  (eq1 : p * q + p + q = 322)
  (eq2 : q * r + q + r = 186)
  (eq3 : r * s + r + s = 154)
  (eq4 : s * t + s + t = 272)
  (product : p * q * r * s * t = 3628800) : -- 3628800 is 10!
  p - t = 6 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3653_365341


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3653_365334

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) = Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3653_365334


namespace NUMINAMATH_CALUDE_jerry_added_two_figures_l3653_365330

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem: Jerry added 2 action figures to his shelf -/
theorem jerry_added_two_figures : action_figures_added 8 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_two_figures_l3653_365330


namespace NUMINAMATH_CALUDE_combined_tax_rate_calculation_l3653_365312

def john_tax_rate : ℚ := 30 / 100
def ingrid_tax_rate : ℚ := 40 / 100
def alice_tax_rate : ℚ := 25 / 100
def ben_tax_rate : ℚ := 35 / 100

def john_income : ℕ := 56000
def ingrid_income : ℕ := 74000
def alice_income : ℕ := 62000
def ben_income : ℕ := 80000

def total_tax : ℚ := john_tax_rate * john_income + ingrid_tax_rate * ingrid_income + 
                     alice_tax_rate * alice_income + ben_tax_rate * ben_income

def total_income : ℕ := john_income + ingrid_income + alice_income + ben_income

def combined_tax_rate : ℚ := total_tax / total_income

theorem combined_tax_rate_calculation : 
  combined_tax_rate = total_tax / total_income :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_calculation_l3653_365312


namespace NUMINAMATH_CALUDE_worker_travel_time_l3653_365388

/-- Proves that if a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, her usual travel time is 60 minutes. -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) : 
  normal_speed * normal_time = (5/6 * normal_speed) * (normal_time + 12) → 
  normal_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l3653_365388


namespace NUMINAMATH_CALUDE_sharp_composition_l3653_365371

def sharp (N : ℕ) : ℕ := 3 * N + 2

theorem sharp_composition : sharp (sharp (sharp 6)) = 188 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_l3653_365371


namespace NUMINAMATH_CALUDE_quadratic_m_bounds_l3653_365302

open Complex

/-- Given a quadratic equation x^2 + z₁x + z₂ + m = 0 with complex coefficients,
    prove that under certain conditions, |m| has specific min and max values. -/
theorem quadratic_m_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  abs (α - β) = 2 * Real.sqrt 7 →
  (abs m = Real.sqrt 41 - 7 ∨ abs m = Real.sqrt 41 + 7) ∧
  ∀ m' : ℂ, (∃ α' β' : ℂ, α'^2 + z₁*α' + z₂ + m' = 0 ∧
                          β'^2 + z₁*β' + z₂ + m' = 0 ∧
                          abs (α' - β') = 2 * Real.sqrt 7) →
    Real.sqrt 41 - 7 ≤ abs m' ∧ abs m' ≤ Real.sqrt 41 + 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_m_bounds_l3653_365302


namespace NUMINAMATH_CALUDE_tree_climbing_average_height_l3653_365325

theorem tree_climbing_average_height : 
  let first_tree_height : ℝ := 1000
  let second_tree_height : ℝ := first_tree_height / 2
  let third_tree_height : ℝ := first_tree_height / 2
  let fourth_tree_height : ℝ := first_tree_height + 200
  let total_height : ℝ := first_tree_height + second_tree_height + third_tree_height + fourth_tree_height
  let num_trees : ℝ := 4
  (total_height / num_trees) = 800 := by sorry

end NUMINAMATH_CALUDE_tree_climbing_average_height_l3653_365325


namespace NUMINAMATH_CALUDE_orange_beads_count_l3653_365321

/-- Represents the number of beads of each color in a necklace -/
structure NecklaceComposition where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
def TotalBeads : ℕ := 45

/-- The composition of beads in each necklace -/
def necklace : NecklaceComposition := {
  green := 9,
  white := 6,
  orange := 9  -- This is what we want to prove
}

/-- The maximum number of necklaces that can be made -/
def maxNecklaces : ℕ := 5

theorem orange_beads_count :
  necklace.orange = 9 ∧
  necklace.green * maxNecklaces = TotalBeads ∧
  necklace.white * maxNecklaces ≤ TotalBeads ∧
  necklace.orange * maxNecklaces = TotalBeads :=
by sorry

end NUMINAMATH_CALUDE_orange_beads_count_l3653_365321


namespace NUMINAMATH_CALUDE_triangle_square_equal_area_l3653_365308

/-- Given a square with perimeter 32 units and a right triangle with height 40 units,
    if the square and triangle have the same area, and the triangle's base is twice
    the length of x, then x = 8/5. -/
theorem triangle_square_equal_area (x : ℝ) : 
  let square_perimeter : ℝ := 32
  let square_side : ℝ := square_perimeter / 4
  let square_area : ℝ := square_side ^ 2
  let triangle_height : ℝ := 40
  let triangle_base : ℝ := 2 * x
  let triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
  square_area = triangle_area → x = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_equal_area_l3653_365308


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3653_365385

theorem inequality_solution_set :
  let S := {x : ℝ | (3*x + 1)*(1 - 2*x) > 0}
  S = {x : ℝ | -1/3 < x ∧ x < 1/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3653_365385


namespace NUMINAMATH_CALUDE_a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l3653_365399

/-- Represents a quadratic equation ax^2 + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x, a * x^2 + 2 * x + 1 = 0

/-- Predicate for an equation having at least one negative root -/
def has_negative_root {a : ℝ} (eq : QuadraticEquation a) : Prop :=
  ∃ x, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- Statement that 'a < 0' is a sufficient condition -/
theorem a_neg_sufficient {a : ℝ} (h : a < 0) : 
  ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Statement that 'a < 0' is not a necessary condition -/
theorem a_neg_not_necessary : 
  ∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Main theorem stating that 'a < 0' is sufficient but not necessary -/
theorem a_neg_sufficient_not_necessary : 
  (∀ a, a < 0 → ∃ (eq : QuadraticEquation a), has_negative_root eq) ∧
  (∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq) :=
sorry

end NUMINAMATH_CALUDE_a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l3653_365399


namespace NUMINAMATH_CALUDE_g_triple_equality_l3653_365315

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3 * x - 50

theorem g_triple_equality (a : ℝ) :
  a < 0 → (g (g (g 15)) = g (g (g a)) ↔ a = -55 / 3) := by
  sorry

end NUMINAMATH_CALUDE_g_triple_equality_l3653_365315


namespace NUMINAMATH_CALUDE_basketball_win_requirement_l3653_365348

theorem basketball_win_requirement (total_games : ℕ) (first_games : ℕ) (wins_so_far : ℕ) (remaining_games : ℕ) 
  (h1 : total_games = first_games + remaining_games)
  (h2 : total_games = 110)
  (h3 : first_games = 60)
  (h4 : wins_so_far = 48)
  (h5 : remaining_games = 50) :
  ∃ (additional_wins : ℕ), 
    (wins_so_far + additional_wins : ℚ) / total_games = 3/4 ∧ 
    additional_wins = 35 := by
  sorry

#check basketball_win_requirement

end NUMINAMATH_CALUDE_basketball_win_requirement_l3653_365348


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3653_365323

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3653_365323


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3653_365381

theorem complex_fraction_evaluation (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3653_365381


namespace NUMINAMATH_CALUDE_cube_surface_area_l3653_365316

/-- Given a cube with side length x and distance d between non-intersecting diagonals
    of adjacent lateral faces, prove that its total surface area is 18d^2. -/
theorem cube_surface_area (d : ℝ) (h : d > 0) :
  let x := d * Real.sqrt 3
  6 * x^2 = 18 * d^2 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3653_365316


namespace NUMINAMATH_CALUDE_martha_driving_distance_martha_driving_distance_proof_l3653_365395

theorem martha_driving_distance : ℝ → ℝ → Prop :=
  fun initial_speed increased_speed =>
    ∀ (d t : ℝ),
      initial_speed = 45 →
      increased_speed = initial_speed + 10 →
      d = initial_speed * (t + 0.75) →
      d - 45 = increased_speed * (t - 1) →
      d = 230.625

-- The proof is omitted
theorem martha_driving_distance_proof :
  martha_driving_distance 45 55 := by sorry

end NUMINAMATH_CALUDE_martha_driving_distance_martha_driving_distance_proof_l3653_365395


namespace NUMINAMATH_CALUDE_whack_a_mole_tickets_kaleb_whack_a_mole_tickets_l3653_365352

/-- Given that Kaleb won tickets from two games and could buy a certain number of candies,
    we prove how many tickets he won from one of the games. -/
theorem whack_a_mole_tickets (skee_ball_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  let total_tickets := candy_cost * candies_bought
  total_tickets - skee_ball_tickets

/-- Proof of the specific problem where Kaleb won 8 tickets playing 'whack a mole' -/
theorem kaleb_whack_a_mole_tickets : whack_a_mole_tickets 7 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_whack_a_mole_tickets_kaleb_whack_a_mole_tickets_l3653_365352


namespace NUMINAMATH_CALUDE_triangle_problem_l3653_365361

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  (2 * a^2 * Real.sin B * Real.sin C = Real.sqrt 3 * (a^2 + b^2 - c^2) * Real.sin A) →
  (a = 1) →
  (b = 2) →
  (D = ((A + B) / 2, 0)) →  -- Assuming A and B are coordinates on the x-axis
  (C = Real.pi / 3) ∧
  (Real.sqrt ((C - D.1)^2 + D.2^2) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3653_365361


namespace NUMINAMATH_CALUDE_cube_divisors_of_four_divisor_number_l3653_365337

/-- If an integer n has exactly 4 positive divisors (including 1 and n),
    then n^3 has 16 positive divisors. -/
theorem cube_divisors_of_four_divisor_number (n : ℕ) :
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 4 ∧ 1 ∈ d ∧ n ∈ d) →
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n^3} ∧ d.card = 16) :=
sorry

end NUMINAMATH_CALUDE_cube_divisors_of_four_divisor_number_l3653_365337


namespace NUMINAMATH_CALUDE_weight_replacement_l3653_365307

theorem weight_replacement (n : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  n = 8 →
  average_increase = 1.5 →
  new_weight = 77 →
  ∃ old_weight : ℝ,
    old_weight = new_weight - n * average_increase ∧
    old_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l3653_365307


namespace NUMINAMATH_CALUDE_trailer_homes_calculation_l3653_365344

/-- The number of new trailer homes added to Maple Drive -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes on Maple Drive -/
def initial_homes : ℕ := 25

/-- The number of years that have passed -/
def years_passed : ℕ := 3

/-- The initial average age of the trailer homes -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

theorem trailer_homes_calculation :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end NUMINAMATH_CALUDE_trailer_homes_calculation_l3653_365344


namespace NUMINAMATH_CALUDE_product_of_fractions_l3653_365356

theorem product_of_fractions : (1/2 : ℚ) * (3/5 : ℚ) * (5/6 : ℚ) = (1/4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3653_365356


namespace NUMINAMATH_CALUDE_rectangle_area_l3653_365387

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_breadth = 10)
  : ∃ (circle_radius : ℝ) (rectangle_length : ℝ),
    circle_radius ^ 2 = square_area ∧
    rectangle_length = (2 / 5) * circle_radius ∧
    rectangle_length * rectangle_breadth = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3653_365387


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l3653_365343

/-- A matrix is its own inverse if and only if its square is the identity matrix. -/
theorem matrix_is_own_inverse (c d : ℚ) : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; c, d]
  A * A = 1 ↔ c = 15/2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l3653_365343


namespace NUMINAMATH_CALUDE_pens_taken_after_first_month_l3653_365373

theorem pens_taken_after_first_month 
  (total_pens : ℕ) 
  (pens_taken_second_month : ℕ) 
  (remaining_pens : ℕ) : 
  total_pens = 315 → 
  pens_taken_second_month = 41 → 
  remaining_pens = 237 → 
  total_pens - (total_pens - remaining_pens - pens_taken_second_month) - pens_taken_second_month = remaining_pens → 
  total_pens - remaining_pens - pens_taken_second_month = 37 := by
  sorry

end NUMINAMATH_CALUDE_pens_taken_after_first_month_l3653_365373


namespace NUMINAMATH_CALUDE_infiniteSum_equals_power_l3653_365322

/-- Number of paths from (0,0) to (k,n) satisfying the given conditions -/
def C (k n : ℕ) : ℕ := sorry

/-- The sum of C_{100j+19,17} for j from 0 to infinity -/
def infiniteSum : ℕ := sorry

/-- Theorem stating that the infinite sum equals 100^17 -/
theorem infiniteSum_equals_power : infiniteSum = 100^17 := by sorry

end NUMINAMATH_CALUDE_infiniteSum_equals_power_l3653_365322


namespace NUMINAMATH_CALUDE_rectangle_width_l3653_365311

theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 100) : w = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3653_365311


namespace NUMINAMATH_CALUDE_inequality_proof_l3653_365394

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3653_365394


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l3653_365310

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l3653_365310


namespace NUMINAMATH_CALUDE_convex_ngon_coincidence_l3653_365319

/-- A convex n-gon in a 2D plane -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry  -- Axiom for convexity

/-- Predicate to check if one n-gon's vertices lie within another -/
def vertices_within (P Q : ConvexNGon n) : Prop := sorry

/-- Predicate to check if two n-gons are congruent -/
def congruent (P Q : ConvexNGon n) : Prop := sorry

/-- Predicate to check if vertices of two n-gons coincide -/
def vertices_coincide (P Q : ConvexNGon n) : Prop := sorry

/-- Theorem: If two congruent convex n-gons have the vertices of one within the other, 
    then their vertices coincide -/
theorem convex_ngon_coincidence (n : ℕ) (P Q : ConvexNGon n) 
  (h1 : vertices_within P Q) (h2 : congruent P Q) : 
  vertices_coincide P Q := by
  sorry

end NUMINAMATH_CALUDE_convex_ngon_coincidence_l3653_365319


namespace NUMINAMATH_CALUDE_committee_selection_l3653_365313

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 15) : Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3653_365313


namespace NUMINAMATH_CALUDE_circle_radius_given_area_and_circumference_sum_l3653_365340

theorem circle_radius_given_area_and_circumference_sum (x y : ℝ) :
  x ≥ 0 →
  y > 0 →
  x = π * (y / (2 * π))^2 →
  x + y = 90 * π →
  y / (2 * π) = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_given_area_and_circumference_sum_l3653_365340


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3653_365300

theorem students_playing_both_sports 
  (total : ℕ) 
  (hockey : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_neither : neither = 4) :
  hockey + basketball - (total - neither) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3653_365300


namespace NUMINAMATH_CALUDE_abs_decreasing_neg_l3653_365304

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := abs x

-- State the theorem
theorem abs_decreasing_neg : ∀ x y : ℝ, x < y → y < 0 → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_abs_decreasing_neg_l3653_365304


namespace NUMINAMATH_CALUDE_ellipse_y_axis_iff_l3653_365369

/-- Predicate to check if an equation represents an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

/-- The condition for m and n is necessary and sufficient for representing an ellipse with foci on the y-axis -/
theorem ellipse_y_axis_iff (m n : ℝ) :
  is_ellipse_y_axis m n ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_iff_l3653_365369


namespace NUMINAMATH_CALUDE_negation_of_implication_is_true_l3653_365355

theorem negation_of_implication_is_true : 
  ¬(∀ a : ℝ, a ≤ 2 → a^2 < 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_is_true_l3653_365355


namespace NUMINAMATH_CALUDE_pants_price_calculation_l3653_365363

/-- Given 10 pairs of pants with a 20% discount, followed by a 10% tax,
    resulting in a final price of $396, prove that the original retail
    price of each pair of pants is $45. -/
theorem pants_price_calculation (quantity : Nat) (discount_rate : Real)
    (tax_rate : Real) (final_price : Real) :
  quantity = 10 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  final_price = 396 →
  ∃ (original_price : Real),
    original_price = 45 ∧
    final_price = quantity * original_price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_pants_price_calculation_l3653_365363


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3653_365386

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 2 * a 5 * a 8 = 64 := by
  sorry

#check geometric_sequence_product

end NUMINAMATH_CALUDE_geometric_sequence_product_l3653_365386


namespace NUMINAMATH_CALUDE_piravena_round_trip_cost_l3653_365346

/-- Represents the cost of a journey between two cities -/
structure JourneyCost where
  distance : ℝ
  rate : ℝ
  bookingFee : ℝ := 0

def totalCost (journey : JourneyCost) : ℝ :=
  journey.distance * journey.rate + journey.bookingFee

def roundTripCost (outbound outboundRate inbound inboundRate bookingFee : ℝ) : ℝ :=
  totalCost { distance := outbound, rate := outboundRate, bookingFee := bookingFee } +
  totalCost { distance := inbound, rate := inboundRate }

theorem piravena_round_trip_cost :
  let distanceAB : ℝ := 4000
  let distanceAC : ℝ := 3000
  let busRate : ℝ := 0.20
  let planeRate : ℝ := 0.12
  let planeBookingFee : ℝ := 120
  roundTripCost distanceAB planeRate distanceAB busRate planeBookingFee = 1400 := by
  sorry

end NUMINAMATH_CALUDE_piravena_round_trip_cost_l3653_365346


namespace NUMINAMATH_CALUDE_inequality_proof_l3653_365317

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3653_365317


namespace NUMINAMATH_CALUDE_antibiotics_cost_l3653_365314

/-- The cost of Antibiotic A per dose in dollars -/
def cost_A : ℚ := 3

/-- The number of doses of Antibiotic A per day -/
def doses_per_day_A : ℕ := 2

/-- The number of days Antibiotic A is taken per week -/
def days_per_week_A : ℕ := 3

/-- The cost of Antibiotic B per dose in dollars -/
def cost_B : ℚ := 9/2

/-- The number of doses of Antibiotic B per day -/
def doses_per_day_B : ℕ := 1

/-- The number of days Antibiotic B is taken per week -/
def days_per_week_B : ℕ := 4

/-- The total cost of antibiotics for Archie for one week -/
def total_cost : ℚ := cost_A * doses_per_day_A * days_per_week_A + cost_B * doses_per_day_B * days_per_week_B

theorem antibiotics_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_antibiotics_cost_l3653_365314


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3653_365358

theorem root_sum_reciprocal (α β γ : ℝ) : 
  (60 * α^3 - 80 * α^2 + 24 * α - 2 = 0) →
  (60 * β^3 - 80 * β^2 + 24 * β - 2 = 0) →
  (60 * γ^3 - 80 * γ^2 + 24 * γ - 2 = 0) →
  (α ≠ β) → (β ≠ γ) → (α ≠ γ) →
  (0 < α) → (α < 1) →
  (0 < β) → (β < 1) →
  (0 < γ) → (γ < 1) →
  (1 / (1 - α) + 1 / (1 - β) + 1 / (1 - γ) = 22) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3653_365358


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_13_l3653_365391

theorem sum_seven_smallest_multiples_of_13 : 
  (Finset.range 7).sum (fun i => 13 * (i + 1)) = 364 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_13_l3653_365391


namespace NUMINAMATH_CALUDE_tuition_room_board_difference_l3653_365318

/-- Given the total cost and tuition cost at State University, prove the difference between tuition and room and board costs. -/
theorem tuition_room_board_difference (total_cost tuition_cost : ℕ) 
  (h1 : total_cost = 2584)
  (h2 : tuition_cost = 1644)
  (h3 : tuition_cost > total_cost - tuition_cost) :
  tuition_cost - (total_cost - tuition_cost) = 704 := by
  sorry

end NUMINAMATH_CALUDE_tuition_room_board_difference_l3653_365318


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_700_l3653_365370

theorem greatest_multiple_of_5_and_7_less_than_700 : 
  ∀ n : ℕ, n % 5 = 0 ∧ n % 7 = 0 ∧ n < 700 → n ≤ 700 := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_700_l3653_365370


namespace NUMINAMATH_CALUDE_jadens_car_count_l3653_365392

/-- Calculates the final number of toy cars Jaden has after a series of transactions -/
def jadensFinalCarCount (initial bought birthday givenSister givenVinnie tradedAway tradedFor : ℕ) : ℕ :=
  initial + bought + birthday - givenSister - givenVinnie - tradedAway + tradedFor

/-- Theorem stating that Jaden ends up with 45 toy cars -/
theorem jadens_car_count : 
  jadensFinalCarCount 14 28 12 8 3 5 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jadens_car_count_l3653_365392


namespace NUMINAMATH_CALUDE_correct_answer_l3653_365376

theorem correct_answer (x : ℚ) (h : 2 * x = 80) : x / 3 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l3653_365376


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_negation_and_disjunction_correct_propositions_l3653_365375

-- Proposition ②
theorem contrapositive_real_roots (q : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1 :=
sorry

-- Proposition ③
theorem negation_and_disjunction (p q : Prop) :
  ¬p ∧ (p ∨ q) → q :=
sorry

-- Main theorem combining both propositions
theorem correct_propositions :
  (∃ q : ℝ, (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1) ∧
  (∀ p q : Prop, ¬p ∧ (p ∨ q) → q) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_negation_and_disjunction_correct_propositions_l3653_365375


namespace NUMINAMATH_CALUDE_min_value_of_function_lower_bound_is_tight_l3653_365364

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (3 * x^2 + 6 * x + 19) / (8 * (1 + x)) ≥ Real.sqrt 3 :=
sorry

theorem lower_bound_is_tight :
  ∃ x : ℝ, x ≥ 0 ∧ (3 * x^2 + 6 * x + 19) / (8 * (1 + x)) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_lower_bound_is_tight_l3653_365364


namespace NUMINAMATH_CALUDE_base_seven_division_1452_14_l3653_365354

/-- Represents a number in base 7 --/
def BaseSevenNum := List Nat

/-- Converts a base 7 number to base 10 --/
def to_base_ten (n : BaseSevenNum) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 --/
def to_base_seven (n : Nat) : BaseSevenNum :=
  sorry

/-- Performs division in base 7 --/
def base_seven_div (a b : BaseSevenNum) : BaseSevenNum :=
  to_base_seven ((to_base_ten a) / (to_base_ten b))

theorem base_seven_division_1452_14 :
  base_seven_div [2, 5, 4, 1] [4, 1] = [3, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_seven_division_1452_14_l3653_365354


namespace NUMINAMATH_CALUDE_min_value_expression_l3653_365389

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  b / a^2 + 4 / b + a / 2 ≥ 2 * Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ b₀ / a₀^2 + 4 / b₀ + a₀ / 2 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3653_365389


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3653_365339

-- Define the function f
def f (x : ℝ) : ℝ := x * (3 - x^2)

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.sqrt 2 }

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = 0 ∧ ∀ x ∈ interval, f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3653_365339
