import Mathlib

namespace NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l605_60534

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of derangements of n objects -/
def derangement (n : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The probability of exactly k people getting the right letter when n letters are randomly distributed to n people -/
def probability_correct_letters (n k : ℕ) : ℚ :=
  (choose n k * derangement (n - k)) / factorial n

theorem probability_three_correct_out_of_seven :
  probability_correct_letters 7 3 = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l605_60534


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l605_60537

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z := (4 : ℂ) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l605_60537


namespace NUMINAMATH_CALUDE_alex_coin_distribution_distribution_satisfies_conditions_l605_60582

/-- The minimum number of additional coins needed -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's scenario -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

/-- Proof that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  ∀ i j, i ≠ j → i ≤ num_friends → j ≤ num_friends → 
  (i : ℕ) ≠ (j : ℕ) ∧ (i : ℕ) ≥ 1 ∧ (j : ℕ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_distribution_satisfies_conditions_l605_60582


namespace NUMINAMATH_CALUDE_distance_to_stream_is_six_l605_60551

/-- Represents a trapezoidal forest with a stream -/
structure TrapezidalForest where
  side1 : ℝ  -- Length of the side closest to Wendy's house
  side2 : ℝ  -- Length of the opposite parallel side
  area : ℝ   -- Total area of the forest
  stream_divides_in_half : Bool  -- Whether the stream divides the area in half

/-- The distance from either parallel side to the stream in the trapezoidal forest -/
def distance_to_stream (forest : TrapezidalForest) : ℝ :=
  sorry

/-- Theorem stating that the distance to the stream is 6 miles for the given forest -/
theorem distance_to_stream_is_six (forest : TrapezidalForest) 
  (h1 : forest.side1 = 8)
  (h2 : forest.side2 = 14)
  (h3 : forest.area = 132)
  (h4 : forest.stream_divides_in_half = true) :
  distance_to_stream forest = 6 :=
  sorry

end NUMINAMATH_CALUDE_distance_to_stream_is_six_l605_60551


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l605_60526

theorem distribute_and_simplify (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l605_60526


namespace NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l605_60512

theorem obtuse_triangle_from_altitudes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = 13) (h5 : b = 11) (h6 : c = 5) :
  (c^2 + b^2 - a^2) / (2 * b * c) < 0 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l605_60512


namespace NUMINAMATH_CALUDE_initial_workers_count_l605_60546

/-- The initial number of workers that can complete a job in 25 days, 
    where adding 10 more workers allows the job to be completed in 15 days -/
def initial_workers : ℕ :=
  sorry

/-- The total amount of work to be done -/
def total_work : ℝ :=
  sorry

theorem initial_workers_count : initial_workers = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l605_60546


namespace NUMINAMATH_CALUDE_ellipse_y_axis_l605_60591

theorem ellipse_y_axis (k : ℝ) (h : k < -1) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (1 - k) * x^2 + y^2 = k^2 - 1 ↔ (x^2 / b^2) + (y^2 / a^2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_l605_60591


namespace NUMINAMATH_CALUDE_last_year_production_l605_60539

/-- The number of eggs produced by farms in Douglas County --/
structure EggProduction where
  lastYear : ℕ
  thisYear : ℕ
  increase : ℕ

/-- Theorem stating the relationship between this year's and last year's egg production --/
theorem last_year_production (e : EggProduction) 
  (h1 : e.thisYear = 4636)
  (h2 : e.increase = 3220)
  (h3 : e.thisYear = e.lastYear + e.increase) :
  e.lastYear = 1416 := by
  sorry

end NUMINAMATH_CALUDE_last_year_production_l605_60539


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l605_60541

/-- Represents the payment for Scheme 1 -/
def scheme1_payment (x : ℕ) : ℚ :=
  90 * (1 - 30/100) * x + 100 * (1 - 15/100) * (2*x + 1)

/-- Represents the payment for Scheme 2 -/
def scheme2_payment (x : ℕ) : ℚ :=
  (90*x + 100*(2*x + 1)) * (1 - 20/100)

/-- Theorem stating that Scheme 2 is more cost-effective for x ≥ 33 -/
theorem scheme2_more_cost_effective (x : ℕ) (h : x ≥ 33) :
  scheme2_payment x < scheme1_payment x :=
sorry

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l605_60541


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l605_60597

theorem cricket_team_right_handed_players
  (total_players : ℕ) (throwers : ℕ) (left_handed_thrower_percent : ℚ)
  (right_handed_thrower_avg : ℕ) (left_handed_thrower_avg : ℕ)
  (total_runs : ℕ) (left_handed_non_thrower_runs : ℕ) :
  total_players = 120 →
  throwers = 55 →
  left_handed_thrower_percent = 1/5 →
  right_handed_thrower_avg = 25 →
  left_handed_thrower_avg = 30 →
  total_runs = 3620 →
  left_handed_non_thrower_runs = 720 →
  ∃ (batsmen_runs all_rounder_runs : ℕ),
    batsmen_runs = 2 * all_rounder_runs ∧
    batsmen_runs + all_rounder_runs = total_runs - (throwers * right_handed_thrower_avg * (1 - left_handed_thrower_percent) +
                                                    throwers * left_handed_thrower_avg * left_handed_thrower_percent) →
  ∃ (left_handed_non_throwers right_handed_non_throwers : ℕ),
    5 * left_handed_non_throwers = right_handed_non_throwers ∧
    left_handed_non_throwers * left_handed_thrower_avg = left_handed_non_thrower_runs →
  throwers * (1 - left_handed_thrower_percent) + right_handed_non_throwers = 164 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l605_60597


namespace NUMINAMATH_CALUDE_rope_sections_l605_60580

theorem rope_sections (initial_rope : ℝ) (art_fraction : ℝ) (friend_fraction : ℝ) (section_length : ℝ) : 
  initial_rope = 50 →
  art_fraction = 1 / 5 →
  friend_fraction = 1 / 2 →
  section_length = 2 →
  let remaining_after_art := initial_rope - (art_fraction * initial_rope)
  let remaining_after_friend := remaining_after_art - (friend_fraction * remaining_after_art)
  let num_sections := remaining_after_friend / section_length
  num_sections = 10 := by sorry

end NUMINAMATH_CALUDE_rope_sections_l605_60580


namespace NUMINAMATH_CALUDE_johns_drive_time_l605_60532

theorem johns_drive_time (speed : ℝ) (total_distance : ℝ) (after_lunch_time : ℝ) 
  (h : speed = 55)
  (h' : total_distance = 275)
  (h'' : after_lunch_time = 3) :
  let before_lunch_time := (total_distance - speed * after_lunch_time) / speed
  before_lunch_time = 2 := by sorry

end NUMINAMATH_CALUDE_johns_drive_time_l605_60532


namespace NUMINAMATH_CALUDE_equal_fractions_k_value_l605_60529

theorem equal_fractions_k_value 
  (x y z k : ℝ) 
  (h : (8 : ℝ) / (x + y + 1) = k / (x + z + 2) ∧ 
       k / (x + z + 2) = (12 : ℝ) / (z - y + 3)) : 
  k = 20 := by sorry

end NUMINAMATH_CALUDE_equal_fractions_k_value_l605_60529


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_four_l605_60515

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem subset_implies_m_equals_four (m : ℝ) : B ⊆ A m → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_four_l605_60515


namespace NUMINAMATH_CALUDE_expression_evaluation_l605_60566

theorem expression_evaluation (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l605_60566


namespace NUMINAMATH_CALUDE_total_time_is_383_l605_60514

def total_time (mac_download : ℕ) (ny_audio_glitch : ℕ) (ny_video_glitch : ℕ) 
  (berlin_audio_glitch : ℕ) (berlin_video_glitch : ℕ) (tokyo_audio_glitch : ℕ) 
  (tokyo_video_glitch : ℕ) (sydney_audio_glitch : ℕ) : ℕ :=
  let windows_download := 3 * mac_download
  let ny_glitch_time := 2 * ny_audio_glitch + ny_video_glitch
  let ny_total := ny_glitch_time + 3 * ny_glitch_time
  let berlin_glitch_time := 3 * berlin_audio_glitch + 2 * berlin_video_glitch
  let berlin_total := berlin_glitch_time + 2 * berlin_glitch_time
  let tokyo_glitch_time := tokyo_audio_glitch + 2 * tokyo_video_glitch
  let tokyo_total := tokyo_glitch_time + 4 * tokyo_glitch_time
  let sydney_glitch_time := 2 * sydney_audio_glitch
  let sydney_total := sydney_glitch_time + 5 * sydney_glitch_time
  mac_download + windows_download + ny_total + berlin_total + tokyo_total + sydney_total

theorem total_time_is_383 : 
  total_time 10 6 8 4 5 7 9 6 = 383 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_383_l605_60514


namespace NUMINAMATH_CALUDE_trig_identities_l605_60517

theorem trig_identities (α : Real) (h : Real.sin α = 2 * Real.cos α) : 
  ((2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l605_60517


namespace NUMINAMATH_CALUDE_walnut_count_l605_60540

theorem walnut_count (total : ℕ) (p a c w : ℕ) : 
  total = 150 →
  p + a + c + w = total →
  a = p / 2 →
  c = 4 * a →
  w = 3 * c →
  w = 96 := by
  sorry

end NUMINAMATH_CALUDE_walnut_count_l605_60540


namespace NUMINAMATH_CALUDE_train_length_l605_60565

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 50 → time = 9 → ∃ length : ℝ, 
  (length ≥ 124.5 ∧ length ≤ 125.5) ∧ length = speed * 1000 / 3600 * time :=
sorry

end NUMINAMATH_CALUDE_train_length_l605_60565


namespace NUMINAMATH_CALUDE_polynomial_degree_example_l605_60518

/-- The degree of a polynomial (3x^5 + 2x^4 - x^2 + 5)(4x^{11} - 8x^8 + 3x^5 - 10) - (x^3 + 7)^6 -/
theorem polynomial_degree_example : 
  let p₁ : Polynomial ℝ := X^5 * 3 + X^4 * 2 - X^2 + 5
  let p₂ : Polynomial ℝ := X^11 * 4 - X^8 * 8 + X^5 * 3 - 10
  let p₃ : Polynomial ℝ := (X^3 + 7)^6
  (p₁ * p₂ - p₃).degree = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_example_l605_60518


namespace NUMINAMATH_CALUDE_f_inequality_l605_60557

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : f (a * b) > f a - f (-b) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l605_60557


namespace NUMINAMATH_CALUDE_adam_final_score_l605_60528

/-- Calculates the final score in a trivia game based on the given conditions --/
def calculate_final_score (
  science_correct : ℕ)
  (history_correct : ℕ)
  (sports_correct : ℕ)
  (literature_correct : ℕ)
  (science_points : ℕ)
  (history_points : ℕ)
  (sports_points : ℕ)
  (literature_points : ℕ)
  (history_multiplier : ℕ)
  (literature_penalty : ℕ) : ℕ :=
  science_correct * science_points +
  history_correct * history_points * history_multiplier +
  sports_correct * sports_points +
  literature_correct * (literature_points - literature_penalty)

/-- Theorem stating that Adam's final score is 99 points --/
theorem adam_final_score :
  calculate_final_score 5 3 1 1 10 5 15 7 2 3 = 99 := by
  sorry

#eval calculate_final_score 5 3 1 1 10 5 15 7 2 3

end NUMINAMATH_CALUDE_adam_final_score_l605_60528


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l605_60522

theorem min_reciprocal_sum (x y : ℝ) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ a b : ℝ, Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l605_60522


namespace NUMINAMATH_CALUDE_xyz_product_l605_60516

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 120 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l605_60516


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l605_60530

def A : Set ℝ := {x | x^2 - 3*x - 4 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l605_60530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l605_60538

theorem arithmetic_sequence_problem (k : ℕ+) : 
  let a : ℕ → ℤ := λ n => 2*n + 2
  let S : ℕ → ℤ := λ n => n^2 + 3*n
  S k - a (k + 5) = 44 → k = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l605_60538


namespace NUMINAMATH_CALUDE_one_third_recipe_sugar_l605_60595

def original_recipe : ℚ := 7 + 3/4

theorem one_third_recipe_sugar (original : ℚ) (reduced : ℚ) : 
  original = 7 + 3/4 → reduced = original * (1/3) → reduced = 2 + 7/12 := by
  sorry

end NUMINAMATH_CALUDE_one_third_recipe_sugar_l605_60595


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_twelve_after_subtracting_seven_l605_60572

theorem smallest_number_divisible_by_twelve_after_subtracting_seven : 
  ∃ N : ℕ, N > 0 ∧ (N - 7) % 12 = 0 ∧ ∀ m : ℕ, m > 0 → (m - 7) % 12 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_twelve_after_subtracting_seven_l605_60572


namespace NUMINAMATH_CALUDE_sqrt_37_between_6_and_7_l605_60510

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_37_between_6_and_7_l605_60510


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l605_60520

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 6 → width = 4.75 → rate = 900 → length * width * rate = 25650 := by sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l605_60520


namespace NUMINAMATH_CALUDE_two_red_balls_probability_l605_60575

/-- Represents a bag of colored balls -/
structure Bag where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a red ball from a given bag -/
def redProbability (bag : Bag) : Rat :=
  bag.red / (bag.red + bag.white)

/-- Theorem: The probability of drawing two red balls, one from each bag, is 1/9 -/
theorem two_red_balls_probability
  (bagA : Bag)
  (bagB : Bag)
  (hA : bagA = { red := 4, white := 2 })
  (hB : bagB = { red := 1, white := 5 }) :
  redProbability bagA * redProbability bagB = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_two_red_balls_probability_l605_60575


namespace NUMINAMATH_CALUDE_inequality_solution_l605_60511

theorem inequality_solution (x : ℝ) : 3 * x^2 - x < 8 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l605_60511


namespace NUMINAMATH_CALUDE_cube_increase_theorem_l605_60527

def cube_edge_increase_percent : ℝ := 60

theorem cube_increase_theorem (s : ℝ) (h : s > 0) :
  let new_edge := s * (1 + cube_edge_increase_percent / 100)
  let original_surface_area := 6 * s^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 156 ∧
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry


end NUMINAMATH_CALUDE_cube_increase_theorem_l605_60527


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l605_60553

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  (a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) → (q = -2 ∨ q = 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l605_60553


namespace NUMINAMATH_CALUDE_lawns_mowed_count_l605_60570

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def driveways_shoveled : ℕ := 5

def total_money : ℕ := shoe_cost + change_after_purchase
def allowance_savings : ℕ := saving_months * monthly_allowance
def shoveling_earnings : ℕ := driveways_shoveled * driveway_shoveling_charge
def mowing_earnings : ℕ := total_money - allowance_savings - shoveling_earnings

theorem lawns_mowed_count : mowing_earnings / lawn_mowing_charge = 4 := by
  sorry

end NUMINAMATH_CALUDE_lawns_mowed_count_l605_60570


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l605_60552

theorem unique_solution_cubic_equation :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 41 ↔ x = 5 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l605_60552


namespace NUMINAMATH_CALUDE_min_shots_13x13_grid_l605_60585

/-- Represents a grid with side length n -/
def Grid (n : ℕ) := Fin n × Fin n

/-- The set of possible moves for the target -/
def neighborMoves : List (ℤ × ℤ) :=
  [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

/-- Check if a move is valid within the grid -/
def isValidMove (n : ℕ) (pos : Grid n) (move : ℤ × ℤ) : Bool :=
  let (x, y) := pos
  let (dx, dy) := move
  0 ≤ x.val + dx ∧ x.val + dx < n ∧ 0 ≤ y.val + dy ∧ y.val + dy < n

/-- The minimum number of shots required to guarantee hitting the target twice -/
def minShotsToDestroy (n : ℕ) : ℕ :=
  n * n + (n * n + 1) / 2

/-- Theorem stating the minimum number of shots required for a 13x13 grid -/
theorem min_shots_13x13_grid :
  minShotsToDestroy 13 = 254 :=
sorry

end NUMINAMATH_CALUDE_min_shots_13x13_grid_l605_60585


namespace NUMINAMATH_CALUDE_parabola_equation_l605_60576

/-- A parabola with directrix x = -7 has the standard equation y² = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ k, ∀ x y, p (x, y) ↔ y^2 = 4 * k * x ∧ k > 0) →  -- general form of parabola equation
  (∀ x y, p (x, y) ↔ y^2 = 28 * x) :=  -- standard equation to be proved
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l605_60576


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l605_60579

theorem manuscript_revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ)
  (initial_cost_per_page : ℚ) (total_cost : ℚ)
  (h1 : total_pages = 100)
  (h2 : revised_once = 30)
  (h3 : revised_twice = 20)
  (h4 : initial_cost_per_page = 5)
  (h5 : total_cost = 710)
  (h6 : total_pages = revised_once + revised_twice + (total_pages - revised_once - revised_twice)) :
  let revision_cost : ℚ := (total_cost - (initial_cost_per_page * total_pages)) / (revised_once + 2 * revised_twice)
  revision_cost = 3 := by sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l605_60579


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l605_60593

theorem factorization_of_difference_of_squares (a b : ℝ) :
  36 * a^2 - 4 * b^2 = 4 * (3*a + b) * (3*a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l605_60593


namespace NUMINAMATH_CALUDE_two_thirds_of_45_plus_10_l605_60521

theorem two_thirds_of_45_plus_10 : ((2 : ℚ) / 3) * 45 + 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_45_plus_10_l605_60521


namespace NUMINAMATH_CALUDE_indefinite_integral_of_3x_squared_plus_1_l605_60561

theorem indefinite_integral_of_3x_squared_plus_1 (x : ℝ) (C : ℝ) :
  deriv (fun x => x^3 + x + C) x = 3 * x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_of_3x_squared_plus_1_l605_60561


namespace NUMINAMATH_CALUDE_line_equation_proof_l605_60596

theorem line_equation_proof (m b k : ℝ) : 
  (∃ k, (k^2 + 4*k + 3 - (m*k + b) = 3 ∨ k^2 + 4*k + 3 - (m*k + b) = -3) ∧ 
        (∀ k', k' ≠ k → ¬(k'^2 + 4*k' + 3 - (m*k' + b) = 3 ∨ k'^2 + 4*k' + 3 - (m*k' + b) = -3))) →
  (m * 2 + b = 5) →
  (b ≠ 0) →
  (m = 9/2 ∧ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l605_60596


namespace NUMINAMATH_CALUDE_chess_tournament_games_l605_60502

theorem chess_tournament_games (n : Nat) (games : Fin n → Nat) :
  n = 5 →
  (∀ i j : Fin n, i ≠ j → games i + games j ≤ n - 1) →
  (∃ p : Fin n, games p = 4) →
  (∃ p : Fin n, games p = 3) →
  (∃ p : Fin n, games p = 2) →
  (∃ p : Fin n, games p = 1) →
  (∃ p : Fin n, games p = 2) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l605_60502


namespace NUMINAMATH_CALUDE_y_derivative_is_zero_l605_60567

noncomputable def y (x : ℝ) : ℝ :=
  5 * x - Real.log (1 + Real.sqrt (1 - Real.exp (10 * x))) - Real.exp (-5 * x) * Real.arcsin (Real.exp (5 * x))

theorem y_derivative_is_zero :
  ∀ x : ℝ, deriv y x = 0 :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_is_zero_l605_60567


namespace NUMINAMATH_CALUDE_perpendicular_lines_product_sum_zero_l605_60569

/-- Two lines in the plane -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Perpendicularity of two lines -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.A * l₂.A + l₁.B * l₂.B = 0

/-- Theorem: If two lines are perpendicular, then the sum of the products of their coefficients is zero -/
theorem perpendicular_lines_product_sum_zero (l₁ l₂ : Line) :
  perpendicular l₁ l₂ → l₁.A * l₂.A + l₁.B * l₂.B = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_product_sum_zero_l605_60569


namespace NUMINAMATH_CALUDE_probability_is_half_l605_60574

/-- Represents a game board as described in the problem -/
structure GameBoard where
  total_regions : ℕ
  shaded_regions : ℕ
  h_total : total_regions = 8
  h_shaded : shaded_regions = 4

/-- The probability of landing in a shaded region on the game board -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem probability_is_half (board : GameBoard) : probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l605_60574


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l605_60590

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l605_60590


namespace NUMINAMATH_CALUDE_tim_final_coin_count_l605_60525

/-- Represents the count of different types of coins -/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ
  pennies : ℕ

/-- Represents a transaction that modifies the coin count -/
inductive Transaction
  | DadGift : Transaction
  | DadExchange : Transaction
  | PaySister : Transaction
  | BuySnack : Transaction
  | ExchangeQuarter : Transaction

def initial_coins : CoinCount :=
  { quarters := 7, nickels := 9, dimes := 12, pennies := 5 }

def apply_transaction (coins : CoinCount) (t : Transaction) : CoinCount :=
  match t with
  | Transaction.DadGift => 
      { quarters := coins.quarters + 2,
        nickels := coins.nickels + 3,
        dimes := coins.dimes,
        pennies := coins.pennies + 5 }
  | Transaction.DadExchange => 
      { quarters := coins.quarters + 4,
        nickels := coins.nickels,
        dimes := coins.dimes - 10,
        pennies := coins.pennies }
  | Transaction.PaySister => 
      { quarters := coins.quarters,
        nickels := coins.nickels - 5,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.BuySnack => 
      { quarters := coins.quarters - 2,
        nickels := coins.nickels - 4,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.ExchangeQuarter => 
      { quarters := coins.quarters - 1,
        nickels := coins.nickels + 5,
        dimes := coins.dimes,
        pennies := coins.pennies }

def final_coins : CoinCount :=
  apply_transaction
    (apply_transaction
      (apply_transaction
        (apply_transaction
          (apply_transaction initial_coins Transaction.DadGift)
          Transaction.DadExchange)
        Transaction.PaySister)
      Transaction.BuySnack)
    Transaction.ExchangeQuarter

theorem tim_final_coin_count :
  final_coins = { quarters := 10, nickels := 8, dimes := 2, pennies := 10 } :=
by sorry

end NUMINAMATH_CALUDE_tim_final_coin_count_l605_60525


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l605_60562

def repeating_decimal : ℚ := 7 + 2/10 + 34/99/100

theorem repeating_decimal_as_fraction : 
  repeating_decimal = 36357 / 4950 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l605_60562


namespace NUMINAMATH_CALUDE_money_left_over_calculation_l605_60587

/-- The amount of money left over after purchasing bread and peanut butter -/
def money_left_over (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) (initial_amount : ℚ) : ℚ :=
  initial_amount - (bread_price * bread_quantity + peanut_butter_price)

/-- Theorem stating the amount of money left over in the given scenario -/
theorem money_left_over_calculation :
  let bread_price : ℚ := 9/4  -- $2.25 as a rational number
  let bread_quantity : ℕ := 3
  let peanut_butter_price : ℚ := 2
  let initial_amount : ℚ := 14
  money_left_over bread_price bread_quantity peanut_butter_price initial_amount = 21/4  -- $5.25 as a rational number
  := by sorry

end NUMINAMATH_CALUDE_money_left_over_calculation_l605_60587


namespace NUMINAMATH_CALUDE_powers_of_i_sum_l605_60531

def i : ℂ := Complex.I

theorem powers_of_i_sum (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + i^34 + i^39 = -1 - i :=
by sorry

end NUMINAMATH_CALUDE_powers_of_i_sum_l605_60531


namespace NUMINAMATH_CALUDE_equation_solution_l605_60524

theorem equation_solution (x : Real) :
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔
  (∃ k : ℤ, x = 2 * π / 3 + 2 * k * π ∨ x = 7 * π / 6 + 2 * k * π ∨ x = -π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l605_60524


namespace NUMINAMATH_CALUDE_field_fencing_l605_60547

/-- A rectangular field with one side of 20 feet and an area of 600 square feet
    requires 80 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length = 20 →
  length * width = 600 →
  length + 2 * width = 80 :=
by sorry

end NUMINAMATH_CALUDE_field_fencing_l605_60547


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l605_60594

/-- Calculates the total driving hours in a year for a person who drives to town twice each month -/
def yearly_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * 12

theorem melissa_driving_hours :
  yearly_driving_hours 2 3 = 72 := by
sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l605_60594


namespace NUMINAMATH_CALUDE_population_growth_proof_l605_60578

/-- Represents the annual growth rate of the population -/
def growth_rate : ℝ := 0.20

/-- Represents the population after one year of growth -/
def final_population : ℝ := 12000

/-- Represents the initial population before growth -/
def initial_population : ℝ := 10000

/-- Theorem stating that if a population grows by 20% in one year to reach 12,000,
    then the initial population was 10,000 -/
theorem population_growth_proof :
  final_population = initial_population * (1 + growth_rate) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_proof_l605_60578


namespace NUMINAMATH_CALUDE_sarah_savings_l605_60589

/-- Represents Sarah's savings pattern over time -/
def savings_pattern : List (Nat × Nat) :=
  [(4, 5), (4, 10), (4, 20)]

/-- Calculates the total amount saved given a savings pattern -/
def total_saved (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, amount) => acc + weeks * amount) 0

/-- Calculates the total number of weeks in a savings pattern -/
def total_weeks (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, _) => acc + weeks) 0

/-- Theorem: Sarah saves $140 in 12 weeks -/
theorem sarah_savings : 
  total_saved savings_pattern = 140 ∧ total_weeks savings_pattern = 12 :=
sorry

end NUMINAMATH_CALUDE_sarah_savings_l605_60589


namespace NUMINAMATH_CALUDE_toy_gift_box_discount_l605_60550

theorem toy_gift_box_discount (cost_price marked_price discount profit_margin : ℝ) : 
  cost_price = 160 →
  marked_price = 240 →
  discount = 20 →
  profit_margin = 20 →
  marked_price * (1 - discount / 100) = cost_price * (1 + profit_margin / 100) :=
by sorry

end NUMINAMATH_CALUDE_toy_gift_box_discount_l605_60550


namespace NUMINAMATH_CALUDE_diagonal_difference_l605_60588

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The difference between the number of diagonals in an octagon and a heptagon -/
theorem diagonal_difference : num_diagonals 8 - num_diagonals 7 = 6 := by sorry

end NUMINAMATH_CALUDE_diagonal_difference_l605_60588


namespace NUMINAMATH_CALUDE_initial_blocks_count_l605_60519

/-- The initial number of blocks Adolfo had -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Adolfo added -/
def added_blocks : ℕ := 30

/-- The total number of blocks after adding -/
def total_blocks : ℕ := 65

/-- Theorem stating that the initial number of blocks is 35 -/
theorem initial_blocks_count : initial_blocks = 35 := by
  sorry

/-- Axiom representing the relationship between initial, added, and total blocks -/
axiom block_relationship : initial_blocks + added_blocks = total_blocks


end NUMINAMATH_CALUDE_initial_blocks_count_l605_60519


namespace NUMINAMATH_CALUDE_real_roots_range_l605_60573

theorem real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) ↔ 
  (m ≤ -3/2 ∨ m ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_real_roots_range_l605_60573


namespace NUMINAMATH_CALUDE_candace_hiking_ratio_l605_60523

/-- Candace's hiking scenario -/
def hiking_scenario (old_speed new_speed hike_duration blister_interval blister_slowdown : ℝ) : Prop :=
  let blisters := hike_duration / blister_interval
  let total_slowdown := blisters * blister_slowdown
  let final_new_speed := new_speed - total_slowdown
  final_new_speed / old_speed = 7 / 6

/-- The theorem representing Candace's hiking problem -/
theorem candace_hiking_ratio :
  hiking_scenario 6 11 4 2 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candace_hiking_ratio_l605_60523


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l605_60563

def is_geometric_sequence (a : ℕ+ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_a8 (a : ℕ+ → ℚ) :
  is_geometric_sequence a →
  a 2 = 1 / 16 →
  a 5 = 1 / 2 →
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l605_60563


namespace NUMINAMATH_CALUDE_function_is_constant_l605_60554

/-- A function f: ℚ → ℝ satisfying |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℚ is constant. -/
theorem function_is_constant (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℚ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l605_60554


namespace NUMINAMATH_CALUDE_phone_bill_increase_l605_60542

theorem phone_bill_increase (original_monthly_bill : ℝ) (new_yearly_bill : ℝ) : 
  original_monthly_bill = 50 → 
  new_yearly_bill = 660 → 
  (new_yearly_bill / (12 * original_monthly_bill) - 1) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l605_60542


namespace NUMINAMATH_CALUDE_simplify_expression_l605_60556

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x + 2) = -2*x - 60 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l605_60556


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negative_numbers_l605_60505

theorem inverse_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negative_numbers_l605_60505


namespace NUMINAMATH_CALUDE_frequency_of_defectives_example_l605_60506

/-- Given a sample of parts, calculate the frequency of defective parts -/
def frequency_of_defectives (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for a sample of 500 parts with 8 defective parts, 
    the frequency of defective parts is 0.016 -/
theorem frequency_of_defectives_example : 
  frequency_of_defectives 500 8 = 16 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_defectives_example_l605_60506


namespace NUMINAMATH_CALUDE_max_value_of_g_l605_60592

-- Define the interval [0,1]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the function y = ax
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x

-- Define the function y = 3ax - 1
def g (a : ℝ) : ℝ → ℝ := λ x ↦ 3 * a * x - 1

-- State the theorem
theorem max_value_of_g (a : ℝ) :
  (∃ (max min : ℝ), (∀ x ∈ interval, f a x ≤ max) ∧
                    (∀ x ∈ interval, min ≤ f a x) ∧
                    max + min = 3) →
  (∃ max : ℝ, (∀ x ∈ interval, g a x ≤ max) ∧
              (∃ y ∈ interval, g a y = max) ∧
              max = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l605_60592


namespace NUMINAMATH_CALUDE_expenditure_recording_l605_60586

/-- Given that income is recorded as positive and an income of 20 yuan is recorded as +20 yuan,
    prove that an expenditure of 75 yuan should be recorded as -75 yuan. -/
theorem expenditure_recording (income_recording : ℤ → ℤ) (h : income_recording 20 = 20) :
  income_recording (-75) = -75 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_recording_l605_60586


namespace NUMINAMATH_CALUDE_parabola_vertex_l605_60558

/-- The parabola defined by the equation y = (3x-1)^2 + 2 has vertex (1/3, 2) -/
theorem parabola_vertex (x y : ℝ) :
  y = (3*x - 1)^2 + 2 →
  (∃ a h k : ℝ, a ≠ 0 ∧ y = a*(x - h)^2 + k ∧ h = 1/3 ∧ k = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l605_60558


namespace NUMINAMATH_CALUDE_middle_position_theorem_l605_60508

/-- Represents the color of a stone -/
inductive Color
  | Black
  | White

/-- Represents the state of the stone line -/
def StoneLine := Fin 2021 → Color

/-- Checks if a position is valid for the operation -/
def validPosition (n : Fin 2021) : Prop :=
  1 < n.val ∧ n.val < 2021

/-- Represents a single operation on the stone line -/
def operation (line : StoneLine) (n : Fin 2021) : StoneLine :=
  fun i => if i = n - 1 ∨ i = n + 1 then
    match line i with
    | Color.Black => Color.White
    | Color.White => Color.Black
    else line i

/-- Checks if all stones in the line are black -/
def allBlack (line : StoneLine) : Prop :=
  ∀ i, line i = Color.Black

/-- Initial configuration with one black stone at position n -/
def initialConfig (n : Fin 2021) : StoneLine :=
  fun i => if i = n then Color.Black else Color.White

/-- Represents the ability to make all stones black through operations -/
def canMakeAllBlack (line : StoneLine) : Prop :=
  ∃ (seq : List (Fin 2021)), 
    (∀ n ∈ seq, validPosition n) ∧
    allBlack (seq.foldl operation line)

/-- The main theorem to be proved -/
theorem middle_position_theorem :
  ∀ n : Fin 2021, canMakeAllBlack (initialConfig n) ↔ n = ⟨1011, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_middle_position_theorem_l605_60508


namespace NUMINAMATH_CALUDE_largest_divisor_of_f_l605_60548

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem largest_divisor_of_f :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ f n) → m ≤ 36 ∧
  ∀ n : ℕ, 36 ∣ f n :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_f_l605_60548


namespace NUMINAMATH_CALUDE_non_officers_count_l605_60501

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Average salary of all employees in the office -/
def avg_salary_all : ℕ := 120

/-- Average salary of officers -/
def avg_salary_officers : ℕ := 430

/-- Average salary of non-officers -/
def avg_salary_non_officers : ℕ := 110

/-- Number of officers -/
def num_officers : ℕ := 15

/-- Theorem stating that the number of non-officers is 465 -/
theorem non_officers_count : num_non_officers = 465 := by
  sorry

end NUMINAMATH_CALUDE_non_officers_count_l605_60501


namespace NUMINAMATH_CALUDE_sum_positive_when_difference_exceeds_absolute_value_l605_60599

theorem sum_positive_when_difference_exceeds_absolute_value
  (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_when_difference_exceeds_absolute_value_l605_60599


namespace NUMINAMATH_CALUDE_trivia_game_total_score_l605_60545

theorem trivia_game_total_score :
  let team_a : Int := 2
  let team_b : Int := 9
  let team_c : Int := 4
  let team_d : Int := -3
  let team_e : Int := 7
  let team_f : Int := 0
  let team_g : Int := 5
  let team_h : Int := -2
  team_a + team_b + team_c + team_d + team_e + team_f + team_g + team_h = 22 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_score_l605_60545


namespace NUMINAMATH_CALUDE_speed_of_sound_346_l605_60555

/-- The speed of sound as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 0.6 * t + 331

/-- Theorem: When the speed of sound is 346 m/s, the temperature is 25°C -/
theorem speed_of_sound_346 :
  ∃ (t : ℝ), speed_of_sound t = 346 ∧ t = 25 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_sound_346_l605_60555


namespace NUMINAMATH_CALUDE_third_side_length_l605_60549

/-- A triangle with two known sides and a known perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- The length of the third side of a triangle -/
def third_side (t : Triangle) : ℝ := t.perimeter - t.side1 - t.side2

/-- Theorem: In a triangle with sides 5 cm and 20 cm, and perimeter 55 cm, the third side is 30 cm -/
theorem third_side_length (t : Triangle) 
  (h1 : t.side1 = 5) 
  (h2 : t.side2 = 20) 
  (h3 : t.perimeter = 55) : 
  third_side t = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l605_60549


namespace NUMINAMATH_CALUDE_y_power_x_equals_25_l605_60564

theorem y_power_x_equals_25 (x y : ℝ) : 
  y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 5 → y^x = 25 := by
  sorry

end NUMINAMATH_CALUDE_y_power_x_equals_25_l605_60564


namespace NUMINAMATH_CALUDE_minus_one_circle_plus_four_equals_zero_l605_60577

-- Define the new operation ⊕
def circle_plus (a b : ℝ) : ℝ := a * b + b

-- Theorem statement
theorem minus_one_circle_plus_four_equals_zero : 
  circle_plus (-1) 4 = 0 := by sorry

end NUMINAMATH_CALUDE_minus_one_circle_plus_four_equals_zero_l605_60577


namespace NUMINAMATH_CALUDE_marathon_runners_finished_l605_60571

theorem marathon_runners_finished (total : ℕ) (difference : ℕ) (finished : ℕ) : 
  total = 1250 → 
  difference = 124 → 
  total = finished + (finished + difference) → 
  finished = 563 := by
sorry

end NUMINAMATH_CALUDE_marathon_runners_finished_l605_60571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l605_60513

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_a1 : a 1 = 2)  -- given a_1 = 2
  (h_a3 : a 3 = 8)  -- given a_3 = 8
  : a 2 - a 1 = 3 :=  -- prove that the common difference is 3
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l605_60513


namespace NUMINAMATH_CALUDE_rabbit_logs_l605_60544

theorem rabbit_logs (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_logs_l605_60544


namespace NUMINAMATH_CALUDE_mandy_current_pages_l605_60533

/-- Calculates the number of pages in books Mandy reads at different ages --/
def pages_at_age (initial_pages : ℕ) (initial_age : ℕ) (current_age : ℕ) : ℕ :=
  if current_age = initial_age then
    initial_pages
  else if current_age = 2 * initial_age then
    5 * initial_pages
  else if current_age = 2 * initial_age + 8 then
    3 * 5 * initial_pages
  else
    4 * 3 * 5 * initial_pages

/-- Theorem stating that Mandy now reads books with 480 pages --/
theorem mandy_current_pages : pages_at_age 8 6 (2 * 6 + 8 + 1) = 480 := by
  sorry

end NUMINAMATH_CALUDE_mandy_current_pages_l605_60533


namespace NUMINAMATH_CALUDE_atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l605_60535

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The sample space of all possible outcomes when tossing 3 coins -/
def sampleSpace : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH, CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At most one heads" -/
def atMostOneHeads : Set CoinToss := {CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At least two heads" -/
def atLeastTwoHeads : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH}

/-- Theorem stating that "At most one heads" and "At least two heads" are mutually exclusive -/
theorem atMostOneHeads_atLeastTwoHeads_mutually_exclusive : 
  atMostOneHeads ∩ atLeastTwoHeads = ∅ := by sorry

end NUMINAMATH_CALUDE_atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l605_60535


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l605_60584

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -11 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l605_60584


namespace NUMINAMATH_CALUDE_custom_mult_example_l605_60503

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := (a + b) / (1 - b)

/-- Theorem stating that (5 * 4) * 2 = 1 using the custom multiplication -/
theorem custom_mult_example : custom_mult (custom_mult 5 4) 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l605_60503


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l605_60583

/-- Proves that the mean profit for the first 15 days is 285 Rs, given the conditions of the problem. -/
theorem shop_profit_calculation (total_days : ℕ) (mean_profit : ℚ) (last_half_mean : ℚ) :
  total_days = 30 →
  mean_profit = 350 →
  last_half_mean = 415 →
  (total_days * mean_profit - (total_days / 2) * last_half_mean) / (total_days / 2) = 285 := by
  sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l605_60583


namespace NUMINAMATH_CALUDE_smallest_valid_digit_set_l605_60581

def isRepresentable (digits : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ digits ∨ ∃ a b, a ∈ digits ∧ b ∈ digits ∧ a + b = n

def isValidDigitSet (digits : Finset ℕ) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 99999999 → isRepresentable digits n

theorem smallest_valid_digit_set :
  ∃ digits : Finset ℕ, 
    isValidDigitSet digits ∧ 
    digits.card = 5 ∧ 
    ∀ otherDigits : Finset ℕ, isValidDigitSet otherDigits → otherDigits.card ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_digit_set_l605_60581


namespace NUMINAMATH_CALUDE_quadratic_solution_system_solution_l605_60559

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 50 = 0

-- Define the system of equations
def system_eq (x y : ℝ) : Prop := x = 2*y + 7 ∧ 2*x + 5*y = -4

-- Theorem for the quadratic equation
theorem quadratic_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, quadratic_eq x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

-- Theorem for the system of equations
theorem system_solution : 
  ∃! x y : ℝ, system_eq x y ∧ x = 3 ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_system_solution_l605_60559


namespace NUMINAMATH_CALUDE_line_equation_l605_60598

-- Define the line l
def Line (l : ℝ → ℝ) : Prop :=
  ∃ m b, ∀ x, l x = m * x + b

-- Define the y-intercept
def YIntercept (l : ℝ → ℝ) (y : ℝ) : Prop :=
  l 0 = y

-- Define the segment length cut off by the coordinate axes
def SegmentLength (l : ℝ → ℝ) (length : ℝ) : Prop :=
  ∃ x, x > 0 ∧ l x = 0 ∧ x^2 + (l 0)^2 = length^2

-- Theorem statement
theorem line_equation (l : ℝ → ℝ) :
  Line l →
  YIntercept l (-3) →
  SegmentLength l 5 →
  (∀ x y, 3*x - 4*y - 12 = 0 ↔ y = l x) ∨
  (∀ x y, 3*x + 4*y + 12 = 0 ↔ y = l x) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l605_60598


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l605_60560

theorem min_value_sum_squares (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a - 1)^3 + (b - 1)^3 ≥ 3*(2 - a - b)) : 
  a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l605_60560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_squared_l605_60568

theorem arithmetic_sequence_sum_squared (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  let seq := List.range n |>.map (fun i => a₁ + i * d)
  3 * (seq.sum)^2 = 1520832 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_squared_l605_60568


namespace NUMINAMATH_CALUDE_divisibility_by_101_l605_60509

theorem divisibility_by_101 (n : ℕ+) :
  (∃ k : ℕ+, n = k * 101 - 1) ↔
  (101 ∣ n^3 + 1) ∧ (101 ∣ n^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l605_60509


namespace NUMINAMATH_CALUDE_inverse_99_mod_101_l605_60543

theorem inverse_99_mod_101 : ∃ x : ℕ, x ∈ Finset.range 101 ∧ (99 * x) % 101 = 1 := by
  use 51
  sorry

end NUMINAMATH_CALUDE_inverse_99_mod_101_l605_60543


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l605_60536

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x - floor x = 1 / (floor x : ℝ)

-- Define a function to check if a real number is a solution
def is_solution (x : ℝ) : Prop := equation x ∧ x > 0

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s1 s2 s3 : ℝ),
    is_solution s1 ∧ is_solution s2 ∧ is_solution s3 ∧
    (∀ (x : ℝ), is_solution x → x ≥ s1) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 → x ≥ s2) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 ∧ x ≠ s2 → x ≥ s3) ∧
    s1 + s2 + s3 = 10 + 1/12 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l605_60536


namespace NUMINAMATH_CALUDE_max_a_value_l605_60507

/-- The quadratic polynomial p(x) -/
def p (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - (a - 1) * x + 2022

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x ∈ Set.Icc 0 1, -2022 ≤ p a x ∧ p a x ≤ 2022) → 
  a ≤ 16177 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l605_60507


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l605_60500

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24,
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768 -/
theorem equilateral_triangle_sum_product (a b c : ℂ) : 
  (∃ (ω : ℂ), ω ^ 3 = 1 ∧ ω ≠ 1 ∧ c - a = (b - a) * ω) →
  Complex.abs (b - a) = 24 →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l605_60500


namespace NUMINAMATH_CALUDE_zero_of_composite_f_l605_60504

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

-- State the theorem
theorem zero_of_composite_f :
  ∃ (x : ℝ), f (f x) = 0 ∧ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_composite_f_l605_60504
