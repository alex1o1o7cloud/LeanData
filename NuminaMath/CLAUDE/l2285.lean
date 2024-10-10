import Mathlib

namespace cube_sum_minus_product_eq_2003_l2285_228535

/-- The equation x^3 + y^3 + z^3 - 3xyz = 2003 has only three integer solutions. -/
theorem cube_sum_minus_product_eq_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(667, 668, 668), (668, 667, 668), (668, 668, 667)} := by
  sorry

#check cube_sum_minus_product_eq_2003

end cube_sum_minus_product_eq_2003_l2285_228535


namespace sqrt_expression_equality_l2285_228536

theorem sqrt_expression_equality : Real.sqrt 3 * Real.sqrt 2 - Real.sqrt 2 + Real.sqrt 8 = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end sqrt_expression_equality_l2285_228536


namespace a_eq_one_sufficient_not_necessary_l2285_228599

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + (a - 1) * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y - 3 = 0

/-- Definition of perpendicular lines -/
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ 
  a * (a - 1) + (a - 1) * (2 * a + 3) = 0

/-- Theorem stating that a = 1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary : 
  (∀ a : ℝ, a = 1 → perpendicular a) ∧ 
  ¬(∀ a : ℝ, perpendicular a → a = 1) := by sorry

end a_eq_one_sufficient_not_necessary_l2285_228599


namespace not_divisible_by_11_l2285_228545

theorem not_divisible_by_11 : ¬(11 ∣ 98473092) := by
  sorry

end not_divisible_by_11_l2285_228545


namespace mother_carrots_count_l2285_228505

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

/-- The number of carrots Haley's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - haley_carrots

theorem mother_carrots_count : mother_carrots = 38 := by
  sorry

end mother_carrots_count_l2285_228505


namespace chord_length_l2285_228597

/-- The length of the chord cut by a line on a circle in polar coordinates -/
theorem chord_length (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 := by
sorry


end chord_length_l2285_228597


namespace intersection_set_complement_l2285_228594

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem intersection_set_complement : A ∩ (𝒰 \ B) = {x | -1 ≤ x ∧ x ≤ 0} := by
  sorry

end intersection_set_complement_l2285_228594


namespace multiply_52_48_l2285_228580

theorem multiply_52_48 : 52 * 48 = 2496 := by
  sorry

end multiply_52_48_l2285_228580


namespace division_problem_l2285_228517

theorem division_problem : (96 : ℚ) / ((8 : ℚ) / 4) = 48 := by sorry

end division_problem_l2285_228517


namespace y_divisibility_l2285_228573

def y : ℕ := 58 + 104 + 142 + 184 + 304 + 368 + 3304

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 4 * k) ∧
  ¬(∀ k : ℕ, y = 8 * k) ∧
  ¬(∀ k : ℕ, y = 16 * k) := by
  sorry

end y_divisibility_l2285_228573


namespace compound_interest_rate_is_ten_percent_l2285_228593

/-- Given the conditions of the problem, prove that the compound interest rate is 10% --/
theorem compound_interest_rate_is_ten_percent
  (simple_principal : ℝ)
  (simple_rate : ℝ)
  (simple_time : ℝ)
  (compound_principal : ℝ)
  (compound_time : ℝ)
  (h1 : simple_principal = 1750.0000000000018)
  (h2 : simple_rate = 8)
  (h3 : simple_time = 3)
  (h4 : compound_principal = 4000)
  (h5 : compound_time = 2)
  (h6 : simple_principal * simple_rate * simple_time / 100 = 
        compound_principal * ((1 + compound_rate / 100) ^ compound_time - 1) / 2)
  : compound_rate = 10 := by
  sorry

end compound_interest_rate_is_ten_percent_l2285_228593


namespace least_number_with_remainders_l2285_228538

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 5 ∧
  n % 11 = 5 ∧
  n % 13 = 5 ∧
  n % 17 = 5 ∧
  n % 23 = 5 ∧
  n % 19 = 0 ∧
  ∀ m : ℕ, m > 0 →
    m % 7 = 5 →
    m % 11 = 5 →
    m % 13 = 5 →
    m % 17 = 5 →
    m % 23 = 5 →
    m % 19 = 0 →
    n ≤ m :=
by sorry

end least_number_with_remainders_l2285_228538


namespace chess_tournament_max_matches_l2285_228567

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  max_matches_per_pair : Nat
  no_triples : Bool

/-- The maximum number of matches any participant can play. -/
def max_matches_per_participant (t : ChessTournament) : Nat :=
  sorry

/-- The theorem stating the maximum number of matches per participant
    in a chess tournament with the given conditions. -/
theorem chess_tournament_max_matches
  (t : ChessTournament)
  (h1 : t.participants = 300)
  (h2 : t.max_matches_per_pair = 1)
  (h3 : t.no_triples = true) :
  max_matches_per_participant t = 200 :=
sorry

end chess_tournament_max_matches_l2285_228567


namespace faster_person_speed_l2285_228566

/-- Given two towns 45 km apart and two people traveling towards each other,
    where one person travels 1 km/h faster than the other and they meet after 5 hours,
    prove that the faster person's speed is 5 km/h. -/
theorem faster_person_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) :
  distance = 45 →
  time = 5 →
  speed_diff = 1 →
  ∃ (speed_slower : ℝ),
    speed_slower > 0 ∧
    speed_slower * time + (speed_slower + speed_diff) * time = distance ∧
    speed_slower + speed_diff = 5 := by
  sorry

end faster_person_speed_l2285_228566


namespace picture_area_l2285_228578

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * x * y + 9 * x + 4 * y = 42) : x * y = 6 := by
  sorry

end picture_area_l2285_228578


namespace ice_pop_cost_l2285_228519

theorem ice_pop_cost (ice_pop_price : ℝ) (pencil_price : ℝ) (ice_pops_sold : ℕ) (pencils_bought : ℕ) : 
  ice_pop_price = 1.50 →
  pencil_price = 1.80 →
  ice_pops_sold = 300 →
  pencils_bought = 100 →
  ice_pops_sold * ice_pop_price = pencils_bought * pencil_price →
  ice_pop_price - (ice_pops_sold * ice_pop_price - pencils_bought * pencil_price) / ice_pops_sold = 0.90 := by
sorry

end ice_pop_cost_l2285_228519


namespace contrapositive_equivalence_l2285_228515

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x ≠ 1 ∧ x ≠ 2) ↔ x^2 - 3*x + 2 = 0) ↔
  (x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2) :=
sorry

end contrapositive_equivalence_l2285_228515


namespace tournament_ceremony_theorem_l2285_228503

def tournament_ceremony_length (initial_players : Nat) (initial_ceremony_length : Nat) (ceremony_increase : Nat) : Nat :=
  let rounds := Nat.log2 initial_players
  let ceremony_lengths := List.range rounds |>.map (λ i => initial_ceremony_length + i * ceremony_increase)
  let winners_per_round := List.range rounds |>.map (λ i => initial_players / (2^(i+1)))
  List.sum (List.zipWith (·*·) ceremony_lengths winners_per_round)

theorem tournament_ceremony_theorem :
  tournament_ceremony_length 16 10 10 = 260 := by
  sorry

end tournament_ceremony_theorem_l2285_228503


namespace sin_2theta_value_l2285_228514

theorem sin_2theta_value (θ : Real) (h : (Real.sqrt 2 * Real.cos (2 * θ)) / Real.cos (π / 4 + θ) = Real.sqrt 3 * Real.sin (2 * θ)) : 
  Real.sin (2 * θ) = -2/3 := by
  sorry

end sin_2theta_value_l2285_228514


namespace square_area_error_l2285_228511

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end square_area_error_l2285_228511


namespace chocolate_count_l2285_228518

/-- Represents the total number of chocolates in the jar -/
def total_chocolates : ℕ := 50

/-- Represents the number of chocolates that are not hazelnut -/
def not_hazelnut : ℕ := 12

/-- Represents the number of chocolates that are not liquor -/
def not_liquor : ℕ := 18

/-- Represents the number of chocolates that are not milk -/
def not_milk : ℕ := 20

/-- Theorem stating that the total number of chocolates is 50 -/
theorem chocolate_count :
  total_chocolates = 50 ∧
  not_hazelnut = 12 ∧
  not_liquor = 18 ∧
  not_milk = 20 ∧
  (total_chocolates - not_hazelnut) + (total_chocolates - not_liquor) + (total_chocolates - not_milk) = 2 * total_chocolates :=
by sorry

end chocolate_count_l2285_228518


namespace definite_integral_evaluation_l2285_228501

theorem definite_integral_evaluation :
  ∫ x in (1 : ℝ)..3, (2 * x - 1 / (x^2)) = 22 / 3 := by
  sorry

end definite_integral_evaluation_l2285_228501


namespace scenario_one_scenario_two_scenario_three_scenario_four_l2285_228544

-- Define the probabilities for A and B
def prob_A : ℚ := 2/3
def prob_B : ℚ := 3/4

-- Define the complementary probabilities
def miss_A : ℚ := 1 - prob_A
def miss_B : ℚ := 1 - prob_B

-- Theorem 1
theorem scenario_one : 
  1 - prob_A^3 = 19/27 := by sorry

-- Theorem 2
theorem scenario_two :
  2 * prob_A^2 * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

-- Theorem 3
theorem scenario_three :
  miss_A^2 * prob_B^2 = 1/16 := by sorry

-- Theorem 4
theorem scenario_four :
  2 * prob_A * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

end scenario_one_scenario_two_scenario_three_scenario_four_l2285_228544


namespace install_time_proof_l2285_228564

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install remaining windows is 48 hours -/
theorem install_time_proof (total : ℕ) (installed : ℕ) (time_per_window : ℕ) 
  (h1 : total = 14)
  (h2 : installed = 8)
  (h3 : time_per_window = 8) :
  time_to_install_remaining total installed time_per_window = 48 := by
  sorry

#eval time_to_install_remaining 14 8 8

end install_time_proof_l2285_228564


namespace triangle_perimeter_for_radius_3_l2285_228561

/-- A configuration of three circles and a triangle -/
structure CircleTriangleConfig where
  /-- The radius of each circle -/
  radius : ℝ
  /-- The circles are externally tangent to each other -/
  circles_tangent : Prop
  /-- Each side of the triangle is tangent to two of the circles -/
  triangle_tangent : Prop

/-- The perimeter of the triangle in the given configuration -/
def triangle_perimeter (config : CircleTriangleConfig) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the triangle for the given configuration -/
theorem triangle_perimeter_for_radius_3 :
  ∀ (config : CircleTriangleConfig),
    config.radius = 3 →
    config.circles_tangent →
    config.triangle_tangent →
    triangle_perimeter config = 18 + 18 * Real.sqrt 3 :=
by
  sorry

end triangle_perimeter_for_radius_3_l2285_228561


namespace tangency_condition_l2285_228500

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 4

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x', y') = (x, y)

/-- The theorem statement -/
theorem tangency_condition (m : ℝ) :
  are_tangent m ↔ m = 8 + 4 * Real.sqrt 3 ∨ m = 8 - 4 * Real.sqrt 3 := by sorry

end tangency_condition_l2285_228500


namespace unique_solution_absolute_value_equation_l2285_228588

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 9| = |x + 3| + 2 :=
by
  -- The proof goes here
  sorry

end unique_solution_absolute_value_equation_l2285_228588


namespace third_beats_seventh_l2285_228543

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  /-- List of player scores in descending order -/
  scores : List ℕ
  /-- Ensure there are exactly 8 scores -/
  score_count : scores.length = 8
  /-- Ensure all scores are different -/
  distinct_scores : scores.Nodup
  /-- Second place score equals sum of last four scores -/
  second_place_condition : scores[1]! = scores[4]! + scores[5]! + scores[6]! + scores[7]!

/-- Represents the result of a game between two players -/
inductive GameResult
  | Win
  | Loss

/-- Function to determine the game result between two players based on their positions -/
def gameResult (t : ChessTournament) (player1 : Fin 8) (player2 : Fin 8) : GameResult :=
  if player1 < player2 then GameResult.Win else GameResult.Loss

theorem third_beats_seventh (t : ChessTournament) :
  gameResult t 2 6 = GameResult.Win :=
sorry

end third_beats_seventh_l2285_228543


namespace quadratic_root_zero_l2285_228584

/-- Given a quadratic equation (m-2)x^2 + 3x - m^2 - m + 6 = 0 where one root is 0,
    prove that m = -3 is the only valid solution. -/
theorem quadratic_root_zero (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x - m^2 - m + 6 = 0 ↔ x = 0 ∨ x = (m^2 + m - 6) / (2 * m - 4)) →
  m - 2 ≠ 0 →
  m = -3 := by
sorry

end quadratic_root_zero_l2285_228584


namespace hall_length_is_18_l2285_228550

/-- Represents the dimensions of a rectangular hall -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the hall dimensions satisfy the given conditions -/
def satisfiesConditions (d : HallDimensions) : Prop :=
  d.width = 9 ∧
  2 * (d.length * d.width) = 2 * (d.length * d.height + d.width * d.height) ∧
  d.length * d.width * d.height = 972

theorem hall_length_is_18 :
  ∃ (d : HallDimensions), satisfiesConditions d ∧ d.length = 18 :=
by sorry

end hall_length_is_18_l2285_228550


namespace fourth_grade_students_at_end_l2285_228579

/-- Calculates the number of students remaining in fourth grade at the end of the year. -/
def studentsAtEnd (initial : Float) (left : Float) (transferred : Float) : Float :=
  initial - left - transferred

/-- Theorem stating that the number of students at the end of the year is 28.0 -/
theorem fourth_grade_students_at_end :
  studentsAtEnd 42.0 4.0 10.0 = 28.0 := by
  sorry

end fourth_grade_students_at_end_l2285_228579


namespace triangle_sides_l2285_228532

theorem triangle_sides (average_length : ℝ) (perimeter : ℝ) (n : ℕ) :
  average_length = 12 →
  perimeter = 36 →
  average_length * n = perimeter →
  n = 3 := by
  sorry

end triangle_sides_l2285_228532


namespace inequality_proof_l2285_228516

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) ≥ 3/2) ∧ 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) = 3/2 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_proof_l2285_228516


namespace system_solution_l2285_228592

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end system_solution_l2285_228592


namespace amoeba_population_after_five_days_l2285_228565

/-- The number of amoebas after n days, given an initial population and daily split rate --/
def amoeba_population (initial_population : ℕ) (split_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * split_rate ^ days

/-- The theorem stating that after 5 days, the amoeba population will be 486 --/
theorem amoeba_population_after_five_days :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end amoeba_population_after_five_days_l2285_228565


namespace parabola_translation_existence_l2285_228558

theorem parabola_translation_existence : ∃ (h k : ℝ),
  (0 = -(0 - h)^2 + k) ∧  -- passes through origin
  ((1/2) * (2*h) * k = 1) ∧  -- triangle area is 1
  (h^2 = k) ∧  -- vertex is (h, k)
  (h = 1 ∨ h = -1) ∧
  (k = 1) :=
by sorry

end parabola_translation_existence_l2285_228558


namespace marching_band_members_l2285_228542

theorem marching_band_members : ∃ n : ℕ,
  150 < n ∧ n < 250 ∧
  n % 3 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  (∀ m : ℕ, 150 < m ∧ m < n →
    ¬(m % 3 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3)) ∧
  n = 203 :=
by sorry

end marching_band_members_l2285_228542


namespace sqrt_sum_inequality_l2285_228555

theorem sqrt_sum_inequality (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end sqrt_sum_inequality_l2285_228555


namespace specific_path_count_l2285_228529

/-- The number of paths on a grid with given dimensions and constraints -/
def numPaths (width height diagonalSteps : ℕ) : ℕ :=
  Nat.choose (width + height - diagonalSteps) diagonalSteps *
  Nat.choose (width + height - 2 * diagonalSteps) height

/-- Theorem stating the number of paths for the specific problem -/
theorem specific_path_count :
  numPaths 7 6 2 = 6930 := by
  sorry

end specific_path_count_l2285_228529


namespace arithmetic_sequence_problem_l2285_228537

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 12 + a 13 = 24) : 
  a 7 = 6 := by sorry

end arithmetic_sequence_problem_l2285_228537


namespace fred_seashells_l2285_228549

/-- The number of seashells Fred found initially -/
def initial_seashells : ℝ := 47.5

/-- The number of seashells Fred gave to Jessica -/
def given_seashells : ℝ := 25.3

/-- The number of seashells Fred has now -/
def remaining_seashells : ℝ := initial_seashells - given_seashells

theorem fred_seashells : remaining_seashells = 22.2 := by sorry

end fred_seashells_l2285_228549


namespace smallest_positive_solution_of_equation_l2285_228554

theorem smallest_positive_solution_of_equation :
  ∃ (x : ℝ), x > 0 ∧ x^4 - 58*x^2 + 841 = 0 ∧ ∀ (y : ℝ), y > 0 ∧ y^4 - 58*y^2 + 841 = 0 → x ≤ y ∧ x = Real.sqrt 29 := by
  sorry

end smallest_positive_solution_of_equation_l2285_228554


namespace quadratic_roots_sum_l2285_228508

theorem quadratic_roots_sum (p : ℝ) : 
  (∃ x y : ℝ, x * y = 9 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) →
  (∃ x y : ℝ, x + y = 7 ∧ 2 * x^2 + p * x - p + 4 = 0 ∧ 2 * y^2 + p * y - p + 4 = 0) :=
by sorry

end quadratic_roots_sum_l2285_228508


namespace equation_solution_l2285_228510

theorem equation_solution : 
  ∃ x : ℝ, (3 * x^2 + 6 = |(-25 + x)|) ∧ 
  (x = (-1 + Real.sqrt 229) / 6 ∨ x = (-1 - Real.sqrt 229) / 6) :=
by sorry

end equation_solution_l2285_228510


namespace six_digit_divisibility_by_seven_l2285_228530

theorem six_digit_divisibility_by_seven (a b c d e f : Nat) 
  (h1 : a ≥ 1 ∧ a ≤ 9)  -- Ensure it's a six-digit number
  (h2 : b ≥ 0 ∧ b ≤ 9)
  (h3 : c ≥ 0 ∧ c ≤ 9)
  (h4 : d ≥ 0 ∧ d ≤ 9)
  (h5 : e ≥ 0 ∧ e ≤ 9)
  (h6 : f ≥ 0 ∧ f ≤ 9)
  (h7 : (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≡ 0 [MOD 7]) :
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ≡ 0 [MOD 7] := by
  sorry

#check six_digit_divisibility_by_seven

end six_digit_divisibility_by_seven_l2285_228530


namespace steve_earnings_l2285_228563

/-- Calculates the amount of money an author keeps after selling books and paying an agent. -/
def authorEarnings (totalCopies : ℕ) (advanceCopies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let copiesForEarnings := totalCopies - advanceCopies
  let totalEarnings := copiesForEarnings * earningsPerCopy
  let agentCut := totalEarnings * agentPercentage
  totalEarnings - agentCut

/-- Proves that given the conditions of Steve's book sales, he keeps $1,620,000 after paying his agent. -/
theorem steve_earnings :
  authorEarnings 1000000 100000 2 (1/10) = 1620000 := by
  sorry

end steve_earnings_l2285_228563


namespace equation_solution_exists_l2285_228557

theorem equation_solution_exists : ∃ (a b c d e : ℕ), 
  a ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  (a + b - c) * d / e = 4 := by
  sorry

end equation_solution_exists_l2285_228557


namespace contrapositive_sine_not_piecewise_l2285_228577

-- Define the universe of functions
variable (F : Type) [Nonempty F]

-- Define predicates for sine function and piecewise function
variable (is_sine : F → Prop)
variable (is_piecewise : F → Prop)

-- State the theorem
theorem contrapositive_sine_not_piecewise :
  (∀ f : F, is_sine f → ¬ is_piecewise f) ↔
  (∀ f : F, is_piecewise f → ¬ is_sine f) :=
by sorry

end contrapositive_sine_not_piecewise_l2285_228577


namespace yogurt_combinations_l2285_228541

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 8 → 
  flavors * (toppings.choose 3) = 280 := by
  sorry

end yogurt_combinations_l2285_228541


namespace systematic_sampling_selection_l2285_228576

theorem systematic_sampling_selection (total_rooms : Nat) (sample_size : Nat) (first_room : Nat) : 
  total_rooms = 64 → 
  sample_size = 8 → 
  first_room = 5 → 
  ∃ k : Nat, k < sample_size ∧ (first_room + k * (total_rooms / sample_size)) % total_rooms = 53 :=
by
  sorry

end systematic_sampling_selection_l2285_228576


namespace min_group_size_l2285_228589

/-- Represents the number of men in a group with various attributes -/
structure MenGroup where
  total : ℕ
  married : ℕ
  hasTV : ℕ
  hasRadio : ℕ
  hasAC : ℕ
  hasAll : ℕ

/-- The minimum number of men in the group is at least the maximum of any single category -/
theorem min_group_size (g : MenGroup) 
  (h1 : g.married = 81)
  (h2 : g.hasTV = 75)
  (h3 : g.hasRadio = 85)
  (h4 : g.hasAC = 70)
  (h5 : g.hasAll = 11)
  : g.total ≥ 85 := by
  sorry

#check min_group_size

end min_group_size_l2285_228589


namespace tan_negative_two_implies_fraction_l2285_228512

theorem tan_negative_two_implies_fraction (θ : Real) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2/5 := by
  sorry

end tan_negative_two_implies_fraction_l2285_228512


namespace wilted_ratio_after_first_night_l2285_228551

/-- Represents the number of roses at different stages --/
structure RoseCount where
  initial : ℕ
  afterFirstNight : ℕ
  afterSecondNight : ℕ

/-- Calculates the ratio of wilted flowers to total flowers after the first night --/
def wiltedRatio (rc : RoseCount) : Rat :=
  (rc.initial - rc.afterFirstNight) / rc.initial

theorem wilted_ratio_after_first_night
  (rc : RoseCount)
  (h1 : rc.initial = 36)
  (h2 : rc.afterSecondNight = 9)
  (h3 : rc.afterFirstNight = 2 * rc.afterSecondNight) :
  wiltedRatio rc = 1/2 := by
  sorry

#eval wiltedRatio { initial := 36, afterFirstNight := 18, afterSecondNight := 9 }

end wilted_ratio_after_first_night_l2285_228551


namespace reading_time_calculation_l2285_228552

theorem reading_time_calculation (total_time math_time spelling_time history_time science_time piano_time break_time : ℕ)
  (h1 : total_time = 180)
  (h2 : math_time = 25)
  (h3 : spelling_time = 30)
  (h4 : history_time = 20)
  (h5 : science_time = 15)
  (h6 : piano_time = 30)
  (h7 : break_time = 20) :
  total_time - (math_time + spelling_time + history_time + science_time + piano_time + break_time) = 40 := by
  sorry

end reading_time_calculation_l2285_228552


namespace range_of_a_l2285_228568

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end range_of_a_l2285_228568


namespace pat_stickers_l2285_228524

theorem pat_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 := by
  sorry

end pat_stickers_l2285_228524


namespace guest_lecturer_fee_l2285_228596

theorem guest_lecturer_fee (B : Nat) (h1 : B < 10) (h2 : (200 + 10 * B + 9) % 13 = 0) : B = 0 := by
  sorry

end guest_lecturer_fee_l2285_228596


namespace four_numbers_product_sum_l2285_228583

theorem four_numbers_product_sum (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) ↔
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) := by
sorry


end four_numbers_product_sum_l2285_228583


namespace arrangement_count_four_objects_five_positions_l2285_228575

theorem arrangement_count : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * arrangement_count n

theorem four_objects_five_positions :
  arrangement_count 5 = 120 := by
  sorry

end arrangement_count_four_objects_five_positions_l2285_228575


namespace shaded_area_calculation_l2285_228504

theorem shaded_area_calculation (S T : ℝ) : 
  (16 / S = 4) → 
  (S / T = 4) → 
  (S^2 + 16 * T^2 = 32) := by
sorry

end shaded_area_calculation_l2285_228504


namespace new_device_improvement_l2285_228572

/-- Represents the data for a device's products -/
structure DeviceData where
  mean : ℝ
  variance : ℝ

/-- Criterion for significant improvement -/
def significant_improvement (old new : DeviceData) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement (old new : DeviceData)
  (h_old : old.mean = 10 ∧ old.variance = 0.036)
  (h_new : new.mean = 10.3 ∧ new.variance = 0.04) :
  significant_improvement old new :=
by sorry

end new_device_improvement_l2285_228572


namespace six_digit_numbers_with_zero_l2285_228522

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_l2285_228522


namespace prob_at_least_one_woman_l2285_228528

theorem prob_at_least_one_woman (men women selected : ℕ) :
  men = 9 →
  women = 5 →
  selected = 3 →
  (1 - (Nat.choose men selected) / (Nat.choose (men + women) selected) : ℚ) = 23/30 := by
  sorry

end prob_at_least_one_woman_l2285_228528


namespace min_value_theorem_l2285_228531

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) ∧
  4/a + 1/(b-1) = 9 :=
sorry

end min_value_theorem_l2285_228531


namespace four_squared_sum_equals_four_cubed_l2285_228569

theorem four_squared_sum_equals_four_cubed : 4^2 + 4^2 + 4^2 + 4^2 = 4^3 := by
  sorry

end four_squared_sum_equals_four_cubed_l2285_228569


namespace work_scaling_l2285_228527

/-- Given that 3 people can do 3 times of a particular work in 3 days,
    prove that 9 people can do 9 times of that particular work in 3 days. -/
theorem work_scaling (work_rate : ℕ → ℕ → ℝ → ℝ) :
  work_rate 3 3 3 = 1 →
  work_rate 9 9 3 = 1 := by
sorry

end work_scaling_l2285_228527


namespace problem_statement_l2285_228587

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, (a * x + b * y : ℕ) = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2 : ℕ)) ∧ (a = x ∧ b = y) := by
sorry


end problem_statement_l2285_228587


namespace height_calculations_l2285_228570

-- Define the conversion rate
def inch_to_cm : ℝ := 2.54

-- Define heights in inches
def maria_height_inches : ℝ := 54
def samuel_height_inches : ℝ := 72

-- Define function to convert inches to centimeters
def inches_to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

-- Theorem statement
theorem height_calculations :
  let maria_height_cm := inches_to_cm maria_height_inches
  let samuel_height_cm := inches_to_cm samuel_height_inches
  let height_difference := samuel_height_cm - maria_height_cm
  (maria_height_cm = 137.16) ∧
  (samuel_height_cm = 182.88) ∧
  (height_difference = 45.72) := by
  sorry

end height_calculations_l2285_228570


namespace shirts_not_washed_l2285_228556

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end shirts_not_washed_l2285_228556


namespace largest_divisible_n_l2285_228591

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 11) ∣ (n^4 + 119) ∧ 
  ∀ (m : ℕ), m > n → ¬((m + 11) ∣ (m^4 + 119)) := by
  sorry

end largest_divisible_n_l2285_228591


namespace line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l2285_228539

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (angle : Line → Line → ℝ)
variable (angle_line_plane : Line → Plane → ℝ)
variable (dihedral_angle : Plane → Plane → ℝ)

-- Theorem statements
theorem line_relationships (l1 l2 : Line) : 
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ intersect l1 l2) ∧ 
  ¬(parallel l1 l2 ∧ skew l1 l2) ∧ 
  ¬(intersect l1 l2 ∧ skew l1 l2) := sorry

theorem skew_lines_angle_range (l1 l2 : Line) (h : skew l1 l2) : 
  0 < angle l1 l2 ∧ angle l1 l2 ≤ Real.pi / 2 := sorry

theorem line_plane_angle_range (l : Line) (p : Plane) : 
  0 ≤ angle_line_plane l p ∧ angle_line_plane l p ≤ Real.pi / 2 := sorry

theorem dihedral_angle_range (p1 p2 : Plane) : 
  0 ≤ dihedral_angle p1 p2 ∧ dihedral_angle p1 p2 ≤ Real.pi := sorry

theorem parallel_line_plane (a b : Line) (α : Plane) 
  (h1 : ¬contained_in a α) (h2 : contained_in b α) (h3 : parallel a b) : 
  parallel_plane a α := sorry

theorem parallel_planes (a b : Line) (α β : Plane) (P : Point)
  (h1 : contained_in a β) (h2 : contained_in b β) 
  (h3 : intersect a b) (h4 : ¬parallel_plane a α) (h5 : ¬parallel_plane b α) : 
  planes_parallel α β := sorry

theorem perpendicular_line_plane (a b l : Line) (α : Plane) (A : Point)
  (h1 : contained_in a α) (h2 : contained_in b α) 
  (h3 : intersect a b) (h4 : perpendicular l a) (h5 : perpendicular l b) : 
  perpendicular_plane l α := sorry

theorem perpendicular_planes (l : Line) (α β : Plane)
  (h1 : perpendicular_plane l α) (h2 : contained_in l β) : 
  planes_perpendicular α β := sorry

end line_relationships_skew_lines_angle_range_line_plane_angle_range_dihedral_angle_range_parallel_line_plane_parallel_planes_perpendicular_line_plane_perpendicular_planes_l2285_228539


namespace desired_weather_probability_l2285_228547

def days : ℕ := 5
def p_sun : ℚ := 1/4
def p_rain : ℚ := 3/4

def probability_k_sunny_days (k : ℕ) : ℚ :=
  (Nat.choose days k) * (p_sun ^ k) * (p_rain ^ (days - k))

theorem desired_weather_probability : 
  probability_k_sunny_days 1 + probability_k_sunny_days 2 = 135/2048 := by
  sorry

end desired_weather_probability_l2285_228547


namespace box_probability_l2285_228574

theorem box_probability (a : ℕ) (h1 : a > 0) : 
  (4 : ℝ) / a = (1 : ℝ) / 5 → a = 20 := by
sorry

end box_probability_l2285_228574


namespace sam_gave_29_cards_l2285_228502

/-- The number of new Pokemon cards Sam gave to Mary -/
def new_cards (initial : ℕ) (torn : ℕ) (final : ℕ) : ℕ :=
  final - (initial - torn)

/-- Proof that Sam gave Mary 29 new Pokemon cards -/
theorem sam_gave_29_cards : new_cards 33 6 56 = 29 := by
  sorry

end sam_gave_29_cards_l2285_228502


namespace complex_fourth_root_of_negative_sixteen_l2285_228548

theorem complex_fourth_root_of_negative_sixteen :
  let solutions := {z : ℂ | z^4 = -16 ∧ z.im ≥ 0}
  solutions = {Complex.mk (Real.sqrt 2) (Real.sqrt 2), Complex.mk (-Real.sqrt 2) (-Real.sqrt 2)} := by
  sorry

end complex_fourth_root_of_negative_sixteen_l2285_228548


namespace corresponds_to_zero_one_l2285_228546

/-- A mapping f from A to B where (x, y) in A corresponds to (x-1, 3-y) in B -/
def f (x y : ℝ) : ℝ × ℝ := (x - 1, 3 - y)

/-- Theorem stating that (1, 2) in A corresponds to (0, 1) in B under mapping f -/
theorem corresponds_to_zero_one : f 1 2 = (0, 1) := by
  sorry

end corresponds_to_zero_one_l2285_228546


namespace max_value_abc_l2285_228560

theorem max_value_abc (a b c : ℝ) (h : a^2 + b^2/4 + c^2/9 = 1) :
  a + b + c ≤ Real.sqrt 14 :=
by sorry

end max_value_abc_l2285_228560


namespace triangle_lines_l2285_228559

structure Triangle where
  B : ℝ × ℝ
  altitude_AB : ℝ → ℝ → ℝ
  angle_bisector_A : ℝ → ℝ → ℝ

def line_AB (t : Triangle) : ℝ → ℝ → ℝ := fun x y => 2*x + y - 8
def line_AC (t : Triangle) : ℝ → ℝ → ℝ := fun x y => x + 2*y + 2

theorem triangle_lines (t : Triangle) 
  (hB : t.B = (3, 2))
  (hAlt : t.altitude_AB = fun x y => x - 2*y + 2)
  (hBis : t.angle_bisector_A = fun x y => x + y - 2) :
  (line_AB t = fun x y => 2*x + y - 8) ∧ 
  (line_AC t = fun x y => x + 2*y + 2) := by
  sorry

end triangle_lines_l2285_228559


namespace quadratic_inequality_solution_set_l2285_228571

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Icc (-1/2 : ℝ) (-1/3) = {x | a * x^2 - b * x - 1 ≥ 0}) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end quadratic_inequality_solution_set_l2285_228571


namespace average_salary_feb_to_may_l2285_228533

def average_salary_jan_to_apr : ℕ := 8000
def salary_jan : ℕ := 6100
def salary_may : ℕ := 6500
def target_average : ℕ := 8100

theorem average_salary_feb_to_may :
  (4 * average_salary_jan_to_apr - salary_jan + salary_may) / 4 = target_average :=
sorry

end average_salary_feb_to_may_l2285_228533


namespace dusty_cake_purchase_l2285_228507

/-- The number of double layer cake slices Dusty bought -/
def double_layer_slices : ℕ := sorry

/-- The price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- The price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- The number of single layer cake slices Dusty bought -/
def single_layer_bought : ℕ := 7

/-- The amount Dusty paid in dollars -/
def amount_paid : ℕ := 100

/-- The change Dusty received in dollars -/
def change_received : ℕ := 37

theorem dusty_cake_purchase : 
  double_layer_slices = 5 ∧
  amount_paid = 
    single_layer_price * single_layer_bought + 
    double_layer_price * double_layer_slices + 
    change_received :=
by sorry

end dusty_cake_purchase_l2285_228507


namespace unique_three_digit_number_exists_l2285_228523

/-- Represents a 3-digit number abc as 100a + 10b + c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the number acb obtained by swapping the last two digits of abc -/
def swap_last_two_digits (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem unique_three_digit_number_exists :
  ∃! (a b c : ℕ),
    (100 ≤ three_digit_number a b c) ∧
    (three_digit_number a b c ≤ 999) ∧
    (1730 ≤ three_digit_number a b c + swap_last_two_digits a b c) ∧
    (three_digit_number a b c + swap_last_two_digits a b c ≤ 1739) ∧
    (three_digit_number a b c = 832) :=
by sorry

end unique_three_digit_number_exists_l2285_228523


namespace parabola_tangent_hyperbola_l2285_228509

/-- The value of m for which the parabola y = x^2 + 4 is tangent to the hyperbola y^2 - mx^2 = 4 -/
def tangency_value : ℝ := 8

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 4

/-- Theorem stating that the parabola is tangent to the hyperbola if and only if m = 8 -/
theorem parabola_tangent_hyperbola :
  ∀ (m : ℝ), (∃ (x : ℝ), hyperbola m x (parabola x) ∧
    ∀ (x' : ℝ), x' ≠ x → ¬(hyperbola m x' (parabola x'))) ↔ m = tangency_value :=
sorry

end parabola_tangent_hyperbola_l2285_228509


namespace base_8_4512_equals_2378_l2285_228534

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 : 
  base_8_to_10 [2, 1, 5, 4] = 2378 := by sorry

end base_8_4512_equals_2378_l2285_228534


namespace mini_quiz_true_false_count_l2285_228585

/-- The number of true-false questions in the mini-quiz. -/
def n : ℕ := 3

/-- The number of multiple-choice questions. -/
def m : ℕ := 2

/-- The number of answer choices for each multiple-choice question. -/
def k : ℕ := 4

/-- The total number of ways to write the answer key. -/
def total_ways : ℕ := 96

theorem mini_quiz_true_false_count :
  (2^n - 2) * k^m = total_ways ∧ n > 0 := by sorry

end mini_quiz_true_false_count_l2285_228585


namespace sally_bread_consumption_l2285_228595

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of bread pieces used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of bread pieces Sally eats across Saturday and Sunday -/
def total_bread_pieces : ℕ := (saturday_sandwiches * bread_per_sandwich) + (sunday_sandwiches * bread_per_sandwich)

theorem sally_bread_consumption :
  total_bread_pieces = 6 := by
  sorry

end sally_bread_consumption_l2285_228595


namespace missing_digit_is_seven_l2285_228590

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem missing_digit_is_seven :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_9 (365000 + d * 100 + 42) ∧ d = 7 :=
by sorry

end missing_digit_is_seven_l2285_228590


namespace square_perimeter_from_area_l2285_228525

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 → 
  area = side * side → 
  perimeter = 4 * side → 
  perimeter = 48 := by
  sorry

end square_perimeter_from_area_l2285_228525


namespace volleyball_team_scoring_l2285_228598

/-- Volleyball team scoring problem -/
theorem volleyball_team_scoring 
  (lizzie_score : ℕ) 
  (nathalie_score : ℕ) 
  (aimee_score : ℕ) 
  (team_score : ℕ) 
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score = lizzie_score + 3)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : team_score = 50) :
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
  sorry


end volleyball_team_scoring_l2285_228598


namespace debby_water_bottles_l2285_228526

/-- Given the initial number of water bottles, daily consumption, and remaining bottles,
    calculate the number of days Debby drank water bottles. -/
theorem debby_water_bottles (initial : ℕ) (daily : ℕ) (remaining : ℕ) :
  initial = 264 →
  daily = 15 →
  remaining = 99 →
  (initial - remaining) / daily = 11 := by
  sorry

end debby_water_bottles_l2285_228526


namespace inscribed_circle_triangle_l2285_228586

theorem inscribed_circle_triangle (r : ℝ) (a b c : ℝ) :
  r = 3 →
  a + b = 7 →
  a = 3 →
  b = 4 →
  c^2 = a^2 + b^2 →
  (a + r)^2 + (b + r)^2 = c^2 →
  (a, b, c) = (3, 4, 5) ∨ (a, b, c) = (4, 3, 5) :=
by sorry

end inscribed_circle_triangle_l2285_228586


namespace tangent_inequality_tan_pi_12_l2285_228521

theorem tangent_inequality (a : Fin 13 → ℝ) (h : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ 0 < (a i - a j) / (1 + a i * a j) ∧ 
    (a i - a j) / (1 + a i * a j) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

theorem tan_pi_12 : Real.tan (π / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

end tangent_inequality_tan_pi_12_l2285_228521


namespace no_rational_solution_for_odd_coefficient_quadratic_l2285_228582

theorem no_rational_solution_for_odd_coefficient_quadratic
  (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end no_rational_solution_for_odd_coefficient_quadratic_l2285_228582


namespace ebook_count_l2285_228562

def total_ebooks (anna_bought : ℕ) (john_diff : ℕ) (john_lost : ℕ) (mary_factor : ℕ) (mary_gave : ℕ) : ℕ :=
  let john_bought := anna_bought - john_diff
  let john_has := john_bought - john_lost
  let mary_bought := mary_factor * john_bought
  let mary_has := mary_bought - mary_gave
  anna_bought + john_has + mary_has

theorem ebook_count :
  total_ebooks 50 15 3 2 7 = 145 := by
  sorry

end ebook_count_l2285_228562


namespace circle_arcs_angle_sum_l2285_228540

theorem circle_arcs_angle_sum (n : ℕ) (x y : ℝ) : 
  n = 18 → 
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end circle_arcs_angle_sum_l2285_228540


namespace even_function_monotonicity_l2285_228520

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define monotonically decreasing in an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

-- Define monotonically increasing in an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Theorem statement
theorem even_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_decreasing : monotone_decreasing_on f (-2) (-1)) :
  monotone_increasing_on f 1 2 ∧ 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f 1 ≤ f x) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ f 2) :=
by sorry

end even_function_monotonicity_l2285_228520


namespace parallelogram_area_l2285_228581

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 9 inches and 12 inches is 54√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 54 * Real.sqrt 3 := by sorry

end parallelogram_area_l2285_228581


namespace cost_price_calculation_l2285_228513

/-- Proves that the cost price is 1250 given the markup percentage and selling price -/
theorem cost_price_calculation (markup_percentage : ℝ) (selling_price : ℝ) : 
  markup_percentage = 60 →
  selling_price = 2000 →
  (100 + markup_percentage) / 100 * (selling_price / ((100 + markup_percentage) / 100)) = 1250 := by
  sorry

end cost_price_calculation_l2285_228513


namespace intersection_contains_two_elements_l2285_228553

-- Define the sets P and Q
def P (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k * (x - 1) + 1}
def Q : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 - 2*y = 0}

-- Theorem statement
theorem intersection_contains_two_elements :
  ∃ (k : ℝ), ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ P k ∩ Q ∧ b ∈ P k ∩ Q ∧
  ∀ (c : ℝ × ℝ), c ∈ P k ∩ Q → c = a ∨ c = b :=
sorry

end intersection_contains_two_elements_l2285_228553


namespace expenditure_ratio_l2285_228506

/-- Given a person's income and savings pattern over two years, prove the ratio of total expenditure to first year expenditure --/
theorem expenditure_ratio (income : ℝ) (h1 : income > 0) : 
  let first_year_savings := 0.25 * income
  let first_year_expenditure := income - first_year_savings
  let second_year_income := 1.25 * income
  let second_year_savings := 2 * first_year_savings
  let second_year_expenditure := second_year_income - second_year_savings
  let total_expenditure := first_year_expenditure + second_year_expenditure
  (total_expenditure / first_year_expenditure) = 2 := by
  sorry


end expenditure_ratio_l2285_228506
