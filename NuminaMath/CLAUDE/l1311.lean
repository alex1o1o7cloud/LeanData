import Mathlib

namespace NUMINAMATH_CALUDE_log_ratio_squared_l1311_131105

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (hprod : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1311_131105


namespace NUMINAMATH_CALUDE_kim_shirts_fraction_l1311_131176

theorem kim_shirts_fraction (initial_shirts : ℕ) (remaining_shirts : ℕ) :
  initial_shirts = 4 * 12 →
  remaining_shirts = 32 →
  (initial_shirts - remaining_shirts : ℚ) / initial_shirts = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_fraction_l1311_131176


namespace NUMINAMATH_CALUDE_batman_game_cost_l1311_131164

def football_cost : ℚ := 14.02
def strategy_cost : ℚ := 9.46
def total_spent : ℚ := 35.52

theorem batman_game_cost :
  ∃ (batman_cost : ℚ),
    batman_cost = total_spent - football_cost - strategy_cost ∧
    batman_cost = 12.04 :=
by sorry

end NUMINAMATH_CALUDE_batman_game_cost_l1311_131164


namespace NUMINAMATH_CALUDE_faster_increase_l1311_131119

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- State the theorem
theorem faster_increase : 
  (∀ x : ℝ, (deriv y₁ x) = (deriv y₂ x)) ∧ 
  (∀ x : ℝ, (deriv y₁ x) > (deriv y₃ x)) := by
  sorry

end NUMINAMATH_CALUDE_faster_increase_l1311_131119


namespace NUMINAMATH_CALUDE_expression_simplification_l1311_131146

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 - y^2) / (x*y) - (x^2*y - y^3) / (x^2*y - x*y^2) = (x^2 - x*y - 2*y^2) / (x*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1311_131146


namespace NUMINAMATH_CALUDE_negation_equivalence_l1311_131156

theorem negation_equivalence :
  (¬ ∃ x : ℕ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, Real.exp x - x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1311_131156


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1311_131180

/-- The constant term in the expansion of (x^2 + a/sqrt(x))^5 -/
def constantTerm (a : ℝ) : ℝ := 5 * a^4

theorem constant_term_expansion (a : ℝ) (h1 : a > 0) (h2 : constantTerm a = 80) : a = 2 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l1311_131180


namespace NUMINAMATH_CALUDE_no_contradiction_in_inequality_l1311_131193

theorem no_contradiction_in_inequality (a b c : ℝ) (h1 : a > b) (h2 : a = b + c) :
  a * (a - b - c) = b * (a - b - c) → ¬ (a = b) :=
by sorry

end NUMINAMATH_CALUDE_no_contradiction_in_inequality_l1311_131193


namespace NUMINAMATH_CALUDE_train_passes_jogger_l1311_131136

/-- Proves that a train passes a jogger in 40 seconds given specific conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 280 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passes_jogger

end NUMINAMATH_CALUDE_train_passes_jogger_l1311_131136


namespace NUMINAMATH_CALUDE_total_peaches_is_twelve_l1311_131172

/-- The number of baskets -/
def num_baskets : ℕ := 2

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 4

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of peaches in all baskets -/
def total_peaches : ℕ := num_baskets * (red_peaches_per_basket + green_peaches_per_basket)

theorem total_peaches_is_twelve : total_peaches = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_is_twelve_l1311_131172


namespace NUMINAMATH_CALUDE_fence_sheets_count_l1311_131127

/-- Represents the number of fence panels in the fence. -/
def num_panels : ℕ := 10

/-- Represents the number of metal beams in each fence panel. -/
def beams_per_panel : ℕ := 2

/-- Represents the number of metal rods in each sheet. -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam. -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence. -/
def total_rods : ℕ := 380

/-- Calculates the number of metal sheets in each fence panel. -/
def sheets_per_panel : ℕ :=
  let total_rods_per_panel := total_rods / num_panels
  let rods_for_beams := beams_per_panel * rods_per_beam
  (total_rods_per_panel - rods_for_beams) / rods_per_sheet

theorem fence_sheets_count : sheets_per_panel = 3 := by
  sorry

end NUMINAMATH_CALUDE_fence_sheets_count_l1311_131127


namespace NUMINAMATH_CALUDE_leadership_combinations_l1311_131114

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem leadership_combinations : 
  (tribe_size) * 
  (tribe_size - num_chiefs) * 
  (tribe_size - num_chiefs - 1) * 
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 3243240 := by
  sorry

end NUMINAMATH_CALUDE_leadership_combinations_l1311_131114


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l1311_131151

theorem cos_sin_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos x = Real.sqrt 2 * Real.sin (2 * (x + Real.pi/4) + Real.pi/4) := by
sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l1311_131151


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1311_131177

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 20

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_bars = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1311_131177


namespace NUMINAMATH_CALUDE_team_can_have_odd_and_even_points_l1311_131150

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (points : Nat)

/-- Represents the football tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (bonus_points : Nat)

/-- Definition of the specific tournament conditions -/
def specific_tournament : Tournament :=
  { teams := sorry,
    num_teams := 10,
    points_for_win := 3,
    points_for_draw := 1,
    bonus_points := 5 }

/-- Theorem stating that a team can end with both odd and even points -/
theorem team_can_have_odd_and_even_points (t : Tournament) 
  (h1 : t.num_teams = 10)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.bonus_points = 5) :
  ∃ (team1 team2 : Team), 
    team1 ∈ t.teams ∧ 
    team2 ∈ t.teams ∧ 
    Odd team1.points ∧ 
    Even team2.points :=
sorry

end NUMINAMATH_CALUDE_team_can_have_odd_and_even_points_l1311_131150


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1311_131163

/-- Given a circle with equation 2x^2 = -2y^2 + 16x - 8y + 40, 
    the area of a square inscribed around it with one pair of sides 
    parallel to the x-axis is 160 square units. -/
theorem inscribed_square_area (x y : ℝ) : 
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 40 → 
  ∃ (s : ℝ), s > 0 ∧ s^2 = 160 ∧ 
  ∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 ≤ (s/2)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1311_131163


namespace NUMINAMATH_CALUDE_function_inequality_l1311_131147

theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  (∀ x, f x = x + 4/x) →
  (∀ x, g x = 2^x + a) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1311_131147


namespace NUMINAMATH_CALUDE_power_function_domain_and_oddness_l1311_131157

def α_set : Set ℝ := {-1, 1/2, 1, 2, 3}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem power_function_domain_and_oddness (α : ℝ) :
  α ∈ α_set →
  (((∀ x : ℝ, ∃ y : ℝ, y = x^α) ∧ is_odd_function (fun x => x^α)) ↔ (α = 1 ∨ α = 3)) :=
sorry

end NUMINAMATH_CALUDE_power_function_domain_and_oddness_l1311_131157


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1311_131122

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1311_131122


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1311_131128

theorem square_plus_reciprocal_square (a : ℝ) (h : (a + 1/a)^4 = 5) :
  a^2 + 1/a^2 = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1311_131128


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1311_131142

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1311_131142


namespace NUMINAMATH_CALUDE_sin_2phi_value_l1311_131134

theorem sin_2phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l1311_131134


namespace NUMINAMATH_CALUDE_gage_received_fraction_l1311_131104

-- Define the initial numbers of cubes
def grady_red : ℕ := 20
def grady_blue : ℕ := 15
def gage_initial_red : ℕ := 10
def gage_initial_blue : ℕ := 12

-- Define the fraction of blue cubes Gage received
def blue_fraction : ℚ := 1/3

-- Define the total number of cubes Gage has after receiving some from Grady
def gage_total : ℕ := 35

-- Define the fraction of red cubes Gage received as a rational number
def red_fraction : ℚ := 2/5

-- Theorem statement
theorem gage_received_fraction :
  (gage_initial_red : ℚ) + red_fraction * grady_red + 
  (gage_initial_blue : ℚ) + blue_fraction * grady_blue = gage_total :=
sorry

end NUMINAMATH_CALUDE_gage_received_fraction_l1311_131104


namespace NUMINAMATH_CALUDE_min_circumcircle_area_l1311_131153

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define points A and B as tangent points on circle C
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_C xA yA ∧ circle_C xB yB

-- Define the theorem
theorem min_circumcircle_area (xP yP xA yA xB yB : ℝ) 
  (h_P : point_P xP yP) 
  (h_AB : tangent_points xA yA xB yB) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π = 5*π/4 ∧ 
  ∀ (r' : ℝ), r' > 0 → (∃ (xP' yP' xA' yA' xB' yB' : ℝ),
    point_P xP' yP' ∧ 
    tangent_points xA' yA' xB' yB' ∧ 
    r'^2 * π ≥ 5*π/4) :=
sorry

end NUMINAMATH_CALUDE_min_circumcircle_area_l1311_131153


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1311_131199

/-- The range of b for which the line y = x + b intersects the circle (x-2)^2 + (y-3)^2 = 4
    within the constraints 0 ≤ x ≤ 4 and 1 ≤ y ≤ 3 -/
theorem line_circle_intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧
   y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1311_131199


namespace NUMINAMATH_CALUDE_tournament_512_players_games_l1311_131192

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games required to determine a champion. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem stating that a tournament with 512 initial players requires 511 games. -/
theorem tournament_512_players_games (tournament : SingleEliminationTournament) 
    (h : tournament.initial_players = 512) : 
    games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, 0⟩

end NUMINAMATH_CALUDE_tournament_512_players_games_l1311_131192


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_exists_and_unique_l1311_131184

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_fourth_element_exists_and_unique
  (total_students : ℕ)
  (sample_size : ℕ)
  (n1 n2 n3 : ℕ)
  (h_total : total_students = 52)
  (h_sample : sample_size = 4)
  (h_n1 : n1 = 6)
  (h_n2 : n2 = 32)
  (h_n3 : n3 = 45)
  (h_distinct : n1 < n2 ∧ n2 < n3)
  (h_valid : n1 ≤ total_students ∧ n2 ≤ total_students ∧ n3 ≤ total_students) :
  ∃! n4 : ℕ,
    ∃ s : SystematicSample,
      s.population_size = total_students ∧
      s.sample_size = sample_size ∧
      isInSample s n1 ∧
      isInSample s n2 ∧
      isInSample s n3 ∧
      isInSample s n4 ∧
      n4 ≠ n1 ∧ n4 ≠ n2 ∧ n4 ≠ n3 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_exists_and_unique_l1311_131184


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1311_131102

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1311_131102


namespace NUMINAMATH_CALUDE_min_value_expression_l1311_131190

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 14 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1311_131190


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l1311_131168

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l1311_131168


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1311_131120

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 2*y = 12 :=
by
  -- The unique solution is y = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1311_131120


namespace NUMINAMATH_CALUDE_reverse_two_digit_number_l1311_131109

/-- For a two-digit number with tens digit x and units digit y,
    the number formed by reversing its digits is 10y + x. -/
theorem reverse_two_digit_number (x y : ℕ) 
  (h1 : x ≥ 1 ∧ x ≤ 9) (h2 : y ≥ 0 ∧ y ≤ 9) : 
  (10 * y + x) = (10 * y + x) := by
  sorry

#check reverse_two_digit_number

end NUMINAMATH_CALUDE_reverse_two_digit_number_l1311_131109


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1311_131135

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1311_131135


namespace NUMINAMATH_CALUDE_equation_solution_and_expression_value_l1311_131169

theorem equation_solution_and_expression_value :
  ∃ y : ℝ, (4 * y - 8 = 2 * y + 18) ∧ (3 * (y^2 + 6 * y + 12) = 777) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_expression_value_l1311_131169


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1311_131121

theorem set_intersection_problem :
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-2, -1, 0, 1}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1311_131121


namespace NUMINAMATH_CALUDE_polygon_sides_l1311_131171

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (180 * (n - 2) : ℝ) / 360 = 5 / 2 → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1311_131171


namespace NUMINAMATH_CALUDE_missing_files_l1311_131107

theorem missing_files (total : ℕ) (organized_afternoon : ℕ) : total = 60 → organized_afternoon = 15 → total - (total / 2 + organized_afternoon) = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_files_l1311_131107


namespace NUMINAMATH_CALUDE_set_equality_implies_power_l1311_131187

theorem set_equality_implies_power (m n : ℝ) : 
  let P : Set ℝ := {1, m}
  let Q : Set ℝ := {2, -n}
  P = Q → m^n = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_power_l1311_131187


namespace NUMINAMATH_CALUDE_janes_age_l1311_131140

/-- Jane's babysitting problem -/
theorem janes_age (start_age : ℕ) (years_since_stop : ℕ) (oldest_babysat_now : ℕ)
  (h1 : start_age = 18)
  (h2 : years_since_stop = 12)
  (h3 : oldest_babysat_now = 23) :
  ∃ (current_age : ℕ),
    current_age = 34 ∧
    current_age ≥ start_age + years_since_stop ∧
    2 * (oldest_babysat_now - years_since_stop) ≤ current_age - years_since_stop :=
by sorry

end NUMINAMATH_CALUDE_janes_age_l1311_131140


namespace NUMINAMATH_CALUDE_square_area_after_cuts_l1311_131148

theorem square_area_after_cuts (x : ℝ) : 
  x > 0 → x - 3 > 0 → x - 5 > 0 → 
  x^2 - (x - 3) * (x - 5) = 81 → 
  x^2 = 144 := by
sorry

end NUMINAMATH_CALUDE_square_area_after_cuts_l1311_131148


namespace NUMINAMATH_CALUDE_inequality_solution_l1311_131103

theorem inequality_solution :
  {x : ℝ | |(6 - x) / 4| < 3 ∧ x ≥ 2} = Set.Ici 2 ∩ Set.Iio 18 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1311_131103


namespace NUMINAMATH_CALUDE_four_valid_a_values_l1311_131108

theorem four_valid_a_values : 
  let equation_solution (a : ℝ) := (a - 2 : ℝ)
  let inequality_system (a y : ℝ) := y + 9 ≤ 2 * (y + 2) ∧ (2 * y - a) / 3 ≥ 1
  let valid_a (a : ℤ) := 
    equation_solution a > 0 ∧ 
    equation_solution a ≠ 3 ∧ 
    (∀ y : ℝ, inequality_system a y ↔ y ≥ 5)
  ∃! (s : Finset ℤ), (∀ a ∈ s, valid_a a) ∧ s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_four_valid_a_values_l1311_131108


namespace NUMINAMATH_CALUDE_evaluate_expression_l1311_131115

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 6) = 15 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1311_131115


namespace NUMINAMATH_CALUDE_equal_money_in_40_days_l1311_131144

/-- The number of days it takes for Taehyung and Minwoo to have the same amount of money -/
def days_to_equal_money (taehyung_initial : ℕ) (minwoo_initial : ℕ) 
  (taehyung_daily : ℕ) (minwoo_daily : ℕ) : ℕ :=
  (taehyung_initial - minwoo_initial) / (minwoo_daily - taehyung_daily)

/-- Theorem stating that it takes 40 days for Taehyung and Minwoo to have the same amount of money -/
theorem equal_money_in_40_days :
  days_to_equal_money 12000 4000 300 500 = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_in_40_days_l1311_131144


namespace NUMINAMATH_CALUDE_mass_of_substance_l1311_131182

-- Define the density of the substance
def density : ℝ := 500

-- Define the volume in cubic centimeters
def volume_cm : ℝ := 2

-- Define the conversion factor from cm³ to m³
def cm3_to_m3 : ℝ := 1e-6

-- Define the mass in kg
def mass : ℝ := density * (volume_cm * cm3_to_m3)

-- Theorem statement
theorem mass_of_substance :
  mass = 1e-3 := by sorry

end NUMINAMATH_CALUDE_mass_of_substance_l1311_131182


namespace NUMINAMATH_CALUDE_four_numbers_puzzle_l1311_131111

theorem four_numbers_puzzle (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_puzzle_l1311_131111


namespace NUMINAMATH_CALUDE_dragon_disc_reassembly_l1311_131112

/-- A circular disc with a dragon painted on it -/
structure DragonDisc where
  center : ℝ × ℝ
  radius : ℝ
  dragon_center : ℝ × ℝ

/-- Two discs are congruent if they have the same radius -/
def congruent (d1 d2 : DragonDisc) : Prop := d1.radius = d2.radius

/-- The dragon covers the center of the disc if its center coincides with the disc's center -/
def dragon_covers_center (d : DragonDisc) : Prop := d.center = d.dragon_center

/-- A disc can be cut and reassembled if there exists a line that divides it into two pieces -/
def can_cut_and_reassemble (d : DragonDisc) : Prop := 
  ∃ (line : ℝ × ℝ → ℝ × ℝ → Prop), 
    ∃ (piece1 piece2 : Set (ℝ × ℝ)), 
      piece1 ∪ piece2 = {p | (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2}

theorem dragon_disc_reassembly 
  (d1 d2 : DragonDisc)
  (h_congruent : congruent d1 d2)
  (h_d1_center : dragon_covers_center d1)
  (h_d2_offset : ¬dragon_covers_center d2) :
  can_cut_and_reassemble d2 ∧ 
  ∃ (d2_new : DragonDisc), congruent d2 d2_new ∧ dragon_covers_center d2_new :=
by sorry

end NUMINAMATH_CALUDE_dragon_disc_reassembly_l1311_131112


namespace NUMINAMATH_CALUDE_joes_honey_purchase_l1311_131179

theorem joes_honey_purchase (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) 
  (plant_price : ℚ) (total_spent : ℚ) (orange_count : ℕ) (juice_count : ℕ) 
  (plant_count : ℕ) :
  orange_price = 9/2 →
  juice_price = 1/2 →
  honey_price = 5 →
  plant_price = 9 →
  total_spent = 68 →
  orange_count = 3 →
  juice_count = 7 →
  plant_count = 4 →
  (total_spent - (orange_price * orange_count + juice_price * juice_count + 
    plant_price * plant_count)) / honey_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_joes_honey_purchase_l1311_131179


namespace NUMINAMATH_CALUDE_intersection_distance_l1311_131160

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the directrix of C
def directrix (x : ℝ) : Prop :=
  x = -2

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    directrix A.1 ∧
    directrix B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1311_131160


namespace NUMINAMATH_CALUDE_no_divisors_between_2_and_100_l1311_131101

theorem no_divisors_between_2_and_100 (n : ℕ+) 
  (h : ∀ k ∈ Finset.range 99, (Finset.sum (Finset.range n) (fun i => (i + 1) ^ (k + 1))) % n = 0) :
  ∀ d ∈ Finset.range 99, d > 1 → ¬(d ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_divisors_between_2_and_100_l1311_131101


namespace NUMINAMATH_CALUDE_binomial_20_19_l1311_131100

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l1311_131100


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1311_131116

theorem min_value_of_function (x : ℝ) (hx : x < 0) :
  (1 - 2*x - 3/x) ≥ 1 + 2*Real.sqrt 6 := by
  sorry

theorem min_value_achieved (x : ℝ) (hx : x < 0) :
  ∃ x₀, x₀ < 0 ∧ (1 - 2*x₀ - 3/x₀) = 1 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1311_131116


namespace NUMINAMATH_CALUDE_opponent_total_score_l1311_131145

def team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def games_lost_by_one (scores : List ℕ) : ℕ := 6

def score_ratio_in_other_games : ℕ := 3

theorem opponent_total_score :
  let opponent_scores := team_scores.map (λ score =>
    if score % 2 = 1 then score + 1
    else score / score_ratio_in_other_games)
  opponent_scores.sum = 60 := by sorry

end NUMINAMATH_CALUDE_opponent_total_score_l1311_131145


namespace NUMINAMATH_CALUDE_points_on_opposite_sides_l1311_131154

-- Define the line
def line (x y : ℝ) : ℝ := 2*y - 6*x + 1

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem points_on_opposite_sides :
  line origin.1 origin.2 * line point.1 point.2 < 0 := by sorry

end NUMINAMATH_CALUDE_points_on_opposite_sides_l1311_131154


namespace NUMINAMATH_CALUDE_choir_members_count_l1311_131124

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (∃ k : ℕ, n + 4 = 10 * k) ∧
  (∃ m : ℕ, n + 5 = 11 * m) := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l1311_131124


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1311_131159

theorem function_passes_through_point 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) - 2
  f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1311_131159


namespace NUMINAMATH_CALUDE_matrix_problem_l1311_131188

-- Define 2x2 matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- State the conditions
axiom cond1 : A * B = A ^ 2 * B ^ 2 - (A * B) ^ 2
axiom cond2 : Matrix.det B = 2

-- Theorem statement
theorem matrix_problem :
  Matrix.det A = 0 ∧ Matrix.det (A + 2 • B) - Matrix.det (B + 2 • A) = 6 := by
  sorry

end NUMINAMATH_CALUDE_matrix_problem_l1311_131188


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1311_131141

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1311_131141


namespace NUMINAMATH_CALUDE_expression_simplification_l1311_131166

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) + ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1311_131166


namespace NUMINAMATH_CALUDE_cube_root_problem_l1311_131155

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1311_131155


namespace NUMINAMATH_CALUDE_committee_selection_l1311_131133

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l1311_131133


namespace NUMINAMATH_CALUDE_no_valid_ratio_l1311_131185

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  b : ℝ  -- Length of the larger base
  a : ℝ  -- Length of the smaller base
  h : ℝ  -- Height of the trapezoid
  is_positive : 0 < b
  smaller_base_eq_diagonal : a = h
  altitude_eq_larger_base : h = b

/-- Theorem stating that no valid ratio exists between the bases of the described trapezoid -/
theorem no_valid_ratio (t : IsoscelesTrapezoid) : False :=
sorry

end NUMINAMATH_CALUDE_no_valid_ratio_l1311_131185


namespace NUMINAMATH_CALUDE_cosine_irrationality_l1311_131106

theorem cosine_irrationality (n : ℕ) (h : n ≥ 2) : Irrational (Real.cos (π / 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_irrationality_l1311_131106


namespace NUMINAMATH_CALUDE_nabla_two_three_l1311_131197

def nabla (a b : ℕ+) : ℕ := a.val ^ b.val * b.val ^ a.val

theorem nabla_two_three : nabla 2 3 = 72 := by sorry

end NUMINAMATH_CALUDE_nabla_two_three_l1311_131197


namespace NUMINAMATH_CALUDE_two_roots_condition_l1311_131178

open Real

theorem two_roots_condition (k : ℝ) : 
  (∃ x y, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ 
          x * log x - k * x + 1 = 0 ∧ 
          y * log y - k * y + 1 = 0) ↔ 
  k ∈ Set.Ioo 1 (1 + 1/Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l1311_131178


namespace NUMINAMATH_CALUDE_long_division_problem_l1311_131181

theorem long_division_problem (dividend divisor quotient : ℕ) 
  (h1 : dividend = divisor * quotient)
  (h2 : dividend % divisor = 0)
  (h3 : (dividend / divisor) * 105 = 2015 * 10) :
  dividend = 20685 := by
  sorry

end NUMINAMATH_CALUDE_long_division_problem_l1311_131181


namespace NUMINAMATH_CALUDE_accident_calculation_highway_accidents_l1311_131113

/-- Given an accident rate and total number of vehicles, calculate the number of vehicles involved in accidents --/
theorem accident_calculation (accident_rate : ℕ) (vehicles_per_set : ℕ) (total_vehicles : ℕ) :
  accident_rate > 0 →
  vehicles_per_set > 0 →
  total_vehicles ≥ vehicles_per_set →
  (total_vehicles / vehicles_per_set) * accident_rate = 
    (total_vehicles * accident_rate) / vehicles_per_set :=
by
  sorry

/-- Calculate the number of vehicles involved in accidents on a highway --/
theorem highway_accidents :
  let accident_rate := 80  -- vehicles involved in accidents per set
  let vehicles_per_set := 100000000  -- vehicles per set (100 million)
  let total_vehicles := 4000000000  -- total vehicles (4 billion)
  (total_vehicles / vehicles_per_set) * accident_rate = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_accident_calculation_highway_accidents_l1311_131113


namespace NUMINAMATH_CALUDE_probability_red_or_white_l1311_131110

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l1311_131110


namespace NUMINAMATH_CALUDE_expression_simplification_l1311_131123

theorem expression_simplification :
  Real.sqrt 5 * (5 ^ (1/2 : ℝ)) + 20 / 4 * 3 - 9 ^ (3/2 : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1311_131123


namespace NUMINAMATH_CALUDE_max_value_ab_l1311_131174

theorem max_value_ab (a b : ℝ) (h : ∀ x : ℝ, Real.exp x ≥ a * x + b) : 
  (∀ c d : ℝ, (∀ y : ℝ, Real.exp y ≥ c * y + d) → a * b ≥ c * d) → a * b = Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ab_l1311_131174


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l1311_131126

theorem simplify_cube_roots (h1 : 343 = 7^3) (h2 : 125 = 5^3) :
  (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l1311_131126


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1311_131132

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1311_131132


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l1311_131167

theorem shoe_selection_probability (num_pairs : ℕ) (prob : ℚ) : 
  num_pairs = 8 ∧ 
  prob = 1/15 ∧
  (∃ (total : ℕ), 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob) →
  ∃ (total : ℕ), total = 16 ∧ 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob :=
by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l1311_131167


namespace NUMINAMATH_CALUDE_nancy_eats_indian_food_three_times_a_week_l1311_131170

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_food_times : ℕ := sorry

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_food_times : ℕ := 2

/-- Represents the number of antacids Nancy takes when eating Indian food -/
def indian_food_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes when eating Mexican food -/
def mexican_food_antacids : ℕ := 2

/-- Represents the number of antacids Nancy takes on other days -/
def other_days_antacids : ℕ := 1

/-- Represents the total number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of weeks in a month (approximation) -/
def weeks_in_month : ℕ := 4

/-- Represents the total number of antacids Nancy takes per month -/
def total_antacids_per_month : ℕ := 60

/-- Theorem stating that Nancy eats Indian food 3 times a week -/
theorem nancy_eats_indian_food_three_times_a_week :
  indian_food_times = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_eats_indian_food_three_times_a_week_l1311_131170


namespace NUMINAMATH_CALUDE_stripe_area_on_silo_l1311_131173

/-- The area of a stripe painted on a cylindrical silo -/
theorem stripe_area_on_silo (d h w r θ : ℝ) (hd : d = 40) (hh : h = 100) (hw : w = 4) (hr : r = 3) (hθ : θ = 30 * π / 180) :
  let circumference := π * d
  let stripe_length := r * circumference
  let effective_height := h / Real.cos θ
  let stripe_area := w * effective_height
  ⌊stripe_area⌋ = 462 := by sorry

end NUMINAMATH_CALUDE_stripe_area_on_silo_l1311_131173


namespace NUMINAMATH_CALUDE_number_division_problem_l1311_131195

theorem number_division_problem (x : ℚ) : x / 2 = 100 + x / 5 → x = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1311_131195


namespace NUMINAMATH_CALUDE_street_crossing_time_l1311_131152

/-- Proves that a person walking at 5.4 km/h takes 12 minutes to cross a 1080 m street -/
theorem street_crossing_time :
  let street_length : ℝ := 1080  -- length in meters
  let speed_kmh : ℝ := 5.4       -- speed in km/h
  let speed_mpm : ℝ := speed_kmh * 1000 / 60  -- speed in meters per minute
  let time_minutes : ℝ := street_length / speed_mpm
  time_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_street_crossing_time_l1311_131152


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l1311_131129

theorem max_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a / (a + 1) + b / (b + 1)) ≤ 2/3 ∧
  (a / (a + 1) + b / (b + 1) = 2/3 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l1311_131129


namespace NUMINAMATH_CALUDE_function_lower_bound_l1311_131149

theorem function_lower_bound (c : ℝ) : ∀ x : ℝ, x^2 - 2*x + c ≥ c - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l1311_131149


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1311_131118

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 4 → a 2 = 6 →
  (a 1 + a 2 + a 3 + a 4 = 28) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1311_131118


namespace NUMINAMATH_CALUDE_rectangle_dimension_difference_l1311_131196

theorem rectangle_dimension_difference (x y : ℝ) 
  (perimeter : x + y = 10)  -- Half of the perimeter is 10
  (diagonal : x^2 + y^2 = 100)  -- Diagonal squared is 100
  : x - y = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_difference_l1311_131196


namespace NUMINAMATH_CALUDE_blueberry_tart_fraction_l1311_131186

theorem blueberry_tart_fraction (total : Real) (cherry : Real) (peach : Real)
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : peach = 0.08) :
  total - (cherry + peach) = 0.75 := by
sorry

end NUMINAMATH_CALUDE_blueberry_tart_fraction_l1311_131186


namespace NUMINAMATH_CALUDE_sentence_has_32_letters_l1311_131161

def original_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ... БУКВ"
def filled_word : String := "ТРИДЦАТЬ ДВЕ"
def full_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

def is_cyrillic_letter (c : Char) : Bool :=
  (c.toNat ≥ 1040 ∧ c.toNat ≤ 1103) ∨ (c = 'Ё' ∨ c = 'ё')

def count_cyrillic_letters (s : String) : Nat :=
  s.toList.filter is_cyrillic_letter |>.length

theorem sentence_has_32_letters : count_cyrillic_letters full_sentence = 32 := by
  sorry

end NUMINAMATH_CALUDE_sentence_has_32_letters_l1311_131161


namespace NUMINAMATH_CALUDE_neg_q_sufficient_but_not_necessary_for_neg_p_l1311_131131

-- Define the propositions p and q
variable (p q : Prop)

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_but_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

-- Theorem statement
theorem neg_q_sufficient_but_not_necessary_for_neg_p
  (h : sufficient_but_not_necessary p q) :
  sufficient_but_not_necessary (¬q) (¬p) :=
sorry

end NUMINAMATH_CALUDE_neg_q_sufficient_but_not_necessary_for_neg_p_l1311_131131


namespace NUMINAMATH_CALUDE_class_duration_theorem_l1311_131139

/-- Calculates the total duration of classes given the number of periods, period length, number of breaks, and break length. -/
def classDuration (numPeriods : ℕ) (periodLength : ℕ) (numBreaks : ℕ) (breakLength : ℕ) : ℕ :=
  numPeriods * periodLength + numBreaks * breakLength

/-- Proves that the total duration of classes with 5 periods of 40 minutes each and 4 breaks of 5 minutes each is 220 minutes. -/
theorem class_duration_theorem :
  classDuration 5 40 4 5 = 220 := by
  sorry

#eval classDuration 5 40 4 5

end NUMINAMATH_CALUDE_class_duration_theorem_l1311_131139


namespace NUMINAMATH_CALUDE_equality_of_ratios_implies_equality_of_squares_l1311_131125

theorem equality_of_ratios_implies_equality_of_squares
  (x y z : ℝ) (h : x / y = 3 / z) :
  9 * y^2 = x^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_equality_of_ratios_implies_equality_of_squares_l1311_131125


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1311_131162

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - a < 0 ∧ 1 - 2 * x ≥ 7) ↔ x ≤ -3) → 
  a > -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1311_131162


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1311_131191

theorem p_or_q_necessary_not_sufficient (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1311_131191


namespace NUMINAMATH_CALUDE_sphere_plane_distance_l1311_131158

/-- The distance between the center of a sphere and a plane intersecting it -/
theorem sphere_plane_distance (r : ℝ) (A : ℝ) (h1 : r = 2) (h2 : A = Real.pi) :
  Real.sqrt (r^2 - (A / Real.pi)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_plane_distance_l1311_131158


namespace NUMINAMATH_CALUDE_inverse_p_is_true_l1311_131198

-- Define the original proposition
def p (x : ℝ) : Prop := x < -3 → x^2 - 2*x - 8 > 0

-- Define the inverse of the proposition
def p_inverse (x : ℝ) : Prop := ¬(x < -3) → ¬(x^2 - 2*x - 8 > 0)

-- Theorem stating that the inverse of p is true
theorem inverse_p_is_true : ∀ x : ℝ, p_inverse x :=
  sorry

end NUMINAMATH_CALUDE_inverse_p_is_true_l1311_131198


namespace NUMINAMATH_CALUDE_extra_fee_is_fifteen_l1311_131165

/-- Represents the data plan charges and fees -/
structure DataPlan where
  normalMonthlyCharge : ℝ
  promotionalRate : ℝ
  totalPaid : ℝ
  extraFee : ℝ

/-- Calculates the extra fee for going over the data limit -/
def calculateExtraFee (plan : DataPlan) : Prop :=
  let firstMonthCharge := plan.normalMonthlyCharge * plan.promotionalRate
  let regularMonthsCharge := plan.normalMonthlyCharge * 5
  let totalWithoutExtra := firstMonthCharge + regularMonthsCharge
  plan.extraFee = plan.totalPaid - totalWithoutExtra

/-- Theorem stating the extra fee is $15 given the problem conditions -/
theorem extra_fee_is_fifteen :
  ∃ (plan : DataPlan),
    plan.normalMonthlyCharge = 30 ∧
    plan.promotionalRate = 1/3 ∧
    plan.totalPaid = 175 ∧
    calculateExtraFee plan ∧
    plan.extraFee = 15 := by
  sorry

end NUMINAMATH_CALUDE_extra_fee_is_fifteen_l1311_131165


namespace NUMINAMATH_CALUDE_math_books_count_l1311_131183

/-- Proves that the number of math books bought is 27 given the conditions of the problem -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price total_price : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_price = 4)
  (h3 : history_book_price = 5)
  (h4 : total_price = 373) :
  ∃ (math_books : ℕ), 
    math_books + (total_books - math_books) = total_books ∧ 
    math_books * math_book_price + (total_books - math_books) * history_book_price = total_price ∧
    math_books = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l1311_131183


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_g_nonnegative_l1311_131175

/-- A function that is monotonically decreasing on an interval -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

/-- The function f(x) = x^3 + ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The function g(x) = 3x^2 + 2ax + b -/
def g (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: If f(x) is monotonically decreasing on (0, 1), then g(0) * g(1) ≥ 0 -/
theorem monotone_decreasing_implies_g_nonnegative (a b c : ℝ) :
  MonotonicallyDecreasing (f a b c) 0 1 → g a b 0 * g a b 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_g_nonnegative_l1311_131175


namespace NUMINAMATH_CALUDE_sodium_hydrogen_sulfate_effect_l1311_131189

-- Define the water ionization equilibrium
def water_ionization (temp : ℝ) : Prop :=
  temp = 25 → ∃ (K : ℝ), K > 0 ∧ ∀ (c_H2O c_H c_OH : ℝ),
    c_H * c_OH = K * c_H2O

-- Define the enthalpy change
def delta_H_positive : Prop := ∃ (ΔH : ℝ), ΔH > 0

-- Define the addition of sodium hydrogen sulfate
def add_NaHSO4 (c_H_initial c_H_final : ℝ) : Prop :=
  c_H_final > c_H_initial

-- Theorem statement
theorem sodium_hydrogen_sulfate_effect
  (h1 : water_ionization 25)
  (h2 : delta_H_positive)
  (h3 : ∃ (c_H_initial c_H_final : ℝ), add_NaHSO4 c_H_initial c_H_final) :
  ∃ (K : ℝ), K > 0 ∧
    (∀ (c_H2O c_H c_OH : ℝ), c_H * c_OH = K * c_H2O) ∧
    (∃ (c_H_initial c_H_final : ℝ), c_H_final > c_H_initial) :=
sorry

end NUMINAMATH_CALUDE_sodium_hydrogen_sulfate_effect_l1311_131189


namespace NUMINAMATH_CALUDE_fourth_quadrant_a_range_l1311_131117

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 - 2*Complex.I) * (a + Complex.I)

-- Define the point M
def M (a : ℝ) : ℝ × ℝ := (a + 2, 1 - 2*a)

-- Theorem statement
theorem fourth_quadrant_a_range (a : ℝ) :
  (M a).1 > 0 ∧ (M a).2 < 0 → a > 1/2 := by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_a_range_l1311_131117


namespace NUMINAMATH_CALUDE_max_xy_value_min_inverse_sum_l1311_131143

-- Part 1
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  xy ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 12 ∧ x * y = 3 :=
sorry

-- Part 2
theorem min_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  1 / x + 1 / y ≥ 1 + 2 * Real.sqrt 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 3 ∧ 1 / x + 1 / y = 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_min_inverse_sum_l1311_131143


namespace NUMINAMATH_CALUDE_no_solution_l1311_131130

-- Define the property that we want to prove impossible
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x - y

-- State the theorem
theorem no_solution :
  ¬ ∃ f : ℝ → ℝ, Continuous f ∧ SatisfiesFunctionalEquation f :=
sorry

end NUMINAMATH_CALUDE_no_solution_l1311_131130


namespace NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l1311_131194

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (contents : BagContents) (color : ℕ) : ℚ :=
  color / (contents.red + contents.yellow)

/-- The initial contents of the bag -/
def initialBag : BagContents := { red := 9, yellow := 6 }

/-- The number of balls to be added -/
def addedBalls : ℕ := 7

/-- The contents of the bag after adding balls -/
def finalBag : BagContents := { red := initialBag.red + 2, yellow := initialBag.yellow + 5 }

theorem equal_probabilities_after_adding_balls :
  probability finalBag finalBag.red = probability finalBag finalBag.yellow :=
by sorry

end NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l1311_131194


namespace NUMINAMATH_CALUDE_lacson_sweet_potato_sales_l1311_131138

/-- The problem of Mrs. Lacson's sweet potato sales -/
theorem lacson_sweet_potato_sales 
  (total : ℕ)
  (sold_to_adams : ℕ)
  (unsold : ℕ)
  (h1 : total = 80)
  (h2 : sold_to_adams = 20)
  (h3 : unsold = 45) :
  total - sold_to_adams - unsold = 15 := by
  sorry

end NUMINAMATH_CALUDE_lacson_sweet_potato_sales_l1311_131138


namespace NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l1311_131137

/-- Two lines in three-dimensional space -/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it abstract

/-- Predicate for two lines intersecting -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem non_intersecting_lines_parallel_or_skew 
  (l1 l2 : Line3D) (h : ¬ intersect l1 l2) : 
  parallel l1 l2 ∨ skew l1 l2 :=
by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l1311_131137
