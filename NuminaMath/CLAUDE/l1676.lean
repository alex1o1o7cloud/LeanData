import Mathlib

namespace NUMINAMATH_CALUDE_square_division_l1676_167642

theorem square_division (a : ℕ) (h1 : a > 0) :
  (a * a = 25) ∧
  (∃ b : ℕ, b > 0 ∧ a * a = 24 * 1 * 1 + b * b) ∧
  (a = 5) :=
by sorry

end NUMINAMATH_CALUDE_square_division_l1676_167642


namespace NUMINAMATH_CALUDE_expression_upper_bound_l1676_167619

theorem expression_upper_bound (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_prod : a * c = b * d)
  (h_sum : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b) ≤ 4 ∧ 
  ∃ (a' b' c' d' : ℝ), a' / c' + c' / a' + b' / d' + d' / b' = 4 :=
sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l1676_167619


namespace NUMINAMATH_CALUDE_zero_at_specific_point_l1676_167623

/-- A polynomial of degree 3 in x and y -/
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*x^2*y + b₈*x*y^2 + b₉*y^3

theorem zero_at_specific_point 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) : 
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (5/19) (16/19) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_at_specific_point_l1676_167623


namespace NUMINAMATH_CALUDE_jeanette_practice_weeks_l1676_167676

/-- The number of objects Jeanette can juggle after w weeks -/
def objects_juggled (w : ℕ) : ℕ := 3 + 2 * w

/-- The theorem stating that Jeanette practiced for 5 weeks -/
theorem jeanette_practice_weeks : 
  ∃ w : ℕ, objects_juggled w = 13 ∧ w = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeanette_practice_weeks_l1676_167676


namespace NUMINAMATH_CALUDE_max_candy_remainder_l1676_167697

theorem max_candy_remainder (n : ℕ) : 
  ∃ (k : ℕ), n^2 = 5 * k + 4 ∧ 
  ∀ (m : ℕ), n^2 = 5 * m + (n^2 % 5) → n^2 % 5 ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_candy_remainder_l1676_167697


namespace NUMINAMATH_CALUDE_call_processing_ratio_l1676_167611

/-- Represents the ratio of Team A members to Team B members -/
def team_ratio : ℚ := 5 / 8

/-- Represents the fraction of total calls processed by Team B -/
def team_b_calls : ℚ := 8 / 9

/-- Proves that the ratio of calls processed by each member of Team A to each member of Team B is 1:5 -/
theorem call_processing_ratio :
  let team_a_calls := 1 - team_b_calls
  let team_a_members := team_ratio * team_b_members
  (team_a_calls / team_a_members) / (team_b_calls / team_b_members) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_call_processing_ratio_l1676_167611


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1676_167689

theorem sufficient_not_necessary (p q : Prop) :
  (∃ (h : p ∧ q), ¬p = False) ∧
  (∃ (h : ¬p = False), ¬(p ∧ q = True)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1676_167689


namespace NUMINAMATH_CALUDE_least_bananas_total_l1676_167648

/-- Represents the number of bananas taken by each monkey -/
structure BananaCounts where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ

/-- Represents the final distribution of bananas for each monkey -/
structure FinalDistribution where
  m₁ : ℕ
  m₂ : ℕ
  m₃ : ℕ

/-- Calculates the final distribution based on the initial banana counts -/
def calculateFinalDistribution (counts : BananaCounts) : FinalDistribution :=
  { m₁ := counts.b₁ / 2 + counts.b₂ / 12 + counts.b₃ * 3 / 32
  , m₂ := counts.b₁ / 6 + counts.b₂ * 2 / 3 + counts.b₃ * 3 / 32
  , m₃ := counts.b₁ / 6 + counts.b₂ / 12 + counts.b₃ * 3 / 4 }

/-- Checks if the final distribution satisfies the 4:3:2 ratio -/
def satisfiesRatio (dist : FinalDistribution) : Prop :=
  3 * dist.m₁ = 4 * dist.m₂ ∧ 2 * dist.m₁ = 3 * dist.m₃

/-- The main theorem stating the least possible total number of bananas -/
theorem least_bananas_total (counts : BananaCounts) :
  (∀ (dist : FinalDistribution), dist = calculateFinalDistribution counts → satisfiesRatio dist) →
  counts.b₁ + counts.b₂ + counts.b₃ ≥ 148 :=
by sorry

end NUMINAMATH_CALUDE_least_bananas_total_l1676_167648


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1676_167696

/-- Represents a football team with a given number of total players and offensive linemen --/
structure FootballTeam where
  total_players : Nat
  offensive_linemen : Nat
  h_offensive_linemen_le_total : offensive_linemen ≤ total_players

/-- Calculates the number of ways to choose a starting lineup --/
def chooseStartingLineup (team : FootballTeam) : Nat :=
  team.offensive_linemen * (team.total_players - 1) * (team.total_players - 2) * (team.total_players - 3)

/-- Theorem stating that for a team of 10 players with 3 offensive linemen, 
    there are 1512 ways to choose a starting lineup --/
theorem starting_lineup_count (team : FootballTeam) 
  (h_total : team.total_players = 10) 
  (h_offensive : team.offensive_linemen = 3) : 
  chooseStartingLineup team = 1512 := by
  sorry

#eval chooseStartingLineup ⟨10, 3, by norm_num⟩

end NUMINAMATH_CALUDE_starting_lineup_count_l1676_167696


namespace NUMINAMATH_CALUDE_orange_box_ratio_l1676_167661

theorem orange_box_ratio (total : ℕ) (given_to_mother : ℕ) (remaining : ℕ) :
  total = 9 →
  given_to_mother = 1 →
  remaining = 4 →
  (total - given_to_mother - remaining) * 2 = total - given_to_mother :=
by sorry

end NUMINAMATH_CALUDE_orange_box_ratio_l1676_167661


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1676_167650

open Set
open Function
open Real

noncomputable def f (x : ℝ) : ℝ := x * (x^2 - Real.cos (x/3) + 2)

theorem solution_set_of_inequality :
  let S := {x : ℝ | x ∈ Ioo (-3) 3 ∧ f (1 + x) + f 2 < f (1 - x)}
  S = Ioo (-2) (-1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1676_167650


namespace NUMINAMATH_CALUDE_factorial_difference_l1676_167610

theorem factorial_difference : Nat.factorial 12 - Nat.factorial 11 = 439084800 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1676_167610


namespace NUMINAMATH_CALUDE_determinant_scaling_l1676_167694

open Matrix

theorem determinant_scaling (x y z a b c p q r : ℝ) :
  det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2 →
  det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l1676_167694


namespace NUMINAMATH_CALUDE_hclo4_moles_required_l1676_167686

theorem hclo4_moles_required (naoh_moles : ℝ) (naclo4_moles : ℝ) (h2o_moles : ℝ) 
  (hclo4_participation_rate : ℝ) :
  naoh_moles = 3 →
  naclo4_moles = 3 →
  h2o_moles = 3 →
  hclo4_participation_rate = 0.8 →
  ∃ (hclo4_moles : ℝ), 
    hclo4_moles = naoh_moles / hclo4_participation_rate ∧ 
    hclo4_moles = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_hclo4_moles_required_l1676_167686


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1676_167618

-- Define the variables
variable (a b c x : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  a^3*b - 2*b^2*c + 5*a^3*b - 3*a^3*b + 2*c*b^2 = 3*a^3*b := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  (2*x^2 - 1/2 + 3*x) - 4*(x - x^2 + 1/2) = 6*x^2 - x - 5/2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1676_167618


namespace NUMINAMATH_CALUDE_second_term_is_seven_general_formula_l1676_167621

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone : Monotone a
  is_arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
  sum_first_three : a 1 + a 2 + a 3 = 21
  product_first_three : a 1 * a 2 * a 3 = 231

/-- The second term of the sequence is 7 -/
theorem second_term_is_seven (seq : ArithmeticSequence) : seq.a 2 = 7 := by
  sorry

/-- The general formula for the n-th term -/
theorem general_formula (seq : ArithmeticSequence) : ∀ n : ℕ, seq.a n = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_seven_general_formula_l1676_167621


namespace NUMINAMATH_CALUDE_magpie_call_not_correlation_l1676_167671

/-- Represents a statement that may or may not indicate a correlation. -/
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- A good teacher produces outstanding students
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- The magpie's call is a sign of happiness

/-- Predicate to determine if a statement represents a correlation. -/
def is_correlation (s : Statement) : Prop :=
  match s with
  | Statement.A => True
  | Statement.B => True
  | Statement.C => True
  | Statement.D => False

/-- Theorem stating that Statement D does not represent a correlation. -/
theorem magpie_call_not_correlation :
  ¬ (is_correlation Statement.D) :=
sorry

end NUMINAMATH_CALUDE_magpie_call_not_correlation_l1676_167671


namespace NUMINAMATH_CALUDE_shaded_to_large_square_ratio_l1676_167690

theorem shaded_to_large_square_ratio :
  let large_square_side : ℕ := 5
  let unit_squares_count : ℕ := large_square_side ^ 2
  let half_squares_in_shaded : ℕ := 5
  let shaded_area : ℚ := (half_squares_in_shaded : ℚ) / 2
  let large_square_area : ℕ := unit_squares_count
  (shaded_area : ℚ) / (large_square_area : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_large_square_ratio_l1676_167690


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_l1676_167665

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  id : Nat
  r_squared : Real

/-- Given a list of regression models, the model with the highest R² value has the best fit -/
theorem best_fit_highest_r_squared (models : List RegressionModel) :
  models ≠ [] →
  ∃ best_model : RegressionModel,
    best_model ∈ models ∧
    (∀ model ∈ models, model.r_squared ≤ best_model.r_squared) ∧
    (∀ model ∈ models, model.r_squared = best_model.r_squared → model = best_model) :=
by sorry

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_l1676_167665


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1676_167652

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → 
  ((5 : ℝ)^2 + m * 5 - 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1676_167652


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1676_167602

theorem cos_pi_half_plus_alpha (α : ℝ) (h : Real.sin (-α) = Real.sqrt 5 / 3) :
  Real.cos (π / 2 + α) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1676_167602


namespace NUMINAMATH_CALUDE_walking_distance_l1676_167691

/-- 
Given a person who walks for time t hours:
- At 12 km/hr, they cover a distance of 12t km
- At 20 km/hr, they cover a distance of 20t km
- The difference between these distances is 30 km

Prove that the actual distance travelled at 12 km/hr is 45 km
-/
theorem walking_distance (t : ℝ) 
  (h1 : 20 * t = 12 * t + 30) : 12 * t = 45 := by sorry

end NUMINAMATH_CALUDE_walking_distance_l1676_167691


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1676_167658

/-- A projection matrix is idempotent (P² = P) -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific form of our matrix -/
def P (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 1/3; b, 2/3]

/-- The theorem stating that the only values of a and b that make P a projection matrix are 1/3 and 2/3 -/
theorem projection_matrix_values :
  ∀ a b : ℚ, is_projection_matrix (P a b) ↔ a = 1/3 ∧ b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l1676_167658


namespace NUMINAMATH_CALUDE_complement_of_intersection_union_condition_l1676_167699

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem 1: Complement of intersection
theorem complement_of_intersection :
  (Aᶜ ∪ Bᶜ : Set ℝ) = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem 2: Condition for B ∪ C = C
theorem union_condition (a : ℝ) :
  B ∪ C a = C a → a ≥ -4 := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_union_condition_l1676_167699


namespace NUMINAMATH_CALUDE_path_area_calculation_l1676_167681

/-- Calculates the area of a path surrounding a rectangular field -/
def pathArea (fieldLength fieldWidth pathWidth : ℝ) : ℝ :=
  let totalLength := fieldLength + 2 * pathWidth
  let totalWidth := fieldWidth + 2 * pathWidth
  totalLength * totalWidth - fieldLength * fieldWidth

theorem path_area_calculation :
  let fieldLength : ℝ := 75
  let fieldWidth : ℝ := 55
  let pathWidth : ℝ := 2.5
  pathArea fieldLength fieldWidth pathWidth = 675 := by
  sorry

end NUMINAMATH_CALUDE_path_area_calculation_l1676_167681


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1676_167631

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1676_167631


namespace NUMINAMATH_CALUDE_jesse_bananas_l1676_167693

theorem jesse_bananas (num_bananas : ℕ) : 
  (num_bananas % 3 = 0 ∧ num_bananas / 3 = 7) → num_bananas = 21 := by
  sorry

end NUMINAMATH_CALUDE_jesse_bananas_l1676_167693


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1676_167685

theorem nested_fraction_evaluation : 
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1676_167685


namespace NUMINAMATH_CALUDE_mike_picked_64_peaches_l1676_167646

/-- Calculates the number of peaches Mike picked from the orchard -/
def peaches_picked (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Given the initial conditions, Mike picked 64 peaches from the orchard -/
theorem mike_picked_64_peaches (initial : ℕ) (given_away : ℕ) (final : ℕ)
    (h1 : initial = 34)
    (h2 : given_away = 12)
    (h3 : final = 86) :
  peaches_picked initial given_away final = 64 := by
  sorry

#eval peaches_picked 34 12 86

end NUMINAMATH_CALUDE_mike_picked_64_peaches_l1676_167646


namespace NUMINAMATH_CALUDE_calculate_expression_l1676_167682

theorem calculate_expression : ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1676_167682


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l1676_167634

/-- Proves that the ratio of weight held in each hand to body weight is 1:1 --/
theorem weight_ratio_proof (body_weight hand_weight total_weight : ℝ) 
  (hw : body_weight = 150)
  (vest_weight : ℝ)
  (hv : vest_weight = body_weight / 2)
  (ht : total_weight = 525)
  (he : total_weight = body_weight + vest_weight + 2 * hand_weight) :
  hand_weight / body_weight = 1 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l1676_167634


namespace NUMINAMATH_CALUDE_unique_number_property_l1676_167679

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l1676_167679


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1676_167644

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n : ℚ) / d = 7 + (789 : ℚ) / 10000 / (1 - 1 / 10000) :=
by
  -- The fraction 365/85 satisfies this property
  use 365, 85
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1676_167644


namespace NUMINAMATH_CALUDE_perimeter_unchanged_after_adding_tiles_l1676_167683

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

/-- The theorem stating that adding two tiles can maintain the same perimeter -/
theorem perimeter_unchanged_after_adding_tiles :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 9 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 2 ∧
    final.perimeter = 16 :=
  sorry

end NUMINAMATH_CALUDE_perimeter_unchanged_after_adding_tiles_l1676_167683


namespace NUMINAMATH_CALUDE_fish_count_l1676_167628

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem fish_count : total_fish = 157 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l1676_167628


namespace NUMINAMATH_CALUDE_like_terms_imply_x_y_equal_one_l1676_167636

/-- Two terms are like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b, ∃ k, term1 a b = k * term2 a b ∨ term2 a b = k * term1 a b

/-- Given two terms 3a^(x+1)b^2 and 7a^2b^(x+y), if they are like terms, then x = 1 and y = 1. -/
theorem like_terms_imply_x_y_equal_one (x y : ℕ) :
  like_terms (λ a b => 3 * a^(x + 1) * b^2) (λ a b => 7 * a^2 * b^(x + y)) →
  x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_x_y_equal_one_l1676_167636


namespace NUMINAMATH_CALUDE_social_gathering_handshakes_l1676_167607

theorem social_gathering_handshakes (n : ℕ) (h : n = 8) : 
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 112 := by
sorry

end NUMINAMATH_CALUDE_social_gathering_handshakes_l1676_167607


namespace NUMINAMATH_CALUDE_expression_equality_l1676_167673

theorem expression_equality : 6 * 1000 + 5 * 100 + 6 * 1 = 6506 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1676_167673


namespace NUMINAMATH_CALUDE_cow_calf_ratio_l1676_167637

def total_cost : ℕ := 990
def cow_cost : ℕ := 880
def calf_cost : ℕ := 110

theorem cow_calf_ratio : 
  ∃ (m : ℕ), m > 0 ∧ cow_cost = m * calf_cost ∧ cow_cost / calf_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_calf_ratio_l1676_167637


namespace NUMINAMATH_CALUDE_power_equation_solution_l1676_167664

theorem power_equation_solution : ∃ K : ℕ, (81 ^ 2) * (27 ^ 3) = 3 ^ K ∧ K = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1676_167664


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1676_167667

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 7 * p - 6 = 0) → 
  (3 * q^2 + 7 * q - 6 = 0) → 
  (p - 2) * (q - 2) = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1676_167667


namespace NUMINAMATH_CALUDE_permutations_with_non_adjacency_l1676_167633

theorem permutations_with_non_adjacency (n : ℕ) (h : n ≥ 4) :
  let total_permutations := n.factorial
  let adjacent_a1_a2 := 2 * (n - 1).factorial
  let adjacent_a3_a4 := 2 * (n - 1).factorial
  let both_adjacent := 4 * (n - 2).factorial
  total_permutations - adjacent_a1_a2 - adjacent_a3_a4 + both_adjacent = (n^2 - 5*n + 8) * (n - 2).factorial :=
by sorry

#check permutations_with_non_adjacency

end NUMINAMATH_CALUDE_permutations_with_non_adjacency_l1676_167633


namespace NUMINAMATH_CALUDE_vector_calculation_l1676_167613

theorem vector_calculation : 
  2 • (((3 : ℝ), -2, 5) + ((-1 : ℝ), 6, -7)) = ((4 : ℝ), 8, -4) := by
sorry

end NUMINAMATH_CALUDE_vector_calculation_l1676_167613


namespace NUMINAMATH_CALUDE_envelope_distribution_theorem_l1676_167626

/-- Represents the number of members in the WeChat group -/
def num_members : ℕ := 5

/-- Represents the number of red envelopes -/
def num_envelopes : ℕ := 4

/-- Represents the number of 2-yuan envelopes -/
def num_2yuan : ℕ := 2

/-- Represents the number of 3-yuan envelopes -/
def num_3yuan : ℕ := 2

/-- Represents the number of specific members (A and B) who must get an envelope -/
def num_specific_members : ℕ := 2

/-- Represents the function that calculates the number of ways to distribute the envelopes -/
noncomputable def num_distribution_ways : ℕ := sorry

theorem envelope_distribution_theorem :
  num_distribution_ways = 18 := by sorry

end NUMINAMATH_CALUDE_envelope_distribution_theorem_l1676_167626


namespace NUMINAMATH_CALUDE_olivias_remaining_money_l1676_167609

def olivias_wallet (initial_amount : ℕ) (atm_amount : ℕ) (extra_spent : ℕ) : ℕ :=
  initial_amount + atm_amount - (atm_amount + extra_spent)

theorem olivias_remaining_money :
  olivias_wallet 53 91 39 = 14 :=
by sorry

end NUMINAMATH_CALUDE_olivias_remaining_money_l1676_167609


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_2560_l1676_167680

/-- Calculates the sale in the fifth month given the sales of the first four months,
    the average sale over six months, and the sale in the sixth month. -/
def fifth_month_sale (sale1 sale2 sale3 sale4 average_sale sixth_month_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale4 + sixth_month_sale)

/-- Proves that the sale in the fifth month is 2560 given the specified conditions. -/
theorem fifth_month_sale_is_2560 :
  fifth_month_sale 2435 2920 2855 3230 2500 1000 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_2560_l1676_167680


namespace NUMINAMATH_CALUDE_popsicle_stick_ratio_l1676_167653

-- Define the number of popsicle sticks for each person
def steve_sticks : ℕ := 12
def total_sticks : ℕ := 108

-- Define the relationship between Sam and Sid's sticks
def sam_sticks (sid_sticks : ℕ) : ℕ := 3 * sid_sticks

-- Theorem to prove
theorem popsicle_stick_ratio :
  ∃ (sid_sticks : ℕ),
    sid_sticks > 0 ∧
    sam_sticks sid_sticks + sid_sticks + steve_sticks = total_sticks ∧
    sid_sticks = 2 * steve_sticks :=
by sorry

end NUMINAMATH_CALUDE_popsicle_stick_ratio_l1676_167653


namespace NUMINAMATH_CALUDE_pet_store_cages_l1676_167668

theorem pet_store_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 54)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 7) :
  total_birds / (parrots_per_cage + parakeets_per_cage) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1676_167668


namespace NUMINAMATH_CALUDE_square_root_of_four_l1676_167608

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1676_167608


namespace NUMINAMATH_CALUDE_thirteenth_result_l1676_167688

theorem thirteenth_result (total_count : Nat) (total_avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) 
  (h_total_count : total_count = 25)
  (h_total_avg : total_avg = 20)
  (h_first_12_avg : first_12_avg = 14)
  (h_last_12_avg : last_12_avg = 17) :
  ∃ (thirteenth : ℚ), 
    (total_count : ℚ) * total_avg = 
      12 * first_12_avg + thirteenth + 12 * last_12_avg ∧ 
    thirteenth = 128 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_result_l1676_167688


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1676_167629

theorem algebraic_simplification (x y : ℝ) (h : y ≠ 0) :
  (25 * x^3 * y) * (8 * x * y) * (1 / (5 * x * y^2)^2) = 8 * x^2 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1676_167629


namespace NUMINAMATH_CALUDE_unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l1676_167669

/-- Predicate that checks if a pair (m, n) satisfies the condition for all (x, y) -/
def satisfies_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, y ≠ 0 → x / y = m → (x + y)^2 = n

/-- Theorem stating that (-1, 0) is the only pair satisfying the condition -/
theorem unique_satisfying_pair :
  ∃! p : ℝ × ℝ, satisfies_condition p.1 p.2 ∧ p = (-1, 0) := by
  sorry

/-- Corollary: If (m, n) satisfies the condition, then m = -1 and n = 0 -/
theorem satisfying_pair_is_negative_one_zero (m n : ℝ) :
  satisfies_condition m n → m = -1 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l1676_167669


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1676_167614

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1

-- Define the condition of shared foci
def shared_foci (e h : (ℝ → ℝ → Prop)) : Prop := 
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ x y, e x y ↔ x^2/(c^2+4) + y^2/4 = 1) ∧
    (∀ x y, h x y ↔ ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ a^2 - b^2 = c^2)

-- Define the asymptote condition
def asymptote_condition (h : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), h x y ∧ x - 2*y = 0

-- The theorem to prove
theorem hyperbola_equation 
  (h : shared_foci ellipse hyperbola_C)
  (a : asymptote_condition hyperbola_C) :
  ∀ x y, hyperbola_C x y ↔ x^2/4 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1676_167614


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l1676_167654

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ + z₂ = 0) →
  (z₂ = -2 + 3*I) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l1676_167654


namespace NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l1676_167647

theorem complex_exp_210_deg_60th_power : 
  (Complex.exp (210 * π / 180 * Complex.I)) ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l1676_167647


namespace NUMINAMATH_CALUDE_cube_face_projections_l1676_167601

/-- Given three faces of a unit cube sharing a common vertex, if their projections onto a fixed plane
have areas in the ratio 6:10:15, then the sum of these areas is 31/19. -/
theorem cube_face_projections (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →  -- Ensure positive areas
  x^2 + y^2 + z^2 = 1 →  -- Sum of squares of projection areas equals 1
  x / 6 = y / 10 ∧ y / 10 = z / 15 →  -- Ratio condition
  x + y + z = 31 / 19 := by
sorry

end NUMINAMATH_CALUDE_cube_face_projections_l1676_167601


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l1676_167666

/-- Represents the dimensions of a decorative window --/
structure WindowDimensions where
  ab : ℝ  -- width of the rectangle and diameter of semicircles
  ad : ℝ  -- length of the rectangle
  h_ab_positive : ab > 0
  h_ad_positive : ad > 0
  h_ratio : ad / ab = 4 / 3

/-- Theorem about the ratio of areas in a decorative window --/
theorem decorative_window_area_ratio 
  (w : WindowDimensions) 
  (h_ab : w.ab = 36) : 
  (w.ad * w.ab) / (π * (w.ab / 2)^2) = 16 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l1676_167666


namespace NUMINAMATH_CALUDE_current_age_of_D_l1676_167640

theorem current_age_of_D (a b c d : ℕ) : 
  a + b + c + d = 108 →
  a - b = 12 →
  c - (a - 34) = 3 * (d - (a - 34)) →
  d = 13 := by
sorry

end NUMINAMATH_CALUDE_current_age_of_D_l1676_167640


namespace NUMINAMATH_CALUDE_two_digit_three_digit_percentage_equality_l1676_167651

theorem two_digit_three_digit_percentage_equality :
  ∃! (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) ∧
    (B ≥ 100 ∧ B ≤ 999) ∧
    (A * (1 + B / 100 : ℚ) = B * (1 - A / 100 : ℚ)) ∧
    A = 40 ∧
    B = 200 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_percentage_equality_l1676_167651


namespace NUMINAMATH_CALUDE_rectangle_area_l1676_167625

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1676_167625


namespace NUMINAMATH_CALUDE_hotel_charges_l1676_167616

theorem hotel_charges (G : ℝ) (h1 : G > 0) : 
  let R := 2 * G
  let P := R * (1 - 0.55)
  P = G * (1 - 0.1) := by
sorry

end NUMINAMATH_CALUDE_hotel_charges_l1676_167616


namespace NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l1676_167622

/-- Given a point P(3, 4) on the terminal side of angle α, prove that sin α + cos α = 8/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : Real × Real := (3, 4)
  (P.1 = 3 ∧ P.2 = 4) →  -- Point P has coordinates (3, 4)
  (P.1^2 + P.2^2 = 5^2) →  -- P is on the unit circle with radius 5
  (Real.sin α = P.2 / 5 ∧ Real.cos α = P.1 / 5) →  -- Definition of sin and cos for this point
  Real.sin α + Real.cos α = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l1676_167622


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l1676_167603

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l1676_167603


namespace NUMINAMATH_CALUDE_alternating_sum_equals_three_to_seven_l1676_167672

theorem alternating_sum_equals_three_to_seven (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ = 3^7 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_three_to_seven_l1676_167672


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1676_167630

/-- Given line equation -/
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

/-- Candidate line equation -/
def candidate_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

/-- Point that the candidate line should pass through -/
def point : ℝ × ℝ := (2, -1)

theorem perpendicular_line_through_point :
  (candidate_line point.1 point.2) ∧ 
  (∀ (x y : ℝ), given_line x y → 
    (3 * 2 + 2 * (-3) = 0)) := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1676_167630


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1676_167605

theorem fraction_equivalence : 
  let x : ℚ := 13/2
  (4 + x) / (7 + x) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1676_167605


namespace NUMINAMATH_CALUDE_remainder_95_equals_12_l1676_167674

theorem remainder_95_equals_12 (x : ℤ) : x % 19 = 12 → x % 95 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_95_equals_12_l1676_167674


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l1676_167641

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 8

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 40

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_to_dops * dops_to_dips)⁻¹ * target_dips = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l1676_167641


namespace NUMINAMATH_CALUDE_abc_product_l1676_167615

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165)
  (h2 : b * (c + a) = 156)
  (h3 : c * (a + b) = 180) :
  a * b * c = 100 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1676_167615


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1676_167678

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1676_167678


namespace NUMINAMATH_CALUDE_magnitude_of_complex_expression_l1676_167655

theorem magnitude_of_complex_expression (z : ℂ) (h : z = 1 - Complex.I) : 
  Complex.abs (1 + Complex.I * z) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_expression_l1676_167655


namespace NUMINAMATH_CALUDE_debby_water_bottles_l1676_167698

theorem debby_water_bottles (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 301)
  (h2 : bottles_per_day = 144)
  (h3 : remaining_bottles = 157) :
  (initial_bottles - remaining_bottles) / bottles_per_day = 1 :=
sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l1676_167698


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1676_167635

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1676_167635


namespace NUMINAMATH_CALUDE_total_bills_count_l1676_167606

/-- Represents the number of bills and their total value -/
structure WalletContents where
  num_five_dollar_bills : ℕ
  num_ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 12 -/
theorem total_bills_count (w : WalletContents) 
  (h1 : w.num_five_dollar_bills = 4)
  (h2 : w.total_value = 100)
  (h3 : w.total_value = 5 * w.num_five_dollar_bills + 10 * w.num_ten_dollar_bills) :
  w.num_five_dollar_bills + w.num_ten_dollar_bills = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_bills_count_l1676_167606


namespace NUMINAMATH_CALUDE_extra_bananas_distribution_l1676_167660

theorem extra_bananas_distribution (total_children absent_children : ℕ) 
  (original_distribution : ℕ) (h1 : total_children = 610) 
  (h2 : absent_children = 305) (h3 : original_distribution = 2) : 
  (total_children * original_distribution) / (total_children - absent_children) - 
   original_distribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_distribution_l1676_167660


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l1676_167656

/-- Represents a die with opposite sides summing to 7 -/
structure Die where
  sides : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents a 4x4x4 cube made of 64 dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Function to calculate the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum of visible faces -/
theorem smallest_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l1676_167656


namespace NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1676_167649

theorem least_coins (n : ℕ) : (n % 7 = 3 ∧ n % 4 = 2) → n ≥ 10 :=
by sorry

theorem ten_coins : (10 % 7 = 3) ∧ (10 % 4 = 2) :=
by sorry

theorem coins_in_wallet : ∃ (n : ℕ), n % 7 = 3 ∧ n % 4 = 2 ∧ ∀ (m : ℕ), (m % 7 = 3 ∧ m % 4 = 2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1676_167649


namespace NUMINAMATH_CALUDE_postal_code_arrangements_l1676_167663

/-- The number of possible arrangements of four distinct digits -/
def fourDigitArrangements : ℕ := 24

/-- The set of digits used in the postal code -/
def postalCodeDigits : Finset ℕ := {2, 3, 5, 8}

/-- Theorem: The number of arrangements of four distinct digits equals 24 -/
theorem postal_code_arrangements :
  Finset.card (Finset.powersetCard 4 postalCodeDigits) = fourDigitArrangements :=
by sorry

end NUMINAMATH_CALUDE_postal_code_arrangements_l1676_167663


namespace NUMINAMATH_CALUDE_polynomial_identity_l1676_167659

theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x, f (x^2 + 1) = x^4 + 6*x^2 + 2) :
  ∀ x, f (x^2 - 3) = x^4 - 2*x^2 - 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1676_167659


namespace NUMINAMATH_CALUDE_condensed_milk_higher_caloric_value_l1676_167677

theorem condensed_milk_higher_caloric_value (a b c : ℝ) : 
  (3*a + 4*b + 2*c > 2*a + 3*b + 4*c) → 
  (3*a + 4*b + 2*c > 4*a + 2*b + 3*c) → 
  b > c := by
sorry

end NUMINAMATH_CALUDE_condensed_milk_higher_caloric_value_l1676_167677


namespace NUMINAMATH_CALUDE_lowercase_count_l1676_167638

/-- Represents the structure of Pat's password -/
structure Password where
  total_length : ℕ
  symbols : ℕ
  lowercase : ℕ
  uppercase_and_numbers : ℕ

/-- Defines the conditions for Pat's password -/
def valid_password (p : Password) : Prop :=
  p.total_length = 14 ∧
  p.symbols = 2 ∧
  p.uppercase_and_numbers = p.lowercase / 2 ∧
  p.total_length = p.lowercase + p.uppercase_and_numbers + p.symbols

/-- Theorem stating that a valid password has 8 lowercase letters -/
theorem lowercase_count (p : Password) (h : valid_password p) : p.lowercase = 8 := by
  sorry

#check lowercase_count

end NUMINAMATH_CALUDE_lowercase_count_l1676_167638


namespace NUMINAMATH_CALUDE_factorization_problems_l1676_167645

theorem factorization_problems :
  (∀ a : ℝ, a^2 - 25 = (a + 5) * (a - 5)) ∧
  (∀ x y : ℝ, 2*x^2*y - 8*x*y + 8*y = 2*y*(x - 2)^2) := by
sorry

end NUMINAMATH_CALUDE_factorization_problems_l1676_167645


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1676_167624

/-- The radius of the inscribed circle in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let area := d1 * d2 / 2
  area / (4 * a) = 30 / Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1676_167624


namespace NUMINAMATH_CALUDE_drug_price_reduction_l1676_167695

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 60)
  (h2 : final_price = 48.6) :
  ∃ (x : ℝ), x = 0.1 ∧ initial_price * (1 - x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l1676_167695


namespace NUMINAMATH_CALUDE_solve_equations_l1676_167662

theorem solve_equations :
  (∃ x : ℝ, 4 * x = 2 * x + 6 ∧ x = 3) ∧
  (∃ x : ℝ, 3 * x + 5 = 6 * x - 1 ∧ x = 2) ∧
  (∃ x : ℝ, 3 * x - 2 * (x - 1) = 2 + 3 * (4 - x) ∧ x = 3) ∧
  (∃ x : ℝ, (x - 3) / 5 - (x + 4) / 2 = -2 ∧ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l1676_167662


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1676_167684

/-- A geometric sequence with first term 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem to be proved -/
theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 7 * a 11 = 100) :
  a 9 = 10 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1676_167684


namespace NUMINAMATH_CALUDE_prime_factor_difference_l1676_167600

theorem prime_factor_difference (n : Nat) (h : n = 278459) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≥ r ∧ r ≥ q) ∧
  p - q = 254 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l1676_167600


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1676_167639

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := L / 2
  let new_area := L * B / 2
  new_length * B = new_area → B = B :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1676_167639


namespace NUMINAMATH_CALUDE_unique_intersection_main_theorem_l1676_167627

/-- The curve C generated by rotating P(t, √(2)t^2 - 2t) by 45° anticlockwise -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 2*p.1*p.2 + p.2^2 - p.1 - 3*p.2 = 0}

/-- The line y = -1/8 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1/8}

/-- The intersection of C and L is a singleton -/
theorem unique_intersection : (C ∩ L).Finite ∧ (C ∩ L).Nonempty := by
  sorry

/-- The main theorem stating that y = -1/8 intersects C at exactly one point -/
theorem main_theorem : ∃! p : ℝ × ℝ, p ∈ C ∧ p ∈ L := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_main_theorem_l1676_167627


namespace NUMINAMATH_CALUDE_spread_combination_exists_l1676_167687

/-- Represents the calories in one piece of bread -/
def bread_calories : ℝ := 100

/-- Represents the calories in one serving of peanut butter -/
def peanut_butter_calories : ℝ := 200

/-- Represents the calories in one serving of strawberry jam -/
def strawberry_jam_calories : ℝ := 120

/-- Represents the calories in one serving of almond butter -/
def almond_butter_calories : ℝ := 180

/-- Represents the total calories needed for breakfast -/
def total_calories : ℝ := 500

/-- Theorem stating that there exist non-negative real numbers p, j, and a
    satisfying the calorie equation and ensuring at least one spread is used -/
theorem spread_combination_exists :
  ∃ (p j a : ℝ), p ≥ 0 ∧ j ≥ 0 ∧ a ≥ 0 ∧
  bread_calories + peanut_butter_calories * p + strawberry_jam_calories * j + almond_butter_calories * a = total_calories ∧
  p + j + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_spread_combination_exists_l1676_167687


namespace NUMINAMATH_CALUDE_water_formed_moles_l1676_167692

/-- Represents a chemical compound -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def naoh : Compound := ⟨"NaOH", 2⟩
def h2so4 : Compound := ⟨"H2SO4", 2⟩

def balanced_reaction : Reaction := {
  reactants := [⟨"NaOH", 2⟩, ⟨"H2SO4", 1⟩],
  products := [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]
}

/-- Calculates the moles of a product formed in a reaction -/
def moles_formed (reaction : Reaction) (product : Compound) (limiting_reactant : Compound) : ℚ :=
  sorry

theorem water_formed_moles :
  moles_formed balanced_reaction ⟨"H2O", 0⟩ naoh = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_moles_l1676_167692


namespace NUMINAMATH_CALUDE_train_length_l1676_167643

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1676_167643


namespace NUMINAMATH_CALUDE_circle_sum_center_radius_l1676_167620

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 8*y - 7 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ (a b r : ℝ), is_center_radius a b r ∧ a + b + r = Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_center_radius_l1676_167620


namespace NUMINAMATH_CALUDE_karlson_expenditure_can_exceed_2000_l1676_167604

theorem karlson_expenditure_can_exceed_2000 :
  ∃ (n m : ℕ), 25 * n + 340 * m > 2000 :=
by sorry

end NUMINAMATH_CALUDE_karlson_expenditure_can_exceed_2000_l1676_167604


namespace NUMINAMATH_CALUDE_root_implies_q_value_l1676_167612

theorem root_implies_q_value (p q : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 : ℂ) * (5 + Complex.I) ^ 2 + p * (5 + Complex.I) + q = 0 →
  q = 78 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l1676_167612


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l1676_167617

/-- Calculates the total profit percentage for a shopkeeper selling apples -/
theorem apple_profit_percentage
  (total_apples : ℝ)
  (percent_sold_at_low_profit : ℝ)
  (percent_sold_at_high_profit : ℝ)
  (low_profit_rate : ℝ)
  (high_profit_rate : ℝ)
  (h1 : total_apples = 280)
  (h2 : percent_sold_at_low_profit = 0.4)
  (h3 : percent_sold_at_high_profit = 0.6)
  (h4 : low_profit_rate = 0.1)
  (h5 : high_profit_rate = 0.3)
  (h6 : percent_sold_at_low_profit + percent_sold_at_high_profit = 1) :
  let cost_price := 1
  let low_profit_quantity := percent_sold_at_low_profit * total_apples
  let high_profit_quantity := percent_sold_at_high_profit * total_apples
  let total_cost := total_apples * cost_price
  let low_profit_revenue := low_profit_quantity * cost_price * (1 + low_profit_rate)
  let high_profit_revenue := high_profit_quantity * cost_price * (1 + high_profit_rate)
  let total_revenue := low_profit_revenue + high_profit_revenue
  let total_profit := total_revenue - total_cost
  let profit_percentage := (total_profit / total_cost) * 100
  profit_percentage = 22 := by sorry

end NUMINAMATH_CALUDE_apple_profit_percentage_l1676_167617


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l1676_167670

theorem ceiling_floor_difference (x : ℝ) : 
  (⌈x⌉ : ℝ) + (⌊x⌋ : ℝ) = 2 * x → (⌈x⌉ : ℝ) - (⌊x⌋ : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l1676_167670


namespace NUMINAMATH_CALUDE_no_movement_after_n_commands_l1676_167657

/-- Represents the state of the line of children -/
inductive ChildState
  | Boy
  | Girl

/-- Represents a line of children -/
def ChildLine := List ChildState

/-- Swaps adjacent boy-girl pairs in the line -/
def swapAdjacent (line : ChildLine) : ChildLine :=
  sorry

/-- Applies the swap command n times -/
def applyNCommands (n : Nat) (line : ChildLine) : ChildLine :=
  sorry

/-- Checks if any more swaps are possible -/
def canSwap (line : ChildLine) : Bool :=
  sorry

/-- Initial line of alternating boys and girls -/
def initialLine (n : Nat) : ChildLine :=
  sorry

theorem no_movement_after_n_commands (n : Nat) :
  canSwap (applyNCommands n (initialLine n)) = false :=
  sorry

end NUMINAMATH_CALUDE_no_movement_after_n_commands_l1676_167657


namespace NUMINAMATH_CALUDE_chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l1676_167632

/-- The probability of a chandelier with 3 parallel-connected bulbs being illuminated, 
    given that the probability of each bulb working properly is 0.7 -/
theorem chandelier_illumination_probability : ℝ :=
  let p : ℝ := 0.7  -- probability of each bulb working properly
  let num_bulbs : ℕ := 3  -- number of bulbs in parallel connection
  1 - (1 - p) ^ num_bulbs

/-- Proof that the probability of the chandelier being illuminated is 0.973 -/
theorem chandelier_illumination_probability_is_correct : 
  chandelier_illumination_probability = 0.973 := by
  sorry


end NUMINAMATH_CALUDE_chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l1676_167632


namespace NUMINAMATH_CALUDE_remaining_milk_james_remaining_milk_l1676_167675

/-- Calculates the remaining milk in ounces and liters after consumption --/
theorem remaining_milk (initial_gallons : ℕ) (ounces_per_gallon : ℕ) 
  (james_consumed : ℕ) (sarah_consumed : ℕ) (mark_consumed : ℕ) 
  (ounce_to_liter : ℝ) : ℕ × ℝ :=
  let initial_ounces := initial_gallons * ounces_per_gallon
  let total_consumed := james_consumed + sarah_consumed + mark_consumed
  let remaining_ounces := initial_ounces - total_consumed
  let remaining_liters := (remaining_ounces : ℝ) * ounce_to_liter
  (remaining_ounces, remaining_liters)

/-- Proves that James has 326 ounces and approximately 9.64 liters of milk left --/
theorem james_remaining_milk :
  remaining_milk 3 128 13 20 25 0.0295735 = (326, 9.641051) :=
by sorry

end NUMINAMATH_CALUDE_remaining_milk_james_remaining_milk_l1676_167675
