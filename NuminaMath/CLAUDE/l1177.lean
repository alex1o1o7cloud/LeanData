import Mathlib

namespace NUMINAMATH_CALUDE_grocery_cost_l1177_117765

-- Define the prices and quantities of groceries
def milk_price : ℚ := 3
def cereal_price : ℚ := 7/2
def banana_price : ℚ := 1/4
def apple_price : ℚ := 1/2
def cereal_quantity : ℕ := 2
def banana_quantity : ℕ := 4
def apple_quantity : ℕ := 4
def cookie_quantity : ℕ := 2

-- Define the total cost of groceries
def total_cost : ℚ :=
  milk_price +
  cereal_price * cereal_quantity +
  banana_price * banana_quantity +
  apple_price * apple_quantity +
  (2 * milk_price) * cookie_quantity

-- Theorem statement
theorem grocery_cost : total_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_l1177_117765


namespace NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_complement_B_l1177_117788

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5 ∨ x = 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- State the theorems to be proved
theorem complement_A : (Set.univ \ A) = {x | x ≤ -1 ∨ (5 < x ∧ x < 6) ∨ x > 6} := by sorry

theorem complement_B : (Set.univ \ B) = {x | x < 2 ∨ x ≥ 5} := by sorry

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 < x ∧ x < 2 ∨ x = 5 ∨ x = 6} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_complement_B_l1177_117788


namespace NUMINAMATH_CALUDE_slips_with_two_l1177_117711

theorem slips_with_two (total : ℕ) (expected_value : ℚ) : 
  total = 15 → expected_value = 46/10 → ∃ x y z : ℕ, 
    x + y + z = total ∧ 
    (2 * x + 5 * y + 8 * z : ℚ) / total = expected_value ∧ 
    x = 8 ∧ y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_slips_with_two_l1177_117711


namespace NUMINAMATH_CALUDE_polynomial_divisible_by_nine_l1177_117742

theorem polynomial_divisible_by_nine (n : ℤ) : ∃ k : ℤ, n^6 - 3*n^5 + 4*n^4 - 3*n^3 + 4*n^2 - 3*n = 9*k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisible_by_nine_l1177_117742


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l1177_117720

/-- The number of ways to put distinguishable balls into distinguishable boxes -/
def ways_to_distribute (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 81 ways to put 4 distinguishable balls into 3 distinguishable boxes -/
theorem four_balls_three_boxes : ways_to_distribute 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l1177_117720


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1177_117794

theorem cube_volume_ratio (edge1 edge2 : ℝ) (h : edge2 = 6 * edge1) :
  (edge1^3) / (edge2^3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1177_117794


namespace NUMINAMATH_CALUDE_range_of_ratio_l1177_117784

theorem range_of_ratio (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) : 
  ∃ (k : ℝ), k = |y / (x + 1)| ∧ k ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_ratio_l1177_117784


namespace NUMINAMATH_CALUDE_rex_saved_100_nickels_l1177_117743

/-- Represents the number of coins of each type saved by the children -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Converts a number of coins to their value in cents -/
def coinsToCents (s : Savings) : ℕ :=
  s.pennies + 5 * s.nickels + 10 * s.dimes

/-- The main theorem: Given the conditions, Rex saved 100 nickels -/
theorem rex_saved_100_nickels (s : Savings) 
    (h1 : s.pennies = 200)
    (h2 : s.dimes = 330)
    (h3 : coinsToCents s = 4000) : 
  s.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_rex_saved_100_nickels_l1177_117743


namespace NUMINAMATH_CALUDE_other_group_cleaned_area_l1177_117795

theorem other_group_cleaned_area
  (total_area : ℕ)
  (lizzies_group_area : ℕ)
  (remaining_area : ℕ)
  (h1 : total_area = 900)
  (h2 : lizzies_group_area = 250)
  (h3 : remaining_area = 385) :
  total_area - remaining_area - lizzies_group_area = 265 :=
by sorry

end NUMINAMATH_CALUDE_other_group_cleaned_area_l1177_117795


namespace NUMINAMATH_CALUDE_annie_mike_toy_ratio_l1177_117766

/-- Represents the number of toys each person has -/
structure ToyCount where
  annie : ℕ
  mike : ℕ
  tom : ℕ

/-- Given the conditions of the problem, proves that the ratio of Annie's toys to Mike's toys is 4:1 -/
theorem annie_mike_toy_ratio 
  (tc : ToyCount) 
  (mike_toys : tc.mike = 6)
  (annie_multiple : ∃ k : ℕ, tc.annie = k * tc.mike)
  (annie_less_than_tom : tc.annie = tc.tom - 2)
  (total_toys : tc.annie + tc.mike + tc.tom = 56) :
  tc.annie / tc.mike = 4 := by
  sorry

#check annie_mike_toy_ratio

end NUMINAMATH_CALUDE_annie_mike_toy_ratio_l1177_117766


namespace NUMINAMATH_CALUDE_papayas_theorem_l1177_117780

def remaining_green_papayas (initial : ℕ) (friday_yellow : ℕ) : ℕ :=
  initial - friday_yellow - (2 * friday_yellow)

theorem papayas_theorem :
  remaining_green_papayas 14 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_papayas_theorem_l1177_117780


namespace NUMINAMATH_CALUDE_root_sum_squares_l1177_117718

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (a-b)^2 + (b-c)^2 + (c-a)^2 = 125 := by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1177_117718


namespace NUMINAMATH_CALUDE_goose_survival_fraction_l1177_117797

theorem goose_survival_fraction (total_eggs : ℕ) 
  (hatch_rate : ℚ) (first_month_survival_rate : ℚ) (first_year_survivors : ℕ) :
  hatch_rate = 1/2 →
  first_month_survival_rate = 3/4 →
  first_year_survivors = 120 →
  (hatch_rate * first_month_survival_rate * total_eggs : ℚ) = first_year_survivors →
  (first_year_survivors : ℚ) / (hatch_rate * first_month_survival_rate * total_eggs) = 1 :=
by sorry

end NUMINAMATH_CALUDE_goose_survival_fraction_l1177_117797


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1177_117739

theorem geometric_sequence_sum (a₁ a₂ a₃ a₄ a₅ : ℕ) (q : ℚ) :
  (a₁ > 0) →
  (a₂ > a₁) → (a₃ > a₂) → (a₄ > a₃) → (a₅ > a₄) →
  (a₂ = a₁ * q) → (a₃ = a₂ * q) → (a₄ = a₃ * q) → (a₅ = a₄ * q) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = 211) →
  (a₁ = 16 ∧ q = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1177_117739


namespace NUMINAMATH_CALUDE_angela_height_l1177_117703

/-- Given the heights of five people with specific relationships, prove Angela's height. -/
theorem angela_height (carl becky amy helen angela : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = becky * 1.2)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4) :
  angela = 295 := by
  sorry

end NUMINAMATH_CALUDE_angela_height_l1177_117703


namespace NUMINAMATH_CALUDE_number_percentage_problem_l1177_117721

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l1177_117721


namespace NUMINAMATH_CALUDE_lavinia_son_older_than_daughter_l1177_117717

/-- Given information about the ages of Lavinia's and Katie's children, prove that Lavinia's son is 21 years older than Lavinia's daughter. -/
theorem lavinia_son_older_than_daughter :
  ∀ (lavinia_daughter lavinia_son katie_daughter katie_son : ℕ),
  lavinia_daughter = katie_daughter / 3 →
  lavinia_son = 2 * katie_daughter →
  lavinia_daughter + lavinia_son = 2 * katie_daughter + 5 →
  katie_daughter = 12 →
  katie_son + 3 = lavinia_son →
  lavinia_son - lavinia_daughter = 21 :=
by sorry

end NUMINAMATH_CALUDE_lavinia_son_older_than_daughter_l1177_117717


namespace NUMINAMATH_CALUDE_dan_bought_18_stickers_l1177_117769

/-- The number of stickers Dan bought -/
def stickers_bought (initial_stickers : ℕ) : ℕ := 18

theorem dan_bought_18_stickers (initial_stickers : ℕ) :
  let cindy_remaining := initial_stickers - 15
  let dan_total := initial_stickers + stickers_bought initial_stickers
  dan_total = cindy_remaining + 33 :=
by
  sorry

end NUMINAMATH_CALUDE_dan_bought_18_stickers_l1177_117769


namespace NUMINAMATH_CALUDE_value_difference_is_50p_minus_250_l1177_117738

/-- The value of a fifty-cent coin in pennies -/
def fifty_cent_value : ℕ := 50

/-- The number of fifty-cent coins Liam has -/
def liam_coins (p : ℕ) : ℕ := 3 * p + 2

/-- The number of fifty-cent coins Mia has -/
def mia_coins (p : ℕ) : ℕ := 2 * p + 7

/-- The difference in total value (in pennies) between Liam's and Mia's fifty-cent coins -/
def value_difference (p : ℕ) : ℤ := fifty_cent_value * (liam_coins p - mia_coins p)

theorem value_difference_is_50p_minus_250 (p : ℕ) :
  value_difference p = 50 * p - 250 := by sorry

end NUMINAMATH_CALUDE_value_difference_is_50p_minus_250_l1177_117738


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1177_117787

theorem quadratic_roots_sum (u v : ℝ) : 
  (u^2 - 5*u + 6 = 0) → 
  (v^2 - 5*v + 6 = 0) → 
  u^2 + v^2 + u + v = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1177_117787


namespace NUMINAMATH_CALUDE_log_sum_equality_l1177_117714

-- Define the theorem
theorem log_sum_equality (p q : ℝ) (h : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + 2*q) → p = 2*q / (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1177_117714


namespace NUMINAMATH_CALUDE_square_neg_sqrt_three_eq_three_l1177_117792

theorem square_neg_sqrt_three_eq_three : (-Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_neg_sqrt_three_eq_three_l1177_117792


namespace NUMINAMATH_CALUDE_food_distribution_l1177_117747

/-- The initial number of men for whom the food lasts 50 days -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial group -/
def initial_days : ℕ := 50

/-- The number of additional men who join -/
def additional_men : ℕ := 20

/-- The number of days the food lasts after additional men join -/
def new_days : ℕ := 25

/-- Theorem stating that the initial number of men is 20 -/
theorem food_distribution :
  initial_men * initial_days = (initial_men + additional_men) * new_days ∧
  initial_men = 20 := by sorry

end NUMINAMATH_CALUDE_food_distribution_l1177_117747


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1177_117761

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1177_117761


namespace NUMINAMATH_CALUDE_periodic_function_2009_l1177_117768

/-- A function satisfying the given functional equation -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem periodic_function_2009 (f : ℝ → ℝ) 
  (h1 : PeriodicFunction f) 
  (h2 : f 5 = 2 + Real.sqrt 3) : 
  f 2009 = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_2009_l1177_117768


namespace NUMINAMATH_CALUDE_grape_banana_difference_l1177_117764

/-- Represents the number of candy pieces in each jar -/
structure CandyJars where
  peanutButter : ℕ
  grape : ℕ
  banana : ℕ

/-- The conditions of the candy jar problem -/
def candyJarProblem (jars : CandyJars) : Prop :=
  jars.peanutButter = 4 * jars.grape ∧
  jars.grape > jars.banana ∧
  jars.banana = 43 ∧
  jars.peanutButter = 192

/-- The theorem stating the difference between grape and banana jars -/
theorem grape_banana_difference (jars : CandyJars) 
  (h : candyJarProblem jars) : jars.grape - jars.banana = 5 := by
  sorry

end NUMINAMATH_CALUDE_grape_banana_difference_l1177_117764


namespace NUMINAMATH_CALUDE_infinitely_many_common_divisors_l1177_117748

theorem infinitely_many_common_divisors :
  ∀ k : ℕ, ∃ n : ℕ, ∃ d : ℕ, d > 1 ∧ d ∣ (2 * n - 3) ∧ d ∣ (3 * n - 2) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_common_divisors_l1177_117748


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1177_117767

/-- Given a line segment with midpoint (5, -8) and one endpoint at (7, 2),
    the sum of the coordinates of the other endpoint is -15. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 = (x + 7) / 2) →
    (-8 = (y + 2) / 2) →
    x + y = -15 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1177_117767


namespace NUMINAMATH_CALUDE_min_matches_25_players_l1177_117751

/-- Represents a chess tournament. -/
structure ChessTournament where
  numPlayers : ℕ
  skillLevels : Fin numPlayers → ℕ
  uniqueSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The minimum number of matches required to determine the two strongest players. -/
def minMatchesForTopTwo (tournament : ChessTournament) : ℕ :=
  -- Definition to be proved
  28

/-- Theorem stating the minimum number of matches for a 25-player tournament. -/
theorem min_matches_25_players (tournament : ChessTournament) 
  (h_players : tournament.numPlayers = 25) :
  minMatchesForTopTwo tournament = 28 := by
  sorry

#check min_matches_25_players

end NUMINAMATH_CALUDE_min_matches_25_players_l1177_117751


namespace NUMINAMATH_CALUDE_sin_180_degrees_l1177_117774

/-- The sine of 180 degrees is 0. -/
theorem sin_180_degrees : Real.sin (π) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l1177_117774


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1177_117796

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, x^2 + 1 < 2*x) ↔ (∀ x : ℝ, x^2 + 1 ≥ 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1177_117796


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1177_117770

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 28 →
  perimeter = 2 * length + 2 * breadth →
  perimeter = 5300 / 26.5 →
  length = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1177_117770


namespace NUMINAMATH_CALUDE_problem_shape_surface_area_l1177_117762

/-- Represents a solid shape made of unit cubes -/
structure CubeShape where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ :=
  sorry

/-- The specific cube shape described in the problem -/
def problem_shape : CubeShape :=
  { base_length := 4
  , base_width := 3
  , top_length := 3
  , top_width := 1
  , total_cubes := 15
  }

/-- Theorem stating that the surface area of the problem_shape is 36 square units -/
theorem problem_shape_surface_area :
  surface_area problem_shape = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_shape_surface_area_l1177_117762


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l1177_117740

/-- The number of boys in the lineup -/
def num_boys : ℕ := 9

/-- The number of girls in the lineup -/
def num_girls : ℕ := 15

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the lineup -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl adjacency in any given pair -/
def prob_boy_girl_adjacency : ℚ := (2 * (num_boys - 1) * (num_girls - 1)) / ((total_people - 2) * (total_people - 3))

/-- The expected number of boy-girl adjacencies in the lineup -/
def expected_adjacencies : ℚ := num_pairs * prob_boy_girl_adjacency

theorem expected_boy_girl_adjacencies :
  expected_adjacencies = 920 / 77 := by sorry

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l1177_117740


namespace NUMINAMATH_CALUDE_max_value_of_d_l1177_117799

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 10 ∧ 
    a' * b' + a' * c' + a' * d' + b' * c' + b' * d' + c' * d' = 20 ∧
    d' = (5 + Real.sqrt 105) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1177_117799


namespace NUMINAMATH_CALUDE_abcd_equality_l1177_117753

theorem abcd_equality (a b c d : ℝ) 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
  (h2 : a^2 + d^2 = 1)
  (h3 : b^2 + c^2 = 1)
  (h4 : a*c + b*d = 1/3) :
  a*b - c*d = 2*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_abcd_equality_l1177_117753


namespace NUMINAMATH_CALUDE_f_composition_one_sixteenth_l1177_117756

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

theorem f_composition_one_sixteenth : f (f (1/16)) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_sixteenth_l1177_117756


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1177_117777

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 6 * x₁ - 9 = 0) → 
  (3 * x₂^2 + 6 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1177_117777


namespace NUMINAMATH_CALUDE_percentage_not_liking_basketball_is_52_percent_l1177_117700

/-- Represents the school population and basketball preferences --/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculates the percentage of students who don't like basketball --/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_count := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_count := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_playing := male_count * s.male_basketball_ratio
  let female_playing := female_count * s.female_basketball_ratio
  let total_not_playing := s.total_students - (male_playing + female_playing)
  total_not_playing / s.total_students * 100

/-- The main theorem to prove --/
theorem percentage_not_liking_basketball_is_52_percent :
  let s : School := {
    total_students := 1000,
    male_ratio := 3/5,
    female_ratio := 2/5,
    male_basketball_ratio := 2/3,
    female_basketball_ratio := 1/5
  }
  percentage_not_liking_basketball s = 52 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_liking_basketball_is_52_percent_l1177_117700


namespace NUMINAMATH_CALUDE_lcm_of_12_18_30_l1177_117731

theorem lcm_of_12_18_30 : Nat.lcm (Nat.lcm 12 18) 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_18_30_l1177_117731


namespace NUMINAMATH_CALUDE_jessica_cut_orchids_l1177_117732

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 2

/-- The number of orchids in the vase after cutting -/
def final_orchids : ℕ := 21

/-- The number of orchids Jessica cut -/
def orchids_cut : ℕ := final_orchids - initial_orchids

theorem jessica_cut_orchids : orchids_cut = 19 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_orchids_l1177_117732


namespace NUMINAMATH_CALUDE_bug_path_tiles_l1177_117763

/-- Represents the number of tiles visited by a bug walking diagonally across a rectangular grid -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The playground dimensions in tile units -/
def playground_width : ℕ := 6
def playground_length : ℕ := 13

theorem bug_path_tiles :
  tilesVisited playground_width playground_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l1177_117763


namespace NUMINAMATH_CALUDE_expression_value_when_b_is_negative_one_l1177_117778

theorem expression_value_when_b_is_negative_one :
  let b : ℚ := -1
  let expr := (3 * b⁻¹ + (2 * b⁻¹) / 3) / b
  expr = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_when_b_is_negative_one_l1177_117778


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_l1177_117737

/-- Given that m and n are natural numbers satisfying 3n^3 = 5m^2, 
    the smallest possible value of m + n is 60. -/
theorem smallest_m_plus_n : ∃ (m n : ℕ), 
  (3 * n^3 = 5 * m^2) ∧ 
  (m + n = 60) ∧ 
  (∀ (m' n' : ℕ), (3 * n'^3 = 5 * m'^2) → (m' + n' ≥ 60)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_l1177_117737


namespace NUMINAMATH_CALUDE_total_potatoes_l1177_117705

-- Define the number of people sharing the potatoes
def num_people : Nat := 3

-- Define the number of potatoes each person received
def potatoes_per_person : Nat := 8

-- Theorem to prove the total number of potatoes
theorem total_potatoes : num_people * potatoes_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l1177_117705


namespace NUMINAMATH_CALUDE_vanessa_savings_time_l1177_117749

def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def arcade_expense : ℕ := 15
def snack_expense : ℕ := 5

def weekly_savings : ℕ := weekly_allowance - arcade_expense - snack_expense

theorem vanessa_savings_time : 
  ∃ (weeks : ℕ), 
    weeks * weekly_savings + initial_savings ≥ dress_cost ∧ 
    (weeks - 1) * weekly_savings + initial_savings < dress_cost ∧
    weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_savings_time_l1177_117749


namespace NUMINAMATH_CALUDE_martha_family_women_without_daughters_l1177_117776

/-- Represents the family structure of Martha and her descendants -/
structure MarthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : MarthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of women without daughters in Martha's family -/
theorem martha_family_women_without_daughters :
  ∀ f : MarthaFamily,
  f.daughters = 8 →
  f.total_descendants = 40 →
  f.daughters_with_children * 8 = f.total_descendants - f.daughters →
  women_without_daughters f = 36 :=
by sorry

end NUMINAMATH_CALUDE_martha_family_women_without_daughters_l1177_117776


namespace NUMINAMATH_CALUDE_lemonade_sum_l1177_117775

theorem lemonade_sum : 
  let first_intermission : Float := 0.25
  let second_intermission : Float := 0.4166666666666667
  let third_intermission : Float := 0.25
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sum_l1177_117775


namespace NUMINAMATH_CALUDE_log_inequality_l1177_117791

theorem log_inequality (k : ℝ) (h : k ≥ 3) :
  Real.log k / Real.log (k - 1) > Real.log (k + 1) / Real.log k := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1177_117791


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1177_117708

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 580)
  (ratio_eq : x / y = 0.75) : 
  y - x = 83 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1177_117708


namespace NUMINAMATH_CALUDE_four_number_sequence_l1177_117712

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def is_geometric_sequence (b c d : ℝ) : Prop := c * c = b * d

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : is_arithmetic_sequence a b c)
  (h2 : is_geometric_sequence b c d)
  (h3 : a + d = 16)
  (h4 : b + c = 12) :
  ((a, b, c, d) = (0, 4, 8, 16)) ∨ ((a, b, c, d) = (15, 9, 3, 1)) := by
  sorry

end NUMINAMATH_CALUDE_four_number_sequence_l1177_117712


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1177_117779

theorem sum_of_three_numbers (a b c : ℝ) 
  (eq1 : 2 * a + b = 46)
  (eq2 : b + 2 * c = 53)
  (eq3 : 2 * c + a = 29) :
  a + b + c = 146.5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1177_117779


namespace NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1177_117727

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 - 2*t.b*t.c*(Real.cos t.A) = (t.b + t.c)^2

/-- Theorem stating the measure of angle A -/
theorem angle_A_measure (t : Triangle) (h : satisfiesCondition t) : t.A = 2*π/3 := by
  sorry

/-- Theorem stating the range of the perimeter when a = 3 -/
theorem perimeter_range (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 3) :
  6 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2*Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1177_117727


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1177_117789

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  r₁ + r₂ = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1177_117789


namespace NUMINAMATH_CALUDE_percentage_in_quarters_calculation_l1177_117752

/-- Given a collection of coins, calculate the percentage of the total value that is in quarters. -/
def percentageInQuarters (dimes nickels quarters : ℕ) : ℚ :=
  let dimesValue : ℕ := dimes * 10
  let nickelsValue : ℕ := nickels * 5
  let quartersValue : ℕ := quarters * 25
  let totalValue : ℕ := dimesValue + nickelsValue + quartersValue
  (quartersValue : ℚ) / (totalValue : ℚ) * 100

theorem percentage_in_quarters_calculation :
  percentageInQuarters 70 40 30 = 750 / 1650 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_quarters_calculation_l1177_117752


namespace NUMINAMATH_CALUDE_partial_multiplication_reconstruction_l1177_117745

/-- Represents a partially visible digit (0-9 or unknown) -/
inductive PartialDigit
  | Known (n : Fin 10)
  | Unknown

/-- Represents a partially visible number -/
def PartialNumber := List PartialDigit

/-- Represents a multiplication step in the written method -/
structure MultiplicationStep where
  multiplicand : PartialNumber
  multiplier : PartialNumber
  partialProducts : List PartialNumber
  result : PartialNumber

/-- Check if a number matches a partial number -/
def matchesPartial (n : ℕ) (pn : PartialNumber) : Prop := sorry

/-- The main theorem to prove -/
theorem partial_multiplication_reconstruction 
  (step : MultiplicationStep)
  (h1 : step.multiplicand.length = 3)
  (h2 : step.multiplier.length = 3)
  (h3 : matchesPartial 56576 step.result)
  : ∃ (a b : ℕ), 
    a * b = 56500 ∧ 
    matchesPartial a step.multiplicand ∧ 
    matchesPartial b step.multiplier :=
sorry

end NUMINAMATH_CALUDE_partial_multiplication_reconstruction_l1177_117745


namespace NUMINAMATH_CALUDE_equation_condition_l1177_117772

theorem equation_condition (x y z : ℕ) 
  (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (10 * x + y) * (10 * x + z) = 100 * x^2 + 110 * x + y * z ↔ y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l1177_117772


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_angle_measure_l1177_117754

-- Define the circle O
variable (O : ℝ × ℝ)

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define that ABCD is an inscribed quadrilateral of circle O
def is_inscribed_quadrilateral (O A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_angle_measure 
  (h_inscribed : is_inscribed_quadrilateral O A B C D)
  (h_ratio : angle_measure B A D / angle_measure B C D = 4 / 5) :
  angle_measure B A D = 80 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_angle_measure_l1177_117754


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1177_117781

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, x^2 + y^2 ≥ 0)) ↔ (∃ x y : ℝ, x^2 + y^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1177_117781


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l1177_117782

/-- A function f: ℝ → ℝ is decreasing if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The function f(x) = -x³ + x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + a*x

theorem decreasing_function_implies_a_bound :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ -1/3 := by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l1177_117782


namespace NUMINAMATH_CALUDE_triangle_ABC_point_C_l1177_117734

-- Define the points
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (-1, -2)
def D : ℝ × ℝ := (2, 2)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem triangle_ABC_point_C : 
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (5, 6) := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_point_C_l1177_117734


namespace NUMINAMATH_CALUDE_mark_work_hours_l1177_117707

/-- Calculates the number of hours Mark needs to work per week to earn a target amount --/
def hours_per_week (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) : ℚ :=
  let hourly_wage := spring_earnings / (spring_hours_per_week * spring_weeks)
  let total_hours_needed := target_earnings / hourly_wage
  total_hours_needed / target_weeks

theorem mark_work_hours 
  (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) :
  spring_hours_per_week = 35 ∧ 
  spring_weeks = 15 ∧ 
  spring_earnings = 4200 ∧ 
  target_weeks = 50 ∧ 
  target_earnings = 21000 →
  hours_per_week spring_hours_per_week spring_weeks spring_earnings target_weeks target_earnings = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_mark_work_hours_l1177_117707


namespace NUMINAMATH_CALUDE_matrix_transpose_inverse_sum_squares_l1177_117704

theorem matrix_transpose_inverse_sum_squares (p q r s : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  B.transpose = B⁻¹ →
  p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_transpose_inverse_sum_squares_l1177_117704


namespace NUMINAMATH_CALUDE_project_workers_needed_l1177_117710

/-- Represents a construction project with workers -/
structure Project where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completionRatio : ℚ
  
/-- Calculates the minimum number of workers needed to complete the project on schedule -/
def minWorkersNeeded (p : Project) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific project -/
theorem project_workers_needed :
  let p : Project := {
    totalDays := 40,
    elapsedDays := 10,
    initialWorkers := 10,
    completionRatio := 2/5
  }
  minWorkersNeeded p = 5 := by sorry

end NUMINAMATH_CALUDE_project_workers_needed_l1177_117710


namespace NUMINAMATH_CALUDE_late_fee_is_150_l1177_117726

/-- Calculates the late fee for electricity payment -/
def calculate_late_fee (cost_per_watt : ℝ) (watts_used : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - cost_per_watt * watts_used

/-- Proves that the late fee is $150 given the problem conditions -/
theorem late_fee_is_150 :
  let cost_per_watt : ℝ := 4
  let watts_used : ℝ := 300
  let total_paid : ℝ := 1350
  calculate_late_fee cost_per_watt watts_used total_paid = 150 := by
  sorry

end NUMINAMATH_CALUDE_late_fee_is_150_l1177_117726


namespace NUMINAMATH_CALUDE_min_value_sum_squared_over_one_plus_l1177_117759

theorem min_value_sum_squared_over_one_plus (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_one : x + y + z = 1) : 
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_over_one_plus_l1177_117759


namespace NUMINAMATH_CALUDE_double_force_quadruple_power_l1177_117706

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  /-- Initial force applied by a single tugboat -/
  F : ℝ
  /-- Coefficient of water resistance -/
  k : ℝ
  /-- Initial speed of the barge -/
  v : ℝ
  /-- Water resistance is proportional to speed -/
  resistance_prop : F = k * v

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let v' := 2 * scenario.v  -- New speed after doubling force
  let P := scenario.F * scenario.v  -- Initial power
  let P' := (2 * scenario.F) * v'  -- New power after doubling force
  P' = 4 * P := by sorry

end NUMINAMATH_CALUDE_double_force_quadruple_power_l1177_117706


namespace NUMINAMATH_CALUDE_walter_chores_l1177_117771

theorem walter_chores (total_days : ℕ) (normal_pay exceptional_pay : ℚ) 
  (total_earnings : ℚ) (min_exceptional_days : ℕ) :
  total_days = 15 →
  normal_pay = 4 →
  exceptional_pay = 6 →
  total_earnings = 70 →
  min_exceptional_days = 5 →
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days ≥ min_exceptional_days ∧
    exceptional_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l1177_117771


namespace NUMINAMATH_CALUDE_go_pieces_probability_l1177_117709

theorem go_pieces_probability (p_black p_white : ℝ) 
  (h_black : p_black = 1/7)
  (h_white : p_white = 12/35) :
  p_black + p_white = 17/35 := by
  sorry

end NUMINAMATH_CALUDE_go_pieces_probability_l1177_117709


namespace NUMINAMATH_CALUDE_find_divisor_l1177_117757

theorem find_divisor (x : ℝ) (y : ℝ) 
  (h1 : (x - 5) / y = 7)
  (h2 : (x - 6) / 8 = 6) : 
  y = 7 := by sorry

end NUMINAMATH_CALUDE_find_divisor_l1177_117757


namespace NUMINAMATH_CALUDE_midnight_probability_l1177_117728

/-- Represents the words from which letters are selected -/
inductive Word
| ROAD
| LIGHTS
| TIME

/-- Represents the target word MIDNIGHT -/
def targetWord : String := "MIDNIGHT"

/-- Number of letters to select from each word -/
def selectCount (w : Word) : Nat :=
  match w with
  | .ROAD => 2
  | .LIGHTS => 3
  | .TIME => 4

/-- The probability of selecting the required letters from a given word -/
def selectionProbability (w : Word) : Rat :=
  match w with
  | .ROAD => 1 / 3
  | .LIGHTS => 1 / 20
  | .TIME => 1 / 4

/-- The total probability of selecting all required letters -/
def totalProbability : Rat :=
  (selectionProbability .ROAD) * (selectionProbability .LIGHTS) * (selectionProbability .TIME)

theorem midnight_probability : totalProbability = 1 / 240 := by
  sorry

end NUMINAMATH_CALUDE_midnight_probability_l1177_117728


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_and_difference_l1177_117760

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term_and_difference
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ d = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_and_difference_l1177_117760


namespace NUMINAMATH_CALUDE_overtime_pay_rate_ratio_l1177_117724

/-- Proves that the ratio of overtime pay rate to regular pay rate is 2:1 given specific conditions -/
theorem overtime_pay_rate_ratio (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  overtime_hours = 10 →
  (total_pay - regular_rate * regular_hours) / overtime_hours / regular_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_rate_ratio_l1177_117724


namespace NUMINAMATH_CALUDE_driving_time_is_55_minutes_l1177_117713

/-- Calculates the driving time per trip given total moving time, number of trips, and car filling time -/
def driving_time_per_trip (total_time_hours : ℕ) (num_trips : ℕ) (filling_time_minutes : ℕ) : ℕ :=
  let total_time_minutes := total_time_hours * 60
  let total_filling_time := num_trips * filling_time_minutes
  let total_driving_time := total_time_minutes - total_filling_time
  total_driving_time / num_trips

/-- Theorem stating that given the problem conditions, the driving time per trip is 55 minutes -/
theorem driving_time_is_55_minutes :
  driving_time_per_trip 7 6 15 = 55 := by
  sorry

#eval driving_time_per_trip 7 6 15

end NUMINAMATH_CALUDE_driving_time_is_55_minutes_l1177_117713


namespace NUMINAMATH_CALUDE_integer_floor_equation_l1177_117773

theorem integer_floor_equation (m n : ℕ+) :
  (⌊(m : ℝ)^2 / n⌋ + ⌊(n : ℝ)^2 / m⌋ = ⌊(m : ℝ) / n + (n : ℝ) / m⌋ + m * n) ↔
  (∃ k : ℕ+, (m = k ∧ n = k^2 + 1) ∨ (m = k^2 + 1 ∧ n = k)) :=
sorry

end NUMINAMATH_CALUDE_integer_floor_equation_l1177_117773


namespace NUMINAMATH_CALUDE_stevens_grapes_l1177_117783

def apple_seeds : ℕ := 6
def pear_seeds : ℕ := 2
def grape_seeds : ℕ := 3
def total_seeds_needed : ℕ := 60
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

theorem stevens_grapes (grapes_set_aside : ℕ) : grapes_set_aside = 9 := by
  sorry

#check stevens_grapes

end NUMINAMATH_CALUDE_stevens_grapes_l1177_117783


namespace NUMINAMATH_CALUDE_parallel_line_plane_intersection_not_always_parallel_l1177_117729

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between lines and planes
variable (parallelLP : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLL : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersectionPP : Plane → Plane → Line)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane_intersection_not_always_parallel 
  (α β : Plane) (m n : Line) : 
  ∃ (α β : Plane) (m n : Line), 
    α ≠ β ∧ m ≠ n ∧ 
    parallelLP m α ∧ 
    intersectionPP α β = n ∧ 
    ¬(parallelLL m n) := by sorry

end NUMINAMATH_CALUDE_parallel_line_plane_intersection_not_always_parallel_l1177_117729


namespace NUMINAMATH_CALUDE_class_factory_arrangements_l1177_117741

/-- The number of classes -/
def num_classes : ℕ := 5

/-- The number of factories -/
def num_factories : ℕ := 4

/-- The number of ways to arrange classes into factories -/
def arrangements : ℕ := 240

/-- Theorem stating the number of arrangements -/
theorem class_factory_arrangements :
  (∀ (arrangement : Fin num_classes → Fin num_factories),
    (∀ f : Fin num_factories, ∃ c : Fin num_classes, arrangement c = f) →
    (∀ c : Fin num_classes, arrangement c < num_factories)) →
  arrangements = 240 :=
sorry

end NUMINAMATH_CALUDE_class_factory_arrangements_l1177_117741


namespace NUMINAMATH_CALUDE_infinite_non_representable_l1177_117730

/-- A natural number is representable if it can be written as p + n^(2k) for some prime p and natural numbers n and k. -/
def Representable (m : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ m = p + n^(2*k)

/-- The set of non-representable natural numbers is infinite. -/
theorem infinite_non_representable :
  {m : ℕ | ¬Representable m}.Infinite :=
sorry

end NUMINAMATH_CALUDE_infinite_non_representable_l1177_117730


namespace NUMINAMATH_CALUDE_functions_for_12_functions_for_2007_functions_for_2_pow_2007_l1177_117716

-- Define the functions
def φ : ℕ → ℕ := sorry
def σ : ℕ → ℕ := sorry
def τ : ℕ → ℕ := sorry

-- Theorem for n = 12
theorem functions_for_12 :
  φ 12 = 4 ∧ σ 12 = 28 ∧ τ 12 = 6 := by sorry

-- Theorem for n = 2007
theorem functions_for_2007 :
  φ 2007 = 1332 ∧ σ 2007 = 2912 ∧ τ 2007 = 6 := by sorry

-- Theorem for n = 2^2007
theorem functions_for_2_pow_2007 :
  φ (2^2007) = 2^2006 ∧ 
  σ (2^2007) = 2^2008 - 1 ∧ 
  τ (2^2007) = 2008 := by sorry

end NUMINAMATH_CALUDE_functions_for_12_functions_for_2007_functions_for_2_pow_2007_l1177_117716


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1177_117790

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + 1)) →
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1177_117790


namespace NUMINAMATH_CALUDE_time_saved_without_tide_change_l1177_117735

/-- The time saved by a rower if the tide direction had not changed -/
theorem time_saved_without_tide_change 
  (speed_with_tide : ℝ) 
  (speed_against_tide : ℝ) 
  (distance_after_reversal : ℝ) 
  (h1 : speed_with_tide = 5)
  (h2 : speed_against_tide = 4)
  (h3 : distance_after_reversal = 40) : 
  distance_after_reversal / speed_with_tide - distance_after_reversal / speed_against_tide = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_saved_without_tide_change_l1177_117735


namespace NUMINAMATH_CALUDE_committee_age_difference_l1177_117785

/-- Proves that the age difference between an old and new member in a committee is 40 years,
    given specific conditions about the committee's average age over time. -/
theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 10 → -- The committee has 10 members
  n * A = n * A + n * 4 - (O - N) → -- The total age after 4 years minus the age difference equals the original total age
  O - N = 40 := by
  sorry

end NUMINAMATH_CALUDE_committee_age_difference_l1177_117785


namespace NUMINAMATH_CALUDE_rhett_salary_l1177_117786

/-- Rhett's monthly salary calculation --/
theorem rhett_salary (monthly_rent : ℝ) (tax_rate : ℝ) (late_payments : ℕ) 
  (after_tax_fraction : ℝ) (salary : ℝ) :
  monthly_rent = 1350 →
  tax_rate = 0.1 →
  late_payments = 2 →
  after_tax_fraction = 3/5 →
  after_tax_fraction * (1 - tax_rate) * salary = late_payments * monthly_rent →
  salary = 5000 := by
sorry

end NUMINAMATH_CALUDE_rhett_salary_l1177_117786


namespace NUMINAMATH_CALUDE_initial_children_count_l1177_117701

/-- The number of children who got off the bus -/
def children_off : ℕ := 22

/-- The number of children left on the bus after some got off -/
def children_left : ℕ := 21

/-- The initial number of children on the bus -/
def initial_children : ℕ := children_off + children_left

theorem initial_children_count : initial_children = 43 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_count_l1177_117701


namespace NUMINAMATH_CALUDE_binomial_512_512_l1177_117733

theorem binomial_512_512 : Nat.choose 512 512 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_512_512_l1177_117733


namespace NUMINAMATH_CALUDE_q_at_zero_l1177_117798

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between r, p, and q
axiom r_eq_p_mul_q : r = p * q

-- Define the constant term of p(x)
axiom p_const_term : p.coeff 0 = 6

-- Define the constant term of r(x)
axiom r_const_term : r.coeff 0 = -18

-- The theorem to prove
theorem q_at_zero : q.eval 0 = -3 := by sorry

end NUMINAMATH_CALUDE_q_at_zero_l1177_117798


namespace NUMINAMATH_CALUDE_opposite_of_negative_2016_l1177_117719

theorem opposite_of_negative_2016 : Int.neg (-2016) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2016_l1177_117719


namespace NUMINAMATH_CALUDE_mutual_acquaintance_exists_l1177_117755

/-- Represents a diplomatic reception with a fixed number of participants. -/
structure DiplomaticReception where
  participants : Nat
  heardOf : Nat → Nat → Prop
  heardOfCount : Nat → Nat

/-- The minimum number of people each participant has heard of that guarantees mutual acquaintance. -/
def minHeardOfCount : Nat := 50

/-- Theorem stating that if each participant has heard of at least 50 others,
    there must be a pair who have heard of each other. -/
theorem mutual_acquaintance_exists (reception : DiplomaticReception)
    (h1 : reception.participants = 99)
    (h2 : ∀ i, i < reception.participants → reception.heardOfCount i ≥ minHeardOfCount)
    (h3 : ∀ i j, i < reception.participants → j < reception.participants → 
         reception.heardOf i j → reception.heardOfCount i > 0) :
    ∃ i j, i < reception.participants ∧ j < reception.participants ∧ 
    i ≠ j ∧ reception.heardOf i j ∧ reception.heardOf j i := by
  sorry

end NUMINAMATH_CALUDE_mutual_acquaintance_exists_l1177_117755


namespace NUMINAMATH_CALUDE_problem1_l1177_117723

theorem problem1 (a b : ℝ) : 3 * (a^2 - a*b) - 5 * (a*b + 2*a^2 - 1) = -7*a^2 - 8*a*b + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l1177_117723


namespace NUMINAMATH_CALUDE_platform_length_l1177_117758

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, prove that the platform length is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 27 →
  pole_time = 18 →
  ∃ (platform_length : ℝ),
    platform_length = 150 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1177_117758


namespace NUMINAMATH_CALUDE_sine_graph_translation_l1177_117736

theorem sine_graph_translation (x : ℝ) :
  5 * Real.sin (2 * (x + π/12) + π/6) = 5 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_translation_l1177_117736


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1177_117725

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 135000

/-- The proposed scientific notation representation -/
def scientific_form : ScientificNotation := {
  coefficient := 1.35
  exponent := 5
  coeff_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1177_117725


namespace NUMINAMATH_CALUDE_plane_perpendicular_from_line_l1177_117702

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_from_line
  (α β γ : Plane) (l : Line)
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular_line_plane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_from_line_l1177_117702


namespace NUMINAMATH_CALUDE_petya_prize_probability_at_least_one_prize_probability_l1177_117722

-- Define the number of players
def num_players : ℕ := 10

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Theorem for Petya's probability of winning a prize
theorem petya_prize_probability :
  (5 / 6 : ℚ) ^ (num_players - 1) = (5 / 6 : ℚ) ^ 9 := by sorry

-- Theorem for the probability of at least one player winning a prize
theorem at_least_one_prize_probability :
  1 - (1 / die_sides : ℚ) ^ (num_players - 1) = 1 - (1 / 6 : ℚ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_petya_prize_probability_at_least_one_prize_probability_l1177_117722


namespace NUMINAMATH_CALUDE_set_intersection_range_l1177_117746

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B : Set ℝ := {x | x < -1 ∨ x > 16}
  A ∩ B = A → a < 6 ∨ a > 7.5 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_range_l1177_117746


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1177_117744

theorem complex_fraction_simplification :
  let numerator := (11/4) / ((11/10) + (10/3))
  let denominator := 5/2 - 4/3
  let left_fraction := numerator / denominator
  let right_fraction := 5/7 - ((13/6 + 9/2) * 3/8) / (11/4 - 3/2)
  left_fraction / right_fraction = -35/9 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1177_117744


namespace NUMINAMATH_CALUDE_grid_toothpick_count_l1177_117793

/-- Calculates the number of toothpicks in a rectangular grid with a missing row and column -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_lines := height
  let vertical_lines := width
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * (height - 1)
  horizontal_toothpicks + vertical_toothpicks

/-- Theorem stating that a 25x15 grid with a missing row and column uses 735 toothpicks -/
theorem grid_toothpick_count : toothpick_count 25 15 = 735 := by
  sorry

#eval toothpick_count 25 15

end NUMINAMATH_CALUDE_grid_toothpick_count_l1177_117793


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l1177_117715

theorem square_difference_equals_one (a b : ℝ) (h : a - b = 1) :
  a^2 - b^2 - 2*b = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l1177_117715


namespace NUMINAMATH_CALUDE_school_garden_flowers_l1177_117750

theorem school_garden_flowers (total : ℕ) (yellow : ℕ) : 
  total = 96 → yellow = 12 → ∃ (green : ℕ), 
    green + 3 * green + (total / 2) + yellow = total ∧ green = 9 := by
  sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l1177_117750
