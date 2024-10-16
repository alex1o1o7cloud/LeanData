import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l2622_262242

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l2622_262242


namespace NUMINAMATH_CALUDE_arrangements_remainder_l2622_262251

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The maximum number of red marbles that satisfies the equal neighbor condition -/
def max_red_marbles : ℕ := 23

/-- The number of possible arrangements -/
def num_arrangements : ℕ := 490314

/-- The theorem stating the remainder when the number of arrangements is divided by 1000 -/
theorem arrangements_remainder :
  num_arrangements % 1000 = 314 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_remainder_l2622_262251


namespace NUMINAMATH_CALUDE_fence_cost_for_square_plot_l2622_262202

theorem fence_cost_for_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 81) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 2088 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_for_square_plot_l2622_262202


namespace NUMINAMATH_CALUDE_scalene_triangle_not_unique_l2622_262288

/-- Represents a scalene triangle -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties of a scalene triangle here
  -- as it's not relevant for this particular proof

/-- Represents the circumscribed circle of a triangle -/
structure CircumscribedCircle where
  radius : ℝ

/-- States that a scalene triangle is not uniquely determined by two of its angles
    and the radius of its circumscribed circle -/
theorem scalene_triangle_not_unique (α β : ℝ) (r : CircumscribedCircle) :
  ∃ (t1 t2 : ScaleneTriangle), t1 ≠ t2 ∧
  (∃ (γ1 γ2 : ℝ), α + β + γ1 = π ∧ α + β + γ2 = π) :=
sorry

end NUMINAMATH_CALUDE_scalene_triangle_not_unique_l2622_262288


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_equality_l2622_262269

theorem sphere_volume_surface_area_equality (r : ℝ) (h : r > 0) :
  (4 / 3 : ℝ) * Real.pi * r^3 = 36 * Real.pi → 4 * Real.pi * r^2 = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_equality_l2622_262269


namespace NUMINAMATH_CALUDE_problem_statement_l2622_262226

open Real

-- Define the propositions
def p : Prop := ∀ x, cos (2*x - π/5) = cos (2*(x - π/5))

def q : Prop := ∀ α, tan α = 2 → (cos α)^2 - 2*(sin α)^2 = -7/4 * sin (2*α)

-- State the theorem
theorem problem_statement : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2622_262226


namespace NUMINAMATH_CALUDE_g_monotonic_intervals_exactly_two_tangent_points_l2622_262283

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else -x^2 + 2*x - 1/2

-- Define g(x) = x * f(x)
noncomputable def g (x : ℝ) : ℝ := x * f x

-- Theorem for monotonic intervals of g(x)
theorem g_monotonic_intervals :
  (∀ x y, x < y ∧ y < -1 → g y < g x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y ≤ 0 → g x < g y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < (4 - Real.sqrt 10) / 6 → g y < g x) ∧
  (∀ x y, (4 - Real.sqrt 10) / 6 < x ∧ x < y ∧ y < (4 + Real.sqrt 10) / 6 → g x < g y) ∧
  (∀ x y, (4 + Real.sqrt 10) / 6 < x ∧ x < y → g y < g x) :=
sorry

-- Theorem for existence of exactly two tangent points
theorem exactly_two_tangent_points :
  ∃! (x₁ x₂ : ℝ), x₁ < x₂ ∧
    ∃ (m b : ℝ), 
      (∀ x, f x ≤ m * x + b) ∧
      f x₁ = m * x₁ + b ∧
      f x₂ = m * x₂ + b :=
sorry

end NUMINAMATH_CALUDE_g_monotonic_intervals_exactly_two_tangent_points_l2622_262283


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2622_262209

theorem divisibility_by_three (a : ℤ) : ¬(3 ∣ a) → (3 ∣ (5 * a^2 + 1)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2622_262209


namespace NUMINAMATH_CALUDE_max_c_value_l2622_262252

theorem max_c_value (a b : ℝ) (h : a + 2*b = 2) : 
  ∃ c_max : ℝ, c_max = 3 ∧ 
  (∀ c : ℝ, (3:ℝ)^a + (9:ℝ)^b ≥ c^2 - c → c ≤ c_max) ∧
  ((3:ℝ)^a + (9:ℝ)^b ≥ c_max^2 - c_max) :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l2622_262252


namespace NUMINAMATH_CALUDE_walking_scenario_l2622_262284

def distance_between_people (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) : ℝ :=
  initial_distance + person1_movement - person2_movement

theorem walking_scenario (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) :
  initial_distance = 400 ∧ person1_movement = 200 ∧ person2_movement = -200 →
  distance_between_people initial_distance person1_movement person2_movement = 400 :=
by
  sorry

#check walking_scenario

end NUMINAMATH_CALUDE_walking_scenario_l2622_262284


namespace NUMINAMATH_CALUDE_different_color_probability_l2622_262268

/-- Given 6 cards with 3 red and 3 yellow, the probability of drawing 2 cards of different colors is 3/5 -/
theorem different_color_probability (total_cards : Nat) (red_cards : Nat) (yellow_cards : Nat) :
  total_cards = 6 →
  red_cards = 3 →
  yellow_cards = 3 →
  (Nat.choose red_cards 1 * Nat.choose yellow_cards 1 : Rat) / Nat.choose total_cards 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2622_262268


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l2622_262239

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingSystem :=
  (total_voters : ℕ)
  (num_districts : ℕ)
  (precincts_per_district : ℕ)
  (voters_per_precinct : ℕ)
  (h_total : total_voters = num_districts * precincts_per_district * voters_per_precinct)

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingSystem) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win := (vs.precincts_per_district + 1) / 2
  let voters_to_win_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win * voters_to_win_precinct

/-- The theorem stating the minimum number of voters required for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingSystem) 
  (h_voters : vs.total_voters = 135)
  (h_districts : vs.num_districts = 5)
  (h_precincts : vs.precincts_per_district = 9)
  (h_voters_per_precinct : vs.voters_per_precinct = 3) :
  min_voters_to_win vs = 30 := by
  sorry

#eval min_voters_to_win { total_voters := 135, num_districts := 5, precincts_per_district := 9, voters_per_precinct := 3, h_total := rfl }

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l2622_262239


namespace NUMINAMATH_CALUDE_children_ages_l2622_262207

theorem children_ages (total_age_first_birth total_age_third_birth total_age_children : ℕ)
  (h1 : total_age_first_birth = 45)
  (h2 : total_age_third_birth = 70)
  (h3 : total_age_children = 14) :
  ∃ (age1 age2 age3 : ℕ),
    age1 = 8 ∧ age2 = 5 ∧ age3 = 1 ∧
    age1 + age2 + age3 = total_age_children :=
by
  sorry


end NUMINAMATH_CALUDE_children_ages_l2622_262207


namespace NUMINAMATH_CALUDE_only_f1_is_even_l2622_262211

-- Define the functions
def f1 (x : ℝ) : ℝ := x^2 - 3*abs x + 2
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x - 1

-- Define the domain for f2
def f2_domain : Set ℝ := Set.Ioc (-2) 2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem only_f1_is_even :
  is_even f1 ∧ ¬(is_even f2) ∧ ¬(is_even f3) ∧ ¬(is_even f4) :=
sorry

end NUMINAMATH_CALUDE_only_f1_is_even_l2622_262211


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l2622_262215

/-- Given a quadratic function f(x) = a x^2 - c, prove that if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20. -/
theorem quadratic_function_bounds (a c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = a * x^2 - c)
  (h_bound1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
  (h_bound2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l2622_262215


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l2622_262224

theorem nested_subtraction_simplification (z : ℝ) :
  1 - (2 - (3 - (4 - (5 - z)))) = 3 - z := by sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l2622_262224


namespace NUMINAMATH_CALUDE_g_of_6_eq_0_l2622_262274

/-- The polynomial g(x) = 3x^4 - 18x^3 + 31x^2 - 29x - 72 -/
def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

/-- Theorem: g(6) = 0 -/
theorem g_of_6_eq_0 : g 6 = 0 := by sorry

end NUMINAMATH_CALUDE_g_of_6_eq_0_l2622_262274


namespace NUMINAMATH_CALUDE_zara_goats_l2622_262266

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of groups for transportation -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals -/
def total_animals : ℕ := num_groups * animals_per_group

/-- The number of goats Zara owns -/
def num_goats : ℕ := total_animals - (num_cows + num_sheep)

theorem zara_goats : num_goats = 113 := by
  sorry

end NUMINAMATH_CALUDE_zara_goats_l2622_262266


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l2622_262247

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def zero_at (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

def count_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ → Prop
  | 0 => True
  | n + 1 => ∃ x, a ≤ x ∧ x ≤ b ∧ zero_at f x ∧ count_zeros f a b n

theorem min_zeros_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3) 
  (h_zero : zero_at f 2) : 
  count_zeros f (-3) 3 9 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l2622_262247


namespace NUMINAMATH_CALUDE_eva_total_score_2019_l2622_262200

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year 2019 -/
structure YearScores where
  firstSemester : SemesterScores
  secondSemester : SemesterScores

/-- Theorem stating Eva's total score for 2019 -/
theorem eva_total_score_2019 (scores : YearScores) : 
  totalScore scores.firstSemester + totalScore scores.secondSemester = 485 :=
  by
  have h1 : scores.firstSemester.maths = scores.secondSemester.maths + 10 := by sorry
  have h2 : scores.firstSemester.arts = scores.secondSemester.arts - 15 := by sorry
  have h3 : scores.firstSemester.science = scores.secondSemester.science - scores.secondSemester.science / 3 := by sorry
  have h4 : scores.secondSemester.maths = 80 := by sorry
  have h5 : scores.secondSemester.arts = 90 := by sorry
  have h6 : scores.secondSemester.science = 90 := by sorry
  sorry

end NUMINAMATH_CALUDE_eva_total_score_2019_l2622_262200


namespace NUMINAMATH_CALUDE_greatest_possible_award_l2622_262291

theorem greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) (min_award : ℝ) 
  (prize_fraction : ℝ) (winner_fraction : ℝ) :
  total_prize = 400 →
  num_winners = 20 →
  min_award = 20 →
  prize_fraction = 2/5 →
  winner_fraction = 3/5 →
  ∃ (max_award : ℝ), 
    max_award = 100 ∧ 
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    (∀ (award : ℝ), 
      award ≤ total_prize ∧ 
      award ≥ min_award → 
      award ≤ max_award) ∧
    (prize_fraction * total_prize ≤ winner_fraction * num_winners * min_award) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l2622_262291


namespace NUMINAMATH_CALUDE_unique_valid_n_l2622_262229

/-- The set of numbers {1, 16, 27} -/
def S : Finset ℕ := {1, 16, 27}

/-- Condition: The product of any two distinct members of S increased by 9 is a perfect square -/
axiom distinct_product_square (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ∃ k : ℕ, a * b + 9 = k^2

/-- Definition: n is a positive integer for which n+9, 16n+9, and 27n+9 are perfect squares -/
def is_valid (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∃ k : ℕ, n + 9 = k^2) ∧ 
  (∃ l : ℕ, 16 * n + 9 = l^2) ∧ 
  (∃ m : ℕ, 27 * n + 9 = m^2)

/-- Theorem: 280 is the unique positive integer satisfying the conditions -/
theorem unique_valid_n : 
  is_valid 280 ∧ ∀ n : ℕ, is_valid n → n = 280 :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_n_l2622_262229


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2622_262221

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2622_262221


namespace NUMINAMATH_CALUDE_f_max_value_l2622_262259

/-- The function f(x) defined as |x+2017| - |x-2016| -/
def f (x : ℝ) := |x + 2017| - |x - 2016|

/-- Theorem stating that the maximum value of f(x) is 4033 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 4033 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l2622_262259


namespace NUMINAMATH_CALUDE_simplify_expression_l2622_262222

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2622_262222


namespace NUMINAMATH_CALUDE_sugar_calculation_l2622_262256

/-- The amount of sugar needed for one chocolate bar in grams -/
def sugar_per_bar : ℝ := 1.5

/-- The number of chocolate bars produced per minute -/
def bars_per_minute : ℕ := 36

/-- The number of minutes of production -/
def production_time : ℕ := 2

/-- Calculates the total amount of sugar used in grams -/
def total_sugar_used : ℝ := sugar_per_bar * bars_per_minute * production_time

theorem sugar_calculation :
  total_sugar_used = 108 := by sorry

end NUMINAMATH_CALUDE_sugar_calculation_l2622_262256


namespace NUMINAMATH_CALUDE_no_global_minimum_and_local_minimum_at_one_l2622_262296

noncomputable def f (c : ℝ) : ℝ := c^3 + (3/2)*c^2 - 6*c + 4

theorem no_global_minimum_and_local_minimum_at_one :
  (∀ m : ℝ, ∃ c : ℝ, f c < m) ∧
  (∃ δ : ℝ, δ > 0 ∧ ∀ c : ℝ, c ≠ 1 → |c - 1| < δ → f c > f 1) :=
sorry

end NUMINAMATH_CALUDE_no_global_minimum_and_local_minimum_at_one_l2622_262296


namespace NUMINAMATH_CALUDE_f_max_value_l2622_262271

-- Define the function
def f (t : ℝ) : ℝ := -6 * t^2 + 36 * t - 18

-- State the theorem
theorem f_max_value :
  (∃ (t_max : ℝ), ∀ (t : ℝ), f t ≤ f t_max) ∧
  (∃ (t_max : ℝ), f t_max = 36) ∧
  (f 3 = 36) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2622_262271


namespace NUMINAMATH_CALUDE_fan_ratio_proof_l2622_262277

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of Mets fans to Red Sox fans is 4:5 -/
def mets_to_red_sox_ratio (fc : FanCount) : Prop :=
  4 * fc.red_sox = 5 * fc.mets

/-- The total number of fans is 330 -/
def total_fans (fc : FanCount) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 330

/-- There are 88 Mets fans -/
def mets_fan_count (fc : FanCount) : Prop :=
  fc.mets = 88

/-- The ratio of Yankees fans to Mets fans is 3:2 -/
def yankees_to_mets_ratio (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees

theorem fan_ratio_proof (fc : FanCount)
    (h1 : mets_to_red_sox_ratio fc)
    (h2 : total_fans fc)
    (h3 : mets_fan_count fc) :
  yankees_to_mets_ratio fc := by
  sorry

end NUMINAMATH_CALUDE_fan_ratio_proof_l2622_262277


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2622_262214

theorem unique_triple_solution : 
  ∃! (x y z : ℕ+), 
    (¬(3 ∣ z ∧ y ∣ z)) ∧ 
    (Nat.Prime y) ∧ 
    (x^3 - y^3 = z^2) ∧
    x = 8 ∧ y = 7 ∧ z = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2622_262214


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_seven_l2622_262241

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Theorem for part (1)
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for part (2)
theorem a_greater_than_seven (h : A ⊆ C a) : a > 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_seven_l2622_262241


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2622_262294

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r) ^ 3) = 1 / 13.5 := by
  sorry

#check sphere_hemisphere_volume_ratio

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2622_262294


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l2622_262231

theorem divisibility_in_sequence (a : ℕ → ℕ) 
  (h : ∀ n ∈ Finset.range 3029, 2 * a (n + 2) = a (n + 1) + 4 * a n) :
  ∃ i ∈ Finset.range 3031, 2^2020 ∣ a i := by
sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l2622_262231


namespace NUMINAMATH_CALUDE_min_value_of_f_l2622_262265

def f (x : ℝ) := x^2 + 2

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2622_262265


namespace NUMINAMATH_CALUDE_tangent_circle_center_l2622_262282

/-- A circle tangent to two parallel lines with its center on a third line --/
structure TangentCircle where
  /-- The x-coordinate of the circle's center --/
  x : ℝ
  /-- The y-coordinate of the circle's center --/
  y : ℝ
  /-- The circle is tangent to the line 4x - 3y = 30 --/
  tangent_line1 : 4 * x - 3 * y = 30
  /-- The circle is tangent to the line 4x - 3y = -10 --/
  tangent_line2 : 4 * x - 3 * y = -10
  /-- The center of the circle lies on the line 2x + y = 0 --/
  center_line : 2 * x + y = 0

/-- The center of the circle satisfies all conditions and has coordinates (1, -2) --/
theorem tangent_circle_center : 
  ∃ (c : TangentCircle), c.x = 1 ∧ c.y = -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l2622_262282


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2622_262244

/-- Given two congruent cylinders with radius 10 inches and height 4 inches,
    where the radius of one cylinder and the height of the other are increased by x inches,
    prove that the only nonzero solution for equal volumes is x = 5. -/
theorem cylinder_volume_equality (x : ℝ) (hx : x ≠ 0) :
  π * (10 + x)^2 * 4 = π * 100 * (4 + x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2622_262244


namespace NUMINAMATH_CALUDE_alice_bob_meet_l2622_262217

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 11

/-- The starting position for both Alice and Bob -/
def start_pos : ℕ := n

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 5

/-- The position on the circle after a given number of clockwise moves -/
def position_after_moves (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves start_pos (meeting_turns * alice_move) =
  position_after_moves start_pos (meeting_turns * (n - bob_move)) :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l2622_262217


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l2622_262245

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l2622_262245


namespace NUMINAMATH_CALUDE_max_balls_count_l2622_262253

/-- Represents the count of balls -/
def n : ℕ := 45

/-- The number of green balls in the first 45 -/
def initial_green : ℕ := 41

/-- The number of green balls in each subsequent batch of 10 -/
def subsequent_green : ℕ := 9

/-- The total number of balls in each subsequent batch -/
def batch_size : ℕ := 10

/-- The minimum percentage of green balls required -/
def min_green_percentage : ℚ := 92 / 100

theorem max_balls_count :
  ∀ m : ℕ, m > n →
    (initial_green : ℚ) / n < min_green_percentage ∨
    (initial_green + (m - n) / batch_size * subsequent_green : ℚ) / m < min_green_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_balls_count_l2622_262253


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l2622_262223

/-- The function f(x) = x^2(x-2) + 1 -/
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

theorem tangent_line_at_one :
  let p : ℝ × ℝ := (1, f 1)
  let m : ℝ := f' 1
  ∀ x y : ℝ, (y - p.2 = m * (x - p.1)) ↔ (x + y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l2622_262223


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2622_262234

theorem arithmetic_calculation : 5 * (7 + 3) - 10 * 2 + 36 / 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2622_262234


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l2622_262279

def n : ℕ := 2014

theorem rearrangement_theorem (x y : Fin n → ℤ) 
  (hx : ∀ i j : Fin n, i ≠ j → x i % n ≠ x j % n)
  (hy : ∀ i j : Fin n, i ≠ j → y i % n ≠ y j % n) :
  ∃ σ : Equiv.Perm (Fin n), 
    ∀ i j : Fin n, i ≠ j → (x i + y (σ i)) % (2 * n) ≠ (x j + y (σ j)) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l2622_262279


namespace NUMINAMATH_CALUDE_jane_change_l2622_262240

def skirt_price : ℕ := 13
def skirt_quantity : ℕ := 2
def blouse_price : ℕ := 6
def blouse_quantity : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_quantity + blouse_price * blouse_quantity

theorem jane_change : amount_paid - total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l2622_262240


namespace NUMINAMATH_CALUDE_refrigerator_temp_difference_l2622_262270

/-- The temperature difference between two compartments in a refrigerator -/
def temperature_difference (refrigeration_temp freezer_temp : ℤ) : ℤ :=
  refrigeration_temp - freezer_temp

/-- Theorem stating the temperature difference between specific compartments -/
theorem refrigerator_temp_difference :
  temperature_difference 3 (-10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_temp_difference_l2622_262270


namespace NUMINAMATH_CALUDE_calculation_proof_l2622_262236

theorem calculation_proof :
  (∃ x, x = Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 ∧ x = 4 + Real.sqrt 6) ∧
  (∃ y, y = (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - Real.sqrt 27 * Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2622_262236


namespace NUMINAMATH_CALUDE_lunchroom_students_l2622_262204

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l2622_262204


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2622_262225

/-- Given a car traveling for 2 hours with a speed of 20 km/h in the first hour
    and an average speed of 25 km/h, prove that the speed of the car in the second hour is 30 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (total_time : ℝ)
  (h1 : speed_first_hour = 20)
  (h2 : average_speed = 25)
  (h3 : total_time = 2) :
  let speed_second_hour := (average_speed * total_time - speed_first_hour)
  speed_second_hour = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l2622_262225


namespace NUMINAMATH_CALUDE_product_abcd_l2622_262299

theorem product_abcd (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 42 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = (367/37) * (76/37) * (93/74) * (-55/74) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l2622_262299


namespace NUMINAMATH_CALUDE_min_upper_base_perimeter_is_12_l2622_262276

/-- Represents a frustum with rectangular bases -/
structure Frustum where
  upperBaseLength : ℝ
  upperBaseWidth : ℝ
  height : ℝ
  volume : ℝ

/-- The minimum perimeter of the upper base of a frustum with given properties -/
def minUpperBasePerimeter (f : Frustum) : ℝ :=
  2 * (f.upperBaseLength + f.upperBaseWidth)

/-- Theorem stating the minimum perimeter of the upper base for a specific frustum -/
theorem min_upper_base_perimeter_is_12 (f : Frustum) 
  (h1 : f.height = 3)
  (h2 : f.volume = 63)
  (h3 : f.upperBaseLength * f.upperBaseWidth * 7 = 63) :
  minUpperBasePerimeter f ≥ 12 ∧ 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    f.upperBaseLength = a ∧ f.upperBaseWidth = b ∧ 
    minUpperBasePerimeter f = 12 :=
  sorry


end NUMINAMATH_CALUDE_min_upper_base_perimeter_is_12_l2622_262276


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2622_262203

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop := a * x - b * y + 8 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line_passes_center : line_eq a b (circle_center.1) (circle_center.2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → line_eq a' b' (circle_center.1) (circle_center.2) → 
    1/a + 1/b ≤ 1/a' + 1/b') ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ line_eq a' b' (circle_center.1) (circle_center.2) ∧ 
    1/a' + 1/b' = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2622_262203


namespace NUMINAMATH_CALUDE_imaginary_product_real_part_l2622_262205

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_product_real_part (z : ℂ) (a b : ℝ) 
  (h1 : is_purely_imaginary z) 
  (h2 : (3 * Complex.I) * z = Complex.mk a b) : 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_product_real_part_l2622_262205


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2622_262290

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2622_262290


namespace NUMINAMATH_CALUDE_vector_problem_l2622_262250

theorem vector_problem (x y : ℝ) (hx : x > 0) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (2^2 + 4^2 + x^2 = (3*Real.sqrt 5)^2) →
  (2*2 + 4*y + x*2 = 0) →
  x + 2*y = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l2622_262250


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2622_262280

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + 2*I
  is_in_second_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2622_262280


namespace NUMINAMATH_CALUDE_inequality_and_nonexistence_l2622_262235

theorem inequality_and_nonexistence (x y z : ℝ) :
  (x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∀ k > Real.sqrt 3, ∃ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 < k*(x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_nonexistence_l2622_262235


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2622_262287

def OddUnitsDigit : Set Nat := {1, 3, 5, 7, 9}
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∃ (d : Nat), d ∈ AllDigits ∧ d ∉ OddUnitsDigit ∧ 
  ∀ (x : Nat), x ∈ AllDigits ∧ x ∉ OddUnitsDigit → d ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2622_262287


namespace NUMINAMATH_CALUDE_total_shells_count_l2622_262267

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_count : morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_count_l2622_262267


namespace NUMINAMATH_CALUDE_division_problem_l2622_262275

theorem division_problem (n : ℕ) : 
  n / 3 = 7 ∧ n % 3 = 1 → n = 22 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2622_262275


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2622_262285

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_quantity : ℕ) (sandwich_price : ℚ) (soda_quantity : ℕ) (soda_price : ℚ) : ℚ :=
  sandwich_quantity * sandwich_price + soda_quantity * soda_price

/-- Proof that the total cost of 2 sandwiches at $1.49 each and 4 sodas at $0.87 each is $6.46 -/
theorem total_cost_calculation : total_cost 2 (149/100) 4 (87/100) = 646/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2622_262285


namespace NUMINAMATH_CALUDE_box_length_l2622_262220

/-- The length of a box with given dimensions and cube requirements -/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (num_cubes : ℕ)
  (h_width : width = 12)
  (h_height : height = 3)
  (h_cube_volume : cube_volume = 3)
  (h_num_cubes : num_cubes = 108) :
  width * height * (num_cubes : ℝ) * cube_volume / (width * height) = 9 := by
  sorry

end NUMINAMATH_CALUDE_box_length_l2622_262220


namespace NUMINAMATH_CALUDE_square_difference_48_3_l2622_262292

theorem square_difference_48_3 : 48^2 - 2*(48*3) + 3^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_48_3_l2622_262292


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l2622_262210

theorem greatest_common_multiple_10_15_under_150 : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, n = 10 * k) ∧
    (∃ k : ℕ, n = 15 * k) ∧
    n < 150 ∧
    ∀ m : ℕ, (∃ k : ℕ, m = 10 * k) ∧ (∃ k : ℕ, m = 15 * k) ∧ m < 150 → m ≤ n →
    n = 120

-- The proof goes here
theorem greatest_common_multiple_10_15_under_150_is_120 :
  greatest_common_multiple_10_15_under_150 120 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l2622_262210


namespace NUMINAMATH_CALUDE_units_digit_of_147_25_50_l2622_262246

-- Define a function to calculate the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a power
def unitsDigitOfPower (base : ℕ) (exponent : ℕ) : ℕ :=
  unitsDigit ((unitsDigit base)^exponent)

-- Theorem to prove
theorem units_digit_of_147_25_50 :
  unitsDigitOfPower (unitsDigitOfPower 147 25) 50 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_147_25_50_l2622_262246


namespace NUMINAMATH_CALUDE_profit_ratio_calculation_l2622_262233

theorem profit_ratio_calculation (p q : ℕ) (investment_ratio_p investment_ratio_q : ℕ) 
  (investment_duration_p investment_duration_q : ℕ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  investment_duration_p = 2 →
  investment_duration_q = 4 →
  (investment_ratio_p * investment_duration_p) / (investment_ratio_q * investment_duration_q) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_calculation_l2622_262233


namespace NUMINAMATH_CALUDE_unique_solution_system_l2622_262201

theorem unique_solution_system (x y z : ℝ) :
  (x + y = 2 ∧ x * y - z^2 = 1) ↔ (x = 1 ∧ y = 1 ∧ z = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2622_262201


namespace NUMINAMATH_CALUDE_kiran_currency_notes_l2622_262237

/-- Represents the currency denominations in Rupees --/
inductive Denomination
  | fifty : Denomination
  | hundred : Denomination

/-- Represents the total amount and number of notes for each denomination --/
structure CurrencyNotes where
  total_amount : ℕ
  fifty_amount : ℕ
  fifty_count : ℕ
  hundred_count : ℕ

/-- Calculates the total number of currency notes --/
def total_notes (c : CurrencyNotes) : ℕ :=
  c.fifty_count + c.hundred_count

/-- Theorem stating that given the conditions, Kiran has 85 currency notes in total --/
theorem kiran_currency_notes :
  ∀ (c : CurrencyNotes),
    c.total_amount = 5000 →
    c.fifty_amount = 3500 →
    c.fifty_count = c.fifty_amount / 50 →
    c.hundred_count = (c.total_amount - c.fifty_amount) / 100 →
    total_notes c = 85 := by
  sorry

end NUMINAMATH_CALUDE_kiran_currency_notes_l2622_262237


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l2622_262249

/-- The sum of the digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A is the sum of the digits of 4444^4444 -/
def A : ℕ := digit_sum (4444^4444)

/-- B is the sum of the digits of A -/
def B : ℕ := digit_sum A

/-- The main theorem: the sum of the digits of B is 7 -/
theorem sum_of_digits_of_B_is_seven : digit_sum B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l2622_262249


namespace NUMINAMATH_CALUDE_ab_nonpositive_l2622_262263

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l2622_262263


namespace NUMINAMATH_CALUDE_calculate_mixed_fraction_expression_l2622_262254

theorem calculate_mixed_fraction_expression : 
  (47 * ((2 + 2/3) - (3 + 1/4))) / ((3 + 1/2) + (2 + 1/5)) = -(4 + 25/38) := by
  sorry

end NUMINAMATH_CALUDE_calculate_mixed_fraction_expression_l2622_262254


namespace NUMINAMATH_CALUDE_triangle_area_l2622_262281

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b^2 + c^2 = a^2 - Real.sqrt 3 * b * c →
  b * c * Real.cos A = -4 →
  (1/2) * b * c * Real.sin A = (2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2622_262281


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2622_262286

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2622_262286


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l2622_262230

theorem sugar_recipe_reduction : 
  (3 + 3 / 4 : ℚ) / 3 = 1 + 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l2622_262230


namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2622_262298

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 4}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Part 1
theorem intersection_and_union (a : ℝ) (h : a = 0) :
  (A a ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A a ∪ (Set.univ \ B) = {x | x < -2 ∨ x ≥ 0}) := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | A a ∪ B = B} = {a : ℝ | -2 ≤ a ∧ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2622_262298


namespace NUMINAMATH_CALUDE_new_person_weight_is_85_l2622_262216

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 85 kg -/
theorem new_person_weight_is_85 :
  new_person_weight 8 2.5 65 = 85 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_85_l2622_262216


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2622_262261

/-- 
Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots,
prove that k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2622_262261


namespace NUMINAMATH_CALUDE_loaf_has_twelve_slices_l2622_262208

/-- Represents a household with bread consumption patterns. -/
structure Household where
  members : ℕ
  breakfast_slices : ℕ
  snack_slices : ℕ
  loaves : ℕ
  days : ℕ

/-- Calculates the number of slices in a loaf of bread for a given household. -/
def slices_per_loaf (h : Household) : ℕ :=
  (h.members * (h.breakfast_slices + h.snack_slices) * h.days) / h.loaves

/-- Theorem stating that for the given household, a loaf of bread contains 12 slices. -/
theorem loaf_has_twelve_slices : 
  slices_per_loaf { members := 4, breakfast_slices := 3, snack_slices := 2, loaves := 5, days := 3 } = 12 := by
  sorry

end NUMINAMATH_CALUDE_loaf_has_twelve_slices_l2622_262208


namespace NUMINAMATH_CALUDE_fixed_point_on_symmetric_line_l2622_262232

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define symmetry about a point
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) →
    ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧
      (x + x') / 2 = p.1 ∧ (y + y') / 2 = p.2

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem fixed_point_on_symmetric_line (k : ℝ) :
  ∀ (l2 : Line), symmetric_about ⟨k, -4*k⟩ l2 (2, 1) →
    point_on_line (0, 2) l2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_symmetric_line_l2622_262232


namespace NUMINAMATH_CALUDE_max_product_762_l2622_262262

def digits : Finset Nat := {2, 4, 6, 7, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

theorem max_product_762 :
  ∀ a b c d e : Nat,
  is_valid_pair a b c d e →
  three_digit 7 6 2 * two_digit 9 4 ≥ three_digit a b c * two_digit d e :=
by sorry

end NUMINAMATH_CALUDE_max_product_762_l2622_262262


namespace NUMINAMATH_CALUDE_total_ladybugs_l2622_262295

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_l2622_262295


namespace NUMINAMATH_CALUDE_swim_meet_car_capacity_l2622_262219

/-- Represents the transportation details for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_per_van : ℕ
  additional_capacity : ℕ

/-- Calculates the maximum capacity per car given the transport details --/
def max_capacity_per_car (t : SwimMeetTransport) : ℕ :=
  let total_people := t.num_cars * t.people_per_car + t.num_vans * t.people_per_van
  let total_capacity := total_people + t.additional_capacity
  let van_capacity := t.num_vans * t.max_per_van
  (total_capacity - van_capacity) / t.num_cars

/-- Theorem stating that the maximum capacity per car is 6 for the given scenario --/
theorem swim_meet_car_capacity :
  let t : SwimMeetTransport := {
    num_cars := 2,
    num_vans := 3,
    people_per_car := 5,
    people_per_van := 3,
    max_per_van := 8,
    additional_capacity := 17
  }
  max_capacity_per_car t = 6 := by
  sorry

end NUMINAMATH_CALUDE_swim_meet_car_capacity_l2622_262219


namespace NUMINAMATH_CALUDE_triangle_angle_sum_impossibility_l2622_262264

theorem triangle_angle_sum_impossibility (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬((α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) ∨
    (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_impossibility_l2622_262264


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2622_262272

/-- Given a hyperbola with equation x²/2 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 2 - y^2 = 1
  ∃ (a b c : ℝ),
    a^2 = 2 ∧ 
    b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x, (h x (x * (b / a)) ∨ h x (-x * (b / a))) ↔ x ≠ 0) ∧
    c / a = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2622_262272


namespace NUMINAMATH_CALUDE_triangle_problem_l2622_262289

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * c * Real.cos C = b * Real.cos A + a * Real.cos B ∧
  a = 6 ∧
  Real.cos A = -4/5 →
  C = π/3 ∧ c = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2622_262289


namespace NUMINAMATH_CALUDE_remainder_17_65_mod_7_l2622_262248

theorem remainder_17_65_mod_7 : 17^65 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_17_65_mod_7_l2622_262248


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2622_262278

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2622_262278


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2622_262293

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {1, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2622_262293


namespace NUMINAMATH_CALUDE_divided_hexagon_areas_l2622_262228

/-- Represents a regular hexagon divided by four diagonals -/
structure DividedHexagon where
  /-- The area of the central quadrilateral -/
  quadrilateralArea : ℝ
  /-- The areas of the six triangles -/
  triangleAreas : Fin 6 → ℝ

/-- Theorem about the areas of triangles in a divided regular hexagon -/
theorem divided_hexagon_areas (h : DividedHexagon) 
  (hq : h.quadrilateralArea = 1.8) : 
  (h.triangleAreas 0 = 1.2 ∧ 
   h.triangleAreas 1 = 1.2 ∧ 
   h.triangleAreas 2 = 0.6 ∧ 
   h.triangleAreas 3 = 0.6 ∧ 
   h.triangleAreas 4 = 1.2 ∧ 
   h.triangleAreas 5 = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_divided_hexagon_areas_l2622_262228


namespace NUMINAMATH_CALUDE_tangent_length_circle_l2622_262258

/-- Given a circle C with center (2, 1) and radius 2, and a point A(-4, -1),
    prove that the length of the tangent from A to C is 6. -/
theorem tangent_length_circle (C : Set (ℝ × ℝ)) (A : ℝ × ℝ) : 
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4} →
  A = (-4, -1) →
  ∃ B ∈ C, (B.1 - 2) * (B.1 - (-4)) + (B.2 - 1) * (B.2 - (-1)) = 0 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_circle_l2622_262258


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l2622_262243

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) 
  (h_geo : is_positive_geometric_sequence a)
  (h_condition : a 3 * a 6 + a 2 * a 7 = 2 * Real.exp 4) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y ≤ 4) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y = 4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l2622_262243


namespace NUMINAMATH_CALUDE_distance_center_to_secant_l2622_262273

/-- Given a circle O with center (0, 0) and radius 5, a tangent line AD of length 4,
    and a secant line ABC where AC = 8, the distance from the center O to the line AC is 4. -/
theorem distance_center_to_secant (O A B C D : ℝ × ℝ) : 
  let r := 5
  let circle := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
  (A ∉ circle) →
  (B ∈ circle) →
  (C ∈ circle) →
  (D ∈ circle) →
  (∀ p ∈ circle, (p.1 - A.1) * (D.1 - A.1) + (p.2 - A.2) * (D.2 - A.2) = 0) →
  (Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4) →
  (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8) →
  (abs ((O.2 - A.2) * (C.1 - A.1) - (O.1 - A.1) * (C.2 - A.2)) / 
   Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_secant_l2622_262273


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2622_262206

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (a b : Line) (α : Plane) :
  parallel_line a b → 
  parallel_line_plane b α → 
  ¬ contained_in a α → 
  parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2622_262206


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2622_262212

/-- Given a curve C in the Cartesian coordinate system with polar equation ρ = 2cosθ - 4sinθ,
    prove that its Cartesian equation is (x - 2)² - 15y² = 68 - (y + 8)² -/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  ρ = 2 * Real.cos θ - 4 * Real.sin θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (x - 2)^2 - 15 * y^2 = 68 - (y + 8)^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2622_262212


namespace NUMINAMATH_CALUDE_dividend_calculation_l2622_262255

theorem dividend_calculation (quotient : ℕ) (k : ℕ) (h1 : quotient = 4) (h2 : k = 14) :
  quotient * k = 56 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2622_262255


namespace NUMINAMATH_CALUDE_triangle_abc_solutions_l2622_262218

theorem triangle_abc_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    Real.sin A / a = Real.sin B / b ∧
    Real.sin C / c = Real.sin B / b :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_solutions_l2622_262218


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l2622_262257

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_log_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_prod : a 2 * a 5 = 10) : 
  Real.log (a 3) + Real.log (a 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l2622_262257


namespace NUMINAMATH_CALUDE_janes_skirts_l2622_262260

/-- Proves that Jane bought 2 skirts given the problem conditions -/
theorem janes_skirts :
  let skirt_price : ℕ := 13
  let blouse_price : ℕ := 6
  let num_blouses : ℕ := 3
  let paid : ℕ := 100
  let change : ℕ := 56
  let total_spent : ℕ := paid - change
  ∃ (num_skirts : ℕ), num_skirts * skirt_price + num_blouses * blouse_price = total_spent ∧ num_skirts = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_skirts_l2622_262260


namespace NUMINAMATH_CALUDE_sum_floor_is_179_l2622_262238

theorem sum_floor_is_179 
  (p q r s : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0)
  (h1 : p^2 + q^2 = 4016) (h2 : r^2 + s^2 = 4016)
  (h3 : p * r = 2000) (h4 : q * s = 2000) : 
  ⌊p + q + r + s⌋ = 179 := by
sorry

end NUMINAMATH_CALUDE_sum_floor_is_179_l2622_262238


namespace NUMINAMATH_CALUDE_non_constant_geometric_sequence_exists_l2622_262213

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is non-constant -/
def NonConstant (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

theorem non_constant_geometric_sequence_exists :
  ∃ a : ℕ → ℝ, GeometricSequence a ∧ NonConstant a ∧
  ∃ r s : ℕ, r ≠ s ∧ a r = a s :=
by sorry

end NUMINAMATH_CALUDE_non_constant_geometric_sequence_exists_l2622_262213


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_30_l2622_262297

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_30 :
  units_digit (sum_factorials 30) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_30_l2622_262297


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2622_262227

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2622_262227
