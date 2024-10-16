import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_sum_l3935_393542

-- Define the polynomials f and g
def f (a b x : ℝ) := x^2 + a*x + b
def g (c d x : ℝ) := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∃ (x : ℝ), f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (f a b (-a/2) = -25) →                 -- minimum value of f is -25
  (g c d (-c/2) = -25) →                 -- minimum value of g is -25
  (f a b 50 = -50) →                     -- f and g intersect at (50, -50)
  (g c d 50 = -50) →                     -- f and g intersect at (50, -50)
  (a ≠ c ∨ b ≠ d) →                      -- f and g are distinct
  a + c = -200 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3935_393542


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3935_393505

-- Part 1
theorem problem_1 : 2023^2 - 2022 * 2024 = 1 := by sorry

-- Part 2
theorem problem_2 (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -1) (h3 : m ≠ 0) :
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3935_393505


namespace NUMINAMATH_CALUDE_g_50_eq_zero_l3935_393588

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) to satisfy the condition that for any positive integer n, 
-- the sum of g(d) over all positive divisors d of n equals φ(n)
def g (n : ℕ) : ℤ :=
  sorry

-- Theorem to prove
theorem g_50_eq_zero : g 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_50_eq_zero_l3935_393588


namespace NUMINAMATH_CALUDE_square_prime_equivalence_l3935_393534

theorem square_prime_equivalence (N : ℕ) (h : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬∃ s : ℕ, 4*n*(N-n)+1 = s^2) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_square_prime_equivalence_l3935_393534


namespace NUMINAMATH_CALUDE_motorcycle_price_is_correct_l3935_393571

/-- Represents the factory's production and profit information -/
structure FactoryInfo where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycles_produced : ℕ
  profit_difference : ℕ

/-- Calculates the price per motorcycle based on the given factory information -/
def calculate_motorcycle_price (info : FactoryInfo) : ℕ :=
  (info.profit_difference + (info.car_material_cost + info.motorcycle_material_cost - info.cars_produced * info.car_price) + info.motorcycle_material_cost) / info.motorcycles_produced

/-- Theorem stating that the calculated motorcycle price is correct -/
theorem motorcycle_price_is_correct (info : FactoryInfo) 
  (h1 : info.car_material_cost = 100)
  (h2 : info.cars_produced = 4)
  (h3 : info.car_price = 50)
  (h4 : info.motorcycle_material_cost = 250)
  (h5 : info.motorcycles_produced = 8)
  (h6 : info.profit_difference = 50) :
  calculate_motorcycle_price info = 50 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_price_is_correct_l3935_393571


namespace NUMINAMATH_CALUDE_tuesday_temperature_l3935_393596

/-- Given the average temperatures for three consecutive days and the temperature of the last day,
    this theorem proves the temperature of the first day. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : ℝ)
  (avg_wed_thurs_fri : ℝ)
  (temp_friday : ℝ)
  (h1 : avg_tues_wed_thurs = 32)
  (h2 : avg_wed_thurs_fri = 34)
  (h3 : temp_friday = 44) :
  ∃ (temp_tuesday temp_wednesday temp_thursday : ℝ),
    (temp_tuesday + temp_wednesday + temp_thursday) / 3 = avg_tues_wed_thurs ∧
    (temp_wednesday + temp_thursday + temp_friday) / 3 = avg_wed_thurs_fri ∧
    temp_tuesday = 38 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_temperature_l3935_393596


namespace NUMINAMATH_CALUDE_class_average_weight_l3935_393533

theorem class_average_weight (num_boys : ℕ) (num_girls : ℕ) 
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) :
  num_boys = 5 →
  num_girls = 3 →
  avg_weight_boys = 60 →
  avg_weight_girls = 50 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l3935_393533


namespace NUMINAMATH_CALUDE_inverse_proportion_change_l3935_393558

theorem inverse_proportion_change (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = c) :
  let a' := 1.2 * a
  let b' := 80
  a' * b' = c →
  b = 96 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_change_l3935_393558


namespace NUMINAMATH_CALUDE_angle_properties_l3935_393510

theorem angle_properties (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) 
  (h3 : (a, 2*a) ∈ Set.range (λ t : ℝ × ℝ => (t.1 * Real.cos α, t.1 * Real.sin α))) :
  Real.cos α = -Real.sqrt 5 / 5 ∧ 
  Real.tan α = 2 ∧ 
  (Real.cos α)^2 / Real.tan α = 1/10 := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l3935_393510


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3935_393513

/-- Represents the voting system in country Y -/
structure VotingSystem where
  total_voters : Nat
  sellable_voters : Nat
  preference_voters : Nat
  initial_votes : Nat
  votes_to_win : Nat

/-- Calculates the number of votes a candidate can secure based on the price offered -/
def supply_function (system : VotingSystem) (price : Nat) : Nat :=
  if price = 0 then system.initial_votes
  else if price ≤ system.sellable_voters then min (system.initial_votes + price) system.total_voters
  else min (system.initial_votes + system.sellable_voters) system.total_voters

/-- Calculates the minimum cost to win the election -/
def min_cost_to_win (system : VotingSystem) : Nat :=
  let required_additional_votes := system.votes_to_win - system.initial_votes
  required_additional_votes * (required_additional_votes + 1)

/-- The main theorem stating the minimum cost to win the election -/
theorem min_cost_theorem (system : VotingSystem) 
    (h1 : system.total_voters = 35)
    (h2 : system.sellable_voters = 14)
    (h3 : system.preference_voters = 21)
    (h4 : system.initial_votes = 10)
    (h5 : system.votes_to_win = 18) :
    min_cost_to_win system = 162 := by
  sorry

#eval min_cost_to_win { total_voters := 35, sellable_voters := 14, preference_voters := 21, initial_votes := 10, votes_to_win := 18 }

end NUMINAMATH_CALUDE_min_cost_theorem_l3935_393513


namespace NUMINAMATH_CALUDE_negation_of_existence_rational_sqrt_two_l3935_393583

theorem negation_of_existence_rational_sqrt_two :
  (¬ ∃ (x : ℚ), x^2 - 2 = 0) ↔ (∀ (x : ℚ), x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_rational_sqrt_two_l3935_393583


namespace NUMINAMATH_CALUDE_prime_sequence_existence_l3935_393594

theorem prime_sequence_existence (k : ℕ) (hk : k > 1) :
  ∃ (p : ℕ) (a : ℕ → ℕ),
    Prime p ∧
    (∀ n m, n < m → a n < a m) ∧
    (∀ n, n > 1 → Prime (p + k * a n)) := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_existence_l3935_393594


namespace NUMINAMATH_CALUDE_one_quarter_between_thirds_l3935_393559

theorem one_quarter_between_thirds (x : ℚ) : 
  (x = 1/3 + 1/4 * (2/3 - 1/3)) → x = 5/12 := by
sorry

end NUMINAMATH_CALUDE_one_quarter_between_thirds_l3935_393559


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3935_393563

/-- A color type with two possible values -/
inductive Color
| black
| white

/-- A function that assigns a color to each point in the grid -/
def coloringFunction : ℤ × ℤ → Color := sorry

/-- Theorem: In an infinite grid with vertices colored in two colors, 
    there exist two horizontal lines and two vertical lines such that 
    their four intersection points are of the same color -/
theorem monochromatic_rectangle_exists : 
  ∃ (x₁ x₂ y₁ y₂ : ℤ) (c : Color), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    coloringFunction (x₁, y₁) = c ∧
    coloringFunction (x₁, y₂) = c ∧
    coloringFunction (x₂, y₁) = c ∧
    coloringFunction (x₂, y₂) = c :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3935_393563


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l3935_393538

def carnival_spending (bumper_car_rides : ℕ) (space_shuttle_rides : ℕ) (ferris_wheel_rides : ℕ)
  (bumper_car_cost : ℕ) (space_shuttle_cost : ℕ) (ferris_wheel_cost : ℕ) : ℕ :=
  bumper_car_rides * bumper_car_cost +
  space_shuttle_rides * space_shuttle_cost +
  2 * ferris_wheel_rides * ferris_wheel_cost

theorem carnival_spending_theorem :
  carnival_spending 2 4 3 2 4 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l3935_393538


namespace NUMINAMATH_CALUDE_correct_article_usage_l3935_393598

/-- Represents the possible article choices --/
inductive Article
  | A
  | The
  | None

/-- Represents a sentence with two article slots --/
structure Sentence where
  firstArticle : Article
  secondArticle : Article

/-- Checks if the article usage is correct for the given sentence --/
def isCorrectArticleUsage (s : Sentence) : Prop :=
  s.firstArticle = Article.A ∧ s.secondArticle = Article.None

/-- Theorem stating that the correct article usage is "a" for the first blank and no article for the second --/
theorem correct_article_usage :
  ∃ (s : Sentence), isCorrectArticleUsage s :=
sorry


end NUMINAMATH_CALUDE_correct_article_usage_l3935_393598


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3935_393584

/-- The equation has exactly one real solution in x if and only if a < 1 -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3935_393584


namespace NUMINAMATH_CALUDE_roots_expression_value_l3935_393554

theorem roots_expression_value (m n : ℝ) : 
  m^2 + 2*m - 2027 = 0 → 
  n^2 + 2*n - 2027 = 0 → 
  2*m - m*n + 2*n = 2023 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_value_l3935_393554


namespace NUMINAMATH_CALUDE_ivan_revival_time_l3935_393515

/-- Represents the scenario of Wolf, Ivan, and Raven --/
structure Scenario where
  distance : ℝ
  wolf_speed : ℝ
  water_needed : ℝ
  water_flow_rate : ℝ
  raven_speed : ℝ
  water_spill_rate : ℝ

/-- Checks if Ivan can be revived within the given time --/
def can_revive (s : Scenario) (time : ℝ) : Prop :=
  let wolf_travel_time := s.distance / s.wolf_speed
  let water_collect_time := s.water_needed / s.water_flow_rate
  let total_time := wolf_travel_time + water_collect_time
  let raven_travel_distance := s.distance / 2
  let raven_travel_time := raven_travel_distance / s.raven_speed
  let water_lost := raven_travel_time * s.water_spill_rate
  let water_remaining := s.water_needed - water_lost
  
  time ≥ total_time ∧ water_remaining > 0

/-- The main theorem to prove --/
theorem ivan_revival_time (s : Scenario) :
  s.distance = 20 ∧
  s.wolf_speed = 3 ∧
  s.water_needed = 1 ∧
  s.water_flow_rate = 0.5 ∧
  s.raven_speed = 6 ∧
  s.water_spill_rate = 0.25 →
  can_revive s 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ivan_revival_time_l3935_393515


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_not_polygon_interior_angle_sum_l3935_393575

theorem polygon_interior_angle_sum (n : ℕ) (sum : ℕ) : sum = (n - 2) * 180 → n ≥ 3 :=
by sorry

theorem not_polygon_interior_angle_sum : ¬ ∃ (n : ℕ), 800 = (n - 2) * 180 ∧ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_not_polygon_interior_angle_sum_l3935_393575


namespace NUMINAMATH_CALUDE_relationship_xyz_l3935_393553

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = Real.log π) 
  (hy : y = Real.log 2 / Real.log 5)
  (hz : z = Real.exp (-1/2)) :
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l3935_393553


namespace NUMINAMATH_CALUDE_opposite_of_seven_l3935_393502

theorem opposite_of_seven : 
  (-(7 : ℝ) = -7) := by sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l3935_393502


namespace NUMINAMATH_CALUDE_expression_evaluation_l3935_393569

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/4
  2 * (x - 2*y) - 1/3 * (3*x - 6*y) + 2*x = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3935_393569


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3935_393517

theorem scientific_notation_of_120_million : 
  ∃ (a : ℝ) (n : ℤ), 120000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3935_393517


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3935_393509

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = f x) →
    (y = m * (x - 1) + f 1) →
    (m * x - y - b = 0) →
    (m = Real.exp 1) ∧
    (b = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3935_393509


namespace NUMINAMATH_CALUDE_negation_of_implication_l3935_393578

theorem negation_of_implication (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3935_393578


namespace NUMINAMATH_CALUDE_prob_no_adjacent_birch_value_l3935_393597

/-- The number of pine trees -/
def num_pine : ℕ := 6

/-- The number of cedar trees -/
def num_cedar : ℕ := 5

/-- The number of birch trees -/
def num_birch : ℕ := 7

/-- The total number of trees -/
def total_trees : ℕ := num_pine + num_cedar + num_birch

/-- The number of slots for birch trees -/
def num_slots : ℕ := num_pine + num_cedar + 1

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := (num_slots.choose num_birch : ℚ) / (total_trees.choose num_birch)

theorem prob_no_adjacent_birch_value : prob_no_adjacent_birch = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_birch_value_l3935_393597


namespace NUMINAMATH_CALUDE_joan_balloons_l3935_393500

/-- The number of blue balloons Joan has now -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Proof that Joan has 11 blue balloons now -/
theorem joan_balloons : total_balloons 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3935_393500


namespace NUMINAMATH_CALUDE_target_hit_probability_l3935_393530

/-- The binomial probability function -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem target_hit_probability :
  let n : ℕ := 6
  let k : ℕ := 5
  let p : ℝ := 0.8
  abs (binomial_probability n k p - 0.3932) < 0.00005 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3935_393530


namespace NUMINAMATH_CALUDE_rectangles_in_five_by_five_grid_l3935_393556

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square array of dots -/
def numRectangles (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_five_by_five_grid :
  numRectangles gridSize = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_five_by_five_grid_l3935_393556


namespace NUMINAMATH_CALUDE_triple_counted_number_l3935_393536

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n % 10) % 5 = 0

def sum_valid_numbers : ℕ := sorry

theorem triple_counted_number (triple_counted : ℕ) 
  (h1 : is_valid_number triple_counted)
  (h2 : sum_valid_numbers + 2 * triple_counted = 1035) :
  triple_counted = 45 := by sorry

end NUMINAMATH_CALUDE_triple_counted_number_l3935_393536


namespace NUMINAMATH_CALUDE_sally_cut_orchids_l3935_393506

/-- The number of red orchids Sally cut -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

theorem sally_cut_orchids : orchids_cut 9 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_cut_orchids_l3935_393506


namespace NUMINAMATH_CALUDE_safeties_count_l3935_393589

/-- Represents the scoring of a football team -/
structure FootballScore where
  fieldGoals : ℕ      -- number of four-point field goals
  threePointGoals : ℕ -- number of three-point goals
  safeties : ℕ        -- number of two-point safeties

/-- Calculates the total score for a given FootballScore -/
def totalScore (score : FootballScore) : ℕ :=
  4 * score.fieldGoals + 3 * score.threePointGoals + 2 * score.safeties

/-- Theorem: Given the conditions, the number of safeties is 6 -/
theorem safeties_count (score : FootballScore) :
  (4 * score.fieldGoals = 2 * 3 * score.threePointGoals) →
  (score.safeties = score.threePointGoals + 2) →
  (totalScore score = 50) →
  score.safeties = 6 :=
by sorry

end NUMINAMATH_CALUDE_safeties_count_l3935_393589


namespace NUMINAMATH_CALUDE_sunday_to_weekday_ratio_is_correct_l3935_393529

/-- The weight ratio of Sunday papers to Monday-Saturday papers --/
def sunday_to_weekday_ratio : ℚ :=
  let weekday_paper_weight : ℚ := 8  -- ounces
  let papers_per_day : ℕ := 250
  let weeks : ℕ := 10
  let weekdays_per_week : ℕ := 6
  let recycling_rate : ℚ := 100 / 2000  -- $/pound

  let total_weekday_papers : ℕ := papers_per_day * weekdays_per_week * weeks
  let total_weekday_weight : ℚ := weekday_paper_weight * total_weekday_papers
  
  let total_sunday_papers : ℕ := papers_per_day * weeks
  let total_sunday_weight : ℚ := 2000 * 16  -- 1 ton in ounces
  
  let sunday_paper_weight : ℚ := total_sunday_weight / total_sunday_papers
  
  sunday_paper_weight / weekday_paper_weight

theorem sunday_to_weekday_ratio_is_correct : sunday_to_weekday_ratio = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_weekday_ratio_is_correct_l3935_393529


namespace NUMINAMATH_CALUDE_power_difference_mod_six_l3935_393592

theorem power_difference_mod_six :
  (47^2045 - 18^2045) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_mod_six_l3935_393592


namespace NUMINAMATH_CALUDE_optimal_hospital_location_l3935_393562

/-- Given three points A, B, and C in a plane, with AB = AC = 13 and BC = 10,
    prove that the point P(0, 4) on the perpendicular bisector of BC
    minimizes the sum of squares of distances PA^2 + PB^2 + PC^2 -/
theorem optimal_hospital_location (A B C P : ℝ × ℝ) :
  A = (0, 12) →
  B = (-5, 0) →
  C = (5, 0) →
  P.1 = 0 →
  (∀ y : ℝ, (0, y).1^2 + (0, y).2^2 + (-5 - 0)^2 + (0 - y)^2 + (5 - 0)^2 + (0 - y)^2 ≥
             (0, 4).1^2 + (0, 4).2^2 + (-5 - 0)^2 + (0 - 4)^2 + (5 - 0)^2 + (0 - 4)^2) →
  P = (0, 4) :=
by sorry

end NUMINAMATH_CALUDE_optimal_hospital_location_l3935_393562


namespace NUMINAMATH_CALUDE_max_price_changes_l3935_393516

/-- Represents the price of the souvenir after n changes -/
def price (initial : ℕ) (x : ℚ) (n : ℕ) : ℚ :=
  initial * ((1 - x/100)^n * (1 + x/100)^n)

/-- The problem statement -/
theorem max_price_changes (initial : ℕ) (x : ℚ) : 
  initial = 10000 →
  0 < x →
  x < 100 →
  (∃ n : ℕ, ¬(price initial x n).isInt ∧ (price initial x (n-1)).isInt) →
  (∃ max_changes : ℕ, 
    (∀ n : ℕ, n ≤ max_changes → (price initial x n).isInt) ∧
    ¬(price initial x (max_changes + 1)).isInt ∧
    max_changes = 5) :=
sorry

end NUMINAMATH_CALUDE_max_price_changes_l3935_393516


namespace NUMINAMATH_CALUDE_arc_length_240_degrees_l3935_393586

theorem arc_length_240_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 240 → l = (θ * π * r) / 180 → l = (40 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_arc_length_240_degrees_l3935_393586


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l3935_393599

/-- The area of a triangle given its three vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 13/96 -/
theorem triangle_area_fraction :
  let a_x := 2
  let a_y := 4
  let b_x := 7
  let b_y := 2
  let c_x := 6
  let c_y := 5
  let grid_width := 8
  let grid_height := 6
  (triangleArea a_x a_y b_x b_y c_x c_y) / (grid_width * grid_height) = 13 / 96 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_fraction_l3935_393599


namespace NUMINAMATH_CALUDE_box_height_l3935_393548

/-- Calculates the height of a box with given specifications -/
theorem box_height (internal_volume : ℕ) (external_side_length : ℕ) : 
  internal_volume = 6912 ∧ 
  external_side_length = 26 → 
  (external_side_length - 2)^2 * 12 = internal_volume := by
  sorry

#check box_height

end NUMINAMATH_CALUDE_box_height_l3935_393548


namespace NUMINAMATH_CALUDE_lcm_problem_l3935_393570

theorem lcm_problem (n : ℕ+) 
  (h1 : Nat.lcm 40 n = 120) 
  (h2 : Nat.lcm n 45 = 180) : 
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3935_393570


namespace NUMINAMATH_CALUDE_closest_to_fraction_l3935_393531

def fraction : ℚ := 501 / (1 / 4)

def options : List ℤ := [1800, 1900, 2000, 2100, 2200]

theorem closest_to_fraction :
  (2000 : ℤ) = (options.argmin (λ x => |↑x - fraction|)).get
    (by sorry) := by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l3935_393531


namespace NUMINAMATH_CALUDE_only_cylinder_not_polyhedron_l3935_393572

-- Define the set of given figures
inductive Figure
  | ObliquePrism
  | Cube
  | Cylinder
  | Tetrahedron

-- Define what a polyhedron is
def isPolyhedron (f : Figure) : Prop :=
  match f with
  | Figure.ObliquePrism => true
  | Figure.Cube => true
  | Figure.Cylinder => false
  | Figure.Tetrahedron => true

-- Theorem statement
theorem only_cylinder_not_polyhedron :
  ∀ f : Figure, ¬(isPolyhedron f) ↔ f = Figure.Cylinder :=
sorry

end NUMINAMATH_CALUDE_only_cylinder_not_polyhedron_l3935_393572


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3935_393537

theorem quadratic_equation_roots (k : ℝ) (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3935_393537


namespace NUMINAMATH_CALUDE_range_of_a_l3935_393595

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ≤ a - 4 ∨ x ≥ a + 4) → (x ≤ 1 ∨ x ≥ 2)) → 
  a ∈ Set.Icc (-2) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3935_393595


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3935_393525

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles trapezoid with bases 7 and 13 and area 40, the side length is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := ⟨7, 13, 40⟩
  side_length t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3935_393525


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3935_393523

/-- Represents an ellipse on a coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h_center : e.center = (1, -3))
  (h_major : e.semiMajorAxis = 7)
  (h_minor : e.semiMinorAxis = 4) :
  e.center.1 + e.center.2 + e.semiMajorAxis + e.semiMinorAxis = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3935_393523


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3935_393503

def f (x : ℝ) : ℝ := x^2

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => f x + b

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem quadratic_transformation :
  shift_up (shift_right f 3) 4 = g := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3935_393503


namespace NUMINAMATH_CALUDE_necessary_to_sufficient_contrapositive_l3935_393526

theorem necessary_to_sufficient_contrapositive (p q : Prop) :
  (q → p) → (¬p → ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_to_sufficient_contrapositive_l3935_393526


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_l3935_393508

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Part 1
theorem part_1 (a : ℝ) (h : circle_C a (a+1)) :
  let P := (a, a+1)
  (‖P - Q‖ = 2 * Real.sqrt 10) ∧ 
  ((P.2 - Q.2) / (P.1 - Q.1) = 1/3) :=
sorry

-- Part 2
theorem part_2 :
  ∀ M : ℝ × ℝ, circle_C M.1 M.2 → 
  2 * Real.sqrt 2 ≤ ‖M - Q‖ ∧ ‖M - Q‖ ≤ 6 * Real.sqrt 2 :=
sorry

-- Part 3
theorem part_3 (m n : ℝ) (h : m^2 + n^2 - 4*m - 14*n + 45 = 0) :
  2 - Real.sqrt 3 ≤ (n - 3) / (m + 2) ∧ 
  (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_l3935_393508


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3935_393566

/-- Represents the cost of tickets in rubles -/
def N : ℝ := sorry

/-- The number of southern cities -/
def num_southern_cities : ℕ := 4

/-- The number of northern cities -/
def num_northern_cities : ℕ := 5

/-- The cost of a one-way ticket between any two connected cities -/
def one_way_cost : ℝ := N

/-- The cost of a round-trip ticket between any two connected cities -/
def round_trip_cost : ℝ := 1.6 * N

/-- A route represents a sequence of city visits -/
def Route := List ℕ

/-- Predicate to check if a route is valid according to the problem constraints -/
def is_valid_route (r : Route) : Prop := sorry

/-- The cost of a given route -/
def route_cost (r : Route) : ℝ := sorry

/-- Theorem stating the minimum cost to visit all southern cities and return to the start -/
theorem min_cost_theorem :
  ∀ (r : Route), is_valid_route r →
    route_cost r ≥ 6.4 * N ∧
    ∃ (optimal_route : Route), 
      is_valid_route optimal_route ∧ 
      route_cost optimal_route = 6.4 * N :=
by sorry

end NUMINAMATH_CALUDE_min_cost_theorem_l3935_393566


namespace NUMINAMATH_CALUDE_john_mean_score_l3935_393532

def john_scores : List ℝ := [95, 88, 90, 92, 94, 89]

theorem john_mean_score : 
  (john_scores.sum / john_scores.length : ℝ) = 91.3333 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l3935_393532


namespace NUMINAMATH_CALUDE_tangent_line_count_possibilities_l3935_393535

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Counts the number of distinct values in a list of natural numbers -/
def countDistinctValues (list : List ℕ) : ℕ :=
  (list.toFinset).card

/-- The possible numbers of tangent lines for two non-overlapping circles -/
def possibleTangentLineCounts : List ℕ := [0, 3, 4]

/-- Theorem stating that for two non-overlapping circles with radii 5 and 8,
    the number of possible distinct values for the count of tangent lines is 3 -/
theorem tangent_line_count_possibilities (circle1 circle2 : Circle)
    (h1 : circle1.radius = 5)
    (h2 : circle2.radius = 8)
    (h_non_overlap : circle1 ≠ circle2) :
    countDistinctValues possibleTangentLineCounts = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_count_possibilities_l3935_393535


namespace NUMINAMATH_CALUDE_equation_proof_l3935_393528

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3935_393528


namespace NUMINAMATH_CALUDE_window_wood_strip_width_l3935_393579

/-- Represents the dimensions of a glass piece in centimeters -/
structure GlassDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  glassDimensions : GlassDimensions
  woodStripWidth : ℝ

/-- Calculates the total area of glass in the window -/
def totalGlassArea (config : WindowConfig) : ℝ :=
  4 * config.glassDimensions.width * config.glassDimensions.height

/-- Calculates the total area of the window -/
def totalWindowArea (config : WindowConfig) : ℝ :=
  (2 * config.glassDimensions.width + 3 * config.woodStripWidth) *
  (2 * config.glassDimensions.height + 3 * config.woodStripWidth)

/-- Theorem: If the total area of glass equals the total area of wood,
    then the wood strip width is 20/3 cm -/
theorem window_wood_strip_width
  (config : WindowConfig)
  (h1 : config.glassDimensions.width = 30)
  (h2 : config.glassDimensions.height = 20)
  (h3 : totalGlassArea config = totalWindowArea config - totalGlassArea config) :
  config.woodStripWidth = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_window_wood_strip_width_l3935_393579


namespace NUMINAMATH_CALUDE_linear_function_property_l3935_393561

/-- A function f satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), f (x₁ + x₂) = f x₁ + f x₂

/-- Theorem: A function of the form f(x) = kx, where k is a non-zero constant,
    satisfies the property f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
theorem linear_function_property (k : ℝ) (hk : k ≠ 0) :
  LinearFunction (fun x ↦ k * x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l3935_393561


namespace NUMINAMATH_CALUDE_left_movement_denoted_negative_l3935_393527

/-- Represents the direction of movement -/
inductive Direction
| Left
| Right

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Denotes a movement as a signed real number -/
def denoteMovement (m : Movement) : ℝ :=
  match m.direction with
  | Direction.Right => m.distance
  | Direction.Left => -m.distance

theorem left_movement_denoted_negative (d : ℝ) (h : d > 0) :
  denoteMovement { distance := d, direction := Direction.Right } = d →
  denoteMovement { distance := d, direction := Direction.Left } = -d :=
by
  sorry

#check left_movement_denoted_negative

end NUMINAMATH_CALUDE_left_movement_denoted_negative_l3935_393527


namespace NUMINAMATH_CALUDE_volume_ratio_is_twenty_l3935_393540

-- Define the dimensions of the boxes
def sehee_side : ℝ := 1  -- 1 meter
def serin_width : ℝ := 0.5  -- 50 cm in meters
def serin_depth : ℝ := 0.5  -- 50 cm in meters
def serin_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volumes of the boxes
def sehee_volume : ℝ := sehee_side ^ 3
def serin_volume : ℝ := serin_width * serin_depth * serin_height

-- State the theorem
theorem volume_ratio_is_twenty :
  sehee_volume / serin_volume = 20 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_twenty_l3935_393540


namespace NUMINAMATH_CALUDE_two_wheeler_wheels_l3935_393511

theorem two_wheeler_wheels (total_wheels : ℕ) (four_wheelers : ℕ) : total_wheels = 46 ∧ four_wheelers = 11 → 
  ∃ (two_wheelers : ℕ), two_wheelers * 2 + four_wheelers * 4 = total_wheels ∧ two_wheelers * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_wheeler_wheels_l3935_393511


namespace NUMINAMATH_CALUDE_math_exam_questions_l3935_393574

theorem math_exam_questions (english_questions : ℕ) (english_time : ℕ) (math_time : ℕ) (extra_time_per_question : ℕ) : 
  english_questions = 30 →
  english_time = 60 →
  math_time = 90 →
  extra_time_per_question = 4 →
  (math_time / (english_time / english_questions + extra_time_per_question) : ℕ) = 15 := by
sorry

end NUMINAMATH_CALUDE_math_exam_questions_l3935_393574


namespace NUMINAMATH_CALUDE_enclosed_area_is_one_l3935_393524

-- Define the curves
def curve (x : ℝ) : ℝ := x^2 + 2
def line (x : ℝ) : ℝ := 3*x

-- Define the boundaries
def left_boundary : ℝ := 0
def right_boundary : ℝ := 2

-- Define the area function
noncomputable def area : ℝ := ∫ x in left_boundary..right_boundary, max (curve x - line x) 0 + max (line x - curve x) 0

-- Theorem statement
theorem enclosed_area_is_one : area = 1 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_one_l3935_393524


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3935_393560

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (∃ (b : ℝ), (1 + a * i) / (2 - i) = b * i) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3935_393560


namespace NUMINAMATH_CALUDE_mothers_age_twice_lucys_l3935_393544

/-- Given Lucy's age and her mother's age in 2012, find the year when the mother's age will be twice Lucy's age -/
theorem mothers_age_twice_lucys (lucy_age_2012 : ℕ) (mother_age_multiplier : ℕ) : 
  lucy_age_2012 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_after_2012 : ℕ),
    (lucy_age_2012 + years_after_2012) * 2 = (lucy_age_2012 * mother_age_multiplier + years_after_2012) ∧
    2012 + years_after_2012 = 2042 :=
by sorry

end NUMINAMATH_CALUDE_mothers_age_twice_lucys_l3935_393544


namespace NUMINAMATH_CALUDE_intersection_sum_l3935_393521

/-- Given two lines that intersect at (2,1), prove that a + b = 2 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) → 
  (1 = (1/3) * 2 + b) → 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3935_393521


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3935_393546

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tray of brownies -/
def tray : Dimensions := ⟨24, 20⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the tray can be divided into exactly 80 pieces -/
theorem brownie_pieces_count : (area tray) / (area piece) = 80 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3935_393546


namespace NUMINAMATH_CALUDE_quinn_reading_challenge_l3935_393519

/-- The number of books Quinn needs to read to get one free donut -/
def books_per_donut (books_per_week : ℕ) (weeks : ℕ) (total_donuts : ℕ) : ℕ :=
  (books_per_week * weeks) / total_donuts

/-- Proof that Quinn needs to read 5 books to get one free donut -/
theorem quinn_reading_challenge :
  books_per_donut 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quinn_reading_challenge_l3935_393519


namespace NUMINAMATH_CALUDE_rhombus_acute_angle_l3935_393518

-- Define a rhombus
structure Rhombus where
  -- We don't need to define all properties of a rhombus, just what we need
  acute_angle : ℝ

-- Define the plane passing through a side
structure Plane where
  -- The angles it forms with the diagonals
  angle1 : ℝ
  angle2 : ℝ

-- The main theorem
theorem rhombus_acute_angle (r : Rhombus) (p : Plane) 
  (h1 : p.angle1 = α)
  (h2 : p.angle2 = 2 * α)
  : r.acute_angle = 2 * Real.arctan (1 / (2 * Real.cos α)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_acute_angle_l3935_393518


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_equals_sqrt_3_l3935_393581

theorem sqrt_12_minus_sqrt_3_equals_sqrt_3 : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_equals_sqrt_3_l3935_393581


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_shift_l3935_393564

theorem quadratic_equation_roots_shift (a h k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 2 ∧ a * (x₁ + h)^2 + k = 0 ∧ a * (x₂ + h)^2 + k = 0) →
  (∃ y₁ y₂ : ℝ, y₁ = -2 ∧ y₂ = 3 ∧ a * (y₁ - 1 + h)^2 + k = 0 ∧ a * (y₂ - 1 + h)^2 + k = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_shift_l3935_393564


namespace NUMINAMATH_CALUDE_line_general_form_l3935_393577

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralForm where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope -3 passing through the point (1, 2),
    its general form equation is 3x + y - 5 = 0 -/
theorem line_general_form (l : Line) 
    (h1 : l.slope = -3)
    (h2 : l.point = (1, 2)) :
    ∃ (g : GeneralForm), g.a = 3 ∧ g.b = 1 ∧ g.c = -5 :=
by sorry

end NUMINAMATH_CALUDE_line_general_form_l3935_393577


namespace NUMINAMATH_CALUDE_choir_average_age_l3935_393576

theorem choir_average_age (female_count : ℕ) (male_count : ℕ) (children_count : ℕ)
  (female_avg_age : ℝ) (male_avg_age : ℝ) (children_avg_age : ℝ)
  (h_female_count : female_count = 12)
  (h_male_count : male_count = 18)
  (h_children_count : children_count = 10)
  (h_female_avg : female_avg_age = 28)
  (h_male_avg : male_avg_age = 36)
  (h_children_avg : children_avg_age = 10) :
  let total_count := female_count + male_count + children_count
  let total_age := female_count * female_avg_age + male_count * male_avg_age + children_count * children_avg_age
  total_age / total_count = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3935_393576


namespace NUMINAMATH_CALUDE_translation_of_points_l3935_393501

/-- Given two points A and B in ℝ², if A is translated to A₁, 
    then B translated by the same vector results in B₁ -/
theorem translation_of_points (A B A₁ B₁ : ℝ × ℝ) : 
  A = (-1, 0) → 
  B = (1, 2) → 
  A₁ = (2, -1) → 
  B₁.1 = B.1 + (A₁.1 - A.1) ∧ B₁.2 = B.2 + (A₁.2 - A.2) → 
  B₁ = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_points_l3935_393501


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3935_393552

-- Problem 1
theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 - (2 * x + 5) * (2 * x - 5) = 8 * x + 29 := by
  sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) : 
  (4 * a * b)^2 * (-1/4 * a^4 * b^3 * c^2) / (-4 * a^3 * b^2 * c^2) = a^3 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3935_393552


namespace NUMINAMATH_CALUDE_fixed_points_bound_l3935_393557

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Evaluate a polynomial at a point -/
def evalPoly (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Compose a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer fixed points of a polynomial -/
def numIntFixedPoints (p : IntPolynomial n) : ℕ := sorry

/-- Main theorem: The number of integer fixed points of Q is at most n -/
theorem fixed_points_bound (n k : ℕ) (p : IntPolynomial n) 
  (h1 : n > 1) (h2 : k > 0) : 
  numIntFixedPoints (composeK p k) ≤ n := by sorry

end NUMINAMATH_CALUDE_fixed_points_bound_l3935_393557


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3935_393541

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3935_393541


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l3935_393565

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 10 11))) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l3935_393565


namespace NUMINAMATH_CALUDE_time_to_find_worm_l3935_393549

/-- Given Kevin's toad feeding scenario, prove the time to find each worm. -/
theorem time_to_find_worm (num_toads : ℕ) (worms_per_toad : ℕ) (total_hours : ℕ) :
  num_toads = 8 →
  worms_per_toad = 3 →
  total_hours = 6 →
  (total_hours * 60) / (num_toads * worms_per_toad) = 15 :=
by sorry

end NUMINAMATH_CALUDE_time_to_find_worm_l3935_393549


namespace NUMINAMATH_CALUDE_range_of_m_l3935_393512

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → m ∈ Set.Icc (-4) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3935_393512


namespace NUMINAMATH_CALUDE_scale_correspondence_l3935_393507

/-- Linear relationship between p-scale and s-scale measurements -/
structure ScaleRelation where
  p_to_s : ℝ → ℝ
  s_to_p : ℝ → ℝ
  linear_p_to_s : ∀ x y : ℝ, p_to_s (x + y) = p_to_s x + p_to_s y - p_to_s 0
  linear_s_to_p : ∀ x y : ℝ, s_to_p (x + y) = s_to_p x + s_to_p y - s_to_p 0
  inverse : ∀ x : ℝ, s_to_p (p_to_s x) = x

/-- Theorem stating the relationship between p-scale and s-scale measurements -/
theorem scale_correspondence (sr : ScaleRelation) 
  (h1 : sr.p_to_s 6 = 30) 
  (h2 : sr.p_to_s 24 = 60) : 
  sr.p_to_s 48 = 100 := by
  sorry

end NUMINAMATH_CALUDE_scale_correspondence_l3935_393507


namespace NUMINAMATH_CALUDE_trapezium_area_l3935_393550

theorem trapezium_area (a b h θ : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) (hθ : θ = 30 * π / 180) :
  (a + b) / 2 * (h * Real.sin θ) = 123.5 :=
sorry

end NUMINAMATH_CALUDE_trapezium_area_l3935_393550


namespace NUMINAMATH_CALUDE_m_value_l3935_393555

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem m_value (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l3935_393555


namespace NUMINAMATH_CALUDE_stadium_sections_theorem_l3935_393543

theorem stadium_sections_theorem : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∃ (A C : ℕ), 7 * A = 11 * C ∧ N = A + C) ∧ 
  (∀ (M : ℕ), M > 0 → 
    (∃ (A C : ℕ), 7 * A = 11 * C ∧ M = A + C) → M ≥ N) ∧
  N = 18 :=
sorry

end NUMINAMATH_CALUDE_stadium_sections_theorem_l3935_393543


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3935_393580

theorem quadrilateral_area (rectangle_area shaded_triangles_area : ℝ) 
  (h1 : rectangle_area = 24)
  (h2 : shaded_triangles_area = 7.5) :
  rectangle_area - shaded_triangles_area = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3935_393580


namespace NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_r_two_satisfies_condition_l3935_393522

theorem largest_r_for_sequence_convergence (r : ℝ) :
  r > 2 →
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) ∧
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))) ∧
    (¬ ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n) :=
by sorry

theorem r_two_satisfies_condition :
  ∀ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) →
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + 2 * a (n + 1))) →
    ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_r_two_satisfies_condition_l3935_393522


namespace NUMINAMATH_CALUDE_range_of_f_l3935_393587

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 1, 2}

theorem range_of_f :
  {y : Int | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3935_393587


namespace NUMINAMATH_CALUDE_problem_solution_l3935_393591

theorem problem_solution (a : ℚ) : a + a / 3 + a / 4 = 4 → a = 48 / 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3935_393591


namespace NUMINAMATH_CALUDE_infinitely_many_representable_terms_l3935_393567

-- Define the sequence type
def PositiveIntegerSequence := ℕ → ℕ+

-- Define the property that the sequence is strictly increasing
def StrictlyIncreasing (a : PositiveIntegerSequence) : Prop :=
  ∀ k, a k < a (k + 1)

-- State the theorem
theorem infinitely_many_representable_terms 
  (a : PositiveIntegerSequence) 
  (h : StrictlyIncreasing a) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
    (∀ m ∈ S, ∃ (p q x y : ℕ), 
      p ≠ q ∧ 
      x > 0 ∧ 
      y > 0 ∧ 
      (a m : ℕ) = x * (a p : ℕ) + y * (a q : ℕ)) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_representable_terms_l3935_393567


namespace NUMINAMATH_CALUDE_towers_count_correct_l3935_393539

def number_of_towers (red green blue : ℕ) (height : ℕ) : ℕ :=
  let total := red + green + blue
  let leftout := total - height
  if leftout ≠ 1 then 0
  else
    (Nat.choose total height) *
    (Nat.factorial height / (Nat.factorial red * Nat.factorial (green - 1) * Nat.factorial blue) +
     Nat.factorial height / (Nat.factorial red * Nat.factorial green * Nat.factorial (blue - 1)) +
     Nat.factorial height / (Nat.factorial (red - 1) * Nat.factorial green * Nat.factorial blue))

theorem towers_count_correct :
  number_of_towers 3 4 4 10 = 26250 := by
  sorry

end NUMINAMATH_CALUDE_towers_count_correct_l3935_393539


namespace NUMINAMATH_CALUDE_constant_shift_invariance_l3935_393568

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def addConstant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sampleStandardDeviation (X : Fin n → ℝ) : ℝ :=
  sorry

def sampleRange (X : Fin n → ℝ) : ℝ :=
  sorry

theorem constant_shift_invariance (hc : c ≠ 0) (hY : Y = addConstant X c) :
  sampleStandardDeviation X = sampleStandardDeviation Y ∧
  sampleRange X = sampleRange Y :=
sorry

end NUMINAMATH_CALUDE_constant_shift_invariance_l3935_393568


namespace NUMINAMATH_CALUDE_f_properties_l3935_393545

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem f_properties :
  (∃ (max_val : ℝ), max_val = -4 ∧ ∀ x ≠ 1, f x ≤ max_val) ∧
  (∀ x ≠ 1, f (1 - x) + f (1 + x) = -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > 1 → f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3935_393545


namespace NUMINAMATH_CALUDE_workshop_workers_count_l3935_393573

/-- Proves that the total number of workers in a workshop is 28 given the salary conditions --/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  W = N + 7 →  -- Total workers = Non-technicians + Technicians
  W * 8000 = 7 * 14000 + N * 6000 →  -- Total salary equation
  W = 28 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l3935_393573


namespace NUMINAMATH_CALUDE_outfit_combinations_l3935_393582

/-- The number of possible outfits given the number of shirts, ties, and pants -/
def number_of_outfits (shirts ties pants : ℕ) : ℕ := shirts * ties * pants

/-- Theorem: Given 8 shirts, 6 ties, and 4 pairs of pants, the number of possible outfits is 192 -/
theorem outfit_combinations : number_of_outfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3935_393582


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3935_393590

theorem increasing_function_inequality (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : ∀ x₁ x₂, f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :
  ∀ x₁ x₂, x₁ + x₂ ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3935_393590


namespace NUMINAMATH_CALUDE_range_of_ratio_l3935_393547

theorem range_of_ratio (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0 → 
  0 ≤ y/x ∧ y/x ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_ratio_l3935_393547


namespace NUMINAMATH_CALUDE_curve_C_symmetric_about_y_axis_l3935_393551

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 4)^2)}

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem curve_C_symmetric_about_y_axis : symmetric_about_y_axis C := by
  sorry

end NUMINAMATH_CALUDE_curve_C_symmetric_about_y_axis_l3935_393551


namespace NUMINAMATH_CALUDE_number_of_divisors_of_90_l3935_393593

theorem number_of_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_90_l3935_393593


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3935_393585

/-- A line in the form kx - y - k + 1 = 0 passes through the point (1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 1 - 1 - k + 1 = 0) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3935_393585


namespace NUMINAMATH_CALUDE_f_min_bound_l3935_393504

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_min_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := by sorry

end NUMINAMATH_CALUDE_f_min_bound_l3935_393504


namespace NUMINAMATH_CALUDE_flight_time_estimate_l3935_393514

/-- The radius of the circular path in miles -/
def radius : ℝ := 3950

/-- The speed of the object in miles per hour -/
def speed : ℝ := 550

/-- The approximate value of π -/
def π_approx : ℝ := 3.14

/-- The theorem stating that the time taken to complete one revolution is approximately 45 hours -/
theorem flight_time_estimate :
  let circumference := 2 * π_approx * radius
  let exact_time := circumference / speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |exact_time - 45| < ε :=
sorry

end NUMINAMATH_CALUDE_flight_time_estimate_l3935_393514


namespace NUMINAMATH_CALUDE_first_book_has_200_words_l3935_393520

/-- The number of words in Jenny's first book --/
def first_book_words : ℕ := sorry

/-- The number of words Jenny can read per hour --/
def reading_speed : ℕ := 100

/-- The number of words in the second book --/
def second_book_words : ℕ := 400

/-- The number of words in the third book --/
def third_book_words : ℕ := 300

/-- The number of days Jenny plans to read --/
def reading_days : ℕ := 10

/-- The average number of minutes Jenny spends reading per day --/
def daily_reading_minutes : ℕ := 54

/-- Theorem stating that the first book has 200 words --/
theorem first_book_has_200_words :
  first_book_words = 200 := by sorry

end NUMINAMATH_CALUDE_first_book_has_200_words_l3935_393520
