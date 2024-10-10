import Mathlib

namespace fair_cost_calculation_l906_90634

/-- Calculate the total cost for Joe and the twins at the fair -/
theorem fair_cost_calculation (entrance_fee_under_18 : ℚ) 
  (entrance_fee_over_18_multiplier : ℚ) (group_discount : ℚ) 
  (low_thrill_under_18 : ℚ) (low_thrill_over_18 : ℚ)
  (medium_thrill_under_18 : ℚ) (medium_thrill_over_18 : ℚ)
  (high_thrill_under_18 : ℚ) (high_thrill_over_18 : ℚ)
  (joe_age : ℕ) (twin_age : ℕ)
  (joe_low : ℕ) (joe_medium : ℕ) (joe_high : ℕ)
  (twin_a_low : ℕ) (twin_a_medium : ℕ)
  (twin_b_low : ℕ) (twin_b_high : ℕ) :
  entrance_fee_under_18 = 5 →
  entrance_fee_over_18_multiplier = 1.2 →
  group_discount = 0.85 →
  low_thrill_under_18 = 0.5 →
  low_thrill_over_18 = 0.7 →
  medium_thrill_under_18 = 1 →
  medium_thrill_over_18 = 1.2 →
  high_thrill_under_18 = 1.5 →
  high_thrill_over_18 = 1.7 →
  joe_age = 30 →
  twin_age = 6 →
  joe_low = 2 →
  joe_medium = 1 →
  joe_high = 1 →
  twin_a_low = 2 →
  twin_a_medium = 1 →
  twin_b_low = 3 →
  twin_b_high = 2 →
  (entrance_fee_under_18 * entrance_fee_over_18_multiplier * group_discount + 
   2 * entrance_fee_under_18 * group_discount +
   joe_low * low_thrill_over_18 + joe_medium * medium_thrill_over_18 + 
   joe_high * high_thrill_over_18 +
   twin_a_low * low_thrill_under_18 + twin_a_medium * medium_thrill_under_18 +
   twin_b_low * low_thrill_under_18 + twin_b_high * high_thrill_under_18) = 24.4 := by
  sorry

end fair_cost_calculation_l906_90634


namespace exam_score_proof_l906_90658

theorem exam_score_proof (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 80 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 :=
by sorry

end exam_score_proof_l906_90658


namespace unsafe_trip_probability_775km_l906_90635

/-- The probability of not completing a trip safely given the probability of an accident per km and the total distance. -/
def unsafe_trip_probability (p : ℝ) (distance : ℕ) : ℝ :=
  1 - (1 - p) ^ distance

/-- Theorem stating that the probability of not completing a 775 km trip safely
    is equal to 1 - (1 - p)^775, where p is the probability of an accident per km. -/
theorem unsafe_trip_probability_775km (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  unsafe_trip_probability p 775 = 1 - (1 - p)^775 := by
  sorry

#check unsafe_trip_probability_775km

end unsafe_trip_probability_775km_l906_90635


namespace exactly_25_sixes_probability_l906_90675

/-- A cube made of 27 dice -/
structure CubeOfDice :=
  (size : Nat)
  (h_size : size = 27)

/-- The probability of a specific outcome on the surface of the cube -/
def surface_probability (c : CubeOfDice) : ℚ :=
  31 / (2^13 * 3^18)

/-- Theorem: The probability of exactly 25 sixes on the surface of a cube made of 27 dice -/
theorem exactly_25_sixes_probability (c : CubeOfDice) : 
  surface_probability c = 31 / (2^13 * 3^18) := by
  sorry

end exactly_25_sixes_probability_l906_90675


namespace count_valid_triples_l906_90637

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 48 ∧
  Nat.lcm x.val z.val = 450 ∧
  Nat.lcm y.val z.val = 600

theorem count_valid_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ+ × ℕ+ × ℕ+)),
    S.card = n ∧
    (∀ (t : ℕ+ × ℕ+ × ℕ+), t ∈ S ↔ valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end count_valid_triples_l906_90637


namespace odd_factors_of_x_squared_plus_one_l906_90665

theorem odd_factors_of_x_squared_plus_one (x : ℤ) (d : ℤ) :
  d > 0 → Odd d → (x^2 + 1) % d = 0 → ∃ h : ℤ, d = 4*h + 1 := by
  sorry

end odd_factors_of_x_squared_plus_one_l906_90665


namespace prob_four_successes_in_five_trials_l906_90689

/-- The probability of exactly 4 successes in 5 independent Bernoulli trials with p = 1/3 -/
theorem prob_four_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/3
  let k : ℕ := 4
  Nat.choose n k * p^k * (1-p)^(n-k) = 10/243 := by
  sorry

end prob_four_successes_in_five_trials_l906_90689


namespace root_difference_l906_90622

/-- The difference between the larger and smaller roots of the quadratic equation
    x^2 - 2px + (p^2 - 4p + 4) = 0, where p is a real number. -/
theorem root_difference (p : ℝ) : 
  let a := 1
  let b := -2*p
  let c := p^2 - 4*p + 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = 4 * Real.sqrt (p - 1) := by
sorry

end root_difference_l906_90622


namespace dance_event_women_count_l906_90677

/-- Proves that given the conditions of the dance event, the number of women who attended is 12 -/
theorem dance_event_women_count :
  ∀ (num_men : ℕ) (men_partners : ℕ) (women_partners : ℕ),
    num_men = 9 →
    men_partners = 4 →
    women_partners = 3 →
    ∃ (num_women : ℕ),
      num_women * women_partners = num_men * men_partners ∧
      num_women = 12 := by
  sorry

end dance_event_women_count_l906_90677


namespace valid_selections_count_l906_90629

/-- The number of ways to select 4 out of 6 people to visit 4 distinct places -/
def total_selections : ℕ := 360

/-- The number of ways where person A visits the restricted place -/
def a_restricted : ℕ := 60

/-- The number of ways where person B visits the restricted place -/
def b_restricted : ℕ := 60

/-- The number of valid selection schemes -/
def valid_selections : ℕ := total_selections - a_restricted - b_restricted

theorem valid_selections_count : valid_selections = 240 := by sorry

end valid_selections_count_l906_90629


namespace total_odd_initial_plums_l906_90683

def is_odd_initial (name : String) : Bool :=
  let initial := name.front
  let position := initial.toUpper.toNat - 'A'.toNat + 1
  position % 2 ≠ 0

def plums_picked (name : String) (amount : ℕ) : ℕ :=
  if is_odd_initial name then amount else 0

theorem total_odd_initial_plums :
  let melanie_plums := plums_picked "Melanie" 4
  let dan_plums := plums_picked "Dan" 9
  let sally_plums := plums_picked "Sally" 3
  let ben_plums := plums_picked "Ben" (2 * (4 + 9))
  let peter_plums := plums_picked "Peter" (((3 * 3) / 4) - ((3 * 1) / 4))
  melanie_plums + dan_plums + sally_plums + ben_plums + peter_plums = 7 :=
by sorry

end total_odd_initial_plums_l906_90683


namespace outfit_combinations_l906_90693

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) : shirts = 7 → ties = 6 → shirts * (ties + 1) = 49 := by
  sorry

end outfit_combinations_l906_90693


namespace negative_three_times_two_l906_90618

theorem negative_three_times_two : (-3) * 2 = -6 := by
  sorry

end negative_three_times_two_l906_90618


namespace sum_of_absolute_differences_l906_90686

theorem sum_of_absolute_differences (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end sum_of_absolute_differences_l906_90686


namespace f_discontinuities_l906_90641

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then
    if x ≠ -2 then (x^2 + 7*x + 10) / (x^2 - 4)
    else 11/4
  else 4*x - 3

theorem f_discontinuities :
  (∃ (L : ℝ), ContinuousAt (fun x => if x ≠ -2 then f x else L) (-2)) ∧
  (¬ ContinuousAt f 2) := by
  sorry

end f_discontinuities_l906_90641


namespace basketball_shots_l906_90657

theorem basketball_shots (shots_made : ℝ) (shots_missed : ℝ) :
  shots_made = 0.8 * (shots_made + shots_missed) →
  shots_missed = 4 →
  shots_made + shots_missed = 20 :=
by
  sorry

end basketball_shots_l906_90657


namespace num_valid_codes_correct_num_valid_codes_positive_l906_90610

/-- The number of possible 5-digit codes where no digit is used more than twice. -/
def num_valid_codes : ℕ := 102240

/-- A function that calculates the number of valid 5-digit codes. -/
def calculate_valid_codes : ℕ :=
  let all_different := 10 * 9 * 8 * 7 * 6
  let one_digit_repeated := 10 * (5 * 4 / 2) * 9 * 8 * 7
  let two_digits_repeated := 10 * 9 * (5 * 4 / 2) * (3 * 2 / 2) * 8
  all_different + one_digit_repeated + two_digits_repeated

/-- Theorem stating that the number of valid codes is correct. -/
theorem num_valid_codes_correct : calculate_valid_codes = num_valid_codes := by
  sorry

/-- Theorem stating that the calculated number of valid codes is positive. -/
theorem num_valid_codes_positive : 0 < num_valid_codes := by
  sorry

end num_valid_codes_correct_num_valid_codes_positive_l906_90610


namespace trisha_annual_take_home_pay_l906_90627

/-- Calculates the annual take-home pay given hourly rate, weekly hours, weeks worked, and withholding rate. -/
def annual_take_home_pay (hourly_rate : ℝ) (weekly_hours : ℝ) (weeks_worked : ℝ) (withholding_rate : ℝ) : ℝ :=
  let gross_pay := hourly_rate * weekly_hours * weeks_worked
  let withheld_amount := withholding_rate * gross_pay
  gross_pay - withheld_amount

/-- Proves that given the specified conditions, Trisha's annual take-home pay is $24,960. -/
theorem trisha_annual_take_home_pay :
  annual_take_home_pay 15 40 52 0.2 = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 0.2

end trisha_annual_take_home_pay_l906_90627


namespace william_claire_game_bounds_l906_90684

/-- A move in William's strategy -/
inductive Move
| reciprocal : Move  -- Replace y with 1/y
| increment  : Move  -- Replace y with y+1

/-- William's strategy for rearranging the numbers -/
def Strategy := List Move

/-- The result of applying a strategy to a sequence of numbers -/
def applyStrategy (s : Strategy) (xs : List ℝ) : List ℝ := sorry

/-- Predicate to check if a list is strictly increasing -/
def isStrictlyIncreasing (xs : List ℝ) : Prop := sorry

/-- The theorem to be proved -/
theorem william_claire_game_bounds :
  ∃ (A B : ℝ) (hA : A > 0) (hB : B > 0),
    ∀ (n : ℕ) (hn : n > 1),
      -- Part (a): William can always succeed in at most An log n moves
      (∀ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∃ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) ∧
          s.length ≤ A * n * Real.log n) ∧
      -- Part (b): Claire can force William to use at least Bn log n moves
      (∃ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∀ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) →
          s.length ≥ B * n * Real.log n) :=
sorry

end william_claire_game_bounds_l906_90684


namespace f_84_value_l906_90621

def is_increasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, f (n + 1) > f n

def multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

def special_condition (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → f m = n ∨ f n = m

theorem f_84_value (f : ℕ+ → ℕ+)
  (h_inc : is_increasing f)
  (h_mult : multiplicative f)
  (h_special : special_condition f) :
  f 84 = 1764 := by
  sorry

end f_84_value_l906_90621


namespace digit_difference_base_4_9_l906_90672

theorem digit_difference_base_4_9 (n : ℕ) (h : n = 523) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by sorry

end digit_difference_base_4_9_l906_90672


namespace geometric_progression_special_ratio_l906_90659

/-- A geometric progression where each term is positive and any term is equal to the sum of the next three following terms has a common ratio that satisfies r³ + r² + r - 1 = 0. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →  -- Any term equals sum of next three
  r^3 + r^2 + r - 1 = 0 := by
sorry

end geometric_progression_special_ratio_l906_90659


namespace smallest_three_digit_solution_l906_90652

theorem smallest_three_digit_solution :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    77 * n ≡ 231 [MOD 385] ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ 77 * m ≡ 231 [MOD 385] → false) ∧
    n = 113 := by
  sorry

end smallest_three_digit_solution_l906_90652


namespace t_equality_and_inequality_l906_90691

/-- The function t(n, s) represents the maximum number of edges in a graph with n vertices
    that does not contain s independent vertices. -/
noncomputable def t (n s : ℕ) : ℕ := sorry

/-- Theorem stating the equality and inequality for t(n, s) -/
theorem t_equality_and_inequality (n s : ℕ) : 
  t (n - s) s + (n - s) * (s - 1) + (s.choose 2) = t n s ∧ 
  t n s ≤ ⌊((s - 1 : ℚ) / (2 * s : ℚ)) * (n^2 : ℚ)⌋ := by sorry

end t_equality_and_inequality_l906_90691


namespace susan_spending_l906_90636

def carnival_spending (initial_budget : ℝ) (food_cost : ℝ) : Prop :=
  let ride_cost : ℝ := 3 * food_cost
  let game_cost : ℝ := (1 / 4) * initial_budget
  let total_spent : ℝ := food_cost + ride_cost + game_cost
  let remaining : ℝ := initial_budget - total_spent
  remaining = 0

theorem susan_spending :
  carnival_spending 80 15 := by
  sorry

end susan_spending_l906_90636


namespace min_value_quadratic_l906_90663

theorem min_value_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → b ≤ x) → b = 4 := by
  sorry

end min_value_quadratic_l906_90663


namespace new_species_growth_pattern_l906_90685

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J
  | S

/-- Represents a species in a new area --/
structure Species where
  isNew : Bool
  populationSize : ℕ → ℕ  -- population size as a function of time
  growthPattern : List GrowthShape
  kValue : ℕ

/-- The maximum population allowed by environmental conditions --/
def environmentalCapacity (s : Species) : ℕ := s.kValue

theorem new_species_growth_pattern (s : Species) 
  (h1 : s.isNew = true) 
  (h2 : ∀ t, s.populationSize (t + 1) ≠ s.populationSize t) 
  (h3 : s.growthPattern.length ≥ 2) 
  (h4 : ∃ t, ∀ t' ≥ t, s.populationSize t' = s.kValue) 
  (h5 : s.kValue = environmentalCapacity s) :
  s.growthPattern = [GrowthShape.J, GrowthShape.S] := by
  sorry

end new_species_growth_pattern_l906_90685


namespace cos_270_degrees_l906_90608

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end cos_270_degrees_l906_90608


namespace line_equation_l906_90646

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A line passes through the origin if its y-intercept is 0 -/
def passes_through_origin (l : Line) : Prop :=
  l.y_intercept = 0

/-- A line has equal x and y intercepts if y_intercept = -slope * y_intercept -/
def equal_intercepts (l : Line) : Prop :=
  l.y_intercept = -l.slope * l.y_intercept

/-- The main theorem -/
theorem line_equation (l m : Line) :
  passes_through_origin l →
  parallel l m →
  equal_intercepts m →
  l.slope = -1 :=
sorry

end line_equation_l906_90646


namespace ab_plus_one_neq_a_plus_b_l906_90645

theorem ab_plus_one_neq_a_plus_b (a b : ℝ) : ab + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end ab_plus_one_neq_a_plus_b_l906_90645


namespace map_area_calculation_l906_90631

/-- Given a map scale and an area on the map, calculate the actual area -/
theorem map_area_calculation (scale : ℝ) (map_area : ℝ) (actual_area : ℝ) :
  scale = 1 / 50000 →
  map_area = 100 →
  actual_area = 2.5 * 10^7 →
  map_area / actual_area = scale^2 :=
by sorry

end map_area_calculation_l906_90631


namespace parallel_vectors_l906_90687

theorem parallel_vectors (a b : ℝ × ℝ) :
  a.1 = 2 ∧ a.2 = -1 ∧ b.2 = 3 ∧ a.1 * b.2 = a.2 * b.1 → b.1 = -6 :=
sorry

end parallel_vectors_l906_90687


namespace average_salary_problem_l906_90698

theorem average_salary_problem (salary_raj roshan : ℕ) : 
  (salary_raj + roshan) / 2 = 4000 →
  ((salary_raj + roshan + 7000) : ℚ) / 3 = 5000 := by
  sorry

end average_salary_problem_l906_90698


namespace sufficient_not_necessary_condition_l906_90617

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficient_not_necessary_condition_l906_90617


namespace machine_output_2023_l906_90653

/-- A function that computes the output of Ava's machine for a four-digit number -/
def machine_output (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * b + c * d

/-- Theorem stating that the machine output for 2023 is 6 -/
theorem machine_output_2023 : machine_output 2023 = 6 := by
  sorry

end machine_output_2023_l906_90653


namespace correct_regression_sequence_l906_90660

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | CalculateCorrelation
  | DrawScatterPlot

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateCorrelation,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Theorem stating that the defined sequence is correct
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.CollectData,
                     RegressionStep.DrawScatterPlot,
                     RegressionStep.CalculateCorrelation,
                     RegressionStep.CalculateEquation,
                     RegressionStep.InterpretEquation] :=
by sorry

end correct_regression_sequence_l906_90660


namespace man_against_stream_speed_l906_90616

/-- Represents the speed of a man rowing a boat in different conditions -/
structure BoatSpeed where
  stillWaterRate : ℝ
  withStreamSpeed : ℝ

/-- Calculates the speed of the boat against the stream -/
def againstStreamSpeed (bs : BoatSpeed) : ℝ :=
  abs (2 * bs.stillWaterRate - bs.withStreamSpeed)

/-- Theorem: Given the man's rate in still water and speed with the stream,
    prove that his speed against the stream is 12 km/h -/
theorem man_against_stream_speed (bs : BoatSpeed)
    (h1 : bs.stillWaterRate = 7)
    (h2 : bs.withStreamSpeed = 26) :
    againstStreamSpeed bs = 12 := by
  sorry

end man_against_stream_speed_l906_90616


namespace geometric_sequence_sum_l906_90628

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a → a 3 + a 5 = 20 → a 4 = 8 → a 2 + a 6 = 34 := by
  sorry

end geometric_sequence_sum_l906_90628


namespace fraction_equality_l906_90656

theorem fraction_equality (x : ℝ) : 
  (3/4 : ℝ) * (1/2 : ℝ) * (2/5 : ℝ) * x = 750.0000000000001 → x = 5000.000000000001 := by
  sorry

end fraction_equality_l906_90656


namespace vector_computation_l906_90673

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -5, 1]
  let v2 : Fin 3 → ℝ := ![-1, 4, -2]
  let v3 : Fin 3 → ℝ := ![2, -1, 3]
  2 • v1 + 3 • v2 - v3 = ![1, 3, -7] := by
  sorry

end vector_computation_l906_90673


namespace min_value_of_a_l906_90650

theorem min_value_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0 ∧ 
                 9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) →
  a ≥ 0 ∧ ∀ ε > 0, ∃ a' : ℝ, a' < ε ∧ 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a') * 3 * x₁ + 4 = 0 ∧ 
                  9 * x₂ - (4 + a') * 3 * x₂ + 4 = 0 :=
by sorry

end min_value_of_a_l906_90650


namespace train_speed_problem_l906_90620

/-- Proves that given a train journey of 3x km, where x km is traveled at 50 kmph
    and 2x km is traveled at speed v, and the average speed for the entire journey
    is 25 kmph, the speed v must be 20 kmph. -/
theorem train_speed_problem (x : ℝ) (v : ℝ) (h_x_pos : x > 0) :
  (x / 50 + 2 * x / v = 3 * x / 25) → v = 20 := by
  sorry

#check train_speed_problem

end train_speed_problem_l906_90620


namespace graph_is_two_parabolas_l906_90607

/-- The equation of the conic sections -/
def equation (x y : ℝ) : Prop :=
  y^6 - 9*x^6 = 3*y^3 - 1

/-- A parabola in cubic form -/
def cubic_parabola (x y : ℝ) (a b : ℝ) : Prop :=
  y^3 + a*x^3 = b

/-- The graph consists of two parabolas -/
theorem graph_is_two_parabolas :
  ∃ (a₁ b₁ a₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (cubic_parabola x y a₁ b₁ ∨ cubic_parabola x y a₂ b₂)) ∧
    a₁ ≠ a₂ :=
  sorry

end graph_is_two_parabolas_l906_90607


namespace opponent_total_score_l906_90667

theorem opponent_total_score (team_scores : List Nat) 
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : ∃ lost_games : List Nat, lost_games.length = 6 ∧ 
    ∀ score ∈ lost_games, score + 1 ∈ team_scores)
  (h3 : ∃ double_score_games : List Nat, double_score_games.length = 5 ∧ 
    ∀ score ∈ double_score_games, 2 * (score / 2) ∈ team_scores)
  (h4 : ∃ tie_score : Nat, tie_score ∈ team_scores ∧ 
    tie_score ∉ (Classical.choose h2) ∧ tie_score ∉ (Classical.choose h3)) :
  (team_scores.sum + 6 - (Classical.choose h3).sum / 2 - (Classical.choose h4)) = 69 := by
sorry

end opponent_total_score_l906_90667


namespace min_value_M_l906_90680

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let M := (((a / (b + c)) ^ (1/4)) + ((b / (c + a)) ^ (1/4)) + ((c / (a + b)) ^ (1/4)) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 * (8 ^ (1/4))) / 8 := by
  sorry

end min_value_M_l906_90680


namespace seal_earnings_l906_90605

/-- Calculates the total earnings of a musician over a given number of years -/
def musician_earnings (songs_per_month : ℕ) (earnings_per_song : ℕ) (years : ℕ) : ℕ :=
  songs_per_month * earnings_per_song * 12 * years

/-- Proves that Seal's earnings over 3 years, given the specified conditions, equal $216,000 -/
theorem seal_earnings : musician_earnings 3 2000 3 = 216000 := by
  sorry

end seal_earnings_l906_90605


namespace perpendicular_bisector_b_l906_90639

/-- The value of b for which the line x + y = b is the perpendicular bisector
    of the line segment connecting (1,4) and (7,10) -/
theorem perpendicular_bisector_b : ∃ b : ℝ,
  (∀ x y : ℝ, x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = 9) ∧ 
    ((x - 1) * (7 - 1) + (y - 4) * (10 - 4) = 0)) ∧
  b = 11 := by
  sorry


end perpendicular_bisector_b_l906_90639


namespace missing_number_l906_90623

theorem missing_number (x z : ℕ) 
  (h1 : x * 2 = 8)
  (h2 : 2 * z = 16)
  (h3 : 8 * 7 = 56)
  (h4 : 16 * 7 = 112) :
  x = 4 := by
  sorry

end missing_number_l906_90623


namespace ellipse_focus_k_value_l906_90613

/-- An ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2) has k = -1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∀ x y : ℝ, 5 * x^2 - k * y^2 = 5) →  -- Ellipse equation
  (∃ x : ℝ, (x, 2) ∈ {p : ℝ × ℝ | 5 * p.1^2 - k * p.2^2 = 5}) →  -- Focus at (0, 2)
  k = -1 :=
by sorry

end ellipse_focus_k_value_l906_90613


namespace product_sum_theorem_l906_90654

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120 →
  a + b + c + d + e = 27 := by
sorry

end product_sum_theorem_l906_90654


namespace n_gon_determination_l906_90682

/-- The number of elements required to determine an n-gon uniquely -/
def elementsRequired (n : ℕ) : ℕ := 2 * n - 3

/-- The minimum number of sides required among the elements -/
def minSidesRequired (n : ℕ) : ℕ := n - 2

/-- Predicate to check if a number is at least 3 -/
def isAtLeastThree (n : ℕ) : Prop := n ≥ 3

/-- Theorem stating the number of elements and minimum sides required to determine an n-gon -/
theorem n_gon_determination (n : ℕ) (h : isAtLeastThree n) :
  elementsRequired n = 2 * n - 3 ∧ minSidesRequired n = n - 2 :=
by sorry

end n_gon_determination_l906_90682


namespace vector_dot_product_sum_l906_90671

theorem vector_dot_product_sum (a b : ℝ × ℝ) : 
  a = (1/2, Real.sqrt 3/2) → 
  b = (-Real.sqrt 3/2, 1/2) → 
  (a.1 + b.1, a.2 + b.2) • a = 1 := by sorry

end vector_dot_product_sum_l906_90671


namespace sin_negative_600_degrees_l906_90690

theorem sin_negative_600_degrees : Real.sin (- 600 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end sin_negative_600_degrees_l906_90690


namespace cos_B_value_triangle_area_l906_90644

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.b = t.c ∧ 2 * Real.sin t.B = Real.sqrt 3 * Real.sin t.A

-- Theorem for part (i)
theorem cos_B_value (t : Triangle) (h : SpecialTriangle t) : 
  Real.cos t.B = Real.sqrt 3 / 3 := by
  sorry

-- Theorem for part (ii)
theorem triangle_area (t : Triangle) (h : SpecialTriangle t) (ha : t.a = 2) :
  (1/2) * t.a * t.b * Real.sin t.B = Real.sqrt 2 := by
  sorry

end cos_B_value_triangle_area_l906_90644


namespace geometric_series_sum_l906_90688

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 4) →
  x = 3/4 := by
sorry

end geometric_series_sum_l906_90688


namespace parabola_focus_l906_90692

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y = 2 * x^2 + 4 * x + 5

-- Define the focus of a parabola
def is_focus (x y : ℝ) (f : ℝ × ℝ) : Prop :=
  f.1 = x ∧ f.2 = y

-- Theorem statement
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), is_focus (-1) (25/8) f ∧
  ∀ (x y : ℝ), parabola_equation x y →
  is_focus x y f :=
sorry

end parabola_focus_l906_90692


namespace fraction_sum_l906_90614

theorem fraction_sum : (3 : ℚ) / 9 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end fraction_sum_l906_90614


namespace alpha_beta_difference_bounds_l906_90669

theorem alpha_beta_difference_bounds (α β : ℝ) (h : -1 < α ∧ α < β ∧ β < 1) :
  -2 < α - β ∧ α - β < 0 := by
sorry

end alpha_beta_difference_bounds_l906_90669


namespace total_distance_is_151_l906_90624

/-- Calculates the total distance Amy biked in a week -/
def total_distance_biked : ℝ :=
  let monday_distance : ℝ := 12
  let tuesday_distance : ℝ := 2 * monday_distance - 3
  let wednesday_distance : ℝ := 2 * 11
  let thursday_distance : ℝ := wednesday_distance + 2
  let friday_distance : ℝ := thursday_distance + 2
  let saturday_distance : ℝ := friday_distance + 2
  let sunday_distance : ℝ := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + 
  friday_distance + saturday_distance + sunday_distance

theorem total_distance_is_151 : total_distance_biked = 151 := by
  sorry

end total_distance_is_151_l906_90624


namespace sqrt_two_equality_l906_90662

theorem sqrt_two_equality : (2 : ℝ) / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_two_equality_l906_90662


namespace milo_run_distance_l906_90611

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard rolling speed in miles per hour -/
def milo_roll_speed : ℝ := milo_run_speed * 2

/-- Cory's wheelchair driving speed in miles per hour -/
def cory_drive_speed : ℝ := 12

/-- Time Milo runs in hours -/
def run_time : ℝ := 2

theorem milo_run_distance : 
  (milo_roll_speed = milo_run_speed * 2) →
  (cory_drive_speed = milo_roll_speed * 2) →
  (cory_drive_speed = 12) →
  (milo_run_speed * run_time = 6) :=
by
  sorry


end milo_run_distance_l906_90611


namespace isosceles_from_cosine_relation_l906_90603

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  angle_bounds : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

/-- A triangle is isosceles if it has at least two equal sides -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The main theorem: if a = 2b cos C, then the triangle is isosceles -/
theorem isosceles_from_cosine_relation (t : Triangle) (h : t.a = 2 * t.b * Real.cos t.C) :
  t.isIsosceles := by
  sorry

end isosceles_from_cosine_relation_l906_90603


namespace min_value_sum_of_inverses_l906_90696

theorem min_value_sum_of_inverses (x y z p q r : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0)
  (sum_eq_10 : x + y + z + p + q + r = 10) :
  1/x + 9/y + 4/z + 25/p + 16/q + 36/r ≥ 441/10 ∧ 
  ∃ (x' y' z' p' q' r' : ℝ), 
    x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    x' + y' + z' + p' + q' + r' = 10 ∧
    1/x' + 9/y' + 4/z' + 25/p' + 16/q' + 36/r' = 441/10 :=
by sorry

end min_value_sum_of_inverses_l906_90696


namespace monotone_decreasing_implies_a_geq_2_l906_90666

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a)

-- State the theorem
theorem monotone_decreasing_implies_a_geq_2 (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y) → a ≥ 2 := by
  sorry

end monotone_decreasing_implies_a_geq_2_l906_90666


namespace range_of_2x_plus_y_min_value_of_c_l906_90630

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem 1: Range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, Circle x y → 1 - Real.sqrt 2 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 2 :=
sorry

-- Theorem 2: Minimum value of c
theorem min_value_of_c :
  (∃ c : ℝ, ∀ x y : ℝ, Circle x y → x + y + c > 0) ∧
  (∀ c' : ℝ, (∀ x y : ℝ, Circle x y → x + y + c' > 0) → c' ≥ -1) :=
sorry

end range_of_2x_plus_y_min_value_of_c_l906_90630


namespace skaters_meeting_time_l906_90609

/-- The time it takes for two skaters to meet on a circular rink -/
theorem skaters_meeting_time 
  (rink_circumference : ℝ) 
  (speed_skater1 : ℝ) 
  (speed_skater2 : ℝ) 
  (h1 : rink_circumference = 3000) 
  (h2 : speed_skater1 = 100) 
  (h3 : speed_skater2 = 150) : 
  rink_circumference / (speed_skater1 + speed_skater2) = 12 := by
  sorry

#check skaters_meeting_time

end skaters_meeting_time_l906_90609


namespace merchant_profit_l906_90699

theorem merchant_profit (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 40 →
  discount_percent = 10 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 26 := by
sorry

end merchant_profit_l906_90699


namespace number_of_divisors_36_l906_90670

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end number_of_divisors_36_l906_90670


namespace wickets_before_last_match_is_55_l906_90638

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the bowler took 55 wickets before the last match -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats)
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 :=
  sorry

end wickets_before_last_match_is_55_l906_90638


namespace hostel_provision_days_l906_90615

/-- Calculates the initial number of days provisions were planned for in a hostel. -/
def initial_provision_days (initial_men : ℕ) (men_left : ℕ) (days_after_leaving : ℕ) : ℕ :=
  ((initial_men - men_left) * days_after_leaving) / initial_men

/-- Theorem stating that given the conditions, the initial provision days is 32. -/
theorem hostel_provision_days :
  initial_provision_days 250 50 40 = 32 := by
  sorry

#eval initial_provision_days 250 50 40

end hostel_provision_days_l906_90615


namespace angle_expression_value_l906_90642

theorem angle_expression_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end angle_expression_value_l906_90642


namespace quadrilateral_is_parallelogram_l906_90601

-- Define a quadrilateral in 2D space
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to check if a quadrilateral is a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  let BA := (q.B.1 - q.A.1, q.B.2 - q.A.2)
  let DC := (q.D.1 - q.C.1, q.D.2 - q.C.2)
  BA = DC

-- Theorem statement
theorem quadrilateral_is_parallelogram (q : Quadrilateral) (O : ℝ × ℝ) :
  (q.A.1 - O.1, q.A.2 - O.2) + (q.C.1 - O.1, q.C.2 - O.2) =
  (q.B.1 - O.1, q.B.2 - O.2) + (q.D.1 - O.1, q.D.2 - O.2) →
  is_parallelogram q :=
by
  sorry

end quadrilateral_is_parallelogram_l906_90601


namespace unique_outstanding_defeats_all_l906_90655

/-- Represents a tournament with n participants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  n_ge_three : n ≥ 3
  -- Defeat relation
  defeats : Fin n → Fin n → Prop
  -- Every match has a definite winner
  winner_exists : ∀ i j : Fin n, i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)

/-- Definition of an outstanding participant -/
def is_outstanding (t : Tournament n) (a : Fin n) : Prop :=
  ∀ b : Fin n, b ≠ a → t.defeats a b ∨ ∃ c : Fin n, t.defeats c b ∧ t.defeats a c

/-- The main theorem -/
theorem unique_outstanding_defeats_all (t : Tournament n) (a : Fin n) :
  (∀ b : Fin n, b ≠ a → is_outstanding t b → b = a) →
  is_outstanding t a →
  ∀ b : Fin n, b ≠ a → t.defeats a b :=
by sorry

end unique_outstanding_defeats_all_l906_90655


namespace find_n_l906_90651

theorem find_n : ∃ n : ℤ, 3^3 - 7 = 4^2 + 2 + n ∧ n = 2 := by
  sorry

end find_n_l906_90651


namespace equation_holds_l906_90674

theorem equation_holds : (8 - 2) + 5 - (3 - 1) = 9 := by
  sorry

end equation_holds_l906_90674


namespace floor_paving_cost_l906_90606

theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : cost_per_sqm = 600) : 
  length * width * cost_per_sqm = 12375 := by
sorry

end floor_paving_cost_l906_90606


namespace symmetry_implies_values_l906_90604

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

/-- The theorem states that if point P with coordinates (a-b, 2a+b) is symmetric to point Q (3, 2) with respect to the y-axis, then a = -1/3 and b = 8/3. -/
theorem symmetry_implies_values (a b : ℝ) :
  symmetric_y_axis (a - b, 2 * a + b) (3, 2) →
  a = -1/3 ∧ b = 8/3 := by
sorry

end symmetry_implies_values_l906_90604


namespace deal_or_no_deal_probability_l906_90695

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 10 →
  eliminated_boxes = 20 →
  (total_boxes - eliminated_boxes : ℚ) / 2 ≤ high_value_boxes :=
by sorry

end deal_or_no_deal_probability_l906_90695


namespace symmetry_properties_l906_90649

noncomputable def f (a b c p q x : ℝ) : ℝ := (a * 4^x + b * 2^x + c) / (p * 2^x + q)

theorem symmetry_properties (a b c p q : ℝ) :
  (p = 0 ∧ q ≠ 0 ∧ a^2 + b^2 ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∧
    ¬(∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p ≠ 0 ∧ q = 0 ∧ a * c ≠ 0 →
    (∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∨
    (∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p * q ≠ 0 ∧ a = 0 ∧ b^2 + c^2 ≠ 0 →
    ∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t) ∧
  (p * q ≠ 0 ∧ a ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x))) :=
by sorry

end symmetry_properties_l906_90649


namespace scientific_notation_13000_l906_90619

theorem scientific_notation_13000 : 13000 = 1.3 * (10 ^ 4) := by
  sorry

end scientific_notation_13000_l906_90619


namespace min_value_reciprocal_sum_l906_90640

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end min_value_reciprocal_sum_l906_90640


namespace replaced_person_weight_l906_90694

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - (initial_count * average_increase)

/-- Theorem stating the weight of the replaced person under given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 97 4 = 65 := by
  sorry

end replaced_person_weight_l906_90694


namespace cos_x_value_l906_90679

theorem cos_x_value (x : Real) (h1 : Real.tan (x + Real.pi / 4) = 2) 
  (h2 : x ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) : 
  Real.cos x = - (3 * Real.sqrt 10) / 10 := by
  sorry

end cos_x_value_l906_90679


namespace frog_walk_probability_l906_90625

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the 6x6 grid -/
def Grid := {p : Point // p.x ≤ 6 ∧ p.y ≤ 6}

/-- Defines the possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a random walk on the grid -/
def RandomWalk := List Direction

/-- Checks if a point is on the boundary of the grid -/
def isBoundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 6

/-- Checks if a point is on the top or bottom horizontal side of the grid -/
def isHorizontalSide (p : Point) : Bool :=
  p.y = 0 ∨ p.y = 6

/-- Calculates the probability of ending on a horizontal side -/
noncomputable def probabilityHorizontalSide (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_walk_probability :
  probabilityHorizontalSide ⟨2, 3⟩ = 8 / 25 := by sorry

end frog_walk_probability_l906_90625


namespace f_2_eq_125_l906_90648

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^4 + 2x^3 - 4x + 5 -/
def f (x : ℝ) : ℝ :=
  horner [5, -4, 0, 2, 3, 2] x

theorem f_2_eq_125 : f 2 = 125 := by
  sorry

end f_2_eq_125_l906_90648


namespace nonagon_diagonal_intersection_probability_l906_90612

/-- The probability of two randomly chosen diagonals intersecting in a regular nonagon -/
theorem nonagon_diagonal_intersection_probability :
  let n : ℕ := 9  -- number of sides in a nonagon
  let total_diagonals : ℕ := n.choose 2 - n
  let diagonal_pairs : ℕ := total_diagonals.choose 2
  let intersecting_pairs : ℕ := n.choose 4
  (intersecting_pairs : ℚ) / diagonal_pairs = 14 / 39 :=
by sorry


end nonagon_diagonal_intersection_probability_l906_90612


namespace specific_polygon_area_l906_90697

/-- A polygon on a grid where dots are spaced one unit apart both horizontally and vertically -/
structure GridPolygon where
  vertices : List (ℤ × ℤ)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℚ :=
  sorry

/-- The specific polygon described in the problem -/
def specificPolygon : GridPolygon :=
  { vertices := [
    (0, 0), (10, 0), (20, 10), (30, 0), (40, 0),
    (30, 10), (30, 20), (20, 30), (20, 40), (10, 40),
    (0, 40), (0, 10)
  ] }

/-- Theorem stating that the area of the specific polygon is 31.5 square units -/
theorem specific_polygon_area :
  area specificPolygon = 31.5 := by sorry

end specific_polygon_area_l906_90697


namespace range_of_sum_equal_product_l906_90643

theorem range_of_sum_equal_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = x * y) :
  x + y ≥ 4 ∧ ∀ z ≥ 4, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + y = z :=
by sorry

end range_of_sum_equal_product_l906_90643


namespace sum_of_solutions_quadratic_l906_90600

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 11) → (x + x = 8) :=
by
  sorry

end sum_of_solutions_quadratic_l906_90600


namespace second_division_remainder_l906_90647

theorem second_division_remainder (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 18291 = 1 :=
by sorry

end second_division_remainder_l906_90647


namespace number_of_walls_l906_90668

/-- Given the following conditions:
  - Each wall has 30 bricks in a single row
  - There are 50 rows in each wall
  - 3000 bricks will be used to make all the walls
  Prove that the number of walls that can be built is 2. -/
theorem number_of_walls (bricks_per_row : ℕ) (rows_per_wall : ℕ) (total_bricks : ℕ) :
  bricks_per_row = 30 →
  rows_per_wall = 50 →
  total_bricks = 3000 →
  total_bricks / (bricks_per_row * rows_per_wall) = 2 :=
by sorry

end number_of_walls_l906_90668


namespace simplify_expression_l906_90664

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 := by
  sorry

end simplify_expression_l906_90664


namespace circle_trajectory_and_intersection_line_l906_90626

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 - y^2 = 1

-- Define the curve C (trajectory of center of circle P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem circle_trajectory_and_intersection_line :
  ∀ (x_A y_A x_B y_B : ℝ),
  -- Circle P is internally tangent to both E and F
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x_P y_P : ℝ), (x_P - x_A)^2 + (y_P - y_A)^2 = r^2 →
      (∃ (x_E y_E : ℝ), circle_E x_E y_E ∧ (x_P - x_E)^2 + (y_P - y_E)^2 = (5 - r)^2) ∧
      (∃ (x_F y_F : ℝ), circle_F x_F y_F ∧ (x_P - x_F)^2 + (y_P - y_F)^2 = (r - 1)^2))) →
  -- A and B are on curve C
  curve_C x_A y_A →
  curve_C x_B y_B →
  -- M is the midpoint of AB
  point_M = ((x_A + x_B)/2, (y_A + y_B)/2) →
  -- A, B are on line l
  line_l x_A y_A →
  line_l x_B y_B →
  -- The equation of curve C is correct
  (∀ (x y : ℝ), curve_C x y ↔ x^2/4 + y^2 = 1) ∧
  -- The equation of line l is correct
  (∀ (x y : ℝ), line_l x y ↔ x + 4*y - 5 = 0) :=
by sorry


end circle_trajectory_and_intersection_line_l906_90626


namespace c_investment_is_2000_l906_90661

/-- Represents a partnership investment and profit distribution --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that under given conditions, C's investment is 2000 --/
theorem c_investment_is_2000 (p : Partnership) 
  (h1 : p.a_investment = 8000)
  (h2 : p.b_investment = 4000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = 
        p.c_investment * p.total_profit) : 
  p.c_investment = 2000 := by
  sorry

#check c_investment_is_2000

end c_investment_is_2000_l906_90661


namespace evaluate_expression_l906_90681

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) :
  2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end evaluate_expression_l906_90681


namespace farm_leg_count_l906_90602

def farm_animals : ℕ := 13
def chickens : ℕ := 4
def chicken_legs : ℕ := 2
def buffalo_legs : ℕ := 4

theorem farm_leg_count : 
  (chickens * chicken_legs) + ((farm_animals - chickens) * buffalo_legs) = 44 := by
  sorry

end farm_leg_count_l906_90602


namespace complex_number_opposite_parts_l906_90676

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end complex_number_opposite_parts_l906_90676


namespace initial_cards_l906_90632

theorem initial_cards (x : ℚ) : 
  (x ≥ 0) → 
  (3 * (1/2) * ((x/3) + (4/3)) = 34) → 
  (x = 64) := by
sorry

end initial_cards_l906_90632


namespace max_planes_from_points_l906_90678

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The theorem stating the maximum number of planes determined by the points -/
theorem max_planes_from_points :
  choose num_points 3 = 455 := by sorry

end max_planes_from_points_l906_90678


namespace gcd_cube_plus_27_and_plus_3_l906_90633

theorem gcd_cube_plus_27_and_plus_3 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end gcd_cube_plus_27_and_plus_3_l906_90633
