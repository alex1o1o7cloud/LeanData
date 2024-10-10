import Mathlib

namespace x_equals_one_l2015_201587

theorem x_equals_one :
  ∀ x : ℝ,
  ((x^31) / (5^31)) * ((x^16) / (4^16)) = 1 / (2 * (10^31)) →
  x = 1 :=
by
  sorry

end x_equals_one_l2015_201587


namespace people_in_line_l2015_201575

/-- The number of people in a line is equal to the number of people behind the first passenger plus one. -/
theorem people_in_line (people_behind : ℕ) : ℕ := 
  people_behind + 1

#check people_in_line 10 -- Should evaluate to 11

end people_in_line_l2015_201575


namespace removal_gives_desired_average_l2015_201577

def original_list : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def removed_number : ℕ := 5
def desired_average : ℚ := 10.5

theorem removal_gives_desired_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = desired_average := by
  sorry

end removal_gives_desired_average_l2015_201577


namespace probability_point_between_F_and_G_l2015_201528

/-- Given a line segment AB with points A, E, F, G, B placed consecutively,
    where AB = 4AE and AB = 8BF, the probability that a randomly selected
    point on AB lies between F and G is 1/2. -/
theorem probability_point_between_F_and_G (AB AE BF FG : ℝ) : 
  AB > 0 → AB = 4 * AE → AB = 8 * BF → FG = AB / 2 → FG / AB = 1 / 2 := by sorry

end probability_point_between_F_and_G_l2015_201528


namespace soccer_league_games_l2015_201580

/-- Calculates the total number of games in a soccer league --/
def total_games (n : ℕ) (k : ℕ) : ℕ :=
  -- Regular season games
  n * (n - 1) +
  -- Playoff games (single elimination format)
  (k - 1)

/-- Theorem stating the total number of games in the soccer league --/
theorem soccer_league_games :
  total_games 20 8 = 767 :=
by
  -- The proof goes here
  sorry

end soccer_league_games_l2015_201580


namespace parabola_c_value_l2015_201542

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 3 = -5 →   -- Vertex at (3, -5)
  p.y_at 4 = -3 →   -- Passes through (4, -3)
  p.c = 13 := by
  sorry

end parabola_c_value_l2015_201542


namespace girls_joined_team_l2015_201578

/-- Proves that 7 girls joined the track team given the initial and final conditions --/
theorem girls_joined_team (initial_girls : ℕ) (initial_boys : ℕ) (boys_quit : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → initial_boys = 15 → boys_quit = 4 → final_total = 36 →
  final_total - (initial_girls + (initial_boys - boys_quit)) = 7 := by
sorry

end girls_joined_team_l2015_201578


namespace max_reciprocal_negative_l2015_201588

theorem max_reciprocal_negative (B : Set ℝ) (a₀ : ℝ) :
  (B.Nonempty) →
  (0 ∉ B) →
  (∀ x ∈ B, x ≤ a₀) →
  (a₀ ∈ B) →
  (a₀ < 0) →
  (∀ x ∈ B, -x⁻¹ ≤ -a₀⁻¹) ∧ (-a₀⁻¹ ∈ {-x⁻¹ | x ∈ B}) :=
by sorry

end max_reciprocal_negative_l2015_201588


namespace binomial_8_choose_5_l2015_201519

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l2015_201519


namespace prob_first_second_given_two_fail_l2015_201572

-- Define the failure probabilities for each component
def p1 : ℝ := 0.2
def p2 : ℝ := 0.4
def p3 : ℝ := 0.3

-- Define the probability of two components failing
def prob_two_fail : ℝ := p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the probability of the first and second components failing
def prob_first_second_fail : ℝ := p1 * p2 * (1 - p3)

-- Theorem statement
theorem prob_first_second_given_two_fail : 
  prob_first_second_fail / prob_two_fail = 0.3 := by sorry

end prob_first_second_given_two_fail_l2015_201572


namespace stating_student_marks_theorem_l2015_201540

/-- 
A function that calculates the total marks secured in an examination
given the following parameters:
- total_questions: The total number of questions in the exam
- correct_answers: The number of questions answered correctly
- marks_per_correct: The number of marks awarded for each correct answer
- marks_per_wrong: The number of marks deducted for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_per_wrong)

/-- 
Theorem stating that given the specific conditions of the exam,
the student secures 140 marks in total.
-/
theorem student_marks_theorem :
  calculate_total_marks 60 40 4 1 = 140 := by
  sorry

end stating_student_marks_theorem_l2015_201540


namespace max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2015_201501

/-- A quadrilateral with one angle of 120 degrees can have at most 3 obtuse angles. -/
theorem max_obtuse_angles_in_quadrilateral_with_120_degree_angle :
  ∀ (a b c d : ℝ),
  a = 120 →
  a + b + c + d = 360 →
  a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90 →
  False :=
by
  sorry

end max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2015_201501


namespace special_geometric_sequence_ratio_l2015_201508

/-- An increasing geometric sequence with specific conditions -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ q > 1, ∀ n, a (n + 1) = q * a n
  sum_condition : a 1 + a 4 = 9
  product_condition : a 2 * a 3 = 8

/-- The common ratio of the special geometric sequence is 2 -/
theorem special_geometric_sequence_ratio (seq : SpecialGeometricSequence) :
  ∃ q > 1, (∀ n, seq.a (n + 1) = q * seq.a n) ∧ q = 2 := by sorry

end special_geometric_sequence_ratio_l2015_201508


namespace children_who_got_on_bus_stop_l2015_201560

/-- The number of children who got on the bus at a stop -/
def ChildrenWhoGotOn (initial final : ℕ) : ℕ := final - initial

theorem children_who_got_on_bus_stop (initial final : ℕ) 
  (h1 : initial = 52) 
  (h2 : final = 76) : 
  ChildrenWhoGotOn initial final = 24 := by
  sorry

end children_who_got_on_bus_stop_l2015_201560


namespace max_reflections_l2015_201535

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 12

/-- The maximum angle of incidence before reflection becomes impossible -/
def max_angle : ℝ := 90

/-- The number of reflections -/
def n : ℕ := 7

/-- Theorem stating that 7 is the maximum number of reflections possible -/
theorem max_reflections :
  (n : ℝ) * angle_between_lines ≤ max_angle ∧
  ((n + 1) : ℝ) * angle_between_lines > max_angle :=
sorry

end max_reflections_l2015_201535


namespace complex_exponential_thirteen_pi_over_two_equals_i_l2015_201524

theorem complex_exponential_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exponential_thirteen_pi_over_two_equals_i_l2015_201524


namespace natural_number_divisibility_l2015_201569

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ (k : ℕ), k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end natural_number_divisibility_l2015_201569


namespace complex_fraction_simplification_l2015_201559

theorem complex_fraction_simplification :
  let z : ℂ := (3 - I) / (1 - I)
  z = 2 + I :=
by
  sorry

end complex_fraction_simplification_l2015_201559


namespace shoe_color_probability_l2015_201505

theorem shoe_color_probability (n : ℕ) (k : ℕ) (p : ℕ) :
  n = 2 * p →
  k = 3 →
  p = 6 →
  (1 - (n * (n - 2) * (n - 4) / (k * (k - 1) * (k - 2))) / (n.choose k)) = 7 / 11 := by
  sorry

end shoe_color_probability_l2015_201505


namespace product_of_roots_l2015_201500

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 24 → ∃ y : ℝ, (x + 2) * (x - 3) = 24 ∧ (y + 2) * (y - 3) = 24 ∧ x * y = -30 := by
  sorry

end product_of_roots_l2015_201500


namespace bridge_length_l2015_201526

/-- The length of a bridge given specific train parameters -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 :=
by sorry

end bridge_length_l2015_201526


namespace negation_of_proposition_l2015_201541

theorem negation_of_proposition (p : Prop) :
  (∃ n : ℕ, n^2 > 2*n - 1) → (¬p ↔ ∀ n : ℕ, n^2 ≤ 2*n - 1) :=
by sorry

end negation_of_proposition_l2015_201541


namespace inequality_proof_l2015_201547

theorem inequality_proof (x y : ℝ) (m n : ℤ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (1 - x^n.toNat)^m.toNat + (1 - y^m.toNat)^n.toNat ≥ 1 := by
  sorry

end inequality_proof_l2015_201547


namespace malfunctioning_mix_sum_l2015_201513

/-- Represents the fractional composition of Papaya Splash ingredients -/
structure PapayaSplash :=
  (soda_water : ℚ)
  (lemon_juice : ℚ)
  (sugar : ℚ)
  (papaya_puree : ℚ)
  (secret_spice : ℚ)
  (lime_extract : ℚ)

/-- The standard formula for Papaya Splash -/
def standard_formula : PapayaSplash :=
  { soda_water := 8/21,
    lemon_juice := 4/21,
    sugar := 3/21,
    papaya_puree := 3/21,
    secret_spice := 2/21,
    lime_extract := 1/21 }

/-- The malfunctioning machine's mixing ratios -/
def malfunction_ratios : PapayaSplash :=
  { soda_water := 1/2,
    lemon_juice := 3,
    sugar := 2,
    papaya_puree := 1,
    secret_spice := 1/5,
    lime_extract := 1 }

/-- Applies the malfunction ratios to the standard formula -/
def apply_malfunction (formula : PapayaSplash) (ratios : PapayaSplash) : PapayaSplash :=
  { soda_water := formula.soda_water * ratios.soda_water,
    lemon_juice := formula.lemon_juice * ratios.lemon_juice,
    sugar := formula.sugar * ratios.sugar,
    papaya_puree := formula.papaya_puree * ratios.papaya_puree,
    secret_spice := formula.secret_spice * ratios.secret_spice,
    lime_extract := formula.lime_extract * ratios.lime_extract }

/-- Calculates the sum of soda water, sugar, and secret spice blend fractions -/
def sum_selected_ingredients (mix : PapayaSplash) : ℚ :=
  mix.soda_water + mix.sugar + mix.secret_spice

/-- Theorem stating that the sum of selected ingredients in the malfunctioning mix is 52/105 -/
theorem malfunctioning_mix_sum :
  sum_selected_ingredients (apply_malfunction standard_formula malfunction_ratios) = 52/105 :=
sorry

end malfunctioning_mix_sum_l2015_201513


namespace y_intercept_of_parallel_line_l2015_201599

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, -6) } →
  b.point = (1, 2) →
  yIntercept b = -1 := by
  sorry

end y_intercept_of_parallel_line_l2015_201599


namespace max_min_sum_of_2x_minus_3y_l2015_201510

theorem max_min_sum_of_2x_minus_3y : 
  ∀ x y : ℝ, 3 ≤ x → x ≤ 5 → 4 ≤ y → y ≤ 6 → 
  (∃ (max min : ℝ), 
    (∀ z : ℝ, z = 2*x - 3*y → z ≤ max) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = max) ∧
    (∀ z : ℝ, z = 2*x - 3*y → min ≤ z) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = min) ∧
    max + min = -14) :=
by sorry

end max_min_sum_of_2x_minus_3y_l2015_201510


namespace pythagorean_triple_identity_l2015_201581

theorem pythagorean_triple_identity (n : ℕ+) :
  (2 * n + 1) ^ 2 + (2 * n ^ 2 + 2 * n) ^ 2 = (2 * n ^ 2 + 2 * n + 1) ^ 2 := by
  sorry

end pythagorean_triple_identity_l2015_201581


namespace survival_rate_definition_correct_l2015_201573

/-- Survival rate is defined as the percentage of living seedlings out of the total seedlings -/
def survival_rate (living_seedlings total_seedlings : ℕ) : ℚ :=
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100

/-- The given definition of survival rate is correct -/
theorem survival_rate_definition_correct :
  ∀ (living_seedlings total_seedlings : ℕ),
  survival_rate living_seedlings total_seedlings =
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100 :=
by
  sorry

end survival_rate_definition_correct_l2015_201573


namespace stratified_sampling_is_appropriate_l2015_201551

/-- Represents a stratum in a population --/
structure Stratum where
  size : ℕ
  characteristic : ℝ

/-- Represents a population with three strata --/
structure Population where
  strata : Fin 3 → Stratum
  total_size : ℕ
  avg_characteristic : ℝ

/-- Represents a sampling method --/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Determines if a sampling method is appropriate for a given population and sample size --/
def is_appropriate_sampling_method (pop : Population) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.StratifiedSampling => true
  | _ => false

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario --/
theorem stratified_sampling_is_appropriate (pop : Population) (sample_size : ℕ) :
  is_appropriate_sampling_method pop sample_size SamplingMethod.StratifiedSampling :=
sorry

end stratified_sampling_is_appropriate_l2015_201551


namespace geometric_sequence_special_case_l2015_201565

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 4th term of the sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 6th term of the sequence -/
def a_6 (a : ℕ → ℝ) : ℝ := a 6

/-- The 8th term of the sequence -/
def a_8 (a : ℕ → ℝ) : ℝ := a 8

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a_4 a)^2 - 3*(a_4 a) + 2 = 0 →
  (a_8 a)^2 - 3*(a_8 a) + 2 = 0 →
  a_6 a = Real.sqrt 2 := by
  sorry

end geometric_sequence_special_case_l2015_201565


namespace widget_earnings_proof_l2015_201586

/-- Calculates the earnings per widget given the hourly rate, weekly hours, target earnings, and required widget production. -/
def earnings_per_widget (hourly_rate : ℚ) (weekly_hours : ℕ) (target_earnings : ℚ) (widget_production : ℕ) : ℚ :=
  (target_earnings - hourly_rate * weekly_hours) / widget_production

/-- Proves that the earnings per widget is $0.16 given the specified conditions. -/
theorem widget_earnings_proof :
  let hourly_rate : ℚ := 25 / 2
  let weekly_hours : ℕ := 40
  let target_earnings : ℚ := 620
  let widget_production : ℕ := 750
  earnings_per_widget hourly_rate weekly_hours target_earnings widget_production = 16 / 100 := by
  sorry

#eval earnings_per_widget (25/2) 40 620 750

end widget_earnings_proof_l2015_201586


namespace newspaper_conference_max_neither_l2015_201571

theorem newspaper_conference_max_neither (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) (N : ℕ) :
  total = 90 →
  writers = 45 →
  editors ≥ 39 →
  writers + editors - x + N = total →
  N = 2 * x →
  N ≤ 12 :=
sorry

end newspaper_conference_max_neither_l2015_201571


namespace number_divisible_by_56_l2015_201534

/-- The number formed by concatenating digits a, 7, 8, 3, and b -/
def number (a b : ℕ) : ℕ := a * 10000 + 7000 + 800 + 30 + b

/-- Theorem stating that 47832 is divisible by 56 -/
theorem number_divisible_by_56 : 
  number 4 2 % 56 = 0 :=
sorry

end number_divisible_by_56_l2015_201534


namespace biology_enrollment_percentage_l2015_201550

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 840 →
  not_enrolled = 546 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end biology_enrollment_percentage_l2015_201550


namespace base_conversion_156_to_234_l2015_201585

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_conversion_156_to_234 :
  156 = base8ToBase10 [4, 3, 2] :=
by sorry

end base_conversion_156_to_234_l2015_201585


namespace average_rate_of_change_cubic_l2015_201562

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Theorem statement
theorem average_rate_of_change_cubic (a b : ℝ) (h : a = 1 ∧ b = 2) :
  (f b - f a) / (b - a) = 7 := by
  sorry

end average_rate_of_change_cubic_l2015_201562


namespace negation_of_universal_inequality_l2015_201523

theorem negation_of_universal_inequality : 
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 0) := by sorry

end negation_of_universal_inequality_l2015_201523


namespace candy_distribution_l2015_201545

/-- The number of candy pieces in the bag -/
def total_candy : ℕ := 120

/-- Predicate to check if a number is a valid count of students -/
def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (total_candy - 1) % n = 0

/-- The theorem stating the possible number of students -/
theorem candy_distribution :
  ∃ (n : ℕ), is_valid_student_count n ∧ (n = 7 ∨ n = 17) :=
sorry

end candy_distribution_l2015_201545


namespace number_of_rolls_not_random_variable_l2015_201553

/-- A type representing the outcome of a single die roll -/
inductive DieRoll
| one | two | three | four | five | six

/-- A type representing a random variable on die rolls -/
def RandomVariable := DieRoll → ℝ

/-- The number of times the die is rolled -/
def numberOfRolls : ℕ := 2

theorem number_of_rolls_not_random_variable :
  ¬ ∃ (f : RandomVariable), ∀ (r₁ r₂ : DieRoll), f r₁ = numberOfRolls ∧ f r₂ = numberOfRolls :=
sorry

end number_of_rolls_not_random_variable_l2015_201553


namespace first_group_factory_count_l2015_201537

theorem first_group_factory_count (total : ℕ) (second_group : ℕ) (remaining : ℕ) 
  (h1 : total = 169) 
  (h2 : second_group = 52) 
  (h3 : remaining = 48) : 
  total - second_group - remaining = 69 := by
  sorry

end first_group_factory_count_l2015_201537


namespace critical_point_of_cubic_l2015_201568

/-- The function f(x) = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem critical_point_of_cubic (x : ℝ) : 
  (f' x = 0 ↔ x = 0) :=
sorry

#check critical_point_of_cubic

end critical_point_of_cubic_l2015_201568


namespace place_value_difference_l2015_201555

def number : ℝ := 135.21

def hundreds_place_value : ℝ := 100
def tenths_place_value : ℝ := 0.1

theorem place_value_difference : 
  hundreds_place_value - tenths_place_value = 99.9 := by
  sorry

end place_value_difference_l2015_201555


namespace quadratic_equation_result_l2015_201592

theorem quadratic_equation_result (a : ℝ) (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := by
  sorry

end quadratic_equation_result_l2015_201592


namespace square_root_of_product_l2015_201554

theorem square_root_of_product : Real.sqrt (64 * Real.sqrt 49) = 8 * Real.sqrt 7 := by
  sorry

end square_root_of_product_l2015_201554


namespace vectors_not_coplanar_l2015_201552

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} form a basis for a space V, prove that {a + b, a - b, c} are not coplanar -/
theorem vectors_not_coplanar (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  ¬ (∃ (x y z : ℝ), x • (a + b) + y • (a - b) + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end vectors_not_coplanar_l2015_201552


namespace abs_S_value_l2015_201595

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The complex number S -/
def S : ℂ := (1 + 2*i)^12 - (1 - i)^12

/-- Theorem stating the absolute value of S -/
theorem abs_S_value : Complex.abs S = 15689 := by sorry

end abs_S_value_l2015_201595


namespace correct_answers_for_given_score_l2015_201567

/-- Represents the scoring system for a test -/
structure TestScore where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculate_score (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

/-- Theorem stating the number of correct answers given the conditions -/
theorem correct_answers_for_given_score
  (test : TestScore)
  (h1 : test.total_questions = 100)
  (h2 : test.correct_answers + test.incorrect_answers = test.total_questions)
  (h3 : test.score = calculate_score test.correct_answers test.incorrect_answers)
  (h4 : test.score = 76) :
  test.correct_answers = 92 := by
  sorry

end correct_answers_for_given_score_l2015_201567


namespace car_endpoint_locus_l2015_201515

-- Define the car's properties
structure Car where
  r₁ : ℝ
  r₂ : ℝ
  start : ℝ × ℝ
  s : ℝ
  α : ℝ

-- Define the circles W₁ and W₂
def W₁ (car : Car) (α₂ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₂)^2}

def W₂ (car : Car) (α₁ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₁)^2}

-- State the theorem
theorem car_endpoint_locus (car : Car) (α₁ α₂ : ℝ) 
  (h₁ : car.r₁ > car.r₂)
  (h₂ : car.α < 2 * Real.pi)
  (h₃ : α₁ + α₂ = car.α)
  (h₄ : car.r₁ * α₁ + car.r₂ * α₂ = car.s) :
  ∃ (endpoints : Set (ℝ × ℝ)), endpoints = W₁ car α₂ ∩ W₂ car α₁ := by
  sorry

end car_endpoint_locus_l2015_201515


namespace rachels_reading_homework_l2015_201533

/-- Given that Rachel had 9 pages of math homework and 7 more pages of math homework than reading homework, prove that she had 2 pages of reading homework. -/
theorem rachels_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 9)
  (h2 : math_homework = reading_homework + 7) :
  reading_homework = 2 := by
  sorry

end rachels_reading_homework_l2015_201533


namespace sin_pi_over_six_l2015_201502

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_pi_over_six_l2015_201502


namespace sum_of_fractions_l2015_201517

theorem sum_of_fractions : (3 / 10 : ℚ) + (29 / 5 : ℚ) = 61 / 10 := by sorry

end sum_of_fractions_l2015_201517


namespace first_15_prime_sums_l2015_201539

/-- Returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Returns the sum of the first n prime numbers -/
def sumOfFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The number of prime sums among the first 15 sums of consecutive primes -/
def numPrimeSums : ℕ := sorry

theorem first_15_prime_sums : 
  numPrimeSums = 6 := by sorry

end first_15_prime_sums_l2015_201539


namespace amount_for_c_l2015_201597

def total_amount : ℕ := 2000
def ratio_b : ℕ := 4
def ratio_c : ℕ := 16

theorem amount_for_c (total : ℕ) (rb : ℕ) (rc : ℕ) (h1 : total = total_amount) (h2 : rb = ratio_b) (h3 : rc = ratio_c) :
  (rc * total) / (rb + rc) = 1600 := by
  sorry

end amount_for_c_l2015_201597


namespace fundraiser_goal_is_750_l2015_201504

/-- Represents the fundraiser goal calculation --/
def fundraiser_goal (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) 
  (bronze_donation : ℕ) (silver_donation : ℕ) (gold_donation : ℕ) 
  (final_day_goal : ℕ) : ℕ :=
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation + 
  final_day_goal

/-- Theorem stating that the fundraiser goal is $750 --/
theorem fundraiser_goal_is_750 : 
  fundraiser_goal 10 7 1 25 50 100 50 = 750 := by
  sorry

end fundraiser_goal_is_750_l2015_201504


namespace rhombus_c_coordinate_sum_l2015_201570

/-- A rhombus with vertices A, B, C, and D in 2D space -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The property that A and D are diagonally opposite in the rhombus -/
def diagonallyOpposite (r : Rhombus) : Prop :=
  (r.A.1 + r.D.1) / 2 = (r.B.1 + r.C.1) / 2 ∧
  (r.A.2 + r.D.2) / 2 = (r.B.2 + r.C.2) / 2

/-- The theorem stating that for a rhombus ABCD with given coordinates,
    the sum of coordinates of C is 9 -/
theorem rhombus_c_coordinate_sum :
  ∀ (r : Rhombus),
  r.A = (-3, -2) →
  r.B = (1, -5) →
  r.D = (9, 1) →
  diagonallyOpposite r →
  r.C.1 + r.C.2 = 9 := by
  sorry

end rhombus_c_coordinate_sum_l2015_201570


namespace particle_position_after_2023_minutes_l2015_201529

def particle_position (t : ℕ) : ℕ × ℕ :=
  let n := (Nat.sqrt (t / 4) : ℕ)
  let remaining_time := t - 4 * n^2
  let side_length := 2 * n + 1
  if remaining_time ≤ side_length then
    (remaining_time, 0)
  else if remaining_time ≤ 2 * side_length then
    (side_length, remaining_time - side_length)
  else if remaining_time ≤ 3 * side_length then
    (3 * side_length - remaining_time, side_length)
  else
    (0, 4 * side_length - remaining_time)

theorem particle_position_after_2023_minutes :
  particle_position 2023 = (87, 0) := by
  sorry

end particle_position_after_2023_minutes_l2015_201529


namespace total_stars_l2015_201509

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 124) 
  (h2 : stars_per_student = 3) : 
  num_students * stars_per_student = 372 := by
sorry

end total_stars_l2015_201509


namespace exists_valid_arrangement_l2015_201522

/-- Represents a connection between two circle indices -/
def Connection := Fin 5 × Fin 5

/-- Checks if two circles are connected -/
def is_connected (connections : List Connection) (i j : Fin 5) : Prop :=
  (i, j) ∈ connections ∨ (j, i) ∈ connections

/-- The theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (numbers : Fin 5 → ℕ+) (connections : List Connection),
  (∀ i j : Fin 5, is_connected connections i j →
    (numbers i).val / (numbers j).val = 3 ∨ (numbers i).val / (numbers j).val = 9) ∧
  (∀ i j : Fin 5, ¬is_connected connections i j →
    (numbers i).val / (numbers j).val ≠ 3 ∧ (numbers i).val / (numbers j).val ≠ 9) :=
by sorry


end exists_valid_arrangement_l2015_201522


namespace smallest_winning_number_l2015_201536

theorem smallest_winning_number : ∃ N : ℕ, 
  N ≤ 499 ∧ 
  27 * N + 360 < 500 ∧ 
  (∀ k : ℕ, k < N → 27 * k + 360 ≥ 500) ∧
  N = 5 := by
sorry

end smallest_winning_number_l2015_201536


namespace opposite_sides_line_constant_range_l2015_201557

/-- Given two points on opposite sides of a line, prove the range of the constant term -/
theorem opposite_sides_line_constant_range :
  ∀ (a : ℝ),
  (((3 * 2 - 2 * 1 + a) * (3 * (-2) - 2 * 3 + a) < 0) ↔ (-4 < a ∧ a < 12)) := by
  sorry

end opposite_sides_line_constant_range_l2015_201557


namespace softball_team_ratio_l2015_201525

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 18 →
  (men : ℚ) / women = 7 / 11 := by
sorry

end softball_team_ratio_l2015_201525


namespace stratified_sample_third_group_l2015_201574

/-- Given a population with three groups in the ratio 2:5:3 and a stratified sample of size 120,
    the number of items from the third group in the sample is 36. -/
theorem stratified_sample_third_group : 
  ∀ (total_ratio : ℕ) (group1_ratio group2_ratio group3_ratio : ℕ) (sample_size : ℕ),
    total_ratio = group1_ratio + group2_ratio + group3_ratio →
    group1_ratio = 2 →
    group2_ratio = 5 →
    group3_ratio = 3 →
    sample_size = 120 →
    (sample_size * group3_ratio) / total_ratio = 36 := by
  sorry

end stratified_sample_third_group_l2015_201574


namespace arithmetic_heptagon_angle_l2015_201579

/-- Represents a heptagon with angles in arithmetic progression -/
structure ArithmeticHeptagon where
  -- First angle of the progression
  a : ℝ
  -- Common difference of the progression
  d : ℝ
  -- Constraint: Sum of angles in a heptagon is 900°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) + (a + 6*d) = 900

/-- Theorem: In a heptagon with angles in arithmetic progression, one of the angles can be 128.57° -/
theorem arithmetic_heptagon_angle (h : ArithmeticHeptagon) : 
  ∃ k : Fin 7, h.a + k * h.d = 128.57 := by
  sorry

end arithmetic_heptagon_angle_l2015_201579


namespace geometric_arithmetic_progression_sum_l2015_201549

/-- Given a, b, c in geometric progression, a, m, b in arithmetic progression,
    and b, n, c in arithmetic progression, prove that m/a + n/c = 2 -/
theorem geometric_arithmetic_progression_sum (a b c m n : ℝ) 
  (h_geom : b/a = c/b)  -- a, b, c in geometric progression
  (h_arith1 : 2*m = a + b)  -- a, m, b in arithmetic progression
  (h_arith2 : 2*n = b + c)  -- b, n, c in arithmetic progression
  : m/a + n/c = 2 := by
  sorry

end geometric_arithmetic_progression_sum_l2015_201549


namespace circle_area_through_points_l2015_201596

/-- The area of a circle with center P and passing through Q is 149π -/
theorem circle_area_through_points (P Q : ℝ × ℝ) : 
  P = (-2, 3) → Q = (8, -4) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 149 * π := by
  sorry

end circle_area_through_points_l2015_201596


namespace robot_energy_cells_l2015_201576

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The robot's reported energy cell count in base 7 -/
def robotReport : List Nat := [1, 2, 3]

theorem robot_energy_cells :
  base7ToBase10 robotReport = 162 := by
  sorry

end robot_energy_cells_l2015_201576


namespace product_digit_sum_l2015_201520

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  hundreds_digit product + units_digit product = 7 := by
  sorry

end product_digit_sum_l2015_201520


namespace power_equation_solution_l2015_201507

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 := by
  sorry

end power_equation_solution_l2015_201507


namespace share_difference_l2015_201514

/-- Represents the share ratio for each person --/
structure ShareRatio :=
  (faruk : ℕ)
  (vasim : ℕ)
  (ranjith : ℕ)
  (kavita : ℕ)
  (neel : ℕ)

/-- Represents the distribution problem --/
structure DistributionProblem :=
  (ratio : ShareRatio)
  (vasim_share : ℕ)
  (x : ℕ+)
  (y : ℕ+)

def total_ratio (r : ShareRatio) : ℕ :=
  r.faruk + r.vasim + r.ranjith + r.kavita + r.neel

def total_amount (p : DistributionProblem) : ℕ :=
  p.vasim_share * (p.x + p.y)

theorem share_difference (p : DistributionProblem) 
  (h1 : p.ratio = ⟨3, 5, 7, 9, 11⟩)
  (h2 : p.vasim_share = 1500)
  (h3 : total_amount p = total_ratio p.ratio * (p.vasim_share / p.ratio.vasim)) :
  p.ratio.ranjith * (p.vasim_share / p.ratio.vasim) - 
  p.ratio.faruk * (p.vasim_share / p.ratio.vasim) = 1200 := by
  sorry

end share_difference_l2015_201514


namespace sum_of_max_min_l2015_201593

theorem sum_of_max_min (a b c d : ℝ) (ha : a = 0.11) (hb : b = 0.98) (hc : c = 3/4) (hd : d = 2/3) :
  (max a (max b (max c d))) + (min a (min b (min c d))) = 1.09 := by
  sorry

end sum_of_max_min_l2015_201593


namespace ratio_sum_problem_l2015_201543

theorem ratio_sum_problem (a b c : ℝ) 
  (ratio : a / 4 = b / 5 ∧ b / 5 = c / 7)
  (sum : a + b + c = 240) :
  2 * b - a + c = 195 := by
sorry

end ratio_sum_problem_l2015_201543


namespace median_average_ratio_l2015_201503

theorem median_average_ratio (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = 4 * b → c / b = 11 := by
  sorry

end median_average_ratio_l2015_201503


namespace child_b_share_l2015_201564

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 5400 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 1800 := by
sorry

end child_b_share_l2015_201564


namespace training_end_time_l2015_201584

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem training_end_time :
  let startTime : Time := ⟨8, 0, sorry⟩
  let sessionDuration : ℕ := 40
  let breakDuration : ℕ := 15
  let numSessions : ℕ := 4
  let totalDuration := numSessions * sessionDuration + (numSessions - 1) * breakDuration
  let endTime := addMinutes startTime totalDuration
  endTime = ⟨11, 25, sorry⟩ := by sorry

end training_end_time_l2015_201584


namespace stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l2015_201583

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 163

/-- Represents the number of test phones available -/
def num_phones : ℕ := 2

/-- Represents a strategy for dropping phones -/
structure DropStrategy where
  num_drops : ℕ
  can_determine_all_cases : Bool

/-- 
Theorem stating that 18 drops is the minimum number required 
to determine the breaking floor or conclude the phone is unbreakable
-/
theorem min_drops_required : 
  ∀ (s : DropStrategy), s.can_determine_all_cases → s.num_drops ≥ 18 := by
  sorry

/-- 
Theorem stating that 18 drops is sufficient to determine 
the breaking floor or conclude the phone is unbreakable
-/
theorem drops_18_sufficient : 
  ∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18 := by
  sorry

/-- 
Main theorem combining the above results to prove that 18 
is the minimum number of drops required
-/
theorem min_drops_is_18 : 
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ 
    (∀ (t : DropStrategy), t.can_determine_all_cases → s.num_drops ≤ t.num_drops)) ∧
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18) := by
  sorry

end stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l2015_201583


namespace school_girls_count_l2015_201548

theorem school_girls_count (total_students sample_size : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sample_size = 200)
  (h_stratified_sample : ∃ (girls_sampled boys_sampled : ℕ), 
    girls_sampled + boys_sampled = sample_size ∧ 
    girls_sampled + 20 = boys_sampled) :
  ∃ (school_girls : ℕ), 
    school_girls * sample_size = 720 * total_students ∧
    school_girls ≤ total_students :=
by sorry

end school_girls_count_l2015_201548


namespace geometric_sequence_a5_l2015_201511

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + 2 * a 2 = 4 →
  a 4 ^ 2 = 4 * a 3 * a 7 →
  a 5 = 1 / 8 := by
  sorry

end geometric_sequence_a5_l2015_201511


namespace square_equation_solution_l2015_201582

theorem square_equation_solution (x : ℝ) : (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end square_equation_solution_l2015_201582


namespace jared_age_difference_l2015_201530

/-- Given three friends Jared, Hakimi, and Molly, this theorem proves that Jared is 10 years older than Hakimi -/
theorem jared_age_difference (jared_age hakimi_age molly_age : ℕ) : 
  hakimi_age = 40 →
  molly_age = 30 →
  (jared_age + hakimi_age + molly_age) / 3 = 40 →
  jared_age - hakimi_age = 10 := by
sorry

end jared_age_difference_l2015_201530


namespace parallel_line_a_value_l2015_201591

/-- Given two points A and B on a line parallel to 2x - y + 1 = 0, prove a = 2 -/
theorem parallel_line_a_value (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (-1, a) ∧ 
    B = (a, 8) ∧ 
    (∃ (m : ℝ), (B.2 - A.2) = m * (B.1 - A.1) ∧ m = 2)) → 
  a = 2 := by
sorry

end parallel_line_a_value_l2015_201591


namespace simplify_expression_l2015_201527

theorem simplify_expression (x : ℝ) (h : x^8 ≠ 1) :
  4 / (1 + x^4) + 2 / (1 + x^2) + 1 / (1 + x) + 1 / (1 - x) = 8 / (1 - x^8) :=
by sorry

end simplify_expression_l2015_201527


namespace inequality_proof_l2015_201566

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x < 1) (hy1 : y < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end inequality_proof_l2015_201566


namespace same_grade_percentage_l2015_201512

/-- Represents the number of students in the classroom -/
def total_students : ℕ := 25

/-- Represents the number of students who scored 'A' on both exams -/
def students_a : ℕ := 3

/-- Represents the number of students who scored 'B' on both exams -/
def students_b : ℕ := 2

/-- Represents the number of students who scored 'C' on both exams -/
def students_c : ℕ := 1

/-- Represents the number of students who scored 'D' on both exams -/
def students_d : ℕ := 3

/-- Calculates the total number of students who received the same grade on both exams -/
def same_grade_students : ℕ := students_a + students_b + students_c + students_d

/-- Theorem: The percentage of students who received the same grade on both exams is 36% -/
theorem same_grade_percentage :
  (same_grade_students : ℚ) / total_students * 100 = 36 := by
  sorry

end same_grade_percentage_l2015_201512


namespace ef_fraction_of_gh_l2015_201532

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH 
    when GE is 3 times EH and GF is 8 times FH. -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F lie on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end ef_fraction_of_gh_l2015_201532


namespace power_multiplication_l2015_201556

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l2015_201556


namespace main_theorem_l2015_201531

/-- Definition of H function -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ + f x₂) / 2 > f ((x₁ + x₂) / 2)

/-- Definition of even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The main theorem -/
theorem main_theorem (c : ℝ) (f : ℝ → ℝ) (h : f = fun x ↦ x^2 + c*x) 
  (h_even : is_even f) : c = 0 ∧ is_H_function f := by
  sorry

end main_theorem_l2015_201531


namespace sqrt_seven_minus_a_l2015_201544

theorem sqrt_seven_minus_a (a : ℝ) : a = -1 → Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_seven_minus_a_l2015_201544


namespace appropriate_grouping_l2015_201546

theorem appropriate_grouping : 
  (43 + 27) + ((-78) + (-52)) = 43 + (-78) + 27 + (-52) := by
  sorry

end appropriate_grouping_l2015_201546


namespace decimal_equivalent_of_one_fifth_squared_l2015_201558

theorem decimal_equivalent_of_one_fifth_squared : (1 / 5 : ℚ) ^ 2 = 0.04 := by
  sorry

end decimal_equivalent_of_one_fifth_squared_l2015_201558


namespace product_cost_reduction_l2015_201590

theorem product_cost_reduction (original_selling_price : ℝ) 
  (original_profit_rate : ℝ) (new_profit_rate : ℝ) (additional_profit : ℝ) :
  original_selling_price = 659.9999999999994 →
  original_profit_rate = 0.1 →
  new_profit_rate = 0.3 →
  additional_profit = 42 →
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_cost := (original_selling_price + additional_profit) / (1 + new_profit_rate)
  (original_cost - new_cost) / original_cost = 0.1 := by
sorry

end product_cost_reduction_l2015_201590


namespace cyclists_speed_problem_l2015_201589

theorem cyclists_speed_problem (total_distance : ℝ) (speed_difference : ℝ) :
  total_distance = 270 →
  speed_difference = 1.5 →
  ∃ (speed1 speed2 time : ℝ),
    speed1 > 0 ∧
    speed2 > 0 ∧
    speed1 = speed2 + speed_difference ∧
    time = speed1 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed1 = 12 ∧
    speed2 = 10.5 := by
  sorry

end cyclists_speed_problem_l2015_201589


namespace first_set_video_count_l2015_201598

/-- The cost of a video cassette in rupees -/
def video_cost : ℕ := 300

/-- The total cost of the first set in rupees -/
def first_set_cost : ℕ := 1110

/-- The total cost of the second set in rupees -/
def second_set_cost : ℕ := 1350

/-- The number of audio cassettes in the first set -/
def first_set_audio : ℕ := 7

/-- The number of audio cassettes in the second set -/
def second_set_audio : ℕ := 5

/-- The number of video cassettes in the second set -/
def second_set_video : ℕ := 4

/-- Theorem stating that the number of video cassettes in the first set is 3 -/
theorem first_set_video_count : 
  ∃ (audio_cost : ℕ) (first_set_video : ℕ),
    first_set_audio * audio_cost + first_set_video * video_cost = first_set_cost ∧
    second_set_audio * audio_cost + second_set_video * video_cost = second_set_cost ∧
    first_set_video = 3 :=
by sorry

end first_set_video_count_l2015_201598


namespace janet_pages_per_day_l2015_201518

/-- Prove that Janet reads 80 pages a day given the conditions -/
theorem janet_pages_per_day :
  let belinda_pages_per_day : ℕ := 30
  let weeks : ℕ := 6
  let days_per_week : ℕ := 7
  let extra_pages : ℕ := 2100
  ∀ (janet_pages_per_day : ℕ),
    janet_pages_per_day * (weeks * days_per_week) = 
      belinda_pages_per_day * (weeks * days_per_week) + extra_pages →
    janet_pages_per_day = 80 := by
  sorry

end janet_pages_per_day_l2015_201518


namespace urban_road_network_renovation_l2015_201538

theorem urban_road_network_renovation (a m k : ℝ) (ha : a > 0) (hm : m > 0) (hk : k ≥ 3) :
  let x := 0.2 * a
  let P := (m * x) / (m * k * (a * x + 5))
  P ≤ 1 / 20 := by
  sorry

end urban_road_network_renovation_l2015_201538


namespace quadratic_two_distinct_roots_l2015_201521

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end quadratic_two_distinct_roots_l2015_201521


namespace rectangle_longest_side_l2015_201506

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧  -- positive length and width
  2 * (l + w) = 240 ∧  -- perimeter is 240 feet
  l * w = 8 * 240 →  -- area is eight times perimeter
  max l w = 101 :=
by sorry

end rectangle_longest_side_l2015_201506


namespace sum_of_multiples_of_6_and_9_l2015_201516

theorem sum_of_multiples_of_6_and_9 (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end sum_of_multiples_of_6_and_9_l2015_201516


namespace lower_limit_of_b_l2015_201594

theorem lower_limit_of_b (a b : ℤ) (h1 : 8 < a ∧ a < 15) (h2 : b < 21) 
  (h3 : (14 : ℚ) / b - (9 : ℚ) / b = (155 : ℚ) / 100) : 4 ≤ b := by
  sorry

end lower_limit_of_b_l2015_201594


namespace min_sticks_to_remove_8x8_l2015_201561

/-- Represents a chessboard with sticks on edges -/
structure Chessboard :=
  (size : Nat)
  (sticks : Nat)

/-- The minimum number of sticks that must be removed to avoid rectangles -/
def min_sticks_to_remove (board : Chessboard) : Nat :=
  Nat.ceil (2 / 3 * (board.size * board.size))

/-- Theorem stating the minimum number of sticks to remove for an 8x8 chessboard -/
theorem min_sticks_to_remove_8x8 :
  let board : Chessboard := ⟨8, 144⟩
  min_sticks_to_remove board = 43 := by
  sorry

#eval min_sticks_to_remove ⟨8, 144⟩

end min_sticks_to_remove_8x8_l2015_201561


namespace arithmetic_sequence_fifth_term_l2015_201563

theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + 2*y)
  (h2 : seq 1 = x - 2*y)
  (h3 : seq 2 = x^2 * y)
  (h4 : seq 3 = x / (2*y))
  (h_arith : ∀ n, seq (n+1) - seq n = seq 1 - seq 0)
  (hy : y = 1)
  (hx : x = 20) :
  seq 4 = 6 := by
sorry

end arithmetic_sequence_fifth_term_l2015_201563
