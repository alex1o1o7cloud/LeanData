import Mathlib

namespace count_1000_digit_integers_l1457_145753

/-- Represents the count of n-digit numbers ending with 1 or 9 -/
def b (n : ℕ) : ℕ := sorry

/-- Represents the count of n-digit numbers ending with 3 or 7 -/
def c (n : ℕ) : ℕ := sorry

/-- Represents the count of n-digit numbers ending with 5 -/
def d (n : ℕ) : ℕ := sorry

/-- All digits are odd -/
axiom all_digits_odd : ∀ n, b n + c n + d n > 0

/-- Adjacent digits differ by 2 -/
axiom adjacent_digits_differ_by_two :
  ∀ n, b (n + 1) = c n ∧ c (n + 1) = 2 * d n + b n ∧ d (n + 1) = c n

/-- Base cases -/
axiom base_cases : b 1 = 2 ∧ c 1 = 2 ∧ d 1 = 1

/-- The main theorem -/
theorem count_1000_digit_integers :
  b 1000 + c 1000 + d 1000 = 8 * 3^499 :=
sorry

end count_1000_digit_integers_l1457_145753


namespace percent_of_whole_l1457_145736

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) : 
  (part / whole) * 100 = 50 → part = 80 ∧ whole = 160 :=
by sorry

end percent_of_whole_l1457_145736


namespace certain_number_proof_l1457_145771

theorem certain_number_proof (x y a : ℤ) 
  (eq1 : 4 * x + y = a) 
  (eq2 : 2 * x - y = 20) 
  (y_squared : y^2 = 4) : 
  a = 46 := by
  sorry

end certain_number_proof_l1457_145771


namespace rock_ratio_l1457_145786

/-- Represents the rock collecting contest between Sydney and Conner --/
structure RockContest where
  sydney_initial : ℕ
  conner_initial : ℕ
  sydney_day1 : ℕ
  conner_day1_multiplier : ℕ
  conner_day2 : ℕ
  conner_day3 : ℕ

/-- Calculates the number of rocks Sydney collected on day 3 --/
def sydney_day3 (contest : RockContest) : ℕ :=
  contest.sydney_initial + contest.sydney_day1 + 
  (contest.conner_initial + contest.sydney_day1 * contest.conner_day1_multiplier + 
   contest.conner_day2 + contest.conner_day3) - 
  (contest.sydney_initial + contest.sydney_day1)

/-- The main theorem stating the ratio of rocks collected --/
theorem rock_ratio (contest : RockContest) 
  (h1 : contest.sydney_initial = 837)
  (h2 : contest.conner_initial = 723)
  (h3 : contest.sydney_day1 = 4)
  (h4 : contest.conner_day1_multiplier = 8)
  (h5 : contest.conner_day2 = 123)
  (h6 : contest.conner_day3 = 27) :
  sydney_day3 contest = 2 * (contest.sydney_day1 * contest.conner_day1_multiplier) := by
  sorry

end rock_ratio_l1457_145786


namespace spaghetti_dinner_cost_l1457_145755

/-- Calculates the cost per serving of a meal given the costs of ingredients and number of servings -/
def cost_per_serving (pasta_cost sauce_cost meatballs_cost : ℚ) (servings : ℕ) : ℚ :=
  (pasta_cost + sauce_cost + meatballs_cost) / servings

/-- Theorem: Given the specific costs and number of servings, the cost per serving is $1.00 -/
theorem spaghetti_dinner_cost :
  cost_per_serving 1 2 5 8 = 1 := by
  sorry

#eval cost_per_serving 1 2 5 8

end spaghetti_dinner_cost_l1457_145755


namespace smallest_three_digit_multiple_of_17_l1457_145777

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_17_l1457_145777


namespace train_journey_solution_l1457_145796

/-- Represents the number of passengers from Zhejiang and Shanghai at a given point in the journey -/
structure PassengerCount where
  zhejiang : Nat
  shanghai : Nat

/-- Represents the train journey with passenger counts at each stage -/
structure TrainJourney where
  initial : PassengerCount
  afterB : PassengerCount
  afterC : PassengerCount
  afterD : PassengerCount
  afterE : PassengerCount
  final : PassengerCount

def total_passengers (pc : PassengerCount) : Nat :=
  pc.zhejiang + pc.shanghai

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  total_passengers j.initial = 19 ∧
  total_passengers j.afterB = 12 ∧
  total_passengers j.afterD = 7 ∧
  total_passengers j.final = 0 ∧
  j.initial.zhejiang = (total_passengers j.initial - total_passengers j.afterB) ∧
  j.afterB.zhejiang = (total_passengers j.afterB - total_passengers j.afterC) ∧
  j.afterC.zhejiang = (total_passengers j.afterC - total_passengers j.afterD) ∧
  j.afterD.zhejiang = (total_passengers j.afterD - total_passengers j.afterE) ∧
  j.afterE.zhejiang = (total_passengers j.afterE - total_passengers j.final)

/-- The theorem stating that given the conditions, the journey matches the solution -/
theorem train_journey_solution (j : TrainJourney) :
  journey_conditions j →
  j.initial = ⟨7, 12⟩ ∧
  j.afterB = ⟨3, 9⟩ ∧
  j.afterC = ⟨2, 7⟩ ∧
  j.afterD = ⟨2, 5⟩ :=
by sorry

end train_journey_solution_l1457_145796


namespace pizza_order_cost_l1457_145788

/- Define the problem parameters -/
def num_pizzas : Nat := 3
def price_per_pizza : Nat := 10
def num_toppings : Nat := 4
def price_per_topping : Nat := 1
def tip : Nat := 5

/- Define the total cost calculation -/
def total_cost : Nat :=
  num_pizzas * price_per_pizza +
  num_toppings * price_per_topping +
  tip

/- Theorem statement -/
theorem pizza_order_cost : total_cost = 39 := by
  sorry

end pizza_order_cost_l1457_145788


namespace intersection_M_naturals_l1457_145751

def M : Set ℤ := {-1, 0, 1}

theorem intersection_M_naturals :
  M ∩ Set.range (Nat.cast : ℕ → ℤ) = {0, 1} := by
  sorry

end intersection_M_naturals_l1457_145751


namespace area_between_circles_l1457_145772

theorem area_between_circles (R : ℝ) (r : ℝ) (d : ℝ) (chord_length : ℝ) :
  R = 12 →
  d = 2 →
  chord_length = 20 →
  r = Real.sqrt (R^2 - d^2 - (chord_length/2)^2) →
  π * (R^2 - r^2) = 100 * π :=
by sorry

end area_between_circles_l1457_145772


namespace cubic_roots_determinant_l1457_145758

theorem cubic_roots_determinant (p q : ℝ) (a b c : ℝ) : 
  a^3 + p*a + q = 0 → 
  b^3 + p*b + q = 0 → 
  c^3 + p*c + q = 0 → 
  Matrix.det !![1 + a, 1, 1; 1, 1 + b, 1; 1, 1, 1 + c] = p - q := by
sorry

end cubic_roots_determinant_l1457_145758


namespace sum_negative_condition_l1457_145723

theorem sum_negative_condition (x y : ℝ) :
  (∃ (x y : ℝ), (x < 0 ∨ y < 0) ∧ x + y ≥ 0) ∧
  (∀ (x y : ℝ), x + y < 0 → (x < 0 ∨ y < 0)) :=
sorry

end sum_negative_condition_l1457_145723


namespace product_remainder_l1457_145778

def sequence_product : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence_product n * (3 + 10 * (n + 1))

def sequence_length : ℕ := (93 - 3) / 10 + 1

theorem product_remainder (n : ℕ) : 
  n = sequence_length - 1 → sequence_product n ≡ 4 [MOD 7] :=
by sorry

end product_remainder_l1457_145778


namespace power_equation_solution_l1457_145791

theorem power_equation_solution (m n : ℕ) : 
  (1/5 : ℚ)^m * (1/4 : ℚ)^n = 1/(10^4 : ℚ) → m = 4 ∧ n = 2 := by
  sorry

end power_equation_solution_l1457_145791


namespace exponential_sum_l1457_145781

theorem exponential_sum (a x : ℝ) (ha : a > 0) 
  (h : a^(x/2) + a^(-x/2) = 5) : a^x + a^(-x) = 23 := by
  sorry

end exponential_sum_l1457_145781


namespace four_from_seven_l1457_145725

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem four_from_seven :
  choose 7 4 = 35 := by
  sorry

end four_from_seven_l1457_145725


namespace smallest_number_problem_l1457_145727

theorem smallest_number_problem (a b c : ℕ) 
  (h1 : 0 < a ∧ a < b ∧ b < c)
  (h2 : (a + b + c) / 3 = 30)
  (h3 : b = 29)
  (h4 : c = b + 6) :
  a = 26 := by sorry

end smallest_number_problem_l1457_145727


namespace toothpick_burning_time_l1457_145701

/-- Represents a rectangular structure of toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  toothpicks : Nat

/-- Represents the burning process of the toothpick structure -/
def BurningProcess (r : ToothpickRectangle) (burn_time : Nat) : Prop :=
  r.rows = 3 ∧
  r.cols = 5 ∧
  r.toothpicks = 38 ∧
  burn_time = 10 ∧
  ∃ (total_time : Nat), total_time = 65 ∧
    (∀ (t : Nat), t ≤ total_time →
      ∃ (burned : Nat), burned ≤ r.toothpicks ∧
        burned = min r.toothpicks (2 * (t / burn_time + 1)))

/-- Theorem stating that the entire structure burns in 65 seconds -/
theorem toothpick_burning_time (r : ToothpickRectangle) (burn_time : Nat) :
  BurningProcess r burn_time →
  ∃ (total_time : Nat), total_time = 65 ∧
    (∀ (t : Nat), t > total_time →
      ∀ (burned : Nat), burned = r.toothpicks) :=
by
  sorry

end toothpick_burning_time_l1457_145701


namespace min_value_when_a_is_one_range_of_a_for_nonempty_solution_l1457_145718

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_when_a_is_one :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ x, f 1 x ≥ min_val :=
sorry

-- Theorem 2: Range of a when the solution set of f(x) ≤ 3 is non-empty
theorem range_of_a_for_nonempty_solution :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ 3) ↔ (0 ≤ a ∧ a ≤ 6) :=
sorry

end min_value_when_a_is_one_range_of_a_for_nonempty_solution_l1457_145718


namespace exam_mean_score_l1457_145749

theorem exam_mean_score (morning_mean : ℝ) (afternoon_mean : ℝ) (class_ratio : ℚ) : 
  morning_mean = 82 →
  afternoon_mean = 68 →
  class_ratio = 4/5 →
  let total_students := class_ratio + 1
  let total_score := morning_mean * class_ratio + afternoon_mean
  total_score / total_students = 74 := by
sorry

end exam_mean_score_l1457_145749


namespace sigma_multiple_inequality_l1457_145738

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

theorem sigma_multiple_inequality (n : ℕ+) (h : sigma n > 2 * n) :
  ∀ m : ℕ+, (∃ k : ℕ+, m = k * n) → sigma m > 2 * m := by sorry

end sigma_multiple_inequality_l1457_145738


namespace normal_prob_equal_zero_l1457_145780

-- Define a normally distributed random variable
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability density function for a normal distribution
noncomputable def pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Define the probability of a continuous random variable being equal to a specific value
def prob_equal (X : Type) (a : ℝ) : ℝ := 0

-- Theorem statement
theorem normal_prob_equal_zero (μ σ : ℝ) (a : ℝ) :
  prob_equal (normal_dist μ σ) a = 0 :=
sorry

end normal_prob_equal_zero_l1457_145780


namespace xu_shou_achievements_l1457_145793

/-- Represents a historical figure in Chinese science and technology -/
structure HistoricalFigure where
  name : String

/-- Represents a scientific achievement -/
inductive Achievement
  | SteamEngine
  | RiverSteamer
  | ChemicalTranslationPrinciples
  | ElementTranslations

/-- Predicate to check if a historical figure accomplished a given achievement in a specific year -/
def accomplished (person : HistoricalFigure) (achievement : Achievement) (year : ℕ) : Prop :=
  match achievement with
  | Achievement.SteamEngine => person.name = "Xu Shou" ∧ year = 1863
  | Achievement.RiverSteamer => person.name = "Xu Shou"
  | Achievement.ChemicalTranslationPrinciples => person.name = "Xu Shou"
  | Achievement.ElementTranslations => person.name = "Xu Shou" ∧ ∃ n : ℕ, n = 36

/-- Theorem stating that Xu Shou accomplished all the mentioned achievements -/
theorem xu_shou_achievements (xu_shou : HistoricalFigure) 
    (h_name : xu_shou.name = "Xu Shou") :
    accomplished xu_shou Achievement.SteamEngine 1863 ∧
    accomplished xu_shou Achievement.RiverSteamer 0 ∧
    accomplished xu_shou Achievement.ChemicalTranslationPrinciples 0 ∧
    accomplished xu_shou Achievement.ElementTranslations 0 :=
  sorry

end xu_shou_achievements_l1457_145793


namespace mod_equivalence_problem_l1457_145756

theorem mod_equivalence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 28726 ≡ n [ZMOD 17] ∧ n = 4 := by
  sorry

end mod_equivalence_problem_l1457_145756


namespace isosceles_right_triangle_roots_l1457_145775

def isIsoscelesRightTriangle (z₁ z₂ : ℂ) : Prop :=
  z₂ = Complex.exp (Real.pi * Complex.I / 4) * z₁

theorem isosceles_right_triangle_roots (a b : ℂ) (z₁ z₂ : ℂ) :
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  isIsoscelesRightTriangle z₁ z₂ →
  a^2 / b = 4 + 2*Complex.I*Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_roots_l1457_145775


namespace dividend_calculation_l1457_145743

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 161 := by
sorry

end dividend_calculation_l1457_145743


namespace two_integers_problem_l1457_145767

theorem two_integers_problem (x y : ℕ+) 
  (h1 : x * y = 18)
  (h2 : x - y = 4) : 
  x + y = 10 := by
  sorry

end two_integers_problem_l1457_145767


namespace money_ratio_l1457_145760

/-- Represents the money of each person -/
structure Money where
  natasha : ℚ
  carla : ℚ
  cosima : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.natasha = 60 ∧
  m.carla = 2 * m.cosima ∧
  (7/5) * (m.natasha + m.carla + m.cosima) - (m.natasha + m.carla + m.cosima) = 36

/-- The theorem to prove -/
theorem money_ratio (m : Money) : 
  problem_conditions m → m.natasha / m.carla = 3 / 1 := by
  sorry


end money_ratio_l1457_145760


namespace bc_cd_ratio_l1457_145792

-- Define the points on the line
variable (a b c d e : ℝ)

-- Define the conditions
axiom consecutive_points : a < b ∧ b < c ∧ c < d ∧ d < e
axiom de_length : e - d = 8
axiom ab_length : b - a = 5
axiom ac_length : c - a = 11
axiom ae_length : e - a = 22

-- Define the theorem
theorem bc_cd_ratio :
  (c - b) / (d - c) = 2 / 1 :=
sorry

end bc_cd_ratio_l1457_145792


namespace evaluate_polynomial_l1457_145774

theorem evaluate_polynomial : 2001^3 - 1998 * 2001^2 - 1998^2 * 2001 + 1998^3 = 35991 := by
  sorry

end evaluate_polynomial_l1457_145774


namespace proof_method_characteristics_proof_method_relationship_l1457_145714

/-- Represents a proof method -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| KnownToConclusion
| ConclusionToKnown

/-- Defines the characteristics of a proof method -/
structure ProofMethodCharacteristics where
  method : ProofMethod
  direction : ReasoningDirection

/-- Defines the relationship between two proof methods -/
structure ProofMethodRelationship where
  method1 : ProofMethod
  method2 : ProofMethod
  oppositeThoughtProcess : Bool
  inverseProcedures : Bool

/-- Theorem stating the characteristics of synthetic and analytic methods -/
theorem proof_method_characteristics :
  ∃ (synthetic analytic : ProofMethodCharacteristics),
    synthetic.method = ProofMethod.Synthetic ∧
    synthetic.direction = ReasoningDirection.KnownToConclusion ∧
    analytic.method = ProofMethod.Analytic ∧
    analytic.direction = ReasoningDirection.ConclusionToKnown :=
  sorry

/-- Theorem stating the relationship between synthetic and analytic methods -/
theorem proof_method_relationship :
  ∃ (relationship : ProofMethodRelationship),
    relationship.method1 = ProofMethod.Synthetic ∧
    relationship.method2 = ProofMethod.Analytic ∧
    relationship.oppositeThoughtProcess = true ∧
    relationship.inverseProcedures = true :=
  sorry

end proof_method_characteristics_proof_method_relationship_l1457_145714


namespace marcy_votes_l1457_145761

theorem marcy_votes (joey_votes : ℕ) (barry_votes : ℕ) (marcy_votes : ℕ) : 
  joey_votes = 8 →
  barry_votes = 2 * (joey_votes + 3) →
  marcy_votes = 3 * barry_votes →
  marcy_votes = 66 := by
  sorry

end marcy_votes_l1457_145761


namespace sams_adventure_books_l1457_145739

/-- The number of adventure books Sam bought at the school's book fair -/
def adventure_books : ℕ := by sorry

/-- The number of mystery books Sam bought -/
def mystery_books : ℕ := 17

/-- The number of new books Sam bought -/
def new_books : ℕ := 15

/-- The number of used books Sam bought -/
def used_books : ℕ := 15

/-- The total number of books Sam bought -/
def total_books : ℕ := new_books + used_books

theorem sams_adventure_books : adventure_books = 13 := by sorry

end sams_adventure_books_l1457_145739


namespace youtube_views_theorem_l1457_145707

/-- Calculates the total number of views for a YouTube video given initial views,
    increase factor after 4 days, and additional views after 2 more days. -/
def total_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) : ℕ :=
  initial_views + (increase_factor * initial_views) + additional_views

/-- Theorem stating that given the specific conditions from the problem,
    the total number of views is 94000. -/
theorem youtube_views_theorem :
  total_views 4000 10 50000 = 94000 := by
  sorry

end youtube_views_theorem_l1457_145707


namespace work_on_different_days_probability_l1457_145708

/-- The number of members in the group -/
def num_members : ℕ := 3

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The probability that the members work on different days -/
def prob_different_days : ℚ := 30 / 49

theorem work_on_different_days_probability :
  (num_members.factorial * (num_days - num_members).choose num_members) / num_days ^ num_members = prob_different_days := by
  sorry

end work_on_different_days_probability_l1457_145708


namespace semicircle_area_with_inscribed_rectangle_l1457_145776

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  1^2 + (3/2)^2 = r^2 → 
  π * r^2 / 2 = 9 * π / 4 := by
sorry

end semicircle_area_with_inscribed_rectangle_l1457_145776


namespace painting_job_theorem_l1457_145784

/-- Represents the time taken to complete a job given the number of painters -/
def time_to_complete (num_painters : ℕ) : ℚ :=
  12 / num_painters

/-- The problem statement -/
theorem painting_job_theorem :
  let initial_painters : ℕ := 6
  let initial_time : ℚ := 2
  let new_painters : ℕ := 8
  (initial_painters : ℚ) * initial_time = time_to_complete new_painters * new_painters ∧
  time_to_complete new_painters = 3/2 := by
  sorry

end painting_job_theorem_l1457_145784


namespace absolute_value_inequality_l1457_145790

theorem absolute_value_inequality (x y : ℝ) :
  (∀ x y : ℝ, x > y ∧ y > 0 → abs x > abs y) ∧
  (∃ x y : ℝ, abs x > abs y ∧ ¬(x > y ∧ y > 0)) := by
  sorry

end absolute_value_inequality_l1457_145790


namespace sum_of_floors_even_l1457_145794

theorem sum_of_floors_even (a b c : ℕ+) (h : a^2 + b^2 + 1 = c^2) :
  Even (⌊(a : ℝ) / 2⌋ + ⌊(c : ℝ) / 2⌋) := by sorry

end sum_of_floors_even_l1457_145794


namespace parallel_lines_distance_l1457_145750

/-- A circle intersected by four equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the four chords created by the parallel lines -/
  chord_lengths : Fin 4 → ℝ
  /-- The chords have the specified lengths -/
  chord_length_values : chord_lengths = ![42, 36, 36, 30]
  /-- The parallel lines are equally spaced -/
  equally_spaced : ∀ i j : Fin 3, d = d

/-- The theorem stating that the distance between adjacent parallel lines is √2 -/
theorem parallel_lines_distance (c : ParallelLinesCircle) : c.d = Real.sqrt 2 := by
  sorry

end parallel_lines_distance_l1457_145750


namespace both_correct_count_l1457_145763

theorem both_correct_count (total : ℕ) (set_correct : ℕ) (func_correct : ℕ) (both_incorrect : ℕ) :
  total = 50 →
  set_correct = 40 →
  func_correct = 31 →
  both_incorrect = 4 →
  total - both_incorrect = set_correct + func_correct - (set_correct + func_correct - (total - both_incorrect)) :=
by
  sorry

#check both_correct_count

end both_correct_count_l1457_145763


namespace problem_solution_l1457_145737

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 30)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end problem_solution_l1457_145737


namespace divisibility_by_x_squared_plus_x_plus_one_l1457_145757

theorem divisibility_by_x_squared_plus_x_plus_one (n : ℕ) :
  ∃ q : Polynomial ℤ, (X + 1 : Polynomial ℤ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end divisibility_by_x_squared_plus_x_plus_one_l1457_145757


namespace volume_equality_l1457_145720

-- Define the region S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |5 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 10}

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the volume of the solid obtained by revolving S around y = x
noncomputable def volume_of_solid : ℝ := sorry

-- Define the volume calculated using the cone formula
noncomputable def volume_by_cones : ℝ := sorry

-- Theorem statement
theorem volume_equality : volume_of_solid = volume_by_cones := by sorry

end volume_equality_l1457_145720


namespace rationalize_sqrt_three_eighths_l1457_145703

theorem rationalize_sqrt_three_eighths : 
  Real.sqrt (3 / 8) = Real.sqrt 6 / 4 := by
  sorry

end rationalize_sqrt_three_eighths_l1457_145703


namespace price_reduction_achieves_target_l1457_145741

/-- Represents the daily sales and profit scenario for a clothing item -/
structure ClothingSales where
  initialSales : ℕ  -- Initial daily sales
  initialProfit : ℕ  -- Initial profit per piece in yuan
  salesIncrease : ℕ  -- Increase in daily sales per yuan of price reduction
  targetProfit : ℕ  -- Target daily profit in yuan

/-- Calculates the daily profit given a price reduction -/
def dailyProfit (cs : ClothingSales) (priceReduction : ℕ) : ℕ :=
  (cs.initialProfit - priceReduction) * (cs.initialSales + cs.salesIncrease * priceReduction)

/-- Theorem stating that price reductions of 4 or 36 yuan achieve the target profit -/
theorem price_reduction_achieves_target (cs : ClothingSales) 
  (h1 : cs.initialSales = 20)
  (h2 : cs.initialProfit = 44)
  (h3 : cs.salesIncrease = 5)
  (h4 : cs.targetProfit = 1600) :
  (dailyProfit cs 4 = cs.targetProfit) ∧ (dailyProfit cs 36 = cs.targetProfit) :=
by
  sorry

#eval dailyProfit { initialSales := 20, initialProfit := 44, salesIncrease := 5, targetProfit := 1600 } 4
#eval dailyProfit { initialSales := 20, initialProfit := 44, salesIncrease := 5, targetProfit := 1600 } 36

end price_reduction_achieves_target_l1457_145741


namespace expression_value_l1457_145795

theorem expression_value : 
  let x : ℤ := -1
  let y : ℤ := 2
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end expression_value_l1457_145795


namespace table_count_l1457_145765

theorem table_count (num_stools : ℕ → ℕ) (num_tables : ℕ) 
  (h1 : num_stools num_tables = 6 * num_tables)
  (h2 : 3 * num_stools num_tables + 4 * num_tables = 484) : 
  num_tables = 22 := by
  sorry

end table_count_l1457_145765


namespace proposition_and_converse_l1457_145766

theorem proposition_and_converse :
  (∀ a b : ℝ, a + b ≥ 2 → a ≥ 1 ∨ b ≥ 1) ∧
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ a + b < 2) :=
by sorry

end proposition_and_converse_l1457_145766


namespace f_neq_for_prime_sum_l1457_145716

/-- Sum of positive integers not relatively prime to n -/
def f (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => if Nat.gcd k n ≠ 1 then k else 0)

/-- Theorem stating that f(n+p) ≠ f(n) for n ≥ 2 and prime p -/
theorem f_neq_for_prime_sum (n : ℕ) (p : ℕ) (h1 : n ≥ 2) (h2 : Nat.Prime p) :
  f (n + p) ≠ f n :=
by
  sorry

end f_neq_for_prime_sum_l1457_145716


namespace difference_between_point_eight_and_one_eighth_l1457_145734

theorem difference_between_point_eight_and_one_eighth (ε : ℝ) :
  0.8 - (1 / 8 : ℝ) = 0.675 := by
  sorry

end difference_between_point_eight_and_one_eighth_l1457_145734


namespace first_covering_triangular_number_l1457_145732

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def covers_all_columns (n : ℕ) : Prop :=
  ∀ k : Fin 10, ∃ m : ℕ, m ≤ n ∧ triangular_number m % 10 = k

theorem first_covering_triangular_number :
  (covers_all_columns 29) ∧ (∀ k < 29, ¬ covers_all_columns k) :=
sorry

end first_covering_triangular_number_l1457_145732


namespace quadrilateral_exterior_interior_angles_equal_l1457_145740

theorem quadrilateral_exterior_interior_angles_equal :
  ∀ n : ℕ, n ≥ 3 →
  (360 : ℝ) = (n - 2) * 180 ↔ n = 4 :=
by sorry

end quadrilateral_exterior_interior_angles_equal_l1457_145740


namespace three_roots_exist_l1457_145706

/-- The cubic function we're analyzing -/
def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

/-- Theorem stating the existence of three roots in the given interval -/
theorem three_roots_exist (ε : ℝ) (hε : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ ∈ Set.Icc (-1 : ℝ) 3 ∧ 
    x₂ ∈ Set.Icc (-1 : ℝ) 3 ∧ 
    x₃ ∈ Set.Icc (-1 : ℝ) 3 ∧ 
    |f x₁| < ε ∧ 
    |f x₂| < ε ∧ 
    |f x₃| < ε ∧ 
    x₁ < x₂ ∧ x₂ < x₃ := by
  sorry

end three_roots_exist_l1457_145706


namespace derivative_ln_x_over_x_l1457_145728

open Real

theorem derivative_ln_x_over_x (x : ℝ) (h : x > 0) :
  deriv (fun x => (log x) / x) x = (1 - log x) / x^2 := by
  sorry

end derivative_ln_x_over_x_l1457_145728


namespace card_draw_not_algorithm_l1457_145744

/-- Represents an algorithm in our discussion -/
structure Algorithm where
  steps : List String
  rules : List String
  problem_type : String
  computable : Bool

/-- Represents the operation of calculating the possibility of reaching 24 by randomly drawing 4 playing cards -/
def card_draw_operation : Algorithm := sorry

/-- The definition of an algorithm in our discussion -/
def is_valid_algorithm (a : Algorithm) : Prop :=
  a.steps.length > 0 ∧ 
  a.steps.all (λ s => s.length > 0) ∧
  a.rules.length > 0 ∧
  a.problem_type.length > 0 ∧
  a.computable

/-- Theorem stating that the card draw operation is not a valid algorithm -/
theorem card_draw_not_algorithm : ¬(is_valid_algorithm card_draw_operation) := by
  sorry

end card_draw_not_algorithm_l1457_145744


namespace x_gt_1_necessary_not_sufficient_for_log_x_gt_1_l1457_145773

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement of the theorem
theorem x_gt_1_necessary_not_sufficient_for_log_x_gt_1 :
  (∀ x : ℝ, log10 x > 1 → x > 1) ∧
  ¬(∀ x : ℝ, x > 1 → log10 x > 1) :=
sorry

end x_gt_1_necessary_not_sufficient_for_log_x_gt_1_l1457_145773


namespace at_least_two_equal_sums_l1457_145782

-- Define the type for cell values
inductive CellValue
  | one
  | three
  | five
  | seven

-- Define the type for the 5x5 square
def Square := Matrix (Fin 5) (Fin 5) CellValue

-- Define a function to represent a valid sum (odd number between 5 and 35)
def ValidSum := {n : ℕ // n % 2 = 1 ∧ 5 ≤ n ∧ n ≤ 35}

-- Define a function to calculate the sum of a line (row, column, or diagonal)
def lineSum (s : Square) (line : List (Fin 5 × Fin 5)) : ValidSum :=
  sorry

-- Define the set of all lines (rows, columns, and diagonals)
def allLines : Set (List (Fin 5 × Fin 5)) :=
  sorry

-- Theorem statement
theorem at_least_two_equal_sums (s : Square) :
  ∃ (l1 l2 : List (Fin 5 × Fin 5)), l1 ∈ allLines ∧ l2 ∈ allLines ∧ l1 ≠ l2 ∧ lineSum s l1 = lineSum s l2 :=
  sorry

end at_least_two_equal_sums_l1457_145782


namespace minus_eight_representation_l1457_145746

-- Define a type for temperature
structure Temperature where
  value : ℤ
  unit : String

-- Define a function to represent temperature above or below zero
def aboveZero (t : Temperature) : Bool :=
  t.value > 0

-- Define the given condition
axiom plus_ten_above_zero : aboveZero (Temperature.mk 10 "°C") = true

-- State the theorem to be proved
theorem minus_eight_representation :
  Temperature.mk (-8) "°C" = Temperature.mk (-8) "°C" :=
sorry

end minus_eight_representation_l1457_145746


namespace prob_king_or_queen_l1457_145721

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (kings : ℕ)
  (queens : ℕ)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , ranks := 13
  , suits := 4
  , kings := 4
  , queens := 4 }

/-- The probability of drawing a King or Queen from a standard deck -/
theorem prob_king_or_queen (d : Deck) (h : d = standard_deck) :
  (d.kings + d.queens : ℚ) / d.total_cards = 2 / 13 := by
  sorry

end prob_king_or_queen_l1457_145721


namespace divisibility_by_five_l1457_145747

theorem divisibility_by_five (n : ℕ) : 
  (2^(3*n + 5) + 3^(n + 1)) % 5 = 0 := by
  sorry

end divisibility_by_five_l1457_145747


namespace negative_integer_squared_plus_self_equals_twenty_l1457_145702

theorem negative_integer_squared_plus_self_equals_twenty (N : ℤ) : 
  N < 0 → 2 * N^2 + N = 20 → N = -4 := by
  sorry

end negative_integer_squared_plus_self_equals_twenty_l1457_145702


namespace total_cards_proof_l1457_145754

/-- The number of baseball cards Carlos has -/
def carlos_cards : ℕ := 20

/-- The difference in cards between Carlos and Matias -/
def difference : ℕ := 6

/-- The number of baseball cards Matias has -/
def matias_cards : ℕ := carlos_cards - difference

/-- The number of baseball cards Jorge has -/
def jorge_cards : ℕ := matias_cards

/-- The total number of baseball cards -/
def total_cards : ℕ := carlos_cards + matias_cards + jorge_cards

theorem total_cards_proof : total_cards = 48 := by
  sorry

end total_cards_proof_l1457_145754


namespace parallel_vectors_imply_m_value_l1457_145726

def a (m : ℝ) : Fin 2 → ℝ := ![1, m]
def b : Fin 2 → ℝ := ![2, 5]
def c (m : ℝ) : Fin 2 → ℝ := ![m, 3]

theorem parallel_vectors_imply_m_value :
  ∀ m : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ (a m + c m) = k • (a m - b)) →
  (m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2) :=
by sorry

end parallel_vectors_imply_m_value_l1457_145726


namespace truck_journey_l1457_145745

theorem truck_journey (north_distance east_distance total_distance : ℝ) 
  (h1 : north_distance = 40)
  (h2 : total_distance = 50)
  (h3 : total_distance^2 = north_distance^2 + east_distance^2) :
  east_distance = 30 := by
  sorry

end truck_journey_l1457_145745


namespace expected_rolls_in_year_l1457_145713

/-- Represents the possible outcomes of rolling an eight-sided die -/
inductive DieOutcome
  | One
  | Prime
  | Composite
  | Eight

/-- The probability distribution of the die outcomes -/
def dieProbability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One => 1/8
  | DieOutcome.Prime => 1/2
  | DieOutcome.Composite => 1/4
  | DieOutcome.Eight => 1/8

/-- The expected number of rolls per day -/
def expectedRollsPerDay : ℚ := 7/5

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- Theorem: The expected number of die rolls in a non-leap year is 511 -/
theorem expected_rolls_in_year :
  expectedRollsPerDay * daysInNonLeapYear = 511 := by
  sorry

end expected_rolls_in_year_l1457_145713


namespace isosceles_triangle_perimeter_l1457_145762

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 9) :
  let perimeter := 2 * b + a
  perimeter = 22 := by
sorry

end isosceles_triangle_perimeter_l1457_145762


namespace joan_missed_games_l1457_145748

/-- The number of baseball games Joan missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Joan missed 469 baseball games -/
theorem joan_missed_games : games_missed 864 395 = 469 := by
  sorry

end joan_missed_games_l1457_145748


namespace no_special_triangle_exists_l1457_145798

/-- A triangle with sides and angles in arithmetic progression, given area, and circumradius -/
structure SpecialTriangle where
  /-- The common difference of the arithmetic progression of sides -/
  d : ℝ
  /-- The middle term of the arithmetic progression of sides -/
  b : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The sides form an arithmetic progression -/
  sides_progression : d ≥ 0 ∧ b > d
  /-- The angles form an arithmetic progression -/
  angles_progression : ∃ (α β γ : ℝ), α + β + γ = 180 ∧ β = 60 ∧ α < β ∧ β < γ
  /-- The area is 50 cm² -/
  area_constraint : area = 50
  /-- The circumradius is 10 cm -/
  circumradius_constraint : circumradius = 10

/-- Theorem stating that no triangle satisfies all the given conditions -/
theorem no_special_triangle_exists : ¬∃ (t : SpecialTriangle), True := by
  sorry

end no_special_triangle_exists_l1457_145798


namespace only_D_cannot_form_triangle_l1457_145709

/-- A set of three line segments that may or may not form a triangle -/
structure SegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a set of segments can form a triangle using the triangle inequality theorem -/
def can_form_triangle (s : SegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The given sets of line segments -/
def set_A : SegmentSet := ⟨2, 3, 4⟩
def set_B : SegmentSet := ⟨3, 6, 7⟩
def set_C : SegmentSet := ⟨5, 6, 7⟩
def set_D : SegmentSet := ⟨2, 2, 6⟩

/-- Theorem stating that set D is the only set that cannot form a triangle -/
theorem only_D_cannot_form_triangle : 
  can_form_triangle set_A ∧ 
  can_form_triangle set_B ∧ 
  can_form_triangle set_C ∧ 
  ¬can_form_triangle set_D := by
  sorry

end only_D_cannot_form_triangle_l1457_145709


namespace purely_imaginary_x_eq_neg_one_l1457_145719

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given a real number x, define z as (x^2 - 1) + (x - 1)i. -/
def z (x : ℝ) : ℂ :=
  ⟨x^2 - 1, x - 1⟩

theorem purely_imaginary_x_eq_neg_one :
  ∀ x : ℝ, IsPurelyImaginary (z x) → x = -1 :=
by sorry

end purely_imaginary_x_eq_neg_one_l1457_145719


namespace spending_difference_l1457_145710

def supermarket_spending (x : ℝ) : Prop :=
  x > 0 ∧ x < 350

def automobile_repair_cost : ℝ := 350

def total_spent : ℝ := 450

theorem spending_difference (x : ℝ) 
  (h1 : supermarket_spending x) 
  (h2 : x + automobile_repair_cost = total_spent) 
  (h3 : automobile_repair_cost > 3 * x) : 
  automobile_repair_cost - 3 * x = 50 := by
sorry

end spending_difference_l1457_145710


namespace owen_burger_purchases_l1457_145722

/-- The number of burgers Owen purchased each day in June -/
def burgers_per_day (days_in_june : ℕ) (burger_cost : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / burger_cost) / days_in_june

/-- Proof that Owen purchased 2 burgers each day in June -/
theorem owen_burger_purchases :
  burgers_per_day 30 12 720 = 2 := by
  sorry

end owen_burger_purchases_l1457_145722


namespace smarties_remainder_l1457_145700

theorem smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end smarties_remainder_l1457_145700


namespace expand_expression_l1457_145787

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * 3 * x^3 = 33 * x^5 + 15 * x^4 - 9 * x^3 := by
  sorry

end expand_expression_l1457_145787


namespace theater_ticket_sales_l1457_145779

/-- Represents the theater ticket sales problem --/
theorem theater_ticket_sales 
  (orchestra_price : ℕ) 
  (balcony_price : ℕ) 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (h1 : orchestra_price = 12)
  (h2 : balcony_price = 8)
  (h3 : total_tickets = 340)
  (h4 : total_revenue = 3320) :
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue ∧
    balcony_tickets - orchestra_tickets = 40 := by
  sorry

end theater_ticket_sales_l1457_145779


namespace total_rats_l1457_145759

/-- The number of rats each person has -/
structure RatCounts where
  hunter : ℕ
  elodie : ℕ
  kenia : ℕ
  teagan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (rc : RatCounts) : Prop :=
  rc.elodie = 30 ∧
  rc.elodie = rc.hunter + 10 ∧
  rc.kenia = 3 * (rc.hunter + rc.elodie) ∧
  ∃ p : ℚ, rc.teagan = rc.elodie + (p / 100) * rc.elodie ∧
           rc.teagan = rc.kenia - 5

/-- The theorem to be proved -/
theorem total_rats (rc : RatCounts) :
  satisfiesConditions rc → rc.hunter + rc.elodie + rc.kenia + rc.teagan = 345 :=
by
  sorry

end total_rats_l1457_145759


namespace line_not_in_third_quadrant_l1457_145785

/-- Given a line Ax + By + C = 0 where AC < 0 and BC < 0, the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (A B C : ℝ) (hAC : A * C < 0) (hBC : B * C < 0) :
  ∃ (x y : ℝ), A * x + B * y + C = 0 ∧ (x ≤ 0 ∧ y ≤ 0 → False) :=
sorry

end line_not_in_third_quadrant_l1457_145785


namespace selection_theorem_l1457_145731

/-- The probability of a student being selected for a visiting group -/
def selection_probability (total : ℕ) (eliminated : ℕ) (group_size : ℕ) : ℚ :=
  group_size / (total - eliminated)

/-- The properties of the selection process -/
theorem selection_theorem (total : ℕ) (eliminated : ℕ) (group_size : ℕ) 
  (h1 : total = 2004) 
  (h2 : eliminated = 4) 
  (h3 : group_size = 50) :
  selection_probability total eliminated group_size = 1 / 40 := by
  sorry

#eval selection_probability 2004 4 50

end selection_theorem_l1457_145731


namespace ellipse_theorem_l1457_145712

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b = 2 ∧ ellipse a b 1 (Real.sqrt 3 / 2)

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) (h : ellipse_conditions a b) :
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ C : ℝ × ℝ, C.1^2 / 4 + C.2^2 = 1 ∧
    ∃ A B : ℝ × ℝ, A.1^2 / 4 + A.2^2 = 1 ∧ B.1^2 / 4 + B.2^2 = 1 ∧
      (A.1 * B.1 + A.2 * B.2 = 0) ∧
      (C.1 = A.1 + B.1) ∧ (C.2 = A.2 + B.2) ∧
      (abs (A.1 * B.2 - A.2 * B.1) = Real.sqrt 3)) :=
by
  sorry


end ellipse_theorem_l1457_145712


namespace house_coloring_l1457_145733

theorem house_coloring (n : ℕ) (h : n ≥ 2) :
  ∃ (f : Fin n → Fin n) (c : Fin n → Fin 3),
    (∀ i : Fin n, f i ≠ i) ∧
    (∀ i j : Fin n, i ≠ j → f i ≠ f j) ∧
    (∀ i : Fin n, c i ≠ c (f i)) :=
by sorry

end house_coloring_l1457_145733


namespace train_crossing_time_l1457_145711

/-- Proves that a train crossing a platform of equal length in 40 seconds will cross a signal pole in 20 seconds -/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (platform_crossing_time : ℝ) (h1 : train_length = 250) 
  (h2 : platform_length = 250) (h3 : platform_crossing_time = 40) : 
  train_length / ((train_length + platform_length) / platform_crossing_time) = 20 := by
  sorry

end train_crossing_time_l1457_145711


namespace semicircular_window_perimeter_l1457_145742

/-- The perimeter of a semicircular window with diameter d is (π * d) / 2 + d -/
theorem semicircular_window_perimeter (d : ℝ) (h : d = 63) :
  (π * d) / 2 + d = (π * 63) / 2 + 63 :=
by sorry

end semicircular_window_perimeter_l1457_145742


namespace flavoring_corn_ratio_comparison_l1457_145768

/-- Standard formulation ratio of flavoring to corn syrup to water -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- Sport formulation contains 2 ounces of corn syrup -/
def sport_corn_syrup : ℚ := 2

/-- Sport formulation contains 30 ounces of water -/
def sport_water : ℚ := 30

/-- Sport formulation ratio of flavoring to water is half that of standard formulation -/
def sport_flavoring_water_ratio : ℚ := (standard_ratio 0) / (standard_ratio 2) / 2

/-- Calculate the amount of flavoring in sport formulation -/
def sport_flavoring : ℚ := sport_water * sport_flavoring_water_ratio

/-- Ratio of flavoring to corn syrup in sport formulation -/
def sport_flavoring_corn_ratio : ℚ := sport_flavoring / sport_corn_syrup

/-- Ratio of flavoring to corn syrup in standard formulation -/
def standard_flavoring_corn_ratio : ℚ := (standard_ratio 0) / (standard_ratio 1)

/-- Main theorem: The ratio of (flavoring to corn syrup in sport formulation) to 
    (flavoring to corn syrup in standard formulation) is 3 -/
theorem flavoring_corn_ratio_comparison : 
  sport_flavoring_corn_ratio / standard_flavoring_corn_ratio = 3 := by
  sorry

end flavoring_corn_ratio_comparison_l1457_145768


namespace circle_radius_half_l1457_145764

theorem circle_radius_half (x y : ℝ) : 
  (π * x^2 = π * y^2) →  -- Circles x and y have the same area
  (2 * π * x = 14 * π) →  -- Circle x has a circumference of 14π
  y / 2 = 3.5 :=  -- Half of the radius of circle y is 3.5
by
  sorry

end circle_radius_half_l1457_145764


namespace final_balance_calculation_l1457_145717

def calculate_final_balance (initial_investment : ℝ) (interest_rates : List ℝ) 
  (deposits : List (Nat × ℝ)) (withdrawals : List (Nat × ℝ)) : ℝ :=
  sorry

theorem final_balance_calculation :
  let initial_investment : ℝ := 10000
  let interest_rates : List ℝ := [0.02, 0.03, 0.04, 0.025, 0.035, 0.04, 0.03, 0.035, 0.04]
  let deposits : List (Nat × ℝ) := [(3, 1000), (6, 1000)]
  let withdrawals : List (Nat × ℝ) := [(9, 2000)]
  calculate_final_balance initial_investment interest_rates deposits withdrawals = 13696.95 := by
  sorry

end final_balance_calculation_l1457_145717


namespace special_sequence_value_of_2_special_sequence_verification_l1457_145724

/-- A sequence where each term n is mapped to 6n, except for 6 which maps to 1 -/
def special_sequence : ℕ → ℕ
| 6 => 1
| n => 6 * n

/-- The theorem states that the value corresponding to 2 in the special sequence is 12 -/
theorem special_sequence_value_of_2 : special_sequence 2 = 12 := by
  sorry

/-- Verification of other given values in the sequence -/
theorem special_sequence_verification :
  special_sequence 1 = 6 ∧
  special_sequence 3 = 18 ∧
  special_sequence 4 = 24 ∧
  special_sequence 5 = 30 ∧
  special_sequence 6 = 1 := by
  sorry

end special_sequence_value_of_2_special_sequence_verification_l1457_145724


namespace arcsin_sin_eq_half_x_solutions_l1457_145729

theorem arcsin_sin_eq_half_x_solutions :
  {x : ℝ | x ∈ Set.Icc (-Real.pi) Real.pi ∧ Real.arcsin (Real.sin x) = x / 2} =
  {-2 * Real.pi / 3, 0, 2 * Real.pi / 3} := by
sorry

end arcsin_sin_eq_half_x_solutions_l1457_145729


namespace min_sum_squares_l1457_145770

theorem min_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 10) :
  x^2 + y^2 + z^2 ≥ 100/29 := by
  sorry

end min_sum_squares_l1457_145770


namespace find_other_number_l1457_145769

theorem find_other_number (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 5040)
  (h_gcd : Nat.gcd x y = 24)
  (h_x : x = 240) :
  y = 504 := by
  sorry

end find_other_number_l1457_145769


namespace contest_end_time_l1457_145797

def contest_start : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight
def contest_duration : Nat := 850   -- total duration in minutes
def break_duration : Nat := 30      -- break duration in minutes

def minutes_in_day : Nat := 24 * 60 -- number of minutes in a day

def contest_end : Nat :=
  (contest_start + contest_duration - break_duration) % minutes_in_day

theorem contest_end_time :
  contest_end = 4 * 60 + 40 := by sorry

end contest_end_time_l1457_145797


namespace same_gate_probability_proof_l1457_145789

/-- The number of ticket gates available -/
def num_gates : ℕ := 3

/-- The probability of two individuals selecting the same ticket gate -/
def same_gate_probability : ℚ := 1 / 3

/-- Theorem stating that the probability of two individuals selecting the same ticket gate
    out of three available gates is 1/3 -/
theorem same_gate_probability_proof :
  same_gate_probability = 1 / num_gates := by sorry

end same_gate_probability_proof_l1457_145789


namespace prism_21_edges_9_faces_l1457_145704

/-- A prism is a polyhedron with two congruent parallel faces (bases) and other faces (lateral faces) that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let n := p.edges / 3  -- number of sides in each base
  n + 2  -- lateral faces + 2 base faces

/-- Theorem: A prism with 21 edges has 9 faces -/
theorem prism_21_edges_9_faces :
  ∀ (p : Prism), p.edges = 21 → num_faces p = 9 := by
  sorry

#eval num_faces { edges := 21 }

end prism_21_edges_9_faces_l1457_145704


namespace ruble_payment_l1457_145735

theorem ruble_payment (x : ℤ) (h : x > 7) : ∃ (a b : ℕ), x = 3 * a + 5 * b := by
  sorry

end ruble_payment_l1457_145735


namespace sum_of_four_digit_numbers_l1457_145799

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end sum_of_four_digit_numbers_l1457_145799


namespace max_M_value_inequality_proof_l1457_145783

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x - 3|

-- Theorem for part 1
theorem max_M_value : 
  (∃ M : ℝ, (∀ x m : ℝ, f x ≥ |m + 1| → m ≤ M) ∧ 
   (∀ ε > 0, ∃ x m : ℝ, f x < |m + 1| ∧ m > M - ε)) → 
  (∃ M : ℝ, M = 3/2 ∧ 
   (∀ x m : ℝ, f x ≥ |m + 1| → m ≤ M) ∧
   (∀ ε > 0, ∃ x m : ℝ, f x < |m + 1| ∧ m > M - ε)) :=
sorry

-- Theorem for part 2
theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 3/2) : 
  b^2/a + c^2/b + a^2/c ≥ 3/2 :=
sorry

end max_M_value_inequality_proof_l1457_145783


namespace stratified_sampling_geometric_sequence_l1457_145730

theorem stratified_sampling_geometric_sequence (total : ℕ) (ratio : ℕ) : 
  total = 140 → ratio = 2 → ∃ (x : ℕ), x + ratio * x + ratio^2 * x = total ∧ ratio * x = 40 := by
  sorry

end stratified_sampling_geometric_sequence_l1457_145730


namespace fifth_power_sum_l1457_145705

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 3004 / 11 := by
sorry

end fifth_power_sum_l1457_145705


namespace sufficient_not_necessary_l1457_145715

theorem sufficient_not_necessary (a : ℝ) :
  (a > 10 → (1 / a < 1 / 10)) ∧ ¬((1 / a < 1 / 10) → a > 10) :=
sorry

end sufficient_not_necessary_l1457_145715


namespace tailor_cut_difference_l1457_145752

theorem tailor_cut_difference (skirt_cut pants_cut : ℝ) 
  (h1 : skirt_cut = 0.75)
  (h2 : pants_cut = 0.5) : 
  skirt_cut - pants_cut = 0.25 := by
sorry

end tailor_cut_difference_l1457_145752
