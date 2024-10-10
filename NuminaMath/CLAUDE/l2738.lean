import Mathlib

namespace tangent_slope_point_M_l2738_273849

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

theorem tangent_slope_point_M :
  ∀ x y : ℝ, f y = f x → f' x = -4 → x = -1 ∧ y = 3 := by
  sorry

end tangent_slope_point_M_l2738_273849


namespace inequality_proof_l2738_273856

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end inequality_proof_l2738_273856


namespace sin_cos_equation_integer_solution_l2738_273842

theorem sin_cos_equation_integer_solution (x : ℤ) :
  (∃ t : ℤ, x = 4 * t + 1 ∨ x = 4 * t - 1) ↔ 
  Real.sin (π * (2 * ↑x - 1)) = Real.cos (π * ↑x / 2) :=
sorry

end sin_cos_equation_integer_solution_l2738_273842


namespace sum_of_factors_l2738_273830

theorem sum_of_factors (p q r s t : ℝ) : 
  (∀ y : ℝ, 512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t)) →
  p + q + r + s + t = 60 := by
sorry

end sum_of_factors_l2738_273830


namespace jerrie_situp_minutes_l2738_273857

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- Theorem stating that Jerrie did sit-ups for 3 minutes -/
theorem jerrie_situp_minutes :
  ∃ (j : ℕ), j * jerrie_situps + barney_minutes * barney_situps + carrie_minutes * carrie_situps = total_situps ∧ j = 3 :=
by sorry

end jerrie_situp_minutes_l2738_273857


namespace line_circle_intersection_l2738_273852

/-- The line y = kx + 1 always intersects with the circle x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 
    for any real k if and only if -1 ≤ a ≤ 3 -/
theorem line_circle_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0) ↔ 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end line_circle_intersection_l2738_273852


namespace fractional_equation_positive_root_l2738_273863

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 := by
  sorry

end fractional_equation_positive_root_l2738_273863


namespace workshop_A_more_stable_l2738_273827

def workshop_A : List ℕ := [102, 101, 99, 98, 103, 98, 99]
def workshop_B : List ℕ := [110, 115, 90, 85, 75, 115, 110]

def variance (data : List ℕ) : ℚ :=
  let mean := (data.sum : ℚ) / data.length
  (data.map (fun x => ((x : ℚ) - mean) ^ 2)).sum / data.length

theorem workshop_A_more_stable :
  variance workshop_A < variance workshop_B :=
sorry

end workshop_A_more_stable_l2738_273827


namespace ferry_problem_l2738_273880

/-- Ferry problem -/
theorem ferry_problem (v_p v_q : ℝ) (d_p d_q : ℝ) (t_p t_q : ℝ) :
  v_p = 8 →
  d_q = 3 * d_p →
  v_q = v_p + 1 →
  t_q = t_p + 5 →
  d_p = v_p * t_p →
  d_q = v_q * t_q →
  t_p = 3 := by
  sorry

#check ferry_problem

end ferry_problem_l2738_273880


namespace fence_painting_combinations_l2738_273846

theorem fence_painting_combinations :
  let color_choices : ℕ := 6
  let method_choices : ℕ := 3
  let finish_choices : ℕ := 2
  color_choices * method_choices * finish_choices = 36 :=
by sorry

end fence_painting_combinations_l2738_273846


namespace exam_scores_sum_l2738_273808

theorem exam_scores_sum (scores : List ℝ) :
  scores.length = 6 ∧
  65 ∈ scores ∧ 75 ∈ scores ∧ 85 ∈ scores ∧ 95 ∈ scores ∧
  scores.sum / scores.length = 80 →
  ∃ x y, x ∈ scores ∧ y ∈ scores ∧ x + y = 160 :=
by sorry

end exam_scores_sum_l2738_273808


namespace third_segment_less_than_quarter_l2738_273819

open Real

/-- Given a triangle ABC with angles A, B, C, and side lengths a, b, c, 
    where angle B is divided into four equal parts, prove that the third segment 
    on AC (counting from A) is less than |AC| / 4 -/
theorem third_segment_less_than_quarter (A B C : ℝ) (a b c : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  3 * A - C < π →
  ∃ (K L M : ℝ), 0 < K ∧ K < L ∧ L < M ∧ M < b ∧
    (L - K = M - L) ∧ (M - L = b - M) ∧
    (L - K < b / 4) :=
by sorry

end third_segment_less_than_quarter_l2738_273819


namespace alice_initial_cookies_count_l2738_273870

/-- The number of chocolate chip cookies Alice initially baked -/
def alices_initial_cookies : ℕ := 91

/-- The number of peanut butter cookies Bob initially baked -/
def bobs_initial_cookies : ℕ := 7

/-- The number of cookies thrown on the floor -/
def thrown_cookies : ℕ := 29

/-- The number of additional cookies Alice baked after the accident -/
def alices_additional_cookies : ℕ := 5

/-- The number of additional cookies Bob baked after the accident -/
def bobs_additional_cookies : ℕ := 36

/-- The total number of edible cookies at the end -/
def total_edible_cookies : ℕ := 93

theorem alice_initial_cookies_count :
  alices_initial_cookies = 91 :=
by
  sorry

#check alice_initial_cookies_count

end alice_initial_cookies_count_l2738_273870


namespace min_value_of_p_l2738_273865

-- Define the polynomial p
def p (a b : ℝ) : ℝ := a^2 + 2*b^2 + 2*a + 4*b + 2008

-- Theorem stating the minimum value of p
theorem min_value_of_p :
  ∃ (min : ℝ), min = 2005 ∧ ∀ (a b : ℝ), p a b ≥ min :=
sorry

end min_value_of_p_l2738_273865


namespace election_votes_proof_l2738_273887

theorem election_votes_proof (total_votes : ℕ) : 
  (∃ (valid_votes_A valid_votes_B : ℕ),
    -- 20% of votes are invalid
    (total_votes : ℚ) * (4/5) = valid_votes_A + valid_votes_B ∧
    -- A's valid votes exceed B's by 15% of total votes
    valid_votes_A = valid_votes_B + (total_votes : ℚ) * (3/20) ∧
    -- B received 2834 valid votes
    valid_votes_B = 2834) →
  total_votes = 8720 := by
sorry

end election_votes_proof_l2738_273887


namespace sams_remaining_marbles_l2738_273862

/-- Given Sam's initial yellow marble count and the number of yellow marbles Joan took,
    prove that Sam's remaining yellow marble count is the difference between the two. -/
theorem sams_remaining_marbles (initial_count : ℕ) (marbles_taken : ℕ) 
    (h : marbles_taken ≤ initial_count) :
  initial_count - marbles_taken = initial_count - marbles_taken :=
by
  sorry

#check sams_remaining_marbles 86 25

end sams_remaining_marbles_l2738_273862


namespace max_vertex_sum_l2738_273824

def parabola (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem max_vertex_sum (a T : ℤ) (h : T ≠ 0) :
  ∃ b c : ℤ,
    (parabola a b c 0 = 0) ∧
    (parabola a b c (3 * T) = 0) ∧
    (parabola a b c (3 * T + 1) = 36) →
    ∃ x y : ℚ,
      (∀ t : ℚ, parabola a b c t ≤ parabola a b c x) ∧
      y = parabola a b c x ∧
      x + y ≤ 62 :=
by sorry

end max_vertex_sum_l2738_273824


namespace infinitely_many_primes_of_year_2022_l2738_273821

/-- A prime p is a prime of the year 2022 if there exists a positive integer n 
    such that p^2022 divides n^2022 + 2022 -/
def IsPrimeOfYear2022 (p : Nat) : Prop :=
  Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ (p^2022 ∣ n^2022 + 2022)

/-- There are infinitely many primes of the year 2022 -/
theorem infinitely_many_primes_of_year_2022 :
  ∀ N : Nat, ∃ p : Nat, p > N ∧ IsPrimeOfYear2022 p := by
  sorry

end infinitely_many_primes_of_year_2022_l2738_273821


namespace zeros_of_f_product_inequality_l2738_273874

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := (1/3) * x^3 + x + 1

theorem zeros_of_f_product_inequality (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) (h₃ : x₁ ≠ x₂) :
  g (x₁ * x₂) > g (Real.exp 2) :=
sorry

end

end zeros_of_f_product_inequality_l2738_273874


namespace arithmetic_geometric_sequence_l2738_273854

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (c * a) * (b * c) = (a * b) * (a * b) →  -- geometric sequence
  a + b + c = 15 →
  a = 20 := by sorry

end arithmetic_geometric_sequence_l2738_273854


namespace algebraic_identities_l2738_273876

theorem algebraic_identities (x y : ℝ) : 
  (3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4) ∧ 
  ((5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2) := by
  sorry

end algebraic_identities_l2738_273876


namespace george_monthly_income_l2738_273848

def monthly_income : ℝ := 240

theorem george_monthly_income :
  let half_income := monthly_income / 2
  let remaining_after_groceries := half_income - 20
  remaining_after_groceries = 100 → monthly_income = 240 :=
by
  sorry

end george_monthly_income_l2738_273848


namespace area_covered_by_strips_l2738_273864

/-- The area covered by four overlapping rectangular strips on a table -/
def area_covered (length width : ℝ) : ℝ :=
  4 * length * width - 4 * width * width

/-- Theorem stating that the area covered by four overlapping rectangular strips,
    each 16 cm long and 2 cm wide, is 112 cm² -/
theorem area_covered_by_strips :
  area_covered 16 2 = 112 := by sorry

end area_covered_by_strips_l2738_273864


namespace max_tulips_is_15_l2738_273818

/-- Represents the cost of yellow and red tulips in rubles -/
structure TulipCosts where
  yellow : ℕ
  red : ℕ

/-- Represents the number of yellow and red tulips in the bouquet -/
structure Bouquet where
  yellow : ℕ
  red : ℕ

/-- Calculates the total cost of a bouquet given the costs of tulips -/
def totalCost (b : Bouquet) (c : TulipCosts) : ℕ :=
  b.yellow * c.yellow + b.red * c.red

/-- Checks if a bouquet satisfies the conditions -/
def isValidBouquet (b : Bouquet) : Prop :=
  (b.yellow + b.red) % 2 = 1 ∧ 
  (b.yellow = b.red + 1 ∨ b.red = b.yellow + 1)

/-- The maximum number of tulips in the bouquet -/
def maxTulips : ℕ := 15

/-- The theorem stating that 15 is the maximum number of tulips -/
theorem max_tulips_is_15 (c : TulipCosts) 
    (h1 : c.yellow = 50) 
    (h2 : c.red = 31) : 
    (∀ b : Bouquet, isValidBouquet b → totalCost b c ≤ 600 → b.yellow + b.red ≤ maxTulips) ∧
    (∃ b : Bouquet, isValidBouquet b ∧ totalCost b c ≤ 600 ∧ b.yellow + b.red = maxTulips) :=
  sorry

end max_tulips_is_15_l2738_273818


namespace watermelon_melon_weight_comparison_l2738_273813

theorem watermelon_melon_weight_comparison (W M : ℝ) 
  (h1 : W > 0) (h2 : M > 0)
  (h3 : (2*W > 3*M) ∨ (3*W > 4*M))
  (h4 : ¬((2*W > 3*M) ∧ (3*W > 4*M))) :
  ¬(12*W > 18*M) := by
sorry

end watermelon_melon_weight_comparison_l2738_273813


namespace claire_apple_pies_l2738_273888

theorem claire_apple_pies :
  ∃! n : ℕ, n < 30 ∧ n % 6 = 4 ∧ n % 8 = 5 :=
by
  -- The proof goes here
  sorry

end claire_apple_pies_l2738_273888


namespace linear_equation_solution_l2738_273817

theorem linear_equation_solution : 
  ∃ x : ℝ, (2 / 3 : ℝ) * x - 2 = 4 ∧ x = 9 := by sorry

end linear_equation_solution_l2738_273817


namespace assign_roles_specific_case_l2738_273896

/-- The number of ways to assign roles in a play. -/
def assignRoles (numMen numWomen numMaleRoles numFemaleRoles numEitherRoles : ℕ) : ℕ :=
  (numMen.choose numMaleRoles) *
  (numWomen.choose numFemaleRoles) *
  ((numMen + numWomen - numMaleRoles - numFemaleRoles).choose numEitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_case :
  assignRoles 7 8 3 3 3 = 35525760 :=
by sorry

end assign_roles_specific_case_l2738_273896


namespace modulus_of_complex_power_l2738_273858

theorem modulus_of_complex_power : 
  Complex.abs ((2 - 3 * Complex.I * Real.sqrt 3) ^ 4) = 961 := by
  sorry

end modulus_of_complex_power_l2738_273858


namespace investment_rate_problem_l2738_273873

theorem investment_rate_problem (total_investment remaining_investment : ℚ)
  (rate1 rate2 required_rate : ℚ) (investment1 investment2 : ℚ) (desired_income : ℚ)
  (h1 : total_investment = 12000)
  (h2 : investment1 = 5000)
  (h3 : investment2 = 4000)
  (h4 : rate1 = 3 / 100)
  (h5 : rate2 = 9 / 200)
  (h6 : desired_income = 600)
  (h7 : remaining_investment = total_investment - investment1 - investment2)
  (h8 : desired_income = investment1 * rate1 + investment2 * rate2 + remaining_investment * required_rate) :
  required_rate = 9 / 100 := by
sorry

end investment_rate_problem_l2738_273873


namespace worker_production_theorem_l2738_273872

/-- Represents the production of two workers before and after a productivity increase -/
structure WorkerProduction where
  initial_total : ℕ
  increase1 : ℚ
  increase2 : ℚ
  final_total : ℕ

/-- Calculates the individual production of two workers after a productivity increase -/
def calculate_production (w : WorkerProduction) : ℕ × ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the workers produce 46 and 40 parts after the increase -/
theorem worker_production_theorem (w : WorkerProduction) 
  (h1 : w.initial_total = 72)
  (h2 : w.increase1 = 15 / 100)
  (h3 : w.increase2 = 25 / 100)
  (h4 : w.final_total = 86) :
  calculate_production w = (46, 40) :=
sorry

end worker_production_theorem_l2738_273872


namespace max_wrong_questions_l2738_273845

theorem max_wrong_questions (total_questions : Nat) (success_percentage : Rat) 
  (h1 : total_questions = 50)
  (h2 : success_percentage = 75 / 100) :
  ∃ (max_wrong : Nat), 
    (max_wrong ≤ total_questions) ∧ 
    ((total_questions - max_wrong : Rat) / total_questions ≥ success_percentage) ∧
    (∀ (n : Nat), n > max_wrong → (total_questions - n : Rat) / total_questions < success_percentage) ∧
    max_wrong = 12 := by
  sorry

end max_wrong_questions_l2738_273845


namespace rain_received_calculation_l2738_273841

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The normal average daily rainfall in inches -/
def normalDailyRainfall : ℚ := 2

/-- The number of days left in the year -/
def daysLeft : ℕ := 100

/-- The required average daily rainfall for the remaining days, in inches -/
def requiredDailyRainfall : ℚ := 3

/-- The amount of rain received so far this year, in inches -/
def rainReceivedSoFar : ℚ := 430

theorem rain_received_calculation :
  rainReceivedSoFar = 
    normalDailyRainfall * daysInYear - requiredDailyRainfall * daysLeft :=
by sorry

end rain_received_calculation_l2738_273841


namespace derivative_of_f_at_2_l2738_273867

-- Define the function f(x) = x
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f_at_2 : 
  HasDerivAt f 1 2 := by sorry

end derivative_of_f_at_2_l2738_273867


namespace conjunction_false_implication_l2738_273882

theorem conjunction_false_implication : ∃ (p q : Prop), (p ∧ q → False) ∧ ¬(p → False ∧ q → False) := by sorry

end conjunction_false_implication_l2738_273882


namespace workshop_workers_l2738_273886

theorem workshop_workers (total_average : ℝ) (technician_count : ℕ) (technician_average : ℝ) (non_technician_average : ℝ) 
  (h1 : total_average = 8000)
  (h2 : technician_count = 7)
  (h3 : technician_average = 12000)
  (h4 : non_technician_average = 6000) :
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * non_technician_average ∧
    total_workers = 21 :=
by sorry

end workshop_workers_l2738_273886


namespace arithmetic_sequence_sum_l2738_273836

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 9 = 16 → a 5 + a 7 = 16 := by
  sorry

end arithmetic_sequence_sum_l2738_273836


namespace fraction_equality_l2738_273829

theorem fraction_equality : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end fraction_equality_l2738_273829


namespace p_subset_q_condition_l2738_273840

def P : Set ℝ := {x : ℝ | |x + 2| ≤ 3}
def Q : Set ℝ := {x : ℝ | x ≥ -8}

theorem p_subset_q_condition : P ⊂ Q ∧ 
  (∀ x : ℝ, x ∈ P → x ∈ Q) ∧ 
  (∃ x : ℝ, x ∈ Q ∧ x ∉ P) := by
  sorry

end p_subset_q_condition_l2738_273840


namespace average_age_after_leaving_l2738_273834

def initial_average : ℝ := 40
def initial_count : ℕ := 8
def leaving_age : ℝ := 25
def final_count : ℕ := 7

theorem average_age_after_leaving :
  let initial_total_age := initial_average * initial_count
  let remaining_total_age := initial_total_age - leaving_age
  let final_average := remaining_total_age / final_count
  final_average = 42 := by sorry

end average_age_after_leaving_l2738_273834


namespace sum_first_15_odd_from_5_l2738_273898

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- The 15th odd positive integer starting from 5 -/
def last_term : ℕ := 5 + 2 * (15 - 1)

theorem sum_first_15_odd_from_5 :
  sum_odd_integers 5 15 = 285 := by sorry

end sum_first_15_odd_from_5_l2738_273898


namespace sport_water_amount_l2738_273809

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount :
  (sport_corn_syrup * sport_ratio.water) / sport_ratio.corn_syrup = 15 := by
  sorry

end sport_water_amount_l2738_273809


namespace trajectory_equation_l2738_273832

theorem trajectory_equation (x y : ℝ) :
  ((x + 3)^2 + y^2) + ((x - 3)^2 + y^2) = 38 → x^2 + y^2 = 10 := by
  sorry

end trajectory_equation_l2738_273832


namespace max_condition_l2738_273801

/-- Given a function f with derivative f' and a parameter a, 
    proves that if f has a maximum at x = a and a < 0, then -1 < a < 0 -/
theorem max_condition (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, HasDerivAt f (a * (x + 1) * (x - a)) x) →
  a < 0 →
  (∀ x, f x ≤ f a) →
  -1 < a ∧ a < 0 :=
sorry

end max_condition_l2738_273801


namespace quadratic_inequality_range_l2738_273837

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ a < -1 ∨ a > 3 := by
  sorry

end quadratic_inequality_range_l2738_273837


namespace sqrt_meaningful_range_l2738_273871

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_meaningful_range_l2738_273871


namespace number_of_subsets_of_intersection_l2738_273899

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {0, 2, 4}

theorem number_of_subsets_of_intersection : Finset.card (Finset.powerset (M ∩ N)) = 8 := by
  sorry

end number_of_subsets_of_intersection_l2738_273899


namespace unique_solution_equation_l2738_273866

/-- There exists exactly one ordered pair of real numbers (x, y) satisfying the given equation -/
theorem unique_solution_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end unique_solution_equation_l2738_273866


namespace n_pointed_star_interior_angle_sum_l2738_273815

/-- An n-pointed star where n is a multiple of 3 and n ≥ 6 -/
structure NPointedStar where
  n : ℕ
  n_multiple_of_3 : 3 ∣ n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of an n-pointed star -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 4)

/-- Theorem: The sum of interior angles of an n-pointed star is 180° (n-4) -/
theorem n_pointed_star_interior_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 4) := by
  sorry

end n_pointed_star_interior_angle_sum_l2738_273815


namespace share_ratio_l2738_273811

theorem share_ratio (total amount : ℕ) (a_share : ℕ) : 
  total = 366 → a_share = 122 → a_share / (total - a_share) = 1 / 2 := by
  sorry

end share_ratio_l2738_273811


namespace property_tax_increase_l2738_273881

/-- Calculates the property tax increase when the assessed value changes, given a fixed tax rate. -/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) :
  new_value * tax_rate - initial_value * tax_rate = 800 :=
by sorry

end property_tax_increase_l2738_273881


namespace election_lead_probability_l2738_273860

theorem election_lead_probability (total votes_A votes_B : ℕ) 
  (h_total : total = votes_A + votes_B)
  (h_A_more : votes_A > votes_B) :
  let prob := (votes_A - votes_B) / total
  prob = 1 / 43 :=
by sorry

end election_lead_probability_l2738_273860


namespace die_roll_invariant_l2738_273877

/-- Represents the faces of a tetrahedral die -/
inductive DieFace
  | one
  | two
  | three
  | four

/-- Represents a position in the triangular grid -/
structure GridPosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the die on the grid -/
structure DieState where
  position : GridPosition
  faceDown : DieFace

/-- Represents a single roll of the die -/
inductive DieRoll
  | rollLeft
  | rollRight
  | rollUp
  | rollDown

/-- Defines the starting corner of the grid -/
def startCorner : GridPosition :=
  { x := 0, y := 0 }

/-- Defines the opposite corner of the grid -/
def endCorner : GridPosition :=
  { x := 1, y := 1 }  -- Simplified for demonstration; actual values depend on grid size

/-- Function to perform a single roll -/
def performRoll (state : DieState) (roll : DieRoll) : DieState :=
  sorry  -- Implementation details omitted

/-- Theorem stating that regardless of the path taken, the die will end with face 1 down -/
theorem die_roll_invariant (path : List DieRoll) :
  let initialState : DieState := { position := startCorner, faceDown := DieFace.four }
  let finalState := path.foldl performRoll initialState
  finalState.position = endCorner → finalState.faceDown = DieFace.one :=
by sorry

end die_roll_invariant_l2738_273877


namespace money_distribution_l2738_273825

/-- Given three people A, B, and C with the following conditions:
  - The total amount between A, B, and C is 900
  - A and C together have 400
  - B and C together have 750
  Prove that C has 250. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 900)
  (h2 : A + C = 400)
  (h3 : B + C = 750) : 
  C = 250 := by
  sorry

end money_distribution_l2738_273825


namespace korona_division_l2738_273835

theorem korona_division (total : ℕ) (a b c d : ℝ) :
  total = 9246 →
  (2 * a = 3 * b) →
  (5 * b = 6 * c) →
  (3 * c = 4 * d) →
  (a + b + c + d = total) →
  ∃ (k : ℝ), k > 0 ∧ a = 1380 * k ∧ b = 2070 * k ∧ c = 2484 * k ∧ d = 3312 * k :=
by sorry

end korona_division_l2738_273835


namespace sum_of_fractions_l2738_273855

theorem sum_of_fractions : 
  (1/10 : ℚ) + (2/10 : ℚ) + (3/10 : ℚ) + (4/10 : ℚ) + (5/10 : ℚ) + 
  (6/10 : ℚ) + (7/10 : ℚ) + (8/10 : ℚ) + (9/10 : ℚ) + (90/10 : ℚ) = 
  (27/2 : ℚ) := by
sorry

end sum_of_fractions_l2738_273855


namespace interior_lattice_points_collinear_l2738_273812

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (p1 p2 p3 p4 : LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (t : Triangle) 
  (h1 : ∀ p, isOnBoundary p t → (p = t.v1 ∨ p = t.v2 ∨ p = t.v3))
  (h2 : ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    ∀ p, isInside p t → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4)) :
  ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    areCollinear p1 p2 p3 p4 := by
  sorry

end interior_lattice_points_collinear_l2738_273812


namespace total_goals_is_fifteen_l2738_273833

/-- The total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ :=
  let kickers_second := 2 * kickers_first
  let spiders_first := kickers_first / 2
  let spiders_second := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that the total goals scored is 15 when The Kickers score 2 goals in the first period -/
theorem total_goals_is_fifteen : total_goals 2 = 15 := by
  sorry

end total_goals_is_fifteen_l2738_273833


namespace shaded_area_sum_l2738_273883

theorem shaded_area_sum (r₁ : ℝ) (r₂ : ℝ) : 
  r₁ > 0 → 
  r₂ > 0 → 
  r₁ = 8 → 
  r₂ = r₁ / 2 → 
  (π * r₁^2) / 2 + (π * r₂^2) / 2 = 40 * π :=
by sorry

end shaded_area_sum_l2738_273883


namespace sequence_problem_l2738_273814

theorem sequence_problem (x y : ℝ) : 
  (∃ r : ℝ, y - 1 - 1 = 1 - 2*x ∧ 1 - 2*x = r) →  -- arithmetic sequence condition
  (∃ q : ℝ, |x+1| / (y+3) = q ∧ |x-1| / |x+1| = q) →  -- geometric sequence condition
  (x+1)*(y+1) = 4 ∨ (x+1)*(y+1) = 2*(Real.sqrt 17 - 3) := by
sorry

end sequence_problem_l2738_273814


namespace tea_blend_cost_l2738_273847

theorem tea_blend_cost (blend_ratio : ℚ) (second_tea_cost : ℚ) (blend_sell_price : ℚ) (gain_percent : ℚ) :
  blend_ratio = 5 / 3 →
  second_tea_cost = 20 →
  blend_sell_price = 21 →
  gain_percent = 12 →
  ∃ first_tea_cost : ℚ,
    first_tea_cost = 18 ∧
    (1 + gain_percent / 100) * ((blend_ratio * first_tea_cost + second_tea_cost) / (blend_ratio + 1)) = blend_sell_price :=
by sorry

end tea_blend_cost_l2738_273847


namespace wall_bricks_count_l2738_273890

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 15

/-- Time taken to complete the wall when working together -/
def time_together : ℕ := 5

theorem wall_bricks_count :
  (time_together : ℝ) * ((total_bricks / time_bricklayer1 : ℝ) +
  (total_bricks / time_bricklayer2 : ℝ) - output_reduction) = total_bricks := by
  sorry

end wall_bricks_count_l2738_273890


namespace parabola_directrix_p_value_l2738_273894

/-- Given a parabola with equation x² = 2py where p > 0,
    if its directrix has equation y = -3, then p = 6 -/
theorem parabola_directrix_p_value (p : ℝ) :
  p > 0 →
  (∀ x y : ℝ, x^2 = 2*p*y) →
  (∀ y : ℝ, y = -3 → (∀ x : ℝ, x^2 ≠ 2*p*y)) →
  p = 6 := by
  sorry

end parabola_directrix_p_value_l2738_273894


namespace minimum_additional_wins_l2738_273884

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def initial_wins : ℕ := 2

theorem minimum_additional_wins : 
  ∃ (n : ℕ), n = (puppy_cost - initial_wins * weekly_prize) / weekly_prize ∧ 
  n * weekly_prize + initial_wins * weekly_prize ≥ puppy_cost ∧
  ∀ m : ℕ, m < n → m * weekly_prize + initial_wins * weekly_prize < puppy_cost :=
by sorry

end minimum_additional_wins_l2738_273884


namespace isosceles_triangle_base_length_l2738_273895

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 15 :=
by sorry

end isosceles_triangle_base_length_l2738_273895


namespace find_other_number_l2738_273838

theorem find_other_number (a b : ℤ) : 
  3 * a + 2 * b = 105 → (a = 15 ∨ b = 15) → (a = 30 ∨ b = 30) := by
sorry

end find_other_number_l2738_273838


namespace mark_soup_donation_l2738_273843

/-- The number of homeless shelters -/
def num_shelters : ℕ := 6

/-- The number of people served by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup bought per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans of soup Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end mark_soup_donation_l2738_273843


namespace second_greatest_number_l2738_273897

def digits : List Nat := [4, 3, 1, 7, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 10) % 10 = 3 ∧
  (∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ 3 ∧ b ≠ 3 ∧ n = 100 * a + 30 + b)

def is_second_greatest (n : Nat) : Prop :=
  is_valid_number n ∧
  (∃ (m : Nat), is_valid_number m ∧ m > n) ∧
  (∀ (k : Nat), is_valid_number k ∧ k ≠ n → k ≤ n ∨ k > n ∧ (∃ (m : Nat), is_valid_number m ∧ m > n ∧ m < k))

theorem second_greatest_number : 
  ∃ (n : Nat), is_second_greatest n ∧ n = 934 := by sorry

end second_greatest_number_l2738_273897


namespace cattle_transport_time_l2738_273828

/-- Calculates the total driving time required to transport cattle to higher ground -/
theorem cattle_transport_time 
  (total_cattle : ℕ) 
  (distance : ℕ) 
  (truck_capacity : ℕ) 
  (speed : ℕ) 
  (h1 : total_cattle = 400)
  (h2 : distance = 60)
  (h3 : truck_capacity = 20)
  (h4 : speed = 60)
  : (total_cattle / truck_capacity) * (2 * distance) / speed = 40 := by
  sorry

end cattle_transport_time_l2738_273828


namespace percentage_problem_l2738_273885

theorem percentage_problem (P : ℝ) : P = 30 :=
by
  -- Define the condition from the problem
  have h1 : P / 100 * 100 = 50 / 100 * 40 + 10 := by sorry
  
  -- Proof goes here
  sorry

end percentage_problem_l2738_273885


namespace bus_trip_distance_l2738_273850

/-- Given a bus trip with specific conditions, prove that the distance traveled is 210 miles. -/
theorem bus_trip_distance (actual_speed : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) 
  (h1 : actual_speed = 30)
  (h2 : speed_increase = 5)
  (h3 : time_reduction = 1)
  (h4 : ∀ (distance : ℝ), distance / actual_speed = distance / (actual_speed + speed_increase) + time_reduction) :
  ∃ (distance : ℝ), distance = 210 := by
  sorry

end bus_trip_distance_l2738_273850


namespace repetend_of_five_seventeenths_l2738_273868

/-- The decimal representation of 5/17 has a 6-digit repetend equal to 294117 -/
theorem repetend_of_five_seventeenths :
  ∃ (a b : ℕ), (5 : ℚ) / 17 = (a : ℚ) + (b : ℚ) / 999999 ∧ b = 294117 := by
  sorry

end repetend_of_five_seventeenths_l2738_273868


namespace fundraising_event_l2738_273831

theorem fundraising_event (p : ℝ) (initial_boys : ℕ) :
  -- Initial conditions
  initial_boys = Int.floor (0.35 * p) →
  -- Changes in group composition
  (initial_boys - 3 + 2) / (p + 3) = 0.3 →
  -- Conclusion
  initial_boys = 13 := by
sorry

end fundraising_event_l2738_273831


namespace volleyball_handshakes_l2738_273859

theorem volleyball_handshakes (total_handshakes : ℕ) (h : total_handshakes = 496) :
  ∃ (n : ℕ), 
    n * (n - 1) / 2 = total_handshakes ∧
    ∀ (coach_handshakes : ℕ), 
      n * (n - 1) / 2 + coach_handshakes = total_handshakes → 
      coach_handshakes ≥ 0 ∧
      (coach_handshakes = 0 → 
        ∀ (other_coach_handshakes : ℕ), 
          n * (n - 1) / 2 + other_coach_handshakes = total_handshakes → 
          other_coach_handshakes ≥ coach_handshakes) :=
by sorry

end volleyball_handshakes_l2738_273859


namespace gcd_45885_30515_l2738_273823

theorem gcd_45885_30515 : Nat.gcd 45885 30515 = 10 := by
  sorry

end gcd_45885_30515_l2738_273823


namespace range_of_g_l2738_273892

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := f x - 2*x

-- Define the interval
def I : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_g :
  {y | ∃ x ∈ I, g x = y} = {y | -1 ≤ y ∧ y ≤ 8} := by sorry

end range_of_g_l2738_273892


namespace student_mistake_difference_l2738_273869

theorem student_mistake_difference (n : ℚ) (h : n = 480) : 5/6 * n - 5/16 * n = 250 := by
  sorry

end student_mistake_difference_l2738_273869


namespace line_divides_polygon_equally_l2738_273810

/-- Polygon type representing a closed shape with vertices --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line type representing a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the area of a polygon using the shoelace formula --/
def polygonArea (p : Polygon) : ℝ := sorry

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a line divides a polygon into two equal areas --/
def dividesEqualArea (l : Line) (p : Polygon) : Prop := sorry

/-- The main theorem --/
theorem line_divides_polygon_equally (polygon : Polygon) (line : Line) :
  polygon.vertices = [(0, 0), (0, 6), (4, 6), (4, 4), (6, 4), (6, 0)] →
  line.slope = -1/3 →
  line.intercept = 11/3 →
  pointOnLine line (2, 3) →
  dividesEqualArea line polygon := by
  sorry

end line_divides_polygon_equally_l2738_273810


namespace polynomial_value_at_root_l2738_273844

theorem polynomial_value_at_root (p : ℝ) : 
  p^3 - 5*p + 1 = 0 → p^4 - 3*p^3 - 5*p^2 + 16*p + 2015 = 2018 := by
  sorry

end polynomial_value_at_root_l2738_273844


namespace container_volume_ratio_l2738_273839

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : 2/3 * A = 5/8 * B) : A / B = 15/16 := by
  sorry

end container_volume_ratio_l2738_273839


namespace existence_of_counterexample_l2738_273893

theorem existence_of_counterexample (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b, c * b^2 ≥ a * b^2 := by
sorry

end existence_of_counterexample_l2738_273893


namespace confidence_95_error_5_l2738_273816

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := {r : ℝ // 0 < r ∧ r < 1}

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := {r : ℝ // 0 ≤ r ∧ r ≤ 1}

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

/-- The theorem states that for a 95% confidence level, the error probability is 5% -/
theorem confidence_95_error_5 :
  let cl95 : ConfidenceLevel := ⟨0.95, by sorry⟩
  calculateErrorProbability cl95 = ⟨0.05, by sorry⟩ :=
sorry

end confidence_95_error_5_l2738_273816


namespace rotate_vector_2_3_l2738_273851

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Rotates a 2D vector 90 degrees clockwise -/
def rotate90Clockwise (v : Vector2D) : Vector2D :=
  { x := v.y, y := -v.x }

/-- The theorem stating that rotating (2, 3) by 90 degrees clockwise results in (3, -2) -/
theorem rotate_vector_2_3 :
  rotate90Clockwise { x := 2, y := 3 } = { x := 3, y := -2 } := by
  sorry

end rotate_vector_2_3_l2738_273851


namespace partial_sum_base_7_l2738_273826

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem partial_sum_base_7 :
  let a := [2, 3, 4, 5, 1]
  let b := [1, 5, 6, 4, 2]
  let sum := [4, 2, 4, 2, 3]
  let base := 7
  (to_decimal a base + to_decimal b base = to_decimal sum base) ∧
  (∀ d ∈ (a ++ b ++ sum), d < base) :=
by sorry

end partial_sum_base_7_l2738_273826


namespace complex_cube_root_l2738_273805

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end complex_cube_root_l2738_273805


namespace cubic_yards_to_cubic_feet_l2738_273878

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Number of cubic yards we're converting -/
def cubic_yards : ℝ := 4

/-- Theorem stating that 4 cubic yards equals 108 cubic feet -/
theorem cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 108 := by sorry

end cubic_yards_to_cubic_feet_l2738_273878


namespace burger_cost_l2738_273804

/-- Proves that the cost of each burger is $3.50 given Selena's expenses --/
theorem burger_cost (tip : ℝ) (steak_price : ℝ) (ice_cream_price : ℝ) (remaining : ℝ) :
  tip = 99 →
  steak_price = 24 →
  ice_cream_price = 2 →
  remaining = 38 →
  ∃ (burger_price : ℝ),
    burger_price = 3.5 ∧
    tip = 2 * steak_price + 2 * burger_price + 3 * ice_cream_price + remaining :=
by
  sorry

end burger_cost_l2738_273804


namespace uniform_color_subgrid_l2738_273806

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- A theorem stating that in any two-color infinite grid, there exist two horizontal
    and two vertical lines forming a subgrid with uniformly colored intersection points -/
theorem uniform_color_subgrid
  (coloring : ColoringFunction) :
  ∃ (x₁ x₂ y₁ y₂ : ℤ) (c : Color),
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    coloring ⟨x₁, y₁⟩ = c ∧
    coloring ⟨x₁, y₂⟩ = c ∧
    coloring ⟨x₂, y₁⟩ = c ∧
    coloring ⟨x₂, y₂⟩ = c :=
by sorry


end uniform_color_subgrid_l2738_273806


namespace rectangle_area_l2738_273889

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end rectangle_area_l2738_273889


namespace some_number_value_l2738_273875

theorem some_number_value (x : ℝ) : (50 + 20/x) * x = 4520 → x = 90 := by
  sorry

end some_number_value_l2738_273875


namespace M_union_N_eq_l2738_273807

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem M_union_N_eq : M ∪ N = {-2, -1, 0, 1} := by sorry

end M_union_N_eq_l2738_273807


namespace milkshake_fraction_l2738_273861

theorem milkshake_fraction (total : ℚ) (milkshake_fraction : ℚ) 
  (lost : ℚ) (remaining : ℚ) : 
  total = 28 →
  lost = 11 →
  remaining = 1 →
  (1 - milkshake_fraction) * total / 2 = lost + remaining →
  milkshake_fraction = 1 / 7 := by
sorry

end milkshake_fraction_l2738_273861


namespace intersection_area_theorem_l2738_273802

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Defines the position of points P, Q, R on the cube edges -/
structure PointsOnCube where
  cube : Cube
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Calculates the area of the intersection polygon -/
def intersectionArea (c : Cube) (pts : PointsOnCube) : ℝ :=
  sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (pts : PointsOnCube) :
  c.sideLength = 30 ∧
  pts.P.x = 10 ∧ pts.P.y = 0 ∧ pts.P.z = 0 ∧
  pts.Q.x = 30 ∧ pts.Q.y = 0 ∧ pts.Q.z = 20 ∧
  pts.R.x = 30 ∧ pts.R.y = 5 ∧ pts.R.z = 30 →
  intersectionArea c pts = 450 := by
  sorry

end intersection_area_theorem_l2738_273802


namespace prob_good_friends_is_one_fourth_l2738_273800

/-- The number of balls in the pocket -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The probability space of drawing two balls with replacement -/
def draw_space : Finset (ℕ × ℕ) := ball_numbers.product ball_numbers

/-- The event of drawing the same number (becoming "good friends") -/
def good_friends : Finset (ℕ × ℕ) := 
  draw_space.filter (fun p => p.1 = p.2)

/-- The probability of becoming "good friends" -/
def prob_good_friends : ℚ :=
  good_friends.card / draw_space.card

theorem prob_good_friends_is_one_fourth : 
  prob_good_friends = 1 / 4 := by sorry

end prob_good_friends_is_one_fourth_l2738_273800


namespace number_division_theorem_l2738_273820

theorem number_division_theorem : 
  ∃ (n : ℕ), (n : ℝ) / 189 = 18.444444444444443 :=
by
  -- The proof would go here
  sorry

end number_division_theorem_l2738_273820


namespace system_solution_l2738_273822

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - 2*y = -k ∧ 2*x - y = 8) → k = 2 := by
sorry

end system_solution_l2738_273822


namespace functional_equation_solutions_l2738_273803

theorem functional_equation_solutions (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a) →
  (∀ x, f x = 0) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 2 = 0 then 0 else k) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 4 = 0 then 0 else if x % 4 = 2 then k else k) ∨
  (∃ k : ℤ, ∀ x, f x = k * x ^ 2) :=
by sorry

end functional_equation_solutions_l2738_273803


namespace alice_bob_number_sum_l2738_273853

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ∈ Finset.range 50 →
    B ∈ Finset.range 50 →
    (∀ x ∈ Finset.range 50, x ≠ A → ¬(A > x ↔ B > x)) →
    (∀ y ∈ Finset.range 50, y ≠ B → (B > y ↔ A < y)) →
    is_prime B →
    B % 2 = 0 →
    is_perfect_square (90 * B + A) →
    A + B = 18 :=
by sorry

end alice_bob_number_sum_l2738_273853


namespace abs_sum_inequality_l2738_273891

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end abs_sum_inequality_l2738_273891


namespace smaller_circle_area_l2738_273879

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  -- Centers of the smaller and larger circles
  S : ℝ × ℝ
  L : ℝ × ℝ
  -- Radii of the smaller and larger circles
  r_small : ℝ
  r_large : ℝ
  -- Point P from which tangents are drawn
  P : ℝ × ℝ
  -- Points of tangency on the circles
  A : ℝ × ℝ
  B : ℝ × ℝ
  -- The circles are externally tangent
  externally_tangent : dist S L = r_small + r_large
  -- PAB is a common tangent
  tangent_line : dist P A = dist A B
  -- A is on the smaller circle, B is on the larger circle
  on_circles : dist S A = r_small ∧ dist L B = r_large
  -- Length condition
  length_condition : dist P A = 4 ∧ dist A B = 4

/-- The area of the smaller circle in the TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r_small ^ 2 = 2 * Real.pi := by
  sorry


end smaller_circle_area_l2738_273879
