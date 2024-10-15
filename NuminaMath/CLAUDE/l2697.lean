import Mathlib

namespace NUMINAMATH_CALUDE_economizable_disjoint_from_non_economizable_l2697_269789

-- Define the type for expenses
inductive Expense
  | LoanPayment
  | TaxPayment
  | QualificationCourse
  | HomeInternet
  | TravelExpense
  | VideoCamera
  | DomainPayment
  | CoffeeShopVisit

-- Define the property of being economizable
def is_economizable (e : Expense) : Prop :=
  match e with
  | Expense.HomeInternet => True
  | Expense.TravelExpense => True
  | Expense.VideoCamera => True
  | Expense.DomainPayment => True
  | Expense.CoffeeShopVisit => True
  | _ => False

-- Define the sets of economizable and non-economizable expenses
def economizable_expenses : Set Expense :=
  {e | is_economizable e}

def non_economizable_expenses : Set Expense :=
  {e | ¬is_economizable e}

-- Theorem statement
theorem economizable_disjoint_from_non_economizable :
  economizable_expenses ∩ non_economizable_expenses = ∅ :=
by sorry

end NUMINAMATH_CALUDE_economizable_disjoint_from_non_economizable_l2697_269789


namespace NUMINAMATH_CALUDE_evaluate_expression_l2697_269769

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2697_269769


namespace NUMINAMATH_CALUDE_light_year_scientific_notation_l2697_269781

theorem light_year_scientific_notation :
  9500000000000 = 9.5 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_light_year_scientific_notation_l2697_269781


namespace NUMINAMATH_CALUDE_bird_families_left_l2697_269729

theorem bird_families_left (total : ℕ) (to_africa : ℕ) (to_asia : ℕ) 
  (h1 : total = 85) 
  (h2 : to_africa = 23) 
  (h3 : to_asia = 37) : 
  total - (to_africa + to_asia) = 25 := by
sorry

end NUMINAMATH_CALUDE_bird_families_left_l2697_269729


namespace NUMINAMATH_CALUDE_solution_volume_l2697_269760

/-- Proves that the total volume of a solution is 10 liters, given that it contains 2.5 liters of pure acid and has a 25% concentration. -/
theorem solution_volume (acid_volume : ℝ) (concentration : ℝ) :
  acid_volume = 2.5 →
  concentration = 0.25 →
  acid_volume / concentration = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l2697_269760


namespace NUMINAMATH_CALUDE_afternoon_fliers_fraction_l2697_269744

theorem afternoon_fliers_fraction (total_fliers : ℕ) 
  (morning_fraction : ℚ) (remaining_fliers : ℕ) :
  total_fliers = 1000 →
  morning_fraction = 1 / 5 →
  remaining_fliers = 600 →
  let morning_sent := total_fliers * morning_fraction
  let after_morning := total_fliers - morning_sent
  let afternoon_sent := after_morning - remaining_fliers
  afternoon_sent / after_morning = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_fliers_fraction_l2697_269744


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2697_269731

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ+) : ℕ := sorry

/-- The product of 30 and 450 -/
def product : ℕ+ := 30 * 450

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2697_269731


namespace NUMINAMATH_CALUDE_remainder_mod_nine_l2697_269710

theorem remainder_mod_nine (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_nine_l2697_269710


namespace NUMINAMATH_CALUDE_uncool_students_l2697_269764

theorem uncool_students (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) (cool_siblings : ℕ) 
  (h1 : total = 50)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 30)
  (h4 : cool_both = 15)
  (h5 : cool_siblings = 10) :
  total - (cool_dads + cool_moms - cool_both) - cool_siblings = 10 := by
  sorry

end NUMINAMATH_CALUDE_uncool_students_l2697_269764


namespace NUMINAMATH_CALUDE_claire_age_l2697_269715

/-- Given the ages of Claire, Leo, and Mia, prove Claire's age --/
theorem claire_age (mia leo claire : ℕ) 
  (h1 : claire = leo - 5) 
  (h2 : leo = mia + 4) 
  (h3 : mia = 20) : 
  claire = 19 := by
sorry

end NUMINAMATH_CALUDE_claire_age_l2697_269715


namespace NUMINAMATH_CALUDE_tom_dance_duration_l2697_269779

/-- Given that Tom dances 4 times a week for 10 years and danced for a total of 4160 hours,
    prove that he dances for 2 hours at a time. -/
theorem tom_dance_duration (
  dances_per_week : ℕ)
  (years : ℕ)
  (total_hours : ℕ)
  (h1 : dances_per_week = 4)
  (h2 : years = 10)
  (h3 : total_hours = 4160) :
  total_hours / (dances_per_week * years * 52) = 2 := by
sorry

end NUMINAMATH_CALUDE_tom_dance_duration_l2697_269779


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l2697_269750

theorem min_value_of_function (θ : ℝ) (h : 1 - Real.cos θ ≠ 0) :
  (2 - Real.sin θ) / (1 - Real.cos θ) ≥ 3/4 :=
sorry

theorem min_value_attained (θ : ℝ) (h : 1 - Real.cos θ ≠ 0) :
  ∃ θ₀, (2 - Real.sin θ₀) / (1 - Real.cos θ₀) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l2697_269750


namespace NUMINAMATH_CALUDE_max_negative_integers_l2697_269798

theorem max_negative_integers (S : Finset ℤ) (h_card : S.card = 18)
  (h_distinct : S.card = S.toList.length)
  (h_greater_than_one : ∀ x ∈ S, x > 1)
  (h_product_negative : (S.prod id) < 0) :
  (S.filter (λ x => x < 0)).card ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_negative_integers_l2697_269798


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2697_269725

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Statement to prove
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2697_269725


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2697_269791

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) →
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2697_269791


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l2697_269792

/-- Calculate the net rate of pay for a driver --/
theorem driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gasoline_cost = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l2697_269792


namespace NUMINAMATH_CALUDE_sum_of_roots_is_nine_halves_l2697_269753

-- Define the polynomials
def p (x : ℝ) : ℝ := 2 * x^3 + x^2 - 8 * x + 20
def q (x : ℝ) : ℝ := 5 * x^3 - 25 * x^2 + 19

-- Define the equation
def equation (x : ℝ) : Prop := p x = 0 ∨ q x = 0

-- Theorem statement
theorem sum_of_roots_is_nine_halves :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, equation x) ∧
    (∀ x, equation x → x ∈ roots) ∧
    (Finset.sum roots id = 9/2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_nine_halves_l2697_269753


namespace NUMINAMATH_CALUDE_digit_difference_1234_l2697_269787

/-- The number of digits in the base-b representation of a positive integer n -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n < b then 1 else Nat.log b n + 1

/-- The difference in the number of digits between base-4 and base-9 representations of 1234 -/
theorem digit_difference_1234 :
  num_digits 1234 4 - num_digits 1234 9 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_difference_1234_l2697_269787


namespace NUMINAMATH_CALUDE_largest_side_of_crate_with_cylinder_l2697_269727

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  width : ℝ
  depth : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (2 * cylinder.radius ≤ crate.width ∧ 2 * cylinder.radius ≤ crate.depth) ∨
  (2 * cylinder.radius ≤ crate.width ∧ 2 * cylinder.radius ≤ crate.height) ∨
  (2 * cylinder.radius ≤ crate.depth ∧ 2 * cylinder.radius ≤ crate.height)

theorem largest_side_of_crate_with_cylinder 
  (crate : CrateDimensions) 
  (cylinder : Cylinder) 
  (h1 : crate.width = 7)
  (h2 : crate.depth = 8)
  (h3 : cylinder.radius = 7)
  (h4 : cylinderFitsInCrate crate cylinder) :
  max crate.width (max crate.depth crate.height) = 14 := by
  sorry

#check largest_side_of_crate_with_cylinder

end NUMINAMATH_CALUDE_largest_side_of_crate_with_cylinder_l2697_269727


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l2697_269773

def M : Set ℕ := {0, 1, 2, 3, 4, 5}
def N : Set ℕ := {0, 2, 3}

theorem complement_of_N_in_M :
  M \ N = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l2697_269773


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2697_269719

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 7)^2 = 36 ∧ (x₂ - 7)^2 = 36 ∧ x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2697_269719


namespace NUMINAMATH_CALUDE_max_a_for_inequality_solution_R_l2697_269702

theorem max_a_for_inequality_solution_R : 
  ∃ (a_max : ℝ), 
    (∀ (a : ℝ), (∀ (x : ℝ), |x - a| + |x - 3| ≥ 2*a) → a ≤ a_max) ∧
    (∀ (x : ℝ), |x - a_max| + |x - 3| ≥ 2*a_max) ∧
    (∀ (a : ℝ), a > a_max → ∃ (x : ℝ), |x - a| + |x - 3| < 2*a) ∧
    a_max = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_solution_R_l2697_269702


namespace NUMINAMATH_CALUDE_investment_interest_difference_l2697_269714

theorem investment_interest_difference 
  (total_investment : ℝ)
  (investment_x : ℝ)
  (rate_x : ℝ)
  (rate_y : ℝ)
  (h1 : total_investment = 100000)
  (h2 : investment_x = 42000)
  (h3 : rate_x = 0.23)
  (h4 : rate_y = 0.17) :
  let investment_y := total_investment - investment_x
  let interest_x := investment_x * rate_x
  let interest_y := investment_y * rate_y
  interest_y - interest_x = 200 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_difference_l2697_269714


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l2697_269745

def inner_radius : ℝ := 25
def track_width : ℝ := 15

def outer_radius : ℝ := inner_radius + track_width

theorem concentric_circles_properties :
  let inner_circ := 2 * Real.pi * inner_radius
  let outer_circ := 2 * Real.pi * outer_radius
  let inner_area := Real.pi * inner_radius^2
  let outer_area := Real.pi * outer_radius^2
  (outer_circ - inner_circ = 30 * Real.pi) ∧
  (outer_area - inner_area = 975 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l2697_269745


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2697_269758

/-- Represents an event in a probability space -/
structure Event (Ω : Type) :=
  (set : Set Ω)

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive {Ω : Type} (A B : Event Ω) : Prop :=
  A.set ∩ B.set = ∅

/-- Represents the sample space for shooting at a target -/
inductive ShootingTarget
  | ring7
  | ring8
  | miss

/-- Represents the sample space for two people shooting -/
inductive TwoPeopleShooting
  | bothHit
  | AHitBMiss
  | AMissBHit
  | bothMiss

/-- Represents the sample space for drawing two balls -/
inductive TwoBallDraw
  | redRed
  | redBlack
  | blackRed
  | blackBlack

/-- Event 1: Hitting the 7th ring -/
def hit7th : Event ShootingTarget :=
  ⟨{ShootingTarget.ring7}⟩

/-- Event 1: Hitting the 8th ring -/
def hit8th : Event ShootingTarget :=
  ⟨{ShootingTarget.ring8}⟩

/-- Event 2: At least one person hits the target -/
def atLeastOneHit : Event TwoPeopleShooting :=
  ⟨{TwoPeopleShooting.bothHit, TwoPeopleShooting.AHitBMiss, TwoPeopleShooting.AMissBHit}⟩

/-- Event 2: A hits, B misses -/
def AHitBMiss : Event TwoPeopleShooting :=
  ⟨{TwoPeopleShooting.AHitBMiss}⟩

/-- Event 3: At least one black ball -/
def atLeastOneBlack : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redBlack, TwoBallDraw.blackRed, TwoBallDraw.blackBlack}⟩

/-- Event 3: Both balls are red -/
def bothRed : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redRed}⟩

/-- Event 4: No black balls -/
def noBlack : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redRed}⟩

/-- Event 4: Exactly one red ball -/
def oneRed : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redBlack, TwoBallDraw.blackRed}⟩

theorem mutually_exclusive_events :
  (mutually_exclusive hit7th hit8th) ∧
  (¬mutually_exclusive atLeastOneHit AHitBMiss) ∧
  (mutually_exclusive atLeastOneBlack bothRed) ∧
  (mutually_exclusive noBlack oneRed) := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2697_269758


namespace NUMINAMATH_CALUDE_desired_depth_is_50_l2697_269726

/-- Calculates the desired depth to be dug given initial and new working conditions -/
def desired_depth (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℕ) 
                  (new_hours : ℕ) (extra_men : ℕ) : ℕ :=
  let total_men := initial_men + extra_men
  let numerator := total_men * new_hours * initial_depth
  let denominator := initial_men * initial_hours
  numerator / denominator

/-- Theorem stating that the desired depth is 50 meters under given conditions -/
theorem desired_depth_is_50 :
  desired_depth 54 8 30 6 66 = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_is_50_l2697_269726


namespace NUMINAMATH_CALUDE_triangle_side_length_l2697_269706

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2697_269706


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2697_269767

theorem simplify_sqrt_sum : Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2697_269767


namespace NUMINAMATH_CALUDE_average_daily_income_l2697_269747

def earnings : List ℝ := [620, 850, 760, 950, 680, 890, 720, 900, 780, 830, 800, 880]

theorem average_daily_income :
  (earnings.sum / earnings.length : ℝ) = 805 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_income_l2697_269747


namespace NUMINAMATH_CALUDE_quadratic_complex_solution_sum_l2697_269761

theorem quadratic_complex_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 6 * x^2 + 1 = 5 * x - 16 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 443/144 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_solution_sum_l2697_269761


namespace NUMINAMATH_CALUDE_chess_game_results_l2697_269778

/-- Represents the outcome of a chess game. -/
inductive GameOutcome
  | Win
  | Loss
  | Draw

/-- Calculates points for a given game outcome. -/
def pointsForOutcome (outcome : GameOutcome) : Int :=
  match outcome with
  | GameOutcome.Win => 3
  | GameOutcome.Loss => -2
  | GameOutcome.Draw => 0

/-- Represents a player's game results. -/
structure PlayerResults :=
  (wins : Nat)
  (losses : Nat)
  (draws : Nat)

/-- Calculates total points for a player given their results. -/
def totalPoints (results : PlayerResults) : Int :=
  (results.wins * pointsForOutcome GameOutcome.Win) +
  (results.losses * pointsForOutcome GameOutcome.Loss) +
  (results.draws * pointsForOutcome GameOutcome.Draw)

theorem chess_game_results : ∃ (petr_losses : Nat),
  let petr := PlayerResults.mk 6 petr_losses 2
  let karel := PlayerResults.mk (petr_losses) 6 2
  totalPoints karel = 9 ∧
  petr.wins + petr.losses + petr.draws = 15 ∧
  totalPoints karel > totalPoints petr :=
by sorry

end NUMINAMATH_CALUDE_chess_game_results_l2697_269778


namespace NUMINAMATH_CALUDE_base5_addition_l2697_269722

/-- Addition of two numbers in base 5 --/
def base5_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 5 --/
def to_base5 (n : ℕ) : ℕ := sorry

/-- Conversion from base 5 to base 10 --/
def from_base5 (n : ℕ) : ℕ := sorry

theorem base5_addition : base5_add 14 132 = 101 := by sorry

end NUMINAMATH_CALUDE_base5_addition_l2697_269722


namespace NUMINAMATH_CALUDE_inequality_proof_l2697_269709

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2697_269709


namespace NUMINAMATH_CALUDE_binomial_expectation_variance_relation_l2697_269720

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If 3E(X) = 10D(X) for a binomial random variable X, then p = 0.7 -/
theorem binomial_expectation_variance_relation (X : BinomialRV) 
  (h : 3 * expectation X = 10 * variance X) : X.p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_variance_relation_l2697_269720


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2697_269783

/-- Represents the amount of lingonberries picked on each day -/
structure BerryPicking where
  monday : ℕ
  tuesday : ℕ
  thursday : ℕ

/-- Represents the job parameters -/
structure JobParameters where
  totalEarningsGoal : ℕ
  payRate : ℕ

theorem tuesday_to_monday_ratio (job : JobParameters) (pick : BerryPicking) :
  job.totalEarningsGoal = 100 ∧
  job.payRate = 2 ∧
  pick.monday = 8 ∧
  pick.thursday = 18 →
  pick.tuesday / pick.monday = 3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2697_269783


namespace NUMINAMATH_CALUDE_number_division_problem_l2697_269784

theorem number_division_problem (n : ℕ) : 
  n % 23 = 19 → n / 23 = 17 → (10 * n) / 23 + (10 * n) % 23 = 184 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2697_269784


namespace NUMINAMATH_CALUDE_roots_sum_expression_l2697_269724

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 3 * x - 4 = 0

theorem roots_sum_expression (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) 
  (h₃ : x₁ ≠ x₂) : 
  2 * x₁^2 + 3 * x₂^2 = 178 / 25 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_expression_l2697_269724


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2697_269716

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₁ d : ℤ) : ℕ → ℤ :=
  fun n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic progression -/
def SumArithmeticProgression (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_progression_first_term
  (a₁ d : ℤ)
  (h_increasing : d > 0)
  (h_condition1 : (ArithmeticProgression a₁ d 9) * (ArithmeticProgression a₁ d 17) >
    (SumArithmeticProgression a₁ d 14) + 12)
  (h_condition2 : (ArithmeticProgression a₁ d 11) * (ArithmeticProgression a₁ d 15) <
    (SumArithmeticProgression a₁ d 14) + 47) :
  a₁ ∈ ({-9, -8, -7, -6, -4, -3, -2, -1} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2697_269716


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2697_269743

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + x - 2 = 0) ∧ (a * 1^2 + 1 - 2 = 0) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2697_269743


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l2697_269793

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 4

/-- The number of gift card styles -/
def gift_card_styles : ℕ := 2

/-- The number of decorative bow options -/
def decorative_bow_options : ℕ := 2

/-- Theorem stating the total number of gift wrapping combinations -/
theorem gift_wrapping_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_card_styles * decorative_bow_options = 480 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l2697_269793


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2697_269728

theorem trigonometric_identity (α : ℝ) :
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * (Real.cos (2 * α + Real.pi))^2 - 1) = 2 * Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2697_269728


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l2697_269734

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The given line -/
def L1 : Set (ℝ × ℝ) := {(x, y) | 2*x - 6*y + 1 = 0}

/-- The line to be proven -/
def L2 : Set (ℝ × ℝ) := {(x, y) | 3*x + y + 6 = 0}

theorem perpendicular_and_tangent :
  (∃ (a b : ℝ), (a, b) ∈ L2 ∧ f a = b) ∧  -- L2 is tangent to the curve
  (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 ∧ (x2, y2) ∈ L1 ∧ x1 ≠ x2 →
    (x1 - x2) * (3 * (y1 - y2)) = -1) -- L1 and L2 are perpendicular
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_and_tangent_l2697_269734


namespace NUMINAMATH_CALUDE_train_distance_theorem_l2697_269737

/-- Calculates the distance traveled by a train given its average speed and travel time. -/
def train_distance (average_speed : ℝ) (travel_time : ℝ) : ℝ :=
  average_speed * travel_time

/-- Represents the train journey details -/
structure TrainJourney where
  average_speed : ℝ
  start_time : ℝ
  end_time : ℝ
  halt_time : ℝ

/-- Theorem stating the distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.average_speed = 87)
  (h2 : journey.start_time = 9)
  (h3 : journey.end_time = 13.75)
  (h4 : journey.halt_time = 0.75) :
  train_distance journey.average_speed (journey.end_time - journey.start_time - journey.halt_time) = 348 := by
  sorry


end NUMINAMATH_CALUDE_train_distance_theorem_l2697_269737


namespace NUMINAMATH_CALUDE_cos_240_deg_l2697_269776

/-- Cosine of 240 degrees is equal to -1/2 -/
theorem cos_240_deg : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_deg_l2697_269776


namespace NUMINAMATH_CALUDE_product_invariant_under_decrease_l2697_269752

theorem product_invariant_under_decrease :
  ∃ (a b c d e : ℝ),
    a * b * c * d * e ≠ 0 ∧
    (a - 1) * (b - 1) * (c - 1) * (d - 1) * (e - 1) = a * b * c * d * e :=
by sorry

end NUMINAMATH_CALUDE_product_invariant_under_decrease_l2697_269752


namespace NUMINAMATH_CALUDE_parallelepiped_theorem_l2697_269742

/-- Represents a parallelepiped with a sphere inscribed --/
structure ParallelepipedWithSphere where
  -- Edge length of the base square
  base_edge : ℝ
  -- Height of the parallelepiped (length of A₁A)
  height : ℝ
  -- Distance from C to K on edge CD
  ck : ℝ
  -- Distance from K to D on edge CD
  kd : ℝ

/-- Properties of the parallelepiped and inscribed sphere --/
def parallelepiped_properties (p : ParallelepipedWithSphere) : Prop :=
  -- Edge A₁A is perpendicular to face ABCD (implied by the structure)
  -- Sphere Ω touches edges BB₁, B₁C₁, C₁C, CB, CD (implied by the structure)
  -- Sphere Ω touches edge CD at point K
  p.ck + p.kd = p.base_edge ∧
  -- Given values for CK and KD
  p.ck = 9 ∧ p.kd = 1 ∧
  -- Sphere Ω touches edge A₁D₁ (implied by the structure)
  -- The base is a square (implied by the problem description)
  p.base_edge = p.height

/-- Main theorem stating the properties to be proven --/
theorem parallelepiped_theorem (p : ParallelepipedWithSphere) 
  (h : parallelepiped_properties p) : 
  p.height = 18 ∧ 
  p.height * p.base_edge * p.base_edge = 1944 ∧ 
  ∃ (r : ℝ), r * r = 90 ∧ r = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_theorem_l2697_269742


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_minus_b_l2697_269756

theorem quadratic_roots_imply_a_minus_b (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_minus_b_l2697_269756


namespace NUMINAMATH_CALUDE_locus_of_point_on_line_intersecting_ellipse_l2697_269775

/-- The locus of point P on a line segment AB, where the line intersects an ellipse --/
theorem locus_of_point_on_line_intersecting_ellipse 
  (x y : ℝ) 
  (h_ellipse : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2/4 = 1 ∧ 
    x₂^2 + y₂^2/4 = 1 ∧ 
    y₁ - x₁ = y₂ - x₂)
  (h_slope : ∃ (c : ℝ), y = x + c)
  (h_ratio : ∃ (x_a y_a x_b y_b : ℝ), 
    (x - x_a)^2 + (y - y_a)^2 = 4 * ((x_b - x)^2 + (y_b - y)^2))
  (h_bound : |y - x| < Real.sqrt 5) :
  4*x + y = (2/3) * Real.sqrt (5 - (y-x)^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_on_line_intersecting_ellipse_l2697_269775


namespace NUMINAMATH_CALUDE_real_part_of_i_over_one_plus_i_l2697_269703

theorem real_part_of_i_over_one_plus_i : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  Complex.re z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_over_one_plus_i_l2697_269703


namespace NUMINAMATH_CALUDE_rita_bought_three_pants_l2697_269786

/-- Represents the shopping trip of Rita -/
structure ShoppingTrip where
  dresses : Nat
  jackets : Nat
  dress_cost : Nat
  jacket_cost : Nat
  pants_cost : Nat
  transport_cost : Nat
  initial_money : Nat
  remaining_money : Nat

/-- Calculates the number of pairs of pants bought given a shopping trip -/
def pants_bought (trip : ShoppingTrip) : Nat :=
  let total_spent := trip.initial_money - trip.remaining_money
  let known_expenses := trip.dresses * trip.dress_cost + trip.jackets * trip.jacket_cost + trip.transport_cost
  let pants_expense := total_spent - known_expenses
  pants_expense / trip.pants_cost

/-- Theorem stating that Rita bought 3 pairs of pants -/
theorem rita_bought_three_pants :
  let trip : ShoppingTrip := {
    dresses := 5,
    jackets := 4,
    dress_cost := 20,
    jacket_cost := 30,
    pants_cost := 12,
    transport_cost := 5,
    initial_money := 400,
    remaining_money := 139
  }
  pants_bought trip = 3 := by sorry

end NUMINAMATH_CALUDE_rita_bought_three_pants_l2697_269786


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l2697_269780

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3*a) < 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem 1: When a = 1, the range of x satisfying both p and q is (2, 3)
theorem range_of_x_when_a_is_one :
  {x : ℝ | p x 1 ∧ q x} = Set.Ioo 2 3 := by sorry

-- Theorem 2: The range of a when p is necessary but not sufficient for q is [1, 2]
theorem range_of_a_necessary_not_sufficient :
  {a : ℝ | a > 0 ∧ (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l2697_269780


namespace NUMINAMATH_CALUDE_paving_cost_theorem_l2697_269705

/-- The cost of paving a rectangular floor -/
theorem paving_cost_theorem (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 300) :
  length * width * rate = 6187.5 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_theorem_l2697_269705


namespace NUMINAMATH_CALUDE_fathers_age_triple_weiweis_l2697_269717

/-- Proves that after 5 years, the father's age will be three times Weiwei's age -/
theorem fathers_age_triple_weiweis (weiwei_age : ℕ) (father_age : ℕ) 
  (h1 : weiwei_age = 8) (h2 : father_age = 34) : 
  ∃ (years : ℕ), years = 5 ∧ father_age + years = 3 * (weiwei_age + years) :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_triple_weiweis_l2697_269717


namespace NUMINAMATH_CALUDE_sum_xyz_equals_negative_one_l2697_269777

theorem sum_xyz_equals_negative_one (x y z : ℝ) : 
  (x + 1)^2 + |y - 2| = -(2*x - z)^2 → x + y + z = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_negative_one_l2697_269777


namespace NUMINAMATH_CALUDE_bankruptcy_division_l2697_269749

/-- Represents the weight of an item -/
structure Weight (α : Type) where
  value : ℕ

/-- Represents the collection of items -/
structure Inventory where
  horns : ℕ
  hooves : ℕ
  weight : ℕ

/-- Represents a person's share of the inventory -/
structure Share where
  horns : ℕ
  hooves : ℕ
  hasWeight : Bool

def totalWeight (inv : Inventory) (w : Weight ℕ) (δ : ℕ) : ℕ :=
  inv.horns * (w.value + δ) + inv.hooves * w.value + inv.weight * (w.value + 2 * δ)

def shareWeight (s : Share) (w : Weight ℕ) (δ : ℕ) : ℕ :=
  s.horns * (w.value + δ) + s.hooves * w.value + (if s.hasWeight then w.value + 2 * δ else 0)

theorem bankruptcy_division (w : Weight ℕ) (δ : ℕ) :
  ∃ (panikovsky balaganov : Share),
    panikovsky.horns + balaganov.horns = 17 ∧
    panikovsky.hooves + balaganov.hooves = 2 ∧
    panikovsky.hasWeight = false ∧
    balaganov.hasWeight = true ∧
    shareWeight panikovsky w δ = shareWeight balaganov w δ ∧
    (panikovsky.horns = 9 ∧ panikovsky.hooves = 2 ∧
     balaganov.horns = 8 ∧ balaganov.hooves = 0) :=
  sorry

end NUMINAMATH_CALUDE_bankruptcy_division_l2697_269749


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l2697_269740

theorem jungkook_balls_count (num_boxes : ℕ) (balls_per_box : ℕ) : 
  num_boxes = 2 → balls_per_box = 3 → num_boxes * balls_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l2697_269740


namespace NUMINAMATH_CALUDE_intersection_condition_l2697_269723

/-- The set A on a 2D plane -/
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- The set B on a 2D plane, parameterized by r -/
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

/-- Theorem stating the conditions for r when A and B intersect at exactly one point -/
theorem intersection_condition (r : ℝ) (h1 : r > 0) 
  (h2 : ∃! p, p ∈ A ∩ B r) : r = 3 ∨ r = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2697_269723


namespace NUMINAMATH_CALUDE_meat_market_sales_l2697_269768

theorem meat_market_sales (thursday_sales : ℝ) : 
  (2 * thursday_sales) + thursday_sales + 130 + (130 / 2) = 500 + 325 → 
  thursday_sales = 210 := by
sorry

end NUMINAMATH_CALUDE_meat_market_sales_l2697_269768


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2697_269712

-- Define the coefficients of a quadratic equation ax^2 + bx + c = 0
def QuadraticCoefficients (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - x + 3 = 0

-- Theorem statement
theorem quadratic_equation_coefficients :
  QuadraticCoefficients 1 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2697_269712


namespace NUMINAMATH_CALUDE_pencil_distribution_l2697_269799

theorem pencil_distribution (num_children : ℕ) (pencils_per_child : ℕ) (total_pencils : ℕ) : 
  num_children = 4 → 
  pencils_per_child = 2 → 
  total_pencils = num_children * pencils_per_child →
  total_pencils = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2697_269799


namespace NUMINAMATH_CALUDE_circle_area_increase_l2697_269795

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_r := 2 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  (new_area - original_area) / original_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2697_269795


namespace NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_point_l2697_269751

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (25/29, -57/29)

/-- First line equation: 6x - 5y = 15 -/
def line1 (x y : ℚ) : Prop := 6*x - 5*y = 15

/-- Second line equation: 8x + 3y = 1 -/
def line2 (x y : ℚ) : Prop := 8*x + 3*y = 1

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_lines : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 :=
sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point (x y : ℚ) :
  line1 x y ∧ line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_point_l2697_269751


namespace NUMINAMATH_CALUDE_function_determination_l2697_269736

-- Define a first-degree function
def first_degree_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- State the theorem
theorem function_determination (f : ℝ → ℝ) 
  (h1 : first_degree_function f)
  (h2 : 2 * f 2 - 3 * f 1 = 5)
  (h3 : 2 * f 0 - f (-1) = 1) :
  ∀ x, f x = 3 * x - 2 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l2697_269736


namespace NUMINAMATH_CALUDE_max_value_problem_l2697_269794

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 12 ∧
  ∃ (a b c : ℝ), (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 12 ∧
                  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l2697_269794


namespace NUMINAMATH_CALUDE_infinite_integers_with_noncounting_divisors_l2697_269701

theorem infinite_integers_with_noncounting_divisors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a ∈ S, a ≥ 1 ∧ 
      (∀ n : ℕ, n ≥ 1 → ¬(Nat.card (Nat.divisors a) = n))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_integers_with_noncounting_divisors_l2697_269701


namespace NUMINAMATH_CALUDE_x_intercepts_count_l2697_269754

theorem x_intercepts_count : ∃! (s : Finset ℝ), 
  (∀ x ∈ s, (x - 3) * (x^2 + 4*x + 3) = 0) ∧ 
  s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l2697_269754


namespace NUMINAMATH_CALUDE_trig_identity_l2697_269785

theorem trig_identity (α : Real) (h : Real.cos (α - π/3) = -1/2) : 
  Real.sin (π/6 + α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2697_269785


namespace NUMINAMATH_CALUDE_total_money_calculation_l2697_269796

/-- The total amount of money Sam, Billy, and Lila have together -/
def total_money (sam_money : ℚ) : ℚ :=
  let billy_money := 4.5 * sam_money - 345.25
  let lila_money := 2.25 * (billy_money - sam_money)
  sam_money + billy_money + lila_money

/-- Theorem stating the total amount of money when Sam has $750.50 -/
theorem total_money_calculation :
  total_money 750.50 = 8915.88 := by
  sorry

end NUMINAMATH_CALUDE_total_money_calculation_l2697_269796


namespace NUMINAMATH_CALUDE_vector_equality_l2697_269772

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

theorem vector_equality : c = (1/2 : ℝ) • a - (3/2 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l2697_269772


namespace NUMINAMATH_CALUDE_sum_increases_l2697_269721

/-- A set of 30 distinct positive real numbers -/
def M : Finset ℝ :=
  sorry

/-- The sum of the first n elements of M -/
def A (n : ℕ) : ℝ :=
  sorry

theorem sum_increases (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 29) : A (n + 1) > A n := by
  sorry

end NUMINAMATH_CALUDE_sum_increases_l2697_269721


namespace NUMINAMATH_CALUDE_log_inequality_for_negative_reals_l2697_269713

theorem log_inequality_for_negative_reals (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  Real.log (-a) > Real.log (-b) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_for_negative_reals_l2697_269713


namespace NUMINAMATH_CALUDE_xyz_value_l2697_269739

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 10 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l2697_269739


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2697_269763

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- Theorem stating the range of a for which f(x) is non-negative for all x -/
theorem f_nonnegative_iff_a_in_range :
  (∀ x : ℝ, f a x ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2697_269763


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l2697_269766

theorem gcd_of_polynomial_and_b (b : ℤ) (h : 792 ∣ b) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 99) b = 99 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l2697_269766


namespace NUMINAMATH_CALUDE_convex_polyhedron_volume_relation_l2697_269730

/-- A convex polyhedron with an inscribed sphere -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  inscribedSphereRadius : ℝ

/-- The relationship between volume, surface area, and inscribed sphere radius for a convex polyhedron -/
theorem convex_polyhedron_volume_relation (P : ConvexPolyhedron) :
  P.volume = (1 / 3) * P.surfaceArea * P.inscribedSphereRadius := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_volume_relation_l2697_269730


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2697_269748

theorem solve_linear_equation (x : ℝ) : 3 * x - 5 = 4 * x + 10 → x = -15 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2697_269748


namespace NUMINAMATH_CALUDE_school_ratio_change_l2697_269707

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student population -/
structure School where
  boarders : ℕ
  day_students : ℕ

def initial_ratio : Ratio := { boarders := 5, day_students := 12 }

def initial_school : School := { boarders := 150, day_students := 360 }

def new_boarders : ℕ := 30

def final_school : School := { 
  boarders := initial_school.boarders + new_boarders,
  day_students := initial_school.day_students
}

def final_ratio : Ratio := { 
  boarders := 1,
  day_students := 2
}

theorem school_ratio_change :
  (initial_ratio.boarders * initial_school.day_students = initial_ratio.day_students * initial_school.boarders) ∧
  (final_school.boarders = initial_school.boarders + new_boarders) ∧
  (final_school.day_students = initial_school.day_students) →
  (final_ratio.boarders * final_school.day_students = final_ratio.day_students * final_school.boarders) :=
by sorry

end NUMINAMATH_CALUDE_school_ratio_change_l2697_269707


namespace NUMINAMATH_CALUDE_sas_congruence_l2697_269771

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- SAS Congruence Theorem
theorem sas_congruence (t1 t2 : Triangle) 
  (h1 : t1.a = t2.a)  -- First side equal
  (h2 : t1.b = t2.b)  -- Second side equal
  (h3 : t1.α = t2.α)  -- Included angle equal
  : congruent t1 t2 :=
by
  sorry


end NUMINAMATH_CALUDE_sas_congruence_l2697_269771


namespace NUMINAMATH_CALUDE_special_function_property_l2697_269755

/-- A function f: ℤ → ℤ satisfying specific properties -/
def special_function (f : ℤ → ℤ) : Prop :=
  f 0 = 2 ∧ ∀ x : ℤ, f (x + 1) + f (x - 1) = f x * f 1

/-- Theorem stating the property to be proved for the special function -/
theorem special_function_property (f : ℤ → ℤ) (h : special_function f) :
  ∀ x y : ℤ, f (x + y) + f (x - y) = f x * f y := by
  sorry


end NUMINAMATH_CALUDE_special_function_property_l2697_269755


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2697_269797

theorem solve_exponential_equation :
  ∀ n : ℝ, (2^16 : ℝ) * (25^n) = 5 * (10^16) → n = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2697_269797


namespace NUMINAMATH_CALUDE_expression_simplification_l2697_269762

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 3) :
  (a - 3) / (a^2 + 6*a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2697_269762


namespace NUMINAMATH_CALUDE_eleventh_term_is_25_l2697_269757

/-- An arithmetic sequence with a given sum of first seven terms and first term -/
structure ArithmeticSequence where
  sum_seven : ℝ  -- Sum of first seven terms
  first_term : ℝ  -- First term
  nth_term : ℕ → ℝ  -- Function to calculate the nth term

/-- The eleventh term of the arithmetic sequence is 25 -/
theorem eleventh_term_is_25 (seq : ArithmeticSequence)
    (h1 : seq.sum_seven = 77)
    (h2 : seq.first_term = 5) :
    seq.nth_term 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_25_l2697_269757


namespace NUMINAMATH_CALUDE_divisibility_condition_l2697_269759

theorem divisibility_condition (k n : ℕ+) :
  (∃ (p : ℕ), Prime p ∧ p ∣ (4 * k ^ 2 - 1) ^ 2 ∧ p = 8 * k * n - 1) ↔ Even k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2697_269759


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l2697_269741

/-- The coefficient of x^2 in the expansion of (1+x)+(1+x)^2+(1+x)^3+...+(1+x)^9 -/
def coefficient_x_squared : ℕ :=
  (Finset.range 9).sum (λ n => Nat.choose (n + 1) 2)

/-- The theorem stating that the coefficient of x^2 in the expansion is 120 -/
theorem coefficient_x_squared_is_120 : coefficient_x_squared = 120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l2697_269741


namespace NUMINAMATH_CALUDE_cos_2017_deg_l2697_269790

theorem cos_2017_deg : Real.cos (2017 * π / 180) = -Real.cos (37 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_2017_deg_l2697_269790


namespace NUMINAMATH_CALUDE_gcf_75_100_l2697_269718

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_100_l2697_269718


namespace NUMINAMATH_CALUDE_paving_cost_l2697_269735

/-- The cost of paving a rectangular floor given its dimensions and the paving rate. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 800) :
  length * width * rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2697_269735


namespace NUMINAMATH_CALUDE_population_growth_duration_l2697_269732

/-- Proves that given specific birth and death rates and a total net increase,
    the duration of the period is 12 hours. -/
theorem population_growth_duration
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (total_net_increase : ℝ)
  (h1 : birth_rate = 2)
  (h2 : death_rate = 1)
  (h3 : total_net_increase = 86400)
  : (total_net_increase / (birth_rate - death_rate)) / 3600 = 12 := by
  sorry


end NUMINAMATH_CALUDE_population_growth_duration_l2697_269732


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_15_6_minus_9_6_l2697_269774

-- Define the valuation function v₂
def v₂ (n : ℤ) : ℕ := (n.natAbs.factors.count 2)

-- State the theorem
theorem largest_power_of_two_dividing_15_6_minus_9_6 :
  2^(v₂ (15^6 - 9^6)) = 32 := by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_15_6_minus_9_6_l2697_269774


namespace NUMINAMATH_CALUDE_function_periodic_l2697_269746

/-- A function satisfying the given functional equation is periodic -/
theorem function_periodic (f : ℝ → ℝ) (a : ℝ) (ha : a > 0)
  (h : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodic_l2697_269746


namespace NUMINAMATH_CALUDE_function_positivity_and_inequality_l2697_269788

/-- The function f(x) = mx² + mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + m * x + 3

/-- The function g(x) = (3m-1)x + 5 -/
def g (m : ℝ) (x : ℝ) : ℝ := (3*m - 1) * x + 5

theorem function_positivity_and_inequality (m : ℝ) :
  (∀ x : ℝ, f m x > 0) ↔ (0 ≤ m ∧ m < 12) ∧
  (∀ x : ℝ, f m x > g m x ↔
    (m < -1/2 ∧ -1/m < x ∧ x < 2) ∨
    (m = -1/2 ∧ False) ∨
    (-1/2 < m ∧ m < 0 ∧ 2 < x ∧ x < -1/m) ∨
    (m = 0 ∧ x > 2) ∨
    (m > 0 ∧ (x < -1/m ∨ x > 2))) :=
sorry

end NUMINAMATH_CALUDE_function_positivity_and_inequality_l2697_269788


namespace NUMINAMATH_CALUDE_probability_different_suits_l2697_269700

def deck_size : ℕ := 60
def num_suits : ℕ := 5
def cards_per_suit : ℕ := 12

theorem probability_different_suits :
  let remaining_cards : ℕ := deck_size - 1
  let cards_not_same_suit : ℕ := remaining_cards - (cards_per_suit - 1)
  (cards_not_same_suit : ℚ) / remaining_cards = 48 / 59 :=
by sorry

end NUMINAMATH_CALUDE_probability_different_suits_l2697_269700


namespace NUMINAMATH_CALUDE_last_two_digits_of_root_sum_power_l2697_269704

theorem last_two_digits_of_root_sum_power : 
  ∃ n : ℤ, (n : ℝ) = (Real.sqrt 29 + Real.sqrt 21)^1984 ∧ n % 100 = 71 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_root_sum_power_l2697_269704


namespace NUMINAMATH_CALUDE_total_oranges_picked_l2697_269738

/-- Represents the number of oranges picked by Jeremy on Monday -/
def monday_pick : ℕ := 100

/-- Represents the number of oranges picked by Jeremy and his brother on Tuesday -/
def tuesday_pick : ℕ := 3 * monday_pick

/-- Represents the number of oranges picked by all three on Wednesday -/
def wednesday_pick : ℕ := 2 * tuesday_pick

/-- Represents the number of oranges picked by the cousin on Wednesday -/
def cousin_wednesday_pick : ℕ := tuesday_pick - (tuesday_pick / 5)

/-- Represents the number of oranges picked by Jeremy on Thursday -/
def jeremy_thursday_pick : ℕ := (7 * monday_pick) / 10

/-- Represents the number of oranges picked by the brother on Thursday -/
def brother_thursday_pick : ℕ := tuesday_pick - monday_pick

/-- Represents the number of oranges picked by the cousin on Thursday -/
def cousin_thursday_pick : ℕ := cousin_wednesday_pick + (3 * cousin_wednesday_pick) / 10

/-- Represents the total number of oranges picked over the four days -/
def total_picked : ℕ := monday_pick + tuesday_pick + wednesday_pick + 
  (jeremy_thursday_pick + brother_thursday_pick + cousin_thursday_pick)

theorem total_oranges_picked : total_picked = 1642 := by sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l2697_269738


namespace NUMINAMATH_CALUDE_product_17_reciprocal_squares_sum_l2697_269782

theorem product_17_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 17 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 290 / 289 := by
  sorry

end NUMINAMATH_CALUDE_product_17_reciprocal_squares_sum_l2697_269782


namespace NUMINAMATH_CALUDE_train_length_l2697_269770

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 3 → ∃ (length_m : ℝ), abs (length_m - 50.01) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2697_269770


namespace NUMINAMATH_CALUDE_negative_abs_neg_three_gt_negative_pi_l2697_269765

theorem negative_abs_neg_three_gt_negative_pi : -|-3| > -π := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_neg_three_gt_negative_pi_l2697_269765


namespace NUMINAMATH_CALUDE_min_additional_votes_to_win_l2697_269711

/-- Represents the number of candidates in the election -/
def num_candidates : ℕ := 5

/-- Represents the percentage of votes received by candidate A -/
def votes_a_percent : ℚ := 35 / 100

/-- Represents the percentage of votes received by candidate B -/
def votes_b_percent : ℚ := 20 / 100

/-- Represents the percentage of votes received by candidate C -/
def votes_c_percent : ℚ := 15 / 100

/-- Represents the percentage of votes received by candidate D -/
def votes_d_percent : ℚ := 10 / 100

/-- Represents the difference in votes between candidate A and B -/
def votes_difference : ℕ := 1200

/-- Represents the minimum percentage of votes needed to win -/
def win_percentage : ℚ := 36 / 100

/-- Theorem stating the minimum additional votes needed for candidate A to win -/
theorem min_additional_votes_to_win :
  ∃ (total_votes : ℕ) (votes_a : ℕ) (votes_needed : ℕ),
    (votes_a_percent : ℚ) * total_votes = votes_a ∧
    (votes_b_percent : ℚ) * total_votes = votes_a - votes_difference ∧
    (win_percentage : ℚ) * total_votes = votes_needed ∧
    votes_needed - votes_a = 80 :=
sorry

end NUMINAMATH_CALUDE_min_additional_votes_to_win_l2697_269711


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_5_l2697_269733

theorem smallest_four_digit_mod_5 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 5 = 4 → n ≥ 1004 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_5_l2697_269733


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2697_269708

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-π/4) (π/4)) 
  (hy : y ∈ Set.Icc (-π/4) (π/4)) 
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2697_269708
