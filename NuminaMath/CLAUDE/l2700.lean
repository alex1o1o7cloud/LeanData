import Mathlib

namespace NUMINAMATH_CALUDE_y_variation_l2700_270094

/-- A function that varies directly as x and inversely as the square of z -/
def y (x z : ℝ) : ℝ := sorry

theorem y_variation (k : ℝ) :
  (∀ x z, y x z = k * x / (z^2)) →
  y 5 1 = 10 →
  y (-10) 2 = -5 := by sorry

end NUMINAMATH_CALUDE_y_variation_l2700_270094


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2700_270069

theorem inequality_and_equality_condition (x y : ℝ) :
  (x^2 + 1) * (y^2 + 1) + 4 * (x - 1) * (y - 1) ≥ 0 ∧
  ((x^2 + 1) * (y^2 + 1) + 4 * (x - 1) * (y - 1) = 0 ↔
    ((x = 1 - Real.sqrt 2 ∧ y = 1 + Real.sqrt 2) ∨
     (x = 1 + Real.sqrt 2 ∧ y = 1 - Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2700_270069


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_1181_over_20_l2700_270051

theorem sqrt_fraction_sum_equals_sqrt_1181_over_20 :
  Real.sqrt (16/25 + 9/4 + 1/16) = Real.sqrt 1181 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_1181_over_20_l2700_270051


namespace NUMINAMATH_CALUDE_weight_of_B_l2700_270098

/-- Given three weights A, B, and C, prove that B equals 31 when:
    - The average of A, B, and C is 45
    - The average of A and B is 40
    - The average of B and C is 43 -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : 
  B = 31 := by
  sorry

#check weight_of_B

end NUMINAMATH_CALUDE_weight_of_B_l2700_270098


namespace NUMINAMATH_CALUDE_tomato_multiple_l2700_270076

theorem tomato_multiple : 
  ∀ (before_vacation after_growth : ℕ),
    before_vacation = 36 →
    after_growth = 3564 →
    (before_vacation + after_growth) / before_vacation = 100 := by
  sorry

end NUMINAMATH_CALUDE_tomato_multiple_l2700_270076


namespace NUMINAMATH_CALUDE_peach_difference_l2700_270070

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 5 →
  steven_peaches = jill_peaches + 18 →
  jake_peaches = 17 →
  steven_peaches - jake_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l2700_270070


namespace NUMINAMATH_CALUDE_seeds_in_big_garden_l2700_270035

/-- Given Nancy's gardening scenario, prove the number of seeds in the big garden. -/
theorem seeds_in_big_garden 
  (total_seeds : ℕ) 
  (num_small_gardens : ℕ) 
  (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : num_small_gardens = 6)
  (h3 : seeds_per_small_garden = 4) : 
  total_seeds - (num_small_gardens * seeds_per_small_garden) = 28 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_big_garden_l2700_270035


namespace NUMINAMATH_CALUDE_min_value_theorem_l2700_270092

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x + x^2

noncomputable def h (c b : ℝ) (x : ℝ) : ℝ := Real.log x - c * x^2 - b * x

theorem min_value_theorem (m c b : ℝ) (h_m : m ≥ 3 * Real.sqrt 2 / 2) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  (∀ x, x > 0 → g m x ≤ g m x₁ ∧ g m x ≤ g m x₂) ∧
  h c b x₁ = 0 ∧ h c b x₂ = 0 ∧
  (∀ y, y = (x₁ - x₂) * h c b ((x₁ + x₂) / 2) → y ≥ -2/3 + Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2700_270092


namespace NUMINAMATH_CALUDE_equation_solution_l2700_270074

theorem equation_solution (x y z k : ℝ) :
  (5 / (x - z) = k / (y + z)) ∧ (k / (y + z) = 12 / (x + y)) → k = 17 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2700_270074


namespace NUMINAMATH_CALUDE_deepak_age_l2700_270033

theorem deepak_age (rahul deepak rohan : ℕ) : 
  rahul = 5 * (rahul / 5) → 
  deepak = 2 * (rahul / 5) → 
  rohan = 3 * (rahul / 5) → 
  rahul + 8 = 28 → 
  deepak = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2700_270033


namespace NUMINAMATH_CALUDE_expression_evaluation_l2700_270000

theorem expression_evaluation :
  let f (x : ℝ) := -7*x + 2*(x^2 - 1) - (2*x^2 - x + 3)
  f 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2700_270000


namespace NUMINAMATH_CALUDE_double_vinegar_theorem_l2700_270054

/-- Represents the ratio of oil to vinegar in a salad dressing -/
structure SaladDressing where
  oil : ℚ
  vinegar : ℚ

/-- The initial ratio of oil to vinegar -/
def initial_ratio : SaladDressing :=
  { oil := 3, vinegar := 1 }

/-- Doubles the amount of vinegar in a salad dressing -/
def double_vinegar (sd : SaladDressing) : SaladDressing :=
  { oil := sd.oil, vinegar := 2 * sd.vinegar }

/-- Calculates the ratio of oil to vinegar -/
def ratio (sd : SaladDressing) : ℚ :=
  sd.oil / sd.vinegar

/-- Theorem: Doubling the vinegar in the initial 3:1 ratio results in a 3:2 ratio -/
theorem double_vinegar_theorem :
  ratio (double_vinegar initial_ratio) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_vinegar_theorem_l2700_270054


namespace NUMINAMATH_CALUDE_additional_people_needed_l2700_270011

/-- Represents the number of person-hours required to mow a lawn -/
def lawn_work : ℕ := 24

/-- The number of people who can mow the lawn in 3 hours -/
def initial_people : ℕ := 8

/-- The initial time taken to mow the lawn -/
def initial_time : ℕ := 3

/-- The desired time to mow the lawn -/
def target_time : ℕ := 2

theorem additional_people_needed : 
  ∃ (additional : ℕ), 
    initial_people * initial_time = lawn_work ∧
    (initial_people + additional) * target_time = lawn_work ∧
    additional = 4 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_needed_l2700_270011


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2700_270018

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f (y * f x - 1) = x^2 * f y - f x) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2700_270018


namespace NUMINAMATH_CALUDE_zou_mei_competition_l2700_270041

theorem zou_mei_competition (n : ℕ) : 
  n^2 + 15 + 18 = (n + 1)^2 → n^2 + 15 = 271 := by
  sorry

end NUMINAMATH_CALUDE_zou_mei_competition_l2700_270041


namespace NUMINAMATH_CALUDE_process_terminates_with_bound_bound_is_tight_l2700_270039

/-- Represents the state of the queue -/
structure QueueState where
  n : ℕ
  positions : Fin n → Fin n

/-- Represents a single move in the process -/
structure Move where
  i : ℕ

/-- The result of applying a move to a queue state -/
inductive MoveResult
  | Continue (new_state : QueueState) (euros_paid : ℕ)
  | End

/-- Applies a move to a queue state -/
def apply_move (state : QueueState) (move : Move) : MoveResult :=
  sorry

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def apply_moves (initial_state : QueueState) (moves : MoveSequence) : ℕ :=
  sorry

theorem process_terminates_with_bound (n : ℕ) :
  ∀ (initial_state : QueueState),
  ∀ (moves : MoveSequence),
  apply_moves initial_state moves ≤ 2^n - n - 1 :=
sorry

theorem bound_is_tight (n : ℕ) :
  ∃ (initial_state : QueueState),
  ∃ (moves : MoveSequence),
  apply_moves initial_state moves = 2^n - n - 1 :=
sorry

end NUMINAMATH_CALUDE_process_terminates_with_bound_bound_is_tight_l2700_270039


namespace NUMINAMATH_CALUDE_curve_symmetric_about_y_axis_l2700_270046

theorem curve_symmetric_about_y_axis : ∀ x y : ℝ, x^2 - y^2 = 1 ↔ (-x)^2 - y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_y_axis_l2700_270046


namespace NUMINAMATH_CALUDE_clock_sum_after_duration_l2700_270052

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the result on a 12-hour clock -/
def addTime (initial : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

theorem clock_sum_after_duration (initialTime finalTime : Time) 
  (h : finalTime = addTime ⟨15, 0, 0⟩ 317 15 30) : 
  finalTime.hours + finalTime.minutes + finalTime.seconds = 53 := by
  sorry

end NUMINAMATH_CALUDE_clock_sum_after_duration_l2700_270052


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_two_l2700_270085

theorem quadratic_roots_greater_than_two (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-1)*x + 4 - m = 0 → x > 2) ↔ -6 < m ∧ m ≤ -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_two_l2700_270085


namespace NUMINAMATH_CALUDE_irrigation_system_fluxes_l2700_270068

-- Define the irrigation system
structure IrrigationSystem where
  channels : Set Char
  nodes : Set Char
  flux : Char → Char → ℝ
  water_entry : Char
  water_exit : Char

-- Define the properties of the irrigation system
def is_valid_system (sys : IrrigationSystem) : Prop :=
  -- Channels and nodes
  sys.channels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'} ∧
  sys.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'} ∧
  -- Water entry and exit points
  sys.water_entry = 'A' ∧
  sys.water_exit = 'E' ∧
  -- Flux conservation property
  ∀ x y z : Char, x ∈ sys.nodes ∧ y ∈ sys.nodes ∧ z ∈ sys.nodes →
    sys.flux x y + sys.flux y z = sys.flux x z

-- Theorem statement
theorem irrigation_system_fluxes (sys : IrrigationSystem) 
  (h_valid : is_valid_system sys) 
  (h_flux_BC : sys.flux 'B' 'C' = q₀) :
  sys.flux 'A' 'B' = 2 * q₀ ∧
  sys.flux 'A' 'H' = 3/2 * q₀ ∧
  sys.flux 'A' 'B' + sys.flux 'A' 'H' = 7/2 * q₀ :=
by sorry

end NUMINAMATH_CALUDE_irrigation_system_fluxes_l2700_270068


namespace NUMINAMATH_CALUDE_equal_domain_interval_exponential_l2700_270096

/-- Definition of an "equal domain interval" for a function --/
def is_equal_domain_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- Theorem: If [1,2] is an "equal domain interval" for f(x) = a⋅2^x + b (a > 0), then a = 1/2 and b = 0 --/
theorem equal_domain_interval_exponential (a b : ℝ) (h : a > 0) :
  is_equal_domain_interval (λ x => a * 2^x + b) 1 2 → a = 1/2 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_equal_domain_interval_exponential_l2700_270096


namespace NUMINAMATH_CALUDE_digits_after_decimal_is_six_l2700_270001

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 8^2) is expressed as a decimal -/
def digits_after_decimal : ℕ :=
  let fraction := (5^7 : ℚ) / ((10^5 * 8^2) : ℚ)
  6

theorem digits_after_decimal_is_six :
  digits_after_decimal = 6 := by sorry

end NUMINAMATH_CALUDE_digits_after_decimal_is_six_l2700_270001


namespace NUMINAMATH_CALUDE_cannot_finish_third_l2700_270002

/-- Represents the relative positions of runners in a race -/
def BeatsIn (runners : Type) := runners → runners → Prop

/-- A race with 5 runners and their relative positions -/
structure Race (runners : Type) :=
  (beats : BeatsIn runners)
  (P Q R S T : runners)
  (p_beats_q : beats P Q)
  (p_beats_r : beats P R)
  (p_beats_s : beats P S)
  (q_beats_s : beats Q S)
  (s_beats_r : beats S R)
  (t_after_p : beats P T)
  (t_before_q : beats T Q)

/-- Represents the finishing position of a runner -/
def FinishPosition (runners : Type) := runners → ℕ

theorem cannot_finish_third (runners : Type) (race : Race runners) 
  (finish : FinishPosition runners) :
  (finish race.P ≠ 3) ∧ (finish race.R ≠ 3) := by sorry

end NUMINAMATH_CALUDE_cannot_finish_third_l2700_270002


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2700_270015

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → ∃ s t : ℝ, s + t = 15 ∧ s*t = 6 ∧ s^2 + t^2 = 213 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2700_270015


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l2700_270095

def lunch_cost : ℝ := 60.80
def total_spent : ℝ := 72.96

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l2700_270095


namespace NUMINAMATH_CALUDE_tan_30_plus_2cos_30_l2700_270004

theorem tan_30_plus_2cos_30 : Real.tan (π / 6) + 2 * Real.cos (π / 6) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_2cos_30_l2700_270004


namespace NUMINAMATH_CALUDE_fundraiser_problem_l2700_270024

/-- Fundraiser Problem -/
theorem fundraiser_problem 
  (total_promised : ℕ)
  (amount_received : ℕ)
  (sally_owed : ℕ)
  (carl_owed : ℕ)
  (h_total : total_promised = 400)
  (h_received : amount_received = 285)
  (h_sally : sally_owed = 35)
  (h_carl : carl_owed = 35)
  : ∃ (amy_owed : ℕ) (derek_owed : ℕ),
    amy_owed = 30 ∧ 
    derek_owed = amy_owed / 2 ∧
    total_promised = amount_received + sally_owed + carl_owed + amy_owed + derek_owed :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_problem_l2700_270024


namespace NUMINAMATH_CALUDE_salt_solution_mixture_salt_solution_mixture_proof_l2700_270034

/-- Proves that adding 50 ounces of 10% salt solution to 50 ounces of 40% salt solution results in a 25% salt solution -/
theorem salt_solution_mixture : ℝ → Prop :=
  λ x : ℝ =>
    let initial_volume : ℝ := 50
    let initial_concentration : ℝ := 0.4
    let added_concentration : ℝ := 0.1
    let final_concentration : ℝ := 0.25
    let final_volume : ℝ := initial_volume + x
    let final_salt : ℝ := initial_volume * initial_concentration + x * added_concentration
    (x = 50) →
    (final_salt / final_volume = final_concentration)

/-- The proof of the theorem -/
theorem salt_solution_mixture_proof : salt_solution_mixture 50 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_salt_solution_mixture_proof_l2700_270034


namespace NUMINAMATH_CALUDE_product_of_numbers_l2700_270090

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : x * y = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2700_270090


namespace NUMINAMATH_CALUDE_projection_theorem_l2700_270009

/-- Given two 2D vectors a and b, and a third vector c that satisfies a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem projection_theorem (a b c : ℝ × ℝ) : 
  a = (2, 3) → 
  b = (-4, 7) → 
  a + c = (0, 0) → 
  let proj_c_onto_b := (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj_c_onto_b = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l2700_270009


namespace NUMINAMATH_CALUDE_sum_special_integers_sum_special_integers_proof_l2700_270064

theorem sum_special_integers : ℕ → ℤ → ℤ → Prop :=
  fun a b c =>
    (∀ n : ℕ, a ≤ n) →  -- a is the smallest natural number
    (0 < b ∧ ∀ m : ℤ, 0 < m → b ≤ m) →  -- b is the smallest positive integer
    (c < 0 ∧ ∀ k : ℤ, k < 0 → k ≤ c) →  -- c is the largest negative integer
    a + b + c = 0

-- The proof of this theorem is omitted
theorem sum_special_integers_proof : ∃ a : ℕ, ∃ b c : ℤ, sum_special_integers a b c :=
  sorry

end NUMINAMATH_CALUDE_sum_special_integers_sum_special_integers_proof_l2700_270064


namespace NUMINAMATH_CALUDE_card_value_proof_l2700_270008

/-- Given four cards W, X, Y, Z with certain conditions, prove Y is tagged with 300 --/
theorem card_value_proof (W X Y Z : ℕ) : 
  W = 200 →
  X = W / 2 →
  Z = 400 →
  W + X + Y + Z = 1000 →
  Y = 300 := by
  sorry

end NUMINAMATH_CALUDE_card_value_proof_l2700_270008


namespace NUMINAMATH_CALUDE_horner_method_first_step_l2700_270025

def f (x : ℝ) : ℝ := 7 * x^6 + 6 * x^5 + 3 * x^2 + 2

def horner_first_step (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_method_first_step :
  horner_first_step 7 6 4 = 34 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_first_step_l2700_270025


namespace NUMINAMATH_CALUDE_certain_value_proof_l2700_270067

theorem certain_value_proof (number : ℤ) (certain_value : ℤ) 
  (h1 : number = 109)
  (h2 : 150 - number = number + certain_value) : 
  certain_value = -68 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l2700_270067


namespace NUMINAMATH_CALUDE_jackie_additional_amount_l2700_270060

/-- The amount required for free shipping -/
def free_shipping_threshold : ℝ := 50

/-- The cost of a bottle of shampoo -/
def shampoo_cost : ℝ := 10

/-- The cost of a bottle of conditioner -/
def conditioner_cost : ℝ := 10

/-- The cost of a bottle of lotion -/
def lotion_cost : ℝ := 6

/-- The number of shampoo bottles Jackie ordered -/
def shampoo_quantity : ℕ := 1

/-- The number of conditioner bottles Jackie ordered -/
def conditioner_quantity : ℕ := 1

/-- The number of lotion bottles Jackie ordered -/
def lotion_quantity : ℕ := 3

/-- The additional amount Jackie needs to spend for free shipping -/
def additional_amount_needed : ℝ :=
  free_shipping_threshold - (shampoo_cost * shampoo_quantity + conditioner_cost * conditioner_quantity + lotion_cost * lotion_quantity)

theorem jackie_additional_amount : additional_amount_needed = 12 := by
  sorry

end NUMINAMATH_CALUDE_jackie_additional_amount_l2700_270060


namespace NUMINAMATH_CALUDE_company_salary_change_l2700_270056

theorem company_salary_change 
  (E : ℕ) -- Original number of employees
  (S : ℝ) -- Original average salary
  (new_E : ℕ) -- New number of employees
  (new_S : ℝ) -- New average salary
  (h1 : new_E = (E * 4) / 5) -- 20% decrease in employees
  (h2 : new_S = S * 1.15) -- 15% increase in average salary
  : (new_E : ℝ) * new_S = 0.92 * ((E : ℝ) * S) :=
by sorry

end NUMINAMATH_CALUDE_company_salary_change_l2700_270056


namespace NUMINAMATH_CALUDE_outdoor_players_count_l2700_270055

/-- Represents the number of players in different categories -/
structure PlayerCounts where
  total : ℕ
  indoor : ℕ
  both : ℕ
  outdoor : ℕ

/-- Theorem stating the number of outdoor players given the conditions -/
theorem outdoor_players_count (p : PlayerCounts)
  (h_total : p.total = 400)
  (h_indoor : p.indoor = 110)
  (h_both : p.both = 60)
  (h_valid : p.total ≥ p.indoor + p.outdoor - p.both) :
  p.outdoor = 350 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_players_count_l2700_270055


namespace NUMINAMATH_CALUDE_function_inequality_l2700_270044

open Real

/-- Given a function f: ℝ → ℝ with derivative f', 
    if x · f'(x) + f(x) < 0 for all x, then 2 · f(2) > 3 · f(3) -/
theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
    (h : ∀ x, HasDerivAt f (f' x) x)
    (h' : ∀ x, x * f' x + f x < 0) :
    2 * f 2 > 3 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2700_270044


namespace NUMINAMATH_CALUDE_inequality_proof_l2700_270077

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_proof (a b : ℝ) (h : ∀ x, f a b x ≥ 0) : b * (a + 1) / 2 < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2700_270077


namespace NUMINAMATH_CALUDE_davids_age_twice_daughters_l2700_270093

/-- 
Given:
- David is currently 40 years old
- David's daughter is currently 12 years old

Prove that in 16 years, David's age will be twice his daughter's age.
-/
theorem davids_age_twice_daughters (david_age : ℕ) (daughter_age : ℕ) (years_passed : ℕ) : 
  david_age = 40 → daughter_age = 12 → years_passed = 16 → 
  david_age + years_passed = 2 * (daughter_age + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_davids_age_twice_daughters_l2700_270093


namespace NUMINAMATH_CALUDE_total_wheels_in_lot_l2700_270050

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 17

/-- Theorem: The total number of car wheels in the parking lot is 68 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_lot_l2700_270050


namespace NUMINAMATH_CALUDE_can_collection_ratio_l2700_270040

theorem can_collection_ratio : 
  ∀ (total LaDonna Yoki Prikya : ℕ),
    total = 85 →
    LaDonna = 25 →
    Yoki = 10 →
    Prikya = total - LaDonna - Yoki →
    (Prikya : ℚ) / LaDonna = 2 := by
  sorry

end NUMINAMATH_CALUDE_can_collection_ratio_l2700_270040


namespace NUMINAMATH_CALUDE_car_travel_time_difference_l2700_270079

theorem car_travel_time_difference : 
  let distance : ℝ := 4.333329
  let speed_slow : ℝ := 72
  let speed_fast : ℝ := 78
  let time_slow : ℝ := distance / speed_slow
  let time_fast : ℝ := distance / speed_fast
  time_slow - time_fast = 0.004629369 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_difference_l2700_270079


namespace NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l2700_270091

theorem exponent_of_five_in_30_factorial :
  ∃ k : ℕ, (30 : ℕ).factorial = 5^7 * k ∧ ¬(5 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l2700_270091


namespace NUMINAMATH_CALUDE_counters_ratio_l2700_270063

/-- Represents a person with counters and marbles -/
structure Person where
  counters : ℕ
  marbles : ℕ

/-- The problem setup -/
def problem : Prop :=
  ∃ (reina kevin : Person),
    kevin.counters = 40 ∧
    kevin.marbles = 50 ∧
    reina.marbles = 4 * kevin.marbles ∧
    reina.counters + reina.marbles = 320 ∧
    reina.counters * 1 = kevin.counters * 3

/-- The theorem stating that the ratio of Reina's counters to Kevin's counters is 3:1 -/
theorem counters_ratio : problem := by
  sorry

end NUMINAMATH_CALUDE_counters_ratio_l2700_270063


namespace NUMINAMATH_CALUDE_compound_weight_l2700_270042

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (oxygen_count : ℕ) 
  (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  (carbon_count : ℝ) * carbon_weight + 
  (hydrogen_count : ℝ) * hydrogen_weight + 
  (oxygen_count : ℝ) * oxygen_weight

theorem compound_weight : 
  molecular_weight 2 4 2 12.01 1.008 16.00 = 60.052 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l2700_270042


namespace NUMINAMATH_CALUDE_library_shelves_l2700_270023

/-- Given a library with 14240 books and shelves that hold 8 books each, 
    the number of shelves required is 1780. -/
theorem library_shelves : 
  ∀ (total_books : ℕ) (books_per_shelf : ℕ),
    total_books = 14240 →
    books_per_shelf = 8 →
    total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l2700_270023


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2700_270032

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  A = π / 3 →
  a = Real.sqrt 6 →
  b = 2 →
  a > b →
  (a / Real.sin A = b / Real.sin B) →
  B = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2700_270032


namespace NUMINAMATH_CALUDE_reconstruct_numbers_l2700_270007

/-- Given 10 real numbers representing pairwise sums of 5 unknown numbers,
    prove that the 5 original numbers can be uniquely reconstructed. -/
theorem reconstruct_numbers (a : Fin 10 → ℝ) :
  ∃! (x : Fin 5 → ℝ), ∀ (i j : Fin 5), i < j →
    ∃ (k : Fin 10), a k = x i + x j :=
by sorry

end NUMINAMATH_CALUDE_reconstruct_numbers_l2700_270007


namespace NUMINAMATH_CALUDE_combination_sum_equality_problem_statement_l2700_270088

theorem combination_sum_equality : ∀ (n k : ℕ), k ≤ n →
  (Nat.choose n k) + (Nat.choose n (k+1)) = Nat.choose (n+1) (k+1) :=
sorry

theorem problem_statement : (Nat.choose 12 5) + (Nat.choose 12 6) = Nat.choose 13 6 :=
sorry

end NUMINAMATH_CALUDE_combination_sum_equality_problem_statement_l2700_270088


namespace NUMINAMATH_CALUDE_news_spread_theorem_l2700_270017

/-- Represents a village with residents and their acquaintance relationships -/
structure Village where
  residents : Finset Nat
  acquaintances : Nat → Nat → Prop

/-- Represents the spread of news in the village over time -/
def news_spread (v : Village) (initial : Finset Nat) (t : Nat) : Finset Nat :=
  sorry

/-- The theorem stating that there exists a subset of 90 residents that can spread news to all residents within 10 days -/
theorem news_spread_theorem (v : Village) : 
  v.residents.card = 1000 → 
  ∃ (subset : Finset Nat), 
    subset.card = 90 ∧ 
    subset ⊆ v.residents ∧
    news_spread v subset 10 = v.residents :=
  sorry

end NUMINAMATH_CALUDE_news_spread_theorem_l2700_270017


namespace NUMINAMATH_CALUDE_power_of_power_l2700_270087

theorem power_of_power (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2700_270087


namespace NUMINAMATH_CALUDE_laptop_purchase_solution_l2700_270012

/-- Represents the laptop purchase problem for a unit -/
structure LaptopPurchase where
  totalStaff : ℕ
  totalBudget : ℕ
  costA1B2 : ℕ
  costA2B1 : ℕ

/-- The cost of one A laptop -/
def costA (lp : LaptopPurchase) : ℕ := 4080

/-- The cost of one B laptop -/
def costB (lp : LaptopPurchase) : ℕ := 3880

/-- The maximum number of A laptops that can be bought -/
def maxA (lp : LaptopPurchase) : ℕ := 26

/-- Theorem stating the correctness of the laptop purchase solution -/
theorem laptop_purchase_solution (lp : LaptopPurchase) 
  (h1 : lp.totalStaff = 36)
  (h2 : lp.totalBudget = 145000)
  (h3 : lp.costA1B2 = 11840)
  (h4 : lp.costA2B1 = 12040) :
  (costA lp + 2 * costB lp = lp.costA1B2) ∧ 
  (2 * costA lp + costB lp = lp.costA2B1) ∧
  (maxA lp * costA lp + (lp.totalStaff - maxA lp) * costB lp ≤ lp.totalBudget) ∧
  (∀ n : ℕ, n > maxA lp → n * costA lp + (lp.totalStaff - n) * costB lp > lp.totalBudget) := by
  sorry

#check laptop_purchase_solution

end NUMINAMATH_CALUDE_laptop_purchase_solution_l2700_270012


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_of_squares_l2700_270013

theorem like_terms_imply_sum_of_squares (m n : ℤ) : 
  (m + 10 = 3*n - m) → (7 - n = n - m) → m^2 - 2*m*n + n^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_of_squares_l2700_270013


namespace NUMINAMATH_CALUDE_principal_amount_l2700_270048

/-- Given a principal amount P and an interest rate R, if the amount after 2 years is 850
    and after 7 years is 1020, then the principal amount P is 782. -/
theorem principal_amount (P R : ℚ) : 
  P + (P * R * 2) / 100 = 850 →
  P + (P * R * 7) / 100 = 1020 →
  P = 782 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_l2700_270048


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2700_270038

/-- The equation has exactly one solution when its discriminant is zero -/
def has_one_solution (a : ℝ) (k : ℝ) : Prop :=
  (a + 1/a + 1)^2 - 4*k = 0

/-- The condition for exactly one positive value of a -/
def unique_positive_a (k : ℝ) : Prop :=
  ∃! a : ℝ, a > 0 ∧ has_one_solution a k

theorem quadratic_equation_unique_solution :
  ∃! k : ℝ, k ≠ 0 ∧ unique_positive_a k ∧ k = 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2700_270038


namespace NUMINAMATH_CALUDE_quadratic_circle_theorem_l2700_270022

/-- Quadratic function f(x) = x^2 + 2x + b --/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

/-- Circle equation C(x, y) = 0 --/
def C (b : ℝ) (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - (b + 1)*y + b

theorem quadratic_circle_theorem (b : ℝ) (hb : b < 1 ∧ b ≠ 0) :
  /- The function intersects the coordinate axes at three points -/
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = 0 ∧ f b x₂ = 0) ∧
  (∃ y : ℝ, f b 0 = y) ∧
  /- The circle C passing through these three points has the given equation -/
  (∀ x y : ℝ, (f b x = y ∨ (x = 0 ∧ y = f b 0) ∨ (y = 0 ∧ f b x = 0)) → C b x y = 0) ∧
  /- Circle C passes through the fixed points (0, 1) and (-2, 1) for all valid b -/
  C b 0 1 = 0 ∧ C b (-2) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_circle_theorem_l2700_270022


namespace NUMINAMATH_CALUDE_false_balance_inequality_l2700_270037

/-- A false balance with two pans A and B -/
structure FalseBalance where
  l : ℝ  -- length of arm A
  l' : ℝ  -- length of arm B
  false_balance : l ≠ l'

/-- The balance condition for the false balance -/
def balances (b : FalseBalance) (w1 w2 : ℝ) (on_a : Bool) : Prop :=
  if on_a then w1 * b.l = w2 * b.l' else w1 * b.l' = w2 * b.l

theorem false_balance_inequality (b : FalseBalance) (p x y : ℝ) 
  (h1 : balances b p x false)
  (h2 : balances b p y true) :
  x + y > 2 * p := by
  sorry

end NUMINAMATH_CALUDE_false_balance_inequality_l2700_270037


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2700_270031

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  (3 * a 1 - 2 * a 2 = (1/2) * a 3 - 2 * a 2) →
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2700_270031


namespace NUMINAMATH_CALUDE_cory_patio_set_cost_l2700_270080

def patio_set_cost (table_cost chair_cost : ℕ) (num_chairs : ℕ) : ℕ :=
  table_cost + num_chairs * chair_cost

theorem cory_patio_set_cost : patio_set_cost 55 20 4 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cory_patio_set_cost_l2700_270080


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2700_270065

theorem rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 24)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 10) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    a * b * c = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2700_270065


namespace NUMINAMATH_CALUDE_basketball_handshakes_l2700_270089

/-- The number of handshakes in a basketball game with two teams and referees -/
theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 := by
  sorry

#check basketball_handshakes

end NUMINAMATH_CALUDE_basketball_handshakes_l2700_270089


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l2700_270003

/-- The circle equation: x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation: ax + 2by + 4 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + 2*b*y + 4 = 0

/-- The chord length is 4 -/
def chord_length_is_4 (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧
    line_equation a b x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem min_value_of_a2_plus_b2 :
  ∀ a b : ℝ, chord_length_is_4 a b →
  ∃ min : ℝ, min = 2 ∧ a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l2700_270003


namespace NUMINAMATH_CALUDE_only_B_in_third_quadrant_l2700_270016

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given points -/
def pointA : Point := ⟨2, 3⟩
def pointB : Point := ⟨-1, -4⟩
def pointC : Point := ⟨-4, 1⟩
def pointD : Point := ⟨5, -3⟩

/-- Theorem stating that only point B is in the third quadrant -/
theorem only_B_in_third_quadrant :
  ¬isInThirdQuadrant pointA ∧
  isInThirdQuadrant pointB ∧
  ¬isInThirdQuadrant pointC ∧
  ¬isInThirdQuadrant pointD :=
sorry

end NUMINAMATH_CALUDE_only_B_in_third_quadrant_l2700_270016


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l2700_270036

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x, x^2 - 3*x + 2 < 0 ∧ ¬(x < 2)) = False ∧
  (∃ x, x < 2 ∧ x^2 - 3*x + 2 ≥ 0) = True :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l2700_270036


namespace NUMINAMATH_CALUDE_individual_can_cost_l2700_270020

-- Define the cost of a 12-pack of soft drinks
def pack_cost : ℚ := 299 / 100

-- Define the number of cans in a pack
def cans_per_pack : ℕ := 12

-- Define the function to calculate the cost per can
def cost_per_can : ℚ := pack_cost / cans_per_pack

-- Theorem to prove
theorem individual_can_cost : 
  (round (cost_per_can * 100) / 100 : ℚ) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_individual_can_cost_l2700_270020


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l2700_270097

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 37

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

theorem fgh_supermarkets_count : 
  us_supermarkets = 37 ∧ 
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l2700_270097


namespace NUMINAMATH_CALUDE_necktie_colors_l2700_270084

-- Define the number of different colored shirts
def num_shirts : ℕ := 4

-- Define the probability of all boxes containing matching colors
def match_probability : ℚ := 1 / 24

-- Theorem statement
theorem necktie_colors (n : ℕ) : 
  (n : ℚ) ^ num_shirts = 1 / match_probability → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_necktie_colors_l2700_270084


namespace NUMINAMATH_CALUDE_rectangular_hall_length_l2700_270061

/-- 
Proves that for a rectangular hall where the breadth is two-thirds of the length 
and the area is 2400 sq metres, the length is 60 metres.
-/
theorem rectangular_hall_length 
  (length breadth : ℝ) 
  (breadth_relation : breadth = (2/3) * length)
  (area : ℝ)
  (area_calculation : area = length * breadth)
  (given_area : area = 2400) : 
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_length_l2700_270061


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l2700_270019

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (x^3 - 18*x^2 + 19*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) → 
  (1 + a) * (1 + b) * (1 + c) = 46 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l2700_270019


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2700_270062

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/15 = 1

-- Define the left focus of the hyperbola
def left_focus : ℝ × ℝ := (-4, 0)

-- Define the line that the circle is tangent to
def tangent_line (x : ℝ) : Prop := x = 4

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_focus : center.1 - 4 = center.2
  tangent_to_line : center.1 + center.2 = 4

-- Theorem statement
theorem trajectory_is_parabola (M : MovingCircle) :
  (M.center.2)^2 = -16 * M.center.1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2700_270062


namespace NUMINAMATH_CALUDE_x_plus_inv_x_equals_three_l2700_270026

theorem x_plus_inv_x_equals_three (x : ℝ) (h_pos : x > 0) 
  (h_eq : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) : 
  x + 1/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_equals_three_l2700_270026


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l2700_270053

theorem simplify_sqrt_product : Real.sqrt 18 * Real.sqrt 32 * Real.sqrt 2 = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l2700_270053


namespace NUMINAMATH_CALUDE_geometry_propositions_l2700_270072

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) : 
  -- Proposition 1 is false
  ¬(∀ m α β, parallel_line_plane m α → parallel_line_plane m β → parallel_plane_plane α β) ∧
  -- Proposition 2 is true
  (∀ m α β, perpendicular_line_plane m α → perpendicular_line_plane m β → parallel_plane_plane α β) ∧
  -- Proposition 3 is false
  ¬(∀ m n α, parallel_line_plane m α → parallel_line_plane n α → parallel_line_line m n) ∧
  -- Proposition 4 is true
  (∀ m n α, perpendicular_line_plane m α → perpendicular_line_plane n α → parallel_line_line m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2700_270072


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2700_270014

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2700_270014


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2700_270082

/-- Represents an ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m * y^2 = 1

/-- Represents the property that the foci of the ellipse lie on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1/m - 1 ∧ c ≠ 0

/-- Represents the property that the length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * (1 / Real.sqrt m) = 2 * 2 * 1

theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : foci_on_y_axis e)
  (h2 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2700_270082


namespace NUMINAMATH_CALUDE_integral_equals_22_over_3_l2700_270071

theorem integral_equals_22_over_3 : ∫ x in (1 : ℝ)..3, (2 * x - 1 / x^2) = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_22_over_3_l2700_270071


namespace NUMINAMATH_CALUDE_sqrt_sum_fourth_power_l2700_270006

theorem sqrt_sum_fourth_power : (Real.sqrt (Real.sqrt 9 + Real.sqrt 1))^4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fourth_power_l2700_270006


namespace NUMINAMATH_CALUDE_total_reading_time_is_seven_weeks_l2700_270083

/-- Represents the reading plan for a section of the Bible -/
structure ReadingPlan where
  weekdaySpeed : ℕ  -- pages per hour on weekdays
  weekdayTime : ℚ   -- hours read on weekdays
  saturdaySpeed : ℕ -- pages per hour on Saturdays
  saturdayTime : ℚ  -- hours read on Saturdays
  pageCount : ℕ     -- total pages in this section

/-- Calculates the number of weeks needed to complete a reading plan -/
def weeksToComplete (plan : ReadingPlan) : ℚ :=
  let pagesPerWeek := plan.weekdaySpeed * plan.weekdayTime * 5 + plan.saturdaySpeed * plan.saturdayTime
  plan.pageCount / pagesPerWeek

/-- The reading plan for the Books of Moses -/
def mosesplan : ReadingPlan := {
  weekdaySpeed := 30,
  weekdayTime := 3/2,
  saturdaySpeed := 40,
  saturdayTime := 2,
  pageCount := 450
}

/-- The reading plan for the rest of the Bible -/
def restplan : ReadingPlan := {
  weekdaySpeed := 45,
  weekdayTime := 3/2,
  saturdaySpeed := 60,
  saturdayTime := 5/2,
  pageCount := 2350
}

/-- Theorem stating that the total reading time is 7 weeks -/
theorem total_reading_time_is_seven_weeks :
  ⌈weeksToComplete mosesplan⌉ + ⌈weeksToComplete restplan⌉ = 7 := by
  sorry


end NUMINAMATH_CALUDE_total_reading_time_is_seven_weeks_l2700_270083


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l2700_270078

/-- The ratio of the volume of a cone to the volume of a cylinder with the same radius,
    where the cone's height is half that of the cylinder, and the cylinder has height 12 and radius 4. -/
theorem cone_cylinder_volume_ratio :
  let cylinder_height : ℝ := 12
  let cylinder_radius : ℝ := 4
  let cone_height : ℝ := cylinder_height / 2
  let cone_radius : ℝ := cylinder_radius
  let cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * cylinder_height
  let cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
  cone_volume / cylinder_volume = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l2700_270078


namespace NUMINAMATH_CALUDE_unique_digits_product_rounding_l2700_270029

theorem unique_digits_product_rounding : ∃! (A B C : ℕ), 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧  -- digits are less than 10
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧     -- digits are distinct
  (⌊((10 * A + B : ℝ) + 0.1 * C) * C + 0.5⌋ = 10 * B + C) ∧ -- main equation
  A = 1 ∧ B = 4 ∧ C = 3 :=
by sorry


end NUMINAMATH_CALUDE_unique_digits_product_rounding_l2700_270029


namespace NUMINAMATH_CALUDE_beverage_total_cups_l2700_270030

/-- Represents the ratio of ingredients in the beverage -/
structure BeverageRatio where
  milk : ℕ
  coffee : ℕ
  sugar : ℕ

/-- Calculates the total number of cups given a ratio and the number of coffee cups -/
def totalCups (ratio : BeverageRatio) (coffeeCups : ℕ) : ℕ :=
  let partSize := coffeeCups / ratio.coffee
  partSize * (ratio.milk + ratio.coffee + ratio.sugar)

/-- Theorem stating that for the given ratio and coffee amount, the total is 18 cups -/
theorem beverage_total_cups :
  let ratio : BeverageRatio := { milk := 3, coffee := 2, sugar := 1 }
  let coffeeCups : ℕ := 6
  totalCups ratio coffeeCups = 18 := by
  sorry


end NUMINAMATH_CALUDE_beverage_total_cups_l2700_270030


namespace NUMINAMATH_CALUDE_remainder_4032_divided_by_125_l2700_270047

theorem remainder_4032_divided_by_125 : 
  4032 % 125 = 32 := by sorry

end NUMINAMATH_CALUDE_remainder_4032_divided_by_125_l2700_270047


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2700_270043

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 10) → (∃ y : ℝ, y^2 = 16*y - 10 ∧ x + y = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2700_270043


namespace NUMINAMATH_CALUDE_intersection_M_N_l2700_270028

-- Define set M
def M : Set ℝ := {x | |x| < 1}

-- Define set N
def N : Set ℝ := {y | ∃ x ∈ M, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2700_270028


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l2700_270075

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l2700_270075


namespace NUMINAMATH_CALUDE_year_end_bonus_recipients_l2700_270057

/-- The total number of people receiving year-end bonuses in a company. -/
def total_award_recipients : ℕ := by sorry

/-- The amount of the first prize in ten thousands of yuan. -/
def first_prize : ℚ := 1.5

/-- The amount of the second prize in ten thousands of yuan. -/
def second_prize : ℚ := 1

/-- The amount of the third prize in ten thousands of yuan. -/
def third_prize : ℚ := 0.5

/-- The total bonus amount in ten thousands of yuan. -/
def total_bonus : ℚ := 100

theorem year_end_bonus_recipients :
  ∃ (x y z : ℕ),
    (x + y + z = total_award_recipients) ∧
    (first_prize * x + second_prize * y + third_prize * z = total_bonus) ∧
    (93 ≤ z - x) ∧ (z - x < 96) ∧
    (total_award_recipients = 147) := by sorry

end NUMINAMATH_CALUDE_year_end_bonus_recipients_l2700_270057


namespace NUMINAMATH_CALUDE_binomial_sum_l2700_270027

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  ((1 + 2 * x) ^ 5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry


end NUMINAMATH_CALUDE_binomial_sum_l2700_270027


namespace NUMINAMATH_CALUDE_smallest_gregory_bottles_l2700_270086

/-- The number of bottles Paul drinks -/
def paul_bottles : ℕ → ℕ := fun p => p

/-- The number of bottles Donald drinks -/
def donald_bottles : ℕ → ℕ := fun p => 2 * paul_bottles p + 3

/-- The number of bottles Gregory drinks -/
def gregory_bottles : ℕ → ℕ := fun p => 3 * donald_bottles p - 5

theorem smallest_gregory_bottles :
  ∀ p : ℕ, p ≥ 1 → gregory_bottles p ≥ 10 ∧ gregory_bottles 1 = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_gregory_bottles_l2700_270086


namespace NUMINAMATH_CALUDE_sequence_properties_l2700_270045

/-- Given a sequence a_n with the specified properties, prove the geometric sequence property and the sum formula. -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, S (n + 1) + a n = S n + 5 * (4 ^ n)) :
  (∀ n : ℕ, a (n + 1) - 4^(n + 1) = -(a n - 4^n)) ∧ 
  (∀ n : ℕ, S n = (4^(n + 1) / 3) - ((-1)^(n + 1) / 2) - (11 / 6)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2700_270045


namespace NUMINAMATH_CALUDE_jennifer_fruit_count_l2700_270005

def fruit_problem (pears oranges : ℕ) : Prop :=
  let apples := 2 * pears
  let cherries := oranges / 2
  let grapes := 3 * apples
  let initial_total := pears + oranges + apples + cherries + grapes
  let pineapples := initial_total
  let remaining_pears := pears - 3
  let remaining_oranges := oranges - 5
  let remaining_apples := apples - 5
  let remaining_cherries := cherries - 7
  let remaining_grapes := grapes - 3
  let remaining_before_pineapples := remaining_pears + remaining_oranges + remaining_apples + remaining_cherries + remaining_grapes
  let remaining_pineapples := pineapples - (pineapples / 2)
  remaining_before_pineapples + remaining_pineapples = 247

theorem jennifer_fruit_count : fruit_problem 15 30 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fruit_count_l2700_270005


namespace NUMINAMATH_CALUDE_log_problem_l2700_270010

theorem log_problem (y : ℝ) : y = (Real.log 3 / Real.log 9) ^ (Real.log 16 / Real.log 4) → Real.log y / Real.log 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2700_270010


namespace NUMINAMATH_CALUDE_geometric_sequence_product_property_geometric_sequence_specific_product_l2700_270066

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- In a geometric sequence, the product of terms equidistant from any center term is constant. -/
theorem geometric_sequence_product_property {a : ℕ → ℝ} (h : IsGeometricSequence a) :
    ∀ n k : ℕ, a (n - k) * a (n + k) = (a n) ^ 2 :=
  sorry

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_specific_product (a : ℕ → ℝ) 
    (h1 : IsGeometricSequence a) (h2 : a 4 = 4) : a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_property_geometric_sequence_specific_product_l2700_270066


namespace NUMINAMATH_CALUDE_intersection_distance_l2700_270081

theorem intersection_distance (n d k : ℝ) (h1 : d ≠ 0) (h2 : n * 0 + d = 3) :
  let f (x : ℝ) := x^2 + 4*x + 3
  let g (x : ℝ) := n*x + d
  let c := |f k - g k|
  (∃! k, f k = g k) → c = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2700_270081


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2700_270059

/-- Represents the state of the game -/
structure GameState :=
  (stones : ℕ)

/-- Represents a valid move in the game -/
inductive Move : Type
  | take_one : Move
  | take_two : Move

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.take_one => ⟨state.stones - 1⟩
  | Move.take_two => ⟨state.stones - 2⟩

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.take_one => state.stones ≥ 1
  | Move.take_two => state.stones ≥ 2

/-- Defines the winning strategy sequence for the first player -/
def winning_sequence : List ℕ := [21, 18, 15, 12, 9, 6, 3, 0]

/-- Theorem: The first player has a winning strategy in the 22-stone game -/
theorem first_player_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_move : Move),
      let initial_state := GameState.mk 22
      let first_move := strategy initial_state
      let state_after_first_move := apply_move initial_state first_move
      let state_after_opponent := apply_move state_after_first_move opponent_move
      is_valid_move initial_state first_move ∧
      is_valid_move state_after_first_move opponent_move ∧
      state_after_opponent.stones ∈ winning_sequence :=
by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2700_270059


namespace NUMINAMATH_CALUDE_range_of_a_l2700_270058

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a+2)*x + b

-- Define the theorem
theorem range_of_a (a b : ℝ) :
  (f a b (-1) = -2) →
  (∀ x, ∃ y, y = Real.log (f a b x + 3)) →
  -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2700_270058


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l2700_270099

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a2 : a 2 = 4)
  (h_a3 : a 3 = 6) :
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l2700_270099


namespace NUMINAMATH_CALUDE_weeklySalesTheorem_l2700_270073

/-- Calculates the total sales for a week given the following conditions:
- Number of houses visited per day
- Percentage of customers who buy something
- Percentages and prices of different products
- Number of working days
- Discount percentage on the last day
-/
def calculateWeeklySales (
  housesPerDay : ℕ)
  (buyPercentage : ℚ)
  (product1Percentage : ℚ) (product1Price : ℚ)
  (product2Percentage : ℚ) (product2Price : ℚ)
  (product3Percentage : ℚ) (product3Price : ℚ)
  (product4Percentage : ℚ) (product4Price : ℚ)
  (workingDays : ℕ)
  (lastDayDiscount : ℚ) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, the total sales for the week
    equal $9624.375 -/
theorem weeklySalesTheorem :
  calculateWeeklySales 50 (30/100)
    (35/100) 50
    (40/100) 150
    (15/100) 75
    (10/100) 200
    6 (10/100) = 9624375/1000 := by sorry

end NUMINAMATH_CALUDE_weeklySalesTheorem_l2700_270073


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2700_270049

theorem fraction_sum_equality (p q r : ℝ) 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2700_270049


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l2700_270021

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l2700_270021
