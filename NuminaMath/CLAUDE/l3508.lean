import Mathlib

namespace NUMINAMATH_CALUDE_no_closed_broken_line_315_l3508_350841

/-- A closed broken line with the given properties -/
structure ClosedBrokenLine where
  segments : ℕ
  intersecting : Bool
  perpendicular : Bool
  symmetric : Bool

/-- The number of segments in our specific case -/
def n : ℕ := 315

/-- Theorem stating the impossibility of constructing the specified closed broken line -/
theorem no_closed_broken_line_315 :
  ¬ ∃ (line : ClosedBrokenLine), 
    line.segments = n ∧
    line.intersecting ∧
    line.perpendicular ∧
    line.symmetric :=
sorry


end NUMINAMATH_CALUDE_no_closed_broken_line_315_l3508_350841


namespace NUMINAMATH_CALUDE_alba_oranges_theorem_l3508_350818

/-- Represents the orange production and sale scenario of the Morales sisters -/
structure OrangeScenario where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  maricela_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℕ
  total_revenue : ℕ

/-- Calculates the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree (scenario : OrangeScenario) : ℕ :=
  let total_cups := scenario.total_revenue / scenario.price_per_cup
  let total_oranges := total_cups * scenario.oranges_per_cup
  let gabriela_oranges := scenario.gabriela_oranges_per_tree * scenario.trees_per_sister
  let maricela_oranges := scenario.maricela_oranges_per_tree * scenario.trees_per_sister
  let alba_total_oranges := total_oranges - gabriela_oranges - maricela_oranges
  alba_total_oranges / scenario.trees_per_sister

/-- The main theorem stating that given the scenario conditions, Alba's trees produce 400 oranges per tree -/
theorem alba_oranges_theorem (scenario : OrangeScenario) 
  (h1 : scenario.trees_per_sister = 110)
  (h2 : scenario.gabriela_oranges_per_tree = 600)
  (h3 : scenario.maricela_oranges_per_tree = 500)
  (h4 : scenario.oranges_per_cup = 3)
  (h5 : scenario.price_per_cup = 4)
  (h6 : scenario.total_revenue = 220000) :
  alba_oranges_per_tree scenario = 400 := by
  sorry

end NUMINAMATH_CALUDE_alba_oranges_theorem_l3508_350818


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3508_350817

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (3*x + 4) = 9*x - 19 → x = (7 + 2*Real.sqrt 7) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3508_350817


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_is_one_l3508_350864

/-- 
Theorem: For any three-digit number where the hundreds digit is 3 more than the units digit,
the tens digit of the sum of this number and its reverse is always 1.
-/
theorem tens_digit_of_sum_is_one (c b : ℕ) (h1 : c < 10) (h2 : b < 10) : 
  (((202 * c + 20 * b + 303) / 10) % 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_is_one_l3508_350864


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l3508_350812

/-- Given an incident light ray along y = 2x + 1 reflected by the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) → -- incident light ray equation
  (y = x) →       -- reflecting line equation
  (x - 2*y - 1 = 0) -- reflected light ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l3508_350812


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3508_350802

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem sufficient_not_necessary_condition
  (f : ℝ → ℝ) (hf : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3508_350802


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l3508_350846

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at most one shot is successful"
def at_most_one_successful (ω : Ω) : Prop :=
  ¬(ω.1 ∧ ω.2)

-- Define the event "both shots are successful"
def both_successful (ω : Ω) : Prop :=
  ω.1 ∧ ω.2

-- Theorem: "both shots are successful" is mutually exclusive to "at most one shot is successful"
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(at_most_one_successful ω ∧ both_successful ω) :=
by
  sorry


end NUMINAMATH_CALUDE_mutually_exclusive_events_l3508_350846


namespace NUMINAMATH_CALUDE_first_berry_count_l3508_350821

/-- A sequence of berry counts where the difference between consecutive counts increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem first_berry_count
  (a : ℕ → ℕ)
  (h_seq : BerrySequence a)
  (h_2 : a 2 = 4)
  (h_3 : a 3 = 7)
  (h_4 : a 4 = 12)
  (h_5 : a 5 = 19) :
  a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_berry_count_l3508_350821


namespace NUMINAMATH_CALUDE_inequality_proof_l3508_350805

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : b^2 / a ≥ 2*b - a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3508_350805


namespace NUMINAMATH_CALUDE_equation_solution_l3508_350823

theorem equation_solution (x : ℚ) : 
  (3 : ℚ) / 4 + 1 / x = (7 : ℚ) / 8 → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3508_350823


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3508_350854

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3508_350854


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3508_350888

theorem units_digit_of_product (n : ℕ) : (4^101 * 5^204 * 9^303 * 11^404) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3508_350888


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3508_350828

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 10

theorem circle_center_coordinates :
  CircleCenter 2 1 := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3508_350828


namespace NUMINAMATH_CALUDE_five_month_practice_time_l3508_350851

/-- Calculates the total piano practice time over a given number of months. -/
def total_practice_time (weekly_hours : ℕ) (weeks_per_month : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * weeks_per_month * months

/-- Theorem stating that practicing 4 hours per week for 5 months results in 80 hours of practice. -/
theorem five_month_practice_time :
  total_practice_time 4 4 5 = 80 := by
  sorry

#eval total_practice_time 4 4 5

end NUMINAMATH_CALUDE_five_month_practice_time_l3508_350851


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3508_350822

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3508_350822


namespace NUMINAMATH_CALUDE_train_arrival_theorem_l3508_350861

/-- Represents a time with day, hour, and minute -/
structure Time where
  day : String
  hour : Nat
  minute : Nat

/-- Represents the journey of the train -/
structure TrainJourney where
  startTime : Time
  firstLegDuration : Nat
  secondLegDuration : Nat
  layoverDuration : Nat
  timeZonesCrossed : Nat
  timeZoneDifference : Nat

def calculateArrivalTime (journey : TrainJourney) : Time :=
  sorry

theorem train_arrival_theorem (journey : TrainJourney) 
  (h1 : journey.startTime = ⟨"Tuesday", 5, 0⟩)
  (h2 : journey.firstLegDuration = 12)
  (h3 : journey.secondLegDuration = 21)
  (h4 : journey.layoverDuration = 3)
  (h5 : journey.timeZonesCrossed = 2)
  (h6 : journey.timeZoneDifference = 1) :
  calculateArrivalTime journey = ⟨"Wednesday", 9, 0⟩ :=
by
  sorry

#check train_arrival_theorem

end NUMINAMATH_CALUDE_train_arrival_theorem_l3508_350861


namespace NUMINAMATH_CALUDE_fraction_increase_condition_l3508_350813

theorem fraction_increase_condition (m n : ℤ) (h1 : n ≠ 0) (h2 : n ≠ -1) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) ↔ (n > 0 ∧ m < n) ∨ (n < -1 ∧ m > n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_condition_l3508_350813


namespace NUMINAMATH_CALUDE_monopoly_prefers_durable_coffee_machine_production_decision_l3508_350850

/-- Represents the type of coffee machine -/
inductive CoffeeMachineType
| Durable
| LowQuality

/-- Represents the market structure -/
inductive MarketStructure
| Monopoly
| PerfectlyCompetitive

/-- Represents a coffee machine -/
structure CoffeeMachine where
  type : CoffeeMachineType
  productionCost : ℝ

/-- Represents the consumer's utility from using a coffee machine -/
def consumerUtility : ℝ := 20

/-- Represents the lifespan of a coffee machine in periods -/
def machineLifespan (t : CoffeeMachineType) : ℕ :=
  match t with
  | CoffeeMachineType.Durable => 2
  | CoffeeMachineType.LowQuality => 1

/-- Calculates the profit for a monopolist selling a coffee machine -/
def monopolyProfit (m : CoffeeMachine) : ℝ :=
  (consumerUtility * machineLifespan m.type) - m.productionCost

/-- Theorem: In a monopoly, durable machines are produced when low-quality machine cost exceeds 6 -/
theorem monopoly_prefers_durable (c : ℝ) :
  let durableMachine : CoffeeMachine := ⟨CoffeeMachineType.Durable, 12⟩
  let lowQualityMachine : CoffeeMachine := ⟨CoffeeMachineType.LowQuality, c⟩
  monopolyProfit durableMachine > 2 * monopolyProfit lowQualityMachine ↔ c > 6 := by
  sorry

/-- Main theorem combining all conditions -/
theorem coffee_machine_production_decision 
  (marketStructure : MarketStructure) 
  (c : ℝ) : 
  (marketStructure = MarketStructure.Monopoly ∧ c > 6) ↔ 
  (∃ (d : CoffeeMachine), d.type = CoffeeMachineType.Durable ∧ 
   ∀ (l : CoffeeMachine), l.type = CoffeeMachineType.LowQuality → 
   monopolyProfit d > monopolyProfit l) := by
  sorry

end NUMINAMATH_CALUDE_monopoly_prefers_durable_coffee_machine_production_decision_l3508_350850


namespace NUMINAMATH_CALUDE_polynomial_equality_l3508_350806

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3508_350806


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3508_350870

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 3 ∧ x = 2 ∧ y = 1) ∧
  -- Second system of equations
  (∃ x y : ℝ, x/4 + y/3 = 3 ∧ 3*x - 2*(y-1) = 11 ∧ x = 6 ∧ y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3508_350870


namespace NUMINAMATH_CALUDE_y_change_when_x_increases_y_decreases_by_1_5_l3508_350865

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 1.5 * x

-- Theorem stating the change in y when x increases by one unit
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

-- Theorem stating that y decreases by 1.5 units when x increases by one unit
theorem y_decreases_by_1_5 (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_y_change_when_x_increases_y_decreases_by_1_5_l3508_350865


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3508_350876

theorem stratified_sampling_problem (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (female_sample : ℕ) (total_sample : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) * total_sample = female_sample →
  total_sample = 192 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3508_350876


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3508_350857

theorem quadratic_factorization (a : ℕ+) :
  (∃ m n p q : ℤ, (21 : ℤ) * x^2 + (a : ℤ) * x + 21 = (m * x + n) * (p * x + q)) →
  ∃ k : ℕ+, a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3508_350857


namespace NUMINAMATH_CALUDE_flag_design_count_l3508_350825

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flags : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  total_flags = 27 :=
by sorry

end NUMINAMATH_CALUDE_flag_design_count_l3508_350825


namespace NUMINAMATH_CALUDE_program_output_l3508_350866

def program (initial_A initial_B : Int) : (Int × Int × Int) :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B * initial_B
  let A₂ := A₁ + B₁
  let C := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l3508_350866


namespace NUMINAMATH_CALUDE_function_properties_l3508_350808

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem function_properties (a b c : ℝ) 
  (h1 : f a b c 0 = 0)
  (h2 : ∀ x : ℝ, f a b c x ≤ f a b c (Real.pi / 3))
  (h3 : ∃ x : ℝ, f a b c x = 1) :
  (∃ x : ℝ, f a b c x = 1) ∧ 
  (∀ x : ℝ, f a b c x ≤ f (Real.sqrt 3) 1 (-1) x) ∧
  (f a b c (b / a) > f a b c (c / a)) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l3508_350808


namespace NUMINAMATH_CALUDE_zero_product_probability_l3508_350895

def S : Finset ℤ := {-3, -2, -1, 0, 0, 2, 4, 5}

def different_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (s.product s).filter (λ (a, b) => a ≠ b)

def zero_product_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (different_pairs s).filter (λ (a, b) => a * b = 0)

theorem zero_product_probability :
  (zero_product_pairs S).card / (different_pairs S).card = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_probability_l3508_350895


namespace NUMINAMATH_CALUDE_function_zeros_l3508_350878

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def count_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

theorem function_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f (2 * Real.pi))
  (h_zero_3 : f 3 = 0)
  (h_zero_4 : f 4 = 0) :
  count_zeros f 0 10 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_function_zeros_l3508_350878


namespace NUMINAMATH_CALUDE_max_weighing_ways_exists_89_ways_l3508_350829

/-- Represents the set of weights as powers of 2 up to 2^9 (512) -/
def weights : Finset ℕ := Finset.range 10

/-- The number of ways a weight P can be measured using weights up to 2^n -/
def K (n : ℕ) (P : ℕ) : ℕ := sorry

/-- The maximum number of ways any weight can be measured using weights up to 2^n -/
def K_max (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 different ways -/
theorem max_weighing_ways : K_max 9 ≤ 89 := sorry

/-- Theorem stating that there exists a load that can be weighed in exactly 89 different ways -/
theorem exists_89_ways : ∃ P : ℕ, K 9 P = 89 := sorry

end NUMINAMATH_CALUDE_max_weighing_ways_exists_89_ways_l3508_350829


namespace NUMINAMATH_CALUDE_polygon_sides_l3508_350837

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  ((n - 2) * 180 = 3 * 360) → -- Interior angles sum is 3 times exterior angles sum
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3508_350837


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_condition_l3508_350834

theorem not_sufficient_nor_necessary_condition (x y : ℝ) : 
  ¬(∀ x y : ℝ, x > y → x^2 > y^2) ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_condition_l3508_350834


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3508_350827

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3508_350827


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3508_350887

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4)^2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3508_350887


namespace NUMINAMATH_CALUDE_zeros_inequality_l3508_350809

open Real

theorem zeros_inequality (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (h₂ : exp (m * x₁) - log x₁ + (m - 1) * x₁ = 0) 
  (h₃ : exp (m * x₂) - log x₂ + (m - 1) * x₂ = 0) : 
  2 * log x₁ + log x₂ > exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_inequality_l3508_350809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3508_350874

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 7 = 4 → a 8 = 1 → a 10 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3508_350874


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3508_350890

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ c = 1/12 ∧ d = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3508_350890


namespace NUMINAMATH_CALUDE_student_allowance_equation_l3508_350819

/-- The student's weekly allowance satisfies the given equation. -/
theorem student_allowance_equation (A : ℝ) : A > 0 → (3/4 : ℝ) * (1/3 : ℝ) * ((2/5 : ℝ) * A + 4) - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_student_allowance_equation_l3508_350819


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3508_350844

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = 1/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3508_350844


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l3508_350892

theorem infinite_solutions_condition (k : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - k) = 3 * (4 * x + 10)) ↔ k = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l3508_350892


namespace NUMINAMATH_CALUDE_labourer_monthly_income_l3508_350868

/-- Represents the financial situation of a labourer over a 10-month period --/
structure LabourerFinances where
  monthlyIncome : ℝ
  firstSixMonthsExpenditure : ℝ
  nextFourMonthsExpenditure : ℝ
  savings : ℝ

/-- Theorem stating the labourer's monthly income given the problem conditions --/
theorem labourer_monthly_income 
  (finances : LabourerFinances)
  (h1 : finances.firstSixMonthsExpenditure = 90 * 6)
  (h2 : finances.monthlyIncome * 6 < finances.firstSixMonthsExpenditure)
  (h3 : finances.nextFourMonthsExpenditure = 60 * 4)
  (h4 : finances.monthlyIncome * 4 = finances.nextFourMonthsExpenditure + finances.savings)
  (h5 : finances.savings = 30) :
  finances.monthlyIncome = 81 := by
  sorry

end NUMINAMATH_CALUDE_labourer_monthly_income_l3508_350868


namespace NUMINAMATH_CALUDE_tickets_difference_l3508_350833

theorem tickets_difference (initial_tickets : ℕ) (toys_tickets : ℕ) (clothes_tickets : ℕ)
  (h1 : initial_tickets = 13)
  (h2 : toys_tickets = 8)
  (h3 : clothes_tickets = 18) :
  clothes_tickets - toys_tickets = 10 := by
  sorry

end NUMINAMATH_CALUDE_tickets_difference_l3508_350833


namespace NUMINAMATH_CALUDE_friday_production_to_meet_target_l3508_350877

/-- The number of toys that need to be produced on Friday to meet the weekly target -/
def friday_production (weekly_target : ℕ) (mon_to_wed_daily : ℕ) (thursday : ℕ) : ℕ :=
  weekly_target - (3 * mon_to_wed_daily + thursday)

/-- Theorem stating the required Friday production to meet the weekly target -/
theorem friday_production_to_meet_target :
  friday_production 6500 1200 800 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_friday_production_to_meet_target_l3508_350877


namespace NUMINAMATH_CALUDE_largest_number_in_L_shape_l3508_350847

/-- Represents the different orientations of the "L" shape -/
inductive LShape
  | First  : LShape  -- (x-8, x-7, x)
  | Second : LShape  -- (x-7, x-6, x)
  | Third  : LShape  -- (x-7, x-1, x)
  | Fourth : LShape  -- (x-8, x-1, x)

/-- Calculates the sum of the three numbers in the "L" shape -/
def sumLShape (shape : LShape) (x : ℕ) : ℕ :=
  match shape with
  | LShape.First  => x - 8 + x - 7 + x
  | LShape.Second => x - 7 + x - 6 + x
  | LShape.Third  => x - 7 + x - 1 + x
  | LShape.Fourth => x - 8 + x - 1 + x

/-- The main theorem to be proved -/
theorem largest_number_in_L_shape : 
  ∃ (shape : LShape) (x : ℕ), sumLShape shape x = 2015 ∧ 
  (∀ (shape' : LShape) (y : ℕ), sumLShape shape' y = 2015 → y ≤ x) ∧ 
  x = 676 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_L_shape_l3508_350847


namespace NUMINAMATH_CALUDE_hugo_rolls_six_given_win_l3508_350835

-- Define the number of players and sides on the die
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the event of Hugo winning
def hugo_wins : Set (Fin num_players → Fin num_sides) := sorry

-- Define the event of Hugo rolling a 6 on his first roll
def hugo_rolls_six : Set (Fin num_players → Fin num_sides) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin num_players → Fin num_sides) → ℝ := sorry

-- Theorem statement
theorem hugo_rolls_six_given_win :
  P (hugo_rolls_six ∩ hugo_wins) / P hugo_wins = 6375 / 32768 := by sorry

end NUMINAMATH_CALUDE_hugo_rolls_six_given_win_l3508_350835


namespace NUMINAMATH_CALUDE_work_completion_time_l3508_350831

theorem work_completion_time (b : ℝ) (a_wage_ratio : ℝ) (a : ℝ) : 
  b = 15 →
  a_wage_ratio = 3/5 →
  (1/a) / ((1/a) + (1/b)) = a_wage_ratio →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3508_350831


namespace NUMINAMATH_CALUDE_larger_denomination_proof_l3508_350801

theorem larger_denomination_proof (total_bills : ℕ) (total_value : ℕ) 
  (ten_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 30 →
  total_value = 330 →
  ten_bills = 27 →
  larger_bills = 3 →
  ten_bills + larger_bills = total_bills →
  10 * ten_bills + larger_bills * (total_value - 10 * ten_bills) / larger_bills = total_value →
  (total_value - 10 * ten_bills) / larger_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_denomination_proof_l3508_350801


namespace NUMINAMATH_CALUDE_ship_passengers_l3508_350826

theorem ship_passengers :
  let total : ℕ := 900
  let north_america : ℚ := 1/4
  let europe : ℚ := 2/15
  let africa : ℚ := 1/5
  let asia : ℚ := 1/6
  let south_america : ℚ := 1/12
  let oceania : ℚ := 1/20
  let other_regions : ℕ := 105
  (north_america + europe + africa + asia + south_america + oceania) * total + other_regions = total :=
by sorry

end NUMINAMATH_CALUDE_ship_passengers_l3508_350826


namespace NUMINAMATH_CALUDE_wire_circle_square_ratio_l3508_350891

theorem wire_circle_square_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (π * (a / (2 * π))^2 = (b / 4)^2) → (a / b = 2 / Real.sqrt π) := by
  sorry

end NUMINAMATH_CALUDE_wire_circle_square_ratio_l3508_350891


namespace NUMINAMATH_CALUDE_small_pizza_slices_l3508_350815

/-- The number of large pizzas --/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas --/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a large pizza --/
def slices_per_large_pizza : ℕ := 16

/-- The total number of slices eaten --/
def total_slices_eaten : ℕ := 48

/-- Theorem: The number of slices in a small pizza is 8 --/
theorem small_pizza_slices : 
  ∃ (slices_per_small_pizza : ℕ), 
    slices_per_small_pizza * num_small_pizzas + 
    slices_per_large_pizza * num_large_pizzas = total_slices_eaten ∧
    slices_per_small_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_slices_l3508_350815


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3508_350860

/-- Given a quadratic equation x^2 + px + q = 0 with roots 2 and -3,
    prove that it can be factored as (x - 2)(x + 3) = 0 -/
theorem quadratic_factorization (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = 2 ∨ x = -3) →
  ∀ x, x^2 + p*x + q = 0 ↔ (x - 2) * (x + 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3508_350860


namespace NUMINAMATH_CALUDE_jake_has_nine_peaches_l3508_350883

/-- Jake has 7 fewer peaches than Steven and 9 more peaches than Jill. Steven has 16 peaches. -/
def peach_problem (jake steven jill : ℕ) : Prop :=
  jake + 7 = steven ∧ jake = jill + 9 ∧ steven = 16

/-- Prove that Jake has 9 peaches. -/
theorem jake_has_nine_peaches :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jake = 9 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_nine_peaches_l3508_350883


namespace NUMINAMATH_CALUDE_oven_usage_calculation_l3508_350845

/-- Represents the problem of calculating oven usage time given electricity price, consumption rate, and total cost. -/
def OvenUsage (price : ℝ) (consumption : ℝ) (total_cost : ℝ) : Prop :=
  let hours := total_cost / (price * consumption)
  hours = 25

/-- Theorem stating that given the specific values in the problem, the oven usage time is 25 hours. -/
theorem oven_usage_calculation :
  OvenUsage 0.10 2.4 6 := by
  sorry

end NUMINAMATH_CALUDE_oven_usage_calculation_l3508_350845


namespace NUMINAMATH_CALUDE_inequality_contradiction_l3508_350849

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l3508_350849


namespace NUMINAMATH_CALUDE_function_inequality_l3508_350842

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (2 - x))

-- State the theorem
theorem function_inequality : f 2.5 > f 1 ∧ f 1 > f 3.5 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3508_350842


namespace NUMINAMATH_CALUDE_line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l3508_350830

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1
theorem line_perpendicular_transitive 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

-- Theorem 2
theorem parallel_lines_from_parallel_planes 
  (l m : Line) (α β γ : Plane) :
  intersect α γ m → intersect β γ l → plane_parallel α β → parallel m l :=
sorry

-- Theorem 3
theorem not_always_parallel_transitive 
  (l m : Line) (α : Plane) :
  ¬(∀ l m α, parallel m l → parallel m α → parallel l α) :=
sorry

-- Theorem 4
theorem not_always_parallel_from_intersections 
  (l m n : Line) (α β γ : Plane) :
  ¬(∀ l m n α β γ, 
    intersect α β l → intersect β γ m → intersect γ α n → 
    parallel l m ∧ parallel m n ∧ parallel l n) :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l3508_350830


namespace NUMINAMATH_CALUDE_jesse_pencils_l3508_350869

/-- Given that Jesse starts with 78 pencils and gives away 44 pencils,
    prove that he ends up with 34 pencils. -/
theorem jesse_pencils :
  let initial_pencils : ℕ := 78
  let pencils_given_away : ℕ := 44
  initial_pencils - pencils_given_away = 34 :=
by sorry

end NUMINAMATH_CALUDE_jesse_pencils_l3508_350869


namespace NUMINAMATH_CALUDE_number_line_percentage_l3508_350832

theorem number_line_percentage : 
  let start : ℝ := -55
  let end_point : ℝ := 55
  let target : ℝ := 5.5
  let total_distance := end_point - start
  let target_distance := target - start
  (target_distance / total_distance) * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_number_line_percentage_l3508_350832


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3508_350852

/-- Given a rectangle with perimeter 60 and one side 5 units longer than the other,
    the maximum area is 218.75 square units. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  y = x + 5 →
  x * y ≤ 218.75 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3508_350852


namespace NUMINAMATH_CALUDE_projectile_speed_problem_l3508_350898

theorem projectile_speed_problem (initial_distance : ℝ) (second_projectile_speed : ℝ) (time_to_meet : ℝ) :
  initial_distance = 1182 →
  second_projectile_speed = 525 →
  time_to_meet = 1.2 →
  ∃ (first_projectile_speed : ℝ),
    first_projectile_speed = 460 ∧
    (first_projectile_speed + second_projectile_speed) * time_to_meet = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_projectile_speed_problem_l3508_350898


namespace NUMINAMATH_CALUDE_original_amount_proof_l3508_350859

def transaction (x : ℚ) : ℚ :=
  ((2/3 * x + 10) * 2/3 + 20)

theorem original_amount_proof :
  ∃ (x : ℚ), x > 0 ∧ transaction x = x ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_proof_l3508_350859


namespace NUMINAMATH_CALUDE_prob_not_six_is_five_sevenths_l3508_350867

/-- A specially designed six-sided die -/
structure SpecialDie :=
  (sides : Nat)
  (odds_six : Rat)
  (is_valid : sides = 6 ∧ odds_six = 2/5)

/-- The probability of rolling a number other than six -/
def prob_not_six (d : SpecialDie) : Rat :=
  1 - (d.odds_six / (1 + d.odds_six))

theorem prob_not_six_is_five_sevenths (d : SpecialDie) :
  prob_not_six d = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_is_five_sevenths_l3508_350867


namespace NUMINAMATH_CALUDE_two_from_three_l3508_350897

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: There are 3 ways to choose 2 items from a set of 3 items -/
theorem two_from_three : combinations 3 2 = 3 := by sorry

end NUMINAMATH_CALUDE_two_from_three_l3508_350897


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_500000_l3508_350810

theorem smallest_n_exceeding_500000 : 
  (∀ k : ℕ, k < 10 → (3 : ℝ) ^ ((k * (k + 1) : ℝ) / 16) ≤ 500000) ∧ 
  (3 : ℝ) ^ ((10 * 11 : ℝ) / 16) > 500000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_500000_l3508_350810


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l3508_350889

/-- Calculate the corrected mean of a dataset with misrecorded observations -/
theorem corrected_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (actual_values : List ℚ) (recorded_values : List ℚ) : 
  n = 25 ∧ 
  incorrect_mean = 50 ∧ 
  actual_values = [20, 35, 70] ∧
  recorded_values = [40, 55, 80] →
  (n * incorrect_mean - (recorded_values.sum - actual_values.sum)) / n = 48 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l3508_350889


namespace NUMINAMATH_CALUDE_example_quadratic_function_l3508_350824

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c where a ≠ 0 such that
    f(x) = ax^2 + bx + c for all x ∈ ℝ -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 + x - 1 is quadratic -/
theorem example_quadratic_function :
  IsQuadratic (fun x => 3 * x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_example_quadratic_function_l3508_350824


namespace NUMINAMATH_CALUDE_family_admission_price_l3508_350816

/-- Calculates the total admission price for a family visiting an amusement park. -/
theorem family_admission_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (h1 : adult_price = 22)
  (h2 : child_price = 7)
  (h3 : num_adults = 2)
  (h4 : num_children = 2) :
  adult_price * num_adults + child_price * num_children = 58 := by
  sorry

#check family_admission_price

end NUMINAMATH_CALUDE_family_admission_price_l3508_350816


namespace NUMINAMATH_CALUDE_sequence_formula_l3508_350839

def S (n : ℕ+) : ℤ := -n^2 + 7*n

def a (n : ℕ+) : ℤ := -2*n + 8

theorem sequence_formula (n : ℕ+) : 
  (∀ k : ℕ+, S k = -k^2 + 7*k) → 
  a n = -2*n + 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3508_350839


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l3508_350858

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- Theorem statement
theorem iron_conducts_electricity 
  (h1 : ∀ x, Metal x → ConductsElectricity x) 
  (h2 : Metal iron) : 
  ConductsElectricity iron := by
  sorry


end NUMINAMATH_CALUDE_iron_conducts_electricity_l3508_350858


namespace NUMINAMATH_CALUDE_lateral_to_base_area_ratio_l3508_350855

/-- A cone with its lateral surface unfolded into a sector with a 90° central angle -/
structure UnfoldedCone where
  r : ℝ  -- radius of the base circle
  R : ℝ  -- radius of the unfolded sector (lateral surface)
  h : R = 4 * r  -- condition from the 90° central angle

/-- The ratio of lateral surface area to base area for an UnfoldedCone is 4:1 -/
theorem lateral_to_base_area_ratio (cone : UnfoldedCone) :
  (π * cone.r * cone.R) / (π * cone.r^2) = 4 := by
  sorry

#check lateral_to_base_area_ratio

end NUMINAMATH_CALUDE_lateral_to_base_area_ratio_l3508_350855


namespace NUMINAMATH_CALUDE_ellipse_equation_l3508_350853

/-- Given an ellipse with equation x²/a² + y²/2 = 1 and one focus at (2,0),
    prove that its specific equation is x²/6 + y²/2 = 1 -/
theorem ellipse_equation (a : ℝ) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/2 = 1) ∧ 
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 - 2) →
  a^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3508_350853


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l3508_350894

/-- The function f(x) -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The function g(x) -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

/-- The derivative of g(x) -/
def g_deriv (x : ℝ) : ℝ := 2*x - 3

theorem tangent_line_theorem (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f_deriv a b 2 = g_deriv 2 →
  a = -3 ∧ b = 1 ∧ ∀ x y, y = x - 2 ↔ f a b x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l3508_350894


namespace NUMINAMATH_CALUDE_inequality_of_means_l3508_350800

theorem inequality_of_means (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > a * b ∧ a * b > 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l3508_350800


namespace NUMINAMATH_CALUDE_binomial_30_3_l3508_350872

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3508_350872


namespace NUMINAMATH_CALUDE_a_is_guilty_l3508_350803

-- Define the set of suspects
inductive Suspect : Type
| A : Suspect
| B : Suspect
| C : Suspect

-- Define the properties of the crime and suspects
class CrimeScene where
  involved : Suspect → Prop
  canDrive : Suspect → Prop
  usedCar : Prop

-- Define the specific conditions of this crime
axiom crime_conditions (cs : CrimeScene) :
  -- The crime was committed using a car
  cs.usedCar ∧
  -- At least one suspect was involved
  (cs.involved Suspect.A ∨ cs.involved Suspect.B ∨ cs.involved Suspect.C) ∧
  -- C never commits a crime without A
  (cs.involved Suspect.C → cs.involved Suspect.A) ∧
  -- B knows how to drive
  cs.canDrive Suspect.B

-- Theorem: A is guilty
theorem a_is_guilty (cs : CrimeScene) : cs.involved Suspect.A :=
sorry

end NUMINAMATH_CALUDE_a_is_guilty_l3508_350803


namespace NUMINAMATH_CALUDE_wire_sharing_l3508_350814

/-- Given a wire of total length 150 cm, where one person's share is 16 cm shorter than the other's,
    prove that the shorter share is 67 cm. -/
theorem wire_sharing (total_length : ℕ) (difference : ℕ) (seokgi_share : ℕ) (yeseul_share : ℕ) :
  total_length = 150 ∧ difference = 16 ∧ seokgi_share + yeseul_share = total_length ∧ 
  yeseul_share = seokgi_share + difference → seokgi_share = 67 :=
by sorry

end NUMINAMATH_CALUDE_wire_sharing_l3508_350814


namespace NUMINAMATH_CALUDE_air_quality_probability_l3508_350820

theorem air_quality_probability (p_one_day p_two_days : ℝ) 
  (h1 : p_one_day = 0.8)
  (h2 : p_two_days = 0.6) :
  p_two_days / p_one_day = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l3508_350820


namespace NUMINAMATH_CALUDE_twenty_people_handshakes_l3508_350893

/-- The number of unique handshakes in a group where each person shakes hands once with every other person -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 20 people, where each person shakes hands once with every other person, there are 190 unique handshakes -/
theorem twenty_people_handshakes :
  number_of_handshakes 20 = 190 := by
  sorry

#eval number_of_handshakes 20

end NUMINAMATH_CALUDE_twenty_people_handshakes_l3508_350893


namespace NUMINAMATH_CALUDE_isaiah_typing_speed_l3508_350811

theorem isaiah_typing_speed 
  (micah_speed : ℕ) 
  (isaiah_hourly_diff : ℕ) 
  (h1 : micah_speed = 20)
  (h2 : isaiah_hourly_diff = 1200) : 
  (micah_speed * 60 + isaiah_hourly_diff) / 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_isaiah_typing_speed_l3508_350811


namespace NUMINAMATH_CALUDE_reassembled_prism_surface_area_l3508_350886

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℝ
  height : ℝ
  length : ℝ

/-- Represents the cuts made to the prism -/
structure PrismCuts where
  first_cut : ℝ
  second_cut : ℝ
  third_cut : ℝ

/-- Calculates the surface area of the reassembled prism -/
def surface_area_reassembled (dim : PrismDimensions) (cuts : PrismCuts) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the reassembled prism is 16 square feet -/
theorem reassembled_prism_surface_area 
  (dim : PrismDimensions) 
  (cuts : PrismCuts) 
  (h1 : dim.width = 1) 
  (h2 : dim.height = 1) 
  (h3 : dim.length = 2) 
  (h4 : cuts.first_cut = 1/4) 
  (h5 : cuts.second_cut = 1/5) 
  (h6 : cuts.third_cut = 1/6) : 
  surface_area_reassembled dim cuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_reassembled_prism_surface_area_l3508_350886


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3508_350843

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b → (a + b)^2008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3508_350843


namespace NUMINAMATH_CALUDE_initial_crayons_l3508_350848

theorem initial_crayons (crayons_left : ℕ) (crayons_lost : ℕ) 
  (h1 : crayons_left = 134) 
  (h2 : crayons_lost = 345) : 
  crayons_left + crayons_lost = 479 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l3508_350848


namespace NUMINAMATH_CALUDE_pen_pencil_length_difference_l3508_350882

theorem pen_pencil_length_difference :
  ∀ (rubber pen pencil : ℝ),
  pen = rubber + 3 →
  pencil = 12 →
  rubber + pen + pencil = 29 →
  pencil - pen = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_length_difference_l3508_350882


namespace NUMINAMATH_CALUDE_total_books_correct_l3508_350871

/-- Calculates the total number of books after a purchase -/
def total_books (initial : Real) (bought : Real) : Real :=
  initial + bought

/-- Theorem: The total number of books is the sum of initial and bought books -/
theorem total_books_correct (initial : Real) (bought : Real) :
  total_books initial bought = initial + bought := by
  sorry

end NUMINAMATH_CALUDE_total_books_correct_l3508_350871


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3508_350880

/-- Given real numbers m, n, p, q, and functions f and g,
    prove that f(g(x)) = g(f(x)) has a unique solution
    if and only if mq = p and q = n -/
theorem unique_solution_condition (m n p q : ℝ)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x^2 + n)
  (hg : ∀ x, g x = p * x + q) :
  (∃! x, f (g x) = g (f x)) ↔ (m * q = p ∧ q = n) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3508_350880


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3508_350879

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 54 →
  a * d + b * c = 105 →
  c * d = 50 →
  a^2 + b^2 + c^2 + d^2 ≤ 124 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3508_350879


namespace NUMINAMATH_CALUDE_bob_payment_bob_acorn_payment_l3508_350856

theorem bob_payment (alice_acorns : ℕ) (alice_price_per_acorn : ℚ) (alice_bob_price_ratio : ℕ) : ℚ :=
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  alice_total_payment / alice_bob_price_ratio

theorem bob_acorn_payment : bob_payment 3600 15 9 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_bob_payment_bob_acorn_payment_l3508_350856


namespace NUMINAMATH_CALUDE_square_region_perimeter_l3508_350885

/-- Given a region formed by eight congruent squares arranged in a vertical rectangle
    with a total area of 512 square centimeters, the perimeter of the region is 160 centimeters. -/
theorem square_region_perimeter : 
  ∀ (side_length : ℝ),
  side_length > 0 →
  8 * side_length^2 = 512 →
  2 * (7 * side_length + 3 * side_length) = 160 :=
by sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l3508_350885


namespace NUMINAMATH_CALUDE_simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l3508_350881

-- Define polynomials A and B
def A (a b : ℝ) : ℝ := 3*a^2 - 6*a*b + b^2
def B (a b : ℝ) : ℝ := -2*a^2 + 3*a*b - 5*b^2

-- Theorem for the simplified form of A + 2B
theorem simplify_A_plus_2B (a b : ℝ) : A a b + 2 * B a b = -a^2 - 9*b^2 := by sorry

-- Theorem for the value of A + 2B when a = 1 and b = -1
theorem value_A_plus_2B_at_1_neg1 : A 1 (-1) + 2 * B 1 (-1) = -10 := by sorry

end NUMINAMATH_CALUDE_simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l3508_350881


namespace NUMINAMATH_CALUDE_triangle_area_l3508_350863

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (2, 3)

theorem triangle_area : 
  let doubled_a : ℝ × ℝ := (2 * a.1, 2 * a.2)
  (1/2) * |doubled_a.1 * b.2 - doubled_a.2 * b.1| = 14 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3508_350863


namespace NUMINAMATH_CALUDE_min_value_expression_l3508_350836

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2050513 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3508_350836


namespace NUMINAMATH_CALUDE_segment_length_product_l3508_350873

theorem segment_length_product (a : ℝ) : 
  (((3 * a - 7)^2 + (a - 7)^2) = 90) → 
  (∃ b : ℝ, (((3 * b - 7)^2 + (b - 7)^2) = 90) ∧ (a * b = 0.8)) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l3508_350873


namespace NUMINAMATH_CALUDE_symmetric_polynomial_property_l3508_350875

theorem symmetric_polynomial_property (p q r : ℝ) :
  let f := λ x : ℝ => p * x^7 + q * x^3 + r * x - 5
  f (-6) = 3 → f 6 = -13 := by
sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_property_l3508_350875


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3508_350804

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 8 * x * y) :
  1 / x + 1 / y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3508_350804


namespace NUMINAMATH_CALUDE_fraction_sum_and_complex_fraction_l3508_350896

theorem fraction_sum_and_complex_fraction (a b m : ℝ) 
  (h1 : a ≠ b) (h2 : m ≠ 1) (h3 : m ≠ 2) : 
  (a / (a - b) + b / (b - a) = 1) ∧ 
  ((m^2 - 4) / (4 + 4*m + m^2) / ((m - 2) / (2*m - 2)) * ((m + 2) / (m - 1)) = 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_complex_fraction_l3508_350896


namespace NUMINAMATH_CALUDE_tangent_line_circle_sum_constraint_l3508_350807

theorem tangent_line_circle_sum_constraint (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ (x' y' : ℝ), (x' - 1)^2 + (y' - 1)^2 ≤ 1 → 
                               (m + 1) * x' + (n + 1) * y' - 2 ≥ 0) →
  m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_sum_constraint_l3508_350807


namespace NUMINAMATH_CALUDE_largest_divisor_is_60_l3508_350862

def is_largest_divisor (n : ℕ) : Prop :=
  n ∣ 540 ∧ n < 80 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 540 → m < 80 → m ∣ 180 → m ≤ n

theorem largest_divisor_is_60 : is_largest_divisor 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_is_60_l3508_350862


namespace NUMINAMATH_CALUDE_sequence_formula_l3508_350840

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2^(n-1)) :
  ∀ n : ℕ, n > 0 → a n = 2^n - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3508_350840


namespace NUMINAMATH_CALUDE_exam_correct_percentage_l3508_350884

/-- Given an exam with two sections, calculate the percentage of correctly solved problems. -/
theorem exam_correct_percentage (y : ℕ) : 
  let total_problems := 10 * y
  let section1_problems := 6 * y
  let section2_problems := 4 * y
  let missed_section1 := 2 * y
  let missed_section2 := y
  let correct_problems := (section1_problems - missed_section1) + (section2_problems - missed_section2)
  (correct_problems : ℚ) / total_problems * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_percentage_l3508_350884


namespace NUMINAMATH_CALUDE_difference_of_squares_l3508_350838

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3508_350838


namespace NUMINAMATH_CALUDE_line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l3508_350899

-- Define the points M and N
def M : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (2, -3)

-- Define the lines given in the conditions
def line1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def line2 (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the parallel and perpendicular conditions
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem for the first line
theorem line_through_M_parallel_to_line1 :
  ∃ (a b c : ℝ), 
    (a * M.1 + b * M.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x - y - 4 = 0) ∧
    parallel (a / b) 2 :=
sorry

-- Theorem for the second line
theorem line_through_N_perpendicular_to_line2 :
  ∃ (a b c : ℝ), 
    (a * N.1 + b * N.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x + y - 1 = 0) ∧
    perpendicular (a / b) (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l3508_350899
