import Mathlib

namespace linear_combination_passes_through_intersection_l3503_350355

/-- Two distinct linear equations in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The point where two linear equations intersect -/
def intersection (eq1 eq2 : LinearEquation) : ℝ × ℝ :=
  sorry

/-- Checks if a point satisfies a linear equation -/
def satisfies (eq : LinearEquation) (point : ℝ × ℝ) : Prop :=
  eq.a * point.1 + eq.b * point.2 + eq.c = 0

/-- Theorem: For any two distinct linear equations and any real k,
    the equation P(x,y) + k P^1(x,y) = 0 passes through their intersection point -/
theorem linear_combination_passes_through_intersection
  (P P1 : LinearEquation) (k : ℝ) (h : P ≠ P1) :
  let intersect_point := intersection P P1
  satisfies ⟨P.a + k * P1.a, P.b + k * P1.b, P.c + k * P1.c, sorry⟩ intersect_point :=
by
  sorry

end linear_combination_passes_through_intersection_l3503_350355


namespace water_depth_in_tank_l3503_350300

/-- Represents a horizontally placed cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank -/
def water_depths (tank : CylindricalTank) (water_surface_area : ℝ) : Set ℝ :=
  sorry

/-- Theorem stating the depths of water in the given cylindrical tank -/
theorem water_depth_in_tank (tank : CylindricalTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : water_surface_area = 48) :
  water_depths tank water_surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} := by
  sorry

end water_depth_in_tank_l3503_350300


namespace john_tax_difference_l3503_350320

/-- Represents the tax rates and incomes before and after the change -/
structure TaxData where
  old_rate : ℝ
  new_rate : ℝ
  old_income : ℝ
  new_income : ℝ

/-- Calculates the difference in tax payments given the tax data -/
def tax_difference (data : TaxData) : ℝ :=
  data.new_rate * data.new_income - data.old_rate * data.old_income

/-- The specific tax data for John's situation -/
def john_tax_data : TaxData :=
  { old_rate := 0.20
    new_rate := 0.30
    old_income := 1000000
    new_income := 1500000 }

/-- Theorem stating that the difference in John's tax payments is $250,000 -/
theorem john_tax_difference :
  tax_difference john_tax_data = 250000 := by
  sorry

end john_tax_difference_l3503_350320


namespace exam_pass_count_l3503_350365

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed failed : ℕ), 
    passed + failed = total ∧
    passed * passed_avg + failed * failed_avg = total * overall_avg ∧
    passed = 100 := by
  sorry

end exam_pass_count_l3503_350365


namespace marks_change_factor_l3503_350304

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h_n : n = 12) (h_initial : initial_avg = 50) (h_final : final_avg = 100) :
  (final_avg * n) / (initial_avg * n) = 2 := by
  sorry

end marks_change_factor_l3503_350304


namespace root_difference_quadratic_specific_quadratic_root_difference_l3503_350382

theorem root_difference_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  2*a*root1^2 + b*root1 = c ∧
  2*a*root2^2 + b*root2 = c ∧
  root1 ≥ root2 →
  root1 - root2 = Real.sqrt discriminant / a :=
by sorry

theorem specific_quadratic_root_difference :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := 12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  root1 - root2 = 5.5 :=
by sorry

end root_difference_quadratic_specific_quadratic_root_difference_l3503_350382


namespace asset_value_increase_l3503_350372

theorem asset_value_increase (initial_value : ℝ) (h : initial_value > 0) :
  let year1_increase := 0.2
  let year2_increase := 0.3
  let year1_value := initial_value * (1 + year1_increase)
  let year2_value := year1_value * (1 + year2_increase)
  let total_increase := (year2_value - initial_value) / initial_value
  total_increase = 0.56 := by
  sorry

end asset_value_increase_l3503_350372


namespace product_eleven_sum_reciprocal_squares_l3503_350329

theorem product_eleven_sum_reciprocal_squares :
  ∀ a b : ℕ,
  a * b = 11 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 122 / 121 :=
by
  sorry

end product_eleven_sum_reciprocal_squares_l3503_350329


namespace triangle_theorem_l3503_350332

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = t.a^2 + t.b * t.c)
  (h2 : Real.sin t.B = Real.sqrt 3 / 3)
  (h3 : t.b = 2) :
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 2 + Real.sqrt 3) / 2) :=
by sorry

end triangle_theorem_l3503_350332


namespace dana_marcus_difference_l3503_350385

/-- The number of pencils Jayden has -/
def jayden_pencils : ℕ := 20

/-- The number of pencils Dana has -/
def dana_pencils : ℕ := jayden_pencils + 15

/-- The number of pencils Marcus has -/
def marcus_pencils : ℕ := jayden_pencils / 2

/-- Theorem stating that Dana has 25 more pencils than Marcus -/
theorem dana_marcus_difference : dana_pencils - marcus_pencils = 25 := by
  sorry

end dana_marcus_difference_l3503_350385


namespace unique_composite_with_bounded_divisors_l3503_350393

def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def isProperDivisor (d n : ℕ) : Prop :=
  1 < d ∧ d < n ∧ n % d = 0

theorem unique_composite_with_bounded_divisors :
  ∃! n : ℕ, isComposite n ∧
    (∀ d : ℕ, isProperDivisor d n → n - 12 ≥ d ∧ d ≥ n - 20) ∧
    n = 24 :=
by
  sorry

end unique_composite_with_bounded_divisors_l3503_350393


namespace cubic_roots_inequality_l3503_350383

theorem cubic_roots_inequality (A B C : ℝ) 
  (h : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
       ∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ (x = a ∨ x = b ∨ x = c)) :
  A^2 + B^2 + 18*C > 0 := by
sorry

end cubic_roots_inequality_l3503_350383


namespace quadratic_form_equivalence_l3503_350346

theorem quadratic_form_equivalence (k : ℝ) :
  (∃ (a b : ℝ), ∀ (x : ℝ), (3*k - 2)*x*(x + k) + k^2*(k - 1) = (a*x + b)^2) ↔ k = 2 :=
by sorry

end quadratic_form_equivalence_l3503_350346


namespace revenue_calculation_impossible_l3503_350330

structure ShoeInventory where
  large_boots : ℕ
  medium_sandals : ℕ
  small_sneakers : ℕ
  large_sandals : ℕ
  medium_boots : ℕ
  small_boots : ℕ

def initial_stock : ShoeInventory :=
  { large_boots := 22
  , medium_sandals := 32
  , small_sneakers := 24
  , large_sandals := 45
  , medium_boots := 35
  , small_boots := 26 }

def prices : ShoeInventory :=
  { large_boots := 80
  , medium_sandals := 60
  , small_sneakers := 50
  , large_sandals := 65
  , medium_boots := 75
  , small_boots := 55 }

def total_pairs (stock : ShoeInventory) : ℕ :=
  stock.large_boots + stock.medium_sandals + stock.small_sneakers +
  stock.large_sandals + stock.medium_boots + stock.small_boots

def pairs_left : ℕ := 78

theorem revenue_calculation_impossible :
  ∀ (final_stock : ShoeInventory),
    total_pairs final_stock = pairs_left →
    ∃ (revenue₁ revenue₂ : ℕ),
      revenue₁ ≠ revenue₂ ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₁ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₂ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) :=
by sorry

end revenue_calculation_impossible_l3503_350330


namespace range_of_m_for_odd_function_with_conditions_l3503_350370

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (3/2 + x) = f (3/2 - x)) ∧
  (f 5 > -2) ∧
  (∃ m : ℝ, f 2 = m - 3/m)

/-- The range of m for the given function f -/
def RangeOfM (f : ℝ → ℝ) : Set ℝ :=
  {m : ℝ | m < -1 ∨ (0 < m ∧ m < 3)}

/-- Theorem stating the range of m for a function satisfying the given conditions -/
theorem range_of_m_for_odd_function_with_conditions (f : ℝ → ℝ) 
  (h : OddFunctionWithConditions f) : 
  ∃ m : ℝ, f 2 = m - 3/m ∧ m ∈ RangeOfM f := by
  sorry

end range_of_m_for_odd_function_with_conditions_l3503_350370


namespace correct_number_of_choices_l3503_350318

/-- Represents the number of junior boys or girls -/
def num_juniors : ℕ := 7

/-- Represents the number of senior boys or girls -/
def num_seniors : ℕ := 8

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders : ℕ :=
  num_genders * (num_juniors * num_seniors + num_seniors * num_juniors)

/-- Theorem stating that the number of ways to choose leaders is 224 -/
theorem correct_number_of_choices : ways_to_choose_leaders = 224 := by
  sorry

end correct_number_of_choices_l3503_350318


namespace circle_radius_is_sqrt_2_l3503_350319

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

-- Theorem statement
theorem circle_radius_is_sqrt_2 :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2 :=
by sorry

end circle_radius_is_sqrt_2_l3503_350319


namespace four_digit_number_problem_l3503_350325

theorem four_digit_number_problem : ∃! (a b c d : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧
  (a + b = c + d) ∧
  (a + d = c) ∧
  (b + d = 2 * (a + c)) ∧
  (1000 * a + 100 * b + 10 * c + d = 1854) :=
by sorry

#check four_digit_number_problem

end four_digit_number_problem_l3503_350325


namespace exam_score_below_mean_l3503_350347

/-- Given an exam with a mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean
  (mean : ℝ)
  (score_above : ℝ)
  (sd_above : ℝ)
  (sd_below : ℝ)
  (h1 : mean = 74)
  (h2 : score_above = 98)
  (h3 : sd_above = 3)
  (h4 : sd_below = 2)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 58 :=
by sorry

end exam_score_below_mean_l3503_350347


namespace parallel_vectors_m_value_l3503_350343

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end parallel_vectors_m_value_l3503_350343


namespace fraction_to_decimal_l3503_350352

theorem fraction_to_decimal : (5 : ℚ) / 50 = 0.1 := by sorry

end fraction_to_decimal_l3503_350352


namespace power_multiplication_l3503_350310

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end power_multiplication_l3503_350310


namespace houses_before_boom_correct_l3503_350366

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County. -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built_during_boom : ℕ := 574

/-- Theorem stating that the number of houses before the boom
    plus the number of houses built during the boom
    equals the current number of houses. -/
theorem houses_before_boom_correct :
  houses_before_boom + houses_built_during_boom = current_houses :=
by sorry

end houses_before_boom_correct_l3503_350366


namespace allowance_multiple_l3503_350326

theorem allowance_multiple (middle_school_allowance senior_year_allowance x : ℝ) :
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = middle_school_allowance * x + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance = 1.5 →
  x = 2 := by
sorry

end allowance_multiple_l3503_350326


namespace two_digit_number_ratio_l3503_350373

theorem two_digit_number_ratio (a b : ℕ) (h1 : 10 * a + b - (10 * b + a) = 36) (h2 : (a + b) - (a - b) = 8) : 
  a = 2 * b := by
sorry

end two_digit_number_ratio_l3503_350373


namespace tangent_perpendicular_range_l3503_350354

theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a + 1/x = 0) → a > 2 := by
  sorry

end tangent_perpendicular_range_l3503_350354


namespace courier_speed_impossibility_l3503_350384

/-- Proves the impossibility of achieving a specific average speed given certain conditions -/
theorem courier_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_avg_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_avg_speed = 12 →
  ¬∃ (remaining_speed : ℝ),
    remaining_speed > 0 ∧
    (2/3 * total_distance / initial_speed + 1/3 * total_distance / remaining_speed) = (total_distance / target_avg_speed) :=
by sorry

end courier_speed_impossibility_l3503_350384


namespace binary_linear_equation_sum_l3503_350323

/-- A binary linear equation is an equation where the exponents of all variables are 1. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y, f x y = a * x + b * y + c

/-- Given that x^(3m-3) - 2y^(n-1) = 5 is a binary linear equation, prove that m + n = 10/3 -/
theorem binary_linear_equation_sum (m n : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(3*m-3) - 2*y^(n-1) - 5) →
  m + n = 10/3 :=
by sorry

end binary_linear_equation_sum_l3503_350323


namespace min_value_quadratic_l3503_350367

theorem min_value_quadratic (x y : ℝ) : 
  3 * x^2 + 2 * x * y + y^2 - 6 * x + 2 * y + 8 ≥ -1 :=
by sorry

end min_value_quadratic_l3503_350367


namespace pythagorean_consecutive_naturals_l3503_350395

theorem pythagorean_consecutive_naturals :
  ∀ x y z : ℕ, y = x + 1 → z = x + 2 →
  (z^2 = y^2 + x^2 ↔ x = 3 ∧ y = 4 ∧ z = 5) :=
by sorry

end pythagorean_consecutive_naturals_l3503_350395


namespace fraction_decomposition_l3503_350344

theorem fraction_decomposition (n : ℕ) 
  (h1 : ∀ n, 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1))
  (h2 : ∀ n, 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))) :
  1 / (n * (n + 1) * (n + 2) * (n + 3)) = 
    1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end fraction_decomposition_l3503_350344


namespace odd_function_value_l3503_350353

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 2 then a * Real.log x - a * x + 1 else 0

-- State the theorem
theorem odd_function_value (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 2, f a x = a * Real.log x - a * x + 1) →  -- definition for x ∈ (0, 2)
  (∃ c ∈ Set.Ioo (-2) 0, ∀ x ∈ Set.Ioo (-2) 0, f a x ≥ f a c) →  -- minimum value exists in (-2, 0)
  (∃ c ∈ Set.Ioo (-2) 0, f a c = 1) →  -- minimum value is 1
  a = 2 :=
by sorry

end odd_function_value_l3503_350353


namespace energy_conservation_train_ball_system_energy_changes_specific_scenario_l3503_350398

/-- Represents the velocity of an object -/
structure Velocity where
  value : ℝ
  unit : String

/-- Represents the kinetic energy of an object -/
structure KineticEnergy where
  value : ℝ
  unit : String

/-- Represents a physical system consisting of a train and a ball -/
structure TrainBallSystem where
  trainVelocity : Velocity
  ballMass : ℝ
  ballThrowingVelocity : Velocity

/-- Calculates the kinetic energy of an object given its mass and velocity -/
def calculateKineticEnergy (mass : ℝ) (velocity : Velocity) : KineticEnergy :=
  { value := 0.5 * mass * velocity.value ^ 2, unit := "J" }

/-- Theorem: Energy conservation in the train-ball system -/
theorem energy_conservation_train_ball_system
  (system : TrainBallSystem)
  (initial_train_energy : KineticEnergy)
  (initial_ball_energy : KineticEnergy)
  (final_ball_energy_forward : KineticEnergy)
  (final_ball_energy_backward : KineticEnergy) :
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_forward.value) ∧
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_backward.value) :=
by sorry

/-- Corollary: Specific energy changes for the given scenario -/
theorem energy_changes_specific_scenario
  (system : TrainBallSystem)
  (h_train_velocity : system.trainVelocity.value = 60 ∧ system.trainVelocity.unit = "km/hour")
  (h_ball_velocity : system.ballThrowingVelocity.value = 60 ∧ system.ballThrowingVelocity.unit = "km/hour")
  (initial_ball_energy : KineticEnergy)
  (h_forward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value + system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 4 * initial_ball_energy.value, unit := initial_ball_energy.unit })
  (h_backward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value - system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 0, unit := initial_ball_energy.unit }) :
  ∃ (compensating_energy : KineticEnergy),
    compensating_energy.value = 3 * initial_ball_energy.value ∧
    compensating_energy.value = initial_ball_energy.value :=
by sorry

end energy_conservation_train_ball_system_energy_changes_specific_scenario_l3503_350398


namespace f_satisfies_conditions_l3503_350321

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem f_satisfies_conditions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0) :=
by sorry

end f_satisfies_conditions_l3503_350321


namespace rick_irons_31_clothes_l3503_350351

/-- Calculates the total number of clothes ironed by Rick -/
def totalClothesIroned (shirtsPerHour dressShirtsHours pantsPerHour dressPantsHours jacketsPerHour jacketsHours : ℕ) : ℕ :=
  shirtsPerHour * dressShirtsHours + pantsPerHour * dressPantsHours + jacketsPerHour * jacketsHours

/-- Proves that Rick irons 31 pieces of clothing given the conditions -/
theorem rick_irons_31_clothes :
  totalClothesIroned 4 3 3 5 2 2 = 31 := by
  sorry

#eval totalClothesIroned 4 3 3 5 2 2

end rick_irons_31_clothes_l3503_350351


namespace inequality_proof_l3503_350335

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b * 2^a + a * 2^(-b) ≥ a + b := by
  sorry

end inequality_proof_l3503_350335


namespace regression_prediction_at_2_l3503_350356

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ
  c : ℝ := 0.2

/-- Calculates the y value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.c

/-- Theorem: Given the conditions, the predicted y value when x = 2 is 2.6 -/
theorem regression_prediction_at_2 
  (model : LinearRegression)
  (h₁ : predict model 4 = 5) -- condition: ȳ = 5 when x̄ = 4
  (h₂ : model.c = 0.2) -- condition: intercept is 0.2
  : predict model 2 = 2.6 := by
  sorry

end regression_prediction_at_2_l3503_350356


namespace percentage_relation_l3503_350397

theorem percentage_relation (a b : ℝ) (h1 : a - b = 1650) (h2 : a = 2475) (h3 : b = 825) :
  (7.5 / 100) * a = (22.5 / 100) * b := by
  sorry

end percentage_relation_l3503_350397


namespace jolene_washed_five_cars_l3503_350303

/-- The number of cars Jolene washed to raise money for a bicycle -/
def cars_washed (families : ℕ) (babysitting_rate : ℕ) (car_wash_rate : ℕ) (total_raised : ℕ) : ℕ :=
  (total_raised - families * babysitting_rate) / car_wash_rate

/-- Theorem: Jolene washed 5 cars given the problem conditions -/
theorem jolene_washed_five_cars :
  cars_washed 4 30 12 180 = 5 := by
  sorry

end jolene_washed_five_cars_l3503_350303


namespace incorrect_proposition_l3503_350349

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end incorrect_proposition_l3503_350349


namespace intersection_M_N_l3503_350316

def M : Set ℝ := {x | Real.log (x + 1) > 0}
def N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end intersection_M_N_l3503_350316


namespace cosine_ratio_equals_one_l3503_350399

theorem cosine_ratio_equals_one (c : ℝ) (h : c = 2 * Real.pi / 7) :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) /
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 := by
  sorry

end cosine_ratio_equals_one_l3503_350399


namespace cutting_tool_geometry_l3503_350376

theorem cutting_tool_geometry (A B C : ℝ × ℝ) : 
  let r : ℝ := 6
  let AB : ℝ := 5
  let BC : ℝ := 3
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 →
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  A.1^2 + A.2^2 = r^2 →
  B.1^2 + B.2^2 = r^2 →
  C.1^2 + C.2^2 = r^2 →
  (B.1^2 + B.2^2 = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) := by
sorry

end cutting_tool_geometry_l3503_350376


namespace outside_trash_count_l3503_350336

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end outside_trash_count_l3503_350336


namespace emilys_dogs_l3503_350309

theorem emilys_dogs (food_per_dog_per_day : ℕ) (vacation_days : ℕ) (total_food_kg : ℕ) :
  food_per_dog_per_day = 250 →
  vacation_days = 14 →
  total_food_kg = 14 →
  (total_food_kg * 1000) / (food_per_dog_per_day * vacation_days) = 4 :=
by sorry

end emilys_dogs_l3503_350309


namespace product_evaluation_l3503_350306

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end product_evaluation_l3503_350306


namespace complex_fraction_equality_l3503_350342

theorem complex_fraction_equality : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l3503_350342


namespace p_necessary_not_sufficient_for_q_l3503_350313

-- Define geometric bodies
structure GeometricBody where
  height : ℝ
  crossSectionalArea : ℝ → ℝ
  volume : ℝ

-- Define the Gougu Principle
def gougu_principle (A B : GeometricBody) : Prop :=
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q (A B : GeometricBody) :
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume ∧
  ¬(A.volume = B.volume →
    ∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l3503_350313


namespace smallest_iteration_for_three_l3503_350338

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 ∧ x % 7 = 0 then x / 14
  else if x % 7 = 0 then 2 * x
  else if x % 2 = 0 then 7 * x
  else x + 2

def f_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem smallest_iteration_for_three :
  (∀ a : ℕ, 1 < a → a < 6 → f_iter a 3 ≠ f 3) ∧
  f_iter 6 3 = f 3 :=
sorry

end smallest_iteration_for_three_l3503_350338


namespace probability_at_least_two_A_plus_specific_l3503_350375

def probability_at_least_two_A_plus (p_physics : ℚ) (p_chemistry : ℚ) (p_politics : ℚ) : ℚ :=
  let p_not_physics := 1 - p_physics
  let p_not_chemistry := 1 - p_chemistry
  let p_not_politics := 1 - p_politics
  p_physics * p_chemistry * p_not_politics +
  p_physics * p_not_chemistry * p_politics +
  p_not_physics * p_chemistry * p_politics +
  p_physics * p_chemistry * p_politics

theorem probability_at_least_two_A_plus_specific :
  probability_at_least_two_A_plus (7/8) (3/4) (5/12) = 151/192 :=
by sorry

end probability_at_least_two_A_plus_specific_l3503_350375


namespace prob_3_heads_12_coins_value_l3503_350348

/-- The probability of getting exactly 3 heads when flipping 12 coins -/
def prob_3_heads_12_coins : ℚ :=
  (Nat.choose 12 3 : ℚ) / 2^12

/-- Theorem stating that the probability of getting exactly 3 heads
    when flipping 12 coins is equal to 220/4096 -/
theorem prob_3_heads_12_coins_value :
  prob_3_heads_12_coins = 220 / 4096 := by
  sorry

end prob_3_heads_12_coins_value_l3503_350348


namespace symmetric_points_sum_power_l3503_350387

def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_sum_power (m n : ℝ) :
  symmetric_about_y_axis m 3 4 n →
  (m + n)^2015 = -1 := by
  sorry

end symmetric_points_sum_power_l3503_350387


namespace product_digit_sum_is_nine_l3503_350380

/-- Represents a strictly increasing sequence of 5 digits -/
def StrictlyIncreasingDigits (a b c d e : Nat) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum_is_nine 
  (a b c d e : Nat) 
  (h : StrictlyIncreasingDigits a b c d e) : 
  sumOfDigits (9 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e)) = 9 := by
  sorry

end product_digit_sum_is_nine_l3503_350380


namespace fifteen_fishers_tomorrow_l3503_350317

/-- Represents the fishing schedule in the coastal village --/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow given the fishing schedule --/
def fishersTomorrow (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow --/
theorem fifteen_fishers_tomorrow :
  let schedule := FishingSchedule.mk 7 8 3 12 10
  fishersTomorrow schedule = 15 := by
  sorry

end fifteen_fishers_tomorrow_l3503_350317


namespace min_value_sum_of_reciprocals_l3503_350374

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  (1 / (1 + 1^n) + 1 / (1 + 1^n) = 1) :=
by sorry

end min_value_sum_of_reciprocals_l3503_350374


namespace cab_driver_income_l3503_350334

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 600)
  (h3 : day3 = 450)
  (h4 : day4 = 400)
  (h5 : day5 = 800)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 500) :
  day2 = 250 := by
sorry

end cab_driver_income_l3503_350334


namespace quadratic_real_roots_range_l3503_350305

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - x - m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Theorem statement
theorem quadratic_real_roots_range (m : ℝ) : has_real_roots m → m ≥ -1/4 := by
  sorry

end quadratic_real_roots_range_l3503_350305


namespace circle_diameter_ratio_l3503_350311

theorem circle_diameter_ratio (C D : Real) :
  -- Circle C is inside circle D
  C < D →
  -- Diameter of circle D is 20 cm
  D = 10 →
  -- Ratio of shaded area to area of circle C is 7:1
  (π * D^2 - π * C^2) / (π * C^2) = 7 →
  -- The diameter of circle C is 5√5 cm
  2 * C = 5 * Real.sqrt 5 := by
sorry

end circle_diameter_ratio_l3503_350311


namespace point_transformation_l3503_350360

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180° clockwise around (2,3) -/
def rotate180 (p : Point) : Point :=
  { x := 4 - p.x, y := 6 - p.y }

/-- Reflects a point about the line y = x -/
def reflectAboutYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Translates a point by the vector (4, -2) -/
def translate (p : Point) : Point :=
  { x := p.x + 4, y := p.y - 2 }

/-- The main theorem -/
theorem point_transformation (Q : Point) :
  (translate (reflectAboutYEqualsX (rotate180 Q)) = Point.mk 1 6) →
  (Q.y - Q.x = 13) := by
  sorry

end point_transformation_l3503_350360


namespace bridge_length_l3503_350340

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time_s - train_length = 275 :=
by sorry

end bridge_length_l3503_350340


namespace last_digit_of_n_is_five_l3503_350350

def sum_powers (n : ℕ) : ℕ := (Finset.range (2*n - 2)).sum (λ i => n^(i + 1))

theorem last_digit_of_n_is_five (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (sum_powers n - 4)) :
  n % 10 = 5 := by
  sorry

end last_digit_of_n_is_five_l3503_350350


namespace divisibility_problem_l3503_350361

theorem divisibility_problem :
  (∃ (a b : Nat), a < 10 ∧ b < 10 ∧
    (∀ n : Nat, 73 ∣ (10 * a + b) * 10^n + (200 * 10^n + 79) / 9)) ∧
  (¬ ∃ (c d : Nat), c < 10 ∧ d < 10 ∧
    (∀ n : Nat, 79 ∣ (10 * c + d) * 10^n + (200 * 10^n + 79) / 9)) :=
by sorry

end divisibility_problem_l3503_350361


namespace complex_number_real_imag_equal_l3503_350362

theorem complex_number_real_imag_equal (b : ℝ) : 
  let z : ℂ := (1 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -3 := by sorry

end complex_number_real_imag_equal_l3503_350362


namespace import_tax_problem_l3503_350377

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount. -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the specific values in the problem. -/
theorem import_tax_problem :
  let total_value : ℚ := 2560
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 109.2
  sorry


end import_tax_problem_l3503_350377


namespace child_admission_price_l3503_350358

-- Define the given conditions
def total_people : ℕ := 610
def adult_price : ℚ := 2
def total_receipts : ℚ := 960
def num_adults : ℕ := 350

-- Define the admission price for children
def child_price : ℚ := 1

-- Theorem to prove
theorem child_admission_price :
  child_price * (total_people - num_adults) + adult_price * num_adults = total_receipts :=
sorry

end child_admission_price_l3503_350358


namespace CH₄_has_most_atoms_l3503_350363

-- Define the molecules and their atom counts
def O₂_atoms : ℕ := 2
def NH₃_atoms : ℕ := 4
def CO_atoms : ℕ := 2
def CH₄_atoms : ℕ := 5

-- Define a function to compare atom counts
def has_more_atoms (a b : ℕ) : Prop := a > b

-- Theorem statement
theorem CH₄_has_most_atoms :
  has_more_atoms CH₄_atoms O₂_atoms ∧
  has_more_atoms CH₄_atoms NH₃_atoms ∧
  has_more_atoms CH₄_atoms CO_atoms :=
by sorry

end CH₄_has_most_atoms_l3503_350363


namespace wrapping_paper_fraction_l3503_350381

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (fraction_per_present : ℚ) :
  total_used = 1/2 →
  num_presents = 5 →
  total_used = fraction_per_present * num_presents →
  fraction_per_present = 1/10 := by
  sorry

end wrapping_paper_fraction_l3503_350381


namespace expression_factorization_l3503_350345

theorem expression_factorization (y : ℝ) : 
  (12 * y^6 + 35 * y^4 - 5) - (2 * y^6 - 4 * y^4 + 5) = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end expression_factorization_l3503_350345


namespace total_fence_cost_l3503_350324

/-- Represents the cost of building a fence for a pentagonal plot -/
def fence_cost (a b c d e : ℕ) (pa pb pc pd pe : ℕ) : ℕ :=
  a * pa + b * pb + c * pc + d * pd + e * pe

/-- Theorem stating the total cost of the fence -/
theorem total_fence_cost :
  fence_cost 9 12 15 11 13 45 55 60 50 65 = 3360 := by
  sorry

end total_fence_cost_l3503_350324


namespace set_operations_and_equality_l3503_350364

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem set_operations_and_equality :
  (∃ m : ℝ, 
    (A ∩ B m = {x | 3 ≤ x ∧ x ≤ 5} ∧
     (Set.univ \ A) ∪ B m = {x | x < 2 ∨ x ≥ 3})) ∧
  (∀ m : ℝ, A = B m ↔ 2 ≤ m ∧ m ≤ 3) := by sorry

end set_operations_and_equality_l3503_350364


namespace inscribed_right_isosceles_hypotenuse_l3503_350371

/-- Represents a triangle with a given base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a right isosceles triangle inscribed in another triangle -/
structure InscribedRightIsoscelesTriangle where
  outer : Triangle
  hypotenuse : ℝ

/-- The hypotenuse of an inscribed right isosceles triangle in a 30x10 triangle is 12 -/
theorem inscribed_right_isosceles_hypotenuse 
  (t : Triangle) 
  (i : InscribedRightIsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.height = 10) 
  (h3 : i.outer = t) : 
  i.hypotenuse = 12 := by
  sorry

end inscribed_right_isosceles_hypotenuse_l3503_350371


namespace min_values_xy_and_x_plus_y_l3503_350369

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 4/x + 3/y = 1) : 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4/x₀ + 3/y₀ = 1 ∧ x₀ * y₀ = 48 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' * y' ≥ 48) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 4/x₁ + 3/y₁ = 1 ∧ x₁ + y₁ = 7 + 4 * Real.sqrt 3 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' + y' ≥ 7 + 4 * Real.sqrt 3) := by
  sorry

end min_values_xy_and_x_plus_y_l3503_350369


namespace orange_bows_count_l3503_350392

theorem orange_bows_count (total : ℕ) (black : ℕ) : 
  black = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 12 + (black : ℚ) / total = 1 →
  (1 : ℚ) / 12 * total = 10 :=
by sorry

end orange_bows_count_l3503_350392


namespace polygon_with_five_triangles_is_heptagon_l3503_350301

/-- The number of triangles formed by diagonals from one vertex in an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ := n - 2

theorem polygon_with_five_triangles_is_heptagon (n : ℕ) :
  (n ≥ 3) → (triangles_from_diagonals n = 5) → n = 7 := by
  sorry

end polygon_with_five_triangles_is_heptagon_l3503_350301


namespace perfect_square_trinomial_condition_l3503_350302

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 4 (-2 * k) 1 → k = 2 ∨ k = -2 := by
  sorry

end perfect_square_trinomial_condition_l3503_350302


namespace magnificent_class_size_l3503_350307

theorem magnificent_class_size :
  ∀ (girls boys chocolates_given : ℕ),
    girls + boys = 33 →
    boys = girls + 3 →
    girls * girls + boys * boys = chocolates_given →
    chocolates_given = 540 - 12 →
    True :=
by
  sorry

end magnificent_class_size_l3503_350307


namespace calculation_result_l3503_350327

theorem calculation_result : 
  let initial := 180
  let percentage := 35 / 100
  let first_calc := initial * percentage
  let one_third_less := first_calc - (1 / 3 * first_calc)
  let remaining := initial - one_third_less
  let three_fifths := 3 / 5 * remaining
  (three_fifths ^ 2) = 6857.84 := by sorry

end calculation_result_l3503_350327


namespace expression_evaluation_l3503_350391

theorem expression_evaluation :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end expression_evaluation_l3503_350391


namespace jeanne_ticket_purchase_l3503_350357

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets (ferris_wheel_cost roller_coaster_cost bumper_cars_cost current_tickets : ℕ) : ℕ :=
  ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost - current_tickets

theorem jeanne_ticket_purchase : additional_tickets 5 4 4 5 = 8 := by
  sorry

end jeanne_ticket_purchase_l3503_350357


namespace cosine_odd_function_phi_l3503_350312

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.cos (x + φ + π/3)) → φ = π/6 ∨ ∃ k : ℤ, φ = k * π + π/6 :=
by sorry

end cosine_odd_function_phi_l3503_350312


namespace number_problem_l3503_350331

theorem number_problem (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end number_problem_l3503_350331


namespace direct_variation_problem_l3503_350359

/-- A function representing direct variation between z and w -/
def directVariation (k : ℝ) (w : ℝ) : ℝ := k * w

theorem direct_variation_problem (k : ℝ) :
  (directVariation k 5 = 10) →
  (directVariation k 15 = 30) :=
by
  sorry

#check direct_variation_problem

end direct_variation_problem_l3503_350359


namespace train_platform_length_equality_l3503_350396

/-- Prove that the length of the platform is equal to the length of the train -/
theorem train_platform_length_equality 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (train_length : ℝ) 
  (h1 : train_speed = 90 * 1000 / 60) -- 90 km/hr converted to m/min
  (h2 : crossing_time = 1) -- 1 minute
  (h3 : train_length = 750) -- 750 meters
  : train_length = train_speed * crossing_time - train_length := by
  sorry

end train_platform_length_equality_l3503_350396


namespace quadrangular_prism_angles_l3503_350314

/-- A quadrangular prism with specific geometric properties -/
structure QuadrangularPrism where
  -- Base angles
  angleASB : ℝ
  angleDCS : ℝ
  -- Dihedral angle between SAD and SBC
  dihedralAngle : ℝ

/-- The theorem stating the possible angle measures in the quadrangular prism -/
theorem quadrangular_prism_angles (prism : QuadrangularPrism)
  (h1 : prism.angleASB = π/6)  -- 30°
  (h2 : prism.angleDCS = π/4)  -- 45°
  (h3 : prism.dihedralAngle = π/3)  -- 60°
  : (∃ (angleBSC angleASD : ℝ),
      (angleBSC = π/2 ∧ angleASD = π - Real.arccos (Real.sqrt 3 / 2)) ∨
      (angleBSC = Real.arccos (2 * Real.sqrt 2 / 3) ∧ 
       angleASD = Real.arccos (5 * Real.sqrt 3 / 9))) := by
  sorry


end quadrangular_prism_angles_l3503_350314


namespace greatest_common_factor_45_75_90_l3503_350394

theorem greatest_common_factor_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end greatest_common_factor_45_75_90_l3503_350394


namespace no_fixed_points_iff_a_in_range_l3503_350322

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + a*x + 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a x : ℝ) : Prop := f a x = x

-- Theorem statement
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬(is_fixed_point a x)) ↔ (-1 < a ∧ a < 3) :=
sorry

end no_fixed_points_iff_a_in_range_l3503_350322


namespace common_chord_equation_l3503_350315

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The equation of the common chord -/
def common_chord (x y : ℝ) : Prop := x - 2*y + 5 = 0

/-- Theorem stating that the common chord of the two circles is x - 2y + 5 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end common_chord_equation_l3503_350315


namespace train_acceleration_time_l3503_350388

/-- Proves that a train starting from rest, accelerating uniformly at 3 m/s², 
    and traveling a distance of 27 m takes sqrt(18) seconds. -/
theorem train_acceleration_time : ∀ (s a : ℝ),
  s = 27 →  -- distance traveled
  a = 3 →   -- acceleration rate
  ∃ t : ℝ,
    s = (1/2) * a * t^2 ∧  -- kinematic equation for uniform acceleration from rest
    t = Real.sqrt 18 := by
  sorry

end train_acceleration_time_l3503_350388


namespace painting_time_theorem_l3503_350390

def grace_rate : ℚ := 1 / 6
def henry_rate : ℚ := 1 / 8
def julia_rate : ℚ := 1 / 12
def grace_break : ℚ := 1
def henry_break : ℚ := 1
def julia_break : ℚ := 2

theorem painting_time_theorem :
  ∃ t : ℚ, t > 0 ∧ (grace_rate + henry_rate + julia_rate) * (t - 2) = 1 ∧ t = 14 / 3 := by
  sorry

end painting_time_theorem_l3503_350390


namespace weight_ratio_after_loss_l3503_350389

def jakes_current_weight : ℝ := 152
def combined_weight : ℝ := 212
def weight_loss : ℝ := 32

def sisters_weight : ℝ := combined_weight - jakes_current_weight
def jakes_new_weight : ℝ := jakes_current_weight - weight_loss

theorem weight_ratio_after_loss : 
  jakes_new_weight / sisters_weight = 2 := by sorry

end weight_ratio_after_loss_l3503_350389


namespace intersection_A_complement_B_union_A_B_equals_B_l3503_350328

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A (-2) ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem union_A_B_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end intersection_A_complement_B_union_A_B_equals_B_l3503_350328


namespace original_raw_silk_amount_l3503_350308

/-- Given information about silk drying process, prove the original amount of raw silk. -/
theorem original_raw_silk_amount 
  (initial_wet : ℚ) 
  (water_loss : ℚ) 
  (final_dry : ℚ) 
  (h1 : initial_wet = 30) 
  (h2 : water_loss = 3) 
  (h3 : final_dry = 12) : 
  (initial_wet * final_dry) / (initial_wet - water_loss) = 40 / 3 := by
  sorry

#check original_raw_silk_amount

end original_raw_silk_amount_l3503_350308


namespace school_population_theorem_l3503_350378

theorem school_population_theorem (b g t : ℕ) : 
  b = 4 * g → g = 10 * t → b + g + t = 51 * t := by sorry

end school_population_theorem_l3503_350378


namespace ninth_term_of_sequence_l3503_350368

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_9 = 17 -/
theorem ninth_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : a 9 = 17 := by
  sorry

end ninth_term_of_sequence_l3503_350368


namespace quadratic_equation_solution_l3503_350379

theorem quadratic_equation_solution (t s : ℝ) : t = 15 * s^2 + 5 → t = 20 → s = 1 ∨ s = -1 := by
  sorry

end quadratic_equation_solution_l3503_350379


namespace adjacent_knights_probability_l3503_350341

def total_knights : ℕ := 30
def chosen_knights : ℕ := 3

def probability_adjacent_knights : ℚ :=
  1 - (27 * 25 * 23) / (3 * total_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 34 / 35 := by sorry

end adjacent_knights_probability_l3503_350341


namespace reflect_x_of_P_l3503_350339

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The given point P -/
def P : Point :=
  { x := -1, y := 2 }

/-- Theorem: The x-axis reflection of P(-1, 2) is (-1, -2) -/
theorem reflect_x_of_P : reflect_x P = { x := -1, y := -2 } := by
  sorry

end reflect_x_of_P_l3503_350339


namespace quadratic_real_roots_l3503_350337

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0) ↔ k ≤ 1 := by
sorry

end quadratic_real_roots_l3503_350337


namespace memorial_visitors_equation_l3503_350333

theorem memorial_visitors_equation (x : ℕ) (h1 : x + (2 * x + 56) = 589) : 2 * x + 56 = 589 - x := by
  sorry

end memorial_visitors_equation_l3503_350333


namespace probability_same_color_l3503_350386

def green_balls : ℕ := 8
def white_balls : ℕ := 6
def red_balls : ℕ := 5
def blue_balls : ℕ := 4

def total_balls : ℕ := green_balls + white_balls + red_balls + blue_balls

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def total_combinations : ℕ := choose total_balls 3

def same_color_combinations : ℕ := 
  choose green_balls 3 + choose white_balls 3 + choose red_balls 3 + choose blue_balls 3

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 90 / 1771 := by sorry

end probability_same_color_l3503_350386
