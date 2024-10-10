import Mathlib

namespace wizard_elixir_combinations_l189_18939

/-- Represents the number of magical herbs -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical stones -/
def num_stones : ℕ := 6

/-- Represents the number of herbs incompatible with one specific stone -/
def incompatible_herbs : ℕ := 3

/-- Calculates the number of valid combinations for the wizard's elixir -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

/-- Proves that the number of valid combinations for the wizard's elixir is 21 -/
theorem wizard_elixir_combinations : valid_combinations = 21 := by
  sorry

end wizard_elixir_combinations_l189_18939


namespace house_transaction_result_l189_18952

theorem house_transaction_result (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  gain_percent = 0.20 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = -240 := by
sorry

end house_transaction_result_l189_18952


namespace dimitri_weekly_calorie_intake_l189_18993

/-- Represents the daily calorie intake from burgers -/
def daily_calorie_intake (burger_a_calories burger_b_calories burger_c_calories : ℕ) 
  (burger_a_count burger_b_count burger_c_count : ℕ) : ℕ :=
  burger_a_calories * burger_a_count + 
  burger_b_calories * burger_b_count + 
  burger_c_calories * burger_c_count

/-- Calculates the weekly calorie intake based on daily intake -/
def weekly_calorie_intake (daily_intake : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_intake * days_in_week

/-- Theorem stating Dimitri's weekly calorie intake from burgers -/
theorem dimitri_weekly_calorie_intake : 
  weekly_calorie_intake 
    (daily_calorie_intake 350 450 550 2 1 3) 
    7 = 19600 := by
  sorry


end dimitri_weekly_calorie_intake_l189_18993


namespace function_inequality_implies_parameter_bound_l189_18994

open Real

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → f x = a / x + x * log x) →
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → g x = x^3 - x^2 - 5) →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (1/2 : ℝ) 2 → x₂ ∈ Set.Icc (1/2 : ℝ) 2 → f x₁ - g x₂ ≥ 2) →
  a ≥ 1 :=
by sorry

end function_inequality_implies_parameter_bound_l189_18994


namespace square_division_problem_l189_18998

theorem square_division_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 100 ∧ x/y = 4/3 ∧ x = 8 ∧ y = 6 := by
  sorry

end square_division_problem_l189_18998


namespace area_units_order_l189_18987

/-- An enumeration of area units -/
inductive AreaUnit
  | SquareKilometer
  | Hectare
  | SquareMeter
  | SquareDecimeter
  | SquareCentimeter

/-- A function to compare two area units -/
def areaUnitLarger (a b : AreaUnit) : Prop :=
  match a, b with
  | AreaUnit.SquareKilometer, _ => a ≠ b
  | AreaUnit.Hectare, AreaUnit.SquareKilometer => False
  | AreaUnit.Hectare, _ => a ≠ b
  | AreaUnit.SquareMeter, AreaUnit.SquareKilometer => False
  | AreaUnit.SquareMeter, AreaUnit.Hectare => False
  | AreaUnit.SquareMeter, _ => a ≠ b
  | AreaUnit.SquareDecimeter, AreaUnit.SquareCentimeter => True
  | AreaUnit.SquareDecimeter, _ => False
  | AreaUnit.SquareCentimeter, _ => False

/-- Theorem stating the correct order of area units from largest to smallest -/
theorem area_units_order :
  areaUnitLarger AreaUnit.SquareKilometer AreaUnit.Hectare ∧
  areaUnitLarger AreaUnit.Hectare AreaUnit.SquareMeter ∧
  areaUnitLarger AreaUnit.SquareMeter AreaUnit.SquareDecimeter ∧
  areaUnitLarger AreaUnit.SquareDecimeter AreaUnit.SquareCentimeter :=
sorry

end area_units_order_l189_18987


namespace nonagon_diagonals_l189_18905

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l189_18905


namespace length_of_AE_l189_18932

/-- Given a coordinate grid where:
    - A is at (0,4)
    - B is at (7,0)
    - C is at (3,0)
    - D is at (5,3)
    - Line segment AB meets line segment CD at point E
    Prove that the length of segment AE is (7√65)/13 -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (7, 0) →
  C = (3, 0) →
  D = (5, 3) →
  E ∈ Set.Icc A B →
  E ∈ Set.Icc C D →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = (7 * Real.sqrt 65) / 13 :=
by sorry

end length_of_AE_l189_18932


namespace first_half_speed_l189_18953

/-- Proves that given a 6-hour journey where the second half is traveled at 48 kmph
    and the total distance is 324 km, the speed during the first half must be 60 kmph. -/
theorem first_half_speed (total_time : ℝ) (second_half_speed : ℝ) (total_distance : ℝ)
    (h1 : total_time = 6)
    (h2 : second_half_speed = 48)
    (h3 : total_distance = 324) :
    let first_half_time := total_time / 2
    let second_half_time := total_time / 2
    let second_half_distance := second_half_speed * second_half_time
    let first_half_distance := total_distance - second_half_distance
    let first_half_speed := first_half_distance / first_half_time
    first_half_speed = 60 := by
  sorry

#check first_half_speed

end first_half_speed_l189_18953


namespace no_valid_box_dimensions_l189_18927

theorem no_valid_box_dimensions :
  ¬∃ (a b c : ℕ), 
    (Prime a) ∧ (Prime b) ∧ (Prime c) ∧
    (a ≤ b) ∧ (b ≤ c) ∧
    (a * b * c = 2 * (a * b + b * c + a * c)) ∧
    (Prime (a + b + c)) :=
by sorry

end no_valid_box_dimensions_l189_18927


namespace common_tangent_range_l189_18949

/-- The range of parameter a for which the curves y = ln x + 1 and y = x² + x + 3a have a common tangent line -/
theorem common_tangent_range :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (1 / x₁ = 2 * x₂ + 1) ∧
    (Real.log x₁ + 1 = x₂^2 + x₂ + 3 * a) ∧
    (Real.log x₁ + x₂^2 = 3 * a)) ↔
  a ≥ (1 - 4 * Real.log 2) / 12 :=
by sorry

end common_tangent_range_l189_18949


namespace modified_cube_edge_count_l189_18962

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  12 * 3 -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with cubes of side length 2 removed from each corner has 36 edges -/
theorem modified_cube_edge_count :
  ∀ (cube : ModifiedCube),
    cube.originalSideLength = 4 →
    cube.removedCubeSideLength = 2 →
    edgeCount cube = 36 :=
by
  sorry

end modified_cube_edge_count_l189_18962


namespace three_propositions_true_l189_18923

-- Define the properties of functions
def IsConstant (f : ℝ → ℝ) : Prop := ∃ C : ℝ, ∀ x : ℝ, f x = C
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Define the propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsConstant f → (IsOdd f ∧ IsEven f)
def Prop2 (f : ℝ → ℝ) : Prop := IsOdd f → HasInverse f
def Prop3 (f : ℝ → ℝ) : Prop := IsOdd f → IsOdd (λ x => Real.sin (f x))
def Prop4 (f g : ℝ → ℝ) : Prop := IsOdd f → IsEven g → IsEven (g ∘ f)

-- The main theorem
theorem three_propositions_true :
  ∃ (f g : ℝ → ℝ),
    (Prop1 f ∧ ¬Prop2 f ∧ Prop3 f ∧ Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ ¬Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ ¬Prop3 f ∧ Prop4 f g) ∨
    (¬Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ Prop4 f g) :=
  sorry

end three_propositions_true_l189_18923


namespace sqrt_equation_solution_l189_18907

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9 * x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 33 := by
sorry

end sqrt_equation_solution_l189_18907


namespace sin_inequalities_l189_18908

theorem sin_inequalities (x : ℝ) (h : x > 0) :
  (Real.sin x ≤ x) ∧
  (Real.sin x ≥ x - x^3 / 6) ∧
  (Real.sin x ≤ x - x^3 / 6 + x^5 / 120) ∧
  (Real.sin x ≥ x - x^3 / 6 + x^5 / 120 - x^7 / 5040) := by
  sorry

end sin_inequalities_l189_18908


namespace athlete_formation_problem_l189_18974

theorem athlete_formation_problem :
  ∃ n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    n = 226 := by
  sorry

end athlete_formation_problem_l189_18974


namespace divisor_count_fourth_power_l189_18995

theorem divisor_count_fourth_power (x d : ℕ) : 
  (∃ n : ℕ, x = n^4) → 
  (d = (Finset.filter (· ∣ x) (Finset.range (x + 1))).card) →
  d ≡ 1 [MOD 4] := by
sorry

end divisor_count_fourth_power_l189_18995


namespace cookie_boxes_problem_l189_18925

theorem cookie_boxes_problem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ), 
    mark_sold = n - 9 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) ↔ 
  n = 10 := by
sorry

end cookie_boxes_problem_l189_18925


namespace gcd_of_48_72_120_l189_18980

theorem gcd_of_48_72_120 : Nat.gcd 48 (Nat.gcd 72 120) = 24 := by
  sorry

end gcd_of_48_72_120_l189_18980


namespace remainder_problem_l189_18940

theorem remainder_problem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end remainder_problem_l189_18940


namespace simplify_expression_l189_18996

theorem simplify_expression (h : Real.pi < 4) : 
  Real.sqrt ((Real.pi - 4)^2) + Real.pi = 4 := by
  sorry

end simplify_expression_l189_18996


namespace alcohol_mixture_exists_l189_18919

theorem alcohol_mixture_exists : ∃ (x y z : ℕ), 
  x + y + z = 560 ∧ 
  (70 * x + 64 * y + 50 * z : ℚ) = 60 * 560 := by
  sorry

end alcohol_mixture_exists_l189_18919


namespace intersection_point_is_two_one_l189_18973

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first line: x - 2y = 0 -/
def line1 (p : IntersectionPoint) : Prop :=
  p.x - 2 * p.y = 0

/-- Definition of the second line: x + y - 3 = 0 -/
def line2 (p : IntersectionPoint) : Prop :=
  p.x + p.y - 3 = 0

/-- Theorem stating that (2, 1) is the unique intersection point of the two lines -/
theorem intersection_point_is_two_one :
  ∃! p : IntersectionPoint, line1 p ∧ line2 p ∧ p.x = 2 ∧ p.y = 1 := by
  sorry

end intersection_point_is_two_one_l189_18973


namespace volume_polynomial_coefficients_ratio_l189_18903

/-- A right rectangular prism with edge lengths 2, 2, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 2
  height : ℝ := 5

/-- The set of points within distance r of any point in the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_polynomial_coefficients_ratio (B : Prism) (coeff : VolumeCoefficients) :
  (∀ r : ℝ, volume B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
  (coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0 ∧ coeff.d > 0) →
  (coeff.b * coeff.c) / (coeff.a * coeff.d) = 50.4 := by
  sorry

end volume_polynomial_coefficients_ratio_l189_18903


namespace parabola_focus_l189_18944

/-- A parabola is defined by the equation x = -1/8 * y^2. Its focus is at (-2, 0). -/
theorem parabola_focus (x y : ℝ) : 
  x = -1/8 * y^2 → (x + 2 = 0 ∧ y = 0) := by sorry

end parabola_focus_l189_18944


namespace alternating_series_sum_l189_18948

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem alternating_series_sum (a b : ℕ) :
  (S a + S b + S (a + b) = 1) ↔ (Odd a ∧ Odd b) :=
sorry

end alternating_series_sum_l189_18948


namespace sum_of_solutions_eq_neg_three_fourths_l189_18922

theorem sum_of_solutions_eq_neg_three_fourths :
  let f : ℝ → ℝ := λ x => 243^(x + 1) - 81^(x^2 + 2*x)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = -3/4 :=
by sorry

end sum_of_solutions_eq_neg_three_fourths_l189_18922


namespace katies_speed_l189_18926

/-- Given the running speeds of Eugene, Brianna, Marcus, and Katie, prove Katie's speed -/
theorem katies_speed (eugene_speed : ℝ) (brianna_ratio : ℝ) (marcus_ratio : ℝ) (katie_ratio : ℝ)
  (h1 : eugene_speed = 5)
  (h2 : brianna_ratio = 3/4)
  (h3 : marcus_ratio = 5/6)
  (h4 : katie_ratio = 4/5) :
  katie_ratio * marcus_ratio * brianna_ratio * eugene_speed = 2.5 := by
  sorry

end katies_speed_l189_18926


namespace football_team_progress_l189_18957

/-- Calculates the net progress of a football team given yards lost and gained -/
def netProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Proves that when a team loses 5 yards and gains 10 yards, their net progress is 5 yards -/
theorem football_team_progress : netProgress 5 10 = 5 := by
  sorry

end football_team_progress_l189_18957


namespace work_completion_time_l189_18978

theorem work_completion_time (a b c : ℝ) (h1 : b = 6) (h2 : c = 12) 
  (h3 : 1/a + 1/b + 1/c = 7/24) : a = 24 := by
  sorry

end work_completion_time_l189_18978


namespace cubic_function_constant_term_l189_18936

/-- Given a cubic function with specific properties, prove that the constant term is 16 -/
theorem cubic_function_constant_term (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℤ → ℤ := λ x => x^3 + a*x^2 + b*x + c
  (f a = a^3) ∧ (f b = b^3) → c = 16 := by
  sorry

end cubic_function_constant_term_l189_18936


namespace flash_catches_ace_l189_18901

/-- The time it takes for Flash to catch up to Ace in a race -/
theorem flash_catches_ace (v a y : ℝ) (hv : v > 0) (ha : a > 0) (hy : y > 0) :
  let t := (v + Real.sqrt (v^2 + 2*a*y)) / a
  2 * (v * t + y) = a * t^2 := by sorry

end flash_catches_ace_l189_18901


namespace tangent_point_coordinates_l189_18941

/-- The function representing the curve y = x^3 + x^2 -/
def f (x : ℝ) : ℝ := x^3 + x^2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2*x

theorem tangent_point_coordinates :
  ∀ a : ℝ, f' a = 4 → (a = 1 ∧ f a = 2) ∨ (a = -1 ∧ f a = -2) :=
by sorry

end tangent_point_coordinates_l189_18941


namespace water_consumed_last_mile_is_three_l189_18931

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  totalDistance : ℝ
  initialWater : ℝ
  hikeDuration : ℝ
  remainingWater : ℝ
  leakRate : ℝ
  consumptionRateFirst6Miles : ℝ

/-- Calculates the water consumed in the last mile of the hike -/
def waterConsumedLastMile (h : HikeScenario) : ℝ :=
  h.initialWater - h.remainingWater - (h.leakRate * h.hikeDuration) - 
  (h.consumptionRateFirst6Miles * (h.totalDistance - 1))

/-- Theorem stating that given the specific conditions, Harry drank 3 cups of water in the last mile -/
theorem water_consumed_last_mile_is_three (h : HikeScenario) 
  (h_totalDistance : h.totalDistance = 7)
  (h_initialWater : h.initialWater = 11)
  (h_hikeDuration : h.hikeDuration = 3)
  (h_remainingWater : h.remainingWater = 2)
  (h_leakRate : h.leakRate = 1)
  (h_consumptionRateFirst6Miles : h.consumptionRateFirst6Miles = 0.5) :
  waterConsumedLastMile h = 3 := by
  sorry

end water_consumed_last_mile_is_three_l189_18931


namespace quadrilateral_I_greater_than_II_l189_18900

/-- Quadrilateral I with vertices at (0,0), (2,0), (2,1), and (0,1) -/
def quadrilateral_I : List (ℝ × ℝ) := [(0,0), (2,0), (2,1), (0,1)]

/-- Quadrilateral II with vertices at (0,0), (1,0), (1,1), (0,2) -/
def quadrilateral_II : List (ℝ × ℝ) := [(0,0), (1,0), (1,1), (0,2)]

/-- Calculate the area of a quadrilateral given its vertices -/
def area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the perimeter of a quadrilateral given its vertices -/
def perimeter (vertices : List (ℝ × ℝ)) : ℝ := sorry

theorem quadrilateral_I_greater_than_II :
  area quadrilateral_I > area quadrilateral_II ∧
  perimeter quadrilateral_I > perimeter quadrilateral_II :=
by sorry

end quadrilateral_I_greater_than_II_l189_18900


namespace train_speed_proof_l189_18969

/-- Proves that a train with given crossing times has a specific speed -/
theorem train_speed_proof (platform_length : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  platform_length = 280 →
  platform_cross_time = 32 →
  man_cross_time = 18 →
  ∃ (train_speed : ℝ), train_speed = 72 ∧ 
    (train_speed * man_cross_time = train_speed * platform_cross_time - platform_length) :=
by
  sorry

#check train_speed_proof

end train_speed_proof_l189_18969


namespace billy_strategy_l189_18937

def FencePainting (n : ℕ) :=
  ∃ (strategy : ℕ → ℕ),
    (∀ k, k ≤ n → strategy k ≤ n) ∧
    (∀ k, k < n → strategy k ≠ k) ∧
    (∀ k, k < n - 1 → strategy k ≠ strategy (k + 1))

theorem billy_strategy (n : ℕ) (h : n > 10) :
  FencePainting n ∧ (n % 2 = 1 → ∃ (winning_strategy : ℕ → ℕ), FencePainting n) :=
sorry

#check billy_strategy

end billy_strategy_l189_18937


namespace not_clear_def_not_set_l189_18966

-- Define what it means for a collection to have a clear definition
def has_clear_definition (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x) ∨ (¬C x)

-- Define what it means to be a set
def is_set (S : Type → Prop) : Prop :=
  has_clear_definition S

-- Theorem: A collection without a clear definition is not a set
theorem not_clear_def_not_set (C : Type → Prop) :
  ¬(has_clear_definition C) → ¬(is_set C) := by
  sorry

end not_clear_def_not_set_l189_18966


namespace greatest_ratio_three_digit_number_l189_18933

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a three-digit number to the sum of its digits -/
def ratio (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem greatest_ratio_three_digit_number :
  (∀ n : ThreeDigitNumber, ratio n ≤ 100) ∧
  (∃ n : ThreeDigitNumber, ratio n = 100) :=
sorry

end greatest_ratio_three_digit_number_l189_18933


namespace at_least_one_good_product_l189_18971

theorem at_least_one_good_product (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 12)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  ∀ (selection : Finset (Fin total)), selection.card = selected → 
    ∃ (x : Fin total), x ∈ selection ∧ x.val ∉ Finset.range defective :=
by sorry

end at_least_one_good_product_l189_18971


namespace smallest_number_with_conditions_l189_18916

def ends_in (n : ℕ) (m : ℕ) : Prop := n % 100 = m

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    ends_in n 56 ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 →
    n ≥ 29899856 :=
by sorry

end smallest_number_with_conditions_l189_18916


namespace a_range_l189_18981

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

/-- f(x) is a decreasing function -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The theorem statement -/
theorem a_range (a : ℝ) :
  is_decreasing (f a) → a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 :=
sorry

end a_range_l189_18981


namespace speed_limit_representation_l189_18988

-- Define the speed limit
def speed_limit : ℝ := 70

-- Define a vehicle's speed
variable (v : ℝ)

-- Theorem stating that v ≤ speed_limit correctly represents the speed limit instruction
theorem speed_limit_representation : 
  (v ≤ speed_limit) ↔ (v ≤ speed_limit ∧ ¬(v > speed_limit)) :=
by sorry

end speed_limit_representation_l189_18988


namespace center_value_is_27_l189_18975

/-- Represents a 7x7 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 7 → Fin 7 → ℤ

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : Fin 7 → ℤ) : ℤ :=
  (seq 6 - seq 0) / 6

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 7 → ℤ) : Prop :=
  ∀ i j : Fin 7, seq j - seq i = (j - i : ℤ) * commonDifference seq

/-- Theorem: The center value of the arithmetic array is 27 -/
theorem center_value_is_27 (A : ArithmeticArray) 
  (h_rows : ∀ i : Fin 7, isArithmeticSequence (λ j ↦ A i j))
  (h_cols : ∀ j : Fin 7, isArithmeticSequence (λ i ↦ A i j))
  (h_first_row : A 0 0 = 3 ∧ A 0 6 = 39)
  (h_last_row : A 6 0 = 10 ∧ A 6 6 = 58) :
  A 3 3 = 27 := by
  sorry

end center_value_is_27_l189_18975


namespace circle_and_line_problem_l189_18990

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define points A, B, and D
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)
def point_D : ℝ × ℝ := (0, 1)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Define line l with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x - 0)

-- Define the property that line m bisects circle C
def bisects (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property that a line intersects a circle at two distinct points
def intersects_at_two_points (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the squared distance between two points on a line
def squared_distance (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_and_line_problem (k : ℝ) :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  bisects line_m circle_C ∧
  intersects_at_two_points (line_l k) circle_C ∧
  squared_distance (line_l k) circle_C = 12 →
  k = 1 := by sorry

end circle_and_line_problem_l189_18990


namespace amy_ticket_cost_l189_18960

/-- The total cost of tickets purchased by Amy at the fair -/
theorem amy_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) (price_per_ticket : ℚ) :
  initial_tickets = 33 →
  additional_tickets = 21 →
  price_per_ticket = 3/2 →
  (initial_tickets + additional_tickets : ℚ) * price_per_ticket = 81 := by
sorry

end amy_ticket_cost_l189_18960


namespace erik_money_left_l189_18945

theorem erik_money_left (initial_money : ℕ) (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (h1 : initial_money = 86) 
  (h2 : bread_quantity = 3) (h3 : juice_quantity = 3) (h4 : bread_price = 3) 
  (h5 : juice_price = 6) : 
  initial_money - (bread_quantity * bread_price + juice_quantity * juice_price) = 59 := by
  sorry

end erik_money_left_l189_18945


namespace compound_interest_rate_l189_18921

/-- Given an initial investment of $7000, invested for 2 years with annual compounding,
    resulting in a final amount of $8470, prove that the annual interest rate is 0.1 (10%). -/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (r : ℝ) : 
  P = 7000 → A = 8470 → t = 2 → n = 1 → 
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.1 := by
  sorry


end compound_interest_rate_l189_18921


namespace max_telephones_is_210_quality_rate_at_least_90_percent_l189_18912

/-- Represents the quality inspection of a batch of telephones. -/
structure TelephoneBatch where
  first_50_high_quality : Nat := 49
  first_50_total : Nat := 50
  subsequent_high_quality : Nat := 7
  subsequent_total : Nat := 8
  quality_threshold : Rat := 9/10

/-- The maximum number of telephones in the batch satisfying the quality conditions. -/
def max_telephones (batch : TelephoneBatch) : Nat :=
  batch.first_50_total + 20 * batch.subsequent_total

/-- Theorem stating that 210 is the maximum number of telephones in the batch. -/
theorem max_telephones_is_210 (batch : TelephoneBatch) :
  max_telephones batch = 210 :=
by sorry

/-- Theorem stating that the quality rate is at least 90% for the maximum batch size. -/
theorem quality_rate_at_least_90_percent (batch : TelephoneBatch) :
  let total := max_telephones batch
  let high_quality := batch.first_50_high_quality + 20 * batch.subsequent_high_quality
  (high_quality : Rat) / total ≥ batch.quality_threshold :=
by sorry

end max_telephones_is_210_quality_rate_at_least_90_percent_l189_18912


namespace solution_in_quadrant_I_l189_18979

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8/5 := by
  sorry

end solution_in_quadrant_I_l189_18979


namespace superhero_advantage_l189_18976

/-- Superhero's speed in miles per minute -/
def superhero_speed : ℚ := 10 / 4

/-- Supervillain's speed in miles per hour -/
def supervillain_speed : ℚ := 100

/-- Minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem superhero_advantage : 
  (superhero_speed * minutes_per_hour) - supervillain_speed = 50 := by sorry

end superhero_advantage_l189_18976


namespace tangent_line_at_pi_l189_18997

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1

theorem tangent_line_at_pi :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - f π = m * (x - π)) ↔ (x * Real.exp π + y - 1 - π * Real.exp π = 0) := by
  sorry

end tangent_line_at_pi_l189_18997


namespace simplify_expression_l189_18930

theorem simplify_expression : (12^0.6) * (12^0.4) * (8^0.2) * (8^0.8) = 96 := by
  sorry

end simplify_expression_l189_18930


namespace min_value_of_expression_l189_18928

theorem min_value_of_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) :=
by sorry

end min_value_of_expression_l189_18928


namespace wy_equals_uv_l189_18911

-- Define the variables
variable (u v w y : ℝ)
variable (α β : ℝ)

-- Define the conditions
axiom sin_roots : (Real.sin α)^2 - u * (Real.sin α) + v = 0 ∧ (Real.sin β)^2 - u * (Real.sin β) + v = 0
axiom cos_roots : (Real.cos α)^2 - w * (Real.cos α) + y = 0 ∧ (Real.cos β)^2 - w * (Real.cos β) + y = 0
axiom right_triangle : Real.sin α = Real.cos β ∧ Real.sin β = Real.cos α

-- State the theorem
theorem wy_equals_uv : wy = uv := by sorry

end wy_equals_uv_l189_18911


namespace calculate_expression_l189_18934

theorem calculate_expression (a : ℝ) : a * a^2 - 2 * a^3 = -a^3 := by
  sorry

end calculate_expression_l189_18934


namespace smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l189_18983

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

theorem smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150
  (a b : ℕ)
  (ha1 : is_two_digit a)
  (ha2 : a % 5 = 0)
  (ha3 : ends_in_five a)
  (ha4 : ∀ n, is_two_digit n → n % 5 = 0 → ends_in_five n → a ≤ n)
  (hb1 : is_three_digit b)
  (hb2 : b % 7 = 0)
  (hb3 : b > 150)
  (hb4 : ∀ n, is_three_digit n → n % 7 = 0 → n > 150 → b ≤ n) :
  a + b = 176 := by
  sorry

end smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l189_18983


namespace coin_toss_probability_l189_18964

def coin_toss_events : ℕ := 2^4

def favorable_events : ℕ := 11

theorem coin_toss_probability : 
  (favorable_events : ℚ) / coin_toss_events = 11 / 16 :=
sorry

end coin_toss_probability_l189_18964


namespace pentatonic_scale_theorem_l189_18992

/-- Calculates the length of the instrument for the nth note in the pentatonic scale,
    given the initial length and the number of alternations between subtracting
    and adding one-third. -/
def pentatonic_length (initial_length : ℚ) (n : ℕ) : ℚ :=
  initial_length * (2/3)^(n/2) * (4/3)^((n-1)/2)

theorem pentatonic_scale_theorem (a : ℚ) :
  pentatonic_length a 3 = 32 → a = 54 := by
  sorry

#check pentatonic_scale_theorem

end pentatonic_scale_theorem_l189_18992


namespace base5_arithmetic_l189_18959

/-- Converts a base 5 number to base 10 --/
def base5_to_base10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

/-- Converts a base 10 number to base 5 --/
noncomputable def base10_to_base5 (n : ℕ) : ℕ × ℕ × ℕ :=
  let d₂ := n / 25
  let r₂ := n % 25
  let d₁ := r₂ / 5
  let d₀ := r₂ % 5
  (d₂, d₁, d₀)

/-- Theorem stating that 142₅ + 324₅ - 213₅ = 303₅ --/
theorem base5_arithmetic : 
  let x := base5_to_base10 1 4 2
  let y := base5_to_base10 3 2 4
  let z := base5_to_base10 2 1 3
  base10_to_base5 (x + y - z) = (3, 0, 3) := by sorry

end base5_arithmetic_l189_18959


namespace movie_theater_child_price_l189_18965

/-- Proves that the price for children is $4.5 given the conditions of the movie theater problem -/
theorem movie_theater_child_price 
  (adult_price : ℝ) 
  (num_children : ℕ) 
  (child_adult_diff : ℕ) 
  (total_receipts : ℝ) 
  (h1 : adult_price = 6.75)
  (h2 : num_children = 48)
  (h3 : child_adult_diff = 20)
  (h4 : total_receipts = 405) :
  ∃ (child_price : ℝ), 
    child_price = 4.5 ∧ 
    (num_children : ℝ) * child_price + ((num_children : ℝ) - (child_adult_diff : ℝ)) * adult_price = total_receipts :=
by
  sorry

end movie_theater_child_price_l189_18965


namespace optimal_selling_price_l189_18909

/-- Represents the profit optimization problem for a product -/
def ProfitOptimization (initialCost initialPrice initialSales : ℝ) 
                       (priceIncrease salesDecrease : ℝ) : Prop :=
  let profitFunction := fun x : ℝ => 
    (initialPrice + priceIncrease * x - initialCost) * (initialSales - salesDecrease * x)
  ∃ (optimalX : ℝ), 
    (∀ x : ℝ, profitFunction x ≤ profitFunction optimalX) ∧
    initialPrice + priceIncrease * optimalX = 14

/-- The main theorem stating the optimal selling price -/
theorem optimal_selling_price :
  ProfitOptimization 8 10 200 0.5 10 := by
  sorry

#check optimal_selling_price

end optimal_selling_price_l189_18909


namespace external_tangent_intersections_collinear_l189_18910

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of disjoint circles
def disjoint (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

-- Define the intersection point of external tangents
def external_tangent_intersection (c1 c2 : Circle) : ℝ × ℝ :=
  sorry -- The actual computation is not needed for the statement

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- The main theorem
theorem external_tangent_intersections_collinear (C1 C2 C3 : Circle)
  (h12 : disjoint C1 C2) (h23 : disjoint C2 C3) (h31 : disjoint C3 C1) :
  let T12 := external_tangent_intersection C1 C2
  let T23 := external_tangent_intersection C2 C3
  let T31 := external_tangent_intersection C3 C1
  collinear T12 T23 T31 :=
sorry

end external_tangent_intersections_collinear_l189_18910


namespace no_prime_valued_polynomial_l189_18984

theorem no_prime_valued_polynomial : ¬ ∃ (P : ℕ → ℤ), (∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > n → P k = 0)) ∧ (∀ k : ℕ, Nat.Prime (P k).natAbs) := by
  sorry

end no_prime_valued_polynomial_l189_18984


namespace common_points_on_line_l189_18938

-- Define the curves and line
def C1 (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 1)^2 = a^2}
def C2 : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 4}
def C3 : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1}

theorem common_points_on_line (a : ℝ) (h : a > 0) : 
  (∀ p, p ∈ C1 a ∩ C2 → p ∈ C3) → a = 1 := by
  sorry

end common_points_on_line_l189_18938


namespace wooden_block_length_is_3070_l189_18967

/-- The length of a wooden block in centimeters, given that it is 30 cm shorter than 31 meters -/
def wooden_block_length : ℕ :=
  let meters_to_cm : ℕ → ℕ := (· * 100)
  meters_to_cm 31 - 30

theorem wooden_block_length_is_3070 : wooden_block_length = 3070 := by
  sorry

end wooden_block_length_is_3070_l189_18967


namespace quadratic_form_h_value_l189_18946

/-- Given a quadratic expression 3x^2 + 9x + 17, when written in the form a(x-h)^2 + k, h = -3/2 -/
theorem quadratic_form_h_value : 
  ∃ (a k : ℝ), ∀ x : ℝ, 3*x^2 + 9*x + 17 = a*(x - (-3/2))^2 + k :=
by sorry

end quadratic_form_h_value_l189_18946


namespace pencil_cost_is_two_l189_18918

/-- Represents the cost of school supplies for Mary --/
structure SchoolSuppliesCost where
  num_classes : ℕ
  folders_per_class : ℕ
  pencils_per_class : ℕ
  erasers_per_pencils : ℕ
  folder_cost : ℚ
  eraser_cost : ℚ
  paint_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of a single pencil given the school supplies cost structure --/
def pencil_cost (c : SchoolSuppliesCost) : ℚ :=
  let total_folders := c.num_classes * c.folders_per_class
  let total_pencils := c.num_classes * c.pencils_per_class
  let total_erasers := (total_pencils + c.erasers_per_pencils - 1) / c.erasers_per_pencils
  let non_pencil_cost := total_folders * c.folder_cost + total_erasers * c.eraser_cost + c.paint_cost
  let pencil_total_cost := c.total_spent - non_pencil_cost
  pencil_total_cost / total_pencils

/-- Theorem stating that the cost of each pencil is $2 --/
theorem pencil_cost_is_two (c : SchoolSuppliesCost) 
  (h1 : c.num_classes = 6)
  (h2 : c.folders_per_class = 1)
  (h3 : c.pencils_per_class = 3)
  (h4 : c.erasers_per_pencils = 6)
  (h5 : c.folder_cost = 6)
  (h6 : c.eraser_cost = 1)
  (h7 : c.paint_cost = 5)
  (h8 : c.total_spent = 80) :
  pencil_cost c = 2 := by
  sorry


end pencil_cost_is_two_l189_18918


namespace second_loan_amount_l189_18904

def initial_loan : ℝ := 40
def final_debt : ℝ := 30

theorem second_loan_amount (half_paid_back : ℝ) (second_loan : ℝ) 
  (h1 : half_paid_back = initial_loan / 2)
  (h2 : final_debt = initial_loan - half_paid_back + second_loan) : 
  second_loan = 10 := by
  sorry

end second_loan_amount_l189_18904


namespace circumcircle_equation_l189_18985

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define an equilateral triangle on the parabola
def equilateral_triangle_on_parabola (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  parabola O.1 O.2 ∧ parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) :
  equilateral_triangle_on_parabola A B →
  ∃ x y : ℝ, (x - 4)^2 + y^2 = 16 ∧
            (x - 0)^2 + (y - 0)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
            (x - 0)^2 + (y - 0)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end circumcircle_equation_l189_18985


namespace tic_tac_toe_wins_l189_18951

theorem tic_tac_toe_wins (total_rounds harry_wins william_wins : ℕ) :
  total_rounds = 15 →
  william_wins = harry_wins + 5 →
  william_wins = 10 := by
sorry

end tic_tac_toe_wins_l189_18951


namespace uncommon_card_ratio_l189_18991

/-- Given a number of card packs, cards per pack, and total uncommon cards,
    prove that the ratio of uncommon cards to total cards per pack is 5:2 -/
theorem uncommon_card_ratio
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (total_uncommon : ℕ)
  (h1 : num_packs = 10)
  (h2 : cards_per_pack = 20)
  (h3 : total_uncommon = 50) :
  (total_uncommon : ℚ) / (num_packs * cards_per_pack : ℚ) = 5 / 2 := by
  sorry

end uncommon_card_ratio_l189_18991


namespace opposite_of_2xyz_l189_18929

theorem opposite_of_2xyz (x y z : ℝ) 
  (h : Real.sqrt (2*x - 1) + Real.sqrt (1 - 2*x) + |x - 2*y| + |z + 4*y| = 0) : 
  -(2*x*y*z) = 1/4 := by
  sorry

end opposite_of_2xyz_l189_18929


namespace not_P_necessary_not_sufficient_for_not_Q_l189_18989

-- Define the propositions P and Q as functions from ℝ to Prop
def P (x : ℝ) : Prop := |2*x - 3| > 1
def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

-- Define the relationship between ¬P and ¬Q
theorem not_P_necessary_not_sufficient_for_not_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  ¬(∀ x, ¬(P x) → ¬(Q x)) :=
sorry

end not_P_necessary_not_sufficient_for_not_Q_l189_18989


namespace faye_candy_count_l189_18956

/-- Calculates the final candy count for Faye after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye's final candy count is 62 pieces. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end faye_candy_count_l189_18956


namespace andrews_age_l189_18913

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 10 * andrew_age →
  grandfather_age - andrew_age = 63 →
  andrew_age = 7 := by
sorry

end andrews_age_l189_18913


namespace longest_side_of_equal_area_rectangles_l189_18942

/-- Represents a rectangle with integer sides -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem longest_side_of_equal_area_rectangles 
  (r1 r2 r3 : Rectangle) 
  (h_equal_areas : area r1 = area r2 ∧ area r2 = area r3)
  (h_one_side_19 : r1.width = 19 ∨ r1.height = 19 ∨ 
                   r2.width = 19 ∨ r2.height = 19 ∨ 
                   r3.width = 19 ∨ r3.height = 19) :
  ∃ (r : Rectangle), (r = r1 ∨ r = r2 ∨ r = r3) ∧ 
    (r.width = 380 ∨ r.height = 380) :=
sorry

end longest_side_of_equal_area_rectangles_l189_18942


namespace technician_salary_l189_18935

theorem technician_salary (total_workers : ℕ) (total_avg_salary : ℝ) 
  (num_technicians : ℕ) (non_tech_avg_salary : ℝ) :
  total_workers = 18 →
  total_avg_salary = 8000 →
  num_technicians = 6 →
  non_tech_avg_salary = 6000 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) / num_technicians = 12000 := by
  sorry

end technician_salary_l189_18935


namespace outfits_count_l189_18999

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The total number of outfit combinations. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of outfits is 600. -/
theorem outfits_count : total_outfits = 600 := by
  sorry

end outfits_count_l189_18999


namespace polynomial_division_remainder_l189_18954

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + 3•X^2 - 4) = (X^2 + 2) * q + (X^2 - 4) :=
by sorry

end polynomial_division_remainder_l189_18954


namespace circle_rotation_l189_18906

theorem circle_rotation (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = 11 * Real.pi) :
  (d / (2 * Real.pi * r)) % 1 = 1/2 := by
  sorry

end circle_rotation_l189_18906


namespace liars_count_l189_18947

/-- Represents the type of inhabitant: Knight or Liar -/
inductive InhabitantType
| Knight
| Liar

/-- Represents an island in the Tenth Kingdom -/
structure Island where
  population : Nat
  knights : Nat

/-- Represents the Tenth Kingdom -/
structure TenthKingdom where
  islands : List Island
  total_islands : Nat
  inhabitants_per_island : Nat

/-- Predicate for islands where everyone answered "Yes" to the first question -/
def first_question_yes (i : Island) : Prop :=
  i.knights = i.population / 2

/-- Predicate for islands where everyone answered "No" to the first question -/
def first_question_no (i : Island) : Prop :=
  i.knights ≠ i.population / 2

/-- Predicate for islands where everyone answered "No" to the second question -/
def second_question_no (i : Island) : Prop :=
  i.knights ≥ i.population / 2

/-- Predicate for islands where everyone answered "Yes" to the second question -/
def second_question_yes (i : Island) : Prop :=
  i.knights < i.population / 2

/-- Main theorem: The number of liars in the Tenth Kingdom is 1013 -/
theorem liars_count (k : TenthKingdom) : Nat := by
  sorry

/-- The Tenth Kingdom setup -/
def tenth_kingdom : TenthKingdom := {
  islands := [],  -- Placeholder for the list of islands
  total_islands := 17,
  inhabitants_per_island := 119
}

#check liars_count tenth_kingdom

end liars_count_l189_18947


namespace tan_alpha_value_l189_18915

theorem tan_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end tan_alpha_value_l189_18915


namespace subset_implies_a_range_l189_18958

theorem subset_implies_a_range (M N : Set ℝ) (a : ℝ) 
  (hM : M = {x : ℝ | x - 2 < 0})
  (hN : N = {x : ℝ | x < a})
  (hSubset : M ⊆ N) :
  a ∈ Set.Ici 2 := by
sorry

end subset_implies_a_range_l189_18958


namespace sector_area_l189_18943

/-- The area of a circular sector with radius 12 meters and central angle 42 degrees -/
theorem sector_area : 
  let r : ℝ := 12
  let θ : ℝ := 42
  let sector_area := (θ / 360) * Real.pi * r^2
  sector_area = (42 / 360) * Real.pi * 12^2 := by sorry

end sector_area_l189_18943


namespace zach_cookies_theorem_l189_18968

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday_cookies : ℕ) : ℕ :=
  let tuesday_cookies := monday_cookies / 2
  let wednesday_cookies := tuesday_cookies * 3
  monday_cookies + tuesday_cookies + wednesday_cookies - 4

/-- Theorem stating that Zach had 92 cookies at the end of three days --/
theorem zach_cookies_theorem :
  total_cookies 32 = 92 :=
by sorry

end zach_cookies_theorem_l189_18968


namespace no_common_point_implies_skew_l189_18972

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (parallel : Line → Line → Prop)
variable (determine_plane : Line → Line → Plane → Prop)
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (has_common_point : Line → Line → Prop)

-- Axioms and definitions
axiom parallel_determine_plane (a b : Line) (p : Plane) :
  parallel a b → determine_plane a b p

axiom non_coplanar_non_collinear (p q r s : Point) :
  ¬coplanar p q r s → ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s

axiom skew_perpendicular (l₁ l₂ : Line) (p : Plane) :
  skew l₁ l₂ → ¬(perpendicular l₁ p ∧ perpendicular l₂ p)

-- The statement to be proved false
theorem no_common_point_implies_skew (l₁ l₂ : Line) :
  ¬has_common_point l₁ l₂ → skew l₁ l₂ := by sorry

end no_common_point_implies_skew_l189_18972


namespace perfect_cube_factors_of_8820_l189_18955

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_8820 :
  let factorization := prime_factorization 8820
  (factorization = [(2, 2), (3, 2), (5, 1), (7, 2)]) →
  count_perfect_cube_factors 8820 = 1 := by sorry

end perfect_cube_factors_of_8820_l189_18955


namespace bakery_boxes_sold_l189_18950

/-- Calculates the number of boxes of doughnuts sold by a bakery -/
theorem bakery_boxes_sold
  (doughnuts_per_box : ℕ)
  (total_doughnuts : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : doughnuts_per_box = 10)
  (h2 : total_doughnuts = 300)
  (h3 : doughnuts_given_away = 30) :
  (total_doughnuts / doughnuts_per_box) - (doughnuts_given_away / doughnuts_per_box) = 27 :=
by sorry

end bakery_boxes_sold_l189_18950


namespace morning_rowers_count_l189_18914

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers : ℕ := 7

/-- The total number of campers who went rowing that day -/
def total_rowers : ℕ := 60

/-- The number of campers who went rowing in the morning -/
def morning_rowers : ℕ := total_rowers - afternoon_rowers

theorem morning_rowers_count : morning_rowers = 53 := by
  sorry

end morning_rowers_count_l189_18914


namespace arithmetic_sequence_problem_l189_18961

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →
  ((a 2 - 1)^3 + 5*(a 2 - 1) = 1) →
  ((a 2010 - 1)^3 + 5*(a 2010 - 1) = -1) →
  (a 2 + a 2010 = 2 ∧ S 2011 = 2011) :=
by sorry

end arithmetic_sequence_problem_l189_18961


namespace apples_per_basket_l189_18924

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) 
  (h1 : total_baskets = 37) 
  (h2 : total_apples = 629) : 
  total_apples / total_baskets = 17 := by
sorry

end apples_per_basket_l189_18924


namespace quadratic_equation_coefficients_l189_18982

/-- Given a quadratic equation 3x^2 = 5x - 1, prove that its standard form coefficients are a = 3 and b = -5 --/
theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x ↦ 3 * x^2 = 5 * x - 1
  let standard_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x ↦ a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ standard_form a b c x) ∧ a = 3 ∧ b = -5 := by
  sorry

end quadratic_equation_coefficients_l189_18982


namespace no_good_tetrahedron_inside_good_parallelepiped_l189_18986

-- Define a good polyhedron
def is_good_polyhedron (volume : ℝ) (surface_area : ℝ) : Prop :=
  volume = surface_area

-- Define a tetrahedron
structure Tetrahedron where
  volume : ℝ
  surface_area : ℝ

-- Define a parallelepiped
structure Parallelepiped where
  volume : ℝ
  surface_area : ℝ
  face_areas : Fin 3 → ℝ
  heights : Fin 3 → ℝ

-- Define the property of a tetrahedron being inside a parallelepiped
def tetrahedron_inside_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  t.volume = (1/3) * t.surface_area * r ∧
  p.heights 0 > 2 * r

-- Theorem statement
theorem no_good_tetrahedron_inside_good_parallelepiped :
  ¬ ∃ (t : Tetrahedron) (p : Parallelepiped),
    is_good_polyhedron t.volume t.surface_area ∧
    is_good_polyhedron p.volume p.surface_area ∧
    tetrahedron_inside_parallelepiped t p :=
sorry

end no_good_tetrahedron_inside_good_parallelepiped_l189_18986


namespace unique_solution_factorial_equation_l189_18977

theorem unique_solution_factorial_equation : 
  ∃! (n : ℕ), (Nat.factorial (n + 2) - Nat.factorial (n + 1) - Nat.factorial n = n^2 + n^4) := by
  sorry

end unique_solution_factorial_equation_l189_18977


namespace toothpick_grid_25_15_l189_18970

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := grid.height * grid.width
  horizontal + vertical + diagonal

/-- Theorem stating the total number of toothpicks in a 25x15 grid -/
theorem toothpick_grid_25_15 :
  total_toothpicks ⟨25, 15⟩ = 1165 := by sorry

end toothpick_grid_25_15_l189_18970


namespace triangle_ABC_properties_l189_18917

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the theorem
theorem triangle_ABC_properties 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_cos_A : Real.cos A = -5/13)
  (h_cos_B : Real.cos B = 3/5)
  (h_BC : 5 = 5) :
  Real.sin C = 16/65 ∧ 
  5 * 5 * Real.sin C / 2 = 8/3 :=
sorry

end triangle_ABC_properties_l189_18917


namespace sophia_age_in_three_years_l189_18902

/-- Represents the current ages of Jeremy, Sebastian, and Sophia --/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The sum of their ages in three years is 150 --/
def sum_ages_in_three_years (ages : Ages) : Prop :=
  ages.jeremy + 3 + ages.sebastian + 3 + ages.sophia + 3 = 150

/-- Sebastian is 4 years older than Jeremy --/
def sebastian_older (ages : Ages) : Prop :=
  ages.sebastian = ages.jeremy + 4

/-- Jeremy's current age is 40 --/
def jeremy_age (ages : Ages) : Prop :=
  ages.jeremy = 40

/-- Sophia's age three years from now --/
def sophia_future_age (ages : Ages) : ℕ :=
  ages.sophia + 3

/-- Theorem stating Sophia's age three years from now is 60 --/
theorem sophia_age_in_three_years (ages : Ages) 
  (h1 : sum_ages_in_three_years ages) 
  (h2 : sebastian_older ages) 
  (h3 : jeremy_age ages) : 
  sophia_future_age ages = 60 := by
  sorry

end sophia_age_in_three_years_l189_18902


namespace marcel_total_cost_l189_18963

def calculate_total_cost (pen_price : ℝ) : ℝ :=
  let briefcase_price := 5 * pen_price
  let notebook_price := 2 * pen_price
  let calculator_price := 3 * notebook_price
  let briefcase_discount := 0.15 * briefcase_price
  let discounted_briefcase_price := briefcase_price - briefcase_discount
  let total_before_tax := pen_price + discounted_briefcase_price + notebook_price + calculator_price
  let tax := 0.10 * total_before_tax
  total_before_tax + tax

theorem marcel_total_cost :
  calculate_total_cost 4 = 58.30 := by sorry

end marcel_total_cost_l189_18963


namespace prob_white_second_given_red_first_l189_18920

/-- The probability of drawing a white ball on the second draw, given that the first ball drawn is red -/
theorem prob_white_second_given_red_first
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 5)
  (h_white : white_balls = 3) :
  (white_balls : ℚ) / (total_balls - 1) = 3 / 7 := by
  sorry

end prob_white_second_given_red_first_l189_18920
