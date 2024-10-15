import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l679_67929

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 16*x - 12) = 0)} = 
  {1, -12, 3, -4} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l679_67929


namespace NUMINAMATH_CALUDE_kelly_points_l679_67907

def golden_state_team (kelly : ℕ) : Prop :=
  let draymond := 12
  let curry := 2 * draymond
  let durant := 2 * kelly
  let klay := draymond / 2
  draymond + curry + kelly + durant + klay = 69

theorem kelly_points : ∃ (k : ℕ), golden_state_team k ∧ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_kelly_points_l679_67907


namespace NUMINAMATH_CALUDE_allocation_schemes_l679_67997

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of tasks --/
def num_tasks : ℕ := 4

/-- The number of volunteers who cannot perform a specific task --/
def num_restricted : ℕ := 2

/-- Calculates the number of ways to allocate tasks to volunteers with restrictions --/
def num_allocations (n v t r : ℕ) : ℕ :=
  -- n: total number of volunteers
  -- v: number of volunteers to be selected
  -- t: number of tasks
  -- r: number of volunteers who cannot perform a specific task
  sorry

/-- Theorem stating the number of allocation schemes --/
theorem allocation_schemes :
  num_allocations num_volunteers num_tasks num_tasks num_restricted = 72 :=
by sorry

end NUMINAMATH_CALUDE_allocation_schemes_l679_67997


namespace NUMINAMATH_CALUDE_tan_double_angle_l679_67995

theorem tan_double_angle (α : Real) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2 →
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l679_67995


namespace NUMINAMATH_CALUDE_pool_filling_time_l679_67938

/-- Proves that filling a 24,000-gallon pool with 5 hoses supplying 3 gallons per minute takes 27 hours (rounded) -/
theorem pool_filling_time :
  let pool_capacity : ℕ := 24000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time := (pool_capacity + total_flow_rate - 1) / total_flow_rate
  filling_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l679_67938


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l679_67984

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l679_67984


namespace NUMINAMATH_CALUDE_fraction_equality_l679_67910

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a / 4 = b / 3) : b / (a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l679_67910


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l679_67902

/-- The number of walnut trees remaining in the park after cutting down damaged trees. -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 29 walnut trees remain after cutting down 13 from the initial 42. -/
theorem walnut_trees_remaining : remaining_walnut_trees 42 13 = 29 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l679_67902


namespace NUMINAMATH_CALUDE_inequality_solution_set_l679_67994

theorem inequality_solution_set (x : ℝ) :
  (3 / (x + 2) + 4 / (x + 8) ≥ 3 / 4) ↔ 
  (x ∈ Set.Icc (-10.125) (-8) ∪ Set.Ico (-2) 4.125) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l679_67994


namespace NUMINAMATH_CALUDE_special_function_property_l679_67957

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 5 - f 1) / f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l679_67957


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l679_67945

-- Define set A
def A : Set ℝ := {x : ℝ | x * Real.sqrt (x^2 - 4) ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x - 1| + |x + 1| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {-2} ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l679_67945


namespace NUMINAMATH_CALUDE_true_false_questions_count_l679_67934

/-- Proves that the number of true/false questions is 6 given the conditions of the problem -/
theorem true_false_questions_count :
  ∀ (T F M : ℕ),
  T + F + M = 45 →
  M = 2 * F →
  F = T + 7 →
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_true_false_questions_count_l679_67934


namespace NUMINAMATH_CALUDE_clothespin_count_total_clothespins_l679_67942

theorem clothespin_count (handkerchiefs : ℕ) (ropes : ℕ) : ℕ :=
  let ends_per_handkerchief := 2
  let pins_for_handkerchiefs := handkerchiefs * ends_per_handkerchief
  let pins_for_ropes := ropes
  pins_for_handkerchiefs + pins_for_ropes

theorem total_clothespins : clothespin_count 40 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_clothespin_count_total_clothespins_l679_67942


namespace NUMINAMATH_CALUDE_inequality_condition_l679_67912

theorem inequality_condition (b : ℝ) : 
  (b > 0) → (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l679_67912


namespace NUMINAMATH_CALUDE_root_transformation_equation_l679_67965

theorem root_transformation_equation : 
  ∀ (p q r s : ℂ),
  (p^4 + 4*p^3 - 5 = 0) → 
  (q^4 + 4*q^3 - 5 = 0) → 
  (r^4 + 4*r^3 - 5 = 0) → 
  (s^4 + 4*s^3 - 5 = 0) → 
  ∃ (x : ℂ),
  (x = (p+q+r)/s^3 ∨ x = (p+q+s)/r^3 ∨ x = (p+r+s)/q^3 ∨ x = (q+r+s)/p^3) →
  (5*x^6 - x^2 + 4*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_equation_l679_67965


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l679_67946

theorem sum_of_coefficients (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x - 2) = (5 * x^2 - 8 * x - 6) / (x - 3)) →
  C + D = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l679_67946


namespace NUMINAMATH_CALUDE_john_profit_l679_67936

/-- Calculates John's profit from selling woodburnings, metal sculptures, and paintings. -/
theorem john_profit : 
  let woodburnings_count : ℕ := 20
  let woodburnings_price : ℚ := 15
  let metal_sculptures_count : ℕ := 15
  let metal_sculptures_price : ℚ := 25
  let paintings_count : ℕ := 10
  let paintings_price : ℚ := 40
  let wood_cost : ℚ := 100
  let metal_cost : ℚ := 150
  let paint_cost : ℚ := 120
  let woodburnings_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05

  let woodburnings_revenue := woodburnings_count * woodburnings_price * (1 - woodburnings_discount)
  let metal_sculptures_revenue := metal_sculptures_count * metal_sculptures_price
  let paintings_revenue := paintings_count * paintings_price
  let total_revenue := woodburnings_revenue + metal_sculptures_revenue + paintings_revenue
  let total_revenue_with_tax := total_revenue * (1 + sales_tax)
  let total_cost := wood_cost + metal_cost + paint_cost
  let profit := total_revenue_with_tax - total_cost

  profit = 727.25
:= by sorry

end NUMINAMATH_CALUDE_john_profit_l679_67936


namespace NUMINAMATH_CALUDE_equal_selection_probability_l679_67939

/-- Represents the selection process for a visiting group from a larger group of students. -/
structure SelectionProcess where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- The probability of a student being selected given the selection process. -/
def selection_probability (process : SelectionProcess) : ℚ :=
  (process.selected_students : ℚ) / (process.total_students : ℚ)

/-- Theorem stating that the selection probability is equal for all students. -/
theorem equal_selection_probability (process : SelectionProcess) 
  (h1 : process.total_students = 2006)
  (h2 : process.selected_students = 50)
  (h3 : process.eliminated_students = 6) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l679_67939


namespace NUMINAMATH_CALUDE_tims_income_percentage_l679_67935

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 1.12 * juan) : 
  tim = 0.7 * juan := by
sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l679_67935


namespace NUMINAMATH_CALUDE_no_real_solutions_l679_67978

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l679_67978


namespace NUMINAMATH_CALUDE_inequality_range_l679_67972

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |2*x - a| > x - 1) ↔ (a < 3 ∨ a > 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l679_67972


namespace NUMINAMATH_CALUDE_museum_paintings_l679_67962

theorem museum_paintings (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 98 → removed = 3 → remaining = initial - removed → remaining = 95 := by
sorry

end NUMINAMATH_CALUDE_museum_paintings_l679_67962


namespace NUMINAMATH_CALUDE_notebooks_in_scenario3_l679_67932

/-- Represents the production scenario in a factory --/
structure ProductionScenario where
  workers : ℕ
  hours : ℕ
  tablets : ℕ
  notebooks : ℕ

/-- The production rate for tablets (time to produce one tablet) --/
def tablet_rate : ℝ := 1

/-- The production rate for notebooks (time to produce one notebook) --/
def notebook_rate : ℝ := 2

/-- The given production scenarios --/
def scenario1 : ProductionScenario := ⟨120, 1, 360, 240⟩
def scenario2 : ProductionScenario := ⟨100, 2, 400, 500⟩
def scenario3 (n : ℕ) : ProductionScenario := ⟨80, 3, 480, n⟩

/-- Theorem stating that the number of notebooks produced in scenario3 is 120 --/
theorem notebooks_in_scenario3 : ∃ n : ℕ, scenario3 n = ⟨80, 3, 480, 120⟩ := by
  sorry


end NUMINAMATH_CALUDE_notebooks_in_scenario3_l679_67932


namespace NUMINAMATH_CALUDE_lisa_sock_collection_l679_67952

/-- The number of sock pairs Lisa ends up with after contributions from various sources. -/
def total_socks (lisa_initial : ℕ) (sandra : ℕ) (mom_extra : ℕ) : ℕ :=
  lisa_initial + sandra + (sandra / 5) + (3 * lisa_initial + mom_extra)

/-- Theorem stating the total number of sock pairs Lisa ends up with. -/
theorem lisa_sock_collection : total_socks 12 20 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lisa_sock_collection_l679_67952


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l679_67974

/-- The surface area of a sphere that circumscribes a cube with edge length 4 -/
theorem sphere_surface_area_with_inscribed_cube : 
  ∀ (cube_edge_length : ℝ) (sphere_radius : ℝ),
    cube_edge_length = 4 →
    sphere_radius = 2 * Real.sqrt 3 →
    4 * Real.pi * sphere_radius^2 = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l679_67974


namespace NUMINAMATH_CALUDE_prob_change_approx_point_54_l679_67914

/-- The number of banks in the country of Alpha -/
def num_banks : ℕ := 5

/-- The initial probability of a bank closing -/
def initial_prob : ℝ := 0.05

/-- The probability of a bank closing after the crisis -/
def crisis_prob : ℝ := 0.25

/-- The probability that at least one bank will close -/
def prob_at_least_one_close (p : ℝ) : ℝ := 1 - (1 - p) ^ num_banks

/-- The change in probability of at least one bank closing -/
def prob_change : ℝ :=
  |prob_at_least_one_close crisis_prob - prob_at_least_one_close initial_prob|

/-- Theorem stating that the change in probability is approximately 0.54 -/
theorem prob_change_approx_point_54 :
  ∃ ε > 0, ε < 0.005 ∧ |prob_change - 0.54| < ε :=
sorry

end NUMINAMATH_CALUDE_prob_change_approx_point_54_l679_67914


namespace NUMINAMATH_CALUDE_bills_height_ratio_l679_67921

/-- Represents the heights of three siblings in inches -/
structure SiblingHeights where
  cary : ℕ
  jan : ℕ
  bill : ℕ

/-- Given the heights of Cary, Jan, and Bill, proves that Bill's height is half of Cary's -/
theorem bills_height_ratio (h : SiblingHeights) 
  (h_cary : h.cary = 72)
  (h_jan : h.jan = 42)
  (h_jan_bill : h.jan = h.bill + 6) :
  h.bill / h.cary = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_bills_height_ratio_l679_67921


namespace NUMINAMATH_CALUDE_supercomputer_additions_in_half_hour_l679_67906

/-- Proves that a supercomputer performing 20,000 additions per second can complete 36,000,000 additions in half an hour. -/
theorem supercomputer_additions_in_half_hour :
  let additions_per_second : ℕ := 20000
  let seconds_in_half_hour : ℕ := 1800
  additions_per_second * seconds_in_half_hour = 36000000 :=
by sorry

end NUMINAMATH_CALUDE_supercomputer_additions_in_half_hour_l679_67906


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l679_67951

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (a^4 - 15*a^2 + 56 = 0) ∧ 
  (b^4 - 15*b^2 + 56 = 0) ∧ 
  (c^4 - 15*c^2 + 56 = 0) ∧ 
  (d^4 - 15*d^2 + 56 = 0) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l679_67951


namespace NUMINAMATH_CALUDE_right_triangle_area_l679_67919

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = (4/3) * b) (h5 : a = (2/3) * c) (h6 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 2/3 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l679_67919


namespace NUMINAMATH_CALUDE_max_min_values_l679_67976

theorem max_min_values (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (1 / (a + 2*b) + 1 / (2*a + b) ≥ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l679_67976


namespace NUMINAMATH_CALUDE_apple_consumption_theorem_l679_67900

/-- Represents the apple's division and consumption rates -/
structure AppleConsumption where
  above_water : ℚ
  below_water : ℚ
  fish_rate : ℚ
  bird_rate : ℚ

/-- Theorem stating the portions of apple eaten by fish and bird -/
theorem apple_consumption_theorem (a : AppleConsumption) 
  (h1 : a.above_water = 1/5)
  (h2 : a.below_water = 4/5)
  (h3 : a.fish_rate = 120)
  (h4 : a.bird_rate = 60) :
  ∃ (fish_portion bird_portion : ℚ),
    fish_portion = 2/3 ∧ 
    bird_portion = 1/3 ∧
    fish_portion + bird_portion = 1 :=
sorry

end NUMINAMATH_CALUDE_apple_consumption_theorem_l679_67900


namespace NUMINAMATH_CALUDE_neznaika_contradiction_l679_67943

theorem neznaika_contradiction (S T : ℝ) 
  (h1 : S ≤ 50 * T) 
  (h2 : 60 * T ≤ S) 
  (h3 : T > 0) : 
  False :=
by sorry

end NUMINAMATH_CALUDE_neznaika_contradiction_l679_67943


namespace NUMINAMATH_CALUDE_nils_geese_count_l679_67931

/-- Represents the number of geese Nils initially has. -/
def initial_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the initial number of geese. -/
def initial_days : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day. -/
def feed_per_goose_per_day : ℝ := sorry

/-- Represents the total amount of feed available. -/
def total_feed : ℝ := sorry

/-- The feed lasts 20 days longer when 50 geese are sold. -/
axiom sell_condition : total_feed = feed_per_goose_per_day * (initial_days + 20) * (initial_geese - 50)

/-- The feed lasts 10 days less when 100 geese are bought. -/
axiom buy_condition : total_feed = feed_per_goose_per_day * (initial_days - 10) * (initial_geese + 100)

/-- The initial amount of feed equals the product of initial days, initial geese, and feed per goose per day. -/
axiom initial_condition : total_feed = feed_per_goose_per_day * initial_days * initial_geese

/-- Theorem stating that Nils initially has 300 geese. -/
theorem nils_geese_count : initial_geese = 300 := by sorry

end NUMINAMATH_CALUDE_nils_geese_count_l679_67931


namespace NUMINAMATH_CALUDE_dependent_variable_influences_l679_67982

/-- Represents a linear regression model --/
structure LinearRegressionModel where
  y : ℝ  -- dependent variable
  x : ℝ  -- independent variable
  b : ℝ  -- slope
  a : ℝ  -- intercept
  e : ℝ  -- random error term

/-- The linear regression equation --/
def linear_regression_equation (model : LinearRegressionModel) : ℝ :=
  model.b * model.x + model.a + model.e

/-- Theorem stating that the dependent variable is influenced by both the independent variable and other factors --/
theorem dependent_variable_influences (model : LinearRegressionModel) :
  ∃ (other_factors : ℝ), model.y = linear_regression_equation model ∧ model.e ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_dependent_variable_influences_l679_67982


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l679_67981

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 37^(37*(28^28)) ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l679_67981


namespace NUMINAMATH_CALUDE_can_determine_ten_gram_coins_can_determine_coin_weight_l679_67949

/-- Represents the weight of coins in grams -/
inductive CoinWeight
  | Ten
  | Eleven
  | Twelve
  | Thirteen
  | Fourteen

/-- Represents a bag of coins -/
structure Bag where
  weight : CoinWeight
  count : Nat
  h_count : count = 100

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a collection of bags -/
structure BagCollection where
  bags : Fin 5 → Bag
  h_distinct : ∀ i j, i ≠ j → (bags i).weight ≠ (bags j).weight

/-- Function to perform a weighing -/
noncomputable def weigh (left right : List Nat) : WeighingResult :=
  sorry

/-- Theorem stating that it's possible to determine if a specific bag contains 10g coins with one weighing -/
theorem can_determine_ten_gram_coins (bags : BagCollection) (pointed : Fin 5) : 
  ∃ (left right : List Nat), 
    (∀ n ∈ left ∪ right, n ≤ 100) ∧ 
    (weigh left right = WeighingResult.Equal ↔ (bags.bags pointed).weight = CoinWeight.Ten) :=
  sorry

/-- Theorem stating that it's possible to determine the weight of coins in a specific bag with at most two weighings -/
theorem can_determine_coin_weight (bags : BagCollection) (pointed : Fin 5) :
  ∃ (left1 right1 left2 right2 : List Nat),
    (∀ n ∈ left1 ∪ right1 ∪ left2 ∪ right2, n ≤ 100) ∧
    (∃ f : WeighingResult → WeighingResult → CoinWeight,
      f (weigh left1 right1) (weigh left2 right2) = (bags.bags pointed).weight) :=
  sorry

end NUMINAMATH_CALUDE_can_determine_ten_gram_coins_can_determine_coin_weight_l679_67949


namespace NUMINAMATH_CALUDE_ellipse_equation_from_shared_focus_l679_67915

/-- Given a parabola and an ellipse with a shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_from_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x^2/a^2 + y^2 = 1 ∧ x = 2) →
  (∀ (x y : ℝ), x^2/8 + y^2/4 = 1 ↔ x^2/a^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_shared_focus_l679_67915


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l679_67908

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 7 = a) :
  (a - (2*a - 1) / a) / ((a - 1) / a^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l679_67908


namespace NUMINAMATH_CALUDE_monotonic_decreasing_intervals_l679_67924

/-- The function f(x) = (x + 1) / x is monotonically decreasing on (-∞, 0) and (0, +∞) -/
theorem monotonic_decreasing_intervals (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x = (x + 1) / x) →
  (StrictMonoOn f (Set.Iio 0) ∧ StrictMonoOn f (Set.Ioi 0)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_intervals_l679_67924


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l679_67901

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x+y-z| + |y+z-x| + |z+x-y| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l679_67901


namespace NUMINAMATH_CALUDE_curve_c_properties_l679_67930

/-- The curve C in a rectangular coordinate system -/
structure CurveC where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on the curve C -/
structure PointOnC (c : CurveC) where
  φ : ℝ
  x : ℝ
  y : ℝ
  h_x : x = c.a * Real.cos φ
  h_y : y = c.b * Real.sin φ

/-- Theorem about the curve C -/
theorem curve_c_properties (c : CurveC) 
  (m : PointOnC c) 
  (h_m_x : m.x = 2) 
  (h_m_y : m.y = Real.sqrt 3) 
  (h_m_φ : m.φ = π / 3) :
  (∀ x y, x^2 / 16 + y^2 / 4 = 1 ↔ ∃ φ, x = c.a * Real.cos φ ∧ y = c.b * Real.sin φ) ∧
  (∀ ρ₁ ρ₂ θ, 
    (∃ φ₁, ρ₁ * Real.cos θ = c.a * Real.cos φ₁ ∧ ρ₁ * Real.sin θ = c.b * Real.sin φ₁) →
    (∃ φ₂, ρ₂ * Real.cos (θ + π/2) = c.a * Real.cos φ₂ ∧ ρ₂ * Real.sin (θ + π/2) = c.b * Real.sin φ₂) →
    1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 16) :=
by sorry

end NUMINAMATH_CALUDE_curve_c_properties_l679_67930


namespace NUMINAMATH_CALUDE_chord_equation_l679_67958

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

theorem chord_equation (m n s t : ℝ) 
  (h_positive : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_sum : m + n = 2)
  (h_ratio : m / s + n / t = 9)
  (h_min : s + t = 4 / 9)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧ 
    m = (x₁ + x₂) / 2 ∧ 
    n = (y₁ + y₂) / 2) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l679_67958


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l679_67960

open Real

theorem indefinite_integral_proof (x : ℝ) : 
  deriv (fun x => (1/2) * log (abs (x^2 - x + 1)) + 
                  Real.sqrt 3 * arctan ((2*x - 1) / Real.sqrt 3) + 
                  (1/2) * log (abs (x^2 + 1))) x = 
  (2*x^3 + 2*x + 1) / ((x^2 - x + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l679_67960


namespace NUMINAMATH_CALUDE_min_side_length_square_l679_67904

theorem min_side_length_square (s : ℝ) : s ≥ 0 → s ^ 2 ≥ 900 → s ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_square_l679_67904


namespace NUMINAMATH_CALUDE_hcd_7560_270_minus_4_l679_67927

theorem hcd_7560_270_minus_4 : Nat.gcd 7560 270 - 4 = 266 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7560_270_minus_4_l679_67927


namespace NUMINAMATH_CALUDE_log_exponent_simplification_l679_67955

theorem log_exponent_simplification :
  Real.log 2 + Real.log 5 - 42 * (8 ^ (1/4 : ℝ)) - (2017 ^ (0 : ℝ)) = -2 :=
by sorry

end NUMINAMATH_CALUDE_log_exponent_simplification_l679_67955


namespace NUMINAMATH_CALUDE_container_initial_percentage_l679_67911

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 20 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_container_initial_percentage_l679_67911


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l679_67961

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  A = 48 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l679_67961


namespace NUMINAMATH_CALUDE_alpha_value_at_negative_four_l679_67923

/-- Given that α is inversely proportional to β², prove that α = 5/4 when β = -4, 
    given that α = 5 when β = 2. -/
theorem alpha_value_at_negative_four (α β : ℝ) (k : ℝ) 
  (h1 : ∀ β, α * β^2 = k)  -- α is inversely proportional to β²
  (h2 : α = 5 ∧ β = 2 → k = 20)  -- α = 5 when β = 2
  : β = -4 → α = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_at_negative_four_l679_67923


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l679_67990

theorem triangle_cosine_problem (A B C : ℝ) (a b : ℝ) :
  -- Conditions
  A + B + C = Real.pi ∧  -- Sum of angles in a triangle
  B = (A + B + C) / 3 ∧  -- Angles form arithmetic sequence
  a = 8 ∧ 
  b = 7 ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Conclusion
  Real.cos C = -11/14 ∨ Real.cos C = -13/14 := by
sorry


end NUMINAMATH_CALUDE_triangle_cosine_problem_l679_67990


namespace NUMINAMATH_CALUDE_expression_evaluation_l679_67989

theorem expression_evaluation : 
  (200^2 - 13^2) / (140^2 - 23^2) * ((140 - 23) * (140 + 23)) / ((200 - 13) * (200 + 13)) + 1/10 = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l679_67989


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l679_67920

theorem beef_weight_before_processing 
  (weight_after : ℝ) 
  (percent_lost : ℝ) 
  (h1 : weight_after = 546) 
  (h2 : percent_lost = 35) : 
  ∃ weight_before : ℝ, 
    weight_before * (1 - percent_lost / 100) = weight_after ∧ 
    weight_before = 840 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l679_67920


namespace NUMINAMATH_CALUDE_intersection_M_N_l679_67944

def M : Set ℤ := {x | ∃ a : ℤ, x = a^2 + 1}
def N : Set ℤ := {y | 1 ≤ y ∧ y ≤ 6}

theorem intersection_M_N : M ∩ N = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l679_67944


namespace NUMINAMATH_CALUDE_equation_solutions_l679_67954

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 9 = 4 * x ∧ x = -9/2) ∧
  (∃ x : ℚ, (5/2) * x - (7/3) * x = (4/3) * 5 - 5 ∧ x = 10) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l679_67954


namespace NUMINAMATH_CALUDE_mary_flour_added_l679_67970

def recipe_flour : ℕ := 12
def recipe_salt : ℕ := 7
def extra_flour : ℕ := 3

theorem mary_flour_added (flour_added : ℕ) : 
  flour_added = recipe_flour - (recipe_salt + extra_flour) → flour_added = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_added_l679_67970


namespace NUMINAMATH_CALUDE_reader_current_page_l679_67991

/-- Represents a book with a given number of pages -/
structure Book where
  pages : ℕ

/-- Represents a reader with a constant reading rate -/
structure Reader where
  rate : ℕ  -- pages per hour

/-- The current state of reading -/
structure ReadingState where
  book : Book
  reader : Reader
  previousPage : ℕ
  hoursAgo : ℕ
  hoursLeft : ℕ

/-- Calculate the current page number of the reader -/
def currentPage (state : ReadingState) : ℕ :=
  state.previousPage + state.reader.rate * state.hoursAgo

/-- Theorem: Given the conditions, prove that the reader's current page is 90 -/
theorem reader_current_page
  (state : ReadingState)
  (h1 : state.book.pages = 210)
  (h2 : state.previousPage = 60)
  (h3 : state.hoursAgo = 1)
  (h4 : state.hoursLeft = 4)
  (h5 : currentPage state + state.reader.rate * state.hoursLeft = state.book.pages) :
  currentPage state = 90 := by
  sorry


end NUMINAMATH_CALUDE_reader_current_page_l679_67991


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l679_67983

theorem students_playing_both_sports 
  (total : ℕ) 
  (hockey : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_neither : neither = 4) :
  hockey + basketball - (total - neither) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l679_67983


namespace NUMINAMATH_CALUDE_test_score_proof_l679_67948

theorem test_score_proof (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_points = 20 →
  incorrect_points = 5 →
  total_score = 325 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 19 :=
by sorry

end NUMINAMATH_CALUDE_test_score_proof_l679_67948


namespace NUMINAMATH_CALUDE_imaginary_part_of_z2_l679_67980

theorem imaginary_part_of_z2 (z₁ : ℂ) (h : z₁ = 1 - 2*I) :
  Complex.im ((z₁ + 1) / (z₁ - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z2_l679_67980


namespace NUMINAMATH_CALUDE_states_fraction_1800_1809_l679_67933

theorem states_fraction_1800_1809 (total_states : Nat) (states_1800_1809 : Nat) :
  total_states = 30 →
  states_1800_1809 = 5 →
  (states_1800_1809 : ℚ) / total_states = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1800_1809_l679_67933


namespace NUMINAMATH_CALUDE_eight_balls_three_boxes_l679_67967

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 45 ways to distribute 8 indistinguishable balls into 3 distinguishable boxes -/
theorem eight_balls_three_boxes : distribute_balls 8 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_eight_balls_three_boxes_l679_67967


namespace NUMINAMATH_CALUDE_remainder_problem_l679_67909

theorem remainder_problem : ∃ q : ℕ, 
  6598574241545098875458255622898854689448911257658451215825362549889 = 
  3721858987156557895464215545212524189541456658712589687354871258 * q + 8 * 23 + r ∧ 
  r < 23 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l679_67909


namespace NUMINAMATH_CALUDE_percentage_of_b_l679_67999

/-- Given that 8 is 4% of a, a certain percentage of b is 4, and c equals b / a, 
    prove that the percentage of b is 1 / (50c) -/
theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 1 / (50 * c) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l679_67999


namespace NUMINAMATH_CALUDE_f_bound_and_g_monotonicity_l679_67903

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

theorem f_bound_and_g_monotonicity :
  (∃ c : ℝ, c = -1 ∧ ∀ x > 0, f x ≤ 2 * x + c) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioo 0 a)) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioi a)) :=
sorry

end NUMINAMATH_CALUDE_f_bound_and_g_monotonicity_l679_67903


namespace NUMINAMATH_CALUDE_used_car_selection_l679_67953

theorem used_car_selection (num_cars num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 15 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 :=
by sorry

end NUMINAMATH_CALUDE_used_car_selection_l679_67953


namespace NUMINAMATH_CALUDE_odd_factors_of_360_is_6_l679_67956

/-- The number of odd factors of 360 -/
def odd_factors_of_360 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_of_360_is_6 : odd_factors_of_360 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_factors_of_360_is_6_l679_67956


namespace NUMINAMATH_CALUDE_brad_read_more_books_l679_67905

def william_last_month : ℕ := 6
def brad_this_month : ℕ := 8

def brad_last_month : ℕ := 3 * william_last_month
def william_this_month : ℕ := 2 * brad_this_month

def william_total : ℕ := william_last_month + william_this_month
def brad_total : ℕ := brad_last_month + brad_this_month

theorem brad_read_more_books : brad_total = william_total + 4 := by
  sorry

end NUMINAMATH_CALUDE_brad_read_more_books_l679_67905


namespace NUMINAMATH_CALUDE_arcsin_negative_half_l679_67925

theorem arcsin_negative_half : Real.arcsin (-1/2) = -π/6 := by sorry

end NUMINAMATH_CALUDE_arcsin_negative_half_l679_67925


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l679_67940

theorem right_triangle_leg_square (a b c : ℝ) : 
  (a^2 + b^2 = c^2) →  -- right triangle condition
  (c = a + 2) →        -- hypotenuse is 2 units longer than leg a
  b^2 = 4*(a + 1) :=   -- square of other leg b
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l679_67940


namespace NUMINAMATH_CALUDE_wilfred_carrot_consumption_l679_67992

/-- The number of carrots Wilfred eats on Tuesday -/
def tuesday_carrots : ℕ := 4

/-- The number of carrots Wilfred eats on Wednesday -/
def wednesday_carrots : ℕ := 6

/-- The number of carrots Wilfred plans to eat on Thursday -/
def thursday_carrots : ℕ := 5

/-- The total number of carrots Wilfred wants to eat from Tuesday to Thursday -/
def total_carrots : ℕ := tuesday_carrots + wednesday_carrots + thursday_carrots

theorem wilfred_carrot_consumption :
  total_carrots = 15 := by sorry

end NUMINAMATH_CALUDE_wilfred_carrot_consumption_l679_67992


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l679_67966

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l679_67966


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l679_67971

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Theorem stating that collinearity of three out of four points is sufficient but not necessary for coplanarity -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → coplanar p q r s) ∧
  (∃ p q r s : Point3D, coplanar p q r s ∧ ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s) :=
sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l679_67971


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l679_67928

theorem line_segment_endpoint (y : ℝ) : y > 0 →
  (Real.sqrt (((-7) - 3)^2 + (y - (-2))^2) = 13) →
  y = -2 + Real.sqrt 69 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l679_67928


namespace NUMINAMATH_CALUDE_inequality_proof_l679_67985

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a / b + b / c > a / c + c / a := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l679_67985


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014_l679_67941

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2014 :
  arithmetic_sequence 4 3 671 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014_l679_67941


namespace NUMINAMATH_CALUDE_never_equal_implies_m_range_l679_67968

theorem never_equal_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6) →
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_never_equal_implies_m_range_l679_67968


namespace NUMINAMATH_CALUDE_propositions_analysis_l679_67947

theorem propositions_analysis :
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ a * c^2 ≤ b * c^2) ∧
  (∃ (a b : ℝ), a < b ∧ 1/a ≤ 1/b) ∧
  (∀ (a b : ℝ), a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ (a b : ℝ), a > abs b → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_propositions_analysis_l679_67947


namespace NUMINAMATH_CALUDE_all_triplets_sum_to_two_l679_67996

theorem all_triplets_sum_to_two :
  (3/4 + 1/4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3/5 + 4/5 + 3/5 = 2) ∧
  (2 + (-3) + 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_all_triplets_sum_to_two_l679_67996


namespace NUMINAMATH_CALUDE_max_money_is_twelve_dollars_l679_67973

/-- Represents the recycling scenario with given rates and collected items -/
structure RecyclingScenario where
  can_rate : Rat -- Money received for 12 cans
  newspaper_rate : Rat -- Money received for 5 kg of newspapers
  bottle_rate : Rat -- Money received for 3 glass bottles
  weight_limit : Rat -- Weight limit in kg
  cans_collected : Nat -- Number of cans collected
  can_weight : Rat -- Weight of each can in kg
  newspapers_collected : Rat -- Weight of newspapers collected in kg
  bottles_collected : Nat -- Number of bottles collected
  bottle_weight : Rat -- Weight of each bottle in kg

/-- Calculates the maximum money received from recycling -/
noncomputable def max_money_received (scenario : RecyclingScenario) : Rat :=
  sorry

/-- Theorem stating that the maximum money received is $12.00 -/
theorem max_money_is_twelve_dollars (scenario : RecyclingScenario) 
  (h1 : scenario.can_rate = 1/2)
  (h2 : scenario.newspaper_rate = 3/2)
  (h3 : scenario.bottle_rate = 9/10)
  (h4 : scenario.weight_limit = 25)
  (h5 : scenario.cans_collected = 144)
  (h6 : scenario.can_weight = 3/100)
  (h7 : scenario.newspapers_collected = 20)
  (h8 : scenario.bottles_collected = 30)
  (h9 : scenario.bottle_weight = 1/2) :
  max_money_received scenario = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_money_is_twelve_dollars_l679_67973


namespace NUMINAMATH_CALUDE_min_value_expression_l679_67979

theorem min_value_expression (n : ℕ+) : 
  (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) ≥ 5 ∧ 
  (∃ n : ℕ+, (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) = 5) := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l679_67979


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l679_67998

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l679_67998


namespace NUMINAMATH_CALUDE_cubic_root_sum_l679_67988

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 28 * a - 2 = 0) →
  (40 * b^3 - 60 * b^2 + 28 * b - 2 = 0) →
  (40 * c^3 - 60 * c^2 + 28 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l679_67988


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l679_67950

/-- Given that if a person walks at 16 km/hr instead of 12 km/hr, they would have walked 20 km more,
    prove that the actual distance traveled is 60 km. -/
theorem actual_distance_traveled (D : ℝ) : 
  (D / 12 = (D + 20) / 16) → D = 60 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l679_67950


namespace NUMINAMATH_CALUDE_proportion_solution_l679_67913

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/6) : x = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l679_67913


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l679_67993

theorem square_ratio_theorem :
  let area_ratio : ℚ := 18 / 98
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    side_ratio = (a : ℝ) * Real.sqrt b / c ∧
    a = 3 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l679_67993


namespace NUMINAMATH_CALUDE_z_value_l679_67922

theorem z_value (x y z : ℝ) (h : 1/x + 1/y = 2/z) : z = x*y/2 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l679_67922


namespace NUMINAMATH_CALUDE_constant_in_exponent_l679_67918

theorem constant_in_exponent (w : ℕ) (h1 : 2^(2*w) = 8^(w-4)) (h2 : w = 12) : 
  ∃ k : ℕ, 2^(2*w) = 8^(w-k) ∧ k = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_in_exponent_l679_67918


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l679_67964

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 3.75 ∧ total_cost = 12375 →
  total_cost / (length * width) = 600 := by
  sorry

#check paving_rate_calculation

end NUMINAMATH_CALUDE_paving_rate_calculation_l679_67964


namespace NUMINAMATH_CALUDE_apples_left_l679_67959

theorem apples_left (frank_apples susan_apples : ℕ) : 
  frank_apples = 36 →
  susan_apples = 3 * frank_apples →
  (frank_apples - frank_apples / 3) + (susan_apples - susan_apples / 2) = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_left_l679_67959


namespace NUMINAMATH_CALUDE_base_k_theorem_l679_67917

theorem base_k_theorem (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_theorem_l679_67917


namespace NUMINAMATH_CALUDE_total_sums_attempted_l679_67986

/-- Given a student's performance on a set of math problems, calculate the total number of problems attempted. -/
theorem total_sums_attempted
  (correct : ℕ)  -- Number of sums solved correctly
  (h1 : correct = 12)  -- The student solved 12 sums correctly
  (h2 : ∃ wrong : ℕ, wrong = 2 * correct)  -- The student got twice as many sums wrong as right
  : ∃ total : ℕ, total = 3 * correct :=
by sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l679_67986


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l679_67916

theorem geometric_progression_solution (b₁ q : ℝ) : 
  b₁ * (1 + q + q^2) = 21 ∧ 
  b₁^2 * (1 + q^2 + q^4) = 189 → 
  ((b₁ = 3 ∧ q = 2) ∨ (b₁ = 12 ∧ q = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l679_67916


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l679_67987

theorem basketball_lineup_combinations (n : ℕ) (k₁ k₂ k₃ : ℕ) 
  (h₁ : n = 12) (h₂ : k₁ = 2) (h₃ : k₂ = 2) (h₄ : k₃ = 1) : 
  (n.choose k₁) * ((n - k₁).choose k₂) * ((n - k₁ - k₂).choose k₃) = 23760 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l679_67987


namespace NUMINAMATH_CALUDE_min_selection_for_tenfold_l679_67926

theorem min_selection_for_tenfold (n : ℕ) (h : n = 2020) :
  ∃ k : ℕ, k = 203 ∧
  (∀ S : Finset ℕ, S.card < k → S ⊆ Finset.range (n + 1) →
    ¬∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) ∧
  (∃ S : Finset ℕ, S.card = k ∧ S ⊆ Finset.range (n + 1) ∧
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) :=
by sorry

end NUMINAMATH_CALUDE_min_selection_for_tenfold_l679_67926


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l679_67937

/-- The number of seats in a row -/
def num_seats : ℕ := 7

/-- The number of persons to be seated -/
def num_persons : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial) / ((n - k).factorial)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_seats num_persons - 
  (num_seats - 1) * seating_arrangements 2 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l679_67937


namespace NUMINAMATH_CALUDE_three_points_on_circle_at_distance_from_line_l679_67963

theorem three_points_on_circle_at_distance_from_line :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 3 ∧
  (∀ p ∈ points, p.1^2 + p.2^2 = 4 ∧
    (|p.1 - p.2 + Real.sqrt 2|) / Real.sqrt 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_three_points_on_circle_at_distance_from_line_l679_67963


namespace NUMINAMATH_CALUDE_negation_of_existence_l679_67969

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l679_67969


namespace NUMINAMATH_CALUDE_subsequence_theorem_l679_67977

theorem subsequence_theorem (seq : List ℕ) (h1 : seq.length = 101) 
  (h2 : ∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 101) 
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 101 → n ∈ seq) :
  ∃ subseq : List ℕ, subseq.length = 11 ∧ 
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ < subseq.get ⟨j, by sorry⟩) ∨
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ > subseq.get ⟨j, by sorry⟩) :=
sorry

end NUMINAMATH_CALUDE_subsequence_theorem_l679_67977


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l679_67975

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ -3 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l679_67975
