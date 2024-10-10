import Mathlib

namespace sue_answer_formula_l421_42107

/-- Given Ben's initial number, calculate Sue's final answer -/
def sueAnswer (x : ℕ) : ℕ :=
  let benResult := 2 * (2 * x + 1)
  2 * (benResult - 1)

/-- Theorem: Sue's answer is always 4x + 2, where x is Ben's initial number -/
theorem sue_answer_formula (x : ℕ) : sueAnswer x = 4 * x + 2 := by
  sorry

#eval sueAnswer 8  -- Should output 66

end sue_answer_formula_l421_42107


namespace marble_distribution_l421_42179

theorem marble_distribution (total_marbles : ℕ) (people : ℕ) : 
  total_marbles = 180 →
  (total_marbles / people : ℚ) - (total_marbles / (people + 2) : ℚ) = 1 →
  people = 18 := by
  sorry

end marble_distribution_l421_42179


namespace mans_swimming_speed_l421_42115

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 2.5 km/h. -/
theorem mans_swimming_speed (v : ℝ) (s : ℝ) (h1 : s = 2.5) 
    (h2 : ∃ t : ℝ, t > 0 ∧ (v + s) * t = (v - s) * (2 * t)) : v = 7.5 := by
  sorry

end mans_swimming_speed_l421_42115


namespace polynomial_product_sum_l421_42187

theorem polynomial_product_sum (p q : ℚ) : 
  (∀ x, (4 * x^2 - 5 * x + p) * (6 * x^2 + q * x - 12) = 
   24 * x^4 - 62 * x^3 - 69 * x^2 + 94 * x - 36) → 
  p + q = 43 / 3 := by
  sorry

end polynomial_product_sum_l421_42187


namespace complex_conversion_l421_42161

theorem complex_conversion :
  3 * Real.sqrt 2 * Complex.exp ((-5 * π * Complex.I) / 4) = -3 - 3 * Complex.I :=
by sorry

end complex_conversion_l421_42161


namespace complex_magnitude_example_l421_42162

theorem complex_magnitude_example : Complex.abs (Complex.mk (7/8) 3) = 25/8 := by
  sorry

end complex_magnitude_example_l421_42162


namespace mildred_oranges_l421_42160

/-- The number of oranges Mildred's father ate -/
def fatherAte : ℕ := 2

/-- The number of oranges Mildred has now -/
def currentOranges : ℕ := 75

/-- The initial number of oranges Mildred collected -/
def initialOranges : ℕ := currentOranges + fatherAte

theorem mildred_oranges : initialOranges = 77 := by
  sorry

end mildred_oranges_l421_42160


namespace minimum_value_implies_k_l421_42192

/-- Given that k is a positive constant and the minimum value of the function
    y = x^2 + k/x (where x > 0) is 3, prove that k = 2. -/
theorem minimum_value_implies_k (k : ℝ) (h1 : k > 0) :
  (∀ x > 0, x^2 + k/x ≥ 3) ∧ (∃ x > 0, x^2 + k/x = 3) → k = 2 := by
  sorry

end minimum_value_implies_k_l421_42192


namespace race_finish_difference_l421_42159

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the race with three runners -/
structure Race where
  runner1 : Runner
  runner2 : Runner
  runner3 : Runner
  constant_speed : Prop

/-- The difference in distance between two runners at the finish line -/
def distance_difference (r1 r2 : Runner) : ℝ :=
  r1.distance - r2.distance

/-- The theorem statement -/
theorem race_finish_difference (race : Race) 
  (h1 : distance_difference race.runner1 race.runner2 = 2)
  (h2 : distance_difference race.runner1 race.runner3 = 4)
  (h3 : race.constant_speed) :
  distance_difference race.runner2 race.runner3 = 2.5 := by
  sorry

end race_finish_difference_l421_42159


namespace blocks_shared_l421_42111

theorem blocks_shared (start_blocks end_blocks : ℝ) (h1 : start_blocks = 86.0) (h2 : end_blocks = 127) : 
  end_blocks - start_blocks = 41 := by
sorry

end blocks_shared_l421_42111


namespace no_valid_equation_l421_42142

/-- Represents a letter in the equation -/
structure Letter where
  value : Nat
  property : value < 10

/-- Represents a two-digit number as a pair of letters -/
structure TwoDigitNumber where
  tens : Letter
  ones : Letter
  different : tens ≠ ones

/-- Represents the equation АБ×ВГ = ДДЕЕ -/
structure Equation where
  ab : TwoDigitNumber
  vg : TwoDigitNumber
  d : Letter
  e : Letter
  different_letters : ab.tens ≠ ab.ones ∧ ab.tens ≠ vg.tens ∧ ab.tens ≠ vg.ones ∧
                      ab.ones ≠ vg.tens ∧ ab.ones ≠ vg.ones ∧ vg.tens ≠ vg.ones ∧
                      d ≠ e
  valid_multiplication : ab.tens.value * 10 + ab.ones.value *
                         (vg.tens.value * 10 + vg.ones.value) =
                         d.value * 1000 + d.value * 100 + e.value * 10 + e.value

theorem no_valid_equation : ¬ ∃ (eq : Equation), True := by
  sorry

end no_valid_equation_l421_42142


namespace system_solution_l421_42174

theorem system_solution (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = -14)
  (eq2 : 6 * u + 5 * v = 7) :
  2 * u - v = -63/13 := by
sorry

end system_solution_l421_42174


namespace prime_divides_product_implies_divides_factor_l421_42125

theorem prime_divides_product_implies_divides_factor 
  (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
  (h_prime : Nat.Prime p) 
  (h_divides_product : p ∣ (Finset.univ.prod a)) : 
  ∃ i, p ∣ a i :=
sorry

end prime_divides_product_implies_divides_factor_l421_42125


namespace travel_ratio_l421_42130

theorem travel_ratio (total : ℕ) (europe : ℕ) (south_america : ℕ) (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : asia = 6)
  (h5 : europe + south_america + asia ≤ total) :
  asia * 2 = total - europe - south_america :=
by sorry

end travel_ratio_l421_42130


namespace race_catch_up_time_l421_42121

/-- Given a 10-mile race with two runners, where the first runner's pace is 8 minutes per mile
    and the second runner's pace is 7 minutes per mile, prove that if the second runner stops
    after 56 minutes, they can remain stopped for 8 minutes before the first runner catches up. -/
theorem race_catch_up_time (race_length : ℝ) (pace1 pace2 stop_time : ℝ) :
  race_length = 10 →
  pace1 = 8 →
  pace2 = 7 →
  stop_time = 56 →
  let distance1 := stop_time / pace1
  let distance2 := stop_time / pace2
  let distance_diff := distance2 - distance1
  distance_diff * pace1 = 8 := by sorry

end race_catch_up_time_l421_42121


namespace even_function_range_l421_42152

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_range (f : ℝ → ℝ) (h_even : IsEven f)
  (h_cond : ∀ x₁ x₂, x₁ ∈ Set.Ici 0 ∧ x₂ ∈ Set.Ici 0 ∧ x₁ ≠ x₂ → 
    (x₁ - x₂) * (f x₁ - f x₂) > 0)
  (m : ℝ) (h_ineq : f (m + 1) ≥ f 2) :
  m ∈ Set.Iic (-3) ∪ Set.Ici 1 := by
  sorry

end even_function_range_l421_42152


namespace min_distance_to_line_l421_42172

/-- The minimum distance from a point on y = e^x + x to the line 2x-y-3=0 -/
theorem min_distance_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x + x
  let P : ℝ × ℝ := (0, f 0)
  let d (x y : ℝ) : ℝ := |2*x - y - 3| / Real.sqrt (2^2 + (-1)^2)
  ∀ x : ℝ, d x (f x) ≥ d P.1 P.2 ∧ d P.1 P.2 = 4 * Real.sqrt 5 / 5 :=
by sorry


end min_distance_to_line_l421_42172


namespace cubic_sum_fraction_l421_42181

theorem cubic_sum_fraction (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)) = -3 / (2*(x^2 + y^2 + x*y)) :=
by sorry

end cubic_sum_fraction_l421_42181


namespace optimal_warehouse_location_l421_42191

/-- The optimal warehouse location problem -/
theorem optimal_warehouse_location 
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) (h₁ : ∀ x > 0, y₁ x = k₁ / x) 
  (h₂ : ∀ x > 0, y₂ x = k₂ * x) (h₃ : k₁ > 0) (h₄ : k₂ > 0)
  (h₅ : y₁ 10 = 4) (h₆ : y₂ 10 = 16) :
  ∃ x₀ > 0, ∀ x > 0, y₁ x + y₂ x ≥ y₁ x₀ + y₂ x₀ ∧ x₀ = 5 :=
sorry

end optimal_warehouse_location_l421_42191


namespace cone_volume_from_triangle_rotation_l421_42155

/-- The volume of a cone formed by rotating a right triangle -/
def cone_volume (S L : ℝ) : ℝ :=
  S * L

/-- Theorem: The volume of a cone formed by rotating a right triangle with area S
    around one of its legs is equal to SL, where L is the length of the circumference
    described by the intersection point of the medians during rotation -/
theorem cone_volume_from_triangle_rotation (S L : ℝ) (h1 : S > 0) (h2 : L > 0) :
  cone_volume S L = S * L :=
by
  sorry

end cone_volume_from_triangle_rotation_l421_42155


namespace linear_transformation_mapping_l421_42119

theorem linear_transformation_mapping (x : ℝ) :
  0 ≤ x ∧ x ≤ 1 → -1 ≤ 4 * x - 1 ∧ 4 * x - 1 ≤ 3 := by
  sorry

end linear_transformation_mapping_l421_42119


namespace most_suitable_sampling_method_l421_42197

-- Define the population structure
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

-- Define the sampling method
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | RemoveOneElderlyThenStratified

-- Define the suitability of a sampling method
def isMostSuitable (pop : Population) (sampleSize : ℕ) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.RemoveOneElderlyThenStratified

-- Theorem statement
theorem most_suitable_sampling_method 
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : ℕ)
  (h4 : sampleSize = 36) :
  isMostSuitable pop sampleSize SamplingMethod.RemoveOneElderlyThenStratified :=
by
  sorry

end most_suitable_sampling_method_l421_42197


namespace Y_two_five_l421_42151

def Y (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2 + 3

theorem Y_two_five : Y 2 5 = 2 := by sorry

end Y_two_five_l421_42151


namespace football_practice_hours_l421_42126

/-- The number of hours a football team practices daily, given their weekly schedule and total practice time. -/
def daily_practice_hours (total_hours : ℕ) (practice_days : ℕ) : ℚ :=
  total_hours / practice_days

/-- Theorem stating that the daily practice hours is 6, given the conditions of the problem. -/
theorem football_practice_hours :
  let total_week_hours : ℕ := 36
  let days_in_week : ℕ := 7
  let rain_days : ℕ := 1
  let practice_days : ℕ := days_in_week - rain_days
  daily_practice_hours total_week_hours practice_days = 6 := by
  sorry

end football_practice_hours_l421_42126


namespace first_shirt_costs_15_l421_42143

/-- The cost of the first shirt given the conditions of the problem -/
def first_shirt_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + 6

/-- The total cost of both shirts -/
def total_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + first_shirt_cost second_shirt_cost

theorem first_shirt_costs_15 :
  ∃ (second_shirt_cost : ℝ),
    first_shirt_cost second_shirt_cost = 15 ∧
    total_cost second_shirt_cost = 24 :=
by
  sorry

end first_shirt_costs_15_l421_42143


namespace div_ratio_problem_l421_42110

theorem div_ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 3 / 4)
  (h3 : c / d = 2 / 3) :
  d / a = 2 / 3 := by
  sorry

end div_ratio_problem_l421_42110


namespace evaluate_expression_l421_42186

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem evaluate_expression (x y : ℝ) : 
  dollar ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 := by
  sorry

end evaluate_expression_l421_42186


namespace intersects_once_impl_a_eq_one_l421_42124

/-- The function f(x) for a given 'a' -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

/-- Predicate to check if f(x) intersects x-axis at exactly one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem: If f(x) intersects x-axis at exactly one point, then a = 1 -/
theorem intersects_once_impl_a_eq_one :
  ∀ a : ℝ, intersects_once a → a = 1 := by
sorry

end intersects_once_impl_a_eq_one_l421_42124


namespace max_abcd_is_one_l421_42146

theorem max_abcd_is_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : (1 + a) * (1 + b) * (1 + c) * (1 + d) = 16) :
  abcd ≤ 1 ∧ ∃ (a' b' c' d' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    (1 + a') * (1 + b') * (1 + c') * (1 + d') = 16 ∧ a' * b' * c' * d' = 1 := by
  sorry

end max_abcd_is_one_l421_42146


namespace sum_of_roots_is_36_l421_42114

def f (x : ℝ) : ℝ := (11 - x)^3 + (13 - x)^3 - (24 - 2*x)^3

theorem sum_of_roots_is_36 :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    x₁ + x₂ + x₃ = 36 := by
  sorry

end sum_of_roots_is_36_l421_42114


namespace flower_count_l421_42164

theorem flower_count : 
  ∀ (flowers bees : ℕ), 
    bees = 3 → 
    bees = flowers - 2 → 
    flowers = 5 := by sorry

end flower_count_l421_42164


namespace extreme_values_of_f_l421_42176

def f (x : ℝ) := x^3 - 12*x + 12

theorem extreme_values_of_f :
  (∃ x, f x = -4 ∧ x = 2) ∧
  (∃ x, f x = 28) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 28) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 28) :=
by sorry

end extreme_values_of_f_l421_42176


namespace fraction_simplification_l421_42102

theorem fraction_simplification (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 + a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 :=
by sorry

end fraction_simplification_l421_42102


namespace intersection_M_N_l421_42169

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = interval_1_2 := by sorry

end intersection_M_N_l421_42169


namespace simplify_expression_l421_42133

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end simplify_expression_l421_42133


namespace quadratic_polynomial_problem_l421_42137

theorem quadratic_polynomial_problem : ∃ (q : ℝ → ℝ),
  (∀ x, q x = -4.5 * x^2 - 13.5 * x + 81) ∧
  q (-6) = 0 ∧
  q 3 = 0 ∧
  q 4 = -45 := by
  sorry

end quadratic_polynomial_problem_l421_42137


namespace quadratic_root_equation_l421_42173

theorem quadratic_root_equation (x : ℝ) : 
  (∃ r : ℝ, x = (2 + r * Real.sqrt (4 - 4 * 3 * (-1))) / (2 * 3) ∧ r^2 = 1) →
  (3 * x^2 - 2 * x - 1 = 0) := by
sorry

end quadratic_root_equation_l421_42173


namespace sqrt_product_plus_one_equals_2549_l421_42189

theorem sqrt_product_plus_one_equals_2549 :
  Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549 := by
  sorry

end sqrt_product_plus_one_equals_2549_l421_42189


namespace pure_imaginary_condition_l421_42127

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let Z : ℂ := (a + i) / (1 + i)
  is_pure_imaginary Z → a = -1 := by
  sorry

end pure_imaginary_condition_l421_42127


namespace interest_calculation_l421_42139

/-- Calculates the compound interest earned over a period of time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the compound interest earned on $2000 at 5% for 5 years is approximately $552.56 -/
theorem interest_calculation :
  let principal := 2000
  let rate := 0.05
  let years := 5
  abs (compoundInterest principal rate years - 552.56) < 0.01 := by
sorry

end interest_calculation_l421_42139


namespace white_line_length_l421_42144

theorem white_line_length : 
  let blue_line_length : Float := 3.3333333333333335
  let difference : Float := 4.333333333333333
  let white_line_length : Float := blue_line_length + difference
  white_line_length = 7.666666666666667 := by
sorry

end white_line_length_l421_42144


namespace percentage_change_difference_l421_42154

theorem percentage_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 60 / 100 →
  initial_no = 40 / 100 →
  final_yes = 80 / 100 →
  final_no = 20 / 100 →
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≥ 0 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 20 / 100 :=
by sorry

end percentage_change_difference_l421_42154


namespace fraction_change_l421_42108

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 - (a * 0.8) / (b * 1.28) / (a / b) = 0.375 := by sorry

end fraction_change_l421_42108


namespace tile_border_ratio_l421_42175

theorem tile_border_ratio (t w : ℝ) (h : t > 0) (h' : w > 0) : 
  (900 * t^2) / ((30 * t + 30 * w)^2) = 81/100 → w/t = 1/9 := by
sorry

end tile_border_ratio_l421_42175


namespace function_properties_l421_42195

-- Define the functions y₁ and y₂
def y₁ (a b x : ℝ) : ℝ := x^2 + a*x + b
def y₂ (x : ℝ) : ℝ := x^2 + x - 2

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, |y₁ a b x| ≤ |y₂ x|) →
  (a = 1 ∧ b = -2) ∧
  (∀ m : ℝ, (∀ x > 1, y₁ a b x > (m - 2)*x - m) → m < 2*Real.sqrt 2 + 5) :=
by sorry

end function_properties_l421_42195


namespace total_miles_equals_484_l421_42182

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group per day -/
def group_miles_per_day : ℕ := 3

/-- The number of days per week the group walks together -/
def group_days_per_week : ℕ := 6

/-- Jamie's additional miles walked per day -/
def jamie_additional_miles : ℕ := 2

/-- Sue's additional miles walked per day (half of Jamie's) -/
def sue_additional_miles : ℕ := jamie_additional_miles / 2

/-- Laura's additional miles walked every two days -/
def laura_additional_miles : ℕ := 1

/-- Melissa's additional miles walked every three days -/
def melissa_additional_miles : ℕ := 2

/-- Katie's additional miles walked per day -/
def katie_additional_miles : ℕ := 1

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate the total miles walked by all ladies in the group during a month -/
def total_miles_per_month : ℕ :=
  let jamie_miles := (group_miles_per_day * group_days_per_week + jamie_additional_miles * group_days_per_week) * weeks_per_month
  let sue_miles := (group_miles_per_day * group_days_per_week + sue_additional_miles * group_days_per_week) * weeks_per_month
  let laura_miles := (group_miles_per_day * group_days_per_week + laura_additional_miles * 3) * weeks_per_month
  let melissa_miles := (group_miles_per_day * group_days_per_week + melissa_additional_miles * 2) * weeks_per_month
  let katie_miles := (group_miles_per_day * group_days_per_week + katie_additional_miles * group_days_per_week) * weeks_per_month
  jamie_miles + sue_miles + laura_miles + melissa_miles + katie_miles

theorem total_miles_equals_484 : total_miles_per_month = 484 := by
  sorry

end total_miles_equals_484_l421_42182


namespace least_coins_in_purse_l421_42100

theorem least_coins_in_purse (n : ℕ) : 
  (n % 7 = 3 ∧ n % 5 = 4) → n ≥ 24 :=
sorry

end least_coins_in_purse_l421_42100


namespace triangle_abc_properties_l421_42138

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  (Real.cos A = Real.sqrt 6 / 3) ∧ (c = 5) := by
  sorry

end triangle_abc_properties_l421_42138


namespace circle_passes_through_points_circle_uniqueness_l421_42149

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 --/
def Circle (D E F : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in --/
def SpecificCircle : ℝ × ℝ → Prop :=
  Circle (-4) (-6) 0

theorem circle_passes_through_points :
  SpecificCircle (0, 0) ∧
  SpecificCircle (4, 0) ∧
  SpecificCircle (-1, 1) :=
by sorry

/-- Uniqueness of the circle --/
theorem circle_uniqueness (D E F : ℝ) :
  Circle D E F (0, 0) →
  Circle D E F (4, 0) →
  Circle D E F (-1, 1) →
  ∀ (x y : ℝ), Circle D E F (x, y) ↔ SpecificCircle (x, y) :=
by sorry

end circle_passes_through_points_circle_uniqueness_l421_42149


namespace place_face_value_difference_l421_42101

def numeral : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  (n / 10) % 10 * 10

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value numeral digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end place_face_value_difference_l421_42101


namespace handshake_problem_l421_42194

/-- The number of handshakes in a group of n people where each person shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of men in the group -/
def num_men : ℕ := 60

theorem handshake_problem :
  handshakes num_men = 1770 :=
sorry

#eval handshakes num_men

end handshake_problem_l421_42194


namespace solve_equation_l421_42116

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1/3) * (6 * x + 45) → x = 0 := by
  sorry

end solve_equation_l421_42116


namespace largest_square_4digits_base7_l421_42112

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem largest_square_4digits_base7 :
  (M^2 ≥ 7^3) ∧ (M^2 < 7^4) ∧ (∀ n : ℕ, n > M → n^2 ≥ 7^4) ∧ (toBase7 M = [6, 6]) := by
  sorry

end largest_square_4digits_base7_l421_42112


namespace class_test_result_l421_42117

theorem class_test_result (boys : ℕ) (grade5 : ℕ) : ∃ (low_grade : ℕ), low_grade ≤ 2 ∧ low_grade > 0 := by
  -- Define the number of girls
  let girls : ℕ := boys + 3
  
  -- Define the number of grade 4s
  let grade4 : ℕ := grade5 + 6
  
  -- Define the number of grade 3s
  let grade3 : ℕ := 2 * grade4
  
  -- Define the total number of students
  let total_students : ℕ := boys + girls
  
  -- Define the total number of positive grades (3, 4, 5)
  let total_positive_grades : ℕ := grade3 + grade4 + grade5
  
  -- Proof goes here
  sorry


end class_test_result_l421_42117


namespace multiple_problem_l421_42103

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : 2 * n = m * n - 25) : m = 3 := by
  sorry

end multiple_problem_l421_42103


namespace additional_people_calculation_l421_42120

/-- Represents Carl's open house scenario -/
structure OpenHouse where
  confirmed_attendees : ℕ
  extravagant_bags : ℕ
  initial_average_bags : ℕ
  additional_bags_needed : ℕ

/-- Calculates the number of additional people Carl hopes will show up -/
def additional_people (oh : OpenHouse) : ℕ :=
  (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees

/-- Theorem stating that the number of additional people Carl hopes will show up
    is equal to the total number of gift bags minus the number of confirmed attendees -/
theorem additional_people_calculation (oh : OpenHouse) :
  additional_people oh = (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees :=
by
  sorry

#eval additional_people {
  confirmed_attendees := 50,
  extravagant_bags := 10,
  initial_average_bags := 20,
  additional_bags_needed := 60
}

end additional_people_calculation_l421_42120


namespace thomas_blocks_total_l421_42136

theorem thomas_blocks_total (stack1 stack2 stack3 stack4 stack5 : ℕ) : 
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack5 = 2 * stack2 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end thomas_blocks_total_l421_42136


namespace amount_distributed_l421_42185

theorem amount_distributed (A : ℝ) : 
  (A / 20 = A / 25 + 100) → A = 10000 := by
  sorry

end amount_distributed_l421_42185


namespace number_of_factors_of_M_l421_42198

def M : Nat := 2^5 * 3^4 * 5^3 * 11^2

theorem number_of_factors_of_M : 
  (Finset.filter (·∣M) (Finset.range (M + 1))).card = 360 := by
  sorry

end number_of_factors_of_M_l421_42198


namespace koschei_stopped_month_l421_42166

/-- The number of children Baba Yaga helps per month -/
def baba_yaga_rate : ℕ := 77

/-- The number of children Koschei helps per month -/
def koschei_rate : ℕ := 12

/-- The number of months between the start and end of the competition -/
def competition_duration : ℕ := 120

/-- The ratio of Baba Yaga's total good deeds to Koschei's at the end -/
def final_ratio : ℕ := 5

/-- Theorem stating when Koschei stopped doing good deeds -/
theorem koschei_stopped_month :
  ∃ (m : ℕ), m * koschei_rate * final_ratio = competition_duration * baba_yaga_rate ∧ m = 154 := by
  sorry

end koschei_stopped_month_l421_42166


namespace arithmetic_sequence_third_term_l421_42165

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumOfFirstFiveIs20 (a : ℕ → ℚ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
    (a : ℕ → ℚ)
    (h_arithmetic : IsArithmeticSequence a)
    (h_sum : SumOfFirstFiveIs20 a) :
    a 3 = 4 := by
  sorry

end arithmetic_sequence_third_term_l421_42165


namespace price_decrease_fifty_percent_l421_42123

/-- Calculates the percentage decrease in price given the original and new prices. -/
def percentage_decrease (original_price new_price : ℚ) : ℚ :=
  (original_price - new_price) / original_price * 100

/-- Theorem stating that the percentage decrease is 50% given the specific prices. -/
theorem price_decrease_fifty_percent (original_price new_price : ℚ) 
  (h1 : original_price = 1240)
  (h2 : new_price = 620) : 
  percentage_decrease original_price new_price = 50 := by
  sorry

#eval percentage_decrease 1240 620

end price_decrease_fifty_percent_l421_42123


namespace max_sum_of_seventh_powers_l421_42157

theorem max_sum_of_seventh_powers (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ (M : ℝ), M = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ 
                        a'^7 + b'^7 + c'^7 + d'^7 = M :=
by
  sorry

end max_sum_of_seventh_powers_l421_42157


namespace initial_kids_on_field_l421_42105

theorem initial_kids_on_field (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 22 → total = 36 → total = initial + joined → initial = 14 := by
sorry

end initial_kids_on_field_l421_42105


namespace farmer_wheat_harvest_l421_42134

theorem farmer_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end farmer_wheat_harvest_l421_42134


namespace fraction_product_l421_42199

theorem fraction_product : (4/5 : ℚ) * (5/6 : ℚ) * (6/7 : ℚ) * (7/8 : ℚ) * (8/9 : ℚ) = 4/9 := by
  sorry

end fraction_product_l421_42199


namespace vector_sum_magnitude_l421_42196

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

-- Define the conditions
def perpendicular (v w : Fin 2 → ℝ) : Prop := 
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop := 
  ∃ (k : ℝ), v = fun i ↦ k * (w i)

-- Theorem statement
theorem vector_sum_magnitude (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end vector_sum_magnitude_l421_42196


namespace pizza_sales_total_l421_42109

theorem pizza_sales_total (pepperoni bacon cheese : ℕ) 
  (h1 : pepperoni = 2) 
  (h2 : bacon = 6) 
  (h3 : cheese = 6) : 
  pepperoni + bacon + cheese = 14 := by
  sorry

end pizza_sales_total_l421_42109


namespace sum_of_three_squares_l421_42163

theorem sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hp_neq_3 : p ≠ 3) :
  ∃ a b c : ℕ, 4 * p^2 + 1 = a^2 + b^2 + c^2 := by
  sorry

end sum_of_three_squares_l421_42163


namespace point_order_on_parabola_l421_42141

-- Define the parabola function
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Define the theorem
theorem point_order_on_parabola (a b c : ℝ) :
  parabola a = -2 →
  parabola b = -2 →
  parabola c = -7 →
  a < b →
  c > 2 →
  a < b ∧ b < c := by
  sorry

end point_order_on_parabola_l421_42141


namespace james_black_spools_l421_42183

/-- Represents the number of spools of yarn needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- Represents the number of spools of red yarn James has -/
def red_spools : ℕ := 12

/-- Represents the number of spools of blue yarn James has -/
def blue_spools : ℕ := 6

/-- Represents the number of berets James can make -/
def total_berets : ℕ := 11

/-- Calculates the number of black yarn spools James has -/
def black_spools : ℕ := 
  spools_per_beret * total_berets - (red_spools + blue_spools)

theorem james_black_spools : black_spools = 15 := by
  sorry

end james_black_spools_l421_42183


namespace initial_apples_l421_42153

theorem initial_apples (initial : ℝ) (received : ℝ) (total : ℝ) : 
  received = 7.0 → total = 27 → initial + received = total → initial = 20.0 := by
sorry

end initial_apples_l421_42153


namespace independence_day_bananas_l421_42170

theorem independence_day_bananas (total_children : ℕ) 
  (present_children : ℕ) (absent_children : ℕ) (bananas : ℕ) : 
  total_children = 260 →
  bananas = 4 * present_children →
  bananas = 2 * total_children →
  present_children + absent_children = total_children →
  absent_children = 130 := by
sorry

end independence_day_bananas_l421_42170


namespace line_relations_l421_42168

def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

theorem line_relations (m : ℝ) :
  (∀ x y, l1 m x y → l2 m x y → (m - 2 + 3 * m = 0) ↔ m = 1/2) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ m ≠ 3 ∧ m ≠ -3) ↔ m = -1) ∧
  (∀ x y, l1 m x y → l2 m x y → ((m - 2) / 1 = 3 / m ∧ 3 / m = 2 * m / 6) ↔ m = 3) ∧
  (∀ x y, l1 m x y → l2 m x y → (m ≠ 3 ∧ m ≠ -1) ↔ (m ≠ 3 ∧ m ≠ -1)) :=
by sorry

end line_relations_l421_42168


namespace chocolate_bar_breaks_chocolate_bar_40_pieces_l421_42113

/-- The minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of breaks for a chocolate bar with n pieces is n - 1 -/
theorem chocolate_bar_breaks (n : ℕ) (h : n > 0) : 
  min_breaks n = n - 1 := by
  sorry

/-- Corollary for the specific case of 40 pieces -/
theorem chocolate_bar_40_pieces : 
  min_breaks 40 = 39 := by
  sorry

end chocolate_bar_breaks_chocolate_bar_40_pieces_l421_42113


namespace cheap_module_count_l421_42171

/-- Represents the stock of modules -/
structure ModuleStock where
  expensive_count : ℕ
  cheap_count : ℕ

/-- The cost of an expensive module -/
def expensive_cost : ℚ := 10

/-- The cost of a cheap module -/
def cheap_cost : ℚ := 3.5

/-- The total value of the stock -/
def total_value (stock : ModuleStock) : ℚ :=
  (stock.expensive_count : ℚ) * expensive_cost + (stock.cheap_count : ℚ) * cheap_cost

/-- The total count of modules in the stock -/
def total_count (stock : ModuleStock) : ℕ :=
  stock.expensive_count + stock.cheap_count

theorem cheap_module_count (stock : ModuleStock) :
  total_value stock = 45 ∧ total_count stock = 11 → stock.cheap_count = 10 :=
by sorry

end cheap_module_count_l421_42171


namespace quadratic_factorization_l421_42147

theorem quadratic_factorization (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end quadratic_factorization_l421_42147


namespace dining_bill_calculation_l421_42106

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 158.40 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧
  total = food_price * (1 + tax_rate) * (1 + tip_rate) →
  food_price = 120 := by
sorry

end dining_bill_calculation_l421_42106


namespace other_candidate_votes_l421_42188

theorem other_candidate_votes
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (winning_candidate_percentage : ℚ)
  (h_total : total_votes = 7500)
  (h_invalid : invalid_percentage = 1/5)
  (h_winning : winning_candidate_percentage = 11/20) :
  ⌊(1 - invalid_percentage) * (1 - winning_candidate_percentage) * total_votes⌋ = 2700 :=
sorry

end other_candidate_votes_l421_42188


namespace correct_matching_probability_l421_42167

theorem correct_matching_probability (n : ℕ) (h : n = 4) : 
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 24 :=
by sorry

end correct_matching_probability_l421_42167


namespace whitney_cant_afford_all_items_l421_42104

def poster_cost : ℕ := 7
def notebook_cost : ℕ := 5
def bookmark_cost : ℕ := 3
def pencil_cost : ℕ := 1

def poster_quantity : ℕ := 3
def notebook_quantity : ℕ := 4
def bookmark_quantity : ℕ := 5
def pencil_quantity : ℕ := 2

def available_funds : ℕ := 2 * 20

theorem whitney_cant_afford_all_items :
  poster_cost * poster_quantity +
  notebook_cost * notebook_quantity +
  bookmark_cost * bookmark_quantity +
  pencil_cost * pencil_quantity > available_funds :=
by sorry

end whitney_cant_afford_all_items_l421_42104


namespace eggs_for_bread_l421_42184

/-- The number of dozens of eggs needed given the total weight required, weight per egg, and eggs per dozen -/
def eggs_needed (total_weight : ℚ) (weight_per_egg : ℚ) (eggs_per_dozen : ℕ) : ℚ :=
  (total_weight / weight_per_egg) / eggs_per_dozen

/-- Theorem stating that 8 dozens of eggs are needed for the given conditions -/
theorem eggs_for_bread : eggs_needed 6 (1/16) 12 = 8 := by
  sorry

end eggs_for_bread_l421_42184


namespace additional_men_needed_l421_42128

/-- Proves that given a work that can be finished by 12 men in 11 days,
    if the work is completed in 8 days (3 days earlier),
    then the number of additional men needed is 5. -/
theorem additional_men_needed
  (original_days : ℕ)
  (original_men : ℕ)
  (actual_days : ℕ)
  (h1 : original_days = 11)
  (h2 : original_men = 12)
  (h3 : actual_days = original_days - 3)
  : ∃ (additional_men : ℕ), 
    (original_men * original_days = (original_men + additional_men) * actual_days) ∧
    additional_men = 5 := by
  sorry

end additional_men_needed_l421_42128


namespace t_formula_correct_t_2022_last_digit_l421_42129

/-- The number of unordered triples of non-empty and pairwise disjoint subsets of a set with n elements -/
def t (n : ℕ+) : ℚ :=
  (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6

/-- The closed form formula for t_n is correct -/
theorem t_formula_correct (n : ℕ+) :
  t n = (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6 := by sorry

/-- The last digit of t_2022 is 1 -/
theorem t_2022_last_digit :
  t 2022 % 1 = 1 / 10 := by sorry

end t_formula_correct_t_2022_last_digit_l421_42129


namespace pictures_in_first_album_l421_42177

theorem pictures_in_first_album 
  (total_pictures : ℕ) 
  (num_albums : ℕ) 
  (pics_per_album : ℕ) 
  (h1 : total_pictures = 65)
  (h2 : num_albums = 6)
  (h3 : pics_per_album = 8) :
  total_pictures - (num_albums * pics_per_album) = 17 :=
by sorry

end pictures_in_first_album_l421_42177


namespace sum_of_reciprocals_l421_42178

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end sum_of_reciprocals_l421_42178


namespace solution_set_l421_42148

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 4 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 4 * Real.sqrt (x - 9)) - 3 →
  x ≥ 12 :=
by sorry

end solution_set_l421_42148


namespace monotone_increasing_condition_l421_42135

open Real

theorem monotone_increasing_condition (b : ℝ) :
  (∃ (a c : ℝ), 1/2 ≤ a ∧ c ≤ 2 ∧ a < c ∧
    ∀ x y, a ≤ x ∧ x < y ∧ y ≤ c →
      exp x * (x^2 - b*x) < exp y * (y^2 - b*y)) →
  b < 8/3 := by
sorry

end monotone_increasing_condition_l421_42135


namespace trees_died_in_typhoon_l421_42158

theorem trees_died_in_typhoon (initial_trees : ℕ) (remaining_trees : ℕ) : 
  initial_trees = 20 → remaining_trees = 4 → initial_trees - remaining_trees = 16 := by
  sorry

end trees_died_in_typhoon_l421_42158


namespace max_x_value_l421_42156

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 9) 
  (prod_sum_eq : x*y + x*z + y*z = 20) : 
  x ≤ (18 + Real.sqrt 312) / 6 := by
  sorry

end max_x_value_l421_42156


namespace sphere_volume_of_inscribed_parallelepiped_l421_42132

/-- The volume of a sphere circumscribing a rectangular parallelepiped with edge lengths 1, √2, and 3 -/
theorem sphere_volume_of_inscribed_parallelepiped : ∃ (V : ℝ),
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 2
  let c : ℝ := 3
  let r : ℝ := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  V = (4 / 3) * π * r^3 ∧ V = 4 * Real.sqrt 3 * π :=
by sorry

end sphere_volume_of_inscribed_parallelepiped_l421_42132


namespace root_product_squared_plus_one_l421_42193

theorem root_product_squared_plus_one (a b c : ℂ) : 
  (a^3 + 20*a^2 + a + 5 = 0) →
  (b^3 + 20*b^2 + b + 5 = 0) →
  (c^3 + 20*c^2 + c + 5 = 0) →
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) = 229 := by
  sorry

end root_product_squared_plus_one_l421_42193


namespace min_distance_points_l421_42131

theorem min_distance_points (a b : ℝ) : 
  a = 2 → 
  (∃ (min_val : ℝ), min_val = 7 ∧ 
    ∀ (x : ℝ), |x - a| + |x - b| ≥ min_val) → 
  (b = -5 ∨ b = 9) :=
by sorry

end min_distance_points_l421_42131


namespace quadratic_roots_l421_42122

-- Define the quadratic equations
def eq1 (x : ℝ) : Prop := x^2 - x + 1 = 0
def eq2 (x : ℝ) : Prop := x * (x - 1) = 0
def eq3 (x : ℝ) : Prop := x^2 + 12*x = 0
def eq4 (x : ℝ) : Prop := x^2 + x = 1

-- Theorem stating that eq1 has no real roots while others have
theorem quadratic_roots :
  (¬ ∃ x : ℝ, eq1 x) ∧
  (∃ x : ℝ, eq2 x) ∧
  (∃ x : ℝ, eq3 x) ∧
  (∃ x : ℝ, eq4 x) := by
  sorry

end quadratic_roots_l421_42122


namespace prob_good_or_excellent_grade_l421_42150

/-- Represents the types of students in the group -/
inductive StudentType
| Excellent
| Good
| Poor

/-- Represents the possible grades a student can receive -/
inductive Grade
| Excellent
| Good
| Satisfactory
| Unsatisfactory

/-- The total number of students -/
def totalStudents : ℕ := 21

/-- The number of excellent students -/
def excellentCount : ℕ := 5

/-- The number of good students -/
def goodCount : ℕ := 10

/-- The number of poorly performing students -/
def poorCount : ℕ := 6

/-- The probability of selecting an excellent student -/
def probExcellent : ℚ := excellentCount / totalStudents

/-- The probability of selecting a good student -/
def probGood : ℚ := goodCount / totalStudents

/-- The probability of selecting a poor student -/
def probPoor : ℚ := poorCount / totalStudents

/-- The probability of an excellent student receiving an excellent grade -/
def probExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | _ => 0

/-- The probability of a good student receiving a good or excellent grade -/
def probGoodOrExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | StudentType.Good => 1
  | StudentType.Poor => 1/3

/-- The probability of a randomly selected student receiving a good or excellent grade -/
theorem prob_good_or_excellent_grade :
  probExcellent * probExcellentGrade StudentType.Excellent +
  probGood * probGoodOrExcellentGrade StudentType.Good +
  probPoor * probGoodOrExcellentGrade StudentType.Poor = 17/21 := by
  sorry


end prob_good_or_excellent_grade_l421_42150


namespace tan_product_eighths_pi_l421_42118

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end tan_product_eighths_pi_l421_42118


namespace quadratic_point_ordering_l421_42180

/-- Given a quadratic function f(x) = x² + 2x + c, prove that for points
    A(-3, y₁), B(-2, y₂), and C(2, y₃) on its graph, y₃ > y₁ > y₂ holds. -/
theorem quadratic_point_ordering (c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x + c
  let y₁ : ℝ := f (-3)
  let y₂ : ℝ := f (-2)
  let y₃ : ℝ := f 2
  y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end quadratic_point_ordering_l421_42180


namespace x_value_when_y_is_two_l421_42145

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end x_value_when_y_is_two_l421_42145


namespace line_through_points_l421_42140

/-- Given two intersecting lines and their intersection point, prove the equation of the line passing through specific points. -/
theorem line_through_points (A₁ B₁ A₂ B₂ : ℝ) :
  (2 * A₁ + 3 * B₁ = 1) →  -- l₁ passes through P(2, 3)
  (2 * A₂ + 3 * B₂ = 1) →  -- l₂ passes through P(2, 3)
  (∀ x y : ℝ, A₁ * x + B₁ * y = 1 → 2 * x + 3 * y = 1) →  -- l₁ equation
  (∀ x y : ℝ, A₂ * x + B₂ * y = 1 → 2 * x + 3 * y = 1) →  -- l₂ equation
  ∀ x y : ℝ, (y - B₁) * (A₂ - A₁) = (x - A₁) * (B₂ - B₁) → 2 * x + 3 * y = 1 :=
by sorry


end line_through_points_l421_42140


namespace knights_on_red_chairs_l421_42190

/-- Represents the type of chair occupant -/
inductive Occupant
| Knight
| Liar

/-- Represents the color of a chair -/
inductive ChairColor
| Blue
| Red

/-- Represents the state of the room -/
structure RoomState where
  totalChairs : ℕ
  knights : ℕ
  liars : ℕ
  knightsOnRed : ℕ
  liarsOnBlue : ℕ

/-- The initial state of the room -/
def initialState : RoomState :=
  { totalChairs := 20
  , knights := 20 - (20 : ℕ) / 2  -- Arbitrary split between knights and liars
  , liars := (20 : ℕ) / 2
  , knightsOnRed := 0
  , liarsOnBlue := 0 }

/-- The state of the room after rearrangement -/
def finalState (initial : RoomState) : RoomState :=
  { totalChairs := initial.totalChairs
  , knights := initial.knights
  , liars := initial.liars
  , knightsOnRed := (initial.totalChairs : ℕ) / 2 - initial.liars
  , liarsOnBlue := (initial.totalChairs : ℕ) / 2 - (initial.knights - ((initial.totalChairs : ℕ) / 2 - initial.liars)) }

theorem knights_on_red_chairs (initial : RoomState) :
  (finalState initial).knightsOnRed = 5 := by
  sorry

end knights_on_red_chairs_l421_42190
