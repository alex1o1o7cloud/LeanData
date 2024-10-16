import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l550_55090

theorem equation_solutions (n : ℕ) : 
  (∃! (solutions : Finset (ℕ × ℕ × ℕ)), 
    solutions.card = 10 ∧ 
    ∀ (x y z : ℕ), (x, y, z) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ z > 0 ∧ 4*x + 6*y + 2*z = n)) ↔ 
  (n = 32 ∨ n = 33) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l550_55090


namespace NUMINAMATH_CALUDE_infinite_sum_equals_two_l550_55055

theorem infinite_sum_equals_two :
  (∑' n : ℕ, (4 * n - 2) / (3 : ℝ)^n) = 2 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_two_l550_55055


namespace NUMINAMATH_CALUDE_even_odd_property_l550_55088

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem even_odd_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_odd : is_odd_function (fun x ↦ f (x - 1)))
  (h_f2 : f 2 = 3) :
  f 5 + f 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_even_odd_property_l550_55088


namespace NUMINAMATH_CALUDE_ice_cream_probability_l550_55058

def probability_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem ice_cream_probability : 
  probability_exactly_k_successes 7 3 (3/4) = 945/16384 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_probability_l550_55058


namespace NUMINAMATH_CALUDE_baseball_team_ratio_l550_55057

def baseball_ratio (games_played : ℕ) (games_won : ℕ) : ℚ := 
  games_played / (games_played - games_won)

theorem baseball_team_ratio : 
  let games_played := 10
  let games_won := 5
  baseball_ratio games_played games_won = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_ratio_l550_55057


namespace NUMINAMATH_CALUDE_peter_double_harriet_age_l550_55005

def mother_age : ℕ := 60
def harriet_age : ℕ := 13

def peter_age : ℕ := mother_age / 2

def years_until_double (x : ℕ) : Prop :=
  peter_age + x = 2 * (harriet_age + x)

theorem peter_double_harriet_age :
  ∃ x : ℕ, years_until_double x ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_peter_double_harriet_age_l550_55005


namespace NUMINAMATH_CALUDE_alpha_plus_three_beta_range_l550_55018

theorem alpha_plus_three_beta_range (α β : ℝ) 
  (h1 : -1 ≤ α + β ∧ α + β ≤ 1) 
  (h2 : 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) : 
  1 ≤ α + 3*β ∧ α + 3*β ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_three_beta_range_l550_55018


namespace NUMINAMATH_CALUDE_store_profit_analysis_l550_55050

/-- Represents the relationship between sales volume and selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 120

/-- Represents the profit function -/
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - 60)

/-- The cost price per item -/
def cost_price : ℝ := 60

/-- The maximum allowed profit percentage -/
def max_profit_percentage : ℝ := 0.45

theorem store_profit_analysis 
  (h1 : ∀ x, x ≥ cost_price)  -- Selling price not lower than cost price
  (h2 : ∀ x, profit x ≤ max_profit_percentage * cost_price * (x - cost_price))  -- Profit not exceeding 45%
  : 
  (∃ max_profit_price : ℝ, 
    max_profit_price = 87 ∧ 
    profit max_profit_price = 891 ∧ 
    ∀ x, profit x ≤ profit max_profit_price) ∧ 
  (∀ x, profit x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 110) := by
  sorry


end NUMINAMATH_CALUDE_store_profit_analysis_l550_55050


namespace NUMINAMATH_CALUDE_cubic_root_relation_l550_55019

theorem cubic_root_relation (m n p x₃ : ℝ) : 
  (∃ (z : ℂ), z^3 + (m/3)*z^2 + (n/3)*z + (p/3) = 0 ∧ 
               (z = 4 + 3*Complex.I ∨ z = 4 - 3*Complex.I ∨ z = x₃)) →
  x₃ > 0 →
  p = -75 * x₃ := by
sorry

end NUMINAMATH_CALUDE_cubic_root_relation_l550_55019


namespace NUMINAMATH_CALUDE_square_area_from_equal_rectangles_l550_55049

/-- Given a square cut into five rectangles of equal area, where one rectangle has a width of 5,
    the area of the square is 400. -/
theorem square_area_from_equal_rectangles (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    a = 5 + 2*x ∧
    a^2 / 5 = 5 * (a^2 / 25) ∧
    a^2 / 5 = x * y ∧
    a^2 / 5 = 3 * y * 5) →
  a^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_equal_rectangles_l550_55049


namespace NUMINAMATH_CALUDE_barbara_typing_time_l550_55048

/-- Calculates the time needed to type a document given the original typing speed,
    speed reduction, and document length. -/
def typing_time (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_reduction)

/-- Theorem stating that with given conditions, it takes 20 minutes to type the document. -/
theorem barbara_typing_time :
  typing_time 212 40 3440 = 20 := by
  sorry

end NUMINAMATH_CALUDE_barbara_typing_time_l550_55048


namespace NUMINAMATH_CALUDE_car_B_speed_l550_55079

/-- Proves that the speed of car B is 90 km/h given the problem conditions -/
theorem car_B_speed (distance : ℝ) (time : ℝ) (speed_ratio : ℝ × ℝ) :
  distance = 88 →
  time = 32 / 60 →
  speed_ratio = (5, 6) →
  ∃ (speed_A speed_B : ℝ),
    speed_A / speed_B = speed_ratio.1 / speed_ratio.2 ∧
    distance = (speed_A + speed_B) * time ∧
    speed_B = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_B_speed_l550_55079


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l550_55093

/-- Calculates the total interest earned from an investment split between two interest rates -/
def total_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := total_investment - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest is $660 given the specified conditions -/
theorem investment_interest_calculation :
  total_interest 18000 0.03 0.05 6000 = 660 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l550_55093


namespace NUMINAMATH_CALUDE_radhika_video_games_l550_55071

/-- The number of video games Radhika received on Christmas. -/
def christmas_games : ℕ := 12

/-- The number of video games Radhika received on her birthday. -/
def birthday_games : ℕ := 8

/-- The number of video games Radhika already owned. -/
def owned_games : ℕ := (christmas_games + birthday_games) / 2

/-- The total number of video games Radhika owns now. -/
def total_games : ℕ := christmas_games + birthday_games + owned_games

theorem radhika_video_games :
  total_games = 30 :=
by sorry

end NUMINAMATH_CALUDE_radhika_video_games_l550_55071


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l550_55087

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l550_55087


namespace NUMINAMATH_CALUDE_water_depth_of_specific_tower_l550_55036

/-- Represents a conical tower -/
structure ConicalTower where
  height : ℝ
  volumeAboveWater : ℝ

/-- Calculates the depth of water at the base of a conical tower -/
def waterDepth (tower : ConicalTower) : ℝ :=
  tower.height * (1 - (tower.volumeAboveWater)^(1/3))

/-- The theorem stating the depth of water for a specific conical tower -/
theorem water_depth_of_specific_tower :
  let tower : ConicalTower := ⟨10000, 1/4⟩
  waterDepth tower = 905 := by sorry

end NUMINAMATH_CALUDE_water_depth_of_specific_tower_l550_55036


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l550_55051

theorem smallest_number_divisible (n : ℕ) : n ≥ 136 →
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 16 = 4 * k ∧ m - 16 = 6 * k ∧ m - 16 = 8 * k ∧ m - 16 = 10 * k)) →
  (∃ k : ℕ, 136 - 16 = 4 * k ∧ 136 - 16 = 6 * k ∧ 136 - 16 = 8 * k ∧ 136 - 16 = 10 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l550_55051


namespace NUMINAMATH_CALUDE_carmen_total_sales_l550_55061

/-- Represents the sales to a house -/
structure HouseSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ
  samoasPrice : ℚ
  thinMintsPrice : ℚ
  fudgeDelightsPrice : ℚ
  sugarCookiesPrice : ℚ

/-- Calculates the total sales for a house -/
def houseSalesTotal (sales : HouseSales) : ℚ :=
  sales.samoas * sales.samoasPrice +
  sales.thinMints * sales.thinMintsPrice +
  sales.fudgeDelights * sales.fudgeDelightsPrice +
  sales.sugarCookies * sales.sugarCookiesPrice

/-- Represents Carmen's total sales -/
def carmenSales : List HouseSales :=
  [
    { samoas := 3, thinMints := 0, fudgeDelights := 0, sugarCookies := 0,
      samoasPrice := 4, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 2, fudgeDelights := 1, sugarCookies := 0,
      samoasPrice := 0, thinMintsPrice := 7/2, fudgeDelightsPrice := 5, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 0, fudgeDelights := 0, sugarCookies := 9,
      samoasPrice := 0, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 2 }
  ]

theorem carmen_total_sales :
  (carmenSales.map houseSalesTotal).sum = 42 := by
  sorry

end NUMINAMATH_CALUDE_carmen_total_sales_l550_55061


namespace NUMINAMATH_CALUDE_zero_function_equals_derivative_l550_55092

theorem zero_function_equals_derivative : ∃ f : ℝ → ℝ, ∀ x, f x = 0 ∧ (deriv f) x = f x := by
  sorry

end NUMINAMATH_CALUDE_zero_function_equals_derivative_l550_55092


namespace NUMINAMATH_CALUDE_negative_sqrt_two_squared_l550_55091

theorem negative_sqrt_two_squared : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_squared_l550_55091


namespace NUMINAMATH_CALUDE_bucket_capacity_change_l550_55011

theorem bucket_capacity_change (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4/5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_change_l550_55011


namespace NUMINAMATH_CALUDE_ellipse_sum_parameters_l550_55084

/-- An ellipse with foci F₁ and F₂, and constant sum of distances 2a -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  a : ℝ

/-- The standard form equation of an ellipse -/
structure EllipseEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, this function returns its standard form equation -/
def ellipse_to_equation (e : Ellipse) : EllipseEquation :=
  sorry

theorem ellipse_sum_parameters (e : Ellipse) (eq : EllipseEquation) :
  e.F₁ = (0, 0) →
  e.F₂ = (6, 0) →
  e.a = 5 →
  eq = ellipse_to_equation e →
  eq.h + eq.k + eq.a + eq.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_parameters_l550_55084


namespace NUMINAMATH_CALUDE_limit_of_polynomial_at_two_l550_55016

theorem limit_of_polynomial_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |4*x^2 - 6*x + 3 - 7| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_polynomial_at_two_l550_55016


namespace NUMINAMATH_CALUDE_odd_ceiling_factorial_fraction_l550_55056

theorem odd_ceiling_factorial_fraction (n : ℕ) (h1 : n > 6) (h2 : Nat.Prime (n + 1)) :
  Odd (⌈(Nat.factorial (n - 1) : ℚ) / (n * (n + 1))⌉) := by
  sorry

end NUMINAMATH_CALUDE_odd_ceiling_factorial_fraction_l550_55056


namespace NUMINAMATH_CALUDE_no_perfect_square_212_b_l550_55024

theorem no_perfect_square_212_b : ¬ ∃ (b : ℕ), b > 2 ∧ ∃ (n : ℕ), 2 * b^2 + b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_212_b_l550_55024


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l550_55028

/-- Given a sphere with volume 36π cubic inches, its surface area is 36π square inches. -/
theorem sphere_surface_area_from_volume : 
  ∀ (r : ℝ), (4 / 3 : ℝ) * π * r^3 = 36 * π → 4 * π * r^2 = 36 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l550_55028


namespace NUMINAMATH_CALUDE_sheridan_fish_count_l550_55077

/-- Calculates the remaining number of fish after giving some away -/
def remaining_fish (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 25.0 fish after giving away 22.0 from her initial 47.0 fish -/
theorem sheridan_fish_count : remaining_fish 47.0 22.0 = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_count_l550_55077


namespace NUMINAMATH_CALUDE_expression_equals_two_l550_55070

theorem expression_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π/2 - α) + Real.cos (π/2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l550_55070


namespace NUMINAMATH_CALUDE_expression_simplification_l550_55030

theorem expression_simplification (y : ℝ) :
  3 * y - 7 * y^2 + 15 - (2 + 6 * y - 7 * y^2) = -3 * y + 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l550_55030


namespace NUMINAMATH_CALUDE_potato_cost_is_correct_l550_55075

/-- The cost of one bag of potatoes from the farmer in rubles -/
def potato_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The additional profit Boris made compared to Andrey in rubles -/
def additional_profit : ℝ := 1200

theorem potato_cost_is_correct : 
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * boris_first_sale +
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100) * boris_second_sale -
  potato_cost * bags_bought * (1 + andrey_increase / 100) * bags_bought = additional_profit := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_is_correct_l550_55075


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l550_55086

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 120) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l550_55086


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l550_55032

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22222222222223 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l550_55032


namespace NUMINAMATH_CALUDE_overlap_area_is_75_l550_55059

-- Define a 30-60-90 triangle
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = hypotenuse / 2
  longLeg_eq : longLeg = shortLeg * Real.sqrt 3

-- Define the overlapping configuration
def overlapArea (t : Triangle30_60_90) : ℝ :=
  t.longLeg * t.longLeg

-- Theorem statement
theorem overlap_area_is_75 (t : Triangle30_60_90) :
  overlapArea t = 75 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_75_l550_55059


namespace NUMINAMATH_CALUDE_jury_deliberation_hours_jury_deliberation_hours_proof_l550_55022

theorem jury_deliberation_hours (jury_selection_days : ℕ) 
                                (trial_multiplier : ℕ) 
                                (full_day_equivalent : ℕ) 
                                (total_days : ℕ) 
                                (hours_per_day : ℕ) : Prop :=
  jury_selection_days = 2 →
  trial_multiplier = 4 →
  full_day_equivalent = 6 →
  total_days = 19 →
  hours_per_day = 24 →
  let trial_days := trial_multiplier * jury_selection_days
  let deliberation_days := total_days - (jury_selection_days + trial_days)
  let total_deliberation_hours := full_day_equivalent * hours_per_day
  total_deliberation_hours / deliberation_days = 16

-- The proof would go here, but we're skipping it as requested
theorem jury_deliberation_hours_proof : 
  jury_deliberation_hours 2 4 6 19 24 := by sorry

end NUMINAMATH_CALUDE_jury_deliberation_hours_jury_deliberation_hours_proof_l550_55022


namespace NUMINAMATH_CALUDE_f_properties_when_a_is_1_f_minimum_on_interval_l550_55081

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1
theorem f_properties_when_a_is_1 :
  let a := 1
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f a x > f a y ∧
  ∀ z : ℝ, f a z ≥ 1 ∧ ∃ w : ℝ, f a w = 1 := by sorry

-- Part 2
theorem f_minimum_on_interval (a : ℝ) (h : a ≥ -1) :
  let min_value := if a < 1 then -a^2 + 2 else 3 - 2*a
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ min_value ∧
  ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = min_value := by sorry

end NUMINAMATH_CALUDE_f_properties_when_a_is_1_f_minimum_on_interval_l550_55081


namespace NUMINAMATH_CALUDE_root_expression_value_l550_55078

theorem root_expression_value (p m n : ℝ) : 
  (m^2 + (p - 2) * m + 1 = 0) → 
  (n^2 + (p - 2) * n + 1 = 0) → 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_root_expression_value_l550_55078


namespace NUMINAMATH_CALUDE_triangle_side_length_l550_55027

theorem triangle_side_length (A B C : ℝ) (AC AB BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  AB = 2 →
  (Real.sqrt 3 * Real.sin angle_A + Real.cos angle_A) / (Real.sqrt 3 * Real.cos angle_A - Real.sin angle_A) = Real.tan (5 * Real.pi / 12) →
  BC = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l550_55027


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l550_55094

def ice_cream_problem (single_cone waffle_bowl banana_split double_cone : ℕ) : Prop :=
  single_cone = 1 ∧ 
  banana_split = 3 * single_cone ∧ 
  waffle_bowl = banana_split + 1 ∧
  double_cone = 2 ∧
  single_cone + double_cone + banana_split + waffle_bowl = 10

theorem ice_cream_scoops : 
  ∃ (single_cone waffle_bowl banana_split double_cone : ℕ),
    ice_cream_problem single_cone waffle_bowl banana_split double_cone :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l550_55094


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l550_55010

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l550_55010


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l550_55065

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c > 0 ↔ x < -1 ∨ x > 2) → 
  b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l550_55065


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l550_55053

theorem binomial_coefficient_equality : Nat.choose 10 8 = Nat.choose 10 2 ∧ Nat.choose 10 8 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l550_55053


namespace NUMINAMATH_CALUDE_problem1_l550_55021

theorem problem1 : Real.sqrt 4 - (1/2)⁻¹ + (2 - 1/7)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l550_55021


namespace NUMINAMATH_CALUDE_prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l550_55089

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : ℕ :=
  bag.red + bag.yellow + bag.green

/-- Calculates the probability of drawing a green ball -/
def probGreen (bag : BagContents) : ℚ :=
  bag.green / (totalBalls bag)

/-- The initial bag contents -/
def initialBag : BagContents :=
  { red := 6, yellow := 9, green := 3 }

/-- Theorem stating the probability of drawing a green ball is 1/6 -/
theorem prob_green_is_one_sixth :
  probGreen initialBag = 1/6 := by sorry

/-- Adds green balls to the bag -/
def addGreenBalls (bag : BagContents) (n : ℕ) : BagContents :=
  { bag with green := bag.green + n }

/-- Theorem stating that adding 2 green balls makes the probability 1/4 -/
theorem adding_two_green_balls_makes_prob_one_fourth :
  probGreen (addGreenBalls initialBag 2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l550_55089


namespace NUMINAMATH_CALUDE_equal_arcs_equal_chords_l550_55080

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc where
  circle : Circle
  start_angle : ℝ
  end_angle : ℝ

/-- Represents a chord of a circle -/
structure Chord where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Function to calculate the length of an arc -/
def arcLength (arc : Arc) : ℝ := sorry

/-- Function to calculate the length of a chord -/
def chordLength (chord : Chord) : ℝ := sorry

/-- Theorem: In a circle, equal arcs correspond to equal chords -/
theorem equal_arcs_equal_chords (c : Circle) (arc1 arc2 : Arc) (chord1 chord2 : Chord) :
  arc1.circle = c → arc2.circle = c →
  chord1.circle = c → chord2.circle = c →
  arcLength arc1 = arcLength arc2 →
  chord1.endpoint1 = (c.center.1 + c.radius * Real.cos arc1.start_angle,
                      c.center.2 + c.radius * Real.sin arc1.start_angle) →
  chord1.endpoint2 = (c.center.1 + c.radius * Real.cos arc1.end_angle,
                      c.center.2 + c.radius * Real.sin arc1.end_angle) →
  chord2.endpoint1 = (c.center.1 + c.radius * Real.cos arc2.start_angle,
                      c.center.2 + c.radius * Real.sin arc2.start_angle) →
  chord2.endpoint2 = (c.center.1 + c.radius * Real.cos arc2.end_angle,
                      c.center.2 + c.radius * Real.sin arc2.end_angle) →
  chordLength chord1 = chordLength chord2 := by sorry

end NUMINAMATH_CALUDE_equal_arcs_equal_chords_l550_55080


namespace NUMINAMATH_CALUDE_ab_equals_six_l550_55017

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l550_55017


namespace NUMINAMATH_CALUDE_book_price_proof_l550_55097

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 75)
  (h2 : profit_percentage = 25) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l550_55097


namespace NUMINAMATH_CALUDE_championship_and_expectation_l550_55020

-- Define the probabilities of class A winning each event
def p_basketball : ℝ := 0.4
def p_soccer : ℝ := 0.8
def p_badminton : ℝ := 0.6

-- Define the points awarded for winning and losing
def win_points : ℕ := 8
def lose_points : ℕ := 0

-- Define the probability of class A winning the championship
def p_championship : ℝ := 
  p_basketball * p_soccer * p_badminton +
  (1 - p_basketball) * p_soccer * p_badminton +
  p_basketball * (1 - p_soccer) * p_badminton +
  p_basketball * p_soccer * (1 - p_badminton)

-- Define the distribution of class B's total score
def p_score (x : ℕ) : ℝ :=
  if x = 0 then (1 - p_basketball) * (1 - p_soccer) * (1 - p_badminton)
  else if x = win_points then 
    p_basketball * (1 - p_soccer) * (1 - p_badminton) +
    (1 - p_basketball) * p_soccer * (1 - p_badminton) +
    (1 - p_basketball) * (1 - p_soccer) * p_badminton
  else if x = 2 * win_points then
    p_basketball * p_soccer * (1 - p_badminton) +
    p_basketball * (1 - p_soccer) * p_badminton +
    (1 - p_basketball) * p_soccer * p_badminton
  else if x = 3 * win_points then
    p_basketball * p_soccer * p_badminton
  else 0

-- Define the expectation of class B's total score
def expectation_B : ℝ :=
  0 * p_score 0 +
  win_points * p_score win_points +
  (2 * win_points) * p_score (2 * win_points) +
  (3 * win_points) * p_score (3 * win_points)

theorem championship_and_expectation :
  p_championship = 0.656 ∧ expectation_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_championship_and_expectation_l550_55020


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l550_55034

theorem triangle_angle_measure (P Q R : ℝ) (h1 : R = 3 * Q) (h2 : Q = 30) :
  P + Q + R = 180 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l550_55034


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l550_55007

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 = 1 →
  a 8 * a 9 = 16 →
  a 6 * a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l550_55007


namespace NUMINAMATH_CALUDE_shortest_paths_correct_l550_55009

def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem shortest_paths_correct (m n : ℕ) : 
  shortest_paths m n = Nat.choose (m + n) m :=
by sorry

end NUMINAMATH_CALUDE_shortest_paths_correct_l550_55009


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l550_55038

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l550_55038


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l550_55085

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  ((2 / (x - 1) - 1 / x) / ((x^2 - 1) / (x^2 - 2*x + 1))) = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l550_55085


namespace NUMINAMATH_CALUDE_max_k_value_l550_55064

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l550_55064


namespace NUMINAMATH_CALUDE_total_insects_on_leaves_l550_55062

/-- The total number of insects on leaves with given conditions -/
def total_insects (
  num_leaves : ℕ
  ) (ladybugs_per_leaf : ℕ
  ) (ants_per_leaf : ℕ
  ) (caterpillars_per_third_leaf : ℕ
  ) : ℕ :=
  (num_leaves * ladybugs_per_leaf) +
  (num_leaves * ants_per_leaf) +
  (num_leaves / 3 * caterpillars_per_third_leaf)

/-- Theorem stating the total number of insects under given conditions -/
theorem total_insects_on_leaves :
  total_insects 84 139 97 53 = 21308 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_on_leaves_l550_55062


namespace NUMINAMATH_CALUDE_no_factor_of_polynomial_l550_55083

theorem no_factor_of_polynomial : ¬ ∃ (p : Polynomial ℝ), 
  (p = X^2 + 4*X + 4 ∨ 
   p = X^2 - 4*X + 4 ∨ 
   p = X^2 + 2*X + 4 ∨ 
   p = X^2 + 4) ∧ 
  (∃ (q : Polynomial ℝ), X^4 - 4*X^2 + 16 = p * q) := by
  sorry

end NUMINAMATH_CALUDE_no_factor_of_polynomial_l550_55083


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_l550_55052

theorem magnitude_of_complex_square : 
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_l550_55052


namespace NUMINAMATH_CALUDE_distance_downstream_l550_55096

/-- Calculates the distance traveled downstream given boat speed, current speed, and time -/
theorem distance_downstream (boat_speed current_speed : ℝ) (time_minutes : ℝ) :
  boat_speed = 20 ∧ current_speed = 4 ∧ time_minutes = 24 →
  (boat_speed + current_speed) * (time_minutes / 60) = 9.6 := by
  sorry

#check distance_downstream

end NUMINAMATH_CALUDE_distance_downstream_l550_55096


namespace NUMINAMATH_CALUDE_smallest_k_for_same_remainder_l550_55026

theorem smallest_k_for_same_remainder : ∃ (k : ℕ), k > 0 ∧
  (∀ (n : ℕ), n > 0 → n < k → ¬((201 + n) % 24 = (9 + n) % 24)) ∧
  ((201 + k) % 24 = (9 + k) % 24) ∧
  (201 % 24 = 9 % 24) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_same_remainder_l550_55026


namespace NUMINAMATH_CALUDE_total_cost_is_180_l550_55060

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let palm_fern_cost : ℝ := 15.00
  let creeping_jenny_cost : ℝ := 4.00
  let geranium_cost : ℝ := 3.50
  let plants_per_pot : ℕ := 1 + 4 + 4
  let cost_per_pot : ℝ := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost
  let corners : ℕ := 4
  corners * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_180_l550_55060


namespace NUMINAMATH_CALUDE_add_specific_reals_l550_55039

theorem add_specific_reals : 1.25 + 47.863 = 49.113 := by
  sorry

end NUMINAMATH_CALUDE_add_specific_reals_l550_55039


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l550_55037

/-- A normally distributed random variable -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (X : NormalRV) (f : ℝ → Prop) : ℝ := sorry

theorem normal_distribution_probability 
  (X : NormalRV)
  (h1 : P X (λ x => x > 5) = 0.2)
  (h2 : P X (λ x => x < -1) = 0.2) :
  P X (λ x => 2 < x ∧ x < 5) = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l550_55037


namespace NUMINAMATH_CALUDE_pool_capacity_l550_55082

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C = 0.5 * C + 300) 
  (h2 : 300 = 0.3 * C) : 
  C = 1000 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l550_55082


namespace NUMINAMATH_CALUDE_tetrahedron_distance_altitude_inequality_l550_55008

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between any pair of opposite edges -/
  d : ℝ
  /-- The length of the shortest altitude -/
  h : ℝ
  /-- Assumption that d and h are positive -/
  d_pos : d > 0
  h_pos : h > 0

/-- Theorem: For any tetrahedron, twice the minimum distance between opposite edges
    is greater than the length of the shortest altitude -/
theorem tetrahedron_distance_altitude_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_altitude_inequality_l550_55008


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l550_55041

-- Define an isosceles triangle with two known side lengths
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b ∨ a = 3 ∨ b = 3

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + 3

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) (h1 : t.a = 3 ∨ t.a = 4) (h2 : t.b = 3 ∨ t.b = 4) :
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l550_55041


namespace NUMINAMATH_CALUDE_inequality_solution_range_l550_55035

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l550_55035


namespace NUMINAMATH_CALUDE_no_perfect_square_9999xxxx_l550_55095

theorem no_perfect_square_9999xxxx : 
  ¬ ∃ x : ℕ, (99990000 ≤ x ∧ x ≤ 99999999) ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_9999xxxx_l550_55095


namespace NUMINAMATH_CALUDE_overlapping_area_l550_55031

theorem overlapping_area (total_length : ℝ) (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    left_length / right_length = (left_only_area + overlap_area) / (right_only_area + overlap_area) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_l550_55031


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l550_55045

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    ∀ r s : ℕ, Prime r → Prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l550_55045


namespace NUMINAMATH_CALUDE_total_toys_count_l550_55069

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The total number of toys given -/
def total_toys : ℕ := toy_cars + dolls

theorem total_toys_count : total_toys = 403 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l550_55069


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l550_55023

theorem pencil_eraser_cost_problem :
  ∃ (p e : ℕ), 
    p > 0 ∧ 
    e > 0 ∧ 
    10 * p + 2 * e = 110 ∧ 
    p < e ∧ 
    p + e = 19 :=
by sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l550_55023


namespace NUMINAMATH_CALUDE_x_congruence_l550_55076

theorem x_congruence (x : ℤ) 
  (h1 : (2 + x) % 4 = 3 % 4)
  (h2 : (4 + x) % 16 = 8 % 16)
  (h3 : (6 + x) % 36 = 7 % 36) :
  x % 48 = 1 % 48 := by
sorry

end NUMINAMATH_CALUDE_x_congruence_l550_55076


namespace NUMINAMATH_CALUDE_no_integer_list_with_mean_6_35_l550_55015

theorem no_integer_list_with_mean_6_35 :
  ¬ ∃ (lst : List ℤ), lst.length = 35 ∧ (lst.sum : ℚ) / 35 = 35317 / 5560 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_list_with_mean_6_35_l550_55015


namespace NUMINAMATH_CALUDE_assistant_coaches_average_age_l550_55073

/-- The average age of assistant coaches in a sports club --/
theorem assistant_coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (girls_count : ℕ) 
  (girls_average : ℕ) 
  (boys_count : ℕ) 
  (boys_average : ℕ) 
  (head_coaches_count : ℕ) 
  (head_coaches_average : ℕ) 
  (assistant_coaches_count : ℕ) 
  (h_total : total_members = 50)
  (h_overall : overall_average = 22)
  (h_girls : girls_count = 30)
  (h_girls_avg : girls_average = 18)
  (h_boys : boys_count = 15)
  (h_boys_avg : boys_average = 20)
  (h_head_coaches : head_coaches_count = 3)
  (h_head_coaches_avg : head_coaches_average = 30)
  (h_assistant_coaches : assistant_coaches_count = 2)
  (h_coaches_total : head_coaches_count + assistant_coaches_count = 5) :
  (total_members * overall_average - 
   girls_count * girls_average - 
   boys_count * boys_average - 
   head_coaches_count * head_coaches_average) / assistant_coaches_count = 85 := by
sorry


end NUMINAMATH_CALUDE_assistant_coaches_average_age_l550_55073


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l550_55054

theorem coin_fraction_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  (lost - found) / x = (1 : ℚ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l550_55054


namespace NUMINAMATH_CALUDE_max_non_managers_l550_55025

/-- Represents the number of managers in the department -/
def managers : ℕ := 8

/-- Represents the maximum total number of employees allowed in the department -/
def max_total : ℕ := 130

/-- Represents the company-wide ratio of managers to non-managers -/
def company_ratio : ℚ := 5 / 24

/-- Represents the department-specific ratio of managers to non-managers -/
def dept_ratio : ℚ := 3 / 5

/-- Theorem stating the maximum number of non-managers in the department -/
theorem max_non_managers :
  ∃ (n : ℕ), n = 13 ∧ 
  (managers : ℚ) / n > company_ratio ∧
  (managers : ℚ) / n ≤ dept_ratio ∧
  managers + n ≤ max_total ∧
  ∀ (m : ℕ), m > n → 
    ((managers : ℚ) / m ≤ company_ratio ∨
     (managers : ℚ) / m > dept_ratio ∨
     managers + m > max_total) :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l550_55025


namespace NUMINAMATH_CALUDE_bike_rental_cost_theorem_l550_55000

/-- The fee structure and rental details for a bicycle rental service. -/
structure BikeRental where
  fee_per_30min : ℕ  -- Fee in won for 30 minutes
  num_bikes : ℕ      -- Number of bikes rented
  duration_hours : ℕ -- Duration of rental in hours
  num_people : ℕ     -- Number of people splitting the cost

/-- Calculate the cost per person for a bike rental. -/
def cost_per_person (rental : BikeRental) : ℕ :=
  let total_cost := rental.fee_per_30min * 2 * rental.duration_hours * rental.num_bikes
  total_cost / rental.num_people

/-- Theorem stating that under the given conditions, each person pays 16000 won. -/
theorem bike_rental_cost_theorem (rental : BikeRental) 
  (h1 : rental.fee_per_30min = 4000)
  (h2 : rental.num_bikes = 4)
  (h3 : rental.duration_hours = 3)
  (h4 : rental.num_people = 6) : 
  cost_per_person rental = 16000 := by
  sorry

end NUMINAMATH_CALUDE_bike_rental_cost_theorem_l550_55000


namespace NUMINAMATH_CALUDE_planes_parallel_conditions_l550_55029

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_conditions 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∃ a : Line, perpendicular a α ∧ perpendicular a β → plane_parallel α β) ∧
  (∃ a b : Line, skew a b ∧ contains α a ∧ contains β b ∧ 
    parallel a β ∧ parallel b α → plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_conditions_l550_55029


namespace NUMINAMATH_CALUDE_compute_expression_l550_55002

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l550_55002


namespace NUMINAMATH_CALUDE_gcd_1248_585_l550_55074

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1248_585_l550_55074


namespace NUMINAMATH_CALUDE_union_of_S_and_T_l550_55047

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_of_S_and_T : S ∪ T = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_S_and_T_l550_55047


namespace NUMINAMATH_CALUDE_impossible_to_reach_in_six_moves_l550_55003

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a single move of the ant -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Applies a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up    => ⟨p.x, p.y + 1⟩
  | Move.Down  => ⟨p.x, p.y - 1⟩
  | Move.Left  => ⟨p.x - 1, p.y⟩
  | Move.Right => ⟨p.x + 1, p.y⟩

/-- Applies a list of moves to a starting position -/
def apply_moves (start : Position) (moves : List Move) : Position :=
  moves.foldl apply_move start

/-- The sum of coordinates of a position -/
def coord_sum (p : Position) : Int := p.x + p.y

/-- Theorem: It's impossible to reach (2,1) or (1,2) from (0,0) in exactly 6 moves -/
theorem impossible_to_reach_in_six_moves :
  ∀ (moves : List Move),
    moves.length = 6 →
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨2, 1⟩) ∧
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨1, 2⟩) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_reach_in_six_moves_l550_55003


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l550_55006

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the circle F
def circle_F (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 3 = 0

-- Define the right focus F of hyperbola C
def right_focus (c : ℝ) : Prop :=
  c = 2

-- Define the distance from F to asymptote
def distance_to_asymptote (b : ℝ) : Prop :=
  b = 1

-- Theorem statement
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∃ x y, hyperbola_C x y a b) →
  (∃ x y, circle_F x y) →
  right_focus c →
  distance_to_asymptote b →
  c^2 = a^2 + b^2 →
  c / a = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l550_55006


namespace NUMINAMATH_CALUDE_only_two_is_sum_of_squares_among_repeating_twos_l550_55043

def is_repeating_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * (10^k - 1) / 9

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

theorem only_two_is_sum_of_squares_among_repeating_twos :
  ∀ n : ℕ, is_repeating_two n → (is_sum_of_two_squares n ↔ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_only_two_is_sum_of_squares_among_repeating_twos_l550_55043


namespace NUMINAMATH_CALUDE_arc_length_from_sector_area_l550_55013

/-- Given a circle with radius 5 cm and a sector with area 13.75 cm²,
    prove that the length of the arc forming the sector is 5.5 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) :
  r = 5 →
  area = 13.75 →
  arc_length = (2 * area) / r →
  arc_length = 5.5 :=
by
  sorry

#check arc_length_from_sector_area

end NUMINAMATH_CALUDE_arc_length_from_sector_area_l550_55013


namespace NUMINAMATH_CALUDE_negation_equivalence_l550_55067

-- Define the original statement
def P : Prop := ∀ n : ℤ, 3 ∣ n → Odd n

-- Define the correct negation
def not_P : Prop := ∃ n : ℤ, 3 ∣ n ∧ ¬(Odd n)

-- Theorem stating that not_P is indeed the negation of P
theorem negation_equivalence : ¬P ↔ not_P := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l550_55067


namespace NUMINAMATH_CALUDE_combined_mpg_calculation_l550_55042

/-- Calculates the combined miles per gallon for three cars given their individual efficiencies and a common distance traveled. -/
def combinedMPG (ray_mpg tom_mpg amy_mpg distance : ℚ) : ℚ :=
  let total_distance := 3 * distance
  let total_gas := distance / ray_mpg + distance / tom_mpg + distance / amy_mpg
  total_distance / total_gas

/-- Theorem stating that the combined MPG for the given conditions is 3600/114 -/
theorem combined_mpg_calculation :
  combinedMPG 50 20 40 120 = 3600 / 114 := by
  sorry

#eval combinedMPG 50 20 40 120

end NUMINAMATH_CALUDE_combined_mpg_calculation_l550_55042


namespace NUMINAMATH_CALUDE_expected_value_of_12_sided_die_l550_55014

/-- A fair 12-sided die -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob (n : ℕ) : ℚ := if n ∈ fair_12_sided_die then 1 / 12 else 0

/-- The expected value of a roll of a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => x * prob x)) / 1

theorem expected_value_of_12_sided_die : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_12_sided_die_l550_55014


namespace NUMINAMATH_CALUDE_system_solution_l550_55033

theorem system_solution :
  ∃ (x y₁ y₂ : ℝ),
    (x / 5 + 3 = 4) ∧
    (x^2 - 4*x*y₁ + 3*y₁^2 = 36) ∧
    (x^2 - 4*x*y₂ + 3*y₂^2 = 36) ∧
    (x = 5) ∧
    (y₁ = 10/3 + Real.sqrt 133 / 3) ∧
    (y₂ = 10/3 - Real.sqrt 133 / 3) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l550_55033


namespace NUMINAMATH_CALUDE_ellipse_distance_theorem_l550_55098

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem about the distance AF₂ in the given ellipse problem -/
theorem ellipse_distance_theorem (E : Ellipse) (F₁ F₂ A B : Point) : 
  -- F₁ and F₂ are the foci of E
  (∀ P : Point, distance P F₁ + distance P F₂ = 2 * E.a) →
  -- Line through F₁ intersects E at A and B
  (∃ t : ℝ, A = ⟨t * F₁.x, t * F₁.y⟩ ∧ B = ⟨(1 - t) * F₁.x, (1 - t) * F₁.y⟩) →
  -- |AF₁| = 3|F₁B|
  distance A F₁ = 3 * distance F₁ B →
  -- |AB| = 4
  distance A B = 4 →
  -- Perimeter of triangle ABF₂ is 16
  distance A B + distance B F₂ + distance F₂ A = 16 →
  -- Then |AF₂| = 5
  distance A F₂ = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_distance_theorem_l550_55098


namespace NUMINAMATH_CALUDE_action_figures_earnings_l550_55040

/-- Calculates the total earnings from selling action figures with discounts -/
def total_earnings (type_a_count type_b_count type_c_count type_d_count : ℕ)
                   (type_a_value type_b_value type_c_value type_d_value : ℕ)
                   (type_a_discount type_b_discount type_c_discount type_d_discount : ℕ) : ℕ :=
  (type_a_count * (type_a_value - type_a_discount)) +
  (type_b_count * (type_b_value - type_b_discount)) +
  (type_c_count * (type_c_value - type_c_discount)) +
  (type_d_count * (type_d_value - type_d_discount))

/-- Theorem stating that the total earnings from selling all action figures is $435 -/
theorem action_figures_earnings :
  total_earnings 6 5 4 5 22 35 45 50 10 14 18 20 = 435 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_earnings_l550_55040


namespace NUMINAMATH_CALUDE_permutation_difference_l550_55046

def permutation (n : ℕ) (r : ℕ) : ℕ :=
  (n - r + 1).factorial / (n - r).factorial

theorem permutation_difference : permutation 8 4 - 2 * permutation 8 2 = 1568 := by
  sorry

end NUMINAMATH_CALUDE_permutation_difference_l550_55046


namespace NUMINAMATH_CALUDE_problem_solution_l550_55066

theorem problem_solution (x y : ℝ) : 
  ((x + 2)^3 < x^3 + 8*x^2 + 42*x + 27) ∧
  ((x^3 + 8*x^2 + 42*x + 27) < (x + 4)^3) ∧
  (y = x + 3) ∧
  ((x + 3)^3 = x^3 + 9*x^2 + 27*x + 27) ∧
  ((x + 3)^3 = x^3 + 8*x^2 + 42*x + 27) ∧
  (x^2 = 15*x) →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l550_55066


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l550_55012

theorem integral_reciprocal_plus_one (u : ℝ) : 
  ∫ x in (0:ℝ)..(1:ℝ), 1 / (x + 1) = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l550_55012


namespace NUMINAMATH_CALUDE_inequality_proof_l550_55072

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b/(a+b))^2 + (a*b/(a+b))*(a*c/(a+c)) + (a*c/(a+c))^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l550_55072


namespace NUMINAMATH_CALUDE_fraction_equality_l550_55044

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64 + 128) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l550_55044


namespace NUMINAMATH_CALUDE_square_difference_value_l550_55063

theorem square_difference_value (a b : ℝ) 
  (h1 : 3 * (a + b) = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l550_55063


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l550_55099

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l550_55099


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l550_55068

/-- The range of m for which the line y = kx + 1 always intersects 
    with the ellipse x²/5 + y²/m = 1 for any real k -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ (m ≥ 1 ∧ m ≠ 5) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l550_55068


namespace NUMINAMATH_CALUDE_acute_triangle_selection_l550_55004

/-- A point on a circle, with a color attribute -/
structure ColoredPoint where
  point : ℝ × ℝ
  color : Nat

/-- Represents a circle with colored points -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  points : List ColoredPoint

/-- Checks if three points form an acute or right-angled triangle -/
def isAcuteOrRightTriangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Checks if a ColoredCircle has at least one point of each color (assuming colors are 1, 2, 3) -/
def hasAllColors (circle : ColoredCircle) : Prop := sorry

/-- The main theorem to be proved -/
theorem acute_triangle_selection (circle : ColoredCircle) 
  (h : hasAllColors circle) : 
  ∃ (p1 p2 p3 : ColoredPoint), 
    p1 ∈ circle.points ∧ 
    p2 ∈ circle.points ∧ 
    p3 ∈ circle.points ∧ 
    p1.color ≠ p2.color ∧ 
    p2.color ≠ p3.color ∧ 
    p1.color ≠ p3.color ∧ 
    isAcuteOrRightTriangle p1.point p2.point p3.point := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_selection_l550_55004


namespace NUMINAMATH_CALUDE_symmetrical_points_product_l550_55001

/-- 
Given two points P₁(a, 5) and P₂(-4, b) that are symmetrical about the x-axis,
prove that their x-coordinate product is -20.
-/
theorem symmetrical_points_product (a b : ℝ) : 
  (a = 4 ∧ b = -5) → a * b = -20 := by sorry

end NUMINAMATH_CALUDE_symmetrical_points_product_l550_55001
