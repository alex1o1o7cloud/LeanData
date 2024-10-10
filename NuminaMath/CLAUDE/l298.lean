import Mathlib

namespace women_fair_hair_percentage_l298_29832

-- Define the total number of employees
variable (E : ℝ)

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_ratio : ℝ := 0.4

-- Define the percentage of employees who have fair hair
def fair_haired_ratio : ℝ := 0.8

-- Define the percentage of employees who are women with fair hair
def women_fair_hair_ratio : ℝ := fair_haired_women_ratio * fair_haired_ratio

-- Theorem statement
theorem women_fair_hair_percentage :
  women_fair_hair_ratio = 0.32 :=
sorry

end women_fair_hair_percentage_l298_29832


namespace snow_probability_first_week_january_l298_29871

def probability_of_snow (days : ℕ) (prob : ℚ) : ℚ :=
  1 - (1 - prob) ^ days

theorem snow_probability_first_week_january : 
  1 - (1 - probability_of_snow 3 (1/2)) * (1 - probability_of_snow 4 (1/3)) = 79/81 := by
  sorry

end snow_probability_first_week_january_l298_29871


namespace sum_of_digits_7_power_1500_l298_29879

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to get the tens digit of a two-digit number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_7_power_1500 :
  tensDigit (lastTwoDigits (7^1500)) + unitsDigit (lastTwoDigits (7^1500)) = 2 := by
  sorry

end sum_of_digits_7_power_1500_l298_29879


namespace only_solutions_are_24_and_42_l298_29896

/-- Reverses the digits of a natural number -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Computes the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number has no zeros in its decimal representation -/
def no_zeros (n : ℕ) : Prop := sorry

/-- The main theorem stating that 24 and 42 are the only solutions -/
theorem only_solutions_are_24_and_42 :
  {X : ℕ | no_zeros X ∧ X * (reverse_digits X) = 1000 + product_of_digits X} = {24, 42} :=
sorry

end only_solutions_are_24_and_42_l298_29896


namespace sum_of_factors_l298_29829

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60 →
  a + b + c + d + e = 24 := by
sorry

end sum_of_factors_l298_29829


namespace perfect_square_factorization_l298_29801

/-- Perfect square formula check -/
def isPerfectSquare (a b c : ℝ) : Prop :=
  ∃ (k : ℝ), a * c = (b / 2) ^ 2 ∧ a > 0

theorem perfect_square_factorization :
  ¬ isPerfectSquare 1 (1/4 : ℝ) (1/4) ∧
  ¬ isPerfectSquare 1 (2 : ℝ) (-1) ∧
  ¬ isPerfectSquare 1 (1 : ℝ) 1 ∧
  isPerfectSquare 4 (4 : ℝ) 1 :=
by sorry

end perfect_square_factorization_l298_29801


namespace gcd_n_cubed_plus_16_and_n_plus_3_l298_29816

theorem gcd_n_cubed_plus_16_and_n_plus_3 (n : ℕ) (h : n > 8) :
  Nat.gcd (n^3 + 16) (n + 3) = 1 := by
  sorry

end gcd_n_cubed_plus_16_and_n_plus_3_l298_29816


namespace area_between_concentric_circles_l298_29804

theorem area_between_concentric_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 60) 
  (h₂ : r₂ = 40) 
  (h₃ : chord_length = 100) 
  (h₄ : r₁ > r₂) 
  (h₅ : chord_length / 2 > r₂) : 
  (r₁^2 - r₂^2) * π = 2500 * π := by
  sorry

end area_between_concentric_circles_l298_29804


namespace trapezoid_xy_length_l298_29840

-- Define the trapezoid and its properties
structure Trapezoid where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  wy_perp_zy : (W.1 - Y.1) * (Z.1 - Y.1) + (W.2 - Y.2) * (Z.2 - Y.2) = 0

-- Define the given conditions
def trapezoid_conditions (t : Trapezoid) : Prop :=
  let (_, y2) := t.Y
  let (_, z2) := t.Z
  let yz_length := Real.sqrt ((t.Y.1 - t.Z.1)^2 + (y2 - z2)^2)
  let tan_z := (t.W.2 - t.Z.2) / (t.W.1 - t.Z.1)
  let tan_x := (t.W.2 - t.X.2) / (t.X.1 - t.W.1)
  yz_length = 15 ∧ tan_z = 2 ∧ tan_x = 2.5

-- State the theorem
theorem trapezoid_xy_length (t : Trapezoid) (h : trapezoid_conditions t) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 6 * Real.sqrt 29 := by
  sorry

end trapezoid_xy_length_l298_29840


namespace average_age_decrease_l298_29867

theorem average_age_decrease (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (original_strength : ℕ) :
  original_average = 40 →
  new_students = 15 →
  new_average = 32 →
  original_strength = 15 →
  let total_students := original_strength + new_students
  let new_total_age := original_average * original_strength + new_average * new_students
  let final_average := new_total_age / total_students
  40 - final_average = 4 :=
by sorry

end average_age_decrease_l298_29867


namespace tank_fill_time_l298_29860

/-- Represents the time it takes to fill a tank given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given specific pipe rates, the fill time is 20 minutes -/
theorem tank_fill_time :
  fill_time (1/18) (1/60) (-1/45) = 20 := by
  sorry

end tank_fill_time_l298_29860


namespace bread_price_for_cash_register_l298_29875

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadSold : ℕ
  cakesPrice : ℝ
  cakesSold : ℕ
  rentCost : ℝ
  electricityCost : ℝ

/-- Calculates the daily profit of the bakery --/
def dailyProfit (b : BakeryFinances) : ℝ :=
  b.breadPrice * b.breadSold + b.cakesPrice * b.cakesSold - b.rentCost - b.electricityCost

/-- The main theorem: The price of bread that allows Marie to buy the cash register in 8 days is $2 --/
theorem bread_price_for_cash_register (b : BakeryFinances) 
    (h1 : b.breadSold = 40)
    (h2 : b.cakesSold = 6)
    (h3 : b.cakesPrice = 12)
    (h4 : b.rentCost = 20)
    (h5 : b.electricityCost = 2)
    (h6 : 8 * dailyProfit b = 1040) : 
  b.breadPrice = 2 := by
  sorry

#check bread_price_for_cash_register

end bread_price_for_cash_register_l298_29875


namespace total_games_in_season_l298_29889

theorem total_games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (h1 : total_teams = 16)
  (h2 : teams_per_division = 8)
  (h3 : total_teams = 2 * teams_per_division)
  (h4 : ∀ (division : Fin 2), ∀ (team : Fin teams_per_division),
    (division.val = 0 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22) ∧
    (division.val = 1 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22)) :
  total_teams * 22 / 2 = 176 := by
sorry

end total_games_in_season_l298_29889


namespace inequality_always_true_l298_29861

theorem inequality_always_true (x : ℝ) : (7 / 20) + |3 * x - (2 / 5)| ≥ (1 / 4) := by
  sorry

end inequality_always_true_l298_29861


namespace students_liking_both_desserts_l298_29827

theorem students_liking_both_desserts
  (total_students : ℕ)
  (like_apple_pie : ℕ)
  (like_chocolate_cake : ℕ)
  (like_neither : ℕ)
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 10) :
  (like_apple_pie + like_chocolate_cake) - (total_students - like_neither) = 5 :=
by
  sorry

end students_liking_both_desserts_l298_29827


namespace symmetric_axis_after_transformation_l298_29890

/-- Given a function f(x) = √3 sin(x - π/6) + cos(x - π/6), 
    after stretching the horizontal coordinate to twice its original length 
    and shifting the graph π/6 units to the left, 
    one symmetric axis of the resulting function is at x = 5π/6 -/
theorem symmetric_axis_after_transformation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (x - π/6) + Real.cos (x - π/6)
  let g : ℝ → ℝ := λ x => f ((x + π/6) / 2)
  ∃ (k : ℤ), g (5*π/6 + 2*π*k) = g (5*π/6 - 2*π*k) := by
  sorry


end symmetric_axis_after_transformation_l298_29890


namespace smallest_benches_arrangement_l298_29822

theorem smallest_benches_arrangement (M : ℕ+) (n : ℕ+) : 
  (9 * M.val = n ∧ 14 * M.val = n) → M.val ≥ 9 :=
by sorry

end smallest_benches_arrangement_l298_29822


namespace probability_three_even_dice_l298_29842

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12

theorem probability_three_even_dice :
  let p := (num_dice.choose 3) * (1 / 2) ^ num_dice / 1
  p = 5 / 16 := by sorry

end probability_three_even_dice_l298_29842


namespace determine_x_with_gcd_queries_l298_29834

theorem determine_x_with_gcd_queries :
  ∀ X : ℕ+, X ≤ 100 →
  ∃ (queries : Fin 7 → ℕ+ × ℕ+),
    (∀ i, (queries i).1 < 100 ∧ (queries i).2 < 100) ∧
    ∀ Y : ℕ+, Y ≤ 100 →
      (∀ i, Nat.gcd (X + (queries i).1) (queries i).2 = Nat.gcd (Y + (queries i).1) (queries i).2) →
      X = Y := by
  sorry

end determine_x_with_gcd_queries_l298_29834


namespace min_value_my_plus_nx_l298_29837

theorem min_value_my_plus_nx (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∀ z : ℝ, m * y + n * x ≥ z → z ≥ -2 := by
  sorry

end min_value_my_plus_nx_l298_29837


namespace volume_central_region_is_one_sixth_l298_29815

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  -- Add necessary fields/axioms for a unit cube

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- Represents the central region (regular octahedron) formed by intersecting planes -/
structure CentralRegion where
  cube : UnitCube
  intersecting_planes : List Plane
  -- Add necessary conditions to ensure the planes intersect at midpoints of edges

/-- Calculate the volume of the central region in a unit cube intersected by specific planes -/
def volume_central_region (region : CentralRegion) : ℝ :=
  sorry

/-- Theorem stating that the volume of the central region is 1/6 -/
theorem volume_central_region_is_one_sixth (region : CentralRegion) :
  volume_central_region region = 1 / 6 :=
sorry

end volume_central_region_is_one_sixth_l298_29815


namespace fixed_point_of_function_l298_29805

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1) + 2
  f (-1) = 3 := by
sorry

end fixed_point_of_function_l298_29805


namespace special_triangle_angles_special_triangle_exists_l298_29803

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Angles of the triangle
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Condition: The sum of angles in a triangle is 180°
  angle_sum : angleA + angleB + angleC = 180
  -- Condition: Angle C is a right angle
  right_angleC : angleC = 90
  -- Condition: Angle A is one-fourth of angle B
  angle_relation : angleA = angleB / 3

/-- Theorem stating the angles of the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angleA = 22.5 ∧ t.angleB = 67.5 ∧ t.angleC = 90 := by
  sorry

/-- The existence of such a triangle -/
theorem special_triangle_exists : ∃ t : SpecialTriangle, True := by
  sorry

end special_triangle_angles_special_triangle_exists_l298_29803


namespace average_problem_l298_29893

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of four numbers
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem average_problem :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end average_problem_l298_29893


namespace correct_pricing_strategy_l298_29878

/-- Represents the cost and pricing structure of items A and B -/
structure ItemPricing where
  cost_A : ℝ
  cost_B : ℝ
  initial_price_A : ℝ
  price_reduction_A : ℝ

/-- Represents the sales data for items A and B -/
structure SalesData where
  initial_sales_A : ℕ
  sales_increase_rate : ℝ
  revenue_B : ℝ

/-- Theorem stating the correct pricing and reduction strategy -/
theorem correct_pricing_strategy 
  (p : ItemPricing) 
  (s : SalesData) 
  (h1 : 5 * p.cost_A + 3 * p.cost_B = 450)
  (h2 : 10 * p.cost_A + 8 * p.cost_B = 1000)
  (h3 : p.initial_price_A = 80)
  (h4 : s.initial_sales_A = 100)
  (h5 : s.sales_increase_rate = 20)
  (h6 : s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A > 200)
  (h7 : s.revenue_B = 7000)
  (h8 : (p.initial_price_A - p.price_reduction_A) * 
        (s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A) + 
        s.revenue_B = 10000) :
  p.cost_A = 60 ∧ p.cost_B = 50 ∧ p.price_reduction_A = 10 := by
  sorry

end correct_pricing_strategy_l298_29878


namespace abs_neg_abs_square_minus_one_eq_zero_l298_29866

theorem abs_neg_abs_square_minus_one_eq_zero :
  |(-|(-1 + 2)|)^2 - 1| = 0 := by
  sorry

end abs_neg_abs_square_minus_one_eq_zero_l298_29866


namespace marie_bike_distance_l298_29855

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that Marie biked 31 miles -/
theorem marie_bike_distance :
  let speed := 12.0
  let time := 2.583333333
  distance speed time = 31 := by
sorry

end marie_bike_distance_l298_29855


namespace bob_benefit_reduction_l298_29885

/-- Calculates the monthly reduction in housing benefit given a raise, work hours, and net increase --/
def monthly_benefit_reduction (raise_per_hour : ℚ) (hours_per_week : ℕ) (net_increase_per_week : ℚ) : ℚ :=
  4 * (raise_per_hour * hours_per_week - net_increase_per_week)

/-- Theorem stating that given the specific conditions, the monthly reduction in housing benefit is $60 --/
theorem bob_benefit_reduction :
  monthly_benefit_reduction (1/2) 40 5 = 60 := by
  sorry

end bob_benefit_reduction_l298_29885


namespace stone_division_impossibility_l298_29854

theorem stone_division_impossibility :
  ¬ ∃ (n : ℕ), n > 0 ∧ 3 * n = 1001 - (n - 1) :=
by
  sorry

end stone_division_impossibility_l298_29854


namespace maximize_subsidy_l298_29891

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1) - x / 10 + 1

theorem maximize_subsidy (m : ℝ) (h_m : m > 0) :
  let max_subsidy := fun x : ℝ => x ≥ 1 ∧ x ≤ 9 ∧ ∀ y, 1 ≤ y ∧ y ≤ 9 → f m x ≥ f m y
  (m ≤ 1/5 ∧ max_subsidy 1) ∨
  (1/5 < m ∧ m < 1 ∧ max_subsidy (10*m - 1)) ∨
  (m ≥ 1 ∧ max_subsidy 9) :=
by sorry

end maximize_subsidy_l298_29891


namespace four_tellers_coins_l298_29892

/-- Calculates the total number of coins for a given number of bank tellers -/
def totalCoins (numTellers : ℕ) (rollsPerTeller : ℕ) (coinsPerRoll : ℕ) : ℕ :=
  numTellers * rollsPerTeller * coinsPerRoll

/-- Theorem: Four bank tellers have 1000 coins in total -/
theorem four_tellers_coins :
  totalCoins 4 10 25 = 1000 := by
  sorry

#eval totalCoins 4 10 25  -- Should output 1000

end four_tellers_coins_l298_29892


namespace eggs_ratio_is_one_to_one_l298_29850

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the total number of eggs Megan initially had --/
def initial_eggs : ℕ := 2 * dozen

/-- Represents the number of eggs Megan used for cooking --/
def used_eggs : ℕ := 2 + 4

/-- Represents the number of eggs Megan plans to use for her meals --/
def planned_meals_eggs : ℕ := 3 * 3

/-- Theorem stating that the ratio of eggs Megan gave to her aunt to the eggs she kept for herself is 1:1 --/
theorem eggs_ratio_is_one_to_one : 
  (initial_eggs - used_eggs - planned_meals_eggs) = planned_meals_eggs := by
  sorry

end eggs_ratio_is_one_to_one_l298_29850


namespace multiple_of_nine_squared_greater_than_80_less_than_30_l298_29872

theorem multiple_of_nine_squared_greater_than_80_less_than_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 80)
  (h3 : x < 30) :
  x = 9 ∨ x = 18 ∨ x = 27 := by
sorry

end multiple_of_nine_squared_greater_than_80_less_than_30_l298_29872


namespace inequality_reversal_l298_29819

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(1 - x > 1 - y) := by
  sorry

end inequality_reversal_l298_29819


namespace rhino_state_reachable_l298_29846

/-- Represents the state of a Rhinoceros with folds on its skin -/
structure RhinoState :=
  (left_vertical : Nat)
  (left_horizontal : Nat)
  (right_vertical : Nat)
  (right_horizontal : Nat)

/-- Represents the direction of scratching -/
inductive ScratchDirection
  | Vertical
  | Horizontal

/-- Represents the side of the Rhinoceros being scratched -/
inductive Side
  | Left
  | Right

/-- Defines a single transition step for a Rhinoceros state -/
def transition (s : RhinoState) (dir : ScratchDirection) (side : Side) : RhinoState :=
  sorry

/-- Defines if a target state is reachable from an initial state -/
def is_reachable (initial : RhinoState) (target : RhinoState) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem rhino_state_reachable :
  is_reachable
    (RhinoState.mk 0 2 2 1)
    (RhinoState.mk 2 0 2 1) :=
  sorry

end rhino_state_reachable_l298_29846


namespace hyperbola_asymptotes_l298_29807

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0, and eccentricity e = 2,
    the equation of its asymptotes is y = ±√3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end hyperbola_asymptotes_l298_29807


namespace compute_expression_l298_29847

theorem compute_expression : 3 * 3^4 - 27^60 / 27^58 = -486 := by
  sorry

end compute_expression_l298_29847


namespace four_twos_polynomial_property_l298_29874

/-- A polynomial that takes the value 2 for four different integer inputs -/
def FourTwosPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  P a = 2 ∧ P b = 2 ∧ P c = 2 ∧ P d = 2

theorem four_twos_polynomial_property (P : ℤ → ℤ) 
  (h : FourTwosPolynomial P) :
  ∀ x : ℤ, P x ≠ 1 ∧ P x ≠ 3 ∧ P x ≠ 5 ∧ P x ≠ 7 ∧ P x ≠ 9 :=
sorry

end four_twos_polynomial_property_l298_29874


namespace inverse_function_ln_l298_29881

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

noncomputable def g (x : ℝ) : ℝ := Real.exp x + 1

theorem inverse_function_ln (x : ℝ) (hx : x > 2) :
  Function.Injective f ∧
  Function.Surjective f ∧
  (∀ y, y > 0 → g y > 2) ∧
  (∀ y, y > 0 → f (g y) = y) ∧
  (∀ x, x > 2 → g (f x) = x) := by
  sorry

end inverse_function_ln_l298_29881


namespace sum_of_nth_row_l298_29841

/-- Represents the sum of numbers in the nth row of the triangular array -/
def row_sum (n : ℕ) : ℕ := 2^n

/-- The first row sum is 2 -/
axiom first_row : row_sum 1 = 2

/-- Each subsequent row sum is double the previous row sum -/
axiom double_previous (n : ℕ) : n ≥ 1 → row_sum (n + 1) = 2 * row_sum n

/-- The sum of numbers in the nth row of the triangular array is 2^n -/
theorem sum_of_nth_row (n : ℕ) : n ≥ 1 → row_sum n = 2^n := by sorry

end sum_of_nth_row_l298_29841


namespace sequence_property_l298_29833

def sequence_sum (n : ℕ) : ℚ := n * (3 * n - 1) / 2

def sequence_term (n : ℕ) : ℚ := 3 * n - 2

theorem sequence_property (m : ℕ) :
  (∀ n, sequence_sum n = n * (3 * n - 1) / 2) →
  (∀ n, sequence_term n = 3 * n - 2) →
  sequence_term 1 * sequence_term m = (sequence_term 4) ^ 2 →
  m = 34 := by
  sorry

end sequence_property_l298_29833


namespace dividend_calculation_l298_29873

/-- Calculates the total dividends received over three years given an initial investment and dividend rates. -/
def total_dividends (initial_investment : ℚ) (share_face_value : ℚ) (initial_premium : ℚ) 
  (dividend_rate1 : ℚ) (dividend_rate2 : ℚ) (dividend_rate3 : ℚ) : ℚ :=
  let cost_per_share := share_face_value * (1 + initial_premium)
  let num_shares := initial_investment / cost_per_share
  let dividend1 := num_shares * share_face_value * dividend_rate1
  let dividend2 := num_shares * share_face_value * dividend_rate2
  let dividend3 := num_shares * share_face_value * dividend_rate3
  dividend1 + dividend2 + dividend3

/-- Theorem stating that the total dividends received is 2640 given the specified conditions. -/
theorem dividend_calculation :
  total_dividends 14400 100 (1/5) (7/100) (9/100) (6/100) = 2640 := by
  sorry

end dividend_calculation_l298_29873


namespace log_sum_equality_l298_29899

theorem log_sum_equality : 10^(Real.log 3 / Real.log 10) + Real.log 25 / Real.log 5 = 5 := by
  sorry

end log_sum_equality_l298_29899


namespace product_after_digit_reversal_l298_29809

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The theorem statement -/
theorem product_after_digit_reversal (x y : ℕ) :
  x ≥ 10 ∧ x < 100 ∧  -- x is a two-digit number
  y > 0 ∧  -- y is positive
  (reverse_digits x) * y = 221 →  -- erroneous product condition
  x * y = 527 ∨ x * y = 923 :=
by sorry

end product_after_digit_reversal_l298_29809


namespace prime_rational_sum_l298_29830

theorem prime_rational_sum (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℚ) (n : ℕ), x > 0 ∧ y > 0 ∧ x + y + p / x + p / y = 3 * n) ↔ 3 ∣ (p + 1) :=
sorry

end prime_rational_sum_l298_29830


namespace inequality_proof_l298_29858

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ 36 / (a - d) := by
  sorry

end inequality_proof_l298_29858


namespace base8_digit_product_7890_l298_29894

/-- Given a natural number n, returns the list of its digits in base 8 --/
def toBase8Digits (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product_7890 :
  listProduct (toBase8Digits 7890) = 336 :=
sorry

end base8_digit_product_7890_l298_29894


namespace always_close_piece_l298_29828

-- Define the grid structure
structure Grid :=
  (points : Set (ℤ × ℤ))
  (adjacent : (ℤ × ℤ) → Set (ℤ × ℤ))
  (initial : ℤ × ℤ)

-- Define the grid distance
def gridDistance (g : Grid) (p : ℤ × ℤ) : ℕ :=
  sorry

-- Define the marking function
def mark (n : ℕ) : ℚ :=
  1 / 2^n

-- Define the sum of markings for pieces
def pieceSum (g : Grid) (pieces : Set (ℤ × ℤ)) : ℚ :=
  sorry

-- Define the sum of markings for points with grid distance ≥ 7
def distantSum (g : Grid) : ℚ :=
  sorry

-- Main theorem
theorem always_close_piece (g : Grid) (pieces : Set (ℤ × ℤ)) :
  pieceSum g pieces > distantSum g :=
sorry

end always_close_piece_l298_29828


namespace bakers_cakes_l298_29865

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 8)
  (h2 : bought_cakes = 139)
  (h3 : sold_cakes = 145) :
  sold_cakes - bought_cakes = 6 := by
  sorry

end bakers_cakes_l298_29865


namespace count_ones_digits_divisible_by_six_l298_29814

/-- A number is divisible by 6 if and only if it is divisible by both 2 and 3 -/
axiom divisible_by_six (n : ℕ) : n % 6 = 0 ↔ n % 2 = 0 ∧ n % 3 = 0

/-- The set of possible ones digits in numbers divisible by 6 -/
def ones_digits_divisible_by_six : Finset ℕ :=
  {0, 2, 4, 6, 8}

/-- The number of possible ones digits in numbers divisible by 6 is 5 -/
theorem count_ones_digits_divisible_by_six :
  Finset.card ones_digits_divisible_by_six = 5 := by sorry

end count_ones_digits_divisible_by_six_l298_29814


namespace ball_placement_count_is_42_l298_29823

/-- The number of ways to place four distinct balls into three labeled boxes
    such that exactly one box remains empty. -/
def ballPlacementCount : ℕ := 42

/-- Theorem stating that the number of ways to place four distinct balls
    into three labeled boxes such that exactly one box remains empty is 42. -/
theorem ball_placement_count_is_42 : ballPlacementCount = 42 := by
  sorry

end ball_placement_count_is_42_l298_29823


namespace louisa_travel_l298_29812

/-- Louisa's travel problem -/
theorem louisa_travel (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 200)
  (h2 : speed = 50)
  (h3 : time_difference = 3)
  (h4 : first_day_distance / speed + time_difference = second_day_distance / speed) :
  second_day_distance = 350 :=
by
  sorry

end louisa_travel_l298_29812


namespace cost_of_dozen_pens_l298_29824

theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  (3 * pen_cost + 5 * pencil_cost = 150) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 450) :=
by
  sorry

end cost_of_dozen_pens_l298_29824


namespace correct_answers_for_given_exam_l298_29811

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  totalScore : ℤ

/-- Calculates the number of correctly answered questions. -/
def correctAnswers (result : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that given the specific exam conditions, 
    the number of correctly answered questions is 44. -/
theorem correct_answers_for_given_exam : 
  let exam : Exam := { totalQuestions := 60, correctScore := 4, wrongScore := -1 }
  let result : ExamResult := { exam := exam, totalScore := 160 }
  correctAnswers result = 44 := by
  sorry

end correct_answers_for_given_exam_l298_29811


namespace count_numbers_with_6_or_8_is_452_l298_29877

/-- The count of three-digit whole numbers containing at least one digit 6 or at least one digit 8 -/
def count_numbers_with_6_or_8 : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let digits_without_6_or_8 := 8  -- 0-5, 7, 9
  let first_digit_choices := 7    -- 1-5, 7, 9
  let numbers_without_6_or_8 := first_digit_choices * digits_without_6_or_8 * digits_without_6_or_8
  total_three_digit_numbers - numbers_without_6_or_8

theorem count_numbers_with_6_or_8_is_452 : count_numbers_with_6_or_8 = 452 := by
  sorry

end count_numbers_with_6_or_8_is_452_l298_29877


namespace inscribed_circle_radius_l298_29818

/-- Given a triangle DEF with side lengths DE = 26, DF = 15, and EF = 17,
    the radius of its inscribed circle is 3√2. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 3 * Real.sqrt 2 := by sorry

end inscribed_circle_radius_l298_29818


namespace parabola_tangent_line_l298_29836

/-- A parabola is tangent to a line if and only if the discriminant of their difference is zero -/
def is_tangent (a : ℝ) : Prop :=
  (4 : ℝ) - 12 * a = 0

/-- The value of a for which the parabola y = ax^2 + 6 is tangent to the line y = 2x + 3 -/
theorem parabola_tangent_line : ∃ (a : ℝ), is_tangent a ∧ a = (1/3 : ℝ) := by
  sorry

end parabola_tangent_line_l298_29836


namespace solution_sets_equal_l298_29817

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OneToOne (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = f y → x = y

def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | f (f x) = x}

theorem solution_sets_equal
  (f : ℝ → ℝ)
  (h_increasing : StrictlyIncreasing f)
  (h_onetoone : OneToOne f) :
  SolutionSetP f = SolutionSetQ f :=
sorry

end solution_sets_equal_l298_29817


namespace geometric_sequence_ratio_l298_29880

/-- Given a geometric sequence {a_n} with all positive terms,
    if a_3, (1/2)a_5, a_4 form an arithmetic sequence,
    then (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (a 3 + a 4 = a 5) →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry


end geometric_sequence_ratio_l298_29880


namespace expression_evaluation_l298_29844

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  2 * a^2 - 3 * b^2 + a * b = 12 := by
  sorry

end expression_evaluation_l298_29844


namespace babysitting_theorem_l298_29821

def babysitting_earnings (initial_charge : ℝ) (hours : ℕ) : ℝ :=
  let rec calc_earnings (h : ℕ) (prev_charge : ℝ) (total : ℝ) : ℝ :=
    if h = 0 then
      total
    else
      calc_earnings (h - 1) (prev_charge * 1.5) (total + prev_charge)
  calc_earnings hours initial_charge 0

theorem babysitting_theorem :
  babysitting_earnings 4 4 = 32.5 := by
  sorry

end babysitting_theorem_l298_29821


namespace midpoint_square_area_l298_29808

theorem midpoint_square_area (A B C D : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (1, 0) → 
  C = (1, 1) → 
  D = (0, 1) → 
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let Q := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 1/2 :=
by sorry

end midpoint_square_area_l298_29808


namespace function_properties_l298_29839

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2 + 1

-- Define the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 2 ∨ a = -2) ∧
  (∀ x y : ℝ, x < y ∧ x ≤ 0 ∧ 0 < y → f a x > f a y) ∧
  (∀ x y : ℝ, x < y ∧ 0 < x → f a x < f a y) :=
by
  sorry

end function_properties_l298_29839


namespace divisibility_problem_l298_29852

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) 
  (hy : y ≠ -1) 
  (h_int : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end divisibility_problem_l298_29852


namespace tank_capacity_l298_29883

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) :
  initial_fraction = 5 / 8 →
  final_fraction = 19 / 24 →
  added_amount = 15 →
  ∃ (total_capacity : ℚ),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 90 := by
  sorry

#check tank_capacity

end tank_capacity_l298_29883


namespace quiz_max_correct_answers_l298_29843

theorem quiz_max_correct_answers :
  ∀ (correct blank incorrect : ℕ),
    correct + blank + incorrect = 60 →
    5 * correct - 2 * incorrect = 150 →
    correct ≤ 38 ∧
    ∃ (c b i : ℕ), c + b + i = 60 ∧ 5 * c - 2 * i = 150 ∧ c = 38 := by
  sorry

end quiz_max_correct_answers_l298_29843


namespace equation_equivalence_l298_29838

theorem equation_equivalence : ∀ x : ℝ, (2 * (x + 1) = x + 7) ↔ (x = 5) := by sorry

end equation_equivalence_l298_29838


namespace odd_power_difference_divisibility_l298_29869

theorem odd_power_difference_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end odd_power_difference_divisibility_l298_29869


namespace geometric_sequence_3_pow_l298_29859

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem geometric_sequence_3_pow (a : ℕ → ℝ) :
  (∀ n, a n = 3^n) →
  geometric_sequence a ∧
  (∀ n, a (n + 1) > a n) ∧
  a 5^2 = a 10 ∧
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1) :=
by sorry

end geometric_sequence_3_pow_l298_29859


namespace triangle_area_l298_29853

/-- Given a triangle with perimeter 28 cm and inradius 2.0 cm, its area is 28 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2 → area = inradius * (perimeter / 2) → area = 28 := by
  sorry

end triangle_area_l298_29853


namespace det_related_matrix_l298_29845

/-- Given a 2x2 matrix with determinant 4, prove that the determinant of a related matrix is 12 -/
theorem det_related_matrix (a b c d : ℝ) (h : a * d - b * c = 4) :
  a * (7 * c + 3 * d) - c * (7 * a + 3 * b) = 12 := by
  sorry


end det_related_matrix_l298_29845


namespace smallest_three_digit_middle_ring_l298_29857

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- Checks if a number can be expressed as a product of numbers from 1 to 26 -/
def is_expressible (n : ℕ) : Prop := ∃ (factors : List ℕ), (factors.all (λ x => 1 ≤ x ∧ x ≤ 26)) ∧ (factors.prod = n)

/-- The smallest three-digit middle ring number -/
def smallest_middle_ring : ℕ := 106

theorem smallest_three_digit_middle_ring :
  is_composite smallest_middle_ring ∧
  ¬(is_expressible smallest_middle_ring) ∧
  ∀ n < smallest_middle_ring, n ≥ 100 → is_composite n → is_expressible n :=
by sorry

end smallest_three_digit_middle_ring_l298_29857


namespace parakeets_per_cage_l298_29826

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 4)
  (h2 : parrots_per_cage = 8)
  (h3 : total_birds = 40)
  : (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end parakeets_per_cage_l298_29826


namespace triangle_equation_implies_right_triangle_l298_29856

/-- A triangle with side lengths satisfying a certain equation is a right triangle -/
theorem triangle_equation_implies_right_triangle 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  a^2 + b^2 = c^2 := by
  sorry

#check triangle_equation_implies_right_triangle

end triangle_equation_implies_right_triangle_l298_29856


namespace slope_implies_y_value_l298_29849

/-- Given two points A(4, y) and B(2, -3), if the slope of the line passing through these points is π/4, then y = -1 -/
theorem slope_implies_y_value (y : ℝ) :
  let A : ℝ × ℝ := (4, y)
  let B : ℝ × ℝ := (2, -3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = π / 4 → y = -1 := by
  sorry

end slope_implies_y_value_l298_29849


namespace carnation_count_l298_29884

theorem carnation_count (total_flowers : ℕ) (roses : ℕ) (carnations : ℕ) : 
  total_flowers = 10 → roses = 5 → total_flowers = roses + carnations → carnations = 5 := by
  sorry

end carnation_count_l298_29884


namespace determinant_scaling_l298_29813

theorem determinant_scaling {x y z w : ℝ} (h : Matrix.det !![x, y; z, w] = 3) :
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 27 := by sorry

end determinant_scaling_l298_29813


namespace simplify_expression_l298_29825

theorem simplify_expression (a b : ℝ) : a * b^2 * (-2 * a^3 * b) = -2 * a^4 * b^3 := by
  sorry

end simplify_expression_l298_29825


namespace smallest_n_with_property_l298_29810

def has_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n - 2) ⊔ {3, 4} → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_property :
  (∀ k < 243, ¬ has_property k) ∧ has_property 243 :=
sorry

end smallest_n_with_property_l298_29810


namespace g_value_at_3056_l298_29802

theorem g_value_at_3056 (g : ℝ → ℝ) 
  (h1 : ∀ x > 0, g x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → g (x - y) = Real.sqrt (g (x * y) + 4))
  (h3 : ∃ x y, x > y ∧ y > 0 ∧ x - y = x * y ∧ x * y = 3056) :
  g 3056 = 2 := by
sorry

end g_value_at_3056_l298_29802


namespace exists_x0_implies_a_value_l298_29898

noncomputable section

def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -1 - Real.log 2 := by
  sorry

end

end exists_x0_implies_a_value_l298_29898


namespace min_separating_edges_l298_29831

/-- Represents a color in the grid -/
inductive Color
| Red
| Green
| Blue

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Fin 33)
  (col : Fin 33)
  (color : Color)

/-- Represents the grid -/
def Grid := Array (Array Cell)

/-- Checks if two cells are adjacent -/
def isAdjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c1.col.val = c2.col.val + 1)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c1.row.val = c2.row.val + 1))

/-- Counts the number of separating edges in the grid -/
def countSeparatingEdges (grid : Grid) : Nat :=
  sorry

/-- Checks if the grid has an equal number of cells for each color -/
def hasEqualColorDistribution (grid : Grid) : Prop :=
  sorry

/-- Theorem: The minimum number of separating edges in a 33x33 grid with three equally distributed colors is 56 -/
theorem min_separating_edges (grid : Grid) 
  (h : hasEqualColorDistribution grid) : 
  countSeparatingEdges grid ≥ 56 := by
  sorry

end min_separating_edges_l298_29831


namespace minimum_dimes_needed_l298_29851

/-- The cost of the jacket in cents -/
def jacket_cost : ℕ := 4550

/-- The value of two $20 bills in cents -/
def bills_value : ℕ := 2 * 2000

/-- The value of five quarters in cents -/
def quarters_value : ℕ := 5 * 25

/-- The value of six nickels in cents -/
def nickels_value : ℕ := 6 * 5

/-- The value of one dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed -/
def min_dimes : ℕ := 40

theorem minimum_dimes_needed :
  ∀ n : ℕ, 
    n ≥ min_dimes → 
    bills_value + quarters_value + nickels_value + n * dime_value ≥ jacket_cost ∧
    ∀ m : ℕ, m < min_dimes → 
      bills_value + quarters_value + nickels_value + m * dime_value < jacket_cost :=
by sorry

end minimum_dimes_needed_l298_29851


namespace decimal_point_problem_l298_29868

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 3 / Real.sqrt 1000 := by
sorry

end decimal_point_problem_l298_29868


namespace complex_difference_magnitude_l298_29870

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 := by
  sorry

end complex_difference_magnitude_l298_29870


namespace max_sum_with_product_2310_l298_29895

theorem max_sum_with_product_2310 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 48 :=
by sorry

end max_sum_with_product_2310_l298_29895


namespace existence_implies_bound_l298_29800

theorem existence_implies_bound :
  (∃ (m : ℝ), ∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) →
  (∀ (m : ℝ), (∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) → m ≤ -2) :=
by sorry

end existence_implies_bound_l298_29800


namespace smallest_sum_of_reciprocals_l298_29806

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x + y : ℕ) ≤ (a + b : ℕ)) ∧ (x + y : ℕ) = 49 :=
sorry

end smallest_sum_of_reciprocals_l298_29806


namespace no_real_solution_for_log_equation_l298_29835

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), 
    (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 7*x - 18)) ∧ 
    (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 7*x - 18 > 0) :=
by sorry

end no_real_solution_for_log_equation_l298_29835


namespace polygon_angles_l298_29848

theorem polygon_angles (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 + (360 / n) = 1500 → n = 10 := by
  sorry

end polygon_angles_l298_29848


namespace cupcake_packages_l298_29887

/-- Given the initial number of cupcakes, the number eaten, and the number of cupcakes per package,
    calculate the number of full packages that can be made. -/
def fullPackages (initial : ℕ) (eaten : ℕ) (perPackage : ℕ) : ℕ :=
  (initial - eaten) / perPackage

/-- Theorem stating that with 60 initial cupcakes, 22 eaten, and 10 cupcakes per package,
    the number of full packages is 3. -/
theorem cupcake_packages : fullPackages 60 22 10 = 3 := by
  sorry

end cupcake_packages_l298_29887


namespace lottery_winnings_l298_29864

/-- Calculates the total money won in a lottery given the number of tickets, winning numbers per ticket, and value per winning number. -/
def total_money_won (num_tickets : ℕ) (winning_numbers_per_ticket : ℕ) (value_per_winning_number : ℕ) : ℕ :=
  num_tickets * winning_numbers_per_ticket * value_per_winning_number

/-- Proves that with 3 lottery tickets, 5 winning numbers per ticket, and $20 per winning number, the total money won is $300. -/
theorem lottery_winnings :
  total_money_won 3 5 20 = 300 := by
  sorry

#eval total_money_won 3 5 20

end lottery_winnings_l298_29864


namespace polynomial_roots_l298_29876

/-- The polynomial x^3 + x^2 - 4x - 2 --/
def f (x : ℂ) : ℂ := x^3 + x^2 - 4*x - 2

/-- The roots of the polynomial --/
def roots : List ℂ := [1, -1 + Complex.I, -1 - Complex.I]

theorem polynomial_roots :
  ∀ r ∈ roots, f r = 0 ∧ (∀ z : ℂ, f z = 0 → z ∈ roots) :=
sorry

end polynomial_roots_l298_29876


namespace at_least_one_not_less_than_two_l298_29888

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l298_29888


namespace geometric_arithmetic_sequence_sum_l298_29886

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4 * y)^2 = 15 * x * z)  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)   -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end geometric_arithmetic_sequence_sum_l298_29886


namespace solution_satisfies_conditions_l298_29862

/-- Represents a 5x6 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 6) ℕ

/-- Checks if a row in the grid has no repeating numbers -/
def rowNoRepeats (g : Grid) (row : Fin 5) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g row i ≠ g row j

/-- Checks if a column in the grid has no repeating numbers -/
def colNoRepeats (g : Grid) (col : Fin 6) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i col ≠ g j col

/-- Checks if all numbers in the grid are between 1 and 6 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

/-- Checks if the sums of specific digits match the given constraints -/
def validSums (g : Grid) : Prop :=
  g 0 0 * 100 + g 0 1 * 10 + g 0 2 = 669 ∧
  g 0 3 * 10 + g 0 4 = 44

/-- The main theorem stating that 41244 satisfies all conditions -/
theorem solution_satisfies_conditions : ∃ (g : Grid),
  (∀ row : Fin 5, rowNoRepeats g row) ∧
  (∀ col : Fin 6, colNoRepeats g col) ∧
  validNumbers g ∧
  validSums g ∧
  g 0 0 = 4 ∧ g 0 1 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 4 ∧ g 0 4 = 4 :=
sorry

end solution_satisfies_conditions_l298_29862


namespace parabola_kite_sum_l298_29882

/-- The sum of coefficients for two parabolas forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- Parabola 1 intersects x-axis
    a * x₁^2 - 4 = 0 ∧ 
    a * x₂^2 - 4 = 0 ∧ 
    x₁ ≠ x₂ ∧
    -- Parabola 2 intersects x-axis
    6 - b * x₁^2 = 0 ∧ 
    6 - b * x₂^2 = 0 ∧
    -- Parabolas intersect y-axis
    y₁ = -4 ∧
    y₂ = 6 ∧
    -- Area of kite formed by intersection points
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 16) →
  a + b = 3.9 := by
sorry

end parabola_kite_sum_l298_29882


namespace absolute_value_integral_l298_29863

theorem absolute_value_integral : ∫ x in (0:ℝ)..4, |x - 2| = 4 := by
  sorry

end absolute_value_integral_l298_29863


namespace tile_border_ratio_l298_29897

theorem tile_border_ratio (s d : ℝ) (h1 : s > 0) (h2 : d > 0) : 
  (25 * s)^2 / ((25 * s + 2 * d)^2) = 0.81 → d / s = 1 / 18 := by
  sorry

end tile_border_ratio_l298_29897


namespace equation_solution_l298_29820

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 := by
  sorry

end equation_solution_l298_29820
