import Mathlib

namespace angle_measure_proof_l363_36379

theorem angle_measure_proof (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end angle_measure_proof_l363_36379


namespace simplify_expression_simplify_and_evaluate_l363_36312

-- Part 1
theorem simplify_expression (x y : ℝ) : 3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = -x * y := by
  sorry

-- Part 2
theorem simplify_and_evaluate (a b : ℝ) (h1 : a = 2) (h2 : b = -3) : 
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 := by
  sorry

end simplify_expression_simplify_and_evaluate_l363_36312


namespace susan_money_left_l363_36362

def susan_problem (swimming_income babysitting_income : ℝ) 
  (clothes_percentage books_percentage gifts_percentage : ℝ) : ℝ :=
  let total_income := swimming_income + babysitting_income
  let after_clothes := total_income * (1 - clothes_percentage)
  let after_books := after_clothes * (1 - books_percentage)
  let final_amount := after_books * (1 - gifts_percentage)
  final_amount

theorem susan_money_left : 
  susan_problem 1200 600 0.4 0.25 0.15 = 688.5 := by
  sorry

end susan_money_left_l363_36362


namespace min_throws_for_repeated_sum_min_throws_is_22_l363_36352

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of distinct possible sums -/
def distinct_sums : ℕ := max_sum - min_sum + 1

/-- 
The minimum number of throws required to guarantee a repeated sum 
when rolling four fair six-sided dice
-/
theorem min_throws_for_repeated_sum : ℕ := distinct_sums + 1

/-- The main theorem to prove -/
theorem min_throws_is_22 : min_throws_for_repeated_sum = 22 := by sorry

end min_throws_for_repeated_sum_min_throws_is_22_l363_36352


namespace dogs_not_doing_anything_l363_36316

def total_dogs : ℕ := 500

def running_dogs : ℕ := (18 * total_dogs) / 100
def playing_dogs : ℕ := (3 * total_dogs) / 20
def barking_dogs : ℕ := (7 * total_dogs) / 100
def digging_dogs : ℕ := total_dogs / 10
def agility_dogs : ℕ := 12
def sleeping_dogs : ℕ := (2 * total_dogs) / 25
def eating_dogs : ℕ := total_dogs / 5

def dogs_doing_something : ℕ := 
  running_dogs + playing_dogs + barking_dogs + digging_dogs + 
  agility_dogs + sleeping_dogs + eating_dogs

theorem dogs_not_doing_anything : 
  total_dogs - dogs_doing_something = 98 := by sorry

end dogs_not_doing_anything_l363_36316


namespace f_is_odd_g_is_even_l363_36384

-- Define the functions
def f (x : ℝ) : ℝ := x + x^3 + x^5
def g (x : ℝ) : ℝ := x^2 + 1

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statements
theorem f_is_odd : IsOdd f := by sorry

theorem g_is_even : IsEven g := by sorry

end f_is_odd_g_is_even_l363_36384


namespace coin_stacking_arrangements_l363_36318

/-- Represents the number of ways to arrange n indistinguishable objects in k positions -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Represents the number of valid orientation arrangements -/
def validOrientations : ℕ := 9

/-- Represents the total number of gold coins -/
def goldCoins : ℕ := 5

/-- Represents the total number of silver coins -/
def silverCoins : ℕ := 3

/-- Represents the total number of coins -/
def totalCoins : ℕ := goldCoins + silverCoins

/-- The number of distinguishable arrangements for stacking coins -/
def distinguishableArrangements : ℕ := 
  binomial totalCoins goldCoins * validOrientations

theorem coin_stacking_arrangements :
  distinguishableArrangements = 504 := by sorry

end coin_stacking_arrangements_l363_36318


namespace special_triangle_properties_l363_36348

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the specified conditions. -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧ t.c = Real.sqrt 7 ∧ Real.cos t.A + (1/2) * t.a = t.b

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.C = π/3 ∧ (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

#check special_triangle_properties

end special_triangle_properties_l363_36348


namespace team_composition_proof_l363_36344

theorem team_composition_proof (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  (22 * x + 47 * y) / (x + y) = 41 → x / (x + y) = 6 / 25 :=
by
  sorry

end team_composition_proof_l363_36344


namespace min_value_inequality_l363_36301

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end min_value_inequality_l363_36301


namespace complex_number_problem_l363_36392

theorem complex_number_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z / (-i) = 1 + 2*i → z = 2 - i := by
  sorry

end complex_number_problem_l363_36392


namespace average_temperature_proof_l363_36390

theorem average_temperature_proof (tuesday wednesday thursday friday : ℝ) : 
  (tuesday + wednesday + thursday) / 3 = 32 →
  friday = 44 →
  tuesday = 38 →
  (wednesday + thursday + friday) / 3 = 34 := by
sorry

end average_temperature_proof_l363_36390


namespace car_speed_problem_l363_36327

/-- Proves that if a car traveling at 94.73684210526315 km/h takes 2 seconds longer to travel 1 kilometer
    compared to a certain faster speed, then that faster speed is 90 km/h. -/
theorem car_speed_problem (current_speed : ℝ) (faster_speed : ℝ) : 
  current_speed = 94.73684210526315 →
  (1 / current_speed) * 3600 = (1 / faster_speed) * 3600 + 2 →
  faster_speed = 90 := by
  sorry

end car_speed_problem_l363_36327


namespace silverware_probability_l363_36323

theorem silverware_probability (forks spoons knives : ℕ) (h1 : forks = 6) (h2 : spoons = 6) (h3 : knives = 6) :
  let total := forks + spoons + knives
  let ways_to_choose_three := Nat.choose total 3
  let ways_to_choose_one_each := forks * spoons * knives
  (ways_to_choose_one_each : ℚ) / ways_to_choose_three = 9 / 34 :=
by
  sorry

end silverware_probability_l363_36323


namespace mary_shopping_total_l363_36313

def store1_total : ℚ := 13.04 + 12.27
def store2_total : ℚ := 44.15 + 25.50
def store3_total : ℚ := 2 * 9.99 * (1 - 0.1)
def store4_total : ℚ := 30.93 + 7.42
def store5_total : ℚ := 20.75 * (1 + 0.05)

def total_spent : ℚ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_shopping_total :
  total_spent = 173.08 := by sorry

end mary_shopping_total_l363_36313


namespace salt_solution_problem_l363_36311

/-- Given a solution with initial volume and salt percentage, prove that
    adding water to reach a specific salt percentage results in the correct
    initial salt percentage. -/
theorem salt_solution_problem (initial_volume : ℝ) (water_added : ℝ) 
    (final_salt_percentage : ℝ) (initial_salt_percentage : ℝ) : 
    initial_volume = 64 →
    water_added = 16 →
    final_salt_percentage = 0.08 →
    initial_salt_percentage * initial_volume = 
      final_salt_percentage * (initial_volume + water_added) →
    initial_salt_percentage = 0.1 := by
  sorry

#check salt_solution_problem

end salt_solution_problem_l363_36311


namespace stock_worth_calculation_l363_36361

theorem stock_worth_calculation (W : ℝ) 
  (h1 : 0.02 * W - 0.024 * W = -400) : W = 100000 := by
  sorry

end stock_worth_calculation_l363_36361


namespace inverse_of_A_l363_36309

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by sorry

end inverse_of_A_l363_36309


namespace cereal_box_bowls_l363_36371

/-- Given a cereal box with the following properties:
  * Each spoonful contains 4 clusters of oats
  * Each bowl has 25 spoonfuls of cereal
  * Each box contains 500 clusters of oats
  Prove that there are 5 bowlfuls of cereal in each box. -/
theorem cereal_box_bowls (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (clusters_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : clusters_per_box = 500) :
  clusters_per_box / (clusters_per_spoon * spoons_per_bowl) = 5 := by
  sorry

end cereal_box_bowls_l363_36371


namespace sour_count_theorem_l363_36369

/-- Represents the number of sours of each type -/
structure SourCounts where
  cherry : ℕ
  lemon : ℕ
  orange : ℕ
  grape : ℕ

/-- Calculates the total number of sours -/
def total_sours (counts : SourCounts) : ℕ :=
  counts.cherry + counts.lemon + counts.orange + counts.grape

/-- Represents the ratio between two quantities -/
structure Ratio where
  num : ℕ
  denom : ℕ

theorem sour_count_theorem (counts : SourCounts) 
  (cherry_lemon_ratio : Ratio) (lemon_grape_ratio : Ratio) :
  counts.cherry = 32 →
  cherry_lemon_ratio = Ratio.mk 4 5 →
  counts.cherry * cherry_lemon_ratio.denom = counts.lemon * cherry_lemon_ratio.num →
  4 * (counts.cherry + counts.lemon + counts.orange) = 3 * (counts.cherry + counts.lemon) →
  lemon_grape_ratio = Ratio.mk 3 2 →
  counts.lemon * lemon_grape_ratio.denom = counts.grape * lemon_grape_ratio.num →
  total_sours counts = 123 := by
  sorry

#check sour_count_theorem

end sour_count_theorem_l363_36369


namespace coin_count_proof_l363_36398

/-- Represents the number of nickels -/
def n : ℕ := 7

/-- Represents the number of dimes -/
def d : ℕ := 3 * n

/-- Represents the number of quarters -/
def q : ℕ := 9 * n

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := 1820

theorem coin_count_proof :
  (n * nickel_value + d * dime_value + q * quarter_value = total_value) →
  (n + d + q = 91) := by
  sorry

end coin_count_proof_l363_36398


namespace saras_money_theorem_l363_36336

/-- Calculates Sara's remaining money after all expenses --/
def saras_remaining_money (hours_per_week : ℕ) (weeks : ℕ) (hourly_rate : ℚ) 
  (tax_rate : ℚ) (insurance_fee : ℚ) (misc_fee : ℚ) (tire_cost : ℚ) : ℚ :=
  let gross_pay := hours_per_week * weeks * hourly_rate
  let taxes := tax_rate * gross_pay
  let net_pay := gross_pay - taxes - insurance_fee - misc_fee - tire_cost
  net_pay

/-- Theorem stating that Sara's remaining money is $292 --/
theorem saras_money_theorem : 
  saras_remaining_money 40 2 (11.5) (0.15) 60 20 410 = 292 := by
  sorry

end saras_money_theorem_l363_36336


namespace power_sum_theorem_l363_36305

theorem power_sum_theorem (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d ∧ a^3 + b^3 = c^3 + d^3 ∧ a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end power_sum_theorem_l363_36305


namespace min_value_squared_sum_l363_36382

theorem min_value_squared_sum (x y z : ℝ) (h : 2*x + 3*y + z = 7) :
  x^2 + y^2 + z^2 ≥ 7/2 :=
by
  sorry

end min_value_squared_sum_l363_36382


namespace length_AB_with_equal_quarter_circles_l363_36387

/-- The length of AB given two circles with equal quarter-circle areas --/
theorem length_AB_with_equal_quarter_circles 
  (r : ℝ) 
  (h_r : r = 4)
  (π_approx : ℝ) 
  (h_π : π_approx = 3) : 
  let quarter_circle_area := (1/4) * π_approx * r^2
  let total_shaded_area := 2 * quarter_circle_area
  let AB := total_shaded_area / (2 * r)
  AB = 6 := by sorry

end length_AB_with_equal_quarter_circles_l363_36387


namespace stationery_prices_l363_36350

theorem stationery_prices (pen_price notebook_price : ℝ) : 
  pen_price + notebook_price = 3.6 →
  pen_price + 4 * notebook_price = 10.5 →
  pen_price = 1.3 ∧ notebook_price = 2.3 := by
sorry

end stationery_prices_l363_36350


namespace decimal_to_binary_conversion_l363_36322

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The decimal number to be converted -/
def decimal_number : ℕ := 2016

/-- The expected binary representation -/
def expected_binary : List Bool := [true, true, true, true, true, false, false, false, false, false, false]

theorem decimal_to_binary_conversion :
  to_binary decimal_number = expected_binary := by
  sorry

end decimal_to_binary_conversion_l363_36322


namespace min_value_theorem_l363_36383

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end min_value_theorem_l363_36383


namespace four_digit_addition_l363_36333

/-- Given four different natural numbers A, B, C, and D that satisfy the equation
    4A5B + C2D7 = 7070, prove that C = 2. -/
theorem four_digit_addition (A B C D : ℕ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq : 4000 * A + 50 * B + 1000 * C + 200 * D + 7 = 7070) : 
  C = 2 := by
  sorry

end four_digit_addition_l363_36333


namespace profit_calculation_l363_36359

theorem profit_calculation (cost1 cost2 cost3 : ℝ) (profit_percentage : ℝ) :
  cost1 = 200 →
  cost2 = 300 →
  cost3 = 500 →
  profit_percentage = 0.1 →
  let total_cost := cost1 + cost2 + cost3
  let total_selling_price := total_cost + total_cost * profit_percentage
  total_selling_price = 1100 := by
  sorry

end profit_calculation_l363_36359


namespace expression_equals_negative_one_l363_36332

theorem expression_equals_negative_one
  (a b y : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ a)
  (hy1 : y ≠ a)
  (hy2 : y ≠ -a) :
  (((a + b) / (a + y) + y / (a - y)) /
   ((y + b) / (a + y) - a / (a - y)) = -1) ↔
  (y = a - b) :=
sorry

end expression_equals_negative_one_l363_36332


namespace largest_number_l363_36356

theorem largest_number : Real.sqrt 2 = max (max (max (-3) 0) 1) (Real.sqrt 2) := by
  sorry

end largest_number_l363_36356


namespace extremal_point_and_range_l363_36342

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 * Real.log x

theorem extremal_point_and_range (e : ℝ) (h_e : Real.exp 1 = e) :
  (∃ a : ℝ, (deriv (f a)) e = 0 ↔ (a = e ∨ a = 3*e)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Ioc 0 (3*e) → f a x ≤ 4*e^2) ↔ 
    a ∈ Set.Icc (3*e - 2*e / Real.sqrt (Real.log (3*e))) (3*e)) :=
by sorry

end extremal_point_and_range_l363_36342


namespace q_satisfies_conditions_l363_36395

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℚ) : ℚ := (20/7) * x^2 + (40/7) * x - 300/7

/-- Proof that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -20 :=
by sorry


end q_satisfies_conditions_l363_36395


namespace min_value_is_neg_one_l363_36363

/-- The system of equations and inequalities -/
def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

/-- The expression to be minimized -/
def expression (x y : ℝ) : ℝ := x^3 + y^3

/-- The theorem stating that the minimum value of the expression is -1 -/
theorem min_value_is_neg_one :
  ∃ (x y : ℝ), system x y ∧
  ∀ (a b : ℝ), system a b → expression x y ≤ expression a b ∧
  expression x y = -1 := by
  sorry

end min_value_is_neg_one_l363_36363


namespace marlas_errand_time_l363_36397

/-- The total time Marla spends on her errand activities -/
def total_time (driving_time grocery_time gas_time parent_teacher_time coffee_time : ℕ) : ℕ :=
  2 * driving_time + grocery_time + gas_time + parent_teacher_time + coffee_time

/-- Theorem stating the total time Marla spends on her errand activities -/
theorem marlas_errand_time : 
  total_time 20 15 5 70 30 = 160 := by sorry

end marlas_errand_time_l363_36397


namespace abs_eq_sqrt_sq_l363_36335

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end abs_eq_sqrt_sq_l363_36335


namespace paper_pieces_difference_paper_pieces_problem_l363_36399

theorem paper_pieces_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_squares initial_corners final_pieces final_corners corner_difference =>
    initial_squares * 4 = initial_corners →
    ∃ (triangles pentagonals : ℕ),
      triangles + pentagonals = final_pieces ∧
      3 * triangles + 5 * pentagonals = final_corners ∧
      triangles - pentagonals = corner_difference

theorem paper_pieces_problem :
  paper_pieces_difference 25 100 50 170 30 := by
  sorry

end paper_pieces_difference_paper_pieces_problem_l363_36399


namespace classify_books_count_l363_36385

/-- The number of ways to classify 6 distinct books into two groups -/
def classify_books : ℕ :=
  let total_books : ℕ := 6
  let intersection_size : ℕ := 3
  let remaining_books : ℕ := total_books - intersection_size
  let ways_to_choose_intersection : ℕ := Nat.choose total_books intersection_size
  let ways_to_distribute_remaining : ℕ := 3^remaining_books
  (ways_to_choose_intersection * ways_to_distribute_remaining) / 2

/-- Theorem stating that the number of ways to classify the books is 270 -/
theorem classify_books_count : classify_books = 270 := by
  sorry

end classify_books_count_l363_36385


namespace equation_solutions_l363_36353

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3 * x - 2|

-- Define the domain of f
def domain_f (x : ℝ) : Prop := x ≠ 3 ∧ x ≠ 0

-- Define the equation to be solved
def equation (x a : ℝ) : Prop := |3 * x - 2| = |x + a|

-- Theorem statement
theorem equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧
  (∀ x, domain_f x → equation x a₁) ∧
  (∀ x, domain_f x → equation x a₂) ∧
  (∀ a, (∀ x, domain_f x → equation x a) → (a = a₁ ∨ a = a₂)) ∧
  a₁ = -2/3 ∧ a₂ = 2 :=
sorry

end equation_solutions_l363_36353


namespace fraction_equation_solution_l363_36321

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end fraction_equation_solution_l363_36321


namespace car_dealer_problem_l363_36341

theorem car_dealer_problem (X Y : ℚ) (h1 : X > 0) (h2 : Y > 0) : 
  1.54 * (X + Y) = 1.4 * X + 1.6 * Y →
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ X * b = Y * a ∧ Nat.gcd a b = 1 ∧ 11 * a + 13 * b = 124 :=
by sorry

end car_dealer_problem_l363_36341


namespace marathon_time_l363_36315

theorem marathon_time (dean_time : ℝ) 
  (h1 : dean_time > 0)
  (h2 : dean_time * (2/3) * (1 + 1/3) + dean_time * (3/2) + dean_time = 23) : 
  dean_time = 23/3 := by
sorry

end marathon_time_l363_36315


namespace special_trapezoid_not_isosceles_l363_36354

/-- A trapezoid with the given properties --/
structure SpecialTrapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  is_trapezoid : base1 ≠ base2
  base_values : base1 = 3 ∧ base2 = 4
  diagonal_length : diagonal = 6
  diagonal_bisects_angle : Bool

/-- Theorem stating that a trapezoid with the given properties cannot be isosceles --/
theorem special_trapezoid_not_isosceles (t : SpecialTrapezoid) : 
  ¬(∃ (side : ℝ), side > 0 ∧ t.base1 < t.base2 → 
    (side = t.diagonal ∧ side^2 = (t.base2 - t.base1)^2 / 4 + side^2 / 4)) := by
  sorry

end special_trapezoid_not_isosceles_l363_36354


namespace ceiling_plus_x_eq_two_x_l363_36326

theorem ceiling_plus_x_eq_two_x (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ + x = 2 * x := by
  sorry

end ceiling_plus_x_eq_two_x_l363_36326


namespace min_sum_of_quadratic_roots_l363_36329

theorem min_sum_of_quadratic_roots (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0) → 
  (∃ x : ℝ, x^2 + 2*n*x + m = 0) → 
  m + n ≥ 6 := by sorry

end min_sum_of_quadratic_roots_l363_36329


namespace simplify_nested_roots_l363_36302

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/12))^(1/4))^6 * (((a^16)^(1/4))^(1/12))^3 = a^3 := by sorry

end simplify_nested_roots_l363_36302


namespace three_digit_divisibility_l363_36386

theorem three_digit_divisibility (a b c : ℕ) (h : ∃ k : ℕ, 100 * a + 10 * b + c = 27 * k ∨ 100 * a + 10 * b + c = 37 * k) :
  ∃ m : ℕ, 100 * b + 10 * c + a = 27 * m ∨ 100 * b + 10 * c + a = 37 * m := by
sorry

end three_digit_divisibility_l363_36386


namespace min_lines_inequality_l363_36377

/-- Represents the minimum number of lines required to compute a function using only disjunctions and conjunctions -/
def M (n : ℕ) : ℕ := sorry

/-- The theorem states that for n ≥ 4, the minimum number of lines to compute f_n is at least 3 more than the minimum number of lines to compute f_(n-2) -/
theorem min_lines_inequality (n : ℕ) (h : n ≥ 4) : M n ≥ M (n - 2) + 3 := by
  sorry

end min_lines_inequality_l363_36377


namespace intersection_line_equation_l363_36368

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    line x1 y1 ∧ line x2 y2 :=
by
  sorry

end intersection_line_equation_l363_36368


namespace function_identity_l363_36364

theorem function_identity (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) :
  ∀ n : ℕ, f n = n := by
  sorry

end function_identity_l363_36364


namespace multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l363_36389

theorem multiple_of_24 : ∃ (n : ℕ), 600 = 24 * n := by
  sorry

theorem multiple_of_3_and_8 (x : ℕ) : x % 24 = 0 ↔ x % 3 = 0 ∧ x % 8 = 0 := by
  sorry

theorem six_hundred_is_multiple_of_24 : 600 % 24 = 0 := by
  sorry

end multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l363_36389


namespace roots_product_plus_one_l363_36381

theorem roots_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  (1+p)*(1+q)*(1+r) = 51 := by
sorry

end roots_product_plus_one_l363_36381


namespace arithmetic_sequence_common_difference_l363_36310

/-- An arithmetic sequence with a_4 = 3 and a_12 = 19 has a common difference of 2 -/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence condition
  a 4 = 3 →
  a 12 = 19 →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l363_36310


namespace abs_negative_six_l363_36328

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by
  sorry

end abs_negative_six_l363_36328


namespace johann_oranges_l363_36367

def orange_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_count := remaining_after_eating - stolen + returned_oranges
  final_count = 30

theorem johann_oranges :
  orange_problem 60 10 2 5 := by sorry

end johann_oranges_l363_36367


namespace square_root_sum_l363_36340

theorem square_root_sum (y : ℝ) : 
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 →
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end square_root_sum_l363_36340


namespace product_of_repeating_decimals_l363_36366

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem product_of_repeating_decimals :
  repeating_decimal_12 * repeating_decimal_34 = 136 / 3267 :=
by sorry

end product_of_repeating_decimals_l363_36366


namespace smallest_angle_triangle_range_l363_36334

theorem smallest_angle_triangle_range (x : Real) : 
  (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4)) →
  (0 < x ∧ x ≤ π/3) →
  ∃ (a b : Real), a = 1 ∧ b = Real.sqrt 2 ∧ 
    (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4) → a < y ∧ y ≤ b) ∧
    (∀ z : Real, a < z ∧ z ≤ b → ∃ x₀ : Real, 0 < x₀ ∧ x₀ ≤ π/3 ∧ z = Real.sqrt 2 * Real.sin (x₀ + π/4)) :=
by sorry

end smallest_angle_triangle_range_l363_36334


namespace solve_equation_l363_36337

/-- Given an equation 19(x + y) + z = 19(-x + y) - 21 where x = 1, prove that z = -59 -/
theorem solve_equation (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end solve_equation_l363_36337


namespace min_omega_for_cos_symmetry_l363_36355

theorem min_omega_for_cos_symmetry (ω : ℕ+) : 
  (∃ k : ℤ, ω = 6 * k + 2) → 
  (∀ ω' : ℕ+, (∃ k' : ℤ, ω' = 6 * k' + 2) → ω ≤ ω') → 
  ω = 2 := by sorry

end min_omega_for_cos_symmetry_l363_36355


namespace deal_or_no_deal_probability_l363_36351

/-- Represents the game setup with total boxes and high-value boxes -/
structure GameSetup where
  total_boxes : ℕ
  high_value_boxes : ℕ

/-- Calculates the probability of holding a high-value box -/
def probability_high_value (g : GameSetup) (eliminated : ℕ) : ℚ :=
  g.high_value_boxes / (g.total_boxes - eliminated)

/-- Theorem stating that eliminating 7 boxes results in at least 50% chance of high-value box -/
theorem deal_or_no_deal_probability 
  (g : GameSetup) 
  (h1 : g.total_boxes = 30) 
  (h2 : g.high_value_boxes = 8) : 
  probability_high_value g 7 ≥ 1/2 := by
  sorry

#eval probability_high_value ⟨30, 8⟩ 7

end deal_or_no_deal_probability_l363_36351


namespace largest_angle_of_triangle_l363_36357

/-- Given a triangle XYZ with sides x, y, and z satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (x y z : ℝ) (h1 : x + 3*y + 4*z = x^2) (h2 : x + 3*y - 4*z = -7) :
  ∃ (X Y Z : ℝ), X + Y + Z = 180 ∧ 0 < X ∧ 0 < Y ∧ 0 < Z ∧ max X (max Y Z) = 120 := by
  sorry

end largest_angle_of_triangle_l363_36357


namespace consecutive_color_draw_probability_l363_36300

def num_green_chips : ℕ := 4
def num_blue_chips : ℕ := 3
def num_red_chips : ℕ := 5
def total_chips : ℕ := num_green_chips + num_blue_chips + num_red_chips

theorem consecutive_color_draw_probability :
  (Nat.factorial 3 * Nat.factorial num_green_chips * Nat.factorial num_blue_chips * Nat.factorial num_red_chips) / 
  Nat.factorial total_chips = 1 / 4620 := by
  sorry

end consecutive_color_draw_probability_l363_36300


namespace michaels_pets_l363_36319

theorem michaels_pets (total_pets : ℕ) (dog_percentage : ℚ) (cat_percentage : ℚ) :
  total_pets = 36 →
  dog_percentage = 25 / 100 →
  cat_percentage = 50 / 100 →
  ↑(total_pets : ℕ) * (1 - dog_percentage - cat_percentage) = 9 := by
  sorry

end michaels_pets_l363_36319


namespace circumradius_side_ratio_not_unique_l363_36325

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a side of a triangle -/
def side_length (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- The shape of a triangle, represented by its angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The ratio of circumradius to one side does not uniquely determine triangle shape -/
theorem circumradius_side_ratio_not_unique (r : ℝ) (side : Fin 3) :
  ∃ t1 t2 : Triangle, 
    circumradius t1 / side_length t1 side = r ∧
    circumradius t2 / side_length t2 side = r ∧
    triangle_shape t1 ≠ triangle_shape t2 := by
  sorry

end circumradius_side_ratio_not_unique_l363_36325


namespace probability_of_odd_product_l363_36375

def range_start : ℕ := 4
def range_end : ℕ := 16

def count_integers : ℕ := range_end - range_start + 1
def count_odd_integers : ℕ := (range_end - range_start + 1) / 2

def total_combinations : ℕ := count_integers.choose 3
def odd_combinations : ℕ := count_odd_integers.choose 3

theorem probability_of_odd_product :
  (odd_combinations : ℚ) / total_combinations = 10 / 143 :=
sorry

end probability_of_odd_product_l363_36375


namespace infinite_geometric_series_first_term_l363_36324

/-- For an infinite geometric series with common ratio -1/3 and sum 12, the first term is 16 -/
theorem infinite_geometric_series_first_term :
  ∀ (a : ℝ), 
    (∃ (S : ℝ), S = a / (1 - (-1/3))) →  -- Infinite geometric series formula
    (a / (1 - (-1/3)) = 12) →             -- Sum of the series is 12
    a = 16 :=                             -- First term is 16
by
  sorry

end infinite_geometric_series_first_term_l363_36324


namespace product_sum_relation_l363_36396

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end product_sum_relation_l363_36396


namespace meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l363_36388

/-- Represents the kiwi sale scenario -/
structure KiwiSale where
  purchase_price : ℝ
  initial_selling_price : ℝ
  initial_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit for a given price reduction -/
def daily_profit (ks : KiwiSale) (price_reduction : ℝ) : ℝ :=
  (ks.initial_selling_price - price_reduction - ks.purchase_price) *
  (ks.initial_sales + ks.sales_increase_rate * price_reduction)

/-- The kiwi sale scenario from the problem -/
def meixian_kiwi_sale : KiwiSale :=
  { purchase_price := 80
    initial_selling_price := 120
    initial_sales := 20
    sales_increase_rate := 2 }

theorem meixian_kiwi_profit_1200 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  daily_profit meixian_kiwi_sale x₁ = 1200 ∧
  daily_profit meixian_kiwi_sale x₂ = 1200 ∧
  (x₁ = 10 ∨ x₁ = 20) ∧ (x₂ = 10 ∨ x₂ = 20) :=
sorry

theorem meixian_kiwi_profit_1800_impossible :
  ¬∃ y : ℝ, daily_profit meixian_kiwi_sale y = 1800 :=
sorry

end meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l363_36388


namespace increase_by_percentage_l363_36374

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 700 ∧ percentage = 85 ∧ final = initial * (1 + percentage / 100) →
  final = 1295 := by
  sorry

end increase_by_percentage_l363_36374


namespace valid_number_count_l363_36360

/-- Represents a valid six-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 6 → Fin 6)
  (no_repetition : Function.Injective digits)
  (one_not_at_ends : digits 0 ≠ 1 ∧ digits 5 ≠ 1)
  (one_adjacent_even_pair : ∃! (i : Fin 5), 
    (digits i).val % 2 = 0 ∧ (digits (i + 1)).val % 2 = 0 ∧
    (digits i).val ≠ (digits (i + 1)).val)

/-- The number of valid six-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- The main theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 288 := by sorry

end valid_number_count_l363_36360


namespace greatest_whole_number_satisfying_inequality_l363_36378

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 6*x - 5 < 3 - 2*x :=
by sorry

end greatest_whole_number_satisfying_inequality_l363_36378


namespace final_amount_correct_l363_36370

/-- Represents the financial transactions in the boot-selling scenario -/
def boot_sale (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ) : Prop :=
  -- Original intended price
  original_price = 25 ∧
  -- Total collected from selling two boots
  total_collected = 2 * price_per_boot ∧
  -- Price per boot
  price_per_boot = 12.5 ∧
  -- Amount to be returned per boot
  return_amount = 2.5 ∧
  -- Cost of candy Hans bought
  candy_cost = 3 ∧
  -- Actual amount returned to each customer
  actual_return = 1

/-- The theorem stating that the final amount Karl received is correct -/
theorem final_amount_correct 
  (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ)
  (h : boot_sale original_price total_collected price_per_boot return_amount candy_cost actual_return) :
  total_collected - (2 * actual_return) = original_price - (2 * return_amount) :=
by sorry

end final_amount_correct_l363_36370


namespace translation_lori_to_alex_l363_36314

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Lori's house location --/
def lori_house : Point := ⟨6, 3⟩

/-- Alex's house location --/
def alex_house : Point := ⟨-2, -4⟩

/-- Calculates the translation between two points --/
def calculate_translation (p1 p2 : Point) : Translation :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

/-- Theorem: The translation from Lori's house to Alex's house is 8 units left and 7 units down --/
theorem translation_lori_to_alex :
  let t := calculate_translation lori_house alex_house
  t.dx = -8 ∧ t.dy = -7 := by sorry

end translation_lori_to_alex_l363_36314


namespace geometric_sequence_sum_l363_36306

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2 + a 6 = 3) →
  (a 6 + a 10 = 12) →
  (a 8 + a 12 = 24) :=
by
  sorry

end geometric_sequence_sum_l363_36306


namespace maria_car_rental_cost_l363_36304

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that Maria's car rental cost is $275 given the specified conditions. -/
theorem maria_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end maria_car_rental_cost_l363_36304


namespace problem_statement_l363_36380

theorem problem_statement (a b c : ℤ) 
  (h1 : 0 < c) (h2 : c < 90) 
  (h3 : Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * Real.sin (c * π / 180)) :
  (a + b) / c = 1/2 := by
  sorry

end problem_statement_l363_36380


namespace arctan_tan_sum_equals_angle_l363_36391

theorem arctan_tan_sum_equals_angle (θ : Real) : 
  θ ≥ 0 ∧ θ ≤ π / 2 → Real.arctan (Real.tan θ + 3 * Real.tan (π / 12)) = θ := by
  sorry

end arctan_tan_sum_equals_angle_l363_36391


namespace f_sin_cos_inequality_l363_36320

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

def f_on_interval (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 3 4, f x = x - 2

theorem f_sin_cos_inequality 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period_two f) 
  (h_interval : f_on_interval f) : 
  f (Real.sin 1) < f (Real.cos 1) := by
  sorry

end f_sin_cos_inequality_l363_36320


namespace solve_for_t_l363_36346

theorem solve_for_t (s t u : ℚ) 
  (eq1 : 12 * s + 6 * t + 3 * u = 180)
  (eq2 : t = s + 2)
  (eq3 : t = u + 3) :
  t = 213 / 21 := by
sorry

end solve_for_t_l363_36346


namespace angle_sum_theorem_l363_36372

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  α + 2*β = 2*π/3 →
  Real.tan (α/2) * Real.tan β = 2 - Real.sqrt 3 →
  α + β = 5*π/12 := by
sorry

end angle_sum_theorem_l363_36372


namespace april_rose_price_l363_36394

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  (total_earnings : ℚ) / ((initial_roses - remaining_roses) : ℚ)

theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end april_rose_price_l363_36394


namespace evening_to_morning_ratio_l363_36339

def morning_miles : ℝ := 2
def total_miles : ℝ := 12

def evening_miles : ℝ := total_miles - morning_miles

theorem evening_to_morning_ratio :
  evening_miles / morning_miles = 5 := by sorry

end evening_to_morning_ratio_l363_36339


namespace parallelepiped_diagonal_l363_36331

/-- The diagonal of a rectangular parallelepiped given its face diagonals -/
theorem parallelepiped_diagonal (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = (m^2 + n^2 + p^2) / 2 := by
  sorry

#check parallelepiped_diagonal

end parallelepiped_diagonal_l363_36331


namespace gray_area_calculation_l363_36317

theorem gray_area_calculation (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 3 * r_small →
  (π * r_large^2 - π * r_small^2) = 32 * π := by
  sorry

end gray_area_calculation_l363_36317


namespace third_side_length_l363_36303

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two sides of a triangle with lengths 1 and 3, the third side must be 3 --/
theorem third_side_length :
  ∀ x : ℝ, is_valid_triangle 1 3 x → x = 3 := by
  sorry

end third_side_length_l363_36303


namespace simplify_sqrt_sum_l363_36393

theorem simplify_sqrt_sum : 
  Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) = 
  1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 := by
  sorry

end simplify_sqrt_sum_l363_36393


namespace alissa_picked_16_flowers_l363_36376

/-- The number of flowers Alissa picked -/
def alissa_flowers : ℕ := sorry

/-- The number of flowers Melissa picked -/
def melissa_flowers : ℕ := sorry

/-- The number of flowers given to their mother -/
def flowers_to_mother : ℕ := 18

/-- The number of flowers left after giving to their mother -/
def flowers_left : ℕ := 14

theorem alissa_picked_16_flowers :
  (alissa_flowers = melissa_flowers) ∧
  (alissa_flowers + melissa_flowers = flowers_to_mother + flowers_left) ∧
  (flowers_to_mother = 18) ∧
  (flowers_left = 14) →
  alissa_flowers = 16 := by sorry

end alissa_picked_16_flowers_l363_36376


namespace polynomial_roots_l363_36338

theorem polynomial_roots (p : ℝ) (hp : p > 5/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
    x₁^4 - 2*p*x₁^3 + x₁^2 - 2*p*x₁ + 1 = 0 ∧
    x₂^4 - 2*p*x₂^3 + x₂^2 - 2*p*x₂ + 1 = 0 :=
sorry

end polynomial_roots_l363_36338


namespace lesser_number_problem_l363_36358

theorem lesser_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 612) :
  min x y = 21.395 := by sorry

end lesser_number_problem_l363_36358


namespace family_reunion_handshakes_count_l363_36330

/-- Represents the number of handshakes at a family reunion --/
def family_reunion_handshakes : ℕ :=
  let twin_sets := 7
  let triplet_sets := 4
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 3) + triplets * (twins / 4)
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 184 --/
theorem family_reunion_handshakes_count : family_reunion_handshakes = 184 := by
  sorry

end family_reunion_handshakes_count_l363_36330


namespace volume_alteration_percentage_l363_36343

def original_volume : ℝ := 20 * 15 * 12

def removed_volume : ℝ := 4 * (4 * 4 * 4)

def added_volume : ℝ := 4 * (2 * 2 * 2)

def net_volume_change : ℝ := removed_volume - added_volume

theorem volume_alteration_percentage :
  (net_volume_change / original_volume) * 100 = 6.22 := by
  sorry

end volume_alteration_percentage_l363_36343


namespace dog_walking_distance_l363_36307

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (days_per_week : ℕ) :
  total_weekly_miles = 70 ∧ 
  dog1_daily_miles = 2 ∧ 
  days_per_week = 7 →
  (total_weekly_miles - dog1_daily_miles * days_per_week) / days_per_week = 8 :=
by sorry

end dog_walking_distance_l363_36307


namespace book_pages_theorem_l363_36365

theorem book_pages_theorem (total_pages : ℕ) : 
  (total_pages / 5 : ℚ) + 24 + (3/2 : ℚ) * ((total_pages / 5 : ℚ) + 24) = (3/4 : ℚ) * total_pages →
  total_pages = 240 := by
sorry

end book_pages_theorem_l363_36365


namespace radius_of_circle_from_spherical_coords_l363_36373

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/3) is √3/2 -/
theorem radius_of_circle_from_spherical_coords :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ,
  let x : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.cos θ
  let y : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end radius_of_circle_from_spherical_coords_l363_36373


namespace diagonal_cubes_200_420_480_l363_36345

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The number of cubes an internal diagonal passes through in a 200×420×480 rectangular solid is 1000 -/
theorem diagonal_cubes_200_420_480 :
  diagonal_cubes 200 420 480 = 1000 := by
  sorry

end diagonal_cubes_200_420_480_l363_36345


namespace junior_score_l363_36349

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let class_average : ℝ := 78
  let senior_average : ℝ := 75
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := class_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end junior_score_l363_36349


namespace right_triangle_legs_l363_36347

/-- A right-angled triangle with a point inside it -/
structure RightTriangleWithPoint where
  /-- Length of one leg of the triangle -/
  x : ℝ
  /-- Length of the other leg of the triangle -/
  y : ℝ
  /-- Distance from the point to one side -/
  d1 : ℝ
  /-- Distance from the point to the other side -/
  d2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : x > 0 ∧ y > 0
  /-- The point is inside the triangle -/
  point_inside : d1 > 0 ∧ d2 > 0 ∧ d1 < y ∧ d2 < x
  /-- The area of the triangle is 100 -/
  area : x * y / 2 = 100
  /-- The distances from the point to the sides are 4 and 8 -/
  distances : d1 = 4 ∧ d2 = 8

/-- The theorem stating the possible leg lengths of the triangle -/
theorem right_triangle_legs (t : RightTriangleWithPoint) :
  (t.x = 40 ∧ t.y = 5) ∨ (t.x = 10 ∧ t.y = 20) :=
sorry

end right_triangle_legs_l363_36347


namespace complement_P_intersect_Q_l363_36308

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end complement_P_intersect_Q_l363_36308
