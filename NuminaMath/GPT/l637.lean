import Mathlib

namespace compute_expression_l637_63760

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l637_63760


namespace smallest_square_area_l637_63728

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) (h4 : d = 5) :
  ∃ s : ℕ, s * s = 64 ∧ (a + c <= s ∧ max b d <= s) ∨ (max a c <= s ∧ b + d <= s) :=
sorry

end smallest_square_area_l637_63728


namespace ratio_of_incomes_l637_63768

variable {I1 I2 E1 E2 S1 S2 : ℝ}

theorem ratio_of_incomes
  (h1 : I1 = 4000)
  (h2 : E1 / E2 = 3 / 2)
  (h3 : S1 = 1600)
  (h4 : S2 = 1600)
  (h5 : S1 = I1 - E1)
  (h6 : S2 = I2 - E2) :
  I1 / I2 = 5 / 4 :=
by
  sorry

end ratio_of_incomes_l637_63768


namespace min_tiles_l637_63743

theorem min_tiles (x y : ℕ) (h1 : 25 * x + 9 * y = 2014) (h2 : ∀ a b, 25 * a + 9 * b = 2014 -> (a + b) >= (x + y)) : x + y = 94 :=
  sorry

end min_tiles_l637_63743


namespace project_completion_in_16_days_l637_63751

noncomputable def a_work_rate : ℚ := 1 / 20
noncomputable def b_work_rate : ℚ := 1 / 30
noncomputable def c_work_rate : ℚ := 1 / 40
noncomputable def days_a_works (X: ℚ) : ℚ := X - 10
noncomputable def days_b_works (X: ℚ) : ℚ := X - 5
noncomputable def days_c_works (X: ℚ) : ℚ := X

noncomputable def total_work (X: ℚ) : ℚ :=
  (a_work_rate * days_a_works X) + (b_work_rate * days_b_works X) + (c_work_rate * days_c_works X)

theorem project_completion_in_16_days : total_work 16 = 1 := by
  sorry

end project_completion_in_16_days_l637_63751


namespace fixed_point_l637_63716

variable (p : ℝ)

def f (x : ℝ) : ℝ := 9 * x^2 + p * x - 5 * p

theorem fixed_point : ∀ c d : ℝ, (∀ p : ℝ, f p c = d) → (c = 5 ∧ d = 225) :=
by
  intro c d h
  -- This is a placeholder for the proof
  sorry

end fixed_point_l637_63716


namespace problem_abc_l637_63706

theorem problem_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by
  sorry

end problem_abc_l637_63706


namespace quadratic_root_condition_l637_63776

theorem quadratic_root_condition (k : ℝ) :
  (∀ (x : ℝ), x^2 + k * x + 4 * k^2 - 3 = 0 → ∃ x1 x2 : ℝ, x1 + x2 = (-k) ∧ x1 * x2 = 4 * k^2 - 3 ∧ x1 + x2 = x1 * x2) →
  k = 3 / 4 :=
by
  sorry

end quadratic_root_condition_l637_63776


namespace compare_abc_l637_63740

noncomputable def a : ℝ := - Real.logb 2 (1/5)
noncomputable def b : ℝ := Real.logb 8 27
noncomputable def c : ℝ := Real.exp (-3)

theorem compare_abc : a = Real.logb 2 5 ∧ 1 < b ∧ b < 2 ∧ c = Real.exp (-3) → a > b ∧ b > c :=
by
  sorry

end compare_abc_l637_63740


namespace max_red_dominated_rows_plus_blue_dominated_columns_l637_63756

-- Definitions of the problem conditions and statement
theorem max_red_dominated_rows_plus_blue_dominated_columns (m n : ℕ)
  (h1 : Odd m) (h2 : Odd n) (h3 : 0 < m ∧ 0 < n) :
  ∃ A : Finset (Fin m) × Finset (Fin n),
  (A.1.card + A.2.card = m + n - 2) :=
sorry

end max_red_dominated_rows_plus_blue_dominated_columns_l637_63756


namespace maria_earnings_l637_63764

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l637_63764


namespace product_of_digits_of_non_divisible_number_l637_63780

theorem product_of_digits_of_non_divisible_number:
  (¬ (3641 % 4 = 0)) →
  ((3641 % 10) * ((3641 / 10) % 10)) = 4 :=
by
  intro h
  sorry

end product_of_digits_of_non_divisible_number_l637_63780


namespace fraction_equivalent_to_decimal_l637_63720

theorem fraction_equivalent_to_decimal : 
  ∃ (x : ℚ), x = 0.6 + 0.0037 * (1 / (1 - 0.01)) ∧ x = 631 / 990 :=
by
  sorry

end fraction_equivalent_to_decimal_l637_63720


namespace max_value_fraction_l637_63750

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 1 = 0

theorem max_value_fraction (a b : ℝ) (H : circle_eq a b) :
  ∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1/2 ∧ b = t * (a - 3) ∧ t = 1 / 2 :=
by sorry

end max_value_fraction_l637_63750


namespace gdp_scientific_notation_l637_63788

theorem gdp_scientific_notation (gdp : ℝ) (h : gdp = 338.8 * 10^9) : gdp = 3.388 * 10^10 :=
by sorry

end gdp_scientific_notation_l637_63788


namespace fourth_term_of_geometric_progression_l637_63770

theorem fourth_term_of_geometric_progression (x : ℝ) (r : ℝ) 
  (h1 : (2 * x + 5) = r * x) 
  (h2 : (3 * x + 10) = r * (2 * x + 5)) : 
  (3 * x + 10) * r = -5 :=
by
  sorry

end fourth_term_of_geometric_progression_l637_63770


namespace part1_problem_part2_problem_l637_63724

/-- Given initial conditions and price adjustment, prove the expected number of helmets sold and the monthly profit. -/
theorem part1_problem (initial_price : ℕ) (initial_sales : ℕ) 
(price_reduction : ℕ) (sales_per_reduction : ℕ) (cost_price : ℕ) : 
  initial_price = 80 → initial_sales = 200 → price_reduction = 10 → 
  sales_per_reduction = 20 → cost_price = 50 → 
  (initial_sales + price_reduction * sales_per_reduction = 400) ∧ 
  ((initial_price - price_reduction - cost_price) * 
  (initial_sales + price_reduction * sales_per_reduction) = 8000) :=
by
  intros
  sorry

/-- Given initial conditions and profit target, prove the expected selling price of helmets. -/
theorem part2_problem (initial_price : ℕ) (initial_sales : ℕ) 
(cost_price : ℕ) (profit_target : ℕ) (x : ℕ) :
  initial_price = 80 → initial_sales = 200 → cost_price = 50 → 
  profit_target = 7500 → (x = 15) → 
  (initial_price - x = 65) :=
by
  intros
  sorry

end part1_problem_part2_problem_l637_63724


namespace length_of_A_l637_63759

structure Point := (x : ℝ) (y : ℝ)

noncomputable def length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem length_of_A'B' (A A' B B' C : Point) 
    (hA : A = ⟨0, 6⟩)
    (hB : B = ⟨0, 10⟩)
    (hC : C = ⟨3, 6⟩)
    (hA'_line : A'.y = A'.x)
    (hB'_line : B'.y = B'.x) 
    (hA'C : ∃ m b, ((C.y = m * C.x + b) ∧ (C.y = b) ∧ (A.y = b))) 
    (hB'C : ∃ m b, ((C.y = m * C.x + b) ∧ (B.y = m * B.x + b)))
    : length A' B' = (12 / 7) * Real.sqrt 2 :=
by
  sorry

end length_of_A_l637_63759


namespace cost_of_bananas_l637_63721

-- Definitions of the conditions from the problem
namespace BananasCost

variables (A B : ℝ)

-- Condition equations
def condition1 : Prop := 2 * A + B = 7
def condition2 : Prop := A + B = 5

-- The theorem to prove the cost of a bunch of bananas
theorem cost_of_bananas (h1 : condition1 A B) (h2 : condition2 A B) : B = 3 := 
  sorry

end BananasCost

end cost_of_bananas_l637_63721


namespace largest_integer_condition_l637_63799

theorem largest_integer_condition (m a b : ℤ) 
  (h1 : m < 150) 
  (h2 : m > 50) 
  (h3 : m = 9 * a - 2) 
  (h4 : m = 6 * b - 4) : 
  m = 106 := 
sorry

end largest_integer_condition_l637_63799


namespace regular_octagon_interior_angle_l637_63723

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l637_63723


namespace total_oranges_and_apples_l637_63777

-- Given conditions as definitions
def bags_with_5_oranges_and_7_apples (m : ℕ) : ℕ × ℕ :=
  (5 * m + 1, 7 * m)

def bags_with_9_oranges_and_7_apples (n : ℕ) : ℕ × ℕ :=
  (9 * n, 7 * n + 21)

theorem total_oranges_and_apples (m n : ℕ) (k : ℕ) 
  (h1 : (5 * m + 1, 7 * m) = (9 * n, 7 * n + 21)) 
  (h2 : 4 * n ≡ 1 [MOD 5]) : 85 = 36 + 49 :=
by
  sorry

end total_oranges_and_apples_l637_63777


namespace perpendicular_line_slope_l637_63782

theorem perpendicular_line_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x - 2 * y + 5 = 0 → x = 2 * y - 5)
  (h2 : ∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = - (2 / m) * x + 6 / m)
  (h3 : (1 / 2 : ℝ) * - (2 / m) = -1) : m = 1 :=
sorry

end perpendicular_line_slope_l637_63782


namespace unique_solution_of_functional_equation_l637_63758

theorem unique_solution_of_functional_equation
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f (x + y)) = f x + y) :
  ∀ x : ℝ, f x = x := 
sorry

end unique_solution_of_functional_equation_l637_63758


namespace average_of_six_numbers_l637_63709

theorem average_of_six_numbers (a b c d e f : ℝ)
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 :=
by sorry

end average_of_six_numbers_l637_63709


namespace roots_square_sum_l637_63755

theorem roots_square_sum {a b c : ℝ} (h1 : 3 * a^3 + 2 * a^2 - 3 * a - 8 = 0)
                                  (h2 : 3 * b^3 + 2 * b^2 - 3 * b - 8 = 0)
                                  (h3 : 3 * c^3 + 2 * c^2 - 3 * c - 8 = 0)
                                  (sum_roots : a + b + c = -2/3)
                                  (product_pairs : a * b + b * c + c * a = -1) : 
  a^2 + b^2 + c^2 = 22 / 9 := by
  sorry

end roots_square_sum_l637_63755


namespace toothpick_250_stage_l637_63731

-- Define the arithmetic sequence for number of toothpicks at each stage
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

-- The proof statement for the 250th stage
theorem toothpick_250_stage : toothpicks 250 = 1001 :=
  by
  sorry

end toothpick_250_stage_l637_63731


namespace number_of_integer_solutions_l637_63778

theorem number_of_integer_solutions (h : ∀ n : ℤ, (2020 - n) ^ 2 / (2020 - n ^ 2) ≥ 0) :
  ∃! (m : ℤ), m = 90 := 
sorry

end number_of_integer_solutions_l637_63778


namespace percentage_increase_l637_63795

theorem percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1.1025 → x = 5.024 := 
sorry

end percentage_increase_l637_63795


namespace triangle_inequality_l637_63729

variable {α : Type*} [LinearOrderedField α]

/-- Given a triangle ABC with sides a, b, c, circumradius R, 
exradii r_a, r_b, r_c, and given 2R ≤ r_a, we need to show that a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : α) (h₁ : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry

end triangle_inequality_l637_63729


namespace cost_of_iphone_l637_63784

def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80
def weeks_worked : ℕ := 7
def total_earnings := weekly_earnings * weeks_worked
def total_money := total_earnings + trade_in_value
def new_iphone_cost : ℕ := 800

theorem cost_of_iphone :
  total_money = new_iphone_cost := by
  sorry

end cost_of_iphone_l637_63784


namespace calculate_v3_l637_63746

def f (x : ℤ) : ℤ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def v0 : ℤ := 2
def v1 (x : ℤ) : ℤ := v0 * x + 5
def v2 (x : ℤ) : ℤ := v1 x * x + 6
def v3 (x : ℤ) : ℤ := v2 x * x + 23

theorem calculate_v3 : v3 (-4) = -49 :=
by
sorry

end calculate_v3_l637_63746


namespace range_of_c_extreme_values_l637_63732

noncomputable def f (c x : ℝ) : ℝ := x^3 - 2 * c * x^2 + x

theorem range_of_c_extreme_values 
  (c : ℝ) 
  (h : ∃ a b : ℝ, a ≠ b ∧ (3 * a^2 - 4 * c * a + 1 = 0) ∧ (3 * b^2 - 4 * c * b + 1 = 0)) :
  c < - (Real.sqrt 3 / 2) ∨ c > (Real.sqrt 3 / 2) :=
by sorry

end range_of_c_extreme_values_l637_63732


namespace circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l637_63741

-- Define the variables
variables {a b c : ℝ} {x y z : ℝ}
variables {α β γ : ℝ}

-- Circumcircle equation
theorem circumcircle_trilinear_eq :
  a * y * z + b * x * z + c * x * y = 0 :=
sorry

-- Incircle equation
theorem incircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt x) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

-- Excircle equation
theorem excircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt (-x)) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

end circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l637_63741


namespace find_triples_l637_63773

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

theorem find_triples :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ is_solution a b c :=
sorry

end find_triples_l637_63773


namespace sides_of_figures_intersection_l637_63791

theorem sides_of_figures_intersection (n p q : ℕ) (h1 : p ≠ 0) (h2 : q ≠ 0) :
  p + q ≤ n + 4 :=
by sorry

end sides_of_figures_intersection_l637_63791


namespace simplify_fraction_l637_63797

theorem simplify_fraction (a b : ℕ) (h : a = 150) (hb : b = 450) : a / b = 1 / 3 := by
  sorry

end simplify_fraction_l637_63797


namespace baby_turtles_on_sand_l637_63708

theorem baby_turtles_on_sand (total_swept : ℕ) (total_hatched : ℕ) (h1 : total_hatched = 42) (h2 : total_swept = total_hatched / 3) :
  total_hatched - total_swept = 28 := by
  sorry

end baby_turtles_on_sand_l637_63708


namespace sequence_properties_l637_63794

-- Define the sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

-- Define the conditions
axiom h1 : a 1 = 1
axiom h2 : b 1 = 1
axiom h3 : ∀ n, b (n + 1) ^ 2 = b n * b (n + 2)
axiom h4 : 9 * (b 3) ^ 2 = b 2 * b 6
axiom h5 : ∀ n, b (n + 1) / a (n + 1) = b n / (a n + 2 * b n)

-- Define the theorem to prove
theorem sequence_properties :
  (∀ n, a n = (2 * n - 1) * 3 ^ (n - 1)) ∧
  (∀ n, (a n) / (b n) = (a (n + 1)) / (b (n + 1)) + 2) := by
  sorry

end sequence_properties_l637_63794


namespace sub_three_five_l637_63702

theorem sub_three_five : 3 - 5 = -2 := 
by 
  sorry

end sub_three_five_l637_63702


namespace shanna_initial_tomato_plants_l637_63712

theorem shanna_initial_tomato_plants (T : ℕ) 
  (h1 : 56 = (T / 2) * 7 + 2 * 7 + 3 * 7) : 
  T = 6 :=
by sorry

end shanna_initial_tomato_plants_l637_63712


namespace tank_capacity_l637_63736

theorem tank_capacity (T : ℝ) (h1 : T * (4 / 5) - T * (5 / 8) = 15) : T = 86 :=
by
  sorry

end tank_capacity_l637_63736


namespace perimeter_of_triangle_l637_63733

noncomputable def ellipse_perimeter (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) : ℝ :=
  let a := 2
  let c := 1
  2 * a + 2 * c

theorem perimeter_of_triangle (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) :
  ellipse_perimeter x y h = 6 :=
by 
  sorry

end perimeter_of_triangle_l637_63733


namespace jelly_beans_problem_l637_63765

/-- Mrs. Wonderful's jelly beans problem -/
theorem jelly_beans_problem : ∃ n_girls n_boys : ℕ, 
  (n_boys = n_girls + 2) ∧
  ((n_girls ^ 2) + ((n_girls + 2) ^ 2) = 394) ∧
  (n_girls + n_boys = 28) :=
by
  sorry

end jelly_beans_problem_l637_63765


namespace PetrovFamilySavings_l637_63787

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l637_63787


namespace proportion_of_ones_l637_63769

theorem proportion_of_ones (m n : ℕ) (h : Nat.gcd m n = 1) : 
  m + n = 275 :=
  sorry

end proportion_of_ones_l637_63769


namespace find_x_intercept_of_line_through_points_l637_63749

-- Definitions based on the conditions
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (0, 3)

-- Statement: The x-intercept of the line passing through the given points is -3/2
theorem find_x_intercept_of_line_through_points :
  let x1 := point1.1
  let y1 := point1.2
  let x2 := point2.1
  let y2 := point2.2
  ∃ x_intercept : ℝ, x_intercept = -3 / 2 ∧ 
    (∀ x, ∀ y, (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) → y = 0 → x = x_intercept) :=
by
  sorry

end find_x_intercept_of_line_through_points_l637_63749


namespace total_amount_distributed_l637_63798

def number_of_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem total_amount_distributed : (number_of_persons * amount_per_person) = 42900 := by
  sorry

end total_amount_distributed_l637_63798


namespace largest_n_binary_operation_l637_63700

-- Define the binary operation @
def binary_operation (n : ℤ) : ℤ := n - (n * 5)

-- Define the theorem stating the desired property
theorem largest_n_binary_operation (x : ℤ) (h : x > -8) :
  ∃ (n : ℤ), n = 2 ∧ binary_operation n < x :=
sorry

end largest_n_binary_operation_l637_63700


namespace kaleb_money_earned_l637_63701

-- Definitions based on the conditions
def total_games : ℕ := 10
def non_working_games : ℕ := 8
def price_per_game : ℕ := 6

-- Calculate the number of working games
def working_games : ℕ := total_games - non_working_games

-- Calculate the total money earned by Kaleb
def money_earned : ℕ := working_games * price_per_game

-- The theorem to prove
theorem kaleb_money_earned : money_earned = 12 := by sorry

end kaleb_money_earned_l637_63701


namespace base3_to_base10_equiv_l637_63761

theorem base3_to_base10_equiv : 
  let repr := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  repr = 142 :=
by
  sorry

end base3_to_base10_equiv_l637_63761


namespace pencils_left_l637_63715

theorem pencils_left (initial_pencils : ℕ := 79) (pencils_taken : ℕ := 4) : initial_pencils - pencils_taken = 75 :=
by
  sorry

end pencils_left_l637_63715


namespace lines_intersect_l637_63737

theorem lines_intersect (a b : ℝ) 
  (h₁ : ∃ y : ℝ, 4 = (3/4) * y + a ∧ y = 3)
  (h₂ : ∃ x : ℝ, 3 = (3/4) * x + b ∧ x = 4) :
  a + b = 7/4 :=
sorry

end lines_intersect_l637_63737


namespace range_of_k_l637_63705

theorem range_of_k (k : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (k + 2) * x1 - 1 > (k + 2) * x2 - 1) → k < -2 := by
  sorry

end range_of_k_l637_63705


namespace weight_of_B_l637_63735

theorem weight_of_B (A B C : ℝ)
(h1 : (A + B + C) / 3 = 45)
(h2 : (A + B) / 2 = 40)
(h3 : (B + C) / 2 = 41)
(h4 : 2 * A = 3 * B ∧ 5 * C = 3 * B)
(h5 : A + B + C = 144) :
B = 43.2 :=
sorry

end weight_of_B_l637_63735


namespace find_a_l637_63774

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) (h2 : x₁ = -2 * a) (h3 : x₂ = 4 * a) (h4 : x₂ - x₁ = 15) : a = 5 / 2 :=
by 
  sorry

end find_a_l637_63774


namespace find_m_l637_63754

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem find_m (a b m : ℝ) (h1 : f m a b = 0) (h2 : 3 * m^2 + 2 * a * m + b = 0)
  (h3 : f (m / 3) a b = 1 / 2) (h4 : m ≠ 0) : m = 3 / 2 :=
  sorry

end find_m_l637_63754


namespace eight_natural_numbers_exist_l637_63742

theorem eight_natural_numbers_exist :
  ∃ (n : Fin 8 → ℕ), (∀ i j : Fin 8, i ≠ j → ¬(n i ∣ n j)) ∧ (∀ i j : Fin 8, i ≠ j → n i ∣ (n j * n j)) :=
by 
  sorry

end eight_natural_numbers_exist_l637_63742


namespace selection_methods_count_l637_63792

-- Define a function to compute combinations (n choose r)
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement
theorem selection_methods_count :
  combination 5 2 * combination 3 1 * combination 2 1 = 60 :=
by
  sorry

end selection_methods_count_l637_63792


namespace manager_salary_l637_63771

theorem manager_salary :
  let avg_salary_employees := 1500
  let num_employees := 20
  let new_avg_salary := 2000
  (new_avg_salary * (num_employees + 1) - avg_salary_employees * num_employees = 12000) :=
by
  sorry

end manager_salary_l637_63771


namespace recurring_decimal_exceeds_by_fraction_l637_63704

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l637_63704


namespace volume_tetrahedron_formula_l637_63713

-- Definitions of the problem elements
def distance (A B C D : Point) : ℝ := sorry
def angle (A B C D : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

-- The problem states you need to prove the volume of the tetrahedron
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ := sorry

-- Conditions
variable (A B C D : Point)
variable (d : ℝ) (phi : ℝ) -- d = distance between lines AB and CD, phi = angle between lines AB and CD

-- Question reformulated as a proof statement
theorem volume_tetrahedron_formula (h1 : d = distance A B C D)
                                   (h2 : phi = angle A B C D) :
  volume_tetrahedron A B C D = (d * length A B * length C D * Real.sin phi) / 6 :=
sorry

end volume_tetrahedron_formula_l637_63713


namespace intersection_of_A_and_B_l637_63762

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 2}
def B := {x : ℝ | x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l637_63762


namespace double_chess_first_player_can_draw_l637_63767

-- Define the basic structure and rules of double chess
structure Game :=
  (state : Type)
  (move : state → state)
  (turn : ℕ → state → state)

-- Define the concept of double move
def double_move (g : Game) (s : g.state) : g.state :=
  g.move (g.move s)

-- Define a condition stating that the first player can at least force a draw
theorem double_chess_first_player_can_draw
  (game : Game)
  (initial_state : game.state)
  (double_move_valid : ∀ s : game.state, ∃ s' : game.state, s' = double_move game s) :
  ∃ draw : game.state, ∀ second_player_strategy : game.state → game.state, 
    double_move game initial_state = draw :=
  sorry

end double_chess_first_player_can_draw_l637_63767


namespace real_roots_of_quadratic_l637_63703

theorem real_roots_of_quadratic (m : ℝ) : ((m - 2) ≠ 0 ∧ (-4 * m + 24) ≥ 0) → (m ≤ 6 ∧ m ≠ 2) := 
by 
  sorry

end real_roots_of_quadratic_l637_63703


namespace ball_bounce_height_l637_63772

theorem ball_bounce_height :
  ∃ k : ℕ, (500 * (2 / 3:ℝ)^k < 10) ∧ (∀ m : ℕ, m < k → ¬(500 * (2 / 3:ℝ)^m < 10)) :=
sorry

end ball_bounce_height_l637_63772


namespace quadratic_inequality_solution_set_l637_63752

variable (a b c : ℝ) (α β : ℝ)

theorem quadratic_inequality_solution_set
  (hαβ : α < β)
  (hα_lt_0 : α < 0) 
  (hβ_lt_0 : β < 0)
  (h_sol_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ (x < α ∨ x > β)) :
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ (-(1 / α) < x ∧ x < -(1 / β))) :=
  sorry

end quadratic_inequality_solution_set_l637_63752


namespace james_calories_ratio_l637_63763

theorem james_calories_ratio:
  ∀ (dancing_sessions_per_day : ℕ) (hours_per_session : ℕ) 
  (days_per_week : ℕ) (calories_per_hour_walking : ℕ) 
  (total_calories_dancing_per_week : ℕ),
  dancing_sessions_per_day = 2 →
  hours_per_session = 1/2 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  total_calories_dancing_per_week = 2400 →
  300 * 2 = 600 →
  (total_calories_dancing_per_week / (dancing_sessions_per_day * hours_per_session * days_per_week)) / calories_per_hour_walking = 2 :=
by
  sorry

end james_calories_ratio_l637_63763


namespace net_rate_of_pay_equals_39_dollars_per_hour_l637_63793

-- Definitions of the conditions
def hours_travelled : ℕ := 3
def speed_per_hour : ℕ := 60
def car_consumption_rate : ℕ := 30
def earnings_per_mile : ℕ := 75  -- expressing $0.75 as 75 cents to avoid floating-point
def gasoline_cost_per_gallon : ℕ := 300  -- expressing $3.00 as 300 cents to avoid floating-point

-- Proof statement
theorem net_rate_of_pay_equals_39_dollars_per_hour : 
  (earnings_per_mile * (speed_per_hour * hours_travelled) - gasoline_cost_per_gallon * ((speed_per_hour * hours_travelled) / car_consumption_rate)) / hours_travelled = 3900 := 
by 
  -- The statement below essentially expresses 39 dollars per hour in cents (i.e., 3900 cents per hour).
  sorry

end net_rate_of_pay_equals_39_dollars_per_hour_l637_63793


namespace geometric_sequence_x_l637_63726

theorem geometric_sequence_x (x : ℝ) (h : 1 * 9 = x^2) : x = 3 ∨ x = -3 :=
by
  sorry

end geometric_sequence_x_l637_63726


namespace line_intersects_x_axis_at_3_0_l637_63775

theorem line_intersects_x_axis_at_3_0 : ∃ (x : ℝ), ∃ (y : ℝ), 2 * y + 5 * x = 15 ∧ y = 0 ∧ (x, y) = (3, 0) :=
by
  sorry

end line_intersects_x_axis_at_3_0_l637_63775


namespace evaluate_power_l637_63707

theorem evaluate_power (x : ℝ) (hx : (8:ℝ)^(2 * x) = 11) : 
  2^(x + 1.5) = 11^(1 / 6) * 2 * Real.sqrt 2 :=
by 
  sorry

end evaluate_power_l637_63707


namespace no_function_satisfies_condition_l637_63747

theorem no_function_satisfies_condition :
  ¬ ∃ (f: ℕ → ℕ), ∀ (n: ℕ), f (f n) = n + 2017 :=
by
  -- Proof details are omitted
  sorry

end no_function_satisfies_condition_l637_63747


namespace value_of_a_l637_63779

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x + a^2 * y + 6 = 0 → (a-2) * x + 3 * a * y + 2 * a = 0) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l637_63779


namespace correct_system_of_equations_l637_63744

-- Definitions based on the conditions
def rope_exceeds (x y : ℝ) : Prop := x - y = 4.5
def rope_half_falls_short (x y : ℝ) : Prop := (1/2) * x + 1 = y

-- Proof statement
theorem correct_system_of_equations (x y : ℝ) :
  rope_exceeds x y → rope_half_falls_short x y → 
  (x - y = 4.5 ∧ (1/2 * x + 1 = y)) := 
by 
  sorry

end correct_system_of_equations_l637_63744


namespace program_output_is_10_l637_63781

def final_value_of_A : ℤ :=
  let A := 2
  let A := A * 2
  let A := A + 6
  A

theorem program_output_is_10 : final_value_of_A = 10 := by
  sorry

end program_output_is_10_l637_63781


namespace ratio_of_green_to_yellow_l637_63796

def envelopes_problem (B Y G X : ℕ) : Prop :=
  B = 14 ∧
  Y = B - 6 ∧
  G = X * Y ∧
  B + Y + G = 46 ∧
  G / Y = 3

theorem ratio_of_green_to_yellow :
  ∃ B Y G X : ℕ, envelopes_problem B Y G X :=
by
  sorry

end ratio_of_green_to_yellow_l637_63796


namespace number_of_adults_l637_63718

theorem number_of_adults
  (A C : ℕ)
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) :
  A = 350 :=
by
  sorry

end number_of_adults_l637_63718


namespace grill_burns_fifteen_coals_in_twenty_minutes_l637_63745

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end grill_burns_fifteen_coals_in_twenty_minutes_l637_63745


namespace total_amount_correct_l637_63711

/-- Meghan has the following cash denominations: -/
def num_100_bills : ℕ := 2
def num_50_bills : ℕ := 5
def num_10_bills : ℕ := 10

/-- Value of each denomination: -/
def value_100_bill : ℕ := 100
def value_50_bill : ℕ := 50
def value_10_bill : ℕ := 10

/-- Meghan's total amount of money: -/
def total_amount : ℕ :=
  (num_100_bills * value_100_bill) +
  (num_50_bills * value_50_bill) +
  (num_10_bills * value_10_bill)

/-- The proof: -/
theorem total_amount_correct : total_amount = 550 :=
by
  -- sorry for now
  sorry

end total_amount_correct_l637_63711


namespace pipes_fill_tank_in_one_hour_l637_63786

theorem pipes_fill_tank_in_one_hour (p q r s : ℝ) (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  1 / (p + q + r + s) = 1 :=
by
  sorry

end pipes_fill_tank_in_one_hour_l637_63786


namespace find_q_of_quadratic_with_roots_ratio_l637_63727

theorem find_q_of_quadratic_with_roots_ratio {q : ℝ} :
  (∃ r1 r2 : ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ r1 / r2 = 3 / 1 ∧ r1 + r2 = -10 ∧ r1 * r2 = q) →
  q = 18.75 :=
by
  sorry

end find_q_of_quadratic_with_roots_ratio_l637_63727


namespace car_speed_proof_l637_63766

noncomputable def car_speed_in_kmh (rpm : ℕ) (circumference : ℕ) : ℕ :=
  (rpm * circumference * 60) / 1000

theorem car_speed_proof : 
  car_speed_in_kmh 400 1 = 24 := 
by
  sorry

end car_speed_proof_l637_63766


namespace alice_meets_john_time_l637_63714

-- Definitions according to conditions
def john_speed : ℝ := 4
def bob_speed : ℝ := 6
def alice_speed : ℝ := 3
def initial_distance_alice_john : ℝ := 2

-- Prove the required meeting time
theorem alice_meets_john_time : 2 / (john_speed + alice_speed) * 60 = 17 := 
by
  sorry

end alice_meets_john_time_l637_63714


namespace chickens_increased_l637_63730

-- Definitions and conditions
def initial_chickens := 45
def chickens_bought_day1 := 18
def chickens_bought_day2 := 12
def total_chickens_bought := chickens_bought_day1 + chickens_bought_day2

-- Proof statement
theorem chickens_increased :
  total_chickens_bought = 30 :=
by
  sorry

end chickens_increased_l637_63730


namespace bees_hatch_every_day_l637_63790

   /-- 
   Given:
   - The queen loses 900 bees every day.
   - The initial number of bees is 12500.
   - After 7 days, the total number of bees is 27201.
   
   Prove:
   - The number of bees hatching from the queen's eggs every day is 3001.
   -/
   
   theorem bees_hatch_every_day :
     ∃ x : ℕ, 12500 + 7 * (x - 900) = 27201 → x = 3001 :=
   sorry
   
end bees_hatch_every_day_l637_63790


namespace fraction_identity_l637_63725

open Real

theorem fraction_identity
  (p q r : ℝ)
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 :=
  sorry

end fraction_identity_l637_63725


namespace percent_decrease_in_hours_l637_63717

theorem percent_decrease_in_hours (W H : ℝ) 
  (h1 : W > 0) 
  (h2 : H > 0)
  (new_wage : ℝ := W * 1.25)
  (H_new : ℝ := H / 1.25)
  (total_income_same : W * H = new_wage * H_new) :
  ((H - H_new) / H) * 100 = 20 := 
by
  sorry

end percent_decrease_in_hours_l637_63717


namespace lisa_total_distance_l637_63734

-- Definitions for distances and counts of trips
def plane_distance : ℝ := 256.0
def train_distance : ℝ := 120.5
def bus_distance : ℝ := 35.2

def plane_trips : ℕ := 32
def train_trips : ℕ := 16
def bus_trips : ℕ := 42

-- Definition of total distance traveled
def total_distance_traveled : ℝ :=
  (plane_distance * plane_trips)
  + (train_distance * train_trips)
  + (bus_distance * bus_trips)

-- The statement to be proven
theorem lisa_total_distance :
  total_distance_traveled = 11598.4 := by
  sorry

end lisa_total_distance_l637_63734


namespace number_of_monomials_is_3_l637_63719

def isMonomial (term : String) : Bool :=
  match term with
  | "0" => true
  | "-a" => true
  | "-3x^2y" => true
  | _ => false

def monomialCount (terms : List String) : Nat :=
  terms.filter isMonomial |>.length

theorem number_of_monomials_is_3 :
  monomialCount ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"] = 3 :=
by
  sorry

end number_of_monomials_is_3_l637_63719


namespace liam_birthday_next_monday_2018_l637_63789

-- Define year advancement rules
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define function to calculate next weekday
def next_weekday (current_day : ℕ) (years_elapsed : ℕ) : ℕ :=
  let advance := (years_elapsed / 4) * 2 + (years_elapsed % 4)
  (current_day + advance) % 7

theorem liam_birthday_next_monday_2018 :
  (next_weekday 4 3 = 0) :=
sorry

end liam_birthday_next_monday_2018_l637_63789


namespace line_does_not_pass_through_third_quadrant_l637_63757

-- Define the Cartesian equation of the line
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 1

-- Define the property that a point (x, y) belongs to the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- State the theorem
theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ in_third_quadrant x y :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l637_63757


namespace total_puppies_count_l637_63722

theorem total_puppies_count (total_cost sale_cost others_cost: ℕ) 
  (three_puppies_on_sale: ℕ) 
  (one_sale_puppy_cost: ℕ)
  (one_other_puppy_cost: ℕ)
  (h1: total_cost = 800)
  (h2: three_puppies_on_sale = 3)
  (h3: one_sale_puppy_cost = 150)
  (h4: others_cost = total_cost - three_puppies_on_sale * one_sale_puppy_cost)
  (h5: one_other_puppy_cost = 175)
  (h6: ∃ other_puppies : ℕ, other_puppies = others_cost / one_other_puppy_cost) :
  ∃ total_puppies : ℕ,
  total_puppies = three_puppies_on_sale + (others_cost / one_other_puppy_cost) := 
sorry

end total_puppies_count_l637_63722


namespace area_of_square_containing_circle_l637_63710

theorem area_of_square_containing_circle (r : ℝ) (hr : r = 4) :
  ∃ (a : ℝ), a = 64 ∧ (∀ (s : ℝ), s = 2 * r → a = s * s) :=
by
  use 64
  sorry

end area_of_square_containing_circle_l637_63710


namespace sum_of_series_l637_63785

noncomputable def infinite_series_sum : ℚ :=
∑' n : ℕ, (3 * (n + 1) - 2) / (((n + 1) : ℚ) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_of_series : infinite_series_sum = 11 / 24 := by
  sorry

end sum_of_series_l637_63785


namespace isosceles_triangle_perimeter_l637_63753

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l637_63753


namespace arithmetic_sequence_abs_sum_l637_63748

theorem arithmetic_sequence_abs_sum :
  ∀ (a : ℕ → ℤ), (∀ n, a (n + 1) - a n = 2) → a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 18) :=
by
  sorry

end arithmetic_sequence_abs_sum_l637_63748


namespace inequality_solution_set_l637_63738

noncomputable def solution_set := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : {x : ℝ | (x - 1) * (3 - x) ≥ 0} = solution_set := by
  sorry

end inequality_solution_set_l637_63738


namespace sum_of_numbers_l637_63739

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l637_63739


namespace textile_firm_looms_l637_63783

theorem textile_firm_looms
  (sales_val : ℝ)
  (manu_exp : ℝ)
  (estab_charges : ℝ)
  (profit_decrease : ℝ)
  (L : ℝ)
  (h_sales : sales_val = 500000)
  (h_manu_exp : manu_exp = 150000)
  (h_estab_charges : estab_charges = 75000)
  (h_profit_decrease : profit_decrease = 7000)
  (hem_equal_contrib : ∀ l : ℝ, l > 0 →
    (l = sales_val / (sales_val / L) - manu_exp / (manu_exp / L)))
  : L = 50 := 
by
  sorry

end textile_firm_looms_l637_63783
