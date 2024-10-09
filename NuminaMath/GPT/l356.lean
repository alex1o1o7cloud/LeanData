import Mathlib

namespace polar_to_rectangular_l356_35614

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 5) (h₂ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5 / 2, -5 * Real.sqrt 3 / 2) :=
by sorry

end polar_to_rectangular_l356_35614


namespace a11_a12_a13_eq_105_l356_35638

variable (a : ℕ → ℝ) -- Define the arithmetic sequence
variable (d : ℝ) -- Define the common difference

-- Assume the conditions given in step a)
axiom arith_seq (n : ℕ) : a n = a 0 + n * d
axiom sum_3_eq_15 : a 0 + a 1 + a 2 = 15
axiom prod_3_eq_80 : a 0 * a 1 * a 2 = 80
axiom pos_diff : d > 0

theorem a11_a12_a13_eq_105 : a 10 + a 11 + a 12 = 105 :=
sorry

end a11_a12_a13_eq_105_l356_35638


namespace error_in_step_one_l356_35671

theorem error_in_step_one : 
  ∃ a b c d : ℝ, 
    (a * (x + 1) - b = c * (x - 2)) = (3 * (x + 1) - 6 = 2 * (x - 2)) → 
    a ≠ 3 ∨ b ≠ 6 ∨ c ≠ 2 := 
by
  sorry

end error_in_step_one_l356_35671


namespace warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l356_35682

noncomputable def netChange (tonnages : List Int) : Int :=
  List.sum tonnages

noncomputable def initialGoods (finalGoods : Int) (change : Int) : Int :=
  finalGoods + change

noncomputable def totalFees (tonnages : List Int) (feePerTon : Int) : Int :=
  feePerTon * List.sum (tonnages.map (Int.natAbs))

theorem warehouseGoodsDecreased 
  (tonnages : List Int) (finalGoods : Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20]) 
  (h2 : finalGoods = 580)
  (h3 : feePerTon = 4) : 
  netChange tonnages < 0 := by
  sorry

theorem initialTonnage 
  (tonnages : List Int) (finalGoods : Int) (change : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : finalGoods = 580)
  (h3 : change = netChange tonnages) : 
  initialGoods finalGoods change = 630 := by
  sorry

theorem totalLoadingFees 
  (tonnages : List Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : feePerTon = 4) : 
  totalFees tonnages feePerTon = 648 := by
  sorry

end warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l356_35682


namespace hulk_jump_exceeds_2000_l356_35606

theorem hulk_jump_exceeds_2000 {n : ℕ} (h : n ≥ 1) :
  2^(n - 1) > 2000 → n = 12 :=
by
  sorry

end hulk_jump_exceeds_2000_l356_35606


namespace min_of_quadratic_l356_35601

theorem min_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, x^2 + 7 * x + 3 ≤ y^2 + 7 * y + 3) ∧ x = -7 / 2 :=
by
  sorry

end min_of_quadratic_l356_35601


namespace calculate_f_ff_f60_l356_35621

def f (N : ℝ) : ℝ := 0.3 * N + 2

theorem calculate_f_ff_f60 : f (f (f 60)) = 4.4 := by
  sorry

end calculate_f_ff_f60_l356_35621


namespace solution_to_quadratic_solution_to_cubic_l356_35653

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end solution_to_quadratic_solution_to_cubic_l356_35653


namespace fred_limes_l356_35675

theorem fred_limes (limes_total : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) (fred_limes : ℕ)
  (h_total : limes_total = 103)
  (h_alyssa : alyssa_limes = 32)
  (h_nancy : nancy_limes = 35)
  (h_fred : fred_limes = limes_total - (alyssa_limes + nancy_limes)) :
  fred_limes = 36 :=
by
  sorry

end fred_limes_l356_35675


namespace hexagon_perimeter_l356_35602

theorem hexagon_perimeter (s : ℕ) (P : ℕ) (h1 : s = 8) (h2 : 6 > 0) 
                          (h3 : P = 6 * s) : P = 48 := by
  sorry

end hexagon_perimeter_l356_35602


namespace reciprocal_of_neg_2023_l356_35629

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l356_35629


namespace problem_statement_l356_35626

noncomputable def nonreal_omega_root (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω^2 + ω + 1 = 0

theorem problem_statement (ω : ℂ) (h : nonreal_omega_root ω) :
  (1 - 2 * ω + ω^2)^6 + (1 + 2 * ω - ω^2)^6 = 1458 :=
sorry

end problem_statement_l356_35626


namespace required_weekly_hours_approx_27_l356_35608

noncomputable def planned_hours_per_week : ℝ := 25
noncomputable def planned_weeks : ℝ := 15
noncomputable def total_amount : ℝ := 4500
noncomputable def sick_weeks : ℝ := 3
noncomputable def increased_wage_weeks : ℝ := 5
noncomputable def wage_increase_factor : ℝ := 1.5 -- 50%

-- Normal hourly wage
noncomputable def normal_hourly_wage : ℝ := total_amount / (planned_hours_per_week * planned_weeks)

-- Increased hourly wage
noncomputable def increased_hourly_wage : ℝ := normal_hourly_wage * wage_increase_factor

-- Earnings in the last 5 weeks at increased wage
noncomputable def earnings_in_last_5_weeks : ℝ := increased_hourly_wage * planned_hours_per_week * increased_wage_weeks

-- Amount needed before the wage increase
noncomputable def amount_needed_before_wage_increase : ℝ := total_amount - earnings_in_last_5_weeks

-- We have 7 weeks before the wage increase
noncomputable def weeks_before_increase : ℝ := planned_weeks - sick_weeks - increased_wage_weeks

-- New required weekly hours before wage increase
noncomputable def required_weekly_hours : ℝ := amount_needed_before_wage_increase / (normal_hourly_wage * weeks_before_increase)

theorem required_weekly_hours_approx_27 :
  abs (required_weekly_hours - 27) < 1 :=
sorry

end required_weekly_hours_approx_27_l356_35608


namespace positive_number_square_roots_l356_35665

theorem positive_number_square_roots (a : ℝ) 
  (h1 : (2 * a - 1) ^ 2 = (a - 2) ^ 2) 
  (h2 : ∃ b : ℝ, b > 0 ∧ ((2 * a - 1) = b ∨ (a - 2) = b)) : 
  ∃ n : ℝ, n = 1 :=
by
  sorry

end positive_number_square_roots_l356_35665


namespace solve_special_sine_system_l356_35616

noncomputable def special_sine_conditions1 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  let z := -(Real.pi / 2) + 2 * Real.pi * k
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = (-1)^n * Real.pi / 6 + Real.pi * n ∧
  z = -Real.pi / 2 + 2 * Real.pi * k

noncomputable def special_sine_conditions2 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := -Real.pi / 2 + 2 * Real.pi * k
  let z := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = -Real.pi / 2 + 2 * Real.pi * k ∧
  z = (-1)^n * Real.pi / 6 + Real.pi * n

theorem solve_special_sine_system (m n k : ℤ) :
  special_sine_conditions1 m n k ∨ special_sine_conditions2 m n k :=
sorry

end solve_special_sine_system_l356_35616


namespace gardener_tree_arrangement_l356_35640

theorem gardener_tree_arrangement :
  let maple_trees := 4
  let oak_trees := 5
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)
  let valid_slots := 9  -- as per slots identified in the solution
  let valid_arrangements := 1 * Nat.choose valid_slots oak_trees
  let probability := valid_arrangements / total_arrangements
  probability = 1 / 75075 →
  (1 + 75075) = 75076 := by {
    sorry
  }

end gardener_tree_arrangement_l356_35640


namespace function_increasing_probability_l356_35610

noncomputable def is_increasing_on_interval (a b : ℤ) : Prop :=
∀ x : ℝ, x > 1 → 2 * a * x - 2 * b > 0

noncomputable def valid_pairs : List (ℤ × ℤ) :=
[(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]

noncomputable def total_pairs : ℕ :=
3 * 4

noncomputable def probability_of_increasing_function : ℚ :=
(valid_pairs.length : ℚ) / total_pairs

theorem function_increasing_probability :
  probability_of_increasing_function = 5 / 12 :=
by
  sorry

end function_increasing_probability_l356_35610


namespace geometric_series_sum_l356_35655

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum_l356_35655


namespace parabola_sum_l356_35651

variables (a b c x y : ℝ)

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_sum (h1 : ∀ x, quadratic a b c x = -(x - 3)^2 + 4)
    (h2 : quadratic a b c 1 = 0)
    (h3 : quadratic a b c 5 = 0) :
    a + b + c = 0 :=
by
  -- We assume quadratic(a, b, c, x) = a * x^2 + b * x + c
  -- We assume quadratic(a, b, c, 1) = 0 and quadratic(a, b, c, 5) = 0
  -- We need to prove a + b + c = 0
  sorry

end parabola_sum_l356_35651


namespace baking_dish_to_recipe_book_ratio_is_2_l356_35652

-- Definitions of costs
def cost_recipe_book : ℕ := 6
def cost_ingredient : ℕ := 3
def num_ingredients : ℕ := 5
def cost_apron : ℕ := cost_recipe_book + 1
def total_spent : ℕ := 40

-- Definition to calculate the total cost excluding the baking dish
def cost_excluding_baking_dish : ℕ :=
  cost_recipe_book + cost_apron + cost_ingredient * num_ingredients

-- Definition of cost of baking dish
def cost_baking_dish : ℕ := total_spent - cost_excluding_baking_dish

-- Definition of the ratio
def ratio_baking_dish_to_recipe_book : ℕ := cost_baking_dish / cost_recipe_book

-- Theorem stating that the ratio is 2
theorem baking_dish_to_recipe_book_ratio_is_2 :
  ratio_baking_dish_to_recipe_book = 2 :=
sorry

end baking_dish_to_recipe_book_ratio_is_2_l356_35652


namespace greatest_product_sum_300_l356_35698

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l356_35698


namespace integer_solutions_of_inequality_l356_35672

theorem integer_solutions_of_inequality (x : ℤ) : 
  (-4 < 1 - 3 * (x: ℤ) ∧ 1 - 3 * (x: ℤ) ≤ 4) ↔ (x = -1 ∨ x = 0 ∨ x = 1) := 
by 
  sorry

end integer_solutions_of_inequality_l356_35672


namespace platform_length_l356_35685

/-- Mathematical proof problem:
The problem is to prove that given the train's length, time taken to cross a signal pole and 
time taken to cross a platform, the length of the platform is 525 meters.
-/
theorem platform_length 
    (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) (P : ℕ) 
    (h_train_length : train_length = 450) (h_time_pole : time_pole = 18) 
    (h_time_platform : time_platform = 39) (h_P : P = 525) : 
    P = 525 := 
  sorry

end platform_length_l356_35685


namespace car_and_cyclist_speeds_and_meeting_point_l356_35670

/-- 
(1) Distance between points $A$ and $B$ is $80 \mathrm{~km}$.
(2) After one hour, the distance between them reduces to $24 \mathrm{~km}$.
(3) The cyclist takes a 1-hour rest but they meet $90$ minutes after their departure.
-/
def initial_distance : ℝ := 80 -- km
def distance_after_one_hour : ℝ := 24 -- km apart after 1 hour
def cyclist_rest_duration : ℝ := 1 -- hour
def meeting_time : ℝ := 1.5 -- hours (90 minutes after departure)

def car_speed : ℝ := 40 -- km/hr
def cyclist_speed : ℝ := 16 -- km/hr

theorem car_and_cyclist_speeds_and_meeting_point :
  initial_distance = 80 → 
  distance_after_one_hour = 24 → 
  cyclist_rest_duration = 1 → 
  meeting_time = 1.5 → 
  car_speed = 40 ∧ cyclist_speed = 16 ∧ meeting_point_from_A = 60 ∧ meeting_point_from_B = 20 :=
by
  sorry

end car_and_cyclist_speeds_and_meeting_point_l356_35670


namespace find_a_l356_35627

noncomputable def set_A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem find_a (a : ℝ) (h : 3 ∈ set_A a) : a = -3 / 2 :=
by
  sorry

end find_a_l356_35627


namespace tangent_line_to_curve_at_P_l356_35646

noncomputable def tangent_line_at_point (x y : ℝ) := 4 * x - y - 2 = 0

theorem tangent_line_to_curve_at_P :
  (∃ (b: ℝ), ∀ (x: ℝ), b = 2 * 1^2 → tangent_line_at_point 1 2)
:= 
by
  sorry

end tangent_line_to_curve_at_P_l356_35646


namespace Amy_homework_time_l356_35667

def mathProblems : Nat := 18
def spellingProblems : Nat := 6
def problemsPerHour : Nat := 4
def totalProblems : Nat := mathProblems + spellingProblems
def totalHours : Nat := totalProblems / problemsPerHour

theorem Amy_homework_time :
  totalHours = 6 := by
  sorry

end Amy_homework_time_l356_35667


namespace intersection_points_l356_35683

theorem intersection_points : 
  (∃ x : ℝ, y = -2 * x + 4 ∧ y = 0 ∧ (x, y) = (2, 0)) ∧
  (∃ y : ℝ, y = -2 * 0 + 4 ∧ (0, y) = (0, 4)) :=
by
  sorry

end intersection_points_l356_35683


namespace sqrt_expr_eq_l356_35613

theorem sqrt_expr_eq : (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 :=
by sorry

end sqrt_expr_eq_l356_35613


namespace arithmetic_sequence_value_l356_35693

theorem arithmetic_sequence_value :
  ∀ (a_n : ℕ → ℤ) (d : ℤ),
    (∀ n : ℕ, a_n n = a_n 0 + ↑n * d) →
    a_n 2 = 4 →
    a_n 4 = 8 →
    a_n 10 = 20 :=
by
  intros a_n d h_arith h_a3 h_a5
  --
  sorry

end arithmetic_sequence_value_l356_35693


namespace alpha_beta_square_eq_eight_l356_35641

open Real

theorem alpha_beta_square_eq_eight :
  ∃ α β : ℝ, 
  (∀ x : ℝ, x^2 - 2 * x - 1 = 0 ↔ x = α ∨ x = β) → 
  (α ≠ β) → 
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_eq_eight_l356_35641


namespace Jane_possible_numbers_l356_35691

def is_factor (a b : ℕ) : Prop := b % a = 0
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 4000

def Jane_number (m : ℕ) : Prop :=
  is_factor 180 m ∧
  is_factor 42 m ∧
  in_range m

theorem Jane_possible_numbers :
  Jane_number 1260 ∧ Jane_number 2520 ∧ Jane_number 3780 :=
by
  sorry

end Jane_possible_numbers_l356_35691


namespace expression_divisible_by_264_l356_35603

theorem expression_divisible_by_264 (n : ℕ) (h : n > 1) : ∃ k : ℤ, 7^(2*n) - 4^(2*n) - 297 = 264 * k :=
by 
  sorry

end expression_divisible_by_264_l356_35603


namespace binom_150_1_eq_150_l356_35677

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l356_35677


namespace greatest_product_two_integers_sum_2004_l356_35633

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l356_35633


namespace find_a_for_square_binomial_l356_35644

theorem find_a_for_square_binomial (a : ℚ) (h: ∃ (b : ℚ), ∀ (x : ℚ), 9 * x^2 + 21 * x + a = (3 * x + b)^2) : a = 49 / 4 := 
by 
  sorry

end find_a_for_square_binomial_l356_35644


namespace greatest_n_l356_35607

def S := { xy : ℕ × ℕ | ∃ x y : ℕ, xy = (x * y, x + y) }

def in_S (a : ℕ) : Prop := ∃ x y : ℕ, a = x * y * (x + y)

def pow_mod (a b m : ℕ) : ℕ := (a ^ b) % m

def satisfies_condition (a : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → in_S (a + pow_mod 2 k 9)

theorem greatest_n (a : ℕ) (n : ℕ) : 
  satisfies_condition a n → n ≤ 3 :=
sorry

end greatest_n_l356_35607


namespace jordan_rectangle_width_l356_35687

theorem jordan_rectangle_width
  (carol_length : ℕ) (carol_width : ℕ) (jordan_length : ℕ) (jordan_width : ℕ)
  (h_carol_dims : carol_length = 12) (h_carol_dims2 : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_area_eq : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := 
sorry

end jordan_rectangle_width_l356_35687


namespace number_of_pairs_l356_35680

theorem number_of_pairs (n : ℕ) (h : n ≥ 3) : 
  ∃ a : ℕ, a = (n-2) * 2^(n-1) + 1 :=
by
  sorry

end number_of_pairs_l356_35680


namespace total_cars_l356_35676

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l356_35676


namespace bridge_length_l356_35699

-- Definitions based on conditions
def Lt : ℕ := 148
def Skm : ℕ := 45
def T : ℕ := 30

-- Conversion from km/h to m/s
def conversion_factor : ℕ := 1000 / 3600
def Sm : ℝ := Skm * conversion_factor

-- Calculation of distance traveled in 30 seconds
def distance : ℝ := Sm * T

-- The length of the bridge
def L_bridge : ℝ := distance - Lt

theorem bridge_length : L_bridge = 227 := sorry

end bridge_length_l356_35699


namespace mary_sugar_cups_l356_35673

theorem mary_sugar_cups (sugar_required : ℕ) (sugar_remaining : ℕ) (sugar_added : ℕ) (h1 : sugar_required = 11) (h2 : sugar_added = 1) : sugar_remaining = 10 :=
by
  -- Placeholder for the proof
  sorry

end mary_sugar_cups_l356_35673


namespace percentage_increase_twice_l356_35635

theorem percentage_increase_twice {P : ℝ} (x : ℝ) :
  (P * (1 + x)^2) = (P * (1 + 0.6900000000000001)) →
  x = 0.30 :=
by
  sorry

end percentage_increase_twice_l356_35635


namespace largest_solution_of_equation_l356_35649

theorem largest_solution_of_equation :
  let eq := λ x : ℝ => x^4 - 50 * x^2 + 625
  ∃ x : ℝ, eq x = 0 ∧ ∀ y : ℝ, eq y = 0 → y ≤ x :=
sorry

end largest_solution_of_equation_l356_35649


namespace necessary_but_not_sufficient_condition_l356_35660

variable {x : ℝ}

theorem necessary_but_not_sufficient_condition 
    (h : -1 ≤ x ∧ x < 2) : 
    (-1 ≤ x ∧ x < 3) ∧ ¬(((-1 ≤ x ∧ x < 3) → (-1 ≤ x ∧ x < 2))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l356_35660


namespace uniformity_comparison_l356_35694

theorem uniformity_comparison (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : S1^2 < S2^2 :=
by {
  sorry
}

end uniformity_comparison_l356_35694


namespace garden_perimeter_l356_35663

theorem garden_perimeter (w l : ℕ) (garden_width : ℕ) (garden_perimeter : ℕ)
  (garden_area playground_length playground_width : ℕ)
  (h1 : garden_width = 16)
  (h2 : playground_length = 16)
  (h3 : garden_area = 16 * l)
  (h4 : playground_area = w * playground_length)
  (h5 : garden_area = playground_area)
  (h6 : garden_perimeter = 2 * l + 2 * garden_width)
  (h7 : garden_perimeter = 56):
  l = 12 :=
by
  sorry

end garden_perimeter_l356_35663


namespace total_students_in_class_l356_35657

theorem total_students_in_class 
  (b : ℕ)
  (boys_jelly_beans : ℕ := b * b)
  (girls_jelly_beans : ℕ := (b + 1) * (b + 1))
  (total_jelly_beans : ℕ := 432) 
  (condition : boys_jelly_beans + girls_jelly_beans = total_jelly_beans) :
  (b + b + 1 = 29) :=
sorry

end total_students_in_class_l356_35657


namespace purely_imaginary_complex_l356_35650

theorem purely_imaginary_complex :
  ∀ (x y : ℤ), (x - 4) ≠ 0 → (y^2 - 3*y - 4) ≠ 0 → (∃ (z : ℂ), z = ⟨0, x^2 + 3*x - 4⟩) → 
    (x = 4 ∧ y ≠ 4 ∧ y ≠ -1) :=
by
  intro x y hx hy hz
  sorry

end purely_imaginary_complex_l356_35650


namespace find_number_l356_35639

theorem find_number (x : ℝ) (h: x - (3 / 5) * x = 58) : x = 145 :=
by {
  sorry
}

end find_number_l356_35639


namespace radical_axis_of_non_concentric_circles_l356_35697

theorem radical_axis_of_non_concentric_circles 
  {a R1 R2 : ℝ} (a_pos : a ≠ 0) (R1_pos : R1 > 0) (R2_pos : R2 > 0) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ x = (R2^2 - R1^2) / (4 * a) :=
by sorry

end radical_axis_of_non_concentric_circles_l356_35697


namespace find_f7_l356_35631

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x^7 + b * x^3 + c * x - 5

theorem find_f7 (a b c : ℝ) (h : f (-7) a b c = 7) : f 7 a b c = -17 :=
by
  sorry

end find_f7_l356_35631


namespace Henry_trays_per_trip_l356_35612

theorem Henry_trays_per_trip (trays1 trays2 trips : ℕ) (h1 : trays1 = 29) (h2 : trays2 = 52) (h3 : trips = 9) :
  (trays1 + trays2) / trips = 9 :=
by
  sorry

end Henry_trays_per_trip_l356_35612


namespace total_employees_in_buses_l356_35695

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l356_35695


namespace quadratic_less_than_zero_for_x_in_0_1_l356_35623

theorem quadratic_less_than_zero_for_x_in_0_1 (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, 0 < x ∧ x < 1 → (a * x^2 + b * x + c) < 0 :=
by
  sorry

end quadratic_less_than_zero_for_x_in_0_1_l356_35623


namespace reb_min_biking_speed_l356_35630

theorem reb_min_biking_speed (driving_time_minutes driving_speed driving_distance biking_distance_minutes biking_reduction_percentage biking_distance_hours : ℕ) 
  (driving_time_eqn: driving_time_minutes = 45) 
  (driving_speed_eqn: driving_speed = 40) 
  (driving_distance_eqn: driving_distance = driving_speed * driving_time_minutes / 60)
  (biking_reduction_percentage_eqn: biking_reduction_percentage = 20)
  (biking_distance_eqn: biking_distance = driving_distance * (100 - biking_reduction_percentage) / 100)
  (biking_distance_hours_eqn: biking_distance_minutes = 120)
  (biking_hours_eqn: biking_distance_hours = biking_distance_minutes / 60)
  : (biking_distance / biking_distance_hours) ≥ 12 := 
by
  sorry

end reb_min_biking_speed_l356_35630


namespace tournament_total_games_l356_35624

theorem tournament_total_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) : 
  (n * (n - 1) / 2) * k = 1740 := by
  -- Given conditions
  have h1 : n = 30 := h_n
  have h2 : k = 4 := h_k

  -- Calculation using provided values
  sorry

end tournament_total_games_l356_35624


namespace vertex_of_parabola_l356_35619

theorem vertex_of_parabola : 
  (exists (a b: ℝ), ∀ x: ℝ, (a * (x - 1)^2 + b = (x - 1)^2 - 2)) → (1, -2) = (1, -2) :=
by
  intro h
  sorry

end vertex_of_parabola_l356_35619


namespace solve_ab_eq_l356_35615

theorem solve_ab_eq:
  ∃ a b : ℝ, (1 + (2 : ℂ) * (Complex.I)) * (a : ℂ) + (b : ℂ) = (2 : ℂ) * (Complex.I) ∧ a = 1 ∧ b = -1 := by
  sorry

end solve_ab_eq_l356_35615


namespace parabola_hyperbola_focus_l356_35605

theorem parabola_hyperbola_focus (p : ℝ) (hp : 0 < p) :
  (∃ k : ℝ, y^2 = 2 * k * x ∧ k > 0) ∧ (x^2 - y^2 / 3 = 1) → (p = 4) :=
by
  sorry

end parabola_hyperbola_focus_l356_35605


namespace problem_statement_l356_35645

-- Given the conditions and the goal
theorem problem_statement (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz_sum : x + y + z = 1) :
  (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 :=
by
  sorry

end problem_statement_l356_35645


namespace common_ratio_of_geometric_sequence_l356_35611

-- Define the problem conditions and goal
theorem common_ratio_of_geometric_sequence 
  (a1 : ℝ)  -- nonzero first term
  (h₁ : a1 ≠ 0) -- first term is nonzero
  (r : ℝ)  -- common ratio
  (h₂ : r > 0) -- ratio is positive
  (h₃ : ∀ n m : ℕ, n ≠ m → a1 * r^n ≠ a1 * r^m) -- distinct terms in sequence
  (h₄ : a1 * r * r * r = (a1 * r) * (a1 * r^3) ∧ a1 * r ≠ (a1 * r^4)) -- arithmetic sequence condition
  : r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l356_35611


namespace race_time_l356_35659

theorem race_time (t_A t_B : ℝ) (v_A v_B : ℝ)
  (h1 : t_B = t_A + 7)
  (h2 : v_A * t_A = 80)
  (h3 : v_B * t_B = 80)
  (h4 : v_A * (t_A + 7) = 136) :
  t_A = 10 :=
by
  sorry

end race_time_l356_35659


namespace speed_with_stream_l356_35642

noncomputable def man_speed_still_water : ℝ := 5
noncomputable def speed_against_stream : ℝ := 4

theorem speed_with_stream :
  ∃ V_s, man_speed_still_water + V_s = 6 :=
by
  use man_speed_still_water - speed_against_stream
  sorry

end speed_with_stream_l356_35642


namespace polynomial_value_l356_35690

noncomputable def polynomial_spec (p : ℝ) : Prop :=
  p^3 - 5 * p + 1 = 0

theorem polynomial_value (p : ℝ) (h : polynomial_spec p) : 
  p^4 - 3 * p^3 - 5 * p^2 + 16 * p + 2015 = 2018 := 
by
  sorry

end polynomial_value_l356_35690


namespace exists_tetrahedra_volume_and_face_area_conditions_l356_35636

noncomputable def volume (T : Tetrahedron) : ℝ := sorry
noncomputable def face_area (T : Tetrahedron) : List ℝ := sorry

-- The existence of two tetrahedra such that the volume of T1 > T2 
-- and the area of each face of T1 does not exceed any face of T2.
theorem exists_tetrahedra_volume_and_face_area_conditions :
  ∃ (T1 T2 : Tetrahedron), 
    (volume T1 > volume T2) ∧ 
    (∀ (a1 : ℝ), a1 ∈ face_area T1 → 
      ∃ (a2 : ℝ), a2 ∈ face_area T2 ∧ a2 ≥ a1) :=
sorry

end exists_tetrahedra_volume_and_face_area_conditions_l356_35636


namespace chord_intersection_probability_l356_35648

theorem chord_intersection_probability
  (points : Finset Point)
  (hp : points.card = 2000)
  (A B C D E : Point)
  (hA : A ∈ points)
  (hB : B ∈ points)
  (hC : C ∈ points)
  (hD : D ∈ points)
  (hE : E ∈ points)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  : probability_chord_intersection := by
    sorry

end chord_intersection_probability_l356_35648


namespace boat_speed_in_still_water_l356_35637

theorem boat_speed_in_still_water (V_b : ℝ) (D : ℝ) (V_s : ℝ) 
  (h1 : V_s = 3) 
  (h2 : D = (V_b + V_s) * 1) 
  (h3 : D = (V_b - V_s) * 1.5) : 
  V_b = 15 := 
by 
  sorry

end boat_speed_in_still_water_l356_35637


namespace find_nat_numbers_l356_35686

theorem find_nat_numbers (a b : ℕ) (c : ℕ) (h : ∀ n : ℕ, a^n + b^n = c^(n+1)) : a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_nat_numbers_l356_35686


namespace perfect_square_trinomial_l356_35661

theorem perfect_square_trinomial (m : ℝ) (h : ∃ a : ℝ, x^2 + 2 * x + m = (x + a)^2) : m = 1 := 
sorry

end perfect_square_trinomial_l356_35661


namespace max_students_can_distribute_equally_l356_35684

-- Define the given numbers of pens and pencils
def pens : ℕ := 1001
def pencils : ℕ := 910

-- State the problem in Lean 4 as a theorem
theorem max_students_can_distribute_equally :
  Nat.gcd pens pencils = 91 :=
sorry

end max_students_can_distribute_equally_l356_35684


namespace max_value_of_a_l356_35674

theorem max_value_of_a
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

example 
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) : 
  (7 - Real.sqrt 46) / 3 ≤ a :=
sorry

end max_value_of_a_l356_35674


namespace find_function_l356_35600

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_function (f : ℝ → ℝ)
  (cond : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + (f y)^3 + (f z)^3 = 3 * x * y * z) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end find_function_l356_35600


namespace neg_power_identity_l356_35668

variable (m : ℝ)

theorem neg_power_identity : (-m^2)^3 = -m^6 :=
sorry

end neg_power_identity_l356_35668


namespace friendly_number_pair_a_equals_negative_three_fourths_l356_35679

theorem friendly_number_pair_a_equals_negative_three_fourths (a : ℚ) (h : (a / 2) + (3 / 4) = (a + 3) / 6) : 
  a = -3 / 4 :=
sorry

end friendly_number_pair_a_equals_negative_three_fourths_l356_35679


namespace part_I_part_II_l356_35654

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II_l356_35654


namespace book_pairs_count_l356_35692

theorem book_pairs_count :
  let mystery_count := 3
  let fantasy_count := 4
  let biography_count := 3
  mystery_count * fantasy_count + mystery_count * biography_count + fantasy_count * biography_count = 33 :=
by 
  sorry

end book_pairs_count_l356_35692


namespace dot_product_two_a_plus_b_with_a_l356_35669

-- Define vector a
def a : ℝ × ℝ := (2, -1)

-- Define vector b
def b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication of vector a by 2
def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)

-- Define the vector addition of 2a and b
def two_a_plus_b : ℝ × ℝ := (two_a.1 + b.1, two_a.2 + b.2)

-- Define dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the dot product of (2 * a + b) and a equals 6
theorem dot_product_two_a_plus_b_with_a :
  dot_product two_a_plus_b a = 6 :=
by
  sorry

end dot_product_two_a_plus_b_with_a_l356_35669


namespace integers_square_less_than_three_times_l356_35678

theorem integers_square_less_than_three_times (x : ℤ) : x^2 < 3 * x ↔ x = 1 ∨ x = 2 :=
by
  sorry

end integers_square_less_than_three_times_l356_35678


namespace tickets_total_l356_35696

theorem tickets_total (T : ℝ) (h1 : T / 2 + (T / 2) / 4 = 3600) : T = 5760 :=
by
  sorry

end tickets_total_l356_35696


namespace part_a_l356_35647

theorem part_a (students : Fin 64 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (A B : Fin 64), (students A).1 ≥ (students B).1 ∧ (students A).2.1 ≥ (students B).2.1 ∧ (students A).2.2 ≥ (students B).2.2 :=
sorry

end part_a_l356_35647


namespace cyclic_sum_inequality_l356_35609

variable (a b c : ℝ)
variable (pos_a : a > 0)
variable (pos_b : b > 0)
variable (pos_c : c > 0)

theorem cyclic_sum_inequality :
  ( (a^3 + b^3) / (a^2 + a * b + b^2) + 
    (b^3 + c^3) / (b^2 + b * c + c^2) + 
    (c^3 + a^3) / (c^2 + c * a + a^2) ) ≥ 
  (2 / 3) * (a + b + c) := 
  sorry

end cyclic_sum_inequality_l356_35609


namespace probability_red_buttons_l356_35620

/-- 
Initial condition: Jar A contains 6 red buttons and 10 blue buttons.
Carla removes the same number of red buttons as blue buttons from Jar A and places them in Jar B.
Jar A's state after action: Jar A retains 3/4 of its original number of buttons.
Question: What is the probability that both selected buttons are red? Express your answer as a common fraction.
-/
theorem probability_red_buttons :
  let initial_red_a := 6
  let initial_blue_a := 10
  let total_buttons_a := initial_red_a + initial_blue_a
  
  -- Jar A after removing buttons
  let retained_fraction := 3 / 4
  let remaining_buttons_a := retained_fraction * total_buttons_a
  let removed_buttons := total_buttons_a - remaining_buttons_a
  let removed_red_buttons := removed_buttons / 2
  let removed_blue_buttons := removed_buttons / 2
  
  -- Remaining red and blue buttons in Jar A
  let remaining_red_a := initial_red_a - removed_red_buttons
  let remaining_blue_a := initial_blue_a - removed_blue_buttons

  -- Total remaining buttons in Jar A
  let total_remaining_a := remaining_red_a + remaining_blue_a

  -- Jar B contains the removed buttons
  let total_buttons_b := removed_buttons
  
  -- Probability calculations
  let probability_red_a := remaining_red_a / total_remaining_a
  let probability_red_b := removed_red_buttons / total_buttons_b

  -- Combined probability of selecting red button from both jars
  probability_red_a * probability_red_b = 1 / 6 :=
by
  sorry

end probability_red_buttons_l356_35620


namespace goldbach_134_l356_35617

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem goldbach_134 (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum : p + q = 134) (h_diff : p ≠ q) : 
  ∃ (d : ℕ), d = 134 - (2 * p) ∧ d ≤ 128 := 
sorry

end goldbach_134_l356_35617


namespace grandmaster_plays_21_games_l356_35632

theorem grandmaster_plays_21_games (a : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ a (i + 1) - a i) ∧ (∀ i, a (i + 7) - a i ≤ 10) →
  ∃ (i j : ℕ), i < j ∧ (a j - a i = 21) :=
sorry

end grandmaster_plays_21_games_l356_35632


namespace binomial_expansion_sum_l356_35681

theorem binomial_expansion_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h₁ : (a * x - 1)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5)
  (h₂ : a₃ = 80) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
sorry

end binomial_expansion_sum_l356_35681


namespace oranges_ratio_l356_35666

theorem oranges_ratio (initial_oranges_kgs : ℕ) (additional_oranges_kgs : ℕ) (total_oranges_three_weeks : ℕ) :
  initial_oranges_kgs = 10 →
  additional_oranges_kgs = 5 →
  total_oranges_three_weeks = 75 →
  (2 * (total_oranges_three_weeks - (initial_oranges_kgs + additional_oranges_kgs)) / 2) / (initial_oranges_kgs + additional_oranges_kgs) = 2 :=
by
  intros h_initial h_additional h_total
  sorry

end oranges_ratio_l356_35666


namespace min_value_l356_35628

theorem min_value (x : ℝ) (h : x > 2) : ∃ y, y = 22 ∧ 
  ∀ z, (z > 2) → (y ≤ (z^2 + 8) / (Real.sqrt (z - 2))) := 
sorry

end min_value_l356_35628


namespace find_number_l356_35656

theorem find_number (x : ℝ) (h : x / 14.5 = 171) : x = 2479.5 :=
by
  sorry

end find_number_l356_35656


namespace minimum_value_of_a_l356_35622

theorem minimum_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y ≥ 9) : ∃ a > 0, a ≥ 4 :=
sorry

end minimum_value_of_a_l356_35622


namespace star_problem_l356_35634

def star_problem_proof (p q r s u : ℤ) (S : ℤ): Prop :=
  (S = 64) →
  ({n : ℤ | n = 19 ∨ n = 21 ∨ n = 23 ∨ n = 25 ∨ n = 27} = {p, q, r, s, u}) →
  (p + q + r + s + u = 115) →
  (9 + p + q + 7 = S) →
  (3 + p + u + 15 = S) →
  (3 + q + r + 11 = S) →
  (9 + u + s + 11 = S) →
  (15 + s + r + 7 = S) →
  (q = 27)

theorem star_problem : ∃ p q r s u S, star_problem_proof p q r s u S := by
  -- Proof goes here
  sorry

end star_problem_l356_35634


namespace larger_number_is_84_l356_35604

theorem larger_number_is_84 (x y : ℕ) (HCF LCM : ℕ)
  (h_hcf : HCF = 84)
  (h_lcm : LCM = 21)
  (h_ratio : x * 4 = y)
  (h_product : x * y = HCF * LCM) :
  y = 84 :=
by
  sorry

end larger_number_is_84_l356_35604


namespace other_root_zero_l356_35618

theorem other_root_zero (b : ℝ) (x : ℝ) (hx_root : x^2 + b * x = 0) (h_x_eq_minus_two : x = -2) : 
  (0 : ℝ) = 0 :=
by
  sorry

end other_root_zero_l356_35618


namespace cody_tickets_l356_35664

theorem cody_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (won_tickets : ℕ) : 
  initial_tickets = 49 ∧ spent_tickets = 25 ∧ won_tickets = 6 → 
  initial_tickets - spent_tickets + won_tickets = 30 :=
by sorry

end cody_tickets_l356_35664


namespace pitchers_of_lemonade_l356_35625

theorem pitchers_of_lemonade (glasses_per_pitcher : ℕ) (total_glasses_served : ℕ)
  (h1 : glasses_per_pitcher = 5) (h2 : total_glasses_served = 30) :
  total_glasses_served / glasses_per_pitcher = 6 := by
  sorry

end pitchers_of_lemonade_l356_35625


namespace general_term_of_sequence_l356_35662

-- Definition of arithmetic sequence with positive common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℤ} {d : ℤ}
axiom positive_common_difference : d > 0
axiom cond1 : a 3 * a 4 = 117
axiom cond2 : a 2 + a 5 = 22

-- Target statement to prove
theorem general_term_of_sequence : is_arithmetic_sequence a d → a n = 4 * n - 3 :=
by sorry

end general_term_of_sequence_l356_35662


namespace h_eq_20_at_y_eq_4_l356_35688

noncomputable def k (y : ℝ) : ℝ := 40 / (y + 5)

noncomputable def h (y : ℝ) : ℝ := 4 * (k⁻¹ y)

theorem h_eq_20_at_y_eq_4 : h 4 = 20 := 
by 
  -- Insert proof here
  sorry

end h_eq_20_at_y_eq_4_l356_35688


namespace probability_of_X_eq_4_l356_35689

noncomputable def probability_X_eq_4 : ℝ :=
  let total_balls := 12
  let new_balls := 9
  let old_balls := 3
  let draw := 3
  -- Number of ways to choose 2 old balls from 3
  let choose_old := Nat.choose old_balls 2
  -- Number of ways to choose 1 new ball from 9
  let choose_new := Nat.choose new_balls 1
  -- Total number of ways to choose 3 balls from 12
  let total_ways := Nat.choose total_balls draw
  -- Probability calculation
  (choose_old * choose_new) / total_ways

theorem probability_of_X_eq_4 : probability_X_eq_4 = 27 / 220 := by
  sorry

end probability_of_X_eq_4_l356_35689


namespace minimum_resistors_required_l356_35643

-- Define the grid configuration and the connectivity condition
def isReliableGrid (m : ℕ) (n : ℕ) (failures : Finset (ℕ × ℕ)) : Prop :=
m * n > 9 ∧ (∀ (a b : ℕ), a ≠ b → (a, b) ∉ failures)

-- Minimum number of resistors ensuring connectivity with up to 9 failures
theorem minimum_resistors_required :
  ∃ (m n : ℕ), 5 * 5 = 25 ∧ isReliableGrid 5 5 ∅ :=
by
  let m : ℕ := 5
  let n : ℕ := 5
  have h₁ : m * n = 25 := by rfl
  have h₂ : isReliableGrid 5 5 ∅ := by
    unfold isReliableGrid
    exact ⟨by norm_num, sorry⟩ -- formal proof omitted for brevity
  exact ⟨m, n, h₁, h₂⟩

end minimum_resistors_required_l356_35643


namespace fruits_in_box_l356_35658

theorem fruits_in_box (initial_persimmons : ℕ) (added_apples : ℕ) (total_fruits : ℕ) :
  initial_persimmons = 2 → added_apples = 7 → total_fruits = initial_persimmons + added_apples → total_fruits = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end fruits_in_box_l356_35658
