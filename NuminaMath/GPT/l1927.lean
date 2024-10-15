import Mathlib

namespace NUMINAMATH_GPT_find_params_l1927_192796

theorem find_params (a b c : ℝ) :
    (∀ x : ℝ, x = 2 ∨ x = -2 → x^5 + 4 * x^4 + a * x = b * x^2 + 4 * c) 
    → a = 16 ∧ b = 48 ∧ c = -32 :=
by
  sorry

end NUMINAMATH_GPT_find_params_l1927_192796


namespace NUMINAMATH_GPT_velocity_division_l1927_192710

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end NUMINAMATH_GPT_velocity_division_l1927_192710


namespace NUMINAMATH_GPT_weigh_grain_with_inaccurate_scales_l1927_192701

theorem weigh_grain_with_inaccurate_scales
  (inaccurate_scales : ℕ → ℕ → Prop)
  (correct_weight : ℕ)
  (bag_of_grain : ℕ → Prop)
  (balanced : ∀ a b : ℕ, inaccurate_scales a b → a = b := sorry)
  : ∃ grain_weight : ℕ, bag_of_grain grain_weight ∧ grain_weight = correct_weight :=
sorry

end NUMINAMATH_GPT_weigh_grain_with_inaccurate_scales_l1927_192701


namespace NUMINAMATH_GPT_B_gain_correct_l1927_192732

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_of_B : ℝ :=
  let principal : ℝ := 3150
  let interest_rate_A_to_B : ℝ := 0.08
  let annual_compound : ℕ := 1
  let time_A_to_B : ℝ := 3

  let interest_rate_B_to_C : ℝ := 0.125
  let semiannual_compound : ℕ := 2
  let time_B_to_C : ℝ := 2.5

  let amount_A_to_B := compound_interest principal interest_rate_A_to_B annual_compound time_A_to_B
  let amount_B_to_C := compound_interest principal interest_rate_B_to_C semiannual_compound time_B_to_C

  amount_B_to_C - amount_A_to_B

theorem B_gain_correct : gain_of_B = 282.32 :=
  sorry

end NUMINAMATH_GPT_B_gain_correct_l1927_192732


namespace NUMINAMATH_GPT_problem_l1927_192764

theorem problem (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2 * a - 1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1927_192764


namespace NUMINAMATH_GPT_shortest_distance_proof_l1927_192747

noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  let f_p := -p^2 + (6 - k) * p + 18
  let d := |f_p|
  d / (Real.sqrt (k^2 + 1))

theorem shortest_distance_proof (k : ℝ) :
  shortest_distance k = 
  |(-(k - 6) / 2^2 + (6 - k) * (k - 6) / 2 + 18)| / (Real.sqrt (k^2 + 1)) :=
sorry

end NUMINAMATH_GPT_shortest_distance_proof_l1927_192747


namespace NUMINAMATH_GPT_second_recipe_cup_count_l1927_192782

theorem second_recipe_cup_count (bottle_ounces : ℕ) (ounces_per_cup : ℕ)
  (first_recipe_cups : ℕ) (third_recipe_cups : ℕ) (bottles_needed : ℕ)
  (total_ounces : bottle_ounces = 16)
  (ounce_to_cup : ounces_per_cup = 8)
  (first_recipe : first_recipe_cups = 2)
  (third_recipe : third_recipe_cups = 3)
  (bottles : bottles_needed = 3) :
  (bottles_needed * bottle_ounces) / ounces_per_cup - first_recipe_cups - third_recipe_cups = 1 :=
by
  sorry

end NUMINAMATH_GPT_second_recipe_cup_count_l1927_192782


namespace NUMINAMATH_GPT_find_M_l1927_192726

theorem find_M : ∃ M : ℕ, M > 0 ∧ 18 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2 ∧ M = 54 := by
  use 54
  sorry

end NUMINAMATH_GPT_find_M_l1927_192726


namespace NUMINAMATH_GPT_class_sizes_l1927_192730

theorem class_sizes
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (garcia_students : ℕ)
  (smith_students : ℕ)
  (h1 : finley_students = 24)
  (h2 : johnson_students = 10 + finley_students / 2)
  (h3 : garcia_students = 2 * johnson_students)
  (h4 : smith_students = finley_students / 3) :
  finley_students = 24 ∧ johnson_students = 22 ∧ garcia_students = 44 ∧ smith_students = 8 :=
by
  sorry

end NUMINAMATH_GPT_class_sizes_l1927_192730


namespace NUMINAMATH_GPT_digit_inequality_l1927_192769

theorem digit_inequality : ∃ (n : ℕ), n = 9 ∧ ∀ (d : ℕ), d < 10 → (2 + d / 10 + 5 / 1000 > 2 + 5 / 1000) → d > 0 :=
by
  sorry

end NUMINAMATH_GPT_digit_inequality_l1927_192769


namespace NUMINAMATH_GPT_allowance_spent_l1927_192772

variable (A x y : ℝ)
variable (h1 : x = 0.20 * (A - y))
variable (h2 : y = 0.05 * (A - x))

theorem allowance_spent : (x + y) / A = 23 / 100 :=
by 
  sorry

end NUMINAMATH_GPT_allowance_spent_l1927_192772


namespace NUMINAMATH_GPT_converse_even_sum_l1927_192706

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem converse_even_sum (a b : Int) :
  (is_even a ∧ is_even b → is_even (a + b)) →
  (is_even (a + b) → is_even a ∧ is_even b) :=
by
  sorry

end NUMINAMATH_GPT_converse_even_sum_l1927_192706


namespace NUMINAMATH_GPT_bryan_bought_4_pairs_of_pants_l1927_192768

def number_of_tshirts : Nat := 5
def total_cost : Nat := 1500
def cost_per_tshirt : Nat := 100
def cost_per_pants : Nat := 250

theorem bryan_bought_4_pairs_of_pants : (total_cost - number_of_tshirts * cost_per_tshirt) / cost_per_pants = 4 := by
  sorry

end NUMINAMATH_GPT_bryan_bought_4_pairs_of_pants_l1927_192768


namespace NUMINAMATH_GPT_find_x_if_arithmetic_mean_is_12_l1927_192746

theorem find_x_if_arithmetic_mean_is_12 (x : ℝ) (h : (8 + 16 + 21 + 7 + x) / 5 = 12) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_arithmetic_mean_is_12_l1927_192746


namespace NUMINAMATH_GPT_calculator_sum_is_large_l1927_192786

-- Definitions for initial conditions and operations
def participants := 50
def initial_calc1 := 2
def initial_calc2 := -2
def initial_calc3 := 0

-- Define the operations
def operation_calc1 (n : ℕ) := initial_calc1 * 2^n
def operation_calc2 (n : ℕ) := (-2) ^ (2^n)
def operation_calc3 (n : ℕ) := initial_calc3 - n

-- Define the final values for each calculator
def final_calc1 := operation_calc1 participants
def final_calc2 := operation_calc2 participants
def final_calc3 := operation_calc3 participants

-- The final sum
def final_sum := final_calc1 + final_calc2 + final_calc3

-- Prove the final result
theorem calculator_sum_is_large :
  final_sum = 2 ^ (2 ^ 50) :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_calculator_sum_is_large_l1927_192786


namespace NUMINAMATH_GPT_smallest_perimeter_of_triangle_with_consecutive_odd_integers_l1927_192749

theorem smallest_perimeter_of_triangle_with_consecutive_odd_integers :
  ∃ (a b c : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
  (a < b) ∧ (b < c) ∧ (c = a + 4) ∧
  (a + b > c) ∧ (b + c > a) ∧ (a + c > b) ∧ 
  (a + b + c = 15) :=
by
  sorry

end NUMINAMATH_GPT_smallest_perimeter_of_triangle_with_consecutive_odd_integers_l1927_192749


namespace NUMINAMATH_GPT_yogurt_combinations_l1927_192756

-- Definitions based on conditions
def flavors : ℕ := 5
def toppings : ℕ := 8
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The problem statement to be proved
theorem yogurt_combinations :
  flavors * choose toppings 3 = 280 :=
by
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l1927_192756


namespace NUMINAMATH_GPT_complex_number_quadrant_l1927_192729

def i_squared : ℂ := -1

def z (i : ℂ) : ℂ := (-2 + i) * i^5

def in_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant 
  (i : ℂ) (hi : i^2 = -1) (z_val : z i = (-2 + i) * i^5) :
  in_quadrant_III (z i) :=
sorry

end NUMINAMATH_GPT_complex_number_quadrant_l1927_192729


namespace NUMINAMATH_GPT_squared_sum_of_a_b_l1927_192770

theorem squared_sum_of_a_b (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) : (a + b) ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_squared_sum_of_a_b_l1927_192770


namespace NUMINAMATH_GPT_juniors_score_l1927_192771

theorem juniors_score (juniors seniors total_students avg_score avg_seniors_score total_score : ℝ)
  (hj: juniors = 0.2 * total_students)
  (hs: seniors = 0.8 * total_students)
  (ht: total_students = 20)
  (ha: avg_score = 78)
  (hp: (seniors * avg_seniors_score + juniors * c) / total_students = avg_score)
  (havg_seniors: avg_seniors_score = 76)
  (hts: total_score = total_students * avg_score)
  (total_seniors_score : ℝ)
  (hts_seniors: total_seniors_score = seniors * avg_seniors_score)
  (total_juniors_score : ℝ)
  (hts_juniors: total_juniors_score = total_score - total_seniors_score)
  (hjs: c = total_juniors_score / juniors) :
  c = 86 :=
sorry

end NUMINAMATH_GPT_juniors_score_l1927_192771


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1927_192780

open Real Nat

theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^4 = (7! : ℝ))
  (h2 : a * r^7 = (8! : ℝ)) : a = 315 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1927_192780


namespace NUMINAMATH_GPT_derivative_of_y_l1927_192735

noncomputable def y (x : ℝ) : ℝ :=
  -1/4 * Real.arcsin ((5 + 3 * Real.cosh x) / (3 + 5 * Real.cosh x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (3 + 5 * Real.cosh x) :=
sorry

end NUMINAMATH_GPT_derivative_of_y_l1927_192735


namespace NUMINAMATH_GPT_solve_equation_l1927_192789

theorem solve_equation (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 / x + 4 / y = 1) : 
  x = 3 * y / (y - 4) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1927_192789


namespace NUMINAMATH_GPT_blue_paint_cans_needed_l1927_192776

-- Definitions of the conditions
def blue_to_green_ratio : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 42
def expected_blue_cans : ℕ := 24

-- Proof statement
theorem blue_paint_cans_needed (r : ℕ × ℕ) (total : ℕ) (expected : ℕ) 
  (h1: r = (4, 3)) (h2: total = 42) : expected = 24 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_cans_needed_l1927_192776


namespace NUMINAMATH_GPT_smallest_n_for_at_least_64_candies_l1927_192719

theorem smallest_n_for_at_least_64_candies :
  ∃ n : ℕ, (n > 0) ∧ (n * (n + 1) / 2 ≥ 64) ∧ (∀ m : ℕ, (m > 0) ∧ (m * (m + 1) / 2 ≥ 64) → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_n_for_at_least_64_candies_l1927_192719


namespace NUMINAMATH_GPT_min_value_expression_l1927_192727

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1927_192727


namespace NUMINAMATH_GPT_part1_part2_part3_l1927_192722

def climbing_function_1_example (x : ℝ) : Prop :=
  ∃ a : ℝ, a^2 = -8 / a

theorem part1 (x : ℝ) : climbing_function_1_example x ↔ (x = -2) := sorry

def climbing_function_2_example (m : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = m*a + m) ∧ ∀ d: ℝ, ((d^2 = m*d + m) → d = a)

theorem part2 (m : ℝ) : (m = -4) ∧ climbing_function_2_example m := sorry

def climbing_function_3_example (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : Prop :=
  ∃ a1 a2 : ℝ, ((a1 + a2 = n/(1-m)) ∧ (a1*a2 = 1/(m-1)) ∧ (|a1 - a2| = p)) ∧ 
  (∀ x : ℝ, (m * x^2 + n * x + 1) ≥ q) 

theorem part3 (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : climbing_function_3_example m n p q h1 h2 ↔ (0 < q) ∧ (q ≤ 4/11) := sorry

end NUMINAMATH_GPT_part1_part2_part3_l1927_192722


namespace NUMINAMATH_GPT_always_positive_inequality_l1927_192753

theorem always_positive_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_always_positive_inequality_l1927_192753


namespace NUMINAMATH_GPT_exists_consecutive_natural_numbers_satisfy_equation_l1927_192745

theorem exists_consecutive_natural_numbers_satisfy_equation :
  ∃ (n a b c d: ℕ), a = n ∧ b = n+2 ∧ c = n-1 ∧ d = n+1 ∧ n>0 ∧ a * b - c * d = 11 :=
by
  sorry

end NUMINAMATH_GPT_exists_consecutive_natural_numbers_satisfy_equation_l1927_192745


namespace NUMINAMATH_GPT_total_cost_is_17_l1927_192742

def taco_shells_cost : ℝ := 5
def bell_pepper_cost_per_unit : ℝ := 1.5
def bell_pepper_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2

def total_spent : ℝ :=
  taco_shells_cost + (bell_pepper_cost_per_unit * bell_pepper_quantity) + (meat_cost_per_pound * meat_quantity)

theorem total_cost_is_17 : total_spent = 17 := 
  by sorry

end NUMINAMATH_GPT_total_cost_is_17_l1927_192742


namespace NUMINAMATH_GPT_sqrt_of_16_is_4_l1927_192765

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_16_is_4_l1927_192765


namespace NUMINAMATH_GPT_maximize_total_profit_maximize_average_annual_profit_l1927_192741

-- Define the profit function
def total_profit (x : ℤ) : ℤ := -x^2 + 18*x - 36

-- Define the average annual profit function
def average_annual_profit (x : ℤ) : ℤ :=
  let y := total_profit x
  y / x

-- Prove the maximum total profit
theorem maximize_total_profit : 
  ∃ x : ℤ, (total_profit x = 45) ∧ (x = 9) := 
  sorry

-- Prove the maximum average annual profit
theorem maximize_average_annual_profit : 
  ∃ x : ℤ, (average_annual_profit x = 6) ∧ (x = 6) :=
  sorry

end NUMINAMATH_GPT_maximize_total_profit_maximize_average_annual_profit_l1927_192741


namespace NUMINAMATH_GPT_quadratic_three_distinct_solutions_l1927_192794

open Classical

variable (a b c : ℝ) (x1 x2 x3 : ℝ)

-- Conditions:
variables (hx1 : a * x1^2 + b * x1 + c = 0)
          (hx2 : a * x2^2 + b * x2 + c = 0)
          (hx3 : a * x3^2 + b * x3 + c = 0)
          (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

-- Proof problem
theorem quadratic_three_distinct_solutions : a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_three_distinct_solutions_l1927_192794


namespace NUMINAMATH_GPT_floor_length_l1927_192792

theorem floor_length (width length : ℕ) 
  (cost_per_square total_cost : ℕ)
  (square_side : ℕ)
  (h1 : width = 64) 
  (h2 : square_side = 8)
  (h3 : cost_per_square = 24)
  (h4 : total_cost = 576) 
  : length = 24 :=
by
  -- Placeholder for the proof, using sorry
  sorry

end NUMINAMATH_GPT_floor_length_l1927_192792


namespace NUMINAMATH_GPT_volume_of_revolution_l1927_192767

theorem volume_of_revolution (a : ℝ) (h : 0 < a) :
  let x (θ : ℝ) := a * (1 + Real.cos θ) * Real.cos θ
  let y (θ : ℝ) := a * (1 + Real.cos θ) * Real.sin θ
  V = (8 / 3) * π * a^3 :=
sorry

end NUMINAMATH_GPT_volume_of_revolution_l1927_192767


namespace NUMINAMATH_GPT_village_foods_sales_l1927_192707

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end NUMINAMATH_GPT_village_foods_sales_l1927_192707


namespace NUMINAMATH_GPT_find_small_pack_size_l1927_192740

-- Define the conditions of the problem
def soymilk_sold_in_packs (pack_size : ℕ) : Prop :=
  pack_size = 2 ∨ ∃ L : ℕ, pack_size = L

def cartons_bought (total_cartons : ℕ) (large_pack_size : ℕ) (num_large_packs : ℕ) (small_pack_size : ℕ) : Prop :=
  total_cartons = num_large_packs * large_pack_size + small_pack_size

-- The problem statement as a Lean theorem
theorem find_small_pack_size (total_cartons : ℕ) (num_large_packs : ℕ) (large_pack_size : ℕ) :
  soymilk_sold_in_packs 2 →
  soymilk_sold_in_packs large_pack_size →
  cartons_bought total_cartons large_pack_size num_large_packs 2 →
  total_cartons = 17 →
  num_large_packs = 3 →
  large_pack_size = 5 →
  ∃ S : ℕ, soymilk_sold_in_packs S ∧ S = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_small_pack_size_l1927_192740


namespace NUMINAMATH_GPT_bottles_drunk_l1927_192783

theorem bottles_drunk (initial_bottles remaining_bottles : ℕ)
  (h₀ : initial_bottles = 17) (h₁ : remaining_bottles = 14) :
  initial_bottles - remaining_bottles = 3 :=
sorry

end NUMINAMATH_GPT_bottles_drunk_l1927_192783


namespace NUMINAMATH_GPT_line_intersects_parabola_at_vertex_l1927_192736

theorem line_intersects_parabola_at_vertex :
  ∃ (a : ℝ), (∀ x : ℝ, -x + a = x^2 + a^2) ↔ a = 0 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_at_vertex_l1927_192736


namespace NUMINAMATH_GPT_future_years_l1927_192717

theorem future_years (P A F : ℝ) (Y : ℝ) 
  (h1 : P = 50)
  (h2 : P = 1.25 * A)
  (h3 : P = 5 / 6 * F)
  (h4 : A + 10 + Y = F) : 
  Y = 10 := sorry

end NUMINAMATH_GPT_future_years_l1927_192717


namespace NUMINAMATH_GPT_problem_I_problem_II_l1927_192754

-- Define the function f as given
def f (x m : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Problem (I)
theorem problem_I (x : ℝ) : -2 < x ∧ x < 1 ↔ f x 2 < 0 := sorry

-- Problem (II)
theorem problem_II (m : ℝ) : ∀ x, f x m + 1 ≥ 0 ↔ -3 ≤ m ∧ m ≤ 1 := sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1927_192754


namespace NUMINAMATH_GPT_all_lines_can_be_paired_perpendicular_l1927_192760

noncomputable def can_pair_perpendicular_lines : Prop := 
  ∀ (L1 L2 : ℝ), 
    L1 ≠ L2 → 
      ∃ (m : ℝ), 
        (m * L1 = -1/L2 ∨ L1 = 0 ∧ L2 ≠ 0 ∨ L2 = 0 ∧ L1 ≠ 0)

theorem all_lines_can_be_paired_perpendicular : can_pair_perpendicular_lines :=
sorry

end NUMINAMATH_GPT_all_lines_can_be_paired_perpendicular_l1927_192760


namespace NUMINAMATH_GPT_range_and_intervals_of_f_l1927_192759

noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2 - 2 * x - 3)

theorem range_and_intervals_of_f :
  (∀ y, y > 0 → y ≤ 81 → (∃ x : ℝ, f x = y)) ∧
  (∀ x y, x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ≥ y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_GPT_range_and_intervals_of_f_l1927_192759


namespace NUMINAMATH_GPT_area_ratio_l1927_192755

variables {rA rB : ℝ} (C_A C_B : ℝ)

#check C_A = 2 * Real.pi * rA
#check C_B = 2 * Real.pi * rB

theorem area_ratio (h : (60 / 360) * C_A = (40 / 360) * C_B) : (Real.pi * rA^2) / (Real.pi * rB^2) = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_area_ratio_l1927_192755


namespace NUMINAMATH_GPT_find_f_6_l1927_192781

def f : ℕ → ℕ := sorry

lemma f_equality (x : ℕ) : f (x + 1) = x := sorry

theorem find_f_6 : f 6 = 5 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_find_f_6_l1927_192781


namespace NUMINAMATH_GPT_largest_value_among_l1927_192723

theorem largest_value_among (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hneq : a ≠ b) :
  max (a + b) (max (2 * Real.sqrt (a * b)) ((a^2 + b^2) / (2 * a * b))) = a + b :=
sorry

end NUMINAMATH_GPT_largest_value_among_l1927_192723


namespace NUMINAMATH_GPT_mom_foster_dog_food_l1927_192795

theorem mom_foster_dog_food
    (puppy_food_per_meal : ℚ := 1 / 2)
    (puppy_meals_per_day : ℕ := 2)
    (num_puppies : ℕ := 5)
    (total_food_needed : ℚ := 57)
    (days : ℕ := 6)
    (mom_meals_per_day : ℕ := 3) :
    (total_food_needed - (num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days)) / (↑days * ↑mom_meals_per_day) = 1.5 :=
by
  -- Definitions translation
  let puppy_total_food := num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days
  let mom_total_food := total_food_needed - puppy_total_food
  let mom_meals := ↑days * ↑mom_meals_per_day
  -- Proof starts with sorry to indicate that the proof part is not included
  sorry

end NUMINAMATH_GPT_mom_foster_dog_food_l1927_192795


namespace NUMINAMATH_GPT_unique_solution_range_l1927_192705
-- import relevant libraries

-- define the functions
def f (a x : ℝ) : ℝ := 2 * a * x ^ 3 + 3
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 2

-- state and prove the main theorem (statement only)
theorem unique_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = g x ∧ ∀ y : ℝ, y > 0 → f a y = g y → y = x) ↔ a ∈ Set.Iio (-1) :=
sorry

end NUMINAMATH_GPT_unique_solution_range_l1927_192705


namespace NUMINAMATH_GPT_triangle_inradius_exradius_l1927_192791

-- Define the properties of the triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the inradius
def inradius (a b c : ℝ) (r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Define the exradius
def exradius (a b c : ℝ) (rc : ℝ) : Prop :=
  rc = (a + b + c) / 2

-- Formalize the Lean statement for the given proof problem
theorem triangle_inradius_exradius (a b c r rc: ℝ) 
  (h_triangle: right_triangle a b c) : 
  inradius a b c r ∧ exradius a b c rc :=
by
  sorry

end NUMINAMATH_GPT_triangle_inradius_exradius_l1927_192791


namespace NUMINAMATH_GPT_calculate_expression_l1927_192737

theorem calculate_expression : (Real.sqrt 4) + abs (3 - Real.pi) + (1 / 3)⁻¹ = 2 + Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1927_192737


namespace NUMINAMATH_GPT_parabola_vertex_calc_l1927_192762

noncomputable def vertex_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_vertex_calc 
  (a b c : ℝ) 
  (h_vertex : vertex_parabola a b c 2 = 5)
  (h_point : vertex_parabola a b c 1 = 8) : 
  a - b + c = 32 :=
sorry

end NUMINAMATH_GPT_parabola_vertex_calc_l1927_192762


namespace NUMINAMATH_GPT_circle_area_l1927_192712

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : π * r^2 = 3 / 2 :=
by
  -- We leave this place for computations and derivations.
  sorry

end NUMINAMATH_GPT_circle_area_l1927_192712


namespace NUMINAMATH_GPT_john_pushups_less_l1927_192713

theorem john_pushups_less (zachary david john : ℕ) 
  (h1 : zachary = 19)
  (h2 : david = zachary + 39)
  (h3 : david = 58)
  (h4 : john < david) : 
  david - john = 0 :=
sorry

end NUMINAMATH_GPT_john_pushups_less_l1927_192713


namespace NUMINAMATH_GPT_same_number_written_every_vertex_l1927_192761

theorem same_number_written_every_vertex (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i > 0) 
(h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (a i) ^ 2 = a (i - 1) + a (i + 1) ) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i = 2 :=
by
  sorry

end NUMINAMATH_GPT_same_number_written_every_vertex_l1927_192761


namespace NUMINAMATH_GPT_team_E_speed_l1927_192731

noncomputable def average_speed_team_E (d t_E t_A v_A v_E : ℝ) : Prop :=
  d = 300 ∧
  t_A = t_E - 3 ∧
  v_A = v_E + 5 ∧
  d = v_E * t_E ∧
  d = v_A * t_A →
  v_E = 20

theorem team_E_speed : ∃ (v_E : ℝ), average_speed_team_E 300 t_E (t_E - 3) (v_E + 5) v_E :=
by
  sorry

end NUMINAMATH_GPT_team_E_speed_l1927_192731


namespace NUMINAMATH_GPT_initial_number_of_men_l1927_192744

theorem initial_number_of_men (x : ℕ) :
    (50 * x = 25 * (x + 20)) → x = 20 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1927_192744


namespace NUMINAMATH_GPT_expression_evaluation_l1927_192784

-- Define the variables and the given condition
variables (x y : ℝ)

-- Define the equation condition
def equation_condition : Prop := x - 3 * y = 4

-- State the theorem
theorem expression_evaluation (h : equation_condition x y) : 15 * y - 5 * x + 6 = -14 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1927_192784


namespace NUMINAMATH_GPT_estimate_red_balls_l1927_192798

-- Define the conditions
variable (total_balls : ℕ)
variable (prob_red_ball : ℝ)
variable (frequency_red_ball : ℝ := prob_red_ball)

-- Assume total number of balls in the bag is 20
axiom total_balls_eq_20 : total_balls = 20

-- Assume the probability (or frequency) of drawing a red ball
axiom prob_red_ball_eq_0_25 : prob_red_ball = 0.25

-- The Lean statement
theorem estimate_red_balls (H1 : total_balls = 20) (H2 : prob_red_ball = 0.25) : total_balls * prob_red_ball = 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end NUMINAMATH_GPT_estimate_red_balls_l1927_192798


namespace NUMINAMATH_GPT_typing_problem_l1927_192704

theorem typing_problem (a b m n : ℕ) (h1 : 60 = a * b) (h2 : 540 = 75 * n) (h3 : n = 3 * m) :
  a = 25 :=
by {
  -- sorry placeholder where the proof would go
  sorry
}

end NUMINAMATH_GPT_typing_problem_l1927_192704


namespace NUMINAMATH_GPT_value_of_x_l1927_192775

theorem value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 6 = 3 * y) : x = 108 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1927_192775


namespace NUMINAMATH_GPT_probability_of_one_failure_l1927_192787

theorem probability_of_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  (p1 * (1 - p2) + (1 - p1) * p2) = 0.14 :=
by
  rw [h1, h2]
  -- Additional leaning code can be inserted here to finalize the proof if this was complete
  sorry

end NUMINAMATH_GPT_probability_of_one_failure_l1927_192787


namespace NUMINAMATH_GPT_correct_articles_l1927_192711

-- Definitions based on conditions provided in the problem
def sentence := "Traveling in ____ outer space is quite ____ exciting experience."
def first_blank_article := "no article"
def second_blank_article := "an"

-- Statement of the proof problem
theorem correct_articles : 
  (first_blank_article = "no article" ∧ second_blank_article = "an") :=
by
  sorry

end NUMINAMATH_GPT_correct_articles_l1927_192711


namespace NUMINAMATH_GPT_math_marks_l1927_192799

theorem math_marks (english physics chemistry biology total_marks math_marks : ℕ) 
  (h_eng : english = 73)
  (h_phy : physics = 92)
  (h_chem : chemistry = 64)
  (h_bio : biology = 82)
  (h_avg : total_marks = 76 * 5) :
  math_marks = 69 := 
by
  sorry

end NUMINAMATH_GPT_math_marks_l1927_192799


namespace NUMINAMATH_GPT_sequence_contains_2017_l1927_192777

theorem sequence_contains_2017 (a1 d : ℕ) (hpos : d > 0)
  (k n m l : ℕ) 
  (hk : 25 = a1 + k * d)
  (hn : 41 = a1 + n * d)
  (hm : 65 = a1 + m * d)
  (h2017 : 2017 = a1 + l * d) : l > 0 :=
sorry

end NUMINAMATH_GPT_sequence_contains_2017_l1927_192777


namespace NUMINAMATH_GPT_number_of_male_employees_l1927_192773

theorem number_of_male_employees (num_female : ℕ) (x y : ℕ) 
  (h1 : 7 * x = y) 
  (h2 : 8 * x = num_female) 
  (h3 : 9 * (7 * x + 3) = 8 * num_female) :
  y = 189 := by
  sorry

end NUMINAMATH_GPT_number_of_male_employees_l1927_192773


namespace NUMINAMATH_GPT_largest_three_digit_number_satisfying_conditions_l1927_192751

def valid_digits (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def sum_of_two_digit_permutations_eq (a b c : ℕ) : Prop :=
  22 * (a + b + c) = 100 * a + 10 * b + c

theorem largest_three_digit_number_satisfying_conditions (a b c : ℕ) :
  valid_digits a b c →
  sum_of_two_digit_permutations_eq a b c →
  100 * a + 10 * b + c ≤ 396 :=
sorry

end NUMINAMATH_GPT_largest_three_digit_number_satisfying_conditions_l1927_192751


namespace NUMINAMATH_GPT_complete_laps_l1927_192718

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end NUMINAMATH_GPT_complete_laps_l1927_192718


namespace NUMINAMATH_GPT_additive_inverse_commutativity_l1927_192700

section
  variable {R : Type} [Ring R] (h : ∀ x : R, x ^ 2 = x)

  theorem additive_inverse (x : R) : -x = x := by
    sorry

  theorem commutativity (x y : R) : x * y = y * x := by
    sorry
end

end NUMINAMATH_GPT_additive_inverse_commutativity_l1927_192700


namespace NUMINAMATH_GPT_evaluate_complex_power_expression_l1927_192709

theorem evaluate_complex_power_expression : (i : ℂ)^23 + ((i : ℂ)^105 * (i : ℂ)^17) = -i - 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_complex_power_expression_l1927_192709


namespace NUMINAMATH_GPT_min_length_intersection_l1927_192728

theorem min_length_intersection
  (m n : ℝ)
  (hM0 : 0 ≤ m)
  (hM1 : m + 3/4 ≤ 1)
  (hN0 : n - 1/3 ≥ 0)
  (hN1 : n ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧
  x = ((m + 3/4) + (n - 1/3)) - 1 :=
sorry

end NUMINAMATH_GPT_min_length_intersection_l1927_192728


namespace NUMINAMATH_GPT_per_capita_income_ratio_l1927_192743

theorem per_capita_income_ratio
  (PL_10 PZ_10 PL_now PZ_now : ℝ)
  (h1 : PZ_10 = 0.4 * PL_10)
  (h2 : PZ_now = 0.8 * PL_now)
  (h3 : PL_now = 3 * PL_10) :
  PZ_now / PZ_10 = 6 := by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_per_capita_income_ratio_l1927_192743


namespace NUMINAMATH_GPT_total_spent_by_pete_and_raymond_l1927_192750

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end NUMINAMATH_GPT_total_spent_by_pete_and_raymond_l1927_192750


namespace NUMINAMATH_GPT_circle_center_radius_l1927_192790

theorem circle_center_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ ((x - 2)^2 + y^2 = 4) ∧ (∃ (c_x c_y r : ℝ), c_x = 2 ∧ c_y = 0 ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l1927_192790


namespace NUMINAMATH_GPT_inequality_holds_l1927_192752

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4 * x + y + 2 * z) * (2 * x + y + 8 * z) ≥ (375 / 2) * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1927_192752


namespace NUMINAMATH_GPT_slope_range_of_line_l1927_192758

/-- A mathematical proof problem to verify the range of the slope of a line
that passes through a given point (-1, -1) and intersects a circle. -/
theorem slope_range_of_line (
  k : ℝ
) : (∃ x y : ℝ, (y + 1 = k * (x + 1)) ∧ (x - 2) ^ 2 + y ^ 2 = 1) ↔ (0 < k ∧ k < 3 / 4) := 
by
  sorry  

end NUMINAMATH_GPT_slope_range_of_line_l1927_192758


namespace NUMINAMATH_GPT_part1_part2_part3_l1927_192757

-- Part 1: Prove that B = 90° given a=20, b=29, c=21

theorem part1 (a b c : ℝ) (h1 : a = 20) (h2 : b = 29) (h3 : c = 21) : 
  ∃ B : ℝ, B = 90 := 
sorry

-- Part 2: Prove that b = 7 given a=3√3, c=2, B=150°

theorem part2 (a c B b : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = 150) : 
  ∃ b : ℝ, b = 7 :=
sorry

-- Part 3: Prove that A = 45° given a=2, b=√2, c=√3 + 1

theorem part3 (a b c A : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3 + 1) : 
  ∃ A : ℝ, A = 45 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1927_192757


namespace NUMINAMATH_GPT_part_I_part_II_l1927_192793

-- Part (I) 
theorem part_I (a b : ℝ) : (∀ x : ℝ, x^2 - 5 * a * x + b > 0 ↔ (x > 4 ∨ x < 1)) → 
(a = 1 ∧ b = 4) :=
by { sorry }

-- Part (II) 
theorem part_II (x y : ℝ) (a b : ℝ) (h : x + y = 2 ∧ a = 1 ∧ b = 4) : 
x > 0 → y > 0 → 
(∃ t : ℝ, t = a / x + b / y ∧ t ≥ 9 / 2) :=
by { sorry }

end NUMINAMATH_GPT_part_I_part_II_l1927_192793


namespace NUMINAMATH_GPT_girls_on_playground_l1927_192748

variable (total_children : ℕ) (boys : ℕ) (girls : ℕ)

theorem girls_on_playground (h1 : total_children = 117) (h2 : boys = 40) (h3 : girls = total_children - boys) : girls = 77 :=
by
  sorry

end NUMINAMATH_GPT_girls_on_playground_l1927_192748


namespace NUMINAMATH_GPT_first_term_arithmetic_sequence_l1927_192763

theorem first_term_arithmetic_sequence (S : ℕ → ℤ) (a : ℤ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2 : ∀ n m, (S (3 * n)) / (S m) = (S (3 * m)) / (S n)) : a = 5 / 2 := 
sorry

end NUMINAMATH_GPT_first_term_arithmetic_sequence_l1927_192763


namespace NUMINAMATH_GPT_pages_revised_once_l1927_192715

-- Definitions
def total_pages : ℕ := 200
def pages_revised_twice : ℕ := 20
def total_cost : ℕ := 1360
def cost_first_time : ℕ := 5
def cost_revision : ℕ := 3

theorem pages_revised_once (x : ℕ) (h1 : total_cost = 1000 + 3 * x + 120) : x = 80 := by
  sorry

end NUMINAMATH_GPT_pages_revised_once_l1927_192715


namespace NUMINAMATH_GPT_algebraic_expression_value_l1927_192724

noncomputable def a : ℝ := Real.sqrt 6 + 1
noncomputable def b : ℝ := Real.sqrt 6 - 1

theorem algebraic_expression_value :
  a^2 + a * b = 12 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1927_192724


namespace NUMINAMATH_GPT_bricks_required_l1927_192721

-- Definitions
def courtyard_length : ℕ := 20  -- in meters
def courtyard_breadth : ℕ := 16  -- in meters
def brick_length : ℕ := 20  -- in centimeters
def brick_breadth : ℕ := 10  -- in centimeters

-- Statement to prove
theorem bricks_required :
  ((courtyard_length * 100) * (courtyard_breadth * 100)) / (brick_length * brick_breadth) = 16000 :=
sorry

end NUMINAMATH_GPT_bricks_required_l1927_192721


namespace NUMINAMATH_GPT_walter_time_at_seals_l1927_192774

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end NUMINAMATH_GPT_walter_time_at_seals_l1927_192774


namespace NUMINAMATH_GPT_solution_set_of_abs_fraction_eq_fraction_l1927_192785

-- Problem Statement
theorem solution_set_of_abs_fraction_eq_fraction :
  { x : ℝ | |x / (x - 1)| = x / (x - 1) } = { x : ℝ | x ≤ 0 ∨ x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_abs_fraction_eq_fraction_l1927_192785


namespace NUMINAMATH_GPT_min_ratio_of_integers_l1927_192714

theorem min_ratio_of_integers (x y : ℕ) (hx : 50 < x) (hy : 50 < y) (h_mean : x + y = 130) : 
  x = 51 → y = 79 → x / y = 51 / 79 := by
  sorry

end NUMINAMATH_GPT_min_ratio_of_integers_l1927_192714


namespace NUMINAMATH_GPT_geom_seq_sum_of_terms_l1927_192703

theorem geom_seq_sum_of_terms
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geometric: ∀ n, a (n + 1) = a n * q)
  (h_q : q = 2)
  (h_sum : a 0 + a 1 + a 2 = 21)
  (h_pos : ∀ n, a n > 0) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_of_terms_l1927_192703


namespace NUMINAMATH_GPT_final_tree_count_l1927_192788

noncomputable def current_trees : ℕ := 39
noncomputable def trees_planted_today : ℕ := 41
noncomputable def trees_planted_tomorrow : ℕ := 20

theorem final_tree_count : current_trees + trees_planted_today + trees_planted_tomorrow = 100 := by
  sorry

end NUMINAMATH_GPT_final_tree_count_l1927_192788


namespace NUMINAMATH_GPT_triangle_angle_construction_l1927_192797

-- Step d): Lean 4 Statement
theorem triangle_angle_construction (a b c : ℝ) (α β : ℝ) (γ : ℝ) (h1 : γ = 120)
  (h2 : a < c) (h3 : c < a + b) (h4 : b < c)  (h5 : c < a + b) :
    (∃ α' β' γ', α' = 60 ∧ β' = α ∧ γ' = 60 + β) ∧ 
    (∃ α'' β'' γ'', α'' = 60 ∧ β'' = β ∧ γ'' = 60 + α) :=
  sorry

end NUMINAMATH_GPT_triangle_angle_construction_l1927_192797


namespace NUMINAMATH_GPT_find_B_l1927_192778

variable (A B : ℝ)

def condition1 : Prop := A + B = 1210
def condition2 : Prop := (4 / 15) * A = (2 / 5) * B

theorem find_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 484 :=
sorry

end NUMINAMATH_GPT_find_B_l1927_192778


namespace NUMINAMATH_GPT_simplify_expression_l1927_192779

variable (a b : ℝ)
variable (h₁ : a = 3 + Real.sqrt 5)
variable (h₂ : b = 3 - Real.sqrt 5)

theorem simplify_expression : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1927_192779


namespace NUMINAMATH_GPT_max_area_of_fenced_rectangle_l1927_192716

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_area_of_fenced_rectangle_l1927_192716


namespace NUMINAMATH_GPT_percent_relation_l1927_192766

variable (x y z : ℝ)

theorem percent_relation (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : x = 0.78 * z :=
by sorry

end NUMINAMATH_GPT_percent_relation_l1927_192766


namespace NUMINAMATH_GPT_solve_for_y_l1927_192702

theorem solve_for_y (x y : ℝ) (h : 2 * y - 4 * x + 5 = 0) : y = 2 * x - 2.5 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1927_192702


namespace NUMINAMATH_GPT_solve_inequality_l1927_192708

theorem solve_inequality :
  {x : ℝ | 0 ≤ x ∧ x ≤ 1 } = {x : ℝ | x * (x - 1) ≤ 0} :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l1927_192708


namespace NUMINAMATH_GPT_roots_polynomial_d_l1927_192720

theorem roots_polynomial_d (c d u v : ℝ) (ru rpush rv rpush2 : ℝ) :
    (u + v + ru = 0) ∧ (u+3 + v-2 + rpush2 = 0) ∧
    (d + 153 = -(u + 3) * (v - 2) * (ru)) ∧ (d + 153 = s) ∧ (s = -(u + 3) * (v - 2) * (rpush2 - 1)) →
    d = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_d_l1927_192720


namespace NUMINAMATH_GPT_triangle_angle_sum_l1927_192733

theorem triangle_angle_sum (x : ℝ) :
  let a := 40
  let b := 60
  let sum_of_angles := 180
  a + b + x = sum_of_angles → x = 80 :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l1927_192733


namespace NUMINAMATH_GPT_ratio_of_groups_l1927_192738

variable (x : ℚ)

-- The total number of people in the calligraphy group
def calligraphy_group (x : ℚ) := x + (2 / 7) * x

-- The total number of people in the recitation group
def recitation_group (x : ℚ) := x + (1 / 5) * x

theorem ratio_of_groups (x : ℚ) (hx : x ≠ 0) : 
    (calligraphy_group x) / (recitation_group x) = (3 : ℚ) / (4 : ℚ) := by
  sorry

end NUMINAMATH_GPT_ratio_of_groups_l1927_192738


namespace NUMINAMATH_GPT_eggs_left_l1927_192725

theorem eggs_left (x : ℕ) : (47 - 5 - x) = (42 - x) :=
  by
  sorry

end NUMINAMATH_GPT_eggs_left_l1927_192725


namespace NUMINAMATH_GPT_minimum_workers_needed_to_make_profit_l1927_192734

-- Given conditions
def fixed_maintenance_fee : ℝ := 550
def setup_cost : ℝ := 200
def wage_per_hour : ℝ := 18
def widgets_per_worker_per_hour : ℝ := 6
def sell_price_per_widget : ℝ := 3.5
def work_hours_per_day : ℝ := 8

-- Definitions derived from conditions
def daily_wage_per_worker := wage_per_hour * work_hours_per_day
def daily_revenue_per_worker := widgets_per_worker_per_hour * work_hours_per_day * sell_price_per_widget
def total_daily_cost (n : ℝ) := fixed_maintenance_fee + setup_cost + n * daily_wage_per_worker

-- Prove that the number of workers needed to make a profit is at least 32
theorem minimum_workers_needed_to_make_profit (n : ℕ) (h : (total_daily_cost (n : ℝ)) < n * daily_revenue_per_worker) :
  n ≥ 32 := by
  -- We fill the sorry for proof to pass Lean check
  sorry

end NUMINAMATH_GPT_minimum_workers_needed_to_make_profit_l1927_192734


namespace NUMINAMATH_GPT_minimum_value_8m_n_l1927_192739

noncomputable def min_value (m n : ℝ) : ℝ :=
  8 * m + n

theorem minimum_value_8m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (8 / n) = 4) : 
  min_value m n = 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_8m_n_l1927_192739
