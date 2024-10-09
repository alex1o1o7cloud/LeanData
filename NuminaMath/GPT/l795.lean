import Mathlib

namespace part1_solution_set_part2_comparison_l795_79586

noncomputable def f (x : ℝ) := -|x| - |x + 2|

theorem part1_solution_set (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 :=
by sorry

theorem part2_comparison (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = Real.sqrt 5) : 
  a^2 + b^2 / 4 ≥ f x + 3 :=
by sorry

end part1_solution_set_part2_comparison_l795_79586


namespace total_action_figures_l795_79582

theorem total_action_figures (initial_figures cost_per_figure total_cost needed_figures : ℕ)
  (h1 : initial_figures = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost = 72)
  (h4 : needed_figures = total_cost / cost_per_figure)
  : initial_figures + needed_figures = 16 :=
by
  sorry

end total_action_figures_l795_79582


namespace polygon_diagonals_eq_sum_sides_and_right_angles_l795_79512

-- Define the number of sides of the polygon
variables (n : ℕ)

-- Definition of the number of diagonals in a convex n-sided polygon
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Definition of the sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Definition of equivalent right angles for interior angles
def num_right_angles (n : ℕ) : ℕ := 2 * (n - 2)

-- The proof statement: prove that the equation holds for n
theorem polygon_diagonals_eq_sum_sides_and_right_angles (h : 3 ≤ n) :
  num_diagonals n = n + num_right_angles n :=
sorry

end polygon_diagonals_eq_sum_sides_and_right_angles_l795_79512


namespace greatest_possible_third_term_l795_79545

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l795_79545


namespace ruby_initial_apples_l795_79506

theorem ruby_initial_apples (apples_taken : ℕ) (apples_left : ℕ) (initial_apples : ℕ) 
  (h1 : apples_taken = 55) (h2 : apples_left = 8) (h3 : initial_apples = apples_taken + apples_left) : 
  initial_apples = 63 := 
by
  sorry

end ruby_initial_apples_l795_79506


namespace sam_dimes_proof_l795_79590

def initial_dimes : ℕ := 9
def remaining_dimes : ℕ := 2
def dimes_given : ℕ := 7

theorem sam_dimes_proof : initial_dimes - remaining_dimes = dimes_given :=
by
  sorry

end sam_dimes_proof_l795_79590


namespace min_sum_of_dimensions_l795_79575

/-- A theorem to find the minimum possible sum of the three dimensions of a rectangular box 
with given volume 1729 inch³ and positive integer dimensions. -/
theorem min_sum_of_dimensions (x y z : ℕ) (h1 : x * y * z = 1729) : x + y + z ≥ 39 :=
by
  sorry

end min_sum_of_dimensions_l795_79575


namespace unique_odd_number_between_500_and_1000_l795_79598

theorem unique_odd_number_between_500_and_1000 :
  ∃! x : ℤ, 500 ≤ x ∧ x ≤ 1000 ∧ x % 25 = 6 ∧ x % 9 = 7 ∧ x % 2 = 1 :=
sorry

end unique_odd_number_between_500_and_1000_l795_79598


namespace sponge_cake_eggs_l795_79581

theorem sponge_cake_eggs (eggs flour sugar total desiredCakeMass : ℕ) 
  (h_recipe : eggs = 300) 
  (h_flour : flour = 120)
  (h_sugar : sugar = 100) 
  (h_total : total = 520) 
  (h_desiredMass : desiredCakeMass = 2600) :
  (eggs * desiredCakeMass / total) = 1500 := by
  sorry

end sponge_cake_eggs_l795_79581


namespace stereographic_projection_reflection_l795_79569

noncomputable def sphere : Type := sorry
noncomputable def point_on_sphere (P : sphere) : Prop := sorry
noncomputable def reflection_on_sphere (P P' : sphere) (e : sphere) : Prop := sorry
noncomputable def arbitrary_point (E : sphere) (P P' : sphere) : Prop := E ≠ P ∧ E ≠ P'
noncomputable def tangent_plane (E : sphere) : Type := sorry
noncomputable def stereographic_projection (E : sphere) (δ : Type) : sphere → sorry := sorry
noncomputable def circle_on_plane (e : sphere) (E : sphere) (δ : Type) : Type := sorry
noncomputable def inversion_in_circle (P P' : sphere) (e_1 : Type) : Prop := sorry

theorem stereographic_projection_reflection (P P' E : sphere) (e : sphere) (δ : Type) (e_1 : Type) :
  point_on_sphere P ∧
  reflection_on_sphere P P' e ∧
  arbitrary_point E P P' ∧
  circle_on_plane e E δ = e_1 →
  inversion_in_circle P P' e_1 :=
sorry

end stereographic_projection_reflection_l795_79569


namespace stella_spent_amount_l795_79531

-- Definitions
def num_dolls : ℕ := 3
def num_clocks : ℕ := 2
def num_glasses : ℕ := 5

def price_doll : ℕ := 5
def price_clock : ℕ := 15
def price_glass : ℕ := 4

def profit : ℕ := 25

-- Calculation of total revenue from profit
def total_revenue : ℕ := num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass

-- Proposition to be proved
theorem stella_spent_amount : total_revenue - profit = 40 :=
by sorry

end stella_spent_amount_l795_79531


namespace solve_inequality_l795_79546

noncomputable def solutionSet := { x : ℝ | 0 < x ∧ x < 1 }

theorem solve_inequality (x : ℝ) : x^2 < x ↔ x ∈ solutionSet := 
sorry

end solve_inequality_l795_79546


namespace simplify_and_evaluate_expression_l795_79522

theorem simplify_and_evaluate_expression (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) : 
  1 - (a^2 + 2 * a * b + b^2) / (a^2 - a * b) / ((a + b) / (a - b)) = -1 := 
sorry

end simplify_and_evaluate_expression_l795_79522


namespace reciprocal_of_fraction_sum_l795_79517

theorem reciprocal_of_fraction_sum : 
  (1 / (1 / 3 + 1 / 4 - 1 / 12)) = 2 := sorry

end reciprocal_of_fraction_sum_l795_79517


namespace gcd_qr_l795_79523

theorem gcd_qr (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 770) : Nat.gcd q r = 70 := sorry

end gcd_qr_l795_79523


namespace jeans_cost_proof_l795_79557

def cheaper_jeans_cost (coat_price: Float) (backpack_price: Float) (shoes_price: Float) (subtotal: Float) (difference: Float): Float :=
  let known_items_cost := coat_price + backpack_price + shoes_price
  let jeans_total_cost := subtotal - known_items_cost
  let x := (jeans_total_cost - difference) / 2
  x

def more_expensive_jeans_cost (cheaper_price : Float) (difference: Float): Float :=
  cheaper_price + difference

theorem jeans_cost_proof : ∀ (coat_price backpack_price shoes_price subtotal difference : Float),
  coat_price = 45 →
  backpack_price = 25 →
  shoes_price = 30 →
  subtotal = 139 →
  difference = 15 →
  cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference = 12 ∧
  more_expensive_jeans_cost (cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference) difference = 27 :=
by
  intros coat_price backpack_price shoes_price subtotal difference
  intros h1 h2 h3 h4 h5
  sorry

end jeans_cost_proof_l795_79557


namespace gift_sequence_count_l795_79549

noncomputable def number_of_gift_sequences (students : ℕ) (classes_per_week : ℕ) : ℕ :=
  (students * students) ^ classes_per_week

theorem gift_sequence_count :
  number_of_gift_sequences 15 3 = 11390625 :=
by
  sorry

end gift_sequence_count_l795_79549


namespace find_m_b_sum_does_not_prove_l795_79533

theorem find_m_b_sum_does_not_prove :
  ∃ m b : ℝ, 
  let original_point := (2, 3)
  let image_point := (10, 9)
  let midpoint := ((original_point.1 + image_point.1) / 2, (original_point.2 + image_point.2) / 2)
  m = -4 / 3 ∧ 
  midpoint = (6, 6) ∧ 
  6 = m * 6 + b 
  ∧ m + b = 38 / 3 := sorry

end find_m_b_sum_does_not_prove_l795_79533


namespace max_covered_squares_by_tetromino_l795_79515

-- Definition of the grid size
def grid_size := (5, 5)

-- Definition of S-Tetromino (Z-Tetromino) coverage covering four contiguous squares
def is_STetromino (coords: List (Nat × Nat)) : Prop := 
  coords.length = 4 ∧ ∃ (x y : Nat), coords = [(x, y), (x, y+1), (x+1, y+1), (x+1, y+2)]

-- Definition of the coverage constraint
def no_more_than_two_tiles (cover: List (Nat × Nat)) : Prop :=
  ∀ (coord: Nat × Nat), cover.count coord ≤ 2

-- Definition of the total tiled squares covered by at least one tile
def tiles_covered (cover: List (Nat × Nat)) : Nat := 
  cover.toFinset.card 

-- Definition of the problem using proof equivalence
theorem max_covered_squares_by_tetromino
  (cover: List (List (Nat × Nat)))
  (H_tiles: ∀ t, t ∈ cover → is_STetromino t)
  (H_coverage: no_more_than_two_tiles (cover.join)) :
  tiles_covered (cover.join) = 24 :=
sorry 

end max_covered_squares_by_tetromino_l795_79515


namespace find_sam_age_l795_79535

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l795_79535


namespace Mia_biking_speed_l795_79505

theorem Mia_biking_speed
    (Eugene_speed : ℝ)
    (Carlos_ratio : ℝ)
    (Mia_ratio : ℝ)
    (Mia_speed : ℝ)
    (h1 : Eugene_speed = 5)
    (h2 : Carlos_ratio = 3 / 4)
    (h3 : Mia_ratio = 4 / 3)
    (h4 : Mia_speed = Mia_ratio * (Carlos_ratio * Eugene_speed)) :
    Mia_speed = 5 :=
by
  sorry

end Mia_biking_speed_l795_79505


namespace problem_l795_79516

def p : Prop := 0 % 2 = 0
def q : Prop := ¬(3 % 2 = 0)

theorem problem : p ∨ q :=
by
  sorry

end problem_l795_79516


namespace area_of_trapezium_l795_79507

-- Definitions for the given conditions
def parallel_side_a : ℝ := 18  -- in cm
def parallel_side_b : ℝ := 20  -- in cm
def distance_between_sides : ℝ := 5  -- in cm

-- Statement to prove the area is 95 cm²
theorem area_of_trapezium : 
  let a := parallel_side_a
  let b := parallel_side_b
  let h := distance_between_sides
  (1 / 2 * (a + b) * h = 95) :=
by
  sorry  -- Proof is not required here

end area_of_trapezium_l795_79507


namespace table_height_is_five_l795_79547

def height_of_table (l h w : ℕ) : Prop :=
  l + h + w = 45 ∧ 2 * w + h = 40

theorem table_height_is_five (l w : ℕ) : height_of_table l 5 w :=
by
  sorry

end table_height_is_five_l795_79547


namespace dave_time_correct_l795_79541

-- Definitions for the given conditions
def chuck_time (dave_time : ℕ) := 5 * dave_time
def erica_time (chuck_time : ℕ) := chuck_time + (3 * chuck_time / 10)
def erica_fixed_time := 65

-- Statement to prove
theorem dave_time_correct : ∃ (dave_time : ℕ), erica_time (chuck_time dave_time) = erica_fixed_time ∧ dave_time = 10 := by
  sorry

end dave_time_correct_l795_79541


namespace find_integer_for_combination_of_square_l795_79530

theorem find_integer_for_combination_of_square (y : ℝ) :
  ∃ (k : ℝ), (y^2 + 14*y + 60) = (y + 7)^2 + k ∧ k = 11 :=
by
  use 11
  sorry

end find_integer_for_combination_of_square_l795_79530


namespace find_fraction_l795_79526

theorem find_fraction (x y : ℤ) (h1 : x + 2 = y + 1) (h2 : 2 * (x + 4) = y + 2) : 
  x = -5 ∧ y = -4 := 
sorry

end find_fraction_l795_79526


namespace find_a_l795_79559

noncomputable def geometric_sum_expression (n : ℕ) (a : ℝ) : ℝ :=
  3 * 2^n + a

theorem find_a (a : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum_expression n a) → a = -3 :=
by
  sorry

end find_a_l795_79559


namespace ursula_annual_salary_l795_79554

def hourly_wage : ℝ := 8.50
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

noncomputable def daily_earnings : ℝ := hourly_wage * hours_per_day
noncomputable def monthly_earnings : ℝ := daily_earnings * days_per_month
noncomputable def annual_salary : ℝ := monthly_earnings * months_per_year

theorem ursula_annual_salary : annual_salary = 16320 := 
  by sorry

end ursula_annual_salary_l795_79554


namespace Bill_has_39_dollars_l795_79519

noncomputable def Frank_initial_money : ℕ := 42
noncomputable def pizza_cost : ℕ := 11
noncomputable def num_pizzas : ℕ := 3
noncomputable def Bill_initial_money : ℕ := 30

noncomputable def Frank_spent : ℕ := pizza_cost * num_pizzas
noncomputable def Frank_remaining_money : ℕ := Frank_initial_money - Frank_spent
noncomputable def Bill_final_money : ℕ := Bill_initial_money + Frank_remaining_money

theorem Bill_has_39_dollars :
  Bill_final_money = 39 :=
by
  sorry

end Bill_has_39_dollars_l795_79519


namespace only_nonneg_int_solution_l795_79521

theorem only_nonneg_int_solution (x y z : ℕ) (h : x^3 = 3 * y^3 + 9 * z^3) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end only_nonneg_int_solution_l795_79521


namespace systematic_sampling_first_number_l795_79574

theorem systematic_sampling_first_number
    (n : ℕ)  -- total number of products
    (k : ℕ)  -- sample size
    (common_diff : ℕ)  -- common difference in the systematic sample
    (x : ℕ)  -- an element in the sample
    (first_num : ℕ)  -- first product number in the sample
    (h1 : n = 80)  -- total number of products is 80
    (h2 : k = 5)  -- sample size is 5
    (h3 : common_diff = 16)  -- common difference is 16
    (h4 : x = 42)  -- 42 is in the sample
    (h5 : x = common_diff * 2 + first_num)  -- position of 42 in the arithmetic sequence
: first_num = 10 := 
sorry

end systematic_sampling_first_number_l795_79574


namespace scientific_notation_135000_l795_79527

theorem scientific_notation_135000 :
  135000 = 1.35 * 10^5 := sorry

end scientific_notation_135000_l795_79527


namespace lemon_heads_distribution_l795_79591

-- Conditions
def total_lemon_heads := 72
def number_of_friends := 6

-- Desired answer
def lemon_heads_per_friend := 12

-- Lean 4 statement
theorem lemon_heads_distribution : total_lemon_heads / number_of_friends = lemon_heads_per_friend := by 
  sorry

end lemon_heads_distribution_l795_79591


namespace security_to_bag_ratio_l795_79511

noncomputable def U_house : ℕ := 10
noncomputable def U_airport : ℕ := 5 * U_house
noncomputable def C_bag : ℕ := 15
noncomputable def W_boarding : ℕ := 20
noncomputable def W_takeoff : ℕ := 2 * W_boarding
noncomputable def T_total : ℕ := 180
noncomputable def T_known : ℕ := U_house + U_airport + C_bag + W_boarding + W_takeoff
noncomputable def T_security : ℕ := T_total - T_known

theorem security_to_bag_ratio : T_security / C_bag = 3 :=
by sorry

end security_to_bag_ratio_l795_79511


namespace evaluate_powers_of_i_l795_79501

-- Define complex number "i"
def i := Complex.I

-- Define the theorem to prove
theorem evaluate_powers_of_i : i^44 + i^444 + 3 = 5 := by
  -- use the cyclic property of i to simplify expressions
  sorry

end evaluate_powers_of_i_l795_79501


namespace find_m_l795_79562

-- Define the vector
def vec2 := (ℝ × ℝ)

-- Given vectors
def a : vec2 := (2, -1)
def c : vec2 := (-1, 2)

-- Definition of parallel vectors
def parallel (v1 v2 : vec2) := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Problem Statement
theorem find_m (m : ℝ) (b : vec2 := (-1, m)) (h : parallel (a.1 + b.1, a.2 + b.2) c) : m = -1 :=
sorry

end find_m_l795_79562


namespace sum_of_digits_6608_condition_l795_79532

theorem sum_of_digits_6608_condition :
  ∀ n1 n2 : ℕ, (6 * 1000 + n1 * 100 + n2 * 10 + 8) % 236 = 0 → n1 + n2 = 6 :=
by 
  intros n1 n2 h
  -- This is where the proof would go. Since we're not proving it, we skip it with "sorry".
  sorry

end sum_of_digits_6608_condition_l795_79532


namespace find_number_l795_79584

-- Definitions and conditions
def unknown_number (x : ℝ) : Prop :=
  (14 / 100) * x = 98

-- Theorem to prove
theorem find_number (x : ℝ) : unknown_number x → x = 700 := by
  sorry

end find_number_l795_79584


namespace diff_implies_continuous_l795_79566

def differentiable_imp_continuous (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀

-- Problem statement: if f is differentiable at x₀, then it is continuous at x₀.
theorem diff_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) : differentiable_imp_continuous f x₀ :=
by
  sorry

end diff_implies_continuous_l795_79566


namespace rubber_boat_fall_time_l795_79558

variable {a b x : ℝ}

theorem rubber_boat_fall_time
  (h1 : 5 - x = (a - b) / (a + b))
  (h2 : 6 - x = b / (a + b)) :
  x = 4 := by
  sorry

end rubber_boat_fall_time_l795_79558


namespace linda_savings_l795_79548

theorem linda_savings :
  let original_price_per_notebook := 3.75
  let discount_rate := 0.15
  let quantity := 12
  let total_price_without_discount := quantity * original_price_per_notebook
  let discount_amount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_amount_per_notebook
  let total_price_with_discount := quantity * discounted_price_per_notebook
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 6.75 :=
by {
  sorry
}

end linda_savings_l795_79548


namespace solve_x_floor_x_eq_72_l795_79578

theorem solve_x_floor_x_eq_72 : ∃ x : ℝ, 0 < x ∧ x * (⌊x⌋) = 72 ∧ x = 9 :=
by
  sorry

end solve_x_floor_x_eq_72_l795_79578


namespace simplify_negative_exponents_l795_79571

theorem simplify_negative_exponents (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ :=
  sorry

end simplify_negative_exponents_l795_79571


namespace red_flowers_count_l795_79536

theorem red_flowers_count (w r : ℕ) (h1 : w = 555) (h2 : w = r + 208) : r = 347 :=
by {
  -- Proof steps will be here
  sorry
}

end red_flowers_count_l795_79536


namespace round_trip_ticket_percentage_l795_79579

theorem round_trip_ticket_percentage (p : ℕ → Prop) : 
  (∀ n, p n → n = 375) → (∀ n, p n → n = 375) :=
by
  sorry

end round_trip_ticket_percentage_l795_79579


namespace final_bill_is_correct_l795_79585

def Alicia_order := [7.50, 4.00, 5.00]
def Brant_order := [10.00, 4.50, 6.00]
def Josh_order := [8.50, 4.00, 3.50]
def Yvette_order := [9.00, 4.50, 6.00]

def discount_rate := 0.10
def sales_tax_rate := 0.08
def tip_rate := 0.20

noncomputable def calculate_final_bill : Float :=
  let subtotal := (Alicia_order.sum + Brant_order.sum + Josh_order.sum + Yvette_order.sum)
  let discount := discount_rate * subtotal
  let discounted_total := subtotal - discount
  let sales_tax := sales_tax_rate * discounted_total
  let pre_tax_and_discount_total := subtotal
  let tip := tip_rate * pre_tax_and_discount_total
  discounted_total + sales_tax + tip

theorem final_bill_is_correct : calculate_final_bill = 84.97 := by
  sorry

end final_bill_is_correct_l795_79585


namespace floor_trig_sum_l795_79572

theorem floor_trig_sum :
  Int.floor (Real.sin 1) + Int.floor (Real.cos 2) + Int.floor (Real.tan 3) +
  Int.floor (Real.sin 4) + Int.floor (Real.cos 5) + Int.floor (Real.tan 6) = -4 := by
  sorry

end floor_trig_sum_l795_79572


namespace num_new_terms_in_sequence_l795_79510

theorem num_new_terms_in_sequence (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end num_new_terms_in_sequence_l795_79510


namespace least_number_subtracted_l795_79576

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted_l795_79576


namespace white_balls_count_l795_79561

theorem white_balls_count
  (total_balls : ℕ)
  (white_balls blue_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls + blue_balls + red_balls = total_balls)
  (h3 : blue_balls = white_balls + 12)
  (h4 : red_balls = 2 * blue_balls) : white_balls = 16 := by
  sorry

end white_balls_count_l795_79561


namespace isosceles_right_triangle_area_l795_79544

noncomputable def triangle_area (p : ℝ) : ℝ :=
  (1 / 8) * ((p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2)) ^ 2

theorem isosceles_right_triangle_area (p : ℝ) :
  let perimeter := p + p * Real.sqrt 2 + 2
  let x := (p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2) / 2
  let area := 1 / 2 * x ^ 2
  area = triangle_area p :=
by
  sorry

end isosceles_right_triangle_area_l795_79544


namespace power_function_decreasing_l795_79589

theorem power_function_decreasing (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 < x → f x = (m^2 + m - 11) * x^(m - 1))
  (hm : m^2 + m - 11 > 0)
  (hm' : m - 1 < 0)
  (hx : 0 < 1):
  f (-1) = -1 := by 
sorry

end power_function_decreasing_l795_79589


namespace students_interested_both_l795_79567

/-- total students surveyed -/
def U : ℕ := 50

/-- students who liked watching table tennis matches -/
def A : ℕ := 35

/-- students who liked watching badminton matches -/
def B : ℕ := 30

/-- students not interested in either -/
def nU_not_interest : ℕ := 5

theorem students_interested_both : (A + B - (U - nU_not_interest)) = 20 :=
by sorry

end students_interested_both_l795_79567


namespace g_difference_l795_79593

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 3) * (n + 5) + 2

theorem g_difference (s : ℕ) : g s - g (s - 1) = (3 * s^2 + 9 * s + 8) / 4 :=
by
  -- skip the proof
  sorry

end g_difference_l795_79593


namespace complete_the_square_l795_79534

-- Define the quadratic expression as a function.
def quad_expr (k : ℚ) : ℚ := 8 * k^2 + 12 * k + 18

-- Define the completed square form.
def completed_square_expr (k : ℚ) : ℚ := 8 * (k + 3 / 4)^2 + 27 / 2

-- Theorem stating the equality of the original expression in completed square form and the value of r + s.
theorem complete_the_square : ∀ k : ℚ, quad_expr k = completed_square_expr k ∧ (3 / 4 + 27 / 2 = 57 / 4) :=
by
  intro k
  sorry

end complete_the_square_l795_79534


namespace square_area_l795_79555

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the side length of the square based on the arrangement of circles
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- State the theorem to prove the area of the square
theorem square_area : (square_side_length * square_side_length) = 144 :=
by
  sorry

end square_area_l795_79555


namespace find_number_of_students_l795_79508

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students_l795_79508


namespace green_red_socks_ratio_l795_79568

theorem green_red_socks_ratio 
  (r : ℕ) -- Number of pairs of red socks originally ordered
  (y : ℕ) -- Price per pair of red socks
  (green_socks_price : ℕ := 3 * y) -- Price per pair of green socks, 3 times the red socks
  (C_original : ℕ := 6 * green_socks_price + r * y) -- Cost of the original order
  (C_interchanged : ℕ := r * green_socks_price + 6 * y) -- Cost of the interchanged order
  (exchange_rate : ℚ := 1.2) -- 20% increase
  (cost_relation : C_interchanged = exchange_rate * C_original) -- Cost relation given by the problem
  : (6 : ℚ) / (r : ℚ) = 2 / 3 := 
by
  sorry

end green_red_socks_ratio_l795_79568


namespace least_possible_value_l795_79540

theorem least_possible_value (x y z : ℕ) (hx : 2 * x = 5 * y) (hy : 5 * y = 8 * z) (hz : 8 * z = 2 * x) (hnz_x: x > 0) (hnz_y: y > 0) (hnz_z: z > 0) :
  x + y + z = 33 :=
sorry

end least_possible_value_l795_79540


namespace ratio_square_l795_79503

theorem ratio_square (x y : ℕ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) (h3 : 2 * y = 3 * x) : (x + y) ^ 2 = 100 := 
by 
  sorry

end ratio_square_l795_79503


namespace value_of_expression_l795_79538

def x : ℝ := 12
def y : ℝ := 7

theorem value_of_expression : (x - y) * (x + y) = 95 := by
  sorry

end value_of_expression_l795_79538


namespace solution_set_l795_79552

def f (x : ℝ) : ℝ := abs x - x + 1

theorem solution_set (x : ℝ) : f (1 - x^2) > f (1 - 2 * x) ↔ x > 2 ∨ x < -1 := by
  sorry

end solution_set_l795_79552


namespace number_of_positions_forming_cube_with_missing_face_l795_79580

-- Define the polygon formed by 6 congruent squares in a cross shape
inductive Square
| center : Square
| top : Square
| bottom : Square
| left : Square
| right : Square

-- Define the indices for the additional square positions
inductive Position
| pos1 : Position
| pos2 : Position
| pos3 : Position
| pos4 : Position
| pos5 : Position
| pos6 : Position
| pos7 : Position
| pos8 : Position
| pos9 : Position
| pos10 : Position
| pos11 : Position

-- Define a function that takes a position and returns whether the polygon can form the missing-face cube
def can_form_cube_missing_face : Position → Bool
  | Position.pos1   => true
  | Position.pos2   => true
  | Position.pos3   => true
  | Position.pos4   => true
  | Position.pos5   => false
  | Position.pos6   => false
  | Position.pos7   => false
  | Position.pos8   => false
  | Position.pos9   => true
  | Position.pos10  => true
  | Position.pos11  => true

-- Count valid positions for forming the cube with one face missing
def count_valid_positions : Nat :=
  List.length (List.filter can_form_cube_missing_face 
    [Position.pos1, Position.pos2, Position.pos3, Position.pos4, Position.pos5, Position.pos6, Position.pos7, Position.pos8, Position.pos9, Position.pos10, Position.pos11])

-- Prove that the number of valid positions is 7
theorem number_of_positions_forming_cube_with_missing_face : count_valid_positions = 7 :=
  by
    -- Implementation of the proof
    sorry

end number_of_positions_forming_cube_with_missing_face_l795_79580


namespace smallest_distance_zero_l795_79553

theorem smallest_distance_zero :
  let r_track (t : ℝ) := (Real.cos t, Real.sin t)
  let i_track (t : ℝ) := (Real.cos (t / 2), Real.sin (t / 2))
  ∀ t₁ t₂ : ℝ, dist (r_track t₁) (i_track t₂) = 0 := by
  sorry

end smallest_distance_zero_l795_79553


namespace age_problem_l795_79513

theorem age_problem (x y : ℕ) (h1 : y - 5 = 2 * (x - 5)) (h2 : x + y + 16 = 50) : x = 13 :=
by sorry

end age_problem_l795_79513


namespace min_xy_l795_79565

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
by sorry

end min_xy_l795_79565


namespace f_l795_79570

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x

-- Define the derivative f'(x)
def f' (a b x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x - 1

-- Problem statement: Prove that f'(-1) = -5 given the conditions
theorem f'_neg_one_value (a b : ℝ) (h : f' a b 1 = 3) : f' a b (-1) = -5 :=
by
  -- Placeholder for the proof
  sorry

end f_l795_79570


namespace complex_number_identity_l795_79551

theorem complex_number_identity (m : ℝ) (h : m + ((m ^ 2 - 4) * Complex.I) = Complex.re 0 + 1 * Complex.I ↔ m > 0): 
  (Complex.mk m 2 * Complex.mk 2 (-2)⁻¹) = Complex.I := sorry

end complex_number_identity_l795_79551


namespace gcd_a_b_l795_79509

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l795_79509


namespace product_of_four_integers_l795_79594

theorem product_of_four_integers:
  ∃ (A B C D : ℚ) (x : ℚ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧
  A + B + C + D = 40 ∧
  A - 3 = x ∧ B + 3 = x ∧ C / 2 = x ∧ D * 2 = x ∧
  A * B * C * D = (9089600 / 6561) := by
  sorry

end product_of_four_integers_l795_79594


namespace number_of_students_in_the_course_l795_79564

variable (T : ℝ)

theorem number_of_students_in_the_course
  (h1 : (1/5) * T + (1/4) * T + (1/2) * T + 40 = T) :
  T = 800 :=
sorry

end number_of_students_in_the_course_l795_79564


namespace teacher_age_l795_79542

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_total : ℕ) (num_total : ℕ) (h1 : avg_age_students = 21) (h2 : num_students = 20) (h3 : avg_age_total = 22) (h4 : num_total = 21) :
  let total_age_students := avg_age_students * num_students
  let total_age_class := avg_age_total * num_total
  let teacher_age := total_age_class - total_age_students
  teacher_age = 42 :=
by
  sorry

end teacher_age_l795_79542


namespace how_many_large_glasses_l795_79588

theorem how_many_large_glasses (cost_small cost_large : ℕ) 
                               (total_money money_left change : ℕ) 
                               (num_small : ℕ) : 
  cost_small = 3 -> 
  cost_large = 5 -> 
  total_money = 50 -> 
  money_left = 26 ->
  change = 1 ->
  num_small = 8 ->
  (money_left - change) / cost_large = 5 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end how_many_large_glasses_l795_79588


namespace problem1_problem2_l795_79587

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : x > -1) (m : ℕ) (hm : 0 < m) : 
  (1 + x)^m ≥ 1 + m * x :=
sorry

end problem1_problem2_l795_79587


namespace final_solution_concentration_l795_79550

def concentration (mass : ℕ) (volume : ℕ) : ℕ := 
  (mass * 100) / volume

theorem final_solution_concentration :
  let volume1 := 4
  let conc1 := 4 -- percentage
  let volume2 := 2
  let conc2 := 10 -- percentage
  let mass1 := volume1 * conc1 / 100
  let mass2 := volume2 * conc2 / 100
  let total_mass := mass1 + mass2
  let total_volume := volume1 + volume2
  concentration total_mass total_volume = 6 :=
by
  sorry

end final_solution_concentration_l795_79550


namespace perfect_square_trinomial_iff_l795_79518

theorem perfect_square_trinomial_iff (m : ℤ) :
  (∃ a b : ℤ, 4 = a^2 ∧ 121 = b^2 ∧ (4 = a^2 ∧ 121 = b^2) ∧ m = 2 * a * b ∨ m = -2 * a * b) ↔ (m = 44 ∨ m = -44) :=
by sorry

end perfect_square_trinomial_iff_l795_79518


namespace percentage_solution_P_mixture_l795_79525

-- Define constants for volumes and percentages
variables (P Q : ℝ)

-- Define given conditions
def percentage_lemonade_P : ℝ := 0.2
def percentage_carbonated_P : ℝ := 0.8
def percentage_lemonade_Q : ℝ := 0.45
def percentage_carbonated_Q : ℝ := 0.55
def percentage_carbonated_mixture : ℝ := 0.72

-- Prove that the percentage of the volume of the mixture that is Solution P is 68%
theorem percentage_solution_P_mixture : 
  (percentage_carbonated_P * P + percentage_carbonated_Q * Q = percentage_carbonated_mixture * (P + Q)) → 
  ((P / (P + Q)) * 100 = 68) :=
by
  -- proof skipped
  sorry

end percentage_solution_P_mixture_l795_79525


namespace trigonometric_identity_l795_79514

theorem trigonometric_identity :
  (Real.sin (18 * Real.pi / 180) * Real.sin (78 * Real.pi / 180)) -
  (Real.cos (162 * Real.pi / 180) * Real.cos (78 * Real.pi / 180)) = 1 / 2 := by
  sorry

end trigonometric_identity_l795_79514


namespace inequality_semi_perimeter_l795_79583

variables {R r p : Real}

theorem inequality_semi_perimeter (h1 : 0 < R) (h2 : 0 < r) (h3 : 0 < p) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 :=
sorry

end inequality_semi_perimeter_l795_79583


namespace sqrt_16_eq_plus_minus_4_l795_79563

theorem sqrt_16_eq_plus_minus_4 : ∀ x : ℝ, (x^2 = 16) ↔ (x = 4 ∨ x = -4) :=
by sorry

end sqrt_16_eq_plus_minus_4_l795_79563


namespace john_draw_on_back_l795_79595

theorem john_draw_on_back (total_pictures front_pictures : ℕ) (h1 : total_pictures = 15) (h2 : front_pictures = 6) : total_pictures - front_pictures = 9 :=
  by
  sorry

end john_draw_on_back_l795_79595


namespace tournament_ranking_sequences_l795_79504

def total_fair_ranking_sequences (A B C D : Type) : Nat :=
  let saturday_outcomes := 2
  let sunday_outcomes := 4 -- 2 possibilities for (first, second) and 2 for (third, fourth)
  let tiebreaker_effect := 2 -- swap second and third
  saturday_outcomes * sunday_outcomes * tiebreaker_effect

theorem tournament_ranking_sequences (A B C D : Type) :
  total_fair_ranking_sequences A B C D = 32 := 
by
  sorry

end tournament_ranking_sequences_l795_79504


namespace union_of_sets_l795_79577

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets_l795_79577


namespace find_side_c_and_area_S_find_sinA_plus_cosB_l795_79500

-- Definitions for the conditions given
structure Triangle :=
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)

noncomputable def givenTriangle : Triangle :=
  { a := 2, b := 4, c := 2 * Real.sqrt 3, angleA := 30, angleB := 90, angleC := 60 }

-- Prove the length of side c and the area S
theorem find_side_c_and_area_S (t : Triangle) (h : t = givenTriangle) :
  t.c = 2 * Real.sqrt 3 ∧ (1 / 2) * t.a * t.b * Real.sin (t.angleC * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

-- Prove the value of sin A + cos B
theorem find_sinA_plus_cosB (t : Triangle) (h : t = givenTriangle) :
  Real.sin (t.angleA * Real.pi / 180) + Real.cos (t.angleB * Real.pi / 180) = 1 / 2 :=
by
  sorry

end find_side_c_and_area_S_find_sinA_plus_cosB_l795_79500


namespace carter_siblings_oldest_age_l795_79520

theorem carter_siblings_oldest_age
    (avg_age : ℕ)
    (sibling1 : ℕ)
    (sibling2 : ℕ)
    (sibling3 : ℕ)
    (sibling4 : ℕ) :
    avg_age = 9 →
    sibling1 = 5 →
    sibling2 = 8 →
    sibling3 = 7 →
    ((sibling1 + sibling2 + sibling3 + sibling4) / 4) = avg_age →
    sibling4 = 16 := by
  intros
  sorry

end carter_siblings_oldest_age_l795_79520


namespace parallelogram_base_length_l795_79560

theorem parallelogram_base_length :
  ∀ (A H : ℝ), (A = 480) → (H = 15) → (A = Base * H) → (Base = 32) := 
by 
  intros A H hA hH hArea 
  sorry

end parallelogram_base_length_l795_79560


namespace bird_family_problem_l795_79524

def initial_bird_families (f s i : Nat) : Prop :=
  i = f + s

theorem bird_family_problem : initial_bird_families 32 35 67 :=
by
  -- Proof would go here
  sorry

end bird_family_problem_l795_79524


namespace jackson_points_l795_79502

theorem jackson_points (team_total_points : ℕ) (other_players_count : ℕ) (other_players_avg_score : ℕ) 
  (total_points_by_team : team_total_points = 72) 
  (total_points_by_others : other_players_count = 7) 
  (avg_points_by_others : other_players_avg_score = 6) :
  ∃ points_by_jackson : ℕ, points_by_jackson = 30 :=
by
  sorry

end jackson_points_l795_79502


namespace value_of_a_plus_b_l795_79592

theorem value_of_a_plus_b (a b : ℝ) (h : (2 * a + 2 * b - 1) * (2 * a + 2 * b + 1) = 99) :
  a + b = 5 ∨ a + b = -5 :=
sorry

end value_of_a_plus_b_l795_79592


namespace minimize_distance_school_l795_79556

-- Define the coordinates for the towns X, Y, and Z
def X_coord : ℕ × ℕ := (0, 0)
def Y_coord : ℕ × ℕ := (200, 0)
def Z_coord : ℕ × ℕ := (0, 300)

-- Define the population of the towns
def X_population : ℕ := 100
def Y_population : ℕ := 200
def Z_population : ℕ := 300

theorem minimize_distance_school : ∃ (x y : ℕ), x + y = 300 := by
  -- This should follow from the problem setup and conditions.
  sorry

end minimize_distance_school_l795_79556


namespace pairs_satisfying_condition_l795_79573

theorem pairs_satisfying_condition (x y : ℤ) (h : x + y ≠ 0) :
  (x^2 + y^2)/(x + y) = 10 ↔ (x, y) = (12, 6) ∨ (x, y) = (-2, 6) ∨ (x, y) = (12, 4) ∨ (x, y) = (-2, 4) ∨ (x, y) = (10, 10) ∨ (x, y) = (0, 10) ∨ (x, y) = (10, 0) :=
sorry

end pairs_satisfying_condition_l795_79573


namespace system_of_equations_l795_79597

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l795_79597


namespace greatest_value_of_x_for_7x_factorial_100_l795_79537

open Nat

theorem greatest_value_of_x_for_7x_factorial_100 : 
  ∃ x : ℕ, (∀ y : ℕ, 7^y ∣ factorial 100 → y ≤ x) ∧ x = 16 :=
by
  sorry

end greatest_value_of_x_for_7x_factorial_100_l795_79537


namespace total_bike_price_l795_79543

theorem total_bike_price 
  (marion_bike_cost : ℝ := 356)
  (stephanie_bike_base_cost : ℝ := 2 * marion_bike_cost)
  (stephanie_discount_rate : ℝ := 0.10)
  (patrick_bike_base_cost : ℝ := 3 * marion_bike_cost)
  (patrick_discount_rate : ℝ := 0.75)
  (stephanie_bike_cost : ℝ := stephanie_bike_base_cost * (1 - stephanie_discount_rate))
  (patrick_bike_cost : ℝ := patrick_bike_base_cost * patrick_discount_rate):
  marion_bike_cost + stephanie_bike_cost + patrick_bike_cost = 1797.80 := 
by 
  sorry

end total_bike_price_l795_79543


namespace dave_shirts_not_washed_l795_79529

variable (short_sleeve_shirts long_sleeve_shirts washed_shirts : ℕ)

theorem dave_shirts_not_washed (h1 : short_sleeve_shirts = 9) (h2 : long_sleeve_shirts = 27) (h3 : washed_shirts = 20) :
  (short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 16) :=
by {
  -- sorry indicates the proof is omitted
  sorry
}

end dave_shirts_not_washed_l795_79529


namespace arcsin_arccos_eq_l795_79596

theorem arcsin_arccos_eq (x : ℝ) (h : Real.arcsin x + Real.arcsin (2 * x - 1) = Real.arccos x) : x = 1 := by
  sorry

end arcsin_arccos_eq_l795_79596


namespace range_a_of_function_has_two_zeros_l795_79528

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem range_a_of_function_has_two_zeros (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) : 
  1 < a :=
sorry

end range_a_of_function_has_two_zeros_l795_79528


namespace number_of_cans_on_third_day_l795_79539

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem number_of_cans_on_third_day :
  (arithmetic_sequence 4 5 2 = 9) →   -- on the second day, he found 9 cans
  (arithmetic_sequence 4 5 7 = 34) →  -- on the seventh day, he found 34 cans
  (arithmetic_sequence 4 5 3 = 14) :=  -- therefore, on the third day, he found 14 cans
by
  intros h1 h2
  sorry

end number_of_cans_on_third_day_l795_79539


namespace river_current_speed_l795_79599

theorem river_current_speed 
  (downstream_distance upstream_distance still_water_speed : ℝ)
  (H1 : still_water_speed = 20)
  (H2 : downstream_distance = 100)
  (H3 : upstream_distance = 60)
  (H4 : (downstream_distance / (still_water_speed + x)) = (upstream_distance / (still_water_speed - x)))
  : x = 5 :=
by
  sorry

end river_current_speed_l795_79599
