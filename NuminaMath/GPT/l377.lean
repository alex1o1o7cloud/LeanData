import Mathlib

namespace NUMINAMATH_GPT_min_value_m_plus_n_l377_37783

theorem min_value_m_plus_n (m n : ℕ) (h : 108 * m = n^3) (hm : 0 < m) (hn : 0 < n) : m + n = 8 :=
sorry

end NUMINAMATH_GPT_min_value_m_plus_n_l377_37783


namespace NUMINAMATH_GPT_f_eq_32x5_l377_37764

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end NUMINAMATH_GPT_f_eq_32x5_l377_37764


namespace NUMINAMATH_GPT_population_after_4_years_l377_37747

theorem population_after_4_years 
  (initial_population : ℕ) 
  (new_people : ℕ) 
  (people_moved_out : ℕ) 
  (years : ℕ) 
  (final_population : ℕ) :
  initial_population = 780 →
  new_people = 100 →
  people_moved_out = 400 →
  years = 4 →
  final_population = initial_population + new_people - people_moved_out →
  final_population / 2 / 2 / 2 / 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_population_after_4_years_l377_37747


namespace NUMINAMATH_GPT_cubic_geometric_progression_l377_37765

theorem cubic_geometric_progression (a b c : ℝ) (α β γ : ℝ) 
    (h_eq1 : α + β + γ = -a) 
    (h_eq2 : α * β + α * γ + β * γ = b) 
    (h_eq3 : α * β * γ = -c) 
    (h_gp : ∃ k q : ℝ, α = k / q ∧ β = k ∧ γ = k * q) : 
    a^3 * c - b^3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_geometric_progression_l377_37765


namespace NUMINAMATH_GPT_tan_of_right_triangle_l377_37700

theorem tan_of_right_triangle (A B C : ℝ) (h : A^2 + B^2 = C^2) (hA : A = 30) (hC : C = 37) : 
  (37^2 - 30^2).sqrt / 30 = (469).sqrt / 30 := by
  sorry

end NUMINAMATH_GPT_tan_of_right_triangle_l377_37700


namespace NUMINAMATH_GPT_vector_parallel_l377_37755

theorem vector_parallel (x : ℝ) : ∃ x, (1, x) = k * (-2, 3) → x = -3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_vector_parallel_l377_37755


namespace NUMINAMATH_GPT_solve_divisor_problem_l377_37744

def divisor_problem : Prop :=
  ∃ D : ℕ, 12401 = (D * 76) + 13 ∧ D = 163

theorem solve_divisor_problem : divisor_problem :=
sorry

end NUMINAMATH_GPT_solve_divisor_problem_l377_37744


namespace NUMINAMATH_GPT_geometric_sequence_n_l377_37730

theorem geometric_sequence_n (a1 an q : ℚ) (n : ℕ) (h1 : a1 = 9 / 8) (h2 : an = 1 / 3) (h3 : q = 2 / 3) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_n_l377_37730


namespace NUMINAMATH_GPT_carrots_eaten_after_dinner_l377_37711

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end NUMINAMATH_GPT_carrots_eaten_after_dinner_l377_37711


namespace NUMINAMATH_GPT_reduction_for_same_profit_cannot_reach_460_profit_l377_37773

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end NUMINAMATH_GPT_reduction_for_same_profit_cannot_reach_460_profit_l377_37773


namespace NUMINAMATH_GPT_greatest_integer_jean_thinks_of_l377_37795

theorem greatest_integer_jean_thinks_of :
  ∃ n : ℕ, n < 150 ∧ (∃ a : ℤ, n + 2 = 9 * a) ∧ (∃ b : ℤ, n + 3 = 11 * b) ∧ n = 142 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_jean_thinks_of_l377_37795


namespace NUMINAMATH_GPT_power_sum_l377_37706

theorem power_sum : 1 ^ 2009 + (-1) ^ 2009 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_power_sum_l377_37706


namespace NUMINAMATH_GPT_find_xyz_area_proof_l377_37758

-- Conditions given in the problem
variable (x y z : ℝ)
-- Side lengths derived from condition of inscribed circle
def conditions :=
  (x + y = 5) ∧
  (x + z = 6) ∧
  (y + z = 8)

-- The proof problem: Show the relationships between x, y, and z given the side lengths
theorem find_xyz_area_proof (h : conditions x y z) :
  (z - y = 1) ∧ (z - x = 3) ∧ (z = 4.5) ∧ (x = 1.5) ∧ (y = 3.5) :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_area_proof_l377_37758


namespace NUMINAMATH_GPT_num_ways_to_distribute_balls_l377_37714

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_num_ways_to_distribute_balls_l377_37714


namespace NUMINAMATH_GPT_parabola_focus_l377_37778

theorem parabola_focus (p : ℝ) (hp : 0 < p) (h : ∀ y x : ℝ, y^2 = 2 * p * x → (x = 2 ∧ y = 0)) : p = 4 :=
sorry

end NUMINAMATH_GPT_parabola_focus_l377_37778


namespace NUMINAMATH_GPT_range_of_a_l377_37790

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → (a ≤ -1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l377_37790


namespace NUMINAMATH_GPT_age_difference_l377_37777

theorem age_difference (Rona Rachel Collete : ℕ) (h1 : Rachel = 2 * Rona) (h2 : Collete = Rona / 2) (h3 : Rona = 8) : Rachel - Collete = 12 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l377_37777


namespace NUMINAMATH_GPT_probability_all_white_is_correct_l377_37799

-- Define the total number of balls
def total_balls : ℕ := 25

-- Define the number of white balls
def white_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 15

-- Define the number of balls drawn
def balls_drawn : ℕ := 4

-- Define combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to choose 4 balls from 25
def total_ways : ℕ := C total_balls balls_drawn

-- Ways to choose 4 white balls from 10 white balls
def white_ways : ℕ := C white_balls balls_drawn

-- Probability that all 4 drawn balls are white
def prob_all_white : ℚ := white_ways / total_ways

theorem probability_all_white_is_correct :
  prob_all_white = (3 : ℚ) / 181 := by
  -- Proof statements go here
  sorry

end NUMINAMATH_GPT_probability_all_white_is_correct_l377_37799


namespace NUMINAMATH_GPT_solve_for_y_l377_37776

theorem solve_for_y : ∃ y : ℝ, y = -2 ∧ y^2 + 6 * y + 8 = -(y + 2) * (y + 6) :=
by
  use -2
  sorry

end NUMINAMATH_GPT_solve_for_y_l377_37776


namespace NUMINAMATH_GPT_cost_of_pears_l377_37748

theorem cost_of_pears (P : ℕ)
  (apples_cost : ℕ := 40)
  (dozens : ℕ := 14)
  (total_cost : ℕ := 1260)
  (h_p : dozens * P + dozens * apples_cost = total_cost) :
  P = 50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pears_l377_37748


namespace NUMINAMATH_GPT_area_of_regular_inscribed_polygon_f3_properties_of_f_l377_37710

noncomputable def f (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (n / 2) * Real.sin (2 * Real.pi / n) else 0

theorem area_of_regular_inscribed_polygon_f3 :
  f 3 = (3 * Real.sqrt 3) / 4 :=
by
  sorry

theorem properties_of_f (n : ℕ) (hn : n ≥ 3) :
  (f n = (n / 2) * Real.sin (2 * Real.pi / n)) ∧
  (f n < f (n + 1)) ∧ 
  (f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n) :=
by
  sorry

end NUMINAMATH_GPT_area_of_regular_inscribed_polygon_f3_properties_of_f_l377_37710


namespace NUMINAMATH_GPT_candy_amount_in_peanut_butter_jar_l377_37782

-- Definitions of the candy amounts in each jar
def banana_jar := 43
def grape_jar := banana_jar + 5
def peanut_butter_jar := 4 * grape_jar
def coconut_jar := (3 / 2) * banana_jar

-- The statement we need to prove
theorem candy_amount_in_peanut_butter_jar : peanut_butter_jar = 192 := by
  sorry

end NUMINAMATH_GPT_candy_amount_in_peanut_butter_jar_l377_37782


namespace NUMINAMATH_GPT_externally_tangent_circles_proof_l377_37723

noncomputable def externally_tangent_circles (r r' : ℝ) (φ : ℝ) : Prop :=
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r')

theorem externally_tangent_circles_proof (r r' φ : ℝ) 
  (h1: r > 0) (h2: r' > 0) (h3: φ ≥ 0 ∧ φ ≤ π) : 
  externally_tangent_circles r r' φ :=
sorry

end NUMINAMATH_GPT_externally_tangent_circles_proof_l377_37723


namespace NUMINAMATH_GPT_distinguishable_large_equilateral_triangles_l377_37793

-- Definitions based on conditions.
def num_colors : ℕ := 8

def same_color_corners : ℕ := num_colors
def two_same_one_diff_colors : ℕ := num_colors * (num_colors - 1)
def all_diff_colors : ℕ := (num_colors * (num_colors - 1) * (num_colors - 2)) / 6

def corner_configurations : ℕ := same_color_corners + two_same_one_diff_colors + all_diff_colors
def triangle_between_center_and_corner : ℕ := num_colors
def center_triangle : ℕ := num_colors

def total_distinguishable_triangles : ℕ := corner_configurations * triangle_between_center_and_corner * center_triangle

theorem distinguishable_large_equilateral_triangles : total_distinguishable_triangles = 7680 :=
by
  sorry

end NUMINAMATH_GPT_distinguishable_large_equilateral_triangles_l377_37793


namespace NUMINAMATH_GPT_fifth_hexagon_dots_l377_37737

-- Definitions as per conditions
def dots_in_nth_layer (n : ℕ) : ℕ := 6 * (n + 2)

-- Function to calculate the total number of dots in the nth hexagon
def total_dots_in_hexagon (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + dots_in_nth_layer k) (dots_in_nth_layer 0)

-- The proof problem statement
theorem fifth_hexagon_dots : total_dots_in_hexagon 5 = 150 := sorry

end NUMINAMATH_GPT_fifth_hexagon_dots_l377_37737


namespace NUMINAMATH_GPT_ratio_of_mistakes_l377_37704

theorem ratio_of_mistakes (riley_mistakes team_mistakes : ℕ) 
  (h_riley : riley_mistakes = 3) (h_team : team_mistakes = 17) : 
  (team_mistakes - riley_mistakes) / riley_mistakes = 14 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_mistakes_l377_37704


namespace NUMINAMATH_GPT_janessa_initial_cards_l377_37745

theorem janessa_initial_cards (X : ℕ)  :
  (X + 45 = 49) →
  X = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_janessa_initial_cards_l377_37745


namespace NUMINAMATH_GPT_age_difference_l377_37772

theorem age_difference (J P : ℕ) 
  (h1 : P = 16 - 10) 
  (h2 : P = (1 / 3) * J) : 
  (J + 10) - 16 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_age_difference_l377_37772


namespace NUMINAMATH_GPT_minimum_sugar_amount_l377_37735

theorem minimum_sugar_amount (f s : ℕ) (h1 : f ≥ 9 + s / 2) (h2 : f ≤ 3 * s) : s ≥ 4 :=
by
  -- Provided conditions: f ≥ 9 + s / 2 and f ≤ 3 * s
  -- Goal: s ≥ 4
  sorry

end NUMINAMATH_GPT_minimum_sugar_amount_l377_37735


namespace NUMINAMATH_GPT_tank_plastering_cost_proof_l377_37798

/-- 
Given a tank with the following dimensions:
length = 35 meters,
width = 18 meters,
depth = 10 meters.
The cost of plastering per square meter is ₹135.
Prove that the total cost of plastering the walls and bottom of the tank is ₹228,150.
-/
theorem tank_plastering_cost_proof (length width depth cost_per_sq_meter : ℕ)
  (h_length : length = 35)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost_per_sq_meter : cost_per_sq_meter = 135) : 
  (2 * (length * depth) + 2 * (width * depth) + length * width) * cost_per_sq_meter = 228150 := 
by 
  -- The proof is not required as per the problem statement
  sorry

end NUMINAMATH_GPT_tank_plastering_cost_proof_l377_37798


namespace NUMINAMATH_GPT_value_of_g_g_2_l377_37753

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end NUMINAMATH_GPT_value_of_g_g_2_l377_37753


namespace NUMINAMATH_GPT_total_cost_of_toys_l377_37708

-- Define the costs of the yoyo and the whistle
def cost_yoyo : Nat := 24
def cost_whistle : Nat := 14

-- Prove the total cost of the yoyo and the whistle is 38 cents
theorem total_cost_of_toys : cost_yoyo + cost_whistle = 38 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_toys_l377_37708


namespace NUMINAMATH_GPT_negation_equiv_l377_37742
variable (x : ℝ)

theorem negation_equiv :
  (¬ ∃ x : ℝ, x^2 + 1 > 3 * x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3 * x) :=
by 
  sorry

end NUMINAMATH_GPT_negation_equiv_l377_37742


namespace NUMINAMATH_GPT_Morse_code_distinct_symbols_l377_37746

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end NUMINAMATH_GPT_Morse_code_distinct_symbols_l377_37746


namespace NUMINAMATH_GPT_revenue_difference_l377_37769

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end NUMINAMATH_GPT_revenue_difference_l377_37769


namespace NUMINAMATH_GPT_worker_b_days_l377_37767

variables (W_A W_B W : ℝ)
variables (h1 : W_A = 2 * W_B)
variables (h2 : (W_A + W_B) * 10 = W)
variables (h3 : W = 30 * W_B)

theorem worker_b_days : ∃ days : ℝ, days = 30 :=
by
  sorry

end NUMINAMATH_GPT_worker_b_days_l377_37767


namespace NUMINAMATH_GPT_basketball_free_throws_l377_37729

/-
Given the following conditions:
1. The players scored twice as many points with three-point shots as with two-point shots: \( 3b = 2a \).
2. The number of successful free throws was one more than the number of successful two-point shots: \( x = a + 1 \).
3. The team’s total score was 84 points: \( 2a + 3b + x = 84 \).

Prove that the number of free throws \( x \) equals 16.
-/
theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = a + 1) 
  (h3 : 2 * a + 3 * b + x = 84) : 
  x = 16 := 
  sorry

end NUMINAMATH_GPT_basketball_free_throws_l377_37729


namespace NUMINAMATH_GPT_smallest_number_to_add_l377_37725

theorem smallest_number_to_add:
  ∃ x : ℕ, x = 119 ∧ (2714 + x) % 169 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_to_add_l377_37725


namespace NUMINAMATH_GPT_bear_hunting_l377_37779

theorem bear_hunting
    (mother_meat_req : ℕ) (cub_meat_req : ℕ) (num_cubs : ℕ) (num_animals_daily : ℕ)
    (weekly_meat_req : mother_meat_req = 210)
    (weekly_meat_per_cub : cub_meat_req = 35)
    (number_of_cubs : num_cubs = 4)
    (animals_hunted_daily : num_animals_daily = 10)
    (total_weekly_meat : mother_meat_req + num_cubs * cub_meat_req = 350) :
    ∃ w : ℕ, (w * num_animals_daily * 7 = 350) ∧ w = 5 :=
by
  sorry

end NUMINAMATH_GPT_bear_hunting_l377_37779


namespace NUMINAMATH_GPT_sale_price_after_discounts_l377_37709

def calculate_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

theorem sale_price_after_discounts :
  calculate_sale_price 500 [0.10, 0.15, 0.20, 0.25, 0.30] = 160.65 :=
by
  sorry

end NUMINAMATH_GPT_sale_price_after_discounts_l377_37709


namespace NUMINAMATH_GPT_round_trip_time_l377_37761

variable (boat_speed standing_water_speed stream_speed distance : ℕ)

theorem round_trip_time (boat_speed := 9) (stream_speed := 6) (distance := 170) : 
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed)) = 68 := by 
  sorry

end NUMINAMATH_GPT_round_trip_time_l377_37761


namespace NUMINAMATH_GPT_third_consecutive_even_integer_l377_37763

theorem third_consecutive_even_integer (n : ℤ) (h : (n + 2) + (n + 6) = 156) : (n + 4) = 78 :=
sorry

end NUMINAMATH_GPT_third_consecutive_even_integer_l377_37763


namespace NUMINAMATH_GPT_circle_has_greatest_symmetry_l377_37787

-- Definitions based on the conditions
def lines_of_symmetry (figure : String) : ℕ∞ := 
  match figure with
  | "regular pentagon" => 5
  | "isosceles triangle" => 1
  | "circle" => ⊤  -- Using the symbol ⊤ to represent infinity in Lean.
  | "rectangle" => 2
  | "parallelogram" => 0
  | _ => 0          -- default case

theorem circle_has_greatest_symmetry :
  ∃ fig, fig = "circle" ∧ ∀ other_fig, lines_of_symmetry fig ≥ lines_of_symmetry other_fig := 
by
  sorry

end NUMINAMATH_GPT_circle_has_greatest_symmetry_l377_37787


namespace NUMINAMATH_GPT_solve_system_eqs_l377_37768

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end NUMINAMATH_GPT_solve_system_eqs_l377_37768


namespace NUMINAMATH_GPT_quadratic_complete_square_l377_37739

theorem quadratic_complete_square : ∃ k : ℤ, ∀ x : ℤ, x^2 + 8*x + 22 = (x + 4)^2 + k :=
by
  use 6
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l377_37739


namespace NUMINAMATH_GPT_ship_passengers_percentage_l377_37789

variables (P R : ℝ)

-- Conditions
def condition1 : Prop := (0.20 * P) = (0.60 * R)

-- Target
def target : Prop := R / P = 1 / 3

theorem ship_passengers_percentage
  (h1 : condition1 P R) :
  target P R :=
by
  sorry

end NUMINAMATH_GPT_ship_passengers_percentage_l377_37789


namespace NUMINAMATH_GPT_syllogism_correct_l377_37771

-- Define that natural numbers are integers
axiom nat_is_int : ∀ (n : ℕ), ∃ (m : ℤ), m = n

-- Define that 4 is a natural number
axiom four_is_nat : ∃ (n : ℕ), n = 4

-- The syllogism's conclusion: 4 is an integer
theorem syllogism_correct : ∃ (m : ℤ), m = 4 :=
by
  have h1 := nat_is_int 4
  have h2 := four_is_nat
  exact h1

end NUMINAMATH_GPT_syllogism_correct_l377_37771


namespace NUMINAMATH_GPT_rectangle_area_l377_37717

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l377_37717


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l377_37703

theorem no_solution_for_inequalities (x : ℝ) : ¬(4 * x ^ 2 + 7 * x - 2 < 0 ∧ 3 * x - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l377_37703


namespace NUMINAMATH_GPT_cos_identity_proof_l377_37734

noncomputable def cos_eq_half : Prop :=
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7)) = 1 / 2

theorem cos_identity_proof : cos_eq_half :=
  by sorry

end NUMINAMATH_GPT_cos_identity_proof_l377_37734


namespace NUMINAMATH_GPT_gcd_of_B_l377_37736

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_B_l377_37736


namespace NUMINAMATH_GPT_jared_annual_salary_l377_37724

def monthly_salary_diploma_holder : ℕ := 4000
def factor_degree_to_diploma : ℕ := 3
def months_in_year : ℕ := 12

theorem jared_annual_salary :
  (factor_degree_to_diploma * monthly_salary_diploma_holder) * months_in_year = 144000 :=
by
  sorry

end NUMINAMATH_GPT_jared_annual_salary_l377_37724


namespace NUMINAMATH_GPT_coeff_matrix_correct_l377_37780

-- Define the system of linear equations as given conditions
def eq1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1
def eq2 (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the coefficient matrix
def coeffMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![1, -2]
]

-- The theorem stating that the coefficient matrix of the system is as defined
theorem coeff_matrix_correct (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : 
  coeffMatrix = ![
    ![2, 3],
    ![1, -2]
  ] :=
sorry

end NUMINAMATH_GPT_coeff_matrix_correct_l377_37780


namespace NUMINAMATH_GPT_superhero_speed_conversion_l377_37786

theorem superhero_speed_conversion
    (speed_km_per_min : ℕ)
    (conversion_factor : ℝ)
    (minutes_in_hour : ℕ)
    (H1 : speed_km_per_min = 1000)
    (H2 : conversion_factor = 0.6)
    (H3 : minutes_in_hour = 60) :
    (speed_km_per_min * conversion_factor * minutes_in_hour = 36000) :=
by
    sorry

end NUMINAMATH_GPT_superhero_speed_conversion_l377_37786


namespace NUMINAMATH_GPT_sector_radian_measure_l377_37731

theorem sector_radian_measure {r l : ℝ} 
  (h1 : 2 * r + l = 12) 
  (h2 : (1/2) * l * r = 8) : 
  (l / r = 1) ∨ (l / r = 4) :=
sorry

end NUMINAMATH_GPT_sector_radian_measure_l377_37731


namespace NUMINAMATH_GPT_num_solutions_l377_37766

theorem num_solutions (h : ∀ n : ℕ, (1 ≤ n ∧ n ≤ 455) → n^3 % 455 = 1) : 
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ (1 ≤ n ∧ n ≤ 455) ∧ n^3 % 455 = 1) ∧ s.card = 9) :=
sorry

end NUMINAMATH_GPT_num_solutions_l377_37766


namespace NUMINAMATH_GPT_natalia_crates_l377_37719

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end NUMINAMATH_GPT_natalia_crates_l377_37719


namespace NUMINAMATH_GPT_proof_problem_l377_37792

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x| > 1}

def B : Set ℝ := {x | (0 : ℝ) < x ∧ x ≤ 2}

def complement_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def intersection (s1 s2 : Set ℝ) : Set ℝ := s1 ∩ s2

theorem proof_problem : (complement_A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l377_37792


namespace NUMINAMATH_GPT_find_a2_plus_b2_l377_37781

theorem find_a2_plus_b2
  (a b : ℝ)
  (h1 : a^3 - 3 * a * b^2 = 39)
  (h2 : b^3 - 3 * a^2 * b = 26) :
  a^2 + b^2 = 13 :=
sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l377_37781


namespace NUMINAMATH_GPT_vector_relation_condition_l377_37707

variables {V : Type*} [AddCommGroup V] (OD OE OM DO EO MO : V)

-- Given condition
theorem vector_relation_condition (h : OD + OE = OM) :

-- Option B
(OM + DO = OE) ∧ 

-- Option C
(OM - OE = OD) ∧ 

-- Option D
(DO + EO = MO) :=
by {
  -- Sorry, to focus on statement only
  sorry
}

end NUMINAMATH_GPT_vector_relation_condition_l377_37707


namespace NUMINAMATH_GPT_problem1_problem2_l377_37756

-- Definitions and conditions:
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := (x^2 - 2 * x + 1 - m^2 ≤ 0) ∧ (m > 0)

-- Question (1) statement: Prove that if p is a sufficient condition for q, then m ≥ 4
theorem problem1 (p_implies_q : ∀ x : ℝ, p x → q x m) : m ≥ 4 := sorry

-- Question (2) statement: Prove that if m = 5 and p ∨ q is true but p ∧ q is false,
-- then the range of x is [-4, -1) ∪ (5, 6]
theorem problem2 (m_eq : m = 5) (p_or_q : ∃ x : ℝ, p x ∨ q x m) (p_and_not_q : ¬ (∃ x : ℝ, p x ∧ q x m)) :
  ∃ x : ℝ, (x < -1 ∧ -4 ≤ x) ∨ (5 < x ∧ x ≤ 6) := sorry

end NUMINAMATH_GPT_problem1_problem2_l377_37756


namespace NUMINAMATH_GPT_max_vector_sum_l377_37702

theorem max_vector_sum
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ := (2, 0))
  (hA : A.1^2 + A.2^2 = 1)
  (hB : B.1^2 + B.2^2 = 1)
  (hC : C.1^2 + C.2^2 = 1)
  (h_perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  |(2,0) - A + (2,0) - B + (2,0) - C| = 7 := sorry

end NUMINAMATH_GPT_max_vector_sum_l377_37702


namespace NUMINAMATH_GPT_jean_jail_time_l377_37785

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end NUMINAMATH_GPT_jean_jail_time_l377_37785


namespace NUMINAMATH_GPT_not_possible_to_list_numbers_l377_37727

theorem not_possible_to_list_numbers :
  ¬ (∃ (f : ℕ → ℕ), (∀ n, f n ≥ 1 ∧ f n ≤ 1963) ∧
                     (∀ n, Nat.gcd (f n) (f (n+1)) = 1) ∧
                     (∀ n, Nat.gcd (f n) (f (n+2)) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_list_numbers_l377_37727


namespace NUMINAMATH_GPT_proof_problem_l377_37701

variables (a b : ℝ)

noncomputable def expr := (2 * a⁻¹ + (a⁻¹ / b)) / a

theorem proof_problem (h1 : a = 1/3) (h2 : b = 3) : expr a b = 21 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l377_37701


namespace NUMINAMATH_GPT_sector_area_l377_37733

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 6) (h_α : α = π / 3) : (1 / 2) * (α * r) * r = 6 * π :=
by
  rw [h_r, h_α]
  sorry

end NUMINAMATH_GPT_sector_area_l377_37733


namespace NUMINAMATH_GPT_range_of_a_second_quadrant_l377_37788

theorem range_of_a_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0) → x < 0 ∧ y > 0) →
  a > 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_second_quadrant_l377_37788


namespace NUMINAMATH_GPT_product_of_two_numbers_l377_37712

-- Define HCF (Highest Common Factor) and LCM (Least Common Multiple) conditions
def hcf_of_two_numbers (a b : ℕ) : ℕ := 11
def lcm_of_two_numbers (a b : ℕ) : ℕ := 181

-- The theorem to prove
theorem product_of_two_numbers (a b : ℕ) 
  (h1 : hcf_of_two_numbers a b = 11)
  (h2 : lcm_of_two_numbers a b = 181) : 
  a * b = 1991 :=
by 
  -- This is where we would put the proof, but we can use sorry for now
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l377_37712


namespace NUMINAMATH_GPT_mean_equivalence_l377_37749

theorem mean_equivalence :
  (20 + 30 + 40) / 3 = (23 + 30 + 37) / 3 :=
by sorry

end NUMINAMATH_GPT_mean_equivalence_l377_37749


namespace NUMINAMATH_GPT_baseball_fans_count_l377_37784

theorem baseball_fans_count
  (Y M R : ℕ) 
  (h1 : Y = (3 * M) / 2)
  (h2 : R = (5 * M) / 4)
  (hM : M = 104) :
  Y + M + R = 390 :=
by
  sorry 

end NUMINAMATH_GPT_baseball_fans_count_l377_37784


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l377_37728

theorem cost_of_adult_ticket (x : ℕ) (total_persons : ℕ) (total_collected : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) (amount_from_children : ℕ) :
  total_persons = 280 →
  total_collected = 14000 →
  adult_tickets = 200 →
  child_ticket_cost = 25 →
  amount_from_children = 2000 →
  200 * x + amount_from_children = total_collected →
  x = 60 :=
by
  intros h_persons h_total h_adults h_child_cost h_children_amount h_eq
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_l377_37728


namespace NUMINAMATH_GPT_smallest_multiple_of_4_and_14_is_28_l377_37762

theorem smallest_multiple_of_4_and_14_is_28 :
  ∃ (a : ℕ), a > 0 ∧ (4 ∣ a) ∧ (14 ∣ a) ∧ ∀ b : ℕ, b > 0 → (4 ∣ b) → (14 ∣ b) → a ≤ b := 
sorry

end NUMINAMATH_GPT_smallest_multiple_of_4_and_14_is_28_l377_37762


namespace NUMINAMATH_GPT_probability_both_hit_l377_37743

-- Define the probabilities of hitting the target for shooters A and B.
def prob_A_hits : ℝ := 0.7
def prob_B_hits : ℝ := 0.8

-- Define the independence condition (not needed as a direct definition but implicitly acknowledges independence).
axiom A_and_B_independent : true

-- The statement we want to prove: the probability that both shooters hit the target.
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.56 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_probability_both_hit_l377_37743


namespace NUMINAMATH_GPT_total_spent_correct_l377_37796

def cost_gifts : ℝ := 561.00
def cost_giftwrapping : ℝ := 139.00
def total_spent : ℝ := cost_gifts + cost_giftwrapping

theorem total_spent_correct : total_spent = 700.00 := by
  sorry

end NUMINAMATH_GPT_total_spent_correct_l377_37796


namespace NUMINAMATH_GPT_gcd_values_count_l377_37740

noncomputable def count_gcd_values (a b : ℕ) : ℕ :=
  if (a * b = 720 ∧ a + b = 50) then 1 else 0

theorem gcd_values_count : 
  (∃ a b : ℕ, a * b = 720 ∧ a + b = 50) → count_gcd_values a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_values_count_l377_37740


namespace NUMINAMATH_GPT_total_birds_and_storks_l377_37741

theorem total_birds_and_storks
  (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ)
  (hb : initial_birds = 3) (hs : initial_storks = 4) (has : additional_storks = 6) :
  initial_birds + (initial_storks + additional_storks) = 13 :=
by
  sorry

end NUMINAMATH_GPT_total_birds_and_storks_l377_37741


namespace NUMINAMATH_GPT_boys_to_girls_ratio_l377_37759

theorem boys_to_girls_ratio (boys girls : ℕ) (h_boys : boys = 1500) (h_girls : girls = 1200) : 
  (boys / Nat.gcd boys girls) = 5 ∧ (girls / Nat.gcd boys girls) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_boys_to_girls_ratio_l377_37759


namespace NUMINAMATH_GPT_initial_speed_is_7_l377_37797

-- Definitions based on conditions
def distance_travelled (S : ℝ) (T : ℝ) : ℝ := S * T

-- Constants from problem
def time_initial : ℝ := 6
def time_final : ℝ := 3
def speed_final : ℝ := 14

-- Theorem statement
theorem initial_speed_is_7 : ∃ S : ℝ, distance_travelled S time_initial = distance_travelled speed_final time_final ∧ S = 7 := by
  sorry

end NUMINAMATH_GPT_initial_speed_is_7_l377_37797


namespace NUMINAMATH_GPT_parking_lot_cars_l377_37713

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_parking_lot_cars_l377_37713


namespace NUMINAMATH_GPT_first_digit_base_8_of_725_is_1_l377_37770

-- Define conditions
def decimal_val := 725

-- Helper function to get the largest power of 8 less than the decimal value
def largest_power_base_eight (n : ℕ) : ℕ :=
  if 8^3 <= n then 8^3 else if 8^2 <= n then 8^2 else if 8^1 <= n then 8^1 else if 8^0 <= n then 8^0 else 0

-- The target theorem
theorem first_digit_base_8_of_725_is_1 : 
  (725 / largest_power_base_eight 725) = 1 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_first_digit_base_8_of_725_is_1_l377_37770


namespace NUMINAMATH_GPT_pentagon_stack_l377_37760

/-- Given a stack of identical regular pentagons with vertices numbered from 1 to 5, rotated and flipped
such that the sums of numbers at each vertex are the same, the number of pentagons in the stacks can be
any natural number except 1 and 3. -/
theorem pentagon_stack (n : ℕ) (h0 : identical_pentagons_with_vertices_1_to_5)
  (h1 : pentagons_can_be_rotated_and_flipped)
  (h2 : stacked_vertex_to_vertex)
  (h3 : sums_at_each_vertex_are_equal) :
  ∃ k : ℕ, k = n ∧ n ≠ 1 ∧ n ≠ 3 :=
sorry

end NUMINAMATH_GPT_pentagon_stack_l377_37760


namespace NUMINAMATH_GPT_calc_dz_calc_d2z_calc_d3z_l377_37757

variables (x y dx dy : ℝ)

def z : ℝ := x^5 * y^3

-- Define the first differential dz
def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

-- Define the second differential d2z
def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

-- Define the third differential d3z
def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem calc_dz : (dz x y dx dy) = (5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) := 
by sorry

theorem calc_d2z : (d2z x y dx dy) = (20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) :=
by sorry

theorem calc_d3z : (d3z x y dx dy) = (60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end NUMINAMATH_GPT_calc_dz_calc_d2z_calc_d3z_l377_37757


namespace NUMINAMATH_GPT_douglas_votes_in_county_D_l377_37721

noncomputable def percent_votes_in_county_D (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
    (total_votes : ℝ) (percent_A percent_B percent_C percent_D total_percent : ℝ) : Prop :=
  (votes_A / (5 * x) = 0.70) ∧
  (votes_B / (3 * x) = 0.58) ∧
  (votes_C / (2 * x) = 0.50) ∧
  (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62 ∧
  (votes_D / (4 * x) = percent_D)

theorem douglas_votes_in_county_D 
  (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
  (total_votes : ℝ := 14 * x) 
  (percent_A percent_B percent_C total_percent percent_D : ℝ)
  (h1 : votes_A / (5 * x) = 0.70) 
  (h2 : votes_B / (3 * x) = 0.58) 
  (h3 : votes_C / (2 * x) = 0.50) 
  (h4 : (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62) : 
  percent_votes_in_county_D x votes_A votes_B votes_C votes_D total_votes percent_A percent_B percent_C 0.61 total_percent :=
by
  constructor
  exact h1
  constructor
  exact h2
  constructor
  exact h3
  constructor
  exact h4
  sorry

end NUMINAMATH_GPT_douglas_votes_in_county_D_l377_37721


namespace NUMINAMATH_GPT_sum_two_integers_l377_37751

theorem sum_two_integers (a b : ℤ) (h1 : a = 17) (h2 : b = 19) : a + b = 36 := by
  sorry

end NUMINAMATH_GPT_sum_two_integers_l377_37751


namespace NUMINAMATH_GPT_weather_station_accuracy_l377_37705

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

theorem weather_station_accuracy :
  binomial_probability 3 2 0.9 = 0.243 :=
by
  sorry

end NUMINAMATH_GPT_weather_station_accuracy_l377_37705


namespace NUMINAMATH_GPT_max_value_expression_l377_37775

theorem max_value_expression  
    (x y : ℝ) 
    (h : 2 * x^2 + y^2 = 6 * x) : 
    x^2 + y^2 + 2 * x ≤ 15 :=
sorry

end NUMINAMATH_GPT_max_value_expression_l377_37775


namespace NUMINAMATH_GPT_remainder_when_c_divided_by_b_eq_2_l377_37794

theorem remainder_when_c_divided_by_b_eq_2 
(a b c : ℕ) 
(hb : b = 3 * a + 3) 
(hc : c = 9 * a + 11) : 
  c % b = 2 := 
sorry

end NUMINAMATH_GPT_remainder_when_c_divided_by_b_eq_2_l377_37794


namespace NUMINAMATH_GPT_option_D_correct_l377_37791

theorem option_D_correct (a b : ℝ) : -a * b + 3 * b * a = 2 * a * b :=
by sorry

end NUMINAMATH_GPT_option_D_correct_l377_37791


namespace NUMINAMATH_GPT_largest_percent_error_l377_37716
noncomputable def max_percent_error (d : ℝ) (d_err : ℝ) (r_err : ℝ) : ℝ :=
  let d_min := d - d * d_err
  let d_max := d + d * d_err
  let r := d / 2
  let r_min := r - r * r_err
  let r_max := r + r * r_err
  let area_actual := Real.pi * r^2
  let area_d_min := Real.pi * (d_min / 2)^2
  let area_d_max := Real.pi * (d_max / 2)^2
  let area_r_min := Real.pi * r_min^2
  let area_r_max := Real.pi * r_max^2
  let error_d_min := (area_actual - area_d_min) / area_actual * 100
  let error_d_max := (area_d_max - area_actual) / area_actual * 100
  let error_r_min := (area_actual - area_r_min) / area_actual * 100
  let error_r_max := (area_r_max - area_actual) / area_actual * 100
  max (max error_d_min error_d_max) (max error_r_min error_r_max)

theorem largest_percent_error 
  (d : ℝ) (d_err : ℝ) (r_err : ℝ) 
  (h_d : d = 30) (h_d_err : d_err = 0.15) (h_r_err : r_err = 0.10) : 
  max_percent_error d d_err r_err = 31.57 := by
  sorry

end NUMINAMATH_GPT_largest_percent_error_l377_37716


namespace NUMINAMATH_GPT_school_spent_total_l377_37726

noncomputable def seminar_fee (num_teachers : ℕ) : ℝ :=
  let base_fee := 150 * num_teachers
  if num_teachers >= 20 then
    base_fee * 0.925
  else if num_teachers >= 10 then
    base_fee * 0.95
  else
    base_fee

noncomputable def seminar_fee_with_tax (num_teachers : ℕ) : ℝ :=
  let fee := seminar_fee num_teachers
  fee * 1.06

noncomputable def food_allowance (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  let num_regular := num_teachers - num_special
  num_regular * 10 + num_special * 15

noncomputable def total_cost (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  seminar_fee_with_tax num_teachers + food_allowance num_teachers num_special

theorem school_spent_total (num_teachers num_special : ℕ) (h : num_teachers = 22 ∧ num_special = 3) :
  total_cost num_teachers num_special = 3470.65 :=
by
  sorry

end NUMINAMATH_GPT_school_spent_total_l377_37726


namespace NUMINAMATH_GPT_each_regular_tire_distance_used_l377_37754

-- Define the conditions of the problem
def total_distance_traveled : ℕ := 50000
def spare_tire_distance : ℕ := 2000
def regular_tires_count : ℕ := 4

-- Using these conditions, we will state the problem as a theorem
theorem each_regular_tire_distance_used : 
  (total_distance_traveled - spare_tire_distance) / regular_tires_count = 12000 :=
by
  sorry

end NUMINAMATH_GPT_each_regular_tire_distance_used_l377_37754


namespace NUMINAMATH_GPT_number_of_guests_l377_37715

def cook_per_minute : ℕ := 10
def time_to_cook : ℕ := 80
def guests_ate_per_guest : ℕ := 5
def guests_to_serve : ℕ := 20 -- This is what we'll prove.

theorem number_of_guests 
    (cook_per_8min : cook_per_minute = 10)
    (total_time : time_to_cook = 80)
    (eat_rate : guests_ate_per_guest = 5) :
    (time_to_cook * cook_per_minute) / guests_ate_per_guest = guests_to_serve := 
by 
  sorry

end NUMINAMATH_GPT_number_of_guests_l377_37715


namespace NUMINAMATH_GPT_quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l377_37752

theorem quadratic_real_roots_discriminant (m : ℝ) :
  (2 * (m + 1))^2 - 4 * m * (m - 1) > 0 ↔ (m > -1/2 ∧ m ≠ 0) := 
sorry

theorem quadratic_real_roots_sum_of_squares (m x1 x2 : ℝ) 
  (h1 : m > -1/2 ∧ m ≠ 0)
  (h2 : x1 + x2 = -2 * (m + 1) / m)
  (h3 : x1 * x2 = (m - 1) / m)
  (h4 : x1^2 + x2^2 = 8) : 
  m = (6 + 2 * Real.sqrt 33) / 8 := 
sorry

end NUMINAMATH_GPT_quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l377_37752


namespace NUMINAMATH_GPT_point_third_quadrant_l377_37732

theorem point_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : 3 * m - 2 < 0 ∧ -n < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_third_quadrant_l377_37732


namespace NUMINAMATH_GPT_compute_fraction_sum_l377_37718

theorem compute_fraction_sum :
  8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_sum_l377_37718


namespace NUMINAMATH_GPT_q_value_l377_37750

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_q_value_l377_37750


namespace NUMINAMATH_GPT_driver_a_driven_more_distance_l377_37722

-- Definitions based on conditions
def initial_distance : ℕ := 787
def speed_a : ℕ := 90
def speed_b : ℕ := 80
def start_difference : ℕ := 1

-- Statement of the problem
theorem driver_a_driven_more_distance :
  let distance_a := speed_a * (start_difference + (initial_distance - speed_a) / (speed_a + speed_b))
  let distance_b := speed_b * ((initial_distance - speed_a) / (speed_a + speed_b))
  distance_a - distance_b = 131 := by
sorry

end NUMINAMATH_GPT_driver_a_driven_more_distance_l377_37722


namespace NUMINAMATH_GPT_sum_of_consecutive_even_integers_is_24_l377_37738

theorem sum_of_consecutive_even_integers_is_24 (x : ℕ) (h_pos : x > 0)
    (h_eq : (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2))) :
    (x - 2) + x + (x + 2) = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_integers_is_24_l377_37738


namespace NUMINAMATH_GPT_students_dont_eat_lunch_l377_37720

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end NUMINAMATH_GPT_students_dont_eat_lunch_l377_37720


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_l377_37774

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_l377_37774
