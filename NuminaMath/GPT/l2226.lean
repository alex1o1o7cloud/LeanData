import Mathlib

namespace NUMINAMATH_GPT_intersecting_functions_k_range_l2226_222633

theorem intersecting_functions_k_range 
  (k : ℝ) (h : 0 < k) : 
    ∃ x : ℝ, -2 * x + 3 = k / x ↔ k ≤ 9 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_intersecting_functions_k_range_l2226_222633


namespace NUMINAMATH_GPT_johannes_cabbage_sales_l2226_222652

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem johannes_cabbage_sales :
  (earnings_wednesday / price_per_kg) + (earnings_friday / price_per_kg) + (earnings_today / price_per_kg) = 48 := by
  sorry

end NUMINAMATH_GPT_johannes_cabbage_sales_l2226_222652


namespace NUMINAMATH_GPT_dean_taller_than_ron_l2226_222694

theorem dean_taller_than_ron (d h r : ℕ) (h1 : d = 15 * h) (h2 : r = 13) (h3 : d = 255) : h - r = 4 := 
by 
  sorry

end NUMINAMATH_GPT_dean_taller_than_ron_l2226_222694


namespace NUMINAMATH_GPT_unique_solution_arithmetic_progression_l2226_222608

variable {R : Type*} [Field R]

theorem unique_solution_arithmetic_progression (a b c m x y z : R) :
  (m ≠ -2) ∧ (m ≠ 1) ∧ (a + c = 2 * b) → 
  (x + y + m * z = a) ∧ (x + m * y + z = b) ∧ (m * x + y + z = c) → 
  ∃ x y z, 2 * y = x + z :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_arithmetic_progression_l2226_222608


namespace NUMINAMATH_GPT_monthly_rate_is_24_l2226_222648

noncomputable def weekly_rate : ℝ := 10
noncomputable def weeks_per_year : ℕ := 52
noncomputable def months_per_year : ℕ := 12
noncomputable def yearly_savings : ℝ := 232

theorem monthly_rate_is_24 (M : ℝ) (h : weeks_per_year * weekly_rate - months_per_year * M = yearly_savings) : 
  M = 24 :=
by
  sorry

end NUMINAMATH_GPT_monthly_rate_is_24_l2226_222648


namespace NUMINAMATH_GPT_tan_double_angle_third_quadrant_l2226_222642

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : sin (π - α) = -3 / 5) :
  tan (2 * α) = 24 / 7 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_third_quadrant_l2226_222642


namespace NUMINAMATH_GPT_v3_at_2_is_15_l2226_222669

-- Define the polynomial f(x)
def f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1

-- Define v3 using Horner's Rule at x
def v3 (x : ℝ) := ((x + 2) * x + 1) * x - 3

-- Prove that v3 at x = 2 equals 15
theorem v3_at_2_is_15 : v3 2 = 15 :=
by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_v3_at_2_is_15_l2226_222669


namespace NUMINAMATH_GPT_distribute_coins_l2226_222661

/-- The number of ways to distribute 25 identical coins among 4 schoolchildren -/
theorem distribute_coins :
  (Nat.choose 28 3) = 3276 :=
by
  sorry

end NUMINAMATH_GPT_distribute_coins_l2226_222661


namespace NUMINAMATH_GPT_max_sides_convex_polygon_with_obtuse_angles_l2226_222612

-- Definition of conditions
def is_convex_polygon (n : ℕ) : Prop := n ≥ 3
def obtuse_angles (n : ℕ) (k : ℕ) : Prop := k = 3 ∧ is_convex_polygon n

-- Statement of the problem
theorem max_sides_convex_polygon_with_obtuse_angles (n : ℕ) :
  obtuse_angles n 3 → n ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_sides_convex_polygon_with_obtuse_angles_l2226_222612


namespace NUMINAMATH_GPT_average_rate_of_interest_l2226_222638

def invested_amount_total : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05
def annual_return (amount : ℝ) (rate : ℝ) : ℝ := amount * rate

theorem average_rate_of_interest : 
  (∃ (x : ℝ), x > 0 ∧ x < invested_amount_total ∧ 
    annual_return (invested_amount_total - x) rate1 = annual_return x rate2) → 
  ((annual_return (invested_amount_total - 1875) rate1 + annual_return 1875 rate2) / invested_amount_total = 0.0375) := 
by
  sorry

end NUMINAMATH_GPT_average_rate_of_interest_l2226_222638


namespace NUMINAMATH_GPT_fraction_unchanged_when_multiplied_by_3_l2226_222667

variable (x y : ℚ)

theorem fraction_unchanged_when_multiplied_by_3 (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x) / (3 * (3 * x + y)) = x / (3 * x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_unchanged_when_multiplied_by_3_l2226_222667


namespace NUMINAMATH_GPT_adjacent_block_permutations_l2226_222629

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the block of digits that must be adjacent
def block : List ℕ := [2, 5, 8]

-- Function to calculate permutations of a list (size n)
def fact (n : ℕ) : ℕ := Nat.factorial n

-- Calculate the total number of arrangements
def total_arrangements : ℕ := fact 8 * fact 3

-- The main theorem statement to be proved
theorem adjacent_block_permutations :
  total_arrangements = 241920 :=
by
  sorry

end NUMINAMATH_GPT_adjacent_block_permutations_l2226_222629


namespace NUMINAMATH_GPT_flute_cost_is_correct_l2226_222690

-- Define the conditions
def total_spent : ℝ := 158.35
def stand_cost : ℝ := 8.89
def songbook_cost : ℝ := 7.0

-- Calculate the cost to be subtracted
def accessories_cost : ℝ := stand_cost + songbook_cost

-- Define the target cost of the flute
def flute_cost : ℝ := total_spent - accessories_cost

-- Prove that the flute cost is $142.46
theorem flute_cost_is_correct : flute_cost = 142.46 :=
by
  -- Here we would provide the proof
  sorry

end NUMINAMATH_GPT_flute_cost_is_correct_l2226_222690


namespace NUMINAMATH_GPT_determine_a_if_slope_angle_is_45_degrees_l2226_222601

-- Define the condition that the slope angle of the given line is 45°
def is_slope_angle_45_degrees (a : ℝ) : Prop :=
  let m := -a / (2 * a - 3)
  m = 1

-- State the theorem we need to prove
theorem determine_a_if_slope_angle_is_45_degrees (a : ℝ) :
  is_slope_angle_45_degrees a → a = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_a_if_slope_angle_is_45_degrees_l2226_222601


namespace NUMINAMATH_GPT_smallest_possible_n_l2226_222672

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def n_is_three_digit (n : ℕ) : Prop := 
  n ≥ 100 ∧ n < 1000

def prime_digits_less_than_10 (p : ℕ) : Prop :=
  p ∈ [2, 3, 5, 7]

def three_distinct_prime_factors (n a b : ℕ) : Prop :=
  a ≠ b ∧ is_prime a ∧ is_prime b ∧ is_prime (10 * a + b) ∧ n = a * b * (10 * a + b)

theorem smallest_possible_n :
  ∃ (n a b : ℕ), n_is_three_digit n ∧ prime_digits_less_than_10 a ∧ prime_digits_less_than_10 b ∧ three_distinct_prime_factors n a b ∧ n = 138 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_possible_n_l2226_222672


namespace NUMINAMATH_GPT_solve_system_l2226_222656

def F (t : ℝ) : ℝ := 32 * t ^ 5 + 48 * t ^ 3 + 17 * t - 15

def system_of_equations (x y z : ℝ) : Prop :=
  (1 / x = (32 / y ^ 5) + (48 / y ^ 3) + (17 / y) - 15) ∧
  (1 / y = (32 / z ^ 5) + (48 / z ^ 3) + (17 / z) - 15) ∧
  (1 / z = (32 / x ^ 5) + (48 / x ^ 3) + (17 / x) - 15)

theorem solve_system : ∃ (x y z : ℝ), system_of_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry -- Proof not included

end NUMINAMATH_GPT_solve_system_l2226_222656


namespace NUMINAMATH_GPT_tickets_difference_vip_general_l2226_222621

theorem tickets_difference_vip_general (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : 40 * V + 10 * G = 7500) : G - V = 34 := 
by
  sorry

end NUMINAMATH_GPT_tickets_difference_vip_general_l2226_222621


namespace NUMINAMATH_GPT_determine_b_l2226_222681

variable (a b c : ℝ)

theorem determine_b
  (h1 : -a / 3 = -c)
  (h2 : 1 + a + b + c = -c)
  (h3 : c = 5) :
  b = -26 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l2226_222681


namespace NUMINAMATH_GPT_bianca_bags_not_recycled_l2226_222620

theorem bianca_bags_not_recycled :
  ∀ (points_per_bag total_bags total_points bags_recycled bags_not_recycled : ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    total_points = 45 →
    bags_recycled = total_points / points_per_bag →
    bags_not_recycled = total_bags - bags_recycled →
    bags_not_recycled = 8 :=
by
  intros points_per_bag total_bags total_points bags_recycled bags_not_recycled
  intros h_points_per_bag h_total_bags h_total_points h_bags_recycled h_bags_not_recycled
  sorry

end NUMINAMATH_GPT_bianca_bags_not_recycled_l2226_222620


namespace NUMINAMATH_GPT_abs_neg_five_not_eq_five_l2226_222670

theorem abs_neg_five_not_eq_five : -(abs (-5)) ≠ 5 := by
  sorry

end NUMINAMATH_GPT_abs_neg_five_not_eq_five_l2226_222670


namespace NUMINAMATH_GPT_cos_identity_l2226_222606

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_identity_l2226_222606


namespace NUMINAMATH_GPT_track_length_l2226_222687

theorem track_length (y : ℝ) 
  (H1 : ∀ b s : ℝ, b + s = y ∧ b = y / 2 - 120 ∧ s = 120)
  (H2 : ∀ b s : ℝ, b + s = y + 180 ∧ b = y / 2 + 60 ∧ s = y / 2 - 60) :
  y = 600 :=
by 
  sorry

end NUMINAMATH_GPT_track_length_l2226_222687


namespace NUMINAMATH_GPT_g_extreme_value_f_ge_g_l2226_222682

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

theorem g_extreme_value :
  ∃ (x : ℝ), x = Real.exp 1 ∧ g x = 1 / Real.exp 1 + 2 :=
by sorry

theorem f_ge_g (x : ℝ) (hx : 0 < x) : f x >= g x :=
by sorry

end NUMINAMATH_GPT_g_extreme_value_f_ge_g_l2226_222682


namespace NUMINAMATH_GPT_winning_percentage_l2226_222675

/-- A soccer team played 158 games and won 63.2 games. 
    Prove that the winning percentage of the team is 40%. --/
theorem winning_percentage (total_games : ℕ) (won_games : ℝ) (h1 : total_games = 158) (h2 : won_games = 63.2) :
  (won_games / total_games) * 100 = 40 :=
sorry

end NUMINAMATH_GPT_winning_percentage_l2226_222675


namespace NUMINAMATH_GPT_find_m_l2226_222609

noncomputable def m : ℕ :=
  let S := {d : ℕ | d ∣ 15^8 ∧ d > 0}
  let total_ways := 9^6
  let strictly_increasing_ways := (Nat.choose 9 3) * (Nat.choose 10 3)
  let probability := strictly_increasing_ways / total_ways
  let gcd := Nat.gcd strictly_increasing_ways total_ways
  strictly_increasing_ways / gcd

theorem find_m : m = 112 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2226_222609


namespace NUMINAMATH_GPT_solve_for_x_l2226_222646

theorem solve_for_x (x y z w : ℕ) 
  (h1 : x = y + 7) 
  (h2 : y = z + 15) 
  (h3 : z = w + 25) 
  (h4 : w = 95) : 
  x = 142 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2226_222646


namespace NUMINAMATH_GPT_jane_drinks_l2226_222623

/-- Jane buys a combination of muffins, bagels, and drinks over five days,
where muffins cost 40 cents, bagels cost 90 cents, and drinks cost 30 cents.
The number of items bought is 5, and the total cost is a whole number of dollars.
Prove that the number of drinks Jane bought is 4. -/
theorem jane_drinks :
  ∃ b m d : ℕ, b + m + d = 5 ∧ (90 * b + 40 * m + 30 * d) % 100 = 0 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_jane_drinks_l2226_222623


namespace NUMINAMATH_GPT_sin_double_angle_l2226_222617

theorem sin_double_angle (θ : ℝ) (h₁ : 3 * (Real.cos θ)^2 = Real.tan θ + 3) (h₂ : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l2226_222617


namespace NUMINAMATH_GPT_floor_pi_plus_four_l2226_222676

theorem floor_pi_plus_four : Int.floor (Real.pi + 4) = 7 := by
  sorry

end NUMINAMATH_GPT_floor_pi_plus_four_l2226_222676


namespace NUMINAMATH_GPT_quadratic_function_solution_l2226_222634

noncomputable def g (x : ℝ) : ℝ := x^2 + 44 * x + 50

theorem quadratic_function_solution (c d : ℝ)
  (h : ∀ x, (g (g x + x)) / (g x) = x^2 + 44 * x + 50) :
  c = 44 ∧ d = 50 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_solution_l2226_222634


namespace NUMINAMATH_GPT_weight_of_new_girl_l2226_222691

theorem weight_of_new_girl (W N : ℝ) (h_weight_replacement: (20 * W / 20 + 40 - 40 + 40) / 20 = W / 20 + 2) :
  N = 80 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_girl_l2226_222691


namespace NUMINAMATH_GPT_width_of_grassy_plot_l2226_222602

-- Definitions
def length_plot : ℕ := 110
def width_path : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.50
def total_cost : ℝ := 425

-- Hypotheses and Target Proposition
theorem width_of_grassy_plot (w : ℝ) 
  (h1 : length_plot = 110)
  (h2 : width_path = 2.5)
  (h3 : cost_per_sq_meter = 0.50)
  (h4 : total_cost = 425)
  (h5 : (length_plot + 2 * width_path) * (w + 2 * width_path) = 115 * (w + 5))
  (h6 : 110 * w = 110 * w)
  (h7 : (115 * (w + 5) - (110 * w)) = total_cost / cost_per_sq_meter) :
  w = 55 := 
sorry

end NUMINAMATH_GPT_width_of_grassy_plot_l2226_222602


namespace NUMINAMATH_GPT_price_percentage_combined_assets_l2226_222658

variable (A B P : ℝ)

-- Conditions
axiom h1 : P = 1.20 * A
axiom h2 : P = 2 * B

-- Statement
theorem price_percentage_combined_assets : (P / (A + B)) * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_price_percentage_combined_assets_l2226_222658


namespace NUMINAMATH_GPT_find_x_l2226_222650

theorem find_x (x : ℝ) : 0.3 * x + 0.2 = 0.26 → x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2226_222650


namespace NUMINAMATH_GPT_variance_of_dataset_l2226_222630

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => y + acc) 0) / (x.length)

noncomputable def variance (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => (y - mean x)^2 + acc) 0) / (x.length)

theorem variance_of_dataset :
  variance dataset = 26 / 5 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_dataset_l2226_222630


namespace NUMINAMATH_GPT_simplified_fraction_l2226_222631

theorem simplified_fraction :
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = (1 / 120) :=
by 
  sorry

end NUMINAMATH_GPT_simplified_fraction_l2226_222631


namespace NUMINAMATH_GPT_total_cost_of_goods_l2226_222660

theorem total_cost_of_goods :
  ∃ (M R F : ℝ),
    (10 * M = 24 * R) ∧
    (6 * F = 2 * R) ∧
    (F = 20.50) ∧
    (4 * M + 3 * R + 5 * F = 877.40) :=
by {
  sorry
}

end NUMINAMATH_GPT_total_cost_of_goods_l2226_222660


namespace NUMINAMATH_GPT_sin_cos_identity_l2226_222635

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := 
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l2226_222635


namespace NUMINAMATH_GPT_additional_oil_needed_l2226_222697

def car_cylinders := 6
def car_oil_per_cylinder := 8
def truck_cylinders := 8
def truck_oil_per_cylinder := 10
def motorcycle_cylinders := 4
def motorcycle_oil_per_cylinder := 6

def initial_car_oil := 16
def initial_truck_oil := 20
def initial_motorcycle_oil := 8

theorem additional_oil_needed :
  let car_total_oil := car_cylinders * car_oil_per_cylinder
  let truck_total_oil := truck_cylinders * truck_oil_per_cylinder
  let motorcycle_total_oil := motorcycle_cylinders * motorcycle_oil_per_cylinder
  let car_additional_oil := car_total_oil - initial_car_oil
  let truck_additional_oil := truck_total_oil - initial_truck_oil
  let motorcycle_additional_oil := motorcycle_total_oil - initial_motorcycle_oil
  car_additional_oil = 32 ∧
  truck_additional_oil = 60 ∧
  motorcycle_additional_oil = 16 :=
by
  repeat (exact sorry)

end NUMINAMATH_GPT_additional_oil_needed_l2226_222697


namespace NUMINAMATH_GPT_trees_chopped_in_first_half_l2226_222653

theorem trees_chopped_in_first_half (x : ℕ) (h1 : ∀ t, t = x + 300) (h2 : 3 * t = 1500) : x = 200 :=
by
  sorry

end NUMINAMATH_GPT_trees_chopped_in_first_half_l2226_222653


namespace NUMINAMATH_GPT_minimum_races_to_determine_top_five_fastest_horses_l2226_222636

-- Defining the conditions
def max_horses_per_race : ℕ := 3
def total_horses : ℕ := 50

-- The main statement to prove the minimum number of races y
theorem minimum_races_to_determine_top_five_fastest_horses (y : ℕ) :
  y = 19 :=
sorry

end NUMINAMATH_GPT_minimum_races_to_determine_top_five_fastest_horses_l2226_222636


namespace NUMINAMATH_GPT_product_of_numbers_in_given_ratio_l2226_222647

theorem product_of_numbers_in_given_ratio :
  ∃ (x y : ℝ), (x - y) ≠ 0 ∧ (x + y) / (x - y) = 9 ∧ (x * y) / (x - y) = 40 ∧ (x * y) = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_numbers_in_given_ratio_l2226_222647


namespace NUMINAMATH_GPT_num_people_in_5_years_l2226_222640

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 12
  | (k+1) => 4 * seq k - 18

theorem num_people_in_5_years : seq 5 = 6150 :=
  sorry

end NUMINAMATH_GPT_num_people_in_5_years_l2226_222640


namespace NUMINAMATH_GPT_find_k_l2226_222600

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem find_k (a b k : ℝ) (h1 : f a b k = 4) (h2 : f a b (f a b k) = 7) (h3 : f a b (f a b (f a b k)) = 19) :
  k = 13 / 4 := 
sorry

end NUMINAMATH_GPT_find_k_l2226_222600


namespace NUMINAMATH_GPT_ones_digit_largest_power_of_3_dividing_18_factorial_l2226_222684

theorem ones_digit_largest_power_of_3_dividing_18_factorial :
  (3^8 % 10) = 1 :=
by sorry

end NUMINAMATH_GPT_ones_digit_largest_power_of_3_dividing_18_factorial_l2226_222684


namespace NUMINAMATH_GPT_latest_start_time_l2226_222689

-- Define the times for each activity
def homework_time : ℕ := 30
def clean_room_time : ℕ := 30
def take_out_trash_time : ℕ := 5
def empty_dishwasher_time : ℕ := 10
def dinner_time : ℕ := 45

-- Define the total time required to finish everything in minutes
def total_time_needed : ℕ := homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time + dinner_time

-- Define the equivalent time in hours
def total_time_needed_hours : ℕ := total_time_needed / 60

-- Define movie start time and the time Justin gets home
def movie_start_time : ℕ := 20 -- (8 PM in 24-hour format)
def justin_home_time : ℕ := 17 -- (5 PM in 24-hour format)

-- Prove the latest time Justin can start his chores and homework
theorem latest_start_time : movie_start_time - total_time_needed_hours = 18 := by
  sorry

end NUMINAMATH_GPT_latest_start_time_l2226_222689


namespace NUMINAMATH_GPT_closed_broken_line_impossible_l2226_222674

theorem closed_broken_line_impossible (n : ℕ) (h : n = 1989) : ¬ (∃ a b : ℕ, 2 * (a + b) = n) :=
by {
  sorry
}

end NUMINAMATH_GPT_closed_broken_line_impossible_l2226_222674


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2226_222639

variables (x y a b : ℝ)

-- Problem 1
theorem simplify_expr1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := 
by sorry

-- Problem 2
theorem simplify_expr2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := 
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2226_222639


namespace NUMINAMATH_GPT_initial_bird_count_l2226_222677

theorem initial_bird_count (B : ℕ) (h₁ : B + 7 = 12) : B = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_bird_count_l2226_222677


namespace NUMINAMATH_GPT_meet_at_midpoint_l2226_222624

open Classical

noncomputable def distance_travel1 (t : ℝ) : ℝ :=
  4 * t

noncomputable def distance_travel2 (t : ℝ) : ℝ :=
  (t / 2) * (3.5 + 0.5 * t)

theorem meet_at_midpoint (t : ℝ) : 
  (4 * t + (t / 2) * (3.5 + 0.5 * t) = 72) → 
  (t = 9) ∧ (4 * t = 36) := 
 by 
  sorry

end NUMINAMATH_GPT_meet_at_midpoint_l2226_222624


namespace NUMINAMATH_GPT_remove_two_fractions_sum_is_one_l2226_222680

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end NUMINAMATH_GPT_remove_two_fractions_sum_is_one_l2226_222680


namespace NUMINAMATH_GPT_quadratic_real_roots_l2226_222607

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3 / 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2226_222607


namespace NUMINAMATH_GPT_intersection_complement_A_B_l2226_222603

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem intersection_complement_A_B :
  U = Set.univ →
  A = {x | -1 < x ∧ x < 1} →
  B = {y | 0 < y} →
  (A ∩ complement B) = {x | -1 < x ∧ x ≤ 0} :=
by
  intros hU hA hB
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_l2226_222603


namespace NUMINAMATH_GPT_proof_problem_l2226_222632

noncomputable def a_n (n : ℕ) : ℕ := n + 2
noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 3
noncomputable def C_n (n : ℕ) : ℚ := 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))
noncomputable def T_n (n : ℕ) : ℚ := (1/4) * (1 - (1/(2 * n + 1)))

theorem proof_problem :
  (∀ n, a_n n = n + 2) ∧
  (∀ n, b_n n = 2 * n + 3) ∧
  (∀ n, C_n n = 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))) ∧
  (∀ n, T_n n = (1/4) * (1 - (1/(2 * n + 1)))) ∧
  (∀ n, (T_n n > k / 54) ↔ k < 9) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2226_222632


namespace NUMINAMATH_GPT_fraction_subtraction_l2226_222686

theorem fraction_subtraction :
  (15 / 45) - (1 + (2 / 9)) = - (8 / 9) :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l2226_222686


namespace NUMINAMATH_GPT_actual_value_wrongly_copied_l2226_222618

theorem actual_value_wrongly_copied (mean_initial : ℝ) (n : ℕ) (wrong_value : ℝ) (mean_correct : ℝ) :
  mean_initial = 140 → n = 30 → wrong_value = 135 → mean_correct = 140.33333333333334 →
  ∃ actual_value : ℝ, actual_value = 145 :=
by
  intros
  sorry

end NUMINAMATH_GPT_actual_value_wrongly_copied_l2226_222618


namespace NUMINAMATH_GPT_roger_initial_candies_l2226_222610

def initial_candies (given_candies left_candies : ℕ) : ℕ :=
  given_candies + left_candies

theorem roger_initial_candies :
  initial_candies 3 92 = 95 :=
by
  sorry

end NUMINAMATH_GPT_roger_initial_candies_l2226_222610


namespace NUMINAMATH_GPT_solve_for_y_l2226_222673

theorem solve_for_y (x : ℝ) (y : ℝ) (h1 : x = 8) (h2 : x^(2*y) = 16) : y = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2226_222673


namespace NUMINAMATH_GPT_investment_initial_amount_l2226_222698

theorem investment_initial_amount (P : ℝ) (h1 : ∀ (x : ℝ), 0 < x → (1 + 0.10) * x = 1.10 * x) (h2 : 1.21 * P = 363) : P = 300 :=
sorry

end NUMINAMATH_GPT_investment_initial_amount_l2226_222698


namespace NUMINAMATH_GPT_female_managers_count_l2226_222644

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end NUMINAMATH_GPT_female_managers_count_l2226_222644


namespace NUMINAMATH_GPT_circles_fit_l2226_222619

noncomputable def fit_circles_in_rectangle : Prop :=
  ∃ (m n : ℕ) (α : ℝ), (m * n * α * α = 1) ∧ (m * n * α / 2 = 1962)

theorem circles_fit : fit_circles_in_rectangle :=
by sorry

end NUMINAMATH_GPT_circles_fit_l2226_222619


namespace NUMINAMATH_GPT_solve_f_sqrt_2009_l2226_222695

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_never_zero : ∀ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

theorem solve_f_sqrt_2009 :
  f (sqrt 2009) = 1 / 2009 := sorry

end NUMINAMATH_GPT_solve_f_sqrt_2009_l2226_222695


namespace NUMINAMATH_GPT_max_min_values_l2226_222663

def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

theorem max_min_values :
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  max (f a) (f b) = f a ∧ f a = 269 / 27 ∧ min (f a) (f b) = f b ∧ f b = -5 :=
by
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  have ha : f a = 269 / 27 := sorry
  have hb : f b = -5 := sorry
  have max_eq : max (f a) (f b) = f a := by sorry
  have min_eq : min (f a) (f b) = f b := by sorry
  exact ⟨max_eq, ha, min_eq, hb⟩

end NUMINAMATH_GPT_max_min_values_l2226_222663


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2226_222693

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2226_222693


namespace NUMINAMATH_GPT_most_pieces_day_and_maximum_number_of_popular_days_l2226_222692

-- Definitions for conditions:
def a_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then 3 * n
else 65 - 2 * n

def S_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then (3 + 3 * n) * n / 2
else 273 + (51 - n) * (n - 13)

-- Propositions to prove:
theorem most_pieces_day_and_maximum :
  ∃ k a_k, (1 ≤ k ∧ k ≤ 31) ∧
           (a_k = a_n k) ∧
           (∀ n, 1 ≤ n ∧ n ≤ 31 → a_n n ≤ a_k) ∧
           k = 13 ∧ a_k = 39 := 
sorry

theorem number_of_popular_days :
  ∃ days_popular,
    (∃ n1, 1 ≤ n1 ∧ n1 ≤ 13 ∧ S_n n1 > 200) ∧
    (∃ n2, 14 ≤ n2 ∧ n2 ≤ 31 ∧ a_n n2 < 20) ∧
    days_popular = (22 - 12 + 1) :=
sorry

end NUMINAMATH_GPT_most_pieces_day_and_maximum_number_of_popular_days_l2226_222692


namespace NUMINAMATH_GPT_total_earnings_per_week_correct_l2226_222668

noncomputable def weekday_fee_kid : ℝ := 3
noncomputable def weekday_fee_adult : ℝ := 6
noncomputable def weekend_surcharge_ratio : ℝ := 0.5

noncomputable def num_kids_weekday : ℕ := 8
noncomputable def num_adults_weekday : ℕ := 10

noncomputable def num_kids_weekend : ℕ := 12
noncomputable def num_adults_weekend : ℕ := 15

noncomputable def weekday_earnings_kids : ℝ := (num_kids_weekday : ℝ) * weekday_fee_kid
noncomputable def weekday_earnings_adults : ℝ := (num_adults_weekday : ℝ) * weekday_fee_adult

noncomputable def weekday_earnings_total : ℝ := weekday_earnings_kids + weekday_earnings_adults

noncomputable def weekday_earning_per_week : ℝ := weekday_earnings_total * 5

noncomputable def weekend_fee_kid : ℝ := weekday_fee_kid * (1 + weekend_surcharge_ratio)
noncomputable def weekend_fee_adult : ℝ := weekday_fee_adult * (1 + weekend_surcharge_ratio)

noncomputable def weekend_earnings_kids : ℝ := (num_kids_weekend : ℝ) * weekend_fee_kid
noncomputable def weekend_earnings_adults : ℝ := (num_adults_weekend : ℝ) * weekend_fee_adult

noncomputable def weekend_earnings_total : ℝ := weekend_earnings_kids + weekend_earnings_adults

noncomputable def weekend_earning_per_week : ℝ := weekend_earnings_total * 2

noncomputable def total_weekly_earnings : ℝ := weekday_earning_per_week + weekend_earning_per_week

theorem total_earnings_per_week_correct : total_weekly_earnings = 798 := by
  sorry

end NUMINAMATH_GPT_total_earnings_per_week_correct_l2226_222668


namespace NUMINAMATH_GPT_allocate_to_Team_A_l2226_222625

theorem allocate_to_Team_A (x : ℕ) :
  31 + x = 2 * (50 - x) →
  x = 23 :=
by
  sorry

end NUMINAMATH_GPT_allocate_to_Team_A_l2226_222625


namespace NUMINAMATH_GPT_inequality_solution_l2226_222613

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) : (x - (1/x) > 0) ↔ (-1 < x ∧ x < 0) ∨ (1 < x) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2226_222613


namespace NUMINAMATH_GPT_expected_value_of_monicas_winnings_l2226_222659

def die_outcome (n : ℕ) : ℤ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then n else if n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 then 0 else -5

noncomputable def expected_winnings : ℚ :=
  (1/2 : ℚ) * 0 + (1/8 : ℚ) * 2 + (1/8 : ℚ) * 3 + (1/8 : ℚ) * 5 + (1/8 : ℚ) * 7 + (1/8 : ℚ) * (-5)

theorem expected_value_of_monicas_winnings : expected_winnings = 3/2 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_monicas_winnings_l2226_222659


namespace NUMINAMATH_GPT_binary_and_ternary_product_l2226_222645

theorem binary_and_ternary_product :
  let binary_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ternary_1021 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  binary_1011 = 11 ∧ ternary_1021 = 34 →
  binary_1011 * ternary_1021 = 374 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_binary_and_ternary_product_l2226_222645


namespace NUMINAMATH_GPT_rate_of_simple_interest_l2226_222622

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end NUMINAMATH_GPT_rate_of_simple_interest_l2226_222622


namespace NUMINAMATH_GPT_divides_polynomial_l2226_222604

theorem divides_polynomial (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∀ x : ℂ, (x^2 + x + 1) ∣ (x^(3 * m + 1) + x^(3 * n + 2) + 1) :=
by
  sorry

end NUMINAMATH_GPT_divides_polynomial_l2226_222604


namespace NUMINAMATH_GPT_find_k_l2226_222649

theorem find_k (k t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 75) : k = 167 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_l2226_222649


namespace NUMINAMATH_GPT_tan_of_diff_l2226_222615

theorem tan_of_diff (θ : ℝ) (hθ : -π/2 + 2 * π < θ ∧ θ < 2 * π) 
  (h : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_of_diff_l2226_222615


namespace NUMINAMATH_GPT_average_of_remaining_six_is_correct_l2226_222643

noncomputable def average_of_remaining_six (s20 s14: ℕ) (avg20 avg14: ℚ) : ℚ :=
  let sum20 := s20 * avg20
  let sum14 := s14 * avg14
  let sum_remaining := sum20 - sum14
  (sum_remaining / (s20 - s14))

theorem average_of_remaining_six_is_correct : 
  average_of_remaining_six 20 14 500 390 = 756.67 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_remaining_six_is_correct_l2226_222643


namespace NUMINAMATH_GPT_value_of_k_plus_p_l2226_222611

theorem value_of_k_plus_p
  (k p : ℝ)
  (h1 : ∀ x : ℝ, 3*x^2 - k*x + p = 0)
  (h_sum_roots : k / 3 = -3)
  (h_prod_roots : p / 3 = -6)
  : k + p = -27 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_plus_p_l2226_222611


namespace NUMINAMATH_GPT_jack_valid_sequences_l2226_222655

-- Definitions based strictly on the conditions from Step a)
def valid_sequence_count : ℕ :=
  -- Count the valid paths under given conditions (mock placeholder definition)
  1  -- This represents the proof statement

-- The main theorem stating the proof problem
theorem jack_valid_sequences :
  valid_sequence_count = 1 := 
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_jack_valid_sequences_l2226_222655


namespace NUMINAMATH_GPT_admission_fee_for_children_l2226_222688

theorem admission_fee_for_children (x : ℝ) :
  (∀ (admission_fee_adult : ℝ) (total_people : ℝ) (total_fees_collected : ℝ) (children_admitted : ℝ) (adults_admitted : ℝ),
    admission_fee_adult = 4 ∧
    total_people = 315 ∧
    total_fees_collected = 810 ∧
    children_admitted = 180 ∧
    adults_admitted = total_people - children_admitted ∧
    total_fees_collected = children_admitted * x + adults_admitted * admission_fee_adult
  ) → x = 1.5 := sorry

end NUMINAMATH_GPT_admission_fee_for_children_l2226_222688


namespace NUMINAMATH_GPT_greatest_gcd_of_rope_lengths_l2226_222605

theorem greatest_gcd_of_rope_lengths : Nat.gcd (Nat.gcd 39 52) 65 = 13 := by
  sorry

end NUMINAMATH_GPT_greatest_gcd_of_rope_lengths_l2226_222605


namespace NUMINAMATH_GPT_ellipse_major_minor_axis_ratio_l2226_222626

theorem ellipse_major_minor_axis_ratio
  (a b : ℝ)
  (h₀ : a = 2 * b):
  2 * a = 4 * b :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_minor_axis_ratio_l2226_222626


namespace NUMINAMATH_GPT_problem_arithmetic_l2226_222616

variable {α : Type*} [LinearOrderedField α] 

def arithmetic_sum (a d : α) (n : ℕ) : α := n * (2 * a + (n - 1) * d) / 2
def arithmetic_term (a d : α) (k : ℕ) : α := a + (k - 1) * d

theorem problem_arithmetic (a3 a2015 : ℝ) 
  (h_roots : a3 + a2015 = 10) 
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_sum : ∀ n, S n = arithmetic_sum a3 ((a2015 - a3) / 2012) n) 
  (h_an : ∀ k, a k = arithmetic_term a3 ((a2015 - a3) / 2012) k) :
  (S 2017) / 2017 + a 1009 = 10 := by
sorry

end NUMINAMATH_GPT_problem_arithmetic_l2226_222616


namespace NUMINAMATH_GPT_sara_quarters_l2226_222683

-- Conditions
def usd_to_eur (usd : ℝ) : ℝ := usd * 0.85
def eur_to_usd (eur : ℝ) : ℝ := eur * 1.15
def value_of_quarter_usd : ℝ := 0.25
def dozen : ℕ := 12

-- Theorem
theorem sara_quarters (sara_savings_usd : ℝ) (usd_to_eur_ratio : ℝ) (eur_to_usd_ratio : ℝ) (quarter_value_usd : ℝ) (doz : ℕ) : sara_savings_usd = 9 → usd_to_eur_ratio = 0.85 → eur_to_usd_ratio = 1.15 → quarter_value_usd = 0.25 → doz = 12 → 
  ∃ dozens : ℕ, dozens = 2 :=
by
  sorry

end NUMINAMATH_GPT_sara_quarters_l2226_222683


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l2226_222665

theorem arithmetic_progression_sum (a d : ℝ)
  (h1 : 10 * (2 * a + 19 * d) = 200)
  (h2 : 25 * (2 * a + 49 * d) = 0) :
  35 * (2 * a + 69 * d) = -466.67 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l2226_222665


namespace NUMINAMATH_GPT_evaluate_expression_l2226_222654

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2226_222654


namespace NUMINAMATH_GPT_max_value_of_x_plus_y_l2226_222685

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ x + y + (1/x) + (1/y) = 5

-- Theorem statement
theorem max_value_of_x_plus_y (x y : ℝ) (h : conditions x y) : x + y ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_value_of_x_plus_y_l2226_222685


namespace NUMINAMATH_GPT_cost_of_one_shirt_l2226_222628

-- Definitions based on the conditions given
variables (J S : ℝ)

-- First condition: 3 pairs of jeans and 2 shirts cost $69
def condition1 : Prop := 3 * J + 2 * S = 69

-- Second condition: 2 pairs of jeans and 3 shirts cost $61
def condition2 : Prop := 2 * J + 3 * S = 61

-- The theorem to prove that the cost of one shirt is $9
theorem cost_of_one_shirt (J S : ℝ) (h1 : condition1 J S) (h2 : condition2 J S) : S = 9 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_shirt_l2226_222628


namespace NUMINAMATH_GPT_other_number_l2226_222679

theorem other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 30) (h_A : A = 770) : B = 90 :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_other_number_l2226_222679


namespace NUMINAMATH_GPT_meaningful_expression_range_l2226_222614

theorem meaningful_expression_range (x : ℝ) (h : 1 - x > 0) : x < 1 := sorry

end NUMINAMATH_GPT_meaningful_expression_range_l2226_222614


namespace NUMINAMATH_GPT_midpoint_AB_find_Q_find_H_l2226_222696

-- Problem 1: Midpoint of AB
theorem midpoint_AB (x1 y1 x2 y2 : ℝ) : 
  let A := (x1, y1)
  let B := (x2, y2)
  let M := ( (x1 + x2) / 2, (y1 + y2) / 2 )
  M = ( (x1 + x2) / 2, (y1 + y2) / 2 )
:= 
  -- The lean statement that shows the midpoint formula is correct.
  sorry

-- Problem 2: Coordinates of Q given midpoint
theorem find_Q (px py mx my : ℝ) : 
  let P := (px, py)
  let M := (mx, my)
  let Q := (2 * mx - px, 2 * my - py)
  ( (px + Q.1) / 2 = mx ∧ (py + Q.2) / 2 = my )
:= 
  -- Lean statement to find Q
  sorry

-- Problem 3: Coordinates of H given midpoints coinciding
theorem find_H (xE yE xF yF xG yG : ℝ) :
  let E := (xE, yE)
  let F := (xF, yF)
  let G := (xG, yG)
  ∃ xH yH : ℝ, 
    ( (xE + xH) / 2 = (xF + xG) / 2 ∧ (yE + yH) / 2 = (yF + yG) / 2 ) ∨
    ( (xF + xH) / 2 = (xE + xG) / 2 ∧ (yF + yH) / 2 = (yE + yG) / 2 ) ∨
    ( (xG + xH) / 2 = (xE + xF) / 2 ∧ (yG + yH) / 2 = (yE + yF) / 2 )
:=
  -- Lean statement to find H
  sorry

end NUMINAMATH_GPT_midpoint_AB_find_Q_find_H_l2226_222696


namespace NUMINAMATH_GPT_area_of_square_l2226_222671

theorem area_of_square (r s l : ℕ) (h1 : l = (2 * r) / 5) (h2 : r = s) (h3 : l * 10 = 240) : s * s = 3600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_l2226_222671


namespace NUMINAMATH_GPT_eggs_eaten_in_afternoon_l2226_222657

theorem eggs_eaten_in_afternoon (initial : ℕ) (morning : ℕ) (final : ℕ) (afternoon : ℕ) :
  initial = 20 → morning = 4 → final = 13 → afternoon = initial - morning - final → afternoon = 3 :=
by
  intros h_initial h_morning h_final h_afternoon
  rw [h_initial, h_morning, h_final] at h_afternoon
  linarith

end NUMINAMATH_GPT_eggs_eaten_in_afternoon_l2226_222657


namespace NUMINAMATH_GPT_mod_equiv_l2226_222699

theorem mod_equiv (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 :=
by
  sorry

end NUMINAMATH_GPT_mod_equiv_l2226_222699


namespace NUMINAMATH_GPT_Total_points_proof_l2226_222664

noncomputable def Samanta_points (Mark_points : ℕ) : ℕ := Mark_points + 8
noncomputable def Mark_points (Eric_points : ℕ) : ℕ := Eric_points + (Eric_points / 2)
def Eric_points : ℕ := 6
noncomputable def Daisy_points (Total_points_Samanta_Mark_Eric : ℕ) : ℕ := Total_points_Samanta_Mark_Eric - (Total_points_Samanta_Mark_Eric / 4)

def Total_points_Samanta_Mark_Eric (Samanta_points Mark_points Eric_points : ℕ) : ℕ := Samanta_points + Mark_points + Eric_points

theorem Total_points_proof :
  let Mk_pts := Mark_points Eric_points
  let Sm_pts := Samanta_points Mk_pts
  let Tot_SME := Total_points_Samanta_Mark_Eric Sm_pts Mk_pts Eric_points
  let D_pts := Daisy_points Tot_SME
  Sm_pts + Mk_pts + Eric_points + D_pts = 56 := by
  sorry

end NUMINAMATH_GPT_Total_points_proof_l2226_222664


namespace NUMINAMATH_GPT_AmandaWillSpend_l2226_222637

/--
Amanda goes shopping and sees a sale where different items have different discounts.
She wants to buy a dress for $50 with a 30% discount, a pair of shoes for $75 with a 25% discount,
and a handbag for $100 with a 40% discount.
After applying the discounts, a 5% tax is added to the final price.
Prove that Amanda will spend $158.81 to buy all three items after the discounts and tax have been applied.
-/
noncomputable def totalAmount : ℝ :=
  let dressPrice := 50
  let dressDiscount := 0.30
  let shoesPrice := 75
  let shoesDiscount := 0.25
  let handbagPrice := 100
  let handbagDiscount := 0.40
  let taxRate := 0.05
  let dressFinalPrice := dressPrice * (1 - dressDiscount)
  let shoesFinalPrice := shoesPrice * (1 - shoesDiscount)
  let handbagFinalPrice := handbagPrice * (1 - handbagDiscount)
  let subtotal := dressFinalPrice + shoesFinalPrice + handbagFinalPrice
  let tax := subtotal * taxRate
  let totalAmount := subtotal + tax
  totalAmount

theorem AmandaWillSpend : totalAmount = 158.81 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_AmandaWillSpend_l2226_222637


namespace NUMINAMATH_GPT_cone_volume_with_same_radius_and_height_l2226_222678

theorem cone_volume_with_same_radius_and_height (r h : ℝ) 
  (Vcylinder : ℝ) (Vcone : ℝ) (h1 : Vcylinder = 54 * Real.pi) 
  (h2 : Vcone = (1 / 3) * Vcylinder) : Vcone = 18 * Real.pi :=
by sorry

end NUMINAMATH_GPT_cone_volume_with_same_radius_and_height_l2226_222678


namespace NUMINAMATH_GPT_orchid_bushes_total_l2226_222666

def current_orchid_bushes : ℕ := 22
def new_orchid_bushes : ℕ := 13

theorem orchid_bushes_total : current_orchid_bushes + new_orchid_bushes = 35 := 
by 
  sorry

end NUMINAMATH_GPT_orchid_bushes_total_l2226_222666


namespace NUMINAMATH_GPT_min_red_chips_l2226_222662

theorem min_red_chips (w b r : ℕ) (h1 : b ≥ w / 3) (h2 : b ≤ r / 4) (h3 : w + b ≥ 75) : r ≥ 76 :=
sorry

end NUMINAMATH_GPT_min_red_chips_l2226_222662


namespace NUMINAMATH_GPT_part1_part2_l2226_222641

-- Defining the function f
def f (x : ℝ) (a : ℝ) : ℝ := a * abs (x + 1) - abs (x - 1)

-- Part 1: a = 1, finding the solution set of the inequality f(x) < 3/2
theorem part1 (x : ℝ) : f x 1 < 3 / 2 ↔ x < 3 / 4 := 
sorry

-- Part 2: a > 1, and existence of x such that f(x) <= -|2m+1|, finding the range of m
theorem part2 (a : ℝ) (h : 1 < a) (m : ℝ) (x : ℝ) : 
  f x a ≤ -abs (2 * m + 1) → -3 / 2 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2226_222641


namespace NUMINAMATH_GPT_sum_in_base_8_l2226_222627

theorem sum_in_base_8 (a b : ℕ) (h_a : a = 3 * 8^2 + 2 * 8 + 7)
                                  (h_b : b = 7 * 8 + 3) :
  (a + b) = 4 * 8^2 + 2 * 8 + 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_in_base_8_l2226_222627


namespace NUMINAMATH_GPT_algebraic_expression_value_l2226_222651

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2226_222651
