import Mathlib

namespace school_robes_l2185_218589

theorem school_robes (total_singers robes_needed : ℕ) (robe_cost total_spent existing_robes : ℕ) 
  (h1 : total_singers = 30)
  (h2 : robe_cost = 2)
  (h3 : total_spent = 36)
  (h4 : total_singers - total_spent / robe_cost = existing_robes) :
  existing_robes = 12 :=
by sorry

end school_robes_l2185_218589


namespace apples_per_pie_l2185_218598

-- Definitions of the conditions
def number_of_pies : ℕ := 10
def harvested_apples : ℕ := 50
def to_buy_apples : ℕ := 30
def total_apples_needed : ℕ := harvested_apples + to_buy_apples

-- The theorem to prove
theorem apples_per_pie :
  (total_apples_needed / number_of_pies) = 8 := 
sorry

end apples_per_pie_l2185_218598


namespace gcd_of_72_and_90_l2185_218529

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l2185_218529


namespace height_of_wall_l2185_218514

theorem height_of_wall (length_brick width_brick height_brick : ℝ)
                        (length_wall width_wall number_of_bricks : ℝ)
                        (volume_of_bricks : ℝ) :
  (length_brick, width_brick, height_brick) = (125, 11.25, 6) →
  (length_wall, width_wall) = (800, 22.5) →
  number_of_bricks = 1280 →
  volume_of_bricks = length_brick * width_brick * height_brick * number_of_bricks →
  volume_of_bricks = length_wall * width_wall * 600 := 
by
  intros h1 h2 h3 h4
  -- proof skipped
  sorry

end height_of_wall_l2185_218514


namespace find_middle_number_l2185_218513

theorem find_middle_number (a b c d x e f g : ℝ) 
  (h1 : (a + b + c + d + x + e + f + g) / 8 = 7)
  (h2 : (a + b + c + d + x) / 5 = 6)
  (h3 : (x + e + f + g + d) / 5 = 9) :
  x = 9.5 := 
by 
  sorry

end find_middle_number_l2185_218513


namespace boa_constrictors_in_park_l2185_218515

theorem boa_constrictors_in_park :
  ∃ (B : ℕ), (∃ (p : ℕ), p = 3 * B) ∧ (B + 3 * B + 40 = 200) ∧ B = 40 :=
by
  sorry

end boa_constrictors_in_park_l2185_218515


namespace vertex_of_quadratic_l2185_218584

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ (h = 1) ∧ (k = -2) :=
by
  sorry

end vertex_of_quadratic_l2185_218584


namespace total_students_in_class_l2185_218594

def current_students : ℕ := 6 * 3
def students_bathroom : ℕ := 5
def students_canteen : ℕ := 5 * 5
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def group4_students : ℕ := 3
def new_group_students : ℕ := group1_students + group2_students + group3_students + group4_students
def germany_students : ℕ := 3
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 2
def spain_students : ℕ := 2
def australia_students : ℕ := 1
def foreign_exchange_students : ℕ :=
  germany_students + france_students + norway_students + italy_students + spain_students + australia_students

def total_students : ℕ :=
  current_students + students_bathroom + students_canteen + new_group_students + foreign_exchange_students

theorem total_students_in_class : total_students = 81 := by
  rfl  -- Reflective equality since total_students already sums to 81 based on the definitions

end total_students_in_class_l2185_218594


namespace complex_fraction_identity_l2185_218550

theorem complex_fraction_identity (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 / 3 :=
by 
  sorry

end complex_fraction_identity_l2185_218550


namespace rectangle_square_overlap_l2185_218595

theorem rectangle_square_overlap (ABCD EFGH : Type) (s x y : ℝ)
  (h1 : 0.3 * s^2 = 0.6 * x * y)
  (h2 : AB = 2 * s)
  (h3 : AD = y)
  (h4 : x * y = 0.5 * s^2) :
  x / y = 8 :=
sorry

end rectangle_square_overlap_l2185_218595


namespace second_integer_value_l2185_218541

theorem second_integer_value (n : ℚ) (h : (n - 1) + (n + 1) + (n + 2) = 175) : n = 57 + 2 / 3 :=
by
  sorry

end second_integer_value_l2185_218541


namespace find_length_of_room_l2185_218526

def length_of_room (L : ℕ) (width verandah_width verandah_area : ℕ) : Prop :=
  (L + 2 * verandah_width) * (width + 2 * verandah_width) - (L * width) = verandah_area

theorem find_length_of_room : length_of_room 15 12 2 124 :=
by
  -- We state the proof here, which is not requested in this exercise
  sorry

end find_length_of_room_l2185_218526


namespace tea_leaves_costs_l2185_218558

theorem tea_leaves_costs (a_1 b_1 a_2 b_2 : ℕ) (c_A c_B : ℝ) :
  a_1 * c_A = 4000 ∧ 
  b_1 * c_B = 8400 ∧ 
  b_1 = a_1 + 10 ∧ 
  c_B = 1.4 * c_A ∧ 
  a_2 + b_2 = 100 ∧ 
  (300 - c_A) * (a_2 / 2) + (300 * 0.7 - c_A) * (a_2 / 2) + 
  (400 - c_B) * (b_2 / 2) + (400 * 0.7 - c_B) * (b_2 / 2) = 5800 
  → c_A = 200 ∧ c_B = 280 ∧ a_2 = 40 ∧ b_2 = 60 := 
sorry

end tea_leaves_costs_l2185_218558


namespace largest_multiple_of_7_gt_neg_150_l2185_218540

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l2185_218540


namespace solve_problem_l2185_218543

def bracket (a b c : ℕ) : ℕ := (a + b) / c

theorem solve_problem :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 :=
by
  sorry

end solve_problem_l2185_218543


namespace correct_operation_l2185_218555

-- Define that m and n are elements of an arbitrary commutative ring
variables {R : Type*} [CommRing R] (m n : R)

theorem correct_operation : (m * n) ^ 2 = m ^ 2 * n ^ 2 := by
  sorry

end correct_operation_l2185_218555


namespace slices_with_both_pepperoni_and_mushrooms_l2185_218590

theorem slices_with_both_pepperoni_and_mushrooms (n : ℕ)
  (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (all_have_topping : ∀ (s : ℕ), s < total_slices → s < pepperoni_slices ∨ s < mushroom_slices ∨ s < (total_slices - pepperoni_slices - mushroom_slices) )
  (total_condition : total_slices = 16)
  (pepperoni_condition : pepperoni_slices = 8)
  (mushroom_condition : mushroom_slices = 12) :
  (8 - n) + (12 - n) + n = 16 → n = 4 :=
sorry

end slices_with_both_pepperoni_and_mushrooms_l2185_218590


namespace find_m_l2185_218577

theorem find_m (m : ℤ) :
  (2 * m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end find_m_l2185_218577


namespace lcm_pair_eq_sum_l2185_218575

theorem lcm_pair_eq_sum (x y : ℕ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : Nat.lcm x y = 1 + 2 * x + 3 * y) :
  (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) :=
by {
  sorry
}

end lcm_pair_eq_sum_l2185_218575


namespace prob_both_students_female_l2185_218580

-- Define the conditions
def total_students : ℕ := 5
def male_students : ℕ := 2
def female_students : ℕ := 3
def selected_students : ℕ := 2

-- Define the function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 2 female students
def probability_both_female : ℚ := 
  (binomial female_students selected_students : ℚ) / (binomial total_students selected_students : ℚ)

-- The actual theorem to be proved
theorem prob_both_students_female : probability_both_female = 0.3 := by
  sorry

end prob_both_students_female_l2185_218580


namespace typing_and_editing_time_l2185_218573

-- Definitions for typing and editing times for consultants together and for Mary and Jim individually
def combined_typing_time := 12.5
def combined_editing_time := 7.5
def mary_typing_time := 30.0
def jim_editing_time := 12.0

-- The total time when Jim types and Mary edits
def total_time := 42.0

-- Proof statement
theorem typing_and_editing_time :
  (combined_typing_time = 12.5) ∧ 
  (combined_editing_time = 7.5) ∧ 
  (mary_typing_time = 30.0) ∧ 
  (jim_editing_time = 12.0) →
  total_time = 42.0 := 
by
  intro h
  -- Proof to be filled later
  sorry

end typing_and_editing_time_l2185_218573


namespace S_n_formula_l2185_218533

def P (n : ℕ) : Type := sorry -- The type representing the nth polygon, not fully defined here.
def S : ℕ → ℝ := sorry -- The sequence S_n defined recursively.

-- Recursive definition of S_n given
axiom S_0 : S 0 = 1

-- This axiom represents the recursive step mentioned in the problem.
axiom S_rec : ∀ (k : ℕ), S (k + 1) = S k + (4^k / 3^(2*k + 2))

-- The main theorem we need to prove
theorem S_n_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

end S_n_formula_l2185_218533


namespace percentage_increase_variable_cost_l2185_218521

noncomputable def variable_cost_first_year : ℝ := 26000
noncomputable def fixed_cost : ℝ := 40000
noncomputable def total_breeding_cost_third_year : ℝ := 71460

theorem percentage_increase_variable_cost (x : ℝ) 
  (h : 40000 + 26000 * (1 + x) ^ 2 = 71460) : 
  x = 0.1 := 
by sorry

end percentage_increase_variable_cost_l2185_218521


namespace quadratic_function_expression_quadratic_function_inequality_l2185_218565

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x : ℝ, f (x + 1) - f x = 2 * x) 
  (h₂ : f 0 = 1) : 
  (f x = x^2 - x + 1) := 
by {
  sorry
}

theorem quadratic_function_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m) ↔ m < -1 := 
by {
  sorry
}

end quadratic_function_expression_quadratic_function_inequality_l2185_218565


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l2185_218505

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l2185_218505


namespace suitable_communication_l2185_218578

def is_suitable_to_communicate (beijing_time : Nat) (sydney_difference : Int) (los_angeles_difference : Int) : Bool :=
  let sydney_time := beijing_time + sydney_difference
  let los_angeles_time := beijing_time - los_angeles_difference
  sydney_time >= 8 ∧ sydney_time <= 22 -- let's assume suitable time is between 8:00 to 22:00

theorem suitable_communication:
  let beijing_time := 18
  let sydney_difference := 2
  let los_angeles_difference := 15
  is_suitable_to_communicate beijing_time sydney_difference los_angeles_difference = true :=
by
  sorry

end suitable_communication_l2185_218578


namespace sector_radius_l2185_218560

theorem sector_radius (θ : ℝ) (s : ℝ) (R : ℝ) 
  (hθ : θ = 150)
  (hs : s = (5 / 2) * Real.pi)
  : (θ / 360) * (2 * Real.pi * R) = (5 / 2) * Real.pi → 
  R = 3 := 
sorry

end sector_radius_l2185_218560


namespace wire_length_l2185_218572

variable (L M l a : ℝ) -- Assume these variables are real numbers.

theorem wire_length (h1 : a ≠ 0) : L = (M / a) * l :=
sorry

end wire_length_l2185_218572


namespace sufficient_but_not_necessary_condition_l2185_218510

theorem sufficient_but_not_necessary_condition (x y : ℝ) : 
  (x > 3 ∧ y > 3 → x + y > 6) ∧ ¬(x + y > 6 → x > 3 ∧ y > 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2185_218510


namespace inequality_proof_l2185_218587

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ((b + c - a)^2) / (a^2 + (b + c)^2) + ((c + a - b)^2) / (b^2 + (c + a)^2) + ((a + b - c)^2) / (c^2 + (a + b)^2) ≥ 3 / 5 :=
  sorry

end inequality_proof_l2185_218587


namespace position_of_z_l2185_218502

theorem position_of_z (total_distance : ℕ) (total_steps : ℕ) (steps_taken : ℕ) (distance_covered : ℕ) (h1 : total_distance = 30) (h2 : total_steps = 6) (h3 : steps_taken = 4) (h4 : distance_covered = total_distance / total_steps) : 
  steps_taken * distance_covered = 20 :=
by
  sorry

end position_of_z_l2185_218502


namespace coin_difference_l2185_218562

variables (x y : ℕ)

theorem coin_difference (h1 : x + y = 15) (h2 : 2 * x + 5 * y = 51) : x - y = 1 := by
  sorry

end coin_difference_l2185_218562


namespace polynomial_diff_l2185_218566

theorem polynomial_diff (m n : ℤ) (h1 : 2 * m + 2 = 0) (h2 : n - 4 = 0) :
  (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := 
by {
  -- This is where the proof would go, so we put sorry for now
  sorry
}

end polynomial_diff_l2185_218566


namespace volume_of_figure_eq_half_l2185_218546

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end volume_of_figure_eq_half_l2185_218546


namespace total_area_rectangle_l2185_218571

theorem total_area_rectangle (BF CF : ℕ) (A1 A2 x : ℕ) (h1 : BF = 3 * CF) (h2 : A1 = 3 * A2) (h3 : 2 * x = 96) (h4 : 48 = x) (h5 : A1 = 3 * 48) (h6 : A2 = 48) : A1 + A2 = 192 :=
  by sorry

end total_area_rectangle_l2185_218571


namespace eval_f_neg2_l2185_218588

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l2185_218588


namespace deal_or_no_deal_min_eliminations_l2185_218520

theorem deal_or_no_deal_min_eliminations (n_boxes : ℕ) (n_high_value : ℕ) 
    (initial_count : n_boxes = 26)
    (high_value_count : n_high_value = 9) :
  ∃ (min_eliminations : ℕ), min_eliminations = 8 ∧
    ((n_boxes - min_eliminations - 1) / 2) ≥ n_high_value :=
sorry

end deal_or_no_deal_min_eliminations_l2185_218520


namespace eliminate_alpha_l2185_218554

theorem eliminate_alpha (α x y : ℝ) (h1 : x = Real.tan α ^ 2) (h2 : y = Real.sin α ^ 2) : 
  x - y = x * y := 
by
  sorry

end eliminate_alpha_l2185_218554


namespace determine_x_l2185_218500

-- Definitions for given conditions
variables (x y z a b c : ℝ)
variables (h₁ : xy / (x - y) = a) (h₂ : xz / (x - z) = b) (h₃ : yz / (y - z) = c)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Main statement to prove
theorem determine_x :
  x = (2 * a * b * c) / (a * b + b * c + c * a) :=
sorry

end determine_x_l2185_218500


namespace polygon_sides_l2185_218596

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l2185_218596


namespace total_wire_length_l2185_218583

theorem total_wire_length
  (A B C D E : ℕ)
  (hA : A = 16)
  (h_ratio : 4 * A = 5 * B ∧ 4 * A = 7 * C ∧ 4 * A = 3 * D ∧ 4 * A = 2 * E)
  (hC : C = B + 8) :
  (A + B + C + D + E) = 84 := 
sorry

end total_wire_length_l2185_218583


namespace negative_x_y_l2185_218538

theorem negative_x_y (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 :=
by
  sorry

end negative_x_y_l2185_218538


namespace part1_part2_l2185_218592

variable (a b : ℝ)
def A : ℝ := 2 * a * b - a
def B : ℝ := -a * b + 2 * a + b

theorem part1 : 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b := by
  sorry

theorem part2 : (∀ b : ℝ, 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b) -> a = 1 / 6 := by
  sorry

end part1_part2_l2185_218592


namespace tutors_meet_after_84_days_l2185_218576

theorem tutors_meet_after_84_days :
  let jaclyn := 3
  let marcelle := 4
  let susanna := 6
  let wanda := 7
  Nat.lcm (Nat.lcm (Nat.lcm jaclyn marcelle) susanna) wanda = 84 := by
  sorry

end tutors_meet_after_84_days_l2185_218576


namespace find_greatest_divisor_l2185_218518

def greatest_divisor_leaving_remainders (n₁ n₁_r n₂ n₂_r d : ℕ) : Prop :=
  (n₁ % d = n₁_r) ∧ (n₂ % d = n₂_r) 

theorem find_greatest_divisor :
  greatest_divisor_leaving_remainders 1657 10 2037 7 1 :=
by
  sorry

end find_greatest_divisor_l2185_218518


namespace arctan_sum_l2185_218553

theorem arctan_sum : 
  Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = Real.pi / 4 := 
by 
  sorry

end arctan_sum_l2185_218553


namespace exchange_ways_count_l2185_218597

theorem exchange_ways_count : ∃ n : ℕ, n = 46 ∧ ∀ x y z : ℕ, x + 2 * y + 5 * z = 20 → n = 46 :=
by
  sorry

end exchange_ways_count_l2185_218597


namespace total_price_all_art_l2185_218567

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l2185_218567


namespace binomial_sixteen_twelve_eq_l2185_218561

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l2185_218561


namespace jed_speeding_l2185_218504

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l2185_218504


namespace linear_regression_increase_l2185_218534

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ :=
  1.6 * x + 2

-- Prove that y increases by 1.6 when x increases by 1
theorem linear_regression_increase (x : ℝ) :
  linear_regression (x + 1) - linear_regression x = 1.6 :=
by sorry

end linear_regression_increase_l2185_218534


namespace time_to_pass_platform_l2185_218508

-- Definitions for the given conditions
def train_length := 1200 -- length of the train in meters
def tree_crossing_time := 120 -- time taken to cross a tree in seconds
def platform_length := 1200 -- length of the platform in meters

-- Calculation of speed of the train and distance to be covered
def train_speed := train_length / tree_crossing_time -- speed in meters per second
def total_distance_to_cover := train_length + platform_length -- total distance in meters

-- Proof statement that given the above conditions, the time to pass the platform is 240 seconds
theorem time_to_pass_platform : 
  total_distance_to_cover / train_speed = 240 :=
  by sorry

end time_to_pass_platform_l2185_218508


namespace population_net_increase_l2185_218512

-- Definitions of conditions
def birth_rate := 7 / 2 -- 7 people every 2 seconds
def death_rate := 1 / 2 -- 1 person every 2 seconds
def seconds_in_a_day := 86400 -- Number of seconds in one day

-- Definition of the total births in one day
def total_births_per_day := birth_rate * seconds_in_a_day

-- Definition of the total deaths in one day
def total_deaths_per_day := death_rate * seconds_in_a_day

-- Proposition to prove the net population increase in one day
theorem population_net_increase : total_births_per_day - total_deaths_per_day = 259200 := by
  sorry

end population_net_increase_l2185_218512


namespace luke_games_l2185_218535

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l2185_218535


namespace part_a_l2185_218528

theorem part_a (m n : ℕ) (hm : m > 1) : n ∣ Nat.totient (m^n - 1) :=
sorry

end part_a_l2185_218528


namespace leopards_count_l2185_218585

theorem leopards_count (L : ℕ) (h1 : 100 + 80 + L + 10 * L + 50 + 2 * (80 + L) = 670) : L = 20 :=
by
  sorry

end leopards_count_l2185_218585


namespace range_of_m_l2185_218574

variable (m t : ℝ)

namespace proof_problem

def proposition_p : Prop :=
  ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1) → (t + 2) * (t - 10) < 0

def proposition_q (m : ℝ) : Prop :=
  -m < t ∧ t < m + 1 ∧ m > 0

theorem range_of_m :
  (∃ t, proposition_q m t) → proposition_p t → 0 < m ∧ m ≤ 2 := by
  sorry

end proof_problem

end range_of_m_l2185_218574


namespace symmetric_line_equation_l2185_218551

-- Define the given lines
def original_line (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := y + 2 = 0

-- Define the problem statement as a theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ), line_of_symmetry x y → (original_line x (2 * (-2 - y) + 1)) ↔ (2 * x + y + 5 = 0) := 
sorry

end symmetric_line_equation_l2185_218551


namespace problem1_l2185_218524

variable (α : ℝ)

theorem problem1 (h : Real.tan α = -3/4) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3/4 := 
sorry

end problem1_l2185_218524


namespace cube_volume_l2185_218591

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 294) : s^3 = 343 := 
by 
  sorry

end cube_volume_l2185_218591


namespace Zhukov_birth_year_l2185_218586

-- Define the conditions
def years_lived_total : ℕ := 78
def years_lived_20th_more_than_19th : ℕ := 70

-- Define the proof problem
theorem Zhukov_birth_year :
  ∃ y19 y20 : ℕ, y19 + y20 = years_lived_total ∧ y20 = y19 + years_lived_20th_more_than_19th ∧ (1900 - y19) = 1896 :=
by
  sorry

end Zhukov_birth_year_l2185_218586


namespace olivia_not_sold_bars_l2185_218542

theorem olivia_not_sold_bars (cost_per_bar : ℕ) (total_bars : ℕ) (total_money_made : ℕ) :
  cost_per_bar = 3 →
  total_bars = 7 →
  total_money_made = 9 →
  total_bars - (total_money_made / cost_per_bar) = 4 :=
by
  intros h1 h2 h3
  sorry

end olivia_not_sold_bars_l2185_218542


namespace range_of_a_l2185_218582

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x - a ≤ -3) → a ∈ Set.Iic (-6) ∪ Set.Ici 2 :=
by
  intro h
  sorry

end range_of_a_l2185_218582


namespace inequality_C_incorrect_l2185_218509

theorem inequality_C_incorrect (x : ℝ) (h : x ≠ 0) : ¬(e^x < 1 + x) → (e^1 ≥ 1 + 1) :=
by {
  sorry
}

end inequality_C_incorrect_l2185_218509


namespace bailey_points_final_game_l2185_218507

def chandra_points (a: ℕ) := 2 * a
def akiko_points (m: ℕ) := m + 4
def michiko_points (b: ℕ) := b / 2
def team_total_points (b c a m: ℕ) := b + c + a + m

theorem bailey_points_final_game (B: ℕ) 
  (M : ℕ := michiko_points B)
  (A : ℕ := akiko_points M)
  (C : ℕ := chandra_points A)
  (H : team_total_points B C A M = 54): B = 14 :=
by 
  sorry

end bailey_points_final_game_l2185_218507


namespace find_speed_of_goods_train_l2185_218559

noncomputable def speed_of_goods_train (v_man : ℝ) (t_pass : ℝ) (d_goods : ℝ) : ℝ := 
  let v_man_mps := v_man * (1000 / 3600)
  let v_relative := d_goods / t_pass
  let v_goods_mps := v_relative - v_man_mps
  v_goods_mps * (3600 / 1000)

theorem find_speed_of_goods_train :
  speed_of_goods_train 45 8 340 = 108 :=
by sorry

end find_speed_of_goods_train_l2185_218559


namespace cubic_binomial_expansion_l2185_218563

theorem cubic_binomial_expansion :
  49^3 + 3 * 49^2 + 3 * 49 + 1 = 125000 :=
by
  sorry

end cubic_binomial_expansion_l2185_218563


namespace train_length_calculation_l2185_218557

theorem train_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (platform_length_m : ℝ) (train_length_m: ℝ) : speed_kmph = 45 → time_seconds = 51.99999999999999 → platform_length_m = 290 → train_length_m = 360 :=
by
  sorry

end train_length_calculation_l2185_218557


namespace percentage_problem_l2185_218579

-- Define the main proposition
theorem percentage_problem (n : ℕ) (a : ℕ) (b : ℕ) (P : ℕ) :
  n = 6000 →
  a = (50 * n) / 100 →
  b = (30 * a) / 100 →
  (P * b) / 100 = 90 →
  P = 10 :=
by
  intros h_n h_a h_b h_Pb
  sorry

end percentage_problem_l2185_218579


namespace valid_license_plates_count_l2185_218536

-- Define the number of choices for letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates
def num_valid_license_plates : ℕ := num_letters^3 * num_digits^3

-- Theorem stating that the number of valid license plates is 17,576,000
theorem valid_license_plates_count :
  num_valid_license_plates = 17576000 :=
by
  sorry

end valid_license_plates_count_l2185_218536


namespace find_m_sum_terms_l2185_218517

theorem find_m (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) 
  (h2 : a 3 + a 6 + a 10 + a 13 = 32) (hm : a m = 8) : m = 8 :=
sorry

theorem sum_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (hS3 : S 3 = 9) (hS6 : S 6 = 36) 
  (a_def : ∀ n, S n = n * (a 1 + a n) / 2) : a 7 + a 8 + a 9 = 45 :=
sorry

end find_m_sum_terms_l2185_218517


namespace find_a_l2185_218581

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = Real.log (-a * x)) (h2 : ∀ x : ℝ, f (-x) = -f x) :
  a = 1 :=
by
  sorry

end find_a_l2185_218581


namespace unique_solution_to_exponential_poly_equation_l2185_218570

noncomputable def polynomial_has_unique_real_solution : Prop :=
  ∃! x : ℝ, (2 : ℝ)^(3 * x + 3) - 3 * (2 : ℝ)^(2 * x + 1) - (2 : ℝ)^x + 1 = 0

theorem unique_solution_to_exponential_poly_equation :
  polynomial_has_unique_real_solution :=
sorry

end unique_solution_to_exponential_poly_equation_l2185_218570


namespace binomial_510_510_l2185_218531

theorem binomial_510_510 : Nat.choose 510 510 = 1 :=
by
  sorry

end binomial_510_510_l2185_218531


namespace lock_and_key_requirements_l2185_218522

/-- There are 7 scientists each with a key to an electronic lock which requires at least 4 scientists to open.
    - Prove that the minimum number of unique features (locks) the electronic lock must have is 35.
    - Prove that each scientist's key should have at least 20 features.
--/
theorem lock_and_key_requirements :
  ∃ (locks : ℕ) (features_per_key : ℕ), 
    locks = 35 ∧ features_per_key = 20 ∧
    (∀ (n_present : ℕ), n_present ≥ 4 → 7 - n_present ≤ 3) ∧
    (∀ (n_absent : ℕ), n_absent ≤ 3 → 7 - n_absent ≥ 4)
:= sorry

end lock_and_key_requirements_l2185_218522


namespace student_ticket_cost_l2185_218501

def general_admission_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 525
def total_revenue : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

def number_of_student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
def revenue_from_general_admission : ℕ := general_admission_tickets_sold * general_admission_ticket_cost

theorem student_ticket_cost : ∃ S : ℕ, number_of_student_tickets_sold * S + revenue_from_general_admission = total_revenue ∧ S = 4 :=
by
  sorry

end student_ticket_cost_l2185_218501


namespace max_value_min_expression_l2185_218525

def f (x y : ℝ) : ℝ :=
  x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_value_min_expression (a b c : ℝ) (h₁: a ≠ b) (h₂: b ≠ c) (h₃: c ≠ a)
  (hab : f a b = f b c) (hbc : f b c = f c a) :
  (max (min (a^4 - 4*a^3 + 4*a^2) (min (b^4 - 4*b^3 + 4*b^2) (c^4 - 4*c^3 + 4*c^2))) 1) = 1 :=
sorry

end max_value_min_expression_l2185_218525


namespace conversion_bah_rah_yah_l2185_218556

theorem conversion_bah_rah_yah (bahs rahs yahs : ℝ) 
  (h1 : 10 * bahs = 16 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) :
  (10 / 16) * (6 / 10) * 500 * yahs = 187.5 * bahs :=
by sorry

end conversion_bah_rah_yah_l2185_218556


namespace mario_time_on_moving_sidewalk_l2185_218549

theorem mario_time_on_moving_sidewalk (d w v : ℝ) (h_walk : d = 90 * w) (h_sidewalk : d = 45 * v) : 
  d / (w + v) = 30 :=
by
  sorry

end mario_time_on_moving_sidewalk_l2185_218549


namespace quadratic_inequality_solution_l2185_218511

theorem quadratic_inequality_solution :
  {x : ℝ | 2 * x ^ 2 - x - 3 > 0} = {x : ℝ | x > 3 / 2 ∨ x < -1} :=
sorry

end quadratic_inequality_solution_l2185_218511


namespace polynomial_root_p_value_l2185_218530

theorem polynomial_root_p_value (p : ℝ) : (3 : ℝ) ^ 3 + p * (3 : ℝ) - 18 = 0 → p = -3 :=
by
  intro h
  sorry

end polynomial_root_p_value_l2185_218530


namespace disputed_piece_weight_l2185_218545

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l2185_218545


namespace intersection_complement_eq_l2185_218537

open Set

-- Definitions from the problem conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | x ≥ 0}
def C_U_N : Set ℝ := {x | x < 0}

-- Statement of the proof problem
theorem intersection_complement_eq : M ∩ C_U_N = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end intersection_complement_eq_l2185_218537


namespace surface_area_ratio_l2185_218593

-- Definitions for side lengths in terms of common multiplier x
def side_length_a (x : ℝ) := 2 * x
def side_length_b (x : ℝ) := 1 * x
def side_length_c (x : ℝ) := 3 * x
def side_length_d (x : ℝ) := 4 * x
def side_length_e (x : ℝ) := 6 * x

-- Definitions for surface areas using the given formula
def surface_area (side_length : ℝ) := 6 * side_length^2

def surface_area_a (x : ℝ) := surface_area (side_length_a x)
def surface_area_b (x : ℝ) := surface_area (side_length_b x)
def surface_area_c (x : ℝ) := surface_area (side_length_c x)
def surface_area_d (x : ℝ) := surface_area (side_length_d x)
def surface_area_e (x : ℝ) := surface_area (side_length_e x)

-- Proof statement for the ratio of total surface areas
theorem surface_area_ratio (x : ℝ) (hx : x ≠ 0) :
  (surface_area_a x) / (surface_area_b x) = 4 ∧
  (surface_area_c x) / (surface_area_b x) = 9 ∧
  (surface_area_d x) / (surface_area_b x) = 16 ∧
  (surface_area_e x) / (surface_area_b x) = 36 :=
by {
  sorry
}

end surface_area_ratio_l2185_218593


namespace share_ratio_l2185_218539

theorem share_ratio (A B C : ℕ) (hA : A = (2 * B) / 3) (hA_val : A = 372) (hB_val : B = 93) (hC_val : C = 62) : B / C = 3 / 2 := 
by 
  sorry

end share_ratio_l2185_218539


namespace no_n_satisfies_mod_5_l2185_218506

theorem no_n_satisfies_mod_5 (n : ℤ) : (n^3 + 2*n - 1) % 5 ≠ 0 :=
by
  sorry

end no_n_satisfies_mod_5_l2185_218506


namespace statement_A_statement_B_statement_C_l2185_218569

variables {p : ℝ} (hp : p > 0) (x0 y0 x1 y1 x2 y2 : ℝ)
variables (h_parabola : ∀ x y, y^2 = 2*p*x) 
variables (h_point_P : ∀ k m, y0 ≠ 0 ∧ x0 = k*y0 + m)

-- Statement A
theorem statement_A (hy0 : y0 = 0) : y1 * y2 = -2 * p * x0 :=
sorry

-- Statement B
theorem statement_B (hx0 : x0 = 0) : 1 / y1 + 1 / y2 = 1 / y0 :=
sorry

-- Statement C
theorem statement_C : (y0 - y1) * (y0 - y2) = y0^2 - 2 * p * x0 :=
sorry

end statement_A_statement_B_statement_C_l2185_218569


namespace smallest_n_for_2007_l2185_218516

/-- The smallest number of positive integers \( n \) such that their product is 2007 and their sum is 2007.
Given that \( n > 1 \), we need to show 1337 is the smallest such \( n \).
-/
theorem smallest_n_for_2007 (n : ℕ) (H : n > 1) :
  (∃ s : Finset ℕ, (s.sum id = 2007) ∧ (s.prod id = 2007) ∧ (s.card = n)) → (n = 1337) :=
sorry

end smallest_n_for_2007_l2185_218516


namespace Queen_High_School_teachers_needed_l2185_218552

def students : ℕ := 1500
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

theorem Queen_High_School_teachers_needed : 
  (students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by 
  sorry

end Queen_High_School_teachers_needed_l2185_218552


namespace parabola_vertex_coordinates_l2185_218519

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = -(x - 1) ^ 2 + 3 → (1, 3) = (1, 3) :=
by
  intros x y h
  sorry

end parabola_vertex_coordinates_l2185_218519


namespace max_discount_l2185_218544

theorem max_discount (C : ℝ) (x : ℝ) (h1 : 1.8 * C = 360) (h2 : ∀ y, y ≥ 1.3 * C → 360 - x ≥ y) : x ≤ 100 :=
by
  have hC : C = 360 / 1.8 := by sorry
  have hMinPrice : 1.3 * C = 1.3 * (360 / 1.8) := by sorry
  have hDiscount : 360 - x ≥ 1.3 * (360 / 1.8) := by sorry
  sorry

end max_discount_l2185_218544


namespace max_soap_boxes_in_carton_l2185_218527

theorem max_soap_boxes_in_carton
  (L_carton W_carton H_carton : ℕ)
  (L_soap_box W_soap_box H_soap_box : ℕ)
  (vol_carton := L_carton * W_carton * H_carton)
  (vol_soap_box := L_soap_box * W_soap_box * H_soap_box)
  (max_soap_boxes := vol_carton / vol_soap_box) :
  L_carton = 25 → W_carton = 42 → H_carton = 60 →
  L_soap_box = 7 → W_soap_box = 6 → H_soap_box = 5 →
  max_soap_boxes = 300 :=
by
  intros hL hW hH hLs hWs hHs
  sorry

end max_soap_boxes_in_carton_l2185_218527


namespace reflection_problem_l2185_218599

theorem reflection_problem 
  (m b : ℝ)
  (h : ∀ (P Q : ℝ × ℝ), 
        P = (2,2) ∧ Q = (8,4) → 
        ∃ mid : ℝ × ℝ, 
        mid = ((P.fst + Q.fst) / 2, (P.snd + Q.snd) / 2) ∧ 
        ∃ m' : ℝ, m' ≠ 0 ∧ P.snd - m' * P.fst = Q.snd - m' * Q.fst) :
  m + b = 15 := 
sorry

end reflection_problem_l2185_218599


namespace virginia_avg_rainfall_l2185_218532

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l2185_218532


namespace negation_of_forall_pos_l2185_218568

open Real

theorem negation_of_forall_pos (h : ∀ x : ℝ, x^2 - x + 1 > 0) : 
  ¬(∀ x : ℝ, x^2 - x + 1 > 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_forall_pos_l2185_218568


namespace discount_percentage_l2185_218547

/-
  A retailer buys 80 pens at the market price of 36 pens from a wholesaler.
  He sells these pens giving a certain discount and his profit is 120%.
  What is the discount percentage he gave on the pens?
-/
theorem discount_percentage
  (P : ℝ)
  (CP SP D DP : ℝ) 
  (h1 : CP = 36 * P)
  (h2 : SP = 2.2 * CP)
  (h3 : D = P - (SP / 80))
  (h4 : DP = (D / P) * 100) :
  DP = 1 := 
sorry

end discount_percentage_l2185_218547


namespace bottles_in_one_bag_l2185_218503

theorem bottles_in_one_bag (total_bottles : ℕ) (cartons bags_per_carton : ℕ)
  (h1 : total_bottles = 180)
  (h2 : cartons = 3)
  (h3 : bags_per_carton = 4) :
  total_bottles / cartons / bags_per_carton = 15 :=
by sorry

end bottles_in_one_bag_l2185_218503


namespace Cheryl_more_eggs_than_others_l2185_218564

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l2185_218564


namespace greatest_distance_is_correct_l2185_218548

-- Define the coordinates of the post.
def post_coordinate : ℝ × ℝ := (6, -2)

-- Define the length of the rope.
def rope_length : ℝ := 12

-- Define the origin.
def origin : ℝ × ℝ := (0, 0)

-- Define the formula to calculate the Euclidean distance between two points in ℝ².
noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := by
  sorry

-- Define the distance from the origin to the post.
noncomputable def distance_origin_to_post : ℝ := euclidean_distance origin post_coordinate

-- Define the greatest distance the dog can be from the origin.
noncomputable def greatest_distance_from_origin : ℝ := distance_origin_to_post + rope_length

-- Prove that the greatest distance the dog can be from the origin is 12 + 2 * sqrt 10.
theorem greatest_distance_is_correct : greatest_distance_from_origin = 12 + 2 * Real.sqrt 10 := by
  sorry

end greatest_distance_is_correct_l2185_218548


namespace probability_of_exactly_nine_correct_placements_is_zero_l2185_218523

-- Define the number of letters and envelopes
def num_letters : ℕ := 10

-- Define the condition of letters being randomly inserted into envelopes
def random_insertion (n : ℕ) : Prop := true

-- Prove that the probability of exactly nine letters being correctly placed is zero
theorem probability_of_exactly_nine_correct_placements_is_zero
  (h : random_insertion num_letters) : 
  (∃ p : ℝ, p = 0) := 
sorry

end probability_of_exactly_nine_correct_placements_is_zero_l2185_218523
