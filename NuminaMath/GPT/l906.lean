import Mathlib

namespace tables_capacity_l906_90644

theorem tables_capacity (invited attended : ℕ) (didn't_show_up : ℕ) (tables : ℕ) (capacity : ℕ) 
    (h1 : invited = 24) (h2 : didn't_show_up = 10) (h3 : attended = invited - didn't_show_up) 
    (h4 : attended = 14) (h5 : tables = 2) : capacity = attended / tables :=
by {
  -- Proof goes here
  sorry
}

end tables_capacity_l906_90644


namespace days_collected_money_l906_90613

-- Defining constants and parameters based on the conditions
def households_per_day : ℕ := 20
def money_per_pair : ℕ := 40
def total_money_collected : ℕ := 2000
def money_from_households : ℕ := (households_per_day / 2) * money_per_pair

-- The theorem that needs to be proven
theorem days_collected_money :
  (total_money_collected / money_from_households) = 5 :=
sorry -- Proof not provided

end days_collected_money_l906_90613


namespace petya_result_less_than_one_tenth_l906_90657

theorem petya_result_less_than_one_tenth 
  (a b c d e f : ℕ) 
  (ha: a.gcd b = 1) (hb: c.gcd d = 1)
  (hc: e.gcd f = 1) 
  (vasya_correct: (a / b) + (c / d) + (e / f) = 1) :
  (a + c + e) / (b + d + f) < 1 / 10 :=
by
  -- proof goes here
  sorry

end petya_result_less_than_one_tenth_l906_90657


namespace wholesale_price_is_90_l906_90653

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l906_90653


namespace minimum_value_exists_l906_90604

-- Definitions of the components
noncomputable def quadratic_expression (k x y : ℝ) : ℝ := 
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12

theorem minimum_value_exists (k : ℝ) :
  (∃ x y : ℝ, quadratic_expression k x y = 0) ↔ k = 2 := 
sorry

end minimum_value_exists_l906_90604


namespace vampire_count_after_two_nights_l906_90607

noncomputable def vampire_growth : Nat :=
  let first_night_new_vampires := 3 * 7
  let total_vampires_after_first_night := first_night_new_vampires + 3
  let second_night_new_vampires := total_vampires_after_first_night * (7 + 1)
  second_night_new_vampires + total_vampires_after_first_night

theorem vampire_count_after_two_nights : vampire_growth = 216 :=
by
  -- Skipping the detailed proof steps for now
  sorry

end vampire_count_after_two_nights_l906_90607


namespace number_of_integers_between_sqrt10_and_sqrt100_l906_90622

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l906_90622


namespace simplify_expression_l906_90685

variable {a : ℝ}

theorem simplify_expression (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by
  sorry

end simplify_expression_l906_90685


namespace probability_select_cooking_l906_90624

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l906_90624


namespace tan_subtraction_modified_l906_90636

theorem tan_subtraction_modified (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = (3 : ℝ) / (157465 : ℝ) := by
  have h3 : Real.tan (α - β) = (Real.tan α - Real.tan β) / (1 + (Real.tan α * Real.tan β)^3) :=
    sorry -- this is assumed as given in the conditions
  sorry -- rest of the proof

end tan_subtraction_modified_l906_90636


namespace digit_to_make_52B6_divisible_by_3_l906_90638

theorem digit_to_make_52B6_divisible_by_3 (B : ℕ) (hB : 0 ≤ B ∧ B ≤ 9) : 
  (5 + 2 + B + 6) % 3 = 0 ↔ (B = 2 ∨ B = 5 ∨ B = 8) := 
by
  sorry

end digit_to_make_52B6_divisible_by_3_l906_90638


namespace range_of_a_l906_90695

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end range_of_a_l906_90695


namespace rope_length_comparison_l906_90645

theorem rope_length_comparison
  (L : ℝ)
  (hL1 : L > 0) 
  (cut1 cut2 : ℝ)
  (hcut1 : cut1 = 0.3)
  (hcut2 : cut2 = 3) :
  L - cut1 > L - cut2 :=
by
  sorry

end rope_length_comparison_l906_90645


namespace mary_finds_eggs_l906_90676

theorem mary_finds_eggs (initial final found : ℕ) (h_initial : initial = 27) (h_final : final = 31) :
  found = final - initial → found = 4 :=
by
  intro h
  rw [h_initial, h_final] at h
  exact h

end mary_finds_eggs_l906_90676


namespace student_game_incorrect_statement_l906_90668

theorem student_game_incorrect_statement (a : ℚ) : ¬ (∀ a : ℚ, -a - 2 < 0) :=
by
  -- skip the proof for now
  sorry

end student_game_incorrect_statement_l906_90668


namespace triangle_sin_double_angle_l906_90648

open Real

theorem triangle_sin_double_angle (A : ℝ) (h : cos (π / 4 + A) = 5 / 13) : sin (2 * A) = 119 / 169 :=
by
  sorry

end triangle_sin_double_angle_l906_90648


namespace rectangle_diagonal_length_l906_90655

theorem rectangle_diagonal_length (P L W k d : ℝ) 
  (h1 : P = 72) 
  (h2 : L / W = 3 / 2) 
  (h3 : L = 3 * k) 
  (h4 : W = 2 * k) 
  (h5 : P = 2 * (L + W))
  (h6 : d = Real.sqrt ((L^2) + (W^2))) :
  d = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l906_90655


namespace rhombus_area_l906_90670

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 150 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l906_90670


namespace renovation_project_cement_loads_l906_90642

theorem renovation_project_cement_loads
  (s : ℚ) (d : ℚ) (t : ℚ)
  (hs : s = 0.16666666666666666) 
  (hd : d = 0.3333333333333333)
  (ht : t = 0.6666666666666666) :
  t - (s + d) = 0.1666666666666666 := by
  sorry

end renovation_project_cement_loads_l906_90642


namespace yonderland_license_plates_l906_90628

/-!
# Valid License Plates in Yonderland

A valid license plate in Yonderland consists of three letters followed by four digits. 

We are tasked with determining the number of valid license plates possible under this format.
-/

def num_letters : ℕ := 26
def num_digits : ℕ := 10
def letter_combinations : ℕ := num_letters ^ 3
def digit_combinations : ℕ := num_digits ^ 4
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem yonderland_license_plates : total_combinations = 175760000 := by
  sorry

end yonderland_license_plates_l906_90628


namespace cos_min_sin_eq_neg_sqrt_seven_half_l906_90608

variable (θ : ℝ)

theorem cos_min_sin_eq_neg_sqrt_seven_half (h1 : Real.sin θ + Real.cos θ = 0.5)
    (h2 : π / 2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = - Real.sqrt 7 / 2 := by
  sorry

end cos_min_sin_eq_neg_sqrt_seven_half_l906_90608


namespace total_assembly_time_l906_90643

-- Define the conditions
def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8
def total_pieces : ℕ := chairs + tables

-- State the theorem
theorem total_assembly_time :
  total_pieces * time_per_piece = 32 :=
sorry

end total_assembly_time_l906_90643


namespace complex_multiplication_l906_90637

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- The theorem stating the equality
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i := 
sorry

end complex_multiplication_l906_90637


namespace cakes_left_correct_l906_90665

def number_of_cakes_left (total_cakes sold_cakes : ℕ) : ℕ :=
  total_cakes - sold_cakes

theorem cakes_left_correct :
  number_of_cakes_left 54 41 = 13 :=
by
  sorry

end cakes_left_correct_l906_90665


namespace weights_system_l906_90614

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l906_90614


namespace total_amount_shared_l906_90641

-- Define the amounts for Ken and Tony based on the conditions
def ken_amt : ℤ := 1750
def tony_amt : ℤ := 2 * ken_amt

-- The proof statement that the total amount shared is $5250
theorem total_amount_shared : ken_amt + tony_amt = 5250 :=
by 
  sorry

end total_amount_shared_l906_90641


namespace repeating_decimal_fraction_eq_l906_90693

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end repeating_decimal_fraction_eq_l906_90693


namespace suki_bag_weight_is_22_l906_90672

noncomputable def weight_of_suki_bag : ℝ :=
  let bags_suki := 6.5
  let bags_jimmy := 4.5
  let weight_jimmy_per_bag := 18.0
  let total_containers := 28
  let weight_per_container := 8.0
  let total_weight_jimmy := bags_jimmy * weight_jimmy_per_bag
  let total_weight_combined := total_containers * weight_per_container
  let total_weight_suki := total_weight_combined - total_weight_jimmy
  total_weight_suki / bags_suki

theorem suki_bag_weight_is_22 : weight_of_suki_bag = 22 :=
by
  sorry

end suki_bag_weight_is_22_l906_90672


namespace geometric_sequence_sum_l906_90688

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h_geo : ∀ n, a (n + 1) = (3 : ℝ) * ((-2 : ℝ) ^ n))
  (h_first : a 1 = 3)
  (h_ratio_ne_1 : -2 ≠ 1)
  (h_arith : 2 * a 3 = a 4 + a 5) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 33 := 
sorry

end geometric_sequence_sum_l906_90688


namespace combined_marble_remainder_l906_90679

theorem combined_marble_remainder (l j : ℕ) (h_l : l % 8 = 5) (h_j : j % 8 = 6) : (l + j) % 8 = 3 := by
  sorry

end combined_marble_remainder_l906_90679


namespace perpendicular_lines_values_of_a_l906_90686

theorem perpendicular_lines_values_of_a (a : ℝ) :
  (∃ (a : ℝ), (∀ x y : ℝ, a * x - y + 2 * a = 0 ∧ (2 * a - 1) * x + a * y = 0) 
    ↔ (a = 0 ∨ a = 1))
  := sorry

end perpendicular_lines_values_of_a_l906_90686


namespace LineChart_characteristics_and_applications_l906_90602

-- Definitions related to question and conditions
def LineChart : Type := sorry
def represents_amount (lc : LineChart) : Prop := sorry
def reflects_increase_or_decrease (lc : LineChart) : Prop := sorry

-- Theorem related to the correct answer
theorem LineChart_characteristics_and_applications (lc : LineChart) :
  represents_amount lc ∧ reflects_increase_or_decrease lc :=
sorry

end LineChart_characteristics_and_applications_l906_90602


namespace coin_heads_probability_l906_90681

theorem coin_heads_probability
    (prob_tails : ℚ := 1/2)
    (prob_specific_sequence : ℚ := 0.0625)
    (flips : ℕ := 4)
    (ht : prob_tails = 1 / 2)
    (hs : prob_specific_sequence = (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) 
    : ∀ (p_heads : ℚ), p_heads = 1 - prob_tails := by
  sorry

end coin_heads_probability_l906_90681


namespace oil_leakage_problem_l906_90656

theorem oil_leakage_problem :
    let l_A := 25  -- Leakage rate of Pipe A (gallons/hour)
    let l_B := 37  -- Leakage rate of Pipe B (gallons/hour)
    let l_C := 55  -- Leakage rate of Pipe C (gallons/hour)
    let l_D := 41  -- Leakage rate of Pipe D (gallons/hour)
    let l_E := 30  -- Leakage rate of Pipe E (gallons/hour)

    let t_A := 10  -- Time taken to fix Pipe A (hours)
    let t_B := 7   -- Time taken to fix Pipe B (hours)
    let t_C := 12  -- Time taken to fix Pipe C (hours)
    let t_D := 9   -- Time taken to fix Pipe D (hours)
    let t_E := 14  -- Time taken to fix Pipe E (hours)

    let leak_A := l_A * t_A  -- Total leaked from Pipe A (gallons)
    let leak_B := l_B * t_B  -- Total leaked from Pipe B (gallons)
    let leak_C := l_C * t_C  -- Total leaked from Pipe C (gallons)
    let leak_D := l_D * t_D  -- Total leaked from Pipe D (gallons)
    let leak_E := l_E * t_E  -- Total leaked from Pipe E (gallons)
  
    let overall_total := leak_A + leak_B + leak_C + leak_D + leak_E
  
    leak_A = 250 ∧
    leak_B = 259 ∧
    leak_C = 660 ∧
    leak_D = 369 ∧
    leak_E = 420 ∧
    overall_total = 1958 :=
by
    sorry

end oil_leakage_problem_l906_90656


namespace solve_system_of_equations_l906_90659

theorem solve_system_of_equations (x y : ℝ) :
  (x^4 + (7/2) * x^2 * y + 2 * y^3 = 0) ∧
  (4 * x^2 + 7 * x * y + 2 * y^3 = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -1) ∨ (x = -11 / 2 ∧ y = -11 / 2) :=
sorry

end solve_system_of_equations_l906_90659


namespace bicycle_final_price_l906_90612

theorem bicycle_final_price : 
  let original_price := 200 
  let weekend_discount := 0.40 * original_price 
  let price_after_weekend_discount := original_price - weekend_discount 
  let wednesday_discount := 0.20 * price_after_weekend_discount 
  let final_price := price_after_weekend_discount - wednesday_discount 
  final_price = 96 := 
by 
  sorry

end bicycle_final_price_l906_90612


namespace janet_counts_total_birds_l906_90639

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end janet_counts_total_birds_l906_90639


namespace total_distance_walked_l906_90626

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l906_90626


namespace jenny_chocolate_squares_l906_90663

theorem jenny_chocolate_squares (mike_chocolates : ℕ) (jenny_chocolates : ℕ) 
  (h_mike : mike_chocolates = 20) 
  (h_jenny : jenny_chocolates = 3 * mike_chocolates + 5) :
  jenny_chocolates = 65 :=
by
  sorry

end jenny_chocolate_squares_l906_90663


namespace right_regular_prism_impossible_sets_l906_90677

-- Define a function to check if a given set of numbers {x, y, z} forms an invalid right regular prism
def not_possible (x y z : ℕ) : Prop := (x^2 + y^2 ≤ z^2)

-- Define individual propositions for the given sets of numbers
def set_a : Prop := not_possible 3 4 6
def set_b : Prop := not_possible 5 5 8
def set_e : Prop := not_possible 7 8 12

-- Define our overall proposition that these sets cannot be the lengths of the external diagonals of a right regular prism
theorem right_regular_prism_impossible_sets : 
  set_a ∧ set_b ∧ set_e :=
by
  -- Proof is omitted
  sorry

end right_regular_prism_impossible_sets_l906_90677


namespace max_total_length_of_cuts_l906_90696

theorem max_total_length_of_cuts (A : ℕ) (n : ℕ) (m : ℕ) (P : ℕ) (Q : ℕ)
  (h1 : A = 30 * 30)
  (h2 : n = 225)
  (h3 : m = A / n)
  (h4 : m = 4)
  (h5 : Q = 4 * 30)
  (h6 : P = 225 * 10 - Q)
  (h7 : P / 2 = 1065) :
  P / 2 = 1065 :=
by 
  exact h7

end max_total_length_of_cuts_l906_90696


namespace polynomial_perfect_square_trinomial_l906_90675

theorem polynomial_perfect_square_trinomial (k : ℝ) :
  (∀ x : ℝ, 4 * x^2 + 2 * k * x + 25 = (2 * x + 5) * (2 * x + 5)) → (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_trinomial_l906_90675


namespace sculpture_and_base_height_l906_90674

def height_sculpture_ft : ℕ := 2
def height_sculpture_in : ℕ := 10
def height_base_in : ℕ := 2

def total_height_in (ft : ℕ) (inch1 inch2 : ℕ) : ℕ :=
  (ft * 12) + inch1 + inch2

def total_height_ft (total_in : ℕ) : ℕ :=
  total_in / 12

theorem sculpture_and_base_height :
  total_height_ft (total_height_in height_sculpture_ft height_sculpture_in height_base_in) = 3 :=
by
  sorry

end sculpture_and_base_height_l906_90674


namespace rachel_pool_fill_time_l906_90629

theorem rachel_pool_fill_time :
  ∀ (pool_volume : ℕ) (num_hoses : ℕ) (hose_rate : ℕ),
  pool_volume = 30000 →
  num_hoses = 5 →
  hose_rate = 3 →
  (pool_volume / (num_hoses * hose_rate * 60) : ℤ) = 33 :=
by
  intros pool_volume num_hoses hose_rate h1 h2 h3
  sorry

end rachel_pool_fill_time_l906_90629


namespace order_of_f_l906_90601

-- Define the function f
variables {f : ℝ → ℝ}

-- Definition of even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of monotonic increasing function on [0, +∞)
def monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x ≤ y) → f x ≤ f y

-- The main problem statement
theorem order_of_f (h_even : even_function f) (h_mono : monotonically_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
  sorry

end order_of_f_l906_90601


namespace range_of_m_l906_90683

theorem range_of_m (m: ℝ) : (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → x^2 - x + 1 > 2*x + m) → m < -1 :=
by
  intro h
  sorry

end range_of_m_l906_90683


namespace find_divisor_l906_90600

theorem find_divisor (d q r : ℕ) :
  (919 = d * q + r) → (q = 17) → (r = 11) → d = 53 :=
by
  sorry

end find_divisor_l906_90600


namespace shorter_leg_of_right_triangle_l906_90605

theorem shorter_leg_of_right_triangle (a b : ℕ) (h1 : a < b)
    (h2 : a^2 + b^2 = 65^2) : a = 16 :=
sorry

end shorter_leg_of_right_triangle_l906_90605


namespace boat_speed_in_still_water_l906_90652

def speed_of_stream : ℝ := 8
def downstream_distance : ℝ := 64
def upstream_distance : ℝ := 32

theorem boat_speed_in_still_water (x : ℝ) (t : ℝ) 
  (HS_downstream : t = downstream_distance / (x + speed_of_stream)) 
  (HS_upstream : t = upstream_distance / (x - speed_of_stream)) :
  x = 24 := by
  sorry

end boat_speed_in_still_water_l906_90652


namespace sushi_cost_l906_90616

variable (x : ℕ)

theorem sushi_cost (h1 : 9 * x = 180) : x + (9 * x) = 200 :=
by 
  sorry

end sushi_cost_l906_90616


namespace cost_of_each_toy_l906_90673

theorem cost_of_each_toy (initial_money spent_money remaining_money toys_count toy_cost : ℕ) 
  (h1 : initial_money = 57)
  (h2 : spent_money = 27)
  (h3 : remaining_money = initial_money - spent_money)
  (h4 : toys_count = 5)
  (h5 : remaining_money / toys_count = toy_cost) :
  toy_cost = 6 :=
by
  sorry

end cost_of_each_toy_l906_90673


namespace total_population_l906_90658

variable (b g t s : ℕ)

theorem total_population (hb : b = 4 * g) (hg : g = 8 * t) (ht : t = 2 * s) :
  b + g + t + s = (83 * g) / 16 :=
by sorry

end total_population_l906_90658


namespace f_divisible_by_13_l906_90609

def f : ℕ → ℤ := sorry

theorem f_divisible_by_13 :
  (f 0 = 0) ∧ (f 1 = 0) ∧
  (∀ n, f (n + 2) = 4 ^ (n + 2) * f (n + 1) - 16 ^ (n + 1) * f n + n * 2 ^ (n ^ 2)) →
  (f 1989 % 13 = 0) ∧ (f 1990 % 13 = 0) ∧ (f 1991 % 13 = 0) :=
by
  intros h
  sorry

end f_divisible_by_13_l906_90609


namespace determine_x_l906_90694

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end determine_x_l906_90694


namespace parallel_lines_slope_l906_90692

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l906_90692


namespace plant_supplier_earnings_l906_90606

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l906_90606


namespace lucas_siblings_product_is_35_l906_90661

-- Definitions based on the given conditions
def total_girls (lauren_sisters : ℕ) : ℕ := lauren_sisters + 1
def total_boys (lauren_brothers : ℕ) : ℕ := lauren_brothers + 1

-- Given conditions
def lauren_sisters : ℕ := 4
def lauren_brothers : ℕ := 7

-- Compute number of sisters (S) and brothers (B) Lucas has
def lucas_sisters : ℕ := total_girls lauren_sisters
def lucas_brothers : ℕ := lauren_brothers

theorem lucas_siblings_product_is_35 : 
  (lucas_sisters * lucas_brothers = 35) := by
  -- Asserting the correctness based on given family structure conditions
  sorry

end lucas_siblings_product_is_35_l906_90661


namespace inverse_proportion_expression_and_calculation_l906_90666

theorem inverse_proportion_expression_and_calculation :
  (∃ k : ℝ, (∀ (x y : ℝ), y = k / x) ∧
   (∀ x y : ℝ, y = 400 ∧ x = 0.25 → k = 100) ∧
   (∀ x : ℝ, 200 = 100 / x → x = 0.5)) :=
by
  sorry

end inverse_proportion_expression_and_calculation_l906_90666


namespace min_value_a_plus_8b_min_value_a_plus_8b_min_l906_90680

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  a + 8 * b ≥ 9 :=
by sorry

-- The minimum value is 9 (achievable at specific values of a and b)
theorem min_value_a_plus_8b_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a * b = a + 2 * b ∧ a + 8 * b = 9 :=
by sorry

end min_value_a_plus_8b_min_value_a_plus_8b_min_l906_90680


namespace emmanuel_jelly_beans_l906_90660

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end emmanuel_jelly_beans_l906_90660


namespace farmer_purchase_l906_90640

theorem farmer_purchase : ∃ r c : ℕ, 30 * r + 45 * c = 1125 ∧ r > 0 ∧ c > 0 ∧ r = 3 ∧ c = 23 := 
by 
  sorry

end farmer_purchase_l906_90640


namespace negation_of_proposition_l906_90631

-- Definitions using the conditions stated
def p (x : ℝ) : Prop := x^2 - x + 1/4 ≥ 0

-- The statement to prove
theorem negation_of_proposition :
  (¬ (∀ x : ℝ, p x)) = (∃ x : ℝ, ¬ p x) :=
by
  -- Proof will go here; replaced by sorry as per instruction
  sorry

end negation_of_proposition_l906_90631


namespace polygon_area_l906_90632

theorem polygon_area (sides : ℕ) (perpendicular_adjacent : Bool) (congruent_sides : Bool) (perimeter : ℝ) (area : ℝ) :
  sides = 32 → 
  perpendicular_adjacent = true → 
  congruent_sides = true →
  perimeter = 64 →
  area = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end polygon_area_l906_90632


namespace tv_show_years_l906_90617

theorem tv_show_years (s1 s2 s3 : ℕ) (e1 e2 e3 : ℕ) (avg : ℕ) :
  s1 = 8 → e1 = 15 →
  s2 = 4 → e2 = 20 →
  s3 = 2 → e3 = 12 →
  avg = 16 →
  (s1 * e1 + s2 * e2 + s3 * e3) / avg = 14 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end tv_show_years_l906_90617


namespace john_has_18_blue_pens_l906_90646

variables (R B Bl : ℕ)

-- Conditions from the problem
def john_has_31_pens : Prop := R + B + Bl = 31
def black_pens_5_more_than_red : Prop := B = R + 5
def blue_pens_twice_black : Prop := Bl = 2 * B

theorem john_has_18_blue_pens :
  john_has_31_pens R B Bl ∧ black_pens_5_more_than_red R B ∧ blue_pens_twice_black B Bl →
  Bl = 18 :=
by
  sorry

end john_has_18_blue_pens_l906_90646


namespace circumcircle_radius_proof_l906_90697

noncomputable def circumcircle_radius (AB A S : ℝ) : ℝ :=
  if AB = 3 ∧ A = 120 ∧ S = 9 * Real.sqrt 3 / 4 then 3 else 0

theorem circumcircle_radius_proof :
  circumcircle_radius 3 120 (9 * Real.sqrt 3 / 4) = 3 := by
  sorry

end circumcircle_radius_proof_l906_90697


namespace distinct_value_expression_l906_90623

def tri (a b : ℕ) : ℕ := min a b
def nabla (a b : ℕ) : ℕ := max a b

theorem distinct_value_expression (x : ℕ) : (nabla 5 (nabla 4 (tri x 4))) = 5 := 
by
  sorry

end distinct_value_expression_l906_90623


namespace euclidean_division_l906_90654

theorem euclidean_division (a b : ℕ) (hb : b ≠ 0) : ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ a = b * q + r :=
by sorry

end euclidean_division_l906_90654


namespace find_f_neg_two_l906_90690

def is_even_function (f : ℝ → ℝ) (h : ℝ → ℝ) := ∀ x, h (-x) = h x

theorem find_f_neg_two (f : ℝ → ℝ) (h : ℝ → ℝ) (hx : ∀ x, h x = f (2*x) + x)
  (h_even : is_even_function f h) 
  (h_f_two : f 2 = 1) : 
  f (-2) = 3 :=
  by
    sorry

end find_f_neg_two_l906_90690


namespace yasna_finish_books_in_two_weeks_l906_90678

theorem yasna_finish_books_in_two_weeks (pages_book1 : ℕ) (pages_book2 : ℕ) (pages_per_day : ℕ) (days_per_week : ℕ) 
  (h1 : pages_book1 = 180) (h2 : pages_book2 = 100) (h3 : pages_per_day = 20) (h4 : days_per_week = 7) : 
  ((pages_book1 + pages_book2) / pages_per_day) / days_per_week = 2 := 
by
  sorry

end yasna_finish_books_in_two_weeks_l906_90678


namespace value_of_b_l906_90610

theorem value_of_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := 
by 
  sorry

end value_of_b_l906_90610


namespace definitely_incorrect_conclusions_l906_90618

theorem definitely_incorrect_conclusions (a b c : ℝ) (x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : a * x2^2 + b * x2 + c = 0)
  (h3 : x1 > 0) 
  (h4 : x2 > 0) 
  (h5 : x1 + x2 = -b / a) 
  (h6 : x1 * x2 = c / a) : 
  (a > 0 ∧ b > 0 ∧ c > 0) = false ∧ 
  (a < 0 ∧ b < 0 ∧ c < 0) = false ∧ 
  (a > 0 ∧ b < 0 ∧ c < 0) = true ∧ 
  (a < 0 ∧ b > 0 ∧ c > 0) = true :=
sorry

end definitely_incorrect_conclusions_l906_90618


namespace ratio_of_speeds_l906_90615

variable (x y n : ℝ)

-- Conditions
def condition1 : Prop := 3 * (x - y) = n
def condition2 : Prop := 2 * (x + y) = n

-- Problem Statement
theorem ratio_of_speeds (h1 : condition1 x y n) (h2 : condition2 x y n) : x = 5 * y :=
by
  sorry

end ratio_of_speeds_l906_90615


namespace floor_equation_solution_l906_90698

open Int

theorem floor_equation_solution (x : ℝ) :
  (⌊ ⌊ 3 * x ⌋ - 1/2 ⌋ = ⌊ x + 4 ⌋) ↔ (7/3 ≤ x ∧ x < 3) := sorry

end floor_equation_solution_l906_90698


namespace annual_income_is_correct_l906_90647

noncomputable def total_investment : ℝ := 4455
noncomputable def price_per_share : ℝ := 8.25
noncomputable def dividend_rate : ℝ := 12 / 100
noncomputable def face_value : ℝ := 10

noncomputable def number_of_shares : ℝ := total_investment / price_per_share
noncomputable def dividend_per_share : ℝ := dividend_rate * face_value
noncomputable def annual_income : ℝ := dividend_per_share * number_of_shares

theorem annual_income_is_correct : annual_income = 648 := by
  sorry

end annual_income_is_correct_l906_90647


namespace tan_sum_identity_l906_90682

open Real

theorem tan_sum_identity : 
  tan (80 * π / 180) + tan (40 * π / 180) - sqrt 3 * tan (80 * π / 180) * tan (40 * π / 180) = -sqrt 3 :=
by
  sorry

end tan_sum_identity_l906_90682


namespace compare_fractions_neg_l906_90651

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l906_90651


namespace maximum_partial_sum_l906_90649

theorem maximum_partial_sum (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
    (h_arith_seq : ∀ n, a n = a 0 + n * d)
    (h8_13 : 3 * a 8 = 5 * a 13)
    (h_pos : a 0 > 0)
    (h_sn_def : ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2) :
  S 20 = max (max (S 10) (S 11)) (max (S 20) (S 21)) := 
sorry

end maximum_partial_sum_l906_90649


namespace hyperbola_correct_l906_90671

noncomputable def hyperbola_properties : Prop :=
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  let b := Real.sqrt ((c^2) - (a^2))
  (h + k + a + b = 4 * Real.sqrt 3 + 6)

theorem hyperbola_correct : hyperbola_properties :=
by
  unfold hyperbola_properties
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  have b : ℝ := Real.sqrt ((c^2) - (a^2))
  sorry

end hyperbola_correct_l906_90671


namespace larger_number_of_two_l906_90650

theorem larger_number_of_two (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 29) (h3 : x * y > 200) : x = 16 :=
by sorry

end larger_number_of_two_l906_90650


namespace arg_cubed_eq_pi_l906_90691

open Complex

theorem arg_cubed_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : 
  arg (z2 / z1) ^ 3 = π :=
by
  sorry

end arg_cubed_eq_pi_l906_90691


namespace range_m_l906_90630

theorem range_m (m : ℝ) : 
  (∀ x : ℝ, ((m * x - 1) * (x - 2) > 0) ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_m_l906_90630


namespace inspection_probability_l906_90684

noncomputable def defective_items : ℕ := 2
noncomputable def good_items : ℕ := 3
noncomputable def total_items : ℕ := defective_items + good_items

/-- Given 2 defective items and 3 good items mixed together,
the probability that the inspection stops exactly after
four inspections is 3/5 --/
theorem inspection_probability :
  (2 * (total_items - 1) * total_items / (total_items * (total_items - 1) * (total_items - 2) * (total_items - 3))) = (3 / 5) :=
by
  sorry

end inspection_probability_l906_90684


namespace min_next_score_to_increase_avg_l906_90621

def Liam_initial_scores : List ℕ := [72, 85, 78, 66, 90, 82]

def current_average (scores: List ℕ) : ℚ :=
  (scores.sum / scores.length : ℚ)

def next_score_requirement (initial_scores: List ℕ) (desired_increase: ℚ) : ℚ :=
  let current_avg := current_average initial_scores
  let desired_avg := current_avg + desired_increase
  let total_tests := initial_scores.length + 1
  let total_required := desired_avg * total_tests
  total_required - initial_scores.sum

theorem min_next_score_to_increase_avg :
  next_score_requirement Liam_initial_scores 5 = 115 := by
  sorry

end min_next_score_to_increase_avg_l906_90621


namespace simplify_expression_l906_90664

theorem simplify_expression (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) (hx1 : x ≠ 1) : 
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by
  sorry

end simplify_expression_l906_90664


namespace intercepts_of_line_l906_90634

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) :
  (∃ x_intercept : ℝ, x_intercept = 7 ∧ (4 * x_intercept + 7 * 0 = 28)) ∧
  (∃ y_intercept : ℝ, y_intercept = 4 ∧ (4 * 0 + 7 * y_intercept = 28)) :=
by
  sorry

end intercepts_of_line_l906_90634


namespace real_imag_equal_complex_l906_90625

/-- Given i is the imaginary unit, and a is a real number,
if the real part and the imaginary part of the complex number -3i(a+i) are equal,
then a = -1. -/
theorem real_imag_equal_complex (a : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_eq : (3 : ℂ) = -(3 : ℂ) * a * i) : a = -1 :=
sorry

end real_imag_equal_complex_l906_90625


namespace employee_pay_l906_90603

variable (X Y Z : ℝ)

-- Conditions
def X_pay (Y : ℝ) := 1.2 * Y
def Z_pay (X : ℝ) := 0.75 * X

-- Proof statement
theorem employee_pay (h1 : X = X_pay Y) (h2 : Z = Z_pay X) (total_pay : X + Y + Z = 1540) : 
  X + Y + Z = 1540 :=
by
  sorry

end employee_pay_l906_90603


namespace angle_perpendicular_vectors_l906_90699

theorem angle_perpendicular_vectors (α : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : (1 : ℝ) * Real.sin α + Real.cos α * (1 : ℝ) = 0) : α = 3 * Real.pi / 4 :=
sorry

end angle_perpendicular_vectors_l906_90699


namespace number_of_cars_l906_90669

theorem number_of_cars (b c : ℕ) (h1 : b = c / 10) (h2 : c - b = 90) : c = 100 :=
by
  sorry

end number_of_cars_l906_90669


namespace complete_collection_probability_l906_90662

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l906_90662


namespace cost_of_replaced_tomatoes_l906_90667

def original_order : ℝ := 25
def delivery_tip : ℝ := 8
def new_total : ℝ := 35
def original_tomatoes : ℝ := 0.99
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00

def increase_in_lettuce := new_lettuce - original_lettuce
def increase_in_celery := new_celery - original_celery
def total_increase_except_tomatoes := increase_in_lettuce + increase_in_celery
def original_total_with_delivery := original_order + delivery_tip
def total_increase := new_total - original_total_with_delivery
def increase_due_to_tomatoes := total_increase - total_increase_except_tomatoes
def replaced_tomatoes := original_tomatoes + increase_due_to_tomatoes

theorem cost_of_replaced_tomatoes : replaced_tomatoes = 2.20 := by
  sorry

end cost_of_replaced_tomatoes_l906_90667


namespace percent_increase_from_first_to_second_quarter_l906_90619

theorem percent_increase_from_first_to_second_quarter 
  (P : ℝ) :
  ((1.60 * P - 1.20 * P) / (1.20 * P)) * 100 = 33.33 := by
  sorry

end percent_increase_from_first_to_second_quarter_l906_90619


namespace find_c_l906_90689

theorem find_c (a b c d y1 y2 : ℝ) (h1 : y1 = a * 2^3 + b * 2^2 + c * 2 + d)
  (h2 : y2 = a * (-2)^3 + b * (-2)^2 + c * (-2) + d)
  (h3 : y1 - y2 = 12) : c = 3 - 4 * a := by
  sorry

end find_c_l906_90689


namespace David_min_max_rides_l906_90627

-- Definitions based on the conditions
variable (Alena_rides : ℕ := 11)
variable (Bara_rides : ℕ := 20)
variable (Cenek_rides : ℕ := 4)
variable (every_pair_rides_at_least_once : Prop := true)

-- Hypotheses for the problem
axiom Alena_has_ridden : Alena_rides = 11
axiom Bara_has_ridden : Bara_rides = 20
axiom Cenek_has_ridden : Cenek_rides = 4
axiom Pairs_have_ridden : every_pair_rides_at_least_once

-- Statement for the minimum and maximum rides of David
theorem David_min_max_rides (David_rides : ℕ) :
  (David_rides = 11) ∨ (David_rides = 29) :=
sorry

end David_min_max_rides_l906_90627


namespace change_sum_equals_108_l906_90633

theorem change_sum_equals_108 :
  ∃ (amounts : List ℕ), (∀ a ∈ amounts, a < 100 ∧ ((a % 25 = 4) ∨ (a % 5 = 4))) ∧
    amounts.sum = 108 := 
by
  sorry

end change_sum_equals_108_l906_90633


namespace min_value_function_l906_90687

open Real

theorem min_value_function (x y : ℝ) 
  (hx : x > -2 ∧ x < 2) 
  (hy : y > -2 ∧ y < 2) 
  (hxy : x * y = -1) : 
  (∃ u : ℝ, u = (4 / (4 - x^2) + 9 / (9 - y^2)) ∧ u = 12 / 5) :=
sorry

end min_value_function_l906_90687


namespace trigonometric_identity_l906_90635

theorem trigonometric_identity (φ : ℝ) 
  (h : Real.cos (π / 2 + φ) = (Real.sqrt 3) / 2) : 
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l906_90635


namespace steven_owes_jeremy_l906_90611

-- Definitions for the conditions
def base_payment_per_room := (13 : ℚ) / 3
def rooms_cleaned := (5 : ℚ) / 2
def additional_payment_per_room := (1 : ℚ) / 2

-- Define the total amount of money Steven owes Jeremy
def total_payment (base_payment_per_room rooms_cleaned additional_payment_per_room : ℚ) : ℚ :=
  let base_payment := base_payment_per_room * rooms_cleaned
  let additional_payment := if rooms_cleaned > 2 then additional_payment_per_room * rooms_cleaned else 0
  base_payment + additional_payment

-- The statement to prove
theorem steven_owes_jeremy :
  total_payment base_payment_per_room rooms_cleaned additional_payment_per_room = 145 / 12 :=
by
  sorry

end steven_owes_jeremy_l906_90611


namespace rectangle_perimeter_l906_90620

-- Define the conditions
variables (z w : ℕ)
-- Define the side lengths of the rectangles
def rectangle_long_side := z - w
def rectangle_short_side := w

-- Theorem: The perimeter of one of the four rectangles
theorem rectangle_perimeter : 2 * (rectangle_long_side z w) + 2 * (rectangle_short_side w) = 2 * z :=
by sorry

end rectangle_perimeter_l906_90620
