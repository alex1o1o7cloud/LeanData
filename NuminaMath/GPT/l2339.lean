import Mathlib

namespace jill_peaches_l2339_233991

-- Definitions based on conditions in a
def Steven_has_peaches : ℕ := 19
def Steven_more_than_Jill : ℕ := 13

-- Statement to prove Jill's peaches
theorem jill_peaches : (Steven_has_peaches - Steven_more_than_Jill = 6) :=
by
  sorry

end jill_peaches_l2339_233991


namespace powers_of_2_form_6n_plus_8_l2339_233968

noncomputable def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2 ^ k

def of_the_form (n : ℕ) : ℕ := 6 * n + 8

def is_odd_greater_than_one (k : ℕ) : Prop := k % 2 = 1 ∧ k > 1

theorem powers_of_2_form_6n_plus_8 (k : ℕ) (n : ℕ) :
  (2 ^ k = of_the_form n) ↔ is_odd_greater_than_one k :=
sorry

end powers_of_2_form_6n_plus_8_l2339_233968


namespace least_value_of_x_for_divisibility_l2339_233961

theorem least_value_of_x_for_divisibility (x : ℕ) (h : 1 + 8 + 9 + 4 = 22) :
  ∃ x : ℕ, (22 + x) % 3 = 0 ∧ x = 2 := by
sorry

end least_value_of_x_for_divisibility_l2339_233961


namespace width_of_wall_l2339_233970

theorem width_of_wall (l : ℕ) (w : ℕ) (hl : l = 170) (hw : w = 5 * l + 80) : w = 930 := 
by
  sorry

end width_of_wall_l2339_233970


namespace find_n_plus_m_l2339_233976

noncomputable def f (x : ℝ) := abs (Real.log x / Real.log 2)

theorem find_n_plus_m (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : f m = f n) (h5 : ∀ x, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
    n + m = 5 / 2 := sorry

end find_n_plus_m_l2339_233976


namespace work_completion_days_l2339_233939

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days_l2339_233939


namespace value_of_a_g_odd_iff_m_eq_one_l2339_233960

noncomputable def f (a x : ℝ) : ℝ := a ^ x

noncomputable def g (m x a : ℝ) : ℝ := m - 2 / (f a x + 1)

theorem value_of_a
  (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_diff : ∀ x y : ℝ, x ∈ (Set.Icc 1 2) → y ∈ (Set.Icc 1 2) → abs (f a x - f a y) = 2) :
  a = 2 :=
sorry

theorem g_odd_iff_m_eq_one
  (a m : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_a_eq : a = 2) :
  (∀ x : ℝ, g m x a = -g m (-x) a) ↔ m = 1 :=
sorry

end value_of_a_g_odd_iff_m_eq_one_l2339_233960


namespace almonds_received_by_amanda_l2339_233965

variable (totalAlmonds : ℚ)
variable (numberOfPiles : ℚ)
variable (pilesForAmanda : ℚ)

-- Conditions
def stephanie_has_almonds := totalAlmonds = 66 / 7
def distribute_equally_into_piles := numberOfPiles = 6
def amanda_receives_piles := pilesForAmanda = 3

-- Conclusion to prove
theorem almonds_received_by_amanda :
  stephanie_has_almonds totalAlmonds →
  distribute_equally_into_piles numberOfPiles →
  amanda_receives_piles pilesForAmanda →
  (totalAlmonds / numberOfPiles) * pilesForAmanda = 33 / 7 :=
by
  sorry

end almonds_received_by_amanda_l2339_233965


namespace negation_of_universal_l2339_233993
-- Import the Mathlib library to provide the necessary mathematical background

-- State the theorem that we want to prove. This will state that the negation of the universal proposition is an existential proposition
theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0)) ↔ (∃ x : ℝ, x ≤ 0) :=
sorry

end negation_of_universal_l2339_233993


namespace opposite_of_negative_seven_l2339_233979

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l2339_233979


namespace boys_and_girls_total_l2339_233916

theorem boys_and_girls_total (c : ℕ) (h_lollipop_fraction : c = 90) 
  (h_one_third_lollipops : c / 3 = 30)
  (h_lollipops_shared : 30 / 3 = 10) 
  (h_candy_caness_shared : 60 / 2 = 30) : 
  10 + 30 = 40 :=
by
  simp [h_one_third_lollipops, h_lollipops_shared, h_candy_caness_shared]

end boys_and_girls_total_l2339_233916


namespace sufficient_not_necessary_condition_l2339_233900

variable (a b c : ℝ)

-- Define the condition that the sequence forms a geometric sequence
def geometric_sequence (a1 a2 a3 a4 a5 : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ a1 * q = a2 ∧ a2 * q = a3 ∧ a3 * q = a4 ∧ a4 * q = a5

-- Lean statement proving the problem
theorem sufficient_not_necessary_condition :
  (geometric_sequence 1 a b c 16) → (b = 4) ∧ ¬ (b = 4 → geometric_sequence 1 a b c 16) :=
sorry

end sufficient_not_necessary_condition_l2339_233900


namespace find_x_l2339_233932

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l2339_233932


namespace solve_for_x_l2339_233983

theorem solve_for_x (x : ℝ) (h : 4^x = Real.sqrt 64) : x = 3 / 2 :=
sorry

end solve_for_x_l2339_233983


namespace maximize_value_l2339_233986

def f (x : ℝ) : ℝ := -3 * x^2 - 8 * x + 18

theorem maximize_value : ∀ x : ℝ, f x ≤ f (-4/3) :=
by sorry

end maximize_value_l2339_233986


namespace louisa_second_day_distance_l2339_233920

-- Definitions based on conditions
def time_on_first_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_on_second_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

def condition (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) : Prop := 
  time_on_first_day distance_first_day speed + time_difference = time_on_second_day x speed

-- The proof statement
theorem louisa_second_day_distance (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) :
  distance_first_day = 240 → 
  speed = 60 → 
  time_difference = 3 → 
  condition distance_first_day speed time_difference x → 
  x = 420 :=
by
  intros h1 h2 h3 h4
  sorry

end louisa_second_day_distance_l2339_233920


namespace percentage_of_women_attended_picnic_l2339_233964

variable (E : ℝ) -- total number of employees
variable (M : ℝ) -- number of men
variable (W : ℝ) -- number of women

-- 45% of all employees are men
axiom h1 : M = 0.45 * E
-- Rest of employees are women
axiom h2 : W = E - M
-- 20% of men attended the picnic
variable (x : ℝ) -- percentage of women who attended the picnic
axiom h3 : 0.20 * M + (x / 100) * W = 0.31000000000000007 * E

theorem percentage_of_women_attended_picnic : x = 40 :=
by
  sorry

end percentage_of_women_attended_picnic_l2339_233964


namespace maximize_garden_area_length_l2339_233914

noncomputable def length_parallel_to_wall (cost_per_foot : ℝ) (fence_cost : ℝ) : ℝ :=
  let total_length := fence_cost / cost_per_foot 
  let y := total_length / 4 
  let length_parallel := total_length - 2 * y
  length_parallel

theorem maximize_garden_area_length :
  ∀ (cost_per_foot fence_cost : ℝ), cost_per_foot = 10 → fence_cost = 1500 → 
  length_parallel_to_wall cost_per_foot fence_cost = 75 :=
by
  intros
  simp [length_parallel_to_wall, *]
  sorry

end maximize_garden_area_length_l2339_233914


namespace option_d_correct_l2339_233980

theorem option_d_correct (a b : ℝ) (h : a * b < 0) : 
  (a / b + b / a) ≤ -2 := by
  sorry

end option_d_correct_l2339_233980


namespace width_of_field_l2339_233955

-- Definitions for the conditions
variables (W L : ℝ) (P : ℝ)
axiom length_condition : L = (7 / 5) * W
axiom perimeter_condition : P = 2 * L + 2 * W
axiom perimeter_value : P = 336

-- Theorem to be proved
theorem width_of_field : W = 70 :=
by
  -- Here will be the proof body
  sorry

end width_of_field_l2339_233955


namespace sum_divisible_by_15_l2339_233957

theorem sum_divisible_by_15 (a : ℤ) : 15 ∣ (9 * a^5 - 5 * a^3 - 4 * a) :=
sorry

end sum_divisible_by_15_l2339_233957


namespace total_games_played_l2339_233994

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l2339_233994


namespace find_k_values_l2339_233969

noncomputable def parallel_vectors (k : ℝ) : Prop :=
  (k^2 / k = (k + 1) / 4)

theorem find_k_values (k : ℝ) : parallel_vectors k ↔ (k = 0 ∨ k = 1 / 3) :=
by sorry

end find_k_values_l2339_233969


namespace net_increase_in_bicycle_stock_l2339_233915

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end net_increase_in_bicycle_stock_l2339_233915


namespace original_number_l2339_233940

theorem original_number (n : ℕ) (h1 : 2319 % 21 = 0) (h2 : 2319 = 21 * (n + 1) - 1) : n = 2318 := 
sorry

end original_number_l2339_233940


namespace f_decreasing_in_interval_l2339_233921

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shifted_g (x : ℝ) : ℝ := g (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := shifted_g (2 * x)

theorem f_decreasing_in_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x :=
by
  sorry

end f_decreasing_in_interval_l2339_233921


namespace minimum_value_quadratic_function_l2339_233938

noncomputable def quadratic_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem minimum_value_quadratic_function : ∀ x, x ≥ 0 → quadratic_function x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_function_l2339_233938


namespace total_players_count_l2339_233951

def kabadi_players : ℕ := 10
def kho_kho_only_players : ℕ := 35
def both_games_players : ℕ := 5

theorem total_players_count : kabadi_players + kho_kho_only_players - both_games_players = 40 :=
by
  sorry

end total_players_count_l2339_233951


namespace hours_rained_l2339_233952

theorem hours_rained (total_hours non_rain_hours rained_hours : ℕ)
 (h_total : total_hours = 8)
 (h_non_rain : non_rain_hours = 6)
 (h_rain_eq : rained_hours = total_hours - non_rain_hours) :
 rained_hours = 2 := 
by
  sorry

end hours_rained_l2339_233952


namespace solution_set_of_inequality_l2339_233918

open Set

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = Ioo (-2 : ℝ) 3 := 
sorry

end solution_set_of_inequality_l2339_233918


namespace tens_digit_3_pow_2016_eq_2_l2339_233928

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem tens_digit_3_pow_2016_eq_2 : tens_digit (3 ^ 2016) = 2 := by
  sorry

end tens_digit_3_pow_2016_eq_2_l2339_233928


namespace weaving_additional_yards_l2339_233937

theorem weaving_additional_yards {d : ℝ} :
  (∃ d : ℝ, (30 * 5 + (30 * 29) / 2 * d = 390) → d = 16 / 29) :=
sorry

end weaving_additional_yards_l2339_233937


namespace midpoint_sum_l2339_233972

theorem midpoint_sum :
  let x1 := 8
  let y1 := -4
  let z1 := 10
  let x2 := -2
  let y2 := 10
  let z2 := -6
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  let midpoint_z := (z1 + z2) / 2
  midpoint_x + midpoint_y + midpoint_z = 8 :=
by
  -- We just need to state the theorem, proof is not required
  sorry

end midpoint_sum_l2339_233972


namespace problem_l2339_233975

variable (a : ℕ → ℝ) (n m : ℕ)

-- Condition: non-negative sequence and a_{n+m} ≤ a_n + a_m
axiom condition (n m : ℕ) : a n ≥ 0 ∧ a (n + m) ≤ a n + a m

-- Theorem: for any n ≥ m
theorem problem (h : n ≥ m) : a n ≤ m * a 1 + ((n / m) - 1) * a m :=
sorry

end problem_l2339_233975


namespace plywood_cut_difference_l2339_233958

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l2339_233958


namespace arcsin_neg_sqrt3_div_2_l2339_233913

theorem arcsin_neg_sqrt3_div_2 : 
  Real.arcsin (- (Real.sqrt 3 / 2)) = - (Real.pi / 3) := 
by sorry

end arcsin_neg_sqrt3_div_2_l2339_233913


namespace silk_dyeing_total_correct_l2339_233925

open Real

theorem silk_dyeing_total_correct :
  let green := 61921
  let pink := 49500
  let blue := 75678
  let yellow := 34874.5
  let total_without_red := green + pink + blue + yellow
  let red := 0.10 * total_without_red
  let total_with_red := total_without_red + red
  total_with_red = 245270.85 :=
by
  sorry

end silk_dyeing_total_correct_l2339_233925


namespace sugar_percentage_l2339_233966

theorem sugar_percentage (S : ℝ) (P : ℝ) : 
  (3 / 4 * S * 0.10 + (1 / 4) * S * P / 100 = S * 0.20) → 
  P = 50 := 
by 
  intro h
  sorry

end sugar_percentage_l2339_233966


namespace fraction_values_l2339_233909

theorem fraction_values (a b c : ℚ) (h1 : a / b = 2) (h2 : b / c = 4 / 3) : c / a = 3 / 8 := 
by
  sorry

end fraction_values_l2339_233909


namespace B_days_solve_l2339_233954

noncomputable def combined_work_rate (A_rate B_rate C_rate : ℝ) : ℝ := A_rate + B_rate + C_rate
noncomputable def A_rate : ℝ := 1 / 6
noncomputable def C_rate : ℝ := 1 / 7.5
noncomputable def combined_rate : ℝ := 1 / 2

theorem B_days_solve : ∃ (B_days : ℝ), combined_work_rate A_rate (1 / B_days) C_rate = combined_rate ∧ B_days = 5 :=
by
  use 5
  rw [←inv_div] -- simplifying the expression of 1/B_days
  have : ℝ := sorry -- steps to cancel and simplify, proving the equality
  sorry

end B_days_solve_l2339_233954


namespace cannot_transform_with_swap_rows_and_columns_l2339_233911

def initialTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def goalTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 4, 7], ![2, 5, 8], ![3, 6, 9]]

theorem cannot_transform_with_swap_rows_and_columns :
  ¬ ∃ (is_transformed_by_swapping : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ → Prop),
    is_transformed_by_swapping initialTable goalTable :=
by sorry

end cannot_transform_with_swap_rows_and_columns_l2339_233911


namespace find_ABC_base10_l2339_233922

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10_l2339_233922


namespace crescent_moon_falcata_area_l2339_233956

/-
Prove that the area of the crescent moon falcata, which is bounded by:
1. A portion of the circle with radius 4 centered at (0,0) in the second quadrant.
2. A portion of the circle with radius 2 centered at (0,2) in the second quadrant.
3. The line segment from (0,0) to (-4,0).
is equal to 6π.
-/
theorem crescent_moon_falcata_area :
  let radius_large := 4
  let radius_small := 2
  let area_large := (1 / 2) * (π * (radius_large ^ 2))
  let area_small := (1 / 2) * (π * (radius_small ^ 2))
  (area_large - area_small) = 6 * π := by
  sorry

end crescent_moon_falcata_area_l2339_233956


namespace sum_in_range_l2339_233933

theorem sum_in_range : 
    let a := (2:ℝ) + 1/8
    let b := (3:ℝ) + 1/3
    let c := (5:ℝ) + 1/18
    10.5 < a + b + c ∧ a + b + c < 11 := 
by 
    sorry

end sum_in_range_l2339_233933


namespace remainder_of_power_sums_modulo_seven_l2339_233982

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l2339_233982


namespace certain_number_is_32_l2339_233907

theorem certain_number_is_32 (k t : ℚ) (certain_number : ℚ) 
  (h1 : t = 5/9 * (k - certain_number))
  (h2 : t = 75) (h3 : k = 167) :
  certain_number = 32 :=
sorry

end certain_number_is_32_l2339_233907


namespace length_of_fountain_built_by_20_men_in_6_days_l2339_233984

noncomputable def work (workers : ℕ) (days : ℕ) : ℕ :=
  workers * days

theorem length_of_fountain_built_by_20_men_in_6_days :
  (work 35 3) / (work 20 6) * 49 = 56 :=
by
  sorry

end length_of_fountain_built_by_20_men_in_6_days_l2339_233984


namespace largest_prime_factor_2999_l2339_233999

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  -- Note: This would require actual computation logic to find the largest prime factor.
  sorry

theorem largest_prime_factor_2999 :
  largest_prime_factor 2999 = 103 :=
by 
  -- Given conditions:
  -- 1. 2999 is an odd number (doesn't need explicit condition in proof).
  -- 2. Sum of digits is 29, thus not divisible by 3.
  -- 3. 2999 is not divisible by 11.
  -- 4. 2999 is not divisible by 7, 13, 17, 19.
  -- 5. Prime factorization of 2999 is 29 * 103.
  admit -- actual proof will need detailed prime factor test results 

end largest_prime_factor_2999_l2339_233999


namespace find_divisor_l2339_233904

theorem find_divisor : 
  ∀ (dividend quotient remainder divisor : ℕ), 
    dividend = 140 →
    quotient = 9 →
    remainder = 5 →
    dividend = (divisor * quotient) + remainder →
    divisor = 15 :=
by
  intros dividend quotient remainder divisor hd hq hr hdiv
  sorry

end find_divisor_l2339_233904


namespace fixed_point_exists_line_intersects_circle_shortest_chord_l2339_233946

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_exists : ∃ P : ℝ × ℝ, (∀ m : ℝ, line_l P.1 P.2 m) ∧ P = (3, 1) :=
by
  sorry

theorem line_intersects_circle : ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

theorem shortest_chord : ∃ m : ℝ, m = -3/4 ∧ (∀ x y, line_l x y m ↔ 2 * x - y - 5 = 0) :=
by
  sorry

end fixed_point_exists_line_intersects_circle_shortest_chord_l2339_233946


namespace value_of_square_l2339_233948

theorem value_of_square (z : ℝ) (h : 3 * z^2 + 2 * z = 5 * z + 11) : (6 * z - 5)^2 = 141 := by
  sorry

end value_of_square_l2339_233948


namespace penguins_count_l2339_233935

variable (P B : ℕ)

theorem penguins_count (h1 : B = 2 * P) (h2 : P + B = 63) : P = 21 :=
by
  sorry

end penguins_count_l2339_233935


namespace train_speed_l2339_233927

/-- 
Theorem: Given the length of the train L = 1200 meters and the time T = 30 seconds, the speed of the train S is 40 meters per second.
-/
theorem train_speed (L : ℕ) (T : ℕ) (hL : L = 1200) (hT : T = 30) : L / T = 40 := by
  sorry

end train_speed_l2339_233927


namespace beautiful_point_coordinates_l2339_233963

-- Define a "beautiful point"
def is_beautiful_point (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = P.1 * P.2

theorem beautiful_point_coordinates (M : ℝ × ℝ) : 
  is_beautiful_point M ∧ abs M.1 = 2 → 
  (M = (2, 2) ∨ M = (-2, 2/3)) :=
by sorry

end beautiful_point_coordinates_l2339_233963


namespace monotonically_increasing_interval_l2339_233971

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^3

theorem monotonically_increasing_interval : ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

end monotonically_increasing_interval_l2339_233971


namespace sum_of_coefficients_l2339_233908

theorem sum_of_coefficients (x : ℝ) : (∃ x : ℝ, 5 * x * (1 - x) = 3) → 5 + (-5) + 3 = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_of_coefficients_l2339_233908


namespace parabola_y_axis_intersection_l2339_233943

theorem parabola_y_axis_intersection:
  (∀ x y : ℝ, y = -2 * (x - 1)^2 - 3 → x = 0 → y = -5) :=
by
  intros x y h_eq h_x
  sorry

end parabola_y_axis_intersection_l2339_233943


namespace lamps_remain_lit_after_toggling_l2339_233924

theorem lamps_remain_lit_after_toggling :
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  1997 - pulled_three_times - pulled_once = 999 := by
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  have h : 1997 - pulled_three_times - (pulled_once) = 999 := sorry
  exact h

end lamps_remain_lit_after_toggling_l2339_233924


namespace vijay_work_alone_in_24_days_l2339_233988

theorem vijay_work_alone_in_24_days (ajay_rate vijay_rate combined_rate : ℝ) 
  (h1 : ajay_rate = 1 / 8) 
  (h2 : combined_rate = 1 / 6) 
  (h3 : ajay_rate + vijay_rate = combined_rate) : 
  vijay_rate = 1 / 24 := 
sorry

end vijay_work_alone_in_24_days_l2339_233988


namespace sum_and_product_of_roots_l2339_233996

theorem sum_and_product_of_roots (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b = 0 → x = -2 ∨ x = 3) → a + b = -7 :=
by
  sorry

end sum_and_product_of_roots_l2339_233996


namespace find_initial_pens_l2339_233997

-- Conditions in the form of definitions
def initial_pens (P : ℕ) : ℕ := P
def after_mike (P : ℕ) : ℕ := P + 20
def after_cindy (P : ℕ) : ℕ := 2 * after_mike P
def after_sharon (P : ℕ) : ℕ := after_cindy P - 19

-- The final condition
def final_pens (P : ℕ) : ℕ := 31

-- The goal is to prove that the initial number of pens is 5
theorem find_initial_pens : 
  ∃ (P : ℕ), after_sharon P = final_pens P → P = 5 :=
by 
  sorry

end find_initial_pens_l2339_233997


namespace solve_for_x_l2339_233903

theorem solve_for_x :
  ∃ x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 = (27 * x) ^ 9 + 81 * x ∧ x = 1 / 3 :=
by
  sorry

end solve_for_x_l2339_233903


namespace min_value_M_l2339_233974

theorem min_value_M 
  (S_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h1 : ∀ n, S_n n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 8)
  (h3 : a 3 + a 5 = 26)
  (h4 : ∀ n, T_n n = S_n n / n^2) :
  ∃ M : ℝ, M = 2 ∧ (∀ n > 0, T_n n ≤ M) :=
by sorry

end min_value_M_l2339_233974


namespace mother_l2339_233929

theorem mother's_age (D M : ℕ) (h1 : 2 * D + M = 70) (h2 : D + 2 * M = 95) : M = 40 :=
sorry

end mother_l2339_233929


namespace siblings_ate_two_slices_l2339_233901

-- Let slices_after_dinner be the number of slices left after eating one-fourth of 16 slices
def slices_after_dinner : ℕ := 16 - 16 / 4

-- Let slices_after_yves be the number of slices left after Yves ate one-fourth of the remaining pizza
def slices_after_yves : ℕ := slices_after_dinner - slices_after_dinner / 4

-- Let slices_left be the number of slices left after Yves's siblings ate some slices
def slices_left : ℕ := 5

-- Let slices_eaten_by_siblings be the number of slices eaten by Yves's siblings
def slices_eaten_by_siblings : ℕ := slices_after_yves - slices_left

-- Since there are two siblings, each ate half of the slices_eaten_by_siblings
def slices_per_sibling : ℕ := slices_eaten_by_siblings / 2

-- The theorem stating that each sibling ate 2 slices
theorem siblings_ate_two_slices : slices_per_sibling = 2 :=
by
  -- Definition of slices_after_dinner
  have h1 : slices_after_dinner = 12 := by sorry
  -- Definition of slices_after_yves
  have h2 : slices_after_yves = 9 := by sorry
  -- Definition of slices_eaten_by_siblings
  have h3 : slices_eaten_by_siblings = 4 := by sorry
  -- Final assertion of slices_per_sibling
  have h4 : slices_per_sibling = 2 := by sorry
  exact h4

end siblings_ate_two_slices_l2339_233901


namespace cost_price_of_article_l2339_233926

theorem cost_price_of_article 
  (CP SP : ℝ)
  (H1 : SP = 1.13 * CP)
  (H2 : 1.10 * SP = 616) :
  CP = 495.58 :=
by
  sorry

end cost_price_of_article_l2339_233926


namespace sum_possible_values_of_y_l2339_233931

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l2339_233931


namespace fraction_equality_l2339_233978

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : b / (a - b) = 3 :=
sorry

end fraction_equality_l2339_233978


namespace max_sum_of_prices_l2339_233953

theorem max_sum_of_prices (R P : ℝ) 
  (h1 : 4 * R + 5 * P ≥ 27) 
  (h2 : 6 * R + 3 * P ≤ 27) : 
  3 * R + 4 * P ≤ 36 :=
by 
  sorry

end max_sum_of_prices_l2339_233953


namespace range_of_m_for_quadratic_sol_in_interval_l2339_233919

theorem range_of_m_for_quadratic_sol_in_interval :
  {m : ℝ // ∀ x, (x^2 + (m-1)*x + 1 = 0) → (0 ≤ x ∧ x ≤ 2)} = {m : ℝ // m < -1} :=
by
  sorry

end range_of_m_for_quadratic_sol_in_interval_l2339_233919


namespace trigonometric_identities_l2339_233942

theorem trigonometric_identities (α : Real) (h1 : 3 * π / 2 < α ∧ α < 2 * π) (h2 : Real.sin α = -3 / 5) :
  Real.tan α = 3 / 4 ∧ Real.tan (α - π / 4) = -1 / 7 ∧ Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end trigonometric_identities_l2339_233942


namespace work_efficiency_ratio_l2339_233941

variables (A_eff B_eff : ℚ) (a b : Type)

theorem work_efficiency_ratio (h1 : B_eff = 1 / 33)
  (h2 : A_eff + B_eff = 1 / 11) :
  A_eff / B_eff = 2 :=
by 
  sorry

end work_efficiency_ratio_l2339_233941


namespace max_OM_ON_value_l2339_233910

noncomputable def maximum_OM_ON (a b : ℝ) : ℝ :=
  (1 + Real.sqrt 2) / 2 * (a + b)

-- Given the conditions in triangle ABC with sides BC and AC having fixed lengths a and b respectively,
-- and that AB can vary such that a square is constructed outward on side AB with center O,
-- and M and N are the midpoints of sides BC and AC respectively, prove the maximum value of OM + ON.
theorem max_OM_ON_value (a b : ℝ) : 
  ∃ OM ON : ℝ, OM + ON = maximum_OM_ON a b :=
sorry

end max_OM_ON_value_l2339_233910


namespace no_real_roots_l2339_233990

def op (m n : ℝ) : ℝ := n^2 - m * n + 1

theorem no_real_roots (x : ℝ) : op 1 x = 0 → ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by {
  sorry
}

end no_real_roots_l2339_233990


namespace arithmetic_sequence_geometric_condition_l2339_233981

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 3)
  (h3 : ∃ k, a (k+3) * a k = (a (k+1)) * (a (k+2))) :
  a 2 = -9 :=
by
  sorry

end arithmetic_sequence_geometric_condition_l2339_233981


namespace minimum_treasure_buried_l2339_233950

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l2339_233950


namespace jim_total_cars_l2339_233973

theorem jim_total_cars (B F C : ℕ) (h1 : B = 4 * F) (h2 : F = 2 * C + 3) (h3 : B = 220) :
  B + F + C = 301 :=
by
  sorry

end jim_total_cars_l2339_233973


namespace piggy_bank_donation_l2339_233945

theorem piggy_bank_donation (total_earnings : ℕ) (cost_of_ingredients : ℕ) 
  (total_donation_homeless_shelter : ℕ) : 
  (total_earnings = 400) → (cost_of_ingredients = 100) → (total_donation_homeless_shelter = 160) → 
  (total_donation_homeless_shelter - (total_earnings - cost_of_ingredients) / 2 = 10) :=
by
  intros h1 h2 h3
  sorry

end piggy_bank_donation_l2339_233945


namespace current_length_of_highway_l2339_233959

def total_length : ℕ := 650
def miles_first_day : ℕ := 50
def miles_second_day : ℕ := 3 * miles_first_day
def miles_still_needed : ℕ := 250
def miles_built : ℕ := miles_first_day + miles_second_day

theorem current_length_of_highway :
  total_length - miles_still_needed = 400 :=
by
  sorry

end current_length_of_highway_l2339_233959


namespace max_AC_not_RS_l2339_233995

theorem max_AC_not_RS (TotalCars NoACCars MinRS MaxACnotRS : ℕ)
  (h1 : TotalCars = 100)
  (h2 : NoACCars = 49)
  (h3 : MinRS >= 51)
  (h4 : (TotalCars - NoACCars) - MinRS = MaxACnotRS)
  : MaxACnotRS = 0 :=
by
  sorry

end max_AC_not_RS_l2339_233995


namespace average_price_per_book_l2339_233902

def books_from_shop1 := 42
def price_from_shop1 := 520
def books_from_shop2 := 22
def price_from_shop2 := 248

def total_books := books_from_shop1 + books_from_shop2
def total_price := price_from_shop1 + price_from_shop2
def average_price := total_price / total_books

theorem average_price_per_book : average_price = 12 := by
  sorry

end average_price_per_book_l2339_233902


namespace range_of_a_l2339_233936

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l2339_233936


namespace part1_part2_l2339_233992

namespace VectorProblem

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def m := 5 / 9
def n := 8 / 9

def k := -16 / 13

-- Statement 1: Prove vectors satisfy the linear combination
theorem part1 : vector_a = (m * vector_b.1 + n * vector_c.1, m * vector_b.2 + n * vector_c.2) :=
by {
  sorry
}

-- Statement 2: Prove vectors are parallel
theorem part2 : (3 + 4 * k) * 2 + (2 + k) * 5 = 0 :=
by {
  sorry
}

end VectorProblem

end part1_part2_l2339_233992


namespace solution_of_system_l2339_233985

noncomputable def system_of_equations (x y : ℝ) :=
  x = 1.12 * y + 52.8 ∧ x = y + 50

theorem solution_of_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ y = -23.33 ∧ x = 26.67 :=
by
  sorry

end solution_of_system_l2339_233985


namespace negate_at_most_two_l2339_233947

def atMost (n : Nat) : Prop := ∃ k : Nat, k ≤ n
def atLeast (n : Nat) : Prop := ∃ k : Nat, k ≥ n

theorem negate_at_most_two : ¬ atMost 2 ↔ atLeast 3 := by
  sorry

end negate_at_most_two_l2339_233947


namespace sum_first_53_odd_numbers_l2339_233934

-- Definitions based on the given conditions
def first_odd_number := 1

def nth_odd_number (n : ℕ) : ℕ :=
  1 + (n - 1) * 2

def sum_n_odd_numbers (n : ℕ) : ℕ :=
  (n * n)

-- Theorem statement to prove
theorem sum_first_53_odd_numbers : sum_n_odd_numbers 53 = 2809 := 
by
  sorry

end sum_first_53_odd_numbers_l2339_233934


namespace cricketer_boundaries_l2339_233912

theorem cricketer_boundaries (total_runs : ℕ) (sixes : ℕ) (percent_runs_by_running : ℝ)
  (h1 : total_runs = 152)
  (h2 : sixes = 2)
  (h3 : percent_runs_by_running = 60.526315789473685) :
  let runs_by_running := round (total_runs * percent_runs_by_running / 100)
  let runs_from_sixes := sixes * 6
  let runs_from_boundaries := total_runs - runs_by_running - runs_from_sixes
  let boundaries := runs_from_boundaries / 4
  boundaries = 12 :=
by
  sorry

end cricketer_boundaries_l2339_233912


namespace initial_number_of_orchids_l2339_233998

theorem initial_number_of_orchids 
  (initial_orchids : ℕ)
  (cut_orchids : ℕ)
  (final_orchids : ℕ)
  (h_cut : cut_orchids = 19)
  (h_final : final_orchids = 21) :
  initial_orchids + cut_orchids = final_orchids → initial_orchids = 2 :=
by
  sorry

end initial_number_of_orchids_l2339_233998


namespace cost_of_fencing_per_meter_l2339_233977

def rectangular_farm_area : Real := 1200
def short_side_length : Real := 30
def total_cost : Real := 1440

theorem cost_of_fencing_per_meter : (total_cost / (short_side_length + (rectangular_farm_area / short_side_length) + Real.sqrt ((rectangular_farm_area / short_side_length)^2 + short_side_length^2))) = 12 :=
by
  sorry

end cost_of_fencing_per_meter_l2339_233977


namespace tangent_line_curve_l2339_233930

theorem tangent_line_curve (a b : ℚ) 
  (h1 : 3 * a + b = 1) 
  (h2 : a + b = 2) : 
  b - a = 3 := 
by 
  sorry

end tangent_line_curve_l2339_233930


namespace batsman_average_after_20th_innings_l2339_233949

theorem batsman_average_after_20th_innings 
    (score_20th_innings : ℕ)
    (previous_avg_increase : ℕ)
    (total_innings : ℕ)
    (never_not_out : Prop)
    (previous_avg : ℕ)
    : score_20th_innings = 90 →
      previous_avg_increase = 2 →
      total_innings = 20 →
      previous_avg = (19 * previous_avg + score_20th_innings) / total_innings →
      ((19 * previous_avg + score_20th_innings) / total_innings) + previous_avg_increase = 52 :=
by 
  sorry

end batsman_average_after_20th_innings_l2339_233949


namespace max_square_test_plots_l2339_233987

theorem max_square_test_plots 
  (length : ℕ) (width : ℕ) (fence_available : ℕ) 
  (side_length : ℕ) (num_plots : ℕ) 
  (h_length : length = 30)
  (h_width : width = 60)
  (h_fencing : fence_available = 2500)
  (h_side_length : side_length = 10)
  (h_num_plots : num_plots = 18) :
  (length * width / side_length^2 = num_plots) ∧
  (30 * (60 / side_length - 1) + 60 * (30 / side_length - 1) ≤ fence_available) := 
sorry

end max_square_test_plots_l2339_233987


namespace percentage_saved_l2339_233917

theorem percentage_saved (saved spent : ℝ) (h_saved : saved = 3) (h_spent : spent = 27) : 
  (saved / (saved + spent)) * 100 = 10 := by
  sorry

end percentage_saved_l2339_233917


namespace crayons_taken_out_l2339_233923

-- Define the initial and remaining number of crayons
def initial_crayons : ℕ := 7
def remaining_crayons : ℕ := 4

-- Define the proposition to prove
theorem crayons_taken_out : initial_crayons - remaining_crayons = 3 := by
  sorry

end crayons_taken_out_l2339_233923


namespace math_problem_l2339_233989

-- Define the conditions
def a := -6
def b := 2
def c := 1 / 3
def d := 3 / 4
def e := 12
def f := -3

-- Statement of the problem
theorem math_problem : a / b + (c - d) * e + f^2 = 1 :=
by
  sorry

end math_problem_l2339_233989


namespace no_triangle_tangent_l2339_233905

open Real

/-- Given conditions --/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0 ∧ (1 / a^2) + (1 / b^2) = 1

theorem no_triangle_tangent (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + 1 / b^2 = 1) :
  ¬∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2) ∧ (C1 B.1 B.2) ∧ (C1 C.1 C.2) ∧
    (∃ (l : ℝ) (m : ℝ) (n : ℝ), C2 l m a b ∧ C2 n l a b) :=
by
  sorry

end no_triangle_tangent_l2339_233905


namespace trees_died_in_typhoon_l2339_233962

theorem trees_died_in_typhoon :
  ∀ (original_trees left_trees died_trees : ℕ), 
  original_trees = 20 → 
  left_trees = 4 → 
  died_trees = original_trees - left_trees → 
  died_trees = 16 :=
by
  intros original_trees left_trees died_trees h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end trees_died_in_typhoon_l2339_233962


namespace smarties_modulo_l2339_233906

theorem smarties_modulo (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end smarties_modulo_l2339_233906


namespace wine_ages_l2339_233967

-- Define the ages of the wines as variables
variable (C F T B Bo M : ℝ)

-- Define the six conditions
axiom h1 : F = 3 * C
axiom h2 : C = 4 * T
axiom h3 : B = (1 / 2) * T
axiom h4 : Bo = 2 * F
axiom h5 : M^2 = Bo
axiom h6 : C = 40

-- Prove the ages of the wines 
theorem wine_ages : 
  F = 120 ∧ 
  T = 10 ∧ 
  B = 5 ∧ 
  Bo = 240 ∧ 
  M = Real.sqrt 240 :=
by
  sorry

end wine_ages_l2339_233967


namespace integral_equals_result_l2339_233944

noncomputable def integral_value : ℝ :=
  ∫ x in 1.0..2.0, (x^2 + 1) / x

theorem integral_equals_result :
  integral_value = (3 / 2) + Real.log 2 := 
by
  sorry

end integral_equals_result_l2339_233944
