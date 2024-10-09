import Mathlib

namespace go_piece_arrangement_l1597_159740

theorem go_piece_arrangement (w b : ℕ) (pieces : List ℕ) 
    (h_w : w = 180) (h_b : b = 181)
    (h_pieces : pieces.length = w + b) 
    (h_black_count : pieces.count 1 = b) 
    (h_white_count : pieces.count 0 = w) :
    ∃ (i j : ℕ), i < j ∧ j < pieces.length ∧ 
    ((j - i - 1 = 178) ∨ (j - i - 1 = 181)) ∧ 
    (pieces.get ⟨i, sorry⟩ = 1) ∧ 
    (pieces.get ⟨j, sorry⟩ = 1) := 
sorry

end go_piece_arrangement_l1597_159740


namespace smaller_circle_radius_l1597_159744

open Real

def is_geometric_progression (a b c : ℝ) : Prop :=
  (b / a = c / b)

theorem smaller_circle_radius 
  (B1 B2 : ℝ) 
  (r2 : ℝ) 
  (h1 : B1 + B2 = π * r2^2) 
  (h2 : r2 = 5) 
  (h3 : is_geometric_progression B1 B2 (B1 + B2)) :
  sqrt ((-1 + sqrt (1 + 100 * π)) / (2 * π)) = sqrt (B1 / π) :=
by
  sorry

end smaller_circle_radius_l1597_159744


namespace required_remaining_speed_l1597_159773

-- Definitions for the given problem
variables (D T : ℝ) 

-- Given conditions from the problem
def speed_first_part (D T : ℝ) : Prop := 
  40 = (2 * D / 3) / (T / 3)

def remaining_distance_time (D T : ℝ) : Prop :=
  10 = (D / 3) / (2 * (2 * D / 3) / 40 / 3)

-- Theorem to be proved
theorem required_remaining_speed (D T : ℝ) 
  (h1 : speed_first_part D T)
  (h2 : remaining_distance_time D T) :
  10 = (D / 3) / (2 * (T / 3)) :=
  sorry  -- Proof is skipped

end required_remaining_speed_l1597_159773


namespace goods_train_speed_l1597_159732

noncomputable def speed_of_goods_train (train_speed : ℝ) (goods_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed_mps := goods_length / passing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  (relative_speed_kmph - train_speed)

theorem goods_train_speed :
  speed_of_goods_train 30 280 9 = 82 :=
by
  sorry

end goods_train_speed_l1597_159732


namespace ellipse_equation_l1597_159775

theorem ellipse_equation
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (eccentricity : ℝ)
  (eccentricity_eq : eccentricity = (Real.sqrt 3 / 3))
  (perimeter_triangle : ℝ)
  (perimeter_eq : perimeter_triangle = 4 * Real.sqrt 3) :
  a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ (a > b) ∧ (eccentricity = 1 / Real.sqrt 3) →
  (∀ x y : ℝ, (x^2 / 3 + y^2 / 2 = 1)) :=
by
  sorry

end ellipse_equation_l1597_159775


namespace find_unit_prices_minimize_total_cost_l1597_159758

def unit_prices_ (x y : ℕ) :=
  x + 2 * y = 40 ∧ 2 * x + 3 * y = 70
  
theorem find_unit_prices (x y: ℕ) (h: unit_prices_ x y): x = 20 ∧ y = 10 := 
  sorry

def total_cost (m: ℕ) := 20 * m + 10 * (60 - m)

theorem minimize_total_cost (m : ℕ) (h1 : 60 ≥ m) (h2 : m ≥ 20) : 
  total_cost m = 800 → m = 20 :=
  sorry

end find_unit_prices_minimize_total_cost_l1597_159758


namespace integer_roots_iff_floor_square_l1597_159716

variable (α β : ℝ)
variable (m n : ℕ)
variable (real_roots : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0)

noncomputable def are_integers (α β : ℝ) : Prop := (∃ (a b : ℤ), α = a ∧ β = b)

theorem integer_roots_iff_floor_square (m n : ℕ) (α β : ℝ)
  (hmn : 0 ≤ m ∧ 0 ≤ n)
  (roots_real : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0) :
  (are_integers α β) ↔ (∃ k : ℤ, (⌊m * α⌋ + ⌊m * β⌋) = k^2) :=
sorry

end integer_roots_iff_floor_square_l1597_159716


namespace jorge_goals_l1597_159752

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l1597_159752


namespace x_squared_plus_y_squared_l1597_159765

theorem x_squared_plus_y_squared (x y : ℝ) (h₀ : x + y = 10) (h₁ : x * y = 15) : x^2 + y^2 = 70 :=
by
  sorry

end x_squared_plus_y_squared_l1597_159765


namespace cricket_team_matches_l1597_159781

theorem cricket_team_matches 
  (M : ℕ) (W : ℕ) 
  (h1 : W = 20 * M / 100) 
  (h2 : (W + 80) * 100 = 52 * M) : 
  M = 250 :=
by
  sorry

end cricket_team_matches_l1597_159781


namespace necessary_but_not_sufficient_for_odd_function_l1597_159708

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f (x)

theorem necessary_but_not_sufficient_for_odd_function (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f :=
sorry

end necessary_but_not_sufficient_for_odd_function_l1597_159708


namespace find_third_side_length_l1597_159704

noncomputable def triangle_third_side_length (a b c : ℝ) (B C : ℝ) 
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) : Prop :=
a = 16

theorem find_third_side_length (a b c : ℝ) (B C : ℝ)
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) :
  triangle_third_side_length a b c B C h1 h2 h3 :=
sorry

end find_third_side_length_l1597_159704


namespace find_P_coordinates_l1597_159799

-- Define points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the theorem
theorem find_P_coordinates :
  ∃ P : ℝ × ℝ, P = (8, -15) ∧ (P.1 - A.1, P.2 - A.2) = (3 * (B.1 - A.1), 3 * (B.2 - A.2)) :=
sorry

end find_P_coordinates_l1597_159799


namespace find_arithmetic_mean_l1597_159745

theorem find_arithmetic_mean (σ μ : ℝ) (hσ : σ = 1.5) (h : 11 = μ - 2 * σ) : μ = 14 :=
by
  sorry

end find_arithmetic_mean_l1597_159745


namespace factor_expression_l1597_159706

variables (b : ℝ)

theorem factor_expression :
  (8 * b ^ 3 + 45 * b ^ 2 - 10) - (-12 * b ^ 3 + 5 * b ^ 2 - 10) = 20 * b ^ 2 * (b + 2) :=
by
  sorry

end factor_expression_l1597_159706


namespace find_inverse_l1597_159751

noncomputable def f (x : ℝ) := (x^7 - 1) / 5

theorem find_inverse :
  (f⁻¹ (-1 / 80) = (15 / 16)^(1 / 7)) :=
sorry

end find_inverse_l1597_159751


namespace homework_checked_on_friday_l1597_159797

theorem homework_checked_on_friday
  (prob_no_check : ℚ := 1/2)
  (prob_check_on_friday_given_check : ℚ := 1/5)
  (prob_a : ℚ := 3/5)
  : 1/3 = prob_check_on_friday_given_check / prob_a :=
by
  sorry

end homework_checked_on_friday_l1597_159797


namespace difference_of_squares_expression_l1597_159778

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l1597_159778


namespace intersection_is_2_to_inf_l1597_159788

-- Define the set A
def setA (x : ℝ) : Prop :=
 x > 1

-- Define the set B
def setB (y : ℝ) : Prop :=
 ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 5)

-- Define the intersection of A and B
def setIntersection : Set ℝ :=
{ y | setA y ∧ setB y }

-- Statement to prove the intersection
theorem intersection_is_2_to_inf : setIntersection = { y | y ≥ 2 } :=
sorry -- Proof is omitted

end intersection_is_2_to_inf_l1597_159788


namespace remainder_of_exponentiated_sum_modulo_seven_l1597_159776

theorem remainder_of_exponentiated_sum_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_of_exponentiated_sum_modulo_seven_l1597_159776


namespace circle_parabola_intersect_l1597_159720

theorem circle_parabola_intersect (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 ∧ y = a * x^2 → (x ≠ 0 ∨ y ≠ 0)) ↔ a > 1 / 2 :=
by
  sorry

end circle_parabola_intersect_l1597_159720


namespace molecular_weight_correct_l1597_159743

noncomputable def molecular_weight : ℝ := 
  let N_count := 2
  let H_count := 6
  let Br_count := 1
  let O_count := 1
  let C_count := 3
  let N_weight := 14.01
  let H_weight := 1.01
  let Br_weight := 79.90
  let O_weight := 16.00
  let C_weight := 12.01
  N_count * N_weight + 
  H_count * H_weight + 
  Br_count * Br_weight + 
  O_count * O_weight +
  C_count * C_weight

theorem molecular_weight_correct :
  molecular_weight = 166.01 := 
by
  sorry

end molecular_weight_correct_l1597_159743


namespace part1_part2_l1597_159783

section

variables {x m : ℝ}

def f (x m : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (x m : ℝ) : ℝ := 2 * x^2 - x - m

theorem part1 (m : ℝ) (h : m = 1) : 
  {x : ℝ | f x m > 0} = {x : ℝ | x < -2 ∨ x > 1} :=
sorry

theorem part2 (m : ℝ) (h : m > 0) : 
  {x : ℝ | f x m ≤ g x m} = {x : ℝ | -5 ≤ x ∧ x ≤ m} :=
sorry
     
end

end part1_part2_l1597_159783


namespace fraction_nonnegative_iff_l1597_159709

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 :=
by
  -- Proof goes here
  sorry

end fraction_nonnegative_iff_l1597_159709


namespace largest_not_sum_of_two_composites_l1597_159715

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l1597_159715


namespace fibonacci_coprime_l1597_159728

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_coprime (n : ℕ) (hn : n ≥ 1) :
  Nat.gcd (fibonacci n) (fibonacci (n - 1)) = 1 := by
  sorry

end fibonacci_coprime_l1597_159728


namespace orange_jellybeans_count_l1597_159702

theorem orange_jellybeans_count (total blue purple red : Nat)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_red : red = 120) :
  ∃ orange : Nat, orange = total - (blue + purple + red) ∧ orange = 40 :=
by
  sorry

end orange_jellybeans_count_l1597_159702


namespace find_sum_a100_b100_l1597_159719

-- Definitions of arithmetic sequences and their properties
structure arithmetic_sequence (an : ℕ → ℝ) :=
  (a1 : ℝ)
  (d : ℝ)
  (def_seq : ∀ n, an n = a1 + (n - 1) * d)

-- Given conditions
variables (a_n b_n : ℕ → ℝ)
variables (ha : arithmetic_sequence a_n)
variables (hb : arithmetic_sequence b_n)

-- Specified conditions
axiom cond1 : a_n 5 + b_n 5 = 3
axiom cond2 : a_n 9 + b_n 9 = 19

-- The goal to be proved
theorem find_sum_a100_b100 : a_n 100 + b_n 100 = 383 :=
sorry

end find_sum_a100_b100_l1597_159719


namespace find_m_l1597_159718

theorem find_m (m : ℝ) :
  (∃ x : ℝ, x^2 - m * x + m^2 - 19 = 0 ∧ (x = 2 ∨ x = 3))
  ∧ (∀ x : ℝ, x^2 - m * x + m^2 - 19 = 0 → x ≠ 2 ∧ x ≠ -4) 
  → m = -2 :=
by
  sorry

end find_m_l1597_159718


namespace opposite_of_neg_one_div_2023_l1597_159747

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l1597_159747


namespace angle_E_measure_l1597_159707

theorem angle_E_measure {D E F : Type} (angle_D angle_E angle_F : ℝ) 
  (h1 : angle_E = angle_F)
  (h2 : angle_F = 3 * angle_D)
  (h3 : angle_D = (1/2) * angle_E) 
  (h_sum : angle_D + angle_E + angle_F = 180) :
  angle_E = 540 / 7 := 
by
  sorry

end angle_E_measure_l1597_159707


namespace pizza_non_crust_percentage_l1597_159784

theorem pizza_non_crust_percentage (total_weight crust_weight : ℕ) (h₁ : total_weight = 200) (h₂ : crust_weight = 50) :
  (total_weight - crust_weight) * 100 / total_weight = 75 :=
by
  sorry

end pizza_non_crust_percentage_l1597_159784


namespace problem_statement_l1597_159786

theorem problem_statement : 25 * 15 * 9 * 5.4 * 3.24 = 3 ^ 10 := 
by 
  sorry

end problem_statement_l1597_159786


namespace range_of_B_l1597_159785

theorem range_of_B (A : ℝ × ℝ) (hA : A = (1, 2)) (h : 2 * A.1 - B * A.2 + 3 ≥ 0) : B ≤ 2.5 :=
by sorry

end range_of_B_l1597_159785


namespace walnut_trees_planted_l1597_159759

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l1597_159759


namespace odd_nat_existence_l1597_159746

theorem odd_nat_existence (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (n : ℕ) :
  ∃ m : ℕ, (a^m * b^2 - 1) % 2^n = 0 ∨ (b^m * a^2 - 1) % 2^n = 0 := 
by
  sorry

end odd_nat_existence_l1597_159746


namespace birds_meeting_distance_l1597_159769

theorem birds_meeting_distance 
  (D : ℝ) (S1 : ℝ) (S2 : ℝ) (t : ℝ)
  (H1 : D = 45)
  (H2 : S1 = 6)
  (H3 : S2 = 2.5)
  (H4 : t = D / (S1 + S2)) :
  S1 * t = 31.76 :=
by
  sorry

end birds_meeting_distance_l1597_159769


namespace exists_positive_integer_divisible_by_15_and_sqrt_in_range_l1597_159730

theorem exists_positive_integer_divisible_by_15_and_sqrt_in_range :
  ∃ (n : ℕ), (n % 15 = 0) ∧ (28 < Real.sqrt n) ∧ (Real.sqrt n < 28.5) ∧ (n = 795) :=
by
  sorry

end exists_positive_integer_divisible_by_15_and_sqrt_in_range_l1597_159730


namespace required_oranges_for_juice_l1597_159717

theorem required_oranges_for_juice (oranges quarts : ℚ) (h : oranges = 36 ∧ quarts = 48) :
  ∃ x, ((oranges / quarts) = (x / 6) ∧ x = 4.5) := 
by sorry

end required_oranges_for_juice_l1597_159717


namespace closest_ratio_to_one_l1597_159771

theorem closest_ratio_to_one (a c : ℕ) (h1 : 2 * a + c = 130) (h2 : a ≥ 1) (h3 : c ≥ 1) : 
  a = 43 ∧ c = 44 :=
by {
    sorry 
}

end closest_ratio_to_one_l1597_159771


namespace find_N_l1597_159749

noncomputable def sum_of_sequence : ℤ :=
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999

theorem find_N : ∃ (N : ℤ), 8000 - N = sum_of_sequence ∧ N = 64 := by
  use 64
  -- The actual proof steps will go here
  sorry

end find_N_l1597_159749


namespace length_of_segment_correct_l1597_159735

noncomputable def length_of_segment (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem length_of_segment_correct :
  length_of_segment 5 (-1) 13 11 = 4 * Real.sqrt 13 := by
  sorry

end length_of_segment_correct_l1597_159735


namespace count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l1597_159738

noncomputable def count_equally_spaced_integers : ℕ := 
  sorry

theorem count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle:
  count_equally_spaced_integers = 4 :=
sorry

end count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l1597_159738


namespace find_length_AD_l1597_159787

noncomputable def length_AD (AB AC BC : ℝ) (is_equal_AB_AC : AB = AC) (BD DC : ℝ) (D_midpoint : BD = DC) : ℝ :=
  let BE := BC / 2
  let AE := Real.sqrt (AB ^ 2 - BE ^ 2)
  AE

theorem find_length_AD (AB AC BC BD DC : ℝ) (is_equal_AB_AC : AB = AC) (D_midpoint : BD = DC) (H1 : AB = 26) (H2 : AC = 26) (H3 : BC = 24) (H4 : BD = 12) (H5 : DC = 12) :
  length_AD AB AC BC is_equal_AB_AC BD DC D_midpoint = 2 * Real.sqrt 133 :=
by
  -- the steps of the proof would go here
  sorry

end find_length_AD_l1597_159787


namespace find_x_plus_y_l1597_159734

theorem find_x_plus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : |x - y| = y - x) :
  (x + y = -1) ∨ (x + y = -5) :=
sorry

end find_x_plus_y_l1597_159734


namespace michael_large_balls_l1597_159741

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l1597_159741


namespace fraction_by_foot_l1597_159791

theorem fraction_by_foot (D distance_by_bus distance_by_car distance_by_foot : ℕ) (h1 : D = 24) 
  (h2 : distance_by_bus = D / 4) (h3 : distance_by_car = 6) 
  (h4 : distance_by_foot = D - (distance_by_bus + distance_by_car)) : 
  (distance_by_foot : ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_by_foot_l1597_159791


namespace five_aliens_have_more_limbs_than_five_martians_l1597_159767

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end five_aliens_have_more_limbs_than_five_martians_l1597_159767


namespace range_of_m_and_n_l1597_159721

theorem range_of_m_and_n (m n : ℝ) : 
  (2 * 2 - 3 + m > 0) → ¬ (2 + 3 - n ≤ 0) → (m > -1 ∧ n < 5) := by
  intros hA hB
  sorry

end range_of_m_and_n_l1597_159721


namespace cubicsum_l1597_159790

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end cubicsum_l1597_159790


namespace total_walnut_trees_in_park_l1597_159711

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end total_walnut_trees_in_park_l1597_159711


namespace number_added_l1597_159761

theorem number_added (x y : ℝ) (h1 : x = 33) (h2 : x / 4 + y = 15) : y = 6.75 :=
by sorry

end number_added_l1597_159761


namespace second_term_deposit_interest_rate_l1597_159705

theorem second_term_deposit_interest_rate
  (initial_deposit : ℝ)
  (first_term_annual_rate : ℝ)
  (first_term_months : ℝ)
  (second_term_initial_value : ℝ)
  (second_term_final_value : ℝ)
  (s : ℝ)
  (first_term_value : initial_deposit * (1 + first_term_annual_rate / 100 / 12 * first_term_months) = second_term_initial_value)
  (second_term_value : second_term_initial_value * (1 + s / 100 / 12 * first_term_months) = second_term_final_value) :
  s = 11.36 :=
by
  sorry

end second_term_deposit_interest_rate_l1597_159705


namespace no_solution_for_m_l1597_159766

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l1597_159766


namespace probability_no_rain_five_days_l1597_159754

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l1597_159754


namespace simplify_polynomial_l1597_159700

theorem simplify_polynomial (x : ℝ) : 
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) = 
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 :=
by sorry

end simplify_polynomial_l1597_159700


namespace train_average_speed_with_stoppages_l1597_159725

theorem train_average_speed_with_stoppages :
  (∀ d t_without_stops t_with_stops : ℝ, t_without_stops = d / 400 → 
  t_with_stops = d / (t_without_stops * (10/9)) → 
  t_with_stops = d / 360) :=
sorry

end train_average_speed_with_stoppages_l1597_159725


namespace problem_1_problem_2_problem_3_l1597_159774

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (2^x + 1)
noncomputable def f_inv (x : ℝ) : ℝ := Real.logb 2 (2^x - 1)

theorem problem_1 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x = m + f x) ↔ 
  m ∈ (Set.Icc (Real.logb 2 (1/3)) (Real.logb 2 (3/5))) :=
sorry

theorem problem_2 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (3/5))) :=
sorry

theorem problem_3 : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (1/3))) :=
sorry

end problem_1_problem_2_problem_3_l1597_159774


namespace sum_of_coefficients_l1597_159763

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 : ℤ) (a_eq : (1 - 2 * (0:ℤ)) ^ 7 = a)
  (hx_eq : ∀ (x : ℤ), (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = -2 :=
by
  sorry

end sum_of_coefficients_l1597_159763


namespace larger_number_is_correct_l1597_159794

theorem larger_number_is_correct : ∃ L : ℝ, ∃ S : ℝ, S = 48 ∧ (L - S = (1 : ℝ) / (3 : ℝ) * L) ∧ L = 72 :=
by
  sorry

end larger_number_is_correct_l1597_159794


namespace perpendicular_vectors_l1597_159757

theorem perpendicular_vectors (x : ℝ) : (2 * x + 3 = 0) → (x = -3 / 2) :=
by
  intro h
  sorry

end perpendicular_vectors_l1597_159757


namespace max_real_solutions_l1597_159710

noncomputable def max_number_of_real_solutions (n : ℕ) (y : ℝ) : ℕ :=
if (n + 1) % 2 = 1 then 1 else 0

theorem max_real_solutions (n : ℕ) (hn : 0 < n) (y : ℝ) :
  max_number_of_real_solutions n y = 1 :=
by
  sorry

end max_real_solutions_l1597_159710


namespace solve_quadratic_eq_l1597_159782

theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - (x^2 - 2 * x + 1) = 0 ↔ x = 1 / 3 ∨ x = -1 := by
  sorry

end solve_quadratic_eq_l1597_159782


namespace value_of_a_l1597_159764

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end value_of_a_l1597_159764


namespace average_children_in_families_with_children_l1597_159742

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l1597_159742


namespace nature_of_roots_Q_l1597_159713

noncomputable def Q (x : ℝ) : ℝ := x^6 - 4 * x^5 + 3 * x^4 - 7 * x^3 - x^2 + x + 10

theorem nature_of_roots_Q : 
  ∃ (negative_roots positive_roots : Finset ℝ),
    (∀ r ∈ negative_roots, r < 0) ∧
    (∀ r ∈ positive_roots, r > 0) ∧
    negative_roots.card = 1 ∧
    positive_roots.card > 1 ∧
    ∀ r, r ∈ negative_roots ∨ r ∈ positive_roots → Q r = 0 :=
sorry

end nature_of_roots_Q_l1597_159713


namespace min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l1597_159760

-- Condition Definitions
def blue := 5
def red := 9
def green := 6
def yellow := 4

-- Theorem Statements
theorem min_pencils_for_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ min_pencils : ℕ, min_pencils = 21 := by
  sorry

theorem max_pencils_remaining_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_pencils : ℕ, max_pencils = 3 := by
  sorry

theorem max_red_pencils_to_ensure_five_remaining :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_red_pencils : ℕ, max_red_pencils = 4 := by
  sorry

end min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l1597_159760


namespace problem1_problem2_l1597_159762

-- Problem 1: Prove that the minimum value of f(x) is at least m for all x ∈ ℝ when k = 0
theorem problem1 (f : ℝ → ℝ) (m : ℝ) (h : ∀ x : ℝ, f x = Real.exp x - x) : m ≤ 1 := 
sorry

-- Problem 2: Prove that there exists exactly one zero of f(x) in the interval (k, 2k) when k > 1
theorem problem2 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) (h : ∀ x : ℝ, f x = Real.exp (x - k) - x) :
  ∃! (x : ℝ), x ∈ Set.Ioo k (2 * k) ∧ f x = 0 := 
sorry

end problem1_problem2_l1597_159762


namespace equilateral_triangle_of_ap_angles_gp_sides_l1597_159727

theorem equilateral_triangle_of_ap_angles_gp_sides
  (A B C : ℝ)
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (h_ap_angles : 2 * β = α + γ)
  (a b c : ℝ)
  (h_gp_sides : b^2 = a * c) :
  α = β ∧ β = γ ∧ a = b ∧ b = c :=
sorry

end equilateral_triangle_of_ap_angles_gp_sides_l1597_159727


namespace correct_completion_l1597_159750

-- Definitions of conditions
def sentence_template := "By the time he arrives, all the work ___, with ___ our teacher will be content."
def option_A := ("will be accomplished", "that")
def option_B := ("will have been accomplished", "which")
def option_C := ("will have accomplished", "it")
def option_D := ("had been accomplished", "him")

-- The actual proof statement
theorem correct_completion : (option_B.fst = "will have been accomplished") ∧ (option_B.snd = "which") :=
by
  sorry

end correct_completion_l1597_159750


namespace value_of_x2_y2_z2_l1597_159722

variable (x y z : ℝ)

theorem value_of_x2_y2_z2 (h1 : x^2 + 3 * y = 4) 
                          (h2 : y^2 - 5 * z = 5) 
                          (h3 : z^2 - 7 * x = -8) : 
                          x^2 + y^2 + z^2 = 20.75 := 
by
  sorry

end value_of_x2_y2_z2_l1597_159722


namespace exp_division_rule_l1597_159737

-- The theorem to prove the given problem
theorem exp_division_rule (x : ℝ) (hx : x ≠ 0) :
  x^10 / x^5 = x^5 :=
by sorry

end exp_division_rule_l1597_159737


namespace tangent_line_at_b_l1597_159779

theorem tangent_line_at_b (b : ℝ) : (∃ x : ℝ, (4*x^3 = 4) ∧ (4*x + b = x^4 - 1)) ↔ (b = -4) := 
by 
  sorry

end tangent_line_at_b_l1597_159779


namespace trains_cross_time_l1597_159701

def length_train1 := 140 -- in meters
def length_train2 := 160 -- in meters

def speed_train1_kmph := 60 -- in km/h
def speed_train2_kmph := 48 -- in km/h

def kmph_to_mps (speed : ℕ) := speed * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

def relative_speed_mps := speed_train1_mps + speed_train2_mps

def total_length := length_train1 + length_train2

def time_to_cross := total_length / relative_speed_mps

theorem trains_cross_time : time_to_cross = 10 :=
  by sorry

end trains_cross_time_l1597_159701


namespace remainder_theorem_div_l1597_159772

noncomputable
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

theorem remainder_theorem_div (A B C : ℝ) (h : p A B C 2 = 13) : p A B C (-2) = 13 :=
by
  -- Proof goes here
  sorry

end remainder_theorem_div_l1597_159772


namespace initial_average_customers_l1597_159793

theorem initial_average_customers (x A : ℕ) (h1 : x = 1) (h2 : (A + 120) / 2 = 90) : A = 60 := by
  sorry

end initial_average_customers_l1597_159793


namespace min_value_f_range_of_a_l1597_159768

-- Define the function f(x) with parameter a.
def f (x a : ℝ) := |x + a| + |x - a|

-- (Ⅰ) Statement: Prove that for a = 1, the minimum value of f(x) is 2.
theorem min_value_f (x : ℝ) : f x 1 ≥ 2 :=
  by sorry

-- (Ⅱ) Statement: Prove that if f(2) > 5, then the range of values for a is (-∞, -5/2) ∪ (5/2, +∞).
theorem range_of_a (a : ℝ) : f 2 a > 5 → a < -5 / 2 ∨ a > 5 / 2 :=
  by sorry

end min_value_f_range_of_a_l1597_159768


namespace insects_per_group_correct_l1597_159731

-- Define the numbers of insects collected by boys and girls
def boys_insects : ℕ := 200
def girls_insects : ℕ := 300
def total_insects : ℕ := boys_insects + girls_insects

-- Define the number of groups
def groups : ℕ := 4

-- Define the expected number of insects per group using total insects and groups
def insects_per_group : ℕ := total_insects / groups

-- Prove that each group gets 125 insects
theorem insects_per_group_correct : insects_per_group = 125 :=
by
  -- The proof is omitted (just setting up the theorem statement)
  sorry

end insects_per_group_correct_l1597_159731


namespace largest_four_digit_number_prop_l1597_159739

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l1597_159739


namespace perpendicular_tangent_l1597_159723

noncomputable def f (x a : ℝ) := (x + a) * Real.exp x -- Defines the function

theorem perpendicular_tangent (a : ℝ) : 
  ∀ (tangent_slope perpendicular_slope : ℝ), 
  (tangent_slope = 1) → 
  (perpendicular_slope = -1) →
  tangent_slope = Real.exp 0 * (a + 1) →
  tangent_slope + perpendicular_slope = 0 → 
  a = 0 := by 
  intros tangent_slope perpendicular_slope htangent hperpendicular hderiv hperpendicular_slope
  sorry

end perpendicular_tangent_l1597_159723


namespace inequality_proof_l1597_159724

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ 2 * (a^3 + b^3 + c^3) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l1597_159724


namespace find_n_l1597_159712

variable {a : ℕ → ℝ} (h1 : a 4 = 7) (h2 : a 3 + a 6 = 16)

theorem find_n (n : ℕ) (h3 : a n = 31) : n = 16 := by
  sorry

end find_n_l1597_159712


namespace percentage_increase_in_overtime_rate_l1597_159755

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_compensation : ℝ := 976
def total_hours_worked : ℝ := 52

theorem percentage_increase_in_overtime_rate :
  ((total_compensation - (regular_rate * regular_hours)) / (total_hours_worked - regular_hours) - regular_rate) / regular_rate * 100 = 75 :=
by
  sorry

end percentage_increase_in_overtime_rate_l1597_159755


namespace problem1_problem2_problem3_l1597_159733

-- Definition of given quantities and conditions
variables (a b x : ℝ) (α β : ℝ)

-- Given Conditions
@[simp] def cond1 := true
@[simp] def cond2 := true
@[simp] def cond3 := true
@[simp] def cond4 := true

-- First Question
theorem problem1 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    a * Real.sin α = b * Real.sin β := sorry

-- Second Question
theorem problem2 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    Real.sin β ≤ a / b := sorry

-- Third Question
theorem problem3 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    x = a * (1 - Real.cos α) + b * (1 - Real.cos β) := sorry

end problem1_problem2_problem3_l1597_159733


namespace mod_division_l1597_159796

theorem mod_division (N : ℕ) (h₁ : N = 5 * 2 + 0) : N % 4 = 2 :=
by sorry

end mod_division_l1597_159796


namespace average_minutes_per_day_l1597_159756

-- Definitions based on the conditions
variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def third_graders_time := 10 * third_graders f
def fourth_graders_time := 12 * fourth_graders f
def fifth_graders_time := 15 * fifth_graders f

def total_students := third_graders f + fourth_graders f + fifth_graders f
def total_time := third_graders_time f + fourth_graders_time f + fifth_graders_time f

-- Proof statement
theorem average_minutes_per_day : total_time f / total_students f = 11 := sorry

end average_minutes_per_day_l1597_159756


namespace complete_square_l1597_159780

theorem complete_square (y : ℝ) : y^2 + 12 * y + 40 = (y + 6)^2 + 4 :=
by {
  sorry
}

end complete_square_l1597_159780


namespace speed_difference_l1597_159714

def anna_time_min := 15
def ben_time_min := 25
def distance_miles := 8

def anna_speed_mph := (distance_miles : ℚ) / (anna_time_min / 60 : ℚ)
def ben_speed_mph := (distance_miles : ℚ) / (ben_time_min / 60 : ℚ)

theorem speed_difference : (anna_speed_mph - ben_speed_mph : ℚ) = 12.8 := by {
  sorry
}

end speed_difference_l1597_159714


namespace percentage_increase_l1597_159703

theorem percentage_increase
  (black_and_white_cost color_cost : ℕ)
  (h_bw : black_and_white_cost = 160)
  (h_color : color_cost = 240) :
  ((color_cost - black_and_white_cost) * 100) / black_and_white_cost = 50 :=
by
  sorry

end percentage_increase_l1597_159703


namespace common_ratio_of_geometric_sequence_l1597_159795

theorem common_ratio_of_geometric_sequence (S : ℕ → ℝ) (a_1 a_2 : ℝ) (q : ℝ)
  (h1 : S 3 = a_1 * (1 + q + q^2))
  (h2 : 2 * S 3 = 2 * a_1 + a_2) : 
  q = -1/2 := 
sorry

end common_ratio_of_geometric_sequence_l1597_159795


namespace bob_hair_length_l1597_159726

-- Define the current length of Bob's hair
def current_length : ℝ := 36

-- Define the growth rate in inches per month
def growth_rate : ℝ := 0.5

-- Define the duration in years
def duration_years : ℕ := 5

-- Define the total growth over the duration in years
def total_growth : ℝ := growth_rate * 12 * duration_years

-- Define the length of Bob's hair when he last cut it
def initial_length : ℝ := current_length - total_growth

-- Theorem stating that the length of Bob's hair when he last cut it was 6 inches
theorem bob_hair_length :
  initial_length = 6 :=
by
  -- Proof omitted
  sorry

end bob_hair_length_l1597_159726


namespace women_count_l1597_159748

def total_passengers : Nat := 54
def men : Nat := 18
def children : Nat := 10
def women : Nat := total_passengers - men - children

theorem women_count : women = 26 :=
sorry

end women_count_l1597_159748


namespace num_small_triangles_l1597_159792

-- Define the lengths of the legs of the large and small triangles
variables (a h b k : ℕ)

-- Define the areas of the large and small triangles
def area_large_triangle (a h : ℕ) : ℕ := (a * h) / 2
def area_small_triangle (b k : ℕ) : ℕ := (b * k) / 2

-- Define the main theorem
theorem num_small_triangles (ha : a = 6) (hh : h = 4) (hb : b = 2) (hk : k = 1) :
  (area_large_triangle a h) / (area_small_triangle b k) = 12 :=
by
  sorry

end num_small_triangles_l1597_159792


namespace greatest_divisor_of_630_lt_35_and_factor_of_90_l1597_159789

theorem greatest_divisor_of_630_lt_35_and_factor_of_90 : ∃ d : ℕ, d < 35 ∧ d ∣ 630 ∧ d ∣ 90 ∧ ∀ e : ℕ, (e < 35 ∧ e ∣ 630 ∧ e ∣ 90) → e ≤ d := 
sorry

end greatest_divisor_of_630_lt_35_and_factor_of_90_l1597_159789


namespace cost_of_each_entree_l1597_159777

def cost_of_appetizer : ℝ := 10
def number_of_entrees : ℝ := 4
def tip_percentage : ℝ := 0.20
def total_spent : ℝ := 108

theorem cost_of_each_entree :
  ∃ E : ℝ, total_spent = cost_of_appetizer + number_of_entrees * E + tip_percentage * (cost_of_appetizer + number_of_entrees * E) ∧ E = 20 :=
by
  sorry

end cost_of_each_entree_l1597_159777


namespace total_hours_correct_l1597_159736

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l1597_159736


namespace video_down_votes_l1597_159770

theorem video_down_votes 
  (up_votes : ℕ)
  (ratio_up_down : up_votes / 1394 = 45 / 17)
  (up_votes_known : up_votes = 3690) : 
  3690 / 1394 = 45 / 17 :=
by
  sorry

end video_down_votes_l1597_159770


namespace bruce_eggs_lost_l1597_159798

theorem bruce_eggs_lost :
  ∀ (initial_eggs remaining_eggs eggs_lost : ℕ), 
  initial_eggs = 75 → remaining_eggs = 5 →
  eggs_lost = initial_eggs - remaining_eggs →
  eggs_lost = 70 :=
by
  intros initial_eggs remaining_eggs eggs_lost h_initial h_remaining h_loss
  sorry

end bruce_eggs_lost_l1597_159798


namespace present_number_of_teachers_l1597_159753

theorem present_number_of_teachers (S T : ℕ) (h1 : S = 50 * T) (h2 : S + 50 = 25 * (T + 5)) : T = 3 := 
by 
  sorry

end present_number_of_teachers_l1597_159753


namespace audrey_sleep_time_l1597_159729

theorem audrey_sleep_time (T : ℝ) (h1 : (3 / 5) * T = 6) : T = 10 :=
by
  sorry

end audrey_sleep_time_l1597_159729
