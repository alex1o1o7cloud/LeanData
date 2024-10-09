import Mathlib

namespace cos_inequality_m_range_l706_70680

theorem cos_inequality_m_range (m : ℝ) : 
  (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end cos_inequality_m_range_l706_70680


namespace find_k_l706_70696

def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 60 < f a b c 9 ∧ f a b c 9 < 70)
  (h3 : 90 < f a b c 10 ∧ f a b c 10 < 100)
  (h4 : ∃ k : ℤ, 10000 * k < f a b c 100 ∧ f a b c 100 < 10000 * (k + 1))
  : k = 2 :=
sorry

end find_k_l706_70696


namespace not_possible_for_runners_in_front_l706_70627

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end not_possible_for_runners_in_front_l706_70627


namespace kelly_games_l706_70648

theorem kelly_games (initial_games give_away in_stock : ℕ) (h1 : initial_games = 50) (h2 : in_stock = 35) :
  give_away = initial_games - in_stock :=
by {
  -- initial_games = 50
  -- in_stock = 35
  -- Therefore, give_away = initial_games - in_stock
  sorry
}

end kelly_games_l706_70648


namespace binomial_12_3_equals_220_l706_70640

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l706_70640


namespace inequality_solution_sum_of_squares_geq_sum_of_products_l706_70687

-- Problem 1
theorem inequality_solution (x : ℝ) : (0 < x ∧ x < 2/3) ↔ (x + 2) / (2 - 3 * x) > 1 :=
by
  sorry

-- Problem 2
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end inequality_solution_sum_of_squares_geq_sum_of_products_l706_70687


namespace number_of_extreme_points_l706_70647

-- Define the function's derivative
def f_derivative (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- State the theorem
theorem number_of_extreme_points : ∃ n : ℕ, n = 2 ∧ 
  (∀ x, (f_derivative x = 0 → ((f_derivative (x - ε) > 0 ∧ f_derivative (x + ε) < 0) ∨ 
                             (f_derivative (x - ε) < 0 ∧ f_derivative (x + ε) > 0))) → 
   (x = 1 ∨ x = 2)) :=
sorry

end number_of_extreme_points_l706_70647


namespace percentage_increase_edge_length_l706_70617

theorem percentage_increase_edge_length (a a' : ℝ) (h : 6 * (a')^2 = 6 * a^2 + 1.25 * 6 * a^2) : a' = 1.5 * a :=
by sorry

end percentage_increase_edge_length_l706_70617


namespace largest_tile_side_length_l706_70650

theorem largest_tile_side_length (w l : ℕ) (hw : w = 120) (hl : l = 96) : 
  ∃ s, s = Nat.gcd w l ∧ s = 24 :=
by
  sorry

end largest_tile_side_length_l706_70650


namespace ellipse_foci_k_value_l706_70675

theorem ellipse_foci_k_value 
    (k : ℝ) 
    (h1 : 5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5): 
    k = 1 := 
by 
  sorry

end ellipse_foci_k_value_l706_70675


namespace range_of_a_l706_70658

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Proposition P: f(x) has a root in the interval [-1, 1]
def P (a : ℝ) : Prop := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0

-- Proposition Q: There is only one real number x satisfying the inequality
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- The theorem stating the range of a if either P or Q is false
theorem range_of_a (a : ℝ) : ¬(P a) ∨ ¬(Q a) → (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) :=
sorry

end range_of_a_l706_70658


namespace cost_of_each_gumdrop_l706_70609

theorem cost_of_each_gumdrop (cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) : 
  cents = 224 → gumdrops = 28 → cost_per_gumdrop = cents / gumdrops → cost_per_gumdrop = 8 :=
by
  intros h_cents h_gumdrops h_cost
  sorry

end cost_of_each_gumdrop_l706_70609


namespace hall_volume_proof_l706_70634

-- Define the given conditions.
def hall_length (l : ℝ) : Prop := l = 18
def hall_width (w : ℝ) : Prop := w = 9
def floor_ceiling_area_eq_wall_area (h l w : ℝ) : Prop := 
  2 * (l * w) = 2 * (l * h) + 2 * (w * h)

-- Define the volume calculation.
def hall_volume (l w h V : ℝ) : Prop := 
  V = l * w * h

-- The main theorem stating that the volume is 972 cubic meters.
theorem hall_volume_proof (l w h V : ℝ) 
  (length : hall_length l) 
  (width : hall_width w) 
  (fc_eq_wa : floor_ceiling_area_eq_wall_area h l w) 
  (volume : hall_volume l w h V) : 
  V = 972 :=
  sorry

end hall_volume_proof_l706_70634


namespace good_coloring_count_l706_70656

noncomputable def c_n (n : ℕ) : ℤ :=
  1 / 2 * (3^(n + 1) + (-1)^(n + 1))

theorem good_coloring_count (n : ℕ) : 
  ∃ c : ℕ → ℤ, c n = c_n n := sorry

end good_coloring_count_l706_70656


namespace division_problem_l706_70686

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l706_70686


namespace inequality_solution_maximum_expression_l706_70670

-- Problem 1: Inequality for x
theorem inequality_solution (x : ℝ) : |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 :=
by
  sorry

-- Problem 2: Maximum value for expression within [0, 1]
theorem maximum_expression (a b : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) : 
  ab + (1 - a - b) * (a + b) ≤ 1/3 :=
by
  sorry

end inequality_solution_maximum_expression_l706_70670


namespace unknown_rate_of_two_towels_l706_70621

theorem unknown_rate_of_two_towels :
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  known_cost + (2 * x) = total_average_price * number_of_towels :=
by
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  show known_cost + (2 * x) = total_average_price * number_of_towels
  sorry

end unknown_rate_of_two_towels_l706_70621


namespace solve_quadratic_for_q_l706_70673

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l706_70673


namespace initial_investment_l706_70664

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment (A : ℝ) (r : ℝ) (n t : ℕ) (P : ℝ) :
  A = 3630.0000000000005 → r = 0.10 → n = 1 → t = 2 → P = 3000 →
  A = compound_interest P r n t :=
by
  intros hA hr hn ht hP
  rw [compound_interest, hA, hr, hP]
  sorry

end initial_investment_l706_70664


namespace smallest_N_l706_70661

theorem smallest_N (l m n : ℕ) (N : ℕ) (h_block : N = l * m * n)
  (h_invisible : (l - 1) * (m - 1) * (n - 1) = 120) :
  N = 216 :=
sorry

end smallest_N_l706_70661


namespace inclination_angle_x_equals_3_is_90_l706_70646

-- Define the condition that line x = 3 is vertical
def is_vertical_line (x : ℝ) : Prop := x = 3

-- Define the inclination angle property for a vertical line
def inclination_angle_of_vertical_line_is_90 (x : ℝ) (h : is_vertical_line x) : ℝ :=
90   -- The angle is 90 degrees

-- Theorem statement to prove the inclination angle of the line x = 3 is 90 degrees
theorem inclination_angle_x_equals_3_is_90 :
  inclination_angle_of_vertical_line_is_90 3 (by simp [is_vertical_line]) = 90 :=
sorry  -- proof goes here


end inclination_angle_x_equals_3_is_90_l706_70646


namespace range_of_a_for_two_critical_points_l706_70615

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - Real.exp 1 * x^2 + 18

theorem range_of_a_for_two_critical_points (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ↔ (a ∈ Set.Ioo (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 1)) :=
sorry

end range_of_a_for_two_critical_points_l706_70615


namespace product_of_consecutive_integers_is_perfect_square_l706_70626

theorem product_of_consecutive_integers_is_perfect_square (n : ℤ) :
    n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 :=
sorry

end product_of_consecutive_integers_is_perfect_square_l706_70626


namespace total_volume_of_barrel_l706_70690

-- Define the total volume of the barrel and relevant conditions.
variable (x : ℝ) -- total volume of the barrel

-- State the given condition about the barrel's honey content.
def condition := (0.7 * x - 0.3 * x = 30)

-- Goal to prove:
theorem total_volume_of_barrel : condition x → x = 75 :=
by
  sorry

end total_volume_of_barrel_l706_70690


namespace remainder_when_divided_by_30_l706_70629

theorem remainder_when_divided_by_30 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 8])
  (h2 : 6 + y ≡ 8 [ZMOD 27])
  (h3 : 8 + y ≡ 27 [ZMOD 125]) :
  y ≡ 4 [ZMOD 30] :=
sorry

end remainder_when_divided_by_30_l706_70629


namespace volume_of_right_prism_with_trapezoid_base_l706_70677

variable (S1 S2 H a b h: ℝ)

theorem volume_of_right_prism_with_trapezoid_base 
  (hS1 : S1 = a * H) 
  (hS2 : S2 = b * H) 
  (h_trapezoid : a ≠ b) : 
  1 / 2 * (S1 + S2) * h = (1 / 2 * (a + b) * h) * H :=
by 
  sorry

end volume_of_right_prism_with_trapezoid_base_l706_70677


namespace point_c_third_quadrant_l706_70699

variable (a b : ℝ)

-- Definition of the conditions
def condition_1 : Prop := b = -1
def condition_2 : Prop := a = -3

-- Definition to check if a point is in the third quadrant
def is_third_quadrant (a b : ℝ) : Prop := a < 0 ∧ b < 0

-- The main statement to be proven
theorem point_c_third_quadrant (h1 : condition_1 b) (h2 : condition_2 a) :
  is_third_quadrant a b :=
by
  -- Proof of the theorem (to be completed)
  sorry

end point_c_third_quadrant_l706_70699


namespace passengers_with_round_trip_tickets_l706_70651

theorem passengers_with_round_trip_tickets (P R : ℝ) : 
  (0.40 * R = 0.25 * P) → (R / P = 0.625) :=
by
  intro h
  sorry

end passengers_with_round_trip_tickets_l706_70651


namespace extreme_points_inequality_l706_70672

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem extreme_points_inequality (a : ℝ) (h_a : 0 < a ∧ a < 1) (x0 : ℝ)
  (h_ext : 4 * x0^2 - 4 * x0 + a = 0) (h_min : ∃ x1, x0 + x1 = 1 ∧ x0 < x1 ∧ x1 < 1) :
  g x0 a > 1 / 2 - Real.log 2 :=
sorry

end extreme_points_inequality_l706_70672


namespace mango_distribution_l706_70616

theorem mango_distribution (harvested_mangoes : ℕ) (sold_fraction : ℕ) (received_per_neighbor : ℕ)
  (h_harvested : harvested_mangoes = 560)
  (h_sold_fraction : sold_fraction = 2)
  (h_received_per_neighbor : received_per_neighbor = 35) :
  (harvested_mangoes / sold_fraction) = (harvested_mangoes / sold_fraction) / received_per_neighbor :=
by
  sorry

end mango_distribution_l706_70616


namespace cost_of_pencil_pen_eraser_l706_70635

variables {p q r : ℝ}

theorem cost_of_pencil_pen_eraser 
  (h1 : 4 * p + 3 * q + r = 5.40)
  (h2 : 2 * p + 2 * q + 2 * r = 4.60) : 
  p + 2 * q + 3 * r = 4.60 := 
by sorry

end cost_of_pencil_pen_eraser_l706_70635


namespace relationship_ab_l706_70602

noncomputable def a : ℝ := Real.log 243 / Real.log 5
noncomputable def b : ℝ := Real.log 27 / Real.log 3

theorem relationship_ab : a = (5 / 3) * b := sorry

end relationship_ab_l706_70602


namespace train_passes_man_in_approx_21_seconds_l706_70694

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def man_speed_kmph : ℝ := 6

-- Convert speeds to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

-- Calculate relative speed
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Calculate time
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_21_seconds : abs (time_to_pass - 21) < 1 :=
by
  sorry

end train_passes_man_in_approx_21_seconds_l706_70694


namespace problem1_problem2_l706_70697

def box (n : ℕ) : ℕ := (10^n - 1) / 9

theorem problem1 (m : ℕ) :
  let b := box (3^m)
  b % (3^m) = 0 ∧ b % (3^(m+1)) ≠ 0 :=
  sorry

theorem problem2 (n : ℕ) :
  (n % 27 = 0) ↔ (box n % 27 = 0) :=
  sorry

end problem1_problem2_l706_70697


namespace remaining_amount_to_be_paid_l706_70600

-- Define the conditions
def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 80

-- Define the total purchase price based on the conditions
def total_price : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_price - deposit_amount

-- State the theorem
theorem remaining_amount_to_be_paid : remaining_amount = 720 := by
  sorry

end remaining_amount_to_be_paid_l706_70600


namespace larger_number_l706_70681

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l706_70681


namespace sin_6_cos_6_theta_proof_l706_70671

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end sin_6_cos_6_theta_proof_l706_70671


namespace positive_integer_divisibility_l706_70693

theorem positive_integer_divisibility (n : ℕ) (h_pos : n > 0) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 := 
sorry

end positive_integer_divisibility_l706_70693


namespace cost_per_ounce_l706_70632

theorem cost_per_ounce (total_cost : ℕ) (num_ounces : ℕ) (h1 : total_cost = 84) (h2 : num_ounces = 12) : (total_cost / num_ounces) = 7 :=
by
  sorry

end cost_per_ounce_l706_70632


namespace minimum_a_plus_2c_l706_70678

theorem minimum_a_plus_2c (a c : ℝ) (h : (1 / a) + (1 / c) = 1) : a + 2 * c ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end minimum_a_plus_2c_l706_70678


namespace maximum_marks_l706_70663

-- Definitions based on the conditions
def passing_percentage : ℝ := 0.5
def student_marks : ℝ := 200
def marks_to_pass : ℝ := student_marks + 20

-- Lean 4 statement for the proof problem
theorem maximum_marks (M : ℝ) 
  (h1 : marks_to_pass = 220)
  (h2 : passing_percentage * M = marks_to_pass) :
  M = 440 :=
sorry

end maximum_marks_l706_70663


namespace riverview_problem_l706_70637

theorem riverview_problem (h c : Nat) (p : Nat := 4 * h) (s : Nat := 5 * c) (d : Nat := 4 * p) :
  (p + h + s + c + d = 52 → false) :=
by {
  sorry
}

end riverview_problem_l706_70637


namespace scientific_notation_l706_70624

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l706_70624


namespace find_number_l706_70614

theorem find_number (x : ℕ) (h : 5 + x = 20) : x = 15 :=
sorry

end find_number_l706_70614


namespace g_at_4_l706_70665

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 5 / x
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8.142857 := by
  sorry

end g_at_4_l706_70665


namespace smallest_number_divisible_l706_70611

   theorem smallest_number_divisible (d n : ℕ) (h₁ : (n + 7) % 11 = 0) (h₂ : (n + 7) % 24 = 0) (h₃ : (n + 7) % d = 0) (h₄ : (n + 7) = 257) : n = 250 :=
   by
     sorry
   
end smallest_number_divisible_l706_70611


namespace zoo_problem_l706_70623

theorem zoo_problem (M B L : ℕ) (h1: 26 ≤ M + B + L) (h2: M + B + L ≤ 32) 
    (h3: M + L > B) (h4: B + L = 2 * M) (h5: M + B = 3 * L + 3) (h6: B = L / 2) : 
    B = 3 :=
by
  sorry

end zoo_problem_l706_70623


namespace ratio_of_surface_areas_l706_70610

-- Definitions based on conditions
def side_length_ratio (a b : ℝ) : Prop := b = 6 * a
def surface_area (a : ℝ) : ℝ := 6 * a ^ 2

-- Theorem statement
theorem ratio_of_surface_areas (a b : ℝ) (h : side_length_ratio a b) :
  (surface_area b) / (surface_area a) = 36 := by
  sorry

end ratio_of_surface_areas_l706_70610


namespace abs_add_lt_abs_sub_l706_70684

variable {a b : ℝ}

theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_add_lt_abs_sub_l706_70684


namespace B_subset_A_iff_l706_70688

namespace MathProofs

def A (x : ℝ) : Prop := -2 < x ∧ x < 5

def B (x : ℝ) (m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem B_subset_A_iff (m : ℝ) :
  (∀ x : ℝ, B x m → A x) ↔ m < 3 :=
by
  sorry

end MathProofs

end B_subset_A_iff_l706_70688


namespace find_value_of_expression_l706_70679

theorem find_value_of_expression
  (x y : ℝ)
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) :
  x^2 * y^2 + 2 * x * y + 1 = 16 :=
sorry

end find_value_of_expression_l706_70679


namespace students_like_apple_chocolate_not_blueberry_l706_70618

theorem students_like_apple_chocolate_not_blueberry
  (n d a b c abc : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : a = 25)
  (h4 : b = 20)
  (h5 : c = 10)
  (h6 : abc = 5)
  (h7 : (n - d) = 35)
  (h8 : (55 - (a + b + c - abc)) = 35) :
  (20 - abc) = (15 : ℕ) :=
by
  sorry

end students_like_apple_chocolate_not_blueberry_l706_70618


namespace find_point_C_l706_70698

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def on_parabola (C : ℝ × ℝ) : Prop := parabola_eq C.1 C.2
def perpendicular_at_C (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Points A and B satisfy both the line and parabola equations
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ parabola_eq A.1 A.2 ∧
  line_eq B.1 B.2 ∧ parabola_eq B.1 B.2

-- Statement to be proven
theorem find_point_C (A B : ℝ × ℝ) (hA : intersection_points A B) :
  ∃ C : ℝ × ℝ, on_parabola C ∧ perpendicular_at_C A B C ∧
    (C = (1, -2) ∨ C = (9, -6)) :=
by
  sorry

end find_point_C_l706_70698


namespace brick_piles_l706_70666

theorem brick_piles (x y z : ℤ) :
  2 * (x - 100) = y + 100 ∧
  x + z = 6 * (y - z) →
  x = 170 ∧ y = 40 :=
by
  sorry

end brick_piles_l706_70666


namespace intersection_A_B_l706_70660

open Set

-- Define sets A and B with given conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

-- Prove the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {0, 3} := 
by
  sorry

end intersection_A_B_l706_70660


namespace probability_A_given_B_l706_70682

def roll_outcomes : ℕ := 6^3 -- Total number of possible outcomes when rolling three dice

def P_AB : ℚ := 60 / 216 -- Probability of both events A and B happening

def P_B : ℚ := 91 / 216 -- Probability of event B happening

theorem probability_A_given_B : (P_AB / P_B) = (60 / 91) := by
  sorry

end probability_A_given_B_l706_70682


namespace working_mom_hours_at_work_l706_70633

-- Definitions corresponding to the conditions
def hours_awake : ℕ := 16
def work_percentage : ℝ := 0.50

-- The theorem to be proved
theorem working_mom_hours_at_work : work_percentage * hours_awake = 8 :=
by sorry

end working_mom_hours_at_work_l706_70633


namespace find_larger_number_l706_70642

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l706_70642


namespace trigonometric_identity_l706_70620

noncomputable def tan_sum (alpha : ℝ) : Prop :=
  Real.tan (alpha + Real.pi / 4) = 2

noncomputable def trigonometric_expression (alpha : ℝ) : ℝ :=
  (Real.sin alpha + 2 * Real.cos alpha) / (Real.sin alpha - 2 * Real.cos alpha)

theorem trigonometric_identity (alpha : ℝ) (h : tan_sum alpha) : 
  trigonometric_expression alpha = -7 / 5 :=
sorry

end trigonometric_identity_l706_70620


namespace whale_population_ratio_l706_70653

theorem whale_population_ratio 
  (W_last : ℕ)
  (W_this : ℕ)
  (W_next : ℕ)
  (h1 : W_last = 4000)
  (h2 : W_next = W_this + 800)
  (h3 : W_next = 8800) :
  (W_this / W_last) = 2 := by
  sorry

end whale_population_ratio_l706_70653


namespace jerry_remaining_debt_l706_70654

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l706_70654


namespace fraction_decomposition_l706_70655

noncomputable def p (n : ℕ) : ℚ :=
  (n + 1) / 2

noncomputable def q (n : ℕ) : ℚ :=
  n * p n

theorem fraction_decomposition (n : ℕ) (h : ∃ k : ℕ, n = 5 + 2*k) :
  (2 / n : ℚ) = (1 / p n) + (1 / q n) :=
by
  sorry

end fraction_decomposition_l706_70655


namespace convert_spherical_coordinates_l706_70603

theorem convert_spherical_coordinates (
  ρ θ φ : ℝ
) (h1 : ρ = 5) (h2 : θ = 3 * Real.pi / 4) (h3 : φ = 9 * Real.pi / 4) : 
ρ = 5 ∧ 0 ≤ 7 * Real.pi / 4 ∧ 7 * Real.pi / 4 < 2 * Real.pi ∧ 0 ≤ Real.pi / 4 ∧ Real.pi / 4 ≤ Real.pi :=
by
  sorry

end convert_spherical_coordinates_l706_70603


namespace no_square_ends_in_2012_l706_70608

theorem no_square_ends_in_2012 : ¬ ∃ a : ℤ, (a * a) % 10 = 2 := by
  sorry

end no_square_ends_in_2012_l706_70608


namespace cost_of_drapes_l706_70645

theorem cost_of_drapes (D: ℝ) (h1 : 3 * 40 = 120) (h2 : D * 3 + 120 = 300) : D = 60 :=
  sorry

end cost_of_drapes_l706_70645


namespace avery_work_time_l706_70619

theorem avery_work_time :
  ∀ (t : ℝ),
    (1/2 * t + 1/4 * 1 = 1) → t = 1 :=
by
  intros t h
  sorry

end avery_work_time_l706_70619


namespace problem_l706_70636

-- Define what it means to be a factor or divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a
def is_divisor (a b : ℕ) : Prop := a ∣ b

-- The specific problem conditions
def statement_A := is_factor 4 28
def statement_B := is_divisor 19 209 ∧ ¬ is_divisor 19 57
def statement_C := ¬ is_divisor 30 90 ∧ ¬ is_divisor 30 76
def statement_D := is_divisor 14 28 ∧ ¬ is_divisor 14 56
def statement_E := is_factor 9 162

-- The proof problem
theorem problem : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D ∧ statement_E :=
by 
  -- You would normally provide the proof here
  sorry

end problem_l706_70636


namespace sam_correct_percent_l706_70674

variable (y : ℝ)
variable (h_pos : 0 < y)

theorem sam_correct_percent :
  ((8 * y - 3 * y) / (8 * y) * 100) = 62.5 := by
sorry

end sam_correct_percent_l706_70674


namespace value_of_a_l706_70657

/-- Given that 0.5% of a is 85 paise, prove that the value of a is 170 rupees. --/
theorem value_of_a (a : ℝ) (h : 0.005 * a = 85) : a = 170 := 
  sorry

end value_of_a_l706_70657


namespace sunil_interest_l706_70631

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := 19828.80 / (1 + 0.08) ^ 2
  P * (1 + r / n) ^ (n * t) = 19828.80 →
  A - P = 2828.80 :=
by
  sorry

end sunil_interest_l706_70631


namespace car_value_reduction_l706_70612

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end car_value_reduction_l706_70612


namespace distance_A_to_B_is_7km_l706_70605

theorem distance_A_to_B_is_7km
  (v1 v2 : ℝ) 
  (t_meet_before : ℝ)
  (t1_after_meet t2_after_meet : ℝ)
  (d1_before_meet d2_before_meet : ℝ)
  (d_after_meet : ℝ)
  (h1 : d1_before_meet = d2_before_meet + 1)
  (h2 : t_meet_before = d1_before_meet / v1)
  (h3 : t_meet_before = d2_before_meet / v2)
  (h4 : t1_after_meet = 3 / 4)
  (h5 : t2_after_meet = 4 / 3)
  (h6 : d1_before_meet + v1 * t1_after_meet = d_after_meet)
  (h7 : d2_before_meet + v2 * t2_after_meet = d_after_meet)
  : d_after_meet = 7 := 
sorry

end distance_A_to_B_is_7km_l706_70605


namespace sin_2alpha_over_cos_alpha_sin_beta_value_l706_70622

variable (α β : ℝ)

-- Given conditions
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < Real.pi / 2
axiom beta_pos : 0 < β
axiom beta_lt_pi_div_2 : β < Real.pi / 2
axiom cos_alpha_eq : Real.cos α = 3 / 5
axiom cos_beta_plus_alpha_eq : Real.cos (β + α) = 5 / 13

-- The results to prove
theorem sin_2alpha_over_cos_alpha : (Real.sin (2 * α) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 12) :=
sorry

theorem sin_beta_value : (Real.sin β = 16 / 65) :=
sorry


end sin_2alpha_over_cos_alpha_sin_beta_value_l706_70622


namespace customers_added_during_lunch_rush_l706_70607

noncomputable def initial_customers := 29.0
noncomputable def total_customers_after_lunch_rush := 83.0
noncomputable def expected_customers_added := 54.0

theorem customers_added_during_lunch_rush :
  (total_customers_after_lunch_rush - initial_customers) = expected_customers_added :=
by
  sorry

end customers_added_during_lunch_rush_l706_70607


namespace original_monthly_bill_l706_70695

-- Define the necessary conditions
def increased_bill (original: ℝ): ℝ := original + 0.3 * original
def total_bill_after_increase : ℝ := 78

-- The proof we need to construct
theorem original_monthly_bill (X : ℝ) (H : increased_bill X = total_bill_after_increase) : X = 60 :=
by {
    sorry -- Proof is not required, only statement
}

end original_monthly_bill_l706_70695


namespace sum_of_0_75_of_8_and_2_l706_70692

theorem sum_of_0_75_of_8_and_2 : 0.75 * 8 + 2 = 8 := by
  sorry

end sum_of_0_75_of_8_and_2_l706_70692


namespace find_angle_A_range_area_of_triangle_l706_70668

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem find_angle_A (h1 : b^2 + c^2 = a^2 - b * c) : A = (2 : ℝ) * Real.pi / 3 :=
by sorry

theorem range_area_of_triangle (h1 : b^2 + c^2 = a^2 - b * c)
(h2 : b * Real.sin A = 4 * Real.sin B) 
(h3 : Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C)) 
(h4 : A = (2 : ℝ) * Real.pi / 3) :
(Real.sqrt 3 / 4 : ℝ) ≤ (1 / 2) * b * c * Real.sin A ∧
(1 / 2) * b * c * Real.sin A ≤ (4 * Real.sqrt 3 / 3 : ℝ) :=
by sorry

end find_angle_A_range_area_of_triangle_l706_70668


namespace simplify_fraction_l706_70601

theorem simplify_fraction : (90 : ℚ) / (126 : ℚ) = 5 / 7 := 
by
  sorry

end simplify_fraction_l706_70601


namespace find_f_600_l706_70639

variable (f : ℝ → ℝ)
variable (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
variable (h2 : f 500 = 3)

theorem find_f_600 : f 600 = 5 / 2 :=
by
  sorry

end find_f_600_l706_70639


namespace triangle_shape_l706_70669

-- Defining the conditions:
variables (A B C a b c : ℝ)
variable (h1 : c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Defining the property to prove:
theorem triangle_shape : 
  (A = Real.pi / 2 ∨ A = B ∨ B = C ∨ C = A + B) :=
sorry

end triangle_shape_l706_70669


namespace altitudes_reciprocal_sum_eq_reciprocal_inradius_l706_70659

theorem altitudes_reciprocal_sum_eq_reciprocal_inradius
  (h1 h2 h3 r : ℝ)
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0)
  (h3_pos : h3 > 0)
  (r_pos : r > 0)
  (triangle_area_eq : ∀ (a b c : ℝ),
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧ a + b + c > 0) :
  1 / h1 + 1 / h2 + 1 / h3 = 1 / r := 
by
  sorry

end altitudes_reciprocal_sum_eq_reciprocal_inradius_l706_70659


namespace largest_of_nine_consecutive_integers_l706_70685

theorem largest_of_nine_consecutive_integers (sum_eq_99: ∃ (n : ℕ), 99 = (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) : 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end largest_of_nine_consecutive_integers_l706_70685


namespace sqrt_solution_l706_70676

theorem sqrt_solution (x : ℝ) (h : x = Real.sqrt (1 + x)) : 1 < x ∧ x < 2 :=
by
  sorry

end sqrt_solution_l706_70676


namespace PointNegativeThreeTwo_l706_70625

def isInSecondQuadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem PointNegativeThreeTwo:
  isInSecondQuadrant (-3) 2 := by
  sorry

end PointNegativeThreeTwo_l706_70625


namespace solution_l706_70691

open Real

variables (a b c A B C : ℝ)

-- Condition: In ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: Given equation relating sides and angles in ΔABC
axiom eq1 : a * sin C / (1 - cos A) = sqrt 3 * c
-- Condition: b + c = 10
axiom eq2 : b + c = 10
-- Condition: Area of ΔABC
axiom eq3 : (1 / 2) * b * c * sin A = 4 * sqrt 3

-- The final statement to prove
theorem solution :
    (A = π / 3) ∧ (a = 2 * sqrt 13) :=
by
    sorry

end solution_l706_70691


namespace smallest_positive_integer_l706_70604

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l706_70604


namespace find_m_plus_n_l706_70643

variable (x n m : ℝ)

def condition : Prop := (x + 5) * (x + n) = x^2 + m * x - 5

theorem find_m_plus_n (hnm : condition x n m) : m + n = 3 := 
sorry

end find_m_plus_n_l706_70643


namespace private_schools_in_district_B_l706_70638

theorem private_schools_in_district_B :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  remaining_private_schools = 4 :=
by
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  sorry

end private_schools_in_district_B_l706_70638


namespace max_d_minus_r_l706_70641

theorem max_d_minus_r (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) : 
  d - r = 35 :=
sorry

end max_d_minus_r_l706_70641


namespace distinct_cube_units_digits_l706_70667

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l706_70667


namespace susan_coins_value_l706_70683

-- Define the conditions as Lean functions and statements.
def total_coins (n d : ℕ) := n + d = 30
def value_if_swapped (n : ℕ) := 10 * n + 5 * (30 - n)
def value_original (n : ℕ) := 5 * n + 10 * (30 - n)
def conditions (n : ℕ) := value_if_swapped n = value_original n + 90

-- The proof statement
theorem susan_coins_value (n d : ℕ) (h1 : total_coins n d) (h2 : conditions n) : 5 * n + 10 * d = 180 := by
  sorry

end susan_coins_value_l706_70683


namespace total_students_sum_is_90_l706_70606

theorem total_students_sum_is_90:
  ∃ (x y z : ℕ), 
  (80 * x - 100 = 92 * (x - 5)) ∧
  (75 * y - 150 = 85 * (y - 6)) ∧
  (70 * z - 120 = 78 * (z - 4)) ∧
  (x + y + z = 90) :=
by
  sorry

end total_students_sum_is_90_l706_70606


namespace circle_radius_l706_70689

theorem circle_radius (P Q : ℝ) (h1 : P = π * r^2) (h2 : Q = 2 * π * r) (h3 : P / Q = 15) : r = 30 :=
by
  sorry

end circle_radius_l706_70689


namespace problem1_problem2_l706_70628

-- Definitions of the three conditions given
def condition1 (x y : Nat) : Prop := x > y
def condition2 (y z : Nat) : Prop := y > z
def condition3 (x z : Nat) : Prop := 2 * z > x

-- Problem 1: If the number of teachers is 4, prove the maximum number of female students is 6.
theorem problem1 (z : Nat) (hz : z = 4) : ∃ y : Nat, (∀ x : Nat, condition1 x y → condition2 y z → condition3 x z) ∧ y = 6 :=
by
  sorry

-- Problem 2: Prove the minimum number of people in the group is 12.
theorem problem2 : ∃ z x y : Nat, (condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ z < y ∧ y < x ∧ x < 2 * z) ∧ z = 3 ∧ x = 5 ∧ y = 4 ∧ x + y + z = 12 :=
by
  sorry

end problem1_problem2_l706_70628


namespace jose_birds_left_l706_70652

-- Define initial conditions
def chickens_initial : Nat := 28
def ducks : Nat := 18
def turkeys : Nat := 15
def chickens_sold : Nat := 12

-- Calculate remaining chickens
def chickens_left : Nat := chickens_initial - chickens_sold

-- Calculate total birds left
def total_birds_left : Nat := chickens_left + ducks + turkeys

-- Theorem statement to prove the number of birds left
theorem jose_birds_left : total_birds_left = 49 :=
by
  -- This is where the proof would typically go
  sorry

end jose_birds_left_l706_70652


namespace sum_difference_l706_70662

noncomputable def sum_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference :
  let S_even := sum_arith_seq 2 2 1001
  let S_odd := sum_arith_seq 1 2 1002
  S_odd - S_even = 1002 :=
by
  sorry

end sum_difference_l706_70662


namespace solve_inequality_range_of_a_l706_70649

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define the set A
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- First part: Solve the inequality f(x) ≤ 3a^2 + 1 when a ≠ 0
-- Solution would be translated in a theorem
theorem solve_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x a ≤ 3 * a^2 + 1 → if a > 0 then -a ≤ x ∧ x ≤ 3 * a else -3 * a ≤ x ∧ x ≤ a :=
sorry

-- Second part: Find the range of a if there exists no x0 ∈ A such that f(x0) ≤ A is false
theorem range_of_a (a : ℝ) :
  (∀ x ∈ A, f x a > 0) ↔ a < 1 :=
sorry

end solve_inequality_range_of_a_l706_70649


namespace right_angled_triangle_hypotenuse_and_altitude_relation_l706_70644

variables (a b c m : ℝ)

theorem right_angled_triangle_hypotenuse_and_altitude_relation
  (h1 : b^2 + c^2 = a^2)
  (h2 : m^2 = (b - c)^2)
  (h3 : b * c = a * m) :
  m = (a * (Real.sqrt 5 - 1)) / 2 := 
sorry

end right_angled_triangle_hypotenuse_and_altitude_relation_l706_70644


namespace point_B_value_l706_70613

/-- Given that point A represents the number 7 on a number line
    and point A is moved 3 units to the right to point B,
    prove that point B represents the number 10 -/
theorem point_B_value (A B : ℤ) (h1: A = 7) (h2: B = A + 3) : B = 10 :=
  sorry

end point_B_value_l706_70613


namespace constant_term_of_product_l706_70630

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + 7
def poly2 (x : ℝ) : ℝ := 4 * x^4 + 2 * x^2 + 10

-- Main statement: Prove that the constant term in the expansion of poly1 * poly2 is 70
theorem constant_term_of_product : (poly1 0) * (poly2 0) = 70 :=
by
  -- The proof would go here
  sorry

end constant_term_of_product_l706_70630
