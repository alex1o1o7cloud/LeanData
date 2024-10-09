import Mathlib

namespace evaluate_expression_l1204_120445

theorem evaluate_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 :=
by
  sorry

end evaluate_expression_l1204_120445


namespace no_preimage_range_l1204_120470

open Set

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem no_preimage_range :
  { k : ℝ | ∀ x : ℝ, f x ≠ k } = Iio 2 := by
  sorry

end no_preimage_range_l1204_120470


namespace sqrt_inequality_l1204_120406

theorem sqrt_inequality (x : ℝ) : abs ((x^2 - 9) / 3) < 3 ↔ -Real.sqrt 18 < x ∧ x < Real.sqrt 18 :=
by
  sorry

end sqrt_inequality_l1204_120406


namespace max_expression_value_l1204_120479

open Real

theorem max_expression_value : 
  ∃ q : ℝ, ∀ q : ℝ, -3 * q ^ 2 + 18 * q + 5 ≤ 32 ∧ (-3 * (3 ^ 2) + 18 * 3 + 5 = 32) :=
by
  sorry

end max_expression_value_l1204_120479


namespace number_of_distinct_real_roots_l1204_120426

theorem number_of_distinct_real_roots (k : ℕ) :
  (∃ k : ℕ, ∀ x : ℝ, |x| - 4 = (3 * |x|) / 2 → 0 = k) :=
  sorry

end number_of_distinct_real_roots_l1204_120426


namespace base8_subtraction_correct_l1204_120468

-- Define what it means to subtract in base 8
def base8_sub (a b : ℕ) : ℕ :=
  let a_base10 := 8 * (a / 10) + (a % 10)
  let b_base10 := 8 * (b / 10) + (b % 10)
  let result_base10 := a_base10 - b_base10
  8 * (result_base10 / 8) + (result_base10 % 8)

-- The given numbers in base 8
def num1 : ℕ := 52
def num2 : ℕ := 31
def expected_result : ℕ := 21

-- The proof problem statement
theorem base8_subtraction_correct : base8_sub num1 num2 = expected_result := by
  sorry

end base8_subtraction_correct_l1204_120468


namespace part1_daily_sales_profit_at_60_part2_selling_price_1350_l1204_120412

-- Definitions from conditions
def cost_per_piece : ℕ := 40
def selling_price_50_sales_volume : ℕ := 100
def sales_decrease_per_dollar : ℕ := 2
def max_selling_price : ℕ := 65

-- Problem Part (1)
def profit_at_60_yuan := 
  let selling_price := 60
  let profit_per_piece := selling_price - cost_per_piece
  let sales_decrease := (selling_price - 50) * sales_decrease_per_dollar
  let sales_volume := selling_price_50_sales_volume - sales_decrease
  let daily_profit := profit_per_piece * sales_volume
  daily_profit

theorem part1_daily_sales_profit_at_60 : profit_at_60_yuan = 1600 := by
  sorry

-- Problem Part (2)
def selling_price_for_1350_profit :=
  let desired_profit := 1350
  let sales_volume (x : ℕ) := selling_price_50_sales_volume - sales_decrease_per_dollar * (x - 50)
  let profit_per_x_piece (x : ℕ) := x - cost_per_piece
  let daily_sales_profit (x : ℕ) := (profit_per_x_piece x) * (sales_volume x)
  daily_sales_profit

theorem part2_selling_price_1350 : 
  ∃ x, x ≤ max_selling_price ∧ selling_price_for_1350_profit x = 1350 ∧ x = 55 := by
  sorry

end part1_daily_sales_profit_at_60_part2_selling_price_1350_l1204_120412


namespace exists_common_plane_l1204_120457

-- Definition of the triangular pyramids
structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

-- Function to represent the area of the intersection produced by a horizontal plane at distance x from the table
noncomputable def sectional_area (P : Pyramid) (x : ℝ) : ℝ :=
  P.base_area * (1 - x / P.height) ^ 2

-- Given seven pyramids
variables {P1 P2 P3 P4 P5 P6 P7 : Pyramid}

-- For any three pyramids, there exists a horizontal plane that intersects them in triangles of equal area
axiom triple_intersection:
  ∀ (Pi Pj Pk : Pyramid), ∃ x : ℝ, x ≥ 0 ∧ x ≤ min (Pi.height) (min (Pj.height) (Pk.height)) ∧
    sectional_area Pi x = sectional_area Pj x ∧ sectional_area Pk x = sectional_area Pi x

-- Prove that there exists a plane that intersects all seven pyramids in triangles of equal area
theorem exists_common_plane :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ min P1.height (min P2.height (min P3.height (min P4.height (min P5.height (min P6.height P7.height))))) ∧
    sectional_area P1 x = sectional_area P2 x ∧
    sectional_area P2 x = sectional_area P3 x ∧
    sectional_area P3 x = sectional_area P4 x ∧
    sectional_area P4 x = sectional_area P5 x ∧
    sectional_area P5 x = sectional_area P6 x ∧
    sectional_area P6 x = sectional_area P7 x :=
sorry

end exists_common_plane_l1204_120457


namespace mean_of_three_digit_multiples_of_8_l1204_120437

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end mean_of_three_digit_multiples_of_8_l1204_120437


namespace point_in_second_quadrant_l1204_120402

theorem point_in_second_quadrant (a : ℝ) (h1 : 2 * a + 1 < 0) (h2 : 1 - a > 0) : a < -1 / 2 := 
sorry

end point_in_second_quadrant_l1204_120402


namespace slopes_of_line_intersecting_ellipse_l1204_120473

noncomputable def possible_slopes : Set ℝ := {m : ℝ | m ≤ -1/Real.sqrt 20 ∨ m ≥ 1/Real.sqrt 20}

theorem slopes_of_line_intersecting_ellipse (m : ℝ) (h : ∃ x y, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) : 
  m ∈ possible_slopes :=
sorry

end slopes_of_line_intersecting_ellipse_l1204_120473


namespace slope_angle_of_line_x_equal_one_l1204_120442

noncomputable def slope_angle_of_vertical_line : ℝ := 90

theorem slope_angle_of_line_x_equal_one : slope_angle_of_vertical_line = 90 := by
  sorry

end slope_angle_of_line_x_equal_one_l1204_120442


namespace sum_of_digits_of_smallest_N_l1204_120400

-- Defining the conditions
def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k
def P (N : ℕ) : ℚ := ((2/3 : ℚ) * N * (1/3 : ℚ) * N) / ((N + 2) * (N + 3))
def S (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

-- The statement of the problem
theorem sum_of_digits_of_smallest_N :
  ∃ N : ℕ, is_multiple_of_6 N ∧ P N < (4/5 : ℚ) ∧ S N = 6 :=
sorry

end sum_of_digits_of_smallest_N_l1204_120400


namespace cos_4pi_over_3_l1204_120408

theorem cos_4pi_over_3 : Real.cos (4 * Real.pi / 3) = -1 / 2 :=
by 
  sorry

end cos_4pi_over_3_l1204_120408


namespace base_radius_of_cone_l1204_120478

-- Definitions of the conditions
def R1 : ℕ := 5
def R2 : ℕ := 4
def R3 : ℕ := 4
def height_radius_ratio := 4 / 3

-- Main theorem statement
theorem base_radius_of_cone : 
  (R1 = 5) → (R2 = 4) → (R3 = 4) → (height_radius_ratio = 4 / 3) → 
  ∃ r : ℚ, r = 169 / 60 :=
by 
  intros hR1 hR2 hR3 hRatio
  sorry

end base_radius_of_cone_l1204_120478


namespace verify_differential_eq_l1204_120452

noncomputable def y (x : ℝ) : ℝ := (2 + 3 * x - 3 * x^2)^(1 / 3 : ℝ)
noncomputable def y_prime (x : ℝ) : ℝ := 
  1 / 3 * (2 + 3 * x - 3 * x^2)^(-2 / 3 : ℝ) * (3 - 6 * x)

theorem verify_differential_eq (x : ℝ) :
  y x * y_prime x = (1 - 2 * x) / y x :=
by
  sorry

end verify_differential_eq_l1204_120452


namespace curves_intersect_at_4_points_l1204_120491

theorem curves_intersect_at_4_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = a^2 ∧ y = x^2 - a → ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x3, y3) ≠ (x4, y4) ∧
  (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x4, y4) ∧
  (x4, y4) ≠ (x3, y3) ∧ x1^2 + (y1 - 1)^2 = a^2 ∧ y1 = x1^2 - a ∧
  x2^2 + (y2 - 1)^2 = a^2 ∧ y2 = x2^2 - a ∧
  x3^2 + (y3 - 1)^2 = a^2 ∧ y3 = x3^2 - a ∧
  x4^2 + (y4 - 1)^2 = a^2 ∧ y4 = x4^2 - a) ↔ a > 0 :=
sorry

end curves_intersect_at_4_points_l1204_120491


namespace sasha_mistake_l1204_120486

/-- If Sasha obtained three numbers by raising 4 to various powers, such that all three units digits are different, 
     then Sasha's numbers cannot have three distinct last digits. -/
theorem sasha_mistake (h : ∀ n1 n2 n3 : ℕ, ∃ k1 k2 k3, n1 = 4^k1 ∧ n2 = 4^k2 ∧ n3 = 4^k3 ∧ (n1 % 10 ≠ n2 % 10) ∧ (n2 % 10 ≠ n3 % 10) ∧ (n1 % 10 ≠ n3 % 10)) :
False :=
sorry

end sasha_mistake_l1204_120486


namespace escalator_ride_time_l1204_120424

theorem escalator_ride_time (x y k t : ℝ)
  (h1 : 75 * x = y)
  (h2 : 30 * (x + k) = y)
  (h3 : t = y / k) :
  t = 50 := by
  sorry

end escalator_ride_time_l1204_120424


namespace pq_r_zero_l1204_120488

theorem pq_r_zero (p q r : ℝ) : 
  (∀ x : ℝ, x^4 + 6 * x^3 + 4 * p * x^2 + 2 * q * x + r = (x^3 + 4 * x^2 + 2 * x + 1) * (x - 2)) → 
  (p + q) * r = 0 :=
by
  sorry

end pq_r_zero_l1204_120488


namespace fraction_division_l1204_120472

theorem fraction_division : 
  ((8 / 4) * (9 / 3) * (20 / 5)) / ((10 / 5) * (12 / 4) * (15 / 3)) = (4 / 5) := 
by
  sorry

end fraction_division_l1204_120472


namespace min_value_of_expression_l1204_120401

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end min_value_of_expression_l1204_120401


namespace solution_set_of_inequality_l1204_120432

theorem solution_set_of_inequality : {x : ℝ | 8 * x^2 + 6 * x ≤ 2} = { x : ℝ | -1 ≤ x ∧ x ≤ (1/4) } :=
sorry

end solution_set_of_inequality_l1204_120432


namespace price_of_each_book_l1204_120492

theorem price_of_each_book (B P : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = 36) -- Number of unsold books is 1/3 of the total books and it equals 36
  (h2 : (2 / 3 : ℚ) * B * P = 144) -- Total amount received for the books sold is $144
  : P = 2 := 
by
  sorry

end price_of_each_book_l1204_120492


namespace Lindas_savings_l1204_120460

theorem Lindas_savings (S : ℝ) (h1 : (1/3) * S = 250) : S = 750 := 
by
  sorry

end Lindas_savings_l1204_120460


namespace infinite_set_k_l1204_120459

theorem infinite_set_k (C : ℝ) : ∃ᶠ k : ℤ in at_top, (k : ℝ) * Real.sin k > C :=
sorry

end infinite_set_k_l1204_120459


namespace solve_for_k_l1204_120482

theorem solve_for_k (x y k : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : 5 * x - k * y - 7 = 0) : k = 1 :=
by
  sorry

end solve_for_k_l1204_120482


namespace sum_of_squares_nonzero_l1204_120446

theorem sum_of_squares_nonzero {a b : ℝ} (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
sorry

end sum_of_squares_nonzero_l1204_120446


namespace sufficient_but_not_necessary_condition_l1204_120484

theorem sufficient_but_not_necessary_condition 
    (a : ℝ) (h_pos : a > 0)
    (h_line : ∀ x y, 2 * a * x - y + 2 * a^2 = 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / 4 = 1) :
    (a ≥ 2) → 
    (∀ x y, ¬ (2 * a * x - y + 2 * a^2 = 0 ∧ x^2 / a^2 - y^2 / 4 = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1204_120484


namespace tangent_parallel_l1204_120499

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

theorem tangent_parallel (a : ℝ) (H : ∀ x1 : ℝ, ∃ x2 : ℝ, (a - 2 * Real.sin x1) = (-Real.exp x2 - 1)) :
  a < -3 := by
  sorry

end tangent_parallel_l1204_120499


namespace solve_for_x_l1204_120427

variable (a b x : ℝ)

def operation (a b : ℝ) : ℝ := (a + 5) * b

theorem solve_for_x (h : operation x 1.3 = 11.05) : x = 3.5 :=
by
  sorry

end solve_for_x_l1204_120427


namespace complex_z_modulus_l1204_120403

noncomputable def i : ℂ := Complex.I

theorem complex_z_modulus (z : ℂ) (h : (1 + i) * z = 2 * i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_z_modulus_l1204_120403


namespace root_of_inverse_f_plus_x_eq_k_l1204_120415

variable {α : Type*} [Nonempty α] [Field α]
variable (f : α → α)
variable (f_inv : α → α)
variable (k : α)

def root_of_f_plus_x_eq_k (x : α) : Prop :=
  f x + x = k

def inverse_function (f : α → α) (f_inv : α → α) : Prop :=
  ∀ y : α, f (f_inv y) = y ∧ f_inv (f y) = y

theorem root_of_inverse_f_plus_x_eq_k
  (h1 : root_of_f_plus_x_eq_k f 5 k)
  (h2 : inverse_function f f_inv) :
  f_inv (k - 5) + (k - 5) = k :=
by
  sorry

end root_of_inverse_f_plus_x_eq_k_l1204_120415


namespace possible_triangular_frames_B_l1204_120461

-- Define the sides of the triangles and the similarity condition
def similar_triangles (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * b₃ = a₃ * b₁ ∧ a₂ * b₃ = a₃ * b₂

def sides_of_triangle_A := (50, 60, 80)

def is_a_possible_triangle (b₁ b₂ b₃ : ℕ) : Prop :=
  similar_triangles 50 60 80 b₁ b₂ b₃

-- Given conditions
def side_of_triangle_B := 20

-- Theorem to prove
theorem possible_triangular_frames_B :
  ∃ (b₂ b₃ : ℕ), (is_a_possible_triangle 20 b₂ b₃ ∨ is_a_possible_triangle b₂ 20 b₃ ∨ is_a_possible_triangle b₂ b₃ 20) :=
sorry

end possible_triangular_frames_B_l1204_120461


namespace drawn_from_grade12_correct_l1204_120489

-- Variables for the conditions
variable (total_students : ℕ) (sample_size : ℕ) (grade10_students : ℕ) 
          (grade11_students : ℕ) (grade12_students : ℕ) (drawn_from_grade12 : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 2400 ∧
  sample_size = 120 ∧
  grade10_students = 820 ∧
  grade11_students = 780 ∧
  grade12_students = total_students - grade10_students - grade11_students ∧
  drawn_from_grade12 = (grade12_students * sample_size) / total_students

-- Theorem to prove
theorem drawn_from_grade12_correct : conditions total_students sample_size grade10_students grade11_students grade12_students drawn_from_grade12 → drawn_from_grade12 = 40 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end drawn_from_grade12_correct_l1204_120489


namespace ball_colors_l1204_120449

theorem ball_colors (R G B : ℕ) (h1 : R + G + B = 15) (h2 : B = R + 1) (h3 : R = G) (h4 : B = G + 5) : false :=
by
  sorry

end ball_colors_l1204_120449


namespace max_expression_value_l1204_120456

theorem max_expression_value (a b c : ℝ) (hb : b > a) (ha : a > c) (hb_ne : b ≠ 0) :
  ∃ M, M = 27 ∧ (∀ a b c, b > a → a > c → b ≠ 0 → (∃ M, (2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2 ≤ M * b^2) → M ≤ 27) :=
  sorry

end max_expression_value_l1204_120456


namespace cone_ratio_l1204_120475

noncomputable def cone_height_ratio : ℚ :=
  let original_height := 40
  let circumference := 24 * Real.pi
  let original_radius := 12
  let new_volume := 432 * Real.pi
  let new_height := 9
  new_height / original_height

theorem cone_ratio (h : cone_height_ratio = 9 / 40) : (9 : ℚ) / 40 = 9 / 40 := by
  sorry

end cone_ratio_l1204_120475


namespace Donovan_Mitchell_current_average_l1204_120417

theorem Donovan_Mitchell_current_average 
    (points_per_game_goal : ℕ) 
    (games_played : ℕ) 
    (total_games_goal : ℕ) 
    (average_needed_remaining_games : ℕ)
    (points_needed : ℕ) 
    (remaining_games : ℕ) 
    (x : ℕ) 
    (h₁ : games_played = 15) 
    (h₂ : total_games_goal = 20) 
    (h₃ : points_per_game_goal = 30) 
    (h₄ : remaining_games = total_games_goal - games_played)
    (h₅ : average_needed_remaining_games = 42) 
    (h₆ : points_needed = remaining_games * average_needed_remaining_games) 
    (h₇ : points_needed = 210)  
    (h₈ : points_per_game_goal * total_games_goal = 600) 
    (h₉ : games_played * x + points_needed = 600) : 
    x = 26 :=
by {
  sorry
}

end Donovan_Mitchell_current_average_l1204_120417


namespace max_m_plus_n_l1204_120496

theorem max_m_plus_n (m n : ℝ) (h : n = -m^2 - 3*m + 3) : m + n ≤ 4 :=
by {
  sorry
}

end max_m_plus_n_l1204_120496


namespace geometry_progressions_not_exhaust_nat_l1204_120493

theorem geometry_progressions_not_exhaust_nat :
  ∃ (g : Fin 1975 → ℕ → ℕ), 
  (∀ i : Fin 1975, ∃ (a r : ℤ), ∀ n : ℕ, g i n = (a * r^n)) ∧
  (∃ m : ℕ, ∀ i : Fin 1975, ∀ n : ℕ, m ≠ g i n) :=
sorry

end geometry_progressions_not_exhaust_nat_l1204_120493


namespace fraction_evaluation_l1204_120476

theorem fraction_evaluation :
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 :=
by
  sorry

end fraction_evaluation_l1204_120476


namespace product_evaluation_l1204_120497

theorem product_evaluation : (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = 5040 := by
  -- sorry
  exact rfl  -- This is just a placeholder. The proof would go here.

end product_evaluation_l1204_120497


namespace maximize_x3y4_l1204_120455

noncomputable def maximize_expr (x y : ℝ) : ℝ :=
x^3 * y^4

theorem maximize_x3y4 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 50 ∧ maximize_expr x y = maximize_expr 30 20 :=
by
  sorry

end maximize_x3y4_l1204_120455


namespace hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l1204_120430

-- Definitions for shooting events for clarity
def hits_9_rings (s : String) := s = "9 rings"
def hits_8_rings (s : String) := s = "8 rings"

def hits_10_rings (s : String) := s = "10 rings"

def hits_target (s: String) := s = "hits target"
def does_not_hit_target (s: String) := s = "does not hit target"

-- Mutual exclusivity:
def mutually_exclusive (E1 E2 : Prop) := ¬ (E1 ∧ E2)

-- Problem 1:
theorem hits_9_and_8_mutually_exclusive :
  mutually_exclusive (hits_9_rings "9 rings") (hits_8_rings "8 rings") :=
sorry

-- Problem 2:
theorem hits_10_and_8_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_10_rings "10 rings" ) (hits_8_rings "8 rings") :=
sorry

-- Problem 3:
theorem both_hit_target_and_neither_hit_target_mutually_exclusive :
  mutually_exclusive (hits_target "both hit target") (does_not_hit_target "neither hit target") :=
sorry

-- Problem 4:
theorem at_least_one_hits_and_A_not_B_does_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_target "at least one hits target") (does_not_hit_target "A not but B does hit target") :=
sorry

end hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l1204_120430


namespace restaurant_discount_l1204_120463

theorem restaurant_discount :
  let coffee_price := 6
  let cheesecake_price := 10
  let discount_rate := 0.25
  let total_price := coffee_price + cheesecake_price
  let discount := discount_rate * total_price
  let final_price := total_price - discount
  final_price = 12 := by
  sorry

end restaurant_discount_l1204_120463


namespace total_area_covered_by_strips_l1204_120422

theorem total_area_covered_by_strips (L W : ℝ) (n : ℕ) (overlap_area : ℝ) (end_to_end_area : ℝ) :
  L = 15 → W = 1 → n = 4 → overlap_area = 15 → end_to_end_area = 30 → 
  (L * W * n - overlap_area + end_to_end_area) = 45 :=
by
  intros hL hW hn hoverlap hend_to_end
  sorry

end total_area_covered_by_strips_l1204_120422


namespace square_side_length_l1204_120487

theorem square_side_length (a b : ℕ) (h : a = 9) (h' : b = 16) (A : ℕ) (h1: A = a * b) :
  ∃ (s : ℕ), s * s = A ∧ s = 12 :=
by
  sorry

end square_side_length_l1204_120487


namespace not_a_function_l1204_120418

theorem not_a_function (angle_sine : ℝ → ℝ) 
                       (side_length_area : ℝ → ℝ) 
                       (sides_sum_int_angles : ℕ → ℝ)
                       (person_age_height : ℕ → Set ℝ) :
  (∃ y₁ y₂, y₁ ∈ person_age_height 20 ∧ y₂ ∈ person_age_height 20 ∧ y₁ ≠ y₂) :=
by {
  sorry
}

end not_a_function_l1204_120418


namespace tetrahedrons_volume_proportional_l1204_120498

-- Define the scenario and conditions.
variable 
  (V V' : ℝ) -- Volumes of the tetrahedrons
  (a b c a' b' c' : ℝ) -- Edge lengths emanating from vertices O and O'
  (α : ℝ) -- The angle between vectors OB and OC which is assumed to be congruent

-- Theorem statement.
theorem tetrahedrons_volume_proportional
  (congruent_trihedral_angles_at_O_and_O' : α = α) -- Condition of congruent trihedral angles
  : (V' / V) = (a' * b' * c') / (a * b * c) :=
sorry

end tetrahedrons_volume_proportional_l1204_120498


namespace max_lateral_surface_area_cylinder_optimizes_l1204_120423

noncomputable def max_lateral_surface_area_cylinder (r m : ℝ) : ℝ × ℝ :=
  let r_c := r / 2
  let h_c := m / 2
  (r_c, h_c)

theorem max_lateral_surface_area_cylinder_optimizes {r m : ℝ} (hr : 0 < r) (hm : 0 < m) :
  let (r_c, h_c) := max_lateral_surface_area_cylinder r m
  r_c = r / 2 ∧ h_c = m / 2 :=
sorry

end max_lateral_surface_area_cylinder_optimizes_l1204_120423


namespace fraction_of_3_5_eq_2_15_l1204_120443

theorem fraction_of_3_5_eq_2_15 : (2 / 15) / (3 / 5) = 2 / 9 := by
  sorry

end fraction_of_3_5_eq_2_15_l1204_120443


namespace compute_a_b_difference_square_l1204_120409

noncomputable def count_multiples (m n : ℕ) : ℕ :=
  (n - 1) / m

theorem compute_a_b_difference_square :
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  (a - b) ^ 2 = 0 :=
by
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  show (a - b) ^ 2 = 0
  sorry

end compute_a_b_difference_square_l1204_120409


namespace smallest_four_digit_divisible_43_l1204_120481

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end smallest_four_digit_divisible_43_l1204_120481


namespace find_y_l1204_120431

theorem find_y (y : ℕ) (h : 2^10 = 32^y) : y = 2 :=
by {
  sorry
}

end find_y_l1204_120431


namespace triangle_perimeter_l1204_120405

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 2) (h2 : (b-2)^2 + |c-3| = 0) : a + b + c = 7 :=
by
  sorry

end triangle_perimeter_l1204_120405


namespace number_of_terms_in_arithmetic_sequence_l1204_120483

theorem number_of_terms_in_arithmetic_sequence 
  (a : ℕ)
  (d : ℕ)
  (an : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : an = 47) :
  ∃ n : ℕ, an = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l1204_120483


namespace perimeter_of_plot_is_340_l1204_120407

def width : ℝ := 80 -- Derived width from the given conditions
def length (w : ℝ) : ℝ := w + 10 -- Length is 10 meters more than width
def perimeter (w : ℝ) : ℝ := 2 * (w + length w) -- Perimeter of the rectangle
def cost_per_meter : ℝ := 6.5 -- Cost rate per meter
def total_cost : ℝ := 2210 -- Total cost given

theorem perimeter_of_plot_is_340 :
  cost_per_meter * perimeter width = total_cost → perimeter width = 340 := 
by
  sorry

end perimeter_of_plot_is_340_l1204_120407


namespace puppies_left_l1204_120447

theorem puppies_left (initial_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : initial_puppies = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_puppies = initial_puppies - given_away) : 
  remaining_puppies = 5 :=
  by
  sorry

end puppies_left_l1204_120447


namespace f_is_odd_max_min_values_l1204_120464

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
variable (f_one : f 1 = -2)
variable (f_neg : ∀ x > 0, f x < 0)

-- Define the statement in Lean for Part 1: proving the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by sorry

-- Define the statement in Lean for Part 2: proving the max and min values on [-3, 3]
theorem max_min_values : 
  ∃ max_value min_value : ℝ, 
  (max_value = f (-3) ∧ max_value = 6) ∧ 
  (min_value = f (3) ∧ min_value = -6) := by sorry

end f_is_odd_max_min_values_l1204_120464


namespace value_of_expression_l1204_120410

theorem value_of_expression : (2207 - 2024)^2 * 4 / 144 = 930.25 := 
by
  sorry

end value_of_expression_l1204_120410


namespace thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l1204_120411

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem thirtieth_triangular_number :
  triangular_number 30 = 465 :=
by
  sorry

theorem sum_thirtieth_thirtyfirst_triangular_numbers :
  triangular_number 30 + triangular_number 31 = 961 :=
by
  sorry

end thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l1204_120411


namespace aurelia_percentage_l1204_120471

variables (P : ℝ)

theorem aurelia_percentage (h1 : 2000 + (P / 100) * 2000 = 3400) : 
  P = 70 :=
by
  sorry

end aurelia_percentage_l1204_120471


namespace find_rate_percent_l1204_120474

theorem find_rate_percent (P : ℝ) (r : ℝ) (A1 A2 : ℝ) (t1 t2 : ℕ)
  (h1 : A1 = P * (1 + r)^t1) (h2 : A2 = P * (1 + r)^t2) (hA1 : A1 = 2420) (hA2 : A2 = 3146) (ht1 : t1 = 2) (ht2 : t2 = 3) :
  r = 0.2992 :=
by
  sorry

end find_rate_percent_l1204_120474


namespace middle_letter_value_l1204_120419

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l1204_120419


namespace simple_interest_correct_l1204_120428

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end simple_interest_correct_l1204_120428


namespace maple_tree_total_l1204_120462

-- Conditions
def initial_maple_trees : ℕ := 53
def trees_planted_today : ℕ := 11

-- Theorem to prove the result
theorem maple_tree_total : initial_maple_trees + trees_planted_today = 64 := by
  sorry

end maple_tree_total_l1204_120462


namespace tangent_line_condition_l1204_120477

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end tangent_line_condition_l1204_120477


namespace area_range_of_triangle_l1204_120414

-- Defining the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 2

-- Function to compute the area of triangle ABP
noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2))

-- The proof goal statement
theorem area_range_of_triangle (P : ℝ × ℝ) (hp : on_circle P) :
  2 ≤ area_of_triangle P ∧ area_of_triangle P ≤ 6 :=
sorry

end area_range_of_triangle_l1204_120414


namespace sum_four_variables_l1204_120441

theorem sum_four_variables 
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + 2 = x)
  (h2 : b + 3 = x)
  (h3 : c + 4 = x)
  (h4 : d + 5 = x)
  (h5 : a + b + c + d + 8 = x) :
  a + b + c + d = -6 :=
by
  sorry

end sum_four_variables_l1204_120441


namespace length_of_jordans_rectangle_l1204_120458

theorem length_of_jordans_rectangle 
  (h1 : ∃ (length width : ℕ), length = 5 ∧ width = 24) 
  (h2 : ∃ (width_area : ℕ), width_area = 30 ∧ ∃ (area : ℕ), area = 5 * 24 ∧ ∃ (L : ℕ), area = L * width_area) :
  ∃ L, L = 4 := by 
  sorry

end length_of_jordans_rectangle_l1204_120458


namespace problem_inequality_l1204_120421

variable {x y : ℝ}

theorem problem_inequality (hx : 2 < x) (hy : 2 < y) : 
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2 / 3 := 
  sorry

end problem_inequality_l1204_120421


namespace range_a_of_tangents_coincide_l1204_120469

theorem range_a_of_tangents_coincide (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (a : ℝ)
  (h3 : -1 / (x2 ^ 2) = 2 * x1 + 1) (h4 : x1 ^ 2 = -a) :
  1/4 < a ∧ a < 1 :=
by
  sorry 

end range_a_of_tangents_coincide_l1204_120469


namespace makarala_meetings_percentage_l1204_120438

def work_day_to_minutes (hours: ℕ) : ℕ :=
  60 * hours

def total_meeting_time (first: ℕ) (second: ℕ) : ℕ :=
  let third := first + second
  first + second + third

def percentage_of_day_spent (meeting_time: ℕ) (work_day_time: ℕ) : ℚ :=
  (meeting_time : ℚ) / (work_day_time : ℚ) * 100

theorem makarala_meetings_percentage
  (work_hours: ℕ)
  (first_meeting: ℕ)
  (second_meeting: ℕ)
  : percentage_of_day_spent (total_meeting_time first_meeting second_meeting) (work_day_to_minutes work_hours) = 37.5 :=
by
  sorry

end makarala_meetings_percentage_l1204_120438


namespace functional_equation_solution_l1204_120495

def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intros h x
  sorry

end functional_equation_solution_l1204_120495


namespace age_of_B_l1204_120448

/--
A is two years older than B.
B is twice as old as C.
The total of the ages of A, B, and C is 32.
How old is B?
-/
theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 32) : B = 12 :=
by
  sorry

end age_of_B_l1204_120448


namespace smallest_omega_l1204_120466

theorem smallest_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω = 6 * k) ∧ (∀ k : ℤ, k > 0 → ω = 6 * k → ω = 6) :=
by sorry

end smallest_omega_l1204_120466


namespace q_investment_time_l1204_120451

theorem q_investment_time (x t : ℝ)
  (h1 : (7 * 20 * x) / (5 * t * x) = 7 / 10) : t = 40 :=
by
  sorry

end q_investment_time_l1204_120451


namespace equation_of_line_perpendicular_and_passing_point_l1204_120444

theorem equation_of_line_perpendicular_and_passing_point :
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = -1 ∧
  (∀ (x y : ℝ), (2 * x - 3 * y + 4 = 0 → y = (2 / 3) * x + 4 / 3) →
  (∀ (x1 y1 : ℝ), x1 = -1 ∧ y1 = 2 →
  (a * x1 + b * y1 + c = 0) ∧
  (∀ (x y : ℝ), (-3 / 2) * (x + 1) + 2 = y) →
  (a * x + b * y + c = 0))) :=
sorry

end equation_of_line_perpendicular_and_passing_point_l1204_120444


namespace cups_filled_with_tea_l1204_120467

theorem cups_filled_with_tea (total_tea ml_each_cup : ℕ)
  (h1 : total_tea = 1050)
  (h2 : ml_each_cup = 65) :
  total_tea / ml_each_cup = 16 := sorry

end cups_filled_with_tea_l1204_120467


namespace lettuce_types_l1204_120440

/-- Let L be the number of types of lettuce. 
    Given that Terry has 3 types of tomatoes, 4 types of olives, 
    and 2 types of soup. The total number of options for his lunch combo is 48. 
    Prove that L = 2. --/

theorem lettuce_types (L : ℕ) (H : 3 * 4 * 2 * L = 48) : L = 2 :=
by {
  -- beginning of the proof
  sorry
}

end lettuce_types_l1204_120440


namespace max_divisor_of_expression_l1204_120465

theorem max_divisor_of_expression 
  (n : ℕ) (hn : n > 0) : ∃ k, k = 8 ∧ 8 ∣ (5^n + 2 * 3^(n-1) + 1) :=
by
  sorry

end max_divisor_of_expression_l1204_120465


namespace area_of_tangency_triangle_l1204_120494

theorem area_of_tangency_triangle (c a b T varrho : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_area : T = (1/2) * a * b) (h_inradius : varrho = (a + b - c) / 2) :
  (area_tangency : ℝ) = (varrho / c) * T :=
sorry

end area_of_tangency_triangle_l1204_120494


namespace halved_r_value_of_n_l1204_120425

theorem halved_r_value_of_n (r a : ℝ) (n : ℕ) (h₁ : a = (2 * r)^n)
  (h₂ : 0.125 * a = r^n) : n = 3 :=
by
  sorry

end halved_r_value_of_n_l1204_120425


namespace percentage_difference_highest_lowest_salary_l1204_120433

variables (R : ℝ)
def Ram_salary := 1.25 * R
def Simran_salary := 0.85 * R
def Rahul_salary := 0.85 * R * 1.10

theorem percentage_difference_highest_lowest_salary :
  let highest_salary := Ram_salary R
  let lowest_salary := Simran_salary R
  (highest_salary ≠ 0) → ((highest_salary - lowest_salary) / highest_salary) * 100 = 32 :=
by
  intros
  -- Sorry in place of proof
  sorry

end percentage_difference_highest_lowest_salary_l1204_120433


namespace train_length_correct_l1204_120420

-- Define the conditions
def train_speed : ℝ := 63
def time_crossing : ℝ := 40
def expected_length : ℝ := 2520

-- The statement to prove
theorem train_length_correct : train_speed * time_crossing = expected_length :=
by
  exact sorry

end train_length_correct_l1204_120420


namespace radical_axis_theorem_l1204_120436

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def power_of_point (p : Point) (c : Circle) : ℝ :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2 - c.radius^2)

theorem radical_axis_theorem (O1 O2 : Circle) :
  ∃ L : ℝ → Point, 
  (∀ p : Point, (power_of_point p O1 = power_of_point p O2) → (L p.x = p)) ∧ 
  (O1.center.y = O2.center.y) ∧ 
  (∃ k : ℝ, ∀ x, L x = Point.mk x k) :=
sorry

end radical_axis_theorem_l1204_120436


namespace pieces_per_box_correct_l1204_120480

-- Define the number of boxes Will bought
def total_boxes_bought := 7

-- Define the number of boxes Will gave to his brother
def boxes_given := 3

-- Define the number of pieces left with Will
def pieces_left := 16

-- Define the function to find the pieces per box
def pieces_per_box (total_boxes : Nat) (given_away : Nat) (remaining_pieces : Nat) : Nat :=
  remaining_pieces / (total_boxes - given_away)

-- Prove that each box contains 4 pieces of chocolate candy
theorem pieces_per_box_correct : pieces_per_box total_boxes_bought boxes_given pieces_left = 4 :=
by
  sorry

end pieces_per_box_correct_l1204_120480


namespace stan_water_intake_l1204_120435

-- Define the constants and parameters given in the conditions
def words_per_minute : ℕ := 50
def pages : ℕ := 5
def words_per_page : ℕ := 400
def water_per_hour : ℚ := 15  -- use rational numbers for precise division

-- Define the derived quantities from the conditions
def total_words : ℕ := pages * words_per_page
def total_minutes : ℕ := total_words / words_per_minute
def water_per_minute : ℚ := water_per_hour / 60

-- State the theorem
theorem stan_water_intake : 10 = total_minutes * water_per_minute := by
  sorry

end stan_water_intake_l1204_120435


namespace product_of_real_values_r_l1204_120485

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end product_of_real_values_r_l1204_120485


namespace total_students_l1204_120413

theorem total_students (S : ℕ) (R : ℕ) :
  (2 * 0 + 12 * 1 + 13 * 2 + R * 3) / S = 2 →
  2 + 12 + 13 + R = S →
  S = 43 :=
by
  sorry

end total_students_l1204_120413


namespace marsupial_protein_l1204_120450

theorem marsupial_protein (absorbed : ℝ) (percent_absorbed : ℝ) (consumed : ℝ) :
  absorbed = 16 ∧ percent_absorbed = 0.4 → consumed = 40 :=
by
  sorry

end marsupial_protein_l1204_120450


namespace chip_final_balance_l1204_120404

noncomputable def finalBalance : ℝ := 
  let initialBalance := 50.0
  let month1InterestRate := 0.20
  let month2NewCharges := 20.0
  let month2InterestRate := 0.20
  let month3NewCharges := 30.0
  let month3Payment := 10.0
  let month3InterestRate := 0.25
  let month4NewCharges := 40.0
  let month4Payment := 20.0
  let month4InterestRate := 0.15

  -- Month 1
  let month1InterestFee := initialBalance * month1InterestRate
  let balanceMonth1 := initialBalance + month1InterestFee

  -- Month 2
  let balanceMonth2BeforeInterest := balanceMonth1 + month2NewCharges
  let month2InterestFee := balanceMonth2BeforeInterest * month2InterestRate
  let balanceMonth2 := balanceMonth2BeforeInterest + month2InterestFee

  -- Month 3
  let balanceMonth3BeforeInterest := balanceMonth2 + month3NewCharges
  let balanceMonth3AfterPayment := balanceMonth3BeforeInterest - month3Payment
  let month3InterestFee := balanceMonth3AfterPayment * month3InterestRate
  let balanceMonth3 := balanceMonth3AfterPayment + month3InterestFee

  -- Month 4
  let balanceMonth4BeforeInterest := balanceMonth3 + month4NewCharges
  let balanceMonth4AfterPayment := balanceMonth4BeforeInterest - month4Payment
  let month4InterestFee := balanceMonth4AfterPayment * month4InterestRate
  let balanceMonth4 := balanceMonth4AfterPayment + month4InterestFee

  balanceMonth4

theorem chip_final_balance : finalBalance = 189.75 := by sorry

end chip_final_balance_l1204_120404


namespace sufficient_but_not_necessary_condition_l1204_120416

def parabola (y : ℝ) : ℝ := y^2
def line (m : ℝ) (y : ℝ) : ℝ := m * y + 1

theorem sufficient_but_not_necessary_condition {m : ℝ} :
  (m ≠ 0) → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ parabola y1 = line m y1 ∧ parabola y2 = line m y2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l1204_120416


namespace max_number_of_children_l1204_120453

theorem max_number_of_children (apples cookies chocolates : ℕ) (remaining_apples remaining_cookies remaining_chocolates : ℕ) 
  (h₁ : apples = 55) 
  (h₂ : cookies = 114) 
  (h₃ : chocolates = 83) 
  (h₄ : remaining_apples = 3) 
  (h₅ : remaining_cookies = 10) 
  (h₆ : remaining_chocolates = 5) : 
  gcd (apples - remaining_apples) (gcd (cookies - remaining_cookies) (chocolates - remaining_chocolates)) = 26 :=
by
  sorry

end max_number_of_children_l1204_120453


namespace factor_expression_l1204_120429

theorem factor_expression (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) /
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) =
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) :=
by
  sorry

end factor_expression_l1204_120429


namespace beacon_population_l1204_120454

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end beacon_population_l1204_120454


namespace probability_of_both_red_is_one_sixth_l1204_120434

noncomputable def probability_both_red (red blue green : ℕ) (balls_picked : ℕ) : ℚ :=
  if balls_picked = 2 ∧ red = 4 ∧ blue = 3 ∧ green = 2 then (4 / 9) * (3 / 8) else 0

theorem probability_of_both_red_is_one_sixth :
  probability_both_red 4 3 2 2 = 1 / 6 :=
by
  unfold probability_both_red
  split_ifs
  · sorry
  · contradiction

end probability_of_both_red_is_one_sixth_l1204_120434


namespace income_is_108000_l1204_120490

theorem income_is_108000 (S I : ℝ) (h1 : S / I = 5 / 9) (h2 : 48000 = I - S) : I = 108000 :=
by
  sorry

end income_is_108000_l1204_120490


namespace polygon_sides_count_l1204_120439

-- Definitions for each polygon and their sides
def pentagon_sides := 5
def square_sides := 4
def hexagon_sides := 6
def heptagon_sides := 7
def nonagon_sides := 9

-- Compute the total number of sides
def total_exposed_sides :=
  (pentagon_sides + nonagon_sides - 2) + (square_sides + hexagon_sides + heptagon_sides - 6)

theorem polygon_sides_count : total_exposed_sides = 23 :=
by
  -- Mathematical proof steps can be detailed here
  -- For now, let's assume it is correctly given as a single number
  sorry

end polygon_sides_count_l1204_120439
