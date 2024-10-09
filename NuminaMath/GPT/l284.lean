import Mathlib

namespace annual_interest_rate_l284_28451

-- Define the initial conditions
def P : ℝ := 5600
def A : ℝ := 6384
def t : ℝ := 2
def n : ℝ := 1

-- The theorem statement:
theorem annual_interest_rate : ∃ (r : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ r = 0.067 :=
by 
  sorry -- proof goes here

end annual_interest_rate_l284_28451


namespace rory_more_jellybeans_l284_28456

-- Definitions based on the conditions
def G : ℕ := 15 -- Gigi has 15 jellybeans
def LorelaiConsumed (R G : ℕ) : ℕ := 3 * (R + G) -- Lorelai has already eaten three times the total number of jellybeans

theorem rory_more_jellybeans {R : ℕ} (h1 : LorelaiConsumed R G = 180) : (R - G) = 30 :=
  by
    -- we can skip the proof here with sorry, as we are only interested in the statement for now
    sorry

end rory_more_jellybeans_l284_28456


namespace max_value_of_m_l284_28419

theorem max_value_of_m (x m : ℝ) (h1 : x^2 - 4*x - 5 > 0) (h2 : x^2 - 2*x + 1 - m^2 > 0) (hm : m > 0) 
(hsuff : ∀ (x : ℝ), (x < -1 ∨ x > 5) → (x > m + 1 ∨ x < 1 - m)) : m ≤ 2 :=
sorry

end max_value_of_m_l284_28419


namespace find_x_l284_28421

theorem find_x (U : Set ℕ) (A B : Set ℕ) (x : ℕ) 
  (hU : U = Set.univ)
  (hA : A = {1, 4, x})
  (hB : B = {1, x ^ 2})
  (h : compl A ⊂ compl B) :
  x = 0 ∨ x = 2 := 
by 
  sorry

end find_x_l284_28421


namespace arcsin_one_eq_pi_div_two_l284_28407

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l284_28407


namespace cost_of_one_of_the_shirts_l284_28485

theorem cost_of_one_of_the_shirts
    (total_cost : ℕ) 
    (cost_two_shirts : ℕ) 
    (num_equal_shirts : ℕ) 
    (cost_of_shirt : ℕ) :
    total_cost = 85 → 
    cost_two_shirts = 20 → 
    num_equal_shirts = 3 → 
    cost_of_shirt = (total_cost - 2 * cost_two_shirts) / num_equal_shirts → 
    cost_of_shirt = 15 :=
by
  intros
  sorry

end cost_of_one_of_the_shirts_l284_28485


namespace find_a_decreasing_l284_28433

-- Define the given function
def f (a x : ℝ) : ℝ := (x - 1) ^ 2 + 2 * a * x + 1

-- State the condition
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y ≤ f x

-- State the proposition
theorem find_a_decreasing :
  ∀ a : ℝ, is_decreasing_on (f a) (Set.Iio 4) → a ≤ -3 :=
by
  intro a
  intro h
  sorry

end find_a_decreasing_l284_28433


namespace quadrant_of_alpha_l284_28429

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end quadrant_of_alpha_l284_28429


namespace percentage_difference_l284_28474

theorem percentage_difference :
  ((75 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25) = 10 := 
by
  sorry

end percentage_difference_l284_28474


namespace opening_night_customers_l284_28461

theorem opening_night_customers
  (matinee_tickets : ℝ := 5)
  (evening_tickets : ℝ := 7)
  (opening_night_tickets : ℝ := 10)
  (popcorn_cost : ℝ := 10)
  (num_matinee_customers : ℝ := 32)
  (num_evening_customers : ℝ := 40)
  (total_revenue : ℝ := 1670) :
  ∃ x : ℝ, 
    (matinee_tickets * num_matinee_customers + 
    evening_tickets * num_evening_customers + 
    opening_night_tickets * x + 
    popcorn_cost * (num_matinee_customers + num_evening_customers + x) / 2 = total_revenue) 
    ∧ x = 58 := 
by
  use 58
  sorry

end opening_night_customers_l284_28461


namespace distinct_real_roots_implies_positive_l284_28403

theorem distinct_real_roots_implies_positive (k : ℝ) (x1 x2 : ℝ) (h_distinct : x1 ≠ x2) 
  (h_root1 : x1^2 + 2*x1 - k = 0) 
  (h_root2 : x2^2 + 2*x2 - k = 0) : 
  x1^2 + x2^2 - 2 > 0 := 
sorry

end distinct_real_roots_implies_positive_l284_28403


namespace infinite_series_sum_l284_28491

theorem infinite_series_sum :
  ∑' n : ℕ, (1 / (n.succ * (n.succ + 2))) = 3 / 4 :=
by sorry

end infinite_series_sum_l284_28491


namespace exp_decreasing_range_l284_28404

theorem exp_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (a-2) ^ x < (a-2) ^ (x - 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end exp_decreasing_range_l284_28404


namespace diameter_increase_l284_28405

theorem diameter_increase (A A' D D' : ℝ)
  (hA_increase: A' = 4 * A)
  (hA: A = π * (D / 2)^2)
  (hA': A' = π * (D' / 2)^2) :
  D' = 2 * D :=
by 
  sorry

end diameter_increase_l284_28405


namespace g_of_36_l284_28415

theorem g_of_36 (g : ℕ → ℕ)
  (h1 : ∀ n, g (n + 1) > g n)
  (h2 : ∀ m n, g (m * n) = g m * g n)
  (h3 : ∀ m n, m ≠ n ∧ m ^ n = n ^ m → (g m = n ∨ g n = m))
  (h4 : ∀ n, g (n ^ 2) = g n * n) :
  g 36 = 36 :=
  sorry

end g_of_36_l284_28415


namespace parsnip_box_fullness_l284_28483

theorem parsnip_box_fullness (capacity : ℕ) (fraction_full : ℚ) (avg_boxes : ℕ) (avg_parsnips : ℕ) :
  capacity = 20 →
  fraction_full = 3 / 4 →
  avg_boxes = 20 →
  avg_parsnips = 350 →
  ∃ (full_boxes : ℕ) (non_full_boxes : ℕ) (parsnips_in_full_boxes : ℕ) (parsnips_in_non_full_boxes : ℕ)
    (avg_fullness_non_full_boxes : ℕ),
    full_boxes = fraction_full * avg_boxes ∧
    non_full_boxes = avg_boxes - full_boxes ∧
    parsnips_in_full_boxes = full_boxes * capacity ∧
    parsnips_in_non_full_boxes = avg_parsnips - parsnips_in_full_boxes ∧
    avg_fullness_non_full_boxes = parsnips_in_non_full_boxes / non_full_boxes ∧
    avg_fullness_non_full_boxes = 10 :=
by
  sorry

end parsnip_box_fullness_l284_28483


namespace find_x_for_g_equal_20_l284_28471

theorem find_x_for_g_equal_20 (g f : ℝ → ℝ) (h₁ : ∀ x, g x = 4 * (f⁻¹ x))
    (h₂ : ∀ x, f x = 30 / (x + 5)) :
    ∃ x, g x = 20 ∧ x = 3 := by
  sorry

end find_x_for_g_equal_20_l284_28471


namespace YooSeung_has_108_marbles_l284_28484

def YoungSoo_marble_count : ℕ := 21
def HanSol_marble_count : ℕ := YoungSoo_marble_count + 15
def YooSeung_marble_count : ℕ := 3 * HanSol_marble_count
def total_marble_count : ℕ := YoungSoo_marble_count + HanSol_marble_count + YooSeung_marble_count

theorem YooSeung_has_108_marbles 
  (h1 : YooSeung_marble_count = 3 * (YoungSoo_marble_count + 15))
  (h2 : HanSol_marble_count = YoungSoo_marble_count + 15)
  (h3 : total_marble_count = 165) :
  YooSeung_marble_count = 108 :=
by sorry

end YooSeung_has_108_marbles_l284_28484


namespace arithmetic_sequence_k_l284_28431

theorem arithmetic_sequence_k (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (ha : ∀ n, S (n + 1) = S n + a (n + 1))
  (hS3_S8 : S 3 = S 8) 
  (hS7_Sk : ∃ k, S 7 = S k)
  : ∃ k, k = 4 :=
by
  sorry

end arithmetic_sequence_k_l284_28431


namespace problem_l284_28416

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l284_28416


namespace conic_sections_of_equation_l284_28453

noncomputable def is_parabola (s : Set (ℝ × ℝ)) : Prop :=
∃ a b c : ℝ, ∀ x y : ℝ, (x, y) ∈ s ↔ y ≠ 0 ∧ y = a * x^3 + b * x + c

theorem conic_sections_of_equation :
  let eq := { p : ℝ × ℝ | p.2^6 - 9 * p.1^6 = 3 * p.2^3 - 1 }
  (is_parabola eq1) → (is_parabola eq2) → (eq = eq1 ∪ eq2) :=
by sorry

end conic_sections_of_equation_l284_28453


namespace karthik_weight_average_l284_28468

theorem karthik_weight_average
  (weight : ℝ)
  (hKarthik: 55 < weight )
  (hBrother: weight < 58 )
  (hFather : 56 < weight )
  (hSister: 54 < weight ∧ weight < 57) :
  (56 < weight ∧ weight < 57) → (weight = 56.5) :=
by 
  sorry

end karthik_weight_average_l284_28468


namespace leak_empties_cistern_in_24_hours_l284_28437

noncomputable def cistern_fill_rate_without_leak : ℝ := 1 / 8
noncomputable def cistern_fill_rate_with_leak : ℝ := 1 / 12

theorem leak_empties_cistern_in_24_hours :
  (1 / (cistern_fill_rate_without_leak - cistern_fill_rate_with_leak)) = 24 :=
by
  sorry

end leak_empties_cistern_in_24_hours_l284_28437


namespace max_x2y_l284_28446

noncomputable def maximum_value_x_squared_y (x y : ℝ) : ℝ :=
  if x ∈ Set.Ici 0 ∧ y ∈ Set.Ici 0 ∧ x^3 + y^3 + 3*x*y = 1 then x^2 * y else 0

theorem max_x2y (x y : ℝ) (h1 : x ∈ Set.Ici 0) (h2 : y ∈ Set.Ici 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  maximum_value_x_squared_y x y = 4 / 27 :=
sorry

end max_x2y_l284_28446


namespace probability_left_oar_works_l284_28497

structure Oars where
  P_L : ℝ -- Probability that the left oar works
  P_R : ℝ -- Probability that the right oar works
  
def independent_prob (o : Oars) : Prop :=
  o.P_L = o.P_R ∧ (1 - o.P_L) * (1 - o.P_R) = 0.16

theorem probability_left_oar_works (o : Oars) (h1 : independent_prob o) (h2 : 1 - (1 - o.P_L) * (1 - o.P_R) = 0.84) : o.P_L = 0.6 :=
by
  sorry

end probability_left_oar_works_l284_28497


namespace quadratic_equation_solution_l284_28444

-- Define the problem statement and the conditions: the equation being quadratic.
theorem quadratic_equation_solution (m : ℤ) :
  (∃ (a : ℤ), a ≠ 0 ∧ (a*x^2 - x - 2 = 0)) →
  m = -1 :=
by
  sorry

end quadratic_equation_solution_l284_28444


namespace problem1_f_x_linear_problem2_f_x_l284_28413

-- Problem 1 statement: Prove f(x) = 2x + 7 given conditions
theorem problem1_f_x_linear (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 :=
by sorry

-- Problem 2 statement: Prove f(x) = 2x - 1/x given conditions
theorem problem2_f_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * f x + f (1 / x) = 3 * x) : 
  ∀ x, f x = 2 * x - 1 / x :=
by sorry

end problem1_f_x_linear_problem2_f_x_l284_28413


namespace triangle_exists_l284_28423

theorem triangle_exists (x : ℕ) (hx : x > 0) :
  (3 * x + 10 > x * x) ∧ (x * x + 10 > 3 * x) ∧ (x * x + 3 * x > 10) ↔ (x = 3 ∨ x = 4) :=
by
  sorry

end triangle_exists_l284_28423


namespace reduced_price_is_25_l284_28457

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price (P : ℝ) := P * 0.85
noncomputable def amount_of_wheat_original (P : ℝ) := 500 / P
noncomputable def amount_of_wheat_reduced (P : ℝ) := 500 / (P * 0.85)

theorem reduced_price_is_25 : 
  ∃ (P : ℝ), reduced_price P = 25 ∧ (amount_of_wheat_reduced P = amount_of_wheat_original P + 3) :=
sorry

end reduced_price_is_25_l284_28457


namespace sports_club_total_members_l284_28443

theorem sports_club_total_members :
  ∀ (B T Both Neither Total : ℕ),
    B = 17 → T = 19 → Both = 10 → Neither = 2 → Total = B + T - Both + Neither → Total = 28 :=
by
  intros B T Both Neither Total hB hT hBoth hNeither hTotal
  rw [hB, hT, hBoth, hNeither] at hTotal
  exact hTotal

end sports_club_total_members_l284_28443


namespace total_volume_of_removed_pyramids_l284_28428

noncomputable def volume_of_removed_pyramids (edge_length : ℝ) : ℝ :=
  8 * (1 / 3 * (1 / 2 * (edge_length / 4) * (edge_length / 4)) * (edge_length / 4) / 6)

theorem total_volume_of_removed_pyramids :
  volume_of_removed_pyramids 1 = 1 / 48 :=
by
  sorry

end total_volume_of_removed_pyramids_l284_28428


namespace dimension_tolerance_l284_28435

theorem dimension_tolerance (base_dim : ℝ) (pos_tolerance : ℝ) (neg_tolerance : ℝ) 
  (max_dim : ℝ) (min_dim : ℝ) 
  (h_base : base_dim = 7) 
  (h_pos_tolerance : pos_tolerance = 0.05) 
  (h_neg_tolerance : neg_tolerance = 0.02) 
  (h_max_dim : max_dim = base_dim + pos_tolerance) 
  (h_min_dim : min_dim = base_dim - neg_tolerance) :
  max_dim = 7.05 ∧ min_dim = 6.98 :=
by
  sorry

end dimension_tolerance_l284_28435


namespace initial_action_figures_l284_28495

theorem initial_action_figures (x : ℕ) (h1 : x + 2 = 10) : x = 8 := 
by sorry

end initial_action_figures_l284_28495


namespace excircle_tangent_segment_length_l284_28412

theorem excircle_tangent_segment_length (A B C M : ℝ) 
  (h1 : A + B + C = 1) 
  (h2 : M = (1 / 2)) : 
  M = 1 / 2 := 
  by
    -- This is where the proof would go
    sorry

end excircle_tangent_segment_length_l284_28412


namespace find_m_and_equation_of_l2_l284_28454

theorem find_m_and_equation_of_l2 (a : ℝ) (M: ℝ × ℝ) (m : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hM : M = (-5, 1)) 
  (hl1 : ∀ {x y : ℝ}, 2 * x - y + 2 = 0) 
  (hl : ∀ {x y : ℝ}, x + y + m = 0) 
  (hl2 : ∀ {x y : ℝ}, (∃ p : ℝ × ℝ, p = M → x - 2 * y + 7 = 0)) : 
  m = -5 ∧ ∀ {x y : ℝ}, x - 2 * y + 7 = 0 :=
by
  sorry

end find_m_and_equation_of_l2_l284_28454


namespace net_effect_transactions_l284_28424

theorem net_effect_transactions {a o : ℝ} (h1 : 3 * a / 4 = 15000) (h2 : 5 * o / 4 = 15000) :
  a + o - (2 * 15000) = 2000 :=
by
  sorry

end net_effect_transactions_l284_28424


namespace arc_length_sector_l284_28478

theorem arc_length_sector (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 3) : 
  α * r = 2 * π / 3 := 
by 
  sorry

end arc_length_sector_l284_28478


namespace parabola_directrix_distance_l284_28455

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end parabola_directrix_distance_l284_28455


namespace rectangle_area_l284_28496

theorem rectangle_area (r l b : ℝ) (h1: r = 30) (h2: l = (2 / 5) * r) (h3: b = 10) : 
  l * b = 120 := 
by
  sorry

end rectangle_area_l284_28496


namespace rickshaw_distance_l284_28460

theorem rickshaw_distance (km1_charge : ℝ) (rate_per_km : ℝ) (total_km : ℝ) (total_charge : ℝ) :
  km1_charge = 13.50 → rate_per_km = 2.50 → total_km = 13 → total_charge = 103.5 → (total_charge - km1_charge) / rate_per_km = 36 :=
by
  intro h1 h2 h3 h4
  -- We would fill in proof steps here, but skipping as required.
  sorry

end rickshaw_distance_l284_28460


namespace maxvalue_on_ellipse_l284_28411

open Real

noncomputable def max_x_plus_y : ℝ := 343 / 88

theorem maxvalue_on_ellipse (x y : ℝ) :
  (x^2 + 3 * x * y + 2 * y^2 - 14 * x - 21 * y + 49 = 0) →
  x + y ≤ max_x_plus_y := 
sorry

end maxvalue_on_ellipse_l284_28411


namespace smallest_ab_41503_539_l284_28470

noncomputable def find_smallest_ab : (ℕ × ℕ) :=
  let a := 41503
  let b := 539
  (a, b)

theorem smallest_ab_41503_539 (a b : ℕ) (h : 7 * a^3 = 11 * b^5) (ha : a > 0) (hb : b > 0) :
  (a = 41503 ∧ b = 539) :=
  by
    -- Add sorry to skip the proof
    sorry

end smallest_ab_41503_539_l284_28470


namespace side_length_of_square_l284_28493

-- Mathematical definitions and conditions
def square_area (side : ℕ) : ℕ := side * side

theorem side_length_of_square {s : ℕ} (h : square_area s = 289) : s = 17 :=
sorry

end side_length_of_square_l284_28493


namespace turtle_speed_l284_28498

theorem turtle_speed
  (hare_speed : ℝ)
  (race_distance : ℝ)
  (head_start : ℝ) :
  hare_speed = 10 → race_distance = 20 → head_start = 18 → 
  (race_distance / (head_start + race_distance / hare_speed) = 1) :=
by
  intros
  sorry

end turtle_speed_l284_28498


namespace rectangular_garden_width_l284_28436

variable (w : ℕ)

/-- The length of a rectangular garden is three times its width.
Given that the area of the rectangular garden is 768 square meters,
prove that the width of the garden is 16 meters. -/
theorem rectangular_garden_width
  (h1 : 768 = w * (3 * w)) :
  w = 16 := by
  sorry

end rectangular_garden_width_l284_28436


namespace diff_of_squares_535_465_l284_28406

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l284_28406


namespace seq_a_eval_a4_l284_28467

theorem seq_a_eval_a4 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) : a 4 = 15 :=
sorry

end seq_a_eval_a4_l284_28467


namespace total_games_equal_684_l284_28445

-- Define the number of players
def n : Nat := 19

-- Define the formula to calculate the total number of games played
def total_games (n : Nat) : Nat := n * (n - 1) * 2

-- The proposition asserting the total number of games equals 684
theorem total_games_equal_684 : total_games n = 684 :=
by
  sorry

end total_games_equal_684_l284_28445


namespace inverse_of_square_positive_is_negative_l284_28441

variable {x : ℝ}

-- Original proposition: ∀ x, x < 0 → x^2 > 0
def original_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 > 0

-- Inverse proposition to be proven: ∀ x, x^2 > 0 → x < 0
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

theorem inverse_of_square_positive_is_negative :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ x : ℝ, x^2 > 0 → x < 0) :=
  sorry

end inverse_of_square_positive_is_negative_l284_28441


namespace cost_of_dvd_player_l284_28464

/-- The ratio of the cost of a DVD player to the cost of a movie is 9:2.
    A DVD player costs $63 more than a movie.
    Prove that the cost of the DVD player is $81. -/
theorem cost_of_dvd_player 
(D M : ℝ)
(h1 : D = (9 / 2) * M)
(h2 : D = M + 63) : 
D = 81 := 
sorry

end cost_of_dvd_player_l284_28464


namespace average_price_of_fruit_l284_28462

theorem average_price_of_fruit :
  ∃ (A O : ℕ), A + O = 10 ∧ (40 * A + 60 * (O - 4)) / (A + O - 4) = 50 → 
  (40 * A + 60 * O) / 10 = 54 :=
by
  sorry

end average_price_of_fruit_l284_28462


namespace meat_division_l284_28452

theorem meat_division (w1 w2 meat : ℕ) (h1 : w1 = 645) (h2 : w2 = 237) (h3 : meat = 1000) :
  ∃ (m1 m2 : ℕ), m1 = 296 ∧ m2 = 704 ∧ w1 + m1 = w2 + m2 := by
  sorry

end meat_division_l284_28452


namespace brick_height_l284_28442

theorem brick_height (H : ℝ) 
    (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
    (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℝ)
    (volume_wall: wall_length = 900 ∧ wall_width = 500 ∧ wall_height = 1850)
    (volume_brick: brick_length = 21 ∧ brick_width = 10)
    (num_bricks_value: num_bricks = 4955.357142857142) :
    (H = 0.8) :=
by {
  sorry
}

end brick_height_l284_28442


namespace point_outside_circle_l284_28427

theorem point_outside_circle (a b : ℝ) (h_intersect : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a*x + b*y = 1) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l284_28427


namespace divisibility_by_5_l284_28466

theorem divisibility_by_5 (B : ℕ) (hB : B < 10) : (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := 
by
  sorry

end divisibility_by_5_l284_28466


namespace total_routes_A_to_B_l284_28425

-- Define the conditions
def routes_A_to_C : ℕ := 4
def routes_C_to_B : ℕ := 2

-- Statement to prove
theorem total_routes_A_to_B : (routes_A_to_C * routes_C_to_B = 8) :=
by
  -- Omitting the proof, but stating that there is a total of 8 routes from A to B
  sorry

end total_routes_A_to_B_l284_28425


namespace mina_numbers_l284_28472

theorem mina_numbers (a b : ℤ) (h1 : 3 * a + 4 * b = 140) (h2 : a = 20 ∨ b = 20) : a = 20 ∧ b = 20 :=
by
  sorry

end mina_numbers_l284_28472


namespace unique_positive_integers_abc_l284_28448

def coprime (a b : ℕ) := Nat.gcd a b = 1

def allPrimeDivisorsNotCongruentTo1Mod7 (n : ℕ) := 
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p % 7 ≠ 1

theorem unique_positive_integers_abc :
  ∀ a b c : ℕ,
    (1 ≤ a) →
    (1 ≤ b) →
    (1 ≤ c) →
    coprime a b →
    coprime b c →
    coprime c a →
    (a * a + b) ∣ (b * b + c) →
    (b * b + c) ∣ (c * c + a) →
    allPrimeDivisorsNotCongruentTo1Mod7 (a * a + b) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end unique_positive_integers_abc_l284_28448


namespace arithmetic_sequence_m_l284_28426

theorem arithmetic_sequence_m (m : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n = 2 * n - 1) →
  (∀ n, S n = n * (2 * n - 1) / 2) →
  S m = (a m + a (m + 1)) / 2 →
  m = 2 :=
by
  sorry

end arithmetic_sequence_m_l284_28426


namespace find_original_expression_l284_28463

theorem find_original_expression (a b c X : ℤ) :
  X + (a * b - 2 * b * c + 3 * a * c) = 2 * b * c - 3 * a * c + 2 * a * b →
  X = 4 * b * c - 6 * a * c + a * b :=
by
  sorry

end find_original_expression_l284_28463


namespace balanced_scale_l284_28477

def children's_book_weight : ℝ := 1.1

def weight1 : ℝ := 0.5
def weight2 : ℝ := 0.3
def weight3 : ℝ := 0.3

theorem balanced_scale :
  (weight1 + weight2 + weight3) = children's_book_weight :=
by
  sorry

end balanced_scale_l284_28477


namespace same_terminal_side_l284_28481

theorem same_terminal_side (k : ℤ): ∃ k : ℤ, 1303 = k * 360 - 137 := by
  -- Proof left as an exercise.
  sorry

end same_terminal_side_l284_28481


namespace initial_overs_l284_28488

theorem initial_overs (initial_run_rate remaining_run_rate target runs initially remaining_overs : ℝ)
    (h_target : target = 282)
    (h_remaining_overs : remaining_overs = 40)
    (h_initial_run_rate : initial_run_rate = 3.6)
    (h_remaining_run_rate : remaining_run_rate = 6.15)
    (h_target_eq : initial_run_rate * initially + remaining_run_rate * remaining_overs = target) :
    initially = 10 :=
by
  sorry

end initial_overs_l284_28488


namespace sixth_power_sum_l284_28449

/-- Given:
     (1) a + b = 1
     (2) a^2 + b^2 = 3
     (3) a^3 + b^3 = 4
     (4) a^4 + b^4 = 7
     (5) a^5 + b^5 = 11
    Prove:
     a^6 + b^6 = 18 -/
theorem sixth_power_sum (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sixth_power_sum_l284_28449


namespace number_of_3_letter_words_with_at_least_one_A_l284_28447

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end number_of_3_letter_words_with_at_least_one_A_l284_28447


namespace no_integer_roots_l284_28465
open Polynomial

theorem no_integer_roots (p : Polynomial ℤ) (c1 c2 c3 : ℤ) (h1 : p.eval c1 = 1) (h2 : p.eval c2 = 1) (h3 : p.eval c3 = 1) (h_distinct : c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3) : ¬ ∃ a : ℤ, p.eval a = 0 :=
by
  sorry

end no_integer_roots_l284_28465


namespace pizza_fraction_eaten_l284_28487

theorem pizza_fraction_eaten :
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  (a * (1 - r ^ n) / (1 - r)) = 63 / 128 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  sorry

end pizza_fraction_eaten_l284_28487


namespace stamps_difference_l284_28410

theorem stamps_difference (x : ℕ) (h1: 5 * x / 3 * x = 5 / 3)
(h2: (5 * x - 12) / (3 * x + 12) = 4 / 3) : 
(5 * x - 12) - (3 * x + 12) = 32 := by
sorry

end stamps_difference_l284_28410


namespace boat_speed_in_still_water_l284_28473

theorem boat_speed_in_still_water (x y : ℝ) :
  (80 / (x + y) + 48 / (x - y) = 9) ∧ 
  (64 / (x + y) + 96 / (x - y) = 12) → 
  x = 12 :=
by
  sorry

end boat_speed_in_still_water_l284_28473


namespace equal_books_for_students_l284_28482

-- Define the conditions
def num_girls : ℕ := 15
def num_boys : ℕ := 10
def total_books : ℕ := 375
def books_for_girls : ℕ := 225
def books_for_boys : ℕ := total_books - books_for_girls -- Calculate books for boys

-- Define the theorem
theorem equal_books_for_students :
  books_for_girls / num_girls = 15 ∧ books_for_boys / num_boys = 15 :=
by
  sorry

end equal_books_for_students_l284_28482


namespace find_a_2016_l284_28432

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (n + 1) / n * a n

theorem find_a_2016 (a : ℕ → ℝ) (h : seq a) : a 2016 = 4032 :=
by
  sorry

end find_a_2016_l284_28432


namespace isosceles_triangle_sides_l284_28459

theorem isosceles_triangle_sides (a b c : ℕ) (h₁ : a + b + c = 10) (h₂ : (a = b ∨ b = c ∨ a = c)) 
  (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) : 
  (a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 4) := 
by
  sorry

end isosceles_triangle_sides_l284_28459


namespace photograph_goal_reach_l284_28476

-- Define the initial number of photographs
def initial_photos : ℕ := 250

-- Define the percentage splits initially
def beth_pct_init : ℝ := 0.40
def my_pct_init : ℝ := 0.35
def julia_pct_init : ℝ := 0.25

-- Define the photographs taken initially by each person
def beth_photos_init : ℕ := 100
def my_photos_init : ℕ := 88
def julia_photos_init : ℕ := 63

-- Confirm initial photographs sum
example (h : beth_photos_init + my_photos_init + julia_photos_init = 251) : true := 
by trivial

-- Define today's decreased productivity percentages
def beth_decrease_pct : ℝ := 0.35
def my_decrease_pct : ℝ := 0.45
def julia_decrease_pct : ℝ := 0.25

-- Define the photographs taken today by each person after decreases
def beth_photos_today : ℕ := 65
def my_photos_today : ℕ := 48
def julia_photos_today : ℕ := 47

-- Sum of photographs taken today
def total_photos_today : ℕ := 160

-- Define the initial plus today's needed photographs to reach goal
def goal_photos : ℕ := 650

-- Define the additional number of photographs needed
def additional_photos_needed : ℕ := 399 - total_photos_today

-- Final proof statement
theorem photograph_goal_reach : 
  (beth_photos_init + my_photos_init + julia_photos_init) + (beth_photos_today + my_photos_today + julia_photos_today) + additional_photos_needed = goal_photos := 
by sorry

end photograph_goal_reach_l284_28476


namespace smallest_whole_number_larger_than_triangle_perimeter_l284_28439

theorem smallest_whole_number_larger_than_triangle_perimeter
  (s : ℝ) (h1 : 5 + 19 > s) (h2 : 5 + s > 19) (h3 : 19 + s > 5) :
  ∃ P : ℝ, P = 5 + 19 + s ∧ P < 48 ∧ ∀ n : ℤ, n > P → n = 48 :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l284_28439


namespace jenny_sold_192_packs_l284_28408

-- Define the conditions
def boxes_sold : ℝ := 24.0
def packs_per_box : ℝ := 8.0

-- The total number of packs sold
def total_packs_sold : ℝ := boxes_sold * packs_per_box

-- Proof statement that total packs sold equals 192.0
theorem jenny_sold_192_packs : total_packs_sold = 192.0 :=
by
  sorry

end jenny_sold_192_packs_l284_28408


namespace units_digit_7_pow_75_plus_6_l284_28430

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_75_plus_6 : units_digit (7 ^ 75 + 6) = 9 := 
by
  sorry

end units_digit_7_pow_75_plus_6_l284_28430


namespace hoseok_has_least_papers_l284_28414

-- Definitions based on the conditions
def pieces_jungkook : ℕ := 10
def pieces_hoseok : ℕ := 7
def pieces_seokjin : ℕ := pieces_jungkook - 2

-- Theorem stating Hoseok has the least pieces of colored paper
theorem hoseok_has_least_papers : pieces_hoseok < pieces_jungkook ∧ pieces_hoseok < pieces_seokjin := by 
  sorry

end hoseok_has_least_papers_l284_28414


namespace inequality_solution_l284_28499

theorem inequality_solution :
  {x : ℝ | (3 * x - 9) * (x - 4) / (x - 1) ≥ 0} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 3} ∪ {x : ℝ | x ≥ 4} :=
by
  sorry

end inequality_solution_l284_28499


namespace units_digit_7_pow_2023_l284_28490

theorem units_digit_7_pow_2023 :
  (7 ^ 2023) % 10 = 3 := 
sorry

end units_digit_7_pow_2023_l284_28490


namespace intersection_points_product_l284_28469

theorem intersection_points_product (x y : ℝ) :
  (x^2 - 2 * x + y^2 - 6 * y + 9 = 0) ∧ (x^2 - 8 * x + y^2 - 6 * y + 28 = 0) → x * y = 6 :=
by
  sorry

end intersection_points_product_l284_28469


namespace a_share_calculation_l284_28438

noncomputable def investment_a : ℕ := 15000
noncomputable def investment_b : ℕ := 21000
noncomputable def investment_c : ℕ := 27000
noncomputable def total_investment : ℕ := investment_a + investment_b + investment_c -- 63000
noncomputable def b_share : ℕ := 1540
noncomputable def total_profit : ℕ := 4620  -- from the solution steps

theorem a_share_calculation :
  (investment_a * total_profit) / total_investment = 1100 := 
by
  sorry

end a_share_calculation_l284_28438


namespace smallest_whole_number_for_inequality_l284_28486

theorem smallest_whole_number_for_inequality:
  ∃ (x : ℕ), (2 : ℝ) / 5 + (x : ℝ) / 9 > 1 ∧ ∀ (y : ℕ), (2 : ℝ) / 5 + (y : ℝ) / 9 > 1 → x ≤ y :=
by
  sorry

end smallest_whole_number_for_inequality_l284_28486


namespace distance_between_homes_l284_28480

-- Define the parameters
def maxwell_speed : ℝ := 4  -- km/h
def brad_speed : ℝ := 6     -- km/h
def maxwell_time_to_meet : ℝ := 2  -- hours
def brad_start_delay : ℝ := 1  -- hours

-- Definitions related to the timings
def brad_time_to_meet : ℝ := maxwell_time_to_meet - brad_start_delay  -- hours

-- Define the distances covered by each
def maxwell_distance : ℝ := maxwell_speed * maxwell_time_to_meet  -- km
def brad_distance : ℝ := brad_speed * brad_time_to_meet  -- km

-- Define the total distance between their homes
def total_distance : ℝ := maxwell_distance + brad_distance  -- km

-- Statement to prove
theorem distance_between_homes : total_distance = 14 :=
by
  -- The proof is omitted; add 'sorry' to indicate this.
  sorry

end distance_between_homes_l284_28480


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l284_28492

-- Define a predicate for consecutive prime numbers
def is_consecutive_primes (a b c d : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
  (b = a + 1 ∨ b = a + 2) ∧
  (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2)

-- Define the main problem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (a b c d : ℕ), is_consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ ∀ (w x y z : ℕ), is_consecutive_primes w x y z ∧ (w + x + y + z) % 5 = 0 → a + b + c + d ≤ w + x + y + z :=
sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l284_28492


namespace females_with_advanced_degrees_eq_90_l284_28409

-- define the given constants
def total_employees : ℕ := 360
def total_females : ℕ := 220
def total_males : ℕ := 140
def advanced_degrees : ℕ := 140
def college_degrees : ℕ := 160
def vocational_training : ℕ := 60
def males_with_college_only : ℕ := 55
def females_with_vocational_training : ℕ := 25

-- define the main theorem to prove the number of females with advanced degrees
theorem females_with_advanced_degrees_eq_90 :
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 90 :=
by
  sorry

end females_with_advanced_degrees_eq_90_l284_28409


namespace minimum_common_perimeter_l284_28422

noncomputable def is_integer (x: ℝ) : Prop := ∃ (n: ℤ), x = n

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ is_triangle a b c

theorem minimum_common_perimeter :
  ∃ (a b : ℝ),
    is_integer a ∧ is_integer b ∧
    4 * a = 5 * b - 18 ∧
    is_isosceles_triangle a a (2 * a - 12) ∧
    is_isosceles_triangle b b (3 * b - 30) ∧
    (2 * a + (2 * a - 12) = 2 * b + (3 * b - 30)) ∧
    (2 * a + (2 * a - 12) = 228) := sorry

end minimum_common_perimeter_l284_28422


namespace symmetry_center_2tan_2x_sub_pi_div_4_l284_28440

theorem symmetry_center_2tan_2x_sub_pi_div_4 (k : ℤ) :
  ∃ (x : ℝ), 2 * (x) - π / 4 = k * π / 2 ∧ x = k * π / 4 + π / 8 :=
by
  sorry

end symmetry_center_2tan_2x_sub_pi_div_4_l284_28440


namespace white_pawn_on_white_square_l284_28418

theorem white_pawn_on_white_square (w b N_b N_w : ℕ) (h1 : w > b) (h2 : N_b < N_w) : ∃ k : ℕ, k > 0 :=
by 
  -- Let's assume a contradiction
  -- The proof steps would be written here
  sorry

end white_pawn_on_white_square_l284_28418


namespace missing_digit_divisibility_by_nine_l284_28402

theorem missing_digit_divisibility_by_nine (x : ℕ) (h : 0 ≤ x ∧ x < 10) :
  9 ∣ (3 + 5 + 2 + 4 + x) → x = 4 :=
by
  sorry

end missing_digit_divisibility_by_nine_l284_28402


namespace ab_range_l284_28479

theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a * b = a + b) : 1 / 4 ≤ a * b :=
sorry

end ab_range_l284_28479


namespace greatest_candies_to_office_l284_28494

-- Problem statement: Prove that the greatest possible number of candies given to the office is 7 when distributing candies among 8 students.

theorem greatest_candies_to_office (n : ℕ) : 
  ∃ k : ℕ, k = n % 8 ∧ k ≤ 7 ∧ k = 7 :=
by
  sorry

end greatest_candies_to_office_l284_28494


namespace new_person_weight_l284_28450

theorem new_person_weight (average_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ) (new_weight : ℝ) 
  (h1 : num_persons = 10) 
  (h2 : average_increase = 3.2) 
  (h3 : replaced_weight = 65) : 
  new_weight = 97 :=
by
  sorry

end new_person_weight_l284_28450


namespace employee_salary_proof_l284_28420

variable (x : ℝ) (M : ℝ) (P : ℝ)

theorem employee_salary_proof (h1 : x + 1.2 * x + 1.8 * x = 1500)
(h2 : M = 1.2 * x)
(h3 : P = 1.8 * x)
: x = 375 ∧ M = 450 ∧ P = 675 :=
sorry

end employee_salary_proof_l284_28420


namespace ratio_of_Steve_speeds_l284_28475

noncomputable def Steve_speeds_ratio : Nat := 
  let d := 40 -- distance in km
  let T := 6  -- total time in hours
  let v2 := 20 -- speed on the way back in km/h
  let t2 := d / v2 -- time taken on the way back in hours
  let t1 := T - t2 -- time taken on the way to work in hours
  let v1 := d / t1 -- speed on the way to work in km/h
  v2 / v1

theorem ratio_of_Steve_speeds :
  Steve_speeds_ratio = 2 := 
  by sorry

end ratio_of_Steve_speeds_l284_28475


namespace gardener_area_l284_28434

-- The definition considers the placement of gardeners and the condition for attending flowers.
noncomputable def grid_assignment (gardener_position: (ℕ × ℕ)) (flower_position: (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry

-- A theorem that states the equivalent proof.
theorem gardener_area (gardener_position: (ℕ × ℕ)) :
  ∀ flower_position: (ℕ × ℕ), (∃ g1 g2 g3, g1 ∈ grid_assignment gardener_position flower_position ∧
                                            g2 ∈ grid_assignment gardener_position flower_position ∧
                                            g3 ∈ grid_assignment gardener_position flower_position) →
  (gardener_position = g1 ∨ gardener_position = g2 ∨ gardener_position = g3) → true :=
by
  sorry

end gardener_area_l284_28434


namespace area_of_triangle_PQS_l284_28458

-- Define a structure to capture the conditions of the trapezoid and its properties.
structure Trapezoid (P Q R S : Type) :=
(area : ℝ)
(PQ : ℝ)
(RS : ℝ)
(area_PQS : ℝ)
(condition1 : area = 18)
(condition2 : RS = 3 * PQ)

-- Here's the theorem we want to prove, stating the conclusion based on the given conditions.
theorem area_of_triangle_PQS {P Q R S : Type} (T : Trapezoid P Q R S) : T.area_PQS = 4.5 :=
by
  -- Proof will go here, but for now we use sorry.
  sorry

end area_of_triangle_PQS_l284_28458


namespace min_positive_numbers_l284_28401

theorem min_positive_numbers (n : ℕ) (numbers : ℕ → ℤ) 
  (h_length : n = 103) 
  (h_consecutive : ∀ i : ℕ, i < n → (∃ (p1 p2 : ℕ), p1 < 5 ∧ p2 < 5 ∧ p1 ≠ p2 ∧ numbers (i + p1) > 0 ∧ numbers (i + p2) > 0)) :
  ∃ (min_positive : ℕ), min_positive = 42 :=
by
  sorry

end min_positive_numbers_l284_28401


namespace value_of_a_l284_28417

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  sorry

end value_of_a_l284_28417


namespace box_volume_max_l284_28400

noncomputable def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem box_volume_max (a : ℝ) (h : 0 < a) :
  ∃ x, 0 < x ∧ x < a / 2 ∧ volume a x = volume a (a / 6) ∧ volume a (a / 6) = (2 * a^3) / 27 :=
by
  sorry

end box_volume_max_l284_28400


namespace union_complement_eq_l284_28489

open Set

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement (what we want to prove)
theorem union_complement_eq :
  A ∪ compl B = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_complement_eq_l284_28489
