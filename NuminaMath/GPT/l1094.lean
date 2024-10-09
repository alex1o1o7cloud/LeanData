import Mathlib

namespace piravena_trip_total_cost_l1094_109437

-- Define the distances
def d_A_to_B : ℕ := 4000
def d_B_to_C : ℕ := 3000

-- Define the costs per kilometer
def bus_cost_per_km : ℝ := 0.15
def airplane_cost_per_km : ℝ := 0.12
def airplane_booking_fee : ℝ := 120

-- Define the individual costs and the total cost
def cost_A_to_B : ℝ := d_A_to_B * airplane_cost_per_km + airplane_booking_fee
def cost_B_to_C : ℝ := d_B_to_C * bus_cost_per_km
def total_cost : ℝ := cost_A_to_B + cost_B_to_C

-- Define the theorem we want to prove
theorem piravena_trip_total_cost :
  total_cost = 1050 := sorry

end piravena_trip_total_cost_l1094_109437


namespace nonnegative_integer_pairs_solution_l1094_109438

open Int

theorem nonnegative_integer_pairs_solution (x y : ℕ) : 
  3 * x ^ 2 + 2 * 9 ^ y = x * (4 ^ (y + 1) - 1) ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) :=
by 
  sorry

end nonnegative_integer_pairs_solution_l1094_109438


namespace doubled_team_completes_half_in_three_days_l1094_109453

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days_l1094_109453


namespace projection_of_a_in_direction_of_b_l1094_109459

noncomputable def vector_projection_in_direction (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_in_direction_of_b :
  vector_projection_in_direction (3, 2) (-2, 1) = -4 * Real.sqrt 5 / 5 := 
by
  sorry

end projection_of_a_in_direction_of_b_l1094_109459


namespace equal_area_condition_l1094_109489

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2)

theorem equal_area_condition : 2 * θ = (Real.tan θ) * (Real.tan (2 * θ)) :=
by {
  sorry
}

end equal_area_condition_l1094_109489


namespace equation_solution_unique_l1094_109408

theorem equation_solution_unique (x y : ℤ) : 
  x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end equation_solution_unique_l1094_109408


namespace anna_reading_time_l1094_109469

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l1094_109469


namespace distance_between_trees_l1094_109468

theorem distance_between_trees (L : ℝ) (n : ℕ) (hL : L = 375) (hn : n = 26) : 
  (L / (n - 1) = 15) :=
by
  sorry

end distance_between_trees_l1094_109468


namespace pencil_price_in_units_l1094_109434

noncomputable def price_of_pencil_in_units (base_price additional_price unit_size : ℕ) : ℝ :=
  (base_price + additional_price) / unit_size

theorem pencil_price_in_units :
  price_of_pencil_in_units 5000 200 10000 = 0.52 := 
  by 
  sorry

end pencil_price_in_units_l1094_109434


namespace range_of_m_l1094_109433

open Set Real

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B (m : ℝ) := {x : ℝ | -1 < x ∧ x < m}

theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → 3 < m :=
by sorry

end range_of_m_l1094_109433


namespace circle_area_l1094_109465

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (π * ((1 / 2) * (1 / 2)) = (π / 4)) := 
by
  intro h
  sorry

end circle_area_l1094_109465


namespace A_salary_less_than_B_by_20_percent_l1094_109460

theorem A_salary_less_than_B_by_20_percent (A B : ℝ) (h1 : B = 1.25 * A) : 
  (B - A) / B * 100 = 20 :=
by
  sorry

end A_salary_less_than_B_by_20_percent_l1094_109460


namespace total_cost_is_135_25_l1094_109443

-- defining costs and quantities
def cost_A : ℕ := 9
def num_A : ℕ := 4
def cost_B := cost_A + 5
def num_B : ℕ := 2
def cost_clay_pot := cost_A + 20
def cost_bag_soil := cost_A - 2
def cost_fertilizer := cost_A + (cost_A / 2)
def cost_gardening_tools := cost_clay_pot - (cost_clay_pot / 4)

-- total cost calculation
def total_cost : ℚ :=
  (num_A * cost_A) + 
  (num_B * cost_B) + 
  cost_clay_pot + 
  cost_bag_soil + 
  cost_fertilizer + 
  cost_gardening_tools

theorem total_cost_is_135_25 : total_cost = 135.25 := by
  sorry

end total_cost_is_135_25_l1094_109443


namespace prob_no_decrease_white_in_A_is_correct_l1094_109409

-- Define the conditions of the problem
def bagA_white : ℕ := 3
def bagA_black : ℕ := 5
def bagB_white : ℕ := 4
def bagB_black : ℕ := 6

-- Define the probabilities involved
def prob_draw_black_from_A : ℚ := 5 / 8
def prob_draw_white_from_A : ℚ := 3 / 8
def prob_put_white_back_into_A_conditioned_on_white_drawn : ℚ := 5 / 11

-- Calculate the combined probability
def prob_no_decrease_white_in_A : ℚ := prob_draw_black_from_A + prob_draw_white_from_A * prob_put_white_back_into_A_conditioned_on_white_drawn

-- Prove the probability is as expected
theorem prob_no_decrease_white_in_A_is_correct : prob_no_decrease_white_in_A = 35 / 44 := by
  sorry

end prob_no_decrease_white_in_A_is_correct_l1094_109409


namespace john_score_l1094_109487

theorem john_score (s1 s2 s3 s4 s5 s6 : ℕ) (h1 : s1 = 85) (h2 : s2 = 88) (h3 : s3 = 90) (h4 : s4 = 92) (h5 : s5 = 83) (h6 : s6 = 102) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 90 :=
by
  sorry

end john_score_l1094_109487


namespace find_x_values_l1094_109461

theorem find_x_values (x : ℝ) :
  x^3 - 9 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
by
  sorry

end find_x_values_l1094_109461


namespace jacob_writing_speed_ratio_l1094_109423

theorem jacob_writing_speed_ratio (N : ℕ) (J : ℕ) (hN : N = 25) (h1 : J + N = 75) : J / N = 2 :=
by {
  sorry
}

end jacob_writing_speed_ratio_l1094_109423


namespace EG_perpendicular_to_AC_l1094_109488

noncomputable def rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 < B.1 ∧ A.2 = B.2 ∧ B.1 < C.1 ∧ B.2 < C.2 ∧ C.1 = D.1 ∧ C.2 > D.2 ∧ D.1 > A.1 ∧ D.2 = A.2

theorem EG_perpendicular_to_AC
  {A B C D E F G: ℝ × ℝ}
  (h1: rectangle A B C D)
  (h2: E = (B.1, C.2) ∨ E = (C.1, B.2)) -- Assuming E lies on BC or BA
  (h3: F = (B.1, A.2) ∨ F = (A.1, B.2)) -- Assuming F lies on BA or BC
  (h4: G = (C.1, D.2) ∨ G = (D.1, C.2)) -- Assuming G lies on CD
  (h5: (F.1, G.2) = (A.1, C.2)) -- Line through F parallel to AC meets CD at G
: ∃ (H : ℝ × ℝ → ℝ × ℝ → ℝ), H E G = 0 := sorry

end EG_perpendicular_to_AC_l1094_109488


namespace Zlatoust_to_Miass_distance_l1094_109421

theorem Zlatoust_to_Miass_distance
  (x g k m : ℝ)
  (H1 : (x + 18) / k = (x - 18) / m)
  (H2 : (x + 25) / k = (x - 25) / g)
  (H3 : (x + 8) / m = (x - 8) / g) :
  x = 60 :=
sorry

end Zlatoust_to_Miass_distance_l1094_109421


namespace math_ineq_problem_l1094_109464

variable (a b c : ℝ)

theorem math_ineq_problem
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : a + b + c ≤ 1)
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 :=
by
  sorry

end math_ineq_problem_l1094_109464


namespace highland_park_science_fair_l1094_109450

noncomputable def juniors_and_seniors_participants (j s : ℕ) : ℕ :=
  (3 * j) / 4 + s / 2

theorem highland_park_science_fair 
  (j s : ℕ)
  (h1 : (3 * j) / 4 = s / 2)
  (h2 : j + s = 240) :
  juniors_and_seniors_participants j s = 144 := by
  sorry

end highland_park_science_fair_l1094_109450


namespace min_questions_to_determine_number_l1094_109415

theorem min_questions_to_determine_number : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 50) → 
  ∃ (q : ℕ), q = 15 ∧ 
  ∀ (primes : ℕ → Prop), 
  (∀ p, primes p → Nat.Prime p ∧ p ≤ 50) → 
  (∀ p, primes p → (n % p = 0 ↔ p ∣ n)) → 
  (∃ m, (∀ k, k < m → primes k → k ∣ n)) :=
sorry

end min_questions_to_determine_number_l1094_109415


namespace dealership_vans_expected_l1094_109444

theorem dealership_vans_expected (trucks vans : ℕ) (h_ratio : 3 * vans = 5 * trucks) (h_trucks : trucks = 45) : vans = 75 :=
by
  sorry

end dealership_vans_expected_l1094_109444


namespace distance_between_Sasha_and_Koyla_is_19m_l1094_109407

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l1094_109407


namespace remainder_eq_52_l1094_109448

noncomputable def polynomial : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 7

theorem remainder_eq_52 : Polynomial.eval (-3) polynomial = 52 :=
by
    sorry

end remainder_eq_52_l1094_109448


namespace max_horizontal_segment_length_l1094_109430

theorem max_horizontal_segment_length (y : ℝ → ℝ) (h : ∀ x, y x = x^3 - x) :
  ∃ a, (∀ x₁, y x₁ = y (x₁ + a)) ∧ a = 2 :=
by
  sorry

end max_horizontal_segment_length_l1094_109430


namespace tan_alpha_eq_two_imp_inv_sin_double_angle_l1094_109466

theorem tan_alpha_eq_two_imp_inv_sin_double_angle (α : ℝ) (h : Real.tan α = 2) : 
  (1 / Real.sin (2 * α)) = 5 / 4 :=
by
  sorry

end tan_alpha_eq_two_imp_inv_sin_double_angle_l1094_109466


namespace maximum_sum_of_squares_l1094_109478

theorem maximum_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 20 :=
sorry

end maximum_sum_of_squares_l1094_109478


namespace smallest_prime_dividing_sum_l1094_109472

theorem smallest_prime_dividing_sum :
  ∃ p : ℕ, Prime p ∧ p ∣ (7^14 + 11^15) ∧ ∀ q : ℕ, Prime q ∧ q ∣ (7^14 + 11^15) → p ≤ q := by
  sorry

end smallest_prime_dividing_sum_l1094_109472


namespace moving_circle_passes_through_fixed_point_l1094_109475

-- Define the parabola x^2 = 12y
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Define the directrix line y = -3
def directrix (y : ℝ) : Prop := y = -3

-- The fixed point we need to show the circle always passes through
def fixed_point : ℝ × ℝ := (0, 3)

-- Define the condition that the moving circle is centered on the parabola and tangent to the directrix
def circle_centered_on_parabola_and_tangent_to_directrix (x y : ℝ) (r : ℝ) : Prop :=
  parabola x y ∧ r = abs (y + 3)

-- Main theorem statement
theorem moving_circle_passes_through_fixed_point :
  (∀ (x y r : ℝ), circle_centered_on_parabola_and_tangent_to_directrix x y r → 
    (∃ (px py : ℝ), (px, py) = fixed_point ∧ (px - x)^2 + (py - y)^2 = r^2)) :=
sorry

end moving_circle_passes_through_fixed_point_l1094_109475


namespace one_belt_one_road_l1094_109420

theorem one_belt_one_road (m n : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + n ↔ (x, y) ∈ { p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 1 }) →
  (∀ x y : ℝ, y = m * x + 1 ↔ (x, y) ∈ { q : ℝ × ℝ | q.1 = 0 ∧ q.2 = 1 }) →
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = 0) →
  m = -1 ∧ n = 1 :=
by
  intros h1 h2 h3
  sorry

end one_belt_one_road_l1094_109420


namespace common_ratio_of_geometric_sequence_l1094_109426

-- Define positive geometric sequence a_n with common ratio q
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

-- Define the relevant conditions
variable {a q : ℝ}
variable (h1 : a * q^4 + 2 * a * q^2 * q^6 + a * q^4 * q^8 = 16)
variable (h2 : (a * q^4 + a * q^8) / 2 = 4)
variable (pos_q : q > 0)

-- Define the goal: proving the common ratio q is sqrt(2)
theorem common_ratio_of_geometric_sequence : q = Real.sqrt 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1094_109426


namespace johns_number_is_thirteen_l1094_109467

theorem johns_number_is_thirteen (x : ℕ) (h1 : 10 ≤ x) (h2 : x < 100) (h3 : ∃ a b : ℕ, 10 * a + b = 4 * x + 17 ∧ 92 ≤ 10 * b + a ∧ 10 * b + a ≤ 96) : x = 13 :=
sorry

end johns_number_is_thirteen_l1094_109467


namespace problem_part1_problem_part2_l1094_109452

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b * x / Real.log x) - (a * x)
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1) / (Real.log x)^2) - a

theorem problem_part1 (a b : ℝ) :
  (f' (Real.exp 2) a b = -(3/4)) ∧ (f (Real.exp 2) a b = -(1/2) * (Real.exp 2)) →
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part2 (a : ℝ) :
  (∃ x1 x2, x1 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x1 a 1 ≤ f' x2 a 1 + a) →
  a ≥ (1/2 - 1/(4 * Real.exp 2)) :=
sorry

end problem_part1_problem_part2_l1094_109452


namespace prove_zero_function_l1094_109457

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f (x ^ 333 + y) = f (x ^ 2018 + 2 * y) + f (x ^ 42)

theorem prove_zero_function : ∀ x : ℝ, f x = 0 :=
by
  sorry

end prove_zero_function_l1094_109457


namespace evaluate_expression_l1094_109498

theorem evaluate_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end evaluate_expression_l1094_109498


namespace pirate_flag_minimal_pieces_l1094_109404

theorem pirate_flag_minimal_pieces (original_stripes : ℕ) (desired_stripes : ℕ) (cuts_needed : ℕ) : 
  original_stripes = 12 →
  desired_stripes = 10 →
  cuts_needed = 1 →
  ∃ pieces : ℕ, pieces = 2 ∧ 
  (∀ (top_stripes bottom_stripes: ℕ), top_stripes + bottom_stripes = original_stripes → top_stripes = desired_stripes → 
   pieces = 1 + (if bottom_stripes = original_stripes - desired_stripes then 1 else 0)) :=
by intros;
   sorry

end pirate_flag_minimal_pieces_l1094_109404


namespace cities_drawn_from_group_b_l1094_109456

def group_b_cities : ℕ := 8
def selection_probability : ℝ := 0.25

theorem cities_drawn_from_group_b : 
  group_b_cities * selection_probability = 2 :=
by
  sorry

end cities_drawn_from_group_b_l1094_109456


namespace abs_quotient_eq_sqrt_7_div_2_l1094_109485

theorem abs_quotient_eq_sqrt_7_div_2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 2) :=
by
  sorry

end abs_quotient_eq_sqrt_7_div_2_l1094_109485


namespace count_distribution_schemes_l1094_109425

theorem count_distribution_schemes :
  let total_pieces := 7
  let pieces_A_B := 2 + 2
  let remaining_pieces := total_pieces - pieces_A_B
  let communities := 5

  -- Number of ways to distribute 7 pieces of equipment such that communities A and B receive at least 2 pieces each
  let ways_one_community := 5
  let ways_two_communities := 20  -- 2 * (choose 5 2)
  let ways_three_communities := 10  -- (choose 5 3)

  ways_one_community + ways_two_communities + ways_three_communities = 35 :=
by
  -- The actual proof steps are omitted here.
  sorry

end count_distribution_schemes_l1094_109425


namespace candy_cost_l1094_109440

theorem candy_cost
  (C : ℝ) -- cost per pound of the first candy
  (w1 : ℝ := 30) -- weight of the first candy
  (c2 : ℝ := 5) -- cost per pound of the second candy
  (w2 : ℝ := 60) -- weight of the second candy
  (w_mix : ℝ := 90) -- total weight of the mixture
  (c_mix : ℝ := 6) -- desired cost per pound of the mixture
  (h1 : w1 * C + w2 * c2 = w_mix * c_mix) -- cost equation for the mixture
  : C = 8 :=
by
  sorry

end candy_cost_l1094_109440


namespace zero_points_ordering_l1094_109481

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x^3 + x - 2

theorem zero_points_ordering :
  ∃ x1 x2 x3 : ℝ,
    f x1 = 0 ∧ x1 < 0 ∧ 
    g x2 = 0 ∧ 0 < x2 ∧ x2 < 1 ∧
    h x3 = 0 ∧ 1 < x3 ∧ x3 < 2 ∧
    x1 < x2 ∧ x2 < x3 := sorry

end zero_points_ordering_l1094_109481


namespace num_pairs_satisfying_inequality_l1094_109424

theorem num_pairs_satisfying_inequality : 
  ∃ (s : Nat), s = 204 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → m^2 + n < 50 → s = 204 :=
by
  sorry

end num_pairs_satisfying_inequality_l1094_109424


namespace cyclist_total_distance_l1094_109483

-- Definitions for velocities and times
def v1 : ℝ := 2  -- velocity in the first minute (m/s)
def v2 : ℝ := 4  -- velocity in the second minute (m/s)
def t : ℝ := 60  -- time interval in seconds (1 minute)

-- Total distance covered in two minutes
def total_distance : ℝ := v1 * t + v2 * t

-- The proof statement
theorem cyclist_total_distance : total_distance = 360 := by
  sorry

end cyclist_total_distance_l1094_109483


namespace race_dead_heat_l1094_109417

theorem race_dead_heat (va vb D : ℝ) (hva_vb : va = (15 / 16) * vb) (dist_a : D = D) (dist_b : D = (15 / 16) * D) (race_finish : D / va = (15 / 16) * D / vb) :
  va / vb = 15 / 16 :=
by sorry

end race_dead_heat_l1094_109417


namespace triangle_side_inequality_l1094_109496

theorem triangle_side_inequality (y : ℕ) (h : 3 < y^2 ∧ y^2 < 19) : 
  y = 2 ∨ y = 3 ∨ y = 4 :=
sorry

end triangle_side_inequality_l1094_109496


namespace num_congruent_mod_7_count_mod_7_eq_22_l1094_109447

theorem num_congruent_mod_7 (n : ℕ) :
  (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) → ∃ k, 0 ≤ k ∧ k ≤ 21 ∧ n = 7 * k + 1 :=
sorry

theorem count_mod_7_eq_22 : 
  (∃ n_set : Finset ℕ, 
    (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) ∧ 
    Finset.card n_set = 22) :=
sorry

end num_congruent_mod_7_count_mod_7_eq_22_l1094_109447


namespace range_of_m_l1094_109477

def p (m : ℝ) : Prop :=
  let Δ := m^2 - 4
  Δ > 0 ∧ -m < 0

def q (m : ℝ) : Prop :=
  let Δ := 16*(m-2)^2 - 16
  Δ < 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ ((1 < m ∧ m ≤ 2) ∨ 3 ≤ m) :=
by {
  sorry
}

end range_of_m_l1094_109477


namespace find_c_l1094_109480

theorem find_c (x c : ℝ) (h : ((5 * x + 38 + c) / 5) = (x + 4) + 5) : c = 7 :=
by
  sorry

end find_c_l1094_109480


namespace no_politics_reporters_l1094_109446

theorem no_politics_reporters (X Y Both XDontY YDontX International PercentageTotal : ℝ) 
  (hX : X = 0.35)
  (hY : Y = 0.25)
  (hBoth : Both = 0.20)
  (hXDontY : XDontY = 0.30)
  (hInternational : International = 0.15)
  (hPercentageTotal : PercentageTotal = 1.0) :
  PercentageTotal - ((X + Y - Both) - XDontY + International) = 0.75 :=
by sorry

end no_politics_reporters_l1094_109446


namespace sum_is_zero_l1094_109403

noncomputable def z : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.sin (3 * Real.pi / 8) * Complex.I

theorem sum_is_zero (hz : z^8 = 1) (hz1 : z ≠ 1) :
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^12)) = 0 :=
by
  sorry

end sum_is_zero_l1094_109403


namespace cuboid_edge_lengths_l1094_109451

theorem cuboid_edge_lengths (a b c : ℕ) (S V : ℕ) :
  (S = 2 * (a * b + b * c + c * a)) ∧ (V = a * b * c) ∧ (V = S) ∧ 
  (∃ d : ℕ, d = Int.sqrt (a^2 + b^2 + c^2)) →
  (∃ a b c : ℕ, a = 4 ∧ b = 8 ∧ c = 8) :=
by
  sorry

end cuboid_edge_lengths_l1094_109451


namespace find_f_1000_l1094_109476

theorem find_f_1000 (f : ℕ → ℕ) 
    (h1 : ∀ n : ℕ, 0 < n → f (f n) = 2 * n) 
    (h2 : ∀ n : ℕ, 0 < n → f (3 * n + 1) = 3 * n + 2) : 
    f 1000 = 1008 :=
by
  sorry

end find_f_1000_l1094_109476


namespace probability_beautiful_equation_l1094_109492

def tetrahedron_faces : Set ℕ := {1, 2, 3, 4}

def is_beautiful_equation (a b : ℕ) : Prop :=
    ∃ m ∈ tetrahedron_faces, a = m + 1 ∨ a = m + 2 ∨ a = m + 3 ∨ a = m + 4 ∧ b = m * (a - m)

theorem probability_beautiful_equation : 
  (∃ a b1 b2, is_beautiful_equation a b1 ∧ is_beautiful_equation a b2) ∧
  (∃ a b1 b2, tetrahedron_faces ⊆ {a} ∧ tetrahedron_faces ⊆ {b1} ∧ tetrahedron_faces ⊆ {b2}) :=
  sorry

end probability_beautiful_equation_l1094_109492


namespace part_a_gray_black_area_difference_l1094_109455

theorem part_a_gray_black_area_difference :
    ∀ (a b : ℕ), 
        a = 4 → 
        b = 3 →
        a^2 - b^2 = 7 :=
by
  intros a b h_a h_b
  sorry

end part_a_gray_black_area_difference_l1094_109455


namespace sum_from_neg_50_to_75_l1094_109486

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end sum_from_neg_50_to_75_l1094_109486


namespace sqrt3_minus1_plus_inv3_pow_minus2_l1094_109432

theorem sqrt3_minus1_plus_inv3_pow_minus2 :
  (Real.sqrt 3 - 1) + (1 / (1/3) ^ 2) = Real.sqrt 3 + 8 :=
by
  sorry

end sqrt3_minus1_plus_inv3_pow_minus2_l1094_109432


namespace nat_triple_solution_l1094_109419

theorem nat_triple_solution (x y n : ℕ) :
  (x! + y!) / n! = 3^n ↔ (x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1) := 
by
  sorry

end nat_triple_solution_l1094_109419


namespace child_admission_charge_l1094_109414

-- Given conditions
variables (A C : ℝ) (T : ℝ := 3.25) (n : ℕ := 3)

-- Admission charge for an adult
def admission_charge_adult : ℝ := 1

-- Admission charge for a child
def admission_charge_child (C : ℝ) : ℝ := C

-- Total cost paid by adult with 3 children
def total_cost (A C : ℝ) (n : ℕ) : ℝ := A + n * C

-- The proof statement
theorem child_admission_charge (C : ℝ) : total_cost 1 C 3 = 3.25 -> C = 0.75 :=
by
  sorry

end child_admission_charge_l1094_109414


namespace cone_height_l1094_109422

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l1094_109422


namespace total_bowling_balls_l1094_109429

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l1094_109429


namespace find_number_l1094_109473

theorem find_number (x : ℝ) : 0.40 * x = 0.80 * 5 + 2 → x = 15 :=
by
  intros h
  sorry

end find_number_l1094_109473


namespace travel_time_is_correct_l1094_109427

-- Define the conditions
def speed : ℕ := 60 -- Speed in km/h
def distance : ℕ := 120 -- Distance between points A and B in km

-- Time calculation from A to B 
def time_AB : ℕ := distance / speed

-- Time calculation from B to A (since speed and distance are the same)
def time_BA : ℕ := distance / speed

-- Total time calculation
def total_time : ℕ := time_AB + time_BA

-- The proper statement to prove
theorem travel_time_is_correct : total_time = 4 := by
  -- Additional steps and arguments would go here
  -- skipping proof
  sorry

end travel_time_is_correct_l1094_109427


namespace probability_of_multiple_of_42_is_zero_l1094_109428

-- Given conditions
def factors_200 : Set ℕ := {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}
def multiple_of_42 (n : ℕ) : Prop := n % 42 = 0

-- Problem statement: the probability of selecting a multiple of 42 from the factors of 200 is 0.
theorem probability_of_multiple_of_42_is_zero : 
  ∀ (n : ℕ), n ∈ factors_200 → ¬ multiple_of_42 n := 
by
  sorry

end probability_of_multiple_of_42_is_zero_l1094_109428


namespace right_triangle_area_l1094_109494

theorem right_triangle_area (x y : ℝ) 
  (h1 : x + y = 4) 
  (h2 : x^2 + y^2 = 9) : 
  (1/2) * x * y = 7 / 4 := 
by
  sorry

end right_triangle_area_l1094_109494


namespace percentage_seeds_from_dandelions_l1094_109482

def Carla_sunflowers := 6
def Carla_dandelions := 8
def seeds_per_sunflower := 9
def seeds_per_dandelion := 12

theorem percentage_seeds_from_dandelions :
  96 / 150 * 100 = 64 := by
  sorry

end percentage_seeds_from_dandelions_l1094_109482


namespace total_output_correct_l1094_109454

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end total_output_correct_l1094_109454


namespace impossible_to_have_only_stacks_of_three_l1094_109458

theorem impossible_to_have_only_stacks_of_three (n J : ℕ) (h_initial_n : n = 1) (h_initial_J : J = 1001) :
  (∀ n J, (n + J = 1002) → (∀ k : ℕ, 3 * k ≤ J → k + 3 * k ≠ 1002)) 
  :=
sorry

end impossible_to_have_only_stacks_of_three_l1094_109458


namespace number_of_yellow_balloons_l1094_109401

-- Define the problem
theorem number_of_yellow_balloons :
  ∃ (Y B : ℕ), 
  B = Y + 1762 ∧ 
  Y + B = 10 * 859 ∧ 
  Y = 3414 :=
by
  -- Proof is skipped, so we use sorry
  sorry

end number_of_yellow_balloons_l1094_109401


namespace central_angle_of_sector_l1094_109400

noncomputable def central_angle (radius perimeter: ℝ) : ℝ :=
  ((perimeter - 2 * radius) / (2 * Real.pi * radius)) * 360

theorem central_angle_of_sector :
  central_angle 28 144 = 180.21 :=
by
  simp [central_angle]
  sorry

end central_angle_of_sector_l1094_109400


namespace balloon_total_l1094_109416

def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

theorem balloon_total :
  total_balloons 40 41 = 81 :=
by
  sorry

end balloon_total_l1094_109416


namespace cello_viola_pairs_l1094_109436

theorem cello_viola_pairs (cellos violas : Nat) (p_same_tree : ℚ) (P : Nat)
  (h_cellos : cellos = 800)
  (h_violas : violas = 600)
  (h_p_same_tree : p_same_tree = 0.00020833333333333335)
  (h_equation : P * ((1 : ℚ) / cellos * (1 : ℚ) / violas) = p_same_tree) :
  P = 100 := 
by
  sorry

end cello_viola_pairs_l1094_109436


namespace second_number_is_22_l1094_109413

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end second_number_is_22_l1094_109413


namespace patternD_cannot_form_pyramid_l1094_109484

-- Define the patterns
inductive Pattern
| A
| B
| C
| D

-- Define the condition for folding into a pyramid with a square base
def canFormPyramidWithSquareBase (p : Pattern) : Prop :=
  p = Pattern.A ∨ p = Pattern.B ∨ p = Pattern.C

-- Goal: Prove that Pattern D cannot be folded into a pyramid with a square base
theorem patternD_cannot_form_pyramid : ¬ canFormPyramidWithSquareBase Pattern.D :=
by
  -- Need to provide the proof here
  sorry

end patternD_cannot_form_pyramid_l1094_109484


namespace part1_part2_l1094_109499

-- Define the first part of the problem
theorem part1 (a b : ℝ) :
  (∀ x : ℝ, |x^2 + a * x + b| ≤ 2 * |x - 4| * |x + 2|) → (a = -2 ∧ b = -8) :=
sorry

-- Define the second part of the problem
theorem part2 (a b m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 + a * x + b ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end part1_part2_l1094_109499


namespace molecular_weight_single_mole_l1094_109411

theorem molecular_weight_single_mole :
  (∀ (w_7m C6H8O7 : ℝ), w_7m = 1344 → (w_7m / 7) = 192) :=
by
  intros w_7m C6H8O7 h
  sorry

end molecular_weight_single_mole_l1094_109411


namespace arithmetic_sequence_30th_term_value_l1094_109449

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

-- Given conditions
def a1 : ℤ := 3
def a2 : ℤ := 15
def a3 : ℤ := 27

-- Calculate the common difference d
def d : ℤ := a2 - a1

-- Define the 30th term
def a30 := arithmetic_sequence a1 d 30

theorem arithmetic_sequence_30th_term_value :
  a30 = 351 := by
  sorry

end arithmetic_sequence_30th_term_value_l1094_109449


namespace solve_for_x_l1094_109435

theorem solve_for_x (x y : ℕ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 :=
sorry

end solve_for_x_l1094_109435


namespace intersection_complement_l1094_109431

def M (x : ℝ) : Prop := x^2 - 2 * x < 0
def N (x : ℝ) : Prop := x < 1

theorem intersection_complement (x : ℝ) :
  (M x ∧ ¬N x) ↔ (1 ≤ x ∧ x < 2) := 
sorry

end intersection_complement_l1094_109431


namespace value_of_one_house_l1094_109445

theorem value_of_one_house
  (num_brothers : ℕ) (num_houses : ℕ) (payment_each : ℕ) 
  (total_money_paid : ℕ) (num_older : ℕ) (num_younger : ℕ)
  (share_per_younger : ℕ) (total_inheritance : ℕ) (value_of_house : ℕ) :
  num_brothers = 5 →
  num_houses = 3 →
  num_older = 3 →
  num_younger = 2 →
  payment_each = 800 →
  total_money_paid = num_older * payment_each →
  share_per_younger = total_money_paid / num_younger →
  total_inheritance = num_brothers * share_per_younger →
  value_of_house = total_inheritance / num_houses →
  value_of_house = 2000 :=
by {
  -- Provided conditions and statements without proofs
  sorry
}

end value_of_one_house_l1094_109445


namespace tangent_function_intersection_l1094_109442

theorem tangent_function_intersection (ω : ℝ) (hω : ω > 0) (h_period : (π / ω) = 3 * π) :
  let f (x : ℝ) := Real.tan (ω * x + π / 3)
  f π = -Real.sqrt 3 :=
by
  sorry

end tangent_function_intersection_l1094_109442


namespace polygon_sides_eq_13_l1094_109490

theorem polygon_sides_eq_13 (n : ℕ) (h : n * (n - 3) = 5 * n) : n = 13 := by
  sorry

end polygon_sides_eq_13_l1094_109490


namespace problem_statement_l1094_109405

theorem problem_statement (x : ℝ) (h₀ : x > 0) (n : ℕ) (hn : n > 0) :
  (x + (n^n : ℝ) / x^n) ≥ (n + 1) :=
sorry

end problem_statement_l1094_109405


namespace range_of_a_l1094_109441

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, 4 ≤ x → (if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a) ≥ 4) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l1094_109441


namespace sin_double_angle_l1094_109418

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_l1094_109418


namespace calories_consumed_Jean_l1094_109471

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l1094_109471


namespace problem_statement_l1094_109493

theorem problem_statement (a b : ℝ) :
  a^2 + b^2 - a - b - a * b + 0.25 ≥ 0 ∧ (a^2 + b^2 - a - b - a * b + 0.25 = 0 ↔ ((a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0))) :=
by 
  sorry

end problem_statement_l1094_109493


namespace volunteer_org_percentage_change_l1094_109479

theorem volunteer_org_percentage_change :
  ∀ (X : ℝ), X > 0 → 
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  (X - spring_decrease) / X * 100 = 11.71 :=
by
  intro X hX
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  show (_ - _) / _ * _ = _
  sorry

end volunteer_org_percentage_change_l1094_109479


namespace white_red_balls_l1094_109462

theorem white_red_balls (w r : ℕ) 
  (h1 : 3 * w = 5 * r)
  (h2 : w + 15 + r = 50) : 
  r = 12 :=
by
  sorry

end white_red_balls_l1094_109462


namespace range_of_a_l1094_109470

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici 3) := 
sorry

end range_of_a_l1094_109470


namespace roses_in_december_l1094_109412

theorem roses_in_december (rOct rNov rJan rFeb : ℕ) 
  (hOct : rOct = 108)
  (hNov : rNov = 120)
  (hJan : rJan = 144)
  (hFeb : rFeb = 156)
  (pattern : (rNov - rOct = 12 ∨ rNov - rOct = 24) ∧ 
             (rJan - rNov = 12 ∨ rJan - rNov = 24) ∧
             (rFeb - rJan = 12 ∨ rFeb - rJan = 24) ∧ 
             (∀ m n, (m - n = 12 ∨ m - n = 24) → 
               ((rNov - rOct) ≠ (rJan - rNov) ↔ 
               (rJan - rNov) ≠ (rFeb - rJan)))) : 
  ∃ rDec : ℕ, rDec = 132 := 
by {
  sorry
}

end roses_in_december_l1094_109412


namespace total_groups_correct_l1094_109497

-- Definitions from conditions
def eggs := 57
def egg_group_size := 7

def bananas := 120
def banana_group_size := 10

def marbles := 248
def marble_group_size := 8

-- Calculate the number of groups for each type of object
def egg_groups := eggs / egg_group_size
def banana_groups := bananas / banana_group_size
def marble_groups := marbles / marble_group_size

-- Total number of groups
def total_groups := egg_groups + banana_groups + marble_groups

-- Proof statement
theorem total_groups_correct : total_groups = 51 := by
  sorry

end total_groups_correct_l1094_109497


namespace fraction_of_milk_in_second_cup_l1094_109439

noncomputable def ratio_mixture (V: ℝ) (x: ℝ) :=
  ((2 / 5 * V + (1 - x) * V) / (3 / 5 * V + x * V))

theorem fraction_of_milk_in_second_cup
  (V: ℝ) 
  (hV: V > 0)
  (hx: ratio_mixture V x = 3 / 7) :
  x = 4 / 5 :=
by
  sorry

end fraction_of_milk_in_second_cup_l1094_109439


namespace no_solutions_l1094_109491

theorem no_solutions (N : ℕ) (d : ℕ) (H : ∀ (i j : ℕ), i ≠ j → d = 6 ∧ d + d = 13) : false :=
by
  sorry

end no_solutions_l1094_109491


namespace probability_correct_l1094_109495

noncomputable def probability_study_group : ℝ :=
  let p_woman : ℝ := 0.5
  let p_man : ℝ := 0.5

  let p_woman_lawyer : ℝ := 0.3
  let p_woman_doctor : ℝ := 0.4
  let p_woman_engineer : ℝ := 0.3

  let p_man_lawyer : ℝ := 0.4
  let p_man_doctor : ℝ := 0.2
  let p_man_engineer : ℝ := 0.4

  (p_woman * p_woman_lawyer + p_woman * p_woman_doctor +
  p_man * p_man_lawyer + p_man * p_man_doctor)

theorem probability_correct : probability_study_group = 0.65 := by
  sorry

end probability_correct_l1094_109495


namespace password_probability_l1094_109463

def isNonNegativeSingleDigit (n : ℕ) : Prop := n ≤ 9

def isOddSingleDigit (n : ℕ) : Prop := isNonNegativeSingleDigit n ∧ n % 2 = 1

def isPositiveSingleDigit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def isVowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

-- Probability that an odd single-digit number followed by a vowel and a positive single-digit number
def prob_odd_vowel_positive_digits : ℚ :=
  let prob_first := 5 / 10 -- Probability of odd single-digit number
  let prob_vowel := 5 / 26 -- Probability of vowel
  let prob_last := 9 / 10 -- Probability of positive single-digit number
  prob_first * prob_vowel * prob_last

theorem password_probability :
  prob_odd_vowel_positive_digits = 9 / 104 :=
by
  sorry

end password_probability_l1094_109463


namespace average_of_r_s_t_l1094_109410

theorem average_of_r_s_t (r s t : ℝ) (h : (5/4) * (r + s + t) = 20) : (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l1094_109410


namespace fisherman_daily_earnings_l1094_109474

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l1094_109474


namespace moving_circle_fixed_point_coordinates_l1094_109402

theorem moving_circle_fixed_point_coordinates (m x y : Real) :
    (∀ m : ℝ, x^2 + y^2 - 2 * m * x - 4 * m * y + 6 * m - 2 = 0) →
    (x = 1 ∧ y = 1 ∨ x = 1 / 5 ∧ y = 7 / 5) :=
  by
    sorry

end moving_circle_fixed_point_coordinates_l1094_109402


namespace min_shots_to_hit_terrorist_l1094_109406

theorem min_shots_to_hit_terrorist : ∀ terrorist_position : ℕ, (1 ≤ terrorist_position ∧ terrorist_position ≤ 10) →
  ∃ shots : ℕ, shots ≥ 6 ∧ (∀ move : ℕ, (shots - move) ≥ 1 → (terrorist_position + move ≤ 10 → terrorist_position % 2 = move % 2)) :=
by
  sorry

end min_shots_to_hit_terrorist_l1094_109406
