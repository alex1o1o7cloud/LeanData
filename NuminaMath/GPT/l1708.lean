import Mathlib

namespace Billy_weighs_more_l1708_170819

-- Variables and assumptions
variable (Billy Brad Carl : ℕ)
variable (b_weight : Billy = 159)
variable (c_weight : Carl = 145)
variable (brad_formula : Brad = Carl + 5)

-- Theorem statement to prove the required condition
theorem Billy_weighs_more :
  Billy - Brad = 9 :=
by
  -- Here we put the proof steps, but it's omitted as per instructions.
  sorry

end Billy_weighs_more_l1708_170819


namespace line_tangent_to_parabola_proof_l1708_170846

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l1708_170846


namespace sequence_has_max_and_min_l1708_170833

noncomputable def a_n (n : ℕ) : ℝ := (4 / 9)^(n - 1) - (2 / 3)^(n - 1)

theorem sequence_has_max_and_min : 
  (∃ N, ∀ n, a_n n ≤ a_n N) ∧ 
  (∃ M, ∀ n, a_n n ≥ a_n M) :=
sorry

end sequence_has_max_and_min_l1708_170833


namespace fruit_problem_l1708_170883

variables (A O x : ℕ) -- Natural number variables for apples, oranges, and oranges put back

theorem fruit_problem :
  (A + O = 10) ∧
  (40 * A + 60 * O = 480) ∧
  (240 + 60 * (O - x) = 45 * (10 - x)) →
  A = 6 ∧ O = 4 ∧ x = 2 :=
  sorry

end fruit_problem_l1708_170883


namespace least_possible_value_of_m_plus_n_l1708_170804

theorem least_possible_value_of_m_plus_n 
(m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) 
(hgcd : Nat.gcd (m + n) 210 = 1) 
(hdiv : ∃ k, m^m = k * n^n)
(hnotdiv : ¬ ∃ k, m = k * n) : 
  m + n = 407 := 
sorry

end least_possible_value_of_m_plus_n_l1708_170804


namespace range_of_a_l1708_170845

noncomputable def proposition_p (a : ℝ) : Prop := 
  0 < a ∧ a < 1

noncomputable def proposition_q (a : ℝ) : Prop := 
  a > 1 / 4

theorem range_of_a (a : ℝ) : 
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by sorry

end range_of_a_l1708_170845


namespace direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l1708_170835

-- Direct Proportional Function
theorem direct_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 1) → m = 1 :=
by 
  sorry

-- Inverse Proportional Function
theorem inverse_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = -1) → m = -1 :=
by 
  sorry

-- Quadratic Function
theorem quadratic_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 2) → (m = (-1 + Real.sqrt 13) / 2 ∨ m = (-1 - Real.sqrt 13) / 2) :=
by 
  sorry

-- Power Function
theorem power_function (m : ℝ) :
  (m^2 + 2 * m = 1) → (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) :=
by 
  sorry

end direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l1708_170835


namespace melissa_points_per_game_l1708_170867

variable (t g p : ℕ)

theorem melissa_points_per_game (ht : t = 36) (hg : g = 3) : p = t / g → p = 12 :=
by
  intro h
  sorry

end melissa_points_per_game_l1708_170867


namespace gcd_correct_l1708_170850

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l1708_170850


namespace simplify_fraction_l1708_170888

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = (65 : ℚ) / 12 := 
by
  sorry

end simplify_fraction_l1708_170888


namespace roxy_total_plants_remaining_l1708_170882

def initial_flowering_plants : Nat := 7
def initial_fruiting_plants : Nat := 2 * initial_flowering_plants
def flowering_plants_bought : Nat := 3
def fruiting_plants_bought : Nat := 2
def flowering_plants_given_away : Nat := 1
def fruiting_plants_given_away : Nat := 4

def total_remaining_plants : Nat :=
  let flowering_plants_now := initial_flowering_plants + flowering_plants_bought - flowering_plants_given_away
  let fruiting_plants_now := initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given_away
  flowering_plants_now + fruiting_plants_now

theorem roxy_total_plants_remaining
  : total_remaining_plants = 21 := by
  sorry

end roxy_total_plants_remaining_l1708_170882


namespace right_triangle_area_l1708_170868

theorem right_triangle_area (a b c : ℝ) (h1 : a + b + c = 90) (h2 : a^2 + b^2 + c^2 = 3362) (h3 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 180 :=
by
  sorry

end right_triangle_area_l1708_170868


namespace marys_birthday_l1708_170891

theorem marys_birthday (M : ℝ) (h1 : (3 / 4) * M - (3 / 20) * M = 60) : M = 100 := by
  -- Leave the proof as sorry for now
  sorry

end marys_birthday_l1708_170891


namespace f_x_plus_1_even_f_x_plus_3_odd_l1708_170865

variable (R : Type) [CommRing R]

variable (f : R → R)

-- Conditions
axiom condition1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom condition2 : ∀ x : R, f (x - 2) + f (-x) = 0

-- Prove that f(x + 1) is an even function
theorem f_x_plus_1_even (x : R) : f (x + 1) = f (-(x + 1)) :=
by sorry

-- Prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd (x : R) : f (x + 3) = - f (-(x + 3)) :=
by sorry

end f_x_plus_1_even_f_x_plus_3_odd_l1708_170865


namespace expected_pairs_of_adjacent_face_cards_is_44_over_17_l1708_170841
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17_l1708_170841


namespace solve_inequality_l1708_170825

theorem solve_inequality :
  {x : ℝ | 8*x^3 - 6*x^2 + 5*x - 5 < 0} = {x : ℝ | x < 1/2} :=
sorry

end solve_inequality_l1708_170825


namespace boy_walking_speed_l1708_170855

theorem boy_walking_speed 
  (travel_rate : ℝ) 
  (total_journey_time : ℝ) 
  (distance : ℝ) 
  (post_office_time : ℝ) 
  (walking_back_time : ℝ) 
  (walking_speed : ℝ): 
  travel_rate = 12.5 ∧ 
  total_journey_time = 5 + 48/60 ∧ 
  distance = 9.999999999999998 ∧ 
  post_office_time = distance / travel_rate ∧ 
  walking_back_time = total_journey_time - post_office_time ∧ 
  walking_speed = distance / walking_back_time 
  → walking_speed = 2 := 
by 
  intros h;
  sorry

end boy_walking_speed_l1708_170855


namespace intersection_complement_M_N_l1708_170839

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_complement_M_N :
  (U \ M) ∩ N = {-3, -4} :=
by {
  sorry
}

end intersection_complement_M_N_l1708_170839


namespace solve_first_system_solve_second_system_l1708_170826

-- Define the first system of equations
def first_system (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)

-- Define the solution to the first system
def solution1 (x y : ℝ) : Prop := (x = 3) ∧ (y = -2)

-- Define the second system of equations
def second_system (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Define the solution to the second system
def solution2 (x y : ℝ) : Prop := (x = 4) ∧ (y = -2)

-- Define the problem statement in Lean
theorem solve_first_system : ∃ x y : ℝ, first_system x y ↔ solution1 x y :=
by
  sorry

theorem solve_second_system : ∃ x y : ℝ, second_system x y ↔ solution2 x y :=
by
  sorry

end solve_first_system_solve_second_system_l1708_170826


namespace molecular_weight_compound_l1708_170806

/-- Definition of atomic weights for elements H, Cr, and O in AMU (Atomic Mass Units) --/
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999

/-- Proof statement to calculate the molecular weight of a compound with 2 H, 1 Cr, and 4 O --/
theorem molecular_weight_compound :
  2 * atomic_weight_H + 1 * atomic_weight_Cr + 4 * atomic_weight_O = 118.008 :=
by
  sorry

end molecular_weight_compound_l1708_170806


namespace tenth_term_geom_seq_l1708_170895

theorem tenth_term_geom_seq :
  let a := (5 : ℚ)
  let r := (4 / 3 : ℚ)
  let n := 10
  (a * r^(n - 1)) = (1310720 / 19683 : ℚ) :=
by
  sorry

end tenth_term_geom_seq_l1708_170895


namespace sum_of_cubes_8001_l1708_170823
-- Import the entire Mathlib library

-- Define a property on integers
def approx (x y : ℝ) := abs (x - y) < 0.000000000000004

-- Define the variables a and b
variables (a b : ℤ)

-- State the theorem
theorem sum_of_cubes_8001 (h : approx (a * b : ℝ) 19.999999999999996) : a^3 + b^3 = 8001 := 
sorry

end sum_of_cubes_8001_l1708_170823


namespace claire_apple_pies_l1708_170848

theorem claire_apple_pies (N : ℤ) 
  (h1 : N % 6 = 4) 
  (h2 : N % 8 = 5) 
  (h3 : N < 30) : 
  N = 22 :=
by
  sorry

end claire_apple_pies_l1708_170848


namespace angle_BDC_proof_l1708_170815

noncomputable def angle_sum_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Given conditions
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- The problem of proving that angle_BDC = 20 degrees
theorem angle_BDC_proof (A E C BDC : ℝ) 
  (hA : A = angle_A)
  (hE : E = angle_E)
  (hC : C = angle_C) :
  BDC = 20 :=
  sorry

end angle_BDC_proof_l1708_170815


namespace smallest_n_l1708_170890

theorem smallest_n (n : ℕ) : 
  (n % 6 = 4) ∧ (n % 7 = 2) ∧ (n > 20) → n = 58 :=
by
  sorry

end smallest_n_l1708_170890


namespace wendy_time_correct_l1708_170857

noncomputable section

def bonnie_time : ℝ := 7.80
def wendy_margin : ℝ := 0.25

theorem wendy_time_correct : (bonnie_time - wendy_margin) = 7.55 := by
  sorry

end wendy_time_correct_l1708_170857


namespace lcm_48_180_l1708_170811

theorem lcm_48_180 : Int.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l1708_170811


namespace multiplicative_inverse_of_AB_l1708_170837

def A : ℕ := 222222
def B : ℕ := 476190
def N : ℕ := 189
def modulus : ℕ := 1000000

theorem multiplicative_inverse_of_AB :
  (A * B * N) % modulus = 1 % modulus :=
by
  sorry

end multiplicative_inverse_of_AB_l1708_170837


namespace centroid_of_quadrant_arc_l1708_170887

def circle_equation (R : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = R^2
def density (ρ₀ x y : ℝ) : ℝ := ρ₀ * x * y

theorem centroid_of_quadrant_arc (R ρ₀ : ℝ) :
  (∃ x y, circle_equation R x y ∧ x ≥ 0 ∧ y ≥ 0) →
  ∃ x_c y_c, x_c = 2 * R / 3 ∧ y_c = 2 * R / 3 :=
sorry

end centroid_of_quadrant_arc_l1708_170887


namespace isosceles_with_base_c_l1708_170859

theorem isosceles_with_base_c (a b c: ℝ) (h: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (triangle_rel: 1/a - 1/b + 1/c = 1/(a - b + c)) : a = c ∨ b = c :=
sorry

end isosceles_with_base_c_l1708_170859


namespace sequence_inequality_l1708_170871

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_condition : ∀ n m, a (n + m) ≤ a n + a m) :
  ∀ n m, n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := by
  sorry

end sequence_inequality_l1708_170871


namespace train_pass_bridge_in_36_seconds_l1708_170899

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/h

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- m/s
noncomputable def total_distance : ℝ := train_length + bridge_length -- meters
noncomputable def passing_time : ℝ := total_distance / train_speed_ms -- seconds

theorem train_pass_bridge_in_36_seconds :
  passing_time = 36 := 
sorry

end train_pass_bridge_in_36_seconds_l1708_170899


namespace rate_of_mangoes_per_kg_l1708_170897

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg_l1708_170897


namespace evaluate_product_l1708_170802

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l1708_170802


namespace find_b_l1708_170898

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 315 * b) : b = 7 :=
by
  -- The actual proof would go here
  sorry

end find_b_l1708_170898


namespace average_viewer_watches_two_videos_daily_l1708_170834

variable (V : ℕ)
variable (video_time : ℕ := 7)
variable (ad_time : ℕ := 3)
variable (total_time : ℕ := 17)

theorem average_viewer_watches_two_videos_daily :
  7 * V + 3 = 17 → V = 2 := 
by
  intro h
  have h1 : 7 * V = 14 := by linarith
  have h2 : V = 2 := by linarith
  exact h2

end average_viewer_watches_two_videos_daily_l1708_170834


namespace cube_volume_of_surface_area_l1708_170843

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l1708_170843


namespace sqrt_exp_cube_l1708_170852

theorem sqrt_exp_cube :
  ((Real.sqrt ((Real.sqrt 5)^4))^3 = 125) :=
by
  sorry

end sqrt_exp_cube_l1708_170852


namespace total_baskets_l1708_170893

theorem total_baskets (Alex_baskets Sandra_baskets Hector_baskets Jordan_baskets total_baskets : ℕ)
  (h1 : Alex_baskets = 8)
  (h2 : Sandra_baskets = 3 * Alex_baskets)
  (h3 : Hector_baskets = 2 * Sandra_baskets)
  (total_combined_baskets := Alex_baskets + Sandra_baskets + Hector_baskets)
  (h4 : Jordan_baskets = total_combined_baskets / 5)
  (h5 : total_baskets = Alex_baskets + Sandra_baskets + Hector_baskets + Jordan_baskets) :
  total_baskets = 96 := by
  sorry

end total_baskets_l1708_170893


namespace Megan_pays_correct_amount_l1708_170821

def original_price : ℝ := 22
def discount : ℝ := 6
def amount_paid := original_price - discount

theorem Megan_pays_correct_amount : amount_paid = 16 := by
  sorry

end Megan_pays_correct_amount_l1708_170821


namespace smallest_factor_of_36_l1708_170878

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l1708_170878


namespace perpendicular_vectors_m_eq_0_or_neg2_l1708_170814

theorem perpendicular_vectors_m_eq_0_or_neg2
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (1, m - 1))
  (h : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) :
  m = 0 ∨ m = -2 := sorry

end perpendicular_vectors_m_eq_0_or_neg2_l1708_170814


namespace midpoint_trajectory_of_circle_l1708_170808

theorem midpoint_trajectory_of_circle 
  (M P : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hx : B = (3, 0))
  (hp : ∃(a b : ℝ), (P = (2 * a - 3, 2 * b)) ∧ (a^2 + b^2 = 1))
  (hm : M = ((P.1 + B.1) / 2, (P.2 + B.2) / 2)) :
  M.1^2 + M.2^2 - 3 * M.1 + 2 = 0 :=
by {
  -- Proof goes here
  sorry
}

end midpoint_trajectory_of_circle_l1708_170808


namespace angle_coincides_with_graph_y_eq_neg_abs_x_l1708_170838

noncomputable def angle_set (α : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

theorem angle_coincides_with_graph_y_eq_neg_abs_x (α : ℝ) :
  α ∈ angle_set α ↔ 
  ∃ k : ℤ, (α = k * 360 + 225 ∨ α = k * 360 + 315) :=
by
  sorry

end angle_coincides_with_graph_y_eq_neg_abs_x_l1708_170838


namespace graph_paper_squares_below_line_l1708_170874

theorem graph_paper_squares_below_line
  (h : ∀ (x y : ℕ), 12 * x + 247 * y = 2976)
  (square_size : ℕ) 
  (xs : ℕ) (ys : ℕ)
  (line_eq : ∀ (x y : ℕ), y = 247 * x / 12)
  (n_squares : ℕ) :
  n_squares = 1358
  := by
    sorry

end graph_paper_squares_below_line_l1708_170874


namespace Carly_applications_l1708_170842

theorem Carly_applications (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : x + 2 * x = 600) : x = 200 :=
sorry

end Carly_applications_l1708_170842


namespace find_n_l1708_170858

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (S_odd : ℝ) (S_even : ℝ)
  (h1 : ∀ k, a (2 * k - 1) = a 0 + (2 * k - 2) * d)
  (h2 : ∀ k, a (2 * k) = a 1 + (2 * k - 1) * d)
  (h3 : 2 * n + 1 = n + (n + 1))
  (h4 : S_odd = (n + 1) * (a 0 + n * d))
  (h5 : S_even = n * (a 1 + (n - 1) * d))
  (h6 : S_odd = 4)
  (h7 : S_even = 3) : n = 3 :=
by
  sorry

end find_n_l1708_170858


namespace proof_problem_l1708_170805

variables {a b c : Real}

theorem proof_problem (h1 : a < 0) (h2 : |a| < |b|) (h3 : |b| < |c|) (h4 : b < 0) :
  (|a * b| < |b * c|) ∧ (a * c < |b * c|) ∧ (|a + b| < |b + c|) :=
by
  sorry

end proof_problem_l1708_170805


namespace distance_between_points_A_and_B_is_240_l1708_170831

noncomputable def distance_between_A_and_B (x y : ℕ) : ℕ := 6 * x * 2

theorem distance_between_points_A_and_B_is_240 (x y : ℕ)
  (h1 : 6 * x = 6 * y)
  (h2 : 5 * (x + 4) = 6 * y) :
  distance_between_A_and_B x y = 240 := by
  sorry

end distance_between_points_A_and_B_is_240_l1708_170831


namespace three_nabla_four_l1708_170832

noncomputable def modified_operation (a b : ℝ) : ℝ :=
  (a + b^2) / (1 + a * b^2)

theorem three_nabla_four : modified_operation 3 4 = 19 / 49 := 
  by 
  sorry

end three_nabla_four_l1708_170832


namespace find_a_l1708_170862

theorem find_a (x y z a : ℝ) (k : ℝ) (h1 : x = 2 * k) (h2 : y = 3 * k) (h3 : z = 5 * k)
    (h4 : x + y + z = 100) (h5 : y = a * x - 10) : a = 2 :=
  sorry

end find_a_l1708_170862


namespace polyhedron_volume_is_correct_l1708_170894

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 12
  let num_squares := 3
  let square_area := side_length * side_length
  let cube_volume := side_length ^ 3
  let polyhedron_volume := cube_volume / 2
  polyhedron_volume

theorem polyhedron_volume_is_correct :
  volume_of_polyhedron = 864 :=
by
  sorry

end polyhedron_volume_is_correct_l1708_170894


namespace exists_congruent_triangle_covering_with_parallel_side_l1708_170861

variable {Point : Type}
variable [MetricSpace Point]
variable {Triangle : Type}
variable {Polygon : Type}

-- Definitions of triangle and polygon covering relationships.
def covers (T : Triangle) (P : Polygon) : Prop := sorry 
def congruent (T1 T2 : Triangle) : Prop := sorry
def side_parallel_or_coincident (T : Triangle) (P : Polygon) : Prop := sorry

-- Statement: Given a triangle covering a polygon, there exists a congruent triangle which covers the polygon 
-- and has one side parallel to or coincident with a side of the polygon.
theorem exists_congruent_triangle_covering_with_parallel_side 
  (ABC : Triangle) (M : Polygon) 
  (h_cover : covers ABC M) : 
  ∃ Δ : Triangle, congruent Δ ABC ∧ covers Δ M ∧ side_parallel_or_coincident Δ M := 
sorry

end exists_congruent_triangle_covering_with_parallel_side_l1708_170861


namespace photograph_perimeter_l1708_170879

-- Definitions of the conditions
def photograph_is_rectangular : Prop := True
def one_inch_border_area (w l m : ℕ) : Prop := (w + 2) * (l + 2) = m
def three_inch_border_area (w l m : ℕ) : Prop := (w + 6) * (l + 6) = m + 52

-- Lean statement of the problem
theorem photograph_perimeter (w l m : ℕ) 
  (h1 : photograph_is_rectangular)
  (h2 : one_inch_border_area w l m)
  (h3 : three_inch_border_area w l m) : 
  2 * (w + l) = 10 := 
by 
  sorry

end photograph_perimeter_l1708_170879


namespace modular_inverse_addition_l1708_170864

theorem modular_inverse_addition :
  (3 * 9 + 9 * 37) % 63 = 45 :=
by
  sorry

end modular_inverse_addition_l1708_170864


namespace p_sq_plus_q_sq_l1708_170813

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := by
  sorry

end p_sq_plus_q_sq_l1708_170813


namespace points_in_quadrants_l1708_170877

theorem points_in_quadrants :
  ∀ (x y : ℝ), (y > 3 * x) → (y > 5 - 2 * x) → ((0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)) :=
by
  intros x y h1 h2
  sorry

end points_in_quadrants_l1708_170877


namespace find_second_number_l1708_170840

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end find_second_number_l1708_170840


namespace initial_crayons_count_l1708_170807

theorem initial_crayons_count (C : ℕ) :
  (3 / 8) * C = 18 → C = 48 :=
by
  sorry

end initial_crayons_count_l1708_170807


namespace subset_range_l1708_170809

open Set

-- Definitions of sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- The statement of the problem
theorem subset_range (a : ℝ) (h : A ⊆ B a) : 2 ≤ a :=
sorry -- Skipping the proof

end subset_range_l1708_170809


namespace total_length_of_sticks_l1708_170800

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l1708_170800


namespace find_the_number_l1708_170869

theorem find_the_number (x : ℕ) (h : x * 9999 = 4691110842) : x = 469211 := by
    sorry

end find_the_number_l1708_170869


namespace red_apples_count_l1708_170880

theorem red_apples_count
  (r y g : ℕ)
  (h1 : r = y)
  (h2 : g = 2 * r)
  (h3 : r + y + g = 28) : r = 7 :=
sorry

end red_apples_count_l1708_170880


namespace narrow_black_stripes_l1708_170873

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l1708_170873


namespace angle_complementary_supplementary_l1708_170830

theorem angle_complementary_supplementary (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle1 + angle3 = 180)
  (h3 : angle3 = 125) :
  angle2 = 35 :=
by 
  sorry

end angle_complementary_supplementary_l1708_170830


namespace pipe_Q_drain_portion_l1708_170886

noncomputable def portion_liquid_drain_by_Q (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) : ℝ :=
  let rate_P := 1 / T_P
  let rate_Q := 1 / T_Q
  let rate_R := 1 / T_R
  let combined_rate := rate_P + rate_Q + rate_R
  (rate_Q / combined_rate)

theorem pipe_Q_drain_portion (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) :
  portion_liquid_drain_by_Q T_Q T_P T_R h1 h2 = 3 / 11 :=
by
  sorry

end pipe_Q_drain_portion_l1708_170886


namespace vector_operation_result_l1708_170872

variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C O E : V)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = (A - E) :=
by
  sorry

end vector_operation_result_l1708_170872


namespace function_inequality_l1708_170836

variable {f : ℕ → ℝ}
variable {a : ℝ}

theorem function_inequality (h : ∀ n : ℕ, f (n + 1) ≥ a^n * f n) :
  ∀ n : ℕ, f n = a^((n * (n - 1)) / 2) * f 1 := 
sorry

end function_inequality_l1708_170836


namespace increasing_sequence_range_l1708_170828

theorem increasing_sequence_range (a : ℝ) (a_seq : ℕ → ℝ)
  (h₁ : ∀ (n : ℕ), n ≤ 5 → a_seq n = (5 - a) * n - 11)
  (h₂ : ∀ (n : ℕ), n > 5 → a_seq n = a ^ (n - 4))
  (h₃ : ∀ (n : ℕ), a_seq n < a_seq (n + 1)) :
  2 < a ∧ a < 5 := 
sorry

end increasing_sequence_range_l1708_170828


namespace sum_of_roots_l1708_170812

theorem sum_of_roots (a1 a2 a3 a4 a5 : ℤ)
  (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
                a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
                a3 ≠ a4 ∧ a3 ≠ a5 ∧
                a4 ≠ a5)
  (h_poly : (104 - a1) * (104 - a2) * (104 - a3) * (104 - a4) * (104 - a5) = 2012) :
  a1 + a2 + a3 + a4 + a5 = 17 := by
  sorry

end sum_of_roots_l1708_170812


namespace not_square_a2_b2_ab_l1708_170820

theorem not_square_a2_b2_ab (n : ℕ) (h_n : n > 2) (a : ℕ) (b : ℕ) (h_b : b = 2^(2^n))
  (h_a_odd : a % 2 = 1) (h_a_le_b : a ≤ b) (h_b_le_2a : b ≤ 2 * a) :
  ¬ ∃ k : ℕ, a^2 + b^2 - a * b = k^2 :=
by
  sorry

end not_square_a2_b2_ab_l1708_170820


namespace simplify_expression_l1708_170801

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  (3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6)) / ((x - 3) * (x + 2)) :=
by
  sorry

end simplify_expression_l1708_170801


namespace monotonically_increasing_interval_l1708_170856

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem monotonically_increasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → (f x > f 0) := 
by
  sorry

end monotonically_increasing_interval_l1708_170856


namespace problem_divisibility_l1708_170853

theorem problem_divisibility (k : ℕ) (hk : k > 1) (p : ℕ) (hp : p = 6 * k + 1) (hprime : Prime p) 
  (m : ℕ) (hm : m = 2^p - 1) : 
  127 * m ∣ 2^(m - 1) - 1 := 
sorry

end problem_divisibility_l1708_170853


namespace reception_time_l1708_170896

-- Definitions of conditions
def noon : ℕ := 12 * 60 -- define noon in minutes
def rabbit_walk_speed (v : ℕ) : Prop := v > 0
def rabbit_run_speed (v : ℕ) : Prop := 2 * v > 0
def distance (D : ℕ) : Prop := D > 0
def delay (minutes : ℕ) : Prop := minutes = 10

-- Definition of the problem
theorem reception_time (v D : ℕ) (h_v : rabbit_walk_speed v) (h_D : distance D) (h_delay : delay 10) :
  noon + (D / v) * 2 / 3 = 12 * 60 + 40 :=
by sorry

end reception_time_l1708_170896


namespace chessboard_L_T_equivalence_l1708_170889

theorem chessboard_L_T_equivalence (n : ℕ) :
  ∃ L_count T_count : ℕ, 
  (L_count = T_count) ∧ -- number of L-shaped pieces is equal to number of T-shaped pieces
  (L_count + T_count = n * (n + 1)) := 
sorry

end chessboard_L_T_equivalence_l1708_170889


namespace andrew_apples_l1708_170881

theorem andrew_apples : ∃ (A n : ℕ), (6 * n = A) ∧ (5 * (n + 2) = A) ∧ (A = 60) :=
by 
  sorry

end andrew_apples_l1708_170881


namespace least_positive_integer_l1708_170863

theorem least_positive_integer (k : ℕ) (h : (528 + k) % 5 = 0) : k = 2 :=
sorry

end least_positive_integer_l1708_170863


namespace value_of_expression_l1708_170875

theorem value_of_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 :=
by sorry

end value_of_expression_l1708_170875


namespace smallest_positive_period_of_f_l1708_170885

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi / 2 :=
by
  sorry

end smallest_positive_period_of_f_l1708_170885


namespace fraction_pattern_l1708_170803

theorem fraction_pattern (n m k : ℕ) (h : n / m = k * n / (k * m)) : (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end fraction_pattern_l1708_170803


namespace average_weight_ten_students_l1708_170884

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l1708_170884


namespace common_tangent_line_range_a_l1708_170829

open Real

theorem common_tangent_line_range_a (a : ℝ) (h_pos : 0 < a) :
  (∃ x₁ x₂ : ℝ, 2 * a * x₁ = exp x₂ ∧ (exp x₂ - a * x₁^2) / (x₂ - x₁) = 2 * a * x₁) →
  a ≥ exp 2 / 4 := 
sorry

end common_tangent_line_range_a_l1708_170829


namespace no_infinite_harmonic_mean_sequence_l1708_170876

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), (∀ n, a n = a 0 → False) ∧
                   (∀ i, 1 ≤ i → a i = (2 * a (i - 1) * a (i + 1)) / (a (i - 1) + a (i + 1))) :=
sorry

end no_infinite_harmonic_mean_sequence_l1708_170876


namespace tan_to_trig_identity_l1708_170892

theorem tan_to_trig_identity (α : ℝ) (h : Real.tan α = 3) : (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
by
  sorry

end tan_to_trig_identity_l1708_170892


namespace lilith_caps_collection_l1708_170860

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l1708_170860


namespace vanessa_deleted_files_l1708_170844

theorem vanessa_deleted_files (initial_music_files : ℕ) (initial_video_files : ℕ) (files_left : ℕ) (files_deleted : ℕ) :
  initial_music_files = 13 → initial_video_files = 30 → files_left = 33 → 
  files_deleted = (initial_music_files + initial_video_files) - files_left → files_deleted = 10 :=
by
  sorry

end vanessa_deleted_files_l1708_170844


namespace loss_percentage_first_book_l1708_170849

theorem loss_percentage_first_book (C1 C2 : ℝ) 
    (total_cost : ℝ) 
    (gain_percentage : ℝ)
    (S1 S2 : ℝ)
    (cost_first_book : C1 = 175)
    (total_cost_condition : total_cost = 300)
    (gain_condition : gain_percentage = 0.19)
    (same_selling_price : S1 = S2)
    (second_book_cost : C2 = total_cost - C1)
    (selling_price_second_book : S2 = C2 * (1 + gain_percentage)) :
    (C1 - S1) / C1 * 100 = 15 :=
by
  sorry

end loss_percentage_first_book_l1708_170849


namespace tangent_line_eq_l1708_170824

theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x - 2 * Real.log x) :
  ∃ m b, (m = -1) ∧ (b = 2) ∧ (∀ x, f x = m * x + b) :=
by
  sorry

end tangent_line_eq_l1708_170824


namespace total_height_of_buildings_l1708_170851

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l1708_170851


namespace num_students_B_l1708_170866

-- Define the given conditions
variables (x : ℕ) -- The number of students who get a B

noncomputable def number_of_A := 2 * x
noncomputable def number_of_C := (12 / 10 : ℤ) * x -- Using (12 / 10) to approximate 1.2 in integers

-- Given total number of students is 42 for integer result
def total_students := 42

-- Lean statement to show number of students getting B is 10
theorem num_students_B : 4.2 * (x : ℝ) = 42 → x = 10 :=
by
  sorry

end num_students_B_l1708_170866


namespace problem_statement_l1708_170818

theorem problem_statement (k : ℕ) (h : 35^k ∣ 1575320897) : 7^k - k^7 = 1 := by
  sorry

end problem_statement_l1708_170818


namespace calculate_bankers_discount_l1708_170822

noncomputable def present_worth : ℝ := 800
noncomputable def true_discount : ℝ := 36
noncomputable def face_value : ℝ := present_worth + true_discount
noncomputable def bankers_discount : ℝ := (face_value * true_discount) / (face_value - true_discount)

theorem calculate_bankers_discount :
  bankers_discount = 37.62 := 
sorry

end calculate_bankers_discount_l1708_170822


namespace principal_amount_l1708_170854

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 5)

theorem principal_amount :
  ((P * R * T) / 100 = P - 2000) → P = 2500 :=
by
  sorry

end principal_amount_l1708_170854


namespace total_cans_needed_l1708_170817

-- Definitions
def cans_per_box : ℕ := 4
def number_of_boxes : ℕ := 203

-- Statement of the problem
theorem total_cans_needed : cans_per_box * number_of_boxes = 812 := 
by
  -- skipping the proof
  sorry

end total_cans_needed_l1708_170817


namespace total_capacity_both_dressers_l1708_170870

/-- Definition of drawers and capacity -/
def first_dresser_drawers : ℕ := 12
def first_dresser_capacity_per_drawer : ℕ := 8
def second_dresser_drawers : ℕ := 6
def second_dresser_capacity_per_drawer : ℕ := 10

/-- Theorem stating the total capacity of both dressers -/
theorem total_capacity_both_dressers :
  (first_dresser_drawers * first_dresser_capacity_per_drawer) +
  (second_dresser_drawers * second_dresser_capacity_per_drawer) = 156 :=
by sorry

end total_capacity_both_dressers_l1708_170870


namespace girls_doctors_percentage_l1708_170816

-- Define the total number of students in the class
variables (total_students : ℕ)

-- Define the proportions given in the problem
def proportion_boys : ℚ := 3 / 5
def proportion_boys_who_want_to_be_doctors : ℚ := 1 / 3
def proportion_doctors_who_are_boys : ℚ := 2 / 5

-- Compute the proportion of boys in the class who want to be doctors
def proportion_boys_as_doctors := proportion_boys * proportion_boys_who_want_to_be_doctors

-- Compute the proportion of girls in the class
def proportion_girls := 1 - proportion_boys

-- Compute the number of girls who want to be doctors compared to boys
def proportion_girls_as_doctors := (1 - proportion_doctors_who_are_boys) / proportion_doctors_who_are_boys * proportion_boys_as_doctors

-- Compute the proportion of girls who want to be doctors
def proportion_girls_who_want_to_be_doctors := proportion_girls_as_doctors / proportion_girls

-- Define the expected percentage of girls who want to be doctors
def expected_percentage_girls_who_want_to_be_doctors : ℚ := 75 / 100

-- The theorem we need to prove
theorem girls_doctors_percentage : proportion_girls_who_want_to_be_doctors * 100 = expected_percentage_girls_who_want_to_be_doctors :=
sorry

end girls_doctors_percentage_l1708_170816


namespace intersection_of_A_and_B_l1708_170810

def setA : Set ℝ := { x | x ≤ 4 }
def setB : Set ℝ := { x | x ≥ 1/2 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | 1/2 ≤ x ∧ x ≤ 4 } := by
  sorry

end intersection_of_A_and_B_l1708_170810


namespace solve_eq_sqrt_exp_l1708_170827

theorem solve_eq_sqrt_exp :
  (∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) → (x = 2 ∨ x = -1)) :=
by
  -- Prove that the solutions are x = 2 and x = -1
  sorry

end solve_eq_sqrt_exp_l1708_170827


namespace max_x_for_integer_fraction_l1708_170847

theorem max_x_for_integer_fraction (x : ℤ) (h : ∃ k : ℤ, x^2 + 2 * x + 11 = k * (x - 3)) : x ≤ 29 :=
by {
    -- This is where the proof would be,
    -- but we skip the proof per the instructions.
    sorry
}

end max_x_for_integer_fraction_l1708_170847
