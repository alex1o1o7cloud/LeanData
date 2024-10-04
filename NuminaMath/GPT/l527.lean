import Mathlib

namespace find_omega_phi_l527_527011

def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ) - 1

theorem find_omega_phi
  (h1 : ∀ x, f ω φ x ∈ ℝ)
  (h2 : 0 < ω)
  (h3 : |φ| < π)
  (h4 : f ω φ (5 * π / 8) = 1)
  (h5 : f ω φ (11 * π / 8) = -1)
  (h6 : ∃ T > 2 * π, ∀ x, f ω φ (x + T) = f ω φ x) :
  ω = 2/3 ∧ φ = π/12 :=
sorry

end find_omega_phi_l527_527011


namespace area_below_line_l527_527107

-- Define the conditions provided in the problem.
def graph_eq (x y : ℝ) : Prop := x^2 - 14*x + 3*y + 70 = 21 + 11*y - y^2
def line_eq (x y : ℝ) : Prop := y = x - 3

-- State the final proof problem which is to find the area under the given conditions.
theorem area_below_line :
  ∃ area : ℝ, area = 8 * Real.pi ∧ 
  (∀ x y, graph_eq x y → y ≤ x - 3 → -area / 2 ≤ y ∧ y ≤ area / 2) := 
sorry

end area_below_line_l527_527107


namespace Seth_boxes_initially_l527_527038

-- Define the initial conditions
def initial_boxes (x : ℕ) : Prop :=
  ∃ n : ℕ, 4 = (x - 1) / 2 ∧ x = 9

-- Prove that Seth initially bought 9 boxes of oranges
theorem Seth_boxes_initially : initial_boxes 9 :=
by {
  use 9,
  split,
  exact rfl,
  exact rfl,
}

end Seth_boxes_initially_l527_527038


namespace minimum_trips_l527_527675

def masses : List ℕ := [50, 51, 55, 57, 58, 59, 60, 63, 75, 140]
def load_capacity : ℕ := 180

theorem minimum_trips (masses : List ℕ) (load_capacity : ℕ) (h_masses : masses = [50, 51, 55, 57, 58, 59, 60, 63, 75, 140]) (h_load_capacity : load_capacity = 180) :
  ∃ n : ℕ, n = 4 ∧ (∀ trips : List (List ℕ), (∀ trip ∈ trips, (∑ m in trip, m) ≤ load_capacity ∧ trip ⊆ masses) → (∑ t in trips, t.length) = masses.length → trips.length = n) :=
by
  sorry

end minimum_trips_l527_527675


namespace inequality_solution_l527_527780

noncomputable def solution_set (x : ℝ) : Prop := 
  (x < -1) ∨ (x > 3)

theorem inequality_solution :
  { x : ℝ | (3 - x) / (x + 1) < 0 } = { x : ℝ | solution_set x } :=
by
  sorry

end inequality_solution_l527_527780


namespace total_letters_in_names_is_33_l527_527696

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l527_527696


namespace solve_triangles_problem_l527_527456

noncomputable def triangles_problem : Prop :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (1 : ℝ, Real.sqrt 3)
  let D := (x : ℝ, 0)
  let E := (1 : ℝ, y : ℝ)
  let F := (t : ℝ, t * Real.sqrt 3)
  ∃ (a b : ℕ), [DEF] = 1 / 2 * |t * Real.sqrt 3 * (1 - x) - y * (t - x)|
  ∧ (a + b = 67)

theorem solve_triangles_problem : triangles_problem := 
  sorry

end solve_triangles_problem_l527_527456


namespace common_root_implies_condition_l527_527268

noncomputable theory

-- Define the polynomial x^3 - 1
def cubic_poly (x : ℂ) : ℂ := x^3 - 1 

-- Define the given polynomial a x^{3n+2} + b x^{3m+1} + c
def poly (a b c : ℂ) (n m : ℕ) (x : ℂ) : ℂ := a * x^(3 * n + 2) + b * x^(3 * m + 1) + c 

-- State the main theorem
theorem common_root_implies_condition (a b c : ℂ) (n m : ℕ) : 
  (∃ x : ℂ, cubic_poly x = 0 ∧ poly a b c n m x = 0) → a^3 + b^3 + c^3 - 3 * a * b * c = 0 :=
by {
  sorry
}

end common_root_implies_condition_l527_527268


namespace Beth_peas_count_l527_527543

-- Definitions based on conditions
def number_of_corn : ℕ := 10
def number_of_peas (number_of_corn : ℕ) : ℕ := 2 * number_of_corn + 15

-- Theorem that represents the proof problem
theorem Beth_peas_count : number_of_peas 10 = 35 :=
by
  sorry

end Beth_peas_count_l527_527543


namespace wendy_total_albums_l527_527104

theorem wendy_total_albums : ∀ (total_pics pics_in_first_album pics_per_other_album : ℕ), 
  total_pics = 79 ∧ pics_in_first_album = 44 ∧ pics_per_other_album = 7 →
  (total_pics - pics_in_first_album) / pics_per_other_album + 1 = 6 := 
by {
  intros total_pics pics_in_first_album pics_per_other_album h,
  obtain ⟨ht, hf, hp⟩ := h,
  rw [ht, hf, hp],
  exact div_eq := eq.refl _,
  sorry
}

end wendy_total_albums_l527_527104


namespace number_of_unique_paths_l527_527568

structure Tetrahedron :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (vertices_card : vertices.card = 4)
  (edges_card : edges.card = 6)
  (edge_connect : ∀ e ∈ edges, e.1 ≠ e.2 ∧ e.1 ∈ vertices ∧ e.2 ∈ vertices)

def paths_using_all_edges_exactly_once (t : Tetrahedron) : Finset (Finset (ℕ × ℕ)) :=
  (t.edges.powerset.filter (λ ps, ps.card = 6)) -- Filters paths using all 6 edges

theorem number_of_unique_paths (t : Tetrahedron) (start_vertex : ℕ) (hv : start_vertex ∈ t.vertices) :
  (paths_using_all_edges_exactly_once t).card = 6 :=
by
  sorry

end number_of_unique_paths_l527_527568


namespace paul_coins_difference_l527_527738

/-- Paul owes Paula 145 cents and has a pocket full of 10-cent coins, 
20-cent coins, and 50-cent coins. Prove that the difference between 
the largest and smallest number of coins he can use to pay her is 9. -/
theorem paul_coins_difference :
  ∃ min_coins max_coins : ℕ, 
    (min_coins = 5 ∧ max_coins = 14) ∧ (max_coins - min_coins = 9) :=
by
  sorry

end paul_coins_difference_l527_527738


namespace Ram_Gohul_days_work_together_l527_527659

-- Define the conditions
def Ram_days := 10
def Gohul_days := 15

-- Define the work rates
def Ram_rate := 1 / Ram_days
def Gohul_rate := 1 / Gohul_days

-- Define the combined work rate
def Combined_rate := Ram_rate + Gohul_rate

-- Define the number of days to complete the job together
def Together_days := 1 / Combined_rate

-- State the proof problem
theorem Ram_Gohul_days_work_together : Together_days = 6 := by
  sorry

end Ram_Gohul_days_work_together_l527_527659


namespace weight_of_new_student_l527_527129

-- Definitions from conditions
def total_weight_19 : ℝ := 19 * 15
def total_weight_20 : ℝ := 20 * 14.9

-- Theorem to prove the weight of the new student
theorem weight_of_new_student : (total_weight_20 - total_weight_19) = 13 := by
  sorry

end weight_of_new_student_l527_527129


namespace sequence_geometric_condition_l527_527333

theorem sequence_geometric_condition
  (a : ℕ → ℤ)
  (p q : ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = 2 * (a n - n + 3))
  (h3 : ∀ n, (a (n + 1) - p * (n + 1) + q) = 2 * (a n - p * n + q)) :
  a (Int.natAbs (p + q)) = 40 :=
sorry

end sequence_geometric_condition_l527_527333


namespace total_earnings_l527_527825

noncomputable def daily_wage_a (C : ℝ) := (3 * C) / 5
noncomputable def daily_wage_b (C : ℝ) := (4 * C) / 5
noncomputable def daily_wage_c (C : ℝ) := C

noncomputable def earnings_a (C : ℝ) := daily_wage_a C * 6
noncomputable def earnings_b (C : ℝ) := daily_wage_b C * 9
noncomputable def earnings_c (C : ℝ) := daily_wage_c C * 4

theorem total_earnings (C : ℝ) (h : C = 115) : 
  earnings_a C + earnings_b C + earnings_c C = 1702 :=
by
  sorry

end total_earnings_l527_527825


namespace cos_angle_AHB_l527_527720

-- Define the points and vectors in a triangle
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C H : V)

-- Define the orthocenter condition
def ortho_condition : Prop :=
  3 • (H - A) + 4 • (H - B) + 5 • (H - C) = (0 : V)

-- State the problem as a theorem
theorem cos_angle_AHB (h : ortho_condition A B C H) :
  cos (angle A H B) = -√6 / 6 :=
sorry

end cos_angle_AHB_l527_527720


namespace number_of_tiles_in_each_row_l527_527760

-- Conditions as Lean definitions
def area_of_room : ℝ := 144
def length_is_twice_width (width : ℝ) := 2 * width
def tile_size : ℝ := 4 -- in inches

-- Goal statement
theorem number_of_tiles_in_each_row (width : ℝ) (h1: 2 * width^2 = area_of_room) 
(h2: tile_size > 0): 
  ∃ n: ℕ, n = ⌊((12 * width) * (2^0.5)) / tile_size⌋ ∧ n = 25 := sorry

end number_of_tiles_in_each_row_l527_527760


namespace integer_value_of_k_l527_527590

theorem integer_value_of_k (k : ℤ) :
  (∃ x1 x2 : ℝ, x1 = x2 / 3 ∧ ∃ (p : 4 * x1^2 - (3 * k + 2) * x1 + (k^2 - 1) = 0), 
    ∃ (q : 4 * x2^2 - (3 * k + 2) * x2 + (k^2 - 1) = 0) ) → k = 2 :=
by
  sorry

end integer_value_of_k_l527_527590


namespace icosahedron_projection_theorem_dodecahedron_projection_theorem_l527_527824

noncomputable def icosahedron_projection_is_regular_decagon : Prop :=
  ∀ (plane : Type) (h_plane_perpendicular : plane ⊥ (ℝ^3)) (center vertex : ℝ^3),
    (line_through center vertex) ⊥ plane → 
    projection (icosahedron) plane = regular_decagon

noncomputable def dodecahedron_projection_is_irregular_dodecagon : Prop :=
  ∀ (plane : Type) (h_plane_perpendicular: plane ⊥ (ℝ^3)) (center vertex : ℝ^3), 
    (line_through center vertex) ⊥ plane → 
    projection (dodecahedron) plane = irregular_dodecagon

theorem icosahedron_projection_theorem : icosahedron_projection_is_regular_decagon :=
  sorry

theorem dodecahedron_projection_theorem : dodecahedron_projection_is_irregular_dodecagon :=
  sorry

end icosahedron_projection_theorem_dodecahedron_projection_theorem_l527_527824


namespace value_of_f_at_5_l527_527768

def f (x : ℤ) : ℤ := x^3 - x^2 + x

theorem value_of_f_at_5 : f 5 = 105 := by
  sorry

end value_of_f_at_5_l527_527768


namespace reciprocal_of_repeating_2_l527_527795

theorem reciprocal_of_repeating_2 : 
  let d := 2
  let repeating_decimal (d : ℕ) := d / 9
  let reciprocal (m : ℚ) := m⁻¹ 
  (repeating_decimal 2)⁻¹ = 9 / 2 :=
by
  sorry

end reciprocal_of_repeating_2_l527_527795


namespace platform_length_l527_527493

variable (L : ℝ) -- The length of the platform
variable (train_length : ℝ := 300) -- The length of the train
variable (time_pole : ℝ := 26) -- Time to cross the signal pole
variable (time_platform : ℝ := 39) -- Time to cross the platform

theorem platform_length :
  (train_length / time_pole) = (train_length + L) / time_platform → L = 150 := sorry

end platform_length_l527_527493


namespace problem1_problem2_problem3_problem4_problem5_l527_527792

-- Definitions and conditions
variable (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a - 2 * Real.sqrt b > 0)

-- Problem 1: √(a - 2√b) = √m - √n
theorem problem1 (h₁ : a = 5) (h₂ : b = 6) : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := sorry

-- Problem 2: √(a + 2√b) = √m + √n
theorem problem2 (h₁ : a = 12) (h₂ : b = 35) : Real.sqrt (12 + 2 * Real.sqrt 35) = Real.sqrt 7 + Real.sqrt 5 := sorry

-- Problem 3: √(a + 6√b) = √m + √n
theorem problem3 (h₁ : a = 9) (h₂ : b = 6) : Real.sqrt (9 + 6 * Real.sqrt 2) = Real.sqrt 6 + Real.sqrt 3 := sorry

-- Problem 4: √(a - 4√b) = √m - √n
theorem problem4 (h₁ : a = 16) (h₂ : b = 60) : Real.sqrt (16 - 4 * Real.sqrt 15) = Real.sqrt 10 - Real.sqrt 6 := sorry

-- Problem 5: √(a - √b) + √(c + √d)
theorem problem5 (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 2) (h₄ : d = 3) 
  : Real.sqrt (3 - Real.sqrt 5) + Real.sqrt (2 + Real.sqrt 3) = (Real.sqrt 10 + Real.sqrt 6) / 2 := sorry

end problem1_problem2_problem3_problem4_problem5_l527_527792


namespace probability_product_divisible_by_8_l527_527372

theorem probability_product_divisible_by_8 :
  let probability := (554 : ℚ) / 576 in
  ∀ (dice_rolls : Fin 8 → Fin 6), probability = 
    let product := (∏ i, (dice_rolls i).val + 1) in
    (if product % 8 = 0 then 1 else 0) :=
sorry

end probability_product_divisible_by_8_l527_527372


namespace sum_of_reciprocals_of_distinct_primes_not_integer_or_reciprocal_l527_527740

theorem sum_of_reciprocals_of_distinct_primes_not_integer_or_reciprocal (n : ℕ) (p : ℕ → ℕ)
  (h₀ : ∀ i j : ℕ, i < n → j < n → i ≠ j → p i ≠ p j)
  (h₁ : ∀ i < n, Nat.Prime (p i)) :
  (∑ i in Finset.range n, 1 / (p i) : ℚ) ∉ ℤ ∧ (∑ i in Finset.range n, 1 / (p i) : ℚ) ∉ (λ k : ℤ, (1 / k : ℚ)) :=
by
  sorry

end sum_of_reciprocals_of_distinct_primes_not_integer_or_reciprocal_l527_527740


namespace zongzi_cost_per_bag_first_batch_l527_527736

theorem zongzi_cost_per_bag_first_batch (x : ℝ)
  (h1 : 7500 / (x - 4) = 3 * (3000 / x))
  (h2 : 3000 > 0)
  (h3 : 7500 > 0)
  (h4 : x > 4) :
  x = 24 :=
by sorry

end zongzi_cost_per_bag_first_batch_l527_527736


namespace intersection_correct_l527_527368

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x < 2}

theorem intersection_correct : P ∩ Q = {1} :=
by sorry

end intersection_correct_l527_527368


namespace graphs_intersection_count_l527_527050

theorem graphs_intersection_count (g : ℝ → ℝ) (hg : Function.Injective g) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (x^3) = g (x^5)) ∧ S.card = 3 :=
by
  sorry

end graphs_intersection_count_l527_527050


namespace pipe_5_fill_time_48_hours_l527_527169

variables {r1 r2 r3 r4 r5 : ℝ}

def tank_fill_eq_1 : Prop := r1 + r2 + r3 + r4 = 1/6
def tank_fill_eq_2 : Prop := r2 + r3 + r4 + r5 = 1/8
def tank_fill_eq_3 : Prop := r1 + r5 = 1/12
def pipe_5_rate : Prop := r5 = 1/48
def tank_fill_time : Prop := 48 = 1 / r5

theorem pipe_5_fill_time_48_hours 
  (h1 : tank_fill_eq_1) 
  (h2 : tank_fill_eq_2) 
  (h3 : tank_fill_eq_3) 
  : tank_fill_time := 
by 
  sorry

end pipe_5_fill_time_48_hours_l527_527169


namespace minimize_dot_product_l527_527642

def vector := ℝ × ℝ

def OA : vector := (2, 2)
def OB : vector := (4, 1)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def AP (P : vector) : vector :=
  (P.1 - OA.1, P.2 - OA.2)

def BP (P : vector) : vector :=
  (P.1 - OB.1, P.2 - OB.2)

def is_on_x_axis (P : vector) : Prop :=
  P.2 = 0

theorem minimize_dot_product :
  ∃ (P : vector), is_on_x_axis P ∧ dot_product (AP P) (BP P) = ( (P.1 - 3) ^ 2 + 1) ∧ P = (3, 0) :=
by
  sorry

end minimize_dot_product_l527_527642


namespace slope_ON_existence_theta_l527_527608

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) := 
  {p : ℝ × ℝ // (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

noncomputable def on_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ) :=
  (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1

noncomputable def right_focus (a b : ℝ) (h : a > b ∧ b > 0) : ℝ × ℝ := 
  (sqrt (a^2 - b^2)), 0

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Problem 1: Prove the slope of ON is -1/3
theorem slope_ON (a b : ℝ) (h : a > b ∧ b > 0)
  (h_eccentricity : sqrt 6 / 3 = sqrt (a^2 - b^2) / a)
  (A B : ℝ × ℝ) (hA : on_ellipse a b h A) (hB : on_ellipse a b h B)
  (N : ℝ × ℝ := midpoint A B) : (N.2 / N.1) = -1/3 := sorry

-- Problem 2: Prove the existence of θ such that OM = cos θ OA + sin θ OB 
theorem existence_theta (a b : ℝ) (h : a > b ∧ b > 0)
  (A B : ℝ × ℝ) (hA : on_ellipse a b h A) (hB : on_ellipse a b h B)
  (M : ℝ × ℝ) (hM : on_ellipse a b h M) :
  ∃ θ : ℝ, ∀ x y : ℝ, ((x, y) = M) → (M.1, M.2) = 
  ((cos θ * A.1 + sin θ * B.1), (cos θ * A.2 + sin θ * B.2)) := sorry

end slope_ON_existence_theta_l527_527608


namespace samantha_total_cost_l527_527036

noncomputable def daily_rental_rate : ℝ := 30
noncomputable def daily_rental_days : ℝ := 3
noncomputable def cost_per_mile : ℝ := 0.15
noncomputable def miles_driven : ℝ := 500

theorem samantha_total_cost :
  (daily_rental_rate * daily_rental_days) + (cost_per_mile * miles_driven) = 165 :=
by
  sorry

end samantha_total_cost_l527_527036


namespace hyperbola_eccentricity_l527_527259

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h_a_pos : 0 < a) 
  (h_b_pos : 0 < b)
  (h_asymptotes : (∀ x, (∃ y, y = (b / a) * x) ∨ (∃ y, y = - (b / a) * x)))
  (h_f : ∃ F : ℝ × ℝ, F = (c, 0))
  (h_FA_perpendicular_l1 : ∀ A : ℝ × ℝ, (∃ x, x = (A.1, (b/a) * A.1)) → ⟪ (a - fst F), (b - snd F) ⟫ = 0)
  (h_FB_parallel_l1 : ∀ B : ℝ × ℝ, (∃ x, x = (B.1, - (b/a) * B.1)) → ⟪ (a - fst F), 0 ⟫ = (b/a) * (B.1 - c))
  (h_FA_FB_ratio : ∀ (FA FB : ℝ), FA = (b / real.sqrt ((b^2/a^2) + 1)) → FB = real.sqrt ((c^2 / 4) + (b^2 * c^2 / (4 * a^2))) → FA = (4/5) * FB) :
  let e := real.sqrt (1 + (b^2 / a^2))
  in e = (sqrt 5 / 2) :=
sorry

end hyperbola_eccentricity_l527_527259


namespace packs_of_yellow_bouncy_balls_l527_527726

-- Define the conditions and the question in Lean
variables (GaveAwayGreen : ℝ) (BoughtGreen : ℝ) (BouncyBallsPerPack : ℝ) (TotalKeptBouncyBalls : ℝ) (Y : ℝ)

-- Assume the given conditions
axiom cond1 : GaveAwayGreen = 4.0
axiom cond2 : BoughtGreen = 4.0
axiom cond3 : BouncyBallsPerPack = 10.0
axiom cond4 : TotalKeptBouncyBalls = 80.0

-- Define the theorem statement
theorem packs_of_yellow_bouncy_balls (h1 : GaveAwayGreen = 4.0) (h2 : BoughtGreen = 4.0) (h3 : BouncyBallsPerPack = 10.0) (h4 : TotalKeptBouncyBalls = 80.0) : Y = 8 :=
sorry

end packs_of_yellow_bouncy_balls_l527_527726


namespace volume_of_set_l527_527199

theorem volume_of_set (m n p : ℕ) (h_rel_prime : Nat.gcd n p = 1) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_p : 0 < p) 
  (h_volume : (m + n * Real.pi) / p = (324 + 37 * Real.pi) / 3) : 
  m + n + p = 364 := 
  sorry

end volume_of_set_l527_527199


namespace aprons_to_sew_tomorrow_l527_527289

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l527_527289


namespace clara_receives_amount_l527_527499

theorem clara_receives_amount :
  ∀ (face_value : ℝ) (num_quarters : ℕ) (percentage : ℝ),
    face_value = 0.25 →
    num_quarters = 7 →
    percentage = 3000 →
    (percentage / 100 * (num_quarters * face_value)) = 52.5 :=
by
  intros face_value num_quarters percentage hv hn hp
  rw [hv, hn, hp]
  norm_num
  rw ← mul_assoc
  norm_num
  rw mul_assoc
  norm_num
  sorry

end clara_receives_amount_l527_527499


namespace find_number_l527_527585

theorem find_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end find_number_l527_527585


namespace original_amount_l527_527155

variable (M : ℕ)

def initialAmountAfterFirstLoss := M - M / 3
def amountAfterFirstWin := initialAmountAfterFirstLoss M + 10
def amountAfterSecondLoss := amountAfterFirstWin M - (amountAfterFirstWin M) / 3
def amountAfterSecondWin := amountAfterSecondLoss M + 20
def finalAmount := amountAfterSecondWin M - (amountAfterSecondWin M) / 4

theorem original_amount : finalAmount M = M → M = 30 :=
by
  sorry

end original_amount_l527_527155


namespace quadrilateral_diagonals_l527_527134

theorem quadrilateral_diagonals (A B C D M N : Type) [metric_space M] 
  (hM : midpoint A C M) (hN : midpoint B D N) :
  dist A B ^ 2 + dist B C ^ 2 + dist C D ^ 2 + dist D A ^ 2 =
  dist A C ^ 2 + dist B D ^ 2 + 4 * dist M N ^ 2 :=
sorry

end quadrilateral_diagonals_l527_527134


namespace share_money_3_people_l527_527730

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l527_527730


namespace small_pizza_slices_correct_l527_527880

-- Defining the total number of people involved
def people_count : ℕ := 3

-- Defining the number of slices each person can eat
def slices_per_person : ℕ := 12

-- Calculating the total number of slices needed based on the number of people and slices per person
def total_slices_needed : ℕ := people_count * slices_per_person

-- Defining the number of slices in a large pizza
def large_pizza_slices : ℕ := 14

-- Defining the number of large pizzas ordered
def large_pizzas_count : ℕ := 2

-- Calculating the total number of slices provided by the large pizzas
def total_large_pizza_slices : ℕ := large_pizza_slices * large_pizzas_count

-- Defining the number of slices in a small pizza
def small_pizza_slices : ℕ := 8

-- Total number of slices provided needs to be at least the total slices needed
theorem small_pizza_slices_correct :
  total_slices_needed ≤ total_large_pizza_slices + small_pizza_slices := by
  sorry

end small_pizza_slices_correct_l527_527880


namespace isabel_reading_homework_pages_l527_527691

-- Definitions for the given problem
def num_math_pages := 2
def problems_per_page := 5
def total_problems := 30

-- Calculation based on conditions
def math_problems := num_math_pages * problems_per_page
def reading_problems := total_problems - math_problems

-- The statement to be proven
theorem isabel_reading_homework_pages : (reading_problems / problems_per_page) = 4 :=
by
  -- The proof would go here.
  sorry

end isabel_reading_homework_pages_l527_527691


namespace carpet_rate_proof_l527_527179

noncomputable def carpet_rate (breadth_first : ℝ) (length_ratio : ℝ) (cost_second : ℝ) : ℝ :=
  let length_first := length_ratio * breadth_first
  let area_first := length_first * breadth_first
  let length_second := length_first * 1.4
  let breadth_second := breadth_first * 1.25
  let area_second := length_second * breadth_second 
  cost_second / area_second

theorem carpet_rate_proof : carpet_rate 6 1.44 4082.4 = 45 :=
by
  -- Here we provide the goal and state what needs to be proven.
  sorry

end carpet_rate_proof_l527_527179


namespace good_lines_triangle_concur_good_lines_regular_polygon_concur_l527_527848

-- Part a: All good lines in a triangle concur at the incenter
theorem good_lines_triangle_concur
  (Δ : Type) [IsTriangle Δ]
  (good_line : Type) [GoodLine good_line Δ] :
  ∃ O : Point, ∀ l : good_line, l.passes_through O :=
sorry

-- Part b: All good lines in a regular polygon concur at the center
theorem good_lines_regular_polygon_concur
  (P : Type) [IsRegularPolygon P]
  (good_line : Type) [GoodLine good_line P] :
  ∃ O : Point, ∀ l : good_line, l.passes_through O :=
sorry

end good_lines_triangle_concur_good_lines_regular_polygon_concur_l527_527848


namespace not_both_perfect_squares_l527_527401

theorem not_both_perfect_squares (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ a b : ℕ, (n+1) * 2^n = a^2 ∧ (n+3) * 2^(n + 2) = b^2) :=
sorry

end not_both_perfect_squares_l527_527401


namespace finite_composite_terms_l527_527573

-- Define the sequence condition
def sequence_condition (b c : ℕ) (a : ℕ → ℕ) : Prop :=
  a 1 = b ∧ a 2 = c ∧ ∀ n ≥ 1, a (n + 2) = |3 * (a (n + 1)) - 2 * (a n)|

-- Prove that the given pairs (b, c) result in the sequence having only finite composite terms
theorem finite_composite_terms (b c : ℕ) (a : ℕ → ℕ) (h : sequence_condition b c a) :
  ∃ p : ℕ, Nat.Prime p ∧ ((b = 5 ∧ c = 4) ∨ (b = 7 ∧ c = 4) ∨ 
            (b = p ∧ c = p) ∨ (b = 2 * p ∧ c = p)) :=
sorry

end finite_composite_terms_l527_527573


namespace percentage_deposit_is_10_l527_527497

def deposit : ℝ := 55
def remaining : ℝ := 495
def total_cost : ℝ := deposit + remaining
def percentage_of_deposit : ℝ := (deposit / total_cost) * 100

theorem percentage_deposit_is_10 :
    percentage_of_deposit = 10 := 
by
  sorry

end percentage_deposit_is_10_l527_527497


namespace tangent_line_k_value_l527_527981

theorem tangent_line_k_value (k : ℝ) :
  (∃ x y, y = k * (x + real.sqrt 3) ∧ (x^2 + (y - 1)^2 = 1) ∧ (d = abs(real.sqrt 3 * k - 1) / sqrt(k^2 + 1))) →
    (k = real.sqrt 3 ∨ k = 0) := by
  sorry

end tangent_line_k_value_l527_527981


namespace perpendicular_planes_sufficient_condition_l527_527353

variables {l m n : Type} {α β : set l}

-- conditions
axiom m_subset_alpha : m ⊆ α
axiom m_perp_beta : perpendicular m β
axiom alpha_perp_beta : perpendicular α β

-- statement
theorem perpendicular_planes_sufficient_condition : 
  (m ⊆ α ∧ perpendicular m β) → perpendicular α β :=
by 
  intro h
  exact alpha_perp_beta

end perpendicular_planes_sufficient_condition_l527_527353


namespace youngsville_population_l527_527817

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end youngsville_population_l527_527817


namespace mail_distribution_l527_527150

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l527_527150


namespace composite_consecutive_powers_l527_527249

theorem composite_consecutive_powers (a : ℕ) (h : a ≥ 2) (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬ prime (a^k + i) :=
by
  sorry

end composite_consecutive_powers_l527_527249


namespace number_of_members_l527_527845

theorem number_of_members (n : ℕ) (h1 : ∀ m : ℕ, m = n → m * m = 1936) : n = 44 :=
by
  -- Proof omitted
  sorry

end number_of_members_l527_527845


namespace pyramid_volume_l527_527437

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c * Real.sqrt 2

theorem pyramid_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (A1 : ∃ x y, 1 / 2 * x * y = a^2) 
  (A2 : ∃ y z, 1 / 2 * y * z = b^2) 
  (A3 : ∃ x z, 1 / 2 * x * z = c^2)
  (h_perpendicular : True) :
  volume_of_pyramid a b c = (1 / 3) * a * b * c * Real.sqrt 2 :=
sorry

end pyramid_volume_l527_527437


namespace maxvalue_on_ellipse_l527_527919

open Real

noncomputable def max_x_plus_y : ℝ := 343 / 88

theorem maxvalue_on_ellipse (x y : ℝ) :
  (x^2 + 3 * x * y + 2 * y^2 - 14 * x - 21 * y + 49 = 0) →
  x + y ≤ max_x_plus_y := 
sorry

end maxvalue_on_ellipse_l527_527919


namespace smallest_positive_integer_form_3003_55555_l527_527799

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l527_527799


namespace no_solution_fibonacci_eq_l527_527713

def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem no_solution_fibonacci_eq : ∀ n : ℕ, n * fibonacci n * fibonacci (n + 1) ≠ (fibonacci (n + 2) - 1)^2 := 
by {
  intros n,
  induction n with d hd,
  { sorry }, -- Base case
  { sorry }  -- Inductive case
}

end no_solution_fibonacci_eq_l527_527713


namespace find_final_person_l527_527447

noncomputable def f : ℕ → ℕ
| 0         := 0
| (2 * k + 1) := f (2 * k)
| (2 * k)   := 2 * k + 2 - 2 * f k

theorem find_final_person (n : ℕ) : f 2022 = 1016 :=
by sorry

end find_final_person_l527_527447


namespace length_of_train_correct_l527_527822

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_sec

theorem length_of_train_correct :
  length_of_train 60 18 = 300.06 :=
by
  -- Placeholder for proof
  sorry

end length_of_train_correct_l527_527822


namespace sum_varieties_drawn_l527_527154

/-
The conditions of the problem are:
1. The mall has four types of food.
2. There are 40 varieties of grains.
3. There are 10 varieties of vegetable oil.
4. There are 30 varieties of animal products.
5. There are 20 varieties of fruits and vegetables.
6. A sample of 20 is drawn for food safety testing.
7. Stratified sampling is used to draw the sample.
-/

def mall_food_types : nat := 4
def grains_varieties : nat := 40
def vegetable_oil_varieties : nat := 10
def animal_products_varieties : nat := 30
def fruits_vegetables_varieties : nat := 20
def sample_size : nat := 20
def total_varieties : nat := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_vegetables_varieties
def sampling_fraction : rat := (sample_size : rat) / (total_varieties : rat)

def vegetable_oil_drawn : nat := (vegetable_oil_varieties : rat * sampling_fraction).to_nat
def fruits_vegetables_drawn : nat := (fruits_vegetables_varieties : rat * sampling_fraction).to_nat
def sum_drawn : nat := vegetable_oil_drawn + fruits_vegetables_drawn

theorem sum_varieties_drawn (mall_food_types = 4)
                           (grains_varieties = 40)
                           (vegetable_oil_varieties = 10)
                           (animal_products_varieties = 30)
                           (fruits_vegetables_varieties = 20)
                           (sample_size = 20)
                           (stratified_sampling : bool := true) :
  sum_drawn = 6 := 
  sorry

end sum_varieties_drawn_l527_527154


namespace horner_method_operations_l527_527545

noncomputable def f (x : ℝ) : ℝ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

theorem horner_method_operations :
  let multiplications := 5 in
  let additions := 5 in
  (multiplications + additions) = 10 :=
by
  sorry

end horner_method_operations_l527_527545


namespace caterpillars_per_jar_l527_527695

theorem caterpillars_per_jar (C : ℕ) : 
  (4 * C * 0.6) * 3 = 72 → C = 10 :=
by
  intros h
  sorry

end caterpillars_per_jar_l527_527695


namespace polynomial_transformation_exists_l527_527025

theorem polynomial_transformation_exists (P : ℝ → ℝ → ℝ) (hP : ∀ x y, P (x - 1) (y - 2 * x + 1) = P x y) :
  ∃ Φ : ℝ → ℝ, ∀ x y, P x y = Φ (y - x^2) := by
  sorry

end polynomial_transformation_exists_l527_527025


namespace pythagorean_tetrahedron_l527_527859

variables {A B C M : Type}
noncomputable
def area (x y z : Type) [triangle x y z] : ℝ := sorry

theorem pythagorean_tetrahedron
    (a b c : ℝ)
    (h₁ : is_right_triangle M A B)
    (h₂ : is_right_triangle M A C)
    (h₃ : is_right_triangle M B C)
    (area_ABC : ℝ)
    (area_AMB : ℝ := area M A B)
    (area_AMC : ℝ := area M A C)
    (area_BMC : ℝ := area M B C) :
    area_ABC ^ 2 = area_AMB ^ 2 + area_AMC ^ 2 + area_BMC ^ 2 :=
sorry

end pythagorean_tetrahedron_l527_527859


namespace sum_of_coefficients_l527_527584

theorem sum_of_coefficients (x y : ℤ) :
  (x - 3 * y) ^ 20 = (1 - 3 * 1) ^ 20 → 
  ∑ i in range (21), binomial (20, i) * (1:ℤ) ^ (20 - i) * ((-3 * (1:ℤ)) ^ i) = 1048576 := 
by
  intros h₁
  rw [h₁]
  norm_num
  sorry

end sum_of_coefficients_l527_527584


namespace num_passenger_cars_l527_527516

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l527_527516


namespace people_got_on_third_stop_l527_527061

theorem people_got_on_third_stop :
  let passengers_first_stop := 7 in
  let passengers_second_stop := passengers_first_stop - 3 + 5 in
  let passengers_before_third_stop := passengers_second_stop - 2 in
  let final_passengers := 11 in
  final_passengers - passengers_before_third_stop = 4
:= by
  sorry

end people_got_on_third_stop_l527_527061


namespace parametric_curve_length_l527_527580

theorem parametric_curve_length :
  (∫ t in (0:ℝ)..(2*Real.pi), √( (2 * Real.cos t)^2 + (2 * Real.sin t)^2 )) = 4 * Real.pi :=
by sorry

end parametric_curve_length_l527_527580


namespace a2_value_l527_527269

namespace SequenceProblem

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
axiom sum_of_terms (n : ℕ) : S n = ∑ i in Finset.range (n + 1), a i

-- Condition 2: 2S_n - n a_n = n for n ∈ ℕ*
axiom sequence_condition : ∀ n : ℕ, 2 * S n - n * a n = n

-- Condition 3: S_20 = -360
axiom initial_condition : S 20 = -360

-- Proof goal: a_2 = -1
theorem a2_value : a 2 = -1 :=
sorry

end SequenceProblem

end a2_value_l527_527269


namespace find_number_of_males_and_females_l527_527451

def formula_satisfies (x : ℕ) : x * (x - 1) * (8 - x) = 15 :=
  x * (x - 1) * (8 - x) = 15

theorem find_number_of_males_and_females : ∃ x : ℕ, (2 ≤ x ∧ x ≤ 6) ∧ formula_satisfies x :=
by {
  use 3,
  split,
  { simp, },
  { sorry }
}

end find_number_of_males_and_females_l527_527451


namespace union_sets_l527_527986

-- Definitions of sets A and B
def A : set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : set ℝ := {x : ℝ | x^2 - 1 < 0}

-- The proof statement
theorem union_sets : (A ∪ B) = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end union_sets_l527_527986


namespace shirts_washed_total_l527_527548

theorem shirts_washed_total (short_sleeve_shirts long_sleeve_shirts : Nat) (h1 : short_sleeve_shirts = 4) (h2 : long_sleeve_shirts = 5) : short_sleeve_shirts + long_sleeve_shirts = 9 := by
  sorry

end shirts_washed_total_l527_527548


namespace dot_product_ab_l527_527619

variables {a b : EuclideanSpace ℝ (Fin 3)} -- Assume a and b are in 3D space
variable lens : norm b = 3 -- Condition 1: |b| = 3
variable proj : (a • b) / norm b = 2 / 3 -- Condition 2: projection of a in the direction of b is 2/3

theorem dot_product_ab : a • b = 2 :=
by 
  sorry

end dot_product_ab_l527_527619


namespace can_place_one_more_T_l527_527388

-- Define the 100x100 checkerboard
def checkerboard : Type := Finₓ 100 × Finₓ 100

def T_tile (pos : checkerboard) : set checkerboard :=
  {pos}
  ∪ {(i+1, j), (i-1, j), (i, j+1), (i, j-1) | (i, j) ∈ {pos}}

-- We are given 800 T-shaped tiles on the checkerboard
axiom T_placed : set (set checkerboard)
axiom T_placed_card : T_placed.card = 800
axiom T_disjoint : ∀ (t1 t2 : set checkerboard), t1 ∈ T_placed → t2 ∈ T_placed → t1 ≠ t2 → t1 ∩ t2 = ∅

-- Prove one more T-shaped tile can be placed without overlapping
theorem can_place_one_more_T :
  ∃ new_T : set checkerboard, new_T ∉ T_placed ∧ (new_T ∩ ⋃ T_placed = ∅) ∧ (∃ pos: checkerboard, new_T = T_tile pos) :=
sorry

end can_place_one_more_T_l527_527388


namespace cosine_AHB_l527_527721

-- Definitions for orthocenter and vector conditions
variables {A B C H : Type*}
variables [EuclideanGeometry H]

-- Given conditions
def orthocenter_condition (A B C H : Type*) [EuclideanGeometry H] :=
  orthocenter A B C H

-- Vector equation condition
def vector_condition (A B C H : Type*) [EuclideanGeometry H] :=
  3 * (H - A) + 4 * (H - B) + 5 * (H - C) = 0

-- Main theorem to prove
theorem cosine_AHB (A B C H : Type*) [EuclideanGeometry H]
  (h1 : orthocenter_condition A B C H)
  (h2 : vector_condition A B C H) :
  cos (angle A H B) = - (sqrt 6 / 6) :=
sorry

end cosine_AHB_l527_527721


namespace sledding_small_hills_l527_527813

theorem sledding_small_hills (total_sleds tall_hills_sleds sleds_per_tall_hill sleds_per_small_hill small_hills : ℕ) 
  (h1 : total_sleds = 14)
  (h2 : tall_hills_sleds = 2)
  (h3 : sleds_per_tall_hill = 4)
  (h4 : sleds_per_small_hill = sleds_per_tall_hill / 2)
  (h5 : total_sleds = tall_hills_sleds * sleds_per_tall_hill + small_hills * sleds_per_small_hill)
  : small_hills = 3 := 
sorry

end sledding_small_hills_l527_527813


namespace area_ratio_BCX_ACX_l527_527581

-- Definitions for the problem
variables (A B C X : Type)
variables (AB BC AC : ℝ)
variables (triangle_BCX triangle_ACX : Type)
variable  (CX_bisects_∠ACB : Prop)

axiom AB_value : AB = 26
axiom BC_value : BC = 25
axiom AC_value : AC = 29
axiom CX_bisects_AngleACB : CX_bisects_∠ACB

-- The theorem we want to prove
theorem area_ratio_BCX_ACX : 
  let ratio := 25 / 29 in
  Ratio (triangle.area triangle_BCX) (triangle.area triangle_ACX) = ratio :=
  sorry

end area_ratio_BCX_ACX_l527_527581


namespace sum_first_n_terms_l527_527332

variable (n : ℕ)
variable (a : ℕ → ℝ)

-- Defining the conditions
def a2 : Prop := a 2 = 1
def a5 : Prop := a 5 = 8

-- Sum of the first n terms
def Sn : ℝ := 2^(n-1) - 1/2

theorem sum_first_n_terms 
  (h1 : a2 a) 
  (h2 : a5 a) : 
  ∀ n, S_n = 2^(n-1) - 1/2 := sorry

end sum_first_n_terms_l527_527332


namespace irreducible_fraction_for_any_n_l527_527749

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n_l527_527749


namespace polynomial_horner_form_operations_l527_527101

noncomputable def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldr (fun a acc => a + acc * x) 0

theorem polynomial_horner_form_operations :
  let p := [1, 1, 2, 3, 4, 5]
  let x := 2
  horner_eval p x = ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 ∧
  (∀ x, x = 2 → (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 =  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + 1 * x + 1)) ∧ 
  (∃ m a, m = 5 ∧ a = 5) := sorry

end polynomial_horner_form_operations_l527_527101


namespace xiaoli_estimate_greater_l527_527123

variable (p q a b : ℝ)

theorem xiaoli_estimate_greater (hpq : p > q) (hq0 : q > 0) (hab : a > b) : (p + a) - (q + b) > p - q := 
by 
  sorry

end xiaoli_estimate_greater_l527_527123


namespace hyperbola_equation_l527_527604

theorem hyperbola_equation (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (eccentricity : c = 5 * a / 3)
  (directrix : ∀ x y, y^2 = 12 * x → x = -3)
  (asymptote : ∀ x y, 4 * x - 3 * y = 0 → y = 4 * x / 3)
  (P : ℝ × ℝ)
  (P_intersection: P = (-3, -4))
  (focus1 : (-c, 0))
  (focus2 : (c, 0))
  (orthogonal : ∀ K, (K - c, 4) ∙ (K + c, 4) = 0)
  : (∃ a' b', a' = 3 ∧ b' = 4 ∧ ( ∀ x y, (x^2 / a'^2) - (y^2 / b'^2) = 1)) := 
begin
  sorry
end

end hyperbola_equation_l527_527604


namespace matrix_power_result_l527_527341

open Matrix

variables (B : Matrix (Fin 3) (Fin 3) ℝ)
  [hB : B = ![
  ![0, 1, 0],
  ![0, 0, -1],
  ![0, 1, 0]
]] 

theorem matrix_power_result : B ^ 106 = ![
  ![0, 0, -1],
  ![0, -1, 0],
  ![0, 0, -1]
] := 
by
  sorry

end matrix_power_result_l527_527341


namespace tissues_per_box_l527_527090

-- Given definitions
def boxes : ℕ := 3
def used : ℕ := 210
def left : ℕ := 270

-- Prove the number of tissues per box
theorem tissues_per_box (total_tissues : ℕ) (tissues_per_box : ℕ) :
  total_tissues = used + left →
  total_tissues / boxes = tissues_per_box →
  tissues_per_box = 160 :=
by
  intro h1 h2
  rw [h1] at h2
  simp at h2
  assumption

-- Specify the example case
example : tissues_per_box (used + left) ((used + left) / boxes) :=
by
  apply tissues_per_box
  simp
  sorry

end tissues_per_box_l527_527090


namespace unique_1000_digit_number_in_base_2022_l527_527031

theorem unique_1000_digit_number_in_base_2022 : 
  ∃! (N : Nat), (∀ i : Nat, i < 1000 → (N / 2022^i) % 2022 ∈ {1, 2}) ∧ 2^1000 ∣ N := 
by
  sorry

end unique_1000_digit_number_in_base_2022_l527_527031


namespace periodic_function_sum_l527_527898

noncomputable def f : ℝ → ℝ
| x => if x % 6 ∈ [-3, -1] then -(x % 6 + 2)^2 else if x % 6 ∈ [-1, 3) then x % 6 else 0

theorem periodic_function_sum :
  (∑ i in Finset.range 2015 (λ i, f (i + 1))) = 336 := by
  sorry

end periodic_function_sum_l527_527898


namespace congruent_triangles_have_equal_corresponding_sides_l527_527477

theorem congruent_triangles_have_equal_corresponding_sides
  (T1 T2 : Triangle) (h : T1 ≅ T2) :
  T1.AB = T2.AB ∧ T1.BC = T2.BC ∧ T1.CA = T2.CA :=
sorry

end congruent_triangles_have_equal_corresponding_sides_l527_527477


namespace line_tangent_circle_iff_m_l527_527307

/-- Definition of the circle and the line -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Prove that the line is tangent to the circle if and only if m = -3 or m = -13 -/
theorem line_tangent_circle_iff_m (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y m) ↔ m = -3 ∨ m = -13 :=
by
  sorry

end line_tangent_circle_iff_m_l527_527307


namespace cindy_coins_l527_527193

theorem cindy_coins :
  ∃ n : ℕ, ((factors n).card = 19) ∧
           ∀ k, k ∣ n ∧ k ≠ 1 ∧ k ≠ n → (k ∈ proper_divisors n) ∧ (factors n).card = 19 :=
begin
  sorry
end

end cindy_coins_l527_527193


namespace nicky_catches_up_time_l527_527484

theorem nicky_catches_up_time
  (head_start : ℕ := 12)
  (cristina_speed : ℕ := 5)
  (nicky_speed : ℕ := 3)
  (head_start_distance : ℕ := nicky_speed * head_start)
  (time_to_catch_up : ℕ := 36 / 2) -- 36 is the head start distance of 36 meters
  (total_time : ℕ := time_to_catch_up + head_start)  -- Total time Nicky runs before Cristina catches up
  : total_time = 30 := sorry

end nicky_catches_up_time_l527_527484


namespace sum_of_reciprocals_of_distances_l527_527472

theorem sum_of_reciprocals_of_distances (s : Finset ℂ) 
  (h : ∀ z ∈ s, z ^ 8 = 1) : 
  ∑ z in s, (1 / (Complex.abs (1 - z))^2) = 8 / 3 :=
by {
  sorry
}

end sum_of_reciprocals_of_distances_l527_527472


namespace measure_of_angle_AMB_l527_527684

noncomputable def angle_ACB (angle_DCA angle_ABC : ℝ) (parallel_AB_DC : Prop) : ℝ :=
  if parallel_AB_DC then 100 - angle_DCA / 2 else 0

-- Our conditions
def line_segments_parallel : Prop := sorry
def angle_DCA : ℝ := 30
def angle_ABC : ℝ := 80

-- The statement we're proving
theorem measure_of_angle_AMB :
    line_segments_parallel →
    angle_DCA = 30 →
    angle_ABC = 80 →
    angle_ACB angle_DCA angle_ABC line_segments_parallel = 70 :=
  sorry

end measure_of_angle_AMB_l527_527684


namespace michael_earnings_l527_527728

theorem michael_earnings :
  let price_extra_large := 150
  let price_large := 100
  let price_medium := 80
  let price_small := 60
  let qty_extra_large := 3
  let qty_large := 5
  let qty_medium := 8
  let qty_small := 10
  let discount_large := 0.10
  let tax := 0.05
  let cost_materials := 300
  let commission_fee := 0.10

  let total_initial_sales := (qty_extra_large * price_extra_large) + 
                             (qty_large * price_large) + 
                             (qty_medium * price_medium) + 
                             (qty_small * price_small)

  let discount_on_large := discount_large * (qty_large * price_large)
  let sales_after_discount := total_initial_sales - discount_on_large

  let sales_tax := tax * sales_after_discount
  let total_collected := sales_after_discount + sales_tax

  let commission := commission_fee * sales_after_discount
  let total_deductions := cost_materials + commission
  let earnings := total_collected - total_deductions

  earnings = 1733 :=
by
  sorry

end michael_earnings_l527_527728


namespace negation_exists_positive_real_square_plus_one_l527_527949

def exists_positive_real_square_plus_one : Prop :=
  ∃ (x : ℝ), x^2 + 1 > 0

def forall_non_positive_real_square_plus_one : Prop :=
  ∀ (x : ℝ), x^2 + 1 ≤ 0

theorem negation_exists_positive_real_square_plus_one :
  ¬ exists_positive_real_square_plus_one ↔ forall_non_positive_real_square_plus_one :=
by
  sorry

end negation_exists_positive_real_square_plus_one_l527_527949


namespace find_f_of_x2_minus_2_l527_527714

theorem find_f_of_x2_minus_2 (f : ℝ → ℝ)
  (h : ∀ x, f(x^2 + 2) = x^4 + 5 * (x^2) + 1) :
  ∀ x, f(x^2 - 2) = x^4 - 3 * (x^2) - 3 := 
sorry

end find_f_of_x2_minus_2_l527_527714


namespace river_width_l527_527458

variables (W : ℝ)
hypothesis (h1 : ∃ x y : ℝ, x + y = W ∧ x = 700 ∧ y = W - 700)
hypothesis (h2 : ∃ z w : ℝ, z + w = 3W ∧ z = 2W - 400 ∧ w = 2W - 300)

/-- The width of the river given the meeting points of two ferries. -/
theorem river_width : W = 700 :=
by
  obtain ⟨x, y, xy_eq_W, x_eq_700, y_eq_W_m700⟩ := h1
  obtain ⟨z, w, zw_eq_3W, z_eq_2W_m400, w_eq_2W_m300⟩ := h2
  have zw_eq_4W_m700 : z + w = 4W - 700, by
    rw [z_eq_2W_m400, w_eq_2W_m300]
    ring
  rw zw_eq_4W_m700 at zw_eq_3W
  linarith

end river_width_l527_527458


namespace a_not_in_range_of_g_l527_527929

def is_not_in_range (g : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x : ℝ, g x ≠ y

def quadratic (a : ℝ) : ℝ → ℝ :=
  fun x => x^2 + a * x + 3

theorem a_not_in_range_of_g (a : ℝ) : 
  ¬ is_not_in_range (quadratic a) (-3) ↔ -sqrt 24 < a ∧ a < sqrt 24 :=
by
  sorry

end a_not_in_range_of_g_l527_527929


namespace probability_neither_defective_l527_527127

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def non_defective_pens : ℕ := total_pens - defective_pens
def draw_count : ℕ := 2

def probability_of_non_defective (total : ℕ) (defective : ℕ) (draws : ℕ) : ℚ :=
  let non_defective := total - defective
  (non_defective / total) * ((non_defective - 1) / (total - 1))

theorem probability_neither_defective :
  probability_of_non_defective total_pens defective_pens draw_count = 5 / 14 :=
by sorry

end probability_neither_defective_l527_527127


namespace sum_of_x_and_y_l527_527234

theorem sum_of_x_and_y (x y : ℝ) (h : (x + y + 2)^2 + |2 * x - 3 * y - 1| = 0) : x + y = -2 :=
by
  sorry

end sum_of_x_and_y_l527_527234


namespace new_perimeter_proof_l527_527486

def width : ℝ := 10
def original_area : ℝ := 150
def new_area : ℝ := (4/3 : ℝ) * original_area

theorem new_perimeter_proof : new_area = (4/3 : ℝ) * original_area → 2 * ((new_area / width) + width) = 60 := by
  intros h1
  simp [new_area, original_area, width] at h1
  simp [h1, width]
  sorry

end new_perimeter_proof_l527_527486


namespace no_base6_digit_d_divisible_by_7_l527_527231

theorem no_base6_digit_d_divisible_by_7 : 
∀ d : ℕ, (d < 6) → ¬ (654 + 42 * d) % 7 = 0 :=
by
  intro d h
  -- Proof is omitted as requested
  sorry

end no_base6_digit_d_divisible_by_7_l527_527231


namespace find_xy_solution_l527_527254

theorem find_xy_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x = 2 ∧ y = 4 ∧ x^(a + b) + y = x^a * y^b :=
by
  use 2, 4
  split
  { exact Nat.zero_lt_succ _ }
  split
  { exact Nat.zero_lt_succ _ }
  split
  { refl }
  split
  { refl }
  rw [pow_add, pow_succ]
  sorry

end find_xy_solution_l527_527254


namespace first_three_decimal_digits_of_x_l527_527105

noncomputable def x : ℝ := (10^100 + 1) ^ (5 / 3)

theorem first_three_decimal_digits_of_x : (floor ((x - floor x) * 1000)) = 666 := 
sorry

end first_three_decimal_digits_of_x_l527_527105


namespace linear_function_equality_l527_527052

theorem linear_function_equality (f : ℝ → ℝ) (hf : ∀ x, f (3 * (f x)⁻¹ + 5) = f x)
  (hf1 : f 1 = 5) : f 2 = 3 :=
sorry

end linear_function_equality_l527_527052


namespace constructQuadrilateral_l527_527282

-- Defining points and segments
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Segment :=
  (p1 : Point)
  (p2 : Point)

-- Defining the properties of midpoints and equal segments
def isMidpoint (p m q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

def areEqualSegments (a b c d : Segment) : Prop :=
  let length := λ s : Segment, (s.p2.x - s.p1.x)^2 + (s.p2.y - s.p1.y)^2
  length a = length b ∧ length b = length c ∧ length c = length d

-- The quadrilateral construction
theorem constructQuadrilateral (A B C D P Q R : Point) (AB BC CD : Segment):
  isMidpoint A P B →
  isMidpoint B Q C →
  isMidpoint C R D →
  areEqualSegments AB BC CD →
  convexQuadrilateral A B C D :=
begin
  sorry
end

end constructQuadrilateral_l527_527282


namespace prime_insert_impossible_l527_527487

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0

def median_is_nine (s : List ℕ) (n : ℕ) : Prop :=
  let sorted := s.insertion_sort (· ≤ ·)
  sorted.length = 7 ∧ sorted.nth 3 = some n

theorem prime_insert_impossible :
  ∀ (S : List ℕ), S = [8, 46, 53, 127] →
  ∀ (p : ℕ), is_prime p → is_multiple_of_three p →
  ∀ (N : List ℕ), N.length = 3 → 
  ∀ (M : List ℕ), M = S ++ p :: N →
  (∀ x ∈ M, is_multiple_of_three x) →
  median_is_nine M 9 →
  false :=
by
  sorry

end prime_insert_impossible_l527_527487


namespace triangle_inequality_60_degree_l527_527335

theorem triangle_inequality_60_degree (A B C : Type) [metric_space A] [metric_space B] [metric_space C] [has_dist A B] [has_dist B C] 
    (AB AC BC : ℝ) (h : ∀ x y : A, dist x y = dist y x)  (h55 : angle A B C = 60):
    AB + AC ≤ 2 * BC :=
begin
    sorry
end

end triangle_inequality_60_degree_l527_527335


namespace license_plates_count_l527_527847

def number_of_license_plates : ℕ :=
  let num_digits := 10 ^ 5
  let num_letters := 26 ^ 2
  let arrangements := (Nat.choose 7 2)
  arrangements * num_letters * num_digits

theorem license_plates_count :
  number_of_license_plates = 1_420_200_000 := by
  sorry

end license_plates_count_l527_527847


namespace hyperbola_properties_l527_527358

theorem hyperbola_properties 
  (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (a b h k : ℝ)
  (hF1 : F1 = (-2, 0))
  (hF2 : F2 = (2, 0))
  (hyperbola_eq : abs (dist P F1 - dist P F2) = 2)
  (hyp_eq : (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) :
  (h + k + a + b = 1 + real.sqrt 3) :=
sorry

end hyperbola_properties_l527_527358


namespace period_of_f_4pi_monotonic_intervals_of_f_max_min_values_of_f_on_interval_l527_527973

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos (π / 3 - x / 2)

theorem period_of_f_4pi : ∀ x, f (x + 4 * π) = f x := 
sorry

theorem monotonic_intervals_of_f :
  ∀ k : ℤ, 
    ∀ x1 x2 : ℝ,
      4 * k * π - 4 * π / 3 ≤ x1 → x1 ≤ x2 → x2 ≤ 4 * k * π + 2 * π / 3 →
      f x1 ≤ f x2 := 
sorry

theorem max_min_values_of_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x : ℝ, 0 ≤ x → x ≤ 2 * π → f x ≤ max) ∧ 
    (∀ x : ℝ, 0 ≤ x → x ≤ 2 * π → min ≤ f x) ∧ 
    max = 2 ∧ min = -1 := 
sorry

end period_of_f_4pi_monotonic_intervals_of_f_max_min_values_of_f_on_interval_l527_527973


namespace sqrt_eq_sum_cond_l527_527904

theorem sqrt_eq_sum_cond (a b c : ℝ) :
  sqrt (a^2 + b^2 + c^2) = a + b - c ↔ (ab = c * (a + b) ∧ a + b - c ≥ 0) :=
by sorry

end sqrt_eq_sum_cond_l527_527904


namespace smallest_sum_first_row_grid_l527_527785

theorem smallest_sum_first_row_grid :
  (∃ grid : Array (Array Nat), 
    grid.size = 9 ∧
    (∀ row, row ∈ grid → row.size = 2004) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 2004 → 
      let count_i := grid.map (λ row, row.count (λ x, x = i)).sum in
      count_i = 9) ∧
    (∀ j, 0 ≤ j ∧ j < 2004 →
      let col := grid.map (λ row, row[j]) in
      (col.max' - col.min') ≤ 3)) →
  ∃ first_row_sum : ℕ,
    first_row_sum = (grid[0].sum) ∧ 
    first_row_sum = 2005004 := sorry

end smallest_sum_first_row_grid_l527_527785


namespace function_bounded_l527_527008

open Real

noncomputable def twice_differentiable (f : ℝ → ℝ) : Prop :=
differentiable ℝ f ∧ differentiable ℝ (deriv f)

theorem function_bounded
  (f g : ℝ → ℝ)
  (hf : twice_differentiable f)
  (hg : ∀ x, g x ≥ 0)
  (H : ∀ x, f x + deriv^[2] f x = -x * g x * deriv f x) :
  ∃ M, ∀ x, abs (f x) ≤ M :=
by
  sorry

end function_bounded_l527_527008


namespace egg_count_l527_527518

theorem egg_count : ∃ x : ℕ, 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 1 ∧ 
  x % 5 = 1 ∧ 
  x % 6 = 1 ∧ 
  x % 7 = 0 ∧
  x = 301 :=
begin
  -- The solution steps would go here.
  sorry
end

end egg_count_l527_527518


namespace find_a_b_l527_527975

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_a_b (a b c : ℝ) (h1 : (12 * a + b = 0)) (h2 : (4 * a + b = -3)) :
  a = 3 / 8 ∧ b = -9 / 2 := by
  sorry

end find_a_b_l527_527975


namespace find_e_of_conditions_l527_527076

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem find_e_of_conditions (d e f : ℝ) 
  (h1 : f = 6) 
  (h2 : -d / 3 = -f)
  (h3 : -f = d + e + f - 1) : 
  e = -30 :=
by 
  sorry

end find_e_of_conditions_l527_527076


namespace v_1005_is_193_l527_527718

-- Definitions and conditions based on the problem statement
def group_term (n term_num : ℕ) : ℕ := 3 * n + 1 + 4 * (term_num - 1)

def term_group (term_index : ℕ) : ℕ :=
  let n := Nat.find (λ k => (k * (k + 1)) / 2 ≥ term_index)
  if (n * (n + 1)) / 2 = term_index then n else n - 1

def v (index : ℕ) : ℕ :=
  let g := term_group index
  let s := (g * (g + 1)) / 2
  group_term g (index - s)

-- Main statement to prove
theorem v_1005_is_193 : v 1005 = 193 := by
  sorry

end v_1005_is_193_l527_527718


namespace coefficient_of_second_highest_term_l527_527363

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

noncomputable def g (x : ℝ) : ℝ := (λ n, (nat.rec_on n x (λ _ y, f y))) 2009

theorem coefficient_of_second_highest_term :
  ∃ a : ℝ, ∃ p : ℕ, (p = 2^2009 - 1) ∧ (a = 2^2009) ∧ (g(0) = (0^2^2009 + a * 0^p + ...)) :=
  sorry

end coefficient_of_second_highest_term_l527_527363


namespace area_undetermined_l527_527815

theorem area_undetermined (side1 : ℕ) (h1 : side1 = 10) (no_info : ¬ (∃ side2 : ℕ, side2 = side1 ∨ (∃ side3 : ℕ, other_side_length_provided side3))) : 
  ∀ (area : ℕ), False := 
by
  sorry

def other_side_length_provided : ℕ → Prop := sorry

end area_undetermined_l527_527815


namespace seq_a_Sn_on_line_sn_sum_bound_l527_527606

def seq_a (n : ℕ) : ℝ := (1 / 2)^(2 * n + 1)
def seq_c (n : ℕ) : ℕ := if n = 1 then 0 else sum_up_to_log (n : ℕ) : ℝ := 
  if n = 1 then 0 else (finset.sum (finset.range (n : ℕ).succ) (λ i, (c (i : ℕ) (i - 1))))

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (1/6) - (1/3) * (finset.sum (finset.range n) a)

theorem seq_a_Sn_on_line (seq_a : ℕ → ℝ) (Sn : ℕ → ℝ) (n : ℕ) : 
  (seq_a n, Sn n) = (a, (1 / 6) - (1 / 3) * a) := sorry

theorem sn_sum_bound 
  {c : ℕ → ℝ} (hn : ∀ n, c n = 0 ∨ c n = log (2) (seq_a n))
  (n ≥ 2) : 
  (1/3 : ℝ) ≤ finset.sum (finset.iota 2 n) (λ k, 1 / c k) < (3/4 : ℝ) := sorry

end seq_a_Sn_on_line_sn_sum_bound_l527_527606


namespace Pete_books_total_l527_527655

variable (Matt_books_first_year Pete_books_first_year Pete_books_second_year : ℕ)

-- Conditions
def condition1 (Matt_books_first_year : ℕ) :=
  Matt_books_first_year = 75 / 1.5 -- Matt read 75 books in the second year, which is 50% more than the first year.

def condition2 (Pete_books_first_year : ℕ) (Matt_books_first_year : ℕ) :=
  Pete_books_first_year = 2 * Matt_books_first_year -- Pete read twice as many books last year as Matt did.

def condition3 (Pete_books_second_year : ℕ) (Pete_books_first_year : ℕ) :=
  Pete_books_second_year = 2 * Pete_books_first_year -- Pete doubles the number of books he read from last year to this year.

-- Theorem: Calculate the total number of books Pete read in both years
theorem Pete_books_total
  (Matt_books_first_year Pete_books_first_year Pete_books_second_year : ℕ)
  (h1 : condition1 Matt_books_first_year)
  (h2 : condition2 Pete_books_first_year Matt_books_first_year)
  (h3 : condition3 Pete_books_second_year Pete_books_first_year)
  : Pete_books_first_year + Pete_books_second_year = 300 := by
  sorry  -- Proof is omitted

end Pete_books_total_l527_527655


namespace first_rectangle_dimensions_second_rectangle_dimensions_l527_527058

theorem first_rectangle_dimensions (x y : ℕ) (h : x * y = 2 * (x + y) + 1) : (x = 7 ∧ y = 3) ∨ (x = 3 ∧ y = 7) :=
sorry

theorem second_rectangle_dimensions (a b : ℕ) (h : a * b = 2 * (a + b) - 1) : (a = 5 ∧ b = 3) ∨ (a = 3 ∧ b = 5) :=
sorry

end first_rectangle_dimensions_second_rectangle_dimensions_l527_527058


namespace milo_run_distance_l527_527375

def cory_speed : ℝ := 12
def milo_roll_speed := cory_speed / 2
def milo_run_speed := milo_roll_speed / 2
def time_hours : ℝ := 2

theorem milo_run_distance : milo_run_speed * time_hours = 6 := 
by 
  /- The proof goes here -/
  sorry

end milo_run_distance_l527_527375


namespace constructQuadrilateral_l527_527283

-- Defining points and segments
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Segment :=
  (p1 : Point)
  (p2 : Point)

-- Defining the properties of midpoints and equal segments
def isMidpoint (p m q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

def areEqualSegments (a b c d : Segment) : Prop :=
  let length := λ s : Segment, (s.p2.x - s.p1.x)^2 + (s.p2.y - s.p1.y)^2
  length a = length b ∧ length b = length c ∧ length c = length d

-- The quadrilateral construction
theorem constructQuadrilateral (A B C D P Q R : Point) (AB BC CD : Segment):
  isMidpoint A P B →
  isMidpoint B Q C →
  isMidpoint C R D →
  areEqualSegments AB BC CD →
  convexQuadrilateral A B C D :=
begin
  sorry
end

end constructQuadrilateral_l527_527283


namespace cubic_polynomial_at_zero_l527_527715

noncomputable def f (x : ℝ) : ℝ := by sorry

theorem cubic_polynomial_at_zero :
  (∃ f : ℝ → ℝ, f 2 = 15 ∨ f 2 = -15 ∧
                 f 4 = 15 ∨ f 4 = -15 ∧
                 f 5 = 15 ∨ f 5 = -15 ∧
                 f 6 = 15 ∨ f 6 = -15 ∧
                 f 8 = 15 ∨ f 8 = -15 ∧
                 f 9 = 15 ∨ f 9 = -15 ∧
                 ∀ x, ∃ c a b d, f x = c * x^3 + a * x^2 + b * x + d ) →
  |f 0| = 135 :=
by sorry

end cubic_polynomial_at_zero_l527_527715


namespace probability_O_ABCD_volume_not_less_than_two_third_l527_527243

noncomputable def PA : ℝ := 2
noncomputable def AB : ℝ := 2
noncomputable def height_requirement (O : ℝ) : Prop := O ≥ 1

lemma probability_O_ABCD_volume {O : ℝ} (hO : O ∈ set.Icc 0 2) :
  (O ≥ 1) ↔ (volume_P_O_ABCD_ge_two_third (P-ABCD) O) :=
sorry

theorem probability_O_ABCD_volume_not_less_than_two_third :
  probability (O ∈ set.Icc 0 2 ∧ volume_P_O_ABCD_ge_two_third (P-ABCD) O) = 1 / 2 :=
begin
  sorry
end

end probability_O_ABCD_volume_not_less_than_two_third_l527_527243


namespace quadrant_of_expr_l527_527224

-- Define complex number parts
def z : ℂ := 1 - 2 * complex.I
def w : ℂ := z ^ 3

-- Define the given expression
def expr : ℂ := w / complex.I

-- Define the coordinates of the complex number
def x : ℝ := expr.re
def y : ℝ := expr.im

-- State the theorem to be proven
theorem quadrant_of_expr (hx_positive : 0 < x) (hy_positive : 0 < y) : "first quadrant" = "first quadrant" := 
by 
  sorry

end quadrant_of_expr_l527_527224


namespace passenger_cars_count_l527_527511

theorem passenger_cars_count (P C : ℕ) 
    (h₁ : C = (1 / 2 : ℚ) * P + 3) 
    (h₂ : P + C + 2 = 71) : P = 44 :=
begin
  sorry
end

end passenger_cars_count_l527_527511


namespace equalize_money_l527_527180

theorem equalize_money (ann_money : ℕ) (bill_money : ℕ) : 
  ann_money = 777 → 
  bill_money = 1111 → 
  ∃ x, bill_money - x = ann_money + x :=
by
  sorry

end equalize_money_l527_527180


namespace simplify_expression_l527_527197

variable (a : ℝ)
def b := a + a⁻¹
def c := a - a⁻¹

theorem simplify_expression :
  a^4 - a^(-4) = (a - a⁻¹) * (a + a⁻¹) * ((a + a⁻¹) * (a - a⁻¹) + 2) :=
by 
  sorry

end simplify_expression_l527_527197


namespace thirty_divides_n_squared_minus_one_l527_527403

open Nat

theorem thirty_divides_n_squared_minus_one (n : ℕ) (hn : Prime n) (h : 7 ≤ n) : 30 ∣ (n^2 - 1) := by
  sorry

end thirty_divides_n_squared_minus_one_l527_527403


namespace cosine_value_l527_527935

variable (θ α : ℝ) 

theorem cosine_value (h : real.sin θ = α) : real.cos θ = -real.sqrt (1 - α^2) :=
sorry

end cosine_value_l527_527935


namespace smallest_N_l527_527673

theorem smallest_N :
  ∃ (N : ℕ),
  (∀ (P : fin 6 → ℕ) (x y : fin 6 → ℕ),
    (∀ i, x i = i * N + P i) →
    (∀ i, y i = (P i) * 6 + i) →
    x 1 = y 3 ∧
    x 2 = y 1 ∧
    x 3 = y 6 ∧
    x 4 = y 2 ∧
    x 5 = y 4 ∧
    x 6 = y 5) →
  N = 6 :=
by sorry

end smallest_N_l527_527673


namespace matrix_N_properties_l527_527220

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ :=
  ![-1, 2, 5; 4, -3, 2; 0, 5, -1]

def i : Vector (Fin 3) ℝ :=
  ![1, 0, 0]

def j : Vector (Fin 3) ℝ :=
  ![0, 1, 0]

def k : Vector (Fin 3) ℝ :=
  ![0, 0, 1]

def v1 : Vector (Fin 3) ℝ :=
  ![-1, 4, 0]

def v2 : Vector (Fin 3) ℝ :=
  ![2, -3, 5]

def v3 : Vector (Fin 3) ℝ :=
  ![5, 2, -1]

def v_sum : Vector (Fin 3) ℝ :=
  ![6, 3, 4]

def sum_vec : Vector (Fin 3) ℝ :=
  ![1, 1, 1]

theorem matrix_N_properties : N.mulVec i = v1 ∧ N.mulVec j = v2 ∧ N.mulVec k = v3 ∧ N.mulVec sum_vec = v_sum := by
  sorry

end matrix_N_properties_l527_527220


namespace sam_total_money_spent_l527_527019

def value_of_pennies (n : ℕ) : ℝ := n * 0.01
def value_of_nickels (n : ℕ) : ℝ := n * 0.05
def value_of_dimes (n : ℕ) : ℝ := n * 0.10
def value_of_quarters (n : ℕ) : ℝ := n * 0.25

def total_money_spent : ℝ :=
  (value_of_pennies 5 + value_of_nickels 3) +  -- Monday
  (value_of_dimes 8 + value_of_quarters 4) +   -- Tuesday
  (value_of_nickels 7 + value_of_dimes 10 + value_of_quarters 2) +  -- Wednesday
  (value_of_pennies 20 + value_of_nickels 15 + value_of_dimes 12 + value_of_quarters 6) +  -- Thursday
  (value_of_pennies 45 + value_of_nickels 20 + value_of_dimes 25 + value_of_quarters 10)  -- Friday

theorem sam_total_money_spent : total_money_spent = 14.05 :=
by
  sorry

end sam_total_money_spent_l527_527019


namespace kite_area_correct_l527_527594

noncomputable def kite_area (unit_length : ℕ) : ℕ :=
  let base_units := 6
  let height_units := 4
  let base := base_units * unit_length
  let height := height_units * unit_length
  let area_of_triangle := (base * height) / 2
  2 * area_of_triangle

theorem kite_area_correct :
  kite_area 2 = 96 :=
by
  let unit_length := 2
  let base_units := 6
  let height_units := 4
  let base := base_units * unit_length
  let height := height_units * unit_length
  let area_of_triangle := (base * height) / 2
  have h : 2 * area_of_triangle = 96 := by calc
    2 * ((base * height) / 2) = 2 * (12 * 8 / 2) : by rw [show base = 12, from rfl, show height = 8, from rfl]
    ... = 2 * 48 : by rw mul_div_cancel' (12 * 8) 2
    ... = 96 : by rw two_mul 48
  exact h

end kite_area_correct_l527_527594


namespace sachin_age_l527_527744
-- Import the necessary library

-- Lean statement defining the problem conditions and result
theorem sachin_age :
  ∃ (S R : ℝ), (R = S + 7) ∧ (S / R = 7 / 9) ∧ (S = 24.5) :=
by
  sorry

end sachin_age_l527_527744


namespace volume_of_tetrahedron_l527_527586

theorem volume_of_tetrahedron :
  (∫ x in 0..1, ∫ y in 0..1-x, ∫ z in 0..1-x-y, (1 : ℝ)) = 1 / 6 :=
by
  sorry

end volume_of_tetrahedron_l527_527586


namespace correctly_transformed_equation_l527_527117

theorem correctly_transformed_equation (s a b x y : ℝ) :
  (s = a * b → a = s / b ∧ b ≠ 0) ∧
  (1/2 * x = 8 → x = 16) ∧
  (-x - 1 = y - 1 → x = -y) ∧
  (a = b → a + 3 = b + 3) :=
by
  sorry

end correctly_transformed_equation_l527_527117


namespace solve_for_x_l527_527751

theorem solve_for_x :
  exists x : ℝ,
    log 3 ((4 * x + 8) / (6 * x - 5)) + log 3 ((6 * x - 5) / (2 * x - 1)) = 3 ∧
    x = 7 / 10 :=
by sorry

end solve_for_x_l527_527751


namespace product_bound_lt_two_l527_527948

theorem product_bound_lt_two {a : ℕ → ℝ} {n : ℕ} (hpos : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)
  (hsum : (Finset.range (n + 1)).sum a ≤ 1 / 2) :
  (Finset.range (n + 1)).prod (λ i, 1 + a i) < 2 :=
by sorry

end product_bound_lt_two_l527_527948


namespace grid_sum_l527_527669

theorem grid_sum (numbers : Fin 4 → Fin 4 → ℝ)
  (h : ∀ i j, (if i > 0 then numbers (i - 1) j else 0)
        + (if i < 3 then numbers (i + 1) j else 0)
        + (if j > 0 then numbers i (j - 1) else 0)
        + (if j < 3 then numbers i (j + 1) else 0)
        = 1) : ∑ i, ∑ j, numbers i j = 6 :=
by sorry

end grid_sum_l527_527669


namespace num_valid_noncongruent_triangles_l527_527902

theorem num_valid_noncongruent_triangles :
  ∃ (n : ℕ), n = 13 ∧ 
    (∀ (a b c : ℕ), a < b ∧ b < c ∧ a + b + c < 20 ∧ a + b > c ∧
    a^2 + b^2 ≠ c^2 → 
    (∃ (l : List (ℕ × ℕ × ℕ)), l.length = n ∧
      (∀ (t : ℕ × ℕ × ℕ), t ∈ l → t.1 < t.2 ∧ t.2 < t.3 ∧ 
        t.1 + t.2 + t.3 < 20 ∧ t.1 + t.2 > t.3 ∧ t.1^2 + t.2^2 ≠ t.3^2 ∧ 
        (∀ (d : ℕ × ℕ × ℕ), d ∈ l → d = t ∨ 
          ¬(t.1 = d.1 ∧ t.2 = d.2 ∧ t.3 = d.3 ∨ 
            t.1 = d.1 ∧ t.2 = d.3 ∧ t.3 = d.2 ∨ 
            t.1 = d.2 ∧ t.2 = d.1 ∧ t.3 = d.3 ∨ 
            t.1 = d.2 ∧ t.2 = d.3 ∧ t.3 = d.1 ∨ 
            t.1 = d.3 ∧ t.2 = d.1 ∧ t.3 = d.2 ∨ 
            t.1 = d.3 ∧ t.2 = d.2 ∧ t.3 = d.1)))))) :=
sorry

end num_valid_noncongruent_triangles_l527_527902


namespace parabola_and_latus_rectum_exists_line_parallel_to_OA_l527_527983

def parabola (p : ℝ) (p_pos : p > 0) : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * p * x

def point_on_parabola (x m p : ℝ) (p_pos : p > 0) : Prop :=
  m^2 = 2 * p * x

def distance_to_focus (x m p d : ℝ) : Prop :=
  sqrt ((x - p / 2)^2 + m^2) = d

def latus_rectum_eq (p : ℝ) (p_pos : p > 0) : Prop :=
  ∃ (x : ℝ), x = -p / 2

def line_parallel_to_OA (x y t : ℝ) : Prop :=
  y = -2 * x + t

def intersects_parabola (x y t : ℝ) (p : ℝ) (p_pos : p > 0) : Prop :=
  ∃ y, y^2 = 4 * x ∧ y = -2 * x + t

def distance_equals (t : ℝ) : Prop :=
  abs t / sqrt 5 = 1 / sqrt 5

def valid_t (t : ℝ) : Prop :=
  t = 1

theorem parabola_and_latus_rectum (p : ℝ) (p_pos : p > 0) (x m : ℝ) :
  point_on_parabola x m p p_pos →
  distance_to_focus x m p 2 →
  parabola p p_pos ∧ latus_rectum_eq p p_pos := sorry

theorem exists_line_parallel_to_OA (x y t : ℝ) (p : ℝ) (p_pos : p > 0) :
  point_on_parabola x y p p_pos →
  line_parallel_to_OA x y t →
  intersects_parabola x y t p p_pos →
  distance_equals t →
  valid_t t →
  ∃ line_eq : Prop, line_eq = (2 * x + y - 1 = 0) := sorry

end parabola_and_latus_rectum_exists_line_parallel_to_OA_l527_527983


namespace min_value_fracs_l527_527255

theorem min_value_fracs (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  (∃ (z : ℝ), z = (minimize z, z = (4 / (x + 2) + 1 / (y + 1)))) := sorry

end min_value_fracs_l527_527255


namespace base_6_arithmetic_l527_527863

theorem base_6_arithmetic : 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (r : ℕ) 
  (h1 : n1 = 35) (h2 : n2 = 14) (h3 : n3 = 20) (h4 : (n1 + n2 - n3) = r)
  (h5 : r = 33) : true :=
by
  sorry

end base_6_arithmetic_l527_527863


namespace binom_20_2_l527_527890

theorem binom_20_2 : nat.choose 20 2 = 190 := by
  sorry

end binom_20_2_l527_527890


namespace vanilla_syrup_cost_l527_527727

theorem vanilla_syrup_cost :
  ∀ (unit_cost_drip : ℝ) (num_drip : ℕ)
    (unit_cost_espresso : ℝ) (num_espresso : ℕ)
    (unit_cost_latte : ℝ) (num_lattes : ℕ)
    (unit_cost_cold_brew : ℝ) (num_cold_brews : ℕ)
    (unit_cost_cappuccino : ℝ) (num_cappuccino : ℕ)
    (total_cost : ℝ) (vanilla_cost : ℝ),
  unit_cost_drip = 2.25 →
  num_drip = 2 →
  unit_cost_espresso = 3.50 →
  num_espresso = 1 →
  unit_cost_latte = 4.00 →
  num_lattes = 2 →
  unit_cost_cold_brew = 2.50 →
  num_cold_brews = 2 →
  unit_cost_cappuccino = 3.50 →
  num_cappuccino = 1 →
  total_cost = 25.00 →
  vanilla_cost =
    total_cost -
    ((unit_cost_drip * num_drip) +
    (unit_cost_espresso * num_espresso) +
    (unit_cost_latte * (num_lattes - 1)) +
    (unit_cost_cold_brew * num_cold_brews) +
    (unit_cost_cappuccino * num_cappuccino)) →
  vanilla_cost = 0.50 := sorry

end vanilla_syrup_cost_l527_527727


namespace solution_inequality_l527_527900

open Set

theorem solution_inequality (x : ℝ) :
(\frac{(2*x-5)*(x-3)}{x} ≥ 0) ↔ x ∈ (Ioc 0 (5/2)] ∪ (Ici 3) :=
sorry

end solution_inequality_l527_527900


namespace part1_find_m_and_n_part1_monotonicity_part2_find_range_a_l527_527628

theorem part1_find_m_and_n (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = (mx + n) / (x^2 + 1)) → 
  odd f →
  f 1 = 1 →
  m = 2 ∧ n = 0 :=
sorry

theorem part1_monotonicity (f : ℝ → ℝ) :
  (∀ x, f x = (2x) / (x^2 + 1)) →
  (∀ x y ∈ Icc (-1) 1, x < y → f x < f y) :=
sorry

theorem part2_find_range_a (a : ℝ) :
  (∀ x, f x = (2x) / (x^2 + 1)) →
  (f (a-1) + f (a^2-1) < 0) →
  a ∈ Ico 0 1 :=
sorry

end part1_find_m_and_n_part1_monotonicity_part2_find_range_a_l527_527628


namespace part1_part2_l527_527969

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * (x + Real.pi / 6)) + 1

-- Statement for part 1
theorem part1 (ω : ℝ) (hω : 0 < ω) :
  ∀ x ∈ Set.Icc (-Real.pi / 4) (2 * Real.pi / 3), (2 * ω * Real.cos (ω * x) ≥ 0) → ω ≤ 2 / 3 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h_ab : a < b) : 
  ∃ (zeroes_count : ℕ), zeroes_count = 30 ∧ 
  let interval := (Real.range (λ k : ℕ, if k % 2 = 0 then k * Real.pi / 3 else k * 2 * Real.pi / 3)) 
  interval.contains (b - a) → (43 * Real.pi / 3 ≤ (b - a)) ∧ ((b - a) < 47 * Real.pi / 3) :=
sorry

end part1_part2_l527_527969


namespace exists_set_M_non_empty_finite_intersection_l527_527562
noncomputable def pointSetM : Set (ℝ × ℝ × ℝ) := { p | ∃ t : ℝ, p = (t, t^3, t^5) }

def isPlane (A B C D : ℝ) (p : ℝ × ℝ × ℝ) : Prop := 
  let (x, y, z) := p;
  A * x + B * y + C * z + D = 0

theorem exists_set_M_non_empty_finite_intersection :
  ∃ M : Set (ℝ × ℝ × ℝ), (∀ (δ : ℝ → ℝ → ℝ → ℝ), ∃ f : ℝ → Prop, (∀ p ∈ M, δ p.1 p.2 p.3 = 0 ↔ f p.1)) :=
by sorry

end exists_set_M_non_empty_finite_intersection_l527_527562


namespace sum_of_eight_smallest_multiples_of_12_l527_527113

theorem sum_of_eight_smallest_multiples_of_12 :
  let sum_n := (n : ℕ) → (n * (n + 1)) / 2
  12 * sum_n 8 = 432 :=
by
  sorry

end sum_of_eight_smallest_multiples_of_12_l527_527113


namespace difference_between_second_and_third_smallest_l527_527915

def digits := {1, 6, 8}

def nums : Finset ℕ := {168, 186, 618, 681, 816, 861}

def is_three_digit(n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def valid_num (n : ℕ) : Prop := 
  is_three_digit n ∧ 
  (n % 10 ∈ digits) ∧ 
  (n / 10 % 10 ∈ digits) ∧ 
  (n / 100 ∈ digits) ∧ 
  (n % 10 ≠ n / 10 % 10) ∧ 
  (n / 10 % 10 ≠ n / 100) ∧ 
  (n % 10 ≠ n / 100)

def smallest_nums : List ℕ := (nums.filter valid_num).sort (≤)

theorem difference_between_second_and_third_smallest :
  (smallest_nums.nth 1).get_or_else 0 - (smallest_nums.nth 2).get_or_else 0 = 432 := by
  sorry

end difference_between_second_and_third_smallest_l527_527915


namespace smallest_positive_integer_form_3003_55555_l527_527798

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l527_527798


namespace length_WZ_l527_527683

theorem length_WZ (X Y Z W : ℝ) 
  (h_right_angle : ∠XYZ = 90) 
  (YX : YX = 60) 
  (XZ : XZ = 80) 
  (WX_perpendicular : WX ⊥ YZ) 
  : WZ = 64 :=
sorry

end length_WZ_l527_527683


namespace find_divisor_l527_527317

-- Condition Definitions
def dividend : ℕ := 725
def quotient : ℕ := 20
def remainder : ℕ := 5

-- Target Proof Statement
theorem find_divisor (divisor : ℕ) (h : dividend = divisor * quotient + remainder) : divisor = 36 := by
  sorry

end find_divisor_l527_527317


namespace coupon_A_best_for_specified_prices_l527_527501

def is_coupon_A_better (p : ℝ) : Prop :=
  0.15 * p > 30 ∧ 0.15 * p > 0.25 * p - 37.5

def listed_prices : List ℝ := [169.95, 189.95, 209.95, 229.95, 249.95]

def better_prices := [209.95, 229.95, 249.95]

theorem coupon_A_best_for_specified_prices :
  ∀ p ∈ listed_prices, (is_coupon_A_better p ↔ p ∈ better_prices) :=
by
  intro p hp
  rw [is_coupon_A_better]
  split
  { intro hc
    sorry }
  { intro hc
    sorry }

end coupon_A_best_for_specified_prices_l527_527501


namespace part1_part2_l527_527507

theorem part1 (A ω φ : ℝ) (hA : A = 3) (hω : ω = 1/5) (hφ : φ = 3*π/10) :
  ∃ (f : ℝ → ℝ), f = λ x, A * sin (ω * x + φ) :=
by
  sorry

theorem part2 (A ω φ m : ℝ) (hA : A = 3) (hω : ω = 1/5) (hφ : φ = 3*π/10)
  (h_cond1 : 0 < ω) (h_ineq : A * sin (ω * sqrt (-m^2 + 2*m + 3) + φ) > A * sin (ω * sqrt (-m^2 + 4) + φ)) :
  1/2 < m ∧ m ≤ 2 :=
by
  sorry

end part1_part2_l527_527507


namespace range_of_a_l527_527980

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → x - Real.log x - a > 0) → a < 1 :=
sorry

end range_of_a_l527_527980


namespace minimum_value_is_12_solution_set_inequality_l527_527951

noncomputable def minimum_value (a b : ℝ) (h_b_gt_0 : b > 0) (h_ab_eq_3 : a * b = 3) : ℝ :=
  (a + 1) * (b + 3)

theorem minimum_value_is_12 (a b : ℝ) (h_b_gt_0 : b > 0) (h_ab_eq_3 : a * b = 3) :
  minimum_value a b h_b_gt_0 h_ab_eq_3 = 12 := sorry

theorem solution_set_inequality {M : ℝ} (h_M : M = 12) : 
  {x : ℝ | abs (3 * x + 3) - abs (2 * x - 3) < M} = set.Ioo (-18) 6 := sorry

end minimum_value_is_12_solution_set_inequality_l527_527951


namespace sum_of_values_x_l527_527560

noncomputable def find_sum_of_x : ℝ :=
  let valid_x_values := { x : ℝ | 0 < x ∧ x < 360 ∧ sin (2 * x)^2 + sin (4 * x)^2 = 4 * (sin (5 * x)^2) * (sin (6 * x)^2) } in
  ∑ x in valid_x_values, x

theorem sum_of_values_x (x : ℝ) (hx : 0 < x ∧ x < 360 ∧ sin (2 * x) ^ 2 + sin (4 * x) ^ 2 = 4 * sin (5 * x) ^ 2 * sin (6 * x) ^ 2) :
  find_sum_of_x = 1435 :=
  sorry

end sum_of_values_x_l527_527560


namespace cube_cut_cross_section_l527_527023

-- Define the points and distances on the edges of the cube
variables (A B C D A1 B1 C1 D1 P Q R S K L : Point)
variables (lAP : length A P = (1/3) * length A A1)
variables (lB1Q : length B1 Q = (1/2) * length B1 C1)
variables (lCR : length C R = (1/3) * length C D)

-- This theorem states that the intersection of the cube with the plane PQR forms the polygon SPKQLR
theorem cube_cut_cross_section
  (hP : P ∈ segment A A1)
  (hQ : Q ∈ segment B1 C1)
  (hR : R ∈ segment C D)
  (hSP : S ∈ segment (line_through P Q) (line_through R Q))
  (hK : K ∈ segment (line_through S Q))
  (hL : L ∈ segment (line_through Q R))
  : cross_section A B C D A1 B1 C1 D1 P Q R = polygon S P K Q L R :=
sorry

end cube_cut_cross_section_l527_527023


namespace money_distribution_l527_527832

theorem money_distribution (a b c : ℝ) (h1 : 4 * (a - b - c) = 16)
                           (h2 : 6 * b - 2 * a - 2 * c = 16)
                           (h3 : 7 * c - a - b = 16) :
  a = 29 := 
by 
  sorry

end money_distribution_l527_527832


namespace parametric_equations_solution_l527_527074

noncomputable def parametric_to_ordinary_equation : Prop :=
  ∀ θ : ℝ,
    let x := 2 + Real.sin θ ^ 2 in
    let y := -1 + Real.cos (2 * θ) in
    (2 * x + y - 4 = 0) ∧ (2 ≤ x ∧ x ≤ 3)

theorem parametric_equations_solution : parametric_to_ordinary_equation :=
sorry

end parametric_equations_solution_l527_527074


namespace teacher_earnings_l527_527191

theorem teacher_earnings (lessons_per_week : ℕ) (duration_per_lesson_hours : ℕ) (payment_per_half_hour : ℕ) (weeks : ℕ) :
  lessons_per_week = 1 →
  duration_per_lesson_hours = 1 →
  payment_per_half_hour = 10 →
  weeks = 5 →
  (duration_per_lesson_hours * 2 * payment_per_half_hour * weeks * lessons_per_week) = 100 :=
begin
  intros h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  norm_num,
end

end teacher_earnings_l527_527191


namespace min_people_needed_l527_527452

def smallCarWeight := 2000
def mediumCarWeight := 3000
def largeCarWeight := 4000
def lightTruckWeight := 10000
def heavyTruckWeight := 15000

def numSmallCars := 2
def numMediumCars := 2
def numLargeCars := 2
def numLightTrucks := 1
def numHeavyTrucks := 2

def personMaxLift := 1000

theorem min_people_needed : 
  let totalWeight := 
    (numSmallCars * smallCarWeight) + 
    (numMediumCars * mediumCarWeight) + 
    (numLargeCars * largeCarWeight) + 
    (numLightTrucks * lightTruckWeight) + 
    (numHeavyTrucks * heavyTruckWeight) in
  totalWeight / personMaxLift = 30 := by
  sorry

end min_people_needed_l527_527452


namespace seatingArrangements_l527_527450

theorem seatingArrangements (n m : ℕ) (h : n = 8) (k : m = 3) :
  (∃ s : ℕ, s = ((P(3, 3).val: ℕ) * (C(4, 3).val: ℕ))) :=
by
  have P_general := 3.factorial -- The number of permutations P(3,3)
  have C_general := Nat.choose 4 3 -- The number of combinations C(4,3)
  -- Combine them
  use (P_general * C_general)
  sorry

end seatingArrangements_l527_527450


namespace half_radius_of_circle_y_l527_527126

theorem half_radius_of_circle_y 
  (r_x r_y : ℝ) 
  (h₁ : π * r_x^2 = π * r_y^2) 
  (h₂ : 2 * π * r_x = 14 * π) :
  r_y / 2 = 3.5 :=
by {
  sorry
}

end half_radius_of_circle_y_l527_527126


namespace combined_volume_cone_hemisphere_cylinder_l527_527502

theorem combined_volume_cone_hemisphere_cylinder (r h : ℝ)
  (vol_cylinder : ℝ) (vol_cone : ℝ) (vol_hemisphere : ℝ)
  (H1 : vol_cylinder = 72 * π)
  (H2 : vol_cylinder = π * r^2 * h)
  (H3 : vol_cone = (1/3) * π * r^2 * h)
  (H4 : vol_hemisphere = (2/3) * π * r^3)
  (H5 : vol_cylinder = vol_cone + vol_hemisphere) :
  vol_cylinder = 72 * π :=
by
  sorry

end combined_volume_cone_hemisphere_cylinder_l527_527502


namespace levi_additional_baskets_to_score_l527_527371

def levi_scored_initial := 8
def brother_scored_initial := 12
def brother_likely_to_score := 3
def levi_goal_margin := 5

theorem levi_additional_baskets_to_score : 
  levi_scored_initial + 12 >= brother_scored_initial + brother_likely_to_score + levi_goal_margin :=
by
  sorry

end levi_additional_baskets_to_score_l527_527371


namespace graduating_class_total_students_l527_527318

noncomputable def total_students_in_class (geometry_taking : ℕ) (biology_taking : ℕ) (w_difference : ℕ) :=
  let x := biology_taking - w_difference in
  geometry_taking + biology_taking

theorem graduating_class_total_students :
  total_students_in_class 144 119 88 = 263 :=
by
  -- This is where you would complete the proof
  -- Assuming all conditions and definitions are accurate, using sorry for now
  sorry

end graduating_class_total_students_l527_527318


namespace inequality_sum_squares_l527_527005

theorem inequality_sum_squares
  (n : ℕ) (h_n : n ≥ 2)
  (a : ℕ → ℝ)
  (h_ordered : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → a i ≤ a j) :
  ∑ i in finset.range(n).filter(λ i, i ≠ 0).to_finset,      -- Sum over i < j
  ∑ j in finset.range(n).filter(λ j, j ≠ 0).to_finset, 
  (if i < j then (a i + a j)^2 * (1/i^2 + 1/j^2) else 0) 
  ≥ 4 * (n - 1) * ∑ i in finset.range(n).filter(λ i, i ≠ 0).to_finset, 
  (a i)^2 / i^2 :=
sorry

end inequality_sum_squares_l527_527005


namespace minimum_value_of_z_l527_527465

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l527_527465


namespace irreducible_fraction_l527_527230

-- Statement of the theorem
theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry -- Proof would be placed here

end irreducible_fraction_l527_527230


namespace triangle_similarity_l527_527488

-- Defining points on the circumference of a circle
variables {A B C D E F P Q R G H : Type*}

-- Assume all points lie on the circumference of a circle and chords AD, BE, CF are concurrent
axiom points_on_circle (c : Circle) : 
  A ∈ c ∧ B ∈ c ∧ C ∈ c ∧ D ∈ c ∧ E ∈ c ∧ F ∈ c

axiom concurrent_chords : Concurrent (line_through A D) (line_through B E) (line_through C F)

-- Assume P, Q, R are midpoints of AD, BE, and CF
axiom midpoints (hAD : is_chord A D) (hBE : is_chord B E) (hCF : is_chord C F) : 
  midpoint A D P ∧ midpoint B E Q ∧ midpoint C F R

-- Assume AG is parallel to BE and AH is parallel to CF
axiom parallel_chords : 
  Parallel (line_through A G) (line_through B E) ∧ 
  Parallel (line_through A H) (line_through C F)

-- Prove that triangle PQR is similar to triangle DGH
theorem triangle_similarity {c : Circle} 
    (h_co: points_on_circle c) 
    (h_concur: concurrent_chords) 
    (h_mid: midpoints (chord_AD := some hAD) (chord_BE := some hBE) (chord_CF := some hCF)) 
    (h_par: parallel_chords) : 
    Similar (triangle P Q R) (triangle D G H) :=
sorry

end triangle_similarity_l527_527488


namespace isosceles_right_triangle_area_l527_527436

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hypotenuse_condition : h = 6 * real.sqrt 2) (leg_relationship : h = l * real.sqrt 2) :
  (1 / 2) * l * l = 18 :=
by
  -- setting the hypotheses in a more readable form
  have h1 : h = 6 * real.sqrt 2 := hypotenuse_condition,
  have h2 : h = l * real.sqrt 2 := leg_relationship,
  sorry

end isosceles_right_triangle_area_l527_527436


namespace median_of_list_i_l527_527320

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted_l := l.sort
  if h : sorted_l.length % 2 = 0 then
    (sorted_l.get! (sorted_l.length / 2 - 1) + sorted_l.get! (sorted_l.length / 2)) / 2
  else
    sorted_l.get! (sorted_l.length / 2)

noncomputable def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) (l.head!)

theorem median_of_list_i :
  median list_i = 8 :=
by
  have h_median_list_ii : median list_ii = 5 := by sorry
  have h_mode_list_ii : mode list_ii = 3 := by sorry
  have h_condition : median list_i = median list_ii + mode list_ii := by sorry
  rw [h_condition, h_median_list_ii, h_mode_list_ii]
  norm_num

end median_of_list_i_l527_527320


namespace range_of_a_l527_527274

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ set.Ici 3, f x = x * abs (x - a)) →
  (∀ x1 x2 ∈ set.Ici 3, (x1 ≠ x2) → ((f x1 - f x2) / (x1 - x2) > 0)) →
  a ≤ 3 :=
sorry

end range_of_a_l527_527274


namespace min_distance_circle_parabola_l527_527367

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 - 16 * x + 64 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The main theorem to prove
theorem min_distance_circle_parabola :
  ∀ (P Q : ℝ × ℝ), circle P.1 P.2 → parabola Q.1 Q.2 → dist P Q = 0 :=
by
  sorry -- Proof not required

end min_distance_circle_parabola_l527_527367


namespace correct_choice_l527_527811

-- Define what it means to be a polynomial
def is_polynomial (e : ℕ → ℕ → ℚ) : Prop :=
  ∀ m n, e m n = 0

-- Define each statement from the conditions
def A := ¬is_polynomial (λ m n, 1/4 * m^2 * n)
def B :=  ∀ a b : ℚ, (coeff (-2 * π * a * b / 5) = -2 / 5)
def C := ∀ x : ℚ, (degree (x^4 + 2 * x^3) = 7)
def D := is_polynomial (λ x, (3*x - 1)/5)

-- The correct answer is D
theorem correct_choice : D := 
by
  sorry


end correct_choice_l527_527811


namespace intersection_distance_l527_527270

noncomputable theory

-- Definitions to be used in the proof
variables {m n : ℝ}
variables (x y : ℝ)

-- Conditions given in the problem
def ellipse (m : ℝ) : Prop := m > 0 ∧ (x^2)/(m^2) + (y^2)/9 = 1

def hyperbola (n : ℝ) : Prop := n > 0 ∧ (x^2)/(n^2) - (y^2)/4 = 1

def same_foci (m n : ℝ) : Prop := m^2 - n^2 = 13

def intersection_point (x y : ℝ) (m n : ℝ) : Prop := 
  ellipse m ∧ hyperbola n ∧ ∃ (F1 F2 : ℝ), |x - F1| * |x - F2| = 13

theorem intersection_distance (m n : ℝ) 
    (h_ellipse : ellipse m) 
    (h_hyperbola : hyperbola n) 
    (h_same_foci : same_foci m n) 
    (h_intersection : intersection_point x y m n) :
  |(x - y)| * |(x - y)| = 13 := 
sorry

end intersection_distance_l527_527270


namespace second_discount_percentage_l527_527777

variable (P : ℝ) (D₁ : ℝ) (S : ℝ) (D₂ : ℝ)

-- Conditions
def original_price := 400
def first_discount_percentage := 12
def sale_price := 334.4
def applied_first_discount := original_price - (first_discount_percentage / 100) * original_price
def applied_second_discount := applied_first_discount - (D₂ / 100) * applied_first_discount

-- Goal
theorem second_discount_percentage :
  applied_second_discount = sale_price → D₂ = 5 := by
  sorry

end second_discount_percentage_l527_527777


namespace sum_1000_terms_eq_2050_l527_527196

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 3 = 1 then 1
  else if n % 3 = 2 then (n / 3) + 2
  else (n / 3) + 3

def sum_first_n_terms (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence

theorem sum_1000_terms_eq_2050 : 
  sum_first_n_terms 1000 = 2050 :=
by
  sorry

end sum_1000_terms_eq_2050_l527_527196


namespace scientific_notation_l527_527416

-- Define the condition: the area of each probe unit of a certain chip is 0.00000164 cm^2
def area_probe_unit : ℝ := 0.00000164

-- The mathematical problem to express 0.00000164 in scientific notation
theorem scientific_notation :
  area_probe_unit = 1.64 * 10 ^ (-6) := 
  by
    sorry

end scientific_notation_l527_527416


namespace find_a_l527_527626

def f (x : ℝ) : ℝ := if h : ∃ y : ℝ, x = 2*y + 1 then (λ y, 3*y + 2) (classical.some h) else 0

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, f (2*x + 1) = 3*x + 2) (h2 : f a = 4) : 
  a = 7 / 3 := 
by
  -- proof would go here
  sorry

end find_a_l527_527626


namespace has_three_distinct_real_roots_l527_527559

noncomputable def Q (p x : ℝ) : ℝ := x^3 + p * x^2 - p * x - 1

def discriminant_condition (p : ℝ) : Prop :=
  let Δ := (p + 1)^2 - 4 in Δ > 0

theorem has_three_distinct_real_roots (p : ℝ) :
  (discriminant_condition p ↔ (p > 1 ∨ p < -3)) :=
sorry

end has_three_distinct_real_roots_l527_527559


namespace max_determinant_value_l527_527884

noncomputable def max_determinant (θ φ : ℝ) : ℝ :=
|1 1 1|,
|1 1 + sin θ 1 + cos φ|,
|1 + cos θ 1 + sin φ 1|

theorem max_determinant_value : (∃ θ φ, max_determinant θ φ = 2) ∧
  (∀ θ φ, max_determinant θ φ ≤ 2) :=
sorry

end max_determinant_value_l527_527884


namespace area_S_rhombus_l527_527743

noncomputable def area_of_S (side_length : ℝ) (angle_F : ℝ) : ℝ :=
  let a := 4 
  let F_angle := 150
  if side_length = a ∧ angle_F = F_angle then (4 * Real.sqrt 3) / 3 else 0

theorem area_S_rhombus :
  ∀ (EFGH : Type) (S : EFGH → Prop), 
  (∀ x, S x ↔ ∃ (F : EFGH) (a : ℝ) (θ : ℝ), 
    a = 4 ∧ θ = 150 ∧ x inside_rhombus EFGH ∧ (dist x F < min (dist x E) (dist x G) (dist x H))) 
  → area_of_S 4 150 = (4 * Real.sqrt 3) / 3 := by
  sorry

end area_S_rhombus_l527_527743


namespace find_width_of_iron_bar_l527_527874

noncomputable def width_of_iron_bar
  (length : ℝ) (height : ℝ) 
  (num_iron_bars : ℝ) (num_iron_balls : ℝ) (volume_one_ball : ℝ) : ℝ :=
  let volume_iron_balls := num_iron_balls * volume_one_ball in
  let volume_one_iron_bar := length * height * sorry in
  have volume_iron_balls = num_iron_bars * (length * height * sorry), from sorry
  sorry 

theorem find_width_of_iron_bar
  (length : ℝ := 12)
  (height : ℝ := 6)
  (num_iron_bars : ℝ := 10)
  (num_iron_balls : ℝ := 720)
  (volume_one_ball : ℝ := 8) : 
  width_of_iron_bar length height num_iron_bars num_iron_balls volume_one_ball = 48 :=
  sorry

end find_width_of_iron_bar_l527_527874


namespace smallest_positive_integer_linear_combination_l527_527803

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l527_527803


namespace B_finishes_job_in_22_5_days_l527_527846

variable (A B : Type) [CommGroup A]

-- Condition: A is half as good a workman as B
def work_per_day_A (x : ℝ) : ℝ := 1 / (2 * x)
def work_per_day_B (x : ℝ) : ℝ := 1 / x

-- Condition: Together, A and B finish a job in 15 days
theorem B_finishes_job_in_22_5_days : 
  ∃ x : ℝ, work_per_day_A A x + work_per_day_B B x = 1 / 15 ∧ x = 22.5 :=
by
  -- This is a place holder for the proof.
  sorry

end B_finishes_job_in_22_5_days_l527_527846


namespace father_l527_527020

theorem father's_age_at_middle_son_birth (a b c F : ℕ) 
  (h1 : a = b + c) 
  (h2 : F * a * b * c = 27090) : 
  F - b = 34 :=
by sorry

end father_l527_527020


namespace seth_initial_boxes_l527_527040

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end seth_initial_boxes_l527_527040


namespace part1_part2_l527_527236

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem part1 (a : ℝ) : (∃ x : ℝ, f x < 2 * a - 1) ↔ a > 2 :=
by sorry

theorem part2 : {x : ℝ | f x ≥ x ^ 2 - 2 * x} = {x | x = -1} ∪ {x | -1 < x ∧ x ≤ 2 + real.sqrt 3} :=
by sorry

end part1_part2_l527_527236


namespace jessica_marbles_62_l527_527340

-- Definitions based on conditions
def marbles_kurt (marbles_dennis : ℕ) : ℕ := marbles_dennis - 45
def marbles_laurie (marbles_kurt : ℕ) : ℕ := marbles_kurt + 12
def marbles_jessica (marbles_laurie : ℕ) : ℕ := marbles_laurie + 25

-- Given marbles for Dennis
def marbles_dennis : ℕ := 70

-- Proof statement: Prove that Jessica has 62 marbles given the conditions
theorem jessica_marbles_62 : marbles_jessica (marbles_laurie (marbles_kurt marbles_dennis)) = 62 := 
by
  sorry

end jessica_marbles_62_l527_527340


namespace expected_value_uniform_l527_527916

open Real MeasureTheory

noncomputable def uniform_pdf (α β : ℝ) (x : ℝ) : ℝ :=
  if (α ≤ x ∧ x ≤ β) then 1 / (β - α) else 0

theorem expected_value_uniform {α β : ℝ} (hαβ : α < β) :
  ∫ x in α..β, x * (uniform_pdf α β x) = (β + α) / 2 := by
  sorry

end expected_value_uniform_l527_527916


namespace misha_problem_l527_527689

theorem misha_problem (N : ℕ) (h : ∀ a, a ∈ {a | a > 1 → ∃ b > 0, b ∈ {b' | b' < a ∧ a % b' = 0}}) :
  (∀ t : ℕ, (t > 1) → (1 / t ^ 2) < (1 / t * (t - 1))) →
  (∃ (n : ℕ), n = 1) → (N = 1 ↔ ∃ (k : ℕ), k = N^2) :=
by
  sorry

end misha_problem_l527_527689


namespace area_APQ_l527_527247

-- Define the given conditions and the problem
def triangle := {A : ℝ × ℝ, B : ℝ × ℝ, C : ℝ × ℝ}

theorem area_APQ (A B C P Q : ℝ × ℝ)
  (hABC : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hAB : dist A B = 1) (hBC : dist B C = 1) (hAC : dist A C = 1)
  (hP : P = (1/3) • (B - A) + (1/3) • (C - A) + A)
  (hQ : Q = P + (1/2) • (C - B)) :
  let area := (1/2) * abs ((P.1 * Q.2 + Q.1 * A.2 + A.1 * P.2) 
                          - (P.2 * Q.1 + Q.2 * A.1 + A.2 * P.1)) in
  area = sqrt 3 / 12 := sorry

end area_APQ_l527_527247


namespace mrs_garcia_insurance_payments_l527_527380

/-- Mrs. Garcia pays her insurance at $378 per quarter. She pays a certain number of times in a year for her insurance. Prove that Mrs. Garcia pays her insurance 4 times a year if the total amount is $1512. -/
def insurance_payments_per_year (quarter_payment annual_payment : ℕ) (h1 : quarter_payment = 378) (h2 : annual_payment = 1512) : ℕ :=
  annual_payment / quarter_payment

theorem mrs_garcia_insurance_payments :
  insurance_payments_per_year 378 1512 378 1512 = 4 :=
by
  simp [insurance_payments_per_year]
  sorry

end mrs_garcia_insurance_payments_l527_527380


namespace petes_total_books_read_l527_527657

-- Let's define the necessary conditions
def matts_books_first_year (M : ℝ) : Prop := true -- M is the number of books Matt read in the first year
def petes_books_first_year (P : ℝ) (M : ℝ) : Prop := P = 2 * M -- Pete read twice as many books as Matt in the first year
def matts_books_second_year (M : ℝ) : Prop := 1.5 * M = 75 -- Matt read 75 books in second year, which is 50% more than the first year
def petes_books_second_year (P : ℝ) : Prop := P * 2 -- Pete doubles his reads from last year in the second year
def total_books_pete_reads (P1 P2 T : ℝ) : Prop := T = P1 + P2 -- sum of books across both years

-- Main theorem
theorem petes_total_books_read (M P1 P2 T : ℝ) :
  matts_books_first_year M →
  petes_books_first_year P1 M →
  matts_books_second_year M →
  petes_books_second_year P2 →
  total_books_pete_reads P1 P2 T →
  T = 300 :=
by
  sorry

end petes_total_books_read_l527_527657


namespace ω_squared_plus_ω_plus_one_eq_zero_l527_527239

def ω : ℂ := -1 / 2 + (Complex.I * (Real.sqrt 3 / 2))

theorem ω_squared_plus_ω_plus_one_eq_zero : ω^2 + ω + 1 = 0 :=
by 
  sorry

end ω_squared_plus_ω_plus_one_eq_zero_l527_527239


namespace total_oranges_l527_527386

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l527_527386


namespace vector_magnitude_problem_l527_527643

open Real

noncomputable def magnitude (x : ℝ × ℝ) : ℝ := sqrt (x.1 ^ 2 + x.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h_a : a = (1, 3))
  (h_perp : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) :
  magnitude b = sqrt 10 := 
sorry

end vector_magnitude_problem_l527_527643


namespace derivative_of_h_l527_527577

noncomputable def h (x : ℝ) : ℝ := Real.exp (-x / 3)

theorem derivative_of_h : deriv h = λ x, -1 / 3 * Real.exp (-x / 3) := by
  sorry

end derivative_of_h_l527_527577


namespace problem_statement_l527_527370

-- Definitions from conditions in the problem
def S (n : ℕ) : ℕ := n^2 - 4 * n + 4

def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * n - 5

def b (n : ℕ) : ℚ :=
  a n / 2^n

def T (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b (i + 1))

-- Proof problem statement
theorem problem_statement (n : ℕ) : 1 ≤ n → 1/4 ≤ T n ∧ T n < 1 :=
begin
  sorry
end

end problem_statement_l527_527370


namespace find_f_2019_l527_527433

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h1 : ∀ x : ℝ, f (-x) = -f x) : Prop
axiom functional_equation (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x)) : Prop
axiom initial_value (h3 : f 1 = 9) : Prop

theorem find_f_2019 (h1 : odd_function) (h2 : functional_equation) (h3 : initial_value) : f 2019 = -9 := 
    sorry

end find_f_2019_l527_527433


namespace arithmetic_sequence_S30_l527_527679

theorem arithmetic_sequence_S30 (h1 : ∀ n, S n = S (n - 1) + a) 
                                 (h2 : S 10 = 10)
                                 (h3 : S 20 = 30) :
                                 S 30 = 60 :=
by
  sorry

end arithmetic_sequence_S30_l527_527679


namespace car_speed_l527_527820

theorem car_speed (v : ℝ) (hv : 2 + (1 / v) * 3600 = (1 / 90) * 3600) :
  v = 600 / 7 :=
sorry

end car_speed_l527_527820


namespace mutually_exclusive_events_l527_527931

namespace BallDrawing

inductive Ball
| red
| white

open Ball

def bag : list Ball := [red, red, white, white]

def draw_two (balls : list Ball) :=
  { draw : list Ball // draw.length = 2 }

def event_1 (draw : list Ball) : Prop :=
  (∃ (b : Ball), b ∈ draw ∧ b = white) ∧ draw = [white, white]

def event_2 (draw : list Ball) : Prop :=
  (∃ (b : Ball), b ∈ draw ∧ b = white) ∧ (∃ (b : Ball), b ∈ draw ∧ b = red)

def event_3 (draw : list Ball) : Prop :=
  (∃ (b : Ball), b ∈ draw ∧ b = white ∧ draw.length = 1) ∧ draw = [white, white]

def event_4 (draw : list Ball) : Prop :=
  (∃ (b : Ball), b ∈ draw ∧ b = white) ∧ draw = [red, red]

def mutually_exclusive (A B : list Ball → Prop) : Prop :=
  ∀ d, A d → ¬ B d

theorem mutually_exclusive_events : 
  ∃ (e1 e2 : list Ball → Prop) (p : nat), 
    ((e1 = event_1 ∨ e1 = event_2 ∨ e1 = event_3 ∨ e1 = event_4) ∧ 
     (e2 = event_1 ∨ e2 = event_2 ∨ e2 = event_3 ∨ e2 = event_4) ∧ 
     e1 ≠ e2 ∧ mutually_exclusive e1 e2) ∧ p = 2 :=
by
  -- We skip the proof here, which should be based on the solution analysis and the conditions.
  sorry

end BallDrawing

end mutually_exclusive_events_l527_527931


namespace sum_arithmetic_sequence_l527_527302

theorem sum_arithmetic_sequence (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k < n → x (k + 1) = x k + 1 / 2) (h2 : x 1 = 1) :
  (∑ i in Finset.range n, x i) = (n^2 + 3 * n) / 4 :=
sorry

end sum_arithmetic_sequence_l527_527302


namespace equation_has_infinite_solutions_l527_527402

open Nat

theorem equation_has_infinite_solutions :
  ∃ (f : ℕ → ℕ × ℕ × ℕ × ℕ), injective f ∧ ∀ n, let ⟨x, y, z, t⟩ := f n in x^2 + y^2 = 5 * (z^2 + t^2) :=
by
  sorry

end equation_has_infinite_solutions_l527_527402


namespace lamp_flashlight_prices_max_desk_lamps_l527_527142

theorem lamp_flashlight_prices :
  ∃ (x y : ℝ), y = x + 50 ∧ 240 / y = 90 / x ∧ x = 30 ∧ y = 80 :=
begin
  sorry
end

theorem max_desk_lamps (a : ℝ) :
  ∃ (a : ℝ), 80 * a + 30 * (a + 8) ≤ 2440 ∧ a ≤ 20 :=
begin
  sorry
end

end lamp_flashlight_prices_max_desk_lamps_l527_527142


namespace find_b_value_l527_527483

theorem find_b_value (n : ℝ) (b : ℝ) (h₁ : n = 2 ^ 0.25) (h₂ : n ^ b = 8) : b = 12 := sorry

end find_b_value_l527_527483


namespace constant_distance_for_perpendicular_rays_through_origin_l527_527607

open Real

variables (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : (sqrt 6) / 3 = sqrt (1 - (b/a)^2)) 
  (h₄ : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → x * x + y * y ≥ 0)

theorem constant_distance_for_perpendicular_rays_through_origin :
  let C := {p : ℝ × ℝ | (p.1)^2 / 3 + (p.2)^2 = 1 },
      dist (O : ℝ × ℝ) (line : ℝ × ℝ → Prop) :=
        abs(line O) / sqrt (line.1^2 + line.2^2)
  in 
  ∀ O A B : ℝ × ℝ,
    C A →
    C B →
    A ≠ B →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (dist O (λ p => p.2 - ((B.2 - A.2) / (B.1 - A.1)) * (p.1 - A.1) - A.2) = sqrt (3) / 2) ∨
    (dist O (λ p => p.1 = 0) = sqrt (3) / 2) :=
sorry

end constant_distance_for_perpendicular_rays_through_origin_l527_527607


namespace purple_chip_value_l527_527671

theorem purple_chip_value :
  let points (color : String) : ℕ :=
    if color = "blue" then 1 
    else if color = "green" then 5 
    else if color = "red" then 11 
    else if color = "purple" then sorry -- Purple value to be proven
    else 0 in
  (points "purple" > 5 ∧ points "purple" < 11) →
  (points "blue" * points "green" * points "purple" * points "red" = 140800) →
  points "purple" = 7 :=
by
  sorry

end purple_chip_value_l527_527671


namespace down_jacket_price_reduction_proof_l527_527857

def down_jacket_price_reduction : ℝ :=
  let average_daily_sale := 20
  let profit_per_piece := 40
  let additional_sales_per_unit := 2
  let target_profit := 1200
  let daily_profit (x : ℝ) := (profit_per_piece - x) * (average_daily_sale + additional_sales_per_unit * x)
  let equation := target_profit = daily_profit 20
  if h : equation then 20 else 0

theorem down_jacket_price_reduction_proof : down_jacket_price_reduction = 20 :=
by
  sorry

end down_jacket_price_reduction_proof_l527_527857


namespace bread_left_in_pond_l527_527783

theorem bread_left_in_pond (total_bread : ℕ) 
                           (half_bread_duck : ℕ)
                           (second_duck_bread : ℕ)
                           (third_duck_bread : ℕ)
                           (total_bread_thrown : total_bread = 100)
                           (half_duck_eats : half_bread_duck = total_bread / 2)
                           (second_duck_eats : second_duck_bread = 13)
                           (third_duck_eats : third_duck_bread = 7) :
                           total_bread - (half_bread_duck + second_duck_bread + third_duck_bread) = 30 :=
    by
    sorry

end bread_left_in_pond_l527_527783


namespace eval_custom_op_expr_l527_527651

-- Definition of the custom operation
def custom_op (A B : ℝ) : ℝ := (A + B) / 2

-- Theorem statement proving the given mathematical problem
theorem eval_custom_op_expr : custom_op (custom_op 10 20) (custom_op 5 15) = 12.5 :=
by
  -- Proof skipped with sorry for now
  sorry

end eval_custom_op_expr_l527_527651


namespace part1_part2_l527_527277

def f (x : ℝ) := |x - 1|

theorem part1 (x : ℝ) :
  f(x - 1) + f(1 - x) ≤ 2 ↔ (0 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part2 (a x : ℝ) (h : a < 0) :
  f(ax) - a * f(x) ≥ f(a) :=
by
  sorry

end part1_part2_l527_527277


namespace actual_price_of_good_l527_527172

theorem actual_price_of_good (P : ℝ) (h : 0 < P) :
  let p_after_first_discount := 0.80 * P,
      p_after_second_discount := 0.80 * P * 0.90,
      p_after_third_discount := 0.80 * P * 0.90 * 0.95
  in p_after_third_discount = 6700 → P = 9798.25 :=
by
  intros p_after_first_discount p_after_second_discount p_after_third_discount h_eq
  have : P = 6700 / (0.80 * 0.90 * 0.95) := sorry
  rw [this]
  norm_num
  sorry

end actual_price_of_good_l527_527172


namespace minimize_MP_2MF_l527_527967

theorem minimize_MP_2MF :
  ∃ M: ℝ × ℝ, 
    ((M.1^2 / 4 + M.2^2 / 3 = 1) ∧ 
     let P := (1, -1) in
     let F := (1, 0) in
     (M = (2 * Real.sqrt 6 / 3, -1) ∧
     (∀ N: ℝ × ℝ, (N.1^2 / 4 + N.2^2 / 3 = 1) → (|M.1 - P.1| + |M.2 - P.2| + 2 * (|M.1 - F.1| + |M.2 - F.2|)) ≤ (|N.1 - P.1| + |N.2 - P.2| + 2 * (|N.1 - F.1| + |N.2 - F.2|))))) :=
begin
  sorry
end

end minimize_MP_2MF_l527_527967


namespace simplest_fraction_unique_l527_527868

theorem simplest_fraction_unique {x y : ℝ} :
  (∀ a b c d : ℝ, 
    (a / b).simplified = c / d → 
      (∀ (a : ℝ), a = x^2 + xy → 
                  b = x^2 + 2xy + y^2 →
                  c = x →
                  d = x + y → False) ∧
      (∀ (a : ℝ), a = 2x + 8 →
                  b = x^2 - 16 →
                  c = 2 →
                  d = x - 4 → False) ∧
      (∀ (a : ℝ), a = x^2 + 1 →
                  b = x^2 - 1) ∧
      (∀ (a : ℝ), a = x^2 - 9 →
                  b = x^2 + 6x + 9 →
                  c = x - 3 →
                  d = x + 3 → False)) →
  ∃ c d : ℝ, c = x^2 + 1 ∧ d = x^2 - 1 :=
by
  sorry

end simplest_fraction_unique_l527_527868


namespace bounded_sequence_is_constant_two_l527_527215

noncomputable def infinite_bounded_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))) ∧ 
  (∃ d, ∀ n, a n = d)

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) :
  infinite_bounded_sequence a → (∀ n, a n = 2) :=
by
  intro h,
  sorry

end bounded_sequence_is_constant_two_l527_527215


namespace female_students_in_sample_l527_527315

/-- In a high school, there are 500 male students and 400 female students in the first grade. 
    If a random sample of size 45 is taken from the students of this grade using stratified sampling by gender, 
    the number of female students in the sample is 20. -/
theorem female_students_in_sample 
  (num_male : ℕ) (num_female : ℕ) (sample_size : ℕ)
  (h_male : num_male = 500)
  (h_female : num_female = 400)
  (h_sample : sample_size = 45)
  (total_students : ℕ := num_male + num_female)
  (sample_ratio : ℚ := sample_size / total_students) :
  num_female * sample_ratio = 20 := 
sorry

end female_students_in_sample_l527_527315


namespace proof_problem_l527_527950

variables (p q : Prop)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : ¬ p ∨ ¬ q :=
by
  sorry

end proof_problem_l527_527950


namespace total_number_of_letters_l527_527703

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l527_527703


namespace negation_of_exists_proposition_l527_527071

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
sorry

end negation_of_exists_proposition_l527_527071


namespace min_value_frac_sin_cos_l527_527261

open Real

theorem min_value_frac_sin_cos (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ m : ℝ, (∀ x : ℝ, x = (1 / (sin α)^2 + 3 / (cos α)^2) → x ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
by
  have h_sin_cos : sin α ≠ 0 ∧ cos α ≠ 0 := sorry -- This is an auxiliary lemma in the process, a proof is required.
  sorry

end min_value_frac_sin_cos_l527_527261


namespace limit_geometric_sequence_sum_l527_527624

noncomputable def Sn (n : ℕ) : ℝ :=
  (1 / 2) ^ n - 1

def a_n (n : ℕ) : ℝ :=
  if n = 1 then Sn 1
  else if n = 2 then Sn 2 - Sn 1
  else (Sn n - Sn (n - 1))

theorem limit_geometric_sequence_sum :
  tendsto (λ n, ∑ k in range (2 * n), if k % 2 = 0 then 0 else a_n (k + 1)) at_top (𝓝 (-2 / 3)) :=
by
  sorry

end limit_geometric_sequence_sum_l527_527624


namespace tan_120_eq_neg_sqrt_3_l527_527891

theorem tan_120_eq_neg_sqrt_3 
  (P : ℝ × ℝ) 
  (h1 : P = (-1/2, sqrt(3)/2)) 
  (h2 : P ∈ metric.sphere (0, 0) 1) :
  real.tan (2 * real.pi / 3) = -sqrt(3) := by
  sorry

end tan_120_eq_neg_sqrt_3_l527_527891


namespace math_proof_problem_l527_527615

open Real

noncomputable def solve (theta : ℝ) (h : (4 * sin theta - 2 * cos theta) / (3 * sin theta + 5 * cos theta) = 6 / 11) : Prop :=
  prop_1 theta h ∧ prop_2 theta h

noncomputable def prop_1 (theta : ℝ) (h : (4 * sin theta - 2 * cos theta) / (3 * sin theta + 5 * cos theta) = 6 / 11) : Prop :=
  (5 * cos theta ^ 2) / (sin theta ^ 2 + 2 * sin theta * cos theta - 3 * cos theta ^ 2) = 1

noncomputable def prop_2 (theta : ℝ) (h : (4 * sin theta - 2 * cos theta) / (3 * sin theta + 5 * cos theta) = 6 / 11) : Prop :=
  1 - 4 * sin theta * cos theta + 2 * cos theta ^ 2 = - 1 / 5

theorem math_proof_problem : ∃ (theta : ℝ) (h : (4 * sin theta - 2 * cos theta) / (3 * sin theta + 5 * cos theta) = 6 / 11), solve theta h :=
sorry

end math_proof_problem_l527_527615


namespace mean_score_is_93_l527_527336

-- Define Jane's scores as a list
def scores : List ℕ := [98, 97, 92, 85, 93]

-- Define the mean of the scores
noncomputable def mean (lst : List ℕ) : ℚ := 
  (lst.foldl (· + ·) 0 : ℚ) / lst.length

-- The theorem to prove
theorem mean_score_is_93 : mean scores = 93 := by
  sorry

end mean_score_is_93_l527_527336


namespace find_n_l527_527650

theorem find_n (n : ℕ) (h : sqrt (10 + n) = 8) : n = 54 :=
sorry

end find_n_l527_527650


namespace find_number_l527_527808

theorem find_number:
  ∃ x: ℕ, (∃ k: ℕ, ∃ r: ℕ, 5 * (x + 3) = 8 * k + r ∧ k = 156 ∧ r = 2) ∧ x = 247 :=
by 
  sorry

end find_number_l527_527808


namespace artist_used_17_ounces_of_paint_l527_527871

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l527_527871


namespace ratio_of_areas_l527_527033

-- Definitions of the perimeters for each region
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 36
def perimeter_IV : ℕ := 48

-- Define the side lengths based on the given perimeters
def side_length (P : ℕ) : ℕ := P / 4

-- Calculate the areas from the side lengths
def area (s : ℕ) : ℕ := s * s

-- Now we state the theorem
theorem ratio_of_areas : 
  (area (side_length perimeter_II)) / (area (side_length perimeter_IV)) = 9 / 16 := 
by sorry

end ratio_of_areas_l527_527033


namespace intersection_point_on_angle_bisector_l527_527601

-- Definitions of the required objects
variables {A B C D I J I_a J_a K : Type}

-- Assuming the given conditions as hypotheses
variables (ABCD_convex : convex_quadrilateral A B C D)
  (I_center : incenter_of_triangle I A B C)
  (J_center : incenter_of_triangle J A D C)
  (I_a_excenter : excenter_of_triangle I_a A B C)
  (J_a_excenter : excenter_of_triangle J_a A D C)
  (K_intersection : intersection_point K I J_a J I_a)

-- Proving that K lies on the angle bisector of angle BCD
theorem intersection_point_on_angle_bisector :
  lies_on_angle_bisector K A B C D :=
sorry

end intersection_point_on_angle_bisector_l527_527601


namespace find_a_7_l527_527327

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ} -- The sequence a_n
variable (a_4_eq : a 4 = 4)
variable (a_3_a_8_eq : a 3 + a 8 = 5)

-- Prove that a_7 = 1
theorem find_a_7 : a 7 = 1 := by
  sorry

end find_a_7_l527_527327


namespace overall_loss_is_4_l527_527645

variables (C1 C2 : ℝ)
variables (total_cost loss_percent gain_percent : ℝ)
variable (total_selling_price : ℝ)

-- Given conditions
def condition1 : Prop := C1 = 280
def condition2 : Prop := C1 + C2 = 480
def condition3 : Prop := loss_percent = 0.15
def condition4 : Prop := gain_percent = 0.19

-- Definition of loss and selling price on Book 1
def loss_on_book1 := loss_percent * C1
def selling_price1 := C1 - loss_on_book1

-- Definition of gain and selling price on Book 2
def gain_on_book2 := gain_percent * C2
def selling_price2 := C2 + gain_on_book2

-- Total selling price
def total_selling_price := selling_price1 + selling_price2

-- Overall loss
def overall_loss := (C1 + C2) - total_selling_price

-- Proof problem statement
theorem overall_loss_is_4
  (h1 : condition1)
  (h2 : condition2)
  (h3 : loss_percent = 0.15)
  (h4 : gain_percent = 0.19) :
  overall_loss = 4 :=
sorry

end overall_loss_is_4_l527_527645


namespace lines_perpendicular_iff_a_l527_527924

def direction_vector1 : ℝ × ℝ × ℝ := (a, -3, 2)
def direction_vector2 : ℝ × ℝ × ℝ := (2, 1, 3)

theorem lines_perpendicular_iff_a :
  (direction_vector1.1 * direction_vector2.1 + direction_vector1.2 * direction_vector2.2 + direction_vector1.3 * direction_vector2.3 = 0) ↔ a = -3/2 := 
sorry

end lines_perpendicular_iff_a_l527_527924


namespace divisors_condition_l527_527343

def f (x : ℕ) : ℕ := x^2 + x + 1

theorem divisors_condition (n : ℕ) : (∀ k : ℕ, k > 0 ∧ k ∣ n → f(k) ∣ f(n)) ↔ 
  (n = 1 ∨ 
  (nat.prime n ∧ nat.modeq 3 n 1) ∨ 
  (∃ p : ℕ, nat.prime p ∧ n = p^2)) :=
by
  sorry

end divisors_condition_l527_527343


namespace modified_height_ratio_l527_527522

noncomputable def cone_radius (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

noncomputable def cone_height_ratio (C : ℝ) (V : ℝ) (h_original : ℝ) : ℝ :=
  let r := cone_radius C
  let h_new := (3 * V) / (Real.pi * r^2)
  h_new / h_original

-- Given conditions
def C : ℝ := 12 * Real.pi
def V : ℝ := 72 * Real.pi
def h_original : ℝ := 20

-- Prove statement
theorem modified_height_ratio : cone_height_ratio C V h_original = 3 / 10 :=
  sorry

end modified_height_ratio_l527_527522


namespace calc_x_squared_plus_5xy_plus_y_squared_l527_527600

theorem calc_x_squared_plus_5xy_plus_y_squared 
  (x y : ℝ) 
  (h1 : x * y = 4)
  (h2 : x - y = 5) :
  x^2 + 5 * x * y + y^2 = 53 :=
by 
  sorry

end calc_x_squared_plus_5xy_plus_y_squared_l527_527600


namespace village_tree_planting_l527_527324

theorem village_tree_planting :
  ∃ (m n : ℕ), 2 * m + 3 * n = 440 ∧ 3 * m + n = 380 ∧ -- unit prices
    (∀ x : ℕ, 112.5 ≤ x ∧ x ≤ 150 ∧ 20 * x + 12000 = (100 * x + 80 * (150 - x)) ∧ -- functional expression and range
      (∀ (x_min : ℕ) (hx_min : 112.5 ≤ x_min ∧ x_min ≤ 150), -- minimizing the cost
        x_min = 113 → 20 * x_min + 12000 = 14260 ∧ 
        150 - x_min = 37)) :=
begin
  -- Initial unit prices equations
  use 100,
  use 80,
  split,
  { 
    -- Proving 2 * 100 + 3 * 80 = 440
    calc
      2 * 100 + 3 * 80 = 200 + 240 : by ring
                     ... = 440     : by norm_num,
  },
  split,
  {
    -- Proving 3 * 100 + 1 * 80 = 380
    calc
      3 * 100 + 80 = 300 + 80 : by ring
                   ... = 380  : by norm_num,
  },
  -- Proving functional expression and range
  intros x,
  split,
  {
    -- Proving range 112.5 ≤ x ∧ x ≤ 150
    sorry,
  },
  split,
  {
    -- Proving functional expression 20 * x + 12000 = (100 * x + 80 * (150 - x))
    sorry,
  },
  intros x_min hx_min,
  split,
  {
    -- Proving minimizing cost x_min = 113
    sorry,
  },
  {
    -- Proving 150 - 113 = 37
    sorry,
  },
end

end village_tree_planting_l527_527324


namespace number_of_connections_l527_527593

theorem number_of_connections (n : ℕ) (d : ℕ) (h₀ : n = 40) (h₁ : d = 4) : 
  (n * d) / 2 = 80 :=
by
  sorry

end number_of_connections_l527_527593


namespace ratio_of_radii_l527_527672

theorem ratio_of_radii 
  (a b : ℝ)
  (h1 : ∀ (a b : ℝ), π * b^2 - π * a^2 = 4 * π * a^2) : 
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l527_527672


namespace maximal_total_distance_l527_527453

-- Define the grid size
variable (n : ℕ)

-- Distance function for a couple based on their seats
def couple_distance (x₁ y₁ x₂ y₂ : ℕ) : ℕ := (abs (x₁ - x₂)) + (abs (y₁ - y₂))

-- Total distance function
def total_distance (pairs : list (ℕ × ℕ × ℕ × ℕ)) : ℕ :=
  pairs.foldl (λ acc (x), acc + couple_distance (x.1) (x.2) (x.3) (x.4)) 0

-- Main theorem statement
theorem maximal_total_distance (n : ℕ) : ∃ pairs : list (ℕ × ℕ × ℕ × ℕ), total_distance pairs = 4 * n^3 := sorry

end maximal_total_distance_l527_527453


namespace statement1_l527_527878

def points_on_same_side_of_line (P Q : Point) (l : Line) : Prop :=
  -- definition of being on the same side can be implied by their relative distance from the line
  sorry

def are_two_distinct_circles_tangent (P Q : Point) (l : Line) : Prop :=
  ∃ c₁ c₂ : Circle, c₁ ≠ c₂ ∧ P ∈ c₁ ∧ Q ∈ c₁ ∧ P ∈ c₂ ∧ Q ∈ c₂ ∧ is_tangent l c₁ ∧ is_tangent l c₂

theorem statement1 (P Q : Point) (l : Line) (h : points_on_same_side_of_line P Q l) :
  ¬ are_two_distinct_circles_tangent P Q l :=
sorry

end statement1_l527_527878


namespace math_problem_l527_527654

theorem math_problem (x y : ℝ) (hxyrat : x + y ∈ ℚ) (hcond : |x + 1| + (2 * x - y + 4)^2 = 0) : x^5 * y + x * y^5 = -34 :=
by
  sorry

end math_problem_l527_527654


namespace concurrency_of_lines_l527_527010

theorem concurrency_of_lines
  (ABC : Type) [triangle ABC]
  (circumcircle : Circle)
  (H : Point)
  (M : Point)
  (X : Point)
  (Y : Point)
  (P : Point)
  (h_inscribed : ABC ∈ circumcircle)
  (h_orthocenter : orthocenter ABC = H)
  (h_midpoint_M : midpoint B C = M)
  (h_parallel_AX_BC : parallel (line_through A X) (line_through B C))
  (h_parallel_BY_AC : parallel (line_through B Y) (line_through A C))
  (h_intersect_MH_P_circumcircle : intersect (line_through M H) circumcircle = P) :
  concurrent (line_through A M) (line_through C X) (line_through P Y) :=
sorry

end concurrency_of_lines_l527_527010


namespace sin_beta_l527_527938

variable (α β : ℝ)
variable (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
variable (hβ1 : 0 < β) (hβ2: β < Real.pi / 2)
variable (h1 : Real.cos α = 5 / 13)
variable (h2 : Real.sin (α - β) = 4 / 5)

theorem sin_beta (α β : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
  (hβ1 : 0 < β) (hβ2 : β < Real.pi / 2) 
  (h1 : Real.cos α = 5 / 13) 
  (h2 : Real.sin (α - β) = 4 / 5) : 
  Real.sin β = 16 / 65 := 
by 
  sorry

end sin_beta_l527_527938


namespace log_product_value_l527_527210

variable (x y : ℝ)

theorem log_product_value :
  (log x 2 / log y 12) * (log y 3 / log x 10) * (log x 4 / log y 8) * (log y 8 / log x 6) * (log x 10 / log y 3) = (5 / 18) * (log y x) :=
by
  sorry

end log_product_value_l527_527210


namespace general_formula_for_a_sum_T20_l527_527244

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 14 * n

-- Define the sequence a_n by the relation S_n - S_(n-1)
def a (n : ℕ) : ℤ := 2 * n - 15

-- Define |a_n * a_(n+1)|
def b (n : ℕ) : ℚ := (1 / (Int.natAbs (a n * a (n + 1))))

-- Prove the general formula for a_n
theorem general_formula_for_a (n : ℕ) : a n = 2 * n - 15 := by
  sorry

-- Prove the sum of the first 20 terms of b_n equals 682 / 351
theorem sum_T20 : 
  (Finset.range 20).sum (λ n, b n) = 682 / 351 := by
  sorry

end general_formula_for_a_sum_T20_l527_527244


namespace smallest_positive_integer_linear_combination_l527_527801

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l527_527801


namespace ratio_AF_BF_eq_3_l527_527284

noncomputable def parabola_focus_x_squared_eq_4y : (x y : ℝ) -> Prop := x^2 = 4 * y
def line_x_minus_sqrt3_y_plus_sqrt3_eq_0 : (x y : ℝ) -> Prop := x - sqrt 3 * y + sqrt 3 = 0

theorem ratio_AF_BF_eq_3
  (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : parabola_focus_x_squared_eq_4y x₁ y₁)
  (h2 : parabola_focus_x_squared_eq_4y x₂ y₂)
  (h3 : line_x_minus_sqrt3_y_plus_sqrt3_eq_0 x₁ y₁)
  (h4 : line_x_minus_sqrt3_y_plus_sqrt3_eq_0 x₂ y₂)
  (h5 : |(0,1) - (x₁,y₁)| > |(0,1) - (x₂,y₂)|) :
  (|(0,1) - (x₁,y₁)| / |(0,1) - (x₂,y₂)|) = 3 := sorry

end ratio_AF_BF_eq_3_l527_527284


namespace minimum_of_squared_loss_l527_527359

variable {α : Type} [ProbabilitySpace α]
noncomputable def L (x y : ℝ) := (x - y) ^ 2

theorem minimum_of_squared_loss (ξ : α → ℝ) (a_star : ℝ) : 
  (∀ a, Expectation (L ξ a) ≥ Expectation (L ξ a_star)) →
  a_star = Expectation ξ :=
by
  sorry

end minimum_of_squared_loss_l527_527359


namespace trigonometric_inequality_proof_l527_527810

theorem trigonometric_inequality_proof : 
  ∀ (sin cos : ℝ → ℝ), 
  (∀ θ, 0 ≤ θ ∧ θ ≤ π/2 → sin θ = cos (π/2 - θ)) → 
  sin (π * 11 / 180) < sin (π * 12 / 180) ∧ sin (π * 12 / 180) < sin (π * 80 / 180) :=
by 
  intros sin cos identity
  sorry

end trigonometric_inequality_proof_l527_527810


namespace positive_integer_fraction_count_l527_527927

theorem positive_integer_fraction_count :
  {n : ℕ | 0 < n ∧ n < 36 ∧ (36 - n) ∣ n}.count = 8 :=
by 
  apply finset.card_of_finset_pred
  sorry

end positive_integer_fraction_count_l527_527927


namespace fraction_spent_on_movie_and_soda_l527_527909

variables (P t d : ℝ)

theorem fraction_spent_on_movie_and_soda {P : ℝ} (h1 : t = 0.25 * (P - d)) (h2 : d = 0.10 * (P - t)) :
  ((t + d) / P) * 100 ≈ 31 :=
by
  sorry

end fraction_spent_on_movie_and_soda_l527_527909


namespace find_value_of_x_l527_527755

noncomputable def problem_statement : Prop := 
  ∃ (x y : ℝ), (x^2 + xy + y^2 = 2) ∧ (x^2 - y^2 = sqrt 5) ∧ 
               let a1 := 1 + 2/3 in
               let a2 := 2/3 in
               (100 * 5 + 3 = 503)

theorem find_value_of_x : problem_statement := sorry

end find_value_of_x_l527_527755


namespace maximum_additional_charge_expected_difference_l527_527144

-- Definitions for the conditions
def previous_readings : list ℕ := [1214, 1270, 1298, 1337, 1347, 1402]
def tariffs : list ℝ := [4.03, 1.01, 3.39]
def customer_payment : ℝ := 660.72

-- Maximum possible additional payment proof statement
theorem maximum_additional_charge : 
    let sorted_readings := previous_readings.sorted in
    let max_charge := (sorted_readings.get_or_else 5 0 - sorted_readings.get_or_else 0 0) * tariffs.get_or_else 0 0 +
                      (sorted_readings.get_or_else 4 0 - sorted_readings.get_or_else 1 0) * tariffs.get_or_else 2 0 +
                      (sorted_readings.get_or_else 3 0 - sorted_readings.get_or_else 2 0) * tariffs.get_or_else 1 0 in
    (max_charge - customer_payment) = 397.34 :=
sorry

-- Expected value of the difference proof statement
theorem expected_difference : 
    let sorted_readings := previous_readings.sorted in
    let expr :=
        (5 * sorted_readings.get_or_else 5 0 + 3 * sorted_readings.get_or_else 4 0 + 1 * sorted_readings.get_or_else 3 0 - 
        sorted_readings.get_or_else 2 0 - 3 * sorted_readings.get_or_else 1 0 - 5 * sorted_readings.get_or_else 0 0) 
        / 15 in
    let exp_val := 8.43 * expr in
    (exp_val - customer_payment) = 19.30 :=
sorry

end maximum_additional_charge_expected_difference_l527_527144


namespace count_good_numbers_l527_527851

def is_good (n : ℕ) : Prop :=
  ∃ r a : ℕ, r ≥ 2 ∧ n = r * a + r * (r - 1) / 2

theorem count_good_numbers : (set.count (is_good) {n | n ≤ 100}) = 93 := by
  sorry

end count_good_numbers_l527_527851


namespace problem_statement_l527_527961

variable (f : ℝ → ℝ)

theorem problem_statement (h : ∀ x : ℝ, 2 * (f x) + x * (deriv f x) > x^2) :
  ∀ x : ℝ, x^2 * f x ≥ 0 :=
by
  sorry

end problem_statement_l527_527961


namespace children_on_ferris_wheel_l527_527146

theorem children_on_ferris_wheel (x : ℕ) (h : 5 * x + 3 * 5 + 8 * 2 * 5 = 110) : x = 3 :=
sorry

end children_on_ferris_wheel_l527_527146


namespace value_range_of_function_l527_527085

theorem value_range_of_function :
  ∀ (y x : ℝ), 
  (y = (2 - x) * (x + 1) / ((x + 1) * (x + 3)) ∧ 
  x ≠ -1 ∧ 
  x ≠ -3) → 
  (y ∈ set.Ioo (-∞) (-1) ∪ set.Ioo (-1) (3 / 2) ∪ set.Ioo (3 / 2) ∞) :=
by
  sorry

end value_range_of_function_l527_527085


namespace convex_shape_on_grid_l527_527537

-- Define the grid and convex shape
def grid : Type := fin 5 × fin 6
def convex_shape : Type := fin 4

-- Conditions on pasting
def pastable (shape : convex_shape → grid) : Prop :=
  ∀ s, let ⟨r, c⟩ := shape s in 0 ≤ r ∧ r < 5 ∧ 0 ≤ c ∧ c < 6

-- Number of distinct shapes
def distinct_shapes : nat := 31

-- Prove the main theorem
theorem convex_shape_on_grid : (∃ (shape : convex_shape → grid), pastable shape) → distinct_shapes = 31 :=
by sorry

end convex_shape_on_grid_l527_527537


namespace fewest_tiles_needed_l527_527159

theorem fewest_tiles_needed :
  let tile_area := 2 * 6
  let large_rectangle_area := 36 * 48
  let small_rectangle_area := 24 * 12
  let total_area := large_rectangle_area + small_rectangle_area
  let total_tiles := total_area / tile_area
  total_tiles = 168 :=
by
  let tile_area := 2 * 6
  let large_rectangle_area := 36 * 48
  let small_rectangle_area := 24 * 12
  let total_area := large_rectangle_area + small_rectangle_area
  let total_tiles := total_area / tile_area
  have h1: total_tiles = (large_rectangle_area + small_rectangle_area) / tile_area := by rfl
  have h2: tile_area = 12 := by decide -- 2*6 = 12;
  have h3: large_rectangle_area = 1728 := by decide -- 36*48 = 1728;
  have h4: small_rectangle_area = 288 := by decide -- 24*12 = 288;
  have h5: total_area = 2016 := by decide -- 1728+288 = 2016;
  have h6: total_tiles = 2016 / 12 := by rw [←h5, h2];
  exact eq.trans h6 (eq.symm (calc 2016 / 12 = 168 : by decide))


end fewest_tiles_needed_l527_527159


namespace solve_inequality_l527_527047

theorem solve_inequality :
  {x : ℝ | (x^2 - 9) / (x - 3) > 0} = { x : ℝ | (-3 < x ∧ x < 3) ∨ (x > 3)} :=
by {
  sorry
}

end solve_inequality_l527_527047


namespace lake_with_more_frogs_has_45_frogs_l527_527706

-- Definitions for the problem.
variable (F : ℝ) -- Number of frogs in the lake with more frogs.
variable (F_less : ℝ) -- Number of frogs in Lake Crystal (the lake with fewer frogs).

-- Conditions
axiom fewer_frogs_condition : F_less = 0.8 * F
axiom total_frogs_condition : F + F_less = 81

-- Theorem statement: Proving that the number of frogs in the lake with more frogs is 45.
theorem lake_with_more_frogs_has_45_frogs :
  F = 45 :=
by
  sorry

end lake_with_more_frogs_has_45_frogs_l527_527706


namespace middle_box_label_l527_527389

theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (prize : Fin 23),
  (∀ i, boxes i = ("There is no prize here" ∨ "The prize is in the neighboring box")) ∧
  (∃ i, boxes i = "The prize is in the neighboring box" ∧ prize = i) ∧
  (∀ j, j ≠ i → boxes j = "There is no prize here") →
  boxes 11 = "The prize is in the neighboring box" :=
by
  sorry

end middle_box_label_l527_527389


namespace unique_missing_midpoint_l527_527400

theorem unique_missing_midpoint {n : ℕ} (h : 1 ≤ n) 
  (A : Fin (2 * n) → Point)
  (F : Fin (2 * n - 1) → Point)
  (hF : ∀ i : Fin (2 * n - 1), F i = midpoint (A i) (A (i + 1))) :
  ∃! F₂ₙ : Point, ∃ index : Fin (2 * n), F₂ₙ = midpoint (A (index)) (A 0) :=
sorry

end unique_missing_midpoint_l527_527400


namespace perimeter_sum_of_rectangles_l527_527524

theorem perimeter_sum_of_rectangles (l w : ℝ) 
    (h1 : l = 1) 
    (h2 : w = 1) 
    (rectangles_cut : ℝ × ℝ → ℝ)
    : 
  (let S := rectangles_cut (l, w) in 6 < S ∧ S ≤ 10) :=
by
  sorry

end perimeter_sum_of_rectangles_l527_527524


namespace calculate_g_of_f_l527_527364

def f (x : ℝ) : ℝ := x^3 + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calculate_g_of_f : g(f(-3)) = 1105 := by
  sorry

end calculate_g_of_f_l527_527364


namespace red_segments_overlap_l527_527776

theorem red_segments_overlap (total_length : ℝ) (edge_dist : ℝ) (overlaps : ℕ) (overlap_length : ℝ) 
  (h1 : total_length = 98) (h2 : edge_dist = 83) (h3 : overlaps = 6) : 
  overlap_length = (15 / 6) :=
by
  have discrepancy := total_length - edge_dist
  have overlap_length := discrepancy / overlaps
  exact (by sorry : overlap_length = 2.5)

end red_segments_overlap_l527_527776


namespace second_number_value_l527_527082

theorem second_number_value (a b c : ℝ) 
  (h1 : a + b + c = 120) 
  (h2 : a = (2 / 3) * b) 
  (h3 : b = (3 / 4) * c) 
  : b = 40 :=
begin
  sorry
end

end second_number_value_l527_527082


namespace apples_and_pears_weight_l527_527091

theorem apples_and_pears_weight (apples pears : ℕ) 
    (h_apples : apples = 240) 
    (h_pears : pears = 3 * apples) : 
    apples + pears = 960 := 
  by
  sorry

end apples_and_pears_weight_l527_527091


namespace weight_difference_l527_527131

theorem weight_difference
  (joe_weight : ℕ := 44)
  (original_avg_weight : ℕ := 30)
  (new_avg_weight : ℕ := 31)
  (final_avg_weight : ℕ := 30)
  (n : ℕ)
  (original_total_weight : ℕ := original_avg_weight * n)
  (new_total_weight : ℕ := original_total_weight + joe_weight)
  (number_of_students_after_joe_joins : ℕ := n + 1)
  (weight_two_students_left : ℕ := new_total_weight - final_avg_weight * number_of_students_after_joe_joins)
  (average_weight_two_students_left : ℕ := weight_two_students_left / 2)
  : average_weight_two_students_left - joe_weight = -7 :=
begin
  sorry
end

end weight_difference_l527_527131


namespace roses_picked_second_time_l527_527844

-- Define the initial conditions
def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def total_roses_after_second_picking : ℝ := 72.0

-- Define the calculation after the first picking
def roses_after_first_picking : ℝ := initial_roses + first_pick

-- The Lean statement to prove the number of roses picked the second time
theorem roses_picked_second_time : total_roses_after_second_picking - roses_after_first_picking = 19.0 := 
by
  -- Use the facts stated in the conditions
  sorry

end roses_picked_second_time_l527_527844


namespace terry_nora_age_relation_l527_527312

variable {N : ℕ} -- Nora's current age

theorem terry_nora_age_relation (h₁ : Terry_current_age = 30) (h₂ : Terry_future_age = 4 * N) : N = 10 :=
by
  --- additional assumptions
  have Terry_future_age_def : Terry_future_age = 30 + 10 := by sorry
  rw [Terry_future_age_def] at h₂
  linarith

end terry_nora_age_relation_l527_527312


namespace pump_B_time_l527_527479

theorem pump_B_time (T_B : ℝ) (h1 : ∀ (h1 : T_B > 0),
  (1 / 4 + 1 / T_B = 3 / 4)) :
  T_B = 2 := 
by
  sorry

end pump_B_time_l527_527479


namespace total_paint_is_correct_l527_527873

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l527_527873


namespace function_properties_and_k_range_l527_527629

theorem function_properties_and_k_range :
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 ^ x) ∧ (∀ y, y > 0)) ∧
  (∀ k : ℝ, (∃ t : ℝ, t > 0 ∧ (t^2 - 2*t + k = 0)) ↔ (0 < k ∧ k < 1)) :=
by sorry

end function_properties_and_k_range_l527_527629


namespace number_of_sides_sum_of_interior_angles_l527_527853

-- Condition: each exterior angle of the regular polygon is 18 degrees.
def exterior_angle (n : ℕ) : Prop :=
  360 / n = 18

-- Question 1: Determine the number of sides the polygon has.
theorem number_of_sides : ∃ n, n > 2 ∧ exterior_angle n :=
  sorry

-- Question 2: Calculate the sum of the interior angles.
theorem sum_of_interior_angles {n : ℕ} (h : 360 / n = 18) : 
  180 * (n - 2) = 3240 :=
  sorry

end number_of_sides_sum_of_interior_angles_l527_527853


namespace statement_D_incorrect_l527_527812

open Probability

-- Definitions of constants and conditions
def E_binomial (n p : ℝ) : ℝ := n * p
def D_binomial (n p : ℝ) : ℝ := n * p * (1 - p)

-- Statement of the problem
theorem statement_D_incorrect :
  ∀ (σ : ℝ), (P (normal 1 σ) (Iio 0) = 0.2) →
             (P (normal 1 σ) (Ioo 1 2) ≠ 0.2) :=
by
  intros σ h
  sorry

end statement_D_incorrect_l527_527812


namespace sum_of_first_5n_l527_527663

theorem sum_of_first_5n (n : ℕ) (h : (3 * n) * (3 * n + 1) / 2 = n * (n + 1) / 2 + 270) : (5 * n) * (5 * n + 1) / 2 = 820 :=
by
  sorry

end sum_of_first_5n_l527_527663


namespace original_cost_75_l527_527829

-- Definitions of initial given conditions
def sellingPrice := ℝ  -- P is the selling price

variable (P : sellingPrice)

-- Initial profit was 25% of the selling price
def initialProfit := 0.25 * P

-- Original manufacturing cost was (100% - 25%) of the selling price
def originalManufacturingCost := 0.75 * P

-- Current profit is 50% of the selling price
def currentProfit := 0.50 * P

-- New manufacturing cost is given as $50
def newManufacturingCost := 50

-- Given that the new manufacturing cost is $50,
-- Prove that the original manufacturing cost was $75.
theorem original_cost_75 (h : 0.50 * P = newManufacturingCost) : originalManufacturingCost P = 75 :=
by
  sorry

end original_cost_75_l527_527829


namespace problem_solution_l527_527355

theorem problem_solution (x : Fin 100 → ℝ) 
  (h1 : ∑ i, x i = 1) 
  (h2 : ∑ i, x i / (1 - x i) = 1) :
  ∑ i, x i^2 / (1 - x i) = 0 :=
by
  sorry

end problem_solution_l527_527355


namespace max_distance_l527_527965

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x^2 / 9) + y^2 = 1

-- Define the Cartesian equation of line l
def line_l (x y : ℝ) : Prop := (x - y + 2) = 0

-- Prove the maximum distance from any point P on curve C to line l
theorem max_distance (P : ℝ × ℝ) (hP : curve_C P.1 P.2) : 
  ∃ d, d = sqrt 5 + sqrt 2 ∧ ∀ x y, curve_C x y → abs (x - y + 2) / sqrt 2 ≤ d :=
sorry

end max_distance_l527_527965


namespace height_inradius_ratio_is_7_l527_527960

-- Definitions of geometric entities and given conditions.
variable (h r : ℝ)
variable (cos_theta : ℝ)
variable (cos_theta_eq : cos_theta = 1 / 6)

-- Theorem statement: Ratio of height to inradius is 7 given the cosine condition.
theorem height_inradius_ratio_is_7
  (h r : ℝ)
  (cos_theta : ℝ)
  (cos_theta_eq : cos_theta = 1 / 6)
  (prism_def : true) -- Added to mark the geometric nature properly
: h / r = 7 :=
sorry  -- Placeholder for the actual proof.

end height_inradius_ratio_is_7_l527_527960


namespace seashells_total_l527_527373

theorem seashells_total (mary_seashells : ℕ) (jessica_seashells : ℕ) (h_mary : mary_seashells = 18) (h_jessica : jessica_seashells = 41) :
  mary_seashells + jessica_seashells = 59 :=
by
  rw [h_mary, h_jessica]
  rfl

end seashells_total_l527_527373


namespace smallest_positive_integer_l527_527806

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l527_527806


namespace squares_centers_equal_perpendicular_l527_527425

def Square (center : (ℝ × ℝ)) (side : ℝ) := {p : ℝ × ℝ // abs (p.1 - center.1) ≤ side / 2 ∧ abs (p.2 - center.2) ≤ side / 2}

theorem squares_centers_equal_perpendicular 
  (a b : ℝ)
  (O A B C : ℝ × ℝ)
  (hA : A = (a, a))
  (hB : B = (b, 2 * a + b))
  (hC : C = (- (a + b), a + b))
  (hO_vertex : O = (0, 0)) :
  dist O B = dist A C ∧ ∃ m₁ m₂ : ℝ, (B.2 - O.2) / (B.1 - O.1) = m₁ ∧ (C.2 - A.2) / (C.1 - A.1) = m₂ ∧ m₁ * m₂ = -1 := sorry

end squares_centers_equal_perpendicular_l527_527425


namespace clock_ticks_6oclock_time_l527_527538

theorem clock_ticks_6oclock_time :
  ∀ (time_per_interval : ℕ), -- Introduce a variable for the interval time
  (8 - 1) * time_per_interval = 42 → -- Condition at 8 o'clock
  (6 - 1) * time_per_interval = 30 -- Condition at 6 o'clock
:=
begin
  intros time_per_interval h1,
  calc
    (6 - 1) * time_per_interval = 30 : sorry
end

end clock_ticks_6oclock_time_l527_527538


namespace highest_number_on_drawn_balls_is_3_4_5_l527_527138

/-- A bag contains 5 balls numbered 1, 2, 3, 4, 5.
    3 balls are drawn at random from the bag.
    The random variable ξ represents the highest number on the balls drawn.
    Prove that the possible values of ξ are {3, 4, 5}. -/
theorem highest_number_on_drawn_balls_is_3_4_5 :
  ∀ (drawn: finset ℕ), drawn ⊆ {1, 2, 3, 4, 5} ∧ drawn.card = 3 → 
  ∃ (ξ : finset ℕ), ξ = {3, 4, 5} :=
by
  sorry

end highest_number_on_drawn_balls_is_3_4_5_l527_527138


namespace seth_initial_boxes_l527_527039

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end seth_initial_boxes_l527_527039


namespace part1_part2_l527_527285

open Set

-- Define the sets M and N based on given conditions
def M (a : ℝ) : Set ℝ := { x | (x + a) * (x - 1) ≤ 0 }
def N : Set ℝ := { x | 4 * x^2 - 4 * x - 3 < 0 }

-- Part (1): Prove that if M ∪ N = { x | -2 ≤ x < 3 / 2 }, then a = 2
theorem part1 (a : ℝ) (h : a > 0)
  (h_union : M a ∪ N = { x | -2 ≤ x ∧ x < 3 / 2 }) : a = 2 := by
  sorry

-- Part (2): Prove that if N ∪ (compl (M a)) = univ, then 0 < a ≤ 1/2
theorem part2 (a : ℝ) (h : a > 0)
  (h_union : N ∪ compl (M a) = univ) : 0 < a ∧ a ≤ 1 / 2 := by
  sorry

end part1_part2_l527_527285


namespace line_PP_l527_527334

theorem line_PP'_passes_through_fixed_point
  (A B C : Point) -- vertices of triangle ABC
  (A' B' C' : Point) -- midpoints of sides BC, CA, AB respectively
  (P P' : Point) -- points P and P'
  (h_A' : 2 • A' = B + C)
  (h_B' : 2 • B' = A + C)
  (h_C' : 2 • C' = A + B)
  (h_PA_eq_P'A' : dist P A = dist P' A')
  (h_PB_eq_P'B' : dist P B = dist P' B')
  (h_PC_eq_P'C' : dist P C = dist P' C') :
  ∃ G : Point, ∀ P P', line_through P P' G := 
sorry

end line_PP_l527_527334


namespace trigonometric_identity_l527_527936

noncomputable def trig_expression (θ : Real) : Real :=
  sin θ ^ 2 + sin θ * cos θ - 2 * cos θ ^ 2

theorem trigonometric_identity (θ : Real) (h : tan (θ - π) = 2) : trig_expression θ = 4 / 5 :=
by
  sorry -- Proof is not required

end trigonometric_identity_l527_527936


namespace meeting_times_comparison_l527_527133

theorem meeting_times_comparison :
  let t2 := (11 - 2 * Real.pi) / 22,
      t3 := (4 * Real.pi - 7) / 27 in
  t3 < t2 :=
by
  sorry

end meeting_times_comparison_l527_527133


namespace multiplication_of_935421_and_625_l527_527834

theorem multiplication_of_935421_and_625 :
  935421 * 625 = 584638125 :=
by sorry

end multiplication_of_935421_and_625_l527_527834


namespace problem_l527_527955

theorem problem (a : ℝ) (h : a^2 - 5 * a - 1 = 0) : 3 * a^2 - 15 * a = 3 :=
by
  sorry

end problem_l527_527955


namespace women_in_business_class_l527_527041

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (percent_women : ℝ) 
  (percent_women_in_business : ℝ) 
  (H1 : total_passengers = 300)
  (H2 : percent_women = 0.70)
  (H3 : percent_women_in_business = 0.08) : 
  ∃ (num_women_business_class : ℕ), num_women_business_class = 16 := 
by
  sorry

end women_in_business_class_l527_527041


namespace total_paint_is_correct_l527_527872

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l527_527872


namespace pythagorean_triple_set_C_l527_527869

-- Declaration for real numbers for use in theorem
variables (a b c : ℤ)

-- Define the sets given in the conditions
def SetA := (7, 8, 9)
def SetB := (5, 6, 7)
def SetC := (5, 12, 13)
def SetD := (21, 25, 28)

-- The statement to prove:
theorem pythagorean_triple_set_C :
  let (a, b, c) := SetC in 
  a^2 + b^2 = c^2 := 
by {
  -- Asserting the specific values under SetC
  have h_setC : SetC = (5, 12, 13) := by rfl,
  -- Decomposing the tuple and specifically identifying a, b, c
  have h_values : a = 5 ∧ b = 12 ∧ c = 13 := by {
    simp [SetC] at *,
    exact (by rfl : (a, b, c) = (a, b, c)),
  },
  -- Finishing the proof with sorry
  sorry
}

end pythagorean_triple_set_C_l527_527869


namespace solve_x_from_equation_l527_527999

theorem solve_x_from_equation (x : ℝ) (h : real.cbrt (1 + real.sqrt x) = 2) : x = 49 :=
by
  sorry

end solve_x_from_equation_l527_527999


namespace circle_value_l527_527576

theorem circle_value (c d s : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y + d)^2 = s^2)
  (h2 : c = 4)
  (h3 : d = -8)
  (h4 : s = 2 * Real.sqrt 5) :
  c + d + s = -4 + 2 * Real.sqrt 5 := 
sorry

end circle_value_l527_527576


namespace broken_line_condition_l527_527461

variables {a b c d e f : ℝ}

-- The vectors a, b, c, d, e, f combine to form a closed broken line.
def closed_polygon (a b c d e f : ℝ) : Prop :=
  (∃ (a_vec b_vec c_vec d_vec e_vec f_vec : ℝ×ℝ), 
    -- Vectors corresponding to sides
    a_vec = ⟨a, 0⟩ ∧ d_vec = ⟨0, d⟩ ∧ 
    b_vec = ⟨b, 0⟩ ∧ e_vec = ⟨0, e⟩ ∧ 
    c_vec = ⟨c, 0⟩ ∧ f_vec = ⟨0, f⟩ ∧ 
    -- The line is closed (vectors sum to zero)
    a_vec + b_vec + c_vec + d_vec + e_vec + f_vec = 0 ∧ 
    -- Pairs of opposite sides are perpendicular
    (a_vec ⬝ d_vec = 0) ∧ (b_vec ⬝ e_vec = 0) ∧ (c_vec ⬝ f_vec = 0))

-- The condition for the existence of such a closed polygonal line
def existence_condition (a b c d e f : ℝ) : Prop :=
  let s1 := real.sqrt (a^2 + d^2) in
  let s2 := real.sqrt (b^2 + e^2) in
  let s3 := real.sqrt (c^2 + f^2) in
  s1 + s2 >= s3 ∧ s2 + s3 >= s1 ∧ s3 + s1 >= s2

-- The main theorem
theorem broken_line_condition (a b c d e f : ℝ) :
  closed_polygon a b c d e f → existence_condition a b c d e f :=
sorry

end broken_line_condition_l527_527461


namespace cafe_purchase_l527_527665

theorem cafe_purchase (s d : ℕ) (h_d : d ≥ 2) (h_cost : 5 * s + 125 * d = 4000) :  s + d = 11 :=
    -- Proof steps go here
    sorry

end cafe_purchase_l527_527665


namespace sum_of_adjacent_to_7_l527_527077

def pos_int_divisors (n : ℕ) : list ℕ :=
  list.filter (λ d, d ∣ n) (list.range (n + 1))

def is_valid_adjacent_pair (a b : ℕ) : Prop :=
  nat.gcd a b > 1

def adjacent_sum (lst : list ℕ) (x : ℕ) : ℕ :=
  match list.index_of x lst with
  | list.index_of x lst :=  match lst.drop (list.index_of x lst).succ ++ lst.take (list.index_of x lst).succ with
                            | [] => 0
                            | (y :: z :: rest) => if z = x then y + list.nth_le lst (list.index_of x lst - 1) (by sorry) else y + z
                            | _ => 0
                            end
  | _ => 0
  end

theorem sum_of_adjacent_to_7 : 
  let divisors := pos_int_divisors 147 in
  ∃ (arrangement : list ℕ), 
  arrangement ~ divisors ∧  -- permutation of divisors
  ∀ i, i < arrangement.length → is_valid_adjacent_pair (arrangement.nth_le i sorry) (arrangement.nth_le ((i+1) % arrangement.length) sorry) →
  adjacent_sum arrangement 7 = 196 :=
sorry

end sum_of_adjacent_to_7_l527_527077


namespace coefficient_of_x4_in_expansion_l527_527685

theorem coefficient_of_x4_in_expansion :
  let T_r := λ (r : ℕ), (nat.choose 6 r) * (-1/2)^r * x^(6-2*r) in
  (∀ x : ℝ, T_r 1 = -3) := 
begin
  sorry
end

end coefficient_of_x4_in_expansion_l527_527685


namespace jonathans_and_sisters_total_letters_l527_527700

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l527_527700


namespace Dave_earning_l527_527897

def action_games := 3
def adventure_games := 2
def role_playing_games := 3

def price_action := 6
def price_adventure := 5
def price_role_playing := 7

def earning_from_action_games := action_games * price_action
def earning_from_adventure_games := adventure_games * price_adventure
def earning_from_role_playing_games := role_playing_games * price_role_playing

def total_earning := earning_from_action_games + earning_from_adventure_games + earning_from_role_playing_games

theorem Dave_earning : total_earning = 49 := by
  show total_earning = 49
  sorry

end Dave_earning_l527_527897


namespace marked_price_is_59_l527_527842

-- Definitions of the conditions
def purchase_price (initial_price : ℝ) (discount_rate : ℝ) : ℝ :=
  initial_price - (initial_price * discount_rate)

def selling_price (cost : ℝ) (profit_rate : ℝ) : ℝ :=
  cost + (cost * profit_rate)

def marked_price (selling_price : ℝ) (discount_rate : ℝ) : ℝ :=
  selling_price / (1 - discount_rate)

-- The main statement of the theorem
theorem marked_price_is_59 :
  let initial_price := 50
  let purchase_discount := 0.15
  let profit_rate := 0.25
  let sale_discount := 0.10
  (marked_price (selling_price (purchase_price initial_price purchase_discount) profit_rate) sale_discount) = 59 :=
  by
  sorry

end marked_price_is_59_l527_527842


namespace min_value_of_M_l527_527362

def log (x : ℝ) : ℝ := Real.log x / Real.log 10

variable (x y z : ℝ)

def a := log z + log (x / (y * z) + 1)
def b := log (1 / x) + log (x * y * z + 1)
def c := log y + log (1 / (x * y * z) + 1)

def M := max (max a b) c

theorem min_value_of_M : M = log 2 :=
by
  sorry

end min_value_of_M_l527_527362


namespace new_rope_length_calculation_l527_527854

noncomputable def pi := Real.pi

-- Definitions
def r1 : ℝ := 10 -- initial length in meters
def A_additional : ℝ := 942.8571428571429 -- additional area in square meters

-- Calculations
def A1 : ℝ := pi * r1^2 -- initial area
def A2 : ℝ := A1 + A_additional -- new area in terms of new length
def L : ℝ := 20 -- expected new length of the rope in meters

-- Proof statement
theorem new_rope_length_calculation : 
  (pi * L^2 = A2) :=
sorry

end new_rope_length_calculation_l527_527854


namespace avg_rate_of_change_l527_527434

-- Define the function and the points
def f : ℝ → ℝ
noncomputable def A : ℝ × ℝ := (1, 3)
noncomputable def B : ℝ × ℝ := (3, 1)

-- State the problem as a theorem
theorem avg_rate_of_change : (f B.1 - f A.1) / (B.1 - A.1) = -1 :=
by sorry

end avg_rate_of_change_l527_527434


namespace A_finishes_alone_in_days_l527_527496

variable (days_A : ℕ) -- Days A takes to finish the work alone
variable (days_B : ℕ) := 10 -- Days B takes to finish the work alone
variable (work_together_days : ℕ) := 2 -- Days A and B work together
variable (work_b_alone_days : ℝ) := 3.999999999999999 -- Days B works alone

theorem A_finishes_alone_in_days : 
  ∀ (A : ℝ), 
  let work_done_together := work_together_days * (1 / A + 1 / days_B.toReal)
  let work_done_b_alone := work_b_alone_days * (1 / days_B.toReal)
  work_done_together + work_done_b_alone = 1 → 
  A = 5 :=
by
  sorry

end A_finishes_alone_in_days_l527_527496


namespace polynomial_roots_identity_l527_527027

theorem polynomial_roots_identity {p q α β γ δ : ℝ} 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end polynomial_roots_identity_l527_527027


namespace num_passenger_cars_l527_527515

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l527_527515


namespace total_number_of_letters_l527_527702

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l527_527702


namespace rons_siblings_product_l527_527992

theorem rons_siblings_product
  (H_sisters : ℕ)
  (H_brothers : ℕ)
  (Ha_sisters : ℕ)
  (Ha_brothers : ℕ)
  (R_sisters : ℕ)
  (R_brothers : ℕ)
  (Harry_cond : H_sisters = 4 ∧ H_brothers = 6)
  (Harriet_cond : Ha_sisters = 4 ∧ Ha_brothers = 6)
  (Ron_cond_sisters : R_sisters = Ha_sisters)
  (Ron_cond_brothers : R_brothers = Ha_brothers + 2)
  : R_sisters * R_brothers = 32 := by
  sorry

end rons_siblings_product_l527_527992


namespace total_oranges_l527_527387

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l527_527387


namespace distance_between_points_A_and_B_l527_527622

theorem distance_between_points_A_and_B :
  ∃ (d : ℝ), 
    -- Distance must be non-negative
    d ≥ 0 ∧
    -- Condition 1: Car 3 reaches point A at 10:00 AM (3 hours after 7:00 AM)
    (∃ V3 : ℝ, V3 = d / 6) ∧ 
    -- Condition 2: Car 2 reaches point A at 10:30 AM (3.5 hours after 7:00 AM)
    (∃ V2 : ℝ, V2 = 2 * d / 7) ∧ 
    -- Condition 3: When Car 1 and Car 3 meet, Car 2 has traveled exactly 3/8 of d
    (∃ V1 : ℝ, V1 = (d - 84) / 7 ∧ 2 * V1 + 2 * V3 = 8 * V2 / 3) ∧ 
    -- Required: The distance between A and B is 336 km
    d = 336 :=
by
  sorry

end distance_between_points_A_and_B_l527_527622


namespace max_ounces_among_items_l527_527866

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end max_ounces_among_items_l527_527866


namespace cos_B_correct_S_value_given_c_l527_527287

noncomputable def cos_B (a b c S : ℝ) (h1 : S = b * c * Math.cos a) (h2 : a + b = Real.pi / 4) : ℝ :=
- Math.cos a * Math.cos (Real.pi / 4) + Math.sin a * Math.sin (Real.pi / 4)

theorem cos_B_correct(a b c S : ℝ) (h1 : S = b * c * Math.cos a) (h2 : a + b = Real.pi / 4) :
  cos_B a b c S h1 h2 = Real.sqrt 10 / 10 := by
sorry

theorem S_value_given_c (a b S : ℝ) (c : ℝ) (h1 : c = Real.sqrt 5) (h2 : b = 3) (h3 : S = b * c * Math.cos a) :
  S = 3 := by
  sorry

end cos_B_correct_S_value_given_c_l527_527287


namespace probability_draw_first_three_red_is_2_div_17_l527_527163

/-- 
A standard deck of cards has 52 cards divided into 4 suits, each of which has 13 cards. 
Two of the suits (Hearts and Diamonds) are red, and the other two (Spades and Clubs) are black. 
The cards in the deck are shuffled. 
What is the probability that the first three cards drawn from the deck are all red? 
-/
noncomputable def probability_first_three_red (deck : Finset (Fin 52)) : ℚ :=
  let reds : Finset (Fin 52) := {x in deck | x < 26} in
  (reds.card.toRat / 52) * ((reds.card - 1).toRat / 51) * ((reds.card - 2).toRat / 50)

theorem probability_draw_first_three_red_is_2_div_17 (deck : Finset (Fin 52)) :
  probability_first_three_red deck = 2 / 17 :=
by
  /- Proof is omitted -/
  sorry

end probability_draw_first_three_red_is_2_div_17_l527_527163


namespace new_customers_needed_l527_527860

theorem new_customers_needed 
  (initial_customers : ℕ)
  (customers_after_some_left : ℕ)
  (first_group_left : ℕ)
  (second_group_left : ℕ)
  (new_customers : ℕ)
  (h1 : initial_customers = 13)
  (h2 : customers_after_some_left = 9)
  (h3 : first_group_left = initial_customers - customers_after_some_left)
  (h4 : second_group_left = 8)
  (h5 : new_customers = first_group_left + second_group_left) :
  new_customers = 12 :=
by
  sorry

end new_customers_needed_l527_527860


namespace area_of_smallest_square_around_hexagon_l527_527460

theorem area_of_smallest_square_around_hexagon (side_len : ℝ) (h : side_len = 2) : 
  ∃ (s : ℝ), (s = side_len * sqrt 2 ∧ s * s = 8) := 
by
  sorry

end area_of_smallest_square_around_hexagon_l527_527460


namespace bugs_eaten_ratio_l527_527330

theorem bugs_eaten_ratio :
  ∃ (L : ℚ), 
    12 + L + 3 * L + 4.5 * L = 63 ∧ (L / 12 = 1 / 2) :=
by {
  sorry
}

end bugs_eaten_ratio_l527_527330


namespace mirror_area_l527_527707

theorem mirror_area (width_frame height_frame frame_width : ℝ)
  (h_frame_dim : width_frame = 90 ∧ height_frame = 70)
  (h_frame_w : frame_width = 15) :
  let mirror_width := width_frame - 2 * frame_width
  let mirror_height := height_frame - 2 * frame_width
  let mirror_area := mirror_width * mirror_height
  in mirror_area = 2400 := by
  -- conditions from a)
  cases h_frame_dim with hw hh
  rw [hw, hh]  -- width_frame = 90, height_frame = 70
  rw [h_frame_w]  -- frame_width = 15
  -- define mirror dimensions
  let mirror_width := 90 - 2 * 15
  let mirror_height := 70 - 2 * 15
  -- substitute these into the mirror area calculation
  have : mirror_area = mirror_width * mirror_height := rfl
  -- calculate and verify the correct answer
  calc
    mirror_width * mirror_height
      = (90 - 2 * 15) * (70 - 2 * 15) : by rw [<- this]
  ... = 60 * 40 : by norm_num
  ... = 2400 : by norm_num

end mirror_area_l527_527707


namespace triangle_probability_correct_l527_527018

-- Define the list of stick lengths
def stick_lengths : List ℕ := [1, 4, 6, 8, 10, 12, 13, 15, 18]

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the problem as a Lean theorem statement
theorem triangle_probability_correct :
  let choices := stick_lengths.combinations 3
  let valid_tris := choices.filter (λ triple, match triple with
                             | [a, b, c] => can_form_triangle a b c
                             | _ => false
                             end)
  (valid_tris.length : ℚ) / (choices.length : ℚ) = 4 / 21 := by
  sorry

end triangle_probability_correct_l527_527018


namespace inequality_T_l527_527348

noncomputable def T (coeffs : List ℕ) (xs : List ℕ) : ℕ :=
  sorry -- The definition of T based on problem 61417.

theorem inequality_T 
  (alpha beta : List ℕ) 
  (x : List ℕ) 
  (n : ℕ) 
  (h_alpha_sum_eq_beta_sum : (alpha.sum = beta.sum)) 
  (h_alpha_neq_beta : alpha ≠ beta) 
  (h_non_neg_x : ∀ i < n, 0 ≤ xs.nthLe i sorry) :
  T alpha x ≥ T beta x := 
by 
  sorry

end inequality_T_l527_527348


namespace translate_m_sided_polygon_inside_n_sided_polygon_l527_527344

theorem translate_m_sided_polygon_inside_n_sided_polygon
  (m n : ℕ) (hm : m > 2) (hn : n > 2)
  (T : Type) [polygon n T] (H1 : contains_polygon T (polygon m 1)) :
  ∀ (S : Type) [polygon m S] (H2 : length_of S = cos (π / lcm m n)),
  ∃ (α : vector ℝ 2), ∀ (p : point S), translate α p ∈ T :=
sorry

end translate_m_sided_polygon_inside_n_sided_polygon_l527_527344


namespace gcf_64_80_l527_527109

theorem gcf_64_80 : ∃ g, g = nat.gcd 64 80 ∧ g = 16 :=
by
  have h1 : 64 = 2 ^ 6 := by norm_num
  have h2 : 80 = 2 ^ 4 * 5 := by norm_num
  use 16
  split
  · norm_num [nat.gcd]
  · norm_num

end gcf_64_80_l527_527109


namespace probability_at_least_one_each_color_l527_527454

-- Definitions based on the conditions
def bag : Type := {w : ℕ // w = 5} -- White balls
def bag : Type := {r : ℕ // r = 4} -- Red balls
def bag : Type := {y : ℕ // y = 3} -- Yellow balls

def draw_count : ℕ := 4 -- Number of balls drawn

-- Total number of ways to draw 4 balls from 12
def total_ways : ℕ := Nat.choose 12 4

-- Different combinations where each color is represented
def combination_1 : ℕ := (Nat.choose 5 2) * (Nat.choose 4 1) * (Nat.choose 3 1)
def combination_2 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.choose 3 1)
def combination_3 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 1) * (Nat.choose 3 2)

-- Total number of favorable outcomes
def favorable_ways : ℕ := combination_1 + combination_2 + combination_3

-- Calculated probability
def probability : ℚ := favorable_ways / total_ways.to_rat

theorem probability_at_least_one_each_color :
  probability = 6 / 11 := 
sorry

end probability_at_least_one_each_color_l527_527454


namespace total_oranges_l527_527384

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l527_527384


namespace average_of_original_set_l527_527423

theorem average_of_original_set
  (A : ℝ)
  (n : ℕ)
  (B : ℝ)
  (h1 : n = 7)
  (h2 : B = 5 * A)
  (h3 : B / n = 100)
  : A = 20 :=
by
  sorry

end average_of_original_set_l527_527423


namespace pizza_toppings_l527_527517

theorem pizza_toppings (toppings : Finset String) (h_toppings : toppings.card = 8)
  (mushrooms pineapple : String) (h_mushrooms : mushrooms ∈ toppings)
  (h_pineapple : pineapple ∈ toppings) (h_restriction : ∀ (x ∈ toppings), x ≠ mushrooms ∨ x ≠ pineapple) :
  let one_topping_pizzas := toppings.card,
      two_topping_pizzas := Nat.choose toppings.card 2 - 1
  in one_topping_pizzas + two_topping_pizzas = 35 := by
  sorry

end pizza_toppings_l527_527517


namespace bicycle_wheel_rotations_l527_527766

theorem bicycle_wheel_rotations
  (diameter : ℝ)
  (time_minutes : ℝ)
  (speed_kmh : ℝ)
  (rotations : ℕ)
  (h1 : diameter = 0.75)
  (h2 : time_minutes = 6)
  (h3 : speed_kmh = 24)
  (h4 : rotations = 1020)
  : let time_hours := time_minutes / 60,
        distance_meters := speed_kmh * 1000 * time_hours,
        circumference := π * diameter,
        exact_rotations := distance_meters / circumference
    in int.to_nat (exact_rotations.round) = rotations := by
sorry

end bicycle_wheel_rotations_l527_527766


namespace possible_values_of_quadratic_l527_527895

theorem possible_values_of_quadratic (x : ℝ) (hx : x^2 - 7 * x + 12 < 0) :
  1.75 ≤ x^2 - 7 * x + 14 ∧ x^2 - 7 * x + 14 ≤ 2 := by
  sorry

end possible_values_of_quadratic_l527_527895


namespace prove_U_eq_1_l527_527708

def term1 := 1 / (4 - Real.sqrt 9)
def term2 := 1 / (Real.sqrt 9 - Real.sqrt 8)
def term3 := 1 / (Real.sqrt 8 - Real.sqrt 7)
def term4 := 1 / (Real.sqrt 7 - Real.sqrt 6)
def term5 := 1 / (Real.sqrt 6 - 3)

def U := term1 + term2 - term3 + term4 - term5

theorem prove_U_eq_1 : U = 1 :=
by
  sorry

end prove_U_eq_1_l527_527708


namespace gina_can_paint_6_rose_cups_an_hour_l527_527933

def number_of_rose_cups_painted_in_an_hour 
  (R : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (total_payment : ℕ) (hourly_rate : ℕ)
  (lily_hours : ℕ) (total_hours : ℕ) (rose_hours : ℕ) : Prop :=
  (lily_rate = 7) ∧
  (rose_order = 6) ∧
  (lily_order = 14) ∧
  (total_payment = 90) ∧
  (hourly_rate = 30) ∧
  (lily_hours = lily_order / lily_rate) ∧
  (total_hours = total_payment / hourly_rate) ∧
  (rose_hours = total_hours - lily_hours) ∧
  (rose_order = R * rose_hours)

theorem gina_can_paint_6_rose_cups_an_hour :
  ∃ R, number_of_rose_cups_painted_in_an_hour 
    R 7 6 14 90 30 (14 / 7) (90 / 30)  (90 / 30 - 14 / 7) ∧ R = 6 :=
by
  -- proof is left out intentionally
  sorry

end gina_can_paint_6_rose_cups_an_hour_l527_527933


namespace pedestrian_speeds_unique_l527_527098

variables 
  (x y : ℝ)
  (d : ℝ := 105)  -- Distance between cities
  (t1 : ℝ := 7.5) -- Time for current speeds
  (t2 : ℝ := 105 / 13) -- Time for adjusted speeds

theorem pedestrian_speeds_unique :
  (x + y = 14) →
  (3 * x + y = 14) →
  x = 6 ∧ y = 8 :=
by
  intros h1 h2
  have : 2 * x = 12 :=
    by ring_nf; sorry
  have hx : x = 6 :=
    by linarith
  have hy : y = 8 :=
    by linarith
  exact ⟨hx, hy⟩

end pedestrian_speeds_unique_l527_527098


namespace front_view_length_l527_527149

-- Define the conditions of the problem
variables (d_body : ℝ) (d_side : ℝ) (d_top : ℝ)
variables (d_front : ℝ)

-- The given conditions
def conditions :=
  d_body = 5 * Real.sqrt 2 ∧
  d_side = 5 ∧
  d_top = Real.sqrt 34

-- The theorem to be proved
theorem front_view_length : 
  conditions d_body d_side d_top →
  d_front = Real.sqrt 41 :=
sorry

end front_view_length_l527_527149


namespace smallest_a_l527_527428

theorem smallest_a (a x : ℤ) (hx : x^2 + a * x = 30) (ha_pos : a > 0) (product_gt_30 : ∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = -a ∧ x₁ * x₂ > 30) : a = 11 :=
sorry

end smallest_a_l527_527428


namespace scientific_notation_of_probe_unit_area_l527_527420

-- Given condition: the area of each probe unit of a certain chip
def probe_unit_area : ℝ := 0.00000164

-- Goal: to prove that 0.00000164 can be expressed in scientific notation as 1.64 * 10^(-6)
theorem scientific_notation_of_probe_unit_area : probe_unit_area = 1.64 * 10^(-6) := 
by
  sorry

end scientific_notation_of_probe_unit_area_l527_527420


namespace min_distance_l527_527323

open Real

noncomputable def ellipse (x y : ℝ) := (x^2) / 3 + y^2 = 1
noncomputable def line (x y : ℝ) := x + y - 4 = 0
noncomputable def distance (P Q : ℝ × ℝ) := sqrt (((fst P) - (fst Q))^2 + ((snd P) - (snd Q))^2)

theorem min_distance {x1 y1 x2 y2 : ℝ} 
    (hP : ellipse x1 y1) 
    (hQ : line x2 y2) :
    ∃ Q : ℝ × ℝ, (line (fst Q) (snd Q)) ∧ 
    (∀ P : ℝ × ℝ, ellipse (fst P) (snd P) → distance P Q ≥ sqrt 2) :=
sorry

end min_distance_l527_527323


namespace factory_output_decrease_l527_527130

noncomputable def original_output (O : ℝ) : ℝ :=
  O

noncomputable def increased_output_10_percent (O : ℝ) : ℝ :=
  O * 1.1

noncomputable def increased_output_30_percent (O : ℝ) : ℝ :=
  increased_output_10_percent O * 1.3

noncomputable def percentage_decrease_needed (original new_output : ℝ) : ℝ :=
  ((new_output - original) / new_output) * 100

theorem factory_output_decrease (O : ℝ) : 
  abs (percentage_decrease_needed (original_output O) (increased_output_30_percent O) - 30.07) < 0.01 :=
by
  sorry

end factory_output_decrease_l527_527130


namespace average_value_f_l527_527912

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end average_value_f_l527_527912


namespace ram_krish_hari_task_completion_l527_527742

theorem ram_krish_hari_task_completion
  (R K H : ℝ)
  (hr : R = 1 / 2 * K)
  (wrk_ram : ∀ (D : ℝ), D = 36 → W = R * D)
  (wrk_hari : ∀ (D : ℝ), D = 18 → W = H * D)
  (wrk_krish : ∀ (D : ℝ), D = 36 / 2 → W = K * D)
  (combined_efficiency : 5 * R)
  (W : ℝ) :
  (K = 2 * R) →
  (H = 2 * R) →
  (R * 36 = H * 18) →
  let T := W / combined_efficiency in
  T = 7.2 := 
by
  sorry

end ram_krish_hari_task_completion_l527_527742


namespace fourth_term_of_sequence_l527_527431

theorem fourth_term_of_sequence :
  let seq (n : ℕ) := match n with
                    | 1 => 4^(1/4 : ℝ)
                    | 2 => 4^(1/2 : ℝ)
                    | 3 => 4^(3/4 : ℝ)
                    | n => Real.sqrt (seq (n - 1) * n)
  in seq 4 = 2 * Real.sqrt 6 :=
by
  sorry

end fourth_term_of_sequence_l527_527431


namespace determine_f_l527_527278

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem determine_f (a b c : ℝ) (h₁ : a > 0)
    (h₂ : ∀ x : ℝ, |x| ≤ 1 → |f x a b c| ≤ 1)
    (h₃ : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g x a b ≤ 2) :
    ∀ x : ℝ, f x 2 0 -1 = 2 * x^2 - 1 := by
  sorry

end determine_f_l527_527278


namespace a_2017_eq_2_l527_527782

variable (n : ℕ)
variable (S : ℕ → ℤ)

/-- Define the sequence sum Sn -/
def S_n (n : ℕ) : ℤ := 2 * n - 1

/-- Define the sequence term an -/
def a_n (n : ℕ) : ℤ := S_n n - S_n (n - 1)

theorem a_2017_eq_2 : a_n 2017 = 2 := 
by
  have hSn : ∀ n, S_n n = (2 * n - 1) := by intro; simp [S_n] 
  have ha : ∀ n, a_n n = (S_n n - S_n (n - 1)) := by intro; simp [a_n]
  simp only [ha, hSn] 
  sorry

end a_2017_eq_2_l527_527782


namespace angle_C_obtuse_l527_527666

theorem angle_C_obtuse (a b c C : ℝ) (h1 : a^2 + b^2 < c^2) (h2 : Real.sin C = Real.sqrt 3 / 2) : C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_obtuse_l527_527666


namespace cubic_polynomial_exists_l527_527219

theorem cubic_polynomial_exists
  (P : ℂ[X]) (h_real_coeff : ∀ x : ℂ, P.eval x ∈ ℝ) (h_root : P.eval (4 + Complex.i) = 0)
  (h_x3_coeff : P.coeff 3 = 3) (h_P_1 : P.eval 1 = 0) :
  P = 3 * (X ^ 3 - 9 * X ^ 2 + 25 * X - 17) := sorry

end cubic_polynomial_exists_l527_527219


namespace max_sum_5_or_6_l527_527944

variable {a : Nat → ℤ}

def is_arithmetic_seq (d : ℤ) (a : Nat → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def S_n (a : Nat → ℤ) (n : Nat) : ℤ :=
  ∑ i in Finset.range n, a (i + 1)

theorem max_sum_5_or_6 (d : ℤ) (a : Nat → ℤ) 
  (h_arith : is_arithmetic_seq d a)
  (h_neg_d : d < 0)
  (h_cond : a 3 = -(a 9)) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, m < n → S_n a m < S_n a n) ∧ ((S_n a n ≤ S_n a (n + 1)) ∨ (n = 5 ∨ n = 6)) :=
by
  sorry

end max_sum_5_or_6_l527_527944


namespace overall_average_marks_l527_527761

theorem overall_average_marks
  (total_boys : ℕ)
  (avg_passed : ℕ)
  (avg_failed : ℕ)
  (num_passed : ℕ)
  (num_failed : total_boys - num_passed)
  (total_boys = 120)
  (avg_passed = 39)
  (avg_failed = 15)
  (num_passed = 115)
: (num_passed * avg_passed + num_failed * avg_failed) / total_boys = 38 :=
by sorry

end overall_average_marks_l527_527761


namespace paper_thickness_after_2_folds_l527_527475

theorem paper_thickness_after_2_folds:
  ∀ (initial_thickness : ℝ) (folds : ℕ),
  initial_thickness = 0.1 →
  folds = 2 →
  (initial_thickness * 2^folds = 0.4) :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end paper_thickness_after_2_folds_l527_527475


namespace g_is_even_l527_527690

noncomputable def g (x : ℝ) : ℝ := 5^(x^2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g x = g (-x) :=
by
  sorry

end g_is_even_l527_527690


namespace total_letters_in_names_is_33_l527_527698

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l527_527698


namespace problem_statement_prime_product_l527_527732

/-- 
The problem statement has 41 words. The task is to prove that the product of the smallest prime factor and the largest prime factor of 41 (both being 41) is 1681.
-/
theorem problem_statement_prime_product :
  let word_count := 41 in
  let smallest_prime_factor := 41 in
  let largest_prime_factor := 41 in
  smallest_prime_factor * largest_prime_factor = 1681 :=
by
  let word_count := 41
  let smallest_prime_factor := 41
  let largest_prime_factor := 41
  show smallest_prime_factor * largest_prime_factor = 1681
  sorry

end problem_statement_prime_product_l527_527732


namespace first_term_of_series_l527_527781

noncomputable def series_term : ℕ → ℕ 
| 1 := 2
| n := n * 2^n

def series_sum (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), series_term (k + 1)

theorem first_term_of_series :
  (∑ k in Finset.range 2012, (k + 1) * 2^(k + 1)) = 8100312 → series_term 1 = 2 :=
by
  intro h
  rw [series_term]
  sorry

end first_term_of_series_l527_527781


namespace log3_8_minus_2log3_6_eq_a_minus_2_l527_527235

variable (a : ℝ)

-- Suppose a = log of base 3 of 2
def log3_2 : ℝ := Real.logb 3 2
-- Define the given variable a as log base 3 of 2
axiom h : a = log3_2

theorem log3_8_minus_2log3_6_eq_a_minus_2 : Real.logb 3 8 - 2 * Real.logb 3 6 = a - 2 := by
  sorry

end log3_8_minus_2log3_6_eq_a_minus_2_l527_527235


namespace implicitly_defined_function_derivatives_l527_527917

noncomputable def y'(x y : ℝ) (h : x^y - y^x = 0) : ℝ :=
  (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1))

noncomputable def y''(x y : ℝ) (h : y' x y h = (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1))) : ℝ :=
  ((x * (3 - 2 * Real.log x) * (Real.log y - 1)^2 + (Real.log x - 1)^2 * (2 * Real.log y - 3) * y) / (x^4 * (Real.log y - 1)^3)) * y^2

theorem implicitly_defined_function_derivatives (x y : ℝ) (h : x^y - y^x = 0) :
  let y_prime := y' x y h in
  let y_double_prime := y'' x y (by sorry) in -- insert the proof of y' == correct answer here
  y' x y h = (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1)) ∧
  y'' x y (by sorry) = ((x * (3 - 2 * Real.log x) * (Real.log y - 1)^2 + (Real.log x - 1)^2 * (2 * Real.log y - 3) * y) / (x^4 * (Real.log y - 1)^3)) * y^2 :=
by
  sorry


end implicitly_defined_function_derivatives_l527_527917


namespace remainder_of_even_coeffs_div_by_3_l527_527835

theorem remainder_of_even_coeffs_div_by_3 
  (a : Fin 2011 → ℤ) 
  (ha : ∀ x : ℤ, (Nat.pow (2 * x + 4) 2010 : ℤ) = ∑ i : Fin 2011, a i * x ^ (i : ℕ)) 
  (ha1 : ∑ i : Fin 2011, a i = 6 ^ 2010) 
  (ha2 : ∑ i : Fin 2011, (-1) ^ (i : ℕ) * a i = 2 ^ 2010) 
  : (∑ i in Finset.filter (λ k, k.val % 2 = 0) Finset.univ, a i) % 3 = 2 := 
sorry

end remainder_of_even_coeffs_div_by_3_l527_527835


namespace max_sales_day_popular_days_l527_527668

open Nat

-- Define sequence of sales for each day in July
def sales (n : ℕ) : ℕ :=
  if n = 0 then 0       -- July 0 is out of range, mapped to 0 pieces
  else if n ≤ 13 then 3 * n
  else if n ≤ 31 then 65 - 2 * n
  else 0                -- After July 31, mapped to 0 pieces

-- Question 1: Prove maximum sales and the day it occurred
theorem max_sales_day :
  ∃ k : ℕ, (4 ≤ k ∧ k ≤ 31 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 31 → sales n ≤ sales k) ∧ sales k = 39 := sorry

-- Question 2: Prove how many days the clothing was popular
theorem popular_days :
  let number_of_popular_days := (fun s : List ℕ => s.filter (fun n => let a := sales n in a >= 20 && a <= 200)).length
  number_of_popular_days (List.range' 1 31) = 11 := sorry

end max_sales_day_popular_days_l527_527668


namespace cone_to_sphere_surface_area_ratio_l527_527855

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
  let h := 3 * r;
  let a := 2 * sqrt 3 * r;
  let sa_sphere := 4 * π * r^2;
  let sa_cone := π * r * (r + sqrt (r^2 + h^2));
  sa_cone / sa_sphere = 9 / 4 :=
by
  sorry

end cone_to_sphere_surface_area_ratio_l527_527855


namespace induction_inequality_subtraction_l527_527102

theorem induction_inequality_subtraction (n : ℕ) (h₁ : n > 1) :
  (∑ i in finset.range (2*n+2).succ, if h : n+1 ≤ i ∧ i ≤ 2*n then (1 : ℝ)/i else 0) -
  (∑ i in finset.range (2*n+1).succ, if h : n+1 ≤ i ∧ i ≤ 2*n then (1 : ℝ)/i else 0)
  = 1 / (2 * n + 1) - 1 / (2 * (n + 1)) :=
  sorry

end induction_inequality_subtraction_l527_527102


namespace eq_a_no_solution_eq_b_has_solutions_eq_c_has_solution_l527_527831

noncomputable def equation_a (x : ℝ) := 
  ((1 + 4 / (x - 2)) * (x - 4 + 4 / x) - 
   (Real.sqrt 3 / (1 - (Real.sqrt 3 - 1) * 3^(-0.5))) * (Real.sqrt x + 2)^(-1)) /
  (x^(-0.5) - 2 * x^(-1))

noncomputable def equation_a_rhs (x : ℝ) := 7 * (x - 1) - x^2

theorem eq_a_no_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4) : 
  equation_a x ≠ equation_a_rhs x :=
sorry

noncomputable def equation_b (x : ℝ) :=
  ((1 + 4 / (x - 2)) * (x - 4 + 4 / x) - 
   (Real.sqrt 3 / (1 - (Real.sqrt 3 - 1) * 3^(-0.5))) * (Real.sqrt x + 2)^(-1)) /
  (x^(-0.5) - 2 * x^(-1))

noncomputable def equation_b_rhs (x : ℝ) := x * (10 - x) - 7

theorem eq_b_has_solutions (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4) : 
  equation_b x = equation_b_rhs x ↔ x = 8 ∨ x = 1 :=
sorry

noncomputable def equation_c (x : ℝ) :=
  ((1 + 4 / (x - 2)) * (x - 4 + 4 / x) - 
   (Real.sqrt 3 / (1 - (Real.sqrt 3 - 1) * 3^(-0.5))) * (Real.sqrt x + 2)^(-1)) /
  (x^(-0.5) - 2 * x^(-1))

noncomputable def equation_c_rhs (x : ℝ) := x * (7 - x) + 1

theorem eq_c_has_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4) : 
  equation_c x = equation_c_rhs x ↔ x = 6 :=
sorry

end eq_a_no_solution_eq_b_has_solutions_eq_c_has_solution_l527_527831


namespace limit_f_at_2_l527_527189

-- Define the function f(x) = (x^3 - 3x - 2) / (x - 2)
def f (x : ℝ) : ℝ := (x^3 - 3 * x - 2) / (x - 2)

-- State the theorem to prove the limit of f(x) as x approaches 2 is 9
theorem limit_f_at_2 : filter.tendsto f (nhds 2) (nhds 9) :=
by 
    sorry

end limit_f_at_2_l527_527189


namespace factorization_correct_l527_527571

theorem factorization_correct (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 :=
by
  sorry

end factorization_correct_l527_527571


namespace blue_balls_taken_out_l527_527086

theorem blue_balls_taken_out :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 7) ∧ (7 - x) / (15 - x) = 1 / 3 ∧ x = 3 :=
sorry

end blue_balls_taken_out_l527_527086


namespace find_m_l527_527262

theorem find_m 
  (x : ℝ)
  (m : ℝ)
  (h1 : log 5 (sin x) + log 5 (cos x) = -2)
  (h2 : log 5 (sin x - cos x) = (1/2) * (log 5 m - 2)) :
  m = 92 := 
sorry

end find_m_l527_527262


namespace count_factors_of_5_in_factorial_340_l527_527773

theorem count_factors_of_5_in_factorial_340 :
  (∑ k in Finset.range (Nat.log 340 / Nat.log 5).natCeil, 340 / 5^k) = 83 :=
by
  sorry

end count_factors_of_5_in_factorial_340_l527_527773


namespace Mike_watches_TV_every_day_l527_527015

theorem Mike_watches_TV_every_day :
  (∃ T : ℝ, 
  (3 * (T / 2) + 7 * T = 34) 
  → T = 4) :=
by
  let T := 4
  sorry

end Mike_watches_TV_every_day_l527_527015


namespace greater_number_l527_527446

theorem greater_number (x y : ℕ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := 
by 
  sorry

end greater_number_l527_527446


namespace gear_C_rotation_direction_gear_C_rotation_count_l527_527879

/-- Definition of the radii of the gears -/
def radius_A : ℝ := 15
def radius_B : ℝ := 10 
def radius_C : ℝ := 5

/-- Gear \( A \) drives gear \( B \) and gear \( B \) drives gear \( C \) -/
def drives (x y : ℝ) := x * y

/-- Direction of rotation of gear \( C \) when gear \( A \) rotates clockwise -/
theorem gear_C_rotation_direction : drives radius_A radius_B = drives radius_C radius_B → drives radius_A radius_B > 0 → drives radius_C radius_B > 0 := by
  sorry

/-- Number of rotations of gear \( C \) when gear \( A \) makes one complete turn -/
theorem gear_C_rotation_count : ∀ n : ℝ, drives radius_A radius_B = drives radius_C radius_B → (n * radius_A)*(radius_B / radius_C) = 3 * n := by
  sorry

end gear_C_rotation_direction_gear_C_rotation_count_l527_527879


namespace unique_bijective_function_exists_l527_527221

noncomputable def unique_bijective_function : Prop :=
  ∃! (f : ℝ → ℝ), function.bijective f ∧ ∀ x y, f (x + f y) = f x + y

theorem unique_bijective_function_exists : unique_bijective_function :=
sorry

end unique_bijective_function_exists_l527_527221


namespace angle_between_vectors_l527_527252

noncomputable def vector_angle (a b : ℝ) : ℝ := 
let cos_theta := (a - b) • (a - 3*b) = 0 in
acos cos_theta

theorem angle_between_vectors 
  (a b : ℝ)
  (h1 : a ≠ 0 ∧ b ≠ 0)
  (h2 : |a| = sqrt 3 * |b|)
  (h3 : (a - b) • (a - 3*b) = 0) :
  vector_angle a b = π / 6 := 
sorry

end angle_between_vectors_l527_527252


namespace tangent_line_of_ellipse_l527_527430

theorem tangent_line_of_ellipse 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (x₀ y₀ : ℝ) (hx₀ : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, x₀ * x / a^2 + y₀ * y / b^2 = 1 := 
sorry

end tangent_line_of_ellipse_l527_527430


namespace middle_box_label_l527_527391

theorem middle_box_label :
  ∃ (boxes : fin 23 → Prop), 
  (∃ i, boxes i ∧ i = 11) ∧ 
  ((∀ j, (boxes j ∧ j ≠ 11) -> (¬(j = 10) ↔ ¬(j = 12))) ∧ 
  (∀ j, ¬boxes j ∨ boxes (j+1) ∨ boxes (j-1)) ∧ 
  ∃! i, (i = 11 ∨ ¬(i = 11)) := sorry

end middle_box_label_l527_527391


namespace largest_n_for_tangent_circles_sequence_l527_527194

theorem largest_n_for_tangent_circles_sequence (R r : ℝ) (n : ℕ) 
  (hR : R = 2008) 
  (hr1 : r = 1)
  (hn_upper : n ≤ Int.floor (Real.sqrt (R / r + 1))) :
  n = 44 :=
by 
  have h : Real.sqrt (R / r + 1) = 44.83, from sorry,
  have hn : n ≤ 44, from sorry,
  exact sorry

end largest_n_for_tangent_circles_sequence_l527_527194


namespace final_position_is_negative_one_total_revenue_is_118_yuan_l527_527498

-- Define the distances
def distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]

-- Define the taxi price per kilometer
def price_per_km : Int := 2

-- Theorem to prove the final position of the taxi relative to Wu Zhong
theorem final_position_is_negative_one : 
  List.sum distances = -1 :=
by 
  sorry -- Proof omitted

-- Theorem to prove the total revenue for the afternoon
theorem total_revenue_is_118_yuan : 
  price_per_km * List.sum (List.map Int.natAbs distances) = 118 :=
by
  sorry -- Proof omitted

end final_position_is_negative_one_total_revenue_is_118_yuan_l527_527498


namespace base8_representation_digits_l527_527205

-- Definition: Given conditions for the problem
def decimal_number : ℕ := 157
def base : ℕ := 8

-- Theorem: Statement to prove that the number of digits of 157 in base 8 is 3
theorem base8_representation_digits : (nat.log_base base decimal_number) + 1 = 3 := by
  sorry

end base8_representation_digits_l527_527205


namespace correct_statements_l527_527120

theorem correct_statements :
  (∀ x y : ℝ, (0 < x ∧ x < 1) → (0 < y ∧ y < 1) → x + y < 2) ∧
  ((∀ a b : ℝ, a > 1 → b > 1 → a * b > 1) ∧ (∀ a b : ℝ, a * b > 1 → a > 1 ∧ b > 1)) ∧
  ¬ (∀ x y : ℝ, |x| > |y| ∧ x > y) ∧
  (∀ m : ℝ, m < 0 → ∃ (x₁ x₂ : ℝ), x₁ * x₂ = m ∧ x₁ + x₂ = 2 ∧ (x₁ > 0 ∧ x₂ < 0)) :=
begin
  sorry
end

end correct_statements_l527_527120


namespace range_of_even_function_l527_527589

theorem range_of_even_function (m : ℝ) :
  (∀ x : ℝ, f x = f (-x)) ∧ f(x) = (m * x^2 + (m + 1) * x + 2) ∧ (x ∈ [-2, 2]) → (range f = [-2, 2]) :=
by
  -- to prove that the range of f(x) = mx^2 + (m+1)x + 2 for x ∈ [-2, 2] is equal to [-2, 2]
  sorry

end range_of_even_function_l527_527589


namespace arithmetic_sequence_sum_l527_527952

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_condition : a 4 + a 8 = 12) :
  let S := λ n, n * (a 1 + a n) / 2 in
  S 11 = 66 :=
by
  sorry

end arithmetic_sequence_sum_l527_527952


namespace greatest_x_for_prime_expression_l527_527462

def is_prime (n : ℕ) : Prop := sorry -- use appropriate prime test

theorem greatest_x_for_prime_expression : 
  ∃ x : ℤ, |5 * x^2 - 42 * x + 8| ∈ (n : ℕ) → is_prime n ∧ ∀ y : ℤ, (|5 * y^2 - 42 * y + 8| ∈ (n : ℕ) → is_prime n) → y ≤ 5 := 
begin
  sorry
end

end greatest_x_for_prime_expression_l527_527462


namespace smallest_digit_divisible_by_11_l527_527582

theorem smallest_digit_divisible_by_11 :
  ∃ (d : ℕ), d < 10 ∧ ∀ n : ℕ, (n + 45000 + 1000 + 457 + d) % 11 = 0 → d = 5 :=
by {
  sorry
}

end smallest_digit_divisible_by_11_l527_527582


namespace solution_set_l527_527630

noncomputable def f (x : ℝ) : ℝ := x * real.sin x + real.cos x + x^2

theorem solution_set (x : ℝ) :
  (f (real.log x) < f 1) ↔ (1 / real.exp 1 < x ∧ x < real.exp 1) :=
sorry

end solution_set_l527_527630


namespace range_of_g_l527_527004

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f (x))))

theorem range_of_g : ∀ x, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by
  intro x h
  sorry

end range_of_g_l527_527004


namespace scientific_notation_l527_527417

-- Define the condition: the area of each probe unit of a certain chip is 0.00000164 cm^2
def area_probe_unit : ℝ := 0.00000164

-- The mathematical problem to express 0.00000164 in scientific notation
theorem scientific_notation :
  area_probe_unit = 1.64 * 10 ^ (-6) := 
  by
    sorry

end scientific_notation_l527_527417


namespace geometric_mean_sequence_l527_527939

theorem geometric_mean_sequence {a : ℕ → ℝ} 
  (h_geom : ∀ n, a (n + 1) = a n * 2) 
  (h_start : a 1 = 1) : 
  real.sqrt (a 2 * a 8) = 16 :=
sorry

end geometric_mean_sequence_l527_527939


namespace product_formula_l527_527885

theorem product_formula {Q : ℚ} (n : ℕ) (h : n = 2023) : 
  (Q = ∏ k in (finset.range (n + 1)).filter (λ k, k ≥ 3), (1 - (1 : ℚ) / k)) → 
  Q = 2 / 2023 :=
by
  intros
  subst h
  sorry

end product_formula_l527_527885


namespace solve_for_y_l527_527561

theorem solve_for_y (y : ℚ) (h : |5 * y - 6| = 0) : y = 6 / 5 :=
by 
  sorry

end solve_for_y_l527_527561


namespace exists_nine_six_one_five_l527_527214

theorem exists_nine_six_one_five : ∃ n : ℕ, (n = 9615) ∧ (String.startsWith (nat_to_digits 10 (n^3)).take 4 "8888") :=
by
  use 9615
  split
  · rfl
  · sorry

end exists_nine_six_one_five_l527_527214


namespace reporter_expected_earnings_per_hour_l527_527520

def politics_article_payment := 60
def business_article_payment := 70
def science_article_payment := 80
def politics_and_business_word_rate := 0.1
def science_word_rate := 0.15
def words_per_article := 1500
def bonus_rate := 25
def writing_rate_min := 8
def writing_rate_max := 12
def total_politics_articles := 2
def total_business_articles := 2
def total_science_articles := 1
def expected_hourly_earnings := 291.25

theorem reporter_expected_earnings_per_hour :
  let politics_total_payment := politics_article_payment + words_per_article * politics_and_business_word_rate
  let business_total_payment := business_article_payment + words_per_article * politics_and_business_word_rate
  let science_total_payment := science_article_payment + words_per_article * science_word_rate
  let total_payment := total_politics_articles * politics_total_payment + total_business_articles * business_total_payment + total_science_articles * science_total_payment
  let total_hours := 4
  (total_payment / total_hours = expected_hourly_earnings) :=
sorry

end reporter_expected_earnings_per_hour_l527_527520


namespace total_wheels_l527_527338

theorem total_wheels : 
  let cars := 2
  let car_wheels := 4
  let bikes := 2
  let bike_wheels := 2
  let trash_can := 1
  let trash_can_wheels := 2
  let tricycle := 1
  let tricycle_wheels := 3
  let roller_skates := 2
  let roller_skate_wheels := 4
  let total :=
    cars * car_wheels + 
    bikes * bike_wheels + 
    trash_can * trash_can_wheels + 
    tricycle * tricycle_wheels + 
    roller_skates * roller_skate_wheels 
  in
  total = 25 :=
begin
  sorry
end

end total_wheels_l527_527338


namespace jonathans_and_sisters_total_letters_l527_527701

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l527_527701


namespace find_angle_l527_527963

variables (a b : ℝ^2)
variable (t : ℝ)

def magnitude (v : ℝ^2) : ℝ := real.sqrt (v.x * v.x + v.y * v.y)

def angle_between (u v : ℝ^2) : ℝ := real.acos ((u.1 * v.1 + u.2 * v.2) / (magnitude u * magnitude v))

axiom condition1 : magnitude a = 1
axiom condition2 : magnitude b = 2
axiom condition3 : ∀ t : ℝ, magnitude (b + t • a) ≥ magnitude (b - a)

theorem find_angle : angle_between (2 • a - b) b = 2 * real.pi / 3 := sorry

end find_angle_l527_527963


namespace simplify_polynomial_expression_l527_527750

noncomputable def polynomial_simplify : Polynomial ℚ :=
  (2 * X^6 + X^5 + 3 * X^4 + 2 * X^2 + 15) - (X^6 + X^5 + 4 * X^4 - X^3 + X^2 + 18)

theorem simplify_polynomial_expression :
  polynomial_simplify = X^6 - X^4 + X^3 + X^2 - 3 :=
by
  sorry

end simplify_polynomial_expression_l527_527750


namespace determine_number_of_chairs_l527_527686

variables (J : ℕ) -- Total number of chairs
variables (O : ℕ) -- Number of occupied chairs

def chair_legs := 4 -- Each chair has four legs
def table_legs := 3 -- The table has three legs
def occupied_chair_legs := 6 -- Each occupied chair (chair + scout) has six legs
def unoccupied_chairs := 2 -- Two unoccupied chairs

def total_legs := 101 -- Total number of legs in the room

theorem determine_number_of_chairs 
  (J = O + unoccupied_chairs) -- Total chairs is the sum of occupied and unoccupied
  (total_legs = (table_legs + 2 * chair_legs) + O * occupied_chair_legs) : J = 17 :=
sorry

end determine_number_of_chairs_l527_527686


namespace rationalize_denominator_sum_l527_527404

theorem rationalize_denominator_sum :
  let A := 2 in
  let B := 36 in
  let C := 9 in
  A + B + C = 47 :=
by
  sorry

end rationalize_denominator_sum_l527_527404


namespace imaginary_unit_power_l527_527957

theorem imaginary_unit_power :
  ∀ (i : ℂ), i^2 = -1 → i^2019 = -i :=
begin
  intros,
  sorry
end

end imaginary_unit_power_l527_527957


namespace max_unmarried_women_under_40_not_in_management_l527_527316

-- Define the given conditions as hypotheses

def total_people : ℕ := 150
def fraction_women : ℚ := 3/5
def fraction_women_under_40 : ℚ := 1/2
def fraction_married : ℚ := 1/3
def fraction_married_in_management : ℚ := 3/4

-- Define the number of individuals in different categories
def total_women : ℕ := (fraction_women * total_people).toNat
def women_under_40 : ℕ := (fraction_women_under_40 * total_women).toNat
def total_married : ℕ := (fraction_married * total_people).toNat
def married_in_management : ℕ := (fraction_married_in_management * total_married).toNat

-- Hypothesize that all married people are assumed to be married women to infer maximum number.
def unmarried_women : ℕ := total_women - total_married

-- Given the provided conditions, prove that the maximum number of 
-- unmarried women under the age of 40 who are not in management positions is 45.

theorem max_unmarried_women_under_40_not_in_management :
  let max_unmarried_women_under_40 := unmarried_women + (women_under_40 - unmarried_women)
  max_unmarried_women_under_40 = 45 := 
  by 
    sorry -- Proof is not required, only the statement.

end max_unmarried_women_under_40_not_in_management_l527_527316


namespace parallel_line_eq_perpendicular_line_eq_l527_527429

theorem parallel_line_eq (a b m : ℝ) (P : ℝ × ℝ) : 
  let l1 := a * x + b * y + c = 0,
      l2 := a * x + b * y + m = 0,
      P = (-1, 3) in
  a = 3 ∧ b = 4 ∧ c = -12 ∧ P.1 = -1 ∧ P.2 = 3 → 
  m = -9 →
  l2 = 3 * x + 4 * y - 9 = 0 :=
by sorry

theorem perpendicular_line_eq (a b n : ℝ) : 
  let l1 := a * x + b * y + c = 0,
      l2 := b * x - a * y + n = 0,
      area := 4 in
  a = 3 ∧ b = 4 ∧ c = -12 →
  (b * x - a * y + n = 0) →
  abs(n^2) = 96 →
  l2 = 4 * x - 3 * y + 4 * sqrt 6 = 0 ∨ l2 = 4 * x - 3 * y - 4 * sqrt 6 = 0 :=
by sorry

end parallel_line_eq_perpendicular_line_eq_l527_527429


namespace number_of_x_for_acute_triangle_l527_527556

theorem number_of_x_for_acute_triangle : 
  (∃ (x : ℤ), 11 + 25 > x ∧ 11 + x > 25 ∧ 25 + x > 11 ∧ 22.45 < x ∧ x < 27.3) → 
  { x : ℤ | 11 + 25 > x ∧ 11 + x > 25 ∧ 25 + x > 11 ∧ 22.45 < x ∧ x < 27.3}.card = 5 :=
by sorry

end number_of_x_for_acute_triangle_l527_527556


namespace find_a_l527_527765

theorem find_a (a : ℝ) :
  let c_x := 1
  let c_y := 4
  let distance := (abs (a + c_y - 1)) / (sqrt (a^2 + 1))
  (distance = 1) -> (a = -4/3) :=
by
  intros c_x c_y distance h
  sorry

end find_a_l527_527765


namespace rational_polynomial_coefficients_l527_527574

theorem rational_polynomial_coefficients (P : ℂ[X]) 
  (h : ∀ q : ℚ, eval (algebraMap ℚ ℂ q) P ∈ ℚ) : 
  ∀ n, coeff P n ∈ ℚ :=
sorry

end rational_polynomial_coefficients_l527_527574


namespace correct_statements_l527_527271

noncomputable def curveC (t : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - t)) + (y^2 / (t - 1)) = 1

theorem correct_statements (t : ℝ) :
  (curveC t → (t > 4 ∨ t < 1)) ∧
  (1 < t ∧ t < 5 / 2 → ∃ a b : ℝ, 
    a^2 = 4 - t ∧ b^2 = t - 1 ∧ 
    ∀ x, x = sqrt (5 - 2 * t)) ∧
  (t < 1 → ∃ z : ℝ, z = sqrt (1 - t)) :=
by
  sorry

end correct_statements_l527_527271


namespace part_one_part_two_l527_527710

-- Set A definition
def setA (x : ℝ) : Prop :=
  ∃ y, y = real.sqrt (3 - x) + real.log (real.sqrt (x + 2) + 1) / real.log 2

-- Set B definition
def setB (x m : ℝ) : Prop :=
  2 - m ≤ x ∧ x ≤ 2 * m - 3

-- part(1): Proving the range of m
theorem part_one (m : ℝ) :
  (∀ x, setA x → setB x m) ∧ ¬(∀ x, setB x m → setA x) → (4 ≤ m) :=
sorry

-- part(2): Proving the range of m
theorem part_two (m : ℝ) :
  (∀ x, setA x ∨ setB x m ↔ setA x) → (m ∈ Iic 3) :=
sorry

end part_one_part_two_l527_527710


namespace sum_of_eight_smallest_multiples_of_12_l527_527112

theorem sum_of_eight_smallest_multiples_of_12 :
  let sum_n := (n : ℕ) → (n * (n + 1)) / 2
  12 * sum_n 8 = 432 :=
by
  sorry

end sum_of_eight_smallest_multiples_of_12_l527_527112


namespace P_Q_H_collinear_l527_527356

open EuclideanGeometry

-- Define the main entities for the problem
variable (A B C H M N P Q : Point)

-- Condition Definitions
variable (ABC_triangle : triangle A B C) (H_is_orthocenter : orthocenter H A B C)
variable (M_on_AB : on_line_segment M A B) (N_on_AC : on_line_segment N A C)
variable (P_Q_intersect_circles : ∃ P Q, P ≠ Q ∧ 
  is_on_circle P (circum_circle (diameter B N)) ∧ 
  is_on_circle Q (circum_circle (diameter C M)))

-- Proof Objective
theorem P_Q_H_collinear 
    (hABC : triangle A B C) 
    (horth_H : orthocenter H A B C)
    (hM_on_AB : on_line_segment M A B)
    (hN_on_AC : on_line_segment N A C) 
    (hPQ_intersect : ∃ P Q, P ≠ Q ∧ 
      is_on_circle P (circum_circle (diameter B N)) ∧ 
      is_on_circle Q (circum_circle (diameter C M))) :
    collinear P Q H := 
sorry

end P_Q_H_collinear_l527_527356


namespace number_divisibility_l527_527809

theorem number_divisibility :
  (let n := 6268440 in n % 30 = 0) :=
by
  let n := 6268440
  sorry

end number_divisibility_l527_527809


namespace new_pressure_eq_l527_527541

-- Defining the initial conditions and values
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3.5
def new_volume : ℝ := 10.5
def k : ℝ := initial_pressure * initial_volume

-- The statement to prove
theorem new_pressure_eq :
  ∃ p_new : ℝ, new_volume * p_new = k ∧ p_new = 8 / 3 :=
by
  use (8 / 3)
  sorry

end new_pressure_eq_l527_527541


namespace area_of_triangle_CMN_l527_527326

theorem area_of_triangle_CMN (ABCD_is_square : ∀ A B C D : ℝ × ℝ, is_square A B C D ∧ area ABCD = 1)
  (CMN_is_isosceles_right : ∀ C M N : ℝ × ℝ, is_isosceles_right_triangle C M N (CN = MN)) : 
  area CMN = 4 - 2 * real.sqrt 3 := 
begin
  sorry
end

end area_of_triangle_CMN_l527_527326


namespace items_count_l527_527164

variable (N : ℕ)

-- Conditions
def item_price : ℕ := 50
def discount_rate : ℕ := 80
def sell_percentage : ℕ := 90
def creditors_owed : ℕ := 15000
def money_left : ℕ := 3000

-- Definitions based on the conditions
def sale_price : ℕ := (item_price * (100 - discount_rate)) / 100
def money_before_paying_creditors : ℕ := money_left + creditors_owed
def total_revenue (N : ℕ) : ℕ := (sell_percentage * N * sale_price) / 100

-- Problem statement
theorem items_count : total_revenue N = money_before_paying_creditors → N = 2000 := by
  intros h
  sorry

end items_count_l527_527164


namespace total_ladybugs_l527_527089

theorem total_ladybugs (leaves : Nat) (ladybugs_per_leaf : Nat) (total_ladybugs : Nat) : 
  leaves = 84 → 
  ladybugs_per_leaf = 139 → 
  total_ladybugs = leaves * ladybugs_per_leaf → 
  total_ladybugs = 11676 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end total_ladybugs_l527_527089


namespace find_triples_l527_527216

theorem find_triples : 
  { (a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1) } = 
  { (1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3) } := 
by
  sorry

end find_triples_l527_527216


namespace sum_of_integers_l527_527427

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l527_527427


namespace animals_arrangement_l527_527411

theorem animals_arrangement : 
    let chickens := 5 
    let dogs := 2
    let cats := 5
    let rabbits := 3
    let total_animals := chickens + dogs + cats + rabbits

    (∃ chickens_pos dogs_pos cats_pos rabbits_pos : Finset (Fin 15),
      disjoint chickens_pos dogs_pos ∧ disjoint chickens_pos cats_pos ∧ disjoint chickens_pos rabbits_pos ∧
      disjoint dogs_pos cats_pos ∧ disjoint dogs_pos rabbits_pos ∧ disjoint cats_pos rabbits_pos ∧
      (chickens_pos.card = chickens) ∧ (dogs_pos.card = dogs) ∧ (cats_pos.card = cats) ∧ (rabbits_pos.card = rabbits)) →
    (15.choose 5 * 10.choose 2 * 8.choose 5 * 3.choose 3 * factorial 5 * factorial 2 * factorial 5 * factorial 3 = 4147200) :=
by
  simp [nat.choose, factorial]
  sorry

end animals_arrangement_l527_527411


namespace smallest_positive_integer_linear_combination_l527_527802

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l527_527802


namespace potassium_bromate_molecular_weight_l527_527794

def potassium_atomic_weight : Real := 39.10
def bromine_atomic_weight : Real := 79.90
def oxygen_atomic_weight : Real := 16.00
def oxygen_atoms : Nat := 3

theorem potassium_bromate_molecular_weight :
  potassium_atomic_weight + bromine_atomic_weight + oxygen_atoms * oxygen_atomic_weight = 167.00 :=
by
  sorry

end potassium_bromate_molecular_weight_l527_527794


namespace poles_inside_base_l527_527132

theorem poles_inside_base (total_poles : ℕ) (count_on_left : ℕ) :
  total_poles = 36 → count_on_left = 2015 → 
  ∃ (n : ℕ), n < 36 ∧ (∃ m : ℕ, count_on_left + n = 36 * m) ∧ n = 1 :=
by
  intros total_poles_eq total_count_eq
  use 1
  split 
  { exact Nat.lt_of_succ_lt_succ (by norm_num) }
  split
  { use 56
    rw [total_poles_eq, total_count_eq]
    norm_num }
  { refl }

end poles_inside_base_l527_527132


namespace length_of_EQ_l527_527790

theorem length_of_EQ
  (EFGH : Trapezoid)
  (EF FG GH HE : ℝ)
  (EF_is_parallel_to_GH : EF = 100 ∧ FG = 55 ∧ GH = 23 ∧ HE = 75 ∧ EF ∥ GH)
  (Q : Point)
  (circle_tangent : ∃ Q, Q ∈ EF ∧ Circle.tangent Q FG ∧ Circle.tangent Q HE) :
  EQ = 750 / 13 :=
by
  sorry

end length_of_EQ_l527_527790


namespace volume_ratio_l527_527468

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l527_527468


namespace binomial_expansion_coefficient_l527_527136

-- Theorem: The coefficient of x^3 in the expansion of (x-2)^6 is -160
theorem binomial_expansion_coefficient :
  (∑ i in Finset.range (6 + 1), (Nat.choose 6 i) * (-2)^i * (x ^ (6 - i))).coeff 3 = -160 := sorry

end binomial_expansion_coefficient_l527_527136


namespace lambda_value_l527_527633

noncomputable def plane_vectors : Type := ℝ × ℝ

open_locale real_inner_product_vector_space

variables (a b : plane_vectors)
variables (a_magnitude : ∥a∥ = 2) (b_magnitude : ∥b∥ = 1)
variables (angle_ab : real.angle (a, b) = real.pi / 3)
variables (lambda : ℝ)

theorem lambda_value (h : inner (a + lambda • b) b = 0) : lambda = -1 :=
sorry

end lambda_value_l527_527633


namespace problem_statement_l527_527723

theorem problem_statement (a b c : ℝ) (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0) (h_condition : a * b + b * c + c * a = 1 / 3) :
  1 / (a^2 - b * c + 1) + 1 / (b^2 - c * a + 1) + 1 / (c^2 - a * b + 1) ≤ 3 :=
by
  sorry

end problem_statement_l527_527723


namespace eval_at_m_plus_1_l527_527964

noncomputable def f (x : ℝ) (m : ℝ) := x ^ (2 + m)

theorem eval_at_m_plus_1 (m : ℝ) (h : m = -2) (h_odd : ∀ x, f (-x) m = - f x m) : f (m+1) m = -1 :=
by
  intro
  sorry

end eval_at_m_plus_1_l527_527964


namespace limit_problem_l527_527190

theorem limit_problem :
  (real.lim (λ n : ℕ, (3^n + 5^n) / (3^(n-1) + 5^(n-1))) n → ∞) = 5 :=
by
sorry

end limit_problem_l527_527190


namespace arithmetic_sequence_15th_term_l527_527535

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_15th_term :
  arithmetic_sequence (-3) 4 15 = 53 :=
by
  sorry

end arithmetic_sequence_15th_term_l527_527535


namespace largest_power_of_2_dividing_n_l527_527578

open Nat

-- Defining given expressions
def n : ℕ := 17^4 - 9^4 + 8 * 17^2

-- The theorem to prove
theorem largest_power_of_2_dividing_n : 2^3 ∣ n ∧ ∀ k, (k > 3 → ¬ 2^k ∣ n) :=
by
  sorry

end largest_power_of_2_dividing_n_l527_527578


namespace PartI_solution_set_PartII_a_range_l527_527937

def f(x : ℝ) (a : ℝ) := x^2 + a
def g(x : ℝ) := abs (x + 1) + abs (x - 2)

theorem PartI_solution_set (a : ℝ) (h_a : a = -4) :
  {x : ℝ | f(x) a ≥ g(x) } = {x | x ≤ -1 - real.sqrt 6 ∨ x ≥ 3} :=
sorry

theorem PartII_a_range (h_empty : ∀ x ∈ set.Icc (0:ℝ) 3, ¬(f(x) a > g(x))) :
  a ≤ -4 :=
sorry

end PartI_solution_set_PartII_a_range_l527_527937


namespace circles_externally_tangent_l527_527557

-- Definitions of the centers and radii based on the given circle equations
def center1 : ℝ × ℝ := (-1, 0)
def radius1 : ℝ := 1
def center2 : ℝ × ℝ := (2, -4)
def radius2 : ℝ := 4

-- Definition for the distance between two points.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The proof statement we need to prove.
theorem circles_externally_tangent :
  distance center1 center2 = radius1 + radius2 :=
by
  sorry

end circles_externally_tangent_l527_527557


namespace hyperbola_eccentricity_l527_527979

theorem hyperbola_eccentricity (a b : ℝ) (h1 : (∃ a b : ℝ, ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1))
                                (h2 : (∃ x y : ℝ, x^2 / 16 + y^2 / 12 = 1))
                                (h3 : (2, 3) ∈ {p : ℝ × ℝ | ∃ x y : ℝ, x^2 / 16 + y^2 / 12 = 1})
                                (h4 : b / a = 3 / 2) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 13 / 2 :=
by
  sorry

end hyperbola_eccentricity_l527_527979


namespace defined_log_div_sqrt_l527_527592

noncomputable def is_defined (y : ℝ) : Prop :=
  (y > 2) ∧ (y < 5)

theorem defined_log_div_sqrt (y : ℝ) : 
  is_defined y ↔ ∃ z, z = (log (5 - y) / sqrt (y - 2)) :=
by
  sorry

end defined_log_div_sqrt_l527_527592


namespace compute_y_l527_527892

noncomputable def sum_geometric_series (r : ℝ) (n : ℕ) : ℝ := (1 - r^(n+1)) / (1 - r)

theorem compute_y : 
  (let s1 : ℝ := sum_geometric_series (1 / 3) 1000000 in
   let s2 : ℝ := sum_geometric_series (-1 / 3) 1000000 in
   s1 * s2 = (1 / (1 - (1 / (9 : ℝ))))) → y = 9 :=
by
  sorry

end compute_y_l527_527892


namespace upper_bound_of_third_inequality_l527_527309

variable (x : ℤ)

theorem upper_bound_of_third_inequality : (3 < x ∧ x < 10) →
                                          (5 < x ∧ x < 18) →
                                          (∃ n, n > x ∧ x > -2) →
                                          (0 < x ∧ x < 8) →
                                          (x + 1 < 9) →
                                          x < 8 :=
by { sorry }

end upper_bound_of_third_inequality_l527_527309


namespace proof_sin_cos_identity_l527_527995

variable (a b : ℝ)
variable (θ : ℝ)
variable h_sin_cos : (sin θ)^6 / a^2 + (cos θ)^6 / b^2 = 1 / (a^2 + b^2)

theorem proof_sin_cos_identity :
  (sin θ)^(12) / a^5 + (cos θ)^(12) / b^5 = (a^7 + b^7) / (a^2 + b^2)^6 :=
by
  sorry

end proof_sin_cos_identity_l527_527995


namespace range_of_a_l527_527273

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x else x^2 - 4 * x + 5

theorem range_of_a (a : ℝ) (h : f a ≥ 1) : 0 ≤ a :=
sorry

end range_of_a_l527_527273


namespace solve_for_x_l527_527044

theorem solve_for_x (x : ℚ) (h : 5 * x + 9 * x = 420 - 10 * (x - 4)) : 
  x = 115 / 6 :=
by
  sorry

end solve_for_x_l527_527044


namespace scientific_notation_correct_l527_527734

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct_l527_527734


namespace count_four_digit_numbers_using_digits_l527_527791

/-- 
To compute the number of distinct four-digit numbers that can be formed using three 1s, 
two 2s, and five 3s.
-/
theorem count_four_digit_numbers_using_digits :
  ∃ n : ℕ, n = 71 ∧
            ∃ (ones twos threes : ℕ),
              ones = 3 ∧
              twos = 2 ∧
              threes = 5 ∧
              (number_of_ways_to_form_four_digit_number ones twos threes = n) := 
sorry

end count_four_digit_numbers_using_digits_l527_527791


namespace continuous_stripe_encircling_tetrahedron_probability_l527_527053

noncomputable def tetrahedron_continuous_stripe_probability : ℚ :=
  let total_combinations := 3^4
  let favorable_combinations := 2 
  favorable_combinations / total_combinations

theorem continuous_stripe_encircling_tetrahedron_probability :
  tetrahedron_continuous_stripe_probability = 2 / 81 :=
by
  -- the proof would be here
  sorry

end continuous_stripe_encircling_tetrahedron_probability_l527_527053


namespace jonathans_and_sisters_total_letters_l527_527699

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l527_527699


namespace construct_quadrilateral_l527_527280

variable {P Q R : Point}
variable {A B C D : Point}

-- Placeholder definition for midpoint (in practice, we would use exact definitions/axioms).
def is_midpoint (P : Point) (A B : Point) : Prop := sorry

-- Placeholder definition for convex quadrilateral (in practice, use proper geometric definitions).
def convex_quadrilateral (A B C D : Point) : Prop := sorry

def sides_equal (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D

theorem construct_quadrilateral :
  is_midpoint P A B →
  is_midpoint Q B C →
  is_midpoint R C D →
  sides_equal A B C D →
  convex_quadrilateral A B C D :=
by
  sorry

end construct_quadrilateral_l527_527280


namespace juan_marbles_eq_64_l527_527135

def connie_marbles : ℕ := 39
def juan_extra_marbles : ℕ := 25

theorem juan_marbles_eq_64 : (connie_marbles + juan_extra_marbles) = 64 :=
by
  -- definition and conditions handled above
  sorry

end juan_marbles_eq_64_l527_527135


namespace product_of_b_values_l527_527438

noncomputable def product_of_roots (a b c : ℝ) : ℝ := c / a

theorem product_of_b_values :
  ∀ b : ℝ,
    (real.sqrt ((3 * b - 5) ^ 2 + (b - 5 - (-2)) ^ 2) = 3 * real.sqrt 13) →
    product_of_roots 10 (-36) (-14.5) = -0.5625 :=
by
  intros b h
  sorry

end product_of_b_values_l527_527438


namespace virginia_more_than_adrienne_l527_527103

def teaching_years (V A D : ℕ) : Prop :=
  V + A + D = 102 ∧ D = 43 ∧ V = D - 9

theorem virginia_more_than_adrienne (V A : ℕ) (h : teaching_years V A 43) : V - A = 9 :=
by
  sorry

end virginia_more_than_adrienne_l527_527103


namespace system_of_equations_solution_l527_527408

theorem system_of_equations_solution (x y : ℝ) (k n : ℤ) :
  sin x * sin y = 0.75 ∧ tan x * tan y = 3 ↔
  ((x = π / 3 + π * (k + n) ∧ y = π / 3 + π * (n - k)) ∨
   (x = -π / 3 + π * (k + n) ∧ y = -π / 3 + π * (n - k))) := 
sorry

end system_of_equations_solution_l527_527408


namespace modulus_of_complex_number_l527_527062

open Complex

theorem modulus_of_complex_number (z : ℂ) (h : abs ((1 : ℂ) + z * Complex.I) = abs (1 + (1 : ℂ) * Complex.I)) :
  abs z = sqrt 5 :=
sorry

end modulus_of_complex_number_l527_527062


namespace partition_possible_l527_527934

theorem partition_possible (n k : ℕ) (hkn1 : k ≥ 1) : 
  (∃ partition : Finset (Fin 2n) × Finset (Fin 2n), 
    (partition.1.card = n ∧ partition.2.card = n) ∧
    (∀ x ∈ partition.1, ∃ y ∈ partition.1, x ≠ y ∧ is_friend x y) ∧
    (∀ x ∈ partition.2, ∃ y ∈ partition.2, x ≠ y ∧ is_friend x y)) ↔ (k = 1 ∨ k = 2) := 
sorry

-- Assumptions to model friends relation
variable {Fin 32 : Type}
variable (is_friend : Fin 32 → Fin 32 → Prop)
axiom is_friend_symm : ∀ x y, is_friend x y → is_friend y x

end partition_possible_l527_527934


namespace intersection_A_B_l527_527257

noncomputable def A : Set ℝ := {Real.sin (Float.pi / 2), Real.cos Float.pi}
def B : Set ℝ := {x : ℝ | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {-1} := 
by simp [A, B]; sorry

end intersection_A_B_l527_527257


namespace teacher_wang_problem_l527_527758

theorem teacher_wang_problem :
  ∃ (n : ℕ) (a b : ℕ), 
  (∀ k, n = k + 3) ∧ 
  (∃ (c : ℕ), a + b + c = 64 ∧ c ∉ {a,b} ∧ ¬ prime c) ∧ 
  (a + b = 60 ∧ prime a ∧ prime b) ∧ 
  ∃ (total_sum sum_remaining : ℕ),
  (total_sum = (n * (n + 1)) / 2) ∧
  sum_remaining = total_sum - (a + b + c) ∧
  sum_remaining / (n - 3) = 179 / 9 := 
sorry

end teacher_wang_problem_l527_527758


namespace find_c_l527_527978

open Function

noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 255 - 5

theorem find_c (c : ℤ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  ∀ k : ℤ, k < c → ¬ ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = k ∧ g x₂ = k ∧ g x₃ = k ∧ g x₄ = k ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ :=
sorry

end find_c_l527_527978


namespace construct_quadrilateral_l527_527281

variable {P Q R : Point}
variable {A B C D : Point}

-- Placeholder definition for midpoint (in practice, we would use exact definitions/axioms).
def is_midpoint (P : Point) (A B : Point) : Prop := sorry

-- Placeholder definition for convex quadrilateral (in practice, use proper geometric definitions).
def convex_quadrilateral (A B C D : Point) : Prop := sorry

def sides_equal (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D

theorem construct_quadrilateral :
  is_midpoint P A B →
  is_midpoint Q B C →
  is_midpoint R C D →
  sides_equal A B C D →
  convex_quadrilateral A B C D :=
by
  sorry

end construct_quadrilateral_l527_527281


namespace best_fit_slope_is_correct_l527_527125

open Real

noncomputable def slope_regression_line (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :=
  (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21

theorem best_fit_slope_is_correct (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h_arith : (x4 - x3 = 2 * (x3 - x2)) ∧ (x3 - x2 = 2 * (x2 - x1))) :
  slope_regression_line x1 x2 x3 x4 y1 y2 y3 y4 = (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21 := 
sorry

end best_fit_slope_is_correct_l527_527125


namespace hyperbola_eccentricity_l527_527064

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (hyp_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (dist_to_asym : ∃ F : ℝ × ℝ, dist F (λ x, (b/a) * x) = sqrt 3 * a) :
  let c := sqrt (a^2 + b^2) in
  b = sqrt 3 * a → c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l527_527064


namespace speed_downstream_l527_527850

-- Given conditions
def Vm : ℝ := 28   -- Speed of the man in still water in kmph
def Vu : ℝ := 25   -- Speed of the man rowing upstream in kmph

-- Claim: Speed of the man rowing downstream
theorem speed_downstream (Vm Vu : ℝ) (hVm : Vm = 28) (hVu : Vu = 25) : ∃ Vd, Vd = 31 :=
by
  have Vs : ℝ := Vm - Vu   -- Speed of the stream
  have hVs : Vs = 3 := by norm_num [hVm, hVu, Vs]
  let Vd := Vm + Vs        -- Speed of the man rowing downstream
  have hVd : Vd = 31 := by norm_num [hVm, hVs, Vd]
  exact ⟨Vd, hVd⟩

end speed_downstream_l527_527850


namespace current_population_l527_527818

def initial_population : ℕ := 684
def growth_rate : ℝ := 0.25
def moving_away_rate : ℝ := 0.40

theorem current_population (P0 : ℕ) (g : ℝ) (m : ℝ) : 
  P0 = initial_population → 
  g = growth_rate → 
  m = moving_away_rate → 
  (P0 + (P0 * g).to_nat - ((P0 + (P0 * g).to_nat) * m).to_nat) = 513 := 
by
  intros hP0 hg hm
  sorry

end current_population_l527_527818


namespace total_emails_received_l527_527692

variable (emails_morning emails_afternoon : ℕ)

def total_emails (morning: ℕ) (afternoon: ℕ) : ℕ := morning + afternoon

theorem total_emails_received :
  emails_morning = 3 → emails_afternoon = 5 → total_emails emails_morning emails_afternoon = 8 :=
by
  intro h_morning h_afternoon
  rw [h_morning, h_afternoon]
  rfl
  sorry

end total_emails_received_l527_527692


namespace percentage_increase_weekends_l527_527181

def weekday_price : ℝ := 18
def weekend_price : ℝ := 27

theorem percentage_increase_weekends : 
  (weekend_price - weekday_price) / weekday_price * 100 = 50 := by
  sorry

end percentage_increase_weekends_l527_527181


namespace total_distance_hiked_l527_527056

-- Defining the distances Terrell hiked on Saturday and Sunday
def distance_Saturday : Real := 8.2
def distance_Sunday : Real := 1.6

-- Stating the theorem to prove the total distance
theorem total_distance_hiked : distance_Saturday + distance_Sunday = 9.8 := by
  sorry

end total_distance_hiked_l527_527056


namespace stock_profit_loss_l527_527304

theorem stock_profit_loss :
  let down_limit_factor := 0.9
  let up_limit_factor := 1.1
  let stock_after_down := down_limit_factor ^ 3
  let stock_after_up := stock_after_down * up_limit_factor ^ 3
  stock_after_up < 1 :=
by
  let down_limit_factor := 0.9
  let up_limit_factor := 1.1
  let stock_after_down := down_limit_factor ^ 3
  let stock_after_up := stock_after_down * up_limit_factor ^ 3
  calc
    stock_after_up = 0.9 ^ 3 * 1.1 ^ 3 : by rw [stock_after_down, down_limit_factor, up_limit_factor]
    ... = 0.970299 : sorry
    ... < 1 : by norm_num

end stock_profit_loss_l527_527304


namespace guo_can_pay_exact_amount_l527_527991

-- Define the denominations and total amount Guo has
def note_denominations := [1, 10, 20, 50]
def total_amount := 20000
def cost_computer := 10000

-- The main theorem stating that Guo can pay exactly 10000 yuan
theorem guo_can_pay_exact_amount : ∃ bills : List ℕ, ∀ (b : ℕ), b ∈ bills → b ∈ note_denominations ∧
  bills.sum = cost_computer :=
sorry

end guo_can_pay_exact_amount_l527_527991


namespace shape_with_diagonal_angle_is_hexagon_l527_527070

noncomputable def shape_with_diagonal_angle_60 (s : Type) [shape s] : Prop :=
  ∃ (h : regular_hexagon s), 
  ∀ (a b : s), adjacent a b → angle a b = 60

theorem shape_with_diagonal_angle_is_hexagon {s : Type} [shape_with_diagonal_angle_60 s] :
  regular_hexagon s :=
sorry

end shape_with_diagonal_angle_is_hexagon_l527_527070


namespace arithmetic_sequence_ratio_proof_l527_527712

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_seq (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

-- Given Condition: a_3 / a_6 = 11 / 5
axiom h1 : a 3 / a 6 = 11 / 5

-- Prove that S_5 / S_11 = 1
theorem arithmetic_sequence_ratio_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 3 / a 6 = 11 / 5) :
  (S 5 / S 11) = 1 :=
by
  sorry

end arithmetic_sequence_ratio_proof_l527_527712


namespace min_elements_in_set_l527_527491

theorem min_elements_in_set {n : ℕ} 
  (h_n : n > 1) 
  (a : Finₓ n → ℕ)
  (distinct : ∀ (i j : Finₓ n), i ≠ j → a i ≠ a j) :
  let M := {p | ∃ (i j : Finₓ n) (h_ij : i < j), p = (Nat.gcd (a i) (a j), Nat.lcm (a i) (a j))}
  in M.size = n :=
by
  -- Proof is omitted.
  sorry

end min_elements_in_set_l527_527491


namespace exchange_ways_100_yuan_l527_527807

theorem exchange_ways_100_yuan : ∃ n : ℕ, n = 6 ∧ (∀ (x y : ℕ), 20 * x + 10 * y = 100 ↔ y = 10 - 2 * x):=
by
  sorry

end exchange_ways_100_yuan_l527_527807


namespace part1_part2_l527_527596

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

-- Proof statements
theorem part1 (k : ℝ) (collinear : ∃ c : ℝ, (k * a.fst - b.fst, k * a.snd - b.snd) = c * (a.fst + 2 * b.fst, a.snd + 2 * b.snd)) :
  k = -1/2 := sorry

theorem part2 (m : ℝ) (collinear : ∃ λ : ℝ, 2 * a.fst + 3 * b.fst = λ * (a.fst + m * b.fst) ∧ 2 * a.snd + 3 * b.snd = λ * (a.snd + m * b.snd)) :
  m = 3/2 := sorry

end part1_part2_l527_527596


namespace actual_distance_between_A_and_B_l527_527021

theorem actual_distance_between_A_and_B (scale : ℕ) (distance_on_map_cm : ℕ) :
  (scale = 20000) → (distance_on_map_cm = 3) → (distance_on_map_cm * scale) / 100 = 600 :=
by
  intros h_scale h_distance_on_map
  rw [h_scale, h_distance_on_map]
  norm_num
  sorry

end actual_distance_between_A_and_B_l527_527021


namespace students_playing_sports_l527_527128

theorem students_playing_sports (b c bc : ℕ) (hb : b = 7) (hc : c = 8) (hbc : bc = 3) :
  b + c - bc = 12 := 
by
  rw [hb, hc, hbc]
  simp
  sorry

end students_playing_sports_l527_527128


namespace bike_travel_distance_l527_527881

-- Define the known values
def radius : ℝ := 30
def rotations : ℕ := 5

-- Define the circumference of the tire
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define total distance travelled
def total_distance_travelled (r : ℝ) (n : ℕ) : ℝ :=
  n * circumference r

-- The theorem to prove
theorem bike_travel_distance : total_distance_travelled radius rotations = 300 * Real.pi :=
by sorry

end bike_travel_distance_l527_527881


namespace proportion_of_triumphal_arch_photographs_l527_527209

-- Define the constants
variables (x y z t : ℕ) -- x = castles, y = triumphal arches, z = waterfalls, t = cathedrals

-- The conditions
axiom half_photographed : t + x + y + z = (3*y + 2*x + 2*z + y) / 2
axiom three_times_cathedrals : ∃ (a : ℕ), t = 3 * a ∧ y = a
axiom same_castles_waterfalls : ∃ (b : ℕ), t + z = x + y
axiom quarter_photographs_castles : x = (t + x + y + z) / 4
axiom second_castle_frequency : t + z = 2 * x
axiom every_triumphal_arch_photographed : ∀ (c : ℕ), y = c ∧ y = c

theorem proportion_of_triumphal_arch_photographs : 
  ∃ (p : ℚ), p = 1 / 4 ∧ p = y / ((t + x + y + z) / 2) :=
sorry

end proportion_of_triumphal_arch_photographs_l527_527209


namespace scientific_notation_of_probe_unit_area_l527_527421

-- Given condition: the area of each probe unit of a certain chip
def probe_unit_area : ℝ := 0.00000164

-- Goal: to prove that 0.00000164 can be expressed in scientific notation as 1.64 * 10^(-6)
theorem scientific_notation_of_probe_unit_area : probe_unit_area = 1.64 * 10^(-6) := 
by
  sorry

end scientific_notation_of_probe_unit_area_l527_527421


namespace tan_is_2_implies_expression_l527_527954

-- Let θ be a real number such that tan(θ) = 2.
variable (θ : ℝ)
hypothesis (h : Real.tan θ = 2)

-- We need to prove that (1 - Real.sin (2 * θ)) / (1 + Real.sin (2 * θ)) = 1 / 9.
theorem tan_is_2_implies_expression (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (1 + Real.sin (2 * θ)) = 1 / 9 :=
sorry

end tan_is_2_implies_expression_l527_527954


namespace collinear_SPT_l527_527682

-- Let ABCD be a convex quadrilateral with the given properties
variables (A B C D I1 I2 E F S T P : Point)
variables (AB AD CB CD : ℝ)

-- The conditions of the problem
variables 
  (hConvex : ConvexQuadrilateral A B C D)
  (hSideLengths : AB + AD = CB + CD)
  (hIncenter1 : Incenter A B C I1)
  (hIncenter2 : Incenter A D C I2)
  (hPerp1 : Perpendicular I1 E A C)
  (hPerp2 : Perpendicular I2 F A C)
  (hIntersect : Intersect AC BD P)
  (hExtend1 : Extend BI1 DE S)
  (hExtend2 : Extend DI2 BF T)

-- The theorem to prove
theorem collinear_SPT 
  (hConvex : ConvexQuadrilateral A B C D)
  (hSideLengths : AB + AD = CB + CD)
  (hIncenter1 : Incenter A B C I1)
  (hIncenter2 : Incenter A D C I2)
  (hPerp1 : Perpendicular I1 E A C)
  (hPerp2 : Perpendicular I2 F A C)
  (hIntersect : Intersect AC BD P)
  (hExtend1 : Extend BI1 DE S)
  (hExtend2 : Extend DI2 BF T) : 
  Collinear S P T := 
sorry

end collinear_SPT_l527_527682


namespace problem_solution_l527_527258

theorem problem_solution
  (x y : ℝ)
  (h1 : (x - y)^2 = 25)
  (h2 : x * y = -10) :
  x^2 + y^2 = 5 := sorry

end problem_solution_l527_527258


namespace time_difference_in_minutes_l527_527141

def speed := 60 -- speed of the car in miles per hour
def distance1 := 360 -- distance of the first trip in miles
def distance2 := 420 -- distance of the second trip in miles
def hours_to_minutes := 60 -- conversion factor from hours to minutes

theorem time_difference_in_minutes :
  ((distance2 / speed) - (distance1 / speed)) * hours_to_minutes = 60 :=
by
  -- proof to be provided
  sorry

end time_difference_in_minutes_l527_527141


namespace consecutive_even_product_l527_527097

theorem consecutive_even_product (x : ℤ) (h : x * (x + 2) = 224) : x * (x + 2) = 224 := by
  sorry

end consecutive_even_product_l527_527097


namespace vertical_asymptotes_count_l527_527204

def f (x : ℝ) := (x - 2) / (x^2 + 8 * x - 9)

theorem vertical_asymptotes_count : 
  {x : ℝ | x^2 + 8 * x - 9 = 0 ∧ x - 2 ≠ 0}.finite ∧ 
  {x : ℝ | x^2 + 8 * x - 9 = 0 ∧ x - 2 ≠ 0}.to_finset.card = 2 :=
by
  sorry

end vertical_asymptotes_count_l527_527204


namespace maximize_revenue_l527_527527

-- Define the conditions
def price (p : ℝ) := p ≤ 30
def toys_sold (p : ℝ) : ℝ := 150 - 4 * p
def revenue (p : ℝ) := p * (toys_sold p)

-- State the theorem to solve the problem
theorem maximize_revenue : ∃ p : ℝ, price p ∧ 
  (∀ q : ℝ, price q → revenue q ≤ revenue p) ∧ p = 18.75 :=
by {
  sorry
}

end maximize_revenue_l527_527527


namespace incorrect_observation_value_l527_527440

def initial_mean (n : ℕ) (sum : ℝ) : ℝ := sum / n

def new_sum (initial_sum : ℝ) (new_mean : ℝ) (n : ℕ) : ℝ := new_mean * n

theorem incorrect_observation_value {n : ℕ} (initial_mean_value : ℝ) (new_mean_value : ℝ) (correct_observation_value : ℝ) (initial_sum new_total_sum error_difference incorrect_value : ℝ) :
  n = 50 →
  initial_mean_value = 36 →
  new_mean_value = 36.5 →
  correct_observation_value = 46 →
  initial_sum = initial_mean_value * n →
  new_total_sum = new_mean_value * n →
  error_difference = new_total_sum - initial_sum →
  incorrect_value = correct_observation_value - error_difference →
  incorrect_value = 21 :=
by
  intros
  sorry

end incorrect_observation_value_l527_527440


namespace solution_intersection_A_B_l527_527609

noncomputable def f (x : ℝ) : ℝ := sorry

def φ (α ξ k : ℝ) : ℝ :=
  (Real.sin α) ^ 2 + (ξ + 1) * Real.cos α - ξ ^ 2 - ξ - k

def A (ξ k : ℝ) : Prop :=
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi → φ α ξ k < 0

def B (ξ k : ℝ) : Prop :=
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi → f (φ α ξ k) > 0

theorem solution_intersection_A_B (k : ℝ) (h_pos : k > 0) :
  { ξ : ℝ | A ξ k } ∩ { ξ : ℝ | B ξ k } = { ξ : ℝ | ξ > 1 ∨ ξ < -5 / 3 } :=
sorry

end solution_intersection_A_B_l527_527609


namespace passenger_cars_count_l527_527512

theorem passenger_cars_count (P C : ℕ) 
    (h₁ : C = (1 / 2 : ℚ) * P + 3) 
    (h₂ : P + C + 2 = 71) : P = 44 :=
begin
  sorry
end

end passenger_cars_count_l527_527512


namespace sum_of_possible_values_of_g_8_l527_527716

noncomputable def f (x : ℝ) : ℝ := x^2 - 7 * x + 18
noncomputable def g (y : ℝ) : ℝ := 2 * (Real.sqrt y) + 3
noncomputable def roots : list ℝ := [2, 5]

theorem sum_of_possible_values_of_g_8 :
  (roots.map (λ x, g (f x))).sum = 20 :=
by
  sorry

end sum_of_possible_values_of_g_8_l527_527716


namespace xy_diff_l527_527653

theorem xy_diff {x y : ℝ} (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end xy_diff_l527_527653


namespace smallest_positive_integer_l527_527805

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l527_527805


namespace range_of_m_l527_527279

noncomputable def distance (m : ℝ) : ℝ := (|m| * Real.sqrt 2 / 2)
theorem range_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 + A.2 + m = 0 ∧ B.1 + B.2 + m = 0) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2) ∧
    (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) + Real.sqrt (B.1 ^ 2 + B.2 ^ 2) ≥ 
     Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧ (distance m < Real.sqrt 2)) ↔ 
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 := 
sorry

end range_of_m_l527_527279


namespace Ramsey_6_l527_527028

theorem Ramsey_6 :
  ∀ (G : SimpleGraph Prop) (V : Fin 6 → Prop),
  (∀ u v, G.adj u v → G.adj v u) →
  (∀ (c : G.EdgeSet → Fin 2), ∃ (T : Finset G.Vertex), T.card = 3 ∧ (∀ (u v : G.Vertex), u ≠ v → u ∈ T → v ∈ T → c ⟨u, v, _⟩ = c ⟨v, u, _⟩)) :=
by
  sorry

end Ramsey_6_l527_527028


namespace tan_of_alpha_cos_of_complement_and_periodic_l527_527953

variable {α : ℝ}

axiom sin_alpha_eq : sin α = 3 / 5
axiom alpha_in_second_quadrant : π / 2 < α ∧ α < π

theorem tan_of_alpha : tan α = -3 / 4 := 
by sorry

theorem cos_of_complement_and_periodic : cos (π / 2 - α) + cos (3 * π + α) = 7 / 5 :=
by sorry

end tan_of_alpha_cos_of_complement_and_periodic_l527_527953


namespace problem_inequality_l527_527613

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 1) : 
  (xy/z + yz/x + zx/y) ≥ sqrt(3) :=
sorry

end problem_inequality_l527_527613


namespace power_function_propositions_l527_527176

theorem power_function_propositions : (∀ n : ℤ, n > 0 → ∀ x : ℝ, x > 0 → (x^n) < x) ∧
  (∀ n : ℤ, n < 0 → ∀ x : ℝ, x > 0 → (x^n) > x) :=
by
  sorry

end power_function_propositions_l527_527176


namespace magnitude_of_angle_between_vectors_norm_of_3a_plus_b_l527_527990

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
theorem magnitude_of_angle_between_vectors (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖3 • a - 2 • b‖ = sqrt 7) :
  inner a b = 1 / 2 :=
sorry

theorem norm_of_3a_plus_b (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖3 • a - 2 • b‖ = sqrt 7) :
  ‖3 • a + b‖ = sqrt 13 :=
sorry

end magnitude_of_angle_between_vectors_norm_of_3a_plus_b_l527_527990


namespace derivative_of_x_log_x_derivative_of_sin_x_over_x_l527_527202

section

theorem derivative_of_x_log_x (x : ℝ) (hx : x ≠ 0) :
  (deriv (λ x : ℝ, x * real.log x)) x = real.log x + 1 :=
sorry

theorem derivative_of_sin_x_over_x (x : ℝ) (hx : x ≠ 0) :
  (deriv (λ x : ℝ, (real.sin x) / x)) x = (x * real.cos x - real.sin x) / (x^2) :=
sorry

end

end derivative_of_x_log_x_derivative_of_sin_x_over_x_l527_527202


namespace range_of_a_l527_527998

def greatest_integer_leq (x : ℝ) : ℤ := ⌊x⌋  -- Define greatest integer less than or equal to x

theorem range_of_a (a : ℝ) : (∃ (x : ℕ), greatest_integer_leq ((x + a) / 3) = 2) → a < 6 :=
by {
    intros H,
    sorry
}

end range_of_a_l527_527998


namespace sum_10_least_values_divisible_by_5_l527_527926

def S_n (n: ℕ) : ℚ := (n-1) * n * (n+1) * (3*n+2) / 24

theorem sum_10_least_values_divisible_by_5 : 
  let values := filter (λ n, S_n n % 5 = 0) (range 20) in
  (values.take 10).sum = 105 :=
sorry

end sum_10_least_values_divisible_by_5_l527_527926


namespace elvin_fixed_monthly_charge_l527_527482

theorem elvin_fixed_monthly_charge
    (F C : ℝ)
    (h1 : F + C = 52)
    (h2 : F + 2C = 76) :
  F = 28 := by
  sorry

end elvin_fixed_monthly_charge_l527_527482


namespace form_of_reasoning_is_incorrect_l527_527444

-- Definitions from the conditions
def some_rational_numbers_are_fractions : Prop := 
  ∃ q : ℚ, ∃ f : ℚ, q = f / 1

def integers_are_rational_numbers : Prop :=
  ∀ z : ℤ, ∃ q : ℚ, q = z

-- The proposition to be proved
theorem form_of_reasoning_is_incorrect (h1 : some_rational_numbers_are_fractions) (h2 : integers_are_rational_numbers) : 
  ¬ ∀ z : ℤ, ∃ f : ℚ, f = z  := sorry

end form_of_reasoning_is_incorrect_l527_527444


namespace sum_of_x_intersections_l527_527627

theorem sum_of_x_intersections (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 + x) = f (3 - x))
  (m : ℕ) (xs : Fin m → ℝ) (ys : Fin m → ℝ)
  (h_intersection : ∀ i : Fin m, f (xs i) = |(xs i)^2 - 4 * (xs i) - 3|) :
  (Finset.univ.sum fun i => xs i) = 2 * m :=
by
  sorry

end sum_of_x_intersections_l527_527627


namespace find_c_l527_527424

theorem find_c (a b c : ℚ) (h1 : ∀ y : ℚ, 1 = a * (3 - 1)^2 + b * (3 - 1) + c) (h2 : ∀ y : ℚ, 4 = a * (1)^2 + b * (1) + c)
  (h3 : ∀ y : ℚ, 1 = a * (y - 1)^2 + 4) : c = 13 / 4 :=
by
  sorry

end find_c_l527_527424


namespace basis_unique_representation_l527_527648

-- Define the vector space and basis
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (e₁ e₂ : V)

-- Assumptions/Conditions
def are_basis (e₁ e₂ : V) : Prop := linear_independent ℝ ![e₁, e₂] ∧ submodule.span ℝ (set.range ![e₁, e₂]) = ⊤

-- Proof statement: We will prove that if e₁ and e₂ are basis, then m • e₁ + n • e₂ = 0 implies m = 0 and n = 0
theorem basis_unique_representation (h : are_basis e₁ e₂) : 
  ∀ (m n : ℝ), (m • e₁ + n • e₂ = (0 : V)) → (m = 0 ∧ n = 0) :=
by
  sorry -- The proof steps should go here

end basis_unique_representation_l527_527648


namespace count_valid_right_triangles_l527_527644

theorem count_valid_right_triangles : 
  (∃ (n : ℕ), n = 7 ∧ 
  ∀ (a b : ℕ), (a^2 = 4 * (b + 1) ∧ b < 50 ∧ (a % 2 = 0)) → 
  (a, b) ∈ (finset.range 50).product (finset.range 100)) := sorry

end count_valid_right_triangles_l527_527644


namespace p_5_l527_527717

-- Define p(x), a monic quintic polynomial
def p (x : ℝ) : ℝ := sorry 
-- p(x) is a monic quintic polynomial
axiom monic_p : ∀ x, p(x) = x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Given conditions
axiom p_1 : p 1 = 2
axiom p_2 : p 2 = 5
axiom p_3 : p 3 = 10
axiom p_4 : p 4 = 17
axiom p_6 : p 6 = 37

-- Objective: Prove that p(5) = 2
theorem p_5 : p 5 = 2 := 
by
  sorry

end p_5_l527_527717


namespace sufficient_but_not_necessary_l527_527253

-- Definitions for the vectors and conditions
variables {m n : EuclideanSpace ℝ (Fin 3)}
variables (h_m : m ≠ 0) (h_n : n ≠ 0) 

-- Statement of the theorem
theorem sufficient_but_not_necessary (h : ∃ (λ : ℝ), λ < 0 ∧ m = λ • n) : 
  m ⬝ n < 0 ∧ ¬ (m ⬝ n < 0 → ∃ (λ : ℝ), λ < 0 ∧ m = λ • n) :=
by sorry

end sufficient_but_not_necessary_l527_527253


namespace frustum_lateral_surface_area_l527_527506

theorem frustum_lateral_surface_area (R r h : ℝ) (hR : R = 8) (hr : r = 2) (hh : h = 6) : 
  (π * (R + r) * sqrt ((R - r)^2 + h^2)) = 60 * sqrt 2 * π :=
by
  rw [hR, hr, hh]
  sorry

end frustum_lateral_surface_area_l527_527506


namespace calc_r_over_s_at_2_l527_527769

def r (x : ℝ) := 3 * (x - 4) * (x - 1)
def s (x : ℝ) := (x - 4) * (x + 3)

theorem calc_r_over_s_at_2 : (r 2) / (s 2) = 3 / 5 := by
  sorry

end calc_r_over_s_at_2_l527_527769


namespace probability_max_area_triangle_ABP_l527_527852

noncomputable def prob_max_area_triangle_ABP (P : ℝ × ℝ) (inside_square : P.1 > 0 ∧ P.1 < 1 ∧ P.2 > 0 ∧ P.2 < 1) : ℝ :=
  if (P.1 ≤ P.2 ∧ P.1 + P.2 ≤ 1) then 1/4 else 0

theorem probability_max_area_triangle_ABP : 
  (∀ P : ℝ × ℝ, (P.1 > 0 ∧ P.1 < 1 ∧ P.2 > 0 ∧ P.2 < 1) → 
    prob_max_area_triangle_ABP P (⟨P.1 > 0 ∧ P.1 < 1, P.2 > 0 ∧ P.2 < 1⟩) = 1/4) :=
sorry

end probability_max_area_triangle_ABP_l527_527852


namespace find_n_l527_527203

noncomputable def tau (n : Nat) : Nat := 
if h : n > 0 then List.length (Nat.divisors n) else 0

theorem find_n (n : Nat) : tau (n * (n + 2) * (n + 4)) ≤ 15
    ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 ∨ n = 9 := 
by
  sorry

end find_n_l527_527203


namespace total_return_at_x_50_optimal_investment_strategy_l527_527494

-- Definitions and conditions from the problem
def total_investment (x y : ℝ) : Prop := x + y = 120
def min_investment (x y : ℝ) : Prop := x ≥ 40 ∧ y ≥ 40
def return_A (x : ℝ) : ℝ := 3 * real.sqrt(2 * x) - 6
def return_B (y : ℝ) : ℝ := (1/4) * y + 2
def total_return (x : ℝ) : ℝ := return_A x + return_B (120 - x)

-- Q1: Total return when x = 50
theorem total_return_at_x_50 : total_return 50 = 43.5 := 
by 
  rw [total_return, return_A, return_B]; 
  -- specific calculations are done in actual proofs
  sorry

-- Q2: Optimal investment strategy
theorem optimal_investment_strategy : 
  (∀ x : ℝ, 40 ≤ x ∧ x ≤ 80 → total_return x ≤ 44) ∧ total_return 72 = 44 := 
by 
  -- actual optimization calculations and proofs
  sorry

end total_return_at_x_50_optimal_investment_strategy_l527_527494


namespace smallest_positive_integer_form_3003_55555_l527_527800

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l527_527800


namespace female_kittens_count_l527_527382

theorem female_kittens_count (initial_cats total_cats male_kittens female_kittens : ℕ)
  (h1 : initial_cats = 2)
  (h2 : total_cats = 7)
  (h3 : male_kittens = 2)
  (h4 : female_kittens = total_cats - initial_cats - male_kittens) :
  female_kittens = 3 :=
by
  sorry

end female_kittens_count_l527_527382


namespace smallest_sum_is_S5_l527_527678

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions of arithmetic sequence sum
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom h1 : a 3 + a 8 > 0
axiom h2 : S 9 < 0

-- Statements relating terms and sums in arithmetic sequence
axiom h3 : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem smallest_sum_is_S5 (seq_a : arithmetic_sequence a) : S 5 ≤ S 1 ∧ S 5 ≤ S 2 ∧ S 5 ≤ S 3 ∧ S 5 ≤ S 4 ∧ S 5 ≤ S 6 ∧ S 5 ≤ S 7 ∧ S 5 ≤ S 8 ∧ S 5 ≤ S 9 :=
by {
    sorry
}

end smallest_sum_is_S5_l527_527678


namespace passenger_cars_count_l527_527513

theorem passenger_cars_count (P C : ℕ) 
    (h₁ : C = (1 / 2 : ℚ) * P + 3) 
    (h₂ : P + C + 2 = 71) : P = 44 :=
begin
  sorry
end

end passenger_cars_count_l527_527513


namespace prob_both_white_prob_at_least_one_white_l527_527314

-- Define the outcomes and the event probabilities
def drawWithReplacement : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

def event_both_white (outcomes : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  outcomes.filter (λ pair => (pair.1 = 2 ∨ pair.1 = 3) ∧ (pair.2 = 2 ∨ pair.2 = 3))

def event_at_least_one_white (outcomes : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  outcomes.filter (λ pair => pair.1 = 2 ∨ pair.1 = 3 ∨ pair.2 = 2 ∨ pair.2 = 3)

theorem prob_both_white : 
  (event_both_white drawWithReplacement).length.toRat / drawWithReplacement.length.toRat = 4 / 9 := by
  sorry

theorem prob_at_least_one_white : 
  (event_at_least_one_white drawWithReplacement).length.toRat / drawWithReplacement.length.toRat = 8 / 9 := by
  sorry

end prob_both_white_prob_at_least_one_white_l527_527314


namespace prob_rains_at_least_one_day_l527_527849

def prob_rain_friday := 0.30
def prob_rain_saturday_if_friday := 0.60
def prob_rain_saturday_if_not_friday := 0.25
def prob_rain_sunday := 0.40

theorem prob_rains_at_least_one_day : 
  let prob_no_rain_friday := 1 - prob_rain_friday
      prob_no_rain_saturday_if_friday := 1 - prob_rain_saturday_if_friday
      prob_no_rain_saturday_if_not_friday := 1 - prob_rain_saturday_if_not_friday
      prob_no_rain_sunday := 1 - prob_rain_sunday
      prob_no_rain_all_three := prob_no_rain_friday * 
                                (prob_no_rain_saturday_if_not_friday * prob_no_rain_friday + 
                                 prob_no_rain_saturday_if_friday * prob_rain_friday) * 
                                prob_no_rain_sunday
      prob_rain_at_least_one := 1 - prob_no_rain_all_three
  in prob_rain_at_least_one = 0.685 := 
by 
  sorry

end prob_rains_at_least_one_day_l527_527849


namespace sum_first_50_even_integers_l527_527445

theorem sum_first_50_even_integers (h : (∑ k in finset.range 50, (2 * k + 1)) = 50^2) :
  (∑ k in finset.range 50, (2 * (k + 1))) = 50^2 + 50 :=
by
  sorry

end sum_first_50_even_integers_l527_527445


namespace area_A_area_A_l527_527001

-- Defining the condition as an acute-angled triangle
structure AcuteTriangle (α : Type) [RealOrComplexRing α] :=
  (A B C : α)
  (angle_A : α)
  (angle_B : α)
  (angle_C : α)
  (angles_sum_180 : angle_A + angle_B + angle_C = 180)
  (acute_A : angle_A < 90)
  (acute_B : angle_B < 90)
  (acute_C : angle_C < 90)

variables {α : Type} [RealOrComplexRing α]
variable (T : AcuteTriangle α)
variables (A' C' : α) (B' : α)

-- Midpoint definition
def midpoint (x y : α) : α := (x + y) / 2

-- B' is the midpoint of AC
def is_midpoint_AC := B' = midpoint T.A T.C

-- Part (a): Area of triangle A'B'C' is at most half the area of triangle ABC
theorem area_A'B'C'_le_half_area_ABC (hB'_mid_AC : is_midpoint_AC T A' C' B') :
  area (triangle A' B' C') ≤ (1/2 : ℝ) * area (triangle T.A T.B T.C) :=
sorry

-- Part (b): Area of triangle A'B'C' is exactly one-quarter the area of triangle ABC if and only if at least one of the points A', C' coincides with the midpoint of the corresponding side
theorem area_A'B'C'_eq_quarter_area_ABC_iff (hB'_mid_AC : is_midpoint_AC T A' C' B') :
  area (triangle A' B' C') = (1/4 : ℝ) * area (triangle T.A T.B T.C) ↔
  (A' = midpoint T.B T.C ∨ C' = midpoint T.A T.B) :=
sorry

end area_A_area_A_l527_527001


namespace center_of_similarity_l527_527073

-- Definitions for the parabolas
def parabola1 (x : ℝ) : ℝ := -x^2 + 6*x - 10
def parabola2 (x : ℝ) : ℝ := (x^2 + 6*x + 13) / 2

-- The main statement to be proved
theorem center_of_similarity :
  ∃ (x y : ℝ), (x, y) = (1, 0) :=
sorry

end center_of_similarity_l527_527073


namespace spherical_to_rectangular_conversion_l527_527201

theorem spherical_to_rectangular_conversion :
  ∃ x y z : ℝ, 
    x = -Real.sqrt 2 ∧ 
    y = 0 ∧ 
    z = Real.sqrt 2 ∧ 
    (∃ rho theta phi : ℝ, 
      rho = 2 ∧
      theta = π ∧
      phi = π/4 ∧
      x = rho * Real.sin phi * Real.cos theta ∧
      y = rho * Real.sin phi * Real.sin theta ∧
      z = rho * Real.cos phi) :=
by
  sorry

end spherical_to_rectangular_conversion_l527_527201


namespace relationship_l527_527947

open Real

variable (b y1 y2 y3 : ℝ)

-- Points A(-3, y1), B(-1, y2), C(2, y3)
def pointA := -3
def pointB := -1
def pointC := 2

-- Hypotheses
def condA : y1 = -pointA^2 - 2*pointA + b := by sorry
def condB : y2 = -pointB^2 - 2*pointB + b := by sorry
def condC : y3 = -pointC^2 - 2*pointC + b := by sorry

theorem relationship (hA : condA) (hB : condB) (hC : condC) : y3 < y1 ∧ y1 < y2 := by
  sorry

end relationship_l527_527947


namespace function_is_odd_l527_527833

def func (x : ℝ) : ℝ := Real.log ((1 + Real.sin x) / Real.cos x)

def is_domain (x : ℝ) : Prop := 
  ∃ k : ℤ, x ∈ (Set.Ioo ((2 * k * Real.pi - Real.pi / 2) : ℝ) (2 * k * Real.pi + Real.pi / 2))

theorem function_is_odd (x : ℝ) (h1 : (1 + Real.sin x) / Real.cos x > 0) (h2 : is_domain x) : 
  func (-x) = -func x := 
sorry

end function_is_odd_l527_527833


namespace shortest_distance_tangent_circle_l527_527084

theorem shortest_distance_tangent_circle :
  ∃ (d : ℝ), d = (4 * real.sqrt 5 / 5) - 1 ∧
  ∀ (P Q : ℝ × ℝ), 
    let l := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} in
    let circle := {q : ℝ × ℝ | (q.1 + 2)^2 + q.2^2 = 1} in
    P ∈ l → Q ∈ circle → sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) >= d := 
begin
  sorry
end

end shortest_distance_tangent_circle_l527_527084


namespace position_independence_point_lies_on_AB_l527_527830

variables (A B X P : Type) [Point A] [Point B] [Point X] [Point P]
variables (λ μ : ℝ) (hλμ : λ + μ = 1)

def vec_XP (X P : Point) : Point := λ • vec_XA + μ • vec_XB

def independent_of_X (X P : Point) : Prop :=
  ∀ (X₁ X₂ : Point), vec_XP X₁ P = vec_XP X₂ P

def lies_on_line (A B P : Point) : Prop :=
  ∃ (t : ℝ), P = (1 - t) • A + t • B

theorem position_independence (A B : Point) (λ μ : ℝ) :
  ∀ (X : Point), independent_of_X A B X P ↔ λ + μ = 1 := sorry

theorem point_lies_on_AB (A B : Point) (λ μ : ℝ) (hλμ : λ + μ = 1) :
  lies_on_line A B P := sorry

end position_independence_point_lies_on_AB_l527_527830


namespace abs_neg_six_l527_527412

theorem abs_neg_six : abs (-6) = 6 :=
by
  -- Proof goes here
  sorry

end abs_neg_six_l527_527412


namespace pipe_length_correct_l527_527165

noncomputable def pipe_length (a : ℝ) (m : ℕ) (n : ℕ) := 
  let y := (168 : ℝ - 80 : ℝ) / (210 + 100) in
  let x := (168 : ℝ - 210 * y) in
  x

theorem pipe_length_correct(x_approx: ℝ) 
  (a : ℝ)
  (m : ℕ)
  (n : ℕ)
  (h_a : a = 0.8)
  (h_m : m = 210)
  (h_n : n = 100)
  : round (pipe_length a m n) = x_approx :=
by {
  unfold pipe_length,
  rw [h_a, h_m, h_n],
  have y_def : y = (168 - 80) / (210 + 100) := rfl,
  rw y_def,
  unfold y,
  simp, -- This step simplifies the expression of y = 8 / 31
  have x_def : x = 168 - (210 * (8 / 31 : ℝ)) := rfl,
  rw x_def,
  unfold x,
  norm_num, -- This simplifies the result to approximately 113.81 in Lean
  norm_cast, -- This converts to natural numbers for rounding correctly
  refl
  sorry -- Simplification to x approximately 108 meters in Lean
}

end pipe_length_correct_l527_527165


namespace beta_value_l527_527647

variable {α β : Real}
open Real

theorem beta_value :
  cos α = 1 / 7 ∧ cos (α + β) = -11 / 14 ∧ 0 < α ∧ α < π / 2 ∧ π / 2 < α + β ∧ α + β < π → 
  β = π / 3 := 
by
  -- Proof would go here
  sorry

end beta_value_l527_527647


namespace probability_diff_grades_l527_527962

section

variable (n10 n11 n12 : ℕ)
variable (total_students selected_students : ℕ)
variable (p_same_grade : ℚ)

-- Conditions
def students_in_grade10 : ℕ := 180
def students_in_grade11 : ℕ := 180
def students_in_grade12 : ℕ := 90
def total_students : ℕ := students_in_grade10 + students_in_grade11 + students_in_grade12
def selected_students : ℕ := 5

-- Number of students selected from each grade
def selected_from_grade10 : ℕ := selected_students * students_in_grade10 / total_students
def selected_from_grade11 : ℕ := selected_students * students_in_grade11 / total_students
def selected_from_grade12 : ℕ := selected_students * students_in_grade12 / total_students

-- Probability that both students are from the same grade
def P_same_grade : ℚ :=
  (selected_from_grade10 * (selected_from_grade10 - 1)) / (selected_students * (selected_students - 1))
  + (selected_from_grade11 * (selected_from_grade11 - 1)) / (selected_students * (selected_students - 1))

-- Probability that the 2 selected students are from different grades
def P_diff_grades : ℚ := 1 - P_same_grade

-- Theorem statement
theorem probability_diff_grades : P_diff_grades = 4 / 5 := by
  -- This is where the proof would go
  sorry

end

end probability_diff_grades_l527_527962


namespace minimum_value_frac_l527_527489

variable {k : Type} [Field k]
variables (ABCD : Quadrilateral k) (s t : k)
variables [Circumscribed ABCD k] [AngleMeasure ABCD k AC B s] [AngleMeasure ABCD k AC D t]

theorem minimum_value_frac :
  s < t → ∃ (min_val : k), min_val = 4 / 5 ∧ 
  (∃ (AB BC CD DA : k) (A B C D : k × k),
    AB = 1 ∧ BC = 1 ∧ CD = sqrt 2 ∧ DA = sqrt 2 ∧
    ∠ ABC = π / 2 ∧ ∠ ADC = π / 2 ∧ [ACB] = s ∧ [ACD] = t) →
  ∀ s t, 
  (4 * s ^ 2 + t ^ 2) / (5 * s * t) = 4 / 5 :=
sorry

end minimum_value_frac_l527_527489


namespace smallest_natural_number_divisible_l527_527226

theorem smallest_natural_number_divisible :
  ∃ n : ℕ, (n^2 + 14 * n + 13) % 68 = 0 ∧ 
          ∀ m : ℕ, (m^2 + 14 * m + 13) % 68 = 0 → 21 ≤ m :=
by 
  sorry

end smallest_natural_number_divisible_l527_527226


namespace mrs_johnson_no_calls_in_2022_l527_527016

theorem mrs_johnson_no_calls_in_2022 
  (calls_every_5_days : ℕ)
  (calls_every_7_days : ℕ)
  (year_days : ℕ) 
  (initial_call_day_2022 : ℕ)
  (h5 : calls_every_5_days = 5)
  (h7 : calls_every_7_days = 7)
  (h_year_days : year_days = 365)
  (h_initial : initial_call_day_2022 = 1) :
  ∃ days_no_calls : ℕ, days_no_calls = 250 := by
  use 250
  sorry

end mrs_johnson_no_calls_in_2022_l527_527016


namespace integral_identity_l527_527365

theorem integral_identity (k a b c : Real) 
  (ha : a > 0) (hk : k > 0) (h : b^2 < a * c) :
  ∫ (x y : Real) in set.univ, (k + a * x^2 + 2 * b * x * y + c * y^2)⁻² = 
  π / (k * Real.sqrt (a * c - b^2)) :=
by
  sorry

end integral_identity_l527_527365


namespace intersect_at_one_point_l527_527183

theorem intersect_at_one_point (a : ℝ) :
  (∃ x : ℝ, ax^2 + 2 * x + 2 = -3 * x - 2) ↔ a = 25 / 16 := by
suffices h : ∀ x : ℝ, ax^2 + 5 * x + 4 = 0 ↔ a = 25 / 16
from ⟨λ ⟨x, hx⟩, ((h x).mp hx), λ ha, ⟨0, (h 0).mpr ha⟩⟩ 
sorry

end intersect_at_one_point_l527_527183


namespace time_away_from_home_l527_527339

-- Definitions for distances and speeds
def distance_to_friend := 80 -- miles
def speed_to_friend := 50 -- mph

def detour1_factor := 1.10
def speed_detour1 := 40 -- mph

def detour2_factor := 1.15
def speed_detour2 := 45 -- mph

def detour3_factor := 1.20
def speed_detour3 := 50 -- mph

def time_at_friend := 0.75 -- hours (45 minutes)

-- Calculate total time away from home
def time_to_friend := distance_to_friend / speed_to_friend
def time_detour1 := (detour1_factor * distance_to_friend) / speed_detour1
def time_detour2 := (detour2_factor * distance_to_friend) / speed_detour2
def time_detour3 := (detour3_factor * distance_to_friend) / speed_detour3

def total_time_away := time_to_friend + time_at_friend + time_detour1 + time_detour2 + time_detour3

theorem time_away_from_home : total_time_away ≈ 8.5144 := by
  sorry

end time_away_from_home_l527_527339


namespace firstQuartile_is_25_l527_527662

open List

-- Declaring the list of numbers
def numbers := [42, 24, 30, 22, 26, 27, 33, 35]

-- Declaring the median of a list
noncomputable def median (l : List ℝ) : ℝ :=
  if length l % 2 = 1 then nthLe l (length l / 2) (by sorry) 
  else (nthLe l (length l / 2) (by sorry) + nthLe l (length l / 2 - 1) (by sorry)) / 2

-- Declaring the first quartile computation
noncomputable def firstQuartile (l : List ℝ) : ℝ :=
  let m := median l
  let lessThanMedian := filter (< m) l
  median lessThanMedian

-- Proof statement
theorem firstQuartile_is_25 : firstQuartile numbers = 25 :=
  sorry

end firstQuartile_is_25_l527_527662


namespace smallest_palindrome_in_bases_2_and_4_l527_527195

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let repr := n.digits base
  repr = repr.reverse

theorem smallest_palindrome_in_bases_2_and_4 (x : ℕ) :
  (x > 15) ∧ is_palindrome x 2 ∧ is_palindrome x 4 → x = 17 :=
by
  sorry

end smallest_palindrome_in_bases_2_and_4_l527_527195


namespace computer_selling_price_l527_527856

variable (C SP : ℝ)

theorem computer_selling_price
  (h1 : 1.5 * C = 2678.57)
  (h2 : SP = 1.4 * C) :
  SP = 2500 :=
by
  sorry

end computer_selling_price_l527_527856


namespace survey_sampling_methods_l527_527549

/-- Given a large number of individuals with significant differences for Survey ①
    and a small number of individuals with minor differences for Survey ②,
    the appropriate sampling methods are ①Stratified sampling and ②Simple random sampling -/
theorem survey_sampling_methods :
  ∀ (num_people_necessary num_people_cruel num_people_indifferent : ℕ)
    (total_students art_students : ℕ),
  num_people_necessary + num_people_cruel + num_people_indifferent > 0 →
  art_students > 0 →
  -- Condition for Survey ①
  let survey_1_significant_differences := num_people_necessary > 0 ∧ num_people_cruel > 0 ∧ num_people_indifferent > 0 in
  -- Condition for Survey ②
  let survey_2_minor_differences := total_students = art_students in
  survey_1_significant_differences → survey_2_minor_differences →
  -- Conclusion
  (stratified_sampling_for_1 : true) ∧ (simple_random_sampling_for_2 : true) :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end survey_sampling_methods_l527_527549


namespace car_mileage_45_mph_l527_527888

-- Define the data of the problem
def mpg_at_60 (mpg_60 : ℕ) (miles : ℕ) (gallons : ℕ) : Prop :=
  mpg_60 = miles / gallons

def mpg_at_45 (mpg_45 mpg_60 : ℕ) : Prop :=
  mpg_45 = mpg_60 / 0.8

-- Given conditions
theorem car_mileage_45_mph (miles_60mph : ℕ) (gallons_60mph : ℕ) :
  (mpg_at_60 mpg_60 miles_60mph gallons_60mph) → 
  (miles_60mph = 400) → (gallons_60mph = 10) → 
  (mpg_at_45 50 mpg_60) :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  have h4 : mpg_60 = 40 := by sorry -- From miles_60mph = 400 and gallons_60mph = 10, mpg_60 is calculated as 40
  rw mpg_at_45 50 40
  sorry -- Continuing from mpg_60 = 40, 50 = 40 / 0.8 will be validated.

end car_mileage_45_mph_l527_527888


namespace sum_eight_smallest_multiples_of_12_l527_527111

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end sum_eight_smallest_multiples_of_12_l527_527111


namespace solve_sqrt_eq_l527_527575

theorem solve_sqrt_eq (x : ℝ) (hx : x ≥ 4/9) :
  (sqrt (9 * x - 4) + 15 / sqrt (9 * x - 4) = 8) ↔ (x = 29/9 ∨ x = 13/9) :=
by
  sorry

end solve_sqrt_eq_l527_527575


namespace range_of_f_A_value_of_b_l527_527688

-- Definitions and conditions for the first part of the question
variables (A : ℝ)
def vec_m := (1, 1)
def vec_n (A : ℝ) := (-Real.cos A, Real.sin A)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def f (A : ℝ) := dot_product (vec_m) (vec_n A)

-- Definitions and conditions for the second part of the question
variables (B C : ℝ) (c : ℝ)
def angle_between_vectors (θ : ℝ) : Prop := dot_product vec_m vec_n = Real.sqrt 2 * Real.cos θ
def sine_law (b c : ℝ) (B C : ℝ) : Prop := (b / Real.sin B) = (c / Real.sin C)

-- Theorem to state the proof problems
theorem range_of_f_A : ∀ A, -Real.sqrt 2 / 2 < f A ∧ f A ≤ Real.sqrt 2 := sorry
theorem value_of_b : angle_between_vectors (π/3) → C = π/3 → c = Real.sqrt 6 → B = π/4 → sine_law b c B C → b = 2 := sorry

end range_of_f_A_value_of_b_l527_527688


namespace moment_of_inertia_z_axis_l527_527162

noncomputable def moment_of_inertia (k : ℝ) : ℝ :=
  ∫ x in 0..2, ∫ y in 0..1, ∫ z in 0..real.sqrt (6 * x), k * z * (x^2 + y^2) * z

theorem moment_of_inertia_z_axis (k : ℝ) : moment_of_inertia k = 14 * k :=
by
  sorry

end moment_of_inertia_z_axis_l527_527162


namespace prove_a_minus_c_l527_527661

-- Define the given conditions as hypotheses
def condition1 (a b d : ℝ) : Prop := (a + d + b + d) / 2 = 80
def condition2 (b c d : ℝ) : Prop := (b + d + c + d) / 2 = 180

-- State the theorem to be proven
theorem prove_a_minus_c (a b c d : ℝ) (h1 : condition1 a b d) (h2 : condition2 b c d) : a - c = -200 :=
by
  sorry

end prove_a_minus_c_l527_527661


namespace sequence_modulo_1000_remainder_l527_527161

theorem sequence_modulo_1000_remainder :
  let a : ℕ → ℤ :=
    λ n, if n = 0 then 0
    else if n = 1 ∨ n = 2 ∨ n = 3 then 1
    else a (n - 1) + a (n - 2) + a (n - 3)
  in
  a 28 = 7020407 →
  a 29 = 13003421 →
  a 30 = 23927447 →
  (∑ k in finset.range 28, a (k + 1)) % 1000 = 925 :=
by
  assume h1 : a 28 = 7020407,
  assume h2 : a 29 = 13003421,
  assume h3 : a 30 = 23927447,
  sorry

end sequence_modulo_1000_remainder_l527_527161


namespace scientific_notation_l527_527418

-- Define the condition: the area of each probe unit of a certain chip is 0.00000164 cm^2
def area_probe_unit : ℝ := 0.00000164

-- The mathematical problem to express 0.00000164 in scientific notation
theorem scientific_notation :
  area_probe_unit = 1.64 * 10 ^ (-6) := 
  by
    sorry

end scientific_notation_l527_527418


namespace find_245th_digit_in_decimal_rep_of_13_div_17_l527_527473

-- Definition of the repeating sequence for the fractional division
def repeating_sequence_13_div_17 : List Char := ['7', '6', '4', '7', '0', '5', '8', '8', '2', '3', '5', '2', '9', '4', '1', '1']

-- Period of the repeating sequence
def period : ℕ := 16

-- Function to find the n-th digit in a repeating sequence
def nth_digit_in_repeating_sequence (seq : List Char) (period : ℕ) (n : ℕ) : Char :=
  seq.get! ((n - 1) % period)

-- Hypothesis: The repeating sequence of 13/17 and its period
axiom repeating_sequence_period : repeating_sequence_13_div_17.length = period

-- The theorem to prove
theorem find_245th_digit_in_decimal_rep_of_13_div_17 : nth_digit_in_repeating_sequence repeating_sequence_13_div_17 period 245 = '7' := 
  by
  sorry

end find_245th_digit_in_decimal_rep_of_13_div_17_l527_527473


namespace problem_solution_l527_527623

variables {V : Type*} [inner_product_space ℝ V]

def vector_AB : V := sorry
def vector_AC : V := sorry
def vector_AM (λ μ : ℝ) : V := λ • vector_AB + μ • vector_AC
def vector_BC : V := vector_AC - vector_AB

lemma angle_AB_AC_is_90 (h : inner_product_space.angle vector_AB vector_AC = 0) : 
inner_product_space.orthogonal vector_AB vector_AC :=
begin
  sorry
end

lemma norm_vector_AB : ∥vector_AB∥ = 2 := sorry
lemma norm_vector_AC : ∥vector_AC∥ = 1 := sorry

noncomputable
def given_AM_BC_orthogonal (λ μ : ℝ) : Prop := inner_product_space.inner (vector_AM λ μ) vector_BC = 0

theorem problem_solution (λ μ : ℝ) 
  (h_angle : inner_product_space.orthogonal vector_AB vector_AC)
  (h_norm_AB : ∥vector_AB∥ = 2)
  (h_norm_AC : ∥vector_AC∥ = 1)
  (h_orth_AM_BC : given_AM_BC_orthogonal λ μ) :
  λ / μ = 4 :=
begin
  sorry
end

end problem_solution_l527_527623


namespace junk_mail_per_red_or_white_house_l527_527153

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l527_527153


namespace coefficient_x4_in_expansion_l527_527329

open Real

noncomputable theory
def binomial_coefficient (n k : ℕ) : ℤ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x4_in_expansion :
  let expr := (2 + (Real.sqrt x : ℂ) - (1 / (x ^ 2016 : ℂ))) ^ 10
  (∀ x, x ≠ 0 ∧ (∃ k, k = 8 ∧ (binomial_coefficient 10 k) * (2 ^ (10 - k)) = 180)) :=
by
  sorry

end coefficient_x4_in_expansion_l527_527329


namespace total_oranges_l527_527385

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l527_527385


namespace sum_of_squares_ge_two_ab_l527_527002

theorem sum_of_squares_ge_two_ab (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := 
  sorry

end sum_of_squares_ge_two_ab_l527_527002


namespace probability_different_last_digit_l527_527051

open BigOperators

def count_ways_different_last_digit : ℕ :=
  10 * 9 * 8 * 7 * 6

def total_combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_different_last_digit :
  (count_ways_different_last_digit : ℚ) / (total_combinations 90 5 : ℚ) = 252 / 366244 :=
by
  sorry

end probability_different_last_digit_l527_527051


namespace simplify_fraction_l527_527923

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end simplify_fraction_l527_527923


namespace largest_possible_percent_error_l527_527705

theorem largest_possible_percent_error 
  (actual_radius : ℝ) (measurement_error_percent : ℝ) 
  (h_radius : actual_radius = 10) (h_error : measurement_error_percent = 0.2) :
  ∃(percent_error : ℝ), percent_error = 44 :=
by
  exists 44
  sorry

end largest_possible_percent_error_l527_527705


namespace Sam_balloon_count_l527_527930

theorem Sam_balloon_count:
  ∀ (F M S : ℕ), F = 5 → M = 7 → (F + M + S = 18) → S = 6 :=
by
  intros F M S hF hM hTotal
  rw [hF, hM] at hTotal
  linarith

end Sam_balloon_count_l527_527930


namespace ratio_of_perimeters_l527_527096

def triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : ℕ := a + b + c

def sides_PQR := (6, 8, 10)
def sides_STU := (9, 12, 15)

theorem ratio_of_perimeters :
  let PQR_perimeter := perimeter 6 8 10 in
  let STU_perimeter := perimeter 9 12 15 in
  PQR_perimeter = 24 ∧ STU_perimeter = 36 ∧
  (PQR_perimeter : ℚ) / STU_perimeter = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l527_527096


namespace solution_set_of_derivative_pos_l527_527652

theorem solution_set_of_derivative_pos (x : ℝ) (h₁ : x > 0)
  (h₂ : ∀ x, f(x) = x^2 - 2x - 4 * real.log x) :
    ∀ y, (2 + infer_instance )? sorry  :=
begin
  sorry
end

end solution_set_of_derivative_pos_l527_527652


namespace minimum_minutes_for_planB_cheaper_l527_527533

-- Define the costs for Plan A and Plan B as functions of minutes
def planACost (x : Nat) : Nat := 1500 + 12 * x
def planBCost (x : Nat) : Nat := 3000 + 6 * x

-- Statement to prove
theorem minimum_minutes_for_planB_cheaper : 
  ∃ x : Nat, (planBCost x < planACost x) ∧ ∀ y : Nat, y < x → planBCost y ≥ planACost y :=
by
  sorry

end minimum_minutes_for_planB_cheaper_l527_527533


namespace cylinder_height_and_diameter_l527_527083

-- Define the given conditions and the theorem to be proved.
theorem cylinder_height_and_diameter:
  ∀ (r_sphere : ℝ) (h d : ℝ),
  r_sphere = 7 →
  4 * real.pi * r_sphere^2 = 2 * real.pi * (d / 2) * h →
  h = d →
  h = 14 ∧ d = 14 :=
by
  intros r_sphere h d r_sphere_eq A_eq h_eq_d
  sorry

end cylinder_height_and_diameter_l527_527083


namespace carrie_spent_money_l527_527828

variable (cost_per_tshirt : ℝ) (num_tshirts : ℕ)

theorem carrie_spent_money (h1 : cost_per_tshirt = 9.95) (h2 : num_tshirts = 20) :
  cost_per_tshirt * num_tshirts = 199 := by
  sorry

end carrie_spent_money_l527_527828


namespace fraction_upgraded_sensors_l527_527821

theorem fraction_upgraded_sensors (N U : ℕ) (h1 : (N : ℚ) = (1 : ℚ) / 3 * (U : ℚ)) :
  let total_sensors := (24 : ℕ) * N + U in
  (U : ℚ) / total_sensors = (1 : ℚ) / 9 :=
by sorry

end fraction_upgraded_sensors_l527_527821


namespace cody_paid_17_l527_527889

-- Definitions for the conditions
def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def final_price_after_discount : ℝ := initial_cost * (1 + tax_rate) - discount
def cody_payment : ℝ := 17

-- The proof statement
theorem cody_paid_17 :
  cody_payment = (final_price_after_discount / 2) :=
by
  -- Proof steps, which we omit by using sorry
  sorry

end cody_paid_17_l527_527889


namespace heather_aprons_l527_527291

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l527_527291


namespace eventC_is_certain_event_l527_527118

def eventA : Prop :=
  ∀ (t : Type), ∃ (ticket : t), (∃ (n : ℕ), Even n)

def eventB : Prop :=
  ∀ (tv : Type), (∃ (channel : ℕ), channel = 1 → broadcast_weather)

def eventC : Prop :=
  ∀ (bag : Type), (∃ (balls : bag), (∀ (ball : balls), (color ball = red)) → (drawn_ball color = red))

def eventD : Prop :=
  ∀ (crossroad : Type), (∃ (traffic_lights : crossroad), (∃ (signal : traffic_lights), signal = red))

def certain_event (e : Prop) : Prop :=
  ∀ h : e, True

theorem eventC_is_certain_event : certain_event eventC :=
  sorry

end eventC_is_certain_event_l527_527118


namespace Alfred_sells_scooter_l527_527173

theorem Alfred_sells_scooter :
  let purchase_price := 4700
  let repair_costs := 600
  let gain_percent := 9.433962264150944 / 100
  let total_cost := purchase_price + repair_costs
  let gain := gain_percent * total_cost
  let selling_price := total_cost + gain
  selling_price = 5800 :=
by 
  -- Definitions
  let purchase_price := 4700
  let repair_costs := 600
  let gain_percent := 9.433962264150944 / 100
  let total_cost := purchase_price + repair_costs
  let gain := gain_percent * total_cost
  let selling_price := total_cost + gain
  -- Sorry to indicate missing proof
  sorry

end Alfred_sells_scooter_l527_527173


namespace range_of_function_l527_527435

def f (x : ℕ) : ℤ := x^2 - 2 * x

theorem range_of_function :
  (set.range f) ∩ {0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_function_l527_527435


namespace mass_of_curve_static_moments_center_of_gravity_moments_of_inertia_l527_527587

section CycloidCurve
variables {a : ℝ} {ρ : ℝ := 1}

def x (t : ℝ) := a * (t - sin t)
def y (t : ℝ) := a * (1 - cos t)
def dl (t : ℝ) := 2 * a * sin (t / 2)

-- 1. Prove that the mass of the curve is 4a
theorem mass_of_curve : (∫ t in 0..π, dl t) = 4 * a := 
sorry

-- 2. Prove the static moments M_x and M_y
theorem static_moments :
  let M_x := ∫ t in 0..π, y t * dl t,
  let M_y := ∫ t in 0..π, x t * dl t
  in M_x = (2 * a^2 * ∫ t in 0..π, (1 - cos t) * sin (t / 2)) ∧
     M_y = (2 * a^2 * ∫ t in 0..π, (t - sin t) * sin (t / 2)) :=
sorry

-- 3. Prove the coordinates of the center of gravity
theorem center_of_gravity :
  let M_x := ∫ t in 0..π, y t * dl t,
  let M_y := ∫ t in 0..π, x t * dl t,
  let m := 4 * a 
  in (M_y / m = ∫ t in 0..π, x t * dl t / (4 * a)) ∧
     (M_x / m = ∫ t in 0..π, y t * dl t / (4 * a)) :=
sorry

-- 4. Prove moments of inertia relative to the coordinate axes Ox and Oy
theorem moments_of_inertia : 
    -- Add theorem proving moments of inertia I_x and I_y
    sorry :=
sorry

end CycloidCurve

end mass_of_curve_static_moments_center_of_gravity_moments_of_inertia_l527_527587


namespace dot_product_expression_calc_magnitude_calc_l527_527956

variables {α : Type*} [inner_product_space ℝ α]

-- Given conditions as variables and constants
variables (a b : α)
def norm_a : ℝ := ∥a∥ = 2
def norm_b : ℝ := ∥b∥ = 3
def angle_ab : ℝ := real.inner (a, b) = ∥a∥ * ∥b∥ * real.cos (2 * real.pi / 3)

-- Define the proof problems (1), (2), and (3)
theorem dot_product : ⟪a, b⟫ = -3 := sorry

theorem expression_calc : ⟪2 • a - b, a + 3 • b⟫ = -34 := sorry

theorem magnitude_calc : ∥a + b∥ = real.sqrt 7 := sorry

end dot_product_expression_calc_magnitude_calc_l527_527956


namespace transformation_identity_l527_527442

theorem transformation_identity (a b : ℝ) 
    (h1 : ∃ a b : ℝ, ∀ x y : ℝ, (y, -x) = (-7, 3) → (x, y) = (3, 7))
    (h2 : ∃ a b : ℝ, ∀ c d : ℝ, (d, c) = (3, -7) → (c, d) = (-7, 3)) :
    b - a = 4 :=
by
    sorry

end transformation_identity_l527_527442


namespace oblique_projection_l527_527116

def oblique_projection_area_relation (original_area perspective_area : ℝ) : Prop :=
  original_area = 2 * sqrt 2 * perspective_area

theorem oblique_projection (original_area perspective_area : ℝ)
  (base_unchanged : true)
  (height_halved : true)
  (area_relation : oblique_projection_area_relation original_area perspective_area) :
  original_area = 2 * sqrt 2 * perspective_area :=
sorry

end oblique_projection_l527_527116


namespace intersection_A_complement_B_l527_527260

-- Definition of real numbers
def R := ℝ

-- Definitions of sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x^2 - x - 2 > 0}

-- Definition of the complement of B in R
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- The final statement we need to prove
theorem intersection_A_complement_B :
  A ∩ B_complement = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_complement_B_l527_527260


namespace three_element_subset_sum_divisible_by_4_l527_527993

/-- Given the set {1, 2, 3, ..., 19}, the number of 3-element subsets
    with the sum of elements divisible by 4 is 244. -/
theorem three_element_subset_sum_divisible_by_4 :
  (finset.univ.filter (λ s : finset ℕ, s.card = 3 ∧ (s.sum id) % 4 = 0)).card = 244 :=
sorry

end three_element_subset_sum_divisible_by_4_l527_527993


namespace combined_earnings_l527_527883

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l527_527883


namespace curve_is_non_square_rhombus_l527_527313

-- Define the problem with the conditions that a and b are unequal positive numbers
variable {a b : ℝ} (h_cond : 0 < a ∧ 0 < b ∧ a ≠ b)

-- State the goal to prove given the conditions
theorem curve_is_non_square_rhombus (h : ∀ x y : ℝ, 
  abs (x + y) / (2 * a) + abs (x - y) / (2 * b) = 1) : 
  curve_is_non_square_rhombus :=
begin
  sorry
end

end curve_is_non_square_rhombus_l527_527313


namespace sequence_geometric_progression_l527_527778

theorem sequence_geometric_progression (p : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = p * a n + 2^n)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 1)^2 = a n * a (n + 2)): 
  ∃ p : ℝ, ∀ n : ℕ, a n = 2^n :=
by
  sorry

end sequence_geometric_progression_l527_527778


namespace abs_value_of_neg3_l527_527301

-- Define the variables and conditions
variable (x : ℝ)
variable (h : x = -3)

-- State the theorem
theorem abs_value_of_neg3 : |x| = 3 :=
by
  have h1 := h 
  rw h1
  rw abs_neg 
  sorry

end abs_value_of_neg3_l527_527301


namespace isosceles_triangle_DF_l527_527676

theorem isosceles_triangle_DF (D E F J : Type) [point D] [point E] [point F] [point J]
  (h_iso : D ≠ E ∧ E ≠ F ∧ D ≠ F ∧ ∀ P : Type, (P = D ∨ P = E ∨ P = F ↔ p D ∨ p E ∨ p F))
  (h_DE_eq_EF : dist D E = 5 ∧ dist E F = 5)
  (h_alt : (∀ G : Type, alt D G))
  (h_div : dist E J = 4 * dist J F) :
  dist D F = real.sqrt 10 := by
  sorry

end isosceles_triangle_DF_l527_527676


namespace youngsville_population_l527_527816

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end youngsville_population_l527_527816


namespace polar_to_cartesian_l527_527896

theorem polar_to_cartesian :
  ∀ (r θ : ℝ), r = 4 → θ = 5 * Real.pi / 6 → (let x := r * Real.cos θ in 
                                                 let y := r * Real.sin θ in 
                                                 (x, y) = (-2 * Real.sqrt 3, 2)) := by
  intros r θ hr hθ
  sorry

end polar_to_cartesian_l527_527896


namespace novels_at_both_ends_l527_527310

noncomputable def arrangements_where_novels_at_both_ends (novels : ℕ) (other_books : ℕ) : ℕ :=
  let total_books := novels + other_books
  if h : novels = 2 ∧ other_books = 3 then
    (factorial 3) * 2
  else
    0

theorem novels_at_both_ends : arrangements_where_novels_at_both_ends 2 3 = 12 := 
  by
  sorry

end novels_at_both_ends_l527_527310


namespace length_of_pipe_l527_527168

-- Define the conditions
def gavrila_step_length : ℝ := 0.8
def steps_in_direction_of_tractor : ℝ := 210
def steps_against_direction_of_tractor : ℝ := 100

-- Define what we need to prove
theorem length_of_pipe :
  let y := 8 / 31 in
  let x := 168 - 210 * y in
  let x_approx := 113.81 in
  (x ≈ 113.81) ∨ (108 ≤ floor x ∧ floor x ≤ 109) :=
by
  let y := 8 / 31
  let x := 168 - 210 * y
  let x_approx := 113.81
  /-
  We have:
  168 - 210y, where y = 8 / 31.
  This gives us approximately 113.81 meters
  
  Since we want the length rounded to the nearest whole number:
  floor(x) should be approximately 108 (since 113.81 rounded is 114).
  We include the flooring condition just to ensure thoroughness and rounding behavior.
  -/
  have step_1 : y = 8 / 31 := rfl
  have step_2 : x = 168 - 210 * y := rfl
  have step_3 : x ≈ 113.81 := by
    norm_num
    linarith

  have step_4 := floor.to_nearby x
  have step_5 : (108 ≤ floor x ∧ floor x ≤ 109) := by sorry
  exact Or.inr step_5

end length_of_pipe_l527_527168


namespace divisor_count_of_45_l527_527293

theorem divisor_count_of_45 : 
  ∃ (n : ℤ), n = 12 ∧ ∀ d : ℤ, d ∣ 45 → (d > 0 ∨ d < 0) := sorry

end divisor_count_of_45_l527_527293


namespace robin_packages_of_gum_l527_527405

theorem robin_packages_of_gum (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 15) (h2 : total_pieces = 135) : 
  total_pieces / pieces_per_package = 9 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end robin_packages_of_gum_l527_527405


namespace first_three_decimal_digits_of_x_l527_527106

noncomputable def x : ℝ := (10^100 + 1) ^ (5 / 3)

theorem first_three_decimal_digits_of_x : (floor ((x - floor x) * 1000)) = 666 := 
sorry

end first_three_decimal_digits_of_x_l527_527106


namespace total_amount_spent_l527_527588

-- Define the variables B and D representing the amounts Ben and David spent.
variables (B D : ℝ)

-- Define the conditions based on the given problem.
def conditions : Prop :=
  (D = 0.60 * B) ∧ (B = D + 14)

-- The main theorem stating the total amount spent by Ben and David is 56.
theorem total_amount_spent (h : conditions B D) : B + D = 56 :=
sorry  -- Proof omitted.

end total_amount_spent_l527_527588


namespace total_amount_spent_l527_527542

def cost_of_soft_drink : ℕ := 2
def cost_per_candy_bar : ℕ := 5
def number_of_candy_bars : ℕ := 5

theorem total_amount_spent : cost_of_soft_drink + cost_per_candy_bar * number_of_candy_bars = 27 := by
  sorry

end total_amount_spent_l527_527542


namespace find_two_angles_of_scalene_obtuse_triangle_l527_527160

def is_scalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_obtuse (a : ℝ) : Prop := a > 90
def is_triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem find_two_angles_of_scalene_obtuse_triangle
  (a b c : ℝ)
  (ha : is_obtuse a) (h_scalene : is_scalene a b c) 
  (h_sum : is_triangle a b c) 
  (ha_val : a = 108)
  (h_half : b = 2 * c) :
  b = 48 ∧ c = 24 :=
by
  sorry

end find_two_angles_of_scalene_obtuse_triangle_l527_527160


namespace sandra_oranges_l527_527567

theorem sandra_oranges (S E B: ℕ) (h1: E = 7 * S) (h2: E = 252) (h3: B = 12) : S / B = 3 := by
  sorry

end sandra_oranges_l527_527567


namespace problem1_problem2_l527_527598

-- Define the function f(theta)
def f (theta : Real) : Real :=
  (sin (theta - 5 * π) * cos (-π / 2 - theta) * cos (8 * π - theta)) /
  (sin (theta - 3 * π / 2) * sin (-theta - 4 * π))

-- Prove that f(theta) = -sin(theta)
theorem problem1 (theta : Real) : f(theta) = -sin(theta) := sorry

-- Prove that f(4/3 * π) = sqrt(3)/2
theorem problem2 : f(4/3 * π) = sqrt(3) / 2 := sorry

end problem1_problem2_l527_527598


namespace base_case_proof_l527_527100

noncomputable def base_case_inequality := 1 + (1 / (2 ^ 3)) < 2 - (1 / 2)

theorem base_case_proof : base_case_inequality := by
  -- The proof would go here
  sorry

end base_case_proof_l527_527100


namespace f_odd_and_period_pi_l527_527971

def f (x : Real) : Real := Real.sin (2 * x - Real.pi)

theorem f_odd_and_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi) :=
by
  sorry

end f_odd_and_period_pi_l527_527971


namespace tan_add_eq_l527_527297

noncomputable def tan (x : ℝ) := sin x / cos x
noncomputable def cot (x : ℝ) := cos x / sin x

theorem tan_add_eq
  (x y : ℝ)
  (h₁ : tan x + tan y = 30)
  (h₂ : cot x + cot y = 40) :
  tan (x + y) = 120 :=
by
  sorry

end tan_add_eq_l527_527297


namespace board_divisible_into_hexominos_l527_527838

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end board_divisible_into_hexominos_l527_527838


namespace number_of_clerks_l527_527060

theorem number_of_clerks 
  (num_officers : ℕ) 
  (num_clerks : ℕ) 
  (avg_salary_staff : ℕ) 
  (avg_salary_officers : ℕ) 
  (avg_salary_clerks : ℕ)
  (h1 : avg_salary_staff = 90)
  (h2 : avg_salary_officers = 600)
  (h3 : avg_salary_clerks = 84)
  (h4 : num_officers = 2)
  : num_clerks = 170 :=
sorry

end number_of_clerks_l527_527060


namespace five_student_committee_l527_527932

theorem five_student_committee : ∀ (students : Finset ℕ) (alice bob : ℕ), 
  alice ∈ students → bob ∈ students → students.card = 8 → ∃ (committees : Finset (Finset ℕ)),
  (∀ committee ∈ committees, alice ∈ committee ∧ bob ∈ committee) ∧
  ∀ committee ∈ committees, committee.card = 5 ∧ committees.card = 20 :=
by
  sorry

end five_student_committee_l527_527932


namespace num_students_play_cricket_l527_527737

theorem num_students_play_cricket 
  (total_students : ℕ)
  (play_football : ℕ)
  (play_both : ℕ)
  (play_neither : ℕ)
  (C : ℕ) :
  total_students = 450 →
  play_football = 325 →
  play_both = 100 →
  play_neither = 50 →
  (total_students - play_neither = play_football + C - play_both) →
  C = 175 := by
  intros h0 h1 h2 h3 h4
  sorry

end num_students_play_cricket_l527_527737


namespace angle_BMC_eq_10_l527_527677

/-- Given conditions: -/
variables (A B C D M : Type)
variables [inner_product_space ℝ (Type)]
variables (AB CD AD BC : Type)
variables (AB_eq_CD : AB = CD)
variables (angle_ABC : inner_product_space.angle A B C = real.pi / 2)
variables (angle_BCD : inner_product_space.angle B C D = 10 * real.pi / 18) -- 100 degrees in radians
variables (M : Type → Type)
variables (bisector_AD : function.bijective (M AD))
variables (bisector_BC : function.bijective (M BC))

/-- The goal: -/
theorem angle_BMC_eq_10 (A B C D : Type)
  (AB_eq_CD : AB = CD)
  (angle_ABC : inner_product_space.angle A B C = real.pi / 2)
  (angle_BCD : inner_product_space.angle B C D = 10 * real.pi / 18) -- 100 degrees in radians
  (bisector_AD : function.bijective (M AD))
  (bisector_BC : function.bijective (M BC)) :
  inner_product_space.angle B M C = real.pi / 18 := -- 10 degrees in radians
sorry

end angle_BMC_eq_10_l527_527677


namespace max_stamps_l527_527308

-- Definitions based on conditions
def price_of_stamp := 28 -- in cents
def total_money := 3600 -- in cents

-- The theorem statement
theorem max_stamps (price_of_stamp total_money : ℕ) : (total_money / price_of_stamp) = 128 := by
  sorry

end max_stamps_l527_527308


namespace roots_reciprocal_sum_l527_527065

-- Define the roots and their properties
variables {a b c : ℝ}

-- State the conditions as Lean hypotheses
def conditions : Prop :=
  a + b + c = 12 ∧
  ab + bc + ca = 14 ∧
  abc = -3

-- State the theorem we want to prove
theorem roots_reciprocal_sum (h : conditions) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 :=
by
  sorry

end roots_reciprocal_sum_l527_527065


namespace sum_of_cubes_of_consecutive_integers_l527_527081

theorem sum_of_cubes_of_consecutive_integers :
  ∃ (a b c d : ℕ), a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a^2 + b^2 + c^2 + d^2 = 9340) ∧ (a^3 + b^3 + c^3 + d^3 = 457064) :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l527_527081


namespace ratio_of_ages_l527_527035

-- Necessary conditions as definitions in Lean
def combined_age (S D : ℕ) : Prop := S + D = 54
def sam_is_18 (S : ℕ) : Prop := S = 18

-- The statement that we need to prove
theorem ratio_of_ages (S D : ℕ) (h1 : combined_age S D) (h2 : sam_is_18 S) : S / D = 1 / 2 := by
  sorry

end ratio_of_ages_l527_527035


namespace find_n_l527_527918

theorem find_n (n : ℤ) (h1 : -90 ≤ n ∧ n ≤ 90) : 
  sin (n * Real.pi / 180) = sin (720 * Real.pi / 180) ↔ n = 0 := 
by
  sorry

end find_n_l527_527918


namespace max_OM_ON_l527_527393

variables {a b : ℝ} (AC BC AB : ℝ)
variables (O : ℝ) -- center of the square on AB
variables (M N : ℝ) -- midpoints of AC and BC, respectively
variables (γ : ℝ) -- angle ACB

-- Given conditions
axiom AC_length : AC = a
axiom BC_length : BC = b
axiom OM_length : M = 1/2 * O
axiom ON_length : N = 1/2 * O
axiom ABC_square : Square_on_AB O -- This represents the setup of the square on AB

-- Prove statement
theorem max_OM_ON : (OM + ON) ≤ ((1 + Real.sqrt 2) / 2) * (a + b) :=
sorry

end max_OM_ON_l527_527393


namespace proof_problem_l527_527946

open Real EuclideanSpace

variables {V : Type*} [inner_product_space ℝ V]

theorem proof_problem 
  (a b : V) 
  (ha_nonzero : a ≠ 0) 
  (hb_nonzero : b ≠ 0) 
  (ha_norm : ‖a‖ = 1) 
  (h_dot : inner (a - b) (a + b) = 3 / 4)
  (h_ab_dot : inner a b = -1 / 4) :
  (‖b‖ = 1 / 2) ∧
  let theta := real.arccos((inner a (a + 2 • b)) / (‖a‖ * ‖a + 2 • b‖))
  in theta = real.pi / 3 :=
by
  sorry

end proof_problem_l527_527946


namespace total_kids_in_Lawrence_county_l527_527564

-- Define the given conditions
def kids_who_went_to_camp : ℕ := 610769
def kids_who_stayed_home : ℕ := 590796
def kids_from_outside : ℕ := 22 -- This is given but not used in the proof.

-- Lean statement representing the proof problem
theorem total_kids_in_Lawrence_county :
  let n := kids_who_went_to_camp + kids_who_stayed_home in
  n = 1201565 :=
by
  sorry

end total_kids_in_Lawrence_county_l527_527564


namespace current_population_l527_527819

def initial_population : ℕ := 684
def growth_rate : ℝ := 0.25
def moving_away_rate : ℝ := 0.40

theorem current_population (P0 : ℕ) (g : ℝ) (m : ℝ) : 
  P0 = initial_population → 
  g = growth_rate → 
  m = moving_away_rate → 
  (P0 + (P0 * g).to_nat - ((P0 + (P0 * g).to_nat) * m).to_nat) = 513 := 
by
  intros hP0 hg hm
  sorry

end current_population_l527_527819


namespace false_propositions_l527_527599

variables (m n : Line) (α : Plane)

def proposition_1 := m ∥ α ∧ n ∥ α → m ∥ n
def proposition_2 := m ⊥ α ∧ n ⊥ α → m ∥ n
def proposition_3 := m ∥ α ∧ n ⊥ α → m ⊥ n
def proposition_4 := m ⊥ α ∧ m ⊥ n → n ∥ α

theorem false_propositions : 
  ¬proposition_1 m n α ∧ ¬proposition_3 m n α :=
sorry

end false_propositions_l527_527599


namespace train_speed_l527_527481

theorem train_speed (distance time : ℝ) (h1 : distance = 400) (h2 : time = 10) : 
  distance / time = 40 := 
sorry

end train_speed_l527_527481


namespace pipe_length_correct_l527_527166

noncomputable def pipe_length (a : ℝ) (m : ℕ) (n : ℕ) := 
  let y := (168 : ℝ - 80 : ℝ) / (210 + 100) in
  let x := (168 : ℝ - 210 * y) in
  x

theorem pipe_length_correct(x_approx: ℝ) 
  (a : ℝ)
  (m : ℕ)
  (n : ℕ)
  (h_a : a = 0.8)
  (h_m : m = 210)
  (h_n : n = 100)
  : round (pipe_length a m n) = x_approx :=
by {
  unfold pipe_length,
  rw [h_a, h_m, h_n],
  have y_def : y = (168 - 80) / (210 + 100) := rfl,
  rw y_def,
  unfold y,
  simp, -- This step simplifies the expression of y = 8 / 31
  have x_def : x = 168 - (210 * (8 / 31 : ℝ)) := rfl,
  rw x_def,
  unfold x,
  norm_num, -- This simplifies the result to approximately 113.81 in Lean
  norm_cast, -- This converts to natural numbers for rounding correctly
  refl
  sorry -- Simplification to x approximately 108 meters in Lean
}

end pipe_length_correct_l527_527166


namespace cosine_AHB_l527_527722

-- Definitions for orthocenter and vector conditions
variables {A B C H : Type*}
variables [EuclideanGeometry H]

-- Given conditions
def orthocenter_condition (A B C H : Type*) [EuclideanGeometry H] :=
  orthocenter A B C H

-- Vector equation condition
def vector_condition (A B C H : Type*) [EuclideanGeometry H] :=
  3 * (H - A) + 4 * (H - B) + 5 * (H - C) = 0

-- Main theorem to prove
theorem cosine_AHB (A B C H : Type*) [EuclideanGeometry H]
  (h1 : orthocenter_condition A B C H)
  (h2 : vector_condition A B C H) :
  cos (angle A H B) = - (sqrt 6 / 6) :=
sorry

end cosine_AHB_l527_527722


namespace original_equation_solution_l527_527328

noncomputable def original_equation : Prop :=
  ∃ Y P A K P O C : ℕ,
  (Y = 5) ∧ (P = 2) ∧ (A = 0) ∧ (K = 2) ∧ (P = 4) ∧ (O = 0) ∧ (C = 0) ∧
  (Y.factorial * P.factorial * A.factorial = K * 10000 + P * 1000 + O * 100 + C * 10 + C)

theorem original_equation_solution : original_equation :=
  sorry

end original_equation_solution_l527_527328


namespace cos_alpha_beta_l527_527233

variable {α β : ℝ}

theorem cos_alpha_beta (h1 : cos α + cos β = 1 / 2) (h2 : sin α + sin β = 1 / 3) :
  cos (α - β) = -59 / 72 :=
by
  sorry

end cos_alpha_beta_l527_527233


namespace circumscribed_sphere_surface_area_l527_527610

theorem circumscribed_sphere_surface_area
  (AB BC : ℝ)
  (AB_eq : AB = 8)
  (BC_eq : BC = 6) :
  let AC := Real.sqrt (AB^2 + BC^2),
      r := AC / 2,
      surface_area := 4 * Real.pi * r^2
  in surface_area = 100 * Real.pi :=
by
  -- Proof is omitted
  sorry

end circumscribed_sphere_surface_area_l527_527610


namespace probability_condition_l527_527007

noncomputable def Q (x : ℝ) : ℝ :=
  x^2 - 5 * x - 4

def floor (x : ℝ) : ℤ :=
  Int.floor x

def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

def condition (x : ℝ) : Prop :=
  sqrt (Q x) = sqrt (Q (floor x))

def valid_interval (x : ℝ) : Prop :=
  (5 : ℝ) ≤ x ∧ x < (6 : ℝ)

theorem probability_condition :
  ∃ p : ℝ, (p = 1 / 7) ∧
    (∀ x : ℝ, (3 : ℝ) ≤ x ∧ x ≤ (10 : ℝ) → valid_interval x → condition x) :=
sorry

end probability_condition_l527_527007


namespace parallel_and_distance_1_l527_527814

variables (x y : ℝ) (b : ℝ)

/-- Definition of the original line -/
def original_line := 3 * x + 4 * y - 1 = 0

/-- Definition of a line parallel to the original -/
def parallel_line (b : ℝ) := 3 * x + 4 * y - 4 * b = 0

/-- Condition for the distance between the lines being 1 -/
def distance_condition (b : ℝ) := abs(1 - 4 * b) / 5 = 1

theorem parallel_and_distance_1 (b : ℝ) :
  distance_condition b → 
  (parallel_line b = (3 * x + 4 * y + 4 = 0) ∨ parallel_line b = (3 * x + 4 * y - 6 = 0)) := 
by sorry

end parallel_and_distance_1_l527_527814


namespace line_conic_tangent_iff_one_intersection_l527_527148

-- Condition definition
def line_conic_one_intersection (L : Line) (C : Conic) : Prop :=
  ∃ P : Point, P ∈ (L ∩ C) ∧ ∀ Q : Point, Q ∈ (L ∩ C) → Q = P

-- Question (to prove that the condition is equivalent to being tangent)
theorem line_conic_tangent_iff_one_intersection (L : Line) (C : Conic) :
  (line_conic_one_intersection L C) ↔ (is_tangent L C) :=
sorry

end line_conic_tangent_iff_one_intersection_l527_527148


namespace number_of_odd_minus_even_S_up_to_2500_eq_49_l527_527928

-- Definitions for the conditions
def tau (n : ℕ) : ℕ :=
  if n = 0 then 0 else (list.range (n+1)).filter (λ d, n % d = 0).length

def S (n : ℕ) : ℕ :=
  (list.range (n+1)).map tau |>.sum

-- Main theorem statement
theorem number_of_odd_minus_even_S_up_to_2500_eq_49 :
  let a := (list.range 2501).countp (λ n, S n % 2 = 1) in
  let b := (list.range 2501).countp (λ n, S n % 2 = 0) in
  |a - b| = 49 :=
by
  -- Leaves the proof as an exercise
  sorry

end number_of_odd_minus_even_S_up_to_2500_eq_49_l527_527928


namespace lattice_points_impossible_l527_527670

/-
  Prove that given a 1994-sided polygon with side lengths a_k = sqrt(4 + k^2)
  for k = 1, 2, ..., 1994, it is impossible for all the vertices of this polygon
  to be lattice points.
-/
theorem lattice_points_impossible :
  ( ∀ k, 1 ≤ k ∧ k ≤ 1994 → ∃ x y : ℤ, ∃ (x1 y1: ℤ), 
    (a_k = real.sqrt (4 + k^2)) ∧ (a_k = (real.sqrt ((x1 - x)^2 + (y1 - y)^2)))) → false :=
begin
  sorry
end

end lattice_points_impossible_l527_527670


namespace milo_running_distance_l527_527377

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end milo_running_distance_l527_527377


namespace factorize_expression_l527_527572

theorem factorize_expression (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) :=
by
  sorry

end factorize_expression_l527_527572


namespace value_of_expression_l527_527616

theorem value_of_expression (a b m n x : ℝ) 
    (hab : a * b = 1) 
    (hmn : m + n = 0) 
    (hxsq : x^2 = 1) : 
    2022 * (m + n) + 2018 * x^2 - 2019 * (a * b) = -1 := 
by 
    sorry

end value_of_expression_l527_527616


namespace find_usual_time_l527_527459

def usual_speed := S : ℝ

def usual_time := T : ℝ

def slower_speed := (4 / 7) * S

def delay_in_hours := (9 / 60) : ℝ

def speed_time_proportion := S * T = slower_speed * (T + delay_in_hours)

theorem find_usual_time (h : speed_time_proportion) : T = 12 / 60 :=
by
  sorry

end find_usual_time_l527_527459


namespace ron_pay_cuts_l527_527034

-- Define percentages as decimals
def cut_1 : ℝ := 0.05
def cut_2 : ℝ := 0.10
def cut_3 : ℝ := 0.15
def overall_cut : ℝ := 0.27325

-- Define the total number of pay cuts
def total_pay_cuts : ℕ := 3

noncomputable def verify_pay_cuts (cut_1 cut_2 cut_3 overall_cut : ℝ) (total_pay_cuts : ℕ) : Prop :=
  (((1 - cut_1) * (1 - cut_2) * (1 - cut_3) = (1 - overall_cut)) ∧ (total_pay_cuts = 3))

theorem ron_pay_cuts 
  (cut_1 : ℝ := 0.05)
  (cut_2 : ℝ := 0.10)
  (cut_3 : ℝ := 0.15)
  (overall_cut : ℝ := 0.27325)
  (total_pay_cuts : ℕ := 3) 
  : verify_pay_cuts cut_1 cut_2 cut_3 overall_cut total_pay_cuts :=
by sorry

end ron_pay_cuts_l527_527034


namespace problem_l527_527237

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x ^ 2

noncomputable def f (gx : ℝ) (x : ℝ) : ℝ := (2 - 3 * x ^ 2) / x ^ 2

theorem problem (x : ℝ) (hx : x ≠ 0) : f (g x) x = 3 / 2 :=
  sorry

end problem_l527_527237


namespace area_common_rectangle_circle_10x4_radius3_same_center_l527_527519

open Real

def rectCircleIntersectionArea (length width radius : ℝ) : ℝ :=
  9 * π - 4 * sqrt 5

theorem area_common_rectangle_circle_10x4_radius3_same_center :
  rectCircleIntersectionArea 10 4 3 = 9 * π - 4 * sqrt 5 :=
sorry

end area_common_rectangle_circle_10x4_radius3_same_center_l527_527519


namespace fill_in_square_l527_527296

variable {α : Type*} [CommRing α]

theorem fill_in_square (a b : α) (square : α) (h : square * 3 * a * b = 3 * a^2 * b) : square = a :=
sorry

end fill_in_square_l527_527296


namespace number_of_dress_designs_l527_527145

-- Definitions and conditions
def fabric_colors := ["red", "green", "blue", "yellow", "purple"]
def total_patterns := 6
def patterns_for_bp := 1 -- Only the 6th pattern is constrained

-- The main theorem to be proved
theorem number_of_dress_designs : 
    let colors_rg_y : ℕ := 3 in
    let colors_bp : ℕ := 2 in
    let patterns_first5 : ℕ := 5 in
    let patterns_all6 : ℕ := 6 in
    colors_rg_y * patterns_first5 + colors_bp * patterns_all6 = 27 :=
by
    sorry

end number_of_dress_designs_l527_527145


namespace evaluate_f_at_3_l527_527272

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

theorem evaluate_f_at_3 : f 3 = -6 := by
  sorry

end evaluate_f_at_3_l527_527272


namespace original_price_of_frozen_yogurt_is_correct_l527_527748

noncomputable def price_of_frozen_yogurt :=
  let F := 3.94 in  -- price of a pint of frozen yogurt
  let total_cost_after_discount := 65 in
  let total_cost_before_discount := total_cost_after_discount / 0.90 in
  let price_of_shrimp := 5 in
  let cost_of_frozen_yogurt := 5 * F in
  let cost_of_chewing_gum := 2 * (F / 2) in
  let cost_of_shrimp := 5 * price_of_shrimp in
  let cost_of_mixed_nuts := 3 * (2 * F) in
  let total_cost_calculated := cost_of_frozen_yogurt + cost_of_chewing_gum + cost_of_shrimp + cost_of_mixed_nuts in
  total_cost_calculated = total_cost_before_discount

theorem original_price_of_frozen_yogurt_is_correct : price_of_frozen_yogurt = 72.22 :=
  by
  sorry

end original_price_of_frozen_yogurt_is_correct_l527_527748


namespace probability_one_hits_l527_527739

theorem probability_one_hits (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.6) :
  (P_A * (1 - P_B) + (1 - P_A) * P_B) = 0.48 :=
by
  sorry

end probability_one_hits_l527_527739


namespace correct_statement_is_B_l527_527121

-- Definitions of the statements as provided in the conditions
def statement_A : Prop :=
  ∀ (l₁ l₂ : Line) (t : transversal l₁ l₂), corresponding_angles l₁ l₂ t = corresponding_angles l₁ l₂ t

def statement_B : Prop :=
  ∀ (R : Rhombus), sides_equal R

def statement_C : Prop :=
  ∀ (P : Pentagon), is_centrally_symmetric P

def statement_D : Prop :=
  degree (monomial 5 (variables [a, b]) [1, 2]) = 4

-- The theorem to prove the correct statement is statement_B
theorem correct_statement_is_B : statement_B :=
sorry

end correct_statement_is_B_l527_527121


namespace probability_of_odd_result_for_large_n_l527_527140

def calc_odd_probability_lim : ℕ → ℝ → Prop
| n, p => p = 1 / 3

theorem probability_of_odd_result_for_large_n : ∀ n, ∃ p, calc_odd_probability_lim n p := by
  intro n
  exists 1 / 3
  rw calc_odd_probability_lim
  sorry

end probability_of_odd_result_for_large_n_l527_527140


namespace original_profit_percentage_l527_527185

theorem original_profit_percentage
  (P SP : ℝ)
  (h1 : SP = 549.9999999999995)
  (h2 : SP = P * (1 + x / 100))
  (h3 : 0.9 * P * 1.3 = SP + 35) :
  x = 10 := 
sorry

end original_profit_percentage_l527_527185


namespace rhombus_side_length_l527_527521

theorem rhombus_side_length (A : ℝ) (hA : 0 ≤ A) : 
  ∃ s : ℝ, s = (λ (short_diag long_diag : ℝ), 
    let s_squared := (short_diag / 2)^2 + (long_diag / 2)^2 in
    (s_squared)^(1/2)) (√(2*A/3)) (3*√(2*A/3)) ∧
    s = (√(5*A)/3) :=
by {
  sorry
}

end rhombus_side_length_l527_527521


namespace cos_angle_AHB_l527_527719

-- Define the points and vectors in a triangle
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C H : V)

-- Define the orthocenter condition
def ortho_condition : Prop :=
  3 • (H - A) + 4 • (H - B) + 5 • (H - C) = (0 : V)

-- State the problem as a theorem
theorem cos_angle_AHB (h : ortho_condition A B C H) :
  cos (angle A H B) = -√6 / 6 :=
sorry

end cos_angle_AHB_l527_527719


namespace find_abc_l527_527298

theorem find_abc (a b c : ℝ) (ha : a + 1 / b = 5)
                             (hb : b + 1 / c = 2)
                             (hc : c + 1 / a = 3) :
    a * b * c = 10 + 3 * Real.sqrt 11 :=
sorry

end find_abc_l527_527298


namespace shooting_range_problem_l527_527395

theorem shooting_range_problem :
  ∃ (P3 V3 : ℕ), 
  let scores := [10, 9, 9, 8, 8, 5, 4, 4, 3, 2] in
  let P_last := [scores[0], scores[1], scores[3]] in -- scores: 10 + 9 + 8 = 27
  let V_last := [scores[8], scores[9], scores[6]] in -- scores: 2 + 3 + 4 = 9
  (P3 = scores[0] ∧ V3 = scores[8]) ∧
  P_last.sum = 3 * V_last.sum ∧ -- comparison of last three scores
  (P_last.sum - P_last[2] = V_last.sum - V_last[2]) := -- same score in first three
begin
  use scores[0],
  use scores[8],
  split,
  { -- firt part of condition
    split, refl, refl },
  { -- second part of condition
    split, refl, 
      rw [P_last,{ sum_cons, sum_nil }, – associative property of addition and commutativity] } sorry.

end shooting_range_problem_l527_527395


namespace jongkook_points_l527_527042

-- Define the conditions in the problem
def num_questions_solved_each : ℕ := 18
def shinhye_points : ℕ := 100
def jongkook_correct_6_points : ℕ := 8
def jongkook_correct_5_points : ℕ := 6
def points_per_question_6 : ℕ := 6
def points_per_question_5 : ℕ := 5
def jongkook_wrong_questions : ℕ := num_questions_solved_each - jongkook_correct_6_points - jongkook_correct_5_points

-- Calculate Jongkook's points from correct answers
def jongkook_points_from_6 : ℕ := jongkook_correct_6_points * points_per_question_6
def jongkook_points_from_5 : ℕ := jongkook_correct_5_points * points_per_question_5

-- Calculate total points
def jongkook_total_points : ℕ := jongkook_points_from_6 + jongkook_points_from_5

-- Prove that Jongkook's total points is 78
theorem jongkook_points : jongkook_total_points = 78 :=
by
  sorry

end jongkook_points_l527_527042


namespace smallest_a_such_that_sqrt_50a_is_integer_l527_527295

theorem smallest_a_such_that_sqrt_50a_is_integer : ∃ a : ℕ, (∀ b : ℕ, (b > 0 ∧ (∃ k : ℕ, 50 * b = k^2)) → (a ≤ b)) ∧ (∃ k : ℕ, 50 * a = k^2) ∧ a = 2 := 
by
  sorry

end smallest_a_such_that_sqrt_50a_is_integer_l527_527295


namespace percentage_of_gold_coins_is_35_percent_l527_527875

-- Definitions of conditions
def percentage_of_objects_that_are_beads : ℝ := 0.30
def percentage_of_coins_that_are_silver : ℝ := 0.25
def percentage_of_coins_that_are_gold : ℝ := 0.50

-- Problem Statement
theorem percentage_of_gold_coins_is_35_percent 
  (h_beads : percentage_of_objects_that_are_beads = 0.30) 
  (h_silver_coins : percentage_of_coins_that_are_silver = 0.25) 
  (h_gold_coins : percentage_of_coins_that_are_gold = 0.50) :
  0.35 = 0.35 := 
sorry

end percentage_of_gold_coins_is_35_percent_l527_527875


namespace smallest_positive_integer_l527_527804

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l527_527804


namespace remainder_2017_div_89_l527_527796

theorem remainder_2017_div_89 : 2017 % 89 = 59 :=
by
  sorry

end remainder_2017_div_89_l527_527796


namespace breadth_of_water_tank_l527_527837

theorem breadth_of_water_tank (L H V : ℝ) (n : ℕ) (avg_displacement : ℝ) (total_displacement : ℝ)
  (h_len : L = 40)
  (h_height : H = 0.25)
  (h_avg_disp : avg_displacement = 4)
  (h_number : n = 50)
  (h_total_disp : total_displacement = avg_displacement * n)
  (h_displacement_value : total_displacement = 200) :
  (40 * B * 0.25 = 200) → B = 20 :=
by
  intro h_eq
  sorry

end breadth_of_water_tank_l527_527837


namespace length_of_diagonal_EG_l527_527321

theorem length_of_diagonal_EG (EF FG GH HE : ℕ) (hEF : EF = 7) (hFG : FG = 15) 
  (hGH : GH = 7) (hHE : HE = 7) (primeEG : Prime EG) : EG = 11 ∨ EG = 13 :=
by
  -- Apply conditions and proof steps here
  sorry

end length_of_diagonal_EG_l527_527321


namespace ratatouille_cost_per_quart_l527_527746

theorem ratatouille_cost_per_quart:
  let eggplant_weight := 5.5
  let eggplant_price := 2.20
  let zucchini_weight := 3.8
  let zucchini_price := 1.85
  let tomatoes_weight := 4.6
  let tomatoes_price := 3.75
  let onions_weight := 2.7
  let onions_price := 1.10
  let basil_weight := 1.0
  let basil_price_per_quarter := 2.70
  let bell_peppers_weight := 0.75
  let bell_peppers_price := 3.15
  let yield_quarts := 4.5
  let eggplant_cost := eggplant_weight * eggplant_price
  let zucchini_cost := zucchini_weight * zucchini_price
  let tomatoes_cost := tomatoes_weight * tomatoes_price
  let onions_cost := onions_weight * onions_price
  let basil_cost := basil_weight * (basil_price_per_quarter * 4)
  let bell_peppers_cost := bell_peppers_weight * bell_peppers_price
  let total_cost := eggplant_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost + bell_peppers_cost
  let cost_per_quart := total_cost / yield_quarts
  cost_per_quart = 11.67 :=
by
  sorry

end ratatouille_cost_per_quart_l527_527746


namespace num_passenger_cars_l527_527514

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l527_527514


namespace find_angle_A_find_range_b_l527_527264

variables {a b c A B C : ℝ}
variable h1 : a * cos B + sqrt 3 * a * sin B - b - c = 0
variable h_area : (1 / 2) * b * c * sin A = sqrt 3
variable h_acute : A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2

theorem find_angle_A (h1 : a * cos B + sqrt 3 * a * sin B - b - c = 0) : A = π / 3 :=
sorry

theorem find_range_b (h_acute : A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2) (h_area : (1 / 2) * b * c * sin A = sqrt 3): sqrt 2 < b ∧ b < 2 * sqrt 2 :=
sorry

end find_angle_A_find_range_b_l527_527264


namespace solve_system_eq_pos_reals_l527_527407

theorem solve_system_eq_pos_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + y^2 + x * y = 7)
  (h2 : x^2 + z^2 + x * z = 13)
  (h3 : y^2 + z^2 + y * z = 19) :
  x = 1 ∧ y = 2 ∧ z = 3 :=
sorry

end solve_system_eq_pos_reals_l527_527407


namespace cards_given_by_Dan_l527_527745

def initial_cards : Nat := 27
def bought_cards : Nat := 20
def total_cards : Nat := 88

theorem cards_given_by_Dan :
  ∃ (cards_given : Nat), cards_given = total_cards - bought_cards - initial_cards :=
by
  use 41
  sorry

end cards_given_by_Dan_l527_527745


namespace rank_matA_l527_527225

def matA : Matrix (Fin 4) (Fin 5) ℤ :=
  ![![5, 7, 12, 48, -14],
    ![9, 16, 24, 98, -31],
    ![14, 24, 25, 146, -45],
    ![11, 12, 24, 94, -25]]

theorem rank_matA : Matrix.rank matA = 3 :=
by
  sorry

end rank_matA_l527_527225


namespace couch_price_after_six_months_l527_527143

-- Definitions based on conditions:
def initial_price : ℝ := 62500
def increase_factor : ℝ := 6 / 5
def decrease_factor : ℝ := 4 / 5

-- Total number of price changes
def price_increase_count : ℕ := 3
def price_decrease_count : ℕ := 3

-- Calculate the expected final price
def expected_final_price : ℝ :=
  initial_price * (increase_factor ^ price_increase_count) * (decrease_factor ^ price_decrease_count)

-- Lean 4 statement to prove that the calculated price equals the expected price
theorem couch_price_after_six_months :
  expected_final_price = 55296 := by
  sorry

end couch_price_after_six_months_l527_527143


namespace equal_poly_terms_l527_527443

theorem equal_poly_terms (p q : ℝ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : 
  (7 * p^6 * q = 21 * p^5 * q^2) -> p = 3 / 4 :=
by
  sorry

end equal_poly_terms_l527_527443


namespace area_in_scientific_notation_l527_527413

theorem area_in_scientific_notation {a : ℝ} 
  (h : a = 0.00000164) : 
  a = 1.64 * 10 ^ (-6) :=
sorry

end area_in_scientific_notation_l527_527413


namespace translation_coordinates_B_l527_527747

theorem translation_coordinates_B :
  ∀ (A B C D : ℝ × ℝ), segment CD is obtained by translating segment AB →
  A = (-1, 4) ∧ C = (4, 7) →
  D = (-4, 1) →
  B = (-9, -2) :=
by
  intros A B C D h_translation h_AC h_D
  sorry

end translation_coordinates_B_l527_527747


namespace solve_for_xy_l527_527558

-- Define vectors
def a (x : ℝ) : ℝ × ℝ × ℝ := (3, x, -8)
def b (y : ℝ) : ℝ × ℝ × ℝ := (6, 7, y)

-- Cross product definition
def cross_prod (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2.2 * v.2.1 - u.2.1 * v.2.2,
 u.2.0 * v.2.2 - u.2.2 * v.2.0,
 u.2.1 * v.2.0 - u.2.0 * v.2.1)

-- Formulate theorem
theorem solve_for_xy (x y : ℝ) (h : cross_prod (a x) (b y) = (0, 0, 0)) :
  x = 7 / 2 ∧ y = -16 :=
by 
  sorry

end solve_for_xy_l527_527558


namespace finding_real_a_l527_527306

noncomputable def a_pure_imaginary_condition (a : ℝ) : Prop := 
  let z := (a + 3 * complex.I) / (1 - complex.I)
  complex.re z = 0

theorem finding_real_a (a : ℝ) (h : a_pure_imaginary_condition a) : a = 3 :=
sorry

end finding_real_a_l527_527306


namespace adam_earnings_after_taxes_l527_527171

theorem adam_earnings_after_taxes
  (daily_earnings : ℕ) 
  (tax_pct : ℕ)
  (workdays : ℕ)
  (H1 : daily_earnings = 40) 
  (H2 : tax_pct = 10) 
  (H3 : workdays = 30) : 
  (daily_earnings - daily_earnings * tax_pct / 100) * workdays = 1080 := 
by
  -- Proof to be filled in
  sorry

end adam_earnings_after_taxes_l527_527171


namespace tank_salt_solution_l527_527858

theorem tank_salt_solution (x : ℝ) :
  (∀ x : ℝ, (0.20 * x + 20) = (1 / 3) * ((3 / 4) * x + 30)) → x = 200 :=
by
  intro h
  -- We will solve this statement in informal steps for clarity, to be formalized in the proof.
  -- Substituting the given in the problem condition
  have h1 : 0.20 * x + 20 = (1 / 3) * ((3 / 4) * x + 30) := by exact h x
  -- Creating equivalent expressions and solving to get x.
  sorry

end tank_salt_solution_l527_527858


namespace distance_midpoint_to_origin_l527_527989

variables {a b c d m k l n : ℝ}

theorem distance_midpoint_to_origin (h1 : b = m * a + k) (h2 : d = m * c + k) (h3 : n = -1 / m) :
  dist (0, 0) ( ((a + c) / 2), ((m * (a + c) + 2 * k) / 2) ) = (1 / 2) * Real.sqrt ((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k) :=
by
  sorry

end distance_midpoint_to_origin_l527_527989


namespace units_digit_G_500_l527_527903

noncomputable def units_digit := (a : ℕ) → (a % 10)

theorem units_digit_G_500 :
  let G_n := λ n : ℕ, 3^(3^n) + 1 in
  units_digit (G_n 500) = 0 :=
by
  sorry

end units_digit_G_500_l527_527903


namespace artist_used_17_ounces_of_paint_l527_527870

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l527_527870


namespace unique_two_digit_integer_l527_527914

theorem unique_two_digit_integer (s : ℕ) (hs : s > 9 ∧ s < 100) (h : 13 * s ≡ 42 [MOD 100]) : s = 34 :=
by sorry

end unique_two_digit_integer_l527_527914


namespace angelina_speed_l527_527827

theorem angelina_speed (v : ℝ) (h1 : 840 / v - 40 = 240 / v) :
  2 * v = 30 :=
by
  sorry

end angelina_speed_l527_527827


namespace new_rate_ratio_l527_527877

/--
Hephaestus charged 3 golden apples for the first six months and raised his rate halfway through the year.
Apollo paid 54 golden apples in total for the entire year.
The ratio of the new rate to the old rate is 2.
-/
theorem new_rate_ratio
  (old_rate new_rate : ℕ)
  (total_payment : ℕ)
  (H1 : old_rate = 3)
  (H2 : total_payment = 54)
  (H3 : ∀ R : ℕ, new_rate = R * old_rate ∧ total_payment = 18 + 18 * R) :
  ∃ (R : ℕ), R = 2 :=
by {
  sorry
}

end new_rate_ratio_l527_527877


namespace largest_possible_median_l527_527614

theorem largest_possible_median 
  (l : List ℕ)
  (h_l : l = [4, 5, 3, 7, 9, 6])
  (h_pos : ∀ n ∈ l, 0 < n)
  (additional : List ℕ)
  (h_additional_pos : ∀ n ∈ additional, 0 < n)
  (h_length : l.length + additional.length = 9) : 
  ∃ median, median = 7 :=
by
  sorry

end largest_possible_median_l527_527614


namespace number_of_ways_to_select_students_l527_527406

theorem number_of_ways_to_select_students : 
  ∃ n : ℕ, (∀ (S : set ℕ) (A : set (set ℕ)), 
                   S.card = 5 ∧ A.card = 4 ∧ (∀ a ∈ A, (S ∩ a).card ≥ 1) ↔ n = 4) :=
by 
  sorry

end number_of_ways_to_select_students_l527_527406


namespace largest_n_digit_number_divisible_by_61_correct_l527_527463

def largest_n_digit_number (n : ℕ) : ℕ :=
10^n - 1

def largest_n_digit_number_divisible_by_61 (n : ℕ) : ℕ :=
largest_n_digit_number n - (largest_n_digit_number n % 61)

theorem largest_n_digit_number_divisible_by_61_correct (n : ℕ) :
  ∃ k : ℕ, largest_n_digit_number_divisible_by_61 n = 61 * k :=
by
  sorry

end largest_n_digit_number_divisible_by_61_correct_l527_527463


namespace total_number_of_letters_l527_527704

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l527_527704


namespace green_light_probability_l527_527182

theorem green_light_probability :
  ∀ (red yellow green : ℕ), red = 30 → yellow = 5 → green = 40 →
  let total_cycle := red + yellow + green in
  total_cycle = 75 →
  (green : ℚ) / total_cycle = 8 / 15 :=
begin
  intros red yellow green h_red h_yellow h_green total_cycle h_total_cycle,
  rw [h_red, h_yellow, h_green] at h_total_cycle,
  rw [h_red, h_yellow, h_green],
  norm_num at h_total_cycle,
  simp [total_cycle, h_total_cycle],
  norm_num,
end

end green_light_probability_l527_527182


namespace problem_l527_527366

namespace MathProof

variable {p a b : ℕ}

theorem problem (h1 : Nat.Prime p) (h2 : p % 2 = 1) (h3 : a > 0) (h4 : b > 0) (h5 : (p + 1)^a - p^b = 1) : a = 1 ∧ b = 1 := 
sorry

end MathProof

end problem_l527_527366


namespace find_monotonic_increasing_interval_l527_527913

def sin_monotonic_increasing_interval (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ b → sin x1 ≤ sin x2

theorem find_monotonic_increasing_interval :
  let f (x : ℝ) := 2 * sin (π / 4 - x) in
  ∃ a b : ℝ, (a = -5 * π / 4 ∧ b = - π / 4 ∧ sin_monotonic_increasing_interval a b) :=
begin
  -- We assume f(x)
  let f := λ x, 2 * sin (π / 4 - x),
  -- Existential quantifiers for the interval
  use [-5 * π / 4, -π / 4],
  split,
  { refl, },   -- prove that a = -5π / 4
  split,
  { refl, },   -- prove that b = -π / 4
  -- prove that sin is monotonically increasing on the interval
  -- This is where the actual proof will be carried out
  sorry
end

end find_monotonic_increasing_interval_l527_527913


namespace hyperbola_equation_l527_527242

theorem hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (F : ℝ × ℝ)
  (P : ℝ × ℝ) (PF_dist : ℝ) (hyperbola_def : ∀ x y, (x, y) ∈ P → x^2 / a^2 - y^2 / b^2 = 1)
  (parabola_def : ∀ x y, (x, y) ∈ P → y^2 = 8 * x)
  (common_focus : F = (2, 0))
  (PF_value : PF_dist = 5)
  (P_coordinates : P = (3, 2 * sqrt 6)) :
  x^2 - (y^2 / 3) = 1 := 
begin
  sorry -- proof to be filled in
end

end hyperbola_equation_l527_527242


namespace problem_statement_l527_527958

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { P | ∃ x y, (x, y) = P ∧ (x^2 / 25 + y^2 / 9 = 1) }

noncomputable def vertical_line (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { (x, y) | x = P.1 }

def focus_left : ℝ × ℝ := (-4, 0)

def line_through_focus (k : ℝ) : Set (ℝ × ℝ) :=
  { (x, y) | y = k * (x + 4) }

def chord_length (k : ℝ) : ℝ :=
  90 * (1 + k^2) / (25 * k^2 + 9)

theorem problem_statement
    (k1 k2 : ℝ)
    (Hk1k2 : k1 * k2 = 1) :
    (∃ (h f g : ℝ → ℝ), (h(x) = f(x) - g(x)) ∧ (λ x y, x^2 / 25 + y^2 / 9 = 1))
    ∧ (1 / chord_length k1 + 1 / chord_length k2 = 17 / 45) :=
  sorry

end problem_statement_l527_527958


namespace min_folds_exceed_12mm_l527_527660

theorem min_folds_exceed_12mm : ∃ n : ℕ, 0.1 * (2: ℝ)^n > 12 ∧ ∀ m < n, 0.1 * (2: ℝ)^m ≤ 12 := 
by
  sorry

end min_folds_exceed_12mm_l527_527660


namespace function_inequality_proof_l527_527046

noncomputable def f (n : ℕ) : ℝ := sorry
noncomputable def g (n : ℕ) : ℝ := sorry

theorem function_inequality_proof (n : ℕ) (h_n2 : n ≥ 2) (h_g : ∀ m, g m ≥ 1) :
    n * f n - (n - 1) * f (n + 1) ≥ 1 ∧ f 2 = 3 :=
begin
    sorry
end

end function_inequality_proof_l527_527046


namespace distance_between_points_l527_527305

theorem distance_between_points 
  (x_P y_P x_F y_F : ℝ)
  (a t : ℝ)
  (h1 : x_P = 4)
  (h2 : x_F = 2)
  (h3 : y_F = 0)
  (h4 : x_P = t / 2)
  (h5 : y_P = 2 * real.sqrt t)
  (h6 : y_P = a) :
  real.sqrt ((x_P - x_F)^2 + (y_P - y_F)^2) = 6 :=
by
  sorry

end distance_between_points_l527_527305


namespace volume_of_rectangular_parallelepiped_l527_527006

theorem volume_of_rectangular_parallelepiped (x y z p q r : ℝ) 
  (h1 : p = x * y) 
  (h2 : q = x * z) 
  (h3 : r = y * z) : 
  x * y * z = Real.sqrt (p * q * r) :=
by
  sorry

end volume_of_rectangular_parallelepiped_l527_527006


namespace total_letters_in_names_is_33_l527_527697

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l527_527697


namespace student_pass_percentage_l527_527528

theorem student_pass_percentage (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ) :
  marks_obtained = 80 → marks_failed_by = 100 → max_marks = 300 → 
  (marks_obtained + marks_failed_by : ℚ) / max_marks * 100 = 60 :=
by intros h1 h2 h3
   rw [h1, h2, h3]
   norm_num
   sorry

end student_pass_percentage_l527_527528


namespace allocate_to_Team_A_l527_527055

theorem allocate_to_Team_A (x : ℕ) :
  31 + x = 2 * (50 - x) →
  x = 23 :=
by
  sorry

end allocate_to_Team_A_l527_527055


namespace volume_increase_l527_527069

-- Given original length, width, and height
variables (L W H : ℝ)

-- Define the new dimensions after the increase
def L_new := L * 1.04
def W_new := W * 1.07
def H_new := H * 1.10

-- Define the original and new volumes
def V_original := L * W * H
def V_new := (L_new L) * (W_new W) * (H_new H)

-- Calculate the percentage increase in volume
def percentage_increase := ((V_new L W H - V_original L W H) / V_original L W H) * 100

theorem volume_increase : percentage_increase L W H = 22.168 := 
by sorry

end volume_increase_l527_527069


namespace prime_cannot_repeatedly_factor_l527_527122

-- Define factorial of prime numbers (Euclid number)
noncomputable def Euclid (n : ℕ) : ℕ := (Finset.range n).prod (λ i, Nat.prime.succ (Finset.range i)) + 1

-- Prove that no prime number can be a factor of Euclid numbers twice
theorem prime_cannot_repeatedly_factor (p : ℕ) (h : Nat.Prime p) :
  ¬ (∃ n m : ℕ, n ≠ m ∧ p ∣ Euclid n ∧ p ∣ Euclid m) :=
sorry

end prime_cannot_repeatedly_factor_l527_527122


namespace num_pipes_needed_l527_527555

-- Define the diameters of the pipes
def small_pipe_diameter := 2
def large_pipe_diameter := 8

-- Define the radius of the pipes
def small_pipe_radius := small_pipe_diameter / 2
def large_pipe_radius := large_pipe_diameter / 2

-- Define the areas of the pipes using the formula of the area of a circle
def small_pipe_area := Real.pi * (small_pipe_radius^2)
def large_pipe_area := Real.pi * (large_pipe_radius^2)

-- Define the number of smaller pipes needed
def num_small_pipes := large_pipe_area / small_pipe_area

theorem num_pipes_needed : num_small_ppipes = 16 := by
  sorry

end num_pipes_needed_l527_527555


namespace optimal_tom_choices_l527_527094

-- Define the range of the numbers
def valid_range : set ℕ := {x | 1 ≤ x ∧ x ≤ 2011}

-- Function to check if a choice is valid and optimal for Tom given Dick and Harry's strategies
def optimal_choice (tom dick harry : ℕ) : Prop :=
  tom ∈ valid_range ∧
  (dick ≠ tom ∧ dick ∈ valid_range) ∧
  (harry ≠ tom ∧ harry ≠ dick ∧ harry ∈ valid_range) ∧
  (∀ n ∈ valid_range, (abs (n - tom) = min (abs (n - tom)) (min (abs (n - dick)) (abs (n - harry)))))
  -- More detailed optimal strategy conditions as per the problem would be added here.

-- Proposition stating the optimal choices for Tom
theorem optimal_tom_choices : optimal_choice 503 1509 1000 ∧ optimal_choice 1509 503 1010 :=
sorry

end optimal_tom_choices_l527_527094


namespace middle_box_label_l527_527392

theorem middle_box_label :
  ∃ (boxes : fin 23 → Prop), 
  (∃ i, boxes i ∧ i = 11) ∧ 
  ((∀ j, (boxes j ∧ j ≠ 11) -> (¬(j = 10) ↔ ¬(j = 12))) ∧ 
  (∀ j, ¬boxes j ∨ boxes (j+1) ∨ boxes (j-1)) ∧ 
  ∃! i, (i = 11 ∨ ¬(i = 11)) := sorry

end middle_box_label_l527_527392


namespace tangency_circumcircles_l527_527759

noncomputable def point (α : Type) := α

variables {α : Type} [geometry α]

structure triangle (α : Type) :=
(A B C : point α)
(A1 B1 H D M N : point α)
(circumcircle_A1DN circumcircle_B1DM : circle α)

-- Declare the points and their relationships
axioms 
(altitude_A : altitude (triangle.A α) (triangle.A1 α))
(altitude_B : altitude (triangle.B α) (triangle.B1 α))
(coll_inter_H : intersect (altitude_A, altitude_B) (triangle.H α))
(semi_C : lies_on (triangle.D α) (semicircle (triangle.A α, triangle.B α) (circle.Diameter (triangle.A1 α, triangle.B1 α))))
(intersect_AM : intersects (line (triangle.A α, triangle.D α), line (triangle.B α, triangle.B1 α)) (triangle.M α))
(intersect_BN : intersects (line (triangle.B α, triangle.D α), line (triangle.A α, triangle.A1 α)) (triangle.N α))
(is_tangent : tangent (circle (triangle.B1 α, triangle.D α, triangle.M α)) 
(circle (triangle.A1 α, triangle.D α, triangle.N α)) (triangle.D α))

theorem tangency_circumcircles 
: tangent (circle (triangle.B1 α, triangle.D α, triangle.M α)) 
(circle (triangle.A1 α, triangle.D α, triangle.N α)) (triangle.D α) := 
begin
  sorry
end

end tangency_circumcircles_l527_527759


namespace tangent_slope_l527_527251

-- Definitions of the line and the circle
def line (k : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = k * (p.1 - 2) + 2
def circle : ℝ × ℝ → Prop := λ p, (p.1 - 1)^2 + (p.2 - 1)^2 = 2

-- The theorem stating the problem
theorem tangent_slope (k : ℝ) (tangent : ∀ p, line k p → circle p → True) : k = -1 :=
sorry

end tangent_slope_l527_527251


namespace largest_k_for_log3_iterations_l527_527899

-- Definition of U(n)
def U : ℕ → ℝ
| 1     := 3
| (n+1) := 3 ^ U n

noncomputable def main_problem : Prop :=
  let U5 := U 5 in
  let X := U5 ^ 2 in
  let Y := U5 ^ X in
  let log3 := log 3 in
  let log3_iter (k : ℕ) : ℝ → ℝ :=
    nat.rec_on k id (λ n f x, log3 (f x)) in  -- Iterative application of log3
  ∃ k : ℕ, k = 7 ∧ (log3_iter k) Y ≠ 0

-- The statement of the problem
theorem largest_k_for_log3_iterations : main_problem :=
by
  sorry

end largest_k_for_log3_iterations_l527_527899


namespace geometric_progression_terms_l527_527223

theorem geometric_progression_terms (b1 b2 bn : ℕ) (q n : ℕ)
  (h1 : b1 = 3) 
  (h2 : b2 = 12)
  (h3 : bn = 3072)
  (h4 : b2 = b1 * q)
  (h5 : bn = b1 * q^(n-1)) : 
  n = 6 := 
by 
  sorry

end geometric_progression_terms_l527_527223


namespace sqrt_inequality_proof_l527_527092

theorem sqrt_inequality_proof :
  (sqrt 2 + sqrt 7)^2 < (sqrt 3 + sqrt 6)^2 :=
sorry

end sqrt_inequality_proof_l527_527092


namespace soccer_ball_selling_price_l527_527525

theorem soccer_ball_selling_price
  (cost_price_per_ball : ℕ)
  (num_balls : ℕ)
  (total_profit : ℕ)
  (h_cost_price : cost_price_per_ball = 60)
  (h_num_balls : num_balls = 50)
  (h_total_profit : total_profit = 1950) :
  (cost_price_per_ball + (total_profit / num_balls) = 99) :=
by 
  -- Note: Proof can be filled here
  sorry

end soccer_ball_selling_price_l527_527525


namespace Mark_same_color_opposite_foot_l527_527012

variable (shoes : Finset (Σ _ : Fin (14), Bool))

def same_color_opposite_foot_probability (shoes : Finset (Σ _ : Fin (14), Bool)) : ℚ := 
  let total_shoes : ℚ := 28
  let num_black_pairs := 7
  let num_brown_pairs := 4
  let num_gray_pairs := 2
  let num_white_pairs := 1
  let black_pair_prob  := (14 / total_shoes) * (7 / (total_shoes - 1))
  let brown_pair_prob  := (8 / total_shoes) * (4 / (total_shoes - 1))
  let gray_pair_prob   := (4 / total_shoes) * (2 / (total_shoes - 1))
  let white_pair_prob  := (2 / total_shoes) * (1 / (total_shoes - 1))
  black_pair_prob + brown_pair_prob + gray_pair_prob + white_pair_prob

theorem Mark_same_color_opposite_foot (shoes : Finset (Σ _ : Fin (14), Bool)) :
  same_color_opposite_foot_probability shoes = 35 / 189 := 
sorry

end Mark_same_color_opposite_foot_l527_527012


namespace part1_part2_i_part2_ii_l527_527976

noncomputable def f (x : ℝ) (a : ℝ) := x * (a * Real.log x - x - 1)
def g (x : ℝ) := -x

theorem part1 :
  ∀ x > 0, (f x 1) is_strictly_decreasing_on (Ioi 0) :=
sorry

theorem part2_i (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = g x1 ∧ f x2 a = g x2) →
  a > Real.exp 1 :=
sorry

theorem part2_ii {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = g x1 ∧ f x2 a = g x2) :
  ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = g x1 ∧ f x2 a = g x2 →
  x1 * x2 > Real.exp 2 :=
sorry

end part1_part2_i_part2_ii_l527_527976


namespace sum_odd_S_eq_n_pow4_l527_527170

theorem sum_odd_S_eq_n_pow4 (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S 1 = 1)
  (h2 : S 2 = 5)
  (h3 : S 3 = 15)
  (h4 : S 4 = 34)
  (h5 : S 5 = 65)
  (h6 : S 6 = 111)
  (h7 : S 7 = 175)
  (h_pattern : ∀ k, S (2 * k - 1) = (k^4 - sum (k' in range (k-1), S (2 * k' + 1)))) :
  (sum (i in range n, S (2 * i + 1))) = n^4 :=
by sorry

end sum_odd_S_eq_n_pow4_l527_527170


namespace factorize_x4_minus_16y4_l527_527911

theorem factorize_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by 
  sorry

end factorize_x4_minus_16y4_l527_527911


namespace onions_removed_l527_527836

noncomputable def number_of_removed_onions (total_weight_onions : ℝ) 
  (remaining_onions : ℕ) (avg_weight_remaining_onions : ℝ) 
  (avg_weight_removed_onions : ℝ) : ℕ :=
  (total_weight_onions * 1000 - (remaining_onions * avg_weight_remaining_onions)) 
  / avg_weight_removed_onions

theorem onions_removed (total_weight_onions : ℝ) 
  (remaining_onions : ℕ) (avg_weight_remaining_onions : ℝ)
  (avg_weight_removed_onions : ℝ) 
  (h1 : total_weight_onions = 7.68)
  (h2 : remaining_onions = 35)
  (h3 : avg_weight_remaining_onions = 190)
  (h4 : avg_weight_removed_onions = 206) :
  number_of_removed_onions total_weight_onions 
    remaining_onions avg_weight_remaining_onions avg_weight_removed_onions = 5 := 
by 
  sorry

end onions_removed_l527_527836


namespace mike_winning_strategy_l527_527374

-- universe setup (only if necessary for specific results related to combinatorics or game theory)
universe u

-- Definitions for the problem conditions
def board_size := 8
def is_HMM_or_MMH_sequence (board : Array (Array Char)) (pos : Fin board_size × Fin board_size) : Bool :=
  -- This function should check if a given position is the start of a sequence "HMM" or "MMH"
  sorry

-- Main theorem statement
theorem mike_winning_strategy : ∃ k : ℕ, k = 16 ∧ ∀ (M_positions : Fin board_size → Fin board_size → Bool),
  (∃ (H_positions : Fin board_size → Fin board_size → Bool),
  (∑ i j, if M_positions i j then 1 else 0 = k) ∧
  (∑ i j, if H_positions i j then 1 else 0 = k + 1) →
  (∃ (i j : Fin board_size), is_HMM_or_MMH_sequence (board_from M_positions H_positions) (i, j) = true)) ∧
  (∀ k' < k, ∃ (M_positions : Fin board_size → Fin board_size → Bool),
  ∃ (H_positions : Fin board_size → Fin board_size → Bool),
  — Ensure configurations where Harry wins for smaller k
  (∑ i j, if M_positions i j then 1 else 0 = k') ∧
  (∑ i j, if H_positions i j then 1 else 0 = k' + 1) ∧
  (¬ ∃ (i j : Fin board_size), is_HMM_or_MMH_sequence (board_from M_positions H_positions) (i, j) = true))) :=
sorry

end mike_winning_strategy_l527_527374


namespace paddy_field_cultivation_equation_l527_527505

theorem paddy_field_cultivation_equation :
  ∀ (x : ℝ),
  (x > 0) →
  (36 / x = 2 * 30 / (x + 4)) → 
  ∃ (y : ℝ), y = (36 / x) :=
by
  intros x hx h
  use 36 / x
  assumption

end paddy_field_cultivation_equation_l527_527505


namespace complex_numbers_symmetric_bisector_l527_527637

theorem complex_numbers_symmetric_bisector (z1 z2 : ℂ) 
    (h1 : z1 = 1 + 2 * Complex.i)
    (h2 : ∃ w, w = Complex.conj z2 ∧ w = Complex.neg z2) 
    : z1 * z2 = 5 * Complex.i :=
by
  sorry

end complex_numbers_symmetric_bisector_l527_527637


namespace mail_distribution_l527_527151

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l527_527151


namespace arithmetic_mean_of_14_22_36_l527_527108

theorem arithmetic_mean_of_14_22_36 : (14 + 22 + 36) / 3 = 24 := by
  sorry

end arithmetic_mean_of_14_22_36_l527_527108


namespace combined_earnings_l527_527882

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l527_527882


namespace first_player_wins_with_optimal_play_l527_527087

-- Define the game setup and rules
def initial_coins : ℕ := 2001

def first_player_can_take (n : ℕ) : Prop := n % 2 = 1 ∧ n ≥ 1 ∧ n ≤ 99
def second_player_can_take (n : ℕ) : Prop := n % 2 = 0 ∧ n ≥ 2 ∧ n ≤ 100

-- Define a predicate stating that the player who cannot move loses
def cannot_move_loses (coins : ℕ) (player_turn : bool) : Prop :=
  (player_turn ∧ ∀ n, first_player_can_take n → (coins < n)) ∨
  (¬player_turn ∧ ∀ n, second_player_can_take n → (coins < n))

-- Define a predicate stating that the first player has a winning strategy
def first_player_wins (coins : ℕ) : Prop :=
  ∃ strategy, (strategy initial_coins true) = "first_player_wins"

theorem first_player_wins_with_optimal_play : first_player_wins initial_coins := by
  sorry

end first_player_wins_with_optimal_play_l527_527087


namespace projection_multiplier_l527_527921

noncomputable def a : ℝ × ℝ := (3, 6)
noncomputable def b : ℝ × ℝ := (-1, 0)

theorem projection_multiplier :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let proj := (dot_product / b_norm_sq) * 2
  (proj * b.1, proj * b.2) = (6, 0) :=
by 
  sorry

end projection_multiplier_l527_527921


namespace painted_by_all_three_l527_527788

-- Define the portions painted by each color.
variable (P_R P_G P_B : ℝ)
-- Represent the unpainted regions as functions of the painted regions.
def U_R := 1 - P_R
def U_G := 1 - P_G
def U_B := 1 - P_B

-- The total unpainted area when summed.
def total_unpainted := U_R + U_G + U_B

-- The portion definitely painted by all three colors.
def definitely_painted := 1 - total_unpainted

-- Proof statement:
theorem painted_by_all_three (hR : P_R = 0.75) (hG : P_G = -0.7) (hB : P_B = -0.65) :
  definitely_painted P_R P_G P_B = 0.10 := by
  unfold definitely_painted total_unpainted U_R U_G U_B
  simp [hR, hG, hB]
  norm_num
  sorry

end painted_by_all_three_l527_527788


namespace system_of_equations_correct_l527_527789

theorem system_of_equations_correct (x y : ℝ) (h1 : x + y = 2000) (h2 : y = x * 0.30) :
  x + y = 2000 ∧ y = x * 0.30 :=
by 
  exact ⟨h1, h2⟩

end system_of_equations_correct_l527_527789


namespace find_S_coords_l527_527636

-- Definitions for the given conditions
def P : ℝ × ℝ × ℝ := (4, -2, 3)
def Q : ℝ × ℝ × ℝ := (0, 3, -5)
def R : ℝ × ℝ × ℝ := (-2, 2, 3)

-- Function to compute the midpoint between two 3D points
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

-- Given that PQRS is a parallelogram, we declare S and aim to prove its coordinates
theorem find_S_coords (S : ℝ × ℝ × ℝ) (h1 : midpoint P R = midpoint Q S) :
  S = (2, -3, 11) :=
sorry

end find_S_coords_l527_527636


namespace second_smallest_packs_of_hot_dogs_l527_527566

theorem second_smallest_packs_of_hot_dogs
    (n : ℤ) 
    (h1 : ∃ m : ℤ, 12 * n = 8 * m + 6) :
    ∃ k : ℤ, n = 4 * k + 7 :=
sorry

end second_smallest_packs_of_hot_dogs_l527_527566


namespace sum_of_integers_l527_527426

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l527_527426


namespace ratio_of_volumes_l527_527471
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l527_527471


namespace bus_trip_distance_l527_527139

theorem bus_trip_distance
  (D S : ℕ) (H1 : S = 55)
  (H2 : D / S - 1 = D / (S + 5))
  : D = 660 :=
sorry

end bus_trip_distance_l527_527139


namespace shaded_areas_comparison_l527_527906

theorem shaded_areas_comparison :
  let area_square (s : ℝ) := s * s in
  let area_triangle (s : ℝ) := area_square s / 4 in
  let area_squareI (s : ℝ) := area_triangle s in
  let area_small_square (s : ℝ) := area_square s / 4 in
  let area_squareII (s : ℝ) := 2 * area_small_square s in
  let area_oct_triangle (s : ℝ) := area_square s / 8 in
  let area_squareIII (s : ℝ) := 4 * area_oct_triangle s in
  ∀ (s : ℝ), 
  s > 0 → 
  area_squareII s = area_squareIII s ∧ 
  area_squareI s ≠ area_squareII s := 
by 
  intros s hs
  let area_square := s * s
  let area_triangle := area_square / 4
  let area_squareI := area_triangle
  let area_small_square := area_square / 4
  let area_squareII := 2 * area_small_square
  let area_oct_triangle := area_square / 8
  let area_squareIII := 4 * area_oct_triangle
  split
  { show area_squareII = area_squareIII
    calc area_squareII = 2 * (area_square / 4) : by refl
                  ...   = area_square / 2     : by ring_nf
                  ...   = 4 * (area_square / 8) : by ring_nf
                  ...   = area_squareIII      : by refl }
  { show area_squareI ≠ area_squareII
    calc area_squareI = area_square / 4 : by refl
                  ...   ≠ area_square / 2 : by norm_num }. 

end shaded_areas_comparison_l527_527906


namespace largest_d_l527_527351

theorem largest_d (a b c d : ℝ) (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := sorry

end largest_d_l527_527351


namespace min_value_function_A_function_B_incorrect_function_C_incorrect_function_D_inconclusive_l527_527175

theorem min_value_function_A :
  ∃ x : ℝ, x = 0 ∧ ∀ x : ℝ, y = (x^2 + 2) / real.sqrt(x^2 + 1) ∧ 2 ≤ y :=
sorry

theorem function_B_incorrect :
  ∀ x : ℝ, 3 < x → (x + 4/(x-1) + 1) > 9/2 :=
sorry

theorem function_C_incorrect :
  ∀ x : ℝ, 0 < x ∧ x < π/2 → (real.sin x + 1 / real.sin x) > 2 :=
sorry

theorem function_D_inconclusive :
  ¬ (∀ x : ℝ, (x + 1/x) = 2) :=
sorry

end min_value_function_A_function_B_incorrect_function_C_incorrect_function_D_inconclusive_l527_527175


namespace probability_purple_or_orange_face_l527_527409

theorem probability_purple_or_orange_face 
  (total_faces : ℕ) (green_faces : ℕ) (purple_faces : ℕ) (orange_faces : ℕ) 
  (h_total : total_faces = 10) 
  (h_green : green_faces = 5) 
  (h_purple : purple_faces = 3) 
  (h_orange : orange_faces = 2) :
  (purple_faces + orange_faces) / total_faces = 1 / 2 :=
by 
  sorry

end probability_purple_or_orange_face_l527_527409


namespace has_one_zero_in_interval_l527_527767

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 4

theorem has_one_zero_in_interval : 
  (∃! x ∈ set.Icc 1 3, f x = 0) :=
by {
  sorry
}

end has_one_zero_in_interval_l527_527767


namespace minimum_value_of_z_l527_527464

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l527_527464


namespace cos_neg_is_increasing_l527_527067

theorem cos_neg_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → cos (-x1) < cos (-x2) :=
by
  sorry

end cos_neg_is_increasing_l527_527067


namespace max_product_of_sum_eq_l527_527360

theorem max_product_of_sum_eq (N k t r : ℕ) (x : ℕ → ℕ) (hx : ∑ i in finset.range k, x i = N) 
  (N_eq : N = k * t + r) (r_range : 0 ≤ r ∧ r < k) (k_range : 1 < k ∧ k ≤ N) :
  ∏ i in finset.range k, x i ≤ t^(k-r) * (t+1)^r := 
sorry

end max_product_of_sum_eq_l527_527360


namespace milo_run_distance_l527_527376

def cory_speed : ℝ := 12
def milo_roll_speed := cory_speed / 2
def milo_run_speed := milo_roll_speed / 2
def time_hours : ℝ := 2

theorem milo_run_distance : milo_run_speed * time_hours = 6 := 
by 
  /- The proof goes here -/
  sorry

end milo_run_distance_l527_527376


namespace petya_result_less_than_one_tenth_l527_527455

theorem petya_result_less_than_one_tenth 
  (a b c d e f : ℕ) 
  (ha: a.gcd b = 1) (hb: c.gcd d = 1)
  (hc: e.gcd f = 1) 
  (vasya_correct: (a / b) + (c / d) + (e / f) = 1) :
  (a + c + e) / (b + d + f) < 1 / 10 :=
by
  -- proof goes here
  sorry

end petya_result_less_than_one_tenth_l527_527455


namespace store_discount_problem_l527_527526

theorem store_discount_problem (original_price : ℝ) :
  let price_after_first_discount := original_price * 0.75
  let price_after_second_discount := price_after_first_discount * 0.90
  let true_discount := 1 - price_after_second_discount / original_price
  let claimed_discount := 0.40
  let difference := claimed_discount - true_discount
  true_discount = 0.325 ∧ difference = 0.075 :=
by
  sorry

end store_discount_problem_l527_527526


namespace horizontal_asymptote_of_f_l527_527770

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f : 
  Tendsto (fun x => f x) (Filter.atTop) (𝓝 (7 / 4)) :=
sorry

end horizontal_asymptote_of_f_l527_527770


namespace length_of_pipe_l527_527167

-- Define the conditions
def gavrila_step_length : ℝ := 0.8
def steps_in_direction_of_tractor : ℝ := 210
def steps_against_direction_of_tractor : ℝ := 100

-- Define what we need to prove
theorem length_of_pipe :
  let y := 8 / 31 in
  let x := 168 - 210 * y in
  let x_approx := 113.81 in
  (x ≈ 113.81) ∨ (108 ≤ floor x ∧ floor x ≤ 109) :=
by
  let y := 8 / 31
  let x := 168 - 210 * y
  let x_approx := 113.81
  /-
  We have:
  168 - 210y, where y = 8 / 31.
  This gives us approximately 113.81 meters
  
  Since we want the length rounded to the nearest whole number:
  floor(x) should be approximately 108 (since 113.81 rounded is 114).
  We include the flooring condition just to ensure thoroughness and rounding behavior.
  -/
  have step_1 : y = 8 / 31 := rfl
  have step_2 : x = 168 - 210 * y := rfl
  have step_3 : x ≈ 113.81 := by
    norm_num
    linarith

  have step_4 := floor.to_nearby x
  have step_5 : (108 ≤ floor x ∧ floor x ≤ 109) := by sorry
  exact Or.inr step_5

end length_of_pipe_l527_527167


namespace milo_running_distance_l527_527378

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end milo_running_distance_l527_527378


namespace prob_at_least_one_success_is_correct_mean_profit_is_correct_l527_527840

-- Definitions of success probabilities
def prob_success_A : ℚ := 2/3
def prob_success_B : ℚ := 3/5

-- Definitions of profit values
def profit_A : ℚ := 120
def profit_B : ℚ := 100

-- Conditions of independence and profit calculations
axiom indep_A_B : ∀ events_A events_B, 
  (prob_success_A * prob_success_B) = (prob_success_A * prob_success_B)

-- Probability calculations
noncomputable def prob_at_least_one_success : ℚ :=
  1 - ((1 - prob_success_A) * (1 - prob_success_B))

theorem prob_at_least_one_success_is_correct : 
  prob_at_least_one_success = 13/15 :=
by sorry

-- Profit distribution calculations
noncomputable def profit_distribution : List (ℚ × ℚ) :=
  [ (0, (1 - prob_success_A) * (1 - prob_success_B)),
    (profit_A, prob_success_A * (1 - prob_success_B)),
    (profit_B, (1 - prob_success_A) * prob_success_B),
    (profit_A + profit_B, prob_success_A * prob_success_B) ]

noncomputable def mean_profit : ℚ :=
  List.sum (List.map (λ (p : ℚ × ℚ), p.1 * p.2) profit_distribution)

theorem mean_profit_is_correct :
  mean_profit = 140 :=
by sorry

end prob_at_least_one_success_is_correct_mean_profit_is_correct_l527_527840


namespace min_value_of_quadratic_l527_527466

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l527_527466


namespace tod_trip_time_l527_527093

noncomputable def total_time (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  d1 / s1 + d2 / s2 + d3 / s3 + d4 / s4

theorem tod_trip_time :
  total_time 55 95 30 75 40 50 20 60 = 6.025 :=
by 
  sorry

end tod_trip_time_l527_527093


namespace license_plate_increase_l527_527687

theorem license_plate_increase : 
  let old_plate_count := 26^3 * 10^2 in
  let new_plate_count := 26^2 * 10^4 in
  new_plate_count / old_plate_count = 50 / 13 :=
by
  sorry

end license_plate_increase_l527_527687


namespace isosceles_triangle_with_common_side_angle_l527_527536

theorem isosceles_triangle_with_common_side_angle 
  (CD CB : ℝ)
  (h_iso : CD = CB)
  (angle_rectangle : ℝ)
  (angle_triangle : ℝ)
  (h_angle_rect : angle_rectangle = 80)
  (h_angle_tri : angle_triangle = 70) :
  ∠CDB = 15 :=
by
  sorry

end isosceles_triangle_with_common_side_angle_l527_527536


namespace find_certain_number_l527_527211

-- Define the conditions
def certain_number (C x : ℕ) := C - |(-x + 6)| = 26

theorem find_certain_number (x : ℕ) (h : x = 10) : ∃ C, certain_number C x ∧ C = 30 := 
by 
  use 30
  rw [h]
  unfold certain_number
  sorry

end find_certain_number_l527_527211


namespace expand_polynomial_l527_527910

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) = 12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end expand_polynomial_l527_527910


namespace intersection_of_sets_l527_527985

open Set

theorem intersection_of_sets (M N : Set ℝ) (hM : M = {2, 4, 6, 8}) (hN : N = {x | -1 < x ∧ x < 5}) : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_sets_l527_527985


namespace natasha_time_to_top_l527_527383

theorem natasha_time_to_top (T : ℝ) 
  (descent_time : ℝ) 
  (whole_journey_avg_speed : ℝ) 
  (climbing_speed : ℝ) 
  (desc_time_condition : descent_time = 2) 
  (whole_journey_avg_speed_condition : whole_journey_avg_speed = 3.5) 
  (climbing_speed_condition : climbing_speed = 2.625) 
  (distance_to_top : ℝ := climbing_speed * T) 
  (avg_speed_condition : whole_journey_avg_speed = 2 * distance_to_top / (T + descent_time)) :
  T = 4 := by
  sorry

end natasha_time_to_top_l527_527383


namespace defeated_candidate_percentage_l527_527540

variable (V : ℕ) (D : ℕ) (P : ℚ)

def votes_valid (total_votes : ℕ) (invalid_votes : ℕ) : ℕ :=
  total_votes - invalid_votes

def votes_received (d_votes : ℕ) : ℕ :=
  2 * d_votes + 500

theorem defeated_candidate_percentage :
  ∀ (total_votes invalid_votes d_votes : ℕ) 
    (h1 : total_votes = 850)
    (h2 : invalid_votes = 10)
    (h3 : votes_valid total_votes invalid_votes = 840)
    (h4 : votes_received d_votes = 840)
    (h5 : d_votes = 170),
  P = (170 : ℚ) / 840 * 100 → P ≈ 20.24 :=
by
  sorry

end defeated_candidate_percentage_l527_527540


namespace largest_number_less_than_200_with_properties_l527_527017

/-- Define the conditions for a number being a perfect square, less than 200, and divisible by 3 -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_less_than_200 (n : ℕ) : Prop :=
  n < 200

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

/-- The main statement we aim to prove -/
theorem largest_number_less_than_200_with_properties :
  ∃ n, is_perfect_square n ∧ is_less_than_200 n ∧ is_divisible_by_3 n ∧ ∀ m, is_perfect_square m → is_less_than_200 m → is_divisible_by_3 m → m ≤ n :=
    ∃ n, n = 144 :=
begin
  sorry
end

end largest_number_less_than_200_with_properties_l527_527017


namespace euler_line_bisects_area_l527_527026

theorem euler_line_bisects_area (Δ : Triangle) (EulerLineBisectsArea : BisectionOfAreaByEulerLine Δ) : 
  IsIsosceles Δ ∨ IsRightAngled Δ := 
by
  sorry

end euler_line_bisects_area_l527_527026


namespace sum_series_l527_527228

noncomputable def u_r (r : ℕ) : ℕ := r * (r + 1) / 2

theorem sum_series :
  (∑ i in Finset.range 100, i + 1) / 2 = 2575 :=
by
  sorry

end sum_series_l527_527228


namespace determine_list_price_l527_527174

theorem determine_list_price 
  (x : ℝ) 
  (alice_selling_price : x - 15 = alice_price)
  (alice_commission : 0.15 * alice_price = alice_comm)
  (bob_selling_price : x - 25 = bob_price)
  (bob_commission : 0.25 * bob_price = bob_comm)
  (equal_commission : alice_comm = bob_comm) : 
  x = 40 := 
by
  have h1 : alice_price = x - 15 := by rw [alice_selling_price]
  have h2 : bob_price = x - 25 := by rw [bob_selling_price]
  have h3 : alice_comm = 0.15 * (x - 15) := by rw [alice_comm, h1]
  have h4 : bob_comm = 0.25 * (x - 25) := by rw [bob_comm, h2]
  have h5 : 0.15 * (x - 15) = 0.25 * (x - 25) := by rw [←h3, ←h4, equal_commission]
  sorry

end determine_list_price_l527_527174


namespace selling_price_same_loss_as_profit_l527_527078

theorem selling_price_same_loss_as_profit (cost_price selling_price_with_profit selling_price_with_loss profit loss : ℝ)
  (h1 : selling_price_with_profit - cost_price = profit)
  (h2 : cost_price - selling_price_with_loss = loss)
  (h3 : profit = loss) :
  selling_price_with_loss = 52 :=
by
  have h4 : selling_price_with_profit = 66 := by sorry
  have h5 : cost_price = 59 := by sorry
  have h6 : profit = 66 - 59 := by sorry
  have h7 : profit = 7 := by sorry
  have h8 : loss = 59 - selling_price_with_loss := by sorry
  have h9 : loss = 7 := by sorry
  have h10 : selling_price_with_loss = 59 - loss := by sorry
  have h11 : selling_price_with_loss = 59 - 7 := by sorry
  have h12 : selling_price_with_loss = 52 := by sorry
  exact h12

end selling_price_same_loss_as_profit_l527_527078


namespace price_per_glass_third_day_l527_527735

-- Define the conditions
def price_first_day := 0.60
def revenue_day_equal (A B : ℝ) := A = B
def volume_first_day (O : ℝ) := 2 * O
def volume_third_day (O : ℝ) := 4 * O

-- Price per glass on the third day
def price_third_day (w : ℝ) (N : ℝ) (O : ℝ) := 
  revenue_day_equal (price_first_day * N) (w * (2 * N))

-- Theorem statement to prove the price per glass on the third day
theorem price_per_glass_third_day :
  ∀ (w O N : ℝ),
  price_third_day w N O →
  volume_first_day O = 2 * O →
  volume_third_day O = 4 * O →
  w = 0.30 
:= 
by intros; sorry

end price_per_glass_third_day_l527_527735


namespace ways_to_stand_l527_527531

-- Definitions derived from conditions
def num_steps : ℕ := 7
def max_people_per_step : ℕ := 2

-- Define a function to count the number of different ways
noncomputable def count_ways : ℕ :=
  336

-- The statement to be proven in Lean 4
theorem ways_to_stand : count_ways = 336 :=
  sorry

end ways_to_stand_l527_527531


namespace problem_statement_l527_527620

-- Define the factorial of 100
def fact100 : ℕ := 100.fact

-- Condition: 100! = 12^n * M
def condition (n : ℕ) (M : ℕ) : Prop :=
  fact100 = 12^n * M

-- Requirements: n = 48 and M = fact100 / 12^48
def M_def : ℕ := fact100 / 12^48

-- The problem statement to prove
theorem problem_statement :
  ∀ (M : ℕ), condition 48 M → M = M_def → M % 2 = 0 ∧ M % 3 ≠ 0 :=
by
  intros M h_condition h_M_def
  sorry

end problem_statement_l527_527620


namespace flowers_per_bug_l527_527381

theorem flowers_per_bug (bugs : ℝ) (flowers : ℝ) (h_bugs : bugs = 2.0) (h_flowers : flowers = 3.0) :
  flowers / bugs = 1.5 :=
by
  sorry

end flowers_per_bug_l527_527381


namespace fraction_of_book_finished_l527_527048

variables (x y : ℝ)

theorem fraction_of_book_finished (h1 : x = y + 90) (h2 : x + y = 270) : x / 270 = 2 / 3 :=
by sorry

end fraction_of_book_finished_l527_527048


namespace games_mike_can_buy_l527_527014

theorem games_mike_can_buy (initial_money spent_money game_cost : ℕ)
    (h1 : initial_money = 69)
    (h2 : spent_money = 24)
    (h3 : game_cost = 5) :
    (initial_money - spent_money) / game_cost = 9 := by
  subst h1
  subst h2
  subst h3
  simp
  sorry

end games_mike_can_buy_l527_527014


namespace reflect_point_center_l527_527762

theorem reflect_point_center :
  ∀ (x y : ℝ), (x, y) = (4, -3) → 
  (let (x', y') := (-y, x) in
   (x', y')) = (3, -4) :=
by
  intros x y h
  rw [h]
  simp
  sorry

end reflect_point_center_l527_527762


namespace problem_AD_times_CD_l527_527095

open EuclideanGeometry

noncomputable theory

variable {A B C P D : Point}
variable {PB PD : ℝ}

theorem problem_AD_times_CD :
  (PB = 4) →
  (PD = 1) →
  (dist P A = dist P B) →
  (angle P A B = 2 * angle A C B) →
  ∃ D, line_through A C → line_through B P →
  dist P D = 1 → dist P C = dist P B → dist P A = dist P B →
  AD * CD = 15 :=
by {
  sorry
}

end problem_AD_times_CD_l527_527095


namespace find_four_numbers_l527_527217

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b = 2024) 
  (h2 : a + c = 2026) 
  (h3 : a + d = 2030) 
  (h4 : b + c = 2028) 
  (h5 : b + d = 2032) 
  (h6 : c + d = 2036) : 
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) := 
sorry

end find_four_numbers_l527_527217


namespace maximum_sum_of_products_l527_527865

theorem maximum_sum_of_products :
  ∃ (a b c d e f : ℕ), 
    {a, b, c, d, e, f} = {2, 3, 4, 5, 6, 7} ∧
    (a + b) * (c + d) * (e + f) ≤ 729 :=
by sorry

end maximum_sum_of_products_l527_527865


namespace find_omega_monotonicity_intervals_l527_527276

def f (ω : ℝ) (x : ℝ) : ℝ := 4 * cos (ω * x) * sin (ω * x + Real.pi / 4)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_omega 
  (h : ∀ x, f ω x = f ω (x + Real.pi))
  (hω_pos : ω > 0) : ω = 1 := 
sorry

theorem monotonicity_intervals 
  (x : ℝ) (hω : ω = 1)
  (hx: 0 ≤ x ∧ x ≤ Real.pi) : 
  (0 ≤ x ∧ x ≤ Real.pi / 8) ∨ 
  (5 * Real.pi / 8 ≤ x ∧ x ≤ Real.pi ∧ (f 1 x) is_monotonically_increasing)
  ∨ (Real.pi / 8 ≤ x ∧ x ≤ 5 * Real.pi / 8 ∧ (f 1 x) is_monotonically_decreasing) :=
sorry

end find_omega_monotonicity_intervals_l527_527276


namespace ryan_learning_hours_l527_527570

theorem ryan_learning_hours : (english_hours - chinese_hours = 4) :=
by
  -- Define the conditions
  let english_hours := 6
  let chinese_hours := 2
  
  -- State the theorem to prove
  show (english_hours - chinese_hours = 4)
  -- Skip the proof with sorry
  sorry

end ryan_learning_hours_l527_527570


namespace polygon_has_twelve_sides_l527_527664

theorem polygon_has_twelve_sides
  (sum_exterior_angles : ℝ)
  (sum_interior_angles : ℝ → ℝ)
  (n : ℝ)
  (h1 : sum_exterior_angles = 360)
  (h2 : ∀ n, sum_interior_angles n = 180 * (n - 2))
  (h3 : ∀ n, sum_interior_angles n = 5 * sum_exterior_angles) :
  n = 12 :=
by
  sorry

end polygon_has_twelve_sides_l527_527664


namespace sum_first_n_terms_geom_seq_l527_527331

def geom_seq (n : ℕ) : ℕ :=
match n with
| 0     => 2
| k + 1 => 3 * geom_seq k

def sum_geom_seq (n : ℕ) : ℕ :=
(geom_seq 0) * (3 ^ n - 1) / (3 - 1)

theorem sum_first_n_terms_geom_seq (n : ℕ) :
sum_geom_seq n = 3 ^ n - 1 := by
sorry

end sum_first_n_terms_geom_seq_l527_527331


namespace color_correct_l527_527905

-- Define the data type for color
inductive Color
| red
| blue

open Color

-- Define the coloring of each number
def color : ℕ → Color
| 1  := blue
| 2  := blue
| 3  := blue
| 4  := blue
| 5  := red
| 6  := blue
| 7  := blue
| 8  := blue
| 9  := blue
| 10 := red
| _ := sorry -- For completeness, but shouldn't reach this in [1, 10]

theorem color_correct :
  (color 5 = red) ∧
  (∃ n, color n = blue) ∧
  (∀ m n, m ≠ n ∧ color m ≠ color n ∧ m + n ≤ 10 → color (m + n) = blue) ∧
  (∀ m n, m ≠ n ∧ color m ≠ color n ∧ m * n ≤ 10 → color (m * n) = red) :=
by
  split
  -- Condition: The number 5 is red
  { exact rfl }
  split
  -- Condition: At least one number is blue
  { use 1
    exact rfl }
  split
  -- Condition: If m and n are different colors and m+n ≤ 10, then m+n is blue
  { intros m n h
    cases h with h1 h2
    cases h2 with h_col h_sum
    cases m <;> cases n <;> simp [h1, h_col] at h_sum { 
      all_goals { try { deciding } } 
    }
  -- sorry -- Fill this part as part of the proof
  }
  -- Condition: If m and n are different colors and mn ≤ 10, then mn is red
  { intros m n h
    cases h with h1 h2
    cases h2 with h_col h_prod
    cases m <;> cases n <;> simp [h1, h_col] at h_prod { 
      all_goals { try { deciding } } 
    }
  -- sorry -- Fill this part as part of the proof
  }


end color_correct_l527_527905


namespace max_lambda_neg_inv_27_equality_conditions_neg_inv_27_l527_527579

variables {α β γ : Real} {x λ : Real}
variables {a b c : Real}

noncomputable def f (x : Real) := x^3 + a * x^2 + b * x + c

theorem max_lambda_neg_inv_27 (hroots : 0 ≤ α ∧ α ≤ β ∧ β ≤ γ) 
  (hpoly : f(x) = (x - α) * (x - β) * (x - γ))
  (hx_geq_0 : x ≥ 0) :
  ∃ (λ : Real), λ = -1 / 27 ∧ (∀ (x : Real), 0 ≤ x → f(x) ≥ λ * (x - a)^3) :=
  sorry

theorem equality_conditions_neg_inv_27 (hroots_eq : α = β ∧ β = γ ∨ α = β ∧ β = 0 ∧ γ = 2 * x) (hx_eq_0 : x = 0):
  (∀ (x : Real), f(x) = -1 / 27 * (x - a)^3) :=
  sorry

end max_lambda_neg_inv_27_equality_conditions_neg_inv_27_l527_527579


namespace find_circle_c_and_n_find_chord_length_l527_527959

-- Definitions
def circle_c_tangent_point := (3 / 2, sqrt 3 / 2)
def circle_c_tangent_line (n : ℝ) := λ x y, x + sqrt 3 * y + n = 0
def circle_m_eqn (r : ℝ) := x^2 + (y - sqrt 15)^2 = r^2

-- The main problem statements
theorem find_circle_c_and_n :
  ∃ (n : ℝ) (cx : ℝ), n = -3 ∧ cx = 1 ∧
  (∀ (x y : ℝ), (x - cx)^2 + y^2 = 1 ↔ true) :=
sorry

theorem find_chord_length (r : ℝ) :
  r = 3 ∨ r = 5 →
  ∃ (L : ℝ), (L = 2 * sqrt 3 ∨ L = 2 * sqrt 19) ∧
  (L = 2 * sqrt (r^2 - (sqrt 6)^2)) :=
sorry

end find_circle_c_and_n_find_chord_length_l527_527959


namespace distance_P_to_x_axis_l527_527966

-- Define the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1

-- Define the foci distances
def foci_distance (x : ℝ) : ℝ := Real.sqrt (16 - 9)

-- Define the right-angled triangle property
def is_right_angle_triangle (x y c : ℝ) (F1 F2 P : Point) : Prop :=
  let c := Real.sqrt (16 - 9)
  in (c < 3) ∧ 
     ((F1.x = -c ∧ F2.x = c ∧ (P.x^2 / 16 + P.y^2 / 9 = 1) ∧
     ∃ angle, angle < 90 ∧ angle + 90 = 180 ∧
     (P.x, P.y) = +sqrt(7))) -- Embedding the condition that right angle must be at P

-- Define the distance function
def point_distance_to_x_axis (P : Point) : ℝ := Real.abs P.y

-- Main theorem statement
theorem distance_P_to_x_axis : 
  ∀ (P F1 F2 : Point), ellipse_eq P.x P.y → is_right_angle_triangle P.x P.y (foci_distance P.x) F1 F2 P → 
  point_distance_to_x_axis P = 9 / 4 := 
sorry

end distance_P_to_x_axis_l527_527966


namespace find_f_minus3_and_f_2009_l527_527266

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Conditions
axiom h1 : is_odd f
axiom h2 : f 1 = 2
axiom h3 : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Questions
theorem find_f_minus3_and_f_2009 : f (-3) = 0 ∧ f 2009 = -2 :=
by 
  sorry

end find_f_minus3_and_f_2009_l527_527266


namespace absent_children_l527_527022

/-- On a school's annual day, sweets were to be equally distributed amongst 112 children. 
But on that particular day, some children were absent. Thus, the remaining children got 6 extra sweets. 
Each child was originally supposed to get 15 sweets. Prove that 32 children were absent. -/
theorem absent_children (A : ℕ) 
  (total_children : ℕ := 112) 
  (sweets_per_child : ℕ := 15) 
  (extra_sweets : ℕ := 6)
  (absent_eq : (total_children - A) * (sweets_per_child + extra_sweets) = total_children * sweets_per_child) : 
  A = 32 := 
by
  sorry

end absent_children_l527_527022


namespace Pete_books_total_l527_527656

variable (Matt_books_first_year Pete_books_first_year Pete_books_second_year : ℕ)

-- Conditions
def condition1 (Matt_books_first_year : ℕ) :=
  Matt_books_first_year = 75 / 1.5 -- Matt read 75 books in the second year, which is 50% more than the first year.

def condition2 (Pete_books_first_year : ℕ) (Matt_books_first_year : ℕ) :=
  Pete_books_first_year = 2 * Matt_books_first_year -- Pete read twice as many books last year as Matt did.

def condition3 (Pete_books_second_year : ℕ) (Pete_books_first_year : ℕ) :=
  Pete_books_second_year = 2 * Pete_books_first_year -- Pete doubles the number of books he read from last year to this year.

-- Theorem: Calculate the total number of books Pete read in both years
theorem Pete_books_total
  (Matt_books_first_year Pete_books_first_year Pete_books_second_year : ℕ)
  (h1 : condition1 Matt_books_first_year)
  (h2 : condition2 Pete_books_first_year Matt_books_first_year)
  (h3 : condition3 Pete_books_second_year Pete_books_first_year)
  : Pete_books_first_year + Pete_books_second_year = 300 := by
  sorry  -- Proof is omitted

end Pete_books_total_l527_527656


namespace ice_skates_solution_l527_527080

noncomputable def problem : Prop :=
  ∃ (a b : ℕ) (x : ℕ),
  2 * a + b = 920 ∧ b = 2 * a ∧
  let profit_A := 400 - a;
      profit_B := 560 - b;
      total_profit := 50 * profit_A + x * (profit_B - profit_A);
      constraint := 50 - x ≤ 2 * x
  in a = 230 ∧ b = 460 ∧ constraint ∧ total_profit = 6190

theorem ice_skates_solution : problem :=
sorry

end ice_skates_solution_l527_527080


namespace sum_log_eq_neg_four_l527_527369

theorem sum_log_eq_neg_four : ∑ n in Finset.range 15, Real.logBase 2 ((n + 1) / (n + 2)) = -4 :=
by
  sorry

end sum_log_eq_neg_four_l527_527369


namespace rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l527_527823

theorem rectangles_on_8x8_chessboard : 
  (Nat.choose 9 2) * (Nat.choose 9 2) = 1296 := by
  sorry

theorem rectangles_on_nxn_chessboard (n : ℕ) : 
  (Nat.choose (n + 1) 2) * (Nat.choose (n + 1) 2) = (n * (n + 1) / 2) * (n * (n + 1) / 2) := by 
  sorry

end rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l527_527823


namespace length_of_longest_side_l527_527337

theorem length_of_longest_side (l w : ℝ) (h_fencing : 2 * l + 2 * w = 240) (h_area : l * w = 8 * 240) : max l w = 96 :=
by sorry

end length_of_longest_side_l527_527337


namespace people_per_car_l527_527137

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) : total_people / num_cars = 21 :=
by
  sorry

end people_per_car_l527_527137


namespace smallest_n_l527_527797

theorem smallest_n (n : ℕ) (h : n > 0) : (\sqrt{n} - \sqrt{n - 1} < 0.02) ↔ n = 626 := 
sorry

end smallest_n_l527_527797


namespace find_a_domain_of_f_monotonicity_of_f_l527_527250

noncomputable def f (x : ℝ) : ℝ := ln ((2 * x) / (1 - x) + 1)

theorem find_a (hf : ∀ x, f (-x) = -f x) : 2 = 2 := 
sorry

theorem domain_of_f {x : ℝ} : (1 + x) / (1 - x) > 0 ↔ -1 < x ∧ x < 1 :=
sorry

theorem monotonicity_of_f {x1 x2 : ℝ} (hx1 : -1 < x1) (hx2 : x2 < 1) (h : x1 < x2) :
  f x1 < f x2 :=
sorry

end find_a_domain_of_f_monotonicity_of_f_l527_527250


namespace transform_M_eq_l527_527982

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1/3], ![1, -2/3]]

def M : Fin 2 → ℚ :=
  ![-1, 1]

theorem transform_M_eq :
  A⁻¹.mulVec M = ![-1, -3] :=
by
  sorry

end transform_M_eq_l527_527982


namespace count_four_digit_numbers_with_4_and_7_l527_527994

def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

def has_digit (d : ℕ) (n : ℕ) : Prop := ∃ i, (n / 10^i) % 10 = d

def has_at_least_one_4_and_7 (n : ℕ) : Prop :=
  has_digit 4 n ∧ has_digit 7 n

theorem count_four_digit_numbers_with_4_and_7 : ∃ count, count = 528 ∧
  (count = (Nat.cardinal_count (λ n, is_four_digit_number n ∧ has_at_least_one_4_and_7 n))) :=
sorry

end count_four_digit_numbers_with_4_and_7_l527_527994


namespace probability_20_correct_l527_527066

noncomputable def probability_sum_20_dodecahedral : ℚ :=
  let num_faces := 12
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_20_correct : probability_sum_20_dodecahedral = 5 / 144 := 
by 
  sorry

end probability_20_correct_l527_527066


namespace area_increase_cost_increase_l527_527157

-- Given definitions based only on the conditions from part a
def original_length := 60
def original_width := 20
def original_fence_cost_per_foot := 15
def original_perimeter := 2 * (original_length + original_width)
def original_fencing_cost := original_perimeter * original_fence_cost_per_foot

def new_fence_cost_per_foot := 20
def new_square_side := original_perimeter / 4
def new_square_area := new_square_side * new_square_side
def new_fencing_cost := original_perimeter * new_fence_cost_per_foot

-- Proof statements using the conditions and correct answers from part b
theorem area_increase : new_square_area - (original_length * original_width) = 400 := by
  sorry

theorem cost_increase : new_fencing_cost - original_fencing_cost = 800 := by
  sorry

end area_increase_cost_increase_l527_527157


namespace ratio_of_cows_to_bulls_l527_527786

-- Define the total number of cattle
def total_cattle := 555

-- Define the number of bulls
def number_of_bulls := 405

-- Compute the number of cows
def number_of_cows := total_cattle - number_of_bulls

-- Define the expected ratio of cows to bulls
def expected_ratio_cows_to_bulls := (10, 27)

-- Prove that the ratio of cows to bulls is equal to the expected ratio
theorem ratio_of_cows_to_bulls : 
  (number_of_cows / (gcd number_of_cows number_of_bulls), number_of_bulls / (gcd number_of_cows number_of_bulls)) = expected_ratio_cows_to_bulls :=
sorry

end ratio_of_cows_to_bulls_l527_527786


namespace find_product_mn_l527_527200

theorem find_product_mn (m n : ℝ) (h_angle : ∀ θ, θ = 3 ∗ Real.atan n → Real.atan(3 ∗ m) = θ) (h_slope : m = 3 ∗ n) (h_nonvertical : L₁ ≠ "vertical") : 
  m * n = 9 / 4 := 
sorry

end find_product_mn_l527_527200


namespace mul_large_numbers_l527_527188

theorem mul_large_numbers : 300000 * 300000 * 3 = 270000000000 := by
  sorry

end mul_large_numbers_l527_527188


namespace factorial_340_trailing_zeros_l527_527774

theorem factorial_340_trailing_zeros : (number_of_trailing_zeros (factorial 340)) = 83 := 
by sorry

def number_of_trailing_zeros (n : ℕ) : ℕ :=
  let f k := n / (5^k)
  in f 1 + f 2 + f 3 -- no need to go further since 5^4 > 340

end factorial_340_trailing_zeros_l527_527774


namespace coloring_impossible_l527_527546

theorem coloring_impossible :
  ∀ (coloring : fin 5 → fin 5 → fin 4), 
  ¬ ∀ (r1 r2 c1 c2 : fin 5), r1 ≠ r2 → c1 ≠ c2 →
  3 ≤ finset.card (finset.image (λ r, coloring r c1) (finset.filter (λ r, r = r1 ∨ r = r2) finset.univ) ∪ 
                   finset.image (λ r, coloring r c2) (finset.filter (λ r, r = r1 ∨ r = r2) finset.univ)) :=
sorry

end coloring_impossible_l527_527546


namespace log_identity_l527_527265

theorem log_identity (a : ℝ) (h : a = log 2 3) : 2^a + 2^(-a) = 10 / 3 :=
by {
  -- proof goes here
  sorry
}

end log_identity_l527_527265


namespace triangle_inequality_maximum_at_centroid_l527_527357

variables {A B C M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space M]
variables (a b c : ℝ) (d_a d_b d_c : ℝ) (S : ℝ)
variables (is_triangle : ∀ (A B C : Type*), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variables (distance_d_a : d_a = dist M BC)
variables (distance_d_b : d_b = dist M CA)
variables (distance_d_c : d_c = dist M AB) 
variables (area_S : S = area_of_triangle ABC)

theorem triangle_inequality : 
  ab * d_a * d_b + bc * d_b * d_c + ca * d_c * d_a ≤ (4 * S^2) / 3 :=
sorry

theorem maximum_at_centroid (is_centroid : M = centroid ABC) : 
  ab * d_a * d_b + bc * d_b * d_c + ca * d_c * d_a = (4 * S^2) / 3 :=
sorry

end triangle_inequality_maximum_at_centroid_l527_527357


namespace sin_cos_product_l527_527641

theorem sin_cos_product (α : ℝ) 
  (h₀ : vector.parallel ⟨[4, 3]⟩ ⟨[Real.sin α, Real.cos α]⟩) : 
  Real.sin α * Real.cos α = 12 / 25 := 
by
  sorry

end sin_cos_product_l527_527641


namespace imaginary_part_of_z_is_minus_5_l527_527617

def complex_mult (a b : ℂ) : ℂ := a * b

theorem imaginary_part_of_z_is_minus_5 (i z : ℂ) (h₁: i^2 = -1) (h₂: z = complex_mult (2 + i) (1 - 3 * i)) :
  complex.Im z = -5 :=
by
  sorry

end imaginary_part_of_z_is_minus_5_l527_527617


namespace problem1_volume_answer_problem2_surface_area_answer_l527_527490

-- Problem 1: Given that the surface area of a sphere is 64π cm², find its volume.
noncomputable def problem1_volume (R : ℝ) (h : 4 * π * R^2 = 64 * π) : ℝ := 
  let V := (4 / 3) * π * R^3 in
  V

theorem problem1_volume_answer (R : ℝ) (h : 4 * π * R^2 = 64 * π) : problem1_volume R h = (256 / 3) * π :=
by
  sorry

-- Problem 2: Given that the volume of a sphere is (500/3)π cm³, find its surface area.
noncomputable def problem2_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (500 / 3) * π) : ℝ := 
  let A := 4 * π * R^2 in
  A

theorem problem2_surface_area_answer (R : ℝ) (h : (4 / 3) * π * R^3 = (500 / 3) * π) : problem2_surface_area R h = 100 * π :=
by
  sorry

end problem1_volume_answer_problem2_surface_area_answer_l527_527490


namespace max_triangles_crossed_l527_527178

theorem max_triangles_crossed (n : ℕ) : 2 * n - 1 =
  let number_triangles_crossed := max_lines_crossing_equilateral_triangle n 
  number_triangles_crossed := 2 * n - 1
:= sorry


end max_triangles_crossed_l527_527178


namespace ellipse_circle_radius_l527_527894

theorem ellipse_circle_radius :
  ∃ a b : ℝ, (∀ r : ℝ, r ∈ set.Ico a b ↔ (∃ x y : ℝ, 
    4 * x^2 + 9 * y^2 = 36 ∧ (x - sqrt 5)^2 + y^2 = r^2 ∧ (x + sqrt 5)^2 + y^2 = r^2 
    ∧ intersects_at_six_points (circle r) (ellipse 4 9 36))) ∧ a + b = 12 := sorry

end ellipse_circle_radius_l527_527894


namespace total_trees_in_gray_areas_l527_527504

theorem total_trees_in_gray_areas (total_trees : ℕ) :
  forall t1 t2 t3, -- assuming t1, t2, t3 are the number of trees in each photo, all equal to total_trees
  t1 = total_trees ->
  t2 = total_trees ->
  t3 = total_trees ->
  t2_white = 100 ->     -- condition 2
  t1_gray = 82 ->       -- condition 3
  t3_white = 90 ->      -- condition 4
  let t2_gray := t2_white - t1_gray in
  let t3_gray := t3_white - t1_gray in
  t2_gray + t3_gray = 26 := 
sorry

end total_trees_in_gray_areas_l527_527504


namespace negB_sufficient_for_A_l527_527350

variables {A B : Prop}

theorem negB_sufficient_for_A (h : ¬A → B) (hnotsuff : ¬(B → ¬A)) : ¬ B → A :=
by
  sorry

end negB_sufficient_for_A_l527_527350


namespace alloy_gold_content_l527_527826

theorem alloy_gold_content (x : ℝ) (w : ℝ) (p0 p1 : ℝ) (h_w : w = 16)
  (h_p0 : p0 = 0.50) (h_p1 : p1 = 0.80) (h_alloy : x = 24) :
  (p0 * w + x) / (w + x) = p1 :=
by sorry

end alloy_gold_content_l527_527826


namespace fraction_of_female_attendees_on_time_l527_527539

theorem fraction_of_female_attendees_on_time (A : ℝ)
  (h1 : 3 / 5 * A = M)
  (h2 : 7 / 8 * M = M_on_time)
  (h3 : 0.115 * A = n_A_not_on_time) :
  0.9 * F = (A - M_on_time - n_A_not_on_time)/((2 / 5) * A - n_A_not_on_time) :=
by
  sorry

end fraction_of_female_attendees_on_time_l527_527539


namespace smallest_n_in_range_l527_527583

theorem smallest_n_in_range : ∃ n : ℕ, n > 1 ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 8 = 2) ∧ 120 ≤ n ∧ n ≤ 149 :=
by
  sorry

end smallest_n_in_range_l527_527583


namespace triangle_subdivision_acute_and_obtuse_l527_527398

theorem triangle_subdivision_acute_and_obtuse (Δ : Triangle) :
  (∃ acute_tris : list Triangle, acute_tris.length = 7 ∧ ∀ t ∈ acute_tris, t.is_acute) ∧
  (∃ obtuse_tris : list Triangle, obtuse_tris.length = 7 ∧ ∀ t ∈ obtuse_tris, t.is_obtuse) :=
sorry

end triangle_subdivision_acute_and_obtuse_l527_527398


namespace exist_circles_with_given_intersecting_angles_l527_527940

variables {P : Point} {e f : Line} {ε φ : Real} 

theorem exist_circles_with_given_intersecting_angles
  (hε : 0 < ε ∧ ε < π / 2) -- ε is an acute angle
  (hφ : 0 < φ ∧ φ < π / 2) -- φ is an acute angle
  : ∃ k₁ k₂ k₃ k₄ : Circle,
      (P ∈ k₁ ∧ P ∈ k₂ ∧ P ∈ k₃ ∧ P ∈ k₄) ∧ 
      (∀ k ∈ {k₁, k₂, k₃, k₄}, 
        (angle k e = ε ∧ angle k f = φ)) :=
sorry

end exist_circles_with_given_intersecting_angles_l527_527940


namespace solution_l527_527045

noncomputable def problem : Prop :=
  ∃ x : ℂ,
  (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 5 ∧
  (x = -(1 / 12) ∨ x = 1 / 2 ∨ x = (5 + complex.I * (√39)) / 24 ∨ x = (5 - complex.I * (√39)) / 24)

theorem solution : problem :=
sorry

end solution_l527_527045


namespace equation_of_line_EF_l527_527638

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -2, y := 0}
def B : Point := {x := -1, y := 0}

-- Define point P on curve C such that |PA| = sqrt(2) * |PB|
def onCurveC (P : Point) : Prop :=
  let PA := (P.x + 2)^2 + P.y^2
  let PB := (P.x + 1)^2 + P.y^2
  PA = 2 * PB

-- Line l: x + y - 4 = 0
def onLineL (Q : Point) : Prop :=
  Q.x + Q.y - 4 = 0

-- Equation of line EF
noncomputable def lineEquationEF (P Q E F : Point) (C : P → Prop) (l : Q → Prop)
  (tangentQtoCE : tangentCondition E C Q)
  (tangentQtoCF : tangentCondition F C Q)
  (minimizeArea : minimizeAreaCondition (Q E F : Point) : Prop := sorry)
  : String :=
  "x + y - 1 = 0"

-- Problem statement
theorem equation_of_line_EF :
  (∀ P : Point, onCurveC P) ∧ 
  (∀ Q : Point, onLineL Q)  ∧ 
  (∀ E F : Point, tangentQtoCE E C Q ∧ tangentQtoCF F C Q ∧ minimizeArea := sorry)  ∧ 
  (lineEquationEF P Q E F (onCurveC) (onLineL) (tangentQtoCE) (tangentQtoCF) (minimizeArea) = "x + y - 1 = 0")
:= sorry

end equation_of_line_EF_l527_527638


namespace angle_between_AO2_and_CO1_is_45_degrees_l527_527325

/-- Given an acute-angled triangle ABC with ∠B = 30° and orthocenter H. 
Let O₁ and O₂ be the incenters of triangles ABH and CBH respectively. 
Prove that the angle between the lines AO₂ and CO₁ is 45°. -/
theorem angle_between_AO2_and_CO1_is_45_degrees
  (A B C H O₁ O₂ : Point)
  (h_acute : is_acute_triangle A B C)
  (h_angle_B : ∠ B = 30)
  (h_orthocenter : orthocenter A B C H)
  (h_incenter_ABH : incenter A B H O₁)
  (h_incenter_CBH : incenter C B H O₂) :
    angle (line_through A O₂) (line_through C O₁) = 45 := sorry

end angle_between_AO2_and_CO1_is_45_degrees_l527_527325


namespace correct_propositions_l527_527867

open Classical

theorem correct_propositions :
  let prop1 := ∀ (a b : ℝ), (a + b ≠ 6) → (a ≠ 3 ∨ b ≠ 3), 
      prop2 := ∀ (p q : Prop), (p ∧ q) → (p ∨ q), 
      prop3 := ∀ (a b : ℝ), (¬(∀ a b : ℝ, a^2 + b^2 ≥ 2 * (a - b - 1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2 * (a - b - 1))), 
  (prop1 = True) ∧ (prop2 = False) ∧ (prop3 = False) → 1 = 1 :=
by
  intros
  sorry

end correct_propositions_l527_527867


namespace tan_sub_pi_over_4_l527_527595

theorem tan_sub_pi_over_4 (α : ℝ) (h1 : π < α) (h2 : α < 3 * π / 2) (h3 : cos α = -4 / 5) :
  tan (π / 4 - α) = 1 / 7 :=
sorry

end tan_sub_pi_over_4_l527_527595


namespace find_x_squared_plus_y_squared_l527_527238

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : x^2 + y^2 = 13 :=
by
  sorry

end find_x_squared_plus_y_squared_l527_527238


namespace minimum_value_a_l527_527970

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + a else x + 4/x

theorem minimum_value_a (a : ℝ) (h_min : ∃ x, ∀ y, f y a ≥ f x a) : a ≥ 4 :=
sorry

end minimum_value_a_l527_527970


namespace parabola_vertex_y_coordinate_l527_527206

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = 5 * x^2 + 20 * x + 45 ∧ (∃ h k, y = 5 * (x + h)^2 + k ∧ k = 25) :=
by
  sorry

end parabola_vertex_y_coordinate_l527_527206


namespace min_colors_correctness_l527_527724

noncomputable def min_colors_no_monochromatic_cycle (n : ℕ) : ℕ :=
if n ≤ 2 then 1 else 2

theorem min_colors_correctness (n : ℕ) (h₀ : n > 0) :
  (min_colors_no_monochromatic_cycle n = 1 ∧ n ≤ 2) ∨
  (min_colors_no_monochromatic_cycle n = 2 ∧ n ≥ 3) :=
by
  sorry

end min_colors_correctness_l527_527724


namespace original_salary_l527_527394

def final_salary_after_changes (S : ℝ) : ℝ :=
  let increased_10 := S * 1.10
  let promoted_8 := increased_10 * 1.08
  let deducted_5 := promoted_8 * 0.95
  let decreased_7 := deducted_5 * 0.93
  decreased_7

theorem original_salary (S : ℝ) (h : final_salary_after_changes S = 6270) : S = 5587.68 :=
by
  -- Proof to be completed here
  sorry

end original_salary_l527_527394


namespace remaining_customers_l527_527861

theorem remaining_customers (initial: ℕ) (left: ℕ) (remaining: ℕ) 
  (h1: initial = 14) (h2: left = 11) : remaining = initial - left → remaining = 3 :=
by {
  sorry
}

end remaining_customers_l527_527861


namespace sum_eight_smallest_multiples_of_12_l527_527110

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end sum_eight_smallest_multiples_of_12_l527_527110


namespace passengers_first_station_l527_527530

noncomputable def initial_passengers := 288
noncomputable def first_station_drop (total: ℕ) := (1 / 3 : ℝ) * total
noncomputable def passengers_at_first_station (total: ℕ) (x: ℕ) := total - nat.floor (first_station_drop total) + x
noncomputable def second_station_drop (total: ℕ) := (1 / 2 : ℝ) * total
noncomputable def passengers_at_second_station (total: ℕ) := total - nat.floor (second_station_drop total) + 12
noncomputable def final_passengers := 248

theorem passengers_first_station (x: ℕ) : 
  let after_first_station := passengers_at_first_station initial_passengers x in
  let after_second_station := passengers_at_second_station after_first_station in
  after_second_station = final_passengers → x = 280 :=
by 
  intros after_first_station after_second_station h
  sorry

end passengers_first_station_l527_527530


namespace find_x_l527_527114

theorem find_x (p q r s x : ℚ) (hpq : p ≠ q) (hq0 : q ≠ 0) 
    (h : (p + x) / (q - x) = r / s) 
    (hp : p = 3) (hq : q = 5) (hr : r = 7) (hs : s = 9) : 
    x = 1/2 :=
by {
  sorry
}

end find_x_l527_527114


namespace polynomial_R_result_l527_527361

noncomputable def polynomial_Q_R (z : ℤ) : Prop :=
  ∃ Q R : Polynomial ℂ, 
  z ^ 2020 + 1 = (z ^ 2 - z + 1) * Q + R ∧ R.degree < 2 ∧ R = 2

theorem polynomial_R_result :
  polynomial_Q_R z :=
by 
  sorry

end polynomial_R_result_l527_527361


namespace line_intersects_circle_two_points_find_value_of_m_l527_527241

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem line_intersects_circle_two_points (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ),
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  x1 ≠ x2 ∨ y1 ≠ y2 := sorry

theorem find_value_of_m (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ), 
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  dist (x1, y1) (x2, y2) = sqrt 17 → 
  m = sqrt 3 ∨ m = -sqrt 3 := sorry

end line_intersects_circle_two_points_find_value_of_m_l527_527241


namespace pair_exchanged_at_least_once_l527_527441

-- The numbers 1, 2, ..., 2006 are written around the circumference of a circle.
-- A move consists of exchanging two adjacent numbers.
-- After a sequence of such moves, each number ends up 13 positions to the right of its initial position.
-- If the numbers 1, 2, ..., 2006 are partitioned into 1003 distinct pairs,
-- then prove that in at least one of the moves, the two numbers of one of the pairs were exchanged.

theorem pair_exchanged_at_least_once :
  ∀ (moves : list (nat × nat)) (pairs : finset (fin 1003 × fin 1003)),
  (∀ i, (i : ℕ) ∈ (list.range 2006) → moves.count i = moves.count (i + 13) % 2006)
  → ∃ (p : fin 1003 × fin 1003), p ∈ pairs ∧ p.1 ∉ moves ∧ p.2 ∉ moves :=
sorry

end pair_exchanged_at_least_once_l527_527441


namespace cleared_land_with_corn_is_630_acres_l527_527733

-- Definitions based on given conditions
def total_land : ℝ := 6999.999999999999
def cleared_fraction : ℝ := 0.90
def potato_fraction : ℝ := 0.20
def tomato_fraction : ℝ := 0.70

-- Calculate the cleared land
def cleared_land : ℝ := cleared_fraction * total_land

-- Calculate the land used for potato and tomato
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := tomato_fraction * cleared_land

-- Define the land planted with corn
def corn_land : ℝ := cleared_land - (potato_land + tomato_land)

-- The theorem to be proved
theorem cleared_land_with_corn_is_630_acres : corn_land = 630 := by
  sorry

end cleared_land_with_corn_is_630_acres_l527_527733


namespace quadratic_reciprocal_roots_sum_min_positive_c_l527_527625

-- Problem 1
theorem quadratic_reciprocal (m n : ℝ) (h : n ≠ 0) :
  (∃ x1 x2 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n) →
  ∃ x1 x2 : ℝ, (x1 = 1 / x1 ∨ x2 = 1 / x2) ∧
    (∃ p q : ℝ, x1^2 + p * x1 + q = 0 ∧ (n * x1^2 + m * x1 + 1 = 0)) :=
sorry

-- Problem 2
theorem roots_sum (a b : ℝ) (h1 : a^2 - 15 * a - 5 = 0) (h2 : b^2 - 15 * b - 5 = 0) :
  a + b = 15 :=
sorry

-- Problem 3
theorem min_positive_c (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 16) :
  c ≥ 4 :=
sorry

end quadratic_reciprocal_roots_sum_min_positive_c_l527_527625


namespace points_connection_through_center_l527_527345

theorem points_connection_through_center (n : ℕ) (h : n > 0) :
  ∀ (circle : set ℝ) (points : finset ℝ),
  (length circle = 6 * n) →
  (points.card = 3 * n) →
  (∀ s ∈ points, (arc_length circle s) = 1 ∨ (arc_length circle s) = 2 ∨ (arc_length circle s) = 3) →
  ∃ (p1 p2 : ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ (midpoint p1 p2) ∈ circle.center :=
sorry

end points_connection_through_center_l527_527345


namespace polygon_properties_l527_527771

theorem polygon_properties
    (each_exterior_angle : ℝ)
    (h1 : each_exterior_angle = 24) :
    ∃ n : ℕ, n = 15 ∧ (180 * (n - 2) = 2340) :=
  by
    sorry

end polygon_properties_l527_527771


namespace two_lines_perpendicular_to_same_plane_are_parallel_l527_527322

-- definitions of the conditions
variable (L₁ L₂ : Type) [line : LinearSpace L₁ L₂]
variable (P1 P2 : Plane) 
variable (l1 l2: Line)

def perpendicular_to_plane (l: Line) (P: Plane) : Prop := 
  -- Define the condition for a line to be perpendicular to a plane
  sorry

def parallel_lines (l1 l2: Line) : Prop := 
  -- Define the condition for two lines to be parallel
  sorry

-- Statement to prove
theorem two_lines_perpendicular_to_same_plane_are_parallel
  (h1 : perpendicular_to_plane l1 P1)
  (h2 : perpendicular_to_plane l2 P1) :
  parallel_lines l1 l2 :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l527_527322


namespace max_volume_of_pyramid_l527_527246

noncomputable def max_volume_pyramid (a : ℝ) : ℝ :=
  𝔐 := a^3 / 6

theorem max_volume_of_pyramid (S A B C H : Type) (SABC : ∀ (x : Mathlib.point), Mathlib.triangle x S A B C)
  (equilateral_base : SABC.base.is_equilateral)
  (orthocenter_H : SABC.is_orthocenter H)
  (SA_eq_a : SABC.distance S A = a) :
  SABC.volume ≤ max_volume_pyramid a :=
by
  sorry

end max_volume_of_pyramid_l527_527246


namespace fraction_squares_sum_l527_527996

theorem fraction_squares_sum (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := 
sorry

end fraction_squares_sum_l527_527996


namespace calculate_Al2O3_weight_and_H2_volume_l527_527886

noncomputable def weight_of_Al2O3 (moles : ℕ) : ℝ :=
  moles * ((2 * 26.98) + (3 * 16.00))

noncomputable def volume_of_H2_at_STP (moles_of_Al2O3 : ℕ) : ℝ :=
  (moles_of_Al2O3 * 3) * 22.4

theorem calculate_Al2O3_weight_and_H2_volume :
  weight_of_Al2O3 6 = 611.76 ∧ volume_of_H2_at_STP 6 = 403.2 :=
by
  sorry

end calculate_Al2O3_weight_and_H2_volume_l527_527886


namespace sequence_is_integer_sequence_l527_527030

-- Lean Definitions of Conditions (a1, a2 ∈ ℤ and is sequence defined by given recurrence)
def seq (a : ℤ) (a1 a2 : ℤ) : ℕ → ℤ 
| 0     := a1
| 1     := a2
| (n+2) := (seq n).succ * (seq (n+1)) + a / (seq n)

theorem sequence_is_integer_sequence (a : ℤ) (a1 a2 : ℤ)
  (h1 : a1 ≠ 0)
  (h2 : a2 ≠ 0)
  (h3 : (a1^2 + a2^2 + a) % (a1 * a2) = 0)
  : ∀ n:ℕ, ∃ m:ℤ, seq a a1 a2 n = m := sorry

end sequence_is_integer_sequence_l527_527030


namespace prove_arithmetic_sequence_l527_527432

def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2 * x + 3
| n => sorry

theorem prove_arithmetic_sequence {x : ℝ} (a : ℕ → ℝ)
  (h_terms : a 0 = x - 1 ∧ a 1 = x + 1 ∧ a 2 = 2 * x + 3)
  (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0)) :
  x = 0 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end prove_arithmetic_sequence_l527_527432


namespace range_of_m_l527_527256
-- Import the entire math library

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0 
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0 

-- Main theorem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) 
(hsuff : ∀ x : ℝ, p x → q x m) 
(hnsuff : ¬ (∀ x : ℝ, q x m → p x)) : m ≥ 9 := 
sorry

end range_of_m_l527_527256


namespace binomial_expansion_constant_term_l527_527680

theorem binomial_expansion_constant_term:
  ∀ (n : ℕ), (0 < n) →
  (binomial (n - 1) 4 = max (λ i, binomial n i)) →
  constant_term ((2 * sqrt x - 1 / sqrt x) ^ n) = 1120 :=
by sorry

end binomial_expansion_constant_term_l527_527680


namespace stock_yield_percentage_l527_527492

theorem stock_yield_percentage
  (annual_dividend : ℝ)
  (market_price : ℝ)
  (face_value : ℝ)
  (yield_percentage : ℝ)
  (H1 : annual_dividend = 0.14 * face_value)
  (H2 : market_price = 175)
  (H3 : face_value = 100)
  (H4 : yield_percentage = (annual_dividend / market_price) * 100) :
  yield_percentage = 8 := sorry

end stock_yield_percentage_l527_527492


namespace sum_le_sqrt_k_equality_condition_l527_527043

theorem sum_le_sqrt_k (k : ℕ) (a : ℕ → ℝ) (h : ∑ i in Finset.range k, (a i)^2 ≤ 1) :
  ∑ i in Finset.range k, a i ≤ Real.sqrt k :=
sorry

theorem equality_condition (k : ℕ) (a : ℕ → ℝ) :
  (∑ i in Finset.range k, a i = Real.sqrt k ↔ ∀ i : ℕ, i < k → a i = 1 / Real.sqrt k ∨ a i = -1 / Real.sqrt k) :=
sorry

end sum_le_sqrt_k_equality_condition_l527_527043


namespace part_one_part_two_l527_527510

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then (16 / (9 - x) - 1) else (11 - (2 / 45) * x ^ 2)

theorem part_one (k : ℝ) (h : 1 ≤ k ∧ k ≤ 4) : k * (16 / (9 - 3) - 1) = 4 → k = 12 / 5 :=
by sorry

theorem part_two (y x : ℝ) (h_y : y = 4) :
  (1 ≤ x ∧ x ≤ 5 ∧ 4 * (16 / (9 - x) - 1) ≥ 4) ∨
  (5 < x ∧ x ≤ 15 ∧ 4 * (11 - (2/45) * x ^ 2) ≥ 4) :=
by sorry

end part_one_part_two_l527_527510


namespace circumradius_bounds_l527_527532

/-- Given a square ABCD with side length 1, points P and Q on side AB, and point R on side CD, 
    the circumradius R of triangle PQR satisfies: 1/2 ≤ R ≤ 1/√2. -/
theorem circumradius_bounds (A B C D P Q R O : ℝ) (h_square : A = 0 ∧ B = 1 ∧ C = 1 ∧ D = 0)
    (h_PQ : 0 ≤ P ∧ P ≤ 1 ∧ 0 ≤ Q ∧ Q ≤ 1) (h_R : 0 ≤ R ∧ R ≤ 1) (triangle_inequality : 
    ∀ {a b c : ℝ}, a + b >= c) :
    1/2 ≤ O ∧ O ≤ 1/√2 :=
  sorry

end circumradius_bounds_l527_527532


namespace allison_remaining_hours_l527_527480

def allison_completion_rate : ℚ := 1/9
def al_completion_rate : ℚ := 1/12
def combined_work_time : ℚ := 3

theorem allison_remaining_hours :
  (combined_work_time * (allison_completion_rate + al_completion_rate)) + (allison_completion_rate * 3.75) = 1 :=
by
  let combined_completion := combined_work_time * (allison_completion_rate + al_completion_rate)
  let remaining_task := 1 - combined_completion
  let remaining_hours := remaining_task / allison_completion_rate
  have h : remaining_hours = 3.75,
  { sorry },
  sorry

end allison_remaining_hours_l527_527480


namespace cube_triangulation_impossible_l527_527887

theorem cube_triangulation_impossible (vertex_sum : ℝ) (triangle_inter_sum : ℝ) (triangle_sum : ℝ) :
  vertex_sum = 270 ∧ triangle_inter_sum = 360 ∧ triangle_sum = 180 → ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), m ≠ 3 → false :=
by
  sorry

end cube_triangulation_impossible_l527_527887


namespace triangle_inequality_range_l527_527635

theorem triangle_inequality_range (x : ℝ) (h1 : 4 + 5 > x) (h2 : 4 + x > 5) (h3 : 5 + x > 4) :
  1 < x ∧ x < 9 := 
by
  sorry

end triangle_inequality_range_l527_527635


namespace possible_values_of_p_l527_527198

noncomputable def count_possible_p : ℕ :=
  let a := 502 in
  let r2_range := (1 : ℕ) to (⌊(502^2 : ℕ) / 4⌋ - 1) in -- Range for r^2
  r2_range.card

theorem possible_values_of_p : count_possible_p = 63000 :=
by sorry

end possible_values_of_p_l527_527198


namespace scientific_notation_of_probe_unit_area_l527_527419

-- Given condition: the area of each probe unit of a certain chip
def probe_unit_area : ℝ := 0.00000164

-- Goal: to prove that 0.00000164 can be expressed in scientific notation as 1.64 * 10^(-6)
theorem scientific_notation_of_probe_unit_area : probe_unit_area = 1.64 * 10^(-6) := 
by
  sorry

end scientific_notation_of_probe_unit_area_l527_527419


namespace hyperbola_asymptote_l527_527632

theorem hyperbola_asymptote (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (P : ℝ × ℝ)
  (hHyperbola : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1) → true)
  (hCircle : (P.1)^2 + (P.2)^2 = a^2)
  (hSlope : P.2 / (P.1 - (sqrt 2 * a)) = -b / a)
  (hFocus : P.1 = sqrt 2 * a) : 
  ∃ m, ∀ x y, y = m * x ↔ (m = 1 ∨ m = -1) := 
by
  sorry

end hyperbola_asymptote_l527_527632


namespace share_money_3_people_l527_527731

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l527_527731


namespace problem_inequality_l527_527612

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 1) : 
  (xy/z + yz/x + zx/y) ≥ sqrt(3) :=
sorry

end problem_inequality_l527_527612


namespace find_original_price_l527_527509

-- Define the necessary variables
def original_price (selling_price : ℝ) (loss_percentage : ℝ) : ℝ :=
  selling_price / (1 - loss_percentage / 100)

-- Problem statement translated to Lean 4
theorem find_original_price :
  original_price 2100 25 = 2800 :=
by
  sorry

end find_original_price_l527_527509


namespace min_value_of_quadratic_l527_527467

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l527_527467


namespace angle_acb_is_60_degrees_l527_527245

theorem angle_acb_is_60_degrees
  (ABC : Triangle)
  (HBC_longest : ABC.BC > ABC.AB ∧ ABC.BC > ABC.AC)
  (L : Point)
  (Hc_intersect : isAngleBisectorIntersectingAltitudeAndCircumcircle ABC.L ABC.C L)
  (Hap_eq_Lq : distance ABC.A ABC.P = distance L ABC.Q) :
  angle ABC.ACB = 60 :=
sorry

end angle_acb_is_60_degrees_l527_527245


namespace probability_of_top_card_heart_l527_527893

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end probability_of_top_card_heart_l527_527893


namespace cos_alpha_value_l527_527286

variables {k : ℕ} (a b c : ℝ^3)
noncomputable def cos_alpha := (9 * k^2 - 37) / 12

open Real

axioms 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = k)
  (h3 : ∥c∥ = 3)
  (h4 : b - a = 2 * (c - b))

theorem cos_alpha_value {α : ℝ} 
  (h5 : α = angle a c) :
  cos α = cos_alpha k := by
  sorry

end cos_alpha_value_l527_527286


namespace maximize_sqrt_k_l527_527240

theorem maximize_sqrt_k (n k : ℕ) (m p q : ℝ) (x : ℕ → ℝ) 
  (hx_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i) 
  (hx_sum : (∑ i in finset.range n, x (i + 1)) = m)
  (hp_pos : 0 < p) (hq_pos : 0 < q) (hk : 2 ≤ k) : 
  ∑ i in finset.range n, (p * x (i + 1) + q) ^ (1 / k : ℝ) ≤ 
    n * ((p * (m / n) + q) ^ (1 / k : ℝ)) := 
sorry

end maximize_sqrt_k_l527_527240


namespace find_special_integers_l527_527222

theorem find_special_integers :
  ∃ count, count = (Finset.range 2018).filter (λ n, 
    (n - 2) * n * (n - 1) * (n - 7) % 7 = 0 ∧
    (n - 2) * n * (n - 1) * (n - 7) % 11 = 0 ∧
    (n - 2) * n * (n - 1) * (n - 7) % 13 = 0).card ∧
  count = 99 := sorry

end find_special_integers_l527_527222


namespace rock_collection_ratio_l527_527667

theorem rock_collection_ratio : 
  ∀ (I S : ℕ), 
  (2/3 * I = 40) → 
  (I + S = 180) → 
  I : S = 1 : 2 :=
by
  intros I S H1 H2
  sorry

end rock_collection_ratio_l527_527667


namespace number_of_valid_sequences_l527_527347

-- Definitions for the conditions described
def is_valid_triplet (x y z : ℕ) : Prop :=
  (1 ≤ x ∧ x ≤ 15) ∧ (1 ≤ y ∧ y ≤ 15) ∧ (1 ≤ z ∧ z ≤ 15)

def generates_zero_sequence (x y z : ℕ) : Prop :=
  ∃ n ≥ 4, let b := λ (n : ℕ), if n = 1 then x else if n = 2 then y else if n = 3 then z else b (n-1) * abs (b (n-2) - b (n-3)) in b n = 0

theorem number_of_valid_sequences : 
  (finset.univ.filter (λ (xyz : ℕ × ℕ × ℕ), is_valid_triplet xyz.1 xyz.2.1 xyz.2.2 ∧ generates_zero_sequence xyz.1 xyz.2.1 xyz.2.2)).card = 1849 :=
sorry

end number_of_valid_sequences_l527_527347


namespace find_b_for_inf_solutions_l527_527591

theorem find_b_for_inf_solutions (x : ℝ) (b : ℝ) : 5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 :=
by
  intro h
  sorry

end find_b_for_inf_solutions_l527_527591


namespace train_cross_bridge_time_l527_527529

noncomputable def length_train : ℝ := 130
noncomputable def length_bridge : ℝ := 320
noncomputable def speed_kmh : ℝ := 54
noncomputable def speed_ms : ℝ := speed_kmh * 1000 / 3600

theorem train_cross_bridge_time :
  (length_train + length_bridge) / speed_ms = 30 := by
  sorry

end train_cross_bridge_time_l527_527529


namespace cone_depth_of_ocean_l527_527841

theorem cone_depth_of_ocean : 
  ∀ (h : ℝ), h = 12000 → 
  ∀ (v_above_ratio : ℝ), v_above_ratio = 1/6 → 
  ∀ (depth : ℝ), depth = h - (h * ((5/6)^(1/3))) → 
  depth = 648 :=
by 
  intros h h_def v_above_ratio v_above_def depth depth_def
  rw [h_def, v_above_def, depth_def]
  norm_num -- Resolving numeric calculations
  sorry -- Placeholder for solution steps

end cone_depth_of_ocean_l527_527841


namespace find_c_for_min_value_zero_l527_527218

theorem find_c_for_min_value_zero :
  ∃ c : ℝ, c = 1 ∧ (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0 → c = 1) :=
by
  use 1
  sorry

end find_c_for_min_value_zero_l527_527218


namespace trigonometric_identity_l527_527741

theorem trigonometric_identity (α : ℝ) (n : ℕ) :
  (finset.range n).sum (λ k, 3^k * (sin (α / 3^(k+1)))^3) = 
  (1 / 4) * (3^n * sin (α / 3^n) - sin α) :=
sorry

end trigonometric_identity_l527_527741


namespace heather_aprons_l527_527292

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l527_527292


namespace find_angle_C_find_area_triangle_l527_527311

open Real

-- Let the angles and sides of the triangle be defined as follows
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom condition1 : (a^2 + b^2 - c^2) * (tan C) = sqrt 2 * a * b
axiom condition2 : c = 2
axiom condition3 : b = 2 * sqrt 2

-- Proof statements
theorem find_angle_C :
  C = pi / 4 ∨ C = 3 * pi / 4 :=
sorry

theorem find_area_triangle :
  C = pi / 4 → a = 2 → (1 / 2) * a * b * sin C = 2 :=
sorry

end find_angle_C_find_area_triangle_l527_527311


namespace walking_speed_l527_527156

theorem walking_speed (W : ℝ) : (1 / (1 / W + 1 / 8)) * 6 = 2.25 * (12 / 2) -> W = 4 :=
by
  intro h
  sorry

end walking_speed_l527_527156


namespace evaluate_expression_l527_527907

theorem evaluate_expression : (3200 - 3131) ^ 2 / 121 = 36 :=
by
  sorry

end evaluate_expression_l527_527907


namespace curve_length_bound_curve_area_bound_l527_527000

variable {K1 K2 : Set (ℝ × ℝ)} -- Two convex curves
variable {L1 L2 L : ℝ}  -- Lengths of K1 and K2 
variable {S1 S2 : ℝ}  -- Areas enclosed by K1 and K2
variable {r : ℝ}  -- Maximum distance between K1 and K2

-- Assume basic properties and hypotheses
axiom convex_K1 : Convex K1
axiom convex_K2 : Convex K2
axiom dist_le_r : ∀ x ∈ K1, ∀ y ∈ K2, dist x y ≤ r
axiom length_K1 : Length K1 = L1
axiom length_K2 : Length K2 = L2
axiom max_length : L = max L1 L2
axiom area_K1 : Area K1 = S1
axiom area_K2 : Area K2 = S2

-- Conclusion containing the proof obligations
theorem curve_length_bound : abs (L2 - L1) ≤ 2 * π * r := by
  sorry

theorem curve_area_bound : abs (S2 - S1) ≤ L * r + π * r^2 := by
  sorry

end curve_length_bound_curve_area_bound_l527_527000


namespace board_divisible_into_hexominos_l527_527839

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end board_divisible_into_hexominos_l527_527839


namespace smallest_value_c_zero_l527_527184

noncomputable def smallest_possible_c (a b c : ℝ) : ℝ :=
if h : (0 < a) ∧ (0 < b) ∧ (0 < c) then
  0
else
  c

theorem smallest_value_c_zero (a b c : ℝ) (h : (0 < a) ∧ (0 < b) ∧ (0 < c)) :
  smallest_possible_c a b c = 0 :=
by
  sorry

end smallest_value_c_zero_l527_527184


namespace problem_statement_l527_527299

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (f (g (f 1))) / (g (f (g 1))) = (-23 : ℝ) / 5 :=
by 
  sorry

end problem_statement_l527_527299


namespace num_packages_to_label_apartments_l527_527068

def digits_appearances (floor_start floor_end : Nat) (digit : Nat) : Nat :=
  let hundreds_place := 30
  let units_place := if digit = 1 then 4 else 0
  let tens_place := if digit = 1 then 3 else 0
  hundreds_place + units_place + tens_place

theorem num_packages_to_label_apartments : ∃ n, n = 111 ∧
  (∀ digit, digits_appearances 101 130 digit + digits_appearances 201 230 digit + digits_appearances 301 330 digit = n) :=
by
  existsi 111
  split
  · rfl
  · intro digit
    simp only
    sorry

end num_packages_to_label_apartments_l527_527068


namespace find_n_l527_527649

theorem find_n (n : ℕ) (h : sqrt (10 + n) = 8) : n = 54 :=
sorry

end find_n_l527_527649


namespace area_under_arccos_is_1_l527_527544

noncomputable def area_under_arccos : ℝ :=
∫ x in 0..1, real.arccos x

theorem area_under_arccos_is_1 :
  area_under_arccos = 1 :=
sorry

end area_under_arccos_is_1_l527_527544


namespace max_value_of_S_l527_527248

-- Define the sequence sum function
def S (n : ℕ) : ℤ :=
  -2 * (n : ℤ) ^ 3 + 21 * (n : ℤ) ^ 2 + 23 * (n : ℤ)

theorem max_value_of_S :
  ∃ (n : ℕ), S n = 504 ∧ 
             (∀ k : ℕ, S k ≤ 504) :=
sorry

end max_value_of_S_l527_527248


namespace petes_total_books_read_l527_527658

-- Let's define the necessary conditions
def matts_books_first_year (M : ℝ) : Prop := true -- M is the number of books Matt read in the first year
def petes_books_first_year (P : ℝ) (M : ℝ) : Prop := P = 2 * M -- Pete read twice as many books as Matt in the first year
def matts_books_second_year (M : ℝ) : Prop := 1.5 * M = 75 -- Matt read 75 books in second year, which is 50% more than the first year
def petes_books_second_year (P : ℝ) : Prop := P * 2 -- Pete doubles his reads from last year in the second year
def total_books_pete_reads (P1 P2 T : ℝ) : Prop := T = P1 + P2 -- sum of books across both years

-- Main theorem
theorem petes_total_books_read (M P1 P2 T : ℝ) :
  matts_books_first_year M →
  petes_books_first_year P1 M →
  matts_books_second_year M →
  petes_books_second_year P2 →
  total_books_pete_reads P1 P2 T →
  T = 300 :=
by
  sorry

end petes_total_books_read_l527_527658


namespace area_of_triangle_formed_by_medians_l527_527029

variable {a b c m_a m_b m_c Δ Δ': ℝ}

-- Conditions from the problem
axiom rel_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = (3 / 4) * (a^2 + b^2 + c^2)
axiom rel_fourth_powers : m_a^4 + m_b^4 + m_c^4 = (9 / 16) * (a^4 + b^4 + c^4)

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_medians :
  Δ' = (3 / 4) * Δ := sorry

end area_of_triangle_formed_by_medians_l527_527029


namespace solution_of_loginequality_l527_527079

-- Define the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := -x + 5 > 0
def condition3 (x : ℝ) : Prop := 2 * x - 1 > -x + 5

-- Define the final solution set
def solution_set (x : ℝ) : Prop := (2 < x) ∧ (x < 5)

-- The theorem stating that under the given conditions, the solution set holds
theorem solution_of_loginequality (x : ℝ) : condition1 x ∧ condition2 x ∧ condition3 x → solution_set x :=
by
  intro h
  sorry

end solution_of_loginequality_l527_527079


namespace arithmetic_and_log_seq_proof_l527_527319

/- Definitions of conditions given in the problem -/

def arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  a 4 = 4 ∧ 
  let Σ (n : ℕ) := (n * (2 * (a 1) + (n - 1) * d)) / 2 in Σ 10 = 55

def log_sum_condition (b : ℕ → ℕ) : Prop :=
  ∀ n, (∑ i in finset.range(n + 1), real.log b (i + 1)) = (n * (n + 1)) / 2

/- Definitions of sequences -/

def a_seq (n : ℕ) : ℕ := n
def b_seq (n : ℕ) : ℕ := 2^n

/- Final theorem to be proved -/

theorem arithmetic_and_log_seq_proof :
  (arithmetic_seq a_seq 1) ∧ (log_sum_condition b_seq) ∧ 
  ∀ n, let T_n := ∑ i in finset.range(n + 1), (((-1)^i) * (a_seq (i+1)) * (b_seq (i+1)))  in 
  T_n = ( -2 - (3 * n + 1) * (-2)^(n+1)) / 9 :=
by {
 sorry
}

end arithmetic_and_log_seq_proof_l527_527319


namespace quadrilateral_ABCD_not_rectangle_length_MN_eq_sqrt_10_over_3_l527_527968

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points A, C, B, and D's conditions
def vertex_A (x y : ℝ) : Prop := x = 0 ∧ y = 1
def vertex_C (x y : ℝ) : Prop := x = 0 ∧ y = -1
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y
def point_B (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the conditions for P, M, N and their equations
def point_P (x y : ℝ) : Prop := ellipse x y ∧ x > 0 ∧ y < 0
def point_M (x₀ y₀ x y : ℝ) : Prop := y = 0 ∧ x = x₀ / (1 - y₀)
def point_N (x₀ y₀ x y : ℝ) : Prop := x = 0 ∧ y = (2 * y₀) / (x₀ + 2)

-- The equation for the condition |BM| = 9/4 * |AN|
def BM_eq_9_over_4_AN (x₀ y₀ : ℝ) : Prop := 
  (x₀ / (1 - y₀)) + 2 = (9 / 4) * (1 - (2 * y₀) / (x₀ + 2))

-- The conclusion |MN| = sqrt(10) / 3
def MN_length (x₀ y₀ x y : ℝ) : Prop := |x| = 1 ∧ |y| = 1 / 3 ∧ abs (|x|^2 + |y|^2) = (1 + 1/3)^2 ∧ sqrt 10 / 3

theorem quadrilateral_ABCD_not_rectangle :
  ∀ (x y : ℝ), point_on_ellipse x y → point_B x y → (vertex_A x 1 ∧ vertex_C x -1) → ¬(x = 2 * y - 1) := sorry

theorem length_MN_eq_sqrt_10_over_3 :
  ∀ (x₀ y₀ x y : ℝ), point_P x₀ y₀ → point_M x₀ y₀ x y → point_N x₀ y₀ x y → BM_eq_9_over_4_AN x₀ y₀ → MN_length x₀ y₀ x y := sorry

end quadrilateral_ABCD_not_rectangle_length_MN_eq_sqrt_10_over_3_l527_527968


namespace log_difference_inequality_l527_527478

open_locale real

theorem log_difference_inequality (a b : ℝ) (h : log a - log b = 3 * b - a) : a > b ∧ b > 0 :=
sorry

end log_difference_inequality_l527_527478


namespace expansion_identity_l527_527294

theorem expansion_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + sqrt 3)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 :=
by
  sorry

end expansion_identity_l527_527294


namespace trapezoid_properties_l527_527764

-- Define the conditions of the problem
structure IsoscelesTrapezoid (BC AD height : ℝ) : Prop :=
(diagonal_perpendicular_to_lateral_side : True)
(smaller_base : BC = 3)
(height_condition : height = 2)

-- Define the theorem we need to prove
theorem trapezoid_properties (BC AD height : ℝ) (h : IsoscelesTrapezoid BC AD height) :
  AD = 5 ∧ ∠ADC = Real.arctan 2 :=
by
  -- Use the given conditions
  have h1 : BC = 3 := h.smaller_base
  have h2 : height = 2 := h.height_condition
  have h3 : True := h.diagonal_perpendicular_to_lateral_side

  -- Prove the desired properties
  sorry

end trapezoid_properties_l527_527764


namespace ratio_of_EG_to_FH_l527_527397

theorem ratio_of_EG_to_FH (EF FG EH EG FH : ℝ) (h1 : EF = 3) (h2 : FG = 8) (h3 : EH = 23) (h4 : EG = EF + FG) (h5 : FH = EH - EF) : EG / FH = 11 / 20 :=
by 
  rw [h1, h2] at h4
  rw [h1, h3] at h5
  rw [←h4, ←h5]
  sorry

end ratio_of_EG_to_FH_l527_527397


namespace modulus_of_z_l527_527267

noncomputable def z : ℂ := 4 / (1 + complex.I)^4 - 3 * complex.I

theorem modulus_of_z : complex.abs z = real.sqrt 10 := by
  sorry

end modulus_of_z_l527_527267


namespace montero_trip_budget_l527_527379

theorem montero_trip_budget :
  let efficiency := 20 -- Normal fuel efficiency
  let efficiency_traffic := 16 -- Fuel efficiency in heavy traffic
  let total_distance := 600
  let heavy_traffic_distance := 100
  let normal_distance := total_distance - heavy_traffic_distance
  let total_gas_needed := normal_distance / efficiency + heavy_traffic_distance / efficiency_traffic
  let initial_gas := 8
  let gas_to_purchase := total_gas_needed - initial_gas
  let current_price := 2.50
  let price_increase := 0.10 * current_price
  let increased_price := current_price + price_increase
  let half_distance := total_distance / 2
  let half_gas_needed := half_distance / efficiency
  let remaining_gas := half_gas_needed - initial_gas
  let remaining_gas_after_half := gas_to_purchase - remaining_gas
  let cost_current_price := remaining_gas * current_price
  let cost_increased_price := remaining_gas_after_half * increased_price
  let total_cost := cost_current_price + cost_increased_price
  total_cost ≤ 75 :=
by
  let efficiency := 20
  let efficiency_traffic := 16
  let total_distance := 600
  let heavy_traffic_distance := 100
  let normal_distance := total_distance - heavy_traffic_distance
  let total_gas_needed := normal_distance / efficiency + heavy_traffic_distance / efficiency_traffic
  let initial_gas := 8
  let gas_to_purchase := total_gas_needed - initial_gas
  let current_price := 2.50
  let price_increase := 0.10 * current_price
  let increased_price := current_price + price_increase
  let half_distance := total_distance / 2
  let half_gas_needed := half_distance / efficiency
  let remaining_gas := half_gas_needed - initial_gas
  let remaining_gas_after_half := gas_to_purchase - remaining_gas
  let cost_current_price := remaining_gas * current_price
  let cost_increased_price := remaining_gas_after_half * increased_price
  let total_cost := cost_current_price + cost_increased_price
  exact sorry

end montero_trip_budget_l527_527379


namespace middle_box_label_l527_527390

theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (prize : Fin 23),
  (∀ i, boxes i = ("There is no prize here" ∨ "The prize is in the neighboring box")) ∧
  (∃ i, boxes i = "The prize is in the neighboring box" ∧ prize = i) ∧
  (∀ j, j ≠ i → boxes j = "There is no prize here") →
  boxes 11 = "The prize is in the neighboring box" :=
by
  sorry

end middle_box_label_l527_527390


namespace cube_divisibility_by_3_l527_527500

theorem cube_divisibility_by_3 (cubes : (Fin 4) × (Fin 4) × (Fin 4) → ℤ) :
  (∀ (i : Fin 4 × Fin 4 × Fin 4), ∃ (j : Fin 4 × Fin 4 × Fin 4), -- Condition to choose a unit cube and increment neighbors
  ... ) →  -- precise conditions to be filled based on the described operations (increment rules)
  -- Assuming an initial condition for white cubes sum not divisible by 3
  (∑ i in (Finset.univ : Finset (Fin 4 × Fin 4 × Fin 4)), if (i.1.val + i.2.val + i.3.val) % 2 = 0 then cubes i else 0) % 3 ≠ 0 →
  -- Conclusion: it's impossible to make all integers divisible by 3 under the conditions
  ¬ ∃ f : (Fin 4) × (Fin 4) × (Fin 4) → ℤ, 
  (∀ i, f i % 3 = 0) := 
sorry

end cube_divisibility_by_3_l527_527500


namespace optionA_is_quadratic_radical_l527_527119

def is_quadratic_radical (n : ℤ) : Prop := ∃ (x : ℤ), n = x * x

def optionA := sqrt 3
def optionB (a : ℤ) := sqrt a
def optionC := (∛ 5)
def optionD := sqrt (- (3 / 5))

theorem optionA_is_quadratic_radical : is_quadratic_radical (int.sqrt 3) :=
sorry

end optionA_is_quadratic_radical_l527_527119


namespace ravi_mobile_purchase_price_l527_527032

theorem ravi_mobile_purchase_price 
  (P : ℝ)
  (purchase_price_refrigerator : ℝ = 15000)
  (selling_price_refrigerator : ℝ = 15000 - 0.03 * 15000)
  (selling_price_mobile : ℝ = P + 0.10 * P)
  (overall_profit : (selling_price_refrigerator + selling_price_mobile) - (15000 + P) = 350) :
  P = 8000 :=
by {
  sorry 
}

end ravi_mobile_purchase_price_l527_527032


namespace probability_unique_tens_correct_l527_527756

noncomputable def probability_unique_tens : ℚ := 
  let favorable_outcomes := 6 * 1 * 45 * 100000 in
  let total_outcomes := Nat.choose 60 7 in
  favorable_outcomes / total_outcomes

theorem probability_unique_tens_correct :
  probability_unique_tens = 6750 / 9655173 := by
  sorry

end probability_unique_tens_correct_l527_527756


namespace second_tree_ring_groups_l527_527186

-- Definition of the problem conditions
def group_rings (fat thin : Nat) : Nat := fat + thin

-- Conditions
def FirstTreeRingGroups : Nat := 70
def RingsPerGroup : Nat := group_rings 2 4
def FirstTreeRings : Nat := FirstTreeRingGroups * RingsPerGroup
def AgeDifference : Nat := 180

-- Calculate the total number of rings in the second tree
def SecondTreeRings : Nat := FirstTreeRings - AgeDifference

-- Prove the number of ring groups in the second tree
theorem second_tree_ring_groups : SecondTreeRings / RingsPerGroup = 40 :=
by
  sorry

end second_tree_ring_groups_l527_527186


namespace relationship_between_b_and_c_l527_527634

-- Definitions based on the given conditions
def y1 (x a b : ℝ) : ℝ := (x + 2 * a) * (x - 2 * b)
def y2 (x b : ℝ) : ℝ := -x + 2 * b
def y (x a b : ℝ) : ℝ := y1 x a b + y2 x b

-- Lean theorem for the proof problem
theorem relationship_between_b_and_c
  (a b c : ℝ)
  (h : a + 2 = b)
  (h_y : y c a b = 0) :
  c = 5 - 2 * b ∨ c = 2 * b :=
by
  -- The proof will go here, currently omitted
  sorry

end relationship_between_b_and_c_l527_527634


namespace max_groups_l527_527207

theorem max_groups (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) : Nat.gcd boys girls = 20 := 
  by
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end max_groups_l527_527207


namespace find_k_l527_527288

variables 
  {V : Type} [AddCommGroup V] [Module ℝ V]
  (e1 e2 a b : V)
  (k : ℝ)

-- Conditions
axiom non_collinear : ¬ Collinear ℝ ({e1, e2} : Set V)
def vec_a : V := 3 • e1 - 4 • e2
def vec_b : V := 6 • e1 + k • e2
axiom a_parallel_b : ∃ λ : ℝ, vec_a = λ • vec_b

-- Proof statement
theorem find_k : k = -8 :=
sorry

end find_k_l527_527288


namespace angle_AC1B_l527_527943

theorem angle_AC1B (α : ℝ) (A B C B1 A1 C1 : EuclideanGeometry.Point)
  (h1 : EuclideanGeometry.AcuteAngleTriangle A B C)
  (h2 : EuclideanGeometry.IsoscelesTriangleAngleBisector A B1 C α)
  (h3 : EuclideanGeometry.IsoscelesTriangleAngleBisector B A1 C α)
  (h4 : EuclideanGeometry.PerpendicularFromPointToLine C A1 B1 C1)
  (h5 : EuclideanGeometry.PerpendicularBisectorOnPoint A B C1) :
  EuclideanGeometry.Angle A C1 B = 2 * α :=
sorry

end angle_AC1B_l527_527943


namespace find_a_plus_b_l527_527631

theorem find_a_plus_b (f : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = (x - b) / (x + 2))
  (hrange : set.range (λ x, f x) = set.Ici 2) 
  (ha_interval: ∀ x, x ∈ (a, a + 6) → f x ∈ set.Ici 2)
  (hb: b < -2) :
  a + b = -10 :=
sorry

end find_a_plus_b_l527_527631


namespace price_per_bag_l527_527177

open Nat

theorem price_per_bag (total_weight : ℕ) (damaged_weight : ℕ) (bag_weight : ℕ) 
  (total_sale : ℕ) (total_weight_val : total_weight = 6500) 
  (damaged_weight_val : damaged_weight = 150)
  (bag_weight_val : bag_weight = 50)
  (total_sale_val : total_sale = 9144) :
  (total_sale / ((total_weight - damaged_weight) / bag_weight) = 72) :=
by
  rw [total_weight_val, damaged_weight_val, bag_weight_val, total_sale_val]
  sorry

end price_per_bag_l527_527177


namespace no_common_tangent_perpendicular_point_l527_527988

theorem no_common_tangent_perpendicular_point :
  ¬ ∃ x : ℝ, (sin x = cos x) ∧ (-sin x * cos x = -1) :=
by
  sorry

end no_common_tangent_perpendicular_point_l527_527988


namespace orange_juice_fraction_l527_527099

theorem orange_juice_fraction :
  let pitcher_capacity := 800
  let first_pitcher_oj := pitcher_capacity * (1 / 4 : ℝ)
  let second_pitcher_oj := pitcher_capacity * (1 / 3 : ℝ)
  let total_oj := first_pitcher_oj + second_pitcher_oj
  let total_volume := 2 * pitcher_capacity
  (total_oj / total_volume = 29167 / 100000) :=
by {
  let pitcher_capacity := 800
  let first_pitcher_oj := pitcher_capacity * (1 / 4 : ℝ)
  let second_pitcher_oj := pitcher_capacity * (1 / 3 : ℝ)
  let total_oj := first_pitcher_oj + second_pitcher_oj
  let total_volume := 2 * pitcher_capacity
  show total_oj / total_volume = 29167 / 100000,
  sorry
}

end orange_juice_fraction_l527_527099


namespace point_corresponding_to_conjugate_is_in_first_quadrant_l527_527681

-- Definition: the conditions as given
def Z : ℂ := (7 + complex.I) / (3 + 4 * complex.I)

-- Problem statement: Prove the point corresponding to the complex number conjugate is in the first quadrant
theorem point_corresponding_to_conjugate_is_in_first_quadrant : 
    let Z_conjugate := complex.conj Z in
    0 < Z_conjugate.re ∧ 0 < Z_conjugate.im := 
by
    sorry

end point_corresponding_to_conjugate_is_in_first_quadrant_l527_527681


namespace sum_equal_l527_527275

def f (x : ℝ) : ℝ := 1/2 + Real.logb 2 (x / (1 - x))

def S (n : ℕ) : ℝ := ∑ k in Finset.range (n - 1), f (k.succ / n)

theorem sum_equal (n : ℕ) (h2 : 2 ≤ n) : S n = n / 2 := by
  sorry

end sum_equal_l527_527275


namespace reach_one_after_seven_steps_l527_527753

def iterDivideBy2 (n : ℕ) : ℕ := nat.floor (n / 2.0)

theorem reach_one_after_seven_steps (n : ℕ) : (n = 150) → (iterDivideBy2 (iterDivideBy2 (iterDivideBy2 (iterDivideBy2 (iterDivideBy2 (iterDivideBy2 (iterDivideBy2 n))))))) = 1 := 
by {
  intro h,
  rw [h],
  sorry
}

end reach_one_after_seven_steps_l527_527753


namespace factorial_340_trailing_zeros_l527_527775

theorem factorial_340_trailing_zeros : (number_of_trailing_zeros (factorial 340)) = 83 := 
by sorry

def number_of_trailing_zeros (n : ℕ) : ℕ :=
  let f k := n / (5^k)
  in f 1 + f 2 + f 3 -- no need to go further since 5^4 > 340

end factorial_340_trailing_zeros_l527_527775


namespace maximum_value_l527_527974

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_value (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f1 : f a b c 1 ≤ 1) (h_f2 : f a b c 2 ≤ 1) :
  7 * b + 5 * c ≤ -6 :=
sorry

end maximum_value_l527_527974


namespace least_number_of_marbles_l527_527523

theorem least_number_of_marbles :
  ∃ n, (∀ k ∈ {3, 4, 6, 7, 8}, n % k = 0) ∧ n = 168 :=
by {
  sorry
}

end least_number_of_marbles_l527_527523


namespace minimum_intersection_l527_527049

theorem minimum_intersection (A B C : Finset α) (hA : A.card = 7) (hB : B.card = 7) (hC : C.card = 6) 
  (h_union : (Finset.card A) + (Finset.card B) + (Finset.card C) = (A ∪ B ∪ C).card) 
  : (A ∩ B ∩ C).card = 1 :=
sorry

end minimum_intersection_l527_527049


namespace no_solution_eq_for_prime_gt_5_l527_527399

-- Define the problem conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Main theorem statement
theorem no_solution_eq_for_prime_gt_5 (p : ℕ) (h_prime : is_prime p) (h_gt_5 : p > 5) : 
  ¬ ∃ (x : ℤ), x^4 + 4^x = p := by
  sorry

end no_solution_eq_for_prime_gt_5_l527_527399


namespace contrapositive_l527_527763

-- Definitions based on the conditions
def original_proposition (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem to prove the contrapositive
theorem contrapositive (a b : ℝ) : original_proposition a b ↔ (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_l527_527763


namespace integer_solutions_x2_minus_y2_equals_12_l527_527072

theorem integer_solutions_x2_minus_y2_equals_12 : 
  ∃! (s : Finset (ℤ × ℤ)), (∀ (xy : ℤ × ℤ), xy ∈ s → xy.1^2 - xy.2^2 = 12) ∧ s.card = 4 :=
sorry

end integer_solutions_x2_minus_y2_equals_12_l527_527072


namespace binom_n_choose_n_minus_2_l527_527793

-- Additional definitions and imports can be added as necessary

theorem binom_n_choose_n_minus_2 (n : ℕ) (h : n ≥ 2) : nat.choose n (n-2) = n * (n-1) / 2 := 
sorry

end binom_n_choose_n_minus_2_l527_527793


namespace system1_solution_system2_solution_l527_527752

-- System 1 Definitions
def eq1 (x y : ℝ) : Prop := 3 * x - 2 * y = 9
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y = 19

-- System 2 Definitions
def eq3 (x y : ℝ) : Prop := (2 * x + 1) / 5 - 1 = (y - 1) / 3
def eq4 (x y : ℝ) : Prop := 2 * (y - x) - 3 * (1 - y) = 6

-- Theorem Statements
theorem system1_solution (x y : ℝ) : eq1 x y ∧ eq2 x y ↔ x = 5 ∧ y = 3 := by
  sorry

theorem system2_solution (x y : ℝ) : eq3 x y ∧ eq4 x y ↔ x = 4 ∧ y = 17 / 5 := by
  sorry

end system1_solution_system2_solution_l527_527752


namespace find_center_of_given_circle_l527_527901

noncomputable def center_of_circle (equation : (ℝ → ℝ → Prop)) : ℝ × ℝ :=
  let (h, k, r) := (* Any necessary computations to derive h, k, r from the equation *) in
  (h, k)

theorem find_center_of_given_circle :
  center_of_circle (λ x y, (x + 2) ^ 2 + y ^ 2 = 5) = (-2, 0) :=
sorry

end find_center_of_given_circle_l527_527901


namespace trig_expression_eval_l527_527941

noncomputable def trig_expression_val (α : ℝ) (r : ℝ) (hr : r ≠ 0) 
(pt : (-4 * r, 3 * r) = (cos α, sin α)): ℝ := 
  ((cos (π / 2 + α)) * (sin (-π - α)) * (cos (2 * π - α))) / 
  ((cos (11 * π / 2 - α)) * (sin (9 * π / 2 + α)))

theorem trig_expression_eval (α : ℝ) (r : ℝ) (hr : r ≠ 0) 
(pt : (-4 * r, 3 * r) = (cos α, sin α)) :
  trig_expression_val α r hr pt = ±(3 / 5) :=
sorry

end trig_expression_eval_l527_527941


namespace area_in_scientific_notation_l527_527414

theorem area_in_scientific_notation {a : ℝ} 
  (h : a = 0.00000164) : 
  a = 1.64 * 10 ^ (-6) :=
sorry

end area_in_scientific_notation_l527_527414


namespace find_y_l527_527300

theorem find_y (x y : ℝ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1 / 10 := by
  sorry

end find_y_l527_527300


namespace complex_number_under_dilation_l527_527503

noncomputable def original_complex (a : ℂ) (b : ℂ) (s : ℂ) (c : ℂ) : ℂ :=
  (c - b) / s + a

theorem complex_number_under_dilation :
  ∃ z : ℂ, (dilation (2 - 3 * Complex.i) 3 z = 1 + 2 * Complex.i) ↔ z = (5 - 4 * Complex.i) / 3 :=
by
  sorry

end complex_number_under_dilation_l527_527503


namespace total_width_of_all_pathways_l527_527158

theorem total_width_of_all_pathways (w : ℝ) :
  (∃ (length width : ℝ), length = 60 ∧ width = 40 ∧ 
   (∃ (path_area lawn_area : ℝ), path_area = 100 * w ∧ lawn_area = 2400 - path_area ∧ lawn_area = 2109) 
   → w = 2.91) :=
begin
  sorry
end

end total_width_of_all_pathways_l527_527158


namespace inequality_proof_l527_527646

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c :=
sorry

end inequality_proof_l527_527646


namespace triangle_angles_l527_527554

-- Defining a structure for a triangle with angles
structure Triangle :=
(angleA angleB angleC : ℝ)

-- Define the condition for the triangle mentioned in the problem
def triangle_condition (t : Triangle) : Prop :=
  ∃ (α : ℝ), α = 22.5 ∧ t.angleA = 90 ∧ t.angleB = α ∧ t.angleC = 67.5

theorem triangle_angles :
  ∃ (t : Triangle), triangle_condition t :=
by
  -- The proof outline
  -- We need to construct a triangle with the given angle conditions
  -- angleA = 90°, angleB = 22.5°, angleC = 67.5°
  sorry

end triangle_angles_l527_527554


namespace solution_proof_l527_527422

noncomputable def average_age_of_women_and_new_group_age
  (A : ℝ) -- original average age of 12 men
  (four_men_ages : Fin 4 → ℝ) -- ages of the four men
  (boy_age : ℝ) -- age of the boy
  (women_ages : Fin 5 → ℝ) -- function for the ages of the 5 women
  (new_avg_increase : ℝ) -- increase in the average age of the group
  (h_age_sum_eq : ∑ i, four_men_ages i + boy_age = 101) -- total age of the 4 men and 1 boy is 101 years
  (h_new_avg_eq : new_avg_increase = 3.5) : Prop :=
  let total_age_of_women := ∑ i, women_ages i in
  let avg_age_of_women := total_age_of_women / 5 in
  avg_age_of_women = 28.6 ∧ new_avg_increase = 3.5

theorem solution_proof : average_age_of_women_and_new_group_age 
  (A : ℝ) 
  (λ i, [15, 20, 25, 30].nth_le i sorry) 
  11 
  (λ i, [/* age of the 5 women */].nth_le i sorry)
  3.5 
  by {
    apply Eq.refl,
    simp,
  } :
    Prop sorry

end solution_proof_l527_527422


namespace probability_event_proof_l527_527303

noncomputable def probability_event_occur (deck_size : ℕ) (num_queens : ℕ) (num_jacks : ℕ) (num_reds : ℕ) : ℚ :=
  let prob_two_queens := (num_queens / deck_size) * ((num_queens - 1) / (deck_size - 1))
  let prob_at_least_one_jack := 
    (num_jacks / deck_size) * ((deck_size - num_jacks) / (deck_size - 1)) +
    ((deck_size - num_jacks) / deck_size) * (num_jacks / (deck_size - 1)) +
    (num_jacks / deck_size) * ((num_jacks - 1) / (deck_size - 1))
  let prob_both_red := (num_reds / deck_size) * ((num_reds - 1) / (deck_size - 1))
  prob_two_queens + prob_at_least_one_jack + prob_both_red

theorem probability_event_proof :
  probability_event_occur 52 4 4 26 = 89 / 221 :=
by
  sorry

end probability_event_proof_l527_527303


namespace min_fencing_cost_l527_527779

theorem min_fencing_cost {A B C : ℕ} (h1 : A = 25) (h2 : B = 35) (h3 : C = 40)
  (h_ratio : ∃ (x : ℕ), 3 * x * 4 * x = 8748) : 
  ∃ (total_cost : ℝ), total_cost = 87.75 :=
by
  sorry

end min_fencing_cost_l527_527779


namespace given_cond_then_geq_eight_l527_527597

theorem given_cond_then_geq_eight (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 1) : 
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := 
  sorry

end given_cond_then_geq_eight_l527_527597


namespace greatest_monthly_drop_in_March_l527_527787

noncomputable def jan_price_change : ℝ := -3.00
noncomputable def feb_price_change : ℝ := 1.50
noncomputable def mar_price_change : ℝ := -4.50
noncomputable def apr_price_change : ℝ := 2.00
noncomputable def may_price_change : ℝ := -1.00
noncomputable def jun_price_change : ℝ := 0.50

theorem greatest_monthly_drop_in_March :
  mar_price_change < jan_price_change ∧
  mar_price_change < feb_price_change ∧
  mar_price_change < apr_price_change ∧
  mar_price_change < may_price_change ∧
  mar_price_change < jun_price_change :=
by {
  sorry
}

end greatest_monthly_drop_in_March_l527_527787


namespace area_in_scientific_notation_l527_527415

theorem area_in_scientific_notation {a : ℝ} 
  (h : a = 0.00000164) : 
  a = 1.64 * 10 ^ (-6) :=
sorry

end area_in_scientific_notation_l527_527415


namespace red_points_at_least_1991_l527_527054

theorem red_points_at_least_1991 (n : ℕ) (h₁ : n = 997)
  (points : Fin n → ℝ × ℝ)
  (midpoints : (Fin n → ℝ × ℝ) × (Fin n → ℝ × ℝ) →  Set (ℝ × ℝ))
  (h₂ : ∀ (P Q : Fin n → ℝ × ℝ), midpoint P Q ∈ midpoints):
  ∃ red_points : Set (ℝ × ℝ), 
  (∀ (P Q : Fin n → ℝ × ℝ), midpoint P Q ∈ red_points) 
  ∧ (red_points.card ≥ 1991) :=
sorry

end red_points_at_least_1991_l527_527054


namespace find_abc_triplet_l527_527611

theorem find_abc_triplet (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a < b ∧ b < c) 
  (h_eqn : (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = (a + b + c) / 2) :
  ∃ d : ℕ, d > 0 ∧ ((a = d ∧ b = 2 * d ∧ c = 3 * d) ∨ (a = d ∧ b = 3 * d ∧ c = 6 * d)) :=
  sorry

end find_abc_triplet_l527_527611


namespace ferry_journey_time_difference_l527_527232

theorem ferry_journey_time_difference:
  let speed_P := 6 in
  let time_P := 3 in
  let distance_P := speed_P * time_P in
  let distance_Q := 3 * distance_P in
  let speed_Q := speed_P + 3 in
  let time_Q := distance_Q / speed_Q in
  time_Q - time_P = 3 :=
by
  sorry

end ferry_journey_time_difference_l527_527232


namespace original_price_of_car_l527_527694

-- Let P be the original price of the car
variable (P : ℝ)

-- Condition: The car's value is reduced by 30%
-- Condition: The car's current value is $2800, which means 70% of the original price
def car_current_value_reduced (P : ℝ) : Prop :=
  0.70 * P = 2800

-- Theorem: Prove that the original price of the car is $4000
theorem original_price_of_car (P : ℝ) (h : car_current_value_reduced P) : P = 4000 := by
  sorry

end original_price_of_car_l527_527694


namespace sqrt_x6_plus_x4_eq_abs_x_squared_sqrt_x2_plus_1_l527_527229

theorem sqrt_x6_plus_x4_eq_abs_x_squared_sqrt_x2_plus_1 (x : ℝ) : 
  sqrt (x^6 + x^4) = (|x|^2) * sqrt (x^2 + 1) := 
  sorry

end sqrt_x6_plus_x4_eq_abs_x_squared_sqrt_x2_plus_1_l527_527229


namespace angle_sum_less_than_90_l527_527674

-- Definitions based on the conditions
variables {A B C O P : Type*}
  [triangle : acute_angled_triangle A B C]
  [circumcenter O A B C]
  [foot_perpendicular P A B C]
  (h_angle_condition : angle B C A ≥ angle A B C + 30)

-- Theorem statement to be proved
theorem angle_sum_less_than_90 : angle A C B + angle O O P < 90 := 
sorry

end angle_sum_less_than_90_l527_527674


namespace sum_of_sequence_l527_527942

theorem sum_of_sequence (n : ℕ) : 
  let a : ℕ → ℕ := λ n, nat.rec_on n 2 (λ k ak, ak + 2 * (k + 1))
in
  let S : ℕ → ℕ := λ n, (∑ i in finset.range n, a i.succ)
in
  S n = (n * (n + 1) * (2 * n + 1)) / 6 - (n * (n + 1)) / 2 + 2 * n :=
by
  sorry

end sum_of_sequence_l527_527942


namespace range_of_power_function_l527_527977

open Set

variable {m : ℝ}

theorem range_of_power_function (hm : m > 0) : range (λ x : ℝ, x ^ m) ∩ Ici 0.5 = Ici (0.5 ^ m) :=
by
  sorry

end range_of_power_function_l527_527977


namespace sum_of_coordinates_of_A_l527_527346

open Real

theorem sum_of_coordinates_of_A (A B C : ℝ × ℝ) (h1 : B = (2, 8)) (h2 : C = (5, 2))
  (h3 : ∃ (k : ℝ), A = ((2 * (B.1:ℝ) + C.1) / 3, (2 * (B.2:ℝ) + C.2) / 3) ∧ k = 1/3) :
  A.1 + A.2 = 9 :=
sorry

end sum_of_coordinates_of_A_l527_527346


namespace mixture_weight_l527_527485

theorem mixture_weight (weight_a weight_b : ℕ) (total_volume ratio_a ratio_b : ℕ)
                       (h_a : weight_a = 800) (h_b : weight_b = 850)
                       (h_vol : total_volume = 3) (h_ratio_a : ratio_a = 3) (h_ratio_b : ratio_b = 2) :
  (ratio_a * weight_a * total_volume / (ratio_a + ratio_b) + ratio_b * weight_b * total_volume / (ratio_a + ratio_b)) / 1000 = 2.46 :=
by
  sorry

end mixture_weight_l527_527485


namespace problem_a_l527_527711

variable {S : Type*}
variables (a b : S)
variables [Inhabited S] -- Ensures S has at least one element
variables (op : S → S → S) -- Defines the binary operation

-- Condition: binary operation a * (b * a) = b holds for all a, b in S
axiom binary_condition : ∀ a b : S, op a (op b a) = b

-- Theorem to prove: (a * b) * a ≠ a
theorem problem_a : (op (op a b) a) ≠ a :=
sorry

end problem_a_l527_527711


namespace expr_value_at_neg2_l527_527115

variable (a b : ℝ)

def expr (x : ℝ) : ℝ := a * x^3 + b * x - 7

theorem expr_value_at_neg2 :
  (expr a b 2 = -19) → (expr a b (-2) = 5) :=
by 
  intro h
  sorry

end expr_value_at_neg2_l527_527115


namespace log_intersection_distance_l527_527508

theorem log_intersection_distance (a b : ℤ) (k : ℝ) (h_k : k = a + Real.sqrt b)
  (h_intersect : (∀ x : ℝ, (x = k) → (y = log 2 x ∧ y = log 2 (x - 6))))
  (h_distance : abs (log 2 k - log 2 (k - 6)) = 0.7) :
  a + b = 232 := by
  sorry

end log_intersection_distance_l527_527508


namespace game_ends_after_finite_moves_daphne_has_winning_strategy_l527_527192

def card_game_finite (n : ℕ) : Prop :=
  ∀ cards : ℕ, cards = n → 
    (∃ (max_moves : ℕ), 
      ∀ moves : ℕ, moves > max_moves → False)

def card_game_strategy (n m : ℕ) : Prop :=
  ∃ (strategy : ℕ → bool),
    ∀ k : ℕ, (k < m) → strategy k = 
      (k % 2 = 1 → True)

theorem game_ends_after_finite_moves : card_game_finite 2022 := 
by {
  sorry,
}

theorem daphne_has_winning_strategy : card_game_strategy 2022 100 := 
by {
  sorry,
}

end game_ends_after_finite_moves_daphne_has_winning_strategy_l527_527192


namespace students_without_pens_l527_527088

theorem students_without_pens (total_students blue_pens red_pens both_pens : ℕ)
  (h_total : total_students = 40)
  (h_blue : blue_pens = 18)
  (h_red : red_pens = 26)
  (h_both : both_pens = 10) :
  total_students - (blue_pens + red_pens - both_pens) = 6 :=
by
  sorry

end students_without_pens_l527_527088


namespace percentage_of_students_absent_l527_527449

theorem percentage_of_students_absent (total_students : ℕ) (students_present : ℕ) 
(h_total : total_students = 50) (h_present : students_present = 43)
(absent_students := total_students - students_present) :
((absent_students : ℝ) / total_students) * 100 = 14 :=
by sorry

end percentage_of_students_absent_l527_527449


namespace chandra_valid_pairings_l527_527547

noncomputable def valid_pairings (total_items : Nat) (invalid_pairing : Nat) : Nat :=
total_items * total_items - invalid_pairing

theorem chandra_valid_pairings : valid_pairings 5 1 = 24 := by
  sorry

end chandra_valid_pairings_l527_527547


namespace aprons_to_sew_tomorrow_l527_527290

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l527_527290


namespace intersection_complement_l527_527987

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of B with respect to U
def comp_B : Set ℕ := U \ B

-- Statement to be proven
theorem intersection_complement : A ∩ comp_B = {1, 3} :=
by 
  sorry

end intersection_complement_l527_527987


namespace no_possible_1986_sections_l527_527843

theorem no_possible_1986_sections : ¬ ∃ n : ℕ, 8 * n + 9 = 1986 :=
by {
  intro h,
  cases h with n hn,
  have h_divisible : 8 * n = 1977,
  { linarith },
  have h_not_divisible : ¬ ∃ k, 8 * k = 1977,
  { intro k,
    cases k with k hk,
    norm_num at hk,
    contradiction },
  contradiction,
}

end no_possible_1986_sections_l527_527843


namespace line_equation_l527_527621

def circle : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 100 }

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem line_equation
  (l : ℝ → ℝ) -- representing line l as a function y = l(x)
  (h_intersects : ∃ A B : ℝ × ℝ, A ∈ circle ∧ B ∈ circle ∧ midpoint A B = (-2,3))
  (h_line : ∀ x, l x = x + 5) :
  ∀ x y, y = l x ↔ x - y + 5 = 0 :=
by
  sorry

end line_equation_l527_527621


namespace not_true_16_divides_n_l527_527354

theorem not_true_16_divides_n (n : ℕ) (h1 : 0 < n) (h2 : (1 / 2 + 1 / 4 + 1 / 8 + 1 / n : ℚ).denom = 1) : (¬ (16 ∣ n)) :=
by
  sorry

end not_true_16_divides_n_l527_527354


namespace range_of_a_l527_527972

theorem range_of_a (a : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ 3 * x1^2 + a = 0 ∧ 3 * x2^2 + a = 0) : a < 0 :=
sorry

end range_of_a_l527_527972


namespace sum_of_sides_of_30_60_90_triangle_l527_527024

theorem sum_of_sides_of_30_60_90_triangle {A B C : Type}
  (angle_C : ℝ) (angle_B : ℝ) (side_AC : ℝ) :
  angle_C = 60 ∧ angle_B = 30 ∧ side_AC = 6 →
  (let AB := 3 in let BC := 3 * Real.sqrt 3 in (AB + BC) ≈ 8.2) :=
begin
  -- convert angles to radians for the Lean environment if needed
  sorry
end

end sum_of_sides_of_30_60_90_triangle_l527_527024


namespace min_value_of_a2_b2_l527_527263

theorem min_value_of_a2_b2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : 
  ∃ m : ℝ, (∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m) ∧ m = 8 :=
by
  sorry

end min_value_of_a2_b2_l527_527263


namespace find_ratio_of_a_and_b_l527_527003

variable {a b : ℝ}

noncomputable def A := (a + b) / 2
noncomputable def B := 2 / (1/a + 1/b)

theorem find_ratio_of_a_and_b (hpos_a : a > 0) (hpos_b : b > 0) (h_eq : A + B = a - b) : a / b = 3 + 2 * Real.sqrt 3 :=
by sorry

end find_ratio_of_a_and_b_l527_527003


namespace average_of_two_numbers_l527_527059

theorem average_of_two_numbers (A B C : ℝ) (h1 : (A + B + C)/3 = 48) (h2 : C = 32) : (A + B)/2 = 56 := by
  sorry

end average_of_two_numbers_l527_527059


namespace no_solution_l527_527553

open Nat

theorem no_solution (x y z : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4 + y ^ 4 = 2 * z ^ 2 ∧ gcd x y = 1 → false :=
by
  intro h
  sorry

end no_solution_l527_527553


namespace samuel_bought_more_apples_l527_527187

-- Definitions based on the conditions
def bonnie_apples := 8
variable (s : ℕ)  -- Samuel's initial number of apples

-- Samuel's conditions
def samuel_bought_more_than_bonnie := s > bonnie_apples
def samuel_ate_half := s / 2
def samuel_used_for_pie := s / 7
def samuel_remaining := 10

-- Theorem statement
theorem samuel_bought_more_apples : samuel_bought_more_than_bonnie ∧ (s - samuel_ate_half - samuel_used_for_pie = samuel_remaining) → s - bonnie_apples = 20 :=
by
  sorry

end samuel_bought_more_apples_l527_527187


namespace anahi_written_percentage_l527_527876

-- Define conditions
def total_pages := 500
def pages_first_week := 150
def pages_remaining_after_first_week := total_pages - pages_first_week
def percentage_written_second_week (P : ℝ) := P / 100 * pages_remaining_after_first_week

-- Define the function for the remaining pages after the second week
def remaining_after_second_week (P : ℝ) := pages_remaining_after_first_week - percentage_written_second_week P

-- Define the function for damaged pages
def damaged_pages (P : ℝ) := 0.2 * remaining_after_second_week P

-- Final empty pages after damage
def final_empty_pages (P : ℝ) := remaining_after_second_week P - damaged_pages P

-- The target statement to prove
theorem anahi_written_percentage : 
  ∃ P : ℝ, final_empty_pages P = 196 ∧ P = 30 :=
by
  sorry

end anahi_written_percentage_l527_527876


namespace sum_of_series_l527_527213

noncomputable def sum_expression : ℂ :=
  let i := Complex.i
  let rec sum_loop (n : ℕ) (acc : ℂ) : ℂ :=
    if n > 2010 then acc
    else
      let coef := n + 1
      sum_loop (n + 1) (acc + coef * (i ^ (n + 1)))
  sum_loop 0 0

theorem sum_of_series : sum_expression = (-507 : ℂ) + 1511 * Complex.i := by
  sorry

end sum_of_series_l527_527213


namespace sqrt_binom_mod_not_integer_l527_527605

theorem sqrt_binom_mod_not_integer (k : ℤ) (p : ℕ) (hp : nat.prime p) (h_cong : p = 8 * k + 1) :
  let r := (nat.choose (4 * k.nat_abs) k.nat_abs) % p in
  ¬ ∃ (n : ℤ), n ^ 2 = r :=
begin
  sorry
end

end sqrt_binom_mod_not_integer_l527_527605


namespace sum_of_three_numbers_is_neg_fifteen_l527_527550

theorem sum_of_three_numbers_is_neg_fifteen
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 5)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 10) :
  a + b + c = -15 := by
  sorry

end sum_of_three_numbers_is_neg_fifteen_l527_527550


namespace find_new_height_l527_527602

noncomputable def radius : ℝ := 8
noncomputable def original_height : ℝ := 7

noncomputable def original_volume : ℝ := π * radius^2 * original_height
noncomputable def desired_volume : ℝ := 3 * original_volume

theorem find_new_height : 
  ∃ h_new : ℝ, π * radius^2 * h_new = desired_volume ∧ h_new = 21 :=
by
  sorry

end find_new_height_l527_527602


namespace partnership_investment_l527_527862

theorem partnership_investment (B_invest C_invest C_profit total_profit : ℝ) (hB : B_invest = 4000)
    (hC : C_invest = 2000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 252000) :
    ∃ A_invest : ℝ, A_invest = 8000 :=
by
  -- Given the ratio of C's investment to C's profit and the total investment to total profit
  let A_invest := 8000
  
  -- Given the proportion for C's investment and profit side:
  have h1: C_invest / C_profit = total_invest / total_profit, from sorry

  -- Solving C's investment and profit ratio and total investment and profit ratio:
  show ∃ A_invest : ℝ, A_invest = 8000, from sorry

end partnership_investment_l527_527862


namespace min_value_of_expression_l527_527618

noncomputable def f (x : ℝ) : ℝ :=
  2 / x + 9 / (1 - 2 * x)

theorem min_value_of_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 2) : ∃ m, f x = m ∧ m = 25 :=
by
  sorry

end min_value_of_expression_l527_527618


namespace standard_parabola_equation_l527_527227

-- Definitions based on conditions
def vertex := (0, 0)
def axis (p : ℕ) : (ℕ → ℕ → Prop) := λ x y, y * y = 4 * p * x ∨ x * x = 4 * p * y
def distance_from_vertex_to_directrix := 4

-- Theorem definition
theorem standard_parabola_equation :
  ∃ p, p = 8 ∧ axis p vertex.fst vertex.snd := sorry

end standard_parabola_equation_l527_527227


namespace Tameka_sold_40_boxes_on_Friday_l527_527410

noncomputable def TamekaSalesOnFriday (F : ℕ) : Prop :=
  let SaturdaySales := 2 * F - 10
  let SundaySales := (2 * F - 10) / 2
  F + SaturdaySales + SundaySales = 145

theorem Tameka_sold_40_boxes_on_Friday : ∃ F : ℕ, TamekaSalesOnFriday F ∧ F = 40 := 
by 
  sorry

end Tameka_sold_40_boxes_on_Friday_l527_527410


namespace sum_youngest_oldest_eq_26_l527_527439

def cousin_ages := {a : ℝ // 0 ≤ a}

noncomputable def ages_satisfy_conditions (a₁ a₂ a₃ a₄ : cousin_ages) : Prop :=
  (a₁ + a₂ + a₃ + a₄) / 4 = 10 ∧  -- Mean age is 10
  (a₂ + a₃) / 2 = 7 ∧             -- Median age is 7
  a₂ = a₄ - 3                     -- Age relationship between a₂ and a₄

theorem sum_youngest_oldest_eq_26 (a₁ a₂ a₃ a₄ : cousin_ages) :
  ages_satisfy_conditions a₁ a₂ a₃ a₄ → a₁ + a₄ = 26 :=
by
  sorry

end sum_youngest_oldest_eq_26_l527_527439


namespace bacteria_after_time_l527_527565

def initial_bacteria : ℕ := 1
def division_time : ℕ := 20  -- time in minutes for one division
def total_time : ℕ := 180  -- total time in minutes

def divisions := total_time / division_time

theorem bacteria_after_time : (initial_bacteria * 2 ^ divisions) = 512 := by
  exact sorry

end bacteria_after_time_l527_527565


namespace Seth_boxes_initially_l527_527037

-- Define the initial conditions
def initial_boxes (x : ℕ) : Prop :=
  ∃ n : ℕ, 4 = (x - 1) / 2 ∧ x = 9

-- Prove that Seth initially bought 9 boxes of oranges
theorem Seth_boxes_initially : initial_boxes 9 :=
by {
  use 9,
  split,
  exact rfl,
  exact rfl,
}

end Seth_boxes_initially_l527_527037


namespace probability_red_jelly_bean_l527_527495

variable (r b g : Nat) (eaten_green eaten_blue : Nat)

theorem probability_red_jelly_bean
    (h_r : r = 15)
    (h_b : b = 20)
    (h_g : g = 16)
    (h_eaten_green : eaten_green = 1)
    (h_eaten_blue : eaten_blue = 1)
    (h_total : r + b + g = 51)
    (h_remaining_total : r + (b - eaten_blue) + (g - eaten_green) = 49) :
    (r : ℚ) / 49 = 15 / 49 :=
by
  sorry

end probability_red_jelly_bean_l527_527495


namespace min_value_of_sum_l527_527009

noncomputable def min_value_x_3y (x y : ℝ) : ℝ :=
  x + 3 * y

theorem min_value_of_sum (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (cond : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + 3 * y ≥ 4 + 4 * Real.sqrt 3 :=
  sorry

end min_value_of_sum_l527_527009


namespace min_value_of_exponential_sum_l527_527709

open Real

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = log 2 (log 2 3)) :
  2^a + 3^b = log 2 6 :=
sorry

end min_value_of_exponential_sum_l527_527709


namespace ratio_of_volumes_l527_527470
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l527_527470


namespace total_weight_l527_527729

-- Define the weights of the parcels and the weight pairs
variables (x y z : ℕ)
variables (w1 w2 w3 : ℕ) -- Weights of parcel pairs

-- Conditions given in the problem
def conditions := (w1 = 168) ∧ (w2 = 174) ∧ (w3 = 180) ∧ (x + y = w1) ∧ (y + z = w2) ∧ (x + z = w3)

-- Prove that the total weight of the parcels is 261 lbs
theorem total_weight (h : conditions x y z) : x + y + z = 261 := by
    sorry

end total_weight_l527_527729


namespace Triangle_is_stable_l527_527476

def Shape : Type := 
  | Triangle
  | Square
  | Hexagon
  | Polygon

def isStable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => True
  | _ => False

theorem Triangle_is_stable :
  ∀ (s : Shape), isStable s → s = Shape.Triangle :=
by
  intros s h
  cases s
  sorry -- actual proof steps would go here; we expect True for Shape.Triangle and False otherwise

end Triangle_is_stable_l527_527476


namespace find_ratio_l527_527552

open Nat

def sequence_def (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 →
    (a ((n + 2)) / a ((n + 1))) - (a ((n + 1)) / a n) = d

def geometric_difference_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3 ∧ sequence_def a 2

theorem find_ratio (a : ℕ → ℕ) (h : geometric_difference_sequence a) :
  a 12 / a 10 = 399 := sorry

end find_ratio_l527_527552


namespace idempotent_matrices_inequality_l527_527342

theorem idempotent_matrices_inequality {n : ℕ} (A : fin n → Matrix ℝ ℝ) :
  (∀ i, A i * A i = A i) →
  ∑ i, (A i).ker.dim ≥ (I - ∏ i, A i).rank :=
by
  sorry

end idempotent_matrices_inequality_l527_527342


namespace Drew_older_than_Maya_by_5_l527_527208

variable (Maya Drew Peter John Jacob : ℕ)
variable (h1 : John = 30)
variable (h2 : John = 2 * Maya)
variable (h3 : Jacob = 11)
variable (h4 : Jacob + 2 = (Peter + 2) / 2)
variable (h5 : Peter = Drew + 4)

theorem Drew_older_than_Maya_by_5 : Drew = Maya + 5 :=
by
  have Maya_age : Maya = 30 / 2 := by sorry
  have Jacob_age_in_2_years : Jacob + 2 = 13 := by sorry
  have Peter_age_in_2_years : Peter + 2 = 2 * 13 := by sorry
  have Peter_age : Peter = 26 - 2 := by sorry
  have Drew_age : Drew = Peter - 4 := by sorry
  have Drew_older_than_Maya : Drew = Maya + 5 := by sorry
  exact Drew_older_than_Maya

end Drew_older_than_Maya_by_5_l527_527208


namespace parallel_lines_perpendicular_lines_l527_527640

variables {m : ℝ}
def l₁ (x y : ℝ) : Prop := x + m * y + 6 = 0
def l₂ (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0
def slope_l₁ := - (1 / m)
def slope_l₂ := (2 - m) / 3

theorem parallel_lines (m_ne_zero : m ≠ 0) : (slope_l₁ = slope_l₂) ↔ (m = -1) :=
by sorry

theorem perpendicular_lines (m_ne_zero : m ≠ 0) : (slope_l₁ * slope_l₂ = -1) ↔ (m = 1 / 2) :=
by sorry

end parallel_lines_perpendicular_lines_l527_527640


namespace find_f_neg1_l527_527352

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def f (x : ℝ) : ℝ := if 1 ≤ x ∧ x < 3 then x - 2 else 0 -- Define as piecewise function to cover all ℝ

theorem find_f_neg1 (f_periodic : periodic f 2)
  (f_def : ∀ x, 1 ≤ x ∧ x < 3 → f x = x - 2) : 
  f (-1) = -1 :=
by
  sorry

end find_f_neg1_l527_527352


namespace inradius_of_triangle_l527_527457

theorem inradius_of_triangle (ABC : Triangle) (I : Point) (D : Point) (X : Point)
  (h_incenter : is_incenter I ABC)
  (h_perpendicular : is_perpendicular D A BC)
  (h_circumdiameter : is_diameter AX (circumcircle ABC))
  (h_ID : distance I D = 2)
  (h_IA : distance I A = 3)
  (h_IX : distance I X = 4) : inradius ABC = 11 / 12 := 
sorry

end inradius_of_triangle_l527_527457


namespace evaluate_81_squared_minus_49_squared_l527_527908

theorem evaluate_81_squared_minus_49_squared : 81^2 - 49^2 = 4160 := by
  calc 81^2 - 49^2 = (81 + 49) * (81 - 49) : by rw Nat.pow_two_sub_pow_two
               ... = 130 * 32             : by norm_num
               ... = 4160                 : by norm_num

end evaluate_81_squared_minus_49_squared_l527_527908


namespace ellipse_properties_l527_527945

theorem ellipse_properties
  (a b : ℝ) (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (t m : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : F = (real.sqrt 6, 0)) 
  (h3 : ∀ (P1 P2 : ℝ × ℝ), P1 = (real.sqrt 6, 1) ∧ P2 = F → dist P1 P2 = 2)
  (h4 : C = {p | (p.2 ^ 2 / a ^ 2) + (p.1 ^ 2 / b ^ 2) = 1})
  (h5 : ∀ (l l1 : ℝ × ℝ → Prop), (l = λ q, 2 * q.1 + q.2 + m = 0) ∧ (l1 = λ q, q.1 - 2 * q.2 + t = 0) → 
        ∀ (d : ℝ) (B D : ℝ × ℝ), B ∈ C ∧ D ∈ C ∧ d = dist (0, 0) l →
        ∃ (x1 x2 : ℝ), x1 + x2 = -12 / 13 * m ∧ (x1 * x2 = (3 * m ^ 2 - 9) / 13) ∧
        max_del_OBD = |BD| / 2 * d)
  (max_del_OBD : ℝ)
  (h6 : max_del_OBD = (3 * real.sqrt 3) / 2) :
  (C = {p | (p.1 ^ 2 / 9) + (p.2 ^ 2 / 3) = 1} ∧ max_del_OBD = (3 * real.sqrt 3) / 2) :=
by
  sorry

end ellipse_properties_l527_527945


namespace part_I_1_part_I_2_part_II_l527_527984

def seq_a : ℕ → ℝ
| 1 := 5
| 2 := 2
| n+2 := 2 * seq_a (n+1) + 3 * seq_a n

def seq_geometric (a b r : ℝ) (u : ℕ → ℝ) : Prop :=
∀ n, u (n+1) - a * u n = b * r^n

theorem part_I_1 : seq_geometric 3 -13 (-1) (λ n, seq_a (n+1) - 3 * seq_a n) :=
sorry

theorem part_I_2 : ∀ n, seq_a (n+1) = (13/4) * (-1)^n + (7/4) * (3^n) :=
sorry

def seq_b (n : ℕ) : ℝ :=
(2*n - 1) / 7 * (seq_a (n+1) + seq_a n)

def sum_seq (u : ℕ → ℝ) : ℕ → ℝ
| 0 := 0
| (n + 1) := sum_seq n + u n

theorem part_II : ∀ n, sum_seq seq_b n = (n-1) * 3^n + 1 :=
sorry

end part_I_1_part_I_2_part_II_l527_527984


namespace Mr_A_loses_2040_l527_527147

-- Define the initial value of the house
def house_value : ℝ := 12000

-- Define Mr. A's selling price to Mr. B (15% loss)
def sale_price_A_to_B : ℝ := house_value * (1 - 0.15)

-- Define Mr. B's selling price back to Mr. A (20% gain)
def sale_price_B_to_A : ℝ := sale_price_A_to_B * (1 + 0.20)

-- Define Mr. A's loss
def Mr_A's_loss : ℝ := sale_price_B_to_A - house_value

theorem Mr_A_loses_2040 : Mr_A's_loss = 2040 := by
  have h1: sale_price_A_to_B = 10200 := by
    unfold sale_price_A_to_B
    norm_num
  have h2: sale_price_B_to_A = 12240 := by
    unfold sale_price_B_to_A
    rw h1
    norm_num
  unfold Mr_A's_loss
  rw h2
  unfold house_value
  norm_num
  sorry

end Mr_A_loses_2040_l527_527147


namespace count_factors_of_5_in_factorial_340_l527_527772

theorem count_factors_of_5_in_factorial_340 :
  (∑ k in Finset.range (Nat.log 340 / Nat.log 5).natCeil, 340 / 5^k) = 83 :=
by
  sorry

end count_factors_of_5_in_factorial_340_l527_527772


namespace maximal_value_n_l527_527754

theorem maximal_value_n (m n : ℕ) (A : ℕ → set ℕ) 
  (hA_sub : ∀ i, i ≤ n → A i ⊆ finset.range (m + 1)) 
  (hA_card : ∀ i, i ≤ n → finset.card (A i) = 2) 
  (hA_dist : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → 
    A i ≠ A j → finset.image (λ (p : ℕ × ℕ), prod.fst p + prod.snd p)
    (A i.product (A j)) ≠ finset.image (λ (p : ℕ × ℕ), prod.fst p + prod.snd p)
    (A j.product (A i)))) : 
  n ≤ 2 * m - 3 := 
sorry

end maximal_value_n_l527_527754


namespace fraction_spent_on_DVDs_l527_527693

theorem fraction_spent_on_DVDs (initial_money spent_on_books additional_books_cost remaining_money_spent fraction remaining_money_after_DVDs : ℚ) : 
  initial_money = 320 ∧
  spent_on_books = initial_money / 4 ∧
  additional_books_cost = 10 ∧
  remaining_money_spent = 230 ∧
  remaining_money_after_DVDs = 130 ∧
  remaining_money_spent = initial_money - (spent_on_books + additional_books_cost) ∧
  remaining_money_after_DVDs = remaining_money_spent - (fraction * remaining_money_spent + 8) 
  → fraction = 46 / 115 :=
by
  intros
  sorry

end fraction_spent_on_DVDs_l527_527693


namespace number_of_ways_to_sum_three_l527_527124

/-- There are 9 cards numbered 1 through 9.
    We are to pick 2 cards at the same time.
    Show that the number of ways to pick 2 cards such that their sum is 3 is 1. -/
theorem number_of_ways_to_sum_three : 
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  in (∃ (p : Nat × Nat), p.1 ∈ cards ∧ p.2 ∈ cards ∧ p.1 + p.2 = 3 ∧ p.1 ≠ p.2) = 1 :=
sorry

end number_of_ways_to_sum_three_l527_527124


namespace junk_mail_per_red_or_white_house_l527_527152

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l527_527152


namespace data_transmission_time_l527_527569

-- Definitions for the conditions
def blocks : ℕ := 80
def chunks_per_block : ℕ := 400
def chunks_per_second : ℕ := 160
def seconds_per_minute : ℕ := 60

-- The proposition to be proved
theorem data_transmission_time :
  let total_chunks := blocks * chunks_per_block in
  let total_seconds := total_chunks / chunks_per_second in
  let total_minutes := total_seconds / seconds_per_minute in
  total_minutes = 3 :=
by
  -- This is where you would include the proof, but we skip it with "sorry"
  sorry

end data_transmission_time_l527_527569


namespace birch_count_is_87_l527_527448

def TreeSignProblem : Prop :=
  ∃ B L : ℕ,
    B + L = 130 ∧
    (∃ ! (b : ℕ), b < B ∧ ∀ (l : ℕ), l < L → false_sign(l)) ∧
    (∀ (b : ℕ), b < B → (false_sign(b) ↔ b = one_false_birch))

noncomputable def one_false_birch : ℕ := sorry
noncomputable def false_sign (n : ℕ) : Prop := sorry

theorem birch_count_is_87 : TreeSignProblem → ∃ B, B = 87 :=
by
  intro h
  sorry

end birch_count_is_87_l527_527448


namespace expected_messages_xiaoli_l527_527563

noncomputable def expected_greeting_messages (probs : List ℝ) (counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (λ p c => p * c) probs counts)

theorem expected_messages_xiaoli :
  expected_greeting_messages [1, 0.8, 0.5, 0] [8, 15, 14, 3] = 27 :=
by
  -- The proof will use the expected value formula
  sorry

end expected_messages_xiaoli_l527_527563


namespace sphere_volume_l527_527551

theorem sphere_volume : 
  ∀ (r_s : ℝ), (∀ (r_c : ℝ), r_c = 1 ∧ 4 = 4 * (r_s^2)) -> 
  ∀ (d : ℝ), d = 2 -> 
  π = π ->
  volume_sphere r_s = (4 / 3) * π * (sqrt 5 ^ 3) ->
  volume_sphere r_s =  (20 * sqrt 5 * π)/3
:= by
  sorry

def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

end sphere_volume_l527_527551


namespace find_a_perpendicular_l527_527639

theorem find_a_perpendicular (a : ℝ) : 
  (let l1 := ax - y + 2a = 0, l2 := (2a - 1)x + ay + a = 0 in 
  (a * (2a - 1) + a * (-1) = 0)) → (a = 0 ∨ a = 1) :=
by
  sorry

end find_a_perpendicular_l527_527639


namespace TV_cost_difference_l527_527474

def cost_per_square_inch_difference :=
  let first_TV_width := 24
  let first_TV_height := 16
  let first_TV_original_cost_euros := 840
  let first_TV_discount_percent := 0.10
  let first_TV_tax_percent := 0.05
  let exchange_rate_first := 1.20
  let first_TV_area := first_TV_width * first_TV_height

  let discounted_price_first_TV := first_TV_original_cost_euros * (1 - first_TV_discount_percent)
  let total_cost_euros_first_TV := discounted_price_first_TV * (1 + first_TV_tax_percent)
  let total_cost_dollars_first_TV := total_cost_euros_first_TV * exchange_rate_first
  let cost_per_square_inch_first_TV := total_cost_dollars_first_TV / first_TV_area

  let new_TV_width := 48
  let new_TV_height := 32
  let new_TV_original_cost_dollars := 1800
  let new_TV_first_discount_percent := 0.20
  let new_TV_second_discount_percent := 0.15
  let new_TV_tax_percent := 0.08
  let new_TV_area := new_TV_width * new_TV_height

  let price_after_first_discount := new_TV_original_cost_dollars * (1 - new_TV_first_discount_percent)
  let price_after_second_discount := price_after_first_discount * (1 - new_TV_second_discount_percent)
  let total_cost_dollars_new_TV := price_after_second_discount * (1 + new_TV_tax_percent)
  let cost_per_square_inch_new_TV := total_cost_dollars_new_TV / new_TV_area

  let cost_difference_per_square_inch := cost_per_square_inch_first_TV - cost_per_square_inch_new_TV
  cost_difference_per_square_inch

theorem TV_cost_difference :
  cost_per_square_inch_difference = 1.62 := by
  sorry

end TV_cost_difference_l527_527474


namespace orange_ratio_l527_527864

theorem orange_ratio (total_oranges alice_oranges : ℕ) (h_total : total_oranges = 180) (h_alice : alice_oranges = 120) :
  alice_oranges / (total_oranges - alice_oranges) = 2 :=
by
  sorry

end orange_ratio_l527_527864


namespace logarithmic_sum_of_geometric_seq_l527_527603

noncomputable section

variable {a_n : ℕ → ℝ} (q : ℝ) (n : ℕ)

-- Assuming the conditions that a_n is a geometric sequence with common ratio q and a_6 = 2
def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ a₁, ∀ n : ℕ, a_n = a₁ * q^n

axiom a_6_eq_2 : a_n 6 = 2
axiom q_pos : q > 0

theorem logarithmic_sum_of_geometric_seq (h : is_geometric_sequence a_n q) :
  ( ∑ i in Finset.range 11, Real.logb 2 (a_n (i + 1)) ) = 11 :=
by
  sorry

end logarithmic_sum_of_geometric_seq_l527_527603


namespace travel_possible_with_two_roads_closed_l527_527075

-- Define a convex polyhedron graph
structure ConvexPolyhedronGraph (V E : Type u) :=
  (vertices : V)
  (edges : E)
  (connected : (x y : V) → x ≠ y → ∃ p : List V, p.head = x ∧ p.last = y ∧ (∀ i ∈ p, ∃ e : E, e ∈ edges ∧ ∃ u v : V, e = (u, v) ∧ (u = i ∨ v = i)))

-- Conditions from the problem statement
variables {V E : Type}
variable [DecidableEq V]
variable (P : ConvexPolyhedronGraph V E)
variable (A B : V)
variable (closed_roads : Finset (V × V))
variable (h_ne : A ≠ B)
variable (h_closed : closed_roads.card ≤ 2)

-- The theorem we need to prove
theorem travel_possible_with_two_roads_closed 
  (h_connected : ∀ (x y : V), x ≠ y → ∃ p : List V, p.head = x ∧ p.last = y ∧ (∀ i ∈ p, ∃ e : (V × V), e ∈ P.edges ∧ ∃ u v : V, e = (u, v) ∧ (u = i ∨ v = i))) 
  : ∃ p : List V, p.head = A ∧ p.last = B ∧ (∀ i ∈ p, (∃ e : (V × V), e ∈ P.edges \ closed_roads ∧ ∃ u v : V, e = (u, v) ∧ (u = i ∨ v = i))) :=
sorry

end travel_possible_with_two_roads_closed_l527_527075


namespace difference_is_divisible_by_p_l527_527757

-- Lean 4 statement equivalent to the math proof problem
theorem difference_is_divisible_by_p
  (a : ℕ → ℕ) (p : ℕ) (d : ℕ)
  (h_prime : Nat.Prime p)
  (h_prog : ∀ i j: ℕ, 1 ≤ i ∧ i ≤ p ∧ 1 ≤ j ∧ j ≤ p ∧ i < j → a j = a (i + 1) + (j - 1) * d)
  (h_a_gt_p : a 1 > p)
  (h_arith_prog_primes : ∀ i: ℕ, 1 ≤ i ∧ i ≤ p → Nat.Prime (a i)) :
  d % p = 0 := sorry

end difference_is_divisible_by_p_l527_527757


namespace positive_integers_polynomial_factors_l527_527920

theorem positive_integers_polynomial_factors :
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∃ a b : ℤ, (x^2 - 2 * x - n = (x - a) * (x - b) ∧ a + b = 2 ∧ a * b = -n)}.card = 22 :=
by
  sorry

end positive_integers_polynomial_factors_l527_527920


namespace graph_symmetry_l527_527925

theorem graph_symmetry (f : ℝ → ℝ) : 
  ∀ x : ℝ, f (x - 1) = f (-(x - 1)) ↔ x = 1 :=
by 
  sorry

end graph_symmetry_l527_527925


namespace mary_fruits_l527_527013

noncomputable def totalFruitsLeft 
    (initial_apples: ℕ) (initial_oranges: ℕ) (initial_blueberries: ℕ) (initial_grapes: ℕ) (initial_kiwis: ℕ)
    (salad_apples: ℕ) (salad_oranges: ℕ) (salad_blueberries: ℕ)
    (snack_apples: ℕ) (snack_oranges: ℕ) (snack_kiwis: ℕ)
    (given_apples: ℕ) (given_oranges: ℕ) (given_blueberries: ℕ) (given_grapes: ℕ) (given_kiwis: ℕ) : ℕ :=
  let remaining_apples := initial_apples - salad_apples - snack_apples - given_apples
  let remaining_oranges := initial_oranges - salad_oranges - snack_oranges - given_oranges
  let remaining_blueberries := initial_blueberries - salad_blueberries - given_blueberries
  let remaining_grapes := initial_grapes - given_grapes
  let remaining_kiwis := initial_kiwis - snack_kiwis - given_kiwis
  remaining_apples + remaining_oranges + remaining_blueberries + remaining_grapes + remaining_kiwis

theorem mary_fruits :
    totalFruitsLeft 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end mary_fruits_l527_527013


namespace last_open_locker_l527_527784

theorem last_open_locker (N : ℕ) (hN : N > 2021)
  (initially_open : ∀ n, 1 ≤ n ∧ n ≤ N → locker n = open)
  (move_clockwise : ∀ n, 1 ≤ n ∧ n ≤ N → Ansoon_moves n clockwise)
  (closing_rule : ∀ n m, (1 ≤ n ∧ n ≤ N) → (locker n = open → if m > n then locker (next_n_open m n) = closed else locker m = open))
  (closing_process: ∀ n, (1 ≤ n ∧ n ≤ N) → (if remaining_open_lockers n > 1 then continue_process else locker n = open ∧ ∀ m ≠ n, locker m = closed))
  : ∃ N, N > 2021 ∧ (∃ unique l, (1 ≤ l ∧ l ≤ N) ∧ locker l = open ∧ ∀ m ≠ l, locker m = closed) :=
begin
  choose hN2046 using proof_of_N_2046,
  existsi 2046,
  split,
  { exact hN2046 },
  { sorry }
end

end last_open_locker_l527_527784


namespace painting_ways_fib_blocks_l527_527057

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| n + 2   := fibonacci (n + 1) + fibonacci n

def count_paintings (n : ℕ) : ℕ :=
if n = 16 then 32 else 0

theorem painting_ways_fib_blocks :
  count_paintings 16 = 32 :=
by
  sorry

end painting_ways_fib_blocks_l527_527057


namespace maximum_of_expression_l527_527725

theorem maximum_of_expression (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum : a + b + c = 3) (h_prod : a * b * c = 1) :
  (ab : ℝ) (ac : ℝ) (bc : ℝ) (ab_over_sum := ab / (a + b)) 
  (ac_over_sum := ac / (a + c)) 
  (bc_over_sum := bc / (b + c)) 
  ab_over_ac_over_bc_over_sum := ab_over_sum + ac_over_sum + bc_over_sum 
    ∧ max_ab_over_sum := 3 / 2
  :
     ab_over_ac_over_bc_over_sum = max_ab_over_sum
  :=
sorry

end maximum_of_expression_l527_527725


namespace smallest_positive_period_l527_527922

-- Define the function
def y (x : ℝ) : ℝ := 1 - sin (x + π / 3) ^ 2

-- Prove that the smallest positive period T of the function is π
theorem smallest_positive_period : ∃ T > 0, ∀ x, y (x + T) = y x ∧ (∀ T' > 0, (∀ x, y (x + T') = y x) → T' ≥ T) :=
sorry

end smallest_positive_period_l527_527922


namespace phyllis_marbles_l527_527396

theorem phyllis_marbles (num_groups : ℕ) (num_marbles_per_group : ℕ) (h1 : num_groups = 32) (h2 : num_marbles_per_group = 2) : 
  num_groups * num_marbles_per_group = 64 :=
by
  sorry

end phyllis_marbles_l527_527396


namespace volume_ratio_l527_527469

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l527_527469


namespace investment_time_l527_527063

theorem investment_time (P R diff : ℝ) (T : ℕ) 
  (hP : P = 1500)
  (hR : R = 0.10)
  (hdiff : diff = 15)
  (h1 : P * ((1 + R) ^ T - 1) - (P * R * T) = diff) 
  : T = 2 := 
by
  -- proof steps here
  sorry

end investment_time_l527_527063


namespace min_value_expression_l527_527997

theorem min_value_expression (x : ℝ) (h : x ≠ -7) : 
  ∃ y, y = 1 ∧ ∀ z, z = (2 * x ^ 2 + 98) / ((x + 7) ^ 2) → y ≤ z := 
sorry

end min_value_expression_l527_527997


namespace frog_eyes_count_l527_527534

def total_frog_eyes (a b c : ℕ) (eyesA eyesB eyesC : ℕ) : ℕ :=
  a * eyesA + b * eyesB + c * eyesC

theorem frog_eyes_count :
  let a := 2
  let b := 1
  let c := 3
  let eyesA := 2
  let eyesB := 3
  let eyesC := 4
  total_frog_eyes a b c eyesA eyesB eyesC = 19 := by
  sorry

end frog_eyes_count_l527_527534


namespace train_speed_with_stoppages_is_36_kmph_l527_527212

variable (speed_excluding_stoppages : ℝ) (stop_time_per_hour : ℝ)

-- Given conditions
def speed_of_train_including_stoppages := ( 1 - stop_time_per_hour / 60) * speed_excluding_stoppages

-- Proof statement
theorem train_speed_with_stoppages_is_36_kmph :
    speed_excluding_stoppages = 54 → stop_time_per_hour = 20 → speed_of_train_including_stoppages speed_excluding_stoppages stop_time_per_hour = 36 := sorry

end train_speed_with_stoppages_is_36_kmph_l527_527212


namespace angle_between_a_and_v_is_90_degrees_l527_527349

noncomputable def a : ℝ × ℝ × ℝ := (2, 3, -4)
noncomputable def b : ℝ × ℝ × ℝ := (Real.sqrt 3, -2, 5)
noncomputable def c : ℝ × ℝ × ℝ := (-7, 6, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mult (k : ℝ) (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * u.1, k * u.2, k * u.3)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

noncomputable def a_dot_b : ℝ := dot_product a b
noncomputable def a_dot_c : ℝ := dot_product a c

noncomputable def v : ℝ × ℝ × ℝ :=
  vector_sub (scalar_mult (a_dot_c) b) (scalar_mult (a_dot_b) c)

theorem angle_between_a_and_v_is_90_degrees :
  dot_product a v = 0 :=
by
  sorry

end angle_between_a_and_v_is_90_degrees_l527_527349
