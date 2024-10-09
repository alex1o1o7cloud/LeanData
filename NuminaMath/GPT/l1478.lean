import Mathlib

namespace proof_f_1_add_g_2_l1478_147843

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 1

theorem proof_f_1_add_g_2 : f (1 + g 2) = 8 := by
  sorry

end proof_f_1_add_g_2_l1478_147843


namespace problem_i31_problem_i32_problem_i33_problem_i34_l1478_147811

-- Problem I3.1
theorem problem_i31 (a : ℝ) :
  a = 1.8 * 5.0865 + 1 - 0.0865 * 1.8 → a = 10 :=
by sorry

-- Problem I3.2
theorem problem_i32 (a b : ℕ) (oh ok : ℕ) (OABC : Prop) :
  oh = ok ∧ oh = a ∧ ok = a ∧ OABC ∧ (b = AC) → b = 10 :=
by sorry

-- Problem I3.3
theorem problem_i33 (b c : ℕ) :
  b = 10 → c = (10 - 2) :=
by sorry

-- Problem I3.4
theorem problem_i34 (c d : ℕ) :
  c = 30 → d = 3 * c → d = 90 :=
by sorry

end problem_i31_problem_i32_problem_i33_problem_i34_l1478_147811


namespace simplify_rationalize_expr_l1478_147877

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l1478_147877


namespace arithmetic_expression_l1478_147830

theorem arithmetic_expression :
  10 + 4 * (5 + 3)^3 = 2058 :=
by
  sorry

end arithmetic_expression_l1478_147830


namespace shelves_in_room_l1478_147842

theorem shelves_in_room
  (n_action_figures_per_shelf : ℕ)
  (total_action_figures : ℕ)
  (h1 : n_action_figures_per_shelf = 10)
  (h2 : total_action_figures = 80) :
  total_action_figures / n_action_figures_per_shelf = 8 := by
  sorry

end shelves_in_room_l1478_147842


namespace no_snow_probability_l1478_147867

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2 / 3) 
  (h2 : p2 = 3 / 4) 
  (h3 : p3 = 5 / 6) 
  (h4 : p4 = 1 / 2) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 144 :=
by
  sorry

end no_snow_probability_l1478_147867


namespace first_of_five_consecutive_sums_60_l1478_147890

theorem first_of_five_consecutive_sums_60 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) : n = 10 :=
by {
  sorry
}

end first_of_five_consecutive_sums_60_l1478_147890


namespace calculate_expression_l1478_147868

-- Defining the main theorem to prove
theorem calculate_expression (a b : ℝ) : 
  3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by 
  sorry

end calculate_expression_l1478_147868


namespace girl_attendance_l1478_147822

theorem girl_attendance (g b : ℕ) (h1 : g + b = 1500) (h2 : (3 / 4 : ℚ) * g + (1 / 3 : ℚ) * b = 900) :
  (3 / 4 : ℚ) * g = 720 :=
by
  sorry

end girl_attendance_l1478_147822


namespace smallest_c_no_real_root_l1478_147805

theorem smallest_c_no_real_root (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 5) ↔ c = -4 :=
by
  sorry

end smallest_c_no_real_root_l1478_147805


namespace initial_marbles_count_l1478_147887

theorem initial_marbles_count (g y : ℕ) 
  (h1 : (g + 3) * 4 = g + y + 3) 
  (h2 : 3 * g = g + y + 4) : 
  g + y = 8 := 
by 
  -- The proof will go here
  sorry

end initial_marbles_count_l1478_147887


namespace sum_to_fraction_l1478_147835

theorem sum_to_fraction :
  (2 / 10) + (3 / 100) + (4 / 1000) + (6 / 10000) + (7 / 100000) = 23467 / 100000 :=
by
  sorry

end sum_to_fraction_l1478_147835


namespace train_speed_is_36_kph_l1478_147894

-- Define the given conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 3

-- Convert distance from meters to kilometers
def distance_kilometers : ℕ -> ℕ := fun d => d / 1000
-- Convert time from minutes to hours
def time_hours : ℕ -> ℚ := fun t => (t : ℚ) / 60

-- Calculate speed in kilometers per hour
def speed_kph (d : ℕ) (t : ℕ) : ℚ :=
  let d_km := d / 1000
  let t_hr := (t : ℚ) / 60
  d_km / t_hr

-- The theorem to prove the speed
theorem train_speed_is_36_kph :
  speed_kph distance_meters time_minutes = 36 := by
  sorry

end train_speed_is_36_kph_l1478_147894


namespace triangle_area_ratio_l1478_147824

-- Define parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : (ℝ × ℝ) := (2, 0)

-- Define the line passing through the focus and intersecting the parabola
def line_through_focus (f : ℝ × ℝ) (a b : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l (f.1) = f.2 ∧ parabola a.1 a.2 ∧ parabola b.1 b.2 ∧   -- line passes through the focus and intersects parabola at a and b
  l a.1 = a.2 ∧ l b.1 = b.2 ∧ 
  |a.1 - f.1| + |a.2 - f.2| = 3 ∧ -- condition |AF| = 3
  (f = (2, 0))

-- The proof problem
theorem triangle_area_ratio (a b : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_line : line_through_focus focus a b l) :
  ∃ r, r = (1 / 2) := 
sorry

end triangle_area_ratio_l1478_147824


namespace point_coordinates_l1478_147864

theorem point_coordinates (m : ℝ) 
  (h1 : dist (0 : ℝ) (Real.sqrt m) = 4) : 
  (-m, Real.sqrt m) = (-16, 4) := 
by
  -- The proof will use the conditions and solve for m to find the coordinates
  sorry

end point_coordinates_l1478_147864


namespace boat_distance_downstream_l1478_147858

-- Let v_s be the speed of the stream in km/h
-- Condition 1: In one hour, a boat goes 5 km against the stream.
-- Condition 2: The speed of the boat in still water is 8 km/h.

theorem boat_distance_downstream (v_s : ℝ) :
  (8 - v_s = 5) →
  (distance : ℝ) →
  8 + v_s = distance →
  distance = 11 := by
  sorry

end boat_distance_downstream_l1478_147858


namespace find_x0_range_l1478_147815

variable {x y x0 : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def angle_condition (x0 : ℝ) : Prop :=
  let OM := Real.sqrt (x0^2 + 3)
  OM ≤ 2

theorem find_x0_range (h1 : circle_eq x y) (h2 : angle_condition x0) :
  -1 ≤ x0 ∧ x0 ≤ 1 := 
sorry

end find_x0_range_l1478_147815


namespace tom_candy_pieces_l1478_147823

/-!
# Problem Statement
Tom bought 14 boxes of chocolate candy, 10 boxes of fruit candy, and 8 boxes of caramel candy. 
He gave 8 chocolate boxes and 5 fruit boxes to his little brother. 
If each chocolate box has 3 pieces inside, each fruit box has 4 pieces, and each caramel box has 5 pieces, 
prove that Tom still has 78 pieces of candy.
-/

theorem tom_candy_pieces 
  (chocolate_boxes : ℕ := 14)
  (fruit_boxes : ℕ := 10)
  (caramel_boxes : ℕ := 8)
  (gave_away_chocolate_boxes : ℕ := 8)
  (gave_away_fruit_boxes : ℕ := 5)
  (chocolate_pieces_per_box : ℕ := 3)
  (fruit_pieces_per_box : ℕ := 4)
  (caramel_pieces_per_box : ℕ := 5)
  : chocolate_boxes * chocolate_pieces_per_box + 
    fruit_boxes * fruit_pieces_per_box + 
    caramel_boxes * caramel_pieces_per_box - 
    (gave_away_chocolate_boxes * chocolate_pieces_per_box + 
     gave_away_fruit_boxes * fruit_pieces_per_box) = 78 :=
by
  sorry

end tom_candy_pieces_l1478_147823


namespace rectangular_region_area_l1478_147849

theorem rectangular_region_area :
  ∀ (s : ℝ), 18 * s * s = (15 * Real.sqrt 2) * (7.5 * Real.sqrt 2) :=
by
  intro s
  have h := 5 ^ 2 = 2 * s ^ 2
  have s := Real.sqrt (25 / 2)
  exact sorry

end rectangular_region_area_l1478_147849


namespace inequality_proof_l1478_147889

variable {a b : ℕ → ℝ}

-- Conditions: {a_n} is a geometric sequence with positive terms, {b_n} is an arithmetic sequence, a_6 = b_8
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

axiom a_pos_terms : ∀ n : ℕ, a n > 0
axiom a_geom_seq : is_geometric a
axiom b_arith_seq : is_arithmetic b
axiom a6_eq_b8 : a 6 = b 8

-- Prove: a_3 + a_9 ≥ b_9 + b_7
theorem inequality_proof : a 3 + a 9 ≥ b 9 + b 7 :=
by sorry

end inequality_proof_l1478_147889


namespace solve_for_nabla_l1478_147826

theorem solve_for_nabla (nabla mu : ℤ) (h1 : 5 * (-3) = nabla + mu - 3) (h2 : mu = 4) : 
  nabla = -16 := 
by
  sorry

end solve_for_nabla_l1478_147826


namespace finite_points_outside_unit_circle_l1478_147869

noncomputable def centroid (x y z : ℝ × ℝ) : ℝ × ℝ := 
  ((x.1 + y.1 + z.1) / 3, (x.2 + y.2 + z.2) / 3)

theorem finite_points_outside_unit_circle
  (A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)
  (D : ℕ → ℝ × ℝ)
  (hA : ∀ n, A (n + 1) = centroid (B n) (C n) (D n))
  (hB : ∀ n, B (n + 1) = centroid (A n) (C n) (D n))
  (hC : ∀ n, C (n + 1) = centroid (A n) (B n) (D n))
  (hD : ∀ n, D (n + 1) = centroid (A n) (B n) (C n))
  (h₀ : A 1 = A₁ ∧ B 1 = B₁ ∧ C 1 = C₁ ∧ D 1 = D₁)
  : ∃ N : ℕ, ∀ n > N, (A n).1 * (A n).1 + (A n).2 * (A n).2 ≤ 1 :=
sorry

end finite_points_outside_unit_circle_l1478_147869


namespace second_train_speed_l1478_147880

theorem second_train_speed :
  ∃ v : ℝ, 
  (∀ t : ℝ, 20 * t = v * t + 50) ∧
  (∃ t : ℝ, 20 * t + v * t = 450) →
  v = 16 :=
by
  sorry

end second_train_speed_l1478_147880


namespace max_stamps_l1478_147834

-- Definitions based on conditions
def price_of_stamp := 28 -- in cents
def total_money := 3600 -- in cents

-- The theorem statement
theorem max_stamps (price_of_stamp total_money : ℕ) : (total_money / price_of_stamp) = 128 := by
  sorry

end max_stamps_l1478_147834


namespace number_of_girls_l1478_147892

theorem number_of_girls (B G : ℕ) (h1 : B * 5 = G * 8) (h2 : B + G = 1040) : G = 400 :=
by
  sorry

end number_of_girls_l1478_147892


namespace proof_problem_l1478_147802

noncomputable def problem_statement : Prop :=
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ 
    x1 + x2 + x3 + x4 = 8 ∧ 
    x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ 
    (x1 + x2) = 2 * 2 ∧ 
    (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 - 4 * 2 * (x1 + x2 + x3 + x4) + 4 * 4) = 4 ∧ 
    (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3)

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l1478_147802


namespace table_price_l1478_147813

theorem table_price (C T : ℝ) (h1 : 2 * C + T = 0.6 * (C + 2 * T)) (h2 : C + T = 96) : T = 84 := by
  sorry

end table_price_l1478_147813


namespace exists_plane_perpendicular_l1478_147872

-- Definitions of line, plane and perpendicularity intersection etc.
variables (Point : Type) (Line Plane : Type)
variables (l : Line) (α : Plane) (intersects : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)
variables (β : Plane) (subset : Line → Plane → Prop)

-- Conditions
axiom line_intersects_plane (h1 : intersects l α) : Prop
axiom line_not_perpendicular_plane (h2 : ¬perpendicular l α) : Prop

-- The main statement to prove
theorem exists_plane_perpendicular (h1 : intersects l α) (h2 : ¬perpendicular l α) :
  ∃ (β : Plane), (subset l β) ∧ (perpendicular_planes β α) :=
sorry

end exists_plane_perpendicular_l1478_147872


namespace upper_bound_expression_4n_plus_7_l1478_147888

theorem upper_bound_expression_4n_plus_7 (U : ℤ) :
  (∃ (n : ℕ),  4 * n + 7 > 1) ∧
  (∀ (n : ℕ), 4 * n + 7 < U → ∃ (k : ℕ), k ≤ 19 ∧ k = n) ∧
  (∃ (n_min n_max : ℕ), n_max = n_min + 19 ∧ 4 * n_max + 7 < U) →
  U = 84 := sorry

end upper_bound_expression_4n_plus_7_l1478_147888


namespace sum_of_solutions_eq_zero_l1478_147866

theorem sum_of_solutions_eq_zero :
  let p := 6
  let q := 150
  (∃ x1 x2 : ℝ, p * x1 = q / x1 ∧ p * x2 = q / x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 0) :=
sorry

end sum_of_solutions_eq_zero_l1478_147866


namespace wine_with_cork_cost_is_2_10_l1478_147898

noncomputable def cork_cost : ℝ := 0.05
noncomputable def wine_without_cork_cost : ℝ := cork_cost + 2.00
noncomputable def wine_with_cork_cost : ℝ := wine_without_cork_cost + cork_cost

theorem wine_with_cork_cost_is_2_10 : wine_with_cork_cost = 2.10 :=
by
  -- skipped proof
  sorry

end wine_with_cork_cost_is_2_10_l1478_147898


namespace original_price_of_apples_l1478_147845

-- Define the conditions and problem
theorem original_price_of_apples 
  (discounted_price : ℝ := 0.60 * original_price)
  (total_cost : ℝ := 30)
  (weight : ℝ := 10) :
  original_price = 5 :=
by
  -- This is the point where the proof steps would go.
  sorry

end original_price_of_apples_l1478_147845


namespace percentage_refund_l1478_147839

theorem percentage_refund
  (initial_amount : ℕ)
  (sweater_cost : ℕ)
  (tshirt_cost : ℕ)
  (shoes_cost : ℕ)
  (amount_left_after_refund : ℕ)
  (refund_percentage : ℕ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  amount_left_after_refund = 51 →
  refund_percentage = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_refund_l1478_147839


namespace integer_part_M_is_4_l1478_147836

-- Define the variables and conditions based on the problem statement
variable (a b c : ℝ)

-- This non-computable definition includes the main mathematical expression we need to evaluate
noncomputable def M (a b c : ℝ) := Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1)

-- The theorem we need to prove
theorem integer_part_M_is_4 (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 
  ⌊M a b c⌋ = 4 := 
by 
  sorry

end integer_part_M_is_4_l1478_147836


namespace probability_of_heads_or_five_tails_is_one_eighth_l1478_147862

namespace coin_flip

def num_heads_or_at_least_five_tails : ℕ :=
1 + 6 + 1

def total_outcomes : ℕ :=
2^6

def probability_heads_or_five_tails : ℚ :=
num_heads_or_at_least_five_tails / total_outcomes

theorem probability_of_heads_or_five_tails_is_one_eighth :
  probability_heads_or_five_tails = 1 / 8 := by
  sorry

end coin_flip

end probability_of_heads_or_five_tails_is_one_eighth_l1478_147862


namespace extra_sweets_l1478_147874

theorem extra_sweets (S : ℕ) (h1 : ∀ n: ℕ, S = 120 * 38) : 
    (38 - (S / 190) = 14) :=
by
  -- Here we will provide the proof 
  sorry

end extra_sweets_l1478_147874


namespace decreasing_exponential_iff_l1478_147860

theorem decreasing_exponential_iff {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 1)^y < (a - 1)^x) ↔ (1 < a ∧ a < 2) :=
by 
  sorry

end decreasing_exponential_iff_l1478_147860


namespace find_x_values_l1478_147891

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 :=
by
  sorry

end find_x_values_l1478_147891


namespace ambiguous_dates_count_l1478_147803

theorem ambiguous_dates_count : 
  ∃ n : ℕ, n = 132 ∧ ∀ d m : ℕ, 1 ≤ d ∧ d ≤ 31 ∧ 1 ≤ m ∧ m ≤ 12 →
  ((d ≥ 1 ∧ d ≤ 12 ∧ m ≥ 1 ∧ m ≤ 12) → n = 132)
  :=
by 
  let ambiguous_days := 12 * 12
  let non_ambiguous_days := 12
  let total_ambiguous := ambiguous_days - non_ambiguous_days
  use total_ambiguous
  sorry

end ambiguous_dates_count_l1478_147803


namespace word_sum_problems_l1478_147841

theorem word_sum_problems (J M O I : Fin 10) (h_distinct : J ≠ M ∧ J ≠ O ∧ J ≠ I ∧ M ≠ O ∧ M ≠ I ∧ O ≠ I) 
  (h_nonzero_J : J ≠ 0) (h_nonzero_I : I ≠ 0) :
  let JMO := 100 * J + 10 * M + O
  let IMO := 100 * I + 10 * M + O
  (JMO + JMO + JMO = IMO) → 
  (JMO = 150 ∧ IMO = 450) ∨ (JMO = 250 ∧ IMO = 750) :=
sorry

end word_sum_problems_l1478_147841


namespace axis_of_symmetry_l1478_147833

-- Definitions for conditions
variable (ω : ℝ) (φ : ℝ) (A B : ℝ)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Hypotheses
axiom ω_pos : ω > 0
axiom φ_bound : 0 ≤ φ ∧ φ < Real.pi
axiom even_func : ∀ x, f x = f (-x)
axiom dist_AB : abs (B - A) = 4 * Real.sqrt 2

-- Proof statement
theorem axis_of_symmetry : ∃ x : ℝ, x = 4 := 
sorry

end axis_of_symmetry_l1478_147833


namespace solution_x_y_zero_l1478_147870

theorem solution_x_y_zero (x y : ℤ) (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 :=
by
sorry

end solution_x_y_zero_l1478_147870


namespace kim_money_l1478_147840

theorem kim_money (S P K : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : S + P = 1.80) : K = 1.12 :=
by sorry

end kim_money_l1478_147840


namespace correct_option_C_l1478_147831

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end correct_option_C_l1478_147831


namespace sound_frequency_and_speed_glass_proof_l1478_147800

def length_rod : ℝ := 1.10 -- Length of the glass rod, l in meters
def nodal_distance_air : ℝ := 0.12 -- Distance between nodal points in air, l' in meters
def speed_sound_air : ℝ := 340 -- Speed of sound in air, V in meters per second

-- Frequency of the sound produced
def frequency_sound_produced : ℝ := 1416.67

-- Speed of longitudinal waves in the glass
def speed_longitudinal_glass : ℝ := 3116.67

theorem sound_frequency_and_speed_glass_proof :
  (2 * nodal_distance_air = 0.24) ∧
  (frequency_sound_produced * (2 * length_rod) = speed_longitudinal_glass) :=
by
  -- Here we will include real equivalent math proof in the future
  sorry

end sound_frequency_and_speed_glass_proof_l1478_147800


namespace road_trip_mileage_base10_l1478_147878

-- Defining the base 8 number 3452
def base8_to_base10 (n : Nat) : Nat :=
  3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 2 * 8^0

-- Stating the problem as a theorem
theorem road_trip_mileage_base10 : base8_to_base10 3452 = 1834 := by
  sorry

end road_trip_mileage_base10_l1478_147878


namespace paving_time_together_l1478_147851

/-- Define the rate at which Mary alone paves the driveway -/
noncomputable def Mary_rate : ℝ := 1 / 4

/-- Define the rate at which Hillary alone paves the driveway -/
noncomputable def Hillary_rate : ℝ := 1 / 3

/-- Define the increased rate of Mary when working together -/
noncomputable def Mary_rate_increased := Mary_rate + (0.3333 * Mary_rate)

/-- Define the decreased rate of Hillary when working together -/
noncomputable def Hillary_rate_decreased := Hillary_rate - (0.5 * Hillary_rate)

/-- Combine their rates when working together -/
noncomputable def combined_rate := Mary_rate_increased + Hillary_rate_decreased

/-- Prove that the time taken to pave the driveway together is approximately 2 hours -/
theorem paving_time_together : abs ((1 / combined_rate) - 2) < 0.0001 :=
by
  sorry

end paving_time_together_l1478_147851


namespace exponent_division_l1478_147855

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division_l1478_147855


namespace best_model_l1478_147896

theorem best_model (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.55) (h2 : R2 = 0.65) (h3 : R3 = 0.79) (h4 : R4 = 0.95) :
  R4 > R3 ∧ R4 > R2 ∧ R4 > R1 :=
by {
  sorry
}

end best_model_l1478_147896


namespace emir_needs_more_money_l1478_147852

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end emir_needs_more_money_l1478_147852


namespace no_integer_roots_l1478_147853

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l1478_147853


namespace B_work_rate_l1478_147829

theorem B_work_rate (A_rate C_rate combined_rate : ℝ) (B_days : ℝ) (hA : A_rate = 1 / 4) (hC : C_rate = 1 / 8) (hCombined : A_rate + 1 / B_days + C_rate = 1 / 2) : B_days = 8 :=
by
  sorry

end B_work_rate_l1478_147829


namespace compute_fg_neg_2_l1478_147809

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 4 * x + 4

theorem compute_fg_neg_2 : f (g (-2)) = -5 :=
by
-- sorry is used to skip the proof
sorry

end compute_fg_neg_2_l1478_147809


namespace initial_tomatoes_l1478_147844

/-- 
Given the conditions:
  - The farmer picked 134 tomatoes yesterday.
  - The farmer picked 30 tomatoes today.
  - The farmer will have 7 tomatoes left after today.
Prove that the initial number of tomatoes in the farmer's garden was 171.
--/

theorem initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (left_tomatoes : ℕ)
  (h1 : picked_yesterday = 134)
  (h2 : picked_today = 30)
  (h3 : left_tomatoes = 7) :
  (picked_yesterday + picked_today + left_tomatoes) = 171 :=
by 
  sorry

end initial_tomatoes_l1478_147844


namespace gain_percent_is_50_l1478_147856

theorem gain_percent_is_50
  (C : ℕ) (S : ℕ) (hC : C = 10) (hS : S = 15) : ((S - C) / C : ℚ) * 100 = 50 := by
  sorry

end gain_percent_is_50_l1478_147856


namespace triangle_side_length_l1478_147816

open Real

/-- Given a triangle ABC with the incircle touching side AB at point D,
where AD = 5 and DB = 3, and given that the angle A is 60 degrees,
prove that the length of side BC is 13. -/
theorem triangle_side_length
  (A B C D : Point)
  (AD DB : ℝ)
  (hAD : AD = 5)
  (hDB : DB = 3)
  (angleA : Real)
  (hangleA : angleA = π / 3) : 
  ∃ BC : ℝ, BC = 13 :=
sorry

end triangle_side_length_l1478_147816


namespace arithmetic_sequence_fifth_term_l1478_147879

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  sorry

end arithmetic_sequence_fifth_term_l1478_147879


namespace water_heater_ratio_l1478_147821

variable (Wallace_capacity : ℕ) (Catherine_capacity : ℕ)
variable (Wallace_fullness : ℚ := 3/4) (Catherine_fullness : ℚ := 3/4)
variable (total_water : ℕ := 45)

theorem water_heater_ratio :
  Wallace_capacity = 40 →
  (Wallace_fullness * Wallace_capacity : ℚ) + (Catherine_fullness * Catherine_capacity : ℚ) = total_water →
  ((Wallace_capacity : ℚ) / (Catherine_capacity : ℚ)) = 2 :=
by
  sorry

end water_heater_ratio_l1478_147821


namespace part1_part2_l1478_147847

variables {R : Type} [LinearOrderedField R]

def setA := {x : R | -1 < x ∧ x ≤ 5}
def setB (m : R) := {x : R | x^2 - 2*x - m < 0}
def complementB (m : R) := {x : R | x ≤ -1 ∨ x ≥ 3}

theorem part1 : 
  {x : R | 6 / (x + 1) ≥ 1} = setA := 
by 
  sorry

theorem part2 (m : R) (hm : m = 3) : 
  setA ∩ complementB m = {x : R | 3 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end part1_part2_l1478_147847


namespace john_age_is_24_l1478_147814

noncomputable def john_age_condition (j d b : ℕ) : Prop :=
  j = d - 28 ∧
  j + d = 76 ∧
  j + 5 = 2 * (b + 5)

theorem john_age_is_24 (d b : ℕ) : ∃ j, john_age_condition j d b ∧ j = 24 :=
by
  use 24
  unfold john_age_condition
  sorry

end john_age_is_24_l1478_147814


namespace video_game_cost_l1478_147854

theorem video_game_cost :
  let september_saving : ℕ := 50
  let october_saving : ℕ := 37
  let november_saving : ℕ := 11
  let mom_gift : ℕ := 25
  let remaining_money : ℕ := 36
  let total_savings : ℕ := september_saving + october_saving + november_saving
  let total_with_gift : ℕ := total_savings + mom_gift
  let game_cost : ℕ := total_with_gift - remaining_money
  game_cost = 87 :=
by
  sorry

end video_game_cost_l1478_147854


namespace total_cost_of_commodities_l1478_147859

theorem total_cost_of_commodities (a b : ℕ) (h₁ : a = 477) (h₂ : a - b = 127) : a + b = 827 :=
by
  sorry

end total_cost_of_commodities_l1478_147859


namespace tires_in_parking_lot_l1478_147806

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l1478_147806


namespace percentage_of_alcohol_in_first_vessel_l1478_147863

variable (x : ℝ) -- percentage of alcohol in the first vessel in decimal form, i.e., x% is represented as x/100

-- conditions
variable (v1_capacity : ℝ := 2)
variable (v2_capacity : ℝ := 6)
variable (v2_alcohol_concentration : ℝ := 0.5)
variable (total_capacity : ℝ := 10)
variable (new_concentration : ℝ := 0.37)

theorem percentage_of_alcohol_in_first_vessel :
  (x / 100) * v1_capacity + v2_alcohol_concentration * v2_capacity = new_concentration * total_capacity -> x = 35 := 
by
  sorry

end percentage_of_alcohol_in_first_vessel_l1478_147863


namespace find_input_values_f_l1478_147825

theorem find_input_values_f (f : ℤ → ℤ) 
  (h_def : ∀ x, f (2 * x + 3) = (x - 3) * (x + 4))
  (h_val : ∃ y, f y = 170) : 
  ∃ (a b : ℤ), (a = -25 ∧ b = 29) ∧ (f a = 170 ∧ f b = 170) :=
by
  sorry

end find_input_values_f_l1478_147825


namespace same_solutions_a_value_l1478_147861

theorem same_solutions_a_value (a x : ℝ) (h1 : 2 * x + 1 = 3) (h2 : 3 - (a - x) / 3 = 1) : a = 7 := by
  sorry

end same_solutions_a_value_l1478_147861


namespace find_f_and_min_g_l1478_147886

theorem find_f_and_min_g (f g : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x : ℝ, f (2 * x - 3) = 4 * x^2 + 2 * x + 1)
  (h2 : ∀ x : ℝ, g x = f (x + a) - 7 * x):
  
  (∀ x : ℝ, f x = x^2 + 7 * x + 13) ∧
  
  (∀ a : ℝ, 
    ∀ x : ℝ, 
      (x = 1 → (a ≥ -1 → g x = a^2 + 9 * a + 14)) ∧
      (-3 < a ∧ a < -1 → g (-a) = 7 * a + 13) ∧
      (x = 3 → (a ≤ -3 → g x = a^2 + 13 * a + 22))) :=
by
  sorry

end find_f_and_min_g_l1478_147886


namespace sample_size_proof_l1478_147850

theorem sample_size_proof (p : ℝ) (N : ℤ) (n : ℤ) (h1 : N = 200) (h2 : p = 0.25) : n = 50 :=
by
  sorry

end sample_size_proof_l1478_147850


namespace ab_geq_3_plus_cd_l1478_147895

theorem ab_geq_3_plus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13) (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := 
sorry

end ab_geq_3_plus_cd_l1478_147895


namespace number_of_pairs_divisible_by_five_l1478_147807

theorem number_of_pairs_divisible_by_five :
  (∃ n : ℕ, n = 864) ↔
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 80) ∧ (1 ≤ b ∧ b ≤ 30) →
  (a * b) % 5 = 0 → (∃ n : ℕ, n = 864) := 
sorry

end number_of_pairs_divisible_by_five_l1478_147807


namespace final_temperature_is_100_l1478_147820

-- Definitions based on conditions
def initial_temperature := 20  -- in degrees
def heating_rate := 5          -- in degrees per minute
def heating_time := 16         -- in minutes

-- The proof statement
theorem final_temperature_is_100 :
  initial_temperature + heating_rate * heating_time = 100 := by
  sorry

end final_temperature_is_100_l1478_147820


namespace distinct_nonzero_reals_product_l1478_147882

theorem distinct_nonzero_reals_product 
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy: x ≠ y)
  (h : x + 3 / x = y + 3 / y) :
  x * y = 3 :=
sorry

end distinct_nonzero_reals_product_l1478_147882


namespace identify_translation_l1478_147884

def phenomenon (x : String) : Prop :=
  x = "translational"

def option_A : Prop := phenomenon "rotational"
def option_B : Prop := phenomenon "rotational"
def option_C : Prop := phenomenon "translational"
def option_D : Prop := phenomenon "rotational"

theorem identify_translation :
  (¬ option_A) ∧ (¬ option_B) ∧ option_C ∧ (¬ option_D) :=
  by {
    sorry
  }

end identify_translation_l1478_147884


namespace kyunghoon_time_to_go_down_l1478_147899

theorem kyunghoon_time_to_go_down (d : ℕ) (t_up t_down total_time : ℕ) : 
  ((t_up = d / 3) ∧ (t_down = (d + 2) / 4) ∧ (total_time = 4) → (t_up + t_down = total_time) → (t_down = 2)) := 
by
  sorry

end kyunghoon_time_to_go_down_l1478_147899


namespace complex_quadrant_l1478_147827

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l1478_147827


namespace find_k_l1478_147871

theorem find_k (k : ℝ) :
  (∀ x, x^2 + k*x + 10 = 0 → (∃ r s : ℝ, x = r ∨ x = s) ∧ r + s = -k ∧ r * s = 10) ∧
  (∀ x, x^2 - k*x + 10 = 0 → (∃ r s : ℝ, x = r + 4 ∨ x = s + 4) ∧ (r + 4) + (s + 4) = k) → 
  k = 4 :=
by
  sorry

end find_k_l1478_147871


namespace probability_of_choosing_perfect_square_is_0_08_l1478_147875

-- Definitions for the conditions
def n : ℕ := 100
def p : ℚ := 1 / 200
def probability (m : ℕ) : ℚ := if m ≤ 50 then p else 3 * p
def perfect_squares_before_50 : Finset ℕ := {1, 4, 9, 16, 25, 36, 49}
def perfect_squares_between_51_and_100 : Finset ℕ := {64, 81, 100}
def total_perfect_squares : Finset ℕ := perfect_squares_before_50 ∪ perfect_squares_between_51_and_100

-- Statement to prove that the probability of selecting a perfect square is 0.08
theorem probability_of_choosing_perfect_square_is_0_08 :
  (perfect_squares_before_50.card * p + perfect_squares_between_51_and_100.card * 3 * p) = 0.08 := 
by
  -- Adding sorry to skip the proof
  sorry

end probability_of_choosing_perfect_square_is_0_08_l1478_147875


namespace total_bouquets_sold_l1478_147832

-- Define the conditions as variables
def monday_bouquets : ℕ := 12
def tuesday_bouquets : ℕ := 3 * monday_bouquets
def wednesday_bouquets : ℕ := tuesday_bouquets / 3

-- The statement to prove
theorem total_bouquets_sold : 
  monday_bouquets + tuesday_bouquets + wednesday_bouquets = 60 :=
by
  -- The proof is omitted using sorry
  sorry

end total_bouquets_sold_l1478_147832


namespace soldiers_height_order_l1478_147819

theorem soldiers_height_order {n : ℕ} (a b : Fin n → ℝ) 
  (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) 
  (h : ∀ i, a i ≤ b i) :
  ∀ i, a i ≤ b i :=
  by sorry

end soldiers_height_order_l1478_147819


namespace intersection_M_N_l1478_147812

def M : Set ℝ := { x | x ≤ 4 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 4 } := 
by 
  sorry

end intersection_M_N_l1478_147812


namespace length_of_XY_correct_l1478_147828

noncomputable def length_of_XY (XZ : ℝ) (angleY : ℝ) (angleZ : ℝ) :=
  if angleZ = 90 ∧ angleY = 30 then 8 * Real.sqrt 3 else panic! "Invalid triangle angles"

theorem length_of_XY_correct : length_of_XY 12 30 90 = 8 * Real.sqrt 3 :=
by
  sorry

end length_of_XY_correct_l1478_147828


namespace least_number_to_add_l1478_147885

theorem least_number_to_add (n : ℕ) (h : n = 28523) : 
  ∃ x, x + n = 29560 ∧ 3 ∣ (x + n) ∧ 5 ∣ (x + n) ∧ 7 ∣ (x + n) ∧ 8 ∣ (x + n) :=
by 
  sorry

end least_number_to_add_l1478_147885


namespace find_b_c_d_sum_l1478_147838

theorem find_b_c_d_sum :
  ∃ (b c d : ℤ), (∀ n : ℕ, n > 0 → 
    a_n = b * (⌊(n : ℝ)^(1/3)⌋.natAbs : ℤ) + d ∧
    b = 2 ∧ c = 0 ∧ d = 0) ∧ (b + c + d = 2) :=
sorry

end find_b_c_d_sum_l1478_147838


namespace product_of_square_roots_l1478_147817

theorem product_of_square_roots (a b : ℝ) (h₁ : a^2 = 9) (h₂ : b^2 = 9) (h₃ : a ≠ b) : a * b = -9 :=
by
  -- Proof skipped
  sorry

end product_of_square_roots_l1478_147817


namespace sum_of_ages_l1478_147876

variables (S F : ℕ)

theorem sum_of_ages
  (h1 : F - 18 = 3 * (S - 18))
  (h2 : F = 2 * S) :
  S + F = 108 :=
by
  sorry

end sum_of_ages_l1478_147876


namespace complex_magnitude_l1478_147893

variable (a b : ℝ)

theorem complex_magnitude :
  ((1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I) →
  Complex.normSq (a + b * Complex.I) = 5/4 :=
by
  intro h
  -- Add missing logic to transform assumption to the norm result
  sorry

end complex_magnitude_l1478_147893


namespace convert_to_spherical_l1478_147804

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if y / x < 0 then Real.arctan (-y / x) + 2 * Real.pi else Real.arctan (y / x)
  (ρ, θ, φ)

theorem convert_to_spherical :
  let x := 1
  let y := -4 * Real.sqrt 3
  let z := 4
  spherical_coordinates x y z = (Real.sqrt 65, Real.arctan (-4 * Real.sqrt 3) + 2 * Real.pi, Real.arccos (4 / (Real.sqrt 65))) :=
by
  sorry

end convert_to_spherical_l1478_147804


namespace sum_of_eight_numbers_l1478_147857

theorem sum_of_eight_numbers (avg : ℝ) (num_of_items : ℕ) (h_avg : avg = 5.3) (h_items : num_of_items = 8) :
  avg * num_of_items = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l1478_147857


namespace gcd_4004_10010_l1478_147865

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 :=
by
  have h1 : 4004 = 4 * 1001 := by norm_num
  have h2 : 10010 = 10 * 1001 := by norm_num
  sorry

end gcd_4004_10010_l1478_147865


namespace necessary_but_not_sufficient_l1478_147801

-- Definitions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- The condition we are given
axiom m : ℝ

-- The quadratic equation specific condition
axiom quadratic_condition : quadratic_eq 1 2 m = 0

-- The necessary but not sufficient condition for real solutions
theorem necessary_but_not_sufficient (h : m < 2) : 
  ∃ x : ℝ, quadratic_eq 1 2 m x = 0 ∧ quadratic_eq 1 2 m x = 0 → m ≤ 1 ∨ m > 1 :=
sorry

end necessary_but_not_sufficient_l1478_147801


namespace total_players_is_28_l1478_147818

def total_players (A B C AB BC AC ABC : ℕ) : ℕ :=
  A + B + C - (AB + BC + AC) + ABC

theorem total_players_is_28 :
  total_players 10 15 18 8 6 4 3 = 28 :=
by
  -- as per inclusion-exclusion principle
  -- T = A + B + C - (AB + BC + AC) + ABC
  -- substituting given values we repeatedly perform steps until final answer
  -- take user inputs to build your final answer.
  sorry

end total_players_is_28_l1478_147818


namespace triangle_DEF_area_l1478_147873

theorem triangle_DEF_area (DE height : ℝ) (hDE : DE = 12) (hHeight : height = 15) : 
  (1/2) * DE * height = 90 :=
by
  rw [hDE, hHeight]
  norm_num

end triangle_DEF_area_l1478_147873


namespace circle_center_is_21_l1478_147810

theorem circle_center_is_21 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 5 = 0 →
                                      ∃ h k : ℝ, h = 2 ∧ k = 1 ∧ (x - h)^2 + (y - k)^2 = 10 :=
by
  intro x y h_eq
  sorry

end circle_center_is_21_l1478_147810


namespace solution_eq1_solution_eq2_l1478_147808

theorem solution_eq1 (x : ℝ) : 
  2 * x^2 - 4 * x - 1 = 0 ↔ 
  (x = 1 + (Real.sqrt 6) / 2 ∨ x = 1 - (Real.sqrt 6) / 2) := by
sorry

theorem solution_eq2 (x : ℝ) :
  (x - 1) * (x + 2) = 28 ↔ 
  (x = -6 ∨ x = 5) := by
sorry

end solution_eq1_solution_eq2_l1478_147808


namespace largest_possible_A_l1478_147848

theorem largest_possible_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 :=
by
  have h3 : A ≤ 10 + 4 := sorry
  exact h3

end largest_possible_A_l1478_147848


namespace solve_equation_and_find_c_d_l1478_147837

theorem solve_equation_and_find_c_d : 
  ∃ (c d : ℕ), (∃ x : ℝ, x^2 + 14 * x = 84 ∧ x = Real.sqrt c - d) ∧ c + d = 140 := 
sorry

end solve_equation_and_find_c_d_l1478_147837


namespace no_natural_pairs_exist_l1478_147897

theorem no_natural_pairs_exist (n m : ℕ) : ¬(n + 1) * (2 * n + 1) = 18 * m ^ 2 :=
by
  sorry

end no_natural_pairs_exist_l1478_147897


namespace find_x_for_parallel_l1478_147846

-- Definitions for vector components and parallel condition.
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_x_for_parallel :
  ∃ x : ℝ, parallel a (b x) ∧ x = -3 / 2 :=
by
  -- The statement to be proven
  sorry

end find_x_for_parallel_l1478_147846


namespace graph_translation_l1478_147883

theorem graph_translation (f : ℝ → ℝ) (x : ℝ) (h : f 1 = -1) :
  f (x - 1) - 1 = -2 :=
by
  sorry

end graph_translation_l1478_147883


namespace game_a_greater_than_game_c_l1478_147881

-- Definitions of probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probabilities for Game A and Game C based on given conditions
def prob_game_a : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)
def prob_game_c : ℚ :=
  (prob_heads ^ 5) +
  (prob_tails ^ 5) +
  (prob_heads ^ 3 * prob_tails ^ 2) +
  (prob_tails ^ 3 * prob_heads ^ 2)

-- Define the difference
def prob_difference : ℚ := prob_game_a - prob_game_c

-- The theorem to be proved
theorem game_a_greater_than_game_c :
  prob_difference = 3 / 64 :=
by
  sorry

end game_a_greater_than_game_c_l1478_147881
