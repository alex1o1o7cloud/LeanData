import Mathlib

namespace simplify_complex_expression_l629_62980

theorem simplify_complex_expression (i : ℂ) (h_i : i * i = -1) : 
  (11 - 3 * i) / (1 + 2 * i) = 3 - 5 * i :=
sorry

end simplify_complex_expression_l629_62980


namespace find_a_from_perpendicular_lines_l629_62996

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l629_62996


namespace result_number_of_edges_l629_62907

-- Define the conditions
def hexagon (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 6 ∧ edges = 6 * side_length)
def triangle (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 3 ∧ edges = 3 * side_length)

-- State the theorem
theorem result_number_of_edges (side_length_hex : ℕ) (side_length_tri : ℕ)
  (h_h : hexagon side_length_hex) (h_t : triangle side_length_tri)
  (aligned_edge_to_edge : side_length_hex = side_length_tri ∧ side_length_hex = 1 ∧ side_length_tri = 1) :
  ∃ edges, edges = 5 :=
by
  -- Proof is not provided, it is marked with sorry
  sorry

end result_number_of_edges_l629_62907


namespace probability_two_queens_or_at_least_one_jack_l629_62923

-- Definitions
def num_jacks : ℕ := 4
def num_queens : ℕ := 4
def total_cards : ℕ := 52

-- Probability calculation for drawing either two Queens or at least one Jack
theorem probability_two_queens_or_at_least_one_jack :
  (4 / 52) * (3 / (52 - 1)) + ((4 / 52) * (48 / (52 - 1)) + (48 / 52) * (4 / (52 - 1)) + (4 / 52) * (3 / (52 - 1))) = 2 / 13 :=
by
  sorry

end probability_two_queens_or_at_least_one_jack_l629_62923


namespace percentage_sum_l629_62912

noncomputable def womenWithRedHairBelow30 : ℝ := 0.07
noncomputable def menWithDarkHair30OrOlder : ℝ := 0.13

theorem percentage_sum :
  womenWithRedHairBelow30 + menWithDarkHair30OrOlder = 0.20 := by
  sorry -- Proof is omitted

end percentage_sum_l629_62912


namespace audrey_dreaming_fraction_l629_62917

theorem audrey_dreaming_fraction
  (total_asleep_time : ℕ) 
  (not_dreaming_time : ℕ)
  (dreaming_time : ℕ)
  (fraction_dreaming : ℚ)
  (h_total_asleep : total_asleep_time = 10)
  (h_not_dreaming : not_dreaming_time = 6)
  (h_dreaming : dreaming_time = total_asleep_time - not_dreaming_time)
  (h_fraction : fraction_dreaming = dreaming_time / total_asleep_time) :
  fraction_dreaming = 2 / 5 := 
by {
  sorry
}

end audrey_dreaming_fraction_l629_62917


namespace probability_of_region_l629_62933

theorem probability_of_region :
  let area_rect := (1000: ℝ) * 1500
  let area_polygon := 500000
  let prob := area_polygon / area_rect
  prob = (1 / 3) := sorry

end probability_of_region_l629_62933


namespace q_at_1_is_zero_l629_62910

-- Define the function q : ℝ → ℝ
-- The conditions imply q(1) = 0
axiom q : ℝ → ℝ

-- Given that (1, 0) is on the graph of y = q(x)
axiom q_condition : q 1 = 0

-- Prove q(1) = 0 given the condition that (1, 0) is on the graph
theorem q_at_1_is_zero : q 1 = 0 :=
by
  exact q_condition

end q_at_1_is_zero_l629_62910


namespace Sandy_total_marks_l629_62988

theorem Sandy_total_marks
  (correct_marks_per_sum : ℤ)
  (incorrect_marks_per_sum : ℤ)
  (total_sums : ℕ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ)
  (total_marks : ℤ) :
  correct_marks_per_sum = 3 →
  incorrect_marks_per_sum = -2 →
  total_sums = 30 →
  correct_sums = 24 →
  incorrect_sums = total_sums - correct_sums →
  total_marks = correct_marks_per_sum * correct_sums + incorrect_marks_per_sum * incorrect_sums →
  total_marks = 60 :=
by
  sorry

end Sandy_total_marks_l629_62988


namespace line_passing_quadrants_l629_62965

-- Definition of the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Prove that the line passes through Quadrants I, III, and IV
theorem line_passing_quadrants :
  ∃ x_1 > 0, line x_1 > 0 ∧      -- Quadrant I
  ∃ x_3 < 0, line x_3 < 0 ∧      -- Quadrant III
  ∃ x_4 > 0, line x_4 < 0 :=     -- Quadrant IV
sorry

end line_passing_quadrants_l629_62965


namespace total_cost_correct_l629_62903

-- Define the conditions
def total_employees : ℕ := 300
def emp_12_per_hour : ℕ := 200
def emp_14_per_hour : ℕ := 40
def emp_17_per_hour : ℕ := total_employees - emp_12_per_hour - emp_14_per_hour

def wage_12_per_hour : ℕ := 12
def wage_14_per_hour : ℕ := 14
def wage_17_per_hour : ℕ := 17

def hours_per_shift : ℕ := 8

-- Define the cost calculations
def cost_12 : ℕ := emp_12_per_hour * wage_12_per_hour * hours_per_shift
def cost_14 : ℕ := emp_14_per_hour * wage_14_per_hour * hours_per_shift
def cost_17 : ℕ := emp_17_per_hour * wage_17_per_hour * hours_per_shift

def total_cost : ℕ := cost_12 + cost_14 + cost_17

-- The theorem to be proved
theorem total_cost_correct :
  total_cost = 31840 :=
by
  sorry

end total_cost_correct_l629_62903


namespace find_x_in_triangle_XYZ_l629_62932

theorem find_x_in_triangle_XYZ (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ) (hx : y = 7) (hz : z = 6) (hcos : cos_Y_minus_Z = 47 / 64) : 
    ∃ x : ℝ, x = Real.sqrt 63.75 :=
by
  -- The proof will go here, but it is skipped for now.
  sorry

end find_x_in_triangle_XYZ_l629_62932


namespace traffic_light_probability_l629_62949

theorem traffic_light_probability :
  let total_cycle_time := 63
  let green_time := 30
  let yellow_time := 3
  let red_time := 30
  let observation_window := 3
  let change_intervals := 3 * 3
  ∃ (P : ℚ), P = change_intervals / total_cycle_time ∧ P = 1 / 7 := 
by
  sorry

end traffic_light_probability_l629_62949


namespace gloria_initial_dimes_l629_62967

variable (Q D : ℕ)

theorem gloria_initial_dimes (h1 : D = 5 * Q) 
                             (h2 : (3 * Q) / 5 + D = 392) : 
                             D = 350 := 
by {
  sorry
}

end gloria_initial_dimes_l629_62967


namespace find_triplet_l629_62999

theorem find_triplet (x y z : ℕ) :
  (1 + (x : ℚ) / (y + z))^2 + (1 + (y : ℚ) / (z + x))^2 + (1 + (z : ℚ) / (x + y))^2 = 27 / 4 ↔ (x, y, z) = (1, 1, 1) :=
by
  sorry

end find_triplet_l629_62999


namespace afternoon_to_morning_ratio_l629_62956

theorem afternoon_to_morning_ratio (total_kg : ℕ) (afternoon_kg : ℕ) (morning_kg : ℕ) 
  (h1 : total_kg = 390) (h2 : afternoon_kg = 260) (h3 : morning_kg = total_kg - afternoon_kg) :
  afternoon_kg / morning_kg = 2 :=
sorry

end afternoon_to_morning_ratio_l629_62956


namespace parabola_min_area_l629_62908

-- Definition of the parabola C with vertex at the origin and focus on the positive y-axis
-- (Conditions 1 and 2)
def parabola_eq (x y : ℝ) : Prop := x^2 = 6 * y

-- Line l defined by mx + y - 3/2 = 0 (Condition 3)
def line_eq (m x y : ℝ) : Prop := m * x + y - 3 / 2 = 0

-- Formal statement combining all conditions to prove the equivalent Lean statement
theorem parabola_min_area :
  (∀ x y : ℝ, parabola_eq x y ↔ x^2 = 6 * y) ∧
  (∀ m x y : ℝ, line_eq m x y ↔ m * x + y - 3 / 2 = 0) →
  (parabola_eq 0 0) ∧ (∃ y > 0, parabola_eq 0 y ∧ line_eq 0 0 (y/2) ∧ y = 3 / 2) ∧
  ∀ A B P : ℝ, line_eq 0 A B ∧ line_eq 0 B P ∧ A^2 + B^2 > 0 → 
  ∃ min_S : ℝ, min_S = 9 :=
by
  sorry

end parabola_min_area_l629_62908


namespace avg_price_per_book_l629_62937

theorem avg_price_per_book (n1 n2 p1 p2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 55) (h3 : p1 = 1380) (h4 : p2 = 900) :
    (p1 + p2) / (n1 + n2) = 19 := by
  sorry

end avg_price_per_book_l629_62937


namespace cake_cubes_with_exactly_two_faces_iced_l629_62992

theorem cake_cubes_with_exactly_two_faces_iced :
  let cake : ℕ := 3 -- cake dimension
  let total_cubes : ℕ := cake ^ 3 -- number of smaller cubes (total 27)
  let cubes_with_two_faces_icing := 4
  (∀ cake icing (smaller_cubes : ℕ), icing ≠ 0 → smaller_cubes = cake ^ 3 → 
    let top_iced := cake - 2 -- cubes with icing on top only
    let front_iced := cake - 2 -- cubes with icing on front only
    let back_iced := cake - 2 -- cubes with icing on back only
    ((top_iced * 2) = cubes_with_two_faces_icing)) :=
  sorry

end cake_cubes_with_exactly_two_faces_iced_l629_62992


namespace anna_baked_60_cupcakes_l629_62970

variable (C : ℕ)
variable (h1 : (1/5 : ℚ) * C - 3 = 9)

theorem anna_baked_60_cupcakes (h1 : (1/5 : ℚ) * C - 3 = 9) : C = 60 :=
sorry

end anna_baked_60_cupcakes_l629_62970


namespace cube_root_of_8_is_2_l629_62930

theorem cube_root_of_8_is_2 : ∃ x : ℝ, x ^ 3 = 8 ∧ x = 2 :=
by
  have h : (2 : ℝ) ^ 3 = 8 := by norm_num
  exact ⟨2, h, rfl⟩

end cube_root_of_8_is_2_l629_62930


namespace check_inequality_l629_62914

theorem check_inequality : 1.7^0.3 > 0.9^3.1 :=
sorry

end check_inequality_l629_62914


namespace digit_matching_equalities_l629_62921

theorem digit_matching_equalities :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 99 → 0 ≤ b ∧ b ≤ 99 →
    ((a = 98 ∧ b = 1 ∧ (98 + 1)^2 = 100*98 + 1) ∨
     (a = 20 ∧ b = 25 ∧ (20 + 25)^2 = 100*20 + 25)) :=
by
  intros a b ha hb
  sorry

end digit_matching_equalities_l629_62921


namespace min_value_of_sum_range_of_x_l629_62991

noncomputable def ab_condition (a b : ℝ) : Prop := a + b = 1
noncomputable def ra_positive (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Problem 1: Minimum value of (1/a + 4/b)

theorem min_value_of_sum (a b : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    ∃ m : ℝ, m = 9 ∧ ∀ a b, ab_condition a b → ra_positive a b → 
    (1 / a + 4 / b) ≥ m :=
by sorry

-- Problem 2: Range of x for which the inequality holds

theorem range_of_x (a b x : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| → x ∈ Set.Icc (-7 : ℝ) 11 :=
by sorry

end min_value_of_sum_range_of_x_l629_62991


namespace inverse_proportion_function_point_l629_62942

theorem inverse_proportion_function_point (k x y : ℝ) (h₁ : 1 = k / (-6)) (h₂ : y = k / x) :
  k = -6 ∧ (x = 2 ∧ y = -3 ↔ y = -k / x) :=
by
  sorry

end inverse_proportion_function_point_l629_62942


namespace equation_of_line_containing_BC_l629_62998

theorem equation_of_line_containing_BC (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (altitude_from_CA : ∀ x y : ℝ, 2 * x - 3 * y + 1 = 0)
  (altitude_from_BA : ∀ x y : ℝ, x + y = 1)
  (eq_BC : ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b) :
  ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b) ∧ 2 * x + 3 * y + 7 = 0 :=
sorry

end equation_of_line_containing_BC_l629_62998


namespace lattice_intersections_l629_62976

theorem lattice_intersections (squares : ℕ) (circles : ℕ) 
        (line_segment : ℤ × ℤ → ℤ × ℤ) 
        (radius : ℚ) (side_length : ℚ) : 
        line_segment (0, 0) = (1009, 437) → 
        radius = 1/8 → side_length = 1/4 → 
        (squares + circles = 430) :=
by
  sorry

end lattice_intersections_l629_62976


namespace largest_int_less_than_100_remainder_4_l629_62977

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l629_62977


namespace boat_travel_distance_per_day_l629_62987

-- Definitions from conditions
def men : ℕ := 25
def water_daily_per_man : ℚ := 1/2
def travel_distance : ℕ := 4000
def total_water : ℕ := 250

-- Main theorem
theorem boat_travel_distance_per_day : 
  ∀ (men : ℕ) (water_daily_per_man : ℚ) (travel_distance : ℕ) (total_water : ℕ), 
  men = 25 ∧ water_daily_per_man = 1/2 ∧ travel_distance = 4000 ∧ total_water = 250 ->
  travel_distance / (total_water / (men * water_daily_per_man)) = 200 :=
by
  sorry

end boat_travel_distance_per_day_l629_62987


namespace length_of_bridge_correct_l629_62963

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time_seconds
  total_distance - train_length

theorem length_of_bridge_correct :
  length_of_bridge 500 42 60 = 200.2 :=
by
  sorry -- Proof of the theorem

end length_of_bridge_correct_l629_62963


namespace safe_dishes_count_l629_62993

theorem safe_dishes_count (total_dishes vegan_dishes vegan_with_nuts : ℕ) 
  (h1 : vegan_dishes = total_dishes / 3) 
  (h2 : vegan_with_nuts = 4) 
  (h3 : vegan_dishes = 6) : vegan_dishes - vegan_with_nuts = 2 :=
by
  sorry

end safe_dishes_count_l629_62993


namespace find_a_for_quadratic_max_l629_62939

theorem find_a_for_quadratic_max :
  ∃ a : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ 1/2 → (x^2 + 2 * x - 2 ≤ 1)) ∧
           (∃ x : ℝ, a ≤ x ∧ x ≤ 1/2 ∧ (x^2 + 2 * x - 2 = 1)) ∧ 
           a = -3 :=
sorry

end find_a_for_quadratic_max_l629_62939


namespace lucas_investment_l629_62943

noncomputable def investment_amount (y : ℝ) : ℝ := 1500 - y

theorem lucas_investment :
  ∃ y : ℝ, (y * 1.04 + (investment_amount y) * 1.06 = 1584.50) ∧ y = 275 :=
by
  sorry

end lucas_investment_l629_62943


namespace find_g_9_l629_62901

-- Define the function g
def g (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 7

-- Given conditions
variables (a b c : ℝ)

-- g(-9) = 9
axiom h : g a b c (-9) = 9

-- Prove g(9) = -23
theorem find_g_9 : g a b c 9 = -23 :=
by
  sorry

end find_g_9_l629_62901


namespace condition_eq_l629_62944

-- We are given a triangle ABC with sides opposite angles A, B, and C being a, b, and c respectively.
variable (A B C a b c : ℝ)

-- Conditions for the problem
def sin_eq (A B : ℝ) := Real.sin A = Real.sin B
def cos_eq (A B : ℝ) := Real.cos A = Real.cos B
def sin2_eq (A B : ℝ) := Real.sin (2 * A) = Real.sin (2 * B)
def cos2_eq (A B : ℝ) := Real.cos (2 * A) = Real.cos (2 * B)

-- The main statement we need to prove
theorem condition_eq (h1 : sin_eq A B) (h2 : cos_eq A B) (h4 : cos2_eq A B) : a = b :=
sorry

end condition_eq_l629_62944


namespace distance_interval_l629_62906

theorem distance_interval (d : ℝ) (h₁ : ¬ (d ≥ 8)) (h₂ : ¬ (d ≤ 6)) (h₃ : ¬ (d ≤ 3)) : 6 < d ∧ d < 8 := by
  sorry

end distance_interval_l629_62906


namespace inequality_solution_l629_62929

theorem inequality_solution (x : ℝ) (h : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1)) : x > 1 := 
sorry

end inequality_solution_l629_62929


namespace hyperbola_focal_distance_and_asymptotes_l629_62919

-- Define the hyperbola
def hyperbola (y x : ℝ) : Prop := (y^2 / 4) - (x^2 / 3) = 1

-- Prove the properties
theorem hyperbola_focal_distance_and_asymptotes :
  (∀ y x : ℝ, hyperbola y x → ∃ c : ℝ, c = 2 * Real.sqrt 7)
  ∧
  (∀ y x : ℝ, hyperbola y x → (y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x)) :=
by
  sorry

end hyperbola_focal_distance_and_asymptotes_l629_62919


namespace scaled_det_l629_62964

variable (x y z a b c p q r : ℝ)
variable (det_orig : ℝ)
variable (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2)

theorem scaled_det (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2) :
  Matrix.det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 :=
by
  sorry

end scaled_det_l629_62964


namespace expected_value_proof_l629_62969

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l629_62969


namespace beta_gt_half_alpha_l629_62985

theorem beta_gt_half_alpha (alpha beta : ℝ) (h1 : Real.sin beta = (3/4) * Real.sin alpha) (h2 : 0 < alpha ∧ alpha ≤ 90) : beta > alpha / 2 :=
by
  sorry

end beta_gt_half_alpha_l629_62985


namespace tangerines_times_persimmons_l629_62918

-- Definitions from the problem conditions
def apples : ℕ := 24
def tangerines : ℕ := 6 * apples
def persimmons : ℕ := 8

-- Statement to be proved
theorem tangerines_times_persimmons :
  tangerines / persimmons = 18 := by
  sorry

end tangerines_times_persimmons_l629_62918


namespace find_n_cosine_l629_62936

theorem find_n_cosine : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (∃ m : ℤ, n = 25 + 360 * m ∨ n = -25 + 360 * m) :=
by
  sorry

end find_n_cosine_l629_62936


namespace mitchell_pencils_l629_62909

/-- Mitchell and Antonio have a combined total of 54 pencils.
Mitchell has 6 more pencils than Antonio. -/
theorem mitchell_pencils (A M : ℕ) 
  (h1 : M = A + 6)
  (h2 : M + A = 54) : M = 30 :=
by
  sorry

end mitchell_pencils_l629_62909


namespace simplify_to_quadratic_l629_62950

noncomputable def simplify_expression (a b c x : ℝ) : ℝ := 
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c + 2)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem simplify_to_quadratic {a b c x : ℝ} (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  simplify_expression a b c x = x^2 - (a + b + c) * x + sorry :=
sorry

end simplify_to_quadratic_l629_62950


namespace max_value_X2_plus_2XY_plus_3Y2_l629_62989

theorem max_value_X2_plus_2XY_plus_3Y2 
  (x y : ℝ) 
  (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  x^2 + 2 * x * y + 3 * y^2 ≤ 30 + 20 * Real.sqrt 3 :=
sorry

end max_value_X2_plus_2XY_plus_3Y2_l629_62989


namespace perimeter_of_triangle_l629_62913

theorem perimeter_of_triangle
  (P : ℝ)
  (r : ℝ := 1.5)
  (A : ℝ := 29.25)
  (h : A = r * (P / 2)) :
  P = 39 :=
by
  sorry

end perimeter_of_triangle_l629_62913


namespace distribute_pictures_l629_62979

/-
Tiffany uploaded 34 pictures from her phone, 55 from her camera,
and 12 from her tablet to Facebook. If she sorted the pics into 7 different albums
with the same amount of pics in each album, how many pictures were in each of the albums?
-/

theorem distribute_pictures :
  let phone_pics := 34
  let camera_pics := 55
  let tablet_pics := 12
  let total_pics := phone_pics + camera_pics + tablet_pics
  let albums := 7
  ∃ k r, (total_pics = k * albums + r) ∧ (r < albums) := by
  sorry

end distribute_pictures_l629_62979


namespace max_cos_y_cos_x_l629_62904

noncomputable def max_cos_sum : ℝ :=
  1 + (Real.sqrt (2 + Real.sqrt 2)) / 2

theorem max_cos_y_cos_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (x y : ℝ), Real.cos y + Real.cos x = max_cos_sum :=
sorry

end max_cos_y_cos_x_l629_62904


namespace selling_price_correct_l629_62972

noncomputable def selling_price (purchase_price : ℝ) (overhead_expenses : ℝ) (profit_percent : ℝ) : ℝ :=
  let total_cost_price := purchase_price + overhead_expenses
  let profit := (profit_percent / 100) * total_cost_price
  total_cost_price + profit

theorem selling_price_correct :
    selling_price 225 28 18.577075098814234 = 300 := by
  sorry

end selling_price_correct_l629_62972


namespace minimum_value_of_2x_plus_y_l629_62974

theorem minimum_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) : 2 * x + y ≥ 12 :=
  sorry

end minimum_value_of_2x_plus_y_l629_62974


namespace fraction_simplify_l629_62938

theorem fraction_simplify:
  (1/5 + 1/7) / (3/8 - 1/9) = 864 / 665 :=
by
  sorry

end fraction_simplify_l629_62938


namespace solve_linear_system_l629_62986

theorem solve_linear_system :
  ∃ (x y : ℝ), (x + 3 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 2) ∧ (y = -1) :=
  sorry

end solve_linear_system_l629_62986


namespace tan_150_degrees_l629_62961

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l629_62961


namespace total_cost_full_units_l629_62968

def total_units : Nat := 12
def cost_1_bedroom : Nat := 360
def cost_2_bedroom : Nat := 450
def num_2_bedroom : Nat := 7
def num_1_bedroom : Nat := total_units - num_2_bedroom

def total_cost : Nat := (num_1_bedroom * cost_1_bedroom) + (num_2_bedroom * cost_2_bedroom)

theorem total_cost_full_units : total_cost = 4950 := by
  -- proof would go here
  sorry

end total_cost_full_units_l629_62968


namespace nublian_total_words_l629_62951

-- Define the problem's constants and conditions
def nublian_alphabet_size := 6
def word_length_one := nublian_alphabet_size
def word_length_two := nublian_alphabet_size * nublian_alphabet_size
def word_length_three := nublian_alphabet_size * nublian_alphabet_size * nublian_alphabet_size

-- Define the total number of words
def total_words := word_length_one + word_length_two + word_length_three

-- Main theorem statement
theorem nublian_total_words : total_words = 258 := by
  sorry

end nublian_total_words_l629_62951


namespace six_digit_square_number_cases_l629_62947

theorem six_digit_square_number_cases :
  ∃ n : ℕ, 316 ≤ n ∧ n < 1000 ∧ (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by {
  sorry
}

end six_digit_square_number_cases_l629_62947


namespace no_afg_fourth_place_l629_62935

theorem no_afg_fourth_place
  (A B C D E F G : ℕ)
  (h1 : A < B)
  (h2 : A < C)
  (h3 : B < D)
  (h4 : C < E)
  (h5 : A < F ∧ F < B)
  (h6 : B < G ∧ G < C) :
  ¬ (A = 4 ∨ F = 4 ∨ G = 4) :=
by
  sorry

end no_afg_fourth_place_l629_62935


namespace smallest_digit_not_found_in_units_place_of_odd_number_l629_62981

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l629_62981


namespace model_N_completion_time_l629_62966

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end model_N_completion_time_l629_62966


namespace relationship_between_b_and_c_l629_62990

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

end relationship_between_b_and_c_l629_62990


namespace f_neg2_eq_neg1_l629_62995

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg2_eq_neg1 : f (-2) = -1 := by
  sorry

end f_neg2_eq_neg1_l629_62995


namespace rectangle_area_l629_62922

def length : ℝ := 2
def width : ℝ := 4
def area := length * width

theorem rectangle_area : area = 8 := 
by
  -- Proof can be written here
  sorry

end rectangle_area_l629_62922


namespace not_divisible_l629_62934

theorem not_divisible (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 12) : ¬∃ k : ℕ, 120 * a + 2 * b = k * (100 * a + b) := 
sorry

end not_divisible_l629_62934


namespace find_c_l629_62931

theorem find_c (a b c : ℝ) (h : 1/a + 1/b = 1/c) : c = (a * b) / (a + b) := 
by
  sorry

end find_c_l629_62931


namespace solve_fraction_equation_l629_62984

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l629_62984


namespace cost_per_adult_meal_l629_62925

theorem cost_per_adult_meal (total_people : ℕ) (num_kids : ℕ) (total_cost : ℕ) (cost_per_kid : ℕ) :
  total_people = 12 →
  num_kids = 7 →
  cost_per_kid = 0 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids)) = 3 :=
by
  intros
  sorry

end cost_per_adult_meal_l629_62925


namespace modified_goldbach_2024_l629_62955

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n > 1 → n < p → ¬ (p % n = 0)

theorem modified_goldbach_2024 :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ is_prime p1 ∧ is_prime p2 ∧ p1 + p2 = 2024 := 
sorry

end modified_goldbach_2024_l629_62955


namespace grid_area_l629_62941

theorem grid_area :
  let B := 10   -- Number of boundary points
  let I := 12   -- Number of interior points
  I + B / 2 - 1 = 16 :=
by
  sorry

end grid_area_l629_62941


namespace smallest_pos_n_l629_62940

theorem smallest_pos_n (n : ℕ) (h : 435 * n % 30 = 867 * n % 30) : n = 5 :=
by
  sorry

end smallest_pos_n_l629_62940


namespace part1_find_a_b_part2_inequality_l629_62926

theorem part1_find_a_b (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 * x + 1| + |x + a|) 
  (h_sol : ∀ x, f x ≤ 3 ↔ b ≤ x ∧ x ≤ 1) : 
  a = -1 ∧ b = -1 :=
sorry

theorem part2_inequality (m n : ℝ) (a : ℝ) (h_m : 0 < m) (h_n : 0 < n) 
  (h_eq : (1 / (2 * m)) + (2 / n) + 2 * a = 0) (h_a : a = -1) : 
  4 * m^2 + n^2 ≥ 4 :=
sorry

end part1_find_a_b_part2_inequality_l629_62926


namespace relationship_between_a_and_b_l629_62911

theorem relationship_between_a_and_b : 
  ∀ (a b : ℝ), (∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → (2 + 2*a)*x + (2 + 2*b)*y - a^2 - 1 = 0) → a^2 + 2*a + 2*b + 5 = 0 :=
by
  intros a b hyp
  sorry

end relationship_between_a_and_b_l629_62911


namespace largest_divisor_360_450_l629_62957

theorem largest_divisor_360_450 : ∃ d, (d ∣ 360 ∧ d ∣ 450) ∧ (∀ e, (e ∣ 360 ∧ e ∣ 450) → e ≤ d) ∧ d = 90 :=
by
  sorry

end largest_divisor_360_450_l629_62957


namespace least_number_of_coins_l629_62959

theorem least_number_of_coins : ∃ (n : ℕ), 
  (n % 6 = 3) ∧ 
  (n % 4 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, (m % 6 = 3) ∧ (m % 4 = 1) ∧ (m % 7 = 2) → n ≤ m) :=
by
  exists 9
  simp
  sorry

end least_number_of_coins_l629_62959


namespace solve_problem_1_solve_problem_2_l629_62971

-- Problem statement 1: Prove that the solutions to x(x-2) = x-2 are x = 1 and x = 2.
theorem solve_problem_1 (x : ℝ) : (x * (x - 2) = x - 2) ↔ (x = 1 ∨ x = 2) :=
  sorry

-- Problem statement 2: Prove that the solutions to 2x^2 + 3x - 5 = 0 are x = 1 and x = -5/2.
theorem solve_problem_2 (x : ℝ) : (2 * x^2 + 3 * x - 5 = 0) ↔ (x = 1 ∨ x = -5 / 2) :=
  sorry

end solve_problem_1_solve_problem_2_l629_62971


namespace seeds_per_flowerbed_l629_62960

theorem seeds_per_flowerbed (total_seeds : ℕ) (flowerbeds : ℕ) (seeds_per_bed : ℕ) 
  (h1 : total_seeds = 45) (h2 : flowerbeds = 9) 
  (h3 : total_seeds = flowerbeds * seeds_per_bed) : seeds_per_bed = 5 :=
by sorry

end seeds_per_flowerbed_l629_62960


namespace sum_of_primes_l629_62952

theorem sum_of_primes (p1 p2 p3 : ℕ) (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) 
    (h : p1 * p2 * p3 = 31 * (p1 + p2 + p3)) :
    p1 + p2 + p3 = 51 := by
  sorry

end sum_of_primes_l629_62952


namespace point_M_coordinates_l629_62916

theorem point_M_coordinates (a : ℤ) (h : a + 3 = 0) : (a + 3, 2 * a - 2) = (0, -8) :=
by
  sorry

end point_M_coordinates_l629_62916


namespace smallest_sector_angle_division_is_10_l629_62948

/-
  Prove that the smallest possible sector angle in a 15-sector division of a circle,
  where the central angles form an arithmetic sequence with integer values and the
  total sum of angles is 360 degrees, is 10 degrees.
-/
theorem smallest_sector_angle_division_is_10 :
  ∃ (a1 d : ℕ), (∀ i, i ∈ (List.range 15) → a1 + i * d > 0) ∧ (List.sum (List.map (fun i => a1 + i * d) (List.range 15)) = 360) ∧
  a1 = 10 := by
  sorry

end smallest_sector_angle_division_is_10_l629_62948


namespace find_prob_A_l629_62973

variable (P : String → ℝ)
variable (A B : String)

-- Conditions
axiom prob_complement_twice : P B = 2 * P A
axiom prob_sum_to_one : P A + P B = 1

-- Statement to be proved
theorem find_prob_A : P A = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end find_prob_A_l629_62973


namespace find_k_l629_62945

-- Definitions based on the conditions
def number := 24
def bigPart := 13
  
theorem find_k (x y k : ℕ) 
  (original_number : x + y = 24)
  (big_part : x = 13 ∨ y = 13)
  (equation : k * x + 5 * y = 146) : k = 7 := 
  sorry

end find_k_l629_62945


namespace minimum_distance_l629_62958

theorem minimum_distance (x y : ℝ) (h : x - y - 1 = 0) : (x - 2)^2 + (y - 2)^2 ≥ 1 / 2 :=
sorry

end minimum_distance_l629_62958


namespace simplify_cosine_expression_l629_62946

theorem simplify_cosine_expression :
  ∀ (θ : ℝ), θ = 30 * Real.pi / 180 → (1 - Real.cos θ) * (1 + Real.cos θ) = 1 / 4 :=
by
  intro θ hθ
  have cos_30 := Real.cos θ
  rewrite [hθ]
  sorry

end simplify_cosine_expression_l629_62946


namespace fraction_difference_l629_62924

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : (1 / a - 1 / b) = -2 := 
by
  sorry

end fraction_difference_l629_62924


namespace two_a_minus_b_l629_62954

variables (a b : ℝ × ℝ)
variables (m : ℝ)

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem two_a_minus_b 
  (ha : a = (1, -2))
  (hb : b = (m, 4))
  (h_parallel : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end two_a_minus_b_l629_62954


namespace set_properties_proof_l629_62994

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := Icc (-2 : ℝ) 2)
variable (N : Set ℝ := Iic (1 : ℝ))

theorem set_properties_proof :
  (M ∪ N = Iic (2 : ℝ)) ∧
  (M ∩ N = Icc (-2 : ℝ) 1) ∧
  (U \ N = Ioi (1 : ℝ)) := by
  sorry

end set_properties_proof_l629_62994


namespace highlighter_difference_l629_62927

theorem highlighter_difference :
  ∀ (yellow pink blue : ℕ),
    yellow = 7 →
    pink = yellow + 7 →
    yellow + pink + blue = 40 →
    blue - pink = 5 :=
by
  intros yellow pink blue h_yellow h_pink h_total
  rw [h_yellow, h_pink] at h_total
  sorry

end highlighter_difference_l629_62927


namespace four_numbers_divisible_by_2310_in_4_digit_range_l629_62997

/--
There exist exactly 4 numbers within the range of 1000 to 9999 that are divisible by 2310.
-/
theorem four_numbers_divisible_by_2310_in_4_digit_range :
  ∃ n₁ n₂ n₃ n₄,
    1000 ≤ n₁ ∧ n₁ ≤ 9999 ∧ n₁ % 2310 = 0 ∧
    1000 ≤ n₂ ∧ n₂ ≤ 9999 ∧ n₂ % 2310 = 0 ∧ n₁ < n₂ ∧
    1000 ≤ n₃ ∧ n₃ ≤ 9999 ∧ n₃ % 2310 = 0 ∧ n₂ < n₃ ∧
    1000 ≤ n₄ ∧ n₄ ≤ 9999 ∧ n₄ % 2310 = 0 ∧ n₃ < n₄ ∧
    ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2310 = 0 → (n = n₁ ∨ n = n₂ ∨ n = n₃ ∨ n = n₄) :=
by
  sorry

end four_numbers_divisible_by_2310_in_4_digit_range_l629_62997


namespace unit_price_solution_purchase_plan_and_costs_l629_62975

-- Definitions based on the conditions (Note: numbers and relationships purely)
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 2 * y = 60 ∧ x + 3 * y = 55

def prize_purchase_conditions (m n : ℕ) : Prop :=
  m + n = 100 ∧ 10 * m + 15 * n ≤ 1160 ∧ m ≤ 3 * n

-- Proving that the unit prices found match the given constraints
theorem unit_price_solution : ∃ x y : ℕ, unit_prices x y := by
  sorry

-- Proving the number of purchasing plans and minimum cost
theorem purchase_plan_and_costs : 
  (∃ (num_plans : ℕ) (min_cost : ℕ), 
    num_plans = 8 ∧ min_cost = 1125 ∧ 
    ∀ m n : ℕ, prize_purchase_conditions m n → 
      ((68 ≤ m ∧ m ≤ 75) →
      10 * m + 15 * (100 - m) = min_cost)) := by
  sorry

end unit_price_solution_purchase_plan_and_costs_l629_62975


namespace similar_triangles_perimeter_l629_62915

theorem similar_triangles_perimeter (P_small P_large : ℝ) 
  (h_ratio : P_small / P_large = 2 / 3) 
  (h_sum : P_small + P_large = 20) : 
  P_small = 8 := 
sorry

end similar_triangles_perimeter_l629_62915


namespace reflection_across_x_axis_l629_62983

theorem reflection_across_x_axis (x y : ℝ) : 
  (x, -y) = (-2, -4) ↔ (x, y) = (-2, 4) :=
by
  sorry

end reflection_across_x_axis_l629_62983


namespace new_cost_percentage_l629_62902

variable (t b : ℝ)

-- Define the original cost
def original_cost : ℝ := t * b ^ 4

-- Define the new cost when b is doubled
def new_cost : ℝ := t * (2 * b) ^ 4

-- The theorem statement
theorem new_cost_percentage (t b : ℝ) : new_cost t b = 16 * original_cost t b := 
by
  -- Proof steps are skipped
  sorry

end new_cost_percentage_l629_62902


namespace sin_equations_solution_l629_62928

theorem sin_equations_solution {k : ℤ} (hk : k ≤ 1 ∨ k ≥ 5) : 
  (∃ x : ℝ, 2 * x = π * k ∧ x = (π * k) / 2) ∨ x = 7 * π / 4 :=
by
  sorry

end sin_equations_solution_l629_62928


namespace area_of_shape_is_correct_l629_62978

noncomputable def square_side_length : ℝ := 2 * Real.pi

noncomputable def semicircle_radius : ℝ := square_side_length / 2

noncomputable def area_of_resulting_shape : ℝ :=
  let area_square := square_side_length^2
  let area_semicircle := (1/2) * Real.pi * semicircle_radius^2
  let total_area := area_square + 4 * area_semicircle
  total_area

theorem area_of_shape_is_correct :
  area_of_resulting_shape = 2 * Real.pi^2 * (Real.pi + 2) :=
sorry

end area_of_shape_is_correct_l629_62978


namespace bookstore_floor_l629_62962

theorem bookstore_floor
  (academy_floor : ℤ)
  (reading_room_floor : ℤ)
  (bookstore_floor : ℤ)
  (h1 : academy_floor = 7)
  (h2 : reading_room_floor = academy_floor + 4)
  (h3 : bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l629_62962


namespace negate_universal_statement_l629_62982

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negate_universal_statement_l629_62982


namespace annulus_area_sufficient_linear_element_l629_62900

theorem annulus_area_sufficient_linear_element (R r : ℝ) (hR : R > 0) (hr : r > 0) (hrR : r < R):
  (∃ d : ℝ, d = R - r ∨ d = R + r) → ∃ A : ℝ, A = π * (R ^ 2 - r ^ 2) :=
by
  sorry

end annulus_area_sufficient_linear_element_l629_62900


namespace decompose_x_l629_62920

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem decompose_x :
  x = (4 : ℝ) • p + q - r :=
sorry

end decompose_x_l629_62920


namespace compare_f_m_plus_2_l629_62905

theorem compare_f_m_plus_2 (a : ℝ) (ha : a > 0) (m : ℝ) 
  (hf : (a * m^2 + 2 * a * m + 1) < 0) : 
  (a * (m + 2)^2 + 2 * a * (m + 2) + 1) > 1 :=
sorry

end compare_f_m_plus_2_l629_62905


namespace bronze_needed_l629_62953

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l629_62953
