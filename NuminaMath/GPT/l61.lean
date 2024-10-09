import Mathlib

namespace back_wheel_revolutions_calculation_l61_6169

def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℕ := 50

noncomputable def back_wheel_revolutions (front_wheel_radius back_wheel_radius gear_ratio : ℝ) (front_wheel_revolutions : ℕ) : ℝ :=
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let distance_traveled := front_circumference * front_wheel_revolutions
  let back_circumference := 2 * Real.pi * back_wheel_radius
  distance_traveled / back_circumference * gear_ratio

theorem back_wheel_revolutions_calculation :
  back_wheel_revolutions front_wheel_radius back_wheel_radius gear_ratio front_wheel_revolutions = 600 :=
sorry

end back_wheel_revolutions_calculation_l61_6169


namespace average_score_for_entire_class_l61_6172

theorem average_score_for_entire_class (n x y : ℕ) (a b : ℝ) (hn : n = 100) (hx : x = 70) (hy : y = 30) (ha : a = 0.65) (hb : b = 0.95) :
    ((x * a + y * b) / n) = 0.74 := by
  sorry

end average_score_for_entire_class_l61_6172


namespace positional_relationship_of_circles_l61_6178

theorem positional_relationship_of_circles 
  (m n : ℝ)
  (h1 : ∃ (x y : ℝ), x^2 - 10 * x + n = 0 ∧ y^2 - 10 * y + n = 0 ∧ x = 2 ∧ y = m) :
  n = 2 * m ∧ m = 8 → 16 > 2 + 8 :=
by
  sorry

end positional_relationship_of_circles_l61_6178


namespace least_xy_value_l61_6149

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l61_6149


namespace triangle_transform_same_l61_6156

def Point := ℝ × ℝ

def reflect_x (p : Point) : Point :=
(p.1, -p.2)

def rotate_180 (p : Point) : Point :=
(-p.1, -p.2)

def reflect_y (p : Point) : Point :=
(-p.1, p.2)

def transform (p : Point) : Point :=
reflect_y (rotate_180 (reflect_x p))

theorem triangle_transform_same (A B C : Point) :
A = (2, 1) → B = (4, 1) → C = (2, 3) →
(transform A = (2, 1) ∧ transform B = (4, 1) ∧ transform C = (2, 3)) :=
by
  intros
  sorry

end triangle_transform_same_l61_6156


namespace find_functions_l61_6159

noncomputable def pair_of_functions_condition (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g (f (x + y)) = f x + (2 * x + y) * g y

theorem find_functions (f g : ℝ → ℝ) :
  pair_of_functions_condition f g →
  (∃ c d : ℝ, ∀ x : ℝ, f x = c * (x + d)) :=
sorry

end find_functions_l61_6159


namespace remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l61_6163

theorem remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12 :
  (7 * 11 ^ 24 + 2 ^ 24) % 12 = 11 := by
sorry

end remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l61_6163


namespace inequality_solution_l61_6105

theorem inequality_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b + c = 1) : (1 / (b * c + a + 1 / a) + 1 / (c * a + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31) :=
by sorry

end inequality_solution_l61_6105


namespace tank_fish_count_l61_6139

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l61_6139


namespace f_2016_plus_f_2015_l61_6189

theorem f_2016_plus_f_2015 (f : ℝ → ℝ) 
  (H1 : ∀ x, f (-x) = -f x) -- Odd function property
  (H2 : ∀ x, f (x + 1) = f (-x + 1)) -- Even function property for f(x+1)
  (H3 : f 1 = 1) : 
  f 2016 + f 2015 = -1 :=
sorry

end f_2016_plus_f_2015_l61_6189


namespace point_not_on_graph_l61_6198

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end point_not_on_graph_l61_6198


namespace probability_one_of_two_sheep_selected_l61_6160

theorem probability_one_of_two_sheep_selected :
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  probability = 3 / 5 :=
by
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  sorry

end probability_one_of_two_sheep_selected_l61_6160


namespace inv_f_zero_l61_6135

noncomputable def f (a b x : Real) : Real := 1 / (2 * a * x + 3 * b)

theorem inv_f_zero (a b : Real) (ha : a ≠ 0) (hb : b ≠ 0) : f a b (1 / (3 * b)) = 0 :=
by 
  sorry

end inv_f_zero_l61_6135


namespace new_price_after_increase_l61_6112

def original_price : ℝ := 220
def percentage_increase : ℝ := 0.15

def new_price (original_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_price + (original_price * percentage_increase)

theorem new_price_after_increase : new_price original_price percentage_increase = 253 := 
by
  sorry

end new_price_after_increase_l61_6112


namespace range_of_a_l61_6124

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) + f (1 - 2 * a) < 0

-- Theorem statement
theorem range_of_a (h_decreasing : decreasing_on f (Set.Ioo (-1) 1))
                   (h_odd : odd_function f)
                   (h_condition : condition f a) :
  0 < a ∧ a < 2 / 3 :=
sorry

end range_of_a_l61_6124


namespace compound_ratio_is_one_fourteenth_l61_6126

theorem compound_ratio_is_one_fourteenth :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) = 1 / 14 :=
by sorry

end compound_ratio_is_one_fourteenth_l61_6126


namespace hcf_of_two_numbers_l61_6136

theorem hcf_of_two_numbers (A B H L : ℕ) (h1 : A * B = 1800) (h2 : L = 200) (h3 : A * B = H * L) : H = 9 :=
by
  sorry

end hcf_of_two_numbers_l61_6136


namespace time_spent_on_road_l61_6193

theorem time_spent_on_road (Total_time_hours Stop1_minutes Stop2_minutes Stop3_minutes : ℕ) 
  (h1: Total_time_hours = 13) 
  (h2: Stop1_minutes = 25) 
  (h3: Stop2_minutes = 10) 
  (h4: Stop3_minutes = 25) : 
  Total_time_hours - (Stop1_minutes + Stop2_minutes + Stop3_minutes) / 60 = 12 :=
by
  sorry

end time_spent_on_road_l61_6193


namespace find_n_l61_6195

theorem find_n (a b : ℤ) (h₁ : a ≡ 25 [ZMOD 42]) (h₂ : b ≡ 63 [ZMOD 42]) :
  ∃ n, 200 ≤ n ∧ n ≤ 241 ∧ (a - b ≡ n [ZMOD 42]) ∧ n = 214 :=
by
  sorry

end find_n_l61_6195


namespace least_number_to_add_l61_6161

theorem least_number_to_add (x : ℕ) : (1056 + x) % 28 = 0 ↔ x = 4 :=
by sorry

end least_number_to_add_l61_6161


namespace cos_960_eq_neg_half_l61_6110

theorem cos_960_eq_neg_half (cos : ℝ → ℝ) (h1 : ∀ x, cos (x + 360) = cos x) 
  (h_even : ∀ x, cos (-x) = cos x) (h_cos120 : cos 120 = - cos 60)
  (h_cos60 : cos 60 = 1 / 2) : cos 960 = -(1 / 2) := by
  sorry

end cos_960_eq_neg_half_l61_6110


namespace total_students_l61_6141

theorem total_students (groups students_per_group : ℕ) (h : groups = 6) (k : students_per_group = 5) :
  groups * students_per_group = 30 := 
by
  sorry

end total_students_l61_6141


namespace students_in_class_l61_6180

theorem students_in_class (n m f r u : ℕ) (cond1 : 20 < n ∧ n < 30)
  (cond2 : f = 2 * m) (cond3 : n = m + f)
  (cond4 : r = 3 * u - 1) (cond5 : r + u = n) :
  n = 27 :=
sorry

end students_in_class_l61_6180


namespace find_m_l61_6166

theorem find_m (m : ℝ) :
  (∃ x a : ℝ, |x - 1| - |x + m| ≥ a ∧ a ≤ 5) ↔ (m = 4 ∨ m = -6) :=
by
  sorry

end find_m_l61_6166


namespace cakes_served_during_lunch_l61_6158

theorem cakes_served_during_lunch (T D L : ℕ) (h1 : T = 15) (h2 : D = 9) : L = T - D → L = 6 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end cakes_served_during_lunch_l61_6158


namespace computation_result_l61_6179

-- Define the vectors and scalar multiplications
def v1 : ℤ × ℤ := (3, -9)
def v2 : ℤ × ℤ := (2, -7)
def v3 : ℤ × ℤ := (-1, 4)

noncomputable def result : ℤ × ℤ := 
  let scalar_mult (m : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (m * v.1, m * v.2)
  scalar_mult 5 v1 - scalar_mult 3 v2 + scalar_mult 2 v3

-- The main theorem
theorem computation_result : result = (7, -16) :=
  by 
    -- Skip the proof as required
    sorry

end computation_result_l61_6179


namespace find_d_value_l61_6197

theorem find_d_value (d : ℝ) :
  (∀ x, (8 * x^3 + 27 * x^2 + d * x + 55 = 0) → (2 * x + 5 = 0)) → d = 39.5 :=
by
  sorry

end find_d_value_l61_6197


namespace latest_time_temperature_84_l61_6102

noncomputable def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem latest_time_temperature_84 :
  ∃ t_max : ℝ, temperature t_max = 84 ∧ ∀ t : ℝ, temperature t = 84 → t ≤ t_max ∧ t_max = 11 :=
by
  sorry

end latest_time_temperature_84_l61_6102


namespace simple_interest_time_l61_6153

-- Definitions based on given conditions
def SI : ℝ := 640           -- Simple interest
def P : ℝ := 4000           -- Principal
def R : ℝ := 8              -- Rate
def T : ℝ := 2              -- Time in years (correct answer to be proved)

-- Theorem statement
theorem simple_interest_time :
  SI = (P * R * T) / 100 := 
by 
  sorry

end simple_interest_time_l61_6153


namespace find_three_digit_number_l61_6119

-- Definitions of digit constraints and the number representation
def is_three_digit_number (N : ℕ) (a b c : ℕ) : Prop :=
  N = 100 * a + 10 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9

-- Definition of the problem condition
def sum_of_digits_condition (N : ℕ) (a b c : ℕ) : Prop :=
  a + b + c = N / 11

-- Lean theorem statement
theorem find_three_digit_number (N a b c : ℕ) :
  is_three_digit_number N a b c ∧ sum_of_digits_condition N a b c → N = 198 :=
by
  sorry

end find_three_digit_number_l61_6119


namespace polyhedron_volume_formula_l61_6164

noncomputable def polyhedron_volume (H S1 S2 S3 : ℝ) : ℝ :=
  (1 / 6) * H * (S1 + S2 + 4 * S3)

theorem polyhedron_volume_formula 
  (H S1 S2 S3 : ℝ)
  (bases_parallel_planes : Prop)
  (lateral_faces_trapezoids_parallelograms_or_triangles : Prop)
  (H_distance : Prop) 
  (S1_area_base : Prop) 
  (S2_area_base : Prop) 
  (S3_area_cross_section : Prop) : 
  polyhedron_volume H S1 S2 S3 = (1 / 6) * H * (S1 + S2 + 4 * S3) :=
sorry

end polyhedron_volume_formula_l61_6164


namespace equation_solutions_equivalence_l61_6106

theorem equation_solutions_equivalence {n k : ℕ} (hn : 1 < n) (hk : 1 < k) (hnk : n > k) :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^(n - k)) :=
by
  sorry

end equation_solutions_equivalence_l61_6106


namespace algebraic_expression_value_l61_6187

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l61_6187


namespace solve_for_x_l61_6184

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end solve_for_x_l61_6184


namespace males_band_not_orchestra_l61_6192

/-- Define conditions as constants -/
def total_females_band := 150
def total_males_band := 130
def total_females_orchestra := 140
def total_males_orchestra := 160
def females_both := 90
def males_both := 80
def total_students_either := 310

/-- The number of males in the band who are NOT in the orchestra -/
theorem males_band_not_orchestra : total_males_band - males_both = 50 := by
  sorry

end males_band_not_orchestra_l61_6192


namespace difference_of_digits_is_three_l61_6107

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem difference_of_digits_is_three :
  ∀ n : ℕ, n = 63 → tens_digit n + ones_digit n = 9 → tens_digit n - ones_digit n = 3 :=
by
  intros n h1 h2
  sorry

end difference_of_digits_is_three_l61_6107


namespace Z_is_1_5_decades_younger_l61_6171

theorem Z_is_1_5_decades_younger (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end Z_is_1_5_decades_younger_l61_6171


namespace tangent_point_x_coordinate_l61_6177

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 2 * x

-- The statement to be proved.
theorem tangent_point_x_coordinate (x : ℝ) (h : derivative x = 4) : x = 2 :=
sorry

end tangent_point_x_coordinate_l61_6177


namespace value_of_f_m_minus_1_pos_l61_6145

variable (a m : ℝ)
variable (f : ℝ → ℝ)
variable (a_pos : a > 0)
variable (fm_neg : f m < 0)
variable (f_def : ∀ x, f x = x^2 - x + a)

theorem value_of_f_m_minus_1_pos : f (m - 1) > 0 :=
by
  sorry

end value_of_f_m_minus_1_pos_l61_6145


namespace arc_length_l61_6129

theorem arc_length (r α : ℝ) (h1 : r = 3) (h2 : α = π / 3) : r * α = π :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is the step where actual simplification and calculation will happen

end arc_length_l61_6129


namespace accurate_to_hundreds_place_l61_6175

def rounded_number : ℝ := 8.80 * 10^4

theorem accurate_to_hundreds_place
  (n : ℝ) (h : n = rounded_number) : 
  exists (d : ℤ), n = d * 100 ∧ |round n - n| < 50 :=
sorry

end accurate_to_hundreds_place_l61_6175


namespace adam_money_ratio_l61_6122

theorem adam_money_ratio 
  (initial_dollars: ℕ) 
  (spent_dollars: ℕ) 
  (remaining_dollars: ℕ := initial_dollars - spent_dollars) 
  (ratio_numerator: ℕ := remaining_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (ratio_denominator: ℕ := spent_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (h_initial: initial_dollars = 91) 
  (h_spent: spent_dollars = 21) 
  (h_gcd: Nat.gcd (initial_dollars - spent_dollars) spent_dollars = 7) :
  ratio_numerator = 10 ∧ ratio_denominator = 3 := by
  sorry

end adam_money_ratio_l61_6122


namespace rectangle_circumference_15pi_l61_6138

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def circumference_of_circle (d : ℝ) : ℝ := 
  Real.pi * d
  
theorem rectangle_circumference_15pi :
  let a := 9
  let b := 12
  let diagonal := rectangle_diagonal a b
  circumference_of_circle diagonal = 15 * Real.pi :=
by 
  sorry

end rectangle_circumference_15pi_l61_6138


namespace cost_of_pencils_and_pens_l61_6134

theorem cost_of_pencils_and_pens (p q : ℝ) 
  (h₁ : 3 * p + 2 * q = 3.60) 
  (h₂ : 2 * p + 3 * q = 3.15) : 
  3 * p + 3 * q = 4.05 :=
sorry

end cost_of_pencils_and_pens_l61_6134


namespace appropriate_investigation_method_l61_6181

theorem appropriate_investigation_method
  (volume_of_investigation_large : Prop)
  (no_need_for_comprehensive_investigation : Prop) :
  (∃ (method : String), method = "sampling investigation") :=
by
  sorry

end appropriate_investigation_method_l61_6181


namespace difference_between_scores_l61_6132

variable (H F : ℕ)
variable (h_hajar_score : H = 24)
variable (h_sum_scores : H + F = 69)
variable (h_farah_higher : F > H)

theorem difference_between_scores : F - H = 21 := by
  sorry

end difference_between_scores_l61_6132


namespace rackets_packed_l61_6104

theorem rackets_packed (total_cartons : ℕ) (cartons_3 : ℕ) (cartons_2 : ℕ) 
  (h1 : total_cartons = 38) 
  (h2 : cartons_3 = 24) 
  (h3 : cartons_2 = total_cartons - cartons_3) :
  3 * cartons_3 + 2 * cartons_2 = 100 := 
by
  -- The proof is omitted
  sorry

end rackets_packed_l61_6104


namespace smallest_three_digit_n_l61_6150

theorem smallest_three_digit_n (n : ℕ) (h_pos : 100 ≤ n) (h_below : n ≤ 999) 
  (cond1 : n % 9 = 2) (cond2 : n % 6 = 4) : n = 118 :=
by {
  sorry
}

end smallest_three_digit_n_l61_6150


namespace additional_money_needed_l61_6176

theorem additional_money_needed :
  let total_budget := 500
  let budget_dresses := 300
  let budget_shoes := 150
  let budget_accessories := 50
  let extra_fraction := 2 / 5
  let discount_rate := 0.15
  let total_without_discount := 
    budget_dresses * (1 + extra_fraction) +
    budget_shoes * (1 + extra_fraction) +
    budget_accessories * (1 + extra_fraction)
  let discounted_total := total_without_discount * (1 - discount_rate)
  discounted_total > total_budget :=
sorry

end additional_money_needed_l61_6176


namespace greatest_integer_x_l61_6125

theorem greatest_integer_x (x : ℤ) : (5 - 4 * x > 17) → x ≤ -4 :=
by
  sorry

end greatest_integer_x_l61_6125


namespace total_beads_in_necklace_l61_6148

noncomputable def amethyst_beads : ℕ := 7
noncomputable def amber_beads : ℕ := 2 * amethyst_beads
noncomputable def turquoise_beads : ℕ := 19
noncomputable def total_beads : ℕ := amethyst_beads + amber_beads + turquoise_beads

theorem total_beads_in_necklace : total_beads = 40 := by
  sorry

end total_beads_in_necklace_l61_6148


namespace average_age_is_35_l61_6131

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l61_6131


namespace jeremy_money_ratio_l61_6191

theorem jeremy_money_ratio :
  let cost_computer := 3000
  let cost_accessories := 0.10 * cost_computer
  let money_left := 2700
  let total_spent := cost_computer + cost_accessories
  let money_before_purchase := total_spent + money_left
  (money_before_purchase / cost_computer) = 2 := by
  sorry

end jeremy_money_ratio_l61_6191


namespace questionnaires_drawn_from_D_l61_6128

theorem questionnaires_drawn_from_D (a1 a2 a3 a4 total sample_b sample_total sample_d : ℕ)
  (h1 : a2 - a1 = a3 - a2)
  (h2 : a3 - a2 = a4 - a3)
  (h3 : a1 + a2 + a3 + a4 = total)
  (h4 : total = 1000)
  (h5 : sample_b = 30)
  (h6 : a2 = 200)
  (h7 : sample_total = 150)
  (h8 : sample_d * total = sample_total * a4) :
  sample_d = 60 :=
by sorry

end questionnaires_drawn_from_D_l61_6128


namespace tetrahedron_perpendicular_distances_inequalities_l61_6173

section Tetrahedron

variables {R : Type*} [LinearOrderedField R]

variables {S_A S_B S_C S_D V d_A d_B d_C d_D h_A h_B h_C h_D : R}

/-- Given areas and perpendicular distances of a tetrahedron, prove inequalities involving these parameters. -/
theorem tetrahedron_perpendicular_distances_inequalities 
  (h1 : S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D = 3 * V) : 
  (min h_A (min h_B (min h_C h_D)) ≤ d_A + d_B + d_C + d_D) ∧ 
  (d_A + d_B + d_C + d_D ≤ max h_A (max h_B (max h_C h_D))) ∧ 
  (d_A * d_B * d_C * d_D ≤ 81 * V ^ 4 / (256 * S_A * S_B * S_C * S_D)) :=
sorry

end Tetrahedron

end tetrahedron_perpendicular_distances_inequalities_l61_6173


namespace second_pipe_filling_time_l61_6167

theorem second_pipe_filling_time (T : ℝ) :
  (∃ T : ℝ, (1 / 8 + 1 / T = 1 / 4.8) ∧ T = 12) :=
by
  sorry

end second_pipe_filling_time_l61_6167


namespace find_particular_number_l61_6154

theorem find_particular_number (x : ℝ) (h : 4 * x * 25 = 812) : x = 8.12 :=
by sorry

end find_particular_number_l61_6154


namespace least_positive_integer_added_to_575_multiple_4_l61_6170

theorem least_positive_integer_added_to_575_multiple_4 :
  ∃ n : ℕ, n > 0 ∧ (575 + n) % 4 = 0 ∧ 
           ∀ m : ℕ, (m > 0 ∧ (575 + m) % 4 = 0) → n ≤ m := by
  sorry

end least_positive_integer_added_to_575_multiple_4_l61_6170


namespace pqr_value_l61_6130

theorem pqr_value (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h1 : p + q + r = 30) 
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + (420 : ℚ) / (p * q * r) = 1) : 
  p * q * r = 1800 := 
sorry

end pqr_value_l61_6130


namespace find_w_l61_6116

variables {x y z w : ℝ}

theorem find_w (h : (1 / x) + (1 / y) + (1 / z) = 1 / w) :
  w = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end find_w_l61_6116


namespace part1_part2_l61_6123

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (b : ℝ) := 0.5 * x^2 - b * x
noncomputable def h (x : ℝ) (b : ℝ) := f x + g x b

theorem part1 (b : ℝ) :
  (∃ (tangent_point : ℝ),
    tangent_point = 1 ∧
    deriv f tangent_point = 1 ∧
    f tangent_point = 0 ∧
    ∃ (y_tangent : ℝ → ℝ), (∀ (x : ℝ), y_tangent x = x - 1) ∧
    ∃ (tangent_for_g : ℝ), (∀ (x : ℝ), y_tangent x = g x b)
  ) → false :=
sorry 

theorem part2 (b : ℝ) :
  ¬ (∀ (x : ℝ) (hx : 0 < x), deriv (h x) b = 0 → deriv (h x) b < 0) →
  2 < b :=
sorry

end part1_part2_l61_6123


namespace students_count_l61_6182

theorem students_count :
  ∃ S : ℕ, (S + 4) % 9 = 0 ∧ S = 23 :=
by
  sorry

end students_count_l61_6182


namespace percentage_x_equals_twenty_percent_of_487_50_is_65_l61_6118

theorem percentage_x_equals_twenty_percent_of_487_50_is_65
    (x : ℝ)
    (hx : x = 150)
    (y : ℝ)
    (hy : y = 487.50) :
    (∃ (P : ℝ), P * x = 0.20 * y ∧ P * 100 = 65) :=
by
  sorry

end percentage_x_equals_twenty_percent_of_487_50_is_65_l61_6118


namespace number_of_cows_is_six_l61_6114

variable (C H : Nat) -- C for cows and H for chickens

-- Number of legs is 12 more than twice the number of heads.
def cows_count_condition : Prop :=
  4 * C + 2 * H = 2 * (C + H) + 12

theorem number_of_cows_is_six (h : cows_count_condition C H) : C = 6 :=
sorry

end number_of_cows_is_six_l61_6114


namespace total_food_for_guinea_pigs_l61_6174

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l61_6174


namespace Mina_age_is_10_l61_6108

-- Define the conditions as Lean definitions
variable (S : ℕ)

def Minho_age := 3 * S
def Mina_age := 2 * S - 2

-- State the main problem as a theorem
theorem Mina_age_is_10 (h_sum : S + Minho_age S + Mina_age S = 34) : Mina_age S = 10 :=
by
  sorry

end Mina_age_is_10_l61_6108


namespace simplify_complex_fraction_l61_6109

open Complex

theorem simplify_complex_fraction :
  (⟨2, 2⟩ : ℂ) / (⟨-3, 4⟩ : ℂ) = (⟨-14 / 25, -14 / 25⟩ : ℂ) :=
by
  sorry

end simplify_complex_fraction_l61_6109


namespace selling_price_eq_l61_6165

theorem selling_price_eq (cp sp L : ℕ) (h_cp: cp = 47) (h_L : L = cp - 40) (h_profit_loss_eq : sp - cp = L) :
  sp = 54 :=
by
  sorry

end selling_price_eq_l61_6165


namespace cos_largest_angle_value_l61_6144

noncomputable def cos_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

theorem cos_largest_angle_value : cos_largest_angle 2 3 4 (by rfl) (by rfl) (by rfl) = -1 / 4 := 
sorry

end cos_largest_angle_value_l61_6144


namespace value_of_z_sub_y_add_x_l61_6186

-- Represent 312 in base 3
def base3_representation : List ℕ := [1, 0, 1, 2, 1, 0] -- 312 in base 3 is 101210

-- Define x, y, z
def x : ℕ := (base3_representation.count 0)
def y : ℕ := (base3_representation.count 1)
def z : ℕ := (base3_representation.count 2)

-- Proposition to be proved
theorem value_of_z_sub_y_add_x : z - y + x = 2 := by
  sorry

end value_of_z_sub_y_add_x_l61_6186


namespace range_of_a_l61_6146

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x
noncomputable def k (x : ℝ) : ℝ := (Real.log x + x) / x^2

theorem range_of_a (a : ℝ) (h_zero : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l61_6146


namespace calculate_total_payment_l61_6127

theorem calculate_total_payment
(adult_price : ℕ := 30)
(teen_price : ℕ := 20)
(child_price : ℕ := 15)
(num_adults : ℕ := 4)
(num_teenagers : ℕ := 4)
(num_children : ℕ := 2)
(num_activities : ℕ := 5)
(has_coupon : Bool := true)
(soda_price : ℕ := 5)
(num_sodas : ℕ := 5)

(total_admission_before_discount : ℕ := 
  num_adults * adult_price + num_teenagers * teen_price + num_children * child_price)
(discount_on_activities : ℕ := if num_activities >= 7 then 15 else if num_activities >= 5 then 10 else if num_activities >= 3 then 5 else 0)
(admission_after_activity_discount : ℕ := 
  total_admission_before_discount - total_admission_before_discount * discount_on_activities / 100)
(additional_discount : ℕ := if has_coupon then 5 else 0)
(admission_after_all_discounts : ℕ := 
  admission_after_activity_discount - admission_after_activity_discount * additional_discount / 100)

(total_cost : ℕ := admission_after_all_discounts + num_sodas * soda_price) :
total_cost = 22165 := 
sorry

end calculate_total_payment_l61_6127


namespace mean_home_runs_per_game_l61_6117

variable (home_runs : Nat) (games_played : Nat)

def total_home_runs : Nat := 
  (5 * 4) + (6 * 5) + (4 * 7) + (3 * 9) + (2 * 11)

def total_games_played : Nat :=
  (5 * 5) + (6 * 6) + (4 * 8) + (3 * 10) + (2 * 12)

theorem mean_home_runs_per_game :
  (total_home_runs : ℚ) / total_games_played = 127 / 147 :=
  by 
    sorry

end mean_home_runs_per_game_l61_6117


namespace find_speed_of_first_train_l61_6162

noncomputable def relative_speed (length1 length2 : ℕ) (time_seconds : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hours := time_seconds / 3600
  total_length_km / time_hours

theorem find_speed_of_first_train
  (length1 : ℕ)   -- Length of the first train in meters
  (length2 : ℕ)   -- Length of the second train in meters
  (speed2 : ℝ)    -- Speed of the second train in km/h
  (time_seconds : ℝ)  -- Time in seconds to be clear from each other
  (correct_speed1 : ℝ)  -- Correct speed of the first train in km/h
  (h_length1 : length1 = 160)
  (h_length2 : length2 = 280)
  (h_speed2 : speed2 = 30)
  (h_time_seconds : time_seconds = 21.998240140788738)
  (h_correct_speed1 : correct_speed1 = 41.98) :
  relative_speed length1 length2 time_seconds = speed2 + correct_speed1 :=
by
  sorry

end find_speed_of_first_train_l61_6162


namespace no_solution_prob1_l61_6196

theorem no_solution_prob1 : ¬ ∃ x : ℝ, x ≠ 2 ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
by
  sorry

end no_solution_prob1_l61_6196


namespace product_of_integers_l61_6115

theorem product_of_integers (a b : ℚ) (h1 : a / b = 12) (h2 : a + b = 144) :
  a * b = 248832 / 169 := 
sorry

end product_of_integers_l61_6115


namespace P_is_sufficient_but_not_necessary_for_Q_l61_6190

def P (x : ℝ) : Prop := (2 * x - 3)^2 < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_is_sufficient_but_not_necessary_for_Q : 
  (∀ x, P x → Q x) ∧ (∃ x, Q x ∧ ¬ P x) :=
by
  sorry

end P_is_sufficient_but_not_necessary_for_Q_l61_6190


namespace find_angle_A_l61_6100

theorem find_angle_A (a b c A B C : ℝ)
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) :
  A = Real.pi / 6 :=
sorry

end find_angle_A_l61_6100


namespace probability_of_neither_tamil_nor_english_l61_6113

-- Definitions based on the conditions
def TotalPopulation := 1500
def SpeakTamil := 800
def SpeakEnglish := 650
def SpeakTamilAndEnglish := 250

-- Use Inclusion-Exclusion Principle
def SpeakTamilOrEnglish : ℕ := SpeakTamil + SpeakEnglish - SpeakTamilAndEnglish

-- Number of people who speak neither Tamil nor English
def SpeakNeitherTamilNorEnglish : ℕ := TotalPopulation - SpeakTamilOrEnglish

-- The probability calculation
def Probability := (SpeakNeitherTamilNorEnglish : ℚ) / (TotalPopulation : ℚ)

-- Theorem to prove
theorem probability_of_neither_tamil_nor_english : Probability = (1/5 : ℚ) :=
sorry

end probability_of_neither_tamil_nor_english_l61_6113


namespace oliver_dishes_count_l61_6151

def total_dishes : ℕ := 42
def mango_salsa_dishes : ℕ := 5
def fresh_mango_dishes : ℕ := total_dishes / 6
def mango_jelly_dishes : ℕ := 2
def strawberry_dishes : ℕ := 3
def pineapple_dishes : ℕ := 5
def kiwi_dishes : ℕ := 4
def mango_dishes_oliver_picks_out : ℕ := 3

def total_mango_dishes : ℕ := mango_salsa_dishes + fresh_mango_dishes + mango_jelly_dishes
def mango_dishes_oliver_wont_eat : ℕ := total_mango_dishes - mango_dishes_oliver_picks_out
def max_strawberry_pineapple_dishes : ℕ := strawberry_dishes

def dishes_left_for_oliver : ℕ := total_dishes - mango_dishes_oliver_wont_eat - max_strawberry_pineapple_dishes

theorem oliver_dishes_count : dishes_left_for_oliver = 28 := 
by 
  sorry

end oliver_dishes_count_l61_6151


namespace hyperbola_with_foci_condition_l61_6185

theorem hyperbola_with_foci_condition (k : ℝ) :
  ( ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 → ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 ∧ (k + 3 > 0 ∧ k + 2 < 0) ) ↔ (-3 < k ∧ k < -2) :=
sorry

end hyperbola_with_foci_condition_l61_6185


namespace problem_statement_l61_6133

theorem problem_statement (x y : ℝ) (M N P : ℝ) 
  (hM_def : M = 2 * x + y)
  (hN_def : N = 2 * x - y)
  (hP_def : P = x * y)
  (hM : M = 4)
  (hN : N = 2) : P = 1.5 :=
by
  sorry

end problem_statement_l61_6133


namespace smallest_positive_integer_l61_6183

theorem smallest_positive_integer (n : ℕ) (h₁ : n > 1) (h₂ : n % 2 = 1) (h₃ : n % 3 = 1) (h₄ : n % 4 = 1) (h₅ : n % 5 = 1) : n = 61 :=
by
  sorry

end smallest_positive_integer_l61_6183


namespace solution_l61_6120

theorem solution (x : ℝ) : (x = -2/5) → (x < x^3 ∧ x^3 < x^2) :=
by
  intro h
  rw [h]
  -- sorry to skip the proof
  sorry

end solution_l61_6120


namespace youtube_likes_l61_6152

theorem youtube_likes (L D : ℕ) 
  (h1 : D = (1 / 2 : ℝ) * L + 100)
  (h2 : D + 1000 = 2600) : 
  L = 3000 := 
by
  sorry

end youtube_likes_l61_6152


namespace probability_math_majors_consecutive_l61_6101

theorem probability_math_majors_consecutive :
  (5 / 12) * (4 / 11) * (3 / 10) * (2 / 9) * (1 / 8) * 12 = 1 / 66 :=
by
  sorry

end probability_math_majors_consecutive_l61_6101


namespace simplify_fraction_l61_6168

theorem simplify_fraction (a b : ℝ) :
  ( (3 * b) / (2 * a^2) )^3 = 27 * b^3 / (8 * a^6) :=
by
  sorry

end simplify_fraction_l61_6168


namespace problem1_range_of_x_problem2_value_of_a_l61_6155

open Set

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |x + 3| + |x - a|

-- Problem 1
theorem problem1_range_of_x (a : ℝ) (h : a = 4) (h_eq : ∀ x : ℝ, f x a = 7 ↔ x ∈ Icc (-3 : ℝ) 4) :
  ∀ x : ℝ, f x 4 = 7 ↔ x ∈ Icc (-3 : ℝ) 4 := by
  sorry

-- Problem 2
theorem problem2_value_of_a (h₁ : ∀ x : ℝ, x ∈ {x : ℝ | f x 4 ≥ 6} ↔ x ≤ -4 ∨ x ≥ 2) :
  f x a ≥ 6 ↔  x ≤ -4 ∨ x ≥ 2 :=
  by
  sorry

end problem1_range_of_x_problem2_value_of_a_l61_6155


namespace problem_statement_l61_6188

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x : ℝ, f x = x^2 + x + 1) 
  (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
sorry

end problem_statement_l61_6188


namespace find_vertex_D_l61_6121

noncomputable def quadrilateral_vertices : Prop :=
  let A : (ℤ × ℤ) := (-1, -2)
  let B : (ℤ × ℤ) := (3, 1)
  let C : (ℤ × ℤ) := (0, 2)
  A ≠ B ∧ A ≠ C ∧ B ≠ C

theorem find_vertex_D (A B C D : ℤ × ℤ) (h_quad : quadrilateral_vertices) :
    (A = (-1, -2)) →
    (B = (3, 1)) →
    (C = (0, 2)) →
    (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) →
    D = (-4, -1) :=
by
  sorry

end find_vertex_D_l61_6121


namespace find_value_of_a_l61_6143

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a_l61_6143


namespace alex_score_l61_6194

theorem alex_score 
    (n : ℕ) -- number of students
    (avg_19 : ℕ) -- average score of first 19 students
    (avg_20 : ℕ) -- average score of all 20 students
    (h_n : n = 20) -- number of students is 20
    (h_avg_19 : avg_19 = 75) -- average score of first 19 students is 75
    (h_avg_20 : avg_20 = 76) -- average score of all 20 students is 76
  : ∃ alex_score : ℕ, alex_score = 95 := 
by
    sorry

end alex_score_l61_6194


namespace min_tip_percentage_l61_6142

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage_l61_6142


namespace quartic_polynomial_eval_l61_6157

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a monic quartic polynomial

theorem quartic_polynomial_eval (h_monic: true)
    (h1 : f (-1) = -1)
    (h2 : f 2 = -4)
    (h3 : f (-3) = -9)
    (h4 : f 4 = -16) : f 1 = 23 :=
sorry

end quartic_polynomial_eval_l61_6157


namespace smallest_positive_four_digit_equivalent_to_5_mod_8_l61_6199

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end smallest_positive_four_digit_equivalent_to_5_mod_8_l61_6199


namespace ball_hits_ground_l61_6140

theorem ball_hits_ground (t : ℝ) :
  (∃ t, -(16 * t^2) + 32 * t + 30 = 0 ∧ t = 1 + (Real.sqrt 46) / 4) :=
sorry

end ball_hits_ground_l61_6140


namespace problem_statement_l61_6147

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h1 : ∀ x > 0, f x = Real.log x / Real.log 3)
variable (h2 : b = 9 * a)

theorem problem_statement : f a - f b = -2 := by
  sorry

end problem_statement_l61_6147


namespace leah_coins_value_l61_6137

theorem leah_coins_value : 
  ∃ (p n : ℕ), p + n = 15 ∧ n + 1 = p ∧ 5 * n + 1 * p = 43 := 
by
  sorry

end leah_coins_value_l61_6137


namespace rectangle_k_value_l61_6103

theorem rectangle_k_value (a k : ℝ) (h1 : k > 0) (h2 : 2 * (3 * a + a) = k) (h3 : 3 * a^2 = k) : k = 64 / 3 :=
by
  sorry

end rectangle_k_value_l61_6103


namespace sin_eq_sqrt3_div_2_l61_6111

open Real

theorem sin_eq_sqrt3_div_2 (theta : ℝ) :
  sin theta = (sqrt 3) / 2 ↔ (∃ k : ℤ, theta = π/3 + 2*k*π ∨ theta = 2*π/3 + 2*k*π) :=
by
  sorry

end sin_eq_sqrt3_div_2_l61_6111
