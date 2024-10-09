import Mathlib

namespace find_tangent_line_equation_l2313_231359

-- Define the curve as a function
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, 3)

-- Define the slope of the tangent line at point P
def slope_at_P : ℝ := curve_derivative P.1

-- Define the expected equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

-- The theorem to prove that the tangent line at point P has the expected equation
theorem find_tangent_line_equation : 
  tangent_line P.1 (curve P.1) :=
  sorry

end find_tangent_line_equation_l2313_231359


namespace choir_members_number_l2313_231367

theorem choir_members_number
  (n : ℕ)
  (h1 : n % 12 = 10)
  (h2 : n % 14 = 12)
  (h3 : 300 ≤ n ∧ n ≤ 400) :
  n = 346 :=
sorry

end choir_members_number_l2313_231367


namespace exists_integer_div_15_sqrt_range_l2313_231300

theorem exists_integer_div_15_sqrt_range :
  ∃ n : ℕ, (25^2 ≤ n ∧ n ≤ 26^2) ∧ (n % 15 = 0) :=
by
  sorry

end exists_integer_div_15_sqrt_range_l2313_231300


namespace polar_coordinates_of_2_neg2_l2313_231390

noncomputable def rect_to_polar_coord (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 
                then 2 * Real.pi - Real.arctan (x / (-y)) 
                else Real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_2_neg2 :
  rect_to_polar_coord 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
by 
  sorry

end polar_coordinates_of_2_neg2_l2313_231390


namespace triangle_base_length_l2313_231366

theorem triangle_base_length (h : 3 = (b * 3) / 2) : b = 2 :=
by
  sorry

end triangle_base_length_l2313_231366


namespace sin_2x_value_l2313_231363

theorem sin_2x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : Real.sin (2 * x) = 7 / 9 := by
  sorry

end sin_2x_value_l2313_231363


namespace equation_no_solution_for_k_7_l2313_231385

theorem equation_no_solution_for_k_7 :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → ¬ (x ^ 2 - 1) / (x - 3) = (x ^ 2 - 7) / (x - 5) :=
by
  intro x h
  have h1 : x ≠ 3 := h.1
  have h2 : x ≠ 5 := h.2
  sorry

end equation_no_solution_for_k_7_l2313_231385


namespace middle_odd_number_is_26_l2313_231307

theorem middle_odd_number_is_26 (x : ℤ) 
  (h : (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 130) : x = 26 := 
by 
  sorry

end middle_odd_number_is_26_l2313_231307


namespace husband_weekly_saving_l2313_231389

variable (H : ℕ)

-- conditions
def weekly_wife : ℕ := 225
def months : ℕ := 6
def weeks_per_month : ℕ := 4
def weeks := months * weeks_per_month
def amount_per_child : ℕ := 1680
def num_children : ℕ := 4

-- total savings calculation
def total_saving : ℕ := weeks * H + weeks * weekly_wife

-- half of total savings divided among children
def half_savings_div_by_children : ℕ := num_children * amount_per_child

-- proof statement
theorem husband_weekly_saving : H = 335 :=
by
  let total_children_saving := half_savings_div_by_children
  have half_saving : ℕ := total_children_saving 
  have total_saving_eq : total_saving = 2 * total_children_saving := sorry
  have total_saving_eq_simplified : weeks * H + weeks * weekly_wife = 13440 := sorry
  have H_eq : H = 335 := sorry
  exact H_eq

end husband_weekly_saving_l2313_231389


namespace solve_for_x_l2313_231346

theorem solve_for_x (x : ℝ) (h : 2 - 1 / (1 - x) = 1 / (1 - x)) : x = 0 :=
sorry

end solve_for_x_l2313_231346


namespace floor_sum_min_value_l2313_231336

theorem floor_sum_min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋) = 7 :=
sorry

end floor_sum_min_value_l2313_231336


namespace f_f_f_f_f_of_1_l2313_231338

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_f_f_f_f_of_1 : f (f (f (f (f 1)))) = 4687 :=
by
  sorry

end f_f_f_f_f_of_1_l2313_231338


namespace daisies_bought_l2313_231372

-- Definitions from the given conditions
def cost_per_flower : ℕ := 6
def num_roses : ℕ := 7
def total_spent : ℕ := 60

-- Proving the number of daisies Maria bought
theorem daisies_bought : ∃ (D : ℕ), D = 3 ∧ total_spent = num_roses * cost_per_flower + D * cost_per_flower :=
by
  sorry

end daisies_bought_l2313_231372


namespace arithmetic_seq_a8_l2313_231396

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h1 : a 5 = 10)
  (h2 : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 := sorry

end arithmetic_seq_a8_l2313_231396


namespace find_t_l2313_231384

theorem find_t (s t : ℝ) (h1 : 15 * s + 7 * t = 236) (h2 : t = 2 * s + 1) : t = 16.793 :=
by
  sorry

end find_t_l2313_231384


namespace find_c_plus_d_l2313_231394

theorem find_c_plus_d (a b c d : ℝ) (h1 : a + b = 12) (h2 : b + c = 9) (h3 : a + d = 6) : 
  c + d = 3 := 
sorry

end find_c_plus_d_l2313_231394


namespace find_m_l2313_231355

-- Define points O, A, B, C
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (1, 5)
def C (m : ℝ) : (ℝ × ℝ) := (m, 3)

-- Define vectors AB and OC
def vector_AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)  -- (B - A)
def vector_OC (m : ℝ) : (ℝ × ℝ) := (m, 3)  -- (C - O)

-- Define the dot product
def dot_product (v₁ v₂ : (ℝ × ℝ)) : ℝ := (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem: vector_AB ⊥ vector_OC implies m = 6
theorem find_m (m : ℝ) (h : dot_product vector_AB (vector_OC m) = 0) : m = 6 :=
by
  -- Proof part not required
  sorry

end find_m_l2313_231355


namespace solve_sqrt_eq_l2313_231318

theorem solve_sqrt_eq (z : ℚ) (h : Real.sqrt (5 - 4 * z) = 10) : z = -95 / 4 := by
  sorry

end solve_sqrt_eq_l2313_231318


namespace quadratic_roots_sum_product_l2313_231303

theorem quadratic_roots_sum_product (m n : ℝ) (h1 : m / 2 = 10) (h2 : n / 2 = 24) : m + n = 68 :=
by
  sorry

end quadratic_roots_sum_product_l2313_231303


namespace sum_of_coeffs_eq_92_l2313_231391

noncomputable def sum_of_integer_coeffs_in_factorization (x y : ℝ) : ℝ :=
  let f := 27 * (x ^ 6) - 512 * (y ^ 6)
  3 - 8 + 9 + 24 + 64  -- Sum of integer coefficients

theorem sum_of_coeffs_eq_92 (x y : ℝ) : sum_of_integer_coeffs_in_factorization x y = 92 :=
by
  -- proof steps go here
  sorry

end sum_of_coeffs_eq_92_l2313_231391


namespace sum_of_ages_l2313_231309

theorem sum_of_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
sorry

end sum_of_ages_l2313_231309


namespace length_of_BD_l2313_231343

noncomputable def points_on_circle (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ) : Prop :=
  BC = 4 ∧ CD = 4 ∧ AE = 6 ∧ (0 < y) ∧ (0 < z) ∧ (AE * 2 = y * z) ∧ (8 > y + z)

theorem length_of_BD (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ)
  (h : points_on_circle A B C D E BD AE BC CD y z) : 
  BD = 7 :=
by
  sorry

end length_of_BD_l2313_231343


namespace d_in_N_l2313_231379

def M := {x : ℤ | ∃ n : ℤ, x = 3 * n}
def N := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def P := {x : ℤ | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c d : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) (hd : d = a - b + c) : d ∈ N :=
by sorry

end d_in_N_l2313_231379


namespace at_least_one_ge_two_l2313_231302

theorem at_least_one_ge_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a + b + c ≥ 6 → (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) :=
by
  intros
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  sorry

end at_least_one_ge_two_l2313_231302


namespace minimize_fraction_l2313_231308

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) → (∀ m : ℕ, 0 < m → (n = m) → (3 * m + 27 / m ≥ 6)) := sorry

end minimize_fraction_l2313_231308


namespace number_properties_l2313_231375

theorem number_properties (a b x : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 375) 
  (h3 : a - b = x) : 
  (a = 25 ∧ b = 15 ∧ x = 10) ∨ (a = 15 ∧ b = 25 ∧ x = 10) :=
by
  sorry

end number_properties_l2313_231375


namespace determine_values_of_a_and_b_l2313_231381

def ab_product_eq_one (a b : ℝ) : Prop := a * b = 1

def given_equation (a b : ℝ) : Prop :=
  (a + b + 2) / 4 = (1 / (a + 1)) + (1 / (b + 1))

theorem determine_values_of_a_and_b (a b : ℝ) (h1 : ab_product_eq_one a b) (h2 : given_equation a b) :
  a = 1 ∧ b = 1 :=
by
  sorry

end determine_values_of_a_and_b_l2313_231381


namespace difference_between_balls_l2313_231331

theorem difference_between_balls (B R : ℕ) (h1 : R - 152 = B + 152 + 346) : R - B = 650 := 
sorry

end difference_between_balls_l2313_231331


namespace triangle_side_lengths_exist_l2313_231382

theorem triangle_side_lengths_exist 
  (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  ∃ (x y z : ℝ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
  (a = y + z) ∧ (b = x + z) ∧ (c = x + y) :=
by
  let x := (a - b + c) / 2
  let y := (a + b - c) / 2
  let z := (-a + b + c) / 2
  have hx : x > 0 := sorry
  have hy : y > 0 := sorry
  have hz : z > 0 := sorry
  have ha : a = y + z := sorry
  have hb : b = x + z := sorry
  have hc : c = x + y := sorry
  exact ⟨x, y, z, hx, hy, hz, ha, hb, hc⟩

end triangle_side_lengths_exist_l2313_231382


namespace expression_that_gives_value_8_l2313_231322

theorem expression_that_gives_value_8 (a b : ℝ) 
  (h_eq1 : a = 2) 
  (h_eq2 : b = 2) 
  (h_roots : ∀ x, (x - a) * (x - b) = x^2 - 4 * x + 4) : 
  2 * (a + b) = 8 :=
by
  sorry

end expression_that_gives_value_8_l2313_231322


namespace interval_after_speed_limit_l2313_231386

noncomputable def car_speed_before : ℝ := 80 -- speed before the sign in km/h
noncomputable def car_speed_after : ℝ := 60 -- speed after the sign in km/h
noncomputable def initial_interval : ℕ := 10 -- interval between the cars in meters

-- Convert speeds from km/h to m/s
noncomputable def v : ℝ := car_speed_before * 1000 / 3600
noncomputable def u : ℝ := car_speed_after * 1000 / 3600

-- Given the initial interval and speed before the sign, calculate the time it takes for the second car to reach the sign
noncomputable def delta_t : ℝ := initial_interval / v

-- Given u and delta_t, calculate the new interval after slowing down
noncomputable def new_interval : ℝ := u * delta_t

-- Theorem statement in Lean
theorem interval_after_speed_limit : new_interval = 7.5 :=
sorry

end interval_after_speed_limit_l2313_231386


namespace triangle_angle_y_l2313_231365

theorem triangle_angle_y (y : ℝ) (h1 : 2 * y + (y + 10) + 4 * y = 180) : 
  y = 170 / 7 := 
by
  sorry

end triangle_angle_y_l2313_231365


namespace problem_k_value_l2313_231353

theorem problem_k_value (a b c : ℕ) (h1 : a + b / c = 101) (h2 : a / c + b = 68) :
  (a + b) / c = 13 :=
sorry

end problem_k_value_l2313_231353


namespace min_cubes_l2313_231321

-- Define the conditions
structure Cube := (x : ℕ) (y : ℕ) (z : ℕ)
def shares_face (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z = c2.z - 1)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

def front_view (cubes : List Cube) : Prop :=
  -- Representation of L-shape in xy-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 1 ∧ c2.y = 0 ∧ c2.z = 0) ∧
  (c3.x = 2 ∧ c3.y = 0 ∧ c3.z = 0) ∧
  (c4.x = 2 ∧ c4.y = 1 ∧ c4.z = 0) ∧
  (c5.x = 1 ∧ c5.y = 2 ∧ c5.z = 0)

def side_view (cubes : List Cube) : Prop :=
  -- Representation of Z-shape in yz-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 0 ∧ c2.y = 1 ∧ c2.z = 0) ∧
  (c3.x = 0 ∧ c3.y = 1 ∧ c3.z = 1) ∧
  (c4.x = 0 ∧ c4.y = 2 ∧ c4.z = 1) ∧
  (c5.x = 0 ∧ c5.y = 2 ∧ c5.z = 2)

-- Proof statement
theorem min_cubes (cubes : List Cube) (h1 : front_view cubes) (h2 : side_view cubes) : cubes.length = 5 :=
by sorry

end min_cubes_l2313_231321


namespace players_taking_physics_l2313_231378

-- Definitions based on the conditions
def total_players : ℕ := 30
def players_taking_math : ℕ := 15
def players_taking_both : ℕ := 6

-- The main theorem to prove
theorem players_taking_physics : total_players - players_taking_math + players_taking_both = 21 := by
  sorry

end players_taking_physics_l2313_231378


namespace grandma_Olga_grandchildren_l2313_231377

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l2313_231377


namespace jordan_rect_width_is_10_l2313_231369

def carol_rect_length : ℕ := 5
def carol_rect_width : ℕ := 24
def jordan_rect_length : ℕ := 12

def carol_rect_area : ℕ := carol_rect_length * carol_rect_width
def jordan_rect_width := carol_rect_area / jordan_rect_length

theorem jordan_rect_width_is_10 : jordan_rect_width = 10 :=
by
  sorry

end jordan_rect_width_is_10_l2313_231369


namespace lines_perpendicular_l2313_231388

theorem lines_perpendicular 
  (a b : ℝ) (θ : ℝ)
  (L1 : ∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ + a = 0)
  (L2 : ∀ x y : ℝ, x * Real.sin θ - y * Real.cos θ + b = 0)
  : ∀ m1 m2 : ℝ, m1 = -(Real.cos θ) / (Real.sin θ) → m2 = (Real.sin θ) / (Real.cos θ) → m1 * m2 = -1 :=
by 
  intros m1 m2 h1 h2
  sorry

end lines_perpendicular_l2313_231388


namespace simplify_expression1_simplify_expression2_l2313_231344

variable {x y : ℝ} -- Declare x and y as real numbers

theorem simplify_expression1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
sorry

theorem simplify_expression2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - (3/2) * x^2 * y) + x^2 * y^2) = - x^2 * y^2 :=
sorry

end simplify_expression1_simplify_expression2_l2313_231344


namespace parabola_opens_downwards_l2313_231335

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l2313_231335


namespace sequence_is_geometric_l2313_231326

theorem sequence_is_geometric {a : ℝ} (h : a ≠ 0) (S : ℕ → ℝ) (H : ∀ n, S n = a^n - 1) 
: ∃ r, ∀ n, (n ≥ 1) → S n - S (n-1) = r * (S (n-1) - S (n-2)) :=
sorry

end sequence_is_geometric_l2313_231326


namespace alcohol_mixture_l2313_231356

theorem alcohol_mixture (y : ℕ) :
  let x_vol := 200 -- milliliters
  let y_conc := 30 / 100 -- 30% alcohol
  let x_conc := 10 / 100 -- 10% alcohol
  let final_conc := 20 / 100 -- 20% target alcohol concentration
  let x_alcohol := x_vol * x_conc -- alcohol in x
  (x_alcohol + y * y_conc) / (x_vol + y) = final_conc ↔ y = 200 :=
by 
  sorry

end alcohol_mixture_l2313_231356


namespace louis_age_currently_31_l2313_231341

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l2313_231341


namespace f_monotone_f_inequality_solution_l2313_231399

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f y = x
axiom f_at_2: f 2 = 1
axiom f_mul : ∀ x y, f (x * y) = f x + f y
axiom f_positive : ∀ x, x > 1 → f x > 0

theorem f_monotone (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem f_inequality_solution (x : ℝ) (hx : x > 2 ∧ x ≤ 4) : f x + f (x - 2) ≤ 3 :=
sorry

end f_monotone_f_inequality_solution_l2313_231399


namespace prove_smallest_number_l2313_231371

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

lemma smallest_number_to_add (n : ℕ) (k : ℕ) (h: sum_of_digits n % k = r) : n % k = r →
  n % k = r → (k - r) = 7 :=
by
  sorry

theorem prove_smallest_number (n : ℕ) (k : ℕ) (r : ℕ) :
  (27452 % 9 = r) ∧ (9 - r = 7) :=
by
  sorry

end prove_smallest_number_l2313_231371


namespace crushing_load_value_l2313_231305

-- Given definitions
def W : ℕ := 3
def T : ℕ := 2
def H : ℕ := 6
def L : ℕ := (30 * W^3 * T^5) / H^3

-- Theorem statement
theorem crushing_load_value :
  L = 120 :=
by {
  -- We provided definitions using the given conditions.
  -- Placeholder for proof is provided
  sorry
}

end crushing_load_value_l2313_231305


namespace isosceles_triangle_area_l2313_231323

theorem isosceles_triangle_area
  (a b : ℝ) -- sides of the triangle
  (inradius : ℝ) (perimeter : ℝ)
  (angle : ℝ) -- angle in degrees
  (h_perimeter : 2 * a + b = perimeter)
  (h_inradius : inradius = 2.5)
  (h_angle : angle = 40)
  (h_perimeter_value : perimeter = 20)
  (h_semiperimeter : (perimeter / 2) = 10) :
  (inradius * (perimeter / 2) = 25) :=
by
  sorry

end isosceles_triangle_area_l2313_231323


namespace find_x_range_l2313_231311

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem find_x_range (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end find_x_range_l2313_231311


namespace distance_interval_l2313_231352

def distance_to_town (d : ℝ) : Prop :=
  ¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 9)

theorem distance_interval (d : ℝ) : distance_to_town d → d ∈ Set.Ioo 7 8 :=
by
  intro h
  have h1 : d < 8 := by sorry
  have h2 : d > 7 := by sorry
  rw [Set.mem_Ioo]
  exact ⟨h2, h1⟩

end distance_interval_l2313_231352


namespace chord_length_circle_l2313_231316

theorem chord_length_circle {x y : ℝ} :
  (x - 1)^2 + (y - 1)^2 = 2 →
  (exists (p q : ℝ), (p-1)^2 = 1 ∧ (q-1)^2 = 1 ∧ p ≠ q ∧ abs (p - q) = 2) :=
by
  intro h
  use (2 : ℝ)
  use (0 : ℝ)
  -- Formal proof omitted
  sorry

end chord_length_circle_l2313_231316


namespace line_slope_and_point_l2313_231317

noncomputable def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem line_slope_and_point (m b : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 5) (h₃ : y₀ = 2) (h₄ : y₀ = line_equation x₀ m b) :
  m + b = 14 :=
by
  sorry

end line_slope_and_point_l2313_231317


namespace second_player_cannot_prevent_first_l2313_231387

noncomputable def player_choice (set_x2_coeff_to_zero : Prop) (first_player_sets : Prop) (second_player_cannot_prevent : Prop) : Prop :=
  ∀ (b : ℝ) (c : ℝ), (set_x2_coeff_to_zero ∧ first_player_sets ∧ second_player_cannot_prevent) → 
  (∀ x : ℝ, x^3 + b * x + c = 0 → ∃! x : ℝ, x^3 + b * x + c = 0)

theorem second_player_cannot_prevent_first (b c : ℝ) :
  player_choice (set_x2_coeff_to_zero := true)
                (first_player_sets := true)
                (second_player_cannot_prevent := true) :=
sorry

end second_player_cannot_prevent_first_l2313_231387


namespace find_z_l2313_231337

/- Definitions of angles and their relationships -/
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

/- Given conditions -/
def ABC : ℝ := 75
def BAC : ℝ := 55
def BCA : ℝ := 180 - ABC - BAC  -- This follows from the angle sum property of triangle ABC
def DCE : ℝ := BCA
def CDE : ℝ := 90

/- Prove z given the above conditions -/
theorem find_z : ∃ (z : ℝ), z = 90 - DCE := by
  use 40
  sorry

end find_z_l2313_231337


namespace total_weight_tommy_ordered_l2313_231383

theorem total_weight_tommy_ordered :
  let apples := 3
  let oranges := 1
  let grapes := 3
  let strawberries := 3
  apples + oranges + grapes + strawberries = 10 := by
  sorry

end total_weight_tommy_ordered_l2313_231383


namespace find_full_price_l2313_231340

-- Defining the conditions
variables (P : ℝ) 
-- The condition that 20% of the laptop's total cost is $240.
def condition : Prop := 0.2 * P = 240

-- The proof goal is to show that the full price P is $1200 given the condition
theorem find_full_price (h : condition P) : P = 1200 :=
sorry

end find_full_price_l2313_231340


namespace num_sequences_to_initial_position_8_l2313_231310

def validSequenceCount : ℕ := 4900

noncomputable def numberOfSequencesToInitialPosition (n : ℕ) : ℕ :=
if h : n = 8 then validSequenceCount else 0

theorem num_sequences_to_initial_position_8 :
  numberOfSequencesToInitialPosition 8 = 4900 :=
by
  sorry

end num_sequences_to_initial_position_8_l2313_231310


namespace probability_none_solve_l2313_231350

theorem probability_none_solve (a b c : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_prob : ((1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15)) : 
  (1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15 := 
by 
  sorry

end probability_none_solve_l2313_231350


namespace calculate_dollar_value_l2313_231347

def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y - 5

theorem calculate_dollar_value : dollar 3 (-1) = -5 := by
  sorry

end calculate_dollar_value_l2313_231347


namespace perpendicular_lines_k_value_l2313_231349

theorem perpendicular_lines_k_value (k : ℚ) : (∀ x y : ℚ, y = 3 * x + 7) ∧ (∀ x y : ℚ, 4 * y + k * x = 4) → k = 4 / 3 :=
by
  sorry

end perpendicular_lines_k_value_l2313_231349


namespace same_terminal_side_l2313_231332

theorem same_terminal_side (k : ℤ) : 
  ∃ (α : ℤ), α = k * 360 + 330 ∧ (α = 510 ∨ α = 150 ∨ α = -150 ∨ α = -390) :=
by
  sorry

end same_terminal_side_l2313_231332


namespace value_of_coupon_l2313_231324

theorem value_of_coupon (price_per_bag : ℝ) (oz_per_bag : ℕ) (cost_per_serving_with_coupon : ℝ) (total_servings : ℕ) :
  price_per_bag = 25 → oz_per_bag = 40 → cost_per_serving_with_coupon = 0.50 → total_servings = 40 →
  (price_per_bag - (cost_per_serving_with_coupon * total_servings)) = 5 :=
by 
  intros hpb hob hcpwcs hts
  sorry

end value_of_coupon_l2313_231324


namespace find_bloom_day_l2313_231319

def days := {d : Fin 7 // 1 ≤ d.val ∧ d.val ≤ 7}

def sunflowers_bloom (d : days) : Prop :=
¬ (d.val = 2 ∨ d.val = 4 ∨ d.val = 7)

def lilies_bloom (d : days) : Prop :=
¬ (d.val = 4 ∨ d.val = 6)

def magnolias_bloom (d : days) : Prop :=
¬ (d.val = 7)

def all_bloom_together (d : days) : Prop :=
sunflowers_bloom d ∧ lilies_bloom d ∧ magnolias_bloom d

def blooms_simultaneously (d : days) : Prop :=
∀ d1 d2 d3 : days, (d1 = d ∧ d2 = d ∧ d3 = d) →
(all_bloom_together d1 ∧ all_bloom_together d2 ∧ all_bloom_together d3)

theorem find_bloom_day :
  ∃ d : days, blooms_simultaneously d :=
sorry

end find_bloom_day_l2313_231319


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l2313_231301

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l2313_231301


namespace max_area_enclosed_by_fencing_l2313_231345

theorem max_area_enclosed_by_fencing (l w : ℕ) (h : 2 * (l + w) = 142) : l * w ≤ 1260 :=
sorry

end max_area_enclosed_by_fencing_l2313_231345


namespace minute_hand_rotation_l2313_231393

theorem minute_hand_rotation (h : ℕ) (radians_per_rotation : ℝ) : h = 5 → radians_per_rotation = 2 * Real.pi → - (h * radians_per_rotation) = -10 * Real.pi :=
by
  intros h_eq rp_eq
  rw [h_eq, rp_eq]
  sorry

end minute_hand_rotation_l2313_231393


namespace monotonicity_f_geq_f_neg_l2313_231360

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) ∧
  (a > 0 →
    (∀ x1 x2 : ℝ, x1 > Real.log a → x2 > Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2) ∧
    (∀ x1 x2 : ℝ, x1 < Real.log a → x2 < Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2)) :=
by sorry

theorem f_geq_f_neg (x : ℝ) (hx : x ≥ 0) : f 1 x ≥ f 1 (-x) :=
by sorry

end monotonicity_f_geq_f_neg_l2313_231360


namespace bird_difference_l2313_231374

-- Variables representing given conditions
def num_migrating_families : Nat := 86
def num_remaining_families : Nat := 45
def avg_birds_per_migrating_family : Nat := 12
def avg_birds_per_remaining_family : Nat := 8

-- Definition to calculate total number of birds for migrating families
def total_birds_migrating : Nat := num_migrating_families * avg_birds_per_migrating_family

-- Definition to calculate total number of birds for remaining families
def total_birds_remaining : Nat := num_remaining_families * avg_birds_per_remaining_family

-- The statement that we need to prove
theorem bird_difference (h : total_birds_migrating - total_birds_remaining = 672) : 
  total_birds_migrating - total_birds_remaining = 672 := 
sorry

end bird_difference_l2313_231374


namespace artist_used_17_ounces_of_paint_l2313_231330

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l2313_231330


namespace minimum_bag_count_l2313_231313

theorem minimum_bag_count (n a b : ℕ) (h1 : 7 * a + 11 * b = 77) (h2 : a + b = n) : n = 17 :=
by
  sorry

end minimum_bag_count_l2313_231313


namespace days_b_worked_l2313_231314

theorem days_b_worked (A_days B_days A_remaining_days : ℝ) (A_work_rate B_work_rate total_work : ℝ)
  (hA_rate : A_work_rate = 1 / A_days)
  (hB_rate : B_work_rate = 1 / B_days)
  (hA_days : A_days = 9)
  (hB_days : B_days = 15)
  (hA_remaining : A_remaining_days = 3)
  (h_total_work : ∀ x : ℝ, (x * B_work_rate + A_remaining_days * A_work_rate = total_work)) :
  ∃ x : ℝ, x = 10 :=
by
  sorry

end days_b_worked_l2313_231314


namespace order_of_magnitudes_l2313_231339

theorem order_of_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : x < x^(x^x) ∧ x^(x^x) < x^x :=
by
  -- Definitions for y and z.
  let y := x^x
  let z := x^(x^x)
  have h1 : x < y := sorry
  have h2 : z < y := sorry
  have h3 : x < z := sorry
  exact ⟨h3, h2⟩

end order_of_magnitudes_l2313_231339


namespace average_weight_of_rock_l2313_231370

-- Define all the conditions
def price_per_pound : ℝ := 4
def total_amount : ℝ := 60
def number_of_rocks : ℕ := 10

-- The statement we need to prove
theorem average_weight_of_rock :
  (total_amount / price_per_pound) / number_of_rocks = 1.5 :=
sorry

end average_weight_of_rock_l2313_231370


namespace intersection_M_N_l2313_231329

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

-- The theorem stating that the intersection of M and N is {1, 5}
theorem intersection_M_N :
  M ∩ N = {1, 5} :=
  sorry

end intersection_M_N_l2313_231329


namespace peanut_butter_revenue_l2313_231376

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l2313_231376


namespace correct_statement_l2313_231373

variables {Line Plane : Type}
variable (a b c : Line)
variable (M N : Plane)

/- Definitions for the conditions -/
def lies_on_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

/- Conditions -/
axiom h1 : lies_on_plane a M
axiom h2 : lies_on_plane b N
axiom h3 : intersection M N = c

/- The correct statement to be proved -/
theorem correct_statement : parallel a b → parallel a c :=
by sorry

end correct_statement_l2313_231373


namespace geometric_sequence_value_of_m_l2313_231392

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_value_of_m (r : ℝ) (hr : r ≠ 1) 
    (h1 : is_geometric_sequence a r)
    (h2 : a 5 * a 6 + a 4 * a 7 = 18) 
    (h3 : a 1 * a m = 9) :
  m = 10 :=
by
  sorry

end geometric_sequence_value_of_m_l2313_231392


namespace speed_in_still_water_l2313_231342

def upstream_speed : ℝ := 35
def downstream_speed : ℝ := 45

theorem speed_in_still_water:
  (upstream_speed + downstream_speed) / 2 = 40 := 
by
  sorry

end speed_in_still_water_l2313_231342


namespace relationship_M_N_l2313_231312

-- Define the sets M and N based on the conditions
def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

-- The statement to be proved
theorem relationship_M_N : ¬ (M ⊆ N) ∧ ¬ (N ⊆ M) :=
by
  sorry

end relationship_M_N_l2313_231312


namespace quadratic_no_solution_l2313_231348

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_no_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  0 < a ∧ discriminant a b c ≤ 0 :=
by
  sorry

end quadratic_no_solution_l2313_231348


namespace problem_statement_l2313_231398

theorem problem_statement (x y : ℝ) (h1 : |x| + x - y = 16) (h2 : x - |y| + y = -8) : x + y = -8 := sorry

end problem_statement_l2313_231398


namespace find_a_find_cos_2C_l2313_231304

noncomputable def triangle_side_a (A B : Real) (b : Real) (cosA : Real) : Real := 
  3

theorem find_a (A : Real) (B : Real) (b : Real) (cosA : Real) 
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3) 
  (h₃ : B = A + Real.pi / 2) : 
  triangle_side_a A B b cosA = 3 := by
  sorry

noncomputable def cos_2C (A B C a b : Real) (cosA sinC : Real) : Real :=
  7 / 9

theorem find_cos_2C (A : Real) (B : Real) (C : Real) (a : Real) (b : Real) (cosA : Real) (sinC: Real)
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3)
  (h₃ : B = A + Real.pi /2)
  (h₄ : a = 3)
  (h₅ : sinC = 1 / 3) :
  cos_2C A B C a b cosA sinC = 7 / 9 := by
  sorry

end find_a_find_cos_2C_l2313_231304


namespace new_shape_perimeter_l2313_231328

-- Definitions based on conditions
def square_side : ℕ := 64 / 4
def is_tri_isosceles (a b c : ℕ) : Prop := a = b

-- Definition of given problem setup and perimeter calculation
theorem new_shape_perimeter
  (side : ℕ)
  (tri_side1 tri_side2 base : ℕ)
  (h_square_side : side = 64 / 4)
  (h_tri1 : tri_side1 = side)
  (h_tri2 : tri_side2 = side)
  (h_base : base = side) :
  (side * 5) = 80 :=
by
  sorry

end new_shape_perimeter_l2313_231328


namespace perimeter_original_square_l2313_231357

theorem perimeter_original_square (s : ℝ) (h1 : (3 / 4) * s^2 = 48) : 4 * s = 32 :=
by
  sorry

end perimeter_original_square_l2313_231357


namespace bert_spent_fraction_at_hardware_store_l2313_231362

variable (f : ℝ)

def initial_money : ℝ := 41.99
def after_hardware (f : ℝ) := (1 - f) * initial_money
def after_dry_cleaners (f : ℝ) := after_hardware f - 7
def after_grocery (f : ℝ) := 0.5 * after_dry_cleaners f

theorem bert_spent_fraction_at_hardware_store 
(h1 : after_grocery f = 10.50) : 
  f = 0.3332 :=
by
  sorry

end bert_spent_fraction_at_hardware_store_l2313_231362


namespace marion_score_correct_l2313_231325

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l2313_231325


namespace y_value_l2313_231327

-- Given conditions
variables (x y : ℝ)
axiom h1 : x - y = 20
axiom h2 : x + y = 14

-- Prove that y = -3
theorem y_value : y = -3 :=
by { sorry }

end y_value_l2313_231327


namespace stick_horisontal_fall_position_l2313_231368

-- Definitions based on the conditions
def stick_length : ℝ := 120 -- length of the stick in cm
def projection_distance : ℝ := 70 -- distance between projections of the ends of the stick on the floor

-- The main theorem to prove
theorem stick_horisontal_fall_position :
  ∀ (L d : ℝ), L = stick_length ∧ d = projection_distance → 
  ∃ x : ℝ, x = 25 :=
by
  intros L d h
  have h1 : L = stick_length := h.1
  have h2 : d = projection_distance := h.2
  -- The detailed proof steps will be here
  sorry

end stick_horisontal_fall_position_l2313_231368


namespace arithmetic_expression_eval_l2313_231306

theorem arithmetic_expression_eval : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 :=
by
  sorry

end arithmetic_expression_eval_l2313_231306


namespace number_of_buses_used_l2313_231364

-- Definitions based on the conditions
def total_students : ℕ := 360
def students_per_bus : ℕ := 45

-- The theorem we need to prove
theorem number_of_buses_used : total_students / students_per_bus = 8 := 
by sorry

end number_of_buses_used_l2313_231364


namespace boys_girls_ratio_l2313_231315

theorem boys_girls_ratio (T G : ℕ) (h : (1/2 : ℚ) * G = (1/6 : ℚ) * T) :
  ((T - G) : ℚ) / G = 2 :=
by 
  sorry

end boys_girls_ratio_l2313_231315


namespace initial_weight_of_mixture_eq_20_l2313_231354

theorem initial_weight_of_mixture_eq_20
  (W : ℝ) (h1 : 0.1 * W + 4 = 0.25 * (W + 4)) :
  W = 20 :=
by
  sorry

end initial_weight_of_mixture_eq_20_l2313_231354


namespace ellipse_through_points_parabola_equation_l2313_231333

-- Ellipse Problem: Prove the standard equation
theorem ellipse_through_points (m n : ℝ) (m_pos : m > 0) (n_pos : n > 0) (m_ne_n : m ≠ n) :
  (m * 0^2 + n * (5/3)^2 = 1) ∧ (m * 1^2 + n * 1^2 = 1) →
  (m = 16 / 25 ∧ n = 9 / 25) → (m * x^2 + n * y^2 = 1) ↔ (16 * x^2 + 9 * y^2 = 225) :=
sorry

-- Parabola Problem: Prove the equation
theorem parabola_equation (p x y : ℝ) (p_pos : p > 0)
  (dist_focus : abs (x + p / 2) = 10) (dist_axis : y^2 = 36) :
  (p = 2 ∨ p = 18) →
  (y^2 = 2 * p * x) ↔ (y^2 = 4 * x ∨ y^2 = 36 * x) :=
sorry

end ellipse_through_points_parabola_equation_l2313_231333


namespace professors_seat_choice_count_l2313_231351

theorem professors_seat_choice_count : 
    let chairs := 11 -- number of chairs
    let students := 7 -- number of students
    let professors := 4 -- number of professors
    ∀ (P: Fin professors -> Fin chairs), 
    (∀ (p : Fin professors), 1 ≤ P p ∧ P p ≤ 9) -- Each professor is between seats 2-10
    ∧ (P 0 < P 1) ∧ (P 1 < P 2) ∧ (P 2 < P 3) -- Professors must be placed with at least one seat gap
    ∧ (P 0 ≠ 1 ∧ P 3 ≠ 11) -- First and last seats are excluded
    → ∃ (ways : ℕ), ways = 840 := sorry

end professors_seat_choice_count_l2313_231351


namespace prove_proposition_false_l2313_231358

def proposition (a : ℝ) := ∃ x : ℝ, x^2 - 4*a*x + 3 < 0

theorem prove_proposition_false : proposition 0 = False :=
by
sorry

end prove_proposition_false_l2313_231358


namespace sum_smallest_largest_l2313_231380

theorem sum_smallest_largest (n a : ℕ) (h_even_n : n % 2 = 0) (y x : ℕ)
  (h_y : y = a + n - 1)
  (h_x : x = (a + 3 * (n / 3 - 1)) * (n / 3)) : 
  2 * y = a + (a + 2 * (n - 1)) :=
by
  sorry

end sum_smallest_largest_l2313_231380


namespace prime_divisibility_l2313_231395

theorem prime_divisibility
  (a b : ℕ) (p q : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hm1 : ¬ p ∣ q - 1)
  (hm2 : q ∣ a ^ p - b ^ p) : q ∣ a - b :=
sorry

end prime_divisibility_l2313_231395


namespace jack_total_books_is_541_l2313_231320

-- Define the number of books in each section
def american_books : ℕ := 6 * 34
def british_books : ℕ := 8 * 29
def world_books : ℕ := 5 * 21

-- Define the total number of books based on the given sections
def total_books : ℕ := american_books + british_books + world_books

-- Prove that the total number of books is 541
theorem jack_total_books_is_541 : total_books = 541 :=
by
  sorry

end jack_total_books_is_541_l2313_231320


namespace faye_earned_total_money_l2313_231334

def bead_necklaces : ℕ := 3
def gem_necklaces : ℕ := 7
def price_per_necklace : ℕ := 7

theorem faye_earned_total_money :
  (bead_necklaces + gem_necklaces) * price_per_necklace = 70 :=
by
  sorry

end faye_earned_total_money_l2313_231334


namespace minimum_value_of_m_plus_n_l2313_231397

noncomputable def m (a b : ℝ) : ℝ := b + (1 / a)
noncomputable def n (a b : ℝ) : ℝ := a + (1 / b)

theorem minimum_value_of_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 1) :
  m a b + n a b = 4 :=
sorry

end minimum_value_of_m_plus_n_l2313_231397


namespace volume_of_pyramid_l2313_231361

noncomputable def volume_of_regular_triangular_pyramid (h R : ℝ) : ℝ :=
  (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4

theorem volume_of_pyramid (h R : ℝ) : volume_of_regular_triangular_pyramid h R = (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4 :=
  by sorry

end volume_of_pyramid_l2313_231361
