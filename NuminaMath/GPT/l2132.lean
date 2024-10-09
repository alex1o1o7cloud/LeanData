import Mathlib

namespace tangent_line_equation_at_1_range_of_a_l2132_213255

noncomputable def f (x a : ℝ) : ℝ := (x+1) * Real.log x - a * (x-1)

-- (I) Tangent line equation when a = 4
theorem tangent_line_equation_at_1 (x : ℝ) (hx : x = 1) :
  let a := 4
  2*x + f 1 a - 2 = 0 :=
sorry

-- (II) Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_equation_at_1_range_of_a_l2132_213255


namespace factor_expression_l2132_213283

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end factor_expression_l2132_213283


namespace quadratic_vertex_position_l2132_213291

theorem quadratic_vertex_position (a p q m : ℝ) (ha : 0 < a) (hpq : p < q) (hA : p = a * (-1 - m)^2) (hB : q = a * (3 - m)^2) : m ≠ 2 :=
by
  sorry

end quadratic_vertex_position_l2132_213291


namespace police_female_officers_l2132_213205

theorem police_female_officers (perc : ℝ) (total_on_duty: ℝ) (half_on_duty : ℝ) (F : ℝ) :
    perc = 0.18 →
    total_on_duty = 144 →
    half_on_duty = total_on_duty / 2 →
    half_on_duty = perc * F →
    F = 400 :=
by
  sorry

end police_female_officers_l2132_213205


namespace compute_expression_l2132_213241

theorem compute_expression:
  let a := 3
  let b := 7
  (a + b) ^ 2 + Real.sqrt (a^2 + b^2) = 100 + Real.sqrt 58 :=
by
  sorry

end compute_expression_l2132_213241


namespace progress_regress_ratio_l2132_213270

theorem progress_regress_ratio :
  let progress_rate := 1.2
  let regress_rate := 0.8
  let log2 := 0.3010
  let log3 := 0.4771
  let target_ratio := 10000
  (progress_rate / regress_rate) ^ 23 = target_ratio :=
by
  sorry

end progress_regress_ratio_l2132_213270


namespace total_bins_correct_l2132_213225

def total_bins (soup vegetables pasta : ℝ) : ℝ :=
  soup + vegetables + pasta

theorem total_bins_correct : total_bins 0.12 0.12 0.5 = 0.74 :=
  by
    sorry

end total_bins_correct_l2132_213225


namespace rectangle_length_is_16_l2132_213230

-- Define the conditions
def side_length_square : ℕ := 8
def width_rectangle : ℕ := 4
def area_square : ℕ := side_length_square ^ 2  -- Area of the square
def area_rectangle (length : ℕ) : ℕ := width_rectangle * length  -- Area of the rectangle

-- Lean 4 statement
theorem rectangle_length_is_16 (L : ℕ) (h : area_square = area_rectangle L) : L = 16 :=
by
  /- Proof will be inserted here -/
  sorry

end rectangle_length_is_16_l2132_213230


namespace month_days_l2132_213217

theorem month_days (letters_per_day packages_per_day total_mail six_months : ℕ) (h1 : letters_per_day = 60) (h2 : packages_per_day = 20) (h3 : total_mail = 14400) (h4 : six_months = 6) : 
  total_mail / (letters_per_day + packages_per_day) / six_months = 30 :=
by sorry

end month_days_l2132_213217


namespace asha_money_remaining_l2132_213261

-- Given conditions as definitions in Lean
def borrowed_from_brother : ℕ := 20
def borrowed_from_father : ℕ := 40
def borrowed_from_mother : ℕ := 30
def gift_from_granny : ℕ := 70
def initial_savings : ℕ := 100

-- Total amount of money Asha has
def total_money : ℕ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + initial_savings

-- Money spent by Asha
def money_spent : ℕ := (3 * total_money) / 4

-- Money remaining with Asha
def money_remaining : ℕ := total_money - money_spent

-- Theorem stating the result
theorem asha_money_remaining : money_remaining = 65 := by
  sorry

end asha_money_remaining_l2132_213261


namespace mary_initial_triangles_l2132_213223

theorem mary_initial_triangles (s t : ℕ) (h1 : s + t = 10) (h2 : 4 * s + 3 * t = 36) : t = 4 :=
by
  sorry

end mary_initial_triangles_l2132_213223


namespace hyperbola_standard_equation_l2132_213242

def a : ℕ := 5
def c : ℕ := 7
def b_squared : ℕ := c * c - a * a

theorem hyperbola_standard_equation (a_eq : a = 5) (c_eq : c = 7) :
    (b_squared = 24) →
    ( ∀ x y : ℝ, x^2 / (a^2 : ℝ) - y^2 / (b_squared : ℝ) = 1 ∨ 
                   y^2 / (a^2 : ℝ) - x^2 / (b_squared : ℝ) = 1) :=
by
  sorry

end hyperbola_standard_equation_l2132_213242


namespace discount_difference_l2132_213216

-- Definitions based on given conditions
def original_bill : ℝ := 8000
def single_discount_rate : ℝ := 0.30
def first_successive_discount_rate : ℝ := 0.26
def second_successive_discount_rate : ℝ := 0.05

-- Calculations based on conditions
def single_discount_final_amount := original_bill * (1 - single_discount_rate)
def first_successive_discount_final_amount := original_bill * (1 - first_successive_discount_rate)
def complete_successive_discount_final_amount := 
  first_successive_discount_final_amount * (1 - second_successive_discount_rate)

-- Proof statement
theorem discount_difference :
  single_discount_final_amount - complete_successive_discount_final_amount = 24 := 
  by
    -- Proof to be provided
    sorry

end discount_difference_l2132_213216


namespace common_ratio_of_geometric_sequence_l2132_213234

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h4 : ∀ n, a n ≤ a (n + 1)) :
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l2132_213234


namespace solve_table_assignment_l2132_213202

noncomputable def table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) : Prop :=
  let Albert := T_4
  let Bogdan := T_2
  let Vadim := T_1
  let Denis := T_3
  (∀ x, x ∈ Vadim ↔ x ∉ (Albert ∪ Bogdan)) ∧
  (∀ x, x ∈ Denis ↔ x ∉ (Bogdan ∪ Vadim)) ∧
  Albert = T_4 ∧
  Bogdan = T_2 ∧
  Vadim = T_1 ∧
  Denis = T_3

theorem solve_table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) :
  table_assignment T_1 T_2 T_3 T_4 :=
sorry

end solve_table_assignment_l2132_213202


namespace tan_double_angle_third_quadrant_l2132_213253

theorem tan_double_angle_third_quadrant
  (α : ℝ)
  (sin_alpha : Real.sin α = -3/5)
  (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.tan (2 * α) = 24 / 7 :=
sorry

end tan_double_angle_third_quadrant_l2132_213253


namespace alpha_values_l2132_213237

noncomputable def α := Complex

theorem alpha_values (α : Complex) :
  (α ≠ 1) ∧ 
  (Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1)) ∧ 
  (Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) ∧ 
  (Real.cos α.arg = 1 / 2) →
  α = Complex.mk ((-1 + Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 + Real.sqrt 33) / 4)^2))) ∨ 
  α = Complex.mk ((-1 - Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 - Real.sqrt 33) / 4)^2))) :=
sorry

end alpha_values_l2132_213237


namespace exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l2132_213280

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- N-gon properties
def is_convex_ngon (N : ℕ) (vertices : List (ℝ × ℝ)) : Prop :=
  -- Placeholder for checking properties; actual implementation would validate convexity and equilateral nature.
  sorry 

-- Statement for 2011-gon
theorem exists_convex_2011_gon_on_parabola :
  ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2011 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

-- Statement for 2012-gon
theorem not_exists_convex_2012_gon_on_parabola :
  ¬ ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2012 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

end exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l2132_213280


namespace abs_sum_less_abs_diff_l2132_213293

theorem abs_sum_less_abs_diff {a b : ℝ} (hab : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_sum_less_abs_diff_l2132_213293


namespace min_unit_cubes_l2132_213224

theorem min_unit_cubes (l w h : ℕ) (S : ℕ) (hS : S = 52) 
  (hSurface : 2 * (l * w + l * h + w * h) = S) : 
  ∃ l w h, l * w * h = 16 :=
by
  -- start the proof here
  sorry

end min_unit_cubes_l2132_213224


namespace problem1_problem2_l2132_213232

-- For problem (1)
noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (1 + x))

theorem problem1 (α : ℝ) (h_alpha : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

-- For problem (2)
theorem problem2 : Real.sin (Real.pi * 50 / 180) * (1 + Real.sqrt 3 * Real.tan (Real.pi * 10 / 180)) = 1 := by
  sorry

end problem1_problem2_l2132_213232


namespace least_x_value_l2132_213286

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_x_value (x : ℕ) (p : ℕ) (hp : is_prime p) (h : x / (12 * p) = 2) : x = 48 := by
  sorry

end least_x_value_l2132_213286


namespace net_percentage_error_in_volume_l2132_213248

theorem net_percentage_error_in_volume
  (a : ℝ)
  (side_error : ℝ := 0.03)
  (height_error : ℝ := -0.04)
  (depth_error : ℝ := 0.02) :
  ((1 + side_error) * (1 + height_error) * (1 + depth_error) - 1) * 100 = 0.8656 :=
by
  -- Placeholder for the proof
  sorry

end net_percentage_error_in_volume_l2132_213248


namespace statement_books_per_shelf_l2132_213243

/--
A store initially has 40.0 coloring books.
Acquires 20.0 more books.
Uses 15 shelves to store the books equally.
-/
def initial_books : ℝ := 40.0
def acquired_books : ℝ := 20.0
def total_shelves : ℝ := 15.0

/-- 
Theorem statement: The number of coloring books on each shelf.
-/
theorem books_per_shelf : (initial_books + acquired_books) / total_shelves = 4.0 := by
  sorry

end statement_books_per_shelf_l2132_213243


namespace raft_sticks_total_l2132_213233

theorem raft_sticks_total : 
  let S := 45 
  let G := (3/5 * 45 : ℝ)
  let M := 45 + G + 15
  let D := 2 * M - 7
  S + G + M + D = 326 := 
by
  sorry

end raft_sticks_total_l2132_213233


namespace train_total_travel_time_l2132_213284

noncomputable def totalTravelTime (d1 d2 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + (d2 / s2)

theorem train_total_travel_time : 
  totalTravelTime 150 200 50 80 = 5.5 :=
by
  sorry

end train_total_travel_time_l2132_213284


namespace coordinates_of_point_P_l2132_213289

theorem coordinates_of_point_P (x y : ℝ) (h1 : x > 0) (h2 : y < 0) (h3 : abs y = 2) (h4 : abs x = 4) : (x, y) = (4, -2) :=
by
  sorry

end coordinates_of_point_P_l2132_213289


namespace exists_within_distance_l2132_213279

theorem exists_within_distance (a : ℝ) (n : ℕ) (h₁ : a > 0) (h₂ : n > 0) :
  ∃ k : ℕ, k < n ∧ ∃ m : ℤ, |k * a - m| < 1 / n :=
by
  sorry

end exists_within_distance_l2132_213279


namespace remainder_7n_mod_5_l2132_213213

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
by 
  sorry

end remainder_7n_mod_5_l2132_213213


namespace problem1_problem2_l2132_213298

noncomputable def f : ℝ → ℝ := -- we assume f is noncomputable since we know its explicit form in the desired interval
sorry

axiom periodic_f (x : ℝ) : f (x + 5) = f x
axiom odd_f {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f x
axiom quadratic_f {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * (x - 2) ^ 2 - 5
axiom minimum_f : f 2 = -5

theorem problem1 : f 1 + f 4 = 0 :=
by
  sorry

theorem problem2 {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * x ^ 2 - 8 * x + 3 :=
by
  sorry

end problem1_problem2_l2132_213298


namespace johnny_ways_to_choose_l2132_213214

def num_ways_to_choose_marbles (total_marbles : ℕ) (marbles_to_choose : ℕ) (blue_must_be_included : ℕ) : ℕ :=
  Nat.choose (total_marbles - blue_must_be_included) (marbles_to_choose - blue_must_be_included)

-- Given conditions
def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_must_be_included : ℕ := 1

-- Theorem to prove the number of ways to choose the marbles
theorem johnny_ways_to_choose :
  num_ways_to_choose_marbles total_marbles marbles_to_choose blue_must_be_included = 56 := by
  sorry

end johnny_ways_to_choose_l2132_213214


namespace inclination_angle_l2132_213277

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end inclination_angle_l2132_213277


namespace find_matrix_l2132_213240

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M^3 - 3 * M^2 + 2 * M = ![![8, 16], ![4, 8]]) : 
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_l2132_213240


namespace sum_of_a_b_l2132_213272

variable {a b : ℝ}

theorem sum_of_a_b (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) : a + b = 1 ∨ a + b = -1 := 
by 
  sorry

end sum_of_a_b_l2132_213272


namespace find_b_l2132_213231

def direction_vector (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - x1, y2 - y1)

theorem find_b (b : ℝ)
  (hx1 : ℝ := -3) (hy1 : ℝ := 1) (hx2 : ℝ := 0) (hy2 : ℝ := 4)
  (hdir : direction_vector hx1 hy1 hx2 hy2 = (3, b)) :
  b = 3 :=
by
  -- Mathematical proof of b = 3 goes here
  sorry

end find_b_l2132_213231


namespace square_area_proof_square_area_square_area_final_square_area_correct_l2132_213239

theorem square_area_proof (x : ℝ) (s1 : ℝ) (s2 : ℝ) (A : ℝ)
  (h1 : s1 = 5 * x - 20)
  (h2 : s2 = 25 - 2 * x)
  (h3 : s1 = s2) :
  A = (s1 * s1) := by
  -- We need to prove A = s1 * s1
  sorry

theorem square_area (x : ℝ) (s : ℝ) (h : s = 85 / 7) :
  s ^ 2 = 7225 / 49 := by
  -- We need to prove s^2 = 7225 / 49
  sorry

theorem square_area_final (x : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (A : ℝ) :
  A = (85 / 7) ^ 2 := by
  -- We need to prove A = (85 / 7) ^ 2
  sorry

theorem square_area_correct (x : ℝ)
  (A : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (h2 : A = (85 / 7) ^ 2) :
  A = 7225 / 49 := by
  -- We need to prove A = 7225 / 49
  sorry

end square_area_proof_square_area_square_area_final_square_area_correct_l2132_213239


namespace abs_inequality_solution_set_l2132_213246

theorem abs_inequality_solution_set :
  { x : ℝ | |x - 1| + |x + 2| ≥ 5 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end abs_inequality_solution_set_l2132_213246


namespace P_eq_Q_l2132_213276

def P (m : ℝ) : Prop := -1 < m ∧ m < 0

def quadratic_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 + 4 * m * x - 4 < 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, quadratic_inequality m x

theorem P_eq_Q : ∀ m : ℝ, P m ↔ Q m := 
by 
  sorry

end P_eq_Q_l2132_213276


namespace expand_binom_l2132_213264

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l2132_213264


namespace raise_3000_yuan_probability_l2132_213221

def prob_correct_1 : ℝ := 0.9
def prob_correct_2 : ℝ := 0.5
def prob_correct_3 : ℝ := 0.4
def prob_incorrect_3 : ℝ := 1 - prob_correct_3

def fund_first : ℝ := 1000
def fund_second : ℝ := 2000
def fund_third : ℝ := 3000

def prob_raise_3000_yuan : ℝ := prob_correct_1 * prob_correct_2 * prob_incorrect_3

theorem raise_3000_yuan_probability :
  prob_raise_3000_yuan = 0.27 :=
by
  sorry

end raise_3000_yuan_probability_l2132_213221


namespace sufficient_but_not_necessary_condition_l2132_213218

theorem sufficient_but_not_necessary_condition (a b : ℝ) : 
  (a ≥ 1 ∧ b ≥ 1) → (a + b ≥ 2) ∧ ¬((a + b ≥ 2) → (a ≥ 1 ∧ b ≥ 1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l2132_213218


namespace problem_extraneous_root_l2132_213295

theorem problem_extraneous_root (m : ℤ) :
  (∃ x, x = -4 ∧ (x + 4 = 0) ∧ ((x-1)/(x+4) = m/(x+4)) ∧ (m = -5)) :=
sorry

end problem_extraneous_root_l2132_213295


namespace additional_tiles_needed_l2132_213235

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l2132_213235


namespace airplane_seats_l2132_213296

theorem airplane_seats (F : ℕ) (h : F + 4 * F + 2 = 387) : F = 77 := by
  -- Proof goes here
  sorry

end airplane_seats_l2132_213296


namespace work_completion_days_l2132_213215

theorem work_completion_days (Ry : ℝ) (R_combined : ℝ) (D : ℝ) :
  Ry = 1 / 40 ∧ R_combined = 1 / 13.333333333333332 → 1 / D + Ry = R_combined → D = 20 :=
by
  intros h_eqs h_combined
  sorry

end work_completion_days_l2132_213215


namespace equal_donations_amount_l2132_213211

def raffle_tickets_sold := 25
def cost_per_ticket := 2
def total_raised := 100
def single_donation := 20
def amount_equal_donations (D : ℕ) : Prop := 2 * D + single_donation = total_raised - (raffle_tickets_sold * cost_per_ticket)

theorem equal_donations_amount (D : ℕ) (h : amount_equal_donations D) : D = 15 :=
  sorry

end equal_donations_amount_l2132_213211


namespace circumscribed_sphere_radius_l2132_213222

/-- Define the right triangular prism -/
structure RightTriangularPrism :=
(AB AC BC : ℝ)
(AA1 : ℝ)
(h_base : AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ BC = 8)
(h_height : AA1 = 6)

/-- The condition that the base is an isosceles right-angled triangle -/
structure IsoscelesRightAngledTriangle :=
(A B C : ℝ)
(AB AC : ℝ)
(BC : ℝ)
(h_isosceles_right : AB = AC ∧ BC = Real.sqrt (AB^2 + AC^2))

/-- The main theorem stating the radius of the circumscribed sphere -/
theorem circumscribed_sphere_radius (prism : RightTriangularPrism) 
    (base : IsoscelesRightAngledTriangle) 
    (h_base_correct : base.AB = prism.AB ∧ base.AC = prism.AC ∧ base.BC = prism.BC):
    ∃ radius : ℝ, radius = 5 := 
by
    sorry

end circumscribed_sphere_radius_l2132_213222


namespace fuel_first_third_l2132_213297

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end fuel_first_third_l2132_213297


namespace max_squared_sum_of_sides_l2132_213251

variable {R : ℝ}
variable {O A B C : EucSpace} -- O is the center, A, B, and C are vertices
variable (a b c : ℝ)  -- Position vectors corresponding to vertices A, B, C

-- Hypotheses based on the problem conditions:
variable (h1 : ‖a‖ = R)
variable (h2 : ‖b‖ = R)
variable (h3 : ‖c‖ = R)
variable (hSumZero : a + b + c = 0)

theorem max_squared_sum_of_sides 
  {AB BC CA : ℝ} -- Side lengths
  (hAB : AB = ‖a - b‖)
  (hBC : BC = ‖b - c‖)
  (hCA : CA = ‖c - a‖) :
  AB^2 + BC^2 + CA^2 = 9 * R^2 :=
sorry

end max_squared_sum_of_sides_l2132_213251


namespace slope_of_line_l2132_213281

theorem slope_of_line
  (k : ℝ) 
  (hk : 0 < k) 
  (h1 : ¬ (2 / Real.sqrt (k^2 + 1) = 3 * 2 * Real.sqrt (1 - 8 * k^2) / Real.sqrt (k^2 + 1))) 
  : k = 1 / 3 :=
sorry

end slope_of_line_l2132_213281


namespace koi_fish_multiple_l2132_213269

theorem koi_fish_multiple (n m : ℕ) (h1 : n = 39) (h2 : m * n - 64 < n) : m * n = 78 :=
by
  sorry

end koi_fish_multiple_l2132_213269


namespace prove_expression_l2132_213256

def otimes (a b : ℚ) : ℚ := a^2 / b

theorem prove_expression : ((otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3))) = -2/3 :=
by 
  sorry

end prove_expression_l2132_213256


namespace not_speaking_hindi_is_32_l2132_213249

-- Definitions and conditions
def total_diplomats : ℕ := 120
def spoke_french : ℕ := 20
def percent_neither : ℝ := 0.20
def percent_both : ℝ := 0.10

-- Number of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi := (percent_neither * total_diplomats : ℝ)

-- Number of diplomats who spoke both French and Hindi
def both_french_and_hindi := (percent_both * total_diplomats : ℝ)

-- Number of diplomats who spoke only French
def only_french := (spoke_french - both_french_and_hindi : ℝ)

-- Number of diplomats who did not speak Hindi
def not_speaking_hindi := (only_french + neither_french_nor_hindi : ℝ)

theorem not_speaking_hindi_is_32 :
  not_speaking_hindi = 32 :=
by
  -- Provide proof here
  sorry

end not_speaking_hindi_is_32_l2132_213249


namespace outfit_combinations_l2132_213229

theorem outfit_combinations :
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6 -- 5 ties + no tie option
  let num_belts := 3 -- 2 belts + no belt option
  num_shirts * num_pants * num_ties * num_belts = 360 :=
by
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6
  let num_belts := 3
  show num_shirts * num_pants * num_ties * num_belts = 360
  sorry

end outfit_combinations_l2132_213229


namespace solve_inequality_when_a_is_one_range_of_values_for_a_l2132_213247

open Real

-- Part (1) Statement
theorem solve_inequality_when_a_is_one (a x : ℝ) (h : a = 1) : 
  |x - a| + |x + 2| ≤ 5 → -3 ≤ x ∧ x ≤ 2 := 
by sorry

-- Part (2) Statement
theorem range_of_values_for_a (a : ℝ) : 
  (∃ x_0 : ℝ, |x_0 - a| + |x_0 + 2| ≤ |2 * a + 1|) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end solve_inequality_when_a_is_one_range_of_values_for_a_l2132_213247


namespace cubic_equation_real_root_l2132_213226

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x + 25 = 0 := 
sorry

end cubic_equation_real_root_l2132_213226


namespace prob_A_and_B_truth_l2132_213220

-- Define the probabilities
def prob_A_truth := 0.70
def prob_B_truth := 0.60

-- State the theorem
theorem prob_A_and_B_truth : prob_A_truth * prob_B_truth = 0.42 :=
by
  sorry

end prob_A_and_B_truth_l2132_213220


namespace scientific_notation_15510000_l2132_213259

theorem scientific_notation_15510000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 15510000 = a * 10^n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_15510000_l2132_213259


namespace esperanza_gross_salary_l2132_213271

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end esperanza_gross_salary_l2132_213271


namespace magic_box_problem_l2132_213219

theorem magic_box_problem (m : ℝ) :
  (m^2 - 2*m - 1 = 2) → (m = 3 ∨ m = -1) :=
by
  intro h
  sorry

end magic_box_problem_l2132_213219


namespace max_integer_inequality_l2132_213273

theorem max_integer_inequality (a b c: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) :
  (a^2 / (b / 29 + c / 31) + b^2 / (c / 29 + a / 31) + c^2 / (a / 29 + b / 31)) ≥ 14 * (a + b + c) :=
sorry

end max_integer_inequality_l2132_213273


namespace possible_values_x_plus_y_l2132_213244

theorem possible_values_x_plus_y (x y : ℝ) (h1 : x = y * (3 - y)^2) (h2 : y = x * (3 - x)^2) :
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end possible_values_x_plus_y_l2132_213244


namespace no_base_b_square_of_integer_l2132_213299

theorem no_base_b_square_of_integer (b : ℕ) : ¬(∃ n : ℕ, n^2 = b^2 + 3 * b + 1) → b < 4 ∨ b > 8 := by
  sorry

end no_base_b_square_of_integer_l2132_213299


namespace arithmetic_sequence_k_value_l2132_213209

theorem arithmetic_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (S_pos : S 2016 > 0) (S_neg : S 2017 < 0)
  (H : ∀ n, |a n| ≥ |a 1009| ): k = 1009 :=
sorry

end arithmetic_sequence_k_value_l2132_213209


namespace probability_of_Z_l2132_213227

/-
  Given: 
  - P(W) = 3 / 8
  - P(X) = 1 / 4
  - P(Y) = 1 / 8

  Prove: 
  - P(Z) = 1 / 4 when P(Z) = 1 - (P(W) + P(X) + P(Y))
-/

theorem probability_of_Z (P_W P_X P_Y P_Z : ℚ) (h_W : P_W = 3 / 8) (h_X : P_X = 1 / 4) (h_Y : P_Y = 1 / 8) (h_Z : P_Z = 1 - (P_W + P_X + P_Y)) : 
  P_Z = 1 / 4 :=
by
  -- We can write the whole Lean Math proof here. However, per the instructions, we'll conclude with sorry.
  sorry

end probability_of_Z_l2132_213227


namespace specialSignLanguage_l2132_213203

theorem specialSignLanguage (S : ℕ) 
  (h1 : (S + 2) * (S + 2) = S * S + 1288) : S = 321 := 
by
  sorry

end specialSignLanguage_l2132_213203


namespace brad_red_balloons_l2132_213288

theorem brad_red_balloons (total balloons green : ℕ) (h1 : total = 17) (h2 : green = 9) : total - green = 8 := 
by {
  sorry
}

end brad_red_balloons_l2132_213288


namespace tangent_line_eq_l2132_213207

theorem tangent_line_eq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_eq_l2132_213207


namespace half_height_of_triangular_prism_l2132_213206

theorem half_height_of_triangular_prism (volume base_area height : ℝ) 
  (h_volume : volume = 576)
  (h_base_area : base_area = 3)
  (h_prism : volume = base_area * height) :
  height / 2 = 96 :=
by
  have h : height = volume / base_area := by sorry
  rw [h_volume, h_base_area] at h
  have h_height : height = 192 := by sorry
  rw [h_height]
  norm_num

end half_height_of_triangular_prism_l2132_213206


namespace total_area_of_L_shaped_figure_l2132_213275

-- Define the specific lengths for each segment
def bottom_rect_length : ℕ := 10
def bottom_rect_width : ℕ := 6
def central_rect_length : ℕ := 4
def central_rect_width : ℕ := 4
def top_rect_length : ℕ := 5
def top_rect_width : ℕ := 1

-- Calculate the area of each rectangle
def bottom_rect_area : ℕ := bottom_rect_length * bottom_rect_width
def central_rect_area : ℕ := central_rect_length * central_rect_width
def top_rect_area : ℕ := top_rect_length * top_rect_width

-- Given the length and width of the rectangles, calculate the total area of the L-shaped figure
theorem total_area_of_L_shaped_figure : 
  bottom_rect_area + central_rect_area + top_rect_area = 81 := by
  sorry

end total_area_of_L_shaped_figure_l2132_213275


namespace range_of_a_l2132_213238

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

noncomputable def g (x a : ℝ) := Real.sqrt x + Real.sqrt (a - x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= a → ∃ x2 : ℝ, 4 ≤ x2 ∧ x2 ≤ 16 ∧ g x1 a = f x2) →
  4 ≤ a ∧ a ≤ 8 :=
sorry 

end range_of_a_l2132_213238


namespace annual_rent_per_square_foot_is_156_l2132_213267

-- Given conditions
def monthly_rent : ℝ := 1300
def length : ℝ := 10
def width : ℝ := 10
def area : ℝ := length * width
def annual_rent : ℝ := monthly_rent * 12

-- Proof statement: Annual rent per square foot
theorem annual_rent_per_square_foot_is_156 : 
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_is_156_l2132_213267


namespace sum_of_series_is_918_l2132_213292

-- Define the first term a, common difference d, last term a_n,
-- and the number of terms n calculated from the conditions.
def first_term : Int := -300
def common_difference : Int := 3
def last_term : Int := 309
def num_terms : Int := 204 -- calculated as per the solution

-- Compute the sum of the arithmetic series
def sum_arithmetic_series (a d : Int) (n : Int) : Int :=
  n * (2 * a + (n - 1) * d) / 2

-- Prove that the sum of the series is 918
theorem sum_of_series_is_918 :
  sum_arithmetic_series first_term common_difference num_terms = 918 :=
by
  sorry

end sum_of_series_is_918_l2132_213292


namespace puzzle_solution_l2132_213268

-- Definitions for the digits
def K : ℕ := 3
def O : ℕ := 2
def M : ℕ := 4
def R : ℕ := 5
def E : ℕ := 6

-- The main proof statement
theorem puzzle_solution : (10 * K + O : ℕ) + (M / 10 + K / 10 + O / 100) = (10 * K + R : ℕ) + (O / 10 + M / 100) := 
  by 
  sorry

end puzzle_solution_l2132_213268


namespace productivity_increase_l2132_213278

theorem productivity_increase :
  (∃ d : ℝ, 
   (∀ n : ℕ, 0 < n → n ≤ 30 → 
      (5 + (n - 1) * d ≥ 0) ∧ 
      (30 * 5 + (30 * 29 / 2) * d = 390) ∧ 
      1 / 100 < d ∧ d < 1) ∧
      d = 0.52) :=
sorry

end productivity_increase_l2132_213278


namespace mike_found_four_more_seashells_l2132_213274

/--
Given:
1. Mike initially found 6.0 seashells.
2. The total number of seashells Mike had after finding more is 10.

Prove:
Mike found 4.0 more seashells.
-/
theorem mike_found_four_more_seashells (initial_seashells : ℝ) (total_seashells : ℝ)
  (h1 : initial_seashells = 6.0)
  (h2 : total_seashells = 10.0) :
  total_seashells - initial_seashells = 4.0 :=
by
  sorry

end mike_found_four_more_seashells_l2132_213274


namespace fraction_books_sold_l2132_213285

theorem fraction_books_sold (B : ℕ) (F : ℚ)
  (hc1 : F * B * 4 = 288)
  (hc2 : F * B + 36 = B) :
  F = 2 / 3 :=
by
  sorry

end fraction_books_sold_l2132_213285


namespace number_of_possible_values_l2132_213287

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l2132_213287


namespace count_perfect_cubes_l2132_213265

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end count_perfect_cubes_l2132_213265


namespace number_of_bugs_seen_l2132_213201

-- Defining the conditions
def flowers_per_bug : ℕ := 2
def total_flowers_eaten : ℕ := 6

-- The statement to prove
theorem number_of_bugs_seen : total_flowers_eaten / flowers_per_bug = 3 :=
by
  sorry

end number_of_bugs_seen_l2132_213201


namespace part_a_part_b_part_c_l2132_213260

def f (n d : ℕ) : ℕ := sorry

theorem part_a (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 ≤ n :=
sorry

theorem part_b (n d : ℕ) (h_even_n_minus_d : (n - d) % 2 = 0) : f n d ≤ (n + d) / (d + 1) :=
sorry

theorem part_c (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 = n :=
sorry

end part_a_part_b_part_c_l2132_213260


namespace geometric_series_sum_l2132_213245

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 5
  ∑' n : ℕ, a * r ^ n = 5 / 4 :=
by
  sorry

end geometric_series_sum_l2132_213245


namespace harmonic_mean_average_of_x_is_11_l2132_213266

theorem harmonic_mean_average_of_x_is_11 :
  let h := (2 * 1008) / (2 + 1008)
  ∃ (x : ℕ), (h + x) / 2 = 11 → x = 18 := by
  sorry

end harmonic_mean_average_of_x_is_11_l2132_213266


namespace same_solution_for_equations_l2132_213212

theorem same_solution_for_equations (b x : ℝ) :
  (2 * x + 7 = 3) → 
  (b * x - 10 = -2) → 
  b = -4 :=
by
  sorry

end same_solution_for_equations_l2132_213212


namespace right_triangular_prism_volume_l2132_213262

theorem right_triangular_prism_volume (R a h V : ℝ)
  (h1 : 4 * Real.pi * R^2 = 12 * Real.pi)
  (h2 : h = 2 * R)
  (h3 : (1 / 3) * (Real.sqrt 3 / 2) * a = R)
  (h4 : V = (1 / 2) * a * a * (Real.sin (Real.pi / 3)) * h) :
  V = 54 :=
by sorry

end right_triangular_prism_volume_l2132_213262


namespace gcd_power_diff_l2132_213258

theorem gcd_power_diff (m n : ℕ) (h1 : m = 2^2021 - 1) (h2 : n = 2^2000 - 1) :
  Nat.gcd m n = 2097151 :=
by sorry

end gcd_power_diff_l2132_213258


namespace average_of_other_two_numbers_l2132_213257

theorem average_of_other_two_numbers
  (avg_5_numbers : ℕ → ℚ)
  (sum_3_numbers : ℕ → ℚ)
  (h1 : ∀ n, avg_5_numbers n = 20)
  (h2 : ∀ n, sum_3_numbers n = 48)
  (h3 : ∀ n, ∃ x y z p q : ℚ, avg_5_numbers n = (x + y + z + p + q) / 5)
  (h4 : ∀ n, sum_3_numbers n = x + y + z) :
  ∃ u v : ℚ, ((u + v) / 2 = 26) :=
by sorry

end average_of_other_two_numbers_l2132_213257


namespace platform_length_is_260_meters_l2132_213204

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform_s : ℝ := 30
noncomputable def time_to_cross_man_s : ℝ := 17

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def length_of_train_m : ℝ := train_speed_mps * time_to_cross_man_s
noncomputable def total_distance_cross_platform_m : ℝ := train_speed_mps * time_to_cross_platform_s
noncomputable def length_of_platform_m : ℝ := total_distance_cross_platform_m - length_of_train_m

theorem platform_length_is_260_meters :
  length_of_platform_m = 260 := by
  sorry

end platform_length_is_260_meters_l2132_213204


namespace choose_starting_team_l2132_213290

-- Definitions derived from the conditions
def team_size : ℕ := 18
def selected_goalie (n : ℕ) : ℕ := n
def selected_players (m : ℕ) (k : ℕ) : ℕ := Nat.choose m k

-- The number of ways to choose the starting team
theorem choose_starting_team :
  let n := team_size
  let k := 7
  selected_goalie n * selected_players (n - 1) k = 222768 :=
by
  simp only [team_size, selected_goalie, selected_players]
  sorry

end choose_starting_team_l2132_213290


namespace symmetric_line_equation_l2132_213252

theorem symmetric_line_equation : ∀ (x y : ℝ), (2 * x + 3 * y - 6 = 0) ↔ (3 * (x + 2) + 2 * (-y - 2) + 16 = 0) :=
by
  sorry

end symmetric_line_equation_l2132_213252


namespace min_letters_required_l2132_213236

theorem min_letters_required (n : ℕ) (hn : n = 26) : 
  ∃ k, (∀ (collectors : Fin n) (leader : Fin n), k = 2 * (n - 1)) := 
sorry

end min_letters_required_l2132_213236


namespace find_m_for_parallel_lines_l2132_213208

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 3 * x - y + 2 = 0 → x + m * y - 3 = 0) →
  m = -1 / 3 := sorry

end find_m_for_parallel_lines_l2132_213208


namespace coordinates_of_E_l2132_213210

theorem coordinates_of_E :
  let A := (-2, 1)
  let B := (1, 4)
  let C := (4, -3)
  let ratio_AB := (1, 2)
  let ratio_CE_ED := (1, 4)
  let D := ( (ratio_AB.1 * B.1 + ratio_AB.2 * A.1) / (ratio_AB.1 + ratio_AB.2),
             (ratio_AB.1 * B.2 + ratio_AB.2 * A.2) / (ratio_AB.1 + ratio_AB.2) )
  let E := ( (ratio_CE_ED.1 * C.1 - ratio_CE_ED.2 * D.1) / (ratio_CE_ED.1 - ratio_CE_ED.2),
             (ratio_CE_ED.1 * C.2 - ratio_CE_ED.2 * D.2) / (ratio_CE_ED.1 - ratio_CE_ED.2) )
  E = (-8 / 3, 11 / 3) := by
  sorry

end coordinates_of_E_l2132_213210


namespace solve_equation_l2132_213228

theorem solve_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x * (x - 1)) = 0) ↔ x = 2 :=
by
  sorry

end solve_equation_l2132_213228


namespace find_the_number_l2132_213263

theorem find_the_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_the_number_l2132_213263


namespace fifth_number_in_ninth_row_l2132_213250

theorem fifth_number_in_ninth_row :
  ∃ (n : ℕ), n = 61 ∧ ∀ (i : ℕ), i = 9 → (7 * i - 2 = n) :=
by
  sorry

end fifth_number_in_ninth_row_l2132_213250


namespace find_n_of_geometric_sum_l2132_213200

-- Define the first term and common ratio of the sequence
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Mathematical statement to be proved
theorem find_n_of_geometric_sum (h : S_n 5 = 80 / 243) : ∃ n, S_n n = 80 / 243 ↔ n = 5 :=
by
  sorry

end find_n_of_geometric_sum_l2132_213200


namespace range_frequency_l2132_213254

-- Define the sample data
def sample_data : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the condition representing the frequency count
def frequency_count : ℝ := 0.2 * 20

-- Define the proof problem
theorem range_frequency (s : List ℝ) (range_start range_end : ℝ) : 
  s = sample_data → 
  range_start = 11.5 →
  range_end = 13.5 → 
  (s.filter (λ x => range_start ≤ x ∧ x < range_end)).length = frequency_count := 
by 
  intros
  sorry

end range_frequency_l2132_213254


namespace lindy_total_distance_traveled_l2132_213282

theorem lindy_total_distance_traveled 
    (initial_distance : ℕ)
    (jack_speed : ℕ)
    (christina_speed : ℕ)
    (lindy_speed : ℕ) 
    (meet_time : ℕ)
    (distance : ℕ) :
    initial_distance = 150 →
    jack_speed = 7 →
    christina_speed = 8 →
    lindy_speed = 10 →
    meet_time = initial_distance / (jack_speed + christina_speed) →
    distance = lindy_speed * meet_time →
    distance = 100 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lindy_total_distance_traveled_l2132_213282


namespace find_line_equation_l2132_213294

theorem find_line_equation 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
  (P : ℝ × ℝ) (P_coord : P = (1, 3/2))
  (line_l : ∀ x : ℝ, ℝ)
  (line_eq : ∀ x : ℝ, y = k * x + b) 
  (intersects : ∀ A B : ℝ × ℝ, A ≠ P ∧ B ≠ P)
  (perpendicular : ∀ A B : ℝ × ℝ, (A.1 - 1) * (B.1 - 1) + (A.2 - 3 / 2) * (B.2 - 3 / 2) = 0)
  (bisected_by_y_axis : ∀ A B : ℝ × ℝ, A.1 + B.1 = 0) :
  ∃ k : ℝ, k = 3 / 2 ∨ k = -3 / 2 :=
sorry

end find_line_equation_l2132_213294
