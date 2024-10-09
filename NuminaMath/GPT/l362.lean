import Mathlib

namespace find_k_l362_36297

-- Define the vectors and the condition that k · a + b is perpendicular to a
theorem find_k 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (k : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (-2, 0))
  (h_perpendicular : ∀ (k : ℝ), (k * a.1 + b.1, k * a.2 + b.2) • a = 0 ) : k = 2 / 5 :=
sorry

end find_k_l362_36297


namespace length_of_goods_train_l362_36264

theorem length_of_goods_train (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) (length_of_train : ℝ) :
  speed_kmph = 96 → platform_length = 360 → crossing_time = 32 → length_of_train = (26.67 * 32 - 360) :=
by
  sorry

end length_of_goods_train_l362_36264


namespace white_pairs_coincide_l362_36288

theorem white_pairs_coincide :
  ∀ (red_triangles blue_triangles white_triangles : ℕ)
    (red_pairs blue_pairs red_blue_pairs : ℕ),
  red_triangles = 4 →
  blue_triangles = 4 →
  white_triangles = 6 →
  red_pairs = 3 →
  blue_pairs = 2 →
  red_blue_pairs = 1 →
  (2 * white_triangles - red_triangles - blue_triangles - red_blue_pairs) = white_triangles →
  6 = white_triangles :=
by
  intros red_triangles blue_triangles white_triangles
         red_pairs blue_pairs red_blue_pairs
         H_red H_blue H_white
         H_red_pairs H_blue_pairs H_red_blue_pairs
         H_pairs
  sorry

end white_pairs_coincide_l362_36288


namespace fraction_of_number_l362_36245

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l362_36245


namespace symmetric_point_l362_36205

theorem symmetric_point (x y : ℝ) (a b : ℝ) :
  (x = 3 ∧ y = 9 ∧ a = -1 ∧ b = -3) ∧ (∀ k: ℝ, k ≠ 0 → (y - 9 = k * (x - 3)) ∧ 
  ((x - 3)^2 + (y - 9)^2 = (a - 3)^2 + (b - 9)^2) ∧ 
  (x >= 0 → (a >= 0 ↔ x = 3) ∧ (b >= 0 ↔ y = 9))) :=
by
  sorry

end symmetric_point_l362_36205


namespace n_salary_eq_260_l362_36227

variables (m n : ℕ)
axiom total_salary : m + n = 572
axiom m_salary : m = 120 * n / 100

theorem n_salary_eq_260 : n = 260 :=
by
  sorry

end n_salary_eq_260_l362_36227


namespace remaining_sausage_meat_l362_36270

-- Define the conditions
def total_meat_pounds : ℕ := 10
def sausage_links : ℕ := 40
def links_eaten_by_Brandy : ℕ := 12
def pounds_to_ounces : ℕ := 16

-- Calculate the remaining sausage meat and prove the correctness
theorem remaining_sausage_meat :
  (total_meat_pounds * pounds_to_ounces - links_eaten_by_Brandy * (total_meat_pounds * pounds_to_ounces / sausage_links)) = 112 :=
by
  sorry

end remaining_sausage_meat_l362_36270


namespace expand_expression_l362_36291

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l362_36291


namespace greatest_common_divisor_b_81_l362_36241

theorem greatest_common_divisor_b_81 (a b : ℤ) 
  (h : (1 + Real.sqrt 2) ^ 2012 = a + b * Real.sqrt 2) : Int.gcd b 81 = 3 :=
by
  sorry

end greatest_common_divisor_b_81_l362_36241


namespace min_value_fraction_l362_36256

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b^2 - 4 * a * c ≤ 0) :
  (a + b + c) / (2 * a) ≥ 2 :=
  sorry

end min_value_fraction_l362_36256


namespace set_intersection_M_N_l362_36262

theorem set_intersection_M_N (x : ℝ) :
  let M := {x | -4 < x ∧ x < -2}
  let N := {x | x^2 + 5*x + 6 < 0}
  M ∩ N = {x | -3 < x ∧ x < -2} :=
by
  sorry

end set_intersection_M_N_l362_36262


namespace max_d_for_range_of_fx_l362_36207

theorem max_d_for_range_of_fx : 
  ∀ (d : ℝ), (∃ x : ℝ, x^2 + 4*x + d = -3) → d ≤ 1 := 
by
  sorry

end max_d_for_range_of_fx_l362_36207


namespace solve_system_equations_l362_36267

theorem solve_system_equations (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 + z^2 = b^2) → 
  b = 0 ∧ (∃ t, (x = 0 ∧ y = t ∧ z = -t) ∨ 
                (x = t ∧ y = 0 ∧ z = -t) ∨ 
                (x = -t ∧ y = t ∧ z = 0)) :=
by
  sorry -- Proof to be provided

end solve_system_equations_l362_36267


namespace speed_of_man_in_still_water_l362_36235

theorem speed_of_man_in_still_water
  (V_m V_s : ℝ)
  (cond1 : V_m + V_s = 5)
  (cond2 : V_m - V_s = 7) :
  V_m = 6 :=
by
  sorry

end speed_of_man_in_still_water_l362_36235


namespace ratio_of_blue_to_red_area_l362_36244

theorem ratio_of_blue_to_red_area :
  let r₁ := 1 / 2
  let r₂ := 3 / 2
  let A_red := Real.pi * r₁^2
  let A_large := Real.pi * r₂^2
  let A_blue := A_large - A_red
  A_blue / A_red = 8 :=
by
  sorry

end ratio_of_blue_to_red_area_l362_36244


namespace initial_carrots_count_l362_36210

theorem initial_carrots_count (x : ℕ) (h1 : x - 2 + 21 = 31) : x = 12 := by
  sorry

end initial_carrots_count_l362_36210


namespace glycerin_percentage_l362_36213

theorem glycerin_percentage (x : ℝ) 
  (h1 : 100 * 0.75 = 75)
  (h2 : 75 + 75 = 100)
  (h3 : 75 * 0.30 + (x/100) * 75 = 75) : x = 70 :=
by
  sorry

end glycerin_percentage_l362_36213


namespace function_value_bounds_l362_36230

theorem function_value_bounds (x : ℝ) : 
  (x^2 + x + 1) / (x^2 + 1) ≤ 3 / 2 ∧ (x^2 + x + 1) / (x^2 + 1) ≥ 1 / 2 := 
sorry

end function_value_bounds_l362_36230


namespace monotonic_intervals_logarithmic_inequality_l362_36254

noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, f x > f (x + 1E-9) ∧ f x < f (x - 1E-9)) ∧ 
  (∀ y ∈ Set.Ioi 1, f y < f (y + 1E-9) ∧ f y > f (y - 1E-9)) := sorry

theorem logarithmic_inequality (a : ℝ) (ha : a > 0) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hneq : x1 ≠ x2)
  (h_eq1 : a * x1 + f x1 = x1^2 - x1) (h_eq2 : a * x2 + f x2 = x2^2 - x2) :
  Real.log x1 + Real.log x2 + 2 * Real.log a < 0 := sorry

end monotonic_intervals_logarithmic_inequality_l362_36254


namespace negation_of_p_l362_36268

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A (x : ℤ) : Prop := is_odd x
def B (x : ℤ) : Prop := is_even x
def p : Prop := ∀ x, A x → B (2 * x)

theorem negation_of_p : ¬ p ↔ ∃ x, A x ∧ ¬ B (2 * x) :=
by
  -- problem statement equivalent in Lean 4
  sorry

end negation_of_p_l362_36268


namespace prove_ab_eq_neg_26_l362_36252

theorem prove_ab_eq_neg_26
  (a b : ℚ)
  (H : ∀ k : ℚ, ∃ x : ℚ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6) :
  a * b = -26 := sorry

end prove_ab_eq_neg_26_l362_36252


namespace hypotenuse_45_45_90_l362_36258

theorem hypotenuse_45_45_90 (leg : ℝ) (h_leg : leg = 10) (angle : ℝ) (h_angle : angle = 45) :
  ∃ hypotenuse : ℝ, hypotenuse = leg * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end hypotenuse_45_45_90_l362_36258


namespace lcm_45_75_l362_36238

theorem lcm_45_75 : Nat.lcm 45 75 = 225 :=
by
  sorry

end lcm_45_75_l362_36238


namespace cube_divided_by_five_tetrahedrons_l362_36250

-- Define the minimum number of tetrahedrons needed to divide a cube
def min_tetrahedrons_to_divide_cube : ℕ := 5

-- State the theorem
theorem cube_divided_by_five_tetrahedrons : min_tetrahedrons_to_divide_cube = 5 :=
by
  -- The proof is skipped, as instructed
  sorry

end cube_divided_by_five_tetrahedrons_l362_36250


namespace average_marks_correct_l362_36218

-- Define constants for the marks in each subject
def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the total number of subjects
def num_subjects : ℕ := 5

-- Define the total marks as the sum of individual subjects
def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is as expected
theorem average_marks_correct : average_marks = 75 :=
by {
  -- skip the proof
  sorry
}

end average_marks_correct_l362_36218


namespace solution_problem_l362_36265

noncomputable def proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : Prop :=
  (-1 < (x - y)) ∧ ((x - y) < 1) ∧ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 1) → (min ((1/x) + (x/y)) = 3))

theorem solution_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  proof_problem x y hx hy h := 
sorry

end solution_problem_l362_36265


namespace initial_nickels_eq_l362_36249

variable (quarters : ℕ) (initial_nickels : ℕ) (nickels_borrowed : ℕ) (nickels_left : ℕ)

-- Assumptions based on the problem
axiom quarters_had : quarters = 33
axiom nickels_left_axiom : nickels_left = 12
axiom nickels_borrowed_axiom : nickels_borrowed = 75

-- Theorem to prove: initial number of nickels
theorem initial_nickels_eq :
  initial_nickels = nickels_left + nickels_borrowed :=
by
  sorry

end initial_nickels_eq_l362_36249


namespace roots_of_quadratic_are_integers_l362_36257

theorem roots_of_quadratic_are_integers
  (b c : ℤ)
  (Δ : ℤ)
  (h_discriminant: Δ = b^2 - 4 * c)
  (h_perfect_square: ∃ k : ℤ, k^2 = Δ)
  : (∃ x1 x2 : ℤ, x1 * x2 = c ∧ x1 + x2 = -b) :=
by
  sorry

end roots_of_quadratic_are_integers_l362_36257


namespace find_value_of_a_l362_36278

theorem find_value_of_a (a : ℝ) (h: (1 + 3 + 2 + 5 + a) / 5 = 3) : a = 4 :=
by
  sorry

end find_value_of_a_l362_36278


namespace mean_score_of_sophomores_l362_36273

open Nat

variable (s j : ℕ)
variable (m m_s m_j : ℝ)

theorem mean_score_of_sophomores :
  (s + j = 150) →
  (m = 85) →
  (j = 80 / 100 * s) →
  (m_s = 125 / 100 * m_j) →
  (s * m_s + j * m_j = 12750) →
  m_s = 94 := by intros; sorry

end mean_score_of_sophomores_l362_36273


namespace remainder_add_l362_36266

theorem remainder_add (a b : ℤ) (n m : ℤ) 
  (ha : a = 60 * n + 41) 
  (hb : b = 45 * m + 14) : 
  (a + b) % 15 = 10 := by 
  sorry

end remainder_add_l362_36266


namespace ab_minus_c_eq_six_l362_36216

theorem ab_minus_c_eq_six (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : 
  a * b - c = 6 := 
by
  sorry

end ab_minus_c_eq_six_l362_36216


namespace solve_equation1_solve_equation2_l362_36243

theorem solve_equation1 (x : ℝ) : 4 - x = 3 * (2 - x) ↔ x = 1 :=
by sorry

theorem solve_equation2 (x : ℝ) : (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1 ↔ x = -3 / 2 :=
by sorry

end solve_equation1_solve_equation2_l362_36243


namespace comparing_exponents_l362_36202

theorem comparing_exponents {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end comparing_exponents_l362_36202


namespace total_wage_calculation_l362_36206

def basic_pay_rate : ℝ := 20
def weekly_hours : ℝ := 40
def overtime_rate : ℝ := basic_pay_rate * 1.25
def total_hours_worked : ℝ := 48
def overtime_hours : ℝ := total_hours_worked - weekly_hours

theorem total_wage_calculation : 
  (weekly_hours * basic_pay_rate) + (overtime_hours * overtime_rate) = 1000 :=
by
  sorry

end total_wage_calculation_l362_36206


namespace graph_intersection_points_l362_36223

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end graph_intersection_points_l362_36223


namespace calculate_x_times_a_l362_36263

-- Define variables and assumptions
variables (a b x y : ℕ)
variable (hb : b = 4)
variable (hy : y = 2)
variable (h1 : a = 2 * b)
variable (h2 : x = 3 * y)
variable (h3 : a + b = x * y)

-- The statement to be proved
theorem calculate_x_times_a : x * a = 48 :=
by sorry

end calculate_x_times_a_l362_36263


namespace balanced_apple_trees_l362_36282

theorem balanced_apple_trees: 
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1 * y2 - x1 * y4 - x3 * y2 + x3 * y4 = 0) ∧
    (x2 * y1 - x2 * y3 - x4 * y1 + x4 * y3 = 0) ∧
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4) :=
  sorry

end balanced_apple_trees_l362_36282


namespace major_airlines_free_snacks_l362_36220

variable (S : ℝ)

theorem major_airlines_free_snacks (h1 : 0.5 ≤ 1) (h2 : 0.5 = 1) :
  0.5 ≤ S :=
sorry

end major_airlines_free_snacks_l362_36220


namespace three_students_two_groups_l362_36292

theorem three_students_two_groups : 
  (2 : ℕ) ^ 3 = 8 := 
by
  sorry

end three_students_two_groups_l362_36292


namespace length_of_chord_MN_l362_36290

theorem length_of_chord_MN 
  (m n : ℝ)
  (h1 : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * M.1 + M.2 * M.2 + m * M.1 + n * M.2 - 4 = 0 ∧ N.1 * N.1 + N.2 * N.2 + m * N.1 + n * N.2 - 4 = 0 
    ∧ N.2 = M.1 ∧ N.1 = M.2) 
  (h2 : x + y = 0)
  : length_of_chord = 4 := sorry

end length_of_chord_MN_l362_36290


namespace acute_triangle_exterior_angles_obtuse_l362_36200

theorem acute_triangle_exterior_angles_obtuse
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  ∀ α β γ, α = A + B → β = B + C → γ = C + A → α > π / 2 ∧ β > π / 2 ∧ γ > π / 2 :=
by
  sorry

end acute_triangle_exterior_angles_obtuse_l362_36200


namespace equal_real_roots_l362_36248

theorem equal_real_roots (m : ℝ) : (∃ x : ℝ, x * x - 4 * x - m = 0) → (16 + 4 * m = 0) → m = -4 :=
by
  sorry

end equal_real_roots_l362_36248


namespace ellipse_range_x_plus_y_l362_36237

/-- The problem conditions:
Given any point P(x, y) on the ellipse x^2 / 144 + y^2 / 25 = 1,
prove that the range of values for x + y is [-13, 13].
-/
theorem ellipse_range_x_plus_y (x y : ℝ) (h : (x^2 / 144) + (y^2 / 25) = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := sorry

end ellipse_range_x_plus_y_l362_36237


namespace hat_price_reduction_l362_36211

theorem hat_price_reduction (original_price : ℚ) (r1 r2 : ℚ) (price_after_reductions : ℚ) :
  original_price = 12 → r1 = 0.20 → r2 = 0.25 →
  price_after_reductions = original_price * (1 - r1) * (1 - r2) →
  price_after_reductions = 7.20 :=
by
  intros original_price_eq r1_eq r2_eq price_calc_eq
  sorry

end hat_price_reduction_l362_36211


namespace sofia_total_time_l362_36204

def distance1 : ℕ := 150
def speed1 : ℕ := 5
def distance2 : ℕ := 150
def speed2 : ℕ := 6
def laps : ℕ := 8
def time_per_lap := (distance1 / speed1) + (distance2 / speed2)
def total_time := 440  -- 7 minutes and 20 seconds in seconds

theorem sofia_total_time :
  laps * time_per_lap = total_time :=
by
  -- Proof steps are omitted and represented by sorry.
  sorry

end sofia_total_time_l362_36204


namespace find_A_B_l362_36219

theorem find_A_B (A B : ℝ) (h : ∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 → 
  (A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3 * x - 10))) :
  A = 3 ∧ B = 2 :=
sorry

end find_A_B_l362_36219


namespace number_properties_l362_36286

theorem number_properties : 
    ∃ (N : ℕ), 
    35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 :=
by 
  sorry

end number_properties_l362_36286


namespace train_length_l362_36277

theorem train_length (L : ℕ) (speed : ℕ) 
  (h1 : L + 1200 = speed * 45) 
  (h2 : L + 180 = speed * 15) : 
  L = 330 := 
sorry

end train_length_l362_36277


namespace calf_rope_length_l362_36212

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * initial_length ^ 2
  let A2 := A1 + additional_area
  let new_length_squared := A2 / Real.pi
  Real.sqrt new_length_squared

theorem calf_rope_length :
  new_rope_length 12 565.7142857142857 = 18 := by
  sorry

end calf_rope_length_l362_36212


namespace renovate_total_time_eq_79_5_l362_36236

-- Definitions based on the given conditions
def time_per_bedroom : ℝ := 4
def num_bedrooms : ℕ := 3
def time_per_kitchen : ℝ := time_per_bedroom * 1.5
def time_per_garden : ℝ := 3
def time_per_terrace : ℝ := time_per_garden - 2
def time_per_basement : ℝ := time_per_kitchen * 0.75

-- Total time excluding the living room
def total_time_excl_living_room : ℝ :=
  (num_bedrooms * time_per_bedroom) +
  time_per_kitchen +
  time_per_garden +
  time_per_terrace +
  time_per_basement

-- Time for the living room
def time_per_living_room : ℝ := 2 * total_time_excl_living_room

-- Total time for everything
def total_time : ℝ := total_time_excl_living_room + time_per_living_room

-- The theorem we need to prove
theorem renovate_total_time_eq_79_5 : total_time = 79.5 := by
  sorry

end renovate_total_time_eq_79_5_l362_36236


namespace inequality_ineq_l362_36214

theorem inequality_ineq (x y : ℝ) (hx: x > Real.sqrt 2) (hy: y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
  sorry

end inequality_ineq_l362_36214


namespace plums_added_l362_36269

-- Definitions of initial and final plum counts
def initial_plums : ℕ := 17
def final_plums : ℕ := 21

-- The mathematical statement to be proved
theorem plums_added (initial_plums final_plums : ℕ) : final_plums - initial_plums = 4 := by
  -- The proof will be inserted here
  sorry

end plums_added_l362_36269


namespace sum_of_coefficients_l362_36233

noncomputable def polynomial_eq (x : ℝ) : ℝ := 1 + x^5
noncomputable def linear_combination (a0 a1 a2 a3 a4 a5 x : ℝ) : ℝ :=
  a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5

theorem sum_of_coefficients (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial_eq 1 = linear_combination a0 a1 a2 a3 a4 a5 1 →
  polynomial_eq 2 = linear_combination a0 a1 a2 a3 a4 a5 2 →
  a0 = 2 →
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  intros h1 h2 h3
  sorry

end sum_of_coefficients_l362_36233


namespace at_least_two_equal_l362_36281

theorem at_least_two_equal (x y z : ℝ) (h : x / y + y / z + z / x = z / y + y / x + x / z) : 
  x = y ∨ y = z ∨ z = x := 
  sorry

end at_least_two_equal_l362_36281


namespace true_discount_different_time_l362_36240

theorem true_discount_different_time (FV TD_initial TD_different : ℝ) (r : ℝ) (initial_time different_time : ℝ) 
  (h1 : r = initial_time / different_time)
  (h2 : FV = 110)
  (h3 : TD_initial = 10)
  (h4 : initial_time / different_time = 1 / 2) :
  TD_different = 2 * TD_initial :=
by
  sorry

end true_discount_different_time_l362_36240


namespace minimum_rows_l362_36255

theorem minimum_rows (n : ℕ) (C : ℕ → ℕ) (hC_bounds : ∀ i, 1 ≤ C i ∧ C i ≤ 39) 
  (hC_sum : (Finset.range n).sum C = 1990) :
  ∃ k, k = 12 ∧ ∀ (R : ℕ) (hR : R = 199), 
    ∀ (seating : ℕ → ℕ) (h_seating : ∀ i, seating i ≤ R) 
    (h_seating_capacity : (Finset.range k).sum seating = 1990),
    True := sorry

end minimum_rows_l362_36255


namespace problem1_problem2_l362_36274

theorem problem1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 :=
  sorry

theorem problem2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 :=
  sorry

end problem1_problem2_l362_36274


namespace ratio_ad_bc_l362_36232

theorem ratio_ad_bc (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 5 * c) (h3 : c = 3 * d) : 
  (a * d) / (b * c) = 4 / 3 := 
by 
  sorry

end ratio_ad_bc_l362_36232


namespace elevenRowTriangleTotalPieces_l362_36272

-- Definitions and problem statement
def numRodsInRow (n : ℕ) : ℕ := 3 * n

def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

def totalRods (rows : ℕ) : ℕ := 3 * (sumFirstN rows)

def totalConnectors (rows : ℕ) : ℕ := sumFirstN (rows + 1)

def totalPieces (rows : ℕ) : ℕ := totalRods rows + totalConnectors rows

-- Lean proof problem
theorem elevenRowTriangleTotalPieces : totalPieces 11 = 276 := 
by
  sorry

end elevenRowTriangleTotalPieces_l362_36272


namespace product_of_functions_l362_36225

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_of_functions (x : ℝ) (h : x ≠ 0) : f x * g x = -6 * x + 2 := by
  sorry

end product_of_functions_l362_36225


namespace number_of_programs_correct_l362_36287

-- Conditions definition
def solo_segments := 5
def chorus_segments := 3

noncomputable def number_of_programs : ℕ :=
  let solo_permutations := Nat.factorial solo_segments
  let available_spaces := solo_segments + 1
  let chorus_placements := Nat.choose (available_spaces - 1) chorus_segments
  solo_permutations * chorus_placements

theorem number_of_programs_correct : number_of_programs = 7200 :=
  by
    -- The proof is omitted
    sorry

end number_of_programs_correct_l362_36287


namespace sum_of_factors_l362_36234

-- Define 36
def num : ℕ := 36

-- Prime factorization of 36
def prime_factorization : (ℕ × ℕ) := (2, 2) -- represents 2^2
def prime_factorization_2 : (ℕ × ℕ) := (3, 2) -- represents 3^2

-- Sum of divisors function formula
def sigma (n : ℕ) : ℕ := 
  (1 + 2 + 2^2) * (1 + 3 + 3^2)

-- Theorem: sum of the positive factors of 36 is 91
theorem sum_of_factors : sigma num = 91 := 
by {
  -- Skipping the proof steps
  sorry
}

end sum_of_factors_l362_36234


namespace no_such_function_exists_l362_36261

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) :=
by
  sorry

end no_such_function_exists_l362_36261


namespace area_of_rectangle_ABCD_l362_36239

theorem area_of_rectangle_ABCD :
  ∀ (short_side long_side width length : ℝ),
    (short_side = 6) →
    (long_side = 6 * (3 / 2)) →
    (width = 2 * short_side) →
    (length = long_side) →
    (width * length = 108) :=
by
  intros short_side long_side width length h_short h_long h_width h_length
  rw [h_short, h_long] at *
  sorry

end area_of_rectangle_ABCD_l362_36239


namespace compound_interest_l362_36247

theorem compound_interest (P R T : ℝ) (SI CI : ℝ)
  (hSI : SI = P * R * T / 100)
  (h_given_SI : SI = 50)
  (h_given_R : R = 5)
  (h_given_T : T = 2)
  (h_compound_interest : CI = P * ((1 + R / 100)^T - 1)) :
  CI = 51.25 :=
by
  -- Since we are only required to state the theorem, we add 'sorry' here.
  sorry

end compound_interest_l362_36247


namespace susan_typing_time_l362_36279

theorem susan_typing_time :
  let Jonathan_rate := 1 -- page per minute
  let Jack_rate := 5 / 3 -- pages per minute
  let combined_rate := 4 -- pages per minute
  ∃ S : ℝ, (1 + 1/S + 5/3 = 4) → S = 30 :=
by
  sorry

end susan_typing_time_l362_36279


namespace algebraic_expression_evaluation_l362_36231

theorem algebraic_expression_evaluation (m : ℝ) (h : m^2 - m - 3 = 0) : m^2 - m - 2 = 1 := 
by
  sorry

end algebraic_expression_evaluation_l362_36231


namespace maria_needs_nuts_l362_36284

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end maria_needs_nuts_l362_36284


namespace trains_distance_l362_36298

theorem trains_distance (t x : ℝ) 
  (h1 : x = 20 * t)
  (h2 : x + 50 = 25 * t) : 
  x + (x + 50) = 450 := 
by 
  -- placeholder for the proof
  sorry

end trains_distance_l362_36298


namespace shifted_quadratic_eq_l362_36209

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end shifted_quadratic_eq_l362_36209


namespace Agnes_birth_year_l362_36203

theorem Agnes_birth_year (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : (11 * x + 2 * y + x * y = 92)) : 1948 = 1900 + (10 * x + y) :=
sorry

end Agnes_birth_year_l362_36203


namespace prime_factors_count_900_l362_36246

theorem prime_factors_count_900 : 
  ∃ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x ∧ x ∣ 900) ∧ S.card = 3 :=
by 
  sorry

end prime_factors_count_900_l362_36246


namespace proposition_judgement_l362_36217

theorem proposition_judgement (p q : Prop) (a b c x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (¬ (a > b → a * c^2 > b * c^2)) ∧
  (¬ (∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  ((x^2 - 3*x + 2 = 0) → (x = 2)) =
  false := sorry

end proposition_judgement_l362_36217


namespace mike_pens_l362_36242

-- Definitions based on the conditions
def initial_pens : ℕ := 25
def pens_after_mike (M : ℕ) : ℕ := initial_pens + M
def pens_after_cindy (M : ℕ) : ℕ := 2 * pens_after_mike M
def pens_after_sharon (M : ℕ) : ℕ := pens_after_cindy M - 19
def final_pens : ℕ := 75

-- The theorem we need to prove
theorem mike_pens (M : ℕ) (h : pens_after_sharon M = final_pens) : M = 22 := by
  have h1 : pens_after_sharon M = 2 * (25 + M) - 19 := rfl
  rw [h1] at h
  sorry

end mike_pens_l362_36242


namespace altitude_eqn_equidistant_eqn_l362_36271

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definition of a line in the form Ax + By + C = 0
structure Line :=
  (A B C : ℝ)
  (non_zero : A ≠ 0 ∨ B ≠ 0)

-- Equation of line l1 (altitude to side BC)
def l1 : Line := { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) }

-- Equation of line l2 (passing through C, equidistant from A and B), two possible values
def l2a : Line := { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) }
def l2b : Line := { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) }

-- Prove the equations for l1 and l2 are correct given the points A, B, and C
theorem altitude_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l1 = { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) } := sorry

theorem equidistant_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l2a = { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) } ∨
  l2b = { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) } := sorry

end altitude_eqn_equidistant_eqn_l362_36271


namespace carbonate_ions_in_Al2_CO3_3_l362_36276

theorem carbonate_ions_in_Al2_CO3_3 (total_weight : ℕ) (formula : String) 
  (molecular_weight : ℕ) (ions_in_formula : String) : 
  formula = "Al2(CO3)3" → molecular_weight = 234 → ions_in_formula = "CO3" → total_weight = 3 := 
by
  intros formula_eq weight_eq ions_eq
  sorry

end carbonate_ions_in_Al2_CO3_3_l362_36276


namespace percent_of_1600_l362_36260

theorem percent_of_1600 (x : ℝ) (h1 : 0.25 * 1600 = 400) (h2 : x / 100 * 400 = 20) : x = 5 :=
sorry

end percent_of_1600_l362_36260


namespace sector_area_eq_13pi_l362_36208

theorem sector_area_eq_13pi
    (O A B C : Type)
    (r : ℝ)
    (θ : ℝ)
    (h1 : θ = 130)
    (h2 : r = 6) :
    (θ / 360) * (π * r^2) = 13 * π := by
  sorry

end sector_area_eq_13pi_l362_36208


namespace rounding_problem_l362_36259

def given_number : ℝ := 3967149.487234

theorem rounding_problem : (3967149.487234).round = 3967149 := sorry

end rounding_problem_l362_36259


namespace required_sampling_methods_l362_36215

-- Defining the given conditions
def total_households : Nat := 2000
def farmer_households : Nat := 1800
def worker_households : Nat := 100
def intellectual_households : Nat := total_households - farmer_households - worker_households
def sample_size : Nat := 40

-- Statement representing the proof problem
theorem required_sampling_methods :
  stratified_sampling_needed ∧ systematic_sampling_needed ∧ simple_random_sampling_needed :=
sorry

end required_sampling_methods_l362_36215


namespace roller_coaster_cars_l362_36228

theorem roller_coaster_cars (n : ℕ) (h : ((n - 1) : ℝ) / n = 0.5) : n = 2 :=
sorry

end roller_coaster_cars_l362_36228


namespace smaller_angle_measure_l362_36295

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l362_36295


namespace part1_solution_set_k_3_part2_solution_set_k_lt_0_l362_36299

open Set

-- Definitions
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Part 1: When k = 3
theorem part1_solution_set_k_3 : ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < (2 / 3) :=
by
  sorry

-- Part 2: When k < 0
theorem part2_solution_set_k_lt_0 :
  ∀ k : ℝ, k < 0 → 
    (k = -2 → ∀ x : ℝ, inequality k x ↔ x ≠ -1) ∧
    (k < -2 → ∀ x : ℝ, inequality k x ↔ x < -1 ∨ x > 2 / k) ∧
    (-2 < k → ∀ x : ℝ, inequality k x ↔ x > -1 ∨ x < 2 / k) :=
by
  sorry

end part1_solution_set_k_3_part2_solution_set_k_lt_0_l362_36299


namespace probability_m_eq_kn_l362_36224

/- 
Define the conditions and question in Lean 4 -/
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_rolls : Finset (ℕ × ℕ) := Finset.product die_faces die_faces

def events_satisfying_condition : Finset (ℕ × ℕ) :=
  {(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), 
   (5, 1), (5, 5), (6, 1), (6, 2), (6, 3), (6, 6)}

theorem probability_m_eq_kn (k : ℕ) (h : k > 0) :
  (events_satisfying_condition.card : ℚ) / (valid_rolls.card : ℚ) = 7/18 := by
  sorry

end probability_m_eq_kn_l362_36224


namespace find_number_l362_36226

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end find_number_l362_36226


namespace sum_of_two_numbers_l362_36221

theorem sum_of_two_numbers :
  ∃ x y : ℝ, (x * y = 9375 ∧ y / x = 15) ∧ (x + y = 400) :=
by
  sorry

end sum_of_two_numbers_l362_36221


namespace intersection_of_sets_l362_36285

def M : Set ℝ := { x | 3 * x - 6 ≥ 0 }
def N : Set ℝ := { x | x^2 < 16 }

theorem intersection_of_sets : M ∩ N = { x | 2 ≤ x ∧ x < 4 } :=
by {
  sorry
}

end intersection_of_sets_l362_36285


namespace Laura_more_than_200_paperclips_on_Friday_l362_36275

theorem Laura_more_than_200_paperclips_on_Friday:
  ∀ (n : ℕ), (n = 4 ∨ n = 0 ∨ n ≥ 1 ∧ (n - 1 = 0 ∨ n = 1) → 4 * 3 ^ n > 200) :=
by
  sorry

end Laura_more_than_200_paperclips_on_Friday_l362_36275


namespace jars_of_peanut_butter_l362_36296

theorem jars_of_peanut_butter (x : Nat) : 
  (16 * x + 28 * x + 40 * x + 52 * x = 2032) → 
  (4 * x = 60) :=
by
  intro h
  sorry

end jars_of_peanut_butter_l362_36296


namespace gcd_of_462_and_330_l362_36294

theorem gcd_of_462_and_330 :
  Nat.gcd 462 330 = 66 :=
sorry

end gcd_of_462_and_330_l362_36294


namespace original_profit_percentage_l362_36253

theorem original_profit_percentage
  (P SP : ℝ)
  (h1 : SP = 549.9999999999995)
  (h2 : SP = P * (1 + x / 100))
  (h3 : 0.9 * P * 1.3 = SP + 35) :
  x = 10 := 
sorry

end original_profit_percentage_l362_36253


namespace pow_div_eq_l362_36283

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l362_36283


namespace shadow_of_tree_l362_36289

open Real

theorem shadow_of_tree (height_tree height_pole shadow_pole shadow_tree : ℝ) 
(h1 : height_tree = 12) (h2 : height_pole = 150) (h3 : shadow_pole = 100) 
(h4 : height_tree / shadow_tree = height_pole / shadow_pole) : shadow_tree = 8 := 
by 
  -- Proof will go here
  sorry

end shadow_of_tree_l362_36289


namespace right_angled_triangle_ratio_3_4_5_l362_36229

theorem right_angled_triangle_ratio_3_4_5 : 
  ∀ (a b c : ℕ), 
  (a = 3 * d) → (b = 4 * d) → (c = 5 * d) → (a^2 + b^2 = c^2) :=
by
  intros a b c h1 h2 h3
  sorry

end right_angled_triangle_ratio_3_4_5_l362_36229


namespace sum_of_ages_l362_36280

theorem sum_of_ages (a b c d : ℕ) (h1 : a * b = 20) (h2 : c * d = 28) (distinct : ∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) : a + b + c + d = 19 :=
sorry

end sum_of_ages_l362_36280


namespace complex_value_l362_36222

open Complex

theorem complex_value (z : ℂ)
  (h : 15 * normSq z = 3 * normSq (z + 3) + normSq (z^2 + 4) + 25) :
  z + (8 / z) = -4 :=
sorry

end complex_value_l362_36222


namespace smallest_n_divisible_by_125000_l362_36293

noncomputable def geometric_term_at (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

noncomputable def first_term : ℚ := 5 / 8
noncomputable def second_term : ℚ := 25
noncomputable def common_ratio : ℚ := second_term / first_term

theorem smallest_n_divisible_by_125000 :
  ∃ n : ℕ, n ≥ 7 ∧ geometric_term_at first_term common_ratio n % 125000 = 0 :=
by
  sorry

end smallest_n_divisible_by_125000_l362_36293


namespace intersection_eq_l362_36251

-- Definitions for M and N
def M : Set ℤ := Set.univ
def N : Set ℤ := {x : ℤ | x^2 - x - 2 < 0}

-- The theorem to be proved
theorem intersection_eq : M ∩ N = {0, 1} := 
  sorry

end intersection_eq_l362_36251


namespace brother_highlighters_spent_l362_36201

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l362_36201
