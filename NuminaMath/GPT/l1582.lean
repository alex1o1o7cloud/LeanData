import Mathlib

namespace eggs_per_group_l1582_158224

-- Conditions
def total_eggs : ℕ := 9
def total_groups : ℕ := 3

-- Theorem statement
theorem eggs_per_group : total_eggs / total_groups = 3 :=
sorry

end eggs_per_group_l1582_158224


namespace sum_of_products_l1582_158289

theorem sum_of_products : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end sum_of_products_l1582_158289


namespace horse_rent_problem_l1582_158285

theorem horse_rent_problem (total_rent : ℝ) (b_payment : ℝ) (a_horses b_horses c_horses : ℝ) 
  (a_months b_months c_months : ℝ) (h_total_rent : total_rent = 870) (h_b_payment : b_payment = 360)
  (h_a_horses : a_horses = 12) (h_b_horses : b_horses = 16) (h_c_horses : c_horses = 18) 
  (h_b_months : b_months = 9) (h_c_months : c_months = 6) : 
  ∃ (a_months : ℝ), (a_horses * a_months * 2.5 + b_payment + c_horses * c_months * 2.5 = total_rent) :=
by
  use 8
  sorry

end horse_rent_problem_l1582_158285


namespace tan_angle_addition_l1582_158213

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l1582_158213


namespace divide_pile_l1582_158203

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l1582_158203


namespace consecutive_integer_cubes_sum_l1582_158287

theorem consecutive_integer_cubes_sum : 
  ∀ (a : ℕ), 
  (a > 2) → 
  (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2)) →
  ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3) = 224 :=
by
  intro a ha h
  sorry

end consecutive_integer_cubes_sum_l1582_158287


namespace compute_expression_l1582_158267

noncomputable def a : ℝ := 125^(1/3)
noncomputable def b : ℝ := (-2/3)^0
noncomputable def c : ℝ := Real.log 8 / Real.log 2

theorem compute_expression : a - b - c = 1 := by
  sorry

end compute_expression_l1582_158267


namespace question1_question2_l1582_158218

noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
noncomputable def f2 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2 - 2*x

theorem question1 (a : ℝ) : 
  (∀ x : ℝ, f1 a x = 0 → ∀ y : ℝ, f1 a y = 0 → x = y) ↔ (a = 0 ∨ a < -4 / Real.exp 2) :=
sorry -- Proof of theorem 1

theorem question2 (a m n x0 : ℝ) (h : a ≠ 0) :
  (f2 a x0 = f2 a ((x0 + m) / 2) * (x0 - m) + n ∧ x0 ≠ m) → False :=
sorry -- Proof of theorem 2

end question1_question2_l1582_158218


namespace rowing_upstream_speed_l1582_158271

theorem rowing_upstream_speed (V_m V_down V_up V_s : ℝ) 
  (hVm : V_m = 40) 
  (hVdown : V_down = 60) 
  (hVdown_eq : V_down = V_m + V_s) 
  (hVup_eq : V_up = V_m - V_s) : 
  V_up = 20 := 
by
  sorry

end rowing_upstream_speed_l1582_158271


namespace distance_from_M0_to_plane_is_sqrt77_l1582_158291

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M1 : Point3D := ⟨1, 0, 2⟩
def M2 : Point3D := ⟨1, 2, -1⟩
def M3 : Point3D := ⟨2, -2, 1⟩
def M0 : Point3D := ⟨-5, -9, 1⟩

noncomputable def distance_to_plane (P : Point3D) (A B C : Point3D) : ℝ := sorry

theorem distance_from_M0_to_plane_is_sqrt77 : 
  distance_to_plane M0 M1 M2 M3 = Real.sqrt 77 := sorry

end distance_from_M0_to_plane_is_sqrt77_l1582_158291


namespace complex_number_problem_l1582_158292

variables {a b c x y z : ℂ}

theorem complex_number_problem (h1 : a = (b + c) / (x - 2))
    (h2 : b = (c + a) / (y - 2))
    (h3 : c = (a + b) / (z - 2))
    (h4 : x * y + y * z + z * x = 67)
    (h5 : x + y + z = 2010) :
    x * y * z = -5892 :=
sorry

end complex_number_problem_l1582_158292


namespace part1_part2_l1582_158216

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 := sorry

theorem part2 (a : ℝ) (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  (f x₁ a - f x₂ a) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) := sorry

end part1_part2_l1582_158216


namespace problem_b_problem_c_problem_d_l1582_158255

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end problem_b_problem_c_problem_d_l1582_158255


namespace compare_squares_l1582_158232

theorem compare_squares (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ (a + b) / 2 * (a + b) / 2 := 
sorry

end compare_squares_l1582_158232


namespace average_speed_of_train_l1582_158222

theorem average_speed_of_train (x : ℝ) (h1 : x > 0): 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 :=
by
  sorry

end average_speed_of_train_l1582_158222


namespace minimum_distance_square_l1582_158282

/-- Given the equation of a circle centered at (2,3) with radius 1, find the minimum value of 
the function z = x^2 + y^2 -/
theorem minimum_distance_square (x y : ℝ) 
  (h : (x - 2)^2 + (y - 3)^2 = 1) : ∃ (z : ℝ), z = x^2 + y^2 ∧ z = 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_distance_square_l1582_158282


namespace positive_difference_of_squares_l1582_158294

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l1582_158294


namespace jo_integer_max_l1582_158235
noncomputable def jo_integer : Nat :=
  let n := 166
  n

theorem jo_integer_max (n : Nat) (h1 : n < 200) (h2 : ∃ k : Nat, n + 2 = 9 * k) (h3 : ∃ l : Nat, n + 4 = 10 * l) : n ≤ jo_integer := 
by
  unfold jo_integer
  sorry

end jo_integer_max_l1582_158235


namespace solve_quadratic_inequality_l1582_158274

theorem solve_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a * x ^ 2 - (2 * a + 1) * x + 2 > 0 ↔
    if a = 0 then
      x < 2
    else if a > 0 then
      if a >= 1 / 2 then
        x < 1 / a ∨ x > 2
      else
        x < 2 ∨ x > 1 / a
    else
      x > 1 / a ∧ x < 2)) :=
sorry

end solve_quadratic_inequality_l1582_158274


namespace scientific_notation_63000_l1582_158211

theorem scientific_notation_63000 : 63000 = 6.3 * 10^4 :=
by
  sorry

end scientific_notation_63000_l1582_158211


namespace max_slope_of_circle_l1582_158246

theorem max_slope_of_circle (x y : ℝ) 
  (h : x^2 + y^2 - 6 * x - 6 * y + 12 = 0) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ ∀ k' : ℝ, (x = 0 → k' = 0) ∧ (x ≠ 0 → y = k' * x → k' ≤ k) :=
sorry

end max_slope_of_circle_l1582_158246


namespace abc_inequality_l1582_158276

theorem abc_inequality (a b c : ℝ) : a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l1582_158276


namespace common_difference_arithmetic_sequence_l1582_158262

theorem common_difference_arithmetic_sequence (a b : ℝ) :
  ∃ d : ℝ, b = a + 6 * d ∧ d = (b - a) / 6 :=
by
  sorry

end common_difference_arithmetic_sequence_l1582_158262


namespace greatest_third_side_l1582_158256

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l1582_158256


namespace theater_revenue_l1582_158260

theorem theater_revenue 
  (seats : ℕ)
  (capacity_percentage : ℝ)
  (ticket_price : ℝ)
  (days : ℕ)
  (H1 : seats = 400)
  (H2 : capacity_percentage = 0.8)
  (H3 : ticket_price = 30)
  (H4 : days = 3)
  : (seats * capacity_percentage * ticket_price * days = 28800) :=
by
  sorry

end theater_revenue_l1582_158260


namespace problem_part1_problem_part2_problem_part3_l1582_158210

variable (a b x : ℝ) (p q : ℝ) (n x1 x2 : ℝ)
variable (h1 : x1 = -2) (h2 : x2 = 3)
variable (h3 : x1 < x2)

def equation1 := x + p / x = q
def solution1_p := p = -6
def solution1_q := q = 1

def equation2 := x + 7 / x = 8
def solution2 := x1 = 7

def equation3 := 2 * x + (n^2 - n) / (2 * x - 1) = 2 * n
def solution3 := (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)

theorem problem_part1 : ∀ (x : ℝ), (x + -6 / x = 1) → (p = -6 ∧ q = 1) := by
  sorry

theorem problem_part2 : (max 7 1 = 7) := by
  sorry

theorem problem_part3 : ∀ (n : ℝ), (∃ x1 x2, x1 < x2 ∧ (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)) := by
  sorry

end problem_part1_problem_part2_problem_part3_l1582_158210


namespace height_difference_l1582_158228

variable {J L R : ℕ}

theorem height_difference
  (h1 : J = L + 15)
  (h2 : J = 152)
  (h3 : L + R = 295) :
  R - J = 6 :=
sorry

end height_difference_l1582_158228


namespace right_triangle_acute_angle_le_45_l1582_158240

theorem right_triangle_acute_angle_le_45
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hright : a^2 + b^2 = c^2):
  ∃ θ φ : ℝ, θ + φ = 90 ∧ (θ ≤ 45 ∨ φ ≤ 45) :=
by
  sorry

end right_triangle_acute_angle_le_45_l1582_158240


namespace find_a_l1582_158221

variable (x y a : ℝ)

theorem find_a (h1 : (a * x + 8 * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : a = 7 :=
sorry

end find_a_l1582_158221


namespace diving_assessment_l1582_158279

theorem diving_assessment (total_athletes : ℕ) (selected_athletes : ℕ) (not_meeting_standard : ℕ) 
  (first_level_sample : ℕ) (first_level_total : ℕ) (athletes : Set ℕ) :
  total_athletes = 56 → 
  selected_athletes = 8 → 
  not_meeting_standard = 2 → 
  first_level_sample = 3 → 
  (∀ (A B C D E : ℕ), athletes = {A, B, C, D, E} → first_level_total = 5 → 
  (∃ proportion_standard number_first_level probability_E, 
    proportion_standard = (8 - 2) / 8 ∧  -- first part: proportion of athletes who met the standard
    number_first_level = 56 * (3 / 8) ∧ -- second part: number of first-level athletes
    probability_E = 4 / 10))           -- third part: probability of athlete E being chosen
:= sorry

end diving_assessment_l1582_158279


namespace intersection_complement_l1582_158244

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem intersection_complement :
  A ∩ (U \ B) = {4, 5} := by
  sorry

end intersection_complement_l1582_158244


namespace quadratic_distinct_real_roots_l1582_158247

-- Definitions
def is_quadratic_eq (a b c x : ℝ) (fx : ℝ) := a * x^2 + b * x + c = fx

-- Theorem statement
theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_quadratic_eq 1 (-2) m x₁ 0 ∧ is_quadratic_eq 1 (-2) m x₂ 0) → m < 1 :=
sorry -- Proof omitted

end quadratic_distinct_real_roots_l1582_158247


namespace find_constants_a_b_l1582_158261

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![2, -2]
]

theorem find_constants_a_b :
  ∃ (a b : ℚ), (M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  a = 1/8 ∧ b = -1/8 :=
by
  sorry

end find_constants_a_b_l1582_158261


namespace range_of_a_l1582_158223

noncomputable def tangent_slopes (a x0 : ℝ) : ℝ × ℝ :=
  let k1 := (a * x0 + a - 1) * Real.exp x0
  let k2 := (x0 - 2) * Real.exp (-x0)
  (k1, k2)

theorem range_of_a (a x0 : ℝ) (h : x0 ∈ Set.Icc 0 (3 / 2))
  (h_perpendicular : (tangent_slopes a x0).1 * (tangent_slopes a x0).2 = -1)
  : 1 ≤ a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l1582_158223


namespace find_unique_f_l1582_158238

theorem find_unique_f (f : ℝ → ℝ) (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f (x) * f (y * z) + 1) : 
    ∀ x : ℝ, f x = 1 :=
by
  sorry

end find_unique_f_l1582_158238


namespace factorize_expr_solve_inequality_solve_equation_simplify_expr_l1582_158219

-- Problem 1
theorem factorize_expr (x y m n : ℝ) : x^2 * (3 * m - 2 * n) + y^2 * (2 * n - 3 * m) = (3 * m - 2 * n) * (x + y) * (x - y) := 
sorry

-- Problem 2
theorem solve_inequality (x : ℝ) : 
  (∃ x, (x - 3) / 2 + 3 > x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) → -2 < x ∧ x < 1 :=
sorry

-- Problem 3
theorem solve_equation (x : ℝ) : 
  (∃ x, (3 - x) / (x - 4) + 1 / (4 - x) = 1) → x = 3 :=
sorry

-- Problem 4
theorem simplify_expr (a : ℝ) (h : a = 3) : 
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a - 1)) = 3 / 4 :=
sorry

end factorize_expr_solve_inequality_solve_equation_simplify_expr_l1582_158219


namespace ratio_unchanged_l1582_158293

-- Define the initial ratio
def initial_ratio (a b : ℕ) : ℚ := a / b

-- Define the new ratio after transformation
def new_ratio (a b : ℕ) : ℚ := (3 * a) / (b / (1/3))

-- The theorem stating that the ratio remains unchanged
theorem ratio_unchanged (a b : ℕ) (hb : b ≠ 0) :
  initial_ratio a b = new_ratio a b :=
by
  sorry

end ratio_unchanged_l1582_158293


namespace smallest_number_divisible_l1582_158242

theorem smallest_number_divisible (n : ℕ) :
  (∀ d ∈ [4, 6, 8, 10, 12, 14, 16], (n - 16) % d = 0) ↔ n = 3376 :=
by {
  sorry
}

end smallest_number_divisible_l1582_158242


namespace undefined_value_of_expression_l1582_158283

theorem undefined_value_of_expression (a : ℝ) : (a^3 - 8 = 0) → (a = 2) := by
  sorry

end undefined_value_of_expression_l1582_158283


namespace min_abs_ab_l1582_158269

theorem min_abs_ab (a b : ℤ) (h : 1009 * a + 2 * b = 1) : ∃ k : ℤ, |a * b| = 504 :=
by
  sorry

end min_abs_ab_l1582_158269


namespace neg_of_forall_sin_ge_neg_one_l1582_158241

open Real

theorem neg_of_forall_sin_ge_neg_one :
  (¬ (∀ x : ℝ, sin x ≥ -1)) ↔ (∃ x0 : ℝ, sin x0 < -1) := by
  sorry

end neg_of_forall_sin_ge_neg_one_l1582_158241


namespace barrel_capacity_is_16_l1582_158245

noncomputable def capacity_of_barrel (midway_tap_rate bottom_tap_rate used_bottom_tap_early_time assistant_use_time : Nat) : Nat :=
  let midway_draw := used_bottom_tap_early_time / midway_tap_rate
  let bottom_draw_assistant := assistant_use_time / bottom_tap_rate
  let total_extra_draw := midway_draw + bottom_draw_assistant
  2 * total_extra_draw

theorem barrel_capacity_is_16 :
  capacity_of_barrel 6 4 24 16 = 16 :=
by
  sorry

end barrel_capacity_is_16_l1582_158245


namespace same_color_difference_perfect_square_l1582_158233

theorem same_color_difference_perfect_square :
  (∃ (f : ℤ → ℕ) (a b : ℤ), f a = f b ∧ a ≠ b ∧ ∃ (k : ℤ), a - b = k * k) :=
sorry

end same_color_difference_perfect_square_l1582_158233


namespace soda_cost_original_l1582_158225

theorem soda_cost_original 
  (x : ℚ) -- note: x in rational numbers to capture fractional cost accurately
  (h1 : 3 * (0.90 * x) = 6) :
  x = 20 / 9 :=
by
  sorry

end soda_cost_original_l1582_158225


namespace minimize_sum_dist_l1582_158230

noncomputable section

variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Conditions
def clusters (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) :=
  Q3 <= Q1 + Q2 + Q4 / 3 ∧ Q3 = (Q1 + 2 * Q2 + 2 * Q4) / 5 ∧
  Q7 <= Q5 + Q6 + Q8 / 3 ∧ Q7 = (Q5 + 2 * Q6 + 2 * Q8) / 5

-- Sum of distances function
def sum_dist (Q : ℝ) (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) +
  abs (Q - Q5) + abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- Theorem
theorem minimize_sum_dist (h : clusters Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) :
  ∃ Q : ℝ, (∀ Q' : ℝ, sum_dist Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ≤ sum_dist Q' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) → Q = Q5 :=
sorry

end minimize_sum_dist_l1582_158230


namespace geometric_sequence_term_number_l1582_158281

theorem geometric_sequence_term_number 
  (a_n : ℕ → ℝ)
  (a1 : ℝ) (q : ℝ) (n : ℕ)
  (h1 : a1 = 1/2)
  (h2 : q = 1/2)
  (h3 : a_n n = 1/32)
  (h4 : ∀ n, a_n n = a1 * (q^(n-1))) :
  n = 5 := 
by
  sorry

end geometric_sequence_term_number_l1582_158281


namespace determine_d_value_l1582_158207

noncomputable def Q (d : ℚ) (x : ℚ) : ℚ := x^3 + 3 * x^2 + d * x + 8

theorem determine_d_value (d : ℚ) : x - 3 ∣ Q d x → d = -62 / 3 := by
  sorry

end determine_d_value_l1582_158207


namespace sum_of_coefficients_l1582_158209

theorem sum_of_coefficients (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = 1) :
    (x - 2 * y + 3 * z) ^ 12 = 4096 :=
by
  sorry

end sum_of_coefficients_l1582_158209


namespace compare_a_b_c_l1582_158227

noncomputable def a : ℝ := (1 / 3)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := Real.logb (1 / 3) (1 / 4)

theorem compare_a_b_c : b < a ∧ a < c := by
  sorry

end compare_a_b_c_l1582_158227


namespace sum_of_parallelogram_sides_l1582_158202

-- Definitions of the given conditions.
def length_one_side : ℕ := 10
def length_other_side : ℕ := 7

-- Theorem stating the sum of the lengths of the four sides of the parallelogram.
theorem sum_of_parallelogram_sides : 
    (length_one_side + length_one_side + length_other_side + length_other_side) = 34 :=
by
    sorry

end sum_of_parallelogram_sides_l1582_158202


namespace gopi_salary_turbans_l1582_158286

-- Define the question and conditions as statements
def total_salary (turbans : ℕ) : ℕ := 90 + 30 * turbans
def servant_receives : ℕ := 60 + 30
def fraction_annual_salary : ℚ := 3 / 4

-- The theorem statement capturing the equivalent proof problem
theorem gopi_salary_turbans (T : ℕ) 
  (salary_eq : total_salary T = 90 + 30 * T)
  (servant_eq : servant_receives = 60 + 30)
  (fraction_eq : fraction_annual_salary = 3 / 4)
  (received_after_9_months : ℚ) :
  fraction_annual_salary * (90 + 30 * T : ℚ) = received_after_9_months → 
  received_after_9_months = 90 →
  T = 1 :=
sorry

end gopi_salary_turbans_l1582_158286


namespace actual_order_correct_l1582_158217

-- Define the actual order of the students.
def actual_order := ["E", "D", "A", "C", "B"]

-- Define the first person's prediction and conditions.
def first_person_prediction := ["A", "B", "C", "D", "E"]
def first_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  (pos1 ≠ "A") ∧ (pos2 ≠ "B") ∧ (pos3 ≠ "C") ∧ (pos4 ≠ "D") ∧ (pos5 ≠ "E") ∧
  (pos1 ≠ "B") ∧ (pos2 ≠ "A") ∧ (pos2 ≠ "C") ∧ (pos3 ≠ "B") ∧ (pos3 ≠ "D") ∧
  (pos4 ≠ "C") ∧ (pos4 ≠ "E") ∧ (pos5 ≠ "D")

-- Define the second person's prediction and conditions.
def second_person_prediction := ["D", "A", "E", "C", "B"]
def second_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  ((pos1 = "D") ∨ (pos2 = "D") ∨ (pos3 = "D") ∨ (pos4 = "D") ∨ (pos5 = "D")) ∧
  ((pos1 = "A") ∨ (pos2 = "A") ∨ (pos3 = "A") ∨ (pos4 = "A") ∨ (pos5 = "A")) ∧
  (pos1 ≠ "D" ∨ pos2 ≠ "A") ∧ (pos2 ≠ "A" ∨ pos3 ≠ "E") ∧ (pos3 ≠ "E" ∨ pos4 ≠ "C") ∧ (pos4 ≠ "C" ∨ pos5 ≠ "B")

-- The theorem to prove the actual order.
theorem actual_order_correct :
  ∃ (pos1 pos2 pos3 pos4 pos5 : String),
    first_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    second_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    [pos1, pos2, pos3, pos4, pos5] = actual_order :=
by sorry

end actual_order_correct_l1582_158217


namespace solve_for_x_l1582_158226
-- Lean 4 Statement

theorem solve_for_x (x : ℝ) (h : 2^(3 * x) = Real.sqrt 32) : x = 5 / 6 := 
sorry

end solve_for_x_l1582_158226


namespace marie_distance_biked_l1582_158268

def biking_speed := 12.0 -- Speed in miles per hour
def biking_time := 2.583333333 -- Time in hours

theorem marie_distance_biked : biking_speed * biking_time = 31 := 
by 
  -- The proof steps go here
  sorry

end marie_distance_biked_l1582_158268


namespace lex_coins_total_l1582_158258

def value_of_coins (dimes quarters : ℕ) : ℕ :=
  10 * dimes + 25 * quarters

def more_quarters_than_dimes (dimes quarters : ℕ) : Prop :=
  quarters > dimes

theorem lex_coins_total (dimes quarters : ℕ) (h : value_of_coins dimes quarters = 265) (h_more : more_quarters_than_dimes dimes quarters) : dimes + quarters = 13 :=
sorry

end lex_coins_total_l1582_158258


namespace inequality_solution_l1582_158234

theorem inequality_solution (x : ℝ) : (2 * x - 3 < x + 1) -> (x < 4) :=
by
  intro h
  sorry

end inequality_solution_l1582_158234


namespace average_of_remaining_two_l1582_158280

theorem average_of_remaining_two (a1 a2 a3 a4 a5 : ℚ)
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 11)
  (h2 : (a1 + a2 + a3) / 3 = 4) :
  ((a4 + a5) / 2 = 21.5) :=
sorry

end average_of_remaining_two_l1582_158280


namespace tobias_time_spent_at_pool_l1582_158206

-- Define the conditions
def distance_per_interval : ℕ := 100
def time_per_interval : ℕ := 5
def pause_interval : ℕ := 25
def pause_time : ℕ := 5
def total_distance : ℕ := 3000
def total_time_in_hours : ℕ := 3

-- Hypotheses based on the problem conditions
def swimming_time_without_pauses := (total_distance / distance_per_interval) * time_per_interval
def number_of_pauses := (swimming_time_without_pauses / pause_interval)
def total_pause_time := number_of_pauses * pause_time
def total_time := swimming_time_without_pauses + total_pause_time

-- Proof statement
theorem tobias_time_spent_at_pool : total_time / 60 = total_time_in_hours :=
by 
  -- Put proof here
  sorry

end tobias_time_spent_at_pool_l1582_158206


namespace vlad_taller_than_sister_l1582_158204

theorem vlad_taller_than_sister : 
  ∀ (vlad_height sister_height : ℝ), 
  vlad_height = 190.5 → sister_height = 86.36 → vlad_height - sister_height = 104.14 :=
by
  intros vlad_height sister_height vlad_height_eq sister_height_eq
  rw [vlad_height_eq, sister_height_eq]
  sorry

end vlad_taller_than_sister_l1582_158204


namespace am_minus_hm_lt_bound_l1582_158236

theorem am_minus_hm_lt_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  (x - y)^2 / (2 * (x + y)) < (x - y)^2 / (8 * x) := 
by
  sorry

end am_minus_hm_lt_bound_l1582_158236


namespace distance_borya_vasya_l1582_158212

-- Definitions of the houses and distances on the road
def distance_andrey_gena : ℕ := 2450
def race_length : ℕ := 1000

-- Variables to represent the distances
variables (y b : ℕ)

-- Conditions
def start_position := y
def finish_position := b / 2 + 1225

axiom distance_eq : distance_andrey_gena = 2 * y
axiom race_distance_eq : finish_position - start_position = race_length

-- Proving the distance between Borya's and Vasya's houses
theorem distance_borya_vasya :
  ∃ (d : ℕ), d = 450 :=
by
  sorry

end distance_borya_vasya_l1582_158212


namespace angle_sum_of_roots_of_complex_eq_32i_l1582_158270

noncomputable def root_angle_sum : ℝ :=
  let θ1 := 22.5
  let θ2 := 112.5
  let θ3 := 202.5
  let θ4 := 292.5
  θ1 + θ2 + θ3 + θ4

theorem angle_sum_of_roots_of_complex_eq_32i :
  root_angle_sum = 630 := by
  sorry

end angle_sum_of_roots_of_complex_eq_32i_l1582_158270


namespace no_prime_degree_measure_l1582_158231

theorem no_prime_degree_measure :
  ∀ n, 10 ≤ n ∧ n < 20 → ¬ Nat.Prime (180 * (n - 2) / n) :=
by
  intros n h1 h2 
  sorry

end no_prime_degree_measure_l1582_158231


namespace vacation_expenses_split_l1582_158296

theorem vacation_expenses_split
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ)
  (hA : A = 180)
  (hB : B = 240)
  (hC : C = 120)
  (ha : a = 0)
  (hb : b = 0)
  : a - b = 0 := 
by
  sorry

end vacation_expenses_split_l1582_158296


namespace geometric_series_common_ratio_l1582_158205

theorem geometric_series_common_ratio (a S r : ℝ)
  (h1 : a = 172)
  (h2 : S = 400)
  (h3 : S = a / (1 - r)) :
  r = 57 / 100 := 
sorry

end geometric_series_common_ratio_l1582_158205


namespace parabola_has_one_x_intercept_l1582_158284

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l1582_158284


namespace max_min_fraction_l1582_158253

-- Given condition
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Problem statement
theorem max_min_fraction (x y : ℝ) (h : circle_condition x y) :
  -20 / 21 ≤ y / (x - 4) ∧ y / (x - 4) ≤ 0 :=
sorry

end max_min_fraction_l1582_158253


namespace determine_X_with_7_gcd_queries_l1582_158273

theorem determine_X_with_7_gcd_queries : 
  ∀ (X : ℕ), (X ≤ 100) → ∃ (f : Fin 7 → ℕ × ℕ), 
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) ∧ (∃ (Y : Fin 7 → ℕ), 
      (∀ i, Y i = Nat.gcd (X + (f i).1) (f i).2) → 
        (∀ (X' : ℕ), (X' ≤ 100) → ((∀ i, Y i = Nat.gcd (X' + (f i).1) (f i).2) → X' = X))) :=
sorry

end determine_X_with_7_gcd_queries_l1582_158273


namespace mike_taller_than_mark_l1582_158257

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l1582_158257


namespace nancy_total_spent_l1582_158266

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l1582_158266


namespace distances_inequality_l1582_158214

theorem distances_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + 
  Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤ 
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + 
  Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 :=
  sorry

end distances_inequality_l1582_158214


namespace find_frac_sin_cos_l1582_158252

theorem find_frac_sin_cos (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin (3 * Real.pi / 2 + α)) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 :=
by
  sorry

end find_frac_sin_cos_l1582_158252


namespace total_area_at_stage_4_l1582_158264

/-- Define the side length of the square at a given stage -/
def side_length (n : ℕ) : ℕ := n + 2

/-- Define the area of the square at a given stage -/
def area (n : ℕ) : ℕ := (side_length n) ^ 2

/-- State the theorem -/
theorem total_area_at_stage_4 : 
  (area 0) + (area 1) + (area 2) + (area 3) = 86 :=
by
  -- proof goes here
  sorry

end total_area_at_stage_4_l1582_158264


namespace inverse_proposition_true_l1582_158272

theorem inverse_proposition_true (x : ℝ) (h : x > 1 → x^2 > 1) : x^2 ≤ 1 → x ≤ 1 :=
by
  intros h₂
  sorry

end inverse_proposition_true_l1582_158272


namespace book_pages_l1582_158201

-- Define the number of pages read each day
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5
def pages_tomorrow : ℕ := 35

-- Total number of pages in the book
def total_pages : ℕ := pages_yesterday + pages_today + pages_tomorrow

-- Proof that the total number of pages is 100
theorem book_pages : total_pages = 100 := by
  -- Skip the detailed proof
  sorry

end book_pages_l1582_158201


namespace find_a_minus_c_l1582_158200

section
variables (a b c : ℝ)
variables (h₁ : (a + b) / 2 = 110) (h₂ : (b + c) / 2 = 170)

theorem find_a_minus_c : a - c = -120 :=
by
  sorry
end

end find_a_minus_c_l1582_158200


namespace paul_mowing_money_l1582_158288

theorem paul_mowing_money (M : ℝ) 
  (h1 : 2 * M = 6) : 
  M = 3 :=
by 
  sorry

end paul_mowing_money_l1582_158288


namespace polynomial_remainder_l1582_158237

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def rem (x : ℝ) : ℝ := 14 * x - 14

theorem polynomial_remainder :
  ∀ x : ℝ, p x % d x = rem x := 
by
  sorry

end polynomial_remainder_l1582_158237


namespace shooting_competition_probabilities_l1582_158278

theorem shooting_competition_probabilities (p_A_not_losing p_B_losing : ℝ)
  (h₁ : p_A_not_losing = 0.59)
  (h₂ : p_B_losing = 0.44) :
  (1 - p_B_losing = 0.56) ∧ (p_A_not_losing - p_B_losing = 0.15) :=
by
  sorry

end shooting_competition_probabilities_l1582_158278


namespace total_lemonade_poured_l1582_158290

def lemonade_poured (first: ℝ) (second: ℝ) (third: ℝ) := first + second + third

theorem total_lemonade_poured :
  lemonade_poured 0.25 0.4166666666666667 0.25 = 0.917 :=
by
  sorry

end total_lemonade_poured_l1582_158290


namespace range_of_f1_3_l1582_158295

noncomputable def f (a b : ℝ) (x y : ℝ) : ℝ :=
  a * (x^3 + 3 * x) + b * (y^2 + 2 * y + 1)

theorem range_of_f1_3 (a b : ℝ)
  (h1 : 1 ≤ f a b 1 2 ∧ f a b 1 2 ≤ 2)
  (h2 : 2 ≤ f a b 3 4 ∧ f a b 3 4 ≤ 5):
  3 / 2 ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 :=
sorry

end range_of_f1_3_l1582_158295


namespace total_tickets_spent_l1582_158215

def tickets_spent_on_hat : ℕ := 2
def tickets_spent_on_stuffed_animal : ℕ := 10
def tickets_spent_on_yoyo : ℕ := 2

theorem total_tickets_spent :
  tickets_spent_on_hat + tickets_spent_on_stuffed_animal + tickets_spent_on_yoyo = 14 := by
  sorry

end total_tickets_spent_l1582_158215


namespace volume_ratio_of_cube_and_cuboid_l1582_158299

theorem volume_ratio_of_cube_and_cuboid :
  let edge_length_meter := 1
  let edge_length_cm := edge_length_meter * 100 -- Convert meter to centimeters
  let cube_volume := edge_length_cm^3
  let cuboid_width := 50
  let cuboid_length := 50
  let cuboid_height := 20
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume = 20 * cuboid_volume := 
by
  sorry

end volume_ratio_of_cube_and_cuboid_l1582_158299


namespace number_of_large_posters_is_5_l1582_158298

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l1582_158298


namespace find_angle_l1582_158265

theorem find_angle (A : ℝ) (deg_to_rad : ℝ) :
  (1/2 * Real.sin (A / 2 * deg_to_rad) + Real.cos (A / 2 * deg_to_rad) = 1) →
  (A = 360) :=
sorry

end find_angle_l1582_158265


namespace travis_apples_l1582_158208

theorem travis_apples
  (price_per_box : ℕ)
  (num_apples_per_box : ℕ)
  (total_money : ℕ)
  (total_boxes : ℕ)
  (total_apples : ℕ)
  (h1 : price_per_box = 35)
  (h2 : num_apples_per_box = 50)
  (h3 : total_money = 7000)
  (h4 : total_boxes = total_money / price_per_box)
  (h5 : total_apples = total_boxes * num_apples_per_box) :
  total_apples = 10000 :=
sorry

end travis_apples_l1582_158208


namespace alyssa_cut_11_roses_l1582_158250

theorem alyssa_cut_11_roses (initial_roses cut_roses final_roses : ℕ) 
  (h1 : initial_roses = 3) 
  (h2 : final_roses = 14) 
  (h3 : initial_roses + cut_roses = final_roses) : 
  cut_roses = 11 :=
by
  rw [h1, h2] at h3
  sorry

end alyssa_cut_11_roses_l1582_158250


namespace geom_sequence_product_l1582_158251

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_product (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : a 0 * a 4 = 4) :
  a 0 * a 1 * a 2 * a 3 * a 4 = 32 ∨ a 0 * a 1 * a 2 * a 3 * a 4 = -32 :=
by
  sorry

end geom_sequence_product_l1582_158251


namespace prism_base_shape_l1582_158243

theorem prism_base_shape (n : ℕ) (hn : 3 * n = 12) : n = 4 := by
  sorry

end prism_base_shape_l1582_158243


namespace max_k_value_l1582_158259

theorem max_k_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = a * b + b * c + c * a →
  (a + b + c) * (1 / (a + b) + 1 / (b + c) + 1 / (c + a) - 1) ≥ 1 :=
by
  intros a b c ha hb hc habc_eq
  sorry

end max_k_value_l1582_158259


namespace simultaneous_equations_solution_exists_l1582_158297

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  -- proof goes here
  sorry

end simultaneous_equations_solution_exists_l1582_158297


namespace both_firms_participate_social_optimality_l1582_158248

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l1582_158248


namespace magic_triangle_max_sum_l1582_158249

/-- In a magic triangle, each of the six consecutive whole numbers 11 to 16 is placed in one of the circles. 
    The sum, S, of the three numbers on each side of the triangle is the same. One of the sides must contain 
    three consecutive numbers. Prove that the largest possible value for S is 41. -/
theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), 
  (a = 11 ∨ a = 12 ∨ a = 13 ∨ a = 14 ∨ a = 15 ∨ a = 16) ∧
  (b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16) ∧
  (c = 11 ∨ c = 12 ∨ c = 13 ∨ c = 14 ∨ c = 15 ∨ c = 16) ∧
  (d = 11 ∨ d = 12 ∨ d = 13 ∨ d = 14 ∨ d = 15 ∨ d = 16) ∧
  (e = 11 ∨ e = 12 ∨ e = 13 ∨ e = 14 ∨ e = 15 ∨ e = 16) ∧
  (f = 11 ∨ f = 12 ∨ f = 13 ∨ f = 14 ∨ f = 15 ∨ f = 16) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  (a + b + c = S) ∧ (c + d + e = S) ∧ (e + f + a = S) ∧
  (∃ k, a = k ∧ b = k+1 ∧ c = k+2 ∨ b = k ∧ c = k+1 ∧ d = k+2 ∨ c = k ∧ d = k+1 ∧ e = k+2 ∨ d = k ∧ e = k+1 ∧ f = k+2) →
  S = 41 :=
by
  sorry

end magic_triangle_max_sum_l1582_158249


namespace rectangle_ratio_ratio_simplification_l1582_158254

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l1582_158254


namespace probability_of_green_l1582_158275

open Classical

-- Define the total number of balls in each container
def balls_A := 12
def balls_B := 14
def balls_C := 12

-- Define the number of green balls in each container
def green_balls_A := 7
def green_balls_B := 6
def green_balls_C := 9

-- Define the probability of selecting each container
def prob_select_container := (1:ℚ) / 3

-- Define the probability of drawing a green ball from each container
def prob_green_A := green_balls_A / balls_A
def prob_green_B := green_balls_B / balls_B
def prob_green_C := green_balls_C / balls_C

-- Define the total probability of drawing a green ball
def total_prob_green := prob_select_container * prob_green_A +
                        prob_select_container * prob_green_B +
                        prob_select_container * prob_green_C

-- Create the proof statement
theorem probability_of_green : total_prob_green = 127 / 252 := 
by
  -- Skip the proof
  sorry

end probability_of_green_l1582_158275


namespace nine_by_nine_chessboard_dark_light_excess_l1582_158220

theorem nine_by_nine_chessboard_dark_light_excess :
  let board_size := 9
  let odd_row_dark := 5
  let odd_row_light := 4
  let even_row_dark := 4
  let even_row_light := 5
  let num_odd_rows := (board_size + 1) / 2
  let num_even_rows := board_size / 2
  let total_dark_squares := (odd_row_dark * num_odd_rows) + (even_row_dark * num_even_rows)
  let total_light_squares := (odd_row_light * num_odd_rows) + (even_row_light * num_even_rows)
  total_dark_squares - total_light_squares = 1 :=
by {
  sorry
}

end nine_by_nine_chessboard_dark_light_excess_l1582_158220


namespace sum_is_24000_l1582_158229

theorem sum_is_24000 (P : ℝ) (R : ℝ) (T : ℝ) : 
  (R = 5) → (T = 2) →
  ((P * (1 + R / 100)^T - P) - (P * R * T / 100) = 60) →
  P = 24000 :=
by
  sorry

end sum_is_24000_l1582_158229


namespace solve_for_x_l1582_158239

theorem solve_for_x (x : ℤ) (h : (3012 + x)^2 = x^2) : x = -1506 := 
sorry

end solve_for_x_l1582_158239


namespace possible_values_of_d_l1582_158263

theorem possible_values_of_d (r s : ℝ) (c d : ℝ)
  (h1 : ∃ u, u = -r - s ∧ r * s + r * u + s * u = c)
  (h2 : ∃ v, v = -r - s - 8 ∧ (r - 3) * (s + 5) + (r - 3) * (u - 8) + (s + 5) * (u - 8) = c)
  (u_eq : u = -r - s)
  (v_eq : v = -r - s - 8)
  (polynomial_relation : d + 156 = -((r - 3) * (s + 5) * (u - 8))) : 
  d = -198 ∨ d = 468 := 
sorry

end possible_values_of_d_l1582_158263


namespace min_odd_integers_l1582_158277

theorem min_odd_integers 
  (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 15)
  (h3 : e + f = 17)
  (h4 : c + d + e + f = 32) :
  ∃ n : ℕ, (n = 2) ∧ (∃ odd_count, 
  odd_count = (if (a % 2 = 0) then 0 else 1) + 
                     (if (b % 2 = 0) then 0 else 1) + 
                     (if (c % 2 = 0) then 0 else 1) + 
                     (if (d % 2 = 0) then 0 else 1) + 
                     (if (e % 2 = 0) then 0 else 1) + 
                     (if (f % 2 = 0) then 0 else 1) ∧
  odd_count = 2) := sorry

end min_odd_integers_l1582_158277
