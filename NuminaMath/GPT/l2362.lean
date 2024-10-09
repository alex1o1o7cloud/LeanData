import Mathlib

namespace circle_equation_tangent_l2362_236262

theorem circle_equation_tangent (h : ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = 25)) :
    ∃ c : ℝ × ℝ, c = (1, 2) ∧ ∃ r : ℝ, r = 5 ∧ ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) := 
by
    sorry

end circle_equation_tangent_l2362_236262


namespace opposite_of_2023_l2362_236231

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l2362_236231


namespace geometric_series_expr_l2362_236276

theorem geometric_series_expr :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4)))))))))) + 100 = 5592504 := 
sorry

end geometric_series_expr_l2362_236276


namespace problem_arithmetic_sequence_l2362_236275

-- Definitions based on given conditions
def a1 : ℕ := 2
def d := (13 - 2 * a1) / 3

-- Definition of the nth term in the arithmetic sequence
def a (n : ℕ) : ℕ := a1 + (n - 1) * d

-- The required proof problem statement
theorem problem_arithmetic_sequence : a 4 + a 5 + a 6 = 42 := 
by
  -- placeholders for the actual proof
  sorry

end problem_arithmetic_sequence_l2362_236275


namespace solve_P_Q_l2362_236223

theorem solve_P_Q :
  ∃ P Q : ℝ, (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 6) + Q / (x * (x - 5)) = (x^2 - 3*x + 15) / (x * (x + 6) * (x - 5)))) ∧
    P = 1 ∧ Q = 5/2 :=
by
  sorry

end solve_P_Q_l2362_236223


namespace total_travel_cost_l2362_236210

noncomputable def calculate_cost : ℕ :=
  let cost_length_road :=
    (30 * 10 * 4) +  -- first segment
    (40 * 10 * 5) +  -- second segment
    (30 * 10 * 6)    -- third segment
  let cost_breadth_road :=
    (20 * 10 * 3) +  -- first segment
    (40 * 10 * 2)    -- second segment
  cost_length_road + cost_breadth_road

theorem total_travel_cost :
  calculate_cost = 6400 :=
by
  sorry

end total_travel_cost_l2362_236210


namespace sum_of_first_n_natural_numbers_single_digit_l2362_236273

theorem sum_of_first_n_natural_numbers_single_digit (n : ℕ) :
  (∃ a : ℕ, a ≤ 9 ∧ (a ≠ 0) ∧ 37 * (3 * a) = n * (n + 1) / 2) ↔ (n = 36) :=
by
  sorry

end sum_of_first_n_natural_numbers_single_digit_l2362_236273


namespace isosceles_triangle_of_cosine_equality_l2362_236200

variable {A B C : ℝ}
variable {a b c : ℝ}

/-- Prove that in triangle ABC, if a*cos(B) = b*cos(A), then a = b implying A = B --/
theorem isosceles_triangle_of_cosine_equality 
(h1 : a * Real.cos B = b * Real.cos A) :
a = b :=
sorry

end isosceles_triangle_of_cosine_equality_l2362_236200


namespace candy_bar_sales_l2362_236222

def max_sales : ℕ := 24
def seth_sales (max_sales : ℕ) : ℕ := 3 * max_sales + 6
def emma_sales (seth_sales : ℕ) : ℕ := seth_sales / 2 + 5
def total_sales (seth_sales emma_sales : ℕ) : ℕ := seth_sales + emma_sales

theorem candy_bar_sales : total_sales (seth_sales max_sales) (emma_sales (seth_sales max_sales)) = 122 := by
  sorry

end candy_bar_sales_l2362_236222


namespace afternoon_registration_l2362_236246

variable (m a t morning_absent : ℕ)

theorem afternoon_registration (m a t morning_absent afternoon : ℕ) (h1 : m = 25) (h2 : a = 4) (h3 : t = 42) (h4 : morning_absent = 3) : 
  afternoon = t - (m - morning_absent + morning_absent + a) :=
by sorry

end afternoon_registration_l2362_236246


namespace geometric_sequence_l2362_236272

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence
def geom_seq (a₁ r : α) (n : ℕ) : α := a₁ * r^(n-1)

theorem geometric_sequence :
  ∀ (a₁ : α), a₁ > 0 → geom_seq a₁ 2 3 * geom_seq a₁ 2 11 = 16 → geom_seq a₁ 2 5 = 1 :=
by
  intros a₁ h_pos h_eq
  sorry

end geometric_sequence_l2362_236272


namespace sheet_length_proof_l2362_236209

noncomputable def length_of_sheet (L : ℝ) : ℝ := 48

theorem sheet_length_proof (L : ℝ) (w : ℝ) (s : ℝ) (V : ℝ) (h : ℝ) (new_w : ℝ) :
  w = 36 →
  s = 8 →
  V = 5120 →
  h = s →
  new_w = w - 2 * s →
  V = (L - 2 * s) * new_w * h →
  L = 48 :=
by
  intros hw hs hV hh h_new_w h_volume
  -- conversion of the mathematical equivalent proof problem to Lean's theorem
  sorry

end sheet_length_proof_l2362_236209


namespace math_problem_solution_l2362_236257

theorem math_problem_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_eq : a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
sorry

end math_problem_solution_l2362_236257


namespace xyz_inequality_l2362_236235

theorem xyz_inequality (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + 
  (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
sorry

end xyz_inequality_l2362_236235


namespace evaluation_of_expression_l2362_236291

theorem evaluation_of_expression: 
  (3^10 + 3^7) / (3^10 - 3^7) = 14 / 13 := 
  sorry

end evaluation_of_expression_l2362_236291


namespace set_representation_l2362_236286

open Nat

def isInPositiveNaturals (x : ℕ) : Prop :=
  x ≠ 0

def isPositiveDivisor (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

theorem set_representation :
  {x | isInPositiveNaturals x ∧ isPositiveDivisor 6 (6 - x)} = {3, 4, 5} :=
by
  sorry

end set_representation_l2362_236286


namespace rakesh_fixed_deposit_percentage_l2362_236214

-- Definitions based on the problem statement
def salary : ℝ := 4000
def cash_in_hand : ℝ := 2380
def spent_on_groceries : ℝ := 0.30

-- The theorem to prove
theorem rakesh_fixed_deposit_percentage (x : ℝ) 
  (H1 : cash_in_hand = 0.70 * (salary - (x / 100) * salary)) : 
  x = 15 := 
sorry

end rakesh_fixed_deposit_percentage_l2362_236214


namespace hindi_speaking_students_l2362_236294

theorem hindi_speaking_students 
    (G M T A : ℕ)
    (Total : ℕ)
    (hG : G = 6)
    (hM : M = 6)
    (hT : T = 2)
    (hA : A = 1)
    (hTotal : Total = 22)
    : ∃ H, Total = G + H + M - (T - A) + A ∧ H = 10 := by
  sorry

end hindi_speaking_students_l2362_236294


namespace mark_more_hours_than_kate_l2362_236271

theorem mark_more_hours_than_kate {K : ℕ} (h1 : K + 2 * K + 6 * K = 117) :
  6 * K - K = 65 :=
by
  sorry

end mark_more_hours_than_kate_l2362_236271


namespace total_games_correct_l2362_236280

noncomputable def number_of_games_per_month : ℕ := 13
noncomputable def number_of_months_in_season : ℕ := 14
noncomputable def total_games_in_season : ℕ := number_of_games_per_month * number_of_months_in_season

theorem total_games_correct : total_games_in_season = 182 := by
  sorry

end total_games_correct_l2362_236280


namespace smallest_scalene_triangle_perimeter_is_prime_l2362_236296

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧
  (p2 = p1 + 2) ∧ (p3 = p1 + 6)

noncomputable def smallest_prime_perimeter : ℕ :=
  5 + 7 + 11

theorem smallest_scalene_triangle_perimeter_is_prime :
  ∃ (p1 p2 p3 : ℕ), p1 < p2 ∧ p2 < p3 ∧ consecutive_primes p1 p2 p3 ∧ is_prime (p1 + p2 + p3) ∧ (p1 + p2 + p3 = smallest_prime_perimeter) :=
by 
  sorry

end smallest_scalene_triangle_perimeter_is_prime_l2362_236296


namespace part1_part2_l2362_236265

open Real

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x) + x * cos x + 1

theorem part1 (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1 / 2) * x^2 := 
sorry

theorem part2 (a x : ℝ) (ha : 1 ≤ a) (hx : 0 ≤ x) : f x a ≥ (1 + sin x)^2 := 
sorry

end part1_part2_l2362_236265


namespace multiple_of_every_positive_integer_is_zero_l2362_236264

theorem multiple_of_every_positive_integer_is_zero :
  ∀ (n : ℤ), (∀ (m : ℕ), ∃ (k : ℤ), n = k * (m : ℤ)) → n = 0 := 
by
  sorry

end multiple_of_every_positive_integer_is_zero_l2362_236264


namespace calc_x_squared_y_squared_l2362_236279

theorem calc_x_squared_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -9) : x^2 + y^2 = 22 := by
  sorry

end calc_x_squared_y_squared_l2362_236279


namespace minimum_dot_product_l2362_236255

noncomputable def min_AE_dot_AF : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60 -- this is 60 degrees, which should be converted to radians if we need to use it
  sorry

theorem minimum_dot_product :
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60
  ∃ (E F : ℝ), (min_AE_dot_AF = 29 / 18) :=
    sorry

end minimum_dot_product_l2362_236255


namespace shaded_area_correct_l2362_236277

def unit_triangle_area : ℕ := 10

def small_shaded_area : ℕ := unit_triangle_area

def medium_shaded_area : ℕ := 6 * unit_triangle_area

def large_shaded_area : ℕ := 7 * unit_triangle_area

def total_shaded_area : ℕ :=
  small_shaded_area + medium_shaded_area + large_shaded_area

theorem shaded_area_correct : total_shaded_area = 110 := 
  by
    sorry

end shaded_area_correct_l2362_236277


namespace chess_tournament_total_players_l2362_236212

theorem chess_tournament_total_players :
  ∃ n : ℕ,
    n + 12 = 35 ∧
    ∀ p : ℕ,
      (∃ pts : ℕ,
        p = n + 12 ∧
        pts = (p * (p - 1)) / 2 ∧
        pts = n^2 - n + 132) ∧
      ( ∃ (gained_half_points : ℕ → Prop),
          (∀ k ≤ 12, gained_half_points k) ∧
          (∀ k > 12, ¬ gained_half_points k)) :=
by
  sorry

end chess_tournament_total_players_l2362_236212


namespace negation_of_p_l2362_236295

def p := ∀ x, x ≤ 0 → Real.exp x ≤ 1

theorem negation_of_p : ¬ p ↔ ∃ x, x ≤ 0 ∧ Real.exp x > 1 := 
by
  sorry

end negation_of_p_l2362_236295


namespace series_sum_equals_one_sixth_l2362_236242

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l2362_236242


namespace sum_abc_eq_8_l2362_236269

theorem sum_abc_eq_8 (a b c : ℝ) 
  (h : (a - 5) ^ 2 + (b - 6) ^ 2 + (c - 7) ^ 2 - 2 * (a - 5) * (b - 6) = 0) : 
  a + b + c = 8 := 
sorry

end sum_abc_eq_8_l2362_236269


namespace projection_of_a_onto_b_is_three_l2362_236226

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (1, 0)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_three : projection vec_a vec_b = 3 := by
  sorry

end projection_of_a_onto_b_is_three_l2362_236226


namespace product_of_roots_cubic_l2362_236230

theorem product_of_roots_cubic:
  (∀ x : ℝ, x^3 - 15 * x^2 + 60 * x - 45 = 0 → x = r_1 ∨ x = r_2 ∨ x = r_3) →
  r_1 * r_2 * r_3 = 45 :=
by
  intro h
  -- the proof should be filled in here
  sorry

end product_of_roots_cubic_l2362_236230


namespace gear_C_rotation_direction_gear_C_rotation_count_l2362_236207

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

end gear_C_rotation_direction_gear_C_rotation_count_l2362_236207


namespace value_of_expression_l2362_236234

variable {a : Nat → Int}

def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n m : Nat, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_expression
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 :=
  sorry

end value_of_expression_l2362_236234


namespace domain_of_function_l2362_236228

noncomputable def function_defined (x : ℝ) : Prop :=
  (x > 1) ∧ (x ≠ 2)

theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, y = (1 / (Real.sqrt (x - 1))) + (1 / (x - 2))) ↔ function_defined x :=
by sorry

end domain_of_function_l2362_236228


namespace projection_of_b_onto_a_l2362_236204
-- Import the entire library for necessary functions and definitions.

-- Define the problem in Lean 4, using relevant conditions and statement.
theorem projection_of_b_onto_a (m : ℝ) (h : (1 : ℝ) * 3 + (Real.sqrt 3) * m = 6) : m = Real.sqrt 3 :=
by
  sorry

end projection_of_b_onto_a_l2362_236204


namespace blocks_for_fort_l2362_236266

theorem blocks_for_fort :
  let length := 15 
  let width := 12 
  let height := 6
  let thickness := 1
  let V_original := length * width * height
  let interior_length := length - 2 * thickness
  let interior_width := width - 2 * thickness
  let interior_height := height - thickness
  let V_interior := interior_length * interior_width * interior_height
  let V_blocks := V_original - V_interior
  V_blocks = 430 :=
by
  sorry

end blocks_for_fort_l2362_236266


namespace find_a_b_l2362_236224

theorem find_a_b (a b : ℤ) (h : ∀ x : ℤ, (x - 2) * (x + 3) = x^2 + a * x + b) : a = 1 ∧ b = -6 :=
by
  sorry

end find_a_b_l2362_236224


namespace perpendicular_line_through_intersection_l2362_236289

theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), (x + y - 2 = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (4 * x - 3 * y - 1 = 0) :=
sorry

end perpendicular_line_through_intersection_l2362_236289


namespace min_y_value_l2362_236249

noncomputable def min_value_y : ℝ :=
  18 - 2 * Real.sqrt 106

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20 * x + 36 * y) : 
  y >= 18 - 2 * Real.sqrt 106 :=
sorry

end min_y_value_l2362_236249


namespace infinite_unlucky_numbers_l2362_236205

def is_unlucky (n : ℕ) : Prop :=
  ¬(∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (n = x^2 - 1 ∨ n = y^2 - 1))

theorem infinite_unlucky_numbers : ∀ᶠ n in at_top, is_unlucky n := sorry

end infinite_unlucky_numbers_l2362_236205


namespace second_job_pay_rate_l2362_236229

-- Definitions of the conditions
def h1 : ℕ := 3 -- hours for the first job
def r1 : ℕ := 7 -- rate for the first job
def h2 : ℕ := 2 -- hours for the second job
def h3 : ℕ := 4 -- hours for the third job
def r3 : ℕ := 12 -- rate for the third job
def d : ℕ := 5   -- number of days
def T : ℕ := 445 -- total earnings

-- The proof statement
theorem second_job_pay_rate (x : ℕ) : 
  d * (h1 * r1 + 2 * x + h3 * r3) = T ↔ x = 10 := 
by 
  -- Implement the necessary proof steps here
  sorry

end second_job_pay_rate_l2362_236229


namespace consecutive_numbers_N_l2362_236220

theorem consecutive_numbers_N (N : ℕ) (h : ∀ k, 0 < k → k < 15 → N + k < 81) : N = 66 :=
sorry

end consecutive_numbers_N_l2362_236220


namespace find_c_l2362_236233

-- Define the function
def f (c x : ℝ) : ℝ := x^4 - 8 * x^2 + c

-- Condition: The function has a minimum value of -14 on the interval [-1, 3]
def condition (c : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3, f c x ≤ f c y ∧ f c x = -14

-- The theorem to be proved
theorem find_c : ∃ c : ℝ, condition c ∧ c = 2 :=
sorry

end find_c_l2362_236233


namespace triangle_area_ratios_l2362_236293

theorem triangle_area_ratios (K : ℝ) 
  (hCD : ∃ AC, ∃ CD, CD = AC / 4) 
  (hAE : ∃ AB, ∃ AE, AE = AB / 5) 
  (hBF : ∃ BC, ∃ BF, BF = BC / 3) :
  ∃ area_N1N2N3, area_N1N2N3 = (8 / 15) * K :=
by
  sorry

end triangle_area_ratios_l2362_236293


namespace maximum_m_l2362_236232

theorem maximum_m (a b c : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b + c = 10)
  (h₅ : a * b + b * c + c * a = 25) :
  ∃ m, (m = min (a * b) (min (b * c) (c * a)) ∧ m = 25 / 9) :=
sorry

end maximum_m_l2362_236232


namespace angle_in_third_quadrant_l2362_236253

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi + β ∧ β ∈ Set.Ioo (0 : ℝ) Real.pi :=
by
  sorry

end angle_in_third_quadrant_l2362_236253


namespace det_B_l2362_236270

open Matrix

-- Define matrix B
def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 2], ![-3, y]]

-- Define the condition B + 2 * B⁻¹ = 0
def condition (x y : ℝ) : Prop :=
  let Binv := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]
  B x y + 2 • Binv = 0

-- Prove that if the condition holds, then det B = 2
theorem det_B (x y : ℝ) (h : condition x y) : det (B x y) = 2 :=
  sorry

end det_B_l2362_236270


namespace four_m0_as_sum_of_primes_l2362_236290

theorem four_m0_as_sum_of_primes (m0 : ℕ) (h1 : m0 > 1) 
  (h2 : ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n) 
  (h3 : ∀ p1 p2 : ℕ, Prime p1 → Prime p2 → (2 * m0 ≠ p1 + p2)) : 
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ (4 * m0 = p1 + p2 + p3 + p4) ∨ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 4 * m0 = p1 + p2 + p3) :=
by sorry

end four_m0_as_sum_of_primes_l2362_236290


namespace add_eq_pm_three_max_sub_eq_five_l2362_236215

-- Define the conditions for m and n
variables (m n : ℤ)
def abs_m_eq_one : Prop := |m| = 1
def abs_n_eq_four : Prop := |n| = 4

-- State the first theorem regarding m + n given mn < 0
theorem add_eq_pm_three (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) (h_mn : m * n < 0) : m + n = 3 ∨ m + n = -3 := 
sorry

-- State the second theorem regarding maximum value of m - n
theorem max_sub_eq_five (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) : ∃ max_val, (max_val = m - n) ∧ (∀ (x : ℤ), (|m| = 1) ∧ (|n| = 4)  → (x = m - n)  → x ≤ max_val) ∧ max_val = 5 := 
sorry

end add_eq_pm_three_max_sub_eq_five_l2362_236215


namespace components_le_20_components_le_n_squared_div_4_l2362_236268

-- Question part b: 8x8 grid, can the number of components be more than 20
theorem components_le_20 {c : ℕ} (h1 : c = 64 / 4) : c ≤ 20 := by
  sorry

-- Question part c: n x n grid, can the number of components be more than n^2 / 4
theorem components_le_n_squared_div_4 (n : ℕ) (h2 : n > 8) {c : ℕ} (h3 : c = n^2 / 4) : 
  c ≤ n^2 / 4 := by
  sorry

end components_le_20_components_le_n_squared_div_4_l2362_236268


namespace tank_capacity_l2362_236259

theorem tank_capacity (T : ℝ) (h1 : 0.6 * T = 0.7 * T - 45) : T = 450 :=
by
  sorry

end tank_capacity_l2362_236259


namespace factor_polynomial_l2362_236211

theorem factor_polynomial (y : ℝ) : 3 * y ^ 2 - 75 = 3 * (y - 5) * (y + 5) :=
by
  sorry

end factor_polynomial_l2362_236211


namespace find_a_l2362_236241

theorem find_a (a b : ℝ) (h₀ : b = 4) (h₁ : (4, b) ∈ {p | p.snd = 0.75 * p.fst + 1}) 
  (h₂ : (a, 5) ∈ {p | p.snd = 0.75 * p.fst + 1}) (h₃ : (a, b+1) ∈ {p | p.snd = 0.75 * p.fst + 1}) : 
  a = 5.33 :=
by 
  sorry

end find_a_l2362_236241


namespace arithmetic_sequence_common_difference_l2362_236206

theorem arithmetic_sequence_common_difference 
  (a : Nat → Int)
  (a1 : a 1 = 5)
  (a6_a8_sum : a 6 + a 8 = 58) :
  ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 4 := 
by 
  sorry

end arithmetic_sequence_common_difference_l2362_236206


namespace ratio_yx_l2362_236217

variable (c x y : ℝ)

theorem ratio_yx (h1: x = 0.80 * c) (h2: y = 1.25 * c) : y / x = 25 / 16 := by
  -- Proof to be written here
  sorry

end ratio_yx_l2362_236217


namespace larry_substitution_l2362_236258

theorem larry_substitution (a b c d e : ℤ)
  (ha : a = 1)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 4)
  (h_ignored : a - b - c - d + e = a - (b - (c - (d + e)))) :
  e = 3 :=
by
  sorry

end larry_substitution_l2362_236258


namespace f_of_13_eq_223_l2362_236239

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_of_13_eq_223 : f 13 = 223 := 
by sorry

end f_of_13_eq_223_l2362_236239


namespace measure_angle_B_triangle_area_correct_l2362_236260

noncomputable def triangle_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) → B = Real.pi / 3

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area1 := (3 + Real.sqrt 3)
  let area2 := Real.sqrt 3
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  let sinA1 := (Real.sqrt 2 / 2)
  let sinA2 := (Real.sqrt 6 - Real.sqrt 2) / 4
  let S1 := (1 / 2) * b * c * sinA1
  let S2 := (1 / 2) * b * c * sinA2
  S1 = area1 ∨ S2 = area2

theorem measure_angle_B :
  ∀ (a b c A B C : ℝ),
    triangle_angle_B a b c A B C := sorry

theorem triangle_area_correct :
  ∀ (a b c A B C : ℝ),
    triangle_area a b c A B C := sorry

end measure_angle_B_triangle_area_correct_l2362_236260


namespace smallest_number_is_28_l2362_236299

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end smallest_number_is_28_l2362_236299


namespace average_price_l2362_236297

theorem average_price (books1 books2 : ℕ) (price1 price2 : ℝ)
  (h1 : books1 = 65) (h2 : price1 = 1380)
  (h3 : books2 = 55) (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2) = 19 :=
by
  sorry

end average_price_l2362_236297


namespace evaluate_expression_l2362_236254

-- Define the expression and the expected result
def expression := -(14 / 2 * 9 - 60 + 3 * 9)
def expectedResult := -30

-- The theorem that states the equivalence
theorem evaluate_expression : expression = expectedResult := by
  sorry

end evaluate_expression_l2362_236254


namespace find_a_l2362_236213

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 2*(a-1)*x + 2

theorem find_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ 2 → 2 ≤ x2 → quadratic_function a x1 ≥ quadratic_function a 2 ∧ quadratic_function a 2 ≤ quadratic_function a x2) →
  a = 3 :=
by
  sorry

end find_a_l2362_236213


namespace gcd_90_405_l2362_236292

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l2362_236292


namespace largest_value_of_a_l2362_236278

noncomputable def largest_possible_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150) : Prop :=
  a = 8924

theorem largest_value_of_a (a b c d : ℕ)
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150)
  (h6 : largest_possible_value_of_a a b c d h1 h2 h3 h4 h5) : a = 8924 := h6

end largest_value_of_a_l2362_236278


namespace solve_for_b_l2362_236256

theorem solve_for_b (a b : ℚ) 
  (h1 : 8 * a + 3 * b = -1) 
  (h2 : a = b - 3 ) : 
  5 * b = 115 / 11 := 
by 
  sorry

end solve_for_b_l2362_236256


namespace simple_interest_rate_l2362_236251

theorem simple_interest_rate (P T A R : ℝ) (hT : T = 15) (hA : A = 4 * P)
  (hA_simple_interest : A = P + (P * R * T / 100)) : R = 20 :=
by
  sorry

end simple_interest_rate_l2362_236251


namespace triangle_area_ratio_l2362_236261

noncomputable def vector_sum_property (OA OB OC : ℝ × ℝ × ℝ) : Prop :=
  OA + (2 : ℝ) • OB + (3 : ℝ) • OC = (0 : ℝ × ℝ × ℝ)

noncomputable def area_ratio (S_ABC S_AOC : ℝ) : Prop :=
  S_ABC / S_AOC = 3

theorem triangle_area_ratio
    (OA OB OC : ℝ × ℝ × ℝ)
    (S_ABC S_AOC : ℝ)
    (h1 : vector_sum_property OA OB OC)
    (h2 : S_ABC = 3 * S_AOC) :
  area_ratio S_ABC S_AOC :=
by
  sorry

end triangle_area_ratio_l2362_236261


namespace desired_percentage_total_annual_income_l2362_236218

variable (investment1 : ℝ)
variable (investment2 : ℝ)
variable (rate1 : ℝ)
variable (rate2 : ℝ)

theorem desired_percentage_total_annual_income (h1 : investment1 = 2000)
  (h2 : rate1 = 0.05)
  (h3 : investment2 = 1000-1e-13)
  (h4 : rate2 = 0.08):
  ((investment1 * rate1 + investment2 * rate2) / (investment1 + investment2) * 100) = 6 := by
  sorry

end desired_percentage_total_annual_income_l2362_236218


namespace total_teams_l2362_236248

theorem total_teams (m n : ℕ) (hmn : m > n) : 
  (m - n) + 1 = m - n + 1 := 
by sorry

end total_teams_l2362_236248


namespace length_of_first_train_is_270_04_l2362_236208

noncomputable def length_of_first_train (speed_first_train_kmph : ℕ) (speed_second_train_kmph : ℕ) 
  (time_seconds : ℕ) (length_second_train_m : ℕ) : ℕ :=
  let combined_speed_mps := ((speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600) 
  let combined_length := combined_speed_mps * time_seconds
  combined_length - length_second_train_m

theorem length_of_first_train_is_270_04 :
  length_of_first_train 120 80 9 230 = 270 :=
by
  sorry

end length_of_first_train_is_270_04_l2362_236208


namespace max_sum_a_b_c_l2362_236236

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem max_sum_a_b_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≥ -1) : a + b + c ≤ 3 :=
sorry

end max_sum_a_b_c_l2362_236236


namespace modulus_sum_l2362_236238

def z1 : ℂ := 3 - 5 * Complex.I
def z2 : ℂ := 3 + 5 * Complex.I

theorem modulus_sum : Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := 
by 
  sorry

end modulus_sum_l2362_236238


namespace difference_of_fractions_l2362_236285

theorem difference_of_fractions (h₁ : 1/10 * 8000 = 800) (h₂ : (1/20) / 100 * 8000 = 4) : 800 - 4 = 796 :=
by
  sorry

end difference_of_fractions_l2362_236285


namespace maximum_area_of_rectangle_with_given_perimeter_l2362_236221

theorem maximum_area_of_rectangle_with_given_perimeter {x y : ℕ} (h₁ : 2 * x + 2 * y = 160) : 
  (∃ x y : ℕ, 2 * x + 2 * y = 160 ∧ x * y = 1600) := 
sorry

end maximum_area_of_rectangle_with_given_perimeter_l2362_236221


namespace max_value_expression_l2362_236216

theorem max_value_expression (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 1) (h2 : 0 < x1) (h3 : 0 < x2) (h4 : 0 < x3) :
    x1 * x2^2 * x3 + x1 * x2 * x3^2 ≤ 27 / 1024 :=
sorry

end max_value_expression_l2362_236216


namespace hash_value_is_minus_15_l2362_236245

def hash (a b c : ℝ) : ℝ := b^2 - 3 * a * c

theorem hash_value_is_minus_15 : hash 2 3 4 = -15 :=
by
  sorry

end hash_value_is_minus_15_l2362_236245


namespace min_sum_of_angles_l2362_236243

theorem min_sum_of_angles (A B C : ℝ) (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin B + Real.sin C ≤ 1) : 
  min (A + B) (min (B + C) (C + A)) < 30 := 
sorry

end min_sum_of_angles_l2362_236243


namespace value_of_f_at_2_and_neg_log2_3_l2362_236281

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^(-x)

theorem value_of_f_at_2_and_neg_log2_3 :
  f 2 * f (-Real.log 3 / Real.log 2) = 3 := by
  sorry

end value_of_f_at_2_and_neg_log2_3_l2362_236281


namespace initial_average_mark_l2362_236267

theorem initial_average_mark (A : ℝ) (n : ℕ) (excluded_avg remaining_avg : ℝ) :
  n = 25 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (A * n = (n - 5) * remaining_avg + 5 * excluded_avg) →
  A = 80 :=
by
  intros hn_hexcluded_avg hremaining_avg htotal_correct
  sorry

end initial_average_mark_l2362_236267


namespace find_x_for_parallel_vectors_l2362_236263

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (2, 2 - x)

def are_parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, are_parallel (a x) (b x) → x = 2/3 :=
by
  sorry

end find_x_for_parallel_vectors_l2362_236263


namespace max_saturdays_l2362_236203

theorem max_saturdays (days_in_month : ℕ) (month : string) (is_leap_year : Prop) (start_day : ℕ) : 
  (days_in_month = 29 → is_leap_year → start_day = 6 → true) ∧ -- February in a leap year starts on Saturday
  (days_in_month = 30 → (start_day = 5 ∨ start_day = 6) → true) ∧ -- 30-day months start on Friday or Saturday
  (days_in_month = 31 → (start_day = 4 ∨ start_day = 5 ∨ start_day = 6) → true) ∧ -- 31-day months start on Thursday, Friday, or Saturday
  (31 ≤ days_in_month ∧ days_in_month ≤ 28 → false) → -- Other case should be false
  ∃ n : ℕ, n = 5 := -- Maximum number of Saturdays is 5
sorry

end max_saturdays_l2362_236203


namespace total_distance_hiked_l2362_236298

def distance_car_to_stream : ℝ := 0.2
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

theorem total_distance_hiked : 
  distance_car_to_stream + distance_stream_to_meadow + distance_meadow_to_campsite = 0.7 := by
  sorry

end total_distance_hiked_l2362_236298


namespace no_n_in_range_l2362_236287

theorem no_n_in_range :
  ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 10467 % 7 := by
  sorry

end no_n_in_range_l2362_236287


namespace johns_speed_l2362_236252

def time1 : ℕ := 2
def time2 : ℕ := 3
def total_distance : ℕ := 225

def total_time : ℕ := time1 + time2

theorem johns_speed :
  (total_distance : ℝ) / (total_time : ℝ) = 45 :=
sorry

end johns_speed_l2362_236252


namespace cube_identity_l2362_236240

theorem cube_identity (a : ℝ) (h : (a + 1/a) ^ 2 = 3) : a^3 + 1/a^3 = 0 := 
by
  sorry

end cube_identity_l2362_236240


namespace geometric_mean_eq_6_l2362_236219

theorem geometric_mean_eq_6 (b c : ℝ) (hb : b = 3) (hc : c = 12) :
  (b * c) ^ (1/2 : ℝ) = 6 := 
by
  sorry

end geometric_mean_eq_6_l2362_236219


namespace odd_power_divisible_by_sum_l2362_236237

theorem odd_power_divisible_by_sum (x y : ℝ) (k : ℕ) (h : k > 0) :
  (x^((2*k - 1)) + y^((2*k - 1))) ∣ (x^(2*k + 1) + y^(2*k + 1)) :=
sorry

end odd_power_divisible_by_sum_l2362_236237


namespace avg_one_fourth_class_l2362_236283

variable (N : ℕ) (A : ℕ)
variable (h1 : ((N : ℝ) * 80) = (N / 4) * A + (3 * N / 4) * 76)

theorem avg_one_fourth_class : A = 92 :=
by
  sorry

end avg_one_fourth_class_l2362_236283


namespace valerie_needs_72_stamps_l2362_236227

noncomputable def total_stamps_needed : ℕ :=
  let thank_you_cards := 5
  let stamps_per_thank_you := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebates_more_than_bills := 3
  let rebate_stamps := 2
  let job_applications_factor := 2
  let job_application_stamps := 1

  let total_thank_you_stamps := thank_you_cards * stamps_per_thank_you
  let total_bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let total_rebates := total_bill_stamps + rebates_more_than_bills
  let total_rebate_stamps := total_rebates * rebate_stamps
  let total_job_applications := total_rebates * job_applications_factor
  let total_job_application_stamps := total_job_applications * job_application_stamps

  total_thank_you_stamps + total_bill_stamps + total_rebate_stamps + total_job_application_stamps

theorem valerie_needs_72_stamps : total_stamps_needed = 72 :=
  by
    sorry

end valerie_needs_72_stamps_l2362_236227


namespace minimum_value_frac_l2362_236201

theorem minimum_value_frac (x y z : ℝ) (h : 2 * x * y + y * z > 0) : 
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 :=
sorry

end minimum_value_frac_l2362_236201


namespace scientific_notation_of_15510000_l2362_236250

/--
Express 15,510,000 in scientific notation.

Theorem: 
Given that the scientific notation for large numbers is of the form \(a \times 10^n\) where \(1 \leq |a| < 10\),
prove that expressing 15,510,000 in scientific notation results in 1.551 × 10^7.
-/
theorem scientific_notation_of_15510000 : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 15510000 = a * 10 ^ n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_of_15510000_l2362_236250


namespace total_weight_l2362_236282

-- Define the weights of almonds and pecans.
def weight_almonds : ℝ := 0.14
def weight_pecans : ℝ := 0.38

-- Prove that the total weight of nuts is 0.52 kilograms.
theorem total_weight (almonds pecans : ℝ) (h_almonds : almonds = 0.14) (h_pecans : pecans = 0.38) :
  almonds + pecans = 0.52 :=
by
  sorry

end total_weight_l2362_236282


namespace asymptotes_of_C2_l2362_236202

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def C1 (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
noncomputable def C2 (x y : ℝ) : Prop := (y^2 / a^2 - x^2 / b^2 = 1)
noncomputable def ecc1 : ℝ := (Real.sqrt (a^2 - b^2)) / a
noncomputable def ecc2 : ℝ := (Real.sqrt (a^2 + b^2)) / a

theorem asymptotes_of_C2 :
  a > b → b > 0 → ecc1 * ecc2 = Real.sqrt 3 / 2 → by exact (∀ x y : ℝ, C2 x y → x = - Real.sqrt 2 * y ∨ x = Real.sqrt 2 * y) :=
sorry

end asymptotes_of_C2_l2362_236202


namespace find_gear_p_rpm_l2362_236244

def gear_p_rpm (r : ℕ) (gear_p_revs : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) : Prop :=
  r = gear_p_revs * 2

theorem find_gear_p_rpm (r : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) :
  gear_q_rpm = 40 ∧ time_seconds = 30 ∧ extra_revs_q_over_p = 15 ∧ gear_p_revs = 10 / 2 →
  r = 10 :=
by
  sorry

end find_gear_p_rpm_l2362_236244


namespace roses_cut_l2362_236247

def r_before := 13
def r_after := 14

theorem roses_cut : r_after - r_before = 1 := by
  sorry

end roses_cut_l2362_236247


namespace math_problem_l2362_236284

theorem math_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 35) : L = 1631 := 
by
  sorry

end math_problem_l2362_236284


namespace BethsHighSchoolStudents_l2362_236288

-- Define the variables
variables (B P : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := B = 4 * P
def condition2 : Prop := B + P = 5000

-- The theorem to be proved
theorem BethsHighSchoolStudents (h1 : condition1 B P) (h2 : condition2 B P) : B = 4000 :=
by
  -- Proof will be here
  sorry

end BethsHighSchoolStudents_l2362_236288


namespace exists_composite_arith_sequence_pairwise_coprime_l2362_236225

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem exists_composite_arith_sequence_pairwise_coprime (n : ℕ) : 
  ∃ seq : Fin n → ℕ, (∀ i, ∃ k, seq i = factorial n + k) ∧ 
  (∀ i j, i ≠ j → gcd (seq i) (seq j) = 1) :=
by
  sorry

end exists_composite_arith_sequence_pairwise_coprime_l2362_236225


namespace rates_of_interest_l2362_236274

theorem rates_of_interest (P_B P_C T_B T_C SI_B SI_C : ℝ) (R_B R_C : ℝ)
  (hB1 : P_B = 5000) (hB2: T_B = 5) (hB3: SI_B = 2200)
  (hC1 : P_C = 3000) (hC2 : T_C = 7) (hC3 : SI_C = 2730)
  (simple_interest : ∀ {P R T SI : ℝ}, SI = (P * R * T) / 100)
  : R_B = 8.8 ∧ R_C = 13 := by
  sorry

end rates_of_interest_l2362_236274
