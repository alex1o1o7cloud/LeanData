import Mathlib

namespace original_price_of_cycle_l2122_212232

/--
A man bought a cycle for some amount and sold it at a loss of 20%.
The selling price of the cycle is Rs. 1280.
What was the original price of the cycle?
-/
theorem original_price_of_cycle
    (loss_percent : ℝ)
    (selling_price : ℝ)
    (original_price : ℝ)
    (h_loss_percent : loss_percent = 0.20)
    (h_selling_price : selling_price = 1280)
    (h_selling_eqn : selling_price = (1 - loss_percent) * original_price) :
    original_price = 1600 :=
sorry

end original_price_of_cycle_l2122_212232


namespace simplify_expression_l2122_212291

theorem simplify_expression :
  ((5 ^ 7 + 2 ^ 8) * (1 ^ 5 - (-1) ^ 5) ^ 10) = 80263680 := by
  sorry

end simplify_expression_l2122_212291


namespace min_AB_plus_five_thirds_BF_l2122_212251

theorem min_AB_plus_five_thirds_BF 
  (A : ℝ × ℝ) (onEllipse : ℝ × ℝ → Prop) (F : ℝ × ℝ)
  (B : ℝ × ℝ) (minFunction : ℝ)
  (hf : F = (-3, 0)) (hA : A = (-2,2))
  (hB : onEllipse B) :
  (∀ B', onEllipse B' → (dist A B' + 5/3 * dist B' F) ≥ minFunction) →
  minFunction = (dist A B + 5/3 * dist B F) →
  B = (-(5 * Real.sqrt 3) / 2, 2) := by
  sorry

def onEllipse (B : ℝ × ℝ) : Prop := (B.1^2) / 25 + (B.2^2) / 16 = 1

end min_AB_plus_five_thirds_BF_l2122_212251


namespace remaining_students_l2122_212225

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l2122_212225


namespace find_b50_l2122_212280

noncomputable def T (n : ℕ) : ℝ := if n = 1 then 2 else 2 / (6 * n - 5)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 2 else T n - T (n - 1)

theorem find_b50 : b 50 = -6 / 42677.5 := by sorry

end find_b50_l2122_212280


namespace part1_proof_l2122_212248

def a : ℚ := 1 / 2
def b : ℚ := -2
def expr : ℚ := 2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b)

theorem part1_proof : expr = 5 := by
  unfold expr
  unfold a
  unfold b
  sorry

end part1_proof_l2122_212248


namespace proof_problem_l2122_212267

-- Definitions for the solution sets
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def intersection : Set ℝ := {x | -1 < x ∧ x < 2}

-- The quadratic inequality solution sets
def solution_set (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- The main theorem statement
theorem proof_problem (a b : ℝ) (h : solution_set a b = intersection) : a + b = -3 :=
sorry

end proof_problem_l2122_212267


namespace exists_between_elements_l2122_212230

noncomputable def M : Set ℝ :=
  { x | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ x = (m + n) / Real.sqrt (m^2 + n^2) }

theorem exists_between_elements (x y : ℝ) (hx : x ∈ M) (hy : y ∈ M) (hxy : x < y) :
  ∃ z ∈ M, x < z ∧ z < y :=
by
  sorry

end exists_between_elements_l2122_212230


namespace symmetric_line_b_value_l2122_212226

theorem symmetric_line_b_value (b : ℝ) : 
  (∃ l1 l2 : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), l1 (x, y) ↔ y = -2 * x + b) ∧ 
    (∃ p2 : ℝ × ℝ, p2 = (1, 6) ∧ l2 p2) ∧
    l2 (-1, 6) ∧ 
    (∀ (x y : ℝ), l1 (x, y) ↔ l2 (-x, y))) →
  b = 4 := 
by
  sorry

end symmetric_line_b_value_l2122_212226


namespace john_took_11_more_chickens_than_ray_l2122_212239

noncomputable def chickens_taken_by_john (mary_chickens : ℕ) : ℕ := mary_chickens + 5
noncomputable def chickens_taken_by_ray (mary_chickens : ℕ) : ℕ := mary_chickens - 6
def ray_chickens : ℕ := 10

-- The theorem to prove:
theorem john_took_11_more_chickens_than_ray :
  ∃ (mary_chickens : ℕ), chickens_taken_by_john mary_chickens - ray_chickens = 11 :=
by
  -- Initial assumptions and derivation steps should be provided here.
  sorry

end john_took_11_more_chickens_than_ray_l2122_212239


namespace proof_inequality_l2122_212217

noncomputable def proof_problem (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : Prop :=
  (1 - p^m)^n + (1 - q^n)^m ≥ 1

theorem proof_inequality (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
by
  sorry

end proof_inequality_l2122_212217


namespace matchsticks_left_l2122_212255

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l2122_212255


namespace trigonometric_identity_l2122_212242

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11 / 12 :=
by
  sorry

end trigonometric_identity_l2122_212242


namespace gcd_210_162_l2122_212283

-- Define the numbers
def a := 210
def b := 162

-- The proposition we need to prove: The GCD of 210 and 162 is 6
theorem gcd_210_162 : Nat.gcd a b = 6 :=
by
  sorry

end gcd_210_162_l2122_212283


namespace roots_sum_l2122_212244

theorem roots_sum (a b : ℝ) 
  (h₁ : 3^(a-1) = 6 - a)
  (h₂ : 3^(6-b) = b - 1) : 
  a + b = 7 := 
by sorry

end roots_sum_l2122_212244


namespace speed_of_faster_train_l2122_212279

noncomputable def speed_of_slower_train : ℝ := 36
noncomputable def length_of_each_train : ℝ := 70
noncomputable def time_to_pass : ℝ := 36

theorem speed_of_faster_train : 
  ∃ (V_f : ℝ), 
    (V_f - speed_of_slower_train) * (1000 / 3600) = 140 / time_to_pass ∧ 
    V_f = 50 :=
by {
  sorry
}

end speed_of_faster_train_l2122_212279


namespace willie_exchange_rate_l2122_212266

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end willie_exchange_rate_l2122_212266


namespace number_of_geese_more_than_ducks_l2122_212254

theorem number_of_geese_more_than_ducks (geese ducks : ℝ) (h1 : geese = 58.0) (h2 : ducks = 37.0) :
  geese - ducks = 21.0 :=
by
  sorry

end number_of_geese_more_than_ducks_l2122_212254


namespace equation_of_line_l_l2122_212238

-- Define the conditions for the parabola and the line
def parabola_vertex : Prop := 
  ∃ C : ℝ × ℝ, C = (0, 0)

def parabola_symmetry_axis : Prop := 
  ∃ l : ℝ → ℝ, ∀ x, l x = -1

def midpoint_of_AB (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

def parabola_equation (A B : ℝ × ℝ) : Prop :=
  A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1

-- State the theorem to be proven
theorem equation_of_line_l (A B : ℝ × ℝ) :
  parabola_vertex ∧ parabola_symmetry_axis ∧ midpoint_of_AB A B ∧ parabola_equation A B →
  ∃ l : ℝ → ℝ, ∀ x, l x = 2 * x - 3 :=
by sorry

end equation_of_line_l_l2122_212238


namespace income_percentage_less_l2122_212210

-- Definitions representing the conditions
variables (T M J : ℝ)
variables (h1 : M = 1.60 * T) (h2 : M = 1.12 * J)

-- The theorem stating the problem
theorem income_percentage_less : (100 - (T / J) * 100) = 30 :=
by
  sorry

end income_percentage_less_l2122_212210


namespace container_fullness_calc_l2122_212281

theorem container_fullness_calc (initial_percent : ℝ) (added_water : ℝ) (total_capacity : ℝ) (result_fraction : ℝ) :
  initial_percent = 0.3 →
  added_water = 27 →
  total_capacity = 60 →
  result_fraction = 3/4 →
  ((initial_percent * total_capacity + added_water) / total_capacity) = result_fraction :=
by
  intros h1 h2 h3 h4
  sorry

end container_fullness_calc_l2122_212281


namespace work_completion_time_extension_l2122_212211

theorem work_completion_time_extension
    (total_men : ℕ) (initial_days : ℕ) (remaining_men : ℕ) (man_days : ℕ) :
    total_men = 100 →
    initial_days = 20 →
    remaining_men = 50 →
    man_days = total_men * initial_days →
    (man_days / remaining_men) - initial_days = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end work_completion_time_extension_l2122_212211


namespace sheets_in_stack_l2122_212269

theorem sheets_in_stack (n : ℕ) (thickness : ℝ) (height : ℝ) 
  (h1 : n = 400) (h2 : thickness = 4) (h3 : height = 10) : 
  n * height / thickness = 1000 := 
by 
  sorry

end sheets_in_stack_l2122_212269


namespace percentage_of_valid_votes_l2122_212273

theorem percentage_of_valid_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℕ) 
  (candidate_valid_votes : ℕ)
  (percentage_invalid : invalid_percentage = 15)
  (total_votes_eq : total_votes = 560000)
  (candidate_votes_eq : candidate_valid_votes = 380800) 
  : (candidate_valid_votes : ℝ) / (total_votes * (0.85 : ℝ)) * 100 = 80 := 
by 
  sorry

end percentage_of_valid_votes_l2122_212273


namespace part1_part2_l2122_212294

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l2122_212294


namespace derivative_at_two_l2122_212202

def f (x : ℝ) : ℝ := x^3 + 4 * x - 5

noncomputable def derivative_f (x : ℝ) : ℝ := 3 * x^2 + 4

theorem derivative_at_two : derivative_f 2 = 16 :=
by
  sorry

end derivative_at_two_l2122_212202


namespace factorize_expression_l2122_212259

theorem factorize_expression (m : ℝ) : m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

end factorize_expression_l2122_212259


namespace solve_for_x_l2122_212258

theorem solve_for_x (x : ℝ) (d : ℝ) (h1 : x > 0) (h2 : x^2 = 4 + d) (h3 : 25 = x^2 + d) : x = Real.sqrt 14.5 := 
by 
  sorry

end solve_for_x_l2122_212258


namespace fraction_exponentiation_l2122_212265

theorem fraction_exponentiation : (3/4 : ℚ)^3 = 27/64 := by
  sorry

end fraction_exponentiation_l2122_212265


namespace remainder_div_29_l2122_212249

theorem remainder_div_29 (k : ℤ) (N : ℤ) (h : N = 899 * k + 63) : N % 29 = 10 :=
  sorry

end remainder_div_29_l2122_212249


namespace possible_six_digit_numbers_divisible_by_3_l2122_212221

theorem possible_six_digit_numbers_divisible_by_3 (missing_digit_condition : ∀ k : Nat, (8 + 5 + 5 + 2 + 2 + k) % 3 = 0) : 
  ∃ count : Nat, count = 13 := by
  sorry

end possible_six_digit_numbers_divisible_by_3_l2122_212221


namespace problem1_problem2_l2122_212241

-- Problem 1
theorem problem1 (x y : ℤ) (h1 : x = 2) (h2 : y = 2016) :
  (3*x + 2*y)*(3*x - 2*y) - (x + 2*y)*(5*x - 2*y) / (8*x) = -2015 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h1 : x = 2) :
  ((x - 3) / (x^2 - 1)) * ((x^2 + 2*x + 1) / (x - 3)) - (1 / (x - 1) + 1) = 1 :=
by
  sorry

end problem1_problem2_l2122_212241


namespace no_integer_regular_pentagon_l2122_212212

theorem no_integer_regular_pentagon 
  (x y : Fin 5 → ℤ) 
  (h_length : ∀ i j : Fin 5, i ≠ j → (x i - x j) ^ 2 + (y i - y j) ^ 2 = (x 0 - x 1) ^ 2 + (y 0 - y 1) ^ 2)
  : False :=
sorry

end no_integer_regular_pentagon_l2122_212212


namespace initial_erasers_in_box_l2122_212220

-- Definitions based on the conditions
def erasers_in_bag_jane := 15
def erasers_taken_out_doris := 54
def erasers_left_in_box := 15

-- Theorem statement
theorem initial_erasers_in_box : ∃ B_i : ℕ, B_i = erasers_taken_out_doris + erasers_left_in_box ∧ B_i = 69 :=
by
  use 69
  -- omitted proof steps
  sorry

end initial_erasers_in_box_l2122_212220


namespace percentage_increase_in_ear_piercing_l2122_212243

def cost_of_nose_piercing : ℕ := 20
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9
def total_amount_made : ℕ := 390

def cost_of_ear_piercing : ℕ := (total_amount_made - (noses_pierced * cost_of_nose_piercing)) / ears_pierced

def percentage_increase (original new : ℕ) : ℚ := ((new - original : ℚ) / original) * 100

theorem percentage_increase_in_ear_piercing : 
  percentage_increase cost_of_nose_piercing cost_of_ear_piercing = 50 := 
by 
  sorry

end percentage_increase_in_ear_piercing_l2122_212243


namespace num_squares_sharing_two_vertices_l2122_212285

-- Define the isosceles triangle and condition AB = AC
structure IsoscelesTriangle (A B C : Type) :=
  (AB AC : ℝ)
  (h_iso : AB = AC)

-- Define the problem statement in Lean
theorem num_squares_sharing_two_vertices 
  (A B C : Type) 
  (iso_tri : IsoscelesTriangle A B C) 
  (planeABC : ∀ P Q R : Type, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) :
  ∃ n : ℕ, n = 4 := sorry

end num_squares_sharing_two_vertices_l2122_212285


namespace hidden_message_is_correct_l2122_212286

def russian_alphabet_mapping : Char → Nat
| 'А' => 1
| 'Б' => 2
| 'В' => 3
| 'Г' => 4
| 'Д' => 5
| 'Е' => 6
| 'Ё' => 7
| 'Ж' => 8
| 'З' => 9
| 'И' => 10
| 'Й' => 11
| 'К' => 12
| 'Л' => 13
| 'М' => 14
| 'Н' => 15
| 'О' => 16
| 'П' => 17
| 'Р' => 18
| 'С' => 19
| 'Т' => 20
| 'У' => 21
| 'Ф' => 22
| 'Х' => 23
| 'Ц' => 24
| 'Ч' => 25
| 'Ш' => 26
| 'Щ' => 27
| 'Ъ' => 28
| 'Ы' => 29
| 'Ь' => 30
| 'Э' => 31
| 'Ю' => 32
| 'Я' => 33
| _ => 0

def prime_p : ℕ := 7 -- Assume some prime number p

def grid_position (p : ℕ) (k : ℕ) := p * k

theorem hidden_message_is_correct :
  ∃ m : String, m = "ПАРОЛЬ МЕДВЕЖАТА" :=
by
  let message := "ПАРОЛЬ МЕДВЕЖАТА"
  have h1 : russian_alphabet_mapping 'П' = 17 := by sorry
  have h2 : russian_alphabet_mapping 'А' = 1 := by sorry
  have h3 : russian_alphabet_mapping 'Р' = 18 := by sorry
  have h4 : russian_alphabet_mapping 'О' = 16 := by sorry
  have h5 : russian_alphabet_mapping 'Л' = 13 := by sorry
  have h6 : russian_alphabet_mapping 'Ь' = 29 := by sorry
  have h7 : russian_alphabet_mapping 'М' = 14 := by sorry
  have h8 : russian_alphabet_mapping 'Е' = 5 := by sorry
  have h9 : russian_alphabet_mapping 'Д' = 10 := by sorry
  have h10 : russian_alphabet_mapping 'В' = 3 := by sorry
  have h11 : russian_alphabet_mapping 'Ж' = 8 := by sorry
  have h12 : russian_alphabet_mapping 'Т' = 20 := by sorry
  have g1 : grid_position prime_p 17 = 119 := by sorry
  have g2 : grid_position prime_p 1 = 7 := by sorry
  have g3 : grid_position prime_p 18 = 126 := by sorry
  have g4 : grid_position prime_p 16 = 112 := by sorry
  have g5 : grid_position prime_p 13 = 91 := by sorry
  have g6 : grid_position prime_p 29 = 203 := by sorry
  have g7 : grid_position prime_p 14 = 98 := by sorry
  have g8 : grid_position prime_p 5 = 35 := by sorry
  have g9 : grid_position prime_p 10 = 70 := by sorry
  have g10 : grid_position prime_p 3 = 21 := by sorry
  have g11 : grid_position prime_p 8 = 56 := by sorry
  have g12 : grid_position prime_p 20 = 140 := by sorry
  existsi message
  rfl

end hidden_message_is_correct_l2122_212286


namespace problem_stmt_l2122_212264

variable (a b : ℝ)

theorem problem_stmt (ha : a > 0) (hb : b > 0) (a_plus_b : a + b = 2):
  3 * a^2 + b^2 ≥ 3 ∧ 4 / (a + 1) + 1 / b ≥ 3 := by
  sorry

end problem_stmt_l2122_212264


namespace greatest_divisor_four_consecutive_squared_l2122_212287

theorem greatest_divisor_four_consecutive_squared :
  ∀ (n: ℕ), ∃ m: ℕ, (∀ (n: ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) ∧ m = 144 := 
sorry

end greatest_divisor_four_consecutive_squared_l2122_212287


namespace triangle_altitude_from_rectangle_l2122_212292

theorem triangle_altitude_from_rectangle (a b : ℕ) (A : ℕ) (h : ℕ) (H1 : a = 7) (H2 : b = 21) (H3 : A = 147) (H4 : a * b = A) (H5 : 2 * A = h * b) : h = 14 :=
sorry

end triangle_altitude_from_rectangle_l2122_212292


namespace minimum_omega_l2122_212272

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l2122_212272


namespace G_at_16_l2122_212206

noncomputable def G : ℝ → ℝ := sorry

-- Condition 1: G is a polynomial, implicitly stated
-- Condition 2: Given G(8) = 21
axiom G_at_8 : G 8 = 21

-- Condition 3: Given that
axiom G_fraction_condition : ∀ (x : ℝ), 
  (x^2 + 6*x + 8) ≠ 0 ∧ ((x+4)*(x+2)) ≠ 0 → 
  (G (2*x) / G (x+4) = 4 - (16*x + 32) / (x^2 + 6*x + 8))

-- The problem: Prove G(16) = 90
theorem G_at_16 : G 16 = 90 := 
sorry

end G_at_16_l2122_212206


namespace triangle_angle_R_measure_l2122_212260

theorem triangle_angle_R_measure :
  ∀ (P Q R : ℝ),
  P + Q + R = 180 ∧ P = 70 ∧ Q = 2 * R + 15 → R = 95 / 3 :=
by
  intros P Q R h
  sorry

end triangle_angle_R_measure_l2122_212260


namespace ellipse_parabola_common_point_l2122_212250

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 := 
by 
  sorry

end ellipse_parabola_common_point_l2122_212250


namespace solution_set_quadratic_l2122_212247

theorem solution_set_quadratic (a x : ℝ) (h : a < 0) : 
  (x^2 - 2 * a * x - 3 * a^2 < 0) ↔ (3 * a < x ∧ x < -a) := 
by
  sorry

end solution_set_quadratic_l2122_212247


namespace decreasing_implies_b_geq_4_l2122_212219

-- Define the function and its derivative
def function (x : ℝ) (b : ℝ) : ℝ := x^3 - 3*b*x + 1

def derivative (x : ℝ) (b : ℝ) : ℝ := 3*x^2 - 3*b

theorem decreasing_implies_b_geq_4 (b : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → derivative x b ≤ 0) → b ≥ 4 :=
by
  intros h
  sorry

end decreasing_implies_b_geq_4_l2122_212219


namespace exp_monotonic_iff_l2122_212293

theorem exp_monotonic_iff (a b : ℝ) : (a > b) ↔ (Real.exp a > Real.exp b) :=
sorry

end exp_monotonic_iff_l2122_212293


namespace product_with_zero_is_zero_l2122_212227

theorem product_with_zero_is_zero :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 0) = 0 :=
by
  sorry

end product_with_zero_is_zero_l2122_212227


namespace Daisy_vs_Bess_l2122_212277

-- Define the conditions
def Bess_daily : ℕ := 2
def Brownie_multiple : ℕ := 3
def total_pails_per_week : ℕ := 77
def days_per_week : ℕ := 7

-- Define the weekly production for Bess
def Bess_weekly : ℕ := Bess_daily * days_per_week

-- Define the weekly production for Brownie
def Brownie_weekly : ℕ := Brownie_multiple * Bess_weekly

-- Farmer Red's total weekly milk production is the sum of Bess, Brownie, and Daisy's production
-- We need to prove the difference in weekly production between Daisy and Bess is 7 pails.
theorem Daisy_vs_Bess (Daisy_weekly : ℕ) (h : Bess_weekly + Brownie_weekly + Daisy_weekly = total_pails_per_week) :
  Daisy_weekly - Bess_weekly = 7 :=
by
  sorry

end Daisy_vs_Bess_l2122_212277


namespace mario_age_is_4_l2122_212256

-- Define the conditions
def sum_of_ages (mario maria : ℕ) : Prop := mario + maria = 7
def mario_older_by_one (mario maria : ℕ) : Prop := mario = maria + 1

-- State the theorem to prove Mario's age is 4 given the conditions
theorem mario_age_is_4 (mario maria : ℕ) (h1 : sum_of_ages mario maria) (h2 : mario_older_by_one mario maria) : mario = 4 :=
sorry -- Proof to be completed later

end mario_age_is_4_l2122_212256


namespace equation_of_line_AB_l2122_212289

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the center C
def C : Point := ⟨1, 0⟩

-- The equation of line AB we want to verify
def line_AB (P : Point) := P.x - P.y - 3 = 0

-- The theorem to prove
theorem equation_of_line_AB :
  (circle_eq P.x P.y ∧ P = ⟨2, -1⟩ ∧ C = ⟨1, 0⟩) → line_AB P :=
by
  sorry

end equation_of_line_AB_l2122_212289


namespace symmetric_point_coordinates_l2122_212295

-- Definition of symmetry in the Cartesian coordinate system
def is_symmetrical_about_origin (A A' : ℝ × ℝ) : Prop :=
  A'.1 = -A.1 ∧ A'.2 = -A.2

-- Given point A and its symmetric property to find point A'
theorem symmetric_point_coordinates (A A' : ℝ × ℝ)
  (hA : A = (1, -2))
  (h_symm : is_symmetrical_about_origin A A') :
  A' = (-1, 2) :=
by
  sorry -- Proof to be filled in (not required as per the instructions)

end symmetric_point_coordinates_l2122_212295


namespace gain_percent_is_150_l2122_212288

variable (C S : ℝ)
variable (h : 50 * C = 20 * S)

theorem gain_percent_is_150 (h : 50 * C = 20 * S) : ((S - C) / C) * 100 = 150 :=
by
  sorry

end gain_percent_is_150_l2122_212288


namespace find_a6_a7_l2122_212218

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Given Conditions
axiom cond1 : arithmetic_sequence a d
axiom cond2 : a 2 + a 4 + a 9 + a 11 = 32

-- Proof Problem
theorem find_a6_a7 : a 6 + a 7 = 16 :=
  sorry

end find_a6_a7_l2122_212218


namespace total_votes_l2122_212235

theorem total_votes (bob_votes total_votes : ℕ) (h1 : bob_votes = 48) (h2 : (2 : ℝ) / 5 * total_votes = bob_votes) :
  total_votes = 120 :=
by
  sorry

end total_votes_l2122_212235


namespace total_time_spent_l2122_212201

noncomputable def time_per_round : ℕ := 30
noncomputable def saturday_rounds : ℕ := 1 + 10
noncomputable def sunday_rounds : ℕ := 15
noncomputable def total_rounds : ℕ := saturday_rounds + sunday_rounds
noncomputable def total_time : ℕ := total_rounds * time_per_round

theorem total_time_spent :
  total_time = 780 := by sorry

end total_time_spent_l2122_212201


namespace points_can_move_on_same_line_l2122_212253

variable {A B C x y x' y' : ℝ}

def transform_x (x y : ℝ) : ℝ := 3 * x + 2 * y + 1
def transform_y (x y : ℝ) : ℝ := x + 4 * y - 3

noncomputable def points_on_same_line (A B C : ℝ) (x y : ℝ) : Prop :=
  A*x + B*y + C = 0 ∧
  A*(transform_x x y) + B*(transform_y x y) + C = 0

theorem points_can_move_on_same_line :
  ∃ (A B C : ℝ), ∀ (x y : ℝ), points_on_same_line A B C x y :=
sorry

end points_can_move_on_same_line_l2122_212253


namespace compound_interest_l2122_212231

variables {a r : ℝ}

theorem compound_interest (a r : ℝ) :
  (a * (1 + r)^10) = a * (1 + r)^(2020 - 2010) :=
by
  sorry

end compound_interest_l2122_212231


namespace solve_for_x_l2122_212222

theorem solve_for_x (x t : ℝ)
  (h₁ : t = 9)
  (h₂ : (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2) :
  x = 3 :=
by
  sorry

end solve_for_x_l2122_212222


namespace inlet_rate_480_l2122_212224

theorem inlet_rate_480 (capacity : ℕ) (T_outlet : ℕ) (T_outlet_inlet : ℕ) (R_i : ℕ) :
  capacity = 11520 →
  T_outlet = 8 →
  T_outlet_inlet = 12 →
  R_i = 480 :=
by
  intros
  sorry

end inlet_rate_480_l2122_212224


namespace winning_candidate_percentage_l2122_212282

theorem winning_candidate_percentage (P: ℝ) (majority diff votes totalVotes : ℝ)
    (h1 : majority = 184)
    (h2 : totalVotes = 460)
    (h3 : diff = P * totalVotes / 100 - (100 - P) * totalVotes / 100)
    (h4 : majority = diff) : P = 70 :=
by
  sorry

end winning_candidate_percentage_l2122_212282


namespace min_dist_of_PQ_l2122_212236

open Real

theorem min_dist_of_PQ :
  ∀ (P Q : ℝ × ℝ),
    (P.fst - 3)^2 + (P.snd + 1)^2 = 4 →
    Q.fst = -3 →
    ∃ (min_dist : ℝ), min_dist = 4 :=
by
  sorry

end min_dist_of_PQ_l2122_212236


namespace maxRegions_four_planes_maxRegions_n_planes_l2122_212278

noncomputable def maxRegions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

theorem maxRegions_four_planes : maxRegions 4 = 11 := by
  sorry

theorem maxRegions_n_planes (n : ℕ) : maxRegions n = 1 + (n * (n + 1)) / 2 := by
  sorry

end maxRegions_four_planes_maxRegions_n_planes_l2122_212278


namespace perfect_square_pairs_l2122_212200

-- Definition of a perfect square
def is_perfect_square (k : ℕ) : Prop :=
∃ (n : ℕ), n * n = k

-- Main theorem statement
theorem perfect_square_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_perfect_square ((2^m - 1) * (2^n - 1)) ↔ (m = n) ∨ (m = 3 ∧ n = 6) ∨ (m = 6 ∧ n = 3) :=
sorry

end perfect_square_pairs_l2122_212200


namespace constant_function_of_inequality_l2122_212271

theorem constant_function_of_inequality (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_inequality_l2122_212271


namespace commodity_price_l2122_212228

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l2122_212228


namespace max_three_cards_l2122_212213

theorem max_three_cards (n m p : ℕ) (h : n + m + p = 8) (sum : 3 * n + 4 * m + 5 * p = 33) 
  (n_le_10 : n ≤ 10) (m_le_10 : m ≤ 10) (p_le_10 : p ≤ 10) : n ≤ 3 := 
sorry

end max_three_cards_l2122_212213


namespace find_m_l2122_212204

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m - 1) > 0) ∧ m = 3 :=
sorry

end find_m_l2122_212204


namespace intersection_of_A_and_B_l2122_212229

def A (x : ℝ) : Prop := x^2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_of_A_and_B_l2122_212229


namespace select_medical_team_l2122_212274

open Nat

theorem select_medical_team : 
  let male_doctors := 5
  let female_doctors := 4
  let selected_doctors := 3
  (male_doctors.choose 1 * female_doctors.choose 2 + male_doctors.choose 2 * female_doctors.choose 1) = 70 :=
by
  sorry

end select_medical_team_l2122_212274


namespace number_of_turns_to_wind_tape_l2122_212290

theorem number_of_turns_to_wind_tape (D δ L : ℝ) 
(hD : D = 22) 
(hδ : δ = 0.018) 
(hL : L = 90000) : 
∃ n : ℕ, n = 791 := 
sorry

end number_of_turns_to_wind_tape_l2122_212290


namespace taimour_paint_time_l2122_212298

theorem taimour_paint_time (T : ℝ) :
  (1 / T + 2 / T) * 7 = 1 → T = 21 :=
by
  intro h
  sorry

end taimour_paint_time_l2122_212298


namespace sum_of_reciprocals_l2122_212275

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 := 
sorry

end sum_of_reciprocals_l2122_212275


namespace inequality_solution_l2122_212234

open Set

theorem inequality_solution :
  {x : ℝ | |x + 1| - 2 > 0} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end inequality_solution_l2122_212234


namespace colton_stickers_final_count_l2122_212268

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end colton_stickers_final_count_l2122_212268


namespace greatest_sum_of_consecutive_integers_l2122_212262

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l2122_212262


namespace find_w_when_x_is_six_l2122_212270

variable {x w : ℝ}
variable (h1 : x = 3)
variable (h2 : w = 16)
variable (h3 : ∀ (x w : ℝ), x^4 * w^(1 / 4) = 162)

theorem find_w_when_x_is_six : x = 6 → w = 1 / 4096 :=
by
  intro hx
  sorry

end find_w_when_x_is_six_l2122_212270


namespace range_of_solutions_l2122_212216

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l2122_212216


namespace sin_2x_plus_one_equals_9_over_5_l2122_212214

theorem sin_2x_plus_one_equals_9_over_5 (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin (2 * x) + 1 = 9 / 5 :=
sorry

end sin_2x_plus_one_equals_9_over_5_l2122_212214


namespace major_minor_axis_lengths_foci_vertices_coordinates_l2122_212245

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

-- Proof Tasks
theorem major_minor_axis_lengths : 
  (∃ a b : ℝ, a = 5 ∧ b = 4 ∧ 2 * a = 10) :=
by sorry

theorem foci_vertices_coordinates : 
  (∃ c : ℝ, 
    (c = 3) ∧ 
    (∀ x y : ℝ, ellipse_eq x y → (x = 0 → y = 4 ∨ y = -4) ∧ (y = 0 → x = 5 ∨ x = -5))) :=
by sorry

end major_minor_axis_lengths_foci_vertices_coordinates_l2122_212245


namespace cost_of_largest_pot_l2122_212299

theorem cost_of_largest_pot
    (x : ℝ)
    (hx : 6 * x + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) = 8.25) :
    (x + 0.5) = 1.625 :=
sorry

end cost_of_largest_pot_l2122_212299


namespace total_payroll_calc_l2122_212207

theorem total_payroll_calc
  (h : ℕ := 129)          -- pay per day for heavy operators
  (l : ℕ := 82)           -- pay per day for general laborers
  (n : ℕ := 31)           -- total number of people hired
  (g : ℕ := 1)            -- number of general laborers employed
  : (h * (n - g) + l * g) = 3952 := 
by
  sorry

end total_payroll_calc_l2122_212207


namespace frances_towel_weight_in_ounces_l2122_212257

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l2122_212257


namespace tangent_line_circle_l2122_212237

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ, (x + y = 0) → ((x - m)^2 + y^2 = 2)) : m = 2 :=
sorry

end tangent_line_circle_l2122_212237


namespace middle_part_of_proportion_l2122_212208

theorem middle_part_of_proportion (x : ℚ) (h : x + (1/4) * x + (1/8) * x = 104) : (1/4) * x = 208 / 11 :=
by
  sorry

end middle_part_of_proportion_l2122_212208


namespace triangle_area_r_l2122_212297

theorem triangle_area_r (r : ℝ) (h₁ : 12 ≤ (r - 3) ^ (3 / 2)) (h₂ : (r - 3) ^ (3 / 2) ≤ 48) : 15 ≤ r ∧ r ≤ 19 := by
  sorry

end triangle_area_r_l2122_212297


namespace necessary_not_sufficient_condition_l2122_212246

variable {x : ℝ}

theorem necessary_not_sufficient_condition (h : x > 2) : x > 1 :=
by
  sorry

end necessary_not_sufficient_condition_l2122_212246


namespace deposit_time_l2122_212276

theorem deposit_time (r t : ℕ) : 
  8000 + 8000 * r * t / 100 = 10200 → 
  8000 + 8000 * (r + 2) * t / 100 = 10680 → 
  t = 3 :=
by 
  sorry

end deposit_time_l2122_212276


namespace staplers_left_l2122_212263

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l2122_212263


namespace flag_count_l2122_212240

def colors := 3

def stripes := 3

noncomputable def number_of_flags (colors stripes : ℕ) : ℕ :=
  colors ^ stripes

theorem flag_count : number_of_flags colors stripes = 27 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end flag_count_l2122_212240


namespace find_other_percentage_l2122_212233

noncomputable def percentage_other_investment
  (total_investment : ℝ)
  (investment_10_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_10_percent : ℝ)
  (other_investment_interest : ℝ) : ℝ :=
  let interest_10_percent := investment_10_percent * interest_rate_10_percent
  let interest_other_investment := total_interest - interest_10_percent
  let amount_other_percentage := total_investment - investment_10_percent
  interest_other_investment / amount_other_percentage

theorem find_other_percentage :
  ∀ (total_investment : ℝ)
    (investment_10_percent : ℝ)
    (total_interest : ℝ)
    (interest_rate_10_percent : ℝ),
    total_investment = 31000 ∧
    investment_10_percent = 12000 ∧
    total_interest = 1390 ∧
    interest_rate_10_percent = 0.1 →
    percentage_other_investment total_investment investment_10_percent total_interest interest_rate_10_percent 190 = 0.01 :=
by
  intros total_investment investment_10_percent total_interest interest_rate_10_percent h
  sorry

end find_other_percentage_l2122_212233


namespace alice_password_prob_correct_l2122_212203

noncomputable def password_probability : ℚ :=
  let even_digit_prob := 5 / 10
  let valid_symbol_prob := 3 / 5
  let non_zero_digit_prob := 9 / 10
  even_digit_prob * valid_symbol_prob * non_zero_digit_prob

theorem alice_password_prob_correct :
  password_probability = 27 / 100 := by
  rfl

end alice_password_prob_correct_l2122_212203


namespace degrees_to_radians_l2122_212261

theorem degrees_to_radians : (800 : ℝ) * (Real.pi / 180) = (40 / 9) * Real.pi :=
by
  sorry

end degrees_to_radians_l2122_212261


namespace rods_in_mile_l2122_212223

theorem rods_in_mile (mile_to_furlongs : 1 = 12) (furlong_to_rods : 1 = 50) : 1 * 12 * 50 = 600 :=
by
  sorry

end rods_in_mile_l2122_212223


namespace quadratic_always_positive_l2122_212296

theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) (hpos : a > 0) (hdisc : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := 
by
  sorry

end quadratic_always_positive_l2122_212296


namespace product_of_two_integers_l2122_212252

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 18) (h2 : x^2 - y^2 = 36) : x * y = 80 :=
by
  sorry

end product_of_two_integers_l2122_212252


namespace problem_statement_l2122_212205

variable (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ)

theorem problem_statement
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 + 36 * y6 + 49 * y7 + 64 * y8 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 + 49 * y6 + 64 * y7 + 81 * y8 = 15)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 + 64 * y6 + 81 * y7 + 100 * y8 = 140) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 + 81 * y6 + 100 * y7 + 121 * y8 = 472 := by
  sorry

end problem_statement_l2122_212205


namespace sequence_value_l2122_212209

theorem sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a n * a (n + 2) = a (n + 1) ^ 2)
  (h2 : a 7 = 16)
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := 
sorry

end sequence_value_l2122_212209


namespace arithmetic_sequence_sum_q_l2122_212284

theorem arithmetic_sequence_sum_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 2) + a (n + 1) = 2 * a n)
  (hq : q ≠ 1) :
  S 5 = 11 :=
sorry

end arithmetic_sequence_sum_q_l2122_212284


namespace iron_per_horseshoe_l2122_212215

def num_farms := 2
def num_horses_per_farm := 2
def num_stables := 2
def num_horses_per_stable := 5
def num_horseshoes_per_horse := 4
def iron_available := 400
def num_horses_riding_school := 36

-- Lean theorem statement
theorem iron_per_horseshoe : 
  (iron_available / (num_farms * num_horses_per_farm * num_horseshoes_per_horse 
  + num_stables * num_horses_per_stable * num_horseshoes_per_horse 
  + num_horses_riding_school * num_horseshoes_per_horse)) = 2 := 
by 
  sorry

end iron_per_horseshoe_l2122_212215
