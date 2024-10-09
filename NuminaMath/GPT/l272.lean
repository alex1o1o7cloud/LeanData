import Mathlib

namespace find_m_l272_27244

noncomputable def m_value (a b c d : Int) (Y : Int) : Int :=
  let l1_1 := a + b
  let l1_2 := b + c
  let l1_3 := c + d
  let l2_1 := l1_1 + l1_2
  let l2_2 := l1_2 + l1_3
  let l3 := l2_1 + l2_2
  if l3 = Y then a else 0

theorem find_m : m_value m 6 (-3) 4 20 = 7 := sorry

end find_m_l272_27244


namespace oranges_packed_in_a_week_l272_27261

open Nat

def oranges_per_box : Nat := 15
def boxes_per_day : Nat := 2150
def days_per_week : Nat := 7

theorem oranges_packed_in_a_week : oranges_per_box * boxes_per_day * days_per_week = 225750 :=
  sorry

end oranges_packed_in_a_week_l272_27261


namespace marker_cost_is_13_l272_27273

theorem marker_cost_is_13 :
  ∃ s m c : ℕ, (s > 20) ∧ (m ≥ 4) ∧ (c > m) ∧ (s * c * m = 3185) ∧ (c = 13) :=
by
  sorry

end marker_cost_is_13_l272_27273


namespace solve_for_x_l272_27220

theorem solve_for_x (x : ℕ) (h : x + 1 = 4) : x = 3 :=
by
  sorry

end solve_for_x_l272_27220


namespace correct_completion_at_crossroads_l272_27241

theorem correct_completion_at_crossroads :
  (∀ (s : String), 
    s = "An accident happened at a crossroads a few meters away from a bank" → 
    (∃ (general_sense : Bool), general_sense = tt)) :=
by
  sorry

end correct_completion_at_crossroads_l272_27241


namespace division_value_l272_27278

theorem division_value (x : ℝ) (h1 : 2976 / x - 240 = 8) : x = 12 := 
by
  sorry

end division_value_l272_27278


namespace problem_a2_sub_b2_problem_a_mul_b_l272_27292

theorem problem_a2_sub_b2 {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
sorry

theorem problem_a_mul_b {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a * b = 12 :=
sorry

end problem_a2_sub_b2_problem_a_mul_b_l272_27292


namespace brown_eyed_brunettes_l272_27275

theorem brown_eyed_brunettes (total_girls blondes brunettes blue_eyed_blondes brown_eyed_girls : ℕ) 
    (h1 : total_girls = 60) 
    (h2 : blondes + brunettes = total_girls) 
    (h3 : blue_eyed_blondes = 20) 
    (h4 : brunettes = 35) 
    (h5 : brown_eyed_girls = 22) 
    (h6 : blondes = total_girls - brunettes) 
    (h7 : brown_eyed_blondes = blondes - blue_eyed_blondes) :
  brunettes - (brown_eyed_girls - brown_eyed_blondes) = 17 :=
by sorry  -- Proof is not required

end brown_eyed_brunettes_l272_27275


namespace boys_without_calculators_l272_27289

-- Definitions based on the conditions
def total_boys : Nat := 20
def students_with_calculators : Nat := 26
def girls_with_calculators : Nat := 15

-- We need to prove the number of boys who did not bring their calculators.
theorem boys_without_calculators : (total_boys - (students_with_calculators - girls_with_calculators)) = 9 :=
by {
    -- Proof goes here
    sorry
}

end boys_without_calculators_l272_27289


namespace tshirts_equation_l272_27201

theorem tshirts_equation (x : ℝ) 
    (hx : x > 0)
    (march_cost : ℝ := 120000)
    (april_cost : ℝ := 187500)
    (april_increase : ℝ := 1.4)
    (cost_increase : ℝ := 5) :
    120000 / x + 5 = 187500 / (1.4 * x) :=
by 
  sorry

end tshirts_equation_l272_27201


namespace arithmetic_expression_evaluation_l272_27272

theorem arithmetic_expression_evaluation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := 
by
  sorry

end arithmetic_expression_evaluation_l272_27272


namespace expand_binomial_l272_27228

theorem expand_binomial (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5 * x - 24 :=
by
  sorry

end expand_binomial_l272_27228


namespace find_p_l272_27274

variable (f w : ℂ) (p : ℂ)
variable (h1 : f = 4)
variable (h2 : w = 10 + 200 * Complex.I)
variable (h3 : f * p - w = 20000)

theorem find_p : p = 5002.5 + 50 * Complex.I := by
  sorry

end find_p_l272_27274


namespace multiply_decimals_l272_27257

noncomputable def real_num_0_7 : ℝ := 7 * 10⁻¹
noncomputable def real_num_0_3 : ℝ := 3 * 10⁻¹
noncomputable def real_num_0_21 : ℝ := 0.21

theorem multiply_decimals :
  real_num_0_7 * real_num_0_3 = real_num_0_21 :=
sorry

end multiply_decimals_l272_27257


namespace floor_inequality_solution_set_l272_27255

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x.
    Prove that the solution set of the inequality ⌊x⌋² - 5⌊x⌋ - 36 ≤ 0 is {x | -4 ≤ x < 10}. -/
theorem floor_inequality_solution_set (x : ℝ) :
  (⌊x⌋^2 - 5 * ⌊x⌋ - 36 ≤ 0) ↔ -4 ≤ x ∧ x < 10 := by
    sorry

end floor_inequality_solution_set_l272_27255


namespace find_N_aN_bN_cN_dN_eN_l272_27252

theorem find_N_aN_bN_cN_dN_eN:
  ∃ (a b c d e : ℝ) (N : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 = 1000) ∧
    (N = c * (a + 3 * b + 4 * d + 6 * e)) ∧
    (N + a + b + c + d + e = 150 + 250 * Real.sqrt 62 + 10 * Real.sqrt 50) := by
  sorry

end find_N_aN_bN_cN_dN_eN_l272_27252


namespace cricket_bat_profit_percentage_l272_27279

theorem cricket_bat_profit_percentage 
  (selling_price profit : ℝ) 
  (h_sp: selling_price = 850) 
  (h_p: profit = 230) : 
  (profit / (selling_price - profit) * 100) = 37.10 :=
by
  sorry

end cricket_bat_profit_percentage_l272_27279


namespace sum_of_integers_l272_27243

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 120) (h2 : (m - 1) * m * (m + 1) = 120) : 
  (n + (n + 1) + (m - 1) + m + (m + 1)) = 36 :=
by
  sorry

end sum_of_integers_l272_27243


namespace problem_solution_l272_27203

theorem problem_solution :
  ∀ (x y : ℚ), 
  4 * x + y = 20 ∧ x + 2 * y = 17 → 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := 
by 
  sorry

end problem_solution_l272_27203


namespace captain_and_vicecaptain_pair_boys_and_girls_l272_27230

-- Problem A
theorem captain_and_vicecaptain (n : ℕ) (h : n = 11) : ∃ ways : ℕ, ways = 110 :=
by
  sorry

-- Problem B
theorem pair_boys_and_girls (N : ℕ) : ∃ ways : ℕ, ways = Nat.factorial N :=
by
  sorry

end captain_and_vicecaptain_pair_boys_and_girls_l272_27230


namespace faye_pencils_l272_27207

theorem faye_pencils :
  ∀ (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) (total_pencils pencils_per_row : ℕ),
  packs = 35 →
  pencils_per_pack = 4 →
  rows = 70 →
  total_pencils = packs * pencils_per_pack →
  pencils_per_row = total_pencils / rows →
  pencils_per_row = 2 :=
by
  intros packs pencils_per_pack rows total_pencils pencils_per_row
  intros packs_eq pencils_per_pack_eq rows_eq total_pencils_eq pencils_per_row_eq
  sorry

end faye_pencils_l272_27207


namespace sequence_sum_l272_27245

theorem sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
    (h1 : a 1 = 1)
    (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1))
    (h6_2 : a 6 = a 2) :
    a 2016 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_sum_l272_27245


namespace min_max_value_l272_27238

theorem min_max_value
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) (h₄ : 0 ≤ x₄) (h₅ : 0 ≤ x₅)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 1) :
  (min (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) = 1 / 3) :=
sorry

end min_max_value_l272_27238


namespace arithmetic_sequence_general_formula_l272_27281

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, 0 < n → (a n - 2 * a (n + 1) + a (n + 2) = 0)) : ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l272_27281


namespace vector_addition_l272_27267

variable {𝕍 : Type} [AddCommGroup 𝕍] [Module ℝ 𝕍]
variable (a b : 𝕍)

theorem vector_addition : 
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by
  sorry

end vector_addition_l272_27267


namespace no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l272_27237

theorem no_nat_nums_x4_minus_y4_eq_x3_plus_y3 : ∀ (x y : ℕ), x^4 - y^4 ≠ x^3 + y^3 :=
by
  intro x y
  sorry

end no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l272_27237


namespace average_salary_correct_l272_27234

/-- The salaries of A, B, C, D, and E. -/
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

/-- The number of people. -/
def number_of_people : ℕ := 5

/-- The total salary is the sum of the salaries. -/
def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

/-- The average salary is the total salary divided by the number of people. -/
def average_salary : ℕ := total_salary / number_of_people

/-- The average salary of A, B, C, D, and E is Rs. 8000. -/
theorem average_salary_correct : average_salary = 8000 := by
  sorry

end average_salary_correct_l272_27234


namespace matrix_inverse_eq_l272_27209

theorem matrix_inverse_eq (d k : ℚ) (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (hA : A = ![![1, 4], ![6, d]]) 
  (hA_inv : A⁻¹ = k • A) :
  (d, k) = (-1, 1/25) :=
  sorry

end matrix_inverse_eq_l272_27209


namespace benjamin_skating_time_l272_27298

-- Defining the conditions
def distance : ℕ := 80 -- Distance in kilometers
def speed : ℕ := 10   -- Speed in kilometers per hour

-- The main theorem statement
theorem benjamin_skating_time : ∀ (T : ℕ), T = distance / speed → T = 8 := by
  sorry

end benjamin_skating_time_l272_27298


namespace valentines_count_l272_27216

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 52) : x * y = 108 :=
by sorry

end valentines_count_l272_27216


namespace inscribed_sphere_radius_l272_27262

variable (a b r : ℝ)

theorem inscribed_sphere_radius (ha : 0 < a) (hb : 0 < b) (hr : 0 < r)
 (h : ∃ A B C D : ℝˣ, true) : r < (a * b) / (2 * (a + b)) := 
sorry

end inscribed_sphere_radius_l272_27262


namespace diff_between_roots_l272_27200

theorem diff_between_roots (p : ℝ) (r s : ℝ)
  (h_eq : ∀ x : ℝ, x^2 - (p+1)*x + (p^2 + 2*p - 3)/4 = 0 → x = r ∨ x = s)
  (h_ge : r ≥ s) :
  r - s = Real.sqrt (2*p + 1 - p^2) := by
  sorry

end diff_between_roots_l272_27200


namespace alyona_final_balances_l272_27246

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end alyona_final_balances_l272_27246


namespace number_of_children_l272_27210

theorem number_of_children (C : ℝ) 
  (h1 : 0.30 * C >= 0)
  (h2 : 0.20 * C >= 0)
  (h3 : 0.50 * C >= 0)
  (h4 : 0.70 * C = 42) : 
  C = 60 := by
  sorry

end number_of_children_l272_27210


namespace real_root_in_interval_l272_27217

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_in_interval : ∃ α : ℝ, f α = 0 ∧ 1 < α ∧ α < 2 :=
sorry

end real_root_in_interval_l272_27217


namespace num_pairs_eq_12_l272_27286

theorem num_pairs_eq_12 :
  ∃ (n : ℕ), (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧
    (a + 1/b : ℚ) / (1/a + b : ℚ) = 7 ↔ (7 * b = a)) ∧ n = 12 :=
sorry

end num_pairs_eq_12_l272_27286


namespace line_AB_eq_x_plus_3y_zero_l272_27242

variable (x y : ℝ)

def circle1 := x^2 + y^2 - 4*x + 6*y = 0
def circle2 := x^2 + y^2 - 6*x = 0

theorem line_AB_eq_x_plus_3y_zero : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B)) → 
  (∀ (x y : ℝ), x + 3*y = 0) := 
by
  sorry

end line_AB_eq_x_plus_3y_zero_l272_27242


namespace solve_equation_correctly_l272_27283

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end solve_equation_correctly_l272_27283


namespace triangle_lines_l272_27264

/-- Given a triangle with vertices A(1, 2), B(-1, 4), and C(4, 5):
  1. The equation of the line l₁ containing the altitude from A to side BC is 5x + y - 7 = 0.
  2. The equation of the line l₂ passing through C such that the distances from A and B to l₂ are equal
     is either x + y - 9 = 0 or x - 2y + 6 = 0. -/
theorem triangle_lines (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 4))
  (hC : C = (4, 5)) :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
  (l₁ = (5, 1, -7)) ∧
  ((l₂ = (1, 1, -9)) ∨ (l₂ = (1, -2, 6))) := by
  sorry

end triangle_lines_l272_27264


namespace ratio_problem_l272_27284

theorem ratio_problem (a b c d : ℚ) (h1 : a / b = 5 / 4) (h2 : c / d = 4 / 1) (h3 : d / b = 1 / 8) :
  a / c = 5 / 2 := by
  sorry

end ratio_problem_l272_27284


namespace arithmetic_sequence_property_l272_27260

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ((a 6 - 1)^3 + 2013 * (a 6 - 1)^3 = 1))
  (h2 : ((a 2008 - 1)^3 = -2013 * (a 2008 - 1)^3))
  (sum_formula : ∀ n, S n = n * a n) : 
  S 2013 = 2013 ∧ a 2008 < a 6 := 
sorry

end arithmetic_sequence_property_l272_27260


namespace probability_p_s_multiple_of_7_l272_27218

section
variables (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 60) (h2 : 1 ≤ b ∧ b ≤ 60) (h3 : a ≠ b)

theorem probability_p_s_multiple_of_7 :
  (∃ k : ℕ, a * b + a + b = 7 * k) → (64 / 1770 : ℚ) = 32 / 885 :=
sorry
end

end probability_p_s_multiple_of_7_l272_27218


namespace bridge_length_l272_27249

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 60 -- in km/hr
noncomputable def crossing_time : ℝ := 20 -- in seconds

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600 -- converting to m/s

noncomputable def total_distance_covered : ℝ := train_speed_ms * crossing_time -- distance covered in 20 seconds

theorem bridge_length : total_distance_covered - train_length = 83.4 :=
by
  -- The proof would go here
  sorry

end bridge_length_l272_27249


namespace arithmetic_sequence_term_number_l272_27205

theorem arithmetic_sequence_term_number :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (a 1 = 1) →
    (∀ m, a (m + 1) = a m + 3) →
    (a n = 2014) →
    n = 672 :=
by
  -- conditions
  intro a n h1 h2 h3
  -- proof skipped
  sorry

end arithmetic_sequence_term_number_l272_27205


namespace pq_sufficient_but_not_necessary_condition_l272_27263

theorem pq_sufficient_but_not_necessary_condition (p q : Prop) (hpq : p ∧ q) :
  ¬¬p = p :=
by
  sorry

end pq_sufficient_but_not_necessary_condition_l272_27263


namespace value_of_x_plus_y_l272_27232

theorem value_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 4) (h3 : x * y > 0) : x + y = 7 ∨ x + y = -7 :=
by
  sorry

end value_of_x_plus_y_l272_27232


namespace computer_table_cost_price_l272_27277

theorem computer_table_cost_price (CP SP : ℝ) (h1 : SP = CP * (124 / 100)) (h2 : SP = 8091) :
  CP = 6525 :=
by
  sorry

end computer_table_cost_price_l272_27277


namespace sector_area_l272_27295

theorem sector_area (arc_length radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) : 
  (1/2) * arc_length * radius = 2 :=
by
  -- sorry placeholder for proof
  sorry

end sector_area_l272_27295


namespace arithmetic_sequence_product_l272_27214

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) (h1 : ∀ n m, n < m → b n < b m) 
(h2 : ∀ n, b (n + 1) - b n = d) (h3 : b 3 * b 4 = 18) : b 2 * b 5 = -80 :=
sorry

end arithmetic_sequence_product_l272_27214


namespace line_equation_correct_l272_27299

-- Definitions for the conditions
def point := ℝ × ℝ
def vector := ℝ × ℝ

-- Given the line has a direction vector and passes through a point
def line_has_direction_vector (l : point → Prop) (v : vector) : Prop :=
  ∀ p₁ p₂ : point, l p₁ → l p₂ → (p₂.1 - p₁.1, p₂.2 - p₁.2) = v

def line_passes_through_point (l : point → Prop) (p : point) : Prop :=
  l p

-- The line equation in point-direction form
def line_equation (x y : ℝ) : Prop :=
  (x - 1) / 2 = y / -3

-- Main statement
theorem line_equation_correct :
  ∃ l : point → Prop, 
    line_has_direction_vector l (2, -3) ∧
    line_passes_through_point l (1, 0) ∧
    ∀ x y, l (x, y) ↔ line_equation x y := 
sorry

end line_equation_correct_l272_27299


namespace find_n_expansion_l272_27204

theorem find_n_expansion : 
  (∃ n : ℕ, 4^n + 2^n = 1056) → n = 5 :=
by sorry

end find_n_expansion_l272_27204


namespace base7_to_base10_245_l272_27222

theorem base7_to_base10_245 : (2 * 7^2 + 4 * 7^1 + 5 * 7^0) = 131 := by
  sorry

end base7_to_base10_245_l272_27222


namespace polar_line_equation_l272_27226

/-- A line that passes through a given point in polar coordinates and is parallel to the polar axis
    has a specific polar coordinate equation. -/
theorem polar_line_equation (r : ℝ) (θ : ℝ) (h : r = 6 ∧ θ = π / 6) : θ = π / 6 :=
by
  /- We are given that the line passes through the point \(C(6, \frac{\pi}{6})\) which means
     \(r = 6\) and \(θ = \frac{\pi}{6}\). Since the line is parallel to the polar axis, 
     the angle \(θ\) remains the same. Therefore, the polar coordinate equation of the line 
     is simply \(θ = \frac{\pi}{6}\). -/
  sorry

end polar_line_equation_l272_27226


namespace total_valid_votes_l272_27269

theorem total_valid_votes (V : ℝ)
  (h1 : ∃ c1 c2 : ℝ, c1 = 0.70 * V ∧ c2 = 0.30 * V)
  (h2 : ∀ c1 c2, c1 - c2 = 182) : V = 455 :=
sorry

end total_valid_votes_l272_27269


namespace factorize_expression_l272_27280

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l272_27280


namespace circle_equation_and_shortest_chord_l272_27202

-- Definitions based on given conditions
def point_P : ℝ × ℝ := (4, -1)
def line_l1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line_l2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- The circle should be such that it intersects line l1 at point P and its center lies on line l2
theorem circle_equation_and_shortest_chord 
  (C : ℝ × ℝ) (r : ℝ) (hC_l2 : line_l2 C.1 C.2)
  (h_intersect : ∃ (k : ℝ), point_P.1 = (C.1 + k * (C.1 - point_P.1)) ∧ point_P.2 = (C.2 + k * (C.2 - point_P.2))) :
  -- Proving (1): Equation of the circle
  ((C.1 = 3) ∧ (C.2 = 5) ∧ r^2 = 37) ∧
  -- Proving (2): Length of the shortest chord through the origin is 2 * sqrt(3)
  (2 * Real.sqrt 3 = 2 * Real.sqrt (r^2 - ((C.1^2 + C.2^2) - (2 * C.1 * 0 + 2 * C.2 * 0)))) :=
by
  sorry

end circle_equation_and_shortest_chord_l272_27202


namespace num_perfect_squares_mul_36_lt_10pow8_l272_27266

theorem num_perfect_squares_mul_36_lt_10pow8 : 
  ∃(n : ℕ), n = 1666 ∧ 
  ∀ (N : ℕ), (1 ≤ N) → (N^2 < 10^8) → (N^2 % 36 = 0) → 
  (N ≤ 9996 ∧ N % 6 = 0) :=
by
  sorry

end num_perfect_squares_mul_36_lt_10pow8_l272_27266


namespace minimum_red_chips_l272_27206

variable (w b r : ℕ)

axiom C1 : b ≥ (1 / 3 : ℚ) * w
axiom C2 : b ≤ (1 / 4 : ℚ) * r
axiom C3 : w + b ≥ 75

theorem minimum_red_chips : r = 76 := by sorry

end minimum_red_chips_l272_27206


namespace crease_length_l272_27233

theorem crease_length (AB : ℝ) (h₁ : AB = 15)
  (h₂ : ∀ (area : ℝ) (folded_area : ℝ), folded_area = 0.25 * area) :
  ∃ (DE : ℝ), DE = 0.5 * AB :=
by
  use 7.5 -- DE
  sorry

end crease_length_l272_27233


namespace cost_of_painting_murals_l272_27296

def first_mural_area : ℕ := 20 * 15
def second_mural_area : ℕ := 25 * 10
def third_mural_area : ℕ := 30 * 8

def first_mural_time : ℕ := first_mural_area * 20
def second_mural_time : ℕ := second_mural_area * 25
def third_mural_time : ℕ := third_mural_area * 30

def total_time : ℚ := (first_mural_time + second_mural_time + third_mural_time) / 60

def total_area : ℕ := first_mural_area + second_mural_area + third_mural_area

def cost (area : ℕ) : ℚ :=
  if area <= 100 then area * 150 else 
  if area <= 300 then 100 * 150 + (area - 100) * 175 
  else 100 * 150 + 200 * 175 + (area - 300) * 200

def total_cost : ℚ := cost total_area

theorem cost_of_painting_murals :
  total_cost = 148000 := by
  sorry

end cost_of_painting_murals_l272_27296


namespace max_n_satisfying_property_l272_27221

theorem max_n_satisfying_property :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, Nat.gcd m n = 1 → m^6 % n = 1) ∧ n = 504 :=
by
  sorry

end max_n_satisfying_property_l272_27221


namespace problem_l272_27247

open Function

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) + a (n - m) = 2 * a n

theorem problem (h_arith : arithmetic_sequence a) (h_eq : a 1 + 3 * a 6 + a 11 = 10) :
  a 5 + a 7 = 4 := 
sorry

end problem_l272_27247


namespace compressor_stations_valid_l272_27294

def compressor_stations : Prop :=
  ∃ (x y z a : ℝ),
    x + y = 3 * z ∧  -- condition 1
    z + y = x + a ∧  -- condition 2
    x + z = 60 ∧     -- condition 3
    0 < a ∧ a < 60 ∧ -- condition 4
    a = 42 ∧         -- specific value for a
    x = 33 ∧         -- expected value for x
    y = 48 ∧         -- expected value for y
    z = 27           -- expected value for z

theorem compressor_stations_valid : compressor_stations := 
  by sorry

end compressor_stations_valid_l272_27294


namespace parallel_vectors_result_l272_27225

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, 4)
noncomputable def m : ℝ := -1 / 2

theorem parallel_vectors_result :
  (b m).1 * a.2 = (b m).2 * a.1 →
  2 * a - b m = (4, -8) :=
by
  intro h
  -- Proof omitted
  sorry

end parallel_vectors_result_l272_27225


namespace range_of_m_l272_27291

def point_P := (1, 1)
def circle_C1 (x y m : ℝ) := x^2 + y^2 + 2*x - m = 0

theorem range_of_m (m : ℝ) :
  (1 + 1)^2 + 1^2 > m + 1 → -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l272_27291


namespace total_time_on_road_l272_27254

def driving_time_day1 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def driving_time_day2 (jade_time krista_time break_time krista_refuel lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + break_time + krista_refuel + lunch_break

def driving_time_day3 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def total_driving_time (day1 day2 day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem total_time_on_road :
  total_driving_time 
    (driving_time_day1 8 6 1 1) 
    (driving_time_day2 7 5 0.5 (1/3) 1) 
    (driving_time_day3 6 4 1 1) 
  = 42.3333 := 
  by 
    sorry

end total_time_on_road_l272_27254


namespace ratio_of_segments_l272_27235

-- Definitions and conditions as per part (a)
variables (a b c r s : ℝ)
variable (h₁ : a / b = 1 / 3)
variable (h₂ : a^2 = r * c)
variable (h₃ : b^2 = s * c)

-- The statement of the theorem directly addressing part (c)
theorem ratio_of_segments (a b c r s : ℝ) 
  (h₁ : a / b = 1 / 3)
  (h₂ : a^2 = r * c)
  (h₃ : b^2 = s * c) :
  r / s = 1 / 9 :=
  sorry

end ratio_of_segments_l272_27235


namespace mika_saucer_surface_area_l272_27229

noncomputable def surface_area_saucer (r h rim_thickness : ℝ) : ℝ :=
  let A_cap := 2 * Real.pi * r * h  -- Surface area of the spherical cap
  let R_outer := r
  let R_inner := r - rim_thickness
  let A_rim := Real.pi * (R_outer^2 - R_inner^2)  -- Area of the rim
  A_cap + A_rim

theorem mika_saucer_surface_area :
  surface_area_saucer 3 1.5 1 = 14 * Real.pi :=
sorry

end mika_saucer_surface_area_l272_27229


namespace ruffy_age_difference_l272_27290

theorem ruffy_age_difference (R O : ℕ) (hR : R = 9) (hRO : R = (3/4 : ℚ) * O) :
  (R - 4) - (1 / 2 : ℚ) * (O - 4) = 1 :=
by 
  sorry

end ruffy_age_difference_l272_27290


namespace UncleVanya_travel_time_l272_27248

-- Define the conditions
variables (x y z : ℝ)
variables (h1 : 2 * x + 3 * y + 20 * z = 66)
variables (h2 : 5 * x + 8 * y + 30 * z = 144)

-- Question: how long will it take to walk 4 km, cycle 5 km, and drive 80 km
theorem UncleVanya_travel_time : 4 * x + 5 * y + 80 * z = 174 :=
sorry

end UncleVanya_travel_time_l272_27248


namespace solve_for_xy_l272_27236

theorem solve_for_xy (x y : ℝ) (h : 2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2)) : x * y = -9 / 4 :=
by sorry

end solve_for_xy_l272_27236


namespace max_median_of_pos_integers_l272_27208

theorem max_median_of_pos_integers
  (k m p r s t u : ℕ)
  (h_avg : (k + m + p + r + s + t + u) / 7 = 24)
  (h_order : k < m ∧ m < p ∧ p < r ∧ r < s ∧ s < t ∧ t < u)
  (h_t : t = 54)
  (h_km_sum : k + m ≤ 20)
  : r ≤ 53 :=
sorry

end max_median_of_pos_integers_l272_27208


namespace problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l272_27293

noncomputable def f (x : ℝ) (k : ℝ) := (Real.log x - k - 1) * x

-- Problem 1: Interval of monotonicity and extremum.
theorem problem1_monotonic_and_extremum (k : ℝ):
  (k ≤ 0 → ∀ x, 1 < x → f x k = (Real.log x - k - 1) * x) ∧
  (k > 0 → (∀ x, 1 < x ∧ x < Real.exp k → f x k = (Real.log x - k - 1) * x) ∧
           (∀ x, Real.exp k < x → f x k = (Real.log x - k - 1) * x) ∧
           f (Real.exp k) k = -Real.exp k) := sorry

-- Problem 2: Range of k.
theorem problem2_range_of_k (k : ℝ):
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f x k < 4 * Real.log x) ↔
  k > 1 - (8 / Real.exp 2) := sorry

-- Problem 3: Inequality involving product of x1 and x2.
theorem problem3_inequality (x1 x2 : ℝ) (k : ℝ):
  x1 ≠ x2 ∧ f x1 k = f x2 k → x1 * x2 < Real.exp (2 * k) := sorry

end problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l272_27293


namespace sequence_non_existence_l272_27270

variable (α β : ℝ)
variable (r : ℝ)

theorem sequence_non_existence 
  (hαβ : α * β > 0) :  
  (∃ (x : ℕ → ℝ), x 0 = r ∧ ∀ n, x (n + 1) = (x n + α) / (β * (x n) + 1) → false) ↔ 
  r = - (1 / β) :=
sorry

end sequence_non_existence_l272_27270


namespace tan_alpha_minus_pi_over_4_eq_neg_3_l272_27285

theorem tan_alpha_minus_pi_over_4_eq_neg_3
  (α : ℝ)
  (h1 : True) -- condition to ensure we define α in ℝ, "True" is just a dummy
  (a : ℝ × ℝ := (Real.cos α, -2))
  (b : ℝ × ℝ := (Real.sin α, 1))
  (h2 : ∃ k : ℝ, a = k • b) : 
  Real.tan (α - Real.pi / 4) = -3 :=
  sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_l272_27285


namespace interest_rate_calculation_l272_27276

theorem interest_rate_calculation
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ)
  (h1 : SI = 2100) (h2 : P = 875) (h3 : T = 20) :
  (SI * 100 = P * R * T) → R = 12 :=
by
  sorry

end interest_rate_calculation_l272_27276


namespace gcd_odd_multiple_1187_l272_27288

theorem gcd_odd_multiple_1187 (b: ℤ) (h1: b % 2 = 1) (h2: ∃ k: ℤ, b = 1187 * k) :
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 1 :=
by
  sorry

end gcd_odd_multiple_1187_l272_27288


namespace ratio_of_areas_ACP_BQA_l272_27211

open EuclideanGeometry

-- Define the geometric configuration
variables (A B C D P Q : Point)
  (is_square : square A B C D)
  (is_bisector_CAD : is_angle_bisector A C D P)
  (is_bisector_ABD : is_angle_bisector B A D Q)

-- Define the areas of triangles
def area_triangle (X Y Z : Point) : Real := sorry -- Placeholder for the area function

-- Lean statement for the proof problem
theorem ratio_of_areas_ACP_BQA 
  (h_square : is_square) 
  (h_bisector_CAD : is_bisector_CAD) 
  (h_bisector_ABD : is_bisector_ABD) :
  (area_triangle A C P) / (area_triangle B Q A) = 2 :=
sorry

end ratio_of_areas_ACP_BQA_l272_27211


namespace profit_difference_is_50_l272_27287

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l272_27287


namespace linear_relationship_selling_price_maximize_profit_l272_27265

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end linear_relationship_selling_price_maximize_profit_l272_27265


namespace water_percentage_in_dried_grapes_l272_27219

noncomputable def fresh_grape_weight : ℝ := 40  -- weight of fresh grapes in kg
noncomputable def dried_grape_weight : ℝ := 5  -- weight of dried grapes in kg
noncomputable def water_percentage_fresh : ℝ := 0.90  -- percentage of water in fresh grapes

noncomputable def water_weight_fresh : ℝ := fresh_grape_weight * water_percentage_fresh
noncomputable def solid_weight_fresh : ℝ := fresh_grape_weight * (1 - water_percentage_fresh)
noncomputable def water_weight_dried : ℝ := dried_grape_weight - solid_weight_fresh
noncomputable def water_percentage_dried : ℝ := (water_weight_dried / dried_grape_weight) * 100

theorem water_percentage_in_dried_grapes : water_percentage_dried = 20 := by
  sorry

end water_percentage_in_dried_grapes_l272_27219


namespace evaluate_expression_l272_27231

-- Define the conditions
def two_pow_nine : ℕ := 2 ^ 9
def neg_one_pow_eight : ℤ := (-1) ^ 8

-- Define the proof statement
theorem evaluate_expression : two_pow_nine + neg_one_pow_eight = 513 := 
by
  sorry

end evaluate_expression_l272_27231


namespace four_sq_geq_prod_sum_l272_27259

variable {α : Type*} [LinearOrderedField α]

theorem four_sq_geq_prod_sum (a b c d : α) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

end four_sq_geq_prod_sum_l272_27259


namespace find_g3_l272_27250

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 1

theorem find_g3 : g 3 = 0 := by
  sorry

end find_g3_l272_27250


namespace half_of_number_l272_27268

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number_l272_27268


namespace maximal_partition_sets_l272_27251

theorem maximal_partition_sets : 
  ∃(n : ℕ), (∀(a : ℕ), a * n = 16657706 → (a = 5771 ∧ n = 2886)) := 
by
  sorry

end maximal_partition_sets_l272_27251


namespace A_plus_B_eq_one_fourth_l272_27282

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth_l272_27282


namespace solve_x_plus_Sx_eq_2001_l272_27297

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem solve_x_plus_Sx_eq_2001 (x : ℕ) (h : x + sum_of_digits x = 2001) : x = 1977 :=
  sorry

end solve_x_plus_Sx_eq_2001_l272_27297


namespace midpoint_trajectory_l272_27224

-- Define the parabola and line intersection conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_focus (A B : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, (∀ P ∈ [A, B, focus], P.2 = m * P.1 + b)

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_trajectory (A B M : ℝ × ℝ) (focus : ℝ × ℝ):
  (parabola A.1 A.2) ∧ (parabola B.1 B.2) ∧ (line_through_focus A B focus) ∧ (midpoint A B M)
  → (M.1 ^ 2 = 2 * M.2 - 2) :=
by
  sorry

end midpoint_trajectory_l272_27224


namespace greatest_k_for_inquality_l272_27253

theorem greatest_k_for_inquality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 > b*c) :
    (a^2 - b*c)^2 > 4 * ((b^2 - c*a) * (c^2 - a*b)) :=
  sorry

end greatest_k_for_inquality_l272_27253


namespace determine_m_l272_27213

theorem determine_m (m : ℝ) : (∀ x : ℝ, (m * x = 1 → x = 1 ∨ x = -1)) ↔ (m = 0 ∨ m = 1 ∨ m = -1) :=
by sorry

end determine_m_l272_27213


namespace student_solved_18_correctly_l272_27258

theorem student_solved_18_correctly (total_problems : ℕ) (correct : ℕ) (wrong : ℕ) 
  (h1 : total_problems = 54) (h2 : wrong = 2 * correct) (h3 : total_problems = correct + wrong) :
  correct = 18 :=
by
  sorry

end student_solved_18_correctly_l272_27258


namespace Isabella_total_items_l272_27239

theorem Isabella_total_items (A_pants A_dresses I_pants I_dresses : ℕ) 
  (h1 : A_pants = 3 * I_pants) 
  (h2 : A_dresses = 3 * I_dresses)
  (h3 : A_pants = 21) 
  (h4 : A_dresses = 18) : 
  I_pants + I_dresses = 13 :=
by
  -- Proof goes here
  sorry

end Isabella_total_items_l272_27239


namespace simplify_expression_l272_27240

theorem simplify_expression :
  1 + (1 / (1 + Real.sqrt 2)) - (1 / (1 - Real.sqrt 5)) =
  1 + ((-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10)) :=
by
  sorry

end simplify_expression_l272_27240


namespace acute_angle_sum_l272_27215

open Real

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
                        (hβ : 0 < β ∧ β < π / 2)
                        (h1 : 3 * (sin α) ^ 2 + 2 * (sin β) ^ 2 = 1)
                        (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end acute_angle_sum_l272_27215


namespace max_area_inscribed_triangle_l272_27212

/-- Let ΔABC be an inscribed triangle in the ellipse given by the equation
    (x^2 / 9) + (y^2 / 4) = 1, where the line segment AB passes through the 
    point (1, 0). Prove that the maximum area of ΔABC is (16 * sqrt 2) / 3. --/
theorem max_area_inscribed_triangle
  (A B C : ℝ × ℝ) 
  (hA : (A.1 ^ 2) / 9 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 9 + (B.2 ^ 2) / 4 = 1)
  (hC : (C.1 ^ 2) / 9 + (C.2 ^ 2) / 4 = 1)
  (hAB : ∃ n : ℝ, ∀ x y : ℝ, (x, y) ∈ [A, B] → x = n * y + 1)
  : ∃ S : ℝ, S = ((16 : ℝ) * Real.sqrt 2) / 3 :=
sorry

end max_area_inscribed_triangle_l272_27212


namespace coin_value_difference_l272_27227

theorem coin_value_difference (p n d : ℕ) (h : p + n + d = 3000) (hp : p ≥ 1) (hn : n ≥ 1) (hd : d ≥ 1) : 
  (p + 5 * n + 10 * d).max - (p + 5 * n + 10 * d).min = 26973 := 
sorry

end coin_value_difference_l272_27227


namespace sequence_arithmetic_l272_27271

variable (a b : ℕ → ℤ)

theorem sequence_arithmetic :
  a 0 = 3 →
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →
  b 3 = -2 →
  b 10 = 12 →
  a 8 = 3 :=
by
  intros h1 ha hb3 hb10
  sorry

end sequence_arithmetic_l272_27271


namespace range_of_x_l272_27256

variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, has_deriv_at f (derivative f x) x)
variable (hf2 : ∀ x : ℝ, derivative f x > - f x)

theorem range_of_x (h : f (Real.log 3) = 1/3) : 
  {x : ℝ | f x > 1 / Real.exp x} = Set.Ioi (Real.log 3) := 
by 
  sorry

end range_of_x_l272_27256


namespace average_discount_rate_l272_27223

theorem average_discount_rate :
  ∃ x : ℝ, (7200 * (1 - x)^2 = 3528) ∧ x = 0.3 :=
by
  sorry

end average_discount_rate_l272_27223
