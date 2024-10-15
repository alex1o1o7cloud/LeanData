import Mathlib

namespace NUMINAMATH_GPT_Bruce_bought_8_kg_of_grapes_l2191_219100

-- Defining the conditions
def rate_grapes := 70
def rate_mangoes := 55
def weight_mangoes := 11
def total_paid := 1165

-- Result to be proven
def cost_mangoes := rate_mangoes * weight_mangoes
def total_cost_grapes (G : ℕ) := rate_grapes * G
def total_cost (G : ℕ) := (total_cost_grapes G) + cost_mangoes

theorem Bruce_bought_8_kg_of_grapes (G : ℕ) (h : total_cost G = total_paid) : G = 8 :=
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_Bruce_bought_8_kg_of_grapes_l2191_219100


namespace NUMINAMATH_GPT_area_percentage_increase_l2191_219188

theorem area_percentage_increase (r1 r2 : ℝ) (π : ℝ) (area1 area2 : ℝ) (N : ℝ) :
  r1 = 6 → r2 = 4 → area1 = π * r1 ^ 2 → area2 = π * r2 ^ 2 →
  N = 125 →
  ((area1 - area2) / area2) * 100 = N :=
by {
  sorry
}

end NUMINAMATH_GPT_area_percentage_increase_l2191_219188


namespace NUMINAMATH_GPT_simplify_expression_l2191_219116

theorem simplify_expression (y : ℝ) : 
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) = 8 * y^3 - 12 * y^2 + 20 * y - 24 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2191_219116


namespace NUMINAMATH_GPT_candies_indeterminable_l2191_219178

theorem candies_indeterminable
  (num_bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) (known_candies : ℕ) :
  num_bags = 26 →
  cookies_per_bag = 2 →
  total_cookies = 52 →
  num_bags * cookies_per_bag = total_cookies →
  ∀ (candies : ℕ), candies = known_candies → false :=
by
  intros
  sorry

end NUMINAMATH_GPT_candies_indeterminable_l2191_219178


namespace NUMINAMATH_GPT_solve_arithmetic_seq_l2191_219137

theorem solve_arithmetic_seq (x : ℝ) (h : x > 0) (hx : x^2 = (4 + 16) / 2) : x = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_solve_arithmetic_seq_l2191_219137


namespace NUMINAMATH_GPT_compute_fraction_at_six_l2191_219125

theorem compute_fraction_at_six (x : ℕ) (h : x = 6) : (x^6 - 16 * x^3 + 64) / (x^3 - 8) = 208 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_at_six_l2191_219125


namespace NUMINAMATH_GPT_find_abc_digits_l2191_219105

theorem find_abc_digits (N : ℕ) (abcd : ℕ) (a b c d : ℕ) (hN : N % 10000 = abcd) (hNsq : N^2 % 10000 = abcd)
  (ha_ne_zero : a ≠ 0) (hb_ne_six : b ≠ 6) (hc_ne_six : c ≠ 6) : (a * 100 + b * 10 + c) = 106 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_find_abc_digits_l2191_219105


namespace NUMINAMATH_GPT_four_digit_number_is_2561_l2191_219152

-- Define the problem domain based on given conditions
def unique_in_snowflake_and_directions (grid : Matrix (Fin 3) (Fin 6) ℕ) : Prop :=
  ∀ (i j : Fin 3), -- across all directions
    ∀ (x y : Fin 6), 
      (x ≠ y) → 
      (grid i x ≠ grid i y) -- uniqueness in i-direction
      ∧ (grid y x ≠ grid y y) -- uniqueness in j-direction

-- Assignment of numbers in the grid fulfilling the conditions
def grid : Matrix (Fin 3) (Fin 6) ℕ :=
![ ![2, 5, 2, 5, 1, 6], ![4, 3, 2, 6, 1, 1], ![6, 1, 4, 5, 3, 2] ]

-- Definition of the four-digit number
def ABCD : ℕ := grid 0 1 * 1000 + grid 0 2 * 100 + grid 0 3 * 10 + grid 0 4

-- The theorem to be proved
theorem four_digit_number_is_2561 :
  unique_in_snowflake_and_directions grid →
  ABCD = 2561 :=
sorry

end NUMINAMATH_GPT_four_digit_number_is_2561_l2191_219152


namespace NUMINAMATH_GPT_sum_of_remainders_l2191_219170

theorem sum_of_remainders (a b c d e : ℕ)
  (h1 : a % 13 = 3)
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9)
  (h5 : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_remainders_l2191_219170


namespace NUMINAMATH_GPT_find_smallest_n_modulo_l2191_219122

theorem find_smallest_n_modulo :
  ∃ n : ℕ, n > 0 ∧ (2007 * n) % 1000 = 837 ∧ n = 691 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_n_modulo_l2191_219122


namespace NUMINAMATH_GPT_ellipse_closer_to_circle_l2191_219169

variables (a : ℝ)

-- Conditions: 1 < a < 2 + sqrt 5
def in_range_a (a : ℝ) : Prop := 1 < a ∧ a < 2 + Real.sqrt 5

-- Ellipse eccentricity should decrease as 'a' increases for the given range 1 < a < 2 + sqrt 5
theorem ellipse_closer_to_circle (h_range : in_range_a a) :
    ∃ b : ℝ, b = Real.sqrt (1 - (a^2 - 1) / (4 * a)) ∧ ∀ a', (1 < a' ∧ a' < 2 + Real.sqrt 5 ∧ a < a') → b > Real.sqrt (1 - (a'^2 - 1) / (4 * a')) := 
sorry

end NUMINAMATH_GPT_ellipse_closer_to_circle_l2191_219169


namespace NUMINAMATH_GPT_maria_zoo_ticket_discount_percentage_l2191_219144

theorem maria_zoo_ticket_discount_percentage 
  (regular_price : ℝ) (paid_price : ℝ) (discount_percentage : ℝ)
  (h1 : regular_price = 15) (h2 : paid_price = 9) :
  discount_percentage = 40 :=
by
  sorry

end NUMINAMATH_GPT_maria_zoo_ticket_discount_percentage_l2191_219144


namespace NUMINAMATH_GPT_dividend_percentage_shares_l2191_219151

theorem dividend_percentage_shares :
  ∀ (purchase_price market_value : ℝ) (interest_rate : ℝ),
  purchase_price = 56 →
  market_value = 42 →
  interest_rate = 0.12 →
  ( (interest_rate * purchase_price) / market_value * 100 = 16) :=
by
  intros purchase_price market_value interest_rate h1 h2 h3
  rw [h1, h2, h3]
  -- Calculations were done in solution
  sorry

end NUMINAMATH_GPT_dividend_percentage_shares_l2191_219151


namespace NUMINAMATH_GPT_dugu_team_prob_l2191_219127

def game_prob (prob_win_first : ℝ) (prob_increase : ℝ) (prob_decrease : ℝ) : ℝ :=
  let p1 := prob_win_first
  let p2 := prob_win_first + prob_increase
  let p3 := prob_win_first + 2 * prob_increase
  let p4 := prob_win_first + 3 * prob_increase
  let p5 := prob_win_first + 4 * prob_increase
  let win_in_3 := p1 * p2 * p3
  let lose_first := (1 - prob_win_first)
  let win_then := prob_win_first
  let win_in_4a := lose_first * (prob_win_first - prob_decrease) * 
    prob_win_first * p2 * p3
  let win_in_4b := win_then * (1 - (prob_win_first + prob_increase)) *
    p2 * p3
  let win_in_4c := win_then * p2 * (1 - prob_win_first + prob_increase - 
    prob_decrease) * p4

  win_in_3 + win_in_4a + win_in_4b + win_in_4c

theorem dugu_team_prob : 
  game_prob 0.4 0.1 0.1 = 0.236 :=
by
  sorry

end NUMINAMATH_GPT_dugu_team_prob_l2191_219127


namespace NUMINAMATH_GPT_solve_for_x_l2191_219129

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (x y : ℝ) (h : 16 * (3:ℝ) ^ x = (7:ℝ) ^ (y + 4)) (hy : y = -4) :
  x = -4 * log 3 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2191_219129


namespace NUMINAMATH_GPT_regular_polygon_sides_l2191_219130

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2191_219130


namespace NUMINAMATH_GPT_rationalize_denominator_l2191_219184

theorem rationalize_denominator (A B C : ℤ) (h : A + B * Real.sqrt C = -(9) - 4 * Real.sqrt 5) : A * B * C = 180 :=
by
  have hA : A = -9 := by sorry
  have hB : B = -4 := by sorry
  have hC : C = 5 := by sorry
  rw [hA, hB, hC]
  norm_num

end NUMINAMATH_GPT_rationalize_denominator_l2191_219184


namespace NUMINAMATH_GPT_range_of_function_l2191_219181

noncomputable def function_y (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem range_of_function : 
  ∃ (a b : ℝ), a = -12 ∧ b = 4 ∧ 
  (∀ y, (∃ x, -5 ≤ x ∧ x ≤ 0 ∧ y = function_y x) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end NUMINAMATH_GPT_range_of_function_l2191_219181


namespace NUMINAMATH_GPT_tan_pi_add_theta_l2191_219102

theorem tan_pi_add_theta (θ : ℝ) (h : Real.tan (Real.pi + θ) = 2) : 
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_add_theta_l2191_219102


namespace NUMINAMATH_GPT_scientific_notation_35_million_l2191_219168

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_35_million_l2191_219168


namespace NUMINAMATH_GPT_union_of_sets_l2191_219107

open Set

theorem union_of_sets :
  ∀ (P Q : Set ℕ), P = {1, 2} → Q = {2, 3} → P ∪ Q = {1, 2, 3} :=
by
  intros P Q hP hQ
  rw [hP, hQ]
  exact sorry

end NUMINAMATH_GPT_union_of_sets_l2191_219107


namespace NUMINAMATH_GPT_lake_with_more_frogs_has_45_frogs_l2191_219171

-- Definitions for the problem.
variable (F : ℝ) -- Number of frogs in the lake with more frogs.
variable (F_less : ℝ) -- Number of frogs in Lake Crystal (the lake with fewer frogs).

-- Conditions
axiom fewer_frogs_condition : F_less = 0.8 * F
axiom total_frogs_condition : F + F_less = 81

-- Theorem statement: Proving that the number of frogs in the lake with more frogs is 45.
theorem lake_with_more_frogs_has_45_frogs :
  F = 45 :=
by
  sorry

end NUMINAMATH_GPT_lake_with_more_frogs_has_45_frogs_l2191_219171


namespace NUMINAMATH_GPT_dot_product_zero_l2191_219189

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the dot product operation for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the scalar multiplication and vector subtraction for 2D vectors
def scalar_mul_vec (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Now we state the theorem we want to prove
theorem dot_product_zero : dot_product a (vec_sub (scalar_mul_vec 2 a) b) = 0 := 
by
  sorry

end NUMINAMATH_GPT_dot_product_zero_l2191_219189


namespace NUMINAMATH_GPT_domain_of_function_l2191_219179

def domain_of_f (x : ℝ) : Prop :=
  (x ≤ 2) ∧ (x ≠ 1)

theorem domain_of_function :
  ∀ x : ℝ, x ∈ { x | (x ≤ 2) ∧ (x ≠ 1) } ↔ domain_of_f x :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2191_219179


namespace NUMINAMATH_GPT_seventh_term_correct_l2191_219180

noncomputable def seventh_term_geometric_sequence (a r : ℝ) (h1 : a = 5) (h2 : a * r = 1/5) : ℝ :=
  a * r ^ 6

theorem seventh_term_correct :
  seventh_term_geometric_sequence 5 (1/25) (by rfl) (by norm_num) = 1 / 48828125 :=
  by
    unfold seventh_term_geometric_sequence
    sorry

end NUMINAMATH_GPT_seventh_term_correct_l2191_219180


namespace NUMINAMATH_GPT_octahedron_tetrahedron_surface_area_ratio_l2191_219134

theorem octahedron_tetrahedron_surface_area_ratio 
  (s : ℝ) 
  (h₁ : s = 1)
  (A_octahedron : ℝ := 2 * Real.sqrt 3)
  (A_tetrahedron : ℝ := Real.sqrt 3)
  (h₂ : A_octahedron = 2 * Real.sqrt 3 * s^2 / 2 * Real.sqrt 3 * (1/4) * s^2) 
  (h₃ : A_tetrahedron = Real.sqrt 3 * s^2 / 4)
  :
  A_octahedron / A_tetrahedron = 2 := 
by
  sorry

end NUMINAMATH_GPT_octahedron_tetrahedron_surface_area_ratio_l2191_219134


namespace NUMINAMATH_GPT_sine_gamma_half_leq_c_over_a_plus_b_l2191_219136

variable (a b c : ℝ) (γ : ℝ)

-- Consider a triangle with sides a, b, c, and angle γ opposite to side c.
-- We need to prove that sin(γ / 2) ≤ c / (a + b).
theorem sine_gamma_half_leq_c_over_a_plus_b (h_c_pos : 0 < c) 
  (h_g_angle : 0 < γ ∧ γ < 2 * π) : 
  Real.sin (γ / 2) ≤ c / (a + b) := 
  sorry

end NUMINAMATH_GPT_sine_gamma_half_leq_c_over_a_plus_b_l2191_219136


namespace NUMINAMATH_GPT_male_teacher_classes_per_month_l2191_219147

theorem male_teacher_classes_per_month (x y a : ℕ) :
  (15 * x = 6 * (x + y)) ∧ (a * y = 6 * (x + y)) → a = 10 :=
by
  sorry

end NUMINAMATH_GPT_male_teacher_classes_per_month_l2191_219147


namespace NUMINAMATH_GPT_total_area_correct_l2191_219115

noncomputable def total_area (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : ℝ :=
  let rect_area := 588 -- Area of the rectangle
  let semi_circle_area := 24.5 * Real.pi -- Area of the semi-circle based on given diameter
  rect_area + semi_circle_area

theorem total_area_correct (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : 
  total_area b l h1 h2 = 588 + 24.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_total_area_correct_l2191_219115


namespace NUMINAMATH_GPT_find_last_score_l2191_219126

/-- The list of scores in ascending order -/
def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

/--
  The problem states that the average score after each entry is an integer.
  Given the scores in ascending order, determine the last score entered.
-/
theorem find_last_score (h : ∀ (n : ℕ) (hn : n < scores.length),
    (scores.take (n + 1) |>.sum : ℤ) % (n + 1) = 0) :
  scores.last' = some 80 :=
sorry

end NUMINAMATH_GPT_find_last_score_l2191_219126


namespace NUMINAMATH_GPT_total_cost_is_225_l2191_219161

def total_tickets : ℕ := 29
def cost_7_dollar_ticket : ℕ := 7
def cost_9_dollar_ticket : ℕ := 9
def number_of_9_dollar_tickets : ℕ := 11
def number_of_7_dollar_tickets : ℕ := total_tickets - number_of_9_dollar_tickets
def total_cost : ℕ := (number_of_9_dollar_tickets * cost_9_dollar_ticket) + (number_of_7_dollar_tickets * cost_7_dollar_ticket)

theorem total_cost_is_225 : total_cost = 225 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_225_l2191_219161


namespace NUMINAMATH_GPT_product_eq_5832_l2191_219145

theorem product_eq_5832 (P Q R S : ℕ) 
(h1 : P + Q + R + S = 48)
(h2 : P + 3 = Q - 3)
(h3 : Q - 3 = R * 3)
(h4 : R * 3 = S / 3) :
P * Q * R * S = 5832 := sorry

end NUMINAMATH_GPT_product_eq_5832_l2191_219145


namespace NUMINAMATH_GPT_exists_sequences_l2191_219163

theorem exists_sequences (m n : Nat → Nat) (h₁ : ∀ k, m k = 2 * k) (h₂ : ∀ k, n k = 5 * k * k)
  (h₃ : ∀ (i j : Nat), (i ≠ j) → (m i ≠ m j) ∧ (n i ≠ n j)) :
  (∀ k, Nat.sqrt (n k + (m k) * (m k)) = 3 * k) ∧
  (∀ k, Nat.sqrt (n k - (m k) * (m k)) = k) :=
by 
  sorry

end NUMINAMATH_GPT_exists_sequences_l2191_219163


namespace NUMINAMATH_GPT_sandy_savings_l2191_219153

theorem sandy_savings (S : ℝ) :
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  P * 100 = 15 :=
by
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  have hP : P = 0.165 / 1.10 := by sorry
  have hP_percent : P * 100 = 15 := by sorry
  exact hP_percent

end NUMINAMATH_GPT_sandy_savings_l2191_219153


namespace NUMINAMATH_GPT_shortest_distance_phenomena_explained_l2191_219135

def condition1 : Prop :=
  ∀ (a b : ℕ), (exists nail1 : ℕ, exists nail2 : ℕ, nail1 ≠ nail2) → (exists wall : ℕ, wall = a + b)

def condition2 : Prop :=
  ∀ (tree1 tree2 tree3 : ℕ), tree1 ≠ tree2 → tree2 ≠ tree3 → (tree1 + tree2 + tree3) / 3 = tree2

def condition3 : Prop :=
  ∀ (A B : ℕ), ∃ (C : ℕ), C = (B - A) → (A = B - (B - A))

def condition4 : Prop :=
  ∀ (dist : ℕ), dist = 0 → exists shortest : ℕ, shortest < dist

-- The following theorem needs to be proven to match our mathematical problem
theorem shortest_distance_phenomena_explained :
  condition3 ∧ condition4 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_phenomena_explained_l2191_219135


namespace NUMINAMATH_GPT_calculate_constants_l2191_219190

noncomputable def parabola_tangent_to_line (a b : ℝ) : Prop :=
  let discriminant := (b - 2) ^ 2 + 28 * a
  discriminant = 0

theorem calculate_constants
  (a b : ℝ)
  (h_tangent : parabola_tangent_to_line a b) :
  a = -((b - 2) ^ 2) / 28 ∧ b ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_constants_l2191_219190


namespace NUMINAMATH_GPT_cos_double_angle_l2191_219155

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l2191_219155


namespace NUMINAMATH_GPT_victor_initial_books_l2191_219165

theorem victor_initial_books (x : ℕ) : (x + 3 = 12) → (x = 9) :=
by
  sorry

end NUMINAMATH_GPT_victor_initial_books_l2191_219165


namespace NUMINAMATH_GPT_cubic_eq_one_real_root_l2191_219198

-- Given a, b, c forming a geometric sequence
variables {a b c : ℝ}

-- Definition of a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Equation ax^3 + bx^2 + cx = 0
def cubic_eq (a b c x : ℝ) : Prop :=
  a * x^3 + b * x^2 + c * x = 0

-- Prove the number of real roots
theorem cubic_eq_one_real_root (h : geometric_sequence a b c) :
  ∃ x : ℝ, cubic_eq a b c x ∧ ¬∃ y ≠ x, cubic_eq a b c y :=
sorry

end NUMINAMATH_GPT_cubic_eq_one_real_root_l2191_219198


namespace NUMINAMATH_GPT_find_first_5digits_of_M_l2191_219156

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end NUMINAMATH_GPT_find_first_5digits_of_M_l2191_219156


namespace NUMINAMATH_GPT_sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l2191_219103

theorem sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three : 
  ((Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3) := by
  sorry

end NUMINAMATH_GPT_sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l2191_219103


namespace NUMINAMATH_GPT_shirt_final_price_is_correct_l2191_219183

noncomputable def final_price_percentage (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * 0.80
  let second_discount := first_discount * 0.90
  let anniversary_addition := second_discount * 1.05
  let final_price := anniversary_addition * 1.15
  final_price / initial_price * 100

theorem shirt_final_price_is_correct (initial_price : ℝ) : final_price_percentage initial_price = 86.94 := by
  sorry

end NUMINAMATH_GPT_shirt_final_price_is_correct_l2191_219183


namespace NUMINAMATH_GPT_distance_foci_of_hyperbola_l2191_219119

noncomputable def distance_between_foci : ℝ :=
  8 * Real.sqrt 5

theorem distance_foci_of_hyperbola :
  ∃ A B : ℝ, (9 * A^2 - 36 * A - B^2 + 4 * B = 40) → distance_between_foci = 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_distance_foci_of_hyperbola_l2191_219119


namespace NUMINAMATH_GPT_ratio_of_length_to_breadth_l2191_219123

theorem ratio_of_length_to_breadth (b l k : ℕ) (h1 : b = 15) (h2 : l = k * b) (h3 : l * b = 675) : l / b = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_length_to_breadth_l2191_219123


namespace NUMINAMATH_GPT_find_constant_k_l2191_219128

theorem find_constant_k (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (h₁ : ∀ n, S n = 3 * 2^n + k)
  (h₂ : ∀ n, 1 ≤ n → a n = S n - S (n - 1))
  (h₃ : ∃ q, ∀ n, 1 ≤ n → a (n + 1) = a n * q ) :
  k = -3 := 
sorry

end NUMINAMATH_GPT_find_constant_k_l2191_219128


namespace NUMINAMATH_GPT_bisect_area_of_trapezoid_l2191_219140

-- Define the vertices of the quadrilateral
structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 16, y := 0 }
def C : Point := { x := 8, y := 8 }
def D : Point := { x := 0, y := 8 }

-- Define the equation of a line
structure Line :=
  (slope : ℚ)
  (intercept : ℚ)

-- Define the condition for parallel lines
def parallel (L1 L2 : Line) : Prop :=
  L1.slope = L2.slope

-- Define the diagonal AC and the required line
def AC : Line := { slope := 1, intercept := 0 }
def bisecting_line : Line := { slope := 1, intercept := -4 }

-- The area of trapezoid
def trapezoid_area : ℚ := (8 * (16 + 8)) / 2

-- Proof that the required line is parallel to AC and bisects the area of the trapezoid
theorem bisect_area_of_trapezoid :
  parallel bisecting_line AC ∧ 
  (1 / 2) * (8 * (16 + bisecting_line.intercept)) = trapezoid_area / 2 :=
by
  sorry

end NUMINAMATH_GPT_bisect_area_of_trapezoid_l2191_219140


namespace NUMINAMATH_GPT_price_per_rose_is_2_l2191_219197

-- Definitions from conditions
def has_amount (total_dollars : ℕ) : Prop := total_dollars = 300
def total_roses (R : ℕ) : Prop := ∃ (j : ℕ) (i : ℕ), R / 3 = j ∧ R / 2 = i ∧ j + i = 125

-- Theorem stating the price per rose
theorem price_per_rose_is_2 (R : ℕ) : 
  has_amount 300 → total_roses R → 300 / R = 2 :=
sorry

end NUMINAMATH_GPT_price_per_rose_is_2_l2191_219197


namespace NUMINAMATH_GPT_intersection_complement_l2191_219192

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {2, 3, 4}) (hB : B = {1, 2})

theorem intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_intersection_complement_l2191_219192


namespace NUMINAMATH_GPT_end_digit_of_number_l2191_219194

theorem end_digit_of_number (n : ℕ) (h_n : n = 2022) (h_start : ∃ (f : ℕ → ℕ), f 0 = 4 ∧ 
    (∀ i < n - 1, (19 ∣ (10 * f i + f (i + 1))) ∨ (23 ∣ (10 * f i + f (i + 1))))) :
  ∃ (f : ℕ → ℕ), f (n - 1) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_end_digit_of_number_l2191_219194


namespace NUMINAMATH_GPT_ball_hits_ground_at_2_72_l2191_219176

-- Define the initial conditions
def initial_velocity (v₀ : ℝ) := v₀ = 30
def initial_height (h₀ : ℝ) := h₀ = 200
def ball_height (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200

-- Prove that the ball hits the ground at t = 2.72 seconds
theorem ball_hits_ground_at_2_72 (t : ℝ) (h : ℝ) 
  (v₀ : ℝ) (h₀ : ℝ) 
  (hv₀ : initial_velocity v₀) 
  (hh₀ : initial_height h₀)
  (h_eq: ball_height t = h) 
  (h₀_eq: ball_height 0 = h₀) : 
  h = 0 -> t = 2.72 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_2_72_l2191_219176


namespace NUMINAMATH_GPT_female_students_transfer_l2191_219101

theorem female_students_transfer (x y z : ℕ) 
  (h1 : ∀ B : ℕ, B = x - 4) 
  (h2 : ∀ C : ℕ, C = x - 5)
  (h3 : ∀ B' : ℕ, B' = x - 4 + y - z)
  (h4 : ∀ C' : ℕ, C' = x + z - 7) 
  (h5 : x - y + 2 = x - 4 + y - z)
  (h6 : x - 4 + y - z = x + z - 7) 
  (h7 : 2 = 2) :
  y = 3 ∧ z = 4 := 
by 
  sorry

end NUMINAMATH_GPT_female_students_transfer_l2191_219101


namespace NUMINAMATH_GPT_guests_equal_cost_l2191_219157

-- Rental costs and meal costs
def rental_caesars_palace : ℕ := 800
def deluxe_meal_cost : ℕ := 30
def premium_meal_cost : ℕ := 40
def rental_venus_hall : ℕ := 500
def venus_special_cost : ℕ := 35
def venus_platter_cost : ℕ := 45

-- Meal distribution percentages
def deluxe_meal_percentage : ℚ := 0.60
def premium_meal_percentage : ℚ := 0.40
def venus_special_percentage : ℚ := 0.60
def venus_platter_percentage : ℚ := 0.40

-- Total costs calculation
noncomputable def total_cost_caesars (G : ℕ) : ℚ :=
  rental_caesars_palace + deluxe_meal_cost * deluxe_meal_percentage * G + premium_meal_cost * premium_meal_percentage * G

noncomputable def total_cost_venus (G : ℕ) : ℚ :=
  rental_venus_hall + venus_special_cost * venus_special_percentage * G + venus_platter_cost * venus_platter_percentage * G

-- Statement to show the equivalence of guest count
theorem guests_equal_cost (G : ℕ) : total_cost_caesars G = total_cost_venus G → G = 60 :=
by
  sorry

end NUMINAMATH_GPT_guests_equal_cost_l2191_219157


namespace NUMINAMATH_GPT_max_value_2ab_3bc_lemma_l2191_219173

noncomputable def max_value_2ab_3bc (a b c : ℝ) : ℝ :=
  2 * a * b + 3 * b * c

theorem max_value_2ab_3bc_lemma
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 = 2) :
  max_value_2ab_3bc a b c ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_2ab_3bc_lemma_l2191_219173


namespace NUMINAMATH_GPT_decimal_to_fraction_l2191_219186

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end NUMINAMATH_GPT_decimal_to_fraction_l2191_219186


namespace NUMINAMATH_GPT_penelope_min_games_l2191_219172

theorem penelope_min_games (m w l: ℕ) (h1: 25 * w - 13 * l = 2007) (h2: m = w + l) : m = 87 := by
  sorry

end NUMINAMATH_GPT_penelope_min_games_l2191_219172


namespace NUMINAMATH_GPT_nate_walks_past_per_minute_l2191_219167

-- Define the conditions as constants
def rows_G := 15
def cars_per_row_G := 10
def rows_H := 20
def cars_per_row_H := 9
def total_minutes := 30

-- Define the problem statement
theorem nate_walks_past_per_minute :
  ((rows_G * cars_per_row_G) + (rows_H * cars_per_row_H)) / total_minutes = 11 := 
sorry

end NUMINAMATH_GPT_nate_walks_past_per_minute_l2191_219167


namespace NUMINAMATH_GPT_inequality_of_factorials_and_polynomials_l2191_219138

open Nat

theorem inequality_of_factorials_and_polynomials (m n : ℕ) (hm : m ≥ n) :
  2^n * n! ≤ (m+n)! / (m-n)! ∧ (m+n)! / (m-n)! ≤ (m^2 + m)^n :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_factorials_and_polynomials_l2191_219138


namespace NUMINAMATH_GPT_smaller_cube_volume_is_correct_l2191_219139

noncomputable def inscribed_smaller_cube_volume 
  (edge_length_outer_cube : ℝ)
  (h : edge_length_outer_cube = 12) : ℝ := 
  let diameter_sphere := edge_length_outer_cube
  let radius_sphere := diameter_sphere / 2
  let space_diagonal_smaller_cube := diameter_sphere
  let side_length_smaller_cube := space_diagonal_smaller_cube / (Real.sqrt 3)
  let volume_smaller_cube := side_length_smaller_cube ^ 3
  volume_smaller_cube

theorem smaller_cube_volume_is_correct 
  (h : 12 = 12) : inscribed_smaller_cube_volume 12 h = 192 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_smaller_cube_volume_is_correct_l2191_219139


namespace NUMINAMATH_GPT_add_pure_chocolate_to_achieve_percentage_l2191_219106

/--
Given:
    Initial amount of chocolate topping: 620 ounces.
    Initial chocolate percentage: 10%.
    Desired total weight of the final mixture: 1000 ounces.
    Desired chocolate percentage in the final mixture: 70%.
Prove:
    The amount of pure chocolate to be added to achieve the desired mixture is 638 ounces.
-/
theorem add_pure_chocolate_to_achieve_percentage :
  ∃ x : ℝ,
    0.10 * 620 + x = 0.70 * 1000 ∧
    x = 638 :=
by
  sorry

end NUMINAMATH_GPT_add_pure_chocolate_to_achieve_percentage_l2191_219106


namespace NUMINAMATH_GPT_probability_all_boys_probability_one_girl_probability_at_least_one_girl_l2191_219148

-- Assumptions and Definitions
def total_outcomes := Nat.choose 5 3
def all_boys_outcomes := Nat.choose 3 3
def one_girl_outcomes := Nat.choose 3 2 * Nat.choose 2 1
def at_least_one_girl_outcomes := one_girl_outcomes + Nat.choose 3 1 * Nat.choose 2 2

-- The probability calculation proofs
theorem probability_all_boys : all_boys_outcomes / total_outcomes = 1 / 10 := by 
  sorry

theorem probability_one_girl : one_girl_outcomes / total_outcomes = 6 / 10 := by 
  sorry

theorem probability_at_least_one_girl : at_least_one_girl_outcomes / total_outcomes = 9 / 10 := by 
  sorry

end NUMINAMATH_GPT_probability_all_boys_probability_one_girl_probability_at_least_one_girl_l2191_219148


namespace NUMINAMATH_GPT_beavers_swimming_correct_l2191_219114

variable (initial_beavers remaining_beavers beavers_swimming : ℕ)

def beavers_problem : Prop :=
  initial_beavers = 2 ∧
  remaining_beavers = 1 ∧
  beavers_swimming = initial_beavers - remaining_beavers

theorem beavers_swimming_correct :
  beavers_problem initial_beavers remaining_beavers beavers_swimming → beavers_swimming = 1 :=
by
  sorry

end NUMINAMATH_GPT_beavers_swimming_correct_l2191_219114


namespace NUMINAMATH_GPT_tan_half_angle_sum_identity_l2191_219146

theorem tan_half_angle_sum_identity
  (α β γ : ℝ)
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) :
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) :=
sorry

end NUMINAMATH_GPT_tan_half_angle_sum_identity_l2191_219146


namespace NUMINAMATH_GPT_simplify_and_calculate_expression_l2191_219199

variable (a b : ℤ)

theorem simplify_and_calculate_expression (h_a : a = -3) (h_b : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 :=
by
  -- We include the proof steps here to achieve the final result.
  sorry

end NUMINAMATH_GPT_simplify_and_calculate_expression_l2191_219199


namespace NUMINAMATH_GPT_cos_of_vector_dot_product_l2191_219196

open Real

noncomputable def cos_value (x : ℝ) : ℝ := cos (x + π / 4)

theorem cos_of_vector_dot_product (x : ℝ)
  (h1 : π / 4 < x)
  (h2 : x < π / 2)
  (h3 : (sqrt 2) * cos x + (sqrt 2) * sin x = 8 / 5) :
  cos_value x = - 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_vector_dot_product_l2191_219196


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_l2191_219141

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 2 = 2) →
  (a 4 = 4) →
  (∀ n, a n = n) →
  (∀ n, b n = 2 ^ (a n)) →
  (∀ n, S n = 2 * (2 ^ n - 1)) :=
by
  intros h1 h2 h3 h4
  -- Proof part is skipped
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_l2191_219141


namespace NUMINAMATH_GPT_box_height_l2191_219162

theorem box_height (h : ℝ) :
  ∃ (h : ℝ), 
  let large_sphere_radius := 3
  let small_sphere_radius := 1.5
  let box_width := 6
  h = 12 := 
sorry

end NUMINAMATH_GPT_box_height_l2191_219162


namespace NUMINAMATH_GPT_net_progress_l2191_219112

-- Define the conditions as properties
def lost_yards : ℕ := 5
def gained_yards : ℕ := 10

-- Prove that the team's net progress is 5 yards
theorem net_progress : (gained_yards - lost_yards) = 5 :=
by
  sorry

end NUMINAMATH_GPT_net_progress_l2191_219112


namespace NUMINAMATH_GPT_jake_weight_l2191_219149

theorem jake_weight (J S B : ℝ) (h1 : J - 8 = 2 * S)
                            (h2 : B = 2 * J + 6)
                            (h3 : J + S + B = 480)
                            (h4 : B = 1.25 * S) :
  J = 230 :=
by
  sorry

end NUMINAMATH_GPT_jake_weight_l2191_219149


namespace NUMINAMATH_GPT_p_implies_q_l2191_219109

def p (x : ℝ) := 0 < x ∧ x < 5
def q (x : ℝ) := -5 < x - 2 ∧ x - 2 < 5

theorem p_implies_q (x : ℝ) (h : p x) : q x :=
  by sorry

end NUMINAMATH_GPT_p_implies_q_l2191_219109


namespace NUMINAMATH_GPT_scientific_notation_of_10900_l2191_219124

theorem scientific_notation_of_10900 : ∃ (x : ℝ) (n : ℤ), 10900 = x * 10^n ∧ x = 1.09 ∧ n = 4 := by
  use 1.09
  use 4
  sorry

end NUMINAMATH_GPT_scientific_notation_of_10900_l2191_219124


namespace NUMINAMATH_GPT_mod_inverse_identity_l2191_219111

theorem mod_inverse_identity : 
  (1 / 5 + 1 / 5^2) % 31 = 26 :=
by
  sorry

end NUMINAMATH_GPT_mod_inverse_identity_l2191_219111


namespace NUMINAMATH_GPT_fraction_susan_can_eat_l2191_219143

theorem fraction_susan_can_eat
  (v t n nf : ℕ)
  (h₁ : v = 6)
  (h₂ : n = 4)
  (h₃ : 1/3 * t = v)
  (h₄ : nf = v - n) :
  nf / t = 1 / 9 :=
sorry

end NUMINAMATH_GPT_fraction_susan_can_eat_l2191_219143


namespace NUMINAMATH_GPT_total_distance_traveled_l2191_219117

theorem total_distance_traveled :
  let car_speed1 := 90
  let car_time1 := 2
  let car_speed2 := 60
  let car_time2 := 1
  let train_speed := 100
  let train_time := 2.5
  let distance_car1 := car_speed1 * car_time1
  let distance_car2 := car_speed2 * car_time2
  let distance_train := train_speed * train_time
  distance_car1 + distance_car2 + distance_train = 490 := by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l2191_219117


namespace NUMINAMATH_GPT_ball_bounce_height_l2191_219104

theorem ball_bounce_height
  (k : ℕ) 
  (h1 : 20 * (2 / 3 : ℝ)^k < 2) : 
  k = 7 :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l2191_219104


namespace NUMINAMATH_GPT_set_roster_method_l2191_219166

open Set

theorem set_roster_method :
  { m : ℤ | ∃ n : ℕ, 12 = n * (m + 1) } = {0, 1, 2, 3, 5, 11} :=
  sorry

end NUMINAMATH_GPT_set_roster_method_l2191_219166


namespace NUMINAMATH_GPT_not_divisible_l2191_219175

theorem not_divisible (n : ℕ) : ¬ ((4^n - 1) ∣ (5^n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_l2191_219175


namespace NUMINAMATH_GPT_value_of_a_minus_b_l2191_219150

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l2191_219150


namespace NUMINAMATH_GPT_girls_picked_more_l2191_219158

variable (N I A V : ℕ)

theorem girls_picked_more (h1 : N > A) (h2 : N > V) (h3 : N > I)
                         (h4 : I ≥ A) (h5 : I ≥ V) (h6 : A > V) :
  N + I > A + V := by
  sorry

end NUMINAMATH_GPT_girls_picked_more_l2191_219158


namespace NUMINAMATH_GPT_solution_of_system_l2191_219185

theorem solution_of_system :
  (∀ x : ℝ,
    (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2)
    → x < 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l2191_219185


namespace NUMINAMATH_GPT_total_tablets_l2191_219191

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end NUMINAMATH_GPT_total_tablets_l2191_219191


namespace NUMINAMATH_GPT_wheel_radius_correct_l2191_219177
noncomputable def wheel_radius (total_distance : ℝ) (n_revolutions : ℕ) : ℝ :=
  total_distance / (n_revolutions * 2 * Real.pi)

theorem wheel_radius_correct :
  wheel_radius 450.56 320 = 0.224 :=
by
  sorry

end NUMINAMATH_GPT_wheel_radius_correct_l2191_219177


namespace NUMINAMATH_GPT_remaining_amount_after_shopping_l2191_219132

theorem remaining_amount_after_shopping (initial_amount spent_percentage remaining_amount : ℝ)
  (h_initial : initial_amount = 4000)
  (h_spent : spent_percentage = 0.30)
  (h_remaining : remaining_amount = 2800) :
  initial_amount - (spent_percentage * initial_amount) = remaining_amount :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_after_shopping_l2191_219132


namespace NUMINAMATH_GPT_spending_Mar_Apr_May_l2191_219187

-- Define the expenditures at given points
def e_Feb : ℝ := 0.7
def e_Mar : ℝ := 1.2
def e_May : ℝ := 4.4

-- Define the amount spent from March to May
def amount_spent_Mar_Apr_May := e_May - e_Feb

-- The main theorem to prove
theorem spending_Mar_Apr_May : amount_spent_Mar_Apr_May = 3.7 := by
  sorry

end NUMINAMATH_GPT_spending_Mar_Apr_May_l2191_219187


namespace NUMINAMATH_GPT_april_revenue_l2191_219160

def revenue_after_tax (initial_roses : ℕ) (initial_tulips : ℕ) (initial_daisies : ℕ)
                      (final_roses : ℕ) (final_tulips : ℕ) (final_daisies : ℕ)
                      (price_rose : ℝ) (price_tulip : ℝ) (price_daisy : ℝ) (tax_rate : ℝ) : ℝ :=
(price_rose * (initial_roses - final_roses) + price_tulip * (initial_tulips - final_tulips) + price_daisy * (initial_daisies - final_daisies)) * (1 + tax_rate)

theorem april_revenue :
  revenue_after_tax 13 10 8 4 3 1 4 3 2 0.10 = 78.10 := by
  sorry

end NUMINAMATH_GPT_april_revenue_l2191_219160


namespace NUMINAMATH_GPT_sum_of_997_lemons_l2191_219174

-- Define x and y as functions of k
def x (k : ℕ) := 1 + 9 * k
def y (k : ℕ) := 110 - 7 * k

-- The theorem we need to prove
theorem sum_of_997_lemons :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ 15 ∧ 7 * (x k) + 9 * (y k) = 997 := 
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_sum_of_997_lemons_l2191_219174


namespace NUMINAMATH_GPT_homework_time_decrease_l2191_219110

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_homework_time_decrease_l2191_219110


namespace NUMINAMATH_GPT_problem_a_proof_l2191_219164

variables {A B C D M K : Point}
variables {triangle_ABC : Triangle A B C}
variables {incircle : Circle} (ht : touches incircle AC D) 
variables (hdm : diameter incircle D M) 
variables (bm_line : Line B M) (intersect_bm_ac : intersects bm_line AC K)

theorem problem_a_proof : 
  AK = DC :=
sorry

end NUMINAMATH_GPT_problem_a_proof_l2191_219164


namespace NUMINAMATH_GPT_smallest_m_div_18_l2191_219108

noncomputable def smallest_multiple_18 : ℕ :=
  900

theorem smallest_m_div_18 : (∃ m: ℕ, (m % 18 = 0) ∧ (∀ d ∈ m.digits 10, d = 9 ∨ d = 0) ∧ ∀ k: ℕ, k % 18 = 0 → (∀ d ∈ k.digits 10, d = 9 ∨ d = 0) → m ≤ k) → 900 / 18 = 50 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_m_div_18_l2191_219108


namespace NUMINAMATH_GPT_intersection_A_B_l2191_219195

def A : Set ℝ := { x | abs x < 3 }
def B : Set ℝ := { x | 2 - x > 0 }

theorem intersection_A_B : A ∩ B = { x : ℝ | -3 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2191_219195


namespace NUMINAMATH_GPT_coupon_savings_difference_l2191_219121

-- Definitions based on conditions
def P (p : ℝ) := 120 + p
def savings_coupon_A (p : ℝ) := 24 + 0.20 * p
def savings_coupon_B := 35
def savings_coupon_C (p : ℝ) := 0.30 * p

-- Conditions
def condition_A_saves_at_least_B (p : ℝ) := savings_coupon_A p ≥ savings_coupon_B
def condition_A_saves_at_least_C (p : ℝ) := savings_coupon_A p ≥ savings_coupon_C p

-- Proof problem
theorem coupon_savings_difference :
  ∀ (p : ℝ), 55 ≤ p ∧ p ≤ 240 → (P 240 - P 55) = 185 :=
by
  sorry

end NUMINAMATH_GPT_coupon_savings_difference_l2191_219121


namespace NUMINAMATH_GPT_pedro_more_squares_l2191_219142

theorem pedro_more_squares (jesus_squares : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : jesus_squares = 60) (h2 : linden_squares = 75) (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end NUMINAMATH_GPT_pedro_more_squares_l2191_219142


namespace NUMINAMATH_GPT_log_inequality_l2191_219182

theorem log_inequality : 
  ∀ (logπ2 log2π : ℝ), logπ2 = 1 / log2π → 0 < logπ2 → 0 < log2π → (1 / logπ2 + 1 / log2π > 2) :=
by
  intros logπ2 log2π h1 h2 h3
  have h4: logπ2 = 1 / log2π := h1
  have h5: 0 < logπ2 := h2
  have h6: 0 < log2π := h3
  -- To be completed with the actual proof steps if needed
  sorry

end NUMINAMATH_GPT_log_inequality_l2191_219182


namespace NUMINAMATH_GPT_fraction_simplifies_l2191_219131

-- Define the integers
def a : ℤ := 1632
def b : ℤ := 1625
def c : ℤ := 1645
def d : ℤ := 1612

-- Define the theorem to prove
theorem fraction_simplifies :
  (a^2 - b^2) / (c^2 - d^2) = 7 / 33 := by
  sorry

end NUMINAMATH_GPT_fraction_simplifies_l2191_219131


namespace NUMINAMATH_GPT_constant_k_value_l2191_219118

theorem constant_k_value 
  (S : ℕ → ℕ)
  (h : ∀ n : ℕ, S n = 4 * 3^(n + 1) - k) :
  k = 12 :=
sorry

end NUMINAMATH_GPT_constant_k_value_l2191_219118


namespace NUMINAMATH_GPT_tree_planting_activity_l2191_219193

variables (trees_first_group trees_second_group people_first_group people_second_group : ℕ)
variable (average_trees_per_person_first_group average_trees_per_person_second_group : ℕ)

theorem tree_planting_activity :
  trees_first_group = 12 →
  trees_second_group = 36 →
  people_second_group = people_first_group + 6 →
  average_trees_per_person_first_group = trees_first_group / people_first_group →
  average_trees_per_person_second_group = trees_second_group / people_second_group →
  average_trees_per_person_first_group = average_trees_per_person_second_group →
  people_first_group = 3 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_tree_planting_activity_l2191_219193


namespace NUMINAMATH_GPT_find_y_value_l2191_219113

theorem find_y_value : 
  (15^2 * 8^3) / y = 450 → y = 256 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l2191_219113


namespace NUMINAMATH_GPT_min_value_y_l2191_219120

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 2) : 4 * a + b ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_y_l2191_219120


namespace NUMINAMATH_GPT_terminal_side_quadrant_l2191_219133

theorem terminal_side_quadrant (α : ℝ) (h : α = 2) : 
  90 < α * (180 / Real.pi) ∧ α * (180 / Real.pi) < 180 := 
by
  sorry

end NUMINAMATH_GPT_terminal_side_quadrant_l2191_219133


namespace NUMINAMATH_GPT_jeff_total_run_is_290_l2191_219154

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end NUMINAMATH_GPT_jeff_total_run_is_290_l2191_219154


namespace NUMINAMATH_GPT_calculate_expression_l2191_219159

variables (a b : ℝ) -- declaring variables a and b to be real numbers

theorem calculate_expression :
  (-a * b^2) ^ 3 + (a * b^2) * (a * b) ^ 2 * (-2 * b) ^ 2 = 3 * a^3 * b^6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2191_219159
