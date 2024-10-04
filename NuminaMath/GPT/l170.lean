import Mathlib

namespace graph_not_pass_through_second_quadrant_l170_170752

theorem graph_not_pass_through_second_quadrant 
    (k : ℝ) (b : ℝ) (h1 : k = 1) (h2 : b = -2) : 
    ¬ ∃ (x y : ℝ), y = k * x + b ∧ x < 0 ∧ y > 0 := 
by
  sorry

end graph_not_pass_through_second_quadrant_l170_170752


namespace minimum_value_property_l170_170976

noncomputable def min_value_expression (x : ℝ) (h : x > 10) : ℝ :=
  (x^2 + 36) / (x - 10)

noncomputable def min_value : ℝ := 4 * Real.sqrt 34 + 20

theorem minimum_value_property (x : ℝ) (h : x > 10) :
  min_value_expression x h >= min_value := by
  sorry

end minimum_value_property_l170_170976


namespace temperature_difference_l170_170431

theorem temperature_difference (last_night_temp current_temp : ℤ) 
  (h_last_night : last_night_temp = -5) 
  (h_current : current_temp = 3) : 
  current_temp - last_night_temp = 8 := 
by {
  rw [h_last_night, h_current],
  norm_num,
}

end temperature_difference_l170_170431


namespace value_of_expression_l170_170640

variable {a : ℝ}

theorem value_of_expression (h : a^2 + 2 * a - 1 = 0) : 2 * a^2 + 4 * a - 2024 = -2022 :=
by
  sorry

end value_of_expression_l170_170640


namespace ratio_of_sopranos_to_altos_l170_170904

theorem ratio_of_sopranos_to_altos (S A : ℕ) :
  (10 = 5 * S) ∧ (15 = 5 * A) → (S : ℚ) / (A : ℚ) = 2 / 3 :=
by sorry

end ratio_of_sopranos_to_altos_l170_170904


namespace point_in_first_quadrant_l170_170598

def i_rotates_90_deg (z : ℂ) : ℂ :=
  i * z

theorem point_in_first_quadrant :
  let z := 2 - 3 * i
  in i_rotates_90_deg z = 3 + 2 * i ∧ re (i_rotates_90_deg z) > 0 ∧ im (i_rotates_90_deg z) > 0 := 
by
  sorry

end point_in_first_quadrant_l170_170598


namespace reach_64_from_2_cannot_reach_2_2011_from_2_l170_170394

theorem reach_64_from_2 : ∃ (steps : list (ℝ → ℝ)), (∀ step ∈ steps, step 2 = 64) :=
by sorry

theorem cannot_reach_2_2011_from_2 : ¬ ∃ (steps : list (ℝ → ℝ)), (∀ step ∈ steps, step 2 = 2^2011) :=
by sorry

end reach_64_from_2_cannot_reach_2_2011_from_2_l170_170394


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170038

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170038


namespace cube_surface_area_l170_170878

theorem cube_surface_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end cube_surface_area_l170_170878


namespace infinite_sum_equals_one_l170_170988

noncomputable def T : ℕ → ℝ
| 0 := 0
| (n + 1) := T n + 1/(n + 1) / (n + 1)!

theorem infinite_sum_equals_one :
  ∑' n : ℕ, 1 / ((n + 2) * T (n + 1) * T (n + 2)) = 1 :=
by
  sorry

end infinite_sum_equals_one_l170_170988


namespace sum_of_digits_of_841_is_13_l170_170963

theorem sum_of_digits_of_841_is_13 :
  let smallest_int := 841 in
  (smallest_int % 10 = 1 ∧  -- the last digit checks
  (smallest_int / 10 % 10 = 4) ∧  -- tens digit check
  (smallest_int / 100 = 8)) →  -- hundreds digit check
  (8 + 4 + 1 = 13) := 
by {
  intros,
  dsimp at *,
  sorry
}

end sum_of_digits_of_841_is_13_l170_170963


namespace tigers_home_games_l170_170743

-- Definitions based on the conditions
def losses : ℕ := 12
def ties : ℕ := losses / 2
def wins : ℕ := 38

-- Statement to prove
theorem tigers_home_games : losses + ties + wins = 56 := by
  sorry

end tigers_home_games_l170_170743


namespace problem_statement_l170_170066

def is_proposition (p : Prop) : Prop := 
  ∃ (b : bool), (if b then p else ¬p)

def not_proposition_1 : Prop := 
  ¬ is_proposition (x^2 = 3)

def not_proposition_2 : Prop := 
  ¬ is_proposition (∃ l1 l2 : line, ∀ x : point, l1 ∩ x ∧ l2 ∩ x → parallel l1 l2)

def not_proposition_4 : Prop := 
  ¬ is_proposition (5 * x - 3 > 6)

theorem problem_statement :
  ¬is_proposition (x^2 = 3) ∧
  ¬is_proposition (∃ l1 l2 : line, ∀ x : point, l1 ∩ x ∧ l2 ∩ x → parallel l1 l2) ∧
  ¬is_proposition (5 * x - 3 > 6) :=
sorry

end problem_statement_l170_170066


namespace function_decreasing_interval_l170_170226

theorem function_decreasing_interval (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ) (hx0 : x > 0), (ln x - (1 / 2) * a * x^2 - 2 * x).deriv ' f x < 0) ↔
  (a ∈ set.Ioo (-1:ℝ) 0 ∪ set.Ioi (0:ℝ)) :=
by {
  sorry
}

end function_decreasing_interval_l170_170226


namespace remaining_money_l170_170716

-- Definitions
def cost_per_app : ℕ := 4
def num_apps : ℕ := 15
def total_money : ℕ := 66

-- Theorem
theorem remaining_money : total_money - (num_apps * cost_per_app) = 6 := by
  sorry

end remaining_money_l170_170716


namespace imo_1995_p6_l170_170179

-- Definitions
def nonCollinear {α} [LinearOrderedField α] (A B C: α × α) : Prop :=
A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
(¬ ∃ k, B.1 = A.1 + k * (C.1 - A.1) ∧ B.2 = A.2 + k * (C.2 - A.2))

def triangleArea {α} [LinearOrderedField α] (A B C: α × α) : α :=
(abs (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

def satisfiesCondition {α} [LinearOrderedField α] (points: Finset (α × α)) (r: Fin points.card → α) : Prop :=
∀ (i j k: Fin points.card), i < j ∧ j < k ∧ nonCollinear (points.nth i) (points.nth j) (points.nth k) →
triangleArea (points.nth i) (points.nth j) (points.nth k) = r i + r j + r k

-- Theorem
theorem imo_1995_p6 {α} [LinearOrderedField α]:
  ∀ (n: ℕ), n > 3 →
  ∃ (points: Finset (α × α)) (r: Fin n → α), points.card = n ∧ satisfiesCondition points r → 
  n = 4 :=
by
  intros n hn points r hcard hcond
  sorry

end imo_1995_p6_l170_170179


namespace total_amount_collected_in_paise_total_amount_collected_in_rupees_l170_170497

-- Definitions and conditions
def num_members : ℕ := 96
def contribution_per_member : ℕ := 96
def total_paise_collected : ℕ := num_members * contribution_per_member
def total_rupees_collected : ℚ := total_paise_collected / 100

-- Theorem stating the total amount collected
theorem total_amount_collected_in_paise :
  total_paise_collected = 9216 := by sorry

theorem total_amount_collected_in_rupees :
  total_rupees_collected = 92.16 := by sorry

end total_amount_collected_in_paise_total_amount_collected_in_rupees_l170_170497


namespace number_of_valid_terminating_decimals_with_nonzero_thousandths_digit_l170_170195

theorem number_of_valid_terminating_decimals_with_nonzero_thousandths_digit :
  {n : ℕ // (∀ k, n = 2^k ∨ n = 5^k ∨ (∃ i j, n = 2^i * 5^j)) ∧
               (n ≤ 200) ∧ 
               (∃ d, (1 / (↑n : ℚ)).nth_d 3 = d) ∧
               (∀ d, (1 / (↑n : ℚ)).nth_d 3 ≠ 0)}.card = 17 :=
sorry

end number_of_valid_terminating_decimals_with_nonzero_thousandths_digit_l170_170195


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170027

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170027


namespace hexagon_cyclic_all_six_conditions_hexagon_cyclic_five_conditions_l170_170986

-- Definitions of the geometric properties in the problem
variables {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space F]

def parallel (a b : Type) [metric_space a] [metric_space b] := sorry
def equal_length (a b : Type) [metric_space a] [metric_space b] := sorry
def cyclic (a b c d e f : Type) [metric_space a] [metric_space b] [metric_space c]
  [metric_space d] [metric_space e] [metric_space f] := sorry

-- Problem (a): All six conditions
theorem hexagon_cyclic_all_six_conditions
  (h1 : parallel AB DE)
  (h2 : equal_length AE BD)
  (h3 : parallel BC EF)
  (h4 : equal_length BF CE)
  (h5 : parallel CD FA)
  (h6 : equal_length CA DF) : 
  cyclic A B C D E F :=
sorry

-- Problem (b): Any five of the six conditions
theorem hexagon_cyclic_five_conditions
  (h1 : parallel AB DE)
  (h2 : equal_length AE BD)
  (h3 : parallel BC EF)
  (h4 : equal_length BF CE)
  (h5 : (parallel CD FA) ∨ (equal_length CA DF)) :
  cyclic A B C D E F :=
sorry

end hexagon_cyclic_all_six_conditions_hexagon_cyclic_five_conditions_l170_170986


namespace problem1_problem2_l170_170625

-- Define the function f(x) in terms of x and a
noncomputable def f (x a : ℝ) := Real.log x + (2 * a) / x

-- Define the derivative of f(x) with respect to x
noncomputable def f' (x a : ℝ) := (1 / x) - (2 * a) / (x^2)

-- First Problem: If f(x) is increasing on [4, +∞), then a ≤ 2
theorem problem1 (a : ℝ) : (∀ x ∈ Icc (4 : ℝ) ∞, f' x a ≥ 0) → a ≤ 2 := sorry

-- Second Problem: If the minimum value of f(x) on [1, e] is 3, then a = e
theorem problem2 (a : ℝ) : (∀ x ∈ Icc (1 : ℝ) Real.exp 1, 
   ∀ y ∈ Icc (1 : ℝ) Real.exp 1, f y a ≥ f x a) ∧ f 1 a = 3 ∧ f (Real.exp 1) a = 3 → a = Real.exp 1 := sorry

end problem1_problem2_l170_170625


namespace infinite_non_square_terms_l170_170354

theorem infinite_non_square_terms (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ᶠ n in at_top, ¬ is_square (n^3 + a * n^2 + b * n + c) :=
by
  sorry

end infinite_non_square_terms_l170_170354


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170014

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170014


namespace trapezoid_area_l170_170859

theorem trapezoid_area (area_PQR : ℝ) (area_small_triangle : ℝ) (num_small_triangles : ℕ)
    (total_area_small_triangles : ℝ) (area_PKL : ℝ) (area_PKQL : ℝ) :
    is_isosceles_triangle P Q R →
    (area_PQR = 90) →
    (area_small_triangle = 2) →
    (num_small_triangles = 9) →
    (total_area_small_triangles = num_small_triangles * area_small_triangle) →
    (area_PKL = (1/√15)^2 * area_PQR) →
    (area_PKQL = area_PQR - area_PKL) →
    area_PKQL = 84 :=
by
  intros h_isosceles h_area_PQR h_area_small_triangle h_num_small_triangles
         h_total_area_small_triangles h_area_PKL h_area_PKQL
  sorry

end trapezoid_area_l170_170859


namespace part1_part2_l170_170633

variables (k : ℝ)

def a : ℝ × ℝ := (-2, 2)
def b : ℝ × ℝ := (5, k)

/-- Part 1: If a is perpendicular to b, then k = 5 -/
theorem part1 (h : (a.1 * b.1 + a.2 * b.2) = 0) : k = 5 :=
by sorry

/-- Part 2: If a + 2b is parallel to 2a - b, then k = -5 -/
theorem part2 (h : (a.1 + 2 * b.1) * (2 * a.2 - b.2) - (a.2 + 2 * b.2) * (2 * a.1 - b.1) = 0) : k = -5 :=
by sorry

end part1_part2_l170_170633


namespace wet_surface_area_is_correct_l170_170481

-- Define the dimensions of the cistern
def cistern_length : ℝ := 6  -- in meters
def cistern_width  : ℝ := 4  -- in meters
def water_depth    : ℝ := 1.25  -- in meters

-- Compute areas for each surface in contact with water
def bottom_area : ℝ := cistern_length * cistern_width
def long_sides_area : ℝ := 2 * (cistern_length * water_depth)
def short_sides_area : ℝ := 2 * (cistern_width * water_depth)

-- Calculate the total area of the wet surface
def total_wet_surface_area : ℝ := bottom_area + long_sides_area + short_sides_area

-- Statement to prove
theorem wet_surface_area_is_correct : total_wet_surface_area = 49 := by
  sorry

end wet_surface_area_is_correct_l170_170481


namespace sqrt_50_between_7_and_8_l170_170798

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170798


namespace badgers_win_prob_l170_170397

def probability_badgers_win_at_least_four_games : ℚ := 1 / 2

theorem badgers_win_prob :
  ∀ (B C : ℕ), let n := 7 in let p := 1 / 2 in 
  (B + C = n) → 
  (B ≥ 4) → 
  (1 / 2) = probability_badgers_win_at_least_four_games := by
  intros B C n p h1 h2
  sorry

end badgers_win_prob_l170_170397


namespace polygon_sides_l170_170919

theorem polygon_sides (P s : ℕ) (h₁ : P = 180) (h₂ : s = 15) : P / s = 12 :=
by
  rw [h₁, h₂]
  exact (Nat.div_eq_of_eq_mul_left (by decide) rfl).symm

end polygon_sides_l170_170919


namespace units_digit_of_fraction_l170_170059

theorem units_digit_of_fraction :
  let factors := [22, 23, 24, 25, 26, 27]
  let divisor := 2000
  let product := List.prod factors
  (product % divisor) % 10 = 8 :=
by
  let factors := [22, 23, 24, 25, 26, 27]
  let divisor := 2000
  let product := factors.product
  have h : product % divisor % 10 = 8 := sorry
  exact h

end units_digit_of_fraction_l170_170059


namespace sqrt_50_between_consecutive_integers_product_l170_170840

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170840


namespace find_j_l170_170080

theorem find_j (n j : ℕ) (h_n_pos : n > 0) (h_j_pos : j > 0) (h_rem : n % j = 28) (h_div : n / j = 142 ∧ (↑n / ↑j : ℝ) = 142.07) : j = 400 :=
by {
  sorry
}

end find_j_l170_170080


namespace parallel_vectors_l170_170200

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors {m : ℝ} (h : (∃ k : ℝ, vector_a = k • vector_b m)) : m = -2 :=
by
  sorry

end parallel_vectors_l170_170200


namespace skittles_division_l170_170642

theorem skittles_division:
  ∀ (Skittles people : ℕ), 
  Skittles = 25 ∧ people = 5 → Skittles / people = 5 :=
by
  intros Skittles people h
  cases h with h1 h2
  rw [h1, h2]
  norm_num

end skittles_division_l170_170642


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170276

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170276


namespace area_inside_C_outside_A_B_l170_170551

-- Define the circles and their properties
def Circle (centre : ℝ × ℝ) (radius : ℝ) : Prop := true

-- Given conditions
def CirclesTangent (A B : (ℝ × ℝ) × ℝ) : Prop :=
  let ((xa, ya), ra) := A in
  let ((xb, yb), rb) := B in
  (xa - xb)^2 + (ya - yb)^2 = (ra + rb)^2

def TangentAtMidpoint (A B C : (ℝ × ℝ) × ℝ) : Prop :=
  let ((xa, ya), ra) := A in
  let ((xb, yb), rb) := B in
  let ((xc, yc), rc) := C in
  xc = (xa + xb) / 2 ∧ yc = (ya + yb) / 2 + 2 ∧ rc = 2

-- Define the circles
def circleA := ((0, 0), 2) -- Circle A at origin with radius 2
def circleB := ((4, 0), 2) -- Circle B such that it is tangent to A
def circleC := ((2, 2), 2) -- Circle C is tangent to midpoint of AB

-- Main theorem
theorem area_inside_C_outside_A_B :
  CirclesTangent circleA circleB ∧ TangentAtMidpoint circleA circleB circleC →
  let area := Real.pi * 2^2 in
  let overlap := (Real.pi - 2) * 2 in
  area - overlap = 2 * Real.pi + 4 :=
by
  sorry

end area_inside_C_outside_A_B_l170_170551


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170037

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170037


namespace year_proof_l170_170928

variable (n : ℕ)

def packaging_waste_exceeds_threshold (y0 : ℝ) (rate : ℝ) (threshold : ℝ) : Prop :=
  let y := y0 * (rate^n)
  y > threshold

noncomputable def year_when_waste_exceeds := 
  let initial_year := 2015
  let y0 := 4 * 10^6 -- in tons
  let rate := (3.0 / 2.0) -- growth rate per year
  let threshold := 40 * 10^6 -- threshold in tons
  ∃ n, packaging_waste_exceeds_threshold n y0 rate threshold ∧ (initial_year + n = 2021)

theorem year_proof : year_when_waste_exceeds :=
  sorry

end year_proof_l170_170928


namespace tangent_line_parabola_l170_170192

-- Define the condition that line 4x + 6y + k = 0 must be tangent to the parabola y^2 = 16x
theorem tangent_line_parabola (k : ℝ) :
  (4 * x + 6 * y + k = 0) → (y^2 = 16 * x) → k = 36 :=
by
  -- Definitions
  intros line_eq parabola_eq
  
  -- Assume the necessary components of the problem conditions
  have h1 : ∀ y, x = -(6 * y + k) / 4, sorry
  have h2 : ∀ y, y^2 + 24 * y + 4 * k = 0, sorry
  have discriminant_zero : 24^2 - 4 * 1 * (4 * k) = 0, sorry
  
  -- Therefore, showing k = 36 as the correct answer
  sorry

end tangent_line_parabola_l170_170192


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170278

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170278


namespace find_15th_term_l170_170750

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term_l170_170750


namespace possible_values_of_k_l170_170269

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170269


namespace range_s_l170_170460

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_s : set.range s = set.Ioi 0 := by
  sorry

end range_s_l170_170460


namespace positive_difference_two_largest_prime_factors_l170_170866

theorem positive_difference_two_largest_prime_factors :
  let n := 187489 in
  (factorize n = [47, 61, 5, 13]) →
  (is_prime 47) ∧ (is_prime 61) →
  (47 < 61) →
  (61 - 47 = 14) :=
by
  intros n h_factorize h_primes h_order
  have h := factorize n
  rw h_factorize at h
  sorry

end positive_difference_two_largest_prime_factors_l170_170866


namespace orange_pyramid_total_l170_170909

theorem orange_pyramid_total (base_length : ℕ) (base_width : ℕ) (height : ℕ)
  (h_base_length : base_length = 6) (h_base_width : base_width = 9) (h_height : height = 7) :
  (∑ i in Finset.range height, (base_length - i) * (base_width - i)) = 155 :=
by
  rw [h_base_length, h_base_width, h_height]
  sorry

end orange_pyramid_total_l170_170909


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170016

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170016


namespace constant_term_expansion_l170_170409

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  let t_r := (λ r, ((nat.choose 8 r) * (1/2)^r * x^(4 - r))) in
  t_r 4 = 35 / 8 :=
by sorry

end constant_term_expansion_l170_170409


namespace k_positive_first_third_quadrants_l170_170289

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170289


namespace elevator_time_to_bottom_l170_170724

theorem elevator_time_to_bottom :
  ∀ (floors : ℕ) (first_half_time: ℕ) 
  (next_floors: ℕ) (next_floors_time_per_floor: ℕ) 
  (final_floors: ℕ) (final_floors_time_per_floor: ℕ)
  (total_floors: ℕ) (total_time_in_hours: ℕ),
  floors = 20 →
  first_half_time = 15 →
  next_floors = 5 →
  next_floors_time_per_floor = 5 →
  final_floors = 5 →
  final_floors_time_per_floor = 16 →
  total_floors = first_half_time +
                 next_floors * next_floors_time_per_floor +
                 final_floors * final_floors_time_per_floor →
  total_time_in_hours = total_floors / 60 →
  total_time_in_hours = 2 :=
begin
  intros,
  sorry
end

end elevator_time_to_bottom_l170_170724


namespace find_digit_property_l170_170182

theorem find_digit_property (a x : ℕ) (h : 10 * a + x = a + x + a * x) : x = 9 :=
sorry

end find_digit_property_l170_170182


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170007

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170007


namespace systematic_sampling_seat_number_l170_170552

theorem systematic_sampling_seat_number (total_students sample_size : ℕ) (sample : set ℕ) (known_seats : set ℕ) (interval fourth_student: ℕ) :
  total_students = 56 → 
  sample_size = 4 → 
  known_seats = {3, 17, 45} → 
  interval = total_students / sample_size →
  17 ∈ known_seats →
  fourth_student = 17 + interval → 
  fourth_student = 31 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h4, h6]
  exact rfl

end systematic_sampling_seat_number_l170_170552


namespace checkerboard_sum_l170_170493

def row_numbering (i j : ℕ) : ℕ := 20 * (i - 1) + j
def column_numbering (i j : ℕ) : ℕ := 15 * (j - 1) + i

theorem checkerboard_sum :
  (∑ (i, j) in [(1, 1), (5, 6), (9, 11), (13, 16), (15, 20)], row_numbering i j) = 809 :=
by sorry

end checkerboard_sum_l170_170493


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170044

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170044


namespace max_consecutive_specials_l170_170454

def is_special (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 20 ∧ ∀ (a b : ℕ), a.digits 10.length ≤ 10 →
                                   b.digits 10.length ≤ 11 →
                                   a * b ≠ n

theorem max_consecutive_specials : ∃ N, N = 10^9 - 1 ∧
  ∀ m, m ∈ (range N) → is_special (10^19 + m) :=
by
  -- placeholder for the proof
  sorry

end max_consecutive_specials_l170_170454


namespace general_formula_sequence_less_than_zero_maximum_sum_value_l170_170609

variable (n : ℕ)

-- Helper definition
def arithmetic_seq (d : ℤ) (a₁ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Conditions given in the problem
def a₁ : ℤ := 31
def a₄ : ℤ := 7
def d : ℤ := (a₄ - a₁) / 3

-- Definitions extracted from problem conditions
def an (n : ℕ) : ℤ := arithmetic_seq d a₁ n
def Sn (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Proving the general formula aₙ = -8n + 39
theorem general_formula :
  ∀ (n : ℕ), an n = -8 * n + 39 :=
by
  sorry

-- Proving when the sequence starts to be less than 0
theorem sequence_less_than_zero :
  ∀ (n : ℕ), n ≥ 5 → an n < 0 :=
by
  sorry

-- Proving that the sum Sn has a maximum value
theorem maximum_sum_value :
  Sn 4 = 76 ∧ ∀ (n : ℕ), Sn n ≤ 76 :=
by
  sorry

end general_formula_sequence_less_than_zero_maximum_sum_value_l170_170609


namespace speed_against_current_l170_170113

noncomputable def man's_speed_with_current : ℝ := 20
noncomputable def current_speed : ℝ := 1

theorem speed_against_current :
  (man's_speed_with_current - 2 * current_speed) = 18 := by
sorry

end speed_against_current_l170_170113


namespace beach_trip_ratio_l170_170920

theorem beach_trip_ratio :
  ∀ (total_students beach_students remaining_students : ℕ),
    total_students = 1000 →
    remaining_students = 250 →
    2 * remaining_students = total_students - beach_students →
    beach_students = 500 →
    (beach_students : ℚ) / total_students = 1 / 2 :=
by
  intros total_students beach_students remaining_students h1 h2 h3 h4
  rw [h4, h1]
  norm_num
  sorry

end beach_trip_ratio_l170_170920


namespace angle_MNA_eq_angle_MNB_l170_170407

-- Definitions to represent the conditions.
variables {C1 C2 : Type} [Circle C1] [Circle C2]
variables (M N : Point) (A : Point C1) (B : Point C2)

-- Statement incorporating the conditions.
theorem angle_MNA_eq_angle_MNB
  (hC1C2 : M ∈ C1 ∧ M ∈ C2 ∧ N ∈ C1 ∧ N ∈ C2 ∧ M ≠ N)
  (hMA_tangent_C2 : tangent_at_point (line_through M A) C2 M)
  (hMB_tangent_C1 : tangent_at_point (line_through M B) C1 M):
  angle M N A = angle M N B :=
sorry

end angle_MNA_eq_angle_MNB_l170_170407


namespace max_volume_and_corresponding_height_l170_170521

-- Definitions for conditions
def steel_bar_length : ℝ := 14.8
def side_length_difference : ℝ := 0.5

-- Definitions for short side length, long side length, and height
def short_side (x : ℝ) : ℝ := x
def long_side (x : ℝ) : ℝ := x + side_length_difference
def height (x : ℝ) : ℝ := (steel_bar_length - 4 * short_side (x) - 4 * long_side (x)) / 4

-- Volume as a function of x
def volume (x : ℝ) : ℝ := (short_side (x)) * (long_side (x)) * (height (x))

-- Proof statement
theorem max_volume_and_corresponding_height :
  ∀ (x : ℝ), 0 < x ∧ x < 1.6 → 
  volume (1) = 1.8 ∧ height (1) = 1.2 := by
  sorry

end max_volume_and_corresponding_height_l170_170521


namespace sum_of_roots_l170_170051

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l170_170051


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170004

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170004


namespace product_of_consecutive_integers_sqrt_50_l170_170792

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170792


namespace h_100_eq_35_l170_170160

noncomputable def h : ℕ → ℕ
| x := if h : (∃ n : ℕ, x = 2^n) then Nat.log 2 x else 1 + h (x + 1)

theorem h_100_eq_35 : h 100 = 35 :=
by {
  -- proof objectives would go here
  sorry
}

end h_100_eq_35_l170_170160


namespace probability_both_blue_buttons_selected_l170_170677

theorem probability_both_blue_buttons_selected :
  let initial_red_C := 6
  let initial_blue_C := 10
  let initial_total_C := initial_red_C + initial_blue_C
  let fraction_remaining_C := (4:ℚ)/5
  let remaining_buttons_C := (fraction_remaining_C * initial_total_C).nat_floor
  let removed_total := initial_total_C - remaining_buttons_C
  let removed_each := removed_total / 2
  let remaining_red_C := initial_red_C - removed_each
  let remaining_blue_C := initial_blue_C - removed_each
  let red_D := removed_each
  let blue_D := removed_each
  let probability_blue_C := remaining_blue_C / (remaining_red_C + remaining_blue_C)
  let probability_blue_D := blue_D / (red_D + blue_D)
  in probability_blue_C * probability_blue_D = 1 / 3 :=
by
  sorry

end probability_both_blue_buttons_selected_l170_170677


namespace neg_half_pow_and_two_pow_l170_170943

theorem neg_half_pow_and_two_pow (n : ℕ) (hn : n = 2016) : (-0.5) ^ n * 2 ^ (n + 1) = 2 :=
by 
  sorry

end neg_half_pow_and_two_pow_l170_170943


namespace cos_decreasing_intervals_l170_170166

open Real

def is_cos_decreasing_interval (k : ℤ) : Prop := 
  let f (x : ℝ) := cos (π / 4 - 2 * x)
  ∀ x y : ℝ, (k * π + π / 8 ≤ x) → (x ≤ k * π + 5 * π / 8) → 
             (k * π + π / 8 ≤ y) → (y ≤ k * π + 5 * π / 8) → 
             x < y → f x > f y

theorem cos_decreasing_intervals : ∀ k : ℤ, is_cos_decreasing_interval k :=
by
  sorry

end cos_decreasing_intervals_l170_170166


namespace exists_fixed_point_P_l170_170545

theorem exists_fixed_point_P 
  (O : Point) (r : ℝ) (hr : r = 1)
  (L : Line) (hL : ¬intersects O.circle(L))
  (M N : Point) (hMN : on_line L M ∧ on_line L N)
  (hMN_circle : circle_with_diameter_MN_touches_C M N O r) :
  ∃ P : Point, ∃ k : Real, 
    ∀ (M' N' : Point), on_line L M' → on_line L N' → circle_with_diameter_MN_touches_C M' N' O r → angle M' P N' = k :=
sorry

end exists_fixed_point_P_l170_170545


namespace line_in_first_and_third_quadrants_l170_170267

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170267


namespace eval_polynomial_at_3_l170_170452

def f (x : ℝ) : ℝ := 2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

theorem eval_polynomial_at_3 : f 3 = 130 :=
by
  -- proof can be completed here following proper steps or using Horner's method
  sorry

end eval_polynomial_at_3_l170_170452


namespace difference_max_min_l170_170150

noncomputable def average (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def process_sequence (seq : List ℝ) : ℝ :=
  let avg1 := average seq[0] seq[1]
  let avg2 := average avg1 seq[2]
  let avg3 := average avg2 seq[3]
  let avg4 := average avg3 seq[4]
  average avg4 seq[5]

example : process_sequence [1, 2, 3, 4, 5, 6] = 5.03125 := sorry

example : process_sequence [6, 5, 4, 3, 2, 1] = 1.96875 := sorry

theorem difference_max_min :
  abs (process_sequence [1, 2, 3, 4, 5, 6] - process_sequence [6, 5, 4, 3, 2, 1]) = 3.0625 :=
by 
  sorry

end difference_max_min_l170_170150


namespace quadrilateral_AD_length_l170_170662

-- Define the given quadrilateral and its properties
variables {A B C D : Type*} [EuclideanGeometry]
variables {AB CD BC AD : ℝ}
variables {angle_B angle_C : ℝ}
variables {is_right_angle : ℝ → Prop} {is_angle : ℝ → ℝ → Prop}
variables {length : Type* → Type* → ℝ}

-- Define the conditions
variables (hAB : length A B = 6) 
          (hBC : length B C = 10) 
          (hCD : length C D = 25) 
          (hAngleB : is_right_angle angle_B) 
          (hAngleC120 : is_angle angle_C 120)

-- The main theorem we want to prove
theorem quadrilateral_AD_length :
  (length A D) = Real.sqrt 351 :=
sorry

end quadrilateral_AD_length_l170_170662


namespace compute_n_v3_l170_170346

open Matrix

-- Definitions used in the conditions
def N : Matrix (Fin 2) (Fin 2) ℝ := sorry

def v1 : Fin 2 → ℝ := ![1, 2]
def v2 : Fin 2 → ℝ := ![4, -1]
def v3 : Fin 2 → ℝ := ![6, 3]

def n_v1 : Fin 2 → ℝ := ![-2, 4]
def n_v2 : Fin 2 → ℝ := ![3, -6]
def n_v3 : Fin 2 → ℝ := ![-1, 2]

-- Given conditions
axiom N_v1 : (N.mul_vec v1) = n_v1
axiom N_v2 : (N.mul_vec v2) = n_v2

-- Proof goal
theorem compute_n_v3 : (N.mul_vec v3) = n_v3 :=
sorry

end compute_n_v3_l170_170346


namespace determine_function_characterization_l170_170163

def isMultiplicative (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y

def infinitySymmetry (f : ℕ → ℕ) : Prop :=
  ∃ᶠ n in atTop, ∀ k < n, f k = f (n - k)

theorem determine_function_characterization :
  ∃ (f : ℕ → ℕ), isMultiplicative f ∧
    (∃ n, f n ≠ 0) ∧
    infinitySymmetry f ∧
    (∃ N : ℕ, ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, f n = N * (Nat.factorization n).find_d p) :=
sorry

end determine_function_characterization_l170_170163


namespace find_matrix_n_l170_170572

theorem find_matrix_n (N : Matrix (Fin 3) (Fin 3) ℚ) :
  N = ![ ![2, 5/14, 0], ![3/7, 1, 0], ![0, 0, 0.5] ] →
  N * ![ ![-4, 5, 0], ![6, -8, 0], ![0, 0, 2] ] = 1 :=
by
  intros hN
  rw hN
  -- proof omitted
  sorry

end find_matrix_n_l170_170572


namespace fibonacci_sum_identity_l170_170419

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0        := 1
| 1        := 1
| (n + 2)  := fibonacci (n + 1) + fibonacci n

-- The Lean statement of the given proof problem
theorem fibonacci_sum_identity :
  let a := fibonacci in
  (∑ n in Finset.range 2020, a n * a (n + 2)) - (∑ n in Finset.range 2019, (a (n + 1))^2) = 1 :=
sorry

end fibonacci_sum_identity_l170_170419


namespace sin_four_arcsin_one_four_eq_seven_sqrt_fifteen_over_thirty_two_l170_170178

theorem sin_four_arcsin_one_four_eq_seven_sqrt_fifteen_over_thirty_two :
  sin (4 * asin (1 / 4)) = (7 * Real.sqrt 15) / 32 :=
by
  sorry

end sin_four_arcsin_one_four_eq_seven_sqrt_fifteen_over_thirty_two_l170_170178


namespace k_positive_first_third_quadrants_l170_170290

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170290


namespace total_roasted_marshmallows_l170_170679

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l170_170679


namespace li_hai_walking_time_and_distance_l170_170997

noncomputable theory

def travel_conditions (x y : ℝ) : Prop :=
  x + y = 16 ∧ 50 * x + (250/3) * y = 1200

theorem li_hai_walking_time_and_distance :
  ∃ x y : ℝ, travel_conditions x y :=
begin
  sorry
end

end li_hai_walking_time_and_distance_l170_170997


namespace exists_special_M_l170_170561

theorem exists_special_M : ∃ M : ℕ, (∀ N : ℕ, (N < 10 ^ 1988) → (¬(divisible_by M N))) := by
  let M := (10 ^ 221 - 1)
  existsi M
  intro N hN
  sorry -- this will contain the proof

end exists_special_M_l170_170561


namespace line_in_first_and_third_quadrants_l170_170265

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170265


namespace slower_bus_speed_l170_170449

-- Definitions based on conditions
def length_bus : ℝ := 3125 -- meters
def speed_faster_bus : ℝ := 40 -- km/hr
def time_pass : ℝ := 50 -- seconds

-- Convert the necessary units
def distance_covered : ℝ := (2 * length_bus) / 1000 -- convert meters to km
def time_pass_hr : ℝ := time_pass / 3600 -- convert seconds to hours

-- The declaration we want to prove
theorem slower_bus_speed : ∃ (v_s : ℝ), v_s = 410 ∧ (speed_faster_bus + v_s) = distance_covered / time_pass_hr :=
by
  sorry

end slower_bus_speed_l170_170449


namespace smallest_positive_n_l170_170255

def a_0 : ℝ := (Real.sin (Real.pi / 45))^2

def a_sequence : ℕ → ℝ
| 0       := a_0
| (n + 1) := 4 * a_sequence n * (1 - a_sequence n)

theorem smallest_positive_n (n : ℕ) : (a_sequence n = a_0) ↔ (n = 12) :=
by
  -- Proof is omitted
  sorry

end smallest_positive_n_l170_170255


namespace calculate_area_closed_figure_l170_170218

-- Define the function f(x) as stated in the problem
noncomputable def f (x : ℝ) : ℝ :=
if (1/4 < x ∧ x ≤ 1) then sqrt x else x ^ 2

-- Define the integral calculation over the regions
def area_segment1 : ℝ := ∫ (x : ℝ) in (Set.Icc (1/4 : ℝ) 1), sqrt x
def area_segment2 : ℝ := ∫ (x : ℝ) in (Set.Icc 1 2), x ^ 2

-- Define the total area
noncomputable def total_area : ℝ := area_segment1 + area_segment2

theorem calculate_area_closed_figure :
    total_area = 35 / 12 := 
by 
  sorry

end calculate_area_closed_figure_l170_170218


namespace megan_probability_correct_number_l170_170369

/-- Define the set of possible last four digits, acknowledging repetition --/
def possible_last_four_digits := {d : ℕ // d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 8}

def num_combinations_last_four_digits : ℕ := 4^4

def num_combinations_first_three_digits : ℕ := 2

def total_num_possible_numbers : ℕ := num_combinations_first_three_digits * num_combinations_last_four_digits

theorem megan_probability_correct_number : 
  let probability_correct : ℚ := 1 / total_num_possible_numbers in
  probability_correct = 1 / 512 :=
by
  /- Proof goes here -/
  sorry

end megan_probability_correct_number_l170_170369


namespace barium_atoms_in_compound_l170_170101

noncomputable def barium_atoms (total_molecular_weight : ℝ) (weight_ba_per_atom : ℝ) (weight_br_per_atom : ℝ) (num_br_atoms : ℕ) : ℝ :=
  (total_molecular_weight - (num_br_atoms * weight_br_per_atom)) / weight_ba_per_atom

theorem barium_atoms_in_compound :
  barium_atoms 297 137.33 79.90 2 = 1 :=
by
  unfold barium_atoms
  norm_num
  sorry

end barium_atoms_in_compound_l170_170101


namespace line_passing_through_first_and_third_quadrants_l170_170293

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170293


namespace ratio_of_parts_l170_170730

theorem ratio_of_parts (N : ℝ) (h1 : (1/4) * (2/5) * N = 14) (h2 : 0.40 * N = 168) : (2/5) * N / N = 1 / 2.5 :=
by
  sorry

end ratio_of_parts_l170_170730


namespace enclosed_area_of_polygon_l170_170171

def polygon := [(0, 0), (5, 0), (5, 5), (0, 5), (0, 3), (3, 3), (3, 0), (0, 0)]

theorem enclosed_area_of_polygon :
  let area := 19 in
  polygon_encloses_area polygon area :=
by
  -- Define the polygon
  let p := [(0, 0), (5, 0), (5, 5), (0, 5), (0, 3), (3, 3), (3, 0), (0, 0)]
  -- Calculate the area
  let large_rectangle_area := 5 * 5
  let cut_out_rectangle_area := 3 * 2
  let total_area := large_rectangle_area - cut_out_rectangle_area
  have h : total_area = 19, by sorry
  exact h

end enclosed_area_of_polygon_l170_170171


namespace train_speed_l170_170119

theorem train_speed (length time : ℝ) (h_length : length = 120) (h_time : time = 11.999040076793857) :
  (length / time) * 3.6 = 36.003 :=
by
  sorry

end train_speed_l170_170119


namespace tan_add_sub_l170_170259

theorem tan_add_sub (γ β : ℝ) (h_tan_γ : Real.tan γ = 5) (h_tan_β : Real.tan β = 3) :
  Real.tan (γ + β) = -4/7 ∧ Real.tan (γ - β) = 1/8 :=
  by
    split
    {
      sorry
    }
    {
      sorry
    }

end tan_add_sub_l170_170259


namespace eat_both_veg_nonveg_l170_170651

theorem eat_both_veg_nonveg (A B C : ℕ) 
    (h1 : A = 28)         -- Total number of people who eat veg
    (h2 : B = 16):        -- Number of people who eat only vegetarian
    C = A - B :=          -- Number of people who eat both veg and non-veg
by {
  rw [h1, h2];
  exact rfl;
}

end eat_both_veg_nonveg_l170_170651


namespace cyclist_distance_traveled_l170_170998

noncomputable def pedestrian_cyclist_problem (distanceAB : ℝ) (vp : ℝ) (vc : ℝ) : ℝ :=
  let t := distanceAB / vp
  in vc * t

theorem cyclist_distance_traveled :
  ∀ (distanceAB : ℝ) (vp : ℝ),
    distanceAB = 5 →
    vc = 2 * vp →
    pedestrian_cyclist_problem distanceAB vp (2 * vp) = 10 :=
by intros; simp [pedestrian_cyclist_problem]; sorry

end cyclist_distance_traveled_l170_170998


namespace range_of_function_l170_170471

theorem range_of_function :
  (set.range (λ x : ℝ, 1 / (2 - x)^2)) = set.Ioi 0 := by
  sorry

end range_of_function_l170_170471


namespace smallest_AAB_value_l170_170528

theorem smallest_AAB_value {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_distinct : A ≠ B) (h_eq : 10 * A + B = (1 / 9) * (100 * A + 10 * A + B)) :
  100 * A + 10 * A + B = 225 :=
by
  -- Insert proof here
  sorry

end smallest_AAB_value_l170_170528


namespace number_of_even_factors_of_n_l170_170247

noncomputable def n := 2^3 * 3^2 * 7^3

theorem number_of_even_factors_of_n : 
  (∃ (a : ℕ), (1 ≤ a ∧ a ≤ 3)) ∧ 
  (∃ (b : ℕ), (0 ≤ b ∧ b ≤ 2)) ∧ 
  (∃ (c : ℕ), (0 ≤ c ∧ c ≤ 3)) → 
  (even_nat_factors_count : ℕ) = 36 :=
by
  sorry

end number_of_even_factors_of_n_l170_170247


namespace find_other_endpoint_of_diameter_l170_170145

noncomputable def circle_center : (ℝ × ℝ) := (4, -2)
noncomputable def one_endpoint_of_diameter : (ℝ × ℝ) := (7, 5)
noncomputable def other_endpoint_of_diameter : (ℝ × ℝ) := (1, -9)

theorem find_other_endpoint_of_diameter :
  let (cx, cy) := circle_center
  let (x1, y1) := one_endpoint_of_diameter
  let (x2, y2) := other_endpoint_of_diameter
  (x2, y2) = (2 * cx - x1, 2 * cy - y1) :=
by
  sorry

end find_other_endpoint_of_diameter_l170_170145


namespace four_equal_angles_in_convex_heptagon_l170_170209

theorem four_equal_angles_in_convex_heptagon
  (heptagon : List ℝ) (h_len : heptagon.length = 7)
  (h_valid : ∀ x, x ∈ heptagon → 0 < x ∧ x < π)
  (sum_const : ∀ (sin_indices : Finset ℕ), sin_indices.card = 4 →
    let cos_indices := (Finset.range 7).erase sin_indices.univ.to_list
    (Finset.sum sin_indices (λ i, Real.sin (heptagon.nth_le i (by linarith)))
     + Finset.sum cos_indices (λ i, Real.cos (heptagon.nth_le i (by linarith))) = c))
  : ∃ (a : ℝ), ∃ (count : ℕ), count ≥ 4 ∧ count = (List.filter (λ x, x = a) heptagon).length :=
begin
  sorry
end

end four_equal_angles_in_convex_heptagon_l170_170209


namespace num_combinations_30_cents_l170_170243

def coins_combinations (penny nickel dime total_value : ℕ) : Prop :=
  ∃ (p n d : ℕ), p * penny + n * nickel + d * dime = total_value

theorem num_combinations_30_cents : coins_combinations 1 5 10 30 = 20 :=
by
  sorry

end num_combinations_30_cents_l170_170243


namespace factorize_poly_l170_170557
open Real Polynomial

theorem factorize_poly : 
  (X^8 + X^4 + 1 : Polynomial ℝ) = 
    (X^2 - (√3) * X + 1) * 
    (X^2 + (√3) * X + 1) * 
    (X^2 - X + 1) * 
    (X^2 + X + 1) :=
by
  sorry

end factorize_poly_l170_170557


namespace possible_values_of_k_l170_170271

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170271


namespace fraction_meaningful_l170_170444

theorem fraction_meaningful (x : ℝ) : (¬ (x - 2 = 0)) ↔ (x ≠ 2) :=
by
  sorry

end fraction_meaningful_l170_170444


namespace rhombus_area_2sqrt2_l170_170448

structure Rhombus (α : Type _) :=
  (side_length : ℝ)
  (angle : ℝ)

theorem rhombus_area_2sqrt2 (R : Rhombus ℝ) (h_side : R.side_length = 2) (h_angle : R.angle = 45) :
  ∃ A : ℝ, A = 2 * Real.sqrt 2 :=
by
  let A := 2 * Real.sqrt 2
  existsi A
  sorry

end rhombus_area_2sqrt2_l170_170448


namespace trigonometric_shift_l170_170445

theorem trigonometric_shift :
  ∀ (x : ℝ),
    (2 * sin (2 * (x - π / 12))) = (sqrt 3 * sin (2 * x) - cos (2 * x)) →
    2 * sin (2 * x) = 2 * sin (2 * (x - π / 12)) :=
by
  intros x h
  sorry

end trigonometric_shift_l170_170445


namespace value_of_f_of_tan_sq_t_l170_170585

theorem value_of_f_of_tan_sq_t (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 0 ∧ x ≠ 1 → f (x / (x - 1)) = 1 / x)
  (t : ℝ) (ht : 0 ≤ t ∧ t ≤ (π / 2)) :
  f (tan t ^ 2) = tan t ^ 2 :=
by
  sorry

end value_of_f_of_tan_sq_t_l170_170585


namespace relationship_among_a_b_c_l170_170351

noncomputable def a : ℝ := 2016^(1/2017)
noncomputable def b : ℝ := Real.log  (sqrt 2017) / Real.log 2016
noncomputable def c : ℝ := Real.log  (sqrt 2016) / Real.log 2017

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l170_170351


namespace range_of_s_l170_170467

def s (x : ℝ) : ℝ := 1 / (2 - x)^2

theorem range_of_s : set.Ioo 0 ∞ = set.range s :=
sorry

end range_of_s_l170_170467


namespace find_z2_l170_170620

-- Definitions of the given conditions

def z1 : ℂ := 2 - complex.I
def z2 (a : ℝ) : ℂ := a + 2 * complex.I

-- Define the conditions as hypotheses
def cond1 : (z1 - 2) * (1 + complex.I) = 1 - complex.I := by
  unfold z1
  simp
  sorry

def cond2 (a : ℝ) : (z1 * z2 a).im = 0 := by
  unfold z1 z2
  simp
  sorry

-- Main theorem
theorem find_z2 : ∃ a : ℝ, (z2 a = 4 + 2 * complex.I ∧ cond2 a) := by
  exists 4
  split
  { unfold z2
    simp
  }
  { unfold cond2
    sorry
  }

end find_z2_l170_170620


namespace problem1_problem2_l170_170213

noncomputable def A (a : ℝ) : set ℝ := { y | y^2 - (a^2 + a + 1) * y + a * (a^2 + 1) > 0 }
noncomputable def B : set ℝ := { y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - x + 1 }

-- Statement of the first problem
theorem problem1 (a : ℝ) (h : A a ∩ B = ∅) : 1 ≤ a ∧ a ≤ 2 := sorry

-- Statement of the second problem
theorem problem2 (a : ℝ) (h_min : -2 = a) :
  (compl (A a) ∩ B) = { y | 2 ≤ y ∧ y ≤ 4 } := sorry

end problem1_problem2_l170_170213


namespace points_after_perfect_games_l170_170914

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l170_170914


namespace sqrt_50_between_7_and_8_l170_170803

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170803


namespace range_of_a_l170_170222

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (a - x)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3) →
  3 < a ∧ a < 13 / 4 :=
by
  sorry

end range_of_a_l170_170222


namespace find_y_l170_170350

def oslash (a b : ℝ) : ℝ := (sqrt (3 * a + 2 * b))^3

theorem find_y (y : ℝ) (h : oslash 3 y = 125) : y = 8 :=
by
  sorry

end find_y_l170_170350


namespace product_of_consecutive_integers_sqrt_50_l170_170783

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170783


namespace bob_distance_when_met_l170_170731

-- Define the constants for the problem
def distance_XY : ℝ := 17
def yolanda_speed_first_half : ℝ := 3
def yolanda_speed_second_half : ℝ := 4
def bob_speed_first_half : ℝ := 4
def bob_speed_second_half : ℝ := 3
def yolanda_head_start : ℝ := 1  -- hour
def total_distance_half : ℝ := distance_XY / 2

-- Define the problem as a Lean theorem
theorem bob_distance_when_met :
  let time_yolanda_first_half := total_distance_half / yolanda_speed_first_half in
  let time_yolanda_second_half := total_distance_half / yolanda_speed_second_half in
  let total_time_yolanda := yolanda_head_start + time_yolanda_first_half + time_yolanda_second_half in
  let total_time_bob := total_time_yolanda - yolanda_head_start in
  let t := (17 - 3 * (total_time_bob - ((total_distance_half / bob_speed_first_half)))) / (bob_speed_first_half - bob_speed_second_half) in
  let bob_distance := bob_speed_first_half * t in
  bob_distance = 8.5004 :=
begin
  sorry
end

end bob_distance_when_met_l170_170731


namespace tangent_line_to_cubic_curve_l170_170412

variable (x : ℝ)

def cubic_curve := x^3

def point_P := (2, 8 : ℝ × ℝ)

-- Slope of the tangent line at x = 2
def derivative_at_2 := 3 * 2^2

-- Equation of the tangent line in point-slope form
def tangent_line_equation := 
  ∀ x y : ℝ, y - 8 = 12 * (x - 2) → y = 12 * x - 16

theorem tangent_line_to_cubic_curve :
  ∃ (f : ℝ → ℝ),
    (∀ x : ℝ, f x = 12 * x - 16) ∧ 
    (∀ x : ℝ, y = x^3 → (∀ x y : ℝ, y = f x)) := 
by
  sorry

end tangent_line_to_cubic_curve_l170_170412


namespace solve_equation_l170_170589

noncomputable def min (a b : ℚ) : ℚ := if a ≤ b then a else b

theorem solve_equation : (∃ x : ℚ, (min x (-x) = 3 * x + 4) ∧ x = -2) :=
by
  use -2
  sorry

end solve_equation_l170_170589


namespace range_of_omega_l170_170601

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ x : ℝ, x ∈ (π / 2, π) ∧ (sin (ω * x - π / 6) = 0)) ∧
  (∀ x : ℝ, x ∈ (π / 2, π) → ¬ ∃ δ, δ > 0 ∧ (sin (ω * x - π / 6 + δ) = sin (ω * x - π / 6 - δ)))
  ↔ ω ∈ (Set.Ioo (1 / 6) (1 / 3) ∪ Set.Icc (4 / 3) (5 / 3)) := 
by {
  sorry
}

end range_of_omega_l170_170601


namespace spring_length_linear_relation_l170_170659

theorem spring_length_linear_relation :
  ∀ (x : ℕ), y x = 2 * x + 20 :=
by
  intros x
  cases x
  case zero { rw y 0, exact rfl }
  case succ {
    cases x
    case zero { rw y 1, exact rfl }
    case succ {
      cases x
      case zero { rw y 2, exact rfl }
      case succ {
        cases x
        case zero { rw y 3, exact rfl }
        case succ {
          cases x
          case zero { rw y 4, exact rfl }
          case succ {
            cases x
            case zero { rw y 5, exact rfl }
            case succ { sorry }
          }
        }
      }
    }
  }

end spring_length_linear_relation_l170_170659


namespace part1_part2_l170_170215

theorem part1 (f : ℝ → ℝ) (a : ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_neg : ∀ x : ℝ, x ∈ Ico (-2) 0 -> f x = -a * x ^ 2 - log (-x) + 1) : 
  a = 1 / 2 → tangent_eq (f 1) (f' 1) = 4 * 1 - 2 * f 1 - 5 := 
sorry

theorem part2 (f : ℝ → ℝ) (a : ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_neg : ∀ x : ℝ, x ∈ Ico (-2) 0 -> f x = -a * x ^ 2 - log (-x) + 1) 
  (h_abs : ∀ x : ℝ, x ∈ Ioo 0 2 -> abs (f x + x) >= 1) : 
  max_a f = -1 := 
sorry

end part1_part2_l170_170215


namespace radius_of_sphere_is_12_l170_170114

-- Define the radius of the wire and the height of the wire
def r_wire : ℝ := 4
def h : ℝ := 144

-- The goal is to prove that the radius of the sphere is 12 cm, given the conditions
theorem radius_of_sphere_is_12 :
  ∃ r_sphere : ℝ, (4/3 * real.pi * r_sphere^3 = real.pi * r_wire^2 * h) ∧ r_sphere = 12 :=
begin
  use 12,
  sorry
end

end radius_of_sphere_is_12_l170_170114


namespace sqrt_50_product_consecutive_integers_l170_170806

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170806


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170021

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170021


namespace polynomials_exist_for_natural_numbers_l170_170587

open BigOperators

noncomputable def exists_polynomials (n : ℕ) (hn : n > 0) : 
  ∃ (P Q : mv_polynomial (fin n) ℤ), 
    (P ≠ 0) ∧ (Q ≠ 0) ∧ 
    (∑ i, mv_polynomial.X (fin i)) * P = Q ∘ mv_polynomial.X ∘ (λ i, (fin i)) ^ 2 := 
sorry

theorem polynomials_exist_for_natural_numbers :
  ∀ n : ℕ, n > 0 → exists_polynomials n :=
sorry

end polynomials_exist_for_natural_numbers_l170_170587


namespace monster_perimeter_l170_170309

theorem monster_perimeter (r : ℝ) (theta : ℝ) (h₁ : r = 2) (h₂ : theta = 90 * π / 180) :
  2 * r + (3 / 4) * (2 * π * r) = 3 * π + 4 := by
  -- Sorry to skip the proof.
  sorry

end monster_perimeter_l170_170309


namespace smallest_model_length_l170_170386

theorem smallest_model_length :
  ∀ (full_size mid_size smallest : ℕ),
  full_size = 240 →
  mid_size = full_size / 10 →
  smallest = mid_size / 2 →
  smallest = 12 :=
by
  intros full_size mid_size smallest h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end smallest_model_length_l170_170386


namespace max_value_of_n_l170_170376

noncomputable def ellipse_max_n (a c d : ℝ) (n : ℝ) : Prop :=
  4 * a^2 + 3 * c^2 = 12 ∧ -- Equation of the ellipse
  abs (a - c) = 1 ∧           -- |P_1F| = |a - c| = 1
  a + c = 3 ∧                  -- |P_nF| = a + c = 3
  d > 1 / 100 ∧                -- Common difference condition
  1 + (n - 1) * d = 3          -- Arithmetic sequence property

theorem max_value_of_n : ∃ (n : ℕ), ellipse_max_n a c d n ∧ n = 200 :=
begin
  sorry
end

end max_value_of_n_l170_170376


namespace product_of_consecutive_integers_sqrt_50_l170_170790

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170790


namespace optimal_days_without_discount_optimal_days_with_discount_l170_170490

-- Define the conditions for the farm's feed purchase problem
def feed_per_day := 200
def feed_price := 1.8
def storage_cost_per_kg_per_day := 0.03
def transportation_fee := 300
def discount_threshold_tons := 5
def discount_rate := 0.85

-- Define the total cost function without discount 
def total_cost_without_discount (x : ℕ) : ℝ := (300 / x) + 3 * x + 357

-- Define the total cost function with discount
def total_cost_with_discount (x : ℕ) : ℝ := (300 / x) + 3 * x + 303

-- Prove the optimal number of days between purchases without discount is 10
theorem optimal_days_without_discount : 
  ∀ x : ℕ, x > 0 → total_cost_without_discount x ≥ total_cost_without_discount 10 :=
by sorry

-- Prove the optimal number of days between purchases with discount is 25
theorem optimal_days_with_discount : 
  ∀ x : ℕ, x ≥ 25 → total_cost_with_discount x ≥ total_cost_with_discount 25 :=
by sorry

end optimal_days_without_discount_optimal_days_with_discount_l170_170490


namespace quadratic_solution_l170_170416

variable (a c : ℝ)

theorem quadratic_solution (h₀ : a ≠ 0) (h₁ : a * 3^2 - 2 * a * 3 + c = 0) : 
  (∃ x1 x2, (x1 = -1 ∧ x2 = 3) ∧ a * x1^2 - 2 * a * x1 + c = 0 ∧ a * x2^2 - 2 * a * x2 + c = 0) :=
begin
  sorry
end

end quadratic_solution_l170_170416


namespace length_of_first_train_l170_170526

-- Define the conditions
def speed_train1 : ℝ := 120  -- Speed of the first train in kmph
def speed_train2 : ℝ := 80  -- Speed of the second train in kmph
def length_train2 : ℝ := 260.04  -- Length of the second train in meters
def crossing_time : ℝ := 9  -- Time to cross in seconds

-- Define the conversion from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Define the relative speed in m/s
def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train1 + speed_train2)

-- Define the combined length of both trains
def combined_length : ℝ :=
  relative_speed_mps * crossing_time

-- Define the length of the first train
def length_train1 : ℝ :=
  combined_length - length_train2

-- Proof problem: Prove the length of the first train is 240 meters
theorem length_of_first_train :
  length_train1 = 240 := 
  sorry  -- proof to be filled in later

end length_of_first_train_l170_170526


namespace solution_set_xf_pos_l170_170614

-- Let f be an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

-- Let d f := f + x f', if x < 0 => f + x f' < 0
def dec_xf_dx_lt_zero {f : ℝ → ℝ} (h1 : ∀ x < 0, f(x) + x * deriv f x < 0) : Prop := ∀ x < 0, f(x) + x * deriv f x < 0

-- Given condition f(-4)= 0
axiom f_minus4_zero (f : ℝ → ℝ) : f (-4) = 0

-- The goal is to find the solution set of xf(x) > 0
theorem solution_set_xf_pos (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : ∀ x < 0, f(x) + x * deriv f x < 0)
  (h : f(-4) = 0) :
  { x : ℝ | x * f x > 0 } = set.Ioo (-∞) (-4) ∪ set.Ioo 0 4 :=
sorry

end solution_set_xf_pos_l170_170614


namespace method_of_moments_estimate_l170_170764

-- Definitions of conditions from the problem

-- X follows a Poisson distribution with parameter λ
def Poisson_pmf (λ : ℝ) (xi : ℕ) : ℝ := (λ^xi * Real.exp (-λ)) / (Nat.factorial xi)

-- Sample values
variables (n : ℕ) (xi : Fin n → ℕ)
variable (λ : ℝ)

-- Statement of the problem: point estimate of λ using method of moments
theorem method_of_moments_estimate :
  let sample_mean := (Finset.univ.sum (λ i => xi i).toReal) / n
  in λ = sample_mean :=
sorry

end method_of_moments_estimate_l170_170764


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170041

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170041


namespace complete_the_square_l170_170530

theorem complete_the_square (x : ℝ) : (x^2 - 8*x + 15 = 0) → ((x - 4)^2 = 1) :=
by
  intro h
  have eq1 : x^2 - 8*x + 15 = 0 := h
  sorry

end complete_the_square_l170_170530


namespace zara_owns_113_goats_l170_170073

-- Defining the conditions
def cows : Nat := 24
def sheep : Nat := 7
def groups : Nat := 3
def animals_per_group : Nat := 48

-- Stating the problem, with conditions as definitions
theorem zara_owns_113_goats : 
  let total_animals := groups * animals_per_group in
  let cows_and_sheep := cows + sheep in
  let goats := total_animals - cows_and_sheep in
  goats = 113 := by
  sorry

end zara_owns_113_goats_l170_170073


namespace remainder_when_x_squared_div_30_l170_170168

theorem remainder_when_x_squared_div_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  (x^2) % 30 = 21 := 
by 
  sorry

end remainder_when_x_squared_div_30_l170_170168


namespace probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l170_170446

-- Definitions for balls and initial conditions
def totalBalls : ℕ := 10
def redBalls : ℕ := 2
def whiteBalls : ℕ := 3
def yellowBalls : ℕ := 5

-- Drawing without replacement
noncomputable def probability_second_ball_red : ℚ :=
  (2/10) * (1/9) + (8/10) * (2/9)

-- Probabilities for each case
noncomputable def probability_first_prize : ℚ := 
  (redBalls.choose 1 * whiteBalls.choose 1) / (totalBalls.choose 2)

noncomputable def probability_second_prize : ℚ := 
  (redBalls.choose 2) / (totalBalls.choose 2)

noncomputable def probability_third_prize : ℚ := 
  (whiteBalls.choose 2) / (totalBalls.choose 2)

-- Probability of at least one yellow ball (no prize)
noncomputable def probability_no_prize : ℚ := 
  1 - probability_first_prize - probability_second_prize - probability_third_prize

-- Probability distribution and expectation for number of winners X
noncomputable def winning_probability : ℚ := probability_first_prize + probability_second_prize + probability_third_prize

noncomputable def P_X (n : ℕ) : ℚ :=
  if n = 0 then (7/9)^3
  else if n = 1 then 3 * (2/9) * (7/9)^2
  else if n = 2 then 3 * (2/9)^2 * (7/9)
  else if n = 3 then (2/9)^3
  else 0

noncomputable def expectation_X : ℚ := 
  3 * winning_probability

-- Lean statements
theorem probability_of_second_ball_red_is_correct :
  probability_second_ball_red = 1 / 5 := by
  sorry

theorem probabilities_of_winning_prizes :
  probability_first_prize = 2 / 15 ∧
  probability_second_prize = 1 / 45 ∧
  probability_third_prize = 1 / 15 := by
  sorry

theorem distribution_and_expectation_of_X :
  P_X 0 = 343 / 729 ∧
  P_X 1 = 294 / 729 ∧
  P_X 2 = 84 / 729 ∧
  P_X 3 = 8 / 729 ∧
  expectation_X = 2 / 3 := by
  sorry

end probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l170_170446


namespace points_after_perfect_games_l170_170912

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l170_170912


namespace system_linear_eq_sum_l170_170238

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l170_170238


namespace number_of_permutations_of_5_exhibits_l170_170930

theorem number_of_permutations_of_5_exhibits : (5.factorial) = 120 := by
  sorry

end number_of_permutations_of_5_exhibits_l170_170930


namespace triangle_area_l170_170673

variables {α : Type*} [Real α] (a b c : α) (A B C S : α)

-- Define the conditions
def condition_1 := c = 2
def condition_2 := sin A = 2 * sin C
def condition_3 := cos B = 1 / 4

-- The proof statement
theorem triangle_area : condition_1 → condition_2 → condition_3 → S = sqrt 15 :=
by intros h1 h2 h3; sorry

end triangle_area_l170_170673


namespace dale_pasta_l170_170156

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l170_170156


namespace fraction_pizza_peter_ate_l170_170377

theorem fraction_pizza_peter_ate (slices_of_pizza : ℕ) (peter_ate : ℕ) (shared_with_paul : ℕ) (shared_equally : ℕ) : 
  slices_of_pizza = 16 → peter_ate = 3 → shared_with_paul = 2 → shared_equally = 1 → 
  (peter_ate * 1 + shared_with_paul * shared_equally) / slices_of_pizza = 5 / 16 :=
by
  intros slices_of_pizza_eq peter_ate_eq shared_with_paul_eq shared_equally_eq
  rw [slices_of_pizza_eq, peter_ate_eq, shared_with_paul_eq, shared_equally_eq]
  norm_num
  rfl

end fraction_pizza_peter_ate_l170_170377


namespace problem_solution_l170_170063

def correct_choice_approx : Prop :=
  (sqrt 2520 ≈ 50.2) ∧
  (∀ a, a ≠ √13.6 ∨ ¬(a ≈ 4.0)) ∧
  (∀ b, b ≠ ∛800 ∨ ¬(b ≈ 12)) ∧
  (∀ c, c ≠ √8958 ∨ ¬(c ≈ 9.4))

theorem problem_solution : correct_choice_approx :=
by
  -- Correct approximation
  have h1 : sqrt 2520 ≈ 50.2 := sorry,
  -- Incorrect approximations
  have h2 : ∀ a, a ≠ √13.6 ∨ ¬(a ≈ 4.0) := sorry,
  have h3 : ∀ b, b ≠ ∛800 ∨ ¬(b ≈ 12) := sorry,
  have h4 : ∀ c, c ≠ √8958 ∨ ¬(c ≈ 9.4) := sorry,
  -- Combine all the conditions
  exact ⟨h1, h2, h3, h4⟩

end problem_solution_l170_170063


namespace op_evaluation_l170_170713

-- Define the custom operation ⊕
def op (a b c : ℝ) : ℝ := b^2 - 3 * a * c

-- Statement of the theorem we want to prove
theorem op_evaluation : op 2 3 4 = -15 :=
by 
  -- This is a placeholder for the actual proof,
  -- which in a real scenario would involve computing the operation.
  sorry

end op_evaluation_l170_170713


namespace luke_payments_difference_l170_170719

theorem luke_payments_difference :
  let principal := 12000
  let rate := 0.08
  let years := 10
  let n_quarterly := 4
  let n_annually := 1
  let quarterly_rate := rate / n_quarterly
  let annually_rate := rate / n_annually
  let balance_plan1_5years := principal * (1 + quarterly_rate)^(n_quarterly * 5)
  let payment_plan1_5years := balance_plan1_5years / 3
  let remaining_balance_plan1_5years := balance_plan1_5years - payment_plan1_5years
  let final_balance_plan1_10years := remaining_balance_plan1_5years * (1 + quarterly_rate)^(n_quarterly * 5)
  let total_payment_plan1 := payment_plan1_5years + final_balance_plan1_10years
  let final_balance_plan2_10years := principal * (1 + annually_rate)^years
  (total_payment_plan1 - final_balance_plan2_10years).abs = 1022 :=
by
  sorry

end luke_payments_difference_l170_170719


namespace max_elements_l170_170121

open Set

def is_valid_set (T : Set ℕ) : Prop :=
  (∀ y ∈ T, ((T.sum id - y) / (T.size - 1)) ∈ ℤ) ∧
  2 ∈ T ∧ T.max' _ = 4018 ∧ (∀ x ∈ T, x > 0)

theorem max_elements (T : Set ℕ) (hT : is_valid_set T) : T.size ≤ 33 :=
sorry

end max_elements_l170_170121


namespace book_loss_percentage_l170_170410

theorem book_loss_percentage 
  (C S : ℝ) 
  (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := 
by 
  sorry

end book_loss_percentage_l170_170410


namespace tangent_at_one_minimum_value_extreme_points_l170_170224

-- Problem (1)
def problem_1 (x : ℝ) : ℝ := x * Real.log x - x^2
theorem tangent_at_one (x : ℝ) (h : x = 1) : problem_1 x + 1 = 0 := sorry

-- Problem (2)
def problem_2 (x : ℝ) : ℝ := x * Real.log x
theorem minimum_value (t : ℝ) (h : t > 0) : 
  (t > 1 / Real.e → ∀ x ∈ Set.Icc t (t + 2), problem_2 x ≥ t * Real.log t ∧ problem_2 t = t * Real.log t) ∧
  (t ≤ 1 / Real.e → ∀ x ∈ Set.Icc t (t + 2), problem_2 x ≥ -1 / Real.e ∧ problem_2 (1 / Real.e) = -1 / Real.e) := sorry

-- Problem (3)
def problem_3 (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x
theorem extreme_points (a x1 x2 : ℝ) (hx : problem_3 x1 a = 0 ∧ problem_3 x2 a = 0) (h_a_pos : a > 0) :
  (1 / Real.log x1) + (1 / Real.log x2) > 2 * a * Real.e := sorry

end tangent_at_one_minimum_value_extreme_points_l170_170224


namespace range_of_a_l170_170622

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → log a (a * x^2 - x + 1/2) > 0) ↔ ((1 / 2 < a ∧ a < 5 / 8) ∨ (a > 3 / 2)) :=
by
  sorry

end range_of_a_l170_170622


namespace remainder_1234567_div_145_l170_170140

theorem remainder_1234567_div_145 : 1234567 % 145 = 67 := by
  sorry

end remainder_1234567_div_145_l170_170140


namespace numbers_have_same_digits_reverse_l170_170390

theorem numbers_have_same_digits_reverse (n : ℕ) :
  let B := n^2 + 1 in
  let num1 := n^2 * (n^2 + 2)^2 in
  let num2 := n^4 * (n^2 + 2)^2 in
  digits_reverse (to_base B num1) (to_base B num2) :=
sorry

end numbers_have_same_digits_reverse_l170_170390


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170018

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170018


namespace sum_dihedral_angles_at_each_vertex_is_180_l170_170556

-- Define a tetrahedron
structure Tetrahedron :=
(vertices : Fin 4 → ℝ × ℝ × ℝ) -- A mapping from the 4 vertices of a tetrahedron to 3D coordinates

-- Define how to obtain the dihedral angles
def dihedralAngle (T : Tetrahedron) (v : Fin 4) : ℝ :=
-- This function should compute the sum of the dihedral angles at vertex 'v'
sorry

-- The main theorem to be proved
theorem sum_dihedral_angles_at_each_vertex_is_180 (T : Tetrahedron) (v : Fin 4) :
  dihedralAngle(T, v) = 180 :=
sorry

end sum_dihedral_angles_at_each_vertex_is_180_l170_170556


namespace scientific_notation_of_area_l170_170400

theorem scientific_notation_of_area :
  (0.0000064 : ℝ) = 6.4 * 10 ^ (-6) := 
sorry

end scientific_notation_of_area_l170_170400


namespace find_triangle_sides_l170_170427

noncomputable def triangle_sides (AB : ℝ) (AH : ℝ) (AE : ℝ) : ℝ × ℝ :=
  sorry

theorem find_triangle_sides
  (AB AE AH : ℝ)
  (h_AB : AB = 5)
  (h_AE : AE = 5.5)
  (h_AH : AH = 2 * real.sqrt 6) :
  triangle_sides AB AH AE = (4 + 8/21, 5 + 20/21) :=
sorry

end find_triangle_sides_l170_170427


namespace inscribed_quadrilateral_non_inscribed_quadrilateral_l170_170582

noncomputable def smallest_perimeter_quadrilateral (A B C D : Pt) (inscribed : Bool) : Prop :=
  if inscribed then
    ∃ (X1 X2 X3 X4 : Pt), X1 = A ∧ X2 = B ∧ X3 = C ∧ X4 = D ∧ 
      ∃ f : LinearMap ℝ (Pt × Pt) Pt, f.inscribes (X1, X2, X3, X4) ∧ ∀ Q, (quadrilateral.perimeter Q > quadrilateral.perimeter ⟨X1, X2, X3, X4⟩)
  else
    ∃! X Y Z W : Pt, False

theorem inscribed_quadrilateral (A B C D : Pt) : 
  (∃ a b c d, (angle a b d = 180) ∧ (angle c) = 180) → 
  smallest_perimeter_quadrilateral A B C D True :=
begin
  sorry
end

theorem non_inscribed_quadrilateral (A B C D : Pt) : 
  (∀ a b c d, (angle a b d ≠ 180) ∨ (angle c ≠ 180)) → 
  smallest_perimeter_quadrilateral A B C D False :=
begin
  sorry
end

end inscribed_quadrilateral_non_inscribed_quadrilateral_l170_170582


namespace sqrt_50_between_7_and_8_l170_170796

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170796


namespace teacups_more_than_hotchocolate_l170_170728

def hot_chocolate_cups_on_rainy_days (n : ℕ) := 2 * n
def tea_cups_on_non_rainy_days := 5 * 4
def total_cups (n : ℕ) := hot_chocolate_cups_on_rainy_days n + tea_cups_on_non_rainy_days
def more_tea_than_hot_chocolate_cups (n : ℕ) := tea_cups_on_non_rainy_days - hot_chocolate_cups_on_rainy_days n

theorem teacups_more_than_hotchocolate (n : ℕ) (h : total_cups n = 26) :
  more_tea_than_hot_chocolate_cups n = 14 :=
begin
  sorry
end

end teacups_more_than_hotchocolate_l170_170728


namespace product_of_roots_l170_170147

theorem product_of_roots :
  let a := 4
  let k := 6
  let polynomial := {p : Polynomial ℝ // p.degree = 4 ∧ p.coeff 4 = 4 ∧ p.coeff 0 = 6} in
  ( ∃ p : polynomial, p.roots.prod = (3 / 2) ) :=
by
  sorry

end product_of_roots_l170_170147


namespace possible_arrangement_of_vectors_l170_170324

open Real

variable {A B C : EuclideanSpace ℝ (Fin 2)}

theorem possible_arrangement_of_vectors 
    (h1 : A + B + C = 0) 
    (h2 : ∥A + B∥ = 1) 
    (h3 : ∥B + C∥ = 1) 
    (h4 : ∥C + A∥ = 1) : 
    ∃ A B C : EuclideanSpace ℝ (Fin 2), 
    A + B + C = 0 ∧ 
    ∥A + B∥ = 1 ∧ 
    ∥B + C∥ = 1 ∧ 
    ∥C + A∥ = 1 :=
by
  -- Skipping the proof as specified by 'sorry'
  sorry

end possible_arrangement_of_vectors_l170_170324


namespace rose_paid_after_discount_l170_170738

-- Define the conditions as given in the problem statement
def original_price : ℕ := 10
def discount_rate : ℕ := 10

-- Define the theorem that needs to be proved
theorem rose_paid_after_discount : 
  original_price - (original_price * discount_rate / 100) = 9 :=
by
  -- Here we skip the proof with sorry
  sorry

end rose_paid_after_discount_l170_170738


namespace max_expression_value_l170_170316

theorem max_expression_value : ∃ a b c d : ℕ, 
  a ∈ {1, 2, 4, 5} ∧ b ∈ {1, 2, 4, 5} ∧ c ∈ {1, 2, 4, 5} ∧ d ∈ {1, 2, 4, 5} ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (c * a ^ b - d = 79) :=
begin
  sorry
end

end max_expression_value_l170_170316


namespace sqrt_50_between_consecutive_integers_product_l170_170839

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170839


namespace proof_problem_solution_l170_170546

noncomputable def proof_problem : ℤ := 
  let n : ℤ := 2006
  in Int.floor ( ( ((n+1)^3 : ℤ) / ((n-1) * n) - ((n+2)^3 : ℤ) / (n * (n+1)) ) )

theorem proof_problem_solution : proof_problem = -4 :=
by {
  sorry
}

end proof_problem_solution_l170_170546


namespace quadrilateral_divided_into_7_equal_triangles_l170_170074

variable {α : Type*} [Nonempty α]

theorem quadrilateral_divided_into_7_equal_triangles :
  (∀ (T : α), ∃ (triangles : list α), length triangles = 4 ∧ AllHaveEqualArea triangles) →
  ∃ (Q : α), ∃ (triangles : list α), length triangles = 7 ∧ AllHaveEqualArea triangles :=
by sorry

end quadrilateral_divided_into_7_equal_triangles_l170_170074


namespace derivative_at_2_l170_170626

-- Define the function f(x) = x * exp(x)
def f (x : ℝ) : ℝ := x * Real.exp x

-- Statement of the proof problem
theorem derivative_at_2 : (deriv f 2) = 3 * Real.exp 2 := by
  sorry

end derivative_at_2_l170_170626


namespace integral_x2_sqrt_9_minus_x2_l170_170484

open Real

theorem integral_x2_sqrt_9_minus_x2 :
  ∫ x in -3..3, x^2 * sqrt (9 - x^2) = (81 * π) / 8 :=
by
  sorry

end integral_x2_sqrt_9_minus_x2_l170_170484


namespace range_of_function_l170_170473

theorem range_of_function :
  (set.range (λ x : ℝ, 1 / (2 - x)^2)) = set.Ioi 0 := by
  sorry

end range_of_function_l170_170473


namespace overall_salary_change_is_3_60_percent_l170_170380

-- Define initial salary
def initial_salary : ℝ := 100  -- Arbitrary value, final proof is independent of the initial salary as it's a multiplier

-- Define the sequence of percentage changes in decimal form
def salary_changes : List ℝ := [
  1.20,  -- First increase by 20%
  0.65,  -- Then decrease by 35%
  1.40,  -- Then increase by 40%
  1.10,  -- Further increase by 10%
  0.75,  -- Then decrease by 25%
  1.15   -- Finally increase by 15%
]

-- Compute the final multiplication factor
def final_salary (initial : ℝ) (changes : List ℝ) : ℝ :=
  List.foldl (*) initial changes

-- Define the overall percentage change
def overall_percentage_change (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

-- The final claim to prove
theorem overall_salary_change_is_3_60_percent :
  overall_percentage_change initial_salary (final_salary initial_salary salary_changes) = 3.6035 :=
by
  sorry  -- Proof is omitted as per the instructions

end overall_salary_change_is_3_60_percent_l170_170380


namespace smallest_positive_period_of_f_l170_170980

-- Define the function and the condition
def f (x: Real) : Real := Real.sin x * Real.cos x

-- State the problem in Lean 4
theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', (T' > 0 ∧ (∀ x, f (x + T') = f x)) → T' ≥ T) := sorry

end smallest_positive_period_of_f_l170_170980


namespace sum_of_solutions_of_given_equation_l170_170960

theorem sum_of_solutions_of_given_equation :
  let f := λ x : ℝ, (x^3 - 3 * x^2 - 9 * x) / (x + 3)
  (x1 x2 : ℝ)
   (hx1 : 7 = f x1)
   (hx2 : 7 = f x2) in
  x1 + x2 = 6 :=
by 
  sorry

end sum_of_solutions_of_given_equation_l170_170960


namespace joe_total_travel_time_is_16_5_l170_170688

def joe_travel_time (d : ℝ) (t_walk : ℝ) (t_wait : ℝ) (ratio_run_walk : ℝ) (total_time_walk : ℝ) : ℝ :=
  let one_third_time_walk := total_time_walk / 2
  let speed_walk := d / 18
  let speed_run := ratio_run_walk * speed_walk
  let one_third_time_run := d / (3 * speed_run)
  total_time_walk + t_wait + one_third_time_run

theorem joe_total_travel_time_is_16_5 :
  joe_travel_time d 6 3 4 12 = 16.5 :=
by
  sorry

end joe_total_travel_time_is_16_5_l170_170688


namespace product_of_consecutive_integers_sqrt_50_l170_170841

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170841


namespace quadrilateral_divided_into_7_equal_triangles_l170_170075

variable {α : Type*} [Nonempty α]

theorem quadrilateral_divided_into_7_equal_triangles :
  (∀ (T : α), ∃ (triangles : list α), length triangles = 4 ∧ AllHaveEqualArea triangles) →
  ∃ (Q : α), ∃ (triangles : list α), length triangles = 7 ∧ AllHaveEqualArea triangles :=
by sorry

end quadrilateral_divided_into_7_equal_triangles_l170_170075


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170012

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170012


namespace B_work_days_l170_170092

-- Define the necessary conditions
def A_days := 15
def work_together_days := 2
def fraction_left := 0.7666666666666666
def fraction_completed := 1 - fraction_left

-- Statement of the problem to prove
theorem B_work_days (x : ℝ) : 
  (x > 0) ∧ (2 * (1 / A_days + 1 / x) = fraction_completed) → 
  x = 20 :=
by
  sorry

end B_work_days_l170_170092


namespace anie_days_to_complete_l170_170440

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l170_170440


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170015

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170015


namespace eccentricity_of_hyperbola_l170_170608

variables {a b c e : ℝ}
variables (F1 F2 M : ℝ × ℝ)

def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def left_focus (F1 : ℝ × ℝ) : Prop := F1 = (-c, 0)
def right_focus (F2 : ℝ × ℝ) : Prop := F2 = (c, 0)
def point_on_hyperbola (M : ℝ × ℝ) : Prop := M.1 > 0 ∧ hyperbola M.1 M.2
def condition_one : Prop := dist M F2 = dist F1 F2
def condition_two (angle_MF1F2 : ℝ) : Prop := e * Real.sin angle_MF1F2 = 1

theorem eccentricity_of_hyperbola
  (a_pos : 0 < a) (b_pos : 0 < b) (e_pos : 0 < e)
  (hF1 : left_focus F1) (hF2 : right_focus F2)
  (hM : point_on_hyperbola M)
  (h1 : condition_one)
  (angle_MF1F2 : ℝ)
  (h2 : condition_two angle_MF1F2) :
  e = 5 / 3 := 
sorry

end eccentricity_of_hyperbola_l170_170608


namespace product_of_consecutive_integers_sqrt_50_l170_170817

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170817


namespace construct_trapezoid_l170_170153

-- Define the lengths of the sides and height
variables (c d b : ℝ)
-- Define the diagonal lengths and conditions for constructing the trapezoid
variables (d_1 d_2 a : ℝ)

-- Define the conditions to ensure a valid trapezoid can be constructed
def valid_trapezoid := d_1 - d_2 < a + b ∧ a + b < d_1 + d_2

-- Statement that the trapezoid can be constructed given the conditions
theorem construct_trapezoid (hc : c > 0) (hd : d > 0) (hb : b > 0) (h_diag : valid_trapezoid d_1 d_2 a b) :
  ∃ (AB CD AD BC AC BD : ℝ), 
    (AB = c) ∧
    (CD = d) ∧ 
    (AD = b) ∧ 
    (AB ∥ CD) ∧
    valid_trapezoid d_1 d_2 a b :=
begin
  sorry
end

end construct_trapezoid_l170_170153


namespace sin_sub_pi_div_4_eq_l170_170201

theorem sin_sub_pi_div_4_eq:
  ∀ α : ℝ, sin α = -3 / 5 → (α > 3 * π / 2 ∧ α < 2 * π) → 
  sin (π / 4 - α) = 7 * real.sqrt 2 / 10 := 
begin
  intros α h₁ h₂,
  sorry,
end

end sin_sub_pi_div_4_eq_l170_170201


namespace both_locks_stall_time_l170_170336

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l170_170336


namespace print_time_l170_170510

theorem print_time (rate : ℕ) (pages : ℕ) (h_rate : rate = 24) (h_pages : pages = 360) :
  (pages / rate : ℕ) = 15 :=
by
  rw [h_rate, h_pages]
  simp
  sorry

end print_time_l170_170510


namespace smallest_of_four_numbers_l170_170669

theorem smallest_of_four_numbers :
  ∃ (n : ℤ), n ∈ ({0, -2, -1, 3} : set ℤ) ∧ ∀ m ∈ ({0, -2, -1, 3} : set ℤ), n ≤ m := 
begin
  use -2,
  split,
  { simp, },
  { intros m hm,
    fin_cases hm;
    linarith, }
end

end smallest_of_four_numbers_l170_170669


namespace solve_inequality_l170_170742

theorem solve_inequality {x : ℝ} : (x^2 - 9 * x + 18 ≤ 0) ↔ 3 ≤ x ∧ x ≤ 6 :=
by
sorry

end solve_inequality_l170_170742


namespace days_in_february_2013_l170_170852

def is_common_year (year : ℕ) : Prop :=
  year % 4 ≠ 0

theorem days_in_february_2013 : is_common_year 2013 → days_in_february 2013 = 28 :=
by
  intros h
  have common_year := h
  sorry

end days_in_february_2013_l170_170852


namespace numerator_trailing_zeros_l170_170428

theorem numerator_trailing_zeros :
  let N := ∑ k in Finset.range 1 46, Nat.factorial 45 / k
  in  trailing_zeros_of_quotient : Nat.trailing_zeros N = 8
:= sorry

end numerator_trailing_zeros_l170_170428


namespace secant_divides_area_perimeter_secant_divides_area_perimeter_inverse_l170_170733

-- Define a triangle with its incenter
structure Triangle where
  A B C : Point
  I : Incenter A B C

-- Definitions of points and lines involved
variables (T : Triangle) (P Q : Point)
variable (secant_through_incenter : Line)

-- Conditions: secant passing through the incenter
def secant_through_incenter (T : Triangle) (P Q : Point) : Prop :=
  lies_on P to T.AB ∧ lies_on Q to T.AC ∧ passes_through secant PQ T.I

-- Proving the theorem
theorem secant_divides_area_perimeter (T : Triangle) (P Q : Point) 
    (secant_through_incenter T P Q) :
    (divides_area_equally T P Q T.I ∧ divides_perimeter_equally T P Q T.I) :=
  sorry

-- Inverse theorem
theorem secant_divides_area_perimeter_inverse (T : Triangle) (P Q : Point) 
    (secant_divides T P Q : divides_area_perimeter_equally) :
    (is_incenter T P Q) :=
  sorry

end secant_divides_area_perimeter_secant_divides_area_perimeter_inverse_l170_170733


namespace anie_days_to_complete_l170_170441

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l170_170441


namespace glasses_total_l170_170877

theorem glasses_total :
  ∃ (S L e : ℕ), 
    (L = S + 16) ∧ 
    (12 * S + 16 * L) / (S + L) = 15 ∧ 
    (e = 12 * S + 16 * L) ∧ 
    e = 480 :=
by
  sorry

end glasses_total_l170_170877


namespace fewest_toothpicks_proof_l170_170995

noncomputable def fewest_toothpicks_to_remove (total_toothpicks : ℕ) (additional_row_and_column : ℕ) (triangles : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) (max_destroyed_per_toothpick : ℕ) (horizontal_toothpicks : ℕ) : ℕ :=
  horizontal_toothpicks

theorem fewest_toothpicks_proof 
  (total_toothpicks : ℕ := 40) 
  (additional_row_and_column : ℕ := 1) 
  (triangles : ℕ := 35) 
  (upward_triangles : ℕ := 15) 
  (downward_triangles : ℕ := 10)
  (max_destroyed_per_toothpick : ℕ := 1)
  (horizontal_toothpicks : ℕ := 15) :
  fewest_toothpicks_to_remove total_toothpicks additional_row_and_column triangles upward_triangles downward_triangles max_destroyed_per_toothpick horizontal_toothpicks = 15 := 
by 
  sorry

end fewest_toothpicks_proof_l170_170995


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170010

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170010


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170280

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170280


namespace p_2015_coordinates_l170_170132

namespace AaronWalk

def position (n : ℕ) : ℤ × ℤ :=
sorry

theorem p_2015_coordinates : position 2015 = (22, 57) := 
sorry

end AaronWalk

end p_2015_coordinates_l170_170132


namespace marked_price_l170_170661

theorem marked_price (x : ℝ) (payment : ℝ) (discount : ℝ) (hx : (payment = 90) ∧ ((x ≤ 100 ∧ discount = 0.1) ∨ (x > 100 ∧ discount = 0.2))) :
  (x = 100 ∨ x = 112.5) := by
  sorry

end marked_price_l170_170661


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170024

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170024


namespace smallest_n_for_f_equality_l170_170704

def f(n : ℕ) : ℕ := 
  (finset.range (n + 1)).sum (λ a, (finset.range (n + 1)).count (λ b, a^2 + b^2 = n) / (if a = n / a then 1 else 2))

theorem smallest_n_for_f_equality :
  (∃ n : ℕ, f(n) = 4 ∧ (∀ m : ℕ, m < n → f(m) ≠ 4)) :=
by
  use 26
  split
  · sorry -- Proof that f(26) = 4
  · intro m h
    sorry -- Proof that for all m < 26, f(m) ≠ 4

end smallest_n_for_f_equality_l170_170704


namespace solve_for_x_l170_170374

theorem solve_for_x (x : ℚ) : x^2 + 125 = (x - 15)^2 → x = 10 / 3 := by
  sorry

end solve_for_x_l170_170374


namespace contradiction_example_l170_170453

theorem contradiction_example : 
  ∀ (a b : ℕ), (a + b ≥ 3) → (a ≥ 2 ∨ b ≥ 2) :=
by
  intros a b h
  by_contradiction h1
  cases h1 with ha hb
  have ha' : a < 2 := ha
  have hb' : b < 2 := hb
  sorry

end contradiction_example_l170_170453


namespace basketball_team_combinations_l170_170732

def total_members : ℕ := 15

def leadership_pool : ℕ := 6

def remaining_members (captain : ℕ) (vice_captain : ℕ) : ℕ :=
  total_members - 2

def lineup_combinations (remaining : ℕ) : ℕ :=
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem basketball_team_combinations :
  let captain_choices := leadership_pool,
      vice_captain_choices := leadership_pool - 1,
      remaining := remaining_members captain_choices vice_captain_choices
  in captain_choices * vice_captain_choices * lineup_combinations remaining = 3_326_400 :=
  sorry

end basketball_team_combinations_l170_170732


namespace prob_odd_sum_l170_170173

-- Define the sets of numbers for each wheel
def wheel1 : Set ℕ := {1, 2, 3, 4, 5, 6}
def wheel2 : Multiset ℕ := {3, 3, 4, 4, 5, 5}

-- Define the property of a wheel element being odd or even
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probability of picking an odd number from a given set
def prob_odd (wheel : Set ℕ) : ℚ :=
  (wheel.filter is_odd).card.to_rat / wheel.card.to_rat

def prob_odd_multiset (wheel : Multiset ℕ) : ℚ :=
  (wheel.filter is_odd).card.to_rat / wheel.card.to_rat

-- Define the probability of picking an even number from a given set
def prob_even (wheel : Set ℕ) : ℚ :=
  (wheel.filter is_even).card.to_rat / wheel.card.to_rat

def prob_even_multiset (wheel : Multiset ℕ) : ℚ :=
  (wheel.filter is_even).card.to_rat / wheel.card.to_rat

-- Define the statement to prove the probability of an odd sum
theorem prob_odd_sum :
  (prob_odd wheel1 * prob_even_multiset wheel2 + prob_even wheel1 * prob_odd_multiset wheel2 = 1 / 2) :=
by sorry

end prob_odd_sum_l170_170173


namespace k_positive_first_third_quadrants_l170_170288

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170288


namespace sqrt_50_between_7_and_8_l170_170797

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170797


namespace line_in_first_and_third_quadrants_l170_170264

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170264


namespace tan_sum_identity_l170_170217

theorem tan_sum_identity (α : ℝ) 
  (hα1 : α ∈ Ioo (π / 2) π)
  (hα2 : sin α = 3 / 5) : 
  tan (α + π / 4) = 1 / 7 := 
sorry

end tan_sum_identity_l170_170217


namespace find_angle_C_l170_170648

theorem find_angle_C (a b c A B C : ℝ) (h₀ : 0 < C) (h₁ : C < Real.pi)
  (h₂ : 2 * c * Real.sin A = a * Real.tan C) :
  C = Real.pi / 3 :=
sorry

end find_angle_C_l170_170648


namespace product_of_consecutive_integers_sqrt_50_l170_170849

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170849


namespace print_time_l170_170509

theorem print_time (rate : ℕ) (pages : ℕ) (h_rate : rate = 24) (h_pages : pages = 360) :
  (pages / rate : ℕ) = 15 :=
by
  rw [h_rate, h_pages]
  simp
  sorry

end print_time_l170_170509


namespace valid_n_values_l170_170568

theorem valid_n_values :
  ∀ (n : ℕ), (n ≥ 2) →
  (∃ (a : ℕ → ℝ) (r : ℝ), r > 0 ∧ strict_mono a ∧
   (∃ (a_set : finset ℝ), 
    a_set = (finset.range (nat.choose n 2)).image (λ ⟨i, j⟩, a (j + 1) - a (i + 1)) ∧ 
    a_set = (finset.range (nat.choose n 2)).image (λ k, r ^ (k + 1)))) →
  n = 2 ∨ n = 3 ∨ n = 4 :=
by
  sorry

end valid_n_values_l170_170568


namespace increasing_function_inv_condition_l170_170873

-- Given a strictly increasing real-valued function f on ℝ with an inverse,
-- satisfying the condition f(x) + f⁻¹(x) = 2x for all x in ℝ,
-- prove that f(x) = x + b, where b is a real constant.

theorem increasing_function_inv_condition (f : ℝ → ℝ) (hf_strict_mono : StrictMono f)
  (hf_inv : ∀ x, f (f⁻¹ x) = x ∧ f⁻¹ (f x) = x)
  (hf_condition : ∀ x, f x + f⁻¹ x = 2 * x) :
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end increasing_function_inv_condition_l170_170873


namespace gasoline_price_increase_percentage_l170_170423

theorem gasoline_price_increase_percentage : 
  ∀ (highest_price lowest_price : ℝ), highest_price = 24 → lowest_price = 18 → 
  ((highest_price - lowest_price) / lowest_price) * 100 = 33.33 :=
by
  intros highest_price lowest_price h_highest h_lowest
  rw [h_highest, h_lowest]
  -- To be completed in the proof
  sorry

end gasoline_price_increase_percentage_l170_170423


namespace polynomial_expansion_sum_l170_170342

theorem polynomial_expansion_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} : ℝ)
  (h : (x^2 + 1) * (2 * x + 1)^9 = a_0 + a_1 * (x + 2) + a_2 * (x + 2)^2 + a_3 * (x + 2)^3 + 
       a_4 * (x + 2)^4 + a_5 * (x + 2)^5 + a_6 * (x + 2)^6 + a_7 * (x + 2)^7 + a_8 * (x + 2)^8 + 
       a_9 * (x + 2)^9 + a_{10} * (x + 2)^{10} + a_{11} * (x + 2)^{11}) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} = -2 := 
by
  sorry

end polynomial_expansion_sum_l170_170342


namespace find_k_l170_170498

-- Define the points and the condition
def point1 := (3 : ℤ, 5 : ℤ)
def point2 (k : ℚ) := (-3 : ℤ, k)
def point3 := (-9 : ℤ, -2 : ℤ)

-- Define a condition to check if three points are collinear based on the slope
def are_collinear (p1 p2 p3 : ℚ × ℚ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

-- The theorem to prove
theorem find_k : ∃ k : ℚ, are_collinear point1 (point2 k) point3 ∧ k = 3 / 2 :=
by
  sorry

end find_k_l170_170498


namespace karen_locks_problem_l170_170339

theorem karen_locks_problem :
  let T1 := 5 in
  let T2 := 3 * T1 - 3 in
  let Combined_Locks_Time := 5 * T2 in
  Combined_Locks_Time = 60 := by
    let T1 := 5
    let T2 := 3 * T1 - 3
    let Combined_Locks_Time := 5 * T2
    sorry

end karen_locks_problem_l170_170339


namespace loss_percentage_on_first_book_is_15_l170_170637

theorem loss_percentage_on_first_book_is_15
  (total_cost : ℝ)
  (cost_first_book : ℝ)
  (gain_percentage_second_book : ℝ)
  (selling_price : ℝ)
  (loss_percentage : ℝ)
  (cost_second_book := total_cost - cost_first_book)
  (selling_price_second_book := cost_second_book + 0.19 * cost_second_book)
  (loss_amount := cost_first_book - selling_price) :
  total_cost = 500 →
  cost_first_book = 291.67 →
  gain_percentage_second_book = 19 →
  selling_price = selling_price_second_book →
  selling_price = 247.91 →
  loss_percentage = (loss_amount / cost_first_book) * 100 →
  loss_percentage = 15 :=
begin
  intros,
  sorry
end

end loss_percentage_on_first_book_is_15_l170_170637


namespace four_equal_angles_of_heptagon_l170_170211

theorem four_equal_angles_of_heptagon (α β γ δ ε ζ η : ℝ)
  (h_concave : convex_heptagon α β γ δ ε ζ η)
  (h_sum_invariant : ∀ (a b c d : ℝ),
    (sine α + sine β + sine γ + sine δ + cosine ε + cosine ζ + cosine η) =
    (sine a + sine b + sine c + sine d + cosine e + cosine f + cosine g)) :
  (∃ (α' β' γ' δ' : ℝ), α = α' ∧ β = β' ∧ γ = γ' ∧ δ = δ') :=
sorry

end four_equal_angles_of_heptagon_l170_170211


namespace oil_depth_solution_l170_170104

theorem oil_depth_solution
  (length diameter surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 12)
  (h_diameter : diameter = 4)
  (h_surface_area : surface_area = 24)
  (r : ℝ := diameter / 2)
  (c : ℝ := surface_area / length) :
  (h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3) :=
by
  sorry

end oil_depth_solution_l170_170104


namespace smallest_model_length_l170_170387

theorem smallest_model_length :
  ∀ (full_size mid_size smallest : ℕ),
  full_size = 240 →
  mid_size = full_size / 10 →
  smallest = mid_size / 2 →
  smallest = 12 :=
by
  intros full_size mid_size smallest h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end smallest_model_length_l170_170387


namespace no_prime_p_such_that_22p2_plus_23_is_prime_l170_170590

theorem no_prime_p_such_that_22p2_plus_23_is_prime :
  ∀ p : ℕ, Prime p → ¬ Prime (22 * p ^ 2 + 23) :=
by
  sorry

end no_prime_p_such_that_22p2_plus_23_is_prime_l170_170590


namespace isosceles_triangle_l170_170588

-- Given: sides a, b, c of a triangle satisfying a specific condition
-- To Prove: the triangle is isosceles (has at least two equal sides)

theorem isosceles_triangle (a b c : ℝ)
  (h : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l170_170588


namespace no_two_identical_snakes_swallow_each_other_l170_170128

variable {Snake : Type} (identical : ∀ (s1 s2 : Snake), s1 = s2)

theorem no_two_identical_snakes_swallow_each_other (s1 s2 : Snake) 
    (identical_snakes : identical s1 s2) : false :=
by
  sorry

end no_two_identical_snakes_swallow_each_other_l170_170128


namespace dale_pasta_l170_170157

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l170_170157


namespace checkerboard_corners_sum_l170_170895

theorem checkerboard_corners_sum :
  let top_left := 1,
      top_right := 9,
      bottom_left := 73,
      bottom_right := 81
  in top_left + top_right + bottom_left + bottom_right = 164 :=
by
  sorry

end checkerboard_corners_sum_l170_170895


namespace different_ways_no_encounter_l170_170137

/-- Define the rooms as a simple example, using integers to represent them. -/
inductive Room
| R1 | R2 | R3 | R4 | R5 | R6

/-- Define the conditions given in the problem -/
def isDifferentRooms (p r1 r2 : Room) : Prop :=
  r1 ≠ r2

/-- Define the movement of policeman and thief -/
def movePolice (curr : Room) : Room := sorry -- The actual move logic would be implemented here (moves through solid lines)
def moveThief (curr : Room) : Room := sorry -- The actual move logic would be implemented here (moves through dashed lines)

/-- Define no encounter condition after 3 moves -/
def noEncounter (startPolice startThief : Room) : Prop :=
  let p1 := movePolice startPolice;
  let p2 := movePolice p1;
  let p3 := movePolice p2;
  let t1 := moveThief startThief;
  let t2 := moveThief t1;
  let t3 := moveThief t2 in
  p3 ≠ t3 ∧ p2 ≠ t2 ∧ p1 ≠ t1

/-- The theorem stating the problem's claim -/
theorem different_ways_no_encounter : ∃ n : Nat, n = 1476 ∧ -- Number of ways n is 1476
  ∀ (startPolice startThief : Room), isDifferentRooms startPolice startThief → noEncounter startPolice startThief := sorry

end different_ways_no_encounter_l170_170137


namespace at_most_seventy_percent_acute_triangles_l170_170599

/--
Given 100 coplanar points with no three points collinear,
prove that at most 70% of the triangles formed by these points have all angles acute.
-/
theorem at_most_seventy_percent_acute_triangles 
  (points : Finₓ 100 → Point ℝ 2) 
  (h_non_collinear : ∀ (a b c : Finₓ 100), a ≠ b → b ≠ c → c ≠ a → ¬Collinear ℝ ({a, b, c} : Set (Finₓ 100))) :
  let triangles := { t : Finset (Finₓ 100) // t.card = 3 }
  let acute_triangles := { t ∈ triangles | all_angles_acute t }
  (acute_triangles.card : ℕ) ≤ 70 * (triangles.card : ℕ) / 100 := 
sorry

end at_most_seventy_percent_acute_triangles_l170_170599


namespace find_f_at_7_l170_170616

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_values (h : 0 < x ∧ x < 2) : f x = 2 * x ^ 2

-- The proof problem
theorem find_f_at_7 : f 7 = -2 :=
by sorry

end find_f_at_7_l170_170616


namespace product_of_consecutive_integers_sqrt_50_l170_170780

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170780


namespace prob_palindrome_div_11_l170_170500

theorem prob_palindrome_div_11 :
  (1000 ≤ n ∧ n < 10000 ∧ (∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10)) →
  (n % 11 = 0) :=
by
  assume h : (1000 ≤ n ∧ n < 10000 ∧ (∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10)),
  -- The proof would go here
  sorry

end prob_palindrome_div_11_l170_170500


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170003

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170003


namespace range_of_s_l170_170466

def s (x : ℝ) : ℝ := 1 / (2 - x)^2

theorem range_of_s : set.Ioo 0 ∞ = set.range s :=
sorry

end range_of_s_l170_170466


namespace pasta_needed_for_family_reunion_l170_170158

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l170_170158


namespace range_of_a_minus_b_l170_170252

theorem range_of_a_minus_b {a b : ℝ} (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) : -6 < a - b ∧ a - b < 1 :=
by
  sorry -- The proof is skipped as per the instructions.

end range_of_a_minus_b_l170_170252


namespace flour_per_new_bread_roll_l170_170088

theorem flour_per_new_bread_roll (p1 f1 p2 f2 c : ℚ)
  (h1 : p1 = 40)
  (h2 : f1 = 1 / 8)
  (h3 : p2 = 25)
  (h4 : c = p1 * f1)
  (h5 : c = p2 * f2) :
  f2 = 1 / 5 :=
by
  sorry

end flour_per_new_bread_roll_l170_170088


namespace ideal_complex_relation_l170_170100

def is_ideal_complex (z : ℂ) : Prop :=
  z.re = -z.im

theorem ideal_complex_relation (a b : ℝ) (h : is_ideal_complex (a / (1 - 2 * complex.I) + b * complex.I)) :
  3 * a + 5 * b = 0 :=
sorry

end ideal_complex_relation_l170_170100


namespace second_piece_length_multiple_of_18_l170_170332

-- Define the conditions as given in the problem
def has_two_pieces (len1 len2 : ℕ) : Prop :=
  len1 = 90 ∧ (∃ k, len2 = k * 18)

-- Define the main proof statement
theorem second_piece_length_multiple_of_18 : ∀ (len2 : ℕ), (∃ k, k * 18 = len2) :=
by
  assume len2,
  sorry

end second_piece_length_multiple_of_18_l170_170332


namespace product_of_distances_is_constant_l170_170212

noncomputable def squared_distance (A B : Point) : ℝ :=
  (A.x - B.x)^2 + (A.y - B.y)^2

theorem product_of_distances_is_constant
    (A : Point)
    (ω : Circle)
    (B C D E : Point)
    (hA_outside : ¬ ω.contains A)
    (hB_on_circle : ω.contains B)
    (hC_on_circle : ω.contains C)
    (hD_on_circle : ω.contains D)
    (hE_on_circle : ω.contains E)
    (h_ray1 : linear_path (A, B))
    (h_ray1 : linear_path (A, C))
    (h_ray2 : linear_path (A, D))
    (h_ray2 : linear_path (A, E))
    : squared_distance A B * squared_distance A C = squared_distance A D * squared_distance A E :=
by
  sorry

end product_of_distances_is_constant_l170_170212


namespace find_angle_ABC_l170_170668

/- Definitions corresponding to given conditions -/
variables (A B C D E F : Type)
variables [Collinear B D E C] (angle : A → A → ℝ)
variables (perpendicular : A → A → ℝ → Prop)
variables (length : A → A → ℝ)
variables (x : ℝ)
variables (BAD DAE : ∀ A B D E : A, angle B A D = 12 ∧ angle D A E = 12)
variables (perpendicular (AD : A) (AC : A) 90)
variables (BC : A → A → ℝ)
variables (AB : ℝ)
variables (AE AF : ℝ)

/- Problem Statement -/
theorem find_angle_ABC 
(Collinear B D E C : Prop)
(angle : ∀ A B C : A, angle B A D = 12 ∧ angle D A E = 12)
(perpendicular : ∀ AC AD : A, 90)
(BC : A → A → ℝ)
(AB AE AF : ℝ)
(AF = AE)
(BC = AB + AE)
: (x = angle B F A → x = 44)
 := 
 sorry

end find_angle_ABC_l170_170668


namespace find_plane_equation_l170_170973

noncomputable def planeEquation (A B C D : ℕ) : Prop :=
  A > 0 ∧ Int.gcd A (Int.gcd B (Int.gcd C D)) = 1 ∧
  ∃ (x y z : ℤ), (x = 0 ∧ y = 2 ∧ z = 1) ∨ (x = 2 ∧ y = 0 ∧ z = 1) → A * x + B * y + C * z + D = 0 ∧
  (∀ (x y z : ℤ), 2 * x - y + 3 * z = 4 → A * x + B * y + C * z = 0)

theorem find_plane_equation :
  ∃ (A B C D : ℕ), planeEquation A B C D ∧ 
  ∀ (x y z : ℤ), (x = 0 ∧ y = 2 ∧ z = 1) → (A * x + B * y + C * z + D = 0) ∧
                  (x = 2 ∧ y = 0 ∧ z = 1) → (A * x + B * y + C * z + D = 0) ∧
                  (A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1) :=
sorry

end find_plane_equation_l170_170973


namespace solve_for_x_l170_170170

theorem solve_for_x : ∃ x, (1 / (x - 1) = 3 / (x - 3)) → x = 0 :=
begin
  sorry
end

end solve_for_x_l170_170170


namespace evaluate_expression_l170_170868

theorem evaluate_expression : 
  (2 ^ 2015 + 2 ^ 2013 + 2 ^ 2011) / (2 ^ 2015 - 2 ^ 2013 + 2 ^ 2011) = 21 / 13 := 
by 
 sorry

end evaluate_expression_l170_170868


namespace set_inter_compl_eq_l170_170715

def U := ℝ
def M : Set ℝ := { x | abs (x - 1/2) ≤ 5/2 }
def P : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def complement_U_M : Set ℝ := { x | x < -2 ∨ x > 3 }

theorem set_inter_compl_eq :
  (complement_U_M ∩ P) = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end set_inter_compl_eq_l170_170715


namespace compute_P_2a_4b_l170_170347

-- Let P be a matrix, and let a and b be vectors
variable {P : Matrix (Fin 2) (Fin 2) ℝ}
variable {a b : Vector (Fin 2) ℝ}

-- Conditions
def P_a := P.mulVec a = ![5, -1]
def P_b := P.mulVec b = ![3, 2]

-- Theorem
theorem compute_P_2a_4b (h1 : P.mulVec a = ![5, -1]) (h2 : P.mulVec b = ![3, 2]) :
  P.mulVec (2 • a - 4 • b) = ![-2, -10] :=
sorry

end compute_P_2a_4b_l170_170347


namespace equal_elements_l170_170425

theorem equal_elements
  (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ) 
  (hperm : ∀ i, x' i ∈ Finset.univ.map x)
  (heq : ∀ i, 
    let j := (i : ℕ)
    let next := (j + 1) % 2011
    x i + x ⟨next, sorry⟩ = 2 * x' i) :
  ∀ i j, x i = x j :=
  sorry

end equal_elements_l170_170425


namespace f_prime_one_third_zero_l170_170357

noncomputable def f (x : ℝ) : ℝ :=
if (∃ k : ℕ, ∀ n ≥ k, (decimal_digits x).nth n = some 9) then undefined
else ∑' n : ℕ, (decimal_digits x).nth (n+1) / 10^(2 * (n+1))

theorem f_prime_one_third_zero : deriv f (1 / 3) = 0 :=
sorry

end f_prime_one_third_zero_l170_170357


namespace gcd_228_1995_base3_to_base6_conversion_l170_170086

-- Proof Problem 1: GCD of 228 and 1995 is 57
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

-- Proof Problem 2: Converting base-3 number 11102 to base-6
theorem base3_to_base6_conversion : Nat.ofDigits 6 [3, 1, 5] = Nat.ofDigits 10 [1, 1, 1, 0, 2] :=
by
  sorry

end gcd_228_1995_base3_to_base6_conversion_l170_170086


namespace possible_values_of_k_l170_170274

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170274


namespace simple_closed_polygon_exists_l170_170593

-- Define the problem statement in Lean 4
theorem simple_closed_polygon_exists 
  (n : ℕ) (h : n ≥ 3)
  (points : set (ℝ × ℝ))
  (h_distinct : points.card = n)
  (h_collinear : ∀ P1 P2 P3 ∈ points, P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 → ¬ collinear ℝ {P1, P2, P3}) :
  ∃ polygon : set (ℝ × ℝ), 
  simple_closed_polygon polygon ∧ 
  ∀ P ∈ points, P ∈ polygon := 
      sorry

end simple_closed_polygon_exists_l170_170593


namespace min_f_value_l170_170258

noncomputable def f (a b : ℝ) := 
  Real.sqrt (2 * a^2 - 8 * a + 10) + 
  Real.sqrt (b^2 - 6 * b + 10) + 
  Real.sqrt (2 * a^2 - 2 * a * b + b^2)

theorem min_f_value : ∃ a b : ℝ, f a b = 2 * Real.sqrt 5 :=
sorry

end min_f_value_l170_170258


namespace elevator_travel_time_l170_170722

-- Definitions corresponding to the conditions
def total_floors : ℕ := 20
def first_half_floors_time : ℕ := 15 -- in minutes
def next_five_floors_time_per_floor : ℕ := 5 -- in minutes per floor
def final_five_floors_time_per_floor : ℕ := 16 -- in minutes per floor

-- Statement of the problem
theorem elevator_travel_time :
  let first_half_floors := total_floors / 2,
      next_five_floors := 5,
      final_five_floors := 5,
      total_time_in_minutes := first_half_floors_time + 
                               (next_five_floors * next_five_floors_time_per_floor) + 
                               (final_five_floors * final_five_floors_time_per_floor)
  in total_time_in_minutes / 60 = 2 := 
sorry

end elevator_travel_time_l170_170722


namespace range_of_a_l170_170061

noncomputable def isIncreasing (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, x < y → f x < f y

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → e ^ x - 1 ≥ x ^ 2 - a * x) → a ≥ 2 - Real.exp 1 :=
by
  sorry

end range_of_a_l170_170061


namespace B_finish_work_in_10_days_l170_170093

variable (W : ℝ) -- amount of work
variable (x : ℝ) -- number of days B can finish the work alone

theorem B_finish_work_in_10_days (h1 : ∀ A_rate, A_rate = W / 4)
                                (h2 : ∀ B_rate, B_rate = W / x)
                                (h3 : ∀ Work_done_together Remaining_work,
                                      Work_done_together = 2 * (W / 4 + W / x) ∧
                                      Remaining_work = W - Work_done_together ∧
                                      Remaining_work = (W / x) * 3.0000000000000004) :
  x = 10 :=
by
  sorry

end B_finish_work_in_10_days_l170_170093


namespace boat_trip_distance_l170_170898

-- Defining the conditions given in the problem
def downstream_speed (boat_speed stream_velocity : ℝ) : ℝ := boat_speed + stream_velocity
def upstream_speed (boat_speed stream_velocity : ℝ) : ℝ := boat_speed - stream_velocity

def total_time (distance : ℝ) (downstream_speed upstream_speed : ℝ) : ℝ :=
  (distance / downstream_speed) + ((distance / 2) / upstream_speed)

def distance_between_A_and_B := 122.14

-- Main theorem statement
theorem boat_trip_distance 
  (boat_speed stream_velocity : ℝ) (total_travel_time : ℝ) :
  boat_speed = 14 → stream_velocity = 4 → total_travel_time = 19 →
  total_time distance_between_A_and_B (downstream_speed boat_speed stream_velocity) (upstream_speed boat_speed stream_velocity) = total_travel_time :=
by sorry

end boat_trip_distance_l170_170898


namespace evelyn_found_caps_l170_170968

theorem evelyn_found_caps (start_caps end_caps found_caps : ℕ) 
    (h1 : start_caps = 18) 
    (h2 : end_caps = 81) 
    (h3 : found_caps = end_caps - start_caps) :
  found_caps = 63 := by
  sorry

end evelyn_found_caps_l170_170968


namespace general_formula_Tn_value_l170_170604

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∀ n, a n = 3 * (3 ^ (n - 1))

def Sn (a : ℕ → ℝ) (n : ℕ) := (finset.range n).sum a

def an_value (a : ℕ → ℝ) (Sn : (ℕ → ℝ) → ℕ → ℝ) (n : ℕ) := 
  3 * Sn a 1 :: 2 * Sn a 2 :: Sn a 3 :: list.nil : list ℝ

theorem general_formula
  (a : ℕ → ℝ)
  (h1: a 1 = 3)
  (h2: geometric_sequence a)
  (arithmetic_seq: list.arith_seq (an_value a Sn) 1 1) :
  ∀ n, a n = 3 ^ n :=
begin
  sorry
end

noncomputable def bn (a : ℕ → ℝ) (n : ℕ) := real.log 3 (a n)

def Tn (a : ℕ → ℝ) (n : ℕ) := 
  (finset.range n).sum (λ k, let i := 2 * k + 1 in bn a i * bn a (i + 1) - bn a (i + 1) * bn a (i + 2))

theorem Tn_value 
  (a : ℕ → ℝ)
  (h1: a 1 = 3)
  (h2: geometric_sequence a)
  (arithmetic_seq: list.arith_seq (an_value a Sn) 1 1) :
  ∀ n, Tn a n = -2 * n ^ 2 - 2 * n :=
begin
  sorry
end

end general_formula_Tn_value_l170_170604


namespace problem_I_problem_II_l170_170596

-- Declaration of function f(x)
def f (x a b : ℝ) := |x + a| - |x - b|

-- Proof 1: When a = 1, b = 1, solve the inequality f(x) > 1
theorem problem_I (x : ℝ) : (f x 1 1) > 1 ↔ x > 1/2 := by
  sorry

-- Proof 2: If the maximum value of the function f(x) is 2, prove that (1/a) + (1/b) ≥ 2
theorem problem_II (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max_f : ∀ x, f x a b ≤ 2) : 1 / a + 1 / b ≥ 2 := by
  sorry

end problem_I_problem_II_l170_170596


namespace total_tickets_l170_170885

-- Definitions based on given conditions
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def additional_tickets : ℕ := 6

-- Proof statement (only statement, proof is not required)
theorem total_tickets : (initial_tickets - spent_tickets + additional_tickets = 30) :=
  sorry

end total_tickets_l170_170885


namespace number_of_children_l170_170371

def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6
def number_of_adults := 2
def total_cost := 77

theorem number_of_children : 
  ∃ (x : ℕ), cost_of_child_ticket * x + cost_of_adult_ticket * number_of_adults = total_cost ∧ x = 3 :=
by
  sorry

end number_of_children_l170_170371


namespace projection_matrix_l170_170573

def u : ℝ^3 := ![3, -1, 4]
def Q : Matrix (Fin 3) (Fin 3) ℝ := !![
  [9/26, -3/26, 12/26],
  [-3/26, 1/26, -4/26],
  [12/26, -4/26, 16/26]
]

theorem projection_matrix :
  ∀ (v : ℝ^3), (Q ⬝ v) = (let dot_uv := u ⬝ᵥ v in (dot_uv * u) / (u ⬝ᵥ u)) :=
by
  sorry

end projection_matrix_l170_170573


namespace minimum_value_cos_2x_l170_170414

theorem minimum_value_cos_2x : ∃ x ∈ Ioc 0 π, cos (2 * x) = -1 :=
by
  -- Use the necessary conditions and properties to construct the proof
  sorry

end minimum_value_cos_2x_l170_170414


namespace scientific_notation_of_area_l170_170401

theorem scientific_notation_of_area :
  (0.0000064 : ℝ) = 6.4 * 10 ^ (-6) := 
sorry

end scientific_notation_of_area_l170_170401


namespace P_n_correct_l170_170343

-- We define the problem conditions
variables (S : finset ℕ) (A B : finset ℕ) (n : ℕ)

-- Maximum number in set A is less than the minimum number in set B
def condition (A B : finset ℕ) : Prop :=
  ∃ (maxA ∈ A) (minB ∈ B), maxA < minB

-- Definition of P_n as the number of satisfying pairs (A, B)
def P : ℕ → ℕ
| 2 := 1
| 3 := 5
| n := (n-2) * 2^(n-1) + 1

-- Main theorem for proving the number of pairs
theorem P_n_correct (n : ℕ) (h : n ≥ 2) : P n = (n-2) * 2^(n-1) + 1 :=
by
  cases n;
  { simp [P], sorry },
  cases n;
  { simp [P], sorry },
  { rw [P], sorry }

end P_n_correct_l170_170343


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170006

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170006


namespace range_s_l170_170459

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_s : set.range s = set.Ioi 0 := by
  sorry

end range_s_l170_170459


namespace students_with_both_pets_l170_170172

theorem students_with_both_pets :
  ∀ (total_students students_with_dog students_with_cat students_with_both : ℕ),
    total_students = 45 →
    students_with_dog = 25 →
    students_with_cat = 34 →
    total_students = students_with_dog + students_with_cat - students_with_both →
    students_with_both = 14 :=
by
  intros total_students students_with_dog students_with_cat students_with_both
  sorry

end students_with_both_pets_l170_170172


namespace product_of_consecutive_integers_sqrt_50_l170_170845

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170845


namespace roommates_condition_l170_170689

def f (x : ℝ) := 3 * x ^ 2 + 5 * x - 1
def g (x : ℝ) := 2 * x ^ 2 - 3 * x + 5

theorem roommates_condition : f 3 = 2 * g 3 + 5 := 
by {
  sorry
}

end roommates_condition_l170_170689


namespace incorrect_statement_C_l170_170932

-- Define the conditions
def number_of_fungi_less_than_bacteria := ∀ (soil : Type), fungi_count(soil) < bacteria_count(soil)
def spreading_same_dilution_accuracy := ∀ (dilution : Type) (plates : Type), ((sample_spread(dilution, plates) = sample_spread(dilution, plates)) → accurate_results(plates))
def averaging_colonies_more_accurate := ∀ (dilutions : Type) (colony_counts : Type), average_colonies(dilutions, colony_counts) = accurate_results(colony_counts)
def control_group_requirement := ∀ (sterile_water : Type), (experiment_with_control(sterile_water) = credible_results)

-- The main theorem
theorem incorrect_statement_C : number_of_fungi_less_than_bacteria ∧ 
                                spreading_same_dilution_accuracy ∧ 
                                averaging_colonies_more_accurate ∧ 
                                control_group_requirement 
                                → ¬ accurate_results (average_colonies(soil_dilutions, colony_counts)) := 
by
    intros h,
    cases h with h1 h_tail,
    cases h_tail with h2 h_tail,
    cases h_tail with h3 h4,
    sorry

end incorrect_statement_C_l170_170932


namespace find_c_value_l170_170124

theorem find_c_value (x y n m c : ℕ) 
  (h1 : 10 * x + y = 8 * n) 
  (h2 : 10 + x + y = 9 * m) 
  (h3 : c = x + y) : 
  c = 8 := 
by
  sorry

end find_c_value_l170_170124


namespace hyperbola_eccentricity_range_l170_170945

theorem hyperbola_eccentricity_range
  {a b x0 : ℝ}
  (ha : a > 0)
  (hb : b > 0)
  (hx0 : x0 > 1)
  (H : ∃ y0 : ℝ, y0^2 = x0 ∧ y0 = (b / a) * x0)
  : 1 < sqrt ((a^2 + b^2) / a^2) ∧ sqrt ((a^2 + b^2) / a^2) < sqrt 2 := 
sorry

end hyperbola_eccentricity_range_l170_170945


namespace line_passing_through_first_and_third_quadrants_l170_170295

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170295


namespace exists_congruent_triangle_cover_l170_170081

-- Define a triangle using three points
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define a polygon as a list of points, with the condition it must be convex
structure ConvexPolygon :=
(points : List Point)
(convex : ... ) -- Convexity condition (to be defined formally)

-- Define congruence of triangles
def congruent_triangles (Δ₁ Δ₂ : Triangle) : Prop :=
sorry -- Congruence conditions to be defined

-- Define that a triangle can cover a polygon
def triangle_covers_polygon (Δ : Triangle) (M : ConvexPolygon) : Prop :=
sorry -- Cover condition to be defined

-- The proof statement
theorem exists_congruent_triangle_cover (ΔABC : Triangle) (M : ConvexPolygon)
  (h_cover : triangle_covers_polygon ΔABC M) :
  ∃ (Δ' : Triangle), congruent_triangles Δ' ΔABC ∧ triangle_covers_polygon Δ' M ∧
  (∃ (side_Δ' side_M : (Point × Point)),
    (side_of_triangle Δ' side_Δ') ∧ (side_of_polygon M side_M) ∧ (side_Δ'.1 - side_Δ'.2).y = (side_M.1 - side_M.2).y) :=
sorry

end exists_congruent_triangle_cover_l170_170081


namespace book_club_pairing_l170_170929

noncomputable def book_club_pairs :=
  let members := ['Alice', 'Bob', 'Carol', 'Dan', 'Emily', 'Frank'] in
  let num_pairs_or_groups := 44 in
  num_pairs_or_groups

theorem book_club_pairing :
  book_club_pairs = 44 := sorry

end book_club_pairing_l170_170929


namespace conditional_probability_l170_170529

variable (pA pB pAB : ℝ)
variable (h1 : pA = 0.2)
variable (h2 : pB = 0.18)
variable (h3 : pAB = 0.12)

theorem conditional_probability : (pAB / pB = 2 / 3) :=
by
  -- sorry is used to skip the proof
  sorry

end conditional_probability_l170_170529


namespace prove_g_of_f_g_l170_170362

noncomputable def f (x : Polynomial ℤ) : Polynomial ℤ := x^2 + 2 * x + 1 -- Example polynomial for f(x)
def g (x : Polynomial ℤ) : Polynomial ℤ := x^2 + 20 * x - 20 -- g(x) we want to prove

theorem prove_g_of_f_g (f g : Polynomial ℤ)
  (h₁ : ∀ x, f.eval (g.eval x) = (f.eval x) * (g.eval x))
  (h₂ : g.eval 3 = 50) :
  g = Polynomial.C(1) * x^2 + Polynomial.C(20) * x - Polynomial.C(20) := 
sorry

end prove_g_of_f_g_l170_170362


namespace roots_reciprocal_sum_l170_170353

theorem roots_reciprocal_sum (p q r : ℝ) (h : polynomial.eval p (X^3 - 2*X + 2) = 0 ∧ 
                                       polynomial.eval q (X^3 - 2*X + 2) = 0 ∧ 
                                       polynomial.eval r (X^3 - 2*X + 2) = 0) :
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2)) = 3 / 5 := 
by
  sorry

end roots_reciprocal_sum_l170_170353


namespace product_of_consecutive_integers_sqrt_50_l170_170847

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170847


namespace car_exhaust_negative_correlation_l170_170550

variable (CarExhaust AirQuality : Type)
variable (affects : CarExhaust → AirQuality → Prop)
variable (increase : CarExhaust → CarExhaust → Prop)
variable (worsen : AirQuality → AirQuality → Prop)

theorem car_exhaust_negative_correlation (c₁ c₂ : CarExhaust) (a₁ a₂ : AirQuality) :
  (affects c₁ a₁) →
  (increase c₁ c₂) →
  (worsen a₁ a₂) →
  (negative_correlation : ∀ c a, affects c a →
                            increase c c₂ → worsen a a₂) :=
  sorry

end car_exhaust_negative_correlation_l170_170550


namespace max_connected_stations_l170_170524

theorem max_connected_stations (n : ℕ) 
  (h1 : ∀ s : ℕ, s ≤ n → s ≤ 3) 
  (h2 : ∀ x y : ℕ, x < y → ∃ z : ℕ, z < 3 ∧ z ≤ n) : 
  n = 10 :=
by 
  sorry

end max_connected_stations_l170_170524


namespace males_only_in_band_l170_170398

theorem males_only_in_band
  (females_in_band : ℕ)
  (males_in_band : ℕ)
  (females_in_orchestra : ℕ)
  (males_in_orchestra : ℕ)
  (females_in_both : ℕ)
  (total_students : ℕ)
  (total_students_in_either : ℕ)
  (hf_in_band : females_in_band = 120)
  (hm_in_band : males_in_band = 90)
  (hf_in_orchestra : females_in_orchestra = 100)
  (hm_in_orchestra : males_in_orchestra = 130)
  (hf_in_both : females_in_both = 80)
  (h_total_students : total_students = 260) :
  total_students_in_either = 260 → 
  (males_in_band - (90 + 130 + 80 - 260 - 120)) = 30 :=
by
  intros h_total_students_in_either
  sorry

end males_only_in_band_l170_170398


namespace limit_expression_l170_170547

open Real

theorem limit_expression :
  (∃ L : ℝ, (filter.tendsto (λ x : ℝ, (1 + cos (π * x)) / (tan (π * x)) ^ 2) (nhds 1) (nhds L)) ∧
            (filter.tendsto (λ x : ℝ, (λ x, ((1 + cos (π * x)) / (tan (π * x)) ^ 2) ^ (x ^ 2))) (nhds 1) (nhds L)) ∧
            L = 1 / 2) :=
sorry

end limit_expression_l170_170547


namespace rectangle_area_l170_170307

variable (x y k : ℝ) -- Define variables
variable (smaller_side larger_side base altitude diagonal : ℝ)

theorem rectangle_area :
  smaller_side = 6 →
  larger_side = 10 →
  k = 3 →
  diagonal = sqrt (smaller_side^2 + larger_side^2) →
  base = 2 * smaller_side + (larger_side - k) →
  altitude = (diagonal / 2) + k →
  base * altitude = (19 * (sqrt 136 + 6)) / 2 :=
by
  sorry

end rectangle_area_l170_170307


namespace no_real_roots_iff_k_gt_2_l170_170196

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l170_170196


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170009

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170009


namespace num_sequences_eq_15_l170_170420

noncomputable def num_possible_sequences : ℕ :=
  let angles_increasing_arith_seq := ∃ (x d : ℕ), x > 0 ∧ x + 4 * d < 140 ∧ 5 * x + 10 * d = 540 ∧ d ≠ 0
  by sorry

theorem num_sequences_eq_15 : num_possible_sequences = 15 := 
  by sorry

end num_sequences_eq_15_l170_170420


namespace anie_days_to_finish_task_l170_170442

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l170_170442


namespace divisibility_theorem_l170_170696

theorem divisibility_theorem (n : ℕ) (h1 : n > 0) (h2 : ¬(2 ∣ n)) (h3 : ¬(3 ∣ n)) (k : ℤ) :
  (k + 1) ^ n - k ^ n - 1 ∣ k ^ 2 + k + 1 :=
sorry

end divisibility_theorem_l170_170696


namespace smallest_k_with_exactly_one_prime_l170_170578

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def count_primes_in_range (k : ℕ) (a b : ℕ) : ℕ :=
((list.range' (b - a + 1) a).map (λ x, k * x + 60)).countp is_prime

theorem smallest_k_with_exactly_one_prime :
  ∃ (k : ℕ), (k > 0) ∧ (count_primes_in_range k 0 10 = 1) ∧ 
  (∀ k' : ℕ, k' > 0 → count_primes_in_range k' 0 10 = 1 → k ≤ k') :=
begin
  use 17,
  split,
  { exact nat.succ_pos' 16 },
  split,
  { -- Proof that count_primes_in_range 17 0 10 = 1
    -- This part would require a detailed proof involving checking each value of x
    -- to ensure there's exactly one prime in the set {17x + 60 | 0 ≤ x ≤ 10}
    sorry },
  { intros k' hk' hcount,
    -- Proof that 17 is the smallest k satisfying the conditions
    -- This part would involve showing no smaller k results in exactly one prime
    sorry }
end

end smallest_k_with_exactly_one_prime_l170_170578


namespace program_of_five_courses_l170_170522

/-
 A student must choose a program of five courses from a list consisting of English, Algebra, Geometry, History, Art, Latin, and Science. 
 This program must contain English, at least one mathematics course (Algebra or Geometry), and History.
 Prove that there are exactly 9 ways to create such a program.
-/

theorem program_of_five_courses (L : list string) 
  (h1 : "English" ∈ L)
  (h2 : "History" ∈ L)
  (h3 : "Algebra" ∈ L ∨ "Geometry" ∈ L)
  (h  : L.length = 5) :
  9 = 
    (nat.choose 5 3) - 
    if "Algebra" ∉ L ∧ "Geometry" ∉ L then 1 else 0 := sorry

end program_of_five_courses_l170_170522


namespace range_sum_f_lt_zero_l170_170621

theorem range_sum_f_lt_zero
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(-x) = -f(x+4))
  (h2 : ∀ x > 2, ∀ y > x, f(y) > f(x))
  (x1 x2 : ℝ)
  (h3 : x1 + x2 < 4)
  (h4 : (x1 - 2) * (x2 - 2) < 0) :
  f(x1) + f(x2) < 0 :=
  sorry

end range_sum_f_lt_zero_l170_170621


namespace smallest_pretty_num_l170_170916

-- Define the notion of a pretty number
def is_pretty (n : ℕ) : Prop :=
  ∃ d1 d2 : ℕ, (1 ≤ d1 ∧ d1 ≤ n) ∧ (1 ≤ d2 ∧ d2 ≤ n) ∧ d2 - d1 ∣ n ∧ (1 < d1)

-- Define the statement to prove that 160400 is the smallest pretty number greater than 401 that is a multiple of 401
theorem smallest_pretty_num (n : ℕ) (hn1 : n > 401) (hn2 : n % 401 = 0) : n = 160400 :=
  sorry

end smallest_pretty_num_l170_170916


namespace product_of_consecutive_integers_between_sqrt_50_l170_170827

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170827


namespace compare_quadratics_maximize_rectangle_area_l170_170889

-- (Ⅰ) Problem statement for comparing quadratic expressions
theorem compare_quadratics (x : ℝ) : (x + 1) * (x - 3) > (x + 2) * (x - 4) := by
  sorry

-- (Ⅱ) Problem statement for maximizing rectangular area with given perimeter
theorem maximize_rectangle_area (x y : ℝ) (h : 2 * (x + y) = 36) : 
  x = 9 ∧ y = 9 ∧ x * y = 81 := by
  sorry

end compare_quadratics_maximize_rectangle_area_l170_170889


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170034

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170034


namespace find_sales_volume_formula_highest_profit_l170_170492

def cost_per_kg : ℝ := 8

def selling_price (x : ℕ) : ℝ := 0.5 * x + 18

def sales_volume (x : ℕ) : ℝ :=
  if x = 2 then 33
  else if x = 5 then 30
  else if x = 9 then 26
  else 0  -- Placeholder for other x values

theorem find_sales_volume_formula :
  ∀ x, 1 ≤ x ∧ x ≤ 10 → x ∈ {2, 5, 9} ∨ (sales_volume x = -x + 35) :=
sorry

theorem highest_profit :
  ∃ d : ℕ, 1 ≤ d ∧ d ≤ 10 ∧ (d = 7 ∨ d = 8) ∧ 
  ∀ x, 1 ≤ x ∧ x ≤ 10 → 
  let w := (-x + 35) * (0.5 * x + 10) in
  w ≤ 378 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 10 → let wy := (-y + 35) * (0.5 * y + 10) in wy ≤ w :=
sorry

end find_sales_volume_formula_highest_profit_l170_170492


namespace transformation_result_l170_170857

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => f (k * x)

theorem transformation_result :
  (compress_horizontal (translate_left initial_function (Real.pi / 3)) 2) x = Real.sin (4 * x + (2 * Real.pi / 3)) :=
sorry

end transformation_result_l170_170857


namespace five_digit_integer_probability_l170_170107

theorem five_digit_integer_probability :
  let prob_units := (2 / 10 : ℚ),
      prob_hundreds := (5 / 10 : ℚ),
      prob_total := prob_units * prob_hundreds
  in prob_total = (1 / 10 : ℚ) :=
by
  sorry

end five_digit_integer_probability_l170_170107


namespace sqrt_50_product_consecutive_integers_l170_170813

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170813


namespace statement_A_statement_C_statement_D_l170_170065

theorem statement_A (x : ℝ) :
  (¬ (∀ x ≥ 3, 2 * x - 10 ≥ 0)) ↔ (∃ x0 ≥ 3, 2 * x0 - 10 < 0) := 
sorry

theorem statement_C {a b c : ℝ} (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

theorem statement_D {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (a / b) > ((a + m) / (b + m)) := 
sorry

end statement_A_statement_C_statement_D_l170_170065


namespace sqrt_50_between_7_and_8_l170_170799

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170799


namespace part_one_solution_exists_solutions_with_associated_value_4_minimum_associated_value_and_range_l170_170586

-- Part (1) - Proving that there exists a solution and its associated value
theorem part_one_solution_exists : ∃ (x y : ℤ), x - 2 * y = 2 ∧ (if |x| ≥ |y| then |x| else |y|) = 1 := 
sorry

-- Part (2) - Solutions such that the associated value is 4
theorem solutions_with_associated_value_4 : 
  { (x, y : ℤ) // x - 2 * y = 2 ∧ (if |x| ≥ |y| then |x| else |y|) = 4 } = { (4, 1), (-4, -3) } := 
sorry

-- Part (3) - Minimum associated value and range of x
theorem minimum_associated_value_and_range :
  ∃ x y : ℚ, x - 2 * y = 2 ∧ 
  (if |x| ≥ |y| then |x| else |y|) = 2/3 ∧ 
  ∀ z : ℚ, (x = z) ↔ (z ≥ (1/3) ∨ z ≤ (-2)) :=
sorry

end part_one_solution_exists_solutions_with_associated_value_4_minimum_associated_value_and_range_l170_170586


namespace alternating_sum_10000_l170_170476

theorem alternating_sum_10000 : 
  ∑ n in finset.range 10000, ((-1) ^ (n + 1)) * (n + 1) = -5000 := 
sorry

end alternating_sum_10000_l170_170476


namespace find_acd_over_b_l170_170187

theorem find_acd_over_b (a b c d : ℤ) (x : ℝ) :
  (x = (a + b * real.sqrt c) / d) →
  (7 * x / 4 + 2 = 4 / x) →
  (7 * x^2 + 8 * x - 16 = 0) →
  (c = 2) →
  ∃ a b d : ℤ, (x = (a + b * real.sqrt c) / d) ∧ (a * c * d / b = -7) :=
begin
  intros,
  sorry
end

end find_acd_over_b_l170_170187


namespace largest_hexagon_area_proof_l170_170744

noncomputable def largest_hexagon_area (s : ℝ) : ℝ :=
(3 * real.sqrt 3 / 2) * s^2

def hexagon_in_rectangle (a b : ℕ) (W H : ℝ) :=
∃ s : ℝ, W = 20 ∧ H = 22 ∧
  largest_hexagon_area s = a * real.sqrt b - c ∧
  100 * a + 10 * b + c = 134610

theorem largest_hexagon_area_proof :
  hexagon_in_rectangle 1326 3 20 22 :=
sorry

end largest_hexagon_area_proof_l170_170744


namespace max_a_condition_l170_170251

theorem max_a_condition (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) → a = -1 :=
by
  sorry

end max_a_condition_l170_170251


namespace car_speed_l170_170901

-- Defining the given conditions as constants
constants (distance time : ℕ) (conversion_factor speed_mps speed_kmph : ℝ)
-- Assuming the conditions are given
axiom distance_value : distance = 450
axiom time_value : time = 15
axiom conversion_factor_value : conversion_factor = 3.6

-- Defining the speed in meters per second
def calculate_speed_mps := distance / time

-- Defining the speed in kilometers per hour
def calculate_speed_kmph := calculate_speed_mps * conversion_factor

-- The Lean statement to prove
theorem car_speed : calculate_speed_kmph = 108 := by
  -- Details of the proof are omitted
  sorry

end car_speed_l170_170901


namespace length_gh_parallel_lines_l170_170948

theorem length_gh_parallel_lines (
    AB CD EF GH : ℝ
) (
    h1 : AB = 300
) (
    h2 : CD = 200
) (
    h3 : EF = (AB + CD) / 2 * (1 / 2)
) (
    h4 : GH = EF * (1 - 1 / 4)
) :
    GH = 93.75 :=
by
    sorry

end length_gh_parallel_lines_l170_170948


namespace sequence_sum_l170_170991

open Real

def a (n : ℕ) := 3 * n + sqrt (n^2 - 1)
def b (n : ℕ) := 2 * (sqrt (n^2 + n) + sqrt (n^2 - n))

theorem sequence_sum :
  ∃ (A B : ℤ), (∑ n in Finset.range 49 | n > 0, sqrt (a n - b n)) = (A : ℝ) + B * sqrt 2 :=
by
  existsi (-5 : ℤ)
  existsi (4 : ℤ)
  sorry

end sequence_sum_l170_170991


namespace product_of_consecutive_integers_sqrt_50_l170_170815

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170815


namespace product_of_consecutive_integers_sqrt_50_l170_170794

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170794


namespace least_k_l170_170879

theorem least_k (k : ℤ) (h : 0.0010101 * 10 ^ k > 100) : k ≥ 6 :=
sorry

end least_k_l170_170879


namespace conic_eccentricity_l170_170094

def point := ℝ × ℝ

def A : point := (-2, 2 * Real.sqrt 3)
def B : point := (1, -3)

-- Define the general equation of a conic section passing through specific points
def conic_section (m n : ℝ) (p : point) : Prop :=
  (m * p.1 ^ 2) + (n * p.2 ^ 2) = 1

theorem conic_eccentricity :
  ∃ m n : ℝ,
  conic_section m n A ∧ conic_section m n B ∧
  (let a := Real.sqrt (if m > 0 then 1/m else -1/m),
       b := Real.sqrt (if n > 0 then 1/n else -1/n),
       c := Real.sqrt (a ^ 2 + b ^ 2) in
    c / a = Real.sqrt 2) :=
begin
  sorry
end

end conic_eccentricity_l170_170094


namespace sum_of_roots_of_P_is_8029_l170_170967

-- Define the polynomial
noncomputable def P : Polynomial ℚ :=
  (Polynomial.X - 1)^2008 + 
  3 * (Polynomial.X - 2)^2007 + 
  5 * (Polynomial.X - 3)^2006 + 
  -- Continue defining all terms up to:
  2009 * (Polynomial.X - 2008)^2 + 
  2011 * (Polynomial.X - 2009)

-- The proof problem statement
theorem sum_of_roots_of_P_is_8029 :
  (P.roots.sum = 8029) :=
sorry

end sum_of_roots_of_P_is_8029_l170_170967


namespace man_age_twice_son_age_in_two_years_l170_170910

theorem man_age_twice_son_age_in_two_years :
  ∀ (S M X : ℕ), S = 30 → M = S + 32 → (M + X = 2 * (S + X)) → X = 2 :=
by
  intros S M X hS hM h
  sorry

end man_age_twice_son_age_in_two_years_l170_170910


namespace bob_winning_strategy_alice_winning_strategy_l170_170533

theorem bob_winning_strategy :
  ∀ (board : Fin 8 × Fin 8) (initial_positions : (Fin 8 × Fin 8) × (Fin 8 × Fin 8)),
  (∀ (rounds : ℕ) (alice_move : (Fin 8 × Fin 8)) (bob_move : (Fin 8 × Fin 8)),
    alice_move = initial_positions.1 →
    bob_move = initial_positions.2 →
    ((rounds < 2012 ∧ alice_move ≠ bob_move) ∨ bob_move = alice_move)) →
  bob_move = initial_positions.2 →
  ¬(∀ (rounds : ℕ), rounds < 2012 → initial_positions.1 = initial_positions.2) →
  ∃ (strategy : (Fin 8 × Fin 8) → (Fin 8 × Fin 8)),
  strategy = initial_positions.2 →
  (∀ (pos : Fin 8 × Fin 8), strategy pos = initial_positions.2) :=
sorry

theorem alice_winning_strategy :
  ∀ (board : Fin 8 × Fin 8) (initial_positions : (Fin 8 × Fin 8) × (Fin 8 × Fin 8)),
  (∀ (rounds : ℕ) (alice_move : (Fin 8 × Fin 8)) (bob_move : (Fin 8 × Fin 8)),
    alice_move = initial_positions.1 →
    bob_move = initial_positions.2 →
    ((rounds < 2012 ∧ alice_move ≠ bob_move) ∨ alice_move = bob_move)) →
  alice_move = initial_positions.1 →
  (∀ (rounds : ℕ), rounds ≤ 14 → alice_move = bob_move) →
  ∃ (strategy : (Fin 8 × Fin 8) → (Fin 8 × Fin 8)),
  strategy = initial_positions.1 →
  (∀ (pos : Fin 8 × Fin 8), strategy pos = bob_move) :=
sorry

end bob_winning_strategy_alice_winning_strategy_l170_170533


namespace angle_between_medians_and_base_l170_170406

-- Define the problem in Lean 4
theorem angle_between_medians_and_base
  (a m : ℝ)
  (h : 4 * a = 2 * real.pi * m) :
  let tan_beta := (real.sqrt 8) / (real.pi * real.sqrt 5)
  in sorry := -- placeholder for angle conversion to degrees if necessary 
  real.arctan tan_beta = -- The required angle is about 21 degrees 55 minutes.
  real.angle.of_degrees 21.9167 :=
sorry

end angle_between_medians_and_base_l170_170406


namespace length_of_CD_l170_170447

variable (A B C D E : Type*)
variable [Ring A]
variable [LinearOrder A]

def Trapezoid (AD BD BC CD : A) (DBAngle BDCAngle : A) (ratioBCDA : A): Prop :=
  AD ∥ BC ∧ BD = 2 ∧ ∠ DBA = 30 ∧ ∠ BDC = 60 ∧ BC / AD = 5 / 3

theorem length_of_CD (AD BD BC CD : A) (DBAngle BDCAngle : A) (ratioBCDA : A)
  (h: Trapezoid AD BD BC CD DBAngle BDCAngle ratioBCDA) : CD = 1 / 2 :=
sorry

end length_of_CD_l170_170447


namespace smallest_c_equality_condition_l170_170359

variable {n : ℕ} (hn : n ≥ 2)
variable {x : ℕ → ℝ} (hx : ∀ i, 0 ≤ x i)

def F (x : ℕ → ℝ) := ∑ i in finset.range n, ∑ j in finset.filter (λ j, i < j) (finset.range n), x i * x j * (x i ^ 2 + x j ^ 2)

theorem smallest_c
  (sum_x_eq_one : ∑ i in finset.range n, x i = 1)
  : (F x) ≤ (1 / 8) * (∑ i in finset.range n, x i) ^ 4 :=
sorry

theorem equality_condition
  (sum_x_eq_one : ∑ i in finset.range n, x i = 1)
  (hF : F x = (1 / 8) * (∑ i in finset.range n, x i) ^ 4)
  : ∃ a b, (∃ i j, i ≠ j ∧ x i = a ∧ x j = b ∧ a > 0 ∧ b > 0) ∧ ∀ k, k ≠ i ∧ k ≠ j → x k = 0 :=
sorry

end smallest_c_equality_condition_l170_170359


namespace initial_dimes_count_l170_170388

theorem initial_dimes_count (dimes_borrowed : ℕ) (dimes_left : ℕ) :
  dimes_borrowed = 4 → dimes_left = 4 → (dimes_left + dimes_borrowed) = 8 :=
by
  intro h_borrowed h_left
  rw [h_borrowed, h_left]
  simp
  sorry

end initial_dimes_count_l170_170388


namespace veggies_count_l170_170908

def initial_tomatoes := 500
def picked_tomatoes := 325
def initial_potatoes := 400
def picked_potatoes := 270
def initial_cucumbers := 300
def planted_cucumber_plants := 200
def cucumbers_per_plant := 2
def initial_cabbages := 100
def picked_cabbages := 50
def planted_cabbage_plants := 80
def cabbages_per_cabbage_plant := 3

noncomputable def remaining_tomatoes : Nat :=
  initial_tomatoes - picked_tomatoes

noncomputable def remaining_potatoes : Nat :=
  initial_potatoes - picked_potatoes

noncomputable def remaining_cucumbers : Nat :=
  initial_cucumbers + planted_cucumber_plants * cucumbers_per_plant

noncomputable def remaining_cabbages : Nat :=
  (initial_cabbages - picked_cabbages) + planted_cabbage_plants * cabbages_per_cabbage_plant

theorem veggies_count :
  remaining_tomatoes = 175 ∧
  remaining_potatoes = 130 ∧
  remaining_cucumbers = 700 ∧
  remaining_cabbages = 290 :=
by
  sorry

end veggies_count_l170_170908


namespace max_min_difference_eq_one_l170_170618

theorem max_min_difference_eq_one (a b : ℝ) (M m : ℝ) 
  (h₁ : a ≠ 0 ∧ b ≠ 0)
  (h₂ : a^2 + b^2 = 1)
  (h₃ : M = max (λ y : ℝ, ∃ x : ℝ, y = (ax + b) / (x^2 + 1)))
  (h₄ : m = min (λ y : ℝ, ∃ x : ℝ, y = (ax + b) / (x^2 + 1))) :
  M - m = 1 :=
by
  sorry

end max_min_difference_eq_one_l170_170618


namespace distribute_consecutive_numbers_l170_170855

theorem distribute_consecutive_numbers (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℕ) 
    (h : {A₁, A₂, A₃, A₄, A₅, A₆, A₇} = {1, 2, 3, 4, 5, 6, 7}) : 
    ∃ (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℕ), 
      (A₁ + A₂ + A₃ = A₄ + A₅ + A₆) ∧ 
      (A₁ + A₂ + A₇ = A₃ + A₄ + A₅) ∧ 
      (A₇ + A₆ + A₂ = A₃ + A₄ + A₅) := sorry

end distribute_consecutive_numbers_l170_170855


namespace shaded_area_l170_170656

theorem shaded_area (side_length : ℝ) (radius : ℝ) (h_side : side_length = 10) (h_radius : radius = 3) :
  let area_of_square := side_length^2,
      area_of_shaded_region := area_of_square - 60 * real.sqrt 3 - 3 * real.pi in
  area_of_shaded_region = 100 - 60 * real.sqrt 3 - 3 * real.pi := 
by
  sorry

end shaded_area_l170_170656


namespace f_2015_l170_170708

noncomputable def f : ℝ+ → ℝ
  := sorry

axiom f_continuous : Continuous f

axiom f_functional_eq (x y : ℝ+) : f(x * y) = f(x) + f(y) + 1

axiom f_at_2 : f 2 = 0

theorem f_2015 : f 2015 = Real.log2 2015 - 1 :=
  by
  sorry

end f_2015_l170_170708


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170019

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170019


namespace groups_needed_for_sampling_l170_170856

def total_students : ℕ := 600
def sample_size : ℕ := 20

theorem groups_needed_for_sampling : (total_students / sample_size = 30) :=
by
  sorry

end groups_needed_for_sampling_l170_170856


namespace product_of_integers_around_sqrt_50_l170_170771

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170771


namespace possible_values_of_k_l170_170272

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170272


namespace total_roasted_marshmallows_l170_170680

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l170_170680


namespace ratio_of_pond_to_field_area_l170_170754

theorem ratio_of_pond_to_field_area
  (l w : ℕ)
  (field_area pond_area : ℕ)
  (h1 : l = 2 * w)
  (h2 : l = 36)
  (h3 : pond_area = 9 * 9)
  (field_area_def : field_area = l * w)
  (pond_area_def : pond_area = 81) :
  pond_area / field_area = 1 / 8 := 
sorry

end ratio_of_pond_to_field_area_l170_170754


namespace product_of_consecutive_integers_sqrt_50_l170_170791

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170791


namespace number_of_rounds_l170_170996

-- Define the initial state of the game
structure GameState :=
  (tokens : ℕ → ℕ) -- mapping from player index (0 to 3) to number of tokens
  (discardPile : ℕ)

-- Initial game state
def initialGameState : GameState :=
  { tokens := λ i, if i = 0 then 18 else if i = 1 then 17 else if i = 2 then 16 else 15,
    discardPile := 0 }

-- Function to simulate one round of the game
def playRound (s : GameState) : GameState :=
  let richest := (List.range 4).maxBy s.tokens
  let new_tokens := λ i, if i = richest then s.tokens i - 4 else s.tokens i + 1
  { tokens := new_tokens, discardPile := s.discardPile + 1 }

-- Function to check if any player is out of tokens
def playerOutOfTokens (s : GameState) : bool :=
  (List.range 4).any (λ i, s.tokens i = 0)

-- Function to simulate multiple rounds until a player runs out of tokens
def playGame (s : GameState) : ℕ × GameState :=
  let rec go (rounds : ℕ) (state : GameState) : ℕ × GameState :=
    if playerOutOfTokens state then (rounds, state)
    else go (rounds + 1) (playRound state)
  go 0 s

-- The proof statement
theorem number_of_rounds : (playGame initialGameState).fst = 57 :=
by sorry

end number_of_rounds_l170_170996


namespace find_l2_l170_170864

-- Definitions according to provided conditions
def Q : ℝ × ℝ := (-2, 3)
def Q'' : ℝ × ℝ := (5, -2)
def l1 (x y : ℝ) : Prop := 3 * x = y

-- Define the l2 to be determined
def candidate_l2a (x y : ℝ) : Prop := x + 4 * y = 0

-- Define a reflection function for a point about a line ax + by = 0
def reflect (a b : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := 
let (x, y) := P in
let denominator := (a^2 + b^2) in
let x' := ((a^2 - b^2) * x - 2 * a * b * y) / denominator in
let y' := ((b^2 - a^2) * y - 2 * a * b * x) / denominator in
(x', y')

-- Proof statement
theorem find_l2 : 
  (∃ l2 : ℝ × ℝ → Prop, 
    (∀ P, reflect 3 (-1) P = reflect 1 4 (reflect 3 (-1) Q)) ∧
    l2 = candidate_l2a) :=
by {
  sorry
}

end find_l2_l170_170864


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170275

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170275


namespace seashells_left_l170_170739

-- Definitions based on conditions
def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

-- Theorem stating the proof problem
theorem seashells_left (initial_seashells seashells_given_away : ℕ) : initial_seashells - seashells_given_away = 17 := 
    by
        sorry

end seashells_left_l170_170739


namespace negation_of_ex_negation_of_specific_proposition_l170_170758

theorem negation_of_ex (P : ℤ → Prop) : 
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, ¬ P x :=
by sorry

def specific_proposition (x : ℤ) : Prop :=
  x^2 = 2*x

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℤ, specific_proposition x) ↔ ∀ x : ℤ, ¬ specific_proposition x :=
by {
  apply negation_of_ex,
}

end negation_of_ex_negation_of_specific_proposition_l170_170758


namespace elevator_time_to_bottom_l170_170725

theorem elevator_time_to_bottom :
  ∀ (floors : ℕ) (first_half_time: ℕ) 
  (next_floors: ℕ) (next_floors_time_per_floor: ℕ) 
  (final_floors: ℕ) (final_floors_time_per_floor: ℕ)
  (total_floors: ℕ) (total_time_in_hours: ℕ),
  floors = 20 →
  first_half_time = 15 →
  next_floors = 5 →
  next_floors_time_per_floor = 5 →
  final_floors = 5 →
  final_floors_time_per_floor = 16 →
  total_floors = first_half_time +
                 next_floors * next_floors_time_per_floor +
                 final_floors * final_floors_time_per_floor →
  total_time_in_hours = total_floors / 60 →
  total_time_in_hours = 2 :=
begin
  intros,
  sorry
end

end elevator_time_to_bottom_l170_170725


namespace product_of_consecutive_integers_between_sqrt_50_l170_170830

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170830


namespace units_digit_of_5_pow_12_l170_170058

theorem units_digit_of_5_pow_12 : ∀ (n : ℕ), n > 0 → (5^n % 10 = 5) → (5^12 % 10 = 5) :=
by
  intros,
  sorry

end units_digit_of_5_pow_12_l170_170058


namespace length_of_room_l170_170333

def area_of_room : ℝ := 10
def width_of_room : ℝ := 2

theorem length_of_room : width_of_room * 5 = area_of_room :=
by
  sorry

end length_of_room_l170_170333


namespace parabola_focus_l170_170183

theorem parabola_focus (x f : ℝ) (hx : ∀ x : ℝ, (x, 2 * x^2) ∈ set_of (λ p, p.snd = 2 * (p.fst)^2)) :
  (0, f) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_l170_170183


namespace alex_score_correct_l170_170727

-- Conditions of the problem
def num_students := 20
def average_first_19 := 78
def new_average := 79

-- Alex's score calculation
def alex_score : ℕ :=
  let total_score_first_19 := 19 * average_first_19
  let total_score_all := num_students * new_average
  total_score_all - total_score_first_19

-- Problem statement: Prove Alex's score is 98
theorem alex_score_correct : alex_score = 98 := by
  sorry

end alex_score_correct_l170_170727


namespace sum_first_n_terms_l170_170220

-- Define the geometric sequence with given conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 > 0 ∧
  a 2 > 0 ∧
  2 * a 1 + 3 * a 2 = 1 ∧
  (a 3) ^ 2 = 9 * a 2 * a 6

-- Define the sequence b_n
def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n+1)).sum (λ i, Real.logBase 3 (a (i+1)))

-- Define the series s_n to prove
def s (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n+1)).sum (λ i, 1 / (b a (i+1)))

-- Final theorem statement
theorem sum_first_n_terms {a : ℕ → ℝ}
  (h : geometric_sequence a) (n : ℕ) : s a n = -((2 : ℝ) * n) / (n + 1) :=
sorry

end sum_first_n_terms_l170_170220


namespace part1_part2_l170_170594

noncomputable def α := arbitrary ℝ -- Assume α is some real number

-- Define the given condition
axiom h1 : (sin α - 2 * cos α) / (sin α + 2 * cos α) = 3

-- From the solution: sin α = -4 * cos α
axiom h2 : sin α = -4 * cos α

-- Statement 1
theorem part1 : (sin α + 2 * cos α) / (5 * cos α - sin α) = -2 / 9 := by
  sorry

-- Statement 2
theorem part2 : (sin α + cos α) ^ 2 = 9 / 17 := by
  sorry

end part1_part2_l170_170594


namespace segment_ratio_l170_170365

/- 
Line segment \overline{AB} is extended past B to point P such that AP:PB = 7:5. 
Determine the constants t and u such that 
\overrightarrow{P} = t \overrightarrow{A} + u \overrightarrow{B}.
The ratio AP:PB = 7:5 leads to the constants t = 5/12 and u = 7/12.
-/
theorem segment_ratio (A B P : V) (AP_BP : 7 * (P - B) = 5 * (A - P)) :
  ∃ t u : ℚ, (t = 5/12 ∧ u = 7/12) ∧
  (P = t • A + u • B) :=
sorry

end segment_ratio_l170_170365


namespace bad_carrots_count_l170_170372

theorem bad_carrots_count (n_t_carrots : ℕ) (m_t_carrots : ℕ) (good_carrots : ℕ) (total_carrots := n_t_carrots + m_t_carrots) :
  n_t_carrots = 38 → m_t_carrots = 47 → good_carrots = 71 → total_carrots - good_carrots = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  simp only
  linarith

end bad_carrots_count_l170_170372


namespace line_passing_through_first_and_third_quadrants_l170_170298

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170298


namespace rhombus_area_proof_l170_170308

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def diagonal1 : ℝ :=
  distance (0, 3.5) (0, -3.5)

def diagonal2 : ℝ :=
  distance (-8, 0) (8, 0)

def area_of_rhombus : ℝ :=
  (diagonal1 * diagonal2) / 2

theorem rhombus_area_proof : area_of_rhombus = 56 :=
by
  -- Define the points
  let p1 := (0, 3.5)
  let p2 := (0, -3.5)
  let p3 := (-8, 0)
  let p4 := (8, 0)
  
  -- Calculate the diagonals
  have d1 : distance p1 p2 = 7 := by sorry
  have d2 : distance p3 p4 = 16 := by sorry
  
  -- Calculate the area
  have area_calc : area_of_rhombus = (7 * 16) / 2 := by sorry
  
  -- Conclude the proof
  have final_area : area_of_rhombus = 56 := by
    rw [area_calc]
    norm_num
    
  exact final_area

end rhombus_area_proof_l170_170308


namespace sqrt_50_between_consecutive_integers_product_l170_170833

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170833


namespace sequence_general_formula_l170_170749

theorem sequence_general_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (a 5 = 16) → ∀ n : ℕ, n > 0 → a n = 2^(n-1) :=
by
  intros h n hn
  sorry

end sequence_general_formula_l170_170749


namespace find_sum_of_m_and_k_l170_170424

theorem find_sum_of_m_and_k
  (d m k : ℤ)
  (h : (9 * d^2 - 5 * d + m) * (4 * d^2 + k * d - 6) = 36 * d^4 + 11 * d^3 - 59 * d^2 + 10 * d + 12) :
  m + k = -7 :=
by sorry

end find_sum_of_m_and_k_l170_170424


namespace difference_of_squares_example_l170_170083

theorem difference_of_squares_example : 169^2 - 168^2 = 337 :=
by
  -- The proof steps using the difference of squares formula is omitted here.
  sorry

end difference_of_squares_example_l170_170083


namespace sqrt_50_product_consecutive_integers_l170_170808

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170808


namespace sum_of_roots_cubic_equation_l170_170053

theorem sum_of_roots_cubic_equation :
  let roots := multiset.to_finset (multiset.filter (λ r, r ≠ 0) (RootSet (6 * (X ^ 3) + 7 * (X ^ 2) + (-12) * X) ℤ))
  (roots.sum : ℤ) / (roots.card : ℤ) = -117 / 100 := sorry

end sum_of_roots_cubic_equation_l170_170053


namespace reachability_in_new_nation_l170_170496

theorem reachability_in_new_nation
  (original_cities : Finset ℕ)
  (new_nation_cities : Finset ℕ)
  (num_original_cities : original_cities.card = 1001)
  (num_new_nation_cities : new_nation_cities.card = 668)
  (roads : ℕ → ℕ → Prop)
  (h_roads_symm : ∀ {a b}, roads a b ↔ roads b a)
  (h_incoming_outgoing : ∀ x ∈ original_cities, (∃ s ∈ Finset.powersetLen 500 original_cities, ∀ y, y ∈ s ↔ roads x y)) :
  (∀ x y ∈ new_nation_cities, ∃ path : List ℕ, path.head = x ∧ path.last = y ∧ ∀ i, (path.nth i) ∈ new_nation_cities ∧ roads (path.nth i) (path.nth (i + 1)))
 :=
sorry

end reachability_in_new_nation_l170_170496


namespace distance_between_trees_l170_170892

-- Define the conditions
def yard_length : ℝ := 325
def number_of_trees : ℝ := 26
def number_of_intervals : ℝ := number_of_trees - 1

-- Define what we need to prove
theorem distance_between_trees:
  (yard_length / number_of_intervals) = 13 := 
  sorry

end distance_between_trees_l170_170892


namespace joe_and_dad_total_marshmallows_roasted_l170_170684

theorem joe_and_dad_total_marshmallows_roasted :
  (let dads_marshmallows := 21
       dads_roasted := dads_marshmallows / 3
       joes_marshmallows := 4 * dads_marshmallows
       joes_roasted := joes_marshmallows / 2
   in dads_roasted + joes_roasted = 49) :=
by
  let dads_marshmallows := 21
  let dads_roasted := dads_marshmallows / 3
  let joes_marshmallows := 4 * dads_marshmallows
  let joes_roasted := joes_marshmallows / 2
  show dads_roasted + joes_roasted = 49 from sorry

end joe_and_dad_total_marshmallows_roasted_l170_170684


namespace min_sum_mul_l170_170422

def is_permutation_of {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1 ~ l2

theorem min_sum_mul (a b c x y z : ℕ) (h_perm : is_permutation_of [a, b, c, x, y, z] [1, 2, 3, 4, 5, 6]) :
  a * b * c + x * y * z = 56 :=
by { sorry }

end min_sum_mul_l170_170422


namespace zara_goats_l170_170070

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l170_170070


namespace boat_trip_distance_l170_170897

-- Defining the conditions given in the problem
def downstream_speed (boat_speed stream_velocity : ℝ) : ℝ := boat_speed + stream_velocity
def upstream_speed (boat_speed stream_velocity : ℝ) : ℝ := boat_speed - stream_velocity

def total_time (distance : ℝ) (downstream_speed upstream_speed : ℝ) : ℝ :=
  (distance / downstream_speed) + ((distance / 2) / upstream_speed)

def distance_between_A_and_B := 122.14

-- Main theorem statement
theorem boat_trip_distance 
  (boat_speed stream_velocity : ℝ) (total_travel_time : ℝ) :
  boat_speed = 14 → stream_velocity = 4 → total_travel_time = 19 →
  total_time distance_between_A_and_B (downstream_speed boat_speed stream_velocity) (upstream_speed boat_speed stream_velocity) = total_travel_time :=
by sorry

end boat_trip_distance_l170_170897


namespace total_toys_is_60_l170_170328

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l170_170328


namespace total_homework_problems_l170_170127

-- Define the conditions as Lean facts
def finished_problems : ℕ := 45
def ratio_finished_to_left := (9, 4)
def problems_left (L : ℕ) := finished_problems * ratio_finished_to_left.2 = L * ratio_finished_to_left.1 

-- State the theorem
theorem total_homework_problems (L : ℕ) (h : problems_left L) : finished_problems + L = 65 :=
sorry

end total_homework_problems_l170_170127


namespace num_squares_7x7_l170_170249

-- Definition of the problem conditions and expected result
def num_noncongruent_squares (n : ℕ) : ℕ :=
  (n - 1 + 1)^2 + (n - 2 + 1)^2 + (n - 3 + 1)^2 + (n - 4 + 1)^2 + (n - 5 + 1)^2 + (n - 6 + 1)^2 + 
  (n - 1)^2 + (n - 2)^2

theorem num_squares_7x7 : num_noncongruent_squares 7 = 200 := 
by {
  have h1 : 49 = (7 - 1 + 1)^2 := by norm_num,
  have h2 : 36 = (7 - 2 + 1)^2 := by norm_num,
  have h3 : 25 = (7 - 3 + 1)^2 := by norm_num,
  have h4 : 16 = (7 - 4 + 1)^2 := by norm_num,
  have h5 : 9 = (7 - 5 + 1)^2 := by norm_num,
  have h6 : 4 = (7 - 6 + 1)^2 := by norm_num,
  have h7 : 36 = (7 - 1)^2 := by norm_num,
  have h8 : 25 = (7 - 2)^2 := by norm_num,
  calc
    num_noncongruent_squares 7 
        = 49 + 36 + 25 + 16 + 9 + 4 + 36 + 25 : by simp [h1, h2, h3, h4, h5, h6, h7, h8]
    ... = 200 : by norm_num 
}

end num_squares_7x7_l170_170249


namespace at_least_one_third_positive_l170_170853

theorem at_least_one_third_positive (nums : Fin 2016 → ℤ) (h_nonzero : ∀ i, nums i ≠ 0) : 
  ∃ p : ℚ, p ≥ 1/3 ∧ ∃ (S : Finset (Fin (2016*(2016-1)/2))), S.card = (2016 * (2016 - 1)) / 2 ∧ 
    p = (S.filter (λ x, (nums x.fst) * (nums x.snd) > 0)).card / S.card :=
by sorry

end at_least_one_third_positive_l170_170853


namespace product_of_consecutive_integers_sqrt_50_l170_170782

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170782


namespace base_seven_product_and_sum_of_digits_l170_170139

theorem base_seven_product_and_sum_of_digits :
  let base_seven_mul (a b : Nat) : Nat :=
        let a₁ := a / 7;
        let a₀ := a % 7;
        let a_base10 := a₁ * 7 + a₀;

        let b₁ := b / 7;
        let b₀ := b % 7;
        let b_base10 := b₁ * 7 + b₀;

        let product_base10 := a_base10 * b_base10;
        
        let product₃ := product_base10 / (7 * 7);
        let r₁ := product_base10 % (7 * 7);
        let product₂ := r₁ / 7;
        let product₁ := r₁ % 7;

        let digits_sum := product₃ + product₂ + product₁;

        let final_sum_quot := digits_sum / 7;
        let final_sum_rem := digits_sum % 7;
        
        final_sum_quot * 10 + final_sum_rem
  in
  base_seven_mul 35 42 = 21 :=
by
  sorry

end base_seven_product_and_sum_of_digits_l170_170139


namespace largest_c_divisor_property_l170_170974

open Nat

-- Define the number of divisors function
def tau (n : ℕ) : ℕ :=
  (divisors n).card

theorem largest_c_divisor_property :
  ∃ c > 0, (∀ n ≥ 2, ∃ d ∣ n, d ≤ sqrt n ∧ tau d ≥ c * sqrt (tau n)) ∧ c = 1 / sqrt 2 :=
by
  sorry

end largest_c_divisor_property_l170_170974


namespace sqrt_50_product_consecutive_integers_l170_170810

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170810


namespace binomial_square_odd_squares_diff_div_by_8_even_squares_diff_div_by_4_even_squares_diff_not_div_by_8_l170_170381

-- Task 1: Proof for multiplication formula of binomial squares
theorem binomial_square (a b : ℤ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 :=
by sorry

-- Task 2: Proof for difference of squares of any two odd numbers
theorem odd_squares_diff_div_by_8 (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) :
  let a := 2 * m + 1,
      b := 2 * n + 1 in
  (a ^ 2 - b ^ 2) % 8 = 0 :=
by sorry

-- Task 3: Proof for difference of squares of two consecutive even numbers
theorem even_squares_diff_div_by_4 (n : ℤ) :
  let a := 2 * n,
      b := 2 * n + 2 in
  (b ^ 2 - a ^ 2) % 4 = 0 :=
by sorry

theorem even_squares_diff_not_div_by_8 (n : ℤ) :
  let a := 2 * n,
      b := 2 * n + 2 in
  ¬((b ^ 2 - a ^ 2) % 8 = 0) :=
by sorry

end binomial_square_odd_squares_diff_div_by_8_even_squares_diff_div_by_4_even_squares_diff_not_div_by_8_l170_170381


namespace sufficient_condition_for_odd_l170_170477

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (Real.sqrt (x^2 + a^2) - x)

theorem sufficient_condition_for_odd (a : ℝ) :
  (∀ x : ℝ, f 1 (-x) = -f 1 x) ∧
  (∀ x : ℝ, f (-1) (-x) = -f (-1) x) → 
  (a = 1 → ∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (a ≠ 1 → ∃ x : ℝ, f a (-x) ≠ -f a x) :=
by
  sorry

end sufficient_condition_for_odd_l170_170477


namespace line_passing_through_first_and_third_quadrants_l170_170294

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170294


namespace sqrt_50_between_7_and_8_l170_170800

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170800


namespace minute_hand_catch_up_l170_170935

theorem minute_hand_catch_up (t : ℚ) 
    (h1 : ∀ (t : ℚ), 0 ≤ t) 
    (h2 : (minute_speed : ℚ) := 6) 
    (h3 : (hour_speed : ℚ) := 0.5) 
    (h4 : (initial_hour_pos : ℚ) := 240) :
    (minute_speed * t = initial_hour_pos + hour_speed * t) → 
    t = 43 + 7 / 11 :=
begin
  sorry
end

end minute_hand_catch_up_l170_170935


namespace rectangle_length_l170_170880

theorem rectangle_length :
  ∀ (side : ℕ) (width : ℕ) (length : ℕ), 
  side = 4 → 
  width = 8 → 
  side * side = width * length → 
  length = 2 := 
by
  -- sorry to skip the proof
  intros side width length h1 h2 h3
  sorry

end rectangle_length_l170_170880


namespace percentage_decrease_in_area_l170_170261

theorem percentage_decrease_in_area {r : ℝ} (h : r > 0) :
  let r' := 0.8 * r,
      A := π * r^2,
      A' := π * (0.8 * r)^2
  in (A - A') / A * 100 = 36 :=
by
  let r' := 0.8 * r
  let A := π * r^2
  let A' := π * (0.8 * r)^2
  have h₁ : A = π * r^2 := rfl
  have h₂ : A' = π * (0.8 * r)^2 := rfl
  have h₃ : A' = π * 0.64 * r^2 := by rw [←(mul_assoc π 0.64 (r^2))]
  have h₄ : (A - A') / A * 100 = ((π * r^2 - π * 0.64 * r^2) / π * r^2) * 100 :=
    by rw [←(h₃), h₂, h₁]
  have h₅ : ((π * r^2 - π * 0.64 * r^2) / π * r^2) * 100 = ((π * r^2 * (1 - 0.64)) / π * r^2) * 100 :=
    by rw [←(sub_mul (1 : ℝ) 0.64 (π * r^2))]
  have h₆ : ((π * r^2 * (1 - 0.64)) / π * r^2) * 100 = (1 - 0.64) * 100 :=
    by rw [mul_div_cancel_left _ (ne_of_gt (mul_pos (pi_pos) (pow_pos h 2)))]
  have h₇ : (1 - 0.64) * 100 = 36 := by norm_num
  exact h₇

end percentage_decrease_in_area_l170_170261


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170030

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170030


namespace cutCornerIntoEqualParts_l170_170103

-- Define the structure representing a corner made up of three squares
structure Corner :=
  (squares : Fin 3 → Square)
  (arrangement : IsLShape squares)

-- Define a function that takes a corner and returns a set of pieces after cutting
noncomputable def cutCorner (corn: Corner) (n : ℕ) : Set (Fin n → Piece)
  := sorry

-- Define the problem statement
theorem cutCornerIntoEqualParts (corn: Corner)
  (h2 : ∃ pieces, pieces ∈ (cutCorner corn 2) ∧ (∀ p1 p2 ∈ pieces, Area p1 = Area p2 ∧ Shape p1 ≅ Shape p2))
  (h3 : ∃ pieces, pieces ∈ (cutCorner corn 3) ∧ (∀ p1 p2 ∈ pieces, Area p1 = Area p2 ∧ Shape p1 ≅ Shape p2))
  (h4 : ∃ pieces, pieces ∈ (cutCorner corn 4) ∧ (∀ p1 p2 ∈ pieces, Area p1 = Area p2 ∧ Shape p1 ≅ Shape p2)) :
  True := sorry

end cutCornerIntoEqualParts_l170_170103


namespace possible_values_of_k_l170_170273

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170273


namespace student_ticket_cost_l170_170395

def general_admission_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 525
def total_revenue : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

def number_of_student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
def revenue_from_general_admission : ℕ := general_admission_tickets_sold * general_admission_ticket_cost

theorem student_ticket_cost : ∃ S : ℕ, number_of_student_tickets_sold * S + revenue_from_general_admission = total_revenue ∧ S = 4 :=
by
  sorry

end student_ticket_cost_l170_170395


namespace product_of_integers_around_sqrt_50_l170_170776

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170776


namespace complement_union_l170_170240

open Set

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ)
  (hA : A = { x : ℝ | x^2 - 3 * x < 4 })
  (hB : B = { x : ℝ | |x| ≥ 2 }) :
  (compl B ∪ A) = Ioo (-2 : ℝ) 4 :=
by
  -- We state that complement and union is as required.
  sorry

end complement_union_l170_170240


namespace coefficient_x2y7_l170_170408

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (x + y)(x - y)^8
def expr (x y : ℤ) := (x + y) * (finset.range 9).sum (λ k, binomial 8 k * x^(8 - k) * (-y)^k)

-- Define the coefficient of x^2y^7 in the expansion of (x + y)(x - y)^8
def coefficient (x y : ℤ) := (finset.range 9).sum (λ k, if 8 - k = 2 ∧ k = 7 then binomial 8 k * (-1)^k else 0)

theorem coefficient_x2y7 (x y : ℤ) : coefficient x y = 20 :=
by {
    sorry
}

end coefficient_x2y7_l170_170408


namespace cos_neg_570_eq_neg_sqrt3_div_2_l170_170191

theorem cos_neg_570_eq_neg_sqrt3_div_2 :
  Real.cos (-(570 : ℝ) * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_neg_570_eq_neg_sqrt3_div_2_l170_170191


namespace stationery_shop_costs_l170_170126

theorem stationery_shop_costs (p n : ℝ) 
  (h1 : 9 * p + 6 * n = 3.21)
  (h2 : 8 * p + 5 * n = 2.84) :
  12 * p + 9 * n = 4.32 :=
sorry

end stationery_shop_costs_l170_170126


namespace mountain_height_third_path_end_l170_170875

-- Define the conditions of the problem as variables and hypotheses
def truncated_square_pyramid (base_edge: ℝ) (upper_base_edge: ℝ) :=
  -- Regular truncated square pyramid
  true

noncomputable def angle_of_first_path : ℝ := 16 -- degrees
noncomputable def angle_of_second_path : ℝ := angle_of_first_path - 7 -- degrees

-- Given base and upper base edge lengths
def lower_base_edge := 320
def upper_base_edge := 135

-- Let h be the height of the mountain
variable (h : ℝ)

-- Define the incline angles for the paths
variable (θ_1 θ_2 : ℝ)
hypothesis hθ_1 : θ_1 = angle_of_first_path
hypothesis hθ_2 : θ_2 = angle_of_second_path

-- Hypothesis parameters for projections and distances
variable (KV' KT' T'U' U'V' : ℝ)
hypothesis h_KV' : KV' = 160 * sqrt 2
hypothesis h_KT' : KT' = 200 * sqrt 2
hypothesis h_T'U' : T'U' = 150 * sqrt 2
hypothesis h_U'V' : U'V' = 112.5 * sqrt 2

-- Total projection length A
noncomputable def A : ℝ := 462.5 * sqrt 2

-- Prove that the height of the mountain is 130.5 meters
theorem mountain_height :
  (truncated_square_pyramid lower_base_edge upper_base_edge) →
  h = 130.5 := sorry

-- Path segments
variable (mid_east_mid_south : ℝ)
hypothesis h_mid_east_mid_south : 
  mid_east_mid_south = lower_base_edge / 2

-- Prove that the end point of the third path is the midpoint of the southern side of the plateau
theorem third_path_end :
  true →
  mid_east_mid_south = 320 / 2 := sorry

end mountain_height_third_path_end_l170_170875


namespace sum_of_angles_l170_170475

theorem sum_of_angles (p q r s t u v w x y : ℝ)
  (H1 : p + r + t + v + x = 360)
  (H2 : q + s + u + w + y = 360) :
  p + q + r + s + t + u + v + w + x + y = 720 := 
by sorry

end sum_of_angles_l170_170475


namespace chess_club_members_l170_170756

theorem chess_club_members {n : ℤ} (h10 : n % 10 = 6) (h11 : n % 11 = 6) (rng : 300 ≤ n ∧ n ≤ 400) : n = 336 :=
  sorry

end chess_club_members_l170_170756


namespace point_on_xaxis_equidistant_l170_170508

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l170_170508


namespace unique_solution_a_eq_e_one_div_e_l170_170970

theorem unique_solution_a_eq_e_one_div_e : 
  ∀ a : ℝ, a > 1 → (∃! x : ℝ, a^x = real.log x / real.log x) ↔ a = real.exp (1 / real.exp 1) :=
by sorry

end unique_solution_a_eq_e_one_div_e_l170_170970


namespace meiosis_and_fertilization_outcome_l170_170961

-- Definitions corresponding to the conditions:
def increases_probability_of_genetic_mutations (x : Type) := 
  ∃ (p : x), false -- Placeholder for the actual mutation rate being low

def inherits_all_genetic_material (x : Type) :=
  ∀ (p : x), false -- Parents do not pass all genes to offspring

def receives_exactly_same_genetic_information (x : Type) :=
  ∀ (p : x), false -- Offspring do not receive exact genetic information from either parent

def produces_genetic_combination_different (x : Type) :=
  ∃ (o : x), true -- The offspring has different genetic information from either parent

-- The main statement to be proven:
theorem meiosis_and_fertilization_outcome (x : Type) 
  (cond1 : ¬ increases_probability_of_genetic_mutations x)
  (cond2 : ¬ inherits_all_genetic_material x)
  (cond3 : ¬ receives_exactly_same_genetic_information x) :
  produces_genetic_combination_different x :=
sorry

end meiosis_and_fertilization_outcome_l170_170961


namespace distinct_license_plates_l170_170112

theorem distinct_license_plates (d : Fin 10 → Fin 10) 
(letters : Fin 21 × Fin 21) 
(cond : letters.fst ≠ letters.snd) 
(block_pos : Fin 6) : 
∃ N : Nat, N = 2,520,000,000 :=
by 
  sorry

end distinct_license_plates_l170_170112


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170036

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170036


namespace dot_product_solution_l170_170639

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)
variables (ha : ‖a‖ = 4) (hb : ‖b‖ = 7) (hc : ‖c‖ = 5)

theorem dot_product_solution :
  (a + b + c) ⬝ (a - b - c) = -58 :=
sorry

end dot_product_solution_l170_170639


namespace print_time_l170_170511

theorem print_time (rate : ℕ) (total_pages : ℕ) : rate = 24 → total_pages = 360 → total_pages / rate = 15 :=
by 
  intros 
  assume h_rate : rate = 24 
  assume h_total_pages : total_pages = 360 
  rw [h_rate, h_total_pages] 
  norm_num
  sorry

end print_time_l170_170511


namespace f_specification_l170_170152

open Function

def f : ℕ → ℕ := sorry -- define function f here

axiom f_involution (n : ℕ) : f (f n) = n

axiom f_functional_property (n : ℕ) : f (f n + 1) = if n % 2 = 0 then n - 1 else n + 3

axiom f_bijective : Bijective f

axiom f_not_two (n : ℕ) : f (f n + 1) ≠ 2

axiom f_one_eq_two : f 1 = 2

theorem f_specification (n : ℕ) : 
  f n = if n % 2 = 1 then n + 1 else n - 1 :=
sorry

end f_specification_l170_170152


namespace sum_invariant_under_permutation_l170_170703

theorem sum_invariant_under_permutation (b : List ℝ) (σ : List ℕ) (hσ : σ.Perm (List.range b.length)) :
  (List.sum b) = (List.sum (σ.map (b.get!))) := by
  sorry

end sum_invariant_under_permutation_l170_170703


namespace geometric_sequence_problem_l170_170317

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 * a 7 = 8)
  (h2 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1)):
  a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l170_170317


namespace lana_total_spending_l170_170694

noncomputable def general_admission_cost : ℝ := 6
noncomputable def vip_cost : ℝ := 10
noncomputable def premium_cost : ℝ := 15

noncomputable def num_general_admission_tickets : ℕ := 6
noncomputable def num_vip_tickets : ℕ := 2
noncomputable def num_premium_tickets : ℕ := 1

noncomputable def discount_general_admission : ℝ := 0.10
noncomputable def discount_vip : ℝ := 0.15

noncomputable def total_spending (gen_cost : ℝ) (vip_cost : ℝ) (prem_cost : ℝ) (gen_num : ℕ) (vip_num : ℕ) (prem_num : ℕ) (gen_disc : ℝ) (vip_disc : ℝ) : ℝ :=
  let general_cost := gen_cost * gen_num
  let general_discount := general_cost * gen_disc
  let discounted_general_cost := general_cost - general_discount
  let vip_cost_total := vip_cost * vip_num
  let vip_discount := vip_cost_total * vip_disc
  let discounted_vip_cost := vip_cost_total - vip_discount
  let premium_cost_total := prem_cost * prem_num
  discounted_general_cost + discounted_vip_cost + premium_cost_total

theorem lana_total_spending : total_spending general_admission_cost vip_cost premium_cost num_general_admission_tickets num_vip_tickets num_premium_tickets discount_general_admission discount_vip = 64.40 := 
sorry

end lana_total_spending_l170_170694


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170020

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170020


namespace incorrect_counting_of_students_l170_170653

open Set

theorem incorrect_counting_of_students
  (total_students : ℕ)
  (english_only : ℕ)
  (german_only : ℕ)
  (french_only : ℕ)
  (english_german : ℕ)
  (english_french : ℕ)
  (german_french : ℕ)
  (all_three : ℕ)
  (reported_total : ℕ)
  (h_total_students : total_students = 100)
  (h_english_only : english_only = 30)
  (h_german_only : german_only = 23)
  (h_french_only : french_only = 50)
  (h_english_german : english_german = 10)
  (h_english_french : english_french = 8)
  (h_german_french : german_french = 20)
  (h_all_three : all_three = 5)
  (h_reported_total : reported_total = 100) :
  (english_only + german_only + french_only + english_german +
   english_french + german_french - 2 * all_three) ≠ reported_total :=
by
  sorry

end incorrect_counting_of_students_l170_170653


namespace point_on_xaxis_equidistant_l170_170507

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l170_170507


namespace sandbox_width_l170_170519

theorem sandbox_width (L A : ℕ) (h₁ : L = 312) (h₂ : A = 45552) : ∃ W : ℕ, W = 146 ∧ L * W = A :=
by
  use 146
  rw [h₁, h₂]
  simp
  norm_num
  sorry

end sandbox_width_l170_170519


namespace triangle_shape_not_determined_by_product_of_two_sides_and_angle_l170_170871

theorem triangle_shape_not_determined_by_product_of_two_sides_and_angle (T : Type) [triangle T] :
  (∃ a b c: ℝ, ∀ (s1 s2 : ℝ), s1 * s2 * (sin c) = a * b * (sin c)) → ¬(unique_shape T a b c) :=
by 
  sorry

end triangle_shape_not_determined_by_product_of_two_sides_and_angle_l170_170871


namespace product_of_consecutive_integers_between_sqrt_50_l170_170831

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170831


namespace committee_count_l170_170591

theorem committee_count (total_students : ℕ) (include_students : ℕ) (choose_students : ℕ) :
  total_students = 8 → include_students = 2 → choose_students = 3 →
  Nat.choose (total_students - include_students) choose_students = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end committee_count_l170_170591


namespace coin_combinations_30_cents_l170_170246

theorem coin_combinations_30_cents : 
  let value_penny := 1
      value_nickel := 5
      value_dime := 10
  in 
  ∃ (p k d : ℕ), (p * value_penny + k * value_nickel + d * value_dime = 30 ∧ 
        p + k + d = 22) :=
by {
  sorry
}

end coin_combinations_30_cents_l170_170246


namespace possible_values_of_k_l170_170270

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l170_170270


namespace measure_angle_CAD_is_15_degrees_l170_170174

open EuclideanGeometry

-- Define the equilateral triangle and square in coplanar geometry
variable (A B C D E : Point)
variable [euclidean_space ℝ Point]
variable (h_triangle : EquilateralTriangle A B C)
variable (h_square : Square B C D E)
variable (h_coplanar : Coplanar {A, B, C, D, E})

-- Prove that the angle CAD is 15 degrees
theorem measure_angle_CAD_is_15_degrees 
  (h_triangle : EquilateralTriangle A B C)
  (h_square : Square B C D E)
  (h_coplanar : Coplanar {A, B, C, D, E}) :
  angle A C D = 15 := 
  sorry

end measure_angle_CAD_is_15_degrees_l170_170174


namespace trajectory_of_M_equation_of_l_area_of_POM_l170_170606

-- Definitions from the conditions
def P : ℝ × ℝ := (2, 2)
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0
def O : ℝ × ℝ := (0, 0)

-- Statements to be proven
theorem trajectory_of_M :
  ∀ (M : ℝ × ℝ), (circle_C M.1 M.2) ∧ 
  (∀ (A B : ℝ × ℝ), ((A.1 - 2)^2 + (A.2 - 2)^2 = 16) ∧ ((B.1 - 2)^2 + (B.2 - 2)^2 = 16) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) 
  → (M.1 - 1) ^ 2 + (M.2 - 3) ^ 2 = 2 := 
sorry

theorem equation_of_l_area_of_POM :
  ∀ (l : ℝ × ℝ → Prop) (M : ℝ × ℝ),
  (| (2:ℝ) - O.2 | = | M.2 - O.2 |) ∧
  (∀ (A B : ℝ × ℝ), l A ∧ l B ∧ (circle_C A.1 A.2) ∧ (circle_C B.1 B.2) ∧ 
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) 
  → (∀ x y, l (x, y) ↔ x + 3 * y - 8 = 0) ∧ (triangle_area P O M = 16 / 5) := 
sorry

end trajectory_of_M_equation_of_l_area_of_POM_l170_170606


namespace algebra_students_count_l170_170540

theorem algebra_students_count {total_students geometry_students both_students: ℕ}
  (h_total: total_students = 15)
  (h_geometry: geometry_students = 9)
  (h_both: both_students = 4)
  (h_union: total_students = geometry_students + (total_students - both_students)) :
  (total_students - geometry_students + both_students) = 10 := 
by
  rw [h_total, h_geometry, h_both]
  sorry

end algebra_students_count_l170_170540


namespace hashN_of_25_l170_170952

def hashN (N : ℝ) : ℝ := 0.6 * N + 2

theorem hashN_of_25 : hashN (hashN (hashN (hashN 25))) = 7.592 :=
by
  sorry

end hashN_of_25_l170_170952


namespace cone_curved_surface_area_l170_170882

def radius (r : ℝ) := r = 3
def slantHeight (l : ℝ) := l = 15
def curvedSurfaceArea (csa : ℝ) := csa = 45 * Real.pi

theorem cone_curved_surface_area 
  (r l csa : ℝ) 
  (hr : radius r) 
  (hl : slantHeight l) 
  : curvedSurfaceArea (Real.pi * r * l) 
  := by
  unfold radius at hr
  unfold slantHeight at hl
  unfold curvedSurfaceArea
  rw [hr, hl]
  norm_num
  sorry

end cone_curved_surface_area_l170_170882


namespace problem_fixed_values_problem_arbitrary_values_l170_170358

noncomputable def minimal_value_fixed (x y z : ℝ) (m n p: ℝ) :=
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem problem_fixed_values (x y z m n p : ℝ) 
  (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) 
  (xyz_eq_8 : x * y * z = 8) (mnp_eq_8 : m * n * p = 8)
  (m_eq_2 : m = 2) (n_eq_2 : n = 2) (p_eq_2 : p = 2) :
  minimal_value_fixed x y z m n p = 36 :=
sorry

theorem problem_arbitrary_values (x y z m n p : ℝ) 
  (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) 
  (xyz_eq_8 : x * y * z = 8) (mnp_eq_8 : m * n * p = 8) :
  minimal_value_fixed x y z m n p = 
  6 * real.cbrt 2 * (real.cbrt (m^2) + real.cbrt (n^2) + real.cbrt (p^2)) :=
sorry

end problem_fixed_values_problem_arbitrary_values_l170_170358


namespace find_Jerrys_age_l170_170370

variables (J : ℕ)

def Mickeys_age_is_six_years_less_than_200_percent_of_Jerrys_age := 
  ∀ M, M = 2 * J - 6

def Mickeys_age_is_sixteen := 16

theorem find_Jerrys_age 
  (hm := Mickeys_age_is_six_years_less_than_200_percent_of_Jerrys_age J)
  (hM := Mickeys_age_is_sixteen) : 
  ∃ J, J = 11 :=
by 
  let M := 16
  have h1 : M = 2 * J - 6 := hm M
  have h2 : M = 16 := hM
  rw [h2] at h1
  sorry

end find_Jerrys_age_l170_170370


namespace symmetric_center_translated_cosine_l170_170751

noncomputable def translated_cosine_function (x : ℝ) : ℝ :=
  Real.cos (2 * x + 7 * Real.pi / 12)

theorem symmetric_center_translated_cosine :
  ∃ k ∈ Set.Ioo (-(1:ℝ/2) : ℝ) (1 / 2), 2 * (k * Real.pi / 2 - Real.pi / 24) + 7 * Real.pi / 12 = k * Real.pi + Real.pi / 2 :=
begin
  use 1,
  split,
  linarith,
  linarith,
  sorry
end

end symmetric_center_translated_cosine_l170_170751


namespace problem_proposition_correctness_l170_170623

theorem problem_proposition_correctness :
  let p := false
  let q := false
  let Sₙ (n : ℕ) : ℝ := (n * (n + 1)) / 2
  let aₙ (n : ℕ) : ℝ := if n = 1 then 1 else 1
  (¬(p ∧ q) → ¬p ∨ ¬q) ∧
  (Sₙ 10 / 10 = (Sₙ 100) / 100) ∧
  (¬(∀ x : ℝ, x^2 + 1 ≥ 1) = (∃ x : ℝ, x^2 + 1 < 1)) ∧
  (∀ (A B : ℝ), A > B ↔ sin A > sin B) →
  2 = 2 :=
by
  sorry

end problem_proposition_correctness_l170_170623


namespace triangle_to_square_ratio_l170_170313

variable {Point : Type}
variables (A B C D M N : Point)
variables {dist : Point → Point → ℝ}
variables {area : Point → Point → Point → ℝ}

noncomputable def square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧
  dist B C = dist C D ∧
  dist C D = dist D A ∧
  dist A C = dist B D

noncomputable def midpoint (P Q M : Point) : Prop :=
  dist P M = dist M Q

theorem triangle_to_square_ratio
  (h_sq : square A B C D)
  (h_mid_M : midpoint D A M)
  (h_mid_N : midpoint B C N) :
  area D M N / (dist A B * dist A B) = 1 / 4 :=
sorry

end triangle_to_square_ratio_l170_170313


namespace inequality_f2n_l170_170204

noncomputable def f (n : ℕ) : ℝ :=
  if n = 0 then 0 else (∑ i in Finset.range (n + 1) \ {0}, (1 : ℝ) / (i : ℝ))

theorem inequality_f2n (n : ℕ) (hn : 2 ≤ n) : f (2^n) > (n + 2) / 2 :=
by
  sorry

end inequality_f2n_l170_170204


namespace number_of_permutations_conditioned_l170_170341

theorem number_of_permutations_conditioned :
  let a : Fin 15 → ℕ := λ i, i + 1
  let permutations := List.permutations (List.range' 1 15)
  let valid_permutations := 
        permutations.filter 
          (λ l, (l.take 7).sorted (· > ·) ∧ (l.drop 6).sorted (· < ·))
  valid_permutations.length = 3003 := 
by
  sorry

end number_of_permutations_conditioned_l170_170341


namespace boat_trip_distance_l170_170900

theorem boat_trip_distance
  (total_time : ℕ := 19)
  (stream_velocity : ℕ := 4)
  (boat_still_water_speed : ℕ := 14)
  (distance_between_A_B : ℕ := 180) :
  let downstream_speed := boat_still_water_speed + stream_velocity,
      upstream_speed := boat_still_water_speed - stream_velocity,
      time_downstream := distance_between_A_B / downstream_speed,
      time_upstream := (distance_between_A_B / 2) / upstream_speed
  in time_downstream + time_upstream = total_time →
     distance_between_A_B = 180 := 
by
  intros h
  have downstream_speed := boat_still_water_speed + stream_velocity
  have upstream_speed := boat_still_water_speed - stream_velocity
  have time_downstream := distance_between_A_B / downstream_speed
  have time_upstream := (distance_between_A_B / 2) / upstream_speed
  have h_eq : time_downstream + time_upstream = total_time := h
  exact eq_of_sub_eq_zero ((by simp [time_downstream, time_upstream, downstream_speed, upstream_speed, distance_between_A_B, nat.div_def, total_time]; linarith) : distance_between_A_B - 180 = 0)

end boat_trip_distance_l170_170900


namespace max_wx_plus_xy_plus_yz_max_wx_plus_xy_plus_yz_is_achievable_l170_170707

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0)
  (h_sum : w + x + y + z = 100) :
  wx + xy + yz ≤ 2500 :=
sorry

theorem max_wx_plus_xy_plus_yz_is_achievable :
  ∃ w x y z : ℝ, w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ (w + x + y + z = 100) ∧ (wx + xy + yz = 2500) :=
begin
  let w := 50,
  let x := 50,
  let y := 0,
  let z := 0,
  use [w, x, y, z],
  repeat { split }, 
  repeat { linarith }, 
  linarith,
end

end max_wx_plus_xy_plus_yz_max_wx_plus_xy_plus_yz_is_achievable_l170_170707


namespace response_rate_percentage_50_l170_170491

def questionnaire_response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℕ :=
  (responses_needed * 100) / questionnaires_mailed

theorem response_rate_percentage_50 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 600) : 
  questionnaire_response_rate_percentage responses_needed questionnaires_mailed = 50 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end response_rate_percentage_50_l170_170491


namespace percentage_of_rotten_oranges_l170_170123

theorem percentage_of_rotten_oranges
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (percentage_good_condition : ℕ)
  (rotted_percentage_bananas : ℕ)
  (total_fruits : ℕ)
  (good_condition_fruits : ℕ)
  (rotted_fruits : ℕ)
  (rotted_bananas : ℕ)
  (rotted_oranges : ℕ)
  (percentage_rotten_oranges : ℕ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : percentage_good_condition = 89)
  (h4 : rotted_percentage_bananas = 5)
  (h5 : total_fruits = total_oranges + total_bananas)
  (h6 : good_condition_fruits = percentage_good_condition * total_fruits / 100)
  (h7 : rotted_fruits = total_fruits - good_condition_fruits)
  (h8 : rotted_bananas = rotted_percentage_bananas * total_bananas / 100)
  (h9 : rotted_oranges = rotted_fruits - rotted_bananas)
  (h10 : percentage_rotten_oranges = rotted_oranges * 100 / total_oranges) : 
  percentage_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l170_170123


namespace problem_I_problem_II_l170_170627

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + (4 / x) - m| + m

-- Proof problem (I): When m = 0, find the minimum value of the function f(x).
theorem problem_I : ∀ x : ℝ, (f x 0) ≥ 4 := by
  sorry

-- Proof problem (II): If the function f(x) ≤ 5 for all x ∈ [1, 4], find the range of m.
theorem problem_II (m : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x m ≤ 5) ↔ m ≤ 9 / 2 := by
  sorry

end problem_I_problem_II_l170_170627


namespace find_number_69_3_l170_170746

theorem find_number_69_3 (x : ℝ) (h : (x * 0.004) / 0.03 = 9.237333333333334) : x = 69.3 :=
by
  sorry

end find_number_69_3_l170_170746


namespace select_computers_l170_170389

-- Definition of given conditions
def lenovo_computers := 4
def crsc_computers := 5
def total_selected := 3

-- Main statement to prove
theorem select_computers :
  ∃ n : ℕ, n = 70 ∧ (∃ lenovo_select crsc_select : ℕ,
    lenovo_select + crsc_select = total_selected ∧
    lenovo_select > 0 ∧ crsc_select > 0 ∧ 
    (comb lenovo_computers lenovo_select) * (comb crsc_computers crsc_select) = n) := sorry

end select_computers_l170_170389


namespace max_good_points_acute_triangle_l170_170915

-- Definitions for acute triangles and cevian conditions
def is_acute_triangle {A B C : Type} (triangle : Triangle A B C) : Prop := 
  true -- Definition omitted for brevity

def is_good_point (triangle : Triangle A B C) (P : Point) : Prop := 
  ∃ A₁ B₁ C₁ Aₚ Bₚ Cₚ : Point, 
    altitude A₁ triangle ∧
    altitude B₁ triangle ∧
    altitude C₁ triangle ∧
    cevian Aₚ P A triangle ∧
    cevian Bₚ P B triangle ∧
    cevian Cₚ P C triangle ∧
    (length A Aₚ / length A A₁ = length B Bₚ / length B B₁) ∧ 
    (length B Bₚ / length B B₁ = length C Cₚ / length C C₁)

-- Main theorem statement
theorem max_good_points_acute_triangle {A B C : Type} 
  (triangle : Triangle A B C) (h_acute : is_acute_triangle triangle) : 
  ∃! H : Point, is_good_point triangle H :=
begin
  sorry
end

end max_good_points_acute_triangle_l170_170915


namespace range_of_f_l170_170560

open Set

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (2 * x - 1)

theorem range_of_f : range f = Ici (1 / 2) :=
by
  sorry

end range_of_f_l170_170560


namespace inequality_C_D_l170_170944

variable {α : Type*} [linear_ordered_field α]

def b_k (a : ℕ → α) (k : ℕ) : α :=
  (list.sum (list.map a (list.fin_range k))) / k

def C (a : ℕ → α) (n : ℕ) : α :=
  list.sum (list.map (λ k, (a k - b_k a k) ^ 2) (list.fin_range n))

def D (a : ℕ → α) (n : ℕ) : α :=
  list.sum (list.map (λ k, (a k - b_k a n) ^ 2) (list.fin_range n))

theorem inequality_C_D (a : ℕ → α) (n : ℕ) :
  C a n ≤ D a n ∧ D a n ≤ 2 * C a n := by
  sorry

end inequality_C_D_l170_170944


namespace necessary_and_sufficient_condition_for_equal_edges_l170_170432

noncomputable def tetrahedron_conditions :=
  {a a' b b' c c' R : ℝ // 
  16 * R^2 = a^2 + a'^2 + b^2 + b'^2 + c^2 + c'^2}

theorem necessary_and_sufficient_condition_for_equal_edges
  (a a' b b' c c' R : ℝ)
  (h : 16 * R^2 = a^2 + a'^2 + b^2 + b'^2 + c^2 + c'^2) :
  (a = a' ∧ b = b' ∧ c = c') ↔ (16 * R^2 = a^2 + a'^2 + b^2 + b'^2 + c^2 + c'^2) :=
sorry

end necessary_and_sufficient_condition_for_equal_edges_l170_170432


namespace product_of_consecutive_integers_sqrt_50_l170_170781

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170781


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170011

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170011


namespace line_passing_through_first_and_third_quadrants_l170_170297

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170297


namespace sqrt_50_between_7_and_8_l170_170801

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170801


namespace product_of_consecutive_integers_sqrt_50_l170_170819

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170819


namespace pencils_in_each_box_l170_170135

theorem pencils_in_each_box (n : ℕ) (h : 10 * n - 10 = 40) : n = 5 := by
  sorry

end pencils_in_each_box_l170_170135


namespace Elmer_savings_l170_170563

variables (x c : ℝ) (h1 : x > 0) (h2 : c > 0)

theorem Elmer_savings :
  let cost_old := (300 : ℝ) * c / x in
  let cost_new := (300 : ℝ) * (1.3 * c) / (1.4 * x) in
  let savings := cost_old - cost_new in
  let percentage_savings := savings / cost_old * 100 in
  abs (percentage_savings - 7.14) < 0.01 :=
by sorry

end Elmer_savings_l170_170563


namespace barycenter_condition_l170_170514

-- Define the type of the problem
variables {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]

-- Define the conditions
variables (circumscribed : ∀ P, has_center P O) (not_parallel : ¬ (is_parallel A B ∧ is_parallel C D))

-- Define the necessary angles for the quadrilateral
variables (α β γ δ : ℝ)
hypothesis angle_defs : α = 1/2 * angle A ∧ β = 1/2 * angle B ∧ γ = 1/2 * angle C ∧ δ = 1/2 * angle D

-- Define the necessary midpoints and distances
variables (M₁ M₂ T₁ T₂ : Type) [metric_space M₁] [metric_space M₂] [metric_space T₁] [metric_space T₂]
variables (midpoint_AD : midpoint A D = M₁) (midpoint_BC : midpoint B C = M₂)
variables (touch_AD : segment AD touches_circle T₁) (touch_BC : segment BC touches_circle T₂)

-- Define the circle radius R
variable (R : ℝ)

-- Express distances in terms of angles
variables
 (OA OB OC OD : ℝ)
 (distance_OA : OA = R / sin α)
 (distance_OB : OB = R / sin β)
 (distance_OC : OC = R / sin γ)
 (distance_OD : OD = R / sin δ)

-- The theorem statement
theorem barycenter_condition :
  (∀ P, is_barycenter P [A, B, C, D] → P = O) ↔ (OA * OC = OB * OD) :=
sorry

end barycenter_condition_l170_170514


namespace part_a_roots_part_b_min_max_l170_170230

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  (4 * (Real.cos x)^4 + 5 * (Real.sin x)^2) / (4 * (Real.sin x)^4 + 3 * (Real.cos x)^2)

-- Part (a): Prove that the roots of the equation g(x) = 4/3 are x = kπ/3 for any k in ℤ
theorem part_a_roots (x : ℝ) : 
  g(x) = 4/3 →
  ∃ k : ℤ, x = k * Real.pi / 3 :=
sorry

-- Part (b): Prove the maximum and minimum values of g(x)
theorem part_b_min_max :
  (∀ x : ℝ, g(x) ≥ 5/4) ∧ (∃ x : ℝ, g(x) = 55/39) :=
sorry

end part_a_roots_part_b_min_max_l170_170230


namespace region_area_l170_170972

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem region_area :
  (∃ (A : ℝ), A = (∫ x in 0..1, ∫ y in 0..(40 * fractional_part x) + floor x, 1) = 20.5) :=
begin
  sorry
end

end region_area_l170_170972


namespace product_of_consecutive_integers_sqrt_50_l170_170779

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170779


namespace f_at_1_l170_170712

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom fg_eq : ∀ x : ℝ, f x + g x = x^3 - x^2 + 1

theorem f_at_1 : f 1 = 1 := by
  sorry

end f_at_1_l170_170712


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170005

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170005


namespace cream_ratio_l170_170334

noncomputable def joe_coffee_initial := 14
noncomputable def joe_coffee_drank := 3
noncomputable def joe_cream_added := 3

noncomputable def joann_coffee_initial := 14
noncomputable def joann_cream_added := 3
noncomputable def joann_mixture_stirred := 17
noncomputable def joann_amount_drank := 3

theorem cream_ratio (joe_coffee_initial joe_coffee_drank joe_cream_added 
                     joann_coffee_initial joann_cream_added joann_mixture_stirred 
                     joann_amount_drank : ℝ) : 
  (joe_coffee_initial - joe_coffee_drank + joe_cream_added) / 
  (joann_cream_added - (joann_amount_drank * (joann_cream_added / joann_mixture_stirred))) = 17 / 14 :=
by
  -- Prove the theorem statement
  sorry

end cream_ratio_l170_170334


namespace part_i_part_ii_l170_170624

-- Define the function f(x, m) with absolute values
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 3|

-- Part I: Prove ∀ x ∈ (-1, 5), f(x, -3) < 9
theorem part_i : ∀ x, -1 < x ∧ x < 5 → f x (-3) < 9 :=
sorry

-- Part II: Prove ∃ x ∈ [2, 4], f(x, m) ≤ 3 implies m ∈ [-2, 2]
theorem part_ii (m : ℝ) : (∃ x ∈ set.Icc (2:ℝ) 4, f x m ≤ 3) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end part_i_part_ii_l170_170624


namespace unique_p_q_inequality_l170_170485

theorem unique_p_q_inequality : 
  ∃! (p q : ℝ), p = -1 ∧ q = (sqrt 2 + 1) / 2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → abs (sqrt (1 - x^2) - p * x - q) ≤ (sqrt 2 - 1) / 2 :=
sorry

end unique_p_q_inequality_l170_170485


namespace rubber_boat_lost_time_l170_170122

theorem rubber_boat_lost_time (a b : ℝ) (x : ℝ) (h : (5 - x) * (a - b) + (6 - x) * b = a + b) : x = 4 :=
  sorry

end rubber_boat_lost_time_l170_170122


namespace determine_ω_l170_170225

def function_period (ω : ℝ) := (∀ x : ℝ, f(x) = (sin (ω * x) + cos (ω * x))^2 + 2 * cos (2 * ω * x)) 
  ∧ ( ∃ T : ℝ, T = 2 * π / 3 ∧ ∀ x : ℝ, f(x + T) = f(x) )

noncomputable def ω_value : ℝ := 3 / 2

theorem determine_ω (ω : ℝ) (h : function_period ω) : ω = ω_value :=
sorry

end determine_ω_l170_170225


namespace bacteria_growth_l170_170903

theorem bacteria_growth (initial_cells : ℕ) (growth_rate : ℕ) (total_days : ℕ) (period_days : ℕ) :
  initial_cells = 4 → growth_rate = 2 → total_days = 10 → period_days = 2 → 
  initial_cells * growth_rate ^ (total_days / period_days - 1) = 64 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end bacteria_growth_l170_170903


namespace smallest_model_length_l170_170385

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end smallest_model_length_l170_170385


namespace line_through_two_quadrants_l170_170284

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170284


namespace polar_coordinates_rectPoint_l170_170950

-- Define the rectangular point
def rectPoint := (2 : ℝ, -2 * Real.sqrt 2)

-- Define the conditions
def r := Real.sqrt (2^2 + (-2 * Real.sqrt 2)^2)
def θ := 5 * Real.pi / 4

-- The proof statement
theorem polar_coordinates_rectPoint : 
  ∃ (r θ : ℝ), r = 2 * Real.sqrt 3 ∧ θ = 5 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
begin
  use 2 * Real.sqrt 3,
  use 5 * Real.pi / 4,
  split,
  -- r = sqrt(2^2 + (-2 sqrt 2)^2)
  { exact Real.sqrt (2^2 + (-2 * Real.sqrt 2)^2) },
  split,
  -- θ = 5π/4
  { exact 5 * Real.pi / 4 },
  split,
  -- r > 0
  { exact Real.sqrt_pos.2 (by norm_num; exact add_pos (by norm_num) (mul_pos (by norm_num) (Real.sqrt_pos.2 (by norm_num)))) },
  split,
  -- 0 ≤ θ
  { exact_div (norm_num.add_pos (Real.pi_div_four_pos' (by norm_num))) },
  -- θ < 2π
  { exact lt_of_lt_of_le (Real.pi_div_four_pos' (by norm_num)) (targ num).left }
end

end polar_coordinates_rectPoint_l170_170950


namespace product_of_consecutive_integers_between_sqrt_50_l170_170826

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170826


namespace determine_q_l170_170631

theorem determine_q (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 4) : q = 2 := 
sorry

end determine_q_l170_170631


namespace sum_of_integers_is_19_l170_170303

theorem sum_of_integers_is_19
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : a - b = 5) 
  (h3 : a * b = 84) : 
  a + b = 19 :=
sorry

end sum_of_integers_is_19_l170_170303


namespace points_after_perfect_games_l170_170913

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l170_170913


namespace product_of_consecutive_integers_sqrt_50_l170_170787

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170787


namespace segment_length_B_to_B_l170_170858

-- Define the points and their reflections
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (-3, 2)

def A' : ℝ × ℝ := (-2, 0)
def B' : ℝ × ℝ := (1, -4)
def C' : ℝ × ℝ := (-3, -2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement problem in Lean 4
theorem segment_length_B_to_B' : distance B B' = 8 := by
  sorry

end segment_length_B_to_B_l170_170858


namespace log_base_exp_l170_170474

theorem log_base_exp : (∀ (b x : ℝ), b > 0 ∧ b ≠ 1 → b^(log b x) = x) → (2:ℝ)^(Real.logb 2 8) = 8 :=
by 
  intros h_log_prop
  have h_log : Real.logb 2 8 = 3 := by 
    rw [Real.logb, log_div_log, log_nat],
    norm_num, norm_cast,
  rw [h_log, Real.pow_eq_rpow, Real.pow_eq_rpow, h_log_prop 2 8] at *,
  simp, sorry


end log_base_exp_l170_170474


namespace partition_into_republics_l170_170884

-- Consider a graph with 2000 vertices and certain edges representing roads.
variable (G : Type) [GraphTheory.Graph G]
variable (V : Finset G) (E : Finset (Sym2 G))
variable (N : ℕ)
variable condition : (∀ v, (find_odd_cycles_through_vertex v E).card ≤ N)

theorem partition_into_republics (h : V.card = 2000) : ∃ (partitions : Finset (Finset G)), 
  partitions.card = 2 * N + 2 ∧ 
  (∀ p ∈ partitions, ∀ v1 v2 ∈ p, ¬(Sym2.mk v1 v2 ∈ E)) :=
sorry

end partition_into_republics_l170_170884


namespace express_in_scientific_notation_l170_170367

theorem express_in_scientific_notation (n : ℝ) (h : n = 456.87 * 10^6) : n = 4.5687 * 10^8 :=
by 
  -- sorry to skip the proof
  sorry

end express_in_scientific_notation_l170_170367


namespace wrapping_paper_area_l170_170091

theorem wrapping_paper_area (a : ℝ) (h : ℝ) : h = a ∧ 1 ≥ 0 → 4 * a^2 = 4 * a^2 :=
by sorry

end wrapping_paper_area_l170_170091


namespace boat_trip_distance_l170_170899

theorem boat_trip_distance
  (total_time : ℕ := 19)
  (stream_velocity : ℕ := 4)
  (boat_still_water_speed : ℕ := 14)
  (distance_between_A_B : ℕ := 180) :
  let downstream_speed := boat_still_water_speed + stream_velocity,
      upstream_speed := boat_still_water_speed - stream_velocity,
      time_downstream := distance_between_A_B / downstream_speed,
      time_upstream := (distance_between_A_B / 2) / upstream_speed
  in time_downstream + time_upstream = total_time →
     distance_between_A_B = 180 := 
by
  intros h
  have downstream_speed := boat_still_water_speed + stream_velocity
  have upstream_speed := boat_still_water_speed - stream_velocity
  have time_downstream := distance_between_A_B / downstream_speed
  have time_upstream := (distance_between_A_B / 2) / upstream_speed
  have h_eq : time_downstream + time_upstream = total_time := h
  exact eq_of_sub_eq_zero ((by simp [time_downstream, time_upstream, downstream_speed, upstream_speed, distance_between_A_B, nat.div_def, total_time]; linarith) : distance_between_A_B - 180 = 0)

end boat_trip_distance_l170_170899


namespace cube_root_of_expr_l170_170383

-- Define the expression under consideration
def expr : ℕ := 2^9 * 3^6 * 7^3

-- State the theorem to be proved
theorem cube_root_of_expr : int.cbrt (expr) = 504 := by 
  sorry

end cube_root_of_expr_l170_170383


namespace pasta_needed_for_family_reunion_l170_170159

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l170_170159


namespace value_of_sum_is_uncertain_l170_170254

noncomputable def verify_condition (a b c : ℤ) : Prop :=
  (|a - b| ^ 19 + |c - a| ^ 95 = 1)

theorem value_of_sum_is_uncertain (a b c : ℤ) (h : verify_condition a b c) :
  ∃ n ∈ {1, 2}, |c - a| + |a - b| + |b - a| = n :=
begin
  sorry
end

end value_of_sum_is_uncertain_l170_170254


namespace gum_pieces_per_package_l170_170737

theorem gum_pieces_per_package :
  (∀ (packages pieces each_package : ℕ), packages = 9 ∧ pieces = 135 → each_package = pieces / packages → each_package = 15) := 
by
  intros packages pieces each_package
  sorry

end gum_pieces_per_package_l170_170737


namespace zara_goats_l170_170069

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l170_170069


namespace impossible_star_placement_l170_170675

theorem impossible_star_placement : 
  ¬ (∃ (star_grid : Fin 10 → Fin 10 → bool), 
    (∀ i j : Fin 10, 
       let square := star_grid i j ∨ star_grid i (j % 9 + 1) ∨ star_grid (i % 9 + 1) j ∨ star_grid (i % 9 + 1) (j % 9 + 1) in
       count_true square = 2) ∧
    (∀ i j : Fin 10, 
       let rect1 := star_grid i j ∧ star_grid i (j % 9 + 1) ∧ star_grid i (j % 9 + 2)
       let rect2 := star_grid i j ∧ star_grid (i % 9 + 1) j ∧ star_grid (i % 9 + 2) j in
       count_true rect1 + count_true rect2 = 1)) :=
sorry

end impossible_star_placement_l170_170675


namespace product_of_consecutive_integers_sqrt_50_l170_170843

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170843


namespace max_chips_in_grid_l170_170312

-- Definition of the problem
def grid := Fin 200 × Fin 200
def color := grid → Option Bool

-- max_chips function captures the maximum number of chips in the grid given the conditions
def max_chips (g : color) : Nat :=
  (Finset.univ : Finset grid).filter_map g |>.card

-- Hypothesis: each chip sees exactly 5 chips of the opposite color
def sees_opposite_color (g : color) (p : grid) : Prop :=
  g p = some true ∧ (Finset.card ((Finset.univ : Finset grid).filter (λ q, ((q.1 = p.1 ∨ q.2 = p.2) ∧ g q = some false))) = 5)
  ∨ g p = some false ∧ (Finset.card ((Finset.univ : Finset grid).filter (λ q, ((q.1 = p.1 ∨ q.2 = p.2) ∧ g q = some true))) = 5)

theorem max_chips_in_grid : 
  ∀ (g : color),
  (∀ p, g p = none ∨ sees_opposite_color g p) →
  max_chips g ≤ 3800 :=
by sorry

end max_chips_in_grid_l170_170312


namespace min_value_expression_l170_170189

theorem min_value_expression : 
  ∃ x : ℝ, 
    let f := λ x, sqrt (x^2 + (2 - x)^2) + sqrt ((2 - x)^2 + (2 + x)^2) in 
    (∀ y : ℝ, f y ≥ 2 * sqrt 5) ∧ f x = 2 * sqrt 5 :=
sorry

end min_value_expression_l170_170189


namespace maximize_prob_l170_170745

-- Define the probability of correctly answering each question
def prob_A : ℝ := 0.6
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.5

-- Define the probability of getting two questions correct in a row for each order
def prob_A_first : ℝ := (prob_A * prob_B * (1 - prob_C) + (1 - prob_A) * prob_B * prob_C) +
                        (prob_A * prob_C * (1 - prob_B) + (1 - prob_A) * prob_C * prob_B)
def prob_B_first : ℝ := (prob_B * prob_A * (1 - prob_C) + (1 - prob_B) * prob_A * prob_C) +
                        (prob_B * prob_C * (1 - prob_A) + (1 - prob_B) * prob_C * prob_A)
def prob_C_first : ℝ := (prob_C * prob_A * (1 - prob_B) + (1 - prob_C) * prob_A * prob_B) +
                        (prob_C * prob_B * (1 - prob_A) + (1 - prob_C) * prob_B * prob_A)

-- Prove that the maximum probability is obtained when question C is answered first
theorem maximize_prob : prob_C_first > prob_A_first ∧ prob_C_first > prob_B_first :=
by
  -- Add the proof details here
  sorry

end maximize_prob_l170_170745


namespace max_intersections_of_circle_and_three_lines_l170_170494

theorem max_intersections_of_circle_and_three_lines:
  ∀ (circle : set (ℝ × ℝ)) (line1 line2 line3 : set (ℝ × ℝ)),
  is_circle circle →
  distinct_lines line1 line2 line3 →
  number_of_intersections circle line1 line2 line3 = 9 :=
sorry

end max_intersections_of_circle_and_three_lines_l170_170494


namespace confidence_interval_a_confidence_interval_sigma2_l170_170763

variables {ξ : Type} [distribution : measure_theory.measure_space ξ] {a : ℝ} {σ^2 : ℝ}
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

noncomputable def sample_mean : ℝ := (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10) / 10
noncomputable def sample_variance : ℝ := (1/9) * ((x1 - sample_mean) ^ 2 + (x2 - sample_mean) ^ 2 + (x3 - sample_mean) ^ 2 + (x4 - sample_mean) ^ 2 + (x5 - sample_mean) ^ 2 + (x6 - sample_mean) ^ 2 + (x7 - sample_mean) ^ 2 + (x8 - sample_mean) ^ 2 + (x9 - sample_mean) ^ 2 + (x10 - sample_mean) ^ 2)

-- Given conditions
axiom (sample_mean : ℝ) (sample_variance : ℝ)
axiom (mean_condition : sample_mean = 1.17)
axiom (variance_condition : sample_variance = 0.25)

-- Given confidence levels
axiom (confidence_level_a : ℝ)
axiom (confidence_level_sigma2 : ℝ)
axiom (confidence_level_a_condition : confidence_level_a = 0.98)
axiom (confidence_level_sigma2_condition : confidence_level_sigma2 = 0.96)

-- Statements to prove
theorem confidence_interval_a : sample_mean - 0.446 ≤ a ∧ a ≤ sample_mean + 0.446 :=
by {
    sorry
}

theorem confidence_interval_sigma2 : 0.114 ≤ σ^2 ∧ σ^2 ≤ 0.889 :=
by {
    sorry
}

end confidence_interval_a_confidence_interval_sigma2_l170_170763


namespace ParallelSegmentsEqualIn3D_l170_170993

-- Let's define the essential properties and propositions in Lean
structure Line (ℝ : Type) := 
  (parallel_to : Line ℝ → Prop)

structure Segment (ℝ : Type) := 
  (line : Line ℝ)
  (parallel_to : Segment ℝ → Prop)

structure Plane (ℝ : Type) := 
  (parallel_to : Plane ℝ → Prop)

structure Segment3D (ℝ : Type) := 
  (plane : Plane ℝ)
  (parallel_to : Segment3D ℝ → Prop)

-- Given Proposition: Parallel line segments between two parallel lines are equal.
axiom ParallelSegmentsEqualIn2D (ℝ : Type) 
  (l1 l2 : Line ℝ) (s1 s2 : Segment ℝ) 
  (h_l1_l2 : l1.parallel_to l2) (h_s1_s2 : s1.parallel_to s2) : 
  l1 = l2 → s1 = s2

-- Proposition to be proved: Parallel line segments between two parallel planes are equal.
theorem ParallelSegmentsEqualIn3D (ℝ : Type) 
  (p1 p2 : Plane ℝ) (s1 s2 : Segment3D ℝ) 
  (h_p1_p2 : p1.parallel_to p2) (h_s1_s2 : s1.parallel_to s2) : 
  p1 = p2 → s1 = s2 := 
sorry

end ParallelSegmentsEqualIn3D_l170_170993


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170042

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170042


namespace geometric_sum_over_term_l170_170429

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def geometric_term (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geometric_sum_over_term (a₁ : ℝ) (q : ℝ) (h₁ : q = 3) :
  (geometric_sum a₁ q 4) / (geometric_term a₁ q 4) = 40 / 27 := by
  sorry

end geometric_sum_over_term_l170_170429


namespace awareness_survey_sampling_l170_170520

theorem awareness_survey_sampling
  (students : Set ℝ) -- assumption that defines the set of students
  (grades : Set ℝ) -- assumption that defines the set of grades
  (awareness : ℝ → ℝ) -- assumption defining the awareness function
  (significant_differences : ∀ g1 g2 : ℝ, g1 ≠ g2 → awareness g1 ≠ awareness g2) -- significant differences in awareness among grades
  (first_grade_students : Set ℝ) -- assumption defining the set of first grade students
  (second_grade_students : Set ℝ) -- assumption defining the set of second grade students
  (third_grade_students : Set ℝ) -- assumption defining the set of third grade students
  (students_from_grades : students = first_grade_students ∪ second_grade_students ∪ third_grade_students) -- assumption that the students are from first, second, and third grades
  (representative_method : (simple_random_sampling → False) ∧ (systematic_sampling_method → False))
  : stratified_sampling_method := 
sorry

end awareness_survey_sampling_l170_170520


namespace joe_and_dad_total_marshmallows_roasted_l170_170682

theorem joe_and_dad_total_marshmallows_roasted :
  (let dads_marshmallows := 21
       dads_roasted := dads_marshmallows / 3
       joes_marshmallows := 4 * dads_marshmallows
       joes_roasted := joes_marshmallows / 2
   in dads_roasted + joes_roasted = 49) :=
by
  let dads_marshmallows := 21
  let dads_roasted := dads_marshmallows / 3
  let joes_marshmallows := 4 * dads_marshmallows
  let joes_roasted := joes_marshmallows / 2
  show dads_roasted + joes_roasted = 49 from sorry

end joe_and_dad_total_marshmallows_roasted_l170_170682


namespace count_n_that_factorizes_l170_170575

open Nat

theorem count_n_that_factorizes (N : Nat) (hN : N = 2000) :
  let possible_ns := {n : Nat | 1 ≤ n ∧ n ≤ N ∧ n % 3 = 0 ∧ ∃ a b c : Int, (X - a) * (X - b) * (X - c) = X^3 + X^2 - n ∧ a + b + c = -1 ∧ a * b * c = -n}
  possible_ns.card = 31 :=
by
  sorry

end count_n_that_factorizes_l170_170575


namespace intersecting_plane_distance_l170_170433

noncomputable def P : ℝ × ℝ × ℝ := (0,3,0)
noncomputable def Q : ℝ × ℝ × ℝ := (2,0,0)
noncomputable def R : ℝ × ℝ × ℝ := (2,5,5)

-- Calculate the distance function
def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

theorem intersecting_plane_distance :
  ∃ S T : ℝ × ℝ × ℝ, S = (2,0,5) ∧ T = (0,5,0) ∧ distance S T = 3 * real.sqrt 6 :=
sorry

end intersecting_plane_distance_l170_170433


namespace number_of_correct_propositions_l170_170536

theorem number_of_correct_propositions :
  let a1 := ((∀ n : ℕ, n > 0 → a_n = 1 / (n * (n + 2))) ∧ (a₁ > 0 ∧ ∀ n : ℕ, n > 0 → a_{n+1} < a_n))
            ∧ (a_10 = 1 / 120)
            ∧ ∃ max_a : ℝ, max_a = a_1 ∧ (∀ k : ℕ, k > 1 → a_k < max_a),
      a2 := (∀ n : ℕ, n > 0 → a_n = sqrt (3 * n - 1)),
      a3 := ∃ k : ℤ, k = 2 ∧ (∀ n : ℤ, a_n = k * n - 5) ∧ a_8 = 11 ∧ a_17 = 29,
      a4 := (∀ a_n : ℤ, (a_{n+1} = a_n + 3) → ∀ n : ℕ, n > 0 → a_{n+1} > a_n)
  in 
    a1 ∧ a2 ∧ a3 ∧ a4 → 4 = 4 := 
by
  sorry

end number_of_correct_propositions_l170_170536


namespace find_two_digit_numbers_l170_170180
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else sum_of_digits (n / 10) + (n % 10)

def is_valid_number (a : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧
  (∀ m ∈ {2, 3, 4, 5, 6, 7, 8, 9}, sum_of_digits (m * a) = sum_of_digits a)

theorem find_two_digit_numbers :
  ∀ a, is_valid_number a ↔ a = 18 ∨ a = 45 ∨ a = 90 :=
by
  sorry

end find_two_digit_numbers_l170_170180


namespace particle_not_illuminated_by_ray_l170_170501

-- Define the problem conditions step by step
def x (t : ℝ) : ℝ := 3
def y (t : ℝ) : ℝ := 3 + sin t * cos t - sin t - cos t

-- The statement of the theorem
theorem particle_not_illuminated_by_ray (c : ℝ) (h1 : 0 < c) :
    ¬∃ t : ℝ, y t = 3 * c ↔ c ∈ (Ioo 0 (1/2)) ∪ Ioi (7/6) :=
by
  sorry

end particle_not_illuminated_by_ray_l170_170501


namespace volume_of_regular_tetrahedron_unit_edge_l170_170434

theorem volume_of_regular_tetrahedron_unit_edge :
  ∀ (a : ℝ), a = 1 → volume_of_regular_tetrahedron a = (sqrt 2) / 6 :=
sorry

end volume_of_regular_tetrahedron_unit_edge_l170_170434


namespace min_m_Rm_l_eq_l_l170_170717

/-- Let l₁ and l₂ be lines passing through the origin, making angles
    π/60 and π/45 radians with the positive x-axis, respectively.
    Let R(l) be a transformation reflecting a line l firstly in l₁ 
    and then the resulting line in l₂. Let R⁽¹⁾(l) = R(l) and 
    R⁽ⁿ⁾(l) = R(R⁽ⁿ⁻¹⁾(l)). Given that l is y = 1/3 x, prove that the 
    smallest positive integer m for which R⁽ᵐ⁾(l) = l is 30.
-/
theorem min_m_Rm_l_eq_l :
  let α : ℝ := π / 60
      β : ℝ := π / 45
      θ : ℝ := Real.arctan (1 / 3)
      l : ℝ → ℝ := λ x, 1 / 3 * x,
      R : (ℝ → ℝ) → (ℝ → ℝ) := λ l, {
          let l' := Reflect(l, α),
          Reflect(l', β)
      },
  ∃ m : ℕ, m > 0 ∧ R^(m) l = l := sorry

end min_m_Rm_l_eq_l_l170_170717


namespace range_of_s_l170_170465

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_of_s : set.range s = set.Ioi 0 := 
by 
  sorry

end range_of_s_l170_170465


namespace evaluate_256_pow_5_div_8_l170_170175

theorem evaluate_256_pow_5_div_8 (h : 256 = 2^8) : 256^(5/8) = 32 :=
by
  sorry

end evaluate_256_pow_5_div_8_l170_170175


namespace value_of_x_l170_170062

theorem value_of_x (x : ℝ) :
  (4 / x) * 12 = 8 ↔ x = 6 :=
by
  sorry

end value_of_x_l170_170062


namespace circle_radius_l170_170905

theorem circle_radius (Q P : ℝ) (h1 : Q = real.pi * r ^ 2) (h2 : P = 2 * real.pi * r) (h3 : Q / P = 10) : r = 20 :=
  sorry

end circle_radius_l170_170905


namespace distance_after_5_hours_l170_170881

-- Conditions
def start_point := (0 : ℝ, 0 : ℝ)  -- Starting from the same point

def speed_girl1 := 5  -- speed of the first girl in km/hr
def speed_girl2 := 10  -- speed of the second girl in km/hr
def time_elapsed := 5  -- time in hours

-- Distance Calculation
def distance_girl1 := speed_girl1 * time_elapsed  -- distance traveled by the first girl
def distance_girl2 := speed_girl2 * time_elapsed  -- distance traveled by the second girl

-- Total distance is the sum of the two distances since they're walking in opposite directions
def total_distance := distance_girl1 + distance_girl2

-- Statement to be proved
theorem distance_after_5_hours : total_distance = 75 := by
  -- You can provide the proof here if needed, but for now we use sorry to indicate that the proof is left out.
  sorry

end distance_after_5_hours_l170_170881


namespace exist_geometric_sequence_l170_170990

theorem exist_geometric_sequence (T : ℕ → ℕ) (g : ℕ → ℝ) :
  (∀ (n : ℕ), n ≥ 4 → 
    T n = finset.card {a : fin n // (∀ i, i < n → a i ∈ {1, 2, 3, 4}) 
      ∧ a 0 = 1 ∧ a (n-1) = 1 ∧ a 1 ≠ 1
      ∧ (∀ j, 2 ≤ j → j < n → a j ≠ a (j-1) ∧ a j ≠ a (j-2))}) →
  (∃ (g : ℕ → ℝ), 
    (∀ (n : ℕ), n ≥ 4 → 
      g n = (3 / 16) * 2 ^ n) ∧ 
    ∀ (n : ℕ), n ≥ 4 →
      g n - 2 * real.sqrt (g n) < T n ∧ 
      T n < g n + 2 * real.sqrt (g n)) :=
begin
  sorry
end

end exist_geometric_sequence_l170_170990


namespace num_combinations_30_cents_l170_170244

def coins_combinations (penny nickel dime total_value : ℕ) : Prop :=
  ∃ (p n d : ℕ), p * penny + n * nickel + d * dime = total_value

theorem num_combinations_30_cents : coins_combinations 1 5 10 30 = 20 :=
by
  sorry

end num_combinations_30_cents_l170_170244


namespace problem_solution_l170_170202

theorem problem_solution (a b : ℕ) (h₀ : b ∈ {0, 1}) (h₁ : a ∈ {0, 1, 2}) (h₂ : 9 + 2 * b = 9 * a + 2) : a = 1 ∧ b = 1 :=
by sorry

end problem_solution_l170_170202


namespace product_of_consecutive_integers_between_sqrt_50_l170_170828

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170828


namespace find_income_given_conditions_l170_170115

noncomputable def monthly_income : Type := ℝ

-- Conditions
def son_distribution (I : monthly_income) : monthly_income := 0.12 * I
def daughter_distribution (I : monthly_income) : monthly_income := 0.10 * I
def joint_investment (I : monthly_income) : monthly_income := 0.12 * I
def taxes (I : monthly_income) : monthly_income := 0.08 * I
def monthly_expenses : monthly_income := 2000
def loan_payment (I : monthly_income) : monthly_income := 0.015 * I

-- Remaining amount before donation
def remaining_after_distributions (I : monthly_income) : monthly_income :=
  I - (son_distribution I + daughter_distribution I + joint_investment I + taxes I + loan_payment I + monthly_expenses)

-- The donation made if the remaining amount is greater than $10,000
def orphanage_donation (R : monthly_income) : monthly_income :=
  if R > 10000 then 0.05 * R else 0

-- Remaining amount after donation
def remaining_after_donation (I : monthly_income) : monthly_income := 
  let R := remaining_after_distributions I in
  R - orphanage_donation R

-- The problem statement to prove
theorem find_income_given_conditions : ∃ I : monthly_income, remaining_after_donation I = 22000 ∧ I ≈ 44517.33 := sorry

end find_income_given_conditions_l170_170115


namespace product_of_integers_around_sqrt_50_l170_170774

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170774


namespace max_handshakes_25_people_l170_170891

theorem max_handshakes_25_people : 
  (∃ n : ℕ, n = 25) → 
  (∀ p : ℕ, p ≤ 24) → 
  ∃ m : ℕ, m = 300 :=
by sorry

end max_handshakes_25_people_l170_170891


namespace solve_fraction_eq_zero_l170_170060

theorem solve_fraction_eq_zero (x : ℝ) (h : x ≠ 0) : 
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_fraction_eq_zero_l170_170060


namespace find_number_added_l170_170863

-- Definitions corresponding to the conditions
def x := λ y : ℕ, 1 / 4 * y
def n := λ x y : ℕ, 1 / 2 * y - x

-- Main theorem statement based on the conditions and the proof problem
theorem find_number_added 
  (y : ℕ)
  (h_y : y = 48)
  (h_ratio1 : x y = 1 / 4 * y)
  (h_ratio2 : n (1 / 4 * y) y = 1 / 2 * y - (1 / 4 * y)) :
  n (1 / 4 * y) y = 12 := by
  sorry

end find_number_added_l170_170863


namespace points_after_perfect_games_l170_170911

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l170_170911


namespace range_s_l170_170461

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_s : set.range s = set.Ioi 0 := by
  sorry

end range_s_l170_170461


namespace minimum_value_inequality_l170_170232

variables {a x : ℝ}

theorem minimum_value_inequality
  (h_inequality : ∀ x, x^2 - 4*a*x + 3*a^2 < 0)
  (h_a_pos : a > 0) :
  (4 * a + 1 / (3 * a)) ≥ 2 * sqrt (4 / 3) ∧ (∃ a, 4 * a + 1 / (3 * a) = (4 * sqrt 3) / 3) :=
by
  -- Here we should provide the actual proof, but we leave it as sorry for now.
  sorry

end minimum_value_inequality_l170_170232


namespace area_of_cyclic_quadrilateral_l170_170697

variables {a b c d S : ℝ}
variables {φ : ℝ}

-- Condition: 2φ is the sum of two opposite angles of a cyclic quadrilateral
def sum_opposite_angles (φ : ℝ) : Prop :=
  ∃ α β γ δ : ℝ, α + β + γ + δ = π ∧ 2 * φ = α + γ

-- Define the sides a, b, c, d and area S of the cyclic quadrilateral
def sides_and_area (a b c d S : ℝ) (φ : ℝ) : Prop := 
  ∃ r, 
    a = r * (tan (φ) + tan (π - φ)) ∧
    b = r * (tan (π - φ) + tan (2*π - φ)) ∧
    c = r * (tan (2*π - φ) + tan (π + φ)) ∧
    d = r * (tan (π + φ) + tan (φ - π)) ∧
    S = r^2 * (tan (φ) + tan (π - φ) + tan (2*π - φ) + tan (π + φ))

-- Prove the desired equation
theorem area_of_cyclic_quadrilateral 
  (h1 : sum_opposite_angles φ) 
  (h2 : sides_and_area a b c d S φ) : 
  S = sqrt (a * b * c * d) * sin φ :=
sorry

end area_of_cyclic_quadrilateral_l170_170697


namespace lcm_48_90_l170_170188

theorem lcm_48_90 : Nat.lcm 48 90 = 720 := by
  have factorization_48 : 48 = 2^4 * 3^1 := by
    norm_num
    sorry
  have factorization_90 : 90 = 2^1 * 3^2 * 5^1 := by
    norm_num
    sorry
  sorry

end lcm_48_90_l170_170188


namespace sum_of_sequence_l170_170940

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) : 
  (∑ k in Finset.range (n + 1), if (k ≥ 2) then (3 * k + 2) else 0) = (3 * n^2 + 7 * n - 10) / 2 :=
by
  -- Proof will go here
  sorry

end sum_of_sequence_l170_170940


namespace no_partition_l170_170562

theorem no_partition (n : ℕ) (h : n > 1) : 
  ¬ ∃ (A : fin n → set ℕ), 
    (∀ i, (A i).nonempty ∧ ∀ (i : fin n), ∀ (S : finset (fin n)), S.card = n - 1 →
    ∃ x ∈ A (↑(finset.univ.val.erase i).max' sorry), 
      S.sum (λ j, x (j)) ∈ A i) :=
sorry

end no_partition_l170_170562


namespace probability_exactly_three_win_l170_170302

theorem probability_exactly_three_win (total_balls: ℕ) (balls: fin total_balls → ℕ) (draws_each: ℕ) (total_people: ℕ) 
(conditions: ∀ a b: fin total_balls, a ≠ b → (balls a) * (balls b) % 4 = 0 → ((a, b), true)) :
  draw_balls = 2 →
  total_balls = 6 →
  draws_each = 2 →
  total_people = 4 →
  ∃ (p: ℚ), p = 96 / 625 :=
by
  sorry

end probability_exactly_three_win_l170_170302


namespace ice_cream_flavors_l170_170638

theorem ice_cream_flavors : 
  let chocolate := 1
  let vanilla := 1
  let strawberry := 1
  let remaining_scoops := 5 - chocolate - vanilla - strawberry
  (finset.card (finset.range (remaining_scoops + 3 - 1)).choose 2) = 6 :=
by
  let chocolate := 1
  let vanilla := 1
  let strawberry := 1
  let remaining_scoops := 5 - chocolate - vanilla - strawberry
  have h : finset.card (finset.range (remaining_scoops + 3 - 1)).choose 2 = 6,
  from sorry,
  exact h

end ice_cream_flavors_l170_170638


namespace convex_quadrilateral_inequality_l170_170102

-- Definitions of the properties and conditions
variables (A B C D P : Type) 
variables (h AD BC : ℝ)
variables (d₁ d₂ : ℝ)

-- Conditions from the problem
axiom AB_eq_AD_plus_BC : ∀ (AB AD BC : ℝ), AB = AD + BC
axiom h_distance_CD : ∀ (P : Type) (CD : ℝ), d₁ = h
axiom AP_eq_h_AD : ∀ (h AD : ℝ), d₂ = h + AD
axiom BP_eq_h_BC : ∀ (h BC : ℝ), d₂ = h + BC

-- Target inequality proof statement
theorem convex_quadrilateral_inequality :
  ∀ (AD BC h : ℝ), 
    (AB_eq_AD_plus_BC AD BC) ∧ (h_distance_CD P h) ∧ (AP_eq_h_AD h AD) ∧ (BP_eq_h_BC h BC) →
    (1 / ℤ.sqrt h) ≥ (1 / ℤ.sqrt AD) + (1 / ℤ.sqrt BC) :=
by 
  sorry

end convex_quadrilateral_inequality_l170_170102


namespace collinear_O_P_F_l170_170883

theorem collinear_O_P_F 
  {A B C D E F G O P : Type*}
  [geometry O A B C D E F G]
  (h_AB_BC : dist A B = dist B C)
  (h_BC_CD : dist B C = dist C D)
  (h_tangent_AB : tangent_circle O A B E)
  (h_tangent_BC : tangent_circle O B C F)
  (h_tangent_CD : tangent_circle O C D G)
  (h_intersection : intersects AC BD P) :
  collinear O P F :=
sorry

end collinear_O_P_F_l170_170883


namespace product_of_integers_around_sqrt_50_l170_170770

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170770


namespace pi_estimation_l170_170382

noncomputable def estimate_pi (m : ℕ) (total : ℕ) : ℝ :=
  (m * 4 / total + 2)

theorem pi_estimation
  (n : ℕ) (m : ℕ)
  (h_n : n = 120) (h_m : m = 34)
  : estimate_pi m n = 47 / 15 :=
by
  rw [estimate_pi, h_m, h_n]
  norm_num
  sorry

end pi_estimation_l170_170382


namespace four_equal_angles_of_heptagon_l170_170210

theorem four_equal_angles_of_heptagon (α β γ δ ε ζ η : ℝ)
  (h_concave : convex_heptagon α β γ δ ε ζ η)
  (h_sum_invariant : ∀ (a b c d : ℝ),
    (sine α + sine β + sine γ + sine δ + cosine ε + cosine ζ + cosine η) =
    (sine a + sine b + sine c + sine d + cosine e + cosine f + cosine g)) :
  (∃ (α' β' γ' δ' : ℝ), α = α' ∧ β = β' ∧ γ = γ' ∧ δ = δ') :=
sorry

end four_equal_angles_of_heptagon_l170_170210


namespace conclusions_correct_l170_170947

def f : ℕ+ → ℕ+ → ℕ+
| ⟨1, _⟩, ⟨1, _⟩ := ⟨1, by norm_num⟩
| m, ⟨1, _⟩ := ⟨2 * f ⟨m.1 - 1, m.2.pred'_proof⟩ ⟨1, by norm_num⟩, by norm_num⟩
| m, ⟨n + 1, h⟩ := ⟨(f m ⟨n, h.pred'_proof⟩).1 + 2, by norm_num⟩

theorem conclusions_correct :
  f ⟨1, by norm_num⟩ ⟨5, by norm_num⟩ = ⟨9, by norm_num⟩ ∧
  f ⟨5, by norm_num⟩ ⟨1, by norm_num⟩ = ⟨16, by norm_num⟩ ∧
  f ⟨5, by norm_num⟩ ⟨6, by norm_num⟩ = ⟨26, by norm_num⟩ :=
by {
  sorry
}

end conclusions_correct_l170_170947


namespace bicycle_spokes_l170_170543

theorem bicycle_spokes (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) (total_spokes : ℕ) 
  (h1 : num_bicycles = 4) (h2 : wheels_per_bicycle = 2) (h3 : total_spokes = 80) :
  total_spokes / (num_bicycles * wheels_per_bicycle) = 10 :=
by
  -- Setting up the given conditions
  have h : num_bicycles * wheels_per_bicycle = 8, from by
    rw [h1, h2],
    calc 4 * 2 = 8 : by norm_num,
  rw h at *,
  -- Calculate the number of spokes per wheel
  calc total_spokes / 8 = 10 : by
    rw h3,
    norm_num,
  -- Adding sorry to the end to skip the proof as instructed
  sorry

end bicycle_spokes_l170_170543


namespace sum_of_roots_cubic_equation_l170_170054

theorem sum_of_roots_cubic_equation :
  let roots := multiset.to_finset (multiset.filter (λ r, r ≠ 0) (RootSet (6 * (X ^ 3) + 7 * (X ^ 2) + (-12) * X) ℤ))
  (roots.sum : ℤ) / (roots.card : ℤ) = -117 / 100 := sorry

end sum_of_roots_cubic_equation_l170_170054


namespace symmetric_line_equation_l170_170411

def line_1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line_2 (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem symmetric_line_equation :
  ∀ x y : ℝ, line_1 x y → line_2 x y → symmetric_line x y := 
sorry

end symmetric_line_equation_l170_170411


namespace length_of_EF_l170_170649

noncomputable def D : ℝ := ⟨0, 0⟩ -- For convenience, consider D at origin in a coordinate system
noncomputable def E : ℝ := ⟨45, 0⟩ -- E coordinates on x-axis, DE = 45
noncomputable def F : Type := sorry -- F point definition, ℝ or vector space TBD

structure Triangle :=
  (DE : ℝ)
  (DF : ℝ)
  (intersects_circle : D ∈ circle_with_center D radius DE)

axiom EY_length : ℤ
axiom FY_length : ℤ

-- Define a theorem to prove that the length of EF is 120
theorem length_of_EF : ∃ EF : ℝ, EF = 120 := 
sorry

end length_of_EF_l170_170649


namespace number_of_dishes_I_can_eat_l170_170250

theorem number_of_dishes_I_can_eat :
  ∀ (total_dishes vegan_gluten vegan_dairy vegan_dishes : ℕ),
    total_dishes = 30 → 
    vegan_dishes = total_dishes // 6 →
    vegan_gluten = vegan_dishes // 2 →
    vegan_dairy = 2 → 
    (vegan_dishes - vegan_gluten <= vegan_dishes - vegan_dairy) → 
    ((vegan_dishes - vegan_gluten) = 2) :=
by 
  intros total_dishes vegan_gluten vegan_dairy vegan_dishes h1 h2 h3 h4 h5
  sorry

end number_of_dishes_I_can_eat_l170_170250


namespace product_of_consecutive_integers_sqrt_50_l170_170786

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170786


namespace computer_cost_l170_170558

theorem computer_cost 
    (initial_money : ℕ)
    (total_spent : ℕ)
    (money_left : ℕ) 
    (printer_cost : ℕ)
    (h1 : initial_money = 450)
    (h2 : total_spent = 40)
    (h3 : money_left = 10)
    (h4 : printer_cost = 40)
  : let computer_cost := initial_money - total_spent - money_left in
    computer_cost = 400 := 
by
  sorry

end computer_cost_l170_170558


namespace center_of_symmetry_exists_l170_170644

def translated_function (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

theorem center_of_symmetry_exists (k : ℤ) : 
  ∃ x : ℝ, translated_function x = 0 ∧ x = (k : ℝ) * π / 2 + π / 12 := 
sorry

end center_of_symmetry_exists_l170_170644


namespace quadrilateral_divisible_into_seven_equal_triangles_l170_170077

theorem quadrilateral_divisible_into_seven_equal_triangles (Q : Type) [quadrilateral Q] : 
  ∃ (triangles : set (triangle Q)), set.card triangles = 7 ∧ (∀ t1 t2 ∈ triangles, area t1 = area t2) := 
sorry

end quadrilateral_divisible_into_seven_equal_triangles_l170_170077


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170013

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170013


namespace symmetric_point_coordinates_l170_170181

theorem symmetric_point_coordinates (Q : ℝ × ℝ × ℝ) 
  (P : ℝ × ℝ × ℝ := (-6, 7, -9)) 
  (A : ℝ × ℝ × ℝ := (1, 3, -1)) 
  (B : ℝ × ℝ × ℝ := (6, 5, -2)) 
  (C : ℝ × ℝ × ℝ := (0, -3, -5)) : Q = (2, -5, 7) :=
sorry

end symmetric_point_coordinates_l170_170181


namespace problem_proof_l170_170581

theorem problem_proof (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 2 * y) / (x - 2 * y) = 23 :=
by sorry

end problem_proof_l170_170581


namespace intersection_proof_range_of_a_l170_170236

-- Definitions for sets A, B, and C
def A : set ℝ := {x | x^2 - x - 12 < 0}
def B : set ℝ := {x | x^2 + 2x - 8 > 0}
def C (a : ℝ) : set ℝ := {x | x^2 - 4 * a * x + 3 * a^2 < 0}

-- Conditions for C
def non_zero_a (a : ℝ) : Prop := a ≠ 0

-- Define the statement A ∩ (C \ B) = { x | -3 < x ≤ 2 }
theorem intersection_proof (a : ℝ) (h : non_zero_a a) :
  A ∩ (C a \ B) = { x | -3 < x ∧ x ≤ 2 } :=
by
  sorry

-- Define the statement that the range of values for a is between 4/3 and 2
theorem range_of_a (a : ℝ) (h : non_zero_a a) :
  (C a ⊇ (A ∩ B)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) :=
by
  sorry

end intersection_proof_range_of_a_l170_170236


namespace coords_D_area_parallelogram_l170_170619

noncomputable theory

-- Definition of the points and vectors
def A := (0, 1, 2 : ℝ × ℝ × ℝ)
def B := (-2, 0, 5 : ℝ × ℝ × ℝ)
def C := (1, -2, 4 : ℝ × ℝ × ℝ)
def D := (3, -1, 1 : ℝ × ℝ × ℝ)

-- Proving the coordinates of point D
theorem coords_D (A B C D : ℝ × ℝ × ℝ) :
  ∃ D : ℝ × ℝ × ℝ, D = (3, -1, 1) := 
by {
  use (3, -1, 1),
  trivial,
}

-- Calculating the area of the parallelogram
theorem area_parallelogram (A B C D : ℝ × ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3),
      BC := (C.1 - B.1, C.2 - B.2, C.3 - B.3),
      BA := (A.1 - B.1, A.2 - B.2, A.3 - B.3),
      cos_theta := ((BC.1 * BA.1 + BC.2 * BA.2 + BC.3 * BA.3) / (real.sqrt (BC.1^2 + BC.2^2 + BC.3^2) * real.sqrt (BA.1^2 + BA.2^2 + BA.3^2))),
      sin_theta := real.sqrt (1 - cos_theta^2),
      area := (real.sqrt (BC.1^2 + BC.2^2 + BC.3^2) * real.sqrt (BA.1^2 + BA.2^2 + BA.3^2) * sin_theta)
  in area = 7 * real.sqrt 3 :=
by {
  sorry
}

end coords_D_area_parallelogram_l170_170619


namespace damaged_percentage_is_20_l170_170538

noncomputable def initial_total_pages : ℕ := 500
noncomputable def pages_written_first_week : ℕ := 150
noncomputable def percentage_written_second_week : ℕ := 30
noncomputable def pages_remaining_after_spill : ℕ := 196

def pages_remaining_first_week : ℕ :=
  initial_total_pages - pages_written_first_week

def pages_written_second_week : ℕ :=
  percentage_written_second_week * pages_remaining_first_week / 100

def pages_remaining_second_week : ℕ :=
  pages_remaining_first_week - pages_written_second_week

def pages_damaged_by_spill : ℕ :=
  pages_remaining_second_week - pages_remaining_after_spill

def percentage_damaged_by_spill : ℕ :=
  (pages_damaged_by_spill * 100) / pages_remaining_second_week

theorem damaged_percentage_is_20 :
  percentage_damaged_by_spill = 20 :=
by
  sorry

end damaged_percentage_is_20_l170_170538


namespace k_positive_first_third_quadrants_l170_170292

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170292


namespace product_of_consecutive_integers_sqrt_50_l170_170793

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170793


namespace system_linear_eq_sum_l170_170237

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l170_170237


namespace no_n_exists_11_div_mod_l170_170084

theorem no_n_exists_11_div_mod (n : ℕ) (h1 : n > 0) (h2 : 3^5 ≡ 1 [MOD 11]) (h3 : 4^5 ≡ 1 [MOD 11]) : ¬ (11 ∣ (3^n + 4^n)) := 
sorry

end no_n_exists_11_div_mod_l170_170084


namespace find_m_plus_n_l170_170701

theorem find_m_plus_n
  (AB AC BC : ℝ)
  (h_AB : AB = 2023)
  (h_AC : AC = 2022)
  (h_BC : BC = 2021) :
  let CH : ℝ := sorry,
      AH : ℝ := (4104530 / 4046),
      BH : ℝ := 2023 - AH,
      R : ℝ := (AH + CH - AC) / 2,
      S : ℝ := (CH + BH - BC) / 2 in
  let RS := |R - S| in 
    let m : ℕ := 1,
        n : ℕ := 8092 in
    m + n = 8093 := 
sorry

end find_m_plus_n_l170_170701


namespace cannot_transform_l170_170720

-- Definitions for the conditions described in the problem
def initial_set := {-Real.sqrt 2, 1 / Real.sqrt 2, Real.sqrt 2, 2}
def target_set := {-Real.sqrt 2, 1, Real.sqrt 2, 2 * Real.sqrt 2}

-- Allowed operation that replaces two numbers.
def operation (a b : ℝ) : ℝ × ℝ := ((a + b) / Real.sqrt 2, (a - b) / Real.sqrt 2)

-- Function to calculate the sum of squares of a set of numbers
def sum_of_squares (s : Set ℝ) : ℝ := s.sum (λ x, x * x)

-- Stating the problem: Prove that it's impossible to transform the initial set to the target set
theorem cannot_transform : ¬ ∃ (sequence : List (Set ℝ)), 
  sequence.head = initial_set ∧
  sequence.ilast = target_set ∧
  ∀ (s' s : Set ℝ), List.chain operation s s' sequence → s'.sum_of_squares = s.sum_of_squares :=
begin
  sorry
end

end cannot_transform_l170_170720


namespace sqrt_50_between_consecutive_integers_product_l170_170838

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170838


namespace quadrilateral_divisible_into_seven_equal_triangles_l170_170076

theorem quadrilateral_divisible_into_seven_equal_triangles (Q : Type) [quadrilateral Q] : 
  ∃ (triangles : set (triangle Q)), set.card triangles = 7 ∧ (∀ t1 t2 ∈ triangles, area t1 = area t2) := 
sorry

end quadrilateral_divisible_into_seven_equal_triangles_l170_170076


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170035

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170035


namespace zara_goats_l170_170068

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l170_170068


namespace probability_sum_even_l170_170439

theorem probability_sum_even (D : ℕ → finset ℕ) (h : ∀ i, i ∈ D i → i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) :
  (∃ (d1 d2 d3 : ℕ), d1 ∈ D 1 ∧ d2 ∈ D 2 ∧ d3 ∈ D 3 ∧ (d1 + d2 + d3) % 2 = 0) →
  (1 / 2) := sorry

end probability_sum_even_l170_170439


namespace both_locks_stall_time_l170_170337

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l170_170337


namespace sum_of_roots_l170_170057

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l170_170057


namespace sqrt_50_between_consecutive_integers_product_l170_170835

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170835


namespace right_triangle_set_C_l170_170537

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

def set_A := (2, 3, 4)
def set_B := (4, 5, 6)
def set_C := (5, 12, 13)
def set_D := (5, 6, 7)

theorem right_triangle_set_C : 
  is_right_triangle 5 12 13 :=
by {
  have h1 : 5^2 = 25 := by norm_num,
  have h2 : 12^2 = 144 := by norm_num,
  have h3 : 13^2 = 169 := by norm_num,
  calc
    5^2 + 12^2 = 25 + 144 : by rw [h1, h2]
          ... = 169 : by norm_num
          ... = 13^2 : by rw [h3]
}

end right_triangle_set_C_l170_170537


namespace right_triangle_shorter_leg_l170_170654

theorem right_triangle_shorter_leg
  (a b c : ℕ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_hypotenuse : c = 65)
  (h_integer_sides : a < b < c) :
  a = 25 :=
by
  sorry

end right_triangle_shorter_leg_l170_170654


namespace roasted_marshmallows_total_l170_170686

def joe_marshmallows (dads_marshmallows : ℕ) := 4 * dads_marshmallows

def roasted_marshmallows (total_marshmallows : ℕ) (fraction : ℕ) := total_marshmallows / fraction

theorem roasted_marshmallows_total :
  let dads_marshmallows := 21 in
  let joe_marshmallows := joe_marshmallows dads_marshmallows in
  let dads_roasted := roasted_marshmallows dads_marshmallows 3 in
  let joe_roasted := roasted_marshmallows joe_marshmallows 2 in
  dads_roasted + joe_roasted = 49 :=
by
  sorry

end roasted_marshmallows_total_l170_170686


namespace find_m_plus_b_l170_170418

-- Define the given equation
def given_line (x y : ℝ) : Prop := x - 3 * y + 11 = 0

-- Define the reflection of the given line about the x-axis
def reflected_line (x y : ℝ) : Prop := x + 3 * y + 11 = 0

-- Define the slope-intercept form of the reflected line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- State the theorem to prove
theorem find_m_plus_b (m b : ℝ) :
  (∀ x y : ℝ, reflected_line x y ↔ slope_intercept_form m b x y) → m + b = -4 :=
by
  sorry

end find_m_plus_b_l170_170418


namespace product_of_consecutive_integers_sqrt_50_l170_170784

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170784


namespace find_point_A_coordinates_l170_170505

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l170_170505


namespace distance_and_circle_verification_l170_170652

-- Step 1: Define the conditions and the point
def point := (12, 5)

-- Step 2: Define the origin and the circle radius
def origin := (0, 0)
def radius := 13

-- Step 3: Define the distance formula function
def distance (p q : ℕ × ℕ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Step 4: Define a function that checks if a point lies on the circle
def on_circle (p : ℕ × ℕ) (r : ℝ) : Prop :=
  (p.1)^2 + (p.2)^2 = r^2

-- Step 5: The main theorem stating both proof problems
theorem distance_and_circle_verification :
  distance origin point = radius ∧ on_circle point radius :=
by
  sorry

end distance_and_circle_verification_l170_170652


namespace sum_of_positive_divisors_eq_360_l170_170430

theorem sum_of_positive_divisors_eq_360 (i j k : ℕ) (h : (Finset.range (i + 1)).sum (λ n, 2^n) *
                                                (Finset.range (j + 1)).sum (λ n, 3^n) *
                                                (Finset.range (k + 1)).sum (λ n, 5^n) = 360) : 
  i + j + k = 6 := 
sorry

end sum_of_positive_divisors_eq_360_l170_170430


namespace sqrt_three_irrational_l170_170133

theorem sqrt_three_irrational : irrational (real.sqrt 3) :=
sorry

end sqrt_three_irrational_l170_170133


namespace mary_needs_more_sugar_l170_170368

theorem mary_needs_more_sugar 
  (sugar_needed flour_needed salt_needed already_added_flour : ℕ)
  (h1 : sugar_needed = 11)
  (h2 : flour_needed = 6)
  (h3 : salt_needed = 9)
  (h4 : already_added_flour = 12) :
  (sugar_needed - salt_needed) = 2 :=
by
  sorry

end mary_needs_more_sugar_l170_170368


namespace least_marked_points_on_cube_l170_170851

theorem least_marked_points_on_cube :
  let corner_cubes := 8
  let edge_cubes := 12
  let face_cubes := 6
  let internal_cubes := 1

  let points_per_corner_cube := 6
  let points_per_edge_cube := 3
  let points_per_face_cube := 1

  corner_cubes * points_per_corner_cube +
  edge_cubes * points_per_edge_cube +
  face_cubes * points_per_face_cube = 90
:= by
  let corner_cubes := 8
  let edge_cubes := 12
  let face_cubes := 6
  let internal_cubes := 1

  let points_per_corner_cube := 6
  let points_per_edge_cube := 3
  let points_per_face_cube := 1

  -- calculation to show the sum of points
  calc
    corner_cubes * points_per_corner_cube +
    edge_cubes * points_per_edge_cube +
    face_cubes * points_per_face_cube
      = 8 * 6 + 12 * 3 + 6 * 1 : by sorry
      = 48 + 36 + 6 : by sorry
      = 90 : by sorry

end least_marked_points_on_cube_l170_170851


namespace expected_value_of_winnings_l170_170090

theorem expected_value_of_winnings (p6 : ℚ) (p_not6 : ℚ) (win6 : ℚ) (lose_not6 : ℚ) :
  p6 = 1/4 ∧ p_not6 = 3/4 ∧ win6 = 4 ∧ lose_not6 = -1 →
  p6 * win6 + p_not6 * lose_not6 = 0.25 :=
by 
  intro h
  have h_p6 : p6 = 1/4 := h.1
  have h_p_not6 : p_not6 = 3/4 := h.2.1
  have h_win6 : win6 = 4 := h.2.2.1
  have h_lose_not6 : lose_not6 = -1 := h.2.2.2
  
  rw [h_p6, h_p_not6, h_win6, h_lose_not6]
  norm_num
  sorry

end expected_value_of_winnings_l170_170090


namespace square_eq_four_implies_two_l170_170645

theorem square_eq_four_implies_two (x : ℝ) (h : x^2 = 4) : x = 2 := 
sorry

end square_eq_four_implies_two_l170_170645


namespace binomial_sum_mod_l170_170148

theorem binomial_sum_mod (n : ℕ) :
  (∑ k in Finset.range(675), 3 * Nat.choose 2023 (3 * k)) % 1000 = 42 :=
by
  sorry

end binomial_sum_mod_l170_170148


namespace curler_ratio_l170_170965

theorem curler_ratio
  (total_curlers : ℕ)
  (pink_curlers : ℕ)
  (blue_curlers : ℕ)
  (green_curlers : ℕ)
  (h1 : total_curlers = 16)
  (h2 : blue_curlers = 2 * pink_curlers)
  (h3 : green_curlers = 4) :
  pink_curlers / total_curlers = 1 / 4 := by
  sorry

end curler_ratio_l170_170965


namespace total_beakers_count_l170_170111

variable (total_beakers_with_ions : ℕ) 
variable (drops_per_test : ℕ)
variable (total_drops_used : ℕ) 
variable (beakers_without_ions : ℕ)

theorem total_beakers_count
  (h1 : total_beakers_with_ions = 8)
  (h2 : drops_per_test = 3)
  (h3 : total_drops_used = 45)
  (h4 : beakers_without_ions = 7) : 
  (total_drops_used / drops_per_test) = (total_beakers_with_ions + beakers_without_ions) :=
by
  -- Proof to be filled in
  sorry

end total_beakers_count_l170_170111


namespace range_of_s_l170_170468

def s (x : ℝ) : ℝ := 1 / (2 - x)^2

theorem range_of_s : set.Ioo 0 ∞ = set.range s :=
sorry

end range_of_s_l170_170468


namespace shortest_paths_avoid_segments_l170_170576

def point := (ℕ × ℕ)

def A := (2, 2) : point
def B := (3, 2) : point
def C := (4, 2) : point
def D := (5, 2) : point
def E := (6, 2) : point
def F := (6, 3) : point
def G := (7, 2) : point
def H := (7, 3) : point

def O : point := (0, 0)
def P : point := (10, 5)

-- The main statement to be proven
theorem shortest_paths_avoid_segments : 
  (number_of_shortest_paths O P) - 
  ((number_of_paths_via_segments O P [A, B]) +
   (number_of_paths_via_segments O P [C, D]) +
   (number_of_paths_via_segments O P [E, F]) +
   (number_of_paths_via_segments O P [G, H]) -
   (number_of_paths_intersection O P [A, B, C, D]) -
   (number_of_paths_intersection O P [A, B, E, F]) -
   (number_of_paths_intersection O P [A, B, G, H]) -
   (number_of_paths_intersection O P [C, D, E, F]) -
   (number_of_paths_intersection O P [C, D, G, H]) -
   (number_of_paths_intersection O P [E, F, G, H])) = 274 :=
sorry

end shortest_paths_avoid_segments_l170_170576


namespace product_of_integers_around_sqrt_50_l170_170769

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170769


namespace correct_operation_l170_170870

theorem correct_operation : ∀ (x : ℝ), x^3 / x^2 = x :=
by
  intro x
  have h : x^3 / x^2 = x^(3 - 2) := by sorry
  rw [h]
  have h1 : x^(3 - 2) = x^1 := by sorry
  rw [h1]
  exact rfl

end correct_operation_l170_170870


namespace sqrt_50_between_consecutive_integers_product_l170_170832

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170832


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170039

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170039


namespace range_of_m_l170_170299

theorem range_of_m (m : ℝ) :
  (∃ n ∈ {3, 4, 5}, 2 < n ∧ n < m ∧ ∃p ∈ {3, 4, 5}, p ≠ n ∧ 2 < p ∧ p < m ∧ ∃q ∈ {3, 4, 5}, q ≠ n ∧ q ≠ p ∧ 2 < q ∧ q < m) 
  ∧ (¬∃x ∈ {1, 2, 7, 8, 9, 10}, (2 < x ∧ x < m)) →
  5 < m ∧ m ≤ 6 := 
sorry

end range_of_m_l170_170299


namespace line_in_first_and_third_quadrants_l170_170268

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170268


namespace quadratic_real_roots_exists_l170_170176

theorem quadratic_real_roots_exists :
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 * x1 - 6 * x1 + 8 = 0) ∧ (x2 * x2 - 6 * x2 + 8 = 0) :=
by
  sorry

end quadratic_real_roots_exists_l170_170176


namespace value_v3_horner_method_l170_170941

noncomputable def polynomial_eval_horner (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  coeffs.foldl (λ acc coeff, (acc.headD 0 * x + coeff) :: acc) []

theorem value_v3_horner_method :
  let coeffs := [2, 5, 6, 23, -8, 10, -3]
  let x := -4
  ∃ v3, polynomial_eval_horner coeffs x !! 3 = some v3 ∧ v3 = -49 := by
  sorry

end value_v3_horner_method_l170_170941


namespace tan_theta_in_terms_of_x_l170_170348

namespace Math_Proof

variables {θ x : ℝ}

-- Given conditions
axiom h1 : θ > 0 ∧ θ < π / 2
axiom h2 : sin (θ / 2) = sqrt ((x - 1) / (2 * x))

-- Goal to prove
theorem tan_theta_in_terms_of_x : tan θ = sqrt (x^2 - 1) :=
by
  sorry

end Math_Proof

end tan_theta_in_terms_of_x_l170_170348


namespace sqrt_50_product_consecutive_integers_l170_170809

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170809


namespace determine_x_l170_170958

theorem determine_x (x : ℝ) (hx : 0 < x) (h : (⌊x⌋ : ℝ) * x = 120) : x = 120 / 11 := 
sorry

end determine_x_l170_170958


namespace total_HVAC_cost_l170_170335

theorem total_HVAC_cost
  (zones : ℕ)
  (vents_per_zone : ℕ)
  (cost_per_vent : ℕ)
  (h_zones : zones = 2)
  (h_vents_per_zone : vents_per_zone = 5)
  (h_cost_per_vent : cost_per_vent = 2000) :
  zones * vents_per_zone * cost_per_vent = 20000 :=
by {
  rw [h_zones, h_vents_per_zone, h_cost_per_vent],
  norm_num,
  sorry
}

end total_HVAC_cost_l170_170335


namespace invertible_functions_product_l170_170417

def func4 (x : ℝ) := x^3
def func5 (x : ℝ) := 5 / x

theorem invertible_functions_product :
  (function.injective func4) ∧ (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → func5 x = func5 y → x = y) →
  4 * 5 = 20 :=
by
  intro h
  sorry

end invertible_functions_product_l170_170417


namespace circumcircle_touches_bisector_l170_170987

-- Defining the triangle ABC, where it is nonisosceles.
variables {A B C : Type} [Inhabited A] [Inhabited B] [Inhabited C]

-- Assuming a nonisosceles triangle ABC.
noncomputable def nonisosceles_triangle (ABC : Type) : Prop := 
  ∃ (a b c : ABC), a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Definitions for altitude (AD), angle bisectors (BE, CF), and incenter I.
noncomputable def altitude_from_A (A B C : Type) : A × B × C → (A → B → Prop) := sorry
noncomputable def angle_bisector_B (A B C : Type) : B → (A → C × B ) := sorry
noncomputable def angle_bisector_C (A B C : Type) : C → (A → B × C ) := sorry
noncomputable def incenter (A B C : Type) (I : Type) : (angle_bisector_B A B C B ) → (angle_bisector_C A B C C) → I := sorry

-- Circumcircle definition for triangle formed by AD, BE, CF.
noncomputable def circumcircle (A B C I : Type) [inh_A: Inhabited A] [inh_B: Inhabited B] [inh_C: Inhabited C] 
  (altitude_AD: A → B → C → Prop) (angle_bisector_BE: B → (A → C × B )) (angle_bisector_CF: C → (A → B × C )) 
  (incenter_I: angle_bisector_BE B → angle_bisector_CF C → I)
  : (Circle) := sorry

-- Main statement: proving the circumcircle touches the angle bisector from A
theorem circumcircle_touches_bisector (A B C I : Type) [inh_A: Inhabited A] [inh_B: Inhabited B] [inh_C: Inhabited C] 
  (nonisosceles_triangle: nonisosceles_triangle (A × B × C))
  (altitude_AD: A → B → C → Prop) 
  (angle_bisector_BE: B → (A → C × B )) 
  (angle_bisector_CF: C → (A → B × C )) 
  (incenter_I: angle_bisector_BE B → angle_bisector_CF C → I)
  (circumcircle_w: Circle)
  (touches_bisector_AI: (circumcircle circumcircle_w = ∃ (circumcircle_w : Circle), (min isosceles_triangle A ∨ nonisosceles_triangle w))
  (tangent: ∃ (tang : tangent), touch alcircle w B (altitude_AD.incenter_I angle_bisector_BE angle_bisector_CF_ibisector AI)

   : touches_bisector_AI = ∀ A B C: triangle_AB, nonisosceles → touches. sorry

end circumcircle_touches_bisector_l170_170987


namespace pirate_coins_total_l170_170378

def total_coins (y : ℕ) := 6 * y

theorem pirate_coins_total : 
  (∃ y : ℕ, y ≠ 0 ∧ y * (y + 1) / 2 = 5 * y) →
  total_coins 9 = 54 :=
by
  sorry

end pirate_coins_total_l170_170378


namespace sum_of_literally_1434_numbers_l170_170108

theorem sum_of_literally_1434_numbers : ∑ n in { 
   n | ∃ (a b c d : ℕ), 
   n = 1000 * a + 100 * b + 10 * c + d ∧
   (0 < a) ∧ (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧
   a % 5 = 1 ∧ 
   b % 5 = 4 ∧ 
   c % 5 = 3 ∧ 
   d % 5 = 4 
}, n = 67384 := 
by {
   sorry
}

end sum_of_literally_1434_numbers_l170_170108


namespace average_mileage_correct_l170_170924

def total_distance : ℕ := 150 * 2
def sedan_mileage : ℕ := 25
def hybrid_mileage : ℕ := 50
def sedan_gas_used : ℕ := 150 / sedan_mileage
def hybrid_gas_used : ℕ := 150 / hybrid_mileage
def total_gas_used : ℕ := sedan_gas_used + hybrid_gas_used
def average_gas_mileage : ℚ := total_distance / total_gas_used

theorem average_mileage_correct :
  average_gas_mileage = 33 + 1 / 3 :=
by
  sorry

end average_mileage_correct_l170_170924


namespace line_through_midpoint_intersects_hyperbola_l170_170499

theorem line_through_midpoint_intersects_hyperbola 
  (A B : Point) 
  (M : Point) 
  (hM : M = midpoint A B) 
  (h1 : (1, 1) = M)
  (h2 : (A.x^2 / 4) - (A.y^2 / 3) = 1) 
  (h3 : (B.x^2 / 4) - (B.y^2 / 3) = 1) :
  ∃ k, k = 4 - 3 * x - 1 := 
sorry

end line_through_midpoint_intersects_hyperbola_l170_170499


namespace anie_days_to_finish_task_l170_170443

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l170_170443


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170023

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170023


namespace cosine_expression_rewrite_l170_170748

theorem cosine_expression_rewrite (x : ℝ) :
  ∃ a b c d : ℕ, 
    a * (Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) = 
    Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (14 * x) + Real.cos (18 * x) 
    ∧ a + b + c + d = 22 := sorry

end cosine_expression_rewrite_l170_170748


namespace product_of_consecutive_integers_sqrt_50_l170_170789

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170789


namespace probability_not_exceed_60W_l170_170436

noncomputable def total_bulbs : ℕ := 250
noncomputable def bulbs_100W : ℕ := 100
noncomputable def bulbs_60W : ℕ := 50
noncomputable def bulbs_25W : ℕ := 50
noncomputable def bulbs_15W : ℕ := 50

noncomputable def probability_of_event (event : ℕ) (total : ℕ) : ℝ := 
  event / total

noncomputable def P_A : ℝ := probability_of_event bulbs_60W total_bulbs
noncomputable def P_B : ℝ := probability_of_event bulbs_25W total_bulbs
noncomputable def P_C : ℝ := probability_of_event bulbs_15W total_bulbs
noncomputable def P_D : ℝ := probability_of_event bulbs_100W total_bulbs

theorem probability_not_exceed_60W : 
  P_A + P_B + P_C = 3 / 5 :=
by
  sorry

end probability_not_exceed_60W_l170_170436


namespace bus_speed_including_stoppages_approx_l170_170566

-- Define the given conditions
def speed_without_stoppages : ℝ := 82
def stoppage_time_per_hour_minutes : ℝ := 5.12

-- Calculate running time in hours
def running_time_per_hour_hours : ℝ := (60 - stoppage_time_per_hour_minutes) / 60

-- Define the condition for the resultant speed
def speed_with_stoppages : ℝ := speed_without_stoppages * running_time_per_hour_hours

theorem bus_speed_including_stoppages_approx :
  abs (speed_with_stoppages - 75) < 1 :=
sorry

end bus_speed_including_stoppages_approx_l170_170566


namespace sum_of_center_coordinates_eq_four_l170_170099

theorem sum_of_center_coordinates_eq_four :
  (∃ (h : ℝ → ℝ → ℝ), (∀ x y, h x y = x ^ 2 + y ^ 2 + 4 * x - 12 * y + 20) →
  (∀ c ∈ { (x, y) | h x y = 0 },
    (fst c + snd c) = 4)) :=
sorry

end sum_of_center_coordinates_eq_four_l170_170099


namespace range_of_s_l170_170469

def s (x : ℝ) : ℝ := 1 / (2 - x)^2

theorem range_of_s : set.Ioo 0 ∞ = set.range s :=
sorry

end range_of_s_l170_170469


namespace parabola_focus_l170_170184

theorem parabola_focus (x f : ℝ) (hx : ∀ x : ℝ, (x, 2 * x^2) ∈ set_of (λ p, p.snd = 2 * (p.fst)^2)) :
  (0, f) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_l170_170184


namespace ellipse_foci_y_axis_range_l170_170221

theorem ellipse_foci_y_axis_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1) ↔ (m < -1 ∨ (1 < m ∧ m < 3 / 2)) :=
sorry

end ellipse_foci_y_axis_range_l170_170221


namespace triangle_cos_b_lt_sin_b_l170_170647

theorem triangle_cos_b_lt_sin_b (A B : ℝ) (h1 : 0 < A) (h2 : A < 45) (h3 : B = 90 - A) :
  cos B < sin B := by
  sorry

end triangle_cos_b_lt_sin_b_l170_170647


namespace triangular_square_is_triangular_l170_170580

-- Definition of a triangular number
def is_triang_number (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) / 2

-- The main theorem statement
theorem triangular_square_is_triangular :
  ∃ x : ℕ, 
    is_triang_number x ∧ 
    is_triang_number (x * x) :=
sorry

end triangular_square_is_triangular_l170_170580


namespace motel_rent_equivalence_l170_170079

noncomputable def motel_total_rent (R40 R60 : ℕ) : ℕ :=
  40 * R40 + 60 * R60

theorem motel_rent_equivalence
  (R40 R60 : ℕ)
  (h1 : 40 * (R40 + 10) + 60 * (R60 - 10) = 0.6 * (40 * R40 + 60 * R60)) :
  motel_total_rent R40 R60 = 500 := by
  sorry

end motel_rent_equivalence_l170_170079


namespace fraction_product_l170_170942

theorem fraction_product :
  (7 / 4 : ℚ) * (14 / 35) * (21 / 12) * (28 / 56) * (49 / 28) * (42 / 84) * (63 / 36) * (56 / 112) = (1201 / 12800) := 
by
  sorry

end fraction_product_l170_170942


namespace find_the_letters_l170_170854

def letters := ['L', 'O', 'T']
def probability_of_lot (l : List Char) : Prop :=
  (l = ['L', 'O', 'T'] ∨ l = ['L', 'T', 'O'] ∨ l = ['O', 'L', 'T'] ∨
   l = ['O', 'T', 'L'] ∨ l = ['T', 'L', 'O'] ∨ l = ['T', 'O', 'L']) →
  1 / (l.length.factorial).toFloat = 0.16666666666666666

theorem find_the_letters : letters → probability_of_lot letters :=
sorry

end find_the_letters_l170_170854


namespace matt_age_l170_170650

-- Let M be Matt's current age
variable (M : ℕ)

-- Define the conditions
def kaylees_current_age : ℕ := 8
def kaylees_future_age : ℕ := kaylees_current_age + 7

-- Condition stating that Kaylee's age in 7 years will be 3 times Matt's current age
def condition_1 : Prop := kaylees_future_age = 3 * M

-- The proof to be constructed (statement only)
theorem matt_age : condition_1 → M = 5 := by
  sorry

end matt_age_l170_170650


namespace no_real_roots_iff_k_gt_2_l170_170197

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l170_170197


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170017

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170017


namespace segment_PM_half_perimeter_l170_170592

variable {α : Type _} [LinearOrderedField α] {A B C M P: Point α}

noncomputable def isTriangle (A B C : Point α) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

noncomputable def arePerpendiculars (A B C M P : Point α) : Prop :=
  -- Definition capturing AM and AP being perpendiculars to external angle bisectors 
  sorry

theorem segment_PM_half_perimeter {A B C M P : Point α} 
  (h1 : isTriangle A B C) 
  (h2 : arePerpendiculars A B C M P) 
  : distance M P = (distance A B + distance B C + distance C A) / 2 :=
sorry

end segment_PM_half_perimeter_l170_170592


namespace statement1_statement2_statement3_statement4_l170_170235

variable {A : Set}

def p : Prop := A ∩ ∅ = ∅
def q : Prop := A ∪ ∅ = A

theorem statement1 (hp : p) (hq : q) : p ∧ q :=
by sorry

theorem statement2 (hp : p) (hq : q) : ¬p ∨ ¬q = false :=
by sorry

theorem statement3 (hp : p) (hq : q) : ¬p ∨ q :=
by sorry

theorem statement4 (hp : p) (hq : q) : ¬p ∧ q = false :=
by sorry

end statement1_statement2_statement3_statement4_l170_170235


namespace proposition_C_is_correct_l170_170064

theorem proposition_C_is_correct :
  ∃ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4) :=
by
  sorry

end proposition_C_is_correct_l170_170064


namespace irrational_count_l170_170931

def is_rational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬is_rational x

def num_irrational_numbers : ℕ :=
  let numbers := [15/7, 3.14, 0, Real.sqrt 9, Real.pi, Real.sqrt 5, 0.1010010001] in
  List.countp is_irrational numbers

theorem irrational_count : num_irrational_numbers = 3 :=
  sorry

end irrational_count_l170_170931


namespace function_properties_l170_170355

-- Definitions based on provided conditions
variable {ℝ : Type*} [LinearOrderedField ℝ]
variable (f : ℝ → ℝ)
variable (h1 : ∀ x : ℝ, f (10 + x) = f (10 - x))
variable (h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x))

-- The Lean 4 statement to be proved
theorem function_properties
  (h1 : ∀ x : ℝ, f (10 + x) = f (10 - x))
  (h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x) = f (x + 40)) :=
sorry

end function_properties_l170_170355


namespace maximum_valid_subset_size_l170_170344

-- Define the subset T and its properties
def is_valid_subset (T : finset ℕ) : Prop :=
  T ⊆ finset.range 60 ∧
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

-- Define the statement to be proven
theorem maximum_valid_subset_size : 
  ∃ T : finset ℕ, is_valid_subset T ∧ T.card = 36 :=
sorry

end maximum_valid_subset_size_l170_170344


namespace additional_interest_rate_l170_170393

variable (P A1 A2 T SI1 SI2 R AR : ℝ)
variable (h_P : P = 9000)
variable (h_A1 : A1 = 10200)
variable (h_A2 : A2 = 10740)
variable (h_T : T = 3)
variable (h_SI1 : SI1 = A1 - P)
variable (h_SI2 : SI2 = A2 - A1)
variable (h_R : SI1 = P * R * T / 100)
variable (h_AR : SI2 = P * AR * T / 100)

theorem additional_interest_rate :
  AR = 2 := by
  sorry

end additional_interest_rate_l170_170393


namespace exactly_one_solves_problem_l170_170118

theorem exactly_one_solves_problem (pA pB pC : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  (pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC) = 11 / 24 :=
by
  sorry

end exactly_one_solves_problem_l170_170118


namespace odd_function_with_period_l170_170535

-- Define the given functions
def f (x : Real) : Real := sin (2 * x) + cos (2 * x)
def g (x : Real) : Real := sin (4 * x + π / 2)
def h (x : Real) : Real := sin (2 * x) * cos (2 * x)
def k (x : Real) : Real := sin (2 * x) ^ 2 - cos (2 * x) ^ 2

-- Prove that h(x) is odd and has a period of π/2
theorem odd_function_with_period :
  Function.Odd h ∧ (∀ x, h (x + π / 2) = h x) ∧ (∀ p > 0, (∀ x, h (x + p) = h x) → p = π / 2) := by
  sorry

end odd_function_with_period_l170_170535


namespace necessary_and_sufficient_condition_l170_170674

variables {A B C D E T P : Type} [point_geometry A B C D E T P]

-- Conditions
variables (triangle_ABC_eq : AB = AC)
          (D_on_AB : D ∈ segment AB)
          (E_on_extension_AC : E ∈ segment_extension AC)
          (DE_eq_AC : DE = AC)
          (T_on_circumcircle_ABC : T ∈ circumcircle ABC)
          (P_on_extension_AT : P ∈ segment_extension AT)

-- Prove: necessary and sufficient condition for PD + PE = AT is P lies on circumcircle ADE
theorem necessary_and_sufficient_condition 
  (h1 : AB = AC)
  (h2 : D ∈ segment AB)
  (h3 : E ∈ segment_extension AC)
  (h4 : DE = AC)
  (h5 : DE ∩ circumcircle ABC = T)
  (h6 : P ∈ segment_extension AT) :
  (PD + PE = AT) ↔ P ∈ circumcircle ADE :=
sorry

end necessary_and_sufficient_condition_l170_170674


namespace four_equal_angles_in_convex_heptagon_l170_170208

theorem four_equal_angles_in_convex_heptagon
  (heptagon : List ℝ) (h_len : heptagon.length = 7)
  (h_valid : ∀ x, x ∈ heptagon → 0 < x ∧ x < π)
  (sum_const : ∀ (sin_indices : Finset ℕ), sin_indices.card = 4 →
    let cos_indices := (Finset.range 7).erase sin_indices.univ.to_list
    (Finset.sum sin_indices (λ i, Real.sin (heptagon.nth_le i (by linarith)))
     + Finset.sum cos_indices (λ i, Real.cos (heptagon.nth_le i (by linarith))) = c))
  : ∃ (a : ℝ), ∃ (count : ℕ), count ≥ 4 ∧ count = (List.filter (λ x, x = a) heptagon).length :=
begin
  sorry
end

end four_equal_angles_in_convex_heptagon_l170_170208


namespace sequence_property_l170_170989

theorem sequence_property (p : ℕ → ℕ)
  (h : ∀ n, p n = number_of_sequences_without_aaaa_and_bbb n) :
  (p 2004 - p 2002 - p 1999) / (p 2000 + p 2001) = 1 := 
sorry

-- Define the condition function
def number_of_sequences_without_aaaa_and_bbb : ℕ → ℕ := sorry

end sequence_property_l170_170989


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170040

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170040


namespace polynomial_remainder_l170_170978

-- Define the polynomials
def f : Polynomial ℝ := X^100
def g : Polynomial ℝ := (X^2 + 1) * (X - 1)

-- Statement of the problem
theorem polynomial_remainder : (f % g) = 1 := 
by sorry

end polynomial_remainder_l170_170978


namespace number_of_real_fifth_powers_l170_170955

theorem number_of_real_fifth_powers (z : ℂ) (hz : z^30 = 1) : 
  (nat.card {w : ℂ // w^30 = 1 ∧ (w^5).re ∈ {1, -1}}) = 11 :=
sorry

end number_of_real_fifth_powers_l170_170955


namespace complex_square_l170_170257

theorem complex_square (a b : ℤ) (i : ℂ) (h1: a = 5) (h2: b = 3) (h3: i^2 = -1) :
  ((↑a) + (↑b) * i)^2 = 16 + 30 * i := by
  sorry

end complex_square_l170_170257


namespace sqrt_50_product_consecutive_integers_l170_170807

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170807


namespace ab_proof_l170_170361

theorem ab_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 90 < a + b) (h4 : a + b < 99) 
  (h5 : 0.9 < (a : ℝ) / b) (h6 : (a : ℝ) / b < 0.91) : a * b = 2346 :=
sorry

end ab_proof_l170_170361


namespace units_digit_probability_l170_170921

noncomputable def probability_units_digit_less_than_seven : ℚ :=
  let favorable_outcomes := 7
  let total_possible_outcomes := 10
  favorable_outcomes / total_possible_outcomes

theorem units_digit_probability :
  probability_units_digit_less_than_seven = 7 / 10 :=
sorry

end units_digit_probability_l170_170921


namespace sum_of_roots_l170_170055

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l170_170055


namespace phase_shift_of_sine_function_l170_170548

theorem phase_shift_of_sine_function :
  let y := λ x : ℝ, 3 * Real.sin (4 * x + (Real.pi / 4)) in
  ∃ (shift : ℝ), shift = -Real.pi / 16 := by
  let A := 3
  let B := 4
  let C := Real.pi / 4
  let phase_shift := -C / B
  use phase_shift
  sorry

end phase_shift_of_sine_function_l170_170548


namespace smallest_perfect_cube_divisor_l170_170709

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  ∃ (a b c : ℕ), a = 6 ∧ b = 6 ∧ c = 6 ∧ (p^a * q^b * r^c) = (p^2 * q^2 * r^2)^3 ∧ 
  (p^a * q^b * r^c) % (p^2 * q^3 * r^4) = 0 := 
by
  sorry

end smallest_perfect_cube_divisor_l170_170709


namespace find_smallest_palindrome_addition_l170_170489

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

theorem find_smallest_palindrome_addition :
  ∃ n : ℕ, 134782 + n = 135531 ∧ is_palindrome (134782 + n) ∧ ∀ m : ℕ, is_palindrome (134782 + m) → 134782 < 134782 + n ∧ 134782 + n ≤ 134782 + m :=
begin
  existsi 749,
  split,
  { refl },
  split,
  { exact dec_trivial },
  { intros m hm,
    sorry }
end

end find_smallest_palindrome_addition_l170_170489


namespace total_swim_distance_five_weeks_total_swim_time_five_weeks_l170_170331

-- Definitions of swim distances and times based on Jasmine's routine 
def monday_laps : ℕ := 10
def tuesday_laps : ℕ := 15
def tuesday_aerobics_time : ℕ := 20
def wednesday_laps : ℕ := 12
def wednesday_time_per_lap : ℕ := 2
def thursday_laps : ℕ := 18
def friday_laps : ℕ := 20

-- Proving total swim distance for five weeks
theorem total_swim_distance_five_weeks : (5 * (monday_laps + tuesday_laps + wednesday_laps + thursday_laps + friday_laps)) = 375 := 
by 
  sorry

-- Proving total swim time for five weeks (partially solvable)
theorem total_swim_time_five_weeks : (5 * (tuesday_aerobics_time + wednesday_laps * wednesday_time_per_lap)) = 220 := 
by 
  sorry

end total_swim_distance_five_weeks_total_swim_time_five_weeks_l170_170331


namespace club_population_after_five_years_l170_170495

noncomputable def a : ℕ → ℕ
| 0     => 18
| (n+1) => 3 * (a n - 5) + 5

theorem club_population_after_five_years : a 5 = 3164 := by
  sorry

end club_population_after_five_years_l170_170495


namespace palindromic_square_iff_l170_170917

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_square_iff (k : ℕ) (n : ℕ) (a : Fin n → ℕ) (h₁ : k = ∑ i, a i * 10^i) (h₂ : is_palindromic k) :
  is_palindromic (k^2) ↔ (∑ i, (a i)^2) < 10 :=
sorry

end palindromic_square_iff_l170_170917


namespace area_of_triangle_eq_l170_170515

-- Define the problem statement and conditions
def octagon_side_length : ℝ := 3
def cosine_135 : ℝ := - (1 / Real.sqrt 2)
def area_triangle_ADG : ℝ := (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2)) / (2 * Real.sqrt 2)

-- Use noncomputable to define the area based on the conditions
noncomputable def area_of_triangle : ℝ :=
  1 / (2 * Real.sqrt 2) * (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2))

-- Claim about the area of the triangle
theorem area_of_triangle_eq :
  area_of_triangle = area_triangle_ADG :=
sorry

end area_of_triangle_eq_l170_170515


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170029

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170029


namespace total_biscuits_l170_170082

-- Define the number of dogs and biscuits per dog
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem stating the total number of biscuits needed
theorem total_biscuits : num_dogs * biscuits_per_dog = 6 := by
  -- sorry to skip the proof
  sorry

end total_biscuits_l170_170082


namespace count_C_sets_l170_170999

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

-- The predicate that a set C satisfies B ∪ C = A
def satisfies_condition (C : Set ℕ) : Prop := B ∪ C = A

-- The claim that there are exactly 4 such sets C
theorem count_C_sets : 
  ∃ (C1 C2 C3 C4 : Set ℕ), 
    (satisfies_condition C1 ∧ satisfies_condition C2 ∧ satisfies_condition C3 ∧ satisfies_condition C4) 
    ∧ 
    (∀ C', satisfies_condition C' → C' = C1 ∨ C' = C2 ∨ C' = C3 ∨ C' = C4)
    ∧ 
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4) := 
sorry

end count_C_sets_l170_170999


namespace chocolates_distribution_l170_170129

theorem chocolates_distribution (boys girls : ℕ) (h_boys : boys = 60) (h_girls : girls = 60) :
  boys + girls = 120 :=
by
  rw [h_boys, h_girls]
  exact Nat.add_eq_add_of_eq rfl rfl

end chocolates_distribution_l170_170129


namespace milk_remaining_and_total_weight_correct_l170_170676

section MilkProblem

-- Define the initial conditions
def whole_milk_gallons : ℚ := 3
def skim_milk_gallons : ℚ := 2
def almond_milk_gallons : ℚ := 1

def whole_milk_drank : ℚ := 13
def skim_milk_drank : ℚ := 20
def almond_milk_drank : ℚ := 25

def ounces_in_gallon : ℚ := 128

def whole_milk_weight : ℚ := 8.6
def skim_milk_weight : ℚ := 8.4
def almond_milk_weight : ℚ := 8.3

-- Define the expected outcomes
def remaining_whole_milk : ℚ := 371
def remaining_skim_milk : ℚ := 236
def remaining_almond_milk : ℚ := 103

def total_weight_approx : ℚ := 47.09296875

-- The proof statement
theorem milk_remaining_and_total_weight_correct :
  remaining_whole_milk = (whole_milk_gallons * ounces_in_gallon - whole_milk_drank) ∧
  remaining_skim_milk = (skim_milk_gallons * ounces_in_gallon - skim_milk_drank) ∧
  remaining_almond_milk = (almond_milk_gallons * ounces_in_gallon - almond_milk_drank) ∧ 
  total_weight_approx ≈ ((remaining_whole_milk / ounces_in_gallon * whole_milk_weight) +
                        (remaining_skim_milk / ounces_in_gallon * skim_milk_weight) +
                        (remaining_almond_milk / ounces_in_gallon * almond_milk_weight)) :=
by sorry

end MilkProblem

end milk_remaining_and_total_weight_correct_l170_170676


namespace right_triangle_side_length_l170_170985

theorem right_triangle_side_length (x : ℝ) (hx : x > 0) (h_area : (1 / 2) * x * (3 * x) = 108) :
  x = 6 * Real.sqrt 2 :=
sorry

end right_triangle_side_length_l170_170985


namespace initial_fraction_of_larger_jar_l170_170964

theorem initial_fraction_of_larger_jar (S L W : ℝ) 
  (h1 : W = 1/6 * S) 
  (h2 : W = 1/3 * L) : 
  W / L = 1 / 3 := 
by 
  sorry

end initial_fraction_of_larger_jar_l170_170964


namespace sqrt_50_between_7_and_8_l170_170802

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170802


namespace unit_prices_max_books_l170_170096

-- Definitions based on conditions 1 and 2
def unit_price_A (x : ℝ) : Prop :=
  x > 5 ∧ (1200 / x = 900 / (x - 5))

-- Definitions based on conditions 3, 4, and 5
def max_books_A (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 300 ∧ 0.9 * 20 * y + 15 * (300 - y) ≤ 5100

theorem unit_prices
  (x : ℝ)
  (h : unit_price_A x) :
  x = 20 ∧ x - 5 = 15 :=
sorry

theorem max_books
  (y : ℝ)
  (hy : max_books_A y) :
  y ≤ 200 :=
sorry

end unit_prices_max_books_l170_170096


namespace product_of_consecutive_integers_sqrt_50_l170_170778

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170778


namespace joe_and_dad_total_marshmallows_roasted_l170_170683

theorem joe_and_dad_total_marshmallows_roasted :
  (let dads_marshmallows := 21
       dads_roasted := dads_marshmallows / 3
       joes_marshmallows := 4 * dads_marshmallows
       joes_roasted := joes_marshmallows / 2
   in dads_roasted + joes_roasted = 49) :=
by
  let dads_marshmallows := 21
  let dads_roasted := dads_marshmallows / 3
  let joes_marshmallows := 4 * dads_marshmallows
  let joes_roasted := joes_marshmallows / 2
  show dads_roasted + joes_roasted = 49 from sorry

end joe_and_dad_total_marshmallows_roasted_l170_170683


namespace total_toys_is_60_l170_170327

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l170_170327


namespace sum_of_roots_l170_170049

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l170_170049


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170043

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170043


namespace parabola_distance_P_to_F_l170_170603

variables {t : ℝ} (P : ℝ × ℝ) (F : ℝ × ℝ)

-- Definitions for conditions
def parabola_param (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)
def P := (3 : ℝ, 2 * Real.sqrt 3)
def F := (1 : ℝ, 0 : ℝ)
def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- The proof statement
theorem parabola_distance_P_to_F : distance P F = 4 := by
  sorry

end parabola_distance_P_to_F_l170_170603


namespace home_electronics_budget_l170_170098

theorem home_electronics_budget (deg_ba: ℝ) (b_deg: ℝ) (perc_me: ℝ) (perc_fa: ℝ) (perc_gm: ℝ) (perc_il: ℝ) : 
  deg_ba = 43.2 → 
  b_deg = 360 → 
  perc_me = 12 →
  perc_fa = 15 →
  perc_gm = 29 →
  perc_il = 8 →
  (b_deg / 360 * 100 = 12) → 
  perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100) = 76 →
  100 - (perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100)) = 24 :=
by
  intro h_deg_ba h_b_deg h_perc_me h_perc_fa h_perc_gm h_perc_il h_ba_12perc h_total_76perc
  sorry

end home_electronics_budget_l170_170098


namespace square_side_length_l170_170765

theorem square_side_length (A : ℝ) (h : A = 9) : ∃ s : ℝ, s = real.sqrt 9 ∧ A = s * s :=
by
  use real.sqrt 9
  split
  { refl }
  { rw h
    exact real.sqrt_mul_self_eq 9 }
-- sorry

end square_side_length_l170_170765


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170033

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170033


namespace find_y_l170_170314

-- Definitions of the given conditions
def is_straight_line (A B : Point) : Prop := 
  ∃ C D, A ≠ C ∧ B ≠ D

def angle (A B C : Point) : ℝ := sorry -- Assume angle is a function providing the angle in degrees

-- The proof problem statement
theorem find_y
  (A B C D X Y Z : Point)
  (hAB : is_straight_line A B)
  (hCD : is_straight_line C D)
  (hAXB : angle A X B = 180) 
  (hYXZ : angle Y X Z = 70)
  (hCYX : angle C Y X = 110) :
  angle X Y Z = 40 :=
sorry

end find_y_l170_170314


namespace problem1_problem2_l170_170142

theorem problem1 : sqrt 6 + sqrt 8 * sqrt 12 = 5 * sqrt 6 := 
by
  sorry

theorem problem2 : sqrt 4 - (sqrt 2 / (sqrt 2 + 1)) = sqrt 2 := 
by
  sorry

end problem1_problem2_l170_170142


namespace correct_statement_l170_170872

theorem correct_statement :
  (The statement "the coefficient of the monomial a is 1" is true) :=
sorry

end correct_statement_l170_170872


namespace area_of_shaded_region_is_correct_l170_170918

-- Define the conditions: side length and the formula for the diagonal
def side_length := 8
def diagonal_length (s : ℕ) : ℝ := s * (1 + Real.sqrt 2)

-- Define the area of the square formed by the diagonals
def square_area (d : ℝ) : ℝ := d * d

-- The theorem statement to prove
theorem area_of_shaded_region_is_correct :
  square_area (diagonal_length side_length) = 192 + 128 * Real.sqrt 2 :=
by
  sorry

end area_of_shaded_region_is_correct_l170_170918


namespace polyhedron_space_diagonals_l170_170907

theorem polyhedron_space_diagonals (V E F T P : ℕ) (total_pairs_of_vertices total_edges total_face_diagonals : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 40)
  (hT : T = 30)
  (hP : P = 10)
  (h_total_pairs_of_vertices : total_pairs_of_vertices = 30 * 29 / 2)
  (h_total_face_diagonals : total_face_diagonals = 5 * 10)
  :
  total_pairs_of_vertices - E - total_face_diagonals = 315 := 
by
  sorry

end polyhedron_space_diagonals_l170_170907


namespace product_value_l170_170138

noncomputable def product_expression : ℚ :=
  ∏ n in finset.range 15 + 1, (n * (n + 1)^2 * (n + 2)) / ((n + 4)^3)

theorem product_value :
  product_expression = 50625 / 543339776 :=
by
  sorry

end product_value_l170_170138


namespace survey_respondents_l170_170375

theorem survey_respondents
  (X Y Z : ℕ) 
  (h1 : X = 360) 
  (h2 : X * 4 = Y * 9) 
  (h3 : X * 3 = Z * 9) : 
  X + Y + Z = 640 :=
by
  sorry

end survey_respondents_l170_170375


namespace quadratic_roots_l170_170555

theorem quadratic_roots (a : ℝ) (k c : ℝ) : 
    (∀ x : ℝ, 2 * x^2 + k * x + c = 0 ↔ (x = 7 ∨ x = a)) →
    k = -2 * a - 14 ∧ c = 14 * a :=
by
  sorry

end quadratic_roots_l170_170555


namespace final_number_divisible_by_1985_l170_170479

theorem final_number_divisible_by_1985:
  let seq := (List.range 1986) in
  let summed_seq := List.sum <$> List.iterate (λ l, (l.tail.zip l).map (λ ⟨x, y⟩, x + y)) seq 1986 in
  (summed_seq.last).get_or_else 0 % 1985 = 0 :=
sorry

end final_number_divisible_by_1985_l170_170479


namespace product_of_consecutive_integers_sqrt_50_l170_170818

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170818


namespace monotonic_increasing_interval_of_function_l170_170421

noncomputable def increasing_interval (y : ℝ → ℝ) : set ℝ :=
  {x : ℝ | ∃ k : ℤ, x ∈ set.Icc (real.pi / 2 + 2 * k * real.pi) (3 * real.pi / 2 + 2 * k * real.pi)}

theorem monotonic_increasing_interval_of_function :
  increasing_interval (λ x, 3 - 2 * real.sin x) = {x : ℝ | ∃ k : ℤ, x ∈ set.Icc (real.pi / 2 + 2 * k * real.pi) (3 * real.pi / 2 + 2 * k * real.pi)} :=
by
  sorry

end monotonic_increasing_interval_of_function_l170_170421


namespace number_of_polynomials_l170_170957

def polynomial_of_form (a : ℕ → ℤ) (n : ℕ) : Prop :=
  n + ∑ i in Finset.range (n + 1), (a i).natAbs = 5

theorem number_of_polynomials : 
  (∑ n in Finset.range 5, if polynomial_of_form (λ i, 5 - i) n then 1 else 0) = 30 :=
sorry

end number_of_polynomials_l170_170957


namespace smallest_positive_number_div_conditions_is_perfect_square_l170_170979

theorem smallest_positive_number_div_conditions_is_perfect_square :
  ∃ n : ℕ,
    (n % 11 = 10) ∧
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    (∃ k : ℕ, n = k * k) ∧
    n = 2782559 :=
by
  sorry

end smallest_positive_number_div_conditions_is_perfect_square_l170_170979


namespace read_book_in_n_days_l170_170480

variable (total_pages : ℝ)
variable (pages_per_night : ℝ)
variable (days : ℝ)

theorem read_book_in_n_days (h1 : total_pages = 1200) (h2 : pages_per_night = 120.0) : days = 10 :=
by
  have h3 : days = total_pages / pages_per_night := by sorry
  rw [h1, h2] at h3
  exact h3

end read_book_in_n_days_l170_170480


namespace polynomial_characterization_l170_170164

noncomputable def conditions (n : ℕ) (a : fin (n+1) → ℝ) : Prop :=
  let f (x : ℝ) := ∑ i : fin (n+1), a i * x^(2*i.1)
  let roots_purely_imaginary (f : ℝ → ℝ) : Prop :=
    ∃ β : fin n → ℝ, (∀ i, β i > 0) ∧ ∀ x, f x = 0 ↔ ∃ i, x = ⟪0, β i⟫ ∨ x = ⟪0, -β i⟫
  a 0 > 0 ∧
  ∑ j in fin.range (n+1), a j * a (⟨2*n - 2*j.1, by ...⟩) ≤ nat.factorial (2*n) / (nat.factorial n)^2 * a 0 * a n ∧
  roots_purely_imaginary f

theorem polynomial_characterization (n : ℕ) (a : fin (n+1) → ℝ) (r : ℝ) (a0 : ℝ) :
  conditions n a →
  ∃ r : ℝ, r > 0 ∧ (∏ i in fin.range n, (X^2 + r)) = ∑ i : fin (n+1), a i * X^(2*i.1) :=
sorry

end polynomial_characterization_l170_170164


namespace no_such_convex_quadrilateral_exists_l170_170325

-- Define what it means to be a convex quadrilateral
structure ConvexQuadrilateral (A B C D : ℝ) :=
(side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) 
(diag1 : ℝ) (diag2 : ℝ)
(is_convex : ∀ (angle : ℝ), angle ≥ 0 ∧ angle ≤ 180)
(diagonal_condition : diag1 ≤ side1 ∧ diag1 ≤ side2 ∧ diag1 ≤ side3 ∧ diag1 ≤ side4 ∧
                      diag2 ≤ side1 ∧ diag2 ≤ side2 ∧ diag2 ≤ side3 ∧ diag2 ≤ side4)

-- Prove that such a convex quadrilateral does not exist
theorem no_such_convex_quadrilateral_exists: 
  ¬ ∃ (A B C D : ℝ), ConvexQuadrilateral A B C D :=
by
  sorry

end no_such_convex_quadrilateral_exists_l170_170325


namespace product_of_consecutive_integers_sqrt_50_l170_170820

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170820


namespace parabola_focus_l170_170186

-- Define the condition of the parabola y = 2x^2
def is_parabola (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, p x = 2 * x^2

-- Define the condition for the focus
def is_focus (p : ℝ → ℝ) (f : ℝ × ℝ) : Prop :=
  f = (0, 1 / 8)

-- Define the main theorem to state and prove focus of the parabola
theorem parabola_focus : 
  ∃ f : ℝ × ℝ, is_parabola (λ x, 2 * x^2) ∧ is_focus (λ x, 2 * x^2) f :=
begin
  -- Proof is omitted
  sorry
end

end parabola_focus_l170_170186


namespace probability_no_more_than_five_girls_between_first_and_last_boys_l170_170860

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else factorial n / (factorial k * factorial (n - k))

theorem probability_no_more_than_five_girls_between_first_and_last_boys :
  let total_arrangements := binom 20 9 in
  let arrangements_with_first_last_within_14 := binom 14 9 in
  let arrangements_with_last_at_specific_positions := 6 * binom 13 8 in
  let favorable_arrangements := arrangements_with_first_last_within_14 + arrangements_with_last_at_specific_positions in
  let probability := favorable_arrangements / total_arrangements in
  probability = 9724 / 167960 :=
by
  sorry

end probability_no_more_than_five_girls_between_first_and_last_boys_l170_170860


namespace students_scoring_80_percent_l170_170095

theorem students_scoring_80_percent
  (x : ℕ)
  (h1 : 10 * 90 + x * 80 = 25 * 84)
  (h2 : x + 10 = 25) : x = 15 := 
by {
  -- Proof goes here
  sorry
}

end students_scoring_80_percent_l170_170095


namespace exists_non_degenerate_triangle_l170_170865

theorem exists_non_degenerate_triangle
  (l : Fin 7 → ℝ)
  (h_ordered : ∀ i j, i ≤ j → l i ≤ l j)
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) :
  ∃ i j k : Fin 7, i < j ∧ j < k ∧ l i + l j > l k ∧ l j + l k > l i ∧ l k + l i > l j := 
sorry

end exists_non_degenerate_triangle_l170_170865


namespace sin_squared_minus_cos_squared_l170_170610

theorem sin_squared_minus_cos_squared {α : ℝ} (h : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 :=
by
  sorry -- Proof is omitted

end sin_squared_minus_cos_squared_l170_170610


namespace line_in_first_and_third_quadrants_l170_170266

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170266


namespace product_of_consecutive_integers_between_sqrt_50_l170_170825

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170825


namespace unit_vector_orthogonal_l170_170567

def u : ℝ × ℝ × ℝ := (2, 2, 0)
def v : ℝ × ℝ × ℝ := (2, 0, 4)
def w : ℝ × ℝ × ℝ := (2/3, -2/3, -1/3)
def w' : ℝ × ℝ × ℝ := (-2/3, 2/3, 1/3)

theorem unit_vector_orthogonal (u v w w' : ℝ × ℝ × ℝ) :
  (w.1 * u.1 + w.2 * u.2 + w.3 * u.3 = 0) ∧ 
  (w.1 * v.1 + w.2 * v.2 + w.3 * v.3 = 0) ∧ 
  (w'.1 * u.1 + w'.2 * u.2 + w'.3 * u.3 = 0) ∧ 
  (w'.1 * v.1 + w'.2 * v.2 + w'.3 * v.3 = 0) :=
sorry

end unit_vector_orthogonal_l170_170567


namespace average_without_ivan_l170_170583

theorem average_without_ivan
  (total_friends : ℕ := 5)
  (avg_all : ℝ := 55)
  (ivan_amount : ℝ := 43)
  (remaining_friends : ℕ := total_friends - 1)
  (total_amount : ℝ := total_friends * avg_all)
  (remaining_amount : ℝ := total_amount - ivan_amount)
  (new_avg : ℝ := remaining_amount / remaining_friends) :
  new_avg = 58 := 
sorry

end average_without_ivan_l170_170583


namespace unique_restore_triangle_l170_170097

axiom Circle (k : Type) (ABC : Triangle) : Prop
axiom OnCircle (A B C : Point) (k : Circle) : Prop
axiom OnSide (P A B : Point) (S : Side) : Prop
axiom Intersection (L1 L2 L3 : Line) (P : Point) : Prop
axiom UniqueTriangle (A B C A1 B1 C1 : Point) (k : Circle) (ABC : Triangle) : Prop

theorem unique_restore_triangle {k : Circle} {A B C A1 B1 C1 : Point} (ABC : Triangle) :
  (Circumscribed k ABC) →
  (OnSide A1 (BC ABC)) →
  (OnSide B1 (CA ABC)) →
  (OnSide C1 (AB ABC)) →
  (∃ (T : Point), Intersection (Line A A1) (Line B B1) (Line C C1) T) ↔
  UniqueTriangle A B C A1 B1 C1 k ABC :=
sorry

end unique_restore_triangle_l170_170097


namespace cos_minus_sin_value_l170_170214

-- Define the conditions
variables (α : Real)
def condition1 := sin α * cos α = 3 / 8
def condition2 := π / 4 < α ∧ α < π / 2

-- The theorem to be proved
theorem cos_minus_sin_value (h1 : condition1 α) (h2 : condition2 α) : cos α - sin α = -1 / 2 :=
by
  sorry

end cos_minus_sin_value_l170_170214


namespace range_s_l170_170458

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_s : set.range s = set.Ioi 0 := by
  sorry

end range_s_l170_170458


namespace tile_position_l170_170893

open Classical

theorem tile_position (x y : ℕ) (h1 : x > 0) (h2 : x ≤ 7) (h3 : y > 0) (h4 : y ≤ 7)
  (h5 : ∃ t : Finset (ℕ × ℕ), t.card = 16 ∧ ∀ p ∈ t, ∃ i j, p = (i, j) ∧ i % 1 = 0 ∧ j % 3 = 0)
  (h6 : ∃! p : ℕ × ℕ, p.1 = x ∧ p.2 = y ∧ p ∉ univ.image2 (λ i j, (i, j)) ({1, 4, 7} × {1, 4, 7})) :
  (x = 4 ∧ y = 4) ∨ (x = 1 ∨ x = 7 ∨ y = 1 ∨ y = 7) :=
sorry

end tile_position_l170_170893


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170025

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170025


namespace zero_point_in_interval_l170_170399

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_in_interval :
  (∃ c ∈ Ioo 1 2, f c = 0) :=
begin
  -- Define that f is a continuous and monotonically decreasing function
  have h_decreasing : ∀ x1 x2, x1 < x2 → f x1 > f x2,
  {
    -- Proof would go here (skipped)
    sorry
  },
  -- We are given f(1) > 0 and f(2) < 0
  have f_1_pos : f 1 > 0,
  { 
    -- Proof of f(1) >
    sorry
  },
  
  have f_2_neg : f 2 < 0,
  { 
    -- Proof of f(2) < 0
    sorry
  },
  
  -- From the intermediate value theorem for monotonically decreasing
  -- function, there exists a c in (1, 2) such that f(c) = 0
  sorry
end

end zero_point_in_interval_l170_170399


namespace arithmetic_mean_end_number_l170_170405

theorem arithmetic_mean_end_number (n : ℤ) :
  (100 + n) / 2 = 150 + 100 → n = 400 := by
  sorry

end arithmetic_mean_end_number_l170_170405


namespace students_chemistry_or_physics_not_both_l170_170177

variables (total_chemistry total_both total_physics_only : ℕ)

theorem students_chemistry_or_physics_not_both
  (h1 : total_chemistry = 30)
  (h2 : total_both = 15)
  (h3 : total_physics_only = 18) :
  total_chemistry - total_both + total_physics_only = 33 :=
by
  sorry

end students_chemistry_or_physics_not_both_l170_170177


namespace product_of_consecutive_integers_sqrt_50_l170_170842

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170842


namespace range_of_s_l170_170464

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_of_s : set.range s = set.Ioi 0 := 
by 
  sorry

end range_of_s_l170_170464


namespace rope_length_correct_l170_170110

noncomputable def rope_length : ℝ :=
  Real.sqrt ((4 * 380.132711084365) / Real.pi)

theorem rope_length_correct :
  rope_length ≈ 22 :=
by
  sorry

end rope_length_correct_l170_170110


namespace triangle_area_hyperbola_l170_170231

noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 24) = 1

def foci_F1 : ℝ × ℝ := (-5, 0)
def foci_F2 : ℝ × ℝ := (5, 0)

theorem triangle_area_hyperbola 
  (P : ℝ × ℝ)
  (on_hyperbola : hyperbola_eq P.1 P.2)
  (right_branch : P.1 > 0)
  (distance_relation : ∀ P : ℝ × ℝ, dist P foci_F1 = (4/3) * dist P foci_F2) :
  ∃ (area : ℝ), area = 24 :=
by
  sorry

end triangle_area_hyperbola_l170_170231


namespace tan_theta_value_l170_170253

theorem tan_theta_value (θ k : ℝ) 
  (h1 : Real.sin θ = (k + 1) / (k - 3)) 
  (h2 : Real.cos θ = (k - 1) / (k - 3)) 
  (h3 : (Real.sin θ ≠ 0) ∧ (Real.cos θ ≠ 0)) : 
  Real.tan θ = 3 / 4 := 
sorry

end tan_theta_value_l170_170253


namespace number_of_drolls_is_one_l170_170306

-- Define species
inductive Species
| trull : Species
| droll : Species
| mroll : Species

-- Define beings and their statements
structure Being (name : String) :=
(statement : String)
(species : Species)

-- The five beings and their statements
def Ann : Being "Ann" := ⟨"I am not the same species as Bob or Carl.", Species.mroll⟩
def Bob : Being "Bob" := ⟨"Eve is a droll.", Species.trull⟩
def Carl : Being "Carl" := ⟨"Dana is a trull.", Species.trull⟩
def Dana : Being "Dana" := ⟨"Of the five of us, more than two are trulls.", Species.trull⟩
def Eve : Being "Eve" := ⟨"Carl is a droll.", Species.droll⟩

-- State the problem: Prove that the number of drolls is 1
theorem number_of_drolls_is_one :
  (Ann.species = Species.droll) + (Bob.species = Species.droll) + (Carl.species = Species.droll) + 
  (Dana.species = Species.droll) + (Eve.species = Species.droll) = 1 := 
  by sorry

end number_of_drolls_is_one_l170_170306


namespace k_positive_first_third_quadrants_l170_170287

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170287


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170028

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170028


namespace random_graph_is_rado_l170_170435

noncomputable theory

open Classical
open_locale BigOperators

theorem random_graph_is_rado (G : Type) [Fintype G] (p : ℝ) (h : 0 < p ∧ p < 1)
  (H : G ∈ G𝔾 Fintype.card ℕ p) : 
  ∃ R : Type, G ≃ R :=
sorry

end random_graph_is_rado_l170_170435


namespace max_perimeter_triangle_l170_170527

theorem max_perimeter_triangle
  (x : ℤ) (h1 : 3 < x) (h2 : x < 19) :
  8 + 11 + x ≤ 37 :=
begin
  have h3 : x ≤ 18, {
    linarith,
  },
  linarith,
end

end max_perimeter_triangle_l170_170527


namespace sum_erased_odd_numbers_l170_170925

theorem sum_erased_odd_numbers (n : ℕ) (h1 : n + 2)^2 = 4147) 
    (a b : ℕ) (h2 : a = 2 * n + 1) (h3 : b = 4 * n + 5) : a + b = 168 :=
sorry

end sum_erased_odd_numbers_l170_170925


namespace line_relationship_l170_170767

-- Define the basic types and propositions
inductive LineRelationship
| parallel : LineRelationship
| intersecting : LineRelationship
| skew : LineRelationship

-- The theorem statement in Lean
theorem line_relationship (l1 l2 : Line) : 
  (l1 and l2 are parallel) ∨ (l1 and l2 intersect) ∨ (l1 and l2 are skew) :=
sorry

end line_relationship_l170_170767


namespace range_of_lambda_l170_170613

theorem range_of_lambda (a : ℕ → ℝ) (λ : ℝ) (h_inc : ∀ n, a (n + 1) > a n) (h_def : ∀ n, a n = n^2 + λ * n) :
  λ > -3 :=
sorry

end range_of_lambda_l170_170613


namespace sum_of_angles_subtended_by_edges_greater_than_540_l170_170735

-- Define a tetrahedron
structure Tetrahedron where
  A B C D : ℝ

-- Define internal point P
structure InternalPoint where
  P : ℝ

-- Define the measure of angles subtended by the edges of a tetrahedron from a point P
noncomputable def angle_subtended_by_edge (P : InternalPoint) (ABCD : Tetrahedron) : ℝ := sorry

-- The correct answer statement we want to prove
theorem sum_of_angles_subtended_by_edges_greater_than_540 (ABCD : Tetrahedron) (P : InternalPoint) :
  angle_subtended_by_edge P ABCD > 540 :=
sorry

end sum_of_angles_subtended_by_edges_greater_than_540_l170_170735


namespace kiki_scarves_count_l170_170691

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l170_170691


namespace collinear_APQ_l170_170655

variables {A B C C₀ B₀ O H P Q : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace C₀] [MetricSpace B₀] [MetricSpace O]
variables [MetricSpace H] [MetricSpace P] [MetricSpace Q]

-- Definitions based on conditions
def scalene_acute_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop := sorry
def midpoint (C₀ : Type) (A B : Type) [MetricSpace C₀] [MetricSpace A] [MetricSpace B] : Prop := sorry
def circumcenter (O : Type) (A B C : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop := sorry
def orthocenter (H : Type) (A B C : Type) [MetricSpace H] [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop := sorry
def intersection (P Q : Type) (L M : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace L] [MetricSpace M] : Prop := sorry
def rhombus (O H P Q : Type) [MetricSpace O] [MetricSpace H] [MetricSpace P] [MetricSpace Q] : Prop := sorry
def collinear (A P Q : Type) [MetricSpace A] [MetricSpace P] [MetricSpace Q] : Prop := sorry

-- Conjecture based on correct answer
theorem collinear_APQ : ∀ (A B C C₀ B₀ O H P Q : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace C₀] [MetricSpace B₀] [MetricSpace O]
  [MetricSpace H] [MetricSpace P] [MetricSpace Q],
  scalene_acute_triangle A B C →
  midpoint C₀ A B →
  midpoint B₀ A C →
  circumcenter O A B C →
  orthocenter H A B C →
  intersection P BH OC₀ →
  intersection Q CH OB₀ →
  rhombus O P H Q →
  collinear A P Q :=
begin
  sorry
end

end collinear_APQ_l170_170655


namespace no_equilateral_triangle_A1BC_l170_170665

-- Defining the points A1, A2, O, P and M based on the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A1 : Point := ⟨-3, 0⟩
def A2 : Point := ⟨3, 0⟩
def O : Point := ⟨0, 0⟩
def M (x : ℝ) : Point := ⟨real.sqrt (x^2 - 9), 0⟩

-- Defining vectors based on the points
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Problem (I): Derive the equation of W based on the given condition
def W_equation (λ : ℝ) (P : Point) : Prop := 
  λ^2 * (vec O (M P.x)).x^2 = (vec A1 P).x * (vec A2 P).x + (vec A1 P).y * (vec A2 P).y

-- Problem (II): Prove impossibility of equilateral triangle A1BC
theorem no_equilateral_triangle_A1BC (λ : ℝ) (B C : Point) : 
  λ = real.sqrt 3 / 3 ∧ B.y = B.x + 3 ∧
  (B.x ≠ -3 ∨ B.y ≠ 0) ∧ (B.y ^ 2 + B.x ^ 2 * 5 + 18 * B.x + 9 = 0) →
  C.x = -9 →
  ¬ ((A1.x - C.x)^2 + (A1.y - C.y)^2 = (A1.x - B.x)^2 + (A1.y - B.y)^2 ∧ 
  (A1.x - B.x)^2 + (A1.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2)
:= sorry

end no_equilateral_triangle_A1BC_l170_170665


namespace power_function_passes_through_point_l170_170219

theorem power_function_passes_through_point : ∃ α : ℝ, (∀ x : ℝ, f x = x^α) ∧ f 2 = 32 -> (α = 5) :=
begin
  sorry
end

end power_function_passes_through_point_l170_170219


namespace part1_part2_l170_170207

-- Problem Condition Definitions
def CircleP : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 16}
def Q (a b : ℝ) := (a, b)
def CircleM (PQ : ℝ) := {p : ℝ × ℝ | (p.1 - PQ / 2) ^ 2 + (p.2 - PQ / 2) ^ 2 = (PQ / 2) ^ 2}

-- Theorem Statements
theorem part1 (a b : ℝ) (hQ_outside : a ^ 2 + b ^ 2 > 16) (hQA_QB : (a - 0) ^ 2 + (b - 0) ^ 2 = 25) :
  a ^ 2 + b ^ 2 = 25 :=
sorry

theorem part2 (a b : ℝ) (Ha : a = 4) (Hb : b = 6) :
  ∀ x y : ℝ, (x - 2) ^ 2 + (y - 3) ^ 2 - 16 = (x - 0) ^ 2 + (y - 0) ^ 2 -> 2 * x + 3 * y - 8 = 0 :=
sorry

end part1_part2_l170_170207


namespace equidistant_points_bisector_plane_l170_170426

-- Definitions of key concepts
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

def isEquidistant (P A B : Point) : Prop :=
  dist P A = dist P B

def PerpendicularBisectorPlane (A B : Point) : set Point :=
  { P : Point | isEquidistant P A B }

-- The proof problem statement
theorem equidistant_points_bisector_plane (A B : Point) :
  { P | isEquidistant P A B } = PerpendicularBisectorPlane A B :=
sorry

end equidistant_points_bisector_plane_l170_170426


namespace minimum_value_l170_170360

open Real

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 9) :
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z) ≥ 9 :=
by sorry

end minimum_value_l170_170360


namespace total_toys_is_correct_l170_170329

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l170_170329


namespace mountain_liquid_volume_l170_170541

theorem mountain_liquid_volume (h1 : ∀ (altitude : ℝ), altitude > 0 → atmospheric_pressure altitude < atmospheric_pressure 0)
    (h2 : ∀ (altitude : ℝ), altitude > 0 → convexity_of_liquid_surface altitude < convexity_of_liquid_surface 0) :
    ∀ (altitude : ℝ), altitude > 0 → volume_of_liquid altitude < volume_of_liquid 0 :=
by
  sorry

end mountain_liquid_volume_l170_170541


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170008

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170008


namespace distance_between_points_l170_170664

theorem distance_between_points (x : ℝ) :
  let M := (-1, 4)
  let N := (x, 4)
  dist (M, N) = 5 →
  (x = -6 ∨ x = 4) := sorry

end distance_between_points_l170_170664


namespace find_g_inv_f_l170_170256

noncomputable def f_inv := λ g_x : ℝ, (g_x ^ 4 - 3)
def g := λ x : ℝ, f (x ^ 4 - 3)
def f (x : ℝ) : ℝ := sorry  -- since the actual function f is not given, we put a placeholder

variable (g_inv : ℝ → ℝ)

-- Assume g has an inverse, i.e., g_inv is the inverse for g
axiom g_inverse : ∀ x : ℝ, g_inv (g x) = x

-- The main theorem to prove
theorem find_g_inv_f :
  g_inv (f 10) = real.root 4 13 := sorry

end find_g_inv_f_l170_170256


namespace simplify_expression_l170_170938

theorem simplify_expression :
  1 + (1 / (1 + Real.sqrt 2)) - (1 / (1 - Real.sqrt 5)) =
  1 + ((-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10)) :=
by
  sorry

end simplify_expression_l170_170938


namespace not_recurring_decimal_l170_170085

-- Definitions based on the provided conditions
def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ d m n : ℕ, d ≠ 0 ∧ (x * d) % 10 ^ n = m

-- Condition: 0.89898989
def number_0_89898989 : ℝ := 0.89898989

-- Proof statement to show 0.89898989 is not a recurring decimal
theorem not_recurring_decimal : ¬ is_recurring_decimal number_0_89898989 :=
sorry

end not_recurring_decimal_l170_170085


namespace parabola_focus_l170_170185

-- Define the condition of the parabola y = 2x^2
def is_parabola (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, p x = 2 * x^2

-- Define the condition for the focus
def is_focus (p : ℝ → ℝ) (f : ℝ × ℝ) : Prop :=
  f = (0, 1 / 8)

-- Define the main theorem to state and prove focus of the parabola
theorem parabola_focus : 
  ∃ f : ℝ × ℝ, is_parabola (λ x, 2 * x^2) ∧ is_focus (λ x, 2 * x^2) f :=
begin
  -- Proof is omitted
  sorry
end

end parabola_focus_l170_170185


namespace cost_price_of_radio_l170_170517

theorem cost_price_of_radio (C : ℝ) 
  (overhead_expenses : ℝ := 15) (selling_price : ℝ := 350) (profit_percent : ℝ := 45.833333333333314) :
  C = 228.41 :=
by
  let profit_fraction := profit_percent / 100
  have total_cost := C + overhead_expenses
  have profit := selling_price - total_cost
  have profit_eq := profit_fraction * C = profit
  have eq1 := ((7 / 15) * C = 335 - C)
  have eqn := (22 / 15) * C = 335
  let C_val := 335 * 15 / 22
  have calc_C := C_val = 228.41
  sorry

end cost_price_of_radio_l170_170517


namespace arithmetic_sequence_geometric_sequence_sum_of_first_n_terms_l170_170611

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 3^n
noncomputable def c_n (n : ℕ) : ℤ := (-1 : ℤ)^n * a_n n * b_n n
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in range (n + 1), c_n i

theorem arithmetic_sequence {d : ℤ} : (a_n 1 = 1) → (a_n 3 + a_n 4 = 12) → d = 2 → a_n n = 2n - 1 := sorry
theorem geometric_sequence {q : ℤ} : (b_n 1 = a_n 2) → (b_n 2 = a_n 5) → q = 3 → b_n n = 3^n := sorry
theorem sum_of_first_n_terms {n : ℕ} : S_n n = (3/8 : ℝ) - (8n - 1)/8 * (-3 : ℝ)^(n + 1) := sorry

end arithmetic_sequence_geometric_sequence_sum_of_first_n_terms_l170_170611


namespace zara_owns_113_goats_l170_170071

-- Defining the conditions
def cows : Nat := 24
def sheep : Nat := 7
def groups : Nat := 3
def animals_per_group : Nat := 48

-- Stating the problem, with conditions as definitions
theorem zara_owns_113_goats : 
  let total_animals := groups * animals_per_group in
  let cows_and_sheep := cows + sheep in
  let goats := total_animals - cows_and_sheep in
  goats = 113 := by
  sorry

end zara_owns_113_goats_l170_170071


namespace sum_of_roots_cubic_equation_l170_170052

theorem sum_of_roots_cubic_equation :
  let roots := multiset.to_finset (multiset.filter (λ r, r ≠ 0) (RootSet (6 * (X ^ 3) + 7 * (X ^ 2) + (-12) * X) ℤ))
  (roots.sum : ℤ) / (roots.card : ℤ) = -117 / 100 := sorry

end sum_of_roots_cubic_equation_l170_170052


namespace elevator_travel_time_l170_170723

-- Definitions corresponding to the conditions
def total_floors : ℕ := 20
def first_half_floors_time : ℕ := 15 -- in minutes
def next_five_floors_time_per_floor : ℕ := 5 -- in minutes per floor
def final_five_floors_time_per_floor : ℕ := 16 -- in minutes per floor

-- Statement of the problem
theorem elevator_travel_time :
  let first_half_floors := total_floors / 2,
      next_five_floors := 5,
      final_five_floors := 5,
      total_time_in_minutes := first_half_floors_time + 
                               (next_five_floors * next_five_floors_time_per_floor) + 
                               (final_five_floors * final_five_floors_time_per_floor)
  in total_time_in_minutes / 60 = 2 := 
sorry

end elevator_travel_time_l170_170723


namespace kiki_scarves_count_l170_170690

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l170_170690


namespace find_k_l170_170632

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-2, k)
def vec_op (a b : ℝ × ℝ) : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

noncomputable def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_prod a (vec_op a (b k)) = 0 → k = 14 :=
by
  sorry

end find_k_l170_170632


namespace total_surface_area_of_tower_is_correct_l170_170373

theorem total_surface_area_of_tower_is_correct :
  let volumes := [1, 8, 27, 64, 125, 216, 343, 512, 729]
  let side_length (v : ℕ) := v.cbrt
  let surface_area (s : ℕ) := 6 * s^2
  let adjusted_surface_area (s : ℕ) (prev_s : Option ℕ) := 
    surface_area s - Option.getOrElse (prev_s.map (λ p_s, p_s^2)) 0
  let total_surface_area :=
    List.sum
      (List.mapIdx (λ idx v, adjusted_surface_area (side_length v) 
        (if idx = 0 then none else some (side_length (volumes.idx (idx-1))))) volumes)
  total_surface_area = 1426 := by
  sorry

end total_surface_area_of_tower_is_correct_l170_170373


namespace dance_class_students_l170_170304

theorem dance_class_students : 
  ∃ (n : ℕ), n > 40 ∧ let total := 5 * n + 2 in total = 207 ∧ total > 200 :=
begin
  sorry
end

end dance_class_students_l170_170304


namespace S30_arithmetic_sequence_l170_170657

-- Define the sum of an arithmetic sequence
def sum_arithmetic_n (a d : ℕ → ℝ) (n : ℕ) := (n / 2) * (2 * a 0 + (n - 1) * d 0)

-- Conditions of the problem in Lean
variable {a : ℕ → ℝ}
variable {d : ℕ → ℝ}
hypothesis (h_seq : ∀ n, a (n + 1) = a n + d 0)
hypothesis (S10 : sum_arithmetic_n a d 10 = 10)
hypothesis (S20 : sum_arithmetic_n a d 20 = 30)

-- Theorem to prove S30 = 50
theorem S30_arithmetic_sequence : sum_arithmetic_n a d 30 = 50 :=
sorry

-- The theorem statement assumes the conditions directly from the problem statement, ensuring it is equivalent to the original mathematical problem.

end S30_arithmetic_sequence_l170_170657


namespace cos_a5_eq_sqrt2_div2_l170_170612

-- Define the arithmetic sequence conditions
variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_sum1 : a 1 + a 2 + a 3 = π / 2)
variable (h_sum2 : a 7 + a 8 + a 9 = π)

-- Prove that cos(a 5) = sqrt(2) / 2
theorem cos_a5_eq_sqrt2_div2 : cos (a 5) = sqrt 2 / 2 := by
  -- Proof goes here
  sorry

end cos_a5_eq_sqrt2_div2_l170_170612


namespace range_of_function_l170_170472

theorem range_of_function :
  (set.range (λ x : ℝ, 1 / (2 - x)^2)) = set.Ioi 0 := by
  sorry

end range_of_function_l170_170472


namespace range_of_function_l170_170470

theorem range_of_function :
  (set.range (λ x : ℝ, 1 / (2 - x)^2)) = set.Ioi 0 := by
  sorry

end range_of_function_l170_170470


namespace standard_equation_of_line_l170_170760

theorem standard_equation_of_line (t : ℝ) (x y : ℝ) (h₁ : x = -2 - √2 * t) (h₂ : y = 3 + √2 * t) :
    x + y - 1 = 0 :=
by
  sorry

end standard_equation_of_line_l170_170760


namespace intersection_points_of_graphs_l170_170946

-- Define the invertibility condition
def is_invertible (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- State the problem in Lean
theorem intersection_points_of_graphs (f : ℝ → ℝ) (h_invertible : is_invertible f) :
  {x : ℝ | f (x ^ 2) = f (x ^ 6)}.finite ∧ 
  {x : ℝ | f (x ^ 2) = f (x ^ 6)}.to_finset.card = 3 :=
by
  sorry

end intersection_points_of_graphs_l170_170946


namespace solution_set_x_squared_minus_3x_lt_0_l170_170766

theorem solution_set_x_squared_minus_3x_lt_0 : { x : ℝ | x^2 - 3 * x < 0 } = { x : ℝ | 0 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_x_squared_minus_3x_lt_0_l170_170766


namespace number_of_tangents_l170_170451

theorem number_of_tangents (A B : Point) (hAB : dist A B = 8) : 
  count_lines_tangent_to_circles A B 3 2 = 2 :=
sorry

end number_of_tangents_l170_170451


namespace smallest_m_value_l170_170143

theorem smallest_m_value : ∃ (m : ℕ), (m * 15 = 180) ∧ (∀ n, n * 15 = 180 → n ≥ m) :=
by
  exists 12
  split
  { sorry }
  { intro n h
    sorry }

end smallest_m_value_l170_170143


namespace shortest_side_of_triangle_l170_170321

noncomputable def shortest_side_length (A B C D E : Point) (r : Real) : Real :=
  let AB := dist A B in
  let AD := dist A D in
  let DB := dist D B in
  if r = 6 ∧ AD = 9 ∧ DB = 15 ∧ AB = AD + DB then
    AB
  else
    0

theorem shortest_side_of_triangle (A B C D E : Point) (r : Real) (AD DB : Real):
  r = 6 → AD = 9 → DB = 15 → dist A B = AD + DB →
  shortest_side_length A B C D E r = dist A B := by
    sorry

end shortest_side_of_triangle_l170_170321


namespace good_function_gcd_representable_l170_170161

def good_function (k : ℕ) (f : ℕ × ℕ → ℕ) : Prop :=
  k = 0 ∧ (∀ a b, f (a, b) = a ∨ f (a, b) = b) ∨
  ∃ p q (f₁ f₂ : ℕ × ℕ → ℕ),
    good_function p f₁ ∧ good_function q f₂ ∧
    (∀ a b, f (a, b) = Nat.gcd (f₁ (a, b)) (f₂ (a, b)) ∨
            f (a, b) = f₁ (a, b) * f₂ (a, b) ∧ k = p + q + 1)

theorem good_function_gcd_representable (f : ℕ × ℕ → ℕ) (n k : ℕ) (h₁ : 3 ≤ n)
  (h₂ : good_function k f) (h₃ : k ≤ Nat.choose n 3) :
  ∃ t ≤ Nat.choose n 2, ∃ xy : List (ℕ × ℕ),
    ∀ a b, f (a, b) = Nat.gcd (xy.map (λ p, a ^ p.1 * b ^ p.2)).prod := by
  sorry

end good_function_gcd_representable_l170_170161


namespace find_angle_BAO_l170_170666

-- Given conditions
variable {O A B C D E : Point}
variable {CD: Segment}
variable {diameter : CD = Segment.between_points O C + Segment.between_points O D}
variable {A_extension : A ∈ line_extension_segment D C}
variable {E_semi_circle : E ∈ semicircle_centered_at O with_diameter CD}
variable {B_intersection : B ∈ line_segment A E ∧ B ≠ E}
variable {AB_OD : Distance AB = Distance OD}
variable {angle_EOD: Angle E O D = 30}

-- Proof problem
theorem find_angle_BAO : Angle B A O = 7.5 :=
  sorry

end find_angle_BAO_l170_170666


namespace solution_set_l170_170165

open Nat

def is_solution (a b c : ℕ) : Prop :=
  a ^ (b + 20) * (c - 1) = c ^ (b + 21) - 1

theorem solution_set (a b c : ℕ) : 
  (is_solution a b c) ↔ ((c = 0 ∧ a = 1) ∨ (c = 1)) := 
sorry

end solution_set_l170_170165


namespace inequality1_inequality2_l170_170741

theorem inequality1 (x : ℝ) : 
  (3 * x - 1) / (2 - x) > 1 ↔ x ∈ set.Ioo (3 / 4) 2 :=
sorry

theorem inequality2 (x : ℝ) : 
  -1 ≤ x^2 + 2 * x - 1 ∧ x^2 + 2 * x - 1 ≤ 2 ↔ x ∈ (set.Icc (-3) (-2) ∪ set.Icc 0 1) :=
sorry

end inequality1_inequality2_l170_170741


namespace product_of_consecutive_integers_between_sqrt_50_l170_170829

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170829


namespace print_time_for_document_l170_170513

def total_print_time (text_pages : ℕ) (graphic_pages : ℕ) (text_rate : ℕ) (graphic_rate : ℕ) : ℝ :=
  (text_pages / text_rate) + (graphic_pages / graphic_rate)

theorem print_time_for_document :
  let text_pages := 250
  let graphic_pages := 90
  let text_rate := 17
  let graphic_rate := 10
  let total_time := total_print_time text_pages graphic_pages text_rate graphic_rate
  Int.round total_time = 24 :=
by
  sorry

end print_time_for_document_l170_170513


namespace Problem1_Problem2_l170_170886

noncomputable def f : ℝ → ℝ := sorry

theorem Problem1 (x : ℝ) (h : f (x + 1) = x^2 + 4 * x + 1) : 
  f(x) = x^2 + 2 * x - 2 := 
sorry

theorem Problem2 (x : ℝ) (h : f(x) - 2 * f(-x) = 9 * x + 2) :
  f(x) = 3 * x - 2 := 
sorry

end Problem1_Problem2_l170_170886


namespace cat_reach_shelter_probability_l170_170305

noncomputable def P : ℕ → ℚ
| 0       := 0
| 14      := 1
| (n + 1) := if h : n + 1 < 14 then (n + 1 : ℚ) / 14 * P n + (1 - (n + 1 : ℚ) / 14) * P (n + 2) else 0

theorem cat_reach_shelter_probability : P 2 = 610/943 := 
sorry

end cat_reach_shelter_probability_l170_170305


namespace largeSquareArea_l170_170136

-- Given conditions
def squareDividedIntoFiveEqualAreas (s : ℝ) := s * s / 5
def lengthAB := 3.6 -- in centimeters

-- Prove that the area of the large square is 12.96 square centimeters
theorem largeSquareArea (s : ℝ) (h1 : 3.6 = s) : s * s = 12.96 := sorry

end largeSquareArea_l170_170136


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170277

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170277


namespace product_of_consecutive_integers_sqrt_50_l170_170788

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170788


namespace product_of_consecutive_integers_between_sqrt_50_l170_170823

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170823


namespace area_of_triangle_PKF_l170_170233

open_locale classical

-- Definitions: parabola, points, and area
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

def F : ℝ × ℝ := (1, 0)

def K : ℝ × ℝ := (-1, 0)

def on_parabola (P : ℝ × ℝ) : Prop := parabola P

def distance (P1 P2 : ℝ × ℝ) : ℝ := real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def triangle_area (A B C : ℝ × ℝ) : ℝ := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement
theorem area_of_triangle_PKF :
  ∃ P : ℝ × ℝ, on_parabola P ∧ distance P F = 5 ∧ triangle_area P K F = 4 := sorry

end area_of_triangle_PKF_l170_170233


namespace A_3_1_equals_10_l170_170951

def A : ℕ → ℕ → ℕ
| 0, n := n + 1
| (m+1), 0 := A m 1
| (m+1), (n+1) := A m (A (m+1) n)

theorem A_3_1_equals_10 : A 3 1 = 10 := 
  by 
    sorry

end A_3_1_equals_10_l170_170951


namespace triangle_crease_length_l170_170602

/-- Given a right triangle with side lengths 5, 12, and 13 inches,
    if point A is folded onto point C, then the length of the crease is 12.25 inches.-/
theorem triangle_crease_length (A B C : Point) (a b c : ℝ) (h : a = 5 ∧ b = 12 ∧ c = 13) :
  length (crease (fold_to C A)) = 12.25 := sorry

end triangle_crease_length_l170_170602


namespace sqrt_50_product_consecutive_integers_l170_170811

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170811


namespace inequality_solution_l170_170169

theorem inequality_solution (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
    (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := by
  sorry

end inequality_solution_l170_170169


namespace PR_minus_PQ_min_value_l170_170151

theorem PR_minus_PQ_min_value :
  ∃ (x y : ℕ),
  2.5 * x + y = 2021 ∧
  (1 : ℕ) * x < (3 / 2) * x ∧
  y = 2021 - (5 / 2) * x ∧
  404.2 < x ∧
  x < 1010.5 ∧
  x % 2 = 0 ∧
  y ∈ ℕ ∧
  2.5 * x + (2021 - (5 / 2) * x) > x ∧
  x + (2021 - (5 / 2) * x) > (3 / 2) * x ∧
  (3 / 2) * x + (2021 - (5 / 2) * x) > x ∧
  (3 / 2) * x - x = 204 :=
begin
  sorry
end

end PR_minus_PQ_min_value_l170_170151


namespace base_k_to_decimal_is_5_l170_170896

theorem base_k_to_decimal_is_5 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 42) : k = 5 := sorry

end base_k_to_decimal_is_5_l170_170896


namespace voter_ratio_l170_170311

theorem voter_ratio (Vx Vy : ℝ) (hx : 0.72 * Vx + 0.36 * Vy = 0.60 * (Vx + Vy)) : Vx = 2 * Vy :=
by
sorry

end voter_ratio_l170_170311


namespace fraction_not_on_time_l170_170876

-- Define the conditions as hypotheses
variables (x : ℝ) -- total number of attendees

-- Define the fraction of males, females, and those who arrived on time
def fraction_males := 2 / 3
def fraction_males_on_time := 3 / 4
def fraction_females_on_time := 5 / 6

-- Define the equations based on the given conditions
def num_males := fraction_males * x
def num_males_on_time := fraction_males_on_time * num_males
def num_females := x - num_males
def num_females_on_time := fraction_females_on_time * num_females

-- Define those who did not arrive on time
def num_males_not_on_time := num_males - num_males_on_time
def num_females_not_on_time := num_females - num_females_on_time

-- Statement to prove the fraction of attendees who did not arrive on time
theorem fraction_not_on_time (x : ℝ) (hx : x ≠ 0) : 
  (num_males_not_on_time x + num_females_not_on_time x) / x = 2 / 9 :=
sorry

end fraction_not_on_time_l170_170876


namespace polynomial_solution_l170_170577

noncomputable def p : ℝ → ℝ :=
  λ x, x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) ∧ p 3 = 10 :=
by
  sorry

end polynomial_solution_l170_170577


namespace units_digit_sum_factorial_squares_l170_170982

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition to get the units digit of a number
def units_digit (n : ℕ) : ℕ :=
n % 10

-- The main theorem statement
theorem units_digit_sum_factorial_squares : units_digit (∑ i in range 5, (factorial i)^2) = 7 :=
by
  sorry

end units_digit_sum_factorial_squares_l170_170982


namespace sum_of_roots_l170_170050

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l170_170050


namespace sqrt_50_product_consecutive_integers_l170_170812

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170812


namespace part1_proof_part2_proof_l170_170349

-- Let {a_n} be a sequence with the sum of its first n terms denoted as S_n.
-- For any n ∈ ℕ⁺, it holds that S_n = (n + 1)(a_n - n).
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def a_n (n : ℕ) : ℝ := 2 * n -- Following from the proof steps in the solution

-- Prove that {a_n} is an arithmetic sequence.
theorem part1_proof : is_arithmetic_sequence a_n := by
  -- Proof steps would go here
  sorry

-- The sum of the first n terms of the sequence {1 / (a_n * a_(n + 1))} is n / (4(n + 1)).
def term (n : ℕ) : ℝ := 1 / (a_n n * a_n (n + 1))
def sum_first_n_terms (n : ℕ) : ℝ := (Finset.range n).sum term

theorem part2_proof (n : ℕ) : sum_first_n_terms n = n / (4 * (n + 1)) := by
  -- Proof steps would go here
  sorry

end part1_proof_part2_proof_l170_170349


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170001

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170001


namespace roasted_marshmallows_total_l170_170685

def joe_marshmallows (dads_marshmallows : ℕ) := 4 * dads_marshmallows

def roasted_marshmallows (total_marshmallows : ℕ) (fraction : ℕ) := total_marshmallows / fraction

theorem roasted_marshmallows_total :
  let dads_marshmallows := 21 in
  let joe_marshmallows := joe_marshmallows dads_marshmallows in
  let dads_roasted := roasted_marshmallows dads_marshmallows 3 in
  let joe_roasted := roasted_marshmallows joe_marshmallows 2 in
  dads_roasted + joe_roasted = 49 :=
by
  sorry

end roasted_marshmallows_total_l170_170685


namespace find_some_number_l170_170646

theorem find_some_number : 
  ∃ x : ℝ, 
  (6 + 9 * 8 / x - 25 = 5) ↔ (x = 3) :=
by 
  sorry

end find_some_number_l170_170646


namespace jill_vs_jack_arrival_time_l170_170326

def distance_to_park : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem jill_vs_jack_arrival_time :
  let jill_time := distance_to_park / jill_speed
  let jack_time := distance_to_park / jack_speed
  let jill_time_minutes := jill_time * 60
  let jack_time_minutes := jack_time * 60
  jill_time_minutes < jack_time_minutes ∧ jack_time_minutes - jill_time_minutes = 5.4 :=
by
  sorry

end jill_vs_jack_arrival_time_l170_170326


namespace travel_time_difference_l170_170729

theorem travel_time_difference :
  (160 / 40) - (280 / 40) = 3 := by
  sorry

end travel_time_difference_l170_170729


namespace coin_combinations_30_cents_l170_170245

theorem coin_combinations_30_cents : 
  let value_penny := 1
      value_nickel := 5
      value_dime := 10
  in 
  ∃ (p k d : ℕ), (p * value_penny + k * value_nickel + d * value_dime = 30 ∧ 
        p + k + d = 22) :=
by {
  sorry
}

end coin_combinations_30_cents_l170_170245


namespace bird_cages_count_l170_170116

theorem bird_cages_count :
  (∃ x : ℕ, (6 * x) + (2 * x) = 48) → ∃ x : ℕ, x = 6 :=
by
  intro h
  cases h with x hx
  use 6
  sorry

end bird_cages_count_l170_170116


namespace FB_length_correct_l170_170320

-- Define a structure for the problem context
structure Triangle (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] where
  AB : ℝ
  CD : ℝ
  AE : ℝ
  altitude_CD : C -> (A -> B -> Prop)  -- CD is an altitude to AB
  altitude_AE : E -> (B -> C -> Prop)  -- AE is an altitude to BC
  angle_bisector_AF : F -> (B -> C -> Prop)  -- AF is the angle bisector of ∠BAC intersecting BC at F
  intersect_AF_BC_at_F : (F -> B -> Prop)  -- AF intersects BC at F

noncomputable def length_of_FB (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : ℝ := 
  2  -- From given conditions and conclusion

-- The main theorem to prove
theorem FB_length_correct (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : 
  t.AB = 8 ∧ t.CD = 3 ∧ t.AE = 4 → length_of_FB A B C D E F t = 2 :=
by
  intro h
  obtain ⟨AB_eq, CD_eq, AE_eq⟩ := h
  sorry

end FB_length_correct_l170_170320


namespace cost_of_large_palm_fern_per_plant_l170_170455

-- Definitions of the given conditions
def cost_creeping_jennies_per_plant := 4.00
def cost_geraniums_per_plant := 3.50
def number_of_creeping_jennies_per_pot := 4
def number_of_geraniums_per_pot := 4
def number_of_pots := 4
def total_cost := 180.00

-- The condition in Lean 4 statement
theorem cost_of_large_palm_fern_per_plant :
  let cost_creeping_jennies_per_pot := number_of_creeping_jennies_per_pot * cost_creeping_jennies_per_plant,
      cost_geraniums_per_pot := number_of_geraniums_per_pot * cost_geraniums_per_plant,
      total_cost_per_pot := cost_creeping_jennies_per_pot + cost_geraniums_per_pot,
      cost_creeping_jennies_and_geraniums_for_all_pots := total_cost_per_pot * number_of_pots,
      remaining_cost := total_cost - cost_creeping_jennies_and_geraniums_for_all_pots,
      cost_large_palm_fern_per_pot := remaining_cost / number_of_pots in
  cost_large_palm_fern_per_pot = 15.00 :=
  by sorry

end cost_of_large_palm_fern_per_plant_l170_170455


namespace cartesian_coordinate_equations_chord_length_l170_170629

noncomputable def circle_o1_polar_equation : ℝ := 2

noncomputable def circle_o2_polar_equation (rho theta : ℝ) : Prop :=
  rho^2 - 2 * real.sqrt 2 * rho * real.cos (theta - real.pi / 4) = 2

theorem cartesian_coordinate_equations :
  (∀ (rho theta : ℝ), circle_o1_polar_equation = rho → (θ = theta) → ∀ (x y : ℝ), (x = rho * real.cos theta) → (y = rho * real.sin theta) → x^2 + y^2 = 4) ∧
  (∀ (rho theta : ℝ), circle_o2_polar_equation rho θ → ∀ (x y : ℝ), (x = rho * real.cos theta) → (y = rho * real.sin theta) → x^2 + y^2 - 2 * x - 2 * y - 2 = 0) :=
by
  split
  . sorry
  . sorry

theorem chord_length :
  ∀ (x y : ℝ), (x^2 + y^2 = 4) → (x^2 + y^2 - 2 * x - 2 * y - 2 = 0) → ∃ A B : ℝ × ℝ, set.inters O1 O2 = {A, B} → |AB| = real.sqrt 14 :=
by
  sorry

end cartesian_coordinate_equations_chord_length_l170_170629


namespace product_of_consecutive_integers_sqrt_50_l170_170785

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l170_170785


namespace problem_1_problem_2_problem_3_problem_4_l170_170141

theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 :=
by sorry

theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 :=
by sorry

theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 :=
by sorry

theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l170_170141


namespace infinite_product_l170_170565

theorem infinite_product (P : ℕ → ℝ) :
  (∀ n, P n = (3:ℝ)^(n * (1 / (3^n)))) → 
  (∏' n, P n) = 3^(3 / 2) :=
by
  intros h
  have : ∑' n, n * (1 / (3^n)) = 3 / 2 := sorry
  calc
    ∏' n, P n = ∏' n, (3:ℝ)^(n * (1 / (3^n))) : by { congr, funext, exact h }
           ... = (3:ℝ)^(∑' n, n * (1 / (3^n))) : sorry
           ... = (3:ℝ)^(3 / 2) : by { rw this }

end infinite_product_l170_170565


namespace f_at_neg_one_l170_170597

def f : ℝ → ℝ := sorry

theorem f_at_neg_one :
  (∀ x : ℝ, f (x / (1 + x)) = x) →
  f (-1) = -1 / 2 :=
by
  intro h
  -- proof omitted for clarity
  sorry

end f_at_neg_one_l170_170597


namespace max_crates_first_trip_l170_170525

theorem max_crates_first_trip (x : ℕ) : (∀ w, w ≥ 120) ∧ (600 ≥ x * 120) → x = 5 := 
by
  -- Condition: The weight of any crate is no less than 120 kg
  intro h
  have h1 : ∀ w, w ≥ 120 := h.left
  
  -- Condition: The maximum weight for the first trip
  have h2 : 600 ≥ x * 120 := h.right 
  
  -- Derivation of maximum crates
  have h3 : x ≤ 600 / 120 := by sorry  -- This inequality follows from h2 by straightforward division
  
  have h4 : x ≤ 5 := by sorry  -- This follows from evaluating 600 / 120 = 5
  
  -- Knowing x is an integer and the maximum possible value is 5
  exact by sorry

end max_crates_first_trip_l170_170525


namespace no_illumination_l170_170504

noncomputable def is_illuminated (t : ℝ) (c : ℝ) : Prop :=
  let x := 3
  let y := 3 + sin t * cos t - sin t - cos t
  y = c * x

theorem no_illumination (c : ℝ) (h1 : c > 0) (h2 : c < 1/2 ∨ c > 7/6) :
  ¬ ∃ t : ℝ, is_illuminated t c :=
by
  sorry

end no_illumination_l170_170504


namespace ellipse_eccentricity_l170_170262

-- Define the geometric sequence condition and the ellipse properties
theorem ellipse_eccentricity :
  ∀ (a b c e : ℝ), 
  (b^2 = a * c) ∧ (a^2 - c^2 = b^2) ∧ (e = c / a) ∧ (0 < e ∧ e < 1) →
  e = (Real.sqrt 5 - 1) / 2 := 
by 
  sorry

end ellipse_eccentricity_l170_170262


namespace gain_percent_l170_170483

theorem gain_percent (C S : ℝ) (h : 50 * C = 15 * S) :
  (S > C) →
  ((S - C) / C * 100) = 233.33 := 
sorry

end gain_percent_l170_170483


namespace union_of_sets_eq_l170_170607

theorem union_of_sets_eq 
  (A : Set ℝ := {x | x > 1}) 
  (B : Set ℝ := {x | x^2 - 2x - 3 > 0}) : 
  A ∪ B = {x : ℝ | x < -1 ∨ x > 1} :=
sorry

end union_of_sets_eq_l170_170607


namespace value_of_exponentiation_l170_170768

theorem value_of_exponentiation : (256: ℝ) ^ 0.16 * 256 ^ 0.09 = 4 :=
by
  sorry

end value_of_exponentiation_l170_170768


namespace largest_n_for_sin_cos_inequality_l170_170975

theorem largest_n_for_sin_cos_inequality :
  ∃ n : ℕ, (∀ x : ℝ, (sin x + cos x)^n ≥ (2 / n : ℝ)) ∧
  (∀ m : ℕ, (∀ x : ℝ, (sin x + cos x)^m ≥ (2 / m : ℝ)) → m ≤ n) :=
begin
  sorry
end

end largest_n_for_sin_cos_inequality_l170_170975


namespace train_speed_approx_to_interval_l170_170762

theorem train_speed_approx_to_interval (s : ℝ) (t : ℝ) (length_of_rails : ℝ) (v : ℝ) :
  length_of_rails = 40 →
  t = 20 / 60 →
  v = 5280 * s / 60 →
  v / length_of_rails * t ≈ s →
  t ≈ 20 / 60 :=
by
  intros h1 h2 h3 h4
  sorry

end train_speed_approx_to_interval_l170_170762


namespace remaining_two_by_two_square_exists_l170_170198

theorem remaining_two_by_two_square_exists (grid_size : ℕ) (cut_squares : ℕ) : grid_size = 29 → cut_squares = 99 → 
  ∃ remaining_square : ℕ, remaining_square = 1 :=
by
  intros
  sorry

end remaining_two_by_two_square_exists_l170_170198


namespace koby_sparklers_correct_l170_170693

-- Define the number of sparklers in each of Koby's boxes as a variable
variable (S : ℕ)

-- Specify the conditions
def koby_sparklers : ℕ := 2 * S
def koby_whistlers : ℕ := 2 * 5
def cherie_sparklers : ℕ := 8
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := koby_sparklers S + koby_whistlers + cherie_sparklers + cherie_whistlers

-- The theorem to prove that the number of sparklers in each of Koby's boxes is 3
theorem koby_sparklers_correct : total_fireworks S = 33 → S = 3 := by
  sorry

end koby_sparklers_correct_l170_170693


namespace stratified_sampling_l170_170089

noncomputable def proportion_heaters (total_heaters : ℕ) (heaters_A : ℕ) : ℚ := 
  heaters_A / total_heaters

def sample_size : ℕ := 14

def num_from_A (total_heaters sample_size heater_A heaters_B : ℕ) :=
  proportion_heaters total_heaters heater_A * (sample_size : ℚ)

def num_from_B (sample_size num_A : ℕ) :=
  sample_size - num_A

theorem stratified_sampling (total_heaters heaters_A heaters_B : ℕ) (sample_size num_A num_B : ℕ) :
  total_heaters = 98 →
  heaters_A = 56 →
  heaters_B = 42 →
  sample_size = 14 →
  num_A = num_from_A total_heaters sample_size heaters_A heaters_B →
  num_B = num_from_B sample_size num_A →
  num_A = 8 ∧ num_B = 6 :=
by
  intros htA htB hsA hs_S nsA nsB
  rw [htA, hsA] at nsA
  rw [hs_S, nsA] at nsB
  exact ⟨by norm_num [proportion_heaters, num_from_A] at nsA,
         by norm_num [num_from_B] at nsB⟩

end stratified_sampling_l170_170089


namespace willie_stickers_l170_170478

theorem willie_stickers :
  let initial_stickers := 36
  let given_away_stickers := 7
  let final_stickers := initial_stickers - given_away_stickers
  final_stickers = 29 :=
by
  let initial_stickers := 36
  let given_away_stickers := 7
  let final_stickers := initial_stickers - given_away_stickers
  show final_stickers = 29 from sorry

end willie_stickers_l170_170478


namespace intersection_shape_parallelogram_l170_170959

theorem intersection_shape_parallelogram :
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    (p1.1 * p1.2 = 16 ∧ 4 * p1.1^2 + p1.2^2 = 36) ∧
    (p2.1 * p2.2 = 16 ∧ 4 * p2.1^2 + p2.2^2 = 36) ∧
    (p3.1 * p3.2 = 16 ∧ 4 * p3.1^2 + p3.2^2 = 36) ∧
    (p4.1 * p4.2 = 16 ∧ 4 * p4.1^2 + p4.2^2 = 36) ∧
    parallelogram p1 p2 p3 p4) :=
sorry

end intersection_shape_parallelogram_l170_170959


namespace quadratic_root_expression_l170_170702

theorem quadratic_root_expression (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 + x - 2023 = 0 → (x = a ∨ x = b)) 
  (ha_neq_b : a ≠ b) :
  a^2 + 2*a + b = 2022 :=
sorry

end quadratic_root_expression_l170_170702


namespace average_score_all_students_l170_170861

theorem average_score_all_students 
  (n1 n2 : Nat) 
  (avg1 avg2 : Nat) 
  (h1 : n1 = 20) 
  (h2 : avg1 = 80) 
  (h3 : n2 = 30) 
  (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 := 
by
  sorry

end average_score_all_students_l170_170861


namespace print_time_l170_170512

theorem print_time (rate : ℕ) (total_pages : ℕ) : rate = 24 → total_pages = 360 → total_pages / rate = 15 :=
by 
  intros 
  assume h_rate : rate = 24 
  assume h_total_pages : total_pages = 360 
  rw [h_rate, h_total_pages] 
  norm_num
  sorry

end print_time_l170_170512


namespace system_of_equations_solution_l170_170971

theorem system_of_equations_solution :
  ∀ (x y z : ℝ), 
    (x^5 = y^3 + 2 * z ∧ y^5 = z^3 + 2 * x ∧ z^5 = x^3 + 2 * y) →
      (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
      (x = √2 ∧ y = √2 ∧ z = √2) ∨ 
      (x = -√2 ∧ y = -√2 ∧ z = -√2) :=
by
  intros x y z h
  sorry

end system_of_equations_solution_l170_170971


namespace simple_interest_sum_l170_170523

theorem simple_interest_sum :
  ∀ {SI R T : ℝ},
  (SI = 4016.25) →
  (R = 1/100) →
  (T = 9) →
  ∃ P : ℝ, P = 44625 :=
by
  intros SI R T hSI hR hT
  use 44625
  sorry

end simple_interest_sum_l170_170523


namespace train_speed_l170_170130

theorem train_speed (train_length : Real) (bridge_length : Real) (time : Real) (speed_in_kmph : Real) : train_length = 70 → bridge_length = 80 → time = 14.998800095992321 → 
speed_in_kmph ≈ 36.0019209356 :=
by
  sorry

end train_speed_l170_170130


namespace smallest_model_length_l170_170384

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end smallest_model_length_l170_170384


namespace roasted_marshmallows_total_l170_170687

def joe_marshmallows (dads_marshmallows : ℕ) := 4 * dads_marshmallows

def roasted_marshmallows (total_marshmallows : ℕ) (fraction : ℕ) := total_marshmallows / fraction

theorem roasted_marshmallows_total :
  let dads_marshmallows := 21 in
  let joe_marshmallows := joe_marshmallows dads_marshmallows in
  let dads_roasted := roasted_marshmallows dads_marshmallows 3 in
  let joe_roasted := roasted_marshmallows joe_marshmallows 2 in
  dads_roasted + joe_roasted = 49 :=
by
  sorry

end roasted_marshmallows_total_l170_170687


namespace product_of_consecutive_integers_sqrt_50_l170_170844

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170844


namespace kiki_scarves_count_l170_170692

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l170_170692


namespace track_width_l170_170518

theorem track_width (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 10 * π) : r1 - r2 = 5 :=
sorry

end track_width_l170_170518


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170002

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170002


namespace average_of_six_starting_from_d_plus_one_l170_170740

theorem average_of_six_starting_from_d_plus_one (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (c + 6) = ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 := 
by 
-- Proof omitted; end with sorry
sorry

end average_of_six_starting_from_d_plus_one_l170_170740


namespace ratio_of_wire_lengths_l170_170544

theorem ratio_of_wire_lengths 
  (bonnie_wire_length : ℕ := 80)
  (roark_wire_length : ℕ := 12000) :
  bonnie_wire_length / roark_wire_length = 1 / 150 :=
by
  sorry

end ratio_of_wire_lengths_l170_170544


namespace original_number_of_men_l170_170109

theorem original_number_of_men (M : ℕ) : 
  (∀ t : ℕ, (t = 8) -> (8:ℕ) * M = 8 * 10 / (M - 3) ) -> ( M = 12 ) :=
by sorry

end original_number_of_men_l170_170109


namespace business_profit_l170_170755

variable (P : ℝ) -- Total profit

-- Conditions
def majority_share : Prop := P / 4
def remaining_profit : Prop := (3 / 4) * P
def partner_share : Prop := (3 / 16) * P
def combined_share : Prop := (P / 4) + 2 * ((3 / 16) * P) = 50000

theorem business_profit (P : ℝ) (h1 : majority_share P) (h2 : remaining_profit P) (h3 : partner_share P) (h4 : combined_share P) :
  P = 80000 :=
by
  sorry

end business_profit_l170_170755


namespace total_money_shared_l170_170902

def A_share (B : ℕ) : ℕ := B / 2
def B_share (C : ℕ) : ℕ := C / 2
def C_share : ℕ := 400

theorem total_money_shared (A B C : ℕ) (h1 : A = A_share B) (h2 : B = B_share C) (h3 : C = C_share) : A + B + C = 700 :=
by
  sorry

end total_money_shared_l170_170902


namespace inscribed_semicircle_radius_l170_170318

-- Define the properties of triangle ABC
structure Triangle :=
  (a b c : ℝ) -- sides
  (right_angle : Prop) -- right angle

-- Definition of the given right triangle
def ABC : Triangle :=
  { a := AC, b := BC, c := AB,
    right_angle := ∠C = π / 2 }

-- Prove that the radius of the inscribed semicircle is 10/3
theorem inscribed_semicircle_radius (AC BC : ℝ) (h1 : AC = 12) (h2 : BC = 5) (h3 : ∠C = π / 2) :
  ∃ r : ℝ, r = 10 / 3 := by
  sorry

end inscribed_semicircle_radius_l170_170318


namespace equation_of_common_chord_l170_170241

theorem equation_of_common_chord :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6 * p.1 - 7 = 0} in
  let C2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6 * p.2 - 27 = 0} in
  let A := {p : ℝ × ℝ | p ∈ C1 ∧ p ∈ C2} in
  ∃ A B : ℝ × ℝ, A ∈ C1 ∧ A ∈ C2 ∧ B ∈ C1 ∧ B ∈ C2 ∧ line_through A B = (λ p, 3 * p.1 - 3 * p.2 - 10 = 0) :=
sorry

end equation_of_common_chord_l170_170241


namespace longest_diagonal_segment_is_d_l170_170922

-- Define the given conditions
variables {A B C D : Type} -- Let the corners of the square be labeled A, B, C, D.
def square_area : ℝ := 30 -- The area of the square is 30 cm²
def triangle_area (base : ℝ) : ℝ := 2 * base -- Each segment corresponds to a base for triangles

-- Given areas for triangles with respective bases
def area_segment_a : ℝ := 2
def area_segment_b : ℝ := 3
def area_segment_c : ℝ := 1
def area_segment_d : ℝ := 5
def area_segment_e : ℝ := 4

-- Define a function to check the segment with the maximum area
def longest_segment (a b c d e : ℝ) : ℝ :=
max (max (max (max a b) c) d) e

-- The proof statement
theorem longest_diagonal_segment_is_d :
  longest_segment area_segment_a area_segment_b area_segment_c area_segment_d area_segment_e = area_segment_d :=
sorry -- Proof to be completed

end longest_diagonal_segment_is_d_l170_170922


namespace sum_of_digits_of_n_plus_1_l170_170699

-- Define the function S which sums the digits of a number n
def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

-- Assertion
theorem sum_of_digits_of_n_plus_1 {n : ℕ} (h : sum_of_digits n = 351) :
  sum_of_digits (n + 1) = 352 :=
by
  sorry

end sum_of_digits_of_n_plus_1_l170_170699


namespace remaining_payment_is_correct_l170_170643

def deposit_percentage: ℝ := 0.10
def deposit_amount: ℝ := 150
def total_cost: ℝ := deposit_amount / deposit_percentage
def remaining_amount_to_be_paid: ℝ := total_cost - deposit_amount

theorem remaining_payment_is_correct:
  remaining_amount_to_be_paid = 1350 := by
  sorry

end remaining_payment_is_correct_l170_170643


namespace relationship_between_a_b_c_l170_170205

noncomputable def a : ℝ := Real.exp (-2)

noncomputable def b : ℝ := a ^ a

noncomputable def c : ℝ := a ^ b

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by {
  sorry
}

end relationship_between_a_b_c_l170_170205


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170031

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170031


namespace measure_of_angle_x_l170_170574

-- Defining the conditions
def angle_ABC : ℝ := 108
def angle_ABD : ℝ := 180 - angle_ABC
def angle_in_triangle_ABD_1 : ℝ := 26
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove
theorem measure_of_angle_x (h1 : angle_ABD = 72)
                           (h2 : angle_in_triangle_ABD_1 = 26)
                           (h3 : sum_of_angles_in_triangle angle_ABD angle_in_triangle_ABD_1 x) :
  x = 82 :=
by {
  -- Since this is a formal statement, we leave the proof as an exercise 
  sorry
}

end measure_of_angle_x_l170_170574


namespace sum_log_series_eq_ln2_squared_l170_170966

theorem sum_log_series_eq_ln2_squared : 
  (∑ k in (Set.Ico 0 ∞ : Set ℕ), (3 * (Real.log (4 * k + 2)) / (4 * k + 2) 
    - Real.log (4 * k + 3) / (4 * k + 3) 
    - Real.log (4 * k + 4) / (4 * k + 4) 
    - Real.log (4 * k + 5) / (4 * k + 5))) = Real.log 2 ^ 2 :=
  sorry

end sum_log_series_eq_ln2_squared_l170_170966


namespace choose_president_and_vp_l170_170660

theorem choose_president_and_vp (n : ℕ) (h : n = 8) : 
  ∃ (ways : ℕ), ways = 8 * 7 :=
by {
  rw h,
  use 56,
  exact rfl,
  sorry
}

end choose_president_and_vp_l170_170660


namespace find_interest_rate_l170_170641

theorem find_interest_rate (P R_C G_B T : ℝ) (H1 : P = 2000) (H2 : R_C = 0.17) (H3 : G_B = 160) (H4 : T = 4) : 
  let I_C := P * R_C * T in
  let I_A := I_C - G_B in
  let R_A := (I_A / (P * T)) * 100 in
  R_A = 15 := 
by
  sorry

end find_interest_rate_l170_170641


namespace find_angle_A_l170_170301

-- Define the triangle and its angles and sides
variables (A B C : ℝ) (a b c : ℝ)
-- Condition: the sides of the triangle opposite to angles A, B, and C respectively
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Additional trigonometric condition given in the problem
variable (h : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A)

-- The main statement to prove
theorem find_angle_A (h : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A) : A = π / 6 :=
sorry

end find_angle_A_l170_170301


namespace correct_scientific_notation_l170_170134

def is_scientific_notation_correct (x : ℝ) (e : ℤ) : Prop :=
  if h : x ≠ 0 then 1 <= Math.abs x ∧ Math.abs x < 10 else False

theorem correct_scientific_notation :
  (¬ is_scientific_notation_correct 0.12 5) ∧
  (¬ is_scientific_notation_correct 12.5 2) ∧
  (¬ is_scientific_notation_correct 12306 0) ∧
  is_scientific_notation_correct 2.34 12 :=
by
  sorry

end correct_scientific_notation_l170_170134


namespace max_M_value_l170_170193

noncomputable def J_k (k : ℕ) : ℕ := if k > 0 then 10^(k+2) + 100 else 0

def M (k : ℕ) : ℕ :=
  have p := (10^(k+2) + 100).factorization,
  if 2 ∈ p.keys then p 2 else 0

theorem max_M_value : ∃ k, M k = 4 :=
by sorry

end max_M_value_l170_170193


namespace largest_tan_A_l170_170663

theorem largest_tan_A (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (h_right_triangle : ∀ x y z : ℝ, (x^2 + y^2 = z^2) → ∃ a b c : A, True)
  (AB BC : ℝ)
  (h_AB : AB = 25)
  (h_BC : BC = 20)
  : ∃ A B C : ℝ, tan A = 4 / 3 :=
begin
  sorry
end

end largest_tan_A_l170_170663


namespace find_min_value_l170_170977

noncomputable def f (x : ℝ) : ℝ := x - 9 / (2 - 2 * x)

theorem find_min_value : ∀ x : ℝ, 1 < x → f x ≥ 3 * real.sqrt 2 + 1 :=
by
  intro x h1
  sorry

end find_min_value_l170_170977


namespace sum_of_fractions_l170_170549

theorem sum_of_fractions :
  (\dfrac{2}{15} + \dfrac{4}{15} + \dfrac{6}{15} + \dfrac{8}{15} + \dfrac{10}{15} + \dfrac{30}{15} : ℚ) = 4 :=
by
  sorry

end sum_of_fractions_l170_170549


namespace exists_unique_decomposition_l170_170736

-- Definitions of A and B based on the decimal representations
def A : Set ℕ := {n | (∀ i, i % 2 = 1 → n.digits 10 !! i = some 0)}
def B : Set ℕ := {n | (∀ i, i % 2 = 0 → n.digits 10 !! i = some 0)}

-- The main theorem statement
theorem exists_unique_decomposition :
  ∃ (A B : Set ℕ), (∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b) :=
by {
  use [A, B], sorry
}

end exists_unique_decomposition_l170_170736


namespace no_integer_n_squared_plus_one_div_by_seven_l170_170734

theorem no_integer_n_squared_plus_one_div_by_seven (n : ℤ) : ¬ (n^2 + 1) % 7 = 0 := 
sorry

end no_integer_n_squared_plus_one_div_by_seven_l170_170734


namespace triangle_area_l170_170672

variables {α : Type*} [Real α] (a b c : α) (A B C S : α)

-- Define the conditions
def condition_1 := c = 2
def condition_2 := sin A = 2 * sin C
def condition_3 := cos B = 1 / 4

-- The proof statement
theorem triangle_area : condition_1 → condition_2 → condition_3 → S = sqrt 15 :=
by intros h1 h2 h3; sorry

end triangle_area_l170_170672


namespace alannah_more_books_than_beatrix_l170_170532

variable (A B Q : ℕ)

-- Given conditions
def condition1 : B = 30 := by rfl
def condition2 : Q = (6 * A) / 5 := by rfl
def condition3 : A + B + Q = 140 := by rfl

-- Prove the question equals the answer
theorem alannah_more_books_than_beatrix :
  B = 30 → 
  Q = (6 * A) / 5 → 
  A + B + Q = 140 → 
  A - B = 20 :=
by
  intros
  rw [condition1, condition2, condition3]
  sorry

end alannah_more_books_than_beatrix_l170_170532


namespace relationship_y1_y2_l170_170234

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem relationship_y1_y2 :
  ∀ (a b c x₀ x₁ x₂ : ℝ),
    (quadratic_function a b c 0 = 4) →
    (quadratic_function a b c 1 = 1) →
    (quadratic_function a b c 2 = 0) →
    1 < x₁ → 
    x₁ < 2 → 
    3 < x₂ → 
    x₂ < 4 → 
    (quadratic_function a b c x₁ < quadratic_function a b c x₂) :=
by 
  sorry

end relationship_y1_y2_l170_170234


namespace hyperbola_focus_distance_l170_170571

def distance_between_foci (a b c : ℝ) : ℝ := 2 * c

def compute_c (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem hyperbola_focus_distance :
  let h := 3 * x ^ 2 - 18 * x - 2 * y ^ 2 - 4 * y = 48
  ∃ (a b : ℝ), let c := compute_c a b in distance_between_foci a b c = 2 * real.sqrt (53 / 2) :=
sorry

end hyperbola_focus_distance_l170_170571


namespace incorrect_percentile_l170_170542

def scores : List ℕ := [7, 5, 9, 7, 4, 8, 9, 9, 7, 5]

def mode (l : List ℕ) : List ℕ := 
  let grouped := l.groupBy id
  let max_freq := grouped.foldl (λ acc x => max acc x.2.length) 0
  let modes := grouped.filter (λ x => x.2.length = max_freq) |>.map (λ x => x.1)
  modes

def mean (l : List ℕ) : ℕ := 
  l.foldl (·+·) 0 / l.length

def variance (l : List ℕ) : ℕ :=
  let m := mean l
  let sq_diffs := l.map (λ x => ((x - m) * (x - m)))
  sq_diffs.foldl (·+·) 0 / l.length

def percentile (l : List ℕ) (p : ℕ) : Float :=
  let sorted := l.qsort (· < ·)
  let index := (Float.ofNat l.length * (Float.ofNat p / 100)).toNat - 1
  0.5 * (sorted.get! index + sorted.get! (index + 1))

theorem incorrect_percentile : percentile scores 70 ≠ 8 :=
by
  simp only [percentile, scores]
  -- Skipping proof details
  sorry

end incorrect_percentile_l170_170542


namespace inscribed_circle_radius_is_2_l170_170048

-- Define the semiperimeter
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area of the triangle
def heron_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  Float.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the radius of the inscribed circle in terms of area and semiperimeter
def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  heron_area a b c / s

-- Prove the radius of the circle inscribed in triangle ABC is 2
theorem inscribed_circle_radius_is_2 : inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end inscribed_circle_radius_is_2_l170_170048


namespace problem_l170_170698

noncomputable def S : set (ℤ × ℤ × ℤ) :=
  {p | let ⟨x, y, z⟩ := p in 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 5}

def midpoint_in_S (a b : ℤ × ℤ × ℤ) : Prop :=
  let ⟨x1, y1, z1⟩ := a in
  let ⟨x2, y2, z2⟩ := b in
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 ∧ (z1 + z2) % 2 = 0

theorem problem (p q : ℕ) (h_rel_prime : nat.coprime p q)
    (h_probability : ∀ a b ∈ S, a ≠ b → ((p : ℚ) / q = 240 / 2256)) :
  p + q = 52 :=
sorry

end problem_l170_170698


namespace vector_dot_product_l170_170634

-- Declare vector space and necessary vector operations for a formal proof
variables {V : Type*} [inner_product_space ℝ V]

-- Vectors a and b
variables (a b : V)

-- Conditions
axiom magnitude_a : ∥a∥ = 3
axiom magnitude_b : ∥b∥ = 1
axiom equal_magnitudes : ∥a - b∥ = ∥a + b∥

-- Prove the target statement
theorem vector_dot_product : a ⋅ (a - b) = 9 :=
by sorry

end vector_dot_product_l170_170634


namespace complex_pure_imaginary_and_circle_graph_l170_170747

theorem complex_pure_imaginary_and_circle_graph (a : ℝ) : 
  (z : ℂ) (ha : z = a * (1 + complex.I) - 2 * complex.I) 
  (hz_pure_imaginary : complex.re z = 0) 
  (hz_norm : complex.abs z = 3) : 
  a = 2 ∧ (∃ c r, c = 0 ∧ r = 3 ∧ (∀ z : ℂ, complex.abs z = 3 ↔ z = (r * (cos θ + complex.I * sin θ) for some θ)) :=
sorry

end complex_pure_imaginary_and_circle_graph_l170_170747


namespace group_cyclic_of_two_cosets_l170_170553

variable {α : Type*} [Group α] (G : Group α) (H : Subgroup α)
  (finite_cosets : (Set (QuotientGroup.Quotient H)).Nonempty ∧ ∃ A B : QuotientGroup.Quotient H, A ≠ B ∧ ∀ x : QuotientGroup.Quotient H, x = A ∨ x = B)

theorem group_cyclic_of_two_cosets (hG_nontrivial : 1 < G.card) (infinite_order : ∀ x ∈ G, x ≠ 1 → orderOf x = 0) 
  (H_cyclic : Cyclic H)
  (two_cosets : ∃ A B : QuotientGroup.Quotient H, A ≠ B ∧ ∀ x : QuotientGroup.Quotient H, x = A ∨ x = B) :
  Cyclic G :=
by
  sorry

end group_cyclic_of_two_cosets_l170_170553


namespace initial_temperature_l170_170678

theorem initial_temperature (T : ℝ) : 
  let final_temp := ((2 * T - 30) * 0.70 + 24) in final_temp = 59 → T = 40 :=
by
  intros h
  sorry

end initial_temperature_l170_170678


namespace number_of_bushes_l170_170260

theorem number_of_bushes :
  let r := 15
  let spacing := 1.5
  let C := 2 * Real.pi * r
  let n := C / spacing
  Real.ceil n = 63 :=
by
  let r := 15
  let spacing := 1.5
  let C := 2 * Real.pi * r
  let n := C / spacing
  -- Sorry is used to skip the actual proof
  sorry

end number_of_bushes_l170_170260


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170026

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170026


namespace valid_license_plates_count_l170_170131

/--
The problem is to prove that the total number of valid license plates under the given format is equal to 45,697,600.
The given conditions are:
1. A valid license plate in Xanadu consists of three letters followed by two digits, and then one more letter at the end.
2. There are 26 choices of letters for each letter spot.
3. There are 10 choices of digits for each digit spot.

We need to conclude that the number of possible license plates is:
26^4 * 10^2 = 45,697,600.
-/

def num_valid_license_plates : Nat :=
  let letter_choices := 26
  let digit_choices := 10
  let total_choices := letter_choices ^ 3 * digit_choices ^ 2 * letter_choices
  total_choices

theorem valid_license_plates_count : num_valid_license_plates = 45697600 := by
  sorry

end valid_license_plates_count_l170_170131


namespace outfits_count_l170_170067

theorem outfits_count 
  (red_shirts : ℕ) (green_shirts : ℕ) (blue_shirts : ℕ) 
  (pants : ℕ) (blue_pants : ℕ)
  (red_hats : ℕ) (green_hats : ℕ) (blue_hats : ℕ) 
  (hr : red_shirts = 7) (hg : green_shirts = 6) (hb : blue_shirts = 7)
  (p : pants = 8) (bp : blue_pants = 3)
  (rh : red_hats = 10) (gh : green_hats = 9) (bh : blue_hats = 4)
  : (red_hats * (green_shirts + blue_shirts) * (pants - red_shirts) +
    green_hats * (red_shirts + blue_shirts) * (pants - green_shirts) +
    blue_hats * (red_shirts + green_shirts) * (pants - blue_pants)) = 2568 :=
by
  -- We substitute the given values
  have red_hat_outfits := red_hats * (green_shirts + blue_shirts) * (pants - red_shirts),
  have green_hat_outfits := green_hats * (red_shirts + blue_shirts) * (pants - green_shirts),
  have blue_hat_outfits := blue_hats * (red_shirts + green_shirts) * (pants - blue_pants),
  -- Prove the final total is 2568
  have outfits := red_hat_outfits + green_hat_outfits + blue_hat_outfits,
  rw [hr, hg, hb, p, bp, rh, gh, bh] at *,
  have eq1 : red_hat_outfits = 10 * (6 + 7) * 8 := rfl,
  have eq2 : green_hat_outfits = 9 * (7 + 7) * 8 := rfl,
  have eq3 : blue_hat_outfits = 4 * (7 + 6) * 10 := rfl,
  have eq4 : outfits = 1040 + 1008 + 520 := by rw [eq1, eq2, eq3],
  have eq5 : 1040 + 1008 + 520 = 2568 := rfl,
  show outfits = 2568 from eq4.trans eq5

end outfits_count_l170_170067


namespace number_of_pairs_l170_170600

theorem number_of_pairs (h : ∀ (a : ℝ) (b : ℕ), 0 < a → 2 ≤ b ∧ b ≤ 200 → (Real.log a / Real.log b) ^ 2017 = Real.log (a ^ 2017) / Real.log b) :
  ∃ n, n = 597 ∧ ∀ b : ℕ, 2 ≤ b ∧ b ≤ 200 → 
    ∃ a1 a2 a3 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 
      (Real.log a1 / Real.log b) = 0 ∧ 
      (Real.log a2 / Real.log b) = 2017^((1:ℝ)/2016) ∧ 
      (Real.log a3 / Real.log b) = -2017^((1:ℝ)/2016) :=
sorry

end number_of_pairs_l170_170600


namespace range_of_f_less_than_zero_l170_170206

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := log ((2 / (1 - x)) + a)

-- Define what it means for the function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem range_of_f_less_than_zero (a : ℝ) : is_odd_function (λ x, f x a) → set.Ioo (-1 : ℝ) 0 = {x : ℝ | f x a < 0} :=
by
  sorry

end range_of_f_less_than_zero_l170_170206


namespace pyramid_volume_correct_l170_170149

def volume_of_pyramid (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let mid_ab := ((fst v1 + fst v2) / 2, (snd v1 + snd v2) / 2)
  let mid_bc := ((fst v2 + fst v3) / 2, (snd v2 + snd v3) / 2)
  let mid_ca := ((fst v3 + fst v1) / 2, (snd v3 + snd v1) / 2)
  let centroid := ((fst v1 + fst v2 + fst v3) / 3, (snd v1 + snd v2 + snd v3) / 3)
  let base_area := 0.5 * Real.abs (fst v1 * (snd v2 - snd v3) + fst v2 * (snd v3 - snd v1) + fst v3 * (snd v1 - snd v2))
  let height := Real.abs (snd centroid)
  (1 / 3) * base_area * height

theorem pyramid_volume_correct :
  volume_of_pyramid (0,0) (30,0) (15,20) = 670 :=
sorry

end pyramid_volume_correct_l170_170149


namespace greatest_sum_of_consecutive_integers_product_lt_500_l170_170032

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l170_170032


namespace stable_performance_l170_170658

theorem stable_performance (s_A s_B s_C s_D : ℝ) (hA : s_A = 6) (hB : s_B = 5.5) (hC : s_C = 10) (hD : s_D = 3.8) :
  s_D < s_B ∧ s_D < s_A ∧ s_D < s_C :=
by
  rw [hA, hB, hC, hD]
  split
  apply lt_trans (lt_trans (by norm_num : 3.8 < 5.5) (by norm_num : 5.5 < 6)) (by norm_num : 6 < 10)
  apply lt_trans (by norm_num : 3.8 < 5.5) (by norm_num : 5.5 < 6)
  apply norm_num; sorry

end stable_performance_l170_170658


namespace sqrt_50_between_consecutive_integers_product_l170_170836

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170836


namespace probability_of_dart_in_circle_l170_170105

noncomputable def side_length : ℝ := 2 + Real.sqrt 2

noncomputable def octagon_area (s : ℝ) : ℝ :=
  2 * s^2 * (1 + Real.sqrt 2)

noncomputable def inscribed_circle_radius (s : ℝ) : ℝ :=
  s * (1 + Real.sqrt 2) / 2

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem probability_of_dart_in_circle :
  let s := side_length
  let r := inscribed_circle_radius s
  let circle_area := circle_area r
  let octagon_area := octagon_area s
  p : (circle_area / octagon_area) = p_correct :=
sorry

end probability_of_dart_in_circle_l170_170105


namespace star_assoc_l170_170345

variable {U : Type*}
variables (X Y Z : set U)

def star (A B : set U) : set U := A ∩ B

theorem star_assoc : star (star X Y) Z = (X ∩ Y) ∩ Z :=
by sorry

end star_assoc_l170_170345


namespace line_through_two_quadrants_l170_170282

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170282


namespace sqrt_50_between_consecutive_integers_product_l170_170837

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170837


namespace range_of_f_l170_170167

def f (x : ℝ) : ℝ := 2 ^ x - 1

theorem range_of_f : Set.range f = Set.Ici (-1) :=
by
  sorry

end range_of_f_l170_170167


namespace number_of_distinct_bad_arrangements_l170_170759

def is_bad_arrangement (l : List ℕ) : Prop :=
  l.perm [1, 2, 3, 4, 6] ∧
  ∀ n : ℕ, n > 0 ∧ n < 17 → 
    ¬ (∃ (a b : ℕ), a < b ∧ (finset.range (b - a) + a) = (finset.of_list l).sum (finset.range (b - a) + a))

def distinct_bad_arrangements : Finset (List ℕ) :=
  (List.permutations [1, 2, 3, 4, 6]).to_finset.filter is_bad_arrangement

theorem number_of_distinct_bad_arrangements : distinct_bad_arrangements.card = 2 := by
  sorry

end number_of_distinct_bad_arrangements_l170_170759


namespace product_of_consecutive_integers_sqrt_50_l170_170795

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l170_170795


namespace smallest_number_diminished_by_8_divisible_by_9_6_12_18_l170_170867

theorem smallest_number_diminished_by_8_divisible_by_9_6_12_18 :
  ∃ x : ℕ, (x - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 ∧ ∀ y : ℕ, (y - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 → x ≤ y → x = 44 :=
by
  sorry

end smallest_number_diminished_by_8_divisible_by_9_6_12_18_l170_170867


namespace cos_diff_trigonometric_identity_l170_170488

-- Problem 1
theorem cos_diff :
  (Real.cos (25 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) - 
   Real.cos (65 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  1/2 :=
sorry

-- Problem 2
theorem trigonometric_identity (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (Real.cos (2 * θ) - Real.sin (2 * θ)) / (1 + (Real.cos θ)^2) = 5/6 :=
sorry

end cos_diff_trigonometric_identity_l170_170488


namespace inscribed_circle_in_triangle_l170_170087

def Point : Type := ℝ × ℝ

-- Define the vertices of the triangle
def A : Point := (-2, 1)
def B : Point := (2, 5)
def C : Point := (5, 2)

-- Define the function that gives the equation of the inscribed circle
def inscribed_circle_eq (A B C : Point) : Prop :=
  ∃ h k r, (h = 2) ∧ (k = 3) ∧ (r = sqrt 2) ∧
    ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2

theorem inscribed_circle_in_triangle :
  inscribed_circle_eq A B C := 
  sorry

end inscribed_circle_in_triangle_l170_170087


namespace find_green_candies_l170_170437

variable (G : ℕ)

theorem find_green_candies (h1 : 3 / (G + 7) = 0.25) : G = 5 :=
by sorry

end find_green_candies_l170_170437


namespace concentric_circles_area_diff_l170_170438

open Real

def radius_large := 15
def radius_small := 7

def area_circle (r : ℝ) : ℝ := π * r^2

theorem concentric_circles_area_diff :
  area_circle radius_large - area_circle radius_small = 176 * π :=
by
  sorry

end concentric_circles_area_diff_l170_170438


namespace part1_part2_l170_170413

-- Definition of the conditions given
def february_parcels : ℕ := 200000
def april_parcels : ℕ := 338000
def monthly_growth_rate : ℝ := 0.3

-- Problem 1: Proving the monthly growth rate is 0.3
theorem part1 (x : ℝ) (h : february_parcels * (1 + x)^2 = april_parcels) : x = monthly_growth_rate :=
  sorry

-- Problem 2: Proving the number of parcels in May is less than 450,000 with the given growth rate
theorem part2 (h : monthly_growth_rate = 0.3 ) : february_parcels * (1 + monthly_growth_rate)^3 < 450000 :=
  sorry

end part1_part2_l170_170413


namespace lois_books_count_l170_170718

theorem lois_books_count 
  (initial_books : ℕ)
  (nephew_fraction : ℚ)
  (library_fraction : ℚ)
  (neighbor_fraction : ℚ)
  (purchased_books : ℕ) :
  initial_books = 120 →
  nephew_fraction = 1/4 →
  library_fraction = 1/5 →
  neighbor_fraction = 1/6 →
  purchased_books = 8 →
  let books_after_nephew := initial_books - (nephew_fraction * initial_books).toNat
  let books_after_library := books_after_nephew - (library_fraction * books_after_nephew).toNat
  let books_after_neighbor := books_after_library - (neighbor_fraction * books_after_library).toNat
  let final_books := books_after_neighbor + purchased_books
  in final_books = 68 :=
by
  intros initial_books_eq nephew_fraction_eq library_fraction_eq neighbor_fraction_eq purchased_books_eq
  rw [initial_books_eq, nephew_fraction_eq, library_fraction_eq, neighbor_fraction_eq, purchased_books_eq]

  let books_after_nephew := 120 - (1/4 * 120).toNat
  let books_after_library := books_after_nephew - (1/5 * books_after_nephew).toNat
  let books_after_neighbor := books_after_library - (1/6 * books_after_library).toNat
  let final_books := books_after_neighbor + 8

  have books_after_nephew_eq : books_after_nephew = 90 := by norm_num
  have books_after_library_eq : books_after_library = 72 := by norm_num
  have books_after_neighbor_eq : books_after_neighbor = 60 := by norm_num

  rw [books_after_nephew_eq, books_after_library_eq, books_after_neighbor_eq]
  norm_num
  sorry

end lois_books_count_l170_170718


namespace greatest_possible_sum_consecutive_product_lt_500_l170_170022

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l170_170022


namespace karen_locks_problem_l170_170338

theorem karen_locks_problem :
  let T1 := 5 in
  let T2 := 3 * T1 - 3 in
  let Combined_Locks_Time := 5 * T2 in
  Combined_Locks_Time = 60 := by
    let T1 := 5
    let T2 := 3 * T1 - 3
    let Combined_Locks_Time := 5 * T2
    sorry

end karen_locks_problem_l170_170338


namespace line_passing_through_first_and_third_quadrants_l170_170296

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l170_170296


namespace sqrt_50_between_consecutive_integers_product_l170_170834

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l170_170834


namespace sum_of_g_is_zero_l170_170705

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_of_g_is_zero :
  (Finset.range 2022).sum (λ k => (-1)^(k + 1) * g ((k + 1 : ℝ) / 2023)) = 0 :=
by
  sorry

end sum_of_g_is_zero_l170_170705


namespace total_toys_is_correct_l170_170330

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l170_170330


namespace terminating_decimal_of_fraction_l170_170954

theorem terminating_decimal_of_fraction (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 624) : 
  (∃ m : ℕ, 10^m * (n / 625) = k) → ∃ m, m = 624 :=
sorry

end terminating_decimal_of_fraction_l170_170954


namespace area_of_region_unique_solution_for_a_l170_170630

-- Part (a)
theorem area_of_region :
  ∀ x y : ℝ, 
  (|9 + 8 * y - x^2 - y^2| + |8 * y| = 16 * y + 9 - x^2 - y^2)
  → (regionArea = 25 * real.pi - 25 * real.arcsin(0.6) + 12) :=
sorry

-- Part (b)
theorem unique_solution_for_a (x y a : ℝ) :
  (|9 + 8 * y - x^2 - y^2| + |8 * y| = 16 * y + 9 - x^2 - y^2)
  ∧ ((a + 4) * x - 13 * y + a = 0)
  → (a = -3 ∨ a = -6) :=
sorry

end area_of_region_unique_solution_for_a_l170_170630


namespace points_collinear_l170_170934

variable {A B C D E F : Type*} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty E] [Nonempty F]

-- Definitions of the points and the triangle condition
variables {A B C : Point}
variables (h_isosceles : IsIsoscelesTriangle A B C)
variables (circumscribed_circle : CircumscribedCircle A B C)
variables (tangent_B : IsTangent circumscribed_circle B)
variables {D : Point}
variables (h_CD_perpendicular : Perpendicular CD tangent_B)
variables {E : Point}
variables {F : Point}
variables (h_AE_altitude : IsAltitude AE)
variables (h_BF_altitude : IsAltitude BF)

-- The main statement we need to prove
theorem points_collinear : Collinear D E F :=
by 
  sorry

end points_collinear_l170_170934


namespace rational_root_l170_170569

def polynomial : ℚ[X] := 6 * X^4 - 5 * X^3 - 17 * X^2 + 7 * X + 3

theorem rational_root : polynomial.eval (1/2) polynomial = 0 :=
by sorry

end rational_root_l170_170569


namespace shopkeeper_loss_percent_is_56_l170_170106

-- The initial assumption about the goods' value
def initial_value : ℝ := 100

-- The profit margin of the shopkeeper as a percentage
def profit_margin : ℝ := 0.10

-- The theft loss as a percentage
def theft_loss : ℝ := 0.60

-- The value of the goods after theft
def remaining_goods_value (initial_value : ℝ) (loss : ℝ) : ℝ :=
  initial_value * (1 - loss)

-- The selling price with profit before theft
def selling_price_with_profit (initial_value : ℝ) (profit : ℝ) : ℝ :=
  initial_value * (1 + profit)

-- The selling price of the remaining goods after theft
def selling_price_of_remaining_goods (initial_value selling_price : ℝ) (loss : ℝ) : ℝ :=
  selling_price * (1 - loss)

-- The shopkeeper's loss in absolute value
def absolute_loss (initial_value remaining_value : ℝ) : ℝ :=
  initial_value - remaining_value

-- The loss percent
def loss_percent (loss initial_value : ℝ) : ℝ :=
  (loss / initial_value) * 100

-- Proof that the shopkeeper's loss percent is 56%
theorem shopkeeper_loss_percent_is_56 :
  ∀ (iv : ℝ) (pm tl : ℝ),
    iv = 100 → pm = 0.10 → tl = 0.60 →
    loss_percent (absolute_loss iv (selling_price_of_remaining_goods iv (selling_price_with_profit iv pm) tl)) iv = 56 := 
by 
  intros iv pm tl h1 h2 h3
  rw [h1, h2, h3, remaining_goods_value, selling_price_with_profit, selling_price_of_remaining_goods, absolute_loss, loss_percent]
  norm_num
  sorry

end shopkeeper_loss_percent_is_56_l170_170106


namespace initial_white_cookies_l170_170155

theorem initial_white_cookies (B W : ℕ) 
  (h1 : B = W + 50)
  (h2 : (1 / 2 : ℚ) * B + (1 / 4 : ℚ) * W = 85) :
  W = 80 :=
by
  sorry

end initial_white_cookies_l170_170155


namespace product_of_consecutive_integers_sqrt_50_l170_170848

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170848


namespace unique_solution_2023_plus_2_pow_n_eq_k_sq_l170_170953

theorem unique_solution_2023_plus_2_pow_n_eq_k_sq (n k : ℕ) (h : 2023 + 2^n = k^2) :
  (n = 1 ∧ k = 45) :=
by
  sorry

end unique_solution_2023_plus_2_pow_n_eq_k_sq_l170_170953


namespace man_speed_is_six_kmph_l170_170926

noncomputable def speed_of_man (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kmph * 1000) / 3600 in
  let relative_speed := train_length / time_seconds in
  let man_speed_mps := relative_speed - train_speed_mps in
  man_speed_mps * (3600 / 1000)

theorem man_speed_is_six_kmph :
  speed_of_man 240 60 13.090909090909092 = 6 :=
by
  sorry

end man_speed_is_six_kmph_l170_170926


namespace geometric_sequence_expression_l170_170670

variable {a : ℕ → ℝ}

-- Define the geometric sequence property
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_expression :
  is_geometric a q →
  a 3 = 2 →
  a 6 = 16 →
  ∀ n, a n = 2^(n-2) := by
  intros h_geom h_a3 h_a6
  sorry

end geometric_sequence_expression_l170_170670


namespace no_illumination_l170_170503

noncomputable def is_illuminated (t : ℝ) (c : ℝ) : Prop :=
  let x := 3
  let y := 3 + sin t * cos t - sin t - cos t
  y = c * x

theorem no_illumination (c : ℝ) (h1 : c > 0) (h2 : c < 1/2 ∨ c > 7/6) :
  ¬ ∃ t : ℝ, is_illuminated t c :=
by
  sorry

end no_illumination_l170_170503


namespace find_stream_speed_l170_170874

noncomputable def speed_of_stream
    (V_b V_s : ℝ)
    (downstream_distance upstream_distance : ℝ)
    (downstream_time upstream_time : ℝ)
    (downstream_speed upstream_speed : ℝ) : Prop :=
  (downstream_distance = downstream_speed * downstream_time) ∧
  (upstream_distance = upstream_speed * upstream_time) ∧
  (downstream_speed = V_b + V_s) ∧
  (upstream_speed = V_b - V_s)

theorem find_stream_speed :
  ∃ V_s : ℝ, ∀ V_b : ℝ,
    speed_of_stream V_b V_s 130 75 10 15 13 5 → V_s = 4 :=
begin
  sorry
end

end find_stream_speed_l170_170874


namespace complement_intersection_A_B_l170_170239

-- Define the universal set I
def I : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x,y) }

-- Define set A
def A : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }

-- Define set B
def B : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }

-- Define the complement of set A in the universal set I
def C_I_A : set (ℝ × ℝ) := I \ A

-- Define the intersection of the complement of set A and set B
def C_I_A_inter_B : set (ℝ × ℝ) := C_I_A ∩ B

-- State the theorem to be proved
theorem complement_intersection_A_B : C_I_A_inter_B = { (2, 3) } :=
by
  sorry

end complement_intersection_A_B_l170_170239


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170045

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170045


namespace sqrt_50_between_7_and_8_l170_170804

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l170_170804


namespace certain_number_l170_170869

theorem certain_number (n : ℕ) : 
  (55 * 57) % n = 6 ∧ n = 1043 :=
by
  sorry

end certain_number_l170_170869


namespace solve_for_g_l170_170162

theorem solve_for_g (g : ℤ[X]) :
  (∀ x : ℤ, 2 * x^5 + 3 * x^3 - 4 * x + 1 + g.eval x = 4 * x^4 - 9 * x^3 + 2 * x^2 + 5)
  → g = -2 * X^5 + 4 * X^4 - 12 * X^3 + 2 * X^2 + 4 * X + 4 :=
  by
  intro h
  sorry

end solve_for_g_l170_170162


namespace trihedral_angle_equality_l170_170391

structure TrihedralAngle :=
(vertex : Point) -- Placeholder for a point structure
(planes : set Plane) -- Placeholder for a set of planes

def dihedral_angle (p1 p2 : Plane) : Angle :=
sorry -- Placeholder for the dihedral angle calculation

theorem trihedral_angle_equality (A B : TrihedralAngle)
  (α₁ β₁ γ₁ : Angle)
  (α₂ β₂ γ₂ : Angle)
  (h1 : ∀ p1 p2 ∈ A.planes, dihedral_angle p1 p2 = α₁ ∨ dihedral_angle p1 p2 = β₁ ∨ dihedral_angle p1 p2 = γ₁)
  (h2 : ∀ p1 p2 ∈ B.planes, dihedral_angle p1 p2 = α₂ ∨ dihedral_angle p1 p2 = β₂ ∨ dihedral_angle p1 p2 = γ₂)
  (h_eq : α₁ = α₂ ∧ β₁ = β₂ ∧ γ₁ = γ₂) :
  A = B :=
sorry

end trihedral_angle_equality_l170_170391


namespace line_through_two_quadrants_l170_170281

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170281


namespace max_tickets_jane_can_buy_l170_170984

theorem max_tickets_jane_can_buy
  (ticket_cost : ℕ → ℕ)
  (jane_money : ℕ)
  (promotion_threshold : ℕ)
  (regular_price : ℕ)
  (discounted_price : ℕ)
  (total_cost : ∀ n : ℕ, n ≤ promotion_threshold → ℕ)
  (total_cost_discounted : ∀ n : ℕ, n > promotion_threshold → ℕ):
  ∀ n, 
    (total_cost n jane_money ≤ total_cost_discounted n jane_money → n ≤ promotion_threshold)
    ∧ (total_cost_discounted n jane_money > total_cost_discounted n jane_money - 1 → n ≤ promotion_threshold + ((jane_money - (promotion_threshold * regular_price)) / discounted_price)) 
    → n ≤ 19 :=
by
  sorry

end max_tickets_jane_can_buy_l170_170984


namespace M_plus_2N_l170_170203

variable (m : ℝ)

def M : ℝ := -m^2 + 3 * m - 4
def N : ℝ := 2 * m^2 - 5 * m + 8

theorem M_plus_2N : M + 2 * N = 3 * m^2 - 7 * m + 12 := by
  sorry

end M_plus_2N_l170_170203


namespace digits_sum_odd_l170_170379

theorem digits_sum_odd (n : ℕ) (hn : n > 0) :
  let digits : ℕ → ℕ := λ m, (Real.floor (Real.log10 m) + 1).toNat in
  Odd (digits (4^n) + digits (25^n)) :=
by
  sorry

end digits_sum_odd_l170_170379


namespace problem1_solution_problem2_solution_l170_170487

noncomputable def problem1 (x y : ℝ) : ℝ :=
  4 * x^(1/4) * (-3 * x^(1/4) * y^(-1/3)) / (-6 * x^(-1/2) * y^(-2/3))

theorem problem1_solution (x y : ℝ) : problem1 x y = 2 * x * y^(1/3) :=
by
  sorry

noncomputable def problem2 : ℝ :=
  real.log 5 / real.log 10 + real.log 2 / real.log 10 - ((-1/3)^(-2)) + (real.sqrt 2 - 1)^0 + (3 : ℝ)

theorem problem2_solution : problem2 = -4 :=
by
  sorry

end problem1_solution_problem2_solution_l170_170487


namespace find_lambda_l170_170242

variables {R : Type*} [linear_ordered_field R] {V : Type*} [add_comm_group V] [module R V]

variables (e1 e2 : V) (a b : V) (λ : R)

-- Conditions
def non_collinear (e1 e2 : V) : Prop := ¬∃ μ : R, e1 = μ • e2
def a_def (e1 e2 : V) : V := 2 • e1 - 3 • e2
def b_def (e1 e2 : V) (λ : R) : V := λ • e1 + 6 • e2
def collinear (a b : V) : Prop := ∃ μ : R, a = μ • b

-- Proof Problem Statement
theorem find_lambda (h_non_collinear : non_collinear e1 e2) (h_a : a = a_def e1 e2) (h_b : b = b_def e1 e2 λ) (h_collinear : collinear a b) : λ = -4 := by
  -- Proof steps will be here
  sorry

end find_lambda_l170_170242


namespace ball_placement_count_l170_170199

theorem ball_placement_count :
  let balls := {1, 2, 3, 4, 5, 6}
  let boxes := {A, B, C, D}
  let valid_placement (placement : balls → boxes) : Prop :=
    (placement 2 ≠ B) ∧ (placement 4 ≠ D)
  ∃ placement : balls → boxes, valid_placement placement ∧ finset.card {p : balls × boxes | valid_placement (λ b, if b = p.1 then p.2 else placement b)} = 252 :=
sorry

end ball_placement_count_l170_170199


namespace find_m_n_condition_l170_170969

theorem find_m_n_condition (m n : ℕ) :
  m ≥ 1 ∧ n > m ∧ (42 ^ n ≡ 42 ^ m [MOD 100]) ∧ m + n = 24 :=
sorry

end find_m_n_condition_l170_170969


namespace product_of_consecutive_integers_sqrt_50_l170_170822

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170822


namespace ethan_arianna_apart_l170_170564

def ethan_distance := 1000 -- the distance Ethan ran
def arianna_distance := 184 -- the distance Arianna ran

theorem ethan_arianna_apart : ethan_distance - arianna_distance = 816 := by
  sorry

end ethan_arianna_apart_l170_170564


namespace converse_prop_parabola_focus_distance_geom_sequence_hyperbola_focus_distance_l170_170890

-- 1. Proof of the converse proposition.
theorem converse_prop (x : ℝ) : x^2 > 1 → x < -1 ∨ x > 1 :=
by
  sorry

-- 2. Proof of the distance from P to the focus of the parabola.
theorem parabola_focus_distance (a b : ℝ) (h₁ : b^2 = 4 * a) (h₂ : abs (a + 2) = 6) : sqrt ((a - 1)^2 + b^2) = 5 :=
by
  sorry

-- 3. Proof for the geometric sequence property.
variables {a : ℕ → ℝ}
theorem geom_sequence (h₁ : ∃ x y, x^2 - 6 * x + 8 = 0 ∧ y^2 - 6 * y + 8 = 0 ∧ (a 3 = x ∨ a 3 = y) ∧ (a 15 = x ∨ a 15 = y))
  (h₂ : ∀ n, a (n + 1) / a n = r) : a 1 * a 17 / a 9 = 2 * sqrt 2 :=
by
  sorry

-- 4. Proof for hyperbola.
theorem hyperbola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ) (h₁ : A = (1, 4)) 
  (h₂ : C = (λ x y, x^2 / 4 - y^2 / 12 = 1)) (h₃ : some_minimized_condition) : distance_to_line F (line_through_points A P) = 32 / 5 :=
by
  sorry

end converse_prop_parabola_focus_distance_geom_sequence_hyperbola_focus_distance_l170_170890


namespace construct_parallelogram_l170_170949

-- Define the problem parameters
variables (α ε : ℝ) (BD_length : ℝ)

-- The main statement we need to prove
theorem construct_parallelogram (hα : α > 0 ∧ α < pi)
                               (hε : ε > 0 ∧ ε < pi)
                               (hBD : BD_length > 0) :
  ∃ (A B C D : ℝ × ℝ), 
  (is_parallelogram A B C D) ∧ 
  (∠BAC = α) ∧ 
  (angle_between_diagonals A C B D = ε) ∧ 
  (length (B, D) = BD_length) :=
sorry

end construct_parallelogram_l170_170949


namespace range_of_s_l170_170462

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_of_s : set.range s = set.Ioi 0 := 
by 
  sorry

end range_of_s_l170_170462


namespace range_of_g_l170_170695

noncomputable def g (x : ℝ) : ℝ := (Real.arctan x)^3 + (Real.arccot x)^3

theorem range_of_g : 
  let inf := (Real.pi^3) / 32
  let sup := (3 * Real.pi^3) / 8
  Set.Icc inf sup = {y | ∃ x : ℝ, g x = y} :=
by
  sorry

end range_of_g_l170_170695


namespace product_of_consecutive_integers_sqrt_50_l170_170846

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l170_170846


namespace parameter_range_exists_solution_l170_170570

theorem parameter_range_exists_solution :
  (∃ b : ℝ, -14 < b ∧ b < 9 ∧ ∃ a : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * b * (b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)) :=
sorry

end parameter_range_exists_solution_l170_170570


namespace integer_solution_in_range_l170_170456

theorem integer_solution_in_range :
  ∃ (n : ℕ), 0 ≤ n ∧ n < 23 ∧ -300 ≡ n [MOD 23] :=
by {
  use 22,
  split,
  exact Nat.zero_le 22,
  split,
  norm_num,
  norm_num,
  exact Nat.modeq.trans (Nat.modeq.symm (Nat.modeq.sub_right 22 (Nat.modeq.symm (Nat.modeq.modeq_of_dvd (dvd_sub (-300) (22) 23 (by norm_num) (by norm_num))))))
      (Nat.modeq_of_modeq_of_dvd (Nat.modeq.trans (Nat.modeq.sub_right -300 (Nat.modeq.modeq_of_dvd (dvd_mul_right 23 10))) (Nat.modeq_of_dvd (dvd_refl 23))))
}

end integer_solution_in_range_l170_170456


namespace product_of_integers_around_sqrt_50_l170_170772

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170772


namespace scientific_notation_l170_170402

theorem scientific_notation :
  (0.0000064 : ℝ) = 6.4 * 10^(-6) :=
by
  sorry

end scientific_notation_l170_170402


namespace product_of_consecutive_integers_sqrt_50_l170_170814

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170814


namespace scientific_notation_l170_170403

theorem scientific_notation :
  (0.0000064 : ℝ) = 6.4 * 10^(-6) :=
by
  sorry

end scientific_notation_l170_170403


namespace product_of_consecutive_integers_sqrt_50_l170_170816

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170816


namespace monotonic_intervals_of_f_range_of_m_for_unique_solution_l170_170223

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (1 + a) * x + Real.log x

-- Monotonic intervals specification
theorem monotonic_intervals_of_f (a : ℝ) (h : 0 ≤ a) :
  (∀ x : ℝ, 0 < x → derivative (λ x, f a x) x ≥ 0 ↔
   (a = 0 ∧ 0 < x ∧ x < 1) ∨
   (0 < a ∧ a < 1 ∧ 0 < x ∧ x < 1) ∨
   (0 < a ∧ a < 1 ∧ x > 1 / a) ∨ 
   (a = 1) ∨
   (a > 1 ∧ 0 < x ∧ x < 1 / a) ∨
   (a > 1 ∧ x > 1)) ∧

  (∀ x : ℝ, 0 < x → derivative (λ x, f a x) x ≤ 0 ↔ 
   (a = 0 ∧ x > 1) ∨
   (0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
   (a > 1 ∧ 1 / a < x ∧ x < 1)) :=
sorry

-- Range of values for m given the unique solution condition
theorem range_of_m_for_unique_solution (a : ℝ) (h : a = 0) :
  ∀ (m : ℝ), (∃ unique x : ℝ, x ∈ Icc 1 (Real.exp 2) ∧ f a x = m * x) ↔
  (m = (Real.log 1 - 1) ∧ m = Real.log 1) ∨
  (m ≥ -1 ∧ m < (2 / Real.exp 2) - 1 ∨ m = (1 / Real.exp 1) - 1) :=
sorry

end monotonic_intervals_of_f_range_of_m_for_unique_solution_l170_170223


namespace problem_solution_l170_170190

theorem problem_solution : ∃ a b : ℝ, (a = 3) ∧ (b = -4) ∧ ∀ X : ℝ, (X - 1)^2 ∣ a * X^4 + b * X^3 + 1 :=
by
  use 3, -4 
  split
  { reflexivity }
  split
  { reflexivity }
  intro X
  use (3 * X^3 - X^2 + X - 4)
  sorry

end problem_solution_l170_170190


namespace problem_solved_l170_170229

-- Define the function f with the given conditions
def satisfies_conditions(f : ℝ × ℝ × ℝ → ℝ) :=
  (∀ x y z t : ℝ, f (x + t, y + t, z + t) = t + f (x, y, z)) ∧
  (∀ x y z t : ℝ, f (t * x, t * y, t * z) = t * f (x, y, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (y, x, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (x, z, y))

-- We'll state the main result to be proven, without giving the proof
theorem problem_solved (f : ℝ × ℝ × ℝ → ℝ) (h : satisfies_conditions f) : f (2000, 2001, 2002) = 2001 :=
  sorry

end problem_solved_l170_170229


namespace largest_4_digit_number_divisible_by_24_l170_170457

theorem largest_4_digit_number_divisible_by_24 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 24 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 24 = 0 → m ≤ n :=
sorry

end largest_4_digit_number_divisible_by_24_l170_170457


namespace gallons_of_water_removed_l170_170850

theorem gallons_of_water_removed
  (length width lowering_inch : ℝ) (cubic_ft_to_gallons : ℝ)
  (h_length : length = 20)
  (h_width : width = 25)
  (h_lowering_inch : lowering_inch = 6)
  (h_conversion : cubic_ft_to_gallons = 7.5) :
  let lowering_ft := lowering_inch / 12
      volume_cubic_ft := length * width * lowering_ft
      gallons := volume_cubic_ft * cubic_ft_to_gallons in
  gallons = 3750 :=
begin
  sorry
end

end gallons_of_water_removed_l170_170850


namespace triangle_b_value_triangle_area_value_l170_170323

noncomputable def triangle_b (a : ℝ) (cosA : ℝ) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := cosA
  (a * sinB) / sinA

noncomputable def triangle_area (a b c : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Given conditions
variable (A B : ℝ) (a : ℝ := 3) (cosA : ℝ := Real.sqrt 6 / 3) (B := A + Real.pi / 2)

-- The assertions to prove
theorem triangle_b_value :
  triangle_b a cosA = 3 * Real.sqrt 2 :=
sorry

theorem triangle_area_value :
  triangle_area 3 (3 * Real.sqrt 2) 1 (1 / 3) = (3 * Real.sqrt 2) / 2 :=
sorry

end triangle_b_value_triangle_area_value_l170_170323


namespace range_of_s_l170_170463

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 2

theorem range_of_s : set.range s = set.Ioi 0 := 
by 
  sorry

end range_of_s_l170_170463


namespace distance_between_foci_l170_170404

-- Define the conditions
def is_asymptote (y x : ℝ) (slope intercept : ℝ) : Prop := y = slope * x + intercept

def passes_through_point (x y x0 y0 : ℝ) : Prop := x = x0 ∧ y = y0

-- The hyperbola conditions
axiom asymptote1 : ∀ x y : ℝ, is_asymptote y x 2 3
axiom asymptote2 : ∀ x y : ℝ, is_asymptote y x (-2) 5
axiom hyperbola_passes : passes_through_point 2 9 2 9

-- The proof problem statement: distance between the foci
theorem distance_between_foci : ∀ {a b c : ℝ}, ∃ c, (c^2 = 22.75 + 22.75) → 2 * c = 2 * Real.sqrt 45.5 :=
by
  sorry

end distance_between_foci_l170_170404


namespace serve_jelly_l170_170906

def amount_of_jelly : ℚ := 37 + 2 / 3
def one_serving : ℚ := 1 + 1 / 2
def number_of_servings : ℚ := 25 + 1 / 9

theorem serve_jelly :
  amount_of_jelly / one_serving = number_of_servings :=
by
  sorry

end serve_jelly_l170_170906


namespace Madelyn_daily_pizza_expense_l170_170937

theorem Madelyn_daily_pizza_expense (total_expense : ℕ) (days_in_may : ℕ) 
  (h1 : total_expense = 465) (h2 : days_in_may = 31) : 
  total_expense / days_in_may = 15 := 
by
  sorry

end Madelyn_daily_pizza_expense_l170_170937


namespace line_through_two_quadrants_l170_170285

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170285


namespace sine_angle_AF_tangent_l170_170415

def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

def derivative_parabola (x : ℝ) : ℝ := (1 / 2) * x

def focus : ℝ × ℝ := (0, 1)

def slope_tangent_at (A : ℝ × ℝ) : ℝ := 2

def point_A : ℝ × ℝ := (4, 4)

theorem sine_angle_AF_tangent :
  let AF_slope := (point_A.2 - focus.2) / (point_A.1 - focus.1),
      theta_tangent := 2,
      theta_AF := AF_slope,
      tan_theta := (theta_tangent - theta_AF) / (1 + theta_tangent * theta_AF),
      sin_theta := math.sqrt((tan_theta ^ 2) / (1 + tan_theta ^ 2))
  in sin_theta = (math.sqrt 5) / 5 :=
sorry

end sine_angle_AF_tangent_l170_170415


namespace triangle_ABC_proof_l170_170322

theorem triangle_ABC_proof :
  ∀ (a b c : ℝ) (A B C : ℝ),
  b = √7 →
  a + c = 5 →
  cos (2 * B) + cos B = 0 →
  B = π / 3 ∧ 
  (0 < a ∧ 0 < b ∧ 0 < c) ∧
  ∀ (S : ℝ), S = (a * c * (real.sin B)) / 2 →
  S = (3 * real.sqrt 3) / 2 := 
by
  sorry

end triangle_ABC_proof_l170_170322


namespace find_i_value_for_S_i_l170_170671

theorem find_i_value_for_S_i :
  ∃ (i : ℕ), (3 * 6 - 2 ≤ i ∧ i < 3 * 6 + 1) ∧ (1000 ≤ 31 * 2^6) ∧ (31 * 2^6 ≤ 3000) ∧ i = 2 :=
by sorry

end find_i_value_for_S_i_l170_170671


namespace exists_infinitely_many_good_sequences_l170_170120

def is_good_sequence (N : ℕ) (as : Fin N → ℕ) : Prop :=
  ∃ (i j : Fin N), i ≠ j ∧ (∑ k in Finset.filter (λ k, k ≠ i ∧ k ≠ j) Finset.univ, as k) ∣ as i * as j

theorem exists_infinitely_many_good_sequences (N : ℕ) (h_even : Even N) (h_N_ge_4 : 4 ≤ N) :
  ∃ (as : ℕ → Fin N → ℕ), ∀ n, is_good_sequence N (as n) :=
sorry

end exists_infinitely_many_good_sequences_l170_170120


namespace geometric_sequence_condition_l170_170319

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (∀ n ≥ 2, a n = 2 * a (n-1)) → 
  (∃ r, r = 2 ∧ ∀ n ≥ 2, a n = r * a (n-1)) ∧ 
  (∃ b, b ≠ 0 ∧ ∀ n, a n = 0) :=
sorry

end geometric_sequence_condition_l170_170319


namespace k_positive_first_third_quadrants_l170_170291

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l170_170291


namespace greatest_sum_of_consecutive_integers_product_less_500_l170_170000

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l170_170000


namespace spider_dressing_orders_l170_170125

open Function

theorem spider_dressing_orders :
  let socks := 8
  let shoes := 8
  let total_actions := socks + shoes
  factorial total_actions / 2 ^ socks = 81729648000 :=
by
  let socks := 8
  let shoes := 8
  let total_actions := socks + shoes
  have h_factorial := Nat.factorial total_actions
  have h_divisor := 2 ^ socks
  show h_factorial / h_divisor = 81729648000
  sorry

end spider_dressing_orders_l170_170125


namespace sum_of_squares_eq_l170_170396

theorem sum_of_squares_eq (x y : Fin 10 → ℕ) (h : ∀ i, x i + y i = 9) :
  (∑ i, (x i) ^ 2) = ∑ i, (y i) ^ 2 :=
sorry

end sum_of_squares_eq_l170_170396


namespace max_value_of_3x_plus_4y_l170_170216

theorem max_value_of_3x_plus_4y (x y : ℝ) 
(h : x^2 + y^2 = 14 * x + 6 * y + 6) : 
3 * x + 4 * y ≤ 73 := 
sorry

end max_value_of_3x_plus_4y_l170_170216


namespace find_point_A_coordinates_l170_170506

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l170_170506


namespace solve_vector_q_l170_170364

-- Define the vectors and the tensor product operation
def vector : Type := ℝ × ℝ

def tensor (a b : vector) : vector :=
  (a.1 * b.1, a.2 * b.2)

-- The condition
variables p q : vector
def p := (1, 2)

-- The question and answer
def proof_question (q : vector) (condition : tensor p q = (-3, -4)) : Prop :=
  q = (-3, -2)

-- The statement to be proved
theorem solve_vector_q : ∃ q : vector, proof_question q (tensor p q = (-3, -4)) :=
by sorry

end solve_vector_q_l170_170364


namespace area_of_smaller_circle_l170_170450

noncomputable def radius_smaller_circle (r : ℝ) (PA AB : ℝ) (three_times_smaller : ℝ) : Prop :=
  PA = 6 ∧ AB = 6 ∧ three_times_smaller = 3 * r

theorem area_of_smaller_circle {r PA AB three_times_smaller : ℝ} 
  (h : radius_smaller_circle r PA AB three_times_smaller) : 
  ∃ (A : ℝ), A = π * (36 / 7) :=
by
  obtain ⟨hPA, hAB, hthree⟩ := h
  have h_r : r^2 = 36 / 7 := 
    by
      sorry -- Proof using Pythagorean Theorem here
  use π * (36 / 7)
  rw ← h_r
  sorry

end area_of_smaller_circle_l170_170450


namespace zara_owns_113_goats_l170_170072

-- Defining the conditions
def cows : Nat := 24
def sheep : Nat := 7
def groups : Nat := 3
def animals_per_group : Nat := 48

-- Stating the problem, with conditions as definitions
theorem zara_owns_113_goats : 
  let total_animals := groups * animals_per_group in
  let cows_and_sheep := cows + sheep in
  let goats := total_animals - cows_and_sheep in
  goats = 113 := by
  sorry

end zara_owns_113_goats_l170_170072


namespace number_of_satisfying_integers_l170_170559

def satisfies_condition (n : ℕ) : Prop := 
  1 ≤ n ∧ n ≤ 50 ∧ (nat.factorial (n^3 - 1)) % (nat.factorial n)^(n + 2) = 0

theorem number_of_satisfying_integers : 
  (finset.filter satisfies_condition (finset.range 51)).card = 2 := 
by
  sorry

end number_of_satisfying_integers_l170_170559


namespace total_wet_surface_area_correct_l170_170482

-- Define the dimensions and depth of the cistern
def cistern_length : ℝ := 8
def cistern_width : ℝ := 6
def water_depth : ℝ := 1 + 25/100  -- 1 m 25 cm converted to meters

-- Calculate the areas involved
def bottom_surface_area : ℝ := cistern_length * cistern_width
def longer_side_wall_area : ℝ := 2 * (water_depth * cistern_length)
def shorter_side_wall_area : ℝ := 2 * (water_depth * cistern_width)
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_wall_area + shorter_side_wall_area

-- State the proof goal
theorem total_wet_surface_area_correct : total_wet_surface_area = 83 := by
  sorry

end total_wet_surface_area_correct_l170_170482


namespace solve_quadratic_inequality_l170_170392

theorem solve_quadratic_inequality :
  { x : ℝ | -3 * x^2 + 8 * x + 5 < 0 } = { x : ℝ | x < -1 ∨ x > 5 / 3 } :=
sorry

end solve_quadratic_inequality_l170_170392


namespace min_value_of_cos_sum_l170_170757

theorem min_value_of_cos_sum : ∃ x : ℝ, (cos (3*x + π/6) + cos (3*x - π/3)) = -√2 :=
by {
  sorry
}

end min_value_of_cos_sum_l170_170757


namespace license_plate_calculation_l170_170248

def license_plate_count : ℕ :=
  let letter_choices := 26^3
  let first_digit_choices := 5
  let remaining_digit_combinations := 5 * 5
  letter_choices * first_digit_choices * remaining_digit_combinations

theorem license_plate_calculation :
  license_plate_count = 455625 :=
by
  sorry

end license_plate_calculation_l170_170248


namespace problem_l170_170227

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem problem (x : ℝ) (h₁ : x > 0) :
  g(x) > f 0 x + 2 :=
by
  sorry

end problem_l170_170227


namespace evaluate_expression_l170_170983

noncomputable def given_expression : ℝ :=
  (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) * Real.exp(3.5) + Real.log(Real.sin(0.785))

theorem evaluate_expression :
  given_expression ≈ 15563.91492641 :=
sorry

end evaluate_expression_l170_170983


namespace candidate_defeated_by_9074_votes_l170_170936

noncomputable def totalPolledVotes : ℕ := 90830
noncomputable def invalidVotes : ℕ := 83
noncomputable def validVotes : ℕ := totalPolledVotes - invalidVotes
noncomputable def percentVotesDefeatedCandidate : ℝ := 0.45
noncomputable def percentVotesWinningCandidate : ℝ := 0.55
noncomputable def votesDefeatedCandidate : ℕ := floor (percentVotesDefeatedCandidate * validVotes)
noncomputable def votesWinningCandidate : ℕ := floor (percentVotesWinningCandidate * validVotes)
noncomputable def defeatedByVotes : ℕ := votesWinningCandidate - votesDefeatedCandidate

theorem candidate_defeated_by_9074_votes :
  defeatedByVotes = 9074 :=
by
  calculate sorry

end candidate_defeated_by_9074_votes_l170_170936


namespace line_through_two_quadrants_l170_170283

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170283


namespace melanie_initial_dimes_l170_170721

theorem melanie_initial_dimes (d1 d2 t i : ℕ) (h1 : d1 = 39) (h2 : d2 = 25) (h3 : t = 83) :
  i = t - (d1 + d2) → i = 19 :=
by
  intros h
  rw [h1, h2, h3]
  simp at h
  assumption

end melanie_initial_dimes_l170_170721


namespace area_of_triangle_eq_l170_170516

-- Define the problem statement and conditions
def octagon_side_length : ℝ := 3
def cosine_135 : ℝ := - (1 / Real.sqrt 2)
def area_triangle_ADG : ℝ := (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2)) / (2 * Real.sqrt 2)

-- Use noncomputable to define the area based on the conditions
noncomputable def area_of_triangle : ℝ :=
  1 / (2 * Real.sqrt 2) * (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2))

-- Claim about the area of the triangle
theorem area_of_triangle_eq :
  area_of_triangle = area_triangle_ADG :=
sorry

end area_of_triangle_eq_l170_170516


namespace rectangle_width_to_length_ratio_l170_170667

theorem rectangle_width_to_length_ratio
  (w : ℕ) (h : ℕ) (P : ℕ) -- Declare the parameters
  (h_eq_8 : h = 8)  -- Condition 1: The length is 8
  (P_eq_24 : P = 24) -- Condition 2: The perimeter is 24
  (P_def : P = 2 * h + 2 * w) -- Condition 3: Definition of perimeter
  : w : h = 1 : 2 := -- Conclusion: the ratio of width to length is 1:2
by
  sorry -- Proof omitted

end rectangle_width_to_length_ratio_l170_170667


namespace sin_phi_zero_l170_170700

variables {a b c : ℝ × ℝ × ℝ}

def norm (v : ℝ × ℝ × ℝ) : ℝ := (v.1 * v.1 + v.2 * v.2 + v.3 * v.3).sqrt

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2 * w.3 - v.3 * w.2,
 v.3 * w.1 - v.1 * w.3,
 v.1 * w.2 - v.2 * w.1)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def angle_between (v w : ℝ × ℝ × ℝ) : ℝ :=
(dot_product v w / (norm v * norm w)).acos

theorem sin_phi_zero (ha : norm a = 1)
                     (hb : norm b = 7)
                     (hc : norm c = 4)
                     (h : cross_product a (cross_product a b) = c) :
  sin (angle_between a c) = 0 :=
by sorry

end sin_phi_zero_l170_170700


namespace cost_per_box_sugar_substitute_l170_170144

theorem cost_per_box_sugar_substitute 
  (packets_per_coffee : ℕ)
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (cost_for_90_days : ℕ)
  (days : ℕ)
  (total_packets_needed : ℕ := days * packets_per_coffee * coffees_per_day)
  (boxes_needed : ℕ := total_packets_needed / packets_per_box) :
  packets_per_coffee = 1 →
  coffees_per_day = 2 →
  packets_per_box = 30 →
  cost_for_90_days = 24 →
  days = 90 →
  cost_for_90_days / boxes_needed = 4 :=
begin
  intros,
  sorry,
end

end cost_per_box_sugar_substitute_l170_170144


namespace amanda_bought_30_candy_bars_l170_170534

noncomputable def candy_bars_bought (c1 c2 c3 c4 : ℕ) : ℕ :=
  let c5 := c4 * c2
  let c6 := c3 - c2
  let c7 := (c6 + c5) - c1
  c7

theorem amanda_bought_30_candy_bars :
  candy_bars_bought 7 3 22 4 = 30 :=
by
  sorry

end amanda_bought_30_candy_bars_l170_170534


namespace minimum_value_frac_l170_170706

theorem minimum_value_frac (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 2) :
  (p + q) / (p * q * r) ≥ 9 :=
sorry

end minimum_value_frac_l170_170706


namespace find_50th_term_arithmetic_sequence_l170_170554

theorem find_50th_term_arithmetic_sequence :
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  a₅₀ = 346 :=
by
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  show a₅₀ = 346
  sorry

end find_50th_term_arithmetic_sequence_l170_170554


namespace product_of_consecutive_integers_between_sqrt_50_l170_170824

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l170_170824


namespace product_of_integers_around_sqrt_50_l170_170777

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170777


namespace total_roasted_marshmallows_l170_170681

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l170_170681


namespace find_a_from_regression_l170_170994

variables {x_i y_i : ℝ} {a : ℝ}

-- Conditions from the problem
axiom Sum_x : x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 = 6
axiom Sum_y : y_1 + y_2 + y_3 + y_4 + y_5 + y_6 + y_7 + y_8 = 9

-- The regression equation
def regression_eq (x y : ℝ) : Prop := y = (1 / 6) * x + a

-- The statement to prove
theorem find_a_from_regression : a = 1 :=
by
  sorry

end find_a_from_regression_l170_170994


namespace greatest_sum_consecutive_integers_product_less_than_500_l170_170046

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l170_170046


namespace line_in_first_and_third_quadrants_l170_170263

theorem line_in_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) :
    (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x < 0) ↔ k > 0 :=
begin
  sorry
end

end line_in_first_and_third_quadrants_l170_170263


namespace least_number_divisor_l170_170047

theorem least_number_divisor (d : ℕ) (n m : ℕ) 
  (h1 : d = 1081)
  (h2 : m = 1077)
  (h3 : n = 4)
  (h4 : ∃ k, m + n = k * d) :
  d = 1081 :=
by
  sorry

end least_number_divisor_l170_170047


namespace total_meters_examined_l170_170933

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end total_meters_examined_l170_170933


namespace product_of_consecutive_integers_sqrt_50_l170_170821

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l170_170821


namespace g_difference_512_256_l170_170584

def σ (n : ℕ) : ℕ := ∑ d in (Finset.range (n + 1)).filter (λ d, n % d = 0), d

def f (n : ℕ) : ℚ := (σ n : ℚ) / n

def g (n : ℕ) : ℚ := f n + 1 / n

theorem g_difference_512_256 : g 512 - g 256 = 0 := by
  sorry

end g_difference_512_256_l170_170584


namespace find_x_plus_y_l170_170615

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2011 + Real.pi :=
sorry

end find_x_plus_y_l170_170615


namespace f_prime_one_third_zero_l170_170356

noncomputable def f (x : ℝ) : ℝ :=
if (∃ k : ℕ, ∀ n ≥ k, (decimal_digits x).nth n = some 9) then undefined
else ∑' n : ℕ, (decimal_digits x).nth (n+1) / 10^(2 * (n+1))

theorem f_prime_one_third_zero : deriv f (1 / 3) = 0 :=
sorry

end f_prime_one_third_zero_l170_170356


namespace x_squared_plus_y_squared_value_l170_170595

theorem x_squared_plus_y_squared_value (x y : ℝ) (h : (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6) : x^2 + y^2 = 1 :=
by
  sorry

end x_squared_plus_y_squared_value_l170_170595


namespace sqrt_50_product_consecutive_integers_l170_170805

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l170_170805


namespace triangle_perimeter_l170_170300

open Real EuclideanGeometry

-- Given a right triangle ABC with ∠C = 90° and AB = 13,
-- Squares ABXY and CBWZ are constructed outside the triangle,
-- and points X, Y, Z, W lie on a circle,
-- prove that the perimeter of ΔABC is 30.
theorem triangle_perimeter
  (A B C X Y Z W : Point)
  (hC90 : angle_tri C B C = 90)
  (hAB : distance A B = 13)
  (hSquares : square_outside_triangle ABXY CBWZ)
  (hCyclic : cyclic_quad W X Y Z) :
  perimeter_triangle A B C = 30 :=
sorry

end triangle_perimeter_l170_170300


namespace matrix_multiplication_correct_l170_170605

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 0], ![0, 3]]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 1], ![-1, 0]]

theorem matrix_multiplication_correct :
    A ⬝ B = ![![4, 2], ![-3, 0]] :=
by
  sorry

end matrix_multiplication_correct_l170_170605


namespace cartons_in_load_l170_170923

theorem cartons_in_load 
  (crate_weight : ℕ)
  (carton_weight : ℕ)
  (num_crates : ℕ)
  (total_load_weight : ℕ)
  (h1 : crate_weight = 4)
  (h2 : carton_weight = 3)
  (h3 : num_crates = 12)
  (h4 : total_load_weight = 96) :
  ∃ C : ℕ, num_crates * crate_weight + C * carton_weight = total_load_weight ∧ C = 16 := 
by 
  sorry

end cartons_in_load_l170_170923


namespace line_through_two_quadrants_l170_170286

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end line_through_two_quadrants_l170_170286


namespace domain_implies_range_a_range_implies_range_a_l170_170228

theorem domain_implies_range_a {a : ℝ} :
  (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 :=
sorry

theorem range_implies_range_a {a : ℝ} :
  (∀ y : ℝ, ∃ x : ℝ, ax^2 + 2 * a * x + 1 = y) → 1 ≤ a :=
sorry

end domain_implies_range_a_range_implies_range_a_l170_170228


namespace odds_against_C_win_l170_170310

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C_win (pA pB : ℚ) (hA : pA = 1/5) (hB : pB = 2/3) :
  odds_against_winning (1 - pA - pB) = 13 / 2 :=
by
  sorry

end odds_against_C_win_l170_170310


namespace inscribed_square_ratio_l170_170962

noncomputable def area_of_square (side : ℝ) : ℝ := side^2

theorem inscribed_square_ratio : 
  ∀ (side_length : ℝ), 
   side_length > 0 → 
   let large_square_area := area_of_square side_length in
   let small_square_side_length := (1 / 2) * side_length in
   let inscribed_square_area := area_of_square small_square_side_length in
   (inscribed_square_area / large_square_area) = 1 / 4 :=
by
  intros side_length h
  let large_square_area := area_of_square side_length
  let small_square_side_length := (1 / 2) * side_length
  let inscribed_square_area := area_of_square small_square_side_length
  sorry

end inscribed_square_ratio_l170_170962


namespace coefficient_of_a_half_l170_170315

theorem coefficient_of_a_half (a : ℂ) (h : a ≠ 0) : 
  coefficient (λ k : ℕ, (choose 9 k) * (2 : ℂ)^k * (a : ℂ)^(9 - k - k / 2)) (1 / 2) = 4032 :=
  sorry

end coefficient_of_a_half_l170_170315


namespace product_of_integers_around_sqrt_50_l170_170773

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170773


namespace part_a_part_b_l170_170194

-- Define the function f(k) assuming it returns the count of numbers with exactly 3 ones in their binary representation
def f (k : ℕ) : ℕ := sorry  -- Placeholder function, actual definition is skipped

-- (a) For any positive integer m, there exists a positive integer k such that f(k) = m
theorem part_a (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, 0 < k ∧ f(k) = m := sorry

-- (b) Define all positive integers m for which there exists exactly one positive integer k such that f(k) = m
theorem part_b (m : ℕ) : ∃! k : ℕ, 0 < k ∧ f(k) = m ↔ ∃ n : ℕ, 2 ≤ n ∧ m = n*(n-1)/2 + 1 := sorry

end part_a_part_b_l170_170194


namespace quadrilateral_has_four_sides_and_angles_l170_170117

-- Define the conditions based on the characteristics of a quadrilateral
def quadrilateral (sides angles : Nat) : Prop :=
  sides = 4 ∧ angles = 4

-- Statement: Verify the property of a quadrilateral
theorem quadrilateral_has_four_sides_and_angles (sides angles : Nat) (h : quadrilateral sides angles) : sides = 4 ∧ angles = 4 :=
by
  -- We provide a proof by the characteristics of a quadrilateral
  sorry

end quadrilateral_has_four_sides_and_angles_l170_170117


namespace k_positive_if_line_passes_through_first_and_third_quadrants_l170_170279

def passes_through_first_and_third_quadrants (k : ℝ) (h : k ≠ 0) : Prop :=
  ∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)

theorem k_positive_if_line_passes_through_first_and_third_quadrants :
  ∀ k : ℝ, k ≠ 0 → passes_through_first_and_third_quadrants k -> k > 0 :=
by
  intros k h₁ h₂
  sorry

end k_positive_if_line_passes_through_first_and_third_quadrants_l170_170279


namespace decimal_to_vulgar_fraction_l170_170154

theorem decimal_to_vulgar_fraction (h : (34 / 100 : ℚ) = 0.34) : (0.34 : ℚ) = 17 / 50 := by
  sorry

end decimal_to_vulgar_fraction_l170_170154


namespace estimated_population_correlation_coefficient_better_sampling_method_l170_170531

variables 
  (x y : Fin 20 → ℝ)
  (plots : ℕ := 200)
  (sample_size : ℕ := 20)
  (sum_x : ℝ := ∑ i, x i)
  (sum_y : ℝ := ∑ i, y i)
  (sum_xx : ℝ := ∑ i, (x i - sum_x / sample_size) ^ 2)
  (sum_yy : ℝ := ∑ i, (y i - sum_y / sample_size) ^ 2)
  (sum_xy : ℝ := ∑ i, (x i - sum_x / sample_size) * (y i - sum_y / sample_size))
  (sqrt2_approx : ℝ := 1.414)

-- Given conditions
def conditions : Prop := 
  sum_x = 60 ∧ sum_y = 1200 ∧ sum_xx = 80 ∧ sum_yy = 9000 ∧ sum_xy = 800

-- Proving estimated population
theorem estimated_population (h : conditions) : 
  (sum_y / sample_size * plots) = 12000 := 
by {
  sorry,
}

-- Proving correlation coefficient
theorem correlation_coefficient (h : conditions) : 
  (sum_xy / (Real.sqrt (sum_xx * sum_yy))) ≈ 0.94 := 
by {
  sorry,
}

-- Suggesting stratified sampling method
theorem better_sampling_method : 
  StratifiedSamplingRecommendation := 
by {
  sorry,
}

end estimated_population_correlation_coefficient_better_sampling_method_l170_170531


namespace count_divisors_32_458_l170_170636

theorem count_divisors_32_458 :
  (Set.toFinset {d ∣ 32458 | d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}}).card = 2 :=
by
  sorry

end count_divisors_32_458_l170_170636


namespace annie_milkshakes_l170_170539

theorem annie_milkshakes
  (A : ℕ) (C_hamburger : ℕ) (C_milkshake : ℕ) (H : ℕ) (L : ℕ)
  (initial_money : A = 120)
  (hamburger_cost : C_hamburger = 4)
  (milkshake_cost : C_milkshake = 3)
  (hamburgers_bought : H = 8)
  (money_left : L = 70) :
  ∃ (M : ℕ), A - H * C_hamburger - M * C_milkshake = L ∧ M = 6 :=
by
  sorry

end annie_milkshakes_l170_170539


namespace sum_sides_half_perimeter_l170_170862

open_locale classical

noncomputable theory
def regular_n_gon (n : ℕ) := {P : fin n → ℝ × ℝ | ∀ i j, P i ≠ P j}

def alternating_vertices 
  (n : ℕ) 
  (P Q : regular_n_gon n) 
  : { C : fin (2 * n) → ℝ × ℝ // ∀ i, C (2 * i) = P.1 i ∧ C (2 * i + 1) = Q.1 i } := 
  sorry

theorem sum_sides_half_perimeter
  (n : ℕ) 
  (P Q : regular_n_gon n) 
  (h_intersect : ∃ C : fin (2 * n) → ℝ × ℝ, 
    ∀ i, C (2 * i) = P.1 i ∧ C (2 * i + 1) = Q.1 i)
  (p_i q_i : fin n → ℝ)
  (h_p : ∀ i, p_i i = dist (P.1 i) (Q.1 i))
  (h_q : ∀ i, q_i i = dist (Q.1 i) (P.1 ((i + 1) % n))) 
:
  (finset.univ.sum p_i) = (finset.univ.sum q_i) :=
sorry

end sum_sides_half_perimeter_l170_170862


namespace two_times_card_A_gte_card_B_l170_170146

def M (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define the sets A and B according to the provided conditions.
def A (n : ℕ) (color : ℕ → ℕ) : set (ℕ × ℕ × ℕ) :=
    {x | ∃ (x y z ∈ M n), x + y + z ≡ 0 [MOD n] ∧ color x = color y ∧ color y = color z}

def B (n : ℕ) (color : ℕ → ℕ) : set (ℕ × ℕ × ℕ) :=
    {x | ∃ (x y z ∈ M n), x + y + z ≡ 0 [MOD n] ∧ color x ≠ color y ∧ color y ≠ color z ∧ color z ≠ color x}

-- Main theorem stating 2|A| ≥ |B| given the conditions
theorem two_times_card_A_gte_card_B {n : ℕ} (color : ℕ → ℕ) :
  2 * (A n color).card ≥ (B n color).card := 
sorry  -- Proof omitted.

end two_times_card_A_gte_card_B_l170_170146


namespace function_increment_l170_170956

theorem function_increment (x₁ x₂ : ℝ) (f : ℝ → ℝ) (h₁ : x₁ = 2) 
                           (h₂ : x₂ = 2.5) (h₃ : ∀ x, f x = x ^ 2) :
  f x₂ - f x₁ = 2.25 :=
by
  sorry

end function_increment_l170_170956


namespace find_ns_mul_l170_170710

noncomputable def S : set ℝ :=
  {x | x ≠ 0}

noncomputable def f : S → S :=
  sorry

axiom f_property1 (x : S) : f ⟨1 / x.1, by { simp [S] }⟩ = x.1^2 * f x

axiom f_property2 (x y : S) (h : x.1 + y.1 ≠ 0) : f ⟨1 / x.1, by { simp [S] }⟩ + f ⟨1 / y.1, by { simp [S] }⟩ = 1 + f ⟨1 / (x.1 + y.1), by { simp [S, h] }⟩

theorem find_ns_mul : 
  let f1 := f ⟨1, by { simp [S] }⟩ in
  (∃! v, f1 = v) ∧ (∑' x : {v // f1 = v}, x.1) = 1 ∧ 1 * 1 = 1 :=
  sorry

end find_ns_mul_l170_170710


namespace speed_ratio_l170_170366

theorem speed_ratio :
  ∀ (v_A v_B : ℝ), (v_A / v_B = 3 / 2) ↔ (v_A = 3 * v_B / 2) :=
by
  intros
  sorry

end speed_ratio_l170_170366


namespace principal_is_2000_l170_170981

-- Define the conditions
def SI (P R T : ℕ) : ℕ := (P * R * T) / 100
def CI (P R T : ℕ) : ℕ := (P * (1 + R / 100)^T) - P

-- Noncomputable because it involves real number operations
noncomputable def find_principal : ℕ :=
  let R := 10 in
  let T := 2 in
  let CI := CI 1 10 2
  let SI := SI 1 10 2
  let diff := CI - SI
  let P := 20 / diff
  P * 100

theorem principal_is_2000 : find_principal = 2000 := by
  sorry

end principal_is_2000_l170_170981


namespace total_accidents_proof_l170_170340

-- Conditions for accidents per highway
def HighwayA_accidents_per_hundred_million := 200
def HighwayB_accidents_per_fifty_million := 150
def HighwayC_accidents_per_hundred_fifty_million := 100

-- Conditions for number of vehicles traveled on each highway
def HighwayA_vehicles_billion := 2
def HighwayB_vehicles_billion := 1.5
def HighwayC_vehicles_billion := 2.5

-- Calculations per billion vehicles (intermediary steps)
def HighwayA_accidents_per_billion := HighwayA_accidents_per_hundred_million * 10
def HighwayB_accidents_per_billion := HighwayB_accidents_per_fifty_million * 20
def HighwayC_accidents_per_billion := (HighwayC_accidents_per_hundred_fifty_million * (100 / 150)).to_nat  -- Rounded whole number

-- Total accidents calculation
def total_accidents := (HighwayA_vehicles_billion * HighwayA_accidents_per_billion) + 
                       (HighwayB_vehicles_billion * HighwayB_accidents_per_billion) + 
                       (HighwayC_vehicles_billion * HighwayC_accidents_per_billion)

-- Proof statement
theorem total_accidents_proof : total_accidents = 10168 :=
by
  have hA : HighwayA_accidents_per_billion = 2000 := rfl
  have hB : HighwayB_accidents_per_billion = 3000 := rfl
  have hC : HighwayC_accidents_per_billion = 667 := rfl
  calc
    total_accidents = (HighwayA_vehicles_billion * 2000) + (HighwayB_vehicles_billion * 3000) + (HighwayC_vehicles_billion * 667) : by congr; rw [hA, hB, hC]
    ... = 2 * 2000 + 1.5 * 3000 + 2.5 * 667 : by rfl
    ... = 4000 + 4500 + 1667.5 : by norm_num
    ... = 10167.5
    ... = 10168 : by norm_num

-- Including sorry to skip the detailed proof steps
sorry

end total_accidents_proof_l170_170340


namespace find_principal_sum_l170_170078

noncomputable def principal_sum (P R : ℝ) : ℝ := P * (R + 6) / 100 - P * R / 100

theorem find_principal_sum (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) : P = 500 :=
by sorry

end find_principal_sum_l170_170078


namespace min_sum_of_product_1806_l170_170761

theorem min_sum_of_product_1806 :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1806 ∧ a + b + c = 153 :=
begin
  sorry
end

end min_sum_of_product_1806_l170_170761


namespace sum_of_roots_l170_170056

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l170_170056


namespace math_problem_l170_170352

noncomputable def f : ℝ → ℝ := sorry

def a_n (n : ℕ) [Fact (0 < n)] : ℝ := f (2^n) / n
def b_n (n : ℕ) [Fact (0 < n)] : ℝ := f (2^n) / 2^n

theorem math_problem (h1 : ∃ x : ℝ, f x ≠ 0)
  (h2 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (h3 : f 2 = 2) :
  f 0 = f 1 ∧
  (∀ n : ℕ, 0 < n → a_n n = 2^n) ∧
  (∀ n : ℕ, 0 < n → b_n n = n) :=
by 
  sorry

end math_problem_l170_170352


namespace particle_not_illuminated_by_ray_l170_170502

-- Define the problem conditions step by step
def x (t : ℝ) : ℝ := 3
def y (t : ℝ) : ℝ := 3 + sin t * cos t - sin t - cos t

-- The statement of the theorem
theorem particle_not_illuminated_by_ray (c : ℝ) (h1 : 0 < c) :
    ¬∃ t : ℝ, y t = 3 * c ↔ c ∈ (Ioo 0 (1/2)) ∪ Ioi (7/6) :=
by
  sorry

end particle_not_illuminated_by_ray_l170_170502


namespace partners_profit_share_l170_170363

theorem partners_profit_share 
  (total_profit : ℝ) (A_ratio B_ratio C_ratio : ℝ)
  (total_profit_eq : total_profit = 22400)
  (ratios : A_ratio = 2 ∧ B_ratio = 3 ∧ C_ratio = 5) :
  let total_parts := A_ratio + B_ratio + C_ratio in
  let A_share := (A_ratio / total_parts) * total_profit in
  let B_share := (B_ratio / total_parts) * total_profit in
  let C_share := (C_ratio / total_parts) * total_profit in
  A_share = 4480 ∧ B_share = 6720 ∧ C_share = 11200 :=
by {
  sorry
}

end partners_profit_share_l170_170363


namespace quadratic_completion_l170_170726

theorem quadratic_completion (x d e f : ℤ) (h1 : 100*x^2 + 80*x - 144 = 0) (hd : d > 0) 
  (hde : (d * x + e)^2 = f) : d + e + f = 174 :=
sorry

end quadratic_completion_l170_170726


namespace circle_tangent_line_eq_l170_170579

theorem circle_tangent_line_eq (m : ℝ) : 
  ∃ (r : ℝ), (x - 2)^2 + (y + 3)^2 = r^2 ∧ 
             (∃ (p : ℝ) (x y : ℝ), r = sqrt ((2 - x)^2 + (-3 - y)^2) ∧ (2*m*x - y - 2*m - 1 = 0)) :=
sorry

end circle_tangent_line_eq_l170_170579


namespace titu_andreescu_dospinescu_problem_l170_170888

-- Define the sum of digits function
def s (m : ℕ) : ℕ := sorry

-- Define the minimal k function
def f (n : ℕ) : ℕ := sorry

theorem titu_andreescu_dospinescu_problem :
  ∃ (C1 C2 : ℝ), (0 < C1 ∧ C1 = 9/2 ∧ C2 = 45) ∧ 
    ∀ (n : ℕ), (2 ≤ n) → 
      (C1 * Real.log10 n ≤ ↑ (f n) ∧ ↑ (f n) ≤ C2 * Real.log10 n) := 
by
  sorry

end titu_andreescu_dospinescu_problem_l170_170888


namespace part1_part2_l170_170617

open Real

-- Definitions for part (1)
def alpha1 : ℝ := 60 * (π / 180)  -- Convert degrees to radians
def R1 : ℝ := 10
def l1 : ℝ := (alpha1 * R1)      -- Arc length when α = 60° and R = 10 cm
def S1 : ℝ := (1 / 2) * l1 * R1  -- Area of the sector

-- Definitions for part (2)
def perimeter : ℝ := 12
def l2 (r : ℝ) : ℝ := perimeter - 2 * r -- Arc length in terms of radius
def S2 (r : ℝ) : ℝ := (1 / 2) * l2(r) * r  -- Area of sector in terms of radius

theorem part1 :
  (alpha1 = π / 3) ∧
  (R1 = 10) ∧
  (l1 = 10 * π / 3) ∧
  (S1 = 50 * π / 3) :=
by
  -- Proof here
  sorry

theorem part2 :
  (∃ r : ℝ, (l2(r) + 2 * r = perimeter) ∧ (S2(r) = - (r - 3)^2 + 9) ∧ (r = 3) ∧ (S2(r) = 9) ∧ (l2(r) = 6) ∧ (l2(r) / r = 2)) :=
by
  -- Proof here
  sorry

end part1_part2_l170_170617


namespace triangle_perimeter_l170_170753

open Real

noncomputable def quadratic_roots (a b c : ℝ) : set ℝ := 
  {r : ℝ | a * r^2 + b * r + c = 0}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem triangle_perimeter :
  ∃ (a b c : ℝ), {a, b, c} = quadratic_roots 1 (-9) 18 ∧
                   is_triangle a b c ∧
                   is_isosceles a b c ∧
                   (a + b + c) = 15 :=
sorry

end triangle_perimeter_l170_170753


namespace max_sum_of_extreme_scores_l170_170486

theorem max_sum_of_extreme_scores (m n : ℕ) (h : 2 ≤ m ∧ 2 ≤ n) (scores : Fin n → Fin m → ℕ) :
  (∃ (p : Fin n → ℕ), (∀ i j : Fin n, i ≤ j → p i ≥ p j) ∧ (∀ i : Fin n, 
  p i = ∑ j, if scores i j < scores j i then m - 1 else 0)
  ∧ ∃ z w : Fin n, p z + p w = m * (n - 1)) :=
begin
  sorry
end

end max_sum_of_extreme_scores_l170_170486


namespace complement_of_B_in_A_l170_170714

open Set

/-- Definitions used in the problem -/
def A := {0, 2, 4, 6, 8, 10} : Set ℕ
def B := {4, 8} : Set ℕ

/-- The proof to show that the complement of B in A is {0, 2, 6, 10} -/
theorem complement_of_B_in_A :
  compl (B : Set ℕ) ∩ (A : Set ℕ) = ({0, 2, 6, 10} : Set ℕ) :=
by
  sorry

end complement_of_B_in_A_l170_170714


namespace train_pass_time_l170_170927

-- Definitions based on conditions
def train_length : Float := 250
def pole_time : Float := 10
def platform_length : Float := 1250
def incline_angle : Float := 5 -- degrees
def speed_reduction_factor : Float := 0.75

-- The statement to be proved
theorem train_pass_time :
  let original_speed := train_length / pole_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  let time_to_pass_platform := total_distance / incline_speed
  time_to_pass_platform = 80 := by
  simp [train_length, pole_time, platform_length, incline_angle, speed_reduction_factor]
  sorry

end train_pass_time_l170_170927


namespace geometric_sequence_sum_l170_170711

noncomputable def f (x : ℝ) (m : ℝ) := log x / log m

variables {a : ℕ → ℝ} {m : ℝ}

theorem geometric_sequence_sum
  (hm : 0 < m)
  (hm1 : m ≠ 1)
  (h_ratio : ∀ n : ℕ, a (n+1) = m * a n)
  (h_product : f (a 2 * a 4 * a 6 * ... * a 2018) m = 7) :
  (finset.sum (finset.range 2018) (λ n, f ((a n) ^ 2) m)) = -1990 :=
sorry

end geometric_sequence_sum_l170_170711


namespace max_value_a_l170_170992

theorem max_value_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) ↔ a ≤ 6 := by
  sorry

end max_value_a_l170_170992


namespace question1_question2_l170_170628

def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem question1 :
  (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 :=
sorry

theorem question2 :
  { x : ℝ | x^2 - 8*x + 15 + f x ≤ 0 } = { x | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 } :=
sorry

end question1_question2_l170_170628


namespace exists_diff_row_col_product_l170_170894

theorem exists_diff_row_col_product :
  ∀ (A : matrix (fin 9) (fin 9) ℤ),
  (∀ i j, 1 ≤ A i j ∧ A i j ≤ 81) →
  ∃ k : fin 9, row_product A k ≠ col_product A k :=
by
  sorry

def row_product (A : matrix (fin 9) (fin 9) ℤ) (i : fin 9) : ℤ :=
  finset.prod (finset.univ) (λ j, A i j)

def col_product (A : matrix (fin 9) (fin 9) ℤ) (j : fin 9) : ℤ :=
  finset.prod (finset.univ) (λ i, A i j)

end exists_diff_row_col_product_l170_170894


namespace range_f_l170_170635

-- Definitions given in the problem
def vector := ℝ × ℝ

def vec_op (a b : vector) : vector := (a.1 * b.1, a.2 * b.2)

def m : vector := (2, 1/2)
def n : vector := (Real.pi / 3, 0)
def P (x : ℝ) : vector := (x, Real.sin x)

def Q (x : ℝ) : vector := vec_op m (P x) + n
def f (x : ℝ) : ℝ := (Q x).snd

-- The proof statement is skipped with sorry
theorem range_f : set.Icc (-1 / 2) (1 / 2) = {y | ∃ x, f x = y} :=
by
  sorry

end range_f_l170_170635


namespace triangle_area_eq_l170_170887

theorem triangle_area_eq :
  ∀ (A B C D D' E F : ℝ) (h1 : ∀ (a b : ℝ), (a = b ↔ a^2 = b^2))
    (h2 : A = 25) (h3 : E = 10) (h4 : D - E = 15) (h5 : F = 5) 
  , 10 * 10 * 5 = 2 * 25 ^ 2 ->
  ∀ (x y : ℝ), x * y / 2 = 25 * sqrt 2 
  := by sorry

end triangle_area_eq_l170_170887


namespace product_of_integers_around_sqrt_50_l170_170775

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l170_170775


namespace complex_fraction_calculation_l170_170939

theorem complex_fraction_calculation :
  (1 - complex.i) * (1 + 2 * complex.i) / (1 + complex.i) = 4 - 2 * complex.i :=
by
  sorry

end complex_fraction_calculation_l170_170939
