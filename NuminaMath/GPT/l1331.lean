import Mathlib

namespace max_quotient_l1331_133173

theorem max_quotient (a b : ℕ) (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 1200 ≤ b) (h₄ : b ≤ 2400) :
  b / a ≤ 24 :=
sorry

end max_quotient_l1331_133173


namespace hash_hash_hash_45_l1331_133153

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_45 : hash (hash (hash 45)) = 7.56 :=
by
  sorry

end hash_hash_hash_45_l1331_133153


namespace smallest_positive_value_l1331_133141

theorem smallest_positive_value (a b c d e : ℝ) (h1 : a = 8 - 2 * Real.sqrt 14) 
  (h2 : b = 2 * Real.sqrt 14 - 8) 
  (h3 : c = 20 - 6 * Real.sqrt 10) 
  (h4 : d = 64 - 16 * Real.sqrt 4) 
  (h5 : e = 16 * Real.sqrt 4 - 64) :
  a = 8 - 2 * Real.sqrt 14 ∧ 0 < a ∧ a < c ∧ a < d :=
by
  sorry

end smallest_positive_value_l1331_133141


namespace sum_of_vertices_l1331_133100

theorem sum_of_vertices (pentagon_vertices : Nat := 5) (hexagon_vertices : Nat := 6) :
  (2 * pentagon_vertices) + (2 * hexagon_vertices) = 22 :=
by
  sorry

end sum_of_vertices_l1331_133100


namespace q_is_false_l1331_133160

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q) = false) (h2 : ¬p = false) : q = false :=
by
  sorry

end q_is_false_l1331_133160


namespace regular_pentagonal_prism_diagonal_count_l1331_133195

noncomputable def diagonal_count (n : ℕ) : ℕ := 
  if n = 5 then 10 else 0

theorem regular_pentagonal_prism_diagonal_count :
  diagonal_count 5 = 10 := 
  by
    sorry

end regular_pentagonal_prism_diagonal_count_l1331_133195


namespace no_integer_solution_for_euler_conjecture_l1331_133152

theorem no_integer_solution_for_euler_conjecture :
  ¬(∃ n : ℕ, 5^4 + 12^4 + 9^4 + 8^4 = n^4) :=
by
  -- Sum of the given fourth powers
  have lhs : ℕ := 5^4 + 12^4 + 9^4 + 8^4
  -- Direct proof skipped with sorry
  sorry

end no_integer_solution_for_euler_conjecture_l1331_133152


namespace smallest_digit_for_divisibility_by_3_l1331_133176

theorem smallest_digit_for_divisibility_by_3 : ∃ x : ℕ, x < 10 ∧ (5 + 2 + 6 + x + 1 + 8) % 3 = 0 ∧ ∀ y : ℕ, y < 10 ∧ (5 + 2 + 6 + y + 1 + 8) % 3 = 0 → x ≤ y := by
  sorry

end smallest_digit_for_divisibility_by_3_l1331_133176


namespace N_is_necessary_but_not_sufficient_l1331_133179

-- Define sets M and N
def M := { x : ℝ | 0 < x ∧ x < 1 }
def N := { x : ℝ | -2 < x ∧ x < 1 }

-- State the theorem to prove that "a belongs to N" is necessary but not sufficient for "a belongs to M"
theorem N_is_necessary_but_not_sufficient (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M → False) :=
by sorry

end N_is_necessary_but_not_sufficient_l1331_133179


namespace ratio_smaller_to_larger_dimension_of_framed_painting_l1331_133101

-- Definitions
def painting_width : ℕ := 16
def painting_height : ℕ := 20
def side_frame_width (x : ℝ) : ℝ := x
def top_frame_width (x : ℝ) : ℝ := 1.5 * x
def total_frame_area (x : ℝ) : ℝ := (painting_width + 2 * side_frame_width x) * (painting_height + 2 * top_frame_width x) - painting_width * painting_height
def frame_area_eq_painting_area (x : ℝ) : Prop := total_frame_area x = painting_width * painting_height

-- Lean statement
theorem ratio_smaller_to_larger_dimension_of_framed_painting :
  ∃ x : ℝ, frame_area_eq_painting_area x → 
  ((painting_width + 2 * side_frame_width x) / (painting_height + 2 * top_frame_width x)) = (3 / 4) :=
by
  sorry

end ratio_smaller_to_larger_dimension_of_framed_painting_l1331_133101


namespace find_angle_between_vectors_l1331_133134

noncomputable def angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : ℝ :=
  60

theorem find_angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : angle_between_vectors a b a_nonzero b_nonzero perp1 perp2 = 60 :=
  by 
  sorry

end find_angle_between_vectors_l1331_133134


namespace inequality_proof_l1331_133159

theorem inequality_proof
  (a b c d : ℝ) (h0 : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0) (h4 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1 / 3 :=
sorry

end inequality_proof_l1331_133159


namespace min_sum_of_diagonals_l1331_133199

theorem min_sum_of_diagonals (x y : ℝ) (α : ℝ) (hx : 0 < x) (hy : 0 < y) (hα : 0 < α ∧ α < π) (h_area : x * y * Real.sin α = 2) : x + y ≥ 2 * Real.sqrt 2 :=
sorry

end min_sum_of_diagonals_l1331_133199


namespace count_scalene_triangles_under_16_l1331_133188

theorem count_scalene_triangles_under_16 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (a b c : ℕ), 
  a < b ∧ b < c ∧ a + b + c < 16 ∧ a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a, b, c) ∈ [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 5), (3, 5, 6), (4, 5, 6)] :=
by sorry

end count_scalene_triangles_under_16_l1331_133188


namespace range_of_a_l1331_133107

noncomputable def f (x a : ℝ) := x^2 - a * x
noncomputable def g (x : ℝ) := Real.exp x
noncomputable def h (x : ℝ) := x - (Real.log x / x)

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (f x a = Real.log x)) ↔ (1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1) :=
by
  sorry

end range_of_a_l1331_133107


namespace remainder_101_mul_103_mod_11_l1331_133132

theorem remainder_101_mul_103_mod_11 : (101 * 103) % 11 = 8 :=
by
  sorry

end remainder_101_mul_103_mod_11_l1331_133132


namespace parametric_line_l1331_133104

theorem parametric_line (s m : ℤ) :
  (∀ t : ℤ, ∃ x y : ℤ, 
    y = 5 * x - 7 ∧
    x = s + 6 * t ∧ y = 3 + m * t ) → 
  (s = 2 ∧ m = 30) :=
by
  sorry

end parametric_line_l1331_133104


namespace cos_B_in_triangle_l1331_133166

theorem cos_B_in_triangle
  (A B C a b c : ℝ)
  (h1 : Real.sin A = 2 * Real.sin C)
  (h2 : b^2 = a * c)
  (h3 : 0 < b)
  (h4 : 0 < c)
  (h5 : a = 2 * c)
  : Real.cos B = 3 / 4 := 
sorry

end cos_B_in_triangle_l1331_133166


namespace length_of_each_lateral_edge_l1331_133120

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end length_of_each_lateral_edge_l1331_133120


namespace determine_n_l1331_133187

variable (x a n : ℕ)

def binomial_term (n k : ℕ) (x a : ℤ) : ℤ :=
  Nat.choose n k * x ^ (n - k) * a ^ k

theorem determine_n (hx : 0 < x) (ha : 0 < a)
  (h4 : binomial_term n 3 x a = 330)
  (h5 : binomial_term n 4 x a = 792)
  (h6 : binomial_term n 5 x a = 1716) :
  n = 7 :=
sorry

end determine_n_l1331_133187


namespace pencils_added_by_mike_l1331_133171

-- Definitions and assumptions based on conditions
def initial_pencils : ℕ := 41
def final_pencils : ℕ := 71

-- Statement of the problem
theorem pencils_added_by_mike : final_pencils - initial_pencils = 30 := 
by 
  sorry

end pencils_added_by_mike_l1331_133171


namespace bacteria_population_at_15_l1331_133109

noncomputable def bacteria_population (t : ℕ) : ℕ := 
  20 * 2 ^ (t / 3)

theorem bacteria_population_at_15 : bacteria_population 15 = 640 := by
  sorry

end bacteria_population_at_15_l1331_133109


namespace sufficient_but_not_necessary_l1331_133110

noncomputable def condition_to_bool (a b : ℝ) : Bool :=
a > b ∧ b > 0

theorem sufficient_but_not_necessary (a b : ℝ) (h : condition_to_bool a b) :
  (a > b ∧ b > 0) → (a^2 > b^2) ∧ (∃ a' b' : ℝ, a'^2 > b'^2 ∧ ¬ (a' > b' ∧ b' > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l1331_133110


namespace zero_exponent_rule_proof_l1331_133121

-- Defining the condition for 818 being non-zero
def eight_hundred_eighteen_nonzero : Prop := 818 ≠ 0

-- Theorem statement
theorem zero_exponent_rule_proof (h : eight_hundred_eighteen_nonzero) : 818 ^ 0 = 1 := by
  sorry

end zero_exponent_rule_proof_l1331_133121


namespace factorize_expression_l1331_133103

theorem factorize_expression (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 :=
by
  sorry

end factorize_expression_l1331_133103


namespace correct_operation_l1331_133113

theorem correct_operation (x : ℝ) : (x^2) * (x^4) = x^6 :=
  sorry

end correct_operation_l1331_133113


namespace find_n_from_equation_l1331_133145

theorem find_n_from_equation : ∃ n : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 22 ∧ n = 4 :=
by
  sorry

end find_n_from_equation_l1331_133145


namespace ab_plus_cd_is_composite_l1331_133155

theorem ab_plus_cd_is_composite 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_eq : a^2 + a * c - c^2 = b^2 + b * d - d^2) : 
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ ab + cd = p * q :=
by
  sorry

end ab_plus_cd_is_composite_l1331_133155


namespace find_r_minus_p_l1331_133197

-- Define the variables and conditions
variables (p q r A1 A2 : ℝ)
noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2

-- Given conditions in the problem
axiom hA1 : arithmetic_mean p q = 10
axiom hA2 : arithmetic_mean q r = 25

-- Statement to prove
theorem find_r_minus_p : r - p = 30 :=
by {
  -- write the necessary proof steps here
  sorry
}

end find_r_minus_p_l1331_133197


namespace curling_teams_l1331_133108

-- Define the problem conditions and state the theorem
theorem curling_teams (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
sorry

end curling_teams_l1331_133108


namespace rationalize_denominator_l1331_133127

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = (Real.sqrt 3) / 5 :=
by
  sorry

end rationalize_denominator_l1331_133127


namespace students_in_each_class_l1331_133150

theorem students_in_each_class (S : ℕ) 
  (h1 : 10 * S * 5 = 1750) : 
  S = 35 := 
by 
  sorry

end students_in_each_class_l1331_133150


namespace worker_C_work_rate_worker_C_days_l1331_133124

theorem worker_C_work_rate (A B C: ℚ) (hA: A = 1/10) (hB: B = 1/15) (hABC: A + B + C = 1/4) : C = 1/12 := 
by
  sorry

theorem worker_C_days (C: ℚ) (hC: C = 1/12) : 1 / C = 12 :=
by
  sorry

end worker_C_work_rate_worker_C_days_l1331_133124


namespace smallest_c_in_range_l1331_133136

-- Define the quadratic function g(x)
def g (x c : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + c

-- Define the condition for c
def in_range_5 (c : ℝ) : Prop :=
  ∃ x : ℝ, g x c = 5

-- The theorem stating that the smallest value of c for which 5 is in the range of g is 7
theorem smallest_c_in_range : ∃ c : ℝ, c = 7 ∧ ∀ c' : ℝ, (in_range_5 c' → 7 ≤ c') :=
sorry

end smallest_c_in_range_l1331_133136


namespace probability_of_receiving_1_l1331_133102

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end probability_of_receiving_1_l1331_133102


namespace point_in_second_quadrant_l1331_133146

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l1331_133146


namespace quadratic_equation_formulation_l1331_133168

theorem quadratic_equation_formulation (a b c : ℝ) (x₁ x₂ : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = -b / a)
  (h₅ : x₁ * x₂ = c / a) :
  ∃ (y : ℝ), a^2 * y^2 + a * (b - c) * y - b * c = 0 :=
by
  sorry

end quadratic_equation_formulation_l1331_133168


namespace race_distance_l1331_133164

theorem race_distance (a b c d : ℝ) 
  (h₁ : d / a = (d - 25) / b)
  (h₂ : d / b = (d - 15) / c)
  (h₃ : d / a = (d - 37) / c) : 
  d = 125 :=
by
  sorry

end race_distance_l1331_133164


namespace perimeter_of_garden_l1331_133190

-- Definitions based on conditions
def length : ℕ := 150
def breadth : ℕ := 150
def is_square (l b : ℕ) := l = b

-- Theorem statement proving the perimeter given conditions
theorem perimeter_of_garden : is_square length breadth → 4 * length = 600 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end perimeter_of_garden_l1331_133190


namespace coins_problem_l1331_133115

theorem coins_problem (x y : ℕ) (h1 : x + y = 20) (h2 : x + 5 * y = 80) : x = 5 :=
by
  sorry

end coins_problem_l1331_133115


namespace bob_buys_nose_sprays_l1331_133193

theorem bob_buys_nose_sprays (cost_per_spray : ℕ) (promotion : ℕ → ℕ) (total_paid : ℕ)
  (h1 : cost_per_spray = 3)
  (h2 : ∀ n, promotion n = 2 * n)
  (h3 : total_paid = 15) : (total_paid / cost_per_spray) * 2 = 10 :=
by
  sorry

end bob_buys_nose_sprays_l1331_133193


namespace gcd_expression_l1331_133129

noncomputable def odd_multiple_of_7771 (a : ℕ) : Prop := 
  ∃ k : ℕ, k % 2 = 1 ∧ a = 7771 * k

theorem gcd_expression (a : ℕ) (h : odd_multiple_of_7771 a) : 
  Int.gcd (8 * a^2 + 57 * a + 132) (2 * a + 9) = 9 :=
  sorry

end gcd_expression_l1331_133129


namespace remainder_division_l1331_133170

theorem remainder_division
  (P E M S F N T : ℕ)
  (h1 : P = E * M + S)
  (h2 : M = N * F + T) :
  (∃ r, P = (EF + 1) * (P / (EF + 1)) + r ∧ r = ET + S - N) :=
sorry

end remainder_division_l1331_133170


namespace surface_area_of_solid_l1331_133175

noncomputable def solid_surface_area (r : ℝ) (h : ℝ) : ℝ :=
  2 * Real.pi * r * h

theorem surface_area_of_solid : solid_surface_area 1 3 = 6 * Real.pi := by
  sorry

end surface_area_of_solid_l1331_133175


namespace hyperbola_eccentricity_is_2_l1331_133135

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) : ℝ :=
c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ)
  (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) :
  hyperbola_eccentricity a b c h H1 H2 = 2 :=
sorry

end hyperbola_eccentricity_is_2_l1331_133135


namespace minimize_blue_surface_l1331_133185

noncomputable def fraction_blue_surface_area : ℚ := 1 / 8

theorem minimize_blue_surface
  (total_cubes : ℕ)
  (blue_cubes : ℕ)
  (green_cubes : ℕ)
  (edge_length : ℕ)
  (surface_area : ℕ)
  (blue_surface_area : ℕ)
  (fraction_blue : ℚ)
  (h1 : total_cubes = 64)
  (h2 : blue_cubes = 20)
  (h3 : green_cubes = 44)
  (h4 : edge_length = 4)
  (h5 : surface_area = 6 * edge_length^2)
  (h6 : blue_surface_area = 12)
  (h7 : fraction_blue = blue_surface_area / surface_area) :
  fraction_blue = fraction_blue_surface_area :=
by
  sorry

end minimize_blue_surface_l1331_133185


namespace rice_price_per_kg_l1331_133154

theorem rice_price_per_kg (price1 price2 : ℝ) (amount1 amount2 : ℝ) (total_cost total_weight : ℝ) (P : ℝ)
  (h1 : price1 = 6.60)
  (h2 : amount1 = 49)
  (h3 : price2 = 9.60)
  (h4 : amount2 = 56)
  (h5 : total_cost = price1 * amount1 + price2 * amount2)
  (h6 : total_weight = amount1 + amount2)
  (h7 : P = total_cost / total_weight) :
  P = 8.20 := 
by sorry

end rice_price_per_kg_l1331_133154


namespace base7_addition_sum_l1331_133143

theorem base7_addition_sum :
  let n1 := 256
  let n2 := 463
  let n3 := 132
  n1 + n2 + n3 = 1214 := sorry

end base7_addition_sum_l1331_133143


namespace solve_equation1_solve_equation2_l1331_133174

-- Define the first equation (x-3)^2 + 2x(x-3) = 0
def equation1 (x : ℝ) : Prop := (x - 3)^2 + 2 * x * (x - 3) = 0

-- Define the second equation x^2 - 4x + 1 = 0
def equation2 (x : ℝ) : Prop := x^2 - 4 * x + 1 = 0

-- Theorem stating the solutions for the first equation
theorem solve_equation1 : ∀ (x : ℝ), equation1 x ↔ x = 3 ∨ x = 1 :=
by
  intro x
  sorry  -- Proof is omitted

-- Theorem stating the solutions for the second equation
theorem solve_equation2 : ∀ (x : ℝ), equation2 x ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  intro x
  sorry  -- Proof is omitted

end solve_equation1_solve_equation2_l1331_133174


namespace largest_possible_s_l1331_133139

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) 
  (h3 : ((r - 2) * 180 : ℚ) / r = (29 / 28) * ((s - 2) * 180 / s)) :
    s = 114 := by sorry

end largest_possible_s_l1331_133139


namespace symmetric_line_eq_l1331_133140

-- Given lines
def line₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def mirror_line (x y : ℝ) : Prop := y = -x

-- Definition of symmetry about the line y = -x
def symmetric_about (l₁ l₂: ℝ → ℝ → Prop) : Prop :=
∀ x y, l₁ x y ↔ l₂ y (-x)

-- Definition of line l₂ that is symmetric to line₁ about the mirror_line
def line₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem stating that the symmetric line to line₁ about y = -x is line₂
theorem symmetric_line_eq :
  symmetric_about line₁ line₂ :=
sorry

end symmetric_line_eq_l1331_133140


namespace three_digit_number_mul_seven_results_638_l1331_133163

theorem three_digit_number_mul_seven_results_638 (N : ℕ) 
  (hN1 : 100 ≤ N) 
  (hN2 : N < 1000)
  (hN3 : ∃ (x : ℕ), 7 * N = 1000 * x + 638) : N = 234 := 
sorry

end three_digit_number_mul_seven_results_638_l1331_133163


namespace distance_from_center_to_point_l1331_133186

theorem distance_from_center_to_point :
  let circle_center := (5, -7)
  let point := (3, -4)
  let distance := Real.sqrt ((3 - 5)^2 + (-4 + 7)^2)
  distance = Real.sqrt 13 := sorry

end distance_from_center_to_point_l1331_133186


namespace smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l1331_133184

theorem smallest_pos_int_greater_than_one_rel_prime_multiple_of_7 (x : ℕ) :
  (x > 1) ∧ (gcd x 210 = 7) ∧ (7 ∣ x) → x = 49 :=
by {
  sorry
}

end smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l1331_133184


namespace village_transportation_problem_l1331_133117

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

variable (total odd : ℕ) (a : ℕ)

theorem village_transportation_problem 
  (h_total : total = 15)
  (h_odd : odd = 7)
  (h_selected : 10 = 10)
  (h_eq : (comb 7 4) * (comb 8 6) / (comb 15 10) = (comb 7 (10 - a)) * (comb 8 a) / (comb 15 10)) :
  a = 6 := 
sorry

end village_transportation_problem_l1331_133117


namespace pirate_coins_l1331_133180

theorem pirate_coins (x : ℕ) : 
  (x * (x + 1)) / 2 = 3 * x → 4 * x = 20 := by
  sorry

end pirate_coins_l1331_133180


namespace k_range_proof_l1331_133191

/- Define points in the Cartesian plane as ordered pairs. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/- Define two points P and Q. -/
def P : Point := { x := -1, y := 1 }
def Q : Point := { x := 2, y := 2 }

/- Define the line equation. -/
def line_equation (k : ℝ) (x : ℝ) : ℝ :=
  k * x - 1

/- Define the range of k. -/
def k_range (k : ℝ) : Prop :=
  1 / 3 < k ∧ k < 3 / 2

/- Theorem statement. -/
theorem k_range_proof (k : ℝ) (intersects_PQ_extension : ∀ k : ℝ, ∀ x : ℝ, ((P.y ≤ line_equation k x ∧ line_equation k x ≤ Q.y) ∧ line_equation k x ≠ Q.y) → k_range k) :
  ∀ k, k_range k :=
by
  sorry

end k_range_proof_l1331_133191


namespace area_of_figure_M_l1331_133189

noncomputable def figure_M_area : Real :=
  sorry

theorem area_of_figure_M :
  figure_M_area = 3 :=
  sorry

end area_of_figure_M_l1331_133189


namespace projectile_hits_ground_at_5_over_2_l1331_133157

theorem projectile_hits_ground_at_5_over_2 :
  ∃ t : ℚ, (-20) * t ^ 2 + 26 * t + 60 = 0 ∧ t = 5 / 2 :=
sorry

end projectile_hits_ground_at_5_over_2_l1331_133157


namespace correct_system_of_equations_l1331_133119

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : x - y = 5) (h2 : y - (1/2) * x = 5) : 
  (x - y = 5) ∧ (y - (1/2) * x = 5) :=
by { sorry }

end correct_system_of_equations_l1331_133119


namespace g_at_4_l1331_133130

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

theorem g_at_4 : g 4 = 11 / 2 :=
by
  sorry

end g_at_4_l1331_133130


namespace quadrilateral_area_l1331_133114

theorem quadrilateral_area {AB BC : ℝ} (hAB : AB = 4) (hBC : BC = 8) :
  ∃ area : ℝ, area = 16 := by
  sorry

end quadrilateral_area_l1331_133114


namespace snowfall_difference_l1331_133169

def baldMountainSnowfallMeters : ℝ := 1.5
def billyMountainSnowfallMeters : ℝ := 3.5
def mountPilotSnowfallCentimeters : ℝ := 126
def cmPerMeter : ℝ := 100

theorem snowfall_difference :
  billyMountainSnowfallMeters * cmPerMeter + mountPilotSnowfallCentimeters - baldMountainSnowfallMeters * cmPerMeter = 326 :=
by
  sorry

end snowfall_difference_l1331_133169


namespace talia_total_distance_l1331_133147

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end talia_total_distance_l1331_133147


namespace correct_statements_l1331_133111

variables (a : Nat → ℤ) (d : ℤ)

-- Suppose {a_n} is an arithmetic sequence with common difference d
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions: S_11 > 0 and S_12 < 0
axiom S11_pos : S a d 11 > 0
axiom S12_neg : S a d 12 < 0

-- The goal is to determine which statements are correct
theorem correct_statements : (d < 0) ∧ (∀ n, 1 ≤ n → n ≤ 12 → S a d 6 ≥ S a d n ∧ S a d 6 ≠ S a d 11 ) := 
sorry

end correct_statements_l1331_133111


namespace amount_spent_on_tumbler_l1331_133112

def initial_amount : ℕ := 50
def spent_on_coffee : ℕ := 10
def amount_left : ℕ := 10
def total_spent : ℕ := initial_amount - amount_left

theorem amount_spent_on_tumbler : total_spent - spent_on_coffee = 30 := by
  sorry

end amount_spent_on_tumbler_l1331_133112


namespace average_age_correct_l1331_133182

def ratio (m w : ℕ) : Prop := w * 8 = m * 9

def average_age_of_group (m w : ℕ) (avg_men avg_women : ℕ) : ℚ :=
  (avg_men * m + avg_women * w) / (m + w)

/-- The average age of the group is 32 14/17 given that the ratio of the number of women to the number of men is 9 to 8, 
    the average age of the women is 30 years, and the average age of the men is 36 years. -/
theorem average_age_correct
  (m w : ℕ)
  (h_ratio : ratio m w)
  (h_avg_women : avg_age_women = 30)
  (h_avg_men : avg_age_men = 36) :
  average_age_of_group m w avg_age_men avg_age_women = 32 + (14 / 17) := 
by
  sorry

end average_age_correct_l1331_133182


namespace product_of_real_values_eq_4_l1331_133198

theorem product_of_real_values_eq_4 : ∀ s : ℝ, 
  (∃ x : ℝ, x ≠ 0 ∧ (1/(3*x) = (s - x)/9) → 
  (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (s - x)/9 → x = s - 3))) → s = 4 :=
by
  sorry

end product_of_real_values_eq_4_l1331_133198


namespace mirka_number_l1331_133137

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b
noncomputable def reversed_number (a b : ℕ) : ℕ := 10 * b + a

theorem mirka_number (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 4) (h2 : b = 2 * a) :
  original_number a b = 12 ∨ original_number a b = 24 ∨ original_number a b = 36 ∨ original_number a b = 48 :=
by
  sorry

end mirka_number_l1331_133137


namespace evaluate_expression_when_c_is_4_l1331_133144

variable (c : ℕ)

theorem evaluate_expression_when_c_is_4 : (c = 4) → ((c^2 - c! * (c - 1)^c)^2 = 3715584) :=
by
  -- This is where the proof would go, but we only need to set up the statement.
  sorry

end evaluate_expression_when_c_is_4_l1331_133144


namespace remainder_when_divided_by_6_l1331_133123

theorem remainder_when_divided_by_6 (n : ℤ) (h_pos : 0 < n) (h_mod12 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l1331_133123


namespace tangent_line_to_curve_at_point_l1331_133131

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ),
  (y = 2 * Real.log x) →
  (x = 2) →
  (y = 2 * Real.log 2) →
  (x - y + 2 * Real.log 2 - 2 = 0) := by
  sorry

end tangent_line_to_curve_at_point_l1331_133131


namespace pencils_total_l1331_133177

theorem pencils_total (p1 p2 : ℕ) (h1 : p1 = 3) (h2 : p2 = 7) : p1 + p2 = 10 := by
  sorry

end pencils_total_l1331_133177


namespace area_union_of_rectangle_and_circle_l1331_133116

theorem area_union_of_rectangle_and_circle :
  let length := 12
  let width := 15
  let r := 15
  let area_rectangle := length * width
  let area_circle := Real.pi * r^2
  let area_overlap := (1/4) * area_circle
  let area_union := area_rectangle + area_circle - area_overlap
  area_union = 180 + 168.75 * Real.pi := by
    sorry

end area_union_of_rectangle_and_circle_l1331_133116


namespace sheets_of_paper_l1331_133151

theorem sheets_of_paper (S E : ℕ) (h1 : S - E = 100) (h2 : E = S / 3 - 25) : S = 120 :=
sorry

end sheets_of_paper_l1331_133151


namespace max_papers_l1331_133158

theorem max_papers (p c r : ℕ) (h1 : p ≥ 2) (h2 : c ≥ 1) (h3 : 3 * p + 5 * c + 9 * r = 72) : r ≤ 6 :=
sorry

end max_papers_l1331_133158


namespace rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l1331_133105

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def transition_rule (n : ℕ) : ℕ :=
  2 * (sum_of_digits n)

theorem rule_for_sequence :
  transition_rule 3 = 6 ∧ transition_rule 6 = 12 :=
by
  sorry

theorem natural_number_self_map :
  ∀ n : ℕ, transition_rule n = n ↔ n = 18 :=
by
  sorry

theorem power_of_2_to_single_digit :
  ∃ x : ℕ, transition_rule (2^1991) = x ∧ x < 10 :=
by
  sorry

end rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l1331_133105


namespace FG_length_of_trapezoid_l1331_133178

-- Define the dimensions and properties of trapezoid EFGH.
def EFGH_trapezoid (area : ℝ) (altitude : ℝ) (EF : ℝ) (GH : ℝ) : Prop :=
  area = 180 ∧ altitude = 9 ∧ EF = 12 ∧ GH = 20

-- State the theorem to prove the length of FG.
theorem FG_length_of_trapezoid : 
  ∀ {E F G H : Type} (area EF GH fg : ℝ) (altitude : ℝ),
  EFGH_trapezoid area altitude EF GH → fg = 6.57 :=
by sorry

end FG_length_of_trapezoid_l1331_133178


namespace general_term_formula_l1331_133118

def Sn (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 2^(n + 1)

theorem general_term_formula (a_n : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → Sn a_n n = (2 * a_n n - 2^(n + 1))) :
  ∀ n : ℕ, n > 0 → a_n n = (n + 1) * 2^n :=
sorry

end general_term_formula_l1331_133118


namespace intersect_sets_l1331_133165

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersect_sets : M ∩ N = {1, 2} :=
by
  sorry

end intersect_sets_l1331_133165


namespace unique_pair_solution_l1331_133167

theorem unique_pair_solution:
  ∃! (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n), a^2 = 2^n + 15 ∧ a = 4 ∧ n = 0 := sorry

end unique_pair_solution_l1331_133167


namespace monthly_production_increase_l1331_133183

/-- A salt manufacturing company produced 3000 tonnes in January and increased its
    production by some tonnes every month over the previous month until the end
    of the year. Given that the average daily production was 116.71232876712328 tonnes,
    determine the monthly production increase. -/
theorem monthly_production_increase :
  let initial_production := 3000
  let daily_average_production := 116.71232876712328
  let days_per_year := 365
  let total_yearly_production := daily_average_production * days_per_year
  let months_per_year := 12
  ∃ (x : ℝ), total_yearly_production = (months_per_year / 2) * (2 * initial_production + (months_per_year - 1) * x) → x = 100 :=
sorry

end monthly_production_increase_l1331_133183


namespace multiplication_result_l1331_133149

theorem multiplication_result
  (h : 16 * 21.3 = 340.8) :
  213 * 16 = 3408 :=
sorry

end multiplication_result_l1331_133149


namespace sq_in_scientific_notation_l1331_133162

theorem sq_in_scientific_notation (a : Real) (h : a = 25000) (h_scientific : a = 2.5 * 10^4) : a^2 = 6.25 * 10^8 :=
sorry

end sq_in_scientific_notation_l1331_133162


namespace percentage_brand_A_l1331_133196

theorem percentage_brand_A
  (A B : ℝ)
  (h1 : 0.6 * A + 0.65 * B = 0.5 * (A + B))
  : (A / (A + B)) * 100 = 60 :=
by
  sorry

end percentage_brand_A_l1331_133196


namespace solve_for_x_minus_y_l1331_133128

theorem solve_for_x_minus_y (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 :=
by
  sorry

end solve_for_x_minus_y_l1331_133128


namespace smallest_n_modulo_l1331_133181

theorem smallest_n_modulo :
  ∃ n : ℕ, 0 < n ∧ 5 * n % 26 = 1846 % 26 ∧ n = 26 :=
by
  sorry

end smallest_n_modulo_l1331_133181


namespace find_x_l1331_133126

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l1331_133126


namespace max_sum_hex_digits_l1331_133106

theorem max_sum_hex_digits 
  (a b c : ℕ) (y : ℕ) 
  (h_a : 0 ≤ a ∧ a < 16)
  (h_b : 0 ≤ b ∧ b < 16)
  (h_c : 0 ≤ c ∧ c < 16)
  (h_y : 0 < y ∧ y ≤ 16)
  (h_fraction : (a * 256 + b * 16 + c) * y = 4096) : 
  a + b + c ≤ 1 :=
sorry

end max_sum_hex_digits_l1331_133106


namespace find_a_plus_b_l1331_133142

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * Real.log x

theorem find_a_plus_b (a b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ f a b x = 1 / 2 ∧ (deriv (f a b)) 1 = 0) →
  a + b = -1/2 :=
by
  sorry

end find_a_plus_b_l1331_133142


namespace matrix_inverse_correct_l1331_133194

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -2], ![5, 3]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/22, 1/11], ![-5/22, 2/11]]

theorem matrix_inverse_correct : A⁻¹ = A_inv :=
  by
    sorry

end matrix_inverse_correct_l1331_133194


namespace ascending_order_perimeters_l1331_133156

noncomputable def hypotenuse (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

theorem ascending_order_perimeters (r : ℝ) (h_r_pos : 0 < r) : 
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

end ascending_order_perimeters_l1331_133156


namespace sin_arcsin_plus_arctan_l1331_133172

theorem sin_arcsin_plus_arctan :
  let a := Real.arcsin (4/5)
  let b := Real.arctan 1
  Real.sin (a + b) = (7 * Real.sqrt 2) / 10 := by
  sorry

end sin_arcsin_plus_arctan_l1331_133172


namespace ellipse_focus_eccentricity_l1331_133122

theorem ellipse_focus_eccentricity (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 2) + (y^2 / m) = 1 → y = 0 ∨ x = 0) ∧
  (∀ e : ℝ, e = 1 / 2) →
  m = 3 / 2 :=
sorry

end ellipse_focus_eccentricity_l1331_133122


namespace total_pairs_of_shoes_tried_l1331_133133

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l1331_133133


namespace sum_diff_reciprocals_equals_zero_l1331_133125

theorem sum_diff_reciprocals_equals_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (1 / (a + 1)) + (1 / (a - 1)) + (1 / (b + 1)) + (1 / (b - 1)) = 0) :
  (a + b) - (1 / a + 1 / b) = 0 :=
by
  sorry

end sum_diff_reciprocals_equals_zero_l1331_133125


namespace negation_exists_ltx2_plus_x_plus_1_lt_0_l1331_133138

theorem negation_exists_ltx2_plus_x_plus_1_lt_0 :
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_exists_ltx2_plus_x_plus_1_lt_0_l1331_133138


namespace no_triangle_possible_l1331_133192

-- Define the lengths of the sticks
def stick_lengths : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

-- The theorem stating the impossibility of forming a triangle with any combination of these lengths
theorem no_triangle_possible : ¬ ∃ (a b c : ℕ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  sorry

end no_triangle_possible_l1331_133192


namespace sequence_general_term_l1331_133161

theorem sequence_general_term (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 5)
  (h4 : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l1331_133161


namespace increasing_function_cond_l1331_133148

theorem increasing_function_cond (f : ℝ → ℝ)
  (h : ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0) :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end increasing_function_cond_l1331_133148
