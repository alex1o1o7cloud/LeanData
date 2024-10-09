import Mathlib

namespace factor_of_lcm_l352_35285

theorem factor_of_lcm (A B hcf : ℕ) (h_gcd : Nat.gcd A B = hcf) (hcf_eq : hcf = 16) (A_eq : A = 224) :
  ∃ X : ℕ, X = 14 := by
  sorry

end factor_of_lcm_l352_35285


namespace max_edges_intersected_by_plane_l352_35228

theorem max_edges_intersected_by_plane (p : ℕ) (h_pos : p > 0) : ℕ :=
  let vertices := 2 * p
  let base_edges := p
  let lateral_edges := p
  let total_edges := 3 * p
  total_edges

end max_edges_intersected_by_plane_l352_35228


namespace common_ratio_geom_seq_l352_35247

variable {a : ℕ → ℝ} {q : ℝ}

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

theorem common_ratio_geom_seq (h₁ : a 5 = 1) (h₂ : a 8 = 8) (hq : geom_seq a q) : q = 2 :=
by
  sorry

end common_ratio_geom_seq_l352_35247


namespace batsman_average_after_17_matches_l352_35271

theorem batsman_average_after_17_matches (A : ℕ) (h : (17 * (A + 3) = 16 * A + 87)) : A + 3 = 39 := by
  sorry

end batsman_average_after_17_matches_l352_35271


namespace george_room_painting_l352_35270

-- Define the number of ways to choose 2 colors out of 9 without considering the restriction
def num_ways_total : ℕ := Nat.choose 9 2

-- Define the restriction that red and pink should not be combined
def num_restricted_ways : ℕ := 1

-- Define the final number of permissible combinations
def num_permissible_combinations : ℕ := num_ways_total - num_restricted_ways

theorem george_room_painting :
  num_permissible_combinations = 35 :=
by
  sorry

end george_room_painting_l352_35270


namespace inequality_solution_set_l352_35286

theorem inequality_solution_set (x : ℝ) : (-2 < x ∧ x ≤ 3) ↔ (x - 3) / (x + 2) ≤ 0 := 
sorry

end inequality_solution_set_l352_35286


namespace deductive_reasoning_option_l352_35255

inductive ReasoningType
| deductive
| inductive
| analogical

-- Definitions based on conditions
def option_A : ReasoningType := ReasoningType.inductive
def option_B : ReasoningType := ReasoningType.deductive
def option_C : ReasoningType := ReasoningType.inductive
def option_D : ReasoningType := ReasoningType.analogical

-- The main theorem to prove
theorem deductive_reasoning_option : option_B = ReasoningType.deductive :=
by sorry

end deductive_reasoning_option_l352_35255


namespace fraction_proof_l352_35275

variables (m n p q : ℚ)

theorem fraction_proof
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 :=
by sorry

end fraction_proof_l352_35275


namespace value_of_x2_plus_4y2_l352_35201

theorem value_of_x2_plus_4y2 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : x * y = -12) : x^2 + 4*y^2 = 84 := 
  sorry

end value_of_x2_plus_4y2_l352_35201


namespace canonical_line_eq_l352_35237

-- Define the system of linear equations
def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x - 3 * y - 2 * z + 6 = 0 ∧ x - 3 * y + z + 3 = 0)

-- Define the canonical equation of the line
def canonical_equation (x y z : ℝ) : Prop :=
  (x + 3) / 9 = y / 4 ∧ (x + 3) / 9 = z / 3 ∧ y / 4 = z / 3

-- The theorem to prove equivalence
theorem canonical_line_eq : 
  ∀ (x y z : ℝ), system_of_equations x y z → canonical_equation x y z :=
by
  intros x y z H
  sorry

end canonical_line_eq_l352_35237


namespace set_intersection_complement_l352_35206

open Set

def I := {n : ℕ | True}
def A := {x ∈ I | 2 ≤ x ∧ x ≤ 10}
def B := {x | Nat.Prime x}

theorem set_intersection_complement :
  A ∩ (I \ B) = {4, 6, 8, 9, 10} := by
  sorry

end set_intersection_complement_l352_35206


namespace probability_non_adjacent_two_twos_l352_35251

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end probability_non_adjacent_two_twos_l352_35251


namespace douglas_vote_percentage_is_66_l352_35283

noncomputable def percentDouglasVotes (v : ℝ) : ℝ :=
  let votesX := 0.74 * (2 * v)
  let votesY := 0.5000000000000002 * v
  let totalVotes := 3 * v
  let totalDouglasVotes := votesX + votesY
  (totalDouglasVotes / totalVotes) * 100

theorem douglas_vote_percentage_is_66 :
  ∀ v : ℝ, percentDouglasVotes v = 66 := 
by
  intros v
  unfold percentDouglasVotes
  sorry

end douglas_vote_percentage_is_66_l352_35283


namespace range_of_m_l352_35248

theorem range_of_m :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x < -1 ∨ x > 3)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l352_35248


namespace largest_three_digit_geometric_sequence_with_8_l352_35209

theorem largest_three_digit_geometric_sequence_with_8 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n = 842 ∧ (∃ (a b c : ℕ), n = 100*a + 10*b + c ∧ a = 8 ∧ (a * c = b^2) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ) :=
by
  sorry

end largest_three_digit_geometric_sequence_with_8_l352_35209


namespace inequality_am_gm_l352_35264

theorem inequality_am_gm 
  (a b c d : ℝ) 
  (h_nonneg_a : 0 ≤ a) 
  (h_nonneg_b : 0 ≤ b) 
  (h_nonneg_c : 0 ≤ c) 
  (h_nonneg_d : 0 ≤ d) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 :=
by
  sorry


end inequality_am_gm_l352_35264


namespace cube_root_eq_self_l352_35252

theorem cube_root_eq_self (a : ℝ) (h : a^(3:ℕ) = a) : a = 1 ∨ a = -1 ∨ a = 0 := 
sorry

end cube_root_eq_self_l352_35252


namespace average_of_last_three_l352_35278

theorem average_of_last_three (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : A + D = 11)
  (h3 : D = 4) : 
  (B + C + D) / 3 = 5 :=
by
  sorry

end average_of_last_three_l352_35278


namespace set_intersection_l352_35219

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x^2 < 4

theorem set_intersection : {x | A x} ∩ {x | B x} = {x | 0 < x ∧ x < 2} := by
  sorry

end set_intersection_l352_35219


namespace symmetric_polynomial_identity_l352_35217

variable (x y z : ℝ)
def σ1 : ℝ := x + y + z
def σ2 : ℝ := x * y + y * z + z * x
def σ3 : ℝ := x * y * z

theorem symmetric_polynomial_identity : 
  x^3 + y^3 + z^3 = σ1 x y z ^ 3 - 3 * σ1 x y z * σ2 x y z + 3 * σ3 x y z := by
  sorry

end symmetric_polynomial_identity_l352_35217


namespace right_triangle_c_l352_35210

theorem right_triangle_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 4)
  (h3 : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2)) :
  c = 5 ∨ c = Real.sqrt 7 :=
by
  -- Proof omitted
  sorry

end right_triangle_c_l352_35210


namespace tricycles_count_l352_35277

theorem tricycles_count (b t : ℕ) 
  (hyp1 : b + t = 10)
  (hyp2 : 2 * b + 3 * t = 26) : 
  t = 6 := 
by 
  sorry

end tricycles_count_l352_35277


namespace smallest_value_4x_plus_3y_l352_35225

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end smallest_value_4x_plus_3y_l352_35225


namespace sqrt_sequence_convergence_l352_35211

theorem sqrt_sequence_convergence :
  ∃ x : ℝ, (x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2) :=
sorry

end sqrt_sequence_convergence_l352_35211


namespace card_sequence_probability_l352_35289

noncomputable def probability_of_sequence : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem card_sequence_probability :
  probability_of_sequence = 4/33150 := 
by 
  sorry

end card_sequence_probability_l352_35289


namespace sum_of_altitudes_less_than_sum_of_sides_l352_35212

-- Define a triangle with sides and altitudes properties
structure Triangle :=
(A B C : Point)
(a b c : ℝ)
(m_a m_b m_c : ℝ)
(sides : a + b > c ∧ b + c > a ∧ c + a > b) -- Triangle Inequality

axiom altitude_property (T : Triangle) :
  T.m_a < T.b ∧ T.m_b < T.c ∧ T.m_c < T.a

-- The theorem to prove
theorem sum_of_altitudes_less_than_sum_of_sides (T : Triangle) :
  T.m_a + T.m_b + T.m_c < T.a + T.b + T.c :=
sorry

end sum_of_altitudes_less_than_sum_of_sides_l352_35212


namespace more_males_l352_35224

theorem more_males {Total_attendees Male_attendees : ℕ} (h1 : Total_attendees = 120) (h2 : Male_attendees = 62) :
  Male_attendees - (Total_attendees - Male_attendees) = 4 :=
by
  sorry

end more_males_l352_35224


namespace teacher_li_sheets_l352_35243

theorem teacher_li_sheets (x : ℕ)
    (h1 : ∀ (n : ℕ), n = 24 → (x / 24) = ((x / 32) + 2)) :
    x = 192 := by
  sorry

end teacher_li_sheets_l352_35243


namespace find_x_minus_y_l352_35233

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + 3 * y = 14) (h2 : x + 4 * y = 11) : x - y = 3 := by
  sorry

end find_x_minus_y_l352_35233


namespace max_quotient_l352_35231

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : (b / a) ≤ 15 :=
  sorry

end max_quotient_l352_35231


namespace range_of_a_l352_35266

def condition1 (a : ℝ) : Prop := (2 - a) ^ 2 < 1
def condition2 (a : ℝ) : Prop := (3 - a) ^ 2 ≥ 1

theorem range_of_a (a : ℝ) (h1 : condition1 a) (h2 : condition2 a) :
  1 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l352_35266


namespace vector_minimization_and_angle_condition_l352_35207

noncomputable def find_OC_condition (C_op C_oa C_ob : ℝ × ℝ) 
  (C : ℝ × ℝ) : Prop := 
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  (CA.1 * CB.1 + CA.2 * CB.2) ≤ (C_op.1 * CB.1 + C_op.2 * CB.2)

theorem vector_minimization_and_angle_condition (C : ℝ × ℝ) 
  (C_op := (2, 1)) (C_oa := (1, 7)) (C_ob := (5, 1)) :
  (C = (4, 2)) → 
  find_OC_condition C_op C_oa C_ob C →
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                 (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
  cos_ACB = -4 * Real.sqrt (17) / 17 :=
  by 
    intro h1 find
    let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
    let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
    let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                   (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
    exact sorry

end vector_minimization_and_angle_condition_l352_35207


namespace ab_range_l352_35254

theorem ab_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b + 8) : a * b ≥ 16 :=
sorry

end ab_range_l352_35254


namespace num_valid_pairs_l352_35205

theorem num_valid_pairs (a b : ℕ) (h1 : b > a) (h2 : a > 4) (h3 : b > 4)
(h4 : a * b = 3 * (a - 4) * (b - 4)) : 
    (1 + (a - 6) = 1 ∧ 72 = b - 6) ∨
    (2 + (a - 6) = 2 ∧ 36 = b - 6) ∨
    (3 + (a - 6) = 3 ∧ 24 = b - 6) ∨
    (4 + (a - 6) = 4 ∧ 18 = b - 6) :=
sorry

end num_valid_pairs_l352_35205


namespace quadratic_equation_with_given_means_l352_35234

theorem quadratic_equation_with_given_means (α β : ℝ)
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 12) : 
  x ^ 2 - 16 * x + 144 = 0 :=
sorry

end quadratic_equation_with_given_means_l352_35234


namespace solution_to_problem_l352_35287

def problem_statement : Prop :=
  (3^202 + 7^203)^2 - (3^202 - 7^203)^2 = 59 * 10^202

theorem solution_to_problem : problem_statement := 
  by sorry

end solution_to_problem_l352_35287


namespace terminal_sides_y_axis_l352_35253

theorem terminal_sides_y_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 2) ∨ 
  (∃ k : ℤ, α = (2 * k + 1) * Real.pi + Real.pi / 2) ↔ 
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 := 
by sorry

end terminal_sides_y_axis_l352_35253


namespace tan_2alpha_and_cos_beta_l352_35257

theorem tan_2alpha_and_cos_beta
    (α β : ℝ)
    (h1 : 0 < β ∧ β < α ∧ α < (Real.pi / 2))
    (h2 : Real.sin α = (4 * Real.sqrt 3) / 7)
    (h3 : Real.cos (β - α) = 13 / 14) :
    Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ Real.cos β = 1 / 2 := by
  sorry

end tan_2alpha_and_cos_beta_l352_35257


namespace masks_purchased_in_first_batch_l352_35276

theorem masks_purchased_in_first_batch
    (cost_first_batch cost_second_batch : ℝ)
    (quantity_ratio : ℝ)
    (unit_price_difference : ℝ)
    (h1 : cost_first_batch = 1600)
    (h2 : cost_second_batch = 6000)
    (h3 : quantity_ratio = 3)
    (h4 : unit_price_difference = 2) :
    ∃ x : ℝ, (cost_first_batch / x) + unit_price_difference = (cost_second_batch / (quantity_ratio * x)) ∧ x = 200 :=
by {
    sorry
}

end masks_purchased_in_first_batch_l352_35276


namespace solve_system_l352_35223

theorem solve_system : ∀ (x y : ℤ), 2 * x + y = 5 → x + 2 * y = 6 → x - y = -1 :=
by
  intros x y h1 h2
  sorry

end solve_system_l352_35223


namespace remaining_gallons_to_fill_tank_l352_35284

-- Define the conditions as constants
def tank_capacity : ℕ := 50
def rate_seconds_per_gallon : ℕ := 20
def time_poured_minutes : ℕ := 6

-- Define the number of gallons poured per minute
def gallons_per_minute : ℕ := 60 / rate_seconds_per_gallon

def gallons_poured (minutes : ℕ) : ℕ :=
  minutes * gallons_per_minute

-- The main statement to prove the remaining gallons needed
theorem remaining_gallons_to_fill_tank : 
  tank_capacity - gallons_poured time_poured_minutes = 32 :=
by
  sorry

end remaining_gallons_to_fill_tank_l352_35284


namespace sum_of_excluded_solutions_l352_35291

noncomputable def P : ℚ := 3
noncomputable def Q : ℚ := 5 / 3
noncomputable def R : ℚ := 25 / 3

theorem sum_of_excluded_solutions :
    (P = 3) ∧
    (Q = 5 / 3) ∧
    (R = 25 / 3) ∧
    (∀ x, (x ≠ -R ∧ x ≠ -10) →
    ((x + Q) * (P * x + 50) / ((x + R) * (x + 10)) = 3)) →
    (-R + -10 = -55 / 3) :=
by
  sorry

end sum_of_excluded_solutions_l352_35291


namespace proof_part1_proof_part2_l352_35238

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l352_35238


namespace total_number_of_athletes_l352_35269

theorem total_number_of_athletes (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
sorry

end total_number_of_athletes_l352_35269


namespace tan_half_alpha_l352_35267

theorem tan_half_alpha (α : ℝ) (h1 : 180 * (Real.pi / 180) < α) 
  (h2 : α < 270 * (Real.pi / 180)) 
  (h3 : Real.sin ((270 * (Real.pi / 180)) + α) = 4 / 5) : 
  Real.tan (α / 2) = -1 / 3 :=
by 
  -- Informal note: proof would be included here.
  sorry

end tan_half_alpha_l352_35267


namespace no_three_digits_all_prime_l352_35214

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that forms a three-digit number from digits a, b, c
def form_three_digit (a b c : ℕ) : ℕ :=
100 * a + 10 * b + c

-- Define a function to check if all permutations of three digits form prime numbers
def all_permutations_prime (a b c : ℕ) : Prop :=
is_prime (form_three_digit a b c) ∧
is_prime (form_three_digit a c b) ∧
is_prime (form_three_digit b a c) ∧
is_prime (form_three_digit b c a) ∧
is_prime (form_three_digit c a b) ∧
is_prime (form_three_digit c b a)

-- The main theorem stating that there are no three distinct digits making all permutations prime
theorem no_three_digits_all_prime : ¬∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  all_permutations_prime a b c :=
sorry

end no_three_digits_all_prime_l352_35214


namespace park_area_l352_35259

theorem park_area (L B : ℝ) (h1 : L = B / 2) (h2 : 6 * 1000 / 60 * 6 = 2 * (L + B)) : L * B = 20000 :=
by
  -- proof will go here
  sorry

end park_area_l352_35259


namespace problem1_problem2_l352_35242

-- Theorem for problem 1
theorem problem1 (a b : ℤ) : (a^3 * b^4) ^ 2 / (a * b^2) ^ 3 = a^3 * b^2 := 
by sorry

-- Theorem for problem 2
theorem problem2 (a : ℤ) : (-a^2) ^ 3 * a^2 + a^8 = 0 := 
by sorry

end problem1_problem2_l352_35242


namespace range_of_b_l352_35218

theorem range_of_b (a b x : ℝ) (ha : 0 < a ∧ a ≤ 5 / 4) (hb : 0 < b) :
  (∀ x, |x - a| < b → |x - a^2| < 1 / 2) ↔ 0 < b ∧ b ≤ 3 / 16 :=
by
  sorry

end range_of_b_l352_35218


namespace min_value_of_expression_l352_35298

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + (1 / (a - 1)) ≥ 3 :=
by sorry

end min_value_of_expression_l352_35298


namespace terry_daily_income_l352_35246

theorem terry_daily_income (T : ℕ) (h1 : ∀ j : ℕ, j = 30) (h2 : 7 * 30 = 210) (h3 : 7 * T - 210 = 42) : T = 36 := 
by
  sorry

end terry_daily_income_l352_35246


namespace modulus_z_eq_sqrt_10_l352_35263

noncomputable def z := (10 * Complex.I) / (3 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_z_eq_sqrt_10_l352_35263


namespace truncatedPyramidVolume_l352_35293

noncomputable def volumeOfTruncatedPyramid (R : ℝ) : ℝ :=
  let h := R * Real.sqrt 3 / 2
  let S_lower := 3 * R^2 * Real.sqrt 3 / 2
  let S_upper := 3 * R^2 * Real.sqrt 3 / 8
  let sqrt_term := Real.sqrt (S_lower * S_upper)
  (1/3) * h * (S_lower + S_upper + sqrt_term)

theorem truncatedPyramidVolume (R : ℝ) (h := R * Real.sqrt 3 / 2)
  (S_lower := 3 * R^2 * Real.sqrt 3 / 2)
  (S_upper := 3 * R^2 * Real.sqrt 3 / 8)
  (V := (1/3) * h * (S_lower + S_upper + Real.sqrt (S_lower * S_upper))) :
  volumeOfTruncatedPyramid R = 21 * R^3 / 16 := by
  sorry

end truncatedPyramidVolume_l352_35293


namespace find_x_in_interval_l352_35274

theorem find_x_in_interval (x : ℝ) : x^2 + 5 * x < 10 ↔ -5 < x ∧ x < 2 :=
sorry

end find_x_in_interval_l352_35274


namespace brenda_mice_left_l352_35204

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l352_35204


namespace people_joined_after_leaving_l352_35296

theorem people_joined_after_leaving 
  (p_initial : ℕ) (p_left : ℕ) (p_final : ℕ) (p_joined : ℕ) :
  p_initial = 30 → p_left = 10 → p_final = 25 → p_joined = p_final - (p_initial - p_left) → p_joined = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end people_joined_after_leaving_l352_35296


namespace problem_l352_35202

open Real

def p (x : ℝ) : Prop := 2*x^2 + 2*x + 1/2 < 0

def q (x y : ℝ) : Prop := (x^2)/4 - (y^2)/12 = 1 ∧ x ≥ 2

def x0_condition (x0 : ℝ) : Prop := sin x0 - cos x0 = sqrt 2

theorem problem (h1 : ∀ x : ℝ, ¬ p x)
               (h2 : ∃ x y : ℝ, q x y)
               (h3 : ∃ x0 : ℝ, x0_condition x0) :
               ∀ x : ℝ, ¬ ¬ p x := 
sorry

end problem_l352_35202


namespace least_number_to_add_l352_35279

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (modulus : ℕ) (h1 : n = 1076) (h2 : divisor = 23) (h3 : n % divisor = 18) :
  modulus = divisor - (n % divisor) ∧ modulus = 5 := 
sorry

end least_number_to_add_l352_35279


namespace slope_intercept_equivalence_l352_35280

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end slope_intercept_equivalence_l352_35280


namespace ratio_u_v_l352_35272

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l352_35272


namespace locus_of_point_P_l352_35294

-- Definitions and conditions
def circle_M (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4
def A_point : ℝ × ℝ := (2, 1)
def chord_BC (x y x₀ y₀ : ℝ) : Prop := (x₀ - 1) * x + y₀ * y - x₀ - 3 = 0
def point_P_locus (x₀ y₀ : ℝ) : Prop := ∃ x y, (chord_BC x y x₀ y₀) ∧ x = 2 ∧ y = 1

-- Lean 4 statement to be proved
theorem locus_of_point_P (x₀ y₀ : ℝ) (h : point_P_locus x₀ y₀) : x₀ + y₀ - 5 = 0 :=
  by
  sorry

end locus_of_point_P_l352_35294


namespace number_of_juniors_in_sample_l352_35236

theorem number_of_juniors_in_sample
  (total_students : ℕ)
  (num_freshmen : ℕ)
  (num_freshmen_sampled : ℕ)
  (num_sophomores_exceeds_num_juniors_by : ℕ)
  (num_sophomores num_juniors num_juniors_sampled : ℕ)
  (h_total : total_students = 1290)
  (h_num_freshmen : num_freshmen = 480)
  (h_num_freshmen_sampled : num_freshmen_sampled = 96)
  (h_exceeds : num_sophomores_exceeds_num_juniors_by = 30)
  (h_equation : total_students - num_freshmen = num_sophomores + num_juniors)
  (h_num_sophomores : num_sophomores = num_juniors + num_sophomores_exceeds_num_juniors_by)
  (h_fraction : num_freshmen_sampled / num_freshmen = 1 / 5)
  (h_num_juniors_sampled : num_juniors_sampled = num_juniors * (num_freshmen_sampled / num_freshmen)) :
  num_juniors_sampled = 78 := by
  sorry

end number_of_juniors_in_sample_l352_35236


namespace sandwich_cost_l352_35292

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end sandwich_cost_l352_35292


namespace find_rate_l352_35244

noncomputable def SI := 200
noncomputable def P := 800
noncomputable def T := 4

theorem find_rate : ∃ R : ℝ, SI = (P * R * T) / 100 ∧ R = 6.25 :=
by sorry

end find_rate_l352_35244


namespace profit_ratio_l352_35260

theorem profit_ratio (P_invest Q_invest : ℕ) (hP : P_invest = 500000) (hQ : Q_invest = 1000000) :
  (P_invest:ℚ) / Q_invest = 1 / 2 := 
  by
  rw [hP, hQ]
  norm_num

end profit_ratio_l352_35260


namespace product_of_two_numbers_l352_35262

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l352_35262


namespace chord_length_of_intersecting_circle_and_line_l352_35213

-- Define the conditions in Lean
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line_equation (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ - 1 = 0

-- Define the problem to prove the length of the chord
theorem chord_length_of_intersecting_circle_and_line 
  (ρ θ : ℝ) (hC : circle_equation ρ θ) (hL : line_equation ρ θ) : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 :=
by 
  sorry

end chord_length_of_intersecting_circle_and_line_l352_35213


namespace minimum_ab_l352_35226

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end minimum_ab_l352_35226


namespace number_of_unique_triangle_areas_l352_35245

theorem number_of_unique_triangle_areas :
  ∀ (G H I J K L : ℝ) (d₁ d₂ d₃ d₄ : ℝ),
    G ≠ H → H ≠ I → I ≠ J → G ≠ I → G ≠ J →
    H ≠ J →
    G - H = 1 → H - I = 1 → I - J = 2 →
    K - L = 2 →
    d₄ = abs d₃ →
    (d₁ = abs (K - G)) ∨ (d₂ = abs (L - G)) ∨ (d₁ = d₂) →
    ∃ (areas : ℕ), 
    areas = 3 :=
by sorry

end number_of_unique_triangle_areas_l352_35245


namespace stratified_sampling_third_year_l352_35295

-- The total number of students in the school
def total_students : ℕ := 2000

-- The probability of selecting a female student from the second year
def prob_female_second_year : ℚ := 0.19

-- The number of students to be selected through stratified sampling
def sample_size : ℕ := 100

-- The total number of third-year students
def third_year_students : ℕ := 500

-- The number of students to be selected from the third year in stratified sampling
def third_year_sample (total : ℕ) (third_year : ℕ) (sample : ℕ) : ℕ :=
  sample * third_year / total

-- Lean statement expressing the goal
theorem stratified_sampling_third_year :
  third_year_sample total_students third_year_students sample_size = 25 :=
by
  sorry

end stratified_sampling_third_year_l352_35295


namespace geometric_sequence_properties_l352_35249

theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) (h1 : ∀ n, S n = 3^n + t) (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 2 = 6 ∧ t = -1 :=
by
  sorry

end geometric_sequence_properties_l352_35249


namespace ratio_QP_l352_35221

theorem ratio_QP {P Q : ℚ} 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 4 → x ≠ -4 → 
    P / (x^2 - 5 * x) + Q / (x + 4) = (x^2 - 3 * x + 8) / (x^3 - 5 * x^2 + 4 * x)) : 
  Q / P = 7 / 2 := 
sorry

end ratio_QP_l352_35221


namespace numerical_identity_l352_35227

theorem numerical_identity :
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 :=
by
  -- conditions and definitions based on a) are directly used here
  sorry -- proof is not required as per instructions

end numerical_identity_l352_35227


namespace number_of_dogs_l352_35290

theorem number_of_dogs 
  (d c b : Nat) 
  (ratio : d / c / b = 3 / 7 / 12) 
  (total_dogs_and_bunnies : d + b = 375) :
  d = 75 :=
by
  -- Using the hypothesis and given conditions to prove d = 75.
  sorry

end number_of_dogs_l352_35290


namespace simplify_to_linear_form_l352_35299

theorem simplify_to_linear_form (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := 
by 
  sorry

end simplify_to_linear_form_l352_35299


namespace find_b_value_l352_35215

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l352_35215


namespace find_x_in_magic_square_l352_35241

def magicSquareProof (x d e f g h S : ℕ) : Prop :=
  (x + 25 + 75 = S) ∧
  (5 + d + e = S) ∧
  (f + g + h = S) ∧
  (x + d + h = S) ∧
  (f = 95) ∧
  (d = x - 70) ∧
  (h = 170 - x) ∧
  (e = x - 145) ∧
  (x + 25 + 75 = 5 + (x - 70) + (x - 145))

theorem find_x_in_magic_square : ∃ x d e f g h S, magicSquareProof x d e f g h S ∧ x = 310 := by
  sorry

end find_x_in_magic_square_l352_35241


namespace find_abs_xyz_l352_35240

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l352_35240


namespace radius_of_semicircular_cubicle_l352_35265

noncomputable def radius_of_semicircle (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem radius_of_semicircular_cubicle :
  radius_of_semicircle 71.9822971502571 = 14 := 
sorry

end radius_of_semicircular_cubicle_l352_35265


namespace casey_pumping_time_l352_35261

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end casey_pumping_time_l352_35261


namespace evaluate_expression_is_15_l352_35232

noncomputable def sumOfFirstNOddNumbers (n : ℕ) : ℕ :=
  n^2

noncomputable def simplifiedExpression : ℕ :=
  sumOfFirstNOddNumbers 1 +
  sumOfFirstNOddNumbers 2 +
  sumOfFirstNOddNumbers 3 +
  sumOfFirstNOddNumbers 4 +
  sumOfFirstNOddNumbers 5

theorem evaluate_expression_is_15 : simplifiedExpression = 15 := by
  sorry

end evaluate_expression_is_15_l352_35232


namespace partition_equation_solution_l352_35222

def partition (n : ℕ) : ℕ := sorry -- defining the partition function

theorem partition_equation_solution (n : ℕ) (h : partition n + partition (n + 4) = partition (n + 2) + partition (n + 3)) :
  n = 1 ∨ n = 3 ∨ n = 5 :=
sorry

end partition_equation_solution_l352_35222


namespace surface_area_calculation_l352_35281

-- Conditions:
-- Original rectangular sheet dimensions
def length : ℕ := 25
def width : ℕ := 35
-- Dimensions of the square corners
def corner_side : ℕ := 7

-- Surface area of the interior calculation
noncomputable def surface_area_interior : ℕ :=
  let original_area := length * width
  let corner_area := corner_side * corner_side
  let total_corner_area := 4 * corner_area
  original_area - total_corner_area

-- Theorem: The surface area of the interior of the resulting box
theorem surface_area_calculation : surface_area_interior = 679 := by
  -- You can fill in the details to compute the answer
  sorry

end surface_area_calculation_l352_35281


namespace count_valid_three_digit_numbers_l352_35288

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
                          a ≥ 1 ∧ a ≤ 9 ∧ 
                          b ≥ 0 ∧ b ≤ 9 ∧ 
                          c ≥ 0 ∧ c ≤ 9 ∧ 
                          (a = b ∨ b = c ∨ a = c ∨ 
                           a + b > c ∧ a + c > b ∧ b + c > a)) ∧
           n = 57 := 
sorry

end count_valid_three_digit_numbers_l352_35288


namespace triangle_inradius_l352_35250

theorem triangle_inradius (A p r : ℝ) 
    (h1 : p = 35) 
    (h2 : A = 78.75) 
    (h3 : A = (r * p) / 2) : 
    r = 4.5 :=
sorry

end triangle_inradius_l352_35250


namespace seats_per_section_correct_l352_35235

-- Define the total number of seats
def total_seats : ℕ := 270

-- Define the number of sections
def sections : ℕ := 9

-- Define the number of seats per section
def seats_per_section (total_seats sections : ℕ) : ℕ := total_seats / sections

theorem seats_per_section_correct : seats_per_section total_seats sections = 30 := by
  sorry

end seats_per_section_correct_l352_35235


namespace probability_at_least_one_multiple_of_4_l352_35230

theorem probability_at_least_one_multiple_of_4 :
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  (probability_at_least_one_multiple_of_4 = 528 / 1250) := 
by
  -- Define the conditions
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  sorry

end probability_at_least_one_multiple_of_4_l352_35230


namespace markers_in_desk_l352_35203

theorem markers_in_desk (pens pencils markers : ℕ) 
  (h_ratio : pens = 2 * pencils ∧ pens = 2 * markers / 5) 
  (h_pens : pens = 10) : markers = 25 :=
by
  sorry

end markers_in_desk_l352_35203


namespace counties_percentage_l352_35282

theorem counties_percentage (a b c : ℝ) (ha : a = 0.2) (hb : b = 0.35) (hc : c = 0.25) :
  a + b + c = 0.8 :=
by
  rw [ha, hb, hc]
  sorry

end counties_percentage_l352_35282


namespace suitable_for_comprehensive_survey_l352_35208

-- Define the four survey options as a custom data type
inductive SurveyOption
  | A : SurveyOption -- Survey on the water quality of the Beijiang River
  | B : SurveyOption -- Survey on the quality of rice dumplings in the market during the Dragon Boat Festival
  | C : SurveyOption -- Survey on the vision of 50 students in a class
  | D : SurveyOption -- Survey by energy-saving lamp manufacturers on the service life of a batch of energy-saving lamps

-- Define feasibility for a comprehensive survey
def isComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => False
  | SurveyOption.B => False
  | SurveyOption.C => True
  | SurveyOption.D => False

-- The statement to be proven
theorem suitable_for_comprehensive_survey : ∃! o : SurveyOption, isComprehensiveSurvey o := by
  sorry

end suitable_for_comprehensive_survey_l352_35208


namespace birds_joined_l352_35220

def numBirdsInitially : Nat := 1
def numBirdsNow : Nat := 5

theorem birds_joined : numBirdsNow - numBirdsInitially = 4 := by
  -- proof goes here
  sorry

end birds_joined_l352_35220


namespace cost_of_items_l352_35258

theorem cost_of_items (x y z : ℝ)
  (h1 : 20 * x + 3 * y + 2 * z = 32)
  (h2 : 39 * x + 5 * y + 3 * z = 58) :
  5 * (x + y + z) = 30 := by
  sorry

end cost_of_items_l352_35258


namespace Seokjin_total_problems_l352_35200

theorem Seokjin_total_problems (initial_problems : ℕ) (additional_problems : ℕ)
  (h1 : initial_problems = 12) (h2 : additional_problems = 7) :
  initial_problems + additional_problems = 19 :=
by
  sorry

end Seokjin_total_problems_l352_35200


namespace eliza_is_18_l352_35216

-- Define the relevant ages
def aunt_ellen_age : ℕ := 48
def dina_age : ℕ := aunt_ellen_age / 2
def eliza_age : ℕ := dina_age - 6

-- Theorem to prove Eliza's age is 18
theorem eliza_is_18 : eliza_age = 18 := by
  sorry

end eliza_is_18_l352_35216


namespace average_of_three_quantities_l352_35239

theorem average_of_three_quantities (a b c d e : ℝ) 
    (h1 : (a + b + c + d + e) / 5 = 8)
    (h2 : (d + e) / 2 = 14) :
    (a + b + c) / 3 = 4 := 
sorry

end average_of_three_quantities_l352_35239


namespace omar_rolls_l352_35256

-- Define the conditions
def karen_rolls : ℕ := 229
def total_rolls : ℕ := 448

-- Define the main theorem to prove the number of rolls by Omar
theorem omar_rolls : (total_rolls - karen_rolls) = 219 := by
  sorry

end omar_rolls_l352_35256


namespace profit_is_correct_l352_35229

-- Definitions of the conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def price_per_set : ℕ := 50
def sets_sold : ℕ := 500

-- Derived calculations
def revenue (sets_sold : ℕ) (price_per_set : ℕ) : ℕ :=
  sets_sold * price_per_set

def manufacturing_costs (initial_outlay : ℕ) (cost_per_set : ℕ) (sets_sold : ℕ) : ℕ :=
  initial_outlay + (cost_per_set * sets_sold)

def profit (revenue : ℕ) (manufacturing_costs : ℕ) : ℕ :=
  revenue - manufacturing_costs

-- Theorem stating the problem
theorem profit_is_correct : 
  profit (revenue sets_sold price_per_set) (manufacturing_costs initial_outlay cost_per_set sets_sold) = 5000 :=
by
  sorry

end profit_is_correct_l352_35229


namespace ratio_payment_shared_side_l352_35297

variable (length_side length_back : ℕ) (cost_per_foot cole_payment : ℕ)
variables (neighbor_back_contrib neighbor_left_contrib total_cost_fence : ℕ)
variables (total_cost_shared_side : ℕ)

theorem ratio_payment_shared_side
  (h1 : length_side = 9)
  (h2 : length_back = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : neighbor_back_contrib = (length_back / 2) * cost_per_foot)
  (h6 : total_cost_fence = (2* length_side + length_back) * cost_per_foot)
  (h7 : total_cost_shared_side = length_side * cost_per_foot)
  (h8 : cole_left_total_payment = cole_payment + neighbor_back_contrib)
  (h9 : neighbor_left_contrib = cole_left_total_payment - cole_payment):
  neighbor_left_contrib / total_cost_shared_side = 1 := 
sorry

end ratio_payment_shared_side_l352_35297


namespace count_points_in_intersection_is_7_l352_35273

def isPointInSetA (x y : ℤ) : Prop :=
  (x - 3)^2 + (y - 4)^2 ≤ (5 / 2)^2

def isPointInSetB (x y : ℤ) : Prop :=
  (x - 4)^2 + (y - 5)^2 > (5 / 2)^2

def isPointInIntersection (x y : ℤ) : Prop :=
  isPointInSetA x y ∧ isPointInSetB x y

def pointsInIntersection : List (ℤ × ℤ) :=
  [(1, 5), (1, 4), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4)]

theorem count_points_in_intersection_is_7 :
  (List.length pointsInIntersection = 7)
  ∧ (∀ (p : ℤ × ℤ), p ∈ pointsInIntersection → isPointInIntersection p.fst p.snd) :=
by
  sorry

end count_points_in_intersection_is_7_l352_35273


namespace diet_cola_cost_l352_35268

theorem diet_cola_cost (T C : ℝ) 
  (h1 : T + 6 + C = 2 * T)
  (h2 : (T + 6 + C) + T = 24) : C = 2 := 
sorry

end diet_cola_cost_l352_35268
