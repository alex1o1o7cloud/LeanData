import Mathlib

namespace area_of_triangle_intercepts_l706_706210

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts_l706_706210


namespace number_of_ways_to_place_digits_l706_706627

theorem number_of_ways_to_place_digits : 
    ∃ n : ℕ, n = 720 ∧ 
             ∀ (digits : Finset ℕ) (grid : Finset (ℕ × ℕ)),
               digits = {1, 2, 3, 4, 5, 6} →
               grid = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)} →
               ∃ f : (ℕ × ℕ) → ℕ, 
                 (∀ b, b ∈ grid → f b ∈ digits) ∧
                 (∀ b₁ b₂, b₁ ≠ b₂ → f b₁ ≠ f b₂) :=
by
  let digits := {1, 2, 3, 4, 5, 6}
  let grid := {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
  existsi 720
  simp [digits, grid] 
  sorry

end number_of_ways_to_place_digits_l706_706627


namespace min_phi_symmetry_l706_706766

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem min_phi_symmetry (φ : ℝ) (hφ: φ > 0) :
  (∀ x : ℝ, f (x - φ) = f (-x - φ)) ↔ φ = π / 3 :=
by
  sorry

end min_phi_symmetry_l706_706766


namespace range_of_a_l706_706633

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (y / 4 - (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) ↔ (-3 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l706_706633


namespace gcd_of_abcd_dcba_l706_706765

theorem gcd_of_abcd_dcba : 
  ∀ (a : ℕ), 0 ≤ a ∧ a ≤ 3 → 
  gcd (2332 * a + 7112) (2332 * (a + 1) + 7112) = 2 ∧ 
  gcd (2332 * (a + 1) + 7112) (2332 * (a + 2) + 7112) = 2 ∧ 
  gcd (2332 * (a + 2) + 7112) (2332 * (a + 3) + 7112) = 2 := 
by 
  sorry

end gcd_of_abcd_dcba_l706_706765


namespace overall_cost_for_all_projects_l706_706676

-- Define the daily salaries including 10% taxes and insurance.
def daily_salary_entry_level_worker : ℕ := 100 + 10
def daily_salary_experienced_worker : ℕ := 130 + 13
def daily_salary_electrician : ℕ := 2 * 100 + 20
def daily_salary_plumber : ℕ := 250 + 25
def daily_salary_architect : ℕ := (35/10) * 100 + 35

-- Define the total cost for each project.
def project1_cost : ℕ :=
  daily_salary_entry_level_worker +
  daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project2_cost : ℕ :=
  2 * daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project3_cost : ℕ :=
  2 * daily_salary_entry_level_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

-- Define the overall cost for all three projects.
def total_cost : ℕ :=
  project1_cost + project2_cost + project3_cost

theorem overall_cost_for_all_projects :
  total_cost = 3399 :=
by
  sorry

end overall_cost_for_all_projects_l706_706676


namespace arithmetic_mean_ge_M_over_N_l706_706082

theorem arithmetic_mean_ge_M_over_N
    (a b : List ℝ) (M N : ℝ) (h1 : ∀ ⦃i j : ℕ⦄, i < j → 0 < j → j ≤ 107 → 0 < a.nth_le j sorry ∧ a.nth_le i sorry ≥ a.nth_le j sorry)
    (h2 : a.sum ≥ M)
    (h3 : ∀ ⦃i j : ℕ⦄, i < j → 0 < j → j ≤ 107 → 0 < b.nth_le j sorry ∧ b.nth_le i sorry ≤ b.nth_le j sorry)
    (h4 : b.sum ≤ M)
    (m : ℕ) (hm : 1 ≤ m) (hm_max : m ≤ 107) :
  (List.range m).sum (λ k, 1 / m * (a.nth_le k sorry / b.nth_le k sorry)) ≥ M / N := by
sorry

end arithmetic_mean_ge_M_over_N_l706_706082


namespace find_value_of_t_find_range_of_m_l706_706601

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (Real.sin(π / 4 + x))^2 - √3 * Real.cos(2 * x) - 1

-- Define the function h
def h (x t : ℝ) : ℝ := f(x + t)

-- First statement
theorem find_value_of_t (t : ℝ) (h_symm : ∀ x, h(x, t) = -h(-x - π / 6, t)) : 
t = π / 3 ∨ t = 5 * π / 6 :=
sorry

-- Second statement
theorem find_range_of_m (m : ℝ) (h_m_range_cond : ∀ x, π / 4 ≤ x → x ≤ π / 2 → abs(f x - m) < 3):
-1 < m ∧ m < 4 :=
sorry

end find_value_of_t_find_range_of_m_l706_706601


namespace max_x_squared_plus_y_squared_l706_706628

theorem max_x_squared_plus_y_squared (x y : ℝ) 
  (h : 3 * x^2 + 2 * y^2 = 2 * x) : x^2 + y^2 ≤ 4 / 9 :=
sorry

end max_x_squared_plus_y_squared_l706_706628


namespace binary_equals_octal_l706_706396

-- Define that 1001101 in binary is a specific integer
def binary_value : ℕ := 0b1001101

-- Define that 115 in octal is a specific integer
def octal_value : ℕ := 0o115

-- State the theorem we need to prove
theorem binary_equals_octal : binary_value = octal_value :=
  by sorry

end binary_equals_octal_l706_706396


namespace distinct_prime_factors_of_M_l706_706885

def num_distinct_prime_factors (n : ℕ) : ℕ := (nat.factors n).erase_dup.length

theorem distinct_prime_factors_of_M :
  ∀ (M : ℕ), log 2 (log 3 (log 5 (log 11 M))) = 7 → num_distinct_prime_factors M = 1 :=
by
  intro M h,
  sorry

end distinct_prime_factors_of_M_l706_706885


namespace distance_eq_3_sqrt_5_l706_706897

-- Defining the points
def point1 : ℝ × ℝ := (1, 1)
def point2 : ℝ × ℝ := (4, 7)

-- Distance function
def distance (p1 p2: ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_eq_3_sqrt_5 : distance point1 point2 = 3 * Real.sqrt 5 :=
  sorry

end distance_eq_3_sqrt_5_l706_706897


namespace total_meeting_arrangements_l706_706719

-- Define the leaders
inductive Leader 
| A | B | C | D | E

open Leader

-- Define the set of meetings that are allowed
def valid_meetings : List (Leader × Leader) :=
  [(A, B), (A, C), (A, D), (A, E), (B, C), (B, D), (C, D), (C, E)]

-- Condition: Each leader can participate in at most one meeting per session
def no_overlap (pairs : List (Leader × Leader)) : Prop := 
  ∀ l : Leader, 
    (pairs.filter (fun pair => pair.fst = l ∨ pair.snd = l)).length ≤ 1

-- Question: Prove that there are exactly 48 valid arrangements
theorem total_meeting_arrangements : 
  { pairs : List (Leader × Leader) // no_overlap pairs ∧ pairs.length = 4 } ->
  ∃! (arrangements : List (List (Leader × Leader))),
    arrangements.length = 2 ∧ arrangements.all no_overlap ∧
    arrangements.bij_on valid_meetings ∧ arrangements.prod = 48 :=
sorry

end total_meeting_arrangements_l706_706719


namespace find_number_l706_706407

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 99) : x = 4400 :=
sorry

end find_number_l706_706407


namespace arithmetic_sequence_value_l706_706637

axiom arithmetic_sequence_sum (n : ℕ) (c : ℕ) : ℕ → Prop :=
S n = 2 * n^2 - n + c

axiom arithmetic_term (n : ℕ) (c : ℕ) : ℕ :=
a n = (S n - S (n - 1))

theorem arithmetic_sequence_value (c : ℕ) : (∃ S : ℕ → ℕ, (∀ n, S n = 2 * n^2 - n + c) ∧ (∀ n, a n = S n - S (n - 1))) → a (c + 5) = 17 :=
by
  sorry

end arithmetic_sequence_value_l706_706637


namespace unique_solution_for_a_l706_706139

theorem unique_solution_for_a (a : ℝ) :
  (∃! (x y : ℝ), 
    (x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a) ∧
    (-3 ≤ x + 2 * y ∧ x + 2 * y ≤ 7) ∧
    (-9 ≤ 3 * x - 4 * y ∧ 3 * x - 4 * y ≤ 1)) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
sorry

end unique_solution_for_a_l706_706139


namespace worth_of_each_gift_is_4_l706_706993

noncomputable def worth_of_each_gift
  (workers_per_block : ℕ)
  (total_blocks : ℕ)
  (total_amount : ℝ) : ℝ :=
  total_amount / (workers_per_block * total_blocks)

theorem worth_of_each_gift_is_4 (workers_per_block total_blocks : ℕ) (total_amount : ℝ)
  (h1 : workers_per_block = 100)
  (h2 : total_blocks = 10)
  (h3 : total_amount = 4000) :
  worth_of_each_gift workers_per_block total_blocks total_amount = 4 :=
by
  sorry

end worth_of_each_gift_is_4_l706_706993


namespace min_value_x_y_l706_706168

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_value_x_y_l706_706168


namespace parallel_planes_of_perp_line_l706_706250

-- Definitions of distinct lines and planes
variables {m n l : Line} {α β : Plane}

-- Conditions
axiom distinct_lines : m ≠ n ∧ n ≠ l ∧ m ≠ l
axiom distinct_planes : α ≠ β

-- Geometric properties
axiom perp_line_to_planes : m ⟂ α ∧ m ⟂ β

-- Theorem statement
theorem parallel_planes_of_perp_line (h_perp : m ⟂ α ∧ m ⟂ β) : α ∥ β := 
  sorry

end parallel_planes_of_perp_line_l706_706250


namespace probability_of_drawing_one_black_ball_l706_706229

theorem probability_of_drawing_one_black_ball 
  (white_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (drawn_balls : ℕ)
  (h_w : white_balls = 3)
  (h_b : black_balls = 2)
  (h_t : total_balls = white_balls + black_balls)
  (h_d : drawn_balls = 2) :
  (Combination (white_balls + black_balls) drawn_balls) 
  ≠ 0 → 
  (2 * Combination white_balls 1 * Combination black_balls 1 / 
  Combination (white_balls + black_balls) drawn_balls : ℚ) = 3 / 5 := by
    sorry

end probability_of_drawing_one_black_ball_l706_706229


namespace count_valid_words_l706_706492

def total_words (n : ℕ) : ℕ := 25 ^ n

def words_with_no_A (n : ℕ) : ℕ := 24 ^ n

def words_with_one_A (n : ℕ) : ℕ := n * 24 ^ (n - 1)

def words_with_less_than_two_As : ℕ :=
  (words_with_no_A 2) + (2 * 24) +
  (words_with_no_A 3) + (3 * 24 ^ 2) +
  (words_with_no_A 4) + (4 * 24 ^ 3) +
  (words_with_no_A 5) + (5 * 24 ^ 4)

def valid_words : ℕ :=
  (total_words 1 + total_words 2 + total_words 3 + total_words 4 + total_words 5) -
  words_with_less_than_two_As

theorem count_valid_words : valid_words = sorry :=
by sorry

end count_valid_words_l706_706492


namespace num_solutions_to_equation_l706_706622

theorem num_solutions_to_equation : 
  let numerator_zeros := (finset.range 50).map (λ n, n + 1)
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49}
  numerator_zeros.card - perfect_squares.card = 43 := 
by
  sorry

end num_solutions_to_equation_l706_706622


namespace sweets_remainder_l706_706553

theorem sweets_remainder (m : ℕ) (h : m % 7 = 6) : (4 * m) % 7 = 3 :=
by
  sorry

end sweets_remainder_l706_706553


namespace sin_of_alpha_l706_706569

theorem sin_of_alpha 
  (α : ℝ) 
  (h : Real.cos (α - Real.pi / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := 
by 
  sorry

end sin_of_alpha_l706_706569


namespace pound_of_rice_cost_l706_706652

theorem pound_of_rice_cost 
(E R K : ℕ) (h1: E = R) (h2: K = 4 * (E / 12)) (h3: K = 11) : R = 33 := by
  sorry

end pound_of_rice_cost_l706_706652


namespace radius_correct_l706_706316

noncomputable def radius_of_circle (chord_length tang_secant_segment : ℝ) : ℝ :=
  let r := 6.25
  r

theorem radius_correct
  (chord_length : ℝ)
  (tangent_secant_segment : ℝ)
  (parallel_secant_internal_segment : ℝ)
  : chord_length = 10 ∧ parallel_secant_internal_segment = 12 → radius_of_circle chord_length parallel_secant_internal_segment = 6.25 :=
by
  intros h
  sorry

end radius_correct_l706_706316


namespace parabola_through_point_has_specific_a_l706_706634

theorem parabola_through_point_has_specific_a :
  (∃ a : ℝ, ∀ x : ℝ, y : ℝ, (y = a * x^2 - 2 * x + 3) → (x = 1) → (y = 2) → a = 1) :=
sorry

end parabola_through_point_has_specific_a_l706_706634


namespace problem_statement_l706_706097

noncomputable def geometric_sum : ℝ :=
  ∑' i : ℕ, 2⁻¹^i

theorem problem_statement :
  (∑' j : ℕ, ∑' k : ℕ, ite (even k) (2 ^ (-(4 * k + 2 * j + (k + j) ^ 2))) 0) = 2 :=
by
  -- proof to be completed
  sorry

end problem_statement_l706_706097


namespace distances_sum_tangents_sum_l706_706419

variable (n : ℕ) [Fact (0 < n)]
variable (A : Point) (A_i : Fin (2 * n + 1) → Point) (r : ℝ)

-- Assume A_1, A_2, ..., A_{2n+1} are the vertices of a regular (2n+1)-gon
-- Assume distances d_i and tangent lengths l_i
def d_i (i : Fin (2 * n + 1)) : ℝ := dist A (A_i i)
def l_i (i : Fin (2 * n + 1)) : ℝ := tangentLength A (A_i i) r

-- Hypothesis: A is on the arc A_1A_{2n+1} of the circumscribed circle S
variable (hArc : on_arc A (A_i 0) (A_i (2 * n)))

-- Part (a)
theorem distances_sum (n : ℕ) [Fact (0 < n)] (A : Point) (A_i : Fin (2 * n + 1) → Point)
    (hReg : is_regular_polygon (A_i : Fintype.shape (2 * n + 1)) )
    (hArc : on_arc A (A_i 0) (A_i (2 * n))):
    (∑ k in Finset.range (n + 1), d_i A (A_i : Fin (2 * n + 1)) (Fin.ofNat (2 * k + 1))) =
    (∑ k in Finset.range (n), d_i A (A_i : Fin (2 * n + 1)) (Fin.ofNat (2 * (k + 1)))) :=
by sorry

-- Part (b)
theorem tangents_sum (n : ℕ) [Fact (0 < n)] (A : Point) (A_i : Fin (2 * n + 1) → Point)
    (r : ℝ) (hReg : is_regular_polygon (A_i : Fintype.shape (2 * n + 1)) )
    (hArc : on_arc A (A_i 0) (A_i (2 * n))) :
    (∑ k in Finset.range (n + 1), l_i A (A_i : Fin (2 * n + 1)) r (Fin.ofNat (2 * k + 1))) =
    (∑ k in Finset.range (n), l_i A (A_i : Fin (2 * n + 1)) r (Fin.ofNat (2 * (k + 1)))) :=
by sorry

end distances_sum_tangents_sum_l706_706419


namespace inequality_solution_l706_706739

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end inequality_solution_l706_706739


namespace candidate_percentage_is_correct_l706_706657

-- Define total number of votes
def total_votes : ℕ := 560000
-- Define the percentage of invalid votes
def invalid_percentage : ℝ := 0.15
-- Define the number of votes the candidate received
def candidate_votes : ℕ := 333200

-- Calculate the number of valid votes
def valid_votes : ℕ := (total_votes : ℝ) * (1 - invalid_percentage) |> Nat.floor

-- Calculate the percentage of valid votes the candidate received
def percentage_valid_votes : ℝ := (candidate_votes : ℝ) / (valid_votes : ℝ) * 100

-- State the theorem: The candidate got 70% of the total valid votes
theorem candidate_percentage_is_correct : percentage_valid_votes = 70 := 
by 
  -- Theorem proof goes here
  sorry

end candidate_percentage_is_correct_l706_706657


namespace largest_w_l706_706979

variable {x y z w : ℝ}

def x_value (x y z w : ℝ) := 
  x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4

theorem largest_w (h : x_value x y z w) : 
  max x (max y (max z w)) = w := 
sorry

end largest_w_l706_706979


namespace EP_Vorontsova_Dashkova_birth_death_years_l706_706762

theorem EP_Vorontsova_Dashkova_birth_death_years :
  ∃ (birth_year death_year : ℕ),
    (death_year - birth_year = 66) ∧ 
    (∃ (x : ℕ), (x + 46 + x = 66) ∧ ((death_year - 1800) = 10) ∧ (1744 = 1800 - (x + 46))) :=
by
  existsi 1744, 1810
  split
  sorry
  existsi 10
  split
  sorry
  split
  sorry
  sorry

end EP_Vorontsova_Dashkova_birth_death_years_l706_706762


namespace proof_completion_l706_706091

namespace MathProof

def p : ℕ := 10 * 7

def r : ℕ := p - 3

def q : ℚ := (3 / 5) * r

theorem proof_completion : q = 40.2 := by
  sorry

end MathProof

end proof_completion_l706_706091


namespace sqrt_of_product_of_powers_of_two_l706_706806

theorem sqrt_of_product_of_powers_of_two :
  sqrt (2^4 * 2^4 * 2^4) = 64 :=
by
  -- Inserting 'sorry' to skip the proof, as required.
  sorry

end sqrt_of_product_of_powers_of_two_l706_706806


namespace goldfish_count_equal_in_6_months_l706_706476

def initial_goldfish_brent : ℕ := 3
def initial_goldfish_gretel : ℕ := 243

def goldfish_brent (n : ℕ) : ℕ := initial_goldfish_brent * 4^n
def goldfish_gretel (n : ℕ) : ℕ := initial_goldfish_gretel * 3^n

theorem goldfish_count_equal_in_6_months : 
  (∃ n : ℕ, goldfish_brent n = goldfish_gretel n) ↔ n = 6 :=
by
  sorry

end goldfish_count_equal_in_6_months_l706_706476


namespace length_PQ_in_right_triangle_l706_706899

-- Definitions based on the given conditions
def is_right_triangle (P Q R : Type) (angle_PQR : ℝ) : Prop :=
  angle_PQR = 45

def length_PR : ℝ := 10

-- Theorem statement
theorem length_PQ_in_right_triangle (P Q R : Type) (h : is_right_triangle P Q R 45) : 
  PQ.length = 5 * real.sqrt 2 :=
sorry

end length_PQ_in_right_triangle_l706_706899


namespace division_result_l706_706702

-- Define n in terms of the given condition
def n : ℕ := 9^2023

theorem division_result : n / 3 = 3^4045 :=
by
  sorry

end division_result_l706_706702


namespace sum_of_cubes_l706_706585

noncomputable def primitive_root := Complex.exp (2 * Real.pi * Complex.I / 3)

lemma cube_roots_of_unity (x : ℂ) (h : x ^ 3 = 1) : x = 1 ∨ x = primitive_root ∨ x = primitive_root ^ 2 :=
begin
  sorry,
end

theorem sum_of_cubes (a b c : ℂ) (n m : ℕ)
  (h_common : ∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n+2) + b * x^(3*m+1) + c = 0) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 0 :=
begin
  obtain ⟨x, hx_unit, hx_poly⟩ := h_common,
  cases (cube_roots_of_unity x hx_unit) with h1 hq;
  {
    by_cases hx : x = 1;
    {
      { sorry },
      {
        { sorry },
        { sorry }
      }
    }
  }
end

end sum_of_cubes_l706_706585


namespace minimum_value_f_on_neg_ab_l706_706176

theorem minimum_value_f_on_neg_ab
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : a < b)
  (h2 : b < 0)
  (odd_f : ∀ x : ℝ, f (-x) = -f (x))
  (decreasing_f : ∀ x y : ℝ, 0 < x ∧ x < y → f y < f x)
  (range_ab : ∀ y : ℝ, a ≤ y ∧ y ≤ b → -3 ≤ f y ∧ f y ≤ 4) :
  ∀ x : ℝ, -b ≤ x ∧ x ≤ -a → -4 ≤ f x ∧ f x ≤ 3 := 
sorry

end minimum_value_f_on_neg_ab_l706_706176


namespace solve_for_n_l706_706736

theorem solve_for_n (n x y : ℤ) (h : n * (x + y) + 17 = n * (-x + y) - 21) (hx : x = 1) : n = -19 :=
by
  sorry

end solve_for_n_l706_706736


namespace intersecting_line_parabola_l706_706839

-- Definitions for the parabola and its properties
def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 8 * x

-- Definition of the focus of the given parabola
def focus_F : ℝ × ℝ := (2, 0)

-- The line equation with an inclination angle of 45 degrees passing through the focus F
def line_l (x y : ℝ) : Prop := y = x - 2

-- Points A and B are intersections of the line and the parabola
def point_A (x y : ℝ) : Prop := line_l x y ∧ parabola x y
def point_B (x y : ℝ) : Prop := line_l x y ∧ parabola x y

-- Distance from focus F to points A and B
def distance_FA (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + y^2)
def distance_FB (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + y^2)

-- The statement for the problem
theorem intersecting_line_parabola :
  ∃ (x1 y1 x2 y2 : ℝ), point_A x1 y1 ∧ point_B x2 y2 ∧ (distance_FA x1 y1) * (distance_FB x2 y2) = 32 :=
sorry

end intersecting_line_parabola_l706_706839


namespace number_of_ways_to_choose_one_person_l706_706837

-- Definitions for the conditions
def people_using_first_method : ℕ := 3
def people_using_second_method : ℕ := 5

-- Definition of the total number of ways to choose one person
def total_ways_to_choose_one_person : ℕ :=
  people_using_first_method + people_using_second_method

-- Statement of the theorem to be proved
theorem number_of_ways_to_choose_one_person :
  total_ways_to_choose_one_person = 8 :=
by 
  sorry

end number_of_ways_to_choose_one_person_l706_706837


namespace cakes_baked_yesterday_l706_706846

noncomputable def BakedToday : ℕ := 5
noncomputable def SoldDinner : ℕ := 6
noncomputable def Left : ℕ := 2

theorem cakes_baked_yesterday (CakesBakedYesterday : ℕ) : 
  BakedToday + CakesBakedYesterday - SoldDinner = Left → CakesBakedYesterday = 3 := 
by 
  intro h 
  sorry

end cakes_baked_yesterday_l706_706846


namespace odd_perfect_has_three_distinct_prime_divisors_l706_706004

-- Defining the concept of a perfect number.
def is_perfect (n : ℕ) : Prop :=
  ∑ d in (finset.divisors n), d = 2 * n

-- Defining the concept of an odd number.
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- Defining the concept of having at least 3 distinct prime divisors.
def has_at_least_three_distinct_prime_divisors (n : ℕ) : Prop :=
  (finset.filter nat.prime (finset.divisors n)).card ≥ 3

-- The theorem statement.
theorem odd_perfect_has_three_distinct_prime_divisors (n : ℕ) (h1 : is_odd n) (h2 : is_perfect n) : 
  has_at_least_three_distinct_prime_divisors n := 
sorry

end odd_perfect_has_three_distinct_prime_divisors_l706_706004


namespace ratio_of_volumes_l706_706786

open Real

theorem ratio_of_volumes
  (W_a : ℝ) (W_b : ℝ) (V : ℝ) (W : ℝ)
  (V_a V_b : ℝ)
  (h1 : V_a + V_b = 4)
  (h2 : 950 * V_a + 850 * V_b = 3640)
  (W_a_def : W_a = 950)
  (W_b_def : W_b = 850)
  (V_def : V = 4)
  (W_def : W = 3640) :
  (V_a / V_b = 1.5) :=
by
  compute_ratio sorry

#check ratio_of_volumes

end ratio_of_volumes_l706_706786


namespace no_good_number_exists_l706_706043

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

theorem no_good_number_exists : ¬ ∃ n : ℕ, is_good n :=
by sorry

end no_good_number_exists_l706_706043


namespace selection_ways_l706_706744

def team_A_size (m : ℕ) : ℕ := 2 * m
def team_B_size (m : ℕ) : ℕ := 3 * m
def drawn_A (m : ℕ) : ℕ := 14 - m
def drawn_B (m : ℕ) : ℕ := 5 * m - 11

theorem selection_ways :
  ∀ (m : ℕ), 
  (m = 5) →
  combinatorics.choose (team_A_size m) (drawn_A m) *
  combinatorics.choose (team_B_size m) (drawn_B m) = 150 := 
by
  intros m hm
  sorry

end selection_ways_l706_706744


namespace ordered_pairs_count_l706_706620

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a * b + 90 = 25 * Nat.lcm a b + 15 * Nat.gcd a b)

def count_valid_pairs : ℕ :=
  (Finset.univ.filter (λ p : ℕ × ℕ, is_valid_pair p.1 p.2)).card

theorem ordered_pairs_count : count_valid_pairs = 2 := 
  sorry

end ordered_pairs_count_l706_706620


namespace sum_x_coordinates_l706_706758

-- Define the equations of the line segments
def segment1 (x : ℝ) := 2 * x + 6
def segment2 (x : ℝ) := -0.5 * x - 1.5
def segment3 (x : ℝ) := 2 * x + 1
def segment4 (x : ℝ) := -0.5 * x + 3.5
def segment5 (x : ℝ) := 2 * x - 4

-- Definition of the problem
theorem sum_x_coordinates (h1 : segment1 (-5) = -4 ∧ segment1 (-3) = 0)
    (h2 : segment2 (-3) = 0 ∧ segment2 (-1) = -1)
    (h3 : segment3 (-1) = -1 ∧ segment3 (1) = 3)
    (h4 : segment4 (1) = 3 ∧ segment4 (3) = 2)
    (h5 : segment5 (3) = 2 ∧ segment5 (5) = 6)
    (hx1 : ∃ x1, segment3 x1 = 2.4 ∧ -1 ≤ x1 ∧ x1 ≤ 1)
    (hx2 : ∃ x2, segment4 x2 = 2.4 ∧ 1 ≤ x2 ∧ x2 ≤ 3)
    (hx3 : ∃ x3, segment5 x3 = 2.4 ∧ 3 ≤ x3 ∧ x3 ≤ 5) :
    (∃ (x1 x2 x3 : ℝ), segment3 x1 = 2.4 ∧ segment4 x2 = 2.4 ∧ segment5 x3 = 2.4 ∧ x1 = 0.7 ∧ x2 = 2.2 ∧ x3 = 3.2 ∧ x1 + x2 + x3 = 6.1) :=
sorry

end sum_x_coordinates_l706_706758


namespace biff_speed_l706_706087

theorem biff_speed :
  ∀ B K D d, (K = 51) → (D = 500) → (d = 10) → (K * (D + d) / (K * D / B) = D) → (B = 50) :=
by
  intros B K D d hK hD hd h_equation
  rw [hK, hD, hd] at h_equation
  have h_time : (D + d) / K = 10 := by linarith
  have h_B : B = D / ((D + d) / K) := by linarith [h_equation]
  rw [h_time] at h_B
  rw [hD, hd] at *

  -- Simplify to conclude that B = 50
  linarith

end biff_speed_l706_706087


namespace scale_fragments_l706_706363

-- definitions for adults and children heights
def height_adults : Fin 100 → ℕ 
def height_children : Fin 100 → ℕ 

-- the main theorem statement
theorem scale_fragments (h₁ : ∀ i, height_children i < height_adults i) :
  ∃ (k : Fin 100 → ℕ), ∀ (i j : Fin 100), 
  k i * height_children i < k i * height_adults i ∧ 
  (i ≠ j → ∀ (a: ℕ), k i * height_children i < a → k i * height_adults i < a) :=
sorry

end scale_fragments_l706_706363


namespace sticker_count_l706_706360

theorem sticker_count (stickers_per_page : ℕ) (pages: ℕ) (h_stickers_per_page : stickers_per_page = 10) (h_pages : pages = 22) : stickers_per_page * pages = 220 :=
by 
  rw [h_stickers_per_page, h_pages]
  norm_num
  sorry

end sticker_count_l706_706360


namespace no_such_fraction_exists_l706_706503

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l706_706503


namespace possible_days_l706_706473

namespace AnyaVanyaProblem

-- Conditions
def AnyaLiesOn (d : String) : Prop := d = "Tuesday" ∨ d = "Wednesday" ∨ d = "Thursday"
def AnyaTellsTruthOn (d : String) : Prop := ¬AnyaLiesOn d

def VanyaLiesOn (d : String) : Prop := d = "Thursday" ∨ d = "Friday" ∨ d = "Saturday"
def VanyaTellsTruthOn (d : String) : Prop := ¬VanyaLiesOn d

-- Statements
def AnyaStatement (d : String) : Prop := d = "Friday"
def VanyaStatement (d : String) : Prop := d = "Tuesday"

-- Proof problem
theorem possible_days (d : String) : 
  (AnyaTellsTruthOn d ↔ AnyaStatement d) ∧ (VanyaTellsTruthOn d ↔ VanyaStatement d)
  → d = "Tuesday" ∨ d = "Thursday" ∨ d = "Friday" := 
sorry

end AnyaVanyaProblem

end possible_days_l706_706473


namespace sum_max_min_values_of_f_l706_706106

noncomputable def f (x : ℝ) : ℝ := 1 - (Real.sin x) / (x^4 + x^2 + 1)

theorem sum_max_min_values_of_f : 
  (let max_f := RealSup (set.range f) in
  let min_f := RealInf (set.range f) in
  max_f + min_f = 2) :=
begin
  sorry
end

end sum_max_min_values_of_f_l706_706106


namespace convert_decimal_to_fraction_l706_706398

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l706_706398


namespace currency_exchange_rate_l706_706032

theorem currency_exchange_rate (b g x : ℕ) (h1 : 1 * b * g = b * g) (h2 : 1 = 1) :
  (b + g) ^ 2 + 1 = b * g * x → x = 5 :=
sorry

end currency_exchange_rate_l706_706032


namespace length_of_CD_l706_706785

theorem length_of_CD {L : ℝ} (h₁ : 16 * Real.pi * L + (256 / 3) * Real.pi = 432 * Real.pi) :
  L = (50 / 3) :=
by
  sorry

end length_of_CD_l706_706785


namespace exists_C_a_n1_minus_a_n_l706_706271

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| 2 => 8
| (n+1) => a (n - 1) + (4 / n) * a n

theorem exists_C (C : ℕ) (hC : C = 2) : ∃ C > 0, ∀ n > 0, a n ≤ C * n^2 := by
  use 2
  sorry

theorem a_n1_minus_a_n (n : ℕ) (h : n > 0) : a (n + 1) - a n ≤ 4 * n + 3 := by
  sorry

end exists_C_a_n1_minus_a_n_l706_706271


namespace number_of_odd_rank_subsets_l706_706829

theorem number_of_odd_rank_subsets (n_cards : ℕ) (n_ranks : ℕ) (n_suits : ℕ) 
  (hn : n_cards = 2014 * 4)
  (hr : n_ranks = 2014)
  (hs : n_suits = 4) :
  (∑ k in (Finset.range (n_ranks + 1)), if k % 2 = 1 then (Finset.choose n_ranks k) * 15^k else 0) 
    = (1/2 * ((16:ℕ)^n_ranks - (14:ℕ)^n_ranks)) :=
by
  sorry

end number_of_odd_rank_subsets_l706_706829


namespace book_total_pages_l706_706009

theorem book_total_pages (n : ℕ) (h1 : 5 * n / 8 - 3 * n / 7 = 33) : n = n :=
by 
  -- We skip the proof as instructed
  sorry

end book_total_pages_l706_706009


namespace range_f_in_0_pi_over_2_increasing_intervals_f_in_0_pi_l706_706600

def f (x : ℝ) := 2 * sin x * cos x - 2 * (sin x)^2 + 1

theorem range_f_in_0_pi_over_2 : set.Icc (-1) (real.sqrt 2) = set.image f (set.Icc 0 (real.pi / 2)) := 
sorry

theorem increasing_intervals_f_in_0_pi : 
  set.Icc 0 (real.pi / 8) ∪ set.Icc (5 * real.pi / 8) real.pi ⊆ {x : ℝ | 0 ≤ x ∧ x ≤ real.pi ∧ is_strictly_increasing_on f (set.Icc 0 real.pi)} := 
sorry

end range_f_in_0_pi_over_2_increasing_intervals_f_in_0_pi_l706_706600


namespace eventually_periodic_sequence_l706_706016

noncomputable def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ m ≥ N, a m = a (m + k)

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_condition : ∀ n, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
  eventually_periodic a :=
sorry

end eventually_periodic_sequence_l706_706016


namespace min_value_of_a2_plus_b2_l706_706630

theorem min_value_of_a2_plus_b2 (a b : ℝ) (h : (∑ k in Finset.range 7, (Nat.choose 6 k) * (a^(6 - k)) * (b^k) * (a^2)^(6 - k) * (b^(-1))^k) = 20) : a^2 + b^2 = 2 :=
sorry

end min_value_of_a2_plus_b2_l706_706630


namespace monotonic_increasing_interval_l706_706189

def f (x b : ℝ) : ℝ := x + b / x

def f_prime (x b : ℝ) : ℝ := 1 - b / (x^2)

theorem monotonic_increasing_interval (b : ℝ) (h_b : 1 < b ∧ b < 4) :
  ∀ x : ℝ, x ∈ (Set.Ioi 2) → f_prime x b > 0 :=
by
  sorry

end monotonic_increasing_interval_l706_706189


namespace evaluate_expression_l706_706259

-- Define the greatest power of 2 and 3 that are factors of 360
def a : ℕ := 3 -- 2^3 is the greatest power of 2 that is a factor of 360
def b : ℕ := 2 -- 3^2 is the greatest power of 3 that is a factor of 360

theorem evaluate_expression : (1 / 4)^(b - a) = 4 := 
by 
  have h1 : a = 3 := rfl
  have h2 : b = 2 := rfl
  rw [h1, h2]
  simp
  sorry

end evaluate_expression_l706_706259


namespace tire_circumference_l706_706014

/-- 
Given:
1. The tire rotates at 400 revolutions per minute.
2. The car is traveling at a speed of 168 km/h.

Prove that the circumference of the tire is 7 meters.
-/
theorem tire_circumference (rpm : ℕ) (speed_km_h : ℕ) (C : ℕ) 
  (h1 : rpm = 400) 
  (h2 : speed_km_h = 168)
  (h3 : C = 7) : 
  C = (speed_km_h * 1000 / 60) / rpm :=
by
  rw [h1, h2]
  exact h3

end tire_circumference_l706_706014


namespace question_1_question_2_l706_706914

-- Define the quadratic function f
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Statement for the first question
theorem question_1 (c : ℝ) : 
  ∀ b ∈ ℝ, b < 0 → 
  (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f b c x ∈ Icc (0 : ℝ) (1 : ℝ)) → 
  b = -2 ∧ c = 1 :=
sorry

-- Define the function g
noncomputable def g (b c x : ℝ) : ℝ := f b c x / x

-- Statement for the second question
theorem question_2 : 
  ∀ x ∈ Icc (3 : ℝ) (5 : ℝ), ∀ y ∈ ℝ, b = -2 → g b c x > y → y < 3 / 2 :=
sorry

end question_1_question_2_l706_706914


namespace lcm_24_30_40_l706_706900

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_24_30_40 : lcm (lcm 24 30) 40 = 120 := by
  -- According to the definition of lcm and verifying with the given conditions
  sorry

end lcm_24_30_40_l706_706900


namespace number_of_terms_in_sequence_l706_706102

theorem number_of_terms_in_sequence : 
  let a := 2.5
  let d := 5
  let l := 62.5
  let n := (l - a) / d + 1
  n = 13 := 
by 
  let a := 2.5
  let d := 5
  let l := 62.5
  have h : n = (l - a) / d + 1 := rfl
  sorry

end number_of_terms_in_sequence_l706_706102


namespace avg_annual_growth_rate_optimal_selling_price_l706_706539

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end avg_annual_growth_rate_optimal_selling_price_l706_706539


namespace average_GPA_school_l706_706750

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l706_706750


namespace factorization_example_l706_706468

theorem factorization_example (a b : ℕ) : (a - 2*b)^2 = a^2 - 4*a*b + 4*b^2 := 
by sorry

end factorization_example_l706_706468


namespace sqrt_identity_l706_706625

theorem sqrt_identity (x : ℝ) (hx : x = Real.sqrt 5 - 3) : Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 :=
by
  sorry

end sqrt_identity_l706_706625


namespace yz_length_l706_706020

noncomputable def find_YZ_length (BC AB XY : ℝ) (AB_val XY_val : ℝ) : ℝ :=
  (BC * XY_val) / AB_val

theorem yz_length (BC AB XY : ℝ) (YZ_expected : ℝ) 
  (h_similar : BC / AB = YZ_expected / XY) 
  (h_BC : BC = 6)
  (h_AB : AB = 7)
  (h_XY : XY = 2.5) : 
  find_YZ_length BC AB XY h_AB h_XY = YZ_expected := 
by
  simp [find_YZ_length]
  norm_num
  exact eq_of_sub_eq_zero (show find_YZ_length BC AB XY h_AB h_XY - YZ_expected = 0, 
    by {simp [find_YZ_length], norm_num})

example : yz_length 6 7 2.5 (15 / 7) sorry sorry sorry := by sorry

end yz_length_l706_706020


namespace log_condition_sufficient_log_condition_not_necessary_l706_706534

theorem log_condition_sufficient (a b : ℝ) (h1 : 0 < b) (h2 : a > 0):
  (log 2 a > log 2 b) → (2^a > 2^b) :=
by sorry

theorem log_condition_not_necessary (a b : ℝ) (h1 : 2^a > 2^b):
  ¬(log 2 a > log 2 b ↔ 0 < b ∧ 0 < a) :=
by sorry

end log_condition_sufficient_log_condition_not_necessary_l706_706534


namespace solution_set_f_over_x_lt_0_l706_706771

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_over_x_lt_0 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x1 x2, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) →
  (f 4 = 0) →
  { x | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
by
  intros _ _ _
  sorry

end solution_set_f_over_x_lt_0_l706_706771


namespace midpoint_quadrilateral_area_l706_706775

theorem midpoint_quadrilateral_area (R : ℝ) (hR : 0 < R) :
  ∃ (Q : ℝ), Q = R / 4 :=
by
  sorry

end midpoint_quadrilateral_area_l706_706775


namespace animals_consuming_hay_l706_706031

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end animals_consuming_hay_l706_706031


namespace f_neg_add_eq_zero_l706_706217

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem f_neg_add_eq_zero (a : ℝ) : f(a) + f(-a) = 0 :=
by sorry

end f_neg_add_eq_zero_l706_706217


namespace proposition_D_incorrect_l706_706145

open Plane_theory

variable (m n : Line) (α β γ : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom parallel_alpha_beta : α ∥ β
axiom parallel_m_alpha : m ∥ α

-- Proposition D to show its incorrectness
theorem proposition_D_incorrect : ¬(m ∥ β) := 
sorry

end proposition_D_incorrect_l706_706145


namespace problem_lean_statement_l706_706245

def integer_exponent_terms (n : ℕ) (f : ℕ → ℕ) : ℕ := 
  ∑ r in finset.range (n + 1), if f r then 1 else 0

theorem problem_lean_statement :
  integer_exponent_terms 20 (λ r, (20 - (4 * r / 3)) % 1 = 0) = 7 :=
  sorry

end problem_lean_statement_l706_706245


namespace school_avg_GPA_l706_706747

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l706_706747


namespace cube_section_volume_ratio_l706_706080

theorem cube_section_volume_ratio (A C K : Point) (EK KF : Real)
  (H_cube : is_cube ABCDEFGH)
  (H_plane : plane_section_through AKC)
  (H_point_on_edge : point_on_edge K EF)
  (H_volume_ratio : volume_ratio (cube_divided_by_plane ABCDEFGH AKC) 3 1) :
  EK / KF = Real.sqrt 3 :=
sorry

end cube_section_volume_ratio_l706_706080


namespace sum_qs_10_l706_706693

noncomputable def S' : Set (Fin 10 → Fin 3) := {s | True}

noncomputable def q_s (s : Fin 10 → Fin 3) : Polynomial ℤ := 
  Polynomial.ofFinFun (λ n => s n)

def q (x : ℕ) : ℤ := ∑ s in S', q_s s x

theorem sum_qs_10 : q 10 = 19683 := 
by 
  sorry

end sum_qs_10_l706_706693


namespace inequality_to_prove_l706_706579

variable {n : ℕ}  -- Number of terms in the sequence

-- Parameter assumptions:
axiom positive_integer_condition : (0.2 * n : ℝ) ∈ ℕ ∧ 0 < n

noncomputable def seq (i : ℕ) : ℝ := sorry  -- Sequence definition which we assume exists from the conditions

axiom positive_elements_condition : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 0 < seq i

axiom sequence_relation :
  (∑ i in Finset.range n, (1 / (i + 1) : ℝ) * (seq (i + 1))^2) +
  2 * (∑ j in Finset.range (n - 1), ∑ k in Finset.Ico (j + 1) n, (1 / (k + 1) : ℝ) * (seq (j + 1)) * (seq (k + 1))) = 
  1

theorem inequality_to_prove :
  (∑ i in Finset.range n, (i + 1) * seq (i + 1)) ≤ 
  Real.sqrt ((4 / 3 : ℝ) * (n : ℝ)^3 - (1 / 3 : ℝ) * (n : ℝ)) :=
sorry

end inequality_to_prove_l706_706579


namespace no_such_fraction_exists_l706_706505

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l706_706505


namespace find_salary_for_january_l706_706315

-- Definitions based on problem conditions
variables (J F M A May : ℝ)
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (hMay : May = 6500)

-- Lean statement
theorem find_salary_for_january : J = 5700 :=
by {
  sorry
}

end find_salary_for_january_l706_706315


namespace ceiling_eval_l706_706547

theorem ceiling_eval : Int.ceil (-3.7 + 1.2) = -2 := by
  sorry

end ceiling_eval_l706_706547


namespace base_b_of_256_has_4_digits_l706_706423

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l706_706423


namespace largest_power_of_2_divides_n_l706_706532

def n : ℤ := 15^4 - 7^4 - 8

theorem largest_power_of_2_divides_n : ∃ k, n = 2^k * m ∧ ¬ ∃ l, n = 2^(k + 1) * l :=
begin
  use 3,
  split,
  sorry,  -- Proof showing n is divisible by 2^3
  intro h,
  cases h with l hl,
  sorry  -- Proof showing n is not divisible by 2^4 or higher
end

end largest_power_of_2_divides_n_l706_706532


namespace count_M_intersect_N_l706_706919

def M (x y : ℝ) : Prop := (Real.tan (π * y) = 0) ∧ (Real.sin (π * x))^2 = 0
def N (x y : ℝ) : Prop := (x^2 + y^2 <= 2)

theorem count_M_intersect_N : set.finite {p : ℝ × ℝ | M p.1 p.2 ∧ N p.1 p.2} ∧ set.card {p : ℝ × ℝ | M p.1 p.2 ∧ N p.1 p.2} = 9 :=
by
  sorry

end count_M_intersect_N_l706_706919


namespace triangle_sides_inequality_l706_706275

-- Define the sides of a triangle and their sum
variables {a b c : ℝ}

-- Define the condition that they are sides of a triangle.
def triangle_sides (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition that their sum is 1
axiom sum_of_sides (a b c : ℝ) (h : triangle_sides a b c) : a + b + c = 1

-- Define the proof theorem for the inequality
theorem triangle_sides_inequality (h : triangle_sides a b c) (h_sum : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_sides_inequality_l706_706275


namespace proj_cycle_identity_l706_706290

variable {A B C D M₁ M₂ M₃ M₄ M₅ M₆ M₇ M₈ M₉ M₁₀ M₁₁ M₁₂ M₁₃ : Point}
variable {AB BC CD DA : Line}

-- Define the projections
def proj (p : Point) (l : Line) (from : Point) : Point := sorry

-- Conditions
axiom M1_on_AB : on_line M₁ AB
axiom M2_is_proj_M1 : M₂ = proj M₁ BC D
axiom M3_is_proj_M2 : M₃ = proj M₂ CD A
axiom M4_is_proj_M3 : M₄ = proj M₃ DA B
axiom M5_is_proj_M4 : M₅ = proj M₄ AB C
axiom M6_is_proj_M5 : M₆ = proj M₅ BC D
axiom M7_is_proj_M6 : M₇ = proj M₆ CD A
axiom M8_is_proj_M7 : M₈ = proj M₇ DA B
axiom M9_is_proj_M8 : M₉ = proj M₈ AB C
axiom M10_is_proj_M9 : M₁₀ = proj M₉ BC D
axiom M11_is_proj_M10 : M₁₁ = proj M₁₀ CD A
axiom M12_is_proj_M11 : M₁₂ = proj M₁₁ DA B
axiom M13_is_proj_M12 : M₁₃ = proj M₁₂ AB C

-- The theorem to prove
theorem proj_cycle_identity : M₁₃ = M₁ :=
sorry

end proj_cycle_identity_l706_706290


namespace jane_quadratic_coefficients_l706_706687

theorem jane_quadratic_coefficients :
  (∃ b c : ℝ, ∀ x : ℝ, (|x-4| = 3 → (x = 1 ∨ x = 7)) ∧ (x^2 + b * x + c = 0 → (x = 1 ∨ x =7)) ∧ (b = -8 ∧ c = 7)) :=
by
  use [-8, 7]
  intros x h1 h2
  sorry

end jane_quadratic_coefficients_l706_706687


namespace sum_reciprocals_common_factors_l706_706389

-- Given conditions
def factors_12 : Set ℕ := {1, 2, 3, 4, 6, 12}
def factors_18 : Set ℕ := {1, 2, 3, 6, 9, 18}

-- Common factors of 12 and 18
def common_factors : Set ℕ := {1, 2, 3, 6}

-- Sum of reciprocals of common factors
noncomputable def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ x, (1 : ℚ) / x)

theorem sum_reciprocals_common_factors : sum_of_reciprocals common_factors = 2 := 
by
  sorry

end sum_reciprocals_common_factors_l706_706389


namespace find_q_l706_706117

theorem find_q : 
  (∃ q : ℝ → ℝ, 
    (q 1 = 0) ∧ 
    (q (-3) = 0) ∧ 
    (∀ x : ℝ, (degree q < 4) ∧ (q 4 = 20) ∧ 
    ((∀ x : ℝ, (x != 1) ∧ (x != -3) → (degrees_of_p(x^4 - 3 * x^3 - 4 * x^2 + 12 * x + 9) > degrees_of_p(q))) → 
    (q(x) = (20/21)*x^2 + (40/21)*x - (60/21)))) :=
by
  sorry

end find_q_l706_706117


namespace part1_part2_l706_706178

noncomputable def polar_eqn (θ : ℝ) : ℝ := (4 / (Real.cos θ ^ 2 + 4 * Real.sin θ ^ 2))

noncomputable def cartesian_eqn (x y : ℝ) : Prop := (x^2 / 4 + y^2 = 1)

theorem part1 {x y : ℝ} : (∃ θ : ℝ, (x = polar_eqn θ * Real.cos θ) ∧ (y = polar_eqn θ * Real.sin θ)) ↔ cartesian_eqn x y :=
sorry

theorem part2 (P Q : ℝ × ℝ) (hP : (∃ θP : ℝ, (P.1 = polar_eqn θP * Real.cos θP) ∧ (P.2 = polar_eqn θP * Real.sin θP)))
                        (hQ : (∃ θQ : ℝ, (Q.1 = polar_eqn (θQ + Real.pi / 2) * Real.cos (θQ + Real.pi / 2)) 
                        ∧ (Q.2 = polar_eqn (θQ + Real.pi / 2) * Real.sin (θQ + Real.pi / 2))))
                        (hbot : (P.1 * Q.1 + P.2 * Q.2 = 0)) :
                        (1 / (P.1^2 + P.2^2) + 1 / (Q.1^2 + Q.2^2) = 5 / 4) :=
sorry

end part1_part2_l706_706178


namespace circumcenters_not_concyclic_l706_706274

/-- 
Let \(ABCD\) be a convex quadrilateral in the plane and let \(O_{A}, O_{B}, O_{C}\) and \(O_{D}\) be the circumcenters 
of the triangles \(BCD, CDA, DAB\) and \(ABC\) respectively. Suppose these four circumcenters are distinct points. 
Prove that these points are not on the same circle.
-/
theorem circumcenters_not_concyclic 
  (A B C D O_A O_B O_C O_D : Type)
  [is_convex_quad A B C D]
  (h_O_A : is_circumcenter O_A B C D)
  (h_O_B : is_circumcenter O_B C D A)
  (h_O_C : is_circumcenter O_C D A B)
  (h_O_D : is_circumcenter O_D A B C)
  (distinct : O_A ≠ O_B ∧ O_B ≠ O_C ∧ O_C ≠ O_D ∧ O_D ≠ O_A) :
  ¬ are_concyclic O_A O_B O_C O_D :=
begin
  sorry
end

end circumcenters_not_concyclic_l706_706274


namespace even_perfect_squares_between_50_and_200_l706_706209

theorem even_perfect_squares_between_50_and_200 : ∃ s : Finset ℕ, 
  (∀ n ∈ s, (n^2 ≥ 50) ∧ (n^2 ≤ 200) ∧ n^2 % 2 = 0) ∧ s.card = 4 := by
  sorry

end even_perfect_squares_between_50_and_200_l706_706209


namespace condition_on_x_l706_706928

theorem condition_on_x (x : ℝ) (h : {1, x, x^2}.card = 3) : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end condition_on_x_l706_706928


namespace cost_of_shoes_l706_706856

theorem cost_of_shoes 
  (budget : ℕ) (shirt : ℕ) (pants : ℕ) (coat : ℕ) (socks : ℕ) (belt : ℕ) (remaining_budget : ℕ) (shoes : ℕ) :
  budget = 200 →
  shirt = 30 →
  pants = 46 →
  coat = 38 →
  socks = 11 →
  belt = 18 →
  remaining_budget = 16 →
  shoes = 41 →
  budget - remaining_budget = (shirt + pants + coat + socks + belt + shoes) :=
begin
  sorry
end

end cost_of_shoes_l706_706856


namespace centroid_quadrilateral_area_ratio_l706_706261

noncomputable def vector := (ℝ × ℝ × ℝ)

def is_convex_quadrilateral (P Q R S : vector) : Prop :=
  -- Assume some condition here to define convexity
  sorry

def centroid (A B C : vector) : vector :=
  (A.1 / 3 + B.1 / 3 + C.1 / 3, A.2 / 3 + B.2 / 3 + C.2 / 3, A.3 / 3 + B.3 / 3 + C.3 / 3)

def area_ratio (PGQG : vector) (PQRS : vector) : ℝ :=
  -- Define the area ratio here
  sorry

-- Lean 4 Statement for the problem
theorem centroid_quadrilateral_area_ratio (P Q R S : vector) :
  is_convex_quadrilateral P Q R S →
  let G_P := centroid Q R S in
  let G_Q := centroid P R S in
  let G_R := centroid P Q S in
  let G_S := centroid P Q R in
  area_ratio (G_P, G_Q, G_R, G_S) (P, Q, R, S) = 1 / 9 :=
sorry

end centroid_quadrilateral_area_ratio_l706_706261


namespace coin_probability_l706_706444

theorem coin_probability :
  let value_quarters : ℚ := 15.00
  let value_nickels : ℚ := 15.00
  let value_dimes : ℚ := 10.00
  let value_pennies : ℚ := 5.00
  let number_quarters := value_quarters / 0.25
  let number_nickels := value_nickels / 0.05
  let number_dimes := value_dimes / 0.10
  let number_pennies := value_pennies / 0.01
  let total_coins := number_quarters + number_nickels + number_dimes + number_pennies
  let probability := (number_quarters + number_dimes) / total_coins
  probability = (1 / 6) := by 
sorry

end coin_probability_l706_706444


namespace part2_l706_706278

noncomputable def parabola_equation (p : ℝ) : (ℝ × ℝ) → Prop := 
  λ (x y : ℝ), y^2 = 2 * p * x

noncomputable def is_equilateral_triangle (a b f : ℝ × ℝ) (side_len : ℝ) : Prop := 
  dist a b = side_len ∧ dist b f = side_len ∧ dist f a = side_len

lemma part1 {p : ℝ} (p_pos : p > 0) (A F B : ℝ × ℝ) 
  (h1 : parabola_equation p A.1 A.2)
  (h2 : B.2 = -(p/2)) 
  (h3 : is_equilateral_triangle A B F 4) :
  p = 2 := sorry

theorem part2 (N : ℝ × ℝ) 
  (h1 : N = (2, 0)) :
  ∃ k : ℝ, ∀ l' : ℝ → Prop, ∀ x y : ℝ, 
  (l' x y) → 
  (dQ : ℝ), (dR : ℝ),
  (intersects_parabola_l' Q (parabola_equation 2) l' N) ∧ 
  (intersects_parabola_l' R (parabola_equation 2) l' N) → 
  ∃ (Q R : ℝ × ℝ), 
  dist N Q * dist N Q + dist N R * dist N R = k :=
sorry

end part2_l706_706278


namespace rowing_speed_l706_706039

theorem rowing_speed (V_m V_w V_upstream V_downstream : ℝ)
  (h1 : V_upstream = 25)
  (h2 : V_downstream = 65)
  (h3 : V_w = 5) :
  V_m = 45 :=
by
  -- Lean will verify the theorem given the conditions
  sorry

end rowing_speed_l706_706039


namespace average_rst_l706_706973

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : (r + s + t) / 3 = 14 / 3 :=
by
  sorry

end average_rst_l706_706973


namespace roots_intervals_l706_706976

theorem roots_intervals (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  ∃ x₁ x₂, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧ (f x₁ = 0) ∧ (f x₂ = 0) 
where
  f(x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) :=
sorry

end roots_intervals_l706_706976


namespace peaches_in_each_basket_l706_706792

variable (R : ℕ)

theorem peaches_in_each_basket (h : 6 * R = 96) : R = 16 :=
by
  sorry

end peaches_in_each_basket_l706_706792


namespace T_n_sum_general_term_b_b_n_comparison_l706_706927

noncomputable def sequence_a (n : ℕ) : ℕ := sorry  -- Placeholder for sequence {a_n}
noncomputable def S (n : ℕ) : ℕ := sorry  -- Placeholder for sum of first n terms S_n
noncomputable def sequence_b (n : ℕ) (q : ℝ) : ℝ := sorry  -- Placeholder for sequence {b_n}

axiom sequence_a_def : ∀ n : ℕ, 2 * sequence_a (n + 1) = sequence_a n + sequence_a (n + 2)
axiom sequence_a_5 : sequence_a 5 = 5
axiom S_7 : S 7 = 28

noncomputable def T (n : ℕ) : ℝ := (2 * n : ℝ) / (n + 1 : ℝ)

theorem T_n_sum : ∀ n : ℕ, T n = 2 * (1 - 1 / (n + 1)) := sorry

axiom b1 : ℝ
axiom b_def : ∀ (n : ℕ) (q : ℝ), q > 0 → sequence_b (n + 1) q = sequence_b n q + q ^ (sequence_a n)

theorem general_term_b (q : ℝ) (n : ℕ) (hq : q > 0) : 
  (if q = 1 then sequence_b n q = n else sequence_b n q = (1 - q ^ n) / (1 - q)) := sorry

theorem b_n_comparison (q : ℝ) (n : ℕ) (hq : q > 0) : 
  sequence_b n q * sequence_b (n + 2) q < (sequence_b (n + 1) q) ^ 2 := sorry

end T_n_sum_general_term_b_b_n_comparison_l706_706927


namespace tangent_line_at_point_l706_706898

theorem tangent_line_at_point (x y : ℝ) (h : y = x / (x - 2)) (hx : x = 1) (hy : y = -1) : y = -2 * x + 1 :=
sorry

end tangent_line_at_point_l706_706898


namespace hare_tortoise_race_l706_706033

theorem hare_tortoise_race :
  ∃ (x y : ℝ) (v : ℝ),
  y = 13 ∧
  5 * v ∈ Set.Ioo 0.0 20.0 ∧ -- v should be positive and less than 20 assuming a positive speed.
  (x + y = 25) ∧ 
  (x^2 + 25 = y^2) ∧ 
  (0 < v) ∧ 
  (v ≠ 0) :
    y = 13 :=
by
  sorry

end hare_tortoise_race_l706_706033


namespace items_sold_increase_by_20_percent_l706_706826

-- Assume initial variables P (price per item without discount) and N (number of items sold without discount)
variables (P N : ℝ)

-- Define the conditions and the final proof goal
theorem items_sold_increase_by_20_percent 
  (h1 : ∀ (P N : ℝ), P > 0 → N > 0 → (P * N > 0))
  (h2 : ∀ (P : ℝ), P' = P * 0.90)
  (h3 : ∀ (P' N' : ℝ), P' * N' = P * N * 1.08)
  : (N' - N) / N * 100 = 20 := 
sorry

end items_sold_increase_by_20_percent_l706_706826


namespace hyperbola_real_axis_length_l706_706151

theorem hyperbola_real_axis_length :
  let a := 2 * b in
  let point := (-1, 4) in
    -- Conditions
  (∃ (a b : ℝ), a = 2 * b ∧ 
    (point.2 ^ 2 / a ^ 2) - (point.1 ^ 2 / b ^ 2) = 1 ∧ 
    a = 2 * Real.sqrt(3)) →
    -- Conclusion
  2 * a = 4 * Real.sqrt 3 :=
by
  sorry

end hyperbola_real_axis_length_l706_706151


namespace bolt_catches_ace_at_correct_distance_l706_706069

-- Define the main variables
variables (v z m : ℝ)

-- z is required to be greater than 0
axiom h_z_pos : z > 0

-- Define the speeds for Ace and Bolt
def ace_speed := v
def bolt_speed := (1 + z / 100) * v

-- Define the problem condition and the target distance
def head_start := m
def meters_to_catch_up : ℝ := (100 + z) * m / (100 - z)

-- The theorem we need to prove
theorem bolt_catches_ace_at_correct_distance 
  (h_head_start : head_start = m)
  (h_bolt_speed : bolt_speed = (1 + z / 100) * v)
  (h_ace_speed : ace_speed = v)
  (h_z_pos : z > 0) :
  ∃ d, d = (100 + z) * m / (100 - z) :=
by
  sorry

end bolt_catches_ace_at_correct_distance_l706_706069


namespace scaling_adults_taller_l706_706364

open Nat

theorem scaling_adults_taller (n : ℕ) (h_c h_a : Fin n → ℚ) (h : ∀ i, h_c i < h_a i) :
  ∃ (A : Fin n → ℕ), ∀ i j, h_c i * (A i) < h_a j * (A j) := by
  sorry

end scaling_adults_taller_l706_706364


namespace chord_length_condition_l706_706952

theorem chord_length_condition {k : ℝ} 
  (h_intersection : ∀ x y : ℝ, (y = k * x) → ((x - 2) ^ 2 + y ^ 2 = 4) → 
                     (∃ p q : ℝ, p ≠ q ∧ x = p ∨ x = q) ∧ ((sqrt (4 - d ^ 2) * 2 = 2))) : 
  |k| = Real.sqrt 3 := 
sorry

end chord_length_condition_l706_706952


namespace new_solution_percentages_l706_706022

noncomputable theory

def original_solution_volume : ℝ := 600
def original_water_percent : ℝ := 53.5 / 100
def original_cola_percent : ℝ := 11.5 / 100
def original_sugar_percent : ℝ := 9.5 / 100
def original_spice_percent : ℝ := 25.25 / 100
def original_tea_percent : ℝ := 0.25 / 100

def added_water_volume : ℝ := 12.3
def added_cola_volume : ℝ := 7.8
def added_sugar_volume : ℝ := 3.6
def added_spice_volume : ℝ := 8.5
def added_tea_volume : ℝ := 1.8

def total_volume : ℝ := original_solution_volume + added_water_volume + added_cola_volume + added_sugar_volume + added_spice_volume + added_tea_volume

def new_water_volume : ℝ := (original_solution_volume * original_water_percent) + added_water_volume
def new_cola_volume : ℝ := (original_solution_volume * original_cola_percent) + added_cola_volume
def new_sugar_volume : ℝ := (original_solution_volume * original_sugar_percent) + added_sugar_volume
def new_spice_volume : ℝ := (original_solution_volume * original_spice_percent) + added_spice_volume
def new_tea_volume : ℝ := (original_solution_volume * original_tea_percent) + added_tea_volume

def new_water_percent : ℝ := (new_water_volume / total_volume) * 100
def new_cola_percent : ℝ := (new_cola_volume / total_volume) * 100
def new_sugar_percent : ℝ := (new_sugar_volume / total_volume) * 100
def new_spice_percent : ℝ := (new_spice_volume / total_volume) * 100
def new_tea_percent : ℝ := (new_tea_volume / total_volume) * 100

theorem new_solution_percentages :
  new_water_percent = 52.57 ∧
  new_cola_percent = 12.12 ∧
  new_sugar_percent = 9.56 ∧
  new_spice_percent = 25.24 ∧
  new_tea_percent = 0.52
:=
by sorry

end new_solution_percentages_l706_706022


namespace unique_fraction_property_l706_706516

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l706_706516


namespace min_is_in_S_max_is_not_in_S_problem_solution_l706_706270

noncomputable theory

def f (x : ℝ) := (3 * x + 4) / (x + 3)
def S := {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = f(x)}

theorem min_is_in_S : (7 / 4) ∈ S :=
  by sorry -- proof to be filled in

theorem max_is_not_in_S : ¬(3 ∈ S) :=
  by sorry -- proof to be filled in

theorem problem_solution :
  (7 / 4 ∈ S) ∧ ¬(3 ∈ S) :=
  by
    apply And.intro
    · exact min_is_in_S
    · exact max_is_not_in_S

end min_is_in_S_max_is_not_in_S_problem_solution_l706_706270


namespace eqn_of_line_CD_eqn_of_circle_P_num_points_Q_l706_706922

def Point2D := (ℝ × ℝ)

def circle_center (P : Point2D) (r : ℝ) (A B : Point2D) (CD_len : ℝ) : Bool :=
  let (a, b) := P
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let mid_AB := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let dist_CD := CD_len
  let radius_squared := dist_CD * dist_CD / 4
  ((x₁ - a)^2 + y₁^2 == radius_squared) && 
  ((x₂ - a)^2 + y₂^2 == radius_squared) && 
  (a + b - 3 = 0)
  
theorem eqn_of_line_CD (A B : Point2D) : Prop :=
∃ (C D : Point2D), 
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let mid_AB := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let k := -((x₂ - x₁) / (y₂ - y₁))
  let line_CD (x y : ℝ) := x + y - 3 = 0
  True  -- asserting the simplified equation directly due to stated solution
  sorry

theorem eqn_of_circle_P (A B : Point2D) (CD_len : ℝ) : Prop :=
∃ (P : Point2D), 
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  ∃ (r : ℝ), 
    circle_center P r A B CD_len ∧
     ((P = (-3, 6) ∧ r = 2 * sqrt 10) ∨ (P = (5, -2) ∧ r = 2 * sqrt 10))
  sorry

theorem num_points_Q (A B : Point2D) (P : Point2D) (CD_len : ℝ) : Prop :=
let r := 2 * sqrt 10
let Q_area := 8
given_triangle_area (A B Q: Point2D) (A B P: Point2D) (Q_area: ℝ) (r : ℝ) : Bool :=
True -- after asserting calculations from solution
2 -- exact gt 2 points Q based on calculation
sorry

end eqn_of_line_CD_eqn_of_circle_P_num_points_Q_l706_706922


namespace flax_plant_relative_frequency_probability_l706_706781

noncomputable def probability_of_deviation_within_epsilon 
  (n : ℕ) 
  (ε : ℝ) 
  (p : ℝ) : ℝ :=
  2 * Real.erf (ε * Real.sqrt (↑n / (p * (1 - p))))

theorem flax_plant_relative_frequency_probability :
  probability_of_deviation_within_epsilon 300 0.05 0.6 ≈ 0.9232 :=
by 
  sorry

end flax_plant_relative_frequency_probability_l706_706781


namespace percentage_of_pines_is_13_l706_706646

/-- Total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- Number of spruces, 10% of total trees -/
def spruces : ℕ := 0.10 * total_trees

/-- Number of birches -/
def birches : ℕ := 2160

/-- Number of oaks equals number of spruces and pines together -/
def oaks (spruces pines : ℕ) : ℕ := spruces + pines

/-- Sum of all trees equals total number of trees -/
lemma trees_total : ∀ (oaks pines : ℕ), oaks + pines + spruces + birches = total_trees :=
by sorry

/-- Calculate the percentage of pines -/
def percentage_pines (pines total : ℕ) : ℕ := (pines * 100) / total

/-- Given the conditions, prove that the percentage of pines is 13% -/
theorem percentage_of_pines_is_13 :
  percentage_pines 520 total_trees = 13 :=
by sorry

end percentage_of_pines_is_13_l706_706646


namespace parallelogram_area_l706_706494

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x, y := p1.y - p2.y, z := p1.z - p2.z }

def vector_cross (v1 v2 : Point3D) : Point3D :=
  {
    x := v1.y * v2.z - v1.z * v2.y,
    y := v1.z * v2.x - v1.x * v2.z,
    z := v1.x * v2.y - v1.y * v2.x
  }

def vector_magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def are_points_forming_parallelogram (A B C D : Point3D) : Prop :=
  vector_sub B A = vector_sub D C

theorem parallelogram_area :
  let A := {x := 2, y := -5, z := 3}
  let B := {x := 6, y := -7, z := 7}
  let C := {x := 3, y := -2, z := 1}
  let D := {x := 7, y := -4, z := 5}
  are_points_forming_parallelogram A B C D →
  vector_magnitude (vector_cross (vector_sub B A) (vector_sub C A)) = real.sqrt 596 :=
by
  sorry

end parallelogram_area_l706_706494


namespace remaining_lawn_area_l706_706996

theorem remaining_lawn_area (lawn_length lawn_width path_width : ℕ) 
  (h_lawn_length : lawn_length = 10) 
  (h_lawn_width : lawn_width = 5) 
  (h_path_width : path_width = 1) : 
  (lawn_length * lawn_width - lawn_length * path_width) = 40 := 
by 
  sorry

end remaining_lawn_area_l706_706996


namespace scaling_adults_taller_l706_706365

open Nat

theorem scaling_adults_taller (n : ℕ) (h_c h_a : Fin n → ℚ) (h : ∀ i, h_c i < h_a i) :
  ∃ (A : Fin n → ℕ), ∀ i j, h_c i * (A i) < h_a j * (A j) := by
  sorry

end scaling_adults_taller_l706_706365


namespace my_and_mothers_ages_l706_706717

-- Definitions based on conditions
noncomputable def my_age (x : ℕ) := x
noncomputable def mothers_age (x : ℕ) := 3 * x
noncomputable def sum_of_ages (x : ℕ) := my_age x + mothers_age x

-- Proposition that needs to be proved
theorem my_and_mothers_ages (x : ℕ) (h : sum_of_ages x = 40) :
  my_age x = 10 ∧ mothers_age x = 30 :=
by
  sorry

end my_and_mothers_ages_l706_706717


namespace man_speed_l706_706040

theorem man_speed
    (distance_meters : ℝ)
    (time_seconds : ℝ)
    (h_dist : distance_meters = 375.03)
    (h_time : time_seconds = 30) :
    (distance_meters / 1000) / (time_seconds / 3600) = 45.0036 :=
by
    rw [h_dist, h_time]
    norm_num
    sorry

end man_speed_l706_706040


namespace quadratic_function_solution_l706_706196

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2*x + 15
def g (x m : ℝ) : ℝ := (1 - 2*m)*x - f x

theorem quadratic_function_solution (f : ℝ → ℝ) (g : ℝ → ℝ → ℝ) (m x : ℝ) :
  (∀ x, f(x+1) - f(x) = -2*x + 1) → (f(2) = 15) →
  f x = -x^2 + 2*x + 15 ∧
  g x m = (1 - 2*m)*x - f x ∧
  (g 0 m ≤ g (m + 1/2) m ∧ g 0 m ≤ g 2 m) → (m ≤ -1/2) → g(0, m) = -15 ∨
  (-1/2 < m ∧ m < 3/2) → g(m + 1/2, m) = -m^2 - m - 61/4 ∨
  (m ≥ 3/2) → g(2, m) = -4*m - 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quadratic_function_solution_l706_706196


namespace proof_of_false_statement_C_l706_706915

noncomputable def geometric_problem (m n : Line) (α β : Plane) : Prop :=
  ∃ (m n : Line) (α β : Plane), 
    m ≠ n ∧ 
    α ≠ β ∧ 
    (m ∥ α ∧ α ⟂ β → ¬ (m ⟂ β))

theorem proof_of_false_statement_C (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : m ∥ α)
  (h4 : α ⟂ β) :
  ¬ (m ⟂ β) :=
sorry

end proof_of_false_statement_C_l706_706915


namespace inverse_value_l706_706953

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value_l706_706953


namespace hyperbola_eccentricity_l706_706924

variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2 * a)
def hyperbola := ∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2 * a) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l706_706924


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l706_706344

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l706_706344


namespace radius_of_circle_l706_706842

theorem radius_of_circle
  (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 7) (h3 : QR = 8) :
  ∃ r : ℝ, r = 2 * Real.sqrt 30 ∧ (PQ * (PQ + QR) = (d - r) * (d + r)) :=
by
  -- All necessary non-proof related statements
  sorry

end radius_of_circle_l706_706842


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l706_706345

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l706_706345


namespace no_fractions_meet_condition_l706_706508

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l706_706508


namespace not_necessarily_congruent_l706_706201

-- Definitions and conditions
structure Triangle :=
(angleA : ℝ) (angleB : ℝ) (angleC : ℝ)
(sideA : ℝ) (sideB : ℝ) (sideC : ℝ)
(angle_sum_eq : angleA + angleB + angleC = 180)
(isosceles : (sideA = sideB) ∨ (sideB = sideC) ∨ (sideC = sideA))
(acute_angled : angleA < 90 ∧ angleB < 90 ∧ angleC < 90)

-- Main theorem
theorem not_necessarily_congruent (T1 T2 : Triangle) :
  (T1.isosceles ∧ T2.isosceles) ∧
  ((T1.angleA = T2.angleA) ∨ (T1.angleB = T2.angleB) ∨ (T1.angleC = T2.angleC)) ∧
  ((T1.sideA = T2.sideA) ∨ (T1.sideB = T2.sideB) ∨ (T1.sideC = T2.sideC)) →
  ¬ (T1 = T2) :=
by
  sorry

end not_necessarily_congruent_l706_706201


namespace find_fx_for_neg_two_to_zero_l706_706701

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) (hx : x ∈ set.Icc 2 3) : f (x - 2) = x - 2 :=
sorry

lemma even_f (x : ℝ) : f (-x) = f (x) :=
sorry

lemma given_property (x : ℝ) (hx : x ∈ set.Icc 2 3) : f (x) = x :=
sorry

theorem find_fx_for_neg_two_to_zero (x : ℝ) (hx : x ∈ set.Icc (-2) 0) : f (x) = 3 - |x + 1| :=
begin
  sorry
end

end find_fx_for_neg_two_to_zero_l706_706701


namespace sum_of_altitudes_of_triangle_l706_706326

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line with the coordinate axes
def forms_triangle_with_axes (x y : ℝ) : Prop := 
  line_eq x 0 ∧ line_eq 0 y

-- Prove the sum of the lengths of the altitudes is 511/17
theorem sum_of_altitudes_of_triangle : 
  ∃ x y : ℝ, forms_triangle_with_axes x y → 
  15 + 8 + (120 / 17) = 511 / 17 :=
by
  sorry

end sum_of_altitudes_of_triangle_l706_706326


namespace at_least_two_equal_sums_l706_706645

theorem at_least_two_equal_sums (grid : Fin 5 → Fin 5 → ℕ) 
  (h1 : ∀ i j, grid i j ∈ {1, 3, 5, 7}) : 
  ∃ (i1 i2 : Fin 28), i1 ≠ i2 ∧ 
    let sums := 
      (Fin 5).toList.map (λ r : Fin 5 => (Fin 5).sum (λ c => grid r c)) ++ -- rows
      (Fin 5).toList.map (λ c : Fin 5 => (Fin 5).sum (λ r => grid r c)) ++ -- columns
      ((0 : Fin 5).mapList (λ d : Fin 5 =>
          ∑ (i : Fin 5), grid (i + d) i) ++
      (0 : Fin 5).mapList (λ d : Fin 5 =>
          ∑ (i : Fin 5), grid i (i + d))) in
    sums.nth i1 = sums.nth i2 :=
sorry

end at_least_two_equal_sums_l706_706645


namespace infinite_sequence_exists_l706_706537

noncomputable def exists_inf_seq_coprime_condition : Prop :=
  ∃ (a : ℕ → ℕ), 
    (∀ (m n : ℕ), m ≠ n → (Nat.gcd (a m) (a n) = 1 ↔ |m - n| = 1))

theorem infinite_sequence_exists : exists_inf_seq_coprime_condition :=
sorry

end infinite_sequence_exists_l706_706537


namespace average_GPA_school_l706_706752

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l706_706752


namespace breadth_of_garden_l706_706986

theorem breadth_of_garden (P L B : ℝ) (hP : P = 1800) (hL : L = 500) : B = 400 :=
by
  sorry

end breadth_of_garden_l706_706986


namespace divisibility_323_l706_706141

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end divisibility_323_l706_706141


namespace ellipse_properties_l706_706160

noncomputable def ellipse_standard_equation : Prop :=
  let a := 2 * Real.sqrt 3
  let c := 2 * Real.sqrt 2
  let b_sq := a^2 - c^2
  ∃ (a b : ℝ), a = 2 * Real.sqrt 3 ∧ b^2 = b_sq ∧
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  ((b^2 = 4) → (∀ (x y : ℝ), x^2 / 12 + y^2 / 4 = 1))

noncomputable def always_constant_dot_product_RM_RN (k : ℝ) : Prop :=
  let a := 2 * Real.sqrt 3
  let c := 2 * Real.sqrt 2
  let b_sq := a^2 - c^2
  let R := (0, -2)
  let M := (0, 1)
  let l_2 (k : ℝ) := (λ x : ℝ, k * x + 1)
  let x1_x2_eq := -6 * k / (1 + 3 * k^2)
  let x1x2_eq := -9 / (1 + 3 * k^2)
  let RM x1 := (x1, (k * x1) + 3)
  let RN x2 := (x2, (k * x2) + 3)
  ∀ (x1 x2 : ℝ), ((x1 + x2 = x1_x2_eq) ∧ (x1 * x2 = x1x2_eq) →
  (RM x1) • (RN x2) = 0)

theorem ellipse_properties :
  ellipse_standard_equation ∧ always_constant_dot_product_RM_RN :=
by
  split
  sorry
  sorry

end ellipse_properties_l706_706160


namespace worst_player_is_son_or_sister_l706_706834

axiom Family : Type
axiom Woman : Family
axiom Brother : Family
axiom Son : Family
axiom Daughter : Family
axiom Sister : Family

axiom are_chess_players : ∀ f : Family, Prop
axiom is_twin : Family → Family → Prop
axiom is_best_player : Family → Prop
axiom is_worst_player : Family → Prop
axiom same_age : Family → Family → Prop
axiom opposite_sex : Family → Family → Prop
axiom is_sibling : Family → Family → Prop

-- Conditions
axiom all_are_chess_players : ∀ f, are_chess_players f
axiom worst_best_opposite_sex : ∀ w b, is_worst_player w → is_best_player b → opposite_sex w b
axiom worst_best_same_age : ∀ w b, is_worst_player w → is_best_player b → same_age w b
axiom twins_relationship : ∀ t1 t2, is_twin t1 t2 → (is_sibling t1 t2 ∨ (t1 = Woman ∧ t2 = Sister))

-- Goal
theorem worst_player_is_son_or_sister :
  ∃ w, (is_worst_player w ∧ (w = Son ∨ w = Sister)) :=
sorry

end worst_player_is_son_or_sister_l706_706834


namespace service_center_location_l706_706443

-- Definitions from conditions
def third_exit := 30
def twelfth_exit := 195
def seventh_exit := 90

-- Concept of distance and service center location
def distance := seventh_exit - third_exit
def service_center_milepost := third_exit + 2 * distance / 3

-- The theorem to prove
theorem service_center_location : service_center_milepost = 70 := by
  -- Sorry is used to skip the proof details.
  sorry

end service_center_location_l706_706443


namespace parabola_focus_circle_intersect_l706_706194

theorem parabola_focus_circle_intersect (p : ℝ) (h : p > 0) :
  (let F := (⟨p / 2, 0⟩ : ℝ × ℝ)
   in let Q := (⟨3 * p / 2, 0⟩ : ℝ × ℝ)
      in let y := p
         in let N := (⟨p / 2, p⟩ : ℝ × ℝ)
            in dist N Q = sqrt 10) →
  p = sqrt 5 :=
by
  intros hF
  -- proof to be provided
  sorry

end parabola_focus_circle_intersect_l706_706194


namespace find_x_coordinate_l706_706248

-- Define points
def point1 : ℝ × ℝ := (-4, -4)
def point2 : ℝ × ℝ := (4, 0)

-- Slope calculation from point1 and point2
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Line equation based on point1 and slope
def line_eq (x : ℝ) : ℝ := (slope point1 point2) * (x + 4) - 4

-- The x-coordinate of the point with the y-coordinate of 3
theorem find_x_coordinate : 
  line_eq 10 = 3 := 
by 
  -- Placeholder for proof
  sorry

end find_x_coordinate_l706_706248


namespace parts_made_by_A_l706_706723

def total_parts : ℕ := 50
def ratio_BC : ℚ := 3 / 4
def parts_C : ℕ := 20
def parts_B : ℕ := 15
def parts_A : ℕ := 15

theorem parts_made_by_A (A B C T : ℕ) (h1 : A = 3 / 10 * T) (h2 : B / C = 3 / 4) (h3 : C = 20) (h4 : 3 / 4 * C = B) (h5 : 7 / 10 * T = B + C) :
  A = 15 :=
by
  rw [←h4] at h2
  rw [h3, h2] at h5
  have h6 : T = 50
  {
    sorry
  }
  rw [h1, h6]
  norm_num

end parts_made_by_A_l706_706723


namespace ratio_is_sqrt_5_div_8_l706_706451

noncomputable theory

-- Define the width and lengths
def width : ℝ 
def length : ℝ := 4 * width

-- Define the original area of the rectangle A
def original_area : ℝ := width * length

-- Coordinates of the top rectangle division points
def point1 := (width, 0)
def point2 := (3 * width, width)

-- Distance between the points (length of the dotted line)
def dotted_line_length : ℝ := real.sqrt ((3 * width - width)^2 + (width - 0)^2)

-- Define the area of the new triangle B
def new_triangle_area : ℝ := (1/2) * dotted_line_length * width

-- The ratio of the areas B/A
def ratio : ℝ := new_triangle_area / original_area

-- Prove that the ratio is correct
theorem ratio_is_sqrt_5_div_8 : ratio = real.sqrt 5 / 8 := by
  sorry

end ratio_is_sqrt_5_div_8_l706_706451


namespace parking_lot_spaces_l706_706369

theorem parking_lot_spaces (n : ℕ) :
  (∃ A B C,
    (A = (n - 2).choose 3) ∧
    (B = 2 * (n - 2).choose 2) ∧
    (C = A ∧ C = B)) →
  n = 10 :=
begin
  intros h,
  cases' h with A h',
  cases' h' with B h'',
  cases' h'' with C h''',
  obtain ⟨hA, hB, hC1, hC2⟩ := h''',
  simp only [hA, hB] at hC1 hC2,
  sorry
end

end parking_lot_spaces_l706_706369


namespace radical_axis_of_circumcircles_passes_through_midpoint_l706_706614

theorem radical_axis_of_circumcircles_passes_through_midpoint
  (triangle_abc : Triangle)
  (I : Circle)
  (D E F : Point)
  (incircle_touches_bc_ca_ab : I.touches_sides triangle_abc D E F)
  (K : Line)
  (K_foot_perp_from_D_to_EF : K.foot_perpendicular_from D E F)
  (AIB AIC : Triangle)
  (circumcircle_AIB AIC_intersect_incirc_at_C1_C2 : Circle)
  (circumcircle_AIB_intersects_incirc_at_C1_C2 : circumcircle_AIB.intersects I at [C1, C2])
  (circumcircle_AIC : Circle)
  (circumcircle_AIC_intersects_incirc_at_B1_B2 : circumcircle_AIC.intersects I at [B1, B2])
  : radical_axis (circumcircle (triangle BB_1B_2)) 
                 (circumcircle (triangle CC_1C_2)) 
                 passes_through (midpoint DK) :=
sorry

end radical_axis_of_circumcircles_passes_through_midpoint_l706_706614


namespace find_incomes_l706_706878

theorem find_incomes (M N O P Q : ℝ) 
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (O + P) / 2 = 6800)
  (h4 : (P + Q) / 2 = 7500)
  (h5 : (M + O + Q) / 3 = 6000) :
  M = 300 ∧ N = 9800 ∧ O = 2700 ∧ P = 10900 ∧ Q = 4100 :=
by
  sorry


end find_incomes_l706_706878


namespace solve_inequality_l706_706737

theorem solve_inequality (x : ℝ) :
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  x ∈ set.Ioo (x) (-∞, 1) ∪ (1, 2) ∪ (2, 4) ∪ (4, 5) ∪ (7, ∞) :=
by sorry

end solve_inequality_l706_706737


namespace max_passwords_possible_l706_706376

theorem max_passwords_possible :
  let digits := {1, 3, 5, 7, 9} in
  ∀ (password : vector ℕ 6), (∀ digit ∈ password.to_list, digit ∈ digits) →
  (password.length = 6) →
  (∃ n : ℕ, n = 15625) :=
by
  sorry

end max_passwords_possible_l706_706376


namespace product_inequality_l706_706586

theorem product_inequality
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i) 
  (h2 : ∑ i, a i = 1) 
  : ∏ i, (1 + 1 / (a i)^2) ≥ (1 + n^2)^(n : ℝ) := 
by
  sorry

end product_inequality_l706_706586


namespace percent_white_area_l706_706653

theorem percent_white_area (r : ℕ) (h1 : r = 3)
  (succ_radius : ∀ n, r + 3 * n)
  (h2 : ∀ n, n < 5)
  (h3 : ¬ r * 2 ≤ 30)
  : ((3 * 3 * π + 6 * 6 * π + 9 * 9 * π + 12 * 12 * π + 15 * 15 * π) / (225 * π)) * 100 = 60 := 
sorry

end percent_white_area_l706_706653


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706341

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706341


namespace sequence_general_term_l706_706610

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n > 1, a n = 2 * a (n-1) + 1) : a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l706_706610


namespace total_turnover_in_first_quarter_l706_706454

theorem total_turnover_in_first_quarter (x : ℝ) : 
  200 + 200 * (1 + x) + 200 * (1 + x) ^ 2 = 1000 :=
sorry

end total_turnover_in_first_quarter_l706_706454


namespace determine_swimming_day_l706_706713

def practices_sport_each_day (sports : ℕ → ℕ → Prop) : Prop :=
  ∀ (d : ℕ), ∃ s, sports d s

def runs_four_days_no_consecutive (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (days : ℕ → ℕ), (∀ i, sports (days i) 0) ∧ 
    (∀ i j, i ≠ j → days i ≠ days j) ∧ 
    (∀ i j, (days i + 1 = days j) → false)

def plays_basketball_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 2 1

def plays_golf_friday_after_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 5 2

def swims_and_plays_tennis_condition (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (swim_day tennis_day : ℕ), swim_day ≠ tennis_day ∧ 
    sports swim_day 3 ∧ 
    sports tennis_day 4 ∧ 
    ∀ (d : ℕ), (sports d 3 → sports (d + 1) 4 → false) ∧ 
    (∀ (d : ℕ), sports d 3 → ∀ (r : ℕ), sports (d + 2) 0 → false)

theorem determine_swimming_day (sports : ℕ → ℕ → Prop) : 
  practices_sport_each_day sports → 
  runs_four_days_no_consecutive sports → 
  plays_basketball_tuesday sports → 
  plays_golf_friday_after_tuesday sports → 
  swims_and_plays_tennis_condition sports → 
  ∃ (d : ℕ), d = 7 := 
sorry

end determine_swimming_day_l706_706713


namespace probability_of_A_l706_706412

noncomputable def probability_that_a_occurs : Prop :=
  ∀ (A B : Event) (P : ProbMeasure),
    independent A B ∧ P.prob A > 0 ∧ P.prob A = 2 * P.prob B ∧ P.prob (A ∪ B) = 14 * P.prob (A ∩ B) →
    P.prob A = 1 / 5

-- The statement definition
theorem probability_of_A : probability_that_a_occurs := by
  sorry

end probability_of_A_l706_706412


namespace johns_profit_l706_706681

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l706_706681


namespace no_such_fraction_exists_l706_706507

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l706_706507


namespace minimal_perimeter_triangle_l706_706251

noncomputable def cos_P : ℚ := 3 / 5
noncomputable def cos_Q : ℚ := 24 / 25
noncomputable def cos_R : ℚ := -1 / 5

theorem minimal_perimeter_triangle
  (P Q R : ℝ) (a b c : ℕ)
  (h0 : a^2 + b^2 + c^2 - 2 * a * b * cos_P - 2 * b * c * cos_Q - 2 * c * a * cos_R = 0)
  (h1 : cos_P^2 + (1 - cos_P^2) = 1)
  (h2 : cos_Q^2 + (1 - cos_Q^2) = 1)
  (h3 : cos_R^2 + (1 - cos_R^2) = 1) :
  a + b + c = 47 :=
sorry

end minimal_perimeter_triangle_l706_706251


namespace binomial_coefficient_30_3_l706_706487

theorem binomial_coefficient_30_3 :
  Nat.choose 30 3 = 4060 := 
by 
  sorry

end binomial_coefficient_30_3_l706_706487


namespace edge_length_of_cube_l706_706358

noncomputable def volume_of_largest_cone (a : ℝ) : ℝ :=
  (π * a^3) / 12

theorem edge_length_of_cube (v : ℝ) (h : ∀ a, volume_of_largest_cone a = v) :
  ∃ a, a ≈ 7 :=
by {
  have volume_correct : volume_of_largest_cone 7 = 89.83333333333333,
  { sorry },
  use 7,
  exact volume_correct
}

end edge_length_of_cube_l706_706358


namespace fish_in_pond_estimate_l706_706624

noncomputable def L (N : ℕ) : ℚ :=
  (nat.choose 20 7) * (((N - 20).choose 43) * (50!).to_rat / 
  ((43!).to_rat * (N.choose 50).to_rat))

theorem fish_in_pond_estimate :
  let n_a := 20
  let m := 50
  let k_1 := 7
  let N := 142 in
  L N = nat.choose 20 7 * (((N - 20).choose 43) * 
  (50!).to_rat / ((43!).to_rat * (N.choose 50).to_rat)) := sorry

end fish_in_pond_estimate_l706_706624


namespace more_polygons_with_red_vertex_l706_706361

/-- There are 1997 white points and 1 red point on a circle. -/
def num_white_points : ℕ := 1997
def num_red_points : ℕ := 1
def total_points := num_white_points + num_red_points

/-- Polygons are formed with vertices at these points, and a polygon must have at least 3 vertices. -/
def is_polygon (vertices : ℕ) : Prop := vertices ≥ 3

/-- Are there more polygons with the red vertex or without it? -/
theorem more_polygons_with_red_vertex :
  (∑ k in range(total_points + 1), if k ≥ 3 then choose (total_points - 1) k else 0) < 
  (∑ k in range(num_white_points + 1), if k ≥ 3 then choose num_white_points k else 0) + 
  (∑ k in range(2, total_points + 1), if k ≥ 3 then choose (num_white_points + 1) k else 0) :=
sorry

end more_polygons_with_red_vertex_l706_706361


namespace selecting_at_least_one_B_question_correctly_answering_at_least_two_l706_706359

theorem selecting_at_least_one_B_question (n m : ℕ) (htypeA htypeB : ℕ) (selection : ℕ) 
  (h_sum : htypeA + htypeB = n)
  (h_typeA : htypeA = 6)
  (h_typeB : htypeB = 4)
  (h_selection : selection = 3) :
  (1 - (nat.choose htypeA selection) / (nat.choose n selection)) = 5 / 6 :=
sorry

theorem correctly_answering_at_least_two (probA probB : ℚ) (n_A n_B total : ℕ) 
  (h_probA : probA = 3 / 5)
  (h_probB : probB = 4 / 5)
  (h_nA : n_A = 2)
  (h_nB : n_B = 1)
  (h_total : total = n_A + n_B) :
  ((probA ^ n_A * (1 - probA)) + (2 * probA * probB * (1 - probA)) + (probA ^ n_A * probB)) = 93 / 125 :=
sorry

end selecting_at_least_one_B_question_correctly_answering_at_least_two_l706_706359


namespace rearrange_2023_prob_l706_706222

theorem rearrange_2023_prob :
  (let total_arrangements := 9 in
   let favorable_arrangements := 5 in
   favorable_arrangements / total_arrangements = 5 / 9) :=
begin
  sorry
end

end rearrange_2023_prob_l706_706222


namespace sum_of_roots_2018_l706_706496

noncomputable def my_polynomial : Polynomial ℂ :=
  (Polynomial.monomial 2018 1) + (Polynomial.monomial 2017 1) + 
  (Polynomial.monomial 2016 1) + (Polynomial.monomial 2015 1) -- continue this pattern
  -- you have to add all terms down to the constant term here
  + (Polynomial.monomial 1 1) + (Polynomial.C 675)

noncomputable def sum_b_n (roots : Finₙ 2018 → ℂ) : ℂ :=
  Finₙ.prod roots (λ n, (1 / (2 - roots n)))

theorem sum_of_roots_2018 :
  ∑ (n : Finₙ 2018), (1 / (2 - (polynomial_roots my_polynomial n))) = 3021 :=
sorry

end sum_of_roots_2018_l706_706496


namespace rearrange_2023_prob_l706_706221

theorem rearrange_2023_prob :
  (let total_arrangements := 9 in
   let favorable_arrangements := 5 in
   favorable_arrangements / total_arrangements = 5 / 9) :=
begin
  sorry
end

end rearrange_2023_prob_l706_706221


namespace part1_extreme_values_part2_unique_zero_l706_706605

-- Defining the polynomial function
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Part 1: For a = 2, finding extreme values
theorem part1_extreme_values (x : ℝ) : 
  (∀ x, max (f 2 x) = f 2 0) ∧ (∀ x, min (f 2 x) = f 2 1) := sorry

-- Defining auxiliary function g
def g (x : ℝ) : ℝ := 3 / x - 1 / x^3

-- Part 2: Proving the range of values for a
theorem part2_unique_zero (a : ℝ) :
  (∀ x, (f a x = 0) → ∃! x > 0, f a x = 0) ↔ (a = 2 ∨ a ∈ set.Iic 0) := sorry

end part1_extreme_values_part2_unique_zero_l706_706605


namespace statement_C_l706_706814

-- Definitions of the properties involved
structure Quadrilateral :=
  (A B C D : Type)

structure IsRhombus (q : Quadrilateral) :=
  (diagPerpendBisect : (diagAC q) ⟂ (diagBD q) ∧ bisects (diagAC q) (diagBD q))

-- Define the diagonals
def diagAC (q : Quadrilateral) := sorry
def diagBD (q : Quadrilateral) := sorry

-- Define the concept of perpendicular bisectors
def (⊥) (d1 d2 : Type) := sorry
def bisects (d1 d2 : Type) := sorry

-- Statement C
theorem statement_C (q : Quadrilateral)
  (h : (diagAC q) ⟂ (diagBD q) ∧ bisects (diagAC q) (diagBD q)) :
  IsRhombus q :=
sorry

end statement_C_l706_706814


namespace find_vec_c_l706_706144

-- Definitions of vectors
def veca := (-1, 2) -- Corresponds to \vec{a} = -\vec{i} + 2\vec{j}
def vecb := (2, -3) -- Corresponds to \vec{b} = 2\vec{i} - 3\vec{j}

-- Vector dot product definition for 2D vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- Conditions in the problem
def cond1 (veca vecc : (ℝ × ℝ)) : Prop := dot_product veca vecc = -7
def cond2 (vecb vecc : (ℝ × ℝ)) : Prop := dot_product vecb vecc = 12

-- The vector to be proven
def vecc := (3, -2) -- Corresponds to \vec{c} = 3\vec{i} - 2\vec{j}

theorem find_vec_c : cond1 veca vecc ∧ cond2 vecb vecc :=
by
  sorry

end find_vec_c_l706_706144


namespace john_paint_area_l706_706256

noncomputable def area_to_paint (length width height openings : ℝ) : ℝ :=
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area - openings

theorem john_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  2 * (area_to_paint length width height openings) = 1300 :=
by
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  sorry

end john_paint_area_l706_706256


namespace Dave_spending_l706_706530

variable (Derek_initial : ℕ) (Derek_spent : ℕ) (Derek_remaining : ℕ)
variable (Dave_initial : ℕ) (Dave_remaining : ℕ)

theorem Dave_spending :
  Derek_initial = 40 →
  Derek_spent = 30 →
  Derek_remaining = Derek_initial - Derek_spent →
  Dave_initial = 50 →
  Dave_remaining = Derek_remaining + 33 →
  (Dave_initial - Dave_remaining) = 7 := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Dave_spending_l706_706530


namespace distinct_products_count_l706_706262

-- Define the prime factorization of the number 72000.
def prime_factorization_72000 := (2^5, 3^2, 5^3)

-- Define the set of all positive integer divisors of 72000.
def divisors_72000 := {d : ℕ | ∃ a b c, d = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 5 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 3}

-- Define the function to count the number of products of two distinct elements.
def count_distinct_products (T : set ℕ) : ℕ :=
  {p : ℕ | ∃ x y ∈ T, x ≠ y ∧ p = x * y}.to_finset.card

-- State the theorem to be proven.
theorem distinct_products_count :
  count_distinct_products divisors_72000 = 383 :=
sorry

end distinct_products_count_l706_706262


namespace denise_removed_bananas_l706_706884

theorem denise_removed_bananas (initial_bananas remaining_bananas : ℕ) 
  (h_initial : initial_bananas = 46) (h_remaining : remaining_bananas = 41) : 
  initial_bananas - remaining_bananas = 5 :=
by
  sorry

end denise_removed_bananas_l706_706884


namespace inequality_solution_l706_706740

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end inequality_solution_l706_706740


namespace cube_edge_length_l706_706357

theorem cube_edge_length (V : ℝ) (h : V = 27) : ∃ x : ℝ, x^3 = V ∧ x = 3 :=
by 
  use 3
  split
  · sorry  -- This will prove 3^3 = 27
  · rfl    -- This will reduce to showing 3 = 3


end cube_edge_length_l706_706357


namespace area_of_triangle_angle_C_l706_706640

-- Condition: cos B = 3/5
def cosB : ℝ := 3 / 5

-- Condition: Dot product of vectors AB and BC
def dotProductAB_BC : ℝ := -21

-- First proof problem: Area of triangle ABC
theorem area_of_triangle (a b c : ℝ) (cosB : ℝ) (dotProductAB_BC : ℝ) (sinB : ℝ) 
    (h_cosB : cosB = 3 / 5) (h_dotProductAB_BC : dotProductAB_BC = -21)
    (h_sinB : sinB = Real.sqrt (1 - cosB ^ 2)) :
    1 / 2 * a * c * sinB = 14 := by
  sorry

-- New condition for second proof problem
def a : ℝ := 7

-- Second proof problem: Angle C
theorem angle_C (a b c : ℝ) (cosB : ℝ) (dotProductAB_BC : ℝ) (sinB : ℝ) (a : ℝ) 
    (h_cosB : cosB = 3 / 5) (h_dotProductAB_BC : dotProductAB_BC = -21)
    (h_a : a = 7) (h_sinB : sinB = Real.sqrt (1 - cosB ^ 2))
    (b : ℝ) (sinC : ℝ) (h_b_eq : b = Real.sqrt (a * a + c * c - 2 * a * c * cosB))
    (h_sinC : sinC = c * sinB / b) :
    Real.arcsin sinC = π / 4 := by
  sorry

end area_of_triangle_angle_C_l706_706640


namespace decreasing_interval_l706_706776

-- Define the inner function t as a function of x
def t (x : ℝ) : ℝ := x^2 - 5 * x - 6

-- Define the outer function y as a function of t
def y (t : ℝ) : ℝ := 2^t

-- Define conditions for the problem
def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂ ∈ I, x₁ < x₂ → f x₁ > f x₂

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂ ∈ I, x₁ < x₂ → f x₁ < f x₂

-- Given conditions
axiom increasing_y : is_increasing y Set.univ
axiom decreasing_t : is_decreasing t (Set.Iio (5 / 2))

-- Proof statement
theorem decreasing_interval : ∀ (x₁ x₂ : ℝ), x₁ ∈ (Set.Iio (5 / 2)) → y (t x₁) > y (t x₂) := by
  intros x₁ x₂ hx₁ hx₂ h
  -- Add proof steps here
  sorry

end decreasing_interval_l706_706776


namespace quadrilateral_area_proof_l706_706705

variables (A B C D M : Point)
variables (angle_DAB angle_ABC angle_BCD : ℝ)
variables (distance_BM distance_DM : ℝ)

def angle_conditions (∠DAB ∠ABC ∠BCD : ℝ) : Prop :=
  ∠DAB = 60 ∧ ∠ABC = 90 ∧ ∠BCD = 120
  
def distance_conditions (BM DM : ℝ) : Prop :=
  BM = 1 ∧ DM = 2

noncomputable def area_of_quadrilateral (A B C D : Point) : ℝ := sorry

theorem quadrilateral_area_proof
  (h_angles : angle_conditions angle_DAB angle_ABC angle_BCD)
  (h_distances : distance_conditions distance_BM distance_DM)
  (h_intersection : intersection A C B D = M) :
  area_of_quadrilateral A B C D = 9 / 2 :=
sorry

end quadrilateral_area_proof_l706_706705


namespace brother_books_total_l706_706867

-- Define the conditions
def sarah_paperbacks : ℕ := 6
def sarah_hardbacks : ℕ := 4
def brother_paperbacks : ℕ := sarah_paperbacks / 3
def brother_hardbacks : ℕ := 2 * sarah_hardbacks

-- Define the statement to be proven
theorem brother_books_total : brother_paperbacks + brother_hardbacks = 10 :=
by
  -- Proof will be added here
  sorry

end brother_books_total_l706_706867


namespace collinear_F_D_E_G_l706_706862

open EuclideanGeometry

variables {A B C H M D E F G : Point}
variables [Nontrivial (AffineSpace ℝ (euclidean_space (fin 3)))]

/-- H is the orthocenter of the acute triangle ABC --/
def orthocenter (H : Point) (A B C : Triangle) : Prop :=
  ∃ F G, altitude A B C F ∧ altitude A B C G ∧ F ∈ Line B C ∧ G ∈ Line C B ∧ H = PointCommon F G

/-- Circle centered at H is tangent to BC at M --/
def tangent_circle (H M B C : Point) (r : ℝ) : Prop :=
  Circle H r = tangentAt M (Line B C)

theorem collinear_F_D_E_G (triangle_ABC  : Triangle) (H circle_center: Point) 
                         (M tangent_point : Point) (B D D_tangent : Point) 
                         (C E E_tangent: Point) 
                         (altitude_CF altitude_BG : Line) 
                         (F G altitude_points : Point)
                         (h1 : orthocenter H triangle_ABC)
                         (h2 : tangent_circle H M B C r)
                         (h3 : tangent_at_point_of_circle tangent_point (Circle H r) (Line B C) M)
                         (h4 : tangent_at_point_of_circle D_tangent (Circle H r) (Line B D) D)
                         (h5 : tangent_at_point_of_circle E_tangent (Circle H r) (Line C E) E)
                         (h6 : altitude_from_point A B C F)
                         (h7 : altitude_from_point A B C G) :
  is_collinear (Line F D E G) :=
sorry

end collinear_F_D_E_G_l706_706862


namespace cost_of_shoes_correct_l706_706854

def budget := 200
def spent_shirt := 30
def spent_pants := 46
def spent_coat := 38
def spent_socks := 11
def spent_belt := 18
def remaining := 16
def cost_shoes := 41
def spent_other_items := spent_shirt + spent_pants + spent_coat + spent_socks + spent_belt

theorem cost_of_shoes_correct :
  cost_shoes + spent_other_items + remaining = budget :=
begin
  -- omitted proof
  sorry
end

end cost_of_shoes_correct_l706_706854


namespace neg_p_lambda_range_l706_706269

theorem neg_p_lambda_range (λ : ℝ) :
  (¬ ∃ x₀ ∈ set.Ioi 0, x₀^2 - λ * x₀ + 1 < 0) ↔ λ ≤ 2 :=
by
  sorry

end neg_p_lambda_range_l706_706269


namespace part_a_part_b_part_c_part_d_l706_706940

variable (M : Set ℝ)

axiom cond1 : (0 ∈ M ∧ 1 ∈ M)
axiom cond2 : ∀ x y ∈ M, (x - y) ∈ M
axiom cond3 : ∀ x ∈ M, x ≠ 0 → (1 / x) ∈ M

theorem part_a : (1 / 3) ∈ M :=
sorry

theorem part_b : (-1) ∈ M :=
sorry

theorem part_c : ∀ x y ∈ M, (x + y) ∈ M :=
sorry

theorem part_d : ∀ x ∈ M, (x^2) ∈ M :=
sorry

end part_a_part_b_part_c_part_d_l706_706940


namespace lowest_score_equals_57_l706_706746

theorem lowest_score_equals_57 (scores : List ℝ)
  (h1 : scores.length = 15)
  (h2 : (scores.sum / scores.length : ℝ) = 85)
  (highest : ℝ)
  (lowest : ℝ)
  (h3 : highest = 100)
  (h4 : scores.max = some highest)
  (h5 : scores.min = some lowest)
  (new_scores := (scores.erase highest).erase lowest)
  (h6 : new_scores.length = 13)
  (h7 : (new_scores.sum / new_scores.length : ℝ) = 86) :
  lowest = 57 :=
sorry

end lowest_score_equals_57_l706_706746


namespace base_b_of_256_has_4_digits_l706_706424

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l706_706424


namespace ratio_of_inscribed_circle_radii_in_rhombus_and_triangle_l706_706056

theorem ratio_of_inscribed_circle_radii_in_rhombus_and_triangle
  (a α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : ∀ (A B C D : ℝ), A = a ∧ B = a ∧ C = a ∧ D = a) -- This represents the property of the rhombus where all sides are equal
  (h3 : ∀ (ABC : ℝ), ABC = 2 * a * cos (α / 2)) -- Length of the longer diagonal in terms of α and a
  : 
  let r1 := a * sin α / 2 in 
  let r2 := (a * sin α) / (2 * (1 + cos (α / 2))) in
  (r1 / r2) = 1 + cos(α / 2) :=
by
  sorry

end ratio_of_inscribed_circle_radii_in_rhombus_and_triangle_l706_706056


namespace unique_fraction_property_l706_706517

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l706_706517


namespace problem_proof_l706_706185

def f (x : ℝ) : ℝ :=
if x > 1 then log 2 x else (1/2) ^ x

theorem problem_proof : f (f (-1/2)) = 1/2 :=
by
  sorry

end problem_proof_l706_706185


namespace part1_part2_l706_706606

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Part 1: Prove the range of k such that f(x) < k * x for all x
theorem part1 (k : ℝ) : (∀ x : ℝ, x > 0 → f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
by sorry

-- Part 2: Define the function g(x) = f(x) - k * x and prove the range of k for which g(x) has two zeros in the interval [1/e, e^2]
noncomputable def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : (∃ x1 x2 : ℝ, 1 / Real.exp 1 ≤ x1 ∧ x1 ≤ Real.exp 2 ∧
                                 1 / Real.exp 1 ≤ x2 ∧ x2 ≤ Real.exp 2 ∧
                                 g x1 k = 0 ∧ g x2 k = 0 ∧ x1 ≠ x2)
                               ↔ 2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
by sorry

end part1_part2_l706_706606


namespace sum_of_six_digit_palindrome_digits_l706_706879

def is_palindrome (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n in
  digits = digits.reverse

def six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ is_palindrome n

theorem sum_of_six_digit_palindrome_digits :
  (∑ n in Finset.filter six_digit_palindrome (Finset.range 1000000), n).digits.sum = 54 := 
sorry

end sum_of_six_digit_palindrome_digits_l706_706879


namespace problem_l706_706945

theorem problem (x : ℝ) (hx : x + x⁻¹ = 3) : x^(3/2) + x^(-3/2) = Real.sqrt 5 :=
sorry

end problem_l706_706945


namespace all_positive_rationals_are_represented_l706_706497

-- Define the sequence according to the given recurrence relations
noncomputable def a : ℕ → ℕ
| 0       := 1
| (2*n+1) := a n
| (2*n+2) := a n + a (n + 1)

-- Define the theorem that needs to be proven
theorem all_positive_rationals_are_represented :
  ∀ q : ℚ, 0 < q → ∃ n : ℕ, (a (n + 1) : ℚ) / (a n) = q :=
sorry

end all_positive_rationals_are_represented_l706_706497


namespace shortest_wire_around_poles_l706_706002

open Real

noncomputable def length_of_shortest_wire (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let diff_r := r2 - r1
  let dist_centers := r1 + r2
  let straight_length := 2 * sqrt(dist_centers ^ 2 - diff_r ^ 2)
  let curved_length_small := (ℕ.toReal 60 / ℕ.toReal 360) * 2 * π * r1
  let curved_length_large := (ℕ.toReal 120 / ℕ.toReal 360) * 2 * π * r2
  straight_length + curved_length_small + curved_length_large

theorem shortest_wire_around_poles :
  length_of_shortest_wire 8 24 = 16 * sqrt 3 + 28 * π / 3 :=
by
  sorry

end shortest_wire_around_poles_l706_706002


namespace max_value_OP_OQ_l706_706663

def circle_1_polar_eq (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

def circle_2_polar_eq (rho theta : ℝ) : Prop :=
  rho = 2 * Real.sin theta

theorem max_value_OP_OQ (alpha : ℝ) :
  (∃ rho1 rho2 : ℝ, circle_1_polar_eq rho1 alpha ∧ circle_2_polar_eq rho2 alpha) ∧
  (∃ max_OP_OQ : ℝ, max_OP_OQ = 4) :=
sorry

end max_value_OP_OQ_l706_706663


namespace distinct_roots_l706_706699

noncomputable def roots (a b c : ℝ) := ((b^2 - 4 * a * c) ≥ 0) ∧ ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) * Real.sqrt (b^2 - 4 * a * c)) ≠ (0 : ℝ)

theorem distinct_roots{ p q r s : ℝ } (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) 
(h5 : q ≠ s) (h6 : r ≠ s)
(h_roots_1 : roots 1 (-12*p) (-13*q))
(h_roots_2 : roots 1 (-12*r) (-13*s)) : 
(p + q + r + s = 2028) := sorry

end distinct_roots_l706_706699


namespace tangent_line_circle_l706_706592

theorem tangent_line_circle (a b : ℝ) :
  (∃ P : ℝ × ℝ, P = (-1, 2) ∧ (ax + by - 3 = 0) ∧ (x^2 + y^2 + 4x - 1 = 0)) ∧
  (∃ C : ℝ × ℝ, C = (-2, 0) ∧ dist C (ax + by - 3 = 0) = sqrt 5) →
  a * b = 2 :=
by sorry

end tangent_line_circle_l706_706592


namespace isosceles_triangle_perimeter_l706_706225

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 3) (h2 : b = 1) :
  (a = 3 ∧ b = 1) ∧ (a + b > b ∨ b + b > a) → a + a + b = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l706_706225


namespace room_length_l706_706773

theorem room_length (L : ℝ) (width : ℝ := 4) (total_cost : ℝ := 20900) (rate : ℝ := 950) :
  L * width = total_cost / rate → L = 5.5 :=
by
  sorry

end room_length_l706_706773


namespace number_of_rational_coefficient_terms_l706_706006

open Nat

def is_rational_coefficient (k : ℕ) : Prop :=
  k % 4 = 0 ∧ (988 - k) % 2 = 0

def rational_coefficient_terms_count : ℕ :=
  (Finset.range 989).filter is_rational_coefficient |>.card

theorem number_of_rational_coefficient_terms :
  rational_coefficient_terms_count = 248 :=
by
  sorry

end number_of_rational_coefficient_terms_l706_706006


namespace greatest_x_is_53_l706_706387

-- Define the polynomial expression
def polynomial (x : ℤ) : ℤ := x^2 + 2 * x + 13

-- Define the condition for the expression to be an integer
def isIntegerWhenDivided (x : ℤ) : Prop := (polynomial x) % (x - 5) = 0

-- Define the theorem to prove the greatest integer value of x
theorem greatest_x_is_53 : ∃ x : ℤ, isIntegerWhenDivided x ∧ (∀ y : ℤ, isIntegerWhenDivided y → y ≤ x) ∧ x = 53 :=
by
  sorry

end greatest_x_is_53_l706_706387


namespace calc_expression_correct_solve_quadratic_eq_correct_l706_706018

-- Definition for problem 1
def calc_expression : ℝ :=
  real.sqrt 12 + real.sqrt 27 / 9 - real.sqrt (1 / 3)

-- Theorem for problem 1
theorem calc_expression_correct : calc_expression = 2 * real.sqrt 3 :=
by sorry

-- Theorem for problem 2
theorem solve_quadratic_eq_correct (x : ℝ) : x^2 - 4 * x + 1 = 0 ↔ (x = 2 + real.sqrt 3 ∨ x = 2 - real.sqrt 3) :=
by sorry

end calc_expression_correct_solve_quadratic_eq_correct_l706_706018


namespace prove_parabola_and_area_l706_706590

variables (C1 C2 : Type) (equation_C2 : (ℝ → ℝ → Prop)) (focus : ℝ × ℝ)
variables (vertex : ℝ × ℝ) (M : ℝ × ℝ) (line_l : ℝ → ℝ × ℝ → Prop)
variables (A B : ℝ × ℝ)
variables (area_ABO : ℝ)

noncomputable def parabola_equation : Prop :=
  ∃ y : ℝ, y^2 = 4 * x

noncomputable def minimum_area_ABO : Prop :=
  ∀ k : ℝ, k ≠ 0 → (let y := k * (x - 4) in Δ = ky^2 - 4y - 16k) → 16 + 64k^2 > 0 →
  (S := 1/2 * |M.1| * |y1 - y2|, S = 2 * sqrt (16/k^2 + 64)) → 
  area_ABO = 16

theorem prove_parabola_and_area :
  (∃ focus : ℝ × ℝ, focus = (1, 0)) ∧
  vertex = (0, 0) ∧ M = (4, 0) ∧
  line_l = (λ k (x, y) := y = k * (x - 4)) ∧
  ∃ A B : ℝ × ℝ, parabola_equation ∧ minimum_area_ABO :=
sorry

end prove_parabola_and_area_l706_706590


namespace odd_function_f_value_odd_function_f_expression_l706_706268

theorem odd_function_f_value :
  (f : ℝ → ℝ) →
  (∀ x, f (-x) = -f x) →
  (∀ x, 0 < x → f x = x^2 + 2 * x - 1) →
  f (-2) = -7 :=
begin
  sorry,
end

noncomputable def piecewise_function (f : ℝ → ℝ) : ℝ → ℝ :=
λ x, if x < 0 then -x^2 + 2 * x + 1 else if x = 0 then 0 else x^2 + 2 * x - 1

theorem odd_function_f_expression :
  (f : ℝ → ℝ) →
  (∀ x, f (-x) = -f x) →
  (∀ x, 0 < x → f x = x^2 + 2 * x - 1) →
  f = piecewise_function f :=
begin
  sorry,
end

end odd_function_f_value_odd_function_f_expression_l706_706268


namespace solve_for_f2_l706_706149

def f : ℝ → ℝ := sorry

theorem solve_for_f2 (h : ∀ x : ℝ, f(x + 1) = x) : f 2 = 1 :=
by
  sorry

end solve_for_f2_l706_706149


namespace pipe_c_empty_time_l706_706414

theorem pipe_c_empty_time :
  (1 / 45 + 1 / 60 - x = 1 / 40) → (1 / x = 72) :=
by
  sorry

end pipe_c_empty_time_l706_706414


namespace standard_form_equation_range_of_y_over_x_l706_706580

variable (θ : ℝ) (x y : ℝ)

def parametric_curve (θ : ℝ) : Prop :=
  x = -2 + Real.cos θ ∧ y = Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem standard_form_equation (h : parametric_curve θ) : (x + 2)^2 + y^2 = 1 := 
sorry

theorem range_of_y_over_x (h : parametric_curve θ) : 
  -Real.sqrt 3 / 3 ≤ y / x ∧ y / x ≤ Real.sqrt 3 / 3 :=
sorry

end standard_form_equation_range_of_y_over_x_l706_706580


namespace problem_solution_l706_706469

-- Define the conditions for each event
def event1 : Prop := ¬(A_hits_9_rings ∧ A_hits_8_rings)
def event2 : Prop := A_hits_10_rings ∧ B_hits_9_rings
def event3 : Prop := ¬(A_hits_target ∧ B_hits_target) ∧ ¬(¬A_hits_target ∧ ¬B_hits_target)
def event4 : Prop := A_hits_target ∨ B_hits_target ∧ ¬(A_hits_target ∧ ¬B_hits_target)

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

-- Define independence
def independent (e1 e2 : Prop) : Prop := (e1 → ¬e2) ∧ (e2 → ¬e1)

-- The theorem statement based on the given problem
theorem problem_solution :
  (mutually_exclusive event1 event2 ∧ mutually_exclusive event1 event3 ∧ independent event2) :=
by
  sorry

end problem_solution_l706_706469


namespace cost_split_equally_l706_706000

variable (t d : ℕ)

theorem cost_split_equally : 
  let tom_paid := 105
  let dorothy_paid := 125
  let sammy_paid := 175
  let total_cost := 405
  let each_should_pay := 135
  let t := each_should_pay - tom_paid
  let d := each_should_pay - dorothy_paid
  t - d = 20 :=
by
  have t_eq : t = 30 := by sorry
  have d_eq : d = 10 := by sorry
  show t - d = 20 from by sorry

end cost_split_equally_l706_706000


namespace angle_AFE_l706_706661

  theorem angle_AFE (A B C D E F : Type)
    [square ABCD] (h1 : angle C D E = 120) (h2 : lies_on_diagonal A C F)
    (h3 : distance D E = distance D F) : angle A F E = 30 :=
  sorry
  
end angle_AFE_l706_706661


namespace distinct_pos_ints_ineq_l706_706694

theorem distinct_pos_ints_ineq (a : ℕ → ℕ) (h1 : ∀ i j, i ≠ j → a i ≠ a j) (h2 : ∀ k, 1 ≤ a k) :
  ∀ n > 0, ∑ k in Finset.range n, (a k : ℝ) / k^2 ≥ ∑ k in Finset.range n, 1 / k :=
begin
  sorry
end

end distinct_pos_ints_ineq_l706_706694


namespace range_of_a_l706_706322

theorem range_of_a 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x, f x = -x^2 + 2*(a - 1)*x + 2)
  (increasing_on : ∀ x < 4, deriv f x > 0) : a ≥ 5 :=
sorry

end range_of_a_l706_706322


namespace find_n_l706_706772

-- Given conditions:
-- 1. Minimum value of the quadratic y = ax^2 + bx + c is -6 at x = -2.
-- 2. The graph passes through point (0, 20).
-- 3. The graph passes through the point (-3, n).

def quadratic_min_value (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + c ≥ -6

def graph_pass_through_point_1 (a b c : ℝ) : Prop :=
  (0, 20) ∈ set_of (λ p : ℝ × ℝ, p.2 = a * p.1^2 + b * p.1 + c)

def graph_pass_through_point_2 (a b c n : ℝ) : Prop :=
  (-3, n) ∈ set_of (λ p : ℝ × ℝ, p.2 = a * p.1^2 + b * p.1 + c)

theorem find_n (a b c n : ℝ) (h1 : quadratic_min_value a b c) (h2 : graph_pass_through_point_1 a b c) :
  graph_pass_through_point_2 a b c n → n = 0.5 :=
sorry

end find_n_l706_706772


namespace new_person_weight_l706_706757

theorem new_person_weight (average_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) 
  (total_increase  : average_increase = 1.8)
  (number_of_people : num_people = 6)
  (weight_of_replaced_person : weight_replaced = 69) :
  let W_new := weight_replaced + average_increase * num_people in
  W_new = 79.8 :=
by
  sorry

end new_person_weight_l706_706757


namespace jane_quadratic_solutions_eq_l706_706685

-- Define the solutions to Lauren's equation
def lauren_solutions : set ℝ := { x | |x - 4| = 3 }

-- Define Jane's quadratic equation
def jane_quadratic (b c : ℝ) (x : ℝ) : Prop := x^2 + b*x + c = 0

-- Define the correctness condition to prove that (b, c) = (-8, 7)
theorem jane_quadratic_solutions_eq (b c : ℝ) :
  (∀ x, x ∈ lauren_solutions ↔ jane_quadratic b c x) ↔ (b = -8 ∧ c = 7) :=
by
  sorry

end jane_quadratic_solutions_eq_l706_706685


namespace probability_diff_by_2_l706_706366

open Finset

def bamboo_poles : Finset ℕ := {1, 2, 3, 4}

def valid_combinations := (bamboo_poles.powerset.filter (λ s, s.card = 2))

def count_diff_by_2 := (valid_combinations.filter (λ s, abs (s.erase (s.min' begin
  apply Fintype.exists_min_begin
end)).min' begin
  apply Fintype.exists_min_begin
end - s.min' begin
  apply Fintype.exists_min_begin
end) = 2)).card

theorem probability_diff_by_2 : (count_diff_by_2 : ℚ) / valid_combinations.card = 1 / 3 := by
  sorry

end probability_diff_by_2_l706_706366


namespace triangle_ABC_conditions_l706_706913

noncomputable def coordinates_C (A : ℝ × ℝ) (median_CM : ℝ → ℝ → Prop) (altitude_BH : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

noncomputable def equation_BC (A : ℝ × ℝ) (C : ℝ × ℝ) (median_CM : ℝ → ℝ → Prop) (altitude_BH : ℝ → ℝ → Prop) : ℝ → ℝ → Prop := sorry

theorem triangle_ABC_conditions :
  let A : ℝ × ℝ := (5, 1),
      median_CM := λ x y, 2*x - y - 5 = 0,
      altitude_BH := λ x y, x - 2*y - 5 = 0,
      C := coordinates_C A median_CM altitude_BH,
      BC := equation_BC A C median_CM altitude_BH in
  C = (4, 3) ∧ (∀ x y, BC x y ↔ 6*x - 5*y - 9 = 0) :=
by
  have C_value : coordinates_C (5, 1) (λ x y, 2*x - y - 5 = 0) (λ x y, x - 2*y - 5 = 0) = (4, 3) := sorry,
  have BC_value : equation_BC (5, 1) (4, 3) (λ x y, 2*x - y - 5 = 0) (λ x y, x - 2*y - 5 = 0) = (λ x y, 6*x - 5*y - 9 = 0) := sorry,
  exact ⟨C_value, λ x y, BC_value x y⟩

end triangle_ABC_conditions_l706_706913


namespace range_function_l706_706805

def function (x : ℝ) : ℝ := (x^2 + 3 * x + 2) / (x + 1)

theorem range_function :
  set.range function = { y : ℝ | y ≠ 1 } :=
sorry

end range_function_l706_706805


namespace largest_divisor_of_n_l706_706015

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l706_706015


namespace pages_with_same_units_digit_count_l706_706435

def same_units_digit (x : ℕ) (y : ℕ) : Prop :=
  x % 10 = y % 10

theorem pages_with_same_units_digit_count :
  ∃! (n : ℕ), n = 12 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ 61) → same_units_digit x (62 - x) → 
  (x % 10 = 2 ∨ x % 10 = 7) :=
by
  sorry

end pages_with_same_units_digit_count_l706_706435


namespace residue_of_neg_1001_mod_37_l706_706887

theorem residue_of_neg_1001_mod_37 : (-1001 : ℤ) % 37 = 35 :=
by
  sorry

end residue_of_neg_1001_mod_37_l706_706887


namespace snowfall_difference_l706_706868

-- Defining all conditions given in the problem
def BaldMountain_snowfall_meters : ℝ := 1.5
def BillyMountain_snowfall_meters : ℝ := 3.5
def MountPilot_snowfall_centimeters : ℝ := 126
def RockstonePeak_snowfall_millimeters : ℝ := 5250
def SunsetRidge_snowfall_meters : ℝ := 2.25

-- Conversion constants
def meters_to_centimeters : ℝ := 100
def millimeters_to_centimeters : ℝ := 0.1

-- Converting snowfall amounts to centimeters
def BaldMountain_snowfall_centimeters : ℝ := BaldMountain_snowfall_meters * meters_to_centimeters
def BillyMountain_snowfall_centimeters : ℝ := BillyMountain_snowfall_meters * meters_to_centimeters
def RockstonePeak_snowfall_centimeters : ℝ := RockstonePeak_snowfall_millimeters * millimeters_to_centimeters
def SunsetRidge_snowfall_centimeters : ℝ := SunsetRidge_snowfall_meters * meters_to_centimeters

-- Defining total combined snowfall
def combined_snowfall_centimeters : ℝ :=
  BillyMountain_snowfall_centimeters + MountPilot_snowfall_centimeters + RockstonePeak_snowfall_centimeters + SunsetRidge_snowfall_centimeters

-- Stating the proof statement
theorem snowfall_difference :
  combined_snowfall_centimeters - BaldMountain_snowfall_centimeters = 1076 := 
  by
    sorry

end snowfall_difference_l706_706868


namespace compare_M_N_l706_706691

variable (a : ℝ)

def M : ℝ := 2 * a^2 - 4 * a
def N : ℝ := a^2 - 2 * a - 3

theorem compare_M_N : M a > N a := by
  sorry

end compare_M_N_l706_706691


namespace championship_outcomes_l706_706367

theorem championship_outcomes (students championships : ℕ)
  (students_eq : students = 8)
  (championships_eq : championships = 3) :
  students ^ championships = 512 :=
by 
  rw [students_eq, championships_eq]
  simp
  norm_num
  sorry

end championship_outcomes_l706_706367


namespace travel_to_shore_prob_l706_706861

noncomputable def probability_of_survival (p : ℝ) (q : ℝ) : ℝ :=
  q / (1 - p * q)

theorem travel_to_shore_prob (p : ℝ := 0.5) (q : ℝ := 1 - p) :
  probability_of_survival p q = 2 / 3 :=
by
  -- Given conditions
  have hp : p = 0.5 := rfl
  have hq : q = 0.5 := by simp [q, hp]
  have h_prob : probability_of_survival p q = q / (1 - p * q) := rfl
  -- Perform the calculation
  sorry

end travel_to_shore_prob_l706_706861


namespace improper_lineups_4_soldiers_improper_lineups_5_soldiers_l706_706447

-- Define the problem for n = 4
def improper_lineups_n4 : ℕ := 10

theorem improper_lineups_4_soldiers :
  ∀ (n : ℕ), n = 4 → 
  (improper_lineups 4 = improper_lineups_n4) :=
by {
  intros,
  unfold improper_lineups_n4,
  sorry
}

-- Define the problem for n = 5
def improper_lineups_n5 : ℕ := 32

theorem improper_lineups_5_soldiers :
  ∀ (n : ℕ), n = 5 → 
  (improper_lineups 5 = improper_lineups_n5) :=
by {
  intros,
  unfold improper_lineups_n5,
  sorry
}

end improper_lineups_4_soldiers_improper_lineups_5_soldiers_l706_706447


namespace minimum_m_value_l706_706932

theorem minimum_m_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  : (∀ (m : ℝ), (
      log 2004 (b / a) + log 2004 (c / b) + log 2004 (d / c) 
      ≥ m * log 2004 (d / a)) → 9 ≤ m) :=
by {
  sorry
}

end minimum_m_value_l706_706932


namespace product_of_consecutive_integers_even_l706_706732

theorem product_of_consecutive_integers_even (n : ℤ) : Even (n * (n + 1)) :=
sorry

end product_of_consecutive_integers_even_l706_706732


namespace simplify_log_expression_l706_706306

variable {p q r s t z : ℝ}
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hz : 0 < z)

theorem simplify_log_expression (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hz : 0 < z) :
  log (p / q) + log (q / r) + log (r / s) - log (pt / (sz)) = log (z / t) :=
by sorry

end simplify_log_expression_l706_706306


namespace travel_time_on_third_day_l706_706869

-- Definitions based on conditions
def speed_first_day : ℕ := 5
def time_first_day : ℕ := 7
def distance_first_day : ℕ := speed_first_day * time_first_day

def speed_second_day_part1 : ℕ := 6
def time_second_day_part1 : ℕ := 6
def distance_second_day_part1 : ℕ := speed_second_day_part1 * time_second_day_part1

def speed_second_day_part2 : ℕ := 3
def time_second_day_part2 : ℕ := 3
def distance_second_day_part2 : ℕ := speed_second_day_part2 * time_second_day_part2

def distance_second_day : ℕ := distance_second_day_part1 + distance_second_day_part2
def total_distance_first_two_days : ℕ := distance_first_day + distance_second_day

def total_distance : ℕ := 115
def distance_third_day : ℕ := total_distance - total_distance_first_two_days

def speed_third_day : ℕ := 7
def time_third_day : ℕ := distance_third_day / speed_third_day

-- The statement to be proven
theorem travel_time_on_third_day : time_third_day = 5 := by
  sorry

end travel_time_on_third_day_l706_706869


namespace intersection_A_B_l706_706578

def A := {x : ℝ | (x - 1) * (x - 4) < 0}
def B := {x : ℝ | x <= 2}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x <= 2} :=
sorry

end intersection_A_B_l706_706578


namespace find_f_x_sq_minus_2_l706_706276

-- Define the polynomial and its given condition
def f (x : ℝ) : ℝ := sorry  -- f is some polynomial, we'll leave it unspecified for now

-- Assume the given condition
axiom f_condition : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6 * x^2 + 4

-- Prove the desired result
theorem find_f_x_sq_minus_2 (x : ℝ) : f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
sorry

end find_f_x_sq_minus_2_l706_706276


namespace xy_value_l706_706978

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 :=
by
  sorry

end xy_value_l706_706978


namespace solve_inequality_l706_706738

theorem solve_inequality (x : ℝ) :
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  x ∈ set.Ioo (x) (-∞, 1) ∪ (1, 2) ∪ (2, 4) ∪ (4, 5) ∪ (7, ∞) :=
by sorry

end solve_inequality_l706_706738


namespace angle_bisector_theorem_l706_706852

noncomputable def angle_bisector_length (a b : ℝ) (C : ℝ) (CX : ℝ) : Prop :=
  C = 120 ∧
  CX = (a * b) / (a + b)

theorem angle_bisector_theorem (a b : ℝ) (C : ℝ) (CX : ℝ) :
  angle_bisector_length a b C CX :=
by
  sorry

end angle_bisector_theorem_l706_706852


namespace ceil_neg3_7_plus_1_2_l706_706545

theorem ceil_neg3_7_plus_1_2 : ⌈-3.7 + 1.2⌉ = -2 := sorry

end ceil_neg3_7_plus_1_2_l706_706545


namespace rope_length_l706_706010

theorem rope_length (x S : ℝ) (H1 : x + 7 * S = 140)
(H2 : x - S = 20) : x = 35 := by
sorry

end rope_length_l706_706010


namespace sequence_sum_l706_706179

-- Definition of the arithmetic sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0 else 2 * n - 5

-- Definition of the geometric sequence b_n
def b (n : ℕ) : ℤ :=
  3 ^ (n - 1)

-- Sum of the first n terms of the sequence a_n + b_n
def T (n : ℕ) : ℤ :=
  (n^2 - 4*n) + (3^n - 1) / 2

-- Theorem to be proved
theorem sequence_sum (n : ℕ) : 
  (finset.range n).sum (λ k, a (k + 1) + b (k + 1)) = T n :=
sorry

end sequence_sum_l706_706179


namespace haley_final_cost_l706_706204

def total_shirt_cost_kept (all_shirts : List ℕ) (returned_shirts : List ℕ) : ℕ :=
  (all_shirts.filter (λ p, ¬ returned_shirts.contains p)).sum

def final_cost (total_cost : ℕ) (discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_price := total_cost * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

theorem haley_final_cost :
  let all_shirts := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
  let returned_shirts := [20, 25, 30, 22, 23, 29]
  let discount_rate := 0.10
  let tax_rate := 0.08
  let kept_cost := total_shirt_cost_kept all_shirts returned_shirts
  final_cost kept_cost discount_rate tax_rate = 82.62 :=
by
  sorry

end haley_final_cost_l706_706204


namespace complex_product_magnitude_l706_706549

-- definitions of complex magnitudes and complex product magnitudes
def magnitude (z : Complex) : Real :=
  match z with
  | ⟨a, b⟩ => Real.sqrt (a * a + b * b)

noncomputable def product_magnitude (z1 z2 : Complex) : Real :=
  magnitude z1 * magnitude z2

-- Given complex numbers
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- The proof statement
theorem complex_product_magnitude : magnitude (z1 * z2) = Real.sqrt 7085 :=
  by 
  sorry

end complex_product_magnitude_l706_706549


namespace contrapositive_even_statement_l706_706759

-- Translate the conditions to Lean 4 definitions
def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem contrapositive_even_statement (a b : Int) :
  (¬ is_even (a + b) → ¬ (is_even a ∧ is_even b)) ↔ 
  (is_even a ∧ is_even b → is_even (a + b)) :=
by sorry

end contrapositive_even_statement_l706_706759


namespace incorrect_equation_is_wrong_l706_706655

-- Specifications and conditions
def speed_person_a : ℝ := 7
def speed_person_b : ℝ := 6.5
def head_start : ℝ := 5

-- Define the time variable
variable (x : ℝ)

-- The correct equation based on the problem statement
def correct_equation : Prop := speed_person_a * x - head_start = speed_person_b * x

-- The incorrect equation to prove incorrect
def incorrect_equation : Prop := speed_person_b * x = speed_person_a * x - head_start

-- The Lean statement to prove that the incorrect equation is indeed incorrect
theorem incorrect_equation_is_wrong (h : correct_equation x) : ¬ incorrect_equation x := by
  sorry

end incorrect_equation_is_wrong_l706_706655


namespace treasure_chest_coins_l706_706678

theorem treasure_chest_coins (hours : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ) :
  hours = 8 → coins_per_hour = 25 → total_coins = hours * coins_per_hour → total_coins = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end treasure_chest_coins_l706_706678


namespace axis_of_symmetry_cosine_stretch_shift_l706_706742

theorem axis_of_symmetry_cosine_stretch_shift :
  ∃ (k : ℤ), let f := λ x, Real.cos (x - Real.pi / 3) in
  let g := λ x, Real.cos ((1 / 2) * x - Real.pi / 4) in
  is_axis_of_symmetry x (g x) (τ := Fractional.pi / 2) :=
sorry

end axis_of_symmetry_cosine_stretch_shift_l706_706742


namespace sum_of_coordinates_l706_706987

def g (x : ℝ) : ℝ := sorry  -- g is some function ℝ → ℝ

theorem sum_of_coordinates :
  g 4 = 8 →
  ∀ (h : ℝ → ℝ), (∀ x, h x = (g x) ^ 2 + 1) → (4 + h 4) = 69 :=
by {
  intros hg h_def,
  rw h_def 4,
  rw hg,
  norm_num,
}

end sum_of_coordinates_l706_706987


namespace find_a_l706_706959

theorem find_a : 
  ∃ a, 
    (a > 0) ∧ 
    (let P := (x₁, 2 * x₁ + 2), Q := (x₂, 2 * x₂ + 2), A := (1 / a, 1 / a^2),
         x₁x₂ := -2 / a, x₁x₂_sum := 2 / a,
  
     (| (⟨1/a, 1/a^2⟩ - ⟨x₁, 2 * x₁ + 2⟩ + ⟨1/a, 1/a^2⟩ - ⟨x₂, 2 * x₂ + 2⟩) | 
      = | (⟨1/a, 1/a^2⟩ - ⟨x₁, 2 * x₁ + 2⟩ - ⟨1/a, 1/a^2⟩ - ⟨x₂, 2 * x₂ + 2⟩)) 
      → a = 2 :=
sorry

end find_a_l706_706959


namespace exists_2011_distinct_positive_integers_l706_706535

theorem exists_2011_distinct_positive_integers : 
  ∃ S : Finset ℕ, S.card = 2011 ∧ (∀ a b ∈ S, a ≠ b → |a - b| = Nat.gcd a b) :=
by
  sorry

end exists_2011_distinct_positive_integers_l706_706535


namespace sum_of_sums_of_three_element_subsets_l706_706277

open Set Finset

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem sum_of_sums_of_three_element_subsets :
  let three_element_subsets := A.powerset.filter (λ s, s.card = 3)
  let subsums := three_element_subsets.sum (λ s, s.sum id)
  subsums = 1980 :=
by
  sorry

end sum_of_sums_of_three_element_subsets_l706_706277


namespace smallest_consecutive_sum_perfect_square_l706_706351

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l706_706351


namespace isogonal_triangle_tangent_l706_706818

noncomputable theory

-- Define the problem in Lean
theorem isogonal_triangle_tangent
  (A B C P M N X Y Z : ℝ)
  (h_iso : ∀ A B C : ℝ, (BC : ℝ) = (AB : ℝ))
  (h_P : P ∈ alt B to AC)
  (h_M : M ∈ (circle A B P) ∩ AC)
  (h_AM_NC : AM = NC ∧ M ≠ N)
  (h_X : X ∈ (NP ∩ (circle A B P)) ∧ X ≠ P)
  (h_Y : Y ∈ (AB ∩ (circle A P N)) ∧ Y ≠ A)
  (h_Z : ins_tangent_from A to (circle A P N) ∩ alt B to AC)
  : CZ is tangent to (circle P X Y) := sorry

end isogonal_triangle_tangent_l706_706818


namespace wholesale_price_l706_706055

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l706_706055


namespace combined_loss_percentage_l706_706062

theorem combined_loss_percentage
  (cost_price_radio : ℕ := 8000)
  (quantity_radio : ℕ := 5)
  (discount_radio : ℚ := 0.1)
  (tax_radio : ℚ := 0.06)
  (sale_price_radio : ℕ := 7200)
  (cost_price_tv : ℕ := 20000)
  (quantity_tv : ℕ := 3)
  (discount_tv : ℚ := 0.15)
  (tax_tv : ℚ := 0.07)
  (sale_price_tv : ℕ := 18000)
  (cost_price_phone : ℕ := 15000)
  (quantity_phone : ℕ := 4)
  (discount_phone : ℚ := 0.08)
  (tax_phone : ℚ := 0.05)
  (sale_price_phone : ℕ := 14500) :
  let total_cost_price := (quantity_radio * cost_price_radio) + (quantity_tv * cost_price_tv) + (quantity_phone * cost_price_phone)
  let total_sale_price := (quantity_radio * sale_price_radio) + (quantity_tv * sale_price_tv) + (quantity_phone * sale_price_phone)
  let total_loss := total_cost_price - total_sale_price
  let loss_percentage := (total_loss * 100 : ℚ) / total_cost_price
  loss_percentage = 7.5 :=
by
  sorry

end combined_loss_percentage_l706_706062


namespace AK_bisects_BC_area_ratio_l706_706264

-- Define the basic entities
variables (A B C O D E F K M : Type) [triangle A B C]
variable [excircle O : Type]
variable [tangent_point D : Type] (hD : tangent O A B C D)
variable [tangent_point E : Type] (hE : tangent O A B C E)
variable [tangent_point F : Type] (hF : tangent O A B C F)
variable [intersection_point K : Type] (hK : intersection_point D E F O K)

-- Condition 1: AK bisects BC
theorem AK_bisects_BC (AK : A K) (BM MC : A B C)
  (BC : BM + MC = BC) : BM = MC :=
sorry

-- Variables for areas and sides
variables (a b c : ℝ) (S_ABC S_BKC : ℝ) [b > c]
variable [triangle_area S_ABC : Type] (hS_ABC : area S_ABC A B C)
variable [triangle_area S_BKC : Type] (hS_BKC : area S_BKC B K C)

-- Condition 2: Area ratio
theorem area_ratio (BC : a) (AC : b) (AB : c) : 
  S_ABC / S_BKC = (b + c) / a :=
sorry

end AK_bisects_BC_area_ratio_l706_706264


namespace permutation_sum_bound_l706_706704

theorem permutation_sum_bound (n : ℕ) 
  (x : Fin n → ℝ) 
  (h_sum_abs : abs (Finset.univ.sum x) = 1)
  (h_bound : ∀ i, abs (x i) ≤ (n + 1) / 2) :
  ∃ y : Fin n → ℝ, 
    (∃ p : Equiv.Perm (Fin n), ∀ i, y i = x (p i)) ∧ 
    abs (Finset.univ.sum (λ i, (i.succ : ℕ) * y i)) ≤ (n + 1) / 2 := 
sorry

end permutation_sum_bound_l706_706704


namespace round_table_vip_arrangements_l706_706237

-- Define the conditions
def number_of_people : ℕ := 10
def vip_seats : ℕ := 2

noncomputable def number_of_arrangements : ℕ :=
  let total_arrangements := Nat.factorial number_of_people
  let vip_choices := Nat.choose number_of_people vip_seats
  let remaining_arrangements := Nat.factorial (number_of_people - vip_seats)
  vip_choices * remaining_arrangements

-- Theorem stating the result
theorem round_table_vip_arrangements : number_of_arrangements = 1814400 := by
  sorry

end round_table_vip_arrangements_l706_706237


namespace avg_annual_growth_rate_optimal_selling_price_l706_706538

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end avg_annual_growth_rate_optimal_selling_price_l706_706538


namespace minimum_cuts_to_divide_cube_l706_706382

open Real

theorem minimum_cuts_to_divide_cube (a b : ℕ) (ha : a = 4) (hb : b = 64) : 
  (∃ n : ℕ, 2^n = b ∧ n = log 64 / log 2) :=
begin
  use 6,
  split,
  { norm_num, },
  { norm_num, },
end

end minimum_cuts_to_divide_cube_l706_706382


namespace smallest_sum_of_consecutive_integers_is_square_l706_706332

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l706_706332


namespace percentage_increase_l706_706317

theorem percentage_increase (x : ℝ) (h1 : 75 + 0.75 * x * 0.8 = 72) : x = 20 :=
by
  sorry

end percentage_increase_l706_706317


namespace parallel_AE_DO_l706_706289

variables {A B C D E O : Type} [geometry A B C D E O]

-- Define the conditions
variables (h1 : Point D ∈ Segment A C)
          (h2 : 2 * Distance D C = Distance A D)
          (O_is_incenter : is_incenter O (Triangle B D C))
          (E_is_tangent : is_tangent_point E (incircle O (Triangle B D C)) (Line B D))
          (h3 : Distance B D = Distance B C)

-- Define the theorem statement
theorem parallel_AE_DO (h1 : Point D ∈ Segment A C)
                       (h2 : 2 * Distance D C = Distance A D)
                       (O_is_incenter : is_incenter O (Triangle B D C))
                       (E_is_tangent : is_tangent_point E (incircle O (Triangle B D C)) (Line B D))
                       (h3 : Distance B D = Distance B C) :
  parallel (Line A E) (Line D O) :=
by sorry

end parallel_AE_DO_l706_706289


namespace number_of_correct_relations_l706_706075

theorem number_of_correct_relations :
  (∃ (a b : Type), {a, b} ⊆ {b, a}) ∧
  (∀ (a b : Type), {a, b} = {b, a}) ∧
  (0 ≠ (∅ : Set)) ∧
  (0 ∈ ({0} : Set)) ∧
  (∅ ∉ ({0} : Set)) ∧
  (∅ ⊆ ({0} : Set)) → 4 :=
begin
  sorry

end number_of_correct_relations_l706_706075


namespace probability_X_equals_1_l706_706912

variable {X : ℕ} -- X is a random variable that takes non-negative integer values

noncomputable def binomial_probability (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_X_equals_1 :
  binomial_probability 4 (1/3) 1 = 32 / 81 :=
by sorry

end probability_X_equals_1_l706_706912


namespace initial_volume_of_mixture_l706_706232

theorem initial_volume_of_mixture (milk water : ℕ) (x : ℕ) (hx : 4 * x = milk) (hratio : milk / (water + 11) = 1.8) (initial_ratio : milk = 4 * water) : 
  milk + water = 45 := 
by 
  sorry

end initial_volume_of_mixture_l706_706232


namespace find_c_l706_706533

theorem find_c (x : ℝ) : (x^2 > 7^(8^9)) ↔ (x > real.sqrt(7^(8^9))) :=
sorry

end find_c_l706_706533


namespace conor_potatoes_per_day_l706_706877

theorem conor_potatoes_per_day :
  ∃ P : ℕ, 4 * (12 + 9 + P) = 116 :=
by {
  use 8,
  sorry
}

end conor_potatoes_per_day_l706_706877


namespace unique_fraction_property_l706_706514

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l706_706514


namespace average_output_l706_706406

theorem average_output (t1 t2 t_total : ℝ) (c1 c2 c_total : ℕ) 
                        (h1 : c1 = 60) (h2 : c2 = 60) 
                        (rate1 : ℝ := 15) (rate2 : ℝ := 60) :
  t1 = c1 / rate1 ∧ t2 = c2 / rate2 ∧ t_total = t1 + t2 ∧ c_total = c1 + c2 → 
  (c_total / t_total = 24) := 
by 
  sorry

end average_output_l706_706406


namespace min_value_of_f_l706_706392

noncomputable def f (x : ℝ) : ℝ := 4 * x + 4 / x

theorem min_value_of_f (x : ℝ) (hx : x > 0) : ∃ y, f x = y ∧ (∀ z, (z > 0) → f z ≥ y) :=
begin
  use f 1,
  split,
  { sorry },
  { intro z,
    sorry }
end

end min_value_of_f_l706_706392


namespace integer_solution_pairs_l706_706894

theorem integer_solution_pairs (a b : ℕ) (h_pos : a > 0 ∧ b > 0):
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, l > 0 ∧ ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
sorry

end integer_solution_pairs_l706_706894


namespace wolf_never_catches_any_sheep_l706_706769

-- Define an infinite plane and positions of pieces
structure Position where
  x : ℝ
  y : ℝ

-- Define movement rules for pieces
def move (pos : Position) (dx dy : ℝ) : Position :=
  ⟨pos.x + dx, pos.y + dy⟩

-- Define the game conditions
structure Game where
  wolf : Position
  sheep : Fin 50 → Position
  wolf_move : ∀ t : ℕ, Position
  sheep_move : ∀ t : ℕ, Fin 50 → Position

axiom init_wolf_pos : ∃ p : Position, Game.wolf_move 0 = p
axiom init_sheep_pos : ∀ i : Fin 50, ∃ p : Position, Game.sheep_move 0 i = p

axiom wolf_move_rule : ∀ t : ℕ, ∥Game.wolf_move (t + 1) - Game.wolf_move t∥ ≤ 1
axiom sheep_move_rule : ∀ t : ℕ, ∀ i : Fin 50, ∥Game.sheep_move (t + 1) i - Game.sheep_move t i∥ ≤ 1

-- Define the reachability or capture condition
def capture (wolf sheep : Position) : Prop :=
  ∥wolf - sheep∥ ≤ 0

-- Prove that the wolf does not catch any sheep
theorem wolf_never_catches_any_sheep :
  ¬∃ t : ℕ, ∃ i : Fin 50, capture (Game.wolf_move t) (Game.sheep_move t i) :=
sorry -- proof omitted

end wolf_never_catches_any_sheep_l706_706769


namespace probability_segments_at_least_100cm_l706_706836

noncomputable def probability_of_usable_segments : ℝ := by 
  let total_area := (400 * 400) / 2 -- Area of the original triangle
  let feasible_area := (200 * 200) / 2 -- Area of the feasible region
  let probability := feasible_area / total_area
  exact probability

-- Theorem stating the probability of all three segments being at least 1 meter long is 1/4
theorem probability_segments_at_least_100cm :
  probability_of_usable_segments = 1 / 4 := 
by 
  sorry

end probability_segments_at_least_100cm_l706_706836


namespace unique_paths_in_modified_hexagonal_lattice_system_l706_706995

theorem unique_paths_in_modified_hexagonal_lattice_system :
  let red_arrows := 1,
      paths_to_blue_from_first_red := 2,
      paths_to_blue_from_second_red := 4,
      total_paths_to_blue := paths_to_blue_from_first_red + paths_to_blue_from_second_red,
      paths_to_each_green_from_blue := 3,
      total_paths_to_each_green := total_paths_to_blue * paths_to_each_green_from_blue,
      paths_to_orange_from_each_green := 2,
      total_paths_to_orange := total_paths_to_each_green * paths_to_orange_from_each_green,
      paths_to_B_from_orange := 2,
      total_paths_to_B := total_paths_to_orange * paths_to_B_from_orange
   in total_paths_to_B = 144 :=
by
  let red_arrows := 1
  let paths_to_blue_from_first_red := 2
  let paths_to_blue_from_second_red := 4
  let total_paths_to_blue := paths_to_blue_from_first_red + paths_to_blue_from_second_red
  let paths_to_each_green_from_blue := 3
  let total_paths_to_each_green := total_paths_to_blue * paths_to_each_green_from_blue
  let paths_to_orange_from_each_green := 2
  let total_paths_to_orange := total_paths_to_each_green * paths_to_orange_from_each_green
  let paths_to_B_from_orange := 2
  let total_paths_to_B := total_paths_to_orange * paths_to_B_from_orange
  exact eq_refl 144

end unique_paths_in_modified_hexagonal_lattice_system_l706_706995


namespace fixed_point_C_D_intersection_l706_706587

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ P.2 ≠ 0

noncomputable def line_CD_fixed_point (t : ℝ) (C D : ℝ × ℝ) : Prop :=
  let x1 := (36 - 2 * t^2) / (18 + t^2)
  let y1 := (12 * t) / (18 + t^2)
  let x2 := (2 * t^2 - 4) / (2 + t^2)
  let y2 := -(4 * t) / (t^2 + 2)
  C = (x1, y1) ∧ D = (x2, y2) →
  let k_CD := (4 * t) / (6 - t^2)
  ∀ (x y : ℝ), y + (4 * t) / (t^2 + 2) = k_CD * (x - (2 * t^2 - 4) / (t^2 + 2)) →
  y = 0 → x = 1

theorem fixed_point_C_D_intersection :
  ∀ (t : ℝ) (C D : ℝ × ℝ), point_on_line (4, t) →
  ellipse_equation C.1 C.2 →
  ellipse_equation D.1 D.2 →
  line_CD_fixed_point t C D :=
by
  intros t C D point_on_line_P ellipse_C ellipse_D
  sorry

end fixed_point_C_D_intersection_l706_706587


namespace smallest_positive_integer_l706_706007

theorem smallest_positive_integer (n : ℕ) (h1 : 0 < n) (h2 : ∃ k1 : ℕ, 3 * n = k1^2) (h3 : ∃ k2 : ℕ, 4 * n = k2^3) : 
  n = 54 := 
sorry

end smallest_positive_integer_l706_706007


namespace derivative_at_one_eq_one_l706_706761

open Real

def func (x : ℝ) : ℝ := (log x) / x

theorem derivative_at_one_eq_one : deriv func 1 = 1 :=
by
  sorry

end derivative_at_one_eq_one_l706_706761


namespace solve_equations_l706_706905

theorem solve_equations :
  (∃ x : ℝ, (x + 2) ^ 3 + 1 = 0 ∧ x = -3) ∧
  (∃ x : ℝ, ((3 * x - 2) ^ 2 = 64 ∧ (x = 10/3 ∨ x = -2))) :=
by {
  -- Prove the existence of solutions for both problems
  sorry
}

end solve_equations_l706_706905


namespace Eunji_total_wrong_questions_l706_706110

theorem Eunji_total_wrong_questions 
  (solved_A : ℕ) (solved_B : ℕ) (wrong_A : ℕ) (right_diff : ℕ) 
  (h1 : solved_A = 12) 
  (h2 : solved_B = 15) 
  (h3 : wrong_A = 4) 
  (h4 : right_diff = 2) :
  (solved_A - (solved_A - (solved_A - wrong_A) + right_diff) + (solved_A - wrong_A) + right_diff - solved_B - (solved_B - (solved_A - (solved_A - wrong_A) + right_diff))) = 9 :=
by {
  sorry
}

end Eunji_total_wrong_questions_l706_706110


namespace wholesale_price_is_90_l706_706050

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l706_706050


namespace smallest_sum_of_consecutive_integers_is_square_l706_706354

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l706_706354


namespace sum_inverses_eq_one_mod17_l706_706089

theorem sum_inverses_eq_one_mod17 :
  let inv_mod := fun (x : ℕ) (p : ℕ) => x^(p-2) % p in
  (inv_mod 3 17 + inv_mod (3^2) 17 + inv_mod (3^3) 17 + inv_mod (3^4) 17 + inv_mod (3^5) 17 + inv_mod (3^6) 17) % 17 = 1 :=
by
  sorry

end sum_inverses_eq_one_mod17_l706_706089


namespace series_sum_l706_706094

theorem series_sum :
  let T : ℝ := ∑ n in finset.range 100, (3 + (n + 1) * 9) / 9^(101 - n)
  in T = 112.75 :=
by
  let T : ℝ := ∑ n in finset.range 100, (3 + (n + 1) * 9) / 9^(101 - n)
  show T = 112.75
  sorry

end series_sum_l706_706094


namespace determine_function_l706_706531

-- Define the function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
def condition1 : Prop :=
  (∃ S : Set ℝ, Finite S ∧ ∀ x ∈ ℝ, x ≠ 0 → f x / x ∈ S)

def condition2 : Prop :=
  ∀ x : ℝ, f (x - 1 - f x) = f x - x - 1

-- The theorem statement
theorem determine_function (h1 : condition1 f) (h2 : condition2 f) : 
  ∀ x : ℝ, f x = x := 
sorry

end determine_function_l706_706531


namespace unit_trip_to_expo_l706_706067

theorem unit_trip_to_expo (n : ℕ) (cost : ℕ) (total_cost : ℕ) :
  (n ≤ 30 → cost = 120) ∧ 
  (n > 30 → cost = 120 - 2 * (n - 30) ∧ cost ≥ 90) →
  (total_cost = 4000) →
  (total_cost = n * cost) →
  n = 40 :=
by
  sorry

end unit_trip_to_expo_l706_706067


namespace select_at_least_two_blue_bikes_l706_706731

def comb (n k : ℕ) : ℕ := nat.choose n k

theorem select_at_least_two_blue_bikes : 
  ∃ (n : ℕ), n = comb 4 4 + (comb 4 3 * comb 6 1) + (comb 4 2 * comb 6 2) ∧ n = 115 :=
by
  use comb 4 4 + (comb 4 3 * comb 6 1) + (comb 4 2 * comb 6 2)
  split
  · reflexivity
  · reflexivity

end select_at_least_two_blue_bikes_l706_706731


namespace ratio_of_lateral_edges_l706_706760

theorem ratio_of_lateral_edges (A B : ℝ) (hA : A > 0) (hB : B > 0) (h : A / B = 4 / 9) : 
  let upper_length_ratio := 2
  let lower_length_ratio := 3
  upper_length_ratio / lower_length_ratio = 2 / 3 :=
by 
  sorry

end ratio_of_lateral_edges_l706_706760


namespace greatest_average_speed_l706_706085

noncomputable def initial_odometer_reading : ℕ := 12321
noncomputable def maximum_speed_limit : ℕ := 80
noncomputable def driving_time : ℕ := 4

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the maximum possible distance April could travel
noncomputable def maximum_possible_distance : ℕ :=
  maximum_speed_limit * driving_time

-- Possible palindromes after the initial odometer reading
def next_palindromes : list ℕ :=
  [12421, 12521, 12621]

-- Define a function to calculate the distance traveled
noncomputable def distance_traveled (start end_ : ℕ) : ℕ :=
  end_ - start

-- Define a function to calculate average speed
noncomputable def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- Condition: End odometer reading is a palindrome and within the maximum distance
theorem greatest_average_speed :
  ∃ (end_odometer : ℕ), is_palindrome end_odometer ∧
  distance_traveled initial_odometer_reading end_odometer ≤ maximum_possible_distance ∧
  average_speed (distance_traveled initial_odometer_reading end_odometer) driving_time = 75 :=
begin
  use 12621,
  split,
  { refl, -- 12621 is_palindrome },
  split,
  { norm_num, -- distance_traveled ≤ maximum_possible_distance },
  { norm_num, -- average_speed = 75 }
end

end greatest_average_speed_l706_706085


namespace lincoln_county_final_houses_l706_706643

def initial_houses : ℕ := 20817

def houses_after_year_1 (h : ℕ) : ℕ := 
  let built := rounded (0.15 * h)
  let demolished := rounded (0.03 * h)
  h + built - demolished

def houses_after_year_2 (h : ℕ) : ℕ := 
  let built := rounded (0.12 * h)
  let demolished := rounded (0.05 * h)
  h + built - demolished

def houses_after_year_3 (h : ℕ) : ℕ := 
  let built := rounded (0.10 * h)
  let demolished := rounded (0.04 * h)
  h + built - demolished

def houses_after_year_4 (h : ℕ) : ℕ := 
  let built := rounded (0.08 * h)
  let demolished := rounded (0.06 * h)
  h + built - demolished

def houses_after_year_5 (h : ℕ) : ℕ := 
  let built := rounded (0.05 * h)
  let demolished := rounded (0.02 * h)
  h + built - demolished

noncomputable def total_houses_after_5_years : ℕ := 
  let after_1 := houses_after_year_1 initial_houses
  let after_2 := houses_after_year_2 after_1
  let after_3 := houses_after_year_3 after_2
  let after_4 := houses_after_year_4 after_3
  houses_after_year_5 after_4

open Nat

def rounded (x : ℚ) : ℕ :=
  round (x : ℝ)

theorem lincoln_county_final_houses : 
  total_houses_after_5_years = 27783 :=
by sorry

end lincoln_county_final_houses_l706_706643


namespace no_solns_to_equation_l706_706118

noncomputable def no_solution : Prop :=
  ∀ (n m r : ℕ), (1 ≤ n) → (1 ≤ m) → (1 ≤ r) → n^5 + 49^m ≠ 1221^r

theorem no_solns_to_equation : no_solution :=
sorry

end no_solns_to_equation_l706_706118


namespace central_angle_of_sector_l706_706314

noncomputable def area_of_sector (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * Real.pi * (r ^ 2)

theorem central_angle_of_sector :
  ∃ θ : ℝ, area_of_sector θ 12 = 49.02857142857143 ∧ θ ≈ 39 :=
by
  sorry

end central_angle_of_sector_l706_706314


namespace ceiling_eval_l706_706548

theorem ceiling_eval : Int.ceil (-3.7 + 1.2) = -2 := by
  sorry

end ceiling_eval_l706_706548


namespace problem_1_problem_2_l706_706267

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.log x / Real.log 3)^2 + (a - 1) * (Real.log x / Real.log 3) + 3 * a - 2

theorem problem_1 (a : ℝ) :
  (∀ y : ℝ, ∃ x > 0, f(a, x) = y) ↔ a = 7 + 4 * Real.sqrt 2 ∨ a = 7 - 4 * Real.sqrt 2 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x ∈ Set.Icc (3 : ℝ) 9, f(a, 3 * x) + Real.log (9 * x) / Real.log 3 ≤ 0) ↔ a ≤ -4 / 3 :=
sorry

end problem_1_problem_2_l706_706267


namespace concert_duration_l706_706254

-- Define the given conditions
def intermission : ℕ := 10
def songs_duration (n : ℕ) : ℕ := if n = 13 then 10 else if n < 13 then 5 else 0
def num_songs : ℕ := 13

-- Prove that the total concert duration is 80 minutes
theorem concert_duration : (intermission + (∑ n in finset.range num_songs, songs_duration (n + 1))) = 80 :=
by
  sorry

end concert_duration_l706_706254


namespace max_planes_from_points_l706_706863

theorem max_planes_from_points (points : Finset ℝ^3) (h_card : points.card = 15) 
  (h_no_three_collinear : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬Collinear ℝ [p1, p2, p3])
  (h_no_four_coplanar : ∀ (p1 p2 p3 p4 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → ¬Coplanar ℝ [p1, p2, p3, p4]) : 
  (Finset.univ.choose 3).card = 455 := 
by
  sorry

end max_planes_from_points_l706_706863


namespace initial_amount_l706_706109

-- Define the given conditions
def amount_spent : ℕ := 16
def amount_left : ℕ := 2

-- Define the statement that we want to prove
theorem initial_amount : amount_spent + amount_left = 18 :=
by
  sorry

end initial_amount_l706_706109


namespace smallest_sum_of_consecutive_integers_is_square_l706_706335

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l706_706335


namespace simplify_to_x5_l706_706393

theorem simplify_to_x5 (x : ℝ) :
  x^2 * x^3 = x^5 :=
by {
  -- proof goes here
  sorry
}

end simplify_to_x5_l706_706393


namespace family_impossible_l706_706718

def family_possible (n : Nat) (k : Nat) (h : Nat) : Prop :=
  n = 9 ∧ k = 5 ∧ h = 3 ∧ 
  let total_handshakes := (n * h) / 2 
  in total_handshakes ∈ Nat

theorem family_impossible : ¬ family_possible 9 5 3 :=
by
  sorry

end family_impossible_l706_706718


namespace log_4_135_eq_half_log_2_45_l706_706946

noncomputable def a : ℝ := Real.log 135 / Real.log 4
noncomputable def b : ℝ := Real.log 45 / Real.log 2

theorem log_4_135_eq_half_log_2_45 : a = b / 2 :=
by
  sorry

end log_4_135_eq_half_log_2_45_l706_706946


namespace leak_drains_tank_in_5_hours_l706_706403

-- Definitions based on the problem conditions
def rate_of_pump : ℝ := 1 / 4  -- The pump's rate in tanks per hour
def rate_with_leak : ℝ := 1 / 20  -- The combined rate of pump and leak in tanks per hour

theorem leak_drains_tank_in_5_hours :
  (rate_of_pump - rate_with_leak) = 1 / 5 := by
  -- Here would be the proof steps, which we omit with sorry.
  sorry

end leak_drains_tank_in_5_hours_l706_706403


namespace total_goals_l706_706684

-- Define constants for goals scored in respective seasons
def goalsLastSeason : ℕ := 156
def goalsThisSeason : ℕ := 187

-- Define the theorem for the total number of goals
theorem total_goals : goalsLastSeason + goalsThisSeason = 343 :=
by
  -- Proof is omitted
  sorry

end total_goals_l706_706684


namespace input_for_output_32_l706_706881

theorem input_for_output_32 (x : ℝ) : 
  (let step1 := x - 8 in
   let step2 := step1 / 2 in
   let output := step2 + 16 in
   output = 32) ↔ x = 40 :=
by 
  sorry

end input_for_output_32_l706_706881


namespace locus_of_midpoints_is_line_l706_706124

noncomputable
def locus_of_midpoints_parallel_segments (l1 l2 : Line) (Pi : Plane) (h_l1_parallel_Pi : parallel l1 Pi) (h_l2_parallel_Pi : parallel l2 Pi) : Set Point :=
  {M | ∃ P Q : Point, P ∈ l1 ∧ Q ∈ l2 ∧ segment P Q ∥ Pi ∧ M = midpoint P Q}

theorem locus_of_midpoints_is_line (l1 l2 : Line) (Pi : Plane) (h_l1_parallel_Pi : parallel l1 Pi) (h_l2_parallel_Pi : parallel l2 Pi) :
  ∃ L : Line, ∀ M, M ∈ locus_of_midpoints_parallel_segments l1 l2 Pi h_l1_parallel_Pi h_l2_parallel_Pi → M ∈ L :=
  sorry

end locus_of_midpoints_is_line_l706_706124


namespace probability_two_books_l706_706659

theorem probability_two_books (A B C D : Type) (books : Finset (Set Type)) (books_card : books.card = 4) :
  let event := ({A, B} : Set Type)
  P(event ∈ books.powerset.filter (λ s, s.card = 2)) = 1 / 6 := 
sorry

end probability_two_books_l706_706659


namespace train_pass_time_correct_l706_706065

noncomputable def relative_speed (train_speed man_speed : ℝ) : ℝ :=
  train_speed + man_speed

noncomputable def convert_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def time_to_pass (distance speed : ℝ) : ℝ :=
  distance / speed

theorem train_pass_time_correct :
  time_to_pass 85
    (convert_kmph_to_mps (relative_speed 90 12)) ≈ 3 :=
by
  sorry

end train_pass_time_correct_l706_706065


namespace find_value_l706_706917

axiom m n : ℝ
axiom m_ne_n : m ≠ n
axiom eq1 : m + 1/m = -4
axiom eq2 : n + 1/n = -4

theorem find_value : m * (n + 1) + n = -3 := 
by 
  sorry

end find_value_l706_706917


namespace angle_B_and_ratio_range_l706_706595

theorem angle_B_and_ratio_range {a b c : ℝ} 
  (h_area : (√3 / 4) * (a^2 + c^2 - b^2) = (1/2) * a * c * sin (real.pi / 3)) 
  (h_C_obtuse : ∃ (C : ℝ), C > real.pi / 2 ∧ C < real.pi ∧ real.cos C = (a^2 + c^2 - b^2) / (2 * a * c)) :
  (∀ (B : ℝ), B = real.pi / 3) ∧ (∀ (ratio : ℝ), 2 < (c / a) ∧ (c / a) < +∞) :=
by
  sorry

end angle_B_and_ratio_range_l706_706595


namespace perfect_square_or_cube_divisors_l706_706969

def is_perfect_square (a b : ℕ) : Prop := (a % 2 = 0) ∧ (b % 2 = 0)
def is_perfect_cube (a b : ℕ) : Prop := (a % 3 = 0) ∧ (b % 3 = 0)
def is_perfect_sixth_power (a b : ℕ) : Prop := (a % 6 = 0) ∧ (b % 6 = 0)

def num_perfect_square_divisors (range : ℕ) : ℕ := (range // 2 + 1) * (range // 2 + 1)
def num_perfect_cube_divisors (range : ℕ) : ℕ := (range // 3 + 1) * (range // 3 + 1)
def num_perfect_sixth_power_divisors (range : ℕ) : ℕ := (range // 6 + 1) * (range // 6 + 1)

theorem perfect_square_or_cube_divisors :
  num_perfect_square_divisors 7 + num_perfect_cube_divisors 7 - num_perfect_sixth_power_divisors 7 = 21 :=
by sorry

end perfect_square_or_cube_divisors_l706_706969


namespace cake_division_l706_706872

noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem cake_division (A B C : ℝ × ℝ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) :
  ∃ (P : ℝ × ℝ), P = centroid A B C ∧ ∀ (cut : ℝ × ℝ), cut.1 + cut.2 = 1 ∧ (P = centroid A B C → cut = centroid B C A ∨ cut = centroid A B C ∨ cut = centroid C A B) → 
  ∃ k m : ℕ, Gek_portion : ℕ, Chuck_portion : ℕ,
  ∑ i in range 4, i = Gek_portion ∧ ∑ i in range 5, i = Chuck_portion ∧ Chuck_portion = 5 ∧ Gek_portion = 4 :=
sorry

end cake_division_l706_706872


namespace alice_stops_in_quarter_D_l706_706722

-- Definitions and conditions
def indoor_track_circumference : ℕ := 40
def starting_point_S : ℕ := 0
def run_distance : ℕ := 1600

-- Desired theorem statement
theorem alice_stops_in_quarter_D :
  (run_distance % indoor_track_circumference = 0) → 
  (0 ≤ (run_distance % indoor_track_circumference) ∧ 
   (run_distance % indoor_track_circumference) < indoor_track_circumference) → 
  true := by
  sorry

end alice_stops_in_quarter_D_l706_706722


namespace symmetric_circle_intersection_angle_l706_706174

/-
    Given that the circle x^2 + y^2 + 8x - 4y = 0 is symmetric to the circle
    x^2 + y^2 = 20 with respect to the line y = kx + b,
    
    Prove:
    1. The values of k and b are 2 and 5 respectively.
    2. The degree measure of ∠AOB at the intersection points of the two circles is 120°.
-/

theorem symmetric_circle_intersection_angle :
  (∃ k b : ℝ, k = 2 ∧ b = 5 ∧ 
    let A := (-4, 2 : ℝ × ℝ),
        O := (0, 0 : ℝ × ℝ) in
    let θ := 120 in
    let R1 := 20 in
    let R2 := 20 in
    let slope_perpendicular_bisector := 2 in
    let y_line := 2 * x + 5 in  -- define the line equation
    ∃ (A B : ℝ × ℝ), -- intersection points
      angle (A,O,B) θ) := sorry

end symmetric_circle_intersection_angle_l706_706174


namespace f_one_value_l706_706570

def f (x a: ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem f_one_value (a : ℝ) (h : ∀ x, f x a ≥ 0) : f 1 a = 4 :=
by
  sorry

end f_one_value_l706_706570


namespace ratio_new_values_l706_706817

theorem ratio_new_values (x y x2 y2 : ℝ) (h1 : x / y = 7 / 5) (h2 : x2 = x * y) (h3 : y2 = y * x) : x2 / y2 = 1 := by
  sorry

end ratio_new_values_l706_706817


namespace number_of_factors_of_product_l706_706566

theorem number_of_factors_of_product (p1 p2 p3 p4 : ℕ)
  (h1 : Nat.Prime p1)
  (h2 : Nat.Prime p2)
  (h3 : Nat.Prime p3)
  (h4 : Nat.Prime p4)
  (hp : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  let a := p1^2
  let b := p2^2
  let c := p3^2
  let d := p4^2
  let n := a^3 * b^2 * c^4 * d^5
  Nat.factors_count n = 3465 :=
  by
    sorry

end number_of_factors_of_product_l706_706566


namespace min_sin_angle_A_l706_706639

-- Define the conditions and prove the minimum value of sin ∠A
theorem min_sin_angle_A (AB AC : ℝ) (h1 : AB + AC = 7) (h2 : (1 / 2) * AB * AC * (Real.sin (angle_of A B C)) = 4) :
  (∀ a b c : ℝ, (1 / 2) * a * b * Real.sin c = 4 → a + b = 7 → c ≥ Real.arcsin (32 / 49)) :=
by
  sorry

end min_sin_angle_A_l706_706639


namespace sum_fraction_l706_706095

theorem sum_fraction (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) :
  (∑ (a b c : ℕ) in (finset.Icc 1 ⊤).filter (λ t, (t.1 < t.2 ∧ t.2 < t.3 : Prop)), 
   (1 : ℚ) / (2^t.1 * 4^t.2 * 6^t.3)) = 1 / 318188825 :=
sorry

end sum_fraction_l706_706095


namespace volume_of_water_displaced_l706_706027

-- Define the cylindrical barrel
def cylindrical_barrel (r h : ℝ) := { radius := r, height := h }

-- Define the cube
def cube (a : ℝ) := { side_length := a }

-- Define volume calculation for the displaced water
def volume_displaced (b : { radius : ℝ × height : ℝ }, c : { side_length : ℝ }) : ℝ :=
  let d := c.side_length * Real.sqrt 3
  let s := b.radius * Real.sqrt 3
  let h := d / 2
  let base_area := (Real.sqrt 3 / 4) * s^2
  (1 / 3) * base_area * h

-- Theorem to prove the volume of water displaced
theorem volume_of_water_displaced :
  volume_displaced (cylindrical_barrel 5 12) (cube 6) = 168.75 := by
  sorry

end volume_of_water_displaced_l706_706027


namespace length_of_AB_l706_706152

theorem length_of_AB :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}
  let focus := (Real.sqrt 3, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - Real.sqrt 3}
  ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line ∧ B ∈ line ∧
  (dist A B = 8 / 5) :=
by
  sorry

end length_of_AB_l706_706152


namespace like_terms_solution_l706_706215

theorem like_terms_solution:
  ∀ (x y: ℝ), 3 * x = 2 - 4 * y ∧ y + 5 = 2 * x → x * y = -2 :=
by
  intros x y
  intro h
  cases h
  have : 3 * x = 2 - 4 * y := h_left
  have : y + 5 = 2 * x := h_right
  sorry

end like_terms_solution_l706_706215


namespace medians_inequality_l706_706299

variables A B C : Type
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (triangle : A → B → C → Prop)
noncomputable def semi_perimeter (p : ℝ) := p
noncomputable def median (m : ℝ) := m

theorem medians_inequality (m_a m_b m_c : ℝ) (p : ℝ)
  (triangle_medians : median m_a ∧ median m_b ∧ median m_c)
  (semi_perimeter_p : semi_perimeter p) :
  2 > (m_a + m_b + m_c) / p ∧ (m_a + m_b + m_c) / p > 1.5 :=
by sorry

end medians_inequality_l706_706299


namespace find_f_prime_at_1_l706_706950

def M : ℝ := 1
def tangent_line (x : ℝ) : ℝ := (1/2) * x + 3
def f (x : ℝ) : ℝ := sorry -- f is unknown

theorem find_f_prime_at_1 (h : ∀ x, tangent_line x = (1/2) * x + 3) : 
  Deriv f 1 = 1/2 :=
by
  sorry

end find_f_prime_at_1_l706_706950


namespace maximum_daily_sales_revenue_l706_706327

noncomputable def sales_price (t : ℕ) : ℕ :=
if h : 0 < t ∧ t < 25 then t + 20
else if h : 25 ≤ t ∧ t ≤ 30 then -t + 100
else 0

def daily_sales_volume (t : ℕ) : ℕ :=
if h : 0 < t ∧ t ≤ 30 then -t + 40
else 0

def daily_sales_revenue (t : ℕ) : ℕ :=
sales_price t * daily_sales_volume t

theorem maximum_daily_sales_revenue :
  ∃ t, 25 ≤ t ∧ t ≤ 30 ∧ daily_sales_revenue t = 1125 :=
sorry

end maximum_daily_sales_revenue_l706_706327


namespace composite_num_div_factorial_l706_706297

theorem composite_num_div_factorial {n : ℕ} (h1 : n > 4) (h2 : ¬ (nat.prime n)) : n ∣ nat.factorial (n - 1) := 
sorry

end composite_num_div_factorial_l706_706297


namespace BMN_is_equilateral_l706_706288

variables {A B C A₁ C₁ M N : Point}

noncomputable theory

def is_midpoint (P Q R : Point) := dist P Q = dist P R

def is_equilateral (P Q R : Point) := dist P Q = dist Q R ∧ dist Q R = dist R P 

-- Assuming we have the following conditions:
-- 1. Points on a line: A, B, C with B between A and C
-- 2. Equilateral triangles on segments AB and BC
-- 3. A₁ and C₁ are on the same side of line AB
-- 4. M is midpoint of AA₁
-- 5. N is midpoint of CC₁

axiom (points_linear_on : Points_Linear A B C)
axiom (B_between_A_and_C : Between B A C)
axiom (equilateral_ABC₁ : IsEquilateralTriangle A B C₁)
axiom (equilateral_BCA₁ : IsEquilateralTriangle B C A₁)
axiom (same_side_AB_A₁_C₁ : SameSide A B A₁ C₁)
axiom (M_midpoint_AA₁ : is_midpoint M A A₁)
axiom (N_midpoint_CC₁ : is_midpoint N C C₁)

theorem BMN_is_equilateral 
  (points_linear_on : Points_Linear A B C)
  (B_between_A_and_C : Between B A C)
  (equilateral_ABC₁ : IsEquilateralTriangle A B C₁)
  (equilateral_BCA₁ : IsEquilateralTriangle B C A₁)
  (same_side_AB_A₁_C₁ : SameSide A B A₁ C₁)
  (M_midpoint_AA₁ : is_midpoint M A A₁)
  (N_midpoint_CC₁ : is_midpoint N C C₁)
: is_equilateral B M N :=
sorry

end BMN_is_equilateral_l706_706288


namespace tara_dad_second_year_games_attended_l706_706889

variables (total_games_per_year : ℕ)
           (attendance_first_year_percentage : ℕ)
           (missed_games_second_year_activity : ℕ)
           (missed_games_second_year_injury : ℕ)
           (cancelled_games_second_year : ℕ)
           (rescheduled_games_second_year : ℕ)
           (missed_games_second_year_due_to_work : ℕ)
           (attend_fewer_games_second_year : ℕ)

-- Definitions of the conditions
def first_year_games_attended := (attendance_first_year_percentage * total_games_per_year) / 100

def second_year_games_can_attend :=
  total_games_per_year - missed_games_second_year_activity - missed_games_second_year_injury - cancelled_games_second_year

def second_year_games_attended :=
  first_year_games_attended - attend_fewer_games_second_year - missed_games_second_year_due_to_work

-- Theorem stating the problem
theorem tara_dad_second_year_games_attended
  (total_games_per_year = 20)
  (attendance_first_year_percentage = 90)
  (missed_games_second_year_activity = 5)
  (missed_games_second_year_injury = 2)
  (cancelled_games_second_year = 1)
  (rescheduled_games_second_year = 2)
  (missed_games_second_year_due_to_work = 3)
  (attend_fewer_games_second_year = 4) :
  second_year_games_attended = 11 :=
by
  sorry

end tara_dad_second_year_games_attended_l706_706889


namespace net_change_decrease_initial_tonnage_total_fees_l706_706025

-- Condition: Goods movement over 6 days
def goods_movement : List ℤ := [+31, -32, -16, +35, -38, -20]

-- Question 1: Has the amount of goods increased or decreased after 6 days?
theorem net_change_decrease : List.sum goods_movement = -40 :=
by
  sorry

-- Condition: Current stock after 6 days
def final_tonnage : ℕ := 460

-- Question 2: Initial amount of goods in the warehouse
theorem initial_tonnage : final_tonnage - List.sum goods_movement = 500 :=
by
  sorry

-- Condition: Loading and unloading fees per ton
def fee_per_ton : ℕ := 5

-- Question 3: Total loading and unloading fees
theorem total_fees : List.sum (List.map Int.natAbs goods_movement) * fee_per_ton = 860 :=
by
  sorry

end net_change_decrease_initial_tonnage_total_fees_l706_706025


namespace find_length_of_floor_l706_706413
open Real

-- Definition of the problem
def length_more_than_breadth_by_200pct (breadth length : ℝ) : Prop :=
  length = breadth + 2 * breadth

def total_paint_cost (area cost_per_unit : ℝ) : Prop :=
  area * cost_per_unit = 624

def floor_area (length breadth : ℝ) : ℝ :=
  length * breadth

noncomputable def length_of_floor : ℝ :=
  3 * sqrt (52)

-- Condition setup
def conditions (breadth cost_per_unit : ℝ) : Prop :=
  let length := 3 * breadth in
  length_more_than_breadth_by_200pct breadth length ∧
  total_paint_cost (floor_area length breadth) cost_per_unit 

-- Goal
theorem find_length_of_floor : ∀ breadth cost_per_unit : ℝ,
  cost_per_unit = 4 → conditions breadth cost_per_unit →
  length_of_floor ≈ 21.63 :=
by
  intros breadth cost_per_unit cost_rate cond
  sorry

end find_length_of_floor_l706_706413


namespace prove_a2_eq_neg6_l706_706929

variable (a₁ a₄ : ℤ)

-- Defining the arithmetic sequence with common difference of 3
def a (n : ℕ) : ℤ := a₁ + (n - 1) * 3

-- Specifying that a₃ is a₁ + 6 according to the arithmetic sequence
lemma a₃_def : a 3 = a₁ + 6 :=
by simp [a]

-- Specifying that a₄ is a₁ + 9 according to the arithmetic sequence
lemma a₄_def : a 4 = a₁ + 9 :=
by simp [a]

-- Defining the geometric sequence condition
lemma geo_seq_cond (h : (a 3) ^ 2 = a₁ * (a 4)) :
  (a 1 + 6)^2 = a₁ * (a₁ + 9) :=
by 
  rw [a₃_def, a₄_def] at h
  exact h

-- Proving a2 = -6 given the geometric sequence condition holds
theorem prove_a2_eq_neg6 (h : (a 1 + 6)^2 = a₁ * (a₁ + 9)) :
  a 2 = -6 :=
by 
  have a₁_val : a₁ = -8,
  { calc (a₁ + 6) ^ 2 = a₁ * (a₁ + 9) : h,
        (a₁ + 6) ^ 2 = a₁ ^ 2 + 9 * a₁ : by ring,
    by_contra h,
    have neq_zero : a₁ ≠ -8 := by intro; linarith
  },
  have a₂ : a 2 = a₁ + 3,
  { unfold a,
    rw [add_assoc],
    simp, },
  rw [a₁_val, a₂],
  simp


end prove_a2_eq_neg6_l706_706929


namespace cos_R_in_right_triangle_l706_706227

theorem cos_R_in_right_triangle (R P QR PR PQ : ℝ) : 
  P = 90 ∧ tan R = (1 / 2) ∧ PQ^2 = PR^2 + QR^2 ∧ 
  PQ = sqrt(PR^2 + QR^2) ∧ cos R = PR / PQ → 
  cos R = (2 * sqrt(5)) / 5 :=
by
  sorry

end cos_R_in_right_triangle_l706_706227


namespace females_dont_listen_l706_706063

theorem females_dont_listen:
  ∀ (males_listen females_listen total_listen total_not_listen survey_total : ℕ),
  males_listen = 75 →
  females_listen = 95 →
  total_listen = 200 →
  total_not_listen = 180 →
  survey_total = 380 →
  let males_not_listen := survey_total - total_listen - total_not_listen in
  let total_females := survey_total - males_listen - males_not_listen in
  let females_dont_listen := total_females - females_listen in
  females_dont_listen = 180 := 
by
  intros males_listen females_listen total_listen total_not_listen survey_total h_males_listen h_females_listen h_total_listen h_total_not_listen h_survey_total
  sorry

end females_dont_listen_l706_706063


namespace sum_of_digits_of_1905_l706_706059

-- Define the sequence tn recursively
def t : ℕ → ℚ
| 1       := 1
| (n + 1) := if (n + 1) % 2 = 0 then 1 + t ((n + 1) / 2) else 1 / t n

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Statement: Given tn = 19/87, prove the sum of digits of n is 15
theorem sum_of_digits_of_1905 (n : ℕ) (h : t n = 19 / 87) : sum_of_digits n = 15 := by
  sorry

end sum_of_digits_of_1905_l706_706059


namespace solve_sqrt_equation_l706_706558

theorem solve_sqrt_equation (z : ℝ) : sqrt (7 + 3 * z) = 15 ↔ z = 218 / 3 :=
by
  sorry

end solve_sqrt_equation_l706_706558


namespace isosceles_triangle_vertex_angle_l706_706895

-- Definitions based on the conditions
variables {A B C D E O : Type} [Point : AffineSpacePoint(A B C D E O)] (triangle_ABC : Triangle A B C)
variables (is_isosceles_triangle : AC = BC) (medians_A_D : is_median AD) (medians_B_E : is_median BE)
variables (perpendicular_medians : ∠AOD = 90°)

-- Statement of the problem to be proved.
theorem isosceles_triangle_vertex_angle :
  angle_at_vertex_isosceles_triangle == arctan(1/3) :=
sorry

end isosceles_triangle_vertex_angle_l706_706895


namespace daily_wage_of_c_l706_706400

theorem daily_wage_of_c
  (a b c : ℚ)
  (h_ratio : a / 3 = b / 4 ∧ a / 3 = c / 5)
  (h_total_earning : 6 * a + 9 * b + 4 * c = 1850) :
  c = 125 :=
by 
  sorry

end daily_wage_of_c_l706_706400


namespace cost_of_shoes_l706_706855

theorem cost_of_shoes 
  (budget : ℕ) (shirt : ℕ) (pants : ℕ) (coat : ℕ) (socks : ℕ) (belt : ℕ) (remaining_budget : ℕ) (shoes : ℕ) :
  budget = 200 →
  shirt = 30 →
  pants = 46 →
  coat = 38 →
  socks = 11 →
  belt = 18 →
  remaining_budget = 16 →
  shoes = 41 →
  budget - remaining_budget = (shirt + pants + coat + socks + belt + shoes) :=
begin
  sorry
end

end cost_of_shoes_l706_706855


namespace range_a_l706_706907

def f (x a : ℝ) := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem range_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l706_706907


namespace coeff_x4_polynomial_expansion_l706_706560

def polynomial_expansion := 4 * (x^4 - x^3) + 3 * (x^2 - 3 * x^4 + 2 * x^6) - (5 * x^3 - 2 * x^4)

theorem coeff_x4_polynomial_expansion : coeff (simplify polynomial_expansion) x^4 = -3 :=
by
  sorry

end coeff_x4_polynomial_expansion_l706_706560


namespace find_mass_m1_approx_l706_706827

noncomputable def find_mass_m1
  -- Given conditions
  (m2 : ℝ) (λ : ℝ) (AB : ℝ) (BC : ℝ) (BO : ℝ) (OC : ℝ)
  -- Required result (approximately)
  : ℝ :=
  
  have λ := 3,
  have AB := 7,
  have BC := 5,
  have BO := 4,
  have OC := 3,
  have m2 := 20,
  
  -- Mass of segments
  let m_AB := λ * AB,
  let m_BC := λ * BC,
  
  -- Equilibrium condition about point B
  have H : m1 * AB + m_AB * (AB / 2) = m2 * BO + m_BC * (BO / 2), 
  from sorry,
  
  -- Solve for m1
  have m1 := (m2 * BO + m_BC * (BO / 2) - m_AB * (AB / 2)) / AB,
  from (m2 * BO + m_BC * (BO / 2) - m_AB * (AB / 2)) / AB,

  m1

theorem find_mass_m1_approx : 
  find_mass_m1 20 3 7 5 4 3 ≈ 5.2 := 
sorry

end find_mass_m1_approx_l706_706827


namespace negation_of_proposition_l706_706777

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → (x + 1/x) ≥ 2

-- Define the negation of the original proposition
def negation_prop : Prop := ∃ x > 0, x + 1/x < 2

-- State that the negation of the original proposition is the stated negation
theorem negation_of_proposition : (¬ ∀ x, original_prop x) ↔ negation_prop := 
by sorry

end negation_of_proposition_l706_706777


namespace no_solution_ineq_positive_exponents_l706_706824

theorem no_solution_ineq (m : ℝ) (h : m < 6) : ¬∃ x : ℝ, |x + 1| + |x - 5| ≤ m := 
sorry

theorem positive_exponents (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) : a^a * b^b - a^b * b^a > 0 := 
sorry

end no_solution_ineq_positive_exponents_l706_706824


namespace orchids_minus_roses_l706_706371

theorem orchids_minus_roses (r o : ℕ) (h1 : r = 11) (h2 : o = 20) : o - r = 9 :=
by
  rw [h1, h2]
  exact Nat.sub_self 2

end orchids_minus_roses_l706_706371


namespace max_value_expression_l706_706266

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : c^2 = a^2 + b^2) :
  ∃ x, 2 * (a - x) * (x + sqrt (x^2 + c^2)) ≤ 2 * a^2 + b^2 :=
sorry

end max_value_expression_l706_706266


namespace simple_random_sampling_methods_proof_l706_706418

-- Definitions based on conditions
def equal_probability (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
∀ s1 s2 : samples, p s1 = p s2

-- Define that Lottery Drawing Method and Random Number Table Method are part of simple random sampling
def is_lottery_drawing_method (samples : Type) : Prop := sorry
def is_random_number_table_method (samples : Type) : Prop := sorry

def simple_random_sampling_methods (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
  equal_probability samples p ∧ is_lottery_drawing_method samples ∧ is_random_number_table_method samples

-- Statement to be proven
theorem simple_random_sampling_methods_proof (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) :
  (∀ s1 s2 : samples, p s1 = p s2) → simple_random_sampling_methods samples p :=
by
  intro h
  unfold simple_random_sampling_methods
  constructor
  exact h
  constructor
  sorry -- Proof for is_lottery_drawing_method
  sorry -- Proof for is_random_number_table_method

end simple_random_sampling_methods_proof_l706_706418


namespace similar_triangle_perimeter_l706_706847

theorem similar_triangle_perimeter 
  (a b : ℝ) 
  (hypotenuse_larger : ℝ)
  (h₁ : a = 5) 
  (h₂ : b = 12) 
  (h₃ : hypotenuse_larger = 39) 
  : 
  let hypotenuse_smaller := real.sqrt (a^2 + b^2) in
  let similarity_ratio := hypotenuse_smaller / hypotenuse_larger in
  let larger_leg1 := a * (hypotenuse_larger / hypotenuse_smaller) in
  let larger_leg2 := b * (hypotenuse_larger / hypotenuse_smaller) in
  larger_leg1 + larger_leg2 + hypotenuse_larger = 90 := 
sorry

end similar_triangle_perimeter_l706_706847


namespace shaded_fraction_of_large_square_l706_706838

theorem shaded_fraction_of_large_square :
  let large_square_area := 1
  let small_square_area := large_square_area / 4
  let shaded_square_area := small_square_area
  let shaded_triangle_area := small_square_area / 2
  let two_shaded_small_squares_area := small_square_area / 2
  shaded_square_area + shaded_triangle_area + two_shaded_small_squares_area = large_square_area / 2 :=
by
  let large_square_area := 1
  let small_square_area := large_square_area / 4
  let shaded_square_area := small_square_area
  let shaded_triangle_area := small_square_area / 2
  let two_shaded_small_squares_area := small_square_area / 2
  have h1: shaded_square_area = 1 / 4 := by sorry
  have h2: shaded_triangle_area = 1 / 8 := by sorry
  have h3: two_shaded_small_squares_area = 1 / 8 := by sorry
  have h4: shaded_square_area + shaded_triangle_area + two_shaded_small_squares_area = 1 / 4 + 1 / 8 + 1 / 8 := by
    rw [h1, h2, h3]
  have h5: 1 / 4 + 1 / 8 + 1 / 8 = 1 / 2 := by sorry
  rw [h4, h5]

end shaded_fraction_of_large_square_l706_706838


namespace magnitude_a_sub_b_l706_706617

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

def a_norm : ℝ := norm a
def b_norm : ℝ := norm b
def a_dot_a_sub_b : ℝ := inner a (a - b)

theorem magnitude_a_sub_b
  (h1 : a_norm a = sqrt 3)
  (h2 : b_norm b = 2)
  (h3 : a_dot_a_sub_b a b = 0) : norm (a - b) = 1 :=
by sorry

end magnitude_a_sub_b_l706_706617


namespace ceil_neg3_7_plus_1_2_l706_706546

theorem ceil_neg3_7_plus_1_2 : ⌈-3.7 + 1.2⌉ = -2 := sorry

end ceil_neg3_7_plus_1_2_l706_706546


namespace average_height_students_l706_706329

/-- Given the average heights of female and male students, and the ratio of men to women, the average height -/
theorem average_height_students
  (avg_female_height : ℕ)
  (avg_male_height : ℕ)
  (ratio_men_women : ℕ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 182)
  (h3 : ratio_men_women = 5) :
  (avg_female_height + 5 * avg_male_height) / (1 + 5) = 180 :=
by
  sorry

end average_height_students_l706_706329


namespace trajectory_of_point_Q_l706_706607

-- Define the variables and constants
variables {x y m n : ℝ}

-- Define points O, P, and Q
def O : (ℝ × ℝ) := (0, 0)
def Q : (ℝ × ℝ) := (x, y)
def P : (ℝ × ℝ) := (m, n)

-- Define the conditions
def line_l := 2 * m + 4 * n + 3 = 0
def collinearity := (m, n) = (3 * x, 3 * y)

-- The statement to be proved
theorem trajectory_of_point_Q : line_l → collinearity → (2 * x + 4 * y + 1 = 0) :=
by
  intro hl hc
  -- The proof will be written here eventually
  sorry

end trajectory_of_point_Q_l706_706607


namespace david_marks_in_mathematics_l706_706525

theorem david_marks_in_mathematics 
  (marks_english : ℕ := 76)
  (marks_physics : ℕ := 82)
  (marks_chemistry : ℕ := 67)
  (marks_biology : ℕ := 85)
  (average_marks : ℕ := 75)
  (marks_math : ℕ) : marks_math = 65 :=
by
  have total_marks_other := marks_english + marks_physics + marks_chemistry + marks_biology
  have total_marks := average_marks * 5
  have marks_math_calculated := total_marks - total_marks_other
  have : marks_math = 65
  guard_target = marks_math = 65
  sorry

end david_marks_in_mathematics_l706_706525


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706338

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706338


namespace unique_fraction_condition_l706_706498

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l706_706498


namespace new_profit_is_122_03_l706_706458

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end new_profit_is_122_03_l706_706458


namespace quadratic_roots_l706_706903

theorem quadratic_roots (p q r : ℝ) (h : p ≠ q) (k : ℝ) :
  (p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) →
  ((p * (q - r)) * k^2 + (q * (r - p)) * k + r * (p - q) = 0) →
  k = - (r * (p - q)) / (p * (q - r)) :=
by
  sorry

end quadratic_roots_l706_706903


namespace cos_B_in_right_triangle_l706_706239

theorem cos_B_in_right_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (angle_A : Real) (AB BC : Real) 
  (right_triangle_ABC : Triangle A B C)
  (h_angle_A : angle_A = π / 2) 
  (h_AB : AB = 16) 
  (h_BC : BC = 24) :
  ∃ AC : Real, (cos B = 3 * Real.sqrt 13 / 13) :=
by sorry

end cos_B_in_right_triangle_l706_706239


namespace area_of_smaller_circle_l706_706802

theorem area_of_smaller_circle 
  (P A A' B B' : Type) (radius_smaller radius_larger : ℝ) 
  (h1 : radius_larger = 3 * radius_smaller)
  (h2 : ∀ (x ∈ {P, A, A', B, B'}), true) -- placeholders for the points and lines
   : 
   (∃ (radius : ℝ), radius = sqrt (12/5) ∧ 
   π * radius ^ 2 = π * (12 / 5)) :=
sorry

end area_of_smaller_circle_l706_706802


namespace cos2_alpha_plus_sin2_alpha_l706_706958

theorem cos2_alpha_plus_sin2_alpha (α : ℝ) :
  (∃ (m : ℝ), x - 2 * y + 1 = 0 ∧ m = tan α) →
  cos α ^ 2 + sin (2 * α) = 8 / 5 :=
begin
  sorry
end

end cos2_alpha_plus_sin2_alpha_l706_706958


namespace abs_lt_one_sufficient_not_necessary_l706_706564

theorem abs_lt_one_sufficient_not_necessary (x : ℝ) : (|x| < 1) -> (x < 1) ∧ ¬(x < 1 -> |x| < 1) :=
by
  sorry

end abs_lt_one_sufficient_not_necessary_l706_706564


namespace relationship_among_a_b_c_l706_706965

noncomputable def a : ℝ := 0.6 ^ 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.6
noncomputable def c : ℝ := Real.log Real.pi

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l706_706965


namespace no_such_fraction_exists_l706_706504

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l706_706504


namespace min_magnitude_inequality_l706_706527

noncomputable def min (x y : ℝ) : ℝ :=
if x ≥ y then y else x

variables (a b : ℝ → ℝ → ℝ)
variables (a_ne_zero : a ≠ 0)
variables (b_ne_zero : b ≠ 0)

theorem min_magnitude_inequality (a b : ℝ → ℝ → ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  min ((∥a + b∥)^2) ((∥a - b∥)^2) ≤ (∥a∥^2 + ∥b∥^2) :=
sorry

end min_magnitude_inequality_l706_706527


namespace correct_operation_l706_706812

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^3 ≠ 2 * a^5) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  (a^3 * a^5 ≠ a^15) ∧
  ((ab^2)^2 = a^2 * b^4) :=
by
  sorry

end correct_operation_l706_706812


namespace shortest_path_no_self_intersections_l706_706147

/-- 
Given n points in the plane, the shortest polygonal chain with vertices 
at these points does not have self-intersections.
-/
theorem shortest_path_no_self_intersections {n : ℕ} (points : fin n → ℝ × ℝ) 
  (h : ∀ (p q r s : ℝ × ℝ), p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ p ∧ 
                              (p = q → r = s) → (∃ (k l : ℕ), k < l ∧ 
                              ∃ (a b c d : ℝ × ℝ), a = points k ∧ b = points (k + 1) ∧ 
                              c = points l ∧ d = points (l + 1))) :
  ¬ ∃ (k l : ℕ), k < l ∧ ∃ (a b c d : ℝ × ℝ), a = points k ∧ b = points (k + 1) ∧ 
  c = points l ∧ d = points (l + 1) ∧ segments_intersect a b c d :=
sorry

end shortest_path_no_self_intersections_l706_706147


namespace intersection_M_N_l706_706612

-- Definitions:
def M := {x : ℝ | 0 ≤ x}
def N := {y : ℝ | -2 ≤ y}

-- The theorem statement:
theorem intersection_M_N : M ∩ N = {z : ℝ | 0 ≤ z} := sorry

end intersection_M_N_l706_706612


namespace travel_distance_is_correct_l706_706890

noncomputable def total_travel_distance (P Q R : Type*) 
  [distance_PQ : has_dist P Q] 
  [distance_PR : has_dist P R] 
  [distance_QR : has_dist Q R] 
  (hP : distance_PR.dist = 4000) 
  (hQ : distance_PQ.dist = 5000) 
  (h_right_angle : dist2 Q P + dist2 Q R = dist2 P R) 
  : ℕ := 
  dist P Q + dist Q R + dist R P

theorem travel_distance_is_correct (P Q R : Type*) 
  [distance_PQ : has_dist P Q] 
  [distance_PR : has_dist P R] 
  [distance_QR : has_dist Q R] 
  (hP : distance_PR.dist = 4000) 
  (hQ : distance_PQ.dist = 5000) 
  (h_right_angle : dist2 Q P + dist2 Q R = dist2 P R) 
  : total_travel_distance P Q R = 16500 :=
sorry

end travel_distance_is_correct_l706_706890


namespace lipstick_cost_correct_l706_706086

noncomputable def cost_of_lipsticks (total_cost: ℕ) (cost_slippers: ℚ) (cost_hair_color: ℚ) (paid: ℚ) (number_lipsticks: ℕ) : ℚ :=
  (paid - (6 * cost_slippers + 8 * cost_hair_color)) / number_lipsticks

theorem lipstick_cost_correct :
  cost_of_lipsticks 6 (2.5:ℚ) (3:ℚ) (44:ℚ) 4 = 1.25 := by
  sorry

end lipstick_cost_correct_l706_706086


namespace basketball_free_throw_probability_l706_706434

-- Define the probabilities
def pFT : ℝ := 0.8
def pHS3 : ℝ := 0.5
def pPro3 : ℝ := 1 / 3
def pAtLeastOne : ℝ := 0.9333333333333333

-- Define the complement probability calculations
def pNone : ℝ := (1 - pFT) * (1 - pHS3) * (1 - pPro3)
def pComputedAtLeastOne : ℝ := 1 - pNone

theorem basketball_free_throw_probability : pFT = 0.8 :=
by
  -- We admit the proof here with "sorry", as instructed.
  sorry

end basketball_free_throw_probability_l706_706434


namespace concyclic_Q_R_S_T_l706_706474

/-- Circle Γ₁ is the circumcircle of quadrilateral ABCD,
lines AC and BD intersect at point E.
lines AD and BC intersect at point F.
Circle Γ₂ is tangent to segments EB and EC at points M and N respectively,
and intersects circle Γ₁ at points Q and R.
Lines BC and AD intersect line MN at points S and T respectively. 
We need to prove the points Q, R, S, and T are concyclic. --/
theorem concyclic_Q_R_S_T
  {Γ₁ Γ₂ : Circle}
  {A B C D E F M N Q R S T : Point}
  (h1 : Γ₁.isCircumcircleOf A B C D)
  (h2 : line A C ∩ line B D = E)
  (h3 : line A D ∩ line B C = F)
  (h4 : Γ₂.isTangentToSegment E B M)
  (h5 : Γ₂.isTangentToSegment E C N)
  (h6 : Γ₂ ∩ Γ₁ = {Q, R})
  (h7 : line B C ∩ line M N = S)
  (h8 : line A D ∩ line M N = T) :
  concyclic Q R S T :=
sorry

end concyclic_Q_R_S_T_l706_706474


namespace find_k_l706_706138

-- Define the equation of the line and the point it must contain
def line (k : ℝ) (x : ℝ) (y : ℝ) : Prop := 1 - 3 * k * x = -2 * y
def given_point : ℝ × ℝ := (1/3, 4)

theorem find_k : ∃ k : ℝ, line k (given_point.1) (given_point.2) ∧ k = 9 :=
sorry

end find_k_l706_706138


namespace unique_fraction_property_l706_706515

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l706_706515


namespace no_positive_integers_satisfy_equation_l706_706819

theorem no_positive_integers_satisfy_equation :
  ¬ ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a^2 = b^11 + 23 :=
by
  sorry

end no_positive_integers_satisfy_equation_l706_706819


namespace sum_of_palindromic_primes_below_70_l706_706129

def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def palindromic_primes_below_70 : List ℕ :=
  [11, 13, 31]

theorem sum_of_palindromic_primes_below_70 : 
  (palindromic_primes_below_70.sum = 55) :=
by
  definition is_prime (n : ℕ) : Prop :=
    n > 1 ∧ ∀ m : ℕ, m ∣ n → 1 ≤ m ∧ m ≤ n → m = 1 ∨ m = n

  definition is_palindromic_prime_below_70 (n : ℕ) : Prop :=
    n < 70 ∧ is_prime n ∧ is_palindromic n

  palindromic_primes_below_70 = {n | is_palindromic_prime_below_70 n}

  sorry -- Proof required here

end sum_of_palindromic_primes_below_70_l706_706129


namespace railway_networks_count_l706_706047

-- Define the main theorem stating that number of railway networks is 125
theorem railway_networks_count 
  (segment_count : ℕ)
  (city_count : ℕ)
  (no_collinear : ∀ (p1 p2 p3 : Point), ¬Collinear p1 p2 p3)
  (segments_intersect : ∀ (s1 s2 : Segment), Intersect s1 s2) 
  : segment_count = 4 → city_count = 5 → railway_networks = 125 :=
by
  intros h1 h2
  sorry

end railway_networks_count_l706_706047


namespace unique_fraction_condition_l706_706501

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l706_706501


namespace trains_cross_time_21_33_seconds_l706_706972

-- Define the lengths of the trains
def length_train1 : ℝ := 280
def length_train2 : ℝ := 360

-- Define the speeds of the trains in km/h
def speed_kmph_train1 : ℝ := 71
def speed_kmph_train2 : ℝ := 37

-- Convert speeds from km/h to m/s
def speed_mps_train1 : ℝ := speed_kmph_train1 * 1000 / 3600
def speed_mps_train2 : ℝ := speed_kmph_train2 * 1000 / 3600

-- Define the relative speed since the trains are moving in opposite directions
def relative_speed : ℝ := speed_mps_train1 + speed_mps_train2

-- Define the total distance to be covered
def total_distance : ℝ := length_train1 + length_train2

-- Calculate the time taken to cross each other
def time_to_cross : ℝ := total_distance / relative_speed

theorem trains_cross_time_21_33_seconds :
  time_to_cross ≈ 21.33 := sorry

end trains_cross_time_21_33_seconds_l706_706972


namespace trig_identity_pi_over_11_l706_706130

theorem trig_identity_pi_over_11 : 
  cos (π / 11) * cos (2 * π / 11) * cos (3 * π / 11) * cos (4 * π / 11) * cos (5 * π / 11) = 1 / 32 := 
sorry

end trig_identity_pi_over_11_l706_706130


namespace interval_length_f_lt_g_eq_1_l706_706882

open Int

def floor (x : ℝ) : ℤ := ⌊x⌋₊  -- Greatest integer less than or equal to x
def frac (x : ℝ) : ℝ := x - floor x  -- Fractional part of x

def f (x : ℝ) : ℝ := (floor x : ℝ) * frac x  -- Definition of function f
def g (x : ℝ) : ℝ := x - 1  -- Definition of function g

-- The length of the interval where f(x) < g(x) within the given bounds
theorem interval_length_f_lt_g_eq_1 : 
    (0 ≤ x ∧ x ≤ 3) → 
    f x < g x →
    length ([2, 3]) = 1 := sorry

end interval_length_f_lt_g_eq_1_l706_706882


namespace smallest_consecutive_sum_perfect_square_l706_706349

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l706_706349


namespace necessary_and_sufficient_condition_l706_706768

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function (c : ℝ) (x : ℝ) : ℝ := (1 / x) + c * x^2

theorem necessary_and_sufficient_condition (c : ℝ) :
  is_odd (function c) ↔ c = 0 := by
  sorry

end necessary_and_sufficient_condition_l706_706768


namespace sum_of_every_third_term_l706_706453

theorem sum_of_every_third_term (a : ℤ) (σ : ℕ → ℤ) (h_seq_def : ∀ n, σ (n + 1) = σ n + 3)
  (h_sum_3000 : (∑ i in Finset.range 3000, σ i) = 12000) :
  (∑ i in Finset.range 1000, σ (3 * i)) = 1000 :=
sorry

end sum_of_every_third_term_l706_706453


namespace factorize_a_cubed_minus_four_a_l706_706555

theorem factorize_a_cubed_minus_four_a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
sorry

end factorize_a_cubed_minus_four_a_l706_706555


namespace divisibility_323_l706_706140

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end divisibility_323_l706_706140


namespace area_ratio_l706_706100

theorem area_ratio (l w h : ℝ) (h1 : w * h = 288) (h2 : l * w = 432) (h3 : l * w * h = 5184) :
  (l * h) / (l * w) = 1 / 2 :=
sorry

end area_ratio_l706_706100


namespace tim_gets_correct_change_l706_706799

-- Assume the necessary constants and computations are inline
def num_loaves := 3
def slices_per_loaf := 20
def cost_per_slice := 0.40
def payment := 40.00

-- Compute the cost of one loaf
def cost_of_one_loaf : Float := slices_per_loaf * cost_per_slice

-- Compute the total cost of three loaves
def total_cost : Float := cost_of_one_loaf * num_loaves

-- Compute the change Tim gets when paying with $40
def change : Float := payment - total_cost

-- The statement we need to prove
theorem tim_gets_correct_change (h : change = 16.00) : change = 16.00 :=
by {
  -- We leave the proof to be filled in
  sorry
}

end tim_gets_correct_change_l706_706799


namespace tetrahedron_ratio_l706_706249

open Real

theorem tetrahedron_ratio (a b : ℝ) (h1 : a = PA ∧ PB = a) (h2 : PC = b ∧ AB = b ∧ BC = b ∧ CA = b) (h3 : a < b) :
  (sqrt 6 - sqrt 2) / 2 < a / b ∧ a / b < 1 :=
by
  sorry

end tetrahedron_ratio_l706_706249


namespace coffee_blend_l706_706292

variable (x : ℝ) (y : ℝ)

theorem coffee_blend (x y : ℝ) 
  (h1 : x = 8) 
  (h2 : y = 12) : 
  x + y = 20 := by
  rw [h1, h2]
  exact rfl

end coffee_blend_l706_706292


namespace jasmine_added_is_8_l706_706076

noncomputable def jasmine_problem (J : ℝ) : Prop :=
  let initial_volume := 80
  let initial_jasmine_concentration := 0.10
  let initial_jasmine_amount := initial_volume * initial_jasmine_concentration

  let added_water := 12
  let final_volume := initial_volume + J + added_water
  let final_jasmine_concentration := 0.16
  let final_jasmine_amount := final_volume * final_jasmine_concentration

  initial_jasmine_amount + J = final_jasmine_amount 

theorem jasmine_added_is_8 : jasmine_problem 8 :=
by
  sorry

end jasmine_added_is_8_l706_706076


namespace max_difference_in_flour_masses_l706_706436

/--
Given three brands of flour with the following mass ranges:
1. Brand A: (48 ± 0.1) kg
2. Brand B: (48 ± 0.2) kg
3. Brand C: (48 ± 0.3) kg

Prove that the maximum difference in mass between any two bags of these different brands is 0.5 kg.
-/
theorem max_difference_in_flour_masses :
  (∀ (a b : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.8 ≤ b ∧ b ≤ 48.2)) →
    |a - b| ≤ 0.5) ∧
  (∀ (a c : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |a - c| ≤ 0.5) ∧
  (∀ (b c : ℝ), ((47.8 ≤ b ∧ b ≤ 48.2) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |b - c| ≤ 0.5) := 
sorry

end max_difference_in_flour_masses_l706_706436


namespace part1_part2_l706_706933

noncomputable def f (x m : ℝ) : ℝ := x + m
noncomputable def g (x m : ℝ) : ℝ := x^2 - mx + (m^2 / 2) + 2 * m - 3

-- Part 1: Prove that if the solution set of g(x) < m^2 / 2 + 1 is (1, a), then a = 2
theorem part1 (m a : ℝ) (h : ∀ x, g x m < (m^2 / 2) + 1 ↔ 1 < x ∧ x < a) : a = 2 :=
sorry

-- Part 2: Prove the range of m where for all x1 in [0, 1], there exists x2 in [1, 2] such that f(x1) > g(x2) is -2 < m < 2
theorem part2 (m : ℝ) : (∀ x1 ∈ set.Icc 0 1, ∃ x2 ∈ set.Icc 1 2, f x1 m > g x2 m) ↔ (-2 < m ∧ m < 2) :=
sorry

end part1_part2_l706_706933


namespace power_neg_two_inverse_l706_706017

theorem power_neg_two_inverse : (-2 : ℤ) ^ (-2 : ℤ) = (1 : ℚ) / (4 : ℚ) := by
  -- Condition: a^{-n} = 1 / a^n for any non-zero number a and any integer n
  have h: ∀ (a : ℚ) (n : ℤ), a ≠ 0 → a ^ (-n) = 1 / a ^ n := sorry
  -- Proof goes here
  sorry

end power_neg_two_inverse_l706_706017


namespace k_plus_r_l706_706883

-- Define the operation for determinant of 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the function f(x) using the determinant
def f (x : ℝ) : ℝ := det (x - 1) 2 (-x) (x + 3)

-- Given the vertex of f(x)
def m : ℝ := -5 / 2
def n : ℝ := -13 / 4

-- Given the terms form an arithmetic sequence
def k : ℝ := m - (-13 / 4 + 5 / 2)
def r : ℝ := n + (-13 / 4 + 5 / 2)

-- Prove the value of k + r
theorem k_plus_r : k + r = -23 / 4 :=
by
  sorry

end k_plus_r_l706_706883


namespace distinct_m_values_l706_706099

theorem distinct_m_values : ∃ m_values : set ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values x1 + x2 = true)) ∧ m_values.card = 8 :=
by
  sorry

end distinct_m_values_l706_706099


namespace no_profit_for_x_in_200_300_min_avg_processing_cost_achieved_at_400_l706_706026

noncomputable def monthly_processing_cost (x : ℝ) : ℝ :=
if x ∈ set.Ico 120 144 then (1 / 3) * x^3 - 80 * x^2 + 5040 * x else
if x ∈ set.Ico 144 500 then (1 / 2) * x^2 - 200 * x + 80000 else 0

theorem no_profit_for_x_in_200_300 (x : ℝ) (hx : x ∈ set.Icc 200 300) :
  let s := 200 * x - monthly_processing_cost x
  0 ≤ -((1 / 2) * (x - 400)^2) → s = -5000 :=
sorry

theorem min_avg_processing_cost_achieved_at_400 (x : ℝ) :
  (monthly_processing_cost x / x) = 200 → x = 400 :=
sorry

end no_profit_for_x_in_200_300_min_avg_processing_cost_achieved_at_400_l706_706026


namespace central_angle_of_sector_l706_706452

theorem central_angle_of_sector (r l θ : ℝ) 
  (h1 : 2 * r + l = 8) 
  (h2 : (1 / 2) * l * r = 4) 
  (h3 : θ = l / r) : θ = 2 := 
sorry

end central_angle_of_sector_l706_706452


namespace smaller_bills_denomination_correct_l706_706379

noncomputable def denomination_of_smaller_bills : ℕ :=
  let total_money := 1000
  let part_smaller_bills := 3 / 10
  let smaller_bills_amount := part_smaller_bills * total_money
  let rest_of_money := total_money - smaller_bills_amount
  let bill_100_denomination := 100
  let total_bills := 13
  let num_100_bills := rest_of_money / bill_100_denomination
  let num_smaller_bills := total_bills - num_100_bills
  let denomination := smaller_bills_amount / num_smaller_bills
  denomination

theorem smaller_bills_denomination_correct : denomination_of_smaller_bills = 50 := by
  sorry

end smaller_bills_denomination_correct_l706_706379


namespace base_conversion_subtraction_l706_706544

theorem base_conversion_subtraction :
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  n1 - n2 = 7422 :=
by
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  show n1 - n2 = 7422
  sorry

end base_conversion_subtraction_l706_706544


namespace unique_fraction_condition_l706_706502

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l706_706502


namespace proof_problem_l706_706480

theorem proof_problem :
  (\(sqrt(2023) - 1) ^ 0 + (1 / 2) ^ (-1) = 3 :=
by
  sorry

end proof_problem_l706_706480


namespace sum_slope_y_intercept_eq_l706_706666

noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)

theorem sum_slope_y_intercept_eq :
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Place the conditions and setup here
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  -- Proof will be provided here eventually
  sorry

end sum_slope_y_intercept_eq_l706_706666


namespace find_solution_l706_706906

variable (y z : ℝ)

def vector1 : ℝ × ℝ × ℝ := (3, -1, 2)
def vector2 : ℝ × ℝ × ℝ := (-4, y, z)
def vector3 : ℝ × ℝ × ℝ := (1, 2, 3)

def orthogonal (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem find_solution (y z : ℝ) :
  orthogonal vector1 vector2 ∧ orthogonal vector2 vector3 → (y = -4 ∧ z = 4) :=
by
  intro h,
  sorry

end find_solution_l706_706906


namespace rectangle_diagonal_l706_706318

theorem rectangle_diagonal (P A: ℝ) (hP : P = 46) (hA : A = 120) : ∃ d : ℝ, d = 17 :=
by
  -- Sorry provides the placeholder for the actual proof.
  sorry

end rectangle_diagonal_l706_706318


namespace find_other_root_and_c_l706_706156

theorem find_other_root_and_c (c : ℝ) (x : ℝ) (h : x^2 - 6 * x + c = 0) (x = 2) : 
  ∃ t, (t = 4) ∧ (c = 8) :=
by sorry

end find_other_root_and_c_l706_706156


namespace number_of_such_fractions_is_one_l706_706521

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l706_706521


namespace sausage_more_than_pepperoni_l706_706711

noncomputable def pieces_of_meat_per_slice : ℕ := 22
noncomputable def slices : ℕ := 6
noncomputable def total_pieces_of_meat : ℕ := pieces_of_meat_per_slice * slices

noncomputable def pieces_of_pepperoni : ℕ := 30
noncomputable def pieces_of_ham : ℕ := 2 * pieces_of_pepperoni

noncomputable def total_pieces_of_meat_without_sausage : ℕ := pieces_of_pepperoni + pieces_of_ham
noncomputable def pieces_of_sausage : ℕ := total_pieces_of_meat - total_pieces_of_meat_without_sausage

theorem sausage_more_than_pepperoni : (pieces_of_sausage - pieces_of_pepperoni) = 12 := by
  sorry

end sausage_more_than_pepperoni_l706_706711


namespace degree_hx4_times_kx3_l706_706743

noncomputable def h (x : ℝ) : ℝ := sorry  -- h(x) is a polynomial of degree 3
noncomputable def k (x : ℝ) : ℝ := sorry  -- k(x) is a polynomial of degree 6

lemma degree_h_of_degree_3 (x : ℝ) (h_poly : isPolynomial h) (h_deg : polynomial.degree h = 3) :
  polynomial.degree (λ x, h(x^4)) = 12 := sorry

lemma degree_k_of_degree_6 (x : ℝ) (k_poly : isPolynomial k) (k_deg : polynomial.degree k = 6) :
  polynomial.degree (λ x, k(x^3)) = 18 := sorry

theorem degree_hx4_times_kx3 (x : ℝ) 
  (h_poly : isPolynomial h) (h_deg : polynomial.degree h = 3) 
  (k_poly : isPolynomial k) (k_deg : polynomial.degree k = 6) :
  polynomial.degree (λ x, h(x^4) * k(x^3)) = 30 :=
by
  have deg_hx4 : polynomial.degree (λ x, h(x^4)) = 12, from degree_h_of_degree_3 x h_poly h_deg
  have deg_kx3 : polynomial.degree (λ x, k(x^3)) = 18, from degree_k_of_degree_6 x k_poly k_deg
  rw [polynomial.degree_mul, deg_hx4, deg_kx3]
  exact dec_trivial -- degree of h(x^4) * k(x^3) = 12 + 18

end degree_hx4_times_kx3_l706_706743


namespace temple_run_red_coins_l706_706246

variables (x y z : ℕ)

theorem temple_run_red_coins :
  x + y + z = 2800 →
  x + 3 * y + 5 * z = 7800 →
  z = y + 200 →
  y = 700 := 
by 
  intro h1 h2 h3
  sorry

end temple_run_red_coins_l706_706246


namespace distance_between_railings_l706_706870

/-- 
Given bicycles are placed every 5 meters and there are 19 bicycles in total,
prove that the distance between the two railings is 95 meters.
-/
theorem distance_between_railings :
  ∃ d : ℕ, d = 5 * 19 ∧ d = 95 :=
by
  use 95
  constructor
  · rfl
  · rfl

end distance_between_railings_l706_706870


namespace find_m_l706_706599

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := real.logb 4 (m * x^2 + 2 * x + 3)

noncomputable def min_quadratic (m : ℝ) : ℝ := -(real.sqrt(4 * (m * 3 - m * m * (4 * (-1)))) - 2) / (2 * m)

theorem find_m : ∃ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ∧ f m (min_quadratic m) = 0 :=
by
  use 1 / 2
  sorry

end find_m_l706_706599


namespace probability_non_adjacent_two_twos_l706_706219

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end probability_non_adjacent_two_twos_l706_706219


namespace num_dessert_menus_l706_706831

-- Define the types and conditions
inductive Dessert
| cake
| pie
| ice_cream
| pudding

open Dessert

def is_valid_menu (menu : List Dessert) :=
  menu.length = 7 ∧
  menu.nth 4 = some cake ∧  -- cake on the 5th index (Friday).
  ∀ i, i < 6 → menu.nth i ≠ menu.nth (i + 1)  -- No two consecutive days have the same dessert.

theorem num_dessert_menus : ∃ (menus : List (List Dessert)), menus.length = 729 ∧
  ∀ menu ∈ menus, is_valid_menu menu :=
sorry

end num_dessert_menus_l706_706831


namespace nature_of_f_on_interval_l706_706767

def f (x : ℝ) (a : ℝ) : ℝ := log a (abs (x - 1))

theorem nature_of_f_on_interval (a : ℝ) (h : 1 < a) :
  (∀ x y : ℝ, 0 < x ∧ y < 1 → x < y → f x a > f y a) →
  (∀ x y : ℝ, 1 < x ∧ 1 < y → x < y → f x a < f y a) :=
by
  sorry

end nature_of_f_on_interval_l706_706767


namespace range_of_x_l706_706635

-- Define the condition: the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := 3 + x ≥ 0

-- Define what we want to prove: the range of x such that the condition holds
theorem range_of_x (x : ℝ) : condition x ↔ x ≥ -3 :=
by
  -- Proof goes here
  sorry

end range_of_x_l706_706635


namespace evaluate_expression_at_zero_l706_706390

theorem evaluate_expression_at_zero :
  (0^2 + 5 * 0 - 10) = -10 :=
by
  sorry

end evaluate_expression_at_zero_l706_706390


namespace complex_product_magnitude_l706_706550

-- definitions of complex magnitudes and complex product magnitudes
def magnitude (z : Complex) : Real :=
  match z with
  | ⟨a, b⟩ => Real.sqrt (a * a + b * b)

noncomputable def product_magnitude (z1 z2 : Complex) : Real :=
  magnitude z1 * magnitude z2

-- Given complex numbers
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- The proof statement
theorem complex_product_magnitude : magnitude (z1 * z2) = Real.sqrt 7085 :=
  by 
  sorry

end complex_product_magnitude_l706_706550


namespace limiting_reactant_and_products_l706_706485

def balanced_reaction 
  (al_moles : ℕ) (h2so4_moles : ℕ) 
  (al2_so4_3_moles : ℕ) (h2_moles : ℕ) : Prop :=
  2 * al_moles >= 0 ∧ 3 * h2so4_moles >= 0 ∧ 
  al_moles = 2 ∧ h2so4_moles = 3 ∧ 
  al2_so4_3_moles = 1 ∧ h2_moles = 3 ∧ 
  (2 : ℕ) * al_moles + (3 : ℕ) * h2so4_moles = 2 * 2 + 3 * 3

theorem limiting_reactant_and_products :
  balanced_reaction 2 3 1 3 :=
by {
  -- Here we would provide the proof based on the conditions and balances provided in the problem statement.
  sorry
}

end limiting_reactant_and_products_l706_706485


namespace smallest_next_divisor_l706_706714

theorem smallest_next_divisor (n : ℕ) (h_even : n % 2 = 0) (h_4_digit : 1000 ≤ n ∧ n < 10000) (h_div_493 : 493 ∣ n) :
  ∃ d : ℕ, (d > 493 ∧ d ∣ n) ∧ ∀ e, (e > 493 ∧ e ∣ n) → d ≤ e ∧ d = 510 := by
  sorry

end smallest_next_divisor_l706_706714


namespace technicans_permanent_50pct_l706_706650

noncomputable def percentage_technicians_permanent (p : ℝ) : Prop :=
  let technicians := 0.5
  let non_technicians := 0.5
  let temporary := 0.5
  (0.5 * (1 - 0.5)) + (technicians * p) = 0.5 ->
  p = 0.5

theorem technicans_permanent_50pct (p : ℝ) :
  percentage_technicians_permanent p :=
sorry

end technicans_permanent_50pct_l706_706650


namespace find_y_l706_706957

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y_l706_706957


namespace a_2007_value_l706_706689

noncomputable def sequence (a : ℕ → ℕ) :=
∀ (m n : ℕ), m ≥ n → a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

theorem a_2007_value (a : ℕ → ℕ) (h_seq : sequence a) (h_a1 : a 1 = 1) : a 2007 = 2007 * 2007 :=
  sorry

end a_2007_value_l706_706689


namespace find_C_D_l706_706556

theorem find_C_D :
  ∃ C D : ℚ, (∀ x : ℚ, x ≠ 9 ∧ x ≠ -4 → (7 * x + 3) / (x^2 - 5 * x - 36) = C / (x - 9) + D / (x + 4))
  ∧ C = 66 / 13
  ∧ D = 25 / 13 :=
by
  existsi (66 / 13)
  existsi (25 / 13)
  split
  {
    intros x hxy
    field_simp [mul_sub, sub_mul]
    norm_num
  }
  {
    split
    {
      reflexivity
    }
    {
      reflexivity
    }
  }

end find_C_D_l706_706556


namespace part1_part2_l706_706575

-- Define the sequence and sum conditions
variable {t : ℝ} (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (h1 : 0 < t) (h2 : t ≠ 1)
variable (h3 : ∀ n:ℕ, S n = (t / (t - 1)) * a n - n)

-- First part: sequence {a_{n+1}} is geometric and general formula for {a_n}
theorem part1 (n : ℕ) : 
  ∃ r : ℝ, ∀ k : ℕ, a (k+1) = r * a k :=
  ∃ r : ℝ, ∀ n:ℕ, a n = t^n - 1 := 
sorry

-- Second part: prove the given inequality for c_n when t=2
noncomputable def c (n : ℕ) := (a n + 1) / (a n * a (n + 1))
variable (h4 : ∀ n:ℕ, a n = 2^n - 1)
variable (cn := λ n:ℕ, c a n)
theorem part2 (n : ℕ) :
  2 / 3 ≤ ∑ k in Finset.range n, cn k ∧ ∑ k in Finset.range n, cn k < 1 :=
sorry

end part1_part2_l706_706575


namespace boy_has_10_good_pairs_l706_706795

theorem boy_has_10_good_pairs 
  (n : ℕ) 
  (circle : Fin (2 * n) → bool) 
  (exists_girl_10_good_pairs : ∃ d, is_girl circle d ∧ count_good_pairs circle d = 10) 
  : ∃ b, is_boy circle b ∧ count_good_pairs circle b = 10 := sorry

def is_girl (circle : Fin (2 * n) → bool) (i : Fin (2 * n)) : Prop :=
  circle i = true

def is_boy (circle : Fin (2 * n) → bool) (i : Fin (2 * n)) : Prop :=
  circle i = false

def count_good_pairs (circle : Fin (2 * n) → bool) (i : Fin (2 * n)) : ℕ :=
sorry

end boy_has_10_good_pairs_l706_706795


namespace problem_proof_l706_706916

variables (m n : Type) [line m] [line n]
variable (α : Type) [plane α]

-- Definitions for perpendicular, parallel, and subset relations for later reference 
def perp (l : Type) (p : Type) [line l] [plane p] : Prop := sorry
def parallel (l₁ l₂ : Type) [line l₁] [line l₂] : Prop := sorry
def subset (l : Type) (p : Type) [line l] [plane p] : Prop := sorry

-- The proof problem in Lean 4
theorem problem_proof :
  (perp m α ∧ perp n α → parallel m n) ∧ (perp m α ∧ subset n α → perp m n) :=
by
  sorry

end problem_proof_l706_706916


namespace count_sequences_is_140_l706_706574

noncomputable def count_sequences : ℕ :=
  fintype.card {s : finset (fin 7) // s.card = 3} * fintype.card {t : finset (fin 4) // t.card = 3}

theorem count_sequences_is_140 :
  count_sequences = 140 :=
sorry

end count_sequences_is_140_l706_706574


namespace sum_of_series_l706_706875

noncomputable def sum_term (k : ℕ) : ℝ :=
  (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

theorem sum_of_series : (∑' k : ℕ, sum_term (k + 1)) = 7 / 4 := by
  sorry

end sum_of_series_l706_706875


namespace part1_part2_part3_l706_706057

-- Define the function f according to the given properties
def f : ℕ → ℕ → ℕ
| m, 1 => if m = 1 then 1 else 2 * f (m - 1) 1
| m, n + 1 => f m n + 2

-- State the theorem for the first part
theorem part1 (n : ℕ) : f 1 n = 2 * n - 1 :=
by sorry

-- State the theorem for the second part
theorem part2 (m : ℕ) : f m 1 = 2 ^ (m - 1) :=
by sorry

-- State the theorem for the third part
theorem part3 : f 2002 9 = 2 ^ 2001 + 16 :=
by sorry

end part1_part2_part3_l706_706057


namespace negation_of_proposition_l706_706778

theorem negation_of_proposition (a b : ℝ) :
  ¬(a > b → 2 * a > 2 * b) ↔ (a ≤ b → 2 * a ≤ 2 * b) :=
by
  sorry

end negation_of_proposition_l706_706778


namespace eval_floor_expr_l706_706114

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end eval_floor_expr_l706_706114


namespace both_readers_l706_706651

theorem both_readers (total S L B : ℕ) (h_total : total = 400) (h_S : S = 250) (h_L : L = 230) (h_eq : S + L - B = total) : B = 80 :=
by {
  subst h_total,
  subst h_S,
  subst h_L,
  linarith,
  sorry
}

end both_readers_l706_706651


namespace quotient_of_base4_l706_706554

noncomputable def base4_to_base10 (num : ℕ) : ℕ :=
  num.digits 4.reverse.enum.map (λ (pair : ℕ × ℕ), pair.fst * 4 ^ pair.snd).sum

noncomputable def base10_to_base4 (num : ℕ) : ℕ :=
  num.to_digits 4.reverse.mk_pair (λ (digit : ℕ), digit.digit_to_char 4).foldr (λ (c : Char) (acc : ℕ), acc * 10 + c.char_to_digit.one_product.to_nat) 0

theorem quotient_of_base4 (n1 n2 : ℕ) (h1 : n1 = 3213) (h2 : n2 = 13) :
  (base10_to_base4 ((base4_to_base10 n1) / (base4_to_base10 n2)) = 201_4) :=
by
  sorry

end quotient_of_base4_l706_706554


namespace spinner_probability_l706_706849

theorem spinner_probability (P_D P_E : ℝ) (hD : P_D = 2/5) (hE : P_E = 1/5) 
  (hTotal : P_D + P_E + P_F = 1) : P_F = 2/5 :=
by
  sorry

end spinner_probability_l706_706849


namespace min_omega_shift_overlap_l706_706188

theorem min_omega_shift_overlap (ω : ℝ) (h : ω > 0) :
  (∀ x, 2 * sin (ω * x + π / 3) - 1 = 2 * sin (ω * (x - π / 3) + π / 3) - 1) → ω = 6 :=
by
  -- Proof is not required
  sorry

end min_omega_shift_overlap_l706_706188


namespace scale_fragments_l706_706362

-- definitions for adults and children heights
def height_adults : Fin 100 → ℕ 
def height_children : Fin 100 → ℕ 

-- the main theorem statement
theorem scale_fragments (h₁ : ∀ i, height_children i < height_adults i) :
  ∃ (k : Fin 100 → ℕ), ∀ (i j : Fin 100), 
  k i * height_children i < k i * height_adults i ∧ 
  (i ≠ j → ∀ (a: ℕ), k i * height_children i < a → k i * height_adults i < a) :=
sorry

end scale_fragments_l706_706362


namespace perpendicular_CF_AE_l706_706433

/-- In the acute-angled triangle ABC, let:
    - D be the perpendicular projection of point C on side AB, so CD ⊥ AB,
    - E be the perpendicular projection of point D on side BC, so DE ⊥ BC,
    - F lie on DE such that the ratio EF/FD = AD/DB.
Prove that CF ⊥ AE. -/
theorem perpendicular_CF_AE
  {A B C D E F : Type}
  [IsTriangle A B C]
  (CD_perp_AB : IsPerpendicular (Line C D) (Line A B))
  (DE_perp_BC : IsPerpendicular (Line D E) (Line B C))
  (EF_FD_AD_DB : ratio (Seg E F) (Seg F D) = ratio (Seg A D) (Seg D B)) :
  IsPerpendicular (Line C F) (Line A E) :=
  sorry

end perpendicular_CF_AE_l706_706433


namespace determine_A_l706_706309

-- Define the polynomial and its factors
def polynomial (x : ℝ) : ℝ := x^3 - x^2 - 21 * x + 45
def factor1 (x : ℝ) : ℝ := x + 5
def factor2 (x : ℝ) : ℝ := (x - 3)^2

-- Theorem stating the equality and finding value of A
theorem determine_A (A B C : ℝ) (h : polynomial = factor1 * factor2) 
    (h_eq : ∀ x, 1 / polynomial x = A / factor1 x + B / (x - 3) + C / (factor2 x)) : 
    A = 1 / 64 :=
begin
  sorry
end

end determine_A_l706_706309


namespace tan_neg_3780_eq_zero_l706_706096

theorem tan_neg_3780_eq_zero : Real.tan (-3780 * Real.pi / 180) = 0 := 
by 
  sorry

end tan_neg_3780_eq_zero_l706_706096


namespace intersect_on_median_l706_706291

theorem intersect_on_median
  (A B C M P K : Point)
  (CA CB CM CP : Real)
  (hM : OnSegment C B M)
  (hP : OnSegment C A P)
  (hRatio : CP / CA = 2 * (CM / CB))
  (hParallel1 : ParallelLineThrough M CA)
  (hParallel2 : ParallelLineThrough P AB)
  (hIntersect : Intersect K (ParallelLineThrough P AB) (ParallelLineThrough M CA))
  (hMedian : CN = Median C A B): 
  IntersectAtMedian K CN := 
sorry

end intersect_on_median_l706_706291


namespace unique_fraction_condition_l706_706500

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l706_706500


namespace area_of_triangle_ABC_l706_706323

theorem area_of_triangle_ABC (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let y_vertex := (4 * a * c - b^2) / (4 * a)
  0.5 * (|x2 - x1|) * |y_vertex| = (b^2 - 4 * a * c) * Real.sqrt (b^2 - 4 * a * c) / (8 * a^2) :=
sorry

end area_of_triangle_ABC_l706_706323


namespace inequality_proof_l706_706559

theorem inequality_proof (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 :=
by
  sorry

end inequality_proof_l706_706559


namespace average_minutes_run_per_day_l706_706994

-- Define the given averages for each grade
def sixth_grade_avg : ℕ := 10
def seventh_grade_avg : ℕ := 18
def eighth_grade_avg : ℕ := 12

-- Define the ratios of the number of students in each grade
def num_sixth_eq_three_times_num_seventh (num_seventh : ℕ) : ℕ := 3 * num_seventh
def num_eighth_eq_half_num_seventh (num_seventh : ℕ) : ℕ := num_seventh / 2

-- Average number of minutes run per day by all students
theorem average_minutes_run_per_day (num_seventh : ℕ) :
  (sixth_grade_avg * num_sixth_eq_three_times_num_seventh num_seventh +
   seventh_grade_avg * num_seventh +
   eighth_grade_avg * num_eighth_eq_half_num_seventh num_seventh) / 
  (num_sixth_eq_three_times_num_seventh num_seventh + 
   num_seventh + 
   num_eighth_eq_half_num_seventh num_seventh) = 12 := 
sorry

end average_minutes_run_per_day_l706_706994


namespace number_of_regular_pencils_l706_706234

def cost_eraser : ℝ := 0.8
def cost_regular : ℝ := 0.5
def cost_short : ℝ := 0.4
def num_eraser : ℕ := 200
def num_short : ℕ := 35
def total_revenue : ℝ := 194

theorem number_of_regular_pencils (num_regular : ℕ) :
  (num_eraser * cost_eraser) + (num_short * cost_short) + (num_regular * cost_regular) = total_revenue → 
  num_regular = 40 :=
by
  sorry

end number_of_regular_pencils_l706_706234


namespace minimum_x_plus_y_l706_706571

noncomputable def min_value_x_plus_y : ℝ :=
  λ x y : ℝ, if (x > 0 ∧ y > 0 ∧ (1/x + 9/y = 2)) then x + y else 0

theorem minimum_x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  min_value_x_plus_y x y = 8 :=
begin
  sorry
end

end minimum_x_plus_y_l706_706571


namespace total_fruit_adam_eq_l706_706071

variables (X Y Z : ℕ)

def apples_sarah := X
def oranges_sarah := Y
def apples_jackie := 2 * apples_sarah
def oranges_jackie := Z + 3
def apples_adam := apples_jackie + 5
def oranges_adam := 2 * oranges_sarah
def total_fruit_adam := apples_adam + oranges_adam

theorem total_fruit_adam_eq (X Y Z : ℕ) :
  total_fruit_adam X Y Z = (2 * X + 5) + (2 * Y) :=
by
  sorry

end total_fruit_adam_eq_l706_706071


namespace gcd_polynomial_l706_706169

theorem gcd_polynomial (b : ℤ) (h : b % 2 = 0 ∧ 1171 ∣ b) : 
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 :=
sorry

end gcd_polynomial_l706_706169


namespace solve_for_z_l706_706735

variable (z : ℂ) (i : ℂ)

theorem solve_for_z
  (h1 : 3 - 2*i*z = 7 + 4*i*z)
  (h2 : i^2 = -1) :
  z = 2*i / 3 :=
by
  sorry

end solve_for_z_l706_706735


namespace problems_per_worksheet_l706_706851

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ)
    (h1 : total_worksheets = 16) (h2 : graded_worksheets = 8) (h3 : remaining_problems = 32) :
    remaining_problems / (total_worksheets - graded_worksheets) = 4 :=
by
  sorry

end problems_per_worksheet_l706_706851


namespace pentagon_quadrilateral_contains_points_l706_706252

theorem pentagon_quadrilateral_contains_points 
  {P : Type} [convex P] (A1 A2 A3 A4 A5 P1 P2 : P) :
  ∃ Q : set P, is_quadrilateral Q ∧ subset_of_vertices Q ({A1, A2, A3, A4, A5} : set P) ∧ P1 ∈ Q ∧ P2 ∈ Q :=
sorry

end pentagon_quadrilateral_contains_points_l706_706252


namespace polar_coordinates_of_point_l706_706745

noncomputable def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x * x + y * y)
  let theta := if y < 0 then 2 * real.pi + real.arctan (y/x) else real.arctan (y/x)
  (r, theta)

theorem polar_coordinates_of_point :
  cartesian_to_polar 1 (-real.sqrt 3) = (2, 5 * real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l706_706745


namespace price_difference_VA_NC_l706_706644

/-- Define the initial conditions -/
def NC_price : ℝ := 2
def NC_gallons : ℕ := 10
def VA_gallons : ℕ := 10
def total_spent : ℝ := 50

/-- Define the problem to prove the difference in price per gallon between Virginia and North Carolina -/
theorem price_difference_VA_NC (NC_price VA_price total_spent : ℝ) (NC_gallons VA_gallons : ℕ) :
  total_spent = NC_price * NC_gallons + VA_price * VA_gallons →
  VA_price - NC_price = 1 := 
by
  sorry -- Proof to be filled in

end price_difference_VA_NC_l706_706644


namespace average_track_width_l706_706438

theorem average_track_width (r1 r2 s1 s2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * s1 - 2 * Real.pi * s2 = 30 * Real.pi) :
  (r1 - r2 + (s1 - s2)) / 2 = 12.5 := 
sorry

end average_track_width_l706_706438


namespace parabola_line_intersect_at_one_point_l706_706391

theorem parabola_line_intersect_at_one_point (a : ℚ) :
  (∃ x : ℚ, ax^2 + 5 * x + 4 = 0) → a = 25 / 16 :=
by
  -- Conditions and computation here
  sorry

end parabola_line_intersect_at_one_point_l706_706391


namespace find_k_l706_706529

def otimes (a b : ℝ) := a * b + a + b^2

theorem find_k (k : ℝ) (h1 : otimes 1 k = 2) (h2 : 0 < k) :
  k = 1 :=
sorry

end find_k_l706_706529


namespace prob_sunglasses_also_wearing_cap_l706_706721

variable (P Q : Type) -- Introduce types for people wearing sunglasses and caps

-- Define the conditions
def people_wearing_sunglasses := 75
def people_wearing_caps := 60
def probability_cap_wearing_sunglasses := 1 / 3

-- Number of people wearing both
def people_wearing_both : ℕ := (people_wearing_caps * probability_cap_wearing_sunglasses).toNat

-- Define the theorem to be proven
theorem prob_sunglasses_also_wearing_cap :
  (people_wearing_both / people_wearing_sunglasses : ℚ) = 4 / 15 := by
  sorry

end prob_sunglasses_also_wearing_cap_l706_706721


namespace gcd_1343_816_l706_706324

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l706_706324


namespace find_t_value_l706_706675

noncomputable def k : ℚ :=
  60 / 13

-- We know that t = k / sqrt(1 - k^2)
def t (k : ℚ) : ℚ :=
  k / Real.sqrt (1 - k ^ 2)

theorem find_t_value :
  t k = 12 / 5 := by
  sorry

end find_t_value_l706_706675


namespace locus_of_M_l706_706615

variables {A B M : Type*}
variables (pA pB : Point) (C1 C2 : Circle) (M : Point)

-- Conditions: Definitions
def C1_tangent_at_A (C1 : Circle) (pA : Point) : Prop := 
  tangent_to_line_at C1 (line_through pA pB) pA

def C2_tangent_at_B (C2 : Circle) (pB : Point) : Prop := 
  tangent_to_line_at C2 (line_through pA pB) pB

def M_common_tangent (C1 C2 : Circle) (M : Point) : Prop := 
  tangent_to_each_other_at C1 C2 M

-- Prove: locus statement
theorem locus_of_M (h1 : C1_tangent_at_A C1 pA)
                   (h2 : C2_tangent_at_B C2 pB)
                   (h3 : M_common_tangent C1 C2 M) :
  (exists O : Point, midpoint pA pB O ∧ distance O M = distance O pA ∧ M ≠ pA ∧ M ≠ pB) ∧ 
  locus (circle O (distance O pA)) :=
sorry

end locus_of_M_l706_706615


namespace fifteenth_term_l706_706542

noncomputable def seq : ℕ → ℝ
| 0       => 3
| 1       => 4
| (n + 2) => 12 / seq (n + 1)

theorem fifteenth_term :
  seq 14 = 3 :=
sorry

end fifteenth_term_l706_706542


namespace seventh_term_of_geometric_sequence_l706_706491

theorem seventh_term_of_geometric_sequence :
  ∀ (a r : ℝ), (a * r ^ 3 = 16) → (a * r ^ 8 = 2) → (a * r ^ 6 = 2) :=
by
  intros a r h1 h2
  sorry

end seventh_term_of_geometric_sequence_l706_706491


namespace democrats_ratio_l706_706793

noncomputable def F : ℕ := 240
noncomputable def M : ℕ := 480
noncomputable def D_F : ℕ := 120
noncomputable def D_M : ℕ := 120

theorem democrats_ratio (total_participants : ℕ := 720)
  (h1 : F + M = total_participants)
  (h2 : D_F = 120)
  (h3 : D_F = 1/2 * F)
  (h4 : D_M = 1/4 * M)
  (h5 : D_F + D_M = 240)
  (h6 : F + M = 720) : (D_F + D_M) / total_participants = 1 / 3 :=
by
  sorry

end democrats_ratio_l706_706793


namespace sector_max_area_l706_706997

theorem sector_max_area (P : ℝ) (R l S : ℝ) :
  (P > 0) → (2 * R + l = P) → (S = 1/2 * R * l) →
  (R = P / 4) ∧ (S = P^2 / 16) :=
by
  sorry

end sector_max_area_l706_706997


namespace total_water_filled_jars_l706_706677

theorem total_water_filled_jars (x : ℕ) (h1 : 7 * 4 = 28)
(h2 : 4 * x + 2 * x + x = 28) : 3 * x = 12 :=
by
  -- Conversion factor from gallons to quarts
  have conv_factor : 28 = 7 * 4 := h1
  -- Proportional water distribution among different sizes of jars
  have proportional_eq : 7 * x = 28 := h2
  -- Solving for the total number of each jar type
  have total_jars : x = 28 / 7 := by linarith
  -- Calculate the total number of jars
  calc
    3 * x = 3 * 4 : by rw total_jars
    ... = 12 : by norm_num

end total_water_filled_jars_l706_706677


namespace jane_quadratic_solutions_eq_l706_706686

-- Define the solutions to Lauren's equation
def lauren_solutions : set ℝ := { x | |x - 4| = 3 }

-- Define Jane's quadratic equation
def jane_quadratic (b c : ℝ) (x : ℝ) : Prop := x^2 + b*x + c = 0

-- Define the correctness condition to prove that (b, c) = (-8, 7)
theorem jane_quadratic_solutions_eq (b c : ℝ) :
  (∀ x, x ∈ lauren_solutions ↔ jane_quadratic b c x) ↔ (b = -8 ∧ c = 7) :=
by
  sorry

end jane_quadratic_solutions_eq_l706_706686


namespace right_angled_triangles_count_l706_706780

theorem right_angled_triangles_count :
    ∃ n : ℕ, n = 31 ∧ ∀ (a b : ℕ), (b < 2011) ∧ (a * a = (b + 1) * (b + 1) - b * b) → n = 31 :=
by
  sorry

end right_angled_triangles_count_l706_706780


namespace problem1_problem2_l706_706921

-- Defining the circle C and lines l1, l2
noncomputable def circle_C (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 4
def line_l2 (p: ℝ × ℝ) : Prop := p.1 + 2 * p.2 + 2 = 0
def point_A : ℝ × ℝ := (1, 0)

-- Function to check if a line is tangent to the circle
def is_tangent_line (line : ℝ → ℝ) : Prop := 
  ∃ x₀ y₀ : ℝ, circle_C x₀ y₀ ∧ y₀ = line x₀ ∧ 
  -- Distance from the circle center to the line equals the radius
  (|3 * line(1) - 4 - line(0)| / Real.sqrt (line(1)^2 + 1) = 2)

-- Function to check if a line intersects the circle and compute AM
def intersects_and_midpoint (line : ℝ → ℝ) (A M N : ℝ × ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, 
  circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ -- P, Q are intersection points
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧ -- M is the midpoint
  ∃ k x : ℝ, N = ((2*k - 2)/(2*k + 1), - (3*k)/(2*k + 1)) ∧ -- Intersection point N
  line(1 * Real.sqrt(1 + k^2)) * Real.sqrt(1 + k^2) = k -- ensuring the correctness of k

-- Statement for the first part
theorem problem1 : 
  (is_tangent_line (λ x, 0) ∨ is_tangent_line (λ x, 3*x/4 - 3/4)) :=
sorry

-- Statement for the second part
theorem problem2 : 
  ∀ M N : ℝ × ℝ, 
  intersects_and_midpoint (λ x, 3*x / 4 - 3 / 4) point_A M N → 
  AM.N (point_A M N) = 6 :=
sorry

end problem1_problem2_l706_706921


namespace carpet_needed_l706_706845

-- Define the conditions
def room_length : ℝ := 15
def room_width : ℝ := 10
def sq_ft_per_sq_yd : ℝ := 9

-- Define the total area in square feet
def area_in_sq_ft : ℝ := room_length * room_width

-- Define the area in square yards (rounded up to the nearest whole number)
def area_in_sq_yds : ℝ := (area_in_sq_ft / sq_ft_per_sq_yd).ceil

-- State the theorem
theorem carpet_needed : area_in_sq_yds = 17 := by
  sorry

end carpet_needed_l706_706845


namespace shadow_length_when_eight_meters_away_l706_706729

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l706_706729


namespace find_r_l706_706937

-- Define the given conditions
variables (A B C D E F M N : Type)
variables (r : ℝ)
variables [regular_hexagon A B C D E F]
variables (div_AC : divides_on_segment A C M r)
variables (div_CE : divides_on_segment C E N r)
variables (collinear : collinear B M N)

-- Define the proof goal
theorem find_r (h1 : div_AC) (h2 : div_CE) (h3 : collinear) : r = 1 / real.sqrt 3 :=
sorry

end find_r_l706_706937


namespace curve_is_circle_with_radius_three_l706_706896

noncomputable def curve_in_polar_coordinates (r : ℝ) : Prop :=
  ∀ (θ : ℝ), r = 3

theorem curve_is_circle_with_radius_three : 
  ∀ (p : curve_in_polar_coordinates 3), 
    ∃ (R : ℝ), R = 3 ∧ ∀ (θ : ℝ), (p = 3) :=
by
  sorry

end curve_is_circle_with_radius_three_l706_706896


namespace proof_problem_l706_706848

-- Given a set of data
def data : List Float := [2, 3, 5, 7, 9]

-- Define median
def median (l : List Float) : Float :=
  match l.nth_le? (l.length / 2) (by simp [List.length, l.length]) with
  | some m => m
  | none => 0  -- should never happen for non-empty lists

-- Define mean
def mean (l : List Float) : Float :=
  l.sum / (l.length)

-- Define the range of the third side of a triangle given sides a and b
def triangle_side_range (a b : Float) : Float × Float :=
  (abs (a - b), a + b)

-- Calculate the perimeters of an isosceles triangle
def isosceles_triangle_perimeter (a b : Float) : Float × Float :=
  (2 * a + b, 2 * b + a)

-- The Lean theorem to prove
theorem proof_problem :
  let a := median data
  let b := mean data
  a = 5 ∧
  b = 5.2 ∧
  let (low, high) := triangle_side_range a b in
  0.2 < low ∧ high < 10.2 ∧
  let (perim1, perim2) := isosceles_triangle_perimeter a b in
  perim1 = 15.2 ∧ perim2 = 15.4 :=
by
  unfold median
  unfold mean
  unfold triangle_side_range
  unfold isosceles_triangle_perimeter
  rw [data]  -- using the given set of data
  -- calculate the median
  have median_eq : median data = 5 := by 
    unfold median
    simp
  -- calculate the mean
  have mean_eq : mean data = 5.2 := by 
    unfold mean
    simp
  -- calculate the range of the third side
  have range_eq : (abs (5 - 5.2), 5 + 5.2) = (0.2, 10.2) := by 
    unfold abs
    simp
  -- calculate the perimeter of an isosceles triangle
  have perim_eq1 : 2 * 5 + 5.2 = 15.2 := by 
    simp
  have perim_eq2 : 2 * 5.2 + 5 = 15.4 := by 
    simp
  exact ⟨median_eq, mean_eq, range_eq, perim_eq1, perim_eq2⟩

end proof_problem_l706_706848


namespace num_starting_lineups_l706_706484

def total_players := 15
def chosen_players := 3 -- Ace, Zeppo, Buddy already chosen
def remaining_players := total_players - chosen_players
def players_to_choose := 2 -- remaining players to choose

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem num_starting_lineups : combinations remaining_players players_to_choose = 66 := by
  sorry

end num_starting_lineups_l706_706484


namespace angle_bisector_slope_l706_706313

theorem angle_bisector_slope 
  (y1 y2 : ℝ → ℝ)
  (hy1 : ∀ x, y1 x = 2 * x)
  (hy2 : ∀ x, y2 x = 5 * x) :
  ∃ k : ℝ, (∀ x, k * x = (y1 x + y2 x + sqrt (1 + (y1 x)^2 + (y2 x)^2)) / (1 - (y1 x) * (y2 x))) ∧
           k = (sqrt 30 - 7) / 9 :=
by
  sorry

end angle_bisector_slope_l706_706313


namespace clock_angle_5_30_l706_706437

/-- 
Given a circular clock face with twelve equally spaced dots representing the hours, 
and a minute hand pointing directly at 12 o'clock, 
prove that the measure in degrees of the smaller angle formed between the minute hand and the hour hand when the time is 5:30 is 15 degrees.
-/
theorem clock_angle_5_30 :
  let hour_angle := 360 / 12
  let hour_deg := 5.5 * hour_angle
  let minute_deg := 180
  |minute_deg - hour_deg| = 15 :=
by
  let hour_angle := 360 / 12
  let hour_deg := 5.5 * hour_angle
  let minute_deg := 180
  have h : abs (minute_deg - hour_deg) = 15 := sorry
  exact h

end clock_angle_5_30_l706_706437


namespace three_n_by_three_n_grid_has_more_partitions_than_two_n_by_two_n_grid_l706_706971

theorem three_n_by_three_n_grid_has_more_partitions_than_two_n_by_two_n_grid
  (n : ℕ) (hn : 0 < n) :
  (#(partition2n2n : Partitions (Grid (2*n) (2*n)) (Blocks.1x2))) ≤
  (#(partition3n3n : Partitions (Grid (3*n) (3*n)) (Blocks.LTrimino))) := 
sorry

end three_n_by_three_n_grid_has_more_partitions_than_two_n_by_two_n_grid_l706_706971


namespace power_mean_convergence_max_power_mean_convergence_min_l706_706303

def power_mean (r : ℚ) (a : Finₓ n → ℝ) : ℝ :=
  (∑ i, (a i)^r / n)^(1/r)

theorem power_mean_convergence_max (a : Finₓ n → ℝ) (ha : ∀ i, 0 < a i) :
  ∀ ε > 0, ∃ R : ℚ, ∀ r > R, |power_mean r a - (finset.univ.sup a)| < ε :=
begin
  sorry
end

theorem power_mean_convergence_min (a : Finₓ n → ℝ) (ha : ∀ i, 0 < a i) :
  ∀ ε > 0, ∃ R : ℚ, ∀ r < -R, |power_mean (-r) a - (finset.univ.inf a)| < ε :=
begin
  sorry
end

end power_mean_convergence_max_power_mean_convergence_min_l706_706303


namespace max_value_of_xy_l706_706584

theorem max_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  xy ≤ 1 / 2 :=
sorry

end max_value_of_xy_l706_706584


namespace boat_trip_duration_l706_706800

noncomputable def boat_trip_time (B P : ℝ) : Prop :=
  (P = 4 * B) ∧ (B + P = 10)

theorem boat_trip_duration (B P : ℝ) (h : boat_trip_time B P) : B = 2 :=
by
  cases h with
  | intro hP hTotal =>
    sorry

end boat_trip_duration_l706_706800


namespace number_of_such_fractions_is_one_l706_706519

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l706_706519


namespace sum_of_sequence_l706_706962

def sequence (n : ℕ) : ℚ := (n^2) / ((2 * n - 1) * (2 * n + 1))

def S (n : ℕ) : ℚ := (n * (n + 1)) / (2 * (2 * n + 1))

theorem sum_of_sequence (n : ℕ) : ∑ i in finset.range (n + 1), sequence i = S n :=
sorry

end sum_of_sequence_l706_706962


namespace geometric_series_denominator_l706_706384

-- We assume all terms are real numbers.
variables {a r : ℝ} {n : ℕ}

-- Conditions: Given the sum of the first n terms of a geometric series,
-- Here, in Lean, we state the finite geometric sum and the infinite geometric sum.
def geometric_sum_n := a * (1 - r^n) / (1 - r)
def geometric_sum_infinite (hr : |r| < 1) := a / (1 - r)

-- Proof to show the denominator function common to both scenarios.
theorem geometric_series_denominator (hr : |r| < 1) : 
  (1 - r) = (1 - r) :=
by
  sorry

end geometric_series_denominator_l706_706384


namespace exists_t_ge_two_l706_706192

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x^2 + a * x + b

theorem exists_t_ge_two (a b : ℝ) :
  ∃ t ∈ set.Icc (0 : ℝ) (4 : ℝ), abs (f a b t) ≥ 2 :=
begin
  sorry
end

end exists_t_ge_two_l706_706192


namespace find_vector_b_l706_706263

noncomputable def vec_a : ℝ × ℝ × ℝ := (5, -3, -6)
noncomputable def vec_c : ℝ × ℝ × ℝ := (-1, -2, 3)
noncomputable def vec_b : ℝ × ℝ × ℝ := (2, -5/2, 3/2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def collinear (u v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, w = (u.1 + t * v.1, u.2 + t * v.2, u.3 + t * v.3)

def bisects_angle (u v w : ℝ × ℝ × ℝ) : Prop :=
  dot_product u v * magnitude w = dot_product v w * magnitude u

theorem find_vector_b :
  collinear vec_a vec_c vec_b ∧ bisects_angle vec_a vec_b vec_c :=
sorry

end find_vector_b_l706_706263


namespace problem_l706_706186

def f (x : ℝ) : ℝ := (x + 3) / (x + 1)

noncomputable def m : ℝ := f 1 + f 2 + f 4 + f 8 + f 16
noncomputable def n : ℝ := f (1/2) + f (1/4) + f (1/8) + f (1/16)

theorem problem : m + n = 18 := by
  have h_f_sum : ∀ x ≠ 0, f x + f (1 / x) = 4 := by
    intro x hx
    calc
      f x + f (1 / x) = (x + 3) / (x + 1) + (1 / x + 3) / (1 / x + 1) : by refl
                     ... = (x + 3) / (x + 1) + (3 * x + 1) / (x + 1) : by field_simp [hx]
                     ... = (x + 3 + 3 * x + 1) / (x + 1) : by rw [add_div]
                     ... = 4 : by ring

  have h_m : m = 2 + 4 * 4 := by
    calc
      m = f 1 + (f 2 + f (1 / 2)) + (f 4 + f (1 / 4)) + (f 8 + f (1 / 8)) + (f 16 + f (1 / 16)) : by firstly rfl
      ... = 2 + 4 + 4 + 4 + 4 : by iterate { congr' 1; exact h_f_sum _ dec_trivial}

  have h_n : n = 4 * 4 := by
    calc
      n = f (1 / 2) + f (1 / 4) + f (1 / 8) + f (1 / 16) : by firstly rfl
      ... = 4 + 4 + 4 + 4 : by iterate { exact h_f_sum _ dec_trivial }

  calc
    m + n = (2 + 4 * 4) + 4 * 4 : by rw [h_m, h_n]
    ... = 2 + 4 * 4 + 4 * 4 : by ring
    ... = 2 + 16 + 16 : by ring
    ... = 34 : by norm_num


end problem_l706_706186


namespace determinant_of_matrix_l706_706489

def mat : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 0, 2],![8, 5, -2],![3, 3, 6]]

theorem determinant_of_matrix : Matrix.det mat = 90 := 
by 
  sorry

end determinant_of_matrix_l706_706489


namespace point_A_inside_circle_l706_706594

-- Define the conditions
def radius (O : Type) [metric_space O] : ℝ := 6
def distance_to_center (O : Type) [metric_space O] (A : O) (center : O) : ℝ := 5

-- Define the property to be proved
theorem point_A_inside_circle (O : Type) [metric_space O] (A : O) (center : O) 
  (h1 : radius O = 6) (h2 : distance_to_center O A center = 5) : 
  (distance A center < radius O) :=
by
  rw [h2, h1]
  exact lt_add_one 5

end

end point_A_inside_circle_l706_706594


namespace log2_a_plus_log2_b_zero_l706_706983

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_a_plus_log2_b_zero 
    (a b : ℝ) 
    (h : (Nat.choose 6 3) * (a^3) * (b^3) = 20) 
    (hc : (a^2 + b / a)^(3) = 20 * x^(3)) :
  log2 a + log2 b = 0 :=
by
  sorry

end log2_a_plus_log2_b_zero_l706_706983


namespace tournament_participants_count_l706_706656

theorem tournament_participants_count (num_matches : ℕ) (num_remaining : ℕ) 
  (losses_to_eliminate : ℕ) : 
  num_matches = 29 ∧ num_remaining = 2 ∧ losses_to_eliminate = 2 → 
  ∃ participants : ℕ, participants = 16 :=
by
  intros
  use 16
  sorry

end tournament_participants_count_l706_706656


namespace find_m_value_l706_706202

-- Defining vectors a, b, and c
def vec_a : ℝ × ℝ := (-1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vec_c : ℝ × ℝ := (3, -2)

-- Function to compute the difference between vectors a and b
def vec_diff (m : ℝ) : ℝ × ℝ :=
  let (ax, ay) := vec_a
  let (bx, by) := vec_b m
  (ax - bx, ay - by)

-- Dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  x1 * x2 + y1 * y2

-- Defining the perpendicularity condition
def perp_condition (m : ℝ) : Prop :=
  dot_product (vec_diff m) vec_c = 0

-- Theorem to prove
theorem find_m_value : perp_condition (-3) :=
  by
    -- This is just the statement, and the proof is omitted with sorry.
    sorry

end find_m_value_l706_706202


namespace cans_collected_by_first_group_l706_706321

def class_total_students : ℕ := 30
def students_didnt_collect : ℕ := 2
def students_collected_4 : ℕ := 13
def total_cans_collected : ℕ := 232

theorem cans_collected_by_first_group :
  let remaining_students := class_total_students - (students_didnt_collect + students_collected_4)
  let cans_by_13_students := students_collected_4 * 4
  let cans_by_first_group := total_cans_collected - cans_by_13_students
  let cans_per_student := cans_by_first_group / remaining_students
  cans_per_student = 12 := by
  sorry

end cans_collected_by_first_group_l706_706321


namespace find_annual_compound_interest_rate_l706_706005

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_compound_interest_rate :
  compound_interest_rate 10000 24882.50 1 7 0.125 :=
by sorry

end find_annual_compound_interest_rate_l706_706005


namespace total_bill_is_95_l706_706072

noncomputable def total_bill := 28 + 8 + 10 + 6 + 14 + 11 + 12 + 6

theorem total_bill_is_95 : total_bill = 95 := by
  sorry

end total_bill_is_95_l706_706072


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706339

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706339


namespace total_lives_after_third_level_l706_706231

variable (x y : ℕ) (s_0 P : ℤ)
variable (s_0_cond : s_0 = 2) (x_cond : x = 5) (y_cond : y = 4) (P_cond : P = 3)

theorem total_lives_after_third_level : 
  let E_1 := 2 * x in 
  let s_1 := s_0 + E_1 - P in 
  let E_2 := 3 * y in 
  let M := s_1 / 2 in 
  let s_2 := s_1 + E_2 - M in 
  let z := x + 2 * y - 5 in 
  s_2 + z = 25 :=
by 
  sorry

end total_lives_after_third_level_l706_706231


namespace eccentricity_is_sqrt_three_div_three_l706_706162

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  sqrt (1 - (b^2 / a^2))

theorem eccentricity_is_sqrt_three_div_three (a b : ℝ) (h : a > b ∧ b > 0) 
  (h_eq : 2 * (sqrt (1 + a^2 / b^2) * (0 - -((2 * a^3 * b^2) / (b^4 + a^4)))) = 
          3 * (sqrt (1 + a^2 / b^2) * (a - (a^5 - a * b^4) / (a^4 + b^4)))) :
  eccentricity_of_ellipse a b h = sqrt 3 / 3 :=
by
  sorry

end eccentricity_is_sqrt_three_div_three_l706_706162


namespace smallest_sum_of_consecutive_integers_is_square_l706_706353

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l706_706353


namespace trisectors_of_triangle_form_equilateral_l706_706726

-- Given an arbitrary triangle ABC with angles A, B, and C
def intersection_points_form_equilateral_triangle (A B C : Type) [linear_ordered_field A]
  (triangle : triangle A) 
  (trisectors : ∀a b c : point A, list (line A))
  (adjacent_intersections : point A) : Prop :=
  is_equilateral_triangle adjacent_intersections

-- The theorem to be proved
theorem trisectors_of_triangle_form_equilateral (A B C : Type) [linear_ordered_field A]
  (triangle : triangle A)
  (trisectors : ∀a b c : point A, list (line A))
  (adjacent_intersections : set (point A)) :
  intersection_points_form_equilateral_triangle A B C triangle trisectors adjacent_intersections :=
sorry

end trisectors_of_triangle_form_equilateral_l706_706726


namespace subset_A_iff_l706_706936

noncomputable def A : set ℝ := {x | x < -1 ∨ x ≥ 3}
noncomputable def B (a : ℝ) : set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_A_iff (a : ℝ) : (B a ⊆ A) ↔ (-1/3 ≤ a ∧ a < 1) :=
sorry

end subset_A_iff_l706_706936


namespace total_beats_james_hears_l706_706253

theorem total_beats_james_hears :
  let monday_beats := 2 * 180 * 60,
      tuesday_beats := (1 * 200 * 60) + (1 * 150 * 60),
      wednesday_beats := (1.5 * 190 * 60) + (0.5 * 210 * 60),
      thursday_beats := (1 * 170 * 60) + (1 * 230 * 60),
      friday_beats := 2 * 220 * 60,
      saturday_beats := (1.5 * 200 * 60) + (0.5 * 140 * 60),
      sunday_beats := 2 * 205 * 60
  in monday_beats + tuesday_beats + wednesday_beats + thursday_beats + friday_beats + saturday_beats + sunday_beats = 163200 := by
  sorry

end total_beats_james_hears_l706_706253


namespace min_spiders_sufficient_spiders_l706_706029

def grid_size : ℕ := 2019

noncomputable def min_k_catch (k : ℕ) : Prop :=
∀ (fly spider1 spider2 : ℕ × ℕ) (fly_move spider1_move spider2_move: ℕ × ℕ → ℕ × ℕ), 
  (fly_move fly = fly ∨ fly_move fly = (fly.1 + 1, fly.2) ∨ fly_move fly = (fly.1 - 1, fly.2)
  ∨ fly_move fly = (fly.1, fly.2 + 1) ∨ fly_move fly = (fly.1, fly.2 - 1))
  ∧ (spider1_move spider1 = spider1 ∨ spider1_move spider1 = (spider1.1 + 1, spider1.2) ∨ spider1_move spider1 = (spider1.1 - 1, spider1.2)
  ∨ spider1_move spider1 = (spider1.1, spider1.2 + 1) ∨ spider1_move spider1 = (spider1.1, spider1.2 - 1))
  ∧ (spider2_move spider2 = spider2 ∨ spider2_move spider2 = (spider2.1 + 1, spider2.2) ∨ spider2_move spider2 = (spider2.1 - 1, spider2.2)
  ∨ spider2_move spider2 = (spider2.1, spider2.2 + 1) ∨ spider2_move spider2 = (spider2.1, spider2.2 - 1))
  → (spider1 = fly ∨ spider2 = fly)

theorem min_spiders (k : ℕ) : min_k_catch k → k ≥ 2 :=
sorry

theorem sufficient_spiders : min_k_catch 2 :=
sorry

end min_spiders_sufficient_spiders_l706_706029


namespace trig_expression_evaluation_l706_706949

theorem trig_expression_evaluation
  (α : ℝ)
  (h_tan_α : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
  sorry

end trig_expression_evaluation_l706_706949


namespace lean_solution_l706_706385

open Nat

theorem lean_solution 
  (h₁ : ∀ x, (5 * x ≡ 1 [MOD 17]) → (x ≡ 7 [MOD 17])) 
  (h₂ : ∀ y, (25 * y ≡ 1 [MOD 17]) → (y ≡ 15 [MOD 17])) 
  (h₃ : ∀ z, (125 * z ≡ 1 [MOD 17]) → (z ≡ 2 [MOD 17])):
  (5⁻¹ + 5⁻² + 5⁻³) % 17 = 7 := by
  sorry

end lean_solution_l706_706385


namespace area_decrease_l706_706471

-- Definitions for the problem
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * (s ^ 2)

-- Given conditions
def initial_area : ℝ := 81 * sqrt 3
def side_length : ℝ := 18  -- derived from the initial area

theorem area_decrease (new_side_length : ℝ := side_length - 3) :
  (area_of_equilateral_triangle side_length - area_of_equilateral_triangle new_side_length) = 24.75 * sqrt 3 :=
by
  sorry

end area_decrease_l706_706471


namespace planes_perpendicular_l706_706944

variables {m n : Type} -- lines
variables {α β : Type} -- planes

axiom lines_different : m ≠ n
axiom planes_different : α ≠ β
axiom parallel_lines : ∀ (m n : Type), Prop -- m ∥ n
axiom parallel_plane_line : ∀ (m α : Type), Prop -- m ∥ α
axiom perp_plane_line : ∀ (n β : Type), Prop -- n ⊥ β
axiom perp_planes : ∀ (α β : Type), Prop -- α ⊥ β

theorem planes_perpendicular 
  (h1 : parallel_lines m n) 
  (h2 : parallel_plane_line m α) 
  (h3 : perp_plane_line n β) 
: perp_planes α β := 
sorry

end planes_perpendicular_l706_706944


namespace smallest_sum_of_consecutive_integers_is_square_l706_706356

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l706_706356


namespace root_condition_l706_706416

noncomputable def f (x t : ℝ) := x^2 + t * x - t

theorem root_condition {t : ℝ} : (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) := 
  sorry

end root_condition_l706_706416


namespace quadratic_function_properties_l706_706223

theorem quadratic_function_properties :
  (∃ (b c : ℝ), (∀ x : ℝ, f x = x^2 + b * x + c) ∧ 
  (f 0 = 2 ∧ f 1 = 0) ∧ b = -3 ∧ c = 2 ∧
  (∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ h = 3 / 2 ∧ k = -1 / 4)) :=
begin
  sorry
end

end quadratic_function_properties_l706_706223


namespace math_problem_l706_706977

theorem math_problem (d r : ℕ) (hd : d > 1)
  (h1 : 1259 % d = r) 
  (h2 : 1567 % d = r) 
  (h3 : 2257 % d = r) : d - r = 1 :=
by
  sorry

end math_problem_l706_706977


namespace crease_lines_location_l706_706370

noncomputable def locus_of_points (R a : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - a / 2)^2 / (R / 2)^2 + p.snd^2 / ((R / 2)^2 - (a / 2)^2) ≥ 1}

theorem crease_lines_location (R a : ℝ) (h : 0 < R) (h' : 0 < a ∧ a < R) : 
  ∀ (p : ℝ × ℝ), p ∈ locus_of_points R a ↔ 
    (∃ A' ∈ (metric.sphere (0 : ℝ × ℝ) R), crease_line_through p A' a) :=
by sorry

end crease_lines_location_l706_706370


namespace number_of_broccoli_l706_706375

variable (cost_chicken : ℝ)
variable (cost_lettuce : ℝ)
variable (cost_tomatoes : ℝ)
variable (cost_potatoes : ℝ)
variable (cost_brussel_sprouts : ℝ)
variable (remaining : ℝ)
variable (broccoli_price : ℝ)

variable (total_minimum : ℝ)
variable (additional_amount : ℝ)

#check (cost_chicken = 1.5 * 6)
#check (cost_lettuce = 3)
#check (cost_tomatoes = 2.5)
#check (cost_potatoes = 4 * 0.75)
#check (cost_brussel_sprouts = 2.5)
#check (remaining = total_minimum - (cost_chicken + cost_lettuce + cost_tomatoes + cost_potatoes + cost_brussel_sprouts))
#check (remaining = 15)
#check (broccoli_price = 2)
#check (additional_amount = remaining - 11)

theorem number_of_broccoli (total_minimum : ℝ) (cost : ℝ) (required_amount : ℝ) : (required_amount = 11) ∧ (total_minimum = 35) ∧ (cost = 20) → 
required_amount + broccoli_price * (required_amount / broccoli_price) = 15 → additional_amount = required_amount :
sorry

end number_of_broccoli_l706_706375


namespace right_angle_triangle_division_l706_706108

theorem right_angle_triangle_division {A B C : Type} [euclidean_space A] 
  (hC : right_angle C A B) (hA : angle A C B = 30) :
  ∃ D E, is_median A D E ∧ is_angle_bisector B E A ∧ 
    parallel (line_through C E) (line_through B D) :=
sorry

end right_angle_triangle_division_l706_706108


namespace distance_from_nashville_to_miami_l706_706467

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_nashville_to_miami :
  distance 1170 1560 1950 780 = 1103 :=
  sorry

end distance_from_nashville_to_miami_l706_706467


namespace range_arcsin_b_l706_706593

open Real

theorem range_arcsin_b (a b : ℝ) (h₁ : a x^{2} + b y^{2} = 1) (h₂ : x^{2} + y^{2} = 2/√3) :
  ⟦π / 6, π / 4⟧ ∪ ⟦π / 4, π / 3⟧ = arcsin b := 
sorry

end range_arcsin_b_l706_706593


namespace intersect_or_parallel_l706_706734

variables {Point : Type*} [metric_space Point]

def are_distinct (A B C D E F : Point) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F

def no_four_on_one_circle (A B C D E F : Point) : Prop := sorry -- Implement geometric condition that no four points lie on one circle

def no_two_parallel_segments (A B C D E F : Point) : Prop := sorry -- Implement geometric condition that no two segments from these points are parallel

def are_intersection_points (A B C D E F P Q R P' Q' R' : Point) : Prop := sorry -- Implement the definitions for P, Q, R, P', Q', R' as specified

theorem intersect_or_parallel
  {A B C D E F P Q R P' Q' R' : Point}
  (hdistinct : are_distinct A B C D E F)
  (hcircle : no_four_on_one_circle A B C D E F)
  (hparallel : no_two_parallel_segments A B C D E F)
  (hintersection : are_intersection_points A B C D E F P Q R P' Q' R') :
  P ≠ P' ∧ Q ≠ Q' ∧ R ≠ R' ∧ (line_through P P', line_through Q Q', line_through R R') intersect_or_parallel :=
sorry

end intersect_or_parallel_l706_706734


namespace school_avg_GPA_l706_706748

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l706_706748


namespace arrange_books_l706_706295

theorem arrange_books :
  let russian_books : Finset String := {"R1", "R2", "R3"}
  let french_books : Finset String := {"F1", "F2"}
  let italian_books : Finset String := {"I1", "I2", "I3"}
  let books := russian_books ∪ french_books ∪ italian_books
  (books.card = 8) →
  (russian_books.card = 3) →
  (french_books.card = 2) →
  (italian_books.card = 3) →
  let arranged_units : ℕ := 5
  (factorial arranged_units) * (factorial 3) * (factorial 2) = 1440 := sorry

end arrange_books_l706_706295


namespace first_nonzero_digit_one_div_1029_l706_706804

theorem first_nonzero_digit_one_div_1029 : 
  (Nat.digits 10 (Nat.div 10290000 1029)).head = 9 := 
sorry

end first_nonzero_digit_one_div_1029_l706_706804


namespace card_draw_count_l706_706790

theorem card_draw_count : 
  let total_cards := 12
  let red_cards := 3
  let yellow_cards := 3
  let blue_cards := 3
  let green_cards := 3
  let total_ways := Nat.choose total_cards 3
  let invalid_same_color := 4 * Nat.choose 3 3
  let invalid_two_red := Nat.choose red_cards 2 * Nat.choose (total_cards - red_cards) 1
  total_ways - invalid_same_color - invalid_two_red = 189 :=
by
  sorry

end card_draw_count_l706_706790


namespace find_a_l706_706602

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (a - 1) * x

theorem find_a {a : ℝ} : 
  (∀ x : ℝ, 0 < x → f x a ≤ x^2 * Real.exp x - Real.log x - 4 * x - 1) → 
  a ≤ -2 :=
sorry

end find_a_l706_706602


namespace smallest_consecutive_sum_perfect_square_l706_706350

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l706_706350


namespace no_fractions_meet_condition_l706_706512

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l706_706512


namespace ticket_price_difference_l706_706716

noncomputable def price_difference (adult_price total_cost : ℕ) (num_adults num_children : ℕ) (child_price : ℕ) : ℕ :=
  adult_price - child_price

theorem ticket_price_difference :
  ∀ (adult_price total_cost num_adults num_children child_price : ℕ),
  adult_price = 19 →
  total_cost = 77 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_price + num_children * child_price = total_cost →
  price_difference adult_price total_cost num_adults num_children child_price = 6 :=
by
  intros
  simp [price_difference]
  sorry

end ticket_price_difference_l706_706716


namespace perpendicular_bisectors_intersect_at_one_point_l706_706577

-- Define the key geometric concepts
variables {Point : Type*} [MetricSpace Point]

-- Define the given conditions 
variables (A B C M : Point)
variables (h1 : dist M A = dist M B)
variables (h2 : dist M B = dist M C)

-- Define the theorem to be proven
theorem perpendicular_bisectors_intersect_at_one_point :
  dist M A = dist M C :=
by 
  -- Proof to be filled in later
  sorry

end perpendicular_bisectors_intersect_at_one_point_l706_706577


namespace area_ratio_of_circles_l706_706982

-- Define the circles and lengths of arcs
variables {R_C R_D : ℝ} (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D))

-- Theorem proving the ratio of the areas
theorem area_ratio_of_circles (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 := sorry

end area_ratio_of_circles_l706_706982


namespace exists_rectangle_with_properties_l706_706523

variables {e a φ : ℝ}

-- Define the given conditions
def diagonal_diff (e a : ℝ) := e - a
def angle_between_diagonals (φ : ℝ) := φ

-- The problem to prove
theorem exists_rectangle_with_properties (e a φ : ℝ) 
  (h_diff : diagonal_diff e a = e - a) 
  (h_angle : angle_between_diagonals φ = φ) : 
  ∃ (rectangle : Type) (A B C D : rectangle), 
    (e - a = e - a) ∧ 
    (φ = φ) := 
sorry

end exists_rectangle_with_properties_l706_706523


namespace f_of_13_eq_223_l706_706218

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_of_13_eq_223 : f 13 = 223 := 
by sorry

end f_of_13_eq_223_l706_706218


namespace count_multiples_of_67_l706_706066

def a (n k : ℕ) : ℕ := 2^(n-1) * (n + 2 * k - 2)

theorem count_multiples_of_67 : (finset.range 51).sum (λ n, (finset.range (51 - n)).count (λ k, 67 ∣ a n k)) = 17 := 
sorry

end count_multiples_of_67_l706_706066


namespace determinant_of_A_l706_706488

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![8, 5, -4], ![3, 3, 7]]  -- Defining matrix A

def A' : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![5, 4, -2], ![0, 2, 9]]  -- Defining matrix A' after row operations

theorem determinant_of_A' : Matrix.det A' = 55 := by -- Proving that the determinant of A' is 55
  sorry

end determinant_of_A_l706_706488


namespace asymptote_of_hyperbola_l706_706319

theorem asymptote_of_hyperbola : 
  ∀ x y : ℝ, (y^2 / 4 - x^2 = 1) → (y = 2 * x) ∨ (y = -2 * x) := 
by
  sorry

end asymptote_of_hyperbola_l706_706319


namespace sin_angle_TPU_in_cube_l706_706823

noncomputable def sin_angle_in_cube {P Q R S T U : Point} (h_cube : IsCube P Q R S T U) : ℝ :=
  sin (angle T P U)

theorem sin_angle_TPU_in_cube 
  {P Q R S T U : Point} 
  (h_cube : IsCube P Q R S T U) 
  : sin_angle_in_cube h_cube = real.sin (real.pi / 3) :=
sorry

end sin_angle_TPU_in_cube_l706_706823


namespace expression_evaluation_l706_706090

theorem expression_evaluation : abs (abs (-abs (-2 + 1) - 2) + 2) = 5 := 
by  
  sorry

end expression_evaluation_l706_706090


namespace nikola_charge_per_leaf_l706_706287

theorem nikola_charge_per_leaf
    (num_ants : ℕ)
    (food_per_ant : ℕ)
    (cost_per_ounce : ℝ)
    (num_leaves : ℕ)
    (num_jobs : ℕ)
    (charge_per_job : ℝ)
    (total_cost : ℝ)
    (h1 : num_ants = 400)
    (h2 : food_per_ant = 2)
    (h3 : cost_per_ounce = 0.1)
    (h4 : num_leaves = 6000)
    (h5 : num_jobs = 4)
    (h6 : charge_per_job = 5)
    (total_saved : ℝ)
    (h7 : total_saved = total_cost)
    : (charge_per_leaf : ℝ) := 
by
  have A1 : total_cost = (num_ants * food_per_ant : ℕ) * cost_per_ounce := sorry
  have A2 : total_saved = num_jobs * charge_per_job + num_leaves * charge_per_leaf := sorry
  have A3 : A1 = A2 := sorry
  have A4 : charge_per_leaf = 0.01 := sorry
  exact A4

end nikola_charge_per_leaf_l706_706287


namespace _l706_706273

noncomputable theorem complex_sum_product_identity :
  let z : ℂ := (1 - complex.I * real.sqrt 3) / 2 in
  (∑ k in finset.range (16 + 1), z ^ (k^2)) * (∑ k in finset.range (16 + 1), (z ^ (k^2))⁻¹) = 9 :=
by
  sorry

end _l706_706273


namespace product_lcm_gcd_9_12_l706_706562

theorem product_lcm_gcd_9_12 : Nat.gcd 9 12 * Nat.lcm 9 12 = 108 := by
  have h1 : Nat.gcd 9 12 = 3 := by
    sorry -- Proof that gcd 9 12 = 3

  have h2 : Nat.lcm 9 12 = 36 := by
    sorry -- Proof that lcm 9 12 = 36
  
  calc
    Nat.gcd 9 12 * Nat.lcm 9 12
    = 3 * 36 : by rw [h1, h2]
    = 108 : by norm_num

end product_lcm_gcd_9_12_l706_706562


namespace count_valid_arrangements_l706_706788

-- Define the conditions
namespace TextbookArrangement

-- Conditions given in the problem
def textbooks : List String := ["Chinese1", "Chinese2", "Math1", "Math2", "Physics"]

-- Function to check if the arrangement is valid
def isValidArrangement (arrangement : List String) : Prop :=
  ∀ i, i < arrangement.length - 1 → (arrangement[i] = "Chinese" → arrangement[i+1] ≠ "Chinese")
                              ∧ (arrangement[i] = "Math" → arrangement[i+1] ≠ "Math")
                              ∧ (arrangement[i] = "Physics" → arrangement[i+1] ≠ "Physics")

-- Main statement to prove
theorem count_valid_arrangements (arrangements : List (List String)) :
  (arrangements.countp isValidArrangement) = 48 := sorry

end TextbookArrangement

end count_valid_arrangements_l706_706788


namespace polygon_at_least_9_sides_l706_706629

theorem polygon_at_least_9_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ θ, θ < 45 ∧ (∀ j, 1 ≤ j ∧ j ≤ n → θ = 360 / n))):
  9 ≤ n :=
sorry

end polygon_at_least_9_sides_l706_706629


namespace num_dessert_menus_l706_706830

-- Define the types and conditions
inductive Dessert
| cake
| pie
| ice_cream
| pudding

open Dessert

def is_valid_menu (menu : List Dessert) :=
  menu.length = 7 ∧
  menu.nth 4 = some cake ∧  -- cake on the 5th index (Friday).
  ∀ i, i < 6 → menu.nth i ≠ menu.nth (i + 1)  -- No two consecutive days have the same dessert.

theorem num_dessert_menus : ∃ (menus : List (List Dessert)), menus.length = 729 ∧
  ∀ menu ∈ menus, is_valid_menu menu :=
sorry

end num_dessert_menus_l706_706830


namespace jugglers_count_l706_706779

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end jugglers_count_l706_706779


namespace no_fractions_meet_condition_l706_706511

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l706_706511


namespace fred_balloons_l706_706567

theorem fred_balloons (original_balloons given_balloons remaining_balloons : ℕ) 
    (h_orig : original_balloons = 709) 
    (h_given : given_balloons = 221) 
    (h_compute : remaining_balloons = original_balloons - given_balloons) : 
    remaining_balloons = 488 :=
by 
  rw [h_orig, h_given, h_compute]
  sorry

end fred_balloons_l706_706567


namespace no_function_f_exists_l706_706536

theorem no_function_f_exists :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n)) = n + 1987 :=
sorry

end no_function_f_exists_l706_706536


namespace root_derivative_in_hull_l706_706155

open Complex

noncomputable def f (n : ℕ) (zs : Fin n → ℂ) : ℂ → ℂ :=
  λ z => ∏ i in Finset.finRange n, (z - zs i)

noncomputable def f_prime (n : ℕ) (zs : Fin n → ℂ) : ℂ → ℂ :=
  λ z => ∑ i in Finset.finRange n, 
    (∏ j in (Finset.finRange n).erase i, (z - zs j))

def convex_hull (n : ℕ) (zs : Fin n → ℂ) : Set ℂ :=
  {ω | ∃ α : Fin n → ℝ, (∀ i, α i ≥ 0) ∧ (Finset.univ.sum α = 1) ∧ (ω = Finset.univ.sum (λ i, α i • zs i))}

theorem root_derivative_in_hull 
  {n : ℕ} (h_pos : 0 < n) (zs : Fin n → ℂ) 
  (Omega : Set ℂ) (Omega_def : Omega = convex_hull n zs)
  (ω : ℂ) (h_omega : f_prime n zs ω = 0) :
  ω ∈ Omega :=
  sorry

end root_derivative_in_hull_l706_706155


namespace no_positive_integer_solutions_infinite_integer_solutions_l706_706304

theorem no_positive_integer_solutions : ¬ ∃ (x y z : ℤ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (4 * x * y - x - y = z^2) := 
sorry

theorem infinite_integer_solutions : ∃ (f : ℕ → ℤ × ℤ × ℤ), ∀ n : ℕ, let (x, y, z) := f n in 4 * x * y - x - y = z^2 := 
sorry

end no_positive_integer_solutions_infinite_integer_solutions_l706_706304


namespace OI_perpendicular_MN_l706_706692

variables {A B C D E F P Q M N : Point}
variables (O I : Point) [Circumcenter O A B C] [Incenter I A B C]
variables [IncircleTouchPoints D E F A B C]
variables [FDIntersectsCA P F D] [DEIntersectsAB Q D E]
variables [Midpoint M P E] [Midpoint N Q F]

theorem OI_perpendicular_MN :
  Perpendicular (LineThrough O I) (LineThrough M N) :=
sorry

end OI_perpendicular_MN_l706_706692


namespace correct_options_l706_706938

-- Define the set M and the conditions
def M (x : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∨
  (∃ (y z : ℝ), M y ∧ M z ∧ x = y - z) ∨
  (∃ (y : ℝ), y ≠ 0 ∧ M y ∧ x = 1 / y)

-- Theorem statement
theorem correct_options (M_set : set ℝ) (h1 : 0 ∈ M_set) (h2 : 1 ∈ M_set)
  (h3 : ∀ x y, x ∈ M_set ∧ y ∈ M_set → (x - y) ∈ M_set)
  (h4 : ∀ x, x ∈ M_set ∧ x ≠ 0 → (1 / x) ∈ M_set) :
  (1 / 3 ∈ M_set) ∧ (-1 ∈ M_set) ∧
  (∀ x y, x ∈ M_set ∧ y ∈ M_set → (x + y) ∈ M_set) ∧
  (∀ x, x ∈ M_set → (x ^ 2) ∈ M_set) :=
by
  -- Placeholder for the proof
  sorry

end correct_options_l706_706938


namespace existence_1978_digits_nonexistence_1977_digits_l706_706703

def digits := Fin 10 -- Representing digits from 0 to 9
def number (n : ℕ) := Fin n -> digits -- Representing an n-digit number as a function from Fin n to digits

-- Problem 1: Existence for 1978 digits
theorem existence_1978_digits : 
  ∃ x y : number 1978, (∀ i : Fin 1978, (x i + y i) % 10 = 9) ∧ (x ≠ y) :=
sorry

-- Problem 2: Non-existence for 1977 digits
theorem nonexistence_1977_digits : 
  ¬ ∃ x y : number 1977, (∀ i : Fin 1977, (x i + y i) % 10 = 9) ∧ (x ≠ y) :=
sorry

end existence_1978_digits_nonexistence_1977_digits_l706_706703


namespace fraction_of_undeclared_students_l706_706230

-- Definitions based on the conditions
def total_students := 100
def first_year_students := total_students * (1/5 : ℚ)
def second_year_students := total_students * (2/5 : ℚ)
def third_year_students := total_students * (1/5 : ℚ)
def fourth_year_students := total_students * (1/10 : ℚ)
def postgraduate_students := total_students * (1/10 : ℚ)

def first_year_undeclared := first_year_students * (4/5 : ℚ)
def second_year_undeclared := second_year_students * (3/4 : ℚ)
def third_year_undeclared := (third_year_students * (1/3 : ℚ)).round
def fourth_year_undeclared := (fourth_year_students * (1/6 : ℚ)).round
def postgraduate_undeclared := (postgraduate_students * (1/12 : ℚ)).round

-- Total number of undeclared students
def total_undeclared := first_year_undeclared + second_year_undeclared + third_year_undeclared + fourth_year_undeclared + postgraduate_undeclared

-- Fraction of all students who have not declared a major
def fraction_undeclared (total_students : ℚ) (total_undeclared : ℚ) : ℚ :=
  total_undeclared / total_students

theorem fraction_of_undeclared_students :
  fraction_undeclared total_students total_undeclared = 14 / 25 :=
by
  sorry

end fraction_of_undeclared_students_l706_706230


namespace evaluation_result_l706_706111

theorem evaluation_result : 
  (Int.floor (Real.ceil ((15/8 : Real)^2) + (11/3 : Real)) = 7) := 
sorry

end evaluation_result_l706_706111


namespace part1_result_part2_result_l706_706092

noncomputable def part1_expr : ℝ := log 5 * log 20 - log 2 * log 50 - log 25

theorem part1_result : part1_expr = -1 :=
by
  sorry

variables {a b : ℝ}

noncomputable def part2_expr (a b : ℝ) : ℝ :=
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6))

theorem part2_result (a b : ℝ) : part2_expr a b = 4 * a :=
by
  sorry

end part1_result_part2_result_l706_706092


namespace initial_investment_l706_706073

noncomputable def compound_interest_inv (A r : ℝ) (n t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem initial_investment :
  compound_interest_inv 7372.46 0.065 1 2 ≈ 6510.00 := by 
  sorry

end initial_investment_l706_706073


namespace number_of_such_fractions_is_one_l706_706522

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l706_706522


namespace probability_inequality_l706_706402

theorem probability_inequality :
  let S := {x : ℕ | x > 0 ∧ x < 10}
  let favorable := {x ∈ S | 8 / x > x}
  (favorable.card : ℚ) / (S.card : ℚ) = 1 / 3 :=
by
  let S := {x : ℕ | x > 0 ∧ x < 10}
  let favorable := {x ∈ S | 8 / x > x}
  have S_card : S.card = 9 := by sorry
  have favorable_card : favorable.card = 3 := by sorry
  calc
    (favorable.card : ℚ) / (S.card : ℚ) = (3 : ℚ) / (9 : ℚ) : by rw [favorable_card, S_card]
    ... = 1 / 3 : by norm_num

end probability_inequality_l706_706402


namespace hat_coloring_possible_l706_706308

-- We define our graph structure and necessary properties.
structure Graph (V : Type) :=
(E : V → V → Prop)
(symm : ∀ {x y : V}, E x y → E y x)
(loopfree : ∀ {x : V}, ¬E x x)

-- The main theorem to show the existence of such a coloring.
theorem hat_coloring_possible (V : Type) [Finite V] 
  (G : Graph V) (deg_bounds : ∀ v : V, 50 ≤ Finset.card (Finset.filter (G.E v) (Finset.univ : Finset V)) ∧ Finset.card (Finset.filter (G.E v) (Finset.univ : Finset V)) ≤ 100) :
  ∃ (coloring : V → Fin 1331), ∀ v : V, ∃ (neighbor_colors : Finset (Fin 1331)), Finset.card neighbor_colors ≥ 20 ∧ (∀ u ∈ Finset.filter (G.E v) Finset.univ, coloring u ∈ neighbor_colors) := 
sorry

end hat_coloring_possible_l706_706308


namespace distance_AB_eq_sqrt_10_l706_706235

def point := (ℝ × ℝ × ℝ)

noncomputable def distance (A B : point) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2.1 - A.2.1) ^ 2 + (B.2.2 - A.2.2) ^ 2)

theorem distance_AB_eq_sqrt_10 :
  distance (1, 0, 2) (1, -3, 1) = real.sqrt 10 := by
  sorry

end distance_AB_eq_sqrt_10_l706_706235


namespace factory_dolls_per_day_l706_706859

-- Define the number of normal dolls made per day
def N : ℝ := 4800

-- Define the total number of dolls made per day as 1.33 times the number of normal dolls
def T : ℝ := 1.33 * N

-- The theorem statement to prove the factory makes 6384 dolls per day
theorem factory_dolls_per_day : T = 6384 :=
by
  -- Proof here
  sorry

end factory_dolls_per_day_l706_706859


namespace compute_fraction_square_l706_706486

theorem compute_fraction_square : 6 * (3 / 7) ^ 2 = 54 / 49 :=
by 
  sorry

end compute_fraction_square_l706_706486


namespace scientific_notation_of_330100000_l706_706728

theorem scientific_notation_of_330100000 :
  let n := 330100000 in
  n = 3.301 * 10^8 := 
by
  sorry

end scientific_notation_of_330100000_l706_706728


namespace count_monomials_l706_706242

def isMonomial (expr : String) : Bool :=
  match expr with
  | "m+n" => false
  | "2x^2y" => true
  | "1/x" => true
  | "-5" => true
  | "a" => true
  | _ => false

theorem count_monomials :
  let expressions := ["m+n", "2x^2y", "1/x", "-5", "a"]
  (expressions.filter isMonomial).length = 3 :=
by { sorry }

end count_monomials_l706_706242


namespace new_bucket_capacity_l706_706428

theorem new_bucket_capacity (init_buckets : ℕ) (init_capacity : ℕ) (new_buckets : ℕ) (total_volume : ℕ) :
  init_buckets * init_capacity = total_volume →
  new_buckets * 9 = total_volume →
  9 = total_volume / new_buckets :=
by
  intros h₁ h₂
  sorry

end new_bucket_capacity_l706_706428


namespace brother_books_total_l706_706864

theorem brother_books_total (pb_sarah hb_sarah : ℕ) (h_pb_sarah : pb_sarah = 6) (h_hb_sarah : hb_sarah = 4) : 
  let pb_brother := pb_sarah / 3 in
  let hb_brother := 2 * hb_sarah in
  pb_brother + hb_brother = 10 :=
by
  have h_pb_brother : pb_brother = 2 := by rw [h_pb_sarah] ; exact Nat.div_eq_of_lt (by decide) -- 6 / 3 = 2
  have h_hb_brother : hb_brother = 8 := by rw [h_hb_sarah] ; exact by norm_num -- 4 * 2 = 8
  rw [h_pb_brother, h_hb_brother]
  norm_num  -- 2 + 8 = 10
  sorry

end brother_books_total_l706_706864


namespace sum_odd_digits_from_1_to_200_l706_706706

/-- Function to compute the sum of odd digits of a number -/
def odd_digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (fun d => d % 2 = 1) |>.sum

/-- Statement of the problem to prove the sum of the odd digits of numbers from 1 to 200 is 1000 -/
theorem sum_odd_digits_from_1_to_200 : (Finset.range 200).sum odd_digit_sum = 1000 := 
  sorry

end sum_odd_digits_from_1_to_200_l706_706706


namespace pizza_problem_l706_706045

def number_of_pizzas (n : ℕ) : ℕ :=
  (finset.card (finset.powerset_len 1 (finset.range n))) +
  (finset.card (finset.powerset_len 2 (finset.range n))) +
  (finset.card (finset.powerset_len 3 (finset.range n)))

theorem pizza_problem : number_of_pizzas 8 = 92 :=
by
  -- The proof details would come here
  sorry

end pizza_problem_l706_706045


namespace three_digit_numbers_divisible_by_5_l706_706568

-- We need to represent the condition of selection of three digits from the set {0,1,2,3,4,5,6},
-- forming a 3-digit number divisible by 5, ensuring no digit repetition

theorem three_digit_numbers_divisible_by_5 :
  let digits := {0, 1, 2, 3, 4, 5, 6}
  ∃ count, count = 55 ∧ 
    count = (∑ d1 in digits, ∑ d2 in (digits \ {d1}), ∑ d3 in (digits \ {d1, d2}),
              if (d3 = 0 ∨ d3 = 5) ∧ d1 ≠ 0 then 1 else 0) :=
by {
  sorry
}

end three_digit_numbers_divisible_by_5_l706_706568


namespace expected_value_of_die_roll_l706_706441

def expectedValueOfWin : ℚ := (1 + 4 + 9 + 16 + 25 + 36) * (1 / 6)

theorem expected_value_of_die_roll
    (p₁ p₂ p₃ p₄ p₅ p₆ : ℚ)
    (h₁ : p₁ = 1 / 6)
    (h₂ : p₂ = 1 / 6)
    (h₃ : p₃ = 1 / 6)
    (h₄ : p₄ = 1 / 6)
    (h₅ : p₅ = 1 / 6)
    (h₆ : p₆ = 1 / 6) :
    Real.round ((p₁ * 1^2 + p₂ * 2^2 + p₃ * 3^2 + p₄ * 4^2 + p₅ * 5^2 + p₆ * 6^2) : ℚ) = 15.17 :=
by
    sorry -- proof is skipped

end expected_value_of_die_roll_l706_706441


namespace find_circle_radius_l706_706122

def circle_radius (r : ℝ) : Prop :=
  let π := Real.pi in
  let C := 2 * π * r in
  let A := π * r^2 in
  C + A = 530.929158456675

theorem find_circle_radius (r : ℝ) (h : circle_radius r) : r = Real.sqrt 170 :=
sorry

end find_circle_radius_l706_706122


namespace shadow_length_when_eight_meters_away_l706_706730

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l706_706730


namespace original_balance_l706_706101

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

theorem original_balance (decrease_percentage : ℝ) (current_balance : ℝ) (original_balance : ℝ) :
  decrease_percentage = 0.10 → current_balance = 90000 → 
  current_balance = (1 - decrease_percentage) * original_balance → 
  original_balance = 100000 := by
  sorry

end original_balance_l706_706101


namespace unique_function_satisfying_inequality_l706_706901

theorem unique_function_satisfying_inequality:
  ∃! (f : ℝ → ℝ), ∀ (x y z : ℝ), f(x * y) + f(x * z) + f(y * z) - f(x) * f(y) * f(z) ≥ 1 := sorry

end unique_function_satisfying_inequality_l706_706901


namespace concurrency_iff_concurrency_l706_706081

theorem concurrency_iff_concurrency 
  {A B C D E F L M N: Type} 
  [triangle A B C] (hD: on_side D B C) (hE: on_side E C A) (hF: on_side F A B)
  (hL: midpoint L E F) (hM: midpoint M F D) (hN: midpoint N D E)
  : (concurrent (line_through A D) (line_through B E) (line_through C F)) ↔
    (concurrent (line_through A L) (line_through B M) (line_through C N)) :=
sorry

end concurrency_iff_concurrency_l706_706081


namespace base_b_for_256_l706_706421

theorem base_b_for_256 (b : ℕ) : b^3 ≤ 256 ∧ 256 < b^4 ↔ b = 5 :=
by
  split
  { intro h
    cases h with h1 h2
    have : 5 ≤ b :=
      Nat.le_of_lt_succ ((Nat.lt_of_le_of_lt h1) (Nat.lt_of_lt_of_le (by norm_num) h2)) sorry
  sorry
  
  sorry

end base_b_for_256_l706_706421


namespace intersection_area_l706_706439

-- Defining the cube's edge length.
def edge_length : ℝ := 12

-- Defining vertices of the cube.
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 12, 12)
def E : ℝ × ℝ × ℝ := (0, 12, 3)
def D : ℝ × ℝ × ℝ := (12, 12, 0)
def F : ℝ × ℝ × ℝ := (12, 3, 0)

-- Definitions of distances where BE and DF are given as 9.
axiom BE_length : dist B E = 9
axiom DF_length : dist D F = 9

-- Prove that the area of the intersection of the plane with the cube is 28*sqrt(34).
theorem intersection_area : 
  ∃ (α : ℝ × ℝ × ℝ → ℝ), 
    (∀ (P : ℝ × ℝ × ℝ), α A = 0 ∧ α E = 0 ∧ α F ≠ 0)
    ∧ (α (B) = α (D))
    ∧ (area_of_section = 28 * real.sqrt 34) := 
by
  sorry

end intersection_area_l706_706439


namespace proper_subsets_count_l706_706963

open Finset

theorem proper_subsets_count (B : Finset ℕ) (h : B = {2, 3, 4}) : B.card = 3 → (2 ^ B.card - 1) = 7 :=
by
  intro hB
  rw [h] at hB
  simp at hB
  sorry

end proper_subsets_count_l706_706963


namespace min_quadratic_expr_l706_706388

noncomputable def quadratic_expr (x : ℝ) := x^2 + 10 * x + 3

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = -22 :=
by
  use -5
  simp [quadratic_expr]
  sorry

end min_quadratic_expr_l706_706388


namespace angle_between_a_b_is_pi_div_3_l706_706616

noncomputable def vector_a : ℝ × ℝ := (sqrt 3 / 2, 1 / 2)
noncomputable def vector_b : ℝ × ℝ := (sqrt 3, -1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def angle_between (v w : ℝ × ℝ) : ℝ :=
  real.arccos ((dot_product v w) / (magnitude v * magnitude w))

theorem angle_between_a_b_is_pi_div_3 : angle_between vector_a vector_b = real.pi / 3 :=
by
  sorry

end angle_between_a_b_is_pi_div_3_l706_706616


namespace find_a_l706_706960

def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + a * y + 6 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a - 2) * x + 3 * a * y + 18 = 0
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := 
  ∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → (x1 ≠ 0 → x2 ≠ 0 → x1 / x2 = a / (a-2)) ∧ x1 = 0 → y1 / y2 = a / (3 * a)

theorem find_a (a : ℝ) :
  parallel (line1 a) (line2 a) → a = 0 := 
sorry

end find_a_l706_706960


namespace three_mutual_friends_l706_706789

-- Define the properties based on the conditions
def graph_conditions (G : SimpleGraph (Fin 11)) : Prop :=
∀ v : Fin 11, G.degree v ≥ 6

-- Define the theorem we need to prove
theorem three_mutual_friends :
  ∃ (G : SimpleGraph (Fin 11)) (h : graph_conditions G), ∃ (a b c : Fin 11), G.adj a b ∧ G.adj b c ∧ G.adj c a :=
begin
  sorry
end

end three_mutual_friends_l706_706789


namespace business_tax_paid_january_l706_706036

def revenue : ℝ := 10_000_000
def tax_rate : ℝ := 0.05
def tax_paid : ℝ := (revenue * tax_rate) / 10_000

theorem business_tax_paid_january :
  tax_paid = 500 := by
  sorry

end business_tax_paid_january_l706_706036


namespace algae_growth_at_810am_l706_706003

/-- 
Given that the initial number of algae in a tank is 50 at 8:00 AM and the colony grows threefold 
every 2 minutes, the following Lean 4 code proves that the number of algae in the tank at 8:10 AM 
is 12150, assuming no algae perish.
-/
theorem algae_growth_at_810am :
  ∀ (initial_algae : ℕ) (growth_rate : ℕ) (initial_time final_time growth_interval : ℕ),
  initial_algae = 50 →
  growth_rate = 3 →
  initial_time = 8 * 60 →
  final_time = (8 * 60 + 10) →
  growth_interval = 2 →
  initial_algae * growth_rate ^ ((final_time - initial_time) / growth_interval) = 12150 :=
by
  intros initial_algae growth_rate initial_time final_time growth_interval
  assume h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end algae_growth_at_810am_l706_706003


namespace number_of_lucky_tickets_even_sum_of_lucky_tickets_divisible_by_999_l706_706820

-- Definition of a lucky ticket
def isLucky (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  d1 + d2 + d3 = d4 + d5 + d6

theorem number_of_lucky_tickets_even : 
  {n : ℕ | n < 1000000 ∧ isLucky n}.finite.toFinset.card % 2 = 0 := 
sorry

theorem sum_of_lucky_tickets_divisible_by_999 : 
  ∑ n in {n : ℕ | n < 1000000 ∧ isLucky n}.finite.toFinset, n % 999 = 0 := 
sorry

end number_of_lucky_tickets_even_sum_of_lucky_tickets_divisible_by_999_l706_706820


namespace number_of_primary_schools_l706_706998

theorem number_of_primary_schools (A B total : ℕ) (h1 : A = 2 * 400)
  (h2 : B = 2 * 340) (h3 : total = 1480) (h4 : total = A + B) :
  2 + 2 = 4 :=
by
  sorry

end number_of_primary_schools_l706_706998


namespace part_a_part_b_l706_706184

-- Conditions
variables (A B C A₁ B₁ C₁ N : Point)
variable AA₁ : Line A A₁
variable BB₁ : Line B B₁
variable CC₁ : Line C C₁
variable (ABC_eq : Plane ABC)
variable (A₁B₁C₁_eq : Plane A₁ B₁ C₁)
variable (parallels: Parallel ABC_eq A₁B₁C₁_eq)
variable (equilateral_BB₁C : EquilateralTriangle B B₁ C)
variable (on_edge : OnLine N AA₁)
variable (ratio_AN_NA₁ : Ratio AN NA₁ 1 2)
variable (sphere_omega : Sphere N sqrt(5))

-- Statement
theorem part_a:
  length(BB₁) = sqrt(15) :=
sorry

theorem part_b:
  angle(AA₁, Plane BB₁C) = π / 4 ∧ length(A₁B₁) = 2 - sqrt(3 / 2) :=
sorry

end part_a_part_b_l706_706184


namespace moores_law_transistors_2010_l706_706715

theorem moores_law_transistors_2010 :
  let initial_transistors := 2500000
  let doubling_period_years := 1.5
  let duration_years := 2010 - 2000
  let periods := (duration_years / doubling_period_years).to_int
  let final_transistors := initial_transistors * (2 ^ periods)
  final_transistors = 160000000 := by
  sorry

end moores_law_transistors_2010_l706_706715


namespace range_of_a_l706_706166

open Real

def proposition_p (a : ℝ) : Prop := ∃ x ∈ Icc (-1 : ℝ) (1 : ℝ), a ^ 2 * x ^ 2 + a * x - 2 = 0

def proposition_q (a : ℝ) : Prop := ∃ x : ℝ, (x ^ 2 + 2 * a * x + 2 * a ≤ 0) ∧ ∀ y : ℝ, y ^ 2 + 2 * a * y + 2 * a = 0 → y = x

theorem range_of_a (a : ℝ) :
  ¬(proposition_p a ∨ proposition_q a) → (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l706_706166


namespace Grant_spending_is_200_l706_706203

def Juanita_daily_spending (day: String) : Float :=
  if day = "Sunday" then 2.0 else 0.5

def Juanita_weekly_spending : Float :=
  6 * Juanita_daily_spending "weekday" + Juanita_daily_spending "Sunday"

def Juanita_yearly_spending : Float :=
  52 * Juanita_weekly_spending

def Grant_yearly_spending := Juanita_yearly_spending - 60

theorem Grant_spending_is_200 : Grant_yearly_spending = 200 := by
  sorry

end Grant_spending_is_200_l706_706203


namespace area_of_rhombus_l706_706411

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 65) (h2 : d2 = 60) : 
  (d1 * d2) / 2 = 1950 :=
by
  rw [h1, h2]
  norm_num
  sorry

end area_of_rhombus_l706_706411


namespace problem_l706_706565

noncomputable def g (a : ℝ) : Polynomial ℝ :=
  Polynomial.monomial 3 1 + Polynomial.monomial 2 a + Polynomial.monomial 1 2 + Polynomial.C 15

noncomputable def f (a b c : ℝ) : Polynomial ℝ :=
  Polynomial.monomial 4 1 + Polynomial.monomial 3 1 + Polynomial.monomial 2 b + Polynomial.monomial 1 120 + Polynomial.C c

theorem problem 
  (a b c : ℝ)
  (h1 : (g a).roots.length = 3)
  (h2 : ∀ x, x ∈ (g a).roots → x ∈ (f a b c).roots) :
  (f a b c).eval 1 = -3682.25 :=
sorry

end problem_l706_706565


namespace avg_GPA_is_93_l706_706753

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l706_706753


namespace apple_distribution_l706_706815

theorem apple_distribution (x : ℕ) (h₁ : 1430 % x = 0) (h₂ : 1430 % (x + 45) = 0) (h₃ : 1430 / x - 1430 / (x + 45) = 9) : 
  1430 / x = 22 :=
by
  sorry

end apple_distribution_l706_706815


namespace knights_possibilities_l706_706648

noncomputable def number_of_knights (n : ℕ) (statements : fin n → ℕ × ℕ) : set ℕ :=
  {k | ∃ l : ℕ, l = n - k ∧ ∀ i : fin n, (statements i = (k, l) → ∃ j ≠ i, statements j = (k, l)) ∧ ∀ m ≠ k, 2 * (count (λ p : ℕ × ℕ, p.1 = m ∨ p.2 = m) ((list.finRange n).map statements)) = n}

theorem knights_possibilities : number_of_knights 10
  ((λ i, match i with
    | fin.of_nat m => if m < 5 then (m, 10 - m) else (10 - m, m)
     end) : fin 10 → ℕ × ℕ) = {0, 1, 2} :=
by
  sorry

end knights_possibilities_l706_706648


namespace maximum_absolute_sum_l706_706980

theorem maximum_absolute_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : |x| + |y| + |z| ≤ 2 :=
sorry

end maximum_absolute_sum_l706_706980


namespace expansion_contains_constant_term_l706_706951

theorem expansion_contains_constant_term (n : ℕ) (h_pos : 0 < n) 
  (h_const_term : ∃ (r : ℕ), (choose n r) * (-2)^r * x^((n - 3 * r) / 2) = 1) :
  n = 6 :=
sorry

end expansion_contains_constant_term_l706_706951


namespace tangent_line_eq_range_of_a_local_max_g_l706_706598

section
-- Given functions
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - 2 * a * x
def g (x : ℝ) (a : ℝ) : ℝ := f x a + 0.5 * x ^ 2

-- Problem 1
theorem tangent_line_eq (a : ℝ) (h : a = 2) : (∀ x, f x a = Real.log x - 2 * 2 * x) ∧ (f 1 a = -4) ∧ (derivative (λ x, f x a) 1 = -3) → (∀ x y, y - (-4) = -3 * (x - 1) ↔ 3*x + y + 1 = 0) :=
sorry

-- Problem 2
theorem range_of_a : (∀ x, f x a ≤ 2) → (a ∈ Set.Ici (1 / (2 * Real.exp 3))) :=
sorry

-- Problem 3
theorem local_max_g (a : ℝ) (x0 : ℝ) (h : ∀ x, differentiable ℝ (g x a)) : (∃ x0, local_maximum x0 (g x0 a)) → (x0 * f x0 + 1 + a * x0 ^ 2 > 0) :=
sorry

end

end tangent_line_eq_range_of_a_local_max_g_l706_706598


namespace find_BD_l706_706671

-- Definitions of the given conditions for the problem
def isIsoscelesTriangle (A B C : Type) (AB AC : ℝ) : Prop :=
  AB = AC

def isAltitudeFrom (C : Type) (D A B : Type) : Prop :=
  ∀ E : Type, E ≠ D → ∠ AED = 90

-- The given lengths
def AB_length : ℝ := 10
def CD_length : ℝ := 6

-- Define the main statement to be proven in Lean
theorem find_BD (A B C D : Type) 
  (hABC_isosceles : isIsoscelesTriangle A B C AB_length AB_length)
  (hCD_altitude : isAltitudeFrom C D A B)
  (hAB_length : AB_length = 10)
  (hCD_length : CD_length = 6) : 
  ∃ (BD : ℝ), BD = 5 := 
sorry

end find_BD_l706_706671


namespace smallest_sum_of_consecutive_integers_is_square_l706_706333

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l706_706333


namespace mitch_family_milk_l706_706990

variable (total_milk soy_milk regular_milk : ℚ)

-- Conditions
axiom cond1 : total_milk = 0.6
axiom cond2 : soy_milk = 0.1
axiom cond3 : regular_milk + soy_milk = total_milk

-- Theorem statement
theorem mitch_family_milk : regular_milk = 0.5 :=
by
  sorry

end mitch_family_milk_l706_706990


namespace alpha_is_pi_over_7_l706_706991

/-- Theorem: In a circle, let triangle ABC be inscribed with \(\angle ABC = \angle ACB\). Let AD and CD be tangents from a point D outside the circle. If \(\angle ABC = 3 \angle D\), where \(\angle ADA\) is an external angle at D formed as AD intersects the circle at another point \(A_D\), and \(\angle BAC = \alpha\), then \(\alpha = \frac{\pi}{7}\). -/
theorem alpha_is_pi_over_7
  (A B C D : Type)
  [has_angle A B C]
  [has_angle B C A]
  [has_angle A D D]
  [inscribed_triangle A B C]
  (angle_ABC_eq_angle_ACB : ∀ A B C, angle A B C = angle B C A)
  (angle_ABC_eq_3angle_D : ∀ A B C D, angle A B C = 3 * angle A D D)
  (angle_BAC_eq_alpha : ∀ A B C α, angle A B C = α) :
  ∀ α : real, α = real.pi / 7 :=
begin
  sorry
end

end alpha_is_pi_over_7_l706_706991


namespace proof_problem_l706_706024

noncomputable def X (Y : ℕ) : ℤ := 6 * Y - 8

def P_X_lt_0 : ℝ := 13 / 256

def distribution_table : List (ℤ × ℝ) := [
  (-8, 1 / 256),
  (-2, 3 / 64),
  (4, 27 / 128),
  (10, 27 / 64),
  (16, 81 / 256)
]

def E_X : ℤ := 10

theorem proof_problem (Y : ℕ) (P_Y : Y ≤ 4) 
  (prob_Y : list ℝ) (hx_dist : list (ℤ × ℝ)) :
  (X Y = 6 * Y - 8) ∧
  (prob_Y = [1/256, 12/256, 54/256, 108/256, 81/256]) ∧
  (P_X_lt_0 = 13 / 256) ∧ 
  (hx_dist = distribution_table) ∧ 
  (E_X = 10) := sorry

end proof_problem_l706_706024


namespace ratio_initial_to_doubled_l706_706841

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 105) : x / (2 * x) = 1 / 2 :=
by
  sorry

end ratio_initial_to_doubled_l706_706841


namespace fraction_of_blueberry_tart_l706_706461

/-- Let total leftover tarts be 0.91.
    Let the tart filled with cherries be 0.08.
    Let the tart filled with peaches be 0.08.
    Prove that the fraction of the tart filled with blueberries is 0.75. --/
theorem fraction_of_blueberry_tart (H_total : Real) (H_cherry : Real) (H_peach : Real)
  (H1 : H_total = 0.91) (H2 : H_cherry = 0.08) (H3 : H_peach = 0.08) :
  (H_total - (H_cherry + H_peach)) = 0.75 :=
sorry

end fraction_of_blueberry_tart_l706_706461


namespace cos_A_value_l706_706642

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
-- a, b, c are the sides opposite to angles A, B, and C respectively.
-- Assumption 1: b - c = (1/4) * a
def condition1 := b - c = (1/4) * a
-- Assumption 2: 2 * sin B = 3 * sin C
def condition2 := 2 * Real.sin B = 3 * Real.sin C

-- The theorem statement: Under these conditions, prove that cos A = -1/4.
theorem cos_A_value (h1 : condition1 a b c) (h2 : condition2 B C) : 
    Real.cos A = -1/4 :=
sorry -- placeholder for the proof

end cos_A_value_l706_706642


namespace intersection_point_unique_l706_706697

def g (x : ℝ) : ℝ := x^3 - 9 * x^2 + 27 * x - 29

theorem intersection_point_unique :
  ∀ x : ℝ, g(x) = x → x = 1 :=
begin
  intros x hx,
  have : x^3 - 9 * x^2 + 26 * x - 29 = 0,
  { rw [g, hx], ring },
  sorry
end

end intersection_point_unique_l706_706697


namespace pet_store_feet_count_l706_706044

theorem pet_store_feet_count (total_heads dogs : ℕ) (dog_feet parakeet_feet : ℕ) :
  total_heads = 15 → dogs = 9 → dog_feet = 4 → parakeet_feet = 2 →
  total_heads * dog_feet + (total_heads - dogs) * parakeet_feet = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have total_parakeets := total_heads - dogs
  have feet_dogs := dogs * dog_feet
  have feet_parakeets := total_parakeets * parakeet_feet
  rw [total_heads, total_parakeets, feet_dogs, feet_parakeets]
  sorry

end pet_store_feet_count_l706_706044


namespace limit_f_div_r2_limit_g_div_rh_l706_706272

noncomputable def f (r : ℝ) : ℕ := sorry

def g (r : ℝ) : ℝ := (f r) - π * r^2

theorem limit_f_div_r2 : 
  tendsto (fun r => (f r : ℝ)/r^2) atTop (𝓝 π) :=
sorry

theorem limit_g_div_rh (h : ℝ) (h_lt : h < 2) : 
  tendsto (fun r => g r / r^h) atTop (𝓝 0) :=
sorry

end limit_f_div_r2_limit_g_div_rh_l706_706272


namespace proposition_one_true_proposition_two_false_correct_choice_l706_706183

theorem proposition_one_true (a b c : ℂ) (h : a^2 + b^2 > c^2) : a^2 + b^2 - c^2 > 0 := 
by
  sorry

theorem proposition_two_false (a b c : ℂ) (h : a^2 + b^2 - c^2 > 0) : ¬(a^2 + b^2 > c^2) := 
by
  sorry

theorem correct_choice : 
  (∀ (a b c : ℂ), (a^2 + b^2 > c^2) → (a^2 + b^2 - c^2 > 0)) ∧
  (∃ (a b c : ℂ), (a^2 + b^2 - c^2 > 0) ∧ ¬(a^2 + b^2 > c^2)) :=
by
  constructor
  { intros a b c h
    exact proposition_one_true a b c h }
  { use [2+i, i, sqrt 2 + sqrt 2 * I]
    have h₁ : (2 + i) ^ 2 + i ^ 2 - (sqrt 2 + sqrt 2 * I) ^ 2 > 0 := by
      sorry
    have h₂ : ¬((2 + i) ^ 2 + i ^ 2 > (sqrt 2 + sqrt 2 * I) ^ 2) := by
      sorry
    use h₁, h₂ }

end proposition_one_true_proposition_two_false_correct_choice_l706_706183


namespace smallest_d_l706_706281

noncomputable def d := 53361

theorem smallest_d :
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧
    10000 * d = (p * q * r) ^ 2 ∧ d = 53361 :=
  by
    sorry

end smallest_d_l706_706281


namespace same_type_l706_706803

variable (X Y : Prop) 

-- Definition of witnesses A and B based on their statements
def witness_A (A : Prop) := A ↔ (X → Y)
def witness_B (B : Prop) := B ↔ (¬X ∨ Y)

-- Proposition stating that A and B must be of the same type
theorem same_type (A B : Prop) (HA : witness_A X Y A) (HB : witness_B X Y B) : 
  (A = B) := 
sorry

end same_type_l706_706803


namespace a_1000_value_l706_706670

noncomputable def a : ℕ → ℤ
| 0       := 0
| 1       := 1
| 2       := 5
| (n + 3) := a (n + 2) - a (n + 1)

lemma a_periodic : ∀ n, a (n + 6) = a n :=
by sorry

theorem a_1000_value : a 1000 = -1 :=
by sorry

end a_1000_value_l706_706670


namespace M_subset_N_l706_706199

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l706_706199


namespace johns_profit_l706_706683

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l706_706683


namespace interest_rate_C_l706_706037

theorem interest_rate_C (P A G : ℝ) (R : ℝ) (t : ℝ := 3) (rate_A : ℝ := 0.10) :
  P = 4000 ∧ rate_A = 0.10 ∧ G = 180 →
  (P * rate_A * t + G) = P * (R / 100) * t →
  R = 11.5 :=
by
  intros h_cond h_eq
  -- proof to be filled, use the given conditions and equations
  sorry

end interest_rate_C_l706_706037


namespace smallest_sum_of_consecutive_integers_is_square_l706_706336

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l706_706336


namespace is_isosceles_triangle_l706_706200

theorem is_isosceles_triangle 
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a * Real.cos B + b * Real.cos C + c * Real.cos A = b * Real.cos A + c * Real.cos B + a * Real.cos C) : 
  (A = B ∨ B = C ∨ A = C) :=
sorry

end is_isosceles_triangle_l706_706200


namespace vova_figure_boundary_segment_l706_706381

theorem vova_figure_boundary_segment (figure : set (ℤ × ℤ)) (cuts_total_length : ℝ) :
  (∀ (cell : ℤ × ℤ), cell ∈ figure) ∧ non_empty (figure) ∧ (cuts_total_length = 2017) →
  ∃ (segment_length : ℝ), segment_length ≥ 2 :=
by sorry

end vova_figure_boundary_segment_l706_706381


namespace option_B_is_correct_l706_706888

-- Define the functions for each option
def fA (x : ℝ) := if x ≠ 0 then x^3 / x else 0
def gA (x : ℝ) := x^2

def fB (x : ℝ) := 1 -- x^0 is always 1 for x ≠ 0
def gB (x : ℝ) := 1

def fC (x : ℝ) := real.sqrt (x^2)
def gC (x : ℝ) := x

def fD (x : ℝ) := abs x
def gD (x : ℝ) := (real.sqrt x)^2

-- Determine if two functions are the same by checking expressions and domains
theorem option_B_is_correct : 
  (∀ x : ℝ, x ≠ 0 → fB x = gB x ∧ ∃ d : set ℝ, 
    (x ∈ d ↔ x ≠ 0) ∧ (x ∈ d ↔ x ≠ 0)) :=
by 
  intros x hx 
  split 
  { 
    refl 
  }
  exact ⟨λ x, x ≠ 0, λ x, ⟨λ hx, hx, λ hx, hx⟩⟩

end option_B_is_correct_l706_706888


namespace base_b_for_256_l706_706420

theorem base_b_for_256 (b : ℕ) : b^3 ≤ 256 ∧ 256 < b^4 ↔ b = 5 :=
by
  split
  { intro h
    cases h with h1 h2
    have : 5 ≤ b :=
      Nat.le_of_lt_succ ((Nat.lt_of_le_of_lt h1) (Nat.lt_of_lt_of_le (by norm_num) h2)) sorry
  sorry
  
  sorry

end base_b_for_256_l706_706420


namespace sum_of_greatest_and_median_of_consecutive_multiples_of_seven_l706_706409

theorem sum_of_greatest_and_median_of_consecutive_multiples_of_seven 
  (s : Set ℤ)
  (h1 : ∃ x, s = {y | ∃ k, k ∈ Finset.range 81 ∧ y = x + 7 * k})
  (h2 : ∃ x, x + (x + 7) = 145) : 
  s.sum + s.median = 978 := 
sorry

end sum_of_greatest_and_median_of_consecutive_multiples_of_seven_l706_706409


namespace hyperbola_eccentricity_l706_706961

-- Definitions of the conditions given in the problem
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def hyperbola (x y : ℝ) (m : ℝ) : Prop := x^2 / m - y^2 = 1
def axis_of_symmetry (x : ℝ) : Prop := x = -2
def right_triangle (A B F : (ℝ × ℝ)) : Prop := ∥A - F∥ + ∥B - F∥ = ∥A - B∥

-- Proving the eccentricity of the given hyperbola
theorem hyperbola_eccentricity 
  (m : ℝ) 
  (h1 : ∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ axis_of_symmetry A.1 ∧ axis_of_symmetry B.1 ∧ hyperbola A.1 A.2 m ∧ hyperbola B.1 B.2 m) 
  (h2 : ∃ F : ℝ × ℝ, F.1 = 2 ∧ F.2 = 0)
  (h3 : ∃ A B F : ℝ × ℝ, right_triangle A B F) : 
  m = 4 / 17 → ∃ e : ℝ, e = (Real.sqrt 21) / 2 :=
sorry

end hyperbola_eccentricity_l706_706961


namespace rodney_guess_probability_l706_706301

-- Definitions and conditions
def is_two_digit_integer (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def tens_digit_is_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def units_digit_is_odd (n : ℕ) : Prop := (n % 10) % 2 = 1
def within_range (n : ℕ) : Prop := n > 50 ∧ n < 90

-- The main theorem
theorem rodney_guess_probability : 
  ∃ (n : ℕ), is_two_digit_integer n ∧ tens_digit_is_even n ∧ units_digit_is_odd n ∧ within_range n → ℝ :=
begin
  intro n,
  intro h,
  existsi (1/10 : ℝ),
  sorry
end

end rodney_guess_probability_l706_706301


namespace triangle_A_Cos_fractions_triangle_AD_length_l706_706236

open Real

-- Definition of an acute triangle
def is_acute_triangle (A B C: ℝ) := 0 < A ∧ A < π / 2 ∧
                                  0 < B ∧ B < π / 2 ∧
                                  0 < C ∧ C < π / 2 ∧
                                  A + B + C = π

-- Definition of the problem in Lean
theorem triangle_A_Cos_fractions (A B C a b c : ℝ) 
    (h1 : is_acute_triangle A B C)
    (h2 : a > 0)
    (h3 : b > 0)
    (h4 : c > 0)
    (h5 : cos A / cos C = a / (2 * b - c)) 
    : A = π / 3 := 
by sorry

-- Additional condition for Part 2
theorem triangle_AD_length (A B C a b c : ℝ) (D : Point) 
    (h1 : is_acute_triangle A B C)
    (h2 : a = sqrt 7)
    (h3 : c = 3)
    (h4 : b = 2)
    (h5 : midpoint (B, C) = D)
    : length (A, D) = sqrt 19 / 2 :=
by sorry

end triangle_A_Cos_fractions_triangle_AD_length_l706_706236


namespace quadratic_must_have_m_eq_neg2_l706_706175

theorem quadratic_must_have_m_eq_neg2 (m : ℝ) (h : (m - 2) * x^|m| - 3 * x - 4 = 0) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by
  sorry

end quadratic_must_have_m_eq_neg2_l706_706175


namespace element_mass_percentage_none_l706_706126

theorem element_mass_percentage_none (H Br O : ℝ) (mass_HBrO3 mass_percentage : ℝ) : 
  H = 1.01 → 
  Br = 79.90 → 
  O = 16.00 → 
  mass_HBrO3 = H + Br + 3 * O → 
  mass_percentage = 37.21 →
  ∀ x, x ∈ {H, Br, 3 * O} → (x / mass_HBrO3) * 100 ≠ mass_percentage := 
by
  intros H Br O mass_HBrO3 mass_percentage hH hBr hO hmass_HBrO3 hmass_percentage x hx
  sorry

end element_mass_percentage_none_l706_706126


namespace convert_decimal_to_fraction_l706_706397

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l706_706397


namespace avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l706_706540

/-

Problem 1:
Given:
- Visitors in 2022: 200000
- Visitors in 2024: 288000

Prove:
- The average annual growth rate of visitors from 2022 to 2024 is 20% 

Problem 2:
Given:
- Cost price per cup: 6 yuan
- Selling price per cup at 25 yuan leads to 300 cups sold per day.
- Each 1 yuan reduction leads to 30 more cups sold per day.
- Desired daily profit in 2024: 6300 yuan

Prove:
- The selling price per cup for maximum discount and desired profit is 20 yuan.

-/

-- Definitions for Problem 1
def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000

-- Definition for annual growth rate
def annual_growth_rate (P Q : ℕ) (y : ℕ) : ℝ :=
  ((Q.to_real / P.to_real) ^ (1 / y.to_real)) - 1

def expected_growth_rate := annual_growth_rate visitors_2022 visitors_2024 2

-- Statement for the first proof
theorem avg_annual_growth_rate_is_20_percent : expected_growth_rate = 0.2 := sorry

-- Definitions for Problem 2
def cost_price_per_cup : ℕ := 6
def initial_price_per_cup : ℕ := 25
def initial_sales_per_day : ℕ := 300
def additional_sales_per_price_reduction : ℕ := 30
def desired_daily_profit : ℕ := 6300

-- Profit function
def daily_profit (price : ℕ) : ℕ := (price - cost_price_per_cup) * (initial_sales_per_day + additional_sales_per_price_reduction * (initial_price_per_cup - price))

-- Statement for the second proof
theorem optimal_selling_price_for_max_discount : (∃ (price : ℕ), daily_profit price = desired_daily_profit ∧ price = 20) := sorry

end avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l706_706540


namespace sum_arithmetic_sequence_has_max_value_l706_706985

noncomputable section
open Classical

-- Defining an arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- The main statement to prove: Sn has a maximum value given conditions a1 > 0 and d < 0
theorem sum_arithmetic_sequence_has_max_value (a1 d : ℝ) (h1 : a1 > 0) (h2 : d < 0) :
  ∃ M, ∀ n, sum_arithmetic_sequence a1 d n ≤ M :=
by
  sorry

end sum_arithmetic_sequence_has_max_value_l706_706985


namespace red_green_difference_and_blue_purple_ratio_l706_706368

-- Define the baskets
structure Basket :=
(red : Nat) (yellow : Nat) (blue : Nat) (purple : Nat) (green : Nat) (white : Nat) (black : Nat) (orange : Nat)

def A := Basket.mk 7 8 5 6 0 0 0 0
def B := Basket.mk 5 4 0 0 10 0 0 2
def C := Basket.mk 0 9 0 0 2 3 4 0
def D := Basket.mk 8 6 3 5 0 0 0 1
def E := Basket.mk 3 7 0 0 6 0 5 0

def difference_red_green (b1 b2 b3 : Basket) : Nat :=
  (b1.green + b2.green + b3.green) - (b1.red + b2.red + b3.red)

def ratio_blue_purple (b : Basket) : Real :=
  b.blue.toReal / b.purple.toReal

theorem red_green_difference_and_blue_purple_ratio :
  difference_red_green B C E = 10 ∧ ratio_blue_purple A = 5.0 / 6.0 := by
  -- here goes the proof
  sorry

end red_green_difference_and_blue_purple_ratio_l706_706368


namespace f_even_function_f_smallest_positive_period_f_not_increasing_on_interval_f_maximum_value_l706_706190

noncomputable def f (x : ℝ) : ℝ :=
  |Real.sin x| + |Real.cos x| + Real.sin (2 * x) ^ 4

theorem f_even_function : ∀ x : ℝ, f (-x) = f x :=
by 
  -- proof to be filled in

theorem f_smallest_positive_period : ∀ x : ℝ, f (x + π/2) = f x :=
by 
  -- proof to be filled in

theorem f_not_increasing_on_interval : ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/3 → f x < f y) :=
by 
  -- proof to be filled in

theorem f_maximum_value : ∃ x : ℝ, 0 ≤ x ∧ x ≤ π/2 ∧ f x = 1 + Real.sqrt 2 :=
by 
  -- proof to be filled in

end f_even_function_f_smallest_positive_period_f_not_increasing_on_interval_f_maximum_value_l706_706190


namespace root_polynomial_sum_l706_706695

theorem root_polynomial_sum {b c : ℝ} (hb : b^2 - b - 1 = 0) (hc : c^2 - c - 1 = 0) : 
  (1 / (1 - b)) + (1 / (1 - c)) = -1 := 
sorry

end root_polynomial_sum_l706_706695


namespace problem_I_problem_II_l706_706576

-- Definitions of the conditions
def a (n : ℕ) : ℝ := if n = 0 then 0 else (1 / 2 ^ (n : ℝ))

/- Problem (Ⅰ): Prove the general formula for the sequence {a_n} is a_n = 1 / 2^n for all n in ℕ* -/
theorem problem_I (n : ℕ) (h : n > 0) :
  a n = 1 / 2 ^ n :=
by
  sorry

/- Problem (Ⅱ): Prove the sum of 1 / (b₁b₂) + 1 / (b₂b₃) + ... + 1 / (bₙbₙ₊₁) is n / (n + 1) -/
theorem problem_II (n : ℕ) (h : n > 0) :
  let b (k : ℕ) := log 2 (a k) in
  (∑ i in Finset.range n, 1 / (b i * b (i + 1))) = n / (n + 1) :=
by
  sorry

end problem_I_problem_II_l706_706576


namespace club_has_ten_members_l706_706649

theorem club_has_ten_members
    (members : ℕ)
    (committees : Finset (Finset ℕ))
    (h_committees_size : committees.card = 5)
    (member_committee_assignments : members → Finset ℕ)
    (h_member_committee : ∀ m, (member_committee_assignments m).card = 2)
    (h_pairwise_unique_member : ∀ (c1 c2 : Finset ℕ), c1 ≠ c2 → c1 ∈ committees → c2 ∈ committees →
      (∀ m1 m2, m1 ≠ m2 → (m1 ∈ c1 ∧ m1 ∈ c2) → (m2 ∈ c1 ∧ m2 ∈ c2) → False))
    (h_committee_members_constraint : ∀ c ∈ committees, c.card <= 3) :
    members = 10 := by
  sorry

end club_has_ten_members_l706_706649


namespace part1_part2_l706_706955

def f (x : ℝ) (a : ℝ) : ℝ :=
  |x + 2 * a| + |2 * x - (1 / a)|

theorem part1 (x : ℝ) :
  f x 1 ≤ 6 ↔ -7/3 ≤ x ∧ x ≤ 5/3 :=
by
  sorry

theorem part2 (x a : ℝ) (h : a ≠ 0) :
  f x a ≥ 2 :=
by
  sorry

end part1_part2_l706_706955


namespace equivalent_propositions_l706_706395

variable {U : Type} {M : set U} {a b : U}

theorem equivalent_propositions (h : a ∈ M → b ∉ M) : b ∈ M → a ∉ M :=
by {
  intro hb_in_M,
  by_contradiction ha_in_M,
  exact h ha_in_M hb_in_M,
}

end equivalent_propositions_l706_706395


namespace sequence_sum_difference_l706_706725

theorem sequence_sum_difference (n : ℕ) : 
  let A_n := (2 * n - 1) * (n^2 - n + 1),
      B_n := n^3 - (n - 1)^3
  in A_n + B_n = 2 * n^3 :=
by
  sorry

end sequence_sum_difference_l706_706725


namespace complex_num_in_third_quadrant_l706_706583

-- Define the complex number and imaginary unit i
def complex_num := (1 - complex.I) * (1 - complex.I) / (1 + complex.I)

-- Proof statement to verify that the point corresponding to the complex number lies in the third quadrant
theorem complex_num_in_third_quadrant :
  let p := (complex.re complex_num, complex.im complex_num) in
  p.1 < 0 ∧ p.2 < 0 :=
by
  -- The actual proof steps are omitted as requested and replaced by 'sorry'
  sorry

end complex_num_in_third_quadrant_l706_706583


namespace tan_phi_l706_706911

theorem tan_phi (φ : ℝ) (h1 : Real.cos (π / 2 + φ) = 2 / 3) (h2 : abs φ < π / 2) : 
  Real.tan φ = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_phi_l706_706911


namespace maple_trees_cut_down_l706_706794

-- Define the initial number of maple trees.
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees after cutting.
def final_maple_trees : ℝ := 7.0

-- Define the number of maple trees cut down.
def cut_down_maple_trees : ℝ := initial_maple_trees - final_maple_trees

-- Prove that the number of cut down maple trees is 2.
theorem maple_trees_cut_down : cut_down_maple_trees = 2 := by
  sorry

end maple_trees_cut_down_l706_706794


namespace complementA_inter_B_l706_706198

noncomputable
def setA : Set ℝ := { y | ∃ x : ℝ, y = real.sqrt (x^2 - 2*x + 5) }

def setB : Set ℝ := { x | -1 < x ∧ x ≤ 4 }

def universalSet : Set ℝ := set.univ

def complementA : Set ℝ := universalSet \ setA

theorem complementA_inter_B:
  (complementA ∩ setB) = { y | -1 < y ∧ y < 2 } :=
by
  sorry

end complementA_inter_B_l706_706198


namespace max_liars_in_circle_l706_706791

-- Definitions of the conditions
def is_truthful (p : ℕ → Prop) (n : ℕ) : Prop :=
  p ((n - 1) % 16) = false ∧ p ((n + 1) % 16) = false

def is_liar (p : ℕ → Prop) (n : ℕ) : Prop :=
  ¬is_truthful p n

def valid_configuration (p : ℕ → Prop) : Prop :=
  ∀ n, (p n = true → is_liar p n) ∧ (p n = false → is_truthful p n)

-- Theorem stating the maximum number of liars is 10
theorem max_liars_in_circle : ∃ p : ℕ → Prop, (∑ n in Finset.range 16, if p n then 1 else 0) = 10 ∧ valid_configuration p :=
sorry

end max_liars_in_circle_l706_706791


namespace probability_non_adjacent_two_twos_l706_706220

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end probability_non_adjacent_two_twos_l706_706220


namespace problem_l706_706293

noncomputable def B_coordinates (θ : ℝ) (sin_θ : ℝ) (h₁ : 0 ≤ θ ∧ θ ≤ π) : ℝ × ℝ :=
  (-√(1 - sin_θ^2), sin_θ)

theorem problem (θ : ℝ) (sin_θ : ℝ) (h₁ : 0 ≤ θ ∧ θ ≤ π) (h₂ : sin_θ = 4 / 5) :
  B_coordinates θ sin_θ h₁ = (-3 / 5, 4 / 5) ∧
  (sin (π + θ) + 2 * sin (π / 2 - θ)) / (2 * cos (π - θ)) = -5 / 3 :=
by 
  calc
    coordinates_proof : B_coordinates θ sin_θ h₁ = (-√(1 - sin_θ^2), sin_θ) 
    proof_proof : coords θ sin_θ = (-√(1 - (4/5)^2), 4/5) = (-3/5, 4/5) 
  by 
  sorry 
  calc
    simpl_proof : (sin (π + θ) + 2 * sin (π / 2 - θ)) / (2 * cos (π - θ)) = (-sin θ + 2 * cos θ) / (-2 * cos θ)
    final_proof : (-4/5 + 2 * (-3/5)) / (-2 * (-3/5)) = 
      (-4/5 - 6/5) / (6/5) = -10/5) / (6/5) = -5/3 
  by
  sorry

end problem_l706_706293


namespace drive_photos_storage_l706_706843

theorem drive_photos_storage (photo_size: ℝ) (num_photos_with_videos: ℕ) (photo_storage_with_videos: ℝ) (video_size: ℝ) (num_videos_with_photos: ℕ) : 
  num_photos_with_videos * photo_size + num_videos_with_photos * video_size = 3000 → 
  (3000 / photo_size) = 2000 :=
by
  sorry

end drive_photos_storage_l706_706843


namespace strawberries_area_calculation_l706_706680

noncomputable def area_of_strawberries (diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  let total_area := π * radius^2
  let fruits_area := total_area / 2
  let strawberries_area := fruits_area / 4
  strawberries_area

theorem strawberries_area_calculation :
  area_of_strawberries 16 ≈ 25.13272 :=
by
  sorry

end strawberries_area_calculation_l706_706680


namespace even_divisors_of_8_factorial_divisible_by_3_l706_706208

theorem even_divisors_of_8_factorial_divisible_by_3 : 
  let fac8 := (40320 : ℕ) in 
  let prime_factors := multiset.of_list ([2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7] : list ℕ) in
  let divisors := (r : ℕ) → (r ∣ fac8) ∧ (∃ a b c d, r = 2^a * 3^b * 5^c * 7^d ∧ 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1) in
  finset.card {r ∈ finset.range (fac8 + 1) | divisors r} = 56 :=
begin
  sorry
end

end even_divisors_of_8_factorial_divisible_by_3_l706_706208


namespace shaded_area_is_correct_l706_706493

noncomputable def total_shaded_area : ℝ :=
  let s := 10
  let R := s / (2 * Real.sin (Real.pi / 8))
  let A := (1 / 2) * R^2 * Real.sin (2 * Real.pi / 8)
  4 * A

theorem shaded_area_is_correct :
  total_shaded_area = 200 * Real.sqrt 2 / Real.sin (Real.pi / 8)^2 := 
sorry

end shaded_area_is_correct_l706_706493


namespace area_swept_by_k8_l706_706481

-- Defining the given conditions
structure CircularDiscs (r : ℝ) :=
  (circle_radii : ℕ → ℝ)
  (is_hexagon_arrangement : ∀ n, n < 7 → circle_radii n = r)
  (touches_two_neighbors_and_center : ∀ n, n < 6 → True) -- Simplified condition

def total_area_swept (r : ℝ) (discs : CircularDiscs r) : ℝ :=
  50.07 * r^2

theorem area_swept_by_k8 (r : ℝ) (discs : CircularDiscs r) : 
  total_area_swept r discs = 50.07 * r^2 :=
begin
  sorry
end

end area_swept_by_k8_l706_706481


namespace find_smaller_number_l706_706330

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l706_706330


namespace inequality_transformations_l706_706216

variable {a b : ℝ}

theorem inequality_transformations (h : a > b) : (a - 3 > b - 3) ∧ (-4a < -4b) :=
by
  have h1 : a - 3 > b - 3 := sorry
  have h2 : -4a < -4b := sorry
  exact ⟨h1, h2⟩

end inequality_transformations_l706_706216


namespace number_of_such_fractions_is_one_l706_706520

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l706_706520


namespace average_first_50_even_numbers_l706_706482

-- Condition: The sequence starts from 2.
-- Condition: The sequence consists of the first 50 even numbers.
def first50EvenNumbers : List ℤ := List.range' 2 100

theorem average_first_50_even_numbers : (first50EvenNumbers.sum / 50 = 51) :=
by
  sorry

end average_first_50_even_numbers_l706_706482


namespace min_keystrokes_1_to_243_l706_706455

-- Defining the operations as per the conditions
inductive Op : Type
| add_one : Op
| mul_two : Op
| mul_three : Op

-- Function to apply an operation
def apply_op (op : Op) (n : ℕ) : Option ℕ :=
  match op with
  | Op.add_one   => some (n + 1)
  | Op.mul_two   => some (n * 2)
  | Op.mul_three => if n % 3 = 0 then some (n * 3) else none

-- Recursive function to calculate minimum keystrokes
def min_keystrokes : ℕ → ℕ → ℕ
| curr, target =>
  if curr = target then 0
  else if curr > target then target - curr
  else
    let opts := [Op.add_one, Op.mul_two, Op.mul_three].filterMap (λ op, apply_op op curr)
    1 + opts.map (λ new_curr => min_keystrokes new_curr target).minimum.getOrElse target

-- The main theorem statement
theorem min_keystrokes_1_to_243 : min_keystrokes 1 243 = 5 := by
  sorry

end min_keystrokes_1_to_243_l706_706455


namespace distance_between_parallel_lines_l706_706763

theorem distance_between_parallel_lines (x y : ℝ) :
    let l1 := λ x y, x - y + 1
    let l2 := λ x y, 3 * x - 3 * y + 1
    ∃ d : ℝ, d = (| (1 - (1 / 3)) | / sqrt (1 + 1)) := d =  (sqrt 2 / 3) :=
begin
  sorry
end

end distance_between_parallel_lines_l706_706763


namespace betty_age_l706_706405

theorem betty_age (A M B : ℕ) (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 22) : B = 11 :=
by
  sorry

end betty_age_l706_706405


namespace pizza_left_percentage_l706_706300

-- Define the fractions eaten by Ravindra and Hongshu
def fraction_eaten_by_ravindra := 2 / 5
def fraction_eaten_by_hongshu := 1 / 2 * fraction_eaten_by_ravindra

-- Define the total fraction eaten and the remaining fraction
def total_fraction_eaten := fraction_eaten_by_ravindra + fraction_eaten_by_hongshu
def fraction_left := 1 - total_fraction_eaten

-- Prove that the remaining fraction as a percentage
theorem pizza_left_percentage : fraction_left * 100 = 40 := by
  -- We skip the proof with sorry
  sorry

end pizza_left_percentage_l706_706300


namespace max_magnitude_of_vector_sum_l706_706241

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (3, 0)

-- Define the locus of point P such that |CP| = 1
def P_locus (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x - C.1)^2 + y^2 = 1

-- Define the vector sum OA + OB + OP
def vector_sum (P : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 + B.1 + P.1, A.2 + B.2 + P.2)

-- Define the magnitude of the vector sum
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- State the main theorem for the problem
theorem max_magnitude_of_vector_sum : ∀ P : ℝ × ℝ, 
  P_locus P → magnitude (vector_sum P) ≤ 6 :=
by sorry

end max_magnitude_of_vector_sum_l706_706241


namespace sum_inverses_eq_one_mod17_l706_706088

theorem sum_inverses_eq_one_mod17 :
  let inv_mod := fun (x : ℕ) (p : ℕ) => x^(p-2) % p in
  (inv_mod 3 17 + inv_mod (3^2) 17 + inv_mod (3^3) 17 + inv_mod (3^4) 17 + inv_mod (3^5) 17 + inv_mod (3^6) 17) % 17 = 1 :=
by
  sorry

end sum_inverses_eq_one_mod17_l706_706088


namespace tangency_points_coplanar_l706_706857

-- Definitions for points, lines, plane, and sphere.
variable {Point : Type} [MetricSpace Point]

-- A quadrilateral touching a sphere.
variable {A B C D P Q R T : Point}

-- Sphere centered at O with radius r.
noncomputable def sphere (O : Point) (r : ℝ) : Set Point := sorry

-- Define the condition: all four sides touch a sphere.
def touches_sphere (A B C D : Point) {s : Set Point} : Prop :=
  ∃ (O : Point) (r : ℝ), s = sphere O r ∧
  (∃ P, P ∈ s ∧ Segment A B ∩ s = {P}) ∧
  (∃ Q, Q ∈ s ∧ Segment B C ∩ s = {Q}) ∧
  (∃ R, R ∈ s ∧ Segment C D ∩ s = {R}) ∧
  (∃ T, T ∈ s ∧ Segment D A ∩ s = {T})

-- Statement of the problem in Lean:
theorem tangency_points_coplanar 
  (A B C D P Q R T : Point)
  (s : Set Point)
  (h1 : touches_sphere A B C D s) :
  ∃ (plane : Plane), 
  P ∈ plane ∧ Q ∈ plane ∧ R ∈ plane ∧ T ∈ plane :=
sorry

end tangency_points_coplanar_l706_706857


namespace sum_of_initial_N_ending_in_2_after_8_steps_l706_706008

-- Define the machine operation function
def machineOperation (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the inverse operation function
def inverseMachineOperation (O : ℕ) : ℕ :=
  if O % 2 = 1 then (O - 2) / 4 else 2 * O

-- Prove that the sum of all integers N that end up as 2 after 8 steps is 1020
theorem sum_of_initial_N_ending_in_2_after_8_steps : 
  (∑ N in {N : ℕ | iterate machineOperation 8 N = 2}, N) = 1020 :=
sorry

end sum_of_initial_N_ending_in_2_after_8_steps_l706_706008


namespace find_white_balls_l706_706023

noncomputable def white_balls_in_bag (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) 
  (p_not_red_nor_purple : ℚ) : ℕ :=
total_balls - (red_balls + purple_balls) - (green_balls + yellow_balls)

theorem find_white_balls :
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  white_balls_in_bag total_balls green_balls yellow_balls red_balls purple_balls p_not_red_nor_purple = 21 :=
by
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  sorry

end find_white_balls_l706_706023


namespace find_hyperbolic_lines_l706_706966

def point := (ℝ, ℝ)

def M : point := (-5, 0)
def N : point := (5, 0)

def is_hyperbolic_line (l : ℝ → ℝ) : Prop :=
  ∃ P : point, (l P.1 = P.2) ∧ ((P.1 * P.1 / 9) - (P.2 * P.2 / 16) = 1)

def hyperbolic_line1 : Prop := is_hyperbolic_line (λ x, x + 1)
def hyperbolic_line2 : Prop := is_hyperbolic_line (λ x, 2)
def hyperbolic_line3 : Prop := is_hyperbolic_line (λ x, (4/3) * x)
def hyperbolic_line4 : Prop := is_hyperbolic_line (λ x, 2 * x + 1)

theorem find_hyperbolic_lines : (hyperbolic_line1 ∧ hyperbolic_line2) ∧ ¬(hyperbolic_line3 ∨ hyperbolic_line4) :=
by {
  sorry
}

end find_hyperbolic_lines_l706_706966


namespace arithmetic_sequence_properties_l706_706103

theorem arithmetic_sequence_properties :
    ∃ n : ℕ, ∀ a₁ d : ℤ, a₁ = 100 → d = -4 → 
      let a_n := a₁ + (n - 1) * d
      in a_n = -4 ∧ ∑ i in finset.range (n + 1), (a₁ + i * d : ℤ) > 1000 ∧ n = 27 :=
by
  sorry

end arithmetic_sequence_properties_l706_706103


namespace solve_for_x_l706_706981

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h_eq : y = 1 / (3 * x^2 + 2 * x + 1)) : x = 0 ∨ x = -2 / 3 :=
by
  sorry

end solve_for_x_l706_706981


namespace linear_eq_zero_l706_706821

variables {a b c d x y : ℝ}

theorem linear_eq_zero (h1 : a * x + b * y = 0) (h2 : c * x + d * y = 0) (h3 : a * d - c * b ≠ 0) :
  x = 0 ∧ y = 0 :=
by
  sorry

end linear_eq_zero_l706_706821


namespace problem_l706_706992

variable (x1 x2 x3 x4 B : ℕ)

-- Conditions
def cond1 := x1 = 2
def cond2 := x2 = 3
def cond3 := x3 = 6
def cond4 := x4 = 16

-- Problem statement
theorem problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : B = 5 := by
  sorry

end problem_l706_706992


namespace zookeeper_fish_total_l706_706787

def fish_given : ℕ := 19
def fish_needed : ℕ := 17

theorem zookeeper_fish_total : fish_given + fish_needed = 36 :=
by
  sorry

end zookeeper_fish_total_l706_706787


namespace calc_problem_l706_706478

theorem calc_problem : (sqrt 2023 - 1) ^ 0 + (1 / 2) ^ (-1) = 3 := by
  -- Sorry allows us to skip the proof itself
  sorry

end calc_problem_l706_706478


namespace wholesale_price_is_90_l706_706051

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l706_706051


namespace total_number_of_toothpicks_l706_706377

theorem total_number_of_toothpicks
  (height : ℕ)
  (width : ℕ)
  (horiz_lines : ℕ)
  (vert_lines : ℕ)
  (diagn_diagonals : ℕ)
  (height = 15)
  (width = 15)
  (horiz_lines = 16)
  (vert_lines = 16)
  (diagn_diagonals = 2)
  (horiz_toothpicks = horiz_lines * width)
  (vert_toothpicks = vert_lines * height)
  (diag_toothpicks = diagn_diagonals * 15) :
  horiz_toothpicks + vert_toothpicks + diag_toothpicks = 510 := by
  sorry

end total_number_of_toothpicks_l706_706377


namespace sesame_seed_mass_l706_706019

theorem sesame_seed_mass :
  (∃ n : ℕ, n = 80000) ∧ (∃ m : ℝ, m = 320) → 
  ∃ s : ℝ, s = 4 * 10^(-3) :=
sorry

end sesame_seed_mass_l706_706019


namespace minimum_exam_participants_l706_706658

-- Each definition should only directly appear in the conditions.
-- Define the problem in Lean 4 statement form.

variable {n : ℕ}

def exam_problem (n : ℕ) : Prop :=
  (∀ p₁ p₂ : ℕ, p₁ ≠ p₂ → ∃ q : ℕ, 1 ≤ q ∧ q ≤ 6 ∧ 
    (p₁ ∉ question_solvers q ∨ p₂ ∉ question_solvers q)) ∧ 
  (∀ q : ℕ, 1 ≤ q ∧ q ≤ 6 → ∃! s : set ℕ, card s = 100 ∧ question_solvers q = s)

noncomputable def minimum_participants : ℕ := 200

theorem minimum_exam_participants: ∃ n, exam_problem n ∧ n = minimum_participants :=
by
  sorry

end minimum_exam_participants_l706_706658


namespace slope_angle_range_l706_706446

theorem slope_angle_range (C : Set ℝ) (l : Set ℝ) (α : ℝ)
  (h1 : C = {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1})
  (h2 : ∃ k, l = {p : ℝ × ℝ | p.2 = k * p.1} ∧ ∀ p ∈ l, p < sqrt 6) :
  (π / 4 ≤ α ∧ α ≤ 3 * π / 4) :=
sorry

end slope_angle_range_l706_706446


namespace number_of_such_fractions_is_one_l706_706518

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l706_706518


namespace xy_parallel_zw_l706_706690

-- Definitions of points in the cyclic quadrilateral and perpendicular feet 
variables {A B C D X Y Z W : Type} [EuclideanGeometry A B C D X Y Z W]

-- Condition: ABCD is cyclic (opposite angles sum to π)
axiom ABCD_is_cyclic : is_cyclic A B C D

-- Condition: X is the foot of the perpendicular from A to BC
axiom X_is_foot_from_A_to_BC : foot_perpendicular A B C X

-- Condition: Y is the foot of the perpendicular from B to AC
axiom Y_is_foot_from_B_to_AC : foot_perpendicular B A C Y

-- Condition: Z is the foot of the perpendicular from A to CD
axiom Z_is_foot_from_A_to_CD : foot_perpendicular A C D Z

-- Condition: W is the foot of the perpendicular from D to AC
axiom W_is_foot_from_D_to_AC : foot_perpendicular D A C W

-- Prove that XY is parallel to ZW
theorem xy_parallel_zw : is_parallel XY ZW :=
sorry

end xy_parallel_zw_l706_706690


namespace exists_tangent_circle_l706_706380

noncomputable def construct_tangent_circle (S1 S2 : Circle) (O : Point) : Circle :=
  sorry

theorem exists_tangent_circle (S1 S2 : Circle) (O : Point) (hO₁ : ¬ O ∈ S1) (hO₂ : ¬ O ∈ S2) :
  ∃ S : Circle, (tangent S S1 ∧ tangent S S2) ∧ (passes_through S O) :=
by
  apply noncomputable.construct_tangent_circle S1 S2 O
  use sorry

end exists_tangent_circle_l706_706380


namespace find_slope_of_line_l706_706608

noncomputable def parametric_equations (t : ℝ) : ℝ × ℝ :=
(2 + t / 2, 3 + (real.sqrt 3) / 2 * t)

def slope_of_line (m : ℝ) : Prop :=
∃ t : ℝ, let (x, y) := parametric_equations t in y = m * x + (3 - 2 * real.sqrt 3)

theorem find_slope_of_line : slope_of_line (real.sqrt 3) :=
sorry

end find_slope_of_line_l706_706608


namespace lateral_surface_area_of_rotated_square_l706_706302

theorem lateral_surface_area_of_rotated_square : 
  ∀ (a : ℝ), a = 1 → (2 * 1 * real.pi) = 2 * real.pi :=
by 
  intros a ha
  rw ha
  sorry

end lateral_surface_area_of_rotated_square_l706_706302


namespace num_tiles_visited_l706_706049

theorem num_tiles_visited (width length : ℕ) (h_w : width = 12) (h_l : length = 18) :
  width + length - Nat.gcd width length = 24 :=
by
  rw [h_w, h_l]
  simp [Nat.gcd]
  sorry

end num_tiles_visited_l706_706049


namespace li_payment_l706_706399

noncomputable def payment_li (daily_payment_per_unit : ℚ) (days_li_worked : ℕ) : ℚ :=
daily_payment_per_unit * days_li_worked

theorem li_payment (work_per_day : ℚ) (days_li_worked : ℕ) (days_extra_work : ℕ) 
  (difference_payment : ℚ) (daily_payment_per_unit : ℚ) (initial_nanual_workdays : ℕ) :
  work_per_day = 1 →
  days_li_worked = 2 →
  days_extra_work = 3 →
  difference_payment = 2700 →
  daily_payment_per_unit = difference_payment / (initial_nanual_workdays + (3 * 3)) → 
  payment_li daily_payment_per_unit days_li_worked = 450 := 
by 
  intros h_work_per_day h_days_li_worked h_days_extra_work h_diff_payment h_daily_payment 
  sorry

end li_payment_l706_706399


namespace coeff_sum_l706_706667

open BigOperators

namespace Problem

def f (m n : ℕ) (f : ℕ → ℕ → ℕ) := f m n

theorem coeff_sum :
  let f : ℕ → ℕ → ℕ := λ m n, (Nat.choose 6 m) * (Nat.choose 4 n)
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 :=
by
  let f := λ m n, (Nat.choose 6 m) * (Nat.choose 4 n)
  show f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120
  sorry

end Problem

end coeff_sum_l706_706667


namespace range_of_a_l706_706698

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < a + 2 → x ≤ 2) ↔ a ≤ 0 := by
  sorry

end range_of_a_l706_706698


namespace lollipop_problem_l706_706679

def arithmetic_sequence_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem lollipop_problem
  (a : ℕ) (h1 : arithmetic_sequence_sum a 5 7 = 175) :
  (a + 15) = 25 :=
by
  sorry

end lollipop_problem_l706_706679


namespace even_fn_solution_set_l706_706310

theorem even_fn_solution_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x ≥ 0, f x = x^3 - 8) :
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by sorry

end even_fn_solution_set_l706_706310


namespace ellipse_foci_distance_l706_706123

noncomputable def distance_between_foci : ℝ :=
  let a := 20
  let b := 10
  2 * Real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance : distance_between_foci = 20 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l706_706123


namespace change_digit_results_in_largest_number_l706_706810

theorem change_digit_results_in_largest_number :
  ∀ d1 d2 d3 d4 d5 d6 : ℕ,
  (0.172839 = d1 * (10:ℝ)^-1 + d2 * (10:ℝ)^-2 + d3 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) →
  (∀ (i:ℕ), i ≠ 1 → d1 = 1 → d2 = 7 → d3 = 2 → d4 = 8 → d5 = 3 → d6 = 9) →
  (9 * (10:ℝ)^-1 + d2 * (10:ℝ)^-2 + d3 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) >
  (d1 * (10:ℝ)^-1 + 9 * (10:ℝ)^-2 + d3 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) ∧
  (9 * (10:ℝ)^-1 + d2 * (10:ℝ)^-2 + d3 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) >
  (d1 * (10:ℝ)^-1 + d2 * (10:ℝ)^-2 + 9 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) ∧
  (9 * (10:ℝ)^-1 + d2 * (10:ℝ)^-2 + d3 * (10:ℝ)^-3 + d4 * (10:ℝ)^-4 + d5 * (10:ℝ)^-5 + d6 * (10:ℝ)^-6) >
  0.172839 := by
  sorry

end change_digit_results_in_largest_number_l706_706810


namespace quadratic_function_vertex_yint_eq_l706_706181

theorem quadratic_function_vertex_yint_eq :
  ∃ (a : ℝ), (∀ x : ℝ, (y = a * (x - 3)^2 - 1) ∧ (a * (-3)^2 - 1 = -4)) →
  (∀ x : ℝ, y = - (1/3) * x^2 + 2 * x - 4) :=
begin
  sorry
end

end quadratic_function_vertex_yint_eq_l706_706181


namespace proof_of_calculation_l706_706456

theorem proof_of_calculation : (7^2 - 5^2)^4 = 331776 := by
  sorry

end proof_of_calculation_l706_706456


namespace divisible_by_323_if_even_l706_706143

theorem divisible_by_323_if_even (n : ℤ) : 
  (20 ^ n + 16 ^ n - 3 ^ n - 1) % 323 = 0 ↔ n % 2 = 0 := 
by 
  sorry

end divisible_by_323_if_even_l706_706143


namespace beavers_still_working_is_one_l706_706429

def initial_beavers : Nat := 2
def beavers_swimming : Nat := 1
def still_working_beavers : Nat := initial_beavers - beavers_swimming

theorem beavers_still_working_is_one : still_working_beavers = 1 :=
by
  sorry

end beavers_still_working_is_one_l706_706429


namespace length_AD_l706_706046

open EuclideanGeometry

-- Define the quadrilateral ABCD being inscribed in a circle
variable {A B C D : Point}

-- Conditions
hypothesis (h1 : Inscribed ABCD)
hypothesis (h2 : SegLength AB = 5)
hypothesis (h3 : SegLength CD = 3)
hypothesis (h4 : ∃ M : Point, IsAngleBisector ∠B AM ∧ IsAngleBisector ∠C DM ∧ M ∈ Seg AD)

-- Theorem: The length of segment AD
theorem length_AD : SegLength AD = 8 :=
by
  sorry

end length_AD_l706_706046


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706337

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706337


namespace base_b_of_256_has_4_digits_l706_706425

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l706_706425


namespace magnolia_trees_below_threshold_l706_706892

-- Define the initial number of trees and the function describing the decrease
def initial_tree_count (N₀ : ℕ) (t : ℕ) : ℝ := N₀ * (0.8 ^ t)

-- Define the year when the number of trees is less than 25% of initial trees
theorem magnolia_trees_below_threshold (N₀ : ℕ) : (t : ℕ) -> initial_tree_count N₀ t < 0.25 * N₀ -> t > 14 := 
-- Provide the required statement but omit the actual proof with "sorry"
by sorry

end magnolia_trees_below_threshold_l706_706892


namespace avg_GPA_is_93_l706_706754

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l706_706754


namespace unique_fraction_property_l706_706513

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l706_706513


namespace digit_sum_l706_706135

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem digit_sum theorem (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 9999) (h3 : n / sum_of_digits n = 112) :
  n = 1008 ∨ n = 1344 ∨ n = 1680 ∨ n = 2688 :=
by
  sorry

end digit_sum_l706_706135


namespace train_speed_l706_706378

theorem train_speed (L1 L2 : ℕ) (V2 : ℕ) (t : ℝ) (V1 : ℝ) : 
  L1 = 200 → 
  L2 = 280 → 
  V2 = 30 → 
  t = 23.998 → 
  (0.001 * (L1 + L2)) / (t / 3600) = V1 + V2 → 
  V1 = 42 :=
by 
  intros
  sorry

end train_speed_l706_706378


namespace eccentricity_range_l706_706150

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (h1 : a > 0) (h2 : b > 0)
def hyperbola : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the relationship of b to a and c
axiom b_rel : b = sqrt(c^2 - a^2)

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := c / a

-- The proof statement for the range of eccentricity
theorem eccentricity_range (h3 : sqrt(c^2 - a^2) < a) : 1 < eccentricity a b c ∧ eccentricity a b c < sqrt 2 := by
  sorry

end eccentricity_range_l706_706150


namespace sum_odd_numbers_3_l706_706013

noncomputable def arithmetic_series : ℕ → ℕ := λ n, n * (n + 1)

theorem sum_odd_numbers_3
  (sum_arith : ∀ n : ℕ, 1 + 2 + 3 + 4 + ... + n = arithmetic_series n) :
  3 * (1 + 3 + 5 + ... + 89) = 6075 :=
by
  sorry

end sum_odd_numbers_3_l706_706013


namespace num_three_digit_nums_l706_706211

theorem num_three_digit_nums : 
  ∀ (digits : Set ℕ) (n : ℕ),
    digits = {1, 2, 3} → 
    n = 3 →
    (∀ x ∈ digits, 1 ≤ x ∧ x ≤ 3) → 
    fintype.card {p : List ℕ // ∀ x ∈ p, x ∈ digits ∧ p.nodup ∧ p.length = n} = 6 :=
by
  sorry

end num_three_digit_nums_l706_706211


namespace probability_of_z_l706_706233

noncomputable def P_x : ℝ := 1 / 7
noncomputable def P_y : ℝ := 1 / 3
noncomputable def total : ℝ := 0.6761904761904762

theorem probability_of_z : 
  ∃ P_z : ℝ, P_z = 0.2 ∧ (P_x + P_y + P_z = total) := 
by
  use (0.2)
  split
  { refl }
  { sorry }

end probability_of_z_l706_706233


namespace trains_clear_time_l706_706415

-- Define the lengths of the trains
def length_train1 : ℝ := 220
def length_train2 : ℝ := 280

-- Define the speeds of the trains in m/s
def speed_train1 : ℝ := 42 * (1000 / 3600)
def speed_train2 : ℝ := 30 * (1000 / 3600)

-- Calculate the total length they need to cover
def total_length : ℝ := length_train1 + length_train2

-- Calculate the relative speed
def relative_speed : ℝ := speed_train1 + speed_train2

-- Calculate the time for them to be clear of each other
def time_to_clear : ℝ := total_length / relative_speed

-- Statement to prove
theorem trains_clear_time :
  time_to_clear = 25 := by
  sorry

end trains_clear_time_l706_706415


namespace average_weight_is_fifteen_l706_706224

noncomputable def regression_equation (x : ℕ) : ℕ := 2 * x + 7

def children_ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

def average (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0 / l.length

theorem average_weight_is_fifteen :
  (regression_equation (average children_ages)) = 15 :=
by
  have avg_age := average children_ages
  have avg_weight := regression_equation avg_age
  show avg_weight = 15
  sorry

end average_weight_is_fifteen_l706_706224


namespace simplify_expression_l706_706733

theorem simplify_expression (n : ℕ) :
  let sum_cubes := (n * (n + 1) / 2) ^ 2 in
  ( ( ∑ k in finset.range n, 8 * k^3 ) / ( ∑ k in finset.range n, 27 * k^3 ) ) ^ (1/3 : ℝ) = (2/3 : ℝ) :=
by {
  -- Mathematical proof goes here
  sorry
}

end simplify_expression_l706_706733


namespace chemist_mixtures_l706_706401

theorem chemist_mixtures (x : ℝ) :
  let pure_water := 1
  let salt_solution_percentage := 0.30
  let salt_solution := x
  let final_solution_percentage := 0.20
  let final_solution := pure_water + salt_solution
  (salt_solution_percentage * salt_solution = final_solution_percentage * final_solution) →
  x = 2 :=
by
  intro h
  simp at h
  sorry

end chemist_mixtures_l706_706401


namespace total_parents_surveyed_l706_706064

-- Define the given conditions
def percent_agree : ℝ := 0.20
def percent_disagree : ℝ := 0.80
def disagreeing_parents : ℕ := 640

-- Define the statement to prove
theorem total_parents_surveyed :
  ∃ (total_parents : ℕ), disagreeing_parents = (percent_disagree * total_parents) ∧ total_parents = 800 :=
by
  sorry

end total_parents_surveyed_l706_706064


namespace minimum_cost_l706_706440

noncomputable def f (x : ℝ) : ℝ := (1000 / (x + 5)) + 5 * x + (1 / 2) * (x^2 + 25)

theorem minimum_cost :
  (2 ≤ x ∧ x ≤ 8) →
  (f 5 = 150 ∧ (∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f 5)) :=
by
  intro h
  have f_exp : f x = (1000 / (x+5)) + 5*x + (1/2)*(x^2 + 25) := rfl
  sorry

end minimum_cost_l706_706440


namespace find_length_PB_l706_706647

open Real

noncomputable def length_of_PB (x : ℝ) : ℝ := 2 * x + 2 

-- Conditions:
variables {M P A B C : Point} {x : ℝ}
variable (H_midpoint_arc : MidpointArc M C A B) 
variable (H_perpendicular : Perpendicular MP AB)
variable (H_mid_P : Midpoint P A B)
variable (H_length_AC : Length A C = x)
variable (H_length_AP : Length A P = 2 * x + 2)

-- Proof Statement: Find the length of PB
theorem find_length_PB (H_midpoint_arc : MidpointArc M C A B) 
  (H_perpendicular : Perpendicular MP AB)
  (H_mid_P : Midpoint P A B)
  (H_length_AC : Length A C = x)
  (H_length_AP : Length A P = 2 * x + 2) : 
  Length P B = 2 * x + 2 :=
sorry

end find_length_PB_l706_706647


namespace ellipse_focus_range_OP_l706_706182

-- Define the ellipse equation with a > b > 0
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Main theorem: proving the conditions lead to the specific ellipse equation and |OP| range
theorem ellipse_focus_range_OP 
  (x y a b : ℝ) 
  (h_ellipse : ellipse x y a b) 
  (hx : x = sqrt 2 ∧ y = 0 ∧ a > b > 0) 
  (hl : ∀ m k, abs k ≤ sqrt 2 / 2 → (∃ x y, x^2 / a^2 + y^2 / b^2 = 1 ∧ y = k * x + m)) :
  (a = 2 ∧ b = sqrt 2 ∧ 
  ∀ k m, |k| ≤ sqrt 2 / 2 →
  ((exists x y, (x^2 / 4 + y^2 / 2 = 1) ∧ y = k * x + m ∧ 
  (sqrt 2 ≤ sqrt (x^2 + y^2) ∧ sqrt (x^2 + y^2) ≤ sqrt 3))) :=
sorry

end ellipse_focus_range_OP_l706_706182


namespace sum_of_special_numbers_l706_706128

theorem sum_of_special_numbers : 
  (∑ N in {N : ℕ | (∃ p : ℕ, prime p ∧ N = p^3 ∧ N ≥ 15 ∧ (N.count_divisors = 4) ∧ ∀ d ∈ {d : ℕ | d ∣ N}, d < 15 ∨ d = N)} ∪ 
    {N : ℕ | (∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ N = p * q ∧ N ≥ 15 ∧ (N.count_divisors = 4) ∧ ∀ d ∈ {d : ℕ | d ∣ N}, d < 15 ∨ d = N)})
  = 649 :=
by
  sorry

end sum_of_special_numbers_l706_706128


namespace solve_for_s_l706_706526

def F (a b c : ℝ) : ℝ := a * b^c

theorem solve_for_s (s : ℝ) (h_pos : 0 < s) (h : F(s, s, 4) = 1296) : s = 6^(4/5) :=
by
  sorry

end solve_for_s_l706_706526


namespace mobius_inversion_l706_706296

noncomputable def F (n : ℕ) : ℕ := sorry
noncomputable def f (n : ℕ) : ℕ := sorry
noncomputable def μ (n : ℕ) : ℕ := sorry

theorem mobius_inversion (F : ℕ → ℕ) (f : ℕ → ℕ)
  (μ : ℕ → ℕ) (n : ℕ)
  (h : F n = ∑ d in (finset.range (n+1)).filter (λ d, d ∣ n), f d) :
  f n = ∑ d in (finset.range (n+1)).filter (λ d, d ∣ n), μ d * F (n / d)
      = ∑ d in (finset.range (n+1)).filter (λ d, d ∣ n), μ (n / d) * F d :=
begin
  sorry
end

end mobius_inversion_l706_706296


namespace length_of_BC_l706_706260

theorem length_of_BC (BD CD : ℝ) (h1 : BD = 3 + 3 * BD) (h2 : CD = 2 + 2 * CD) (h3 : 4 * BD + 3 * CD + 5 = 20) : 2 * CD + 2 = 4 :=
by {
  sorry
}

end length_of_BC_l706_706260


namespace parabola_and_line_existence_l706_706589

noncomputable def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

noncomputable def midpoint_condition (x₁ y₁ x₂ y₂ px py : ℝ) : Prop :=
  px = (x₁ + x₂) / 2 ∧ py = (y₁ + y₂) / 2

theorem parabola_and_line_existence :
  ∃ (p : ℝ), ∃ (x y : ℝ), 
    let M := (5, y),
    let F := (p / 2, 0),
    let d := real.sqrt ((5 - p/2) ^ 2 + (y - 0) ^ 2),
    (d = 6) ∧ parabola_equation 2 5 y
    ∧
    ∃ (k : ℝ), ∀ x₁ y₁ x₂ y₂, 
      let l_eq := (x = k * (y + 1) + 2),
      let P := (2, -1),
      midpoint_condition x₁ y₁ x₂ y₂ 2 (-1) →
      parabola_equation 2 x₁ y₁ ∧
      parabola_equation 2 x₂ y₂ →
      l_eq = (2*x + y - 3 = 0) := 
sorry

end parabola_and_line_existence_l706_706589


namespace smallest_term_l706_706105

theorem smallest_term (a1 d : ℕ) (h_a1 : a1 = 7) (h_d : d = 7) :
  ∃ n : ℕ, (a1 + (n - 1) * d) > 150 ∧ (a1 + (n - 1) * d) % 5 = 0 ∧
  (∀ m : ℕ, (a1 + (m - 1) * d) > 150 ∧ (a1 + (m - 1) * d) % 5 = 0 → (a1 + (m - 1) * d) ≥ (a1 + (n - 1) * d)) → a1 + (n - 1) * d = 175 :=
by
  -- We need to prove given the conditions.
  sorry

end smallest_term_l706_706105


namespace calc_problem_l706_706477

theorem calc_problem : (sqrt 2023 - 1) ^ 0 + (1 / 2) ^ (-1) = 3 := by
  -- Sorry allows us to skip the proof itself
  sorry

end calc_problem_l706_706477


namespace sum_sequence_100_l706_706984

-- Define the sequence a_n
def a (n : ℕ) : ℤ := (-1)^(n + 1) * n

-- Define S_n as the sum of the first n terms of the sequence a
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a (i + 1)

-- State the theorem to prove that S_100 = -50
theorem sum_sequence_100 : S 100 = -50 := 
by
  sorry

end sum_sequence_100_l706_706984


namespace num_distinct_sets_of_8_positive_odd_integers_sum_to_20_l706_706207

def numDistinctOddSets (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem num_distinct_sets_of_8_positive_odd_integers_sum_to_20 :
  numDistinctOddSets 6 8 = 1716 :=
by
  sorry

end num_distinct_sets_of_8_positive_odd_integers_sum_to_20_l706_706207


namespace fraction_of_students_who_walk_home_l706_706228

theorem fraction_of_students_who_walk_home :
  let busFraction := 1 / 3
  let carpoolFraction := 1 / 5
  let scooterFraction := 1 / 8
  -- Calculate total fraction of students who use the bus, carpool, or scooter
  let totalNonWalkingFraction := busFraction + carpoolFraction + scooterFraction
  -- Find the fraction of students who walk home
  let walkingFraction := 1 - totalNonWalkingFraction
  walkingFraction = 41 / 120 :=
by
  -- Define the fractions
  let busFraction := 1 / 3
  let carpoolFraction := 1 / 5
  let scooterFraction := 1 / 8

  -- Calculate totalNonWalkingFraction
  have h1 : totalNonWalkingFraction = busFraction + carpoolFraction + scooterFraction := rfl
  have h2 : totalNonWalkingFraction = 1 / 3 + 1 / 5 + 1 / 8 := by rw [h1]
  
  -- Use common denominators
  have h3 : 1 / 3 = 40 / 120 := by norm_num
  have h4 : 1 / 5 = 24 / 120 := by norm_num
  have h5 : 1 / 8 = 15 / 120 := by norm_num
  have h6 : totalNonWalkingFraction = 40 / 120 + 24 / 120 + 15 / 120 := by rw [h2, h3, h4, h5]
  
  -- Add the fractions
  have h7 : totalNonWalkingFraction = 79 / 120 := by norm_num
  
  -- Calculate walkingFraction
  have h8 : walkingFraction = 1 - (79 / 120) := by rw h7
  have h9 : walkingFraction = 120 / 120 - 79 / 120 := by rw h8
  
  -- Simplify to find the desired fraction
  have h10 : walkingFraction = 41 / 120 := by norm_num
  
  -- Completed proof
  exact h10

end fraction_of_students_who_walk_home_l706_706228


namespace largest_prime_factor_always_divides_l706_706490

theorem largest_prime_factor_always_divides (digits_sum : ℕ) (T : ℕ) 
  (h1 : ∀ (seq : List ℕ), seq.length = 5 → 
        (∀ (n : ℕ), n ∈ seq → ∃ (a b c d e : ℕ), 
        seq = [a * 10000 + b * 1000 + c * 100 + d * 10 + e + n,
               e * 10000 + a * 1000 + b * 100 + c * 10 + d, 
               d * 10000 + e * 1000 + a * 100 + b * 10 + c,
               c * 10000 + d * 1000 + e * 100 + a * 10 + b,
               b * 10000 + c * 1000 + d * 100 + e * 10 + a])) :
  41 * 271 ∣ T →
  (∀ (seq : List ℕ), seq.length = 5 → 
    T = 11111 * digits_sum) →
  ∃ (p : ℕ), p.prime ∧ p = 271 := by
    sorry

end largest_prime_factor_always_divides_l706_706490


namespace necessary_but_not_sufficient_condition_for_x_gt_2_l706_706947

theorem necessary_but_not_sufficient_condition_for_x_gt_2 :
  ∀ (x : ℝ), (2 / x < 1 → x > 2) ∧ (x > 2 → 2 / x < 1) → (¬ (x > 2 → 2 / x < 1) ∨ ¬ (2 / x < 1 → x > 2)) :=
by
  intro x h
  sorry

end necessary_but_not_sufficient_condition_for_x_gt_2_l706_706947


namespace acme_vowel_soup_6letter_words_l706_706070

def acme_vowel_soup_total_words
: Nat :=
  let aeo_count := 7
  let iu_count := 5
  let words_with_0_I := 4^6
  let words_with_1_I := 6 * 4^5
  let words_with_2_I := 15 * 4^4
  words_with_0_I + words_with_1_I + words_with_2_I

theorem acme_vowel_soup_6letter_words : acme_vowel_soup_total_words = 14080 := by
  let words_with_0_I := 4^6
  let words_with_1_I := 6 * 4^5
  let words_with_2_I := 15 * 4^4
  have h1: words_with_0_I = 4096 := by sorry
  have h2: words_with_1_I = 6144 := by sorry
  have h3: words_with_2_I = 3840 := by sorry
  have h_total: words_with_0_I + words_with_1_I + words_with_2_I = 14080 := by 
    rw [h1, h2, h3]
    calc 
      4096 + 6144 + 3840 = 14080 := by linarith
  exact h_total

end acme_vowel_soup_6letter_words_l706_706070


namespace smallest_sum_of_consecutive_integers_is_square_l706_706355

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l706_706355


namespace sum_arithmetic_sequence_of_cis_angles_l706_706891

def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ)

theorem sum_arithmetic_sequence_of_cis_angles :
  let θs := (list.range (5 + 1)).map (λ k, 80 + 10 * k)
  let s := θs.sum (λ θ, cis (θ * real.pi / 180))
  ∃ r : ℝ, r > 0 ∧
  ∃ θ ∈ finset.Ico 0 360, s = (r : ℂ) * cis (θ * real.pi / 180) ∧ θ = 105 :=
sorry

end sum_arithmetic_sequence_of_cis_angles_l706_706891


namespace volume_of_regular_triangular_pyramid_l706_706131

theorem volume_of_regular_triangular_pyramid {h Q : ℝ} :
  volume_of_pyramid h Q = (1 / 2) * h * sqrt 3 * sqrt(h^2 + (4 * Q^2 / 3)) - h^2 :=
sorry

noncomputable def volume_of_pyramid (h Q : ℝ) : ℝ :=
  (1 / 3) * h * (sqrt 3 / 4) *
    (sqrt ((-4 * h^2 + sqrt (16 * h^4 + 192 * Q^2)) / 6) ^ 2)

end volume_of_regular_triangular_pyramid_l706_706131


namespace range_of_x_l706_706636

-- Define the condition: the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := 3 + x ≥ 0

-- Define what we want to prove: the range of x such that the condition holds
theorem range_of_x (x : ℝ) : condition x ↔ x ≥ -3 :=
by
  -- Proof goes here
  sorry

end range_of_x_l706_706636


namespace edward_spent_on_books_l706_706543

def money_spent_on_books (initial_amount spent_on_pens amount_left : ℕ) : ℕ :=
  initial_amount - amount_left - spent_on_pens

theorem edward_spent_on_books :
  ∃ (x : ℕ), x = 6 → 
  ∀ {initial_amount spent_on_pens amount_left : ℕ},
    initial_amount = 41 →
    spent_on_pens = 16 →
    amount_left = 19 →
    x = money_spent_on_books initial_amount spent_on_pens amount_left :=
by
  sorry

end edward_spent_on_books_l706_706543


namespace average_GPA_school_l706_706751

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l706_706751


namespace evaluation_result_l706_706112

theorem evaluation_result : 
  (Int.floor (Real.ceil ((15/8 : Real)^2) + (11/3 : Real)) = 7) := 
sorry

end evaluation_result_l706_706112


namespace third_prize_probability_any_prize_probability_l706_706061

def events := finset (ℕ × ℕ)
def draws : events := {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
                        (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)}

noncomputable def prob (s : events) : ℚ := (s.card : ℚ) / (draws.card : ℚ)

def third_prize_draws : events := {(1, 3), (2, 2), (3, 1), (0, 3), (1, 2), (2, 1), (3, 0)}
noncomputable def prob_third_prize : ℚ := prob third_prize_draws

def any_prize_draws : events := {(0, 3), (1, 2), (2, 1), (3, 0), (1, 3), (2, 2), (3, 1),
                                 (2, 3), (3, 2), (3, 3)}
noncomputable def prob_any_prize : ℚ := prob any_prize_draws

theorem third_prize_probability : prob_third_prize = 7 / 16 := by sorry

theorem any_prize_probability : prob_any_prize = 5 / 8 := by sorry

end third_prize_probability_any_prize_probability_l706_706061


namespace sphere_surface_area_of_inscribed_cube_l706_706784

-- Definition of the problem conditions
def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

def cube_side_length_from_surface_area : ℝ := 3

def sphere_radius_from_cube_side_length (a : ℝ) : ℝ :=
  (a * Real.sqrt 3) / 2

def sphere_surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

-- Statement to prove
theorem sphere_surface_area_of_inscribed_cube :
  cube_surface_area cube_side_length_from_surface_area = 54 →
  sphere_surface_area (sphere_radius_from_cube_side_length cube_side_length_from_surface_area) = 27 * Real.pi :=
by
  sorry

end sphere_surface_area_of_inscribed_cube_l706_706784


namespace find_hypotenuse_square_l706_706874

noncomputable def complex_zeros (p q r : ℂ) : Prop :=
  ∃ (s t : ℂ), p^3 - 2*p^2 + s*p + t = 0 ∧ q^3 - 2*q^2 + s*q + t = 0 ∧ r^3 - 2*r^2 + s*r + t = 0 

def right_triangle (p q r : ℂ) : Prop :=
  ∃ (k : ℝ), k^2 = complex.abs (p - q)^2 + complex.abs (q - r)^2 ∧ k^2 = complex.abs (r - p)^2

theorem find_hypotenuse_square (p q r : ℂ) (k : ℝ) :
  complex_zeros p q r →
  (complex.abs p)^2 + (complex.abs q)^2 + (complex.abs r)^2 = 300 →
  right_triangle p q r →
  k^2 = 300 :=
by
  sorry

end find_hypotenuse_square_l706_706874


namespace no_real_roots_l706_706495

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^6 - 3 * x^5 + 6 * x^4 - 6 * x^3 - x + 8

-- The problem can be stated as proving that Q(x) has no real roots
theorem no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end no_real_roots_l706_706495


namespace set_contains_infinite_integer_points_l706_706244

noncomputable def M : set (ℤ × ℝ) :=
  {p | ∃ (x : ℤ), p = (x, floor (real.sqrt 2 * x)) ∧ 
     (real.sqrt 2 * x - 1 : ℝ) < floor (real.sqrt 2 * x) ∧ floor (real.sqrt 2 * x) < real.sqrt 2 * x}

theorem set_contains_infinite_integer_points : 
  ∃ (M : set (ℤ × ℝ)), 
    (∀ (n : ℕ), ∃ (p : ℤ × ℝ), p ∈ M) ∧
    (∀ (l : ℝ → ℝ), ∃ (N : ℕ), ∀ (p : ℤ × ℝ), p ∈ M → p.2 ≠ l p.1) :=
by
  use M
  sorry

end set_contains_infinite_integer_points_l706_706244


namespace eval_floor_expr_l706_706113

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end eval_floor_expr_l706_706113


namespace trajectory_of_center_of_moving_circle_is_hyperbola_branch_l706_706450

theorem trajectory_of_center_of_moving_circle_is_hyperbola_branch
  (M : ℝ × ℝ)
  (r : ℝ)
  (O C : ℝ × ℝ)
  (hO : O = (0, 0))
  (hC : C = (3, 0))
  (tangent_to_O : dist M O = r + 1)
  (tangent_to_C : dist M C = r - 1) :
  ∃ H : ℝ × ℝ → ℝ, ∃ k : ℝ, is_branch_of_hyperbola_with_foci O C k H M :=
sorry

end trajectory_of_center_of_moving_circle_is_hyperbola_branch_l706_706450


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l706_706343

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l706_706343


namespace sum_of_rationals_l706_706563

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_lowest_terms (n d : ℕ) : Prop := is_coprime n d

def valid_rationals (n : ℕ) : Prop := 1 ≤ n ∧ n < 300 ∧ is_lowest_terms n 30

theorem sum_of_rationals : 
  (∑ n in Finset.filter valid_rationals (Finset.range 300), (n / 30 : ℚ)) = 400 :=
sorry

end sum_of_rationals_l706_706563


namespace liars_count_in_room_l706_706654

theorem liars_count_in_room (n : ℕ) (liars : ℕ) (knights : ℕ) 
  (h1 : n = liars + knights)
  (h2 : liars > 1) 
  (h3 : ∀ k, k < knights → (k = liars → k = eye_color_group k).succ = n) :
  liars = 2 ∨ liars = 3 ∨ liars = 5 ∨ liars = 6 ∨ liars = 10 ∨ liars = 15 ∨ liars = 30 := 
by {
  sorry
}

end liars_count_in_room_l706_706654


namespace oj_fraction_is_11_over_30_l706_706001

-- Define the capacity of each pitcher
def pitcher_capacity : ℕ := 600

-- Define the fraction of orange juice in each pitcher
def fraction_oj_pitcher1 : ℚ := 1 / 3
def fraction_oj_pitcher2 : ℚ := 2 / 5

-- Define the amount of orange juice in each pitcher
def oj_amount_pitcher1 := pitcher_capacity * fraction_oj_pitcher1
def oj_amount_pitcher2 := pitcher_capacity * fraction_oj_pitcher2

-- Define the total amount of orange juice after both pitchers are poured into the large container
def total_oj_amount := oj_amount_pitcher1 + oj_amount_pitcher2

-- Define the total volume of the mixture in the large container
def total_mixture_volume := 2 * pitcher_capacity

-- Define the fraction of the mixture that is orange juice
def oj_fraction_in_mixture := total_oj_amount / total_mixture_volume

-- Prove that the fraction of the mixture that is orange juice is 11/30
theorem oj_fraction_is_11_over_30 : oj_fraction_in_mixture = 11 / 30 := by
  sorry

end oj_fraction_is_11_over_30_l706_706001


namespace school_avg_GPA_l706_706749

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l706_706749


namespace midpoint_trajectory_rhombus_l706_706148

theorem midpoint_trajectory_rhombus (cube : Set Point) 
  (A B C D A' B' C' D' X Y Z : Point) 
  (h_cube : is_cube cube A B C D A' B' C' D')
  (h_path_X : moves_along_square_const_speed X [A, B, C, D, A]) 
  (h_path_Y : moves_along_square_const_speed Y [B', C', C, B, B'])
  (h_start_X : starts_from X A)
  (h_start_Y : starts_from Y B'):
  trajectory_of_midpoint Z X Y = rhombus E F C G :=
sorry

end midpoint_trajectory_rhombus_l706_706148


namespace average_milk_correct_l706_706720

-- Definitions of container counts and corresponding milk volumes
def n1 : ℕ := 6
def m1 : ℝ := 1.5
def n2 : ℕ := 4
def m2 : ℝ := 0.67
def n3 : ℕ := 5
def m3 : ℝ := 0.875
def n4 : ℕ := 3
def m4 : ℝ := 2.33
def n5 : ℕ := 2
def m5 : ℝ := 1.25

-- The total amount of milk
def total_milk : ℝ := n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4 + n5 * m5

-- The total number of containers
def total_containers : ℕ := n1 + n2 + n3 + n4 + n5

-- The average amount of milk per container
def average_milk_per_container : ℝ := total_milk / total_containers

-- The given proof statement
theorem average_milk_correct : average_milk_per_container = 1.27725 := by
  sorry

end average_milk_correct_l706_706720


namespace simplify_sqrt_expression_l706_706807

theorem simplify_sqrt_expression : sqrt (36 * sqrt (18 * sqrt 9)) = 6 * sqrt 6 :=
by sorry

end simplify_sqrt_expression_l706_706807


namespace snowman_volume_l706_706257

theorem snowman_volume
  (r1 r2 r3 : ℝ)
  (volume : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h_volume : volume = (4.0 / 3.0) * Real.pi * (r1 ^ 3 + r2 ^ 3 + r3 ^ 3)) :
  volume = (1124.0 / 3.0) * Real.pi :=
by
  sorry

end snowman_volume_l706_706257


namespace plates_required_for_week_l706_706283

theorem plates_required_for_week :
  let plates_used_3_days := 3 * 2 in
  let plates_used_4_days := 4 * 4 * 2 in
  plates_used_3_days + plates_used_4_days = 38 :=
by
  let plates_used_3_days := 3 * 2
  let plates_used_4_days := 4 * 4 * 2
  show plates_used_3_days + plates_used_4_days = 38
  sorry

end plates_required_for_week_l706_706283


namespace brother_books_total_l706_706866

-- Define the conditions
def sarah_paperbacks : ℕ := 6
def sarah_hardbacks : ℕ := 4
def brother_paperbacks : ℕ := sarah_paperbacks / 3
def brother_hardbacks : ℕ := 2 * sarah_hardbacks

-- Define the statement to be proven
theorem brother_books_total : brother_paperbacks + brother_hardbacks = 10 :=
by
  -- Proof will be added here
  sorry

end brother_books_total_l706_706866


namespace new_credit_card_balance_l706_706282

theorem new_credit_card_balance (i g x r n : ℝ)
    (h_i : i = 126)
    (h_g : g = 60)
    (h_x : x = g / 2)
    (h_r : r = 45)
    (h_n : n = (i + g + x) - r) :
    n = 171 :=
sorry

end new_credit_card_balance_l706_706282


namespace semi_monotonous_count_l706_706528

def is_digit_sequence_increasing (digits : List ℕ) : Prop :=
  ∀ i j, i < j → digits.nth i < digits.nth j

def is_digit_sequence_decreasing (digits : List ℕ) : Prop :=
  ∀ i j, i < j → digits.nth i > digits.nth j

def is_semi_monotonous (n : ℕ) : Prop :=
  let digits := n.digits 10
  n ≥ 10 ∧
  (is_digit_sequence_increasing digits ∨ is_digit_sequence_decreasing digits) ∧
  digits.erase_dup.length > 1 ∧
  (digits.head ≠ 0 ∨ digits.drop 1.head ≠ 0)

theorem semi_monotonous_count : 
  ∃ (count : ℕ), count = 2026 ∧ 
  count = (List.range' 10 (9876543210 - 10 + 1)).count is_semi_monotonous := sorry

end semi_monotonous_count_l706_706528


namespace find_pizza_price_without_discount_l706_706449

-- Define the given conditions
def total_cost_after_discount := 37
def num_pizzas := 3
def discount_per_pizza := 0.04

-- Define the variable to solve for
noncomputable def pizza_price_without_discount (P : ℚ) : Prop :=
  num_pizzas * P * (1 - (discount_per_pizza * (num_pizzas - 1))) = total_cost_after_discount

-- Prove that the cost of one pizza without discount is approximately $13.41
theorem find_pizza_price_without_discount : pizza_price_without_discount 13.4057971 :=
by
  -- Proof omitted
  sorry

end find_pizza_price_without_discount_l706_706449


namespace min_distance_from_P_to_Q_l706_706934

noncomputable def point_P_moves_on_curve (P : ℝ × ℝ) : Prop :=
  ∃ (ρ θ : ℝ), P = (ρ * cos θ, ρ * sin θ) ∧ ρ^2 * cos θ - 2 * ρ = 0

def polar_to_cart (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem min_distance_from_P_to_Q :
  let Q := polar_to_cart 1 (π / 3) in
  Q = (1/2, real.sqrt 3 / 2) →
  ∃ P : ℝ × ℝ, 
  point_P_moves_on_curve P →
  distance P Q = 3 / 2 :=
by
  intro Q hQ
  rw hQ
  use (2, 0) -- The point (2, 0) lies on the line x = 2
  split
  sorry -- Proof that (2, 0) satisfies the curve equation
  sorry -- Calculation of the minimum distance

end min_distance_from_P_to_Q_l706_706934


namespace wholesale_price_is_90_l706_706052

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l706_706052


namespace sin_I_l706_706660

theorem sin_I (k : ℝ) (hGHI : ∠GHI = 90) (sin_G : sin G = 3/5) : sin I = 3/5 :=
sorry

end sin_I_l706_706660


namespace factorial_div_add_two_l706_706876

def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_div_add_two :
  (factorial 50) / (factorial 48) + 2 = 2452 :=
by
  sorry

end factorial_div_add_two_l706_706876


namespace quadratic_function_proof_l706_706157

noncomputable def quadratic_function_condition (a b c : ℝ) :=
  ∀ x : ℝ, ((-3 ≤ x ∧ x ≤ 1) → (a * x^2 + b * x + c) ≤ 0) ∧
           ((x < -3 ∨ 1 < x) → (a * x^2 + b * x + c) > 0) ∧
           (a * 2^2 + b * 2 + c) = 5

theorem quadratic_function_proof (a b c : ℝ) (m : ℝ)
  (h : quadratic_function_condition a b c) :
  (a = 1 ∧ b = 2 ∧ c = -3) ∧ (m ≥ -7/9 ↔ ∃ x : ℝ, a * x^2 + b * x + c = 9 * m + 3) :=
by
  sorry

end quadratic_function_proof_l706_706157


namespace trapezoid_count_in_regular_18_gon_l706_706573

theorem trapezoid_count_in_regular_18_gon : 
  let M := regular_polygon 18 in
  number_of_trapezoid_sets M = 504 :=
by sorry

end trapezoid_count_in_regular_18_gon_l706_706573


namespace combined_percent_increase_proof_l706_706083

variable (initial_stock_A_price : ℝ := 25)
variable (initial_stock_B_price : ℝ := 45)
variable (initial_stock_C_price : ℝ := 60)
variable (final_stock_A_price : ℝ := 28)
variable (final_stock_B_price : ℝ := 50)
variable (final_stock_C_price : ℝ := 75)

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  (percent_increase initial_a final_a + percent_increase initial_b final_b + percent_increase initial_c final_c) / 3

theorem combined_percent_increase_proof :
  combined_percent_increase initial_stock_A_price initial_stock_B_price initial_stock_C_price
                            final_stock_A_price final_stock_B_price final_stock_C_price = 16.04 := by
  sorry

end combined_percent_increase_proof_l706_706083


namespace simplify_expression_of_triangle_side_lengths_l706_706942

theorem simplify_expression_of_triangle_side_lengths
  (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  |a - b - c| - |c - a + b| = 0 :=
by
  sorry

end simplify_expression_of_triangle_side_lengths_l706_706942


namespace aluminum_foil_thickness_l706_706850

-- Define the variables and constants
variables (d l m w t : ℝ)

-- Define the conditions
def density_condition : Prop := d = m / (l * w * t)
def volume_formula : Prop := t = m / (d * l * w)

-- The theorem to prove
theorem aluminum_foil_thickness (h1 : density_condition d l m w t) : volume_formula d l m w t :=
sorry

end aluminum_foil_thickness_l706_706850


namespace min_value_pq_l706_706146

theorem min_value_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h1 : p^2 - 8 * q ≥ 0)
  (h2 : 4 * q^2 - 4 * p ≥ 0) :
  p + q ≥ 6 :=
sorry

end min_value_pq_l706_706146


namespace train_crossing_time_l706_706459

variable (train_speed_km_hr : ℕ) (train_length_meters : ℕ)

def convert_speed (speed_km_hr : ℕ) : ℝ :=
  speed_km_hr * 1000 / 3600

def time_to_cross (speed_m_s : ℝ) (length_m : ℕ) : ℝ :=
  length_m / speed_m_s

theorem train_crossing_time 
  (h₁ : train_speed_km_hr = 60)
  (h₂ : train_length_meters = 300) :
  time_to_cross (convert_speed train_speed_km_hr) train_length_meters ≈ 18 := by
  sorry

end train_crossing_time_l706_706459


namespace smallest_consecutive_sum_perfect_square_l706_706348

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l706_706348


namespace triangle_area_l706_706311

theorem triangle_area (A P Q : Point)
  (hA : A = (8, 6))
  (hPerpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1)
  (hSumYIntercepts : P.2 + Q.2 = -2) :
  ∃ area : ℝ, area = 70 :=
sorry

end triangle_area_l706_706311


namespace value_of_expression_l706_706265

variable {a b : ℝ}
variables (h1 : ∀ x, 3 * x^2 + 9 * x - 18 = 0 → x = a ∨ x = b)

theorem value_of_expression : (3 * a - 2) * (6 * b - 9) = 27 :=
by
  sorry

end value_of_expression_l706_706265


namespace intersection_of_M_and_N_l706_706964

-- Define the sets M and N
def M : set ℝ := {x | (x + 2) * (x - 2) > 0}
def N : set ℝ := {-3, -2, 2, 3, 4}

-- State the theorem we aim to prove
theorem intersection_of_M_and_N : M ∩ N = {-3, 3, 4} :=
by
  sorry

end intersection_of_M_and_N_l706_706964


namespace find_a_l706_706280

theorem find_a 
  (a b k : ℚ) 
  (h_sum_a : ∑ i in finset.range 9, abs (a - (k + i)) = 294) 
  (h_sum_b : ∑ i in finset.range 9, abs (b - (k + i)) = 1932)
  (h_ab : a + b = 256) 
  : a = 37 / 3 := sorry

end find_a_l706_706280


namespace unique_row_and_column_sums_possible_l706_706674

theorem unique_row_and_column_sums_possible :
  ∃ (table : ℕ × ℕ → ℤ), 
    (∀ i j, table (i, j) ∈ {1, -1, 0}) ∧
    (∀ i1 i2, i1 ≠ i2 → ((finset.range 10).sum (λ j, table (i1, j))) ≠ ((finset.range 10).sum (λ j, table (i2, j)))) ∧
    (∀ j1 j2, j1 ≠ j2 → ((finset.range 10).sum (λ i, table (i, j1))) ≠ ((finset.range 10).sum (λ i, table (i, j2)))) :=
sorry

end unique_row_and_column_sums_possible_l706_706674


namespace range_of_a_strictly_increasing_l706_706191

theorem range_of_a_strictly_increasing :
  (∀ x ∈ (Set.Ici 2 : Set ℝ), (differentiable ℝ (λ x, (a : ℝ) * x + 2) / (x - 1)) → (∀ x, x > 2 → deriv (λ x, (a * x + 2) / (x - 1)) x > 0)) → 
  (a < -2) :=
sorry

end range_of_a_strictly_increasing_l706_706191


namespace max_stones_removable_l706_706988

/-- Definition for "many stones" on the table, given a list of piles with stone counts -/
def many_stones (piles : list ℕ) : Prop :=
  ∃ (indices : finset ℕ), indices.card = 50 ∧
  (∀ i ∈ indices, piles.nth_le i (finset.mem_univ_subtype_indices.mpr i.in)) ≥ (i.val + 1)

/-- The statement of the problem -/
theorem max_stones_removable (piles : list ℕ) (h_length : piles.length = 100) :
  (∀ n ≤ 10000, ∀ removed_index_subset : finset ℕ, removed_index_subset.card = n →
  (monotone_decreasing_remove_indices removed_index_subset piles).length = 100  →
  many_stones (monotone_decreasing_remove_indices removed_index_subset piles)) ↔ n ≤ 5099 :=
begin
  sorry,
end

/-- Helper function to remove stones as described in the problem -/
def monotone_decreasing_remove_indices (indices : finset ℕ) (piles : list ℕ) : list ℕ :=
  let remove_counts := λ i, if i ∈ indices then i else 0 in
  list.map (λ (i : Σ n, n < piles.length), piles.nth_le i.1 i.2 - remove_counts i.1) (list.fin_range ⟨piles.length⟩)

end max_stones_removable_l706_706988


namespace sum_of_sequence_l706_706609

noncomputable def S (x : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range(n), x i

def recurrence_relation (x : ℕ → ℝ) : Prop :=
  x 1 = 2 ∧ ∀ k, 1 ≤ k → x (k + 1) = x k + 3/4 + 1/(4 * k)

theorem sum_of_sequence (x : ℕ → ℝ) (n : ℕ) (h : recurrence_relation x) :
  S x n = 2 * n + (3 * (n * n + n)) / 8 + (Real.log (n - 1) + Real.eulerMascheroni) / 4 := by
  sorry

end sum_of_sequence_l706_706609


namespace farthest_point_from_origin_l706_706394

/-
Given the points:
p1 = (0, 7),
p2 = (2, 4),
p3 = (4, -6),
p4 = (8, 0),
p5 = (-2, -3)

and the condition that each of these points satisfies the inequality: y <= -1/2 * x + 7,
prove that the point p3 = (4, -6) is the farthest from the origin.
-/
theorem farthest_point_from_origin :
  let p1 := (0, 7)
  let p2 := (2, 4)
  let p3 := (4, -6)
  let p4 := (8, 0)
  let p5 := (-2, -3)

  ∀ (x y : ℤ), (x, y) ∈ {p1, p2, p3, p4, p5} → y ≤ - (1 / 2 : ℚ) * x + 7 →

  let distance (p : ℤ × ℤ) := Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

  distance p3 = max (distance p1) (max (distance p2) 
                          (max (distance p3) 
                               (max (distance p4) (distance p5)))) :=
sorry

end farthest_point_from_origin_l706_706394


namespace sum_of_cubes_minus_tripled_product_l706_706975

theorem sum_of_cubes_minus_tripled_product (a b c d : ℝ) 
  (h1 : a + b + c + d = 15)
  (h2 : ab + ac + ad + bc + bd + cd = 40) :
  a^3 + b^3 + c^3 + d^3 - 3 * a * b * c * d = 1695 :=
by
  sorry

end sum_of_cubes_minus_tripled_product_l706_706975


namespace cookie_pans_l706_706206

theorem cookie_pans (T C : ℕ) (h1 : T = 40) (h2 : C = 8) : T / C = 5 :=
by
  rw [h1, h2]
  norm_num

end cookie_pans_l706_706206


namespace composite_expression_l706_706298

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = 6 * 2^(2^(4 * n)) + 1) :=
by
  sorry

end composite_expression_l706_706298


namespace correct_microorganism_dilution_statement_l706_706813

def microorganism_dilution_conditions (A B C D : Prop) : Prop :=
  (A ↔ ∀ (dilutions : ℕ) (n : ℕ), 1000 ≤ dilutions ∧ dilutions ≤ 10000000) ∧
  (B ↔ ∀ (dilutions : ℕ) (actinomycetes : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (C ↔ ∀ (dilutions : ℕ) (fungi : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (D ↔ ∀ (dilutions : ℕ) (bacteria_first_time : ℕ), 10 ≤ dilutions ∧ dilutions ≤ 10000000)

theorem correct_microorganism_dilution_statement (A B C D : Prop)
  (h : microorganism_dilution_conditions A B C D) : D :=
sorry

end correct_microorganism_dilution_statement_l706_706813


namespace part_a_part_b_part_c_part_d_l706_706941

variable (M : Set ℝ)

axiom cond1 : (0 ∈ M ∧ 1 ∈ M)
axiom cond2 : ∀ x y ∈ M, (x - y) ∈ M
axiom cond3 : ∀ x ∈ M, x ≠ 0 → (1 / x) ∈ M

theorem part_a : (1 / 3) ∈ M :=
sorry

theorem part_b : (-1) ∈ M :=
sorry

theorem part_c : ∀ x y ∈ M, (x + y) ∈ M :=
sorry

theorem part_d : ∀ x ∈ M, (x^2) ∈ M :=
sorry

end part_a_part_b_part_c_part_d_l706_706941


namespace problem_statement_l706_706923

-- Conditions
def circle_M (x y : ℝ) : Prop := (x + real.sqrt 3)^2 + y^2 = 16
def passes_through_F (x : ℝ) (y : ℝ) : Prop := x = real.sqrt 3 ∧ y = 0

-- Target Ellipse: Equation of trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Symmetric Points and Length Conditions
def is_symmetric_to_origin (A B : ℝ × ℝ) : Prop := A.1 = -B.1 ∧ A.2 = -B.2
def distance_AC_CB (A C B : ℝ × ℝ) : Prop := real.dist A C = real.dist C B

-- Minimum area of triangle ABC
def min_area_triangle (A B C : ℝ × ℝ) : ℝ := 
  1/2 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Line equations AB
def line_eq (A B : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1

-- Lean Statement
theorem problem_statement :
  (∀ x y, circle_M x y) →
  (∃ N_center, passes_through_F N_center.1 N_center.2 ∧ trajectory_E N_center.1 N_center.2) ∧
  (
    ∀ A B C,
    is_symmetric_to_origin A B →
    distance_AC_CB A C B →
    A ∈ trajectory_E ∧ B ∈ trajectory_E ∧ C ∈ trajectory_E →
    line_eq A B ↔ min_area_triangle A B C = 8 / 5
  ) :=
by sorry

end problem_statement_l706_706923


namespace part1_part2_l706_706588

-- Part (1)
theorem part1 {a b : ℝ^n} (hb : ∥b∥ = 1) (ha : ∥a∥ = 1) (angle_ab : real.angle_between a b = real.pi / 3) : 
  ∥2 • a + b∥ = real.sqrt 7 := 
by sorry

-- Part (2)
theorem part2 {a b : ℝ^n} (ha : ∥a∥ = 2) (angle_ab : real.angle_between a b = real.pi / 3) 
  (perpendicular : (a + b) ⬝ (2 • a - 5 • b) = 0) : ∥b∥ = 1 := 
by sorry

end part1_part2_l706_706588


namespace divide_nuts_equal_l706_706084

-- Define the conditions: sequence of 64 nuts where adjacent differ by 1 gram
def is_valid_sequence (seq : List Int) :=
  seq.length = 64 ∧ (∀ i < 63, (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ + 1) ∨ (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ - 1))

-- Main theorem statement: prove that the sequence can be divided into two groups with equal number of nuts and equal weights
theorem divide_nuts_equal (seq : List Int) (h : is_valid_sequence seq) :
  ∃ (s1 s2 : List Int), s1.length = 32 ∧ s2.length = 32 ∧ (s1.sum = s2.sum) :=
sorry

end divide_nuts_equal_l706_706084


namespace arithmetic_seq_question_l706_706664

theorem arithmetic_seq_question (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := 
sorry

end arithmetic_seq_question_l706_706664


namespace find_average_of_three_angles_l706_706756

noncomputable theory

variables (α β γ b : ℝ)

def average (x y : ℝ) : ℝ := (x + y) / 2
def average_three (x y z : ℝ) : ℝ := (x + y + z) / 3

theorem find_average_of_three_angles
    (h1 : average α β = 105)
    (h2 : 180 - β + γ = α) :
    average_three α β γ = 80 :=
sorry

end find_average_of_three_angles_l706_706756


namespace dot_product_of_vectors_l706_706226

noncomputable def dot_product_eq : Prop :=
  ∀ (AB AC : ℝ) (BAC_deg : ℝ),
  AB = 3 → AC = 4 → BAC_deg = 30 →
  (AB * AC * (Real.cos (BAC_deg * Real.pi / 180))) = 6 * Real.sqrt 3

theorem dot_product_of_vectors :
  dot_product_eq := by
  sorry

end dot_product_of_vectors_l706_706226


namespace pepe_is_4_feet_6_inches_l706_706871

def big_joe_height : ℝ := 8
def ben_taller_by : ℝ := 1
def larry_taller_by : ℝ := 1
def frank_taller_by : ℝ := 1
def pepe_taller_by : ℝ := 0.5

def ben_height : ℝ := big_joe_height - ben_taller_by
def larry_height : ℝ := ben_height - larry_taller_by
def frank_height : ℝ := larry_height - frank_taller_by
def pepe_height : ℝ := frank_height - pepe_taller_by

def pepe_height_feet : ℕ := pepe_height.to_int
def pepe_height_inches : ℕ := ((pepe_height - (pepe_height_feet : ℝ)) * 12).to_int

theorem pepe_is_4_feet_6_inches :
  pepe_height_feet = 4 ∧ pepe_height_inches = 6 :=
by
  sorry

end pepe_is_4_feet_6_inches_l706_706871


namespace max_x_y_l706_706060

-- Define the variables for the elements of the set and their pairwise sums
variables {a b c d e x y : ℕ}

-- Define the condition that S contains the pairwise sums including x and y
def pairwise_sums (S : set ℕ) :=
  S = {201, 345, 278, 369, x, y, 412, 295, 328, 380}

-- The statement that the greatest possible value of x + y is 2257
theorem max_x_y (S : set ℕ) (h : pairwise_sums S) :
  x + y ≤ 2257 :=
begin
  sorry,
end

end max_x_y_l706_706060


namespace machineA_finishing_time_l706_706712

theorem machineA_finishing_time
  (A : ℝ)
  (hA : 0 < A)
  (hB : 0 < 12)
  (hC : 0 < 6)
  (h_total_time : 0 < 2)
  (h_work_done_per_hour : (1 / A) + (1 / 12) + (1 / 6) = 1 / 2) :
  A = 4 := sorry

end machineA_finishing_time_l706_706712


namespace general_equation_line_BC_standard_equation_circumscribed_circle_l706_706163

-- Define points A, B, and C.
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove that the general equation of line BC is x + 1 = 0.
theorem general_equation_line_BC : ∀ (x y : ℝ), y = B.snd ∧ y = C.snd → x + 1 = 0 := by
  intros x y h
  cases h
  rw [B.fst, C.fst]
  sorry

-- Prove the standard equation of the circumscribed circle of △ABC.
theorem standard_equation_circumscribed_circle : 
  ∀ (x y : ℝ), (x + 5/2)^2 + (y - 3/2)^2 = 5/2 := by
  intros x y
  sorry

end general_equation_line_BC_standard_equation_circumscribed_circle_l706_706163


namespace find_m_for_line_transformation_l706_706710

noncomputable def reflect_angle (theta alpha : Real) : Real :=
  2 * alpha - theta

noncomputable def S (theta alpha beta : Real) : Real :=
  reflect_angle (reflect_angle theta alpha) beta

noncomputable def Sn (theta alpha beta : Real) (n : Nat) : Real :=
  (Nat.iterate (S $ theta) n alpha beta)

def smallest_positive_integer_m (theta alpha beta : Real) (l : Real) : Nat :=
  Nat.find (λ m, Sn theta alpha beta m = theta)

theorem find_m_for_line_transformation :
  let l1_angle := Real.pi / 60
  let l2_angle := Real.pi / 45
  let l := Real.atan (1 / 3)
  let m := smallest_positive_integer_m l l1_angle l2_angle l
  m = 360 :=
  by
  let l1_angle := Real.pi / 60
  let l2_angle := Real.pi / 45
  let l := Real.atan (1 / 3)
  let m := smallest_positive_integer_m l l1_angle l2_angle l
  show m = 360
  sorry 

end find_m_for_line_transformation_l706_706710


namespace magnitude_product_l706_706552

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end magnitude_product_l706_706552


namespace triangle_bisector_altitude_intersection_l706_706460

theorem triangle_bisector_altitude_intersection :
  ∀ (A B C : ℝ × ℝ),
  A = (4, 6) →
  B = (-3, 0) →
  C = (2, -3) →
  let AD := λ P : ℝ × ℝ, 5 * P.1 - 3 * P.2 - 2 = 0 in
  let CE := λ P : ℝ × ℝ, 7 * P.1 + 6 * P.2 + 4 = 0 in
  ∃ F : ℝ × ℝ,
  AD F ∧ CE F ∧ F = (0, -2 / 3) := sorry

end triangle_bisector_altitude_intersection_l706_706460


namespace total_distance_correct_l706_706373

noncomputable def total_distance_covered (rA rB rC : ℝ) (revA revB revC : ℕ) : ℝ :=
  let pi := Real.pi
  let circumference (r : ℝ) := 2 * pi * r
  let distance (r : ℝ) (rev : ℕ) := circumference r * rev
  distance rA revA + distance rB revB + distance rC revC

theorem total_distance_correct :
  total_distance_covered 22.4 35.7 55.9 600 450 375 = 316015.4 :=
by
  sorry

end total_distance_correct_l706_706373


namespace banana_permutations_l706_706104

theorem banana_permutations : 
  let total_letters := 6
      a_count := 3
      n_count := 2
      b_count := 1 in
  (total_letters.factorial / (a_count.factorial * n_count.factorial)) = 60 :=
by
  sorry

end banana_permutations_l706_706104


namespace num_arrangements_l706_706247

theorem num_arrangements (families : ℕ) (cities : ℕ) (choices : ℕ) (h1 : families = 4) (h2 : cities = 3) :
  choices = cities ^ families := by
  rw [h1, h2]
  exact (pow_succ cities 3).rec sorry

#eval num_arrangements

end num_arrangements_l706_706247


namespace correct_conclusions_count_l706_706180

theorem correct_conclusions_count (a : ℕ → ℝ) (S : ℕ → ℝ) (hn_pos : ∀ n, 0 < a n) (hn_sum : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (mut_rule : ∀ n, a n * S n = 9) :
  (if (a 1 = 3) -- Since a1 = sqrt(9) = 3
      ∧ (let a2 := 3 * (real.sqrt 5 - 1) / 2 in a2 < 3) -- Checking the second term is indeed less than 3
      ∧ (∃ n, a n < 1 / 100) -- There exists a term less than 1/100
      ∧ (∀ i j, i < j → a i ≥ a j) -- Sequence is decreasing
     then 3
     else 0) = 3 := sorry

end correct_conclusions_count_l706_706180


namespace proof_problem_l706_706479

theorem proof_problem :
  (\(sqrt(2023) - 1) ^ 0 + (1 / 2) ^ (-1) = 3 :=
by
  sorry

end proof_problem_l706_706479


namespace max_int_difference_l706_706408

theorem max_int_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) : 
  y - x = 5 :=
sorry

end max_int_difference_l706_706408


namespace mass_percentage_H_in_CaH₂_l706_706125

def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_H : ℝ := 1.008
def molar_mass_CaH₂ : ℝ := atomic_mass_Ca + 2 * atomic_mass_H

theorem mass_percentage_H_in_CaH₂ :
  (2 * atomic_mass_H / molar_mass_CaH₂) * 100 = 4.79 := 
by
  -- Skipping the detailed proof for now
  sorry

end mass_percentage_H_in_CaH₂_l706_706125


namespace simplest_quadratic_radical_l706_706811

theorem simplest_quadratic_radical :
  (∀ x : ℝ, (x ≠ 6 → (x ≠ 12 ∨ 2 * real.sqrt 3 < real.sqrt 6) ∧ (x ≠ 0.3 ∨ (real.sqrt 3 / (2 * real.sqrt 5) < real.sqrt 6)) ∧ (x ≠ 1/2 ∨ ((1 / 2) * real.sqrt 2 < real.sqrt 6)))) →
  (∀ y : ℝ, (y = 6) → (y = 6)) :=
begin
  sorry
end

end simplest_quadratic_radical_l706_706811


namespace tangent_line_smallest_slope_l706_706074

noncomputable def curve : ℝ → ℝ := λ x, x^3 - 3 * x + 1

theorem tangent_line_smallest_slope :
  let slope := deriv curve 0,
      point := (0, curve 0)
  in slope = -3 ∧ point = (0, 1) ∧
     (∀ z : ℝ, (curve z - curve 0) = slope * (z - 0) ↔ (z = 0)) :=
by
  -- Proof goes here, containing the calculations of the slope,
  -- the point of tangency, and the equation of the tangent line.
  sorry

end tangent_line_smallest_slope_l706_706074


namespace tan_A_in_triangle_ABC_l706_706641

theorem tan_A_in_triangle_ABC 
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a = b * (1 + sqrt 3 * c / (b * a)))
  (h2 : sin C = 2 * sqrt 3 * sin B)
  (h3 : sin A = a / (2 * sqrt 3 * b)) 
  :
  tan A = sqrt 3 / 3 := 
  sorry

end tan_A_in_triangle_ABC_l706_706641


namespace isosceles_triangle_perimeter_l706_706931

-- Define the isosceles triangle sides and conditions
structure IsoscelesTriangle (a b : ℕ) :=
  (is_isosceles : a = b ∨ a = c)

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : IsoscelesTriangle a b c) (h_sides : {a, b, c} = {2, 4, 4} ∨ {a, b, c} = {4, 4, 2}) : a + b + c = 10 := by
  sorry

end isosceles_triangle_perimeter_l706_706931


namespace line_intersection_y_axis_l706_706475

theorem line_intersection_y_axis :
  ∃ y : ℝ, let x1 := 2, y1 := 5, x2 := 6, y2 := 17 in
           (∀ x : ℝ, (y2 - y1) * (x - x1) = (x2 - x1) * (y - y1)) ∧
           (∀ x, x = 0 → y = -1) :=
sorry

end line_intersection_y_axis_l706_706475


namespace find_interest_rate_l706_706042

variables (P1 P2 R1 R2 T total_interest interest1 interest2 : ℝ)

-- Conditions
def conditions := 
  P1 = 1000 ∧  -- Principal 1
  P2 = 1400 ∧  -- Principal 2
  R1 = 3/100 ∧ -- Rate 1 as a fraction
  T = 3.5 ∧    -- Time period in years
  total_interest = 350 ∧
  interest1 = P1 * R1 * T ∧ -- Interest from principal 1
  interest2 = total_interest - interest1 -- Interest from principal 2

-- Theorem to prove
theorem find_interest_rate (h : conditions) : 
  (interest2 = P2 * (R2/100) * T) → R2 = 5 :=
sorry

end find_interest_rate_l706_706042


namespace solve_eq_in_integers_l706_706307

theorem solve_eq_in_integers :
  ∃ (x y : ℤ), (x ^ 2 - 5 * y ^ 2 = 1) ↔
    ∃ (n : ℕ), (x, y) = (9 + 4 * √5)^n ∨ (x, y) = -(9 + 4 * √5)^n :=
sorry

end solve_eq_in_integers_l706_706307


namespace count_valid_n_l706_706621

theorem count_valid_n :
  {n : ℕ // n > 0 ∧ (⌊(n + 2000) / 60⌋ = ⌊sqrt n⌋)}.card = 6 :=
by
  sorry

end count_valid_n_l706_706621


namespace minimum_value_of_f_l706_706954

noncomputable def f (a x : ℝ) := a * x - Real.log x

theorem minimum_value_of_f (a : ℝ) (e : ℝ) (h_e : e = Real.exp 1) 
  (h_min : ∃ x ∈ Ioo 0 e, f a x = 3) : a = Real.exp 2 := 
by
  sorry

end minimum_value_of_f_l706_706954


namespace proportion_of_derangements_l706_706404

-- Definition of derangements for n numbers
noncomputable def derangements (n : ℕ) : ℕ :=
  let factorial := ∏ i in finset.range (n + 1), if i > 0 then i else 1
  factorial * (finset.range (n + 1)).sum (λ k, (-1 : ℕ)^k / (∏ i in finset.range (k + 1), if i > 0 then i else 1))

-- The proportion of derangements tends to 1/e as n approaches infinity
theorem proportion_of_derangements (n : ℕ) : tendsto (λ n, (derangements n) / (∏ i in finset.range (n + 1), if i > 0 then i else 1 : ℝ)) at_top (𝓝 (1 / real.exp 1)) :=
  sorry

end proportion_of_derangements_l706_706404


namespace larger_factor_of_lcm_l706_706325

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm X Y : ℕ) 
  (h_hcf: hcf = 63)
  (h_A: A = 1071)
  (h_lcm: lcm = hcf * X * Y)
  (h_X: X = 11)
  (h_factors: ∃ k: ℕ, A = hcf * k ∧ lcm = A * (B / k)):
  Y = 17 := 
by sorry

end larger_factor_of_lcm_l706_706325


namespace num_nonempty_proper_subsets_range_of_m_l706_706708

/-- Let A be the set {x | 1/32 ≤ 2^(-x) ≤ 4 with x in ℤ. The number of non-empty proper subsets of A is 254. -/
theorem num_nonempty_proper_subsets (A : Set ℤ) (hA : A = {x : ℤ | (1 / 32 : ℚ) ≤ 2^(-(x : ℚ)) ∧ 2^(-(x : ℚ)) ≤ 4}) : 
  ∃ S : ℕ, S = 254 :=
by
  sorry

/-- Let A be the set {x | 1/32 ≤ 2^(-x) ≤ 4, and B be the set {x | x^2 - 3mx + 2m^2 - m - 1 < 0}. 
Given A ⊇ B, the range of m is -1 ≤ m ≤ 2. -/
theorem range_of_m (A B : Set ℚ) (m : ℚ) 
  (hA : A = {x : ℚ | (1 / 32 : ℚ) ≤ 2^(-(x)) ∧ 2^(-(x)) ≤ 4}) 
  (hB : B = {x : ℚ | (x^2 - 3*x*m + 2*m^2 - m - 1 < 0)}) 
  (hAB : A ⊇ B) : 
  -1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end num_nonempty_proper_subsets_range_of_m_l706_706708


namespace smallest_consecutive_sum_perfect_square_l706_706347

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l706_706347


namespace students_above_620_approx_l706_706058

noncomputable def num_students_above_620 (total_students: ℕ) (mean: ℝ) (variance: ℝ) 
    (prob_0_280: ℝ) : ℕ :=
    let prob_620_750 := prob_0_280 in
    Nat.round (total_students * prob_620_750)

theorem students_above_620_approx {total_students: ℕ} {mean: ℝ} {variance: ℝ}
    {prob_0_280: ℝ}
    (h1: total_students = 1400)
    (h2: mean = 450)
    (h3: variance = 130 ^ 2)
    (h4: prob_0_280 = 0.107) :
  num_students_above_620 total_students mean variance prob_0_280 = 150 :=
by
  simp [num_students_above_620, h1, h4]
  sorry  -- Providing the actual proof is not necessary

end students_above_620_approx_l706_706058


namespace decagon_intersection_possible_hendecagon_intersection_impossible_l706_706012

-- Define the conditions for part (a): Decagon
def decagon := {sides : Nat // sides = 10}
def circle_intersection_decagon (d : decagon) : Prop := 
  ∃ (O : Point), ∀ (i : Fin d.sides), ∃ (circle : Circle), 
  (circle.center = O ∧ circle.diameter = side_length d i)

-- Define the conditions for part (b): Hendecagon
def hendecagon := {sides : Nat // sides = 11}
def circle_intersection_hendecagon (h : hendecagon) : Prop := 
  ∃ (O : Point), ∀ (i : Fin h.sides), ∃ (circle : Circle),
  (circle.center = O ∧ circle.diameter = side_length h i)

-- Prove/Disprove statements
theorem decagon_intersection_possible : ∀ (d : decagon), circle_intersection_decagon d :=
sorry

theorem hendecagon_intersection_impossible : ∀ (h : hendecagon), ¬ circle_intersection_hendecagon h :=
sorry

end decagon_intersection_possible_hendecagon_intersection_impossible_l706_706012


namespace sophomores_sampled_correct_l706_706442

def stratified_sampling_sophomores (total_students num_sophomores sample_size : ℕ) : ℕ :=
  (num_sophomores * sample_size) / total_students

theorem sophomores_sampled_correct :
  stratified_sampling_sophomores 4500 1500 600 = 200 :=
by
  sorry

end sophomores_sampled_correct_l706_706442


namespace iodine_solution_problem_l706_706214

theorem iodine_solution_problem (init_concentration : Option ℝ) (init_volume : ℝ)
  (final_concentration : ℝ) (added_volume : ℝ) : 
  init_concentration = none 
  → ∃ x : ℝ, init_volume + added_volume = x :=
by
  sorry

end iodine_solution_problem_l706_706214


namespace iterative_average_difference_l706_706077

def iterative_average (l : List ℚ) : ℚ :=
  l.tail.foldl (λ acc x, (acc + x) / 2) l.head

theorem iterative_average_difference :
  let nums := [2, 4, 6, 8, 10, 12]
  let max_avg := iterative_average [2, 4, 6, 8, 10, 12] -- Ordering that gives iter_avg 10.0625
  let min_avg := iterative_average [10, 8, 6, 4, 2, 12] -- Ordering that gives iter_avg 7.9375
  max_avg - min_avg = 2.125 :=
by 
  let nums := [2, 4, 6, 8, 10, 12]
  let max_avg := iterative_average [2, 4, 6, 8, 10, 12]
  let min_avg := iterative_average [10, 8, 6, 4, 2, 12]
  have h : max_avg = 10.0625 := sorry
  have h' : min_avg = 7.9375 := sorry
  rw [h, h']
  norm_num

end iterative_average_difference_l706_706077


namespace area_triangle_COB_l706_706724

-- Define the points C, O, and B
variable (p : ℝ) (C O B : ℝ × ℝ)
variable hC : C = (0, p)
variable hO : O = (0, 0)
variable hB : B = (12, 0)

-- Prove the area of triangle COB is 6 * p
theorem area_triangle_COB : 
  let CO := abs (p - 0)
  let OB := abs (12 - 0)
  (1 / 2) * OB * CO = 6 * p := 
by
  sorry

end area_triangle_COB_l706_706724


namespace ratio_proof_l706_706974

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : (a + 2 * b) / (3 * b + c) = 9 / 32 :=
by
  sorry

end ratio_proof_l706_706974


namespace John_makes_72_dollars_l706_706255

def num_jars : ℕ := 4
def caterpillars_per_jar : ℕ := 10
def success_rate : ℝ := 0.60
def price_per_butterfly : ℝ := 3

theorem John_makes_72_dollars :
  let total_caterpillars := num_jars * caterpillars_per_jar in
  let butterflies := total_caterpillars * success_rate in
  let total_income := butterflies * price_per_butterfly in
  total_income = 72 := 
by 
  sorry

end John_makes_72_dollars_l706_706255


namespace brother_books_total_l706_706865

theorem brother_books_total (pb_sarah hb_sarah : ℕ) (h_pb_sarah : pb_sarah = 6) (h_hb_sarah : hb_sarah = 4) : 
  let pb_brother := pb_sarah / 3 in
  let hb_brother := 2 * hb_sarah in
  pb_brother + hb_brother = 10 :=
by
  have h_pb_brother : pb_brother = 2 := by rw [h_pb_sarah] ; exact Nat.div_eq_of_lt (by decide) -- 6 / 3 = 2
  have h_hb_brother : hb_brother = 8 := by rw [h_hb_sarah] ; exact by norm_num -- 4 * 2 = 8
  rw [h_pb_brother, h_hb_brother]
  norm_num  -- 2 + 8 = 10
  sorry

end brother_books_total_l706_706865


namespace avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l706_706541

/-

Problem 1:
Given:
- Visitors in 2022: 200000
- Visitors in 2024: 288000

Prove:
- The average annual growth rate of visitors from 2022 to 2024 is 20% 

Problem 2:
Given:
- Cost price per cup: 6 yuan
- Selling price per cup at 25 yuan leads to 300 cups sold per day.
- Each 1 yuan reduction leads to 30 more cups sold per day.
- Desired daily profit in 2024: 6300 yuan

Prove:
- The selling price per cup for maximum discount and desired profit is 20 yuan.

-/

-- Definitions for Problem 1
def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000

-- Definition for annual growth rate
def annual_growth_rate (P Q : ℕ) (y : ℕ) : ℝ :=
  ((Q.to_real / P.to_real) ^ (1 / y.to_real)) - 1

def expected_growth_rate := annual_growth_rate visitors_2022 visitors_2024 2

-- Statement for the first proof
theorem avg_annual_growth_rate_is_20_percent : expected_growth_rate = 0.2 := sorry

-- Definitions for Problem 2
def cost_price_per_cup : ℕ := 6
def initial_price_per_cup : ℕ := 25
def initial_sales_per_day : ℕ := 300
def additional_sales_per_price_reduction : ℕ := 30
def desired_daily_profit : ℕ := 6300

-- Profit function
def daily_profit (price : ℕ) : ℕ := (price - cost_price_per_cup) * (initial_sales_per_day + additional_sales_per_price_reduction * (initial_price_per_cup - price))

-- Statement for the second proof
theorem optimal_selling_price_for_max_discount : (∃ (price : ℕ), daily_profit price = desired_daily_profit ∧ price = 20) := sorry

end avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l706_706541


namespace sum_of_solutions_eq_neg_six_l706_706107

theorem sum_of_solutions_eq_neg_six (x r s : ℝ) :
  (81 : ℝ) - 18 * x - 3 * x^2 = 0 →
  (r + s = -6) :=
by
  sorry

end sum_of_solutions_eq_neg_six_l706_706107


namespace find_certain_number_l706_706432

theorem find_certain_number :
  let a := 3 * 15,
      b := 3 * 16,
      c := 3 * 19 in
  ∃ x : ℤ, a + b + c + x = 161 ∧ x = 11 :=
by
  sorry

end find_certain_number_l706_706432


namespace range_of_x_for_positive_y_l706_706956

theorem range_of_x_for_positive_y (x : ℝ) : 
  (-1 < x ∧ x < 3) ↔ (-x^2 + 2*x + 3 > 0) :=
sorry

end range_of_x_for_positive_y_l706_706956


namespace question_l706_706825
-- Importing necessary libraries

-- Stating the problem
theorem question (x : ℤ) (h : (x + 12) / 8 = 9) : 35 - (x / 2) = 5 :=
by {
  sorry
}

end question_l706_706825


namespace pascals_triangle_first_25_rows_count_l706_706619

theorem pascals_triangle_first_25_rows_count :
  (∑ n in Finset.range 25, n + 1) = 325 := 
sorry

end pascals_triangle_first_25_rows_count_l706_706619


namespace car_graph_representation_l706_706093

theorem car_graph_representation (v t : ℝ) :
  ∃ (M_graph N_graph : ℝ × ℝ → Prop),
    (∀ x, M_graph (x, v) ↔ x ≤ t) ∧
    (∀ x, N_graph (x, 2 * v) ↔ x ≤ t / 2) :=
begin
  sorry
end

end car_graph_representation_l706_706093


namespace polynomials_equal_l706_706920

noncomputable def polynomial.degree_le {R : Type*} [CommRing R] (f : R[X]) 
  (n : ℕ) : Prop := f.degree ≤ n

theorem polynomials_equal 
  (f g : ℝ[X]) (n : ℕ) 
  (hf : polynomial.degree_le f n) 
  (hg : polynomial.degree_le g n) 
  (h_eq : ∃ (x : ℝ) → (f - g).roots_nodup x ≥ n + 1) : 
  f = g :=
sorry

end polynomials_equal_l706_706920


namespace average_production_before_today_l706_706137

theorem average_production_before_today (n : ℕ) (A : ℕ) (total_even_days : ℕ) 
  (total_production_with_today : ℕ) (new_average : ℕ) (today_production : ℕ) 
  (day_count : ℕ) 
  (h1: n = 9) 
  (h2: today_production = 90) 
  (h3: new_average = 45) 
  (h4: day_count = n + 1) 
  (h5: total_even_days = n * A) 
  (h6: total_production_with_today = total_even_days + today_production) 
  (h7: total_production_with_today = (n + 1) * new_average) : 
  A = 40 :=
by {
  intro,
  sorry
}

end average_production_before_today_l706_706137


namespace solve_for_x_l706_706926

-- Define the condition that the equation holds for some real number x
def equation_holds (x : ℝ) : Prop := 5^(x-1) * 10^(3 * x) = 8^x

-- Theorem stating that if the equation holds for some x in ℝ, then x must be 1/4
theorem solve_for_x (x : ℝ) (h : equation_holds x) : x = 1/4 :=
sorry

end solve_for_x_l706_706926


namespace count_valid_two_digit_numbers_l706_706623

theorem count_valid_two_digit_numbers :
  let digits := {d : ℕ | 1 ≤ d ∧ d ≤ 9}
  let two_digit_numbers_with_condition := 
    {n : ℕ × ℕ | n.snd ∈ digits ∧ n.fst ∈ digits ∧ n.snd ≠ n.fst ∧ (n.snd = 2 * n.fst ∨ n.fst = 2 * n.snd)}
  (two_digit_numbers_with_condition.card = 8) :=
begin
  sorry
end

end count_valid_two_digit_numbers_l706_706623


namespace dessert_menus_count_l706_706832

-- Define the dessert options
inductive Dessert
| cake
| pie
| ice_cream
| pudding

-- Define the conditions
def valid_menu : List Dessert → Prop :=
  λ menu, menu.length = 7 ∧
          ∀ i < 6, menu.get i ≠ menu.get (i + 1) ∧
          menu.get 4 = Dessert.cake

-- The proof problem statement
theorem dessert_menus_count : 
  ∃ menu_count : ℕ, menu_count = 729 ∧
  (∃ menus : List (List Dessert),
     ∀ menu ∈ menus, valid_menu menu) :=
begin
  use 729,
  sorry
end

end dessert_menus_count_l706_706832


namespace findA_l706_706948

variable (x y a : ℝ)

-- Definitions of circle and line are given as conditions
def circle (x y a : ℝ) : Prop := x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0
def line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

-- Point A is on the circle
def pointOnCircle (A : ℝ × ℝ) : Prop := circle A.1 A.2 a

-- Definition of the symmetric point with respect to the line
def symmetricPoint (A : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let (x, y) := A in
  let (u, v) : ℝ × ℝ := if L (a', b') then (x - 2*(x + 2*y + 1)/(1 + 4) * 1,
                                              y - 2*(x + 2*y + 1)/(1 + 4) * 2) else (x, y)
  in (u, v)

-- Symmetric point of A with respect to the line is also on the circle
def symmetricPointOnCircle (A : ℝ × ℝ) : Prop :=
  let symA := symmetricPoint A line in
  circle symA.1 symA.2 a

-- Proving the value of a
theorem findA (A : ℝ × ℝ) : pointOnCircle A → symmetricPointOnCircle A → a = 3 :=
by
  intro h1 h2
  sorry

end findA_l706_706948


namespace no_such_fraction_exists_l706_706506

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l706_706506


namespace solution_set_l706_706943

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - 2 else 2^(-x) - 2

theorem solution_set (x : ℝ) (h1 : ∀ x, f x = f (-x)) (h2 : ∀ x ≥ 0, f x = 2^x - 2)
  (h3 : ∀ x y, x ≥ 0 → y ≥ 0 → x < y → f x < f y) : 
  f (x - 1) ≤ 6 ↔ -2 ≤ x ∧ x ≤ 4 :=
sorry

end solution_set_l706_706943


namespace ab_value_l706_706727

theorem ab_value (a b : ℝ) (h1 : 2^a = 64^(b + 1)) (h2 : 216^b = 6^(a - 4)) : a * b = 56 / 3 :=
by
  sorry

end ab_value_l706_706727


namespace range_of_k_l706_706572

def f : ℝ → ℝ := sorry

axiom cond1 (a b : ℝ) : f (a + b) = f a + f b + 2 * a * b
axiom cond2 (k : ℝ) : ∀ x : ℝ, f (x + k) = f (k - x)
axiom cond3 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y

theorem range_of_k (k : ℝ) : k ≤ 1 :=
sorry

end range_of_k_l706_706572


namespace find_certain_number_l706_706011

def digits : List ℕ := [1, 3, 5, 8]

def smallest_number : ℕ := 13 -- formed by the smallest 2 digits from 1, 3, 5, 8

theorem find_certain_number : ∃ (X : ℕ), (smallest_number + X = 88) ∧ X = 75 :=
by
  let X := 75
  use X
  split
  - exact (show smallest_number + X = 88 by rfl)
  - rfl

end find_certain_number_l706_706011


namespace avg_GPA_is_93_l706_706755

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l706_706755


namespace point_on_parabola_dist_3_from_focus_l706_706154

def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 4 * p.fst

def focus : ℝ × ℝ := (1, 0)

theorem point_on_parabola_dist_3_from_focus :
  ∃ y: ℝ, ∃ x: ℝ, (parabola (x, y) ∧ (x = 2) ∧ (y = 2 * Real.sqrt 2 ∨ y = -2 * Real.sqrt 2) ∧ (Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 3)) :=
by
  sorry

end point_on_parabola_dist_3_from_focus_l706_706154


namespace horizontal_asymptote_l706_706098

theorem horizontal_asymptote :
  let f := λ x : ℝ, (16 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2) / (4 * x^4 + x^3 + 5 * x^2 + 2 * x + 1)
  ∃ L : ℝ, isHorizontalAsymptote (f) L :=
  ∃ L : ℝ, L = 4.

end horizontal_asymptote_l706_706098


namespace total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l706_706374

-- Define the conditions
def total_people (A B : ℕ) : Prop := A + B = 92
def valid_class_A (A : ℕ) : Prop := 51 < A ∧ A < 55
def total_cost (sets : ℕ) (cost_per_set : ℕ) : ℕ := sets * cost_per_set

-- Prices per set for different ranges of number of sets
def price_per_set (n : ℕ) : ℕ :=
  if n > 90 then 30 else if n > 50 then 40 else 50

-- Question 1
theorem total_amount_for_uniforms (A B : ℕ) (h1 : total_people A B) : total_cost 92 30 = 2760 := sorry

-- Question 2
theorem students_in_classes (A B : ℕ) (h1 : total_people A B) (h2 : valid_class_A A) (h3 : 40 * A + 50 * B = 4080) : A = 52 ∧ B = 40 := sorry

-- Question 3
theorem cost_effective_purchase_plan (A : ℕ) (h1 : 51 < A ∧ A < 55) (B : ℕ) (h2 : 92 - A = B) (h3 : A - 8 + B = 91) :
  ∃ (cost : ℕ), cost = total_cost 91 30 ∧ cost = 2730 := sorry

end total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l706_706374


namespace math_problem_equivalent_l706_706860

-- Proposition 1: correlation coefficient r
def prop1 (r : ℝ) : Prop := (r = 0) → (∀ c, c < 1 → linear_dependence := c)
-- Proposition 2: negation of an existential quantifier
def prop2 : Prop := (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)
-- Proposition 3: logical proposition
def prop3 (p q : Prop) : Prop := (p ∧ q) → (p ∨ q)
-- Proposition 4: a function having an extreme value
def prop4 (a b : ℝ) : Prop :=
  let f := (λ x : ℝ, x^3 + 3 * a * x^2 + b * x + a^2) in
  let f' := (λ x : ℝ, 3 * x^2 + 6 * a * x + b) in
  (∃ x : ℝ, f' x = 0) → (a = 2 ∧ b = 9)

-- The main theorem that encapsulates the equivalence to the given problem
theorem math_problem_equivalent :
  let num_correct := (if prop1 0 then 1 else 0) +
                     (if prop2 then 1 else 0) +
                     (if ∀ (p q : Prop), prop3 p q then 1 else 0) +
                     (if ∀ a b : ℝ, prop4 a b then 1 else 0)
  in num_correct = 0 :=
by
  sorry

end math_problem_equivalent_l706_706860


namespace cost_of_mixture_l706_706672

theorem cost_of_mixture (C1 C2 R Cm : ℝ) (hC1 : C1 = 5.5) (hC2 : C2 = 8.75) (hR : R = 0.625) :
  Cm = 7.5 :=
by
  -- let the ratio be R = Q1 / Q2 = 5 / 8
  let Q1 := 5
  let Q2 := 8
  
  -- calculate the mixture cost
  let x := 1  -- assume x as an arbitrary positive volume
  
  have hQ1 : Q1 = 5 * x := by sorry
  have hQ2 : Q2 = 8 * x := by sorry
  
  have hQ1AndQ2 : Q1 = 5 ∧ Q2 = 8 := by
    rw [hQ1, hQ2]
    exact ⟨rfl, rfl⟩
  
  -- substitute Q1 and Q2 back into the weighted average formula
  have hCm := (C1 * Q1 + C2 * Q2) / (Q1 + Q2)
  
  -- Verify Cm is 7.5
  have : Cm = 7.5 := by 
    rw [hCm, hC1, hC2, hQ1AndQ2]
    sorry

end cost_of_mixture_l706_706672


namespace smallest_sum_of_consecutive_integers_is_square_l706_706352

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l706_706352


namespace dayAfter73DaysFromFridayAnd9WeeksLater_l706_706383

-- Define the days of the week as a data type
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Function to calculate the day of the week after a given number of days
def addDays (start_day : Weekday) (days : ℕ) : Weekday :=
  match start_day with
  | Sunday    => match days % 7 with | 0 => Sunday    | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday | _ => Sunday
  | Monday    => match days % 7 with | 0 => Monday    | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday | _ => Monday
  | Tuesday   => match days % 7 with | 0 => Tuesday   | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday | _ => Tuesday
  | Wednesday => match days % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday | _ => Wednesday
  | Thursday  => match days % 7 with | 0 => Thursday  | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday | _ => Thursday
  | Friday    => match days % 7 with | 0 => Friday    | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday | _ => Friday
  | Saturday  => match days % 7 with | 0 => Saturday  | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday | _ => Saturday

-- Theorem that proves the required solution
theorem dayAfter73DaysFromFridayAnd9WeeksLater : addDays Friday 73 = Monday ∧ addDays Monday (9 * 7) = Monday := 
by
  -- Placeholder to acknowledge proof requirements
  sorry

end dayAfter73DaysFromFridayAnd9WeeksLater_l706_706383


namespace multiple_of_first_number_l706_706331

theorem multiple_of_first_number (F S M : ℕ) (hF : F = 15) (hS : S = 55) (h_relation : S = M * F + 10) : M = 3 :=
by
  -- We are given that F = 15, S = 55 and the relation S = M * F + 10
  -- We need to prove that M = 3
  sorry

end multiple_of_first_number_l706_706331


namespace abs_not_always_eq_self_l706_706472

theorem abs_not_always_eq_self : ∃ (a : ℝ), |a| ≠ a :=
by
  use -2023
  have h : |(-2023 : ℝ)| = 2023 := abs_of_neg (by norm_num : (-2023 : ℝ) < 0)
  have h' : (-2023 : ℝ) ≠ 2023 := by norm_num
  rw h
  exact h'

end abs_not_always_eq_self_l706_706472


namespace percentage_error_is_correct_l706_706673

-- Definition of the conditions and question
def original_number (N : ℝ) := N

-- Correct result of multiplying the number by 5
def correct_result (N : ℝ) := N * 5

-- Incorrect result of dividing the number by 10
def incorrect_result (N : ℝ) := N / 10

-- Absolute error between the correct and incorrect results
def absolute_error (N : ℝ) := abs (correct_result N - incorrect_result N)

-- Percentage error calculation
def percentage_error (N : ℝ) := (absolute_error N / correct_result N) * 100

-- Main theorem statement
theorem percentage_error_is_correct (N : ℝ) : percentage_error N = 98 := by
  sorry

end percentage_error_is_correct_l706_706673


namespace question_solution_l706_706591

theorem question_solution
  (f : ℝ → ℝ)
  (h_decreasing : ∀ ⦃x y : ℝ⦄, -3 < x ∧ x < 0 → -3 < y ∧ y < 0 → x < y → f y < f x)
  (h_symmetry : ∀ x : ℝ, f (x) = f (-x + 6)) :
  f (-5) < f (-3/2) ∧ f (-3/2) < f (-7/2) :=
sorry

end question_solution_l706_706591


namespace bisect_perimeter_and_parallel_angle_bisector_l706_706079

-- Definition of the geometric context
variables {A B C D E M N : Point}
variables (Δ : Triangle A B C)  -- Triangle ABC
variables (MD : Midpoint D E M) -- M is the midpoint of D and E
variables (NB : Midpoint B C N) -- N is the midpoint of B and C

-- Conditions of tangent excircles and midpoints
variables (HD : TangentExcircle Δ B CA D) -- Excircle at B tangent to CA at D
variables (HE : TangentExcircle Δ C AB E) -- Excircle at C tangent to AB at E

-- Proof goal
theorem bisect_perimeter_and_parallel_angle_bisector :
  BisectionPerimeter Δ M N ∧ ParallelAngleBisector Δ A M N :=
by
  sorry

end bisect_perimeter_and_parallel_angle_bisector_l706_706079


namespace johns_profit_l706_706682

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l706_706682


namespace correct_options_l706_706939

-- Define the set M and the conditions
def M (x : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∨
  (∃ (y z : ℝ), M y ∧ M z ∧ x = y - z) ∨
  (∃ (y : ℝ), y ≠ 0 ∧ M y ∧ x = 1 / y)

-- Theorem statement
theorem correct_options (M_set : set ℝ) (h1 : 0 ∈ M_set) (h2 : 1 ∈ M_set)
  (h3 : ∀ x y, x ∈ M_set ∧ y ∈ M_set → (x - y) ∈ M_set)
  (h4 : ∀ x, x ∈ M_set ∧ x ≠ 0 → (1 / x) ∈ M_set) :
  (1 / 3 ∈ M_set) ∧ (-1 ∈ M_set) ∧
  (∀ x y, x ∈ M_set ∧ y ∈ M_set → (x + y) ∈ M_set) ∧
  (∀ x, x ∈ M_set → (x ^ 2) ∈ M_set) :=
by
  -- Placeholder for the proof
  sorry

end correct_options_l706_706939


namespace range_a_extreme_point_l706_706632

noncomputable def f (x a : ℝ) := (x^3) / 3 - (a / 2) * x^2 + x + 1

noncomputable def f' (x a : ℝ) := x^2 - a * x + 1

theorem range_a_extreme_point :
  (∃ a : ℝ, (∀ x : ℝ, f' x a = 0 → x ∈ set.Ioo 0.5 3) ∧ (∀ x : ℝ, f' x a ≠ 0 → (∃ y : ℝ, f y a = 0)) ∧ ∃ a : ℝ, 2 < a ∧ a < 10 / 3) :=
sorry

end range_a_extreme_point_l706_706632


namespace smallest_sum_of_consecutive_integers_is_square_l706_706334

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l706_706334


namespace minimum_distance_is_correct_l706_706774

noncomputable def minimum_distance_AB : ℝ :=
  let f : ℝ → ℝ := λ x, Real.exp x
  let g : ℝ → ℝ := λ x, 2 * x
  Real.abs (f (real.log 2) - g (real.log 2))

theorem minimum_distance_is_correct : minimum_distance_AB = 2 - 2 * real.log 2 := by
  sorry

end minimum_distance_is_correct_l706_706774


namespace symmetric_and_distance_l706_706164

def pointA : ℝ × ℝ × ℝ := (-3, 1, 4)
def symmetricPoint (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, -p.2, -p.3)
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

theorem symmetric_and_distance :
    let B := symmetricPoint pointA
    B = (3, -1, -4) ∧ distance pointA B = 2 * Real.sqrt 26 :=
by
  let B := symmetricPoint pointA
  sorry

end symmetric_and_distance_l706_706164


namespace length_of_AB_is_6_l706_706294

theorem length_of_AB_is_6 (A B C D E : Point)
  (collinear : collinear {A, B, C, D})
  (order : A < B ∧ B < C ∧ C < D)
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_15 : dist B C = 15)
  (BE_eq_CE : dist B E = 12 ∧ dist C E = 12)
  (perimeter_condition : 1.5 * (dist B E + dist C E + dist B C)
                         = dist A E + dist E D + dist A D) :
  dist A B = 6 :=
by
  sorry

end length_of_AB_is_6_l706_706294


namespace purely_imaginary_z_imp_m_eq_neg2_second_quadrant_z_imp_m_range_l706_706243

variable (m : ℝ)
def z : ℂ := (m^2 - m - 6) + (m^2 - 2m - 3) * Complex.I

/- Part (1) -/
theorem purely_imaginary_z_imp_m_eq_neg2 :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by
  sorry

/- Part (2) -/
theorem second_quadrant_z_imp_m_range :
  (z m).re < 0 ∧ (z m).im > 0 → -2 < m ∧ m < -1 := by
  sorry

end purely_imaginary_z_imp_m_eq_neg2_second_quadrant_z_imp_m_range_l706_706243


namespace min_value_of_ratio_l706_706603

-- Define the conditions given in the problem
variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_of_ratio :
  (0 < 2 * a ∧ 2 * a < b) →
  (∀ x : ℝ, 0 ≤ f x) →
  (∀ x : ℝ, f x = a * x^2 + b * x + c) →
  (∃ m : ℝ, ∀ x : ℝ, m ≤ (f 1) / (f 0 - f (-1)) ∧ m = 3) :=
begin
  sorry
end

end min_value_of_ratio_l706_706603


namespace monogram_count_l706_706286

theorem monogram_count :
  ∃ (n : ℕ), n = 156 ∧
    (∃ (beforeM : Fin 13) (afterM : Fin 14),
      ∀ (a : Fin 13) (b : Fin 14),
        a < b → (beforeM = a ∧ afterM = b) → n = 12 * 13
    ) :=
by {
  sorry
}

end monogram_count_l706_706286


namespace ladybird_routes_l706_706445

theorem ladybird_routes (X Y : Hexagon) (unshaded : Fin 7 → Hexagon) :
  (adjacent : Hexagon → Hexagon → Prop) → 
  (∀ i j, i ≠ j → ¬ adjacent (unshaded i) (unshaded j)) →
  (∀ i, adjacent X (unshaded i) ∨ adjacent Y (unshaded i)) →
  ∃ (routes : Fin 7 → Hexagon), 
    (∀ i, adjacent (unshaded i) (routes i)) ∧
    routes 0 = X ∧
    routes 6 = Y ∧
    (forall i, adjacent (routes i) (routes (i + 1))) ∧
    (count_routes routes = 5) :=
sorry

end ladybird_routes_l706_706445


namespace number_of_solutions_l706_706967

theorem number_of_solutions :
  (∃ (xs : List ℤ), (∀ x ∈ xs, |3 * x + 4| ≤ 10) ∧ xs.length = 7) := sorry

end number_of_solutions_l706_706967


namespace _l706_706153

noncomputable def construct_triangle_median_circumcircle (A B C A1 : Point)
  (r1 r2 : Real) (h : r2 ≥ r1 ∧ r1 ≥ (distance A A1) / 2) : 
  ∃ (ABC : Triangle), is_median_triangle ABC A A1 ∧ 
                      circumradius (triangle_sub ABC ABA1) = r1 ∧
                      circumradius (triangle_sub ABC ACA1) = r2 := 
sorry

-- Definitions to support the goals mentioned in the problem.
structure Point := (x y : Real)
structure Triangle := (A B C : Point)

def distance (p1 p2 : Point) : Real := 
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

def is_median_triangle (ABC : Triangle) (A A1 : Point) : Prop := 
  -- placeholder for actual median checking logic
  sorry

def triangle_sub (ABC : Triangle) (sub : Triangle) : Triangle := 
  -- placeholder for the sub-triangle extraction logic
  sorry

def circumradius (T : Triangle) : Real := 
  -- placeholder for the calculation of circumradius of a triangle
  sorry

# This Lean 4 code sets up the problem with necessary structures and definitions
# and then states the problem formally as a theorem that needs to be proven.


end _l706_706153


namespace magnitude_product_l706_706551

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end magnitude_product_l706_706551


namespace tom_total_distance_l706_706801

-- Define the conditions
def swimming_time : ℝ := 2
def swimming_speed : ℝ := 2
def running_time := (1 / 2) * swimming_time
def running_speed := 4 * swimming_speed

-- Define the expected total distance
def total_distance := (swimming_speed * swimming_time) + (running_speed * running_time)

-- Prove the total distance covered by Tom is 12 miles
theorem tom_total_distance : total_distance = 12 :=
by
  -- Proof goes here
  sorry

end tom_total_distance_l706_706801


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l706_706342

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l706_706342


namespace no_fractions_meet_condition_l706_706509

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l706_706509


namespace in_circle_implies_range_l706_706177

theorem in_circle_implies_range (a : ℝ) :
  (∃ x y, x = 1 ∧ y = 1 ∧ (x - a)^2 + (y + a)^2 < 4) → (-1 < a ∧ a < 1) :=
by
  intro h
  rcases h with ⟨x, y, hx, hy, hc⟩
  rw [hx, hy] at hc
  simp at hc
  sorry

end in_circle_implies_range_l706_706177


namespace cross_product_magnitude_l706_706596

-- Definition of input conditions
variables (a b : EuclideanSpace ℝ (fin 3)) (θ : ℝ)
noncomputable def norm_a : ℝ := 2
noncomputable def norm_b : ℝ := 5
noncomputable def dot_ab : ℝ := -6
def cross_ab : ℝ := norm a * norm b * real.sin θ

-- Theorem statement
theorem cross_product_magnitude :
  ∥a∥ = norm_a → ∥b∥ = norm_b → inner a b = dot_ab →
  |a × b| = 8 :=
by
  intros h1 h2 h3
  sorry

end cross_product_magnitude_l706_706596


namespace wedge_volume_128pi_l706_706828

/-- A cylindrical log has a diameter of 16 inches. 
A wedge is cut from the log by making two planar cuts through it.
The first cut is perpendicular to the axis of the cylinder.
The second cut's plane forms a 60° angle with the plane of the first cut.
The intersection of these two planes touches the cylinder at exactly one point. 
Calculate the volume of the wedge expressed as mπ, where m is a positive integer, and prove that m = 128. -/
theorem wedge_volume_128pi 
    (diameter : ℝ) (radius : ℝ) (effective_radius : ℝ) (angle : ℝ) 
    (cut1_perpendicular : Prop) (cut2_angle : ℝ) (touch_point : Prop)
    (cylinder_height : ℝ)
    (diameter_eq : diameter = 16)
    (radius_eq : radius = diameter / 2)
    (angle_eq : angle = 60)
    (cos60 : effective_radius = radius * real.cos (angle * real.pi / 180))
    (height_eq : cylinder_height = diameter) :
    ∃ (m : ℕ), m * real.pi = 128 * real.pi :=
by
  use 128
  sorry

end wedge_volume_128pi_l706_706828


namespace isosceles_vertex_angle_l706_706386

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem isosceles_vertex_angle (a b θ : ℝ)
  (h1 : a = golden_ratio * b) :
  ∃ θ, θ = 36 :=
by
  sorry

end isosceles_vertex_angle_l706_706386


namespace wholesale_price_l706_706054

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l706_706054


namespace find_sin_theta_l706_706582

theorem find_sin_theta (θ : ℝ) (h₁ : θ ∈ set.Icc (Real.pi / 4) (Real.pi / 2))
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 := sorry

end find_sin_theta_l706_706582


namespace complete_square_transform_l706_706464

theorem complete_square_transform (x : ℝ) :
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := 
sorry

end complete_square_transform_l706_706464


namespace cost_of_shoes_correct_l706_706853

def budget := 200
def spent_shirt := 30
def spent_pants := 46
def spent_coat := 38
def spent_socks := 11
def spent_belt := 18
def remaining := 16
def cost_shoes := 41
def spent_other_items := spent_shirt + spent_pants + spent_coat + spent_socks + spent_belt

theorem cost_of_shoes_correct :
  cost_shoes + spent_other_items + remaining = budget :=
begin
  -- omitted proof
  sorry
end

end cost_of_shoes_correct_l706_706853


namespace equation_of_ellipse_distance_between_points_l706_706470

/-- Given ellipse properties and eccentricity conditions -/
variable (a b : ℝ) (h1 : a = sqrt 3) (h2 : b = 1)
variable (e : ℝ) (h3 : e = sqrt 6 / 3)
variable (ellipse_eq : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1)

theorem equation_of_ellipse : 
  (∀ x y : ℝ, (x^2) / 3 + (y^2) = 1) :=
sorry

/-- Given the intersection of a line with the ellipse -/
def line_eq (x : ℝ) := x + 1

theorem distance_between_points (x1 y1 x2 y2 : ℝ)
  (h4 : line_eq x1 = y1) (h5 : line_eq x2 = y2)
  (A : x1 = 0) (B : x2 = -3/2) (C : y1 = 1) (D : y2 = -1/2) :
  sqrt ((-3/2 - 0)^2 + (-1/2 - 1)^2) = 3/2 * sqrt 2 :=
sorry

end equation_of_ellipse_distance_between_points_l706_706470


namespace trig_expression_l706_706167

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end trig_expression_l706_706167


namespace problem_statement_l706_706709

-- Defining the sets U, M, and N
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

-- Complement of N in U
def complement_U_N : Set ℕ := U \ N

-- Problem statement
theorem problem_statement : M ∩ complement_U_N = {0, 3} :=
by
  sorry

end problem_statement_l706_706709


namespace f_odd_max_f_l706_706238

noncomputable def f (x : ℝ) : ℝ := ∑ i in finset.range 7, (Real.sin ((2 * (i + 1) - 1) * x) / (2 * (i + 1) - 1))

-- Statement 1: Prove that f(x) is an odd function
theorem f_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  sorry

-- Derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := ∑ i in finset.range 7, (Real.cos ((2 * (i + 1) - 1) * x))

-- Statement 2: Prove that the maximum value of f'(x) is 7
theorem max_f'_val : ∃ x : ℝ, f'(x) = 7 :=
by
  sorry

end f_odd_max_f_l706_706238


namespace coefficient_of_x3_in_expansion_l706_706668

noncomputable def binomial_expansion_coefficient (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem coefficient_of_x3_in_expansion : 
  (∀ k : ℕ, binomial_expansion_coefficient 6 k ≤ binomial_expansion_coefficient 6 3) →
  binomial_expansion_coefficient 6 3 = 20 :=
by
  intro h
  -- skipping the proof
  sorry

end coefficient_of_x3_in_expansion_l706_706668


namespace comm_add_comm_mul_distrib_l706_706116

variable {α : Type*} [AddCommMonoid α] [Mul α] [Distrib α]

theorem comm_add (a b : α) : a + b = b + a :=
by sorry

theorem comm_mul (a b : α) : a * b = b * a :=
by sorry

theorem distrib (a b c : α) : (a + b) * c = a * c + b * c :=
by sorry

end comm_add_comm_mul_distrib_l706_706116


namespace find_integer_pairs_l706_706557

theorem find_integer_pairs :
  ∃ (x y : ℤ), (x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30) ∧ (x^2 + y^2 + 27 = 456 * Int.sqrt (x - y)) :=
by
  sorry

end find_integer_pairs_l706_706557


namespace arrangement_count_l706_706312

-- Define the conditions
def boys : ℕ := 3
def girls : ℕ := 4

-- Define the main problem statement
theorem arrangement_count :
  (∃ (arrangements : ℕ),
   arrangements = boys * (boys - 1) * 2 *
   girls * (girls - 1) * (girls - 2) *
   factorial girls ∧
   arrangements = 432) :=
sorry

end arrangement_count_l706_706312


namespace intersection_of_S_and_T_l706_706613

open Set

def setS : Set ℝ := { x | (x-2)*(x+3) > 0 }
def setT : Set ℝ := { x | 3 - x ≥ 0 }

theorem intersection_of_S_and_T : setS ∩ setT = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_S_and_T_l706_706613


namespace small_canteens_needed_l706_706809

theorem small_canteens_needed 
  (f t a v : ℕ)
  (hf : f = 9)
  (ht : t = 8)
  (ha : a = 7)
  (hv : v = 6) :
  let total_water := f * t + a,
  small_canteens := (total_water + v - 1) / v
  in small_canteens = 14 :=
by
  sorry

end small_canteens_needed_l706_706809


namespace solve_functional_equation_l706_706696

-- Given: Conditions
variables {f : ℝ → ℝ} (h_mono : ∀ x y, x ≤ y → f(x) ≤ f(y))
          (h_eq : ∀ x y, f(x) * f(y) = f(x + y))

-- To Prove: f(x) = a^x for some a > 0
theorem solve_functional_equation :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f(x) = a^x :=
sorry

end solve_functional_equation_l706_706696


namespace no_fractions_meet_condition_l706_706510

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l706_706510


namespace configuration_not_possible_l706_706700

/-- 
  Given:
  - a and b are positive integers.
  - a and b each have exactly m (where m ≥ 3) positive divisors.
  - a₁, a₂, a₃, ..., aₘ is a permutation of all positive divisors of a.
  Prove that it is impossible for {a₁, a₁ + a₂, a₂ + a₃, ..., aₘ₋₁ + aₘ} to be exactly the set of all positive divisors of b.
-/
theorem configuration_not_possible (a b : ℕ) (m : ℕ) (hm : m ≥ 3)
  (ha_div : ∀ d, d ∣ a ↔ d = 1 ∨ d = a ∧ 0 < d ≤ a)
  (hb_div : ∀ d, d ∣ b ↔ d = 1 ∨ d = b ∧ 0 < d ≤ b)
  (exists_perm : ∃ (perm : Fin m → ℕ), ∀ i, (perm i) ∣ a) :
  ¬ (∃ (set_b : Fin (m-1) → ℕ), ∀ i, set_b i = (perm i) + (perm (i+1)) ∧ ∀ i, (set_b i) ∣ b) :=
by {
  sorry
}

end configuration_not_possible_l706_706700


namespace extreme_values_l706_706320

theorem extreme_values (x : ℝ) : 
  ∃ (x_m x_M : ℝ), (∀ x < x_m, f' x < 0) ∧ (∀ x > x_m, f' x > 0) ∧
  (∀ x < x_M, f' x > 0) ∧ (∀ x > x_M, f' x < 0) := 
  let f := λ x : ℝ, -x^3 - x^2 + 2
  let f' := λ x : ℝ, -3 * x^2 - 2 * x
  sorry

end extreme_values_l706_706320


namespace probabilities_equal_l706_706133

variable {N n : ℕ}
variable (P1 P2 P3 : ℚ)

-- Hypotheses based on the conditions:
hypothesis h1 : P1 = n / N
hypothesis h2 : P2 = n / N
hypothesis h3 : P3 = n / N

-- Concluding the theorem to prove:
theorem probabilities_equal :
  P1 = P2 ∧ P2 = P3 :=
by
  sorry

end probabilities_equal_l706_706133


namespace sum_of_first_11_terms_l706_706665

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a n + d

theorem sum_of_first_11_terms (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) 
  (h_cond : a 2 + a 8 = 4) : 
  ∑ i in Finset.range 11, a i = 22 :=
sorry

end sum_of_first_11_terms_l706_706665


namespace value_of_x_l706_706808

theorem value_of_x (x : ℤ) (h : 3 * x = (26 - x) + 26) : x = 13 :=
by
  sorry

end value_of_x_l706_706808


namespace c_investment_l706_706463

theorem c_investment (investment_A investment_B total_profit share_A : ℝ) (C_investment : ℝ) : 
  investment_A = 6300 → 
  investment_B = 4200 → 
  total_profit = 12500 → 
  share_A = 3750 → 
  let total_investment := investment_A + investment_B + C_investment in
  let proportion_A := investment_A / total_investment in
  let proportion_A_profit := share_A / total_profit in
  proportion_A = proportion_A_profit → 
  C_investment = 10500 :=
by
  intros hA hB hTotalProfit hShareA
  let total_investment := 6300 + 4200 + C_investment
  let proportion_A := 6300 / total_investment
  let proportion_A_profit := 3750 / 12500
  have hProportionEq : proportion_A = proportion_A_profit,
  from sorry
  exact sorry

end c_investment_l706_706463


namespace circle_tangent_to_lines_l706_706127

noncomputable def circle_standard_equation (a : ℝ) (r : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (x - a) ^ 2 + (y - 1) ^ 2 = r ^ 2

theorem circle_tangent_to_lines (a : ℝ) : 
  (∃ a, circle_standard_equation a (sqrt 5) 1 1) ∧ ((2 * a - 1) = 1) → 
  (circle_standard_equation 1 (sqrt 5) 1 1) :=
by
  intros h
  sorry

end circle_tangent_to_lines_l706_706127


namespace rational_solutions_exist_l706_706372

theorem rational_solutions_exist (x p q : ℚ) (h : p^2 - x * q^2 = 1) :
  ∃ (a b : ℤ), p = (a^2 + x * b^2) / (a^2 - x * b^2) ∧ q = (2 * a * b) / (a^2 - x * b^2) :=
by
  sorry

end rational_solutions_exist_l706_706372


namespace jasmine_rosewater_mint_percentage_correct_l706_706021

variable (initial_volume : ℕ) (jasmine_initial_frac rosewater_initial_frac mint_initial_frac : ℚ)
variable (jasmine_added rosewater_added mint_added water_added : ℕ)

def initial_jasmine_volume : ℚ := jasmine_initial_frac * initial_volume
def initial_rosewater_volume : ℚ := rosewater_initial_frac * initial_volume
def initial_mint_volume : ℚ := mint_initial_frac * initial_volume

def final_volume : ℚ := initial_volume + jasmine_added + rosewater_added + mint_added + water_added
def final_jasmine_volume : ℚ := initial_jasmine_volume + jasmine_added
def final_rosewater_volume : ℚ := initial_rosewater_volume + rosewater_added
def final_mint_volume : ℚ := initial_mint_volume + mint_added

def percent (volume total_volume : ℚ) : ℚ := (volume / total_volume) * 100

def total_percent_jasmine_rosewater_mint : ℚ :=
  percent final_jasmine_volume final_volume + percent final_rosewater_volume final_volume + percent final_mint_volume final_volume

theorem jasmine_rosewater_mint_percentage_correct :
  initial_volume = 150 → 
  jasmine_initial_frac = 3/100 →
  rosewater_initial_frac = 5/100 →
  mint_initial_frac = 2/100 →
  jasmine_added = 12 →
  rosewater_added = 9 →
  mint_added = 3 →
  water_added = 4 →
  total_percent_jasmine_rosewater_mint = 21.91 :=
by sorry

end jasmine_rosewater_mint_percentage_correct_l706_706021


namespace unique_fraction_condition_l706_706499

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l706_706499


namespace simplify_and_evaluate_expression_l706_706305

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2023) :
  (x + y)^2 + (x + y) * (x - y) - 2 * x^2 = 2023 :=
by
  sorry

end simplify_and_evaluate_expression_l706_706305


namespace roots_cubed_l706_706707

noncomputable def q (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + b^2 - c^2
noncomputable def p (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * (b^2 + 3 * c^2) * x + (b^2 - c^2)^3 
def x1 (b c : ℝ) := b + c
def x2 (b c : ℝ) := b - c

theorem roots_cubed (b c : ℝ) :
  (q b c (x1 b c) = 0 ∧ q b c (x2 b c) = 0) →
  (p b c ((x1 b c)^3) = 0 ∧ p b c ((x2 b c)^3) = 0) :=
by
  sorry

end roots_cubed_l706_706707


namespace find_n_l706_706158

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop := ∀ n > 1, 2 * a n = a (n + 1) + a (n - 1)

def sum_condition (S : ℕ → ℝ) : Prop := S 3 < S 5 ∧ S 5 < S 4

theorem find_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : sequence_condition a)
  (h2 : sum_condition S)
  (h3 : ∀ n, S n = (Σ i in finset.range n, a i)) :
  (∃ n > 1, S (n - 1) * S n < 0) ↔ n = 9 :=
sorry

end find_n_l706_706158


namespace notebook_cost_l706_706840

theorem notebook_cost (n c : ℝ) (h1 : n + c = 2.50) (h2 : n = c + 2) : n = 2.25 :=
by
  sorry

end notebook_cost_l706_706840


namespace sum_of_valid_b_l706_706902

theorem sum_of_valid_b :
  let is_perfect_square (n : ℤ) := ∃ m : ℤ, m * m = n in
  let valid_b (b : ℤ) := (b > 0) ∧ (18 % b = 0) ∧ is_perfect_square (36 - 12 * b) in
  (Finset.sum (Finset.filter valid_b (Finset.range 19))) = 3 :=
by
  sorry

end sum_of_valid_b_l706_706902


namespace find_p_plus_q_l706_706581

noncomputable def p_q_sum : ℕ :=
  let S := ∑ k in finset.range 45, real.sin (4 * (k + 1))
  let p_q := real.tan (S / (q : ℝ)) = real.tan (4 * real.pi / 180) in
  if (p, q).gcd = 1 ∧ (p : ℝ) / ↑q < 90 then
    p + q
  else
    0

theorem find_p_plus_q :
  ∃ p q : ℕ, (p + q = 5 ∧ (p, q).gcd = 1 ∧ (p : ℝ) / ↑q < 90 ∧ S = real.tan (p / q)) :=
sorry

end find_p_plus_q_l706_706581


namespace man_speed_l706_706041

theorem man_speed (time_in_minutes : ℝ) (distance_in_km : ℝ) (T : time_in_minutes = 24) (D : distance_in_km = 4) : 
  (distance_in_km / (time_in_minutes / 60)) = 10 := by
  sorry

end man_speed_l706_706041


namespace max_overlap_l706_706430

variable (A : Type) [Fintype A] [DecidableEq A]
variable (P1 P2 : A → Prop)

theorem max_overlap (hP1 : ∃ X : Finset A, (X.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ X, P1 a)
                    (hP2 : ∃ Y : Finset A, (Y.card : ℝ) / Fintype.card A = 0.70 ∧ ∀ a ∈ Y, P2 a) :
  ∃ Z : Finset A, (Z.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ Z, P1 a ∧ P2 a :=
sorry

end max_overlap_l706_706430


namespace clock_90_degrees_44_times_l706_706212

theorem clock_90_degrees_44_times:
  (count_90_degree_occurrences : ℕ -> ℕ → ℕ) -- a function that counts the occurrences 
  (minute_hand_angle : ℕ → ℝ) -- function representing the angle of the minute hand over time 
  (hour_hand_angle : ℕ → ℝ) -- function representing the angle of the hour hand over time 
  (minute_hand_angle t = 360 * t) -- minute hand completes one full rotation in one hour 
  (hour_hand_angle t = 30 * t) -- hour hand moves 30 degrees per hour 
  (count_90_degree_occurrences 24 (λ t, |minute_hand_angle t - hour_hand_angle t|) = |330 t|):
  count_90_degree_occurrences 24  (λ t, |minute_hand_angle t - hour_hand_angle t|) = 44 := 
sorry

end clock_90_degrees_44_times_l706_706212


namespace probability_distribution_correct_l706_706028

variables (X : Type) (μ σ : ℝ)
variables (x1 x2 : ℝ) (p1 p2 : ℝ)

-- Conditions
def prob_dist :=
  (x2 > x1) ∧ (p1 = 0.6) ∧ (p2 = 0.4) ∧ 
  (μ = 1.4) ∧ (σ = 0.24) ∧
  (μ = p1 * x1 + p2 * x2) ∧ 
  (σ = p1 * x1^2 + p2 * x2^2 - μ^2)

-- Correct Answer
def solution := 
  (x1 = 1) ∧ (x2 = 2) ∧ (p1 = 0.6) ∧ (p2 = 0.4)

-- Theorem
theorem probability_distribution_correct
  (h : prob_dist X μ σ x1 x2 p1 p2) : solution x1 x2 p1 p2 := 
  sorry

end probability_distribution_correct_l706_706028


namespace minimize_cost_l706_706835

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 2) * (x + 5)^2 + 1000 / (x + 5)

theorem minimize_cost :
  (∀ x, 2 ≤ x ∧ x ≤ 8 → cost_function x ≥ 150) ∧ cost_function 5 = 150 :=
by
  sorry

end minimize_cost_l706_706835


namespace fraction_to_terminating_decimal_l706_706115

theorem fraction_to_terminating_decimal :
  (47 : ℚ) / (2^2 * 5^4) = 0.0188 :=
sorry

end fraction_to_terminating_decimal_l706_706115


namespace f_comp_f_1_l706_706597

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else -3^x + 1

theorem f_comp_f_1 : f (f 1) = 0 :=
by
  -- Notice that we are purely stating the theorem here. The proof is not required.
  sorry

end f_comp_f_1_l706_706597


namespace total_pairs_of_shoes_l706_706796

-- Conditions as Definitions
def blue_shoes := 540
def purple_shoes := 355
def green_shoes := purple_shoes  -- The number of green shoes is equal to the number of purple shoes

-- The theorem we need to prove
theorem total_pairs_of_shoes : blue_shoes + green_shoes + purple_shoes = 1250 := by
  sorry

end total_pairs_of_shoes_l706_706796


namespace rectangle_area_l706_706048

-- Definitions
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def length (w : ℝ) : ℝ := 2 * w
def area (l w : ℝ) : ℝ := l * w

-- Main Statement
theorem rectangle_area (w l : ℝ) (h_p : perimeter l w = 120) (h_l : l = length w) :
  area l w = 800 :=
by
  sorry

end rectangle_area_l706_706048


namespace divisible_by_323_if_even_l706_706142

theorem divisible_by_323_if_even (n : ℤ) : 
  (20 ^ n + 16 ^ n - 3 ^ n - 1) % 323 = 0 ↔ n % 2 = 0 := 
by 
  sorry

end divisible_by_323_if_even_l706_706142


namespace abs_diff_condition_l706_706626

theorem abs_diff_condition {a b : ℝ} (h1 : |a| = 1) (h2 : |b - 1| = 2) (h3 : a > b) : a - b = 2 := 
sorry

end abs_diff_condition_l706_706626


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l706_706346

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l706_706346


namespace projection_of_AB_in_direction_of_CD_l706_706165

structure Point2D :=
 (x : ℝ)
 (y : ℝ)

def vector (p1 p2 : Point2D) : Point2D :=
{ x := p2.x - p1.x,
  y := p2.y - p1.y }

def dot_product (v1 v2 : Point2D) : ℝ :=
v1.x * v2.x + v1.y * v2.y

def magnitude (v : Point2D) : ℝ :=
real.sqrt (v.x ^ 2 + v.y ^ 2)

def projection (v1 v2 : Point2D) : ℝ :=
dot_product(v1, v2) / magnitude(v2)

theorem projection_of_AB_in_direction_of_CD :
  let A := Point2D.mk (-1) 1
  let B := Point2D.mk 1 2
  let C := Point2D.mk (-2) (-1)
  let D := Point2D.mk 2 2
  projection (vector A B) (vector C D) = 11 / 5 :=
by
  sorry

end projection_of_AB_in_direction_of_CD_l706_706165


namespace solution_l706_706604

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

lemma even_function (x : ℝ) : f (-x) = f x := 
by sorry

lemma decreasing_on_negative_reals (x : ℝ) (h : x < 0) : (Real.exp x - Real.exp (-x)) < 0 :=
by sorry

theorem solution : (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → (Real.exp x - Real.exp (-x)) < 0) :=
by { split, exact even_function, exact decreasing_on_negative_reals }

end solution_l706_706604


namespace range_of_a_l706_706193

variable (a : ℝ) (x : ℝ)

def y (a : ℝ) (x : ℝ) : ℝ := (2 * a - 3) / x

theorem range_of_a (h : x > 0) (increasing : (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → y a x₁ < y a x₂)) : a < 3 / 2 :=
by
  sorry

end range_of_a_l706_706193


namespace dessert_menus_count_l706_706833

-- Define the dessert options
inductive Dessert
| cake
| pie
| ice_cream
| pudding

-- Define the conditions
def valid_menu : List Dessert → Prop :=
  λ menu, menu.length = 7 ∧
          ∀ i < 6, menu.get i ≠ menu.get (i + 1) ∧
          menu.get 4 = Dessert.cake

-- The proof problem statement
theorem dessert_menus_count : 
  ∃ menu_count : ℕ, menu_count = 729 ∧
  (∃ menus : List (List Dessert),
     ∀ menu ∈ menus, valid_menu menu) :=
begin
  use 729,
  sorry
end

end dessert_menus_count_l706_706833


namespace number_of_preferred_groups_l706_706431

def preferred_group_sum_multiple_5 (n : Nat) : Nat := 
  (2^n) * ((2^(4*n) - 1) / 5 + 1) - 1

theorem number_of_preferred_groups :
  preferred_group_sum_multiple_5 400 = 2^400 * (2^1600 - 1) / 5 + 1 - 1 :=
sorry

end number_of_preferred_groups_l706_706431


namespace wholesale_price_l706_706053

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l706_706053


namespace AB_mul_CD_geq_BC_squared_over_four_equality_case_l706_706038

-- Geometric entities: Triangle, Incenter, Circumcenter, Incircle, Circumcircle
variables {α : Type*} [EuclideanGeometry α]

-- Definitions and variables
variables {triangle : Triangle α}
variables {I O : Point α} -- Incenter and circumcenter
variables {r R : ℝ} -- Inradius and circumradius
variables (A B C D : Point α)

-- Conditions
def incenter (triangle : Triangle α) : Point α := I
def circumcenter (triangle : Triangle α) : Point α := O
def inradius (triangle : Triangle α) : ℝ := r
def circumradius (triangle : Triangle α) : ℝ := R

-- Axioms and assumptions
axiom line_through_incenter : ℕ → Line ℕ
axiom meets_circumcircle : meets (line_through_incenter I) (circle O R) A ∧ meets (line_through_incenter I) (circle O R) B 
axiom meets_incircle : meets (line_through_incenter I) (circle I r) C ∧ meets (line_through_incenter I) (circle I r) D

-- Theorem Statement
theorem AB_mul_CD_geq_BC_squared_over_four (triangle : Triangle α) (I O : Point α)
  (r R : ℝ) (A B C D : Point α)
  (h1 : incenter triangle = I) (h2 : circumcenter triangle = O) 
  (h3 : inradius triangle = r) (h4 : circumradius triangle = R)
  (h5 : meets_circumcircle I O A B) (h6 : meets_incircle I r C D) :
  (dist A B) * (dist C D) ≥ (dist B C)^2 / 4 :=
sorry

-- Equality case equivalent statement
theorem equality_case (triangle : Triangle α) (I O : Point α)
  (r R : ℝ) (A B C D : Point α)
  (h1 : incenter triangle = I) (h2 : circumcenter triangle = O) 
  (h3 : inradius triangle = r) (h4 : circumradius triangle = R)
  (h5 : meets_circumcircle I O A B) (h6 : meets_incircle I r C D) :
  (dist A B) * (dist C D) = (dist B C)^2 / 4 ↔ (line_through_incenter I O) :=
sorry


end AB_mul_CD_geq_BC_squared_over_four_equality_case_l706_706038


namespace coterminal_angle_l706_706121

theorem coterminal_angle (k : ℤ) : ∃ θ : ℤ, θ = -463 + 360 * k → θ = 257 + 360 * k :=
by {
  intro k,
  use -463 + 360 * k,
  use 257 + 360 * k,
  sorry
}

end coterminal_angle_l706_706121


namespace balls_into_boxes_l706_706970

def ways_to_distribute_balls_in_boxes (balls boxes : ℕ) : ℕ :=
  ∑ (p : Multiset ℕ) in Multiset.powersetLen boxes (Multiset.replicate balls (1 : ℕ)).eraseDup,
  if p.sum = balls then p.map (fun x => x.card).prod else 0

theorem balls_into_boxes :
  ways_to_distribute_balls_in_boxes 6 4 = 84 :=
  sorry

end balls_into_boxes_l706_706970


namespace interesting_ads_percentage_l706_706205

-- Initial conditions
variables {total_ads not_blocked blocked not_interesting : ℝ}

-- Ad conditions
def ad_conditions : Prop :=
  blocked = 0.80 * total_ads ∧    -- 80% of ads are blocked
  not_blocked = 0.20 * total_ads ∧  -- 20% of ads are not blocked
  not_interesting = 0.16 * total_ads -- 16% of ads are not interesting (and not blocked)

-- Question to answer
theorem interesting_ads_percentage (h : ad_conditions) :
  let interesting_unblocked := not_blocked - not_interesting in
  let percentage_interesting_unblocked := (interesting_unblocked / not_blocked) * 100 in
  percentage_interesting_unblocked = 20 :=
by
  intros
  sorry

end interesting_ads_percentage_l706_706205


namespace tailor_trim_length_l706_706457

theorem tailor_trim_length (x : ℕ) : 
  (18 - x) * 15 = 120 → x = 10 := 
by
  sorry

end tailor_trim_length_l706_706457


namespace triangle_AC_length_l706_706989

theorem triangle_AC_length (A B C D E: Point) : 
  (D ∈ segment A C) → 
  (E ∈ segment A B) → 
  (AB ⊥ AC) → 
  (DE ⊥ BC) → 
  (BD = 2) → 
  (DC = 2) → 
  (EC = 2) → 
  (AC = 4) :=
by 
  sorry

end triangle_AC_length_l706_706989


namespace expand_product_l706_706893

noncomputable def question_expression (x : ℝ) := -3 * (2 * x + 4) * (x - 7)
noncomputable def correct_answer (x : ℝ) := -6 * x^2 + 30 * x + 84

theorem expand_product (x : ℝ) : question_expression x = correct_answer x := 
by sorry

end expand_product_l706_706893


namespace flowerbed_width_l706_706844

theorem flowerbed_width (w : ℝ) (h₁ : 22 = 2 * (2 * w - 1) + 2 * w) : w = 4 :=
sorry

end flowerbed_width_l706_706844


namespace range_of_a_l706_706426

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → a ≥ Real.exp x

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4 * x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∧ q a) → a ∈ Set.Icc Real.exp 1 4 := by 
  sorry

end range_of_a_l706_706426


namespace a_fraction_of_capital_l706_706068

theorem a_fraction_of_capital (T : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (profit_A : ℝ) (total_profit : ℝ)
  (h1 : B = T * (1 / 4))
  (h2 : C = T * (1 / 5))
  (h3 : D = T - (T * (1 / 4) + T * (1 / 5) + T * x))
  (h4 : profit_A = 805)
  (h5 : total_profit = 2415) :
  x = 161 / 483 :=
by
  sorry

end a_fraction_of_capital_l706_706068


namespace goldbach_140_largest_difference_l706_706328

open Nat

theorem goldbach_140_largest_difference :
  ∃ (p q : ℕ), p ≠ q ∧ prime p ∧ prime q ∧ p + q = 140 ∧ abs (p - q) = 134 :=
by
  sorry

end goldbach_140_largest_difference_l706_706328


namespace harmonic_mean_closest_to_2_l706_706886

theorem harmonic_mean_closest_to_2 (a : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : b = 4032) : 
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  rw [h₁, h₂]
  -- The rest of the proof follows from here, skipped with sorry
  sorry

end harmonic_mean_closest_to_2_l706_706886


namespace largest_n_possible_no_integer_centroid_l706_706816

-- Defining the concept of the centroid
def centroid (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) : (ℚ × ℚ) :=
  ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3)

-- Given 8 points such that the centroid of any triangle formed by 3 points has non-integer coordinates
def no_integer_centroid {n : ℕ} (points : Fin₈ (ℤ × ℤ)) : Prop :=
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k →
    ¬ ∃ x : ℤ, ∃ y : ℤ, centroid (points i).fst (points j).fst (points k).fst (points i).snd (points j).snd (points k).snd = (x, y) 

-- Prove that it is possible to have 8 such points
theorem largest_n_possible_no_integer_centroid : ∃ points : Fin₈ (ℤ × ℤ), no_integer_centroid points :=
sorry

end largest_n_possible_no_integer_centroid_l706_706816


namespace polynomial_p20_value_l706_706908

theorem polynomial_p20_value :
  ∀ (k : ℝ), (p : ℝ → ℝ) (p1_cond : p 1 = p 10) (h : ∀ x, p x = 3 * x^2 + k * x + 117),
  p 20 = 657 :=
by
  intro k p p1_cond h
  have h1 : p 1 = k + 120 := by sorry
  have h2 : p 10 = 10 * k + 417 := by sorry
  have p1_10_eq : k + 120 = 10 * k + 417 := by sorry
  have k_val : k = -33 := by sorry
  have p20_val : p 20 = 3 * 20^2 + k * 20 + 117 := by sorry
  have p20_val_simplified : p 20 = 657 := by sorry
  exact p20_val_simplified 

end polynomial_p20_value_l706_706908


namespace area_of_plot_land_l706_706764

-- Define the dimensions, scale, and conversion factors
def bottom_side_cm := 12
def top_side_cm := 18
def height_cm := 8
def scale_cm_to_miles := 5
def square_mile_to_acres := 640

-- Define the areas and conversion formulas
def area_cm_squared := (bottom_side_cm + top_side_cm) * height_cm / 2
def area_miles_squared := area_cm_squared * scale_cm_to_miles^2
def area_acres := area_miles_squared * square_mile_to_acres

theorem area_of_plot_land :
  area_acres = 1920000 := by
  sorry

#print axiom area_of_plot_land

end area_of_plot_land_l706_706764


namespace proof_m_range_l706_706197

variable {x m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

theorem proof_m_range (h : A m ∩ B = ∅) : m ≤ -2 := 
sorry

end proof_m_range_l706_706197


namespace gray_region_area_l706_706873

noncomputable def area_gray_region : ℝ :=
  let area_rectangle := (12 - 4) * (12 - 4)
  let radius_c := 4
  let radius_d := 4
  let area_quarter_circle_c := 1/4 * Real.pi * radius_c^2
  let area_quarter_circle_d := 1/4 * Real.pi * radius_d^2
  let overlap_area := area_quarter_circle_c + area_quarter_circle_d
  area_rectangle - overlap_area

theorem gray_region_area :
  area_gray_region = 64 - 8 * Real.pi := by
  sorry

end gray_region_area_l706_706873


namespace moment_of_inertia_correct_l706_706561

variable (h R ρ : ℝ)
def moment_of_inertia_cylinder : ℝ :=
  (π * ρ * h * R^4) / 2

theorem moment_of_inertia_correct
  (h R ρ : ℝ) :
  moment_of_inertia_cylinder h R ρ = 
    (π * ρ * h * R^4) / 2 :=
by sorry

end moment_of_inertia_correct_l706_706561


namespace tiffany_pictures_in_each_album_l706_706798

theorem tiffany_pictures_in_each_album (phone_pics camera_pics total_albums : ℕ) 
  (h1 : phone_pics = 7) 
  (h2 : camera_pics = 13) 
  (h3 : total_albums = 5) 
  (total_pics : ℕ) 
  (h4 : total_pics = phone_pics + camera_pics) 
  : total_pics / total_albums = 4 := 
begin
  sorry
end

end tiffany_pictures_in_each_album_l706_706798


namespace sum_inequality_l706_706935

theorem sum_inequality (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, 0 ≤ x i)
  (h_sum : (∑ i, x i) = n) :
  (∑ i, x i / (1 + (x i)^2)) ≤ (∑ i, 1 / (1 + x i)) :=
sorry

end sum_inequality_l706_706935


namespace animals_consuming_hay_l706_706030

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end animals_consuming_hay_l706_706030


namespace polar_circle_equation_l706_706427

theorem polar_circle_equation (ρ θ : ℝ) (O pole : ℝ) (eq_line : ρ * Real.cos θ + ρ * Real.sin θ = 2) :
  (∃ ρ, ρ = 2 * Real.cos θ) :=
sorry

end polar_circle_equation_l706_706427


namespace find_original_class_strength_l706_706410

-- Definitions based on given conditions
def original_average_age : ℝ := 40
def additional_students : ℕ := 12
def new_students_average_age : ℝ := 32
def decrease_in_average : ℝ := 4
def new_average_age : ℝ := original_average_age - decrease_in_average

-- The equation setup
theorem find_original_class_strength (N : ℕ) (T : ℝ) 
  (h1 : T = original_average_age * N) 
  (h2 : T + additional_students * new_students_average_age = new_average_age * (N + additional_students)) : 
  N = 12 := 
sorry

end find_original_class_strength_l706_706410


namespace arithmetic_sequence_lambda_l706_706159

theorem arithmetic_sequence_lambda (λ : ℝ) :
  (∀ n ≥ 7, (n^2 + (1 + λ) * n) > (n^2 + (1 + λ) * (n - 1))) → λ > -16 :=
by {
  intros h,
  -- Proof goes here.
  sorry
}

end arithmetic_sequence_lambda_l706_706159


namespace sum_f_values_l706_706170

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def odd_shifted_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x - 1) = -f (-(x - 1))

theorem sum_f_values : 
  ∀ (f : ℝ → ℝ), 
  even_function f → 
  odd_shifted_function f → 
  f 2 = -1 → 
  (∑ i in finset.range 2009, f (i + 1)) = 0 := 
by 
  sorry

end sum_f_values_l706_706170


namespace problem_l706_706195

def p : Prop := ∃ x : ℝ, x - 2 > log x / log 2
def q : Prop := ∀ x : ℝ, x^2 > 0

theorem problem : p ∧ ¬q :=
by
  -- Placeholder for the proof
  sorry

end problem_l706_706195


namespace sum_of_digits_of_all_six_digit_palindromes_l706_706880

def is_palindrome_form (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ (range 9).succ ∧ 
                b ∈ range 10 ∧ 
                c ∈ range 10 ∧ 
                n = 100001 * a + 10010 * b + 1100 * c

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_string.to_list.map (λ c => c.to_nat - '0'.to_nat)).sum

theorem sum_of_digits_of_all_six_digit_palindromes : 
  sum_of_digits (∑ n in (finset.filter is_palindrome_form (finset.range 1000000)), n) = 45 :=
by sorry

end sum_of_digits_of_all_six_digit_palindromes_l706_706880


namespace chad_bbq_people_l706_706483

theorem chad_bbq_people (ice_cost_per_pack : ℝ) (packs_included : ℕ) (total_money_spent : ℝ) (pounds_needed_per_person : ℝ) :
  total_money_spent = 9 → 
  ice_cost_per_pack = 3 → 
  packs_included = 10 → 
  pounds_needed_per_person = 2 → 
  ∃ (people : ℕ), people = 15 :=
by intros; sorry

end chad_bbq_people_l706_706483


namespace problem_1_problem_2_problem_3_problem_4_l706_706136

def A := { x // 1 ≤ x ∧ x ≤ 23 }

def is_harmonic (B : Set Int) :=
  (B.card = 12) ∧ ∃ a b, a ∈ B ∧ b ∈ B ∧ b < a ∧ b ∣ a

theorem problem_1 (q r : Int) (hq : q = 22) (hr : r = 9) :
  2011 = 91 * q + r ∧ 0 ≤ r ∧ r < 91 := by
  sorry

def B := { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }
def C := { 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23 }

theorem problem_2 : is_harmonic B := by
  sorry

theorem problem_3 : ¬ is_harmonic C := by
  sorry

theorem problem_4 : ∀ (m : Int), m ∈ A → (∀ (S : Set Int), S.card = 12 → m ∈ S → is_harmonic S) ↔ m ≤ 7 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l706_706136


namespace prove_ellipse_and_lambda_l706_706930

def ellipse_equation {a b : ℝ} (Q : ℝ × ℝ) (e : ℝ) (P : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 1/2 ∧ 
  (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧ P = (1, 2)

def lambda_existence {λ : ℝ} (P A B M N : ℝ × ℝ) : Prop :=
  let PA := dist P A
  let PB := dist P B
  let PM := dist P M
  let PN := dist P N
  λ = 1 → PA * PB = λ * PM * PN

theorem prove_ellipse_and_lambda :
  ∃ a b : ℝ, ellipse_equation (1, 3/2) 1/2 (1, 2) ∧ (a = 2 ∧ b = sqrt 3) ∧
  λ = 1 ∧ λ_existence (1, 2) A B M N :=
by
  -- Proof omitted
  sorry

end prove_ellipse_and_lambda_l706_706930


namespace target_eq_complement_intersection_l706_706279

open Set

variable {U : Type} [ord : LinearOrder U] 

def E : Set U := {x | x ≤ -3 ∨ x ≥ 2}
def F : Set U := {x | -1 < x ∧ x < 5}
def target_set : Set U := {x | -1 < x ∧ x < 2}
def CU (A : Set U) : Set U := {x | ¬ (A x)}

theorem target_eq_complement_intersection (Ueq : U = ℝ) :
  target_set = CU E ∩ F :=
by
  sorry

end target_eq_complement_intersection_l706_706279


namespace horse_distance_traveled_l706_706035

theorem horse_distance_traveled :
  let r2 := 12
  let n2 := 120
  let D2 := n2 * 2 * Real.pi * r2
  D2 = 2880 * Real.pi :=
by
  sorry

end horse_distance_traveled_l706_706035


namespace length_AD_proof_l706_706999

variable (A B C D : ℝ)
variable (AB BC CD AD : ℝ)
variable [Quadrilateral ABCD]
variable (B_obtuse : ∠B > π/2)
variable (sin_B : real.sin (∠B) = 4 / 5)
variable (cos_C : real.cos (∠C) = 4 / 5)

theorem length_AD_proof :
  AB = 3 → BC = 4 → CD = 12 → AD = 9.1 :=
by
  intros h1 h2 h3
  sorry

end length_AD_proof_l706_706999


namespace combine_like_terms_substitute_expression_complex_expression_l706_706417

-- Part 1
theorem combine_like_terms (a b : ℝ) : 
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 :=
by
  sorry

-- Part 2
theorem substitute_expression (x y : ℝ) (h1 : x^2 - 2 * y = -5) : 
  4 * x^2 - 8 * y + 24 = 4 :=
by
  sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2 * b = 1009.5) 
  (h2 : 2 * b - c = -2024.6666)
  (h3 : c - d = 1013.1666) : 
  (a - c) + (2 * b - d) - (2 * b - c) = -2 :=
by
  sorry

end combine_like_terms_substitute_expression_complex_expression_l706_706417


namespace area_BQW_l706_706669

open EuclideanGeometry

variables {Point : Type} [MetricSpace Point] [EuclideanSpace Point]

noncomputable def is_midpoint (Q Z W : Point) : Prop :=
dist Z Q = dist Q W

noncomputable def is_trapezoid (A B C D : Point) : Prop := 
∃ (Z W : Point), dist A Z = dist B W ∧ dist Z W = 10 ∧ dist A B = 20

noncomputable def area_trapezoid (Z W C D : Point) : ℝ := 
let base1 := dist Z W in
let base2 := dist C D in
let height := 20 in -- calculated from trapezoid area
(base1 + base2) * height / 2

noncomputable def area_triangle (X Y Z : Point) : ℝ :=
let base := dist X Y in
let height := dist Z (lineFrom X Y) in -- height from Z to line XY
(base * height) / 2

theorem area_BQW (A B C D Z W Q : Point)
  (h1 : is_rectangle A B C D)
  (h2 : dist A Z = 10)
  (h3 : dist B W = 10)
  (h4 : dist A B = 20)
  (h5 : area_trapezoid Z W C D = 200)
  (h6 : is_midpoint Q Z W) :
  area_triangle B Q W = 150 :=
sorry

end area_BQW_l706_706669


namespace monotonic_intervals_of_f_range_of_a_if_f_has_solution_l706_706187

noncomputable def f (a b x : ℝ) : ℝ :=
  (2 * a * x^2 + b * x + 1) * Real.exp (-x)

theorem monotonic_intervals_of_f (b : ℝ) (h_b : b ≥ 0) :
  ∃ I_decreasing I_increasing : set ℝ,
  (a = 1/2) → (I_decreasing = (-∞, 1) ∪ (1 - b, ∞) ∨  I_increasing = (1 - b, 1)) ∧
    ∀ x ∈ I_decreasing, ∀ y ∈ I_increasing, f 1/2 b x ≤ f 1/2 b y :=
sorry

theorem range_of_a_if_f_has_solution (a : ℝ) :
  f a (Real.exp 1 - 1 - 2 * a) 1 = 1 ∧
  (∃ x ∈ set.Ioo 0 1, f a (Real.exp 1 - 1 - 2 * a) x = 1) →
  a ∈ set.Ioo ((Real.exp 1 - 2) / 2) (1 / 2) :=
sorry

end monotonic_intervals_of_f_range_of_a_if_f_has_solution_l706_706187


namespace distribution_of_balls_l706_706213

theorem distribution_of_balls :
  let balls := 5
  let boxes := 3
  (∃ (distribution : (fin balls) → fin boxes),
    ∀ b : fin boxes, ∃ ball : fin balls, distribution ball = b) →
  150 :=
begin
  sorry
end

end distribution_of_balls_l706_706213


namespace analytical_expression_C_not_on_graph_l706_706925

structure Point (α : Type) :=
  (x : α)
  (y : α)

noncomputable
def linear_function (k b : ℝ) : ℝ → ℝ := λ x, k * x + b

-- Given conditions in part 1
axiom A (h1 : Point ℝ)
axiom B (h2 : Point ℝ)
axiom A_value : A = Point.mk 2 1
axiom B_value : B = Point.mk (-3) 6

-- Part 1: Proving the analytical expression of the linear function
theorem analytical_expression : ∃ k b, 
  linear_function k b 2 = 1 ∧
  linear_function k b (-3) = 6 ∧ 
  ∀ x, linear_function k b x = -x + 3 := by sorry

-- Part 2: Determine whether point C lies on the graph
axiom C (h : Point ℝ)
axiom C_value : C = Point.mk (-1) 5

theorem C_not_on_graph : ∃ k b,
  linear_function k b (-1) ≠ 5 := by sorry

end analytical_expression_C_not_on_graph_l706_706925


namespace solve_eqn_l706_706120

noncomputable def a : ℝ := 5 + 2 * Real.sqrt 6
noncomputable def b : ℝ := 5 - 2 * Real.sqrt 6

theorem solve_eqn (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 10) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_eqn_l706_706120


namespace sum_of_gray_squares_is_20_l706_706822

/-- In a grid where each "\(\hookleftarrow\)"-shaped region must be filled with either 
    the set {1, 3, 5, 7} or {2, 4, 6, 8}, and no two adjacent cells have consecutive numbers,
    the sum of the numbers in the gray squares is 20. -/
theorem sum_of_gray_squares_is_20
    (hook1 hook2 hook3 hook4 : Finset ℕ)
    (gray1 gray2 gray3 gray4 : ℕ)
    (H1 : hook1 = {1, 3, 5, 7} ∨ hook1 = {2, 4, 6, 8})
    (H2 : hook2 = {1, 3, 5, 7} ∨ hook2 = {2, 4, 6, 8})
    (H3 : hook3 = {1, 3, 5, 7} ∨ hook3 = {2, 4, 6, 8})
    (H4 : hook4 = {1, 3, 5, 7} ∨ hook4 = {2, 4, 6, 8})
    (nonconsecutive : ∀ a b ∈ hook1 ∪ hook2 ∪ hook3 ∪ hook4, abs (a - b) ≠ 1)
    (gray_in_hooks : gray1 ∈ hook1 ∧ gray2 ∈ hook2 ∧ gray3 ∈ hook3 ∧ gray4 ∈ hook4) :
    gray1 + gray2 + gray3 + gray4 = 20 := by
    sorry

end sum_of_gray_squares_is_20_l706_706822


namespace sequence_term_l706_706611

noncomputable def a : ℕ → ℚ
| 1    := 2
| (n+1) := a n / (1 + 3 * a n)

theorem sequence_term (n : ℕ) : a n = 2 / (6 * n - 5) :=
sorry

end sequence_term_l706_706611


namespace non_congruent_triangles_count_l706_706968

-- Define the points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the given points
def p1 : Point := ⟨0, 0⟩
def p2 : Point := ⟨1, 0⟩
def p3 : Point := ⟨2, 0⟩
def p4 : Point := ⟨3, 0⟩
def p5 : Point := ⟨0, 1⟩
def p6 : Point := ⟨1, 1⟩
def p7 : Point := ⟨2, 1⟩
def p8 : Point := ⟨3, 1⟩
def p9 : Point := ⟨0.5, 0.5⟩
def p10 : Point := ⟨2.5, 0.5⟩

-- List of all points
def points : List Point := [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

-- Placeholder for non-congruence predicate
noncomputable def non_congruent (Δ1 Δ2 : List Point) : Prop := sorry

-- Theorem statement
theorem non_congruent_triangles_count :
  (List.filter (λ (Δ : List Point), 
    Δ.length = 3 ∧ (∀ Δ', Δ'.length = 3 ∧ Δ' ∈ List.combinations 3 points → non_congruent Δ Δ')) 
    (List.combinations 3 points)).length = 12 :=
  sorry

end non_congruent_triangles_count_l706_706968


namespace ellipse_problem_l706_706161

noncomputable def ellipse_equation (a b c : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (a + c = 2 + real.sqrt 3) ∧ (a - c = 2 - real.sqrt 3) ∧ (b^2 = a^2 - c^2)

noncomputable def line_slopes_geometric (k b x1 x2 : ℝ) : Prop :=
  let y1 := k * x1 + b in
  let y2 := k * x2 + b in
  k^2 = (y1 * y2) / (x1 * x2)

noncomputable def delta_discriminant (b k : ℝ) : Prop :=
  let delta := (8 * k * b)^2 - 4 * (4 * k^2 + 1) * (4 * b^2 - 4) in
  delta > 0

noncomputable def line_properties (k : ℝ) : Prop :=
  4 * k^2 + 1 - b^2 > 0

noncomputable def max_triangle_area (b k : ℝ) : ℝ :=
  if 0 < b ∧ b < real.sqrt 2 ∧ 4 * k^2 = 1 then
    1
  else
    0

theorem ellipse_problem (a b c k b_: ℝ) (x1 x2 : ℝ) :
  ellipse_equation a b c →
  line_slopes_geometric k b_ x1 x2 →
  delta_discriminant b_ k →
  line_properties k →
  max_triangle_area b_ k = 1 := sorry

end ellipse_problem_l706_706161


namespace average_lawn_cuts_per_month_l706_706285

theorem average_lawn_cuts_per_month :
  let cuts_in_summer := 6 * 15,
      cuts_in_winter := 6 * 3,
      total_cuts := cuts_in_summer + cuts_in_winter,
      months_in_year := 12
  in
  total_cuts / months_in_year = 9 := by
  sorry

end average_lawn_cuts_per_month_l706_706285


namespace good_point_pair_at_least_one_l706_706797

noncomputable def good_point_pair_exists (A B : Point) (C1 C2 C3 C4 : Point) : Prop :=
  ∃ i j : Fin 4, i ≠ j ∧ |Real.sin (∠ A (Fin.cases1 [C1, C2, C3, C4] i.val) B) - Real.sin (∠ A (Fin.cases1 [C1, C2, C3, C4] j.val) B)| ≤ 1/3

theorem good_point_pair_at_least_one (A B : Point) (C1 C2 C3 C4 : Point) 
  (hC_distinct : C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4)
  (hAB_distinct : A ≠ B ∧ A ≠ C1 ∧ A ≠ C2 ∧ A ≠ C3 ∧ A ≠ C4 ∧ B ≠ C1 ∧ B ≠ C2 ∧ B ≠ C3 ∧ B ≠ C4) :
  good_point_pair_exists A B C1 C2 C3 C4 :=
sorry

end good_point_pair_at_least_one_l706_706797


namespace circumscribed_quadrilateral_midpoint_square_l706_706782

theorem circumscribed_quadrilateral_midpoint_square 
  (ABCD PQRS : ℝ → ℝ → Prop) 
  (P Q R S: ℝ × ℝ)
  (A B C D: ℝ × ℝ)
  (hmidpoint1: P = midpoint A B) 
  (hmidpoint2: Q = midpoint B C) 
  (hmidpoint3: R = midpoint C D) 
  (hmidpoint4: S = midpoint D A)
  (hPQRS_square : is_square PQRS):
  (diagonal_length_eq ABD ABCD ∧ diagonal_perpendicular ABCD) := 
sorry -- Proof to be completed

end circumscribed_quadrilateral_midpoint_square_l706_706782


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706340

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l706_706340


namespace pow_m_plus_n_l706_706638

theorem pow_m_plus_n (x m n : ℝ) (h : (log 10 x)^2 - log 10 x + log 10 2 * log 10 5 = 0) (hm : 2 ≤ x) (hn : x ≤ 5) :
  2 ^ (m + n) = 128 :=
sorry

end pow_m_plus_n_l706_706638


namespace balloon_height_l706_706132

noncomputable def distanceBetween (p1 p2: ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem balloon_height :
  let O := (0, 0, 0)
  let A := (0, 1, 0)
  let B := (-1, 0, 0)
  let C := (0, -1, 0)
  let D := (1, 0, 0)
  ∀ H : ℝ × ℝ,
    distanceBetween O H = H.1 →  -- H.1 is the height OH
    distanceBetween C H = 150 →
    distanceBetween D H = 130 →
    distanceBetween C D = 140 →
    H.1 = 30 * Real.sqrt 11 :=
by
  intros O A B C D H hOH hCH hDH hCD
  sorry -- Proof goes here

end balloon_height_l706_706132


namespace length_of_short_pieces_l706_706462

def total_length : ℕ := 27
def long_piece_length : ℕ := 4
def number_of_long_pieces : ℕ := total_length / long_piece_length
def remainder_length : ℕ := total_length % long_piece_length
def number_of_short_pieces : ℕ := 3

theorem length_of_short_pieces (h1 : remainder_length = 3) : (remainder_length / number_of_short_pieces) = 1 :=
by
  sorry

end length_of_short_pieces_l706_706462


namespace parabola_properties_l706_706770

noncomputable def parabola_equation (a b c d e f : ℤ) : Prop :=
  a = 10 ∧ b = 0 ∧ c = 0 ∧ d = -100 ∧ e = -9 ∧ f = 250

theorem parabola_properties :
  (∃ (a b c d e f : ℤ), parabola_equation a b c d e f) ∧
  ∀ (a b c d e f : ℤ), parabola_equation a b c d e f → 
    (passes_through (λ x y, a * x^2 + b * x * y + c * y^2 + d * x + e * y + f) (2, 10))
    ∧ (focus_x_eq (λ x y, a * x^2 + b * x * y + c * y^2 + d * x + e * y + f) 5)
    ∧ (axis_of_symmetry_parallel_to_y (λ x y, a * x^2 + b * x * y + c * y^2 + d * x + e * y + f))
    ∧ (vertex_on_x_axis (λ x y, a * x^2 + b * x * y + c * y^2 + d * x + e * y + f)) :=
begin
  sorry
end

-- Definitions of the conditions
def passes_through (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  f P.1 P.2 = 0

def focus_x_eq (f : ℝ → ℝ → ℝ) (x0 : ℝ) : Prop :=
  -- Placeholder definition. Actual geometric properties need to be captured
  sorry

def axis_of_symmetry_parallel_to_y (f : ℝ → ℝ → ℝ) : Prop :=
  -- Placeholder definition. Actual geometric properties need to be captured
  sorry

def vertex_on_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  -- Placeholder definition. Actual geometric properties need to be captured
  sorry

end parabola_properties_l706_706770


namespace modeq_inequality_l706_706134

noncomputable def s (n : ℕ) : ℕ :=
  if ∃ a : ℕ, n = a ^ 2010 then 1 else 0  -- Placeholder, requires correct definition

theorem modeq_inequality (x : ℝ) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ s i = x) :=
begin
  sorry
end

end modeq_inequality_l706_706134


namespace solutions_to_equation_l706_706119

theorem solutions_to_equation :
  (∃ x : ℝ, (real.rpow (64 - 2 * x) (1 / 4) + real.rpow (48 + 2 * x) (1 / 4) = 6) ∧ (x = 32 ∨ x = -8)) :=
sorry

end solutions_to_equation_l706_706119


namespace quadratic_to_vertex_form_l706_706524

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (x^2 - 2*x + 3 = (x-1)^2 + 2) :=
by intro x; sorry

end quadratic_to_vertex_form_l706_706524


namespace part_1_part_2_1_part_2_2_l706_706918

variable {k x : ℝ}
def y (k : ℝ) (x : ℝ) := k * x^2 - 2 * k * x + 2 * k - 1

theorem part_1 (k : ℝ) : (∀ x, y k x ≥ 4 * k - 2) ↔ (0 ≤ k ∧ k ≤ 1 / 3) := by
  sorry

theorem part_2_1 (k : ℝ) : ¬∃ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1^2 + x2^2 = 3 * x1 * x2 - 4 := by
  sorry

theorem part_2_2 (k : ℝ) : (∀ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1 > 0 ∧ x2 > 0) ↔ (1 / 2 < k ∧ k < 1) := by
  sorry

end part_1_part_2_1_part_2_2_l706_706918


namespace max_quadratic_eqns_with_roots_l706_706858

theorem max_quadratic_eqns_with_roots
  (S : Finset ℕ)
  (h₁ : ∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ (n / 100 = 3 ∨ n / 100 = 5 ∨ n / 100 = 7 ∨ n / 100 = 9))
  (h₂ : S.card ≥ 100) :
  ∃ a b c ∈ S, ∃ T ⊆ S, T.card = 100 ∧ ∀ ⦃a b c : ℕ⦄, a ∈ T → b ∈ T → c ∈ T → b^2 - 4 * a * c ≥ 0 :=
sorry

end max_quadratic_eqns_with_roots_l706_706858


namespace blue_red_marble_ratio_l706_706448

-- Define the initial counts and conditions
def initial_red_marbles := 20
def initial_blue_marbles := 30
def red_marbles_taken := 3
def total_marbles_left := 35

-- Formulate the equivalent proof problem
theorem blue_red_marble_ratio :
  let red_marbles_left := initial_red_marbles - red_marbles_taken in
  let blue_marbles_left := total_marbles_left - red_marbles_left in
  let blue_marbles_taken := initial_blue_marbles - blue_marbles_left in
  (blue_marbles_taken : ℚ) / red_marbles_taken = 4 :=
by {
  sorry
}

end blue_red_marble_ratio_l706_706448


namespace greatest_sum_of_vertex_products_l706_706466

theorem greatest_sum_of_vertex_products:
  ∃ (a b c d e f : ℕ), {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧
  (a + b) * (c + d) * (e + f) = 343 :=
by
  sorry

end greatest_sum_of_vertex_products_l706_706466


namespace total_brownies_correct_l706_706284

def brownies_initial : Nat := 24
def father_ate : Nat := brownies_initial / 3
def remaining_after_father : Nat := brownies_initial - father_ate
def mooney_ate : Nat := remaining_after_father / 4
def remaining_after_mooney : Nat := remaining_after_father - mooney_ate
def benny_ate : Nat := (remaining_after_mooney * 2) / 5
def remaining_after_benny : Nat := remaining_after_mooney - benny_ate
def snoopy_ate : Nat := 3
def remaining_after_snoopy : Nat := remaining_after_benny - snoopy_ate
def new_batch : Nat := 24
def total_brownies : Nat := remaining_after_snoopy + new_batch

theorem total_brownies_correct : total_brownies = 29 :=
by
  sorry

end total_brownies_correct_l706_706284


namespace anna_candy_per_house_l706_706078

theorem anna_candy_per_house :
  ∃ x : ℕ, 
    let y := 11 in 
    let m := 60 in 
    let n := 75 in 
    60 * x = 75 * y + 15 ∧ 
    x = 14 := 
by
  use 14
  have y : ℕ := 11
  have m : ℕ := 60
  have n : ℕ := 75
  sorry

end anna_candy_per_house_l706_706078


namespace part1_part2_l706_706909

-- Definitions of the quadratic equation and discriminant
def quadratic_eq (k x : ℝ) := x^2 - k*x + k - 1 = 0

def discriminant (a b c : ℝ) := b^2 - 4*a*c

-- Prove that the equation always has two real roots
theorem part1 (k : ℝ) : 
  let a := 1
  let b := -k
  let c := k - 1
  discriminant a b c ≥ 0 :=
by
  let a := 1
  let b := -k
  let c := k - 1
  have h_discriminant : discriminant a b c = (k - 2)^2 := 
    calc
      discriminant a b c = b^2 - 4 * a * c : by sorry
      ... = k^2 - 4 * k + 4 : by sorry
      ... = (k - 2)^2 : by sorry
  show (k - 2)^2 ≥ 0 from
    by apply sq_nonneg
  
-- Range of values for k when one root is less than 0
theorem part2 (k : ℝ) :
  (∃ x : ℝ, quadratic_eq k x ∧ x < 0) → k < 1 :=
by
  intro h
  cases h with x hx
  cases hx with h_eq h_lt
  have h_roots : x = k - 1 ∨ x = 1 := by sorry
  cases h_roots with h_root_km1 h_root_1
  { 
    have h_km1_lt : k - 1 < 0 := by 
      rw h_root_km1 at h_lt
      exact h_lt
    show k < 1 from by linarith
  }
  {
    have h_1_gt0 : 1 ≥ 0 := by linarith
    linarith
  }

end part1_part2_l706_706909


namespace cartesian_eq_of_C2_min_dist_MN_eq_l706_706662

noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

def C2 : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - 1)^2 + y^2 = 1

theorem cartesian_eq_of_C2 : ∀ (ρ : ℝ) (θ : ℝ), (ρ = 2 * Real.cos θ) → (ρ^2 = (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2) :=
by
  intros ρ θ h
  rw [h, ←sq_eq_iff_mono_nonneg (by norm_num : (0 : ℝ) ≤ 2 * Real.cos θ), mul_pow, pow_two, pow_two, Real.sqr_cos_add_sqr_sin, mul_one]
  apply eq.refl

theorem min_dist_MN_eq : ∀ θ : ℝ, IsMin (λ θ, Real.sqrt ((3 * Real.cos θ - 1)^2 + (2 * Real.sin θ)^2) - 1) (Real.arccos (3 / 5)) :=
by
  intros θ
  use Real.arccos (3 / 5)
  sorry

end cartesian_eq_of_C2_min_dist_MN_eq_l706_706662


namespace prove_g_neg_one_l706_706171

variables {R : Type*} [NontrivialRing R] (f g : R → R)

-- Condition 1: f(x) is an odd function
def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

-- Condition 2: g(x) = f(x) + 2
def g_def (f g : R → R) : Prop :=
  ∀ x : R, g x = f x + 2

-- Condition 3: g(1) = 1
def g_one (g : R → R) : Prop :=
  g 1 = 1

-- Prove that g(-1) = 3 under the given conditions
theorem prove_g_neg_one (f g : R → R)
  (h1 : odd_function f)
  (h2 : g_def f g)
  (h3 : g_one g) :
  g (-1) = 3 :=
sorry

end prove_g_neg_one_l706_706171


namespace count_distinct_orientations_of_cubes_l706_706910

theorem count_distinct_orientations_of_cubes :
  let num_black_cubes := 4,
      num_white_cubes := 4,
      cube_dimension := 2,
      volume := cube_dimension * cube_dimension * cube_dimension,
      rotation_equivalent (cube1 cube2 : Fin 3 → Fin 3 → Fin 3 → Prop) :=
        ∃ r : Matrix (Fin 3) (Fin 3) ℤ, is_rotation_matrix r ∧ ∀ x y z, cube1 x y z ↔ cube2 (r.mul_vec ⟨x, y, z⟩).1 
  in ∀ (cubes : Fin volume → bool), 
     (∑ i, if cubes i then 1 else 0) = num_white_cubes →
     ∃ (dist_orients : Fin volume → bool -> Fin (num_white_cubes + num_black_cubes)),
     (∃ (configs : Fin volume → bool) → 
           ∀ (cube1 cube2 : Fin volume → bool),
           rotation_equivalent 
             (λ x y z, cube1 (⟨x, y, z⟩)) 
             (λ x y z, cube2 (⟨x, y, z⟩))) :=
     7 := 
begin
  sorry
end

end count_distinct_orientations_of_cubes_l706_706910


namespace inequality_solution_a_gt_1_inequality_solution_a_lt_1_l706_706741

theorem inequality_solution_a_gt_1 (a : ℝ) (x : ℝ) 
  (ha : a > 1) (hlog : log a (2 * x - 5) > log a (x - 1)) : x > 4 :=
sorry

theorem inequality_solution_a_lt_1 (a : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : a < 1) (hlog : log a (2 * x - 5) > log a (x - 1)) : 5/2 < x ∧ x < 4 :=
sorry

end inequality_solution_a_gt_1_inequality_solution_a_lt_1_l706_706741


namespace inscribed_square_area_l706_706783

theorem inscribed_square_area (R : ℝ) (h : (R^2 * (π - 2) / 4) = (2 * π - 4)) : 
  ∃ (a : ℝ), a^2 = 16 := by
  sorry

end inscribed_square_area_l706_706783


namespace class_inspection_arrangements_l706_706034

-- Define the problem conditions
def liberal_arts_classes : ℕ := 2
def science_classes : ℕ := 4

-- Main theorem statement
theorem class_inspection_arrangements :
  ∃ (num_arrangements : ℕ), num_arrangements = 168 :=
begin
  -- The proof will use combinatorial arguments and constraints described in the problem
  sorry
end

end class_inspection_arrangements_l706_706034


namespace new_average_l706_706173

open Nat

-- The Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Sum of the first 35 Fibonacci numbers
def sum_fibonacci_first_35 : ℕ :=
  (List.range 35).map fibonacci |>.sum -- or critical to use: List.foldr (λ x acc, fibonacci x + acc) 0 (List.range 35) 

theorem new_average (n : ℕ) (avg : ℕ) (Fib_Sum : ℕ) 
  (h₁ : n = 35) 
  (h₂ : avg = 25) 
  (h₃ : Fib_Sum = sum_fibonacci_first_35) : 
  (25 * Fib_Sum / 35) = avg * (sum_fibonacci_first_35) / n := 
by 
  sorry

end new_average_l706_706173


namespace log_base_2_of_fraction_l706_706904

theorem log_base_2_of_fraction :
  logBase 2 (8 / 5) = 0.6774 := by
sorry

end log_base_2_of_fraction_l706_706904


namespace jane_quadratic_coefficients_l706_706688

theorem jane_quadratic_coefficients :
  (∃ b c : ℝ, ∀ x : ℝ, (|x-4| = 3 → (x = 1 ∨ x = 7)) ∧ (x^2 + b * x + c = 0 → (x = 1 ∨ x =7)) ∧ (b = -8 ∧ c = 7)) :=
by
  use [-8, 7]
  intros x h1 h2
  sorry

end jane_quadratic_coefficients_l706_706688


namespace coeffs_sum_coeffs_squares_diff_l706_706258

section PolynomialCoefficients

variables {a_0 a_1 a_2 a_3 a_4 a_5 : ℤ}

/-- Define the polynomial -/
def polynomial := λ x : ℤ, (2 * x - 1) ^ 5

/-- Condition: Polynomial expansion coefficients -/
def polynomial_expansion := (a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5)

/-- First proof problem: proving the sum of certain coefficients is -31-/
theorem coeffs_sum : (polynomial_expansion 1) = -31 := 
sorry

/-- Second proof problem: proving the difference of squares of sums of certain coefficients is -243-/
theorem coeffs_squares_diff : ((a_0 + a_2 + a_4) ^ 2) - ((a_1 + a_3 + a_5) ^ 2) = -243 :=
sorry

end PolynomialCoefficients

end coeffs_sum_coeffs_squares_diff_l706_706258


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l706_706172

/-- General term formula for the arithmetic sequence -/
theorem arithmetic_sequence_general_term (a d : ℕ) (h1 : a + d = 4) (h2 : a + 5 * d = 16) :
  ∃ (a_n : ℕ → ℕ), a_n = λ n => 3 * n - 2 := by
  sorry

/-- Sum of odd numbered terms of geometric sequence -/
theorem geometric_sequence_sum (b q : ℕ) (h3 : b * q ^ 2 = 4) (h4 : b * q ^ 4 = 16) 
  (n : ℕ) :
  ∃ (S_n : ℕ), S_n = ∑ i in range n, 4^i / 3 := by
  sorry

end arithmetic_sequence_general_term_geometric_sequence_sum_l706_706172


namespace license_plate_count_l706_706618

theorem license_plate_count : 
  let num_letters := 26^3 in
  let num_positions := 3 in
  let odd_choices := 5 in
  let even_choices := 5 in
  let odd_combinations := odd_choices ^ 2 in
  let even_combinations := even_choices in
  let digit_arrangement := num_positions.choose 2 in
  let total_digit_combinations := digit_arrangement * odd_combinations * even_combinations in
  let total_plates := num_letters * total_digit_combinations in
  total_plates = 6591000 :=
by
  sorry

end license_plate_count_l706_706618


namespace base_b_for_256_l706_706422

theorem base_b_for_256 (b : ℕ) : b^3 ≤ 256 ∧ 256 < b^4 ↔ b = 5 :=
by
  split
  { intro h
    cases h with h1 h2
    have : 5 ≤ b :=
      Nat.le_of_lt_succ ((Nat.lt_of_le_of_lt h1) (Nat.lt_of_lt_of_le (by norm_num) h2)) sorry
  sorry
  
  sorry

end base_b_for_256_l706_706422


namespace equation_of_trajectory_exists_fixed_point_Q_l706_706240

-- Definitions
def F1 : (ℝ × ℝ) := ( - Real.sqrt 3, 0 )
def F2 : (ℝ × ℝ) := ( Real.sqrt 3, 0 )

def is_on_trajectory (M : ℝ × ℝ) : Prop :=
  ((M.1 + Real.sqrt 3)^2 + M.2^2)^0.5 + ((M.1 - Real.sqrt 3)^2 + M.2^2)^0.5 = 4

-- Proof statement for part (1)
theorem equation_of_trajectory :
  ∀ M : ℝ × ℝ, is_on_trajectory M ↔ (M.1^2 / 4 + M.2^2 = 1) := sorry

-- Definitions for part (2)
def P : (ℝ × ℝ) := (3, 0)
def Q : (ℝ × ℝ) := (19/8, 0)
def is_on_line (k : ℝ) (l : (ℝ × ℝ) → Prop) :=
  ∀ x : ℝ, ∃ y : ℝ, l (x, y) ∧ y = k * (x - P.1)

-- Proof statement for part (2)
theorem exists_fixed_point_Q (k : ℝ) (h : k ≠ 0) :
  ∃ (Q : ℝ × ℝ), 
  (∀ A B : ℝ × ℝ, is_on_trajectory A ∧ is_on_trajectory B ∧ is_on_line k (λ P, P.2 = k * (P.1 - 3))
  → (Q.1 - A.1, Q.2 - A.2) • (Q.1 - B.1, Q.2 - B.2) = 105 / 64) := sorry

end equation_of_trajectory_exists_fixed_point_Q_l706_706240


namespace Alan_finish_time_third_task_l706_706465

theorem Alan_finish_time_third_task :
  let start_time := 480 -- 8:00 AM in minutes from midnight
  let finish_time_second_task := 675 -- 11:15 AM in minutes from midnight
  let total_tasks_time := 195 -- Total time spent on first two tasks
  let first_task_time := 65 -- Time taken for the first task calculated as per the solution
  let second_task_time := 130 -- Time taken for the second task calculated as per the solution
  let third_task_time := 65 -- Time taken for the third task
  let finish_time_third_task := 740 -- 12:20 PM in minutes from midnight
  start_time + total_tasks_time + third_task_time = finish_time_third_task :=
by
  -- proof here
  sorry

end Alan_finish_time_third_task_l706_706465


namespace smallest_twin_balanced_sum_l706_706631

def balanced_number (n : ℕ) : Prop :=
  ∃ (d1 d2 : list ℕ), d1.sum = d2.sum ∧ (d1 ++ d2).join = n.digits

def twin_balanced_numbers (n m : ℕ) : Prop :=
  n + 1 = m ∧ balanced_number n ∧ balanced_number m

theorem smallest_twin_balanced_sum :
  ∃ (n m : ℕ), twin_balanced_numbers n m ∧ n + m = 1099 :=
by {
  -- The proof will be provided here
  sorry
}

end smallest_twin_balanced_sum_l706_706631
