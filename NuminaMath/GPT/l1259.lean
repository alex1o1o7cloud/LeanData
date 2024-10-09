import Mathlib

namespace value_of_k_l1259_125936

open Real

theorem value_of_k {k : ℝ} : 
  (∃ x : ℝ, k * x ^ 2 - 2 * k * x + 4 = 0 ∧ (∀ y : ℝ, k * y ^ 2 - 2 * k * y + 4 = 0 → x = y)) → k = 4 := 
by
  intros h
  sorry

end value_of_k_l1259_125936


namespace number_of_solutions_l1259_125939

theorem number_of_solutions :
  (∀ (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 → x ≠ 0 ∧ x ≠ 5) →
  ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 :=
by
  sorry

end number_of_solutions_l1259_125939


namespace sequence_comparison_l1259_125962

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def arith_seq (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n-1) * d

theorem sequence_comparison
  (a₁ b₁ q d : ℝ)
  (h₃ : geom_seq a₁ q 3 = arith_seq b₁ d 3)
  (h₇ : geom_seq a₁ q 7 = arith_seq b₁ d 7)
  (q_pos : 0 < q)
  (d_pos : 0 < d) :
  geom_seq a₁ q 5 < arith_seq b₁ d 5 ∧
  geom_seq a₁ q 1 > arith_seq b₁ d 1 ∧
  geom_seq a₁ q 9 > arith_seq b₁ d 9 :=
by
  sorry

end sequence_comparison_l1259_125962


namespace perfect_square_A_perfect_square_D_l1259_125960

def is_even (n : ℕ) : Prop := n % 2 = 0

def A : ℕ := 2^10 * 3^12 * 7^14
def D : ℕ := 2^20 * 3^16 * 7^12

theorem perfect_square_A : ∃ k : ℕ, A = k^2 :=
by
  sorry

theorem perfect_square_D : ∃ k : ℕ, D = k^2 :=
by
  sorry

end perfect_square_A_perfect_square_D_l1259_125960


namespace isosceles_triangle_vertex_angle_l1259_125973

theorem isosceles_triangle_vertex_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = β)
  (h2: α = 70) 
  (h3 : α + β + γ = 180) : 
  γ = 40 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l1259_125973


namespace find_original_number_l1259_125928

-- Definitions of the conditions
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem find_original_number (n x y : ℕ) 
  (h1 : isFiveDigitNumber n) 
  (h2 : n = 10 * x + y) 
  (h3 : n - x = 54321) : 
  n = 60356 := 
sorry

end find_original_number_l1259_125928


namespace sum_of_squares_of_roots_of_quadratic_l1259_125934

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l1259_125934


namespace total_cost_of_purchase_l1259_125949

variable (x y z : ℝ)

theorem total_cost_of_purchase (h₁ : 4 * x + (9 / 2) * y + 12 * z = 6) (h₂ : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 :=
sorry

end total_cost_of_purchase_l1259_125949


namespace smaller_circle_x_coordinate_l1259_125922

theorem smaller_circle_x_coordinate (h : ℝ) 
  (P : ℝ × ℝ) (S : ℝ × ℝ)
  (H1 : P = (9, 12))
  (H2 : S = (h, 0))
  (r_large : ℝ)
  (r_small : ℝ)
  (H3 : r_large = 15)
  (H4 : r_small = 10) :
  S.1 = 10 ∨ S.1 = -10 := 
sorry

end smaller_circle_x_coordinate_l1259_125922


namespace determine_guilty_resident_l1259_125942

structure IslandResident where
  name : String
  is_guilty : Bool
  is_knight : Bool
  is_liar : Bool
  is_normal : Bool -- derived condition: ¬is_knight ∧ ¬is_liar

def A : IslandResident := { name := "A", is_guilty := false, is_knight := false, is_liar := false, is_normal := true }
def B : IslandResident := { name := "B", is_guilty := true, is_knight := true, is_liar := false, is_normal := false }
def C : IslandResident := { name := "C", is_guilty := false, is_knight := false, is_liar := true, is_normal := false }

-- Condition: Only one of them is guilty.
def one_guilty (A B C : IslandResident) : Prop :=
  A.is_guilty ≠ B.is_guilty ∧ A.is_guilty ≠ C.is_guilty ∧ B.is_guilty ≠ C.is_guilty ∧ (A.is_guilty ∨ B.is_guilty ∨ C.is_guilty)

-- Condition: The guilty one is a knight.
def guilty_is_knight (A B C : IslandResident) : Prop :=
  (A.is_guilty → A.is_knight) ∧ (B.is_guilty → B.is_knight) ∧ (C.is_guilty → C.is_knight)

-- Statements made by each resident.
def statements_made (A B C : IslandResident) : Prop :=
  (A.is_guilty = false) ∧ (B.is_guilty = false) ∧ (B.is_normal = false)

theorem determine_guilty_resident (A B C : IslandResident) :
  one_guilty A B C →
  guilty_is_knight A B C →
  statements_made A B C →
  B.is_guilty ∧ B.is_knight :=
by
  sorry

end determine_guilty_resident_l1259_125942


namespace jasmine_paperclips_l1259_125943

theorem jasmine_paperclips :
  ∃ k : ℕ, (4 * 3^k > 500) ∧ (∀ n < k, 4 * 3^n ≤ 500) ∧ k = 5 ∧ (n = 6) :=
by {
  sorry
}

end jasmine_paperclips_l1259_125943


namespace percentage_of_all_students_with_cars_l1259_125986

def seniors := 300
def percent_seniors_with_cars := 0.40
def lower_grades := 1500
def percent_lower_grades_with_cars := 0.10

theorem percentage_of_all_students_with_cars :
  (120 + 150) / 1800 * 100 = 15 := by
  sorry

end percentage_of_all_students_with_cars_l1259_125986


namespace area_of_rectangle_l1259_125995

theorem area_of_rectangle (s : ℝ) (h1 : 4 * s = 100) : 2 * s * 2 * s = 2500 := by
  sorry

end area_of_rectangle_l1259_125995


namespace part1_part2_l1259_125930

variables (a b : ℝ) (f g : ℝ → ℝ)

-- Step 1: Given a > 0, b > 0 and f(x) = |x - a| - |x + b|, prove that if max(f) = 3, then a + b = 3.
theorem part1 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3) :
  a + b = 3 :=
sorry

-- Step 2: For g(x) = -x^2 - ax - b, if g(x) < f(x) for all x ≥ a, prove that 1/2 < a < 3.
theorem part2 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3)
    (hg : ∀ x, g x = -x^2 - a * x - b) (hcond : ∀ x, x ≥ a → g x < f x) :
    1 / 2 < a ∧ a < 3 :=
sorry

end part1_part2_l1259_125930


namespace ratio_of_cone_to_sphere_l1259_125964

theorem ratio_of_cone_to_sphere (r : ℝ) (h := 2 * r) : 
  (1 / 3 * π * r^2 * h) / ((4 / 3) * π * r^3) = 1 / 2 :=
by 
  sorry

end ratio_of_cone_to_sphere_l1259_125964


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l1259_125902

variable (a : ℕ → ℝ)

-- Conditions
def sequence_positive : Prop :=
  ∀ n, a n > 0

def recurrence_relation : Prop :=
  ∀ n, a (n + 1) ^ 2 - a (n + 1) = a n

-- Correct conclusions to prove:

-- Conclusion ①
theorem conclusion_1 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ∀ n ≥ 2, a n > 1 := 
sorry

-- Conclusion ②
theorem conclusion_2 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ¬∀ n, a n = a (n + 1) := 
sorry

-- Conclusion ③
theorem conclusion_3 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h3 : 0 < a 1 ∧ a 1 < 2) :
  ∀ n, a (n + 1) > a n :=
sorry

-- Conclusion ④
theorem conclusion_4 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h4 : a 1 > 2) :
  ∀ n ≥ 2, 2 < a n ∧ a n < a 1 :=
sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l1259_125902


namespace muffin_cost_relation_l1259_125910

variable (m b : ℝ)

variable (S := 5 * m + 4 * b)
variable (C := 10 * m + 18 * b)

theorem muffin_cost_relation (h1 : C = 3 * S) : m = 1.2 * b :=
  sorry

end muffin_cost_relation_l1259_125910


namespace product_of_equal_numbers_l1259_125911

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l1259_125911


namespace difference_between_two_numbers_l1259_125948

theorem difference_between_two_numbers : 
  ∃ (a b : ℕ),
    (a + b = 21780) ∧
    (a % 5 = 0) ∧
    ((a / 10) = b) ∧
    (a - b = 17825) :=
sorry

end difference_between_two_numbers_l1259_125948


namespace strictly_increasing_range_l1259_125987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x + 1 else a ^ x

theorem strictly_increasing_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 < a ∧ a ≤ 2) :=
sorry

end strictly_increasing_range_l1259_125987


namespace remainder_div_1356_l1259_125981

theorem remainder_div_1356 :
  ∃ R : ℝ, ∃ L : ℝ, ∃ S : ℝ, S = 268.2 ∧ L - S = 1356 ∧ L = 6 * S + R ∧ R = 15 :=
by
  sorry

end remainder_div_1356_l1259_125981


namespace a_finishes_job_in_60_days_l1259_125997

theorem a_finishes_job_in_60_days (A B : ℝ)
  (h1 : A + B = 1 / 30)
  (h2 : 20 * (A + B) = 2 / 3)
  (h3 : 20 * A = 1 / 3) :
  1 / A = 60 :=
by sorry

end a_finishes_job_in_60_days_l1259_125997


namespace compute_diff_of_squares_l1259_125944

theorem compute_diff_of_squares : (65^2 - 35^2 = 3000) :=
by
  sorry

end compute_diff_of_squares_l1259_125944


namespace problem_solution_l1259_125965

theorem problem_solution (p q r : ℝ) 
    (h1 : (p * r / (p + q) + q * p / (q + r) + r * q / (r + p)) = -8)
    (h2 : (q * r / (p + q) + r * p / (q + r) + p * q / (r + p)) = 9) 
    : (q / (p + q) + r / (q + r) + p / (r + p) = 10) := 
by
  sorry

end problem_solution_l1259_125965


namespace numberOfHandshakes_is_correct_l1259_125966

noncomputable def numberOfHandshakes : ℕ :=
  let gremlins := 30
  let imps := 20
  let friendlyImps := 5
  let gremlinHandshakes := gremlins * (gremlins - 1) / 2
  let impGremlinHandshakes := imps * gremlins
  let friendlyImpHandshakes := friendlyImps * (friendlyImps - 1) / 2
  gremlinHandshakes + impGremlinHandshakes + friendlyImpHandshakes

theorem numberOfHandshakes_is_correct : numberOfHandshakes = 1045 := by
  sorry

end numberOfHandshakes_is_correct_l1259_125966


namespace Q_at_one_is_zero_l1259_125998

noncomputable def Q (x : ℚ) : ℚ := x^4 - 2 * x^2 + 1

theorem Q_at_one_is_zero :
  Q 1 = 0 :=
by
  -- Here we would put the formal proof in Lean language
  sorry

end Q_at_one_is_zero_l1259_125998


namespace equality_proof_l1259_125950

variable {a b c : ℝ}

theorem equality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (1 / 2) * (a + b + c) :=
by
  sorry

end equality_proof_l1259_125950


namespace path_problem_l1259_125992

noncomputable def path_bounds (N : ℕ) (h : 0 < N) : Prop :=
  ∃ p : ℕ, 4 * N ≤ p ∧ p ≤ 2 * N^2 + 2 * N

theorem path_problem (N : ℕ) (h : 0 < N) : path_bounds N h :=
  sorry

end path_problem_l1259_125992


namespace arithmetic_sequence_l1259_125940

-- Given conditions
variables {a x b : ℝ}

-- Statement of the problem in Lean 4
theorem arithmetic_sequence (h1 : x - a = b - x) (h2 : b - x = 2 * x - b) : a / b = 1 / 3 :=
sorry

end arithmetic_sequence_l1259_125940


namespace completing_the_square_l1259_125941

theorem completing_the_square (x : ℝ) :
  x^2 - 6 * x + 2 = 0 →
  (x - 3)^2 = 7 :=
by sorry

end completing_the_square_l1259_125941


namespace rectangle_length_increase_decrease_l1259_125983

theorem rectangle_length_increase_decrease
  (L : ℝ)
  (width : ℝ)
  (increase_percentage : ℝ)
  (decrease_percentage : ℝ)
  (new_width : ℝ)
  (initial_area : ℝ)
  (new_length : ℝ)
  (new_area : ℝ)
  (HLW : width = 40)
  (Hinc : increase_percentage = 0.30)
  (Hdec : decrease_percentage = 0.17692307692307693)
  (Hnew_width : new_width = 40 - (decrease_percentage * 40))
  (Hinitial_area : initial_area = L * 40)
  (Hnew_length : new_length = 1.30 * L)
  (Hequal_area : new_length * new_width = L * 40) :
  L = 30.76923076923077 :=
by
  sorry

end rectangle_length_increase_decrease_l1259_125983


namespace problem_solution_l1259_125912

noncomputable def set_M (x : ℝ) : Prop := x^2 - 4*x < 0
noncomputable def set_N (m x : ℝ) : Prop := m < x ∧ x < 5
noncomputable def set_intersection (x : ℝ) : Prop := 3 < x ∧ x < 4

theorem problem_solution (m n : ℝ) :
  (∀ x, set_M x ↔ (0 < x ∧ x < 4)) →
  (∀ x, set_N m x ↔ (m < x ∧ x < 5)) →
  (∀ x, (set_M x ∧ set_N m x) ↔ set_intersection x) →
  m + n = 7 :=
by
  intros H1 H2 H3
  sorry

end problem_solution_l1259_125912


namespace sum_of_squares_of_roots_eq_1853_l1259_125925

theorem sum_of_squares_of_roots_eq_1853
  (α β : ℕ) (h_prime_α : Prime α) (h_prime_beta : Prime β) (h_sum : α + β = 45)
  (h_quadratic_eq : ∀ x, x^2 - 45*x + α*β = 0 → x = α ∨ x = β) :
  α^2 + β^2 = 1853 := 
by
  sorry

end sum_of_squares_of_roots_eq_1853_l1259_125925


namespace find_range_of_a_l1259_125980

noncomputable def A (a : ℝ) := { x : ℝ | 1 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = 5 * x - 6 ∧ 1 ≤ x ∧ x ≤ a }
noncomputable def C (a : ℝ) := { m : ℝ | ∃ x : ℝ, m = x^2 ∧ 1 ≤ x ∧ x ≤ a }

theorem find_range_of_a (a : ℝ) (h : B a ∩ C a = C a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end find_range_of_a_l1259_125980


namespace sequence_sum_property_l1259_125914

theorem sequence_sum_property {a S : ℕ → ℚ} (h1 : a 1 = 3/2)
  (h2 : ∀ n : ℕ, 2 * a (n + 1) + S n = 3) :
  (∀ n : ℕ, a n = 3 * (1/2)^n) ∧
  (∃ (n_max : ℕ),  (∀ n : ℕ, n ≤ n_max → (S n = 3 * (1 - (1/2)^n)) ∧ ∀ n : ℕ, (S (2 * n)) / (S n) > 64 / 63 → n_max = 5)) :=
by {
  -- The proof would go here
  sorry
}

end sequence_sum_property_l1259_125914


namespace solve_for_x_l1259_125953

theorem solve_for_x (x : ℝ) (h : (x^2 + 2*x + 3) / (x + 1) = x + 3) : x = 0 :=
by
  sorry

end solve_for_x_l1259_125953


namespace arrangement_count_l1259_125993

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l1259_125993


namespace percentage_increase_in_efficiency_l1259_125926

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l1259_125926


namespace apple_baskets_l1259_125909

theorem apple_baskets (total_apples : ℕ) (apples_per_basket : ℕ) (total_apples_eq : total_apples = 495) (apples_per_basket_eq : apples_per_basket = 25) :
  total_apples / apples_per_basket = 19 :=
by
  sorry

end apple_baskets_l1259_125909


namespace intersection_eq_l1259_125979

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def C : Set ℝ := {2}

theorem intersection_eq : A ∩ B = C := 
by {
  sorry
}

end intersection_eq_l1259_125979


namespace product_of_consecutive_even_numbers_divisible_by_24_l1259_125947

theorem product_of_consecutive_even_numbers_divisible_by_24 (n : ℕ) :
  (2 * n) * (2 * n + 2) * (2 * n + 4) % 24 = 0 :=
  sorry

end product_of_consecutive_even_numbers_divisible_by_24_l1259_125947


namespace min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l1259_125917

-- Condition 1: Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- Proof Problem 1: Minimum value of f(x) when a = 1
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f x 1 ≥ 2) :=
sorry

-- Proof Problem 2: Range of values for a when f(x) ≤ 3 has solutions
theorem range_of_a_if_f_leq_3_non_empty : 
  (∃ x : ℝ, f x a ≤ 3) → abs (3 - a) ≤ 3 :=
sorry

end min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l1259_125917


namespace matching_pair_probability_l1259_125975

-- Given conditions
def total_gray_socks : ℕ := 12
def total_white_socks : ℕ := 10
def total_socks : ℕ := total_gray_socks + total_white_socks

-- Proof statement
theorem matching_pair_probability (h_grays : total_gray_socks = 12) (h_whites : total_white_socks = 10) :
  (66 + 45) / (total_socks.choose 2) = 111 / 231 :=
by
  sorry

end matching_pair_probability_l1259_125975


namespace factorial_inequality_l1259_125982

theorem factorial_inequality (n : ℕ) (h : n > 1) : n! < ( (n + 1) / 2 )^n := by
  sorry

end factorial_inequality_l1259_125982


namespace fraction_dehydrated_l1259_125956

theorem fraction_dehydrated (total_men tripped fraction_dnf finished : ℕ) (fraction_tripped fraction_dehydrated_dnf : ℚ)
  (htotal_men : total_men = 80)
  (hfraction_tripped : fraction_tripped = 1 / 4)
  (htripped : tripped = total_men * fraction_tripped)
  (hfinished : finished = 52)
  (hfraction_dnf : fraction_dehydrated_dnf = 1 / 5)
  (hdnf : total_men - finished = tripped + fraction_dehydrated_dnf * (total_men - tripped) * x)
  (hx : x = 2 / 3) :
  x = 2 / 3 := sorry

end fraction_dehydrated_l1259_125956


namespace simplify_and_evaluate_l1259_125994

variable (x y : ℤ)

noncomputable def given_expr := (x + y) ^ 2 - 3 * x * (x + y) + (x + 2 * y) * (x - 2 * y)

theorem simplify_and_evaluate : given_expr 1 (-1) = -3 :=
by
  -- The proof is to be completed here
  sorry

end simplify_and_evaluate_l1259_125994


namespace caterpillar_to_scorpion_ratio_l1259_125900

theorem caterpillar_to_scorpion_ratio 
  (roach_count : ℕ) (scorpion_count : ℕ) (total_insects : ℕ) 
  (h_roach : roach_count = 12) 
  (h_scorpion : scorpion_count = 3) 
  (h_cricket : cricket_count = roach_count / 2) 
  (h_total : total_insects = 27) 
  (h_non_cricket_count : non_cricket_count = roach_count + scorpion_count + cricket_count) 
  (h_caterpillar_count : caterpillar_count = total_insects - non_cricket_count) : 
  (caterpillar_count / scorpion_count) = 2 := 
by 
  sorry

end caterpillar_to_scorpion_ratio_l1259_125900


namespace product_of_zero_multiples_is_equal_l1259_125984

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end product_of_zero_multiples_is_equal_l1259_125984


namespace notebook_cost_l1259_125927

-- Define the cost of notebook (n) and cost of cover (c)
variables (n c : ℝ)

-- Given conditions as definitions
def condition1 := n + c = 3.50
def condition2 := n = c + 2

-- Prove that the cost of the notebook (n) is 2.75
theorem notebook_cost (h1 : condition1 n c) (h2 : condition2 n c) : n = 2.75 := 
by
  sorry

end notebook_cost_l1259_125927


namespace smallest_integer_to_make_square_l1259_125976

noncomputable def y : ℕ := 2^37 * 3^18 * 5^6 * 7^8

theorem smallest_integer_to_make_square : ∃ z : ℕ, z = 10 ∧ ∃ k : ℕ, (y * z) = k^2 :=
by
  sorry

end smallest_integer_to_make_square_l1259_125976


namespace total_trees_in_gray_regions_l1259_125905

theorem total_trees_in_gray_regions (trees_rectangle1 trees_rectangle2 trees_rectangle3 trees_gray1 trees_gray2 trees_total : ℕ)
  (h1 : trees_rectangle1 = 100)
  (h2 : trees_rectangle2 = 90)
  (h3 : trees_rectangle3 = 82)
  (h4 : trees_total = 82)
  (h_gray1 : trees_gray1 = trees_rectangle1 - trees_total)
  (h_gray2 : trees_gray2 = trees_rectangle2 - trees_total)
  : trees_gray1 + trees_gray2 = 26 := 
sorry

end total_trees_in_gray_regions_l1259_125905


namespace minimum_disks_needed_l1259_125989

-- Definition of the conditions
def disk_capacity : ℝ := 2.88
def file_sizes : List (ℝ × ℕ) := [(1.2, 5), (0.9, 10), (0.6, 8), (0.3, 7)]

/-- 
Theorem: Given the capacity of each disk and the sizes and counts of different files,
we can prove that the minimum number of disks needed to store all the files without 
splitting any file is 14.
-/
theorem minimum_disks_needed (capacity : ℝ) (files : List (ℝ × ℕ)) : 
  capacity = disk_capacity ∧ files = file_sizes → ∃ m : ℕ, m = 14 :=
by
  sorry

end minimum_disks_needed_l1259_125989


namespace sector_area_l1259_125937

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 120) : 
  (theta / 360) * π * r^2 = 3 * π :=
by 
  sorry

end sector_area_l1259_125937


namespace find_number_l1259_125990

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l1259_125990


namespace board_game_cost_correct_l1259_125945

-- Definitions
def jump_rope_cost : ℕ := 7
def ball_cost : ℕ := 4
def saved_money : ℕ := 6
def gift_money : ℕ := 13
def needed_money : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_money + gift_money

-- Total cost of all items
def total_cost : ℕ := total_money + needed_money

-- Combined cost of jump rope and ball
def combined_cost_jump_rope_ball : ℕ := jump_rope_cost + ball_cost

-- Cost of the board game
def board_game_cost : ℕ := total_cost - combined_cost_jump_rope_ball

-- Theorem to prove
theorem board_game_cost_correct : board_game_cost = 12 :=
by 
  -- Proof omitted
  sorry

end board_game_cost_correct_l1259_125945


namespace tetrahedron_volume_minimum_l1259_125919

theorem tetrahedron_volume_minimum (h1 h2 h3 : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (h3_pos : 0 < h3) :
  ∃ V : ℝ, V ≥ (1/3) * (h1 * h2 * h3) :=
sorry

end tetrahedron_volume_minimum_l1259_125919


namespace intersection_complement_eq_l1259_125958

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set M within U
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N within U
def N : Set ℕ := {5, 6, 7}

-- Define the complement of M in U
def CU_M : Set ℕ := U \ M

-- Define the complement of N in U
def CU_N : Set ℕ := U \ N

-- Mathematically equivalent proof problem
theorem intersection_complement_eq : CU_M ∩ CU_N = {2, 4, 8} := by
  sorry

end intersection_complement_eq_l1259_125958


namespace intersection_A_complement_B_eq_interval_l1259_125955

-- We define universal set U as ℝ
def U := Set ℝ

-- Definitions provided in the problem
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y >= 2 }

-- Complement of B in U
def C_U_B : Set ℝ := { y | y < 2 }

-- Now we state the theorem
theorem intersection_A_complement_B_eq_interval :
  A ∩ C_U_B = { x | 1 < x ∧ x < 2 } :=
by 
  sorry

end intersection_A_complement_B_eq_interval_l1259_125955


namespace determine_x_l1259_125969

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l1259_125969


namespace minute_hand_gain_per_hour_l1259_125977

theorem minute_hand_gain_per_hour (h_start h_end : ℕ) (time_elapsed : ℕ) 
  (total_gain : ℕ) (gain_per_hour : ℕ) 
  (h_start_eq_9 : h_start = 9)
  (time_period_eq_8 : time_elapsed = 8)
  (total_gain_eq_40 : total_gain = 40)
  (time_elapsed_eq : h_end = h_start + time_elapsed)
  (gain_formula : gain_per_hour * time_elapsed = total_gain) :
  gain_per_hour = 5 := 
by 
  sorry

end minute_hand_gain_per_hour_l1259_125977


namespace first_place_team_ties_l1259_125931

noncomputable def teamPoints (wins ties: ℕ) : ℕ := 2 * wins + ties

theorem first_place_team_ties {T : ℕ} : 
  teamPoints 13 1 + teamPoints 8 10 + teamPoints 12 T = 81 → T = 4 :=
by
  sorry

end first_place_team_ties_l1259_125931


namespace sum_of_squares_of_rates_l1259_125938

theorem sum_of_squares_of_rates (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 66) 
  (h2 : 3 * j + 2 * s + 4 * b = 96) : 
  b^2 + j^2 + s^2 = 612 := 
by 
  sorry

end sum_of_squares_of_rates_l1259_125938


namespace solve_dividend_and_divisor_l1259_125991

-- Definitions for base, digits, and mathematical relationships
def base := 5
def P := 1
def Q := 2
def R := 3
def S := 4
def T := 0
def Dividend := 1 * base^6 + 2 * base^5 + 3 * base^4 + 4 * base^3 + 3 * base^2 + 2 * base^1 + 1 * base^0
def Divisor := 2 * base^2 + 3 * base^1 + 2 * base^0

-- The conditions given in the math problem
axiom condition_1 : Q + R = base
axiom condition_2 : P + 1 = Q
axiom condition_3 : Q + P = R
axiom condition_4 : S = 2 * Q
axiom condition_5 : Q^2 = S
axiom condition_6 : Dividend = 24336
axiom condition_7 : Divisor = 67

-- The goal
theorem solve_dividend_and_divisor : Dividend = 24336 ∧ Divisor = 67 :=
by {
  sorry
}

end solve_dividend_and_divisor_l1259_125991


namespace calculate_expression_l1259_125968

theorem calculate_expression : (-1) ^ 47 + 2 ^ (3 ^ 3 + 4 ^ 2 - 6 ^ 2) = 127 := 
by 
  sorry

end calculate_expression_l1259_125968


namespace probability_inside_circle_is_2_div_9_l1259_125929

noncomputable def probability_point_in_circle : ℚ := 
  let total_points := 36
  let points_inside := 8
  points_inside / total_points

theorem probability_inside_circle_is_2_div_9 :
  probability_point_in_circle = 2 / 9 :=
by
  -- we acknowledge the mathematical computation here
  sorry

end probability_inside_circle_is_2_div_9_l1259_125929


namespace no_possible_k_l1259_125933
open Classical

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_possible_k : 
  ∀ (k : ℕ), 
    (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ (p + q = 74) ∧ (x^2 - 74*x + k = 0)) -> False :=
by sorry

end no_possible_k_l1259_125933


namespace polynomial_has_integer_root_l1259_125907

noncomputable def P : Polynomial ℤ := sorry

theorem polynomial_has_integer_root
  (P : Polynomial ℤ)
  (h_deg : P.degree = 3)
  (h_infinite_sol : ∀ (x y : ℤ), x ≠ y → x * P.eval x = y * P.eval y → 
  ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y) :
  ∃ k : ℤ, P.eval k = 0 :=
sorry

end polynomial_has_integer_root_l1259_125907


namespace max_jogs_possible_l1259_125961

theorem max_jogs_possible :
  ∃ (x y z : ℕ), (3 * x + 4 * y + 10 * z = 100) ∧ (x + y + z ≥ 20) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧
  (∀ (x' y' z' : ℕ), (3 * x' + 4 * y' + 10 * z' = 100) ∧ (x' + y' + z' ≥ 20) ∧ (x' ≥ 1) ∧ (y' ≥ 1) ∧ (z' ≥ 1) → z' ≤ z) :=
by
  sorry

end max_jogs_possible_l1259_125961


namespace lunks_needed_for_20_apples_l1259_125971

-- Definitions based on given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks / 3) * 5

-- The main statement to be proven
theorem lunks_needed_for_20_apples :
  ∃ l : ℕ, (kunks_to_apples (lunks_to_kunks l)) = 20 ∧ l = 24 :=
by
  sorry

end lunks_needed_for_20_apples_l1259_125971


namespace prove_two_minus_a_l1259_125978

theorem prove_two_minus_a (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := 
by 
  sorry

end prove_two_minus_a_l1259_125978


namespace fencing_cost_per_meter_l1259_125946

-- Definitions based on given conditions
def area : ℚ := 1200
def short_side : ℚ := 30
def total_cost : ℚ := 1800

-- Definition to represent the length of the long side
def long_side := area / short_side

-- Definition to represent the diagonal of the rectangle
def diagonal := (long_side^2 + short_side^2).sqrt

-- Definition to represent the total length of the fence
def total_length := long_side + short_side + diagonal

-- Definition to represent the cost per meter
def cost_per_meter := total_cost / total_length

-- Theorem statement asserting that cost_per_meter == 15
theorem fencing_cost_per_meter : cost_per_meter = 15 := 
by 
  sorry

end fencing_cost_per_meter_l1259_125946


namespace max_non_managers_l1259_125954

theorem max_non_managers (N : ℕ) (h : (9:ℝ) / (N:ℝ) > (7:ℝ) / (32:ℝ)) : N ≤ 41 :=
by
  -- Proof skipped
  sorry

end max_non_managers_l1259_125954


namespace hannah_jerry_difference_l1259_125999

-- Define the calculations of Hannah (H) and Jerry (J)
def H : Int := 10 - (3 * 4)
def J : Int := 10 - 3 + 4

-- Prove that H - J = -13
theorem hannah_jerry_difference : H - J = -13 := by
  sorry

end hannah_jerry_difference_l1259_125999


namespace value_of_y_at_48_l1259_125996

open Real

noncomputable def collinear_points (x : ℝ) : ℝ :=
  if x = 2 then 5
  else if x = 6 then 17
  else if x = 10 then 29
  else if x = 48 then 143
  else 0 -- placeholder value for other x (not used in proof)

theorem value_of_y_at_48 :
  (∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 
    ∃ (m : ℝ), m = (collinear_points x2 - collinear_points x1) / (x2 - x1) ∧ 
               m = (collinear_points x3 - collinear_points x2) / (x3 - x2)) →
  collinear_points 48 = 143 :=
by
  sorry

end value_of_y_at_48_l1259_125996


namespace eliot_account_balance_l1259_125935

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A - E = (1/12) * (A + E))
  (h2 : A * 1.10 = E * 1.15 + 30) :
  E = 857.14 := by
  sorry

end eliot_account_balance_l1259_125935


namespace area_of_original_square_l1259_125903

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end area_of_original_square_l1259_125903


namespace big_white_toys_l1259_125901

/-- A store has two types of toys, Big White and Little Yellow, with a total of 60 toys.
    The price ratio of Big White to Little Yellow is 6:5.
    Selling all of them results in a total of 2016 yuan.
    We want to determine how many Big Whites there are. -/
theorem big_white_toys (x k : ℕ) (h1 : 6 * x + 5 * (60 - x) = 2016) (h2 : k = 6) : x = 36 :=
by
  sorry

end big_white_toys_l1259_125901


namespace maxwell_meets_brad_l1259_125970

-- Define the given conditions
def distance_between_homes : ℝ := 94
def maxwell_speed : ℝ := 4
def brad_speed : ℝ := 6
def time_delay : ℝ := 1

-- Define the total time it takes Maxwell to meet Brad
theorem maxwell_meets_brad : ∃ t : ℝ, maxwell_speed * (t + time_delay) + brad_speed * t = distance_between_homes ∧ (t + time_delay = 10) :=
by
  sorry

end maxwell_meets_brad_l1259_125970


namespace failed_both_l1259_125932

-- Defining the conditions based on the problem statement
def failed_hindi : ℝ := 0.34
def failed_english : ℝ := 0.44
def passed_both : ℝ := 0.44

-- Defining a proposition to represent the problem and its solution
theorem failed_both (x : ℝ) (h1 : x = failed_hindi + failed_english - (1 - passed_both)) : 
  x = 0.22 :=
by
  sorry

end failed_both_l1259_125932


namespace ellipse_foci_on_y_axis_l1259_125908

theorem ellipse_foci_on_y_axis (theta : ℝ) (h1 : 0 < theta ∧ theta < π)
  (h2 : Real.sin theta + Real.cos theta = 1 / 2) :
  (0 < theta ∧ theta < π / 2) → 
  (0 < theta ∧ theta < 3 * π / 4) → 
  -- The equation x^2 * sin theta - y^2 * cos theta = 1 represents an ellipse with foci on the y-axis
  ∃ foci_on_y_axis : Prop, foci_on_y_axis := 
sorry

end ellipse_foci_on_y_axis_l1259_125908


namespace arithmetic_sequence_9th_term_l1259_125913

variables {a_n : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_9th_term
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 3 = 6)
  (h2 : a 6 = 3)
  (h_seq : arithmetic_sequence a d) :
  a 9 = 0 :=
sorry

end arithmetic_sequence_9th_term_l1259_125913


namespace additional_track_length_needed_l1259_125985

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed_l1259_125985


namespace coefficient_x3_in_expansion_l1259_125924

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3_in_expansion :
  let x := 1
  let a := x
  let b := 2
  let n := 50
  let k := 47
  let coefficient := binom n (n - k) * b^k
  coefficient = 19600 * 2^47 := by
  sorry

end coefficient_x3_in_expansion_l1259_125924


namespace usual_time_is_60_l1259_125906

variable (S T T' D : ℝ)

-- Defining the conditions
axiom condition1 : T' = T + 12
axiom condition2 : D = S * T
axiom condition3 : D = (5 / 6) * S * T'

-- The theorem to prove
theorem usual_time_is_60 (S T T' D : ℝ) 
  (h1 : T' = T + 12)
  (h2 : D = S * T)
  (h3 : D = (5 / 6) * S * T') : T = 60 := 
sorry

end usual_time_is_60_l1259_125906


namespace remainder_div_x_plus_1_l1259_125951

noncomputable def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_div_x_plus_1 : 
  (f (-1) = 4) := 
by
  sorry

end remainder_div_x_plus_1_l1259_125951


namespace negation_correct_l1259_125957

-- Define the statement to be negated
def original_statement (x : ℕ) : Prop := ∀ x : ℕ, x^2 ≠ 4

-- Define the negation of the original statement
def negated_statement (x : ℕ) : Prop := ∃ x : ℕ, x^2 = 4

-- Prove that the negation of the original statement is the given negated statement
theorem negation_correct : (¬ (∀ x : ℕ, x^2 ≠ 4)) ↔ (∃ x : ℕ, x^2 = 4) :=
by sorry

end negation_correct_l1259_125957


namespace shopkeeper_gain_l1259_125904

theorem shopkeeper_gain
  (true_weight : ℝ)
  (cheat_percent : ℝ)
  (gain_percent : ℝ) :
  cheat_percent = 0.1 ∧
  true_weight = 1000 →
  gain_percent = 20 :=
by
  sorry

end shopkeeper_gain_l1259_125904


namespace min_rectilinear_distance_l1259_125988

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem min_rectilinear_distance : ∀ (M : ℝ × ℝ), (M.1 - M.2 + 4 = 0) → rectilinear_distance (1, 1) M ≥ 4 :=
by
  intro M hM
  -- We only need the statement, not the proof
  sorry

end min_rectilinear_distance_l1259_125988


namespace mul_103_97_l1259_125915

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l1259_125915


namespace solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l1259_125963

-- Problem 1: Solution set for the inequality \( f(x) ≤ 6 \)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Problem 2: Prove \( a^2 + b^2 + c^2 ≥ 16/3 \)
variables (a b c : ℝ)
axiom pos_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom sum_abc : a + b + c = 4

theorem sum_of_squares_geq_16_div_3 :
  a^2 + b^2 + c^2 ≥ 16 / 3 :=
sorry

end solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l1259_125963


namespace alfonzo_visit_l1259_125920

-- Define the number of princes (palaces) as n
variable (n : ℕ)

-- Define the type of connections (either a "Ruelle" or a "Canal")
inductive Transport
| Ruelle
| Canal

-- Define the connection between any two palaces
noncomputable def connection (i j : ℕ) : Transport := sorry

-- The theorem states that Prince Alfonzo can visit all his friends using only one type of transportation
theorem alfonzo_visit (h : ∀ i j, i ≠ j → ∃ t : Transport, ∀ k, k ≠ i → connection i k = t) :
  ∃ t : Transport, ∀ i j, i ≠ j → connection i j = t :=
sorry

end alfonzo_visit_l1259_125920


namespace min_value_of_quadratic_l1259_125959

theorem min_value_of_quadratic :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∃ c : ℝ, c = c → z = 12) :=
by
  sorry

end min_value_of_quadratic_l1259_125959


namespace simplify_and_evaluate_expression_l1259_125952

variable (x y : ℝ)

theorem simplify_and_evaluate_expression (h₁ : x = -2) (h₂ : y = 1/2) :
  (x + 2 * y) ^ 2 - (x + y) * (3 * x - y) - 5 * y ^ 2 / (2 * x) = 2 + 1 / 2 := 
sorry

end simplify_and_evaluate_expression_l1259_125952


namespace simplify_fraction_1_simplify_fraction_2_l1259_125972

variables (a b c : ℝ)

theorem simplify_fraction_1 :
  (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c) :=
sorry

theorem simplify_fraction_2 :
  (a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c) :=
sorry

end simplify_fraction_1_simplify_fraction_2_l1259_125972


namespace principal_calc_l1259_125923

noncomputable def principal (r : ℝ) : ℝ :=
  (65000 : ℝ) / r

theorem principal_calc (P r : ℝ) (h : 0 < r) :
    (P * 0.10 + P * 1.10 * r / 100 - P * (0.10 + r / 100) = 65) → 
    P = principal r :=
by
  sorry

end principal_calc_l1259_125923


namespace polynomial_remainder_l1259_125921

-- Define the polynomial
def poly (x : ℝ) : ℝ := 3 * x^8 - x^7 - 7 * x^5 + 3 * x^3 + 4 * x^2 - 12 * x - 1

-- Define the divisor
def divisor : ℝ := 3

-- State the theorem
theorem polynomial_remainder :
  poly divisor = 15951 :=
by
  -- Proof omitted, to be filled in later
  sorry

end polynomial_remainder_l1259_125921


namespace expected_value_is_350_l1259_125974

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l1259_125974


namespace cafeteria_extra_fruits_l1259_125967

def num_apples_red := 75
def num_apples_green := 35
def num_oranges := 40
def num_bananas := 20
def num_students := 17

def total_fruits := num_apples_red + num_apples_green + num_oranges + num_bananas
def fruits_taken_by_students := num_students
def extra_fruits := total_fruits - fruits_taken_by_students

theorem cafeteria_extra_fruits : extra_fruits = 153 := by
  -- proof goes here
  sorry

end cafeteria_extra_fruits_l1259_125967


namespace no_natural_number_solution_l1259_125916

theorem no_natural_number_solution :
  ¬∃ (n : ℕ), ∃ (k : ℕ), (n^5 - 5*n^3 + 4*n + 7 = k^2) :=
sorry

end no_natural_number_solution_l1259_125916


namespace find_f_1988_l1259_125918

namespace FunctionalEquation

def f (n : ℕ) : ℕ :=
  sorry -- definition placeholder, since we only need the statement

axiom f_properties (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem find_f_1988 (h : ∀ n : ℕ, 0 < n → f n = n) : f 1988 = 1988 :=
  sorry

end FunctionalEquation

end find_f_1988_l1259_125918
