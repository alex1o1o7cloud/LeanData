import Mathlib

namespace negative_integer_solutions_l1025_102557

theorem negative_integer_solutions (x : ℤ) : 3 * x + 1 ≥ -5 ↔ x = -2 ∨ x = -1 := 
by
  sorry

end negative_integer_solutions_l1025_102557


namespace initial_average_quiz_score_l1025_102565

theorem initial_average_quiz_score 
  (n : ℕ) (A : ℝ) (dropped_avg : ℝ) (drop_score : ℝ)
  (students_before : n = 16)
  (students_after : n - 1 = 15)
  (dropped_avg_eq : dropped_avg = 64.0)
  (drop_score_eq : drop_score = 8) 
  (total_sum_before_eq : n * A = 16 * A)
  (total_sum_after_eq : (n - 1) * dropped_avg = 15 * 64):
  A = 60.5 := 
by
  sorry

end initial_average_quiz_score_l1025_102565


namespace check_correct_options_l1025_102590

noncomputable def f (x a b: ℝ) := x^3 - a*x^2 + b*x + 1

theorem check_correct_options :
  (∀ (b: ℝ), b = 0 → ¬(∃ x: ℝ, 3 * x^2 - 2 * a * x = 0)) ∧
  (∀ (a: ℝ), a = 0 → (∀ x: ℝ, f x a b + f (-x) a b = 2)) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), b = a^2 / 4 ∧ a > -4 → ∃ x1 x2 x3: ℝ, f x1 a b = 0 ∧ f x2 a b = 0 ∧ f x3 a b = 0) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), (∀ x: ℝ, 3 * x^2 - 2 * a * x + b ≥ 0) → (a^2 ≤ 3*b)) := sorry

end check_correct_options_l1025_102590


namespace GE_eq_GH_l1025_102537

variables (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
          [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
          
variables (AC : Line A C) (AB : Line A B) (BE : Line B E) (DE : Line D E)
          (BG : Line B G) (AF : Line A F) (DE' : Line D E') (angleC : Angle C = 90)

variables (circB : Circle B BC) (tangentDE : Tangent DE circB E) (perpAB : Perpendicular AC AB)
          (intersectionF : Intersect (PerpendicularLine C AB) BE F)
          (intersectionG : Intersect AF DE G) (intersectionH : Intersect (ParallelLine A BG) DE H)

theorem GE_eq_GH : GE = GH := sorry

end GE_eq_GH_l1025_102537


namespace student_chose_124_l1025_102549

theorem student_chose_124 (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 := 
by {
  sorry
}

end student_chose_124_l1025_102549


namespace approx_values_relationship_l1025_102599

theorem approx_values_relationship : 
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a = b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a > b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a < b) :=
by sorry

end approx_values_relationship_l1025_102599


namespace part1_part2_l1025_102512

variable (a b : ℝ)

theorem part1 (h : |a - 3| + |b + 6| = 0) : a + b - 2 = -5 := sorry

theorem part2 (h : |a - 3| + |b + 6| = 0) : a - b - 2 = 7 := sorry

end part1_part2_l1025_102512


namespace find_N_l1025_102541

theorem find_N
  (N : ℕ)
  (h : (4 / 10 : ℝ) * (16 / (16 + N : ℝ)) + (6 / 10 : ℝ) * (N / (16 + N : ℝ)) = 0.58) :
  N = 144 :=
sorry

end find_N_l1025_102541


namespace sqrt_64_eq_8_l1025_102504

theorem sqrt_64_eq_8 : Real.sqrt 64 = 8 := 
by
  sorry

end sqrt_64_eq_8_l1025_102504


namespace find_a_l1025_102570

def setA (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def setB : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) : setA a ⊆ setB ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l1025_102570


namespace num_solutions_abcd_eq_2020_l1025_102574

theorem num_solutions_abcd_eq_2020 :
  ∃ S : Finset (ℕ × ℕ × ℕ × ℕ), 
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔ (a^2 + b^2) * (c^2 - d^2) = 2020 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧
    S.card = 6 :=
sorry

end num_solutions_abcd_eq_2020_l1025_102574


namespace solve_equation_2021_l1025_102526

theorem solve_equation_2021 (x : ℝ) (hx : 0 ≤ x) : 
  2021 * x = 2022 * (x ^ (2021 : ℕ)) ^ (1 / (2021 : ℕ)) - 1 → x = 1 := 
by
  sorry

end solve_equation_2021_l1025_102526


namespace product_of_possible_b_values_l1025_102586

theorem product_of_possible_b_values : 
  ∀ b : ℝ, 
    (abs (b - 2) = 2 * (4 - 1)) → 
    (b = 8 ∨ b = -4) → 
    (8 * (-4) = -32) := by
  sorry

end product_of_possible_b_values_l1025_102586


namespace minimum_value_exists_l1025_102587

theorem minimum_value_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_condition : x + 4 * y = 2) : 
  ∃ z : ℝ, z = (x + 40 * y + 4) / (3 * x * y) ∧ z ≥ 18 :=
by
  sorry

end minimum_value_exists_l1025_102587


namespace range_of_a_l1025_102538

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, (a * x^2 - 3 * x - 4 = 0) ∧ (a * y^2 - 3 * y - 4 = 0) → x = y) ↔ (a ≤ -9 / 16 ∨ a = 0) := 
by
  sorry

end range_of_a_l1025_102538


namespace simplify_expression_l1025_102523

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l1025_102523


namespace consumption_increase_l1025_102548

theorem consumption_increase (T C : ℝ) (P : ℝ) (h : 0.82 * (1 + P / 100) = 0.943) :
  P = 15.06 := by
  sorry

end consumption_increase_l1025_102548


namespace factorize_difference_of_squares_l1025_102559

-- We are proving that the factorization of m^2 - 9 is equal to (m+3)(m-3)
theorem factorize_difference_of_squares (m : ℝ) : m ^ 2 - 9 = (m + 3) * (m - 3) := 
by 
  sorry

end factorize_difference_of_squares_l1025_102559


namespace jackies_lotion_bottles_l1025_102596

theorem jackies_lotion_bottles (L: ℕ) : 
  (10 + 10) + 6 * L + 12 = 50 → L = 3 :=
by
  sorry

end jackies_lotion_bottles_l1025_102596


namespace circles_intersect_at_two_points_l1025_102529

noncomputable def point_intersection_count (A B : ℝ × ℝ) (rA rB d : ℝ) : ℕ :=
  let distance := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  if rA + rB >= d ∧ d >= |rA - rB| then 2 else if d = rA + rB ∨ d = |rA - rB| then 1 else 0

theorem circles_intersect_at_two_points :
  point_intersection_count (0, 0) (8, 0) 3 6 8 = 2 :=
by 
  -- Proof for the statement will go here
  sorry

end circles_intersect_at_two_points_l1025_102529


namespace root_in_interval_imp_range_m_l1025_102588

theorem root_in_interval_imp_range_m (m : ℝ) (f : ℝ → ℝ) (h : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0) : 2 < m ∧ m < 4 :=
by
  have exists_x : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 := h
  sorry

end root_in_interval_imp_range_m_l1025_102588


namespace find_c_and_general_formula_l1025_102597

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) := ∀ n : ℕ, a (n + 1) = a n + c * 2^n

theorem find_c_and_general_formula : 
  ∀ (c : ℕ) (a : ℕ → ℕ),
    (a 1 = 2) →
    (seq a c) →
    ((a 3) = (a 1) * ((a 2) / (a 1))^2) →
    ((a 2) = (a 1) * (a 2) / (a 1)) →
    c = 1 ∧ (∀ n, a n = 2^n) := 
by
  sorry

end find_c_and_general_formula_l1025_102597


namespace number_of_employees_is_five_l1025_102513

theorem number_of_employees_is_five
  (rudy_speed : ℕ)
  (joyce_speed : ℕ)
  (gladys_speed : ℕ)
  (lisa_speed : ℕ)
  (mike_speed : ℕ)
  (average_speed : ℕ)
  (h1 : rudy_speed = 64)
  (h2 : joyce_speed = 76)
  (h3 : gladys_speed = 91)
  (h4 : lisa_speed = 80)
  (h5 : mike_speed = 89)
  (h6 : average_speed = 80) :
  (rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed) / average_speed = 5 :=
by
  sorry

end number_of_employees_is_five_l1025_102513


namespace point_of_tangency_l1025_102584

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem point_of_tangency (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) 
  (h_slope : ∃ x : ℝ, Real.exp x - 1 / Real.exp x = 3 / 2) :
  ∃ x : ℝ, x = Real.log 2 :=
by
  sorry

end point_of_tangency_l1025_102584


namespace total_wheels_l1025_102551

theorem total_wheels (bicycles tricycles : ℕ) (wheels_per_bicycle wheels_per_tricycle : ℕ) 
  (h1 : bicycles = 50) (h2 : tricycles = 20) (h3 : wheels_per_bicycle = 2) (h4 : wheels_per_tricycle = 3) : 
  (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160) :=
by
  sorry

end total_wheels_l1025_102551


namespace divisor_of_a_l1025_102515

theorem divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 45) 
  (h3 : Nat.gcd c d = 75) (h4 : 80 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
  7 ∣ a :=
by
  sorry

end divisor_of_a_l1025_102515


namespace value_of_m_l1025_102500

theorem value_of_m (x m : ℝ) (h : 2 * x + m - 6 = 0) (hx : x = 1) : m = 4 :=
by
  sorry

end value_of_m_l1025_102500


namespace gnome_count_l1025_102555

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l1025_102555


namespace geo_seq_a6_eight_l1025_102563

-- Definitions based on given conditions
variable (a : ℕ → ℝ) -- the sequence
variable (q : ℝ) -- common ratio
-- Conditions for a_1 * a_3 = 4 and a_4 = 4
def geometric_sequence := ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q
def condition1 := a 1 * a 3 = 4
def condition2 := a 4 = 4

-- Proof problem: Prove a_6 = 8 given the conditions above
theorem geo_seq_a6_eight (h1 : condition1 a) (h2 : condition2 a) (hs : geometric_sequence a) : 
  a 6 = 8 :=
sorry

end geo_seq_a6_eight_l1025_102563


namespace find_k_l1025_102524

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l1025_102524


namespace quadratic_discriminant_l1025_102540

theorem quadratic_discriminant {a b c : ℝ} (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4 * a * c :=
sorry

end quadratic_discriminant_l1025_102540


namespace f_2a_eq_3_l1025_102581

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / 2^x

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2 * a) = 3 := by
  sorry

end f_2a_eq_3_l1025_102581


namespace combined_instruments_correct_l1025_102534

-- Definitions of initial conditions
def Charlie_flutes : Nat := 1
def Charlie_horns : Nat := 2
def Charlie_harps : Nat := 1
def Carli_flutes : Nat := 2 * Charlie_flutes
def Carli_horns : Nat := Charlie_horns / 2
def Carli_harps : Nat := 0

-- Calculation of total instruments
def Charlie_total_instruments : Nat := Charlie_flutes + Charlie_horns + Charlie_harps
def Carli_total_instruments : Nat := Carli_flutes + Carli_horns + Carli_harps
def combined_total_instruments : Nat := Charlie_total_instruments + Carli_total_instruments

-- Theorem statement
theorem combined_instruments_correct : combined_total_instruments = 7 := 
by
  sorry

end combined_instruments_correct_l1025_102534


namespace fraction_habitable_l1025_102595

theorem fraction_habitable : (1 / 3) * (1 / 3) = 1 / 9 := 
by 
  sorry

end fraction_habitable_l1025_102595


namespace sin_2x_equals_neg_61_div_72_l1025_102578

variable (x y : Real)
variable (h1 : Real.sin y = (3 / 2) * Real.sin x + (2 / 3) * Real.cos x)
variable (h2 : Real.cos y = (2 / 3) * Real.sin x + (3 / 2) * Real.cos x)

theorem sin_2x_equals_neg_61_div_72 : Real.sin (2 * x) = -61 / 72 :=
by
  -- Proof goes here
  sorry

end sin_2x_equals_neg_61_div_72_l1025_102578


namespace bonnets_per_orphanage_l1025_102546

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l1025_102546


namespace amount_of_water_in_first_tank_l1025_102585

theorem amount_of_water_in_first_tank 
  (C : ℝ)
  (H1 : 0 < C)
  (H2 : 0.45 * C = 450)
  (water_in_first_tank : ℝ)
  (water_in_second_tank : ℝ := 450)
  (additional_water_needed : ℝ := 1250)
  (total_capacity : ℝ := 2 * C)
  (total_water_needed : ℝ := 2000) : 
  water_in_first_tank = 300 :=
by 
  sorry

end amount_of_water_in_first_tank_l1025_102585


namespace find_lines_and_intersections_l1025_102536

-- Define the intersection point conditions
def intersection_point (m n : ℝ) : Prop :=
  (2 * m - n + 7 = 0) ∧ (m + n - 1 = 0)

-- Define the perpendicular line to l1 passing through (-2, 3)
def perpendicular_line_through_A (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

-- Define the parallel line to l passing through (-2, 3)
def parallel_line_through_A (x y : ℝ) : Prop :=
  2 * x - 3 * y + 13 = 0

-- main theorem
theorem find_lines_and_intersections :
  ∃ m n : ℝ, intersection_point m n ∧ m = -2 ∧ n = 3 ∧
  ∃ l3 : ℝ → ℝ → Prop, l3 = perpendicular_line_through_A ∧
  ∃ l4 : ℝ → ℝ → Prop, l4 = parallel_line_through_A :=
sorry

end find_lines_and_intersections_l1025_102536


namespace two_digit_sequence_partition_property_l1025_102572

theorem two_digit_sequence_partition_property :
  ∀ (A B : Set ℕ), (A ∪ B = {x | x < 100 ∧ x % 10 < 10}) →
  ∃ (C : Set ℕ), (C = A ∨ C = B) ∧ 
  ∃ (lst : List ℕ), (∀ (x : ℕ), x ∈ lst → x ∈ C) ∧ 
  (∀ (x y : ℕ), (x, y) ∈ lst.zip lst.tail → (y = x + 1 ∨ y = x + 10 ∨ y = x + 11)) :=
by
  intros A B partition_condition
  sorry

end two_digit_sequence_partition_property_l1025_102572


namespace geom_seq_result_l1025_102592

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ)

-- Conditions
axiom h1 : a 1 + a 3 = 5 / 2
axiom h2 : a 2 + a 4 = 5 / 4

-- General properties
axiom geom_seq_common_ratio : ∃ q : ℚ, ∀ n, a (n + 1) = a n * q

-- Sum of the first n terms of the geometric sequence
axiom S_def : S n = (2 * (1 - (1 / 2)^n)) / (1 - 1 / 2)

-- General term of the geometric sequence
axiom a_n_def : a n = 2 * (1 / 2)^(n - 1)

-- Result to be proved
theorem geom_seq_result : S n / a n = 2^n - 1 := 
  by sorry

end geom_seq_result_l1025_102592


namespace find_f_neg2_l1025_102502

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem find_f_neg2 : f (-2) = 15 :=
by
  sorry

end find_f_neg2_l1025_102502


namespace smallest_a_b_sum_l1025_102521

theorem smallest_a_b_sum :
  ∃ (a b : ℕ), 3^6 * 5^3 * 7^2 = a^b ∧ a + b = 317 := 
sorry

end smallest_a_b_sum_l1025_102521


namespace syllogistic_reasoning_problem_l1025_102579

theorem syllogistic_reasoning_problem
  (H1 : ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I)
  (H2 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.re z = 2)
  (H3 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.im z = 3) :
  (¬ ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I) → "The conclusion is wrong due to the incorrect major premise" = "A" :=
sorry

end syllogistic_reasoning_problem_l1025_102579


namespace remainder_t100_mod_7_l1025_102554

theorem remainder_t100_mod_7 :
  ∀ T : ℕ → ℕ, (T 1 = 3) →
  (∀ n : ℕ, n > 1 → T n = 3 ^ (T (n - 1))) →
  (T 100 % 7 = 6) :=
by
  intro T h1 h2
  -- sorry to skip the actual proof
  sorry

end remainder_t100_mod_7_l1025_102554


namespace prime_divisors_of_390_l1025_102561

theorem prime_divisors_of_390 : 
  (2 * 195 = 390) → 
  (3 * 65 = 195) → 
  (5 * 13 = 65) → 
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (S.card = 4) ∧ 
    (∀ d ∈ S, d ∣ 390) := 
by
  sorry

end prime_divisors_of_390_l1025_102561


namespace ab_abs_value_l1025_102562

theorem ab_abs_value {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : ∃ r s : ℤ, (x - r)^2 * (x - s) = x^3 + a * x^2 + b * x + 9 * a) :
  |a * b| = 1344 := 
sorry

end ab_abs_value_l1025_102562


namespace min_value_of_frac_l1025_102535

theorem min_value_of_frac (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : 2 * m + n = 1) (hm : m > 0) (hn : n > 0) :
  (1 / m) + (2 / n) = 8 :=
sorry

end min_value_of_frac_l1025_102535


namespace gumball_sharing_l1025_102543

theorem gumball_sharing (init_j : ℕ) (init_jq : ℕ) (mult_j : ℕ) (mult_jq : ℕ) :
  init_j = 40 → init_jq = 60 → mult_j = 5 → mult_jq = 3 →
  (init_j + mult_j * init_j + init_jq + mult_jq * init_jq) / 2 = 240 :=
by
  intros h1 h2 h3 h4
  sorry

end gumball_sharing_l1025_102543


namespace max_value_of_p_l1025_102518

theorem max_value_of_p
  (p q r s : ℕ)
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90)
  (h5 : 0 < s)
  (h6 : 0 < r)
  (h7 : 0 < q)
  (h8 : 0 < p):
  p ≤ 5324 :=
by
  sorry

end max_value_of_p_l1025_102518


namespace bill_bathroom_visits_per_day_l1025_102519

theorem bill_bathroom_visits_per_day
  (squares_per_use : ℕ)
  (rolls : ℕ)
  (squares_per_roll : ℕ)
  (days_supply : ℕ)
  (total_uses : squares_per_use = 5)
  (total_rolls : rolls = 1000)
  (squares_from_each_roll : squares_per_roll = 300)
  (total_days : days_supply = 20000) :
  ( (rolls * squares_per_roll) / days_supply / squares_per_use ) = 3 :=
by
  sorry

end bill_bathroom_visits_per_day_l1025_102519


namespace fraction_subtraction_l1025_102591

theorem fraction_subtraction : (18 : ℚ) / 45 - (3 : ℚ) / 8 = (1 : ℚ) / 40 := by
  sorry

end fraction_subtraction_l1025_102591


namespace A_n_eq_B_n_l1025_102558

open Real

noncomputable def A_n (n : ℕ) : ℝ :=
  1408 * (1 - (1 / (2 : ℝ) ^ n))

noncomputable def B_n (n : ℕ) : ℝ :=
  (3968 / 3) * (1 - (1 / (-2 : ℝ) ^ n))

theorem A_n_eq_B_n : A_n 5 = B_n 5 := sorry

end A_n_eq_B_n_l1025_102558


namespace number_of_even_red_faces_cubes_l1025_102527

def painted_cubes_even_faces : Prop :=
  let block_length := 4
  let block_width := 4
  let block_height := 1
  let edge_cubes_count := 8  -- The count of edge cubes excluding corners
  edge_cubes_count = 8

theorem number_of_even_red_faces_cubes : painted_cubes_even_faces := by
  sorry

end number_of_even_red_faces_cubes_l1025_102527


namespace complement_inter_proof_l1025_102583

open Set

variable (U : Set ℕ) (A B : Set ℕ)

def complement_inter (U A B : Set ℕ) : Set ℕ :=
  compl (A ∩ B)

theorem complement_inter_proof (hU : U = {1, 2, 3, 4, 5, 6, 7, 8} )
  (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) :
  complement_inter U A B = {1, 4, 5, 6, 7, 8} :=
by
  sorry

end complement_inter_proof_l1025_102583


namespace sides_of_polygons_l1025_102509

theorem sides_of_polygons (p : ℕ) (γ : ℝ) (n1 n2 : ℕ) (h1 : p = 5) (h2 : γ = 12 / 7) 
    (h3 : n2 = n1 + p) 
    (h4 : 360 / n1 - 360 / n2 = γ) : 
    n1 = 30 ∧ n2 = 35 := 
  sorry

end sides_of_polygons_l1025_102509


namespace roger_has_more_candy_l1025_102573

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end roger_has_more_candy_l1025_102573


namespace total_sharks_l1025_102520

-- Define the number of sharks at each beach.
def N : ℕ := 22
def D : ℕ := 4 * N
def H : ℕ := D / 2

-- Proof that the total number of sharks on the three beaches is 154.
theorem total_sharks : N + D + H = 154 := by
  sorry

end total_sharks_l1025_102520


namespace matches_played_by_team_B_from_city_A_l1025_102507

-- Define the problem setup, conditions, and the conclusion we need to prove
structure Tournament :=
  (cities : ℕ)
  (teams_per_city : ℕ)

-- Assuming each team except Team A of city A has played a unique number of matches,
-- find the number of matches played by Team B of city A.
theorem matches_played_by_team_B_from_city_A (t : Tournament)
  (unique_match_counts_except_A : ∀ (i j : ℕ), i ≠ j → (i < t.cities → (t.teams_per_city * i ≠ t.teams_per_city * j)) ∧ (i < t.cities - 1 → (t.teams_per_city * i ≠ t.teams_per_city * (t.cities - 1)))) :
  (t.cities = 16) → (t.teams_per_city = 2) → ∃ n, n = 15 :=
by
  sorry

end matches_played_by_team_B_from_city_A_l1025_102507


namespace compare_travel_times_l1025_102525

variable (v : ℝ) (t1 t2 : ℝ)

def travel_time_first := t1 = 100 / v
def travel_time_second := t2 = 200 / v

theorem compare_travel_times (h1 : travel_time_first v t1) (h2 : travel_time_second v t2) : 
  t2 = 2 * t1 :=
by
  sorry

end compare_travel_times_l1025_102525


namespace find_three_digit_number_l1025_102516

theorem find_three_digit_number : 
  ∀ (c d e : ℕ), 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 
  (10 * c + d) / 99 + (100 * c + 10 * d + e) / 999 = 44 / 99 → 
  100 * c + 10 * d + e = 400 :=
by {
  sorry
}

end find_three_digit_number_l1025_102516


namespace partition_nats_100_subsets_l1025_102560

theorem partition_nats_100_subsets :
  ∃ (S : ℕ → ℕ), (∀ n, 1 ≤ S n ∧ S n ≤ 100) ∧
    (∀ a b c : ℕ, a + 99 * b = c → S a = S c ∨ S a = S b ∨ S b = S c) :=
by
  sorry

end partition_nats_100_subsets_l1025_102560


namespace gain_percent_l1025_102510

theorem gain_percent (C S : ℝ) (h : 50 * C = 30 * S) : ((S - C) / C) * 100 = 200 / 3 :=
by 
  sorry

end gain_percent_l1025_102510


namespace nonneg_int_values_of_fraction_condition_l1025_102593

theorem nonneg_int_values_of_fraction_condition (n : ℕ) : (∃ k : ℤ, 30 * n + 2 = k * (12 * n + 1)) → n = 0 := by
  sorry

end nonneg_int_values_of_fraction_condition_l1025_102593


namespace time_to_pick_up_dog_l1025_102568

def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90
def dinner_time_in_minutes : ℕ := 180  -- 7:00 pm - 4:00 pm in minutes

def total_known_time : ℕ := commute_time + grocery_time + dry_cleaning_time + cooking_time

theorem time_to_pick_up_dog : (dinner_time_in_minutes - total_known_time) = 20 :=
by
  -- Proof goes here.
  sorry

end time_to_pick_up_dog_l1025_102568


namespace remainder_of_98_pow_50_mod_50_l1025_102547

theorem remainder_of_98_pow_50_mod_50 : (98 ^ 50) % 50 = 0 := by
  sorry

end remainder_of_98_pow_50_mod_50_l1025_102547


namespace find_four_digit_numbers_l1025_102580

noncomputable def four_digit_number_permutations_sum (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) : Prop :=
  6 * (x + y + z + t) * (1000 + 100 + 10 + 1) = 10 * (1111 * x)

theorem find_four_digit_numbers (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) :
  four_digit_number_permutations_sum x y z t distinct nonzero :=
  sorry

end find_four_digit_numbers_l1025_102580


namespace estimate_total_children_l1025_102503

variables (k m n T : ℕ)

/-- There are k children initially given red ribbons. 
    Then m children are randomly selected, 
    and n of them have red ribbons. -/

theorem estimate_total_children (h : n * T = k * m) : T = k * m / n :=
by sorry

end estimate_total_children_l1025_102503


namespace question1_solution_question2_solution_l1025_102545

-- Define the function f for any value of a
def f (a : ℝ) (x : ℝ) : ℝ :=
  abs (x + 1) - abs (a * x - 1)

-- Definition specifically for question (1) setting a = 1
def f1 (x : ℝ) : ℝ :=
  f 1 x

-- Definition of the set for the inequality in (1)
def solution_set_1 : Set ℝ :=
  { x | f1 x > 1 }

-- Theorem for question (1)
theorem question1_solution :
  solution_set_1 = { x : ℝ | x > 1/2 } :=
sorry

-- Condition for question (2)
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  f a x > x

-- Define the interval for x in question (2)
def interval_0_1 (x : ℝ) : Prop :=
  0 < x ∧ x < 1

-- Theorem for question (2)
theorem question2_solution {a : ℝ} :
  (∀ x ∈ {x | interval_0_1 x}, inequality_condition a x) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end question1_solution_question2_solution_l1025_102545


namespace golden_ticket_problem_l1025_102553

open Real

/-- The golden ratio -/
noncomputable def φ := (1 + sqrt 5) / 2

/-- Assume the proportions and the resulting area -/
theorem golden_ticket_problem
  (a b : ℝ)
  (h : 0 + b * φ = 
        φ - (5 + sqrt 5) / (8 * φ)) :
  b / a = -4 / 3 :=
  sorry

end golden_ticket_problem_l1025_102553


namespace Robin_total_distance_walked_l1025_102571

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l1025_102571


namespace star_in_S_star_associative_l1025_102530

def S (x : ℕ) : Prop :=
  x > 1 ∧ x % 2 = 1

def f (x : ℕ) : ℕ :=
  Nat.log2 x

def star (a b : ℕ) : ℕ :=
  a + 2 ^ (f a) * (b - 3)

theorem star_in_S (a b : ℕ) (h_a : S a) (h_b : S b) : S (star a b) :=
  sorry

theorem star_associative (a b c : ℕ) (h_a : S a) (h_b : S b) (h_c : S c) :
  star (star a b) c = star a (star b c) :=
  sorry

end star_in_S_star_associative_l1025_102530


namespace sqrt_of_product_eq_540_l1025_102589

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l1025_102589


namespace magic_8_ball_probability_l1025_102582

theorem magic_8_ball_probability :
  let p_pos := 1 / 3
  let p_neg := 2 / 3
  let n := 6
  let k := 3
  (Nat.choose n k * (p_pos ^ k) * (p_neg ^ (n - k)) = 160 / 729) :=
by
  sorry

end magic_8_ball_probability_l1025_102582


namespace smallest_non_factor_product_of_factors_of_72_l1025_102505

theorem smallest_non_factor_product_of_factors_of_72 : 
  ∃ x y : ℕ, x ≠ y ∧ x * y ∣ 72 ∧ ¬ (x * y ∣ 72) ∧ x * y = 32 := 
by
  sorry

end smallest_non_factor_product_of_factors_of_72_l1025_102505


namespace min_rows_required_to_seat_students_l1025_102544

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end min_rows_required_to_seat_students_l1025_102544


namespace principal_amount_borrowed_l1025_102567

theorem principal_amount_borrowed (SI R T : ℝ) (h_SI : SI = 2000) (h_R : R = 4) (h_T : T = 10) : 
    ∃ P, SI = (P * R * T) / 100 ∧ P = 5000 :=
by
    sorry

end principal_amount_borrowed_l1025_102567


namespace Tim_cookie_packages_l1025_102539

theorem Tim_cookie_packages 
    (cookies_in_package : ℕ)
    (packets_in_package : ℕ)
    (min_packet_count : ℕ)
    (h1 : cookies_in_package = 5)
    (h2 : packets_in_package = 7)
    (h3 : min_packet_count = 30) :
  ∃ (cookie_packages : ℕ) (packet_packages : ℕ),
    cookie_packages = 7 ∧ packet_packages = 5 ∧
    cookie_packages * cookies_in_package = packet_packages * packets_in_package ∧
    packet_packages * packets_in_package ≥ min_packet_count :=
by
  sorry

end Tim_cookie_packages_l1025_102539


namespace park_will_have_9_oak_trees_l1025_102531

def current_oak_trees : Nat := 5
def additional_oak_trees : Nat := 4
def total_oak_trees : Nat := current_oak_trees + additional_oak_trees

theorem park_will_have_9_oak_trees : total_oak_trees = 9 :=
by
  sorry

end park_will_have_9_oak_trees_l1025_102531


namespace total_weight_of_7_moles_CaO_l1025_102564

/-- Definitions necessary for the problem --/
def atomic_weight_Ca : ℝ := 40.08 -- atomic weight of calcium in g/mol
def atomic_weight_O : ℝ := 16.00 -- atomic weight of oxygen in g/mol
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O -- molecular weight of CaO in g/mol
def number_of_moles_CaO : ℝ := 7 -- number of moles of CaO

/-- The main theorem statement --/
theorem total_weight_of_7_moles_CaO :
  molecular_weight_CaO * number_of_moles_CaO = 392.56 :=
by
  sorry

end total_weight_of_7_moles_CaO_l1025_102564


namespace c_share_l1025_102533

theorem c_share (A B C : ℕ) (h1 : A = B / 2) (h2 : B = C / 2) (h3 : A + B + C = 392) : C = 224 :=
by
  sorry

end c_share_l1025_102533


namespace max_value_sqrt_sum_l1025_102522

theorem max_value_sqrt_sum {x y z : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  ∃ (M : ℝ), M = (Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x))) ∧ M = Real.sqrt 2 + 1 :=
by sorry

end max_value_sqrt_sum_l1025_102522


namespace simplify_expression_l1025_102542

theorem simplify_expression :
  (4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75 / 1 + 53/68 / ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 :=
by sorry

end simplify_expression_l1025_102542


namespace dogs_with_pointy_ears_l1025_102575

theorem dogs_with_pointy_ears (total_dogs with_spots with_pointy_ears: ℕ) 
  (h1: with_spots = total_dogs / 2)
  (h2: total_dogs = 30) :
  with_pointy_ears = total_dogs / 5 :=
by
  sorry

end dogs_with_pointy_ears_l1025_102575


namespace problem1_problem2_l1025_102576

variables (a b c d e f : ℝ)

-- Define the probabilities and the sum condition
def total_probability (a b c d e f : ℝ) : Prop := a + b + c + d + e + f = 1

-- Define P and Q
def P (a b c d e f : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + e^2 + f^2
def Q (a b c d e f : ℝ) : ℝ := (a + c + e) * (b + d + f)

-- Problem 1
theorem problem1 (h : total_probability a b c d e f) : P a b c d e f ≥ 1/6 := sorry

-- Problem 2
theorem problem2 (h : total_probability a b c d e f) : 
  1/4 ≥ Q a b c d e f ∧ Q a b c d e f ≥ 1/2 - 3/2 * P a b c d e f := sorry

end problem1_problem2_l1025_102576


namespace actual_cost_of_article_l1025_102552

noncomputable def article_actual_cost (x : ℝ) : Prop :=
  (0.58 * x = 1050) → x = 1810.34

theorem actual_cost_of_article : ∃ x : ℝ, article_actual_cost x :=
by
  use 1810.34
  sorry

end actual_cost_of_article_l1025_102552


namespace fixed_real_root_l1025_102550

theorem fixed_real_root (k x : ℝ) (h : x^2 + (k + 3) * x + (k + 2) = 0) : x = -1 :=
sorry

end fixed_real_root_l1025_102550


namespace barneys_grocery_store_items_left_l1025_102577

theorem barneys_grocery_store_items_left 
    (ordered_items : ℕ) 
    (sold_items : ℕ) 
    (storeroom_items : ℕ) 
    (damaged_percentage : ℝ)
    (h1 : ordered_items = 4458) 
    (h2 : sold_items = 1561) 
    (h3 : storeroom_items = 575) 
    (h4 : damaged_percentage = 5/100) : 
    ordered_items - (sold_items + ⌊damaged_percentage * ordered_items⌋) + storeroom_items = 3250 :=
by
    sorry

end barneys_grocery_store_items_left_l1025_102577


namespace king_gvidon_descendants_l1025_102508

def number_of_sons : ℕ := 5
def number_of_descendants_with_sons : ℕ := 100
def number_of_sons_each : ℕ := 3
def number_of_grandsons : ℕ := number_of_descendants_with_sons * number_of_sons_each

def total_descendants : ℕ := number_of_sons + number_of_grandsons

theorem king_gvidon_descendants : total_descendants = 305 :=
by
  sorry

end king_gvidon_descendants_l1025_102508


namespace least_number_of_trees_l1025_102517

theorem least_number_of_trees (n : ℕ) :
  (∃ k₄ k₅ k₆, n = 4 * k₄ ∧ n = 5 * k₅ ∧ n = 6 * k₆) ↔ n = 60 :=
by 
  sorry

end least_number_of_trees_l1025_102517


namespace problems_left_to_grade_l1025_102501

-- Defining all the conditions
def worksheets_total : ℕ := 14
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 2

-- Stating the proof problem
theorem problems_left_to_grade : 
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 14 := 
by
  sorry

end problems_left_to_grade_l1025_102501


namespace sixty_percent_is_240_l1025_102511

variable (x : ℝ)

-- Conditions
def forty_percent_eq_160 : Prop := 0.40 * x = 160

-- Proof problem
theorem sixty_percent_is_240 (h : forty_percent_eq_160 x) : 0.60 * x = 240 :=
sorry

end sixty_percent_is_240_l1025_102511


namespace sqrt_six_greater_two_l1025_102598

theorem sqrt_six_greater_two : Real.sqrt 6 > 2 :=
by
  sorry

end sqrt_six_greater_two_l1025_102598


namespace find_k_l1025_102532

theorem find_k (k a : ℤ)
  (h₁ : 49 + k = a^2)
  (h₂ : 361 + k = (a + 2)^2)
  (h₃ : 784 + k = (a + 4)^2) :
  k = 6035 :=
by sorry

end find_k_l1025_102532


namespace bens_old_car_cost_l1025_102569

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l1025_102569


namespace expressions_equal_constant_generalized_identity_l1025_102528

noncomputable def expr1 := (Real.sin (13 * Real.pi / 180))^2 + (Real.cos (17 * Real.pi / 180))^2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180)
noncomputable def expr2 := (Real.sin (15 * Real.pi / 180))^2 + (Real.cos (15 * Real.pi / 180))^2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def expr3 := (Real.sin (-18 * Real.pi / 180))^2 + (Real.cos (48 * Real.pi / 180))^2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180)
noncomputable def expr4 := (Real.sin (-25 * Real.pi / 180))^2 + (Real.cos (55 * Real.pi / 180))^2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)

theorem expressions_equal_constant :
  expr1 = 3/4 ∧ expr2 = 3/4 ∧ expr3 = 3/4 ∧ expr4 = 3/4 :=
sorry

theorem generalized_identity (α : ℝ) :
  (Real.sin α)^2 + (Real.cos (30 * Real.pi / 180 - α))^2 - Real.sin α * Real.cos (30 * Real.pi / 180 - α) = 3 / 4 :=
sorry

end expressions_equal_constant_generalized_identity_l1025_102528


namespace sugar_needed_in_two_minutes_l1025_102566

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l1025_102566


namespace num_distinct_terms_expansion_a_b_c_10_l1025_102514

-- Define the expansion of (a+b+c)^10
def num_distinct_terms_expansion (n : ℕ) : ℕ :=
  Nat.choose (n + 3 - 1) (3 - 1)

-- Theorem statement
theorem num_distinct_terms_expansion_a_b_c_10 : num_distinct_terms_expansion 10 = 66 :=
by
  sorry

end num_distinct_terms_expansion_a_b_c_10_l1025_102514


namespace expression_value_l1025_102506

theorem expression_value
  (x y a b : ℤ)
  (h1 : x = 1)
  (h2 : y = 2)
  (h3 : a + 2 * b = 3) :
  2 * a + 4 * b - 5 = 1 := 
by sorry

end expression_value_l1025_102506


namespace closest_vector_l1025_102594

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -4 + 7 * s, 2 + 4 * s)
def b : ℝ × ℝ × ℝ := (5, 1, -3)
def direction : ℝ × ℝ × ℝ := (3, 7, 4)

theorem closest_vector (s : ℝ) :
  (u s - b) • direction = 0 ↔ s = 27 / 74 :=
sorry

end closest_vector_l1025_102594


namespace number_of_female_democrats_l1025_102556

-- Definitions and conditions
variables (F M D_F D_M D_T : ℕ)
axiom participant_total : F + M = 780
axiom female_democrats : D_F = 1 / 2 * F
axiom male_democrats : D_M = 1 / 4 * M
axiom total_democrats : D_T = 1 / 3 * (F + M)

-- Target statement to be proven
theorem number_of_female_democrats : D_T = 260 → D_F = 130 :=
by
  intro h
  sorry

end number_of_female_democrats_l1025_102556
