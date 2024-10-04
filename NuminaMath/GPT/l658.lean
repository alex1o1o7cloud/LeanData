import Mathlib

namespace Mona_unique_players_l658_658611

theorem Mona_unique_players :
  ∀ (g : ℕ) (p : ℕ) (r1 : ℕ) (r2 : ℕ),
  g = 9 → p = 4 → r1 = 2 → r2 = 1 →
  (g * p) - (r1 + r2) = 33 :=
by {
  intros g p r1 r2 hg hp hr1 hr2,
  rw [hg, hp, hr1, hr2],
  norm_num,
  sorry -- skipping proof as per instructions
}

end Mona_unique_players_l658_658611


namespace base_conversion_equivalence_l658_658731

theorem base_conversion_equivalence :
  ∃ (n : ℕ), (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 9 * C + B) ∧
             (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 6 * B + C) ∧
             n = 0 := 
by 
  sorry

end base_conversion_equivalence_l658_658731


namespace time_to_reach_julia_via_lee_l658_658251

theorem time_to_reach_julia_via_lee (d1 d2 d3 : ℕ) (t1 t2 : ℕ) :
  d1 = 2 → 
  t1 = 6 → 
  d3 = 3 → 
  (∀ v, v = d1 / t1) → 
  t2 = d3 / v → 
  t2 = 9 :=
by
  intros h1 h2 h3 hv ht2
  sorry

end time_to_reach_julia_via_lee_l658_658251


namespace value_of_p_is_arithmetic_seq_l658_658116

noncomputable theory

-- Define the sequence and the sum of the first n terms
def S (a : ℕ → ℝ) (p : ℝ) (n : ℕ) : ℝ := n * p * a n - n * p + n

-- Define the conditions that will be used in the problem
variables (a : ℕ → ℝ)
variable (n : ℕ)
variable (p : ℝ)
variable (h1 : a 1 ≠ a 2)

-- State that the sequence satisfies the given equation
axiom cond_S : ∀ n > 0, S a p n = n * p * a n - n * p + n

-- Proof goal 1: The value of p
theorem value_of_p : p = 1/2 :=
sorry

-- Proof goal 2: The sequence {a_n} is an arithmetic sequence
theorem is_arithmetic_seq (n : ℕ) (h2 : n ≥ 2) : a (n + 1) - a n = a n - a (n - 1) :=
sorry

end value_of_p_is_arithmetic_seq_l658_658116


namespace range_of_m_l658_658196

theorem range_of_m (m : ℝ) : (∃ x ∈ Icc (2 : ℝ) 4, x^2 - 2 * x + 5 - m < 0) ↔ m > 5 :=
by
  sorry

end range_of_m_l658_658196


namespace abs_differentiable_except_at_zero_l658_658293

noncomputable def abs_diff (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

theorem abs_differentiable_except_at_zero (x : ℝ) :
  differentiable_at ℝ (λ x : ℝ, abs x) x ↔ x ≠ 0 :=
by
  sorry

end abs_differentiable_except_at_zero_l658_658293


namespace extremely_large_number_l658_658437

noncomputable def a : ℕ → ℕ
| 1       := 1
| 2       := 2
| 3       := 3
| 4       := 4
| (n + 1) := (Finset.sum (Finset.range n) (λ k, a k.succ))^2 - 1

def P : ℕ := (Finset.range 100).prod (λ i, a (i + 1))

def S : ℕ := (Finset.range 100).sum (λ i, (a (i + 1))^2)

theorem extremely_large_number :
  P - S = -- Extremely large number (actual numeric result) goes here
:= sorry

end extremely_large_number_l658_658437


namespace bananas_to_oranges_equivalence_l658_658037

noncomputable def bananas_to_apples (bananas apples : ℕ) : Prop :=
  4 * apples = 3 * bananas

noncomputable def apples_to_oranges (apples oranges : ℕ) : Prop :=
  5 * oranges = 2 * apples

theorem bananas_to_oranges_equivalence (x y : ℕ) (hx : bananas_to_apples 24 x) (hy : apples_to_oranges x y) :
  y = 72 / 10 := by
  sorry

end bananas_to_oranges_equivalence_l658_658037


namespace max_volume_prism_l658_658035

-- Definitions required by the problem
def point := ℝ × ℝ × ℝ -- representing points in 3D space

-- Assume A, B, C, D, E, F are points in 3D space
variables A B C D E F : point

-- Additional conditions as hypotheses
variables (h_angle_ACB : ∠ACB = 90)
variables (h_DA_perp_ABC : is_perp D A (plane ABC)) -- DA ⊥ plane ABC
variables (h_AE_perp_DB : is_perp A E (line BD)) -- AE ⊥ DB at E
variables (h_AF_perp_DC : is_perp A F (line DC) ) -- AF ⊥ DC at F
variables (h_AD_EQ_AB : dist A D = 2 ∧ dist A B = 2) -- AD = AB = 2

-- Target statement in Lean 4
theorem max_volume_prism : volume_prism D A E F ≤ (sqrt 2) / 6 :=
by sorry

end max_volume_prism_l658_658035


namespace probability_divisible_by_4_l658_658631

-- Define the set S = {1, 2, 3, ..., 2009}
def S : set ℕ := {n | 1 ≤ n ∧ n ≤ 2009}

-- Define the condition for divisibility by 4
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Define the expression we are examining
def expr (a b c : ℕ) : ℕ := a * b * c + a * b + a

-- Main theorem statement
theorem probability_divisible_by_4 :
  let prob := (502 : ℚ) / 2009 in
  (502 / 2009) * 1 + (1 - prob) * ((1 : ℚ) / 2) = 641 / 2009 :=
by
  -- Placeholder for the actual proof
  sorry

end probability_divisible_by_4_l658_658631


namespace log_piece_weight_l658_658248

variable (length_of_log : ℕ) (weight_per_foot : ℕ) (number_of_pieces : ℕ)
variable (original_length : length_of_log = 20)
variable (weight_per_linear_foot : weight_per_foot = 150)
variable (cuts_in_half : number_of_pieces = 2)

theorem log_piece_weight : (length_of_log / number_of_pieces) * weight_per_foot = 1500 := by
  have length_of_piece : length_of_log / number_of_pieces = 10 := by
    rw [original_length, cuts_in_half]
    norm_num
  rw [length_of_piece, weight_per_linear_foot]
  norm_num
  -- Proof complete

#print log_piece_weight

end log_piece_weight_l658_658248


namespace compute_expression_l658_658048

theorem compute_expression :
  (real.sqrt 3 - 1)^0 + |(-3)| - (1/2)^(-2) = 4 :=
by
  sorry

end compute_expression_l658_658048


namespace curious_number_is_digit_swap_divisor_l658_658598

theorem curious_number_is_digit_swap_divisor (a b : ℕ) (hab : a ≠ 0 ∧ b ≠ 0) :
  (10 * a + b) ∣ (10 * b + a) → (10 * a + b) = 11 ∨ (10 * a + b) = 22 ∨ (10 * a + b) = 33 ∨ 
  (10 * a + b) = 44 ∨ (10 * a + b) = 55 ∨ (10 * a + b) = 66 ∨ 
  (10 * a + b) = 77 ∨ (10 * a + b) = 88 ∨ (10 * a + b) = 99 :=
by
  sorry

end curious_number_is_digit_swap_divisor_l658_658598


namespace one_person_remains_dry_l658_658282

theorem one_person_remains_dry (n : ℕ) :
  ∃ (person_dry : ℕ -> Bool), (∀ i : ℕ, i < 2 * n + 1 -> person_dry i = tt) := 
sorry

end one_person_remains_dry_l658_658282


namespace parallelogram_area_l658_658281

theorem parallelogram_area {a b : ℝ} (h₁ : a = 9) (h₂ : b = 12) (angle : ℝ) (h₃ : angle = 150) : 
  ∃ (area : ℝ), area = 54 * Real.sqrt 3 :=
by
  sorry

end parallelogram_area_l658_658281


namespace fifth_element_row_20_pascal_triangle_l658_658718

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l658_658718


namespace problem1_problem2_problem3_l658_658499

noncomputable def f (x : ℝ) : ℝ := 1 / x + x

-- Problem 1: Prove that f(x) is odd
theorem problem1 : ∀ x : ℝ, x ≠ 0 → f (-x) = -f (x) :=
by sorry

-- Problem 2: Prove that f(x) is increasing in the interval (1, +∞)
theorem problem2 : ∀ x₁ x₂ : ℝ, 1 < x₁ ∧ 1 < x₂ ∧ x₁ < x₂ → f (x₁) < f (x₂) :=
by sorry

-- Problem 3: Find the extreme values of f(x) in the interval [1, 3] 
theorem problem3 :
  let minimum := f 1 in
  minimum = 2 ∧ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (f x ≥ minimum) ∧
  let maximum := f 3 in
  maximum = 10/3 ∧ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (f x ≤ maximum) :=
by sorry

end problem1_problem2_problem3_l658_658499


namespace odd_coeffs_binomial_thm_l658_658982

noncomputable def sum_binary_digits : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_binary_digits (n / 2) + (n % 2)

theorem odd_coeffs_binomial_thm (n : ℕ) (s : ℕ) (hs : s = sum_binary_digits n) :
  let binomial_coeff_is_odd (k : ℕ) : Prop :=
    (nat.choose n k) % 2 = 1
  in
  (∑ k in finset.range (n + 1), if binomial_coeff_is_odd k then 1 else 0) = 2^s :=
sorry

end odd_coeffs_binomial_thm_l658_658982


namespace behavior_of_sine_function_l658_658496

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := Real.sin (ω * (x + Real.pi / 3) + φ)

theorem behavior_of_sine_function
  (ω φ : ℝ)
  (hω : ω > 0)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (h_shift : g ω φ (-Real.pi / 6) = 1) :
  ∀ x, -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 6 → 
  Real.deriv (f ω φ) x > 0 := 
sorry

end behavior_of_sine_function_l658_658496


namespace mona_unique_players_l658_658609

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l658_658609


namespace red_balls_in_bag_l658_658751

theorem red_balls_in_bag : 
  ∃ (r : ℕ), (r * (r - 1) = 22) ∧ (r ≤ 12) :=
by { sorry }

end red_balls_in_bag_l658_658751


namespace mappings_functions_count_l658_658784

-- Define sets and correspondences
def A1 := {x // x ∈ {x | x.is_triangle}}
def B1 := {x // x ∈ {x | x.is_circle}}
def f1 (a : A1) : B1 := sorry  -- mapping from triangle to its circumcircle

def A2 := {x // x ∈ {x | x.is_triangle}}
def B2 := ℝ
def f2 (a : A2) : B2 := sorry  -- mapping from triangle to its area

def A3 := ℝ
def B3 := ℝ
def f3 (x : A3) : B3 := real.cbrt x  -- mapping from real number to its cube root

def A4 := ℝ
def B4 := ℝ
def f4 (x : A4) : B4 := real.sqrt x  -- mapping from real number to its square root

theorem mappings_functions_count :
  let mappings := [f1, f2, f3],
      functions := [f2, f3]
  in mappings.length = 3 ∧ functions.length = 2 :=
by {
  -- Here we would provide the proof, but for now we just state the theorem.
  sorry
}

end mappings_functions_count_l658_658784


namespace man_rowed_downstream_l658_658007

variable (v c d : ℝ)

theorem man_rowed_downstream :
  (∀ d : ℝ, c = 1.5 ∧ (d = (v + c) * 6) ∧ (14 = (v - c) * 6) → d = 32) :=
begin
  sorry
end

end man_rowed_downstream_l658_658007


namespace difference_in_spectators_l658_658225

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l658_658225


namespace no_integer_solutions_l658_658443

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
  -x^2 + 4 * y * z + 3 * z^2 = 36 ∧
  x^2 + 2 * x * y + 9 * z^2 = 121 → false :=
by
  sorry

end no_integer_solutions_l658_658443


namespace count_even_integers_between_a_b_number_of_even_integers_between_a_b_l658_658170

def a : ℝ := 21 / 5
def b : ℝ := 45 / 3

def even_integers_in_range (a b : ℝ) : List ℤ :=
  List.filter (λ x, Int.even x) (List.range' ⟨⌊a⌋.to_int + 1, (⌊b⌋.to_int + 1) - (⌊a⌋.to_int + 1)⟩)

theorem count_even_integers_between_a_b :
  even_integers_in_range a b = [6, 8, 10, 12, 14] := 
by
  sorry

theorem number_of_even_integers_between_a_b :
  List.length (even_integers_in_range a b) = 5 := 
by
  sorry

end count_even_integers_between_a_b_number_of_even_integers_between_a_b_l658_658170


namespace probability_sum_greater_than_six_l658_658558

def boxA : List ℕ := [1, 2]
def boxB : List ℕ := [3, 4, 5, 6]

def all_pairs (boxA : List ℕ) (boxB : List ℕ) : List (ℕ × ℕ) :=
  List.product boxA boxB
  
def favorable_pairs (pairs : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  pairs.filter (λ p, p.fst + p.snd > 6)

theorem probability_sum_greater_than_six :
  (favorable_pairs (all_pairs boxA boxB)).length = 3 →
  (∣ (all_pairs boxA boxB)).length = 8 →
  (favorable_pairs (all_pairs boxA boxB)).length.toFloat / (all_pairs boxA boxB).length.toFloat = 3 / 8 := 
by
  intros h1 h2 
  have total_cases := h2
  have successful_cases := h1
  sorry

end probability_sum_greater_than_six_l658_658558


namespace mona_unique_players_l658_658610

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l658_658610


namespace completion_time_B_l658_658743

-- Definitions based on conditions
def work_rate_A : ℚ := 1 / 10 -- A's rate of completing work per day

def efficiency_B : ℚ := 1.75 -- B is 75% more efficient than A

def work_rate_B : ℚ := efficiency_B * work_rate_A -- B's work rate per day

-- The main theorem that we need to prove
theorem completion_time_B : (1 : ℚ) / work_rate_B = 40 / 7 :=
by 
  sorry

end completion_time_B_l658_658743


namespace bisecting_line_through_D_eq_l658_658690

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (4, 1)
def C : point := (4, 0)
def D : point := (0, -1)

def midpoint (p1 p2 : point) : point :=
  (((p1.1 + p2.1) / 2), ((p1.2 + p2.2) / 2))

def area (p1 p2 p3 : point) : ℝ :=
  0.5 * (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

noncomputable def bisecting_line (A B C D : point) : Prop :=
  ∃ m b : ℝ, (∀ x, (x, m * x + b).snd ≠ (D.snd)) ∧
  (m = 0.5 ∧ b = -1)

theorem bisecting_line_through_D_eq :
  bisecting_line A B C D :=
by
  sorry

end bisecting_line_through_D_eq_l658_658690


namespace find_y_l658_658533

theorem find_y (x y : ℝ) (h1 : 2 * x - 3 * y = 24) (h2 : x + 2 * y = 15) : y = 6 / 7 :=
by sorry

end find_y_l658_658533


namespace number_of_x0_such_that_x0_eq_x6_l658_658841
noncomputable def recurrence_relation (x: ℝ) (n: ℕ) : ℝ :=
  if 2^(n + 1) * x % 1 < 0.5 then 2^n * x else 2^n * x - (floor (2^n * x) + 1).mod 2

theorem number_of_x0_such_that_x0_eq_x6 :
  (∃ (x_0 : ℝ), 0 ≤ x_0 ∧ x_0 < 1 ∧ recurrence_relation x_0 6 = x_0) :=
begin
  sorry
end

end number_of_x0_such_that_x0_eq_x6_l658_658841


namespace neither_sufficient_nor_necessary_l658_658665

variable {m n : ℝ}

-- Definitions based on the provided conditions
def foci_on_y_axis (m n : ℝ) := m > n ∧ n > 0
def foci_on_x_axis (m n : ℝ) := n > m ∧ m > 0

-- Main theorem stating the problem
theorem neither_sufficient_nor_necessary (h : m > n ∧ n > 0) :
  ¬(∀ h₁ : m > n ∧ n > 0, foci_on_y_axis m n → foci_on_x_axis m n) ∧
  ¬(∀ h₂ : foci_on_x_axis m n → foci_on_y_axis m n) :=
by
  sorry

end neither_sufficient_nor_necessary_l658_658665


namespace probability_sum_is_four_l658_658695

theorem probability_sum_is_four : 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ x + y = 4) ∧ 
  (∀ a b : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) → (1 / 36)) →
  (1 / 36) * 3 = 1 / 12 :=
sorry

end probability_sum_is_four_l658_658695


namespace min_sum_of_factors_l658_658586

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l658_658586


namespace radioactive_ball_identification_l658_658486

/-- Given 11 balls where exactly 2 are radioactive and each test can detect the presence of 
    radioactive balls in a group, prove that fewer than 7 tests are insufficient to guarantee 
    identification of the 2 radioactive balls, but 7 tests are always sufficient. -/
theorem radioactive_ball_identification (total_balls radioactive_balls : ℕ) (tests : ℕ) 
  (test_group : set ℕ → Prop) :
  total_balls = 11 → radioactive_balls = 2 → 
  (∀ (S : set ℕ), test_group S ↔ ∃ (x : ℕ), x ∈ S ∧ is_radioactive x) →
  (tests < 7 → ¬identify_radioactive_balls total_balls radioactive_balls test_group tests) ∧ 
  (tests = 7 → identify_radioactive_balls total_balls radioactive_balls test_group tests) := 
by
  intros htotal hradioactive htest
  sorry

end radioactive_ball_identification_l658_658486


namespace intersection_M_N_l658_658151

open Set

def M : Set ℝ := { x | log 2 x < 1 }
def N : Set ℝ := { x | x^2 - 1 ≤ 0 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_M_N_l658_658151


namespace distinct_triangles_from_tetrahedron_l658_658161

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l658_658161


namespace family_eat_only_vegetarian_l658_658925

theorem family_eat_only_vegetarian (e_nonveg: ℕ) (e_both: ℕ) (e_veg: ℕ) 
  (h1: e_nonveg = 6) 
  (h2: e_both = 9) 
  (h3: e_veg = 20) : 
  e_veg - e_both = 11 := 
by
  rw [h2, h3]
  norm_num
  sorry

end family_eat_only_vegetarian_l658_658925


namespace Emilia_needs_more_cartons_l658_658068

theorem Emilia_needs_more_cartons :
  ∀ (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ), 
    total_needed = 42 →
    strawberries = 2 →
    blueberries = 7 →
    total_needed - (strawberries + blueberries) = 33 :=
by
  intros total_needed strawberries blueberries h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Emilia_needs_more_cartons_l658_658068


namespace alice_wins_stick_game_l658_658380

theorem alice_wins_stick_game :
  ∃ (strategy : ℕ → ℕ → ℕ), (∀ n (h : 1 ≤ n ∧ n ≤ 3), 
  ∀ m (hm : m = 2022 - 2) (hd : m % 4 = 0), 
  strategy (2022 - 2 - n) (hd) = 4 - n) := sorry

end alice_wins_stick_game_l658_658380


namespace c_is_younger_l658_658373

variables (a b c d : ℕ) -- assuming ages as natural numbers

-- Conditions
axiom cond1 : a + b = b + c + 12
axiom cond2 : b + d = c + d + 8
axiom cond3 : d = a + 5

-- Question
theorem c_is_younger : c = a - 12 :=
sorry

end c_is_younger_l658_658373


namespace compute_complex_power_l658_658431

noncomputable def complex_number := Complex.exp (Complex.I * 125 * Real.pi / 180)

theorem compute_complex_power :
  (complex_number ^ 28) = Complex.ofReal (-Real.cos (40 * Real.pi / 180)) + Complex.I * Real.sin (40 * Real.pi / 180) :=
by
  sorry

end compute_complex_power_l658_658431


namespace det_B_eq_2_l658_658965

variable {x y : ℝ}

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![x,  2], ![-3, y]]
def Binv : Matrix (Fin 2) (Fin 2) ℝ := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]

-- Given conditions
def B_constraint : (B + Binv = 0) := by sorry

theorem det_B_eq_2 (h : B_constraint) : B.det = 2 := by
  -- Proof goes here
  sorry

end det_B_eq_2_l658_658965


namespace thirteenth_number_value_l658_658124

variables (P Q R X Y : ℝ) (S1 S2 S3 O M : ℝ)

-- Define the conditions as Lean variables
def avg_25_numbers (n : ℝ) := n = P
def avg_first_10_numbers (n : ℝ) := n = Q
def avg_last_10_numbers (n : ℝ) := n = R
def seventh_number (n : ℝ) := n = X
def nineteenth_number (n : ℝ) := n = Y
def sum_first_10 := S1 = 10 * Q
def sum_last_10 := S2 = 10 * R
def sum_all := S1 + S3 + S2 = 25 * P
def sum_middle_5 := S3 = M + O

-- Lean statement to prove the value of the 13th number given conditions
theorem thirteenth_number_value (h1 : avg_25_numbers P) (h2 : avg_first_10_numbers Q) (h3 : avg_last_10_numbers R) 
  (h4 : seventh_number X) (h5 : nineteenth_number Y) (hS1 : sum_first_10) (hS2 : sum_last_10) 
  (hS3 : sum_all) (hS4 : sum_middle_5) : M = 25 * P - 10 * Q - 10 * R - O := by
  sorry

end thirteenth_number_value_l658_658124


namespace inscribed_pentagon_angles_sum_l658_658767

theorem inscribed_pentagon_angles_sum (α β γ δ ε : ℝ) (h1 : α + β + γ + δ + ε = 360) 
(h2 : α / 2 + β / 2 + γ / 2 + δ / 2 + ε / 2 = 180) : 
(α / 2) + (β / 2) + (γ / 2) + (δ / 2) + (ε / 2) = 180 :=
by
  sorry

end inscribed_pentagon_angles_sum_l658_658767


namespace angle_OQP_is_90_degrees_l658_658601

theorem angle_OQP_is_90_degrees (A B C D O P Q : Point) 
    (h1 : IsConvexQuadrilateral A B C D)
    (h2 : InscribedInCircle O A B C D)
    (h3 : P = intersection (line_through A C) (line_through B D))
    (h4 : Q ∈ circumcircle (triangle A P D))
    (h5 : Q ∈ circumcircle (triangle B P C))
    (h6 : Q ≠ P) :
    ∠ O Q P = 90 := 
begin
    sorry
end

end angle_OQP_is_90_degrees_l658_658601


namespace sqrt_meaningful_range_l658_658541

theorem sqrt_meaningful_range (x : ℝ) (h : 3 * x - 5 ≥ 0) : x ≥ 5 / 3 :=
sorry

end sqrt_meaningful_range_l658_658541


namespace sum_of_consecutive_integers_345_l658_658215

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end sum_of_consecutive_integers_345_l658_658215


namespace swans_not_ducks_l658_658577

theorem swans_not_ducks (total_birds ducks swans herons cormorants : ℕ) 
(h_total_birds : total_birds = 200) 
(h_ducks : ducks = (40 * total_birds) / 100) 
(h_swans : swans = (30 * total_birds) / 100) 
(h_herons : herons = (20 * total_birds) / 100) 
(h_cormorants : cormorants = (10 * total_birds) / 100) : 
(100 * swans / (total_birds - ducks) = 50) :=
by
  have h1 : ducks = 80 := by rw [h_total_birds, h_ducks]
  have h2 : total_non_ducks = total_birds - ducks := by rw [h1, h_total_birds]
  have h3 : swans = 60 := by rw [h_total_birds, h_swans]
  have h4 : total_non_ducks = 120 := by rw [h2]
  have h5 : (100 * swans / total_non_ducks = 50) := by sorry
  exact h5
  

end swans_not_ducks_l658_658577


namespace paint_usage_total_l658_658574

theorem paint_usage_total (paint_total : ℕ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) :
  paint_total = 360 → first_week_fraction = 2 / 3 → second_week_fraction = 1 / 5 →
  let first_week_usage := first_week_fraction * paint_total in
  let remaining_after_first_week := paint_total - first_week_usage.natAbs in
  let second_week_usage := second_week_fraction * remaining_after_first_week in
  let total_usage := first_week_usage + second_week_usage in
  total_usage.natAbs = 264 :=
by
  intros hpaint hfirst_week_fraction hsecond_week_fraction
  simp [hpaint, hfirst_week_fraction, hsecond_week_fraction]
  let first_week_usage := (2 / 3 : ℚ) * 360
  have p1 : first_week_usage = 240 := by norm_num
  simp [p1]
  let remaining_after_first_week := 360 - first_week_usage.natAbs
  have p2 : remaining_after_first_week = 120 := by norm_num
  simp [p2]
  let second_week_usage := (1 / 5 : ℚ) * remaining_after_first_week
  have p3 : second_week_usage = 24 := by norm_num
  simp [p3]
  let total_usage := first_week_usage + second_week_usage
  have p4 : total_usage = 240 + 24 := by norm_num
  have p5 : total_usage.natAbs = 264 := by norm_num
  exact p5

end paint_usage_total_l658_658574


namespace sums_of_consecutive_8_distinct_l658_658235

noncomputable def sum_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sums_of_consecutive_8_distinct (arr : List ℕ):
  (∀ (i : ℕ), i < 2018 → arr.nthLe i sorry ≠ 0) → -- To ensure all elements are not zero and valid indices.
  arr.length = 2018 →
  (∀ i, 8 * sum_natural_numbers 2018 % 2018 = 0 →
    (arr ++ arr).take 2025 % 2018 ≠ [0, 1, 2, 3, 4, 5, 6, 7, 8, ... , 2017]) :=
begin
  intro h,
  intros,
  sorry,
end

end sums_of_consecutive_8_distinct_l658_658235


namespace Victor_bought_6_decks_l658_658342

theorem Victor_bought_6_decks (V : ℕ) (h1 : 2 * 8 + 8 * V = 64) : V = 6 := by
  sorry

end Victor_bought_6_decks_l658_658342


namespace determinant_scaled_l658_658103

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end determinant_scaled_l658_658103


namespace skateboard_travel_distance_l658_658773

theorem skateboard_travel_distance :
  let a_1 := 8
  let d := 10
  let n := 40
  let a_n := a_1 + (n - 1) * d
  let S_n := (n / 2) * (a_1 + a_n)
in S_n = 8120 :=
by
  sorry

end skateboard_travel_distance_l658_658773


namespace regular_polygon_sides_l658_658915

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) :
  ∃ (n : ℕ), n > 2 ∧ (interior_angle = (n - 2) * 180 / n) :=
by
  sorry
 
end regular_polygon_sides_l658_658915


namespace exists_integer_n_l658_658597
open Real

theorem exists_integer_n (a b : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ + x₂ = -a ∧ x₁ * x₂ = b) : ∃ n : ℤ, (n^2 + a * n + b : ℝ) ≤ max (1 / 4) (1 / 2 * √(a^2 - 4 * b)) :=
by
  sorry

end exists_integer_n_l658_658597


namespace order_365_cards_less_than_2000_questions_l658_658844

noncomputable def can_order_365_cards_using_2000_questions : Prop := 
  ∃ f : ℕ → ℕ, 
    (∀ n, f (3 * n) = 3 * f n + 3 * n - 2) ∧ 
    f 4 = 4 ∧
    let f365 := f 365 in 
    f365 < 2000

theorem order_365_cards_less_than_2000_questions :
  can_order_365_cards_using_2000_questions :=
sorry

end order_365_cards_less_than_2000_questions_l658_658844


namespace intersection_points_l658_658335

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def x1 : ℝ := (3 + Real.sqrt 209) / 4
noncomputable def x2 : ℝ := (3 - Real.sqrt 209) / 4

noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

theorem intersection_points :
  (parabola1 x1 = parabola2 x1) ∧ (parabola1 x2 = parabola2 x2) :=
by
  sorry

end intersection_points_l658_658335


namespace zero_of_derivative_l658_658305

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3 * x + c

theorem zero_of_derivative:
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) →
  (c = 2 ∨ c = -2) :=
begin
  intro hz,
  have fc : ∀ x, f' x = 3 * x^2 - 3 := sorry,
  have f'_increasing : ∀ x, (3 * x^2 - 3 > 0) → (f x c > f (-x) c) := sorry,
  have critical_values : f (-1) c = c + 2 ∧ f 1 c = c - 2 := sorry,
  cases hz with x1 hx1,
  cases hx1 with x2 hx2,
  cases hx2 with ne_zero hx3,
  cases hx3 with hx1_0 hx2_0,
  by_cases hxc : x1 = -1 ∨ x2 = -1,
  { have c1 : c = -2 := sorry, 
    tauto },
  by_cases hyc : x1 = 1 ∨ x2 = 1, 
  { have c2 : c = 2 := sorry,
    tauto },
  sorry
end

end zero_of_derivative_l658_658305


namespace β_possible_values_l658_658121

-- Define the context in which the angles exist and their relationship.
axiom equal_or_supplementary {α β : ℝ} (h_parallel_sides : by sorry) : β = α ∨ β = 180 - α

-- Define the given angle and its measure.
def α : ℝ := 60

-- Define the specific conditions of the problem.
axiom h_alpha_parallel_beta_sides : by sorry

-- State the theorem or problem needing proof.
theorem β_possible_values (h_alpha : α = 60) (h_parallel : by sorry) : β = 60 ∨ β = 120 :=
  equal_or_supplementary h_parallel

end β_possible_values_l658_658121


namespace cos_sum_of_intersections_l658_658276

theorem cos_sum_of_intersections :
  ∃ x₁ x₂ x₃ ∈ Ioo 0 (Real.pi / 2),
    Real.sin x₁ = Real.sqrt 3 / 3 ∧ 
    Real.cos x₂ = Real.sqrt 3 / 3 ∧ 
    Real.tan x₃ = Real.sqrt 3 / 3 ∧ 
    Real.cos (x₁ + x₂ + x₃) = -1 / 2 :=
by
  sorry

end cos_sum_of_intersections_l658_658276


namespace true_proposition_is_p_and_not_q_l658_658510

/-- Proposition for log condition -/
def p : Prop := ∀ x : ℝ, x ≥ 4 → log x / log 2 ≥ 2

/-- Proposition for quadratic condition -/
def q : Prop := ∃ x : ℝ, x^2 + 2*x + 3 = 0

/-- The correct answer is (p ∧ ¬q) -/
theorem true_proposition_is_p_and_not_q (hp : p) (hq : ¬q) : 
  ((p ∧ q) = false) ∧
  ((p ∧ ¬q) = true) ∧
  ((¬p ∧ ¬q) = false) ∧
  ((¬p ∨ q) = false) :=
by
  sorry

end true_proposition_is_p_and_not_q_l658_658510


namespace great_wall_scientific_notation_l658_658662

theorem great_wall_scientific_notation : 
  (21200000 : ℝ) = 2.12 * 10^7 :=
by
  sorry

end great_wall_scientific_notation_l658_658662


namespace shaded_fraction_of_rectangle_l658_658749

theorem shaded_fraction_of_rectangle (a b : ℕ) (h_dim : a = 15 ∧ b = 24) (h_shaded : ∃ s, s = (1/3 : ℚ)) :
  ∃ f, f = (1/9 : ℚ) := 
by
  sorry

end shaded_fraction_of_rectangle_l658_658749


namespace nine_chapters_compensation_difference_l658_658561

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l658_658561


namespace tetrahedron_triangle_count_l658_658167

theorem tetrahedron_triangle_count : 
  let vertices := 4 in
  let choose_three := Nat.choose vertices 3 in
  choose_three = 4 :=
by
  have vertices : Nat := 4
  have choose_three := Nat.choose vertices 3
  show choose_three = 4
  sorry

end tetrahedron_triangle_count_l658_658167


namespace eccentricity_of_conic_section_l658_658120

theorem eccentricity_of_conic_section :
  ∃ m : ℝ, (m^2 = 16) ∧
  (let e1 := (Real.sqrt 3) / 2,
       e2 := Real.sqrt 5 in
       ∃ (e : ℝ), ((m = 4) → (e = e1)) ∧ ((m = -4) → (e = e2))) :=
by {
  use 4,
  split,
  { -- Prove that 4^2 = 16
    exact pow_two 4 },
  { 
    use (Real.sqrt 3) / 2,   -- when m = 4
    use Real.sqrt 5,         -- when m = -4
    split,
    { -- Prove that when m = 4, e = (sqrt 3) / 2
      intro h,
      rw h,
      refl },
    { -- Prove that when m = -4, e = sqrt 5
      intro h,
      rw h,
      refl }
  }
}

end eccentricity_of_conic_section_l658_658120


namespace remainder_31_l658_658536

theorem remainder_31 (x : ℤ) (h : x % 62 = 7) : (x + 11) % 31 = 18 := by
  sorry

end remainder_31_l658_658536


namespace area_abc0_eq_twice_area_hex_area_abc0_gte_four_times_area_abc_l658_658596

variables (A B C A1 B1 C1 A0 B0 C0 : Type) 
  [Triangle A B C] -- acute triangle 
  [InternalBisector A A1] [InternalBisector B B1] [InternalBisector C C1] -- bisectors intersect circumcircle
  [Intersection A A1 (ExternalBisector B) (ExternalBisector C) A0] -- intersection of bisectors
  [Intersection B B1 (ExternalBisector A) (ExternalBisector C) B0]
  [Intersection C C1 (ExternalBisector A) (ExternalBisector B) C0]

-- Auxiliary definitions for areas
variable [Area A0 B0 C0 AC1BA1CB1 ABC : Type]

-- Statement of the problems
-- Part (i)
theorem area_abc0_eq_twice_area_hex :
  Area A0 B0 C0 = 2 * Area AC1BA1CB1 :=
sorry

-- Part (ii)
theorem area_abc0_gte_four_times_area_abc :
  Area A0 B0 C0 ≥ 4 * Area ABC :=
sorry

end area_abc0_eq_twice_area_hex_area_abc0_gte_four_times_area_abc_l658_658596


namespace joe_money_left_l658_658245

theorem joe_money_left
  (joe_savings : ℕ := 6000)
  (flight_cost : ℕ := 1200)
  (hotel_cost : ℕ := 800)
  (food_cost : ℕ := 3000) :
  joe_savings - (flight_cost + hotel_cost + food_cost) = 1000 :=
by
  sorry

end joe_money_left_l658_658245


namespace evaluate_expression_l658_658079

theorem evaluate_expression (a b c : ℤ)
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3 : ℚ) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 := 
sorry

end evaluate_expression_l658_658079


namespace parallel_tangent_line_l658_658822

theorem parallel_tangent_line (c : ℝ) : 
  (2 * (0 : ℝ) + 1 * (0 : ℝ) + c) / √(2^2 + 1^2) = √5 ↔ c = 5 ∨ c = -5 :=
by
  sorry

end parallel_tangent_line_l658_658822


namespace max_pieces_is_seven_l658_658289

noncomputable def max_pieces_of_pie : ℕ :=
  let KUSOK := λ n : ℕ, -- Assume that KUSOK is a function returning a ℕ value
  let PIROG := λ kusok : ℕ, kusok * n in
  if h : KUSOK * n < 100000 then 7 else 0

theorem max_pieces_is_seven {PIROG KUSOK : ℕ} (h₁ : ∀ n : ℕ, PIROG = KUSOK * n → PIREG < 100000) :
  max_pieces_of_pie = 7 :=
by sorry

end max_pieces_is_seven_l658_658289


namespace complex_number_solution_l658_658191

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l658_658191


namespace ordered_pairs_count_l658_658515

noncomputable def num_ordered_pairs (U : Finset ℕ) (A B : Finset ℕ) : ℕ :=
  if (A ∪ B = U ∧ A.nonempty ∧ B.nonempty) then 1 else 0

noncomputable def count_ordered_pairs (U : Finset ℕ) : ℕ :=
  ∑ A in (Finset.powerset U).filter Finset.nonempty, ∑ B in (Finset.powerset U).filter Finset.nonempty, num_ordered_pairs U A B

theorem ordered_pairs_count : count_ordered_pairs (Finset.range 4) = 25 :=
by
  -- Proof skipped
  sorry

end ordered_pairs_count_l658_658515


namespace solve_quadratic_inequality_l658_658527

theorem solve_quadratic_inequality (a x : ℝ) (h : a < 1) : 
  x^2 - (a + 1) * x + a < 0 ↔ (a < x ∧ x < 1) :=
by
  sorry

end solve_quadratic_inequality_l658_658527


namespace election_total_votes_l658_658210

theorem election_total_votes (A1 B1 C1 D1 E1 F1 G1 H1 A2 B2 C2 : ℕ) (H1 : A1 + B1 + C1 + D1 + E1 + F1 + G1 + H1 = 100) (H2 : A1 = 25) (H3 : B1 = 22) (H4 : C1 = 20) (H5 : D1 = 15) (H6 : E1 = 7) (H7 : F1 = 6) (H8 : G1 = 4) (H9 : H1 = 1) (H10 : A2 = 50) (H11 : B2 = 35) (H12 : C2 = 15) (H13 : ∀ V, 0.15 * V = 960) :
  ∃ V : ℕ, 2 * V = 12800 :=
by
  sorry

end election_total_votes_l658_658210


namespace optionC_forms_set_l658_658734

-- Definitions for options
def ConsiderablyLargeNumbers := {x : ℕ // x > 1000000}  -- Example concept of large numbers.
def StudentsWithPoorEyesight := {x : ℕ // x ∈ {1, 2, 3, 4, 5}}  -- placeholder for students with poor eyesight.
def StudentsOfGrade2014 := {x : ℕ // x ∈ {1, 2, 3, 4, 5}}  -- placeholder for students of Grade 2014.
def FamousMathematicians := {y : ℕ // y ∈ {1, 2, 3}}  -- placeholder for famous mathematicians.

-- Conditions for being a set
def is_definite (S : Set α) : Prop := ∀ (x : α), x ∈ S → definite x
def is_distinct (S : Set α) : Prop := ∀ ⦃x y : α⦄, x ∈ S → y ∈ S → x = y → x = y
def is_unordered (S : Set α) : Prop := true  -- Assuming unordered property for simplification.

-- The condition that determines if an option forms a set
def forms_set (S : Set α) : Prop := is_definite S ∧ is_distinct S ∧ is_unordered S

-- Theorem to be proved
theorem optionC_forms_set : forms_set StudentsOfGrade2014 ∧ 
  ¬ forms_set ConsiderablyLargeNumbers ∧
  ¬ forms_set StudentsWithPoorEyesight ∧
  ¬ forms_set FamousMathematicians := 
by
  sorry

end optionC_forms_set_l658_658734


namespace find_second_number_l658_658324

theorem find_second_number (x y z : ℚ) (h_sum : x + y + z = 120)
  (h_ratio1 : x = (3 / 4) * y) (h_ratio2 : z = (7 / 4) * y) :
  y = 240 / 7 :=
by {
  -- Definitions provided from conditions
  sorry  -- Proof omitted
}

end find_second_number_l658_658324


namespace poly_divisible_l658_658633

noncomputable def poly_p (x : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  x^n * sin α - x * sin (n * α) + sin ((n - 1) * α)

noncomputable def poly_Q (x : ℝ) (α : ℝ) : ℝ :=
  x^2 - 2 * x * cos α + 1

theorem poly_divisible {n : ℕ} (α : ℝ) (hn : n ≠ 1) (hsin : sin α ≠ 0) (pn : 0 < n) :
  ∃ (Q : polynomial ℝ), polynomial.divisible (poly_p Q n α) := by
  sorry

end poly_divisible_l658_658633


namespace arithmetic_sequence_find_area_l658_658946

-- Define the conditions
variable (a b c A B C : ℝ)
variable (h1 : a * (cos C / 2)^2 + c * (cos A / 2)^2 = 3 / 2 * b)
variable (h2 : a + c = 2 * b)
variable (h3 : b = 2 * real.sqrt 2)
variable (h4 : B = real.pi / 3)

-- Prove that a, b, c form an arithmetic sequence.
theorem arithmetic_sequence 
    (a b c : ℝ) 
    (h : a * (cos (C / 2))^2 + c * (cos (A / 2))^2 = 3 / 2 * b) 
    (hypothesis : a + c = 2 * b) : 
    true := 
by
  sorry

-- Prove the area of triangle ABC given specific conditions
theorem find_area 
    (a b c A B C : ℝ) 
    (h1 : a * (cos (C / 2))^2 + c * (cos (A / 2))^2 = 3 / 2 * b) 
    (h2 : a + c = 2 * b) 
    (h3 : b = 2 * real.sqrt 2) 
    (h4 : B = real.pi / 3) : 
    (1 / 2) * a * c * sin B = 2 * real.sqrt 3 := 
by
  sorry

end arithmetic_sequence_find_area_l658_658946


namespace apples_per_adult_l658_658041

theorem apples_per_adult (A C a_c D : ℕ) (h₁ : A = 450) (h₂ : C = 33)
  (h₃ : a_c = 10) (h₄ : D = 40) 
  (h₅ : A - C * a_c = 120) (h₆ : 120 / D = 3) :
  ∃ a_d, a_d = 3 :=
by {
  use 3,
  sorry
}

end apples_per_adult_l658_658041


namespace min_vertical_distance_between_graphs_l658_658678

noncomputable def min_distance (x : ℝ) : ℝ :=
  |x| - (-x^2 - 4 * x - 2)

theorem min_vertical_distance_between_graphs :
  ∃ x : ℝ, ∀ y : ℝ, min_distance x ≤ min_distance y := 
    sorry

end min_vertical_distance_between_graphs_l658_658678


namespace only_B_correct_l658_658416

theorem only_B_correct : ¬ (sin (5/7 * π) > sin (4/7 * π)) ∧ (tan (15/8 * π) > tan (-π/7)) ∧ ¬ (sin (-π/5) > sin (-π/6)) ∧ ¬ (cos (-3/5 * π) > cos (-9/4 * π)) :=
by 
  sorry

end only_B_correct_l658_658416


namespace simplify_and_rationalize_l658_658652

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l658_658652


namespace find_line_eq_l_l658_658476

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 3 }
def B : Point := { x := 5, y := 4 }
def C : Point := { x := 3, y := 7 }
def D : Point := { x := 7, y := 1 }
def E : Point := { x := 10, y := 2 }
def F : Point := { x := 8, y := 6 }

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def line_eq (p : Point) (m : ℝ) : ℝ → ℝ :=
  fun x => m * (x - p.x) + p.y
  
theorem find_line_eq_l (A B C D E F : Point)
  (l : ℝ → ℝ) (k : ℝ) : 
  slope D F = 5 →
  (∀ x : ℝ, l x = 5 * x + k) →
  (∀ P : Point, (P = A ∨ P = B ∨ P = C) → dist_point_line P l = dist_point_line P (line_eq D 5)) →
  l = λ x, 5 * x - 55 / 2 :=
sorry

end find_line_eq_l_l658_658476


namespace probability_more_sons_or_daughters_l658_658617

theorem probability_more_sons_or_daughters 
  (children : ℕ)
  (genders : Fin children → Bool)
  (probability : ℚ := 0.5) 
  (n : Nat := 8) :
  (∀ i : Fin n, genders i = true ∨ genders i = false) → 
  ∑ b in Finset.univ.filter (λ g, (Finset.filter id g).card ≠ n / 2), probability^(Finset.card b) = (93 / 128) :=
sorry

end probability_more_sons_or_daughters_l658_658617


namespace negative_square_inequality_l658_658862

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l658_658862


namespace car_maintenance_fraction_l658_658798

variable (p : ℝ) (f : ℝ)

theorem car_maintenance_fraction (hp : p = 5200)
  (he : p - f * p - (p - 320) = 200) : f = 3 / 130 :=
by
  have hp_pos : p ≠ 0 := by linarith [hp]
  sorry

end car_maintenance_fraction_l658_658798


namespace length_AB_l658_658283

variable (A B P Q : Type) [linear_ordered_field B]
variable (AP PB AQ QB : B) 
variable (PQ : B)

-- Conditions
def segment_ratio_P : Prop := AP / PB = 1 / 4
def segment_ratio_Q : Prop := AQ / QB = 2 / 5
def length_PQ : Prop := PQ = 3

-- Prove that the length of \(AB\) is 48
theorem length_AB (h1 : segment_ratio_P AP PB) (h2 : segment_ratio_Q AQ QB) (h3 : length_PQ PQ) : 
  AP + PB + PQ + QB = 48 :=
by 
  -- Mathematical proof skipped (to be done manually as part of proof)
  sorry

end length_AB_l658_658283


namespace monotonically_increasing_intervals_find_a_l658_658508

noncomputable def f (x : ℝ) : ℝ :=
  let m : ℝ × ℝ := (2 - Real.sin (2 * x + π / 6), -2)
  let n : ℝ × ℝ := (1, Real.sin x ^ 2)
  m.1 * n.1 + m.2 * n.2

theorem monotonically_increasing_intervals (k : ℤ) :
  ∀ x, (k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6) ↔ f x = Real.cos (2 * x + π / 3) + 1 :=
sorry

theorem find_a (B : ℝ) (a b c : ℝ) (hB2 : B = π / 6) (hb : b = 1) (hc : c = sqrt 3)
  (hab : f (B / 2) = 1) :
  a = 1 ∨ a = 2 :=
sorry

end monotonically_increasing_intervals_find_a_l658_658508


namespace probability_X_eq_Y_l658_658783

theorem probability_X_eq_Y {x y : ℝ} (hx : -15 * real.pi ≤ x)
    (hy : y ≤ 15 * real.pi) (hxy : sin (cos x) = sin (cos y)) :
    probability (λ (X Y : ℝ), X = Y) = 1 / 31 := 
begin
  sorry
end

end probability_X_eq_Y_l658_658783


namespace purely_imaginary_complex_number_l658_658919

theorem purely_imaginary_complex_number (m : ℝ) (z : ℂ) 
(hz : z = (2 * m^2 - 3 * m - 2 : ℝ) + (6 * m^2 + 5 * m + 1 : ℝ) * complex.i) :
  (2 * m^2 - 3 * m - 2 = 0) → ¬ (6 * m^2 + 5 * m + 1 = 0) → (m = -1 ∨ m = 2) :=
by
  sorry

end purely_imaginary_complex_number_l658_658919


namespace number_of_ways_to_fill_grid_l658_658819

-- Define our 4x4 grid with the pre-filled cells as a fixed parameter
def grid : Type := Array (Array (Option ℕ))

-- The condition that each row and each column must have exactly three 2s and one 0.
def valid_row (r : Array (Option ℕ)) : Prop :=
  r.filter (λ x => x = some 2).size = 3 ∧ r.filter (λ x => x = some 0).size = 1

def valid_col (g : grid) (c : ℕ) : Prop :=
  g.map (λ r => r[c]).filter (λ x => x = some 2).size = 3 ∧ g.map (λ r => r[c]).filter (λ x => x = some 0).size = 1

-- The pre-filled cells condition
def prefilled_cells (g : grid) : Prop := 
  -- Example: You need to provide specific places for the 4 prefilled cells which should be given.
  g[0][0] = some 2 ∧ g[1][1] = some 0 ∧ g[2][2] = some 2 ∧ g[3][3] = some 0

-- Main problem, defining the proof problem
theorem number_of_ways_to_fill_grid :
  ∃ (g : grid), 
    (∀ r, r < 4 → valid_row (g[r])) ∧ 
    (∀ c, c < 4 → valid_col g c) ∧ 
    prefilled_cells g ∧ 
    (∃ n : ℕ, n = 16) :=
sorry

end number_of_ways_to_fill_grid_l658_658819


namespace tan_alpha_plus_pi_over_4_eq_7_over_17_l658_658913

noncomputable def tan_alpha_plus_pi_over_4 (α : ℝ) : ℝ :=
  let sin_alpha : ℝ := -5 / 13
  let cos_alpha : ℝ := real.sqrt (1 - (sin_alpha ^ 2))
  let tan_alpha : ℝ := sin_alpha / cos_alpha
  in (1 + tan_alpha) / (1 - tan_alpha)

theorem tan_alpha_plus_pi_over_4_eq_7_over_17 (α : ℝ) (h1 : real.sin α = -5/13) (h2 : α ∈ Set.Icc (3 * π / 2) (2 * π)) :
  tan_alpha_plus_pi_over_4 α = 7 / 17 :=
by
  sorry

end tan_alpha_plus_pi_over_4_eq_7_over_17_l658_658913


namespace trigonometric_simplification_l658_658294

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.cos α ^ 2 - 1) /
  (2 * Real.tan (π / 4 - α) * Real.sin (π / 4 + α) ^ 2) = 1 :=
sorry

end trigonometric_simplification_l658_658294


namespace value_of_a5_l658_658489

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- Given conditions
def condition_a2_eq_2 : Prop := a 2 = 2
def condition_S3_eq_7 : Prop := S 3 = 7

-- We need to prove that a_5 = 16
theorem value_of_a5 
  (h1 : geometric_sequence a)
  (h2 : monotonically_increasing a)
  (h3 : sum_first_n_terms S a)
  (h4 : condition_a2_eq_2)
  (h5 : condition_S3_eq_7) : 
  a 5 = 16 := 
sorry

end value_of_a5_l658_658489


namespace problem_I_problem_II_l658_658141

-- Problem (I)
theorem problem_I (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  { x : ℝ | |2*x + a| + |2*x - 2*b| + 3 > 8 } = 
  { x : ℝ | x < -1 ∨ x > 1.5 } := by
  sorry

-- Problem (II)
theorem problem_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x : ℝ, |2*x + a| + |2*x - 2*b| + 3 ≥ 5) :
  (1 / a + 1 / b) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end problem_I_problem_II_l658_658141


namespace probability_of_picking_quarter_l658_658763

variable (valueQuarters valueNickels valuePennies : ℝ)
variable (valueQuarter valueNickel valuePenny : ℝ)

noncomputable def numQuarters := valueQuarters / valueQuarter
noncomputable def numNickels := valueNickels / valueNickel
noncomputable def numPennies := valuePennies / valuePenny
noncomputable def totalCoins := numQuarters + numNickels + numPennies

def probabilityQuarter := numQuarters / totalCoins

theorem probability_of_picking_quarter :
  probabilityQuarter valueQuarters valueNickels valuePennies valueQuarter valueNickel valuePenny = 1 / 31 :=
by
  -- Conditions:
  -- 1. The value of quarters is $10.00.
  -- 2. The value of nickels is $10.00.
  -- 3. The value of pennies is $10.00.
  -- 4. Each quarter is worth $0.25.
  -- 5. Each nickel is worth $0.05.
  -- 6. Each penny is worth $0.01.
  let valQuarters := 10.0
  let valNickels := 10.0
  let valPennies := 10.0
  let valQuarter := 0.25
  let valNickel := 0.05
  let valPenny := 0.01
  
  -- Compute results based on provided conditions
  have h_numQuarters : numQuarters valQuarters valQuarter := (10 : ℝ) / (0.25 : ℝ)
  have h_numNickels : numNickels valNickels valNickel := (10 : ℝ) / (0.05 : ℝ)
  have h_numPennies : numPennies valPennies valPenny := (10 : ℝ) / (0.01 : ℝ)
  have h_TotalCoins : totalCoins valQuarters valNickels valPennies valQuarter valNickel valPenny := h_numQuarters + h_numNickels + h_numPennies

  -- Show the final answer
  have h_ProbabilityQuarter := (h_numQuarters : ℝ) / (h_TotalCoins : ℝ)
  
  -- Expected probability
  have h_expected := 1 / 31

  sorry

end probability_of_picking_quarter_l658_658763


namespace min_distance_between_lines_t_l658_658893

theorem min_distance_between_lines_t (t : ℝ) :
  (∀ x y : ℝ, x + 2 * y + t^2 = 0) ∧ (∀ x y : ℝ, 2 * x + 4 * y + 2 * t - 3 = 0) →
  t = 1 / 2 := by
  sorry

end min_distance_between_lines_t_l658_658893


namespace range_of_a_l658_658921

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (a ∈ set.interval (-2 * real.sqrt 2) (2 * real.sqrt 2)) :=
by
  sorry

end range_of_a_l658_658921


namespace set_union_inter_example_l658_658961

open Set

theorem set_union_inter_example :
  let A := ({1, 2} : Set ℕ)
  let B := ({1, 2, 3} : Set ℕ)
  let C := ({2, 3, 4} : Set ℕ)
  (A ∩ B) ∪ C = ({1, 2, 3, 4} : Set ℕ) := by
    let A := ({1, 2} : Set ℕ)
    let B := ({1, 2, 3} : Set ℕ)
    let C := ({2, 3, 4} : Set ℕ)
    sorry

end set_union_inter_example_l658_658961


namespace smallest_even_n_divisible_by_1947_l658_658758

theorem smallest_even_n_divisible_by_1947 :
  ∃ n : ℕ, (1947 ∣ (∏ i in (finset.filter (λ x, x % 2 = 0) (finset.range (n + 1))), i)) ∧ (2 ∣ n) ∧ n = 3894 := 
sorry

end smallest_even_n_divisible_by_1947_l658_658758


namespace fred_red_marbles_l658_658788

theorem fred_red_marbles (total_marbles : ℕ) (third_marbles : ℕ) (green_marbles : ℕ) (yellow_marbles : ℕ) :
  total_marbles = 120 →
  third_marbles = total_marbles / 3 →
  third_marbles ≥ total_marbles / 3 →
  green_marbles = 10 →
  yellow_marbles = 5 →
  (total_marbles - (third_marbles + green_marbles + yellow_marbles)) = 65 :=
by
  intros h_total h_third h_third_ge h_green h_yellow
  rw h_total at *
  rw h_green at *
  rw h_yellow at *
  rw h_third at *
  simp at *
  sorry

end fred_red_marbles_l658_658788


namespace find_A_l658_658512

variable (U A CU_A : Set ℕ)

axiom U_is_universal : U = {1, 3, 5, 7, 9}
axiom CU_A_is_complement : CU_A = {5, 7}

theorem find_A (h1 : U = {1, 3, 5, 7, 9}) (h2 : CU_A = {5, 7}) : 
  A = {1, 3, 9} :=
by
  sorry

end find_A_l658_658512


namespace arccos_cos_eq_periodic_l658_658432

theorem arccos_cos_eq_periodic (theta : ℝ) (k : ℤ) (h : 0 ≤ theta - 2 * π * k ∧ theta - 2 * π * k ≤ π) : 
  Real.arccos (Real.cos theta) = theta - 2 * π * k := 
sorry

example : Real.arccos (Real.cos 9) = 9 - 2 * π := 
by 
  apply arccos_cos_eq_periodic
  sorry

end arccos_cos_eq_periodic_l658_658432


namespace new_mean_and_variance_l658_658389

/-- The mean of a list of numbers. -/
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

/-- The variance of a list of numbers. -/
def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem new_mean_and_variance (salaries : List ℝ) (h_len : salaries.length = 10) (mean_x : ℝ) (var_x : ℝ)
  (h_mean : mean salaries = mean_x) (h_var : variance salaries = var_x) :
  let new_salaries := salaries.map (λ x => x + 100)
  mean new_salaries = mean_x + 100 ∧ variance new_salaries = var_x :=
by {
  sorry
}

end new_mean_and_variance_l658_658389


namespace H_range_l658_658360

noncomputable def H (x : ℝ) : ℝ :=
  if x < -2 then -4
  else if x < 2 then 2 * x
  else 4

theorem H_range : Set.range H = set.Icc (-4) 4 :=
sorry

end H_range_l658_658360


namespace determine_m_factorial_l658_658769

theorem determine_m_factorial :
  (6! * 11!) = 18 * 8! * 2 :=
by
  sorry

end determine_m_factorial_l658_658769


namespace face_value_of_shares_l658_658000

def dividend_rate : ℝ := 0.185
def investment_return_rate : ℝ := 0.25
def investment_amount : ℝ := 37

theorem face_value_of_shares : ∃ (F : ℝ), (dividend_rate * F = investment_return_rate * investment_amount) ∧ F = 50 := 
by
  use 50
  split
  { simp [dividend_rate, investment_return_rate, investment_amount],
    norm_num }
  { refl }

end face_value_of_shares_l658_658000


namespace intersection_eq_l658_658150

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | y > 1}

theorem intersection_eq : {x | ∃ y, (y = log (4 - x^2) ∧ y > 1)} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_l658_658150


namespace finitely_many_solutions_for_nonconstant_polynomial_l658_658255

open Real

-- Define the polynomial P
variable {P : ℝ → ℝ} [Polynomial P]

-- Define the conditions for the integrals
def integral_sin_eq_zero (x : ℝ) : Prop :=
  ∫ t in 0..x, P t * sin t = 0

def integral_cos_eq_zero (x : ℝ) : Prop :=
  ∫ t in 0..x, P t * cos t = 0

-- The combination of both conditions
def both_integrals_zero (x : ℝ) : Prop :=
  integral_sin_eq_zero x ∧ integral_cos_eq_zero x

-- The main theorem
theorem finitely_many_solutions_for_nonconstant_polynomial 
  (P_nonconstant : ¬ (∀ t, P t = 0)) :
  {x : ℝ | both_integrals_zero x}.finite :=
sorry

end finitely_many_solutions_for_nonconstant_polynomial_l658_658255


namespace find_m_for_local_maximum_l658_658889

open Set Filter

variable {ℝ : Type*} [normed_field ℝ] [normed_space ℝ ℝ]

noncomputable def f (x m : ℝ) := x * (x - m)^2

theorem find_m_for_local_maximum :
  ∃ m : ℝ, has_local_max (λ x : ℝ, f x m) 1 ∧ m = 3 :=
begin
  sorry
end

end find_m_for_local_maximum_l658_658889


namespace find_f_cos_x_l658_658484

theorem find_f_cos_x (f : ℝ → ℝ) (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
  (h2 : f (sin x) = 2 - cos (2 * x)) : f (cos x) = 2 + cos (2 * x) :=
sorry

end find_f_cos_x_l658_658484


namespace sum_of_coordinates_l658_658579

theorem sum_of_coordinates (S : Finset (ℝ × ℝ)) (hS : ∀ p ∈ S, |p.1 - 5| = |p.2 - 7| ∧ |p.1 - 7| = 3 * |p.2 - 5|) :
  (∑ p in S, p.1 + p.2) = 28 :=
sorry

end sum_of_coordinates_l658_658579


namespace minimum_g_value_l658_658442

noncomputable def g (x : ℝ) := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem minimum_g_value :
  ∀ x ≥ (1 : ℝ), g x = (47 / 16) := sorry

end minimum_g_value_l658_658442


namespace find_angle_between_vectors_l658_658966
open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

noncomputable def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1
def is_linearly_independent (S : Set (EuclideanSpace ℝ (Fin 3))) : Prop := LinearIndependent ℝ (fun x => x : S → EuclideanSpace ℝ (Fin 3))

theorem find_angle_between_vectors 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : is_unit_vector c) 
  (h_lin_indep : is_linearly_independent ({a, b, c} : Set (EuclideanSpace ℝ (Fin 3))))
  (h_cross : a × (b × c) = (b + c) / (sqrt 2)) : 
  angle a b = 135 :=
sorry

end find_angle_between_vectors_l658_658966


namespace geometric_and_arithmetic_sequence_solution_l658_658205

theorem geometric_and_arithmetic_sequence_solution:
  ∃ a b : ℝ, 
    (a > 0) ∧                  -- a is positive
    (∃ r : ℝ, 10 * r = a ∧ a * r = 1 / 2) ∧   -- geometric sequence condition
    (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) ∧        -- arithmetic sequence condition
    a = Real.sqrt 5 ∧
    b = 10 - Real.sqrt 5 := 
by 
  sorry

end geometric_and_arithmetic_sequence_solution_l658_658205


namespace mona_unique_players_l658_658616

theorem mona_unique_players (groups : ℕ) (players_per_group : ℕ) (repeated1 : ℕ) (repeated2 : ℕ) :
  (groups = 9) → (players_per_group = 4) → (repeated1 = 2) → (repeated2 = 1) →
  (groups * players_per_group - (repeated1 + repeated2) = 33) :=
begin
  intros h_groups h_players_per_group h_repeated1 h_repeated2,
  rw [h_groups, h_players_per_group, h_repeated1, h_repeated2],
  norm_num,
end

end mona_unique_players_l658_658616


namespace column_1000_is_B_l658_658781

-- Definition of the column pattern
def columnPattern : List String := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]

-- Function to determine the column for a given integer
def columnOf (n : Nat) : String :=
  columnPattern.get! ((n - 2) % 10)

-- The theorem we want to prove
theorem column_1000_is_B : columnOf 1000 = "B" :=
by
  sorry

end column_1000_is_B_l658_658781


namespace problemI_proof_problemII_proof_l658_658148

-- Problem Ⅰ: Definitions and statement
def f (a b : ℤ) (x : ℝ) : ℝ := a * x^2 - 4 * b * x + 1

def setP : set ℤ := {1, 2, 3}
def setQ : set ℤ := {-1, 1, 2, 3, 4}

def isIncreasingOnI (a b : ℤ) : Prop := a > 0 ∧ (2 * b ≤ a)
def probIncreasing1 : ℚ := (5 : ℚ) / (15 : ℚ)

theorem problemI_proof : 
  (∃ (a ∈ setP) (b ∈ setQ), isIncreasingOnI a b) ↔ (probIncreasing1 = 1 / 3) := 
sorry

-- Problem Ⅱ: Definitions and statement
def region (a b : ℝ) : Prop := a + b - 8 ≤ 0 ∧ a > 0 ∧ b > 0
def isIncreasingOnII (a b : ℝ) : Prop := a > 0 ∧ (2 * b ≤ a)
def probIncreasing2 : ℚ := 1 / 3

theorem problemII_proof : 
  (∃ (a b : ℝ), region a b ∧ isIncreasingOnII a b) ↔ (probIncreasing2 = 1 / 3) :=
sorry

end problemI_proof_problemII_proof_l658_658148


namespace range_of_H_l658_658355

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_of_H_l658_658355


namespace square_area_l658_658560

theorem square_area (A B C D P Q : ℝ) (h_square: true)
  (hP_on_CD: true) (hQ_on_BC: true)
  (angle_APQ: angle A P Q = π / 2)
  (AP: dist A P = 4) (PQ: dist P Q = 3) :
  let side := (16 : ℝ) / (sqrt 17) in
  side * side = 256 / 17 :=
sorry

end square_area_l658_658560


namespace number_of_complete_remainder_sequences_l658_658095

-- Definitions and conditions
def is_remainder_sequence (P : ℤ[X]) (r : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 512 → r (2 * i - 1) = (P.eval (2 * i - 1)) % 1024

def is_complete (r : ℕ → ℕ) : Prop :=
  (multiset.map r (multiset.range 1 512)) = multiset.range_val (1, 1024, 2)

-- Theorem statement
theorem number_of_complete_remainder_sequences (P : ℤ[X]) :
  (∃ r, is_remainder_sequence P r ∧ is_complete r) → ∃ n, n <= 2^35 :=
sorry

end number_of_complete_remainder_sequences_l658_658095


namespace max_m_n_l658_658337

def lcm (a b : ℕ) : ℕ := a / Nat.gcd a b * b

def valid_mn (m n : ℕ) : Prop :=
  0 < m ∧ m < 500 ∧
  0 < n ∧ n < 500 ∧
  lcm m n = (m - n) ^ 2

theorem max_m_n : ∃ m n : ℕ, valid_mn m n ∧ m + n = 840 := 
by 
  sorry

end max_m_n_l658_658337


namespace vertex_of_quadratic_function_l658_658668

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := 3 * (x + 4)^2 - 5

-- Statement: Prove the coordinates of the vertex of the quadratic function
theorem vertex_of_quadratic_function : vertex quadratic_function = (-4, -5) :=
sorry

end vertex_of_quadratic_function_l658_658668


namespace exists_powers_of_7_difference_div_by_2021_l658_658998

theorem exists_powers_of_7_difference_div_by_2021 :
  ∃ n m : ℕ, n > m ∧ 2021 ∣ (7^n - 7^m) := 
by
  sorry

end exists_powers_of_7_difference_div_by_2021_l658_658998


namespace hyperbola_eccentricity_l658_658993

-- Definition of the hyperbola and its conditions
def hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 → True

-- The relationship defined by points P and Q on the asymptotes and a circle passing through the origin
def points_on_asymptotes (P Q : ℝ × ℝ) (a b : ℝ) (origin : ℝ × ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ,
    ((P.1 / a + P.2 / b = 1) ∧ (Q.1 / a - Q.2 / b = 1)) ∧
    ((P = origin) ∨ (Q = origin))

-- Eccentricity of the hyperbola
def eccentricity (a b : ℝ) : ℝ := (Math.sqrt ((a^2) + (b^2))) / a

-- The proof statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hyp : hyperbola a b ha hb) (orig : points_on_asymptotes (0, 0) (0, 0) a b (0, 0)) :
  eccentricity a a = Math.sqrt 2 :=
sorry

end hyperbola_eccentricity_l658_658993


namespace tangent_line_eq_l658_658086

theorem tangent_line_eq : 
  ∀ (x y : ℝ), y = sqrt x ∧ (x, y) = (1, 1) → x - 2*y + 1 = 0 := 
begin
  intros x y h,
  sorry
end

end tangent_line_eq_l658_658086


namespace solution_of_ab_l658_658835

theorem solution_of_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (ax^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3)) : 
  a * b = 24 := 
sorry

end solution_of_ab_l658_658835


namespace enrique_total_commission_l658_658072

def earning_commission (men_suits: ℕ) (men_suits_price: ℝ) (women_blouses: ℕ) (women_blouses_price: ℝ) (men_ties: ℕ) (men_ties_price: ℝ) (women_dresses: ℕ) (women_dresses_price: ℝ) (sales_tax_rate: ℝ) : ℝ := 
  let total_sales_before_tax := men_suits * men_suits_price + women_blouses * women_blouses_price + men_ties * men_ties_price + women_dresses * women_dresses_price
  let sales_tax := total_sales_before_tax * sales_tax_rate
  let total_sales_incl_tax := total_sales_before_tax + sales_tax
  let commission_first_1000 := min total_sales_before_tax 1000 * 0.10
  let remaining_sales_above_1000 := max 0 (total_sales_before_tax - 1000)
  let commission_above_1000 := remaining_sales_above_1000 * 0.15
  let women_sales := women_blouses * women_blouses_price + women_dresses * women_dresses_price
  let extra_women_commission := women_sales * 0.05
  commission_first_1000 + commission_above_1000 + extra_women_commission

theorem enrique_total_commission : earning_commission 2 600 6 50 4 30 3 150 0.06 = 298 := by
  sorry

end enrique_total_commission_l658_658072


namespace pascal_fifth_element_row_20_l658_658709

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l658_658709


namespace general_term_an_sum_Sn_l658_658566

-- Initial definitions and conditions
def seq_a : ℕ → ℕ
| 0       := 1  -- We use a_0 for simplicity, mapping it to a_1 from the problem
| (n + 1) := 2 * (seq_a n) + 1

def seq_b (n : ℕ) : ℕ := (2 * n + 1) * (seq_a n + 1)

def S (n : ℕ) : ℕ := ∑ i in range n, seq_b i

-- Statement 1: General term for {a_n}
theorem general_term_an (n : ℕ) : seq_a n + 1 = 2^n :=
sorry

-- Statement 2: Sum of the first n terms S_n
theorem sum_Sn (n : ℕ) : S n = 2 + (2 * n - 1) * 2^(n + 1) :=
sorry

end general_term_an_sum_Sn_l658_658566


namespace distinct_triangles_from_tetrahedron_l658_658160

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l658_658160


namespace polynomial_prime_count_unique_l658_658834

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m, m > 1 ∧ m < p → p % m ≠ 0

theorem polynomial_prime_count_unique :
  (Finset.card ({n : ℕ | n > 0 ∧ is_prime (n^3 - 9 * n^2 + 23 * n - 15)} : Finset ℕ)) = 1 :=
sorry

end polynomial_prime_count_unique_l658_658834


namespace pairs_count_l658_658174

noncomputable def count_pairs : ℕ :=
  {m : ℕ | ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = 51}.count id

theorem pairs_count : count_pairs = 2 :=
sorry

end pairs_count_l658_658174


namespace committee_with_chairperson_l658_658099

/-- 
There is a group of eight students. Each committee must consist of five students, and one member of each committee must be chosen as the chairperson. Prove that the number of different five-student committees with a chairperson formed is 280.
-/
theorem committee_with_chairperson (n k : ℕ) (hn : n = 8) (hk : k = 5) : ∑ (com : Finset (Fin n)) in (Finset.chooseStep k), com.card * k = 280 :=
by 
  sorry

end committee_with_chairperson_l658_658099


namespace A_share_of_profit_l658_658410

theorem A_share_of_profit
  (A_investment : ℤ) (B_investment : ℤ) (C_investment : ℤ)
  (A_profit_share : ℚ) (B_profit_share : ℚ) (C_profit_share : ℚ)
  (total_profit : ℤ) :
  A_investment = 6300 ∧ B_investment = 4200 ∧ C_investment = 10500 ∧
  A_profit_share = 0.45 ∧ B_profit_share = 0.3 ∧ C_profit_share = 0.25 ∧ 
  total_profit = 12200 →
  A_profit_share * total_profit = 5490 :=
by sorry

end A_share_of_profit_l658_658410


namespace impossible_unique_remainders_l658_658237

theorem impossible_unique_remainders (f : ℕ → ℕ) :
  (∀ i, 1 ≤ f i ∧ f i ≤ 2018) →
  (∀ i j, f i = f j → i = j) →
  (∀ i, ∃ j, (j = (i % 2018)) ∧ 
             (f i + f (nat.succ i % 2018) + 
              f (nat.succ (nat.succ i % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) % 2018) % 2018)) % 2018) :=
  false := by
  sorry

end impossible_unique_remainders_l658_658237


namespace solution_to_eqn_l658_658685

theorem solution_to_eqn (x : ℝ) : 9^x = 3^x + 2 ↔ x = Real.log 2 / Real.log 3 := 
by
  sorry

end solution_to_eqn_l658_658685


namespace unique_solution_value_k_l658_658061

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l658_658061


namespace probability_sum_even_probability_form_triangle_l658_658203
open Finset

-- Definitions based on the problem conditions
def cards : Finset ℕ := {1, 2, 3, 4, 5}
def draws (c : Finset ℕ) : Finset (ℕ × ℕ) := c.product c

-- Event A: Sum of the drawn cards is even
def eventA (d : ℕ × ℕ) : Prop := (d.fst + d.snd) % 2 = 0

-- Event B: Can form a triangle
def can_form_triangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- (I) Probability of sum being even
def probability_eventA := (filter eventA (draws cards)).card / (draws cards).card

-- (II) Probability of forming a triangle with remaining cards
def remaining_cards (a b : ℕ) : Finset (ℕ × ℕ × ℕ) := (cards.erase a).erase b

def eventB (d : ℕ) : Prop :=
  ∃ x y z, (x ∈ cards.erase d ∧ y ∈ cards.erase d ∧ z ∈ cards.erase d) ∧ can_form_triangle x y z

def probability_eventB (d : ℕ) := (filter eventB (cards.erase d)).card / (remaining_cards d (cards.erase d).min').card

-- The theorem statements
theorem probability_sum_even : probability_eventA = (2 / 5) :=
by sorry

theorem probability_form_triangle (d : ℕ) : probability_eventB d = (3 / 10) :=
by sorry

end probability_sum_even_probability_form_triangle_l658_658203


namespace lottery_probability_l658_658552

theorem lottery_probability (total_tickets winners non_winners people : ℕ)
  (h_total : total_tickets = 10) (h_winners : winners = 3) (h_non_winners : non_winners = 7) (h_people : people = 5) :
  1 - (Nat.choose non_winners people : ℚ) / (Nat.choose total_tickets people : ℚ) = 77 / 84 := 
by
  sorry

end lottery_probability_l658_658552


namespace aztec_pyramid_ramp_intersections_l658_658420

theorem aztec_pyramid_ramp_intersections
    (base_edge : ℝ)
    (side_edge : ℝ)
    (top_edge : ℝ)
    (ramp_vertex_base : ℝ)
    (ramp_vertex_top : ℝ)
    (h_base : base_edge = 81)
    (h_side : side_edge = 65)
    (h_top : top_edge = 16)
    (h_ramp_uniform : true)  -- This states that the ramp rises uniformly. More details can be filled out if precise mathematical formulation is given.
  : set (set ℝ) :=
  {s | s = {27, 45, 57}} :=
by
  sorry

end aztec_pyramid_ramp_intersections_l658_658420


namespace max_area_and_shape_triangle_cos_half_angle_l658_658199

-- Problem 1
theorem max_area_and_shape_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : A + C = 2 * B) (h2 : b = 2) :
  let area_max := (Real.sin 60)^2 * Real.sqrt 3 / 4 in
  let shape_equilateral := (A = 60) ∧ (C = 60) in
  -- area and shape when maximum is achieved
  (area_max = 3 * Real.sqrt 3 / 4) ∧ shape_equilateral := 
sorry

-- Problem 2
theorem cos_half_angle (A B C : ℝ) (h1 : A + C = 2 * B) 
  (h2 : (1 / Real.cos A) + (1 / Real.cos C) = -Real.sqrt 2 / Real.cos B) :
  let cos_half := Real.cos ((A - C) / 2) in
  cos_half = Real.sqrt 2 / 2 :=
sorry

end max_area_and_shape_triangle_cos_half_angle_l658_658199


namespace least_value_in_set_l658_658260

theorem least_value_in_set (S : Set ℕ) (hS_card : S.card = 8) (hS_sub : S ⊆ (Finset.range 16).toSet) 
  (h_property : ∀ a b ∈ S, a < b → ¬ (b % a = 0)) : 
  ∃ s ∈ S, ∀ t ∈ S, t ≥ s :=
  ∀ t ∈ S, t ≥ 5 :=
begin
  sorry
end

end least_value_in_set_l658_658260


namespace arithmetic_sequence_sum_l658_658474

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) :
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * n) →
  S (k + 2) - S k = 24 →
  k = 5 :=
by
  intros a1 ha hS hSk
  sorry

end arithmetic_sequence_sum_l658_658474


namespace simplify_and_rationalize_l658_658653

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l658_658653


namespace area_of_given_polygon_is_9_l658_658795

def point := (ℝ × ℝ)

def vertices : List point := [(0, 0), (4, 3), (7, 3), (3, 0)]

noncomputable def area (vertices: List point) : ℝ :=
  (0.5 : ℝ) * ((
    (vertices[0].1 * vertices[1].2 + 
     vertices[1].1 * vertices[2].2 + 
     vertices[2].1 * vertices[3].2 + 
     vertices[3].1 * vertices[0].2) - 
    (vertices[0].2 * vertices[1].1 + 
     vertices[1].2 * vertices[2].1 + 
     vertices[2].2 * vertices[3].1 + 
     vertices[3].2 * vertices[0].1)).abs)

theorem area_of_given_polygon_is_9 : area vertices = 9 := by
  sorry

end area_of_given_polygon_is_9_l658_658795


namespace carl_required_hours_last_week_l658_658042

-- Defining the conditions
def practice_hours_first_7_weeks : List ℕ := [14, 16, 12, 18, 15, 13, 17]
def required_average_hours_per_week : ℕ := 15
def total_weeks : ℕ := 8

-- The theorem statement
theorem carl_required_hours_last_week (x : ℕ) :
  (Sum practice_hours_first_7_weeks + x) / total_weeks = required_average_hours_per_week ↔ x = 15 := by sorry

end carl_required_hours_last_week_l658_658042


namespace max_value_of_y_l658_658366

-- Define the function based on the given conditions
def y (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

-- State the theorem to prove that y reaches its maximum at x = 5π/6 in the given domain
theorem max_value_of_y : 
  ∃ x ∈ set.Ico 0 (2 * π), y x = 2 ∧ x = 5 * π / 6 :=
by
  use (5 * π / 6)
  split
  {
    -- prove 5π/6 ∈ [0, 2π)
    sorry
  },
  {
    -- prove y(5π/6) = 2
    sorry
  }

end max_value_of_y_l658_658366


namespace action_figure_collection_complete_l658_658955

theorem action_figure_collection_complete (act_figures : ℕ) (cost_per_fig : ℕ) (extra_money_needed : ℕ) (total_collection : ℕ) 
    (h1 : act_figures = 7) 
    (h2 : cost_per_fig = 8) 
    (h3 : extra_money_needed = 72) : 
    total_collection = 16 :=
by
  sorry

end action_figure_collection_complete_l658_658955


namespace bowling_prize_orders_l658_658425

/--
In a professional bowling tournament playoff among the top 6 bowlers with the given conditions,
the total number of different prize orders that can occur is 32.
--/
theorem bowling_prize_orders :
  let number_of_choices := 2 in
  let matches := 5 in
  number_of_choices ^ matches = 32 :=
by
  sorry

end bowling_prize_orders_l658_658425


namespace value_of_sum_l658_658136

def f (x : ℝ) : ℝ :=
  1 - x + Real.log2 ((1 - x) / (1 + x))

theorem value_of_sum :
  f (1 / 2) + f (- (1 / 2)) = 2 := by
  sorry

end value_of_sum_l658_658136


namespace Molly_total_distance_l658_658607

theorem Molly_total_distance (sat_dist sun_dist : ℕ) (h_sat : sat_dist = 250) (h_sun : sun_dist = 180) :
  sat_dist + sun_dist = 430 :=
by
  rw [h_sat, h_sun]
  simp
  sorry

end Molly_total_distance_l658_658607


namespace find_angle_B_min_tan_sum_l658_658263

-- Part (1) statement
theorem find_angle_B (A B : ℝ) (a : ℝ) :
  A = π / 2 ∧ h = (sqrt 3 / 4) * a →
  (B = π / 6 ∨ B = π / 3) :=
sorry

-- Part (2) statement
theorem min_tan_sum (B C : ℝ) :
  0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧
  h = (sqrt 3 / 4) * a →
  tan B + 4 * tan C ≥ 9 * sqrt 3 / 4 :=
sorry

end find_angle_B_min_tan_sum_l658_658263


namespace holder_inequality_l658_658267

variable {p q : ℝ}
variable {n : ℕ}
variable {a b : Fin n → ℝ}

theorem holder_inequality 
  (hp_pos : 0 < p) 
  (hq_pos : 0 < q) 
  (h_reci : 1/p + 1/q = 1)
  (ha_pos : ∀ i, 0 < a i) 
  (hb_pos : ∀ i, 0 < b i) : 
  (∑ i, (a i) * (b i)) ≤ 
  ((∑ i, (a i)^p)^(1/p)) * ((∑ i, (b i)^q)^(1/q)) :=
sorry

end holder_inequality_l658_658267


namespace limit_S_n_div_n_a_n_l658_658565

noncomputable theory

open_locale big_operators

def a : ℕ → ℝ
| 1 := 1
| (n+2) := if h : a (n + 1) > 0 then classical.some (exists_imp_exists (λ _ _, true.intro) 
             (by {
                have h : a (n + 2) > a (n + 1), from sorry,
                have : a (n + 2) ^ 2 + a (n + 1) ^ 2 + 1 = 2 * (a (n + 2) * a (n + 1) + a (n + 2) + a (n + 1)), from sorry,
                exact ⟨_, this⟩,
             })) else 0

def S (n : ℕ) : ℝ :=
∑ k in finset.range n, a (k + 1)

theorem limit_S_n_div_n_a_n :
  tendsto (λ n, S n / (n * a n)) at_top (𝓝 (1/3)) :=
sorry

end limit_S_n_div_n_a_n_l658_658565


namespace measure_angle_A_of_triangle_l658_658230

theorem measure_angle_A_of_triangle {a b c : ℝ}
  (h_condition : b^2 + c^2 - a^2 = -b * c) :
  ∠A = 120 :=
sorry

end measure_angle_A_of_triangle_l658_658230


namespace median_red_envelopes_l658_658291

/--
Given the amounts and their frequencies:
- $1.78$ yuan appearing 2 times,
- $6.6$ yuan appearing 3 times,
- $8.8$ yuan appearing 3 times,
- $9.9$ yuan appearing 1 time,
prove that the median amount of money in these 9 red envelopes is $6.6$ yuan.
-/
theorem median_red_envelopes :
  let amounts : List (ℕ × ℕ) := [(178, 2), (660, 3), (880, 3), (990, 1)] in
  let total_envelopes := 9 in
  (total_envelopes = 2 + 3 + 3 + 1) →
  (List.sorted (le) [(178, 178, 660, 660, 660, 880, 880, 880, 990]) → 660) :=
sorry

end median_red_envelopes_l658_658291


namespace sum_of_adjacent_to_seven_l658_658318

-- Define the number and its properties
def number := 175
def divisors : set ℕ := {d | d ∣ number ∧ d ≠ 1}

-- State the main problem in terms of Lean
theorem sum_of_adjacent_to_seven (h : 7 ∈ divisors ∧ ∀ x y ∈ divisors, x ≠ y → gcd x y > 1 → (x ~ y)) :
  ∃ a b, a ∈ divisors ∧ b ∈ divisors ∧ 7 = gcd 7 a ∧ 7 = gcd 7 b ∧ a + b = 210 :=
begin
  sorry
end

end sum_of_adjacent_to_seven_l658_658318


namespace unique_triple_count_l658_658057

def valid_triples : set (ℤ × ℤ × ℤ) := 
  { t | let (a, b, c) := t in 
    a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ real.log b / real.log a = (c : ℝ)^3 ∧ a + b + c = 100 }

theorem unique_triple_count :
  ∃! t : ℤ × ℤ × ℤ, t ∈ valid_triples :=
sorry

end unique_triple_count_l658_658057


namespace trigonometric_identity_proof_l658_658105

theorem trigonometric_identity_proof (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = - (Real.sqrt 3 + 2) / 3 :=
by
  sorry

end trigonometric_identity_proof_l658_658105


namespace trip_duration_calc_l658_658904

def degrees_per_hour := 30
def degrees_per_minute := 6
def hour_hand_position (hour : ℕ) : ℕ := hour * degrees_per_hour
def minute_hand_catchup_rate := 5.5

def start_time_between_hands_together : ℚ := 210 / minute_hand_catchup_rate
def end_time_angle_formed (angle_degree : ℕ) : ℕ := (angle_degree + hour_hand_position 3) / degrees_per_minute

theorem trip_duration_calc :
  let start_hour := 7 in
  let end_hour := 15 in
  let start_minute := 38.18 in
  let end_minute := end_time_angle_formed 240 in
  let trip_duration := (end_hour - start_hour) * 60 + (end_minute - start_minute) in
  trip_duration = 8 * 60 + 2 := by
    sorry

end trip_duration_calc_l658_658904


namespace probability_1_le_xi_le_2_l658_658115

noncomputable def xi : ℝ → ℝ := sorry

def random_variable_normal : Prop := ∀ x : ℝ, xi x = pdf_normal x 0 1

theorem probability_1_le_xi_le_2 (H : random_variable_normal) :
  P(1 ≤ xi ≤ 2) = 0.1359 :=
 by 
  have h1 : P(1 ≤ xi ≤ 2) = P(0 ≤ xi ≤ 2) - P(0 ≤ xi ≤ 1), from sorry,
  have h2 : P(0 ≤ xi ≤ 1) = 0.3143, from sorry, -- Using given reference data
  have h3 : P(0 ≤ xi ≤ 2) = 0.4772, from sorry, -- Using given reference data
  rw [h1, h2, h3],
  norm_num
  -- 0.4772 - 0.3143 = 0.1359
  sorry

end probability_1_le_xi_le_2_l658_658115


namespace distance_AB_distance_CD_l658_658796

-- Definition of distance function in 3D space
noncomputable def distance (p q : (ℝ × ℝ × ℝ)) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Proving the distance between points A and B
theorem distance_AB :
  distance (1, 1, 0) (1, 1, 1) = 1 := by
  sorry

-- Proving the distance between points C and D
theorem distance_CD :
  distance (-3, 1, 5) (0, -2, 3) = real.sqrt 22 := by
  sorry

end distance_AB_distance_CD_l658_658796


namespace pascal_fifth_element_row_20_l658_658712

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l658_658712


namespace tetrahedron_triangle_count_l658_658168

theorem tetrahedron_triangle_count : 
  let vertices := 4 in
  let choose_three := Nat.choose vertices 3 in
  choose_three = 4 :=
by
  have vertices : Nat := 4
  have choose_three := Nat.choose vertices 3
  show choose_three = 4
  sorry

end tetrahedron_triangle_count_l658_658168


namespace distinct_students_count_l658_658786

open Set

theorem distinct_students_count 
  (germain_students : ℕ := 15) 
  (newton_students : ℕ := 12) 
  (young_students : ℕ := 9)
  (overlap_students : ℕ := 3) :
  (germain_students + newton_students + young_students - overlap_students) = 33 := 
by
  sorry

end distinct_students_count_l658_658786


namespace mixer_cost_difference_l658_658687

theorem mixer_cost_difference:
  let in_store_price := 129.99
  let radio_payment := 30.49
  let shipping and_handling := 9.99
  let radio_price := 4 * radio_payment + shipping_and_handling
  in_store_price < radio_price
proof
  (radio_price - in_store_price) * 100 = 196 :=
by
  sorry

end mixer_cost_difference_l658_658687


namespace local_maximum_at_1_2_l658_658949

noncomputable def f (x1 x2 : ℝ) : ℝ := x2^2 - x1^2
def constraint (x1 x2 : ℝ) : Prop := x1 - 2 * x2 + 3 = 0
def is_local_maximum (f : ℝ → ℝ → ℝ) (x1 x2 : ℝ) : Prop := 
∃ ε > 0, ∀ (y1 y2 : ℝ), (constraint y1 y2 ∧ (y1 - x1)^2 + (y2 - x2)^2 < ε^2) → f y1 y2 ≤ f x1 x2

theorem local_maximum_at_1_2 : is_local_maximum f 1 2 :=
sorry

end local_maximum_at_1_2_l658_658949


namespace pascal_triangle_row_20_element_5_l658_658726

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l658_658726


namespace length_of_each_part_l658_658015

theorem length_of_each_part (total_length_in_inches : ℕ) (num_parts : ℕ) 
  (h1 : total_length_in_inches = 188) (h2 : num_parts = 8) : 
  total_length_in_inches / num_parts = 23.5 := by
sorry

end length_of_each_part_l658_658015


namespace min_sum_max_sum_l658_658327

-- Define the conditions
def num_circle := { n : ℕ // n = 999 }
def values_set := { x : ℤ // x = 1 ∨ x = -1 }
axiom num_circle_valid : ∃ (s : num_circle → values_set), (∃ i j, s i ≠ s j)

-- Define the main theorem statements
theorem min_sum (s : num_circle → values_set) :
(∑ i in finset.range 999, ∏ j in finset.range 10, s ((i + j) % 999)) = -997 :=
sorry

theorem max_sum (s : num_circle → values_set) :
(∑ i in finset.range 999, ∏ j in finset.range 10, s ((i + j) % 999)) = 995 :=
sorry

end min_sum_max_sum_l658_658327


namespace distanceToFocus_l658_658481

variable {P : ℝ × ℝ} {O : ℝ × ℝ} (b : ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)

-- Defining the conditions
-- P is a point on the ellipse x^2/25 + y^2/b^2 = 1 where 0 < b < 5
def onEllipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  0 < b ∧ b < 5 ∧ (P.1^2 / 25 + P.2^2 / b^2 = 1)

-- P is not at the vertices of the ellipse
def notVertices (P : ℝ × ℝ) : Prop :=
  P.2 ≠ 0

-- F1 is the left focus of the ellipse
def isLeftFocus (P : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  F1 = (F1.1, 0) ∧ abs F1.1 = 5 - sqrt(25 - b^2) / b

-- |OP + OF1| = 8
def vectorCondition (P : ℝ × ℝ) (O : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  abs ((O.1 - P.1 + O.1 - F1.1)^2 + (O.2 - P.2 + O.2 - F1.2)^2) = 8

-- Theorem: Prove that the distance from P to the left focus F1 is 2
theorem distanceToFocus (h_P : onEllipse P b) (h_NV : notVertices P) (h_F1 : isLeftFocus P F1) (h_VC : vectorCondition P O F1) :
  abs ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = 2 :=
  sorry

end distanceToFocus_l658_658481


namespace radius_circle_B_l658_658799

theorem radius_circle_B (rA rB rD : ℝ) 
  (hA : rA = 2) (hD : rD = 2 * rA) (h_tangent : (rA + rB) ^ 2 = rD ^ 2) : 
  rB = 2 :=
by
  sorry

end radius_circle_B_l658_658799


namespace geometric_seq_common_ratio_arithmetic_seq_general_term_sk_abs_le_half_sk_not_expected_seq_l658_658660

-- Definitions for the given conditions
def expected_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (Σ i in finset.range n, a (i + 1)) = 0 ∧
  (Σ i in finset.range n, abs (a (i + 1))) = 1

-- Question 1
theorem geometric_seq_common_ratio (a : ℕ → ℝ) (k : ℕ) (hk : 0 < k) :
  expected_seq a (2 * k) →
  (∃ q : ℝ, ∀ i : ℕ, a (i + 1) = a 1 * q ^ (i - 1)) →
  ∃ q : ℝ, q = -1 := 
sorry

-- Question 2
theorem arithmetic_seq_general_term (a : ℕ → ℝ) (k : ℕ) (hk : 0 < k) :
  expected_seq a (2 * k) →
  (∀ i j : ℕ, i < j → a (i + 1) < a (j + 1)) →
  ∀ n : ℕ, a (n + 1) = (2 * n - 2 * k - 1) / (2 * k ^ 2) := 
sorry 

-- Question 3(i)
theorem sk_abs_le_half (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h : ∀ k, S k = Σ i in finset.range k, a (i + 1)) :
  expected_seq a n → 
  ∀ k : ℕ, k ≤ n → abs (S k) ≤ 1 / 2 := 
sorry

-- Question 3(ii)
theorem sk_not_expected_seq (a : ℕ → ℝ) (n m : ℕ) (S : ℕ → ℝ)
  (hS_m : S m = 1 / 2) (h : ∀ k, S k = Σ i in finset.range k, a (i + 1)) :
  expected_seq a n →
  ∀ i : ℕ, i ≤ n → ¬ expected_seq S n :=
sorry

end geometric_seq_common_ratio_arithmetic_seq_general_term_sk_abs_le_half_sk_not_expected_seq_l658_658660


namespace smallest_d_l658_658010

theorem smallest_d (d : ℝ) : 
  (∃ d, d > 0 ∧ (4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2))) → d = 2 :=
sorry

end smallest_d_l658_658010


namespace merchant_marked_price_percent_l658_658393

theorem merchant_marked_price_percent (L : ℝ) (hL : L = 100) (purchase_price : ℝ) (h1 : purchase_price = L * 0.70) (x : ℝ)
  (selling_price : ℝ) (h2 : selling_price = x * 0.75) :
  (selling_price - purchase_price) / selling_price = 0.30 → x = 133.33 :=
by
  sorry

end merchant_marked_price_percent_l658_658393


namespace pablo_days_to_complete_puzzles_l658_658626

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l658_658626


namespace zephyr_most_likely_l658_658376

-- Define the involved propositions
variables (X Y Z : Prop)

-- Conditions
axiom condition1 : Z → X
axiom condition2 : (X ∨ Y) → ¬Z

-- Problem statement
theorem zephyr_most_likely : Z :=
begin
  sorry
end

end zephyr_most_likely_l658_658376


namespace number_of_possible_b2_values_l658_658016

open Nat

theorem number_of_possible_b2_values (b : ℕ → ℕ) 
  (h_seq : ∀ n, b (n + 2) = abs (b (n + 1) - b n))
  (h_b1 : b 1 = 1024)
  (h_b2_lt_1024 : b 2 < 1024)
  (h_b1004 : b 1004 = 1) : 
  { k : ℕ | k < 1024 ∧ gcd 1024 k = 1 ∧ odd k }.card = 512 := sorry

end number_of_possible_b2_values_l658_658016


namespace proof_height_difference_l658_658183

noncomputable def height_in_inches_between_ruby_and_xavier : Prop :=
  let janet_height_inches := 62.75
  let inch_to_cm := 2.54
  let janet_height_cm := janet_height_inches * inch_to_cm
  let charlene_height := 1.5 * janet_height_cm
  let pablo_height := charlene_height + 1.85 * 100
  let ruby_height := pablo_height - 0.5
  let xavier_height := charlene_height + 2.13 * 100 - 97.75
  let paul_height := ruby_height + 50
  let height_diff_cm := xavier_height - ruby_height
  let height_diff_inches := height_diff_cm / inch_to_cm
  height_diff_inches = -18.78

theorem proof_height_difference :
  height_in_inches_between_ruby_and_xavier :=
by
  sorry

end proof_height_difference_l658_658183


namespace distance_from_focus_to_asymptote_l658_658143

def hyperbola_equation : Prop :=
  ∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1

def hyperbola_parameters (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

def right_focus : ℝ × ℝ := (5, 0)

def asymptote_equation (x y : ℝ) : Prop :=
  3 * y = 4 * x

def distance_point_to_line 
  (A B C x0 y0 : ℝ) 
  : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_from_focus_to_asymptote :
  hyperbola_equation ∧ 
  hyperbola_parameters 3 4 5 ∧ 
  asymptote_equation 4 (-3) ∧ 
  right_focus = (5, 0) →
  distance_point_to_line 4 (-3) 0 5 0 = 4 :=
by
  sorry

end distance_from_focus_to_asymptote_l658_658143


namespace pascal_fifth_element_row_20_l658_658711

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l658_658711


namespace ratio_WX_XY_l658_658679

theorem ratio_WX_XY (p q : ℝ) (h : 3 * p = 4 * q) : (4 * q) / (3 * p) = 12 / 7 := by
  sorry

end ratio_WX_XY_l658_658679


namespace count_powers_of_two_not_powers_of_four_below_100000_l658_658522

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k
def is_power_of_four (n : ℕ) : Prop := ∃ k : ℕ, n = 4^k

theorem count_powers_of_two_not_powers_of_four_below_100000 :
  (nat.filter (λ n, is_power_of_two n ∧ ¬ is_power_of_four n) (list.range 100000)).length = 8 := by sorry

end count_powers_of_two_not_powers_of_four_below_100000_l658_658522


namespace rachel_age_is_19_l658_658635

def rachel_and_leah_ages (R L : ℕ) : Prop :=
  (R = L + 4) ∧ (R + L = 34)

theorem rachel_age_is_19 : ∃ L : ℕ, rachel_and_leah_ages 19 L :=
by {
  sorry
}

end rachel_age_is_19_l658_658635


namespace sum_of_coordinates_l658_658580

theorem sum_of_coordinates (S : Finset (ℝ × ℝ)) (hS : ∀ p ∈ S, |p.1 - 5| = |p.2 - 7| ∧ |p.1 - 7| = 3 * |p.2 - 5|) :
  (∑ p in S, p.1 + p.2) = 28 :=
sorry

end sum_of_coordinates_l658_658580


namespace parabola_vertex_l658_658941

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, y = 1 / 2 * (x + 1) ^ 2 - 1 / 2) →
    (h = -1 ∧ k = -1 / 2) :=
by
  sorry

end parabola_vertex_l658_658941


namespace sum_series_eq_one_l658_658440

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (2^n + 1) / (3^(2^n) + 1)

theorem sum_series_eq_one : sum_series = 1 := 
by 
  sorry

end sum_series_eq_one_l658_658440


namespace Mona_unique_players_l658_658612

theorem Mona_unique_players :
  ∀ (g : ℕ) (p : ℕ) (r1 : ℕ) (r2 : ℕ),
  g = 9 → p = 4 → r1 = 2 → r2 = 1 →
  (g * p) - (r1 + r2) = 33 :=
by {
  intros g p r1 r2 hg hp hr1 hr2,
  rw [hg, hp, hr1, hr2],
  norm_num,
  sorry -- skipping proof as per instructions
}

end Mona_unique_players_l658_658612


namespace least_sugar_pounds_l658_658794

theorem least_sugar_pounds (f s : ℕ) (hf1 : f ≥ 7 + s / 2) (hf2 : f ≤ 3 * s) : s ≥ 3 :=
by
  have h : (5 * s) / 2 ≥ 7 := sorry
  have s_ge_3 : s ≥ 3 := sorry
  exact s_ge_3

end least_sugar_pounds_l658_658794


namespace pablo_days_to_complete_all_puzzles_l658_658624

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l658_658624


namespace jose_cupcakes_l658_658958

theorem jose_cupcakes (lemons_needed : ℕ) (tablespoons_per_lemon : ℕ) (tablespoons_per_dozen : ℕ) (target_lemons : ℕ) : 
  (lemons_needed = 12) → 
  (tablespoons_per_lemon = 4) → 
  (target_lemons = 9) → 
  ((target_lemons * tablespoons_per_lemon / lemons_needed) = 3) :=
by
  intros h1 h2 h3
  sorry

end jose_cupcakes_l658_658958


namespace trench_coat_sales_range_l658_658019

theorem trench_coat_sales_range :
  ∀ (x : ℝ), (0 < x ∧ x < 80) → ((160 - 2 * x) * x - (500 + 30 * x) ≥ 1300) ↔ (20 ≤ x ∧ x ≤ 45) :=
by
  intro x hx
  sorry

end trench_coat_sales_range_l658_658019


namespace find_WY_l658_658570

variable (X Y Z W : Type)
variable (dXZ dYZ dXW dCW dWY : ℝ)
variable (h : X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X)

def triangle : Prop := 
  dXZ = 10 ∧ dYZ = 10 ∧ dXW = 12 ∧ dCW = 4

theorem find_WY 
  (h : triangle X Y Z W ∧ dWY = 8.19) : 
  ∃ (WY : ℝ), WY = 8.19 :=
sorry

end find_WY_l658_658570


namespace seq_inequality_l658_658684

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = ((∑ i in finset.range n, a (i + 1)) / n) + 1) 

theorem seq_inequality (a : ℕ → ℚ) (h : seq a) : a 2016 < (1 / 2) + a 1000 := 
sorry

end seq_inequality_l658_658684


namespace probability_of_exactly_nine_correct_placements_is_zero_l658_658778

-- Define the number of letters and envelopes
def num_letters : ℕ := 10

-- Define the condition of letters being randomly inserted into envelopes
def random_insertion (n : ℕ) : Prop := true

-- Prove that the probability of exactly nine letters being correctly placed is zero
theorem probability_of_exactly_nine_correct_placements_is_zero
  (h : random_insertion num_letters) : 
  (∃ p : ℝ, p = 0) := 
sorry

end probability_of_exactly_nine_correct_placements_is_zero_l658_658778


namespace moles_of_water_formed_l658_658089

-- Definitions (conditions)
def reaction : String := "NaOH + HCl → NaCl + H2O"

def initial_moles_NaOH : ℕ := 1
def initial_moles_HCl : ℕ := 1
def mole_ratio_NaOH_HCl : ℕ := 1
def mole_ratio_NaOH_H2O : ℕ := 1

-- The proof problem
theorem moles_of_water_formed :
  initial_moles_NaOH = mole_ratio_NaOH_HCl →
  initial_moles_HCl = mole_ratio_NaOH_HCl →
  mole_ratio_NaOH_H2O * initial_moles_NaOH = 1 :=
by
  intros h1 h2
  sorry

end moles_of_water_formed_l658_658089


namespace volume_of_water_displaced_prove_remaining_volume_prove_l658_658002

noncomputable def volume_of_water_displaced_by_cube
  (r : ℝ) (h : ℝ) (s : ℝ) : ℝ :=
  let v := 3.125 * real.sqrt(6) in
  v 

noncomputable def remaining_volume_of_water
  (r : ℝ) (h : ℝ) (s : ℝ) : ℝ :=
  (π * r^2 * h) - (3.125 * real.sqrt(6))

theorem volume_of_water_displaced_prove
  (r : ℝ := 5) (h : ℝ := 12) (s : ℝ := 6) :
  (volume_of_water_displaced_by_cube r h s)^2 = 351.5625 :=
by
  sorry

theorem remaining_volume_prove
  (r : ℝ := 5) (h : ℝ := 12) (s : ℝ := 6) :
  remaining_volume_of_water r h s = (300 * π) - (3.125 * real.sqrt(6)) :=
by
  sorry

end volume_of_water_displaced_prove_remaining_volume_prove_l658_658002


namespace find_a_l658_658493

noncomputable def tangent_slope_at_point (x₀ : ℝ) : ℝ :=
  (x₀ - 1 - (x₀ + 1)) / (x₀ - 1) ^ 2

theorem find_a (a : ℝ) :
  let slope_tangent := tangent_slope_at_point 3,
      slope_line := -a in
  slope_tangent = -1 / 2 →
  (slope_tangent * slope_line = -1) → a = -2 :=
begin
  intros h1 h2,
  rw [h1, mul_comm] at h2,
  exact eq_of_mul_eq_mul_left (by norm_num) h2,
end

#eval find_a (3)

end find_a_l658_658493


namespace percentage_markup_l658_658014

def wholesale_cost := 200
def employee_price := 168
def discount := 0.30

theorem percentage_markup (M : ℝ) (retail_price : ℝ) :
  retail_price = wholesale_cost * (1 + M) →
  employee_price = retail_price * (1 - discount) →
  M = 0.2 :=
by
  intros h1 h2
  sorry

end percentage_markup_l658_658014


namespace time_required_for_production_l658_658029

variable (time_ratio : ℕ → ℕ)

-- Define the conditions
def time_ratio_A := 1
def time_ratio_B := 2
def time_ratio_C := 3

axiom ratio_correctness :
  time_ratio time_ratio_A = 1 ∧
  time_ratio time_ratio_B = 2 ∧
  time_ratio time_ratio_C = 3

axiom worker_efficiency (hours : ℕ) :
  ∃ x : ℕ,
  2 * x +
  3 * (2 * x) +
  4 * (3 * x) = hours

noncomputable def time_needed_to_produce (parts_A parts_B parts_C : ℕ) : ℕ :=
  (parts_A * 1 * 0.5) +
  (parts_B * 2 * 0.5) +
  (parts_C * 3 * 0.5)

theorem time_required_for_production :
  worker_efficiency 10 →
  14 * 1 * 0.5 + 10 * 2 * 0.5 + 2 * 3 * 0.5 = 20 :=
by
  intro h
  sorry

end time_required_for_production_l658_658029


namespace sin_alpha_def_cos_beta_def_sin_alpha_minus_beta_tan_alpha_plus_beta_l658_658463

-- Define the conditions
variable (α β : ℝ)

-- Conditions
def sin_alpha : ℝ := 4/5
def cos_beta : ℝ := -5/13
def alpha_in_interval : α ∈ Set.Ioo (π / 2) π := sorry
def beta_in_third_quadrant : β ∈ Set.Ioo π (3 * π / 2) := sorry

-- Theorems to prove
theorem sin_alpha_def : Real.sin α = sin_alpha := sorry
theorem cos_beta_def : Real.cos β = cos_beta := sorry

-- The first proof problem
theorem sin_alpha_minus_beta :
  Real.sin (α - β) = -56 / 65 := sorry

-- The second proof problem
theorem tan_alpha_plus_beta :
  Real.tan (α + β) = 16 / 63 := sorry

end sin_alpha_def_cos_beta_def_sin_alpha_minus_beta_tan_alpha_plus_beta_l658_658463


namespace D_periodic_l658_658433

def D (x : ℝ) : ℝ := if is_rational x then 1 else 0

theorem D_periodic : ∃ T > 0, ∀ x, D (x + T) = D x :=
  by
  use 1
  sorry

end D_periodic_l658_658433


namespace acceleration_of_piston_l658_658208

-- Defining the constants and variables.
variables (Q M τ c R : ℝ)

-- Defining the function representing the acceleration calculation.
def acceleration (Q M τ c R : ℝ) : ℝ :=
  real.sqrt (2 * Q / (M * τ^2 * (1 + c / R)))

-- Statement of the theorem.
theorem acceleration_of_piston (Q M τ c R : ℝ) :
  acceleration Q M τ c R = real.sqrt (2 * Q / (M * τ^2 * (1 + c / R))) :=
by
  sorry

end acceleration_of_piston_l658_658208


namespace remainder_of_3_pow_19_mod_5_l658_658361

theorem remainder_of_3_pow_19_mod_5 : (3 ^ 19) % 5 = 2 := by
  have h : 3 ^ 4 % 5 = 1 := by sorry
  sorry

end remainder_of_3_pow_19_mod_5_l658_658361


namespace mutually_exclusive_conditional_probability_l658_658793

variables (BagA BagB : Type) [Fintype BagA] [Fintype BagB]
variables (balls_in_A : {w : ℕ // w = 3} × {r : ℕ // r = 3} × {b : ℕ // b = 2})
variables (balls_in_B : {w : ℕ // w = 2} × {r : ℕ // r = 2} × {b : ℕ // b = 1})

-- Defining events
def A1 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'white }
def A2 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'red }
def A3 : Event := { ω | ω ∈ BagA ∧ ω.1 = 'black }
def B : Event := { ω | ω ∈ BagB ∧ ω.1 = 'red }

-- Proof outlines
theorem mutually_exclusive (A1 A2 A3 : Event) : 
  (∀ ω, ¬(A1 ω ∧ A2 ω) ∧ ¬(A2 ω ∧ A3 ω) ∧ ¬(A1 ω ∧ A3 ω)) :=
sorry

theorem conditional_probability : 
  P(B | A1) = 1/3 :=
sorry

end mutually_exclusive_conditional_probability_l658_658793


namespace alpha_eq_beta_plus_2_l658_658448

def α (n : ℕ) : ℕ :=
  -- Definition based on the number of representations of n as a sum of 1's and 2's
  sorry

def β (n : ℕ) : ℕ :=
  -- Definition based on the number of representations of n as a sum of integers greater than 1
  sorry

theorem alpha_eq_beta_plus_2 (n : ℕ) : α(n) = β(n + 2) :=
  sorry

end alpha_eq_beta_plus_2_l658_658448


namespace simplify_and_rationalize_l658_658654

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l658_658654


namespace loss_percentage_eq_100_div_9_l658_658765

theorem loss_percentage_eq_100_div_9 :
  ( ∀ C : ℝ,
    (11 * C > 1) ∧ 
    (8.25 * (1 + 0.20) * C = 1) →
    ((C - 1/11) / C * 100) = 100 / 9) 
  :=
by sorry

end loss_percentage_eq_100_div_9_l658_658765


namespace regular_polygon_sides_l658_658918

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end regular_polygon_sides_l658_658918


namespace total_ants_employed_l658_658667

-- Definitions of the given conditions
def red_ants_carry := 413
def black_ants_carry := 487
def red_ants_dig := 356
def black_ants_dig := 518
def red_ants_assemble := 298
def black_ants_assemble := 392

-- Assertion about the total number of ants
theorem total_ants_employed :
  red_ants_carry + black_ants_carry +
  red_ants_dig + black_ants_dig +
  red_ants_assemble + black_ants_assemble = 2464 :=
by
  -- Calculate each task's total
  have carry_total : red_ants_carry + black_ants_carry = 900 := by rfl
  have dig_total : red_ants_dig + black_ants_dig = 874 := by rfl
  have assemble_total : red_ants_assemble + black_ants_assemble = 690 := by rfl
  -- Combine all the totals
  calc
    red_ants_carry + black_ants_carry + red_ants_dig + black_ants_dig + red_ants_assemble + black_ants_assemble
     = (red_ants_carry + black_ants_carry) + (red_ants_dig + black_ants_dig) + (red_ants_assemble + black_ants_assemble) : by rw [add_assoc, add_assoc, add_assoc]
  ... = 900 + 874 + 690 : by rw [carry_total, dig_total, assemble_total]
  ... = 2464 := by rfl

-- sorry to skip the proof part
by sorry

end total_ants_employed_l658_658667


namespace license_plate_palindrome_prob_l658_658605

theorem license_plate_palindrome_prob :
  let num = 775
  let denom = 67600
  let gcd_val := Int.gcd num denom
  (num / gcd_val) + (denom / gcd_val) = 68475 := 
by
  let num := 775
  let denom := 67600
  let gcd_val := Int.gcd num denom
  have num_rel_prime : gcd_val = 1 := by sorry
  have simplified_num : num / gcd_val = 775 := by sorry
  have simplified_denom : denom / gcd_val = 67600 := by sorry
  calc
    (num / gcd_val) + (denom / gcd_val) = 775 + 67600 := by rw [simplified_num, simplified_denom]
    ... = 68475 := by norm_num

end license_plate_palindrome_prob_l658_658605


namespace find_n_and_d_l658_658266

theorem find_n_and_d (n d : ℕ) (hn_pos : 0 < n) (hd_digit : d < 10)
    (h1 : 3 * n^2 + 2 * n + d = 263)
    (h2 : 3 * n^2 + 2 * n + 4 = 1 * 8^3 + 1 * 8^2 + d * 8 + 1) :
    n + d = 12 := 
sorry

end find_n_and_d_l658_658266


namespace magnitude_of_T_l658_658261

noncomputable def complex_condition (i : ℂ) : ℂ :=
  (1 + i) ^ 15 - (1 - i) ^ 15

theorem magnitude_of_T (T : ℂ) (i : ℂ) (h_i : i = complex.I)
  (h_T : T = complex_condition i) : complex.abs T = 128 * real.sqrt 2 :=
by
  sorry

end magnitude_of_T_l658_658261


namespace sum_of_angles_satisfying_equation_l658_658826

open Real

theorem sum_of_angles_satisfying_equation :
  ∑ x in { x | 0 ≤ x ∧ x ≤ 360 ∧ sin x ^ 5 - cos x ^ 5 = (1 / cos x) - (1 / sin x) }, x = 270 := sorry

end sum_of_angles_satisfying_equation_l658_658826


namespace hexagon_inequality_l658_658268

-- Definitions and conditions
variables {A B C D E F G H : Type} -- Points
variables [Add α] [LT α] [add_left_cancel_semigroup α] [lt_min_order α] -- Numeric properties
variables (d_AB d_BC d_CD d_DE d_EF d_FA d_CF : α) -- Distances

-- Conditions
variables (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) (h3 : A ≠ G ∧ D ≠ H)
variables (h2 : d_AB = d_BC) (h4 : d_CD = d_DE) (h5 : d_EF = d_FA)
variables (angle_BCD angle_EFA : α) (h6 : angle_BCD = 60) (h7 : angle_EFA = 60)
variables (angle_AGB angle_DHE : α) (h8 : angle_AGB = 120) (h9 : angle_DHE = 120)

-- The proof goal
theorem hexagon_inequality
  (h1 : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧ (A ≠ G ∧ D ≠ H))
  (h2 : d_AB = d_BC)
  (h4 : d_CD = d_DE) 
  (h5 : d_EF = d_FA)
  (h6 : angle_BCD = 60)
  (h7 : angle_EFA = 60)
  (h8 : angle_AGB = 120)
  (h9 : angle_DHE = 120) :
  AG + GB + GH + DH + HE ≥ CF :=
sorry

end hexagon_inequality_l658_658268


namespace train_crossing_time_l658_658948

-- Define the given conditions
def length_of_train : ℝ := 750
def speed_km_hr : ℝ := 300
def conversion_factor : ℝ := 1000 / 3600

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℝ := speed_km_hr * conversion_factor

-- Desired result: time taken for the train to cross the pole
def correct_time : ℝ := 9

-- Define the theorem
theorem train_crossing_time (length_of_train = 750) (speed_km_hr = 300) (conversion_factor = 1000 / 3600) : 
  (length_of_train / (speed_km_hr * conversion_factor)) ≈ 9 :=
sorry

end train_crossing_time_l658_658948


namespace union_correct_l658_658119

variable (x : ℝ)
def A := {x | -2 < x ∧ x < 1}
def B := {x | 0 < x ∧ x < 3}
def unionSet := {x | -2 < x ∧ x < 3}

theorem union_correct : ( {x | -2 < x ∧ x < 1} ∪ {x | 0 < x ∧ x < 3} ) = {x | -2 < x ∧ x < 3} := by
  sorry

end union_correct_l658_658119


namespace distinct_triangles_from_tetrahedron_l658_658158

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l658_658158


namespace lock_code_count_l658_658683

def even_digits : set ℕ := {2, 4, 6, 8}
def prime_digits : set ℕ := {2, 3, 5, 7}
def digits : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_lock_code (code : list ℕ) : Prop :=
  (∀ d ∈ code, d ∈ digits) ∧
  (code.nodup) ∧
  (code.length = 6) ∧
  (code.nth_le 0 (by simp) ∈ even_digits) ∧
  (code.nth_le 2 (by simp) ∈ even_digits) ∧
  (code.nth_le 4 (by simp) ∈ even_digits) ∧
  (code.nth_le 1 (by simp) ∈ prime_digits) ∧
  (∀ i, i < 5 → abs (code.nth_le i (by simpa using i)) (by simpa using i+1) > 1)

noncomputable def number_of_valid_codes : ℕ :=
  (finset.univ.filter valid_lock_code).card

theorem lock_code_count : number_of_valid_codes = 1440 := 
  by sorry

end lock_code_count_l658_658683


namespace solve_a_b_solve_t_CM_eq_BN_solve_t_meeting_scenarios_l658_658123

-- Definition of points A, B, and C
variables (a b t : ℝ)
def point_A := -12
def point_B := 20
def point_C := 4

-- Given condition
def given_condition := |a + 12| + (b - 20)^2 = 0

-- Proof problem 1: Solve for a and b
theorem solve_a_b (h : given_condition) : a = -12 ∧ b = 20 := by sorry

-- Points M and N with their speeds
def speed_M := 5
def speed_N := 3

-- Scenario where CM = BN
def scenario_1_M_eq_N (t : ℝ) : 3 * t = (4 - 5 * t) + 16 := by sorry

-- Proof problem 2: Solve for t when CM = BN
theorem solve_t_CM_eq_BN (h1 : scenario_1_M_eq_N t) : t = 2 := by sorry

-- Total distances in various paths of points M and N
def total_distance_N := 48
def total_time_N := total_distance_N / speed_N

-- Distances combined 
def meeting_scenario_1 (t : ℝ) : 5 * t + 3 * t = 32 := by sorry
def meeting_scenario_2 (t : ℝ) : 3 * t - 32 + 5 * t - 32 = 32 := by sorry
def meeting_scenario_3 (t : ℝ) : 5 * t - 32 * 2 = 3 * t - 32 := by sorry

-- Proof problem 3: Solve for t when points M and N meet under given conditions
theorem solve_t_meeting_scenarios 
  (h2 : meeting_scenario_1 t ∨ meeting_scenario_2 t ∨ meeting_scenario_3 t) 
  : t = 4 ∨ t = 12 ∨ t = 16 := by sorry

end solve_a_b_solve_t_CM_eq_BN_solve_t_meeting_scenarios_l658_658123


namespace class_average_l658_658531

theorem class_average
  (n : ℕ) (h1 : n = 100)
  (p1 p2 p3 : ℝ) (h2 : p1 = 20.5) (h3 : p2 = 55.5) (h4 : p3 = 100 - (20.5 + 55.5))
  (avg1 avg2 avg3 : ℝ) (h5 : avg1 = 98.5) (h6 : avg2 = 76.8) (h7 : avg3 = 67.1) :
  (⟦(p1 * avg1) + (p2 * avg2) + (p3 * avg3) / 100⟧.floor : int) = 79 :=
sorry

end class_average_l658_658531


namespace find_a_l658_658189

open Complex

theorem find_a (a : ℝ) (h : (⟨a, 1⟩ * ⟨1, -a⟩ = 2)) : a = 1 :=
sorry

end find_a_l658_658189


namespace min_questions_to_determine_grid_l658_658960

theorem min_questions_to_determine_grid (m n : ℕ) (hm : m > 0) (hn : n > 0)
  (hmn : ∀ (x y : ℤ), 
    (∑ i in range m, ∑ j in range n, f (x + i) (y + j)) = 0 ∧ 
    (∑ i in range n, ∑ j in range m, f (x + i) (y + j)) = 0) : 
  (∃ (k : ℕ), if gcd m n = 1 then k = (m-1)^2 + (n-1)^2 else false) :=
sorry

end min_questions_to_determine_grid_l658_658960


namespace bracelet_arrangements_l658_658936

theorem bracelet_arrangements : (8.factorial / (8 * 2)) = 2520 := by
  sorry

end bracelet_arrangements_l658_658936


namespace intersection_sets_l658_658511

open Set

theorem intersection_sets:
  let M := {x : ℝ | x > 1}
  let N := {x : ℝ | x^2 - 2 * x ≥ 0}
  M ∩ N = {x : ℝ | x ≥ 2} :=
by {
  let M := {x : ℝ | x > 1},
  let N := {x : ℝ | x^2 - 2 * x ≥ 0},
  have eq_N : N = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by {
    sorry
  },
  have eq_inter : M ∩ N = {x : ℝ | x ≥ 2} := by {
    sorry
  },
  rw [eq_N, eq_inter],
}

end intersection_sets_l658_658511


namespace relationship_between_variables_l658_658636

theorem relationship_between_variables
  (a b x y : ℚ)
  (h1 : x + y = a + b)
  (h2 : y - x < a - b)
  (h3 : b > a) :
  y < a ∧ a < b ∧ b < x :=
sorry

end relationship_between_variables_l658_658636


namespace proof_problem_l658_658367

-- Definitions based on the conditions from the problem
def optionA (A : Set α) : Prop := ∅ ∩ A = ∅

def optionC : Prop := { y | ∃ x, y = 1 / x } = { z | ∃ t, z = 1 / t }

-- The main theorem statement
theorem proof_problem (A : Set α) : optionA A ∧ optionC := by
  -- Placeholder for the proof
  sorry

end proof_problem_l658_658367


namespace arithmetic_sequence_a3_l658_658555

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3_l658_658555


namespace lateral_surface_area_of_cone_l658_658125

-- Define the axial section of the cone as an equilateral triangle with side length 2
def axial_section_is_equilateral_triangle (side_length : ℝ) : Prop :=
  side_length = 2

-- Define the base radius derived from the equilateral triangle
def base_radius (side_length : ℝ) : ℝ :=
  side_length / 2

-- Define the circumference of the base of the cone
def circumference_of_base (radius : ℝ) : ℝ :=
  2 * Real.pi * radius

-- Define the lateral surface area of the cone based on provided conditions
def lateral_surface_area (side_length : ℝ) : ℝ :=
  (1 / 2) * (circumference_of_base (base_radius side_length)) * side_length

-- The theorem statement asserting the lateral surface area of the cone
theorem lateral_surface_area_of_cone : axial_section_is_equilateral_triangle 2 → lateral_surface_area 2 = 2 * Real.pi :=
by
  intros h
  unfold lateral_surface_area base_radius circumference_of_base
  rw [axial_section_is_equilateral_triangle] at h
  rw [h]
  -- (further proof steps would be here)
  sorry

end lateral_surface_area_of_cone_l658_658125


namespace concurrency_and_cocyclicity_l658_658582

theorem concurrency_and_cocyclicity
  (A B C D E F : Point)
  (Γ1 Γ2 : Circle)
  (hABCD : cyclic_quadrilateral A B C D Γ1)
  (hCDEF : cyclic_quadrilateral C D E F Γ2) 
  (hNonParallel : ¬parallel (line_through A B) (line_through C D) ∧ 
                   ¬parallel (line_through C D) (line_through E F) ∧ 
                   ¬parallel (line_through E F) (line_through A B)) :
  (concurrent (line_through A B) (line_through C D) (line_through E F)) ↔ 
  (cocyclic A B E F) := 
sorry

end concurrency_and_cocyclicity_l658_658582


namespace relationship_among_abc_l658_658589

noncomputable def a : ℝ := 0.3^3
noncomputable def b : ℝ := 3^0.3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem relationship_among_abc : b > a ∧ a > c :=
by
  sorry

end relationship_among_abc_l658_658589


namespace more_people_this_week_l658_658228

-- Define the conditions
variables (second_game first_game third_game : ℕ)
variables (total_last_week total_this_week : ℕ)

-- Conditions
def condition1 : Prop := second_game = 80
def condition2 : Prop := first_game = second_game - 20
def condition3 : Prop := third_game = second_game + 15
def condition4 : Prop := total_last_week = 200
def condition5 : Prop := total_this_week = second_game + first_game + third_game

-- Theorem statement
theorem more_people_this_week (h1 : condition1)
                             (h2 : condition2)
                             (h3 : condition3)
                             (h4 : condition4)
                             (h5 : condition5) : total_this_week - total_last_week = 35 :=
sorry

end more_people_this_week_l658_658228


namespace quadratic_has_one_real_solution_l658_658836

theorem quadratic_has_one_real_solution (m : ℝ) : (∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → m = 6 :=
by
  sorry

end quadratic_has_one_real_solution_l658_658836


namespace find_minimum_value_find_range_of_x_l658_658885

section
variables {x a b : ℝ}

def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 2|

theorem find_minimum_value :
  ∃ x, f(x) = 5 / 2 := by
  sorry

theorem find_range_of_x (a : ℝ) (b : ℝ) (h : a ≠ 0) :
  |2 * b - a| + |b + 2 * a| ≥ |a| * (|x + 1| + |x - 1|) →
  -5 / 4 ≤ x ∧ x ≤ 5 / 4 := by
  sorry
end

end find_minimum_value_find_range_of_x_l658_658885


namespace part1_part2_l658_658154

-- Definitions for vectors and conditions
def a (x : ℝ) (m : ℝ) := (Real.sin x, m * Real.cos x)
def b := (3 : ℝ, -1 : ℝ)

-- Part 1: Prove the value given the parallel condition and m = 1
theorem part1 (x : ℝ) (h : a x 1 = b' → b' = b) : 2 * Real.sin x ^ 2 - 3 * Real.cos x ^ 2 = 3 / 2 :=
sorry

-- Part 2: Prove the range given the symmetry condition
def f (x : ℝ) := 3 * Real.sin x - Real.sqrt 3 * Real.cos x

theorem part2 : Set.range (λ (x : ℝ), f (2 * x)) = Set.Interval (Real.sqrt 3, -Real.sqrt 3) :=
sorry

end part1_part2_l658_658154


namespace Mike_hours_before_break_l658_658987

theorem Mike_hours_before_break :
  ∃ (h : ℕ), 
  (let mike_hours_before := h
       mike_speed_before := 600
       mike_pamphlets_before := mike_speed_before * h
       mike_speed_after := mike_speed_before / 3
       mike_pamphlets_after := mike_speed_after * 2
       leo_hours := h / 3
       leo_speed := 2 * mike_speed_before
       leo_pamphlets := leo_speed * leo_hours
       total_pamphlets := mike_pamphlets_before + mike_pamphlets_after + leo_pamphlets
       total_pamphlets = 9400) in h = 9 := by
{ sorry }

end Mike_hours_before_break_l658_658987


namespace geometric_statements_correct_l658_658735

theorem geometric_statements_correct :
  (let is_prism (P : Type) := ∀ (x y : P), true in
   let is_section_of_cylinder (S : Type) := ∀ (x : S), true in
   is_prism prism → 
   is_section_of_cylinder section → 
   (statement_A ↔ true) ∧ (statement_C ↔ true)) :=
begin
  sorry
end

end geometric_statements_correct_l658_658735


namespace staircase_tile_cover_possible_l658_658344
-- Import the necessary Lean Lean libraries

-- We use natural numbers here
open Nat

-- Declare the problem as a theorem in Lean
theorem staircase_tile_cover_possible (m n : ℕ) (h_m : 6 ≤ m) (h_n : 6 ≤ n) :
  (∃ a b, m = 12 * a ∧ n = b ∧ a ≥ 1 ∧ b ≥ 6) ∨ 
  (∃ c d, m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) :=
sorry

end staircase_tile_cover_possible_l658_658344


namespace average_removal_inequalities_l658_658031

theorem average_removal_inequalities
  (a : ℕ → ℝ) 
  (n m : ℕ)
  (h_seq : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → a i ≤ a j)
  (h_mn : 1 ≤ m ∧ m < n) :
  (∑ i in finset.range (m + 1), a i.succ) / m ≤ (∑ i in finset.range (n + 1), a i.succ) / n ∧
  (∑ i in finset.range (n - m), a (m + 1 + i)) / (n - m) ≥ (∑ i in finset.range (n + 1), a i.succ) / n :=
sorry

end average_removal_inequalities_l658_658031


namespace difference_in_spectators_l658_658224

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l658_658224


namespace problem1_problem2_l658_658886

-- Definition of f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

-- Definition of g(a)
def g (a : ℝ) : ℝ := f (1 / a) a

-- Problem (1): Prove that for a = 1, f(x) ≤ 5 if and only if -3 ≤ x ≤ 2
theorem problem1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 :=
by sorry

-- Problem (2): Prove that for a ≠ 0, g(a) ≤ 4 if and only if 1/2 ≤ a ≤ 3/2
theorem problem2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ 1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end problem1_problem2_l658_658886


namespace mr_callen_total_loss_l658_658278

noncomputable def total_loss : ℤ :=
  let bought_paintings_price := 15 * 60
  let bought_wooden_toys_price := 12 * 25
  let bought_handmade_hats_price := 20 * 15
  let total_bought_price := bought_paintings_price + bought_wooden_toys_price + bought_handmade_hats_price
  let sold_paintings_price := 15 * (60 - (60 * 18 / 100))
  let sold_wooden_toys_price := 12 * (25 - (25 * 25 / 100))
  let sold_handmade_hats_price := 20 * (15 - (15 * 10 / 100))
  let total_sold_price := sold_paintings_price + sold_wooden_toys_price + sold_handmade_hats_price
  total_bought_price - total_sold_price

theorem mr_callen_total_loss : total_loss = 267 := by
  sorry

end mr_callen_total_loss_l658_658278


namespace common_difference_l658_658942

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Given condition
def seequence_condition (n : ℕ+) : Prop := a n + a (n + 1) = 4 * n

-- To prove
theorem common_difference (h : ∀ n : ℕ+, sequence_condition a n) : d = 2 :=
sorry

end common_difference_l658_658942


namespace find_n_for_constant_term_l658_658666

-- Considering x is a real number and r is an integer such that the expansion is valid.
noncomputable def general_term (n : ℕ) (r : ℕ) : ℝ := (Nat.choose n r) * (-2)^r * x^((n - 3 * r) / 2)

theorem find_n_for_constant_term :
  ∃ n : ℕ, (∃ r : ℕ, r = 4 ∧ (n - 3 * r) / 2 = 0) → n = 12 :=
by
  use 12
  sorry

end find_n_for_constant_term_l658_658666


namespace find_probability_p_l658_658333

theorem find_probability_p (p : ℝ) (h1 : 0.01 ≠ 1) (h2 : coe p ≠ 1) 
    (h3 : (1 - 0.01) * (1 - p) = 0.9603) :
    p = 0.03 :=
by
  sorry

end find_probability_p_l658_658333


namespace minimum_value_for_blank_space_l658_658058

theorem minimum_value_for_blank_space :
  ∃ k : ℕ, (341 * k >= 1000 ∧ 341 * k < 10000) ∧ (∀ n : ℕ, (341 * n >= 1000 ∧ 341 * n < 10000) → k ≤ n) := 
begin
  use 3,
  split,
  { -- Proving that 341 * 3 is a four-digit number
    split,
    { exact nat.le_of_lt 1023, },
    { exact nat.lt_of_lt_of_le 1023 10000, },
  },
  { -- Proving minimality
    intros n H,
    cases H with H1 H2,
    cases H1,
    cases H2,
    exact nat.le_refl 3,
  }
end

end minimum_value_for_blank_space_l658_658058


namespace geometry_inequality_l658_658630

variables {A B C M N : Type} [MetricSpace ℝ]

def AM : ℝ
def AN : ℝ
def BM : ℝ
def NC : ℝ
def MN : ℝ

axiom area_equality : AM * AN = BM * AN + AM * NC + BM * NC

theorem geometry_inequality 
  (h1 : AM * AN = BM * AN + AM * NC + BM * NC) :
  (BM + MN + NC) / (AM + AN) > 1 / 3 :=
sorry

end geometry_inequality_l658_658630


namespace determinant_scaled_matrix_l658_658101

example (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 9 * |Matrix![[x, y], [z, w]]| := by
  sorry

theorem determinant_scaled_matrix (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 63 := by
  rw [Matrix.det_smul, h]
  norm_num
  sorry

end determinant_scaled_matrix_l658_658101


namespace polynomial_no_in_interval_l658_658317

theorem polynomial_no_in_interval (P : Polynomial ℤ) (x₁ x₂ x₃ x₄ x₅ : ℤ) :
  (-- Conditions
  P.eval x₁ = 5 ∧ P.eval x₂ = 5 ∧ P.eval x₃ = 5 ∧ P.eval x₄ = 5 ∧ P.eval x₅ = 5 ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅)
  -- No x such that -6 <= P(x) <= 4 or 6 <= P(x) <= 16
  → (∀ x : ℤ, ¬(-6 ≤ P.eval x ∧ P.eval x ≤ 4) ∧ ¬(6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
by
  intro h
  sorry

end polynomial_no_in_interval_l658_658317


namespace vector_projection_sum_seven_l658_658052

theorem vector_projection_sum_seven
(l m : ℝ → ℝ × ℝ)
(C : ℝ → ℝ)
(D : ℝ → ℝ)
(Q : ℝ → ℝ)
(v1 v2 : ℝ)
(h1 : ∀ t, l t = (2 + 5 * t, 1 - 2 * t))
(h2 : ∀ s, m s = (3 - 5 * s, 7 - 2 * s))
(h3 : ∀ t, C t = l t)
(h4 : ∀ s, D s = m s)
(h5 : ∀ t, ∃ s, Q t = (2 + 5 * t, (2 + 5 * t) / 5 + 7))
(h6 : v1 + v2 = 7) :
v1 = 2 ∧ v2 = 5 := sorry

end vector_projection_sum_seven_l658_658052


namespace pairs_count_l658_658175

noncomputable def count_pairs : ℕ :=
  {m : ℕ | ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = 51}.count id

theorem pairs_count : count_pairs = 2 :=
sorry

end pairs_count_l658_658175


namespace correct_statements_l658_658050

variables {R : Type*} [linear_ordered_field R]

def symmetric_about_line (f : R → R) (a : R) : Prop := ∀ x, f(a - x) = f(a + x)
def symmetric_about_y_axis (f : R → R) : Prop := ∀ x, f (-x) = f x
def graph_symmetric_transform (f : R → R) (g : R → R) (h : R → R) : Prop := ∀ x, f(x) = g(h(x))

theorem correct_statements (f : R → R) :
  (∀ x, f(4 - x) = f(4 + x)) ∧
  (∀ x, f(4 - x) = f(x - 4)) ∧
  (∀ x, f(4 - x) = f(4 + x)) ∧
  (∀ x, f(4 - x) = f(x - 4)) →
  (symmetric_about_line f 4) ∧
  (symmetric_about_y_axis f) ∧
  (graph_symmetric_transform (λ x, f(4 - x)) (λ x, f(4 + x)) id) ∧
  (graph_symmetric_transform (λ x, f(4 - x)) id (λ x, x - 4)) :=
by { intros, sorry }

end correct_statements_l658_658050


namespace ratio_goats_to_chickens_l658_658988

noncomputable theory

def cows : ℕ := 9
def goats (cows : ℕ) := 4 * cows
def chickens : ℕ := 18

theorem ratio_goats_to_chickens (h1 : cows = 9) (h2 : ∀ (cows : ℕ), goats cows = 4 * cows) (h3 : chickens = 18) :
  ∀ (cows chickens goats : ℕ), goats = 4 * cows → chickens = 18 → (goats / chickens) = 2 :=
by
  intros cows chickens goats
  intros h_goats h_chickens
  rw [h_goats, h_chickens]
  sorry

end ratio_goats_to_chickens_l658_658988


namespace acme_vowel_soup_word_count_l658_658030

theorem acme_vowel_soup_word_count : 
  let count_words := ∑ k in {0, 1, 2, 3}, (nat.choose 5 k) * (3^k) * (5^(5-k))
  in count_words = 31450 := sorry

end acme_vowel_soup_word_count_l658_658030


namespace zeroes_ordering_l658_658891

-- Define the functions f, g, h.
def f (x: ℝ) : ℝ := Real.exp x + x
def g (x: ℝ) : ℝ := Real.log x + x
def h (x: ℝ) : ℝ := Real.log x - 1

-- Define the zeros of these functions.
def a : ℝ := sorry  -- the zero of function f
def b : ℝ := sorry  -- the zero of function g
def c : ℝ := sorry  -- the zero of function h

-- Conditions based on the problem
axiom zero_a : f a = 0
axiom zero_b : g b = 0
axiom zero_c : h c = 0

-- State the proof problem.
theorem zeroes_ordering : a < b ∧ b < c := 
  sorry

end zeroes_ordering_l658_658891


namespace determinant_of_matrix_l658_658047

theorem determinant_of_matrix :
  matrix.det !![
    [1, 2, 0],
    [4, 5, -3],
    [7, 8, 6]
  ] = -36 := 
sorry

end determinant_of_matrix_l658_658047


namespace count_valid_ns_l658_658832

open scoped BigOperators

def divisor_sum (n : ℕ) : ℕ := ∑ d in finset.Icc 1 n, if n % d = 0 then d else 0

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, 2 ≤ m → m < n → ¬ (m ∣ n)

theorem count_valid_ns : 
  ∃ (count : ℕ), count = 26 ∧ 
  count = finset.card { n ∈ finset.Icc 1 100 | divisor_sum n < n + int.sqrt n } := 
begin
  sorry
end

end count_valid_ns_l658_658832


namespace expected_value_of_product_l658_658908

-- Define a set of 7 marbles numbered 1 through 7
def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Definition to compute the product of the numbers on two chosen marbles
def product_of_marbles (a b : ℕ) : ℕ :=
  a * b

-- Expected value function to be defined
noncomputable def expected_value_product : ℚ :=
  295 / 21

-- Lean statement for the problem
theorem expected_value_of_product :
  ∑ x in (marbles.off_diag).map (λ p, product_of_marbles p.1 p.2) / marbles.off_diag.card = expected_value_product :=
sorry

end expected_value_of_product_l658_658908


namespace relationship_abc_l658_658872

-- Define the given function and conditions
variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Hypothesis 1: Symmetry after translation
axiom h1 : ∀ x, f(x) = f(2 - x)
-- Hypothesis 2: Monotonic decreasing property on (1, +∞)
axiom h2 : ∀ x1 x2, (1 < x1 ∧ x1 < x2) → (f(x2) - f(x1)) * (x2 - x1) < 0
-- Definitions of a, b, c
noncomputable def a := f (-1/2)
noncomputable def b := f 2
noncomputable def c := f (Real.exp 1)

-- Theorem statement
theorem relationship_abc : b > a ∧ a > c := by
  -- Sorry added to skip the proof, as requested
  sorry

end relationship_abc_l658_658872


namespace find_norm_b_find_cos_theta_l658_658466

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
axiom norm_a_eq_one : ‖a‖ = 1
axiom dot_product_a_b : ⟪a, b⟫ = 1 / 4
axiom dot_product_sum_diff : ⟪a + b, a - b⟫ = 1 / 2

-- Proof statements
theorem find_norm_b : ‖b‖ = (Real.sqrt 2) / 2 := by
  sorry

theorem find_cos_theta : 
  let theta := arccos (⟪a + b, a - b⟫ / (‖a + b‖ * ‖a - b‖))
  in cos theta = (Real.sqrt 2) / 4 := by
  sorry

end find_norm_b_find_cos_theta_l658_658466


namespace negation_of_p_l658_658602

theorem negation_of_p : (¬ ∃ x : ℕ, x^2 > 4^x) ↔ (∀ x : ℕ, x^2 ≤ 4^x) :=
by
  sorry

end negation_of_p_l658_658602


namespace n_divides_2n_plus_1_implies_multiple_of_3_l658_658292

theorem n_divides_2n_plus_1_implies_multiple_of_3 {n : ℕ} (h₁ : n ≥ 2) (h₂ : n ∣ (2^n + 1)) : 3 ∣ n :=
sorry

end n_divides_2n_plus_1_implies_multiple_of_3_l658_658292


namespace number_of_arithmetic_triplets_l658_658812

theorem number_of_arithmetic_triplets : 
  let S := {0, 1, 2, ..., 15}
  let arithmetic_triplets := 
    { (a, b, c) ∈ S × S × S | 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      (exists d ∈ {1, 2, 3, 4, 5}, a < b ∧ b < c ∧ b - a = d ∧ c - b = d) 
    }
  in
  arithmetic_triplets.card = 50 := 
by 
  to_string sorry

end number_of_arithmetic_triplets_l658_658812


namespace count_non_k_nice_numbers_lt_1200_l658_658459

-- Define the concept of being k-nice
def is_k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (a ^ k).nat_divisors.count = N

-- Define the main proof statement
theorem count_non_k_nice_numbers_lt_1200 (k₆ k₉ : ℕ) (h₆ : k₆ = 6) (h₉ : k₉ = 9) :
  let count := (1200 - 1) 
  (count_non_6_nice_9_nice := count -
    (set.card {n ∈ Ico 1 1200 | n % 6 = 1} + set.card {n ∈ Ico 1 1200 | n % 9 = 1} - set.card {n ∈ Ico 1 1200 | n % 18 = 1})) in
  count_non_6_nice_9_nice = 933 :=
by
  sorry

end count_non_k_nice_numbers_lt_1200_l658_658459


namespace compare_neg_rationals_l658_658045

theorem compare_neg_rationals : (- (2 / 3) > - (3 / 4)) :=
by
  have h1 : abs (- (2 / 3)) = (2 / 3), from abs_neg (2 / 3)
  have h2 : abs (- (3 / 4)) = (3 / 4), from abs_neg (3 / 4)
  have frac_comparison : (2 / 3) < (3 / 4), sorry
  -- Given the conditions from h1 and h2, conclude the theorem
  sorry

end compare_neg_rationals_l658_658045


namespace second_smallest_natural_number_greater_than_500_has_remainder_3_when_divided_by_7_l658_658454

theorem second_smallest_natural_number_greater_than_500_has_remainder_3_when_divided_by_7 : 
  ∃ n : ℕ, n > 500 ∧ n % 7 = 3 ∧ ∀ m : ℕ, (m > 500 ∧ m % 7 = 3 ∧ m < n) → m = 507 →
  ∃ k : ℕ, n = 7 * k + 3 ∧ k = 73 :=
begin
  sorry
end

end second_smallest_natural_number_greater_than_500_has_remainder_3_when_divided_by_7_l658_658454


namespace residents_in_raduzhny_l658_658222

theorem residents_in_raduzhny (population_znoynoe : ℕ) (exceeds_avg_by : ℕ)
  (total_population_others : ℕ) (num_villages : ℕ) (total_villages : ℕ)
  (average_population : ℕ):
  population_znoynoe = 1000 →
  exceeds_avg_by = 90 →
  num_villages = 9 →
  total_villages = 10 →
  (average_population = (population_znoynoe + total_population_others) / total_villages) →
  total_population_others = (average_population * num_villages) →
  (average_population - exceeds_avg_by = 1000 - 1000) →
  average_population = 900 →
  ∃ population_raduzhny, population_raduzhny = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  use 900
  sorry

end residents_in_raduzhny_l658_658222


namespace slow_train_passing_time_l658_658761

theorem slow_train_passing_time (length_fast_train length_slow_train : ℕ) 
    (time_fast_train_pass : ℕ) :
    length_fast_train = 315 → 
    length_slow_train = 300 → 
    time_fast_train_pass = 21 → 
    let relative_speed := length_fast_train / time_fast_train_pass in
    (length_slow_train / relative_speed) = 20 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end slow_train_passing_time_l658_658761


namespace find_b_l658_658314

-- Variables representing the terms in the equations
variables (a b t : ℝ)

-- Conditions given in the problem
def cond1 : Prop := a - (t / 6) * b = 20
def cond2 : Prop := a - (t / 5) * b = -10
def t_value : Prop := t = 60

-- The theorem we need to prove
theorem find_b (H1 : cond1 a b t) (H2 : cond2 a b t) (H3 : t_value t) : b = 15 :=
by {
  -- Assuming the conditions are true
  sorry
}

end find_b_l658_658314


namespace unique_solution_exists_l658_658062

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l658_658062


namespace balance_difference_after_20_years_l658_658033

/-- Alice deposits $10,000 into an account that pays 6% interest compounded semi-annually.
    Bob deposits $10,000 into an account that pays 8% simple annual interest.
    Prove the positive difference between their balances after 20 years is $6,620. --/
theorem balance_difference_after_20_years :
  let alice_principal : ℝ := 10000
  let bob_principal : ℝ := 10000
  let alice_rate : ℝ := 0.06 / 2
  let bob_rate : ℝ := 0.08
  let compounding_periods : ℕ := 2 * 20
  let alice_balance : ℝ := alice_principal * (1 + alice_rate) ^ compounding_periods
  let bob_balance : ℝ := bob_principal * (1 + bob_rate * 20)
  (alice_balance - bob_balance).round = 6620 := 
by
  sorry

end balance_difference_after_20_years_l658_658033


namespace vector_subtraction_magnitude_l658_658868

theorem vector_subtraction_magnitude (a b : ℝ^3) 
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (angle_ab : real.angle a b = real.pi * 2 / 3) :
  ∥a - 2 • b∥ = real.sqrt 7 :=
sorry

end vector_subtraction_magnitude_l658_658868


namespace set_has_six_elements_l658_658262

theorem set_has_six_elements : 
  let Z := Int 
  let A := {x : Z | x^2 - 5 * x < 6 } 
  #elementsofA = A.card = 6 
: ∃ (A : set Int), (∀ x ∈ A, x^2 - 5 * x < 6) ∧ A.card = 6 :=
begin
  existsi {x : Int | x^2 - 5 * x < 6},
  split,
  {
    intros x hx,
    exact hx
  },
  {
    sorry
  }
end

end set_has_six_elements_l658_658262


namespace concurrent_cevians_product_l658_658947

variables {X Y Z X' Y' Z' O : Type}
variables [eqX : X] [eqY : Y] [eqZ : Z] [eqX' : X'] [eqY' : Y'] [eqZ' : Z'] [eqO : O]

-- ΔXYZ is split by cevians through point O forming concurrencies at X', Y', Z' on sides YZ, XZ, XY respectively
axiom cevians_concurrent
  (X Y Z X' Y' Z' O : Type) (hX : X) (hY : Y) (hZ : Z) (hX' : X') (hY' : Y') (hZ' : Z') (hO : O) : 
  concurrent (XX' YY' ZZ') O

-- Given \(\frac{XO}{OX'} + \frac{YO}{OY'} + \frac{ZO}{OZ'} = 105\)
axiom cevian_ratio_sum (XO OX' YO OY' ZO OZ' : ℝ) :
  (XO / OX') + (YO / OY') + (ZO / OZ') = 105

-- The theorem we aim to prove
theorem concurrent_cevians_product (XO OX' YO OY' ZO OZ' : ℝ)
  (hoc : cevians_concurrent X Y Z X' Y' Z' O) (hcrs : cevian_ratio_sum XO OX' YO OY' ZO OZ') :
  (XO / OX') * (YO / OY') * (ZO / OZ') = 107 :=
sorry

end concurrent_cevians_product_l658_658947


namespace eventually_repeating_last_two_digits_l658_658008

theorem eventually_repeating_last_two_digits (K : ℕ) : ∃ N : ℕ, ∃ t : ℕ, 
    (∃ s : ℕ, t = s * 77 + N) ∨ (∃ u : ℕ, t = u * 54 + N) ∧ (t % 100) / 10 = (t % 100) % 10 :=
sorry

end eventually_repeating_last_two_digits_l658_658008


namespace cos_double_angle_l658_658876

theorem cos_double_angle (y0 : ℝ) (h : (1 / 3)^2 + y0^2 = 1) : 
  Real.cos (2 * Real.arccos (1 / 3)) = -7 / 9 := 
by
  sorry

end cos_double_angle_l658_658876


namespace sets_intersection_l658_658881

def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

def A : Set ℤ := {-2, -1, 0, 1}

def gx (x : ℝ) : ℤ := ⌊f(x) - 1/2⌋
def g1x (x : ℝ) : ℤ := ⌊f(1 - x) - 1/2⌋

def B : Set ℤ := {y | ∃ x : ℝ, y = gx x + g1x x}

theorem sets_intersection : A ∩ B = {-1, 0} :=
sorry

end sets_intersection_l658_658881


namespace area_of_union_of_rectangle_and_circle_l658_658772

theorem area_of_union_of_rectangle_and_circle :
  let width := 8
  let length := 12
  let radius := 12
  let A_rectangle := length * width
  let A_circle := Real.pi * radius ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_rectangle + A_circle - A_overlap = 96 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_rectangle_and_circle_l658_658772


namespace equilateral_triangle_side_length_l658_658487

theorem equilateral_triangle_side_length :
  ∃ a b c : ℝ, a^2 = √3 * b + sqrt(3)/4 ∧ c^2 = √3 * b + sqrt(3)/4 ∧ 
  (a = (2 * sqrt(3) + 3) ∧ b = 0 ∧ c = (2 * sqrt(3) + 3)) :=
begin
  sorry
end

end equilateral_triangle_side_length_l658_658487


namespace saltwater_animals_l658_658903

theorem saltwater_animals (num_aquariums_saltwater : ℕ) (animals_per_aquarium : ℕ) (h1 : num_aquariums_saltwater = 22) (h2 : animals_per_aquarium = 46) : 
  num_aquariums_saltwater * animals_per_aquarium = 1012 :=
by {
  rw [h1, h2],
  norm_num,
}

end saltwater_animals_l658_658903


namespace find_Y_l658_658828

theorem find_Y :
  ∃ Y : ℤ, (19 + Y / 151) * 151 = 2912 ∧ Y = 43 :=
by
  use 43
  sorry

end find_Y_l658_658828


namespace exists_smaller_than_sum_proper_divisors_l658_658995

theorem exists_smaller_than_sum_proper_divisors (n : ℕ) (h : n > 0) :
  ∃ k ∈ (setOf (λ m, n ≤ m ∧ m < n + 12)),
  ∃ d ∈ (setOf (λ d, d ∣ k ∧ d ≠ 1 ∧ d ≠ k)), k < ∑ d) :=
sorry

end exists_smaller_than_sum_proper_divisors_l658_658995


namespace average_next_3_numbers_l658_658664

theorem average_next_3_numbers 
  (a1 a2 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_avg_total : (a1 + a2 + b1 + b2 + b3 + c1 + c2 + c3) / 8 = 25)
  (h_avg_first2: (a1 + a2) / 2 = 20)
  (h_c1_c2 : c1 + 4 = c2)
  (h_c1_c3 : c1 + 6 = c3)
  (h_c3_value : c3 = 30) :
  (b1 + b2 + b3) / 3 = 26 := 
sorry

end average_next_3_numbers_l658_658664


namespace midpoint_AO_on_γ_l658_658595

open_locale classical
noncomputable theory

-- Definitions of geometry entities
variables {A B C D E O M : Type} [is_isosceles_triangle A B C] (midpoint D A C) (circumcircle γ A B D)
          (tangent_intersects E γ A B C) (circumcenter O A B E) (midpoint M A O)

-- Theorem statement
theorem midpoint_AO_on_γ
  (h_iso : is_isosceles_triangle A B C)
  (h_D_mid : midpoint D A C)
  (h_γ : circumcircle γ A B D)
  (h_tan_int : tangent_intersects E γ A B C)
  (h_O_center : circumcenter O A B E)
  (h_M_mid : midpoint M A O) : 
  M ∈ γ := sorry

end midpoint_AO_on_γ_l658_658595


namespace triangle_inequality_l658_658023

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality_l658_658023


namespace security_to_bag_ratio_l658_658986

noncomputable def U_house : ℕ := 10
noncomputable def U_airport : ℕ := 5 * U_house
noncomputable def C_bag : ℕ := 15
noncomputable def W_boarding : ℕ := 20
noncomputable def W_takeoff : ℕ := 2 * W_boarding
noncomputable def T_total : ℕ := 180
noncomputable def T_known : ℕ := U_house + U_airport + C_bag + W_boarding + W_takeoff
noncomputable def T_security : ℕ := T_total - T_known

theorem security_to_bag_ratio : T_security / C_bag = 3 :=
by sorry

end security_to_bag_ratio_l658_658986


namespace value_of_alpha_beta_l658_658909

variable (α β : ℝ)

-- Conditions
def quadratic_eq (x: ℝ) : Prop := x^2 + 2*x - 2005 = 0

-- Lean 4 statement
theorem value_of_alpha_beta 
  (hα : quadratic_eq α) 
  (hβ : quadratic_eq β)
  (sum_roots : α + β = -2) :
  α^2 + 3*α + β = 2003 :=
sorry

end value_of_alpha_beta_l658_658909


namespace triangle_area_eq_BK_KC_cot_alpha_div_2_l658_658306

-- Define the data involved in the problem
variables {A B C K : Type} 
variables (a b c : ℝ) -- sides opposite vertices A, B, C respectively
variables (α : ℝ) -- angle at vertex A

-- Define conditions explicitly
-- BK and KC are defined based on properties of the inscribed circle
def BK : ℝ := (a + c - b) / 2
def KC : ℝ := (a + b - c) / 2

-- Definition of the cotangent function
noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

-- Statement of the problem in Lean
theorem triangle_area_eq_BK_KC_cot_alpha_div_2 
    (Δ : ℝ) :
    Δ = BK a b c * KC a b c * cot (α / 2) := sorry

end triangle_area_eq_BK_KC_cot_alpha_div_2_l658_658306


namespace intercept_count_l658_658811

-- Define the function y = sin(1/x)
def f (x : ℝ) : ℝ := sin (1 / x)

-- Define the interval (0.00005, 0.0005)
def interval (x : ℝ) : Prop := 0.00005 < x ∧ x < 0.0005

def x_intercepts_count_in_interval : ℕ :=
  let lower_bound := floor (2000 / Real.pi)
  let upper_bound := floor (20000 / Real.pi)
  upper_bound - lower_bound

theorem intercept_count : x_intercepts_count_in_interval = 5730 := by
  sorry

end intercept_count_l658_658811


namespace variance_xi_l658_658320

noncomputable def xi_values : Finset ℕ :=
  {0, 1, 2}

def P (xi : ℕ) : ℚ :=
  if xi = 0 then 1/4 else if xi = 1 then 1/2 else if xi = 2 then 1/4 else 0

def E_xi : ℚ :=
  ∑ xi in xi_values, xi * P xi

noncomputable def D_xi : ℚ :=
  ∑ xi in xi_values, (xi - E_xi) ^ 2 * P xi

theorem variance_xi : D_xi = 1/2 :=
by
  have h1 : E_xi = 1 := sorry
  have h2 : D_xi = (1/4 * (0 - 1)^2) + (1/2 * (1 - 1)^2) + (1/4 * (2 - 1)^2) := sorry
  have h3 : D_xi = 1/2 := sorry
  exact h3

end variance_xi_l658_658320


namespace range_of_a_l658_658857

def A (x : ℝ) : Prop := x^2 < x
def B (x : ℝ) (a : ℝ) : Prop := x^2 < Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (B ⊆ A ∧ B ≠ A) ↔ (a ∈ Set.Ioo 0 1 ∨ a ∈ Set.Ici (Real.exp(1 / (2 * Real.exp 1)))) :=
by
  sorry

end range_of_a_l658_658857


namespace vec_combination_l658_658110

variables (a b : ℝ × ℝ)

axiom vec_a : a = (1, 3)
axiom vec_b : b = (-2, 5)

theorem vec_combination : 3 • a - 2 • b = (7, -1) :=
by 
  rw [vec_a, vec_b]
  simp
  sorry

end vec_combination_l658_658110


namespace trigonometric_identity_proof_l658_658378

-- Define the trigonometric constants
def cos_50 : Real := Real.cos (50 * Real.pi / 180)
def tan_40 : Real := Real.tan (40 * Real.pi / 180)

-- State the theorem to prove
theorem trigonometric_identity_proof :
  4 * cos_50 - tan_40 = Real.sqrt 3 :=
  sorry

end trigonometric_identity_proof_l658_658378


namespace other_diagonal_of_rhombus_l658_658670

-- Define variables and conditions
variables (d1 d2 : ℝ) (A : ℝ)

-- Given conditions
def rhombus_conditions : Prop := d2 = 20 ∧ A = 140

-- Prove the correct answer
theorem other_diagonal_of_rhombus (h : rhombus_conditions d1 d2 A) : d1 = 14 :=
by sorry

end other_diagonal_of_rhombus_l658_658670


namespace general_term_seq_l658_658473

open Nat

-- Definition of the sequence given conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, n ≥ 1 → (n - 1) * a (n + 1) - n * a n + 1 = 0

-- To prove that the general term is a_n = n
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end general_term_seq_l658_658473


namespace shape_of_triangle_l658_658198

variable {α : Type*} [LinearOrderedField α]

def is_isosceles_triangle (a b c : α) (C : α) : Prop :=
a = 2 * b * Real.cos C

theorem shape_of_triangle {a b c : α} {C : α} 
  (h : is_isosceles_triangle a b c C) : 
  a = c :=
sorry

end shape_of_triangle_l658_658198


namespace find_values_of_a_and_omega_find_interval_monotonically_decreasing_l658_658137

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (a : ℝ) := 4 * cos (ω * x) * sin (ω * x + π / 6) + a

theorem find_values_of_a_and_omega (h1 : ∃ a ω, ω > 0 ∧ ∀ x : ℝ, f x ω a = 4 * cos (ω * x) * sin (ω * x + π / 6) + a) 
(h2 : ∃ x, ∃ y, y = 2 ∧ f x ω a = y) 
(h3 : ∃ d, d = π ∧ ∀ x1 x2 : ℝ, (x2 - x1 = d) → (f x1 ω a = f x2 ω a)) : 
(a = -1) ∧ (ω = 1) := sorry

theorem find_interval_monotonically_decreasing (h1 : ∃ a ω, ω > 0 ∧ ∀ x : ℝ, f x ω a = 4 * cos (ω * x) * sin (ω * x + π / 6) + a)
(h2 : ∃ x, ∃ y, y = 2 ∧ f x ω a = y) 
(h3 : ∃ d, d = π ∧ ∀ x1 x2 : ℝ, (x2 - x1 = d) → (f x1 ω a = f x2 ω a)) 
(h4 : a = -1) (h5 : ω = 1) : 
[π / 6, 2 * π / 3] :=
begin
  sorry
end

end find_values_of_a_and_omega_find_interval_monotonically_decreasing_l658_658137


namespace dvds_still_fit_in_book_l658_658384

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end dvds_still_fit_in_book_l658_658384


namespace locus_of_centroid_of_equilateral_triangles_l658_658472

theorem locus_of_centroid_of_equilateral_triangles {A B C : Point} 
  (nonequilateral_triangle : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (collinear_ABC' : collinear A B' C') 
  (collinear_A'BC' : collinear A' B C') 
  (collinear_A'B'C : collinear A' B' C) :
  ∃ K L M P : Point, 
  equilateral_triangle K L M ∧
  torricelli_point P K L M ∧ 
  ∀ X : Point, centroid A' B' C' X → 
  (X ∈ circumcircle K L M \ {P}) := 
begin 
  sorry 
end

end locus_of_centroid_of_equilateral_triangles_l658_658472


namespace pascal_triangle_row_20_element_5_l658_658725

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l658_658725


namespace log_16_2_eq_one_fourth_l658_658075

theorem log_16_2_eq_one_fourth (a b c : ℝ) (h1 : a = 2^4) (h2 : b = log 2 2) (h3 : c = log 2 (2^4)) : 
  log 16 2 = 1 / 4 := 
by 
  sorry

end log_16_2_eq_one_fourth_l658_658075


namespace opposite_of_neg_2022_l658_658315

theorem opposite_of_neg_2022 : ∃ y : ℤ, -2022 + y = 0 ∧ y = 2022 := 
by 
  use 2022
  split
  { exact rfl }
  { exact rfl }

end opposite_of_neg_2022_l658_658315


namespace function_C_is_the_only_odd_and_increasing_function_l658_658414

theorem function_C_is_the_only_odd_and_increasing_function :
  (∀ x : ℝ, x^3 = -(-x)^3) ∧ (∀ x y : ℝ, x < y → x^3 < y^3) ∧
  ¬((∀ x : ℝ, x - 1 = -(-x - 1)) ∧ (∀ x y : ℝ, x < y → x - 1 < y - 1)) ∧
  ¬(∀ x : ℝ, tan x = -tan (-x) ∧ (∀ x y : ℝ, x < y → tan x < tan y)) ∧
  ¬(∀ x : ℝ, log x = -log (-x) ∧ (∀ x y : ℕ, x < y → log x < log y)) :=
by
  sorry

end function_C_is_the_only_odd_and_increasing_function_l658_658414


namespace percentage_increase_is_2_l658_658411

def alan_price := 2000
def john_price := 2040
def percentage_increase (alan_price : ℕ) (john_price : ℕ) : ℕ := (john_price - alan_price) * 100 / alan_price

theorem percentage_increase_is_2 (alan_price john_price : ℕ) (h₁ : alan_price = 2000) (h₂ : john_price = 2040) :
  percentage_increase alan_price john_price = 2 := by
  rw [h₁, h₂]
  sorry

end percentage_increase_is_2_l658_658411


namespace train_pass_platform_time_l658_658402

theorem train_pass_platform_time :
  ∀ (length_train length_platform : ℕ) (speed_kmph : ℕ),
  length_train = 140 →
  length_platform = 260 →
  speed_kmph = 60 →
  (length_train + length_platform : ℝ) / (speed_kmph * 1000 / 3600 : ℝ) ≈ 24 := by
  intros length_train length_platform speed_kmph 
  intros h_train h_platform h_speed 
  rw [h_train, h_platform, h_speed]
  sorry

end train_pass_platform_time_l658_658402


namespace proof_problem_l658_658651

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l658_658651


namespace mowing_lawn_together_l658_658755

noncomputable def A_rate : ℝ := 1 / 130
noncomputable def B_rate : ℝ := 1 / 100
noncomputable def C_rate : ℝ := 1 / 150

noncomputable def combined_rate : ℝ := A_rate + B_rate + C_rate
noncomputable def total_time : ℝ := 1 / combined_rate

theorem mowing_lawn_together :
  total_time ≈ 41.05 :=
by
  sorry

end mowing_lawn_together_l658_658755


namespace fraction_to_decimal_conversion_l658_658744

theorem fraction_to_decimal_conversion : (2 : ℚ) / 25 = 0.08 := sorry

end fraction_to_decimal_conversion_l658_658744


namespace possible_values_of_x_l658_658846

theorem possible_values_of_x (x : ℤ) (p : Prop) (q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) :
  (-2 < x ∧ x < 3) → x ∈ {-1, 0, 1, 2} :=
by 
  sorry

end possible_values_of_x_l658_658846


namespace bicyclist_speed_remainder_l658_658620

noncomputable def speed_of_bicyclist (total_distance first_distance remaining_distance time_for_first_distance total_time : ℝ) : ℝ :=
  remaining_distance / (total_time - time_for_first_distance)

theorem bicyclist_speed_remainder 
  (total_distance : ℝ)
  (first_distance : ℝ)
  (remaining_distance : ℝ)
  (first_speed : ℝ)
  (average_speed : ℝ)
  (correct_speed : ℝ) :
  total_distance = 250 → 
  first_distance = 100 →
  remaining_distance = total_distance - first_distance →
  first_speed = 20 →
  average_speed = 16.67 →
  correct_speed = 15 →
  speed_of_bicyclist total_distance first_distance remaining_distance (first_distance / first_speed) (total_distance / average_speed) = correct_speed :=
by
  sorry

end bicyclist_speed_remainder_l658_658620


namespace derek_walked_distance_l658_658055

theorem derek_walked_distance
  (biking_speed : ℝ := 20)
  (walking_speed : ℝ := 4)
  (total_time_hours : ℝ := 54 / 60) :
  ∃ distance_walked : ℝ, distance_walked = 3 :=
by
  -- Definitions equivalent to the conditions
  let total_time := total_time_hours
  let total_distance_fraction := total_time * (biking_speed * walking_speed) / (biking_speed + walking_speed)
  have h : total_distance_fraction = 3, from sorry
  use total_distance_fraction
  exact h

end derek_walked_distance_l658_658055


namespace mean_of_numbers_l658_658311

theorem mean_of_numbers (a b c d : ℕ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 16) (h4 : d = 18) :
  (a + b + c + d) / 4 = 15 :=
by
  have H : a + b + c + d = 60, sorry
  have N : 4 = 4, sorry
  rw [H, N]
  norm_num
  sorry

end mean_of_numbers_l658_658311


namespace train_pass_time_l658_658404

theorem train_pass_time (train_length platform_length : ℕ) (speed_kmph : ℕ)
  (convert_factor : ℝ) (h1 : train_length = 140) (h2 : platform_length = 260)
  (h3 : speed_kmph = 60) (h4 : convert_factor = 1/3.6) : 
  let total_distance := train_length + platform_length in
  let speed_mps := speed_kmph * convert_factor in
  let time := total_distance / speed_mps in
  time ≈ 24 :=
by
  sorry

end train_pass_time_l658_658404


namespace value_of_T_l658_658681

variables {A M T E H : ℕ}

theorem value_of_T (H : ℕ) (MATH : ℕ) (MEET : ℕ) (TEAM : ℕ) (H_eq : H = 8) (MATH_eq : MATH = 47) (MEET_eq : MEET = 62) (TEAM_eq : TEAM = 58) :
  T = 9 :=
by
  sorry

end value_of_T_l658_658681


namespace rectangle_A_path_length_l658_658288

theorem rectangle_A_path_length 
  (A B C D : Point)
  (h_AB_CD : distance A B = 3 ∧ distance C D = 3)
  (h_BC_DA : distance B C = 8 ∧ distance D A = 8)
  (rot1 : rotate_90_clockwise D A = A1)
  (rot2 : ∃ C1, rotate_90_clockwise C1 A1 = A')
  : distance_traveled_by_point(A, A1, A') = (π * (8 + sqrt 73) / 2) := 
sorry

end rectangle_A_path_length_l658_658288


namespace impossible_unique_remainders_l658_658236

theorem impossible_unique_remainders (f : ℕ → ℕ) :
  (∀ i, 1 ≤ f i ∧ f i ≤ 2018) →
  (∀ i j, f i = f j → i = j) →
  (∀ i, ∃ j, (j = (i % 2018)) ∧ 
             (f i + f (nat.succ i % 2018) + 
              f (nat.succ (nat.succ i % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) % 2018) + 
              f (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ (nat.succ i % 2018) % 2018) % 2018) % 2018) % 2018) % 2018) % 2018)) % 2018) :=
  false := by
  sorry

end impossible_unique_remainders_l658_658236


namespace squirrel_travel_distance_l658_658370

-- Definitions based on the problem's conditions
def post_height : ℝ := 16
def post_circumference : ℝ := 2
def rise_per_circuit : ℝ := 4

-- Statement of the problem as a Lean theorem
theorem squirrel_travel_distance :
  let number_of_circuits := post_height / rise_per_circuit,
  one_circuit_length := real.sqrt (rise_per_circuit ^ 2 + post_circumference ^ 2),
  total_distance := number_of_circuits * one_circuit_length
  in total_distance ≈ 17.888 :=
by {
  sorry
}

end squirrel_travel_distance_l658_658370


namespace distance_A_B_l658_658628

noncomputable def distance_between_points (v_A v_B : ℝ) (t : ℝ) : ℝ := 5 * (6 * t / (2 / 3 * t))

theorem distance_A_B
  (v_A v_B : ℝ)
  (t : ℝ)
  (h1 : v_A = 1.2 * v_B)
  (h2 : ∃ distance_broken, distance_broken = 5)
  (h3 : ∃ delay, delay = (1 / 6) * 6 * t)
  (h4 : ∃ v_B_new, v_B_new = 1.6 * v_B)
  (h5 : distance_between_points v_A v_B t = 45) :
  distance_between_points v_A v_B t = 45 :=
sorry

end distance_A_B_l658_658628


namespace log_base_change_log_base_evaluation_l658_658077

-- Define the conditions as functions or constants used in the statement
theorem log_base_change 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  : log a (x) / log a (b) = log b (x) := sorry

theorem log_base_evaluation 
  : log 16 2 = 1 / 4 := by 
  have h16 : 16 = 2 ^ 4 := by norm_num
  have log_identity : log 16 2 = log (2 ^ 4) 2 := by rw h16
  have log_change : log (2 ^ 4) 2 = log 2 2 / log 2 (2 ^ 4) := log_base_change 2 2⁴ 2 (by norm_num) (by norm_num) (by norm_num)
  rw [log_change, log_self, log_pow] at log_identity
  exact log_identity

end log_base_change_log_base_evaluation_l658_658077


namespace median_number_of_children_is_three_l658_658677

/-- Define the context of the problem with total number of families. -/
def total_families : Nat := 15

/-- Prove that given the conditions, the median number of children is 3. -/
theorem median_number_of_children_is_three 
  (h : total_families = 15) : 
  ∃ median : Nat, median = 3 :=
by
  sorry

end median_number_of_children_is_three_l658_658677


namespace acute_angle_between_bisectors_l658_658207

theorem acute_angle_between_bisectors
  (A B C D : Type)
  [convex_quadrilateral A B C D]  -- Define property of convex quadrilateral
  (angle_bisectors_parallel : parallel (bisector A) (bisector C))  -- angle bisectors of A and C are parallel
  (intersection_angle : ∠(bisector B) (bisector D) = 46) : -- angle bisectors of B and D intersect at 46 degrees
  ∠(bisector A) (bisector B) = 67 := -- prove acute angle between bisectors of A and B is 67 degrees
sorry

end acute_angle_between_bisectors_l658_658207


namespace probability_sunflower_seed_l658_658421

theorem probability_sunflower_seed :
  ∀ (sunflower_seeds green_bean_seeds pumpkin_seeds : ℕ),
  sunflower_seeds = 2 →
  green_bean_seeds = 3 →
  pumpkin_seeds = 4 →
  (sunflower_seeds + green_bean_seeds + pumpkin_seeds = 9) →
  (sunflower_seeds : ℚ) / (sunflower_seeds + green_bean_seeds + pumpkin_seeds) = 2 / 9 := 
by 
  intros sunflower_seeds green_bean_seeds pumpkin_seeds h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h1, h2, h3]
  sorry -- Proof omitted as per instructions.

end probability_sunflower_seed_l658_658421


namespace game_spinner_probability_l658_658390

theorem game_spinner_probability (P_A P_B P_D P_C : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) (h₄ : P_A + P_B + P_C + P_D = 1) :
  P_C = 1/4 :=
by
  sorry

end game_spinner_probability_l658_658390


namespace trigonometric_identity_l658_658914

variable {θ u : ℝ} {n : ℤ}

-- Given condition
def cos_condition (θ u : ℝ) : Prop := 2 * Real.cos θ = u + (1 / u)

-- Theorem to prove
theorem trigonometric_identity (h : cos_condition θ u) : 2 * Real.cos (n * θ) = u^n + (1 / u^n) :=
sorry

end trigonometric_identity_l658_658914


namespace race_distance_l658_658204

theorem race_distance (T_A T_B : ℝ) (D : ℝ) (V_A V_B : ℝ)
  (h1 : T_A = 23)
  (h2 : T_B = 30)
  (h3 : V_A = D / 23)
  (h4 : V_B = (D - 56) / 30)
  (h5 : D = (D - 56) * (23 / 30) + 56) :
  D = 56 :=
by
  sorry

end race_distance_l658_658204


namespace geom_inequality_l658_658974

variables {Point : Type} [MetricSpace Point] {O A B C K L H M : Point}

/-- Conditions -/
def circumcenter_of_triangle (O A B C : Point) : Prop := 
 -- Definition that O is the circumcenter of triangle ABC
 sorry 

def midpoint_of_arc (K B C A : Point) : Prop := 
 -- Definition that K is the midpoint of the arc BC not containing A
 sorry

def lies_on_line (K L A : Point) : Prop := 
 -- Definition that K lies on line AL
 sorry

def similar_triangles (A H L K M : Point) : Prop := 
 -- Definition that triangles AHL and KML are similar
 sorry 

def segment_inequality (AL KL : ℝ) : Prop := 
 -- Definition that AL < KL
 sorry 

/-- Proof Problem -/
theorem geom_inequality (h1 : circumcenter_of_triangle O A B C) 
                       (h2: midpoint_of_arc K B C A)
                       (h3: lies_on_line K L A)
                       (h4: similar_triangles A H L K M)
                       (h5: segment_inequality (dist A L) (dist K L)) : 
  dist A K < dist B C := 
sorry

end geom_inequality_l658_658974


namespace unique_solution_value_k_l658_658060

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l658_658060


namespace correct_statements_l658_658547

def studentsPopulation : Nat := 70000
def sampleSize : Nat := 1000
def isSamplePopulation (s : Nat) (p : Nat) : Prop := s < p
def averageSampleEqualsPopulation (sampleAvg populationAvg : ℕ) : Prop := sampleAvg = populationAvg
def isPopulation (p : Nat) : Prop := p = studentsPopulation

theorem correct_statements (p s : ℕ) (h1 : isSamplePopulation s p) (h2 : isPopulation p) 
  (h4 : s = sampleSize) : 
  (isSamplePopulation s p ∧ ¬averageSampleEqualsPopulation 1 1 ∧ isPopulation p ∧ s = sampleSize) := 
by
  sorry

end correct_statements_l658_658547


namespace part_a_part_b_l658_658746

/-- Part (a): Given any natural number N, there exists a strictly increasing sequence of N positive integers in harmonic progression. -/
theorem part_a (N : ℕ) (hN : N > 0) : 
  ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (∀ i : ℕ, i < N → (1 : ℚ) / a i + (d : ℚ) = 1 / a (i + 1)) :=
sorry

/-- Part (b): There cannot exist a strictly increasing infinite sequence of positive integers which is in harmonic progression. -/
theorem part_b : 
  ¬ ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (∀ n : ℕ, (1 : ℚ) / a n + (d : ℚ) = 1 / a (n + 1)) :=
sorry

end part_a_part_b_l658_658746


namespace symmetric_polynomial_equality_l658_658972

variable {R : Type*} [CommRing R] [IsDomain R] 

noncomputable def roots_of_polynomial (P : R[X]) (n : ℕ) :=
  {x : R | xroot x P}.to_finset

noncomputable def roots_of_derivative (P : R[X]) (n : ℕ) :=
  {x : R | xroot x (P.derivative)}.to_finset

noncomputable def sigma_k (s : Finset R) (k : ℕ) :=
  (Finset.powersetLen k s).sum (λ t, t.prod id)

theorem symmetric_polynomial_equality {P : R[X]} {n k : ℕ} :
  let x_roots := roots_of_polynomial P n
      y_roots := roots_of_derivative P n
  in sigma_k x_roots k = sigma_k y_roots k :=
sorry

end symmetric_polynomial_equality_l658_658972


namespace value_of_a_l658_658623

theorem value_of_a :
  ∀ (a : ℤ) (BO CO : ℤ), 
  BO = 2 → 
  CO = 2 * BO → 
  |a + 3| = CO → 
  a < 0 → 
  a = -7 := by
  intros a BO CO hBO hCO hAbs ha_neg
  sorry

end value_of_a_l658_658623


namespace price_of_uniform_l658_658737

theorem price_of_uniform (u : ℝ) : 
  let total_amount := 800 in
  let months_worked := 9 in
  let salary_received := 400 in
  let frac_of_year := 3 / 4 in
  total_amount * frac_of_year - salary_received = u ->
  u = 200 :=
by
  intro h
  sorry

end price_of_uniform_l658_658737


namespace max_area_triangle_BDF_l658_658860

theorem max_area_triangle_BDF (D E A B C F : Type)
  (hD_midpoint : midpoint D A B)
  (h_ABC : area (triangle A B C) = 1)
  (hE_AC : E ∈ line_segment A C)
  (lambda₁ lambda₂ : ℝ)
  (hDF_DE : ∃ (DF DE : Type), ratio DF DE = lambda₁)
  (hAE_AC : ∃ (AE AC : Type), ratio AE AC = lambda₂)
  (h_sum_lambda : lambda₁ + lambda₂ = 1 / 2) :
  (∃ S, area (triangle B D F) = S) →
  max S = 1 / 32 := sorry

end max_area_triangle_BDF_l658_658860


namespace min_sum_ab_l658_658588

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l658_658588


namespace negation_proof_l658_658855

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := sorry

end negation_proof_l658_658855


namespace minimum_n_exists_choice_signs_l658_658362

def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem minimum_n_exists_choice_signs
    (n : ℕ) (h₁ : ∃ (signs : fin n → bool), ∑ i in finset.range n, (if signs i then 1 else -1) * (i + 1) ^ 2 = 0) : 
    n = 7 :=
begin
  sorry
end

end minimum_n_exists_choice_signs_l658_658362


namespace min_length_of_EC_l658_658256

open Real EuclideanGeometry

theorem min_length_of_EC {a b : ℝ} {ρ1 ρ2 ρ3 : ℝ} 
  (hP : ∀ x, (x^3 + a*x^2 + b*x + 1 = (x - ρ1)*(x - ρ2)*(x - ρ3)) 
  (h_rho_ordered : abs ρ1 < abs ρ2 ∧ abs ρ2 < abs ρ3)
  (h_roots : ρ1 * ρ2 * ρ3 = -1) :
  (∃ (P : ℝ → ℝ), ∀ x, P x = x^3 + a*x^2 + b*x + 1 ∧ 
    ∀ (x : ℝ), (P 0 = 1 ∧ P ρ1 = 0 ∧ P ρ2 = 0 ∧ P ρ3 = 0)) →
  ∃ E : ℝ × ℝ, 
    E = (0, -1 / ρ2) ∧ dist ⟨(0, -1/ρ2), (ρ2, 0)⟩ = Real.sqrt 2 := 
sorry

end min_length_of_EC_l658_658256


namespace min_triangles_needed_l658_658400

-- Problem Definition: Given K points inside a square, where K > 2,
-- Prove: The minimum number of triangles needed to divide the square so that each triangle contains at most one point is K + 1.

theorem min_triangles_needed (K : ℕ) (hK : K > 2) : 
  (∃ n : ℕ, n = K + 1 ∧ ∀ (triangles : fin n → (triangular_region : set (set (ℝ × ℝ)))), 
    (∀ i : fin n, ∃ (point_count : ℕ), ∃ P ⊆ triangular_region i, card P ≤ 1)) → n = K + 1 := 
sorry

end min_triangles_needed_l658_658400


namespace quadratic_binomial_plus_int_l658_658675

theorem quadratic_binomial_plus_int (y : ℝ) : y^2 + 14*y + 60 = (y + 7)^2 + 11 :=
by sorry

end quadratic_binomial_plus_int_l658_658675


namespace probability_of_Z_l658_658025

-- Defining the probabilities of landing on sections X and Y
def prob_X : ℚ := 3 / 8
def prob_Y : ℚ := 1 / 4

-- Given sum of probabilities of all sections equals 1, we need to prove the probability of landing on section Z.
theorem probability_of_Z : prob_X + prob_Y + ?m_1 = 1 → ?m_1 = 3 / 8 :=
by
  -- Leave the proof as sorry
  sorry

end probability_of_Z_l658_658025


namespace integer_solutions_of_system_l658_658657

theorem integer_solutions_of_system :
  {x : ℤ | - 2 * x + 7 < 10 ∧ (7 * x + 1) / 5 - 1 ≤ x} = {-1, 0, 1, 2} :=
by
  sorry

end integer_solutions_of_system_l658_658657


namespace log_piece_weight_l658_658247

variable (length_of_log : ℕ) (weight_per_foot : ℕ) (number_of_pieces : ℕ)
variable (original_length : length_of_log = 20)
variable (weight_per_linear_foot : weight_per_foot = 150)
variable (cuts_in_half : number_of_pieces = 2)

theorem log_piece_weight : (length_of_log / number_of_pieces) * weight_per_foot = 1500 := by
  have length_of_piece : length_of_log / number_of_pieces = 10 := by
    rw [original_length, cuts_in_half]
    norm_num
  rw [length_of_piece, weight_per_linear_foot]
  norm_num
  -- Proof complete

#print log_piece_weight

end log_piece_weight_l658_658247


namespace total_cost_of_suits_l658_658952

theorem total_cost_of_suits : 
    ∃ o t : ℕ, o = 300 ∧ t = 3 * o + 200 ∧ o + t = 1400 :=
by
  sorry

end total_cost_of_suits_l658_658952


namespace binary_to_base4_conversion_l658_658808

theorem binary_to_base4_conversion 
  (b : Nat) (h : b = 0b1011111010) : nat_to_base b 4 = "23322" := by
  sorry

end binary_to_base4_conversion_l658_658808


namespace no_polynomial_satisfies_conditions_l658_658521

noncomputable def polynomial_function_degree_2 (f : ℚ[X]) : Prop := degree f = 2

theorem no_polynomial_satisfies_conditions (f : ℚ[X]) (h : polynomial_function_degree_2 f)
    (h1 : ∀ x : ℚ, polynomial.eval (x^2) f = x^4)
    (h2 : ∀ x : ℚ, polynomial.eval (f.eval x) f = (x^2 + 1)^4) :
    false :=
sorry

end no_polynomial_satisfies_conditions_l658_658521


namespace number_of_zeros_l658_658969

def sequence_eqn (a : Fin 50 → Int) : Prop :=
  (∑ i, a i = 9) ∧ (∑ i, (a i + 1)^2 = 107)

theorem number_of_zeros (a : Fin 50 → Int) (h : ∀ i, a i ∈ {-1, 0, 1}) :
  sequence_eqn a → (∑ i, if a i = 0 then 1 else 0) = 11 := sorry

end number_of_zeros_l658_658969


namespace range_of_omega_l658_658540

theorem range_of_omega (ω : ℝ) (h₀ : ω > 0) :
  (∀ x y, -π/6 < x ∧ x < π/6 ∧ -π/6 < y ∧ y < π/6 ∧ x < y → sin (ω * x) < sin (ω * y)) →
  0 < ω ∧ ω ≤ 3 :=
by
  sorry

end range_of_omega_l658_658540


namespace max_value_of_x_plus_y_l658_658108

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ x + y + (1/x) + (1/y) = 5

-- Theorem statement
theorem max_value_of_x_plus_y (x y : ℝ) (h : conditions x y) : x + y ≤ 4 := 
sorry

end max_value_of_x_plus_y_l658_658108


namespace train_speed_proof_l658_658382

noncomputable def train_length : ℝ := 620
noncomputable def crossing_time : ℝ := 30.99752019838413
noncomputable def man_speed_kmh : ℝ := 8

noncomputable def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := train_length / crossing_time
noncomputable def train_speed_ms : ℝ := relative_speed + man_speed_ms
noncomputable def train_speed_kmh : ℝ := train_speed_ms * (3600 / 1000)

theorem train_speed_proof : abs (train_speed_kmh - 80) < 0.0001 := by
  sorry

end train_speed_proof_l658_658382


namespace tom_bought_new_books_l658_658698

def original_books : ℕ := 5
def sold_books : ℕ := 4
def current_books : ℕ := 39

def new_books (original_books sold_books current_books : ℕ) : ℕ :=
  current_books - (original_books - sold_books)

theorem tom_bought_new_books :
  new_books original_books sold_books current_books = 38 :=
by
  sorry

end tom_bought_new_books_l658_658698


namespace profit_percent_l658_658374

noncomputable def cost_price (C : ℝ) := C
noncomputable def selling_price_fraction (P : ℝ) := (2 / 3) * P
noncomputable def sell_price_loss (P C : ℝ) := 0.9 * C

theorem profit_percent (C P : ℝ) (h : (2 / 3) * P = 0.9 * C) :
  (P = 1.35 * C) → ((P - C) / C * 100 = 35) := by
suffices : P = 1.35 * C, from
  calc (P - C) / C * 100 = ((1.35 * C - C) / C) * 100 : by rw this
  ... = (0.35 * C / C) * 100 : by rw sub_div
  ... = 0.35 * 100 : by rw div_self (ne_of_lt (by norm_num : (0 : ℝ) < C))
  ... = 35 : by norm_num,
assumption

end profit_percent_l658_658374


namespace minimal_socks_to_ensure_pairs_l658_658003

/-- Prove the smallest number of socks needed to ensure at least 12 pairs of socks are picked. --/
theorem minimal_socks_to_ensure_pairs :
  ∀ (num_red num_green num_blue num_yellow num_purple total_pairs : ℕ),
  num_red = 120 → num_green = 100 → num_blue = 80 →
  num_yellow = 60 → num_purple = 40 → total_pairs = 12 →
  (∃ n, n ≥ 27 ∧
   (∀ picked_socks: list ℕ, picked_socks.length = n →
    (∃ (pairs_of_red pairs_of_green pairs_of_blue pairs_of_yellow pairs_of_purple : ℕ),
     pairs_of_red + pairs_of_green + pairs_of_blue + pairs_of_yellow + pairs_of_purple ≥ total_pairs))) :=
  sorry

end minimal_socks_to_ensure_pairs_l658_658003


namespace find_range_of_k_compare_y_values_l658_658146

theorem find_range_of_k (k : ℝ) (x : ℝ) (y : ℝ) (hx : x ≠ 0) (hy : y = (k - 2) / x)
  (h_quadrant : ∀ x, x < 0 → (k - 2) < 0) : k < 2 :=
by
  intro
  have h := h_quadrant x hx.lt
  linarith
  sorry

theorem compare_y_values (k : ℝ) (y1 y2 : ℝ) (hy1 : y1 = (k - 2) / (-4)) (hy2 : y2 = (k - 2) / (-1))
  (hk : k < 2) : y1 < y2 :=
by
  linarith
  sorry

end find_range_of_k_compare_y_values_l658_658146


namespace simplify_function_l658_658467

-- Problem statement
theorem simplify_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt ((1 - x) / (1 + x))) (α : ℝ)
  (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  f (Real.cos α) + f (-Real.cos α) = 2 * Real.csc α := 
sorry

end simplify_function_l658_658467


namespace eq_y_eq_x_l658_658368

theorem eq_y_eq_x : ∀ x : ℝ, (∛(x^3)) = x :=
by
  sorry

end eq_y_eq_x_l658_658368


namespace area_of_figure_outside_three_circles_touching_each_other_l658_658694

theorem area_of_figure_outside_three_circles_touching_each_other (r : ℝ) :
  let S := (r^2 * (2 * Real.sqrt 3 - Real.pi)) / 2
  in true :=
by sorry

end area_of_figure_outside_three_circles_touching_each_other_l658_658694


namespace possible_values_for_b2_eq_359_l658_658017

-- Define the sequence based on given conditions and statement of the problem
def seq (a b : ℕ) : ℕ → ℕ
| 0     := a
| 1     := b
| (n+2) := abs (seq n.succ - seq n)

-- Define the conditions based on the problem statement
def problem_conditions (b2 : ℕ) : Prop :=
  b2 < 950 ∧
  (∃ b2010, seq 950 b2 2010 = 0) ∧
  (∃ b1008, seq 950 b2 1008 = 1)

-- Statement that provides the number of different possible values for b2
theorem possible_values_for_b2_eq_359 :
  ∃ (S : Set ℕ), (∀ b2, b2 ∈ S ↔ problem_conditions b2) ∧ S.card = 359 :=
sorry

end possible_values_for_b2_eq_359_l658_658017


namespace ellipse_and_line_segment_l658_658877

noncomputable def ellipse_equation (a b : ℝ) (C : ℝ) : Prop :=
  √(a^2 - b^2) = 1 / 2 * a ∧ (0, √3) ∈ { (x, y) | x^2 / a^2 + y^2 / b^2 = C }

noncomputable def line_segment_length (a b : ℝ) (k : ℝ) : ℝ :=
let l := { (x, y) | y = k * x - 1 } in
  let ell := { (x, y) | x^2 / a^2 + y^2 / b^2 = 1 } in
  let intersections := l ∩ ell in
  let (x1, y1) := some (intersections.some) in
  let (x2, y2) := some ((intersections.diff { (x1, y1) }).some) in
  dist (x1, y1) (x2, y2)

theorem ellipse_and_line_segment :
  ∃ a b : ℝ, a = 2 ∧ b = √3 ∧
  ellipse_equation a b 1 ∧
  line_segment_length 2 √3 1 = 24 / 7 :=
by
  use [2, √3]
  split
  { exact rfl }
  split
  { exact rfl }
  split
  {
    simp [ellipse_equation]
    split
    {
      norm_num,
      simp,
      sorry
    }
    {
      sorry
    }
  }
  {
    simp [line_segment_length],
    sorry
  }

end ellipse_and_line_segment_l658_658877


namespace range_of_m_l658_658145

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (2 <= x ∧ x < 3) → (2 * x^2 - 9 * x + m < 0)) → m ≤ 9 := by
  -- Required conditions logic
  assume h : ∀ x : ℝ, (2 <= x ∧ x < 3) → (2 * x^2 - 9 * x + m < 0)
  have h_m_bound : m < -2*3^2 + 9*3 := sorry
  -- Use the given interval to derive the final range for m
  exact sorry

end range_of_m_l658_658145


namespace pascal_fifth_element_row_20_l658_658708

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l658_658708


namespace total_raining_time_correct_l658_658240

-- Define individual durations based on given conditions
def duration_day1 : ℕ := 10        -- 17:00 - 07:00 = 10 hours
def duration_day2 : ℕ := duration_day1 + 2    -- Second day: 10 hours + 2 hours = 12 hours
def duration_day3 : ℕ := duration_day2 * 2    -- Third day: 12 hours * 2 = 24 hours

-- Define the total raining time over three days
def total_raining_time : ℕ := duration_day1 + duration_day2 + duration_day3

-- Formally state the theorem to prove the total rain time is 46 hours
theorem total_raining_time_correct : total_raining_time = 46 := by
  sorry

end total_raining_time_correct_l658_658240


namespace amy_soup_total_cost_l658_658419

theorem amy_soup_total_cost :
  let chicken_soup_cost := 6 * 1.50
  let tomato_soup_cost := 3 * 1.25
  let vegetable_soup_cost := 4 * 1.75
  let clam_chowder_cost := 2 * 2.00
  let french_onion_soup_cost := 1 * 1.80
  let minestrone_soup_cost := 5 * 1.70
  let total_cost := chicken_soup_cost + tomato_soup_cost + vegetable_soup_cost + clam_chowder_cost + french_onion_soup_cost + minestrone_soup_cost
  in total_cost = 34.05 :=
by
  sorry

end amy_soup_total_cost_l658_658419


namespace total_area_of_pyramid_faces_l658_658730

theorem total_area_of_pyramid_faces (base_edge lateral_edge : ℝ) (h : base_edge = 8) (k : lateral_edge = 5) : 
  4 * (1 / 2 * base_edge * 3) = 48 :=
by
  -- Base edge of the pyramid
  let b := base_edge
  -- Lateral edge of the pyramid
  let l := lateral_edge
  -- Half of the base
  let half_b := 4
  -- Height of the triangular face using Pythagorean theorem
  let h := 3
  -- Total area of four triangular faces
  have triangular_face_area : 1 / 2 * base_edge * h = 12 := sorry
  have total_area_of_faces : 4 * (1 / 2 * base_edge * h) = 48 := sorry
  exact total_area_of_faces

end total_area_of_pyramid_faces_l658_658730


namespace fliers_remaining_l658_658740

theorem fliers_remaining (initial_fliers : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ) :
  initial_fliers = 3000 → morning_fraction = 1/5 → afternoon_fraction = 1/4 →
  (initial_fliers - initial_fliers * morning_fraction - (initial_fliers - initial_fliers * morning_fraction) * afternoon_fraction) = 1800 :=
by { intros, sorry }

end fliers_remaining_l658_658740


namespace pure_ghee_percentage_l658_658937

theorem pure_ghee_percentage (Q : ℝ) (vanaspati_percentage : ℝ:= 0.40) (additional_pure_ghee : ℝ := 10) (new_vanaspati_percentage : ℝ := 0.20) (original_quantity : ℝ := 10) :
  (Q = original_quantity) ∧ (vanaspati_percentage = 0.40) ∧ (additional_pure_ghee = 10) ∧ (new_vanaspati_percentage = 0.20) →
  (100 - (vanaspati_percentage * 100)) = 60 :=
by
  sorry

end pure_ghee_percentage_l658_658937


namespace train_crossing_time_l658_658233

-- Define given conditions as constants
constant train_length : ℝ := 40  -- Length of the train in meters
constant speed_kmh : ℝ := 144     -- Speed of the train in km/hr
constant conversion_factor : ℝ := 5 / 18  -- Conversion factor from km/hr to m/s

-- Define the speed of the train in m/s
def speed_ms : ℝ := speed_kmh * conversion_factor

-- Define the time it takes for the train to cross the electric pole
def time_to_cross (d : ℝ) (v : ℝ) : ℝ := d / v

-- Theorem to prove the time it takes to cross the electric pole
theorem train_crossing_time : time_to_cross train_length speed_ms = 1 :=
by
  -- Proof would go here
  sorry

end train_crossing_time_l658_658233


namespace exists_subset_with_property_l658_658785

theorem exists_subset_with_property (r : ℕ) (A : fin r → set ℕ)
  (hdisjoint : ∀ i j : fin r, i ≠ j → disjoint (A i) (A j))
  (hcover : ⋃ i, A i = set.univ) :
  ∃ i : fin r, ∃ m : ℕ, ∀ k : ℕ,
    ∃ α : fin k → ℕ, ∀ j : fin (k - 1), α j.succ - α j ≤ m :=
sorry

end exists_subset_with_property_l658_658785


namespace cost_of_ticket_when_Matty_was_born_l658_658364

theorem cost_of_ticket_when_Matty_was_born 
    (cost : ℕ → ℕ) 
    (h_halved : ∀ t : ℕ, cost (t + 10) = cost t / 2) 
    (h_age_30 : cost 30 = 125000) : 
    cost 0 = 1000000 := 
by 
  sorry

end cost_of_ticket_when_Matty_was_born_l658_658364


namespace ratio_of_third_to_first_l658_658576

-- Define the conditions
def cost_first_floor : ℕ := 15
def cost_second_floor : ℕ := 20
def num_rooms_per_floor : ℕ := 3
def total_earnings_per_month : ℕ := 165

-- Define the total earnings from the first and second floors
def earnings_first_floor := num_rooms_per_floor * cost_first_floor
def earnings_second_floor := num_rooms_per_floor * cost_second_floor

-- Define the earnings from the third floor
def earnings_third_floor := total_earnings_per_month - (earnings_first_floor + earnings_second_floor)

-- Define the cost per room on the third floor
def cost_third_floor := earnings_third_floor / num_rooms_per_floor

-- Define the required ratio of the cost of the rooms on the third floor to the cost of the rooms on the first floor
def ratio_third_to_first := cost_third_floor.to_rat / cost_first_floor.to_rat

-- The proof statement
theorem ratio_of_third_to_first : ratio_third_to_first = (4 : ℚ) / 3 :=
by 
  -- Basic formal proof structure
  sorry

end ratio_of_third_to_first_l658_658576


namespace triangle_area_is_correct_l658_658229

noncomputable def area_triangle : ℝ :=
  let AB := 1; -- Without loss of generality, let AB = 1
  let BC := 1; -- Because AB = BC
  let BE := 10; 
  let cot x := 1 / Real.tan x in
  let α := Real.pi / 4 in -- Given that α = π/4, which is the solution step conclusion
  let β := Real.atan (1 / 2) in -- Assume an arbitrary angle β for solution context
  let BD := BE / Real.sqrt 2 in -- Given BD = BE / sqrt(2)
  let DC := BE / (3 * Real.sqrt 2) in -- Given DC = BE / 3sqrt(2)
  let area := (BD * DC) / 2 in -- Calculate the area of triangle ABC using 1/2 * base * height
  Real.ofInt (area)

theorem triangle_area_is_correct :
  area_triangle = 50 / 3 := by
  sorry

end triangle_area_is_correct_l658_658229


namespace problem_1_problem_2_problem_3_l658_658902

-- Definition of vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (k : ℝ) : ℝ × ℝ := (-3, k)

-- Proof problem 1: Proving that if vec_a is parallel to vec_b, then |vec_b| = 3√5
theorem problem_1 (k : ℝ) (H1 : vec_a = (1, 2)) (H2 : vec_b k = (-3, k)) (H_parallel : vec_a.1 / vec_b k.1 = vec_a.2 / vec_b k.2) :
  |vec_b (-6)| = 3 * Real.sqrt 5 :=
by
  sorry

-- Proof problem 2: Proving that if vec_a is perpendicular to (vec_a + 2 * vec_b), then k = 1/4
theorem problem_2 (k : ℝ) (H1 : vec_a = (1, 2)) (H2 : vec_b k = (-3, k)) (H_perp : (vec_a.1 * (vec_a.1 + 2 * vec_b k.1)) + (vec_a.2 * (vec_a.2 + 2 * vec_b k.2)) = 0) :
  k = 1 / 4 :=
by
  sorry

-- Proof problem 3: Proving that if the angle between vec_a and vec_b is obtuse, then k < 3/2 and k ≠ -6
theorem problem_3 (k : ℝ) (H1 : vec_a = (1, 2)) (H2 : vec_b k = (-3, k)) (H_obtuse : vec_a.1 * vec_b k.1 + vec_a.2 * vec_b k.2 < 0) : 
  k < Real.sqrt (3 / 2) ∧ k ≠ -6 :=
by
  sorry

end problem_1_problem_2_problem_3_l658_658902


namespace min_n_value_arithmetic_sequence_l658_658851

theorem min_n_value_arithmetic_sequence :
  ∀ (a : ℕ → ℝ) (d : ℝ),
    (∀ n m, a n = a m + d * (n - m)) →
    0 < d ∧ d < 1 →
    (sin(a 3) ^ 2 - sin(a 7) ^ 2) / sin(a 3 + a 7) = -1 →
    -5 * (Real.pi / 4) < a 1 ∧ a 1 < -9 * (Real.pi / 8) →
    ∃ n : ℕ, (∀ m < n, a m ≤ 0) ∧ n = 10 :=
begin
  intros,
  sorry
end

end min_n_value_arithmetic_sequence_l658_658851


namespace vectors_orthogonal_dot_product_l658_658445

theorem vectors_orthogonal_dot_product (y : ℤ) :
  (3 * -2) + (4 * y) + (-1 * 5) = 0 → y = 11 / 4 :=
by
  sorry

end vectors_orthogonal_dot_product_l658_658445


namespace proof_problem_l658_658650

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l658_658650


namespace factorial_cannot_be_all_ones_l658_658621

theorem factorial_cannot_be_all_ones (N : ℕ) (hN : N ≥ 3) : ¬ (∃ k : ℕ, (N.factorial : ℕ) = k * 10^(digits_count k) - 1)
where digits_count (n : ℕ) : ℕ :=
  if n = 0 then 1 else ⌊log 10 (n + 1)⌋₊

end factorial_cannot_be_all_ones_l658_658621


namespace trajectory_of_center_is_two_rays_or_circle_or_ellipse_l658_658114

noncomputable def trajectory_of_center (P : Point) (C : Circle) (O : Circle) : Set Point :=
sorry

theorem trajectory_of_center_is_two_rays_or_circle_or_ellipse 
  (P : Point) (O : Circle) (C : Circle):
  (P ∈ interior O ∨ P ∈ O) →
  (C.passes_through P ∧ C.is_tangent_to O) →
  (trajectory_of_center P C O = two_rays ∨ trajectory_of_center P C O = circle ∨ trajectory_of_center P C O = ellipse) :=
sorry

end trajectory_of_center_is_two_rays_or_circle_or_ellipse_l658_658114


namespace find_a_l658_658129

noncomputable def f (a x : ℝ) : ℝ := a * x * (x - 2)^2

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∃ x : ℝ, f a x = 32) :
  a = 27 :=
sorry

end find_a_l658_658129


namespace new_weight_is_77_l658_658302

theorem new_weight_is_77 (weight_increase_per_person : ℝ) (number_of_persons : ℕ) (old_weight : ℝ) 
  (total_weight_increase : ℝ) (new_weight : ℝ) 
  (h1 : weight_increase_per_person = 1.5)
  (h2 : number_of_persons = 8)
  (h3 : old_weight = 65)
  (h4 : total_weight_increase = number_of_persons * weight_increase_per_person)
  (h5 : new_weight = old_weight + total_weight_increase) :
  new_weight = 77 :=
sorry

end new_weight_is_77_l658_658302


namespace simplify_and_rationalize_correct_l658_658646

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l658_658646


namespace probability_match_ends_in_two_games_l658_658544

-- Definitions based on conditions in the problem
def P_A_wins_game : ℝ := 0.6
def P_B_wins_game : ℝ := 0.4
def A_wins (i : ℕ) : Prop := true  -- Placeholder, no need for actual game number
def B_wins (j : ℕ) : Prop := true  -- Placeholder, no need for actual game number

/-- The main theorem stating the probability that the match ends after two more games is 0.52. -/
theorem probability_match_ends_in_two_games : 
  (P((A_wins 3 ∧ A_wins 4) ∨ (B_wins 3 ∧ B_wins 4))) = 0.52 :=
by
  let P_A3_A4 := P_A_wins_game * P_A_wins_game
  let P_B3_B4 := P_B_wins_game * P_B_wins_game
  have P_A_or_B := P_A3_A4 + P_B3_B4
  have h_P : P_A_or_B = 0.52 := by norm_num
  exact h_P

end probability_match_ends_in_two_games_l658_658544


namespace minimum_k_l658_658134

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem minimum_k (e : ℝ) (h : e = Real.exp 1) :
  (∀ m : ℝ, m ∈ Set.Icc (-2 : ℝ) 4 → f (-2 * m^2 + 2 * m - 1) + f (8 * m + e^4) > 0) → 4 = 4 := 
sorry

end minimum_k_l658_658134


namespace exists_point_M_on_circle_l658_658465

variables {n : ℕ} (A : fin n → ℝ × ℝ) (radius : ℝ := 1)

theorem exists_point_M_on_circle
  (hA : ∀ i, (A i).1^2 + (A i).2^2 ≠ 0) :
  ∃ M : ℝ × ℝ, (M.1^2 + M.2^2 = radius^2) ∧ 
  (∑ k in finset.fin_range n, real.dist M (A k) ≥ n) :=
sorry

end exists_point_M_on_circle_l658_658465


namespace problem_maximum_value_problem_smallest_period_problem_sum_x_i_l658_658498

def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sqrt 3 * sin x * cos x

theorem problem_maximum_value :
  ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) :=
sorry

theorem problem_smallest_period :
  ∃ T, ∀ x, f (x + T) = f x ∧ T > 0 :=
sorry

theorem problem_sum_x_i :
  let M := 2 in
  let T := π in
  ∃ x : ℕ → ℝ, (∀ i, 1 ≤ i ∧ i ≤ 10 → f (x i) = M ∧ x i < 10 * π) ∧
  ∑ i in (finset.range 10).image (λ n, n + 1), x i = (140/3) * π :=
sorry

end problem_maximum_value_problem_smallest_period_problem_sum_x_i_l658_658498


namespace log_graph_transformation_l658_658697

theorem log_graph_transformation :
  ∀ x, (y = log x → y = log (x-1) - 2) ↔ (shifting_right 1 unit ∧ shifting_down 2 units) :=
by
  intro x
  sorry

end log_graph_transformation_l658_658697


namespace expression_value_l658_658107

theorem expression_value (x y : ℝ) (h1 : x + y = 0.2) (h2 : x + 3y = 1) : x^2 + 4 * x * y + 4 * y^2 = 0.36 :=
by
  sorry

end expression_value_l658_658107


namespace theo_cookies_per_sitting_l658_658691

-- Definitions from conditions
def sittings_per_day : ℕ := 3
def days_per_month : ℕ := 20
def cookies_in_3_months : ℕ := 2340

-- Calculation based on conditions
def sittings_per_month : ℕ := sittings_per_day * days_per_month
def sittings_in_3_months : ℕ := sittings_per_month * 3

-- Target statement
theorem theo_cookies_per_sitting :
  cookies_in_3_months / sittings_in_3_months = 13 :=
sorry

end theo_cookies_per_sitting_l658_658691


namespace perfect_square_free_sets_count_l658_658963

-- Definition of the sets T_i
def T (i : ℕ) : set ℕ := {n | 50 * i ≤ n ∧ n < 50 * (i + 1)}

-- Predicate to check if a set contains a perfect square
def contains_perfect_square (s : set ℕ) : Prop :=
  ∃ n, n ∈ s ∧ ∃ k, n = k * k

-- The main statement to prove
theorem perfect_square_free_sets_count :
  (finset.range 2000).filter (λ i, ¬ contains_perfect_square (T i)).card = 1858 :=
sorry

end perfect_square_free_sets_count_l658_658963


namespace mona_unique_players_l658_658615

theorem mona_unique_players (groups : ℕ) (players_per_group : ℕ) (repeated1 : ℕ) (repeated2 : ℕ) :
  (groups = 9) → (players_per_group = 4) → (repeated1 = 2) → (repeated2 = 1) →
  (groups * players_per_group - (repeated1 + repeated2) = 33) :=
begin
  intros h_groups h_players_per_group h_repeated1 h_repeated2,
  rw [h_groups, h_players_per_group, h_repeated1, h_repeated2],
  norm_num,
end

end mona_unique_players_l658_658615


namespace intersection_A_B_l658_658479

-- Definitions of sets A and B
def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

-- Prove that the intersection of sets A and B is {2, 3, 5}
theorem intersection_A_B :
  A ∩ B = {2, 3, 5} :=
sorry

end intersection_A_B_l658_658479


namespace fred_bought_books_l658_658098

theorem fred_bought_books (initial_money : ℕ) (remaining_money : ℕ) (book_cost : ℕ)
  (h1 : initial_money = 236)
  (h2 : remaining_money = 14)
  (h3 : book_cost = 37) :
  (initial_money - remaining_money) / book_cost = 6 :=
by {
  sorry
}

end fred_bought_books_l658_658098


namespace transfers_l658_658928

variable (x : ℕ)
variable (gA gB gC : ℕ)

noncomputable def girls_in_A := x + 4
noncomputable def girls_in_B := x
noncomputable def girls_in_C := x - 1

variable (trans_A_to_B : ℕ)
variable (trans_B_to_C : ℕ)
variable (trans_C_to_A : ℕ)

axiom C_to_A_girls : trans_C_to_A = 2
axiom equal_girls : gA = x + 1 ∧ gB = x + 1 ∧ gC = x + 1

theorem transfers (hA : gA = girls_in_A - trans_A_to_B + trans_C_to_A)
                  (hB : gB = girls_in_B - trans_B_to_C + trans_A_to_B)
                  (hC : gC = girls_in_C - trans_C_to_A + trans_B_to_C) :
  trans_A_to_B = 5 ∧ trans_B_to_C = 4 :=
by
  sorry

end transfers_l658_658928


namespace determinant_scaled_matrix_l658_658102

example (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 9 * |Matrix![[x, y], [z, w]]| := by
  sorry

theorem determinant_scaled_matrix (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 63 := by
  rw [Matrix.det_smul, h]
  norm_num
  sorry

end determinant_scaled_matrix_l658_658102


namespace sum_of_solutions_l658_658093

theorem sum_of_solutions :
  (∑ x in (set_of (λ x : ℝ => ((1 / Real.sin x) + (1 / Real.cos x) = 4) ∧ (0 ≤ x) ∧ (x ≤ 2 * Real.pi))), x) = Real.pi :=
by
  sorry

end sum_of_solutions_l658_658093


namespace second_smallest_dimension_of_crate_l658_658001

-- Define the dimensions of the crate. Assume d is the unknown dimension.
variables (d : ℝ)

-- Define the radius and diameter of the pillar.
def pillar_radius : ℝ := 6
def pillar_diameter : ℝ := 2 * pillar_radius

-- Given the crate dimensions
def crate_dimensions : set ℝ := {6, d, 12}

-- The condition that the pillar must fit upright in the crate.
def fits_upright_in_crate : Prop := 
  ∃ (height width length : ℝ), 
    height ∈ crate_dimensions ∧
    width ∈ crate_dimensions ∧
    length ∈ crate_dimensions ∧ 
    height ≥ pillar_diameter ∧ width ≠ height ∧ width ≠ length ∧ length ≠ height

-- The problem statement to be proved
theorem second_smallest_dimension_of_crate : fits_upright_in_crate d → d = 12 := 
by sorry

end second_smallest_dimension_of_crate_l658_658001


namespace miles_driven_on_Monday_l658_658409

def miles_Tuesday : ℕ := 18
def miles_Wednesday : ℕ := 21
def avg_miles_per_day : ℕ := 17

theorem miles_driven_on_Monday (miles_Monday : ℕ) :
  (miles_Monday + miles_Tuesday + miles_Wednesday) / 3 = avg_miles_per_day →
  miles_Monday = 12 :=
by
  intro h
  sorry

end miles_driven_on_Monday_l658_658409


namespace range_H_l658_658351

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_H : set.range H = {-4, 4} := 
by
  sorry

end range_H_l658_658351


namespace largest_n_for_factorization_l658_658823

theorem largest_n_for_factorization :
  ∃ (n : ℤ), (∀ (A B : ℤ), AB = 96 → n = 4 * B + A) ∧ (n = 385) := by
  sorry

end largest_n_for_factorization_l658_658823


namespace fifth_element_row_20_pascal_triangle_l658_658720

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l658_658720


namespace binom_14_11_l658_658801

open Nat

theorem binom_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end binom_14_11_l658_658801


namespace even_function_derivative_is_odd_l658_658989

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Definitions based on the conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def derivative (f : ℝ → ℝ) := ∀ x : ℝ, g x = (f' x)

theorem even_function_derivative_is_odd {f g : ℝ → ℝ}
  (h : is_even f)
  (h_deriv : ∀ x : ℝ, g x = (f' x)) :
  is_odd g :=
sorry

end even_function_derivative_is_odd_l658_658989


namespace remainder_problem_l658_658180

theorem remainder_problem (d r : ℤ) (h1 : 1237 % d = r)
    (h2 : 1694 % d = r) (h3 : 2791 % d = r) (hd : d > 1) :
    d - r = 134 := sorry

end remainder_problem_l658_658180


namespace angle_C_value_l658_658923

theorem angle_C_value (a b c C : ℝ) (h : a^2 + b^2 - c^2 + ab = 0) (h1 : 0 < C) (h2 : C < π) :
  C = (2 / 3) * π :=
begin
  sorry
end

end angle_C_value_l658_658923


namespace sum_of_roots_l658_658729

-- States that the sum of the values of x that satisfy the given quadratic equation is 7
theorem sum_of_roots (x : ℝ) :
  (x^2 - 7 * x + 12 = 4) → (∃ a b : ℝ, x^2 - 7 * x + 8 = 0 ∧ a + b = 7) :=
by
  sorry

end sum_of_roots_l658_658729


namespace range_of_H_l658_658356

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_of_H_l658_658356


namespace average_velocity_correct_l658_658395

def particle_motion_law (t : ℝ) : ℝ :=
  t^2 + 3

def average_velocity (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem average_velocity_correct (Δx : ℝ) :
  average_velocity particle_motion_law 3 (3 + Δx) = 6 + Δx :=
by
  sorry

end average_velocity_correct_l658_658395


namespace ellipse_eq_of_hyperbola_l658_658504

-- Define parameters and equations
def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = -1

-- Main statement to be proved
theorem ellipse_eq_of_hyperbola :
  (∃ x y : ℝ, hyperbola_eq x y) →
  ∃ (a b : ℝ), 
    a = 4 ∧ 
    b = √(a^2 - (2*√3)^2) ∧
    (x^2 / 4) + (y^2 / 16) = 1 :=
by
  sorry

end ellipse_eq_of_hyperbola_l658_658504


namespace compute_fraction_l658_658802

theorem compute_fraction : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := 
by
  sorry

end compute_fraction_l658_658802


namespace number_of_pairs_eq_2_l658_658177

theorem number_of_pairs_eq_2 : 
  { (x, y) : ℕ × ℕ // x > 0 ∧ y > 0 ∧ x^2 - y^2 = 51 }.to_finset.card = 2 :=
by
  sorry

end number_of_pairs_eq_2_l658_658177


namespace percentage_born_in_july_l658_658676

def total_scientists : ℕ := 150
def scientists_born_in_july : ℕ := 15

theorem percentage_born_in_july : (scientists_born_in_july * 100 / total_scientists) = 10 := by
  sorry

end percentage_born_in_july_l658_658676


namespace yacht_AC_squared_range_l658_658777

-- Define angle θ in radians for easier computation in Lean.
def degrees_to_radians (d : ℝ) : ℝ := d * (Float.pi / 180)

theorem yacht_AC_squared_range :
  let AB := 15
  let BC := 25
  let θ_min := degrees_to_radians 30
  let θ_max := degrees_to_radians 75
  let AC_squared θ := AB^2 + BC^2 - 2 * AB * BC * Real.cos θ in
  200 ≤ AC_squared θ_min ∧ AC_squared θ_max ≤ 656 :=
by
  sorry

end yacht_AC_squared_range_l658_658777


namespace curve_is_two_intersecting_lines_l658_658059

def equation := ∀ (x y : ℝ), 2 * x^2 - y^2 - 4 * x - 4 * y - 2 = 0

theorem curve_is_two_intersecting_lines : 
  (∀ (x y : ℝ), 2 * x^2 - y^2 - 4 * x - 4 * y - 2 = 0) → 
  (∀ (x y : ℝ), y = sqrt 2 * x - sqrt 2 - 2 ∨ y = -sqrt 2 * x + sqrt 2 - 2) :=
sorry

end curve_is_two_intersecting_lines_l658_658059


namespace range_of_H_l658_658354

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_of_H_l658_658354


namespace probability_sum_not_less_than_15_l658_658385

open ProbTheory

noncomputable def probability_not_less_than_15 : ℚ :=
let prob_individual : ℚ := 1 / 8 in
let prob_pair_7_8    := prob_individual * prob_individual in
let prob_pair_8_8    := prob_individual * prob_individual in
prob_pair_7_8 + prob_pair_7_8 + prob_pair_8_8

theorem probability_sum_not_less_than_15 :
  probability_not_less_than_15 = 3 / 64 :=
by
  sorry

end probability_sum_not_less_than_15_l658_658385


namespace new_average_score_l658_658663

theorem new_average_score (avg_score : ℝ) (num_students : ℕ) (dropped_score : ℝ) (new_num_students : ℕ) :
  num_students = 16 →
  avg_score = 61.5 →
  dropped_score = 24 →
  new_num_students = num_students - 1 →
  (avg_score * num_students - dropped_score) / new_num_students = 64 :=
by
  sorry

end new_average_score_l658_658663


namespace solve_math_problem_l658_658478

noncomputable def math_problem (m n : ℝ) : Prop :=
  (m * exp m / (4 * n^2) = (log n + log 2) / exp m) ∧
  (exp (2 * m) = 1 / m) →
  ((n = exp m / 2) ∧
  (m * n^2 ≠ 1) ∧
  (m + n < 7 / 5) ∧
  (1 < 2 * n - m^2 ∧ 2 * n - m^2 < 3 / 2))

theorem solve_math_problem (m n : ℝ) : math_problem m n :=
sorry

end solve_math_problem_l658_658478


namespace max_value_of_x2_y2_on_circle_l658_658494

theorem max_value_of_x2_y2_on_circle : 
  let C := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 4 * p.1 - 2 * p.2 - 4 = 0}
  in ∀ p ∈ C, (p.1)^2 + (p.2)^2 ≤ 14 + 6 * real.sqrt 5 :=
sorry

end max_value_of_x2_y2_on_circle_l658_658494


namespace henry_distance_proof_l658_658517

noncomputable def henry_distance_from_start (north_meters south_meters meters_to_feet east_feet : ℝ) : ℝ :=
  let north_feet := north_meters * meters_to_feet
  let south_total_feet := (south_meters * meters_to_feet) + 48
  let net_south_feet := south_total_feet - north_feet
  real.sqrt ((east_feet ^ 2) + (net_south_feet ^ 2))

theorem henry_distance_proof :
  henry_distance_from_start 15 15 3.28084 40 = 62.48 :=
by
  -- here should be the proof block to be filled out
  sorry

end henry_distance_proof_l658_658517


namespace simplify_and_rationalize_l658_658655

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l658_658655


namespace ball_count_l658_658756

theorem ball_count (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 8)
  (h4 : red = 5)
  (h5 : purple = 7) 
  (h6 : (1 - 0.8 : ℝ) = 0.2) :
  let T := white + green + yellow + red + purple in
  (12 : ℝ) / T = 0.2 → T = 60 :=
by
  intro T_def
  have h7 : 12 = 0.2 * T := by sorry
  exact sorry

end ball_count_l658_658756


namespace sqrt_problem_l658_658858

theorem sqrt_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : a = (3 * m - 1) ^ 2) 
  (h2 : a = (-2 * m - 2) ^ 2) : 
  a = 64 ∨ a = 64 / 25 := 
sorry

end sqrt_problem_l658_658858


namespace laundromat_cost_is_35_32_l658_658791

def service_fee : ℝ := 3
def cost_first_hour_large_load : ℝ := 10
def cost_per_additional_hour_large_load : ℝ := 15
def membership_discount : ℝ := 0.10
def total_time_hours : ℝ := 2.75
def discount_amount (total: ℝ) : ℝ := total * membership_discount

theorem laundromat_cost_is_35_32 : 
  let additional_time := total_time_hours - 1
  let additional_cost := additional_time * cost_per_additional_hour_large_load
  let total_laundry_cost := cost_first_hour_large_load + additional_cost
  let total_cost := total_laundry_cost + service_fee
  let discounted_cost := total_cost - discount_amount(total_cost)
  discounted_cost = 35.32 :=
by
  sorry

end laundromat_cost_is_35_32_l658_658791


namespace polynomial_value_at_3_l658_658340

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem polynomial_value_at_3 : f 3 = 1209.4 := 
by
  sorry

end polynomial_value_at_3_l658_658340


namespace sufficient_but_not_necessary_l658_658122

-- Definitions for lines and planes
def line : Type := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

-- Predicate for perpendicularity of a line to a plane
def perp_to_plane (l : line) (α : plane) : Prop := sorry

-- Predicate for parallelism of two planes
def parallel_planes (α β : plane) : Prop := sorry

-- Predicate for perpendicularity of two lines
def perp_lines (l m : line) : Prop := sorry

-- Predicate for a line being parallel to a plane
def parallel_to_plane (m : line) (β : plane) : Prop := sorry

-- Given conditions
variable (l : line)
variable (m : line)
variable (alpha : plane)
variable (beta : plane)
variable (H1 : perp_to_plane l alpha) -- l ⊥ α
variable (H2 : parallel_to_plane m beta) -- m ∥ β

-- Theorem statement
theorem sufficient_but_not_necessary :
  (parallel_planes alpha beta → perp_lines l m) ∧ ¬(perp_lines l m → parallel_planes alpha beta) :=
sorry

end sufficient_but_not_necessary_l658_658122


namespace equilateral_complex_sum_l658_658430

noncomputable theory

open Complex

theorem equilateral_complex_sum {p q r : ℂ} (h1 : abs (p + q + r) = 48)
  (h2 : abs (p - q) = 24) (h3 : abs (q - r) = 24) (h4 : abs (r - p) = 24) :
  abs (p*q + p*r + q*r) = 768 :=
by
  -- Proof is omitted as per the instructions
  sorry

end equilateral_complex_sum_l658_658430


namespace minLengthPolygonalChainIsNonSelfIntersecting_l658_658468

variable (V : Type*) [MetricSpace V]

def PolygonalChain (vertices : List V) : Prop :=
  ∀ (otherVertices : List V), otherVertices ≠ vertices → length otherVertices > length vertices

theorem minLengthPolygonalChainIsNonSelfIntersecting
  {vertices : List V}
  (h : PolygonalChain vertices) :
  ¬ ∃ (intersectingSegments : List (V × V)), vertices ∈ intersectingSegments → False :=
by
  sorry

end minLengthPolygonalChainIsNonSelfIntersecting_l658_658468


namespace license_plate_combinations_l658_658792

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem license_plate_combinations : 
  let letters := 26
  let digits := 10
  let repeated_letter_choices := letters
  let distinct_letter_choices := letters - 1
  let repeated_letter_positions := choose 3 2
  let distinct_digit_combinations := choose digits 3
  in repeated_letter_choices * distinct_letter_choices * repeated_letter_positions * distinct_digit_combinations = 23400 := by 
  sorry

end license_plate_combinations_l658_658792


namespace nat_numbers_count_l658_658171

-- Defining the property that a natural number must satisfy
def satisfies_property (n : ℕ) : Prop :=
  let digits := List.map (fun c => Char.toNat c - Char.toNat '0') (String.toList (toString n)) in
  let product := List.foldr (*) 1 digits in
  product * List.length digits = 2014

-- The main theorem
theorem nat_numbers_count : ∃ n : ℕ, satisfies_property n → n = 1008 := sorry

end nat_numbers_count_l658_658171


namespace range_of_a_l658_658491

theorem range_of_a (a : ℝ) :
  let x := 3 * a - 9
  let y := a + 2
  (sin (2 * α) ≤ 0) → (sin α > 0) → x ≤ 0 → y > 0 → (-2 < a ∧ a ≤ 3) :=
by
  intros _ _ _ _
  sorry

end range_of_a_l658_658491


namespace no_integer_area_of_prime_sided_triangle_l658_658307

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_integer_area_of_prime_sided_triangle (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) : 
  ¬ ∃ (S : ℤ), area a b c = S :=
sorry

end no_integer_area_of_prime_sided_triangle_l658_658307


namespace sin_of_angle_through_point_l658_658195

/-- 
If the terminal side of angle α passes through the point (2*sin(60°), -2*cos(60°)),
then sin(α) = -1/2.
-/
theorem sin_of_angle_through_point (α : ℝ) :
  let p := (2 * Real.sin (Real.pi / 3), -2 * Real.cos (Real.pi / 3))
  in (sin α = -1/2) :=
sorry

end sin_of_angle_through_point_l658_658195


namespace count_nonempty_subsets_of_odds_l658_658905

theorem count_nonempty_subsets_of_odds : 
  let odd_numbers := {1, 3, 5, 7}
  in Finset.card (odd_numbers.powerset \ {∅}) = 15 := 
sorry

end count_nonempty_subsets_of_odds_l658_658905


namespace log_16_2_eq_one_fourth_l658_658074

theorem log_16_2_eq_one_fourth (a b c : ℝ) (h1 : a = 2^4) (h2 : b = log 2 2) (h3 : c = log 2 (2^4)) : 
  log 16 2 = 1 / 4 := 
by 
  sorry

end log_16_2_eq_one_fourth_l658_658074


namespace standard_equations_and_area_l658_658939

noncomputable def curve_param_eqns (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

def polar_line_eqn (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ - Real.sin θ) = 4

theorem standard_equations_and_area :
  (∀ t, curve_param_eqns t = (4 * t^2, 4 * t)) ∧
  (∀ ρ θ, polar_line_eqn ρ θ ↔ ρ * (Real.cos θ - Real.sin θ) = 4) ∧
  (∃ f : ℝ → ℝ, ∀ t, f t = 4 * t ∧ (f t)^2 = 4 * t^2) ∧
  (∃ g : ℝ → ℝ, ∀ ρ θ, g ρ θ = ρ * (Real.cos θ - Real.sin θ) ∧ g ρ θ = 4) ∧
  (∃ A B : ℝ × ℝ, (A.1, A.2^2 = A.1) ∧ (B.1, B.2^2 = B.1) ∧ A ≠ B ∧
  let D := Real.sqrt (A.2 - B.2)^2,
      H := 2 * Real.sqrt 2
  in ½ * D * H = 8 * Real.sqrt 5) :=
sorry

end standard_equations_and_area_l658_658939


namespace cube_volume_from_lateral_surface_area_l658_658680

theorem cube_volume_from_lateral_surface_area (A : ℝ) (hA : A = 100) : 
  (∃ s : ℝ, s^2 = 25) → (∃ V : ℝ, V = 125) :=
by
  assume exists_s
  existsi 125
  sorry

end cube_volume_from_lateral_surface_area_l658_658680


namespace tangents_intersection_perpendicular_parabola_l658_658979

theorem tangents_intersection_perpendicular_parabola :
  ∀ (C D : ℝ × ℝ), C.2 = 4 * C.1 ^ 2 → D.2 = 4 * D.1 ^ 2 → 
  (8 * C.1) * (8 * D.1) = -1 → 
  ∃ Q : ℝ × ℝ, Q.2 = -1 / 16 :=
by
  sorry

end tangents_intersection_perpendicular_parabola_l658_658979


namespace mary_james_not_next_to_each_other_l658_658209

/-- In a row of 10 chairs, two chairs (numbered 5 and 6) are broken and cannot be used.
    Mary and James each choose their remaining seats at random. 
    Prove that the probability that they do not sit next to each other is 11/14. -/
theorem mary_james_not_next_to_each_other : 
  let total_chairs := 10 in
  let broken := {5, 6} in
  let usable_chairs := total_chairs - broken.card in
  let total_ways := Nat.choose usable_chairs 2 in
  let adjacent_pairs := 6 in
  let prob_adjacent := adjacent_pairs / total_ways in
  let prob_not_adjacent := 1 - prob_adjacent in
  prob_not_adjacent = 11 / 14 :=
by
  let total_chairs := 10
  let broken := {5, 6}
  let usable_chairs := total_chairs - broken.to_finset.card
  let total_ways := Nat.choose usable_chairs 2
  let adjacent_pairs := 6
  let prob_adjacent := (adjacent_pairs : ℚ) / total_ways
  let prob_not_adjacent := 1 - prob_adjacent
  show prob_not_adjacent = 11 / 14
  sorry

end mary_james_not_next_to_each_other_l658_658209


namespace train_pass_platform_time_l658_658403

theorem train_pass_platform_time :
  ∀ (length_train length_platform : ℕ) (speed_kmph : ℕ),
  length_train = 140 →
  length_platform = 260 →
  speed_kmph = 60 →
  (length_train + length_platform : ℝ) / (speed_kmph * 1000 / 3600 : ℝ) ≈ 24 := by
  intros length_train length_platform speed_kmph 
  intros h_train h_platform h_speed 
  rw [h_train, h_platform, h_speed]
  sorry

end train_pass_platform_time_l658_658403


namespace cos_difference_identity_max_value_T_l658_658637

-- Problem I
theorem cos_difference_identity (A B α β : ℝ) (h1 : cos (α + β) = cos α * cos β - sin α * sin β)
  (h2 : cos (α - β) = cos α * cos β + sin α * sin β)
  (h3 : α + β = A) (h4 : α - β = B) :
  cos A - cos B = 2 * sin ((A + B) / 2) * sin ((A - B) / 2) :=
by sorry

-- Problem II
theorem max_value_T (A B C : ℝ) (h1 : A + B + C = π) :
  ∃ (T : ℝ), T = sin A + sin B + sin C + sin (π / 3) ∧ T ≤ 2 * sqrt 3 :=
by sorry

end cos_difference_identity_max_value_T_l658_658637


namespace bisect_chords_proof_l658_658111

noncomputable def bisect_chords (O : Point) (r : ℝ) (A B C D M N : Point) (circle : Circle O r) (chord_AB chord_CD : Line)
  (h_AB : is_chord circle A B) (h_CD : is_chord circle C D) 
  (h_parallel : is_parallel chord_AB chord_CD) 
  (h_AD : intersects (line_through A D) (line_through B C) = M) 
  (h_AC : intersects (line_through A C) (line_through B D) = N)
  (h_OP : ∃ P, is_diameter (line_through O P) ∧ is_perpendicular (line_through O P) chord_AB ∧ is_perpendicular (line_through O P) chord_CD) 
  : Prop :=
  is_perpendicular_bisector (line_through M N) chord_AB ∧ 
  is_perpendicular_bisector (line_through M N) chord_CD

theorem bisect_chords_proof (O : Point) (r : ℝ) (A B C D M N : Point) (circle : Circle O r) (chord_AB chord_CD : Line)
  (h_AB : is_chord circle A B) (h_CD : is_chord circle C D) 
  (h_parallel : is_parallel chord_AB chord_CD) 
  (h_AD : intersects (line_through A D) (line_through B C) = M) 
  (h_AC : intersects (line_through A C) (line_through B D) = N)
  (h_OP : ∃ P, is_diameter (line_through O P) ∧ is_perpendicular (line_through O P) chord_AB ∧ is_perpendicular (line_through O P) chord_CD) 
  : bisect_chords O r A B C D M N circle chord_AB chord_CD h_AB h_CD h_parallel h_AD h_AC h_OP :=
  sorry

end bisect_chords_proof_l658_658111


namespace airline_routes_coloring_l658_658742

theorem airline_routes_coloring (N : ℕ) (E : finset (ℕ × ℕ)) 
  (h1 : ∀ k : ℕ, 2 ≤ k ∧ k ≤ N → ∀ (S : finset ℕ), S.card = k → (E.filter (λ e, e.fst ∈ S ∧ e.snd ∈ S)).card ≤ 2 * k - 2) :
  ∃ f : (ℕ × ℕ) → fin 2, ∀ c : finset (ℕ × ℕ), (c ⊆ E) → ((c.image f).card = 1 → ¬cycle c) :=
sorry

end airline_routes_coloring_l658_658742


namespace train_length_proof_l658_658776

-- Define the given conditions as variables
def train_speed_kmh : ℝ := 45                 -- Speed in km/hr
def crossing_time_s : ℝ := 30                 -- Time in seconds
def bridge_length_m : ℝ := 215                -- Length of the bridge in meters

-- Convert speed from km/hr to m/s
def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

-- Calculate total distance covered by the train while crossing the bridge
def total_distance_m : ℝ := train_speed_ms * crossing_time_s

-- Define the length of the train
def train_length_m : ℝ := total_distance_m - bridge_length_m

-- Lean statement to prove the length of the train
theorem train_length_proof : train_length_m = 160 :=
by
  sorry

end train_length_proof_l658_658776


namespace donuts_combinations_l658_658039

theorem donuts_combinations : 
  ∀ (kinds : ℕ) (donuts : ℕ), kinds = 5 → donuts = 8 → 
    (∃ f : fin kinds → ℕ, (∀ i, 1 ≤ f i) ∧ (finset.univ.sum f = donuts)) → 
    (number_of_combinations kinds donuts = 35) :=
by {
  intros kinds donuts hkinds hdonuts hcond,
  have hsum : number_of_combinations kinds donuts = 35,
  { sorry },
  exact hsum
}

end donuts_combinations_l658_658039


namespace arithmetic_sequence_general_term_and_sum_l658_658852

theorem arithmetic_sequence_general_term_and_sum 
  (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS3 : S 3 = 0) (hS5 : S 5 = -5)
  (ha : ∀ n, S n = n * (2 + (n - 1) * (-1)) / 2)
  (b : ℕ → ℚ) (hb : ∀ n, b n = (1 : ℚ) / (a (2 * n - 1) * a (2 * n + 1)))
  (T : ℕ → ℚ) (hT : ∀ n, T n = ∑ i in finset.range n, b (i + 1)) :
  (∀ n, a n = 2 - n) ∧ (∀ n, T n = -n / (2 * n - 1)) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_l658_658852


namespace f_at_5pi_over_4_f_monotonically_increasing_intervals_f_max_min_on_interval_l658_658132

noncomputable def f : ℝ → ℝ := λ x, 2 * cos x * (sin x + cos x)

open Real

theorem f_at_5pi_over_4 : f (5 * π / 4) = 2 := 
sorry

theorem f_monotonically_increasing_intervals (k : ℤ) : 
  ∀ x, x ∈ Icc (k * π - 3 * π / 8) (k * π + π / 8) → monotone f := 
sorry

theorem f_max_min_on_interval : 
  ∃ (x_max x_min ∈ Icc 0 (π / 2)), 
  f x_max = sqrt 2 + 1 ∧ f x_min = 0 := 
sorry

end f_at_5pi_over_4_f_monotonically_increasing_intervals_f_max_min_on_interval_l658_658132


namespace unique_solution_exists_l658_658063

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l658_658063


namespace prove_n_equals_five_l658_658064

noncomputable def tan (x : ℝ) : ℂ := complex.sin x / complex.cos x

def root_of_unity (n : ℕ) : ℂ :=
  complex.cos ((2 * n * real.pi) / 14) + complex.sin ((2 * n * real.pi) / 14) * complex.I

def given_expression : ℂ :=
  (tan (real.pi / 7) + complex.I) / (tan (real.pi / 7) - complex.I)

theorem prove_n_equals_five :
  ∃ (n : ℕ), n = 5 ∧ given_expression = root_of_unity n :=
  by
    exists 5
    split
    { refl }
    { sorry }

end prove_n_equals_five_l658_658064


namespace cube_integer_condition_l658_658833

theorem cube_integer_condition :
  ∃! (n : ℤ), 0 ≤ n ∧ n < 30 ∧ ∃ (k : ℤ), n = (30 - n) * k^3 :=
by
  sorry

end cube_integer_condition_l658_658833


namespace unique_max_point_f_prime_two_zeros_of_f_l658_658878

noncomputable def f(x : ℝ) := Real.sin x - Real.log (x + 1)
def f' (x : ℝ) := Real.cos x - 1 / (x + 1)

-- Prove: f'(x) has a unique maximum point in the interval (-1, π / 2)
theorem unique_max_point_f_prime : 
  ∃! α ∈ Ioo (-1 : ℝ) (Real.pi / 2),
    ∀ x ∈ Ioo (-1 : ℝ) (Real.pi / 2),
      f' α ≥ f' x := sorry

-- Prove: f(x) has exactly two zeros
theorem two_zeros_of_f :
  ∃! x1 x2 : ℝ,
    x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 
    := sorry

end unique_max_point_f_prime_two_zeros_of_f_l658_658878


namespace line_intersects_circle_l658_658444

theorem line_intersects_circle 
  (k : ℝ)
  (x y : ℝ)
  (h_line : x = 0 ∨ y = -2)
  (h_circle : (x - 1)^2 + (y + 2)^2 = 16) :
  (-2 - -2)^2 < 16 := by
  sorry

end line_intersects_circle_l658_658444


namespace sphere_surface_area_l658_658894

theorem sphere_surface_area (r : ℝ) (hr : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi :=
by
  rw [hr]
  norm_num
  sorry

end sphere_surface_area_l658_658894


namespace cost_of_27_lilies_l658_658424

theorem cost_of_27_lilies
  (cost_18 : ℕ)
  (price_ratio : ℕ → ℕ → Prop)
  (h_cost_18 : cost_18 = 30)
  (h_price_ratio : ∀ n m c : ℕ, price_ratio n m ↔ c = n * 5 / 3 ∧ m = c * 3 / 5) :
  ∃ c : ℕ, price_ratio 27 c ∧ c = 45 := 
by
  sorry

end cost_of_27_lilies_l658_658424


namespace water_breaks_frequency_l658_658242

theorem water_breaks_frequency :
  ∃ W : ℕ, (240 / 120 + 10) = 240 / W :=
by
  existsi (20 : ℕ)
  sorry

end water_breaks_frequency_l658_658242


namespace Andy_collects_16_balls_l658_658032

-- Define the number of balls collected by Andy, Roger, and Maria.
variables (x : ℝ) (r : ℝ) (m : ℝ)

-- Define the conditions
def Andy_twice_as_many_as_Roger : Prop := r = x / 2
def Andy_five_more_than_Maria : Prop := m = x - 5
def Total_balls : Prop := x + r + m = 35

-- Define the main theorem to prove Andy's number of balls
theorem Andy_collects_16_balls (h1 : Andy_twice_as_many_as_Roger x r) 
                               (h2 : Andy_five_more_than_Maria x m) 
                               (h3 : Total_balls x r m) : 
                               x = 16 := 
by 
  sorry

end Andy_collects_16_balls_l658_658032


namespace line_through_intersection_perpendicular_l658_658439

theorem line_through_intersection_perpendicular (x y : ℝ) :
  (2 * x - 3 * y + 10 = 0) ∧ (3 * x + 4 * y - 2 = 0) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a = 2) ∧ (b = 3) ∧ (c = -2) ∧ (3 * a + 2 * b = 0)) :=
by
  sorry

end line_through_intersection_perpendicular_l658_658439


namespace domain_of_composition_l658_658870

def domain_f : set ℝ := {x | 0 < x ∧ x < 1}

theorem domain_of_composition : 
  ∀ (f : ℝ → ℝ), (∀ x, x ∈ domain_f → is_open {x | f x ∈ domain_f}) →
  domain (λ x, f (2 ^ x)) = {x | x < 0} :=
sorry

end domain_of_composition_l658_658870


namespace sin_pi_minus_alpha_l658_658910

theorem sin_pi_minus_alpha (α : ℝ) (h1 : cos (2 * π - α) = (√5) / 3) (h2 : α ∈ Ioo (-π / 2) 0) : 
  sin (π - α) = -2 / 3 := 
by 
  sorry

end sin_pi_minus_alpha_l658_658910


namespace find_y_value_l658_658456

theorem find_y_value : ∃ y : ℝ, y = 79 / 12 ∧ 16 ^ (-3) = (4 ^ (72 / y)) / (8 ^ (37 / y) * 16 ^ (28 / y)) :=
by
  use 79 / 12
  sorry

end find_y_value_l658_658456


namespace total_cost_l658_658202

def CategoryA := 15
def CategoryB := 12
def CategoryC := 10
def CategoryD := 7.5
def CategoryE := 5

def BooksA := 6
def BooksB := 4
def BooksC := 8
def BooksD := 10
def BooksE := 5

def DiscountA := 0.20
def DiscountB := 0.15
def DiscountC := 0.10
def DiscountD := 0.25
def DiscountE := 0.05

def CostA := BooksA * (CategoryA * (1 - DiscountA))
def CostB := BooksB * (CategoryB * (1 - DiscountB))
def CostC := (5 * (CategoryC * (1 - DiscountC))) + (3 * CategoryC)
def CostD := (8 * (CategoryD * (1 - DiscountD))) + (2 * CategoryD)
def CostE := BooksE * (CategoryE * (1 - DiscountE))

theorem total_cost : CostA + CostB + CostC + CostD + CostE = 271.55 :=
by
  simp [CostA, CostB, CostC, CostD, CostE, CategoryA, CategoryB, CategoryC, CategoryD, CategoryE,
        BooksA, BooksB, BooksC, BooksD, BooksE, DiscountA, DiscountB, DiscountC, DiscountD, DiscountE]
  sorry

end total_cost_l658_658202


namespace question1_question2_l658_658883

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem question1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem question2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2 := sorry

end question1_question2_l658_658883


namespace find_x0_l658_658503

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5

theorem find_x0 :
  (∃ x0 : ℝ, f (g x0) = 1) → (∃ x0 : ℝ, x0 = 4/3) :=
by
  sorry

end find_x0_l658_658503


namespace pascal_fifth_element_row_20_l658_658715

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l658_658715


namespace proof_problem_l658_658648

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l658_658648


namespace range_of_function_sin_cos_l658_658825

theorem range_of_function_sin_cos : 
  (∃ y, y = sin x + cos x + sin x * cos x) → 
  ∃ (a b : ℝ), ∀ x, a ≤ sin x + cos x + sin x * cos x ∧ sin x + cos x + sin x * cos x ≤ b :=
by
  use [-1, (1/2) + sqrt 2]
  sorry

end range_of_function_sin_cos_l658_658825


namespace yen_per_cad_l658_658750

theorem yen_per_cad (yen cad : ℝ) (h : yen / cad = 5000 / 60) : yen = 83 := by
  sorry

end yen_per_cad_l658_658750


namespace coloring_PQR_coloring_PQRS_l658_658807

-- Prove for coloring points P, Q, R
theorem coloring_PQR 
  (A B C : Finset α) 
  (hA : A.nonempty) 
  (hB : B.card > 1) 
  (hC : C.card > 1) : 
  (∃ P Q R : α, P ∈ A ∧ Q ∈ B ∧ Q ≠ P ∧ R ∈ C ∧ R ≠ Q) → 
  (A.card * (B.card - 1) * (C.card - 1)) = 
  A.card * (B.card - 1) * (C.card - 1) :=
sorry

-- Prove for coloring points P, Q, R, S
theorem coloring_PQRS 
  (A B C D : Finset α) 
  (hA : A.nonempty) 
  (hB : B.card > 1) 
  (hC : C.card > 1) 
  (hD : D.card > 1) : 
  (∃ P Q R S : α, P ∈ A ∧ Q ∈ B ∧ Q ≠ P ∧ R ∈ C ∧ R ≠ Q ∧ S ∈ D ∧ S ≠ R) → 
  (A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)) = 
  A.card * (B.card - 1) * (C.card - 1) * (D.card - 1) :=
sorry

end coloring_PQR_coloring_PQRS_l658_658807


namespace number_of_real_number_pairs_eq_3_l658_658135

-- Define the function
def f (x : ℝ) : ℝ := if x ≥ 0 then (2 * x) / (x + 1) else (2 * x) / (1 - x)

-- Definition of the set N
def N (M : set ℝ) : set ℝ := {y | ∃ x ∈ M, f x = y}

-- Main theorem statement
theorem number_of_real_number_pairs_eq_3 (a b : ℝ) (h : a < b) :
  {M : set ℝ | M = Icc a b ∧ N (Icc a b) = Icc a b}.to_finset.card = 3 :=
sorry

end number_of_real_number_pairs_eq_3_l658_658135


namespace count_integer_n_for_8000_l658_658097

theorem count_integer_n_for_8000 (n : ℤ) :
  (8000 : ℚ) * (2/3)^n ∈ ℤ ↔ n = 0 := 
sorry

example : finset.card {n : ℤ | (8000 : ℚ) * (2/3)^n ∈ ℤ} = 1 :=
by
  have h : ∀ n, (8000 * (2/3)^n : ℚ) ∈ ℤ ↔ n = 0 := count_integer_n_for_8000
  rw [←finset.card_singleton 0, finset.card_congr (λ n h, ⟨h.1, by simp⟩) h.2]
  sorry

end count_integer_n_for_8000_l658_658097


namespace area_triangle_DEF_l658_658009

theorem area_triangle_DEF (Q : Point) (u1 u2 u3 : Triangle) (area_u1 : ℝ) (area_u2 : ℝ) (area_u3 : ℝ) (area_segment : ℝ) (area_DEF : ℝ) :
  interior Q u1 ∧ interior Q u2 ∧ interior Q u3 ∧
  lines_parallel Q u1 ∧ lines_parallel Q u2 ∧ lines_parallel Q u3 ∧ 
  area u1 = 16 ∧ area u2 = 25 ∧ area u3 = 36 ∧
  segment_cut_by_circle Q u3 = 9 →
  area DEF = 225 :=
sorry

end area_triangle_DEF_l658_658009


namespace distinct_numbers_impossible_l658_658013

theorem distinct_numbers_impossible
  (a b c d : ℝ)
  (h1 : a * c = b * d)
  (h2 : a + c = b + d) :
  ({a, b, c, d}.card < 4) :=
by
  sorry

end distinct_numbers_impossible_l658_658013


namespace shaded_area_is_10_l658_658334

-- Definitions based on conditions:
def rectangle_area : ℕ := 12
def unshaded_triangle_area : ℕ := 2

-- Proof statement without the actual proof.
theorem shaded_area_is_10 : rectangle_area - unshaded_triangle_area = 10 := by
  sorry

end shaded_area_is_10_l658_658334


namespace main_theorem_l658_658488

-- Definitions and conditions
noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_condition_1 : ∀ x > 0, f (x + 2) = -1 / f x
axiom f_condition_2 : ∀ x ∈ set.Ico 0 2, f x = real.log (x + 1) / real.log 2

-- Main theorem to prove
theorem main_theorem : f 2015 + f 2016 = -1 :=
by
  sorry

end main_theorem_l658_658488


namespace determine_a_l658_658967

theorem determine_a (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (eq1 : a^2 - b^2 - c^2 + a * b = 2100)
  (eq2 : a^2 + 2 * b^2 + 4 * c^2 - 3 * a * b - 2 * a * c - b * c = -2000) :
  a = 11 :=
by sorry

end determine_a_l658_658967


namespace triangles_A2B2C2_A3B3C3_similar_l658_658962

noncomputable section

open Geometry

variables {A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point} 
          {Γ : Circle} {AB AC BC : Line} {AB1C1_circ BC1A1_circ CA1B1_circ : Circle}

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom circumcircle_ABC : Circle Γ
axiom A1_on_BC : On A1 BC
axiom B1_on_CA : On B1 CA
axiom C1_on_AB : On C1 AB
axiom A3_reflection : Reflection A1 (Midpoint B C) = A3
axiom B3_reflection : Reflection B1 (Midpoint A C) = B3
axiom C3_reflection : Reflection C1 (Midpoint A B) = C3
axiom A2_second_intersection : ∃ x, On x Γ ∧ On x AB1C1_circ ∧ x ≠ A ∧ x = A2
axiom B2_second_intersection : ∃ x, On x Γ ∧ On x BC1A1_circ ∧ x ≠ B ∧ x = B2
axiom C2_second_intersection : ∃ x, On x Γ ∧ On x CA1B1_circ ∧ x ≠ C ∧ x = C2

-- The main goal
theorem triangles_A2B2C2_A3B3C3_similar :
  Similar (Triangle.mk A2 B2 C2) (Triangle.mk A3 B3 C3) :=
sorry -- The proof is omitted as instructed

end triangles_A2B2C2_A3B3C3_similar_l658_658962


namespace common_tangent_line_slope_2_l658_658537

noncomputable def tangent_point_ln (f' : ℝ → ℝ) := 
  ∃ x, f' x = 2

noncomputable def curve_ln (x : ℝ) := log x 

noncomputable def curve_quadratic (a x : ℝ) := a * x^2

noncomputable def common_tangent_slope (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
    (x1 ≠ x2) ∧ 
    ((curve_quadratic a x1).deriv = 2) ∧ 
    ((curve_ln x2).deriv = 2)

theorem common_tangent_line_slope_2 (a : ℝ) :
  common_tangent_slope a → a = 1 / (Real.log 2 + 1) := 
sorry

end common_tangent_line_slope_2_l658_658537


namespace cos_identity_l658_658482

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = √3 / 3) : 
  Real.cos (5 * π / 6 - θ) = -(√3 / 3) :=
by 
  sorry

end cos_identity_l658_658482


namespace eccentricity_of_ellipse_is_sqrt2_div_2_l658_658258

-- Define the ellipse and its properties
structure Ellipse where
  a b : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)

-- Define the points and conditions
structure Points where
  F1 F2 A B : ℝ × ℝ

-- Conditions for the points and intersection
structure IntersectionCondition where
  k : ℝ
  h1 : |A - F1| = 3 * k
  h2 : |B - F1| = k
  h3 : |A - B| = 4 * k
  cos_angle : cos (angle F2 A B) = 3 / 5

-- The definition to state the problem
def ellipse_eccentricity_proof (E : Ellipse) (points : Points) (cond : IntersectionCondition) : ℝ :=
  let c := (sqrt 2 / 2) * E.a
  let e := c / E.a
  e

-- statement of the problem
theorem eccentricity_of_ellipse_is_sqrt2_div_2 (E : Ellipse) (points : Points) (cond : IntersectionCondition) :
  ellipse_eccentricity_proof E points cond = sqrt 2 / 2 :=
sorry

end eccentricity_of_ellipse_is_sqrt2_div_2_l658_658258


namespace weight_lifting_ratio_l658_658933

theorem weight_lifting_ratio :
  ∀ (F S : ℕ), F + S = 600 ∧ F = 300 ∧ 2 * F = S + 300 → F / S = 1 :=
by
  intro F S
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_lifting_ratio_l658_658933


namespace perpendicular_lines_l658_658892

theorem perpendicular_lines (a : ℝ) :
  let l1 := (fun x y : ℝ => (3 * a + 2) * x + (a - 1) * y - 2 = 0) in
  let l2 := (fun x y : ℝ => (a - 1) * x + y + 1 = 0) in
  (∃ x1 y1, l1 x1 y1 ∧ ∃ x2 y2, l2 x2 y2) → 
  (-((3 * a + 2) / (a - 1)) * -(a - 1)) = -1 → 
  a = 1 ∨ a = -1 :=
by
  intros l1 l2 h1 h2
  sorry

end perpendicular_lines_l658_658892


namespace total_watermelons_l658_658572

/-- Proof statement: Jason grew 37 watermelons and Sandy grew 11 watermelons. 
    Prove that they grew a total of 48 watermelons. -/
theorem total_watermelons (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ) 
                         (h1 : jason_watermelons = 37) (h2 : sandy_watermelons = 11) :
  total_watermelons = 48 :=
by
  sorry

end total_watermelons_l658_658572


namespace second_triangle_weight_l658_658774

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def weight_of_second_triangle (m_1 : ℝ) (s_1 s_2 : ℝ) : ℝ :=
  m_1 * (area_equilateral_triangle s_2 / area_equilateral_triangle s_1)

theorem second_triangle_weight :
  let m_1 := 12   -- weight of the first triangle in ounces
  let s_1 := 3    -- side length of the first triangle in inches
  let s_2 := 5    -- side length of the second triangle in inches
  weight_of_second_triangle m_1 s_1 s_2 = 33.3 :=
by
  sorry

end second_triangle_weight_l658_658774


namespace geometric_sum_a4_a6_l658_658943

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_a4_a6 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_pos : ∀ n, a n > 0) 
(h_cond : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) : a 4 + a 6 = 10 :=
by
  sorry

end geometric_sum_a4_a6_l658_658943


namespace area_PQRS_l658_658699

-- Defining the given conditions
def PQ : ℝ := 8
def PR : ℝ := 10
def PS : ℝ := 18

-- Defining key points and shapes
structure Point :=
(x y : ℝ)

-- Consider the coordinates in a general planar geometry setup
def P : Point := ⟨0, 0⟩
def Q : Point := ⟨PQ, 0⟩
def R : Point := ⟨PR / 2, (sqrt (PR ^ 2 - (PR / 2) ^2))⟩ -- Since PR is the hypotenuse in triangle PRS and it is right triangle

-- Calculating RS where PRS is a right triangle with PR and RS legs, and PS hypotenuse
def RS : ℝ := sqrt (PS ^ 2 - PR ^ 2)

-- Calculating the areas
def area_PQR : ℝ := (1 / 2) * PQ * PR
def area_PRS : ℝ := (1 / 2) * PR * RS

-- Stating the final area of quadrilateral PQRS
theorem area_PQRS : area_PQR + area_PRS = 40 + 30 * sqrt 14 :=
by
  -- Proof would go here
  sorry

end area_PQRS_l658_658699


namespace diagonal_less_than_midpoints_distance_l658_658856

-- Define the quadrilateral and its properties
variables {A B C D K L : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K] [MetricSpace L]
variables [LieGroup A] [LieGroup B] [LieGroup C] [LieGroup D] [LieGroup K] [LieGroup L]
variables (AB CD AC : ℝ)

-- Conditions: Midpoints and tangency properties
def is_midpoint (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] (X_mid : X → Y → Z → Prop) := sorry
def circumcircle_tangent (T1 T2 E : Type) [MetricSpace T1] [MetricSpace T2] [MetricSpace E] (Tangent : T1 → T2 → E → Prop)= sorry

-- Proof problem
theorem diagonal_less_than_midpoints_distance (A B C D K L : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K] [MetricSpace L] 
  [LieGroup A] [LieGroup B] [LieGroup C] [LieGroup D] [LieGroup K] [LieGroup L]
  (is_midpoint K A B)
  (is_midpoint L C D)
  (circumcircle_tangent A B C D)
  (circumcircle_tangent A C D B) :
  AC < distance K L :=
sorry

end diagonal_less_than_midpoints_distance_l658_658856


namespace vehicle_gap_l658_658341

theorem vehicle_gap (initial_gap : ℕ) (speed_x speed_y hours : ℕ) : 
  initial_gap = 22 → speed_x = 36 → speed_y = 45 → hours = 5 → 
  (speed_y * hours - speed_x * hours - initial_gap) = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end vehicle_gap_l658_658341


namespace prime_divides_square_l658_658978

theorem prime_divides_square (p a b : ℤ)
  (p_prime : Prime p)
  (p_geq_5 : p ≥ 5)
  (h : a / b = (1 + ∑ i in Finset.range (p - 1) + 1, 1 / (i + 1))) :
  p ∣ a^2 :=
by
  sorry

end prime_divides_square_l658_658978


namespace H_fractal_sequence_l658_658326

theorem H_fractal_sequence :
  let a : ℕ → ℕ := λ n, (nat.rec_on n 1 (λ n aₙ, 2 * aₙ + 1))
  a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7 → a 4 = 15 :=
by
  intro a h
  cases h with ha1 h
  cases h with ha2 ha3
  rw [←ha1, ←ha2, ←ha3]
  sorry

end H_fractal_sequence_l658_658326


namespace distance_to_other_focus_l658_658112

def is_hyperbola (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

def is_tangent_to_asymptote (a b : ℝ) (center circle_radius : ℝ × ℝ) : Prop :=
  let (h, k) := center in
  (circle_radius = sqrt 20) ∧ (6 * b / sqrt (a^2 + b^2) = sqrt 20)

theorem distance_to_other_focus (a b c : ℝ) (F1 P : ℝ × ℝ) : 
  c = 6 → a^2 + b^2 = c^2 → b = sqrt 20 → is_hyperbola a b → 
  is_tangent_to_asymptote a b (6, 0) (sqrt 20) →
  let F2 := (-c, 0) in 
  dist P F1 = 9 → dist P F2 = 17 :=
by
  intros
  sorry

end distance_to_other_focus_l658_658112


namespace part_I_solution_set_part_II_range_of_a_l658_658887

section

variable (x a : ℝ)

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 1 / 2)

-- Part I: Prove the solution set for f(x) <= x + 10 when a = 5 / 2 is { x | -7 / 3 ≤ x ≤ 13 }
theorem part_I_solution_set (x : ℝ) :
    (f x (5 / 2) ≤ x + 10) ↔ (-7 / 3 ≤ x ∧ x ≤ 13) :=
by
    sorry

-- Part II: Prove the range of a such that f(x) ≥ a for all x ∈ ℝ is (-∞, 1 / 4]
theorem part_II_range_of_a (a : ℝ) :
    (∀ x : ℝ, f x a ≥ a) ↔ (a ≤ 1 / 4) :=
by
    sorry

end

end part_I_solution_set_part_II_range_of_a_l658_658887


namespace min_distance_ellipse_line_l658_658312

theorem min_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x ^ 2) / 16 + (y ^ 2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
             (∀ (x y : ℝ), ellipse x y → ∃ (d' : ℝ), line x y → d' ≥ d) :=
  sorry

end min_distance_ellipse_line_l658_658312


namespace difference_in_spectators_l658_658223

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l658_658223


namespace simplify_and_rationalize_correct_l658_658645

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l658_658645


namespace boat_distance_along_stream_l658_658559

-- Define the conditions
def boat_speed_still_water := 15 -- km/hr
def distance_against_stream_one_hour := 9 -- km

-- Define the speed of the stream
def stream_speed := boat_speed_still_water - distance_against_stream_one_hour -- km/hr

-- Define the effective speed along the stream
def effective_speed_along_stream := boat_speed_still_water + stream_speed -- km/hr

-- Define the proof statement
theorem boat_distance_along_stream : effective_speed_along_stream = 21 :=
by
  -- Given conditions and definitions, the steps are assumed logically correct
  sorry

end boat_distance_along_stream_l658_658559


namespace triangle_is_right_l658_658850

variable {a b c : ℝ}

theorem triangle_is_right
  (h : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  (a * a + b * b = c * c) :=
sorry

end triangle_is_right_l658_658850


namespace jenny_spent_on_toys_l658_658954

-- Define the conditions
def adoption_fee : ℕ := 50
def vet_visits_cost : ℕ := 500
def monthly_food_cost : ℕ := 25
def yearly_food_cost : ℕ := 12 * monthly_food_cost
def total_joint_cost : ℕ := adoption_fee + vet_visits_cost + yearly_food_cost
def jenny_total_spent : ℕ := 625
def jenny_share_of_joint_cost : ℕ := total_joint_cost / 2

-- Prove Jenny's total spend on toys
theorem jenny_spent_on_toys : 
  let toys_cost := jenny_total_spent - jenny_share_of_joint_cost in
  toys_cost = 200 := 
by 
  sorry -- proof goes here

end jenny_spent_on_toys_l658_658954


namespace length_of_MN_l658_658506

noncomputable def parabola := {p : ℝ × ℝ // p.2 ^ 2 = 8 * p.1}
def focus : ℝ × ℝ := (2, 0)
def directrix : ℝ := -2

def on_directrix (P : ℝ × ℝ) : Prop := P.1 = directrix

def intersection_points (l : ℝ → ℝ) (M N : parabola) : Prop :=
∃ x1 y1 x2 y2,
  M.val = (x1, y1) ∧ N.val = (x2, y2) ∧
  l x1 = y1 ∧ l x2 = y2

theorem length_of_MN (P : ℝ × ℝ) (M N : parabola) 
  (hP : on_directrix P)
  (h_intersection : intersection_points (λ x, real.sqrt 3 * (x - 2)) M N)
  (h_vector : real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) =
               3 * real.sqrt ((M.val.1 - focus.1)^2 + (M.val.2 - focus.2)^2)) :
  real.sqrt ((M.val.1 - N.val.1)^2 + (M.val.2 - N.val.2)^2) = 32/3 := sorry

end length_of_MN_l658_658506


namespace isosceles_triangle_l658_658922

variables {A B C : ℝ} -- angles in the triangle
variables (a b c : ℝ) -- sides opposite to angles A, B, C respectively

-- condition: sin A = 2 * sin C * cos B
def sin_identity : Prop := sin A = 2 * sin C * cos B

-- assuming the given condition
axiom triangle_condition : sin_identity A B C

-- target: prove that triangle is isosceles
theorem isosceles_triangle (h : sin_identity A B C) : is_isosceles A B C := sorry

end isosceles_triangle_l658_658922


namespace convert_to_polar_coordinates_l658_658436

theorem convert_to_polar_coordinates :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (let r := 4 in r > 0 ∧ (2, -2 * Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ)) :=
  sorry

end convert_to_polar_coordinates_l658_658436


namespace fifth_element_row_20_pascal_triangle_l658_658717

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l658_658717


namespace minimum_area_triangle_AOB_l658_658944

theorem minimum_area_triangle_AOB :
  ∃ k : ℝ, ∃ x_A x_B y_A y_B : ℝ,
  (∀ x y : ℝ, y = k * x + (1 / k) ∧ y^2 = 4 * x → k ≥ 0 ) →
  (x_A^2 - y_A^2 = 1 ∧ x_B^2 - y_B^2 = 1) →
  (x_A * y_B - x_B * y_A = 2 * (2 * sqrt 5)) :=
begin
  sorry
end

end minimum_area_triangle_AOB_l658_658944


namespace pascal_fifth_element_row_20_l658_658714

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l658_658714


namespace simplify_and_rationalize_correct_l658_658644

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l658_658644


namespace odd_divisors_count_100_l658_658906

theorem odd_divisors_count_100 : 
  (finset.filter (λn, ∃ m : ℕ, m * m = n) (finset.range 101)).card = 10 :=
by
  sorry

end odd_divisors_count_100_l658_658906


namespace haley_deleted_pictures_l658_658736

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (remaining_pictures : ℕ) (deleted_pictures : ℕ)

theorem haley_deleted_pictures :
  zoo_pictures = 50 → museum_pictures = 8 → remaining_pictures = 20 →
  deleted_pictures = zoo_pictures + museum_pictures - remaining_pictures →
  deleted_pictures = 38 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end haley_deleted_pictures_l658_658736


namespace equation_of_line_passing_through_A_equation_of_circle_l658_658557

variable {α β γ : ℝ}
variable {a b c u v w : ℝ}
variable (A : ℝ × ℝ × ℝ) -- Barycentric coordinates of point A

-- Statement for the equation of a line passing through point A in barycentric coordinates
theorem equation_of_line_passing_through_A (A : ℝ × ℝ × ℝ) : 
  ∃ (u v w : ℝ), u * α + v * β + w * γ = 0 := by
  sorry

-- Statement for the equation of a circle in barycentric coordinates
theorem equation_of_circle {u v w : ℝ} :
  -a^2 * β * γ - b^2 * γ * α - c^2 * α * β +
  (u * α + v * β + w * γ) * (α + β + γ) = 0 := by
  sorry

end equation_of_line_passing_through_A_equation_of_circle_l658_658557


namespace sum_of_intervals_length_l658_658873

theorem sum_of_intervals_length (m : ℝ) (h : m ≠ 0) (h_pos : m > 0) :
  (∃ l : ℝ, ∀ x : ℝ, (1 < x ∧ x ≤ x₁) ∨ (2 < x ∧ x ≤ x₂) → 
  l = x₁ - 1 + x₂ - 2) → 
  l = 3 / m :=
sorry

end sum_of_intervals_length_l658_658873


namespace hexagon_not_necesarily_planar_l658_658186

structure Hexagon :=
  (v1 v2 v3 v4 v5 v6 : ℝ^3)
  (opp_parallel : (v1 - v4) ∥ (v2 - v5) ∥ (v3 - v6))

def is_planar (h: Hexagon) : Prop :=
  ∃ (n : ℝ^3), ∀ (v : ℝ^3), v ∈ {h.v1, h.v2, h.v3, h.v4, h.v5, h.v6} → (n ⬝ v = n ⬝ h.v1)

theorem hexagon_not_necesarily_planar (h: Hexagon) : ¬ is_planar h := 
sorry

end hexagon_not_necesarily_planar_l658_658186


namespace length_XY_10_minus_5_sqrt2_l658_658563

noncomputable def length_XY (r : ℝ) : ℝ :=
  let OY := r
  let OX := r / Real.sqrt 2
  OY - OX

theorem length_XY_10_minus_5_sqrt2 :
  ∀ (r : ℝ), r = 10 → length_XY r = 10 - 5 * Real.sqrt 2 :=
by 
  intros r hr
  rw [hr]
  simp [length_XY, Real.sqrt, Real.div_eq_mul_inv, -Real.sqrt_pow_two]
  sorry

end length_XY_10_minus_5_sqrt2_l658_658563


namespace estate_area_correct_l658_658396

-- Define the basic parameters given in the problem
def scale : ℝ := 500  -- 500 miles per inch
def width_on_map : ℝ := 5  -- 5 inches
def height_on_map : ℝ := 3  -- 3 inches

-- Define actual dimensions based on the scale
def actual_width : ℝ := width_on_map * scale  -- actual width in miles
def actual_height : ℝ := height_on_map * scale  -- actual height in miles

-- Define the expected actual area of the estate
def actual_area : ℝ := 3750000  -- actual area in square miles

-- The main theorem to prove
theorem estate_area_correct :
  (actual_width * actual_height) = actual_area := by
  sorry

end estate_area_correct_l658_658396


namespace find_value_of_k_l658_658392

noncomputable def line_parallel_and_point_condition (k : ℝ) :=
  ∃ (m : ℝ), m = -5/4 ∧ (22 - (-8)) / (k - 3) = m

theorem find_value_of_k : ∃ k : ℝ, line_parallel_and_point_condition k ∧ k = -21 :=
by
  sorry

end find_value_of_k_l658_658392


namespace john_rent_savings_l658_658957

theorem john_rent_savings : 
  let former_annual_cost := 1500 * 12
  let new_first_half := 2800 * 6
  let new_last_half := (2800 + 0.05 * 2800) * 6
  let new_annual_rent := new_first_half + new_last_half
  let new_winter_utilities := 200 * 3
  let new_other_utilities := 150 * 9
  let new_annual_utilities := new_winter_utilities + new_other_utilities
  let new_annual_cost := new_annual_rent + new_annual_utilities
  let johns_cost := new_annual_cost / 2
  in johns_cost - former_annual_cost = 195 :=
sorry

end john_rent_savings_l658_658957


namespace conditional_expected_value_l658_658053

noncomputable def mean_X : ℝ := 15.5
noncomputable def mean_Y : ℝ := 10.0
noncomputable def stddev_X : ℝ := 1.5
noncomputable def stddev_Y : ℝ := 2.0
noncomputable def correlation_XY : ℝ := 0.8
noncomputable def x : ℝ := mean_X - 2 * stddev_X

theorem conditional_expected_value : 
  E[Y|X = x] = 7 :=
by
  -- Proof steps would go here
  sorry

end conditional_expected_value_l658_658053


namespace carol_first_six_l658_658412

noncomputable def prob_six : ℚ := 1 / 6
noncomputable def not_six : ℚ := 5 / 6
noncomputable def first_term : ℚ := not_six * not_six * prob_six
noncomputable def common_ratio : ℚ := (not_six) ^ 4
noncomputable def carol_prob_first_six : ℚ := first_term / (1 - common_ratio)

theorem carol_first_six : carol_prob_first_six = 125 / 671 := sorry

end carol_first_six_l658_658412


namespace volume_of_solid_rotation_l658_658803

noncomputable def volume_of_solid := 
  (∫ y in (0:ℝ)..(1:ℝ), (y^(2/3) - y^2)) * Real.pi 

theorem volume_of_solid_rotation :
  volume_of_solid = (4 * Real.pi / 15) :=
by
  sorry

end volume_of_solid_rotation_l658_658803


namespace blanket_thickness_l658_658243

theorem blanket_thickness (n : ℕ) (initial_thickness : ℕ) (fold_multiplier : ℕ → ℕ) (total_thickness : ℕ) : 
  n = 5 ∧ initial_thickness = 3 ∧ fold_multiplier = (λ c : ℕ, 2 ^ c) → 
  total_thickness = (∑ c in (finset.range 5).map (λ c, 3 * 2 ^ (c + 1))) :=
sorry

end blanket_thickness_l658_658243


namespace intersecting_chords_l658_658748

theorem intersecting_chords (n : ℕ) (h1 : 0 < n) :
  ∃ intersecting_points : ℕ, intersecting_points ≥ n :=
  sorry

end intersecting_chords_l658_658748


namespace cannot_rearrange_pairs_l658_658950

theorem cannot_rearrange_pairs :
  ¬∃ P : list ℕ, 
    (∀ i ∈ P, 1 ≤ i ∧ i ≤ 1986) ∧ 
    (multiset.card (multiset.filter (λ x, x = i) (P : multiset ℕ)) = 2) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 1986 → list.indexes_of i P).last - (list.indexes_of i P).head = i :=
begin
  sorry
end

end cannot_rearrange_pairs_l658_658950


namespace find_magnitude_of_sum_l658_658900

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, -2)
def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : vector) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
def vector_add (u v : vector) : vector := (u.1 + v.1, u.2 + v.2)

theorem find_magnitude_of_sum :
  ∀ x, dot_product a (b x) = 0 → magnitude (vector_add a (b x)) = 5 :=
by
  sorry

end find_magnitude_of_sum_l658_658900


namespace triangular_pyramid_volume_l658_658780

noncomputable def volume_of_pyramid (b a x : ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * b * x * x

theorem triangular_pyramid_volume :
  ∀ (a b : ℝ), (b = √(2 * a^2 - b^2 / 1)) ∧
  (b^2 = (1 / 3) * a^2) ∧
  (1 = b / 2) ∧
  (a = √3) →
  volume_of_pyramid b a √2 = 2 / 3 :=
by
  intros a b h
  sorry

end triangular_pyramid_volume_l658_658780


namespace consecutive_probability_is_two_fifths_l658_658331

-- Conditions
def total_days : ℕ := 5
def select_days : ℕ := 2

-- Total number of basic events (number of ways to choose 2 days out of 5)
def total_events : ℕ := Nat.choose total_days select_days -- This is C(5, 2)

-- Number of basic events where 2 selected days are consecutive
def consecutive_events : ℕ := 4

-- Probability that the selected 2 days are consecutive
def consecutive_probability : ℚ := consecutive_events / total_events

-- Theorem to be proved
theorem consecutive_probability_is_two_fifths :
  consecutive_probability = 2 / 5 :=
by
  sorry

end consecutive_probability_is_two_fifths_l658_658331


namespace range_H_l658_658346

noncomputable def H (x : ℝ) : ℝ :=
  abs (x + 2) - abs (x - 2)

theorem range_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_H_l658_658346


namespace area_of_triangle_ABC_l658_658328

-- Define points A, B, C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the coordinates of points A, B, C meeting the conditions
def A : Point := {x := -5, y := 2}
def B : Point := {x := 0, y := 3}
def C : Point := {x := 7, y := 4}

-- Function to calculate the area of a triangle given three points
def triangle_area (A B C : Point) : ℝ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))/2)

-- The theorem to prove
theorem area_of_triangle_ABC : triangle_area A B C = 1 :=
by
  -- Proof construction is omitted as per the instructions
  sorry

end area_of_triangle_ABC_l658_658328


namespace arithmetic_sequence_a6_l658_658323

theorem arithmetic_sequence_a6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d a1 n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : S 3 = 12)
  (h3 : S = λ n, n * (2 * a 1 + (n - 1) * d) / 2) 
  (h4 : a = λ n, a 1 + (n - 1) * d) :
  a 6 = 12 :=
sorry

end arithmetic_sequence_a6_l658_658323


namespace problem_lean_l658_658526

theorem problem_lean (n m : ℕ) (coprime : Nat.coprime n m) (h : (2013 * 2013 : ℚ) / (2014 * 2014 + 2012) = n / m) : n + m = 1343 :=
sorry

end problem_lean_l658_658526


namespace custom_op_eval_l658_658528

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b - a^2 * b

theorem custom_op_eval :
  custom_op 3 4 = -4 :=
by
  sorry

end custom_op_eval_l658_658528


namespace largest_variance_is_B_l658_658930

open Finset
open Real

-- Define the frequencies and the constraint sum to 1
variable (p_1 p_2 p_3 p_4 : ℝ)
variable (h_sum : p_1 + p_2 + p_3 + p_4 = 1)

-- Define the expected value calculation
def expected_value (p_1 p_2 p_3 p_4 : ℝ) : ℝ :=
  1 * p_1 + 2 * p_2 + 3 * p_3 + 4 * p_4

-- Define the variance calculation
def variance (p_1 p_2 p_3 p_4 : ℝ) (E : ℝ) : ℝ := 
  (1 - E) ^ 2 * p_1 + (2 - E) ^ 2 * p_2 + (3 - E) ^ 2 * p_3 + (4 - E) ^ 2 * p_4

-- Define each option's frequencies
def option_A := (0.1, 0.4, 0.4, 0.1)
def option_B := (0.4, 0.1, 0.1, 0.4)
def option_C := (0.2, 0.3, 0.3, 0.2)
def option_D := (0.3, 0.2, 0.2, 0.3)

-- Proof problem: option B has the highest variance
theorem largest_variance_is_B : 
  let E_A := expected_value 0.1 0.4 0.4 0.1,
      E_B := expected_value 0.4 0.1 0.1 0.4,
      E_C := expected_value 0.2 0.3 0.3 0.2,
      E_D := expected_value 0.3 0.2 0.2 0.3,
      V_A := variance 0.1 0.4 0.4 0.1 E_A,
      V_B := variance 0.4 0.1 0.1 0.4 E_B,
      V_C := variance 0.2 0.3 0.3 0.2 E_C,
      V_D := variance 0.3 0.2 0.2 0.3 E_D
  in V_B > V_A ∧ V_B > V_C ∧ V_B > V_D :=
sorry

end largest_variance_is_B_l658_658930


namespace problem1_problem2_problem3_problem4_l658_658109

section

variables (x y : Real)

-- Given conditions
def x_def : x = 3 + 2 * Real.sqrt 2 := sorry
def y_def : y = 3 - 2 * Real.sqrt 2 := sorry

-- Problem 1: Prove x + y = 6
theorem problem1 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x + y = 6 := 
by sorry

-- Problem 2: Prove x - y = 4 * sqrt 2
theorem problem2 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x - y = 4 * Real.sqrt 2 :=
by sorry

-- Problem 3: Prove xy = 1
theorem problem3 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x * y = 1 := 
by sorry

-- Problem 4: Prove x^2 - 3xy + y^2 - x - y = 25
theorem problem4 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x^2 - 3 * x * y + y^2 - x - y = 25 :=
by sorry

end

end problem1_problem2_problem3_problem4_l658_658109


namespace gcd_k_n_condition_l658_658257

theorem gcd_k_n_condition {n : ℕ} (hn : 3 < n) :
  ∀ k : ℕ, (1 ≤ k ∧ k ≤ n) →
  (∀ (x : ℕ → ℝ), (∀ i, (∑ j in finset.range k, x ((i + j) % n)) = 0) → (∀ i, x i = 0)) ↔ (nat.gcd k n = 1) :=
begin
sorry
end

end gcd_k_n_condition_l658_658257


namespace plane_divides_segment_ratio_l658_658768

variables {A B C D K L M N F : Type*}
variables [field A] [field B] [field C] [field D]

-- Defining the midpoints
def midpoint (x y : Type*) : Type* := (x + y) / 2

-- Condition definitions
def K := midpoint A D
def L := midpoint B D
def M := midpoint A B
def N := midpoint C D

-- The ratio division problem statement
theorem plane_divides_segment_ratio (C K L M N : Type*) :
  ∃ F, F ∈ plane C K L ∧ divides_segment F M N (1:2) :=
sorry

end plane_divides_segment_ratio_l658_658768


namespace find_m_l658_658901

variables a b : ℝ × ℝ
variable m : ℝ

-- Given conditions
def vec_a := (1 : ℝ, Real.sqrt 3)
def vec_b := (m : ℝ, Real.sqrt 3)
def angle_between := Real.pi / 3

-- Magnitudes
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def magnitude_a := magnitude vec_a
def magnitude_b := magnitude vec_b

-- Dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def dot_product_ab := dot_product vec_a vec_b

-- Formula relating dot product and angle
def scalar_product : Prop := 
  dot_product_ab = magnitude_a * magnitude_b * Real.cos angle_between

-- The proof statement
theorem find_m : scalar_product → m = -1 :=
by
  -- We will fill this in the proof. Just hinting that it will be completed in the required step.
  sorry

end find_m_l658_658901


namespace sum_squares_projections_eq_half_na_squared_l658_658634

-- Define a regular n-gon in terms of its properties
def is_regular_ngon (n : ℕ) (vertices : ℕ → ℝ × ℝ) (a : ℝ) : Prop :=
  ∃ (A : fin n → ℝ × ℝ),
  (∀ i : fin n, dist (A i) (A ((i + 1) % n)) = a) ∧
  (∀ i j : fin n, i ≠ j → ∠ (A i) = ∠ (A j))

-- Define the projection of a vector on a line
def projection_length_squared (side : ℝ × ℝ) (l : ℝ × ℝ) : ℝ :=
  let u := side / ∥side∥ in
  (u.1 * l.1 + u.2 * l.2) ^ 2

-- The main theorem statement
theorem sum_squares_projections_eq_half_na_squared
  {n : ℕ} (h : 2 ≤ n) (a : ℝ) (vertices : ℕ → ℝ × ℝ)
  (h_reg : is_regular_ngon n vertices a) :
  ∀ (l : ℝ × ℝ), ∑ i in finset.range n, projection_length_squared (vertices i - vertices ((i + 1) % n)) l = (1 / 2) * n * a^2 :=
  sorry

end sum_squares_projections_eq_half_na_squared_l658_658634


namespace initial_bottle_count_l658_658254

variable (B: ℕ)

-- Conditions: Each bottle holds 15 stars, bought 3 more bottles, total 75 stars to fill
def bottle_capacity := 15
def additional_bottles := 3
def total_stars := 75

-- The main statement we want to prove
theorem initial_bottle_count (h : (B + additional_bottles) * bottle_capacity = total_stars) : 
    B = 2 :=
by sorry

end initial_bottle_count_l658_658254


namespace ellipse_properties_l658_658853

theorem ellipse_properties :
  ∀ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ x y : ℝ, (x, y) = (1, -3/2) →
    (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1) ∧
    (a^2 - b^2 = 1) →
    (((4,0)) ∈ ({ p : ℝ × ℝ | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}) ∧
    (∀ E F A1 A2 : ℝ × ℝ, (A1.1, A1.2) = (-2, 0) → (A2.1, A2.2) = (2, 0) →
    (E = (x_1, y_1)) → (F = (x_2, y_2)) → ∃ x : ℝ, x = 1)) :=
begin
  intros a b,
  sorry
end

end ellipse_properties_l658_658853


namespace triangle_properties_l658_658492

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨-4, 3⟩

-- Definitions for distance and area calculations
def distance (P Q : Point) : ℝ :=
  sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Area function using appropriate formula (absolute value of determinant)
def triangle_area (A B C : Point) : ℝ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

-- Check perpendicularity of lines
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def are_perpendicular (P Q R : Point) : Prop :=
  slope P Q * slope P R = -1

-- Midpoint of a line segment
def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Radius of circumcircle
def circum_radius (A B : Point) (M : Point) : ℝ :=
  distance A M

-- Standard circumcircle equation
def circumcircle_equation (O : Point) (r : ℝ) (x y : ℝ) : Prop :=
  (x - O.x)^2 + (y - O.y)^2 = r^2

theorem triangle_properties :
  triangle_area A B C = 20 ∧
  ∃ O : Point, ∃ r : ℝ, are_perpendicular A B C ∧ midpoint B C = O ∧ circum_radius A O = 2 * sqrt 5 ∧ circumcircle_equation O (2 * sqrt 5) x y :=
by
  sorry

end triangle_properties_l658_658492


namespace statements_incorrect_l658_658369

-- Definitions for vectors and their properties
variables {V : Type*} [inner_product_space ℝ V]

-- Defining the conditions for each statement
def condition_A (a b : V) : Prop := by
  exact (∥a∥ = ∥b∥ → (a = b ∨ a = -b))

def condition_B (A B C D : V) : Prop := by
  exact (collinear A B C D → on_the_same_line A B C D)

def condition_C (A B : V) : Prop := by
  exact (parallel (A - B) (B - A))

def condition_D (u v : V) : Prop := by
  exact (∥u∥ = 1 ∧ ∥v∥ = 1 → u = v)

-- Defining the incorrectness of statements A, B, and D
def incorrect_A (a b : V) : Prop := by
  exact ¬ condition_A a b

def incorrect_B (A B C D : V) : Prop := by
  exact ¬ condition_B A B C D

def incorrect_D (u v : V) : Prop := by
  exact ¬ condition_D u v

-- Main theorem stating that A, B, and D are incorrect
theorem statements_incorrect (A B : V) (C D : V) (u v : V) (a b : V) :
  incorrect_A a b ∧ incorrect_B A B C D ∧ incorrect_D u v := by
  sorry  

end statements_incorrect_l658_658369


namespace Trishul_invested_less_than_Raghu_l658_658343

-- Definitions based on conditions
def Raghu_investment : ℝ := 2500
def Total_investment : ℝ := 7225

def Vishal_invested_more_than_Trishul (T V : ℝ) : Prop :=
  V = 1.10 * T

noncomputable def percentage_decrease (original decrease : ℝ) : ℝ :=
  (decrease / original) * 100

theorem Trishul_invested_less_than_Raghu (T V : ℝ) 
  (h1 : Vishal_invested_more_than_Trishul T V)
  (h2 : T + V + Raghu_investment = Total_investment) :
  percentage_decrease Raghu_investment (Raghu_investment - T) = 10 := by
  sorry

end Trishul_invested_less_than_Raghu_l658_658343


namespace kenny_jumping_jacks_l658_658253

variable (S M T W X Sa : ℕ)
variable (T_last_week : ℕ := 324)
variable (T_this_week := S + M + T + W + X + Sa)

theorem kenny_jumping_jacks (hS : S = 34) (hM : M = 20) (hT : T = 0)
                           (hW : W = 123) (hX : X = 23) (hSa : Sa = 61) :
  let Th := T_last_week - (S + M + T + W + X + Sa) + 1 in
  Th ≤ 3 :=
by
  sorry

end kenny_jumping_jacks_l658_658253


namespace log_base_change_log_base_evaluation_l658_658076

-- Define the conditions as functions or constants used in the statement
theorem log_base_change 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  : log a (x) / log a (b) = log b (x) := sorry

theorem log_base_evaluation 
  : log 16 2 = 1 / 4 := by 
  have h16 : 16 = 2 ^ 4 := by norm_num
  have log_identity : log 16 2 = log (2 ^ 4) 2 := by rw h16
  have log_change : log (2 ^ 4) 2 = log 2 2 / log 2 (2 ^ 4) := log_base_change 2 2⁴ 2 (by norm_num) (by norm_num) (by norm_num)
  rw [log_change, log_self, log_pow] at log_identity
  exact log_identity

end log_base_change_log_base_evaluation_l658_658076


namespace roots_sum_prod_l658_658051

-- Given polynomial
def polynomial : Polynomial ℝ := Polynomial.C (-3) + (Polynomial.monomial 1 6) * Polynomial.X - Polynomial.X ^ 4

-- Roots definitions
def is_root (p : Polynomial ℝ) (x : ℝ) : Prop := p.eval x = 0

-- Given conditions
def roots : {c d : ℝ} × (is_root polynomial c) × (is_root polynomial d)

-- The proof statement
theorem roots_sum_prod : ∃ c d : ℝ, is_root polynomial c ∧ is_root polynomial d ∧ (c * d + c + d = 3 + Real.sqrt 2) := by
  sorry

end roots_sum_prod_l658_658051


namespace parabola_sum_l658_658092

-- Define the quadratic equation
noncomputable def quadratic_eq (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Given conditions
variables (a b c : ℝ)
variables (h1 : (∀ x y : ℝ, y = quadratic_eq a b c x → y = a * (x - 6)^2 - 2))
variables (h2 : quadratic_eq a b c 3 = 0)

-- Prove the sum a + b + c
theorem parabola_sum :
  a + b + c = 14 / 9 :=
sorry

end parabola_sum_l658_658092


namespace distance_example_max_m_distance_lt_2016_max_elements_in_T_l658_658981

-- Define sequences and their properties
def distance_seq (a b : list ℤ) : ℤ :=
  a.zip_with (λ x y, abs (x - y)) b |>.sum

-- Distance calculation for part (Ⅰ)
theorem distance_example :
  distance_seq [1, 4, 6, 7] [3, 4, 11, 8] = 8 := 
by
  sorry

-- Recursive sequence definition for part (Ⅱ)
def rec_seq (a₁ : ℤ) : ℕ → ℤ 
| 0     => a₁
| (n+1) => (1 + rec_seq n) / (1 - rec_seq n)

-- Distance calculation between sequences with given properties for part (Ⅱ)
theorem max_m_distance_lt_2016 :
  ∀ m : ℕ, m % 4 = 0 → let b := rec_seq 2 in let c := rec_seq 3 in m * distance_seq (list.map b (list.range m)) (list.map c (list.range m)) < 2016 → m ≤ 3024 := 
by
  sorry

-- Definition of set and distance condition for part (Ⅲ)
def seq_set : finset (vector ℕ 7) :=
  (finset.pi (finset.range 7) (λ _, finset.of_list [0, 1]))

def valid_subset (T : finset (vector ℕ 7)) : Prop :=
  ∀ x y ∈ T, x ≠ y → 3 ≤ finset.card (x.to_set.symm_diff y.to_set)

-- Bounding the size of valid subsets in part (Ⅲ)
theorem max_elements_in_T :
  ∀ T : finset (vector ℕ 7), valid_subset T → T.card ≤ 16 :=
by
  sorry

end distance_example_max_m_distance_lt_2016_max_elements_in_T_l658_658981


namespace length_of_CD_l658_658284

theorem length_of_CD (x y : ℝ) (h1 : x = (1/5) * (4 + y))
  (h2 : (x + 4) / y = 2 / 3) (h3 : 4 = 4) : x + y + 4 = 17.143 :=
sorry

end length_of_CD_l658_658284


namespace Mona_unique_players_l658_658613

theorem Mona_unique_players :
  ∀ (g : ℕ) (p : ℕ) (r1 : ℕ) (r2 : ℕ),
  g = 9 → p = 4 → r1 = 2 → r2 = 1 →
  (g * p) - (r1 + r2) = 33 :=
by {
  intros g p r1 r2 hg hp hr1 hr2,
  rw [hg, hp, hr1, hr2],
  norm_num,
  sorry -- skipping proof as per instructions
}

end Mona_unique_players_l658_658613


namespace smallest_period_intervals_of_monotonic_increase_find_lambda_l658_658882

-- Condition definitions
def f (x : Real) : Real := sin (5 * Real.pi / 6 - 2 * x)
                          - 2 * sin (x - Real.pi / 4) * cos (x + 3 * Real.pi / 4)

def F (x : Real) (λ : Real) : Real := -4 * λ * f(x) - cos (4 * x - Real.pi / 3)

-- Statement for the smallest positive period
theorem smallest_period : (∃ T > 0, ∀ x : Real, f(x + T) = f(x)) ∧ T = Real.pi := sorry

-- Statement for intervals of monotonic increase
theorem intervals_of_monotonic_increase :
  ∀ k : Int, ∀ x : Real, (x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) ∧ strict_mono_on f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) := sorry

-- Statement for finding lambda
theorem find_lambda (λ : Real) :
  (∀ x : Real, x ∈ Set.Icc (Real.pi / 12) (Real.pi / 3) → F(x, λ) ≥ -3 / 2) ∧ F(Real.pi / 4, λ) = -3 / 2 → λ = 1/4 := sorry

end smallest_period_intervals_of_monotonic_increase_find_lambda_l658_658882


namespace area_of_trapezoid_ABCD_l658_658594

-- Define the entities in the problem
variables (A B C D O : Type) [IsoscelesTrapezoid A B C D] 

-- Given conditions
axiom condition1 : LargerBase A B
axiom condition2 : DiagonalsIntersectAt O
axiom condition3 : RatioOA_OC O A C 2
axiom condition4 : AreaBOC O B C 10

-- Defining the hypothesis
theorem area_of_trapezoid_ABCD : AreaABCD A B C D = 45 :=
sorry

end area_of_trapezoid_ABCD_l658_658594


namespace sum_first_n_terms_l658_658874

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (a_3 a_4 a_5 : ℝ)

-- Conditions
def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) := ∀ n, a_n (n + 1) = a_n n * q
def condition_1 : Prop := geometric_sequence a_n q
def condition_2 : Prop := q > 1
def condition_3 : Prop := a_n 3 + a_n 5 = 20
def condition_4 : Prop := a_n 4 = 8

-- Statement to be proved
theorem sum_first_n_terms : condition_1 a_n q → condition_2 q → condition_3 (a_n 3) (a_n 5) → condition_4 (a_n 4) → (∑ i in Finset.range n, a_n i) = 2^n - 1 := by
  sorry

end sum_first_n_terms_l658_658874


namespace andrew_workday_length_l658_658423

-- Define conditions as hypotheses
variables (appointments : ℕ) (hours_per_appointment : ℕ) (permits_per_hour : ℕ) (total_permits : ℕ)

-- Given conditions
def conditions : Prop :=
  appointments = 2 ∧
  hours_per_appointment = 3 ∧
  permits_per_hour = 50 ∧
  total_permits = 100

-- Define the workday length function based on the conditions
noncomputable def workday_length (appointments hours_per_appointment permits_per_hour total_permits : ℕ) : ℕ :=
  (appointments * hours_per_appointment) + (total_permits / permits_per_hour)

-- The proof
theorem andrew_workday_length : 
  conditions appointments hours_per_appointment permits_per_hour total_permits → 
  workday_length appointments hours_per_appointment permits_per_hour total_permits = 8 :=
by
  intro h
  cases h with ha h1
  cases h1 with hb h2
  cases h2 with hc hd
  unfold workday_length
  rw [ha, hb, hc, hd]
  norm_num
  sorry -- skipping actual proof steps here

-- Provide specific values (example instantiation for conditions)
lemma specific_andrew_workday_length : 
  workday_length 2 3 50 100 = 8 :=
by
  unfold workday_length
  norm_num
  sorry -- skipping actual proof steps here

end andrew_workday_length_l658_658423


namespace equal_area_triangles_l658_658206

noncomputable def length_of_df (AB AC DE : ℝ) (h : ∀ x y : ℝ, x * y = 5.1 * 5.1) : ℝ :=
classical.some (classical.some_spec h DE)

theorem equal_area_triangles (AB AC DE DF : ℝ) (h : ∀ x y : ℝ, x * y = 5.1 * 5.1) :
  AB = 5.1 → AC = 5.1 → DE = 1.7 → DF = 15.3 :=
by
  sorry

end equal_area_triangles_l658_658206


namespace num_ways_to_write_3050_l658_658259

theorem num_ways_to_write_3050 : 
  let N := (∑ a3 in {0, 1, 2}, 100) + 6 in N = 306 :=
by
  -- Definitions
  let a_0 := 0
  let a_2 := 0
  let a_3 := 0
  let a_1 := set.Icc 0 99

  -- Additional conditions and requirements
  have h_a0 : 0 ≤ a_0 ∧ a_0 ≤ 99 := by sorry
  have h_a1 : ∀ x ∈ a_1, 0 ≤ x ∧ x ≤ 99 := by sorry

  -- Needed values
  let valid_a1_count := 100
  
  let N := ((3 : nat) * valid_a1_count) + 6
  
  -- Proof
  have h_a3 : ∀ a, a ∈ {0,1,2,3} → ((a = 3 → valid_a1_count = 6) ∧ (0 ≤ a ∧ a ≤ 3)) := by sorry

  -- Summation
  let sum1 := ((3 : nat) * valid_a1_count)
  
  -- Combine results
  rw [sum1, add_comm, ←add_assoc]
  norm_num
  exact N

#print num_ways_to_write_3050

end num_ways_to_write_3050_l658_658259


namespace eval_expression_l658_658447

def base : ℝ := 64
def exponent : ℝ := -2 ^ -3
def expected_result : ℝ := 1 / 2 ^ 0.75

theorem eval_expression (e1 : ℝ) (e2 : ℝ) (e_res : ℝ) 
  (h_base : e1 = 64) 
  (h_exp : e2 = -2 ^ -3) 
  (h_expected_res : e_res = 1 / 2 ^ 0.75) : 
  e1 ^ e2 = e_res := 
by 
  rw [h_base, h_exp, h_expected_res]
  sorry

end eval_expression_l658_658447


namespace area_new_rectangle_greater_than_square_l658_658012

theorem area_new_rectangle_greater_than_square (a b : ℝ) (h : a > b) : 
  (2 * (a + b) * (2 * b + a) / 3) > ((a + b) * (a + b)) := 
sorry

end area_new_rectangle_greater_than_square_l658_658012


namespace prime_sum_eq_14_l658_658509

theorem prime_sum_eq_14 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 := 
sorry

end prime_sum_eq_14_l658_658509


namespace magnitude_of_z_l658_658182

noncomputable def z : ℂ := (1 - complex.I) / (1 + complex.I) + 4 - 2 * complex.I

theorem magnitude_of_z :
  complex.abs z = 5 :=
sorry

end magnitude_of_z_l658_658182


namespace students_in_line_l658_658365

theorem students_in_line (y e b between : ℕ) (hy : y = 1) (he : e = 1) (hb : b = 4) (hbetween : between = 30) : y + e + b + between + 1 = 36 :=
by {
  rw [hy, he, hb, hbetween],
  norm_num,
  sorry,
}

end students_in_line_l658_658365


namespace maximize_intersection_area_when_common_center_of_symmetry_l658_658381

def strip (l1 l2 : set (ℝ × ℝ)) : set (ℝ × ℝ) := 
  { p : ℝ × ℝ | p ∈ l1 ∧ p ∉ l2 ∨ p ∉ l1 ∧ p ∈ l2 }

noncomputable def common_center_of_symmetry (strips : list (set (ℝ × ℝ))) (center : ℝ × ℝ) : Prop :=
  ∀ s ∈ strips, ∃ (shift : ℝ × ℝ → ℝ × ℝ), 
  (∀ (p : ℝ × ℝ), p ∈ s ↔ shift p ∈ s) ∧ 
  (∃ (sym_point : ℝ × ℝ), sym_point = center ∧ (shift sym_point = sym_point))

theorem maximize_intersection_area_when_common_center_of_symmetry 
  (strips : list (set (ℝ × ℝ)))
  (h1 : ∀ (s t : set (ℝ × ℝ)), s ∈ strips → t ∈ strips → s ≠ t → ∀ (p : ℝ × ℝ), p ∈ s → p ∉ t)
  : ∃ center, common_center_of_symmetry strips center :=
sorry

end maximize_intersection_area_when_common_center_of_symmetry_l658_658381


namespace matchsticks_20th_stage_l658_658303

theorem matchsticks_20th_stage 
  (a₁ : ℕ)
  (d : ℕ)
  (h_a₁ : a₁ = 4)
  (h_d : d = 3) :
  (∑ i in finset.range 20, d) + a₁ - d = 61 :=
by
  sorry

end matchsticks_20th_stage_l658_658303


namespace solve_for_x_l658_658524

variable (x : ℤ)

theorem solve_for_x: -2 * x - 7 = 7 * x + 2 → x = -1 := by
  intro h
  have h1 : -2 * x - 7 + 2 * x = 7 * x + 2 + 2 * x := by linarith
  have h2 : -7 = 9 * x + 2 := by linarith
  have h3 : -7 - 2 = 9 * x + 2 - 2 := by linarith
  have h4 : -9 = 9 * x := by linarith
  have h5 : -9 / 9 = 9 * x / 9 := by linarith
  have h6 : -1 = x := by linarith
  exact eq.symm h6

end solve_for_x_l658_658524


namespace emilia_should_buy_more_l658_658071

-- Define the variables and conditions
variables (total_needed strawberries blueberries: ℕ)
-- Define the conditions
def needs_42_cartons : Prop := total_needed = 42
def has_strawberries : Prop := strawberries = 2
def has_blueberries  : Prop := blueberries = 7

-- Define the total cartons Emilia already has
def total_cartons_already_has : ℕ := strawberries + blueberries

-- Define the number of additional cartons she needs to buy
def additional_cartons_needed : ℕ := total_needed - total_cartons_already_has

-- Prove that the number of additional cartons needed is 33
theorem emilia_should_buy_more : 
  needs_42_cartons → has_strawberries → has_blueberries → additional_cartons_needed = 33 :=
by
  intros h1 h2 h3
  rw [needs_42_cartons, has_strawberries, has_blueberries] at *
  simp [total_cartons_already_has, additional_cartons_needed]
  sorry

end emilia_should_buy_more_l658_658071


namespace num_lighting_methods_l658_658929

-- Definitions of the problem's conditions
def total_lights : ℕ := 15
def lights_off : ℕ := 6
def lights_on : ℕ := total_lights - lights_off
def available_spaces : ℕ := lights_on - 1

-- Statement of the mathematically equivalent proof problem
theorem num_lighting_methods : Nat.choose available_spaces lights_off = 28 := by
  sorry

end num_lighting_methods_l658_658929


namespace perpendicular_edges_count_eq_eight_l658_658549

noncomputable def number_of_perpendicular_edges_in_cube (A A₁ : ℝ) [cube : Cube (A A₁)] : ℕ :=
  8

theorem perpendicular_edges_count_eq_eight {A A₁ : ℝ} [cube : Cube (A A₁)] :
  number_of_perpendicular_edges_in_cube A A₁ = 8 :=
by
  exact number_of_perpendicular_edges_in_cube A A₁

end perpendicular_edges_count_eq_eight_l658_658549


namespace intersection_A_B_l658_658480

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B_l658_658480


namespace f_half_lt_3_l658_658865

variable {R : Type*} [OrderedRing R]

def arithmetic_sequence (a d : R) (n : ℕ) : R := a + (n - 1) * d

def f (a d : R) (n : ℕ) (x : R) : R :=
  Finset.sum (Finset.range n) (λ k, (arithmetic_sequence a d (k+1)) * x^(k+1))

theorem f_half_lt_3 (a d : R) (n : ℕ) (h1 : 0 < n ∧ 2 ∣ n) (h2 : f a d n 1 = n^2) (h3 : f a d n (-1) = n) :
  f a d n (1 / 2) < 3 :=
sorry

end f_half_lt_3_l658_658865


namespace number_of_subsets_l658_658106

theorem number_of_subsets (a : ℝ) : ∃ M : set ℝ, (M = {x | x ^ 2 - 3 * x - a ^ 2 + 2 = 0}) ∧ (2 ^ (M.card) = 4) :=
by
  sorry

end number_of_subsets_l658_658106


namespace time_to_cross_bridge_l658_658022

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 150
def speed_in_kmhr : ℕ := 72
def speed_in_ms : ℕ := (speed_in_kmhr * 1000) / 3600

theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_in_ms = 20 :=
by
  have total_distance := length_of_train + length_of_bridge
  have speed := speed_in_ms
  sorry

end time_to_cross_bridge_l658_658022


namespace no_integer_area_of_prime_sided_triangle_l658_658308

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_integer_area_of_prime_sided_triangle (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) : 
  ¬ ∃ (S : ℤ), area a b c = S :=
sorry

end no_integer_area_of_prime_sided_triangle_l658_658308


namespace lines_concurrent_l658_658964

noncomputable def circle (O : Point) (r : ℝ) := { P : Point | dist O P = r }

variables {O1 O2 K L M N : Point}
variables {r1 r2 d : ℝ}
variables (Γ1 Γ2 : set Point) 

-- Conditions
axiom circle1 : Γ1 = circle O1 r1
axiom circle2 : Γ2 = circle O2 r2
axiom distance_O1_O2 : dist O1 O2 = d
axiom sum_radii_less_than_distance : r1 + r2 < d
axiom tangent_contact_external : tangent_contact K L Γ1 Γ2 True  -- true for external
axiom tangent_contact_internal : tangent_contact M N Γ1 Γ2 False -- false for internal

-- Theorem Statement
theorem lines_concurrent : are_concurrent (KM : Line) (NL : Line) (O1O2 : Line) := sorry

end lines_concurrent_l658_658964


namespace pasha_can_ensure_chip_reaches_last_cell_l658_658265

theorem pasha_can_ensure_chip_reaches_last_cell (k n : ℕ) (hk : k = 4) (hn : n = 3) : 
  ∃ (m : ℕ), m = n ∧ -- there is a cell number m which equals n
  ∃ (x₀ : ℕ), x₀ = k ∧ -- initially, there are k chips
  ∃ y₀ z₀ y₁ z₁, -- intermediate states for chips in rows
  y₀ + z₀ = k ∧ -- initial chips in both rows sum to k
  (y₀ - 2 ≥ 0) ∧ (z₀ - 2 ≥ 0) ∧ -- Pasha moves 2 chips from both rows to second cell
  y₁ = (y₀ - 2) ∧ z₁ = (z₀ - 2) ∧ -- state after Pasha's move
  y₁ = z₁ ∧ -- Roma removes equal chips from the same cell
  ∃ (final_y final_z : ℕ), 
  final_y = (y₁ - 1) ∧ final_z = (z₁ - 1) ∧ -- Roma's move condition
  (final_y ≥ 0) ∧ (final_z ≥ 0) ∧ 
  (final_y + 1 ≥ 1 ∨ final_z + 1 ≥ 1). -- ensuring the last cell gets a chip
  sorry

end pasha_can_ensure_chip_reaches_last_cell_l658_658265


namespace problem_1_problem_2_problem_3_l658_658126

open Function

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y)
axiom h3 : f(4) = 1

theorem problem_1 : f(1) = 0 := sorry

theorem problem_2 : ∃ m : ℝ, f(m) = 2 ∧ m = 16 := sorry

theorem problem_3 : ∀ x : ℝ, f(x^2 - 4 * x - 5) < 2 ↔ (-3 < x ∧ x < -1) ∨ (5 < x ∧ x < 7) := sorry

end problem_1_problem_2_problem_3_l658_658126


namespace log_tan_eq_l658_658843

variable (b x a : ℝ)
variable (h1 : b > 1)
variable (h2 : sin x > 0)
variable (h3 : cos x > 0)
variable (h4 : log b (sin x) = a)

theorem log_tan_eq : log b (tan x) = a - (1 / 2) * log b (1 - b ^ (2 * a)) := by
  sorry

end log_tan_eq_l658_658843


namespace frustum_lateral_surface_area_l658_658762

theorem frustum_lateral_surface_area:
  ∀ (R r h : ℝ), R = 7 → r = 4 → h = 6 → (∃ L, L = 33 * Real.pi * Real.sqrt 5) := by
  sorry

end frustum_lateral_surface_area_l658_658762


namespace planB_lower_avg_price_planD_higher_final_price_l658_658556

variables (a b m n : ℝ) (h_ab : a ≠ b)
variables (p q : ℝ) (h_pq : p ≠ q)

-- Problem (1): Prove that Plan B has a lower average price than Plan A
theorem planB_lower_avg_price :
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by {
  have h1 : (a + b) / 2 - (2 * a * b) / (a + b) = (a - b)^2 / (2 * (a + b)),
  calc
    (a + b) / 2 - (2 * a * b) / (a + b) = ((a + b)^2 - 4 * a * b) / (2 * (a + b)) : by sorry,
  exact_mod_cast sorry,
  have h2 : (a - b)^2 > 0,
  exact_mod_cast sorry,
  linarith
}

-- Problem (2): Prove that Plan D results in a higher price after the increases than Plan C
theorem planD_higher_final_price :
  100 * (1 + (p + q) / 2)^2 > 100 * (1 + p) * (1 + q) :=
by {
  have h1 : 100 * (1 + (p + q) / 2)^2 - 100 * (1 + p) * (1 + q) = 100 * (p - q)^2 / 4,
  calc
    100 * (1 + (p + q) / 2)^2 - 100 * (1 + p) * (1 + q) = 100 * (((p - q)^2) / 4) : by sorry,
  exact_mod_cast sorry,
  have h2 : (p - q)^2 > 0,
  exact_mod_cast sorry,
  linarith
}

end planB_lower_avg_price_planD_higher_final_price_l658_658556


namespace triangle_incircle_relation_l658_658554

theorem triangle_incircle_relation
  (A B C I D : Type)
  [triABC : Triangle A B C]
  [incenterI : Incenter I A B C]
  [pointD : Point D]
  [D_intersection : IntersectionPoint D (segment A I) (line B C)] :
  AI + CD = AC ↔ ∠B = 60° + (1 / 3) * ∠C :=
sorry

end triangle_incircle_relation_l658_658554


namespace find_f_neg2_l658_658813

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem find_f_neg2 : f (-2) = 15 :=
by
  sorry

end find_f_neg2_l658_658813


namespace H_range_l658_658359

noncomputable def H (x : ℝ) : ℝ :=
  if x < -2 then -4
  else if x < 2 then 2 * x
  else 4

theorem H_range : Set.range H = set.Icc (-4) 4 :=
sorry

end H_range_l658_658359


namespace sum_of_arithmetic_sequence_l658_658277

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7/60) :
  S 9 = 63/2 :=
by
  sorry

end sum_of_arithmetic_sequence_l658_658277


namespace coupon_percentage_correct_l658_658316

variable (original_price increased_price final_price : ℝ)
variable (percentage_off : ℝ)

axiom original_price_value : original_price = 150
axiom increased_price_percentage : increased_price = original_price * 1.2
axiom final_price_value : final_price = 144

theorem coupon_percentage_correct :
  (original_price * 1.2 * (1 - percentage_off / 100)) = final_price →
  percentage_off = 20 :=
by
  intros h

  have h₁ : increased_price = 180 :=
    calc 
      increased_price = original_price * 1.2 : by rw [increased_price_percentage]
      ... = 150 * 1.2 : by rw [original_price_value]
      ... = 180 : by norm_num

  have h₂ : final_price = 180 * (1 - percentage_off / 100) :=
    by rw [h₁, h]

  have h₃ : 180 * (1 - percentage_off / 100) = 144 :=
    by rw [final_price_value, h₂]

  have h₄ : 1 - percentage_off / 100 = 0.8 :=
    by norm_num [←h₃]

  have h₅ : percentage_off / 100 = 0.2 :=
    by linarith

  have h₆ : percentage_off = 20 :=
    by linarith

  exact h₆


end coupon_percentage_correct_l658_658316


namespace a_n_expression_l658_658218

noncomputable def a_n : ℕ → ℕ
| 1       := 0
| (n + 1) := a_n n + 6 * n - 6

noncomputable def b_n : ℕ → ℕ
| 1       := 0
| (n + 1) := b_n n + 6

theorem a_n_expression (n : ℕ) (h : n > 0) : 
  a_n n = 3 * n^2 - 9 * n + 6 :=
sorry

end a_n_expression_l658_658218


namespace function_has_local_minimum_at_zero_l658_658457

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * (x - 1))

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y

theorem function_has_local_minimum_at_zero :
  -4 < 0 ∧ 0 < 1 ∧ is_local_minimum f 0 := 
sorry

end function_has_local_minimum_at_zero_l658_658457


namespace regression_equation_y_compare_models_min_broadcast_cycles_l658_658984

theorem regression_equation_y (x : Fin 6 → ℝ) (y : Fin 6 → ℝ) (z : Fin 6 → ℝ)
  (sum_x : ∀ i : Fin 6, x i = i + 1)
  (sum_y : y = ![3, 7, 15, 30, 40])
  (log2_y_to_z : ∀ i : Fin 6, z i = logBase 2 (y i))
  (average_z : (∑ i, z i) / 5 = 3.7)
  (sum_x_squared : (∑ i, (x i)^2) = 55)
  (sum_x_y : (∑ i, (x i) * (y i)) = 382)
  (sum_x_z : (∑ i, (x i) * (z i)) = 65)
  (sum_squared_diff_y : (∑ i, (y i - 19)^2) = 978)
  (sum_squared_diff_hat_y : (∑ i, (y i - (2^((0.95 * (x i)) + 0.85)))^2) = 101) :
  ∃ a b, ∀ x : ℝ, y = 2^(b * x + a) :=
sorry

theorem compare_models (ra_squared : ℝ) (rb_squared : ℝ)
  (ra_squared_val : ra_squared = 0.90)
  (rb_squared_val : rb_squared = 0.98) :
  rb_squared > ra_squared :=
sorry

theorem min_broadcast_cycles (sales_bound : ℝ)
  (model_b : ∀ x : ℝ, sales_bound = 9.7 * x - 10.1)
  (sales_target : ℝ)
  (sales_target_val : sales_target = 80) :
  ∃ n : ℕ, n ≥ 10 :=
sorry

end regression_equation_y_compare_models_min_broadcast_cycles_l658_658984


namespace sum_345_consecutive_sequences_l658_658214

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l658_658214


namespace find_points_number_l658_658091

noncomputable def find_points
  (a : EuclideanGeometry.Line)
  (A : EuclideanGeometry.Point)
  (s : EuclideanGeometry.Plane)
  (m n p : ℝ) : ℕ := 8

theorem find_points_number
  (a : EuclideanGeometry.Line)
  (A : EuclideanGeometry.Point)
  (s : EuclideanGeometry.Plane)
  (m n p : ℝ) :
  find_points a A s m n p = 8 :=
by sorry

end find_points_number_l658_658091


namespace tetrahedron_circumscribed_sphere_diameter_l658_658406

noncomputable def median_length (a b c : ℝ) : ℝ :=
  real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

-- Conditions: the side lengths of the triangle.
def side_a : ℝ := 10
def side_b : ℝ := 12
def side_c : ℝ := 14

-- Calculate medians of the triangle with given sides.
def median_a : ℝ := median_length side_a side_b side_c
def median_b : ℝ := median_length side_b side_c side_a
def median_c : ℝ := median_length side_c side_a side_b

-- Define the lengths of the segments formed after folding.
def AB : ℝ := 5
def CD : ℝ := 5
def AC : ℝ := 7
def BD : ℝ := 7
def AD : ℝ := 6
def BC : ℝ := 6

-- Deduce the circumradius
def diameter_of_circumscribed_sphere : ℝ :=
  let d := real.sqrt 19 in
  2 * real.sqrt ((d / 2)^2 + (AD / 2)^2)

theorem tetrahedron_circumscribed_sphere_diameter :
  diameter_of_circumscribed_sphere = real.sqrt 55 :=
sorry

end tetrahedron_circumscribed_sphere_diameter_l658_658406


namespace apples_vs_cherries_l658_658658

def pies_per_day : Nat := 12
def apple_days_per_week : Nat := 3
def cherry_days_per_week : Nat := 2

theorem apples_vs_cherries :
  (apple_days_per_week * pies_per_day) - (cherry_days_per_week * pies_per_day) = 12 := by
  sorry

end apples_vs_cherries_l658_658658


namespace map_distance_l658_658372

theorem map_distance
  (s d_m : ℝ) (d_r : ℝ)
  (h1 : s = 0.4)
  (h2 : d_r = 5.3)
  (h3 : d_m = 64) :
  (d_m * d_r / s) = 848 := by
  sorry

end map_distance_l658_658372


namespace pascal_triangle_row_20_element_5_l658_658723

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l658_658723


namespace log_16_2_eq_one_fourth_l658_658073

theorem log_16_2_eq_one_fourth (a b c : ℝ) (h1 : a = 2^4) (h2 : b = log 2 2) (h3 : c = log 2 (2^4)) : 
  log 16 2 = 1 / 4 := 
by 
  sorry

end log_16_2_eq_one_fourth_l658_658073


namespace quadratic_two_roots_l658_658682

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, (x = x₁ ∨ x = x₂) ↔ (x^2 + b*x - 3 = 0)) :=
by
  -- Indicate that a proof is required here
  sorry

end quadratic_two_roots_l658_658682


namespace four_digit_number_proof_l658_658441

def four_digit_number_div (n m k : ℕ) : Prop :=
  n / m = k

def digit_product_condition (d : ℕ) (a b c : ℕ) :=
  100 * d + 10 * b + a

def valid_digits (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ d ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

noncomputable def find_four_digit_number : ℕ :=
  let ABCD := 4000 + 900 + 10 + 6
  ABCD

theorem four_digit_number_proof : 
  ∃ (ABCD : ℕ), valid_digits 4 9 1 6 ∧ 
  four_digit_number_div ABCD 4 (digit_product_condition 4 9 1) :=
begin
  use 4916,
  split,
  { unfold valid_digits,
    exact ⟨dec_trivial, dec_trivial, dec_trivial, dec_trivial, dec_trivial, dec_trivial, dec_trivial, dec_trivial⟩ },
  { unfold four_digit_number_div,
    unfold digit_product_condition,
    exact dec_trivial }
end

end four_digit_number_proof_l658_658441


namespace geometry_problem_l658_658211

open EuclideanGeometry

variables {A B C D E F M : Point}
variables [hcA : AcuteTriangle A B C]
variables [external_angle_bisector : ExternalAngleBisector A B C D]
variables [midpoint_M : Midpoint M B C]
variables [perp_ME_AD : Perp M E (Line A D)]
variables [perp_MF_BC : Perp M F (Line B C)]

theorem geometry_problem
  (h1 : is_acute_triangle A B C)
  (h2 : external_angle_bisector A B C D)
  (h3 : midpoint M B C)
  (h4 : perpendicular M E (line A D))
  (h5 : perpendicular M F (line B C)) :
  Length B C ^ 2 = 4 * (Length A E) * (Length D F) :=
sorry

end geometry_problem_l658_658211


namespace first_player_can_win_if_m_gt_2n_first_player_can_win_if_m_gt_alpha_n_l658_658692

-- Definitions and theorems necessary for the proof
def matches_property (m n : ℕ) := m > n ∧ ∀ k < n, exists l < m, l = k * n

theorem first_player_can_win_if_m_gt_2n (m n : ℕ) (h₁ : m > n) (h₂ : m > 2 * n) : 
  (∃ f : ℕ → ℕ, f 0 = m ∧ 
    (∀ i, matches_property (f i) n → matches_property (f (i + 1)) n) ∧ 
    (∃ i, f i = 0)) :=
sorry

noncomputable def alpha := (1 + Real.sqrt 5) / 2

theorem first_player_can_win_if_m_gt_alpha_n (m n : ℕ) (h₁ : m > n) (h₂ : (m : ℝ) > alpha * n) :
  (∃ f : ℕ → ℕ, f 0 = m ∧ 
    (∀ i, matches_property (f i) n → matches_property (f (i + 1)) n) ∧ 
    (∃ i, f i = 0)) :=
sorry

end first_player_can_win_if_m_gt_2n_first_player_can_win_if_m_gt_alpha_n_l658_658692


namespace pascal_triangle_row_20_element_5_l658_658705

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l658_658705


namespace simplify_rationalize_expr_l658_658640

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l658_658640


namespace trader_profit_l658_658775

theorem trader_profit
  (CP : ℝ)
  (MP : ℝ)
  (SP : ℝ)
  (h1 : MP = CP * 1.12)
  (discount_percent : ℝ)
  (h2 : discount_percent = 0.09821428571428571)
  (discount : ℝ)
  (h3 : discount = MP * discount_percent)
  (actual_SP : ℝ)
  (h4 : actual_SP = MP - discount)
  (h5 : CP = 100) :
  (actual_SP / CP = 1.01) :=
by
  sorry

end trader_profit_l658_658775


namespace log_sqrt5_plus_4_neg_half_eq_one_l658_658046

theorem log_sqrt5_plus_4_neg_half_eq_one :
  (\log (5 : ℝ) (sqrt 5) + 4 ^ (- (1 : ℝ) / 2)) = 1 :=
by
  -- Use Lean definitions of logarithmic and exponential properties
  have h1: log 5 (sqrt 5) = (1 : ℝ) / 2, from sorry,
  have h2: (4 : ℝ) ^ (- (1 : ℝ)/2) = 1 / 2, from sorry,
  -- Add the two simplified parts
  rw [h1, h2],
  norm_num

end log_sqrt5_plus_4_neg_half_eq_one_l658_658046


namespace no_solutions_to_equation_l658_658313

theorem no_solutions_to_equation :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ^ 2 - 2 * y ^ 2 = 5 := by
  sorry

end no_solutions_to_equation_l658_658313


namespace orthocenter_altitudes_ratio_two_l658_658156

open Real EuclideanGeometry

theorem orthocenter_altitudes_ratio_two (A B C H P Q R : Point) :
  is_orthocenter H A B C → (altitude A B C).meets_\( A \) (altitude_foot A B C) P 
  → (altitude B A C).meets_\( B \) (altitude_foot B A C) Q →
  (altitude C A B).meets_\( C \) (altitude_foot C A B) R →
  (\frac{AH}{AP}) + (\frac{BH}{BQ}) + (\frac{CH}{CR}) = 2 :=
by
  sorry

end orthocenter_altitudes_ratio_two_l658_658156


namespace distance_to_fourth_buoy_l658_658428

theorem distance_to_fourth_buoy
  (buoy_interval_distance : ℕ)
  (total_distance_to_third_buoy : ℕ)
  (h : total_distance_to_third_buoy = buoy_interval_distance * 3) :
  (buoy_interval_distance * 4 = 96) :=
by
  sorry

end distance_to_fourth_buoy_l658_658428


namespace incorrect_propositions_l658_658592

variables (m n : line)
variables (α β : plane)

def proposition1 := m ⟂ α ∧ n ∈ β ∧ m ⟂ n → α ⟂ β
def proposition3 := α ⟂ β ∧ m ⟂ α ∧ n ∥ β → m ⟂ n
def proposition4 := α ⟂ β ∧ α ∩ β = m ∧ n ⟂ m → n ⟂ β

theorem incorrect_propositions :
  ¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4 :=
by sorry

end incorrect_propositions_l658_658592


namespace no_hats_left_probability_l658_658066

noncomputable def harmonic (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.range n).sum (λ i, 1 / (i + 1 : ℚ))

noncomputable def pn (n : ℕ) : ℚ :=
  if n = 0 then 1 else (harmonic n * pn (n - 1)) / nat.factorial n

theorem no_hats_left_probability :
  pn 10 ≈ 0.000516 := 
sorry

end no_hats_left_probability_l658_658066


namespace dvds_still_fit_in_book_l658_658383

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end dvds_still_fit_in_book_l658_658383


namespace fifth_element_row_20_pascal_triangle_l658_658719

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l658_658719


namespace profit_share_difference_l658_658298

theorem profit_share_difference :
  let 
    suresh_investment := 18000 * 12,
    rohan_investment := 12000 * 9,
    sudhir_investment := 9000 * 8,
    ankit_investment := 15000 * 6,
    deepak_investment := 10000 * 4,
    total_investment := suresh_investment + rohan_investment + sudhir_investment + ankit_investment + deepak_investment,
    total_profit := 5680
  in
    abs (((rohan_investment * total_profit) / total_investment) - ((sudhir_investment * total_profit) / total_investment)) = 388.97 := 
sorry

end profit_share_difference_l658_658298


namespace n_four_plus_n_squared_plus_one_not_prime_l658_658272

theorem n_four_plus_n_squared_plus_one_not_prime (n : ℤ) (h : n ≥ 2) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end n_four_plus_n_squared_plus_one_not_prime_l658_658272


namespace johns_total_payment_l658_658956

theorem johns_total_payment
  (packs_of_gum : ℕ := 5)
  (candy_bars : ℕ := 4)
  (bags_of_chips : ℕ := 2)
  (cost_candy_bar : ℝ := 1.50)
  (discount : ℝ := 0.10) :
  let cost_gum := cost_candy_bar / 2,
      cost_chip := cost_candy_bar * 2,
      total_cost_gum := packs_of_gum * cost_gum,
      total_cost_candy_bars := candy_bars * cost_candy_bar,
      total_cost_chips := bags_of_chips * cost_chip,
      total_cost_before_discount := total_cost_gum + total_cost_candy_bars + total_cost_chips,
      discount_amount := total_cost_before_discount * discount,
      discount_rounded := (Real.round (discount_amount * 100)) / 100,
      total_cost_after_discount := total_cost_before_discount - discount_rounded
  in total_cost_after_discount = 14.17 :=
by sorry

end johns_total_payment_l658_658956


namespace total_votes_is_240_l658_658399

variable {x : ℕ} -- Total number of votes (natural number)
variable {S : ℤ} -- Score (integer)

-- Given conditions
axiom score_condition : S = 120
axiom votes_condition : 3 * x / 4 - x / 4 = S

theorem total_votes_is_240 : x = 240 :=
by
  -- Proof should go here
  sorry

end total_votes_is_240_l658_658399


namespace a_general_formula_sum_b_first_10_terms_l658_658875

-- Definitions of the sequences
def S (n : ℕ) : ℕ := sorry
def a : ℕ → ℕ
| 1 := 2
| (n+2) := 3 * 2^n
-- Define the dependent sequence b
def b : ℕ → ℚ
| n := if n % 2 = 1 then a n else (n+1 : ℚ)/(n-1 : ℚ) + (n-1 : ℚ)/(n+1 : ℚ)

-- Given Conditions
axiom S_n_eq_a_n_plus_1_minus_1 (n : ℕ) : S n = a (n + 1) - 1

-- Proving the general form of a_n
theorem a_general_formula (n : ℕ) : a n = if n = 1 then 2 else 3 * 2 ^ (n - 2) := sorry

-- Proving the sum of the first 10 terms of b_n
theorem sum_b_first_10_terms : (Finset.range 10).sum b = (5762 : ℚ) / 11 := sorry

end a_general_formula_sum_b_first_10_terms_l658_658875


namespace smallest_integer_n_l658_658971

theorem smallest_integer_n (m n : ℕ) (r : ℝ) (h1 : 0 < n)
  (h2 : 0 < r) (h3 : r < 1/500) (h4 : m = (n + r)^3)
  (h5 : ∀ k : ℕ, k < n → ∃ r' : ℝ, (0 < r' ∧ r' < 1/500) ∧ m ≠ (k + r')^3) :
  n = 13 :=
sorry

end smallest_integer_n_l658_658971


namespace revenue_increase_by_2_point_6_percent_l658_658814

-- Definitions for the given problem conditions
variable (P V: Real) 

def P_new := 1.5 * P
def V_new1 := 0.8 * V
def P_seasonal := 1.35 * P
def V_final := 0.76 * V

-- Proving the final revenue based on given conditions
theorem revenue_increase_by_2_point_6_percent (P V: Real) : 
  let R_initial := P * V
  let R_final := P_seasonal * V_final
  R_final = 1.026 * R_initial := by
  -- Initial Revenue
  let R_initial := P * V
  -- New price after 50% increase
  let P_new := 1.5 * P
  -- New sales volume after 20% decrease
  let V_new1 := 0.8 * V
  -- Price after 10% seasonal discount
  let P_seasonal := 1.35 * P
  -- Final sales volume after additional 5% decrease
  let V_final := 0.76 * V
  -- Final Revenue computation
  let R_final := P_seasonal * V_final
  show R_final = 1.026 * R_initial, by sorry

end revenue_increase_by_2_point_6_percent_l658_658814


namespace sqrt_sequence_identity_sqrt_sequence_conjecture_l658_658505

noncomputable def sequence_2 (x : ℝ) : ℝ :=
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 4 * sqrt (1 + 5 * sqrt (1 + x)))))

noncomputable def sequence_3 (x : ℝ) : ℝ :=
  sqrt (1 + 3 * sqrt (1 + 4 * sqrt (1 + 5 * sqrt (1 + 6 * sqrt (1 + x)))))

noncomputable def sequence_n (n : ℕ) (x : ℝ) : ℝ :=
  sqrt (1 + n * sqrt (1 + (n + 1) * sqrt (1 + (n + 2) * sqrt (1 + (n + 3) * sqrt (1 + x)))))

theorem sqrt_sequence_identity :
  (sequence_2 0 = 3) →
  (sequence_3 0 = 4) :=
by
  intros h1
  sorry

theorem sqrt_sequence_conjecture (n : ℕ) :
  (sequence_n n 0 = n + 1) :=
by
  sorry

end sqrt_sequence_identity_sqrt_sequence_conjecture_l658_658505


namespace intersecting_points_in_circle_l658_658639

theorem intersecting_points_in_circle :
  let n := 4 ∨ n := 5 ∨ n := 6 ∨ n := 7,
  ∃! n, (n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7) ∧
   (∀ (P : ℝ → Prop) (Q : ℝ → Prop) (R : ℝ → Prop),
    ¬(P ∧ Q ∧ R) ∧ ∀ (i j : ℕ), 4 ≤ i ∧ i ≤ 7 → 4 ≤ j ∧ j ≤ 7 → i ≠ j →
    let k := 2 * min i j in
    8 ∨ k = 10 ∨ k = 12) →
  (8 * 3 + 10 * 2 + 12 = 56) := sorry

end intersecting_points_in_circle_l658_658639


namespace readers_of_science_fiction_l658_658926

variable (Total S L B : Nat)

theorem readers_of_science_fiction 
  (h1 : Total = 400) 
  (h2 : L = 230) 
  (h3 : B = 80) 
  (h4 : Total = S + L - B) : 
  S = 250 := 
by
  sorry

end readers_of_science_fiction_l658_658926


namespace find_x_l658_658094

theorem find_x (x : ℝ) (h : log 8 x = 1.75) : x = 32 * real.root 4 2 := sorry

end find_x_l658_658094


namespace least_money_don_l658_658550

variables (Money : Type)
variables [LinearOrder Money]
variables (Alan Bea Cy Don Eva Fay : Money)

theorem least_money_don :
  (Fay < Alan) ∧ (Bea < Alan) ∧
  (Alan > Don) ∧ (Cy > Don) ∧
  (Fay > Don) ∧ (Fay < Alan) ∧
  (Eva > Bea) ∧ (Eva < Cy) ∧
  (Cy < Alan) →
  (Don = LinearOrder.min Alan (LinearOrder.min Bea (LinearOrder.min Cy (LinearOrder.min Eva Fay)))) :=
by
  sorry

end least_money_don_l658_658550


namespace factorization_is_correct_l658_658800

variable {b : ℕ}

-- Define the polynomial expression
def polynomial : ℕ := (4 * b^3 - 84 * b^2 - 12 * b) - (-3 * b^3 - 9 * b^2 + 3 * b)

-- State the factorized form
def factorized_form : ℕ := b * (7 * b^2 - 75 * b - 15)

-- State the quadratic factorization
def quadratic_factorization : ℕ := (7 * b + 3) * (b - 5)

-- The Lean theorem to state that the polynomial equals its factorized form
theorem factorization_is_correct : polynomial = b * quadratic_factorization :=
by
  sorry

end factorization_is_correct_l658_658800


namespace who_to_send_l658_658043

-- Definitions based on conditions
variable (A B C D : Type)
variable (score : A → ℝ) (variance : A → ℝ)
noncomputable def S_A : ℝ := 0.15
noncomputable def S_B : ℝ := 0.2
noncomputable def S_C : ℝ := 0.4
noncomputable def S_D : ℝ := 0.35
variable [Participant : ∀ a, score a = 9]

-- Define the problem statement in Lean: 
-- Prove that A is the most suitable to send to the competition
theorem who_to_send (a_is_suitable: ∀ (P : A), variance P ≥ variance A) : A := by
  sorry

end who_to_send_l658_658043


namespace fewer_green_than_yellow_l658_658754

  def num_purple := 10
  def num_total := 36
  def num_yellow := num_purple + 4
  def num_non_green := num_purple + num_yellow
  def num_green := num_total - num_non_green

  theorem fewer_green_than_yellow :
    (num_green - num_yellow) = -2 :=
  by
    sorry
  
end fewer_green_than_yellow_l658_658754


namespace minimum_area_triangle_l658_658304

def Point := ℝ × ℝ × ℝ

def Cube := {A B C D A₁ B₁ C₁ D₁ : Point}

def is_midpoint (p₁ p₂ m : Point) : Prop := 2 * m = p₁ + p₂

noncomputable def Cube.Area_Min_Triangle (A B C D A₁ B₁ C₁ D₁ : Point) (a : ℝ) : ℝ :=
  let E := ((B.1 + B₁.1) / 2, (B.2 + B₁.2) / 2, (B.3 + B₁.3) / 2)
  let F := ((C.1 + C₁.1) / 2, (C.2 + C₁.2) / 2, (C.3 + C₁.3) / 2)
  sorry

theorem minimum_area_triangle (A B C D A₁ B₁ C₁ D₁ : Point) (a : ℝ) 
  (hA: A = (0, 0, 0)) 
  (hB: B = (a, 0, 0)) 
  (hC: C = (a, a, 0)) 
  (hD: D = (0, a, 0)) 
  (hA₁: A₁ = (0, 0, a)) 
  (hB₁: B₁ = (a, 0, a)) 
  (hC₁: C₁ = (a, a, a)) 
  (hD₁: D₁ = (0, a, a)) 
  (hE: is_midpoint B B₁ ((B.1 + B₁.1) / 2, (B.2 + B₁.2) / 2, (B.3 + B₁.3) / 2))
  (hF: is_midpoint C C₁ ((C.1 + C₁.1) / 2, (C.2 + C₁.2) / 2, (C.3 + C₁.3) / 2)) :
  Cube.Area_Min_Triangle A B C D A₁ B₁ C₁ D₁ a = 7 * (a ^ 2) / 32 := sorry

end minimum_area_triangle_l658_658304


namespace pioneer_club_attendance_l658_658815

theorem pioneer_club_attendance :
  ∀ (pioneers clubs : Type) [fintype pioneers] [fintype clubs],
  fintype.card pioneers = 11 →
  fintype.card clubs = 5 →
  ∃ (A B : pioneers), ∀ (C : clubs), C ∈ A → C ∈ B :=
by sorry

end pioneer_club_attendance_l658_658815


namespace inverse_value_of_f_l658_658464

theorem inverse_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2^x - 2) : f⁻¹ 2 = 3 :=
sorry

end inverse_value_of_f_l658_658464


namespace triangle_area_not_integer_l658_658310

theorem triangle_area_not_integer (a b c : ℕ) (hp_a : Nat.Prime a) (hp_b : Nat.Prime b) (hp_c : Nat.Prime c) (h_triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a): ¬ ∃ S : ℕ, S = sqrt (p (p - a) (p - b) (p - c)) :=
  sorry

end triangle_area_not_integer_l658_658310


namespace simplify_expression_l658_658179

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) : 
  (∛((2 - a) ^ 3) + (∜((3 - a) ^ 4)) = 5 - 2 * a) :=
by 
  sorry

end simplify_expression_l658_658179


namespace equivalent_inequalities_l658_658733

noncomputable def condition_1 (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0
noncomputable def condition_2 (x : ℝ) : Prop := log (x - 2) ≤ 0

theorem equivalent_inequalities :
  (∀ x, condition_1 x ↔ condition_2 x) := 
sorry

end equivalent_inequalities_l658_658733


namespace number_of_minutes_l658_658817

-- Definitions based on the conditions of the problem
def car_collision_interval := 10 -- seconds
def big_crash_interval := 20 -- seconds
def accidents_in_minute := (60 / car_collision_interval) + (60 / big_crash_interval)
def total_accidents := 36

-- Statement that we need to prove
theorem number_of_minutes (car_collision_interval big_crash_interval : ℕ)
  (accidents_in_minute total_accidents : ℕ) : 
  car_collision_interval = 10 → 
  big_crash_interval = 20 → 
  accidents_in_minute = 9 → 
  total_accidents = 36 → 
  36 / 9 = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end number_of_minutes_l658_658817


namespace tetrahedron_triangle_count_l658_658169

theorem tetrahedron_triangle_count : 
  let vertices := 4 in
  let choose_three := Nat.choose vertices 3 in
  choose_three = 4 :=
by
  have vertices : Nat := 4
  have choose_three := Nat.choose vertices 3
  show choose_three = 4
  sorry

end tetrahedron_triangle_count_l658_658169


namespace P_Ravi_is_02_l658_658700

def P_Ram : ℚ := 6 / 7
def P_Ram_and_Ravi : ℚ := 0.17142857142857143

theorem P_Ravi_is_02 (P_Ravi : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ravi = 0.2 :=
by
  intro h
  sorry

end P_Ravi_is_02_l658_658700


namespace pat_climb_8_stairs_l658_658818

noncomputable def f : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 2
| (n + 3) := f (n + 2) + f (n + 1) + f n

theorem pat_climb_8_stairs : f 8 = 81 :=
by
  -- the proof steps would go here
  sorry

end pat_climb_8_stairs_l658_658818


namespace min_max_distance_CD_l658_658325

-- Define the sides of triangles ABC and ABD.
variables {a b c a1 b1 : ℝ}

-- Define m_c and m_d as the heights from C and D perpendicular to AB.
noncomputable def m_c : ℝ := real.sqrt (b^2 - (b^2 + c^2 - a^2)^2 / (4 * c^2))
noncomputable def m_d : ℝ := real.sqrt (b1^2 - (b1^2 + c^2 - a1^2)^2 / (4 * c^2))

-- State the theorem proving the min and max distances from C to D.
theorem min_max_distance_CD (ABC_fixed : true) (ABD_rotates : true) :
  min_max_distances_CD = (abs (m_c - m_d), m_c + m_d) := 
sorry

end min_max_distance_CD_l658_658325


namespace large_bucket_capacity_l658_658005

variable (S L : ℕ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
sorry

end large_bucket_capacity_l658_658005


namespace range_H_l658_658348

noncomputable def H (x : ℝ) : ℝ :=
  abs (x + 2) - abs (x - 2)

theorem range_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_H_l658_658348


namespace range_of_a_l658_658603

noncomputable def proposition_p (a : ℝ) : Prop := 
  0 < a ∧ a < 1

noncomputable def proposition_q (a : ℝ) : Prop := 
  a > 1 / 4

theorem range_of_a (a : ℝ) : 
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by sorry

end range_of_a_l658_658603


namespace angle_C_in_congruent_triangle_l658_658842

theorem angle_C_in_congruent_triangle
  (A B C D E F : ℝ) 
  (congruent : triangle ABC ≃ triangle DEF)
  (angle_A : ∠A = 40)
  (angle_E : ∠E = 80) :
  ∠C = 60 := by
    sorry

end angle_C_in_congruent_triangle_l658_658842


namespace range_of_a_l658_658542

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 :=
by {
  sorry
}

end range_of_a_l658_658542


namespace pablo_days_to_complete_all_puzzles_l658_658625

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l658_658625


namespace sum_of_prime_factors_204204_l658_658363

theorem sum_of_prime_factors_204204 : 
  let prime_factors := [2, 3, 7, 11, 13, 17] in
  prime_factors.sum = 53 :=
by {
  have h : prime_factors = [2, 3, 7, 11, 13, 17] := rfl,
  sorry
}

end sum_of_prime_factors_204204_l658_658363


namespace general_term_a_seq_l658_658849

noncomputable def a_seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 / 6 ∧
  a 2 = 19 / 36 ∧
  (∀ n, (log 2 (a (n + 1) - a n / 3)) =
    - ((n + 1):ℝ)) ∧
  (∀ n, (a (n + 1) - a n / 2) =
    (1 / 3) ^ (n + 1))

def a_formula (n : ℕ) : ℝ :=
  6 * ((1 / 2) ^ (n + 1) - (1 / 3) ^ (n + 1))

theorem general_term_a_seq :
  ∀ (a : ℕ → ℝ), a_seq a → (∀ n, a n = a_formula n) :=
by
  intros a h
  sorry  -- Proof goes here

end general_term_a_seq_l658_658849


namespace tiffany_first_level_treasures_l658_658696

-- Conditions
def treasure_points : ℕ := 6
def treasures_second_level : ℕ := 5
def total_points : ℕ := 48

-- Definition for the number of treasures on the first level
def points_from_second_level : ℕ := treasures_second_level * treasure_points
def points_from_first_level : ℕ := total_points - points_from_second_level
def treasures_first_level : ℕ := points_from_first_level / treasure_points

-- The theorem to prove
theorem tiffany_first_level_treasures : treasures_first_level = 3 :=
by
  sorry

end tiffany_first_level_treasures_l658_658696


namespace monotone_f_a_l658_658890

def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≥ 0 then a * x^2 + 1 else (a + 2) * Real.exp(a * x)

theorem monotone_f_a (a : ℝ) : (a ∈ Icc (-1 : ℝ) 0) → (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ∨ (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) :=
sorry

end monotone_f_a_l658_658890


namespace area_ABQDP_l658_658583

noncomputable def area_of_quadrilateral (P Q A B C D : Point) (hA : Angle) (hD : Angle) : ℝ :=
  let A_equal_10 : AB = 10
  let B_equal_6 : BC = 6
  let C_equal_22 : CD = 22
  let D_equal_8 : DA = 8
  let parallel_AB_CD : AB ∥ CD
  let bisectors_intersect_P : is_bisector P A hA 
  let bisectors_intersect_P : is_bisector P D hD
  let midpoint_Q_BC : is_midpoint Q BC 
in 50 * sqrt 2

theorem area_ABQDP (A B C D P Q : Point) (AB CD : Line) (AB_CD_par : AB ∥ CD)
  (AB_eq_10 : length AB = 10) (BC_eq_6 : length BC = 6)
  (CD_eq_22 : length CD = 22) (DA_eq_8 : length DA = 8)
  (angle_bisectors_P_A : bisector_of_angle P A)
  (angle_bisectors_P_D : bisector_of_angle P D)
  (midpoint_Q : midpoint Q BC) :
  area_of_quadrilateral P Q A B C D (angle_of_vertices A) (angle_of_vertices D) = 50 * sqrt 2 :=
sorry

end area_ABQDP_l658_658583


namespace no_integer_polynomial_exists_l658_658286

theorem no_integer_polynomial_exists 
    (a b c d : ℤ) (h : a ≠ 0) (P : ℤ → ℤ) 
    (h1 : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    (h2 : P 4 = 1) (h3 : P 7 = 2) : 
    false := 
by
    sorry

end no_integer_polynomial_exists_l658_658286


namespace H_range_l658_658358

noncomputable def H (x : ℝ) : ℝ :=
  if x < -2 then -4
  else if x < 2 then 2 * x
  else 4

theorem H_range : Set.range H = set.Icc (-4) 4 :=
sorry

end H_range_l658_658358


namespace distance_point_to_line_parametric_l658_658507

theorem distance_point_to_line_parametric :
  ∀ (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ),
  (P = (4, 0)) →
  (l = (λ t, (1 + t, -1 + t))) →
    let a : ℝ := 1 in
    let b : ℝ := -1 in
    let c : ℝ := -2 in
    let (x0, y0) := P in
    abs (a * x0 + b * y0 + c) / real.sqrt (a^2 + b^2) = real.sqrt 2 :=
by
  intros P l hP hl a b c x0 y0
  sorry

end distance_point_to_line_parametric_l658_658507


namespace pascal_fifth_element_row_20_l658_658710

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l658_658710


namespace find_quadratic_function_l658_658845

theorem find_quadratic_function (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a*x^2 + b*x + c)
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x + 1) = f x + x + 1) :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 1/2 ∧ f = λ x, (1/2)*x^2 + (1/2)*x :=
by
  sorry

end find_quadratic_function_l658_658845


namespace first_dimension_length_l658_658397

-- Definitions for conditions
def tank_surface_area (x : ℝ) : ℝ := 14 * x + 20
def cost_per_sqft : ℝ := 20
def total_cost (x : ℝ) : ℝ := (tank_surface_area x) * cost_per_sqft

-- The theorem we need to prove
theorem first_dimension_length : ∃ x : ℝ, total_cost x = 1520 ∧ x = 4 := by 
  sorry

end first_dimension_length_l658_658397


namespace combined_shoe_size_l658_658953

theorem combined_shoe_size :
  let jasmine_shoe_size := 7
  let alexa_shoe_size := 2 * jasmine_shoe_size
  let clara_shoe_size := 3 * jasmine_shoe_size
  let molly_shoe_size := 1.5 * jasmine_shoe_size
  let molly_sandal_size := molly_shoe_size - 0.5
  alexa_shoe_size + clara_shoe_size + molly_shoe_size + molly_sandal_size + jasmine_shoe_size = 62.5 :=
begin
  sorry
end

end combined_shoe_size_l658_658953


namespace stability_of_scores_requires_variance_l658_658330

-- Define the conditions
variable (scores : List ℝ)

-- Define the main theorem
theorem stability_of_scores_requires_variance : True :=
  sorry

end stability_of_scores_requires_variance_l658_658330


namespace geometric_seq_b_sum_first_n_terms_l658_658149

def seq_a (a : ℕ → ℝ) : Prop := a 1 = 3 / 2 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n - 2 / n + 3 / (n + 1) - 1 / (n + 2))

def seq_b (a b : ℕ → ℝ) : Prop := ∀ n : ℕ, b n = a n - 1 / (n * (n + 1))

theorem geometric_seq_b (a b : ℕ → ℝ) (hn : seq_a a) (hb : seq_b a b) : ∃ r : ℝ, ∀ n ≥ 1, b (n + 1) = r * b n := 
sorry

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)

theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) (hn : seq_a a) : S_n a n = 2^n - 1 / (n + 1) := 
sorry

end geometric_seq_b_sum_first_n_terms_l658_658149


namespace parabola_behavior_l658_658232

-- Definitions for the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- The proof statement
theorem parabola_behavior (a : ℝ) (x : ℝ) (ha : 0 < a) : 
  (0 < a ∧ a < 1 → parabola a x < x^2) ∧
  (a > 1 → parabola a x > x^2) ∧
  (∀ ε > 0, ∃ δ > 0, δ ≤ a → |parabola a x - 0| < ε) := 
sorry

end parabola_behavior_l658_658232


namespace A_inter_B_eq_14_l658_658275

def A := {1, 2, 3, 4}
def B := {x | ∃ m ∈ A, x = 3 * m - 2}

theorem A_inter_B_eq_14 : { x | x ∈ A ∧ x ∈ B } = {1, 4} :=
by 
  sorry

end A_inter_B_eq_14_l658_658275


namespace possible_values_of_expression_l658_658485

theorem possible_values_of_expression (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  ∃ (vals : Finset ℤ), vals = {6, 2, 0, -2, -6} ∧
  (∃ val ∈ vals, val = (if p > 0 then 1 else -1) + 
                         (if q > 0 then 1 else -1) + 
                         (if r > 0 then 1 else -1) + 
                         (if s > 0 then 1 else -1) + 
                         (if (p * q * r) > 0 then 1 else -1) + 
                         (if (p * r * s) > 0 then 1 else -1)) :=
by
  sorry

end possible_values_of_expression_l658_658485


namespace find_f_of_given_conditions_l658_658501

theorem find_f_of_given_conditions
  (A m : ℝ)
  (ω : ℝ)
  (ϕ : ℝ)
  (A_pos : A > 0)
  (ω_pos : ω > 0)
  (ϕ_range : 0 < ϕ ∧ ϕ < π/2)
  (max_eqn : A + m = 4)
  (min_eqn : m - A = 0)
  (period_eqn : (2 * π) / ω = π / 2)
  (symmetry_eqn : (4 * π / 3) + ϕ = 3 * π / 2):
  (∀ x : ℝ, f(x) = 2 * sin(4 * x - π / 6) + 2) :=
by
  sorry

end find_f_of_given_conditions_l658_658501


namespace find_angle_between_vectors_l658_658155

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def angle_between (a b : V) : ℝ :=
real.acos ((⟪a, b⟫) / (∥a∥ * ∥b∥))

theorem find_angle_between_vectors (A B C D : V) 
  (h_ab : ∥A - B∥ = 2)
  (h_cd : ∥C - D∥ = 1)
  (h_ab_cd : ∥(A - B) - 2 * (C - D)∥ = 2 * real.sqrt 3) :
  angle_between (A - B) (C - D) = 2 * real.pi / 3 :=
sorry

end find_angle_between_vectors_l658_658155


namespace bill_receives_26_l658_658021

theorem bill_receives_26 (M : ℝ)
  (h1 : 1 + (1 / 3) * (M - 1) + 6 + (1 / 3) * (2 / 3 * M - 2 / 3) + (4 / 9 * M - 58 / 9) = M)
  (h2 : 4 / 9 * M - 58 / 9 = 40) :
  6 + (1 / 3) * (2 / 3 * M - 2 / 3) = 26 :=
begin
  sorry
end

end bill_receives_26_l658_658021


namespace locus_D_range_of_MN_l658_658462

open Real

-- Define the given points and conditions
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 1 0

def on_circle (C : Point) : Prop :=
  (C.x + 1)^2 + C.y^2 = 8

def perpendicular_bisector (A C : Point) : set Point :=
  {D | ∃ m b : ℝ, (D.y = m * D.x + b ∧ m * (A.x + C.x) / 2 + b = (A.y + C.y) / 2)}

def intersects_AC_and_BC (A B C D : Point) : Prop :=
  perpendicular_bisector A C D ∧ ∃ α β : ℝ, (D.x = α * (C.x - B.x) + B.x ∧ D.y = α * (C.y - B.y) + B.y) ∧ (D.x = β * (A.x - C.x) + C.x ∧ D.y = β * (A.y - C.y) + C.y)

-- Statement (1): Equation of the locus E of point D
theorem locus_D (C D : Point) (hC : on_circle C) (hD : intersects_AC_and_BC A (Point.mk -1 0) C D) :
  (D.x^2 / 2 + D.y^2 = 1) :=
sorry

-- Statement (2): Range of |MN|
def origin := Point.mk 0 0
def is_on_line (E : Point) (M N P : Point) := 
  ∃ k m : ℝ, M.y = k * M.x + m ∧ N.y = k * N.x + m ∧ P.y = k * P.x + m

def is_balanced (M N P : Point) :=
  (M.x + N.x + P.x = 0) ∧ (M.y + N.y + P.y = 0)

def distance (P1 P2 : Point) := 
  sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2)

theorem range_of_MN (M N P : Point)
  (hM_line : is_on_line E M N P)
  (hBalance : is_balanced M N P) :
  ∃ MN_min MN_max : ℝ, (MN_min = sqrt 3 ∧ MN_max = sqrt 6) ∧ 
    ( ∀ x : ℝ, distance M N = x → MN_min ≤ x ∧ x ≤ MN_max ) :=
sorry

end locus_D_range_of_MN_l658_658462


namespace solution_proved_l658_658217

noncomputable def problem_statement : Prop :=
  let C1 (x y : ℝ) := x^2 + y^2 = 1 in
  let l (ρ θ : ℝ) := ρ * (Real.cos θ - Real.sin θ) = 4 in
  let C2 (x' y' : ℝ) := (x' / 2)^2 + (y' / sqrt 3)^2 = 1 in
  let l_cartesian (x y : ℝ) := x - y = 4 in
  let C2_cartesian (x' y' : ℝ) := (x' ^ 2) / 4 + (y' ^ 2) / 3 = 1 in
  let P := (1, 2) in
  let l1_passes_through_P (x y : ℝ) (t : ℝ) := x = 1 + (Real.sqrt 2 / 2) * t ∧ y = 2 + (Real.sqrt 2 / 2) * t in
  let intersects (t : ℝ) := t ≠ 0 ∧ (7 / 2) * t^2 + 11 * Real.sqrt 2 * t + 7 = 0 in
  (∀ ρ θ, l ρ θ → l_cartesian ρ θ) ∧
  (∀ x' y', C1 (x' / 2) (y' / sqrt 3) → C2_cartesian x' y') ∧
  (∀ M N : ℝ × ℝ, intersects (M.1 - 1 / (Real.sqrt 2 / 2)) → intersects (N.1 - 1 / (Real.sqrt 2 / 2)) → 
                   |M.1 - P.1| * |N.1 - P.1| = 2)

theorem solution_proved : problem_statement :=
sorry

end solution_proved_l658_658217


namespace small_cubes_with_painted_faces_l658_658028

-- Define the original conditions
def original_cube_edges : ℕ := 5
def total_small_cubes (n : ℕ) : ℕ := n * n * n
def internal_small_cubes (n : ℕ) : ℕ := (n - 2) * (n - 2) * (n - 2)
def painted_faces_small_cubes (total : ℕ) (internal : ℕ) : ℕ := total - internal

-- State the theorem
theorem small_cubes_with_painted_faces :
  ∃ n : ℕ, ∃ total : ℕ, ∃ internal : ℕ, 
    (total = total_small_cubes n) ∧ (internal = internal_small_cubes n) ∧ (painted_faces_small_cubes total internal = 98) :=
by
  let n := original_cube_edges
  let total := total_small_cubes n
  let internal := internal_small_cubes n
  exists n
  exists total
  exists internal
  split
  { unfold total_small_cubes, exact rfl }
  split
  { unfold internal_small_cubes, exact rfl }
  unfold painted_faces_small_cubes
  exact rfl

end small_cubes_with_painted_faces_l658_658028


namespace find_q_l658_658859

theorem find_q (q: ℕ) (h: 81^10 = 3^q) : q = 40 :=
by
  sorry

end find_q_l658_658859


namespace pascal_triangle_row_20_element_5_l658_658724

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l658_658724


namespace more_people_this_week_l658_658227

-- Define the conditions
variables (second_game first_game third_game : ℕ)
variables (total_last_week total_this_week : ℕ)

-- Conditions
def condition1 : Prop := second_game = 80
def condition2 : Prop := first_game = second_game - 20
def condition3 : Prop := third_game = second_game + 15
def condition4 : Prop := total_last_week = 200
def condition5 : Prop := total_this_week = second_game + first_game + third_game

-- Theorem statement
theorem more_people_this_week (h1 : condition1)
                             (h2 : condition2)
                             (h3 : condition3)
                             (h4 : condition4)
                             (h5 : condition5) : total_this_week - total_last_week = 35 :=
sorry

end more_people_this_week_l658_658227


namespace triangle_area_not_integer_l658_658309

theorem triangle_area_not_integer (a b c : ℕ) (hp_a : Nat.Prime a) (hp_b : Nat.Prime b) (hp_c : Nat.Prime c) (h_triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a): ¬ ∃ S : ℕ, S = sqrt (p (p - a) (p - b) (p - c)) :=
  sorry

end triangle_area_not_integer_l658_658309


namespace H_range_l658_658357

noncomputable def H (x : ℝ) : ℝ :=
  if x < -2 then -4
  else if x < 2 then 2 * x
  else 4

theorem H_range : Set.range H = set.Icc (-4) 4 :=
sorry

end H_range_l658_658357


namespace circumcenter_on_median_l658_658973

variable (A B C O P Q H : Point)
variable (h_b h_c : Line)
variable [Triangle ABC] [AcuteTriangle ABC] [Circumcenter O ABC] [Altitude h_b B] [Altitude h_c C] [Intersection OA_restrictionA h_b P] [Intersection OA_restrictionA h_c Q] [Orthocenter H ABC]

-- Prove that the center of the circumcircle of triangle PQH lies on the median of triangle ABC that passes through A
theorem circumcenter_on_median :
  let circumcenter_PQH := Circumcenter (triangle PQH) in
  let AM_median := median_through A ABC in
  lies_on AM_median circumcenter_PQH :=
sorry

end circumcenter_on_median_l658_658973


namespace total_cost_is_131000_l658_658401

noncomputable def watch_listed_price : ℕ := 50000
noncomputable def watch_first_discount : ℝ := 0.12
noncomputable def necklace_listed_price : ℕ := 75000
noncomputable def necklace_second_discount : ℝ := 0.24
noncomputable def handbag_listed_price : ℕ := 40000
noncomputable def handbag_third_discount : ℝ := 0.25

noncomputable def price_after_discount (listed_price : ℕ) (discount : ℝ) : ℕ :=
  listed_price - (listed_price * discount).toNat

noncomputable def total_cost : ℕ :=
  price_after_discount watch_listed_price watch_first_discount +
  price_after_discount necklace_listed_price necklace_second_discount +
  price_after_discount handbag_listed_price handbag_third_discount

theorem total_cost_is_131000 : total_cost = 131000 := by
  sorry

end total_cost_is_131000_l658_658401


namespace range_H_l658_658352

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_H : set.range H = {-4, 4} := 
by
  sorry

end range_H_l658_658352


namespace number_of_valid_4_digit_numbers_l658_658088

def is_valid_4_digit_number (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits.length = 4 ∧
  digits.all (λ d, d ∈ [0, 1, 2, 3, 4, 5]) ∧
  digits.nodup ∧
  (∀ i, i < 3 → ¬ (digits.nth i % 2 = 0 ∧ digits.nth (i + 1) % 2 = 0))

theorem number_of_valid_4_digit_numbers : (finset.filter is_valid_4_digit_number (finset.Icc 1000 5999)).card = 150 :=
  sorry

end number_of_valid_4_digit_numbers_l658_658088


namespace pascal_triangle_row_20_element_5_l658_658702

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l658_658702


namespace solution_to_inequality_l658_658297

noncomputable def initial_inequality (x : ℝ) : Prop :=
  log (3 + sin x - cos x) (3 - (cos (2 * x) / (cos x + sin x))) ≥ exp (sqrt x)

theorem solution_to_inequality (x : ℝ) (h1 : cos x + sin x ≠ 0) (h2 : 3 + sin x - cos x > 1) :
  initial_inequality x ↔ x = 0 := 
sorry

end solution_to_inequality_l658_658297


namespace hyperbola_eccentricity_l658_658142

theorem hyperbola_eccentricity (m : ℝ) (a b c e : ℝ) (h1 : x^2 + m * y^2 = 1) 
  (h2 : b = 2 * a) (h3 : c = sqrt (a^2 + b^2)) (h4 : e = c / a) : 
  e = sqrt 5 :=
  sorry

end hyperbola_eccentricity_l658_658142


namespace count_transformations_return_t_l658_658584

def Point := (ℝ × ℝ)
def Triangle := {v1 : Point // ∃ v2 v3 : Point, ({v1, v2, v3} : set Point).card = 3 }

def rotation (θ : ℝ) (p : Point) : Point :=
  match p with
  | (x, y) => (x * real.cos θ - y * real.sin θ, x * real.sin θ + y * real.cos θ)

def reflection_x (p : Point) : Point :=
  match p with
  | (x, y) => (x, -y)

def reflection_y (p : Point) : Point :=
  match p with
  | (x, y) => (-x, y)

def translation (v : Point) (p : Point) : Point :=
  match (v, p) with
  | ((vx, vy), (x, y)) => (x + vx, y + vy)

def apply_transformations (t : Triangle) (transforms : list (Point → Point)) : Triangle :=
  transforms.foldr (λ f acc, {acc with v1 := f acc.v1}) t -- This generalizes the transformation application

noncomputable def transformed_sequences (t : Triangle) : list (list (Point → Point)) :=
  let rotations := [rotation (real.pi / 2), rotation real.pi, rotation (3 * real.pi / 2)]
  let reflections := [reflection_x, reflection_y]
  let translation_v := [(2, -1)]
  let all_transforms := rotations ++ reflections ++ translation_v
  (all_transforms.product (all_transforms.product all_transforms)).map (λ triple, [triple.0, triple.1.0, triple.1.1])

theorem count_transformations_return_t (
  T : Triangle := ⟨(0, 0), ⟨(6, 0), (0, 4), by simp [finset.card]⟩⟩
) :
  (transformed_sequences T).count (λ transforms, apply_transformations T transforms = T) = 9 :=
sorry

end count_transformations_return_t_l658_658584


namespace intersection_finite_nonempty_l658_658238

open Real

noncomputable def S : Set (ℝ × ℝ × ℝ) := {p | ∃ t : ℝ, p = (t^5, t^3, t)}

theorem intersection_finite_nonempty (a b c d : ℝ) (h_plane : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0):
  ∃ t₁ t₂ t₃ : Finite (S ∩ {p | a * p.1 + b * p.2 + c * p.3 + d = 0}),
  t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₃ ≠ t₁ ∧ t₁ ≠ none ∧ t₂ ≠ none ∧ t₃ ≠ none :=
begin
  sorry
end

end intersection_finite_nonempty_l658_658238


namespace proof_problem_l658_658898

def setA : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}

def complementB : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def intersection : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem proof_problem :
  (setA ∩ complementB) = intersection := 
by
  sorry

end proof_problem_l658_658898


namespace bisectors_intersect_on_BC_l658_658567

noncomputable def midpoint (A B : Point) : Point := ⟨ (A + B) / 2 ⟩ -- Not actual Lean syntax; just a placeholder for intuition

variables (A B C D E P : Point)
variables (AB AC BC : ℝ)

-- Given conditions
def D_midpoint_AB : midpoint A B = D := sorry
def E_midpoint_AC : midpoint A C = E := sorry
def BC_mean_length : BC = (AB + AC) / 2 := sorry

-- Objective to prove
theorem bisectors_intersect_on_BC :
  bisector_through D E P ∧ bisector_through E D P → intersect_on BC P :=
sorry

end bisectors_intersect_on_BC_l658_658567


namespace chord_length_of_intersection_l658_658938

/-- 
Given the parametric equations of circle C and line l, 
prove that the length of the chord formed by their intersection is 4 * sqrt 6.
-/
theorem chord_length_of_intersection 
  (θ t : ℝ)
  (hC_x : ∀ θ, x = 5 * Real.cos θ - 1)
  (hC_y : ∀ θ, y = 5 * Real.sin θ + 2)
  (hl_x : ∀ t, x = 4 * t + 6)
  (hl_y : ∀ t, y = -3 * t - 2) : 
  let d := 1 
  in 2 * Real.sqrt(25 - d^2) = 4 * Real.sqrt 6 := 
sorry

end chord_length_of_intersection_l658_658938


namespace fifth_element_row_20_pascal_triangle_l658_658721

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l658_658721


namespace mona_unique_players_l658_658608

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l658_658608


namespace perpendicular_line_plane_implies_perpendicular_lines_l658_658745

-- Define the relevant types and properties
variables {Line Plane : Type}
variables [IsLine Line] [IsPlane Plane]

-- Define the perpendicular and subset relations
noncomputable def perpendicular (l : Line) (p : Plane) : Prop := sorry
noncomputable def subset (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between lines
noncomputable def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Given conditions: m, n are lines, and α is a plane. 
variables (m n : Line) (α : Plane)

-- Assuming m ⊥ α and n ⊆ α
axiom m_perpendicular_alpha : perpendicular m α
axiom n_subset_alpha : subset n α

-- Prove that m ⊥ n
theorem perpendicular_line_plane_implies_perpendicular_lines :
  perpendicular m α ∧ subset n α → perpendicular_lines m n :=
by
  intro h,
  sorry

end perpendicular_line_plane_implies_perpendicular_lines_l658_658745


namespace arithmetic_seq_a12_l658_658934

variable {a : ℕ → ℝ}

theorem arithmetic_seq_a12 :
  (∀ n, ∃ d, a (n + 1) = a n + d)
  ∧ a 5 + a 11 = 30
  ∧ a 4 = 7
  → a 12 = 23 :=
by
  sorry


end arithmetic_seq_a12_l658_658934


namespace interval_for_n_l658_658831

theorem interval_for_n (n : ℕ) (h1 : n < 500) 
                       (h2 : ∃ k, (1 : ℚ) / n = (0.abcd(k))) 
                       (h3 : ∃ k, (1 : ℚ) / (n + 5) = (0.uvw(k))) 
                       (h4 : n ∣ (10^5 - 1)) 
                       (h5 : (n + 5) ∣ (10^3 - 1)) : 
  126 ≤ n ∧ n ≤ 250 :=
sorry

end interval_for_n_l658_658831


namespace golden_chest_diamonds_rubies_l658_658790

theorem golden_chest_diamonds_rubies :
  ∀ (diamonds rubies : ℕ), diamonds = 421 → rubies = 377 → diamonds - rubies = 44 :=
by
  intros diamonds rubies
  sorry

end golden_chest_diamonds_rubies_l658_658790


namespace range_of_3x_minus_y_l658_658461

-- Defining the conditions in Lean
variable (x y : ℝ)

-- Condition 1: -1 ≤ x + y ≤ 1
def cond1 : Prop := -1 ≤ x + y ∧ x + y ≤ 1

-- Condition 2: 1 ≤ x - y ≤ 3
def cond2 : Prop := 1 ≤ x - y ∧ x - y ≤ 3

-- The theorem statement to prove that the range of 3x - y is [1, 7]
theorem range_of_3x_minus_y (h1 : cond1 x y) (h2 : cond2 x y) : 1 ≤ 3 * x - y ∧ 3 * x - y ≤ 7 := by
  sorry

end range_of_3x_minus_y_l658_658461


namespace hexagon_perimeter_sum_l658_658004

theorem hexagon_perimeter_sum :
  let distances := [-- List of distances between points
    real.sqrt (1^2 + 1^2),               -- (0,0) to (1,1)
    real.sqrt ((2 - 1)^2 + (1 - 1)^2),   -- (1,1) to (2,1)
    real.sqrt ((3 - 2)^2 + (0 - 1)^2),   -- (2,1) to (3,0)
    real.sqrt ((2 - 3)^2 + (-1 - 0)^2),  -- (3,0) to (2,-1)
    real.sqrt ((1 - 2)^2 + (-1 + 1)^2),  -- (2,-1) to (1,-1)
    real.sqrt ((0 - 1)^2 + (0 + 1)^2)    -- (1,-1) to (0,0)
  ]
  let perimeter := distances.sum
  let (a, b, c) := (2, 4, 0)  -- coefficients of 2 + 4sqrt(2) + 0sqrt(3)
  a + b + c = 6 :=
by
  sorry

end hexagon_perimeter_sum_l658_658004


namespace solution_set_of_inequality_l658_658673

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, f(x) ∈ ℝ
axiom f_at_neg1 : f (-1) = 2
axiom f_derivative_pos : ∀ x : ℝ, (derivative f x) > 2

theorem solution_set_of_inequality : { x : ℝ | f x > 2*x + 4 } = Ioi (-1) :=
by sorry

end solution_set_of_inequality_l658_658673


namespace sufficient_condition_l658_658983

-- Definitions:
-- 1. Arithmetic sequence with first term a_1 and common difference d
-- 2. Define the sum of the first n terms of the arithmetic sequence

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + n * d

def sum_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Conditions given in the problem:
-- Let a_6 = a_1 + 5d
-- Let a_7 = a_1 + 6d
-- Condition p: a_6 + a_7 > 0

def p (a_1 d : ℤ) : Prop := a_1 + 5 * d + a_1 + 6 * d > 0

-- Sum of first 9 terms S_9 and first 3 terms S_3
-- Condition q: S_9 >= S_3

def q (a_1 d : ℤ) : Prop := sum_first_n_terms a_1 d 9 ≥ sum_first_n_terms a_1 d 3

-- The statement to prove:
theorem sufficient_condition (a_1 d : ℤ) : (p a_1 d) -> (q a_1 d) :=
sorry

end sufficient_condition_l658_658983


namespace inv_prop_function_quadrants_l658_658127

theorem inv_prop_function_quadrants (k : ℝ) (h : ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → has_point (λ x : ℝ, k / x) (-3, 3)) :
    ∃ k : ℝ, (k = -9) ∧ (∀ x : ℝ, x ≠ 0 → if x < 0 then (k / x > 0) else (k / x < 0)) :=
by 
  obtain ⟨k, h_k⟩ := h  
  have A := h_k (-3) (by linarith)
  rw [<-A] at h_k 
  use k
  fa sorry
  sorry
  sorry

end inv_prop_function_quadrants_l658_658127


namespace infinite_non_overlapping_squares_l658_658408

theorem infinite_non_overlapping_squares :
  ∃ (seq : ℕ → ℝ × ℝ × ℕ), (∀ n, seq n).snd = \frac{1}{n+1} ∧ 
  (∀ m n, m ≠ n → disjoint (seq m) (seq n)) :=
sorry

end infinite_non_overlapping_squares_l658_658408


namespace value_of_f_at_third_l658_658920

theorem value_of_f_at_third :
  ∀ (ω : ℝ), ω > 0 ∧ (∀ x, f(x) = sin(ω * real.pi * x - real.pi / 6)) ∧ (min_positive_period f = 1 / 5) →
    f(1 / 3) = -1 / 2 :=
by
  intros ω h
  sorry

end value_of_f_at_third_l658_658920


namespace moles_of_SO3_l658_658520

-- Define chemical species
inductive ChemicalSpecies
| SO3
| H2O
| H2SO4

open ChemicalSpecies

-- Hypothesis: The balanced chemical equation for the reaction
def balanced_eq : ∀ (s1 s2 s3 : ChemicalSpecies), (s1 = SO3) ∧ (s2 = H2O) ∧ (s3 = H2SO4) → Prop :=
λ s1 s2 s3 h, h = (SO3, H2O, H2SO4)

-- Theorem: The number of moles of SO3 required to react with 2 moles of H2O to form 2 moles of H2SO4 is 2
theorem moles_of_SO3 (n_H2O : ℕ) (n_H2SO4 : ℕ) :
  (∀ (s1 s2 s3 : ChemicalSpecies), balanced_eq s1 s2 s3 (s1, s2, s3)) → 
  n_H2O = 2 → 
  n_H2SO4 = 2 → 
  ∃ n_SO3 : ℕ, n_SO3 = 2 :=
by { intros h_eq h_H2O h_H2SO4, existsi 2, sorry }

end moles_of_SO3_l658_658520


namespace nine_chapters_compensation_difference_l658_658562

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l658_658562


namespace determinant_scaled_l658_658104

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end determinant_scaled_l658_658104


namespace recurrence_solution_l658_658212

noncomputable def recurrence_relation (c : ℝ) : ℕ → ℝ
| 0       := c
| (n + 1) := 2 * (recurrence_relation c n - 1)^2

theorem recurrence_solution (c : ℝ) (n : ℕ) :
  recurrence_relation c n = 
    (1 / 2) * (( (2 * (c - 1) + real.sqrt ((2 * (c - 1))^2 - 4)) / 2)^(2^n) + 
    ((2 * (c - 1) - real.sqrt((2 * (c - 1))^2 - 4)) / 2)^(2^n) + 2) :=
by
  sorry

end recurrence_solution_l658_658212


namespace number_of_pairs_eq_2_l658_658176

theorem number_of_pairs_eq_2 : 
  { (x, y) : ℕ × ℕ // x > 0 ∧ y > 0 ∧ x^2 - y^2 = 51 }.to_finset.card = 2 :=
by
  sorry

end number_of_pairs_eq_2_l658_658176


namespace contradiction_in_triangle_l658_658287

theorem contradiction_in_triangle (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by
  sorry

end contradiction_in_triangle_l658_658287


namespace subsets_of_A_value_of_a_l658_658581

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 - a*x + 2 = 0}

theorem subsets_of_A : 
  (A = {1, 2} ∧ (∀ S, S ⊆ A → S = ∅ ∨ S = {1} ∨ S = {2} ∨ S = {1, 2}))  :=
by
  sorry

theorem value_of_a (a : ℝ) (B_non_empty : B a ≠ ∅) (B_subset_A : ∀ x, x ∈ B a → x ∈ A): 
  a = 3 :=
by
  sorry

end subsets_of_A_value_of_a_l658_658581


namespace range_H_l658_658345

noncomputable def H (x : ℝ) : ℝ :=
  abs (x + 2) - abs (x - 2)

theorem range_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_H_l658_658345


namespace log_cut_piece_weight_l658_658249

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end log_cut_piece_weight_l658_658249


namespace eval_expression_l658_658797

theorem eval_expression : (-2 ^ 4) + 3 * (-1) ^ 6 - (-2) ^ 3 = -5 := by
  sorry

end eval_expression_l658_658797


namespace total_pencils_l658_658816

def num_boxes : ℕ := 12
def pencils_per_box : ℕ := 17

theorem total_pencils : num_boxes * pencils_per_box = 204 := by
  sorry

end total_pencils_l658_658816


namespace comparison_of_a_b_c_l658_658968

def a : ℝ := Real.log 0.3 / Real.log 3
def b : ℝ := 2^0.3
def c : ℝ := 0.3^2

theorem comparison_of_a_b_c : b > c ∧ c > a :=
by 
  -- Here to defer the actual proof
  sorry

end comparison_of_a_b_c_l658_658968


namespace tetrahedron_triangle_count_l658_658166

theorem tetrahedron_triangle_count : 
  let vertices := 4 in
  let choose_three := Nat.choose vertices 3 in
  choose_three = 4 :=
by
  have vertices : Nat := 4
  have choose_three := Nat.choose vertices 3
  show choose_three = 4
  sorry

end tetrahedron_triangle_count_l658_658166


namespace limit_solution_l658_658375

noncomputable def limit_problem : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (↑(Real.pi) / 2)| < δ →
  |(((2^(Real.cos(x)^2)) - 1) / (Real.log(Real.sin(x)))) - (-(2 * Real.log(2)))| < ε

theorem limit_solution : limit_problem :=
sorry

end limit_solution_l658_658375


namespace pascal_triangle_row_20_element_5_l658_658703

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l658_658703


namespace probability_of_multiple_of_3_or_7_l658_658329

theorem probability_of_multiple_of_3_or_7 :
  let total_tickets := finset.range 100 + 1 in
  let multiples_of_3 := finset.filter (λ n, n % 3 = 0) total_tickets in
  let multiples_of_7 := finset.filter (λ n, n % 7 = 0) total_tickets in
  let multiples_of_21 := finset.filter (λ n, n % 21 = 0) total_tickets in
  multiples_of_3.card + multiples_of_7.card - multiples_of_21.card = 43 :=
by
  sorry

end probability_of_multiple_of_3_or_7_l658_658329


namespace ratio_of_rise_in_liquid_level_l658_658701

theorem ratio_of_rise_in_liquid_level 
(h1 h2 : ℝ) 
(V1 V2 : ℝ) 
(h1_prime h2_prime : ℝ)
(r1 r2 : ℝ := 5) -- radii of the tops of the liquid surfaces in cm
(sphere_r : ℝ := 2) -- radius of the spherical marble in cm
(vol_sphere : ℝ := (4/3) * Math.pi * sphere_r ^ 3) 
(h1_eq : V1 = (1/3) * Math.pi * r1 ^ 2 * h1)
(h2_eq : V2 = (1/3) * Math.pi * r2 ^ 2 * h2)
(V1_eq_V2 : V1 = V2)
(volume_gap_1 : V1 + vol_sphere = (1/3) * Math.pi * r1 ^ 2 * h1_prime)
(volume_gap_2 : V2 + vol_sphere = (1/3) * Math.pi * r2 ^ 2 * h2_prime)
(height_ratio : h1 / h2 = 4)
: (h1_prime - h1) / (h2_prime - h2) = 4 :=
sorry

end ratio_of_rise_in_liquid_level_l658_658701


namespace poly_roots_nature_l658_658087

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 3 * Polynomial.X^2 + 4 * Polynomial.X - 12

theorem poly_roots_nature : 
  ∃ r : ℝ, IsRoot poly r ∧ ∃ c1 c2 : ℂ, c1.im ≠ 0 ∧ c2.im ≠ 0 ∧ IsRoot poly c1 ∧ IsRoot poly c2 :=
sorry

end poly_roots_nature_l658_658087


namespace sum_of_digits_3n_l658_658766

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_3n (n : ℕ) (hn1 : digit_sum n = 100) (hn2 : digit_sum (44 * n) = 800) : digit_sum (3 * n) = 300 := by
  sorry

end sum_of_digits_3n_l658_658766


namespace proof_problem_l658_658649

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l658_658649


namespace express_P_as_polynomial_l658_658659

noncomputable def P (x : ℝ) : ℝ := sorry -- To satisfy the condition it must be defined as real polynomial

lemma polynomial_condition (θ : ℝ) : 
  P (complex.cos θ + complex.sin θ) = P (complex.cos θ - complex.sin θ) :=
sorry -- Given as a condition

theorem express_P_as_polynomial :
  ∃ (a : list ℝ) (n : ℕ), 
    ∀ x, P x = a.foldr (λ (coeff acc : ℝ), acc + coeff * (1 - x^2)^(2 * (a.indexOf coeff))) 0 :=
sorry -- To prove

end express_P_as_polynomial_l658_658659


namespace complex_exp_l658_658529

theorem complex_exp {i : ℂ} (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end complex_exp_l658_658529


namespace focal_length_of_hyperbola_equation_of_hyperbola_range_of_slope_k_l658_658980

-- Given conditions for the hyperbola
variable (t : ℝ) (ht : t > 0)
def hyperbola_eq (x y : ℝ) := x^2 / t^2 - y^2 = 1

-- Given eccentricity
variable (e : ℝ) (he : e = Real.sqrt 10 / 3)

-- Proof Problem 1: Focal length of the hyperbola
theorem focal_length_of_hyperbola :
  let a := t in let c := Real.sqrt (t^2 + 1) in 2 * c = 2 * Real.sqrt 10 :=
sorry

-- Additional conditions for proof problem 2
variable (F₁ : ℝ → ℝ) (O : ℝ → ℝ) (M : ℝ → ℝ)
variable (normal_vector : ℝ × ℝ) (area_triangle : ℝ)
variable (hnv : normal_vector = (t, -1)) (ha : area_triangle = 1/2)

-- Proof Problem 2: Equation of the hyperbola
theorem equation_of_hyperbola :
  let c := Real.sqrt (t^2 + 1) in hyperbola_eq 1 0 :=
sorry

-- Additional conditions for proof problem 3
variable (k m : ℝ) (hk : k > 0)
variable (intersection_pts : ℝ → ℝ) (vector_sum : ℝ × ℝ)
variable (hinter : vector_sum = 4)

-- Proof Problem 3: Range of the slope k
theorem range_of_slope_k :
  let a := Real.sqrt 2 in 0 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end focal_length_of_hyperbola_equation_of_hyperbola_range_of_slope_k_l658_658980


namespace line_equation_intersecting_ellipse_midpoint_l658_658490

theorem line_equation_intersecting_ellipse_midpoint :
  let midpoint : ℝ × ℝ := (4, 2)
  let ellipse (θ : ℝ) : ℝ × ℝ := (6 * Real.cos θ, 3 * Real.sin θ)
  (intersection : ∃ (l : ℝ → ℝ × ℝ),
    l = λ x, ⟨x, -(1 / 2) * x⟩ ∧
    ∃ t₁ t₂,
      let p1 := ellipse t₁,
      let p2 := ellipse t₂ in
      (p1 + p2) / 2 = midpoint) →
  ∃ a b c, a = 1 ∧ b = 2 ∧ c = -8 ∧
    ∀ x y, y = -(1 / 2) * x ↔ a * x + b * y + c = 0 :=
by {
  intro inteq,
  use [1, 2, -8],
  split; try {ring},
  intros x y,
  exact ⟨λ h, by rw [h], λ h, congr_arg (λ z, -z / 2) h⟩,
}

end line_equation_intersecting_ellipse_midpoint_l658_658490


namespace sequence_not_progression_l658_658863

-- Define the conditions
variable (a b c : ℝ) (n : ℕ)
variable (h1 : 1 < n) (h2 : a > 0) (h3 : b = 2 * a) (h4 : c = 4 * a)
variable (h5 : a ≠ b) (h6 : b ≠ c) (h7 : c ≠ a)

-- Define the logarithmic and sine expressions
noncomputable def log_a (n : ℝ) (a : ℝ) := log n / log a
noncomputable def log_b (n : ℝ) (b : ℝ) := log n / log b
noncomputable def log_c (n : ℝ) (c : ℝ) := log n / log c

noncomputable def sin_log_a (n : ℝ) (a : ℝ) := Real.sin (log_a n a)
noncomputable def sin_log_b (n : ℝ) (b : ℝ) := Real.sin (log_b n b)
noncomputable def sin_log_c (n : ℝ) (c : ℝ) := Real.sin (log_c n c)

-- Problem statement
theorem sequence_not_progression :
  ¬ ((sin_log_a n a), (sin_log_b n b), (sin_log_c n c)).isArithmeticSequence ∧
  ¬ ((sin_log_a n a), (sin_log_b n b), (sin_log_c n c)).isGeometricSequence ∧
  ¬ ((1 / sin_log_a n a), (1 / sin_log_b n b), (1 / sin_log_c n c)).isArithmeticSequence ∧
  ¬ ((sin_log_a n a) = (sin_log_b n b) ∧ (sin_log_b n b) = (sin_log_c n c)) ∧
  ¬ none_of_these := sorry

end sequence_not_progression_l658_658863


namespace line_TC_bisects_DB_l658_658924

variables {A B C D E F Q P R S T M M1 M2 : Type*}
variables [LinearOrder A] [LinearOrder B] [LinearOrder C]
variables [LinearOrder D] [LinearOrder E] [LinearOrder F]
variables [LinearOrder Q] [LinearOrder P] [LinearOrder R] [LinearOrder S] [LinearOrder T] 
variables [LinearOrder M] [LinearOrder M1] [LinearOrder M2]

-- Defining the conditions of the problem
def is_cyclic_hexagon (A B C D E F : Type*) : Prop := sorry -- Definition of a convex cyclic hexagon
def BC_eq_EF (B C E F : Type*) (h1 : B ≠ C ≠ E ≠ F) : B = E * F = C := sorry
def CD_eq_AF (C D A F : Type*) (h2 : C ≠ D ≠ A ≠ F) : C = A * D = F := sorry
def intersection_AC_BF (AC BF Q : Type*) (h3 : AC ≠ BF) : AC ∩ BF = Q := sorry
def intersection_EC_DF (EC DF P : Type*) (h4 : EC ≠ DF) : EC ∩ DF = P := sorry
def on_segments (D F B Q P R S T: Type*) : Prop := R ∈ seg DF ∧ S ∈ seg BF := sorry
def segments_equality (FR PD BQ FS: Type*) : FR = PD ∧ BQ = FS := sorry

-- Theorem statement about the bisection
theorem line_TC_bisects_DB
  (hexagon : is_cyclic_hexagon A B C D E F)
  (cond1 : BC_eq_EF B C E F)
  (cond2 : CD_eq_AF C D A F)
  (cond3 : intersection_AC_BF C F Q)
  (cond4 : intersection_EC_DF E C P)
  (cond5 : on_segments D F B Q P R S T)
  (cond6 : segments_equality FR PD BQ FS)
  :
  (line_TC_bisects_DB T C D B) :=
sorry -- Proof omitted

end line_TC_bisects_DB_l658_658924


namespace basement_water_pumping_l658_658752

noncomputable def pump_time (length width depth : ℕ) (pump_rate num_pumps : ℕ) (cubic_feet_to_gallons : ℕ → ℕ) : ℕ :=
  let volume_cubic_feet := depth * length * width
  let volume_gallons := cubic_feet_to_gallons volume_cubic_feet
  let total_pumping_rate := pump_rate * num_pumps
  let time := volume_gallons / total_pumping_rate
  time

theorem basement_water_pumping :
  let length := 30
  let width := 40
  let depth := 2
  let pump_rate := 12
  let num_pumps := 2
  let cubic_feet_to_gallons := λ x : ℕ, x * 75 / 10
  pump_time length width depth pump_rate num_pumps cubic_feet_to_gallons = 750 :=
by
  sorry

end basement_water_pumping_l658_658752


namespace silver_zinc_battery_statements_l658_658407

-- Definitions based on conditions
def reaction1 := "Zn + 2OH⁻ - 2e⁻ = Zn(OH)₂"
def reaction2 := "Ag₂O + H₂O + 2e⁻ = 2Ag + 2OH⁻"

-- Prove that given the reactions, the correct statements are ①, ② and ③, but not ④.
theorem silver_zinc_battery_statements :
  (reaction1 ∧ reaction2) →
  (1: Zinc is the negative electrode, \(\rm{Ag_{2}O}\) is the positive electrode) ∧
  (2: During discharge, the concentration of \(\rm{OH^{-}}\) near the positive electrode increases) ∧
  (3: During operation, the current flows from the \(\rm{Ag_{2}O}\) electrode through the external circuit to the \(\rm{Zn}\) electrode) ∧
  ¬ (4: In the solution, anions move towards the positive electrode, and cations move towards the negative electrode) :=
by
  -- proof goes here
  sorry

end silver_zinc_battery_statements_l658_658407


namespace range_H_l658_658349

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_H : set.range H = {-4, 4} := 
by
  sorry

end range_H_l658_658349


namespace sufficient_but_not_necessary_condition_l658_658377

theorem sufficient_but_not_necessary_condition (x y : ℝ) (h : x < y ∧ y < 0) :
  x^2 > y^2 := 
by
  sorry

example : ¬(∀ x y : ℝ, x^2 > y^2 → x < y ∧ y < 0) :=
by
  use 2
  use 1
  intro h
  linarith

end sufficient_but_not_necessary_condition_l658_658377


namespace find_k_and_b_l658_658591

variables (k b : ℝ)

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (k * p.1, p.2 + b)

theorem find_k_and_b
  (h : f k b (6, 2) = (3, 1)) :
  k = 2 ∧ b = -1 :=
by {
  -- proof steps would go here
  sorry
}

end find_k_and_b_l658_658591


namespace regular_tetrahedron_triangles_l658_658164

theorem regular_tetrahedron_triangles :
  let vertices := 4
  ∃ triangles : ℕ, (triangles = Nat.choose vertices 3) ∧ (triangles = 4) :=
by {
  let vertices := 4,
  use Nat.choose vertices 3,
  split,
  { 
    refl,
  },
  {
    norm_num,
  }
}

end regular_tetrahedron_triangles_l658_658164


namespace graph_shift_cos_sin_l658_658138

theorem graph_shift_cos_sin {φ : ℝ} (hφ : |φ| ≤ π/2)
  (h_symm : ∀ x, cos (2 * (8 * π / 3 - x) + φ) = -cos (2 * x + φ)) :
  ∃ c : ℝ, c = π / 6 ∧ ∀ x, sin (2 * (x + c)) = cos (2 * x + φ) :=
by
  sorry

end graph_shift_cos_sin_l658_658138


namespace x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l658_658184

theorem x_plus_inv_x_eq_two_implies_x_pow_six_eq_one
  (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 :=
sorry

end x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l658_658184


namespace part1_part2_l658_658483

section Problem

variable (α : ℝ)

def f (α : ℝ) : ℝ :=
  (sqrt ((1 - sin α) / (1 + sin α)) + sqrt ((1 + sin α) / (1 - sin α))) * cos(α)^3 + 2*sin ((π/2) + α) * cos ((3*π/2) + α)

theorem part1 (h1 : tan α = 2) (h2 : π ≤ α ∧ α ≤ 3 * π / 2) : f α = 2 / 5 :=
  sorry

theorem part2 (h1 : f α = 2 / 5 * cos α) (h2 : π ≤ α ∧ α ≤ 3 * π / 2) : tan α = 3 / 4 :=
  sorry

end Problem

end part1_part2_l658_658483


namespace valid_elixir_combinations_l658_658027

theorem valid_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := herbs * crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  total_combinations - incompatible_combinations = 18 :=
by
  sorry

end valid_elixir_combinations_l658_658027


namespace exists_polynomial_l658_658976

theorem exists_polynomial (m : ℤ) (n : ℕ) (hn : 0 < n) :
  ∃ p : polynomial ℤ, ∃ I : set ℝ, (∃ a b : ℝ, a < b ∧ b - a = 1 / n ∧ I = set.Icc a b) ∧
  ∀ x ∈ I, abs (polynomial.eval x p - m / n) < 1 / n^2 :=
by
  sorry

end exists_polynomial_l658_658976


namespace percentage_voting_for_biff_equals_45_l658_658985

variable (total : ℕ) (votingForMarty : ℕ) (undecidedPercent : ℝ)

theorem percentage_voting_for_biff_equals_45 :
  total = 200 →
  votingForMarty = 94 →
  undecidedPercent = 0.08 →
  let totalDecided := (1 - undecidedPercent) * total
  let votingForBiff := totalDecided - votingForMarty
  let votingForBiffPercent := (votingForBiff / total) * 100
  votingForBiffPercent = 45 :=
by
  intros h1 h2 h3
  let totalDecided := (1 - 0.08 : ℝ) * 200
  let votingForBiff := totalDecided - 94
  let votingForBiffPercent := (votingForBiff / 200) * 100
  sorry

end percentage_voting_for_biff_equals_45_l658_658985


namespace min_value_expr_l658_658100

theorem min_value_expr : ∀ (x : ℝ), 0 < x ∧ x < 4 → ∃ y : ℝ, y = (1 / (4 - x) + 2 / x) ∧ y = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_expr_l658_658100


namespace sum_of_consecutive_integers_345_l658_658216

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end sum_of_consecutive_integers_345_l658_658216


namespace statement_C_l658_658911

theorem statement_C (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end statement_C_l658_658911


namespace triangles_in_figure_l658_658178

-- Define the conditions of the problem.
def bottom_row_small := 4
def next_row_small := 3
def following_row_small := 2
def topmost_row_small := 1

def small_triangles := bottom_row_small + next_row_small + following_row_small + topmost_row_small

def medium_triangles := 3
def large_triangle := 1

def total_triangles := small_triangles + medium_triangles + large_triangle

-- Lean proof statement that the total number of triangles is 14
theorem triangles_in_figure : total_triangles = 14 :=
by
  unfold total_triangles
  unfold small_triangles
  unfold bottom_row_small next_row_small following_row_small topmost_row_small
  unfold medium_triangles large_triangle
  sorry

end triangles_in_figure_l658_658178


namespace optionB_incorrect_l658_658417

/-- Definitions of sets A and B, and a mapping function -/
variable {A B : Type} (f : A → B)

/-- Condition 1: Every element in A has an image in B -/
axiom condition1 (a : A) : ∃ b : B, f a = b

/-- Condition 2: An element in B may have more than one preimage in A -/
axiom condition4 (b : B) : ∃ (a1 a2 : A), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b

/-- Proof problem: The images of two different elements in A must be different in B is incorrect -/
theorem optionB_incorrect : ¬ (∀ a1 a2 : A, a1 ≠ a2 → f a1 ≠ f a2) :=
by
  intros h
  sorry

end optionB_incorrect_l658_658417


namespace task1_intervals_of_monotonicity_task2_range_of_m_l658_658880

-- Define the function f(x) = x^3 - 3ax - 1
def f (x a : ℝ) : ℝ := x^3 - 3 * a * x - 1

-- First task: Prove the intervals of monotonicity of f(x)

theorem task1_intervals_of_monotonicity (a : ℝ) (h : a ≠ 0) :
  (if a < 0 then ∀ x : ℝ, deriv (f x a) x > 0 else 
   if a > 0 then ∀ x : ℝ, (x < -real.sqrt a ∨ x > real.sqrt a) → deriv (f x a) x > 0 ∧
                                      (-real.sqrt a < x ∧ x < real.sqrt a) → deriv (f x a) x < 0 else
   false) :=
by sorry

-- Define the condition for extremum
def has_extremum (a : ℝ) (h : a ≠ 0) := 3 * (-1)^2 - 3 * a = 0

-- Second task: Prove range of values for m

theorem task2_range_of_m (a : ℝ) (m : ℝ) (h : has_extremum a (by linarith)) :
  f (-1, 1) -3 ∧ f (1, 1) 1 ∧ (∃ x₁ x₂ x₃ : ℝ, f (x₁, 1) m ∧ f (x₂, 1) m ∧ f (x₃, 1) m) → 
  m ∈ Ioo (-3 : ℝ) (1 : ℝ)) :=
by sorry

end task1_intervals_of_monotonicity_task2_range_of_m_l658_658880


namespace equation1_solution_equation2_solution_l658_658656

-- Equation 1: x^2 + 2x - 8 = 0 has solutions x = -4 and x = 2.
theorem equation1_solution (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := by
  sorry

-- Equation 2: 2(x+3)^2 = x(x+3) has solutions x = -3 and x = -6.
theorem equation2_solution (x : ℝ) : 2 * (x + 3)^2 = x * (x + 3) ↔ x = -3 ∨ x = -6 := by
  sorry

end equation1_solution_equation2_solution_l658_658656


namespace minimize_y_l658_658502

def y (a : ℕ) (x : ℕ) : ℕ := (a + 2) * x^2 - 2 * (a^2 - 1) * x + 1

theorem minimize_y (a : ℕ) (x : ℕ) (hx : 0 < x) (ha : 0 < a) : 
  x = if 1 < a ∧ a < 4 then a - 1
      else if a = 4 then 2 ∨ x = 3
      else if 4 < a then a - 2
      else x := 
sorry

end minimize_y_l658_658502


namespace correct_proposition_l658_658994

noncomputable def K_squared : ℝ := sorry
noncomputable def S_squared : ℝ := sorry
noncomputable def R_squared : ℝ := sorry

def proposition1 (K_squared : ℝ) : Prop :=
  ∀ x y, (K_squared x y > 0) → ¬(x is related to y)

def proposition2 (S_squared : ℝ) : Prop :=
  ∀ residuals, (S_squared residuals > 0) → (fitting effect is better)

def proposition3 (R_squared : ℝ) : Prop :=
  ∀ degree_of_fit, (R_squared degree_of_fit > 0) → (degree_of_fit is better)

theorem correct_proposition :
  ¬(proposition1 K_squared) ∧ ¬(proposition2 S_squared) ∧ proposition3 R_squared :=
by
  sorry

end correct_proposition_l658_658994


namespace problem_1_problem_2_l658_658604

noncomputable def set_A : Set ℝ := {x : ℝ | x^2 - 9*x + 18 ≥ 0}
noncomputable def set_B : Set ℝ := {x : ℝ | -2 < x ∧ x < 9}
noncomputable def set_C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a+1}
noncomputable def set_C_complement_A : Set ℝ := {x : ℝ | 3 < x ∧ x < 6}

theorem problem_1 : (set_C_complement_A ∩ set_B) = {x : ℝ | 3 < x ∧ x < 6} :=
by
  -- proof goes here
  sorry

theorem problem_2 (a : ℝ) (h : set_C(a) ⊆ set_B) : -2 ≤ a ∧ a ≤ 8 :=
by
  -- proof goes here
  sorry

end problem_1_problem_2_l658_658604


namespace log_cut_piece_weight_l658_658250

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end log_cut_piece_weight_l658_658250


namespace estimate_vehicles_in_section_l658_658688

-- Define the constants and conditions
def northbound_speed := 65 -- mph
def southbound_speed := 55 -- mph
def vehicles_passed := 30 -- number of vehicles passed in 10 minutes
def time_interval := 10 / 60 -- hours

-- Prove that the number of southbound vehicles in a 150-mile section of the highway is 225
theorem estimate_vehicles_in_section :
    let distance_northbound := northbound_speed * time_interval in
    let relative_speed := northbound_speed + southbound_speed in
    let relative_distance := relative_speed * time_interval in
    let vehicle_density := vehicles_passed / relative_distance in
    let section_length := 150 in
    let estimated_vehicles := vehicle_density * section_length in
    estimated_vehicles = 225 :=
by
    sorry

end estimate_vehicles_in_section_l658_658688


namespace total_revenue_l658_658394

theorem total_revenue (price_adult price_child : ℕ) (total_tickets child_tickets : ℕ) : 
  price_adult = 7 → price_child = 4 → total_tickets = 900 → child_tickets = 400 → 
  let adult_tickets := total_tickets - child_tickets in
  let revenue_adult := adult_tickets * price_adult in
  let revenue_child := child_tickets * price_child in
  let total_revenue := revenue_adult + revenue_child in
  total_revenue = 5100 := by
  intros h1 h2 h3 h4
  let adult_tickets := 900 - 400
  let revenue_adult := adult_tickets * 7
  let revenue_child := 400 * 4
  let total_revenue := revenue_adult + revenue_child
  sorry

end total_revenue_l658_658394


namespace problem_1_problem_2_problem_3_l658_658470

noncomputable def z (m : ℝ) : ℂ := (m^2 - 1) + (m^2 - 3m + 2) * complex.i

-- (1) If z = 0, then m = 1
theorem problem_1 (m : ℝ) (h : z m = 0) : m = 1 :=
sorry

-- (2) If z is purely imaginary, then m = -1
theorem problem_2 (m : ℝ) (h_real : z m.re = 0) (h_imag : z m.im ≠ 0) : m = -1 :=
sorry

-- (3) If z is in the second quadrant, then m ∈ (-1, 1)
theorem problem_3 (m : ℝ) (h1 : (m^2 - 1) < 0) (h2 : (m^2 - 3m + 2) > 0) : -1 < m ∧ m < 1 :=
sorry

end problem_1_problem_2_problem_3_l658_658470


namespace pascal_fifth_element_row_20_l658_658716

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l658_658716


namespace cost_of_soccer_basketball_balls_max_basketballs_l658_658332

def cost_of_balls (x y : ℕ) : Prop :=
  (7 * x = 5 * y) ∧ (40 * x + 20 * y = 3400)

def cost_constraint (x y m : ℕ) : Prop :=
  (x = 50) ∧ (y = 70) ∧ (70 * m + 50 * (100 - m) ≤ 6300)

theorem cost_of_soccer_basketball_balls (x y : ℕ) (h : cost_of_balls x y) : x = 50 ∧ y = 70 :=
  by sorry

theorem max_basketballs (x y m : ℕ) (h : cost_constraint x y m) : m ≤ 65 :=
  by sorry

end cost_of_soccer_basketball_balls_max_basketballs_l658_658332


namespace find_a_l658_658188

open Complex

theorem find_a (a : ℝ) (h : (⟨a, 1⟩ * ⟨1, -a⟩ = 2)) : a = 1 :=
sorry

end find_a_l658_658188


namespace sum_of_x_coordinates_l658_658991

def points : List (ℝ × ℝ) := [(4, 15), (7, 25), (13, 40), (19, 45), (21, 55), (25, 60)]

def line (x : ℝ) : ℝ := 3 * x + 5

theorem sum_of_x_coordinates :
  (∑ (p : ℝ × ℝ) in points.filter (λ p, p.snd > line p.fst), p.fst) = 0 :=
by
  sorry

end sum_of_x_coordinates_l658_658991


namespace polynomial_remainder_l658_658453

theorem polynomial_remainder (x : ℝ) :
  (x^4 + 3 * x^2 - 4) % (x^2 + 2) = x^2 - 4 :=
sorry

end polynomial_remainder_l658_658453


namespace solve_rational_numbers_l658_658082

theorem solve_rational_numbers:
  ∃ (a b c d : ℚ),
    8 * a^2 - 3 * b^2 + 5 * c^2 + 16 * d^2 - 10 * a * b + 42 * c * d + 18 * a + 22 * b - 2 * c - 54 * d = 42 ∧
    15 * a^2 - 3 * b^2 + 21 * c^2 - 5 * d^2 + 4 * a * b + 32 * c * d - 28 * a + 14 * b - 54 * c - 52 * d = -22 ∧
    a = 4 / 7 ∧ b = 19 / 7 ∧ c = 29 / 19 ∧ d = -6 / 19 :=
  sorry

end solve_rational_numbers_l658_658082


namespace min_sum_ab_l658_658587

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l658_658587


namespace primes_between_4900_8100_l658_658907

theorem primes_between_4900_8100 :
  ∃ (count : ℕ),
  count = 5 ∧ ∀ n : ℤ, 70 < n ∧ n < 90 ∧ (n * n > 4900 ∧ n * n < 8100 ∧ Prime n) → count = 5 :=
by
  sorry

end primes_between_4900_8100_l658_658907


namespace square_ratio_condition_l658_658339

theorem square_ratio_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  b ≥ (a / 2) * (Real.sqrt 3 + 1) ↔ 
  ∃ M : ℝ × ℝ, ∀ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, -- Note: ℝ × ℝ represents a 2D point (P, Q)
  (P ∈ square_with_side a) ∧ (Q ∈ square_with_side b) ∧ equilateral_triangle M P Q :=
sorry

end square_ratio_condition_l658_658339


namespace find_f_at_one_l658_658133

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * x^3 + n * x + 1

theorem find_f_at_one (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : f (-1) m n = 5) : f (1) m n = 7 :=
by
  -- proof goes here
  sorry

end find_f_at_one_l658_658133


namespace polynomial_characterization_l658_658741

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization_l658_658741


namespace range_of_a_l658_658884

-- Mathematical statement
theorem range_of_a (a : ℝ) : 
  (∃ x_max x_min : ℝ, (3 * x_max^2 + 2 * (a + 1) * x_max + (a + 1) = 0) ∧ (3 * x_min^2 + 2 * (a + 1) * x_min + (a + 1) = 0) ∧ x_max ≠ x_min) ↔ (a < -1 ∨ a > 2) :=
begin
  sorry
end

end range_of_a_l658_658884


namespace order_of_abcd_l658_658535

-- Define the rational numbers a, b, c, d
variables {a b c d : ℚ}

-- State the conditions as assumptions
axiom h1 : a + b = c + d
axiom h2 : a + d < b + c
axiom h3 : c < d

-- The goal is to prove the correct order of a, b, c, d
theorem order_of_abcd (a b c d : ℚ) (h1 : a + b = c + d) (h2 : a + d < b + c) (h3 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end order_of_abcd_l658_658535


namespace angle_C_in_triangle_l658_658569

theorem angle_C_in_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2 + ab) : 
  ∠(angle A B C) = 120 :=
sorry

end angle_C_in_triangle_l658_658569


namespace smallest_of_5_consecutive_natural_numbers_sum_100_l658_658686

theorem smallest_of_5_consecutive_natural_numbers_sum_100
  (n : ℕ)
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) :
  n = 18 := sorry

end smallest_of_5_consecutive_natural_numbers_sum_100_l658_658686


namespace f_continuous_at_all_points_l658_658271

noncomputable def f : ℝ → ℝ
| x := if x = 0 then 0 else x * real.sin (1 / x)

theorem f_continuous_at_all_points : ∀ x, continuous_at f x :=
by
  sorry

end f_continuous_at_all_points_l658_658271


namespace probability_of_2a_ge_5b_l658_658760

noncomputable def probability_favorable_outcomes : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes :=
    (finset.range 6).sum (λ b, (finset.range 6).count (λ a, 2 * (a + 1) ≥ 5 * (b + 1)))
  favorable_outcomes / total_outcomes

theorem probability_of_2a_ge_5b : probability_favorable_outcomes = 1 / 6 :=
by sorry

end probability_of_2a_ge_5b_l658_658760


namespace max_large_sculptures_l658_658446

theorem max_large_sculptures (x y : ℕ) (h1 : 1 * x = x) 
  (h2 : 3 * y = y + y + y) 
  (h3 : ∃ n, n = (x + y) / 2) 
  (h4 : x + 3 * y + (x + y) / 2 ≤ 30) 
  (h5 : x > y) : 
  y ≤ 4 := 
sorry

end max_large_sculptures_l658_658446


namespace fido_yard_fraction_l658_658080

theorem fido_yard_fraction (a b : ℕ) (r : ℝ) (hex_area fraction_reachable : ℝ) :
  let hex_area := 3 * r^2 * Real.sqrt 3 / 2,
      fraction_reachable := 2 * Real.pi / (3 * Real.sqrt 3)
  in fraction_reachable = (Real.sqrt (a : ℝ) / b) * Real.pi -> 
     a * b = 18 := 
sorry

end fido_yard_fraction_l658_658080


namespace somu_current_age_l658_658739

variable (S F : ℕ)

theorem somu_current_age
  (h1 : S = F / 3)
  (h2 : S - 10 = (F - 10) / 5) :
  S = 20 := by
  sorry

end somu_current_age_l658_658739


namespace no_sum_2015_l658_658959

theorem no_sum_2015 (x a : ℤ) : 3 * x + 3 * a ≠ 2015 := by
  sorry

end no_sum_2015_l658_658959


namespace problem_part_1_problem_part_2_problem_part_3_l658_658117

-- Definitions based on the conditions
def sequence_a (t : ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → let S := (λ n : ℕ, ∑ i in Finset.range n, a i) in 
  S n = t * (S n - a n + 1)

def forms_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) = r * a n

def sequence_b (t : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) := let S := (λ n : ℕ, ∑ i in Finset.range n, a i) in 
  ∀ n : ℕ, n > 0 → b n = a n ^ 2 + S n * a n

def sequence_b_geometric (b : ℕ → ℝ) := forms_geometric_sequence b

def sequence_c (a : ℕ → ℝ) (c : ℕ → ℝ) := ∀ n : ℕ, n > 0 → c n = 4 * a n + 1

def T_n (c : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, c i

def inequality_holds (c : ℕ → ℝ) (k : ℝ) := ∀ n : ℕ, n > 0 → 12 * k / (4 + n - T_n c n) ≥ 2 * n - 7

-- Theorem statements
theorem problem_part_1 (t : ℝ) (h_t_nonzero : t ≠ 0) (h_t_ne_one : t ≠ 1) (a : ℕ → ℝ) 
  (h_seq_a : sequence_a t a) : forms_geometric_sequence a :=
sorry

theorem problem_part_2 (t : ℝ) (h_t_nonzero : t ≠ 0) (h_t_ne_one : t ≠ 1) (a : ℕ → ℝ) 
  (b : ℕ → ℝ) (h_seq_a : sequence_a t a) (h_seq_b : sequence_b t a b) 
  (h_seq_b_geometric : sequence_b_geometric b): t = 1 / 2 :=
sorry

theorem problem_part_3 (a : ℕ → ℝ) (c : ℕ → ℝ) (h_part_2 : ∀ t, t ≠ 0 → t ≠ 1 → 
  (sequence_a t a → sequence_b_geometric (λ n, (t ^ n) ^ 2 + S (t ^ n) * (t ^ n)) → 
  t = 1 / 2)) (h_seq_c : sequence_c a c) : 
  (c_seq_c_and_geometric_a_implies_k_ge_one_over_32 : ∀ k : ℝ, 
  inequality_holds c k → k ≥ 1 / 32) :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l658_658117


namespace vector_norm_range_l658_658475

variables {V : Type*} [inner_product_space ℝ V]

variables (a b c : V)
variables (h₁ : ∥a∥ = real.sqrt 5) 
variables (h₂ : ∥b∥ = real.sqrt 5)
variables (h₃ : ∥c∥ = 1)
variables (h₄ : inner (a - c) (b - c) = 0)

theorem vector_norm_range : 2 ≤ ∥a - b∥ ∧ ∥a - b∥ ≤ 4 :=
sorry

end vector_norm_range_l658_658475


namespace part_I_part_II_l658_658139

noncomputable def f (x a : ℝ) := sin x - a * cos x
noncomputable def g (x : ℝ) := (λ x, sin x - cos x) x * (λ x, sin x - cos x) (-x) + 2 * sqrt 3 * sin x * cos x

theorem part_I {a : ℝ} (h : f (π / 4) a = 0) : a = 1 :=
by
  simp [f] at h
  sorry

theorem part_II : set.range (g ∘ id) = set.Icc (-1) 2 :=
by
  sorry

end part_I_part_II_l658_658139


namespace cistern_length_l658_658757

theorem cistern_length (L : ℝ) (H : 0 < L) :
    (∃ (w d A : ℝ), w = 14 ∧ d = 1.25 ∧ A = 233 ∧ A = L * w + 2 * L * d + 2 * w * d) →
    L = 12 :=
by
  sorry

end cistern_length_l658_658757


namespace more_people_this_week_l658_658226

-- Define the conditions
variables (second_game first_game third_game : ℕ)
variables (total_last_week total_this_week : ℕ)

-- Conditions
def condition1 : Prop := second_game = 80
def condition2 : Prop := first_game = second_game - 20
def condition3 : Prop := third_game = second_game + 15
def condition4 : Prop := total_last_week = 200
def condition5 : Prop := total_this_week = second_game + first_game + third_game

-- Theorem statement
theorem more_people_this_week (h1 : condition1)
                             (h2 : condition2)
                             (h3 : condition3)
                             (h4 : condition4)
                             (h5 : condition5) : total_this_week - total_last_week = 35 :=
sorry

end more_people_this_week_l658_658226


namespace proposition_A_iff_proposition_B_l658_658274

def interior_angle_of_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

def one_angle_is_60 {A B C : ℝ} : Prop :=
A = 60 ∨ B = 60 ∨ C = 60

def angles_form_arithmetic_sequence {A B C : ℝ} : Prop :=
∃ d, (B = A + d ∧ C = A + 2 * d) ∨ (A = B + d ∧ C = B + 2 * d) ∨ (A = C + d ∧ B = C + 2 * d)

theorem proposition_A_iff_proposition_B {A B C : ℝ} :
  interior_angle_of_triangle A B C →
  (one_angle_is_60 ∩ interior_angle_of_triangle A B C ↔ angles_form_arithmetic_sequence ∩ interior_angle_of_triangle A B C) :=
by sorry

end proposition_A_iff_proposition_B_l658_658274


namespace find_fprime_one_l658_658131

theorem find_fprime_one (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + log x) :
  f' 1 = -1 :=
sorry

end find_fprime_one_l658_658131


namespace simplify_cosine_tangent_product_of_cosines_l658_658295

-- Problem 1
theorem simplify_cosine_tangent :
  Real.cos 40 * (1 + Real.sqrt 3 * Real.tan 10) = 1 :=
sorry

-- Problem 2
theorem product_of_cosines :
  (Real.cos (2 * Real.pi / 7)) * (Real.cos (4 * Real.pi / 7)) * (Real.cos (6 * Real.pi / 7)) = 1 / 8 :=
sorry

end simplify_cosine_tangent_product_of_cosines_l658_658295


namespace unique_rational_point_enlargement_rotation_l658_658011

noncomputable def is_rational_point (P : ℝ × ℝ) : Prop :=
∃ (a b : ℚ), P = (a, b)

theorem unique_rational_point_enlargement_rotation
  (A B C D : ℝ × ℝ)
  (hA : is_rational_point A)
  (hB : is_rational_point B)
  (hC : is_rational_point C)
  (hD : is_rational_point D)
  (hne : A ≠ B ∨ C ≠ D)
  (hnparallel : (A.1 - B.1) * (C.2 - D.2) ≠ (C.1 - D.1) * (A.2 - B.2))
  : ∃! P : ℝ × ℝ, is_rational_point P ∧ 
    ∃ (k : ℝ) (θ : ℝ),
    k ≠ 0 ∧ 
    (∀ (PA PB PC PD : ℝ × ℝ), 
       PA = (fst P + k * ((fst A - fst P) * cos θ - (snd A - snd P) * sin θ)) ∧
       PB = (fst P + k * ((fst B - fst P) * cos θ - (snd B - snd P) * sin θ)) ∧
       PC = (fst P + k * ((fst C - fst P) * cos θ - (snd C - snd P) * sin θ)) ∧
       PD = (fst P + k * ((fst D - fst P) * cos θ - (snd D - snd P) * sin θ)) ∧
       PPCD PA PB = PPCD PC PD) :=
begin
  sorry
end

end unique_rational_point_enlargement_rotation_l658_658011


namespace original_deck_card_count_l658_658759

theorem original_deck_card_count (r b : ℕ) 
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
sorry

end original_deck_card_count_l658_658759


namespace pascal_triangle_row_20_element_5_l658_658706

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l658_658706


namespace yellow_pill_cost_22_5_l658_658040

-- Definitions based on conditions
def number_of_days := 3 * 7
def total_cost := 903
def daily_cost := total_cost / number_of_days
def blue_pill_cost (yellow_pill_cost : ℝ) := yellow_pill_cost - 2

-- Prove that the cost of one yellow pill is 22.5 dollars
theorem yellow_pill_cost_22_5 : 
  ∃ (yellow_pill_cost : ℝ), 
    number_of_days = 21 ∧
    total_cost = 903 ∧ 
    (∀ yellow_pill_cost, daily_cost = yellow_pill_cost + blue_pill_cost yellow_pill_cost → yellow_pill_cost = 22.5) :=
by 
  sorry

end yellow_pill_cost_22_5_l658_658040


namespace train_pass_time_l658_658405

theorem train_pass_time (train_length platform_length : ℕ) (speed_kmph : ℕ)
  (convert_factor : ℝ) (h1 : train_length = 140) (h2 : platform_length = 260)
  (h3 : speed_kmph = 60) (h4 : convert_factor = 1/3.6) : 
  let total_distance := train_length + platform_length in
  let speed_mps := speed_kmph * convert_factor in
  let time := total_distance / speed_mps in
  time ≈ 24 :=
by
  sorry

end train_pass_time_l658_658405


namespace minimize_I_l658_658525

def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

def I (p : ℝ) : ℝ := max (9 * p - 5) (-5 * p + 2)

theorem minimize_I : 
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 1 → (I p = I (1/2)) → p = 1/2 := 
by 
  sorry

end minimize_I_l658_658525


namespace range_of_H_l658_658353

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_of_H_l658_658353


namespace remaining_amount_to_be_paid_l658_658185

-- Define the conditions
def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 80

-- Define the total purchase price based on the conditions
def total_price : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_price - deposit_amount

-- State the theorem
theorem remaining_amount_to_be_paid : remaining_amount = 720 := by
  sorry

end remaining_amount_to_be_paid_l658_658185


namespace value_of_nested_f_l658_658970

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_nested_f : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end value_of_nested_f_l658_658970


namespace quadratic_solution_correctness_l658_658427

-- Definition of the quadratic equation and its solution
def quadratic_eq (x : ℝ) : Prop := x^2 + 4 * x - 1 = 0

-- Correct solutions
def solution1 (x : ℝ) : Prop := x = -2 + sqrt 5
def solution2 (x : ℝ) : Prop := x = -2 - sqrt 5

-- The theorem to prove
theorem quadratic_solution_correctness (x : ℝ) : quadratic_eq x ↔ solution1 x ∨ solution2 x := by
  sorry

end quadratic_solution_correctness_l658_658427


namespace ellipse_eqn_l658_658006

theorem ellipse_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
    (h_line : ∃ x₁ x₂ y₁ y₂ : ℝ, let P := (x₁ + x₂) / 2, (y₁ + y₂) / 2 in
    (y₁ - y₂) / (x₁ - x₂) = -1 ∧
    ∃ (c : ℝ), c = sqrt(3) ∧ a^2 = b^2 + c^2 ∧ (x₁, y₁) ∈ M a b ∧ (x₂, y₂) ∈ M a b)
    (h_slope : ∃ (P : ℝ × ℝ), P = ((P.1 + P.2) / 2, (P.1 + P.2) / 2) ∧ P.2 / P.1 = 1 / 2) : 
    (∀ x y : ℝ, (x,y) ∈ M a b ↔ x^2 / 6 + y^2 / 3 = 1) := 
begin
    sorry
end

def M (a b : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

end ellipse_eqn_l658_658006


namespace simplify_rationalize_expr_l658_658641

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l658_658641


namespace distance_center_to_line_eq_sqrt2_l658_658821

noncomputable def circle_center : ℝ × ℝ :=
  let h := 1
  let k := -2
  (h, k)

def line_equation (x y : ℝ) : Prop := x - y = 1

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / real.sqrt (A^2 + B^2)

theorem distance_center_to_line_eq_sqrt2 :
  distance_point_to_line (circle_center.fst) (circle_center.snd) 1 (-1) (-1) = real.sqrt 2 :=
by
  sorry

end distance_center_to_line_eq_sqrt2_l658_658821


namespace magic_square_common_sum_is_neg_2_l658_658049

open Int

def common_sum_4x4 (mat : Matrix (Fin 4) (Fin 4) ℤ) : ℤ :=
  (∑ i, ∑ j, mat i j) / 4

def is_magic_square (mat : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  ∀ i j, (∑ k, mat i k) = (∑ k, mat k j) ∧ 
  (∑ k, mat k k) = common_sum_4x4 mat ∧ 
  (∑ k, mat k (3 - k)) = common_sum_4x4 mat

theorem magic_square_common_sum_is_neg_2 
  (mat : Matrix (Fin 4) (Fin 4) ℤ)
  (h1 : ∀ i j, mat i j ∈ Icc (-8) 7)
  (h2 : is_magic_square mat) :
  common_sum_4x4 mat = -2 :=
  sorry

end magic_square_common_sum_is_neg_2_l658_658049


namespace OMO_sum_l658_658805

theorem OMO_sum (EVIL LOVE IMO OMO : ℚ)
  (hEVIL : EVIL = 5 / 31)
  (hLOVE : LOVE = 6 / 29)
  (hIMO : IMO = 7 / 3)
  (hOMO : OMO = (LOVE / EVIL) * IMO) :
  let m := 434
  let n := 145
  (m.gcd n = 1) → (OMO = m / n ∧ m + n = 579) :=
by {
  intros,
  sorry
}

end OMO_sum_l658_658805


namespace regular_polygon_sides_l658_658917

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end regular_polygon_sides_l658_658917


namespace probability_remainder_is_4_5_l658_658422

def probability_remainder_1 (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1)

theorem probability_remainder_is_4_5 : 
  ∀ N, N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1) → (number_of_successful_outcomes / total_outcomes = 4 / 5) :=
sorry

end probability_remainder_is_4_5_l658_658422


namespace students_in_class_l658_658299

theorem students_in_class (n S : ℕ) 
    (h1 : S = 15 * n)
    (h2 : (S + 56) / (n + 1) = 16) : n = 40 :=
by
  sorry

end students_in_class_l658_658299


namespace monotonicity_of_f_l658_658590

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonicity_of_f : 
  ∀ x : ℝ, (0 < x) ∧ (x < 1) → (f' x < 0) := by
  sorry

end monotonicity_of_f_l658_658590


namespace number_of_elements_in_B_l658_658897

def A : set ℕ := {1, 2, 3, 4, 5}
def B : set (ℕ × ℕ) := {(x, y) | x ∈ A ∧ y ∈ A ∧ (x - y) ∈ A}

theorem number_of_elements_in_B : (#{ p : ℕ × ℕ | p ∈ B } = 10) :=
sorry

end number_of_elements_in_B_l658_658897


namespace Emilia_needs_more_cartons_l658_658069

theorem Emilia_needs_more_cartons :
  ∀ (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ), 
    total_needed = 42 →
    strawberries = 2 →
    blueberries = 7 →
    total_needed - (strawberries + blueberries) = 33 :=
by
  intros total_needed strawberries blueberries h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Emilia_needs_more_cartons_l658_658069


namespace r_squared_eq_one_l658_658629

-- Define the parabola and points A, B, and C
def parabola (x : ℝ) : ℝ := x^2

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C (x : ℝ) : ℝ × ℝ := (x, parabola x)

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the condition AC = CB
def AC_eq_CB (x : ℝ) : Prop :=
  distance (A) (C x) = distance (C x) (B)

-- Prove that r^2 = 1 under the above conditions
theorem r_squared_eq_one : ∃ x : ℝ, AC_eq_CB x → (distance (A) (C x))^2 = 1 :=
by
  sorry

end r_squared_eq_one_l658_658629


namespace equal_distribution_l658_658829

def earnings : List ℕ := [30, 35, 45, 55, 65]

def total_earnings : ℕ := earnings.sum

def equal_share (total: ℕ) : ℕ := total / earnings.length

def redistribution_amount (earner: ℕ) (equal: ℕ) : ℕ := earner - equal

theorem equal_distribution :
  redistribution_amount 65 (equal_share total_earnings) = 19 :=
by
  sorry

end equal_distribution_l658_658829


namespace percentage_error_l658_658770

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem percentage_error (n : ℕ) (h : n = factorial 5) : 
  let correct_result := 3 * n in
  let incorrect_result := n / 3 in
  let error := correct_result - incorrect_result in
  let error_percentage := ((error: ℚ) / correct_result) * 100 in
  error_percentage = 89 := 
by
  sorry

end percentage_error_l658_658770


namespace value_of_expression_l658_658600

theorem value_of_expression {p q : ℝ} (hp : 3 * p^2 + 9 * p - 21 = 0) (hq : 3 * q^2 + 9 * q - 21 = 0) : 
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end value_of_expression_l658_658600


namespace alpha_plus_beta_l658_658500

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem alpha_plus_beta (α β : ℝ) (hα : 0 ≤ α) (hαβ : α < Real.pi) (hβ : 0 ≤ β) (hββ : β < Real.pi)
  (hα_neq_β : α ≠ β) (hf_α : f α = 1 / 2) (hf_β : f β = 1 / 2) : α + β = (7 * Real.pi) / 6 :=
by
  sorry

end alpha_plus_beta_l658_658500


namespace calculate_p_l658_658252

theorem calculate_p (f : ℝ) (w p : ℂ) (h_f : f = 8) (h_w : w = 9 - 75 * complex.I)
  (h_eq : f * p - w = 8000) : p = 1001.13 - 9.38 * complex.I := by
  sorry

end calculate_p_l658_658252


namespace coins_on_table_l658_658992

theorem coins_on_table (R r : ℝ) (n : ℕ) (h_pos_r : r > 0) (h_coins_nonoverlapping : ∀ i j, i ≠ j → (coincenter i - coincenter j).dist ≥ 2 * r) (h_no_more_coins : ∀ p, ¬ (∀ i, (p - coincenter i).dist ≥ 2 * r)) :
    R / r ≤ 2 * real.sqrt n + 1 := 
sorry

end coins_on_table_l658_658992


namespace sum_largest_odd_divisors_eq_n_squared_l658_658599

open Nat

def largestOddDivisor (x : ℕ) : ℕ :=
  if x % 2 = 1 then
    x
  else
    largestOddDivisor (x / 2)

theorem sum_largest_odd_divisors_eq_n_squared (n : ℕ) (h : n > 0) :
  (∑ k in Finset.range (n + 1), largestOddDivisor (k + n + 1)) = n * n :=
by
  sorry

end sum_largest_odd_divisors_eq_n_squared_l658_658599


namespace emilia_should_buy_more_l658_658070

-- Define the variables and conditions
variables (total_needed strawberries blueberries: ℕ)
-- Define the conditions
def needs_42_cartons : Prop := total_needed = 42
def has_strawberries : Prop := strawberries = 2
def has_blueberries  : Prop := blueberries = 7

-- Define the total cartons Emilia already has
def total_cartons_already_has : ℕ := strawberries + blueberries

-- Define the number of additional cartons she needs to buy
def additional_cartons_needed : ℕ := total_needed - total_cartons_already_has

-- Prove that the number of additional cartons needed is 33
theorem emilia_should_buy_more : 
  needs_42_cartons → has_strawberries → has_blueberries → additional_cartons_needed = 33 :=
by
  intros h1 h2 h3
  rw [needs_42_cartons, has_strawberries, has_blueberries] at *
  simp [total_cartons_already_has, additional_cartons_needed]
  sorry

end emilia_should_buy_more_l658_658070


namespace rotten_bananas_percentage_l658_658398

-- Define the conditions
def total_oranges := 600
def total_bananas := 400
def percentage_rotten_oranges := 15 / 100
def percentage_good_fruits := 89.8 / 100

-- Prove the percentage of bananas that were rotten is 3%
theorem rotten_bananas_percentage :
  let total_fruits := total_oranges + total_bananas in
  let good_oranges := total_oranges * (1 - percentage_rotten_oranges) in
  let total_good_fruits := total_fruits * percentage_good_fruits in
  let good_bananas := total_good_fruits - good_oranges in
  let rotten_bananas := total_bananas - good_bananas in
  (rotten_bananas / total_bananas) * 100 = 3 :=
by sorry

end rotten_bananas_percentage_l658_658398


namespace identify_counterfeit_l658_658553

-- Define the problem context.
def problem_context :=
  ∃ (coins : Finset ℕ), 
    cards coins = 17 ∧
    ∃ (counterfeit : Finset ℕ), 
      counterfeit.card = 2 ∧
      (∀ c ∈ counterfeit, ∀ c' ∈ genuine coins, c ≠ c')

-- Define the question as a theorem.
theorem identify_counterfeit (h : problem_context) : 
  ¬(∀ sequences : List (α × α), true) := sorry

end identify_counterfeit_l658_658553


namespace area_of_region_B_l658_658434

-- Define complex number z as x + yi
def z (x y : ℝ) : ℂ := x + y * complex.I

-- Define the region B's conditions
def region_B_condition1 (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 60 ∧ 0 ≤ y ∧ y ≤ 60

def region_B_condition2 (x y : ℝ) : Prop :=
  let norm_sq := x^2 + y^2 in
  60 * x ≤ norm_sq ∧ 60 * y ≤ norm_sq

-- Define the region B
def region_B (x y : ℝ) : Prop :=
  region_B_condition1 x y ∧ region_B_condition2 x y

-- Prove that the area of region B is 3600 - 450π
theorem area_of_region_B : ∃ (A : ℝ), A = 3600 - 450 * Real.pi ∧ 
  ∀ (x y : ℝ), region_B x y ↔ (0 ≤ x ∧ x ≤ 60 ∧ 0 ≤ y ∧ y ≤ 60 ∧
    (x - 30)^2 + y^2 ≥ 900 ∧ x^2 + (y - 30)^2 ≥ 900) :=
by {
  sorry
}

end area_of_region_B_l658_658434


namespace vertex_of_quadratic_function_l658_658669

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := 3 * (x + 4)^2 - 5

-- Statement: Prove the coordinates of the vertex of the quadratic function
theorem vertex_of_quadratic_function : vertex quadratic_function = (-4, -5) :=
sorry

end vertex_of_quadratic_function_l658_658669


namespace part1_part2_part3_l658_658469

-- Definition of complex number z
def z (m : ℝ) : ℂ := complex.mk (1 - m^2) (m^2 - 3 * m + 2)

-- I) If z = 0, find the value of m
theorem part1 (m : ℝ) (h : z m = 0) : m = 1 := 
by sorry

-- II) If z is purely imaginary, find the value of m
theorem part2 (m : ℝ) (h : z m = complex.im z m * complex.I) : m = -1 :=
by sorry

-- III) If z is in the third quadrant, find the range of values for m
theorem part3 (m : ℝ) (h1 : 1 - m^2 < 0) (h2 : m^2 - 3 * m + 2 < 0) : 1 < m ∧ m < 2 := 
by sorry

end part1_part2_part3_l658_658469


namespace find_initial_money_l658_658044
 
theorem find_initial_money (x : ℕ) (gift_grandma gift_aunt_uncle gift_parents total_money : ℕ) 
  (h1 : gift_grandma = 25) 
  (h2 : gift_aunt_uncle = 20) 
  (h3 : gift_parents = 75) 
  (h4 : total_money = 279) 
  (h : x + (gift_grandma + gift_aunt_uncle + gift_parents) = total_money) : 
  x = 159 :=
by
  sorry

end find_initial_money_l658_658044


namespace transformed_mean_variance_l658_658128

-- Given conditions
def mean_original (data : Fin 10 → ℝ) : Prop :=
  (∑ i, data i) / 10 = 2

def variance_original (data : Fin 10 → ℝ) : Prop :=
  (∑ i, (data i - 2) ^ 2) / 10 = 3

-- The theorem to be proven
theorem transformed_mean_variance (data : Fin 10 → ℝ) (h_mean : mean_original data) (h_variance : variance_original data) :
  (∑ i, (2 * data i + 3)) / 10 = 7 ∧ (∑ i, ((2 * data i + 3) - 7) ^ 2) / 10 = 12 :=
by
  sorry

end transformed_mean_variance_l658_658128


namespace sqrt_2_times_sqrt_3_eq_sqrt_6_l658_658732

theorem sqrt_2_times_sqrt_3_eq_sqrt_6
    (h1 : ¬ (sqrt 2 + sqrt 3 = sqrt 5))
    (h2 : ¬ (4 * sqrt 3 - 3 * sqrt 3 = 1))
    (h3 : ¬ (sqrt 12 = 3 * sqrt 2)) : sqrt 2 * sqrt 3 = sqrt 6 := by
  sorry

end sqrt_2_times_sqrt_3_eq_sqrt_6_l658_658732


namespace maximum_value_of_modulus_l658_658530

theorem maximum_value_of_modulus (z : ℂ) (hz : |z| = 1) : ∃ w : ℂ, |w| = 1 ∧ |w - complex.I| = 2 :=
by
  -- The proof will be added here.
  sorry

end maximum_value_of_modulus_l658_658530


namespace regression_line_equation_l658_658018

theorem regression_line_equation 
  (n : ℕ) (x y : ℕ → ℝ) 
  (h1 : (1 / n) * (∑ i in finset.range n, x i) = 1)
  (h2 : (1 / n) * (∑ i in finset.range n, y i) = 1)
  (h3 : ∑ i in finset.range n, (x i - 1) * (y i - 1) = 4) 
  (h4 : ∑ i in finset.range n, (x i - 1)^2 = 2) : 
  ∃ b a : ℝ, (b = 2) ∧ (a = -1) ∧ (∀ x, b * x + a = 2 * x - 1) :=
sorry

end regression_line_equation_l658_658018


namespace assignment_schemes_with_at_least_one_girl_l658_658839

theorem assignment_schemes_with_at_least_one_girl
  (boys girls : ℕ)
  (tasks : ℕ)
  (hb : boys = 4)
  (hg : girls = 3)
  (ht : tasks = 3)
  (total_choices : ℕ := (boys + girls).choose tasks * tasks.factorial)
  (all_boys : ℕ := boys.choose tasks * tasks.factorial) :
  total_choices - all_boys = 186 :=
by
  sorry

end assignment_schemes_with_at_least_one_girl_l658_658839


namespace range_of_a_l658_658438

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-1 : ℝ) 1 then 2^(abs x) - 1 else sorry -- Defined based on conditions

theorem range_of_a (a : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_period : ∀ x, f (x + 2) = f x) 
  (h_eq : ∀ x ∈ Ico (-1 : ℝ) 3, (f x - Real.logBase a (x + 1)) = 0) :
  2 < a ∧ a < 4 := 
  sorry

end range_of_a_l658_658438


namespace time_to_paint_house_l658_658830

theorem time_to_paint_house (total_work : ℝ) (experts_contribution_rate : ℝ) (nonexperts_contribution_rate : ℝ) (time_5_painters : ℝ) :
  let experts_1_hr := 2
  let nonexperts_1_hr := 1
  let total_units_done := 2 * experts_1_hr * time_5_painters + 3 * nonexperts_1_hr * time_5_painters in
  
  total_units_done = total_work →
  (4 * total_work) / (4 * experts_1_hr + 2 * nonexperts_1_hr) = 28 / 6 ∧
  28 / 6 ≈ 4.67 :=
by
  let total_work := 28
  let experts_contribution_rate := 2
  let nonexperts_contribution_rate := 1
  let time_5_painters := 4
  let units_done_by_2_experts := 2 * experts_contribution_rate * time_5_painters
  let units_done_by_3_nonexperts := 3 * nonexperts_contribution_rate * time_5_painters
  let total_units := units_done_by_2_experts + units_done_by_3_nonexperts
  
  have h0 : total_units = total_work := by
    simp [units_done_by_2_experts, units_done_by_3_nonexperts, total_units, total_work]
    sorry

  exact h0
  exact h0

end time_to_paint_house_l658_658830


namespace base_9_numbers_with_5_or_6_digit_l658_658172

def uses_digit_5_or_6_in_base_9 (n : ℕ) : Prop :=
  ∃ d ∈ [5, 6], ∃ k, n = (d * 9 ^ k)

theorem base_9_numbers_with_5_or_6_digit:
  ∃ (count : ℕ), count = 386 ∧ (∀ n < 729, uses_digit_5_or_6_in_base_9 n → n ∈ finset.range 729)
:= sorry

end base_9_numbers_with_5_or_6_digit_l658_658172


namespace regular_tetrahedron_triangles_l658_658163

theorem regular_tetrahedron_triangles :
  let vertices := 4
  ∃ triangles : ℕ, (triangles = Nat.choose vertices 3) ∧ (triangles = 4) :=
by {
  let vertices := 4,
  use Nat.choose vertices 3,
  split,
  { 
    refl,
  },
  {
    norm_num,
  }
}

end regular_tetrahedron_triangles_l658_658163


namespace boys_in_fifth_grade_l658_658564

noncomputable def total_students : ℕ := 420
noncomputable def students_playing_soccer : ℕ := 250
noncomputable def percentage_boys_playing_soccer : ℝ := 0.78
noncomputable def girl_students_not_playing_soccer : ℕ := 53

noncomputable def total_boys : ℕ :=
  let boys_playing_soccer := (percentage_boys_playing_soccer * students_playing_soccer).to_nat
  let students_not_playing_soccer := total_students - students_playing_soccer
  let boys_not_playing_soccer := students_not_playing_soccer - girl_students_not_playing_soccer
  boys_playing_soccer + boys_not_playing_soccer

theorem boys_in_fifth_grade (h : total_boys = 312) : total_boys = 312 :=
by {
  apply h,
  sorry
}

end boys_in_fifth_grade_l658_658564


namespace perpendicular_k_value_l658_658899

open Real

def a := (sqrt 3, 1)
def b := (0, -1)
def c (k : ℝ) := (k, sqrt 3)
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_k_value (k : ℝ) : 
  dot_prod (a.1, a.2 - 2 * b.2) (c k) = 0 → k = -3 
  :=
by
  sorry

end perpendicular_k_value_l658_658899


namespace xy_minimization_l658_658477

theorem xy_minimization (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (1 / (x : ℝ)) + 1 / (3 * y) = 1 / 11) : x * y = 176 ∧ x + y = 30 :=
by
  sorry

end xy_minimization_l658_658477


namespace pablo_days_to_complete_puzzles_l658_658627

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l658_658627


namespace phi_approximation_l658_658285

theorem phi_approximation (δ : ℝ) (ε : ℝ) (h1 : 0 ≤ δ) (h2 : δ ≤ 1) (h3 : 0 < ε) :
  ∃ n : ℕ, |(φ n / n) - δ| < ε := 
sorry

end phi_approximation_l658_658285


namespace probability_proof_l658_658573

noncomputable def probability_greater_than_ten : ℚ :=
  let outcomes : List (ℤ × ℤ) := [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 2), (2, 1), (2, 2)]
  let valid_positions (card_start: ℤ) (spin_outcomes: List (ℤ × ℤ)) : List (ℤ × ℤ) :=
    spin_outcomes.filter (λ (x1, x2) => card_start + x1 + x2 > 10) 
  let valid_probabilities (starting_points : List ℤ) : ℚ :=
    (1 / 12) * starting_points.map (λ start => 
      (valid_positions start outcomes).length / outcomes.length
    ).sum
  valid_probabilities [8, 9, 10] + 
  valid_probabilities [11, 12]

theorem probability_proof : probability_greater_than_ten = 23 / 54 := by
  sorry

end probability_proof_l658_658573


namespace total_cost_of_fencing_l658_658083

def diameter : ℝ := 28
def cost_per_meter : ℝ := 1.50
def pi_approx : ℝ := 3.14159

noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem total_cost_of_fencing : total_cost = 131.94 :=
by
  sorry

end total_cost_of_fencing_l658_658083


namespace PA_plus_PB_eq_3sqrt2_l658_658940

-- Define the center of the circle and its radius
def center : ℝ × ℝ := (2, 0)
def radius : ℝ := sqrt 2

-- Parametric equation of the line
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-t, 1 + t)

-- Coordinates of point P in Cartesian coordinates
def point_P : ℝ × ℝ := (0, 1)

-- Polar equation of circle C
def polar_eq_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ + 2 = 0

-- Polar equation of line l
def polar_eq_line (ρ θ : ℝ) : Prop :=
  ρ * (cos θ + sin θ) = 1

-- Main theorem statement
theorem PA_plus_PB_eq_3sqrt2 : 
  let A := parametric_line (t : ℝ) in
  let B := parametric_line (t' : ℝ) in
  by { 
      exists t t', 
      intersection (polar_eq_circle (-t) t) 
      ∧ intersection (polar_eq_line t t') 
      ∧ t < 0 
      ∧ t' < 0 
      ∧ |P - A| + |P - B| = 3 * sqrt 2 
  } :=
sorry

end PA_plus_PB_eq_3sqrt2_l658_658940


namespace prism_propositions_correctness_l658_658415

/-- 
Among the following four propositions about a prism:
1. If two lateral faces are perpendicular to the base, then the prism is a right prism;
2. If the sections through opposite lateral edges are both perpendicular to the base, then the prism is a right prism;
3. If all four lateral faces are pairwise congruent, then the prism is a right prism;
4. If the four diagonals of the prism are pairwise equal, then the prism is a right prism.
Prove that Propositions ① and ③ are false, and Propositions ② and ④ are true.
-/
theorem prism_propositions_correctness :
  (¬ (∀ (P: Prism), (P.two_lateral_faces_perpendicular_to_base → P.is_right_prism))) ∧
  (∀ (P: Prism), (P.sections_opposite_lateral_edges_perpendicular_to_base → P.is_right_prism)) ∧
  (¬ (∀ (P: Prism), (P.lateral_faces_pairwise_congruent → P.is_right_prism))) ∧
  (∀ (P: Prism), (P.diagonals_pairwise_equal → P.is_right_prism)) :=
by
  sorry

end prism_propositions_correctness_l658_658415


namespace min_fill_boxes_l658_658081

/-- 
  Given numbers 1, 2, 3, 4, 5, 6 to be used exactly once to fill into six boxes
  and arranging them into two-digit and one-digit numbers to find the 
  minimal possible result.
-/
theorem min_fill_boxes :
  ∃ (a b c d e f : ℕ), 
    {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧ 
    a * b + c * d = 342 :=
by 
  sorry

end min_fill_boxes_l658_658081


namespace collinear_vectors_l658_658840

noncomputable def vector_a : ℝ × ℝ := (3, Real.logb 2 15)
noncomputable def vector_b : ℝ × ℝ := (2, Real.logb 2 3)
noncomputable def vector_c (m : ℝ) : ℝ × ℝ := (2, Real.logb 2 m)
noncomputable def vector_diff : ℝ × ℝ := (1, Real.logb 2 5)

theorem collinear_vectors {m : ℝ} :
  vector_diff = (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) →
  (vector_a.1 - vector_b.1) / vector_c m.1 = (vector_a.2 - vector_b.2) / vector_c m.2 →
  m = 25 :=
by {
  intro hv,
  intro hc,
  sorry
}

end collinear_vectors_l658_658840


namespace pascal_fifth_element_row_20_l658_658707

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l658_658707


namespace max_interval_length_l658_658460

def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
def n (x : ℝ) : ℝ := 2 * x - 3

def are_close_functions (m n : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |m x - n x| ≤ 1

theorem max_interval_length
  (h : are_close_functions m n 2 3) :
  3 - 2 = 1 :=
sorry

end max_interval_length_l658_658460


namespace shiela_used_seven_colors_l658_658413

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ)
  (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : (total_blocks / blocks_per_color) = 7 :=
by
  sorry

end shiela_used_seven_colors_l658_658413


namespace middle_group_frequency_l658_658220

theorem middle_group_frequency 
  (sample_size : ℕ)
  (num_rectangles : ℕ)
  (middle_rect_area_ratio : ℚ)
  (frequency : ℚ)
  (frequency_of_middle_group : ℕ) : 
  num_rectangles = 9 →
  middle_rect_area_ratio = 1/3 →
  sample_size = 200 →
  frequency = 1/4 →
  frequency_of_middle_group = sample_size * frequency :=
begin
  intros h1 h2 h3 h4,
  rw [h3, h4],
  exact rfl,
end

end middle_group_frequency_l658_658220


namespace average_annual_growth_rate_correct_l658_658546

-- Define the given problem conditions
variable (x : ℝ) -- x represents the average annual growth rate
constant initial_price : ℝ := 5000 -- Initial price in 2018
constant final_price : ℝ := 6500 -- Final price in 2020

-- The statement we need to prove
theorem average_annual_growth_rate_correct :
  initial_price * (1 + x) ^ 2 = final_price :=
sorry -- Proof is to be provided

end average_annual_growth_rate_correct_l658_658546


namespace total_sample_size_l658_658024

theorem total_sample_size
    (undergrad_count : ℕ) (masters_count : ℕ) (doctoral_count : ℕ)
    (total_students : ℕ) (sample_size_doctoral : ℕ) (proportion_sample : ℕ)
    (n : ℕ)
    (H1 : undergrad_count = 12000)
    (H2 : masters_count = 1000)
    (H3 : doctoral_count = 200)
    (H4 : total_students = undergrad_count + masters_count + doctoral_count)
    (H5 : sample_size_doctoral = 20)
    (H6 : proportion_sample = sample_size_doctoral / doctoral_count)
    (H7 : n = proportion_sample * total_students) :
  n = 1320 := 
sorry

end total_sample_size_l658_658024


namespace mona_unique_players_l658_658614

theorem mona_unique_players (groups : ℕ) (players_per_group : ℕ) (repeated1 : ℕ) (repeated2 : ℕ) :
  (groups = 9) → (players_per_group = 4) → (repeated1 = 2) → (repeated2 = 1) →
  (groups * players_per_group - (repeated1 + repeated2) = 33) :=
begin
  intros h_groups h_players_per_group h_repeated1 h_repeated2,
  rw [h_groups, h_players_per_group, h_repeated1, h_repeated2],
  norm_num,
end

end mona_unique_players_l658_658614


namespace intersection_area_is_30_l658_658336

-- Defining the conditions: points, lengths and angle
variables (A B C D : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (distance : A → A → ℝ)
variables (angle : A → A → A → ℝ)

-- Conditions of the problem
def points_on_same_side_of_AB : Prop := sorry
def triangles_on_plane : Prop := sorry
def AB_eq_13 : Prop := distance A B = 13
def BC_eq_12 : Prop := distance B C = 12
def CA_eq_5 : Prop := distance C A = 5
def AD_eq_5 : Prop := distance A D = 5
def DB_eq_12 : Prop := distance D B = 12
def angles_equal : Prop := angle B A C = angle B A D

-- Prove the intersection area of the two triangles is 30
theorem intersection_area_is_30 (h1 : points_on_same_side_of_AB) 
                                (h2 : triangles_on_plane) 
                                (h3 : AB_eq_13) 
                                (h4 : BC_eq_12) 
                                (h5 : CA_eq_5) 
                                (h6 : AD_eq_5) 
                                (h7 : DB_eq_12) 
                                (h8 : angles_equal) : ∃ (m n : ℕ), m + n = 31 ∧ gcd m n = 1 ∧ 30 = (m : ℕ) / (n : ℕ) :=
begin
  sorry
end

end intersection_area_is_30_l658_658336


namespace spinner_prime_probability_l658_658804
  
theorem spinner_prime_probability :
  let sectors := [2, 4, 7, 8, 11, 13, 14, 17]
  ∃ prime_sectors : Finset ℕ, (∀ x ∈ prime_sectors, x ∈ sectors ∧ Nat.Prime x) ∧
  (prime_sectors.card.to_nat.to_rat = 5) ∧
  (sectors.length.to_rat = 8) →
  (prime_sectors.card.to_nat.to_rat / sectors.length.to_rat = 5 / 8) :=
by
  sorry

end spinner_prime_probability_l658_658804


namespace log_base_change_log_base_evaluation_l658_658078

-- Define the conditions as functions or constants used in the statement
theorem log_base_change 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  : log a (x) / log a (b) = log b (x) := sorry

theorem log_base_evaluation 
  : log 16 2 = 1 / 4 := by 
  have h16 : 16 = 2 ^ 4 := by norm_num
  have log_identity : log 16 2 = log (2 ^ 4) 2 := by rw h16
  have log_change : log (2 ^ 4) 2 = log 2 2 / log 2 (2 ^ 4) := log_base_change 2 2⁴ 2 (by norm_num) (by norm_num) (by norm_num)
  rw [log_change, log_self, log_pow] at log_identity
  exact log_identity

end log_base_change_log_base_evaluation_l658_658078


namespace seven_k_plus_four_l658_658239

theorem seven_k_plus_four (k m n : ℕ) (h1 : 4 * k + 5 = m^2) (h2 : 9 * k + 4 = n^2) (hk : k = 5) : 
  7 * k + 4 = 39 :=
by 
  -- assume conditions
  have h1' := h1
  have h2' := h2
  have hk' := hk
  sorry

end seven_k_plus_four_l658_658239


namespace sums_of_consecutive_8_distinct_l658_658234

noncomputable def sum_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sums_of_consecutive_8_distinct (arr : List ℕ):
  (∀ (i : ℕ), i < 2018 → arr.nthLe i sorry ≠ 0) → -- To ensure all elements are not zero and valid indices.
  arr.length = 2018 →
  (∀ i, 8 * sum_natural_numbers 2018 % 2018 = 0 →
    (arr ++ arr).take 2025 % 2018 ≠ [0, 1, 2, 3, 4, 5, 6, 7, 8, ... , 2017]) :=
begin
  intro h,
  intros,
  sorry,
end

end sums_of_consecutive_8_distinct_l658_658234


namespace range_H_l658_658350

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_H : set.range H = {-4, 4} := 
by
  sorry

end range_H_l658_658350


namespace find_liar_l658_658693

theorem find_liar : 
  ∃ (A B C D : Prop),
  (∃ (shots_A : Nat), shots_A = 5 ∧ (∃ p1 p2 p3 p4 p5, p1 + p2 + p3 + p4 + p5 = 35 ∧ p1 ∈ {1, 3, 5, 7, 9} ∧ p2 ∈ {1, 3, 5, 7, 9} ∧ p3 ∈ {1, 3, 5, 7, 9} ∧ p4 ∈ {1, 3, 5, 7, 9} ∧ p5 ∈ {1, 3, 5, 7, 9})) ∧ 
  (∃ (shots_B : Nat), shots_B = 6 ∧ (∃ q1 q2 q3 q4 q5 q6, q1 + q2 + q3 + q4 + q5 + q6 = 36 ∧ q1 ∈ {1, 3, 5, 7, 9} ∧ q2 ∈ {1, 3, 5, 7, 9} ∧ q3 ∈ {1, 3, 5, 7, 9} ∧ q4 ∈ {1, 3, 5, 7, 9} ∧ q5 ∈ {1, 3, 5, 7, 9} ∧ q6 ∈ {1, 3, 5, 7, 9})) ∧
  (∃ (shots_C : Nat), shots_C = 3 ∧ (∃ r1 r2 r3, r1 + r2 + r3 = 24 ∧ r1 ∈ {1, 3, 5, 7, 9} ∧ r2 ∈ {1, 3, 5, 7, 9} ∧ r3 ∈ {1, 3, 5, 7, 9})) ∧
  (∃ (shots_D : Nat), shots_D = 4 ∧ (∃ s1 s2 s3, s1 + s2 + s3 = 21 ∧ s1 ∈ {1, 3, 5, 7, 9} ∧ s2 ∈ {1, 3, 5, 7, 9} ∧ s3 ∈ {1, 3, 5, 7, 9})) ∧ 
  (A ∨ B ∨ C ∨ D) ∧ ((C ∧ ¬A ∧ ¬B ∧ ¬D) ∨ (¬C ∧ (A ∧ ¬B ∧ ¬D) ∨ (B ∧ ¬A ∧ ¬C ∧ ¬D) ∨ (D ∧ ¬A ∧ ¬B ∧ ¬C))) 
:=
sorry

end find_liar_l658_658693


namespace seq_solution_l658_658945

-- Define the sequence using initial condition and recursive condition
def seq (a : ℕ → ℝ) : Prop := 
  (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 2)

-- State the problem to be proven
theorem seq_solution (a : ℕ → ℝ) (h : seq a) : 
  ∀ n : ℕ, a (n + 1) = 3 * 2^n - 2 :=
by
  -- Use the definition of the sequence
  cases h with h₁ h₂
  intro n
  induction n with n ih
  -- Base case
  { simp [h₁] }
  -- Inductive step
  { simp [h₂, ih]
    sorry }

end seq_solution_l658_658945


namespace f_2015_eq_neg_2014_l658_658912

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f1_value : f 1 = 2014 := sorry

-- Theorem to prove
theorem f_2015_eq_neg_2014 :
  isOddFunction f → isPeriodic f 3 → (f 1 = 2014) → f 2015 = -2014 :=
by
  intros hOdd hPeriodic hF1
  sorry

end f_2015_eq_neg_2014_l658_658912


namespace find_c_d_l658_658674

-- Define the equation and the variables c and d
def equation := ∀ (x : ℝ), x^2 - 18 * x = 80

-- State the theorem we want to prove
theorem find_c_d (c d : ℕ) (x : ℝ) (h1 : equation x) (h2 : x = Real.sqrt c + d) : 
  c + d = 170 :=
sorry

end find_c_d_l658_658674


namespace dave_tickets_after_challenge_l658_658789
-- Importing the necessary library for the proof.

-- Defining the theorem with the necessary conditions and required proof.
theorem dave_tickets_after_challenge : 
  let initial_tickets := 11 in
  let tickets_spent_on_candy := 3 in
  let tickets_spent_on_beanie := 5 in
  let tickets_won_in_racing_game := 10 in
  let tickets_left_after_spending := initial_tickets - (tickets_spent_on_candy + tickets_spent_on_beanie) in
  let tickets_after_winning := tickets_left_after_spending + tickets_won_in_racing_game in
  let final_tickets := tickets_after_winning * 2 in
  final_tickets = 26 :=
by
  sorry

end dave_tickets_after_challenge_l658_658789


namespace min_n_exceeds_target_product_l658_658147

theorem min_n_exceeds_target_product :
  ∀ (n : ℕ), 
  let product := (∏ k in (range (n+1)).map (λ x, x + 2), 8 ^ (k / 11)) in
  product > 1000000 ↔ n >= 11 := by
  sorry

end min_n_exceeds_target_product_l658_658147


namespace div64_by_expression_l658_658632

theorem div64_by_expression {n : ℕ} (h : n > 0) : ∃ k : ℤ, (3^(2 * n + 2) - 8 * ↑n - 9) = 64 * k :=
by
  sorry

end div64_by_expression_l658_658632


namespace alex_needs_packs_of_buns_l658_658779

-- Definitions (conditions)
def guests : ℕ := 10
def burgers_per_guest : ℕ := 3
def meat_eating_guests : ℕ := guests - 1
def bread_eating_ratios : ℕ := meat_eating_guests - 1
def buns_per_pack : ℕ := 8

-- Theorem (question == answer)
theorem alex_needs_packs_of_buns : 
  (burgers_per_guest * meat_eating_guests - burgers_per_guest) / buns_per_pack = 3 := by
  sorry

end alex_needs_packs_of_buns_l658_658779


namespace find_a_b_find_m_l658_658888

-- Problem 1 conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a^2 * x^3 + 3 * a * x^2 - b * x - 1

-- Statement for part 1: Finding a and b
theorem find_a_b (a b : ℝ) 
  (h1 : f 1 a b = 0)
  (h2 : (derange f) 1 a b = 0) :
  (a = -1/2 ∧ b = -9/4) :=
begin
  sorry
end

-- Problem 2 conditions
def g (x : ℝ) : ℝ := (1/4) * x^3 - (3/2) * x^2 + (9/4) * x - 1

-- Statement for part 2: Finding the range of m
theorem find_m (m : ℝ) 
  (h : ∀ x ∈ set.Ici 0, g x ≥ m) :
  m ∈ Iic (-1) :=
begin
  sorry
end

end find_a_b_find_m_l658_658888


namespace angle_equality_l658_658036

-- Define the entities in the problem
variables (O A B P D E F C : Type)
variables (h_semicircle_midpoint : C = midpoint O A B)
variables (h_diameter_extension : ∃ P, P ∈ line_extension A B)
variables (h_PD_tangent : tangent PD)
variables (h_point_tangency : ∃ D, point_of_tangency PD D)
variables (h_bisector_intersects_AC : bisector_intersects DPB AC E)
variables (h_bisector_intersects_BC : bisector_intersects DPB BC F)

-- Statement to prove: 
theorem angle_equality (H : O A B P D E F C) :
  ∠ PDA = ∠ CDF := 
sorry

end angle_equality_l658_658036


namespace odd_function_f1_eq_4_l658_658871

theorem odd_function_f1_eq_4 (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x < 0 → f x = x^2 + a * x)
  (h3 : f 2 = 6) : 
  f 1 = 4 :=
by sorry

end odd_function_f1_eq_4_l658_658871


namespace jerry_mowing_hours_l658_658244

theorem jerry_mowing_hours (painting_hours : ℕ) (kitchen_hours_multiplier : ℕ) (hourly_rate : ℕ) (total_paid : ℕ)
  (painting_eq : painting_hours = 8) 
  (kitchen_eq : kitchen_hours_multiplier = 3) 
  (rate_eq : hourly_rate = 15) 
  (total_eq : total_paid = 570) : 
  let kitchen_hours := kitchen_hours_multiplier * painting_hours,
      total_work_hours := painting_hours + kitchen_hours,
      total_work_paid := total_work_hours * hourly_rate,
      remaining_paid := total_paid - total_work_paid,
      mowing_hours := remaining_paid / hourly_rate
  in mowing_hours = 6 := 
by
  sorry

end jerry_mowing_hours_l658_658244


namespace intervals_of_monotonicity_and_m_range_l658_658379

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem intervals_of_monotonicity_and_m_range :
  (∀ x : ℝ, x ∈ Ioo (-∞) (-2) ∨ x ∈ Ioo 0 (+∞) → f' x > 0) ∧
  (∀ x : ℝ, x ∈ Ioo (-2) 0 → f' x < 0) ∧
  (∀ x : ℝ, x ∈ Icc (-2) 2 → f x > m ↔ m < 0) :=
by
  sorry

end intervals_of_monotonicity_and_m_range_l658_658379


namespace valentines_given_l658_658280

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given_l658_658280


namespace sum_of_numbers_l658_658738

theorem sum_of_numbers (x y : ℕ) (h1 : x * y = 9375) (h2 : y / x = 15) : x + y = 400 :=
by
  sorry

end sum_of_numbers_l658_658738


namespace gcd_sq_of_nat_fracs_l658_658273

theorem gcd_sq_of_nat_fracs (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) :
  ∃ k : ℕ, k^2 = Int.gcd x y z * (y - x) :=
sorry

end gcd_sq_of_nat_fracs_l658_658273


namespace distinct_triangles_from_tetrahedron_l658_658159

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l658_658159


namespace angle_B_area_of_triangle_l658_658513

/-
Given a triangle ABC with angle A, B, C and sides a, b, c opposite to these angles respectively.
Consider the conditions:
- A = π/6
- b = (4 + 2 * sqrt 3) * a * cos B
- b = 1

Prove:
1. B = 5 * π / 12
2. The area of triangle ABC = 1 / 4
-/

namespace TriangleProof

open Real

def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  b = (4 + 2 * sqrt 3) * a * cos B ∧
  b = 1

theorem angle_B (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  B = 5 * π / 12 :=
sorry

theorem area_of_triangle (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  1 / 2 * b * c * sin A = 1 / 4 :=
sorry

end TriangleProof

end angle_B_area_of_triangle_l658_658513


namespace pascal_fifth_element_row_20_l658_658713

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l658_658713


namespace construct_triangle_l658_658054

theorem construct_triangle {s_c r_1 r_2 : ℝ} 
  (s_c_pos : s_c > 0)
  (r1_r2_nonneg : r_1 ≥ 0 ∧ r_2 ≥ 0)
  (r1_r2_order : r_1 ≤ r_2)
  (r1_condition : r_1 ≥ s_c / 2)
  (r2_condition : r_2 > s_c / 2) :
  ∃ (A B C F O1 O2 : ℝ × ℝ), 
    (dist C F = s_c) ∧ 
    (dist C O1 = r_1) ∧ (dist F O1 = r_1) ∧ 
    (dist C O2 = r_2) ∧ (dist F O2 = r_2) ∧ 
    (¬ collinear {C, F, O1, O2}) ∧
    (dist A F = dist B F) ∧
   (¬ collinear {A, B, C}) := sorry

end construct_triangle_l658_658054


namespace number_of_female_students_l658_658927

theorem number_of_female_students (T S f_sample : ℕ) (H_total : T = 1600) (H_sample_size : S = 200) (H_females_in_sample : f_sample = 95) : 
  ∃ F, 95 / 200 = F / 1600 ∧ F = 760 := by 
sorry

end number_of_female_students_l658_658927


namespace number_of_positive_integers_N_l658_658458

theorem number_of_positive_integers_N (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, 48 = k * (n + 3)) :
  (finset.filter (λ n, ∃ m : ℕ, 48 = m * (n + 3)) (finset.range 46).filter (λ n, n > 0)).card = 7 :=
sorry

end number_of_positive_integers_N_l658_658458


namespace intersections_of_ten_streets_l658_658606

theorem intersections_of_ten_streets (n : ℕ) (h : n = 10) 
    (h1 : ∀ i j, i ≠ j → ¬parallel (street i) (street j))
    (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬collinear (point_of_intersection (street i) (street j)) 
                                                           (point_of_intersection (street j) (street k)) 
                                                           (point_of_intersection (street k) (street i))) :
    ∑ i in finset.range n, i = 45 :=
by
  -- Assuming no parallel streets and no three streets meet at a single point.
  -- Calculation of intersections:
  -- ∑ i in finset.range 10, i = 45
  sorry

end intersections_of_ten_streets_l658_658606


namespace find_lambda_l658_658153

variables (i j : ℝ → ℝ) [noncollinear i j]
  (A B C D : ℝ)
  (l : ℝ)
  (h1 : (λ x, x = A) - B = i + 2 * j)
  (h2 : (λ x, x = C) - B = i + l * j)
  (h3 : (λ x, x = D) - C = -2 * i + j)
  (h_collinear : collinear (λ x, x = A) B (λ x, x = D))

theorem find_lambda : l = 7 :=
sorry

end find_lambda_l658_658153


namespace guarantee_advancement_at_least_7_l658_658219

-- Definitions of the conditions
def teams := {A, B, C, D : Type} -- 4 teams in total
def points_for_win := 3
def points_for_draw := 1
def points_for_loss := 0

-- Total matches in a round-robin tournament with 4 teams
def total_matches := 4 * 3 / 2

-- Total points accumulated in all matches
def total_points := total_matches * 3 -- in case all matches are wins

-- To guarantee advancement, a team must earn at least 7 points
def guarantee_advancement(points : ℕ) : Prop := points ≥ 7

-- Theorem statement
theorem guarantee_advancement_at_least_7 
  (total_points : ℕ) (team_points : ℕ) (team1_points team2_points team3_points team4_points : ℕ) 
  (h1 : total_points = 18)
  (h2 : team1_points + team2_points + team3_points + team4_points = total_points)
  (h3 : team_points ≥ 7) : (team_points ≥ team1_points ∨ team_points ≥ team2_points ∨ team_points ≥ team3_points ∨ team_points ≥ team4_points) :=
by
  sorry

end guarantee_advancement_at_least_7_l658_658219


namespace josh_total_candies_l658_658575

def josh_initial_candies (initial_candies given_siblings : ℕ) : Prop :=
  ∃ (remaining_1 best_friend josh_eats share_others : ℕ),
    (remaining_1 = initial_candies - given_siblings) ∧
    (best_friend = remaining_1 / 2) ∧
    (josh_eats = 16) ∧
    (share_others = 19) ∧
    (remaining_1 = 2 * (josh_eats + share_others))

theorem josh_total_candies : josh_initial_candies 100 30 :=
by
  sorry

end josh_total_candies_l658_658575


namespace arithmetic_sequence_problem_l658_658514

variable (a_n b_n : ℕ → ℕ)
variable (A_n B_n : ℕ → ℕ)

-- Conditions
def arithmetic_sum_condition := ∀ n : ℕ, A_n n / B_n n = (7 * n + 45) / (n + 3)

-- Proof statement
theorem arithmetic_sequence_problem (h1 : ∀ n : ℕ, A_n n = n * (2 * (a_n n)))
                                   (h2 : ∀ n : ℕ, B_n n = n * (2 * (b_n n)))
                                   (h3 : arithmetic_sum_condition a_n b_n A_n B_n) :
  a_n 5 / b_n 5 = 9 :=
by
  sorry

end arithmetic_sequence_problem_l658_658514


namespace mike_total_cans_collected_l658_658618

theorem mike_total_cans_collected :
  let monday := 450
  let tuesday := monday + Int.ofNat (Float.ceil (0.33 * monday))
  let wednesday := tuesday + Int.ofNat (Float.ceil (0.05 * tuesday))
  let thursday := wednesday + Int.ofNat (Float.ceil (0.05 * wednesday))
  let friday := thursday + Int.ofNat (Float.ceil (0.05 * thursday))
  let saturday := friday + Int.ofNat (Float.ceil (0.05 * friday))
  let sunday := saturday + Int.ofNat (Float.ceil (0.05 * saturday))
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 4523 := by
  sorry

end mike_total_cans_collected_l658_658618


namespace part_i_l658_658747

theorem part_i (n : ℕ) (h₁ : n ≥ 1) (h₂ : n ∣ (2^n - 1)) : n = 1 :=
sorry

end part_i_l658_658747


namespace range_of_a_l658_658264

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x a : ℝ) := x^a

theorem range_of_a (h : ∃ l : ℝ → ℝ, ∀ x > 0, ∀ a ≠ 0, 
  (f(x) = l(x) ∧ g(x, a) = l(x))):
  {a : ℝ | ∃ l : ℝ → ℝ, ∀ x > 0, (f x = l x) ∧ (g x a = l x)} = 
  {a | (0 < a ∧ a ≤ 1 / Real.exp 1) ∨ (1 < a)} :=
sorry

end range_of_a_l658_658264


namespace curve_is_circle_l658_658084

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b r : ℝ), (r > 0) ∧ ((x + a)^2 + (y + b)^2 = r^2) :=
by
  sorry

end curve_is_circle_l658_658084


namespace variance_of_data_set_l658_658689

-- Defining the data set as a constant vector
def data_set : Vector ℝ 5 := ⟨[1, 3, 2, 5, 4], by decide⟩

-- Definition of mean of the data set
def mean (v : Vector ℝ n) : ℝ :=
  (v.toList.sum) / (n : ℝ)

-- Definition of variance of the data set
def variance (v : Vector ℝ n) : ℝ :=
  let m := mean v
  1 / (n : ℝ) * (v.toList.map (λ x => (x - m)^2)).sum

-- Theorem stating that the variance of the data set is 2
theorem variance_of_data_set : variance data_set = 2 :=
by
  sorry

end variance_of_data_set_l658_658689


namespace homer_second_try_points_l658_658518

theorem homer_second_try_points (x : ℕ) :
  400 + x + 2 * x = 1390 → x = 330 :=
by
  sorry

end homer_second_try_points_l658_658518


namespace true_propositions_count_l658_658418

theorem true_propositions_count :
  let p := (x: ℝ) → (x = 1 → |x| = 1)
      q := (x: ℝ) → (|x| = 1 → x = 1)
      r := (x: ℝ) → (x ≠ 1 → |x| ≠ 1)
      s := (x: ℝ) → (|x| ≠ 1 → x ≠ 1) in
  (∀ x : ℝ, p x) ∧ (∀ x : ℝ, s x) ∧ ¬ (∀ x : ℝ, q x) ∧ ¬ (∀ x : ℝ, r x) → 
  (finset.card (finset.filter (λ f : (ℝ → Prop) × bool, f.2) 
    ((∅ : finset (ℝ → Prop) × bool : finset.map 
      ⟨λ p : (ℝ → Prop), (p, ∀ x, p x), by ext; simp⟩ 
      ([(p, q, r, s) : list (ℝ → Prop)]))))) = 2 := 
by
  sorry

end true_propositions_count_l658_658418


namespace polygon_sides_of_T_l658_658270

def condition1 (b x : ℝ) : Prop := (b ≤ x ∧ x ≤ 3 * b)
def condition2 (b y : ℝ) : Prop := (b ≤ y ∧ y ≤ 3 * b)
def condition3 (b x y : ℝ) : Prop := (x + y ≥ 2 * b)
def condition4 (b x y : ℝ) : Prop := (x + 2 * b ≥ 2 * y)
def condition5 (b x y : ℝ) : Prop := (y + 2 * b ≥ 2 * x)

def set_T (b : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧
               condition1 b x ∧
               condition2 b y ∧
               condition3 b x y ∧
               condition4 b x y ∧
               condition5 b x y }

theorem polygon_sides_of_T (b : ℝ) (hb : 0 < b) : 
  ∃ (n : ℕ), n = 7 ∧ 
  ∃ vertices : vector (ℝ × ℝ) n, 
  is_polygon vertices ∧ 
  (∀ (p: ℝ × ℝ), p ∈ set_T b → p ∈ convex_hull (set.to_finset vertices)) :=
sorry

end polygon_sides_of_T_l658_658270


namespace moles_of_water_formed_l658_658090

def reaction_moles_cl_h2o (a b : ℕ) (reaction_ratio : ℕ → ℕ → Prop) :=
  ∀ a b : ℕ, reaction_ratio a b → a = b → a = 3 → b = 3

theorem moles_of_water_formed :
  reaction_moles_cl_h2o 3 3 (λ a b, a = b) := by
    sorry

end moles_of_water_formed_l658_658090


namespace lambda_equals_four_l658_658391

noncomputable def hyperbola : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 1}
def right_focus : ℝ × ℝ := (1, 0)

theorem lambda_equals_four :
  ∃ (l : ℝ → ℝ) (A B : ℝ × ℝ), (right_focus.pair_mem hyperbola.on_line l) ∧ 
                                (A ∈ hyperbola) ∧ 
                                (B ∈ hyperbola) ∧ 
                                (A ≠ B) ∧ 
                                (∣(dist A B)∣ = 4) ∧ 
                                (card {l | λ ∈ Lines(A, B) ∧ right_focus ∈ l.line ∧ intersects_hyperbola(l)} = 3) :=
sorry

end lambda_equals_four_l658_658391


namespace range_of_m_l658_658879

noncomputable def f (m : ℝ) : 𝔽 x => √(m * x^2 + m * x + 1)

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l658_658879


namespace regular_tetrahedron_triangles_l658_658165

theorem regular_tetrahedron_triangles :
  let vertices := 4
  ∃ triangles : ℕ, (triangles = Nat.choose vertices 3) ∧ (triangles = 4) :=
by {
  let vertices := 4,
  use Nat.choose vertices 3,
  split,
  { 
    refl,
  },
  {
    norm_num,
  }
}

end regular_tetrahedron_triangles_l658_658165


namespace derivative_of_y_l658_658820

noncomputable def y (x : ℝ) : ℝ :=
  cot (5^(1/3 : ℝ)) - (1/8) * ((cos (4 * x))^2 / sin (8 * x))

theorem derivative_of_y (x : ℝ) : (deriv y x) = 1 / (4 * (sin (4 * x))^2) :=
by
  sorry

end derivative_of_y_l658_658820


namespace line_equation_through_points_l658_658838

theorem line_equation_through_points (x y : ℝ) :
  (3 * x + 2 * y - 4 = 0) ∧ (x - y + 5 = 0) ∧ (P : ℝ × ℝ) := (2, -3) →
  ∃ λ : ℝ, 3.4 * x + 1.6 * y - 2 = 0 :=
sorry

end line_equation_through_points_l658_658838


namespace books_on_desk_none_useful_l658_658279

theorem books_on_desk_none_useful :
  ∃ (answer : String), answer = "none" ∧ 
  (answer = "nothing" ∨ answer = "no one" ∨ answer = "neither" ∨ answer = "none")
  → answer = "none"
:= by
  sorry

end books_on_desk_none_useful_l658_658279


namespace curve_C_eq_tangents_through_N_l658_658866

-- Definition of curve C based on the given conditions.
def curve_C (x y: ℝ) : Prop :=
  (x^2 + y^2 + 2 * x - 3 = 0)

-- Tangent line x = 1
def tangent_line_1 (x y: ℝ) : Prop :=
  (x = 1)

-- Tangent line 5x - 12y + 31 = 0
def tangent_line_2 (x y: ℝ) : Prop :=
  (5 * x - 12 * y + 31 = 0)

theorem curve_C_eq :
  ∀ (M: ℝ×ℝ), (let (x, y) := M in 
    ∃ (O: ℝ×ℝ) (A: ℝ×ℝ), 
    O = (0, 0) ∧ A = (3, 0) ∧
    ∃ r: ℚ, 
    r = 1/2 ∧
    dist (x, y) O / dist (x, y) A = r) →
  curve_C (M.1) (M.2) :=
by sorry

theorem tangents_through_N :
  ∀ (N: ℝ×ℝ),
  N = (1, 3) →
  (∃ x y, (tangent_line_1 x y ∨ tangent_line_2 x y) ∧ curve_C x y) :=
by sorry

end curve_C_eq_tangents_through_N_l658_658866


namespace find_k_even_function_find_a_intersection_l658_658495

noncomputable def f (x : ℝ) (k : ℝ) := log 4 (4^x + 1) + k * x
noncomputable def g (x : ℝ) (a : ℝ) := log 4 (a * 2^x - 4/3 * a)

theorem find_k_even_function (k : ℝ) : 
  (∀ x : ℝ, f x k = f (-x) k) ↔ k = -1/2 :=
by sorry

theorem find_a_intersection (a : ℝ) :
  (∃! x : ℝ, f x (-1/2) = g x a) ↔ (a = -3 ∨ a > 1) :=
by sorry

end find_k_even_function_find_a_intersection_l658_658495


namespace identical_solutions_of_quadratic_linear_l658_658837

theorem identical_solutions_of_quadratic_linear (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k ∧ x^2 = 4 * x + k) ↔ k = -4 :=
by
  sorry

end identical_solutions_of_quadratic_linear_l658_658837


namespace pascal_triangle_row_20_element_5_l658_658704

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l658_658704


namespace x_intercept_correct_l658_658450

noncomputable def x_intercept_of_line : ℝ × ℝ :=
if h : (-4 : ℝ) ≠ 0 then (24 / (-4), 0) else (0, 0)

theorem x_intercept_correct : x_intercept_of_line = (-6, 0) := by
  -- proof will be given here
  sorry

end x_intercept_correct_l658_658450


namespace probability_of_three_specific_suits_l658_658532

noncomputable def probability_at_least_one_from_each_of_three_suits : ℚ :=
  1 - (1 / 4) ^ 5

theorem probability_of_three_specific_suits (hearts clubs diamonds : ℕ) :
  hearts = 0 ∧ clubs = 0 ∧ diamonds = 0 → 
  probability_at_least_one_from_each_of_three_suits = 1023 / 1024 := 
by 
  sorry

end probability_of_three_specific_suits_l658_658532


namespace prime_divides_sum_l658_658449

-- Define the main problem statement
theorem prime_divides_sum (p : ℕ) (hp : Nat.Prime p) (hgt : p > 3) :
  p^2 ∣ ∑ k in Finset.range (p - 1), k^(2*p + 1) :=
sorry

end prime_divides_sum_l658_658449


namespace m_range_if_circle_eq_holds_a_range_if_nec_not_suff_cond_l658_658854

-- Given conditions
variables {x y m a : ℝ}
def circle_eq (m : ℝ) : Prop := ∃ k ∈ ℝ, (x - 2 * m) ^ 2 + y ^ 2 = - m ^ 2 - m + k

def m_inequality (m a : ℝ) : Prop := (m - a) * (m - a - 4) < 0

-- Required proofs
theorem m_range_if_circle_eq_holds (h : circle_eq m) : -2 < m ∧ m < 1 := sorry

theorem a_range_if_nec_not_suff_cond (h1 : -2 < m ∧ m < 1) (h2 : ∃ m, m_inequality m a) : -3 ≤ a ∧ a ≤ -2 := sorry

end m_range_if_circle_eq_holds_a_range_if_nec_not_suff_cond_l658_658854


namespace find_m_l658_658197

theorem find_m (x y m : ℝ) 
  (h1 : x + y = 8)
  (h2 : y - m * x = 7)
  (h3 : y - x = 7.5) : m = 3 := 
  sorry

end find_m_l658_658197


namespace measure_angle_C_l658_658869

variables {A B C O : Type*}
-- Conditions
variables [inner_product_space ℝ O] [metric_space A] [metric_space B] [metric_space C]
variables (OA OB OC : O)

-- Given that the circumcenter of triangle ABC is O
def circumcenter (O A B C : Type*) := sorry -- A formal definition of the circumcenter

-- The given condition on vector sums
axiom vector_sum : 3 • OA + 4 • OB + 5 • OC = (0 : O)

-- Main theorem statement
theorem measure_angle_C (circumcenter O A B C) (vector_sum : 3 • OA + 4 • OB + 5 • OC = (0 : O)) : measure_angle O B A = 45 := sorry

end measure_angle_C_l658_658869


namespace general_formula_sum_of_b_n_l658_658847

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)

-- Condition for the geometric sequence
axiom a_3_4 : a 3 = 4
axiom a_6_32 : a 6 = 32

-- Defining b_n in terms of a_n
def b (n : ℕ) : ℝ := a n - 3 * n 

-- The general formula for the sequence a_n
theorem general_formula 
    (a : ℕ → ℝ) 
    (h_3 : a 3 = 4) 
    (h_6 : a 6 = 32) : 
  ∀ n, a n = 2^(n-1) := 
sorry

-- The sum of the first n terms of the sequence b_n
theorem sum_of_b_n 
    (a : ℕ → ℝ)
    (h_3 : a 3 = 4) 
    (h_6 : a 6 = 32) : 
  ∀ n, (∑ i in Finset.range n, b a i) = 2^n - 1 - (3 * n^2 + 3 * n) / 2 :=
sorry

end general_formula_sum_of_b_n_l658_658847


namespace three_digit_numbers_l658_658977

theorem three_digit_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n^2 % 1000 = n % 1000) : 
  n = 376 ∨ n = 625 :=
by
  sorry

end three_digit_numbers_l658_658977


namespace function_inverse_example_l658_658538

theorem function_inverse_example:
  (∃ f : ℝ → ℝ, (∀ x < 0, f (x^2) = x) ∧ (f (9) = -3)) :=
begin
  let f : ℝ → ℝ := sorry,
  existsi f,
  split,
  { intros x hx,
    sorry, },
  { sorry, }
end

end function_inverse_example_l658_658538


namespace min_sum_of_factors_l658_658585

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l658_658585


namespace probability_of_card_leq_nine_is_one_l658_658727

-- Define the set of cards
def cardSet : Set ℕ := {1, 3, 4, 6, 7, 9}

-- Define the property of the drawn card having a number less than or equal to 9
def isLessThanOrEqualToNine (n : ℕ) : Prop := n ≤ 9

-- Define the probability calculation function
def probability (s : Set ℕ) (P : ℕ → Prop) : ℚ :=
  (s.filter P).toFinset.card / s.toFinset.card

-- The math proof problem
theorem probability_of_card_leq_nine_is_one :
  probability cardSet isLessThanOrEqualToNine = 1 := 
sorry

end probability_of_card_leq_nine_is_one_l658_658727


namespace find_whole_number_N_l658_658026

theorem find_whole_number_N (N : ℕ) (h1 : 6.75 < (N / 4 : ℝ)) (h2 : (N / 4 : ℝ) < 7.25) : N = 28 := 
by 
  sorry

end find_whole_number_N_l658_658026


namespace largest_divisor_of_expression_l658_658181

theorem largest_divisor_of_expression (x : ℤ) (hx : x % 2 = 1) :
  864 ∣ (12 * x + 2) * (12 * x + 6) * (12 * x + 10) * (6 * x + 3) :=
sorry

end largest_divisor_of_expression_l658_658181


namespace sum_345_consecutive_sequences_l658_658213

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l658_658213


namespace complex_number_solution_l658_658192

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l658_658192


namespace prob_of_one_letter_each_sister_l658_658809

noncomputable def prob_one_letter_each_sister : ℚ :=
  let total_cards := 10
  let letters_cybil := 5
  let letters_ronda := 5
  let prob_cybil_then_ronda := (letters_cybil / total_cards) * (letters_ronda / (total_cards - 1))
  let prob_ronda_then_cybil := (letters_ronda / total_cards) * (letters_cybil / (total_cards - 1))
  prob_cybil_then_ronda + prob_ronda_then_cybil

theorem prob_of_one_letter_each_sister :
  prob_one_letter_each_sister = 5 / 9 :=
sorry

end prob_of_one_letter_each_sister_l658_658809


namespace sauce_per_burger_proof_l658_658951

def total_sauce (ketchup vinegar honey : ℕ) : ℕ := ketchup + vinegar + honey
def sauce_for_pulled_pork (pp_sandwich_count : ℕ) (sauce_per_pp : ℚ) : ℚ := pp_sandwich_count * sauce_per_pp
def remaining_sauce (total_sauce used_sauce : ℚ) : ℚ := total_sauce - used_sauce
def sauce_per_burger (remaining_sauce : ℚ) (burger_count : ℕ) : ℚ := remaining_sauce / burger_count

theorem sauce_per_burger_proof :
  let ketchup := 3 
  let vinegar := 1 
  let honey := 1 
  let pp_sandwich_count := 18
  let sauce_per_pp := 1 / 6
  let burger_count := 8 in
  sauce_per_burger 
    (remaining_sauce 
      (total_sauce ketchup vinegar honey) 
      (sauce_for_pulled_pork pp_sandwich_count sauce_per_pp)) 
    burger_count = 1 / 4 :=
by
  sorry

end sauce_per_burger_proof_l658_658951


namespace difference_between_projections_l658_658671

def parallel_projection (P : Type) (lines : set (set P)) : Prop :=
  ∀ l1 l2 ∈ lines, l1 = l2 ∨ disjoint l1 l2

def central_projection (P : Type) (lines : set (set P)) (C : P) : Prop :=
  ∀ l ∈ lines, C ∈ l

theorem difference_between_projections (P : Type) (lines₁ lines₂ : set (set P)) (C : P) :
  parallel_projection P lines₁ ∧ central_projection P lines₂ C →
  (∃ D₁ D₂, D₁ = "In parallel projection, the projection lines are parallel to each other" 
   ∧ D₂ = "In central projection, the projection lines converge at a single point")
:= sorry

end difference_between_projections_l658_658671


namespace pulled_pork_sandwiches_l658_658571

/-
  Jack uses 3 cups of ketchup, 1 cup of vinegar, and 1 cup of honey.
  Each burger takes 1/4 cup of sauce.
  Each pulled pork sandwich takes 1/6 cup of sauce.
  Jack makes 8 burgers.
  Prove that Jack can make exactly 18 pulled pork sandwiches.
-/
theorem pulled_pork_sandwiches :
  (3 + 1 + 1) - (8 * (1/4)) = 3 -> 
  3 / (1/6) = 18 :=
sorry

end pulled_pork_sandwiches_l658_658571


namespace new_supervisor_salary_l658_658300

-- Definitions
def average_salary_old (W : ℕ) : Prop :=
  (W + 870) / 9 = 430

def average_salary_new (W : ℕ) (S_new : ℕ) : Prop :=
  (W + S_new) / 9 = 430

-- Problem statement
theorem new_supervisor_salary (W : ℕ) (S_new : ℕ) :
  average_salary_old W →
  average_salary_new W S_new →
  S_new = 870 :=
by
  sorry

end new_supervisor_salary_l658_658300


namespace relay_race_order_count_l658_658551

-- Definitions based on the given conditions
def team_members : List String := ["Sam", "Priya", "Jordan", "Luis"]
def first_runner := "Sam"
def last_runner := "Jordan"

-- Theorem stating the number of different possible orders
theorem relay_race_order_count {team_members first_runner last_runner} :
  (team_members = ["Sam", "Priya", "Jordan", "Luis"]) →
  (first_runner = "Sam") →
  (last_runner = "Jordan") →
  (2 = 2) :=
by
  intros _ _ _
  sorry

end relay_race_order_count_l658_658551


namespace initial_peaches_l658_658290

theorem initial_peaches :
  ∀ (total_peaches picked_at_orchard : ℝ),
  total_peaches = 85.0 →
  picked_at_orchard = 24.0 →
  (total_peaches - picked_at_orchard = 61.0) :=
by
  intros total_peaches picked_at_orchard h_total h_picked
  rw [h_total, h_picked]
  norm_num
  sorry

end initial_peaches_l658_658290


namespace lg_sum_eq_lg_double_diff_l658_658534

theorem lg_sum_eq_lg_double_diff (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_harmonic : 2 / y = 1 / x + 1 / z) : 
  Real.log (x + z) + Real.log (x - 2 * y + z) = 2 * Real.log (x - z) := 
by
  sorry

end lg_sum_eq_lg_double_diff_l658_658534


namespace max_students_above_average_l658_658548

theorem max_students_above_average (students : ℕ) (h_students : students = 150) (scores : fin students → ℝ) : 
  ∃ max_above_average, max_above_average = 149 := 
by
  have h_nonzero_scores: (∑ i, if scores i > (∑ j, scores j) / students then 1 else 0) ≤ 149 := sorry
  use 149
  exact h_nonzero_scores

end max_students_above_average_l658_658548


namespace payroll_amount_l658_658020

theorem payroll_amount (tax_paid : ℝ) (threshold : ℝ) (rate : ℝ) (payroll : ℝ) :
  tax_paid = rate * (payroll - threshold) → payroll = 400000 :=
by
  sorry

-- Usage example:
#eval payroll_amount 400 200000 0.002 400000

end payroll_amount_l658_658020


namespace tournament_matches_divisible_by_7_l658_658932

-- Define the conditions of the chess tournament
def single_elimination_tournament_matches (players byes: ℕ) : ℕ :=
  players - 1

theorem tournament_matches_divisible_by_7 :
  single_elimination_tournament_matches 120 40 = 119 ∧ 119 % 7 = 0 :=
by
  sorry

end tournament_matches_divisible_by_7_l658_658932


namespace exp_sum_ln_l658_658861

theorem exp_sum_ln (a b : ℝ) (h₁ : ln 2 = a) (h₂ : ln 3 = b) : exp a + exp b = 5 :=
by
  sorry

end exp_sum_ln_l658_658861


namespace regular_tetrahedron_triangles_l658_658162

theorem regular_tetrahedron_triangles :
  let vertices := 4
  ∃ triangles : ℕ, (triangles = Nat.choose vertices 3) ∧ (triangles = 4) :=
by {
  let vertices := 4,
  use Nat.choose vertices 3,
  split,
  { 
    refl,
  },
  {
    norm_num,
  }
}

end regular_tetrahedron_triangles_l658_658162


namespace bunch_of_bananas_cost_l658_658034

def cost_of_bananas (A : ℝ) : ℝ := 5 - A

theorem bunch_of_bananas_cost (A B T : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = T) : B = cost_of_bananas A :=
by
  sorry

end bunch_of_bananas_cost_l658_658034


namespace circles_radius_difference_l658_658321

variable (s : ℝ)

theorem circles_radius_difference (h : (π * (2*s)^2) / (π * s^2) = 4) : (2 * s - s) = s :=
by
  sorry

end circles_radius_difference_l658_658321


namespace length_of_other_side_l658_658386

theorem length_of_other_side (length_one_side : ℕ) (total_pieces : ℕ) (piece_side : ℕ)
  (h1 : length_one_side = 18) (h2 : total_pieces = 522) (h3 : piece_side = 1) :
  total_pieces / length_one_side = 29 := by
  rw [h1, h2]
  sorry

end length_of_other_side_l658_658386


namespace evaluate_trig_expression_l658_658867

def P : ℝ × ℝ := (-4, 3)
def θ_r := 5
def sin_θ : ℝ := 3 / 5
def cos_θ : ℝ := -4 / 5

theorem evaluate_trig_expression :
  2 * sin_θ + cos_θ = 2 / 5 :=
by
  sorry

end evaluate_trig_expression_l658_658867


namespace monotonically_decreasing_a_extrema_in_interval_l658_658497

-- Define the function f
def f (a x : ℝ) : ℝ := -x^2 + a * x - Math.log x

-- Problem 1: The range of a such that f(x) is monotonically decreasing
theorem monotonically_decreasing_a (a : ℝ) :
  (∀ x > 0, deriv (λ x, f a x) x ≤ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

-- Problem 2: The range of a such that f(x) has both a maximum and minimum in the interval (0, 3)
theorem extrema_in_interval (a : ℝ) :
  (∃ x1 x2 ∈ (0, 3), x1 ≠ x2 ∧ deriv (λ x, f a x) x1 = 0 ∧ deriv (λ x, f a x) x2 = 0) ↔ 2 * Real.sqrt 2 < a ∧ a < 19 / 3 :=
sorry

end monotonically_decreasing_a_extrema_in_interval_l658_658497


namespace pyramid_vertices_l658_658771

theorem pyramid_vertices (n : ℕ) (h : 2 * n = 14) : n + 1 = 8 :=
by {
  sorry
}

end pyramid_vertices_l658_658771


namespace tangent_lines_to_circle_l658_658085

open Real

def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def is_tangent_line (line : ℝ → ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  @exists (p : ℝ × ℝ), line p.1 p.2 ∧ (p ∈ circle center radius) ∧ ∀ q ≠ p, q ∈ circle center radius → line q.1 q.2 → False

def tangent_line_1 (x y : ℝ) : Prop := y = 4
def tangent_line_2 (x y : ℝ) : Prop := 3 * x + 4 * y - 13 = 0

def A : ℝ × ℝ := (-1, 4)

theorem tangent_lines_to_circle:
  let center := (2, 3)
  let radius := 1
  (tangent_line_1 A.1 A.2 ∧ is_tangent_line tangent_line_1 center radius) ∨
  (tangent_line_2 A.1 A.2 ∧ is_tangent_line tangent_line_2 center radius) :=
sorry

end tangent_lines_to_circle_l658_658085


namespace smallest_number_of_cubes_l658_658753

def box_length : ℕ := 24
def box_width  : ℕ := 40
def box_depth  : ℕ := 16
def gcd_dim    : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes : 
  let cube_side := gcd_dim in
  let num_cubes_length := box_length / cube_side in
  let num_cubes_width := box_width / cube_side in
  let num_cubes_depth := box_depth / cube_side in
  num_cubes_length * num_cubes_width * num_cubes_depth = 30 :=
by
  sorry

end smallest_number_of_cubes_l658_658753


namespace log_base_2_of_0_l658_658455

noncomputable def log_base_2 : ℝ → ℝ
| x := log10 x / log10 2

theorem log_base_2_of_0.375 : log_base_2 0.375 = -1.415 := by
  have h₀ : 0.375 = 3 / 8 := by norm_num
  have h₁ : log_base_2 (3 / 8) = log_base_2 3 - log_base_2 8 := by rw [log_div, log_base_2]
  have h₂ : log_base_2 8 = 3 := by norm_num [log_base_2, log10, log2]
  have h₃ : log10 3 ≈ 0.4771 := by norm_num
  have h₄ : log10 2 ≈ 0.3010 := by norm_num
  have h₅ : log_base_2 3 ≈ 0.4771 / 0.3010 := by rw [←h₃, ←h₄, log_base_2]
  have h₆ : (0.4771 / 0.3010 - 3) = -1.415 := by norm_num
  show log_base_2 0.375 = -1.415, by rw [←h₀, ←h₁, ←h₂, ←h₆]

#print log_base_2_of_0.375

end log_base_2_of_0_l658_658455


namespace liquid_level_ratio_l658_658338

noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
noncomputable def volume_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

theorem liquid_level_ratio
  (h1 h2 : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (h1_eq_4h2 : h1 = 4 * h2)
  (V1 V2 VM : ℝ)
  (V1_def : V1 = volume_cone 5 h1)
  (V2_def : V2 = volume_cone 10 h2)
  (VM_def : VM = volume_sphere 2)
  (V1_eq_V2 : V1 = V2) :
  let Δh1 := VM / (volume_cone 5 1 / h1),
      Δh2 := VM / (volume_cone 10 1 / h2)
  in Δh1 / Δh2 = 4 := 
by {
  sorry
}

end liquid_level_ratio_l658_658338


namespace average_salary_all_workers_l658_658301

-- Definitions based on the conditions
def num_technicians : ℕ := 7
def num_other_workers : ℕ := 7
def avg_salary_technicians : ℕ := 12000
def avg_salary_other_workers : ℕ := 8000
def total_workers : ℕ := 14

-- Total salary calculations based on the conditions
def total_salary_technicians : ℕ := num_technicians * avg_salary_technicians
def total_salary_other_workers : ℕ := num_other_workers * avg_salary_other_workers
def total_salary_all_workers : ℕ := total_salary_technicians + total_salary_other_workers

-- The statement to be proved
theorem average_salary_all_workers : total_salary_all_workers / total_workers = 10000 :=
by
  -- proof will be added here
  sorry

end average_salary_all_workers_l658_658301


namespace sum_vectors_centroid_midpoints_zero_l658_658231

-- Definitions related to the problem conditions
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C G D E F : V}

-- Existing conditions
def is_centroid (G : V) (A B C : V) : Prop :=
  G = (A + B + C) / 3

def is_midpoint (D : V) (A B : V) : Prop :=
  D = (A + B) / 2

-- The theorem to prove
theorem sum_vectors_centroid_midpoints_zero
  (hG : is_centroid G A B C)
  (hD : is_midpoint D B C)
  (hE : is_midpoint E C A)
  (hF : is_midpoint F A B) :
  (G - D) + (G - E) + (G - F) = 0 :=
sorry

end sum_vectors_centroid_midpoints_zero_l658_658231


namespace intersection_S_T_l658_658152

noncomputable def S : set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def T : set ℝ := {z | ∃ x : ℝ, z = -2 * x}
noncomputable def intersection (A B : set ℝ) := {y | y ∈ A ∧ y ∈ B}

theorem intersection_S_T : intersection S T = {y | y ∈ set.Ici 1} :=
by
  sorry

end intersection_S_T_l658_658152


namespace fat_ant_faster_l658_658067

/-- Two ants need to carry 150 g of cargo each from point A to point B, which is 15 meters away. 
    The Fat ant moves at 3 m/min and can carry 5 g of cargo at a time. The Thin ant moves at 5 m/min 
    and can carry 3 g of cargo at a time. The speed does not differ with or without the cargo.
    We need to prove that the Fat Ant will deliver all his load faster than the Thin Ant. -/
theorem fat_ant_faster : 
  ∀ (total_cargo distance : ℕ) (fat_speed fat_capacity thin_speed thin_capacity : ℕ),
    total_cargo = 150 → distance = 15 → fat_speed = 3 → fat_capacity = 5 → thin_speed = 5 → thin_capacity = 3 →
    (let fat_total_trips := 2 * (total_cargo / fat_capacity) - 1,
         thin_total_trips := 2 * (total_cargo / thin_capacity) - 1,
         fat_time := fat_total_trips * (distance / fat_speed),
         thin_time := thin_total_trips * (distance / thin_speed)
     in fat_time < thin_time) := 
begin
  intros total_cargo distance fat_speed fat_capacity thin_speed thin_capacity,
  intros h_total_cargo h_distance h_fat_speed h_fat_capacity h_thin_speed h_thin_capacity,
  have fat_total_trips := 2 * (total_cargo / fat_capacity) - 1,
  have thin_total_trips := 2 * (total_cargo / thin_capacity) - 1,
  have fat_time := fat_total_trips * (distance / fat_speed),
  have thin_time := thin_total_trips * (distance / thin_speed),
  let pf : fat_time < thin_time := sorry,
  exact pf,
end

end fat_ant_faster_l658_658067


namespace rect_prism_sum_is_correct_l658_658578

noncomputable def rect_prism_area_sum : ℕ :=
  let largest_area := 240
  let smallest_area := 48
  let possible_values := 
    (do
      let ab_vals := List.range 49 |>.filter (λ ab, 48 % ab == 0)
      let res := List.filterMap (λ (ab: ℕ),
        let a := ab
        let b := 48 / a
        let c := 240 / b
        if ab * c == 240 then
          if a * c != 48 ∧ a * c != 240 then some (a * c) else none
        else
          none
      ) ab_vals
      res.sum
    )
  possible_values 

theorem rect_prism_sum_is_correct : rect_prism_area_sum = 260 := sorry

end rect_prism_sum_is_correct_l658_658578


namespace simplify_expression_l658_658296

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x ^ 2 + 10 - (5 - 4 * x + 8 * x ^ 2) = -16 * x ^ 2 + 8 * x + 5 :=
by
  sorry

end simplify_expression_l658_658296


namespace param_values_a_l658_658221

noncomputable def same_side_of_line (a : ℝ) : Prop :=
  let A := (a / 2, a / 2)
  let B := (-2 * a, 2 / a)
  let f : (ℝ × ℝ) → ℝ := λ p, p.1 + p.2 - 3
  (f A) * (f B) > 0

theorem param_values_a (a : ℝ) : same_side_of_line a ↔
  (a ∈ Set.Ioo (-2 : ℝ) 0) ∨ (a ∈ Set.Ioo (1 / 2) 3) := sorry

end param_values_a_l658_658221


namespace problem_statement_l658_658975

theorem problem_statement (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (h_prime : Nat.Prime (m + k + 1)) (h_gt : m + k + 1 > n + 1) :
  (∏ i in Finset.range n, (m + 1 + i) * (m + 2 + i) - k * (k + 1)) ∣ (∏ i in Finset.range n, i * (i + 1)) :=
sorry

end problem_statement_l658_658975


namespace sum_of_valid_a_is_42_l658_658827

theorem sum_of_valid_a_is_42 :
  let A := {a ∈ ℕ | ∃ n m : ℕ, a = 2^n * 3^m ∧ ¬ (a^6 ∣ 6^a)} in ∑ a in A, a = 42 :=
by
  sorry

end sum_of_valid_a_is_42_l658_658827


namespace fraction_of_foreign_males_l658_658038

theorem fraction_of_foreign_males
  (total_students : ℕ)
  (female_ratio : ℚ)
  (non_foreign_males : ℕ)
  (foreign_male_fraction : ℚ)
  (h1 : total_students = 300)
  (h2 : female_ratio = 2/3)
  (h3 : non_foreign_males = 90) :
  foreign_male_fraction = 1/10 :=
by
  sorry

end fraction_of_foreign_males_l658_658038


namespace simplify_and_rationalize_correct_l658_658647

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l658_658647


namespace distinct_prime_factors_126_l658_658519

theorem distinct_prime_factors_126 : Nat.card {p ∈ Nat.prime_factors 126 | true} = 3 := by
  sorry

end distinct_prime_factors_126_l658_658519


namespace solvable_for_all_n_geq_3_l658_658056

open Nat

def solvable_system (n : ℕ) : Prop :=
  ∃ (x : Fin n → ℝ), (∑ k, x k = 27) ∧ (∏ k, x k = (3 / 2) ^ 24)

theorem solvable_for_all_n_geq_3 : ∀ n, n ≥ 3 → solvable_system n :=
by
  intros n hn
  sorry

end solvable_for_all_n_geq_3_l658_658056


namespace elixir_concentration_l658_658157

theorem elixir_concentration (x a : ℝ) 
  (h1 : (x * 100) / (100 + a) = 9) 
  (h2 : (x * 100 + a * 100) / (100 + 2 * a) = 23) : 
  x = 11 :=
by 
  sorry

end elixir_concentration_l658_658157


namespace amount_of_loan_l658_658638

def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem amount_of_loan (P : ℝ) :
  P = 632 * 100 / (5.93 * 5.93) :=
by
  sorry

end amount_of_loan_l658_658638


namespace find_a_range_of_a_l658_658140

noncomputable def f (x a : ℝ) := x + a * Real.log x

-- Proof problem 1: Prove that a = 2 given f' (1) = 3 for f (x) = x + a log x
theorem find_a (a : ℝ) : 
  (1 + a = 3) → (a = 2) := sorry

-- Proof problem 2: Prove that the range of a such that f(x) ≥ a always holds is [-e^2, 0]
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ a) → (-Real.exp 2 ≤ a ∧ a ≤ 0) := sorry

end find_a_range_of_a_l658_658140


namespace B_subset_A_A_inter_B_empty_l658_658896

noncomputable def setA (m : ℝ) : set ℝ := { x | (x + 2 * m) * (x - m + 4) < 0 }
noncomputable def setB : set ℝ := { x | (1 - x) / (x + 2) > 0 }

theorem B_subset_A {m : ℝ} : 
  (setB ⊆ setA m) ↔ (m ≥ 5 ∨ m ≤ -1 / 2) := 
sorry

theorem A_inter_B_empty {m : ℝ} :
  (setA m ∩ setB = ∅) ↔ (1 ≤ m ∧ m ≤ 2) := 
sorry

end B_subset_A_A_inter_B_empty_l658_658896


namespace not_proportional_eqn_exists_l658_658065

theorem not_proportional_eqn_exists :
  ∀ (x y : ℝ), (4 * x + 2 * y = 8) → ¬ ((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) :=
by
  intros x y h
  sorry

end not_proportional_eqn_exists_l658_658065


namespace regular_polygon_sides_l658_658916

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) :
  ∃ (n : ℕ), n > 2 ∧ (interior_angle = (n - 2) * 180 / n) :=
by
  sorry
 
end regular_polygon_sides_l658_658916


namespace complex_number_solution_l658_658193

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l658_658193


namespace closing_price_l658_658426

theorem closing_price (opening_price : ℝ) (percent_increase : ℝ) (closing_price : ℝ) 
  (h₀ : opening_price = 6) (h₁ : percent_increase = 0.3333) : closing_price = 8 :=
by
  sorry

end closing_price_l658_658426


namespace both_shots_missed_l658_658931

axiom p q : Prop
theorem both_shots_missed : ¬p ∧ ¬q := sorry

end both_shots_missed_l658_658931


namespace simplify_rationalize_expr_l658_658643

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l658_658643


namespace distance_earth_sun_approx_l658_658130

theorem distance_earth_sun_approx:
  let speed_of_light := 3 * 10^8 in
  let time_to_earth := 5 * 10^2 in
  speed_of_light * time_to_earth = 1.5 * 10^11 :=
by
  sorry

end distance_earth_sun_approx_l658_658130


namespace find_m_l658_658144

-- Define the hyperbola equation
def hyperbola1 (x y : ℝ) (m : ℝ) : Prop := (x^3 / m) - (y^2 / 3) = 1
def hyperbola2 (x y : ℝ) : Prop := (x^3 / 8) - (y^2 / 4) = 1

-- Define the condition for eccentricity equivalence
def same_eccentricity (m : ℝ) : Prop :=
  let e1_sq := 1 + (4 / 2^2)
  let e2_sq := 1 + (3 / m)
  e1_sq = e2_sq

-- The main theorem statement
theorem find_m (m : ℝ) : hyperbola1 x y m → hyperbola2 x y → same_eccentricity m → m = 6 :=
by
  -- Proof can be skipped with sorry to satisfy the statement-only requirement
  sorry

end find_m_l658_658144


namespace angle_B_sine_angle_A_l658_658200

variables (A B C : ℝ)
variables (a b c : ℝ)

-- Basic geometric conditions
def angle_opposite_sides (a b c : ℝ) : Prop := b * sin (C + π/6) = (a + c) / 2

-- Result from part (1)
theorem angle_B (h : angle_opposite_sides a b c) : B = π / 3 := 
  sorry

-- Extra conditions for part (2)
variables (M : ℝ)
def is_midpoint (M B C : ℝ) : Prop := 2 * M = B + C
def am_eq_ac (AM AC : ℝ) : Prop := AM = AC 

-- Result from part (2)
theorem sine_angle_A (h1 : is_midpoint M B C) (h2 : am_eq_ac a c) (h3 : B = π / 3) : sin A = sqrt 21 / 7 := 
  sorry

end angle_B_sine_angle_A_l658_658200


namespace higher_probability_winner_l658_658619

/--
Given a dice game where:
- Dupont and Durand take turns rolling two dice.
- Durand wins a point if the sum of the faces is 7.
- Dupont wins a point if the sum of the faces is 8.

Prove that Durand has a higher probability of winning compared to Dupont.
-/
theorem higher_probability_winner :
  let p7 := 6 / 36,  -- Probability of getting a sum of 7
      p8 := 5 / 36   -- Probability of getting a sum of 8
  in p7 > p8 :=
by {
  let p7 := 6 / 36,
  let p8 := 5 / 36,
  exact (by norm_num : (6 / 36) > (5 / 36))
}

end higher_probability_winner_l658_658619


namespace f_10_l658_658194

noncomputable def f : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * f n

theorem f_10 : f 10 = 2^10 :=
by
  -- This would be filled in with the necessary proof steps to show f(10) = 2^10
  sorry

end f_10_l658_658194


namespace range_x0_l658_658118

theorem range_x0 :
  ∀ x0 y0 : ℝ, (x0 - y0 - 2 = 0) →
  (∃ Q : ℝ×ℝ, (Q.1^2 + Q.2^2 = 1) ∧ (angle (0, 0) (x0, y0) Q = π / 6)) →
  (0 ≤ x0 ∧ x0 ≤ 2) :=
by
  intros x0 y0 hyp_line hyp_angle
  sorry

end range_x0_l658_658118


namespace least_n_base_seventeen_l658_658096

def base_five_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else Nat.digits 5 n

def base_nine_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else Nat.digits 9 n

def sum_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_digits (base_five_digits n)

def g (n : ℕ) : ℕ :=
  sum_digits (base_nine_digits (f n))

def in_base_seventeen_using_digits_0_to_9 (n : ℕ) : Bool :=
  ∀ digit ∈ Nat.digits 17 n, digit < 10

theorem least_n_base_seventeen : ℕ :=
  let M : ℕ := 4 * 5^10 - 1
  ∃ (n : ℕ), g(n) ≥ 10 ∧ ¬ in_base_seventeen_using_digits_0_to_9 (g n) ∧ M % 1000 = 500
  sorry

end least_n_base_seventeen_l658_658096


namespace find_first_image_l658_658113

-- Define the types we'll use
variable (Point Line : Type)

-- Define the conditions
variable (a : Line)               -- The original line
variable (a'' : Line)             -- The second image of the line
variable (a₂ : Line)              -- The second shadow of the line under 45° lighting

-- Define the function that returns the first image given the conditions
def first_image (a a'' a₂ : Line) : Line := sorry

-- The main theorem to state the problem
theorem find_first_image (a'' a₂ : Line) : ∃ a' : Line, first_image a a'' a₂ = a' :=
by
  sorry

end find_first_image_l658_658113


namespace recoloring_pattern_exists_1000_l658_658593

open Function

-- Define the problem in Lean
def recolor_rule (P neighbors : ℕ → Prop) : Prop :=
  ∀ i, P i ↔ (neighbors (i+1) mod 2001 = neighbors (i+2000) mod 2001)

def monochromatic_segment (F : ℕ → ℕ) (d : ℕ) : Prop :=
  ∃ (start : ℕ), ∀ j, j < d → F (start + j) = F start

def max_monochromatic_segment_length (F : ℕ → ℕ) : ℕ := sorry

theorem recoloring_pattern_exists_1000 (F : ℕ → ℕ) :
  (∃ n_0 ≤ 1000, F (n_0 + 2) = F n_0) ∧ 
  ¬ (∃ n_0 ≤ 999, F (n_0 + 2) = F n_0) :=
sorry

end recoloring_pattern_exists_1000_l658_658593


namespace max_movies_watched_l658_658516

-- Conditions given in the problem
def movie_duration : Nat := 90
def tuesday_minutes : Nat := 4 * 60 + 30
def tuesday_movies : Nat := tuesday_minutes / movie_duration
def wednesday_movies : Nat := 2 * tuesday_movies

-- Problem statement: Total movies watched in two days
theorem max_movies_watched : 
  tuesday_movies + wednesday_movies = 9 := 
by
  -- We add the placeholder for the proof here
  sorry

end max_movies_watched_l658_658516


namespace polynomial_real_roots_l658_658997

theorem polynomial_real_roots (a b c d e : ℝ) (h : 2 * a^2 < 5 * b) :
  ¬ ∀ x ∈ ({0, 1, 2, 3, 4, 5} : Fin 6 → ℝ), x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 :=
sorry

end polynomial_real_roots_l658_658997


namespace B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l658_658545

namespace GoGame

-- Define the players: A, B, C
inductive Player
| A
| B
| C

open Player

-- Define the probabilities as given
def P_A_beats_B : ℝ := 0.4
def P_B_beats_C : ℝ := 0.5
def P_C_beats_A : ℝ := 0.6

-- Define the game rounds and logic
def probability_B_winning_four_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
(1 - P_A_beats_B)^2 * P_B_beats_C^2

def probability_C_winning_three_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
  P_A_beats_B * P_C_beats_A^2 * P_B_beats_C + 
  (1 - P_A_beats_B) * P_B_beats_C^2 * P_C_beats_A

-- Proof statements
theorem B_wins_four_rounds_prob_is_0_09 : 
  probability_B_winning_four_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.09 :=
by
  sorry

theorem C_wins_three_rounds_prob_is_0_162 : 
  probability_C_winning_three_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.162 :=
by
  sorry

end GoGame

end B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l658_658545


namespace arithmetic_sequence_term_l658_658543

section ArithmeticSequence

variables (a : ℕ → ℕ)
variables (a_1 d : ℕ)

-- Given conditions
def sum_first_five_terms := (5:ℕ) // 2 * (2 * a_1 + 4 * d) = 25
def second_term := a 2 = a_1 + d = 3

-- Goal
theorem arithmetic_sequence_term :
  sum_first_five_terms →
  second_term →
  a 7 = a_1 + 6 * d := by
  sorry

end ArithmeticSequence

end arithmetic_sequence_term_l658_658543


namespace smallest_k_does_not_end_largest_k_always_ends_l658_658471

-- Define the game board and necessary properties
variables (n : ℕ) (k : ℕ)
-- n is the size of the game board and k is the number of game pieces

-- Assuming n ≥ 2
axiom (n_ge_two : n ≥ 2)

-- Problem (a): Determine the smallest k for which the game does not end for any initial distribution
theorem smallest_k_does_not_end : k = 3 * n^2 - 4 * n + 1 := 
  sorry

-- Problem (b): Determine the largest k for which the game always ends for any initial distribution
theorem largest_k_always_ends : k = 2 * n^2 - 2 * n - 1 := 
  sorry

end smallest_k_does_not_end_largest_k_always_ends_l658_658471


namespace problem_1_problem_2_l658_658848

noncomputable def ellipse_equation : Prop :=
  ∀ x y : ℝ, x^2 / 27 + y^2 / 36 = 1

noncomputable def hyperbola_condition : Prop :=
  ∀ x y : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9 ∧ 
    (y^2 / a^2 - x^2 / b^2 = 1) ∧ (y = 4 ∧ x = sqrt 15)

noncomputable def hyperbola_c : Prop :=
  ∀ x y : ℝ, y^2 / 4 - x^2 / 5 = 1

noncomputable def triangle_area : Prop :=
  let F1 : ℝ × ℝ := (0, -3) in
  let F2 : ℝ × ℝ := (0, 3) in
  let a := 2 in
  let b := sqrt 5 in
  let c := 3 in
  let angle := 120 in
  ∃ P : ℝ × ℝ, P ∈ { p : ℝ × ℝ | (p.2^2 / 4 - p.1^2 / 5 = 1) } ∧
  ∠ F1 P F2 = angle ∧
  let m := |P.1 - F1.1| + |P.2 - F1.2| in
  let n := |P.1 - F2.1| + |P.2 - F2.2| in
  (m - n)^2 / 2 + mn = 36 ∧
  (1 / 2) * (m * n) * sin 120 = 5 * sqrt 3 / 3

theorem problem_1 (h : ellipse_equation ∧ hyperbola_condition) : hyperbola_c :=
  sorry

theorem problem_2 (h : hyperbola_c) : triangle_area :=
  sorry

end problem_1_problem_2_l658_658848


namespace lotion_cost_l658_658241

variable (shampoo_conditioner_cost lotion_total_spend: ℝ)
variable (num_lotions num_lotions_cost_target: ℕ)
variable (free_shipping_threshold additional_spend_needed: ℝ)

noncomputable def cost_of_each_lotion := lotion_total_spend / num_lotions

theorem lotion_cost
    (h1 : shampoo_conditioner_cost = 10)
    (h2 : num_lotions = 3)
    (h3 : additional_spend_needed = 12)
    (h4 : free_shipping_threshold = 50)
    (h5 : (shampoo_conditioner_cost * 2) + additional_spend_needed + lotion_total_spend = free_shipping_threshold) :
    cost_of_each_lotion = 10 :=
by
  sorry

end lotion_cost_l658_658241


namespace cost_price_per_meter_l658_658371

-- We define the given conditions
def meters_sold : ℕ := 60
def selling_price : ℕ := 8400
def profit_per_meter : ℕ := 12

-- We need to prove that the cost price per meter is Rs. 128
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 128 :=
by
  sorry

end cost_price_per_meter_l658_658371


namespace bisect_segment_of_tangents_on_circle_l658_658622

-- Definitions based on the problem statement
variable {α : Type*} [MetricSpace α]
variable (circle : @circle α) (A B M N T1 T2 : α)
variable (MA NB : ℝ)

-- Conditions given in the problem statement
def chord_ext_equal_segments (A B M N : α) (MA NB : ℝ) : Prop :=
  distance M A = MA ∧ distance N B = NB

def tangents_from_points (circle : @circle α) (M N T1 T2 : α) : Prop :=
  circle.is_tangent M T1 ∧ circle.is_tangent N T2 ∧ M ≠ N

def tangents_opposite_sides (M N T1 T2 : α) : Prop :=
  ∠ M N T1 = 90° ∧ ∠ N M T2 = 90°

-- Lean statement for the mathematical proof problem
theorem bisect_segment_of_tangents_on_circle
  (circle : @circle α) (A B M N T1 T2 : α) (MA NB : ℝ)
  (H1 : chord_ext_equal_segments A B M N MA NB) 
  (H2 : tangents_from_points circle M N T1 T2)
  (H3 : tangents_opposite_sides M N T1 T2) :
  bisects_segment T1 T2 M N :=
by
  sorry

end bisect_segment_of_tangents_on_circle_l658_658622


namespace probability_two_blue_marbles_l658_658764

theorem probability_two_blue_marbles (h_red: ℕ := 3) (h_blue: ℕ := 4) (h_white: ℕ := 9) :
  (h_blue / (h_red + h_blue + h_white)) * ((h_blue - 1) / ((h_red + h_blue + h_white) - 1)) = 1 / 20 :=
by sorry

end probability_two_blue_marbles_l658_658764


namespace height_from_side_AC_l658_658568

theorem height_from_side_AC (A B C : Type) [EuclideanGeometry A B C] 
  (AB AC BC : ℝ) (hAB : AB = 3) (hAC : AC = 4) (hBC : BC = sqrt 13) :
  ∃ (h : ℝ), h = (3 / 2) * sqrt 3 :=
by
  -- Assume a perpendicular from B to AC and let the intersection point be D
  -- Let AD = x, CD = 4 - x
  -- Solve for x using Pythagorean theorem. Eventually prove that height is 3/2 * sqrt 3
  sorry

end height_from_side_AC_l658_658568


namespace trees_planted_l658_658782

theorem trees_planted (yard_length : ℕ) (distance_between_trees : ℕ) (n_trees : ℕ) 
  (h1 : yard_length = 434) 
  (h2 : distance_between_trees = 14) 
  (h3 : n_trees = yard_length / distance_between_trees + 1) : 
  n_trees = 32 :=
by
  sorry

end trees_planted_l658_658782


namespace distance_Y_to_B_l658_658451

-- Definitions
variables (A B C Y : Type) [euclidean_space A B C Y]
variable h_eq_triangle : equilateral_triangle A B C
variable h_side_length : ∀s [side s], length s = 5
variable M : Type [midpoint B C M]
variable h_AM_perpendicular : perpendicular A (line_segment B C)
variable h_MY_length : length (line_segment M Y) = 1 / 2

-- Proof Problem
theorem distance_Y_to_B (h_eq_triangle : equilateral_triangle A B C)
  (h_side_length : ∀(s [side s]), length s = 5)
  (h_midpoint : midpoint B C M)
  (h_AM_perpendicular : perpendicular (line_segment A M) (line_segment B C))
  (h_MY_length : length (line_segment M Y) = 1 / 2) : 
  (distance Y B = 2) :=
sorry

end distance_Y_to_B_l658_658451


namespace largest_integer_square_in_base_7_has_four_digits_l658_658269

noncomputable def largest_integer_square_has_four_digits_in_base_7 : ℕ :=
  let m := 48 in m

theorem largest_integer_square_in_base_7_has_four_digits :
  let M := largest_integer_square_has_four_digits_in_base_7 in
  base_to_nat 7 66 = M :=
by
  sorry

end largest_integer_square_in_base_7_has_four_digits_l658_658269


namespace symmetric_point_l658_658187

theorem symmetric_point :
  ∀ (a : ℝ), a < 0 → let A := (a, 4) in let m := 2 in (4 - a, 4) = (4 - a, 4) :=
by
  intros a h
  dsimp
  sorry

end symmetric_point_l658_658187


namespace second_closest_location_l658_658672
-- Import all necessary modules from the math library

-- Define the given distances (conditions)
def distance_library : ℝ := 1.912 * 1000  -- distance in meters
def distance_park : ℝ := 876              -- distance in meters
def distance_clothing_store : ℝ := 1.054 * 1000  -- distance in meters

-- State the proof problem
theorem second_closest_location :
  (distance_library = 1912) →
  (distance_park = 876) →
  (distance_clothing_store = 1054) →
  (distance_clothing_store = 1054) :=
by
  intros h1 h2 h3
  -- sorry to skip the proof
  sorry

end second_closest_location_l658_658672


namespace closest_fraction_l658_658787

theorem closest_fraction (won : ℚ) (options : List ℚ) (closest : ℚ) 
  (h_won : won = 25 / 120) 
  (h_options : options = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]) 
  (h_closest : closest = 1 / 5) :
  ∃ x ∈ options, abs (won - x) = abs (won - closest) := 
sorry

end closest_fraction_l658_658787


namespace complex_number_pure_imaginary_l658_658864

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem complex_number_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = complex.I)
  (h2 : is_pure_imaginary ((1 + a * i) * (2 + i))) : a = 2 :=
by
  sorry

end complex_number_pure_imaginary_l658_658864


namespace sum_of_distances_constant_l658_658996

-- Define the problem context
def regular_polygon (n : ℕ) : Type := sorry  -- Definition of a regular polygon with n sides
def inside_point (P : Point) (polygon : regular_polygon n) : Prop := sorry  -- P is a point inside a regular polygon
def perpendicular_distance (P : Point) (line : Line) : ℝ := sorry  -- Perpendicular distance from point P to line

-- The actual theorem we wish to prove
theorem sum_of_distances_constant
  (n : ℕ) (polygon : regular_polygon n)
  (a : ℝ) (S : ℝ) 
  (h_side_length : ∀ i, side_length polygon i = a)
  (h_area : area polygon = S)
  (P : Point) (h_inside : inside_point P polygon):
    (∑ i in finset.range n, perpendicular_distance P (side_line polygon i)) = 2 * S / a := 
sorry

end sum_of_distances_constant_l658_658996


namespace pascal_triangle_row_20_element_5_l658_658722

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l658_658722


namespace unique_not_in_range_of_g_l658_658806

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g (p q r s : ℝ) (hps_qr_zero : p * s + q * r = 0) 
  (hpr_rs_zero : p * r + r * s = 0) (hg3 : g p q r s 3 = 3) 
  (hg81 : g p q r s 81 = 81) (h_involution : ∀ x ≠ (-s / r), g p q r s (g p q r s x) = x) :
  ∀ x : ℝ, x ≠ 42 :=
sorry

end unique_not_in_range_of_g_l658_658806


namespace RebeccaHasTwentyMarbles_l658_658999

variable (groups : ℕ) (marbles_per_group : ℕ) (total_marbles : ℕ)

def totalMarbles (g m : ℕ) : ℕ :=
  g * m

theorem RebeccaHasTwentyMarbles
  (h1 : groups = 5)
  (h2 : marbles_per_group = 4)
  (h3 : total_marbles = totalMarbles groups marbles_per_group) :
  total_marbles = 20 :=
by {
  sorry
}

end RebeccaHasTwentyMarbles_l658_658999


namespace range_H_l658_658347

noncomputable def H (x : ℝ) : ℝ :=
  abs (x + 2) - abs (x - 2)

theorem range_H : set.range H = set.Icc (-4) 4 := by
  sorry

end range_H_l658_658347


namespace count_less_than_one_is_3_l658_658173

def listOfNumbers : List ℝ := [0.03, 1.5, -0.2, 0.76]

def countLessThanOne (lst : List ℝ) : Nat :=
  lst.count (λ x => x < 1)

theorem count_less_than_one_is_3 :
  countLessThanOne listOfNumbers = 3 := by
  sorry

end count_less_than_one_is_3_l658_658173


namespace find_a_l658_658190

open Complex

theorem find_a (a : ℝ) (h : (⟨a, 1⟩ * ⟨1, -a⟩ = 2)) : a = 1 :=
sorry

end find_a_l658_658190


namespace greatest_integer_bound_l658_658810

def integer_bound_expression : ℤ :=
  let numerator   := 4 ^ 100 + 3 ^ 100
  let denominator := 4 ^ 94 + 3 ^ 94
  let quotient    := (numerator / denominator : ℝ)
  ⌊quotient⌋

theorem greatest_integer_bound :
  integer_bound_expression ≤ 4096 := by sorry

end greatest_integer_bound_l658_658810


namespace weight_second_pair_l658_658429

-- Given conditions
def weight_first_pair : ℕ := 3
def weight_third_pair : ℕ := 8
def total_weight_system : ℕ := 32

-- The statement to prove
theorem weight_second_pair : (weight_first_pair * 2) + (weight_third_pair * 2) + ?weight_second * 2 = total_weight_system → ?weight_second = 5 := by
  sorry

end weight_second_pair_l658_658429


namespace sum_of_60_nonconsecutive_odds_l658_658728

def sequence_arithmetic (n : ℕ) :=
  ∀ k : ℕ, (k < n) → ∃ (a : ℤ), a = -29 + 4 * k

theorem sum_of_60_nonconsecutive_odds :
  sequence_arithmetic 60 →
  ∑ k in Finset.range 60, (-29 + 4 * k) = 5340 :=
by sorry

end sum_of_60_nonconsecutive_odds_l658_658728


namespace find_ordered_pair_l658_658452

theorem find_ordered_pair (x y : ℤ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - y = (x - 2) + (y - 2))
  : (x, y) = (5, 2) := 
sorry

end find_ordered_pair_l658_658452


namespace simplify_rationalize_expr_l658_658642

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l658_658642


namespace part1_part2_l658_658201

variable (A B C: ℝ)
variable (a b c : ℝ)
variable (triangle_ABC : a = cos A ∧ b = cos B ∧ c = cos C)
variable (cos_A_eq : cos A = 1/2)

theorem part1 
  (h1 : (2 * sin C - sin B) / sin B = (a * cos B) / (b * cos A) )
  (h2 : a = cos A) (h3 : b = cos B) (h4 : c = cos C) : 
  A = π/3 :=
sorry

theorem part2 
  (h1 : a = 3) (h2 : sin C = 2 * sin B):
  b = sqrt 3 ∧ c = 2 * sqrt 3 :=
sorry

end part1_part2_l658_658201


namespace prob_both_hit_prob_only_one_hit_prob_at_least_one_hit_l658_658319

-- Definitions for shooters' probabilities
variables {p1 p2 : ℝ} (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)

-- Proposition 1: Probability that both shooters hit the target
theorem prob_both_hit (h_indep : ∀ A B, independent A B) :
  p1 * p2 = sorry :=
by sorry

-- Proposition 2: Probability that only one shooter hits the target
theorem prob_only_one_hit :
  (p1 * (1 - p2)) + ((1 - p1) * p2) = p1 + p2 - 2 * p1 * p2 :=
by sorry

-- Proposition 3: Probability that at least one shooter hits the target
theorem prob_at_least_one_hit :
  1 - (1 - p1) * (1 - p2) = p1 + p2 - p1 * p2 :=
by sorry

end prob_both_hit_prob_only_one_hit_prob_at_least_one_hit_l658_658319


namespace sequence_has_minus_one_iff_l658_658322

def sequence (p : ℕ) (u : ℕ → ℤ) : Prop :=
  u 0 = 0 ∧ u 1 = 1 ∧ ∀ n, u (n + 2) = 2 * u (n + 1) - p * u n

def is_odd_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ p % 2 = 1

theorem sequence_has_minus_one_iff (p : ℕ) (u : ℕ → ℤ)
  (hp : is_odd_prime p) (seq : sequence p u) :
  (∃ n, u n = -1) ↔ p = 5 := sorry

end sequence_has_minus_one_iff_l658_658322


namespace base9_4318_is_base10_3176_l658_658435

def base9_to_base10 (n : Nat) : Nat :=
  let d₀ := (n % 10) * 9^0
  let d₁ := ((n / 10) % 10) * 9^1
  let d₂ := ((n / 100) % 10) * 9^2
  let d₃ := ((n / 1000) % 10) * 9^3
  d₀ + d₁ + d₂ + d₃

theorem base9_4318_is_base10_3176 :
  base9_to_base10 4318 = 3176 :=
by
  sorry

end base9_4318_is_base10_3176_l658_658435


namespace rope_piece_length_l658_658661

-- Define the conditions
def rope_length_ft := 6
def rope_cost := 5
def indiv_rope_length_ft := 1
def indiv_rope_cost := 1.25
def pieces := 10
def min_cost := 5

-- State the theorem
theorem rope_piece_length :
  (rope_cost = min_cost) →
  (rope_length_ft * 12 / pieces = 7.2) :=
by
  intros
  sorry

end rope_piece_length_l658_658661


namespace find_m_integers_l658_658895

open Set

noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B (m : ℚ) : Set ℤ := {x : ℤ | m * x + 1 = 0}

theorem find_m_integers : {m : ℚ | A ∪ B m = A} = {0, 1, -1/2} :=
by
  sorry

end find_m_integers_l658_658895


namespace locus_of_centers_of_circumscribed_rectangles_l658_658824

theorem locus_of_centers_of_circumscribed_rectangles (ABC : Triangle) (h_acute : ABC.isAcute) :
  locus_of_centers_of_rectangles_circumscribed_around_ABC (ABC) =
  curvilinear_triangle_constrained_by_semicircles_on_midsegments (ABC) :=
sorry

end locus_of_centers_of_circumscribed_rectangles_l658_658824


namespace pond_depth_l658_658935

theorem pond_depth (L W V : ℝ) (hL : L = 20) (hW : W = 12) (hV : V = 1200) :
  ∃ D : ℝ, V = L * W * D ∧ D = 5 :=
by
  exists 5
  rw [hL, hW, hV]
  norm_num
  sorry

end pond_depth_l658_658935


namespace change_combinations_l658_658523

def isValidCombination (nickels dimes quarters : ℕ) : Prop :=
  nickels * 5 + dimes * 10 + quarters * 25 = 50 ∧ quarters ≤ 1

theorem change_combinations : {n // ∃ (combinations : ℕ) (nickels dimes quarters : ℕ), 
  n = combinations ∧ isValidCombination nickels dimes quarters ∧ 
  ((nickels, dimes, quarters) = (10, 0, 0) ∨
   (nickels, dimes, quarters) = (8, 1, 0) ∨
   (nickels, dimes, quarters) = (6, 2, 0) ∨
   (nickels, dimes, quarters) = (4, 3, 0) ∨
   (nickels, dimes, quarters) = (2, 4, 0) ∨
   (nickels, dimes, quarters) = (0, 5, 0) ∨
   (nickels, dimes, quarters) = (5, 0, 1) ∨
   (nickels, dimes, quarters) = (3, 1, 1) ∨
   (nickels, dimes, quarters) = (1, 2, 1))}
  :=
  ⟨9, sorry⟩

end change_combinations_l658_658523


namespace john_bought_correct_number_of_packs_l658_658246

-- Define the number of packs per student
def packs_per_student := 4

-- Define the number of students in each class
def num_students_class1 := 24
def num_students_class2 := 18
def num_students_class3 := 30
def num_students_class4 := 20
def num_students_class5 := 28

-- Define the additional packs
def additional_packs := 10

-- Calculate total packs a teacher bought
def total_packs_bought := 
  num_students_class1 * packs_per_student +
  num_students_class2 * packs_per_student +
  num_students_class3 * packs_per_student +
  num_students_class4 * packs_per_student +
  num_students_class5 * packs_per_student +
  additional_packs

-- The statement that needs to be proved
theorem john_bought_correct_number_of_packs : total_packs_bought = 490 :=
by 
  rw [total_packs_bought]
  rw [num_students_class1, num_students_class2, num_students_class3, num_students_class4, num_students_class5, packs_per_student, additional_packs]
  norm_num
  sorry -- Placeholder for further steps if needed

end john_bought_correct_number_of_packs_l658_658246


namespace decreasing_range_of_m_l658_658539

theorem decreasing_range_of_m (m : ℝ) : (∀ x : ℝ, 1 < x → deriv (λ x, (x - m) / (x - 1)) x < 0) ↔ m < 1 :=
by
  sorry

end decreasing_range_of_m_l658_658539


namespace sum_of_possible_radii_l658_658388

-- Define the geometric and algebraic conditions of the problem
noncomputable def circleTangentSum (r : ℝ) : Prop :=
  let center_C := (r, r)
  let center_other := (3, 3)
  let radius_other := 2
  (∃ r : ℝ, (r > 0) ∧ ((center_C.1 - center_other.1)^2 + (center_C.2 - center_other.2)^2 = (r + radius_other)^2))

-- Define the theorem statement
theorem sum_of_possible_radii : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ circleTangentSum r1 ∧ circleTangentSum r2 ∧ r1 + r2 = 16 :=
sorry

end sum_of_possible_radii_l658_658388


namespace part1_1_part1_2_part2_part3_sum_of_integers_l658_658990

-- 1. Prove the equality for the first blank filling problem
theorem part1_1 (x : ℝ) (hx : x ≠ -1) : (3 * x + 5) / (x + 1) = 3 + 2 / (x + 1) :=
by
  sorry

-- 2. Prove the equality for the second blank filling problem
theorem part1_2 (a b c x : ℝ) (hx : x ≠ -c) : (a * x + b) / (x + c) = a + (b - a * c) / (x + c) :=
by
  sorry

-- 3. Find the minimum value of the fraction where x ≥ 0
theorem part2 : ∃ x, x ≥ 0 ∧ ∀ y, y ≥ 0 → (2 * x - 8) / (x + 2) ≤ (2 * y - 8) / (y + 2) :=
by
  use 0
  sorry

-- 4. Sum of all integer values of x that make the fraction an integer
theorem part3_sum_of_integers : ∑ x in ({-4, -2, 0, 2} : Finset ℤ), x = -4 :=
by
  sorry

end part1_1_part1_2_part2_part3_sum_of_integers_l658_658990


namespace circle_intersects_y_axis_length_l658_658387

theorem circle_intersects_y_axis_length (A B C M N : ℝ × ℝ)
    (hA : A = (1, 3)) (hB : B = (4, 2)) (hC : C = (1, -7))
    (hM : M.fst = 0) (hN : N.fst = 0)
    (h_circ : ∃ D E F, (∀ P : ℝ × ℝ, (P = A ∨ P = B ∨ P = C) →
      P.fst^2 + P.snd^2 + D * P.fst + E * P.snd + F = 0))
    (h_y_int : ∀ x : ℝ, x = 0 → ∃ y1 y2 : ℝ, (0, y1) = M ∧ (0, y2) = N ∧
      x^2 + y1^2 + D * x + E * y1 + F = 0 ∧ x^2 + y2^2 + D * x + E * y2 + F = 0) :
  abs ((M.snd - N.snd)) = 4 * real.sqrt 6 :=
sorry

end circle_intersects_y_axis_length_l658_658387
