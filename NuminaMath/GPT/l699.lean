import Mathlib

namespace calc_expr_l699_699223

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l699_699223


namespace T_positive_l699_699408

noncomputable def T (a : Real) : Real :=
  (Real.sin a + Real.tan a) / (Real.cos a + Real.cot a)

theorem T_positive (a : Real) (h : ¬ ∃ (k : Int), a = k * Real.pi / 2) : T a > 0 := 
sorry

end T_positive_l699_699408


namespace spinner_prob_l699_699563

theorem spinner_prob (PD PE PF_PG : ℚ) (hD : PD = 1/4) (hE : PE = 1/3) 
  (hTotal : PD + PE + PF_PG = 1) : PF_PG = 5/12 := by
  sorry

end spinner_prob_l699_699563


namespace relationship_M_N_l699_699639

theorem relationship_M_N (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) 
  (M : ℝ) (hM : M = a * b) (N : ℝ) (hN : N = a + b - 1) : M > N :=
by
  sorry

end relationship_M_N_l699_699639


namespace sin_square_identity_l699_699177

theorem sin_square_identity (x : ℝ) : sin(43 * (π / 180)) ^ 2 + sin(133 * (π / 180)) ^ 2 = 1 :=
by
  sorry

end sin_square_identity_l699_699177


namespace Yura_catches_up_in_five_minutes_l699_699170

-- Define the speeds and distances
variables (v_Lena v_Yura d_Lena d_Yura : ℝ)
-- Assume v_Yura = 2 * v_Lena (Yura is twice as fast)
axiom h1 : v_Yura = 2 * v_Lena 
-- Assume Lena walks for 5 minutes before Yura starts
axiom h2 : d_Lena = v_Lena * 5
-- Assume they walk at constant speeds
noncomputable def t_to_catch_up := 10 / 2 -- time Yura takes to catch up Lena

-- Define the proof problem
theorem Yura_catches_up_in_five_minutes :
    t_to_catch_up = 5 :=
by
    sorry

end Yura_catches_up_in_five_minutes_l699_699170


namespace binomial_expansion_equivalence_l699_699085

-- Define our given conditions
variables (p : ℕ) (a b : ℝ)

-- State the main theorem to be proven
theorem binomial_expansion_equivalence (hp : p > 0) :
  (∑ i in finset.range(p+1).filter (λ i, i ≠ 0), (nat.choose p i) * (nat.choose p i) * a^(p-i) * b^i) =
  (∑ i in finset.range(p+1), (nat.choose p i) * (nat.choose (i+1) i) * (a-b)^(p-i) * b^i) :=
sorry

end binomial_expansion_equivalence_l699_699085


namespace each_kid_uses_140_bags_of_corn_seeds_l699_699973

theorem each_kid_uses_140_bags_of_corn_seeds
  (ears_per_row : ℕ)
  (seeds_per_bag : ℕ)
  (seeds_per_ear : ℕ)
  (pay_per_row : ℝ)
  (dinner_cost : ℝ)
  (money_spent_on_dinner : ℝ)
  (money_earned_per_kid : ℝ)
  (rows_per_kid : ℕ)
  (seeds_per_row : ℕ)
  (total_seeds_per_kid : ℕ)
  (bags_per_kid : ℕ) :
  ears_per_row = 70 →
  seeds_per_bag = 48 →
  seeds_per_ear = 2 →
  pay_per_row = 1.5 →
  dinner_cost = 36 →
  money_spent_on_dinner = dinner_cost / 2 →
  money_earned_per_kid = dinner_cost * 2 →
  rows_per_kid = money_earned_per_kid / pay_per_row →
  seeds_per_row = ears_per_row * seeds_per_ear →
  total_seeds_per_kid = rows_per_kid * seeds_per_row →
  bags_per_kid = total_seeds_per_kid / seeds_per_bag →
  bags_per_kid = 140 :=
  by
    intros,
    sorry

end each_kid_uses_140_bags_of_corn_seeds_l699_699973


namespace sum_of_thirteen_equal_to_8900098_l699_699032

theorem sum_of_thirteen_equal_to_8900098 (N : ℕ) (H : ∀ i : fin 13, (i : ℕ) = N) :
  (13 * N = 8900098) → false :=
begin
  sorry
end

end sum_of_thirteen_equal_to_8900098_l699_699032


namespace right_triangle_area_proof_l699_699923

noncomputable def right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1 / 2) * a * b

theorem right_triangle_area_proof (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 40) (hc : c = 41) :
right_triangle_area a b c h = 180 :=
by
  rw [ha, hc]
  have hb := sqrt (c^2 - a^2)
  sorry

end right_triangle_area_proof_l699_699923


namespace find_f_4_1981_l699_699461

noncomputable def f : ℕ × ℕ → ℕ
| (0, y)       := y + 1
| (x+1, 0)     := f (x, 1)
| (x+1, y+1)   := f (x, f (x+1, y))

theorem find_f_4_1981 : f (4, 1981) = 2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^{2^2}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} - 3 := 
sorry

end find_f_4_1981_l699_699461


namespace collinear_probability_5x5_l699_699369

theorem collinear_probability_5x5 :
  let total_ways := (25.choose 4) in
  let collinear_sets := 28 in
  (collinear_sets : ℚ) / total_ways = 4 / 1807 :=
by 
  sorry

end collinear_probability_5x5_l699_699369


namespace range_of_a_l699_699671

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x^2

theorem range_of_a (a : ℝ) (x : ℝ) (h : 0 < x) : 
    (f x a ≤ a - 1) ↔ (a ∈ set.Ici (1 / 2)) :=
by
  sorry

end range_of_a_l699_699671


namespace squares_ap_if_reciprocals_ap_l699_699852

variable (a b c : ℝ)

-- Given: The reciprocals of sums are in arithmetic progression
def reciprocals_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * (1 / (c + a)) = (1 / (b + c)) + (1 / (b + a))

-- To prove: The squares form an arithmetic progression
def squares_arithmetic_progression (a b c : ℝ) : Prop :=
  a^2 + c^2 = 2 * b^2

theorem squares_ap_if_reciprocals_ap (h : reciprocals_arithmetic_progression a b c) :
  squares_arithmetic_progression a b c :=
begin
  sorry
end

end squares_ap_if_reciprocals_ap_l699_699852


namespace basketball_surface_area_l699_699180

theorem basketball_surface_area (d : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) :
  d = 24 → r = d / 2 → A = 4 * π * r ^ 2 → A = 576 * π :=
by { intros h1 h2 h3, sorry }

end basketball_surface_area_l699_699180


namespace length_F_to_F_l699_699497

def point := (ℝ × ℝ)

def reflect_y (p : point) : point := (-p.1, p.2)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_F_to_F'_is_10 : 
  let F : point := (-5, 3)
  let F' : point := reflect_y F
  distance F F' = 10 :=
by
  sorry

end length_F_to_F_l699_699497


namespace simplest_square_root_l699_699574

def sqrt_4 : ℝ := real.sqrt 4
def cube_root_5 : ℝ := real.cbrt 5
def sqrt_3 : ℝ := real.sqrt 3
def sqrt_1_9 : ℝ := real.sqrt (1 / 9)

theorem simplest_square_root : sqrt_3 = real.sqrt 3 :=
by
  sorry

end simplest_square_root_l699_699574


namespace problem_equivalent_l699_699736

noncomputable def curve_C1_parametric (α : ℝ) : ℝ × ℝ :=
(x : ℝ) (y : ℝ) := (2 * Real.cos α, 2 + 2 * Real.sin α) 

def curve_C1_polar (θ : ℝ) : ℝ := 4 * Real.sin θ

def curve_C2_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)

def curve_C2_polar (θ : ℝ) : ℝ := 8 * Real.sin θ

def curve_theta_polar (ρ : ℝ) : ℝ := 
  ∀ θ, θ = Real.pi / 3 → ρ > 0 → ρ

def point_A : ℝ := 4 * Real.sin (Real.pi / 3)
def point_B : ℝ := 8 * Real.sin (Real.pi / 3)

def distance_AB : ℝ := |point_B - point_A|

theorem problem_equivalent :
  curve_C1_polar = 4 * Real.sin θ ∧
  distance_AB = 2 * Real.sqrt 3 := sorry

end problem_equivalent_l699_699736


namespace number_in_100th_position_l699_699077

def bidirectional_bubble_sort (sequence : List ℕ) : List ℕ :=
  -- Function definition for bidirectional bubble sort (not implemented here)
  sorry

theorem number_in_100th_position (numbers : List ℕ)
  (h_len: numbers.length = 1982)
  (h_unique: numbers = List.range (1983).tail)
  (h_sorted: bidirectional_bubble_sort numbers = numbers)
  (h_stationary: (∀ seq, bidirectional_bubble_sort seq = seq → seq.get! 99 = numbers.get! 99)) :
  numbers.get! 99 = 100 :=
  sorry

end number_in_100th_position_l699_699077


namespace permutation_fixed_points_sum_l699_699753

-- Definition of the set S with n elements and the function p_n(k)
def S (n : ℕ) : Type := Fin n

def p_n (n k : ℕ) : ℕ := sorry -- This would be the function that returns the number of permutations of S with exactly k fixed points

-- The main statement to prove
theorem permutation_fixed_points_sum (n : ℕ) : 
  (∑ k in Finset.range (n + 1), k * p_n n k) = nat.factorial n :=
sorry

end permutation_fixed_points_sum_l699_699753


namespace power_eq_fractions_l699_699001

theorem power_eq_fractions (x : ℝ) (h : 81^4 = 27^x) : 3^(-x) = 1 / 3^(16/3) :=
by {
  sorry
}

end power_eq_fractions_l699_699001


namespace boy_reaches_school_early_l699_699502

theorem boy_reaches_school_early (usual_time : ℕ) (factor : ℚ) (new_time : ℕ) : 
  usual_time = 49 ∧ factor = 7/6 ∧ new_time = (usual_time * 6) / 7 → (usual_time - new_time = 7) :=
by
  intros h
  cases h with h1 h_factors
  cases h_factors with h2 h3
  rw h1 at h3
  rw h2 at h3
  have : new_time = 42 := sorry
  rw this
  have : usual_time = 49 := by assumption
  rw this
  norm_num

end boy_reaches_school_early_l699_699502


namespace find_angle_BCD_l699_699024

variable {A B C D : Type} [EuclideanSpace A]
variable [AB : Real]
variable [AC : Real]
variable [AD : Real]
variable [BAD : Real]

theorem find_angle_BCD (hAB : AB = 1) (hAC : AC = 1) (hAD : AD = 1) (hBAD : BAD = 100) :
  BCD = 130 :=
sorry

end find_angle_BCD_l699_699024


namespace percent_non_cyclists_play_basketball_l699_699582

theorem percent_non_cyclists_play_basketball
  (N : ℕ) 
  (h1 : 0.75 * N = A) 
  (h2 : 0.45 * N = B) 
  (h3 : 0.6 * A = C) 
  : (0.3 * N / 0.55 * N) * 100 ≈ 55 := 
sorry

end percent_non_cyclists_play_basketball_l699_699582


namespace polyhedron_edge_ratio_l699_699555

theorem polyhedron_edge_ratio (x y : ℝ) (F V E : ℕ) 
  (hF : F = 12) (hV : V = 8) (hE : E = 18)
  (euler_formula : V - E + F = 2)
  (triakis_tetrahedron : ∀ (f : ℕ), f ∈ [12] → (all_faces_is_isosceles_triangles f ∧ 
  all_edges_have_length (x ∨ y) ∧ 
  at_each_vertex (edges_meet (3 ∨ 6)) ∧ 
  all_dihedral_angles_are_equal)) :
  x / y = 3 / 5 :=
by sorry

end polyhedron_edge_ratio_l699_699555


namespace monotonic_intervals_g_exactly_two_tangent_points_existence_l699_699678

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then exp x else -x^2 + 2*x - 1/2

def g : ℝ → ℝ :=
λ x, x * f x

theorem monotonic_intervals_g :
  (∀ x, x < -1 → deriv g x < 0) ∧
  (∀ x, -1 < x ∧ x ≤ 0 → deriv g x > 0) ∧
  (∀ x, 0 < x ∧ x < (4 - real.sqrt 10) / 6 → deriv g x < 0) ∧
  (∀ x, (4 - real.sqrt 10) / 6 < x ∧ x < (4 + real.sqrt 10) / 6 → deriv g x > 0) ∧
  (∀ x, x > (4 + real.sqrt 10) / 6 → deriv g x < 0) :=
sorry

theorem exactly_two_tangent_points_existence :
  ∃ t ∈ Ioc 0 1, t^2 - 8*t + 4*t*real.log t + 2 = 0 :=
sorry

end monotonic_intervals_g_exactly_two_tangent_points_existence_l699_699678


namespace shaded_area_l699_699565

-- Conditions
def side_length_square : ℝ := 12
def radius_quarter_circle : ℝ := side_length_square / 4

-- Areas
def area_square : ℝ := side_length_square ^ 2
def area_full_circle : ℝ := Real.pi * radius_quarter_circle ^ 2

-- Statement
theorem shaded_area : (area_square - area_full_circle) = 144 - 9 * Real.pi := by
  sorry

end shaded_area_l699_699565


namespace triangles_not_necessarily_congruent_l699_699384

-- Define the triangles and their properties
structure Triangle :=
  (A B C : ℝ)

-- Define angles and measures for heights and medians
def angle (t : Triangle) : ℝ := sorry
def height_from (t : Triangle) (v : ℝ) : ℝ := sorry
def median_from (t : Triangle) (v : ℝ) : ℝ := sorry

theorem triangles_not_necessarily_congruent
  (T₁ T₂ : Triangle)
  (h_angle : angle T₁ = angle T₂)
  (h_height : height_from T₁ T₁.B = height_from T₂ T₂.B)
  (h_median : median_from T₁ T₁.C = median_from T₂ T₂.C) :
  ¬ (T₁ = T₂) := 
sorry

end triangles_not_necessarily_congruent_l699_699384


namespace A_finishes_in_20_days_l699_699943

-- Define the rates and the work
variable (A B W : ℝ)

-- First condition: A and B together can finish the work in 12 days
axiom together_rate : (A + B) * 12 = W

-- Second condition: B alone can finish the work in 30.000000000000007 days
axiom B_rate : B * 30.000000000000007 = W

-- Prove that A alone can finish the work in 20 days
theorem A_finishes_in_20_days : (1 / A) = 20 :=
by 
  sorry

end A_finishes_in_20_days_l699_699943


namespace number_of_transformations_returning_to_original_l699_699052

-- Definitions for the vertices of triangle T
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (4, 0)
def vertex3 : ℝ × ℝ := (0, 3)

-- Transformation functions
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Main statement
theorem number_of_transformations_returning_to_original :
  let transformations := [rotate90, rotate180, rotate270, reflect_x, reflect_y]
  in let sequences_with_3 := (transformations.product transformations).product transformations
  in let valid_sequences := sequences_with_3.filter (λ seq,
    let t1 := seq.1.1
    let t2 := seq.1.2
    let t3 := seq.2
    (t3 (t2 (t1 vertex1)) = vertex1) ∧ (t3 (t2 (t1 vertex2)) = vertex2) ∧ (t3 (t2 (t1 vertex3)) = vertex3))
  in valid_sequences.length = 12 := 
sorry

end number_of_transformations_returning_to_original_l699_699052


namespace integer_solution_count_l699_699635

theorem integer_solution_count :
  ∃ n : ℕ, n = 10 ∧
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 15 ∧ (0 ≤ x1 ∧ x1 ≤ 5) ∧ (0 ≤ x2 ∧ x2 ≤ 6) ∧ (0 ≤ x3 ∧ x3 ≤ 7) := 
sorry

end integer_solution_count_l699_699635


namespace min_distance_of_complex_abs_eq_one_l699_699646

noncomputable def min_distance (z : ℂ) (h : complex.abs z = 1) : ℝ :=
  complex.abs (z - (3 + 4 * complex.I))

theorem min_distance_of_complex_abs_eq_one (z : ℂ) (h : complex.abs z = 1) : min_distance z h = 4 :=
sorry

end min_distance_of_complex_abs_eq_one_l699_699646


namespace problem_statement_l699_699058

theorem problem_statement : ∃ n : ℤ, 0 < n ∧ (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).den = 1 ∧ ¬ n > 96 := 
by 
  sorry

end problem_statement_l699_699058


namespace exists_n_with_six_consecutive_zeros_l699_699428

theorem exists_n_with_six_consecutive_zeros :
  ∃ n : ℕ, n < 10^6 ∧ 5^n % 10^20 = 95367431640625 :=
begin
  use 524308,
  split,
  { -- Proof that n < 10^6
    norm_num,
  },
  { -- Proof that 5^524308 % 10^20 = 95367431640625
    sorry
  }
end

end exists_n_with_six_consecutive_zeros_l699_699428


namespace expected_value_8_sided_die_l699_699547

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8] in
  let divisible_by_3 := [3, 6] in
  let prob_div_3 := 1 / 4 in
  let prob_not_div_3 := 3 / 4 in
  let winnings_div_3 := (3 + 6) / 4 in
  let winnings_not_div_3 := 0 in
  let expected_value := winnings_div_3 + winnings_not_div_3 in
  expected_value = 2.25 :=
by
  sorry

end expected_value_8_sided_die_l699_699547


namespace sum_rational_coefficients_correct_l699_699630

noncomputable def findSumRationalTermCoefficients : Nat := do
  let binomial := ({2 * Real.sqrt x - 1 / x} ^ 6)
  let result := 64 + 240 + 60 + 1
  return result

theorem sum_rational_coefficients_correct : 
  findSumRationalTermCoefficients = 365 :=
by
  -- The proof is omitted
  sorry

end sum_rational_coefficients_correct_l699_699630


namespace sin_2alpha_tan_alpha_g_range_l699_699669

noncomputable def P : ℝ × ℝ := (-3, real.sqrt 3)

theorem sin_2alpha_tan_alpha (α : ℝ) (h1 : P.1 = -3) (h2 : P.2 = real.sqrt 3) :
  2 * real.sin α * real.cos α - real.tan α = -real.sqrt 3 / 6 :=
sorry

noncomputable def f (x α : ℝ) : ℝ :=
  real.cos (x - α) * real.cos α - real.sin (x - α) * real.sin α

noncomputable def g (x α : ℝ) : ℝ :=
  real.sqrt 3 * f (π / 2 - 2 * x) α - 2 * (f x α) ^ 2

theorem g_range (α : ℝ) (h1 : P.1 = -3) (h2 : P.2 = real.sqrt 3) :
  set.Icc 0 (2 * π / 3) ⊆ set_of (λ x, g x α >= -2 ∧ g x α ≤ 1) :=
sorry

end sin_2alpha_tan_alpha_g_range_l699_699669


namespace ratio_of_areas_inequality_l699_699949

theorem ratio_of_areas_inequality (a x m : ℝ) (h1 : a > 0) (h2 : x > 0) (h3 : x < a) :
  m = (3 * x^2 - 3 * a * x + a^2) / a^2 →
  (1 / 4 ≤ m ∧ m < 1) :=
sorry

end ratio_of_areas_inequality_l699_699949


namespace katy_brownies_total_l699_699395

theorem katy_brownies_total : 
  (let monday_brownies := 5 in
   let tuesday_brownies := 2 * monday_brownies in
   let total_brownies := monday_brownies + tuesday_brownies in
   total_brownies = 15) := 
by 
  let monday_brownies := 5 in
  let tuesday_brownies := 2 * monday_brownies in
  let total_brownies := monday_brownies + tuesday_brownies in
  show total_brownies = 15 by
  sorry

end katy_brownies_total_l699_699395


namespace katy_brownies_l699_699398

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l699_699398


namespace total_cost_price_correct_l699_699981

def cost_price_of_apple (sp_a : ℝ) (l_a : ℝ) : ℝ := sp_a / (1 - l_a)
def cost_price_of_orange (sp_o : ℝ) (l_o : ℝ) : ℝ := sp_o / (1 - l_o)
def cost_price_of_banana (sp_b : ℝ) (l_b : ℝ) : ℝ := sp_b / (1 - l_b)

def total_cost_price (sp_a sp_o sp_b l_a l_o l_b : ℝ) : ℝ :=
  cost_price_of_apple sp_a l_a + cost_price_of_orange sp_o l_o + cost_price_of_banana sp_b l_b

theorem total_cost_price_correct :
  total_cost_price 30 45 15 (1/5) (1/4) (1/6) = 115.5 :=
by
  sorry

end total_cost_price_correct_l699_699981


namespace find_cubic_polynomial_l699_699259

noncomputable def q (x : ℝ) : ℝ := 4 * x^3 - 19 * x^2 + 5 * x + 6

theorem find_cubic_polynomial : 
  ∃ (a b c d : ℝ), q 0 = 6 ∧ q 1 = -4 ∧ q 2 = 0 ∧ q 3 = 10 :=
by {
  let a := 4,
  let b := -19,
  let c := 5,
  let d := 6,
  have h0 : q 0 = 6 := by { unfold q, linarith },
  have h1 : q 1 = -4 := by { unfold q, linarith },
  have h2 : q 2 = 0 := by { unfold q, linarith },
  have h3 : q 3 = 10 := by { unfold q, linarith },
  exact ⟨a, b, c, d, h0, h1, h2, h3⟩,
}
sorry

end find_cubic_polynomial_l699_699259


namespace fifth_term_is_67_l699_699724

noncomputable def satisfies_sequence (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :=
  (a = 3) ∧ (d = 27) ∧ 
  (a = (1/3 : ℚ) * (3 + b)) ∧
  (b = (1/3 : ℚ) * (a + 27)) ∧
  (27 = (1/3 : ℚ) * (b + e))

theorem fifth_term_is_67 :
  ∃ (e : ℕ), satisfies_sequence 3 a b 27 e ∧ e = 67 :=
sorry

end fifth_term_is_67_l699_699724


namespace decreasing_intervals_l699_699876

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem decreasing_intervals : 
  (∀ x y : ℝ, x < y → ((y < -1 ∨ x > -1) → f y < f x)) ∧
  (∀ x y : ℝ, x < y → (y ≥ -1 ∧ x ≤ -1 → f y < f x)) :=
by 
  intros;
  sorry

end decreasing_intervals_l699_699876


namespace triangle_solution_l699_699019

variables (a b c : ℝ) (B : ℝ)

theorem triangle_solution (h1 : a = 1) (h2 : b = 2) (h3 : real.cos B = 1 / 4) :
  c = 2 ∧ (1 / 2) * a * c * real.sin B = sqrt 15 / 4 :=
by
  sorry -- The proof is omitted as per the instructions.

end triangle_solution_l699_699019


namespace sin_cos_condition_necessary_but_not_sufficient_l699_699952

noncomputable def condition_sin_cos (k x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ k * sin x * cos x < x

noncomputable def necessary_but_not_sufficient (k : ℝ) : Prop :=
  (∀ x, condition_sin_cos k x) → k < 1

theorem sin_cos_condition_necessary_but_not_sufficient (k : ℝ) :
  necessary_but_not_sufficient k :=
sorry

end sin_cos_condition_necessary_but_not_sufficient_l699_699952


namespace global_phone_company_customers_l699_699548

theorem global_phone_company_customers :
  (total_customers = 25000) →
  (us_percentage = 0.20) →
  (canada_percentage = 0.12) →
  (australia_percentage = 0.15) →
  (uk_percentage = 0.08) →
  (india_percentage = 0.05) →
  (us_customers = total_customers * us_percentage) →
  (canada_customers = total_customers * canada_percentage) →
  (australia_customers = total_customers * australia_percentage) →
  (uk_customers = total_customers * uk_percentage) →
  (india_customers = total_customers * india_percentage) →
  (mentioned_countries_customers = us_customers + canada_customers + australia_customers + uk_customers + india_customers) →
  (other_countries_customers = total_customers - mentioned_countries_customers) →
  (other_countries_customers = 10000) ∧ (us_customers / other_countries_customers = 1 / 2) :=
by
  -- The further proof steps would go here if needed
  sorry

end global_phone_company_customers_l699_699548


namespace count_5_primable_lt_1000_is_16_l699_699193

def is_n_primable (n pos_num : ℕ) : Prop :=
  pos_num % n = 0 ∧ (∀ d ∈ pos_num.digits 10, d ∈ [2, 3, 5, 7])

def is_5_primable := is_n_primable 5

def count_5_primable_less_than (limit : ℕ) : ℕ :=
  (List.range limit).count (λ x, is_5_primable x)

theorem count_5_primable_lt_1000_is_16 :
  count_5_primable_less_than 1000 = 16 :=
by
  sorry

end count_5_primable_lt_1000_is_16_l699_699193


namespace perimeter_difference_l699_699037

theorem perimeter_difference (x : ℝ) :
  let small_square_perimeter := 4 * x
  let large_square_perimeter := 4 * (x + 8)
  large_square_perimeter - small_square_perimeter = 32 :=
by
  sorry

end perimeter_difference_l699_699037


namespace slope_tangent_line_at_0_l699_699476

def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem slope_tangent_line_at_0 : (f' 0 = 2) :=
by
  -- definition of derivative
  sorry

end slope_tangent_line_at_0_l699_699476


namespace probability_of_valid_ticket_l699_699907

-- Define the number of tickets
def total_tickets := 100

-- Define the condition: A number must be a multiple of 3 and divisible by a prime number
def is_valid_ticket (n : ℕ) : Prop :=
  (n % 3 = 0) ∧ (∃ p, Nat.Prime p ∧ p ∣ n)

-- Define the number of valid tickets
noncomputable def count_valid_tickets : ℕ :=
  (Finset.range 101).count is_valid_ticket

-- Calculate probability
noncomputable def probability : ℚ :=
  count_valid_tickets / total_tickets

-- Theorem statement
theorem probability_of_valid_ticket :
  probability = 33 / 100 :=
sorry

end probability_of_valid_ticket_l699_699907


namespace four_digit_number_exists_l699_699013

noncomputable def findN : Nat :=
  let K := 45 -- This is derived from the modular condition checks
  K * K

theorem four_digit_number_exists (X a b c d K N : Nat) 
  (hX : X = 1000 * a + 100 * b + 10 * c + d)
  (ha_pos : a ≠ 0)
  (hN : N = X - (a + b + c + d))
  (hK : N = K * K)
  (hK_mod_20 : K % 20 = 5)
  (hK_mod_21 : K % 21 = 3) :
  N = 2025 :=
by
  have hK_val : K = 45 := by
    sorry
  have hN_val : N = K * K := by
    rw [hK_val]
    exact by rfl
  rw [hK_val, hN_val]
  rfl

end four_digit_number_exists_l699_699013


namespace day_of_50th_in_year_N_minus_1_l699_699386

theorem day_of_50th_in_year_N_minus_1
  (N : ℕ)
  (day250_in_year_N_is_sunday : (250 % 7 = 0))
  (day150_in_year_N_plus_1_is_sunday : (150 % 7 = 0))
  : 
  (50 % 7 = 1) := 
sorry

end day_of_50th_in_year_N_minus_1_l699_699386


namespace paper_clips_distribution_l699_699831

theorem paper_clips_distribution (total_clips : ℕ) (num_boxes : ℕ) (clip_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : num_boxes = 9) : clip_per_box = 9 :=
by sorry

end paper_clips_distribution_l699_699831


namespace least_positive_integer_x_20y_l699_699016

theorem least_positive_integer_x_20y (x y : ℤ) (h : Int.gcd x (20 * y) = 4) : 
  ∃ k : ℕ, k > 0 ∧ k * (x + 20 * y) = 4 := 
sorry

end least_positive_integer_x_20y_l699_699016


namespace circle_tangent_parabola_height_difference_l699_699966

theorem circle_tangent_parabola_height_difference :
  ∀ (a b : ℝ), 
  let y := λ x : ℝ, x^2 + x in
  let tangency_points := (a, y a) ∧ (-a, y (-a)) in
  let circle_center := (0, b) in
  -- Given the parabola equation and tangency condition
  (∀ x : ℝ, (x^2 + (y x - b)^2 = (y a - b)^2) ∧ y x = x^2 + x) →
  -- The difference in height between the center of the circle and the tangency points
  b - a^2 - a = 1 :=
sorry

end circle_tangent_parabola_height_difference_l699_699966


namespace initial_violet_balloons_l699_699039

-- Let's define the given conditions
def red_balloons : ℕ := 4
def violet_balloons_lost : ℕ := 3
def violet_balloons_now : ℕ := 4

-- Define the statement to prove
theorem initial_violet_balloons :
  (violet_balloons_now + violet_balloons_lost) = 7 :=
by
  sorry

end initial_violet_balloons_l699_699039


namespace chris_average_price_l699_699597

noncomputable def total_cost_dvd (price_per_dvd : ℝ) (num_dvds : ℕ) (discount : ℝ) : ℝ :=
  (price_per_dvd * (1 - discount)) * num_dvds

noncomputable def total_cost_bluray (price_per_bluray : ℝ) (num_blurays : ℕ) : ℝ :=
  price_per_bluray * num_blurays

noncomputable def total_cost_ultra_hd (price_per_ultra_hd : ℝ) (num_ultra_hds : ℕ) : ℝ :=
  price_per_ultra_hd * num_ultra_hds

noncomputable def total_cost (cost_dvd cost_bluray cost_ultra_hd : ℝ) : ℝ :=
  cost_dvd + cost_bluray + cost_ultra_hd

noncomputable def total_with_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

noncomputable def average_price (total_with_tax : ℝ) (total_movies : ℕ) : ℝ :=
  total_with_tax / total_movies

theorem chris_average_price :
  let price_per_dvd := 15
  let num_dvds := 5
  let discount := 0.20
  let price_per_bluray := 20
  let num_blurays := 8
  let price_per_ultra_hd := 25
  let num_ultra_hds := 3
  let tax_rate := 0.10
  let total_movies := num_dvds + num_blurays + num_ultra_hds
  let cost_dvd := total_cost_dvd price_per_dvd num_dvds discount
  let cost_bluray := total_cost_bluray price_per_bluray num_blurays
  let cost_ultra_hd := total_cost_ultra_hd price_per_ultra_hd num_ultra_hds
  let pre_tax_total := total_cost cost_dvd cost_bluray cost_ultra_hd
  let total := total_with_tax pre_tax_total tax_rate
  average_price total total_movies = 20.28 :=
by
  -- substitute each definition one step at a time
  -- to show the average price exactly matches 20.28
  sorry

end chris_average_price_l699_699597


namespace folded_paper_has_16_on_top_l699_699941

theorem folded_paper_has_16_on_top :
  let paper := [
    [1, 2], [3, 4], [5, 6], [7, 8],
    [9, 10], [11, 12], [13, 14], [15, 16]
  ],
  step1 := [[15, 16], [13, 14], [11, 12], [9, 10]],
  step2 := [[9, 10], [11, 12], [13, 14], [15, 16]],
  step3 := [[15, 16], [13, 14]],
  step4 := [16, 14],
  step4.head = 16 :=
by
  -- skip the proof
  sorry

end folded_paper_has_16_on_top_l699_699941


namespace angle_KDL_is_90_l699_699750

variables {A B C D E F P Q S K L : Type*}
variables [CyclicHexagon ABCDEF] [Perpendicular AB BD] [EqualLength BC EF]
variables [LineIntersection BC AD P] [LineIntersection EF AD Q]
variables [SameSide P Q D] [OppositeSide A D] [Midpoint S AD]
variables [Incenter K (Triangle BPS)] [Incenter L (Triangle EQS)]

theorem angle_KDL_is_90° :
  Angle KDL = 90 :=
sorry

end angle_KDL_is_90_l699_699750


namespace last_digit_of_1_div_2_pow_15_l699_699928

theorem last_digit_of_1_div_2_pow_15 :
  let last_digit_of := (n : ℕ) → n % 10
  last_digit_of (5^15) = 5 → 
  (∀ (n : ℕ),  ∃ (k : ℕ), n = 2^k →  last_digit_of (5 ^ k) = last_digit_of (1 / 2 ^ 15)) := 
by 
  intro last_digit_of h proof
  exact sorry

end last_digit_of_1_div_2_pow_15_l699_699928


namespace age_of_James_when_Thomas_reaches_current_age_l699_699486
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l699_699486


namespace iron_weighs_more_l699_699426

-- Define the weights of the metal pieces
def weight_iron : ℝ := 11.17
def weight_aluminum : ℝ := 0.83

-- State the theorem to prove that the difference in weights is 10.34 pounds
theorem iron_weighs_more : weight_iron - weight_aluminum = 10.34 :=
by sorry

end iron_weighs_more_l699_699426


namespace sally_turnip_count_l699_699088

theorem sally_turnip_count (total_turnips : ℕ) (mary_turnips : ℕ) (sally_turnips : ℕ) 
  (h1: total_turnips = 242) 
  (h2: mary_turnips = 129) 
  (h3: total_turnips = mary_turnips + sally_turnips) : 
  sally_turnips = 113 := 
by 
  sorry

end sally_turnip_count_l699_699088


namespace right_triangle_sides_l699_699475

theorem right_triangle_sides (a b c : ℝ) (h_ratio : ∃ x : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x) 
(h_area : 1 / 2 * a * b = 24) : a = 6 ∧ b = 8 ∧ c = 10 :=
by
  sorry

end right_triangle_sides_l699_699475


namespace solve_diamond_l699_699342

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end solve_diamond_l699_699342


namespace orthogonal_dot_product_l699_699407

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem orthogonal_dot_product (x : ℝ) : let a := vec 1 2 in
  let b := vec (-1) x in
  dot_product a b = 0 → x = 1 / 2 :=
by
  intros a b h
  sorry

#eval orthogonal_dot_product (1 / 2) 

end orthogonal_dot_product_l699_699407


namespace coordinates_of_P_l699_699081

noncomputable theory

def point_moves_and_coordinates (a : ℤ) : Prop :=
  let P := (a + 1, a)
  let P1 := (a + 4, a)
  a + 4 = 0 →
  P = (-3, -4)

theorem coordinates_of_P :
  ∃ a : ℤ, point_moves_and_coordinates a :=
begin
  use -4,
  unfold point_moves_and_coordinates,
  simp,
  exact rfl,
end

end coordinates_of_P_l699_699081


namespace range_of_a_l699_699783

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699783


namespace unique_perpendicular_points_l699_699064

-- Definitions for lines and points
structure Line := (equation : ℝ → ℝ → Prop)
structure Point := (x : ℝ) (y : ℝ)

def is_perpendicular_to (l1 l2 : Line) : Prop :=
  ∃ X : Point, (l1.equation X.x X.y) ∧ (l2.equation X.x X.y)
  
-- Assumptions
variables {n : ℕ} (l : ℕ → Line)
  (h_lines_non_parallel : ∃ i j, i ≠ j ∧ ¬ is_perpendicular_to (l i) (l j))

-- The problem to prove
theorem unique_perpendicular_points :
  ∃ X : fin n → Point,
  (∀ k : fin n, 
    (is_perpendicular_to (l k) (l (k + 1))) ∧ 
    (is_perpendicular_to (l (k + 1)) (l k)))
  ∧ 
  (is_perpendicular_to (l 0) (l (fin.last n))) :=
sorry

end unique_perpendicular_points_l699_699064


namespace minimum_value_of_f_l699_699469

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3 * x + 3) + Real.sqrt (x^2 - 3 * x + 3)

theorem minimum_value_of_f : (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ f 0 = 2 * Real.sqrt 3 :=
by
  sorry

end minimum_value_of_f_l699_699469


namespace number_of_triangles_l699_699113

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l699_699113


namespace flat_path_time_l699_699418

/-- Malcolm's walking time problem -/
theorem flat_path_time (x : ℕ) (h1 : 6 + 12 + 6 = 24)
                       (h2 : 3 * x = 24 + 18) : x = 14 := 
by
  sorry

end flat_path_time_l699_699418


namespace coefficient_x2_term_expansion_l699_699047

noncomputable def n : ℝ := ∫ x in 0..(Real.pi / 2), 6 * Real.sin x

theorem coefficient_x2_term_expansion : 
  (∫ x in 0..(Real.pi / 2), 6 * Real.sin x = 6) → 
  ∀ n = 6, ∀ x : ℝ, (x - (2/x))^n.expand.coeff 2 = 60 :=
by
  intros h1 h2
  have h : n = 6, from h2
  rw h
  sorry

end coefficient_x2_term_expansion_l699_699047


namespace katy_brownies_l699_699393

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l699_699393


namespace range_of_f_area_of_triangle_l699_699322

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

-- Problem Part (I)
theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -1/2 ≤ f x ∧ f x ≤ 1/4) :=
sorry

-- Problem Part (II)
theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ) 
  (hA0 : 0 < A ∧ A < Real.pi)
  (hS1 : a = Real.sqrt 3)
  (hS2 : b = 2 * c)
  (hF : f A = 1/4) :
  (∃ (area : ℝ), area = (1/2) * b * c * Real.sin A ∧ area = Real.sqrt 3 / 3)
:=
sorry

end range_of_f_area_of_triangle_l699_699322


namespace max_plus_min_value_of_f_l699_699713

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := ⨆ x, f x
  let m := ⨅ x, f x
  M + m = 4 :=
by 
  sorry

end max_plus_min_value_of_f_l699_699713


namespace algebraic_expression_value_l699_699693

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -2) : 
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 7 :=
by
  sorry

end algebraic_expression_value_l699_699693


namespace monotonically_increasing_range_l699_699773

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699773


namespace smallest_N_for_circular_table_l699_699969

/--
  Given a circular table with 60 chairs, prove that the smallest number of people, N,
  such that any additional person must sit next to someone already seated is 20.
-/
theorem smallest_N_for_circular_table (N : ℕ) (h : N = 20) : 
  ∀ (next_seated : ℕ), next_seated ≤ N → (∃ i : ℕ, i < N ∧ next_seated = i + 1 ∨ next_seated = i - 1) :=
by
  sorry

end smallest_N_for_circular_table_l699_699969


namespace number_of_triangles_l699_699108

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l699_699108


namespace simplify_sum_of_cosines_l699_699443

noncomputable def omega := Complex.exp (Complex.I * 2 * Real.pi / 17)

theorem simplify_sum_of_cosines :
  (Complex.cos (2 * Real.pi / 17) + Complex.cos (6 * Real.pi / 17) + Complex.cos (10 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  have omega_17_eq_one : omega ^ 17 = 1 :=
    by
    rw [omega, Complex.exp_eq_exp, Complex.exp_mul_I, Complex.exp_pi_mul_I, Complex.exp_zero]
    sorry
  have _ := Complex.cos_exp_eq_cos_mul_real
  sorry

end simplify_sum_of_cosines_l699_699443


namespace total_earnings_correct_l699_699090

-- Define the earnings of each individual
def SalvadorEarnings := 1956
def SantoEarnings := SalvadorEarnings / 2
def MariaEarnings := 3 * SantoEarnings
def PedroEarnings := SantoEarnings + MariaEarnings

-- Define the total earnings calculation
def TotalEarnings := SalvadorEarnings + SantoEarnings + MariaEarnings + PedroEarnings

-- State the theorem to prove
theorem total_earnings_correct :
  TotalEarnings = 9780 :=
sorry

end total_earnings_correct_l699_699090


namespace trig_identity_l699_699942

theorem trig_identity (α : ℝ) :
  (cos α)⁻⁶ - (tan α)⁶ = 3 * (tan α)² * (cos α)⁻² + 1 :=
  sorry

end trig_identity_l699_699942


namespace equal_inradii_implication_l699_699751

theorem equal_inradii_implication 
  (A B C D E F : Type) 
  [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F]
  (AB AC BC : Line)
  (radius_AEF radius_BFD radius_CDE radius_DEF radius_ABC : ℝ)
  (h1 : incidence A B AB) (h2 : incidence A C AC) (h3 : incidence B C BC)
  (h4 : incidence D B BC) (h5 : incidence E C AC) (h6 : incidence F A AB)
  (h7 : inradius (triangle A E F) = radius_AEF) 
  (h8 : inradius (triangle B F D) = radius_BFD)
  (h9 : inradius (triangle C D E) = radius_CDE)
  (h10 : inradius (triangle D E F) = radius_DEF)
  (h11 : inradius (triangle A B C) = radius_ABC)
  (h_radii_aequal : radius_AEF = radius_BFD ∧ radius_BFD = radius_CDE)
  : radius_AEF + radius_DEF = radius_ABC :=
begin
  sorry
end

end equal_inradii_implication_l699_699751


namespace track_circumference_is_720_l699_699955

variable (P Q : Type) -- Define the types of P and Q, e.g., as points or runners.

noncomputable def circumference_of_the_track (C : ℝ) : Prop :=
  ∃ y : ℝ, 
  (∃ first_meeting_condition : Prop, first_meeting_condition = (150 = y - 150) ∧
  ∃ second_meeting_condition : Prop, second_meeting_condition = (2*y - 90 = y + 90) ∧
  C = 2 * y)

theorem track_circumference_is_720 :
  circumference_of_the_track 720 :=
by
  sorry

end track_circumference_is_720_l699_699955


namespace part1_intersection_part1_union_complement_part2_subset_l699_699690

open Set

variable {x a : ℝ}

def A : Set ℝ := {x | 3 ≤ 3^x ∧ 3^x ≤ 27}
def B : Set ℝ := {x | log 2 x > 1}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem part1_intersection :
  A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem part1_union_complement :
  compl B ∪ A = {x | x ≤ 3} :=
sorry

theorem part2_subset (a : ℝ) :
  C a ⊆ A → 1 < a ∧ a ≤ 3 :=
sorry

end part1_intersection_part1_union_complement_part2_subset_l699_699690


namespace find_u_over_p_l699_699005

constants (p r t u : ℚ)

axiom h1 : p / r = 8
axiom h2 : t / r = 5
axiom h3 : t / u = 2 / 3

theorem find_u_over_p : u / p = 15 / 16 :=
by
  sorry

end find_u_over_p_l699_699005


namespace solve_diamond_l699_699340

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l699_699340


namespace second_week_data_l699_699844

-- Define the conditions and known values
def planData : ℕ := 8 -- GB
def extraCostPerGB : ℕ := 10 -- dollars
def firstWeekData : ℕ := 2 -- GB
def thirdWeekData : ℕ := 5 -- GB
def fourthWeekData : ℕ := 10 -- GB
def extraCostPaid : ℕ := 120 -- dollars

-- Define the correct answer for the second week usage
def secondWeekDataCorrect : ℕ := 3 -- GB

-- Prove that the data usage for the second week is 3 GB
theorem second_week_data {x : ℕ} : 
  let extraGB := extraCostPaid / extraCostPerGB in
  let totalData := planData + extraGB in
  firstWeekData + x + thirdWeekData + fourthWeekData = totalData → 
  x = secondWeekDataCorrect :=
by
  sorry

end second_week_data_l699_699844


namespace volume_of_solid_of_revolution_l699_699232

-- Definitions of the curves
def curve1 (x : ℝ) := (x^2) / 2
def curve2 (x : ℝ) := (3 - 2*x) / 2

-- The bounds for x where the curves intersect
def x_lower : ℝ := -3
def x_upper : ℝ := 1

-- Integral for the volume of the first solid
def solid1_volume : ℝ := (π / 4) * (9*4 - (12*4*(-1)) + (4/3)*4^3 - 
                                 (9*(-3) - 6*(-3)^2 + (4*(-3)^3)/3))

-- Integral for the volume of the second solid
def solid2_volume : ℝ := (π / 4) * ((x_upper^5 / 5) - (x_lower^5 / 5))

-- The final volume is the difference between the two volumes calculated
def final_volume : ℝ := solid1_volume - solid2_volume

theorem volume_of_solid_of_revolution :
  final_volume = (272/15) * π :=
begin
  sorry
end

end volume_of_solid_of_revolution_l699_699232


namespace area_of_intersecting_lines_triangle_l699_699083

theorem area_of_intersecting_lines_triangle (ABC : Triangle) (area_ABC : ℝ)
    (A1 B1 C1 : Point) (A B C : Point)
    (h1 : A1 ∈ Segment B C ∧ ratio B A1 A1 C = 1 / 3)
    (h2 : B1 ∈ Segment A C ∧ ratio C B1 B1 A = 1 / 3)
    (h3 : C1 ∈ Segment A B ∧ ratio A C1 C1 B = 1 / 3)
    (area_ABC : area ABC = 1) :
    ∃ MNK : Triangle, formed_by_intersections AA1 BB1 CC1,
    area MNK = 4 / 13 := sorry

end area_of_intersecting_lines_triangle_l699_699083


namespace part1_part2_l699_699815

noncomputable section

open Real

def f (x a : ℝ) : ℝ := abs (2*x - 1) + abs (x + a)

theorem part1 (x : ℝ) : f x 1 ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ abs (a - 1)) ↔ a ∈ Iic (1/4) :=
by
  sorry

end part1_part2_l699_699815


namespace average_age_l699_699723

theorem average_age (women men : ℕ) (avg_age_women avg_age_men : ℝ) 
  (h_women : women = 12) 
  (h_men : men = 18) 
  (h_avg_women : avg_age_women = 28) 
  (h_avg_men : avg_age_men = 40) : 
  (12 * 28 + 18 * 40) / (12 + 18) = 35.2 :=
by {
  sorry
}

end average_age_l699_699723


namespace number_of_bad_cards_l699_699745

-- Define the initial conditions
def janessa_initial_cards : ℕ := 4
def father_given_cards : ℕ := 13
def ordered_cards : ℕ := 36
def cards_given_to_dexter : ℕ := 29
def cards_kept_for_herself : ℕ := 20

-- Define the total cards and cards in bad shape calculation
theorem number_of_bad_cards : 
  let total_initial_cards := janessa_initial_cards + father_given_cards;
  let total_cards := total_initial_cards + ordered_cards;
  let total_distributed_cards := cards_given_to_dexter + cards_kept_for_herself;
  total_cards - total_distributed_cards = 4 :=
by {
  sorry
}

end number_of_bad_cards_l699_699745


namespace four_people_possible_l699_699206

structure Person :=
(first_name : String)
(patronymic : String)
(surname : String)

def noThreePeopleShareSameAttribute (people : List Person) : Prop :=
  ∀ (attr : Person → String), ¬ ∃ (a b c : Person),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ (attr a = attr b) ∧ (attr b = attr c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def anyTwoPeopleShareAnAttribute (people : List Person) : Prop :=
  ∀ (a b : Person), a ∈ people ∧ b ∈ people ∧ a ≠ b →
    (a.first_name = b.first_name ∨ a.patronymic = b.patronymic ∨ a.surname = b.surname)

def validGroup (people : List Person) : Prop :=
  noThreePeopleShareSameAttribute people ∧ anyTwoPeopleShareAnAttribute people

theorem four_people_possible : ∃ (people : List Person), people.length = 4 ∧ validGroup people :=
sorry

end four_people_possible_l699_699206


namespace trains_combined_distance_l699_699913

-- Define speeds in kmph
def train_a_speed_kmph : ℝ := 120
def train_b_speed_kmph : ℝ := 160

-- Define the conversion factor from kmph to kmpm
def kmph_to_kmpm : ℝ := 1 / 60

-- Define speeds in kmpm
def train_a_speed_kmpm : ℝ := train_a_speed_kmph * kmph_to_kmpm
def train_b_speed_kmpm : ℝ := train_b_speed_kmph * kmph_to_kmpm

-- Define time in minutes
def time_minutes : ℝ := 45

-- Define distances covered by both trains
def distance_covered_by_train_a : ℝ := train_a_speed_kmpm * time_minutes
def distance_covered_by_train_b : ℝ := train_b_speed_kmpm * time_minutes

-- Define combined distance
def combined_distance : ℝ := distance_covered_by_train_a + distance_covered_by_train_b

-- The statement we want to prove
theorem trains_combined_distance :
  combined_distance = 210 := by
  sorry

end trains_combined_distance_l699_699913


namespace February_March_Ratio_l699_699584

theorem February_March_Ratio (J F M : ℕ) (h1 : F = 2 * J) (h2 : M = 8800) (h3 : J + F + M = 12100) : F / M = 1 / 4 :=
by
  sorry

end February_March_Ratio_l699_699584


namespace num_valid_sequences_is_12_l699_699054

-- Definitions of the vertices of the triangle
def T : List (ℝ × ℝ) := [(0, 0), (4, 0), (0, 3)]

-- Transformation Definitions
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- List of transformations
def transformations : List ((ℝ × ℝ) → (ℝ × ℝ)) := 
    [rotate90, rotate180, rotate270, reflectX, reflectY]

-- Function to apply a list of transformations to a point
def apply_transformations (p : ℝ × ℝ) (trans : List ((ℝ × ℝ) → (ℝ × ℝ))) : ℝ × ℝ :=
    trans.foldl (fun acc t => t acc) p

-- Function to check if a sequence of transformations returns T to its original position
def returns_to_original (trans : List ((ℝ × ℝ) → (ℝ × ℝ))) : Bool :=
    T.all (fun p => apply_transformations p trans ∈ T)

-- Number of valid sequences that return T to its original position
def count_valid_sequences : ℕ :=
    (List.permutationsN 3 transformations).count returns_to_original

theorem num_valid_sequences_is_12 : count_valid_sequences = 12 := 
by sorry

end num_valid_sequences_is_12_l699_699054


namespace current_age_of_oldest_person_babysat_l699_699522

theorem current_age_of_oldest_person_babysat (jane_age_stopped_babysitting : ℕ) (years_since_stopped : ℕ) 
  (half_age_when_stopped : jane_age_stopped_babysitting / 2 = 10)  : 
  let age_of_oldest_child_then := jane_age_stopped_babysitting / 2 in
  let current_age_of_oldest_child := age_of_oldest_child_then + years_since_stopped in
  jane_age_stopped_babysitting = 20 ∧ years_since_stopped = 12 → current_age_of_oldest_child = 22 :=
by
  intros h
  sorry

end current_age_of_oldest_person_babysat_l699_699522


namespace remainder_sum_l699_699937

theorem remainder_sum (a b c d : ℕ) 
  (h_a : a % 30 = 15) 
  (h_b : b % 30 = 7) 
  (h_c : c % 30 = 22) 
  (h_d : d % 30 = 6) : 
  (a + b + c + d) % 30 = 20 := 
by
  sorry

end remainder_sum_l699_699937


namespace semicircle_area_percent_increase_125_l699_699559

noncomputable def percent_increase_semicircle_area (length width : ℝ) : ℝ :=
  let area_large_semicircles := 2 * (π * (length / 2)^2 / 2)
  let area_small_semicircles := 2 * (π * (width / 2)^2 / 2)
  100 * (area_large_semicircles / area_small_semicircles - 1)

theorem semicircle_area_percent_increase_125 (length width : ℝ) (h1 : length = 12) (h2 : width = 8) :
  percent_increase_semicircle_area length width = 125 :=
by
  rw [h1, h2]
  simp [percent_increase_semicircle_area]
  sorry

end semicircle_area_percent_increase_125_l699_699559


namespace prime_geq_7_div_240_l699_699850

theorem prime_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 240 ∣ p^4 - 1 :=
sorry

end prime_geq_7_div_240_l699_699850


namespace rainbow_preschool_full_day_students_l699_699994

theorem rainbow_preschool_full_day_students (total_students : ℕ) (half_day_percent : ℝ)
  (h1 : total_students = 80) (h2 : half_day_percent = 0.25) :
  (total_students * (1 - half_day_percent)).to_nat = 60 :=
by
  -- Transform percentage to a fraction
  let fraction_full_day := 1 - half_day_percent
  -- Calculate full-day students
  have h_full_day_students : ℝ := total_students * fraction_full_day
  -- Convert to natural number
  exact (floor h_full_day_students).to_nat = 60

end rainbow_preschool_full_day_students_l699_699994


namespace find_ABC_l699_699880

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := 
  x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  (∀ x : ℝ, x > 5 → g x 2 (-2) (-24) > 0.5) ∧
  (A = 2) ∧
  (B = -2) ∧
  (C = -24) ∧
  (∀ x, A * x^2 + B * x + C = A * (x + 3) * (x - 4)) → 
  A + B + C = -24 := 
by
  sorry

end find_ABC_l699_699880


namespace number_of_triangles_l699_699112

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l699_699112


namespace count_unique_solutions_l699_699470

theorem count_unique_solutions :
  ∀ (a b c : ℕ),
  0 < a → 0 < b → 0 < c → ab + bc = 44 ∧ ac + bc = 23 →
  let S := { (a, b, c) | 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 44 ∧ ac + bc = 23 } in
  S.card = 2 :=
by
  intro a b c ha hb hc h
  sorry

end count_unique_solutions_l699_699470


namespace ratio_of_areas_l699_699471

theorem ratio_of_areas (s r : ℝ) (h : 3 * s = 2 * π * r) : 
  ( (√ 3) * π / 9)  =
  ( (√ 3 * π^2 * r^2 / 9) / (π * r^2)) :=
by
  sorry

end ratio_of_areas_l699_699471


namespace complex_conjugate_square_l699_699643

theorem complex_conjugate_square (b : ℝ) (i : ℂ) (h_imag : i = Complex.I)
  (h_conj : 2 - i = Complex.conj (2 + b * i)) : (2 - b * i) ^ 2 = 3 - 4 * i :=
by
  -- Proof would go here.
  sorry

end complex_conjugate_square_l699_699643


namespace count_valid_rationals_less_than_pi_l699_699698

open Real Rat

-- Define the set of rational numbers with denominator at most 7 in lowest terms
def validRationals : Finset ℚ := 
  Finset.filter (λ q, q < π ∧ q.denom ≤ 7) 
    (Finset.range (7 * 7 + 1)).map (λ n, (1 : ℚ) * n / 1)

-- The statement to prove
theorem count_valid_rationals_less_than_pi :
  (Finset.count (λ q, q < π) (validRationals)) = 54 :=
sorry

end count_valid_rationals_less_than_pi_l699_699698


namespace number_of_triangles_l699_699105

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l699_699105


namespace solid_id_views_not_cylinder_l699_699982

theorem solid_id_views_not_cylinder :
  ∀ (solid : Type),
  (∃ (shape1 shape2 shape3 : solid),
    shape1 = shape2 ∧ shape2 = shape3) →
  solid ≠ cylinder :=
by 
  sorry

end solid_id_views_not_cylinder_l699_699982


namespace probability_larger_number_is_3_l699_699315

-- Defining the conditions
def cards : List ℕ := [1, 2, 3, 4, 5]

def combinations (lst : List ℕ) : List (ℕ × ℕ) :=
  List.bind lst (fun x => List.map (fun y => (x, y)) lst)

-- Statement of the problem to prove
theorem probability_larger_number_is_3 : 
  (combinations cards).count (λ (x : ℕ × ℕ), x.1 < x.2 ∧ x.2 = 3) / (combinations cards).length = 1 / 5 :=
by
  sorry

end probability_larger_number_is_3_l699_699315


namespace austons_total_height_in_cm_l699_699211

noncomputable def austons_height_in := 65
noncomputable def box_height_in := 12
noncomputable def inch_to_cm := 2.54
noncomputable def total_height_cm := (austons_height_in + box_height_in) * inch_to_cm

theorem austons_total_height_in_cm :
  Float.round (total_height_cm * 10) / 10 = 195.6 := 
by
  sorry

end austons_total_height_in_cm_l699_699211


namespace range_of_a_l699_699319

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 * a - 1) * x + 4 * a else log a x

theorem range_of_a (a : ℝ) : 
(∀ (x1 x2 : ℝ), x1 < x2 → f a x1 - f a x2 > 0) ↔ (a ∈ set.Ico (1 / 7) (1 / 3)) :=
sorry

end range_of_a_l699_699319


namespace inequality_proof_l699_699412

-- Definitions
variables {n : ℕ} {s : ℕ} {n_i : ℕ → ℕ} (h_distinct : (function.injective n_i))
noncomputable def M (n_i : ℕ → ℕ) (s : ℕ) := ∑ i in finset.range s, 2 ^ (n_i i)

-- Goal
theorem inequality_proof (h_distinct : function.injective n_i) :
  (∑ i in finset.range s, 2 ^ (n_i i / 2)) < (1 + real.sqrt 2) * real.sqrt (M n_i s) :=
sorry

end inequality_proof_l699_699412


namespace problem_x0_value_problem_m_value_l699_699247

theorem problem_x0_value : 
  ( ∃ x, |x + 3| - 2 * x - 1 < 0 ) → ∃ x0 : ℝ, x0 = 2 :=
by
  intro h
  use 2
  sorry

theorem problem_m_value (x0 : ℝ) (hx0 : x0 = 2) :
  ( ∃ x : ℝ, ∃ m : ℝ, m > 0 ∧ |x - m| + |x + 1/m| - x0 = 0) → ∃ m : ℝ, m = 1 :=
by
  intro h
  use 1
  sorry

end problem_x0_value_problem_m_value_l699_699247


namespace tom_climbing_time_in_hours_l699_699492

variable (t_Elizabeth t_Tom_minutes t_Tom_hours : ℕ)

-- Conditions:
def elizabeth_time : ℕ := 30
def tom_time_relation (t_Elizabeth : ℕ) : ℕ := 4 * t_Elizabeth
def tom_time_hours (t_Tom_minutes : ℕ) : ℕ := t_Tom_minutes / 60

-- Theorem statement:
theorem tom_climbing_time_in_hours :
  tom_time_hours (tom_time_relation elizabeth_time) = 2 :=
by 
  -- Reiterate the conditions with simplified relations
  have t_Elizabeth := elizabeth_time
  have t_Tom_minutes := tom_time_relation t_Elizabeth
  have t_Tom_hours := tom_time_hours t_Tom_minutes
  show t_Tom_hours = 2
  -- Placeholder for actual proof
  sorry

end tom_climbing_time_in_hours_l699_699492


namespace f_is_polynomial_l699_699951

open Nat

noncomputable def f (n : ℕ) : ℕ := ∑ k in range (n^2 + 1), (⌊ 2 * sqrt (k : ℝ) ⌋ : ℕ)

theorem f_is_polynomial (n : ℕ) (h : 0 < n) : ∃ (p : polynomial ℕ), ∀ k : ℕ, k = n → (f k) = p.eval k :=
sorry

end f_is_polynomial_l699_699951


namespace range_of_a_l699_699764

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699764


namespace monotonic_increasing_range_l699_699797

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699797


namespace chord_intersections_probability_l699_699632

theorem chord_intersections_probability :
  let points := 2020
  let num_choices := 5
  let A B C D E : ℕ
  in A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E
  -> let total_possible_choices := combin points num_choices
  -> let chord_intersections :=
      ∑ (p : Finset (Finset (Range points))), 
          (p.card = 5 ∧ 
           (p.contains A ∧ p.contains B ∧ p.contains C ∧ p.contains D ∧ p.contains E) ∧ 
           (intersects (make_chord A B) (make_chord C D) ∧ 
           ¬intersects (make_chord A E) (make_chord C D) ∧ 
           ¬intersects (make_chord A E) (make_chord A B)))
  -> chord_intersections.to_float / total_possible_choices.to_float = (1 / 3) :=
begin
  intros points num_choices A B C D E h_diff total_possible_choices chord_intersections,
  sorry
end

end chord_intersections_probability_l699_699632


namespace sum_of_sequence_l699_699893

theorem sum_of_sequence (a : ℕ → ℤ) (h1 : a 0 = 1) (h2 : a 1 = -1) 
  (h_property : ∀ n, 1 ≤ n ∧ n ≤ 2007 → a n = a (n - 1) + a (n + 1)) :
  (finset.range 2009).sum a = -2 :=
sorry

end sum_of_sequence_l699_699893


namespace equivalent_proof_problem_l699_699345

noncomputable def problem_statement (a : ℝ) : Prop :=
  let z := complex.mk (a - real.sqrt 2) a in -- z = (a - sqrt(2)) + ai
  z.im = complex.im z ∧ z.re = 0 → (a + complex.mk 0 1 ^ 7) / (1 + complex.mk 0 1 * a) = -complex.mk 0 1

-- statement asserting this is true for some a in ℝ
theorem equivalent_proof_problem : ∀ a : ℝ, problem_statement a :=
sorry

end equivalent_proof_problem_l699_699345


namespace expression_evaluation_l699_699219

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l699_699219


namespace problem1_correct_problem2_correct_l699_699233

-- Definition of conditions for problem (1)
def problem1_expr := (∛27) - ((-1)^2) + (√4)

-- Definition of conditions for problem (2)
def problem2_expr := - (2^2) - (∛8) - |1 - (√2)| + (-6 / 2)

-- Statement of the proofs
theorem problem1_correct : problem1_expr = 4 := 
by 
  sorry

theorem problem2_correct : problem2_expr = - (8 + √2) := 
by 
  sorry

end problem1_correct_problem2_correct_l699_699233


namespace cos_plus_sin_l699_699304

theorem cos_plus_sin (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : sin α * cos α = 1 / 8) : cos α + sin α = sqrt 5 / 2 := 
by
  sorry

end cos_plus_sin_l699_699304


namespace baseball_team_earnings_l699_699961

theorem baseball_team_earnings (S : ℝ) (W : ℝ) (Total : ℝ) 
    (h1 : S = 2662.50) 
    (h2 : W = S - 142.50) 
    (h3 : Total = W + S) : 
  Total = 5182.50 :=
sorry

end baseball_team_earnings_l699_699961


namespace range_of_a_l699_699806

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699806


namespace total_pictures_l699_699433

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end total_pictures_l699_699433


namespace solution_exists_l699_699097

theorem solution_exists :
  ∃ x y : ℝ, x = 19/4 ∧ y = 17/8 ∧ (x + real.sqrt (x + 2 * y) - 2 * y = 7 / 2) ∧ (x^2 + x + 2 * y - 4 * y^2 = 27 / 2) :=
by
  use 19 / 4, 17 / 8
  split ; norm_num [real.sqrt, pow_two],
  split ; norm_num [real.sqrt, pow_two],
  sorry -- Proof steps are omitted.

end solution_exists_l699_699097


namespace true_propositions_l699_699318

theorem true_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + 2*x - m = 0) ∧            -- Condition 1
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧                    -- Condition 2
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ) ∧
  (∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧              -- Condition 3
  ¬ ( (∀ p q : Prop, ¬p → ¬ (p ∧ q)) ∧ (¬ ¬p → p ∧ q) ) ∧   -- Condition 4
  (∃ x : ℝ, x^2 + x + 3 ≤ 0)                                 -- Condition 5
:= by {
  sorry
}

end true_propositions_l699_699318


namespace total_time_taken_l699_699987

variable (x y : ℝ)

def time_first_segment (x : ℝ) : ℝ := x / 50
def time_second_segment (x : ℝ) : ℝ := (2 * x) / 75
def time_stop : ℝ := 1 / 6
def time_third_segment (y : ℝ) : ℝ := y / 100

theorem total_time_taken (x y : ℝ) : 
  (time_first_segment x + time_second_segment x + time_stop + time_third_segment y) = (14 * x + 3 * y + 50) / 300 :=
by
  sorry

end total_time_taken_l699_699987


namespace find_intersection_and_sum_l699_699829

def point := ℝ × ℝ

def A : point := (0, 0)
def B : point := (2, 4)
def C : point := (6, 6)
def D : point := (8, 0)

def quadrilateral_area (A B C D : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 - B.1 * A.2) + (B.1 * C.2 - C.1 * B.2) + (C.1 * D.2 - D.1 * C.2) + (D.1 * A.2 - A.1 * D.2))

def line_through_A_and_point (A : point) (x y : ℝ) : ℝ := y / x

def intersects_at (A : point) (C D : point) : point :=
  let m := (D.2 - C.2) / (D.1 - C.1)
  let line_eq := fun x : ℝ => m * (x - C.1) + C.2
  let y_intersect := line_eq 0
  (16 / 3, y_intersect)

theorem find_intersection_and_sum (p q r s : ℝ) : p + q + r + s = 28 := by
  sorry

end find_intersection_and_sum_l699_699829


namespace smallest_polynomial_degree_l699_699865

theorem smallest_polynomial_degree :
  ∃ (P : Polynomial ℚ), P ≠ 0 ∧
    P.root_set ℚ = {3 - Real.sqrt 8, 3 + Real.sqrt 8, 
                    5 - Real.sqrt 12, 5 + Real.sqrt 12, 
                    12 - 2 * Real.sqrt 11, 12 + 2 * Real.sqrt 11, 
                    -2 * Real.sqrt 3, 2 * Real.sqrt 3} ∧ P.degree = 8 :=
sorry

end smallest_polynomial_degree_l699_699865


namespace centroid_A_l699_699368

-- Scalars triangle ABC with orthocenter H and centroid M
variables {A B C H M : Point}

-- Points defining the orthocenter and centroid
variables (orthocenter_ABC : orthocenter A B C = H)
          (centroid_ABC : centroid A B C = M)

-- Lines through A, B, C perpendicular to AM, BM, CM respectively
variables (A_perp : Line.through_perpendicular A (line_through A M))
          (B_perp : Line.through_perpendicular B (line_through B M))
          (C_perp : Line.through_perpendicular C (line_through C M))

-- Triangle defined by these lines
abbreviation A' := point_on_line A_perp
abbreviation B' := point_on_line B_perp
abbreviation C' := point_on_line C_perp

-- Defining the triangle A'B'C'
def ΔA'B'C' := triangle A' B' C'

-- Proposition to prove
theorem centroid_A'B'C'_on_MH (scalene_ABC : scalene_triangle A B C) :
  line_through M H ∋ (centroid A' B' C') :=
sorry

end centroid_A_l699_699368


namespace units_digit_G_1000_l699_699603

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 2 :=
by
  sorry

end units_digit_G_1000_l699_699603


namespace min_area_of_triangle_l699_699953

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (3 * p - 5 * q)

theorem min_area_of_triangle :
  (∀ p q : ℤ, p ≠ 0 ∨ q ≠ 0 → area_of_triangle p q ≥ (1 / 2 : ℚ)) ∧
  (∃ p q : ℤ, p ≠ 0 ∨ q ≠ 0 ∧ area_of_triangle p q = (1 / 2 : ℚ)) := 
by { 
  sorry 
}

end min_area_of_triangle_l699_699953


namespace power_logarithm_eighth_root_l699_699165

theorem power_logarithm_eighth_root :
  ((256 ^ (Real.log 2018 / Real.log 2)) ^ (1 / 8) = 2018) :=
by
  have h1 : 256 = 2 ^ 8 := by norm_num
  have h2 : (a:ℝ) ^ m ^ n = a ^ (m * n) := sorry
  have h3 : Real.log b (a ^ n) = n * Real.log b a := sorry
  have h4 : (a:ℝ) ^ (Real.log a x) = x := sorry
  have h5 : (a ^ n) ^ (1 / m) = a ^ (n / m) := sorry
  sorry

end power_logarithm_eighth_root_l699_699165


namespace number_of_triangles_l699_699101

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l699_699101


namespace probability_divisible_by_15_in_S_l699_699756

def S : finset ℕ := {n | ∃ j k : ℕ, j < k ∧ k ≤ 39 ∧ n = 2^j + 2^k}

def is_divisible_by_15 (n : ℕ) : Prop := n % 15 = 0

theorem probability_divisible_by_15_in_S :
  (let count_divisible := (S.filter is_divisible_by_15).card
   let total_count := S.card
   let p := count_divisible
   let q := total_count / nat.gcd count_divisible total_count
   p + q = 49) :=
by
  sorry

end probability_divisible_by_15_in_S_l699_699756


namespace sqrt_lt_diff_of_reverse_l699_699828

def digits_reverse (n : ℕ) : ℕ :=
-- Function to find the reverse of the digits of a number (details to be filled in the actual function)
sorry

theorem sqrt_lt_diff_of_reverse (N : ℕ) (h : ∀ F, F = digits_reverse N → N > F) : 
  ∃ F, F = digits_reverse N ∧ sqrt (N) < N - F :=
by
  sorry

end sqrt_lt_diff_of_reverse_l699_699828


namespace gamma_needs_7_hours_l699_699205

noncomputable def gamma_time (A B C D : ℕ) : ℕ := sorry

theorem gamma_needs_7_hours :
  ∃ (A B C D : ℕ), 
  (A = 12) ∧ (B = 15) ∧ 
  (1 / C + 1 / D = 1 / 10) ∧
  (1 / 12 + 1 / 15 + 1 / C + 1 / D = 1 / (C - 3)) → 
  C = 7 :=
begin
  sorry
end

end gamma_needs_7_hours_l699_699205


namespace prove_monotonic_increasing_range_l699_699804

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699804


namespace problem_statement_l699_699202

def round_to_nearest_hundredth (x: ℝ) : ℝ :=
  let hundredths := (Real.floor (x * 100 + 0.5)) / 100
  hundredths

theorem problem_statement : round_to_nearest_hundredth (78.1563 + 24.3981) = 102.55 :=
by
  sorry

end problem_statement_l699_699202


namespace new_stationary_points_relationship_l699_699608

theorem new_stationary_points_relationship :
  (∃ a : ℝ, 0 < a ∧ a < π ∧ sin a = cos a) ∧
  (∃ b : ℝ, 1 < b ∧ b < real.exp 1 ∧ ln b = 1 / b) ∧
  (∃ c : ℝ, c = 3 ∧ c ≠ 0) →
  ∃ (a b c : ℝ), (0 < a) ∧ (a < π) ∧ (sin a = cos a) ∧
  (1 < b) ∧ (b < real.exp 1) ∧ (ln b = 1 / b) ∧ 
  (c = 3) ∧ (a < b ∧ b < c) :=
by sorry

end new_stationary_points_relationship_l699_699608


namespace elevator_passengers_probability_l699_699991

noncomputable def binomial_pdf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem elevator_passengers_probability :
  binomial_pdf 5 (1/3) 4 = 10 / 243 :=
by
  sorry

end elevator_passengers_probability_l699_699991


namespace rainbow_preschool_full_day_students_l699_699996

theorem rainbow_preschool_full_day_students (total_students : ℕ) (half_day_percent : ℝ)
  (h1 : total_students = 80) (h2 : half_day_percent = 0.25) :
  (total_students * (1 - half_day_percent)).to_nat = 60 :=
by
  -- Transform percentage to a fraction
  let fraction_full_day := 1 - half_day_percent
  -- Calculate full-day students
  have h_full_day_students : ℝ := total_students * fraction_full_day
  -- Convert to natural number
  exact (floor h_full_day_students).to_nat = 60

end rainbow_preschool_full_day_students_l699_699996


namespace distance_origin_to_line_l699_699878

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the line equation as a function
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the point-to-line distance formula
def distance_from_origin_to_line (A B C x0 y0 : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The main theorem stating that the distance is √5
theorem distance_origin_to_line : distance_from_origin_to_line 2 1 (-5) 0 0 = real.sqrt 5 := 
sorry

end distance_origin_to_line_l699_699878


namespace rainbow_preschool_full_day_students_l699_699995

theorem rainbow_preschool_full_day_students (total_students : ℕ) (half_day_percent : ℝ)
  (h1 : total_students = 80) (h2 : half_day_percent = 0.25) :
  (total_students * (1 - half_day_percent)).to_nat = 60 :=
by
  -- Transform percentage to a fraction
  let fraction_full_day := 1 - half_day_percent
  -- Calculate full-day students
  have h_full_day_students : ℝ := total_students * fraction_full_day
  -- Convert to natural number
  exact (floor h_full_day_students).to_nat = 60

end rainbow_preschool_full_day_students_l699_699995


namespace squared_sum_inverse_l699_699652

theorem squared_sum_inverse (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 :=
by
  sorry

end squared_sum_inverse_l699_699652


namespace slope_of_equally_dividing_line_l699_699901

noncomputable def circle := { center : ℝ × ℝ, radius : ℝ }

def circle1 : circle := { center := (10, 50), radius := 2 }
def circle2 : circle := { center := (12, 34), radius := 2 }
def circle3 : circle := { center := (15, 40), radius := 2 }

def line_passing_through (m : ℝ) (p : ℝ × ℝ) := {
  a : ℝ // ∃ b : ℝ, line_eq m b p = 0
}


-- Proving that the line passing through (12, 34) with slope m = 2.76 equally divides the circle areas.
theorem slope_of_equally_dividing_line :
  ∃ m, ∀ line_passing_through m (12, 34), abs m = 2.76 :=
sorry

end slope_of_equally_dividing_line_l699_699901


namespace exists_k_m_for_n_l699_699275

-- Define the function f
def f (m k : ℕ) : ℕ :=
  ((List.range 5).sum (λ i, ⌊m * Real.sqrt ((k + 1) / (i + 1))⌋))

-- Prove the existence of k and m for any n
theorem exists_k_m_for_n (n : ℕ) (h₀ : 0 < n) : 
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 5 ∧ f m k = n :=
by sorry

end exists_k_m_for_n_l699_699275


namespace fraction_addition_solution_is_six_l699_699347

theorem fraction_addition_solution_is_six :
  (1 / 9) + (1 / 18) = 1 / 6 := 
sorry

end fraction_addition_solution_is_six_l699_699347


namespace area_of_triangle_ABC_l699_699379

theorem area_of_triangle_ABC {A B C M H : Point} (h_right_angle : is_right_angle ∠ A C B)
  (h_c_trisects : ∃ CH CM : Segment, is_altitude (triangle A C B) CH ∧ is_median (triangle A C B) CM ∧ trisects_angle ∠ A C B CH CM)
  (h_area_chm : area (triangle C H M) = k) :
  area (triangle A B C) = 4 * k :=
sorry

end area_of_triangle_ABC_l699_699379


namespace street_lights_equilateral_triangle_l699_699480

-- Define the context and conditions
def side_length : ℝ := 10
def interval : ℝ := 3
def perimeter (side_length : ℝ) : ℝ := 3 * side_length
def num_street_lights (perimeter interval : ℝ) : ℝ := perimeter / interval

-- State the theorem
theorem street_lights_equilateral_triangle :
  num_street_lights (perimeter side_length) interval = 10 :=
by
  sorry

end street_lights_equilateral_triangle_l699_699480


namespace max_rectangle_area_with_prime_dimension_l699_699979

theorem max_rectangle_area_with_prime_dimension :
  ∃ (l w : ℕ), 2 * (l + w) = 120 ∧ (Prime l ∨ Prime w) ∧ l * w = 899 :=
by
  sorry

end max_rectangle_area_with_prime_dimension_l699_699979


namespace sum_of_coefficients_l699_699349

theorem sum_of_coefficients (n : ℕ) (x : ℝ) :
  (∀ k, k ≠ 0 → k ≠ 6 + 1 - 1 → binomial 6 k < binomial 6 (6 + 1 - 1)) →
  (x ^ 3 - 1 / (2 * x)) ^ 6 = 1 / 64 :=
by
  sorry

end sum_of_coefficients_l699_699349


namespace quadratic_has_two_roots_l699_699743

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l699_699743


namespace math_problem_l699_699704
-- Import necessary modules

-- Define the condition as a hypothesis and state the theorem
theorem math_problem (x : ℝ) (h : 8 * x - 6 = 10) : 50 * (1 / x) + 150 = 175 :=
sorry

end math_problem_l699_699704


namespace selection_properties_l699_699822

theorem selection_properties (n : ℕ) (s : Finset ℕ) (h : s.card = n + 1 ∧ ∀ x ∈ s, x ≤ 2 * n):
  ∃ a b ∈ s, Nat.coprime a b ∧ ∃ u v ∈ s, u ≠ v ∧ (u % v = 0 ∨ v % u = 0) :=
by
  sorry

end selection_properties_l699_699822


namespace complex_expression_solve_combination_l699_699959

-- First Problem
theorem complex_expression (i : ℂ) (h : i = complex.I) :
  ((1 + i) / (1 - i))^2 + complex.abs (3 + 4 * i) - i^2017 = 4 - i :=
  sorry

-- Second Problem
theorem solve_combination (x : ℕ) (hx : x > 6) :
  2 * nat.choose (x - 3) (x - 6) = 5 * nat.choose (x - 4) 2 → x = 18 :=
  sorry

end complex_expression_solve_combination_l699_699959


namespace geometric_sequence_product_l699_699733

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n, a (n + 1) = a n * r

noncomputable def quadratic_roots (a1 a10 : ℝ) : Prop :=
3 * a1^2 - 2 * a1 - 6 = 0 ∧ 3 * a10^2 - 2 * a10 - 6 = 0

theorem geometric_sequence_product {a : ℕ → ℝ}
  (h_geom : geometric_sequence a)
  (h_roots : quadratic_roots (a 1) (a 10)) :
  a 4 * a 7 = -2 :=
sorry

end geometric_sequence_product_l699_699733


namespace sum_of_digits_of_product_l699_699472

def is_thousands_digit (n : ℕ) : ℕ :=
  (n / 1000) % 10

def is_units_digit (n : ℕ) : ℕ :=
  n % 10

def hundred_digit_number (repeated_pattern : ℕ) : ℕ :=
  repeated_pattern * 10^97 + repeated_pattern * 10^94 + ... + repeated_pattern

theorem sum_of_digits_of_product :
  let num1 := hundred_digit_number 707
  let num2 := hundred_digit_number 606
  let product := num1 * num2
  is_thousands_digit product + is_units_digit product = 10 :=
sorry

end sum_of_digits_of_product_l699_699472


namespace smallest_sum_of_primes_with_unique_digits_l699_699169

def isPrime (n : ℕ) : Prop := sorry
def usesAllDigitsOnce (numbers : List ℕ) : Prop :=
  (numbers.joinDigits.to_list.perm [1, 2, 3, 4, 5, 6, 7, 8, 9])

theorem smallest_sum_of_primes_with_unique_digits : ∃ (ps : List ℕ), 
  ps.length = 4 ∧ 
  (∀ p ∈ ps, isPrime p) ∧ 
  usesAllDigitsOnce ps ∧ 
  ps.sum = 620 :=
sorry

end smallest_sum_of_primes_with_unique_digits_l699_699169


namespace count_paths_matematika_count_paths_matematika_after_elimination_l699_699517

-- Define the chart and movement conditions for the word MATEMATIKA
def adj (x y: α) : α → α → Prop := sorry -- adjacency relation

def in_chart (s: list α) : Prop := sorry -- elements must be in chart

theorem count_paths_matematika (chart: list (list α)) : 
    (count_paths "MATEMATIKA" chart) = 528 :=
sorry

-- Theorem statement for eliminating paths to reduce total to exactly 500 ways
theorem count_paths_matematika_after_elimination (chart: list (list α)) (elims: list (list α)): 
    apply_elimination chart elims -> 
    (count_paths "MATEMATIKA" (elims + chart)) = 500 :=
sorry

end count_paths_matematika_count_paths_matematika_after_elimination_l699_699517


namespace point_symmetry_x_axis_l699_699708

structure Point :=
(x : ℝ)
(y : ℝ)

def is_symmetric_about_x_axis (A B : Point) : Prop :=
A.x = B.x ∧ A.y = -B.y

theorem point_symmetry_x_axis : 
  ∀ (x y : ℝ), 
  is_symmetric_about_x_axis 
    {x := x, y := y} 
    {x := x, y := -y} :=
by
  intros x y
  unfold is_symmetric_about_x_axis
  split
  . refl
  . refl

end point_symmetry_x_axis_l699_699708


namespace new_number_is_100t_plus_10u_plus_3_l699_699010

theorem new_number_is_100t_plus_10u_plus_3 (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  new_number = 100 * t + 10 * u + 3 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  show new_number = 100 * t + 10 * u + 3
  sorry

end new_number_is_100t_plus_10u_plus_3_l699_699010


namespace math_problem_l699_699258

theorem math_problem (a b c d : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) (hd : d < 9)
  (hab : Nat.gcd a 9 = 1) (hbb : Nat.gcd b 9 = 1) (hcb : Nat.gcd c 9 = 1) (hdb : Nat.gcd d 9 = 1)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * Nat.gcd_inv (a * b * c * d) 9) % 9 = 6 := by
  sorry

end math_problem_l699_699258


namespace perpendicularity_of_intersections_l699_699718

open EuclideanGeometry

theorem perpendicularity_of_intersections
  (ABC : Triangle)
  (A B C E F : Point)
  (H1 : E ∈ line(A, C))
  (H2 : F ∈ line(A, B))
  (O : Point)
  (O' : Point)
  (H3 : circumcenter(O, triangle(A, B, C)))
  (H4 : circumcenter(O', triangle(A, E, F)))
  (P : Point)
  (Q : Point)
  (H5 : P ∈ segment(B, E))
  (H6 : Q ∈ segment(C, F))
  (H7 : BP :/ PE = FQ :/ QC = (BF^2 / CE^2)) :
  orthogonal(line(O,O'), line(P,Q)) :=
by
  sorry

end perpendicularity_of_intersections_l699_699718


namespace triangles_from_ten_points_l699_699118

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l699_699118


namespace problem_equivalent_statement_l699_699665

-- Conditions as Lean definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_property (f : ℝ → ℝ) := ∀ x, x ≥ 0 → f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 8

-- The main theorem
theorem problem_equivalent_statement (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_periodic : periodic_property f) 
  (hf_specific : specific_interval f) :
  f (-2013) + f 2014 = 1 / 3 := 
sorry

end problem_equivalent_statement_l699_699665


namespace triangles_from_ten_points_l699_699116

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l699_699116


namespace salary_increase_by_90_per_1000_unit_l699_699885

theorem salary_increase_by_90_per_1000_unit (x : ℕ) : 
  let salary := 60 + 90 * x in
  let new_salary := 60 + 90 * (x + 1) in
  new_salary - salary = 90 :=
by
  sorry

end salary_increase_by_90_per_1000_unit_l699_699885


namespace calculate_expression_l699_699227

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l699_699227


namespace tables_needed_for_luncheon_l699_699970

theorem tables_needed_for_luncheon (invited attending remaining tables_needed : ℕ) (H1 : invited = 24) (H2 : remaining = 10) (H3 : attending = invited - remaining) (H4 : tables_needed = attending / 7) : tables_needed = 2 :=
by
  sorry

end tables_needed_for_luncheon_l699_699970


namespace u_seq_limit_l699_699446

noncomputable def u_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := Real.sqrt (12 + u_seq n)

theorem u_seq_limit : ∃ l : ℝ, l = 4 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u_seq n - l| < ε :=
by
  have h₀ : u_seq 0 = 0 := rfl
  have h_rec : ∀ n, u_seq (n + 1) = Real.sqrt (12 + u_seq n) := by intros; refl
  
  -- using these facts and induction, we aim to show that the limit is 4
  sorry

end u_seq_limit_l699_699446


namespace intersection_complement_l699_699314

open Set

variables {U : Type} [LinearOrder U] [ArchimedeanLinearOrder U]

def A : Set U := {x | x^2 - 2 * x < 0}
def B : Set U := {x | x >= 1}

theorem intersection_complement :
  A ∩ (U \ B) = {x | 0 < x ∧ x < 1} :=
begin
  sorry
end

end intersection_complement_l699_699314


namespace prank_combinations_l699_699488

-- Conditions stated as definitions
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 6
def friday_choices : ℕ := 2

-- Theorem to prove
theorem prank_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 180 :=
by
  sorry

end prank_combinations_l699_699488


namespace part1_part2_l699_699814

noncomputable section

open Real

def f (x a : ℝ) : ℝ := abs (2*x - 1) + abs (x + a)

theorem part1 (x : ℝ) : f x 1 ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ abs (a - 1)) ↔ a ∈ Iic (1/4) :=
by
  sorry

end part1_part2_l699_699814


namespace probability_at_least_three_consecutive_heads_l699_699186

def is_fair_coin (p : ℝ) := p = 1/2

def fair_coin_toss (n : ℕ) : Set (Vector Bool n) := 
  {s | ∀ i < n, s.get i = tt ∨ s.get i = ff}

def at_least_three_consecutive_heads (s : Vector Bool 4) : Prop :=
  (s = [tt, tt, tt, tt] ∨
   s = [tt, tt, tt, ff] ∨
   s = [ff, tt, tt, tt])

def probability_of_event (event : Set (Vector Bool 4)) (total : Set (Vector Bool 4)) : ℝ :=
  (event.card / total.card : ℝ)

theorem probability_at_least_three_consecutive_heads :
  ∀ (total : Set (Vector Bool 4)), 
  total = (fair_coin_toss 4) →
  probability_of_event {s | at_least_three_consecutive_heads s} total = 3 / 16 :=
by 
  sorry

end probability_at_least_three_consecutive_heads_l699_699186


namespace reject_null_hypothesis_likelihood_estimation_expectation_value_l699_699537

def contingency_table : Type := 
  { a: ℕ // a = 50 } × { b: ℕ // b = 30 } × { c: ℕ // c = 40 } × { d: ℕ // d = 80 } × 
  { n: ℕ // n = 200 } × { total1: ℕ // total1 = 90 } × { total2: ℕ // total2 = 110 } ×
  { total3: ℕ // total3 = 80 } × { total4: ℕ // total4 = 120 }

def chi_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2) / ((a + b)*(c + d)*(a + c)*(b + d))

theorem reject_null_hypothesis (ct : contingency_table) : Prop :=
  let ⟨⟨a, ha⟩, ⟨b, hb⟩, ⟨c, hc⟩, ⟨d, hd⟩, ⟨n, hn⟩, ⟨total1, ht1⟩, ⟨total2, ht2⟩, ⟨total3, ht3⟩, ⟨total4, ht4⟩⟩ := ct in
  chi_squared n a b c d > 6.635

def likelihood_ratio (ct : contingency_table) : ℚ :=
  let ⟨⟨a, ha⟩, ⟨b, hb⟩, ⟨c, hc⟩, ⟨d, hd⟩, ⟨n, hn⟩, ⟨total1, ht1⟩, ⟨total2, ht2⟩, ⟨total3, ht3⟩, ⟨total4, ht4⟩⟩ := ct in
  (d : ℚ) / (b : ℚ)

theorem likelihood_estimation (ct : contingency_table) : likelihood_ratio ct = 8 / 3 :=
  sorry

def probability_distribution (ct : contingency_table) : ℕ → ℚ :=
  λ x, match x with
  | 0 => 1 / 56
  | 1 => 15 / 56
  | 2 => 15 / 28
  | 3 => 5 / 28
  | _ => 0
  end

def expectation (ct : contingency_table) : ℚ :=
  0 * (1 / 56) + 1 * (15 / 56) + 2 * (15 / 28) + 3 * (5 / 28)

theorem expectation_value (ct : contingency_table) : expectation ct = 15 / 8 :=
  sorry

end reject_null_hypothesis_likelihood_estimation_expectation_value_l699_699537


namespace inequality_proof_l699_699320

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem inequality_proof (α β : ℝ) (m : ℕ) (h1 : 1 < α) (h2 : 1 < β) (h3 : m = 1) 
  (h4 : f α m + f β m = 2) : (4 / α) + (1 / β) ≥ 9 / 2 := by
  sorry

end inequality_proof_l699_699320


namespace range_of_a_l699_699759

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699759


namespace sandy_initial_amount_l699_699437

theorem sandy_initial_amount 
  (cost_shirt : ℝ) (cost_jacket : ℝ) (found_money : ℝ)
  (h1 : cost_shirt = 12.14) (h2 : cost_jacket = 9.28) (h3 : found_money = 7.43) : 
  (cost_shirt + cost_jacket + found_money = 28.85) :=
by
  rw [h1, h2, h3]
  norm_num

end sandy_initial_amount_l699_699437


namespace beautiful_ratio_l699_699134

theorem beautiful_ratio (A B C : Type) (l1 l2 b : ℕ) 
  (h : l1 + l2 + b = 20) (h1 : l1 = 8 ∨ l2 = 8 ∨ b = 8) :
  (b / l1 = 1/2) ∨ (b / l2 = 1/2) ∨ (l1 / l2 = 4/3) ∨ (l2 / l1 = 4/3) :=
by
  sorry

end beautiful_ratio_l699_699134


namespace eccentricity_of_hyperbola_l699_699683

open Real

-- Conditions as parameters
variables (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
variables (A : ℝ × ℝ) (h_A : A = (a, 0))
variables (M N : ℝ × ℝ) (angle_MAN : real.angle MAN = 60)
variables (asymptote_eq : ∀ x y, bx + ay = 0)

-- Define the hyperbola C and its properties
def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle centered at A with radius b that intersects the asymptote
def circle (x y : ℝ) := (x - a)^2 + y^2 = b^2

-- The proof we need to complete is to show 
theorem eccentricity_of_hyperbola : 
  let c := sqrt (a^2 + b^2) in 
  c / a = 2 / sqrt 3 :=
sorry

end eccentricity_of_hyperbola_l699_699683


namespace sum_of_n_maximized_l699_699298

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def max_sum_term (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∑ i in range n, a i = ∑ i in range n, a i
  
noncomputable def max_n (a : ℕ → ℝ) (h1 : a 17 + a 18 + a 19 > 0) (h2 : a 17 + a 20 < 0) : ℕ :=
  18

theorem sum_of_n_maximized (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 17 + a 18 + a 19 > 0) (h2 : a 17 + a 20 < 0) : max_sum_term a (max_n a h1 h2) := 
sorry

end sum_of_n_maximized_l699_699298


namespace sequence_check_l699_699208

theorem sequence_check : 
  (∃ n : ℕ, 380 = n * (n + 1)) ∧ 
  ¬ (∃ n : ℕ, 39 = n * (n + 1)) ∧
  ¬ (∃ n : ℕ, 35 = n * (n + 1)) ∧
  ¬ (∃ n : ℕ, 23 = n * (n + 1)) :=
begin
  sorry
end

end sequence_check_l699_699208


namespace find_valid_numbers_l699_699421

def is_valid_number (n : ℕ) : Prop :=
  let h := n / 100 in
  let t := (n / 10) % 10 in
  let u := n % 10 in
  h < 4 ∧ n = (t + u)^h

theorem find_valid_numbers :
  ∃ n1 n2 : ℕ, is_valid_number n1 ∧ is_valid_number n2 ∧ n1 = 289 ∧ n2 = 343 := by
  let n1 := 289
  let n2 := 343
  have h1 : is_valid_number n1 := by
    -- Proof that 289 satisfies the conditions -- 
    sorry
  have h2 : is_valid_number n2 := by
    -- Proof that 343 satisfies the conditions -- 
    sorry
  use [n1, n2]
  exact ⟨h1, h2, rfl, rfl⟩

end find_valid_numbers_l699_699421


namespace flat_tyre_problem_l699_699570

theorem flat_tyre_problem
    (x : ℝ)
    (h1 : 0 < x)
    (h2 : 1 / x + 1 / 6 = 1 / 5.6) :
  x = 84 :=
sorry

end flat_tyre_problem_l699_699570


namespace chips_needed_per_console_l699_699976

-- Definitions based on the conditions
def chips_per_day : ℕ := 467
def consoles_per_day : ℕ := 93

-- The goal is to prove that each video game console needs 5 computer chips
theorem chips_needed_per_console : chips_per_day / consoles_per_day = 5 :=
by sorry

end chips_needed_per_console_l699_699976


namespace last_digit_of_1_div_2_pow_15_l699_699929

theorem last_digit_of_1_div_2_pow_15 :
  let last_digit_of := (n : ℕ) → n % 10
  last_digit_of (5^15) = 5 → 
  (∀ (n : ℕ),  ∃ (k : ℕ), n = 2^k →  last_digit_of (5 ^ k) = last_digit_of (1 / 2 ^ 15)) := 
by 
  intro last_digit_of h proof
  exact sorry

end last_digit_of_1_div_2_pow_15_l699_699929


namespace circle_radius_l699_699628

theorem circle_radius :
  ∃ r : ℝ, ∀ x y : ℝ, (x^2 - 8 * x + y^2 + 4 * y + 16 = 0) → r = 2 :=
sorry

end circle_radius_l699_699628


namespace problem_statement_l699_699066

def X0_area : ℝ := (1 / 2) * 3 * 4

def perimeter_X0 : ℝ := 3 + 4 + 5

def area_Xn (n : ℕ) : ℝ := 
  X0_area + n * perimeter_X0 + (n * n) * Real.pi

def area_difference (X_n1 X_n : ℝ) : ℝ := X_n1 - X_n

def a : ℤ := 41
def b : ℤ := 12

theorem problem_statement : 100 * a + (b : ℤ) = 4112 := by
  sorry

end problem_statement_l699_699066


namespace cost_prices_correct_l699_699196

noncomputable def cost_price (selling_price profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage)

theorem cost_prices_correct :
  cost_price 400 0.25 = 320 ∧ cost_price 600 0.20 = 500 ∧ cost_price 800 0.15 ≈ 695.65 :=
by
  sorry

end cost_prices_correct_l699_699196


namespace triangle_AD_squared_eq_four_R_squared_minus_AB_AC_l699_699036

open Classical

variable 
  (A B C D I O : Type)
  [Incenter I A B C]
  [IsSymmetric D I O]
  (R : ℝ) -- radius of the circumcircle

theorem triangle_AD_squared_eq_four_R_squared_minus_AB_AC
  (ABC : Triangle A B C)
  (symm_DI_O : ∀ P, IsSymmetric P I O ↔ P = D)
  (circumradius : ∀ P Q, Circumradius P Q R) :
  AD^2 = 4 * R^2 - AB * AC :=
by
  sorry

end triangle_AD_squared_eq_four_R_squared_minus_AB_AC_l699_699036


namespace range_of_function_l699_699133

theorem range_of_function : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ 1 - 2 * sin x ∧ 1 - 2 * sin x ≤ 3 :=
by 
  intros x h
  cases h with h1 h2
  have : -2 * sin x ≤ 2 := mul_le_mul_of_nonpos_right h1 (by linarith [sin_nonneg h2])
  split; linarith [sin_nonpos h1, sin_nonneg h2]

end range_of_function_l699_699133


namespace halfway_fraction_l699_699879

theorem halfway_fraction (a b : ℚ) (h1 : a = 1/5) (h2 : b = 1/3) : (a + b) / 2 = 4 / 15 :=
by 
  rw [h1, h2]
  norm_num

end halfway_fraction_l699_699879


namespace find_point_D_l699_699049

-- Define the parabola equation y = x^2 + 1
def parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the point C
def C : ℝ × ℝ := (2, 4)

-- Define the tangent slope at C
def tangent_slope (x : ℝ) : ℝ := 2 * x

-- Define the normal slope at C
def normal_slope (x : ℝ) : ℝ := -1 / (tangent_slope x)

-- Define the normal line equation at C
def normal_line (x : ℝ) : ℝ :=
  let (cx, cy) := C
  in normal_slope cx * (x - cx) + cy

-- The statement that point D intersects the parabola
theorem find_point_D : ∃ D : ℝ × ℝ, D = (-2, 5) ∧ parabola D.1 = D.2 ∧ normal_line D.1 = D.2 := 
by
  sorry

end find_point_D_l699_699049


namespace monotonic_increasing_on_interval_l699_699672

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem monotonic_increasing_on_interval (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / (2 * ω) = 4 * Real.pi) :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 2) Real.pi) → (y ∈ Set.Icc (Real.pi / 2) Real.pi) → x ≤ y → f ω x ≤ f ω y := 
by
  sorry

end monotonic_increasing_on_interval_l699_699672


namespace diana_slow_speed_l699_699243

def initial_speed : ℝ := 3   -- Diana's initial speed in mph
def initial_time : ℝ := 2    -- Time she bikes at initial speed in hours
def total_distance : ℝ := 10 -- Total distance Diana needs to bike in miles
def total_time : ℝ := 6      -- Total time Diana takes to get home in hours

theorem diana_slow_speed :
  let distance_covered := initial_speed * initial_time in
  let remaining_distance := total_distance - distance_covered in
  let remaining_time := total_time - initial_time in
  remaining_time ≠ 0 ∧
  (remaining_distance / remaining_time = 1) := 
by 
  sorry

end diana_slow_speed_l699_699243


namespace find_m_range_l699_699301

def p (m : ℝ) : Prop := (4 - 4 * m) ≤ 0
def q (m : ℝ) : Prop := (5 - 2 * m) > 1

theorem find_m_range (m : ℝ) (hp_false : ¬ p m) (hq_true : q m) : 1 ≤ m ∧ m < 2 :=
by {
 sorry
}

end find_m_range_l699_699301


namespace gcd_of_12347_and_9876_l699_699155

theorem gcd_of_12347_and_9876 : Nat.gcd 12347 9876 = 7 :=
by
  sorry

end gcd_of_12347_and_9876_l699_699155


namespace max_value_2016_inequality_l699_699825

theorem max_value_2016_inequality 
  (x y z v w : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < v ∧ 0 < w) 
  (h_sum : x^2 + y^2 + z^2 + v^2 + w^2 = 2016) 
  (h_eq : x = 4 ∧ y = 8 ∧ z = 12 * real.sqrt 7 ∧ v = 12 ∧ w = 28) : 
  (let M := x * z + 2 * y * z + 3 * z * v + 7 * z * w in
  M + x + y + z + v + w = 52 + 3036 * real.sqrt 7) := 
by 
  sorry

end max_value_2016_inequality_l699_699825


namespace bike_covered_distance_l699_699962

theorem bike_covered_distance
  (time : ℕ) 
  (truck_distance : ℕ) 
  (speed_difference : ℕ) 
  (bike_speed truck_speed : ℕ)
  (h_time : time = 8)
  (h_truck_distance : truck_distance = 112)
  (h_speed_difference : speed_difference = 3)
  (h_truck_speed : truck_speed = truck_distance / time)
  (h_speed_relation : truck_speed = bike_speed + speed_difference) :
  bike_speed * time = 88 :=
by
  -- The proof is omitted
  sorry

end bike_covered_distance_l699_699962


namespace lunch_to_read_ratio_l699_699890

theorem lunch_to_read_ratio 
  (total_pages : ℕ) (pages_per_hour : ℕ) (lunch_hours : ℕ)
  (h₁ : total_pages = 4000)
  (h₂ : pages_per_hour = 250)
  (h₃ : lunch_hours = 4) :
  lunch_hours / (total_pages / pages_per_hour) = 1 / 4 := by
  sorry

end lunch_to_read_ratio_l699_699890


namespace sequence_sum_l699_699231

theorem sequence_sum : 
  let seq := list.range' 3 18 -- This generates the sequence 3, 7, 11, ..., up to 71
  let signs := list.cycle [1, -1] -- This generates the alternating signs
  let full_seq := list.zip_with (*) seq signs -- Element-wise product with alternating signs
  list.sum full_seq = -36 := sorry

end sequence_sum_l699_699231


namespace least_positive_integer_with_12_factors_consecutive_primes_l699_699930

theorem least_positive_integer_with_12_factors_consecutive_primes :
  ∃ k : ℕ, (factors_count k = 12) ∧ (consecutive_primes_used k) ∧ (∀ n : ℕ, (factors_count n = 12) ∧ (consecutive_primes_used n) → k ≤ n) := 
sorry

noncomputable def factors_count (n : ℕ) : ℕ := 
-- returns the number of positive factors of n
sorry

noncomputable def consecutive_primes_used (n : ℕ) : Prop :=
-- returns true if the factorization of n uses only consecutive primes
sorry

end least_positive_integer_with_12_factors_consecutive_primes_l699_699930


namespace percent_bluegrass_in_X_equals_60_l699_699858

variable (X_ryegrass Y_ryegrass Y_fescue mixture_ryegrass X_ratio : ℝ)

-- Conditions
axiom h1 : X_ryegrass = 0.40
axiom h2 : Y_ryegrass = 0.25
axiom h3 : Y_fescue = 0.75
axiom h4 : mixture_ryegrass = 0.38
axiom h5 : X_ratio = 0.8667

-- Question: Prove that the percent of seed mixture X is bluegrass equals 60%
theorem percent_bluegrass_in_X_equals_60 : (1 - X_ryegrass) * 100 = 60 :=
by
  have h6 : 1 - X_ryegrass = 0.60 :=
  by
    calc
      1 - X_ryegrass = 1 - 0.4 : by rw [h1]
      ... = 0.60 : by norm_num
  calc
    (1 - X_ryegrass) * 100 = 0.60 * 100 : by rw [h6]
    ... = 60 : by norm_num

end percent_bluegrass_in_X_equals_60_l699_699858


namespace female_democrats_count_l699_699145

variable (F M D_F D_M : ℕ)

-- Conditions
def total_participants : Prop := F + M = 990
def half_female_democrats : Prop := D_F = 1/2 * F
def quarter_male_democrats : Prop := D_M = 1/4 * M
def total_democrats : Prop := D_F + D_M = 1/3 * 990

-- Goal
theorem female_democrats_count (h1 : total_participants) (h2 : half_female_democrats) 
    (h3 : quarter_male_democrats) (h4 : total_democrats) : D_F = 165 := 
by 
  sorry

end female_democrats_count_l699_699145


namespace fraction_of_darker_tiles_is_4_by_9_l699_699543

-- Definitions for conditions
def entire_floor_tiled : Prop := true  -- Placeholder for the condition stating the entire floor is tiled in this way

def corner_same_as_others (block : ℕ → ℕ → bool) : Prop := 
  ∀ i j, (1 ≤ i ∧ i ≤ 6) → (1 ≤ j ∧ j ≤ 6) → block i j = block (7 - i) j ∧ block i j = block i (7 - j)

def repeating_unit_6x6 (block : ℕ → ℕ → bool) : Prop := 
  ∀ i j, (1 ≤ i ∧ i ≤ 6) → (1 ≤ j ∧ j ≤ 6) → block i j = block (i + 6) j ∧ block i j = block i (j + 6)

-- Lean statement for the proof
theorem fraction_of_darker_tiles_is_4_by_9 (block : ℕ → ℕ → bool) 
  (h1 : entire_floor_tiled) 
  (h2 : corner_same_as_others block) 
  (h3 : repeating_unit_6x6 block) : 
  (∑ i in finset.range 3, ∑ j in finset.range 3, if block i j then 1 else 0) / 9 = 4 / 9 :=
by
  sorry

end fraction_of_darker_tiles_is_4_by_9_l699_699543


namespace intersect_fourth_line_l699_699067

noncomputable def intersects_all_or_none (Δ : Type) (AB' BC' CD' DA': Δ → Prop) :=
  (AB' Δ ∧ BC' Δ ∧ CD' Δ) → DA' Δ

-- Given a parallelepiped and a line Δ, if Δ intersects three of the lines AB', BC', CD', DA', then it intersects the fourth.
theorem intersect_fourth_line
  (parallelepiped : Type)
  (A B C D A' B' C' D': parallelepiped)
  (Δ : parallelepiped → Prop)
  (AB' BC' CD' DA' : parallelepiped → Prop) 
  (h_AB' : AB' = (λ p, p = A ∨ p = B'))
  (h_BC' : BC' = (λ p, p = B ∨ p = C'))
  (h_CD' : CD' = (λ p, p = C ∨ p = D'))
  (h_DA' : DA' = (λ p, p = D ∨ p = A')) :
  intersects_all_or_none Δ AB' BC' CD' DA' :=
by sorry

end intersect_fourth_line_l699_699067


namespace guessing_secret_number_probability_l699_699087

theorem guessing_secret_number_probability :
  let S := {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ h t u, n = 100 * h + 10 * t + u ∧ h % 2 = 0 ∧ t % 2 = 1 ∧ u % 2 = 0 ∧ n > 500)} in
  1 / S.card = 1 / 50 :=
by
  -- state the proof problem without proof details
  sorry

end guessing_secret_number_probability_l699_699087


namespace total_time_l699_699747

def T (n : ℕ) : ℝ := 5 + 3 * Real.log n

theorem total_time (h1 : T 1 = 5) 
    (h2 : ∀ n, n > 0 → n ≤ 5 → T n = 5 + 3 * Real.log n) :
    (Float.round ((T 1) + (T 2) + (T 3) + (T 4) + (T 5))).toNat = 39 :=
by 
  sorry

end total_time_l699_699747


namespace number_of_triangles_l699_699106

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l699_699106


namespace mother_l699_699520

theorem mother's_age (D M : ℕ) (h1 : 2 * D + M = 70) (h2 : D + 2 * M = 95) : M = 40 :=
sorry

end mother_l699_699520


namespace range_of_a_l699_699760

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699760


namespace problem_l699_699706

theorem problem (m : ℝ) (h : m + 1/m = 6) : m^2 + 1/m^2 + 3 = 37 :=
by
  sorry

end problem_l699_699706


namespace tractor_cost_higher_in_january_l699_699020

theorem tractor_cost_higher_in_january :
  (λ (jan_screw_price : ℝ) (jan_bolt_price : ℝ) (feb_set_price : ℝ) (tractor_screws : ℕ) (tractor_bolts : ℕ),
    jan_screw_price = 40 ∧ jan_bolt_price = 60 ∧ feb_set_price = 1
    ∧ tractor_screws = 600 ∧ tractor_bolts = 600 →
    let jan_cost := (tractor_screws / jan_screw_price) + (tractor_bolts / jan_bolt_price) in
    let feb_cost := ((tractor_screws / 25) : ℝ) * feb_set_price in
    jan_cost > feb_cost) :=
by
  intros jan_screw_price jan_bolt_price feb_set_price tractor_screws tractor_bolts h
  rcases h with ⟨hscrew_price, hbolt_price, hset_price, hscrews, hbolts⟩
  rw [hscrew_price, hbolt_price, hset_price, hscrews, hbolts]
  have jan_cost := (600 / 40) + (600 / 60)
  have feb_cost := (600 / 25) * 1
  change jan_cost > feb_cost
  sorry

end tractor_cost_higher_in_january_l699_699020


namespace ellipse_equation_l699_699653

noncomputable def point := (ℝ × ℝ)

theorem ellipse_equation (a b : ℝ) (P Q : point) (h1 : a > b) (h2: b > 0) (e : ℝ) (h3 : e = 1/2)
  (h4 : P = (2, 3)) (h5 : Q = (2, -3))
  (h6 : (P.1^2)/(a^2) + (P.2^2)/(b^2) = 1) (h7 : (Q.1^2)/(a^2) + (Q.2^2)/(b^2) = 1) :
  (∀ x y: ℝ, (x^2/16 + y^2/12 = 1) ↔ (x^2/a^2 + y^2/b^2 = 1)) :=
sorry

end ellipse_equation_l699_699653


namespace cubic_root_sum_cubic_root_sum_no_solution_l699_699827

theorem cubic_root_sum (p : ℝ) (hp : 0 < p ∧ p ≤ 2) :
  ∃ x : ℝ, (∛(1 - x) + ∛(1 + x) = p) ↔ 
    x = sqrt(1 - (p^3 - 2) / (3 * p)) ∨ 
    x = -sqrt(1 - (p^3 - 2) / (3 * p)) :=
by
  sorry

theorem cubic_root_sum_no_solution (p : ℝ) (hp : p ≤ 0 ∨ 2 < p) :
  ¬ ∃ x : ℝ, (∛(1 - x) + ∛(1 + x) = p) :=
by
  sorry

end cubic_root_sum_cubic_root_sum_no_solution_l699_699827


namespace distance_from_M_to_OP_l699_699210

-- Definitions for the given conditions
variable {x : ℝ}
def A : ℝ × ℝ := (1, 0)
def P (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def M (x : ℝ) : ℝ × ℝ := (Real.cos x, 0)

-- Definition for the function f(x)
def f (x : ℝ) : ℝ := |Real.sin x * Real.cos x|

-- Theorem stating the proof goal
theorem distance_from_M_to_OP :
  ∀ x : ℝ, True → f x = |Real.sin x * Real.cos x|
:= by
  intro x h
  trivial

end distance_from_M_to_OP_l699_699210


namespace advertising_time_l699_699881

-- Define the conditions
def total_duration : ℕ := 30
def national_news : ℕ := 12
def international_news : ℕ := 5
def sports : ℕ := 5
def weather_forecasts : ℕ := 2

-- Calculate total content time
def total_content_time : ℕ := national_news + international_news + sports + weather_forecasts

-- Define the proof problem
theorem advertising_time (h : total_duration - total_content_time = 6) : (total_duration - total_content_time) = 6 :=
by
sorry

end advertising_time_l699_699881


namespace m_minus_n_eq_six_l699_699701

theorem m_minus_n_eq_six (m n : ℝ) (h : ∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) : m - n = 6 := by
  sorry

end m_minus_n_eq_six_l699_699701


namespace cone_lateral_surface_area_l699_699290

noncomputable def lateral_surface_area_cone (r h : ℝ) := r * Real.sqrt (r^2 + h^2) * π

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 1) (hh : h = Real.sqrt 3) :
  lateral_surface_area_cone r h = 2 * π :=
by
  unfold lateral_surface_area_cone
  rw [hr, hh]
  norm_num
  have : Real.sqrt (1^2 + (Real.sqrt 3)^2) = 2 := sorry
  rw [this]
  norm_num
  sorry

end cone_lateral_surface_area_l699_699290


namespace min_moves_to_equalize_coins_l699_699440

theorem min_moves_to_equalize_coins :
  let boxes := [2, 3, 5, 10, 15, 17, 20]
  let target := 10
  let moves := λ b : Nat, if b < target then target - b else b - target
  (sum (List.map moves boxes) = 22) :=
by
  sorry

end min_moves_to_equalize_coins_l699_699440


namespace median_on_hypotenuse_l699_699726

theorem median_on_hypotenuse (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hb : b = 12) (hc : c = 13) :
  median a b c = c / 2 :=
by
  sorry

end median_on_hypotenuse_l699_699726


namespace gifts_receiving_ribbon_l699_699041

def total_ribbon := 18
def ribbon_per_gift := 2
def remaining_ribbon := 6

theorem gifts_receiving_ribbon : (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 := by
  sorry

end gifts_receiving_ribbon_l699_699041


namespace range_of_a_l699_699761

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699761


namespace find_room_length_l699_699882

variable (w : ℝ) (C : ℝ) (r : ℝ)

theorem find_room_length (h_w : w = 4.75) (h_C : C = 29925) (h_r : r = 900) : (C / r) / w = 7 := by
  sorry

end find_room_length_l699_699882


namespace concurrency_of_lines_l699_699183

theorem concurrency_of_lines
  (A B C D E : Point)
  (c : Circle)
  (hcenter : c.center = A)
  (hexists_B : c ∈ {B})
  (hexists_E : c ∈ {E})
  (hregular : is_regular_pentagon A B C D E)
  (F G : Point)
  (hBC_second_intersection : second_intersection (line_through B C) c = F)
  (hG_condition : (G ∈ c) ∧ (distance F B = distance F G) ∧ (B ≠ G)) :
  concurrent (line_through A B) (line_through E F) (line_through D G) :=
sorry

end concurrency_of_lines_l699_699183


namespace basketball_scores_distinct_counts_l699_699533

theorem basketball_scores_distinct_counts (baskets: ℕ) (two_point : ℕ) (three_point: ℕ) 
(h1: baskets = 5)
(h2: ∀ x, x ∈ [2, 3]):
  ∃ total_points: finset ℕ, total_points.card = 6 :=
by sorry

end basketball_scores_distinct_counts_l699_699533


namespace worm_distance_after_15_days_l699_699173

theorem worm_distance_after_15_days : 
  let distance_per_day := 5 - 3 in
  15 * distance_per_day = 30 :=
by
  let distance_per_day := 5 - 3
  sorry

end worm_distance_after_15_days_l699_699173


namespace part_I_part_II_l699_699297

section PartI
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
-- Conditions
axiom a1 : a 1 = 1
axiom H : ∀ n : ℕ, n ≥ 2 → S n * S n = a n * (S n - 1 / 2)

-- Part (I)
theorem part_I : ∀ n : ℕ, n ≥ 2 → ∃ d : ℝ, ∀ m : ℕ, m ≥ 1 → (1 / S (m + 1) - 1 / S m = d) := sorry
end PartI

section PartII
variables (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
-- Conditions
axiom Sn_form : ∀ n : ℕ, n ≥ 2 → S n = 1 / (2 * n - 1)
axiom bn_form : ∀ n : ℕ, n ≥ 2 → b n = S n * S (n + 1)
axiom Tn_form : ∀ n : ℕ, n ≥ 2 → T n = ∑ k in finset.range n, b (k + 1)

-- Part (II)
theorem part_II : ∀ n : ℕ, n ≥ 2 → T n < 1 / 2 := sorry
end PartII

end part_I_part_II_l699_699297


namespace exists_nontrivial_normal_subgroup_l699_699752

theorem exists_nontrivial_normal_subgroup
  (n : ℕ) (hn_even : even n) (hn_geq_4 : 4 ≤ n)
  (G : Subgroup (GL(2, ℂ))) (hG_card : fintype.card G = n) :
  ∃ (H : Subgroup G), H ≠ ⊥ ∧ H ≠ ⊤ ∧ ∀ X ∈ G, ∀ Y ∈ H, X * Y * X⁻¹ ∈ H :=
sorry

end exists_nontrivial_normal_subgroup_l699_699752


namespace translated_parabola_shift_l699_699495

theorem translated_parabola_shift (x : ℝ) :
  let y1 := -x^2 + 2 in
  let y2 := -(x + 3)^2 + 2 in
  (∀ x : ℝ, y2 = -(x + 3)^2 + 2) := 
by
  sorry

end translated_parabola_shift_l699_699495


namespace range_of_m_l699_699717

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
noncomputable def g (x m : ℝ) : ℝ := (1 / 2) * x - m

theorem range_of_m (h1 : ∀ x ∈ Set.Icc 0 3, f x ≥ 0)
                   (h2 : ∀ x ∈ Set.Icc 1 2, g x m ≤ (1 / 2) - m) :
  Set.Ici (1 / 2) ⊆ {m : ℝ | ∀ x ∈ Set.Icc 0 3, ∀ x' ∈ Set.Icc 1 2, f x ≥ g x' m } := 
by
  intros m hm
  sorry

end range_of_m_l699_699717


namespace od_oe_of_eq_cg_l699_699359

variable (α : ℝ) -- Angle α formed with the sides of the triangle
variable (x y z m : ℝ) -- Lengths of perpendiculars and height of triangle

-- Assume x + y + z = m
axiom h1 : x + y + z = m

-- Assume the trigonometric relationships
def OD := x / Real.sin α
def OE := y / Real.sin α
def OF := z / Real.sin α

-- Define CG in terms of the height of the triangle and angle α
def CG := m / Real.sin α

-- Theorem statement to prove OD + OE + OF = CG
theorem od_oe_of_eq_cg : OD α x y z + OE α x y z + OF α x y z = CG α m := by
  sorry

end od_oe_of_eq_cg_l699_699359


namespace maximum_profit_l699_699569

def profit (x : ℝ) : ℝ :=
  if x ≤ 20 then 800 * x - 10000
  else -10 * x^2 + 1000 * x - 10000

theorem maximum_profit : ∃ x ∈ set.Icc (0 : ℝ) 75, profit x = 15000 :=
by
  -- Proof omitted
  sorry

end maximum_profit_l699_699569


namespace num_even_digit_numbers_l699_699336

/-- The number of distinct three-digit positive integers with only even digits and no leading zero
    is 100. -/
theorem num_even_digit_numbers : 
  let even_digits := {0, 2, 4, 6, 8}
  let non_zero_even_digits := {2, 4, 6, 8}
  (card non_zero_even_digits) * (card even_digits) * (card even_digits) = 100 :=
by
  sorry

end num_even_digit_numbers_l699_699336


namespace sum_of_a_and_b_l699_699356

theorem sum_of_a_and_b (a b : ℕ) 
  (h : (3/2 : ℝ) * (4/3) * (5/4) * (6/5) * ... * (a/b) = 9) : 
  a + b = 35 
:= 
sorry

end sum_of_a_and_b_l699_699356


namespace regular_tetrahedron_is_convex_regular_hexahedron_is_convex_cube_regular_octahedron_is_convex_l699_699518

-- a) Regular Tetrahedron
theorem regular_tetrahedron_is_convex :
  ∀ (T : Type) [regular_tetrahedron T], convex_regular_tetrahedron T := by
  sorry

-- b) Regular Hexahedron (Cube)
theorem regular_hexahedron_is_convex_cube :
  ∀ (H : Type) [regular_hexahedron H], convex_regular_hexahedron H := by
  sorry

-- c) Regular Octahedron
theorem regular_octahedron_is_convex :
  ∀ (O : Type) [regular_octahedron O], convex_regular_octahedron O := by
  sorry

end regular_tetrahedron_is_convex_regular_hexahedron_is_convex_cube_regular_octahedron_is_convex_l699_699518


namespace unique_point_value_l699_699100

noncomputable def unique_point_condition : Prop :=
  ∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + 12 = 0

theorem unique_point_value (d : ℝ) : unique_point_condition ↔ d = 12 := 
sorry

end unique_point_value_l699_699100


namespace slope_line_through_origin_divides_parallelogram_l699_699604

-- Definition of points and conditions for the parallelogram
def A := (3, 15) : ℕ × ℕ
def B := (3, 42) : ℕ × ℕ
def C := (12, 66) : ℕ × ℕ
def D := (12, 39) : ℕ × ℕ

-- Given a line through the origin symmetrically divides the parallelogram
-- into two congruent polygons, prove the sum of the coprime integers
-- forming the slope of this line is 32.

theorem slope_line_through_origin_divides_parallelogram :
  ∃ (m n : ℕ), (nat.gcd m n = 1) ∧ (m / n = 27 / 5) ∧ (m + n = 32) :=
sorry

end slope_line_through_origin_divides_parallelogram_l699_699604


namespace real_number_m_l699_699700

theorem real_number_m (m : ℝ) (h : (m^2 - 5 * m + 4) + (m^2 - 2 * m) * complex.I > 0) : m = 0 :=
by {
    sorry
}

end real_number_m_l699_699700


namespace woman_lawyer_probability_l699_699530

noncomputable def probability_of_woman_lawyer : ℚ :=
  let total_members : ℚ := 100
  let women_percentage : ℚ := 0.80
  let lawyer_percentage_women : ℚ := 0.40
  let women_members := women_percentage * total_members
  let women_lawyers := lawyer_percentage_women * women_members
  let probability := women_lawyers / total_members
  probability

theorem woman_lawyer_probability :
  probability_of_woman_lawyer = 0.32 := by
  sorry

end woman_lawyer_probability_l699_699530


namespace scientific_notation_of_11700000_l699_699175

theorem scientific_notation_of_11700000 :
  hasScientificNotation 11700000 1.17 (7 : ℤ) := 
sorry

end scientific_notation_of_11700000_l699_699175


namespace remainder_of_875_div_by_170_l699_699464

theorem remainder_of_875_div_by_170 :
  ∃ r, (∀ x, x ∣ 680 ∧ x ∣ (875 - r) → x ≤ 170) ∧ 170 ∣ (875 - r) ∧ r = 25 :=
by
  sorry

end remainder_of_875_div_by_170_l699_699464


namespace log2_x_condition_l699_699526

noncomputable def sufficient_condition (x : ℝ) : Prop := log 2 x < 1
noncomputable def necessary_condition (x : ℝ) : Prop := x^2 < x

theorem log2_x_condition : ∀ x : ℝ, sufficient_condition x → necessary_condition x ∧ ¬(necessary_condition x → sufficient_condition x) :=
by
  intro x
  sorry

end log2_x_condition_l699_699526


namespace lucy_fish_moved_l699_699071

theorem lucy_fish_moved (original_count moved_count remaining_count : ℝ)
  (h1: original_count = 212.0)
  (h2: remaining_count = 144.0) :
  moved_count = original_count - remaining_count :=
by sorry

end lucy_fish_moved_l699_699071


namespace moles_of_NH3_combined_l699_699627

/-- 
Given that each mole of NH3 forms exactly 1 mole of NH4Cl, 
and that we start with 1 mole of HCl, 
prove that 1 mole of NH3 is combined to form NH4Cl.
-/
theorem moles_of_NH3_combined (moles_HCl moles_NH4Cl: ℕ) 
    (one_mole_NH3_forms_one_mole_NH4Cl: ∀ NH3: ℕ, NH3 == moles_NH4Cl): 
    moles_HCl = 1 → moles_NH4Cl = 1 → ∃ moles_NH3, moles_NH3 = 1 := 
by 
  intro h_hcl h_nh4cl 
  use 1 
  simp [h_hcl, h_nh4cl, one_mole_NH3_forms_one_mole_NH4Cl]
  sorry

end moles_of_NH3_combined_l699_699627


namespace seq_sum_min_no_max_l699_699458

theorem seq_sum_min_no_max (a_1 d : ℝ) (S : ℕ → ℝ)
    (hSn : ∀ n, S n = n * (a_1 + (n - 1) * d / 2))
    (hd_pos : d > abs a_1) :
    (∀ n, S n ≥ S 1 ∧ ∀ m n, n > m → S n > S m ∧ (∀ k, S k → ∃ N, ∀ n > N, S n > k)) ∧
    ¬ (d = abs a_1 → ∀ n, S n ≥ S 1 ∧ ∀ m n, n > m → S n > S m ∧ (∀ k, S k → ∃ N, ∀ n > N, S n > k)) :=
by
  sorry

end seq_sum_min_no_max_l699_699458


namespace original_faculty_number_l699_699561

theorem original_faculty_number (x : ℝ) (h : 0.85 * x = 195) : x = 229 := by
  sorry

end original_faculty_number_l699_699561


namespace num_12_digit_with_consecutive_1s_l699_699695

def total_12_digit_numbers : ℕ := 2^12

def F : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| (n+1) := F n + F (n-1)

def F_12 : ℕ := F 12

theorem num_12_digit_with_consecutive_1s : total_12_digit_numbers - F_12 = 3719 :=
by
  unfold total_12_digit_numbers F_12 F
  sorry

end num_12_digit_with_consecutive_1s_l699_699695


namespace no_rhombus_l699_699038

-- Definition of convex polygon
structure ConvexPolygon where
  vertices : Set Point
  isConvex : ∀ (v1 v2 : Point), (v1 ∈ vertices) → (v2 ∈ vertices) → (v1 ≠ v2) → 
    ∀ (p : Point), (p ∈ LineSegment v1 v2) → (p ∈ vertices)

-- Definition of inscribed quadrilateral
def IsInscribed (quad : Quadrilateral) (poly : ConvexPolygon) : Prop :=
  ∀ v ∈ quad.vertices, v ∈ poly.vertices

-- Definition of inscribed rhombus
def IsInscribedRhombus (rhombus : Rhombus) (poly : ConvexPolygon) : Prop :=
  ∀ v ∈ rhombus.vertices, v ∈ poly.vertices

-- Main statement to show
theorem no_rhombus (poly : ConvexPolygon) :
  ¬ ∃ (rhombus : Rhombus), IsInscribedRhombus rhombus poly ∧
  ∀ (quad : Quadrilateral), IsInscribed quad poly → 
  ∃ s, s ∈ quad.sides ∧ s ≤ rhombus.side :=
sorry

end no_rhombus_l699_699038


namespace probability_reaching_target_l699_699854

-- Definitions for points
def Point : Type := (ℤ × ℤ × ℤ)

-- Definitions for vertices of the pyramid
def E : Point := (10, 10, 0)
def A : Point := (10, -10, 0)
def R : Point := (-10, -10, 0)
def L : Point := (-10, 10, 0)
def Y : Point := (0, 0, 10)

-- Movement rules
def possibleMoves (p : Point) : List Point := 
  let (x, y, z) := p
  [(x, y, z-1), (x+1, y, z-1), (x-1, y, z-1),
   (x, y+1, z-1), (x, y-1, z-1), 
   (x+1, y+1, z-1), (x-1, y+1, z-1),
   (x+1, y-1, z-1), (x-1, y-1, z-1)]

-- Starting at point Y
def initialPosition : Point := Y

-- Theorem: Probability of reaching (8, 9, 0)
theorem probability_reaching_target : 
  let target := (8, 9, 0)
  let steps := 10
  let probability := 550 / (9^10 : ℚ)
  Sean_probability initialPosition target steps = probability :=
sorry

end probability_reaching_target_l699_699854


namespace value_of_f_at_3_l699_699460

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end value_of_f_at_3_l699_699460


namespace monotonic_increasing_range_l699_699792

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699792


namespace food_expenditure_increase_l699_699568

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end food_expenditure_increase_l699_699568


namespace number_of_triangles_l699_699111

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l699_699111


namespace max_value_of_f_l699_699264

def f (x : ℝ) := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem max_value_of_f : ∃ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f x ≥ f y ∧ f x = 5 :=
by
  sorry

end max_value_of_f_l699_699264


namespace vertices_in_parallel_planes_l699_699546

noncomputable theory

-- Assume a defined structure for 3D point and polyhedra
structure Point3D : Type :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Polyhedron : Type :=
  (vertices : list Point3D)
  (faces : list (list Point3D))

-- We define dodecahedron and icosahedron as specific polyhedra
def dodecahedron : Polyhedron := sorry
def icosahedron : Polyhedron := sorry

-- Assume a predicate PlaneParallel to determine plane parallelism
def PlaneParallel (p1 p2 : list Point3D) := sorry

-- We state the condition that one face of each polyhedron and their opposite face lie in the same planes
axiom face_dodecahedron_in_plane (f : list Point3D) (H_f : f ∈ dodecahedron.faces) 
  (g : list Point3D) (H_g : g ∈ icosahedron.faces) : face_parallel f g

axiom face_opposite_in_plane (f1 f2 : list Point3D) 
  (H_f1 : f1 ∈ dodecahedron.faces) (H_f2 : f2 ∈ icosahedron.faces):  (opposite_face f1 dodecahedron) = (opposite_face f2 icosahedron)

-- The theorem to be proven
theorem vertices_in_parallel_planes (dodecahedron icosahedron : Polyhedron)
  (face1 face2 : list Point3D)
  (Hf1 : face1 ∈ dodecahedron.faces)
  (Hf2 : face2 ∈ icosahedron.faces)
  (H_parallel : PlaneParallel face1 face2)
  (H_opposite_parallel : PlaneParallel (opposite_face face1 dodecahedron) (opposite_face face2 icosahedron)): 
  ∀ v ∈ (vertices_not_in_faces dodecahedron face1 (opposite_face face1 dodecahedron)), 
    ∃ p q : list Point3D, PlaneParallel p face1 ∧ PlaneParallel q (opposite_face face1 dodecahedron) := 
sorry

end vertices_in_parallel_planes_l699_699546


namespace triangles_from_ten_points_l699_699120

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l699_699120


namespace max_m_value_l699_699867

variables {a b c t x m : ℝ}
variables f : ℝ → ℝ

-- Definitions and conditions
def quadratic_fn (f : ℝ → ℝ) := ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

def condition1 := ∀ x: ℝ, f (x-4) = f (2-x) ∧ f x ≥ x

def condition2 := ∀ x: ℝ, 0 < x ∧ x < 2 → f x ≤ (x+1)/2^2

def condition3 := ∀ x: ℝ, ∀ y, f y = 0

-- Main theorem
theorem max_m_value : 
  quadratic_fn f → condition1 f → condition2 f → condition3 f →
  (∃ m > 1, ∃ t: ℝ, ∀ x ∈ set.Icc 1 m, f (x + t) ≤ x) → m = 9 :=
begin
  sorry
end

end max_m_value_l699_699867


namespace perpendicular_line_through_point_l699_699457

theorem perpendicular_line_through_point 
  (P : Point)
  (H : P = (-1, 3) ∧ is_perpendicular P (x - 2y + 3 = 0))
  : (line_eq P = 2 * x + y - 1 = 0) :=
by
  sorry

end perpendicular_line_through_point_l699_699457


namespace inverse_function_of_f_l699_699263

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

theorem inverse_function_of_f:
  ∀ x : ℝ, x ≠ 3 → f (f_inv x) = x ∧ f_inv (f x) = x := by
sorry

end inverse_function_of_f_l699_699263


namespace bananas_left_correct_l699_699613

def initial_bananas : ℕ := 12
def eaten_bananas : ℕ := 1
def bananas_left (initial eaten : ℕ) := initial - eaten

theorem bananas_left_correct : bananas_left initial_bananas eaten_bananas = 11 :=
by
  sorry

end bananas_left_correct_l699_699613


namespace right_triangle_area_proof_l699_699924

noncomputable def right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1 / 2) * a * b

theorem right_triangle_area_proof (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 40) (hc : c = 41) :
right_triangle_area a b c h = 180 :=
by
  rw [ha, hc]
  have hb := sqrt (c^2 - a^2)
  sorry

end right_triangle_area_proof_l699_699924


namespace part1_part2_l699_699282

def f (x : ℝ) : ℝ := 9^x / (9^x + 3)

theorem part1 (a : ℝ) (h : 0 < a ∧ a < 1) : f(a) + f(1 - a) = 1 := 
by sorry

theorem part2 : 
  (∑ k in finset.range 999, f ((k + 1) / 1000)) = (999 / 2) := 
by sorry

end part1_part2_l699_699282


namespace pq_on_perpendicular_left_mn_l699_699887

variables (A B C D P Q M N O : Type) [circumscribed_quadrilateral A B C D O]

-- Definitions of properties
def circle_o (A B C D O : Type) : Prop :=
  inscribed_quadrilateral A B C D O ∧ 
  center_of_arc B C M O ∧ 
  center_of_arc D A N O

def intersection_of_bisectors (A B C D P Q : Type) : Prop :=
  intersection_of_angle_bisectors D A B P ∧ 
  intersection_of_angle_bisectors B C D Q


-- The corresponding proof problem statement
theorem pq_on_perpendicular_left_mn :
  circle_o A B C D O →
  intersection_of_bisectors A B C D P Q →
  perpendicular_line_through P Q M N :=
sorry

end pq_on_perpendicular_left_mn_l699_699887


namespace student_rank_from_left_l699_699201

theorem student_rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_right = 16) 
  (h3 : total_students = rank_from_right + rank_from_left - 1) 
  : rank_from_left = 6 := 
by 
  sorry

end student_rank_from_left_l699_699201


namespace multiples_of_4_between_88_and_104_l699_699894

theorem multiples_of_4_between_88_and_104 : 
  ∃ n, (104 - 4 * 23 = n) ∧ n = 88 ∧ ( ∀ x, (x ≥ 88 ∧ x ≤ 104 ∧ x % 4 = 0) → ( x - 88) / 4 < 24) :=
by
  sorry

end multiples_of_4_between_88_and_104_l699_699894


namespace communication_system_connections_l699_699722

theorem communication_system_connections (n : ℕ) (h : ∀ k < 2001, ∃ l < 2001, l ≠ k ∧ k ≠ l) :
  (∀ k < 2001, ∃ l < 2001, k ≠ l) → (n % 2 = 0 ∧ n ≤ 2000) ∨ n = 0 :=
sorry

end communication_system_connections_l699_699722


namespace quadratic_trinomials_with_roots_eq_without_l699_699237

open Nat Int

theorem quadratic_trinomials_with_roots_eq_without :
  let quadratics_with_roots := {p : ℕ × ℕ | p.1 ≠ p.2 ∧ p.1 ≥ 1 ∧ p.2 ≥ 1 ∧ p.1 ≤ 100 ∧ p.2 ≤ 100 ∧ p.1 ≥ p.2},
      quadratics_without_roots := {p : ℕ × ℕ | p.1 ≠ p.2 ∧ p.1 ≥ 1 ∧ p.2 ≥ 1 ∧ p.1 ≤ 100 ∧ p.2 ≤ 100 ∧ p.1 < p.2}
  in quadratics_with_roots.card = quadratics_without_roots.card := 
sorry

end quadratic_trinomials_with_roots_eq_without_l699_699237


namespace sequence_geometric_general_term_l699_699331

def a : ℕ → ℝ
| 0       := 2/3
| (n + 1) := 2 * a n / (a n + 1)

theorem sequence_geometric :
  (∀ n : ℕ, n ≠ 0 → (1 / (a (n + 1)) - 1 = 1/2 * (1 / (a n) - 1))) →
  (1 / (a 1) - 1 = 1 / 2) →
  (∀ n : ℕ, (1 / (a n) - 1) = (1 / 2) * (1 / (a (n - 1)) - 1)) ∧ 
    (∀ m : ℕ, 
        let seq : ℕ → ℝ := λ n, 1 / (a n) - 1 in 
        seq m = (1/2)^m) :=
  sorry

theorem general_term :
  (a 0 = 2/3) →
  (∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 1)) →
  (∀ n : ℕ, a n = 2^n / (1 + 2^n)) :=
  sorry

end sequence_geometric_general_term_l699_699331


namespace reduced_price_l699_699553

variable (original_price : ℝ) (final_amount : ℝ)

noncomputable def sales_tax (price : ℝ) : ℝ :=
  if price <= 2500 then price * 0.04
  else if price <= 4500 then 2500 * 0.04 + (price - 2500) * 0.07
  else 2500 * 0.04 + 2000 * 0.07 + (price - 4500) * 0.09

noncomputable def discount (price : ℝ) : ℝ :=
  if price <= 2000 then price * 0.02
  else if price <= 4000 then 2000 * 0.02 + (price - 2000) * 0.05
  else 2000 * 0.02 + 2000 * 0.05 + (price - 4000) * 0.10

theorem reduced_price (P : ℝ) (original_price := 5000) (final_amount := 2468) :
  P = original_price - discount original_price + sales_tax original_price → P = 2423 :=
by
  sorry

end reduced_price_l699_699553


namespace calc_expr_l699_699224

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l699_699224


namespace garden_width_l699_699749

theorem garden_width (w : ℕ) (h1 : ∀ l : ℕ, l = w + 12 → l * w ≥ 120) : w = 6 := 
by
  sorry

end garden_width_l699_699749


namespace area_of_triangle_tangent_at_pi_div_two_l699_699870

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_at_pi_div_two :
  let x := Real.pi / 2
  let slope := 1 + Real.cos x
  let point := (x, f x)
  let intercept_y := f x - slope * x
  let x_intercept := -intercept_y / slope
  let y_intercept := intercept_y
  (1 / 2) * x_intercept * y_intercept = 1 / 2 := 
by
  sorry

end area_of_triangle_tangent_at_pi_div_two_l699_699870


namespace mark_owe_triple_amount_l699_699072

theorem mark_owe_triple_amount (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 2000) (hr : r = 0.04) :
  (1 + r)^t > 3 → t = 30 :=
by
  intro h
  norm_cast at h
  sorry

end mark_owe_triple_amount_l699_699072


namespace first_problem_second_problem_l699_699060

-- First problem:
theorem first_problem (a m n : ℕ) (h_a_gt_one : a > 1) (h_div : (a^m + 1) % (a^n + 1) = 0) : n ∣ m := 
sorry

-- Second problem:
theorem second_problem (a b m n : ℕ) (h_coprime : Nat.coprime a b) (h_a_gt_one : a > 1) (h_div : (a^m + b^m) % (a^n + b^n) = 0) : n ∣ m := 
sorry

end first_problem_second_problem_l699_699060


namespace least_positive_x_l699_699159

theorem least_positive_x (x : ℕ) (h : (2 * x + 45)^2 % 43 = 0) : x = 42 :=
  sorry

end least_positive_x_l699_699159


namespace calculate_expression_l699_699594

theorem calculate_expression :
  (-1: ℝ) ^ 2023 + real.sqrt 36 - real.cbrt 8 + |real.sqrt 5 - 2| = real.sqrt 5 + 1 := by
  sorry

end calculate_expression_l699_699594


namespace tetrahedron_min_edges_diff_lengths_l699_699707

noncomputable def min_distinct_edge_lengths_if_not_isosceles (T : Tetrahedron) : ℕ :=
  if ∀ f ∈ faces T, ¬Isosceles f then 3 else 0

theorem tetrahedron_min_edges_diff_lengths :
  ∀ (T : Tetrahedron), (∀ f ∈ faces T, ¬Isosceles f) → min_distinct_edge_lengths_if_not_isosceles T = 3 := 
by
  intros T h
  sorry

end tetrahedron_min_edges_diff_lengths_l699_699707


namespace number_of_incorrect_conditions_l699_699207

open Set

-- Definitions of each condition
def condition1 := {0} ∈ ({0, 2, 3} : Set (Set Nat))
def condition2 := (∅: Set Nat) ⊆ ({0}: Set Nat)
def condition3 := ({0, 1, 2}: Set Nat) ⊆ ({0, 1, 2} : Set Nat)
def condition4 := (0 ∈ ∅: Set Nat)
def condition5 := (∅ = (∅: Set Nat): Set Nat)

-- Statement of the proof problem
theorem number_of_incorrect_conditions : 
  ((¬condition1) + (¬condition2) + (¬condition3) + (condition4) + (condition5)) = 2 :=
by
  sorry

end number_of_incorrect_conditions_l699_699207


namespace factorization_identity_l699_699253

theorem factorization_identity (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2 * m - 1)^2 :=
sorry

end factorization_identity_l699_699253


namespace fastest_rate_ammonia_production_l699_699135

def reaction_rate (v_H2 v_N2 v_NH3 : ℕ → ℕ → Prop) : Prop :=
  ∀ (A B C D : ℕ),
  let v_H2_A := 0.3
  let v_N2_B := 0.2
  let v_NH3_C := 0.25
  let v_H2_D := 0.4 in
  (v_N2_B = 0.2) →
  (v_H2 _ A = 3 * v_N2 _ B) → (v_H2 _ B = 0.6) →
  (v_H2 _ A = 1.5 * v_NH3 _ C) → (v_H2 _ C = 0.375) →
  v_H2 _ D = 0.4 →
  max (v_H2 _ A) (max (v_H2 _ B) (max (v_H2 _ C) (v_H2 _ D))) = v_H2 _ B

theorem fastest_rate_ammonia_production : reaction_rate v_H2 v_N2 v_NH3 := by
  sorry

end fastest_rate_ammonia_production_l699_699135


namespace unique_zero_range_l699_699283

-- Define the given function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the unique zero condition and the proof requirement
theorem unique_zero_range (a : ℝ) (x₀ : ℝ) (h₀ : f a x₀ = 0) (h₁ : ∀ x, x ≠ x₀ → f a x ≠ 0) (h₂ : x₀ > 0) :
  a ∈ Iio (-2) := 
sorry

end unique_zero_range_l699_699283


namespace find_k_series_sum_l699_699269

theorem find_k_series_sum :
  (∃ k : ℝ, 5 + ∑' n : ℕ, ((5 + (n + 1) * k) / 5^n.succ) = 10) →
  k = 12 :=
sorry

end find_k_series_sum_l699_699269


namespace quadratic_has_two_roots_l699_699744

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l699_699744


namespace area_difference_l699_699179

-- Definitions of the conditions
def length_rect := 60 -- length of the rectangular garden in feet
def width_rect := 20 -- width of the rectangular garden in feet

-- Compute the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Compute the perimeter of the rectangular garden
def perimeter_rect := 2 * (length_rect + width_rect)

-- Compute the side length of the square garden from the same perimeter
def side_square := perimeter_rect / 4

-- Compute the area of the square garden
def area_square := side_square * side_square

-- The goal is to prove the area difference
theorem area_difference : area_square - area_rect = 400 := by
  sorry -- Proof to be completed

end area_difference_l699_699179


namespace bisectors_concurrency_l699_699045

noncomputable theory

section HexagonBisectors

variables {ABCDEF : Type}
variables (A B C D E F : ABCDEF)
variables [ConvexHexagon ABCDEF] 
variables (h1 : ∠ A = ∠ C)
variables (h2 : ∠ C = ∠ E)
variables (h3 : ∠ B = ∠ D)
variables (h4 : ∠ D = ∠ F)
variables (h_concurrent1 : is_concurrent (angle_bisector A) (angle_bisector C) (angle_bisector E))

theorem bisectors_concurrency :
  is_concurrent (angle_bisector B) (angle_bisector D) (angle_bisector F) :=
sorry

end HexagonBisectors

end bisectors_concurrency_l699_699045


namespace complex_division_l699_699956

theorem complex_division (i : ℂ) (h : i = complex.I) : (i / (2 + i)) = (1 + 2 * i) / 5 := by
  sorry

end complex_division_l699_699956


namespace caramels_distribution_l699_699835

theorem caramels_distribution (x y : ℕ) 
  (h1 : y + x = 5)
  (h2 : 2 * x = 6)
  (h3 : x + 2 * y + x = 10)
  (h4 : 26 = y + y + x + (x + 2 * y + x)) : 
  3x + 2y = 13 :=
by
  sorry

end caramels_distribution_l699_699835


namespace sin_squared_range_l699_699288

theorem sin_squared_range (α β : ℝ) (h : 3 * (sin α)^2 + 2 * (sin β)^2 = 2 * sin α) : 
  0 ≤ (sin α)^2 + (sin β)^2 ∧ (sin α)^2 + (sin β)^2 ≤ 4 / 9 := 
sorry

end sin_squared_range_l699_699288


namespace tetrahedron_ineq_l699_699725

noncomputable theory
open classical

theorem tetrahedron_ineq (A B C D : Point) 
    (angle_BDC : angle B D C = 90) 
    (H_as_ortho : is_orthocenter H A B C) : 
    (dist A B + dist B C + dist C A)^2 ≤ 6 * (dist A D ^ 2 + dist B D ^ 2 + dist C D ^ 2) :=
sorry

end tetrahedron_ineq_l699_699725


namespace solve_quadratic_eq_solve_cubic_eq_l699_699444

-- Problem 1: 4x^2 - 9 = 0 implies x = ± 3/2
theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2 :=
by sorry

-- Problem 2: 64 * (x + 1)^3 = -125 implies x = -9/4
theorem solve_cubic_eq (x : ℝ) : 64 * (x + 1)^3 = -125 ↔ x = -9/4 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l699_699444


namespace ratio_of_cream_max_to_maxine_l699_699837

def ounces_of_cream_in_max (coffee_sipped : ℕ) (cream_added: ℕ) : ℕ := cream_added

def ounces_of_remaining_cream_in_maxine (initial_coffee : ℚ) (cream_added: ℚ) (sipped : ℚ) : ℚ :=
  let total_mixture := initial_coffee + cream_added
  let remaining_mixture := total_mixture - sipped
  (initial_coffee / total_mixture) * cream_added

theorem ratio_of_cream_max_to_maxine :
  let max_cream := ounces_of_cream_in_max 4 3
  let maxine_cream := ounces_of_remaining_cream_in_maxine 16 3 5
  (max_cream : ℚ) / maxine_cream = 19 / 14 := by 
  sorry

end ratio_of_cream_max_to_maxine_l699_699837


namespace find_c_l699_699276

-- Define the polynomial equation and conditions
noncomputable def poly (c d x : ℝ) := 4 * x^3 + 5 * c * x^2 + 3 * d * x + c

theorem find_c {c d : ℝ} (h_roots : ∃ (p q r : ℝ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  poly c d p = 0 ∧ poly c d q = 0 ∧ poly c d r = 0)
  (h_sum_log : ∀ (p q r : ℝ), (poly c d p = 0 ∧ poly c d q = 0 ∧ poly c d r = 0) → (log 3 p + log 3 q + log 3 r = 3)) :
  c = -108 :=
sorry

end find_c_l699_699276


namespace percentage_sum_l699_699703

theorem percentage_sum {A B : ℝ} 
  (hA : 0.40 * A = 160) 
  (hB : (2/3) * B = 160) : 
  0.60 * (A + B) = 384 :=
by
  sorry

end percentage_sum_l699_699703


namespace card_statements_l699_699965

def statement_A : Prop := (count true statements = 1)
def statement_B : Prop := (count true statements = 2)
def statement_C : Prop := (count true statements = 3)
def statement_D : Prop := (count true statements = 4)

theorem card_statements :
  (statement_A → ¬ statement_B ∧ ¬ statement_C ∧ ¬ statement_D)
  ∧ (statement_B → ¬ statement_A ∧ ¬ statement_C ∧ ¬ statement_D)
  ∧ (statement_C → ¬ statement_A ∧ ¬ statement_B ∧ ¬ statement_D)
  ∧ (statement_D → ¬ statement_A ∧ ¬ statement_B ∧ ¬ statement_C)
  →
  statement_A := 
by 
  sorry

end card_statements_l699_699965


namespace exists_m_n_l699_699610

theorem exists_m_n (a b : ℕ) : 
  ∃ (m n : ℕ), (-2 * a^n * b^n)^m + (3 * a^m * b^m)^n = a^6 * b^6 := sorry

end exists_m_n_l699_699610


namespace right_triangle_area_l699_699921

theorem right_triangle_area (a b c : ℕ) (h_right : c^2 = a^2 + b^2) (h_a : a = 40) (h_c : c = 41) : 
  1 / 2 * a * b = 180 := by
  have h_b_squared : b^2 = 81 :=
  calc
    b^2 = c^2 - a^2 : by rw [←h_right]
    ... = 41^2 - 40^2 : by rw [h_a, h_c]
    ... = 1681 - 1600 : rfl
    ... = 81 : rfl,
  have h_b : b = 9 := by 
    rw [eq_comm, Nat.pow_two, Nat.pow_eq_iff_eq] at h_b_squared,
    exact h_b_squared,
  simp [h_a, h_b],
  sorry

end right_triangle_area_l699_699921


namespace alpha_beta_solution_l699_699631

theorem alpha_beta_solution (n k : ℤ) :
  (∀ n : ℤ, α = ± π / 6 + 2 * π * n → β = ± π / 4 + 2 * π * k) ∨ 
  (∀ n : ℤ, α = ± π / 4 + 2 * π * n → β = ± π / 6 + 2 * π * k) := 
by
  sorry

end alpha_beta_solution_l699_699631


namespace abc_arithmetic_sequence_abc_not_geometric_sequence_l699_699280

noncomputable def a : ℝ := Real.log (3) / Real.log (4)
noncomputable def b : ℝ := Real.log (6) / Real.log (4)
noncomputable def c : ℝ := Real.log (12) / Real.log (4)

-- Proving the arithmetic sequence part
theorem abc_arithmetic_sequence : 2 * b = a + c := by
  calc
    2 * b = 2 * (Real.log 6 / Real.log 4)         : rfl
        ... = (Real.log 6 ^ 2) / Real.log 4       : by rw [mul_div_cancel (Real.log 6) (Real.log_ne_zero.2 (by norm_num [Real.log, Real.logf]))]
        ... = (Real.log (3 * 12)) / Real.log 4    : by rw [Real.log_mul (by norm_num) (by norm_num)]
        ... = (Real.log 3 + Real.log 12) / Real.log 4 : by rw [← Real.log_mul (by norm_num : 3 ≠ 1) (by norm_num : 12 ≠ 1)]
        ... = a + c                               : by rw [a, c, add_div]

-- Proving it's not a geometric sequence
theorem abc_not_geometric_sequence : ¬ ∃ r : ℝ, b = r * a ∧ c = r * b := by
  intro h
  cases h with r hr
  cases hr with hr1 hr2
  calc
    b = Real.log 6 / Real.log 4 := rfl
    ... = Real.log 6 / Real.log 4 * (Real.log 3 / Real.log 4) : by sorry -- Remaining steps rely on contradiction exploring logs property

end abc_arithmetic_sequence_abc_not_geometric_sequence_l699_699280


namespace ratio_of_stock_values_l699_699836

/-- Definitions and conditions -/
def value_expensive := 78
def shares_expensive := 14
def shares_other := 26
def total_assets := 2106

/-- The proof problem -/
theorem ratio_of_stock_values : 
  ∃ (V_other : ℝ), 26 * V_other = total_assets - (shares_expensive * value_expensive) ∧ 
  (value_expensive / V_other) = 2 :=
by
  sorry

end ratio_of_stock_values_l699_699836


namespace chocolate_orders_l699_699438

theorem chocolate_orders (v c : ℕ) (h1 : v + c = 220) (h2 : v = 0.20 * 220) (h3 : v = 2 * c) : 
  c = 22 := by
  sorry

end chocolate_orders_l699_699438


namespace probability_of_odd_roll_l699_699940

theorem probability_of_odd_roll : 
  let n := 6 in 
  let m := 3 in 
  let p := (m : ℚ) / n in 
  p = 1 / 2 :=
by 
  sorry

end probability_of_odd_roll_l699_699940


namespace katy_brownies_total_l699_699394

theorem katy_brownies_total : 
  (let monday_brownies := 5 in
   let tuesday_brownies := 2 * monday_brownies in
   let total_brownies := monday_brownies + tuesday_brownies in
   total_brownies = 15) := 
by 
  let monday_brownies := 5 in
  let tuesday_brownies := 2 * monday_brownies in
  let total_brownies := monday_brownies + tuesday_brownies in
  show total_brownies = 15 by
  sorry

end katy_brownies_total_l699_699394


namespace product_of_a_values_l699_699124

theorem product_of_a_values :
  let a1 := 0
  let a2 := 48 / 13
  ((3 * a - 6)^2 + (2 * a - 3)^2 = 45) → 
  (a * (13 * a - 48) = 0 → a = 0 ∨ a = 48 / 13) → 
  (a1 * a2 = 0) :=
by
  intro a1 a2 h_dist h_values
  have h_prod : a1 * a2 = 0 := by
    calc
      a1 * a2 = 0 * (48 / 13) : by rw [a1]
            ... = 0 : by rw [zero_mul]
  exact h_prod

end product_of_a_values_l699_699124


namespace monotonic_increasing_range_l699_699777

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699777


namespace area_of_quadrilateral_l699_699636

noncomputable def quadratic_solution (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b * b - 4 * a * c in
  ((-b + real.sqrt discriminant) / (2 * a), (-b - real.sqrt discriminant) / (2 * a))

theorem area_of_quadrilateral :
  ∃ (A B C D : point) (r : ℝ), 
    r = 1 → 
    (dist A B = 2) ∧ (dist B C = 2) ∧ (dist C D = 2) ∧ (dist D A = 2) ∧
    (dist A C = dist B D ∨ dist A D = dist B C) →
  (area A B C D = 4 ∨ area A B C D = 3) :=
by
  sorry

end area_of_quadrilateral_l699_699636


namespace hyperbola_standard_eqn_line_eqn_l699_699664

-- Conditions
def ellipse_major_axis_ends : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}
def hyperbola_eccentricity : ℝ := 3 / 2
def midpoint_x_coordinate : ℝ := 4 * Real.sqrt (2)

-- Standard equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 
  (x^2) / 4 - (y^2) / 5 = 1

-- Equation of the line l
def line_equation (t : ℝ) (x y : ℝ) : Prop := 
  y = x + t

theorem hyperbola_standard_eqn :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ (x^2 / 4 - y^2 / 5 = 1) :=
sorry

theorem line_eqn :
  ∃ t : ℝ, midpoint_x_coordinate = 4 * Real.sqrt (2) →
  (∀ (x y : ℝ), line_equation t x y ↔ x - y + Real.sqrt (2) = 0) :=
sorry

end hyperbola_standard_eqn_line_eqn_l699_699664


namespace range_of_a_l699_699716

theorem range_of_a (a : ℝ) : (∃ x₀ ∈ set.Icc (0 : ℝ) 1, 2^x₀ * (3 * x₀ + a) < 1) → a < 1 := 
by 
  sorry

end range_of_a_l699_699716


namespace factorization_result_l699_699126

theorem factorization_result (a b : ℤ) (h : (16:ℚ) * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) : a + 2 * b = -23 := by
  sorry

end factorization_result_l699_699126


namespace hyperbola_eccentricity_l699_699312

open Real

-- Definition of ellipse C1
def C1 (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- The problem to prove the eccentricity of hyperbola C2
theorem hyperbola_eccentricity (h_same_foci : ∀ x y, C1 x y → (x^2 + y^2 = 1)) :
    ∃ c, √2 = c := 
by
  sorry

end hyperbola_eccentricity_l699_699312


namespace max_additional_packages_l699_699838

-- Define the basic constants and conditions
def max_packages_per_day : ℕ := 35
def max_days_with_max_packages : ℕ := 3
def days_with_other_packages : ℕ := 3
def total_other_packages : ℕ := 67
def two_fifths_max_packages : ℕ := (2 * max_packages_per_day) / 5
def three_fourths_max_packages : ℕ := (3 * max_packages_per_day) / 4
def two_days_sum_three_fourths : ℕ := 2 * three_fourths_max_packages
def half_difference : ℕ := ((max_packages_per_day - three_fourths_max_packages) / 2)

-- Calculate the total packages Max delivered
def total_packages_delivered : ℕ :=
  (max_days_with_max_packages * max_packages_per_day) +
  total_other_packages +
  two_fifths_max_packages +
  two_days_sum_three_fourths +
  half_difference

-- Calculate the maximum possible packages
def max_possible_packages : ℕ := 10 * max_packages_per_day

-- Calculate the difference in packages
def additional_packages : ℕ := max_possible_packages - total_packages_delivered

-- Statement to prove
theorem max_additional_packages (h : additional_packages = 108) : h = 108 := by
  sorry

end max_additional_packages_l699_699838


namespace soda_cost_l699_699934

theorem soda_cost (total_cost sandwich_price : ℝ) (num_sandwiches num_sodas : ℕ) (total : total_cost = 8.38)
  (sandwich_cost : sandwich_price = 2.45) (total_sandwiches : num_sandwiches = 2) (total_sodas : num_sodas = 4) :
  ((total_cost - (num_sandwiches * sandwich_price)) / num_sodas) = 0.87 :=
by
  sorry

end soda_cost_l699_699934


namespace calc_expr_l699_699225

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l699_699225


namespace probability_point_closer_to_D_l699_699738

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2;
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem probability_point_closer_to_D
  (A B C : ℝ)
  (h1 : A = 6)
  (h2 : B = 8)
  (h3 : C = 10) :
  let DEF_area := area_of_triangle A B C in
  let small_triangle_area := DEF_area / 4 in
  (2 * small_triangle_area) / DEF_area = 1 / 2 :=
by {
  sorry
}

end probability_point_closer_to_D_l699_699738


namespace squares_per_row_l699_699073

theorem squares_per_row (x : ℕ) (h1 : 4 * 6 = 24)
  (h2 : ∀ x, 4 * x = 4 * x) (h3 : 66 = 66) 
  (h4 : 10 * x = 10 * x) : x = 15 :=
by
  have eq1 : 24 + 4 * x + 66 = 10 * x,
  { rw [h1, add_assoc],
    exact add_assoc _ _ _ },
  have eq2 : 24 + 4 * x + 66 = 90 + 4 * x,
  { exact congr_arg2 (+) rfl.symm rfl },
  have eq3 : 90 + 4 * x = 10 * x,
  { exact add_comm 66 24,
    exact add_comm 4 6 },
  exact add_comm x x

#print axioms squares_per_row

end squares_per_row_l699_699073


namespace range_of_a_l699_699788

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699788


namespace johns_weekly_allowance_l699_699334

-- Define the conditions
variables (A : ℝ) (remaining_after_arcade : ℝ) (spent_at_toy_store : ℝ) (remaining_after_toy_store : ℝ)
variables (final_spent : ℝ)

-- Conditions in Lean
def johns_weekly_allowance_conditions (A : ℝ) : Prop :=
  (remaining_after_arcade = (2 / 5) * A) ∧
  (spent_at_toy_store = (1 / 3) * remaining_after_arcade) ∧
  (remaining_after_toy_store = remaining_after_arcade - spent_at_toy_store) ∧
  (final_spent = 1.28) ∧
  (remaining_after_toy_store = final_spent)

-- Goal
theorem johns_weekly_allowance (A : ℝ) : johns_weekly_allowance_conditions A → A = 4.80 :=
by
  sorry

end johns_weekly_allowance_l699_699334


namespace friend_selling_price_l699_699975

noncomputable def original_cost_price : ℝ := 52941.17647058824
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

theorem friend_selling_price :
  let loss_amount := loss_percentage * original_cost_price in
  let sp_man := original_cost_price - loss_amount in
  let gain_amount := gain_percentage * sp_man in
  let sp_friend := sp_man + gain_amount in
  sp_friend = 54000 := by
  sorry

end friend_selling_price_l699_699975


namespace find_range_of_a_l699_699958

variable (x a : ℝ)

/-- Given p: 2 * x^2 - 9 * x + a < 0 and q: the negation of p is sufficient 
condition for the negation of q,
prove to find the range of the real number a. -/
theorem find_range_of_a (hp: 2 * x^2 - 9 * x + a < 0) (hq: ¬ (2 * x^2 - 9 * x + a < 0) → ¬ q) :
  ∃ a : ℝ, sorry := sorry

end find_range_of_a_l699_699958


namespace find_a_value_l699_699685

theorem find_a_value (a : ℝ) (h : ∀ x : ℝ, ax^2 + (a - 1) * x - 1 > 0 -> x ∈ Ioo (-1 : ℝ) (-1 / 2)) : a = -2 := 
sorry

end find_a_value_l699_699685


namespace simple_interest_rate_l699_699523

theorem simple_interest_rate (P I : ℕ) (hP : P = 1200) (hI : I = 108) (R : ℝ) :
  I = P * R * R / 100 → R = 3 :=
by
  intro h
  rw [hP, hI] at h
  have h1 : 108 * 100 = 1200 * (R * R),
  { rw [mul_comm, ←mul_assoc, ←mul_comm 1200, mul_assoc, div_mul_cancel] at h, norm_num at h, rw mul_comm, exact h },
  have h2 : 10800 = 1200 * R^2,
  { norm_num, exact h1 },
  have h3 : 10800 / 1200 = R^2,
  { rw ← h2, norm_num },
  have h4 : 9 = R^2,
  { norm_num, exact h3 },
  exact pow_eq_pow h4

end simple_interest_rate_l699_699523


namespace sum_of_digits_of_square_l699_699164

theorem sum_of_digits_of_square (n : ℤ) (h : n = 1111111) : 
  ∃ d : ℕ, d = 235 ∧ nat.digits 10 (n * n) = d :=
by
  use 235
  sorry

end sum_of_digits_of_square_l699_699164


namespace tatiana_age_full_years_l699_699449

theorem tatiana_age_full_years :
  let years := 72
  let months := 72
  let weeks := 72
  let days := 72
  let hours := 72
  (years + months / 12 + weeks / 52 + days / 365 + hours / (24 * 365)).toInt = 79 :=
by
  sorry

end tatiana_age_full_years_l699_699449


namespace concyclic_points_l699_699405

open_locale geometry

-- Define the given problem
variables (Γ : Type*) [metric_space Γ] [normed_group Γ] [normed_space ℝ Γ]
variables {BC D E F G : Γ} (A : Γ)
variables [circle Γ (Γ × ℝ)]   -- this assumes Γ is a metric space and a circle
variables (D E F G : Γ)
variables (AD AE : ↥(circle Γ))

-- Given conditions
variable (H1 : circle Γ BC)
variable (H2 : A = midpoint_arc BC)
variable (H3 : chord_through A AD)
variable (H4 : chord_through A AE)
variable (H5 : F = intersection_chord AD BC)
variable (H6 : G = intersection_chord AE BC)

-- Goal
theorem concyclic_points : concyclic {D, E, F, G} :=
sorry

end concyclic_points_l699_699405


namespace largest_possible_sum_of_digits_l699_699545

/--
  A digital clock displays hours, minutes, and seconds in a 12-hour format with AM and PM.
  The hours range from 01 to 12, minutes and seconds range from 00 to 59.
  The largest possible sum of the digits in this display is 37.
-/
theorem largest_possible_sum_of_digits :
  let hours := {n // 1 ≤ n ∧ n ≤ 12},
      minutes := {n // 0 ≤ n ∧ n < 60},
      seconds := {n // 0 ≤ n ∧ n < 60},
      digit_sum (n : ℕ) : ℕ := n / 10 + n % 10 
  in
  ∀ (h : hours) (m : minutes) (s : seconds),
    (digit_sum h.val + digit_sum m.val + digit_sum s.val ≤ 37) ∧ 
    ∃ (h : hours) (m : minutes) (s : seconds),
    digit_sum h.val + digit_sum m.val + digit_sum s.val = 37 :=
by sorry

end largest_possible_sum_of_digits_l699_699545


namespace dinosaur_book_cost_l699_699614

theorem dinosaur_book_cost (dictionary_cost cookbook_cost total_cost : ℕ) 
  (h1: dictionary_cost = 5) 
  (h2: cookbook_cost = 5) 
  (h3: total_cost = 21) : 
  total_cost - (dictionary_cost + cookbook_cost) = 11 := 
by
  rw [h1, h2, h3]
  norm_num

end dinosaur_book_cost_l699_699614


namespace seq_natural_product_plus_one_is_square_l699_699137

noncomputable def seq (n : ℕ) : ℕ 
| 0       := 0
| (n + 1) := 1/2 * (3 * seq n + (Real.sqrt(5 * (seq n)^2 + 4)))

theorem seq_natural (n : ℕ) : Nat :=
begin
  sorry
end

theorem product_plus_one_is_square (n : ℕ) : ∃ k : ℕ, seq n * seq (n+1) + 1 = k^2 ∧
                                                    seq (n+1) * seq (n+2) + 1 = k^2 ∧
                                                    seq n * seq (n+2) + 1 = k^2 :=
begin
  sorry
end

end seq_natural_product_plus_one_is_square_l699_699137


namespace harold_august_tips_fraction_l699_699945

noncomputable def tips_fraction : ℚ :=
  let A : ℚ := sorry -- average monthly tips for March to July and September
  let august_tips := 6 * A -- Tips for August
  let total_tips := 6 * A + 6 * A -- Total tips for all months worked
  august_tips / total_tips

theorem harold_august_tips_fraction :
  tips_fraction = 1 / 2 :=
by
  sorry

end harold_august_tips_fraction_l699_699945


namespace intervals_of_monotonic_increase_max_area_acute_triangle_l699_699069

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (sin x, (sqrt 3 / 2) * (sin x - cos x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, sin x + cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

-- Problem 1: Proving the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  ∀ x₁ x₂ : ℝ, (k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12) → f x₁ ≤ f x₂ :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC
theorem max_area_acute_triangle (A : ℝ) (a b c : ℝ) :
  (f A = 1 / 2) → (a = sqrt 2) →
  ∀ S : ℝ, S ≤ (1 + sqrt 2) / 2 :=
sorry

end intervals_of_monotonic_increase_max_area_acute_triangle_l699_699069


namespace men_required_l699_699538

theorem men_required (W M : ℕ) (h1 : M * 20 * W = W) (h2 : (M - 4) * 25 * W = W) : M = 16 := by
  sorry

end men_required_l699_699538


namespace selection_properties_l699_699821

theorem selection_properties (n : ℕ) (s : Finset ℕ) (h : s.card = n + 1 ∧ ∀ x ∈ s, x ≤ 2 * n):
  ∃ a b ∈ s, Nat.coprime a b ∧ ∃ u v ∈ s, u ≠ v ∧ (u % v = 0 ∨ v % u = 0) :=
by
  sorry

end selection_properties_l699_699821


namespace N_coincide_N1_MN_passes_fixed_point_l699_699687

-- Define the conditions
variable (A B M : Point)
variable (AMCD MBEF : Square) (hAM : AMCD.base = AM) (hMB : MBEF.base = MB)
variable (circumcircleAMCD circumcircleMBEF : Circle) 
variable (hAMcirc : AMCD.circumcircle = circumcircleAMCD) 
variable (hMBcirc : MBEF.circumcircle = circumcircleMBEF)
variable (N N1: Point) (hN: N ≠ M) (hNInt: circumcircleAMCD ∩ circumcircleMBEF = {M, N}) 
variable (BC AF : Line) 
variable (hBC : BC ∈ linesFrom B ∩ linesFrom C)
variable (hAF : AF ∈ linesFrom A ∩ linesFrom F)
variable (hN1 : BC ∩ AF = {N1 : Point})
variable (MN: Line) (hMN: MN ∈ linesFrom M ∩ linesFrom N)

-- Prove that N coincides with N1
theorem N_coincide_N1 : N = N1 :=
by
  sorry 

-- Prove that MN passes through a fixed point P as M moves
theorem MN_passes_fixed_point (P: Point) : ∃ P, ∀ M, lineThroughMAndN M N ⊇ {P} :=
by
  sorry

end N_coincide_N1_MN_passes_fixed_point_l699_699687


namespace no_integer_areas_sum_twenty_nineteen_l699_699075

noncomputable def set_of_points : Type := Finset (EuclideanSpace ℝ (Fin 2))

def all_integer_areas (points : Finset (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ (p1 p2 p3 : (EuclideanSpace ℝ (Fin 2))), 
    {p1, p2, p3} ⊆ points → 
    ∃ (area : ℤ), area_of_triangle p1 p2 p3 = area

def sum_of_areas (points : Finset (EuclideanSpace ℝ (Fin 2))) : ℤ :=
  ∑ (p1 p2 p3 : (EuclideanSpace ℝ (Fin 2))) in points.triangle_combinations, area_of_triangle p1 p2 p3

theorem no_integer_areas_sum_twenty_nineteen (points : Finset (EuclideanSpace ℝ (Fin 2))) 
  (h : points.card = 6) 
  (h1 : all_integer_areas points) 
  : ¬ (sum_of_areas points = 2019) := 
sorry

end no_integer_areas_sum_twenty_nineteen_l699_699075


namespace l_shape_area_is_42_l699_699187

-- Defining the dimensions of the larger rectangle
def large_rect_length : ℕ := 10
def large_rect_width : ℕ := 7

-- Defining the smaller rectangle dimensions based on the given conditions
def small_rect_length : ℕ := large_rect_length - 3
def small_rect_width : ℕ := large_rect_width - 3

-- Defining the areas of the rectangles
def large_rect_area : ℕ := large_rect_length * large_rect_width
def small_rect_area : ℕ := small_rect_length * small_rect_width

-- Defining the area of the "L" shape
def l_shape_area : ℕ := large_rect_area - small_rect_area

-- The theorem to prove
theorem l_shape_area_is_42 : l_shape_area = 42 :=
by
  sorry

end l_shape_area_is_42_l699_699187


namespace ball_falls_into_top_left_l699_699980

def billiard_table := (26, 1965)
def bottom_left := (0, 0)
def top_left := (0, 26)
def shot_angle := 45

theorem ball_falls_into_top_left :
  ∀ (billiard_table : (ℕ × ℕ)) (bottom_left top_left : (ℕ × ℕ)) (shot_angle : ℝ), 
  billiard_table = (26, 1965) ∧ bottom_left = (0, 0) ∧ top_left = (0, 26) ∧ shot_angle = 45 
  → (⟹ ball will fall into top_left pocket after several reflections) :=
begin
  sorry
end

end ball_falls_into_top_left_l699_699980


namespace dive_point_value_l699_699021

def dive_scores : List ℝ := [7.5, 8.3, 9.0, 6.0, 8.6]
def difficulty : ℝ := 3.2

theorem dive_point_value :
  let scores_dropped := dive_scores.erase (dive_scores.maximum).get! in
  let final_scores := scores_dropped.erase (scores_dropped.minimum).get! in
  let sum_scores := final_scores.sum in
  let point_value := sum_scores * difficulty in
  point_value = 78.08 :=
by {
  have h_dive_scores_drop_max : scores_dropped = [7.5, 8.3, 6.0, 8.6],
  {simp [dive_scores]},        -- remove 9.0 (maximum)
  have h_dive_scores_drop_both : final_scores = [7.5, 8.3, 8.6],
  {simp [h_dive_scores_drop_max]}, -- remove 6.0 (minimum)
  have h_sum_scores : sum_scores = 24.4,
  {simp [h_dive_scores_drop_both]}, -- sum 7.5 + 8.3 + 8.6
  have h_point_value: point_value = 78.08,
  {simp [h_sum_scores]},
  exact h_point_value
}

end dive_point_value_l699_699021


namespace tom_climbing_time_l699_699494

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end tom_climbing_time_l699_699494


namespace sphere_volume_ratio_l699_699898

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * (Real.pi) * (r ^ 3)

theorem sphere_volume_ratio (a : ℝ) (h : 0 < a) :
  let r1 := a / 2
  let r2 := (a * Real.sqrt 2) / 2
  let r3 := (a * Real.sqrt 3) / 2
  (volume r1) : (volume r2) : (volume r3) = 1 : 2 * Real.sqrt 2 : 3 * Real.sqrt 3 :=
by
  let r1 := a / 2
  let r2 := (a * Real.sqrt 2) / 2
  let r3 := (a * Real.sqrt 3) / 2
  have V1 : volume r1 = (4 / 3) * Real.pi * (r1^3) := by sorry
  have V2 : volume r2 = (4 / 3) * Real.pi * (r2^3) := by sorry
  have V3 : volume r3 = (4 / 3) * Real.pi * (r3^3) := by sorry
  have ratio : (volume r1) : (volume r2) : (volume r3) =
                (r1^3) : (r2^3) : (r3^3) := by sorry
  have ratio_simplified : (r1^3) : (r2^3) : (r3^3) = 1 : 2 * Real.sqrt 2 : 3 * Real.sqrt 3 := by sorry
  exact ratio_simplified

end sphere_volume_ratio_l699_699898


namespace centroid_inverse_square_sum_l699_699050

theorem centroid_inverse_square_sum
  (α β γ p q r : ℝ)
  (h1 : 1/α^2 + 1/β^2 + 1/γ^2 = 1)
  (hp : p = α / 3)
  (hq : q = β / 3)
  (hr : r = γ / 3) :
  (1/p^2 + 1/q^2 + 1/r^2 = 9) :=
sorry

end centroid_inverse_square_sum_l699_699050


namespace calc_expr_l699_699226

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l699_699226


namespace number_of_zeros_among_50_numbers_l699_699286

theorem number_of_zeros_among_50_numbers :
  ∀ (m n p : ℕ), (m + n + p = 50) → (m * p = 500) → n = 5 :=
by
  intros m n p h1 h2
  sorry

end number_of_zeros_among_50_numbers_l699_699286


namespace math_problem_l699_699861

noncomputable def problem_statement (α : ℝ) : Prop :=
  (2 * tan (real.pi / 4 - α)) / (1 - tan (real.pi / 4 - α) ^ 2) * (sin α * cos α) / (cos α ^ 2 - sin α ^ 2) = 1 / 2

theorem math_problem (α : ℝ) : problem_statement α := 
  by 
  sorry

end math_problem_l699_699861


namespace complex_multiplication_l699_699620

-- Define i such that i^2 = -1
def i : ℂ := Complex.I

theorem complex_multiplication : (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i := by
  sorry

end complex_multiplication_l699_699620


namespace right_triangle_correct_option_l699_699513

def triangle_option (BC AC AB : ℝ) : Prop := BC^2 + AC^2 = AB^2

def triangle_ratio_option (BC_ratio AC_ratio AB_ratio : ℝ) : Prop := 
  (BC_ratio^2 + AC_ratio^2 = AB_ratio^2) ∧ (BC_ratio = 3 ∧ AC_ratio = 4 ∧ AB_ratio = 5)

def angle_ratio_not_right_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  let total_angles := angle_A + angle_B + angle_C in
  (angle_A + angle_B + angle_C = 180) ∧ (90 < angle_C) ∧ (angle_C = 180 * (5 / (3 + 4 + 5)))

theorem right_triangle_correct_option :
  triangle_option 2 3 4 = false ∧
  triangle_option 2 3 3 = false ∧
  triangle_ratio_option 3 4 5 = true ∧
  angle_ratio_not_right_triangle (3/12*180) (4/12*180) (5/12*180) = false :=
by
  sorry

end right_triangle_correct_option_l699_699513


namespace clarks_number_is_23_l699_699578

-- Defining the set of prime numbers less than 100
def primes_less_than_100 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

-- Function to check if a number is prime and less than 100
def is_prime_and_less_than_100 (n : ℕ) : Prop := n ∈ primes_less_than_100

-- Main theorem statement
theorem clarks_number_is_23 : 
  ∃ (n : ℕ), is_prime_and_less_than_100 n ∧ 
  (∃ (d1 d2 : ℕ), n = d1 * 10 + d2 ∧ 
  is_prime_and_less_than_100 n ∧ 
  (d1 ≠ d2) ∧ 
  (∀ (x : ℕ), x ∈ primes_less_than_100 → 
    (∃ (a b : ℕ), x = a * 10 + b → 
    (a ≠ d1 ∧ b ≠ d2) ∨ (a ≠ d2 ∧ b ≠ d1)) ∧
    ∃ (y : ℕ), x = y)) := sorry
  
end clarks_number_is_23_l699_699578


namespace monotonic_increasing_range_l699_699776

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699776


namespace total_capacity_is_correct_l699_699478

-- Define small and large jars capacities
def small_jar_capacity : ℕ := 3
def large_jar_capacity : ℕ := 5

-- Define the total number of jars and the number of small jars
def total_jars : ℕ := 100
def small_jars : ℕ := 62

-- Define the number of large jars based on the total jars and small jars
def large_jars : ℕ := total_jars - small_jars

-- Calculate capacities
def small_jars_total_capacity : ℕ := small_jars * small_jar_capacity
def large_jars_total_capacity : ℕ := large_jars * large_jar_capacity

-- Define the total capacity
def total_capacity : ℕ := small_jars_total_capacity + large_jars_total_capacity

-- Prove that the total capacity is 376 liters
theorem total_capacity_is_correct : total_capacity = 376 := by
  sorry

end total_capacity_is_correct_l699_699478


namespace monotonically_increasing_range_l699_699771

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699771


namespace proof_2LN_eq_AC_l699_699947

namespace GeometryLean

open_locale euclidean_geometry

variables {A B C N S L M : Point}

theorem proof_2LN_eq_AC
  (hAB_gt_BC : (AB ≠ BC))
  (h2BN_eq_AB_plus_BC : 2 * (dist B N) = (dist A B) + (dist B C))
  (hBS_bisector : is_angle_bisector B S (triangle A B C))
  (hM_midpoint : midpoint AC M)
  (hML_parallel_AB : parallel (line M L) (line A B))
  : 2 * (dist L N) = dist A C := sorry

end GeometryLean

end proof_2LN_eq_AC_l699_699947


namespace math_problem_l699_699849

theorem math_problem (n : ℕ) (a : Fin n → ℝ) 
  (hn : 0 < n) 
  (ha_pos : ∀ i, 0 < a i) 
  (ha_prod : ∏ i, a i = 1) : 
  (∑ i, a i / Real.sqrt (a i ^ 4 + 3)) ≤ (1 / 2) * ∑ i, 1 / a i :=
by
  sorry

end math_problem_l699_699849


namespace cube_plus_eleven_mul_divisible_by_six_l699_699442

theorem cube_plus_eleven_mul_divisible_by_six (a : ℤ) : 6 ∣ (a^3 + 11 * a) := 
by sorry

end cube_plus_eleven_mul_divisible_by_six_l699_699442


namespace polygon_vertices_l699_699896

theorem polygon_vertices 
  (n : ℕ) 
  (k : ℕ) 
  (h : k > 56)
  (h57 : n = 57) 
  (h_poly : ∀ i, i ≤ k → is_polygon_with_n_gons (i) n) : 
  False :=
by
  sorry

end polygon_vertices_l699_699896


namespace square_area_l699_699375

/-- Given:
  - PQRS is a square.
  - M is the midpoint of PQ.
  - Area of the triangle MQR is 100.

  Prove that the area of the square PQRS is 400.
-/
theorem square_area (PQRS : Type) [square PQRS] (M Q R P S : PQRS) 
  (mid_M : M = midpoint P Q) 
  (triangle_area_100 : area (triangle M Q R) = 100) :
  area (square PQRS) = 400 :=
sorry

end square_area_l699_699375


namespace katy_brownies_l699_699397

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l699_699397


namespace smallest_palindrome_x_l699_699160

-- Define what a palindrome is
def is_palindrome (n : Nat) : Prop :=
  let s := n.toString
  s == s.reverse

-- Condition: x must be positive
def positive (x : Nat) : Prop := x > 0

-- Prove that the smallest positive x such that x + 7890 is a palindrome is 338
theorem smallest_palindrome_x :
  ∃ (x : Nat), positive x ∧ is_palindrome (x + 7890) ∧ ∀ y, positive y ∧ is_palindrome (y + 7890) → x ≤ y :=
  Exists.intro 338
    (and.intro
      (Nat.zero_lt_succ 337) -- 338 > 0
      (and.intro
        (by
          let x := 338
          have h : x + 7890 = 8228 := rfl
          show is_palindrome 8228
          calc
            8228.toString = "8228" : rfl
            "8228".reverse = "8228" : rfl
        sorry) -- Proof of smallest x
  )

end smallest_palindrome_x_l699_699160


namespace integral_sqrt_minus_inverse_sqrt_l699_699009

noncomputable def a : ℝ := 2

theorem integral_sqrt_minus_inverse_sqrt :
    (\int x in 1..a, (sqrt x - (1 / x))) = (4 * sqrt 2 - 2) / 3 - log 2 := by
  sorry

end integral_sqrt_minus_inverse_sqrt_l699_699009


namespace window_total_width_l699_699245

theorem window_total_width 
  (panes : Nat := 6)
  (ratio_height_width : ℤ := 3)
  (border_width : ℤ := 1)
  (rows : Nat := 2)
  (columns : Nat := 3)
  (pane_width : ℤ := 12) :
  3 * pane_width + 2 * border_width + 2 * border_width = 40 := 
by
  sorry

end window_total_width_l699_699245


namespace average_of_next_seven_consecutive_integers_l699_699441

theorem average_of_next_seven_consecutive_integers
  (a b : ℕ)
  (hb : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) :
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7) = a + 6 :=
by
  sorry

end average_of_next_seven_consecutive_integers_l699_699441


namespace proof_mn_squared_l699_699641

theorem proof_mn_squared (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end proof_mn_squared_l699_699641


namespace right_triangle_area_l699_699918

theorem right_triangle_area (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : a = 40) (hc : c = 41) : 
  1 / 2 * a * √(c^2 - a^2) = 180 :=
by
  sorry

end right_triangle_area_l699_699918


namespace find_AO_length_l699_699910

theorem find_AO_length
  {A B C X Y O D : Type*}
  (h : angle A B C = 90)
  (h1 : dist B A = 3)
  (h2 : dist C A = 4)
  (h3 : dist A X = 9 / 4)
  (XY_midpoint_O : midpoint A B X Y O)
  (XY_tangent_point : tangent_point B C X Y D)
  : dist A O = 39 / 32 :=
sorry

end find_AO_length_l699_699910


namespace tom_climbing_time_l699_699493

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end tom_climbing_time_l699_699493


namespace dual_cassette_recorder_price_l699_699567

theorem dual_cassette_recorder_price :
  ∃ (x y : ℝ),
    (x - 0.05 * x = 380) ∧
    (y = x + 0.08 * x) ∧ 
    (y = 432) :=
by
  -- sorry to skip the proof.
  sorry

end dual_cassette_recorder_price_l699_699567


namespace expression_evaluation_l699_699221

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l699_699221


namespace CF_perpendicular_to_AB_l699_699373

variables {A B C P Q F : Point}
variables (hABC : is_acute_triangle A B C)
          (hAB_diameter : is_diameter (circle_through A B C) A B)
          (hCircle_intersects_AC : circle_intersects_side (circle_through A B C) A C P)
          (hCircle_intersects_BC : circle_intersects_side (circle_through A B C) B C Q)
          (hTangents_intersect : tangents_intersect (circle_through A B C) P Q F)
          (hTangents : tangent (circle_through A B C) P ∧ tangent (circle_through A B C) Q)

theorem CF_perpendicular_to_AB :
  is_perpendicular F C F B := sorry

end CF_perpendicular_to_AB_l699_699373


namespace find_ellipse_area_l699_699654

noncomputable def ellipse_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
(∃ x y : ℝ, (x = real.sqrt 6 ∧ y = real.sqrt 2 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (real.sqrt(a^2 - b^2) / a = real.sqrt 6 / 3)))

noncomputable def area_triangle (a b : ℝ) (ha : a^2 = 12) (hb : b^2 = 4) : ℝ :=
let d := real.abs ((-3 - -(-2)) / real.sqrt 2) in
let ab := real.sqrt (2 * (((-3) + 2)^2 - 4 * 0)) in
(1 / 2) * d * ab

theorem find_ellipse_area (a b : ℝ) (ha : a^2 = 12) (hb : b^2 = 4)
  (ellipse_exists : ellipse_equation a b (sqrt_pos.mp ha) (sqrt_pos.mp hb)) :
  area_triangle a b ha hb = 9 / 2 :=
-- Proof would go here, but we insert sorry for now.
sorry

end find_ellipse_area_l699_699654


namespace incorrect_option_A_l699_699512

-- Definitions of the conditions based on given problem statements.
def regression_line (x : ℝ) : ℝ := 0.5 * x - 85
def is_incorrect_option_A : Prop := ¬∀ x = 200, regression_line x = 15
-- Note: For simplicity, the detailed definitions and conditions of B, C, and D are omitted as they are not required for proving A is incorrect.

-- Target lean statement proving that option A is incorrect.
theorem incorrect_option_A : is_incorrect_option_A :=
by
  sorry

end incorrect_option_A_l699_699512


namespace solution_l699_699586

noncomputable def problem : Prop := 
  (\frac{1}{3} * \frac{4}{7} * \frac{9}{13} + \frac{1}{2} = \frac{49}{78})

theorem solution : problem := by
  sorry

end solution_l699_699586


namespace train_distance_travelled_l699_699986

theorem train_distance_travelled 
  (t : ℕ) (v : ℕ) (h_t : t = 6) (h_v : v = 7) : 
  (v * t = 42) :=
  by
    rw [h_t, h_v]
    exact rfl

end train_distance_travelled_l699_699986


namespace alex_correct_percentage_l699_699204

theorem alex_correct_percentage 
  (score_quiz : ℤ) (problems_quiz : ℤ)
  (score_test : ℤ) (problems_test : ℤ)
  (score_exam : ℤ) (problems_exam : ℤ)
  (h1 : score_quiz = 75) (h2 : problems_quiz = 30)
  (h3 : score_test = 85) (h4 : problems_test = 50)
  (h5 : score_exam = 80) (h6 : problems_exam = 20) :
  (75 * 30 + 85 * 50 + 80 * 20) / (30 + 50 + 20) = 81 := 
sorry

end alex_correct_percentage_l699_699204


namespace triangles_from_ten_points_l699_699119

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l699_699119


namespace expr_value_l699_699166

-- Define the constants
def w : ℤ := 3
def x : ℤ := -2
def y : ℤ := 1
def z : ℤ := 4

-- Define the expression
def expr : ℤ := (w^2 * x^2 * y * z) - (w * x^2 * y * z^2) + (w * y^3 * z^2) - (w * y^2 * x * z^4)

-- Statement to be proved
theorem expr_value : expr = 1536 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end expr_value_l699_699166


namespace determine_ABC_l699_699447

noncomputable def digits_are_non_zero_distinct_and_not_larger_than_5 (A B C : ℕ) : Prop :=
  0 < A ∧ A ≤ 5 ∧ 0 < B ∧ B ≤ 5 ∧ 0 < C ∧ C ≤ 5 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def first_condition (A B : ℕ) : Prop :=
  A * 6 + B + A = B * 6 + A -- AB_6 + A_6 = BA_6 condition translated into arithmetics

noncomputable def second_condition (A B C : ℕ) : Prop :=
  A * 6 + B + B = C * 6 + 1 -- AB_6 + B_6 = C1_6 condition translated into arithmetics

theorem determine_ABC (A B C : ℕ) (h1 : digits_are_non_zero_distinct_and_not_larger_than_5 A B C)
    (h2 : first_condition A B) (h3 : second_condition A B C) :
    A * 100 + B * 10 + C = 5 * 100 + 1 * 10 + 5 := -- Final transformation of ABC to 515
  sorry

end determine_ABC_l699_699447


namespace brick_length_is_20_cm_l699_699544

theorem brick_length_is_20_cm
    (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
    (brick_length_cm : ℕ) (brick_width_cm : ℕ)
    (total_bricks_required : ℕ)
    (h1 : courtyard_length_m = 25)
    (h2 : courtyard_width_m = 16)
    (h3 : brick_length_cm = 20)
    (h4 : brick_width_cm = 10)
    (h5 : total_bricks_required = 20000) :
    brick_length_cm = 20 := 
by
    sorry

end brick_length_is_20_cm_l699_699544


namespace equal_product_of_distances_l699_699192

-- Define the setup
variable (circle : Type) [MetricSpace circle]
variable {polygon_outer polygon_inner : Finset circle}
variable {M : circle}

-- Conditions related to the problem
axiom circumscribed_around_circle : polygon_outer.Circumscribed circle
axiom inscribed_in_circle : polygon_inner.Inscribed circle
axiom M_on_circle : M ∈ circle

-- Define the distances from M to the sides of the polygons
def distance_to_sides (polygon : Finset circle) (point : circle) : Finset ℝ :=
  polygon.map (λ vertex, dist point vertex)

-- Statement to prove
theorem equal_product_of_distances :
  ∏ s in distance_to_sides polygon_outer M = ∏ s in distance_to_sides polygon_inner M :=
sorry

end equal_product_of_distances_l699_699192


namespace smallest_positive_x_7890_palindrome_l699_699163

def is_palindrome (n : ℕ) : Prop :=
    let s := n.to_string in
    s = s.reverse

theorem smallest_positive_x_7890_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 7890) ∧ x = 107 :=
by
  sorry

end smallest_positive_x_7890_palindrome_l699_699163


namespace radii_of_original_triangles_equal_l699_699914

theorem radii_of_original_triangles_equal 
  {T1 T2 : Triangle} 
  (H: ∃ Hx : Hexagon, Hx.subdivides T1 T2)
  (equal_radii : ∀ t ∈ Hx.triangles, t.incircle.radius = t'! : real := sorry


end radii_of_original_triangles_equal_l699_699914


namespace number_of_triangles_l699_699114

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l699_699114


namespace math_problem_l699_699593

theorem math_problem :
  (real.sqrt 3 - 1)^0 + (-1 / 3 : ℝ)^(-1 : ℤ) - 2 * real.cos (real.pi / 6) + real.sqrt (1 / 2) * real.sqrt 6 = -2 :=
by {
  -- Step-by-step simplification based on given conditions
  have h1 : (real.sqrt 3 - 1)^0 = 1 := by simp,
  have h2 : (-1 / 3 : ℝ)^(-1 : ℤ) = -3 := by norm_num,
  have h3 : real.cos (real.pi / 6) = real.sqrt 3 / 2 := by norm_num,
  have h4 : real.sqrt (1 / 2) * real.sqrt 6 = real.sqrt (1 / 2 * 6) := by rw real.sqrt_mul',
  rw [h1, h2, h3, h4],
  norm_num,
  ring,
  sorry,
}

end math_problem_l699_699593


namespace evaluate_expression_l699_699615

theorem evaluate_expression : 
  (2 ^ 2003 * 3 ^ 2002 * 5) / (6 ^ 2003) = (5 / 3) :=
by sorry

end evaluate_expression_l699_699615


namespace symmetric_center_of_trig_function_l699_699267

theorem symmetric_center_of_trig_function :
  ∃ x : ℝ, ∃ y : ℝ, (y = sin x - sqrt 3 * cos x + 1) ∧
                 (∀ k : ℤ, x = k * π + π / 3) ∧
                 (y = 1) :=
by
  use π / 3
  use 1
  split
  · simp
    sorry
  · intro k
    field_simp
    linarith

end symmetric_center_of_trig_function_l699_699267


namespace manager_salary_l699_699524

theorem manager_salary :
  let avg_salary_employees := 1500
  let num_employees := 20
  let new_avg_salary := 2000
  (new_avg_salary * (num_employees + 1) - avg_salary_employees * num_employees = 12000) :=
by
  sorry

end manager_salary_l699_699524


namespace min_value_f_l699_699884

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3 / x

theorem min_value_f : ∃ y : ℝ, y > 0 ∧ (∀ x : ℝ, x > 0 → f x ≥ y) ∧ y = 2 * Real.sqrt 6 :=
by
  -- Define the function and the problem statement
  let f := λ (x : ℝ), 2 * x + 3 / x,
  -- Actual proof steps for finding the minimum value are omitted
  sorry

end min_value_f_l699_699884


namespace fraction_covered_by_mat_l699_699968

def radius_mat := 10 -- radius of the mat in inches
def side_tabletop := 24 -- side length of the tabletop in inches

noncomputable def area_mat : ℝ := Real.pi * (radius_mat ^ 2)
def area_tabletop : ℝ := side_tabletop ^ 2

theorem fraction_covered_by_mat :
  (area_mat / area_tabletop) = (100 * Real.pi) / 576 := 
by
  sorry

end fraction_covered_by_mat_l699_699968


namespace multiplicative_inverse_152_mod_367_l699_699599

theorem multiplicative_inverse_152_mod_367 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 367 ∧ (152 * a) % 367 = 1 :=
begin
  use 248,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end multiplicative_inverse_152_mod_367_l699_699599


namespace interval_contains_real_root_l699_699938

-- Define the function f(x)
def f (x : ℝ) : ℝ := real.log10 x + x

-- Define the conditions to be used
def f_at_0_1 : f 0.1 < 0 := by
  calc
    f 0.1 = real.log10 0.1 + 0.1 : by refl
    ... = -1 + 0.1 : by rw [real.log10_eq_log10]
    ... = -0.9 : by norm_num

def f_at_1 : f 1 > 0 := by
  calc
    f 1 = real.log10 1 + 1 : by refl
    ... = 0 + 1 : by rw [real.log10_one]
    ... = 1 : by norm_num

-- The theorem to be proved
theorem interval_contains_real_root : ∃ c ∈ set.Icc (0.1 : ℝ) 1, f c = 0 := by
  have h1 : f 0.1 < 0 := f_at_0_1
  have h2 : f 1 > 0 := f_at_1
  sorry

end interval_contains_real_root_l699_699938


namespace right_triangle_leg_length_l699_699869

theorem right_triangle_leg_length
  (A : ℝ)
  (b h : ℝ)
  (hA : A = 800)
  (hb : b = 40)
  (h_area : A = (1 / 2) * b * h) :
  h = 40 :=
by
  sorry

end right_triangle_leg_length_l699_699869


namespace function_strictly_increasing_on_interval_l699_699462

noncomputable def function_increasing : Real -> Real :=
  fun x => Real.cos ((1/2) * x - (Real.pi / 3))

theorem function_strictly_increasing_on_interval :
  ∀ x : Real, x ∈ Ioo (-4 * Real.pi / 3) (2 * Real.pi / 3) →
  ∃ ε > 0, ∀ y, y ∈ Ioo x (x + ε) → function_increasing y > function_increasing x :=
by sorry

end function_strictly_increasing_on_interval_l699_699462


namespace geometric_sequence_find_AB_length_l699_699018

variable {A B C: ℝ}
variable {BC AC R: ℝ} (hRpos : R > 0)
variable {area: ℝ} (hArea : 2 * area = 1)

axiom angle_eq_c : C = 3 * Real.pi / 4
axiom trig_eq : sin (A + C) = BC / R * cos (A + B)

theorem geometric_sequence 
(h1 : A + B + C = Real.pi)
(h2 : sin (A + C) = BC / R * cos (A + B))
(h3 : C = 3 * Real.pi / 4) :
AC = sqrt 2 * BC :=
sorry

theorem find_AB_length 
(h1 : A + B + C = Real.pi)
(h2 : sin (A + C) = BC / R * cos (A + B))
(h3 : C = 3 * Real.pi / 4)
(h4 : 2 * area = 1) 
(hGeom : AC = sqrt 2 * BC) :
∃ AB, AB = sqrt 10 :=
sorry

end geometric_sequence_find_AB_length_l699_699018


namespace rectangle_width_in_circle_diagonal_l699_699558

-- Define the problem as a Lean theorem statement
theorem rectangle_width_in_circle_diagonal :
  ∃ w : Real, let diameter := 24 in let length := 2 * w in (w^2 + length^2 = diameter^2) → w = Real.sqrt 115.2 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_width_in_circle_diagonal_l699_699558


namespace santa_claus_gifts_l699_699856

theorem santa_claus_gifts :
  let bags := [1, 2, 3, 4, 5, 6, 7, 8]
  (∃ (selections : list (list ℕ)), ∀ sel ∈ selections, (sel.all_different ∧ sel.sum ∈ {8, 16, 24, 32})) ∧
  selections.length = 31 :=
by sorry

end santa_claus_gifts_l699_699856


namespace length_of_platform_l699_699985

theorem length_of_platform (v t_m t_p L_t L_p : ℝ)
    (h1 : v = 33.3333333)
    (h2 : t_m = 22)
    (h3 : t_p = 45)
    (h4 : L_t = v * t_m)
    (h5 : L_t + L_p = v * t_p) :
    L_p = 766.666666 :=
by
  sorry

end length_of_platform_l699_699985


namespace expand_simplify_correct_l699_699618

noncomputable def expand_and_simplify (x : ℕ) : ℕ :=
  (x + 4) * (x - 9)

theorem expand_simplify_correct (x : ℕ) : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by
  sorry

end expand_simplify_correct_l699_699618


namespace direction_vector_b_l699_699466

theorem direction_vector_b (b : ℝ) 
  (P Q : ℝ × ℝ) (hP : P = (-3, 1)) (hQ : Q = (1, 5))
  (hdir : 3 - (-3) = 3 ∧ 5 - 1 = b) : b = 3 := by
  sorry

end direction_vector_b_l699_699466


namespace select_coprime_and_divide_l699_699823

theorem select_coprime_and_divide (n : ℕ) (selected_numbers : Finset ℕ) (h₁ : selected_numbers.card = n + 1)
  (h₂ : ∀ x ∈ selected_numbers, 1 ≤ x ∧ x ≤ 2 * n) :
  (∃ a b ∈ selected_numbers, Nat.coprime a b) ∧ (∃ c d ∈ selected_numbers, c ∣ d) :=
by
  -- Here we would proceed with the proof using Lean tactics
  sorry

end select_coprime_and_divide_l699_699823


namespace min_value_f_l699_699350

noncomputable def f (x : ℝ) (h : x > 3) : ℝ := x + 1 / (x - 3)

theorem min_value_f : (∃ x > 3, ∀ y > 3, f x ‹x > 3› ≤ f y ‹y > 3› ∧ f x ‹x > 3› = 5) :=
by
  sorry

end min_value_f_l699_699350


namespace surface_area_of_sphere_l699_699140

theorem surface_area_of_sphere (r : ℝ) (π : ℝ) (base_area : ℝ = 3) (hemisphere_area : ℝ = 9) : 
  4 * π * r^2 =
    4 * π * (base_area / π) := 
begin
  sorry,
end

end surface_area_of_sphere_l699_699140


namespace find_length_DE_l699_699515

-- Point definitions
variables (A O M : Point)
-- Segment definitions
variables (AB AC DE : LineSegment)
-- Circle definition
variables (circle : Circle)

-- Given conditions
axiom tangent_AB : Tangent AB circle
axiom tangent_AC : Tangent AC circle
axiom center_O : circle.center = O
axiom radius_15 : circle.radius = 15
axiom length_AO : dist A O = 39
axiom M_on_line_AO : OnLine M (line A O)
axiom M_on_circle : OnCircle M circle
axiom DE_through_M : TangentThrough M DE circle

-- Goal to prove
theorem find_length_DE (A O M : Point) (AB AC DE : LineSegment) (circle : Circle)
  (tangent_AB : Tangent AB circle) (tangent_AC : Tangent AC circle) (center_O : circle.center = O)
  (radius_15 : circle.radius = 15) (length_AO : dist A O = 39)
  (M_on_line_AO : OnLine M (line A O)) (M_on_circle : OnCircle M circle)
  (DE_through_M : TangentThrough M DE circle) : 
  length DE = 20 :=
sorry

end find_length_DE_l699_699515


namespace description_of_T_l699_699404

-- Define the set T based on the given conditions
def T : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | 
     let (x, y) := p in 
     (x = 8 ∧ y ≥ 4) ∨ 
     (y = 4 ∧ x ≥ 8) ∨ 
     (y = x - 4 ∧ x ≥ 8)}

-- State the theorem
theorem description_of_T : 
  T = {p : ℝ × ℝ | 
       let (x, y) := p in 
       (x = 8 ∧ y ≥ 4) ∨ 
       (y = 4 ∧ x ≥ 8) ∨ 
       (y = x - 4 ∧ x ≥ 8)} :=
sorry

end description_of_T_l699_699404


namespace remove_five_maximizes_probability_l699_699503

open Finset

theorem remove_five_maximizes_probability :
  ∀ (l : List ℤ), l = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] →
  (∃ x : ℤ, x ∈ l ∧ 
    (∀ a b : ℤ, a ≠ b → a ∈ l.erase x → b ∈ l.erase x → a + b = 10 → (a, b) ∉ (l.product l) \ {(x, 10 - x), (10 - x, x)})) →
  x = 5 :=
begin
  sorry
end

end remove_five_maximizes_probability_l699_699503


namespace line_equation_is_correct_area_of_triangle_is_correct_l699_699306

noncomputable def line_equation (x y : ℝ) : Prop :=
  ∃ m b: ℝ, m = Real.sqrt 3 ∧ b = -2 ∧ y = m * x + b

noncomputable def triangle_area (x y : ℝ) : ℝ :=
  1 / 2 * x * (-y)

theorem line_equation_is_correct :
  ∀ x y : ℝ, line_equation x y ↔ (Real.sqrt 3 * x - y - 2 = 0) := 
by
  intros x y
  split
  · rintro ⟨m, b, hm, hb, hy⟩
    rw [hm, hb] at hy
    exact Eq.symm hy
  · intro h
    use (Real.sqrt 3), -2
    simp [h]

theorem area_of_triangle_is_correct :
  ∃ x y : ℝ, line_equation x y ∧ triangle_area (2 / Real.sqrt 3) (-2) = (2 * Real.sqrt 3) / 3 :=
by
  use 2 / Real.sqrt 3, -2
  split
  · apply line_equation_is_correct
  · simp [triangle_area]
  sorry

end line_equation_is_correct_area_of_triangle_is_correct_l699_699306


namespace Randy_initial_money_l699_699434

theorem Randy_initial_money (M : ℝ) (r1 : M + 200 - 1200 = 2000) : M = 3000 :=
by
  sorry

end Randy_initial_money_l699_699434


namespace number_of_elements_in_set_B_l699_699689

theorem number_of_elements_in_set_B :
  let A := {1, 2, 4}
  let B := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}
  (B.card = 6) := 
by 
  let A := {1, 2, 4}
  let B := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}
  have : B = {2, 3, 4, 5, 6, 8} := sorry
  exact sorry

end number_of_elements_in_set_B_l699_699689


namespace fraction_pizza_covered_by_pepperoni_l699_699094

/--
Given that six pepperoni circles fit exactly across the diameter of a 12-inch pizza
and a total of 24 circles of pepperoni are placed on the pizza without overlap,
prove that the fraction of the pizza covered by pepperoni is 2/3.
-/
theorem fraction_pizza_covered_by_pepperoni : 
  (∃ d r : ℝ, 6 * r = d ∧ d = 12 ∧ (r * r * π * 24) / (6 * 6 * π) = 2 / 3) := 
sorry

end fraction_pizza_covered_by_pepperoni_l699_699094


namespace solid_volume_l699_699477

noncomputable def volume_of_solid (s : ℝ) : ℝ :=
  if s = 4 * Real.sqrt 2 then 42.67 * Real.sqrt 2 else 0

theorem solid_volume (s : ℝ) (h : s = 4 * Real.sqrt 2) : 
  volume_of_solid s = 42.67 * Real.sqrt 2 := 
by
  rw [h]
  unfold volume_of_solid
  simp
  sorry

end solid_volume_l699_699477


namespace ms_warren_running_time_l699_699422

theorem ms_warren_running_time 
  (t : ℝ) 
  (ht_total_distance : 6 * t + 2 * 0.5 = 3) : 
  60 * t = 20 := by 
  sorry

end ms_warren_running_time_l699_699422


namespace exists_magic_grid_l699_699080

def grid : Type := list (list ℕ)

def is_valid_grid (g : grid) : Prop := 
  ∀ (i j : ℕ), i < 3 → j < 3 → 
  (664 ≤ g[i][j] ∧ g[i][j] ≤ 671) ∧ 
  list.nodup (list.join g)

def sum_to_2001 (lst : list ℕ) : Prop := list.sum lst = 2001

def magic_grid (g : grid) : Prop :=
  is_valid_grid g ∧
  sum_to_2001 (g[0]) ∧ 
  sum_to_2001 (g[1]) ∧ 
  sum_to_2001 (g[2]) ∧ 
  sum_to_2001 (g.map (λ row, row[0])) ∧ 
  sum_to_2001 (g.map (λ row, row[1])) ∧ 
  sum_to_2001 (g.map (λ row, row[2])) ∧ 
  sum_to_2001 [g[0][0], g[1][1], g[2][2]] ∧ 
  sum_to_2001 [g[0][2], g[1][1], g[2][0]]

theorem exists_magic_grid : ∃ g : grid, magic_grid g :=
sorry

end exists_magic_grid_l699_699080


namespace tyudejo_subsets_count_l699_699504

def is_tyudejo (S : Set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → abs (x - y) ≠ 2

theorem tyudejo_subsets_count :
  {S : Set ℕ | is_tyudejo S ∧ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}.card = 169 := 
by 
  sorry

end tyudejo_subsets_count_l699_699504


namespace rectangle_length_l699_699132

theorem rectangle_length (P L W : ℕ) (h1 : P = 48) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : L = 16 := by
  sorry

end rectangle_length_l699_699132


namespace canoe_upstream_speed_l699_699182

namespace canoe_speed

def V_c : ℝ := 12.5            -- speed of the canoe in still water in km/hr
def V_downstream : ℝ := 16     -- speed of the canoe downstream in km/hr

theorem canoe_upstream_speed :
  ∃ (V_upstream : ℝ), V_upstream = V_c - (V_downstream - V_c) ∧ V_upstream = 9 := by
  sorry

end canoe_speed

end canoe_upstream_speed_l699_699182


namespace area_of_figure_EFGH_l699_699371

-- Define the radius of the circle
def radius : ℝ := 10

-- Define the angle of each sector in degrees
def sector_angle_degrees : ℝ := 45

-- Convert the angle to radians since Lean's trigonometric functions use radians
def sector_angle_radians : ℝ := (sector_angle_degrees / 360) * 2 * Real.pi

-- Define the formula for the area of a circle
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the area of one sector
def sector_area (r : ℝ) (angle : ℝ) : ℝ := (angle / (2 * Real.pi)) * circle_area r

-- Define the total area as the sum of two such sectors
def total_area (r : ℝ) (angle : ℝ) : ℝ := 2 * sector_area r angle

-- The theorem we want to prove: The total area of figure EFGH is 25π
theorem area_of_figure_EFGH : total_area radius sector_angle_radians = 25 * Real.pi :=
  by sorry

end area_of_figure_EFGH_l699_699371


namespace solve_system_of_equations_l699_699863

theorem solve_system_of_equations
  (a b c : ℝ) (x y z : ℝ)
  (h1 : x + y = a)
  (h2 : y + z = b)
  (h3 : z + x = c) :
  x = (a + c - b) / 2 ∧ y = (a + b - c) / 2 ∧ z = (b + c - a) / 2 :=
by
  sorry

end solve_system_of_equations_l699_699863


namespace arithmetic_sqrt_64_eq_8_l699_699872

theorem arithmetic_sqrt_64_eq_8 : real.sqrt 64 = 8 :=
by
  sorry

end arithmetic_sqrt_64_eq_8_l699_699872


namespace evaluate_expression_l699_699249

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := 
by
  sorry

end evaluate_expression_l699_699249


namespace total_production_first_three_days_max_minus_min_production_total_wage_calculation_l699_699535

-- Definitions for the given conditions
def planned_production_week : ℕ := 1400
def average_daily_production : ℕ := 200
def deviations : List ℤ := [7, -4, -5, 11, -10, 16, -6]

-- Total production in the first three days
theorem total_production_first_three_days : ((average_daily_production * 3 : ℤ) + deviations.take 3.sum = 598) :=
  by sorry

-- Difference between highest and lowest production days
theorem max_minus_min_production : (deviations.maximum - deviations.minimum = 26) :=
  by sorry

-- Total wage calculation for the week
theorem total_wage_calculation : (let total_production := (average_daily_production * 7 : ℤ) + deviations.sum,
                                       base_wage := total_production * 60,
                                       bonus := (total_production - planned_production_week) * 10
                                   in base_wage + bonus = 84570) :=
  by sorry

end total_production_first_three_days_max_minus_min_production_total_wage_calculation_l699_699535


namespace diving_club_capacity_l699_699451

theorem diving_club_capacity :
  (3 * ((2 * 5 + 4 * 2) * 5) = 270) :=
by
  sorry

end diving_club_capacity_l699_699451


namespace ball_transfer_probabilities_equal_l699_699366

theorem ball_transfer_probabilities_equal :
  let initial_red_balls := 100
  let initial_green_balls := 100
  let red_box_initial := initial_red_balls
  let green_box_initial := initial_green_balls
  let red_to_green := 8
  let green_box_after_first_transfer := green_box_initial + red_to_green
  let red_box_after_first_transfer := red_box_initial - red_to_green
  let total_transfer_back := 8

  -- After transferring back
  let red_box_final := red_box_after_first_transfer + total_transfer_back
  let green_box_final := green_box_after_first_transfer - total_transfer_back

  let green_balls_in_red_box := total_transfer_back
  let red_balls_in_green_box := red_to_green

  Prob_green_ball_red_box := (green_balls_in_red_box : ℚ) / (red_box_final : ℚ),
  Prob_red_ball_green_box := (red_balls_in_green_box : ℚ) / (green_box_final : ℚ)
  in

  Prob_green_ball_red_box = Prob_red_ball_green_box :=
sorry

end ball_transfer_probabilities_equal_l699_699366


namespace inequality_relationship_l699_699640

noncomputable def a := 1 / 2023
noncomputable def b := Real.exp (-2022 / 2023)
noncomputable def c := (Real.cos (1 / 2023)) / 2023

theorem inequality_relationship : b > a ∧ a > c :=
by
  -- Initializing and defining the variables
  let a := a
  let b := b
  let c := c
  -- Providing the required proof
  sorry

end inequality_relationship_l699_699640


namespace intersection_example_l699_699691

open Set

theorem intersection_example : 
  M = {1, 2} → N = {2, 3, 4} → M ∩ N = {2} :=
by
  intros hM hN
  sorry

end intersection_example_l699_699691


namespace min_varphi_shift_symmetric_l699_699711

noncomputable def f (x : ℝ) : ℝ := 3 * real.sin x + real.sqrt 3 * real.cos x

theorem min_varphi_shift_symmetric :
  ∃ (φ : ℝ), φ = real.pi / 6 ∧
  (∀ x, f (x + φ) = -f (-x - φ)) ∧
  φ > 0 :=
begin
  sorry
end

end min_varphi_shift_symmetric_l699_699711


namespace equation_is_hyperbola_l699_699240

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := x^2 - 36 * y^2 - 12 * x + y + 64 = 0

-- The theorem stating the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ (x y : ℝ), equation x y → 
(is_hyperbola x y) := 
sorry

end equation_is_hyperbola_l699_699240


namespace parallel_MN_AB_l699_699082

variables {α : Type*} [preorder α] [add_comm_monoid α]

theorem parallel_MN_AB {A B C D M N : α} 
  (h₁ : M ∈ parallelogram ABCD)
  (h₂ : N ∈ triangle AMD)
  (h₃ : ∠MNA + ∠MCB = 180)
  (h₄ : ∠MND + ∠MBC = 180) :
  MN ∥ AB :=
by
  sorry

end parallel_MN_AB_l699_699082


namespace pythagorean_triple_correct_l699_699209

-- Defining what a Pythagorean triple is
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Sets of numbers to be checked
def set_A := (sqrt 3, sqrt 4, sqrt 5)
def set_B := (1, sqrt 2, sqrt 3)
def set_C := (7, 24, 25)
def set_D := (2, 3, 4)

-- The main theorem to be proven
theorem pythagorean_triple_correct :
  is_pythagorean_triple 7 24 25 ∧ 
  ¬ is_pythagorean_triple (sqrt 3) (sqrt 4) (sqrt 5) ∧
  ¬ is_pythagorean_triple 1 (sqrt 2) (sqrt 3) ∧
  ¬ is_pythagorean_triple 2 3 4 := 
sorry

end pythagorean_triple_correct_l699_699209


namespace smallest_positive_x_7890_palindrome_l699_699162

def is_palindrome (n : ℕ) : Prop :=
    let s := n.to_string in
    s = s.reverse

theorem smallest_positive_x_7890_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 7890) ∧ x = 107 :=
by
  sorry

end smallest_positive_x_7890_palindrome_l699_699162


namespace no_digit_assignment_exists_l699_699740

theorem no_digit_assignment_exists :
  ¬ ∃ (C T Ц И Φ P A : ℕ), 
      C ≠ T ∧ C ≠ Ц ∧ C ≠ И ∧ C ≠ Φ ∧ C ≠ P ∧ C ≠ A ∧
      T ≠ Ц ∧ T ≠ И ∧ T ≠ Φ ∧ T ≠ P ∧ T ≠ A ∧
      Ц ≠ И ∧ Ц ≠ Φ ∧ Ц ≠ P ∧ Ц ≠ A ∧
      И ≠ Φ ∧ И ≠ P ∧ И ≠ A ∧
      Φ ≠ P ∧ Φ ≠ A ∧
      P ≠ A ∧
      C ∈ finset.range 10 ∧
      T ∈ finset.range 10 ∧
      Ц ∈ finset.range 10 ∧
      И ∈ finset.range 10 ∧
      Φ ∈ finset.range 10 ∧
      P ∈ finset.range 10 ∧
      A ∈ finset.range 10 ∧
      C * T * 0 = Ц * И * Φ * P * A :=
by
  sorry

end no_digit_assignment_exists_l699_699740


namespace measure_angle_FCA_is_correct_l699_699684

noncomputable def measure_angle_FCA (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (F A B C : ℝ × ℝ) 
  (Gamma : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (l : ∀ x, x ∈ line_through F A B C)
  (angle_FAB : ∃ α : ℝ, α = 50)
  (angle_FBA : ∃ β : ℝ, β = 20) :
  ℝ := -2 * real.sqrt 3 / 3

theorem measure_angle_FCA_is_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (F A B C : ℝ × ℝ) 
  (Gamma : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (l : ∀ x, x ∈ line_through F A B C)
  (angle_FAB : ∃ α : ℝ, α = 50)
  (angle_FBA : ∃ β : ℝ, β = 20) :
  measure_angle_FCA a b ha hb F A B C Gamma l angle_FAB angle_FBA = -2 * real.sqrt 3 / 3 :=
by
  sorry

end measure_angle_FCA_is_correct_l699_699684


namespace monotonically_increasing_range_l699_699770

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699770


namespace pants_original_price_l699_699380

theorem pants_original_price (P : ℝ) (h1 : P * 0.6 = 50.40) : P = 84 :=
sorry

end pants_original_price_l699_699380


namespace range_of_a_same_side_of_line_l699_699307

theorem range_of_a_same_side_of_line 
  {P Q : ℝ × ℝ} 
  (hP : P = (3, -1)) 
  (hQ : Q = (-1, 2)) 
  (h_side : (3 * a - 3) * (-a + 3) > 0) : 
  a > 1 ∧ a < 3 := 
by 
  sorry

end range_of_a_same_side_of_line_l699_699307


namespace fungi_population_exceeds_1000_at_day_6_l699_699022

theorem fungi_population_exceeds_1000_at_day_6 : 
  ∃ n : ℕ, 4 * 3^n > 1000 ∧ 
  ∀ m : ℕ, m < n → 4 * 3^m ≤ 1000 :=
begin
  sorry
end

end fungi_population_exceeds_1000_at_day_6_l699_699022


namespace monotonic_increasing_range_l699_699790

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699790


namespace cube_root_25360000_l699_699344

theorem cube_root_25360000 :
  ∀ (a b : ℝ), (∛a = 2.938) → (∛b = 6.329) → a * 1000000 = 25360000 → ∛(a * 1000000) = 293.8 :=
by 
  intros a b ha hb habm
  sorry

end cube_root_25360000_l699_699344


namespace max_equal_distance_points_l699_699348

theorem max_equal_distance_points (n : ℕ) (h : ∀ (i j : ℕ), i ≠ j → i < n → j < n → dist (points i) (points j) = d) : 
  n ≤ 4 :=
sorry

end max_equal_distance_points_l699_699348


namespace base3_to_base10_conversion_l699_699238

theorem base3_to_base10_conversion : ∀ n : ℕ, n = 120102 → (1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = 416 :=
by
  intro n hn
  sorry

end base3_to_base10_conversion_l699_699238


namespace calculate_remaining_area_l699_699027

/-- In a rectangular plot of land ABCD, where AB = 20 meters and BC = 12 meters, 
    a triangular garden ABE is installed where AE = 15 meters and BE intersects AE at a perpendicular angle, 
    the area of the remaining part of the land which is not occupied by the garden is 150 square meters. -/
theorem calculate_remaining_area 
  (AB BC AE : ℝ) 
  (hAB : AB = 20) 
  (hBC : BC = 12) 
  (hAE : AE = 15)
  (h_perpendicular : true) : -- BE ⊥ AE implying right triangle ABE
  ∃ area_remaining : ℝ, area_remaining = 150 :=
by
  sorry

end calculate_remaining_area_l699_699027


namespace gcd_g50_g52_l699_699820

def g (x : ℕ) : ℕ := x^2 - 2 * x + 2021

theorem gcd_g50_g52 : Nat.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g50_g52_l699_699820


namespace smallest_perfect_square_div_l699_699932

theorem smallest_perfect_square_div :
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ 4 ∣ n ∧ 10 ∣ n ∧ 18 ∣ n ∧ n = 900 :=
by
  sorry

end smallest_perfect_square_div_l699_699932


namespace quadrilateral_area_EFGH_l699_699728

theorem quadrilateral_area_EFGH :
  ∀ (EF FG GH : ℝ) (mE mG : ℝ),
  mE = 135 ∧ mG = 135 ∧ EF = 4 ∧ FG = 6 ∧ GH = 7 → 
  let sin135 := Real.sin (135 * Real.pi / 180)
  in (1/2 * EF * FG * sin135 + 1/2 * FG * GH * sin135 = 27 * Real.sqrt 2) :=
by
  intro EF FG GH mE mG
  intro h
  -- define the required values from the conditions
  let sin135 := Real.sin (135 * Real.pi / 180)
  -- replace the proof with sorry to indicate the need for further formal proof steps
  sorry

end quadrilateral_area_EFGH_l699_699728


namespace g_42_value_l699_699818

-- Definitions based on the given conditions:
def increasing (g : ℕ → ℕ) : Prop := ∀ m n : ℕ, m < n → g(m) < g(n)

def multiplicative (g : ℕ → ℕ) : Prop := ∀ m n : ℕ, g(m * n) = g(m) * g(n)

def special_property (g : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m ≠ n ∧ m^n = n^m → (g(m) = n ∨ g(n) = m)

-- The main proof problem statement:
noncomputable def g : ℕ → ℕ :=
  sorry

theorem g_42_value (g : ℕ → ℕ) :
  increasing g →
  multiplicative g →
  special_property g →
  g(42) = 8668 :=
sorry

end g_42_value_l699_699818


namespace number_of_triangles_l699_699107

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l699_699107


namespace tan_half_sum_of_angles_l699_699410

theorem tan_half_sum_of_angles (p q : ℝ) 
    (h1 : Real.cos p + Real.cos q = 3 / 5) 
    (h2 : Real.sin p + Real.sin q = 1 / 4) :
    Real.tan ((p + q) / 2) = 5 / 12 := by
  sorry

end tan_half_sum_of_angles_l699_699410


namespace shortest_distance_MN_l699_699730

-- Definitions of the vertices and properties of the tetrahedron
def A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (2 : ℝ, 0 : ℝ, 0 : ℝ)
def C := (1 : ℝ, (Real.sqrt 3), 0 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 2 * Real.sqrt 2)

-- Midpoints of edges AB and CD
def M := ((0 + 2) / 2, (0 + 0) / 2, (0 + 0) / 2)
def N := ((1 + 0) / 2, (Real.sqrt 3 + 0) / 2, (0 + 2 * Real.sqrt 2) / 2)

-- Function to calculate distance between two points in 3D
noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Theorem to prove the shortest distance between M and N is 2
theorem shortest_distance_MN : distance M N = 2 :=
by {
  sorry
}

end shortest_distance_MN_l699_699730


namespace Sum_of_A_and_B_is_7_l699_699035

-- Definitions of the digits and the condition that they are different
variables {A B C D: ℕ}
-- Digits are from 0 to 9 (but here we need them to remain digits after checking multiplicative condition)
def is_digit (n: ℕ) : Prop := n ≥ 0 ∧ n < 10

-- All digits are different
def all_digits_different (A B C D: ℕ) : Prop := A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
                                                   B ≠ C ∧ B ≠ D ∧ 
                                                   C ≠ D

-- Multiplication and struct condition
def multiplication_condition (A B C D: ℕ) : Prop :=
  let num1 := 100 * A + 10 * B + C in
  let num2 := D in
  let result_num := 100 * B + 10 * C + D in
  num1 * num2 = result_num

-- The modulo condition
def modulo_condition (C D: ℕ) : Prop := (C * D) % 10 = D

-- Main conjecture: given all conditions, A + B = 7
theorem Sum_of_A_and_B_is_7
  (h_digits_A: is_digit A) (h_digits_B: is_digit B) (h_digits_C: is_digit C) (h_digits_D: is_digit D)
  (h_diff: all_digits_different A B C D) (h_mult: multiplication_condition A B C D) (h_mod: modulo_condition C D):
  A + B = 7 :=
sorry

end Sum_of_A_and_B_is_7_l699_699035


namespace value_of_expression_l699_699167

theorem value_of_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) : x + 2 * y = 10 :=
by
  -- Proof goes here
  sorry

end value_of_expression_l699_699167


namespace margo_total_distance_l699_699832

-- Definitions based on the conditions
def time_to_friends_house_min : ℕ := 15
def time_to_return_home_min : ℕ := 25
def total_walking_time_min : ℕ := time_to_friends_house_min + time_to_return_home_min
def total_walking_time_hours : ℚ := total_walking_time_min / 60
def average_walking_rate_mph : ℚ := 3
def total_distance_miles : ℚ := average_walking_rate_mph * total_walking_time_hours

-- The statement of the proof problem
theorem margo_total_distance : total_distance_miles = 2 := by
  sorry

end margo_total_distance_l699_699832


namespace inequality_not_always_hold_l699_699303

variable (a b c : ℝ)

theorem inequality_not_always_hold (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) : ¬ (∀ (a b : ℝ), |a - b| + 1 / (a - b) ≥ 2) :=
by
  sorry

end inequality_not_always_hold_l699_699303


namespace ratio_of_areas_of_similar_isosceles_triangles_l699_699912

theorem ratio_of_areas_of_similar_isosceles_triangles (A B : ℝ) (h1 h2 b1 b2 : ℝ) 
  (h_ratio : h1 / h2 = 2 / 3) (b_ratio : b1 / b2 = 2 / 3) :
  (A = (b1 * h1) / 2) → (B = (b2 * h2) / 2) → (A / B = 4 / 9) :=
begin
  sorry
end

end ratio_of_areas_of_similar_isosceles_triangles_l699_699912


namespace intersection_point_correct_l699_699333

noncomputable def find_intersection_point (a b : ℝ) (h1 : a = b + 5) (h2 : a = 4 * b + 2) : ℝ × ℝ :=
  let x := -3
  let y := -14
  (x, y)

theorem intersection_point_correct : 
  ∀ (a b : ℝ) (h1 : a = b + 5) (h2 : a = 4 * b + 2), 
  find_intersection_point a b h1 h2 = (-3, -14) := by
  sorry

end intersection_point_correct_l699_699333


namespace count_valid_m_l699_699274

theorem count_valid_m (h : 1260 > 0) :
  ∃! (n : ℕ), n = 3 := by
  sorry

end count_valid_m_l699_699274


namespace max_value_of_sin_minus_cos_l699_699467

variable (x : ℝ)

def y := sin x - cos x

theorem max_value_of_sin_minus_cos : (∀ x, y x ≤ real.sqrt 2) ∧ (∃ x, y x = real.sqrt 2) := by
  sorry

end max_value_of_sin_minus_cos_l699_699467


namespace sum_divisors_equal_l699_699046

theorem sum_divisors_equal (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (hk : n = 1990.factorial)
  (hdivisors : ∀ i, 1 ≤ i ∧ i ≤ k → (d i ∣ n)) :
  (∑ i in finset.range k, (d i) / (n.sqrt)) = (∑ i in finset.range k, (n.sqrt) / (d i)) :=
sorry

end sum_divisors_equal_l699_699046


namespace evaluate_expression_l699_699250

noncomputable def log_base_10 (x : ℝ) : ℝ := log x / log 10

theorem evaluate_expression :
  (log_base_10 (sqrt 10 * log_base_10 1000)) ^ 2 = (log_base_10 3 + 0.5) ^ 2 :=
by
  sorry

end evaluate_expression_l699_699250


namespace sequence_nth_term_16_l699_699329

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end sequence_nth_term_16_l699_699329


namespace board_transformation_l699_699601

def transformation_possible (a b : ℕ) : Prop :=
  6 ∣ (a * b)

theorem board_transformation (a b : ℕ) (h₁ : 2 ≤ a) (h₂ : 2 ≤ b) : 
  transformation_possible a b ↔ 6 ∣ (a * b) := by
  sorry

end board_transformation_l699_699601


namespace determine_k_l699_699281

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 2018).sum (λ i, |x - (i:ℝ) - 1| + |x + (i:ℝ) + 1|)

noncomputable def g (x k: ℝ) : ℝ :=
  (x^2 * (x^2 + k^2 + 2*k - 4) + 4) / ((x^2 + 2)^2 - 2*x^2)

theorem determine_k (k : ℝ) : 
  (∃ m n : ℕ, f ((a:ℕ) ^ 2 - 3 * a + 2) = f ((a:ℕ) - 1) 
    ∧ g x k = 1 ∧ m + n = 3 ∧ n = 2)
  → k = -1 + real.sqrt 7 ∨ k = -1 - real.sqrt 7 :=
begin
  intros,
  sorry
end

end determine_k_l699_699281


namespace min_value_modulus_sub_l699_699647

open Complex

theorem min_value_modulus_sub : ∀ (z : ℂ), abs(z) = 1 → abs(z - (3 + 4 * complex.I)) ≥ 4 :=
by
  intros z h
  sorry

noncomputable def min_modulus_sub : ℂ → ℝ :=
  λ z, if abs(z) = 1 then 4 else 0

end min_value_modulus_sub_l699_699647


namespace correct_option_D_l699_699511

variables (a b c : ℤ)

theorem correct_option_D : -2 * a + 3 * (b - 1) = -2 * a + 3 * b - 3 := 
by
  sorry

end correct_option_D_l699_699511


namespace monotonically_increasing_range_l699_699772

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699772


namespace bottles_left_after_purchase_l699_699144

def initial_bottles : ℕ := 35
def jason_bottles : ℕ := 5
def harry_bottles : ℕ := 6
def jason_effective_bottles (n : ℕ) : ℕ := n  -- Jason buys 5 bottles
def harry_effective_bottles (n : ℕ) : ℕ := n + 1 -- Harry gets one additional free bottle

theorem bottles_left_after_purchase (j_b h_b i_b : ℕ) (j_effective h_effective : ℕ → ℕ) :
  j_b = 5 → h_b = 6 → i_b = 35 → j_effective j_b = 5 → h_effective h_b = 7 →
  i_b - (j_effective j_b + h_effective h_b) = 23 :=
by
  intros
  sorry

end bottles_left_after_purchase_l699_699144


namespace max_quantity_of_Pl_at_ln_2_l699_699079

noncomputable theory

def x (t : ℝ) : ℝ := 10 * Real.exp (-t)
def y (t : ℝ) : ℝ := 10 * Real.exp (-t) - 10 * Real.exp (-2 * t)
def y' (t : ℝ) : ℝ := -10 * Real.exp (-t) + 20 * Real.exp (-2 * t)

theorem max_quantity_of_Pl_at_ln_2 :
  ∀ t : ℝ, y' t = 0 → t = Real.ln 2 :=
by
  intro t
  unfold y'
  sorry

end max_quantity_of_Pl_at_ln_2_l699_699079


namespace change_order_of_integration_l699_699596

-- Define the function f over ℝ × ℝ → ℝ
variable (f : ℝ → ℝ → ℝ)

-- Define the region of integration and its conditions
theorem change_order_of_integration :
  (∫ x in 0..1, ∫ y in (sqrt x)..(2 - x), f x y) = 
  (∫ y in 0..1, ∫ x in 0..(y^2), f x y) + 
  (∫ y in 1..2, ∫ x in 0..(2 - y), f x y) :=
by
  sorry

end change_order_of_integration_l699_699596


namespace radius_semicircle_proof_l699_699198

noncomputable def radius_of_semicircle_inscribed_in_isosceles_triangle
  (base height : ℝ) (h_base : base = 24) (h_height : height = 18) : ℝ :=
  let r : ℝ := 36 * Real.sqrt 13 / 13
  in r

theorem radius_semicircle_proof :
  radius_of_semicircle_inscribed_in_isosceles_triangle 24 18 rfl rfl =
  36 * Real.sqrt 13 / 13 :=
sorry

end radius_semicircle_proof_l699_699198


namespace clarence_initial_oranges_l699_699598

variable (initial_oranges : ℕ)
variable (obtained_from_joyce : ℕ := 3)
variable (total_oranges : ℕ := 8)

theorem clarence_initial_oranges (initial_oranges : ℕ) :
  initial_oranges + obtained_from_joyce = total_oranges → initial_oranges = 5 :=
by
  sorry

end clarence_initial_oranges_l699_699598


namespace number_of_triples_l699_699402

-- Defining the set X with n elements
constant (X : Type) (n : ℕ)

-- Question: find the number of ordered triples (A, B, C) of subsets of X such that A ⊆ B ⊊ C
theorem number_of_triples (X : Type) (n : ℕ) (h : fintype.card X = n) :
  let subsets := {A : set X // ∃ (B C : set X), A ⊆ B ∧ B ⊊ C}
  in finset.card subsets = 4^n - 3^n :=
sorry

end number_of_triples_l699_699402


namespace monotonic_increasing_range_l699_699791

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699791


namespace right_triangle_area_l699_699920

theorem right_triangle_area (a b c : ℕ) (h_right : c^2 = a^2 + b^2) (h_a : a = 40) (h_c : c = 41) : 
  1 / 2 * a * b = 180 := by
  have h_b_squared : b^2 = 81 :=
  calc
    b^2 = c^2 - a^2 : by rw [←h_right]
    ... = 41^2 - 40^2 : by rw [h_a, h_c]
    ... = 1681 - 1600 : rfl
    ... = 81 : rfl,
  have h_b : b = 9 := by 
    rw [eq_comm, Nat.pow_two, Nat.pow_eq_iff_eq] at h_b_squared,
    exact h_b_squared,
  simp [h_a, h_b],
  sorry

end right_triangle_area_l699_699920


namespace marks_lost_per_incorrect_sum_l699_699091

variables (marks_per_correct : ℕ) (total_attempts total_marks correct_sums : ℕ)
variable (marks_per_incorrect : ℕ)
variable (incorrect_sums : ℕ)

def calc_marks_per_incorrect_sum : Prop :=
  marks_per_correct = 3 ∧ 
  total_attempts = 30 ∧ 
  total_marks = 50 ∧ 
  correct_sums = 22 ∧ 
  incorrect_sums = total_attempts - correct_sums ∧ 
  (marks_per_correct * correct_sums) - (marks_per_incorrect * incorrect_sums) = total_marks ∧ 
  marks_per_incorrect = 2

theorem marks_lost_per_incorrect_sum : calc_marks_per_incorrect_sum 3 30 50 22 2 (30 - 22) :=
sorry

end marks_lost_per_incorrect_sum_l699_699091


namespace find_divisors_of_10_pow_10_sum_157_l699_699899

theorem find_divisors_of_10_pow_10_sum_157 
  (x y : ℕ) 
  (hx₁ : 0 < x) 
  (hy₁ : 0 < y) 
  (hx₂ : x ∣ 10^10) 
  (hy₂ : y ∣ 10^10) 
  (hxy₁ : x ≠ y) 
  (hxy₂ : x + y = 157) : 
  (x = 32 ∧ y = 125) ∨ (x = 125 ∧ y = 32) := 
by
  sorry

end find_divisors_of_10_pow_10_sum_157_l699_699899


namespace factorization_correct_l699_699254

theorem factorization_correct (x : ℝ) :
    x^2 - 3 * x - 4 = (x + 1) * (x - 4) :=
  sorry

end factorization_correct_l699_699254


namespace measure_of_angle_AOC_l699_699557

-- Definitions of the rotations based on the conditions.
def rotation_OA_OB := 270 -- Counterclockwise rotation from OA to OB
def rotation_OB_OC := -360 -- Clockwise rotation from OB to OC (one full turn)

-- The measure of angle ∠AOC is the absolute value of the total rotation from OA to OC.
def angle_AOC := | rotation_OA_OB + rotation_OB_OC |

-- The main statement to be proved.
theorem measure_of_angle_AOC : angle_AOC = 90 := by
  sorry

end measure_of_angle_AOC_l699_699557


namespace find_surface_area_of_sphere_l699_699308

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 := a * b = 6
def condition2 := b * c = 2
def condition3 := a * c = 3
def vertices_on_sphere := true  -- Assuming vertices on tensor sphere condition for mathematical completion

theorem find_surface_area_of_sphere
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c)
  (h4 : vertices_on_sphere) :
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 14 * Real.pi :=
  sorry

end find_surface_area_of_sphere_l699_699308


namespace keith_total_spending_l699_699042

-- Define the various spending amounts
def speakers_usd : Float := 136.01
def cd_player_usd : Float := 139.38
def tires_gbp : Float := 85.62
def num_tires : Int := 4
def printer_cables_eur : Float := 12.54
def num_cables : Int := 2
def blank_cds_jpy : Float := 9800

-- Define the local sales tax rate
def sales_tax_rate : Float := 8.25 / 100

-- Define the currency exchange rates
def gbp_to_usd : Float := 1.38
def eur_to_usd : Float := 1.12
def jpy_to_usd : Float := 0.0089

-- Define the total amount Keith spent in USD (expected outcome)
def total_spent_usd : Float := 886.04

-- The proof statement
theorem keith_total_spending : 
    speakers_usd + cd_player_usd * (1 + sales_tax_rate) + 
    (num_tires * tires_gbp * gbp_to_usd) + 
    (num_cables * printer_cables_eur * eur_to_usd) + 
    (blank_cds_jpy * jpy_to_usd) = total_spent_usd := 
by 
    sorry

end keith_total_spending_l699_699042


namespace rain_over_weekend_l699_699891

open ProbabilityTheory

theorem rain_over_weekend (P_R_S : ℚ)
  (P_R_Sn : ℚ)
  (P_R_Sn_given_no_R_S : ℚ) :
  P_R_S = 0.60 →
  P_R_Sn = 0.40 →
  P_R_Sn_given_no_R_S = 0.70 →
  let P_no_R_S := 1 - P_R_S in
  let P_no_R_Sn_given_no_R_S := 1 - P_R_Sn_given_no_R_S in
  let P_no_R_Sn_given_R_S := 1 - P_R_Sn in
  1 - (P_no_R_S * P_no_R_Sn_given_no_R_S) - (P_R_S * P_no_R_Sn_given_R_S) = 0.88 :=
begin
  intros h1 h2 h3,
  let P_no_R_S := 1 - P_R_S,
  let P_no_R_Sn_given_no_R_S := 1 - P_R_Sn_given_no_R_S,
  let P_no_R_Sn_given_R_S := 1 - P_R_Sn,
  have h4 : 1 - (P_no_R_S * P_no_R_Sn_given_no_R_S) - (P_R_S * P_no_R_Sn_given_R_S) = 0.88,
  { simp [h1, h2, h3, P_no_R_S, P_no_R_Sn_given_no_R_S, P_no_R_Sn_given_R_S],
    norm_num },
  exact h4,
end

end rain_over_weekend_l699_699891


namespace number_of_zeros_of_f_l699_699311

noncomputable def f (x : ℝ) : ℝ := if x > 0 then 2017^x + log x / log 2017 else if x < 0 then -(2017^(-x) + log (-x) / log 2017) else 0

theorem number_of_zeros_of_f :
  ∃! z₀ z₁ z₂ : ℝ, z₀ ≠ z₁ → z₀ ≠ z₂ → z₁ ≠ z₂ → f z₀ = 0 ∧ f z₁ = 0 ∧ f z₂ = 0 := sorry

end number_of_zeros_of_f_l699_699311


namespace obtuse_angle_between_line_and_plane_l699_699990

-- Define the problem conditions
def is_obtuse_angle (θ : ℝ) : Prop := θ > 90 ∧ θ < 180

-- Define what we are proving
theorem obtuse_angle_between_line_and_plane (θ : ℝ) (h1 : θ = angle_between_line_and_plane) :
  is_obtuse_angle θ :=
sorry

end obtuse_angle_between_line_and_plane_l699_699990


namespace midpoint_t2_is_correct_l699_699092

-- Defining the points and the translation operation
def point1 : ℝ × ℝ := (5, -3)
def point2 : ℝ × ℝ := (-7, 9)
def translation : ℝ × ℝ := (-3, -2)

-- Calculating the midpoint of segment t1
def midpoint_t1 : ℝ × ℝ :=
  ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)

-- Applying the translation to the midpoint of t1 to get the midpoint of t2
def midpoint_t2 : ℝ × ℝ :=
  (midpoint_t1.1 + translation.1, midpoint_t1.2 + translation.2)

-- The statement to prove that the midpoint of segment t2 equals (-4, 1)
theorem midpoint_t2_is_correct : midpoint_t2 = (-4, 1) :=
  sorry

end midpoint_t2_is_correct_l699_699092


namespace surface_area_calculation_l699_699531

def total_surface_area_after_removals (n : ℕ) : ℕ :=
  let initial_cubes := 64
  let removed_cubes := 12
  let remaining_cubes := initial_cubes - removed_cubes
  let surface_area_per_cube := 54 + 24
  let total_surface_area := remaining_cubes * surface_area_per_cube - 36
  total_surface_area

theorem surface_area_calculation :
  total_surface_area_after_removals 12 = 4020 :=
by
  unfold total_surface_area_after_removals
  rw [Nat.mul_sub_left_distrib, Nat.sub_mul, Nat.add_sub_cancel]
  norm_num
  sorry

end surface_area_calculation_l699_699531


namespace perimeter_ABCDFG_l699_699148

def triangle (A B C : Type) := sorry

def midpoint (A B C : Type) := sorry

def equilateral (A B C : Type) := sorry

noncomputable def perimeter (points : List ℕ) := points.sum

theorem perimeter_ABCDFG : 
  ∀ (A B C D E F G : Type),
  (equilateral A B C) →
  (equilateral A D E) →
  (equilateral D F G) →
  (midpoint D A C) →
  (midpoint F D E) →
  (∀ (x : Type), x = A → x = B → 6) →
  perimeter [6, 6, 3, 1.5, 1.5, 3] = 21 :=
by
  sorry

end perimeter_ABCDFG_l699_699148


namespace polynomial_expression_constant_l699_699062

variable {R : Type*} [CommRing R]

/-- Let f and g be polynomials of degree n. Prove that the expression 
    f g^{(n)} - f' g^{(n-1)} + f'' g^{(n-2)} - f^{(3)} g^{(n-3)} + ... + (-1)^n f^{(n)} g is a constant. -/
theorem polynomial_expression_constant {f g : R[X]} (n : ℕ) (hf : f.degree = n) (hg : g.degree = n) :
  ∃ C : R, ∀ x, (f.coeff x * polynomial.derivative^[n] g.coeff x - 
                 polynomial.derivative f.coeff x * polynomial.derivative^[n-1] g.coeff x + 
                 polynomial.derivative^[2] f.coeff x * polynomial.derivative^[n-2] g.coeff x - 
                 polynomial.derivative^[3] f.coeff x * polynomial.derivative^[n-3] g.coeff x + ... + 
                 (-1)^n * polynomial.derivative^[n] f.coeff x * g.coeff x = C) :=
sorry

end polynomial_expression_constant_l699_699062


namespace min_value_modulus_sub_l699_699648

open Complex

theorem min_value_modulus_sub : ∀ (z : ℂ), abs(z) = 1 → abs(z - (3 + 4 * complex.I)) ≥ 4 :=
by
  intros z h
  sorry

noncomputable def min_modulus_sub : ℂ → ℝ :=
  λ z, if abs(z) = 1 then 4 else 0

end min_value_modulus_sub_l699_699648


namespace num_proper_subsets_of_intersection_l699_699658

def A : Set ℕ := {2, 3, 5, 7, 9}
def B : Set ℕ := {1, 2, 3, 5, 7}

theorem num_proper_subsets_of_intersection : 
  (Finset.card ((A ∩ B).powerset \ {A ∩ B}) = 15) :=
by
  sorry

end num_proper_subsets_of_intersection_l699_699658


namespace solve_diamond_l699_699341

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end solve_diamond_l699_699341


namespace number_of_triangles_l699_699109

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l699_699109


namespace range_of_a_l699_699784

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699784


namespace intersecting_point_value_l699_699463

theorem intersecting_point_value
  (b a : ℤ)
  (h1 : a = -2 * 2 + b)
  (h2 : 2 = -2 * a + b) :
  a = 2 :=
by
  sorry

end intersecting_point_value_l699_699463


namespace rattlesnake_tail_percentage_difference_l699_699246

-- Definitions for the problem
def eastern_segments : Nat := 6
def western_segments : Nat := 8

-- The statement to prove
theorem rattlesnake_tail_percentage_difference :
  100 * (western_segments - eastern_segments) / western_segments = 25 := by
  sorry

end rattlesnake_tail_percentage_difference_l699_699246


namespace neznaika_made_mistake_l699_699843

-- Define the total digits used from 1 to N pages
def totalDigits (N : ℕ) : ℕ :=
  let single_digit_pages := min N 9
  let double_digit_pages := if N > 9 then N - 9 else 0
  single_digit_pages * 1 + double_digit_pages * 2

-- The main statement we want to prove
theorem neznaika_made_mistake : ¬ ∃ N : ℕ, totalDigits N = 100 :=
by
  sorry

end neznaika_made_mistake_l699_699843


namespace largest_term_k_l699_699011

-- Definition of the sequence
def seq (n : ℕ) : ℝ := n * (n + 4) * (2 / 3) ^ n

-- Conditions for the sequence term k being maximum
theorem largest_term_k :
  ∃ k : ℕ, (k ≥ 1) ∧ (seq k ≥ seq (k + 1)) ∧ (seq k ≥ seq (k - 1))
  → k = 4 :=
sorry

end largest_term_k_l699_699011


namespace negation_example_l699_699886

theorem negation_example :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - x₀ > 0) :=
by
  sorry

end negation_example_l699_699886


namespace carla_total_time_l699_699234

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end carla_total_time_l699_699234


namespace twenty_five_x_one_eq_375_add_twenty_five_x_l699_699270

theorem twenty_five_x_one_eq_375_add_twenty_five_x (x : ℝ) : 25^(x + 1) = 375 + 25^x → x = 2 / 3 :=
by
  intros h
  sorry

end twenty_five_x_one_eq_375_add_twenty_five_x_l699_699270


namespace iterative_average_diff_l699_699200

def iterative_average (s : List ℕ) : ℚ :=
  s.foldl (λ acc n, (acc + n) / 2) 0

theorem iterative_average_diff :
  let nums := [0, 1, 2, 3, 4, 5],
      max_val := iterative_average [5, 4, 3, 2, 1, 0],
      min_val := iterative_average [0, 1, 2, 3, 4, 5]
  in
  max_val - min_val = 2.5625 := by
  sorry

end iterative_average_diff_l699_699200


namespace jarris_expected_value_minimum_l699_699746

noncomputable def minimum_expected_value_of_k (r R : ℝ) (prob : ℕ → ℝ) (areas : ℕ → ℝ) : ℝ :=
  if h : r = 3 ∧ R = 10
  then 12
  else 0

theorem jarris_expected_value_minimum :
  minimum_expected_value_of_k 3 10 (λ i, (areas i) / (areas 1 + areas 2 + areas 3 + areas 4)) (λ i, areas i) = 12 :=
by sorry

end jarris_expected_value_minimum_l699_699746


namespace upper_limit_of_people_l699_699617

theorem upper_limit_of_people
  (T : ℕ)
  (h1 : 3 * T = 189)
  (h2 : 5 * T = 315)
  (h3 : T > 50) :
  (T < 126) → T = 125 :=
by 
  -- We need to find the upper limit that is less than 126
  let T := 126 - 1  -- since 125 is the largest number less than 126
  have h4 : T = 125, from sorry,
  exact h4

end upper_limit_of_people_l699_699617


namespace xavier_yvonne_not_zelda_prob_l699_699174

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end xavier_yvonne_not_zelda_prob_l699_699174


namespace prove_monotonic_increasing_range_l699_699805

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699805


namespace total_pictures_l699_699430

-- Conditions: Randy drew 5 pictures, Peter drew 3 more pictures than Randy.
-- Quincy drew 20 more pictures than Peter. 
-- We need to prove the total number of pictures they drew is 41.

theorem total_pictures : 
  let randy_pictures := 5 in
  let peter_pictures := randy_pictures + 3 in
  let quincy_pictures := peter_pictures + 20 in
  randy_pictures + peter_pictures + quincy_pictures = 41 := 
by 
  let randy_pictures := 5
  let peter_pictures := randy_pictures + 3
  let quincy_pictures := peter_pictures + 20
  show randy_pictures + peter_pictures + quincy_pictures = 41 from sorry

end total_pictures_l699_699430


namespace non_working_games_count_l699_699839

-- Definitions based on conditions
def total_games : Nat := 15
def total_earnings : Nat := 30
def price_per_game : Nat := 5

-- Definition to be proved
def working_games : Nat := total_earnings / price_per_game
def non_working_games : Nat := total_games - working_games

-- Statement to be proved
theorem non_working_games_count : non_working_games = 9 :=
by
  sorry

end non_working_games_count_l699_699839


namespace calc_expression_l699_699592

theorem calc_expression : 
  (Real.cos (60 * Real.pi / 180) - Real.pow 2 (-1) + Real.sqrt (Real.pow (-2) 2) - Real.pow (Real.pi - 3) 0) = 1 := 
by
  -- Definitions based on the conditions
  have h1 : Real.cos (60 * Real.pi / 180) = 1 / 2 := by sorry
  have h2 : Real.pow 2 (-1) = 1 / 2 := by sorry
  have h3 : Real.sqrt (Real.pow (-2) 2) = 2 := by sorry
  have h4 : Real.pow (Real.pi - 3) 0 = 1 := by sorry
  
  -- Using these definitions to simplify the main expression
  sorry

end calc_expression_l699_699592


namespace proof_problem_l699_699673

noncomputable def f (a x : ℝ) : ℝ := a^x
noncomputable def g (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem proof_problem (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_f : f a 2 = 9) : 
    g a (1/9) + f a 3 = 25 :=
by
  -- Definitions and assumptions based on the provided problem
  sorry

end proof_problem_l699_699673


namespace original_purchase_price_l699_699978

-- Define the conditions as constants
def commission_percent : ℝ := 0.06
def commission_amount : ℝ := 8880
def improvements_cost : ℝ := 20000
def appreciation_annual_rate : ℝ := 0.02
def holding_period_years : ℕ := 3
def transfer_tax_rate : ℝ := 0.02
def closing_costs : ℝ := 3000
def legal_fees : ℝ := 1200

-- Define the calculated values as constants
def selling_price : ℝ := commission_amount / commission_percent
def total_appreciation : ℝ := selling_price * (appreciation_annual_rate * holding_period_years)
def property_transfer_tax : ℝ := selling_price * transfer_tax_rate
def total_expenses : ℝ := commission_amount + improvements_cost + total_appreciation + property_transfer_tax + closing_costs + legal_fees

-- Prove the original purchase price
theorem original_purchase_price : 
  let original_price := selling_price - total_expenses
  original_price = 103080 := 
by 
  sorry

end original_purchase_price_l699_699978


namespace find_correct_sum_l699_699720

theorem find_correct_sum 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100)
  (h2 : ∃ (a' : ℕ), (nat.reverseDigits a = a') ∧ (a' * b + 35 = 226))
  : a * b + 35 = 54 :=
sorry

end find_correct_sum_l699_699720


namespace probability_negative_product_l699_699902

open BigOperators
open Finset

def set_of_integers : Finset ℤ := {-5, -8, 7, 4, -2, 3}

def count_combinations (n k : ℕ) : ℕ := (Finset.range n).powerset.filter (λ s, s.card = k).card

def count_favorable_outcomes (s : Finset ℤ) : ℕ :=
  let neg := s.filter (λ x, x < 0)
  let pos := s.filter (λ x, x > 0)
  count_combinations neg.card 1 * count_combinations pos.card 2 + 
  count_combinations neg.card 3

theorem probability_negative_product : 
  count_favorable_outcomes set_of_integers / count_combinations set_of_integers.card 3 = 1/2 :=
by sorry

end probability_negative_product_l699_699902


namespace calories_per_slice_l699_699216

theorem calories_per_slice (n k t c : ℕ) (h1 : n = 8) (h2 : k = n / 2) (h3 : k * c = t) (h4 : t = 1200) : c = 300 :=
by sorry

end calories_per_slice_l699_699216


namespace log_expression_val_l699_699510

theorem log_expression_val : [log 10 (3 * log 10 (3 * log 10 (1000)))]^2 = 0.2086 :=
by
  have h₁ : log 10 (1000) = 3 := sorry
  have h₂ : log 10 (9) ≈ 0.9542 := sorry
  have h₃ : log 10 (2.8626) ≈ 0.4567 := sorry
  sorry

end log_expression_val_l699_699510


namespace area_of_cylinder_section_l699_699888

theorem area_of_cylinder_section (r : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π/2) :
  ∃ A : ℝ, A = (π * r^2 / cos α) :=
sorry

end area_of_cylinder_section_l699_699888


namespace range_of_a_l699_699787

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699787


namespace intersection_of_planes_is_line_l699_699939

-- Define the conditions as Lean 4 statements
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y + z - 8 = 0
def plane2 (x y z : ℝ) : Prop := x - 2 * y - 2 * z + 1 = 0

-- Define the canonical form of the line as a Lean 4 proposition
def canonical_line (x y z : ℝ) : Prop := 
  (x - 3) / -4 = y / 5 ∧ y / 5 = (z - 2) / -7

-- The theorem to state equivalence between conditions and canonical line equations
theorem intersection_of_planes_is_line :
  ∀ (x y z : ℝ), plane1 x y z → plane2 x y z → canonical_line x y z :=
by
  intros x y z h1 h2
  -- TODO: Insert proof here
  sorry

end intersection_of_planes_is_line_l699_699939


namespace range_of_a_l699_699762

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699762


namespace expression_evaluation_l699_699220

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l699_699220


namespace problem_l699_699004

theorem problem (x : ℝ) (h : 81^4 = 27^x) : 3^(-x) = 1/(3^(16/3)) := by {
  sorry
}

end problem_l699_699004


namespace max_distance_between_points_on_spheres_l699_699157

-- Define the centers of the spheres
def center1 : ℝ × ℝ × ℝ := (-2, -10, 5)
def center2 : ℝ × ℝ × ℝ := (12, 8, -16)

-- Define the radii of the spheres
def radius1 : ℝ := 19
def radius2 : ℝ := 87

-- Define the distance function between two points in 3D space
def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2 + (a.3 - b.3) ^ 2)

-- The statement of our proof problem in Lean 4
theorem max_distance_between_points_on_spheres :
  ∀ (A B : ℝ × ℝ × ℝ), distance A center1 = radius1 →
                       distance B center2 = radius2 →
                       distance A B ≤ radius1 + distance center1 center2 + radius2 :=
by
  sorry

end max_distance_between_points_on_spheres_l699_699157


namespace correct_statement_h2o_eq_l699_699583

theorem correct_statement_h2o_eq
  (H2O_eq : ∀ (T : ℝ), ∃ ΔH > 0, H2O ∈ equilibrium ↔ (H2O ↔ H^+ + OH^-))
  (B_statement : ∀ (T : ℝ), increase_T (K_w (H2O, T)) ∧ decrease_T (pH (H2O, T))):
  (∀ (T : ℝ), H2O_eq T) →
  (∀ (T : ℝ), ΔH > 0) →
  (heating_increases_Kw_and_decreases_pH : ∀ (T : ℝ), B_statement T) :=
by
  sorry

end correct_statement_h2o_eq_l699_699583


namespace monotonic_intervals_extreme_value_closer_l699_699681

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → (∀ x : ℝ, x < Real.log a → f x a > f (x + 1) a) ∧ (∀ x : ℝ, x > Real.log a → f x a < f (x + 1) a)) :=
sorry

theorem extreme_value_closer (a : ℝ) :
  a > e - 1 →
  ∀ x : ℝ, x ≥ 1 → |Real.exp 1/x - Real.log x| < |Real.exp (x - 1) + a - Real.log x| :=
sorry

end monotonic_intervals_extreme_value_closer_l699_699681


namespace reflection_composition_equivalence_l699_699416

-- Definitions of our points and lines
constants (O : Point) 
constants (l m l1 m1 : Line)

-- Assumptions about the intersection point and rotation
axiom intersect_at_O : intersect l m O
axiom rotated_around_O : rotate_around l m O l1 m1

-- The theorem we want to prove
theorem reflection_composition_equivalence :
  composition_of_reflections l m = composition_of_reflections l1 m1 :=
by sorry

end reflection_composition_equivalence_l699_699416


namespace first_player_advantage_l699_699147

def strip_width (bar : ℕ × ℕ) : ℕ := 
   if bar.1 > bar.2 then 1 else 2

def remaining_bar (bar : ℕ × ℕ) (strip : ℕ) : ℕ × ℕ := 
   (bar.1 - strip, bar.2)

def game (bar : ℕ × ℕ) (first_player_turn : Bool) : ℕ := 
   if bar = (0, 0) then 0
   else if first_player_turn then 
     let strip := strip_width bar
     strip + game (remaining_bar bar strip) false
   else 
     let strip := strip_width bar
     game (remaining_bar bar strip) true

theorem first_player_advantage (initial_bar : ℕ × ℕ)
  (h_initial_bar : initial_bar = (9,6)) :
  ∃ p1 p2 : ℕ, p1 ≥ p2 + 6 ∧ game (8, 6) true = p1 + p2 :=
sorry

end first_player_advantage_l699_699147


namespace range_of_a_l699_699656

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x ∈ set.Icc 0 1, a ≥ real.exp x

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + a * x0 + 4 = 0

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) ↔ (e ≤ a ∧ a < 4) ∨ (a ≤ -4) :=
by
   sorry

end range_of_a_l699_699656


namespace problem1_problem2_l699_699197

open scoped Classical
open ProbabilityTheory

noncomputable def prob_second_level_after_3_shots : ℚ :=
  (2.choose 1 * (2/3) * (1/3) * (2/3))

theorem problem1 : prob_second_level_after_3_shots = 8 / 27 :=
by {
  -- calculation steps here.
  sorry
}

noncomputable def prob_selected : ℚ :=
  ((2/3)^3) + (3.choose 2 * (2/3)^2 * (1/3) * (2/3)) +
  (4.choose 2 * (2/3)^2 * (1/3)^2 * (2/3))

noncomputable def prob_selected_and_shoot_5_times : ℚ :=
  (4.choose 2 * (2/3)^2 * (1/3)^2 * (2/3))

theorem problem2 : prob_selected_and_shoot_5_times / prob_selected = 1 / 4 :=
by {
  -- calculation steps here.
  sorry
}

end problem1_problem2_l699_699197


namespace inverse_of_A_l699_699625

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := 
  !![3, 4; -2, 9]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/35, -4/35; 2/35, 3/35]

theorem inverse_of_A : A⁻¹ = A_inv :=
by
  sorry

end inverse_of_A_l699_699625


namespace trig_problem_l699_699659

variable {α β : ℝ}

theorem trig_problem
  (h1 : sin β = 1 / 3)
  (h2 : sin (α - β) = 3 / 5)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 7 / 9 ∧ sin α = (6 * real.sqrt 2 + 4) / 15 :=
sorry

end trig_problem_l699_699659


namespace area_at_10_l699_699906

noncomputable theory

variables (A B C : ℝ → ℝ × ℝ) -- positions of runners A, B, and C as functions of time
variable t : ℝ -- time variable
variable area : ℝ → ℝ  -- area of triangle ABC as a function of time

-- Initial conditions and properties
axiom runners_parallel_tracks_at_constant_speeds :
  ∃ a b c : ℝ × ℝ, 
    (∀ t, A t = a * t + A 0 ∧ B t = b * t + B 0 ∧ C t = c * t + C 0) ∧ 
    ∃ k : ℝ, (b - a) = k * (c - a) -- parallel velocity vectors, constant speeds
  
axiom initial_area :
  area 0 = 2

axiom area_after_5_seconds :
  area 5 = 3

-- Definition of area of triangle ABC at time t
def S (t : ℝ) : ℝ :=
  1 / 2 * abs ((B t - A t) × (C t - A t))

-- Problem statement: Prove the area at t = 10 can be 4 or 8
theorem area_at_10 (h₁ : ∀ t, area t = S t) : area 10 = 4 ∨ area 10 = 8 :=
  sorry

end area_at_10_l699_699906


namespace range_of_a_l699_699810

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699810


namespace katy_brownies_l699_699392

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l699_699392


namespace range_of_a_l699_699789

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699789


namespace weight_of_person_replaced_l699_699452

theorem weight_of_person_replaced (W_new : ℝ) (h1 : W_new = 74) (h2 : (W_new - W_old) = 9) : W_old = 65 := 
by
  sorry

end weight_of_person_replaced_l699_699452


namespace smallest_palindrome_x_l699_699161

-- Define what a palindrome is
def is_palindrome (n : Nat) : Prop :=
  let s := n.toString
  s == s.reverse

-- Condition: x must be positive
def positive (x : Nat) : Prop := x > 0

-- Prove that the smallest positive x such that x + 7890 is a palindrome is 338
theorem smallest_palindrome_x :
  ∃ (x : Nat), positive x ∧ is_palindrome (x + 7890) ∧ ∀ y, positive y ∧ is_palindrome (y + 7890) → x ≤ y :=
  Exists.intro 338
    (and.intro
      (Nat.zero_lt_succ 337) -- 338 > 0
      (and.intro
        (by
          let x := 338
          have h : x + 7890 = 8228 := rfl
          show is_palindrome 8228
          calc
            8228.toString = "8228" : rfl
            "8228".reverse = "8228" : rfl
        sorry) -- Proof of smallest x
  )

end smallest_palindrome_x_l699_699161


namespace trapezoid_area_of_hexagon_l699_699560

noncomputable def hexagonArea : ℝ := 72 * Real.sqrt 3

theorem trapezoid_area_of_hexagon { a : ℝ } (h : a = 12) :
  let side_length := a,
      midpoints_distance := 12 * Real.sqrt 3,
      height := 6,
      area := (midpoints_distance + midpoints_distance) / 2 * height
  in area = hexagonArea :=
by {
  -- Definitions
  let side_length := a,
      midpoints_distance := 12 * Real.sqrt 3,
      height := 6,
      area := (midpoints_distance + midpoints_distance) / 2 * height,

  -- Proof skipped
  sorry
}

end trapezoid_area_of_hexagon_l699_699560


namespace range_of_m_no_zeros_inequality_when_m_zero_l699_699285

-- Statement for Problem 1
theorem range_of_m_no_zeros (m : ℝ) (h : ∀ x : ℝ, (x^2 + m * x + m) * Real.exp x ≠ 0) : 0 < m ∧ m < 4 :=
sorry

-- Statement for Problem 2
theorem inequality_when_m_zero (x : ℝ) : 
  (x^2) * (Real.exp x) ≥ x^2 + x^3 :=
sorry

end range_of_m_no_zeros_inequality_when_m_zero_l699_699285


namespace sum_of_digits_greatest_prime_divisor_of_8191_is_10_l699_699589

-- Definitions of the conditions
def num := 8191
def form1 := 2^13 - 1
def form2 := 127 * 65
def is_prime_127 : Nat.Prime 127 := by norm_num -- Directly using the fact from mathlib

-- Function to compute the sum of digits of a number
def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + if n >= 10 then sum_of_digits (n / 10) else 0

-- Main theorem to prove
theorem sum_of_digits_greatest_prime_divisor_of_8191_is_10 :
  sum_of_digits (Nat.gcd num 127) = 10 :=
by
  -- Using the conditions defined above
  have h1: num = form1 := by norm_num
  have h2: num = form2 := by norm_num
  have h3: Nat.gcd num 127 = 127 := by
    rw [Nat.gcd_comm, ←form2]
    norm_num
  have sum_eq := sum_of_digits 127
  exact sum_eq
  sorry

end sum_of_digits_greatest_prime_divisor_of_8191_is_10_l699_699589


namespace find_smallest_positive_integer_l699_699266

noncomputable def smallest_positive_integer_sum_consecutive_integers : ℕ :=
  let t := 495 in
  t

theorem find_smallest_positive_integer
  (l m n : ℤ)
  (h1 : ∃ l, 495 = 9 * (l + 4))
  (h2 : ∃ m, 495 = 5 * (2 * m + 9))
  (h3 : ∃ n, 495 = 11 * (n + 5)) :
  smallest_positive_integer_sum_consecutive_integers = 495 :=
by
  sorry

end find_smallest_positive_integer_l699_699266


namespace reciprocal_sum_evaluation_l699_699131

-- Definition of the reciprocal sum operation
def reciprocal_sum (a b : ℝ) : ℝ := (1 / (1 / a + 1 / b))

-- Proof statement of the given problem
theorem reciprocal_sum_evaluation :
  reciprocal_sum 4 (reciprocal_sum 2 (reciprocal_sum 4 (reciprocal_sum 3 (
    reciprocal_sum 4 (reciprocal_sum 4 (reciprocal_sum 2 (
      reciprocal_sum 3 (reciprocal_sum 2 (reciprocal_sum 4 (
        reciprocal_sum 4 (reciprocal_sum 3 3))))))))))) = 3 / 7 := 
by
  sorry

end reciprocal_sum_evaluation_l699_699131


namespace GwenShelves_l699_699694

theorem GwenShelves (M : ℕ) :
  (∀ m : ℕ, 4 * m = 32 - 4 * 3 → M = m) :=
by
  intro m h
  have h1 : 4 * 3 = 12 := rfl
  have h2 : 32 - 12 = 20 := rfl
  have h3 : 4 * m = 20 := h
  have m_val : m = 5 := by linarith
  exact m_val

end GwenShelves_l699_699694


namespace distance_foci_ellipse_l699_699634

noncomputable def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

theorem distance_foci_ellipse (a b : ℝ) (h₁ : a = 8) (h₂ : b = 3) :
  distance_between_foci a b = 2 * Real.sqrt 55 :=
by
  rw [distance_between_foci, h₁, h₂]
  norm_num
  exact Real.sqrt_eq_rfl (by norm_num)

end distance_foci_ellipse_l699_699634


namespace train_speed_l699_699189

theorem train_speed
    (goods_speed_kmph : ℝ)
    (goods_length_m : ℝ)
    (pass_time_s : ℝ)
    (relative_speed : goods_speed_kmph = 108)
    (length : goods_length_m = 340)
    (time : pass_time_s = 8) :
  let goods_speed_mps := goods_speed_kmph * (1000 / 3600),
      V_mps := (340 / 8) - goods_speed_mps,
      V_kmph := V_mps * (3600 / 1000)
  in V_kmph = 45 :=
by
  sorry

end train_speed_l699_699189


namespace solve_system_of_equations_l699_699445

theorem solve_system_of_equations :
  ∀ (x y z : ℚ), 
    (x * y = x + 2 * y ∧
     y * z = y + 3 * z ∧
     z * x = z + 4 * x) ↔
    (x = 0 ∧ y = 0 ∧ z = 0) ∨
    (x = 25 / 9 ∧ y = 25 / 7 ∧ z = 25 / 4) := by
  sorry

end solve_system_of_equations_l699_699445


namespace last_digit_of_frac_l699_699927

noncomputable theory
open_locale classical

theorem last_digit_of_frac (N : ℤ) (hN : N = 2^15) :
  (∃ k : ℤ, (1 / (N : ℝ)) = (5^15) / 10^15 * 10^(-15 * k)) → last_digit((1 / (N : ℝ))) = 5 :=
by {
  sorry
}

end last_digit_of_frac_l699_699927


namespace number_of_triangles_l699_699103

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l699_699103


namespace meeting_point_jane_hector_l699_699387

theorem meeting_point_jane_hector
  (s : ℕ) -- Hector's speed in blocks per unit time
  (l w : ℕ) -- length and width of the rectangle in blocks
  (h_speed : s > 0)
  (j_speed : 2 * s > 0)
  (h_start := (0, 0)) -- Hector starts at corner A (0, 0)
  (j_start := (0, 0)) -- Jane starts at corner A (0, 0)
  (perimeter : 2 * l + 2 * w = 24)
  (meet_time : ∃ t : ℕ, s * t + 2 * s * t = 24) :
  (exists
    (t : ℕ),
    let h_distance := s * t,
    let j_distance := 2 * s * t,
    h_distance = 8 ∧ j_distance = 16 ∧
    (h_distance + j_distance) % (2 * l + 2 * w) = 0 ∧
    (h_distance mod (2 * l + 2 * w) = 8 ∨ j_distance mod (2 * l + 2 * w) = 16)) :=
sorry

end meeting_point_jane_hector_l699_699387


namespace sum_of_first_six_terms_l699_699293

theorem sum_of_first_six_terms
  (a : ℕ → ℝ) -- The geometric sequence
  (S : ℕ → ℝ) -- The sum of the first n terms of the sequence
  (x1 x3 : ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 * (2^(n : ℝ)))
  (h2 : S = λ n, a 0 * (\frac{1 - 2^(n)}{1 - 2}))
  (eq_roots : x1^2 - 5*x1 + 4 = 0 ∧ x3^2 - 5*x3 + 4 = 0)
  (h_increase : ∀ n m : ℕ, n < m → a n < a m)
  (h_a1 : x1 = a 0)
  (h_a3 : x3 = a 2) :
  S 5 = 63 :=
by
  sorry

end sum_of_first_six_terms_l699_699293


namespace select_coprime_and_divide_l699_699824

theorem select_coprime_and_divide (n : ℕ) (selected_numbers : Finset ℕ) (h₁ : selected_numbers.card = n + 1)
  (h₂ : ∀ x ∈ selected_numbers, 1 ≤ x ∧ x ≤ 2 * n) :
  (∃ a b ∈ selected_numbers, Nat.coprime a b) ∧ (∃ c d ∈ selected_numbers, c ∣ d) :=
by
  -- Here we would proceed with the proof using Lean tactics
  sorry

end select_coprime_and_divide_l699_699824


namespace sequence_nth_term_16_l699_699330

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end sequence_nth_term_16_l699_699330


namespace prove_monotonic_increasing_range_l699_699802

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699802


namespace smallest_three_digit_int_satisfies_conditions_l699_699933

theorem smallest_three_digit_int_satisfies_conditions :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (45 * n ≡ 135 [MOD 280]) ∧ (n ≡ 3 [MOD 7]) ∧ n = 115 :=
begin
  sorry
end

end smallest_three_digit_int_satisfies_conditions_l699_699933


namespace sin_angle_RPS_l699_699033

theorem sin_angle_RPS {RPQ RPS : ℝ} (h : Real.sin RPQ = 7 / 25) : 
  Real.sin RPS = 7 / 25 :=
begin
  sorry
end

end sin_angle_RPS_l699_699033


namespace quadratic_has_two_roots_l699_699742

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l699_699742


namespace height_of_water_in_cylinder_l699_699577

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem height_of_water_in_cylinder :
  let cone_radius := 10
      cone_height := 15
      cylinder_radius := 20 in
  volume_of_cylinder cylinder_radius 1.25 = volume_of_cone cone_radius cone_height :=
by
  let cone_volume := volume_of_cone 10 15
  let cylinder_height := 1.25
  let cylinder_volume := volume_of_cylinder 20 cylinder_height
  show cylinder_volume = cone_volume
  sorry

end height_of_water_in_cylinder_l699_699577


namespace quiz_competition_top_three_orders_l699_699572

theorem quiz_competition_top_three_orders :
  let participants := 4
  let top_positions := 3
  let permutations := (Nat.factorial participants) / (Nat.factorial (participants - top_positions))
  permutations = 24 := 
by
  sorry

end quiz_competition_top_three_orders_l699_699572


namespace sin_A_possible_values_l699_699365

theorem sin_A_possible_values (ABC : Triangle)
  (non_obtuse : ¬ obtuse ABC)
  (hAB_AC : ABC.A > ABC.C)
  (angle_B_45 : ABC.B.angle = 45)
  (O : Point := circumcenter ABC)
  (I : Point := incenter ABC)
  (h_OI_condition : real.sqrt 2 * distance O I = ABC.A - ABC.C) :
  sin ABC.A = real.sqrt 2 / 2 ∨ sin ABC.A = (1 / 2) * real.sqrt(4 * real.sqrt 2 - 2) := 
sorry

end sin_A_possible_values_l699_699365


namespace student_average_grade_l699_699172

theorem student_average_grade (points_first_year courses_first_year : ℕ) (avg_first_year : ℚ)
                              (points_second_year courses_second_year : ℕ) (avg_second_year : ℚ)
                              (h1 : points_first_year = 6) (h2 : avg_first_year = 100)
                              (h3 : points_second_year = 5) (h4 : avg_second_year = 50) :
  (points_first_year * avg_first_year + points_second_year * avg_second_year) / 
  (points_first_year + points_second_year) ≈ 77.3 :=
by
  sorry

end student_average_grade_l699_699172


namespace blood_donation_selections_l699_699540

theorem blood_donation_selections :
  let number_O := 18
  let number_A := 10
  let number_B := 8
  let number_AB := 3
  (number_O * number_A * number_B * number_AB = 4320) :=
by
  let number_O := 18
  let number_A := 10
  let number_B := 8
  let number_AB := 3
  show number_O * number_A * number_B * number_AB = 4320, by sorry

end blood_donation_selections_l699_699540


namespace carla_total_time_l699_699235

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end carla_total_time_l699_699235


namespace monotonic_increasing_range_l699_699774

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699774


namespace sum_of_four_digit_numbers_l699_699916

theorem sum_of_four_digit_numbers :
  (∑ x in (finset.range 10000).filter (λ n, 1000 < n ∧ n < 10000 ∧ ∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3]), x) = 179982 := 
sorry

end sum_of_four_digit_numbers_l699_699916


namespace true_statement_l699_699575

variables {Plane Line : Type}
variables (α β γ : Plane) (a b m n : Line)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect_line (p q : Plane) : Line := sorry

-- Given conditions for the problem
variables (h1 : (α ≠ β)) (h2 : (parallel α β))
variables (h3 : (intersect_line α γ = a)) (h4 : (intersect_line β γ = b))

-- Statement verifying the true condition based on the above givens
theorem true_statement : parallel a b :=
by sorry

end true_statement_l699_699575


namespace right_triangle_area_l699_699922

theorem right_triangle_area (a b c : ℕ) (h_right : c^2 = a^2 + b^2) (h_a : a = 40) (h_c : c = 41) : 
  1 / 2 * a * b = 180 := by
  have h_b_squared : b^2 = 81 :=
  calc
    b^2 = c^2 - a^2 : by rw [←h_right]
    ... = 41^2 - 40^2 : by rw [h_a, h_c]
    ... = 1681 - 1600 : rfl
    ... = 81 : rfl,
  have h_b : b = 9 := by 
    rw [eq_comm, Nat.pow_two, Nat.pow_eq_iff_eq] at h_b_squared,
    exact h_b_squared,
  simp [h_a, h_b],
  sorry

end right_triangle_area_l699_699922


namespace distance_to_origin_l699_699372

theorem distance_to_origin : 
  let x := 3
  let y := -2
  sqrt (x^2 + y^2) = sqrt 13 :=
by
  sorry

end distance_to_origin_l699_699372


namespace marian_balance_proof_l699_699833

noncomputable def marian_new_credit_card_balance (initial_balance groceries_cost clothes_cost electronics_cost groceries_discount clothes_discount electronics_discount gas_ratio return_amount monthly_interest_rate : ℝ) : ℝ :=
let groceries_discounted := groceries_cost * (1 - groceries_discount / 100)
let clothes_discounted := clothes_cost * (1 - clothes_discount / 100)
let electronics_discounted := electronics_cost * (1 - electronics_discount / 100)
let gas_cost := groceries_discounted * gas_ratio
let balance_before_interest := initial_balance + groceries_discounted + clothes_discounted + electronics_discounted + gas_cost - return_amount
let interest := balance_before_interest * (monthly_interest_rate / 100)
in balance_before_interest + interest

theorem marian_balance_proof : marian_new_credit_card_balance 126 60 80 120 10 15 5 0.5 45 1.5 = 349.16 := by
  -- Definitions
  let groceries_discounted := 60 * 0.90
  let clothes_discounted := 80 * 0.85
  let electronics_discounted := 120 * 0.95
  let gas_cost := groceries_discounted * 0.50
  let balance_before_interest := 126 + groceries_discounted + clothes_discounted + electronics_discounted + gas_cost - 45
  let interest := balance_before_interest * 0.015
  -- Calculation of the final balance
  have balance_after_interest := balance_before_interest + interest
  -- Final equality check
  rw [groceries_discounted, clothes_discounted, electronics_discounted, gas_cost, balance_before_interest, interest, balance_after_interest]
  norm_num
  exact rfl


end marian_balance_proof_l699_699833


namespace trig_identity_proof_l699_699142

-- Definitions of the angles and corresponding trigonometric functions
def angle_a : Real := 263
def angle_b : Real := 203
def angle_c : Real := 83
def angle_d : Real := 23

-- Definition of the problem statement to be proven
theorem trig_identity_proof :
  cos (angle_a * (π / 180)) * cos (angle_b * (π / 180)) + sin (angle_c * (π / 180)) * sin (angle_d * (π / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_proof_l699_699142


namespace factorial_equation_solution_l699_699305

noncomputable theory

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_equation_solution :
  ∀ (x y : ℕ), 0 < x → 0 < y → 20 * factorial x + 2 * factorial y = factorial (2 * x + y) → (x = 1 ∧ y = 2) :=
by
  assume x y hx hy hxy
  sorry

end factorial_equation_solution_l699_699305


namespace minimum_cost_of_20_oranges_l699_699549

-- Define the main conditions
def cost_of_4_oranges : ℕ := 12
def cost_of_7_oranges : ℕ := 21
def needed_oranges : ℕ := 20

-- Prove that the minimum cost of purchasing exactly 20 oranges is 60 cents
theorem minimum_cost_of_20_oranges : ∃ (x y : ℕ), 4 * x + 7 * y = 20 ∧ 12 * x + 21 * y = 60 :=
begin
  -- Introduction of values to prove existence
  use [5, 0],
  -- Provide proofs for the conditions
  split,
  { -- Proof for the exact number of oranges
    calc
      4 * 5 + 7 * 0 = 20 : by simp,
  },
  { -- Proof for the minimum cost
    calc
      12 * 5 + 21 * 0 = 60 : by simp,
  },
end

end minimum_cost_of_20_oranges_l699_699549


namespace unique_functional_equation_solution_l699_699622

theorem unique_functional_equation_solution (f : ℕ+ → ℕ+)
  (h : ∀ x y : ℕ+, f (x + y * f x) = x + f x * f y) :
  ∀ x : ℕ+, f x = x :=
by
  sorry

end unique_functional_equation_solution_l699_699622


namespace range_of_a_l699_699813

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699813


namespace monotonic_increasing_range_l699_699781

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699781


namespace total_pictures_l699_699431

-- Conditions: Randy drew 5 pictures, Peter drew 3 more pictures than Randy.
-- Quincy drew 20 more pictures than Peter. 
-- We need to prove the total number of pictures they drew is 41.

theorem total_pictures : 
  let randy_pictures := 5 in
  let peter_pictures := randy_pictures + 3 in
  let quincy_pictures := peter_pictures + 20 in
  randy_pictures + peter_pictures + quincy_pictures = 41 := 
by 
  let randy_pictures := 5
  let peter_pictures := randy_pictures + 3
  let quincy_pictures := peter_pictures + 20
  show randy_pictures + peter_pictures + quincy_pictures = 41 from sorry

end total_pictures_l699_699431


namespace sum_of_solutions_l699_699242

theorem sum_of_solutions : 
  let P (x : ℝ) := (x^2 - 6 * x + 3) ^ (x^2 - 7 * x + 12) = 1 in
  (∀ x, P x → x = 3 ∨ x = 4 ∨ x = 3 + Real.sqrt 7 ∨ x = 3 - Real.sqrt 7 ∨ x = 2)
  → (∑ x in {3, 4, 3 + Real.sqrt 7, 3 - Real.sqrt 7, 2}, x) = 19 :=
by
  sorry

end sum_of_solutions_l699_699242


namespace max_value_m_n_l699_699739

variables {A B C : Type} [InnerProductSpace ℝ A]

-- Define the conditions
def side_a := sqrt 3
def side_c_add_2sqrt3_cos_C (c b cosC : ℝ) : Prop := 
  c + 2 * sqrt 3 * cosC = 2 * b

-- Define the vectors and parameters
variables (O AB AC : A) (m n b c cosC : ℝ)

-- Define the problem
theorem max_value_m_n 
  (h1 : side_a = sqrt 3)
  (h2 : side_c_add_2sqrt3_cos_C c b cosC)
  (h3 : O = m • AB + n • AC) : 
  ∃ m n, m + n = 2 / 3 :=
sorry

end max_value_m_n_l699_699739


namespace savings_equal_in_days_l699_699040
-- Import required library

-- Define the conditions as hypotheses
theorem savings_equal_in_days :
  (12_000 + 300 * d = 4_000 + 500 * d) ->
  (d = 40) :=
by
  sorry

end savings_equal_in_days_l699_699040


namespace tim_younger_than_jenny_l699_699908

def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2
def combined_ages_rommel_jenny : ℕ := rommel_age + jenny_age
def uncle_age : ℕ := 2 * combined_ages_rommel_jenny
noncomputable def aunt_age : ℝ := (uncle_age + jenny_age : ℕ) / 2

theorem tim_younger_than_jenny : jenny_age - tim_age = 12 :=
by {
  -- Placeholder proof
  sorry
}

end tim_younger_than_jenny_l699_699908


namespace min_value_of_f_l699_699338

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (9 / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f x = 16 :=
by
  sorry

end min_value_of_f_l699_699338


namespace hyperbola_properties_l699_699309

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

theorem hyperbola_properties (P Q : ℝ × ℝ) (hp : hyperbola_equation P.1 P.2) (hq : hyperbola_equation Q.1 Q.2) 
  (hangle : angle P Q = 45) :
  ∃ c : ℝ, (1 / (dist 0 P)^4) + (1 / (dist 0 Q)^4) = c := 
begin
  use 1 / 4,
  sorry
end

end hyperbola_properties_l699_699309


namespace find_divisor_l699_699946

theorem find_divisor (d q r : ℕ) (h1 : d = 265) (h2 : q = 12) (h3 : r = 1) :
  ∃ x : ℕ, d = (x * q) + r ∧ x = 22 :=
by {
  sorry
}

end find_divisor_l699_699946


namespace part1_part2_l699_699817

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs(2 * x - 1) + abs(x + a)

-- Part (1)
theorem part1 (x : ℝ) : f x 1 ≥ 3 → (x ≥ 1 ∨ x ≤ -1) :=
by sorry

-- Part (2)
theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ abs (a - 1)) → a ≤ 1/4 :=
by sorry

end part1_part2_l699_699817


namespace number_of_triangles_l699_699110

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l699_699110


namespace bowler_overs_l699_699963

theorem bowler_overs (x : ℕ) (h1 : ∀ y, y ≤ 3 * x) 
                     (h2 : y = 10) : x = 4 := by
  sorry

end bowler_overs_l699_699963


namespace full_day_students_count_l699_699998

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l699_699998


namespace find_c_l699_699007

-- Given conditions:
def c_is_integer (c : ℤ) : Prop := 
  ∃ c : ℤ, c^3 + 3*c + 3*c⁻¹ + (c⁻¹)^3 = 8

-- Proof showing that c = 1
theorem find_c : ∀ (c : ℤ), c_is_integer c → c = 1 := 
by
  intros c hc
  sorry

end find_c_l699_699007


namespace expand_polynomial_product_l699_699619

variable (x : ℝ)

def P (x : ℝ) : ℝ := 5 * x ^ 2 + 3 * x - 4
def Q (x : ℝ) : ℝ := 6 * x ^ 3 + 2 * x ^ 2 - x + 7

theorem expand_polynomial_product :
  (P x) * (Q x) = 30 * x ^ 5 + 28 * x ^ 4 - 23 * x ^ 3 + 24 * x ^ 2 + 25 * x - 28 :=
by
  sorry

end expand_polynomial_product_l699_699619


namespace num_valid_sequences_is_12_l699_699053

-- Definitions of the vertices of the triangle
def T : List (ℝ × ℝ) := [(0, 0), (4, 0), (0, 3)]

-- Transformation Definitions
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- List of transformations
def transformations : List ((ℝ × ℝ) → (ℝ × ℝ)) := 
    [rotate90, rotate180, rotate270, reflectX, reflectY]

-- Function to apply a list of transformations to a point
def apply_transformations (p : ℝ × ℝ) (trans : List ((ℝ × ℝ) → (ℝ × ℝ))) : ℝ × ℝ :=
    trans.foldl (fun acc t => t acc) p

-- Function to check if a sequence of transformations returns T to its original position
def returns_to_original (trans : List ((ℝ × ℝ) → (ℝ × ℝ))) : Bool :=
    T.all (fun p => apply_transformations p trans ∈ T)

-- Number of valid sequences that return T to its original position
def count_valid_sequences : ℕ :=
    (List.permutationsN 3 transformations).count returns_to_original

theorem num_valid_sequences_is_12 : count_valid_sequences = 12 := 
by sorry

end num_valid_sequences_is_12_l699_699053


namespace probability_of_specific_combination_l699_699363

theorem probability_of_specific_combination :
  let shirts := 6
  let shorts := 8
  let socks := 7
  let total_clothes := shirts + shorts + socks
  let ways_total := Nat.choose total_clothes 4
  let ways_shirts := Nat.choose shirts 2
  let ways_shorts := Nat.choose shorts 1
  let ways_socks := Nat.choose socks 1
  let ways_favorable := ways_shirts * ways_shorts * ways_socks
  let probability := (ways_favorable: ℚ) / ways_total
  probability = 56 / 399 :=
by
  simp
  sorry

end probability_of_specific_combination_l699_699363


namespace hexagon_coloring_count_l699_699611

-- Defining the number of vertices for a hexagon
def vertices : ℕ := 6

-- Defining the number of available colors
def available_colors : ℕ := 7

-- We consider a function that counts the valid colorings of the hexagon
noncomputable def count_valid_colorings_hexagon (vertices : ℕ) (available_colors : ℕ) : ℕ :=
  if vertices = 6 then
    7 * 6 * 5 * 4 * 3 * 3
  else
    0 -- For non-hexagon shapes, we have a different problem

theorem hexagon_coloring_count : count_valid_colorings_hexagon vertices available_colors = 7560 :=
by {
  -- We simplify the problem given vertices and available colors
  -- And we need to show that the coloring count equals 7560
  simp [vertices, available_colors, count_valid_colorings_hexagon],
  norm_num,
}

end hexagon_coloring_count_l699_699611


namespace loads_ratio_l699_699915

noncomputable def loads_wednesday : ℕ := 6
noncomputable def loads_friday (T : ℕ) : ℕ := T / 2
noncomputable def loads_saturday : ℕ := loads_wednesday / 3
noncomputable def total_loads_week (T : ℕ) : ℕ := loads_wednesday + T + loads_friday T + loads_saturday

theorem loads_ratio (T : ℕ) (h : total_loads_week T = 26) : T / loads_wednesday = 2 := 
by 
  -- proof steps would go here
  sorry

end loads_ratio_l699_699915


namespace monotonic_intervals_inequality_holds_sum_bound_l699_699679

-- Problem 1
def f (x : ℝ) : ℝ := log x - x^2 + x + 1

theorem monotonic_intervals :
  (∀ x ∈ Ioo 0 1, f' x > 0) ∧ (∀ x ∈ Ioi 1, f' x < 0) := sorry

-- Problem 2
theorem inequality_holds (a : ℝ) (x : ℝ) (h : a ≥ 2) :
  f x < (a / 2 - 1) * x^2 + a * x := sorry

-- Problem 3
theorem sum_bound (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h : f x₁ + f x₂ + 2 * (x₁^2 + x₂^2) + x₁ * x₂ - 2 = 0) :
  (x₁ + x₂) ≥ (sqrt 5 - 1) / 2 := sorry

end monotonic_intervals_inequality_holds_sum_bound_l699_699679


namespace axis_of_symmetry_condition_l699_699883

theorem axis_of_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
    (h_sym : ∀ x y, y = -x → y = (p * x + q) / (r * x + s)) : p = s :=
by
  sorry

end axis_of_symmetry_condition_l699_699883


namespace base8_subtraction_works_l699_699255

-- Definitions based on conditions
def base8_sub (a b : Nat) := a - b -- This simple definition assumes borrowing logic is inherent in our Nat representation.
def convert_base_8_to_nat (n : Nat) := 
  let d := n / 100
  let rem1 := n % 100
  let c := rem1 / 10
  let b := rem1 % 10
  d*8^2 + c*8 + b

theorem base8_subtraction_works (a b : Nat) : 
  convert_base_8_to_nat a = 537 → 
  convert_base_8_to_nat b = 261 → 
  convert_base_8_to_nat (base8_sub a b) = 256 := by
  intros ha hb
  rw [←ha, ←hb]
  sorry

end base8_subtraction_works_l699_699255


namespace exists_set_1992_positive_integers_l699_699244

noncomputable def is_power (n : ℕ) : Prop :=
  ∃ (m k : ℕ), k ≥ 2 ∧ n = m^k

theorem exists_set_1992_positive_integers :
  ∃ (S : Finset ℕ), S.card = 1992 ∧
  ∀ (T : Finset ℕ), T ⊆ S → is_power (T.sum) :=
sorry

end exists_set_1992_positive_integers_l699_699244


namespace simplify_proof_l699_699859

noncomputable def simplify_expression : ℝ :=
  (1 / (Real.logBase 18 3 + 1/2)) + (1 / (Real.logBase 12 4 + 1/2)) + (1 / (Real.logBase 8 6 + 1/2))

theorem simplify_proof : simplify_expression = 3/2 :=
by
  -- The actual proof steps would go here, but we'll leave it as a placeholder with sorry
  sorry

end simplify_proof_l699_699859


namespace angle_sum_in_hexagon_l699_699026

theorem angle_sum_in_hexagon (P Q R s t : ℝ) 
    (hP: P = 40) (hQ: Q = 88) (hR: R = 30)
    (hex_sum: 6 * 180 - 720 = 0): 
    s + t = 312 :=
by
  have hex_interior_sum: 6 * 180 - 720 = 0 := hex_sum
  sorry

end angle_sum_in_hexagon_l699_699026


namespace two_digit_sum_of_original_and_reverse_l699_699456

theorem two_digit_sum_of_original_and_reverse
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9) -- a is a digit
  (h2 : 0 ≤ b ∧ b ≤ 9) -- b is a digit
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_of_original_and_reverse_l699_699456


namespace smallest_y_l699_699190

def x := 5 * 30 * 60

theorem smallest_y (y : ℕ) : 
  (y * x = 2^3 * 3^3 * 5^3 ↔ {y = 3 : ℕ}) :=
sorry

end smallest_y_l699_699190


namespace line_perpendicular_value_of_a_l699_699355

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l699_699355


namespace greatest_possible_before_third_wave_l699_699857

theorem greatest_possible_before_third_wave :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a + b + c = 35 ∧ c = 31 :=
by
  use 2, 2, 31
  split; apply Prime.prime_2
  split; apply Prime.prime_2
  split
  apply Prime.prime_31
  split
  norm_num
  exact eq.refl 31

end greatest_possible_before_third_wave_l699_699857


namespace cube_placement_count_l699_699479

theorem cube_placement_count : 
  (∃ r b g : ℕ, r + b + g = 10 ∧ r ≤ 7 ∧ b ≤ 3 ∧ g ≤ 9) ∧
  ∑ (r b : ℕ) in finset.range (8) ×ˢ finset.range (4), if (10 - r - b) ≤ 9 then 1 else 0 = 31 := 
sorry

end cube_placement_count_l699_699479


namespace lemon_heads_each_person_l699_699214

-- Define the constants used in the problem
def totalLemonHeads : Nat := 72
def numberOfFriends : Nat := 6

-- The theorem stating the problem and the correct answer
theorem lemon_heads_each_person :
  totalLemonHeads / numberOfFriends = 12 := 
by
  sorry

end lemon_heads_each_person_l699_699214


namespace bracelet_arrangements_l699_699028

theorem bracelet_arrangements (n : ℕ) (h : n = 8) : (nat.factorial n) / (n * 2) = 2520 := by
  sorry

end bracelet_arrangements_l699_699028


namespace not_second_year_students_percentage_l699_699367

theorem not_second_year_students_percentage :
  (third_year_students not_third_year_second_year_students : ℝ)
  (h₁ : third_year_students = 50 / 100)
  (h₂ : not_third_year_second_year_students = 2 / 3 * (1 - third_year_students)) :
  1 - not_third_year_second_year_students = 66.66666666666667 / 100 :=
by
  sorry

end not_second_year_students_percentage_l699_699367


namespace question_equiv_answer_l699_699866

open Nat Set

-- Definitions based on the given conditions
def isClosedUnderAddition (S : Set ℕ) : Prop :=
  ∀ a b ∈ S, a + b ∈ S

noncomputable def finiteOmission (S : Set ℕ) : Prop :=
  ∃ (N : Finset ℕ), ∀ n ∈ S, n ∈ N ∨ (n ∈ S ∧ ∀ k, k < n → k ∉ S)

theorem question_equiv_answer {S : Set ℕ} (h_finiteOmission : finiteOmission S)
    (h_closedAddition : isClosedUnderAddition S) (k : ℕ) (h_k_in_S : k ∈ S) :
  ∃ (n : ℕ), n = k ∧ ∃ (N : Finset ℕ), ∀ x ∈ S, (x - k ∉ S → x ∈ N) ∧ N.card = k :=
sorry

end question_equiv_answer_l699_699866


namespace find_x_point_on_circle_l699_699731

theorem find_x_point_on_circle :
  ∃ x : ℝ, (let cx := 8 in let cy := 0 in let r := 15 in (x - cx)^2 + (10 - cy)^2 = r^2) ∧ 
  (x = 8 + 5 * Real.sqrt 5 ∨ x = 8 - 5 * Real.sqrt 5) := sorry

end find_x_point_on_circle_l699_699731


namespace tangent_line_eqn_l699_699381

theorem tangent_line_eqn {f : ℝ → ℝ} (h1 : ∀ x, f(x) = Real.log x) :
  tangent_line_eqn_at f e = fun x => (1 / e) * x :=
by
  sorry

end tangent_line_eqn_l699_699381


namespace percentage_fescue_in_Y_l699_699439

-- Define the seed mixtures and their compositions
structure SeedMixture :=
  (ryegrass : ℝ)  -- percentage of ryegrass

-- Seed mixture X
def X : SeedMixture := { ryegrass := 0.40 }

-- Seed mixture Y
def Y : SeedMixture := { ryegrass := 0.25 }

-- Mixture of X and Y contains 32 percent ryegrass
def mixture_percentage := 0.32

-- 46.67 percent of the weight of this mixture is X
def weight_X := 0.4667

-- Question: What percent of seed mixture Y is fescue
theorem percentage_fescue_in_Y : (1 - Y.ryegrass) = 0.75 := by
  sorry

end percentage_fescue_in_Y_l699_699439


namespace visitors_on_rachel_day_l699_699203

theorem visitors_on_rachel_day :
  ∀ (T P V : ℕ), T = 829 → P = 246 → V = T - P → V = 583 :=
by
  intros T P V hT hP hV
  rw [hT, hP] at hV
  rw [Nat.sub_eq_iff_eq_add.mpr rfl, Nat.add_sub_cancel] at hV
  exact hV
-- sorry

end visitors_on_rachel_day_l699_699203


namespace repair_roads_exists_l699_699143

open Set

-- Define the type of vertices and the type of edges
def Vertex : Type := Fin 100

-- Define a graph as a finite graph with 100 vertices
structure Graph (V : Type) :=
(E : Set (V × V))
(connected : ∀ u v : V, u ≠ v → u ∈ E ∧ v ∈ E)

-- Theorem to be proven
theorem repair_roads_exists :
  ∃ (G : Graph Vertex) (H : Graph Vertex),
    (∀ v : Vertex, even (degree H v)) :=
sorry

end repair_roads_exists_l699_699143


namespace like_terms_monomials_l699_699714

theorem like_terms_monomials (a b : ℕ) (x y : ℝ) (c : ℝ) (H1 : x^(a+1) * y^3 = c * y^b * x^2) : a = 1 ∧ b = 3 :=
by
  -- Proof will be provided here
  sorry

end like_terms_monomials_l699_699714


namespace cyclic_quadrilateral_inradii_l699_699519

-- Defining the cyclic quadrilateral and the inradii of the triangles formed.
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (r1 r2 : ℝ)

-- The statement proving that the sum of the inradii does not depend on the diagonal chosen.
theorem cyclic_quadrilateral_inradii (A₁ A₂ A₃ A₄ : A)
  (in_circle : ∃ (O : A) (R : ℝ), 
        dist O A₁ = R ∧ dist O A₂ = R ∧
        dist O A₃ = R ∧ dist O A₄ = R)
    (r1_triangle1 : ∃ (r1 : ℝ), 
        ∃ (A₁A₂A₃_inscribed : ∃ I₁ : A, 
            ∃ I₁_A₁ = I₁_A₂ ∧ I₁_A₁ = I₁_A₃ ∧ dist I₁ A₁ = r1 ∧ 
            dist I₁ A₂ = r1 ∧ dist I₁ A₃ = r1)
    (r2_triangle2 : ∃ (r2 : ℝ), 
        ∃ (A₃A₄A₁_inscribed : ∃ I₂ : A, 
            ∃ I₂_A₃ = I₂_A₄ ∧ I₂_A₃ = I₂_A₁ ∧ dist I₂ A₃ = r2 ∧ 
            dist I₂ A₄ = r2 ∧ dist I₂ A₁ = r2)
    ) : (r1 + r2 = r1 + r2) := 
sorry

end cyclic_quadrilateral_inradii_l699_699519


namespace sum_of_roots_100a_plus_b_l699_699065

theorem sum_of_roots_100a_plus_b :
  let solutions := { (x, y) | (x^2 + y^2)^6 = (x^2 - y^2)^4 ∧ (x^2 - y^2)^4 = (2*x^3 - 6*x*y^2)^3 } in
  let distinct_solutions := { (x, y) ∈ solutions | ∀ (x', y') ∈ solutions, (x, y) = (x', y') → (x, y) = (x, y) } in
  let sum := ∑ (x, y) in distinct_solutions, x + y in
  ∃ a b : ℕ, nat.coprime a b ∧ sum = (a:ℚ) / (b:ℚ) ∧ 100 * a + b = 516 :=
sorry

end sum_of_roots_100a_plus_b_l699_699065


namespace solve_for_a_b_and_extrema_l699_699661

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end solve_for_a_b_and_extrema_l699_699661


namespace xyz_inequality_l699_699851

theorem xyz_inequality (x y z : ℝ) (h_condition : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end xyz_inequality_l699_699851


namespace sum_of_integers_l699_699877

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 :=
by
  sorry

end sum_of_integers_l699_699877


namespace cube_has_max_volume_l699_699084

theorem cube_has_max_volume (x y z a : ℝ) (h : x + y + z = a) : 
  x * y * z ≤ (a / 3) ^ 3 :=
begin
  sorry,
end

end cube_has_max_volume_l699_699084


namespace unique_function_solution_l699_699256

variable (f : ℝ → ℝ)

theorem unique_function_solution :
  (∀ x y : ℝ, f (f x - y^2) = f x ^ 2 - 2 * f x * y^2 + f (f y))
  → (∀ x : ℝ, f x = x^2) :=
by
  sorry

end unique_function_solution_l699_699256


namespace centimeters_per_inch_l699_699423

theorem centimeters_per_inch (miles_per_map_inch : ℝ) (cm_measured : ℝ) (approx_miles : ℝ) (miles_per_inch : ℝ) (inches_from_cm : ℝ) : 
  miles_per_map_inch = 16 →
  inches_from_cm = 18.503937007874015 →
  miles_per_map_inch = 24 / 1.5 →
  approx_miles = 296.06299212598424 →
  cm_measured = 47 →
  (cm_measured / inches_from_cm) = 2.54 :=
by
  sorry

end centimeters_per_inch_l699_699423


namespace inequality_proof_l699_699528

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y * z) + y / (1 + z * x) + z / (1 + x * y)) ∧
  (x / (1 + y * z) + y / (1 + z * x) + z / (1 + x * y)) ≤ √2 := by
  sorry

end inequality_proof_l699_699528


namespace trig_identity_l699_699268

noncomputable def sin_40 := Real.sin (40 * Real.pi / 180)
noncomputable def tan_10 := Real.tan (10 * Real.pi / 180)
noncomputable def sqrt_3 := Real.sqrt 3

theorem trig_identity : sin_40 * (tan_10 - sqrt_3) = -1 := by
  sorry

end trig_identity_l699_699268


namespace four_digit_positive_integers_count_l699_699697

theorem four_digit_positive_integers_count :
  let p := 17
  let a := 4582 % p
  let b := 902 % p
  let c := 2345 % p
  ∃ (n : ℕ), 
    (1000 ≤ 14 + p * n ∧ 14 + p * n ≤ 9999) ∧ 
    (4582 * (14 + p * n) + 902 ≡ 2345 [MOD p]) ∧ 
    n = 530 := sorry

end four_digit_positive_integers_count_l699_699697


namespace find_a_l699_699459

-- Define the function f(x)
def f (a : ℚ) (x : ℚ) : ℚ := x^2 + (2 * a + 3) * x + (a^2 + 1)

-- State that the discriminant of f(x) is non-negative
def discriminant_nonnegative (a : ℚ) : Prop :=
  let Δ := (2 * a + 3)^2 - 4 * (a^2 + 1)
  Δ ≥ 0

-- Final statement expressing the final condition on a and the desired result |p| + |q|
theorem find_a (a : ℚ) (p q : ℤ) (h_relprime : Int.gcd p q = 1) (h_eq : a = -5 / 12) (h_abs : p * q = -5 * 12) :
  discriminant_nonnegative a →
  |p| + |q| = 17 :=
by sorry

end find_a_l699_699459


namespace complement_intersection_l699_699332

open Set

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N = {-2, 2}) :=
by sorry

end complement_intersection_l699_699332


namespace comic_books_collection_l699_699043

theorem comic_books_collection (initial_ky: ℕ) (rate_ky: ℕ) (initial_la: ℕ) (rate_la: ℕ) (months: ℕ) :
  initial_ky = 50 → rate_ky = 1 → initial_la = 20 → rate_la = 7 → months = 33 →
  initial_la + rate_la * months = 3 * (initial_ky + rate_ky * months) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end comic_books_collection_l699_699043


namespace time_to_nth_mile_eq_l699_699191

variable (n : ℕ)

def speed (d : ℕ) : ℝ := 1 / (d - 1 : ℝ)

def time_to_travel_nth_mile (t : ℕ → ℝ) :=
  ∃ c : ℝ, (∀ d ≥ 2, t d = c * (d - 1)) ∧ t 2 = 2

theorem time_to_nth_mile_eq (t : ℕ → ℝ): time_to_travel_nth_mile t → t n = 2 * (n - 1) :=
by
  intros h
  obtain ⟨c, hc, h2⟩ := h
  rw h2 at hc
  rw hc (2 : ℕ) (by norm_num : 2 ≥ 2)
  have c_eq : c = 2 := by linarith
  specialize hc n 
  by_cases hn: n < 2
  { sorry }
  { simp [c_eq, hc (by linarith)] }
  sorry

end time_to_nth_mile_eq_l699_699191


namespace problem_statement_l699_699299

variables (C : set (ℝ × ℝ)) (a b : ℝ) (A B P Q M : ℝ × ℝ)

noncomputable def ellipse_eq := ∀ p : ℝ × ℝ, p ∈ C ↔ p.1^2 / a^2 + p.2^2 / b^2 = 1
noncomputable def ecc := a > b ∧ b > 0 ∧ a^2 = b^2 + (a * √(3) / 2)^2

noncomputable def vertices := A = (-a, 0) ∧ B = (a, 0)
noncomputable def P_conditions := P ≠ A ∧ P ≠ B ∧ P ∈ C
noncomputable def triangle_isosceles := let area (X Y Z : ℝ × ℝ) := |X.1 * (Y.2 - Z.2) + Y.1 * (Z.2 - X.2) + Z.1 * (X.2 - Y.2)| / 2 in
  (area P A B = 2) ∧ (dist P A = dist P B ∨ dist P B = dist P A)

noncomputable def collinear (X Y Z : ℝ × ℝ) := (Y.2 - X.2) * (Z.1 - X.1) = (Z.2 - X.2) * (Y.1 - X.1)
noncomputable def intersection_M := collinear A P M ∧ M.1 = 4
noncomputable def Q_conditions := Q ∈ C ∧ collinear M B Q

noncomputable def passes_through_fixed_point (X Y : ℝ × ℝ) := ∃ k : ℝ, Y.2 = k * (Y.1 - 1)

theorem problem_statement :
  ecc a b →
  ellipse_eq C a b →
  vertices A B →
  P_conditions P A B C →
  triangle_isosceles P A B →
  intersection_M A P M →
  Q_conditions Q M B C →
  passes_through_fixed_point P Q :=
sorry

end problem_statement_l699_699299


namespace permutation_modulo_1000_l699_699755

theorem permutation_modulo_1000:
  let N := ∑ k in Finset.range 3, Nat.choose 4 k * Nat.choose 5 (5 - k) * Nat.choose 6 (4 - k) * Nat.choose 2 k 
  N % 1000 = 715 :=
by
  sorry

end permutation_modulo_1000_l699_699755


namespace imaginary_part_of_complex_number_l699_699128

-- Define the imaginary unit 'i' with its properties
def i := Complex.I

-- Define the given complex number
def complex_number := (1 + i^3) / (1 + i)

-- State the theorem to prove the imaginary part of the complex_number is -1
theorem imaginary_part_of_complex_number : complex.im (complex_number) = -1 := by
  sorry

end imaginary_part_of_complex_number_l699_699128


namespace largest_n_divisible_l699_699156

theorem largest_n_divisible (n : ℕ) (h : (n ^ 3 + 144) % (n + 12) = 0) : n ≤ 84 :=
sorry

end largest_n_divisible_l699_699156


namespace range_of_b_l699_699325

noncomputable def quadratic_f (b c : ℝ) : (ℝ → ℝ) :=
  λ x, x^2 + b*x + c

def A (b c : ℝ) : set ℝ :=
  {x | quadratic_f b c x = 0}

def B (b c : ℝ) : set ℝ :=
  {x | quadratic_f b c (quadratic_f b c x) = 0}

theorem range_of_b (b c : ℝ) :
  (∃ x₀ ∈ B b c, x₀ ∉ A b c) →
  (b < 0 ∨ b ≥ 4) :=
by
  sorry

end range_of_b_l699_699325


namespace radius_ratio_of_spheres_l699_699550

theorem radius_ratio_of_spheres
  (V_large : ℝ) (V_small : ℝ) (r_large r_small : ℝ)
  (h1 : V_large = 324 * π)
  (h2 : V_small = 0.25 * V_large)
  (h3 : (4/3) * π * r_large^3 = V_large)
  (h4 : (4/3) * π * r_small^3 = V_small) :
  (r_small / r_large) = (1/2) := 
sorry

end radius_ratio_of_spheres_l699_699550


namespace simplify_expression_l699_699862

theorem simplify_expression (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 :=
by
  -- We acknowledge this is the placeholder for the proof.
  -- This statement follows directly from the difference of squares identity.
  sorry

end simplify_expression_l699_699862


namespace equation_represents_two_rays_and_circle_l699_699125

theorem equation_represents_two_rays_and_circle :
  ∀ (x y : ℝ), (x + y - 2) * real.sqrt (x^2 + y^2 - 9) = 0 ↔ 
    ((x^2 + y^2 = 9) ∨ (x + y = 2 ∧ x^2 + y^2 ≥ 9)) :=
by
  sorry

end equation_represents_two_rays_and_circle_l699_699125


namespace katy_brownies_total_l699_699396

theorem katy_brownies_total : 
  (let monday_brownies := 5 in
   let tuesday_brownies := 2 * monday_brownies in
   let total_brownies := monday_brownies + tuesday_brownies in
   total_brownies = 15) := 
by 
  let monday_brownies := 5 in
  let tuesday_brownies := 2 * monday_brownies in
  let total_brownies := monday_brownies + tuesday_brownies in
  show total_brownies = 15 by
  sorry

end katy_brownies_total_l699_699396


namespace check_range_a_l699_699403

open Set

def A : Set ℝ := {x | x < -1/2 ∨ x > 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem check_range_a :
  (∃! x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ : ℝ) ∈ A ∩ B a ∧ (x₂ : ℝ) ∈ A ∩ B a) →
  a ∈ Icc (4/3 : ℝ) (15/8 : ℝ) :=
sorry

end check_range_a_l699_699403


namespace area_projection_equal_area_times_cos_angle_l699_699853

-- Defining our planar polygon and its orthogonal projection
structure PlanarPolygon :=
  (vertices : List (ℝ × ℝ)) -- This could be further specified; for now, list of 2D points

structure OrthogonalProjection (polygon : PlanarPolygon) :=
  (vertices : List (ℝ × ℝ)) -- Also a list of 2D points (projection)

def angle_between_planes (polygon_plane projection_plane : ℝ) : ℝ := sorry -- Placeholder for the angle calculation

noncomputable def area (polygon : PlanarPolygon) : ℝ := sorry -- Placeholder for polygon area calculation
noncomputable def projected_area (projection : OrthogonalProjection) : ℝ := sorry -- Placeholder for projected area calculation

theorem area_projection_equal_area_times_cos_angle (polygon : PlanarPolygon)
  (projection : OrthogonalProjection polygon)
  (φ : ℝ)
  (h_angle: φ = angle_between_planes (area polygon) (projected_area projection)) :
  projected_area projection = area polygon * Real.cos φ := 
sorry

end area_projection_equal_area_times_cos_angle_l699_699853


namespace problem_solution_l699_699236

theorem problem_solution : 25 * ((216 / 3) + (49 / 7) + (16 / 25) + 2) = 2041 :=
by
    have h1 : 25 * (216 / 3) = 1800 := by norm_num
    have h2 : 25 * (49 / 7) = 175 := by norm_num
    have h3 : 25 * (16 / 25) = 16 := by norm_num
    have h4 : 25 * 2 = 50 := by norm_num
    calc
    25 * ((216 / 3) + (49 / 7) + (16 / 25) + 2)
        = 25 * (216 / 3) + 25 * (49 / 7) + 25 * (16 / 25) + 25 * 2 : by ring
    ... = 1800 + 175 + 16 + 50 : by rw [h1, h2, h3, h4]
    ... = 2041 : by norm_num

end problem_solution_l699_699236


namespace cost_of_item_is_200_l699_699983

noncomputable def cost_of_each_item (x : ℕ) : ℕ :=
  let before_discount := 7 * x -- Total cost before discount
  let discount_part := before_discount - 1000 -- Part of the cost over $1000
  let discount := discount_part / 10 -- 10% of the part over $1000
  let after_discount := before_discount - discount -- Total cost after discount
  after_discount

theorem cost_of_item_is_200 :
  (∃ x : ℕ, cost_of_each_item x = 1360) ↔ x = 200 :=
by
  sorry

end cost_of_item_is_200_l699_699983


namespace parabola_y_intercepts_l699_699696

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem parabola_y_intercepts : 
  let a := 3
  let b := -4
  let c := 5 in
  discriminant a b c < 0 → 0 = 0 :=
by
  sorry

end parabola_y_intercepts_l699_699696


namespace sum_of_c_and_d_l699_699006

def digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem sum_of_c_and_d 
  (c d : ℕ)
  (hcd : digit c)
  (hdd : digit d)
  (h1: (4*c) * 5 % 10 = 5)
  (h2: 215 = (10 * (4*(d*5) + c*5)) + d*10 + 5) :
  c + d = 5 := 
  sorry

end sum_of_c_and_d_l699_699006


namespace probability_of_valid_triangle_l699_699195

noncomputable def prob_triangle_from_segments_in_15gon : ℚ :=
  321 / 455

theorem probability_of_valid_triangle : 
  let segments := (15 * (15 - 1)) / 2 in 
  let total_ways := Nat.choose segments 3 in
  let valid_triangle_count := (total_ways * prob_triangle_from_segments_in_15gon) in
  valid_triangle_count / total_ways = prob_triangle_from_segments_in_15gon :=
by
  sorry

end probability_of_valid_triangle_l699_699195


namespace distance_between_parallel_lines_l699_699487

theorem distance_between_parallel_lines (r d : ℝ)
  (h1 : ∀ (A B C : ℝ), A + B + C = 40 * 36 + 36 * 32 = 32 * 40)
  (h2 : ∀ (A B C : ℝ), A * 40 * A + B * 36 * B = B * 32 * B)
  (h3 : 400 + (5/4) * d^2 = r^2)
  (h4 : 256 + (36/4) * d^2 = r^2) :
  d = real.sqrt (576 / 31) :=
sorry

end distance_between_parallel_lines_l699_699487


namespace unchanged_cardinality_of_set_l699_699481

theorem unchanged_cardinality_of_set 
  (s : Finset ℝ) 
  (h_avg : (s.sum (λ x, x) / s.card) = 24) 
  (h_new_avg : (s.map (λ x, x * 5)).sum (λ x, x) / s.card = 120) : 
  (∃ n, s.card = n) :=
begin
  sorry
end

end unchanged_cardinality_of_set_l699_699481


namespace problem_1_problem_2_l699_699590

-- Problem 1: Proof
theorem problem_1 : ( ( Real.sqrt 8 - Real.sqrt 24 ) / Real.sqrt 2 + abs ( 1 - Real.sqrt 3 ) ) = 1 - Real.sqrt 3 := by
    sorry

-- Problem 2: Proof
theorem problem_2 : ( ( Real.pi - 3.14 ) ^ 0 + Real.sqrt ( (-2) ^ 2 ) - Real.cbrt ( -27 ) ) = 6 := by
    sorry

end problem_1_problem_2_l699_699590


namespace possible_values_g_l699_699819

def x_k (k : ℕ) : ℤ := (-1)^(k+1)

def g (n : ℕ) : ℚ :=
  if n = 0 then 0 else (∑ k in Finset.range n, x_k (k + 1)) / n

theorem possible_values_g (n : ℕ) (hn : (n > 0)) :
  (∃ k : ℚ, k ∈ {0, 1 / n} ∧ g n = k) :=
sorry

end possible_values_g_l699_699819


namespace topic_preference_order_l699_699362

noncomputable def astronomy_fraction := (8 : ℚ) / 21
noncomputable def botany_fraction := (5 : ℚ) / 14
noncomputable def chemistry_fraction := (9 : ℚ) / 28

theorem topic_preference_order :
  (astronomy_fraction > botany_fraction) ∧ (botany_fraction > chemistry_fraction) :=
by
  sorry

end topic_preference_order_l699_699362


namespace find_M_l699_699337

theorem find_M (a b M : ℝ) (h : (a + 2 * b)^2 = (a - 2 * b)^2 + M) : M = 8 * a * b :=
by sorry

end find_M_l699_699337


namespace partition_kingdom_into_republics_l699_699536

universe u

open scoped Classical

noncomputable section

/-- Definition of the kingdom's properties -/
structure Kingdom (α : Type u) :=
(cities : set α)
(highways : α → α → Prop)
(connected : ∀ a b ∈ cities, ∃ path : list α, path.head = a ∧ path.last = b ∧ ∀ p ∈ path, p ∈ cities ∧ ∀ p, p.tail.head_opt = some p → highways p.tail.head p.head)

/-- Definition of metropolises as a subset of cities -/
structure Rep (α : Type u) :=
(k : nat)
(metropolises : set α)
(h_metropolises : metropolises ⊆ (Kingdom.cities α))

/-- The proof problem statement -/
theorem partition_kingdom_into_republics {α : Type u} (kgdom : Kingdom α) (republics : Rep α)
  (h1 : republics.metropolises.card = republics.k) :
  ∃ (f : α → α), ∀ (c ∈ kgdom.cities), (f c ∈ republics.metropolises ∧ 
                                          ∀ (m' ∈ republics.metropolises), 
                                            shortest_path_length kgdom.highways c (f c) 
                                            ≤ shortest_path_length kgdom.highways c m') :=
sorry

end partition_kingdom_into_republics_l699_699536


namespace divisibility_condition_l699_699061

theorem divisibility_condition (a p q : ℕ) (hp : p > 0) (ha : a > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ↔ p ∣ a^q) :=
sorry

end divisibility_condition_l699_699061


namespace radius_of_third_circle_l699_699150

noncomputable def circle_radius {r1 r2 : ℝ} (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let A_shaded := (25^2 * Real.pi) - (15^2 * Real.pi)
  let r := Real.sqrt (A_shaded / Real.pi)
  r

theorem radius_of_third_circle (r1 r2 r3 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) :
  circle_radius h1 h2 = 20 :=
by 
  sorry

end radius_of_third_circle_l699_699150


namespace cone_on_sphere_l699_699295

theorem cone_on_sphere (R : ℝ) (h1 : ∃ (ABC : Triangle), ABC.is_equilateral ∧ ABC.on_sphere_surface R)
  (h2 : ∃ (S : Cone), S.volume = 16 * Real.sqrt 3 ∧ S.has_base ABC) :
  sphere_surface_area R = 64 * π :=
sorry

end cone_on_sphere_l699_699295


namespace area_of_convex_quad_a_add_b_add_c_l699_699727

open Real

def a := 585
def b := 20.25
def c := 3

theorem area_of_convex_quad :
  (PQ = 7) ∧ (QR = 3) ∧ (RS = 9) ∧ (SP = 9) ∧ (angle RSP = 60) ∧ is_convex PQRS →
  area PQRS = sqrt(a) + b * sqrt(c) :=
sorry

theorem a_add_b_add_c :
  (PQ = 7) ∧ (QR = 3) ∧ (RS = 9) ∧ (SP = 9) ∧ (angle RSP = 60) ∧ is_convex PQRS →
  a + b + c = 608.25 :=
sorry

end area_of_convex_quad_a_add_b_add_c_l699_699727


namespace range_of_a_l699_699782

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699782


namespace problem_correct_l699_699057

-- Definitions (conditions) directly appear in the problem condition a)
variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Conditions: vectors a and b are non-zero
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0

-- Conditions of the problem translated to Lean statements
def statement1 : Prop := (a ⬝ b) = 0 → ‖a + b‖ = ‖a - b‖
def statement4 : Prop := ‖a + b‖ = ‖a‖ - ‖b‖ → ∃ λ : ℝ, a = λ • b

-- The proof problem: proving B) corresponding to ① and ④
theorem problem_correct : statement1 ∧ statement4 :=
sorry

end problem_correct_l699_699057


namespace shift_graph_to_cos_equivalent_l699_699324

def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + (Real.pi / 8))

def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem shift_graph_to_cos_equivalent :
  ∀ (x : ℝ), ∃ d : ℝ,
    d = (3 * Real.pi / 16) ∧
    (∀ (ω : ℝ), ω = 2 →
    f ω x = g ω (x - d)) :=
by
  sorry

end shift_graph_to_cos_equivalent_l699_699324


namespace katy_brownies_l699_699391

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l699_699391


namespace range_of_b_l699_699068

noncomputable def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem range_of_b (b : ℝ) : (∀ n : ℕ, 0 < n → a_n (n+1) b > a_n n b) ↔ (-3 < b) :=
by
    sorry

end range_of_b_l699_699068


namespace students_both_fruits_l699_699023

theorem students_both_fruits (A B C : ℕ) (hA : A = 12) (hB : B = 8) (hC : C = 10) :
  ∃ D : ℕ, D = A + B - hC := by
  existsi (A + B - hC)
  sorry

end students_both_fruits_l699_699023


namespace find_y_given_conditions_l699_699702

theorem find_y_given_conditions : ∀ (x y : ℕ), (x^2 + x + 6 = y - 6) → (x = -5) → (y = 32) :=
by
  intros x y h1 h2
  sorry

end find_y_given_conditions_l699_699702


namespace age_of_James_when_Thomas_reaches_current_age_l699_699485
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l699_699485


namespace max_handshakes_l699_699895

theorem max_handshakes (n : ℕ) (m : ℕ)
  (h_n : n = 25)
  (h_m : m = 20)
  (h_mem : n - m = 5)
  : ∃ (max_handshakes : ℕ), max_handshakes = 250 :=
by
  sorry

end max_handshakes_l699_699895


namespace sum_inverse_combinations_le_one_sum_combinations_ge_m_squared_l699_699754

variables (n m : ℕ) (A : Finset (Fin n)) (A_i : Fin m → Finset (Fin n))

/-- Pairwise disjoint subsets condition --/
def pairwise_disjoint (A_i : Fin m → Finset (Fin n)) : Prop :=
  ∀ i j, i ≠ j → Disjoint (A_i i) (A_i j)

/-- Natural number combination function --/
noncomputable def C (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_inverse_combinations_le_one
  (hA : A.card = n) (hm : ∀ i, (A_i i).card = (A_i i).card) 
  (hdisjoint : pairwise_disjoint A_i) :
  (Finset.univ.sum (λ i, (1 : ℝ) / C n ((A_i i).card))) ≤ 1 :=
sorry

theorem sum_combinations_ge_m_squared
  (hA : A.card = n) (hm : ∀ i, (A_i i).card = (A_i i).card)
  (hdisjoint : pairwise_disjoint A_i) :
  (Finset.univ.sum (λ i, (C n ((A_i i).card) : ℕ))) ≥ m^2 :=
sorry

end sum_inverse_combinations_le_one_sum_combinations_ge_m_squared_l699_699754


namespace hyperbola_distance_vertices_l699_699260

theorem hyperbola_distance_vertices :
  let eq := 16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 60 = 0 in
  (distance_between_vertices eq) = 1 :=
by
  have h : eq = 16 * (x - 2)^2 - 4 * (y - 1)^2 = 0, from sorry,
  have eq_standard : (x - 2)^2 / (1 / 4) - (y - 1)^2 / 1 = 1, from sorry,
  have a2 : (1 / 4) > 0, from sorry,
  have b2 : 1 > 0, from sorry,
  have a : 1 / 2 = real.sqrt (1 / 4), from sorry,
  show 2 * (1 / 2) = 1, from sorry

end hyperbola_distance_vertices_l699_699260


namespace problem_l699_699003

theorem problem (x : ℝ) (h : 81^4 = 27^x) : 3^(-x) = 1/(3^(16/3)) := by {
  sorry
}

end problem_l699_699003


namespace expression_evaluation_l699_699222

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l699_699222


namespace find_second_number_l699_699139

theorem find_second_number (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4)
  (h3 : y / z = 4 / 7) :
  y = 240 / 7 :=
by sorry

end find_second_number_l699_699139


namespace relationship_bx_l699_699287

variable {a b t x : ℝ}

-- Given conditions
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : t > 0)
variable (h4 : a ^ x = a + t)

theorem relationship_bx (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a ^ x = a + t) : b ^ x > b + t :=
by
  sorry

end relationship_bx_l699_699287


namespace integral_of_sin_squared_l699_699587

theorem integral_of_sin_squared:
  ∫ (z : ℂ) in segment ℂ 0 complex.I, (sin z) ^ 2 = (complex.I / 4) * (2 - sinh 2) :=
by
  sorry

end integral_of_sin_squared_l699_699587


namespace monotonic_increasing_range_l699_699775

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699775


namespace number_of_triangles_l699_699102

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l699_699102


namespace numeral_system_is_perfect_square_l699_699385

def numeral_is_perfect_square (B : ℕ) : Prop :=
  ∃ n : ℕ, n^2 = B^4 + B^3 + B^2 + B + 1

theorem numeral_system_is_perfect_square : 
  numeral_is_perfect_square 3 :=
begin
  sorry
end

end numeral_system_is_perfect_square_l699_699385


namespace zero_to_any_power_l699_699591

theorem zero_to_any_power (n : ℕ) : (0 : ℝ) ^ n = 0 := by
  sorry

# Check for specific case where n=2014:
# theorem zero_to_pow_2014 : (0 : ℝ) ^ 2014 = 0 := sorry

end zero_to_any_power_l699_699591


namespace AD_bisects_angle_PAB_l699_699406

-- Definitions based on the conditions
variables {Point : Type} [planar : EuclideanGeometry] 
(open EuclideanGeometry)
variable (P Q A B C D E : Point)
variable (Γ ω : Circle)

-- Stating the conditions
axiom semicircle_with_diameter_PQ (hΓ : semicircle Γ P Q) :
  (∃ A B, is_perpendicular (line_segment P Q) (line_through P A) ∧ A ∈ Γ ∧ B ∈ (line_segment P Q) ∧ 
    (∃ ω C D E, tangent_to_circle ω Γ C ∧ tangent_to_line_segment ω (line_segment P B) D ∧ tangent_to_line_segment ω (line_segment A B) E))

-- The main theorem to be proved
theorem AD_bisects_angle_PAB (hΓ : semicircle Γ P Q) :
  bisect_angle (line_through A D) (angle P A B) :=
sorry

end AD_bisects_angle_PAB_l699_699406


namespace area_is_ln2_l699_699122

noncomputable def area_enclosed_by_curve := ∫ x in 1..2, 1/x

theorem area_is_ln2 : area_enclosed_by_curve = Real.log 2 := by
  sorry

end area_is_ln2_l699_699122


namespace exists_positive_integer_N_l699_699474

noncomputable def sequence (a : ℕ → ℝ) := ∀ n : ℕ, n ≥ 2 → a n - 2 = |a (n + 1)| - a n

theorem exists_positive_integer_N (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 2 → a n - 2 = |a (n + 1)| - a n) →
  ∃ N : ℕ, (N > 0) ∧ (∀ n : ℕ, n ≥ N → a (n - 9) = a n) :=
begin
  sorry
end

end exists_positive_integer_N_l699_699474


namespace rodney_probability_correct_l699_699436

def is_valid_guess (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧
  n / 10 % 2 = 1 ∧
  n % 10 % 2 = 1 ∧
  n > 75

def valid_guesses : List ℕ :=
  [77, 79, 91, 93, 95, 97, 99]

theorem rodney_probability_correct :
  (have_correct_guess : ∃ n, is_valid_guess n ∧ n ∈ valid_guesses) →
  (∃ correct_number, is_valid_guess correct_number ∧ ∀ n ∈ valid_guesses, n = correct_number → 1 / list.length valid_guesses = 1 / 7) :=
by
  sorry

end rodney_probability_correct_l699_699436


namespace digits_in_product_l699_699642

def num_digits (n : ℕ) : ℕ :=
  nat.log10 n + 1

theorem digits_in_product :
  num_digits (3659893456789325678 * 342973489379256) = 34 :=
by
  sorry

end digits_in_product_l699_699642


namespace lines_are_concurrent_l699_699429

variables (A B C D P O O1 O2 O3 O4 G : Type) [InCircle A B C D O] [DiagonalsIntersect A C B D P]
  [CircumcenterOfTriangle A B P O1] [CircumcenterOfTriangle B C P O2]
  [CircumcenterOfTriangle C D P O3] [CircumcenterOfTriangle D A P O4]

theorem lines_are_concurrent (h : ConcurrentLines O P O1 O3 O2 O4 G) : ConcurrentLines O P O1 O3 O2 O4 G :=
  sorry

end lines_are_concurrent_l699_699429


namespace pascal_triangle_parallelogram_sum_l699_699847

theorem pascal_triangle_parallelogram_sum (n k : ℕ) :
  let a := nat.choose n k in 
  a - 1 = (∑ i in finset.range n, ∑ j in finset.range k, if i + j < n + k ∧ i + j > 0 then nat.choose i j else 0) := by
  sorry

end pascal_triangle_parallelogram_sum_l699_699847


namespace number_of_integers_congruent_7_mod_9_lessthan_1000_l699_699000

theorem number_of_integers_congruent_7_mod_9_lessthan_1000 : 
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → 7 + 9 * n < 1000 → k + 1 = 111 :=
by
  sorry

end number_of_integers_congruent_7_mod_9_lessthan_1000_l699_699000


namespace distance_from_A_to_focus_l699_699300

def point : Type := ℝ × ℝ

def parabola (p : point) : Prop := p.snd^2 = 4 * p.fst

def focus_of_parabola : point := (1, 0)

def line_through (p q : point) : point → Prop :=
    λ r, (q.snd - p.snd) * (r.fst - p.fst) = (q.fst - p.fst) * (r.snd - p.snd)

def distance (p q : point) : ℝ := 
    real.sqrt ((p.fst - q.fst)^2 + (p.snd - q.snd)^2)

theorem distance_from_A_to_focus :
    ∀ (A B P : point),
    P = (-2, 0) →
        parabola A →
        parabola B →
        line_through P A A →
        line_through P A B →
        distance P A = (1/2) * distance A B →
        distance A focus_of_parabola = 5/3 :=
sorry

end distance_from_A_to_focus_l699_699300


namespace solve_inequality_1_range_of_t_l699_699663

-- Definitions
variable (f : ℝ → ℝ)
variable (x m n t a : ℝ)

-- Conditions
def is_odd_function_on_Icc := 
  ∀ x ∈ Icc (-1) 1, f (-x) = -f x

def condition_1 := 
  (m ∈ Icc (-1) 1) ∧ (n ∈ Icc (-1) 1) ∧ (m + n ≠ 0) ∧ ((f m + f n) / (m + n) < 0)

def function_bound := 
  ∀ x ∈ Icc (-1) 1, f x ≤ t^2 - 2 * a * t + 1

def max_f := f (-1) = 1

-- Question 1: Prove inequality
theorem solve_inequality_1
  (h_odd : is_odd_function_on_Icc f)
  (h_cond1 : ∀ m n, condition_1 f m n)
  (h_maxf : max_f f) :
  ∀ x, (1 / 4 < x ∧ x ≤ 1 / 2) → f (x + 1 / 2) < f (1 - x) := 
sorry

-- Question 2: Find range of t
theorem range_of_t
  (h_bound : function_bound f t a)
  (h_maxf : max_f f) :
  t ≤ -2 ∨ t ≥ 2 := 
sorry

end solve_inequality_1_range_of_t_l699_699663


namespace cesaro_sum_of_200_terms_l699_699316

theorem cesaro_sum_of_200_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h_sum_199 : (∑ k in finset.range 199, S k) / 199 = 500) :
  (∑ k in finset.range 200, if k = 0 then 2 else 2 + S (k - 1 - 1)) / 200 = 499.5 :=
by
  sorry

end cesaro_sum_of_200_terms_l699_699316


namespace sqrt_sum_of_eq_l699_699448

theorem sqrt_sum_of_eq (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
sorry

end sqrt_sum_of_eq_l699_699448


namespace average_of_first_20_even_numbers_not_divisible_by_3_or_5_l699_699154

def first_20_valid_even_numbers : List ℕ :=
  [2, 4, 8, 14, 16, 22, 26, 28, 32, 34, 38, 44, 46, 52, 56, 58, 62, 64, 68, 74]

-- Check the sum of these numbers
def sum_first_20_valid_even_numbers : ℕ :=
  first_20_valid_even_numbers.sum

-- Define average calculation
def average_first_20_valid_even_numbers : ℕ :=
  sum_first_20_valid_even_numbers / 20

theorem average_of_first_20_even_numbers_not_divisible_by_3_or_5 :
  average_first_20_valid_even_numbers = 35 :=
by
  sorry

end average_of_first_20_even_numbers_not_divisible_by_3_or_5_l699_699154


namespace minimum_number_of_tiles_l699_699171

noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def courtyard_length_cm : ℕ := 378
def courtyard_breadth_cm : ℕ := 525
def square_tile_side_cm : ℕ := gcd courtyard_length_cm courtyard_breadth_cm

def area_of_tile (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_of_courtyard (length breadth : ℕ) : ℕ :=
  length * breadth

def num_tiles_needed (total_area tile_area : ℕ) : ℕ :=
  total_area / tile_area

theorem minimum_number_of_tiles : num_tiles_needed (area_of_courtyard courtyard_length_cm courtyard_breadth_cm) (area_of_tile square_tile_side_cm) = 450 := 
sorry

end minimum_number_of_tiles_l699_699171


namespace probability_sum_10_with_replacement_probability_sum_10_without_replacement_l699_699146

-- Define the problem conditions
def discs_basket := {unmarked := 5, marked := [1, 2, 3, 4, 5]}
def draws_with_replacement := 3
def draws_without_replacement := 3

-- Define the proof statements
theorem probability_sum_10_with_replacement : 
  (∑ i in (finset.range (10)), if i ∈ {0, 1, 2, 3, 4, 5} then 1 else 0)^3 / (10^3) = 0.033 := 
by sorry

theorem probability_sum_10_without_replacement : 
  (∑ i in (finset.range (10)), if i ∈ {1, 2, 3, 4, 5} then 1 else 0) 
  * (∑ i in (finset.erase (finset.range (10)) 1), if i ∈ {2, 3, 4, 5} then 1 else 0)
  * (∑ i in (finset.erase (finset.erase (finset.range (10)) 1) 2), if i ∈ {3, 4, 5} then 1 else 0) 
  / (10 * 9 * 8) = 0.017 :=
by sorry

end probability_sum_10_with_replacement_probability_sum_10_without_replacement_l699_699146


namespace contrapositive_statement_l699_699875

-- Condition definitions
def P (x : ℝ) := x^2 < 1
def Q (x : ℝ) := -1 < x ∧ x < 1
def not_Q (x : ℝ) := x ≤ -1 ∨ x ≥ 1
def not_P (x : ℝ) := x^2 ≥ 1

theorem contrapositive_statement (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_statement_l699_699875


namespace ratio_of_shares_l699_699566

-- Definitions for the given conditions
def capital_A : ℕ := 4500
def capital_B : ℕ := 16200
def months_A : ℕ := 12
def months_B : ℕ := 5 -- B joined after 7 months

-- Effective capital contributions
def effective_capital_A : ℕ := capital_A * months_A
def effective_capital_B : ℕ := capital_B * months_B

-- Defining the statement to prove
theorem ratio_of_shares : effective_capital_A / Nat.gcd effective_capital_A effective_capital_B = 2 ∧ effective_capital_B / Nat.gcd effective_capital_A effective_capital_B = 3 := by
  sorry

end ratio_of_shares_l699_699566


namespace part_a_part_b_l699_699651

-- Define the function d(t)
def d (t : ℝ) (xs : List ℝ) : ℝ :=
  (minList (List.map (λ x => |x - t|) xs) + maxList (List.map (λ x => |x - t|) xs)) / 2

-- Prove that d(t) does not take its minimum value at a unique point for any set of numbers xs
theorem part_a (xs : List ℝ) : ¬∃! t, d t xs = List.minimum (List.map (λ t => d t xs) xs) :=
sorry

-- Define the mid-point c
def c (xs : List ℝ) : ℝ :=
  (List.minimum xs + List.maximum xs) / 2

-- Define the median m (assuming a helper function that calculates the median exists)
def median (xs : List ℝ) : ℝ :=
  sorry -- Placeholder for median calculation

-- Prove d(c) ≤ d(m)
theorem part_b (xs : List ℝ) : d (c xs) xs ≤ d (median xs) xs :=
sorry


end part_a_part_b_l699_699651


namespace real_part_of_z_is_neg3_l699_699317

noncomputable def z : ℂ := (1 + 2 * Complex.I) ^ 2

theorem real_part_of_z_is_neg3 : z.re = -3 := by
  sorry

end real_part_of_z_is_neg3_l699_699317


namespace percentage_running_wickets_l699_699944

-- Conditions provided as definitions and assumptions in Lean
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def boundary_runs (b : ℕ) := b * 4
def six_runs (s : ℕ) := s * 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries := boundary_runs boundaries
def runs_from_sixes := six_runs sixes
def runs_not_from_boundaries_and_sixes := total_runs - (runs_from_boundaries + runs_from_sixes)

-- Proof that the percentage of the total score by running between the wickets is 50%
theorem percentage_running_wickets :
  (runs_not_from_boundaries_and_sixes : ℝ) / (total_runs : ℝ) * 100 = 50 :=
by
  sorry

end percentage_running_wickets_l699_699944


namespace quadratic_has_two_roots_l699_699741

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l699_699741


namespace column_of_1000_is_C_l699_699573

def column_of_integer (n : ℕ) : String :=
  ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"].get! ((n - 2) % 10)

theorem column_of_1000_is_C :
  column_of_integer 1000 = "C" :=
by
  sorry

end column_of_1000_is_C_l699_699573


namespace triangles_from_ten_points_l699_699117

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l699_699117


namespace max_candy_one_student_l699_699542

theorem max_candy_one_student (n : ℕ) (mu : ℕ) (at_least_two : ℕ → Prop) :
  n = 35 → mu = 6 →
  (∀ x, at_least_two x → x ≥ 2) →
  ∃ max_candy : ℕ, (∀ x, at_least_two x → x ≤ max_candy) ∧ max_candy = 142 :=
by
sorry

end max_candy_one_student_l699_699542


namespace area_of_field_l699_699194

theorem area_of_field (L W A : ℕ) (h₁ : L = 20) (h₂ : L + 2 * W = 80) : A = 600 :=
by
  sorry

end area_of_field_l699_699194


namespace unique_y_values_count_l699_699239

theorem unique_y_values_count :
  {y : ℤ | ∃ x : ℤ, x ≥ 0 ∧ (x - y)^2 + x^2 = 25}.to_finset.card = 5 := by
  sorry

end unique_y_values_count_l699_699239


namespace largest_non_shaded_region_l699_699605

/-- Definitions of the figures' geometric properties and areas -/
def FigureX_sq_side := 3
def FigureX_circle_radius := 1.5
def FigureX_sq_area := FigureX_sq_side ^ 2
def FigureX_circle_area := Real.pi * (FigureX_circle_radius ^ 2)
def FigureX_non_shaded_area := FigureX_circle_area

def FigureY_sq_side := 3
def FigureY_circle_radius := 1
def FigureY_sq_area := FigureY_sq_side ^ 2
def FigureY_circle_area := Real.pi * (FigureY_circle_radius ^ 2)
def FigureY_non_shaded_area := FigureY_circle_area

def FigureZ_sq_side := 4
def FigureZ_circle_radius := FigureZ_sq_side / 2
def FigureZ_sq_area := FigureZ_sq_side ^ 2
def FigureZ_circle_area := Real.pi * (FigureZ_circle_radius ^ 2)
def FigureZ_non_shaded_area := FigureZ_sq_area

theorem largest_non_shaded_region :
  FigureZ_non_shaded_area > FigureX_non_shaded_area ∧ FigureZ_non_shaded_area > FigureY_non_shaded_area :=
by
  sorry

end largest_non_shaded_region_l699_699605


namespace average_weight_of_a_and_b_is_40_l699_699453

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B + C) / 3 = 42
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 40

-- Theorem statement
theorem average_weight_of_a_and_b_is_40 (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : 
    (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_is_40_l699_699453


namespace missy_yells_total_l699_699840

variable {O S M : ℕ}
variable (yells_at_obedient : ℕ)

-- Conditions:
def yells_stubborn (yells_at_obedient : ℕ) : ℕ := 4 * yells_at_obedient
def yells_mischievous (yells_at_obedient : ℕ) : ℕ := 2 * yells_at_obedient

-- Prove the total yells equal to 84 when yells_at_obedient = 12
theorem missy_yells_total (h : yells_at_obedient = 12) :
  yells_at_obedient + yells_stubborn yells_at_obedient + yells_mischievous yells_at_obedient = 84 :=
by
  sorry

end missy_yells_total_l699_699840


namespace bananas_per_chimp_per_day_l699_699450

theorem bananas_per_chimp_per_day (total_chimps total_bananas : ℝ) (h_chimps : total_chimps = 45) (h_bananas : total_bananas = 72) :
  total_bananas / total_chimps = 1.6 :=
by
  rw [h_chimps, h_bananas]
  norm_num

end bananas_per_chimp_per_day_l699_699450


namespace fourth_term_binomial_expansion_l699_699261

theorem fourth_term_binomial_expansion :
  (∑ (k : ℕ) in range 9, (binom 8 k) * ((2 * a / sqrt x)^(8 - k) * ((2 * sqrt x) / a^3)^k)).nth 3
  = 7168 * a^(-4) * x^(-3 / 2) := by
sorry

end fourth_term_binomial_expansion_l699_699261


namespace sarah_return_speed_l699_699273

-- Definitions for conditions:
def north_speed : ℝ := 3 -- Sarah's speed walking north in mph
def total_distance : ℝ := 6 -- Total distance walked in miles 
def round_trip_time : ℝ := 3.5 -- Total round trip time in hours

-- The proof problem:
theorem sarah_return_speed : 
  ∀ (return_speed : ℝ), 
    north_speed * 2 + return_speed * (round_trip_time - total_distance / north_speed) = total_distance 
    → return_speed = 4 :=
by 
  intro return_speed
  intro h
  -- Auxiliary definitions
  let north_time := total_distance / north_speed
  have south_time := round_trip_time - north_time
  calc 
  return_speed = total_distance / south_time : by sorry -- Here we replace the 'sorry' with the actual proof steps
end

end sarah_return_speed_l699_699273


namespace total_pictures_l699_699432

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end total_pictures_l699_699432


namespace angle_between_a_and_b_is_pi_over_3_l699_699056

noncomputable def vector_space :=
  sorry

-- Definitions based on the problem's condition
variables (a b : vector_space) (x y : ℕ → ℕ → vector_space) (α : ℝ) 
variable [inner_product_space ℝ vector_space]

-- Let $\overrightarrow{a}$ and $\overrightarrow{b}$ be non-zero vectors,
-- where $\| \overrightarrow{b} \| = 2 \| \overrightarrow{a} \| $.
-- Four pairs of vectors $\overrightarrow{x_{1}}$, $\overrightarrow{x_{2}}$, $\overrightarrow{x_{3}}$, $\overrightarrow{x_{4}}$ and
-- $\overrightarrow{y_{1}}$, $\overrightarrow{y_{2}}$, $\overrightarrow{y_{3}}$, $\overrightarrow{y_{4}}$ are composed of 2 $\overrightarrow{a}$'s and 2 $\overrightarrow{b}$'s each.
-- If the minimum possible value of $\overrightarrow{x_{1}} \cdot \overrightarrow{y_{1}} + \overrightarrow{x_{2}} \cdot \overrightarrow{y_{2}} + \overrightarrow{x_{3}} \cdot \overrightarrow{y_{3}} + \overrightarrow{x_{4}} \cdot \overrightarrow{y_{4}}$ 
-- is $4 \| \overrightarrow{a} \| ^{2}$, then the angle between $\overrightarrow{a}$ and $\overrightarrow{b}$ is $\frac{\pi}{3}$.

theorem angle_between_a_and_b_is_pi_over_3 
  (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) 
  (h₃ : ∥b∥ = 2 * ∥a∥)
  (h₄ : (∀ x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄, 
    (∃ (k1 k2 k3 k4 : ℕ), 
      x₁ = vector_space k1 k2 ∧ 
      x₂ = vector_space k1 k3 ∧ 
      x₃ = vector_space k4 k1 ∧ 
      x₄ = vector_space k4 k2 ∧ 
      y₁ = vector_space k2 k3 ∧ 
      y₂ = vector_space k3 k1 ∧ 
      y₃ = vector_space k3 k4 ∧ 
      y₄ = vector_space k4 k3) →  
      (inner a a + inner a a + inner b b + inner b b = 4 * ∥a∥ ^ 2))
  : α = π / 3 :=
sorry

end angle_between_a_and_b_is_pi_over_3_l699_699056


namespace worker_saves_one_third_l699_699516

variable {P : ℝ} 
variable {f : ℝ}

theorem worker_saves_one_third (h : P ≠ 0) (h_eq : 12 * f * P = 6 * (1 - f) * P) : 
  f = 1 / 3 :=
sorry

end worker_saves_one_third_l699_699516


namespace radius_of_inscribed_sphere_in_pyramid_l699_699138

namespace RegularTriangularPyramid

variable (a : ℝ)

def inscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 13 - 1) / 12

theorem radius_of_inscribed_sphere_in_pyramid
  (h : ∀ (lateral_edge_angle_with_base : ℝ), lateral_edge_angle_with_base = Real.pi / 3) :
  inscribed_sphere_radius a = a * (Real.sqrt 13 - 1) / 12 :=
by
  intros
  sorry

end RegularTriangularPyramid

end radius_of_inscribed_sphere_in_pyramid_l699_699138


namespace area_triangle_eq_area_quadrilateral_l699_699127

variables {A B C D E P Q : Point}
variables [acute_triangle ABC]
variables [is_bisector AD]
variables [circumscribed_circle_intersect AD extended at E]
variables [perpendicular_from_point DP AB]
variables [perpendicular_from_point DQ AC]

theorem area_triangle_eq_area_quadrilateral :
  area (triangle A B C) = area (quadrilateral A P Q E) :=
sorry

end area_triangle_eq_area_quadrilateral_l699_699127


namespace smallest_valid_number_correct_l699_699629

/-
  Problem Statement:
  Find the smallest integer starting with the digit 7 that becomes three times smaller when this digit is moved to the end.
  Given conditions:
  - The number starts with digit '7'.
  - The number becomes three times smaller when the first digit '7' is moved to the end.
-/

/-- 
   Use the notation n to represent the number. The number should start with 7 
   and moving 7 to the end should give a number that is three times the original number.  
-/
def is_valid_number (n : ℕ) :=
  let first_digit := Nat.digits 10 n |> List.head in
  let moved_number := Nat.digits 10 n |> (List.tail ·) |> List.reverse |> List.cons 7 |> List.reverse |> List.foldl (λ sum d => sum * 10 + d) 0 in
  first_digit = some 7 ∧ moved_number = 3 * n

/-- 
  Find the smallest valid number according to the problem statement.
-/
def smallest_valid_number : ℕ := 7241379310344827586206896551

theorem smallest_valid_number_correct : is_valid_number smallest_valid_number :=
  by
  -- By computation and checking against the conditions
  -- We assert this number is true according to steps computed above:
  sorry

end smallest_valid_number_correct_l699_699629


namespace population_at_seven_years_l699_699136

theorem population_at_seven_years (a x : ℕ) (y: ℝ) (h₀: a = 100) (h₁: x = 7) (h₂: y = a * Real.logb 2 (x + 1)):
  y = 300 :=
by
  -- We include the conditions in the theorem statement
  sorry

end population_at_seven_years_l699_699136


namespace expression_simplification_l699_699633
noncomputable def expression (x y z : ℝ) : ℝ :=
  (2 * x + 2 * y + 2 * z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (xy + yz + xz)⁻¹ * (2 * (xy)⁻¹ + 2 * (yz)⁻¹ + 2 * (xz)⁻¹)

theorem expression_simplification (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  expression x y z = (x * y * z)⁻² := 
by 
  sorry

end expression_simplification_l699_699633


namespace range_of_f_inequality_l699_699674

def f (x : ℝ) : ℝ := Real.exp (|x|) - 1 / (x^2 + 2)

theorem range_of_f_inequality (x : ℝ) : f x > f (2 * x - 1) ↔ (1 / 3) < x ∧ x < 1 := 
by
  sorry

end range_of_f_inequality_l699_699674


namespace range_of_a_l699_699812

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699812


namespace triangle_proof_l699_699370

variables (a b c p : ℝ) (h1 : a > b > c) (h2 : a = 2 * (b - c)) (h3 : p * a = c * b) -- Assuming p is given as a projection constant related to sides a and c

theorem triangle_proof (a b c p : ℝ) (h1 : a > b > c) (h2 : a = 2 * (b - c)) (h3 : p = c * (a / b)) : 
  4 * c + 8 * p = 3 * a :=
sorry

end triangle_proof_l699_699370


namespace little_john_remaining_money_l699_699417

noncomputable def initial_amount: ℝ := 8.50
noncomputable def spent_on_sweets: ℝ := 1.25
noncomputable def given_to_each_friend: ℝ := 1.20
noncomputable def number_of_friends: ℝ := 2

theorem little_john_remaining_money : 
  initial_amount - (spent_on_sweets + given_to_each_friend * number_of_friends) = 4.85 :=
by
  sorry

end little_john_remaining_money_l699_699417


namespace complex_real_number_iff_imaginary_part_zero_l699_699709

theorem complex_real_number_iff_imaginary_part_zero (m : ℝ) : (1 + Complex.i) * (1 + m * Complex.i) ∈ ℝ → m = -1 :=
by
  -- Given that the imaginary part of the complex number must be zero
  set z := (1 + Complex.i) * (1 + m * Complex.i)
  rw Complex.mul_def at z
  rw Complex.add_def at z
  rw Complex.mul_im at z
  rw Complex.mul_real at z
  sorry

end complex_real_number_iff_imaginary_part_zero_l699_699709


namespace exponential_model_better_fit_l699_699992
open Real

noncomputable def correlation_coefficient (sum_xy : ℝ) (sum_x2 : ℝ) (sum_y2 : ℝ) : ℝ :=
  sum_xy / (sqrt sum_x2 * sqrt sum_y2)

theorem exponential_model_better_fit 
  (x_bar : ℝ) (y_bar : ℝ) (u_bar : ℝ) (v_bar : ℝ) 
  (sum_x_i_sub_xbar_sq : ℝ) (sum_u_i_sub_ubar_sq : ℝ) (sum_u_i_sub_ubar_yi_sub_ybar : ℝ) 
  (sum_y_i_sub_ybar_sq : ℝ) (sum_v_i_sub_vbar_sq : ℝ) (sum_x_i_sub_xbar_v_i_sub_vbar : ℝ)
  (ln2 : ℝ) (ln5 : ℝ) : 
  (r1 : ℝ := correlation_coefficient sum_u_i_sub_ubar_yi_sub_ybar sum_u_i_sub_ubar_sq sum_y_i_sub_ybar_sq)
  (r2 : ℝ := correlation_coefficient sum_x_i_sub_xbar_v_i_sub_vbar sum_x_i_sub_xbar_sq sum_v_i_sub_vbar_sq) :
  r2 > r1 → 
  (λ : ℝ := sum_x_i_sub_xbar_v_i_sub_vbar / sum_x_i_sub_xbar_sq)
  (t : ℝ := v_bar - (λ * x_bar)) :
  ∃ x : ℝ, (ln 800 = λ * x + t) ∧ (x = (ln 800 - t) / λ) :=
sorry

end exponential_model_better_fit_l699_699992


namespace cos_2theta_plus_sin_theta_cos_theta_l699_699715

theorem cos_2theta_plus_sin_theta_cos_theta (θ : ℝ) (h : ∀ x, f(2*θ - x) = f(x)) :
  (cos (2 * θ) + sin θ * cos θ) = -1 := sorry

def f (x : ℝ) : ℝ := sin x + 2 * cos x

end cos_2theta_plus_sin_theta_cos_theta_l699_699715


namespace total_pieces_is_252_l699_699855

-- Define the number of packages and pieces per package
def packages_gum : Nat := 28
def packages_candy : Nat := 14
def pieces_per_package : Nat := 6

-- Define the total pieces Robin has
def total_pieces := (packages_gum * pieces_per_package) + (packages_candy * pieces_per_package)

-- Prove that the total pieces is 252
theorem total_pieces_is_252 : total_pieces = 252 := by
  have pieces_gum := packages_gum * pieces_per_package
  have pieces_candy := packages_candy * pieces_per_package
  have total := pieces_gum + pieces_candy
  show total = 252
  sorry

end total_pieces_is_252_l699_699855


namespace monotonic_increasing_range_l699_699778

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699778


namespace f_g_2_eq_14_l699_699409

def f (x : ℝ) : ℝ := 3 * real.sqrt(x) + 15 / real.sqrt(x)
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

theorem f_g_2_eq_14 : f (g 2) = 14 :=
by
  sorry

end f_g_2_eq_14_l699_699409


namespace polygon_sides_l699_699554

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_l699_699554


namespace total_bottles_in_box_l699_699964

def dozens (n : ℕ) := 12 * n

def water_bottles : ℕ := dozens 2

def apple_bottles : ℕ := water_bottles + 6

def total_bottles : ℕ := water_bottles + apple_bottles

theorem total_bottles_in_box : total_bottles = 54 := 
by
  sorry

end total_bottles_in_box_l699_699964


namespace find_AX_circle_diameter_l699_699425

theorem find_AX_circle_diameter (
  A B C D X : Point,
  h1 : diameter 1,
  h2 : X ∈ diameter AD,
  h3 : dist B X = dist C X,
  h4 : 3 * ∠ A B C = ∠ B X C = 72
) :
  AX = cos(24) * csc(24) * cos(18) :=
by
  sorry

end find_AX_circle_diameter_l699_699425


namespace appropriate_sampling_method_l699_699539

theorem appropriate_sampling_method (total_staff teachers admin_staff logistics_personnel sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : teachers = 120)
  (h3 : admin_staff = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20) :
  (sample_method : String) -> sample_method = "Stratified sampling" :=
sorry

end appropriate_sampling_method_l699_699539


namespace monotonically_increasing_range_l699_699768

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699768


namespace full_day_students_count_l699_699999

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l699_699999


namespace overtime_hours_l699_699988

theorem overtime_hours (x y : ℕ) 
  (h1 : 60 * x + 90 * y = 3240) 
  (h2 : x + y = 50) : 
  y = 8 :=
by
  sorry

end overtime_hours_l699_699988


namespace find_code_l699_699993

theorem find_code (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 11 * (A + B + C) = 242) :
  A = 5 ∧ B = 8 ∧ C = 9 ∨ A = 5 ∧ B = 9 ∧ C = 8 :=
by
  sorry

end find_code_l699_699993


namespace range_of_a_l699_699763

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699763


namespace set_M_properties_l699_699957

def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem set_M_properties :
  M = { x | 0 < x ∧ x < 2 } ∧
  (∀ a, a ∈ M → 
    ((0 < a ∧ a < 1) → (a^2 - a + 1 < 1 / a)) ∧
    (a = 1 → (a^2 - a + 1 = 1 / a)) ∧
    ((1 < a ∧ a < 2) → (a^2 - a + 1 > 1 / a))) := 
by
  sorry

end set_M_properties_l699_699957


namespace classify_curve_l699_699609

-- Define the curve equation
def curve_equation (m : ℝ) : Prop := 
  ∃ (x y : ℝ), ((m - 3) * x^2 + (5 - m) * y^2 = 1)

-- Define the conditions for types of curves
def is_circle (m : ℝ) : Prop := 
  m = 4 ∧ (curve_equation m)

def is_ellipse (m : ℝ) : Prop := 
  (3 < m ∧ m < 5 ∧ m ≠ 4) ∧ (curve_equation m)

def is_hyperbola (m : ℝ) : Prop := 
  ((m > 5 ∨ m < 3) ∧ (curve_equation m))

-- Main theorem stating the type of curve
theorem classify_curve (m : ℝ) : 
  (is_circle m) ∨ (is_ellipse m) ∨ (is_hyperbola m) :=
sorry

end classify_curve_l699_699609


namespace baker_flour_l699_699960

/-- 
A baker uses 6 2/3 cups of flour for 5/3 recipes of rolls. She will use 9 3/4 cups of flour for m/n recipes of rolls, 
where m and n are relatively prime positive integers. We need to find m + n.
-/

theorem baker_flour (frac1 frac2 : ℚ)
    (h_frac1 : frac1 = 6 + 2/3)
    (h_frac2 : frac2 = 9 + 3/4)
    (h_recipes : ∀ (m n : ℕ), coprime m n → frac2 / frac1 = ((m : ℚ) / n)) :
  ∃ (m n : ℕ), coprime m n ∧ m + n = 55 := sorry

end baker_flour_l699_699960


namespace find_starting_number_l699_699361

open Nat

theorem find_starting_number (m n : ℕ) (h₁ : n = 140) (h₂ : ∃ k : ℕ, k = 44 ∧ ∀ x, m ≤ x ∧ x ≤ n → even x → ¬ (x % 3 = 0 ↔ m + 2 * (x / 2) =. x)) : 
  m = 10 :=
by
   sorry

end find_starting_number_l699_699361


namespace not_consecutive_prime_powers_l699_699556

theorem not_consecutive_prime_powers (N : ℕ) : 
  ∃ M : ℕ, ∀ i : ℕ, i < N → ¬ ∃ (p : ℕ) (k : ℕ), prime p ∧ k ≥ 1 ∧ M + i = p^k := 
sorry

end not_consecutive_prime_powers_l699_699556


namespace correct_distribution_l699_699030

noncomputable def a (n : ℕ) : ℤ := 4 * n - 16

axiom a2_eq_neg8 : a 2 = -8
axiom a3_eq_neg4 : a 3 = -4

def first_10_terms := list.range' 1 10

def positive_terms := list.filter (λ n, a n > 0) first_10_terms

def X (s : finset ℕ) : ℕ := (s ∩ (finset.of_list positive_terms)).card

theorem correct_distribution : 
  ∃ (X_dist : hypergeom 10 6 3), 
    (∀ s : finset ℕ, s.card = 3 → X s = X_dist.sample s) ∧
    ( ∑ k in finset.range 4, k * hypergeom.probability_mass_function X_dist k) = 9 / 5 :=
sorry

end correct_distribution_l699_699030


namespace percentage_increase_is_correct_l699_699971

-- Define the wholesale price and the price paid by the customer
def wholesale_price : ℝ := 4.0
def price_paid_by_customer : ℝ := 4.75

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define retail price before discount
def retail_price : ℝ := price_paid_by_customer / (1 - discount_rate)

-- Define the percentage increase from wholesale price to retail price
def percentage_increase (wholesale_price retail_price : ℝ) : ℝ :=
  ((retail_price - wholesale_price) / wholesale_price) * 100

-- State the theorem to prove
theorem percentage_increase_is_correct :
  percentage_increase wholesale_price retail_price = 25 := by
  sorry

end percentage_increase_is_correct_l699_699971


namespace find_X_l699_699562

noncomputable def X (Y Z : ℕ) := 8

variables (X Y Z W V U T S R Q : ℕ)

-- conditions
axiom H1 : X > Y > Z
axiom H2 : W > V > U
axiom H3 : T > S > R > Q
axiom H4 : W % 2 = 0 ∧ U % 2 = 0  -- W and U are even
axiom H5 : V % 2 = 1  -- V is odd
axiom H6 : T % 2 = 0  -- T is even
axiom H7 : (S % 2 = 1) ∧ (R % 2 = 1) ∧ (Q % 2 = 1) ∧ (S = R + 2) ∧ (R = Q + 2)
axiom H8 : X + Y + Z = 13

theorem find_X : X = 8 := by
  sorry

end find_X_l699_699562


namespace bone_pile_count_l699_699977

theorem bone_pile_count :
  ∃ (n : ℕ), let dog1 := 3,
                dog2 := dog1 - 1,
                dog3 := 2 * dog2,
                dog4 := 1,
                dog5 := 2 * dog4 in
             n = dog1 + dog2 + dog3 + dog4 + dog5 ∧ n = 12 :=
by
  use 12
  let dog1 := 3
  let dog2 := dog1 - 1
  let dog3 := 2 * dog2
  let dog4 := 1
  let dog5 := 2 * dog4
  have h : 12 = dog1 + dog2 + dog3 + dog4 + dog5 := sorry
  exact ⟨h, rfl⟩

end bone_pile_count_l699_699977


namespace coin_flip_prob_difference_l699_699508

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem coin_flip_prob_difference :
  let p1 := binomial_prob 3 2 (1/2)
  let p2 := binomial_prob 3 3 (1/2)
  abs (p1 - p2) = 1 / 4 :=
by
  sorry

end coin_flip_prob_difference_l699_699508


namespace range_of_a_l699_699712

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) → a ∈ Set.Ioo (-∞) 1 ∪ Set.Ioo 3 ∞ := by
  sorry

end range_of_a_l699_699712


namespace coordinates_of_N_l699_699310

-- Define the problem parameters
def M : ℝ × ℝ := (1, -2)
def MN_length : ℝ := 3
def MN_parallel_x_axis : Prop := true  -- This is inherent in the nature of the problem

-- Define the set of possible coordinates for point N
def possible_N_coordinates : set (ℝ × ℝ) :=
  {(-2, -2), (4, -2)}

-- The theorem to be proved
theorem coordinates_of_N :
  ∃ N : ℝ × ℝ, (N ∈ possible_N_coordinates) ∧ (N.1 = M.1 - MN_length ∨ N.1 = M.1 + MN_length ∧ N.2 = M.2) :=
sorry

end coordinates_of_N_l699_699310


namespace right_triangle_area_proof_l699_699925

noncomputable def right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1 / 2) * a * b

theorem right_triangle_area_proof (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 40) (hc : c = 41) :
right_triangle_area a b c h = 180 :=
by
  rw [ha, hc]
  have hb := sqrt (c^2 - a^2)
  sorry

end right_triangle_area_proof_l699_699925


namespace quadrilaterals_similarity_l699_699291

theorem quadrilaterals_similarity (ABCD A1 B1 C1 D1 A2 B2 C2 D2 : Type) [quadrilateral ABCD] 
[circumcenter_of_triangles A1 B1 C1 D1 ABCD] [circumcenter_of_triangles A2 B2 C2 D2 (quadrilateral A1 B1 C1 D1)] 
(h1 : convex_quadrilateral ABCD)
(h2 : circumcenter B1 ABCD = circumcenter B1 ABCD_triangle) 
(h3 : circumcenter C1 BCD = circumcenter C1 BCD_triangle)
(h4 : circumcenter D1 CDA = circumcenter D1 CDA_triangle)
(h5 : circumcenter A1 DAB = circumcenter A1 DAB_triangle)
(h6 : circumcenter A2 A1B1C1D1 = circumcenter A2 A1B1C1D1_triangle)
(h7 : circumcenter B2 A1B1C1D1 = circumcenter B2 A1B1C1D1_triangle)
(h8 : circumcenter C2 A1B1C1D1 = circumcenter C2 A1B1C1D1_triangle)
(h9 : circumcenter D2 A1B1C1D1 = circumcenter D2 A1B1C1D1_triangle) :
similarity_coefficient quadrilaterals_\(ABCD A2 B2 C2 D2\) = 
( ∣ ( (cot A + cot C) (cot B + cot D) ) / 4 ∣ ) :=
sorry

end quadrilaterals_similarity_l699_699291


namespace detectives_sons_ages_l699_699151

theorem detectives_sons_ages (x y : ℕ) (h1 : x < 5) (h2 : y < 5) (h3 : x * y = 4) (h4 : (∃ x₁ y₁ : ℕ, (x₁ * y₁ = 4 ∧ x₁ < 5 ∧ y₁ < 5) ∧ x₁ ≠ x ∨ y₁ ≠ y)) :
  (x = 1 ∨ x = 4) ∧ (y = 1 ∨ y = 4) :=
by
  sorry

end detectives_sons_ages_l699_699151


namespace find_angle_BDC_l699_699343

-- Define the angles and their measures as per the conditions
def angle_A : Real := 45
def angle_E : Real := 55
def angle_C : Real := 25

-- Define the final proof statement in Lean
theorem find_angle_BDC
    (angleA: Real := angle_A)
    (angleE: Real := angle_E)
    (angleC: Real := angle_C)
    (B_intersects_AC_DE: ∀ (A B C D E: Point), 
        B ∈ (AC ∩ DE) -- Assuming AC and DE are lines
    ):
  ∠BDC = 155 :=
by
  sorry

end find_angle_BDC_l699_699343


namespace total_length_of_sticks_l699_699935

-- Definitions based on conditions
def num_sticks := 30
def length_per_stick := 25
def overlap := 6
def effective_length_per_stick := length_per_stick - overlap

-- Theorem statement
theorem total_length_of_sticks : num_sticks * effective_length_per_stick - effective_length_per_stick + length_per_stick = 576 := sorry

end total_length_of_sticks_l699_699935


namespace range_of_a_l699_699807

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699807


namespace range_of_a_exists_x_l699_699328

noncomputable def f (x a : ℝ) : ℝ := |x - a| + a
noncomputable def g (x : ℝ) : ℝ := 4 - x^2

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x : ℝ, g x ≥ f x a) ↔ a ∈ Iic (17/8) := 
sorry

end range_of_a_exists_x_l699_699328


namespace distance_between_parallel_lines_l699_699904

theorem distance_between_parallel_lines 
  (r : ℝ) 
  (d : ℝ)
  (h1 : ∀ P Q R S : ℝ, P = 40 ∧ Q = 40 ∧ R = 36 → (4 * r^2 - 400 = 10 * d^2))
  (h2 : ∀ U V W X : ℝ, U = 40 ∧ V = 40 ∧ W = 36 → (9 * r^2 - 324 = 729 * d^2))
  : d = real.sqrt (76 / 719) :=
  sorry  -- Proof is omitted

end distance_between_parallel_lines_l699_699904


namespace baldness_frequency_l699_699874

def gene_frequency (p : ℝ) (q : ℝ) (AA_male : ℝ) (Aa_male : ℝ) (AA_female : ℝ) : Prop :=
  p = 0.3 ∧ q = 0.7 ∧ AA_male = p^2 ∧ Aa_male = 2 * p * q ∧ AA_female = p^2

theorem baldness_frequency (p q : ℝ) (AA_male Aa_male AA_female : ℝ) (h : gene_frequency p q AA_male Aa_male AA_female) :
  (AA_male + Aa_male = 0.51) ∧ (AA_female = 0.09) :=
by
  cases h with h1 h2
  rw [h1.left, h2.right.left, h2.right.right.left, h2.right.right.right]
  split
  { rw [Real.pow_two], linarith }
  { rw [Real.pow_two] }
  sorry

end baldness_frequency_l699_699874


namespace monotonically_increasing_range_l699_699767

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699767


namespace part1_part2_l699_699321

-- Define the sequence {a_n} according to the given recurrence relation
def a_seq (a : ℕ → ℝ) := ∀ n, 2 * a (n + 1) - 2 * a n + a (n + 1) * a n = 0

-- The sequence {b_n} defined using function f and a_seq
def f (x : ℝ) := (7 * x + 5) / (x + 1)

def b_seq (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  if n = 0 then f 0 
  else f (a n - 1)

-- Prove that {1 / a_n} is an arithmetic sequence
theorem part1 {a : ℕ → ℝ} (h : a_seq a) (h₀ : ∀ n, a n ≠ 0) : 
  ∃ d, ∀ n, (1 / a (n + 1)) - (1 / a n) = d := 
  sorry

-- Prove the sum of { |b_n| } as described in the solution
theorem part2 {a : ℕ → ℝ} (h : a_seq a) (h₀ : ∀ n, a n ≠ 0) :
  ∀ n, let b := λ n, |if n = 0 then f 0 else f (a n - 1)|
        in (∑ i in finset.range n, b i) = 
           if n ≤ 6 then n * (11 - n) / 2 
           else (n ^ 2 - 11 * n + 60) / 2 := 
  sorry

end part1_part2_l699_699321


namespace area_ratio_S_T_l699_699757

-- Define the set T as the set of ordered triples (x, y, z) of nonnegative real numbers that lie in the plane x + y + z = 1
def T : Set (ℝ × ℝ × ℝ) := {p | ∃ x y z, p = (x, y, z) ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

-- Define the supports condition
def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

-- Define the set S as those triples in T that support (1/3, 1/4, 1/5)
def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/3, 1/4, 1/5)}

-- The statement to prove that the area of S divided by the area of T is 347 / 500
theorem area_ratio_S_T : 
  (area S) / (area T) = 347 / 500 := 
sorry

end area_ratio_S_T_l699_699757


namespace garden_length_l699_699842

open Nat

def perimeter : ℕ → ℕ → ℕ := λ l w => 2 * (l + w)

theorem garden_length (width : ℕ) (perimeter_val : ℕ) (length : ℕ) 
  (h1 : width = 15) 
  (h2 : perimeter_val = 80) 
  (h3 : perimeter length width = perimeter_val) :
  length = 25 := by
  sorry

end garden_length_l699_699842


namespace equivalent_discount_l699_699972

theorem equivalent_discount (original_price : ℝ) (d1 d2 single_discount : ℝ) :
  original_price = 50 →
  d1 = 0.15 →
  d2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - d1) * (1 - d2) = original_price * (1 - single_discount) :=
by
  intros
  sorry

end equivalent_discount_l699_699972


namespace James_future_age_when_Thomas_reaches_James_current_age_l699_699483

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l699_699483


namespace proof_solution_l699_699529

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  0 < x ∧ 7.61 * log x / log 2 + 2 * log x / log 4 = x ^ (log 16 / log 3 / log x / log 9)

theorem proof_solution : proof_problem (16 / 3) :=
by
  sorry

end proof_solution_l699_699529


namespace time_calculation_proof_l699_699903

noncomputable def time_first_machine := 6
noncomputable def time_second_machine := 8
noncomputable def time_third_machine := 12

theorem time_calculation_proof :
  let x := time_first_machine,
      y := time_second_machine,
      z := time_third_machine in
  y = x + 2 ∧ 
  z = 2 * x ∧ 
  (1 / x + 1 / y + 1 / z = 3 / 8) :=
by
  let x := time_first_machine
  let y := time_second_machine
  let z := time_third_machine
  have h1 : y = x + 2 := by rfl
  have h2 : z = 2 * x := by rfl
  have h3 : 1 / x + 1 / y + 1 / z = 3 / 8 := by
    calc
      1 / x + 1 / y + 1 / z
          = 1 / 6 + 1 / 8 + 1 / 12  : by rfl
      ... = 4 / 24 + 3 / 24 + 2 / 24 : by norm_num
      ... = 9 / 24                  : by calc
                                   ... = 3 / 8  : by norm_num
  exact ⟨h1, h2, h3⟩

end time_calculation_proof_l699_699903


namespace sum_of_sixth_powers_l699_699055

theorem sum_of_sixth_powers (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 0) 
  (h2 : α₁^2 + α₂^2 + α₃^2 = 2) 
  (h3 : α₁^3 + α₂^3 + α₃^3 = 4) : 
  α₁^6 + α₂^6 + α₃^6 = 7 :=
sorry

end sum_of_sixth_powers_l699_699055


namespace perpendicular_lines_l699_699353

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l699_699353


namespace simplify_expression_l699_699595

noncomputable def sqrt2_minus_sqrt3 : ℝ :=
  real.sqrt 2 - real.sqrt 3

noncomputable def calc_root_expression : ℝ :=
  abs sqrt2_minus_sqrt3 + real.cbrt 8 - real.sqrt 2 * (real.sqrt 2 - 1)

theorem simplify_expression :
  calc_root_expression = real.sqrt 3 := by
  sorry

end simplify_expression_l699_699595


namespace crackers_per_person_l699_699607

variable (darrenA : Nat)
variable (darrenB : Nat)
variable (aCrackersPerBox : Nat)
variable (bCrackersPerBox : Nat)
variable (calvinA : Nat)
variable (calvinB : Nat)
variable (totalPeople : Nat)

-- Definitions based on the conditions
def totalDarrenCrackers := darrenA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCalvinA := 2 * darrenA - 1
def totalCalvinCrackers := totalCalvinA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCrackers := totalDarrenCrackers + totalCalvinCrackers
def crackersPerPerson := totalCrackers / totalPeople

-- The theorem to prove the question equals the answer given the conditions
theorem crackers_per_person :
  darrenA = 4 →
  darrenB = 2 →
  aCrackersPerBox = 24 →
  bCrackersPerBox = 30 →
  calvinA = 7 →
  calvinB = darrenB →
  totalPeople = 5 →
  crackersPerPerson = 76 :=
by
  intros
  sorry

end crackers_per_person_l699_699607


namespace find_a_l699_699031

-- Definition of the points and conditions
def P : ℝ × ℝ := (-5, a)

def circle (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*y - 1 = 0

def tangent_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) + (x1 + x2 - 2) / (y1 + y2) = 0

noncomputable def a_value (a : ℝ) : Prop :=
  circle a (-5) a ∧ (∃ x1 y1 x2 y2, tangent_condition x1 y1 x2 y2) ∧ (a = 3 ∨ a = -2)

theorem find_a (a : ℝ) : a_value a := 
  sorry

end find_a_l699_699031


namespace merchant_marked_price_l699_699551

theorem merchant_marked_price (L : ℤ) (M : ℤ) :
  L = 100 ∧ 
  let purchase_price := L - (30 * L / 100) in
  purchase_price = 70 ∧ 
  let selling_price := (75 * M / 100) in
  selling_price - purchase_price = (30 * selling_price / 100) → 
  M = 13333 / 100 :=
by
  sorry

end merchant_marked_price_l699_699551


namespace line_plane_intersection_l699_699262

theorem line_plane_intersection :
  (∃ t : ℝ, (x, y, z) = (3 + t, 1 - t, -5) ∧ (3 + t) + 7 * (1 - t) + 3 * (-5) + 11 = 0) →
  (x, y, z) = (4, 0, -5) :=
sorry

end line_plane_intersection_l699_699262


namespace subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l699_699048

def A : Set ℝ := {x | x ^ 2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_if_a_neg_third (a : ℝ) (h : a = -1/3) : B a ⊆ A := by
  sorry

theorem set_of_real_numbers_for_A_union_B_eq_A : {a : ℝ | A ∪ B a = A} = {0, -1/3, -1/5} := by
  sorry

end subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l699_699048


namespace tom_climbing_time_in_hours_l699_699491

variable (t_Elizabeth t_Tom_minutes t_Tom_hours : ℕ)

-- Conditions:
def elizabeth_time : ℕ := 30
def tom_time_relation (t_Elizabeth : ℕ) : ℕ := 4 * t_Elizabeth
def tom_time_hours (t_Tom_minutes : ℕ) : ℕ := t_Tom_minutes / 60

-- Theorem statement:
theorem tom_climbing_time_in_hours :
  tom_time_hours (tom_time_relation elizabeth_time) = 2 :=
by 
  -- Reiterate the conditions with simplified relations
  have t_Elizabeth := elizabeth_time
  have t_Tom_minutes := tom_time_relation t_Elizabeth
  have t_Tom_hours := tom_time_hours t_Tom_minutes
  show t_Tom_hours = 2
  -- Placeholder for actual proof
  sorry

end tom_climbing_time_in_hours_l699_699491


namespace arithmetic_seq_sum_inequality_l699_699657

variable (a_n b_n : ℕ → ℝ)

-- Conditions
axiom cond_a (n : ℕ) : exp (a_n n) + a_n n = n
axiom cond_b (n : ℕ) : log (b_n n) + b_n n = n

-- Problem 1: Prove that a_n + b_n = n
theorem arithmetic_seq (n : ℕ) : a_n n + b_n n = n := sorry

-- Definition of c_n
noncomputable def c_n (n : ℕ) : ℝ := (a_n n + b_n n) * (exp (a_n n) + log (b_n n))

-- Problem 2: Prove the sum inequality
theorem sum_inequality (n : ℕ) : (∑ k in Finset.range n, 1 / c_n k.succ) < (5 / 3) := sorry

end arithmetic_seq_sum_inequality_l699_699657


namespace right_triangle_area_l699_699917

theorem right_triangle_area (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : a = 40) (hc : c = 41) : 
  1 / 2 * a * √(c^2 - a^2) = 180 :=
by
  sorry

end right_triangle_area_l699_699917


namespace knights_count_l699_699579

-- Define the possible types of people: Knight or Liar
inductive Tribe
  | knight
  | liar

open Tribe

-- Define each person (Anton, Borya, Vasya, Grisha)
variable (Anton Borya Vasya Grisha : Tribe)

-- Conditions
def condition1 : Prop := ∀ x, x = knight ∨ x = liar
def condition2 : Prop := (Anton = knight ∧ Grisha = liar) ∨ (Anton = liar ∧ Grisha = knight)
def condition3 : Prop := (Borya = knight ∧ Vasya = liar) ∨ (Borya = liar ∧ Vasya = knight)
def condition4 : Prop := (Grisha = knight → ¬∃ (a b : Tribe), a = knight ∧ b = knight)

-- Theorem: There is exactly one knight among Anton, Borya, Vasya, and Grisha
theorem knights_count : condition1 → condition2 → condition3 → condition4 → ∃! x, x = knight :=
by
  intros
  sorry

end knights_count_l699_699579


namespace num_primes_between_squares_3500_8000_l699_699699

def prime_in_square_interval (a b : ℕ) (p : ℕ) : Prop :=
  a < p * p ∧ p * p < b

def primes_in_square_interval (a b : ℕ) : ℕ :=
  (List.range' (Nat.ceil (Real.sqrt a)) (Nat.floor (Real.sqrt b) - Nat.ceil (Real.sqrt a) + 1)).filter Prime.prime.filter (λ p, prime_in_square_interval a b p)

theorem num_primes_between_squares_3500_8000 :
  primes_in_square_interval 3500 8000 = 7 :=
sorry

end num_primes_between_squares_3500_8000_l699_699699


namespace root_in_interval_l699_699095

open Real

noncomputable def f (x : ℝ) : ℝ := log10 x + x - 2

theorem root_in_interval (n : ℤ) (h_n : n = 1) : 
  ∃ x, n < x ∧ x < n + 1 ∧ f x = 0 :=
by
  sorry

end root_in_interval_l699_699095


namespace sufficient_but_not_necessary_condition_l699_699525

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x + m = 0) ↔ m < 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l699_699525


namespace probability_of_Y_l699_699149

theorem probability_of_Y (P_X P_both : ℝ) (h1 : P_X = 1/5) (h2 : P_both = 0.13333333333333333) : 
    (0.13333333333333333 / (1 / 5)) = 0.6666666666666667 :=
by sorry

end probability_of_Y_l699_699149


namespace problem_1_problem_2_l699_699086

theorem problem_1 :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
by
  sorry

theorem problem_2 :
  (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) = (3^32 - 1) / 2 :=
by
  sorry

end problem_1_problem_2_l699_699086


namespace monotonic_increasing_range_l699_699795

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699795


namespace integral_evaluation_l699_699251

theorem integral_evaluation :
  ∫ x in 0..1, sqrt (1 - x ^ 2) + ∫ x in 1..2, 1 / x = Real.pi / 4 + Real.log 2 :=
by
  sorry

end integral_evaluation_l699_699251


namespace geometric_series_solves_y_eqn_l699_699600

theorem geometric_series_solves_y_eqn:
  (∑ n:ℕ, (1/3 : ℝ)^n) * (∑ n:ℕ, (-1/3 : ℝ)^n) = 1 + ∑ n:ℕ, (1 / (9:ℝ))^n := by
sorry

end geometric_series_solves_y_eqn_l699_699600


namespace cannot_pass_through_all_faces_l699_699153

-- Definitions based on conditions
variable (polyhedron : Type) [Nonempty polyhedron]
variable (E : polyhedron → polyhedron → Prop) -- E for edge relationship
variable (T : polyhedron → Prop) -- T for triangular face
variable (P : polyhedron) -- An arbitrary point on an edge of the polyhedron

-- Conditions based on problem statement
axiom all_faces_triangular : ∀ f, T f
axiom P_not_midpoint_endpoint : ∀ x y, E x y → (P ≠ x) ∧ (P ≠ (0.5 : ℝ) • x + (0.5 : ℝ) • y)

-- Centroid function
noncomputable def centroid (f : polyhedron) [T f] : polyhedron := sorry

-- Next point based on process
noncomputable def next_point (p : polyhedron) (f : polyhedron) [E p f ∧ T f] : polyhedron := sorry

-- Theorem statement (translated proof problem)
theorem cannot_pass_through_all_faces :
  ¬ ∃ (f : polyhedron) (p : polyhedron) [T f], ∀ (p_i : polyhedron) [E p_i f],
    ∃ (p_{i+1} : polyhedron), next_point p_i f = p_{i+1} := sorry

end cannot_pass_through_all_faces_l699_699153


namespace range_of_a_l699_699758

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699758


namespace solve_diamond_l699_699339

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l699_699339


namespace ripe_unripe_difference_after_five_days_l699_699181

/--
A bowl of fruit holds 18 peaches. Four of the peaches are ripe initially,
two more peaches ripen every day, but on the third day, three ripe peaches are eaten.
After five days, there are 4 more ripe peaches than unripe peaches in the bowl.
-/
theorem ripe_unripe_difference_after_five_days :
  (let total_peaches := 18
       initial_ripe := 4
       ripen_every_day := 2
       eaten_on_third_day := 3
       days := 5 in
    let ripe_peaches := initial_ripe + ripen_every_day * days - eaten_on_third_day in
    let unripe_peaches := total_peaches - ripe_peaches in
    ripe_peaches - unripe_peaches = 4) :=
by 
  let total_peaches  := 18
  let initial_ripe := 4
  let ripen_every_day := 2
  let eaten_on_third_day := 3
  let days := 5
  let ripe_peaches := initial_ripe + ripen_every_day * days - eaten_on_third_day
  let unripe_peaches := total_peaches - ripe_peaches
  have h : ripe_peaches - unripe_peaches = 4 := sorry
  exact h

end ripe_unripe_difference_after_five_days_l699_699181


namespace problem_statement_l699_699063

noncomputable def f : Polynomial ℂ := Polynomial.C 1 + Polynomial.X + Polynomial.C (-2) * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3

noncomputable def g (x : ℂ) : ℂ :=
  let r := Complex (some polynomial root calculations here)
  let s := Complex (some polynomial root calculations here)
  let t := Complex (some polynomial root calculations here)
  (x - r^2)*(x - s^2)*(x - t^2)

theorem problem_statement : g 4 = 105 := by {
  -- Problem to be proved
  sorry
}

end problem_statement_l699_699063


namespace band_formation_l699_699532

theorem band_formation (r x m : ℕ) (h1 : r * x + 3 = m) (h2 : (r - 1) * (x + 2) = m) (h3 : m < 100) : m = 69 :=
by
  sorry

end band_formation_l699_699532


namespace sqrt_domain_l699_699490

theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (2 * x - 4)) ↔ x ≥ 2 := by
  sorry

end sqrt_domain_l699_699490


namespace tangent_line_at_half_max_min_values_l699_699676

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x - Real.log x

theorem tangent_line_at_half : 
  ∃ m b, (m = 2) ∧ (b = -2 + Real.log 2) ∧ (∀ x : ℝ, f' x = m * (x - (1/2)) + b) :=
  sorry

theorem max_min_values :
  f 1 = 0 ∧ min (f (1/4)) (f e) = f (1/4) ∧ f (1/4) = Real.log 4 - 3 :=
  sorry

end tangent_line_at_half_max_min_values_l699_699676


namespace man_saves_percentage_of_salary_l699_699188

variable (S : ℝ) (P : ℝ) (S_s : ℝ)

def problem_statement (S : ℝ) (S_s : ℝ) (P : ℝ) : Prop :=
  S_s = S - 1.2 * (S - (P / 100) * S)

theorem man_saves_percentage_of_salary
  (h1 : S = 6250)
  (h2 : S_s = 250) :
  problem_statement S S_s 20 :=
by
  sorry

end man_saves_percentage_of_salary_l699_699188


namespace num_uniforms_needed_l699_699666

noncomputable def mean : ℝ := 165
noncomputable def variance : ℝ := 25
noncomputable def std_dev : ℝ := 5
noncomputable def num_students : ℕ := 1000
noncomputable def percentage_within_2_sigma : ℝ := 0.954

theorem num_uniforms_needed : 
  let interval_start := mean - 2 * std_dev
  let interval_end := mean + 2 * std_dev in
  (interval_start = 155 ∧ interval_end = 175) → 
  num_students * percentage_within_2_sigma = 954 :=
by
  sorry

end num_uniforms_needed_l699_699666


namespace twelfth_term_geometric_sequence_l699_699936

theorem twelfth_term_geometric_sequence : 
  let a : ℝ := 27 
  let r : ℝ := 1 / 3 
  let n : ℕ := 12 
  let a_n : ℝ := a * r^(n - 1)
  in a_n = 1 / 6561 := 
by 
  sorry

end twelfth_term_geometric_sequence_l699_699936


namespace not_all_pieces_found_l699_699335

theorem not_all_pieces_found (k p v : ℕ) (h1 : p + v > 0) (h2 : k % 2 = 1) : k + 4 * p + 8 * v ≠ 1988 :=
by
  sorry

end not_all_pieces_found_l699_699335


namespace final_net_worth_l699_699841

noncomputable def initial_cash_A := (20000 : ℤ)
noncomputable def initial_cash_B := (22000 : ℤ)
noncomputable def house_value := (20000 : ℤ)
noncomputable def vehicle_value := (10000 : ℤ)

noncomputable def transaction_1_cash_A := initial_cash_A + 25000
noncomputable def transaction_1_cash_B := initial_cash_B - 25000

noncomputable def transaction_2_cash_A := transaction_1_cash_A - 12000
noncomputable def transaction_2_cash_B := transaction_1_cash_B + 12000

noncomputable def transaction_3_cash_A := transaction_2_cash_A + 18000
noncomputable def transaction_3_cash_B := transaction_2_cash_B - 18000

noncomputable def transaction_4_cash_A := transaction_3_cash_A + 9000
noncomputable def transaction_4_cash_B := transaction_3_cash_B + 9000

noncomputable def final_value_A := transaction_4_cash_A
noncomputable def final_value_B := transaction_4_cash_B + house_value + vehicle_value

theorem final_net_worth :
  final_value_A - initial_cash_A = 40000 ∧ final_value_B - initial_cash_B = 8000 :=
by
  sorry

end final_net_worth_l699_699841


namespace range_of_f_l699_699677

theorem range_of_f :
  (range (λ x : ℝ, 1/2 * (Real.sin x + Real.cos x - |Real.sin x - Real.cos x|)))
  = Set.Icc (-1 : ℝ) (Real.sqrt 2 / 2) := by
  sorry

end range_of_f_l699_699677


namespace triangle_type_l699_699660

-- Let's define what it means for a triangle to be acute, obtuse, and right in terms of angle
def is_acute_triangle (a b c : ℝ) : Prop := (a < 90) ∧ (b < 90) ∧ (c < 90)
def is_obtuse_triangle (a b c : ℝ) : Prop := (a > 90) ∨ (b > 90) ∨ (c > 90)
def is_right_triangle (a b c : ℝ) : Prop := (a = 90) ∨ (b = 90) ∨ (c = 90)

-- The problem statement
theorem triangle_type (A B C : ℝ) (h : A = 100) : is_obtuse_triangle A B C :=
by {
  -- Sorry is used to indicate a placeholder for the proof
  sorry
}

end triangle_type_l699_699660


namespace find_c_l699_699705

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end find_c_l699_699705


namespace find_f_three_l699_699099

noncomputable def f : ℝ → ℝ := sorry -- f(x) is a linear function

axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

axiom equation : ∀ x, f x = 3 * (f⁻¹ x) + 9

axiom f_zero : f 0 = 3

axiom f_inv_three : f⁻¹ 3 = 0

theorem find_f_three : f 3 = 6 * Real.sqrt 3 := 
by sorry

end find_f_three_l699_699099


namespace even_numbers_in_row_2008_of_pascals_triangle_l699_699514

theorem even_numbers_in_row_2008_of_pascals_triangle :
  (number_even_numbers_in_row 2008) = 1881 := by
  sorry

end even_numbers_in_row_2008_of_pascals_triangle_l699_699514


namespace relationship_between_a_b_c_l699_699279

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (Real.cos (34 * Real.pi / 180) - Real.sin (34 * Real.pi / 180))
noncomputable def b : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (128 * Real.pi / 180) + Real.cos (40 * Real.pi / 180) * Real.cos (38 * Real.pi / 180)
noncomputable def c : ℝ := (1 / 2) * (Real.cos (80 * Real.pi / 180) - 2 * (Real.cos (50 * Real.pi / 180))^2 + 1)

theorem relationship_between_a_b_c : b > a ∧ a > c :=
  sorry

end relationship_between_a_b_c_l699_699279


namespace shaded_area_6_5_l699_699581

/-- Given a rectangle of dimensions 8x5,
we need to show that the area of the shaded region is 6.5. --/
theorem shaded_area_6_5 :
  let area_rectangle := 8 * 5 in
  let area_small_rectangles := 2 * (4 * 2) in
  let area_triangles := 4 * (5 + 3) in
  area_rectangle - (area_small_rectangles + area_triangles) = 6.5 :=
by
  sorry

end shaded_area_6_5_l699_699581


namespace polygonal_chain_max_length_not_exceed_200_l699_699289

-- Define the size of the board
def board_size : ℕ := 15

-- Define the concept of a polygonal chain length on a symmetric board
def polygonal_chain_length (n : ℕ) : ℕ := sorry -- length function yet to be defined

-- Define the maximum length constant to be compared with
def max_length : ℕ := 200

-- Define the theorem statement including all conditions and constraints
theorem polygonal_chain_max_length_not_exceed_200 :
  ∃ (n : ℕ), n = board_size ∧ 
             (∀ (length : ℕ),
             length = polygonal_chain_length n →
             length ≤ max_length) :=
sorry

end polygonal_chain_max_length_not_exceed_200_l699_699289


namespace balls_ratio_l699_699435

theorem balls_ratio (initial_Robert_balls : ℕ) (initial_Tim_balls : ℕ) (final_Robert_balls : ℕ)
  (h_initial_Robert : initial_Robert_balls = 25)
  (h_initial_Tim : initial_Tim_balls = 40)
  (h_final_Robert : final_Robert_balls = 45) :
  let balls_given_by_Tim := final_Robert_balls - initial_Robert_balls in
  let ratio := (balls_given_by_Tim : ℚ) / initial_Tim_balls in
  ratio = 1 / 2 := 
by 
  sorry

end balls_ratio_l699_699435


namespace toothpicks_required_to_expand_4_to_7_steps_l699_699212

-- Define the sequence for the number of toothpicks needed per step
def sequence (n : ℕ) : ℕ :=
  match n with
  | 1 => 4
  | 2 => 10
  | 3 => 18
  | 4 => 30
  | _ => sequence (n - 1) + 12 + 2 * (n - 4)

theorem toothpicks_required_to_expand_4_to_7_steps :
  sequence 4 = 30 →
  sequence 7 - sequence 4 = 48 :=
by
  intros h
  refine Eq.trans _ (by norm_num)
  -- Please add proof steps when necessary.
  sorry

end toothpicks_required_to_expand_4_to_7_steps_l699_699212


namespace statement_C_statement_D_l699_699692

-- Define the vectors with the given m
def a (m : ℝ) : ℝ × ℝ := (m + 1, -1)
def b (m : ℝ) : ℝ × ℝ := (1 - m, 2)

-- Define the vector addition
def vector_add (u : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define the magnitude of a vector
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

-- Define the dot product of two vectors
def dot_product (u : ℝ × ℝ) (v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the projection of vector u onto vector v
def projection (u : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

-- The Lean proof statements for C and D
theorem statement_C (m : ℝ) : magnitude (vector_add (a m) (b m)) = Real.sqrt 5 :=
  sorry

theorem statement_D : projection (a 1) (b 1) = (0, -1) :=
  sorry

end statement_C_statement_D_l699_699692


namespace number_of_zero_terms_in_sequence_l699_699732

theorem number_of_zero_terms_in_sequence :
  ∃ (a : Fin 2015 → ℤ), (∀ (i : Fin 2015), a i = -1 ∨ a i = 0 ∨ a i = 1) ∧
  (∑ i, a i = 427) ∧
  (∑ i, (a i + 1) ^ 2 = 3869) →
  (Finset.card {i : Fin 2015 | a i = 0}.toFinset = 1015) :=
by
  sorry

end number_of_zero_terms_in_sequence_l699_699732


namespace proof_problem_l699_699527

-- Define the triangle and its properties
variables (a b c : ℝ) (A B C : ℝ)
variables (D : Point)
variables (AD : ℝ) (S : ℝ) (P : ℝ)

-- Hypotheses from part (a)
def Conditions : Prop :=
  ∃ (A B C : ℝ), A + B + C = π ∧ -- Sum of angles in a triangle
  a*sin(A - B) = (c - b)*sin(A) ∧
  AD = 3 ∧
  ∠ ADC = π/3 ∧
  S = (3*sqrt(3)) -- Area of triangle ABC

-- Questions from part (c)
def Question1 : Prop := A = π/3
def Question2 : Prop := P = 4 + 2*sqrt(13)

-- Proof problem
theorem proof_problem (A B C : ℝ) (a b c : ℝ) (A B C : ℝ) : Conditions a b c A B C AD S ∧ Question1 A ∧ Question2 P :=
by
  -- proof here
  sorry -- skipping the proof as per instructions

end proof_problem_l699_699527


namespace problem_convex_quadrilateral_l699_699025

theorem problem_convex_quadrilateral
  (A B C D E F : Type)
  [convex_quadrilateral A B C D]
  (angle_bac_eq_cad : ∠BAC = ∠CAD)
  (angle_abc_eq_acd : ∠ABC = ∠ACD)
  (E_condition : E = extension_point_ad_bc AD BC)
  (F_condition : F = extension_point_ab_dc AB DC) :
  ∃ AB_length DE_length BC_length CE_length,
    (AB_length * DE_length) / (BC_length * CE_length) = 1 :=
by
  sorry

end problem_convex_quadrilateral_l699_699025


namespace tangent_line_at_1_l699_699323

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_1 :
  let x := (1 : ℝ)
  let y := (f 1)
  ∃ m b : ℝ, (∀ x, y - m * (x - 1) + b = 0)
  ∧ (m = -2)
  ∧ (b = -1) :=
by
  sorry

end tangent_line_at_1_l699_699323


namespace triangle_pqr_conditions_l699_699911

noncomputable def area_of_triangle_PQR : ℕ := 30 * 30

theorem triangle_pqr_conditions (QR : ℕ) (k p : ℕ) (QR_eq : QR = 30) 
 (incircle_bisects_median_PM : ∀ PM : ℕ, PM = 2 * m → median_bisected_by_incircle) :
  k + p = 60 :=
by
  have area_eq : triangle_area = 30 * 30 → k = 30 ∧ p = 30,
  sorry

  exact k + p

end triangle_pqr_conditions_l699_699911


namespace find_natural_integer_l699_699257

theorem find_natural_integer (n : ℕ) :
  ((n^3 + 39 * n - 2) * n.factorial + 17 * 21^n + 5).is_square ↔ n = 1 :=
sorry

end find_natural_integer_l699_699257


namespace mean_of_second_set_l699_699012

def mean (l: List ℕ) : ℚ :=
  (l.sum: ℚ) / l.length

theorem mean_of_second_set (x: ℕ) 
  (h: mean [28, x, 42, 78, 104] = 90): 
  mean [128, 255, 511, 1023, x] = 423 :=
by
  sorry

end mean_of_second_set_l699_699012


namespace mary_regular_hours_l699_699419

theorem mary_regular_hours (x y : ℕ) (h1 : 8 * x + 10 * y = 560) (h2 : x + y = 60) : x = 20 :=
by
  sorry

end mary_regular_hours_l699_699419


namespace integer_solution_of_inequalities_l699_699098

theorem integer_solution_of_inequalities :
  (∀ x : ℝ, 3 * x - 4 ≤ 6 * x - 2 → (2 * x + 1) / 3 - 1 < (x - 1) / 2 → (x = 0)) :=
sorry

end integer_solution_of_inequalities_l699_699098


namespace simplify_expression_l699_699860

-- Definitions as per constraints given in step a)
def a1 : ℝ := 0.7
def a2 : ℝ := 0.6
def a3 : ℝ := 0.3
def a4 : ℝ := -0.1
def a5 : ℝ := 0.5

-- Statement of the proof problem
theorem simplify_expression :
  (10^a1) * (10^a2) * (10^a3) * (10^a4) * (10^a5) = 10^2 :=
  by
    sorry

end simplify_expression_l699_699860


namespace calculate_expression_l699_699230

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l699_699230


namespace complex_imaginary_part_l699_699624

theorem complex_imaginary_part : 
  Complex.im ((1 : ℂ) / (-2 + Complex.I) + (1 : ℂ) / (1 - 2 * Complex.I)) = 1/5 := 
  sorry

end complex_imaginary_part_l699_699624


namespace full_day_students_count_l699_699997

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l699_699997


namespace general_formula_sequence_sum_first_n_terms_sequence_bn_l699_699830

variable {n : ℕ}

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := (n * (a 1 + a n)) / 2

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ b = 2 * a ∧ c = a * r + 1

theorem general_formula_sequence 
  (a : ℕ → ℕ) 
  (h1 : Sn a 5 = 15) 
  (h2 : is_geometric_sequence (2 * a 2) (a 6) (a 8 + 1)) : 
  ∀ n, a n = n :=
sorry

def bn (a : ℕ → ℕ) (n : ℕ) := (a n : ℚ) / 2^n

def Tn (a : ℕ → ℕ) (n : ℕ) : ℚ := ∑ i in finset.range n, bn a (i + 1)

theorem sum_first_n_terms_sequence_bn 
  (a : ℕ → ℕ)
  (h : ∀ n, a n = n) :
  ∀ n, Tn a n = 2 - (n + 2) / (2^n) :=
sorry

end general_formula_sequence_sum_first_n_terms_sequence_bn_l699_699830


namespace max_square_area_in_triangle_l699_699506

-- Let ABC be a triangle with area T.
variables (A B C : Type) [hm: has_area (triangle A B C)]
variable (T : ℝ) -- area of the triangle ABC

-- Let XYZV be a square with vertices on the sides of the triangle ABC.
variables (X Y Z V : Type) 
variable [hmsq: has_area (square X Y Z V)]

-- Given that side VZ of the square is parallel to side AB of the triangle.
variable (VZ_parallel_AB: ∀ (s1 s2 : Type), parallel s1 s2)

-- Prove that the maximum area of a square that fits the conditions is half the area of the triangle.
theorem max_square_area_in_triangle (h: hm = T) (h_vz_ab: VZ_parallel_AB VZ AB) : 
    (area (square X Y Z V)) ≤ T / 2 := 
sorry

end max_square_area_in_triangle_l699_699506


namespace monotonic_increasing_range_l699_699794

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699794


namespace inverse_self_implies_a_eq_2_max_value_of_y_when_a_gt_1_l699_699292

def f (a : ℝ) (x : ℝ) := log a (8 - 2*x)

theorem inverse_self_implies_a_eq_2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x, f a (f a x) = x) : a = 2 := 
sorry

theorem max_value_of_y_when_a_gt_1 (a : ℝ) (h1 : a > 1) : 
  (∃ x : ℝ, -4 < x ∧ x < 4 ∧ f a x + f a (-x) = 6) :=
sorry

end inverse_self_implies_a_eq_2_max_value_of_y_when_a_gt_1_l699_699292


namespace pos_int_solutions_eq_115_l699_699265

theorem pos_int_solutions_eq_115 :
  ∃ (x1 x2 x3 x4 : ℕ), x1 + x2 + x3 + x4 = 23 ∧
                           x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧
                           x1 ≤ 9 ∧ x2 ≤ 8 ∧ x3 ≤ 7 ∧ x4 ≤ 6 → finset.card { (x1, x2, x3, x4) // x1 + x2 + x3 + x4 = 23 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ x1 ≤ 9 ∧ x2 ≤ 8 ∧ x3 ≤ 7 ∧ x4 ≤ 6 } = 115 :=
sorry

end pos_int_solutions_eq_115_l699_699265


namespace range_of_a_l699_699811

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699811


namespace proof_tangent_length_l699_699868

def right_triangle (A B C : Type) [metric_space A] (a b c : A) : Prop :=
  ∃ (E : A), metric.dist a E = 0 ∧ metric.dist b E = 0 ∧ metric.dist c E = π / 2

noncomputable def tangent_length {A : Type} [metric_space A]
  (a b c : A) (DE DF : ℝ) (cond1 : DE = 7) (cond2 : DF = real.sqrt 85)
  (right_angle : right_triangle A b a c) : ℝ :=
  6 -- Correct answer

theorem proof_tangent_length :
  ∀ (A : Type) [metric_space A] (D E F : A) (DE DF : ℝ),
    DE = 7 →
    DF = real.sqrt 85 →
    right_triangle A D E F →
    ∃ Q : A, tangent_length D E F DE DF = 6 :=
by
  assume A _ D E F DE DF cond1 cond2 right_angle
  exists sorry

end proof_tangent_length_l699_699868


namespace cos_product_identity_l699_699846

theorem cos_product_identity :
  (cos (12 * π / 180) * cos (24 * π / 180) * cos (36 * π / 180) * cos (48 * π / 180) * cos (60 * π / 180) * cos (72 * π / 180) * cos (84 * π / 180) = (1 / 2) ^ 7) :=
by
  sorry

end cos_product_identity_l699_699846


namespace tom_saves_80_dollars_l699_699909

def normal_doctor_cost : ℝ := 200
def discount_percentage : ℝ := 0.7
def discount_clinic_cost_per_visit : ℝ := normal_doctor_cost * (1 - discount_percentage)
def number_of_visits : ℝ := 2
def total_discount_clinic_cost : ℝ := discount_clinic_cost_per_visit * number_of_visits
def savings : ℝ := normal_doctor_cost - total_discount_clinic_cost

theorem tom_saves_80_dollars : savings = 80 := by
  sorry

end tom_saves_80_dollars_l699_699909


namespace monotonic_increasing_range_l699_699780

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699780


namespace perpendicular_lines_l699_699352

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l699_699352


namespace largest_factor_and_smallest_multiple_of_18_l699_699241

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ x, (x ∈ {d : ℕ | d ∣ 18}) ∧ (∀ y, y ∈ {d : ℕ | d ∣ 18} → y ≤ x) ∧ x = 18)
  ∧ (∃ y, (y ∈ {m : ℕ | 18 ∣ m}) ∧ (∀ z, z ∈ {m : ℕ | 18 ∣ m} → y ≤ z) ∧ y = 18) :=
by
  sorry

end largest_factor_and_smallest_multiple_of_18_l699_699241


namespace convex_x_power_2n_convex_x_power_2n_plus_1_not_convex_cos_convex_exp_not_convex_step_function_l699_699580

noncomputable def step_function (x : ℝ) : ℝ :=
if x ≤ 0 then 0 else 1

theorem convex_x_power_2n {n : ℕ} (h : n ≥ 1) : convex_on ℝ (λ x : ℝ, x^(2 * n)) :=
sorry

theorem convex_x_power_2n_plus_1 {n : ℕ} (h : n ≥ 1) : convex_on (Set.Ioi 0) (λ x : ℝ, x^(2 * n + 1)) :=
sorry

theorem not_convex_cos :
  ¬ convex_on ℝ (λ x : ℝ, Real.cos x) :=
sorry

theorem convex_exp : convex_on ℝ (λ x : ℝ, Real.exp x) :=
sorry

theorem not_convex_step_function :
  ¬ convex_on ℝ step_function :=
sorry

end convex_x_power_2n_convex_x_power_2n_plus_1_not_convex_cos_convex_exp_not_convex_step_function_l699_699580


namespace larger_number_is_seventy_two_l699_699889

theorem larger_number_is_seventy_two (a b : ℕ) (h1 : a.gcd b = 1) (h2 : a*b = 120) (h3 : a ≠ b) : 
  (3 * nat.coprime.gcd_iff_dvd_of_nat.prime_dvd_four_mod_pone a think-of-a-b<Value>) = 72 :=
by
  sorry

end larger_number_is_seventy_two_l699_699889


namespace prove_monotonic_increasing_range_l699_699798

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699798


namespace calculate_expression_l699_699229

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l699_699229


namespace range_of_a_l699_699786

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699786


namespace monotonically_increasing_range_l699_699766

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699766


namespace john_spends_total_amount_l699_699390

theorem john_spends_total_amount :
  let total_time := (4 * 60 + 35) in
  let break_time := 5 * 10 in
  let actual_playing_time := total_time - break_time in
  let first_hour_expenditure := (60 / 4) * 0.50 in
  let second_hour_expenditure := (60 / 5) * 0.75 in
  let third_hour_expenditure := (60 / 3) * 1.00 in
  let fourth_hour_time := 35 in
  let fourth_hour_expenditure := (fourth_hour_time / 7) * 1.25 in
  let total_expenditure := first_hour_expenditure + second_hour_expenditure +
                           third_hour_expenditure + fourth_hour_expenditure in
  total_expenditure = 42.75 :=
by
  let total_time := (4 * 60 + 35)
  let break_time := 5 * 10
  let actual_playing_time := total_time - break_time
  let first_hour_expenditure := (60 / 4) * 0.50
  let second_hour_expenditure := (60 / 5) * 0.75
  let third_hour_expenditure := (60 / 3) * 1.00
  let fourth_hour_time := 35
  let fourth_hour_expenditure := (fourth_hour_time / 7) * 1.25
  let total_expenditure := first_hour_expenditure + second_hour_expenditure +
                           third_hour_expenditure + fourth_hour_expenditure
  show total_expenditure = 42.75
  sorry  -- Proof omitted

end john_spends_total_amount_l699_699390


namespace arithmetic_sequence_a7_l699_699296

theorem arithmetic_sequence_a7 :
  (∃ (a : ℕ → ℕ), (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ (a 4 + a 9 = 24) ∧ (a 6 = 11)) →
  (∃ d a1 : ℕ, ∀ n : ℕ, a n = a1 + n * d) →
  (a 7 = 13) :=
sorry

end arithmetic_sequence_a7_l699_699296


namespace arithmetic_mean_minimal_l699_699649

theorem arithmetic_mean_minimal (a1 a2 a3 : ℝ) (h1 : 2 * a1 + 3 * a2 + a3 = 1) (h2 : a1 > 0) (h3 : a2 > 0) (h4 : a3 > 0) : 
  (let a := a1 + a2 in let b := a2 + a3 in (min ((1:ℝ) / a) ((1:ℝ) / b) = (3 + 2 * Real.sqrt 2) / 2)) := 
sorry

end arithmetic_mean_minimal_l699_699649


namespace AQB_perpendicular_l699_699950

-- Given Conditions:
variables {Ω1 Ω2 : Type} [metric_space Ω1] [metric_space Ω2]
variables {O1 O2 P Q A B : Ω1}
variables {r1 r2 : ℝ}
variables {line_through_P : set Ω1}

-- Conditions:
-- O1 and O2 are centers of circles ω1 and ω2
def is_center (O : Ω1) (C : set Ω1) := ∀ x ∈ C, dist O x = r2

-- P and Q are intersection points
def on_both_circles (P : Ω1) (ω1 ω2 : set Ω1) := P ∈ ω1 ∧ P ∈ ω2
def on_circle (P : Ω1) (C : set Ω1) := dist O1 P = r1 ∧ dist O2 P = r2

-- Tangent conditions
def is_tangent (O1 P : Ω1) (ω2 : set Ω1) := tangent_line through_point P on_circle ω2

-- Line through P intersects both circles at A and B respectively
def intersects (P : Ω1) (line : set Ω1) (C : set Ω1) := ∃ A B ∈ line, (A ∈ C ∧ B ∈ C)

-- To Prove:
theorem AQB_perpendicular :
  is_center O1 Ω1 ∧ is_center O2 Ω2 ∧
  on_both_circles P {O1, O2} ∧ on_both_circles Q {O1, O2} ∧
  is_tangent O1 P {O2} ∧ is_tangent O1 Q {O2} ∧
  intersects P line_through_P {O1, O2} →
  ∀ A B : Ω1, A ∈ Ω1 ∧ B ∈ Ω1 →
  ∠AQB = 90 :=
begin
  sorry
end

end AQB_perpendicular_l699_699950


namespace coloring_ways_l699_699078

theorem coloring_ways (multiples_of_three : Set ℕ) (color : ℕ → ℕ → Prop) :
  (∀ b r ∈ multiples_of_three, (color b 1 = 1 ∧ color r 0 = 0) → color b 0 = 0 → color (b + r) 0 = 0) ∧
  (∀ b r ∈ multiples_of_three, (color b 1 = 1 ∧ color r 0 = 0) → color b 0 = 0 → color (b * r) 1 = 1) ∧
  (∃ n ∈ multiples_of_three, n = 546 ∧ color n 1 = 1) ↔ 
  ∃ (count = 7), True :=
by
  sorry

end coloring_ways_l699_699078


namespace hamiltonian_exists_2015_l699_699454

axiom checkerboard_hamiltonian_path_exists :
  ∀ n : ℕ, 
    let dim := 2 * n + 1,
        cells := { (i, j) | 0 ≤ i ∧ i < dim ∧ 0 ≤ j ∧ j < dim },
        black_cells := { (i, j) ∈ cells | (i + j) % 2 = 0 },
        adjacent (c1 c2 : ℕ × ℕ) := 
          (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
          (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1)),
        valid_tour (path : list (ℕ × ℕ)) := 
          ∀ p ∈ cells, ∃ i, path.nth i = some p ∧
          (∀ i j, path.nth i ≠ none ∧ path.nth j ≠ none 
            → i ≠ j → path.nth i ≠ path.nth j) ∧
          ∀ i, i < path.length - 1 → 
            adjacent (path.nth_le i (Nat.lt_sub_right_of_add (Nat.lt_sub_right_of_add path.length (by simp)) i).left) 
                     (path.nth_le (i + 1) (Nat.lt_sub_right_of_add 
                     path.length (by simp)).left),
        black_cells_non_empty : ∃ (A B : (ℕ × ℕ)), 
          A ∈ black_cells ∧ B ∈ black_cells :=
  ∀ A B, A ∈ black_cells → B ∈ black_cells 
    → ∃ path : list (ℕ × ℕ), valid_tour path ∧ path.head = some A ∧ path.last = some B
 
theorem hamiltonian_exists_2015 :
  checkerboard_hamiltonian_path_exists 1007 := 
begin 
  sorry 
end

end hamiltonian_exists_2015_l699_699454


namespace area_of_square_with_given_diagonal_l699_699564

theorem area_of_square_with_given_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : ∃ (A : ℝ), A = 64 :=
by
  use (8 * 8)
  sorry

end area_of_square_with_given_diagonal_l699_699564


namespace particle_movement_probabilities_min_time_to_reach_C_probability_of_simultaneous_arrival_l699_699498

theorem particle_movement_probabilities :
  let x : ℚ := 1 - (1/4 + 1/4 + 1/3)
  let y : ℚ := 1 / 4
  x = 1/6 ∧ y = 1/4 := by
  simp
  sorry

theorem min_time_to_reach_C : 
  ∀ (A_moves B_moves : ℕ), 
  (A_moves = 3 ∧ B_moves = 1) → (min_time : ℕ) (min_time = 3) := by
  intro A_moves B_moves
  intro h
  exact 3
  sorry

theorem probability_of_simultaneous_arrival :
  ∀ (pA pB : ℚ),
  (pA = 3 * (1/4)^2 * (1/3)) ∧ (pB = (1/4)^3 * 9) →
  let combined_prob := pA * pB
  combined_prob = 9 / 1024 := by
  intro pA pB
  intro h
  exact (pA * pB)
  sorry

end particle_movement_probabilities_min_time_to_reach_C_probability_of_simultaneous_arrival_l699_699498


namespace melissa_bananas_l699_699420

theorem melissa_bananas (a b : ℕ) (h1 : a = 88) (h2 : b = 4) : a - b = 84 :=
by
  sorry

end melissa_bananas_l699_699420


namespace inequality_solution_set_inequality_proof_l699_699326

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem inequality_solution_set :
  ∀ x : ℝ, -2 < f x ∧ f x < 0 ↔ -1/2 < x ∧ x < 1/2 :=
by
  sorry

theorem inequality_proof (m n : ℝ) (h_m : -1/2 < m ∧ m < 1/2) (h_n : -1/2 < n ∧ n < 1/2) :
  |1 - 4 * m * n| > 2 * |m - n| :=
by
  sorry

end inequality_solution_set_inequality_proof_l699_699326


namespace ratio_of_area_ABJMO_to_EFCDMO_l699_699029

noncomputable def decagon_area_ratio : Prop :=
  ∀ (A B C D E F G H I J O M N: Point), 
    regular_decagon A B C D E F G H I J →
    midpoint B C M →
    midpoint H I N →
    O ≠ M →
    (area (polygon_to_list [A, B, J, M, O]) /
     area (polygon_to_list [E, F, C, D, M, O])) = 2 / 3

theorem ratio_of_area_ABJMO_to_EFCDMO :
  decagon_area_ratio :=
sorry

end ratio_of_area_ABJMO_to_EFCDMO_l699_699029


namespace range_of_a_l699_699765

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699765


namespace dropped_student_score_l699_699873

theorem dropped_student_score (avg_all : ℚ) (avg_remaining : ℚ) (n : ℕ) (score : ℚ) (x : ℚ)  :
  avg_all = 60.5 → avg_remaining = 64 → n = 16 → 
  score = n * avg_all - (n-1) * avg_remaining → 
  x = score → 
  x = 8 :=
by
  intro h1 h2 h3 h4 h5
  have H : n * avg_all - (n-1) * avg_remaining = 8 := sorry 
  rw ← h5
  rw ← H
  exact h4

end dropped_student_score_l699_699873


namespace formula_for_an_l699_699650

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 3 * a n + 2)

theorem formula_for_an (a : ℕ → ℝ) (h : sequence a) : 
  ∀ n, a n = 2 * 3 ^ (n - 1) - 1 :=
by sorry

end formula_for_an_l699_699650


namespace James_future_age_when_Thomas_reaches_James_current_age_l699_699484

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l699_699484


namespace joes_team_draws_l699_699748

theorem joes_team_draws 
  (win_points tie_points : ℕ) 
  (joes_wins joes_points_diff first_place_wins first_place_ties : ℕ)
  (first_place_diff_points : joes_points_diff = 2)
  (joes_wins_eq : joes_wins = 1)
  (first_place_eq : first_place_wins = 2 ∧ first_place_ties = 2)
  (score_system : win_points = 3 ∧ tie_points = 1) 
  : ∃ (joes_ties : ℕ), joes_ties = 3 := 
by
  have points_first_place := first_place_wins * win_points + first_place_ties * tie_points
  have points_joes := joes_wins * win_points + tie_points * joes_ties
  have h1 : points_first_place - joes_points_diff = points_joes by
    rw [first_place_eq, score_system, joes_wins_eq, first_place_diff_points]
    have points_first_place_eq := 2 * 3 + 2 * 1
    have points_joes_eq := 1 * 3 + tie_points * joes_ties
    rw [points_first_place_eq, points_joes] at h1
    linarith
  have h2 : 6 + tie_points * joes_ties = 6 by
    rw [score_system, joes_wins_eq, joes_ties]
    linarith
  have h3 : tie_points * joes_ties = 3 by
    rw [score_system]
    linarith
  have h4 : joes_ties = 3 by
    linarith
  exact ⟨joes_ties, h4⟩

end joes_team_draws_l699_699748


namespace pizza_shared_cost_l699_699585

theorem pizza_shared_cost (total_price : ℕ) (num_people : ℕ) (share: ℕ)
  (h1 : total_price = 40) (h2 : num_people = 5) : share = 8 :=
by
  sorry

end pizza_shared_cost_l699_699585


namespace number_from_1198th_to_1200th_digit_is_473_l699_699213

def sequence_with_first_digit_1_or_2 : List Nat :=
  List.filter (λ n => Nat.digits 10 n ≠ [] ∧ (List.head (Nat.digits 10 n) = some 1 ∨ List.head (Nat.digits 10 n) = some 2))
    (List.range (299 + 1))

def digits_up_to_index (lst : List Nat) (idx : Nat) : List Nat :=
  lst.foldl (λ acc n => acc.append (Nat.digits 10 n).reverse) [] |>.take idx

def number_from_1198th_to_1200th_digit : Nat :=
  Nat.ofDigits 10 ((digits_up_to_index sequence_with_first_digit_1_or_2 1200)[1197:1200])

theorem number_from_1198th_to_1200th_digit_is_473 :
  number_from_1198th_to_1200th_digit = 473 :=
sorry

end number_from_1198th_to_1200th_digit_is_473_l699_699213


namespace probability_sum_of_two_dice_equals_third_die_l699_699905

def eight_sided_die_probability (a b c : ℕ) : Prop :=
  a + b = c ∨ a + c = b ∨ b + c = a

theorem probability_sum_of_two_dice_equals_third_die :
  let outcomes := finset.univ.product (finset.univ.product finset.univ)
  let favorable_outcomes := outcomes.filter (λ ((a, b), c), eight_sided_die_probability a b c)
  let total_outcomes := 512
  let probability := favorable_outcomes.card
  probability / total_outcomes = 267 / 512 :=
sorry

end probability_sum_of_two_dice_equals_third_die_l699_699905


namespace georgia_black_buttons_l699_699278

theorem georgia_black_buttons : 
  ∀ (B : ℕ), 
  (4 + B + 3 = 9) → 
  B = 2 :=
by
  introv h
  linarith

end georgia_black_buttons_l699_699278


namespace perimeter_of_DEF_l699_699360

-- Define the given conditions
variables (D E F G H : Type)
variables (DE DF : ℝ) (EF : ℝ)
variables (isosceles_right_triangle : Prop)
variables (DE_eq_10 : DE = 10)
variables (angle_E_90 : ∃ E, ∠ DEF = 90)
variables (DE_eq_DF : DE = DF)

-- State the theorem to prove the perimeter
theorem perimeter_of_DEF (D E F G H : Type)
    (DE DF : ℝ) (EF : ℝ)
    (isosceles_right_triangle: ∃ DEF, isosceles_right_triangle ∧ ∠ DEF = 90)
    (DE_eq_10 : DE = 10)
    (DE_eq_DF : DE = DF) :
    DE + DF + EF = 20 + 10 * real.sqrt 2 :=
sorry

end perimeter_of_DEF_l699_699360


namespace martha_cakes_l699_699834

theorem martha_cakes :
  ∀ (n : ℕ), (∀ (c : ℕ), c = 3 → (∀ (k : ℕ), k = 6 → n = c * k)) → n = 18 :=
by
  intros n h
  specialize h 3 rfl 6 rfl
  exact h

end martha_cakes_l699_699834


namespace derivative_at_minus_one_l699_699327
open Real

def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1 / 2) * (3 + x^3)^(1 / 3)

theorem derivative_at_minus_one : deriv f (-1) = sqrt 3 * 2^(1 / 3) :=
by sorry

end derivative_at_minus_one_l699_699327


namespace employed_females_percentage_l699_699521

variable (P : ℝ) -- Total population of town X
variable (E_P : ℝ) -- Percentage of the population that is employed
variable (M_E_P : ℝ) -- Percentage of the population that are employed males

-- Conditions
axiom h1 : E_P = 0.64
axiom h2 : M_E_P = 0.55

-- Target: Prove the percentage of employed people in town X that are females
theorem employed_females_percentage (h : P > 0) : 
  (E_P * P - M_E_P * P) / (E_P * P) * 100 = 14.06 := by
sorry

end employed_females_percentage_l699_699521


namespace division_probability_l699_699465

def r_set := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}
def k_set := {3, 5, 7, 9}

def valid_division_pairs := 
  {p : ℤ × ℤ // p.1 ∈ r_set ∧ p.2 ∈ k_set ∧ p.1 % p.2 = 0}

theorem division_probability : 
  (valid_division_pairs.to_finset.card : ℚ) / ((r_set.card * k_set.card) : ℚ) = 1 / 6 :=
by
  sorry

end division_probability_l699_699465


namespace hash_value_is_minus_15_l699_699415

def hash (a b c : ℝ) : ℝ := b^2 - 3 * a * c

theorem hash_value_is_minus_15 : hash 2 3 4 = -15 :=
by
  sorry

end hash_value_is_minus_15_l699_699415


namespace value_of_sqrt_2a_plus_b_l699_699668

theorem value_of_sqrt_2a_plus_b
  (a b : ℝ)
  (h1 : sqrt (2 * a - 1) = 3 ∨ sqrt (2 * a - 1) = -3)
  (h2 : real.cbrt (a - 2 * b + 1) = 2) :
  sqrt (2 * a + b) = 3 :=
sorry

end value_of_sqrt_2a_plus_b_l699_699668


namespace area_of_quadrilateral_PQRS_l699_699602

/-- Representation of a rectangular prism and slicing plane conditions. -/
structure PrismSlice where
  P : ℝ × ℝ × ℝ := (0,0,0)
  Q : ℝ × ℝ × ℝ := (2,1,3)
  R : ℝ × ℝ × ℝ := (0,1,1.5)
  S : ℝ × ℝ × ℝ := (2,0,1.5)

/-- The area of the quadrilateral PQRS formed by slicing the rectangular prism. -/
def quadrilateral_area (slice : PrismSlice) : ℝ :=
  1/2 * real.sqrt 70

theorem area_of_quadrilateral_PQRS (slice : PrismSlice) :
  quadrilateral_area slice = real.sqrt 70 / 2 := by
  sorry

end area_of_quadrilateral_PQRS_l699_699602


namespace quadratic_function_incorrect_statement_l699_699277

theorem quadratic_function_incorrect_statement (x : ℝ) : 
  ∀ y : ℝ, y = -(x + 2)^2 - 1 → ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = 0 ∧ -(x1 + 2)^2 - 1 = 0 ∧ -(x2 + 2)^2 - 1 = 0) :=
by 
sorry

end quadratic_function_incorrect_statement_l699_699277


namespace ratio_of_means_l699_699414

-- Variables for means
variables (xbar ybar zbar : ℝ)
-- Variables for sample sizes
variables (m n : ℕ)

-- Given conditions
def mean_x (x : ℕ) (xbar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ x → xbar = xbar
def mean_y (y : ℕ) (ybar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ y → ybar = ybar
def combined_mean (m n : ℕ) (xbar ybar zbar : ℝ) := zbar = (1/4) * xbar + (3/4) * ybar

-- Assertion to be proved
theorem ratio_of_means (h1 : mean_x m xbar) (h2 : mean_y n ybar)
  (h3 : xbar ≠ ybar) (h4 : combined_mean m n xbar ybar zbar) :
  m / n = 1 / 3 := sorry

end ratio_of_means_l699_699414


namespace katy_brownies_l699_699399

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l699_699399


namespace seq_properties_l699_699313

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x

theorem seq_properties :
  (∀ n, a_n = -2 * (1 / 3) ^ n) ∧
  (∀ n, b_n = 2 * n - 1) ∧
  (∀ t m, (-1 ≤ m ∧ m ≤ 1) → (t^2 - 2 * m * t + 1/2 > T_n) ↔ (t < -2 ∨ t > 2)) ∧
  (∃ m n, 1 < m ∧ m < n ∧ T_1 * T_n = T_m^2 ∧ m = 2 ∧ n = 12) :=
sorry

end seq_properties_l699_699313


namespace rectangle_longer_side_length_l699_699967

theorem rectangle_longer_side_length (r : ℝ) (h1 : r = 4) 
  (h2 : ∃ w l, w * l = 2 * (π * r^2) ∧ w = 2 * r) : 
  ∃ l, l = 4 * π :=
by 
  obtain ⟨w, l, h_area, h_shorter_side⟩ := h2
  sorry

end rectangle_longer_side_length_l699_699967


namespace number_of_valid_pairings_l699_699892

-- Definition for the problem
def validPairingCount (n : ℕ) (k: ℕ) : ℕ :=
  sorry -- Calculating the valid number of pairings is deferred

-- The problem statement to be proven:
theorem number_of_valid_pairings : validPairingCount 12 3 = 14 :=
sorry

end number_of_valid_pairings_l699_699892


namespace bran_tuition_fee_l699_699217

theorem bran_tuition_fee (P : ℝ) (S : ℝ) (M : ℕ) (R : ℝ) (T : ℝ) 
  (h1 : P = 15) (h2 : S = 0.30) (h3 : M = 3) (h4 : R = 18) 
  (h5 : 0.70 * T - (M * P) = R) : T = 90 :=
by
  sorry

end bran_tuition_fee_l699_699217


namespace least_positive_integer_y_l699_699015

theorem least_positive_integer_y (x k y: ℤ) (h1: 24 * x + k * y = 4) (h2: ∃ x: ℤ, ∃ y: ℤ, 24 * x + k * y = 4) : y = 4 :=
sorry

end least_positive_integer_y_l699_699015


namespace tree_boy_growth_ratio_l699_699552

theorem tree_boy_growth_ratio 
    (initial_tree_height final_tree_height initial_boy_height final_boy_height : ℕ) 
    (h₀ : initial_tree_height = 16) 
    (h₁ : final_tree_height = 40) 
    (h₂ : initial_boy_height = 24) 
    (h₃ : final_boy_height = 36) 
:
  (final_tree_height - initial_tree_height) / (final_boy_height - initial_boy_height) = 2 := 
by {
    -- Definitions and given conditions used in the statement part of the proof
    sorry
}

end tree_boy_growth_ratio_l699_699552


namespace line_perpendicular_value_of_a_l699_699354

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l699_699354


namespace monotonic_increasing_range_l699_699796

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699796


namespace sin_value_in_right_triangle_l699_699729

theorem sin_value_in_right_triangle 
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2)
  (h_sin_cos : 3 * (b / c) = 4 * (a / c)) 
  : b / c = 4 / 5 := 
begin
  sorry
end

end sin_value_in_right_triangle_l699_699729


namespace triangle_ratio_property_l699_699017

theorem triangle_ratio_property
  (A : ℝ) (a : ℝ) (c : ℝ) (C : ℝ)
  (hA : A = 60) (ha : a = sqrt 3) :
  c / sin C = 2 :=
sorry

end triangle_ratio_property_l699_699017


namespace partial_fractions_l699_699623

theorem partial_fractions (A B C : ℝ) :
  A = -5 → B = 1 → C = 3 →
  ∀ x : ℝ, x ≠ 0 → x ≠ ι * sqrt (-1) → -- ensures no division by zero
  (x^3 - 2*x^2 + x - 5) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1) :=
by
  intros hA hB hC x hx_ne_zero hx_ne_i
  sorry

end partial_fractions_l699_699623


namespace find_a_for_parallel_lines_l699_699070

-- Define the lines l1 and l2 as given in the problem
def line_l1 (a : ℝ) : Prop := ∀ x y : ℝ, ax + 3 * y + a^2 - 5 = 0
def line_l2 (a : ℝ) : Prop := ∀ x y : ℝ, x + (a - 2) * y + 4 = 0

-- Define the condition for the lines to be parallel
def lines_parallel (a : ℝ) : Prop :=
  a / 1 = 3 / (a - 2)

-- The main problem statement
theorem find_a_for_parallel_lines (a : ℝ) : lines_parallel a → a = 3 :=
sorry

end find_a_for_parallel_lines_l699_699070


namespace contrapositive_l699_699076

variables (Player : Type) (attends_all_sessions : Player → Prop) (arrives_on_time : Player → Prop) (considered_for_captain : Player → Prop)

def P (p : Player) := attends_all_sessions p ∧ arrives_on_time p
def Q (p : Player) := considered_for_captain p

theorem contrapositive 
    (h : ∀ p, P p → Q p) 
    (p : Player) : 
    ¬ (considered_for_captain p) → (¬ (attends_all_sessions p) ∨ ¬ (arrives_on_time p)) :=
by sorry

end contrapositive_l699_699076


namespace probability_of_valid_pair_is_one_seventh_l699_699358

open_locale BigOperators

def numbers : Set ℕ := {4, 6, 20, 25, 30, 75, 100}

def is_valid_pair (a b : ℕ) : Prop :=
  (a * b) % 200 = 0

def valid_pairs_count : ℕ :=
  (numbers.to_finset.powerset.filter (λ s, s.card = 2 ∧ ∃ a b, a ∈ s ∧ b ∈ s ∧ is_valid_pair a b)).card

def total_pairs_count : ℕ :=
  (numbers.to_finset.powerset.filter (λ s, s.card = 2)).card

def probability_valid_pair : ℚ :=
  valid_pairs_count / total_pairs_count

theorem probability_of_valid_pair_is_one_seventh : probability_valid_pair = 1 / 7 :=
by {
  -- Proof ellided
  sorry
}

end probability_of_valid_pair_is_one_seventh_l699_699358


namespace artist_paints_49_square_meters_l699_699576

/-- Proof that the artist paints 49 square meters of exposed surfaces -/
theorem artist_paints_49_square_meters :
  let edge_length := 1
  let total_cubes := 16
  let first_layer := (3, 3)  -- 3x3 square
  let second_layer := (2, 2) -- 2x2 square
  let exposed_area_first_layer := 29  -- pre-computed exposed area for the first layer
  let exposed_area_second_layer := 20 -- pre-computed exposed area for the second layer
  in exposed_area_first_layer + exposed_area_second_layer = 49 :=
by
  intro edge_length total_cubes first_layer second_layer exposed_area_first_layer exposed_area_second_layer
  have h1 : edge_length = 1 := by sorry
  have h2 : total_cubes = 16 := by sorry
  have h3 : first_layer = (3, 3) := by sorry
  have h4 : second_layer = (2, 2) := by sorry
  have h5 : exposed_area_first_layer = 29 := by sorry
  have h6 : exposed_area_second_layer = 20 := by sorry
  show exposed_area_first_layer + exposed_area_second_layer = 49 from
    by rw [h5, h6]; exact rfl

end artist_paints_49_square_meters_l699_699576


namespace min_value_inequality_l699_699655

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) : 
  (2 / a + 3 / b) ≥ 14 :=
sorry

end min_value_inequality_l699_699655


namespace total_distance_traveled_l699_699612

noncomputable def travel_distance : ℝ :=
  1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640

theorem total_distance_traveled :
  let n := 8
  let r := 40
  let theta := 2 * Real.pi / n
  let d_2arcs := 2 * r * Real.sin (theta)
  let d_3arcs := r * (2 + Real.sqrt (2))
  let d_4arcs := 2 * r
  (8 * (4 * d_2arcs + 2 * d_3arcs + d_4arcs)) = travel_distance := by
  sorry

end total_distance_traveled_l699_699612


namespace problem_statement_l699_699346

theorem problem_statement (c d : ℤ) (h1 : 5 + c = 7 - d) (h2 : 6 + d = 10 + c) : 5 - c = 6 := 
by {
  sorry
}

end problem_statement_l699_699346


namespace positive_integer_not_in_S_l699_699848

noncomputable def S : Set ℤ :=
  {n | ∃ (i : ℕ), n = 4^i * 3 ∨ n = -4^i * 2}

theorem positive_integer_not_in_S (n : ℤ) (hn : 0 < n) (hnS : n ∉ S) :
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ x + y = n :=
sorry

end positive_integer_not_in_S_l699_699848


namespace part1_part2_l699_699284

-- Define the function f(x)
def f (x a k : ℝ) := Real.exp (x - a) - (k / a) * x^2

-- Part 1: Prove that f(x) is monotonically increasing given specific conditions
theorem part1 (x : ℝ) (h_a : 1 = 1) (h_k : k = 1/2) : ∀ x, (0 ≤ Real.exp (x - 1) - x) :=
sorry

-- Part 2: Prove the range of k and show x2 + x3 > 4 given the condition
theorem part2 (a : ℝ) (h_a : a > 0) (k: ℝ) (h_k_range : k > Real.exp (2 - a) * a / 4) : 
∃ x1 x2 x3 : ℝ, (x1 < x2) ∧ (x2 < x3) ∧ (f x1 a k = 0) ∧ (f x2 a k = 0) ∧ (f x3 a k = 0) ∧ (x2 + x3 > 4) :=
sorry

end part1_part2_l699_699284


namespace P_sum_positive_l699_699401

noncomputable def Q (x : ℝ) : ℝ := sorry  -- Q(x) is a quadratic trinomial

def P (x : ℝ) : ℝ := x^2 * Q x

axiom Q_increasing_on_positive : ∀ x y : ℝ, 0 < x → x < y → P x < P y

theorem P_sum_positive (x y z : ℝ) (h1 : x + y + z > 0) (h2 : x * y * z > 0) : P x + P y + P z > 0 :=
sorry

end P_sum_positive_l699_699401


namespace csc_240_eq_l699_699621

-- Define the sine function for the given angle in radians
noncomputable def sin_240 : ℝ := - (Mathlib.sqrt 3) / 2

-- Define the cosecant function
noncomputable def csc (x : ℝ) : ℝ := 1 / Real.sin x 

-- State the theorem
theorem csc_240_eq : csc (4 * Real.pi / 3) = - (2 * Mathlib.sqrt 3) / 3 := by
  sorry

end csc_240_eq_l699_699621


namespace two_non_coincident_planes_divide_space_l699_699152

-- Define conditions for non-coincident planes
def non_coincident_planes (P₁ P₂ : Plane) : Prop :=
  ¬(P₁ = P₂)

-- Define the main theorem based on the conditions and the question
theorem two_non_coincident_planes_divide_space (P₁ P₂ : Plane) 
  (h : non_coincident_planes P₁ P₂) :
  ∃ n : ℕ, n = 3 ∨ n = 4 :=
by
  sorry

end two_non_coincident_planes_divide_space_l699_699152


namespace last_digit_of_frac_l699_699926

noncomputable theory
open_locale classical

theorem last_digit_of_frac (N : ℤ) (hN : N = 2^15) :
  (∃ k : ℤ, (1 / (N : ℝ)) = (5^15) / 10^15 * 10^(-15 * k)) → last_digit((1 / (N : ℝ))) = 5 :=
by {
  sorry
}

end last_digit_of_frac_l699_699926


namespace reflection_midpoints_l699_699294

-- Defining the reflections of points
noncomputable definition reflect (p₁ p₂ : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p₂.1 - p₁.1, 2 * p₂.2 - p₁.2)

-- The main theorem statement for the problem
theorem reflection_midpoints (A B C D : ℝ × ℝ) :
  let A1 := reflect A B
  let B1 := reflect B C
  let C1 := reflect C D
  let D1 := reflect D A
  let E := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let F := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let G := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let H := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let E1 := ((A1.1 + B1.1) / 2, (A1.2 + B1.2) / 2)
  let F1 := ((B1.1 + C1.1) / 2, (B1.2 + C1.2) / 2)
  let G1 := ((C1.1 + D1.1) / 2, (C1.2 + D1.2) / 2)
  let H1 := ((D1.1 + A1.1) / 2, (D1.2 + A1.2) / 2)
  (E1, F1, G1, H1) = (E, F, G, H) :=
by
  sorry

end reflection_midpoints_l699_699294


namespace dan_car_mileage_l699_699606

noncomputable def cost_per_gallon := 4 : ℝ
noncomputable def total_gas_cost := 38 : ℝ
noncomputable def total_miles := 304 : ℝ

theorem dan_car_mileage : (total_miles / (total_gas_cost / cost_per_gallon)) = 32 := by
  sorry

end dan_car_mileage_l699_699606


namespace words_lost_equal_137_l699_699734

-- Definitions based on conditions
def letters_in_oz : ℕ := 68
def forbidden_letter_index : ℕ := 7

def words_lost_due_to_forbidden_letter : ℕ :=
  let one_letter_words_lost : ℕ := 1
  let two_letter_words_lost : ℕ := 2 * (letters_in_oz - 1)
  one_letter_words_lost + two_letter_words_lost

-- Theorem stating that the words lost due to prohibition is 137
theorem words_lost_equal_137 :
  words_lost_due_to_forbidden_letter = 137 :=
sorry

end words_lost_equal_137_l699_699734


namespace incenter_proof_l699_699737

variables {A B C P Q R S : Type}
variables [triangle A B C] [altitude AD BE CF] [circumcenter P of triangle A B C]
variables {circumradius : ℝ} [circumradius_eq PQ QR RS circumradius]
variables {directions : ℝ} [direction PQ AD] [direction QR BE] [direction RS CF]

theorem incenter_proof :
  is_incentre S (triangle A B C) →
  circumcenter P (triangle A B C) →
  circumradius_eq PQ QR RS circumradius →
  direction PQ AD →
  direction QR BE →
  direction RS CF →
  S = incenter (triangle A B C) := sorry

end incenter_proof_l699_699737


namespace percentage_of_silver_in_final_solution_l699_699721

noncomputable section -- because we deal with real numbers and division

variable (volume_4pct : ℝ) (percentage_4pct : ℝ)
variable (volume_10pct : ℝ) (percentage_10pct : ℝ)

def final_percentage_silver (v4 : ℝ) (p4 : ℝ) (v10 : ℝ) (p10 : ℝ) : ℝ :=
  let total_silver := v4 * p4 + v10 * p10
  let total_volume := v4 + v10
  (total_silver / total_volume) * 100

theorem percentage_of_silver_in_final_solution :
  final_percentage_silver 5 0.04 2.5 0.10 = 6 := by
  sorry

end percentage_of_silver_in_final_solution_l699_699721


namespace find_monthly_salary_l699_699974

variable (S : ℝ) (savings : ℝ)
variables (initial_savings_rate : ℝ) (increased_expense_rate : ℝ)
variable (post_increase_savings : ℝ)

-- Given conditions
def conditions : Prop := 
  initial_savings_rate = 0.25 ∧ 
  increased_expense_rate = 0.825 ∧ 
  post_increase_savings = 175

-- Man's monthly salary to be proven
theorem find_monthly_salary (h : conditions) : S = 1000 :=
  by
    sorry

end find_monthly_salary_l699_699974


namespace tetrahedron_OABC_volume_l699_699496

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ := 
  (1 / 6) * a * b * c

theorem tetrahedron_OABC_volume :
  ∃ (a b c : ℝ),
    (√(a^2 + b^2) = 5) ∧
    (√(b^2 + c^2) = 6) ∧
    (√(c^2 + a^2) = 7) ∧
    tetrahedron_volume a b c = √95 :=
begin
  sorry
end

end tetrahedron_OABC_volume_l699_699496


namespace mean_first_set_l699_699130

noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length

theorem mean_first_set (x : ℚ) (h : mean [128, 255, 511, 1023, x] = 423) :
  mean [28, x, 42, 78, 104] = 90 :=
sorry

end mean_first_set_l699_699130


namespace solve_triang_c_l699_699719

noncomputable theory

def problem_triang_c (A B : ℝ) (a : ℝ) : ℝ :=
  if A = 45 ∧ B = 45 ∧ a = Real.sqrt 6 then Real.sqrt 12 else 0

theorem solve_triang_c (A B a : ℝ) (hA : A = 45) (hB : B = 45) (ha : a = Real.sqrt 6) :
  problem_triang_c A B a = 2 * Real.sqrt 3 :=
by
  subst hA
  subst hB
  subst ha
  simp [problem_triang_c, Real.sqrt_six_eq_two_sqrt_three, Real.sqrt_eq_sqrt]
  sorry

end solve_triang_c_l699_699719


namespace min_value_of_a_l699_699638

theorem min_value_of_a (a : ℝ) (x : ℝ) (h1: 0 < a) (h2: a ≠ 1) (h3: 1 ≤ x → a^x ≥ a * x) : a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l699_699638


namespace number_of_hardbacks_l699_699897

theorem number_of_hardbacks (H P : ℕ) (books total_books selections : ℕ) (comb : ℕ → ℕ → ℕ) :
  total_books = 8 →
  P = 2 →
  comb total_books 3 - comb H 3 = 36 →
  H = 6 :=
by sorry

end number_of_hardbacks_l699_699897


namespace arithmetic_mean_first_n_odd_integers_l699_699871

theorem arithmetic_mean_first_n_odd_integers (n : ℕ) :
  let sum_first_n_odd := ∑ i in finset.range n, (2 * i + 1)
  sum_first_n_odd = n^2 → (sum_first_n_odd / n : ℚ) = n := by
  sorry

end arithmetic_mean_first_n_odd_integers_l699_699871


namespace smallest_6_digit_divisible_by_111_l699_699509

theorem smallest_6_digit_divisible_by_111 :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 111 = 0 ∧ x = 100011 :=
  by
    sorry

end smallest_6_digit_divisible_by_111_l699_699509


namespace range_of_a_l699_699785

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l699_699785


namespace gray_part_area_l699_699500

theorem gray_part_area (A B : Type) [has_add A] [has_mul A] (a1 a2 : A) (b1 b2 : A) (black_part : A) (gray_part : A) : 
  a1 = 8 → a2 = 10 → b1 = 12 → b2 = 9 → black_part = 37 → gray_part = 65 →
  ((a1 * a2) - black_part) + ((b1 * b2) - ((a1 * a2) - black_part)) = gray_part :=
by 
  intros
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end gray_part_area_l699_699500


namespace find_f_prime_at_1_l699_699644

def f (x : ℝ) : ℝ := (1 / 3) * x^3 + 3 * x * f' 0

theorem find_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end find_f_prime_at_1_l699_699644


namespace rooks_knight_move_attack_l699_699845

theorem rooks_knight_move_attack : 
  ∀ (board : array (fin 15) (fin 15) bool),
  (∀ i j, (i ≠ j → (∀ (kij lmn : fin 15), board.get! i k ≠ board.get! j l ∧ board.get! k i ≠ board.get! l j))) →
  (∀ (rook_pos : fin 15 → fin 15 × fin 15),
  (∀ i, (rook_pos i ≠ rook_pos j → board.get! (rook_pos i).1 (rook_pos i).2 = tt)) →
  (∀ (k_moves : fin 15 → fin 15 × fin 15),
  (∀ i, k_moves i = 
    ((rook_pos i).1 + 2, (rook_pos i).2 + 1) 
    ∨ ((rook_pos i).1 + 2, (rook_pos i).2 - 1) 
    ∨ ((rook_pos i).1 - 2, (rook_pos i).2 + 1) 
    ∨ ((rook_pos i).1 - 2, (rook_pos i).2 - 1) 
    ∨ ((rook_pos i).1 + 1, (rook_pos i).2 + 2) 
    ∨ ((rook_pos i).1 + 1, (rook_pos i).2 - 2) 
    ∨ ((rook_pos i).1 - 1, (rook_pos i).2 + 2) 
    ∨ ((rook_pos i).1 - 1, (rook_pos i).2 - 2)) →
  ∃ i j, i ≠ j ∧ (k_moves i).1 = (k_moves j).1 ∨ (k_moves i).2 = (k_moves j).2) :=
  sorry

end rooks_knight_move_attack_l699_699845


namespace number_of_triangles_l699_699115

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l699_699115


namespace Bruce_paid_correct_amount_l699_699218

def grape_kg := 9
def grape_price_per_kg := 70
def mango_kg := 7
def mango_price_per_kg := 55
def orange_kg := 5
def orange_price_per_kg := 45
def apple_kg := 3
def apple_price_per_kg := 80

def total_cost := grape_kg * grape_price_per_kg + 
                  mango_kg * mango_price_per_kg + 
                  orange_kg * orange_price_per_kg + 
                  apple_kg * apple_price_per_kg

theorem Bruce_paid_correct_amount : total_cost = 1480 := by
  sorry

end Bruce_paid_correct_amount_l699_699218


namespace final_price_scarf_l699_699199

variables (P : ℝ) (D1 D2 : ℝ) (P₁ P₂ : ℝ)

theorem final_price_scarf :
  P = 15 → D1 = 0.20 → D2 = 0.25 → P₁ = P * (1 - D1) → P₂ = P₁ * (1 - D2) → P₂ = 9 :=
by
  intros hP hD1 hD2 hP₁ hP₂
  rw [hP, hD1, hD2] at hP₁ hP₂
  have hP₁' : P₁ = 15 * (1 - 0.20), from hP₁
  rw [hP₁'] at hP₂
  have hP₁'' : P₁ = 12, by norm_num
  rw [hP₁''] at hP₂
  have hP₂' : P₂ = 12 * (1 - 0.25), from hP₂
  rw [hP₂'] 
  norm_num
  sorry

end final_price_scarf_l699_699199


namespace find_area_between_circles_l699_699141

noncomputable def area_between_two_circles
  (O F G : Type) [metric_space O] [metric_space F]
  (radius_outer : ℝ) (chord_length : ℝ) : ℝ :=
  let radius_inner := 2 * (sqrt 11) in
  let area_outer := real.pi * (radius_outer ^ 2) in
  let area_inner := real.pi * (radius_inner ^ 2) in
  area_outer - area_inner

theorem find_area_between_circles : 
  area_between_two_circles 
    ℝ ℝ ℝ 12 20 = 100 * real.pi :=
by sorry

end find_area_between_circles_l699_699141


namespace n_prime_or_power_of_2_l699_699948

theorem n_prime_or_power_of_2
  (n : ℤ)
  (a : ℕ → ℕ)
  (d : ℕ)
  (h_n_gt_6 : 6 < n)
  (h_all_nat_lt_n_rel_prime : ∀ i j, i < j → j < n → a (j-1) ∈ ℕ ∧ a i = j ∧ Nat.coprime a i n)
  (h_diff_eq_d : ∀ m, m < n - 1 → a (m + 1) - a m = d)
  (h_d_pos : 0 < d) :
  Prime n ∨ ∃ k : ℕ, n = 2^k := 
sorry

end n_prime_or_power_of_2_l699_699948


namespace monotonic_intervals_sum_ge_two_log_sum_inequality_l699_699680

-- Condition: Definition of f(x)
def f (a x : ℝ) : ℝ := a * real.log x + 0.5 * x^2 - (a + 1) * x + 1.5

-- Part (1): Monotonic intervals
theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) : true := sorry

-- Part (2): Prove x_1 + x_2 >= 2 when a = 1 and f(x_1) + f(x_2) = 0
theorem sum_ge_two {x1 x2 : ℝ} (h : f 1 x1 + f 1 x2 = 0) : x1 + x2 ≥ 2 := sorry

-- Part (3): Prove inequality for natural numbers
theorem log_sum_inequality (n : ℕ) (h : 0 < n) : 
  2 * real.log (n + 1) + (finset.range n).sum (λ i, ((i : ℝ) / (i + 1))^2) > n := sorry

end monotonic_intervals_sum_ge_two_log_sum_inequality_l699_699680


namespace integral_sqrt_a_squared_minus_x_squared_l699_699616

open Real

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) :
  (∫ x in -a..a, sqrt (a^2 - x^2)) = 1/2 * π * a^2 :=
by
  sorry

end integral_sqrt_a_squared_minus_x_squared_l699_699616


namespace polynomial_no_integer_roots_polynomial_no_integer_roots_if_odd_l699_699826

theorem polynomial_no_integer_roots
  (f : ℤ[X]) (k : ℕ) (h1 : k > 1)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → f.eval i % k ≠ 0) :
  ∀ a : ℤ, f.eval a % k ≠ 0 :=
by
  sorry

theorem polynomial_no_integer_roots_if_odd
  (f : ℤ[X])
  (h1 : ∃ k : ℕ, k > 1 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → f.eval i % k ≠ 0)
  (h2 : f.eval 1 % 2 ≠ 0)
  (h3 : f.eval 2 % 2 ≠ 0) :
  ¬ ∃ r : ℤ, f.eval r = 0 :=
by
  sorry

end polynomial_no_integer_roots_polynomial_no_integer_roots_if_odd_l699_699826


namespace amount_of_paint_in_jar_l699_699074

theorem amount_of_paint_in_jar :
  let mary_paint := 3
  let mike_paint := mary_paint + 2
  let sun_paint  := 5
  in mary_paint + mike_paint + sun_paint = 13 :=
by
  let mary_paint := 3
  let mike_paint := mary_paint + 2
  let sun_paint := 5
  show mary_paint + mike_paint + sun_paint = 13
  sorry

end amount_of_paint_in_jar_l699_699074


namespace reasoning_type_is_deductive_l699_699989

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_even_function (f : ℝ → ℝ) : Prop := 
even_function f

def f : ℝ → ℝ := λ x, x^2

theorem reasoning_type_is_deductive :
  even_function f → is_even_function f :=
by
  intro h
  exact h

end reasoning_type_is_deductive_l699_699989


namespace range_of_a_l699_699808

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699808


namespace sum_powers_divisible_by_odd_n_l699_699427

theorem sum_powers_divisible_by_odd_n (n : ℕ) (h_odd : n % 2 = 1) : 
  (∑ k in finset.range n, k^n) % n = 0 := 
sorry

end sum_powers_divisible_by_odd_n_l699_699427


namespace part1_part2_l699_699816

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs(2 * x - 1) + abs(x + a)

-- Part (1)
theorem part1 (x : ℝ) : f x 1 ≥ 3 → (x ≥ 1 ∨ x ≤ -1) :=
by sorry

-- Part (2)
theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ abs (a - 1)) → a ≤ 1/4 :=
by sorry

end part1_part2_l699_699816


namespace max_value_f_l699_699626

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x

theorem max_value_f : 
  ∃ x ∈ set.Icc 0 20, ∀ y ∈ set.Icc 0 20, f y ≤ f x ∧ f x = 8 * Real.sqrt 3 / 3 :=
by
  -- Sorry to skip the proof.
sorry

end max_value_f_l699_699626


namespace prove_monotonic_increasing_range_l699_699801

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699801


namespace train_crossing_pole_time_l699_699984

def train_speed_km_per_hr : ℝ := 60
def train_length_meters : ℝ := 700

-- We define the conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed in m/s given the speed in km/hr
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * conversion_factor

-- Calculate the time to cross the pole
def time_to_cross_pole : ℝ := train_length_meters / train_speed_m_per_s

theorem train_crossing_pole_time :
  time_to_cross_pole ≈ 42 :=
by
  sorry

end train_crossing_pole_time_l699_699984


namespace monotonicity_of_f_when_a_is_zero_range_of_a_given_f_ge_zero_l699_699682

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * Real.exp (x - 1) - (a + 2) * x + a

-- Part (I)
theorem monotonicity_of_f_when_a_is_zero (b : ℝ) :
  (∀ x > 1, f 0 b x < f 0 b (x + 1)) ↔
  (b ≤ 0) ∨ (b > 0 ∧ x < log (2 / b) + 1) ∨ (b >= 2) :=
sorry

-- Part (II)
theorem range_of_a_given_f_ge_zero (a : ℝ) :
  (∀ x ≥ 1, f a 2 x ≥ 0) → (a ∈ Icc (-∞ : ℝ) 2) :=
sorry

end monotonicity_of_f_when_a_is_zero_range_of_a_given_f_ge_zero_l699_699682


namespace joe_travel_time_l699_699389

theorem joe_travel_time
  (d : ℝ) -- Total distance
  (rw : ℝ) (rr : ℝ) -- Walking and running rates
  (tw : ℝ) -- Walking time
  (tr : ℝ) -- Running time
  (h1 : tw = 9)
  (h2 : rr = 4 * rw)
  (h3 : rw * tw = d / 3)
  (h4 : rr * tr = 2 * d / 3) :
  tw + tr = 13.5 :=
by 
  sorry

end joe_travel_time_l699_699389


namespace Darren_in_fourth_car_l699_699571

def Darren_sits_in_fourth_car : Prop :=
  ∀ (pos : ℕ → ℕ), 
    (pos 1 ≠ 7) ∧ 
    (pos 2 = pos 1 + 1) ∧ 
    (pos 3 = pos 5 - 1) ∧ 
    (pos 1 = pos 7 - 3) ∧ 
    ∃ (k a : ℕ), (2 ≤ |pos k - pos a|) →
    pos 3 = 4

theorem Darren_in_fourth_car : Darren_sits_in_fourth_car :=
by 
  intros pos,
  have h1 : pos 1 ≠ 7 := by sorry,
  have h2 : pos 2 = pos 1 + 1 := by sorry,
  have h3 : pos 3 = pos 5 - 1 := by sorry,
  have h4 : pos 1 = pos 7 - 3 := by sorry,
  have h5 : ∃ (k a : ℕ), (2 ≤ |pos k - pos a|) := by sorry,
  show pos 3 = 4, from sorry

end Darren_in_fourth_car_l699_699571


namespace simplify_expression_l699_699093

theorem simplify_expression (x : ℝ) :
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) = x^4 - 1 :=
  by 
    sorry

end simplify_expression_l699_699093


namespace probability_within_margin_l699_699489

noncomputable theory

def sample_size : ℕ := 626
def sample_mean : ℝ := 16.8
def sample_variance : ℝ := 4
def standard_deviation : ℝ := real.sqrt sample_variance
def standard_error_of_mean : ℝ := standard_deviation / real.sqrt (sample_size - 1)
def margin : ℝ := 0.2

theorem probability_within_margin :
  (probability_within_margin (sample_mean : ℝ) (margin / standard_error_of_mean) 
  ≈ 0.98758) :=
sorry

end probability_within_margin_l699_699489


namespace min_distance_of_complex_abs_eq_one_l699_699645

noncomputable def min_distance (z : ℂ) (h : complex.abs z = 1) : ℝ :=
  complex.abs (z - (3 + 4 * complex.I))

theorem min_distance_of_complex_abs_eq_one (z : ℂ) (h : complex.abs z = 1) : min_distance z h = 4 :=
sorry

end min_distance_of_complex_abs_eq_one_l699_699645


namespace number_of_intersection_points_line_circle_l699_699735
-- Import the necessary library

-- Define the problem
theorem number_of_intersection_points_line_circle
  -- Polar coordinates definitions
  (line_eq : ∀ (ρ θ : ℝ), 4 * ρ * real.cos (θ - real.pi / 6) + 1 = 0)
  (circle_eq : ∀ (ρ θ : ℝ), ρ = 2 * real.sin θ) :
  -- Prove number of common points is 2
  ∃ (n : ℕ), n = 2 := 
sorry

end number_of_intersection_points_line_circle_l699_699735


namespace coeff_x3_in_p_cubed_l699_699411

def p (x : ℝ) : ℝ := x^4 + x^3 - 3 * x + 2 

theorem coeff_x3_in_p_cubed :
  polynomial.coeff (polynomial.expand ℝ 3 (polynomial.C (p 1))) 3 = -27 :=
sorry

end coeff_x3_in_p_cubed_l699_699411


namespace least_positive_integer_satisfying_conditions_l699_699505

theorem least_positive_integer_satisfying_conditions :
  ∃ b : ℕ, b > 0 ∧ (b % 7 = 6) ∧ (b % 11 = 10) ∧ (b % 13 = 12) ∧ b = 1000 :=
by
  sorry

end least_positive_integer_satisfying_conditions_l699_699505


namespace steve_marbles_after_trans_l699_699089

def initial_marbles (S T L H : ℕ) : Prop :=
  S = 2 * T ∧
  L = S - 5 ∧
  H = T + 3

def transactions (S T L H : ℕ) (new_S new_T new_L new_H : ℕ) : Prop :=
  new_S = S - 10 ∧
  new_L = L - 4 ∧
  new_T = T + 4 ∧
  new_H = H - 6

theorem steve_marbles_after_trans (S T L H new_S new_T new_L new_H : ℕ) :
  initial_marbles S T L H →
  transactions S T L H new_S new_T new_L new_H →
  new_S = 6 →
  new_T = 12 :=
by
  sorry

end steve_marbles_after_trans_l699_699089


namespace prove_monotonic_increasing_range_l699_699800

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699800


namespace total_canoes_built_l699_699215

theorem total_canoes_built (boats_jan : ℕ) (h : boats_jan = 5)
    (boats_feb : ℕ) (h1 : boats_feb = boats_jan * 3)
    (boats_mar : ℕ) (h2 : boats_mar = boats_feb * 3)
    (boats_apr : ℕ) (h3 : boats_apr = boats_mar * 3) :
  boats_jan + boats_feb + boats_mar + boats_apr = 200 :=
sorry

end total_canoes_built_l699_699215


namespace relationship_of_y_l699_699378

theorem relationship_of_y {k y1 y2 y3 : ℝ} (hk : k > 0) :
  (y1 = k / -1) → (y2 = k / 2) → (y3 = k / 3) → y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_l699_699378


namespace monotonically_increasing_range_l699_699769

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l699_699769


namespace angle_between_a_b_is_45_degrees_l699_699302

-- Given conditions
def e1 : Vector ℝ 2 := Vector.ofList [1, 0]
def e2 : Vector ℝ 2 := Vector.ofList [0, 1]

-- Define a and b based on the problem's conditions
def a : Vector ℝ 2 := 3 • e1 - e2
def b : Vector ℝ 2 := 2 • e1 + e2

-- Prove that the angle between a and b is 45 degrees
theorem angle_between_a_b_is_45_degrees : 
  ∃ θ : ℝ, θ = Real.arccos ((a.dot b) / (∥a∥ * ∥b∥)) ∧ θ = π / 4 :=
by
  sorry

end angle_between_a_b_is_45_degrees_l699_699302


namespace AO_AQ_AR_sum_l699_699954

-- Define the properties of a regular pentagon
structure Pentagon :=
  (A B C D E : Point)
  (O : Point)
  (is_regular : regular_pentagon A B C D E)

-- Define the perpendicular dropped from a point onto a line
def Perpendicular (P : Point) (l1 l2 : Line) : Prop :=
  Perpendicular_proj P l1.origin l2

-- Define the main theorem stating the relationship
theorem AO_AQ_AR_sum (pent : Pentagon) (AP AQ AR : Length)
  (OP_eq_one : pent.O.dist pent.P = 1)
  (AP_condition : Perpendicular pent.A pent.CD)
  (AQ_condition : Perpendicular pent.A pent.CB.extended)
  (AR_condition : Perpendicular pent.A pent.DE.extended) :
  (pent.A.dist pent.O) + AQ + AR = 4 :=
sorry

end AO_AQ_AR_sum_l699_699954


namespace calculate_expression_l699_699228

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l699_699228


namespace Thabo_harcdover_nonfiction_books_l699_699121

theorem Thabo_harcdover_nonfiction_books 
  (H P F : ℕ)
  (h1 : P = H + 20)
  (h2 : F = 2 * P)
  (h3 : H + P + F = 180) : 
  H = 30 :=
by
  sorry

end Thabo_harcdover_nonfiction_books_l699_699121


namespace combined_weight_of_candles_l699_699248

theorem combined_weight_of_candles (candles : ℕ) (weight_per_candle : ℕ) (total_weight : ℕ) :
  candles = 10 - 3 →
  weight_per_candle = 8 + 1 →
  total_weight = candles * weight_per_candle →
  total_weight = 63 :=
by
  intros
  subst_vars
  sorry

end combined_weight_of_candles_l699_699248


namespace problem1a_problem1b_problem2_problem3_l699_699357

def is_diff_1_eqn (a b c : ℝ) : Prop :=
  let Δ := b*b - 4*a*c in
  Δ ≥ 0 ∧ (sqrt Δ / a = 1 ∨ sqrt Δ / a = -1)

theorem problem1a : ¬ is_diff_1_eqn 1 (-5) (-6) := by sorry

theorem problem1b : is_diff_1_eqn 1 (-sqrt 5) 1 := by sorry

theorem problem2 (m : ℝ) (h : is_diff_1_eqn 1 (-(m-1)) (-m)) : m = 0 ∨ m = -2 := by sorry

theorem problem3 (a b : ℝ) (h₀ : a > 0) (h : is_diff_1_eqn a b 1) :
  let t := 10 * a - b * b in t = 9 := by sorry

end problem1a_problem1b_problem2_problem3_l699_699357


namespace janice_bottle_caps_l699_699388

-- Define the conditions
def num_boxes : ℕ := 79
def caps_per_box : ℕ := 4

-- Define the question as a theorem to prove
theorem janice_bottle_caps : num_boxes * caps_per_box = 316 :=
by
  sorry

end janice_bottle_caps_l699_699388


namespace unique_solution_m_l699_699271

theorem unique_solution_m :
  ∃! m : ℝ, ∀ x y : ℝ, (y = x^2 ∧ y = 4*x + m) → m = -4 :=
by 
  sorry

end unique_solution_m_l699_699271


namespace matrix_solution_l699_699044

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ [1, 2, 3],
      [2, 1, 2],
      [3, 2, 1] ]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  0

theorem matrix_solution (p q r : ℝ) (h : p = -2 ∧ q = -26 ∧ r = -91) :
   B^3 + p • B^2 + q • B + r • I = zero_matrix :=
by
  sorry

end matrix_solution_l699_699044


namespace no_chord_arrangement_l699_699501

/--
Given any group of n people, it is not always possible to arrange chords on a circle
such that intersecting chords represent acquaintances and non-intersecting chords represent strangers.
Note that touching endpoints of chords is considered an intersection.
-/
theorem no_chord_arrangement (n : ℕ) : ¬ (∃ (chords : fin n → set (ℝ × ℝ)), 
  (∀ i j, i ≠ j → (
    (∃ (a b : ℝ), (a, b) ∈ chords i ∧ (a, b) ∈ chords j) ↔ acquaintances i j))) := 
sorry

end no_chord_arrangement_l699_699501


namespace find_bottom_width_of_canal_l699_699455

noncomputable def bottom_width (area depth top_width : ℝ) : ℝ :=
  let w := (2 * area - top_width * depth) / depth in
  w

theorem find_bottom_width_of_canal :
  bottom_width 10290 257.25 6 = 74.02 :=
by
  -- We state that w = (2 * 10290 - 6 * 257.25) / 257.25
  -- and we need to prove that it equals to 74.02
  sorry

end find_bottom_width_of_canal_l699_699455


namespace volume_ratio_of_cones_l699_699931

def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem volume_ratio_of_cones (rC hC rD hD : ℝ) 
    (hrC : rC = 16.4) (hhC : hC = 32.8) 
    (hrD : rD = 32.8) (hhD : hD = 16.4) :
    (cone_volume rC hC) / (cone_volume rD hD) = 1 / 2 := by
  sorry

end volume_ratio_of_cones_l699_699931


namespace diagonal_ratio_of_squares_l699_699473

theorem diagonal_ratio_of_squares (P d : ℝ) (h : ∃ s S, 4 * S = 4 * s * 4 ∧ P = 4 * s ∧ d = s * Real.sqrt 2) : 
    (∃ D, D = 4 * d) :=
by
  sorry

end diagonal_ratio_of_squares_l699_699473


namespace function_equality_check_l699_699168

-- Define the candidate functions
def fA (x : ℝ) : ℝ := (Real.sqrt x)^2
def fB (x : ℝ) : ℝ := x^2 / x
def fC (x : ℝ) : ℝ := Real.sqrt (x^2)
def fD (x : ℝ) : ℝ := Real.cbrt (x^3)

-- Define the target function
def f (x : ℝ) : ℝ := x

-- Prove that fD is equal to f and the others are not
theorem function_equality_check :
  (∀ x, fA x ≠ f x) ∧
  (∀ x, fB x ≠ f x) ∧
  (∀ x, fC x ≠ f x) ∧
  (∀ x, fD x = f x) := 
by
  sorry

end function_equality_check_l699_699168


namespace determine_triangle_type_l699_699383

-- Definition of the problem: in triangle ABC, with sides a, b, c opposite angles A, B, and C respectively

variables {A B C a b c : ℝ}

-- Given condition in the problem: a / cos B = b / cos A
def cond1 : Prop := a / Real.cos B = b / Real.cos A

-- Define what it means for a triangle to be either isosceles or right
def is_isosceles_or_right_triangle : Prop :=
  (A = B ∨ B = C ∨ C = A) ∨ 90 = C ∨ 90 = A ∨ 90 = B

-- The desired theorem statement
theorem determine_triangle_type (h : cond1) : is_isosceles_or_right_triangle :=
sorry

end determine_triangle_type_l699_699383


namespace find_X_l699_699382

namespace Geometry

-- Define the coordinates of the midpoints D, E, and F
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def D : Point3D := ⟨2, 6, -2⟩
def E : Point3D := ⟨1, 5, -3⟩
def F : Point3D := ⟨3, 4, 5⟩

-- Theorem statement to prove the coordinates of X
theorem find_X (X : Point3D) : 
  let midpoint := λ (P Q : Point3D), ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩ in
  midpoint X D = ⟨2, 4.5, 1⟩ →
  X = ⟨2, 3, 4⟩ := 
  by
    sorry

end Geometry

end find_X_l699_699382


namespace problem_Ⅰ_problem_Ⅱ_l699_699670

-- Variables denoting points and curve
variable (x y : ℝ)
def parabola (x y : ℝ) := y^2 = 4 * x
def point_O := (0 : ℝ, 0 : ℝ)
def midpoint_P := (3, 2)
def passes_through (l : ℝ) (x₀ y₀ : ℝ) := y₀ = l * x₀ - 2
def area_triangle_Omn := 6

-- Problem Ⅰ
def point_A (x₁ y₁ : ℝ) := parabola x₁ y₁
def point_B (x₂ y₂ : ℝ) := parabola x₂ y₂
def midpoint (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = 6 ∧ y₁ + y₂ = 4
def line_AB_eq := ∀ (x₁ y₁ x₂ y₂ : ℝ), (midpoint x₁ y₁ x₂ y₂) ∧ (point_A x₁ y₁) ∧ (point_B x₂ y₂) → (∀ x y, (y = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) + y₁) ↔ x - y - 1 = 0)

-- Problem Ⅱ
def line_l_eq (m : ℤ) := (m = 1/2 ∨ m = -1/2) → (∀ x y, (passes_through m x y) → (2 * x - y - 4 = 0 ∨ 2 * x + y - 4 = 0))

-- Theorems to prove
theorem problem_Ⅰ : ∀ (x₁ y₁ x₂ y₂ : ℝ), (midpoint x₁ y₁ x₂ y₂) ∧ (point_A x₁ y₁) ∧ (point_B x₂ y₂) → line_AB_eq x₁ y₁ x₂ y₂ := by sorry

theorem problem_Ⅱ : ∀ (m : ℤ), (passes_through m 2 0) ∧ (area_triangle_Omn = 6) → line_l_eq m := by sorry

end problem_Ⅰ_problem_Ⅱ_l699_699670


namespace interval_sum_l699_699900

theorem interval_sum (a b : ℝ) (h : ∀ x,  |3 * x - 80| ≤ |2 * x - 105| ↔ (a ≤ x ∧ x ≤ b)) :
  a + b = 12 :=
sorry

end interval_sum_l699_699900


namespace range_of_a_l699_699809

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l699_699809


namespace laura_bought_4_shirts_l699_699400

-- Definitions for the conditions
def pants_price : ℕ := 54
def num_pants : ℕ := 2
def shirt_price : ℕ := 33
def given_money : ℕ := 250
def change_received : ℕ := 10

-- Proving the number of shirts bought is 4
theorem laura_bought_4_shirts :
  (num_pants * pants_price) + (shirt_price * 4) + change_received = given_money :=
by
  sorry

end laura_bought_4_shirts_l699_699400


namespace divide_line_segment_l699_699374

theorem divide_line_segment (z1 z2 : ℂ) (m n : ℚ) (h1 : z1 = -7 + 5 * complex.I) (h2 : z2 = 5 - 3 * complex.I) (hm : m = 2) (hn : n = 1) :
  (m * z2 + n * z1) / (m + n) = 1 - (1 / 3) * complex.I :=
by {
  rw [h1, h2, hm, hn],
  simp,
  norm_num,
  sorry
}

end divide_line_segment_l699_699374


namespace power_eq_fractions_l699_699002

theorem power_eq_fractions (x : ℝ) (h : 81^4 = 27^x) : 3^(-x) = 1 / 3^(16/3) :=
by {
  sorry
}

end power_eq_fractions_l699_699002


namespace club_membership_l699_699468

theorem club_membership (n : ℕ) : 
  n ≡ 6 [MOD 10] → n ≡ 6 [MOD 11] → 200 ≤ n ∧ n ≤ 300 → n = 226 :=
by
  intros h1 h2 h3
  sorry

end club_membership_l699_699468


namespace negation_proposition_l699_699008

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ∈ set.Ioo (-(π / 2)) (π / 2) → tan x > sin x) ↔
  (∃ x : ℝ, x ∈ set.Ioo (-(π / 2)) (π / 2) ∧ tan x ≤ sin x) :=
by
  sorry

end negation_proposition_l699_699008


namespace white_squares_in_20th_row_l699_699364

def num_squares_in_row (n : ℕ) : ℕ :=
  3 * n

def num_white_squares (n : ℕ) : ℕ :=
  (num_squares_in_row n - 2) / 2

theorem white_squares_in_20th_row: num_white_squares 20 = 30 := by
  -- Proof skipped
  sorry

end white_squares_in_20th_row_l699_699364


namespace min_a_plus_b_l699_699662

theorem min_a_plus_b (a b : ℕ) (h1 : ∃ l : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + l)
  (h2 : b^2 - 4 * a * l > 0)
  (h3 : -1 < -b / (2 * a) ∧ -b / (2 * a) < 0) :
  (∃ a b : ℕ, a + b = 10) :=
sorry

end min_a_plus_b_l699_699662


namespace product_of_integers_l699_699637

theorem product_of_integers
  (A B C D : ℕ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (hD : D > 0)
  (h_sum : A + B + C + D = 72)
  (h_eq : A + 3 = B - 3 ∧ B - 3 = C * 3 ∧ C * 3 = D / 2) :
  A * B * C * D = 68040 := 
by
  sorry

end product_of_integers_l699_699637


namespace sum_of_smallest_positive_solutions_l699_699864

def floor (x : ℝ) : ℤ :=
  ⌊x⌋

noncomputable def sum_smallest_solutions : ℝ :=
  let solutions := [3, 11/3, 9/2]
  solutions.sum

theorem sum_of_smallest_positive_solutions :
  ((∀ x : ℝ, (x - floor x = 2 / floor x) → x ∈ {3, 11 / 3, 9 / 2}) →
  sum_smallest_solutions =  67 / 6) :=
sorry

end sum_of_smallest_positive_solutions_l699_699864


namespace f_x_minus_3_odd_l699_699710

noncomputable def A : ℝ := sorry  -- A > 0
axiom (A_pos : A > 0)

noncomputable def f : ℝ → ℝ := λ x, A * Real.sin ((Real.pi / 2) * x + φ)

axiom (f_at_1_zero : f 1 = 0)

theorem f_x_minus_3_odd (x : ℝ) : f (x - 3) = -f (-(x - 3)) :=
by
  sorry

end f_x_minus_3_odd_l699_699710


namespace prove_monotonic_increasing_range_l699_699799

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699799


namespace line_tangent_circle_has_specific_m_l699_699351

theorem line_tangent_circle_has_specific_m (m : ℝ) :
  (∃ m, let C := (0, 1) in 
         let r := 1 in 
         x^2 + y^2 - 2 * y = 0 ∧ 
         sqrt 3 * x - y + m = 0 ∧ 
         abs (-1 + m) / sqrt (1 + 3) = 1) → 
    m = -1 ∨ m = 3 :=
by sorry

end line_tangent_circle_has_specific_m_l699_699351


namespace largest_sum_of_two_3_digit_numbers_l699_699158

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ),
    {a, b, c, d, e, f} = {1, 2, 3, 7, 8, 9} ∧
    100 * (max a d) + 10 * (max b e) + max c f = 1803 := 
sorry

end largest_sum_of_two_3_digit_numbers_l699_699158


namespace part1_part2_l699_699413

noncomputable def f (x : ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  log ((∑ k in Finset.range (n + 1), (k:ℝ)^x + a) / n)

theorem part1 (a : ℝ) (n : ℕ) (x : ℝ)
  (h1 : 2 ≤ n) (h2 : x ≤ 1) : 
  (a > -(n-1)) → ∃ y, y ∈ set.Icc (-∞) 1 ∧ f y a n := sorry

theorem part2 (a : ℝ) (n : ℕ) (x : ℝ)
  (h1 : 0 < a ∧ a ≤ 1) (h2 : 2 ≤ n) (h3 : x ≠ 0) : 
  2 * f x a n < f (2 * x) a n := sorry

end part1_part2_l699_699413


namespace solution_set_inequality_l699_699667

open Set

variable {a b : ℝ}

/-- Proof Problem Statement -/
theorem solution_set_inequality (h : ∀ x : ℝ, -3 < x ∧ x < -1 ↔ a * x^2 - 1999 * x + b > 0) : 
  ∀ x : ℝ, 1 < x ∧ x < 3 ↔ a * x^2 + 1999 * x + b > 0 :=
sorry

end solution_set_inequality_l699_699667


namespace cube_root_of_64_eq_two_pow_m_l699_699176

theorem cube_root_of_64_eq_two_pow_m (m : ℕ) (h : (64 : ℝ) ^ (1 / 3) = (2 : ℝ) ^ m) : m = 2 := 
sorry

end cube_root_of_64_eq_two_pow_m_l699_699176


namespace cell_phone_customers_us_l699_699534

theorem cell_phone_customers_us (total_customers : ℕ) (other_country_customers : ℕ) :
  total_customers = 7422 → other_country_customers = 6699 → (total_customers - other_country_customers) = 723 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end cell_phone_customers_us_l699_699534


namespace min_value_y_l699_699507

theorem min_value_y (x : ℝ) : ∃ x : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ z : ℝ, (y = z^2 + 16 * z + 20) → y ≥ -44 := 
sorry

end min_value_y_l699_699507


namespace expected_value_of_winnings_l699_699185

noncomputable def expected_value : ℝ :=
  (1 / 8) * (1 / 2) + (1 / 8) * (3 / 2) + (1 / 8) * (5 / 2) + (1 / 8) * (7 / 2) +
  (1 / 8) * 2 + (1 / 8) * 4 + (1 / 8) * 6 + (1 / 8) * 8

theorem expected_value_of_winnings : expected_value = 3.5 :=
by
  -- the proof steps will go here
  sorry

end expected_value_of_winnings_l699_699185


namespace apples_sold_calculation_l699_699482

def initial_apples := 1238
def pear_difference := 374
def pears_bought := 276
def total_after_transactions := 2527

theorem apples_sold_calculation : 
  let initial_pears := initial_apples + pear_difference,
      new_total_pears := initial_pears + pears_bought,
      apples_sold := 1238 - (total_after_transactions - new_total_pears)
  in apples_sold = 599 := 
by 
  let initial_pears := initial_apples + pear_difference;
  let new_total_pears := initial_pears + pears_bought;
  let apples_sold := initial_apples - (total_after_transactions - new_total_pears);
  have h : apples_sold = 1238 - (2527 - (1612 + 276)) := rfl;
  calc
    apples_sold = 1238 - (2527 - 1888) : by rw h
            ... = 1238 - 639 : by rfl
            ... = 599 : by rfl

end apples_sold_calculation_l699_699482


namespace number_of_transformations_returning_to_original_l699_699051

-- Definitions for the vertices of triangle T
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (4, 0)
def vertex3 : ℝ × ℝ := (0, 3)

-- Transformation functions
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Main statement
theorem number_of_transformations_returning_to_original :
  let transformations := [rotate90, rotate180, rotate270, reflect_x, reflect_y]
  in let sequences_with_3 := (transformations.product transformations).product transformations
  in let valid_sequences := sequences_with_3.filter (λ seq,
    let t1 := seq.1.1
    let t2 := seq.1.2
    let t3 := seq.2
    (t3 (t2 (t1 vertex1)) = vertex1) ∧ (t3 (t2 (t1 vertex2)) = vertex2) ∧ (t3 (t2 (t1 vertex3)) = vertex3))
  in valid_sequences.length = 12 := 
sorry

end number_of_transformations_returning_to_original_l699_699051


namespace rectangle_area_is_correct_l699_699129

   noncomputable def side_of_square : ℝ := real.sqrt 625
   noncomputable def radius_of_circle : ℝ := side_of_square
   def breadth_of_rectangle : ℝ := 10
   def length_of_rectangle : ℝ := (2/5) * radius_of_circle
   def angle_between_diagonal_and_breadth : ℝ := 30

   theorem rectangle_area_is_correct :
     let A := length_of_rectangle * breadth_of_rectangle in
     A = 100 := 
   by
     -- proof steps to be filled here
     sorry
   
end rectangle_area_is_correct_l699_699129


namespace petya_sequences_l699_699424

theorem petya_sequences (n : ℕ) (h : n = 100) : 
  let total_sequences := 3 ^ (n - 1),
      without_three := 2 ^ n 
  in total_sequences - without_three = 3 ^ 100 - 2 ^ 100 :=
by
  sorry

end petya_sequences_l699_699424


namespace quadratic_equation_completes_to_square_l699_699096

theorem quadratic_equation_completes_to_square :
  ∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x + 2)^2 = 2 :=
by
  intro x
  intro h
  sorry

end quadratic_equation_completes_to_square_l699_699096


namespace vector_solution_l699_699014

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_solution (a x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by sorry

end vector_solution_l699_699014


namespace right_triangle_area_l699_699919

theorem right_triangle_area (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : a = 40) (hc : c = 41) : 
  1 / 2 * a * √(c^2 - a^2) = 180 :=
by
  sorry

end right_triangle_area_l699_699919


namespace prove_monotonic_increasing_range_l699_699803

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l699_699803


namespace difference_of_areas_l699_699034

def right_angle (A B C : Type) := ∠BAC = 90
def length (A B : Type) (l : ℕ) := A != B ∧ dist A B = l

-- For simplicity, we define A, B, C, D, E, F to be points of some type
constant Point : Type
constant A B C D E F : Point

-- Declare the appropriate conditions for our problem as hypotheses
axiom h1 : right_angle A B C
axiom h2 : length A B 5
axiom h3 : length A C 3
axiom h4 : length B E 10
axiom h5 : intersects B D C E = F
axiom h6 : perpendicular BD AC

-- Area definition for triangles
def area (P Q R : Point) : ℝ := sorry   -- Assume we have a function to compute area of a triangle

-- Define u, v, and w areas
def u : ℝ := area A F B
def v : ℝ := area C F D
def w : ℝ := area A B C

-- Define the mathematically equivalent statement we need to prove
theorem difference_of_areas :
  u - v = 7.5 := by 
    sorry

end difference_of_areas_l699_699034


namespace minimal_tetrahedron_volume_l699_699059

-- Let us define the problem according to the conditions given in the math problem
variables {a b c : ℝ} (h_ellipsoid : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

-- Define the ellipsoid equation
def ellipsoid (x y z : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1

-- Define the tangent plane
def tangent_plane (x0 y0 z0 x y z : ℝ) : Prop := 
  (x0 * x / a^2) + (y0 * y / b^2) + (z0 * z / c^2) = 1

-- Define the volume of the tetrahedron under the provided conditions
noncomputable def tetrahedron_volume {x0 y0 z0 : ℝ} (hx0 : x0 ≠ 0) (hy0 : y0 ≠ 0) (hz0 : z0 ≠ 0) : ℝ := 
  (a^2 / x0) * (b^2 / y0) * (c^2 / z0) / 6

-- Define maximizing product condition for the minimal volume
def maximizing_product (x0 y0 z0 : ℝ) : Prop :=
  x0 ≠ 0 ∧ y0 ≠ 0 ∧ z0 ≠ 0 ∧ ((x0^2 / a^2) = (y0^2 / b^2) ∧ (y0^2 / b^2) = (z0^2 / c^2) ∧ 
                                 (x0^2 / a^2) + (y0^2 / b^2) + (z0^2 / c^2) = 1)

-- The statement of the mathematical proof problem in Lean 4
theorem minimal_tetrahedron_volume
  {x0 y0 z0 : ℝ} (hx0 : x0 ≠ 0) (hy0 : y0 ≠ 0) (hz0 : z0 ≠ 0)
  (h_maximizing : maximizing_product x0 y0 z0)
  : tetrahedron_volume hx0 hy0 hz0 = (Real.sqrt 3 * a * b * c / 2) :=
sorry

end minimal_tetrahedron_volume_l699_699059


namespace triangle_construction_possible_l699_699686

-- Define a structure for representing a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a function to calculate the distance between two points
def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- Define a predicate to check if two line segments are perpendicular
def are_perpendicular (A B C D : Point) : Prop :=
  let slope1 := if A.x = B.x then 0 else (B.y - A.y) / (B.x - A.x)
  let slope2 := if C.x = D.x then 0 else (D.y - C.y) / (D.x - C.x) in
  slope1 * slope2 = -1

-- Define the main theorem to state the conditions for triangle construction
theorem triangle_construction_possible
  (A B C : Point)
  (AB_length : ℝ)
  (BC_length : ℝ)
  (medians_perpendicular : are_perpendicular (Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)) B (Point.mk ((B.x + C.x) / 2) ((B.y + C.y) / 2)) C) : 
  distance A B = AB_length ∧ distance B C = BC_length → ∃ ABC_triangle : Type, True :=
sorry

end triangle_construction_possible_l699_699686


namespace arrangements_I_arrangements_II_arrangements_III_l699_699178

noncomputable def num_of_arrangements_I := 24

theorem arrangements_I : 
  -- Ball 1 can only go into box 1, and ball 2 can only go into box 2
  num_of_arrangements_I = 24 :=
begin
  sorry
end

noncomputable def num_of_arrangements_II := 192

theorem arrangements_II : 
  -- Ball 3 can only be placed in box 1 or 2, and ball 4 cannot be placed in box 4
  num_of_arrangements_II = 192 :=
begin
  sorry
end

noncomputable def num_of_arrangements_III := 240

theorem arrangements_III : 
  -- Balls 5 and 6 can only be placed into two boxes with consecutive numbers
  num_of_arrangements_III = 240 :=
begin
  sorry
end

end arrangements_I_arrangements_II_arrangements_III_l699_699178


namespace intersection_points_l699_699123

theorem intersection_points : 
  (∃ x : ℝ, y = -2 * x + 4 ∧ y = 0 ∧ (x, y) = (2, 0)) ∧
  (∃ y : ℝ, y = -2 * 0 + 4 ∧ (0, y) = (0, 4)) :=
by
  sorry

end intersection_points_l699_699123


namespace arithmetic_sequence_ineq_l699_699272

variable {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)

-- Assume arithmetic sequence with common difference d > 0
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a n = a 0 + n * d

-- S_n is the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in finset.range n, a i

theorem arithmetic_sequence_ineq {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_seq a)
  (h_sum : sum_first_n_terms a S) (h_d : d > 0)
  (h_ineq : (S 8 - S 5) * (S 9 - S 5) < 0) :
  |a 7| < |a 8| :=
sorry

end arithmetic_sequence_ineq_l699_699272


namespace pyramid_lateral_edges_equal_l699_699541

theorem pyramid_lateral_edges_equal (n : ℕ)
  (A B : Type) [metric_space A] [add_group A] [module ℝ A] [metric_space B] [add_group B] [module ℝ B]
  (S : B) (A_i : fin n → B) 
  (O : B)
  (midpoints : fin n → B)
  (R : ℝ)
  (inscribed_circle : ∀ (i j : fin n), dist (A_i i) (A_i j) = R)
  (circumcircle_center_dist : ∀ (i : fin n), dist O (midpoints i) = R / 2)
  (eqidistance_condition : ∀ (i j : fin n), dist (midpoints i) (midpoints j) = dist (O) (midpoints j)) :
  ∀ (i j : fin n), dist (S - A_i i) (S - A_i j) = 0 :=
by
  sorry

end pyramid_lateral_edges_equal_l699_699541


namespace problem_1_problem_2_l699_699688

open Real

-- Step 1: Define the line and parabola conditions
def line_through_focus (k n : ℝ) : Prop := ∀ (x y : ℝ),
  y = k * (x - 1) ∧ (y = 0 → x = 1)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Step 2: Prove x_1 x_2 = 1 if line passes through the focus
theorem problem_1 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k 1)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1))
  (h_non_zero : x1 * x2 ≠ 0) :
  x1 * x2 = 1 :=
sorry

-- Step 3: Prove n = 4 if x_1 x_2 + y_1 y_2 = 0
theorem problem_2 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k n)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - n) ∧ y2 = k * (x2 - n))
  (h_product_relate : x1 * x2 + y1 * y2 = 0) :
  n = 4 :=
sorry

end problem_1_problem_2_l699_699688


namespace number_of_triangles_l699_699104

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l699_699104


namespace expand_and_simplify_l699_699252

theorem expand_and_simplify (x : ℝ) : 
  (2 * x + 6) * (x + 10) = 2 * x^2 + 26 * x + 60 :=
sorry

end expand_and_simplify_l699_699252


namespace monotonic_increasing_range_l699_699779

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l699_699779


namespace expected_value_range_l699_699499

variable (p q : ℝ) (ξ : ℕ → ℝ)

-- Define q as dependent on p
def q := 1 - p

-- Define u as 2pq
def u := 2 * p * q

-- Condition on p and q
axiom h_p : 0 < p ∧ p < 1
axiom h_q : 0 < q ∧ q < 1

-- Expected value of ξ statement
def E_ξ := (2 * (1 - u^10)) / (1 - u)

-- Theorem statement for the expected value range
theorem expected_value_range (p q : ℝ) (E_ξ : ℝ) (u : ℝ) (h_p : 0 < p ∧ p < 1)  :
  2 < E_ξ ∧ E_ξ ≤ 1023 / 256 :=
by
  sorry

end expected_value_range_l699_699499


namespace coefficient_x2_y3_binomial_expansion_l699_699377

theorem coefficient_x2_y3_binomial_expansion :
  ∀ (x y : ℝ), 
  ∃ c : ℝ, c = -10/3 ∧ (expand (3*x - 1/3*y)^5).coeff (x^2 * y^3) = c :=
by
  sorry

end coefficient_x2_y3_binomial_expansion_l699_699377


namespace expansion_of_binomials_l699_699588

theorem expansion_of_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 :=
  sorry

end expansion_of_binomials_l699_699588


namespace monotonic_increasing_range_l699_699793

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l699_699793


namespace angle_bisector_CD_ADF_l699_699376

-- Definitions based on the problem conditions
variables (O P A B C D E F : Type)

-- Point P is outside the circle centered at O
variable {circle : O → Type}

-- PA and PB are tangential to the circle at points A and B respectively
variables {tangent_PA : P → A → circle(O)}
variables {tangent_PB : P → B → circle(O)}

-- PCD is a secant line intersecting the circle at points C and D
variables {secant_PCD : P → C → D → circle(O)}

-- CO intersects the circle again at E
variable {co_intersect : C → E → circle(O)}

-- AC and EB intersect at F
variables {intersect_AC_EB : A → C → E → B → F}

-- Definition of the angle bisector property to be proven
theorem angle_bisector_CD_ADF
  (tangent_PA : P → A → circle(O))
  (tangent_PB : P → B → circle(O))
  (secant_PCD : P → C → D → circle(O))
  (co_intersect : C → E → circle(O))
  (intersect_AC_EB : A → C → E → B → F) :
  bisect_angle CD (ADF) :=
sorry

end angle_bisector_CD_ADF_l699_699376


namespace minimize_costs_l699_699184

noncomputable def k1 : ℝ := 200000
noncomputable def k2 : ℝ := 8000

def y1 (x : ℝ) : ℝ := k1 / x
def y2 (x : ℝ) : ℝ := k2 * x
def y_total (x : ℝ) : ℝ := y1 x + y2 x

theorem minimize_costs : (∀ x > 0, true) → ∃ (x : ℝ), x = 5 ∧ (∀ (x' : ℝ), x' > 0 → y_total x ≤ y_total x') :=
by
  sorry

end minimize_costs_l699_699184


namespace monotonicity_intervals_range_of_a_l699_699675

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x - a) / x

-- Question 1: monotonicity intervals for g(x) = f(x) / x on (1, +∞)
theorem monotonicity_intervals (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < Real.exp (a + 0.5)) → (deriv (λ x, f a x / x)) x > 0) ∧ 
  (∀ x : ℝ, (Real.exp (a + 0.5) < x) → (deriv (λ x, f a x / x)) x < 0) :=
sorry

-- Define the condition for the inequality
def condition (a : ℝ) (x : ℝ) : Prop := x^2 * f a x + a ≥ 2 - Real.exp 1

-- Question 2: find the range of a for the inequality to hold
theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  (∀ x : ℝ, x > 0 → condition a x) ↔ a ∈ Set.Icc 0 2 :=
sorry

end monotonicity_intervals_range_of_a_l699_699675
