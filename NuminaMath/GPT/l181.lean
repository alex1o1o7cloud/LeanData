import Mathlib

namespace num_terms_divisible_by_b_eq_gcd_l181_181678

theorem num_terms_divisible_by_b_eq_gcd (a b d : ℕ) (h_gcd : Nat.gcd a b = d) :
  (∃ count : ℕ, count = d ∧ ∀ k, (1 ≤ k ∧ k ≤ b) → (a * k) % b = 0 → k = (b / d) * i for some i : ℕ) :=
sorry

end num_terms_divisible_by_b_eq_gcd_l181_181678


namespace prove_ratio_l181_181047

variables (A B C D E : Type)
variables (a b c : ℝ) (h₁ h₂ : A)
variables (BD DE : ℝ)

-- Assuming h₁ and h₂ represent the given triangles ABC and BDE, respectively 
-- Isosceles triangle BDE with equal sides BD and DE, and sides of triangle ABC are a, b, c
def isosceles_triangle (A B C : Type) (BD DE : ℝ) : Prop :=
  BD = DE

-- Function representing the ratio of sides in the triangles
noncomputable def triangle_ratio (A B C D E : Type) (a b c : ℝ) (BD DE : ℝ) : ℝ :=
  let BE := sqrt (DE^2 + BD^2 - 2 * DE * BD * cos (acos ((a^2 + b^2 - c^2) / (2 * a * b))))
  in BE / BD

-- Proof statement that must be shown
theorem prove_ratio (h₁: (A B C : Type)) (h₂ : Ӏsosceles_triangle A B C BD DE) (a b c : ℝ) :
  triangle_ratio A B C D E a b c BD DE = abs (a^2 - b^2 + c^2) / (a * c) :=
sorry

end prove_ratio_l181_181047


namespace train_and_car_speed_l181_181467

theorem train_and_car_speed 
  (train_length : ℝ := 100)
  (time_to_cross_pole : ℝ := 20)
  (platform_length : ℝ := 200)
  (time_to_cross_platform : ℝ)
  (car_time_to_cross : ℝ) :
  (train_speed : ℝ := train_length / time_to_cross_pole)
  (new_train_speed : ℝ := train_speed * 1.25)
  (total_distance : ℝ := train_length + platform_length)
  (time_to_cross_platform = total_distance / new_train_speed)
  (car_speed := platform_length / time_to_cross_platform)
  (V1 = train_speed) ∧ (V1 = 5) ∧ (car_speed ≈ 4.17) :=
begin
  sorry
end

end train_and_car_speed_l181_181467


namespace circle_eqn_l181_181427

variable (a b r : ℝ)
variable (A : ℝ × ℝ := (3, 6))
variable (B : ℝ × ℝ := (5, 2))
variable (l : ℝ → ℝ → Prop := λ x y, 4 * x - 3 * y + 6 = 0)

theorem circle_eqn
  (h1 : l A.1 A.2)
  (h2 : (A.1 - a) ^ 2 + (A.2 - b) ^ 2 = (B.1 - a) ^ 2 + (B.2 - b) ^ 2)
  (h3 : (b - A.2) / (a - A.1) * 4 / 3 = -1) :
  (∀ x y, (x - 5) ^ 2 + (y - 9 / 2) ^ 2 = 25 / 4 → (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) :=
by sorry

end circle_eqn_l181_181427


namespace part1_part2_part3_l181_181937

-- Definitions for (1)
def S (n : ℕ) := (1/2 : ℝ) * n ^ 2 + (11/2 : ℝ) * n
def a (n : ℕ) := S n - S (n - 1)
def b (n : ℕ) := 3 * n + 2

-- Definitions for (2)
def c (n : ℕ) := 3 / ((2 * (a n) - 11) * (2 * (b n) - 1))
def T (n : ℕ) := (1/2 : ℝ) * (1 - 1 / (2 * n + 1))

-- Definitions for (3)
def f (n : ℕ) :=
  if n % 2 = 1 then a n else b n

theorem part1 : ∀ n, a n = n + 5 ∧ b n = 3 * n + 2 := sorry

theorem part2 : ∃ k : ℕ, (∀ n : ℕ, T n > k / 57) ∧ k = 18 := sorry

theorem part3 : ∃ m : ℕ, f (m + 15) = 5 * f m ∧ m = 11 := sorry

end part1_part2_part3_l181_181937


namespace find_angle_AMH_l181_181132

variables {A B C D H M : Type}
variables [Parallelogram A B C D] [Midpoint M A B]
variables (angle_B : ℝ) (equal_sides: ℝ) (angle_BHD : ℝ)
variables (midpoint_M: M ∈ [midpoint A B])
variables (point_H: H ∈ segment B C)

-- condition: angle of the parallelogram ABCD at B is 111 degrees
def angle_at_B : ℝ := angle_B = 111

-- condition: sides BC and BD are equal
def lengths_BC_BD : ℝ := equal_sides BC BD

-- condition: angle BHD is 90 degrees
def angle_B_H_D : ℝ := angle_BHD = 90

-- goal: find angle AMH in degrees
theorem find_angle_AMH :
  angle AMH = 132 :=
sorry

end find_angle_AMH_l181_181132


namespace trailing_zeros_of_expanded_5000_pow_50_l181_181766

theorem trailing_zeros_of_expanded_5000_pow_50 :
  (5000 : ℝ)^50 = 5^50 * 10^150 →
  ∃ n : ℕ, 5000^50 = (5^50 : ℝ) * 10^(n : ℝ) ∧ n = 150 :=
by
  intro h
  use 150
  rw h
  sorry

end trailing_zeros_of_expanded_5000_pow_50_l181_181766


namespace probability_sum_of_primes_is_odd_l181_181924

def first_ten_primes : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
def total_primes : Nat := 10

noncomputable def comb (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_sum_of_primes_is_odd : 
  let favorable : Nat := comb 9 3
  let total : Nat := comb 10 4
  (favorable.to_rat / total.to_rat) = (2 : ℚ) / 5 := 
by {
  let favorable := comb 9 3
  let total := comb 10 4
  have h : favorable = 84 := by sorry
  have h_total : total = 210 := by sorry
  have h_div : 84 * 5 = 210 * 2 := by norm_num
  rw [h, h_total],
  field_simp,
  exact h_div
}

end probability_sum_of_primes_is_odd_l181_181924


namespace integral_is_Gaussian_l181_181652

noncomputable def isGaussian (X : Ω → ℝ) : Prop :=
sorry

variable {Ω : Type*} {X : Ω → ℝ}

variables (measurable_X : ae_measurable X (measure_space.measure_space Ω)) 
          (finite_integral : ∫⁻ x, |X x| ∂(measure_space.measure_space Ω) < ∞)
          (is_gaussian_system : ∀ ⦃sD : set ℝ⦄ (hsD : is_open sD), measurable_set (λ ω, X ω ∈ sD))

theorem integral_is_Gaussian : isGaussian (λ ω, ∫ t in 0..1, X ω t) :=
sorry

end integral_is_Gaussian_l181_181652


namespace count_perfect_squares_l181_181201

theorem count_perfect_squares :
  {N : ℕ // N < 100}.count (λ N, ∃ k, k * k = N ∧ 36 ∣ k * k) = 8 := sorry

end count_perfect_squares_l181_181201


namespace units_digit_7_pow_5_l181_181765

theorem units_digit_7_pow_5 : (7 ^ 5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l181_181765


namespace acute_angle_tan_eq_one_l181_181946

theorem acute_angle_tan_eq_one (A : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 :=
by
  sorry

end acute_angle_tan_eq_one_l181_181946


namespace triangle_area_l181_181409

noncomputable def heron_formula (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area :
  heron_formula 30 28 14 = 194.98 :=
by
  sorry

end triangle_area_l181_181409


namespace max_trig_expr_l181_181533

theorem max_trig_expr (x y z : ℝ) :
  (\sin (2 * x) + \sin y + \sin (3 * z)) * (\cos (2 * x) + \cos y + \cos (3 * z)) ≤ 9 / 2 :=
sorry

end max_trig_expr_l181_181533


namespace hyperbola_conjugate_axis_length_three_times_transverse_l181_181958

theorem hyperbola_conjugate_axis_length_three_times_transverse
  (m : ℝ) 
  (h : (∀ (a b : ℝ), (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * b = 3 * 2 * a))) :
  m = 1 / 9 :=
begin
  sorry
end

end hyperbola_conjugate_axis_length_three_times_transverse_l181_181958


namespace at_least_one_genuine_certain_l181_181925

theorem at_least_one_genuine_certain
    (total_products : ℕ := 12)
    (genuine : ℕ := 10)
    (defective : ℕ := 2)
    (selected : ℕ := 3) :
    total_products = genuine + defective →
    genuine + defective = 12 →
    selected = 3 → 
    ∃ g d, g + d = selected ∧ g ≥ 1 ∧ g ≤ genuine ∧ d ≤ defective :=
by
  intro h_total h_sum h_sel
  use 1
  use 2
  split
  { rw h_sel
    exact rfl }
  split
  { exact nat.one_le_of_lt (lt_add_of_pos_right _ zero_lt_two) }
  split
  { exact nat.le_of_lt (lt_add_of_pos_right _ zero_lt_two) }
  { exact nat.le_refl _ }
  done

end at_least_one_genuine_certain_l181_181925


namespace sum_of_valid_k_equals_26_l181_181889

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181889


namespace option_C_is_not_proposition_l181_181011

namespace MathProof

def is_proposition (s : String) : Prop := 
  ∃ b : Bool, (s → b = tt) ∨ (s → b = ff)

def statement_A : String := "The shortest line segment between two points."
def statement_B : String := "Non-parallel lines have only one intersection point."
def statement_C : String := "Is the difference between x and y equal to x-y?"
def statement_D : String := "Equal angles are vertical angles."

theorem option_C_is_not_proposition : ¬ is_proposition statement_C := 
  sorry

end MathProof

end option_C_is_not_proposition_l181_181011


namespace exactly_one_matching_pair_l181_181493

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, _ + 1 => 0
| n + 1, m + 1 => binomial n m + binomial n (m + 1)

theorem exactly_one_matching_pair (pairs : ℕ) (choose_four : ℕ) : pairs = 4 → choose_four = 4 → 
  (pairs * binomial (pairs - 1) 2 * 4 = 48) := 
by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end exactly_one_matching_pair_l181_181493


namespace b_10_is_105_over_2_l181_181087

noncomputable def b (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n = 2 then 5
  else (b (n - 1))^2 / (b (n - 1) - b (n - 2))

theorem b_10_is_105_over_2 : b 10 = 105 / 2 :=
by sorry

end b_10_is_105_over_2_l181_181087


namespace largest_n_sine_cosine_inequality_l181_181881

theorem largest_n_sine_cosine_inequality :
  ∃ (n : ℕ), (∀ x : ℝ, (sin x)^n + (cos x)^n ≥ 1 / (2 * n)) ∧ (∀ m : ℕ, (∀ x : ℝ, (sin x)^m + (cos x)^m ≥ 1 / (2 * m)) → m ≤ n) :=
sorry

end largest_n_sine_cosine_inequality_l181_181881


namespace find_solution_to_inequality_l181_181525

open Set

noncomputable def inequality_solution : Set ℝ := {x : ℝ | 0.5 ≤ x ∧ x < 2 ∨ 3 ≤ x}

theorem find_solution_to_inequality :
  {x : ℝ | (x^2 + 1) / (x - 2) + (2 * x + 3) / (2 * x - 1) ≥ 4} = inequality_solution := 
sorry

end find_solution_to_inequality_l181_181525


namespace smallest_positive_period_of_f_monotonically_increasing_interval_of_f_min_max_of_f_on_interval_l181_181178

noncomputable def f (x : ℝ) : ℝ := real.sqrt 2 * real.sin (2 * x + real.pi / 4) + 1

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
sorry

theorem monotonically_increasing_interval_of_f (k : ℤ) :
  ∀ x₁ x₂ : ℝ, (k * real.pi - 3 * real.pi / 8) ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ (k * real.pi + real.pi / 8) → f x₁ ≤ f x₂ :=
sorry

theorem min_max_of_f_on_interval :
  ∃ (xmin xmax : ℝ), xmin = -real.pi / 4 ∧ xmax = real.pi / 4 ∧ 
                     ∀ x : ℝ, xmin ≤ x ∧ x ≤ xmax → 
                          (f x = 0 ∨ f x = real.sqrt 2 + 1) :=
sorry

end smallest_positive_period_of_f_monotonically_increasing_interval_of_f_min_max_of_f_on_interval_l181_181178


namespace no_positive_root_polynomial_l181_181029

theorem no_positive_root_polynomial (a : ℕ → ℕ) (n k M : ℕ) (ha_pos : ∀ i, 1 ≤ a i) (hk : ∑ i in finset.range n, (1 / (a i)) = k) (hM : ∏ i in finset.range n, a i = M) (hM_pos : M > 1) :
  ¬ ∃ x > 0, M * (x + 1)^k = (finset.range n).prod (λ i, x + (a i)) :=
sorry

end no_positive_root_polynomial_l181_181029


namespace binomial_sum_sum_of_binomial_solutions_l181_181902

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181902


namespace num_correct_statements_is_2_l181_181578

axiom statement_1 : Prop
axiom statement_2 : Prop
axiom statement_3 : Prop
axiom statement_4 : Prop

def correct_statements_count : ℕ :=
  (if statement_1 then 1 else 0) +
  (if statement_2 then 1 else 0) +
  (if statement_3 then 1 else 0) +
  (if statement_4 then 1 else 0)

theorem num_correct_statements_is_2 
  (h1 : statement_1 = true)
  (h2 : statement_2 = false)
  (h3 : statement_3 = true)
  (h4 : statement_4 = false) :
  correct_statements_count = 2 :=
by {
  sorry
}

end num_correct_statements_is_2_l181_181578


namespace combinatorial_identity_l181_181117

open BigOperators

theorem combinatorial_identity (n : ℕ) :
  ∑ k in Finset.range (n + 1), n.choose k * 2^k * ((n - k) / 2).choose (n - k) = (2 * n + 1).choose n :=
by
  sorry

end combinatorial_identity_l181_181117


namespace problem_statement_l181_181648

variables {A B C : Type} [MetricSpace A] [MetricSpace B] 

variables (R r_a : ℝ) -- circumradius and excircle radius
variables (O O_a : A) -- centers of the circumcircle and excircle

def is_circumradius (R : ℝ) (O : A) (triangle : Triangle A) : Prop :=
  -- definition of circumradius for the current implementation

def is_excircle_radius (r_a : ℝ) (O_a : A) (triangle : Triangle A) (side : Segment A) : Prop :=
  -- definition of excircle radius for the current implementation

def are_circle_centers (O O_a : A) (triangle : Triangle A) : ℝ :=
  dist O O_a

theorem problem_statement
  (triangle : Triangle A)
  (side : Segment A) -- segment BC
  (hR : is_circumradius R O triangle)
  (h_r_a : is_excircle_radius r_a O_a triangle side) :
  (are_circle_centers O O_a triangle)^2 = R^2 + 2 * R * r_a :=
sorry

end problem_statement_l181_181648


namespace total_items_l181_181040

-- Define variables and constants based on the given problem.
constants (x y : ℕ)

-- Define the conditions given in the problem statement.
axiom price_A : ℕ := 8
axiom price_B : ℕ := 9
axiom total_cost : ℕ := 172
axiom items_equal : 2 * x = x + y
axiom total_spent : price_A * x + price_B * y = total_cost

-- The proof goal is to show that the total number of items is 20.
theorem total_items (hx : 2 * x = x + y) (ht : price_A * x + price_B * y = total_cost) : x + y = 20 :=
by
  sorry

end total_items_l181_181040


namespace infinite_series_sum_l181_181650

noncomputable def t : ℝ :=
  Classical.choose (exists_unique_positive_real_solution (λ x : ℝ, x ^ 3 - (1/4) * x - 1 = 0))

theorem infinite_series_sum (t : ℝ) (ht : t ^ 3 - (1/4) * t - 1 = 0) : 
  (∑' n : ℕ, (n + 1) * t ^ (3 * n + 2)) = 16 :=
sorry

-- This auxiliary lemma states that there exists a unique positive real solution.
lemma exists_unique_positive_real_solution (P : ℝ → Prop) :
  ∃! (r : ℝ), r > 0 ∧ P r :=
sorry

end infinite_series_sum_l181_181650


namespace problem_proof_l181_181504

theorem problem_proof (θ : ℝ) :
  (\sin θ * 3 - \cos θ * 2 = 0) →
  (3 * \sin θ + 2 * \cos θ) / (3 * \sin θ - \cos θ) = 4 := 
by 
  sorry

end problem_proof_l181_181504


namespace value_of_fraction_l181_181951

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = (-4 - (-1)) / (4 - 1)

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
  b2 * b2 = (-4) * (-1) ∧ b2 < 0

theorem value_of_fraction (a1 a2 b2 : ℝ)
  (h1 : arithmetic_sequence a1 a2)
  (h2 : geometric_sequence b2) :
  (a2 - a1) / b2 = 1 / 2 :=
by
  sorry

end value_of_fraction_l181_181951


namespace find_c_n_l181_181120

-- Definitions based on conditions
def sequence (a_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ (∀ n, ∃ (c_n : ℕ), ∀ (p q : ℕ), (p = a_n n ∧ q = a_n (n+1)) → p + q = 3 * n ∧ p * q = c_n)

-- Proof problem statement
theorem find_c_n (a_n : ℕ → ℕ) (h_seq : sequence a_n) :
  (∀ n, a_n (2 * n - 1) * a_n (2 * n) = 9 * n ^ 2 - 9 * n + 2) ∧
  (∀ n, a_n (2 * n) * a_n (2 * n + 1) = 9 * n ^ 2 - 1) :=
sorry

end find_c_n_l181_181120


namespace not_representative_l181_181448

variable (A B : Prop)

axiom has_email : A
axiom uses_internet : B
axiom dependent : A → B
axiom sample_size : Nat := 2000

theorem not_representative (hA : has_email) (hB : uses_internet) (hD : dependent) (hS : sample_size = 2000) : ¬(∀ x, A x) :=
  sorry

end not_representative_l181_181448


namespace limit_of_a_seq_l181_181144

noncomputable def a_seq : ℕ → ℝ
| 0 := 1
| 1 := 2
| (n + 2) := 2 * a_seq n * a_seq (n + 1) / (a_seq n + a_seq (n + 1))

theorem limit_of_a_seq : tendsto a_seq at_top (𝓝 (3 / 2)) :=
sorry

end limit_of_a_seq_l181_181144


namespace find_first_discount_l181_181804

noncomputable def first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * 1.34 * (1 - d) * 0.85

theorem find_first_discount (P : ℝ) (D : ℝ) (h : first_discount P D = 1.0251 * P) :
  D ≈ 0.1001 :=
by
  sorry

end find_first_discount_l181_181804


namespace common_points_line_graph_l181_181736

variable {α : Type*} [DecidableEq α]

def number_of_common_points (f : ℝ → ℝ) : ℕ :=
  if h : ∃ y, f 1 = y then 1 else 0

theorem common_points_line_graph (f : ℝ → ℝ) : 
  number_of_common_points f = 0 ∨ number_of_common_points f = 1 :=
by
  unfold number_of_common_points
  split_ifs
  · left
    exact nat.zero
  · right
    exact nat.one
  · left
    exact nat.zero
  sorry -- exact nat.one would also work here and would result in essentially the same proof

end common_points_line_graph_l181_181736


namespace find_original_price_l181_181358

theorem find_original_price 
  (x: ℝ)
  (h : ((x * 1.20 + 5) * 0.80 - 5 = 120)) :
  x ≈ 126.04 := 
sorry

end find_original_price_l181_181358


namespace perpendiculars_intersect_at_single_point_l181_181133

noncomputable def orthocenter (Δ : Triangle ℝ) : Point ℝ := sorry -- Define orthocenter

noncomputable def quadrilateral (A B C D : Point ℝ) : Prop := sorry -- Define a quadrilateral

theorem perpendiculars_intersect_at_single_point
  (A B C D A1 B1 C1 : Point ℝ)
  (h_quad : quadrilateral A B C D)
  (h_A1 : A1 = orthocenter (Triangle.mk B C D))
  (h_B1 : B1 = orthocenter (Triangle.mk A C D))
  (h_C1 : C1 = orthocenter (Triangle.mk A B D)) :
  ∃ X : Point ℝ,
    is_perpendicular A (Line.mk B1 C1) X ∧
    is_perpendicular B (Line.mk C1 A1) X ∧
    is_perpendicular C (Line.mk A1 B1) X :=
begin
  sorry
end

end perpendiculars_intersect_at_single_point_l181_181133


namespace inequality_solution_l181_181868

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Ioo (-1/4) 0 ∪ Set.Ioo 3/2 2) ↔ 
  (1 ≤ (x - 2) * 4 ∧ x ≠ 2) ∧ (x > 0 ∨ x ≠ 0) := 
sorry

end inequality_solution_l181_181868


namespace problem_statement_l181_181566

def f (x : ℝ) := if x ≥ 0 then 2*x^2 - x + m + 1 else -2*x^2 - x

theorem problem_statement (m : ℝ) (x : ℝ) (h : x < 0) : 
  f x = -2*x^2 - x := 
by 
  sorry

end problem_statement_l181_181566


namespace range_of_f_l181_181594

variable {a b c : ℝ}
variable (a_pos : 0 < a)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_f :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → 
  f x ∈ set.Icc (min ((-b^2 / (4 * a)) + c) (min (a - b + c) (a + b + c))) (max (a - b + c) (a + b + c)) :=
by
  -- Proof omitted
  sorry

end range_of_f_l181_181594


namespace inequality_solution_l181_181869

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Ioo (-1/4) 0 ∪ Set.Ioo 3/2 2) ↔ 
  (1 ≤ (x - 2) * 4 ∧ x ≠ 2) ∧ (x > 0 ∨ x ≠ 0) := 
sorry

end inequality_solution_l181_181869


namespace option_d_correct_l181_181400

theorem option_d_correct (x : ℝ) : (-3 * x + 2) * (-3 * x - 2) = 9 * x^2 - 4 := 
  sorry

end option_d_correct_l181_181400


namespace find_N_l181_181214

theorem find_N (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) :
  (x + y) / 3 = 1.222222222222222 := 
by
  -- We state the conditions.
  -- Lean will check whether these assumptions are consistent 
  sorry

end find_N_l181_181214


namespace distance_between_parallel_lines_l181_181378

-- Definitions based on given conditions
structure Semicircle (r : ℝ) (center : ℝ × ℝ)

def line1_length : ℝ := 24
def line2_length : ℝ := 10

def distance_between_lines (d : ℝ) : Prop :=
  let r := sqrt (d^2 + (line1_length/2)^2) in
  let radius_relation := r^2 = ((d + 6)^2 + (line2_length / 2)^2) in
  12 * d = 83

theorem distance_between_parallel_lines :
  ∃ (d : ℝ), distance_between_lines d ∧ d = 6 + 11/12 := sorry

end distance_between_parallel_lines_l181_181378


namespace even_function_iff_a_eq_1_l181_181159

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- Lean 4 statement for the given math proof problem
theorem even_function_iff_a_eq_1 :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 1 :=
sorry

end even_function_iff_a_eq_1_l181_181159


namespace intersection_A_B_l181_181138

open set

def A : set ℝ := { x | log 3 (3 * x - 2) < 1 }
def B : set ℝ := { x | x < 1 }
def intersection : set ℝ := { x | (2 / 3 : ℝ) < x ∧ x < 1 }

theorem intersection_A_B : A ∩ B = intersection := sorry

end intersection_A_B_l181_181138


namespace find_a4_l181_181611

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * a (n - 1) = a n * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 2 + a 6 = 34 ∧ a 3 * a 5 = 64

-- Statement
theorem find_a4 (a : ℕ → ℝ) (h : given_sequence_conditions a) : a 4 = 8 :=
sorry

end find_a4_l181_181611


namespace time_to_pass_platform_l181_181421

def train_length : ℝ := 1200  -- train length in meters
def tree_crossing_time : ℝ := 80  -- time to cross a tree in seconds
def platform_length : ℝ := 1000  -- platform length in meters
def speed (distance time : ℝ) : ℝ := distance / time  -- speed in m/s

theorem time_to_pass_platform : 
  speed train_length tree_crossing_time = 15 →
  (train_length + platform_length) / 15 = 2200 / 15 := 
by
  intros h1
  rw [h1]
  apply rfl

end time_to_pass_platform_l181_181421


namespace triangulation_polygon_l181_181522

theorem triangulation_polygon (k m : Nat) (h_k : k = 80) (h_m : m = 50) : 
  2 * m + k - 2 = 178 := by
  rw [h_k, h_m]
  exact Nat.add_sub_cancel' (2 * 50 + 80) 2

end triangulation_polygon_l181_181522


namespace binomial_sum_sum_of_binomial_solutions_l181_181898

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181898


namespace evaluate_expression_l181_181524

theorem evaluate_expression (x : ℤ) (h : x + 1 = 4) : 
  (-3)^3 + (-3)^2 + (-3 * x) + 3 * x + 3^2 + 3^3 = 18 :=
by
  -- Since we know the condition x + 1 = 4
  have hx : x = 3 := by linarith
  -- Substitution x = 3 into the expression
  rw [hx]
  -- The expression after substitution and simplification
  sorry

end evaluate_expression_l181_181524


namespace university_pays_per_box_l181_181007

theorem university_pays_per_box 
  (box_length : ℕ)
  (box_width : ℕ)
  (box_height : ℕ)
  (total_volume_needed : ℕ)
  (total_min_cost : ℕ) :
  (box_length = 20) →
  (box_width = 20) →
  (box_height = 15) →
  (total_volume_needed = 3060000) →
  (total_min_cost = 255) →
  let volume_of_one_box := box_length * box_width * box_height in
  let number_of_boxes_needed := total_volume_needed / volume_of_one_box in
  let cost_per_box := total_min_cost / number_of_boxes_needed in
  cost_per_box = 0.50 :=
by
  -- Proof steps will be here
  sorry

end university_pays_per_box_l181_181007


namespace smallest_sum_B_c_l181_181208

theorem smallest_sum_B_c : 
  ∃ (B : ℕ) (c : ℕ), (0 ≤ B ∧ B ≤ 4) ∧ (c ≥ 6) ∧ 31 * B = 4 * (c + 1) ∧ B + c = 8 := 
sorry

end smallest_sum_B_c_l181_181208


namespace line_through_P_opp_intercepts_l181_181089

theorem line_through_P_opp_intercepts :
  ∀ (a b : ℝ),
  (a, b) ∈ { (2, 3) } →
  ∃ (m : ℝ), (m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ { (a, b) } → m * x + y = 0) ∨
               (∀ (x: ℝ), (1 + x - y = 0 ∨ 3 * x - 2 * y = 0)) := 
by
  sorry

end line_through_P_opp_intercepts_l181_181089


namespace inequality_solution_l181_181873

theorem inequality_solution (x : ℝ) :
  (\frac{x + 1}{x - 2} + \frac{x + 3}{3*x} ≥ 4) ↔ (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
sorry

end inequality_solution_l181_181873


namespace expression_of_f_l181_181929

theorem expression_of_f (f : ℤ → ℤ) (h : ∀ x, f (x - 1) = x^2 + 4 * x - 5) : ∀ x, f x = x^2 + 6 * x :=
by
  sorry

end expression_of_f_l181_181929


namespace maximum_triangle_area_l181_181083

noncomputable def tangent_line_eq (t : ℝ) : Prop :=
  ∀ x y : ℝ, y = -Real.exp(-t) * (x - t) + Real.exp(-t) ↔ x + Real.exp(t) * y = t + 1

noncomputable def triangle_area (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp(t))

theorem maximum_triangle_area :
  ∃ t ≥ 0, ∀ u ≥ 0, triangle_area t ≥ triangle_area u ∧ triangle_area t = 2 / Real.exp 1 := by
  sorry

end maximum_triangle_area_l181_181083


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181915

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181915


namespace inequality_solution_l181_181872

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l181_181872


namespace magnitude_of_z_l181_181934

theorem magnitude_of_z (z : ℂ) (h : sqrt 2 * I * z = 1 + I) : complex.abs z = 1 := 
by 
  sorry

end magnitude_of_z_l181_181934


namespace sample_not_representative_l181_181443

-- Define the events A and B
def A : Prop := ∃ (x : Type), (x → Prop) -- A person has an email address
def B : Prop := ∃ (x : Type), (x → Prop) -- A person uses the internet

-- Define the dependence and characteristics
def is_dependent (A B : Prop) : Prop := A ∧ B -- A and B are dependent
def linked_to_internet_usage (A : Prop) (B : Prop) : Prop := A → B -- A is closely linked to B
def not_uniform_distribution (B : Prop) : Prop := ¬ (∀ x, B x) -- Internet usage is not uniformly distributed

-- Define the subset condition
def is_subset_of_regular_users (A B : Prop) : Prop := ∀ x, A x → B x -- A represents regular internet users

-- Prove the given statement
theorem sample_not_representative
  (A B : Prop)
  (h_dep : is_dependent A B)
  (h_link : linked_to_internet_usage A B)
  (h_not_unif : not_uniform_distribution B)
  (h_subset : is_subset_of_regular_users A B) :
  ¬ represents_urban_population A :=
sorry

end sample_not_representative_l181_181443


namespace solve_equation_l181_181698

theorem solve_equation (x y z : ℝ) (n k m : ℤ)
  (h1 : sin x ≠ 0)
  (h2 : cos y ≠ 0)
  (h_eq : (sin x ^ 2 + 1 / sin x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * cos z)
  : ∃ n k m : ℤ, x = π / 2 + π * n ∧ y = π * k ∧ z = 2 * π * m :=
by
  sorry

end solve_equation_l181_181698


namespace check_perpendicular_counterparts_l181_181188

def is_perpendicular_counterpart_set (M : Set (ℝ × ℝ)) :=
  ∀ (x1 y1 : ℝ), (x1, y1) ∈ M → ∃ (x2 y2 : ℝ), (x2, y2) ∈ M ∧ x1 * x2 + y1 * y2 = 0

def M1 : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 / x)}
def M2 : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 ** x))}
def M3 : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.exp x - 2)}
def M4 : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.sin x + 1)}

theorem check_perpendicular_counterparts :
  ¬ is_perpendicular_counterpart_set M1 ∧
  ¬ is_perpendicular_counterpart_set M2 ∧
  is_perpendicular_counterpart_set M3 ∧
  is_perpendicular_counterpart_set M4 :=
by
  sorry -- Proof is not required

end check_perpendicular_counterparts_l181_181188


namespace sample_not_representative_l181_181432

-- Definitions
def has_email_address (person : Type) : Prop := sorry
def uses_internet (person : Type) : Prop := sorry

-- Problem statement: prove that the sample is not representative of the urban population.
theorem sample_not_representative (person : Type)
  (sample : set person)
  (h_sample_size : set.size sample = 2000)
  (A : person → Prop)
  (A_def : ∀ p, A p ↔ has_email_address p)
  (B : person → Prop)
  (B_def : ∀ p, B p ↔ uses_internet p)
  (dependent : ∀ p, A p → B p)
  : ¬ is_representative_sample sample := 
sorry

end sample_not_representative_l181_181432


namespace two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l181_181857

theorem two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one :
  (2.85 = 2850 * 0.001) := by
  sorry

end two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l181_181857


namespace boundary_of_locus_l181_181613

noncomputable def boundary_equation (a S : ℝ) (h₁ : a > 0) : ℝ → ℝ :=
  λ x, - (a / (2 * S)) * x^2 + (S / (2 * a))

theorem boundary_of_locus
  {a S : ℝ} (h₁ : a > 0) (P Q : ℝ × ℝ)
  (h₂ : P.2 > 0) (h₃ : Q.2 > 0)
  (AP_perp_PQ : ((P.2) / (P.1 + a)) * ((Q.2) / (Q.1 - a)) = -1)
  (quadrilateral_area : ((P.1 - P.1) * (Q.2 - P.2) - (Q.1 - P.1) * (0 - 0)) / 2 = S)
  : ∀ x, P.2 = boundary_equation a S h₁ x :=
sorry

end boundary_of_locus_l181_181613


namespace mark_any_integer_point_l181_181779

theorem mark_any_integer_point (N : ℕ) (points : List ℕ)
  (h1 : ∀ i j, i ≠ j → Nat.gcd (points.nth i - points.nth j) N = 1 ∧ i, j < points.length)
  (h2 : ∃ i j, (points.nth i - points.nth j) % 3 = 0) :
  ∀ (M : ℕ), M ≤ N → ∃ (ops : List (ℕ × ℕ)), ops.length < N ∧
    ∀ (i : ℕ) (op : ℕ × ℕ), i < ops.length → 
    let op := ops.nth i
    let pointA := points.nth op.fst
    let pointB := points.nth op.snd
    pointA ≤ N ∧ pointB ≤ N ∧
    (points := points.map (λ x, if x = pointB then (pointA + pointB) / 3 else x)) ∧
    pointA = M ∨ pointB = M := sorry

end mark_any_integer_point_l181_181779


namespace triangle_similarities_l181_181381

variables {A B C D E F : Type} [linear_ordered_field A B C D E F]
variables {EF BC AC AB : Prop}

theorem triangle_similarities 
  (h1 : EF ∥ BC)
  (h2 : FD ∥ AC)
  (h3 : ED ∥ AB) : 
  triangle.similar A E F F D B ∧ 
  triangle.similar F D B E D C ∧ 
  triangle.similar E D C A B C ∧ 
  triangle.similar A E F A B C :=
by
  sorry

end triangle_similarities_l181_181381


namespace contradiction_proof_l181_181382

theorem contradiction_proof (a b : ℕ) (h : a + b ≥ 3) : (a ≥ 2) ∨ (b ≥ 2) :=
sorry

end contradiction_proof_l181_181382


namespace fudge_piece_size_l181_181787

theorem fudge_piece_size (side1 side2 num_pieces : ℕ) (h1 : side1 = 18) (h2 : side2 = 29) (h3 : num_pieces = 522)
: (((side1 * side2) / num_pieces) = 1) → sqrt (side1 * side2 / num_pieces) = 1 := 
by sorry

end fudge_piece_size_l181_181787


namespace vector_sum_is_correct_l181_181621

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (0, 1)

-- Define the vectors AB and AC
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vectorAC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- State the theorem
theorem vector_sum_is_correct : vectorAB + vectorAC = (-3, -1) :=
by
  sorry

end vector_sum_is_correct_l181_181621


namespace fraction_of_90_l181_181001

theorem fraction_of_90 : (1 / 2) * (1 / 3) * (1 / 6) * (90 : ℝ) = (5 / 2) := by
  sorry

end fraction_of_90_l181_181001


namespace hexagon_interior_angle_Q_l181_181194

theorem hexagon_interior_angle_Q 
  (A B C D E F : ℕ)
  (hA : A = 135) (hB : B = 150) (hC : C = 120) (hD : D = 130) (hE : E = 100)
  (hex_angle_sum : A + B + C + D + E + F = 720) :
  F = 85 :=
by
  rw [hA, hB, hC, hD, hE] at hex_angle_sum
  sorry

end hexagon_interior_angle_Q_l181_181194


namespace tangent_line_at_1_monotonically_decreasing_F_l181_181965

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x - Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.ln x
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x a - g x

theorem tangent_line_at_1
  (a : ℝ) (ha : a = Real.exp 1 - 1)
  (hf : ∀ x, f x a = x^2 + a * x - Real.exp x) :
  (∃ m b, ∀ x, m = 1 ∧ b = -1 → (f 1 a + m * (x - 1)) = (m * x + b)) := by
  sorry

theorem monotonically_decreasing_F
  (a : ℝ)
  (hf : ∀ x, f x a = x^2 + a * x - Real.exp x)
  (hg : ∀ x, g x = Real.ln x)
  (hF : ∀ x, F x a = f x a - g x) :
  (∀ x ∈ Set.Ioc 0 1, Fderiv ℝ (λ x, F x a) ≤ 0) → a ≤ Real.exp 1 - 1 := by
  sorry

end tangent_line_at_1_monotonically_decreasing_F_l181_181965


namespace escalator_rate_is_15_l181_181483

noncomputable def rate_escalator_moves (escalator_length : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_length / time) - person_speed

theorem escalator_rate_is_15 :
  rate_escalator_moves 200 5 10 = 15 := by
  sorry

end escalator_rate_is_15_l181_181483


namespace prev_geng_yin_year_2010_is_1950_l181_181816

def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "You", "Xu", "Hai"]

def cycleLength : Nat := Nat.lcm 10 12

def prev_geng_yin_year (current_year : Nat) : Nat :=
  if cycleLength ≠ 0 then
    current_year - cycleLength
  else
    current_year -- This line is just to handle the case where LCM is incorrectly zero, which shouldn't happen practically.

theorem prev_geng_yin_year_2010_is_1950 : prev_geng_yin_year 2010 = 1950 := by
  sorry

end prev_geng_yin_year_2010_is_1950_l181_181816


namespace domain_of_f_range_of_f_symmetry_of_f_l181_181313

-- Define the function
def f (x : ℝ) : ℝ := (Real.sqrt (x^2 - x^4)) / (|x - 1| - 1)

-- Prove the equivalent statements
theorem domain_of_f : {x : ℝ | -1 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1} = {x : ℝ | f(x).domain } := 
sorry

theorem range_of_f : set.range f = set.Ioo (-1) 1 := 
sorry

theorem symmetry_of_f : ∀ x : ℝ, f(-x) = -f(x) := 
sorry

end domain_of_f_range_of_f_symmetry_of_f_l181_181313


namespace technician_round_trip_percentage_l181_181771

theorem technician_round_trip_percentage (D : ℝ) (hD : D > 0) :
  let round_trip_distance := 2 * D
  let distance_traveled := D + 0.20 * D in
  (distance_traveled / round_trip_distance) * 100 = 60 :=
  by
  sorry

end technician_round_trip_percentage_l181_181771


namespace person_b_lap_time_l181_181298

noncomputable def lap_time_b (a_lap_time : ℕ) (meet_time : ℕ) : ℕ :=
  let combined_speed := 1 / meet_time
  let a_speed := 1 / a_lap_time
  let b_speed := combined_speed - a_speed
  1 / b_speed

theorem person_b_lap_time 
  (a_lap_time : ℕ) 
  (meet_time : ℕ) 
  (h1 : a_lap_time = 80) 
  (h2 : meet_time = 30) : 
  lap_time_b a_lap_time meet_time = 48 := 
by 
  rw [lap_time_b, h1, h2]
  -- Provided steps to solve the proof, skipped here only for statement
  sorry

end person_b_lap_time_l181_181298


namespace pumps_to_fill_tires_l181_181499

-- Definitions based on conditions
def AirPerTire := 500
def PumpVolume := 50
def FlatTires := 2
def PercentFullTire1 := 0.40
def PercentFullTire2 := 0.70

-- The formal statement/proof problem
theorem pumps_to_fill_tires : (FlatTires * AirPerTire + AirPerTire * (1 - PercentFullTire1) + AirPerTire * (1 - PercentFullTire2)) / PumpVolume = 29 := 
by sorry

end pumps_to_fill_tires_l181_181499


namespace sequence_inequality_l181_181187

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n + 1)
  (general_term : ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1) :
  (∑ k in finset.range n, (a k / a (k + 1)) < n / 2) := by
  sorry

end sequence_inequality_l181_181187


namespace correct_result_value_at_neg_one_l181_181575

theorem correct_result (x : ℝ) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (A - (incorrect - A)) = 4 * x^2 + x + 4 :=
by sorry

theorem value_at_neg_one (x : ℝ := -1) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (4 * x^2 + x + 4) = 7 :=
by sorry

end correct_result_value_at_neg_one_l181_181575


namespace sin_trig_identity_l181_181097

theorem sin_trig_identity : 
  sin 50 * (1 + sqrt 3 * tan 10) = 1 := 
sorry

end sin_trig_identity_l181_181097


namespace objects_meet_l181_181384

variables (v0 a g : ℝ) (t : ℝ)
-- Conditions
axiom initial_velocity_positive : v0 > 0
axiom height_above_a : a > 0
axiom gravity_positive : g > 0

def motion_upward (v0 t g : ℝ) := v0 * t - (1/2) * g * t^2
def motion_downward (t g : ℝ) := (1/2) * g * t^2

-- Meeting condition
theorem objects_meet (h : a = v0 * t) : 
  motion_upward v0 t g + motion_downward t g = a :=
begin
  unfold motion_upward motion_downward,
  rw h,
  linarith,
end

#check objects_meet

end objects_meet_l181_181384


namespace total_kids_in_lawrence_county_l181_181635

theorem total_kids_in_lawrence_county :
  ∀ (h c T : ℕ), h = 274865 → c = 38608 → T = h + c → T = 313473 :=
by
  intros h c T h_eq c_eq T_eq
  rw [h_eq, c_eq] at T_eq
  exact T_eq

end total_kids_in_lawrence_county_l181_181635


namespace rectangular_coordinate_equation_length_of_chord_AB_l181_181968

open Set
open Real
open Complex

noncomputable def parametric_line_equation := 
  ∀ t : ℝ, ∃ x y: ℝ, 
    x = 2 + (1 / 2) * t ∧ 
    y = (sqrt 3 / 2) * t

noncomputable def polar_curve_C :=
  ∀ (ρ θ : ℝ), ρ ≥ 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * π → 
    ρ * (sin θ)^2 = 8 * (cos θ)

theorem rectangular_coordinate_equation : 
  (∀ (ρ θ : ℝ), ρ ≥ 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * π → ρ * (sin θ)^2 = 8 * (cos θ)) →
  ∀ x y : ℝ, (y^2 = 8 * x) :=
sorry

theorem length_of_chord_AB :
  (∀ (ρ θ : ℝ), ρ ≥ 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * π → ρ * (sin θ)^2 = 8 * (cos θ)) →
  (∀ t a b : ℝ, 
    (∃ x y: ℝ, x = 2 + (1 / 2) * t ∧ y = (sqrt 3 / 2) * t) → 
    a = 6 ∧ b = 2 / 3 ∧
    sqrt ((6 - (2 / 3))^2 + (4 * sqrt 3 + (4 * sqrt 3 / 3))^2) = 32 / 3) :=
sorry

end rectangular_coordinate_equation_length_of_chord_AB_l181_181968


namespace change_amount_l181_181516

theorem change_amount 
    (tank_capacity : ℕ) 
    (current_fuel : ℕ) 
    (price_per_liter : ℕ) 
    (total_money : ℕ) 
    (full_tank : tank_capacity = 150) 
    (fuel_in_truck : current_fuel = 38) 
    (cost_per_liter : price_per_liter = 3) 
    (money_with_donny : total_money = 350) : 
    total_money - ((tank_capacity - current_fuel) * price_per_liter) = 14 :=
by
sorr

end change_amount_l181_181516


namespace sequence_third_value_l181_181615

theorem sequence_third_value :
  ∀ (a : ℕ → ℕ), a 1 = 6 ∧ a 2 = 12 ∧ a 4 = 24 ∧ a 5 = 30 ∧ (∀ n, a (n + 5) = a n) → a 3 = 18 :=
by
  intros a h
  cases h with h1 h,
  cases h with h2 h,
  cases h with h4 h,
  cases h with h5 h_cycle,
  sorry

end sequence_third_value_l181_181615


namespace part_b_distance_AC_BF_equals_l181_181258

-- Define everything given in the problem as Lean types/constants.

constant Point : Type
constant Line : Type
constant Plane : Type
constant AB : ℝ
constant O : Point
constant AE BF : Line
constant planes_perpendicular : Plane -> Plane -> Prop
constant square_side : Plane -> Point -> Point -> ℝ
constant line_intersection : Line -> Line -> Point

axiom AB_eq_4 : AB = 4
axiom squares_perpendicular_planes {A B C D E F : Point} {plane1 plane2: Plane} :
  -- Assuming squares have vertices ABCD and ABEF respectively
  square_side plane1 A B = AB ∧ 
  square_side plane1 B C = AB ∧ 
  square_side plane1 C D = AB ∧
  square_side plane1 D A = AB ∧
  square_side plane2 A B = AB ∧
  square_side plane2 B E = AB ∧
  square_side plane2 E F = AB ∧
  square_side plane2 F A = AB ∧ 
  planes_perpendicular plane1 plane2

axiom point_of_intersection :
  O = line_intersection AE BF

noncomputable def distance_from_B_to_line_of_intersection_between_planes_DOC_DAF {B D O C F A M N : Point} (DM : Line) : ℝ := 
  -- The line of intersection between planes (DOC) and (DAF)
  -- Assuming DM is correctly defined.
  sorry

noncomputable def distance_between_lines_AC_BF {A C B F : Point} : ℝ := 
  -- Distance calculation between lines AC and BF
  (4*real.sqrt 3)/3

theorem part_b_distance_AC_BF_equals:
  ∀ (A B C D E F O: Point) (plane1 plane2: Plane),
  square_side plane1 A B = AB ∧ 
  square_side plane1 B C = AB ∧ 
  square_side plane1 C D = AB ∧
  square_side plane1 D A = AB ∧
  square_side plane2 A B = AB ∧
  square_side plane2 B E = AB ∧
  square_side plane2 E F = AB ∧
  square_side plane2 F A = AB ∧
  planes_perpendicular plane1 plane2 ∧ 
  O = line_intersection AE BF → 
  distance_between_lines_AC_BF = (4*real.sqrt 3)/3 := 
  by 
    sorry

end part_b_distance_AC_BF_equals_l181_181258


namespace largest_D_l181_181291

theorem largest_D (D : ℝ) : (∀ x y : ℝ, x^2 + 2 * y^2 + 3 ≥ D * (3 * x + 4 * y)) → D ≤ Real.sqrt (12 / 17) :=
by
  sorry

end largest_D_l181_181291


namespace proof_a_plus_b_sqrt_ab_gt_c_l181_181559

theorem proof_a_plus_b_sqrt_ab_gt_c (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let a := x + y + sqrt (x * y)
  let b := y + z + sqrt (y * z)
  let c := z + x + sqrt (z * x)
  in a + b + sqrt (a * b) > c :=
by
  sorry

end proof_a_plus_b_sqrt_ab_gt_c_l181_181559


namespace initial_volume_of_solution_l181_181426

theorem initial_volume_of_solution (V : ℝ) (h0 : 0.10 * V = 0.08 * (V + 20)) : V = 80 :=
by
  sorry

end initial_volume_of_solution_l181_181426


namespace math_problem_l181_181598

theorem math_problem 
  (x : ℂ) 
  (h : x^3 + x^2 + x = -1) : 
  (x^(-28) + x^(-27) + ... + x^(-2) + x^(-1) + 1 + x^(1) + x^(2) + ... + x^(27) + x^(28)) = 1 := 
by 
  sorry

end math_problem_l181_181598


namespace spherical_caps_ratio_l181_181000

theorem spherical_caps_ratio (r : ℝ) (m₁ m₂ : ℝ) (σ₁ σ₂ : ℝ)
  (h₁ : r = 1)
  (h₂ : σ₁ = 2 * π * m₁ + π * (1 - (1 - m₁)^2))
  (h₃ : σ₂ = 2 * π * m₂ + π * (1 - (1 - m₂)^2))
  (h₄ : σ₁ + σ₂ = 5 * π)
  (h₅ : m₁ + m₂ = 2) :
  (2 * m₁ + (1 - (1 - m₁)^2)) / (2 * m₂ + (1 - (1 - m₂)^2)) = 3.6 :=
sorry

end spherical_caps_ratio_l181_181000


namespace class1_median_is_32_class2_mode_is_35_l181_181233

def class1_scores : List ℕ := [20, 32, 31, 32, 31, 25, 32, 36, 38, 39]
def class2_scores : List ℕ := [25, 27, 35, 30, 34, 35, 35, 27, 36, 32]

theorem class1_median_is_32 (sorted_class1 : List ℕ := class1_scores.qsort (≤)) :
  (sorted_class1.nth 4).getD 0 + (sorted_class1.nth 5).getD 0 = 64 → sorted_class1.median = 32 := by
  sorry

theorem class2_mode_is_35 : class2_scores.mode = some 35 := by
  sorry

end class1_median_is_32_class2_mode_is_35_l181_181233


namespace initial_average_is_16_l181_181336

def average_of_six_observations (A : ℝ) : Prop :=
  ∃ s : ℝ, s = 6 * A

def new_observation (A : ℝ) (new_obs : ℝ := 9) : Prop :=
  ∃ t : ℝ, t = 7 * (A - 1)

theorem initial_average_is_16 (A : ℝ) (new_obs : ℝ := 9) :
  (average_of_six_observations A) → (new_observation A new_obs) → A = 16 :=
by
  intro h1 h2
  sorry

end initial_average_is_16_l181_181336


namespace number_of_differences_l181_181196

def is_difference (x y d : ℕ) : Prop := x ≠ y ∧ d = abs (x - y)

theorem number_of_differences : 
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃! n : ℕ, n = 9 ∧ ∀ d ∈ insert 0 (s.bind (λ x, s.image (λ y, abs (x - y)))), d ≠ 0 → d <= n :=
sorry

end number_of_differences_l181_181196


namespace change_received_l181_181293

theorem change_received (saved_money : Real := 1500) (vip_tickets : Nat := 5)
    (regular_tickets : Nat := 7) (discount_tickets : Nat := 4) 
    (wheelchair_tickets : Nat := 2) (vip_price : Real := 120) 
    (regular_price : Real := 60) (discount_price : Real := 30)
    (wheelchair_price : Real := 40) (tax_rate : Real := 0.05)
    (processing_fee : Real := 25) (payment : Real := 2000) 
    : payment - (vip_tickets * vip_price + regular_tickets * regular_price + discount_tickets * discount_price + wheelchair_tickets * wheelchair_price)
    * (1 + tax_rate) + processing_fee = 694 :=
by 
  -- Lean code to calculate the change
  let vip_cost := vip_tickets * vip_price
  let regular_cost := regular_tickets * regular_price
  let discount_cost := discount_tickets * discount_price
  let wheelchair_cost := wheelchair_tickets * wheelchair_price
  let total_cost_before_tax := vip_cost + regular_cost + discount_cost + wheelchair_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_before_processing_fee := total_cost_before_tax + tax
  let total_cost := total_cost_before_processing_fee + processing_fee
  let change := payment - total_cost
  change = 694

  sorry

end change_received_l181_181293


namespace grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l181_181830

theorem grid_spiral_infinite_divisible_by_68 (n : ℕ) :
  ∃ (k : ℕ), ∃ (m : ℕ), ∃ (t : ℕ), 
  let A := t + 0;
  let B := t + 4;
  let C := t + 12;
  let D := t + 8;
  (k = n * 68 ∧ (n ≥ 1)) ∧ 
  (m = A + B + C + D) ∧ (m % 68 = 0) := by
  sorry

theorem grid_spiral_unique_center_sums (n : ℕ) :
  ∀ (i j : ℕ), 
  let Si := n * 68 + i;
  let Sj := n * 68 + j;
  ¬ (Si = Sj) := by
  sorry

end grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l181_181830


namespace count_integer_values_l181_181118

-- Statement of the problem in Lean 4
theorem count_integer_values (x : ℤ) : 
  (7 * x^2 + 23 * x + 20 ≤ 30) → 
  ∃ (n : ℕ), n = 6 :=
sorry

end count_integer_values_l181_181118


namespace variance_transformation_l181_181571

noncomputable def D (X : Type) [RandomVariable X] : ℝ := sorry

variable {X : Type} [RandomVariable X]

theorem variance_transformation (h : D X = 2) : D (3 * X + 2) = 18 :=
by
  sorry

end variance_transformation_l181_181571


namespace function_properties_l181_181957

variable {α : Type*} [LinearOrderedField α]

def increasing_on (g : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → g x < g y

theorem function_properties
  (g : α → α) (m n : α)
  (h_inc : increasing_on g (Set.Ioo m n))
  (h_n_lt_negm : 0 < n ∧ n < -m)
  : let f := λ x, (g x)^2 - (g (-x))^2 in
    (∀ x, x ∈ Set.Ioo (-n) n → x ∈ Set.Ioo (-n) n ∧ f x = -(f (-x))) :=
by
  sorry

example (g : α → α) (m n : α)
  (h_inc : increasing_on g (Set.Ioo m n))
  (h_n_lt_negm : 0 < n ∧ n < -m)
  : let f := λ x, (g x)^2 - (g (-x))^2 in
    (∀ x, x ∈ Set.Ioo (-n) n → x ∈ Set.Ioo (-n) n ∧ f x = -(f (-x))) :=
  function_properties g m n h_inc h_n_lt_negm

end function_properties_l181_181957


namespace ant_meeting_point_YW_l181_181756

noncomputable def perimeter (XY YZ XZ : ℝ) : ℝ := XY + YZ + XZ

theorem ant_meeting_point_YW (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 10) (hXZ : XZ = 12) :
  let P := perimeter XY YZ XZ in
  let half_P := P / 2 in
    (XY + (half_P - XY)) = 8 + 7 → YW = 3 :=
by sorry

end ant_meeting_point_YW_l181_181756


namespace colors_diff_l181_181759

def coloring (c : ℤ → ℕ) : Prop :=
  ∀ a b c d : ℤ, 
  (b - a) = (d - c) → 
  (c a = c c) → 
  ((c b = c d) → 
  (∀ x, 0 ≤ x ∧ x ≤ b - a → c (a + x) = c (c + x)))

theorem colors_diff (c : ℤ → ℕ) (h : coloring c) : c (-1982) ≠ c (1982) := 
sorry

end colors_diff_l181_181759


namespace count_perfect_squares_l181_181200

theorem count_perfect_squares :
  {N : ℕ // N < 100}.count (λ N, ∃ k, k * k = N ∧ 36 ∣ k * k) = 8 := sorry

end count_perfect_squares_l181_181200


namespace solve_trig_eq_l181_181015

theorem solve_trig_eq (k : ℤ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * (Real.sin t)^2 - Real.sin (2 * t) + 3 * Real.cos t^2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
sorry

end solve_trig_eq_l181_181015


namespace number_of_solutions_l181_181354

-- Define the universal set
def universal_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset that must be included
def required_subset : Finset ℕ := {2, 3}

-- Define the function counting the valid subsets
def count_valid_subsets (universal_set required_subset : Finset ℕ) : ℕ :=
universal_set.powerset.count (λ X, required_subset ⊆ X)

-- The theorem that asserts the answer is 16
theorem number_of_solutions : count_valid_subsets universal_set required_subset = 16 :=
by sorry

end number_of_solutions_l181_181354


namespace derivative_f_l181_181340

noncomputable def f (x : ℝ) := x * Real.cos x - Real.sin x

theorem derivative_f :
  ∀ x : ℝ, deriv f x = -x * Real.sin x :=
by
  sorry

end derivative_f_l181_181340


namespace change_amount_l181_181514

theorem change_amount 
    (tank_capacity : ℕ) 
    (current_fuel : ℕ) 
    (price_per_liter : ℕ) 
    (total_money : ℕ) 
    (full_tank : tank_capacity = 150) 
    (fuel_in_truck : current_fuel = 38) 
    (cost_per_liter : price_per_liter = 3) 
    (money_with_donny : total_money = 350) : 
    total_money - ((tank_capacity - current_fuel) * price_per_liter) = 14 :=
by
sorr

end change_amount_l181_181514


namespace solve_system_of_floor_eqs_l181_181034

noncomputable def floor_function (x : ℝ) : ℤ := int.floor x

theorem solve_system_of_floor_eqs (x y : ℝ) (hx : floor_function (x + y - 3) = 2 - x) (hy : floor_function (x + 1) + floor_function (y - 7) + x = y) :
  x = 3 ∧ y = -1 :=
by
  sorry

end solve_system_of_floor_eqs_l181_181034


namespace find_x_l181_181398

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x_l181_181398


namespace ellipse_parabola_area_l181_181565

theorem ellipse_parabola_area
  (a b : ℝ) (ha : a > b) (hb : b > 0) 
  (hecc : real.sqrt (6)/3 = c/a)
  (hfocal : 2 * c = 4 * real.sqrt 2)
  (parabola_focus : p > 0) :
  (∃ a b, (a > b ∧ b > 0) ∧
  (real.sqrt (6)/3 = c/a) ∧ 
  (2 * c = 4 * real.sqrt 2) ∧
  ((x: ℝ)(y: ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1)  → ∀ p F, 
  F = (0, 2) → 
  x^2 = 8y → 
  ∀ P Q, distinct P Q ∧ 
  ∃ xy, (line_PQ: ℝ → ℝ) → y - 1 = k * x ∧ 
  (overrightarrow_EP : ℝ) ∧ 
  (overrightarrow_FQ * ℝ) =
  ∃ x1 x2, x1 + x2 = (∃ k m, k - 2 * k * x1 + x2 -> 
  x1 ⋅ x2 = (0) ∧ unique_solution
  (SΔ_FP Q)) = ( (\triangle_FP Q = 1/2 * 3 * real.sqrt(72/25) = 0) :

end ellipse_parabola_area_l181_181565


namespace square_perimeter_l181_181334

theorem square_perimeter (a : ℝ) (side : ℝ) (perimeter : ℝ) (h1 : a = 144) (h2 : side = Real.sqrt a) (h3 : perimeter = 4 * side) : perimeter = 48 := by
  sorry

end square_perimeter_l181_181334


namespace Donny_change_l181_181509

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l181_181509


namespace polynomial_possible_n_values_l181_181845

theorem polynomial_possible_n_values :
  let a := 448
  let zeros := {a, (a / 2 + irr), (a / 2 - irr)}
  ∃ (n : ℤ), (∀ (x ∈ zeros), x > 0) ∧ 
  (x ≠ y → x > 0 ∧ y > 0) ∧ 
  (∀ (z : ℤ), z ≠ 0) →
  49952 := sorry

end polynomial_possible_n_values_l181_181845


namespace point_geq_l181_181719

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

variables
  (O A B C M X Y Z : Point)
  (c : Circle)
  (h1 : c.center = O)
  (h2 : dist O A = c.radius)
  (h3 : is_perpendicular O A B C M)
  (h4 : is_on_larger_arc B C X)
  (h5 : intersects_at X A B C Y)
  (h6 : intersects_again X M c Z)

theorem point_geq :
  dist A Y ≥ dist M Z :=
sorry

end point_geq_l181_181719


namespace teapot_volume_proof_l181_181369

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem teapot_volume_proof (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0.5)
  (h2 : arithmetic_sequence a d 7 + arithmetic_sequence a d 8 + arithmetic_sequence a d 9 = 2.5) :
  arithmetic_sequence a d 5 = 0.5 :=
by {
  sorry
}

end teapot_volume_proof_l181_181369


namespace hyperbola_equation_l181_181046

/-- Given that:
1. A hyperbola and an ellipse share common foci.
2. The ellipse equation is 4x² + y² = 64.
3. The eccentricities of the hyperbola and the ellipse are reciprocals of each other.
Prove that the equation of the hyperbola is y²/36 - x²/12 = 1. -/
theorem hyperbola_equation (a b : ℝ) (h1 : ∀ (x y : ℝ), 4 * x^2 + y^2 = 64) 
  (h2 : a^2 + b^2 = 48) (h3 : (real.sqrt (a^2 + b^2)) / a = (2 * real.sqrt 3) / 3) :
  ∃ (a b : ℝ), a = 6 ∧ b = 2 * real.sqrt 3 ∧ (λ x y, (y^2 / a^2) - (x^2 / b^2) = 1) :=
by sorry

end hyperbola_equation_l181_181046


namespace angelfish_goldfish_difference_l181_181078

-- Given statements
variables {A G : ℕ}
def goldfish := 8
def total_fish := 44

-- Conditions
axiom twice_as_many_guppies : G = 2 * A
axiom total_fish_condition : A + G + goldfish = total_fish

-- Theorem
theorem angelfish_goldfish_difference : A - goldfish = 4 :=
by
  sorry

end angelfish_goldfish_difference_l181_181078


namespace tank_ratio_two_l181_181926

variable (T1 : ℕ) (F1 : ℕ) (F2 : ℕ) (T2 : ℕ)

-- Assume the given conditions
axiom h1 : T1 = 48
axiom h2 : F1 = T1 / 3
axiom h3 : F1 - 1 = F2 + 3
axiom h4 : T2 = F2 * 2

-- The theorem to prove
theorem tank_ratio_two (h1 : T1 = 48) (h2 : F1 = T1 / 3) (h3 : F1 - 1 = F2 + 3) (h4 : T2 = F2 * 2) : T1 / T2 = 2 := by
  sorry

end tank_ratio_two_l181_181926


namespace Donny_change_l181_181508

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l181_181508


namespace largest_value_of_x_l181_181763

theorem largest_value_of_x :
  ∃ (x : ℝ), (x / 7 + 3 / (7 * x) = 2 / 3) ∧ (x = (7 + real.sqrt 22) / 3) :=
sorry

end largest_value_of_x_l181_181763


namespace intersecting_lines_a_value_l181_181349

theorem intersecting_lines_a_value :
  ∀ t a b : ℝ, (b = 12) ∧ (b = 2 * a + t) ∧ (t = 4) → a = 4 :=
by
  intros t a b h
  obtain ⟨hb1, hb2, ht⟩ := h
  sorry

end intersecting_lines_a_value_l181_181349


namespace min_value_l181_181930

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_l181_181930


namespace sample_not_representative_l181_181442

-- Define the events A and B
def A : Prop := ∃ (x : Type), (x → Prop) -- A person has an email address
def B : Prop := ∃ (x : Type), (x → Prop) -- A person uses the internet

-- Define the dependence and characteristics
def is_dependent (A B : Prop) : Prop := A ∧ B -- A and B are dependent
def linked_to_internet_usage (A : Prop) (B : Prop) : Prop := A → B -- A is closely linked to B
def not_uniform_distribution (B : Prop) : Prop := ¬ (∀ x, B x) -- Internet usage is not uniformly distributed

-- Define the subset condition
def is_subset_of_regular_users (A B : Prop) : Prop := ∀ x, A x → B x -- A represents regular internet users

-- Prove the given statement
theorem sample_not_representative
  (A B : Prop)
  (h_dep : is_dependent A B)
  (h_link : linked_to_internet_usage A B)
  (h_not_unif : not_uniform_distribution B)
  (h_subset : is_subset_of_regular_users A B) :
  ¬ represents_urban_population A :=
sorry

end sample_not_representative_l181_181442


namespace two_equal_squares_exists_l181_181465

theorem two_equal_squares_exists (s : ℕ) :
  ∃ a1 a2, a1 ≠ a2 ∧ side_length_eq s a1 ∧ side_length_eq s a2 :=
by
  -- Given conditions:
  -- A square divided by 18 lines
  -- 9 lines parallel to one side of the square
  -- 9 lines parallel to the other side
  -- These lines result in 100 rectangles
  -- Exactly 9 of these rectangles are squares
  sorry

noncomputable def side_length_eq (n : ℕ) (a : ℕ) : Prop :=
  a^2 = n

end two_equal_squares_exists_l181_181465


namespace valid_outfit_count_l181_181204

def total_shirts : Nat := 8
def total_pants : Nat := 5
def total_hats : Nat := 7
def total_colors : Nat := 5

def total_outfits := total_shirts * total_pants * total_hats
def matching_pants_hat_outfits := total_colors * total_shirts
def valid_outfits := total_outfits - matching_pants_hat_outfits

theorem valid_outfit_count :
  valid_outfits = 240 :=
by
  -- let's assert the calculations with sorries for academics
  -- total outfits
  have t_outfits : total_outfits = 280 := by 
    sorry
  -- matching pants and hat outfits
  have m_outfits : matching_pants_hat_outfits = 40 := by 
    sorry
  -- therefore, the valid outfits count
  show valid_outfits = 240 from 
    sorry

end valid_outfit_count_l181_181204


namespace average_speed_is_correct_l181_181061

-- Definitions based on given conditions.
def distance1 : ℕ := 210
def time1 : ℕ := 3
def distance2 : ℕ := 270
def time2 : ℕ := 4

-- Definitions for total distance and total time.
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- The average speed calculation.
def average_speed := total_distance / total_time

-- The proof statement that needs to be shown.
theorem average_speed_is_correct :
  average_speed = 68.57 :=
by
  sorry

end average_speed_is_correct_l181_181061


namespace find_side_length_of_left_square_l181_181345

theorem find_side_length_of_left_square (x : ℕ) 
  (h1 : x + (x + 17) + (x + 11) = 52) : 
  x = 8 :=
by
  -- The proof will go here
  sorry

end find_side_length_of_left_square_l181_181345


namespace solve_equation_l181_181711

theorem solve_equation (x y z : ℝ) (m n : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1 / (sin^2 x ))^3 + (cos^2 y + 1 / (cos^2 y))^3 = 16 * cos z) →
  (cos z = 1) ∧ 
  (∃ m : ℤ, x = π / 2 + π * m) ∧ 
  (∃ n : ℤ, y = π * n) ∧ 
  (∃ m : ℤ, z = 2 * π * m) :=
by
  intros hsin_cos hcos_sin heq
  sorry

end solve_equation_l181_181711


namespace magnitude_of_a_minus_2b_l181_181933

def vec_a : ℝ × ℝ × ℝ := (2, -3, 5)
def vec_b : ℝ × ℝ × ℝ := (-3, 1, -4)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem magnitude_of_a_minus_2b :
  magnitude (vec_a.1 - 2 * vec_b.1, vec_a.2 - 2 * vec_b.2, vec_a.3 - 2 * vec_b.3) = real.sqrt 258 :=
by
  sorry

end magnitude_of_a_minus_2b_l181_181933


namespace president_vice_president_ways_l181_181721

theorem president_vice_president_ways (total_members : ℕ) (boys : ℕ) (girls : ℕ) (h1 : total_members = 30) (h2 : boys = 15) (h3 : girls = 15) :
  (boys * girls) + (girls * boys) = 450 :=
by
  rw [h2, h3]
  norm_num
  sorry

end president_vice_president_ways_l181_181721


namespace magnitude_of_OP_l181_181948

noncomputable def unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 * v.1 + v.2 * v.2 = 1

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 * v.1 + v.2 * v.2).sqrt

theorem magnitude_of_OP :
  ∀ (e1 e2 : ℝ × ℝ),
  unit_vector e1 →
  unit_vector e2 →
  dot_product e1 e2 = 1 / 2 →
  let OP := (3 * e1.1 + 2 * e2.1, 3 * e1.2 + 2 * e2.2) in
  magnitude OP = real.sqrt 19 :=
by
  intros e1 e2 he1 he2 he_dot
  let OP := (3 * e1.1 + 2 * e2.1, 3 * e1.2 + 2 * e2.2)
  show magnitude OP = real.sqrt 19
  sorry

end magnitude_of_OP_l181_181948


namespace length_of_AB_l181_181773

theorem length_of_AB (x y : ℝ) (A B C : ℝ × ℝ) (h1 : ∠ B = 90)
  (h2 : dist A C = 225) (h3 : (C.2 - A.2) / (C.1 - A.1) = 4 / 3) : 
  dist A B = 180 := by
  sorry

end length_of_AB_l181_181773


namespace correct_operation_l181_181404

theorem correct_operation (h1 : ¬ (sqrt 9 = 3 ∨ sqrt 9 = -3))
                         (h2 : (-2) ^ 3 = -8)
                         (h3 : - (abs (-3)) = -3)
                         (h4 : -(2 ^ 2) = -4) : 
                         ∃ operation, operation = D :=
by
  sorry

end correct_operation_l181_181404


namespace largest_possible_value_of_EH_l181_181843

def is_cyclic_quadrilateral (EFGH : List ℕ) : Prop :=
  EFGH.length = 4 ∧ 
  (∀ x ∈ EFGH, x < 20) ∧
  EFGH[0] * EFGH[2] = EFGH[1] * EFGH[3] ∧ 
  (List.nodup EFGH)

theorem largest_possible_value_of_EH (EFGH : List ℕ) : 
  is_cyclic_quadrilateral EFGH →
  ∃ EH, EH = sqrt 394 ∧
  let e := EFGH[0]
  let f := EFGH[1]
  let g := EFGH[2]
  let h := EFGH[3]
  2 * EH * EH = e * e + f * f + g * g + h * h :=
by 
  sorry

-- Example usage to indicate that you want to demonstrate EH = sqrt(394) given the conditions.

end largest_possible_value_of_EH_l181_181843


namespace number_of_students_l181_181315

theorem number_of_students (n : ℕ) (H : n > 0) (candy : ℕ) :
  (candy = 100) → 
  (∀ k, k < candy → (k % n ≠ 0 → (k + n) % candy = k + 1 → k = 0 ∨ k = candy - 1)) → 
  n = 11 :=
begin
  sorry,
end

end number_of_students_l181_181315


namespace buratino_got_the_result_l181_181030

namespace BlobsProblem

variables (x : ℕ)

def buratino_operations (x : ℕ) : ℝ :=
  (7 * x - 8) / 6 + 9

theorem buratino_got_the_result (x : ℕ) :
  buratino_operations x = 18^(1 / 6 : ℝ) :=
  sorry

end BlobsProblem

end buratino_got_the_result_l181_181030


namespace rectangle_area_y_coords_l181_181122

theorem rectangle_area_y_coords (h : set (ℝ × ℝ)) 
  (h_hyp : ∃ a b c d : ℝ × ℝ, a.2 = 2 ∧ b.2 = 5 ∧ c.2 = 10 ∧ d.2 = 7 ∧ 
      (a.1 = d.1) ∧ (b.1 = c.1) ∧ 
      ((a + c) / 2 = (b + d) / 2)) 
  : let height := 5 in let width := 8 in  (height * width) = 40 :=
by
  let height := 5
  let width := 8
  show height * width = 40
  sorry

end rectangle_area_y_coords_l181_181122


namespace upstream_distance_l181_181064

theorem upstream_distance
  (v : ℕ) (c := 2 : ℕ)
  (downstream_distance := 45 : ℕ)
  (upstream_time := 5 : ℕ)
  (effective_downstream_speed := v + c)
  (effective_upstream_speed := v - c)
  (downstream_time := downstream_distance / effective_downstream_speed = 5)
  (v_value : v = 7):
  upstream_distance = effective_upstream_speed * upstream_time → upstream_distance = 25 :=
by
  sorry

end upstream_distance_l181_181064


namespace range_of_f_l181_181360

-- Definition of the function and the domain restriction
def f (x : ℝ) : ℝ := log 2 x + 3

-- Definition of the domain
def domain (x : ℝ) : Prop := x ≥ 1

-- The theorem statement: the range of the function f for x ≥ 1
theorem range_of_f : ∀ y : ℝ, ∃ x : ℝ, domain x ∧ y = f x ↔ y ≥ 3 := sorry

end range_of_f_l181_181360


namespace solution_set_inequality_l181_181268

-- Define the function and its properties.
def isOdd (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = -f(x)
def isIncreasingOn (f : ℝ → ℝ) (S : Set ℝ) := ∀ x y ∈ S, x < y → f(x) < f(y)

theorem solution_set_inequality (f : ℝ → ℝ) 
  (h_odd : isOdd f)
  (h_increasing : isIncreasingOn f (Set.Ioi 0))
  (h_f1_eq_0 : f 1 = 0) :
  { x : ℝ | x * (f x - f (-x)) < 0 } = Set.Ioo (-1:ℝ) 0 ∪ Set.Ioo 0 1 := 
by
  sorry

end solution_set_inequality_l181_181268


namespace largest_circle_at_A_l181_181341

/--
Given a pentagon with side lengths AB = 16 cm, BC = 14 cm, CD = 17 cm, DE = 13 cm, and EA = 14 cm,
and given five circles with centers A, B, C, D, and E such that each pair of circles with centers at
the ends of a side of the pentagon touch on that side, the circle with center A
has the largest radius.
-/
theorem largest_circle_at_A
  (rA rB rC rD rE : ℝ) 
  (hAB : rA + rB = 16)
  (hBC : rB + rC = 14)
  (hCD : rC + rD = 17)
  (hDE : rD + rE = 13)
  (hEA : rE + rA = 14) :
  rA ≥ rB ∧ rA ≥ rC ∧ rA ≥ rD ∧ rA ≥ rE := 
sorry

end largest_circle_at_A_l181_181341


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181910

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181910


namespace convex_polyhedron_face_parity_l181_181318

theorem convex_polyhedron_face_parity 
  (P : Polyhedron) 
  (convex : is_convex P) 
  (odd_faces : odd (number_of_faces P)) : 
  ∃ F : Face P, even (number_of_edges F) :=
sorry

end convex_polyhedron_face_parity_l181_181318


namespace binomial_probability_l181_181280

namespace binomial_proof

open ProbabilityTheory

def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- Given a random variable ξ that follows a binomial distribution B(6, 1/2),
  prove that the probability that ξ equals 3 is 5/16. -/
theorem binomial_probability : 
  let ξ : ℝ → ℝ := λ ω, ite (binom 6 3 = ω) 1 0
  in (binom 6 3) * (0.5^3) * (0.5^(6-3)) = 5 / 16 :=
by
  sorry

end binomial_proof

end binomial_probability_l181_181280


namespace koala_fiber_consumption_l181_181256

theorem koala_fiber_consumption (x : ℝ) (h : 0.40 * x = 8) : x = 20 :=
sorry

end koala_fiber_consumption_l181_181256


namespace perimeter_triangle_APR_l181_181379

-- Define the basic structure of the problem with relevant conditions.
variable {A B C P R Q : Point}
variable {circle : Circle}
variable {AB AC : ℝ}

-- Tangent properties and lengths
axiom tangent_eq_AB : TangentFromPoint A B circle
axiom tangent_eq_AC : TangentFromPoint A C circle

-- Known lengths
axiom length_AB : AB = 24
axiom length_diff_AP_AR : ∀ AP AR, AP = AR + 3

-- Segment PR calculation
axiom segment_intersect : TangentIntersectSegments A B P circle ∧ TangentIntersectSegments A C R circle 
axiom tangent_touch_Q : TangentTouchCircle Q circle

-- Goal: prove the perimeter of triangle APR is 57
theorem perimeter_triangle_APR : 
  ∃ (AP AR PR : ℝ), 
    (AR = AP - 3) ∧ 
    (PR = 45 - 2 * AR) ∧ 
    (AB = 24) ∧ 
    (AC = 24) ∧ 
    (AP + AR + PR = 57) := 
sorry

end perimeter_triangle_APR_l181_181379


namespace ascending_order_of_x_y_z_l181_181557

variable {a b x y z : ℝ}
variable (ha : 0 < a)
variable (hb : a < b)
variable (hc : b < 1)
variable (hx : x = a ^ b)
variable (hy : y = b ^ a)
variable (hz : z = Real.logBase b a)

theorem ascending_order_of_x_y_z : x < y ∧ y < z := by
  sorry

end ascending_order_of_x_y_z_l181_181557


namespace meteorological_period_l181_181288

-- Definitions based on problem conditions
def rainy_day (d : ℕ) : Prop := d = 1
def mixed_rainy_day (d : ℕ) : Prop := d = 9
def clear_nights (n : ℕ) : Prop := n = 6
def clear_days (d : ℕ) : Prop := d = 7

theorem meteorological_period
  (rainy_days mixed_rainy_days clear_nights clear_days : ℕ) 
  (h1 : rainy_days = 1) 
  (h2 : mixed_rainy_days = 9) 
  (h3 : clear_nights = 6)
  (h4 : clear_days = 7) :
  ∃ total_days fully_clear_days : ℕ, total_days = 12 ∧ fully_clear_days = 2 :=
by
  exists 12, 2
  split
  · exact rfl
  · exact rfl

end meteorological_period_l181_181288


namespace proof_problem_l181_181564

variables {f : ℝ → ℝ} {f' : ℝ → ℝ}

-- Conditions
def condition_1 (h : ℝ → ℝ) (h' : ℝ → ℝ) : Prop :=
∀ x : ℝ, differentiable_at ℝ h x ∧ ∀ x ∈ Ioi 0, x * (deriv h x) + x^2 < h x

theorem proof_problem (h : ℝ → ℝ) (h' : ℝ → ℝ) 
    (h_diff : ∀ x, differentiable_at ℝ h x)
    (H : ∀ x ∈ Ioi 0, x * (deriv h x) + x^2 < h x) : 
    (2 * h 1 > h 2 + 2) ∧ (3 * h 1 > h 3 + 3) :=
sorry

end proof_problem_l181_181564


namespace john_salary_increase_l181_181022

namespace MathProof

def percentage_increase (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem john_salary_increase :
  percentage_increase 60 70 = 16.67 :=
by
  sorry

end MathProof

end john_salary_increase_l181_181022


namespace stewart_farm_sheep_count_l181_181827

theorem stewart_farm_sheep_count :
  (∃ S H : ℕ, (S + 7*H = 0) ∧ (230 * H = 12880) ∧ (150 * S = 6300) ∧ (S = 8)) :=
proof
  sorry

end stewart_farm_sheep_count_l181_181827


namespace min_kinder_surprises_l181_181092

theorem min_kinder_surprises (gnomes : Finset ℕ) (hs: gnomes.card = 12) :
  ∃ k, k ≤ 166 ∧ ∀ kinder_surprises : Finset (Finset ℕ), kinder_surprises.card = k → 
  (∀ s ∈ kinder_surprises, s.card = 3 ∧ s ⊆ gnomes ∧ (∀ t ∈ kinder_surprises, s ≠ t → s ≠ t)) → 
  ∀ g ∈ gnomes, ∃ s ∈ kinder_surprises, g ∈ s :=
sorry

end min_kinder_surprises_l181_181092


namespace find_b_minus_a_l181_181749

/-- Proof to find the value of b - a given the inequality conditions on x.
    The conditions are:
    1. x - a < 1
    2. x + b > 2
    3. 0 < x < 4
    We need to show that b - a = -1.
-/
theorem find_b_minus_a (a b x : ℝ) 
  (h1 : x - a < 1) 
  (h2 : x + b > 2) 
  (h3 : 0 < x) 
  (h4 : x < 4) 
  : b - a = -1 := 
sorry

end find_b_minus_a_l181_181749


namespace cos_squared_pi_over_4_minus_alpha_l181_181555

theorem cos_squared_pi_over_4_minus_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 3 / 4) :
  Real.cos (Real.pi / 4 - α) ^ 2 = 9 / 25 :=
by
  sorry

end cos_squared_pi_over_4_minus_alpha_l181_181555


namespace Donny_change_l181_181512

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l181_181512


namespace even_function_f_f_for_pos_x_l181_181032

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then x^3 + Real.log(x + 1) else -x^3 + Real.log(1 - x)

theorem even_function_f (x : ℝ) : (f (-x) = f x) :=
begin
  intros,
  rw f,
  rw f,
  split_ifs,
  { rw h_1,
    split_ifs;
    rw [neg_neg, Real.log_neg_eq_log_sub, Real.log_one_sub_eq_neg_log, neg_add, add_neg_cancel_right] },
  { rw h_2,
    split_ifs;
    rw [neg_neg, Real.log_neg_eq_log_sub, Real.log_one_sub_eq_neg_log, neg_add, add_neg_cancel_right] },
  { linarith },
  { linarith }
end

theorem f_for_pos_x (x : ℝ) (hx : 0 < x) : f x = -x^3 + Real.log(1 - x) :=
begin
  intros,
  have h_f_eq : f x = f (-x) := even_function_f x,
  rw f at h_f_eq,
  rw if_neg (not_le.mpr hx) at h_f_eq,
  rw f at h_f_eq,
  rw if_pos (le_of_lt (neg_lt_zero.mpr hx)) at h_f_eq,
  exact h_f_eq,
end

end even_function_f_f_for_pos_x_l181_181032


namespace bobby_ate_chocolate_l181_181074

variable (C : ℕ)

axiom ate_candy_38 : ∃ pieces, pieces = 38
axiom ate_candy_36_more : ∃ pieces, pieces = 36
axiom ate_candy_more_than_chocolate : ∃ extra, extra = 58

theorem bobby_ate_chocolate (C : ℕ) (h1 : ∃ pieces, pieces = 38) 
(h2 : ∃ pieces, pieces = 36) (h3 : ∃ extra, extra = 58) : 
C = 58 :=
begin
  sorry
end

end bobby_ate_chocolate_l181_181074


namespace find_a_for_even_l181_181153

def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

theorem find_a_for_even (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = f a x) ↔ a = 1 :=
by
  -- proof steps here
  sorry

end find_a_for_even_l181_181153


namespace smallest_real_constant_inequality_l181_181506

theorem smallest_real_constant_inequality (n : ℕ) (h_n : 0 < n) (x : ℕ → ℝ) (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i) :
  ∑ k in finset.range n, (1 / (k + 1) * ∑ j in finset.range (k + 1), x (j + 1))^2 ≤ 4 * ∑ k in finset.range n, (x (k + 1))^2 :=
by {
  sorry
}

end smallest_real_constant_inequality_l181_181506


namespace circle_equation_passing_through_P_l181_181935

-- Define the problem conditions
def P : ℝ × ℝ := (3, 1)
def l₁ (x y : ℝ) := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) := x + 2 * y - 7 = 0

-- The main theorem statement
theorem circle_equation_passing_through_P :
  ∃ (α β : ℝ), 
    ((α = 4 ∧ β = -1) ∨ (α = 4 / 5 ∧ β = 3 / 5)) ∧ 
    ((x - α)^2 + (y - β)^2 = 5) :=
  sorry

end circle_equation_passing_through_P_l181_181935


namespace solve_equation_l181_181708

theorem solve_equation (x y z : ℝ) (m n : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1 / (sin^2 x ))^3 + (cos^2 y + 1 / (cos^2 y))^3 = 16 * cos z) →
  (cos z = 1) ∧ 
  (∃ m : ℤ, x = π / 2 + π * m) ∧ 
  (∃ n : ℤ, y = π * n) ∧ 
  (∃ m : ℤ, z = 2 * π * m) :=
by
  intros hsin_cos hcos_sin heq
  sorry

end solve_equation_l181_181708


namespace log_base_problem_l181_181210

noncomputable def log_of_base (base value : ℝ) : ℝ := Real.log value / Real.log base

theorem log_base_problem (x : ℝ) (h : log_of_base 16 (x - 3) = 1 / 4) : 1 / log_of_base (x - 3) 2 = 1 := 
by
  sorry

end log_base_problem_l181_181210


namespace Tim_balloon_count_l181_181847

variables (Dan Tim : ℝ)
constant hDan : Dan = 29.0
constant hRatio : Dan = 7 * Tim

theorem Tim_balloon_count : Tim = 4 := by
  have h1 : Tim = Dan / 7 := by linarith [hDan, hRatio]
  have h2 : Dan / 7 = 4 := by simp [hDan] -- Calculation simplification
  linarith [h1, h2]

end Tim_balloon_count_l181_181847


namespace cos_value_third_quadrant_l181_181139

theorem cos_value_third_quadrant (x : Real) (h1 : Real.sin x = -1 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_value_third_quadrant_l181_181139


namespace total_grocery_bill_l181_181309

theorem total_grocery_bill
    (hamburger_meat_cost : ℝ := 5.00)
    (crackers_cost : ℝ := 3.50)
    (frozen_vegetables_bags : ℝ := 4)
    (frozen_vegetables_cost_per_bag : ℝ := 2.00)
    (cheese_cost : ℝ := 3.50)
    (discount_rate : ℝ := 0.10) :
    let total_cost_before_discount := hamburger_meat_cost + crackers_cost + (frozen_vegetables_bags * frozen_vegetables_cost_per_bag) + cheese_cost
    let discount := total_cost_before_discount * discount_rate
    let total_cost_after_discount := total_cost_before_discount - discount
in
total_cost_after_discount = 18.00 :=
by
   -- total_cost_before_discount = 5.00 + 3.50 + (4 * 2.00) + 3.50 = 20.00
   -- discount = 20.00 * 0.10 = 2.00
   -- total_cost_after_discount = 20.00 - 2.00 = 18.00
   sorry

end total_grocery_bill_l181_181309


namespace second_group_men_count_l181_181985

theorem second_group_men_count
  (M B : ℕ)
  (h1 : M = 2 * B)
  (h2 : ∀ (x : ℕ), (12 * M + 16 * B) * 5 = (x * M + 24 * B) * 4) :
  ∃ (x : ℕ), x = 13 :=
by
  use 13
  sorry

end second_group_men_count_l181_181985


namespace actors_duration_l181_181238

-- Definition of conditions
def actors_at_a_time := 5
def total_actors := 20
def total_minutes := 60

-- Main statement to prove
theorem actors_duration : total_minutes / (total_actors / actors_at_a_time) = 15 := 
by
  sorry

end actors_duration_l181_181238


namespace sequence_terms_divisible_by_b_l181_181679

theorem sequence_terms_divisible_by_b (a b : ℕ) :
  let d := Nat.gcd a b in
  (d = (List.range (b + 1)).filter (λ n, (a * n) % b = 0).length) :=
by
  sorry

end sequence_terms_divisible_by_b_l181_181679


namespace probability_B_given_A_l181_181542

def probability_of_event (total_outcomes favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

def conditional_probability (total_outcomes A_outcomes B_outcomes : ℕ) : ℚ :=
  let P_A := probability_of_event total_outcomes A_outcomes
  let P_B := probability_of_event total_outcomes B_outcomes
  let P_A_and_B := probability_of_event total_outcomes (A_outcomes ∩ B_outcomes).to_nat
  P_A_and_B / P_A

theorem probability_B_given_A :
  let total_outcomes := 10
  let A_outcomes := 4
  let B_outcomes := 3
  let A_and_B_outcomes := 3
  conditional_probability total_outcomes A_outcomes A_and_B_outcomes = 3 / 4 :=
by sorry

end probability_B_given_A_l181_181542


namespace sum_of_valid_k_equals_26_l181_181886

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181886


namespace green_valley_ratio_l181_181485

variable (j s : ℕ)

theorem green_valley_ratio (h : (3 / 4 : ℚ) * j = (1 / 2 : ℚ) * s) : s = 3 / 2 * j :=
by
  sorry

end green_valley_ratio_l181_181485


namespace even_function_iff_a_eq_1_l181_181160

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- Lean 4 statement for the given math proof problem
theorem even_function_iff_a_eq_1 :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 1 :=
sorry

end even_function_iff_a_eq_1_l181_181160


namespace convex_polygon_perimeter_bounded_l181_181653

theorem convex_polygon_perimeter_bounded (P : Type) [Polygon P] (h_convex : IsConvex P) (h_contained : IsContainedInSquare P 1) : 
  perimeter P ≤ 4 := 
sorry

end convex_polygon_perimeter_bounded_l181_181653


namespace probability_prime_product_of_6_dice_roll_l181_181223

theorem probability_prime_product_of_6_dice_roll :
  (6 : ℕ) = 6 ∧ (∀ (die : ℕ), die ∈ {1, 2, 3, 4, 5, 6, 7, 8}) →
  let favorable_outcomes : ℕ := 24 in
  let total_outcomes : ℕ := 8^6 in
  favorable_outcomes / total_outcomes = (3 : ℚ) / 32768 :=
begin
  sorry
end

end probability_prime_product_of_6_dice_roll_l181_181223


namespace trajectory_equation_l181_181624

theorem trajectory_equation (x y : ℝ) : | |x| - |y| | = 4 ↔ |x| - |y| = 4 ∨ |x| - |y| = -4 := by
  sorry

end trajectory_equation_l181_181624


namespace pumps_to_fill_tires_l181_181500

-- Definitions based on conditions
def AirPerTire := 500
def PumpVolume := 50
def FlatTires := 2
def PercentFullTire1 := 0.40
def PercentFullTire2 := 0.70

-- The formal statement/proof problem
theorem pumps_to_fill_tires : (FlatTires * AirPerTire + AirPerTire * (1 - PercentFullTire1) + AirPerTire * (1 - PercentFullTire2)) / PumpVolume = 29 := 
by sorry

end pumps_to_fill_tires_l181_181500


namespace two_positive_real_roots_one_positive_one_negative_real_root_l181_181920

def quadraticRoots (m : ℝ) : list ℝ := 
  let a := 1
  let b := -2 * (m + 2)
  let c := m^2 - 1
  let discriminant := b^2 - 4 * a * c
  if discriminant < 0 then [] 
  else if discriminant = 0 then [ -b / (2*a) ] 
  else 
    let sqrt_disc := Real.sqrt discriminant
    [ (-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a) ]

theorem two_positive_real_roots (m : ℝ) : 
  (quadraticRoots m).length = 2 ∧ (∀ x ∈ quadraticRoots m, x > 0) ↔ 
  (m ∈ set.Icc (-5/4) (-1) ∪ set.Ioi 1) := by
  sorry

theorem one_positive_one_negative_real_root (m : ℝ) :
  (quadraticRoots m).length = 2 ∧ (∃ x y ∈ quadraticRoots m, x > 0 ∧ y < 0) ↔ 
  (m ∈ set.Ioo (-1) 1) := by
  sorry

end two_positive_real_roots_one_positive_one_negative_real_root_l181_181920


namespace bullets_shot_per_person_l181_181367

-- Definitions based on conditions
def num_people : ℕ := 5
def initial_bullets_per_person : ℕ := 25
def total_remaining_bullets : ℕ := 25

-- Statement to prove
theorem bullets_shot_per_person (x : ℕ) :
  (initial_bullets_per_person * num_people - num_people * x) = total_remaining_bullets → x = 20 :=
by
  sorry

end bullets_shot_per_person_l181_181367


namespace range_of_m_l181_181088

def triangle (x y : ℝ) : ℝ := x * (2 - y)

theorem range_of_m (m : ℝ) : (∀ x : ℝ, triangle (x + m) x < 1) ↔ m ∈ Ioo (-4) 0 :=
by
  sorry

end range_of_m_l181_181088


namespace minimum_h10_l181_181067

def intense_function (f : ℕ → ℤ) : Prop :=
  ∀ (x y z : ℕ), x ≠ z → y ≠ z → f(x) + f(y) + f(z) > z^3

def minimum_T (h : ℕ → ℤ) : Prop := 
  (∀ (x y z: ℕ), x ≠ z → y ≠ z → h(x) + h(y) + h(z) > z^3) ∧ 
   h(1) + h(2) + h(3) + h(4) + h(5) + h(6) + h(7) + 
   h(8) + h(9) + h(10) + h(11) + h(12) + h(13) + h(14) + h(15) is minimized

theorem minimum_h10 (h : ℕ → ℤ) (a : ℤ) (T : ℤ):
  intense_function h ∧ minimum_T h → h(1) = 1 → h(2) = 1 → h(10) = 999 :=
by
  sorry

end minimum_h10_l181_181067


namespace cylinder_surface_area_is_correct_l181_181569

-- Define the conditions and necessary variables
noncomputable def sphere_radius : ℝ := sqrt 6
noncomputable def cylinder_height : ℝ := 2
noncomputable def cylinder_radius : ℝ := sqrt 5

-- Define the surface area calculation function for the cylinder
noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

-- The main theorem stating the problem
theorem cylinder_surface_area_is_correct :
  cylinder_surface_area cylinder_radius cylinder_height = (10 + 4 * sqrt 5) * Real.pi := by
  sorry

end cylinder_surface_area_is_correct_l181_181569


namespace sum_abc_geq_half_l181_181190

theorem sum_abc_geq_half (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
(h_abs_sum : |a - b| + |b - c| + |c - a| = 1) : 
a + b + c ≥ 0.5 := 
sorry

end sum_abc_geq_half_l181_181190


namespace initial_investment_proof_l181_181518

noncomputable def initial_investment (A : ℝ) (r t : ℕ) : ℝ := 
  A / (1 + r / 100) ^ t

theorem initial_investment_proof : 
  initial_investment 1000 8 8 = 630.17 := sorry

end initial_investment_proof_l181_181518


namespace patrick_savings_l181_181667

/-- 
  Given:
  1. The cost of the bicycle is $150.
  2. Patrick saved half the price of the bicycle.
  3. Patrick lent $50 to his friend at a 5% annual interest rate.
  4. The loan was repaid after 8 months.
  
  Prove that Patrick now has $126.67.
-/
theorem patrick_savings (total_price : ℝ) (half_saved : ℝ) (loan : ℝ) (interest_rate : ℝ) 
                        (repayment_months : ℝ) (final_amount : ℝ) : 
  total_price = 150 ∧
  half_saved = total_price / 2 ∧
  loan = 50 ∧
  interest_rate = 0.05 ∧
  repayment_months = 8 / 12 ∧
  final_amount = half_saved + loan * (1 + interest_rate * repayment_months) → 
  final_amount = 126.67 :=
begin
  intros,
  sorry
end

end patrick_savings_l181_181667


namespace inequality_solution_l181_181874

theorem inequality_solution (x : ℝ) :
  (\frac{x + 1}{x - 2} + \frac{x + 3}{3*x} ≥ 4) ↔ (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
sorry

end inequality_solution_l181_181874


namespace option_B_incorrect_l181_181009

variable {α β : ℝ}

noncomputable def trig_identity_A : Prop := sin (α + real.pi) = -sin α
noncomputable def trig_identity_B : Prop := cos (-α + β) = -cos (α - β)
noncomputable def trig_identity_C : Prop := sin (-α - 2 * real.pi) = -sin α
noncomputable def trig_identity_D : Prop := cos (-α - β) = cos (α + β)

theorem option_B_incorrect (α β : ℝ) : 
  ¬ trig_identity_B :=
by
  intro h
  -- by cosine properties we know cos (-α + β) = cos (α - β)
  -- thus, ¬ trig_identity_B states
  have : cos (-α + β) = cos (α - β), by sorry
  rw this at h
  -- then, part of assumption states cos (α - β) = -cos (α - β),
  -- which is equivalent to false
  have : cos (α - β) = -cos (α - β), by sorry
  have h_false : false := sorry
  exact h_false

end option_B_incorrect_l181_181009


namespace swimmers_meetings_l181_181038

theorem swimmers_meetings 
  (length : ℕ) (speed1 speed2 : ℕ) (rest1 : ℕ) (time : ℕ) 
  (h1 : length = 120) 
  (h2 : speed1 = 4) 
  (h3 : speed2 = 3) 
  (h4 : rest1 = 30) 
  (h5 : time = 15 * 60):
  let total_meetings := 17 in
  total_meetings = 17 :=
by
  sorry

end swimmers_meetings_l181_181038


namespace minimum_distance_MN_l181_181109

-- Definitions of the parabola and circle
def parabola (M : ℝ × ℝ) : Prop := (M.2)^2 = M.1
def circle (C : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) := {P | (P.1 - C.1)^2 + (P.2 - C.2)^2 = radius^2}

-- Definition of symmetry line and symmetrical point function
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0
def symmetric_point (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry  -- placeholder for the symmetry function

-- Center of the original circle
def center_C1 := (-1, 4)
-- Radius of the original circle
def radius_C1 := 1
-- Center of the symmetric circle
def center_C : ℝ × ℝ := symmetric_point center_C1 (λ x y, symmetry_line x y)

-- Symmetric circle
def symmetric_circle := circle center_C radius_C1

-- Distance function |MN|
def distance (M N : ℝ × ℝ) : ℝ := real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Main theorem
theorem minimum_distance_MN :
  ∃ (M : ℝ × ℝ) (N : ℝ × ℝ), parabola M ∧ symmetric_circle N ∧ distance M N = (real.sqrt 11) / 2 - 1 :=
begin
  sorry
end

end minimum_distance_MN_l181_181109


namespace total_sales_volume_l181_181473

/-- Sales of the two types of suites after 12 months given the initial conditions and growth rates -/
theorem total_sales_volume (a₁ a₂ : ℕ) 
  (r₁ : ℝ) (d₂ : ℕ)
  (initial_sales : ℕ) 
  (reference_data : ℕ → ℝ)
  (h1 : initial_sales = 20)
  (h2 : r₁ = 1.1)
  (h3 : d₂ = 10)
  (h4 : reference_data 11 = 2.9)
  (h5 : reference_data 12 = 3.1)
  (h6 : reference_data 13 = 3.5) :
  let sales_110_geom := 20 * (1 - reference_data 12) / (1 - r₁)
      sales_90_arith := 12 * 20 + (12 * 11 / 2) * 10 in
  sales_110_geom + sales_90_arith = 1320 :=
by
  sorry

end total_sales_volume_l181_181473


namespace number_of_blocks_l181_181355

-- Abstract definition to represent the orthographic views of the figure
def orthographic_views : Type := sorry  

-- The definition for the geometric body constructed from small cubic blocks with the given orthographic views
def geometric_body (views: orthographic_views) : Type := sorry

-- The number of small cubic blocks used to construct the geometric body is 8
theorem number_of_blocks (views: orthographic_views) :
  ∃ (body : geometric_body views), blocks_count body = 8 :=
sorry

end number_of_blocks_l181_181355


namespace sum_k_binomial_l181_181909

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181909


namespace ellipse_major_axis_length_l181_181454

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l181_181454


namespace sum_of_first_70_odd_cubed_is_12003775_l181_181658

theorem sum_of_first_70_odd_cubed_is_12003775 (x : ℕ) :
  let sum_even := 70 / 2 * (2 + 140)
  sum_even = 4970 →
  ∑ k in Finset.range 70, (2 * k + 1) ^ 3 = 12003775 :=
by
  let sum_even := 70 / 2 * (2 + 140)
  intro h1
  have h2 := by rw [sum_even, ((70:ℕ) / 2 * (2 + 140))] ; exact h1
  sorry

end sum_of_first_70_odd_cubed_is_12003775_l181_181658


namespace binomial_coeff_sum_l181_181623

theorem binomial_coeff_sum (a x : ℝ) (h : x ≠ 0) (ha : a ≠ 0) : 
  let binomial_sum := ∑ k in range (6 + 1), (binom 6 k) * (x^2)^(6 - k) * ((1 / (a * x)) ^ k) in
  binomial_sum = 64 :=
by sorry

end binomial_coeff_sum_l181_181623


namespace major_axis_length_of_intersecting_ellipse_l181_181457

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l181_181457


namespace num_correct_relations_l181_181481

theorem num_correct_relations :
  (¬ (sqrt 2).isRat ∧ 0 ∈ ℕ ∧ 2 ∈ ({1, 2} : Set ℕ) ∧ ∅ ≠ ({0} : Set ℕ)) → 1 = 1 := by
sorry

end num_correct_relations_l181_181481


namespace solve_for_x_l181_181987

theorem solve_for_x (x : ℝ) (h : x + real.sqrt 25 = real.sqrt 36) : x = 1 :=
  sorry

end solve_for_x_l181_181987


namespace sequence_sum_eq_l181_181745

noncomputable def x : ℕ → ℚ
| 0       := 2 / 3
| (n + 1) := x n / (2 * (2 * n + 1) * x n + 1)

theorem sequence_sum_eq :
  (Finset.range 2014).sum (λ n, x n) = 4028 / 4029 := 
sorry

end sequence_sum_eq_l181_181745


namespace total_fish_caught_l181_181834

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l181_181834


namespace log_power_function_value_l181_181568

theorem log_power_function_value :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) → f 3 = √3 → log 4 (f 2) = 1 / 4 :=
by
  intros h1 h2
  sorry

end log_power_function_value_l181_181568


namespace fib_gen_func_correct_lucas_gen_func_correct_l181_181528

noncomputable def fib_gen_func (x z : ℤ) : ℤ := z / (1 - x * z - z^2)
noncomputable def lucas_gen_func (x z : ℤ) : ℤ := (2 - x * z) / (1 - x * z - z^2)

theorem fib_gen_func_correct (x : ℤ) : 
  ∑ n in finset.range ∞, fib_poly n x * z^n = fib_gen_func x z := sorry

theorem lucas_gen_func_correct (x : ℤ) : 
  ∑ n in finset.range ∞, lucas_poly n x * z^n = lucas_gen_func x z := sorry

end fib_gen_func_correct_lucas_gen_func_correct_l181_181528


namespace product_of_two_numbers_l181_181411
noncomputable def find_product (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : ℝ :=
x * y

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : find_product x y h1 h2 = 200 :=
sorry

end product_of_two_numbers_l181_181411


namespace proof_problem_l181_181205

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a + b = 2 * c) ∧ (a * b = -5 * d) ∧ (c + d = 2 * a) ∧ (c * d = -5 * b)

theorem proof_problem (a b c d : ℝ) (h : problem_statement a b c d) : a + b + c + d = 30 :=
by
  sorry

end proof_problem_l181_181205


namespace find_difference_l181_181741

theorem find_difference (a b : ℝ)
    (h1 : rotate (2, 4) 90 (a, b) = P1)
    (h2 : reflect_y_eq_neg_x P1 = (-4, 2)) :
    b - a = 10 := by
  sorry

end find_difference_l181_181741


namespace triangle_area_heron_l181_181017

theorem triangle_area_heron (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 5) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 30 :=
by
  -- here we include the setup but leave the proof as sorry, as instructed
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  sorry

end triangle_area_heron_l181_181017


namespace solve_inequality_l181_181211

variable {R : Type} [LinearOrderedField R]

theorem solve_inequality (a b x : R) :
  ((a > 0) → (ax > b ↔ x > b / a)) ∧
  ((a < 0) → (ax > b ↔ x < b / a)) ∧
  ((a = 0) → ((b ≥ 0 → ∀ x, ¬ (ax > b)) ∧ (b < 0 → ∀ x, ax > b))) :=
by
  sorry

end solve_inequality_l181_181211


namespace ray_total_grocery_bill_l181_181310

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l181_181310


namespace time_for_A_to_complete_race_l181_181237

noncomputable def time_to_complete_race (distance race_distance : ℕ) (beat_by_meter beat_by_second : ℕ) : ℕ :=
  let V_B := 2 in -- since B's speed is known to be 2 meters/second
  race_distance / V_B

theorem time_for_A_to_complete_race : 
  ∀ (race_distance beat_by_meter beat_by_second : ℕ), 
  race_distance = 1000 → beat_by_meter = 20 → beat_by_second = 10 → 
  time_to_complete_race beat_by_meter race_distance beat_by_second = 490 := 
by
  intros race_distance beat_by_meter beat_by_second race_distance_eq beat_by_meter_eq beat_by_second_eq
  simp [time_to_complete_race, race_distance_eq, beat_by_meter_eq, beat_by_second_eq]
  sorry

end time_for_A_to_complete_race_l181_181237


namespace vector_subtraction_l181_181972

-- Given vectors a and b
def a : ℝ × ℝ × ℝ := (-7, 0, 1)
def b : ℝ × ℝ × ℝ := (6, 2, -1)

-- Desired result
def result : ℝ × ℝ × ℝ := (-37, -10, 6)

-- Proof statement
theorem vector_subtraction :
  (a.1 - 5*b.1, a.2 - 5*b.2, a.3 - 5*b.3) = result :=
sorry

end vector_subtraction_l181_181972


namespace remaining_pieces_l181_181519

theorem remaining_pieces (S : Set ℕ) (h1 : S = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (odds_removed : ∀ (s : Set ℕ), s ⊆ {1, 3, 5, 7, 9} → s.card = 4 → S \ s = (S.diff {1, 3, 5, 7, 9}).insert 9)
  (product_24_removed : ∃ (x y z : ℕ), Set.toFinset {x, y, z}.prod = 24)
  (remaining_even: ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ {x, y} ⊆ {2, 4, 6, 8}) :
  {2, 8} ⊆ S ∨ {6, 8} ⊆ S :=
by
  intros
  sorry

end remaining_pieces_l181_181519


namespace pictures_vertical_l181_181283

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l181_181283


namespace mode_median_proof_l181_181043

noncomputable def mode (data : List ℝ) : ℝ :=
  data.groupBy id data.cmp.count.maxBy (λ a, a.2.length).toList.head.1

noncomputable def median (data : List ℝ) : ℝ :=
  let sorted := data.qsort (· < ·)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

variables (scores : List ℝ)
#reduce scores = [9.1, 9.8, 9.1, 9.2, 9.9, 9.1, 9.9, 9.1]

theorem mode_median_proof :
  mode scores = 9.1 ∧ median scores = 9.15 := by sorry

end mode_median_proof_l181_181043


namespace parabola_vertex_l181_181727

theorem parabola_vertex :
  (∃ x y : ℝ, y^2 + 6 * y + 4 * x - 7 = 0 ∧ (x, y) = (4, -3)) :=
sorry

end parabola_vertex_l181_181727


namespace limit_b_n_zero_l181_181746

noncomputable def a_n : ℕ → ℝ := sorry -- Define the positive sequence

def b_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / n) * (Finset.sum (Finset.range n) (λ i, a_n (i + 1) / (1 + a_n(i + 1))))

theorem limit_b_n_zero (a_n : ℕ → ℝ) (h : ∀ n, 0 < a_n n) (h_lim : Filter.Tendsto a_n Filter.atTop (Filter.principal {0})) :
  Filter.Tendsto (b_n a_n) Filter.atTop (Filter.principal {0}) :=
sorry

end limit_b_n_zero_l181_181746


namespace probability_of_same_color_l181_181614

theorem probability_of_same_color (balls : Fin 4 → Bool) (h : ∀ i, balls i = tt ∨ balls i = ff) :
  let w := (Finset.univ.filter (λ i, balls i = tt)).card,
      b := (Finset.univ.filter (λ i, balls i = ff)).card in
  w = 3 ∧ b = 1 →
  (∑ i in (Finset.filter (λ x, (balls x.1) = (balls x.2)) (Finset.univ.product (Finset.univ.filter (λ i, i.1 < i.2)))),
   1) / (Finset.card ((Finset.univ.product (Finset.univ.filter (λ i, i.1 < i.2))))) = 1 / 2 :=
begin
  intros hw,
  sorry
end

end probability_of_same_color_l181_181614


namespace ratio_of_triangle_areas_l181_181249

theorem ratio_of_triangle_areas
  (XY XZ YZ : ℝ)
  (hXY : XY = 10)
  (hXZ : XZ = 15)
  (hYZ : YZ = 18)
  (angle_bisector : XW_is_angle_bisector_of_XYZ : ∀ W: point, is_angle_bisector (triangle.mk X Y Z) X W) :
  (area (triangle.mk X Y W) / area (triangle.mk X Z W) = (2 / 3)) :=
by
  sorry

end ratio_of_triangle_areas_l181_181249


namespace deductive_reasoning_syllogism_correct_l181_181240

-- Define the available options as enumerations
inductive SyllogismDefinition
| FirstPartSecondPartThirdPart
| MajorPremiseMinorPremiseConclusion
| InductionConjectureProof
| DiscussingInThreeParts

-- Define the problem in Lean 4
theorem deductive_reasoning_syllogism_correct :
  ∀ (s : SyllogismDefinition),
    (s = SyllogismDefinition.MajorPremiseMinorPremiseConclusion) ↔
    s = SyllogismDefinition.MajorPremiseMinorPremiseConclusion :=
by
  intro s
  split
  . intro hs
    exact hs
  . intro hs
    exact hs

end deductive_reasoning_syllogism_correct_l181_181240


namespace sample_not_representative_l181_181434

-- Definitions
def has_email_address (person : Type) : Prop := sorry
def uses_internet (person : Type) : Prop := sorry

-- Problem statement: prove that the sample is not representative of the urban population.
theorem sample_not_representative (person : Type)
  (sample : set person)
  (h_sample_size : set.size sample = 2000)
  (A : person → Prop)
  (A_def : ∀ p, A p ↔ has_email_address p)
  (B : person → Prop)
  (B_def : ∀ p, B p ↔ uses_internet p)
  (dependent : ∀ p, A p → B p)
  : ¬ is_representative_sample sample := 
sorry

end sample_not_representative_l181_181434


namespace find_other_root_l181_181744

theorem find_other_root (a b c x : ℝ) (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h₄ : a * (b + 2 * c) * x^2 + b * (2 * c - a) * x + c * (2 * a - b) = 0)
  (h₅ : a * (b + 2 * c) - b * (2 * c - a) + c * (2 * a - b) = 0) :
  ∃ y : ℝ, y = - (c * (2 * a - b)) / (a * (b + 2 * c)) :=
sorry

end find_other_root_l181_181744


namespace problem_solution_l181_181119

theorem problem_solution (m n : ℕ) (h1 : m + 7 < n + 3) 
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2 * n) / 6 = n + 3) 
  (h3 : (m + 7 + n + 3) / 2 = n + 3) : m + n = 12 := 
  sorry

end problem_solution_l181_181119


namespace solve_equation_l181_181699

theorem solve_equation (x y z : ℝ) (n k m : ℤ)
  (h1 : sin x ≠ 0)
  (h2 : cos y ≠ 0)
  (h_eq : (sin x ^ 2 + 1 / sin x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * cos z)
  : ∃ n k m : ℤ, x = π / 2 + π * n ∧ y = π * k ∧ z = 2 * π * m :=
by
  sorry

end solve_equation_l181_181699


namespace second_group_men_count_l181_181986

theorem second_group_men_count
  (M B : ℕ)
  (h1 : M = 2 * B)
  (h2 : ∀ (x : ℕ), (12 * M + 16 * B) * 5 = (x * M + 24 * B) * 4) :
  ∃ (x : ℕ), x = 13 :=
by
  use 13
  sorry

end second_group_men_count_l181_181986


namespace pairs_count_l181_181383

theorem pairs_count (n r : ℕ) (h : r ≤ n) :
  ∑ x in finset.range (r + 1), ((2 * (n + 1) - x) * (nat.choose n (n - x))) = nat.choose n r * (n! / (n - r)!) := by
  sorry

end pairs_count_l181_181383


namespace probability_A_greater_B_l181_181490

theorem probability_A_greater_B :
  let A := [10, 10, 1, 1, 1]
  let B := [5, 5, 5, 5, 1, 1, 1]
  let remaining_value (bag : List ℕ) (drawn : List ℕ) :=
    (bag.sum - drawn.sum)
  let valid_pairs := do
    a_drawn ← A.combinations 2
    b_drawn ← B.combinations 2
    guard $ remaining_value A a_drawn > remaining_value B b_drawn
    pure (a_drawn, b_drawn)
  let total_pairs := A.combinations 2.product B.combinations 2
  (valid_pairs.length / total_pairs.length : ℚ) = 9 / 35 :=
by
  sorry

end probability_A_greater_B_l181_181490


namespace hillary_stops_short_of_summit_l181_181974

noncomputable def distance_to_summit_from_base_camp : ℝ := 4700
noncomputable def hillary_climb_rate : ℝ := 800
noncomputable def eddy_climb_rate : ℝ := 500
noncomputable def hillary_descent_rate : ℝ := 1000
noncomputable def time_of_departure : ℝ := 6
noncomputable def time_of_passing : ℝ := 12

theorem hillary_stops_short_of_summit :
  ∃ x : ℝ, 
    (time_of_passing - time_of_departure) * hillary_climb_rate = distance_to_summit_from_base_camp - x →
    (time_of_passing - time_of_departure) * eddy_climb_rate = x →
    x = 2900 :=
by
  sorry

end hillary_stops_short_of_summit_l181_181974


namespace arithmetic_seq_a7_constant_l181_181556

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) : Prop :=
∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given arithmetic sequence {a_n}
variable (a : ℕ → α)
-- Given the property that a_2 + a_4 + a_{15} is a constant
variable (C : α)
variable (h : is_arithmetic_seq a)
variable (h_constant : a 2 + a 4 + a 15 = C)

-- Prove that a_7 is a constant
theorem arithmetic_seq_a7_constant (h : is_arithmetic_seq a) (h_constant : a 2 + a 4 + a 15 = C) : ∃ k : α, a 7 = k :=
by
  sorry

end arithmetic_seq_a7_constant_l181_181556


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181892
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181892


namespace not_representative_l181_181447

variable (A B : Prop)

axiom has_email : A
axiom uses_internet : B
axiom dependent : A → B
axiom sample_size : Nat := 2000

theorem not_representative (hA : has_email) (hB : uses_internet) (hD : dependent) (hS : sample_size = 2000) : ¬(∀ x, A x) :=
  sorry

end not_representative_l181_181447


namespace imaginary_part_z_l181_181146

theorem imaginary_part_z (z : ℂ) (h : (z - complex.i) / (z - 2) = complex.i) : z.im = - 1 / 2 := 
sorry

end imaginary_part_z_l181_181146


namespace determine_S6_l181_181780

-- Definitions and conditions
noncomputable def x : ℝ := sorry
def S (m : ℕ) : ℝ := x^m + 1 / x^m

axiom h : x + 1 / x = 4

-- The main problem statement
theorem determine_S6 : S 6 = 2700 :=
by sorry

end determine_S6_l181_181780


namespace largest_base4_is_largest_l181_181479

theorem largest_base4_is_largest 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (n4 : ℕ)
  (h1 : n1 = 31) (h2 : n2 = 52) (h3 : n3 = 54) (h4 : n4 = 46) :
  n3 = Nat.max (Nat.max n1 n2) (Nat.max n3 n4) :=
by
  sorry

end largest_base4_is_largest_l181_181479


namespace extremum_condition_l181_181581

open Real

-- Define the function f(x) = x^2 (log x - a)
def f (x a : ℝ) : ℝ := x^2 * (log x - a)

-- Define the derivative f'(x)
def f' (x a : ℝ) : ℝ := 2 * x * (log x - a) + x^2 / x

-- Problem statement
theorem extremum_condition (a x1 x2 : ℝ) (h_deriv : f' x1 a = f' x2 a ∧ x1 < x2) (hex : x1 + x2 = e) : 2 < x1 + x2 ∧ x1 + x2 < e :=
by
  sorry

end extremum_condition_l181_181581


namespace anika_age_l181_181068

/-- Given:
 1. Anika is 10 years younger than Clara.
 2. Clara is 5 years older than Ben.
 3. Ben is 20 years old.
 Prove:
 Anika's age is 15 years.
 -/
theorem anika_age (Clara Anika Ben : ℕ) 
  (h1 : Anika = Clara - 10) 
  (h2 : Clara = Ben + 5) 
  (h3 : Ben = 20) : Anika = 15 := 
by
  sorry

end anika_age_l181_181068


namespace correct_operation_l181_181403

theorem correct_operation (h1 : ¬ (sqrt 9 = 3 ∨ sqrt 9 = -3))
                         (h2 : (-2) ^ 3 = -8)
                         (h3 : - (abs (-3)) = -3)
                         (h4 : -(2 ^ 2) = -4) : 
                         ∃ operation, operation = D :=
by
  sorry

end correct_operation_l181_181403


namespace cos_75_degree_l181_181501

theorem cos_75_degree (cos : ℝ → ℝ) (sin : ℝ → ℝ) :
    cos 75 = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_l181_181501


namespace distribute_books_l181_181856

theorem distribute_books : 
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5
  total_ways - subtract_one_student_none + add_two_students_none = 240 :=
by
  -- Definitions based on conditions in a)
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5

  -- The final calculation
  have h : total_ways - subtract_one_student_none + add_two_students_none = 240 := by sorry
  exact h

end distribute_books_l181_181856


namespace base_h_addition_eq_l181_181536

theorem base_h_addition_eq (h : ℕ) (h_eq : h = 9) : 
  (8 * h^3 + 3 * h^2 + 7 * h + 4) + (6 * h^3 + 9 * h^2 + 2 * h + 5) = 1 * h^4 + 5 * h^3 + 3 * h^2 + 0 * h + 9 :=
by
  rw [h_eq]
  sorry

end base_h_addition_eq_l181_181536


namespace log_6_14_eq_l181_181126

-- Given conditions
variables (a b : ℝ)
axiom log7_3_eq_a : Real.logBase 7 3 = a
axiom pow7_b_eq_2 : 7^b = 2

-- Statement: Express log_6 14 in terms of a and b
theorem log_6_14_eq : Real.logBase 6 14 = (b + 1) / (a + b) :=
by
  -- Lean has no built-in logBase function, so create it to proceed
  let logBase (b x : ℝ) := (Real.log x) / (Real.log b)
  
  -- Assuming conditions
  have h1 : logBase 7 3 = a := log7_3_eq_a
  have h2 : 7^b = 2 := pow7_b_eq_2
  
  sorry

end log_6_14_eq_l181_181126


namespace distinct_geometric_progression_roots_l181_181728

theorem distinct_geometric_progression_roots (a r : ℝ) (h1 : r ≠ 0) 
  (h2 : a ≠ 0) (h3 : ∃ k : ℝ, polynomial.eval₂ (algebra_map ℝ ℝ) (a^4 * r^6 * X^4 + j * X^2 + k * X - 405) = 0)
  (h4 : ∀ x ∈ [a, a * r, a * r^2, a * r^3], polynomial.eval₂ (algebra_map ℝ ℝ) x (a^4 * r^6 * X^4 + j * X^2 + k * X - 405) = 0) :
  j = -250 :=
  sorry

end distinct_geometric_progression_roots_l181_181728


namespace percent_of_b_l181_181990

variables (a b c : ℝ)

theorem percent_of_b (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : c = 0.25 * b :=
by sorry

end percent_of_b_l181_181990


namespace sculpture_and_base_height_l181_181838

def height_sculpture_ft : ℕ := 2
def height_sculpture_in : ℕ := 10
def height_base_in : ℕ := 2

def total_height_in (ft : ℕ) (inch1 inch2 : ℕ) : ℕ :=
  (ft * 12) + inch1 + inch2

def total_height_ft (total_in : ℕ) : ℕ :=
  total_in / 12

theorem sculpture_and_base_height :
  total_height_ft (total_height_in height_sculpture_ft height_sculpture_in height_base_in) = 3 :=
by
  sorry

end sculpture_and_base_height_l181_181838


namespace inequality_solution_l181_181870

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l181_181870


namespace speed_of_persons_times_of_encounters_l181_181786

theorem speed_of_persons (d : ℝ) (t : ℝ) (S : ℝ) : (d = 100) → (t = 3) → (S = 4) → 
  let speed_B := (100 / 15) in
  let speed_A := (4 * speed_B) in
  (speed_B = 20 / 3) ∧ (speed_A = 80 / 3) :=
begin
  intros d_eq t_eq S_eq,
  let speed_B := 100 / (S * t + t),
  let speed_A := 4 * speed_B,
  split,
  { exact eq_of_sub_eq_zero (calc
      speed_B = S * 100 / (S * t + t) - 0 : by sorry
             ... = 20 / 3 - 0 : by sorry
  ) },
  { exact eq_of_sub_eq_zero (calc
      4 * speed_B - 80 / 3 = 4 * (100 / (S * t + t)) - 80 / 3 : by sorry) 
             ... = 0 - 80 / 3 : by sorry ) }
end

theorem times_of_encounters (d : ℝ) (speed_A speed_B : ℝ) :
  (speed_B = 20 / 3) → (speed_A = 80 / 3) →
  let meeting_times := [3, 5, 9, 15] in
  (meeting_times = [3, 5, 9, 15]) :=
begin
  intros speed_B_eq speed_A_eq,
  let meeting_times := [3, 5, 9, 15],
  exact eq_of_sub_eq_zero (calc
    meeting_times - [3, 5, 9, 15]
      = [] - [] : by sorry
  )
end

end speed_of_persons_times_of_encounters_l181_181786


namespace express_y_l181_181860

theorem express_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 :=
by {
  sorry
}

end express_y_l181_181860


namespace balance_proof_l181_181121

variables (a b c : ℝ)

theorem balance_proof (h1 : 4 * a + 2 * b = 12 * c) (h2 : 2 * a = b + 3 * c) : 3 * b = 4.5 * c :=
sorry

end balance_proof_l181_181121


namespace outfit_combinations_l181_181431

theorem outfit_combinations (tshirts pants hats : ℕ) (h_tshirts : tshirts = 8) (h_pants : pants = 6) (h_hats : hats = 3) : 
  tshirts * pants * hats = 144 :=
by
  rw [h_tshirts, h_pants, h_hats]
  exact (8 * 6 * 3).symm
  sorry -- conclude the proof

end outfit_combinations_l181_181431


namespace find_a_for_even_function_l181_181156

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l181_181156


namespace triangle_ratio_b_c_l181_181225

theorem triangle_ratio_b_c (A B C a b c : ℝ)
  (hA : A = π / 3)
  (hSin : sin (B + C) = 6 * cos B * sin C) :
  b / c = (1 + Real.sqrt 21) / 2 :=
sorry

end triangle_ratio_b_c_l181_181225


namespace snarks_are_twerks_and_quarks_l181_181232

theorem snarks_are_twerks_and_quarks
  (Snarks Garbles Twerks Quarks : Type)
  (h1 : Snarks ⊆ Garbles)
  (h2 : Twerks ⊆ Garbles)
  (h3 : Snarks ⊆ Quarks)
  (h4 : Quarks ⊆ Twerks) :
  Snarks ⊆ Twerks ∧ Snarks ⊆ Quarks :=
by
  sorry

end snarks_are_twerks_and_quarks_l181_181232


namespace problem1_problem2_l181_181495

noncomputable theory

-- Problem 1
theorem problem1 : 3 * Real.sqrt 7 - 3 * (2 + Real.sqrt 7) + 4 = -2 :=
by
  sorry

-- Problem 2
theorem problem2 : -(Real.cbrt (-2)^3) / Real.sqrt (49 / 16) + Real.sqrt ((-1)^2) = 15 / 7 :=
by
  sorry

end problem1_problem2_l181_181495


namespace total_number_of_tiles_l181_181056

theorem total_number_of_tiles {s : ℕ} 
  (h1 : ∃ s : ℕ, (s^2 - 4*s + 896 = 0))
  (h2 : 225 = 2*s - 1 + s^2 / 4 - s / 2) :
  s^2 = 1024 := by
  sorry

end total_number_of_tiles_l181_181056


namespace problem_solution_l181_181134
-- Importing the Mathlib library to access geometric definitions and theorems

-- Given conditions and definitions for the right triangular prism and the sphere intersection.
variables 
  (A B C A1 B1 C1 T1 L1 S : ℝ)  -- Points in 3D space represented as real numbers
  (AC : ℝ)                     -- Given side length AC
  (AL1 : ℝ := 7)               -- Given side length AL1
  (ST1 : ℝ := 2)               -- Given length ST1

-- Assuming the right-triangular nature and defining volumes and ratios
axiom prism_geometry :
  (S T1 A1 : Type) →
  (right_triangle : A B C A1 B1 C1) → 
  (sphere_diameter : A1 B1) → 
  (intersect_points : sphere_diameter ∩ (A1 C1) = T1) →
  (intersect_points : sphere_diameter ∩ (B1 C1) = L1) →
  (intersection : B T1 ∩ A L1 = S)

--  Lean statement proving the required angle, ratio, and volume of the prism.
theorem problem_solution :
  prism_geometry →
  (A1 T1 ⊥ T1 C1) →  -- Orthogonality in volume calculations
  (T1 L1 || A B) →        -- Parallelism for similar triangles
  (T1 L1 = A1 B1) →       -- Length equality conditions
  (volume_of_prism = 35 * sqrt(3)) :=
  by { sorry }

end problem_solution_l181_181134


namespace inequality_part1_l181_181415

theorem inequality_part1 (n : ℕ) (hn : n > 2) :
  3 - (2 / (n-1)!) < (∑ i in Finset.range (n - 1), (i^2 + 3*i + 4) / (i+2)!) < 3 :=
sorry

end inequality_part1_l181_181415


namespace range_of_a_l181_181277

open Set

variable {a x : ℝ}

def A (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem range_of_a (h : A a ∩ B = ∅) : a ≤ 0 ∨ a ≥ 6 := 
by 
  sorry

end range_of_a_l181_181277


namespace hyperbola_asymptotes_l181_181966

variable (a b x y : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (hyperbola_eq : x^2 / a^2 - y^2 / b^2 = 1)
variable (eccentricity_eq : b^2 / a^2 = 3)

theorem hyperbola_asymptotes : ∀ x y, a > 0 → b > 0 → x^2 / a^2 - y^2 / b^2 = 1 → b^2 / a^2 = 3 → (sqrt 3 * x + y = 0) ∨ (sqrt 3 * x - y = 0) :=
by 
  intros x y h1 h2 hyperbola_eq eccentricity_eq
  sorry

end hyperbola_asymptotes_l181_181966


namespace tangent_angle_between_lines_in_cube_l181_181670

theorem tangent_angle_between_lines_in_cube :
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let B1 := (1, 0, 1)
  let E := (1, 0, 1 / 2)
  let A1 := (0, 0, 1)
  let C := (1, 1, 0)
  let CA1 := Math.sqrt( (1 - 0)^2 + (1 - 0)^2 + (0 - 1)^2 )
  let F_dist := Math.sqrt( 1 + (1 / 2)^2 ) / 2
  let CF := Math.sqrt( 1 + (3 / 2)^2 ) / 2
  let cos_theta := Math.sqrt(15) / 15
  let sin_theta := Math.sqrt(210) / 15
  let tan_theta := sin_theta / cos_theta
  in tan_theta = Math.sqrt(14) := 
by
  sorry

end tangent_angle_between_lines_in_cube_l181_181670


namespace range_of_k_for_real_roots_l181_181993

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l181_181993


namespace polar_bear_daily_food_l181_181093

-- Definitions based on the conditions
def bucketOfTroutDaily : ℝ := 0.2
def bucketOfSalmonDaily : ℝ := 0.4

-- The proof statement
theorem polar_bear_daily_food : bucketOfTroutDaily + bucketOfSalmonDaily = 0.6 := by
  sorry

end polar_bear_daily_food_l181_181093


namespace positive_number_l181_181380

theorem positive_number (x : ℝ) (h1 : 0 < x) (h2 : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := sorry

end positive_number_l181_181380


namespace tickets_needed_l181_181254

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l181_181254


namespace seq_sum_l181_181643

def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := a (n+1) + a n

theorem seq_sum : (∑ n, a n / 4^(n+1)) = 1/11 :=
by
  sorry

end seq_sum_l181_181643


namespace min_cut_length_no_triangle_l181_181063

theorem min_cut_length_no_triangle (a b c x : ℝ) 
  (h_y : a = 7) 
  (h_z : b = 24) 
  (h_w : c = 25) 
  (h1 : a - x > 0)
  (h2 : b - x > 0)
  (h3 : c - x > 0)
  (h4 : (a - x) + (b - x) ≤ (c - x)) :
  x = 6 :=
by
  sorry

end min_cut_length_no_triangle_l181_181063


namespace num_true_propositions_l181_181969

-- Define the original proposition
def original_proposition (f : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0 ∨ x <= 0) -- This represents that the graph does not pass through the fourth quadrant

-- Define the converse of the original proposition
def converse_proposition (f : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0 ∨ x <= 0) → (∃ n : ℝ, f = λ x, x^n)

-- Define the inverse of the original proposition
def inverse_proposition (f : ℝ → ℝ) : Prop :=
  (∃ n : ℝ, f ≠ λ x, x^n) → (∃ x, f x < 0 ∧ x > 0)

-- Define the contrapositive of the original proposition
def contrapositive_proposition (f : ℝ → ℝ) : Prop :=
  (∃ x, f x < 0 ∧ x > 0) → (∃ n : ℝ, f ≠ λ x, x^n)

theorem num_true_propositions (f : ℝ → ℝ) (h : original_proposition f) : 
  -- List of true propositions: contrapositive_proposition
  1 = (if contrapositive_proposition f then 1 else 0) +
      (if converse_proposition f then 1 else 0) +
      (if inverse_proposition f then 1 else 0) :=
by sorry

end num_true_propositions_l181_181969


namespace original_price_of_silk_blanket_l181_181048

theorem original_price_of_silk_blanket (
  (cotton_blankets : Nat) (woolen_blankets : Nat) (silk_blankets : Nat) 
  (price_cotton : ℝ) (price_woolen : ℝ) (discount_cotton : ℝ) (discount_woolen : ℝ)
  (average_price : ℝ) (total_blankets : Nat) (total_spent : ℝ)
) : 
  cotton_blankets = 4 →
  woolen_blankets = 3 →
  price_cotton = 100 →
  price_woolen = 150 →
  discount_cotton = 0.10 →
  discount_woolen = 0.05 →
  average_price = 130 →
  total_blankets = 9 →
  total_spent = average_price * total_blankets →
  let total_cotton_after_discount := price_cotton * cotton_blankets * (1 - discount_cotton),
      total_woolen_after_discount := price_woolen * woolen_blankets * (1 - discount_woolen),
      total_silk := total_spent - total_cotton_after_discount - total_woolen_after_discount,
      price_silk := total_silk / silk_blankets
  in price_silk = 191.25 :=
sorry

end original_price_of_silk_blanket_l181_181048


namespace symmetric_periodic_l181_181149

theorem symmetric_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, f (a - x) = f (a + x))
  (h3 : ∀ x : ℝ, f (b - x) = f (b + x)) :
  ∀ x : ℝ, f x = f (x + 2 * (b - a)) :=
by
  sorry

end symmetric_periodic_l181_181149


namespace martin_probability_360_feet_l181_181287

noncomputable def probability_walking_distance_within_360_feet : ℚ :=
  let total_gates := 15
  let distance_between_gates := 90
  let max_distance := 360
  let total_possible_changes := total_gates * (total_gates - 1)

  let feasible_choices_per_gate :=
    [4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4].sum

  (feasible_choices_per_gate : ℚ) / total_possible_changes

theorem martin_probability_360_feet : probability_walking_distance_within_360_feet = 59 / 105 :=
by
  sorry

end martin_probability_360_feet_l181_181287


namespace two_digit_number_count_four_digit_number_count_l181_181782

-- Defining the set of digits
def digits : Finset ℕ := {1, 2, 3, 4}

-- Problem 1 condition and question
def two_digit_count := Nat.choose 4 2 * 2

-- Problem 2 condition and question
def four_digit_count := Nat.choose 4 4 * 24

-- Theorem statement for Problem 1
theorem two_digit_number_count : two_digit_count = 12 :=
sorry

-- Theorem statement for Problem 2
theorem four_digit_number_count : four_digit_count = 24 :=
sorry

end two_digit_number_count_four_digit_number_count_l181_181782


namespace max_trig_sum_product_l181_181530

theorem max_trig_sum_product (x y z : ℝ) :
  (\sin (2 * x) + \sin y + \sin (3 * z)) * (\cos (2 * x) + \cos y + \cos (3 * z)) ≤ 9 / 2 := 
  sorry

end max_trig_sum_product_l181_181530


namespace power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l181_181681

-- Definitions and conditions
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

def is_sum_of_two_or_more_consecutive_naturals (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = (k * a) + (k * (k - 1)) / 2

-- Proofs to be stated
theorem power_of_2_not_sum_of_consecutive (n : ℕ) (h : is_power_of_2 n) : ¬ is_sum_of_two_or_more_consecutive_naturals n :=
by
    sorry

theorem not_power_of_2_is_sum_of_consecutive (M : ℕ) (h : ¬ is_power_of_2 M) : is_sum_of_two_or_more_consecutive_naturals M :=
by
    sorry

end power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l181_181681


namespace quadrant_of_halved_angle_l181_181592

theorem quadrant_of_halved_angle (k : ℤ) : 
  let α := 150 + k * 360 in 
    let β := α / 2 in 
      (β = 75 + k * 180) → ((0 ≤ β % 360 ∧ β % 360 < 90) ∨ (180 ≤ β % 360 ∧ β % 360 < 270)) :=
sorry

end quadrant_of_halved_angle_l181_181592


namespace ratio_of_m_l181_181639

theorem ratio_of_m (a b m m1 m2 : ℝ)
  (h1 : a * m^2 + b * m + c = 0)
  (h2 : (a / b + b / a) = 3 / 7)
  (h3 : a + b = (3 * m - 2) / m)
  (h4 : a * b = 7 / m)
  (h5 : (a + b)^2 = ab / (m * (7/ m)) - 2) :
  (m1 + m2 = 21) ∧ (m1 * m2 = 4) → 
  (m1/m2 + m2/m1 = 108.25) := sorry

end ratio_of_m_l181_181639


namespace exists_composite_carmichael_number_l181_181323

theorem exists_composite_carmichael_number :
  ∃ n : ℕ, (1 < n ∧ ¬Prime n) ∧ ∀ a : ℤ, (a^n ≡ a [ZMOD n]) :=
by
  use 561
  split
  sorry
  sorry

end exists_composite_carmichael_number_l181_181323


namespace problem1_problem2_l181_181837

-- Problem 1
theorem problem1 : sqrt ((-5) ^ 2) - cbrt (-64) + abs (1 - sqrt 2) = 8 + sqrt 2 :=
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 3 * (x - 1) ^ 2 - 75 = 0) : x = 6 ∨ x = -4 :=
  sorry

end problem1_problem2_l181_181837


namespace circle_center_radius_sum_l181_181261

def find_circle_center_radius_sum (x y : ℝ) : ℝ :=
  let a := -4
  let b := 2
  let r := 3 * Real.sqrt 3
  a + b + r

theorem circle_center_radius_sum (x y : ℝ) (h : x^2 + 8 * x - 4 * y = - y^2 + 2 * y - 7) :
  find_circle_center_radius_sum x y = -2 + 3 * Real.sqrt 3 :=
by {
  sorry
}

end circle_center_radius_sum_l181_181261


namespace sample_not_representative_l181_181436

-- Definitions
def has_email_address (person : Type) : Prop := sorry
def uses_internet (person : Type) : Prop := sorry

-- Problem statement: prove that the sample is not representative of the urban population.
theorem sample_not_representative (person : Type)
  (sample : set person)
  (h_sample_size : set.size sample = 2000)
  (A : person → Prop)
  (A_def : ∀ p, A p ↔ has_email_address p)
  (B : person → Prop)
  (B_def : ∀ p, B p ↔ uses_internet p)
  (dependent : ∀ p, A p → B p)
  : ¬ is_representative_sample sample := 
sorry

end sample_not_representative_l181_181436


namespace tiger_distance_correct_l181_181471

-- Define the parameters as constants
constant escape_time : ℕ := 1 -- 1 AM
constant notice_time : ℕ := 4 -- 4 AM
constant initial_speed : ℕ := 25 -- 25 mph
constant slow_speed : ℕ := 10 -- 10 mph
constant chase_speed : ℕ := 50 -- 50 mph
constant chase_duration : ℕ := 1 / 2 -- 0.5 hours
constant slow_duration : ℕ := 2 -- 2 hours
constant initial_duration : ℕ := notice_time - escape_time -- 4 AM - 1 AM

-- Define distances covered in different segments
def initial_distance := initial_speed * initial_duration
def slow_distance := slow_speed * slow_duration
def chase_distance := chase_speed * chase_duration

-- Define total distance
def total_distance := initial_distance + slow_distance + chase_distance

-- Main statement to verify the total distance from the zoo when the tiger is caught.
theorem tiger_distance_correct : total_distance = 120 := by
  sorry

end tiger_distance_correct_l181_181471


namespace triangle_ratio_problem_l181_181028

theorem triangle_ratio_problem
  (ABC : Type)
  [IsTriangle ABC]
  {A B C D E F : ABC}
  (h1 : ∃ (D : ABC), D ∈ line_segment B C ∧ (BD / DC) = 2 / 3)
  (h2 : ∃ (E : ABC), E ∈ line_segment A C ∧ (AE / EC) = 3 / 4) :
  (AF / FD) * (BF / FE) = 35 / 12 :=
sorry

end triangle_ratio_problem_l181_181028


namespace integral_equals_result_l181_181096

noncomputable def integral_value : ℝ :=
  ∫ x in 1.0..2.0, (x^2 + 1) / x

theorem integral_equals_result :
  integral_value = (3 / 2) + Real.log 2 := 
by
  sorry

end integral_equals_result_l181_181096


namespace circle_area_proof_l181_181672

open Real EuclideanGeometry -- Open necessary modules for real numbers and Euclidean Geometry

noncomputable def point : Type := ℝ × ℝ

def A : point := (4, 16)
def B : point := (10, 14)
def intersect_x_axis := (3, 0)

def circle_area (radius : ℝ) : ℝ := π * r^2 -- Define the area of the circle function

theorem circle_area_proof:
  let ω_center := (x, y) -- The center of circle ω
  let radius := dist ω_center A
  tangent_line A ω_center
  tangent_line B ω_center
  intersect_line_tangent ω_center A (3, 0) in 
  circle_area radius = 10.4 * π := 
sorry -- Proof omitted

end circle_area_proof_l181_181672


namespace roots_sum_l181_181113

def equation (x : ℝ) : Prop :=
  (1/x) + (1/(x + 4)) - (1/(x + 8)) - (1/(x + 12)) - (1/(x + 16)) - (1/(x + 20)) + (1/(x + 24)) + (1/(x + 28)) = 0

theorem roots_sum (a b c d : ℕ) (h : ∀ x : ℝ, equation x → ∃ (g : ℝ), x = -a + g ∨ x = -a - g ∧ g^2 = b + c * real.sqrt d ) :
  a + b + c + d = 123 :=
sorry

end roots_sum_l181_181113


namespace opens_door_on_third_attempt_l181_181800

def probability_opens_door_on_third_attempt (keys : List ℕ) (correct_key : ℕ) : ℕ → ℝ :=
  sorry

noncomputable def solution : ℝ :=
  0.2

theorem opens_door_on_third_attempt :
  ∀ (keys : List ℕ) (correct_key : ℕ), 
    (keys.length = 5) →
    (List.mem correct_key keys) →
    (probability_opens_door_on_third_attempt keys correct_key 3 = solution) :=
by
  intros keys correct_key hlength hmem
  sorry

end opens_door_on_third_attempt_l181_181800


namespace gift_box_spinning_tops_l181_181682

theorem gift_box_spinning_tops
  (red_box_cost : ℕ) (red_box_tops : ℕ)
  (yellow_box_cost : ℕ) (yellow_box_tops : ℕ)
  (total_spent : ℕ) (total_boxes : ℕ)
  (h_red_box_cost : red_box_cost = 5)
  (h_red_box_tops : red_box_tops = 3)
  (h_yellow_box_cost : yellow_box_cost = 9)
  (h_yellow_box_tops : yellow_box_tops = 5)
  (h_total_spent : total_spent = 600)
  (h_total_boxes : total_boxes = 72) :
  ∃ (red_boxes : ℕ) (yellow_boxes : ℕ), (red_boxes + yellow_boxes = total_boxes) ∧
  (red_box_cost * red_boxes + yellow_box_cost * yellow_boxes = total_spent) ∧
  (red_box_tops * red_boxes + yellow_box_tops * yellow_boxes = 336) :=
by
  sorry

end gift_box_spinning_tops_l181_181682


namespace sum_of_valid_k_equals_26_l181_181890

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181890


namespace definite_integral_evaluation_l181_181365

noncomputable def integral_value : ℝ :=
  ∫ x in 0..π, abs (sin x - cos x)

theorem definite_integral_evaluation : integral_value = 2 * real.sqrt 2 :=
by
  sorry

end definite_integral_evaluation_l181_181365


namespace ray_total_grocery_bill_l181_181312

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l181_181312


namespace number_of_license_plates_l181_181328

/--
Suppose license plates are formed with six letters using only the letters in the Engan alphabet.
The Engan alphabet contains 15 letters: {A, B, C, D, E, F, G, H, I, J, K, L, M, N, O}.
How many license plates of six letters are possible that begin with either A or B, end with O,
cannot contain the letter I, and have no letters that repeat.
-/
theorem number_of_license_plates :
  let eng_alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'],
      available_letters := List.erase (List.erase eng_alphabet 'I') 'O' in
  ∑ A B ∈ ['A', 'B'], A ≠ B →
  ∑ D ∈ available_letters, D ≠ A → D ≠ 'O' →
  ∑ E ∈ List.erase available_letters A, E ≠ A → E ≠ D →
  ∑ F ∈ List.erase (List.erase available_letters D) A, F ≠ A → F ≠ D → F ≠ E →
  ∑ G ∈ List.erase (List.erase (List.erase available_letters E) D) A, G ≠ A → G ≠ D → G ≠ E → G ≠ F →
  (2 * 13 * 12 * 11 * 10 = 34320) :=
sorry

end number_of_license_plates_l181_181328


namespace three_legged_reptiles_count_l181_181659

noncomputable def total_heads : ℕ := 300
noncomputable def total_legs : ℕ := 798

def number_of_three_legged_reptiles (b r m : ℕ) : Prop :=
  b + r + m = total_heads ∧
  2 * b + 3 * r + 4 * m = total_legs

theorem three_legged_reptiles_count (b r m : ℕ) (h : number_of_three_legged_reptiles b r m) :
  r = 102 :=
sorry

end three_legged_reptiles_count_l181_181659


namespace total_plates_l181_181791

-- define the variables for the number of plates
def plates_lobster_rolls : Nat := 25
def plates_spicy_hot_noodles : Nat := 14
def plates_seafood_noodles : Nat := 16

-- state the problem as a theorem
theorem total_plates :
  plates_lobster_rolls + plates_spicy_hot_noodles + plates_seafood_noodles = 55 := by
  sorry

end total_plates_l181_181791


namespace decreasing_interval_l181_181923

theorem decreasing_interval (a : ℝ) :
  (∀ x ∈ Icc 1 (2 - a), (5 : ℝ) / (x^2 - 2 * x + 20) ≤ (5 : ℝ) / ((x + ε)^2 - 2 * (x + ε) + 20) for all ε > 0) ↔ 
  (1 / 2 ≤ a ∧ a < 2 / 3) :=
by
-- We will provide the detailed proof here
sorry

end decreasing_interval_l181_181923


namespace range_of_m_l181_181184

theorem range_of_m (m : ℝ) (h1 : ∀ x ∈ set.Icc 0 m, (x^2 - 2*x + 3 ≤ 3)) (h2 : ∀ x ∈ set.Icc 0 m, (x^2 - 2*x + 3 ≥ 2)) : 
  1 ≤ m ∧ m ≤ 2 := 
by 
  sorry

end range_of_m_l181_181184


namespace trig_identity_proof_l181_181082

theorem trig_identity_proof :
  sin (-15 * real.pi / 6) * cos (20 * real.pi / 3) * tan (-7 * real.pi / 6) = real.sqrt 3 / 6 :=
by sorry

end trig_identity_proof_l181_181082


namespace contains_third_order_point_l181_181774
open FiniteField

variables {F_q : Type} [Field F_q] [Fintype F_q]

-- Definition of V as a 2-dimensional vector space over F_q
def V : Type := F_q × F_q

-- Definition of L being a set of lines in all directions
variable (L : Set (Set V))

-- Definition of the number of lines through a point
def order_of_point (p : V) : ℕ := { l ∈ L | p ∈ l }.card

-- The main theorem statement
theorem contains_third_order_point {q : ℕ} (hq : ringChar F_q ≠ 2) (hq_pos : 2 < q)
  (hL : ∀ (m b : F_q), ∃ l ∈ L, ∀ x : F_q, (x, m*x + b) ∈ l ∧
    ∀ a : F_q, ∃ l ∈ L, ∀ y : F_q, (a, y) ∈ l) :
  ∃ l ∈ L, ∃ p ∈ l, order_of_point L p ≥ 3 :=
sorry

end contains_third_order_point_l181_181774


namespace matchsticks_per_house_l181_181289

theorem matchsticks_per_house
  (original_matchsticks : ℕ)
  (used_fraction : ℚ)
  (number_of_houses : ℕ)
  (h_original : original_matchsticks = 600)
  (h_used_fraction : used_fraction = 1/2)
  (h_number_of_houses : number_of_houses = 30) :
  (original_matchsticks * used_fraction) / number_of_houses = 10 :=
by
  -- Given the conditions
  have h_used_matchsticks : original_matchsticks * used_fraction = 300, from
    by rw [h_original, h_used_fraction]; norm_num,
  -- Therefore the matchsticks per house
  have h_per_house : (300 : ℕ) / number_of_houses = 10, from
    by rw [h_number_of_houses]; norm_num,
  -- Completing the proof
  rw h_used_matchsticks,
  exact h_per_house

end matchsticks_per_house_l181_181289


namespace first_three_digits_of_expression_l181_181492

theorem first_three_digits_of_expression :
  let n := (2007 : ℤ)
  let exponent := (12 / 11 : ℝ)
  let number := (10 ^ n + 1 : ℝ)
  let result := number ^ exponent
  decimal.first_three_digits_right_of_decimal result = 909 :=
sorry

end first_three_digits_of_expression_l181_181492


namespace polynomial_roots_sum_l181_181600

theorem polynomial_roots_sum (a b c : ℂ) (x1 x2 x3 : ℂ) (h1 : x1 = 1) (h2 : x2 = 1 - complex.I) (h3 : x3 = 1 + complex.I)
(h4 : x1 + x2 + x3 = -a)
(h5 : x1 * x2 + x2 * x3 + x3 * x1 = b)
(h6 : x1 * x2 * x3 = -c) : (a + b - c) = 3 := 
sorry

end polynomial_roots_sum_l181_181600


namespace fraction_of_painted_surface_area_l181_181806

def total_surface_area_of_smaller_prisms : ℕ := 
  let num_smaller_prisms := 27
  let num_square_faces := num_smaller_prisms * 3
  let num_triangular_faces := num_smaller_prisms * 2
  num_square_faces + num_triangular_faces

def painted_surface_area_of_larger_prism : ℕ :=
  let painted_square_faces := 3 * 9
  let painted_triangular_faces := 2 * 9
  painted_square_faces + painted_triangular_faces

theorem fraction_of_painted_surface_area : 
  (painted_surface_area_of_larger_prism : ℚ) / (total_surface_area_of_smaller_prisms : ℚ) = 1 / 3 :=
by sorry

end fraction_of_painted_surface_area_l181_181806


namespace find_value_of_c_l181_181215

variable (c b : ℝ)
noncomputable def isSolution := (sin (c^2 - 3 * c + 17) * Real.pi / 180 = 4 / (b - 2)) ∧ (0 < c^2 - 3 * c + 17) ∧ (c^2 - 3 * c + 17 < 90) ∧ (c > 0)

theorem find_value_of_c (h : isSolution c b) : c = 7 :=
sorry

end find_value_of_c_l181_181215


namespace deer_distribution_l181_181327

theorem deer_distribution :
  ∃ a : ℕ → ℚ,
    (a 1 + a 2 + a 3 + a 4 + a 5 = 5) ∧
    (a 4 = 2 / 3) ∧ 
    (a 3 = 1) ∧ 
    (a 1 = 5 / 3) :=
by
  sorry

end deer_distribution_l181_181327


namespace sin_pi_minus_alpha_l181_181570

variable (x y : ℝ)
variable (r : ℝ)

def point_on_terminal_side_of_angle (x y: ℝ) (r: ℝ) :=
  x = -4 ∧ y = 3 ∧ r = Real.sqrt ((-4)^2 + 3^2)

theorem sin_pi_minus_alpha 
  (h : point_on_terminal_side_of_angle x y r):
  sin (π - (Real.arcsin (y / r))) = 3 / 5 :=
by
  sorry

end sin_pi_minus_alpha_l181_181570


namespace O_on_MN_l181_181606

-- Define Points and Triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

structure Segment where
  P Q : Point

-- Define midpoint function
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Condition definitions
variable (A B C : Point) (D E : Point)

def angle_A_eq_60 (T : Triangle) : Prop :=
  sorry -- This would involve trigonometric checks to ensure ∠A = 60°.

def midpoint_M (B C : Point) : Point :=
  midpoint B C

def M_eq_midpoint (M : Point) : Prop :=
  M = midpoint_M B C

def angle_MNB_eq_30 (M N B : Point) : Prop :=
  sorry -- This would involve trigonometric checks to ensure ∠MNB = 30°.

def midpoint_F (B E : Point) : Point :=
  midpoint B E

def midpoint_G (C D : Point) : Point :=
  midpoint C D

def midpoint_H (D E : Point) : Point :=
  midpoint D E

def circumcenter (F G H : Point) : Point :=
  sorry -- True circumcenter computation.

def point_O_is_circumcenter (O F G H : Point) : Prop :=
  O = circumcenter F G H

def O_lies_on_MN (O M N : Point) : Prop :=
  sorry -- This would be a geometric check for collinearity.

-- The main theorem
theorem O_on_MN
  (T : Triangle)
  (D E : Point)
  (F : Point := midpoint_F B E)
  (G : Point := midpoint_G C D)
  (H : Point := midpoint_H D E)
  (O : Point := circumcenter F G H)
  (M : Point := midpoint_M B C)
  (N : Point) :
  angle_A_eq_60 T →
  M_eq_midpoint M →
  angle_MNB_eq_30 M N B →
  point_O_is_circumcenter O F G H →
  O_lies_on_MN O M N :=
sorry

end O_on_MN_l181_181606


namespace uniqueness_point_S_point_T_properties_l181_181971

-- Define the sequence of points such that for n >= 3, every A_n is the centroid of the triangle (A_(n-3), A_(n-2), A_(n-1)).
noncomputable def centroid (p1 p2 p3 : Point) : Point :=
  (p1 + p2 + p3) / 3

-- Given conditions as per the problem statement
variable (A1 A2 A3 : Point) (h_noncollinear : ¬Collinear ℝ {A1, A2, A3})

-- Define the sequence {Ai}
noncomputable def A (i : ℕ) : Point :=
  if i = 0 then A1
  else if i = 1 then A2
  else if i = 2 then A3
  else centroid (A (i - 3)) (A (i - 2)) (A (i - 1))

-- Proving uniqueness of point S
theorem uniqueness_point_S : ∃! S : Point, (∀ n ≥ 3, S ∈ triangle (A (n-3)) (A (n-2)) (A (n-1))) :=
begin
  sorry
end

-- Finding the ratios
variable (S : Point) (T : Point)
variable (h_S : ∀ n ≥ 3, S ∈ triangle (A (n-3)) (A (n-2)) (A (n-1)))
variable (h_T : T ∈ line_SA3 ∩ line_SA1A2)

theorem point_T_properties : (A1, A2, A3 : Point) (S : Point) (T : Point)
  (line_SA3 = line S A3)
  (line_SA1A2 = line S A3 ∩ line A1 A2) :
  (dist A1 T / dist T A2 = 2) ∧ (dist T S / dist S A3 = 1) :=
begin
  sorry
end

end uniqueness_point_S_point_T_properties_l181_181971


namespace percent_increase_between_maintenance_checks_l181_181406

theorem percent_increase_between_maintenance_checks (original_time new_time : ℕ) (h_orig : original_time = 50) (h_new : new_time = 60) :
  ((new_time - original_time : ℚ) / original_time) * 100 = 20 := by
  sorry

end percent_increase_between_maintenance_checks_l181_181406


namespace Donny_change_l181_181510

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l181_181510


namespace max_trig_sum_product_l181_181531

theorem max_trig_sum_product (x y z : ℝ) :
  (\sin (2 * x) + \sin y + \sin (3 * z)) * (\cos (2 * x) + \cos y + \cos (3 * z)) ≤ 9 / 2 := 
  sorry

end max_trig_sum_product_l181_181531


namespace solution_set_l181_181784

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set (c1 : ∀ x : ℝ, f x + f' x > 1)
                     (c2 : f 0 = 2) :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | 0 < x} :=
sorry

end solution_set_l181_181784


namespace probability_of_c_l181_181795

theorem probability_of_c (
  P_A : ℚ,
  P_B : ℚ,
  P_D : ℚ,
  P_total : P_A + P_B + P_C + P_D = 1
) : P_C = 1/4 :=
by
  assume h1 : P_A = 1/4,
  assume h2 : P_B = 1/3,
  assume h3 : P_D = 1/6,
  sorry

end probability_of_c_l181_181795


namespace hot_drink_sales_l181_181058

theorem hot_drink_sales (x y : ℝ) (h : y = -2.35 * x + 147.7) (hx : x = 2) : y = 143 := 
by sorry

end hot_drink_sales_l181_181058


namespace replace_all_cardio_machines_cost_l181_181796

noncomputable def totalReplacementCost : ℕ :=
  let numGyms := 20
  let bikesPerGym := 10
  let treadmillsPerGym := 5
  let ellipticalsPerGym := 5
  let costPerBike := 700
  let costPerTreadmill := costPerBike * 3 / 2
  let costPerElliptical := costPerTreadmill * 2
  let totalBikes := numGyms * bikesPerGym
  let totalTreadmills := numGyms * treadmillsPerGym
  let totalEllipticals := numGyms * ellipticalsPerGym
  (totalBikes * costPerBike) + (totalTreadmills * costPerTreadmill) + (totalEllipticals * costPerElliptical)

theorem replace_all_cardio_machines_cost :
  totalReplacementCost = 455000 :=
by
  -- All the calculation steps provided as conditions and intermediary results need to be verified here.
  sorry

end replace_all_cardio_machines_cost_l181_181796


namespace oil_price_reduction_l181_181425

theorem oil_price_reduction (P P_reduced : ℝ) (h1 : P_reduced = 50) (h2 : 1000 / P_reduced - 5 = 5) :
  ((P - P_reduced) / P) * 100 = 25 := by
  sorry

end oil_price_reduction_l181_181425


namespace minimum_phi_l181_181602

noncomputable def initial_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * x + ϕ)

noncomputable def translated_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * (x - (Real.pi / 6)) + ϕ)

theorem minimum_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * Real.pi + 7 * Real.pi / 6) →
  (∃ ϕ_min : ℝ, (ϕ_min = ϕ ∧ ϕ_min = Real.pi / 6)) :=
by
  sorry

end minimum_phi_l181_181602


namespace YoongiHasSevenPets_l181_181768

def YoongiPets (dogs cats : ℕ) : ℕ := dogs + cats

theorem YoongiHasSevenPets : YoongiPets 5 2 = 7 :=
by
  sorry

end YoongiHasSevenPets_l181_181768


namespace num_adult_tickets_l181_181374

theorem num_adult_tickets (adult_ticket_cost child_ticket_cost total_tickets_sold total_receipts : ℕ) 
  (h1 : adult_ticket_cost = 12) 
  (h2 : child_ticket_cost = 4) 
  (h3 : total_tickets_sold = 130) 
  (h4 : total_receipts = 840) :
  ∃ A C : ℕ, A + C = total_tickets_sold ∧ adult_ticket_cost * A + child_ticket_cost * C = total_receipts ∧ A = 40 :=
by {
  sorry
}

end num_adult_tickets_l181_181374


namespace concave_probability_l181_181543

def is_concave (a : Fin 5 → Fin 5) : Prop :=
  a 0 > a 1 ∧ a 1 > a 2 ∧ a 2 < a 3 ∧ a 3 < a 4

def five_digit_numbers := {a : Fin 5 → Fin 5 | ∀ i, a i ∈ ({0, 1, 2, 3, 4} : Set (Fin 5))}

def concave_numbers := {a ∈ five_digit_numbers | is_concave a}

theorem concave_probability :
  let total := 2500
  let count := 46
  total ≠ 0 →
  (count.toRat / total.toRat) = (23 / 1250 : ℚ) :=
by
  sorry

end concave_probability_l181_181543


namespace compute_result_l181_181644

def f (x : ℕ) : ℕ := 2 * x + 3
def g (x : ℕ) : ℕ := 4 * x + 1

theorem compute_result : f (g 2) - g (f 2) = -8 := by
  sorry

end compute_result_l181_181644


namespace solve_trig_eq_l181_181691

   theorem solve_trig_eq (x y z : ℝ) (m n : ℤ): 
     sin x ≠ 0 → cos y ≠ 0 →
     (sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z →
     (∃ m n : ℤ, x = (π / 2) + π * m ∧ y = π * n ∧ z = 2 * π * m) := 
   by
     intros h1 h2 h_eq
     sorry
   
end solve_trig_eq_l181_181691


namespace area_of_given_region_l181_181106

open Real

def area_under_inequality : ℝ :=
  let region := {p : ℝ × ℝ | |2 * p.1 + 3 * p.2| + |2 * p.1 - 3 * p.2| ≤ 12}
  measure (set.univ.restrict region)

theorem area_of_given_region : area_under_inequality = 12 :=
  sorry

end area_of_given_region_l181_181106


namespace mean_exercise_days_correct_l181_181617

def students_exercise_days : List (Nat × Nat) := 
  [ (2, 0), (4, 1), (5, 2), (7, 3), (5, 4), (3, 5), (1, 6)]

def total_days_exercised : Nat := 
  List.sum (students_exercise_days.map (λ (count, days) => count * days))

def total_students : Nat := 
  List.sum (students_exercise_days.map Prod.fst)

def mean_exercise_days : Float := 
  total_days_exercised.toFloat / total_students.toFloat

theorem mean_exercise_days_correct : Float.round (mean_exercise_days * 100) / 100 = 2.81 :=
by
  sorry -- proof not required

end mean_exercise_days_correct_l181_181617


namespace find_angle_4_l181_181552

theorem find_angle_4 (
  (angle1 : ℝ) (angle2 : ℝ) (angle3 : ℝ) (angle4 : ℝ) (angle5 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 80)
  (h3 : angle1 + angle2 + angle3 = 360)
  (h4 : angle3 = angle4)
  (h5 : angle4 = angle5)
) : angle4 = 60 :=
by
  -- the proof is omitted
  sorry

end find_angle_4_l181_181552


namespace car_speed_l181_181016

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) (h_dist : distance = 642) (h_time : time = 6.5) (h_speed_def : speed = distance / time) : speed = 99 :=
by 
  have h1 : speed = 642 / 6.5, from (by rw [h_dist, h_time]; exact h_speed_def),
  have h2 : speed = 98.76923076923077, from by rwa h1,
  -- We round off 98.76923076923077 to 99
  sorry

end car_speed_l181_181016


namespace student_weighted_avg_larger_l181_181810

variable {u v w : ℚ}

theorem student_weighted_avg_larger (h1 : u < v) (h2 : v < w) :
  (4 * u + 6 * v + 20 * w) / 30 > (2 * u + 3 * v + 4 * w) / 9 := by
  sorry

end student_weighted_avg_larger_l181_181810


namespace minimize_theta_l181_181520

theorem minimize_theta (K : ℤ) : ∃ θ : ℝ, -495 = K * 360 + θ ∧ |θ| ≤ 180 ∧ θ = -135 :=
by
  sorry

end minimize_theta_l181_181520


namespace negation_of_p_l181_181743

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : (¬p) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_p_l181_181743


namespace part_one_part_two_l181_181608

-- Definitions based on the input conditions
variables {α : Type*} [linear_ordered_field α]

-- Given triangle ABC with sides a, b, c opposite to angles A, B, C respectively
def triangle (a b c A B C : α) :=
  2 * b * cos B = a * cos C + c * cos A

-- Part (1): Determine the measure of angle B
theorem part_one (a b c A B C : α) (hb : b * cos B = (a * cos C + c * cos A) / 2) 
  (h : triangle a b c A B C) : 
  B = π / 3 :=
sorry

-- Part (2): Find the value of a + c
theorem part_two (a b c A B C : α) (hb : b = sqrt 3) (area : α) (harea : area = (3 * sqrt 3) / 4) 
  (h : triangle a b c A B C) : 
  a + c = 2 * sqrt 3 :=
sorry

end part_one_part_two_l181_181608


namespace modulus_z_l181_181173

noncomputable def z : ℂ := (5 : ℂ) / (1 - (2 * complex.I))

theorem modulus_z : complex.abs z = real.sqrt 5 := 
by 
  sorry

end modulus_z_l181_181173


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181913

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181913


namespace Ariella_total_amount_l181_181070

-- We define the conditions
def Daniella_initial (daniella_amount : ℝ) := daniella_amount = 400
def Ariella_initial (daniella_amount : ℝ) (ariella_amount : ℝ) := ariella_amount = daniella_amount + 200
def simple_interest_rate : ℝ := 0.10
def investment_period : ℕ := 2

-- We state the goal to prove
theorem Ariella_total_amount (daniella_amount ariella_amount : ℝ) :
  Daniella_initial daniella_amount →
  Ariella_initial daniella_amount ariella_amount →
  ariella_amount + ariella_amount * simple_interest_rate * (investment_period : ℝ) = 720 :=
by
  sorry

end Ariella_total_amount_l181_181070


namespace subset_implies_x_eq_1_l181_181655

theorem subset_implies_x_eq_1 (x : ℤ) :
  let M := {2, 0, x}
  let N := {0, 1}
  N ⊆ M → x = 1 := 
by 
  intros h
  sorry

end subset_implies_x_eq_1_l181_181655


namespace biased_sample_non_representative_l181_181437

/-- 
A proof problem verifying the representativeness of a sample of 2000 email address owners concerning 
the urban population's primary sources of news.
-/
theorem biased_sample_non_representative (
  (U : Type) 
  (email_population : finset U) 
  (sample : finset U) :
  sample.card = 2000 
  ∧ sample ⊆ email_population 
  ∧ ∃ (u : U), u ∈ sample 
  → email_population.card < finset.card email_population
  := sorry

end biased_sample_non_representative_l181_181437


namespace cyclic_quadrilateral_diagonals_l181_181386

theorem cyclic_quadrilateral_diagonals (
  R : ℝ,
  θ1 θ2 θ3 θ4 : ℝ,
  arc1 := 3,
  arc2 := 4,
  arc3 := 5,
  arc4 := 6,
  h_sum_angles : θ1 + θ2 + θ3 + θ4 = 2 * Real.pi,
  h_angles : θ1 = arc1 / R ∧ θ2 = arc2 / R ∧ θ3 = arc3 / R ∧ θ4 = arc4 / R
) : ∀ (a b c d : ℝ),
  a = 2 * R * Real.sin (θ1 / 2) →
  b = 2 * R * Real.sin (θ2 / 2) →
  c = 2 * R * Real.sin (θ3 / 2) →
  d = 2 * R * Real.sin (θ4 / 2) →
  (let diagonal := Real.sqrt ((a * b + c * d) * (a * c + b * d) * (a * d + b * c)) in
  diagonal = 9) :=
sorry

end cyclic_quadrilateral_diagonals_l181_181386


namespace area_rhombus_center_square_l181_181243

theorem area_rhombus_center_square :
  ∀ (A B C D F E G H : Point) (AB_length : ℝ),
  square A B C D →
  (midpoint A B F) →
  (midpoint C D E) →
  (F = midpoint A B) →
  (E = midpoint C D) →
  (AB_length = 4) →
  area (rhombus F G E H) = 4 := 
by
  sorry

end area_rhombus_center_square_l181_181243


namespace substring_012_appears_148_times_l181_181517

noncomputable def count_substring_012_in_base_3_concat (n : ℕ) : ℕ :=
  -- The function that counts the "012" substrings in the concatenated base-3 representations
  sorry

theorem substring_012_appears_148_times :
  count_substring_012_in_base_3_concat 728 = 148 :=
  sorry

end substring_012_appears_148_times_l181_181517


namespace sum_of_valid_k_equals_26_l181_181887

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181887


namespace tetrahedron_vertex_edge_condition_l181_181675

theorem tetrahedron_vertex_edge_condition (a b c d e f : ℝ) (h1 : a ≥ b) (h2 : a ≥ c) (h3 : a ≥ d) 
  (h4 : a ≥ e) (h5 : a ≥ f) :
  ∃ (u v w : ℝ), (u, v, w ∈ {a, b, c, d, e, f}) ∧ (u + v > w) ∧ (u + w > v) ∧ (v + w > u) :=
  sorry

end tetrahedron_vertex_edge_condition_l181_181675


namespace inequality_solution_l181_181875

theorem inequality_solution (x : ℝ) :
  (\frac{x + 1}{x - 2} + \frac{x + 3}{3*x} ≥ 4) ↔ (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
sorry

end inequality_solution_l181_181875


namespace midpoint_on_nine_point_circle_l181_181748

def triangle (A B C : Point) : Prop :=
  ∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def diameter_of_circle (B C : Point) (k : Circle) : Prop :=
  diameter B C k

def intersect_line_at (circle : Circle) (line : Line) (point : Point) : Prop :=
  intersects circle line point

def circumcircle_of_triangle (A E F : Point) (k' : Circle) : Prop :=
  circumcircle A E F k'

theorem midpoint_on_nine_point_circle
  {A B C E F P Q M : Point}
  {k k' : Circle} :
  triangle A B C →
  diameter_of_circle B C k →
  intersect_line_at k (Line.of_points C A) E →
  intersect_line_at k (Line.of_points B A) F →
  circumcircle_of_triangle A E F k' →
  Line.exists_midpoint P Q M →
  M ∈ nine_point_circle (triangle A B C) :=
begin
  intros h_triangle h_diameter h_intersect_CE h_intersect_BA h_circumcircle heqx_M,
  sorry -- Proof goes here.
end

end midpoint_on_nine_point_circle_l181_181748


namespace complex_coordinate_l181_181722

-- Define the complex number i
noncomputable def i : ℂ := complex.I

-- Define the given condition, z as 1 / i^3
noncomputable def z : ℂ := 1 / (i^3)

-- The statement to prove
theorem complex_coordinate : z = i → (0, 1) :=
by 
  sorry

end complex_coordinate_l181_181722


namespace number_of_boys_l181_181424

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end number_of_boys_l181_181424


namespace exponent_of_4_in_g_24_is_11_l181_181018

open Nat

def g (x : ℕ) : ℕ :=
  ∏ (k : ℕ) in (Finset.filter (fun k => even k) (Finset.range x.succ)), k

theorem exponent_of_4_in_g_24_is_11 :
  (g 24).factorization 4 = 11 :=
by
  sorry

end exponent_of_4_in_g_24_is_11_l181_181018


namespace selling_price_is_300_l181_181462

-- Definitions from the conditions:
def purchase_price := 225
def overhead_expenses := 30
def profit_percent := 17.64705882352942 / 100

-- The total cost is the sum of purchase price and overhead expenses.
def total_cost := purchase_price + overhead_expenses

-- The profit is calculated as profit percent times the total cost.
def profit := profit_percent * total_cost

-- The selling price is the sum of the total cost and profit.
def selling_price := total_cost + profit

-- The theorem statement to be proven.
theorem selling_price_is_300 : selling_price = 300 := by sorry

end selling_price_is_300_l181_181462


namespace exists_hamiltonian_path_transitivity_iff_victories_condition_l181_181466

-- Define a structure for a tournament and transitivity
structure Tournament (n : ℕ) :=
(wins : Fin n → Fin n → Bool)

def transitive (t : Tournament n) : Prop :=
  ∀ {i j k : Fin n}, t.wins i j → t.wins j k → t.wins i k

-- Show the existence of a Hamiltonian path
theorem exists_hamiltonian_path {n : ℕ} (t : Tournament n) :
  ∃ (order : Fin n → Fin n), ∀ (i : Fin (n - 1)), t.wins (order i) (order (Fin.succ i)) := sorry

-- Define the condition for a tournament to be transitive based on victories
def victories_condition {n : ℕ} (victories : Fin n → ℕ) : Prop :=
  ∑ i, victories i ^ 2 = (n * (n - 1) * (2 * n - 1)) / 6

theorem transitivity_iff_victories_condition {n : ℕ} (t : Tournament n) (victories : Fin n → ℕ) :
  (transitive t ↔ victories_condition victories) := sorry

end exists_hamiltonian_path_transitivity_iff_victories_condition_l181_181466


namespace sum_of_squares_of_non_zero_digits_from_10_to_99_l181_181263

-- Definition of the sum of squares of digits from 1 to 9
def P : ℕ := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)

-- Definition of the sum of squares of the non-zero digits of the integers from 10 to 99
def T : ℕ := 20 * P

-- Theorem stating that T equals 5700
theorem sum_of_squares_of_non_zero_digits_from_10_to_99 : T = 5700 :=
by
  sorry

end sum_of_squares_of_non_zero_digits_from_10_to_99_l181_181263


namespace average_minutes_per_day_l181_181486

theorem average_minutes_per_day
  (g : ℕ) -- number of fifth graders
  (h1 : ℕ) -- number of fourth graders
  (h2 : ℕ) -- number of sixth graders
  (h1_eq : h1 = 3 * g) -- Fourth graders are three times fifth graders
  (h2_eq : h2 = g) -- Sixth graders are equal to fifth graders
  (average_fourth : ℚ := 18) -- Average minutes ran by fourth graders
  (average_fifth : ℚ := 12) -- Average minutes ran by fifth graders
  (average_sixth : ℚ := 9) -- Average minutes ran by sixth graders
:
  ((h1 * average_fourth + g * average_fifth + h2 * average_sixth) / (h1 + g + h2)) = 15
:=
  sorry

end average_minutes_per_day_l181_181486


namespace arithmetic_sequence_S9_l181_181574

axiom arithmetic_sequence 
  (a : ℕ → ℝ) -- Assume the sequence is of real numbers for generality
  (S : ℕ → ℝ) -- The sum of the first n terms
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 3 + a 7 = 10) : 
  S 9 = 45

theorem arithmetic_sequence_S9 : S 9 = 45 :=
by
  apply arithmetic_sequence,
  -- Proof goes here
  sorry

end arithmetic_sequence_S9_l181_181574


namespace monotonic_intervals_intersection_range_m_l181_181962

/-- Part I: Monotonic Intervals -/
theorem monotonic_intervals (a : ℝ) (h : a < 0) : 
  (if (-1/2 < a) then
    ∃ I1 I2, (I1 = set.Icc (-(2*a + 1) / a) 0 ∧ ∀ x ∈ I1, f' x > 0) ∧
              (I2 = set.Icc 0 (-(2*a + 1) / a) ∧ ∀ x ∈ I2, f' x < 0)
  else if (a = -1/2) then
    ∀ x, f' x ≤ 0
  else
    ∃ I3 I4, (I3 = set.Icc (-(2*a + 1) / a) 0 ∧ ∀ x ∈ I3, f' x < 0) ∧
              (I4 = set.Icc 0 (-(2*a + 1) / a) ∧ ∀ x ∈ I4, f' x > 0)) := sorry

/-- Part II: Intersection and Range of m -/
theorem intersection_range_m (m : ℝ) :
  let f (x : ℝ) := (-x^2 + x - 1) * exp x
  let g (x : ℝ) := (1/3 * x^3 + 1/2 * x^2 + m)
  f (-1) < g (-1) ∧ f 0 > g 0 ↔ -3 / exp 1 - 1/6 < m ∧ m < -1 := sorry

end monotonic_intervals_intersection_range_m_l181_181962


namespace exists_x0_and_in_middle_l181_181964

open Real

def f (a x : ℝ) : ℝ := ln x - a * x^2 + (2 - a) * x

theorem exists_x0_and_in_middle (a x1 x2 : ℝ) (h₀ : a < -1/2) (h₁ : 1 < x1) (h₂ : 1 < x2) (h₃ : x1 < x2) :
  ∃ x0 ∈ Ioo x1 x2, deriv (f a) x0 = (f a x2 - f a x1) / (x2 - x1) ∧ x1 + x2 / 2 < x0 :=
begin
  sorry
end

end exists_x0_and_in_middle_l181_181964


namespace annies_initial_amount_l181_181069

theorem annies_initial_amount :
  let hamburger_cost := 4
  let cheeseburger_cost := 5
  let french_fries_cost := 3
  let milkshake_cost := 5
  let smoothie_cost := 6
  let people_count := 8
  let burger_discount := 1
  let milkshake_discount := 2
  let smoothie_discount_buy2_get1free := 6
  let sales_tax := 0.08
  let tip_rate := 0.15
  let max_single_person_cost := cheeseburger_cost + french_fries_cost + smoothie_cost
  let total_cost := people_count * max_single_person_cost
  let total_burger_discount := people_count * burger_discount
  let total_milkshake_discount := 4 * milkshake_discount
  let total_smoothie_discount := smoothie_discount_buy2_get1free
  let total_discount := total_burger_discount + total_milkshake_discount + total_smoothie_discount
  let discounted_cost := total_cost - total_discount
  let tax_amount := discounted_cost * sales_tax
  let subtotal_with_tax := discounted_cost + tax_amount
  let original_total_cost := people_count * max_single_person_cost
  let tip_amount := original_total_cost * tip_rate
  let final_amount := subtotal_with_tax + tip_amount
  let annie_has_left := 30
  let annies_initial_money := final_amount + annie_has_left
  annies_initial_money = 144 :=
by
  sorry

end annies_initial_amount_l181_181069


namespace abs_div_sq_is_integer_l181_181922

theorem abs_div_sq_is_integer (x : ℝ) (hx : x ≠ 0) : (|x - |x||^2) / x ∈ ℤ :=
by
  sorry

end abs_div_sq_is_integer_l181_181922


namespace inequality_solution_l181_181713

theorem inequality_solution :
  {x : ℝ // -1 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 1} = 
  {x : ℝ // x > 1/6} :=
sorry

end inequality_solution_l181_181713


namespace smallest_k_inequality_l181_181090

theorem smallest_k_inequality :
  ∃ k : ℕ, (∀ a b c ∈ [0, 1], ∀ n : ℕ, a^k * (1 - a)^n < 1 / (n + 1)^3) :=
begin
  use 4,
  intros a ha b hb c hc n hn,
  sorry
end

end smallest_k_inequality_l181_181090


namespace fruit_weights_l181_181664

def weights := {140, 150, 160, 170, 1700}

variables (B P M O K : ℕ)

theorem fruit_weights :
  M = 1700 ∧
  (B + K = P + O) ∧
  (K < P ∧ P < O) ∧
  {B, P, M, O, K} = weights ∧
  B ≠ P ∧ B ≠ M ∧ B ≠ O ∧ B ≠ K ∧
  P ≠ M ∧ P ≠ O ∧ P ≠ K ∧
  M ≠ O ∧ M ≠ K ∧
  O ≠ K
:=
  sorry

end fruit_weights_l181_181664


namespace find_m_l181_181947

theorem find_m (x y m : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x + m + y = 0) : m = -1 := by
  -- Proof can be completed here
  sorry

end find_m_l181_181947


namespace find_a_for_even_function_l181_181155

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l181_181155


namespace running_speed_equiv_l181_181050

variable (R : ℝ)
variable (walking_speed : ℝ) (total_distance : ℝ) (total_time: ℝ) (distance_walked : ℝ) (distance_ran : ℝ)

theorem running_speed_equiv :
  walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4 →
  1 + (4 / R) = 1.5 →
  R = 8 :=
by
  intros H1 H2
  -- H1: Condition set (walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4)
  -- H2: Equation (1 + (4 / R) = 1.5)
  sorry

end running_speed_equiv_l181_181050


namespace inequality_proof_l181_181599

theorem inequality_proof {x y z : ℝ} (n : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1)
  : (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) 
    ≥ (3^n) / (3^(n - 2) - 9) :=
by
  sorry

end inequality_proof_l181_181599


namespace carson_pumps_needed_l181_181497

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end carson_pumps_needed_l181_181497


namespace quadrilateral_min_side_length_l181_181079

theorem quadrilateral_min_side_length (a b c d: ℝ×ℝ) (h1 : a.1 = 0 ∧ 0 ≤ a.2 ∧ a.2 ≤ 1)
(h2 : b.2 = 1 ∧ 0 ≤ b.1 ∧ b.1 ≤ 1) 
(h3 : c.1 = 1 ∧ 0 ≤ c.2 ∧ c.2 ≤ 1) 
(h4 : d.2 = 0 ∧ 0 ≤ d.1 ∧ d.1 ≤ 1) :
∃ (e f: ℝ×ℝ), e ∈ {a, b, c, d} ∧ f ∈ {a, b, c, d} ∧ e ≠ f ∧ dist e f ≥ (Real.sqrt 2) / 2 :=
by sorry

end quadrilateral_min_side_length_l181_181079


namespace hyperbola_foci_coordinates_l181_181339

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), x^2 - (y^2 / 3) = 1 → (∃ c : ℝ, c = 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_coordinates_l181_181339


namespace odd_function_a_value_l181_181991

theorem odd_function_a_value :
  (∃ (a : ℝ), ∀ (x : ℝ), (f : ℝ → ℝ) = (λ x, (2 / (3^x + 1) - a)) ∧ (f (-x) = -f x)) → a = 1 := 
by
  intro ⟨a, hf⟩
  have h0 := hf 0
  rw [hf] at h0
  sorry

end odd_function_a_value_l181_181991


namespace isosceles_triangle_ineq_l181_181997

open Real

theorem isosceles_triangle_ineq :
  ∀ (A B C D E : Point) (AB AC BD BE BC : ℝ),
  is_isosceles_triangle A B C ∧
  foot_perpendicular C A B D ∧
  foot_perpendicular B A C E ∧
  length AB = length AC ∧
  length BC = BC ∧
  length BD = BD ∧
  length BE = BE →
  (BC^3 < BD^3 + BE^3) :=
by
  sorry

end isosceles_triangle_ineq_l181_181997


namespace smallest_int_cond_l181_181396

theorem smallest_int_cond (b : ℕ) :
  (b % 9 = 5) ∧ (b % 11 = 7) → b = 95 :=
by
  intro h
  sorry

end smallest_int_cond_l181_181396


namespace fraction_operations_l181_181084

theorem fraction_operations :
  let a := 1 / 3
  let b := 1 / 4
  let c := 1 / 2
  (a + b = 7 / 12) ∧ ((7 / 12) / c = 7 / 6) := by
{
  sorry
}

end fraction_operations_l181_181084


namespace convex_polyhedron_has_even_face_l181_181316

-- Definitions for the conditions
variables {V : Type*} [DecidableEq V] -- V for vertices type with decidable equality
structure Face (V : Type*) :=
(edges : list (V × V))

structure ConvexPolyhedron (V : Type*) :=
(faces : list (Face V))
(is_convex : Prop)

variables (P : ConvexPolyhedron V)
variable (odd_faces : P.faces.length % 2 = 1)

-- The statement we need to prove
theorem convex_polyhedron_has_even_face (P : ConvexPolyhedron V) (odd_faces : P.faces.length % 2 = 1) :
  ∃ f ∈ P.faces, (f.edges.length % 2 = 0) :=
sorry

end convex_polyhedron_has_even_face_l181_181316


namespace find_f_inv_64_l181_181729

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_mul_add (x y : ℝ) (hx : x > 0) (hy : y > 0) : f(x * y) = f(x) + f(y)
axiom f_two : f(2) = 1

theorem find_f_inv_64 : f(1 / 64) = -6 := by
  sorry

end find_f_inv_64_l181_181729


namespace tangent_normal_line_correct_l181_181541

noncomputable def tangent_normal_line (a : ℝ) : (ℝ × ℝ) :=
  let t0 := Real.pi / 4
  let x0 := a * (t0 * Real.sin t0 + Real.cos t0)
  let y0 := a * (Real.sin t0 - t0 * Real.cos t0)
  let y_x' := Real.tan t0
  let tangent := (fun x => x + ((a * Real.sqrt 2 * Real.pi) / 4))
  let normal := (fun x => -x + a * Real.sqrt 2)
  (tangent, normal)

theorem tangent_normal_line_correct:
  ∀ (a : ℝ), tangent_normal_line a = (fun x => x + ((a * Real.sqrt 2 * Real.pi) / 4),
                                     fun x => -x + a * Real.sqrt 2) := 
by
  intros
  sorry

end tangent_normal_line_correct_l181_181541


namespace prism_area_l181_181247

noncomputable def prism_lateral_area (BC CC1 : ℝ) (angle : ℝ) : ℝ :=
  if H : angle = 60 then (2 + 1 / 2 + sqrt 15 / 2) * 1 else 0

theorem prism_area :
  prism_lateral_area 2 1 60 = (5 + sqrt 15) / 2 :=
  by
    unfold prism_lateral_area
    split_ifs
    . simp
    sorry

end prism_area_l181_181247


namespace square_circle_area_ratio_l181_181055

theorem square_circle_area_ratio (r : ℝ) (s : ℝ) 
  (h1 : ∀ (chord : ℝ), chord = r / 2 → 
    ∃ (x : ℝ), (2 * x = chord) ∧ 
               (2 * r / 4 = √3 * x / 4))
  (h2 : s = (r * √3) / 2) :
  (s ^ 2) / (π * r ^ 2) = 3 / (4 * π) :=
by
  sorry

end square_circle_area_ratio_l181_181055


namespace probability_point_on_graph_is_1_over_12_l181_181812

def is_on_graph (x y : ℕ) : Prop :=
  y = 2 * x

def is_valid_roll (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 6

theorem probability_point_on_graph_is_1_over_12 :
  let outcomes := (fin 6).val.prod (fin 6)
  let favorable_events := filter (λ (x, y), is_on_graph x y) outcomes
  (1 : ℝ) / (sizeof outcomes : ℝ) = 1 / 12 :=
by {
  sorry
}

end probability_point_on_graph_is_1_over_12_l181_181812


namespace proof_min_value_a3_and_a2b2_l181_181276

noncomputable def min_value_a3_and_a2b2 (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧
  (a2 = a1 + b1) ∧ (a3 = a1 + 2 * b1) ∧ (b2 = b1 * a1) ∧ 
  (b3 = b1 * a1^2) ∧ (a3 = b3) ∧ 
  (a3 = 3 * Real.sqrt 6 / 2) ∧
  (a2 * b2 = 15 * Real.sqrt 6 / 8) 

theorem proof_min_value_a3_and_a2b2 : ∃ (a1 a2 a3 b1 b2 b3 : ℝ), min_value_a3_and_a2b2 a1 a2 a3 b1 b2 b3 :=
by
  use 2*Real.sqrt 6/3, 5*Real.sqrt 6/4, 3*Real.sqrt 6/2, Real.sqrt 6/4, 3/2, 3*Real.sqrt 6/2
  sorry

end proof_min_value_a3_and_a2b2_l181_181276


namespace lives_lost_l181_181405

-- Conditions given in the problem
def initial_lives : ℕ := 83
def current_lives : ℕ := 70

-- Prove the number of lives lost
theorem lives_lost : initial_lives - current_lives = 13 :=
by
  sorry

end lives_lost_l181_181405


namespace area_of_triangle_values_of_sides_l181_181224

variables {a b c : ℝ}
variables {A B C : ℝ}
variables {S : ℝ}

-- Define the given conditions
def triangle_sides := a > 0 ∧ b > 0 ∧ c > 0
def sides_relationship := b * c = 5
def cos_A_half := cos (A / 2) = 3 * sqrt 10 / 10
def sin_relationship := sin B = 5 * sin C

-- Part (1): Prove the area of ΔABC
theorem area_of_triangle (h1 : triangle_sides) (h2 : sides_relationship) (h3 : cos_A_half) :
  S = 3 / 2 :=
sorry

-- Part (2): Find the values of a, b, c
theorem values_of_sides (h1 : triangle_sides) (h2 : sides_relationship) (h3 : cos_A_half) (h4 : sin_relationship) :
  a = 3 * sqrt 2 ∧ b = 5 ∧ c = 1 :=
sorry

end area_of_triangle_values_of_sides_l181_181224


namespace zuminglish_8_letter_words_l181_181618

-- Define the sequences a_n, b_n, c_n with initial conditions and recurrence relations
def a : ℕ → ℕ
| 2 := 4
| (n + 1) := if n = 1 then a n else 2 * (a n + c n)
-- We can define b and c similarly
def b : ℕ → ℕ
| 2 := 2
| (n + 1) := if n = 1 then a n else a n

def c : ℕ → ℕ
| 2 := 4
| (n + 1) := if n = 1 then a n else 2 * b n

-- Calculate N by summing a_8, b_8, and c_8
def N : ℕ := a 8 + b 8 + c 8

-- Statement to prove
theorem zuminglish_8_letter_words :
  N = a 8 + b 8 + c 8 :=
by
  sorry

end zuminglish_8_letter_words_l181_181618


namespace triangle_inside_symmetric_polygon_l181_181469

noncomputable def T_inv (P : Point) (A B C : Point) : (Point × Point × Point) :=
  let A' := 2 * P - A
  let B' := 2 * P - B
  let C' := 2 * P - C
  (A', B', C')

theorem triangle_inside_symmetric_polygon
  (M : Set Point)
  (hM_convex : convex ℝ M)
  (hM_symmetric : ∀ x ∈ M, ∃ y ∈ M, y = -x)
  (A B C : Point)
  (T := triangle A B C)
  (hT_M : T ⊆ M)
  (P : Point)
  (hP_T : P ∈ T)
  (A' B' C' : Point)
  (hT'_sym : (A', B', C') = T_inv P A B C) :
  A' ∈ M ∨ B' ∈ M ∨ C' ∈ M :=
  sorry

end triangle_inside_symmetric_polygon_l181_181469


namespace find_S6_l181_181638

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * a 1

def geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (∑ i in finset.range n, a i)

theorem find_S6 (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_geo_seq : is_geometric_sequence a)
  (h_geo_sum : geometric_sum S a)
  (h2 : S 2 = 4)
  (h4 : S 4 = 6) :
  S 6 = 7 := 
sorry

end find_S6_l181_181638


namespace probability_of_white_inclusion_l181_181463

-- Define the set of colors
def colors : Finset String := {"yellow", "white", "blue", "red"}

-- All combinations of selecting 2 colors
def combinations : Finset (Finset String) := Finset.powersetLen 2 colors

-- Combinations that include white
def include_white : Finset (Finset String) := 
  combinations.filter (λ s => "white" ∈ s)

-- The probability theorem to be proved
theorem probability_of_white_inclusion : 
  (include_white.card : ℚ) / (combinations.card : ℚ) = 1 / 2 := 
by
  sorry

end probability_of_white_inclusion_l181_181463


namespace temperature_at_midnight_is_minus4_l181_181299

-- Definitions of initial temperature and changes
def initial_temperature : ℤ := -2
def temperature_rise_noon : ℤ := 6
def temperature_drop_midnight : ℤ := 8

-- Temperature at midnight
def temperature_midnight : ℤ :=
  initial_temperature + temperature_rise_noon - temperature_drop_midnight

theorem temperature_at_midnight_is_minus4 :
  temperature_midnight = -4 := by
  sorry

end temperature_at_midnight_is_minus4_l181_181299


namespace prove_radius_of_circle_D_l181_181563

noncomputable def radius_of_circle_D (R : ℝ) : Prop :=
  let C := (λ x y : ℝ, x^2 + (y - 4)^2 = 18)
  let D := (λ x y : ℝ, (x - 1)^2 + (y - 1)^2 = R^2)
  let common_chord_length := (6 * Real.sqrt 2)
  ∃ (line : ℝ → ℝ → Prop), 
    (forall x y, C x y → line x y) ∧ 
    (forall x y, D x y → line x y) ∧ 
    ((4 - R^2)^2 / (4 + 9) = common_chord_length^2)

theorem prove_radius_of_circle_D : radius_of_circle_D (2 * Real.sqrt 7) :=
sorry

end prove_radius_of_circle_D_l181_181563


namespace length_of_other_parallel_side_l181_181526

theorem length_of_other_parallel_side 
  (a : ℝ) (h : ℝ) (A : ℝ) (x : ℝ) 
  (h_a : a = 16) (h_h : h = 15) (h_A : A = 270) 
  (h_area_formula : A = 1 / 2 * (a + x) * h) : 
  x = 20 :=
sorry

end length_of_other_parallel_side_l181_181526


namespace combined_avg_score_l181_181627

noncomputable def classA_student_count := 45
noncomputable def classB_student_count := 55
noncomputable def classA_avg_score := 110
noncomputable def classB_avg_score := 90

theorem combined_avg_score (nA nB : ℕ) (avgA avgB : ℕ) 
  (h1 : nA = classA_student_count) 
  (h2 : nB = classB_student_count) 
  (h3 : avgA = classA_avg_score) 
  (h4 : avgB = classB_avg_score) : 
  (nA * avgA + nB * avgB) / (nA + nB) = 99 := 
by 
  rw [h1, h2, h3, h4]
  -- Substitute the values to get:
  -- (45 * 110 + 55 * 90) / (45 + 55) 
  -- = (4950 + 4950) / 100 
  -- = 9900 / 100 
  -- = 99
  sorry

end combined_avg_score_l181_181627


namespace leak_drain_time_l181_181460

theorem leak_drain_time (P L : ℝ) (fill_time : P = 0.5) (leak_fill_time : 2 + 1/3 = 7/3)
  (combined_rate : P - L = 3/7) : 1 / L = 14 :=
by
  -- Assume P = 0.5 tanks/hour because the pump can fill the tank in 2 hours
  have h_P : P = 0.5 := fill_time,
  -- Combined rate P - L = 3/7 tanks/hour
  have h_combined : P - L = 3/7 := combined_rate,
  -- Solve for leak rate L
  have h_L : L = 0.5 - 3/7 := by sorry,
  -- Prove 1 / L = 14
  show 1 / L = 14, from by sorry

end leak_drain_time_l181_181460


namespace trigonometric_identity_l181_181364

theorem trigonometric_identity :
  sin (50 * (real.pi / 180)) * (1 + real.sqrt 3 * tan (10 * (real.pi / 180))) = 1 :=
by sorry

end trigonometric_identity_l181_181364


namespace avg_transformed_std_dev_transformed_l181_181561

variables (n : ℕ) (x : Fin n → ℝ)
hypothesis (avg_x : (∑ i, x i) / n = 4)
hypothesis (std_dev_x : (∑ i, (x i - (∑ i, x i) / n) ^ 2 / n).sqrt = 7)

theorem avg_transformed : (∑ i, 3 * x i + 2) / n = 14 :=
sorry

theorem std_dev_transformed : 
  (∑ i, ((3 * x i + 2) - (∑ i, 3 * x i + 2) / n) ^ 2 / n).sqrt = 21 :=
sorry

end avg_transformed_std_dev_transformed_l181_181561


namespace root_exists_l181_181098

noncomputable def rational_poly_of_deg_4 : Polynomial ℚ :=
  Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 0 * Polynomial.X^3 - Polynomial.C 10 * Polynomial.X^2 + Polynomial.C 0 * Polynomial.X + Polynomial.C 1

theorem root_exists (α : ℚ) (β : ℚ) :
  α = Real.sqrt (2^(2/3) - 3) ∧ β = 2^(1/3) → IsRoot (Polynomial.map (algebraMap ℚ ℝ) rational_poly_of_deg_4) (α + Real.sqrt 3) :=
  by sorry

end root_exists_l181_181098


namespace leggings_needed_l181_181587

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l181_181587


namespace count_even_digit_divisible_by_4_is_500_l181_181589

def even_digits : Finset ℕ := { 0, 2, 4, 6, 8 }

def is_even_digit (n : ℕ) : Prop := n ∈ even_digits

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def count_valid_numbers : ℕ :=
  (Finset.filter (λ n, is_four_digit n ∧
                     (∀ d : ℕ, d ∈ n.digits 10 → is_even_digit d) ∧
                     is_divisible_by_4 n)
                 (Finset.range 10000)).card

theorem count_even_digit_divisible_by_4_is_500 : count_valid_numbers = 500 := sorry

end count_even_digit_divisible_by_4_is_500_l181_181589


namespace find_a_for_even_function_l181_181165

open Function

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- State the problem in Lean.
theorem find_a_for_even_function :
  (∀ x, f a x = f a (-x)) → a = 1 :=
sorry

end find_a_for_even_function_l181_181165


namespace borrowed_amount_l181_181660

variables (R T A : ℝ)

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem borrowed_amount (R_value : R = 6) (T_value : T = 9) (A_value : A = 8310) : 
  let P := 5396.10 in 
  let SI := simple_interest P R T in
  A = P + SI :=
begin
  -- include the needed variables
  have R_pos : 0 < R := by sorry,
  have T_pos : 0 < T := by sorry,
  have A_pos : 0 < A := by sorry,
  
  -- We substitute R_value, T_value, and A_value into the equality
  rw [R_value, T_value, A_value],

  -- P = 5396.10 is our hypothesis
  let P := 5396.10,

  -- Calculate the Simple Interest using the function
  let SI := simple_interest P R T,

  -- Calculate total amount and verify the equality
  suffices : A = P + SI,
  by sorry
end

#check borrowed_amount

end borrowed_amount_l181_181660


namespace find_angle_A_find_area_of_triangle_l181_181609

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) (h1 : a = 3 * (Real.sqrt 3) / 2) (h2 : b = Real.sqrt 3) : Prop :=
  h3 : 4 * (Real.sin (B + C) / 2) ^ 2 - (Real.cos (2 * A)) = 7 / 2

theorem find_angle_A {a b c A B C : ℝ} :
  a = 3 * (Real.sqrt 3) / 2 →
  b = Real.sqrt 3 →
  4 * (Real.sin (B + C) / 2) ^ 2 - Real.cos (2 * A) = 7 / 2 →
  A = 60 :=
by
  sorry

theorem find_area_of_triangle {a b c A B C : ℝ} :
  a + c = 3 * (Real.sqrt 3) / 2 →
  b = Real.sqrt 3 →
  A = 60 →
  let area := (1 / 2) * b * c * Real.sin (A) in
  area = 15 * (Real.sqrt 3) / 32 :=
by
  sorry

end find_angle_A_find_area_of_triangle_l181_181609


namespace cos_double_angle_identity_l181_181127

theorem cos_double_angle_identity (α : ℝ) (h : sin (α + π / 5) = sqrt 3 / 3) : cos (2 * α + 2 * π / 5) = 1 / 3 :=
by 
  sorry

end cos_double_angle_identity_l181_181127


namespace common_difference_l181_181656

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h_seq : ∀ n, a n = 1 + (n - 1) * d) 
  (h_geom : (a 3) ^ 2 = (a 1) * (a 13)) (h_ne_zero: d ≠ 0) : d = 2 :=
by
  sorry

end common_difference_l181_181656


namespace unique_f_l181_181273

open Rat

def pos_rat := {q : ℚ // q > 0}

noncomputable def f : pos_rat → pos_rat := sorry

axiom f_eq (x y : pos_rat) : f(x) = f(x + y) + f(x + x^2 * f(y))

theorem unique_f (f : pos_rat → pos_rat) 
(h : ∀ x y : pos_rat, f(x) = f(x + y) + f(x + x^2 * f(y))) : 
(f = λ x, ⟨1 / x.1, by simp ⟩) := sorry

end unique_f_l181_181273


namespace max_volume_of_sphere_in_prism_l181_181807

noncomputable def volume_of_inscribed_sphere
  (AB BC AA1 : ℝ) (h1 : AB = 6) (h2 : BC = 8) (h3 : AA1 = 3) : ℝ :=
  (4 / 3) * Real.pi * (3 / 2) ^ 3

theorem max_volume_of_sphere_in_prism :
  ∀ (AB BC AA1 : ℝ), AB = 6 → BC = 8 → AA1 = 3 → volume_of_inscribed_sphere AB BC AA1 = 9 * Real.pi / 2 :=
by
  intros AB BC AA1 h1 h2 h3
  unfold volume_of_inscribed_sphere
  sorry

end max_volume_of_sphere_in_prism_l181_181807


namespace min_lines_to_cover_point_l181_181792

/--
On a plane, there is a circle. Prove that the minimum number of lines needed such that by reflecting the given circle symmetrically relative to these lines (in any finite order), one can cover any given point on the plane is exactly 3.
-/
theorem min_lines_to_cover_point (O : Point) (R : ℝ) : 
  ∃ n ≥ 3, (∀ P : Point, ∃ lines : List Line, List.length lines = n ∧ 
    (∃ Q : Point, circle_reflected Q lines = P)) :=
sorry

end min_lines_to_cover_point_l181_181792


namespace sample_not_representative_l181_181446

-- Define the events A and B
def A : Prop := ∃ (x : Type), (x → Prop) -- A person has an email address
def B : Prop := ∃ (x : Type), (x → Prop) -- A person uses the internet

-- Define the dependence and characteristics
def is_dependent (A B : Prop) : Prop := A ∧ B -- A and B are dependent
def linked_to_internet_usage (A : Prop) (B : Prop) : Prop := A → B -- A is closely linked to B
def not_uniform_distribution (B : Prop) : Prop := ¬ (∀ x, B x) -- Internet usage is not uniformly distributed

-- Define the subset condition
def is_subset_of_regular_users (A B : Prop) : Prop := ∀ x, A x → B x -- A represents regular internet users

-- Prove the given statement
theorem sample_not_representative
  (A B : Prop)
  (h_dep : is_dependent A B)
  (h_link : linked_to_internet_usage A B)
  (h_not_unif : not_uniform_distribution B)
  (h_subset : is_subset_of_regular_users A B) :
  ¬ represents_urban_population A :=
sorry

end sample_not_representative_l181_181446


namespace find_cost_price_of_ball_l181_181410

variable (x : ℝ) -- assuming the cost price is a real number

-- Conditions: Selling 17 balls at Rs. 720 results in a loss equal to the cost price of 5 balls
def correct_answer : Prop :=
  let selling_price := 720
  let total_cost := 17 * x
  let loss := 5 * x
  total_cost - loss = selling_price ∧ x = 60

theorem find_cost_price_of_ball (x : ℝ) (h : let selling_price := 720 in
                                             let total_cost := 17 * x in
                                             let loss := 5 * x in
                                             total_cost - loss = selling_price) : x = 60 := 
sorry

end find_cost_price_of_ball_l181_181410


namespace finite_operations_second_player_wins_initial_l181_181260

-- Definitions representing the conditions
def Cell := Bool -- true represents "L", false represents "R"
def PiecePosition := Nat

structure Configuration where
  cells : List Cell
  piece : PiecePosition

def move_piece (c : Configuration) : Configuration :=
  sorry -- details of this function based on the movement rules

-- Part 1: Prove that only a finite number of operations can be performed for any initial configuration
theorem finite_operations (n : Nat) (c : Configuration)
  (h1 : 0 < n)
  (h2 : c.cells.length = n + 2)
  (h3 : c.piece > 0)
  (h4 : c.piece < n + 1) :
  ∃ k, ∀ k' > k, move_piece^[k'] c = move_piece^[k] c :=
sorry

-- Part 2: Determine initial configurations such that the second player can always win
def is_losing_configuration (c : Configuration) : Prop :=
  (c.cells.nth c.piece = some false ∧ ∀ i > c.piece, c.cells.nth i ≠ some false) ∨
  (c.cells.nth c.piece = some true ∧ ∀ i < c.piece, c.cells.nth i ≠ some true)

def is_winning_strategy (c : Configuration) : Prop :=
  ¬ is_losing_configuration c

theorem second_player_wins_initial (n : Nat) (c : Configuration)
  (h1 : 0 < n)
  (h2 : c.cells.length = n + 2)
  (h3 : c.piece > 0)
  (h4 : c.piece < n + 1) :
  is_winning_strategy c :=
sorry

end finite_operations_second_player_wins_initial_l181_181260


namespace largest_prime_difference_for_126_l181_181344

-- Comment: We define the conditions for the problem
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_pair_126 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 126

-- Now we assert the proposition we need to prove
theorem largest_prime_difference_for_126 :
  ∃ p q : ℕ, is_pair_126 p q ∧ (∀ r s : ℕ, is_pair_126 r s → (p - q).nat_abs ≥ (r - s).nat_abs) ∧ (p - q).nat_abs = 100 :=
by
  sorry

end largest_prime_difference_for_126_l181_181344


namespace not_representative_l181_181449

variable (A B : Prop)

axiom has_email : A
axiom uses_internet : B
axiom dependent : A → B
axiom sample_size : Nat := 2000

theorem not_representative (hA : has_email) (hB : uses_internet) (hD : dependent) (hS : sample_size = 2000) : ¬(∀ x, A x) :=
  sorry

end not_representative_l181_181449


namespace part_a_a_part_a_b_part_a_c_part_b_l181_181026

/-- Definition of the function f according to the given conditions --/
def f : ℕ → ℕ
| 0     := 0
| (m+1) := if m % 2 = 0 then 2 * f (m / 2) else (m / 2) + 2 * f (m / 2)

/-- Set definitions for L, E, and G according to the conditions --/
def L := {n : ℕ | ∃ k > 0, n = 2 * k}
def E := {n : ℕ | n = 0 ∨ ∃ k ≥ 0, n = 4 * k + 1}
def G := {n : ℕ | ∃ k ≥ 0, n = 4 * k + 3}

/-- Proving L, E, and G are as defined given the conditions on f --/
theorem part_a_a : L = {n : ℕ | f(n) < f(n + 1)} := sorry
theorem part_a_b : E = {n : ℕ | f(n) = f(n + 1)} := sorry
theorem part_a_c : G = {n : ℕ | f(n) > f(n + 1)} := sorry

/-- Definition of the maximum function a_k --/
def a_k (k : ℕ) : ℕ := k * 2 ^ (k - 1) - 2 ^ k + 1

/-- Prove the formula for a_k given the conditions on f --/
theorem part_b : ∀ k ≥ 0, a_k k = max {f n | 0 ≤ n ∧ n ≤ 2^k} := sorry

end part_a_a_part_a_b_part_a_c_part_b_l181_181026


namespace range_of_g_l181_181257

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x)^4 + (arcsin x)^4

theorem range_of_g : 
  ∀ x ∈ Icc (-1 : ℝ) 1, 
  g x ∈ set.Icc (π^4 / 16) (17 * π^4 / 16) :=
by 
  sorry

end range_of_g_l181_181257


namespace exponential_inequality_l181_181950

variable (a b : ℝ)
#check lt_of_pow_lt_pow_of_pow_gt_pow

theorem exponential_inequality (h : a > b) : 3^(-a) < 3^(-b) :=
by
  apply pow_lt_pow_of_lt_left
  sorry

end exponential_inequality_l181_181950


namespace convex_polyhedron_face_parity_l181_181319

theorem convex_polyhedron_face_parity 
  (P : Polyhedron) 
  (convex : is_convex P) 
  (odd_faces : odd (number_of_faces P)) : 
  ∃ F : Face P, even (number_of_edges F) :=
sorry

end convex_polyhedron_face_parity_l181_181319


namespace original_pencils_count_l181_181371

theorem original_pencils_count (total_pencils : ℕ) (added_pencils : ℕ) (original_pencils : ℕ) : total_pencils = original_pencils + added_pencils → original_pencils = 2 :=
by
  sorry

end original_pencils_count_l181_181371


namespace poly_product_even_not_all_div_4_l181_181192

noncomputable def poly_even_coeff {n : Nat} (P : Polynomial ℤ) : Prop :=
  ∀ i : Nat, i ≤ n → P.coeff i % 2 = 0

noncomputable def some_coeff_odd {n : Nat} (P : Polynomial ℤ) : Prop :=
  ∃ i : Nat, i ≤ n ∧ P.coeff i % 2 = 1

theorem poly_product_even_not_all_div_4 
    (P Q : Polynomial ℤ)
    (hP: ∀ i, P.coeff i ∈ ℤ)
    (hQ: ∀ i, Q.coeff i ∈ ℤ)
    (hprod_even: ∀ i, (P * Q).coeff i % 2 = 0)
    (hnot_all_div4: ∃ i, (P * Q).coeff i % 4 ≠ 0) :
  (poly_even_coeff P ∧ some_coeff_odd Q) ∨ (poly_even_coeff Q ∧ some_coeff_odd P) := 
sorry

end poly_product_even_not_all_div_4_l181_181192


namespace complex_equation_solution_l181_181137

theorem complex_equation_solution (x y : ℝ)
  (h : (x / (1 - (-ⅈ)) + y / (1 - 2 * (-ⅈ)) = 5 / (1 - 3 * (-ⅈ)))) :
  x + y = 4 :=
sorry

end complex_equation_solution_l181_181137


namespace exists_line_through_point_l181_181798

noncomputable def point (x : ℝ) (y : ℝ) (z : ℝ) : Vector := ⟨x, y, z⟩

noncomputable def line (a d : Vector) (t : ℝ) : Vector := a + t • d

noncomputable def is_perpendicular (m d : Vector) : Prop :=
  m ⬝ d = 0

noncomputable def forms_angle (m n : Vector) (θ : ℝ) : Prop :=
  (m ⬝ n) / (‖m‖ * ‖n‖) = Real.cos θ

theorem exists_line_through_point 
  (P : Vector)
  (a d : Vector)
  (n : Vector) :
  ∃ m : Vector, is_perpendicular m d ∧ forms_angle m n (Real.pi / 6) :=
sorry

end exists_line_through_point_l181_181798


namespace digit_150_of_fraction_17_over_98_is_9_l181_181002

theorem digit_150_of_fraction_17_over_98_is_9 :
  ∃ r:ℚ, r = 17 / 98 ∧ (let ds := ((rat.digits (17 / 98)).dropWhile (= 0)).take 200 in
    (ds.drop 149).head! = 9) :=
sorry

end digit_150_of_fraction_17_over_98_is_9_l181_181002


namespace area_of_square_l181_181057

-- Define the problem setting and the conditions
def square (side_length : ℝ) : Prop :=
  ∃ (width height : ℝ), width * height = side_length^2
    ∧ width = 5
    ∧ side_length / height = 5 / height

-- State the theorem to be proven
theorem area_of_square (side_length : ℝ) (width height : ℝ) (h1 : width = 5) (h2: side_length = 5 + 2 * height): 
  square side_length → side_length^2 = 400 :=
by
  intro h
  sorry

end area_of_square_l181_181057


namespace length_of_second_platform_l181_181060

theorem length_of_second_platform (train_length first_platform_length : ℕ) (time_to_cross_first_platform time_to_cross_second_platform : ℕ) 
  (H1 : train_length = 110) (H2 : first_platform_length = 160) (H3 : time_to_cross_first_platform = 15) 
  (H4 : time_to_cross_second_platform = 20) : ∃ second_platform_length, second_platform_length = 250 := 
by
  sorry

end length_of_second_platform_l181_181060


namespace trig_eqn_solution_l181_181706

noncomputable def solve_trig_eqn (x y z : ℝ) (m n : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (cos y ≠ 0) ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) ∧
  (x = π / 2 + π * m) ∧
  (y = π * n) ∧
  (z = 2 * π * m)

theorem trig_eqn_solution (x y z : ℝ) (m n : ℤ) :
  sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) →
  x = π / 2 + π * m ∧ y = π * n ∧ z = 2 * π * m :=
by
  sorry

end trig_eqn_solution_l181_181706


namespace largest_k_exists_l181_181952

theorem largest_k_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n → (c - b) ≥ k ∧ (b - a) ≥ k ∧ (a + b ≥ c + 1)) ∧ 
  (k = (n - 1) / 3) :=
  sorry

end largest_k_exists_l181_181952


namespace sphere_can_be_circumscribed_l181_181241

structure ConvexPolyhedron (M : Type) :=
  (vertices : Set M)
  (edges : Set (M × M))
  (faces : Set (Set M))
  (condition1 : ∀ v ∈ vertices, ∃! e e1 e2 ∈ edges, 
    (v = (e.1 ∧ e.2) ∧ (v = (e1.1 ∧ e1.2)) ∧ (v = (e2.1 ∧ e2.2))))
  (condition2 : ∀ f ∈ faces, ∃ circumscribed_circle : Set M, 
    ∀ edge ∈ f, edge ∈ circumscribed_circle)

theorem sphere_can_be_circumscribed (M : Type) [ConvexPolyhedron M] :
  ∃ circumscribed_sphere : Set M, ∀ v ∈ ConvexPolyhedron.vertices M, v ∈ circumscribed_sphere :=
sorry

end sphere_can_be_circumscribed_l181_181241


namespace volume_bowling_ball_after_drilling_correct_l181_181790

noncomputable def volume_bowling_ball_after_drilling : ℝ :=
  let r_ball := 12 in -- radius of the ball in cm
  let V_ball := (4 / 3) * Real.pi * r_ball^3 in -- volume of the ball
  let r1 := 1 in -- radius of hole 1 in cm
  let r2 := 1.5 in -- radius of hole 2 in cm
  let r3 := 2 in -- radius of hole 3 in cm
  let h := 10 in -- depth of the holes in cm
  let V1 := Real.pi * r1^2 * h in -- volume of hole 1
  let V2 := Real.pi * r2^2 * h in -- volume of hole 2
  let V3 := Real.pi * r3^2 * h in -- volume of hole 3
  V_ball - (V1 + V2 + V3)

theorem volume_bowling_ball_after_drilling_correct :
  volume_bowling_ball_after_drilling = 2231.5 * Real.pi :=
by simp[volume_bowling_ball_after_drilling]; sorry

end volume_bowling_ball_after_drilling_correct_l181_181790


namespace find_value_of_x_l181_181246

theorem find_value_of_x (a b c d e f x : ℕ) (h1 : a ≠ 1 ∧ a ≠ 6 ∧ b ≠ 1 ∧ b ≠ 6 ∧ c ≠ 1 ∧ c ≠ 6 ∧ d ≠ 1 ∧ d ≠ 6 ∧ e ≠ 1 ∧ e ≠ 6 ∧ f ≠ 1 ∧ f ≠ 6 ∧ x ≠ 1 ∧ x ≠ 6)
  (h2 : a + x + d = 18)
  (h3 : b + x + f = 18)
  (h4 : c + x + 6 = 18)
  (h5 : a + b + c + d + e + f + x + 6 + 1 = 45) :
  x = 7 :=
sorry

end find_value_of_x_l181_181246


namespace range_of_n_l181_181560

theorem range_of_n (n : ℝ) (x : ℝ) (h1 : 180 - n > 0) (h2 : ∀ x, 180 - n != x ∧ 180 - n != x + 24 → 180 - n + x + x + 24 = 180 → 44 ≤ x ∧ x ≤ 52 → 112 ≤ n ∧ n ≤ 128)
  (h3 : ∀ n, 180 - n = max (180 - n) (180 - n) - 24 ∧ min (180 - n) (180 - n) = n - 24 → 104 ≤ n ∧ n ≤ 112)
  (h4 : ∀ n, 180 - n = min (180 - n) (180 - n) ∧ max (180 - n) (180 - n) = 180 - n + 24 → 128 ≤ n ∧ n ≤ 136) :
  104 ≤ n ∧ n ≤ 136 :=
by sorry

end range_of_n_l181_181560


namespace binomial_sum_sum_of_binomial_solutions_l181_181903

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181903


namespace penelope_saving_days_l181_181300

theorem penelope_saving_days :
  ∀ (daily_savings total_saved : ℕ),
  daily_savings = 24 ∧ total_saved = 8760 →
    total_saved / daily_savings = 365 :=
by
  rintro _ _ ⟨rfl, rfl⟩
  sorry

end penelope_saving_days_l181_181300


namespace part_a_part_b_l181_181420

-- Defining the arithmetic progression and problem setup
structure AP (r a : ℤ) :=
(is_arithmetic_progression : ∀ n : ℤ, a + n * r ∈ ℤ)

def M := { ap : AP // ∃ r, r > 1 }

-- Part (a) Problem
theorem part_a : 
  ∃ (aps : List AP), (∀ (ap ∈ aps), ∃ r, r > 1) ∧ (∀ x : ℤ, ∃ ap ∈ aps, ∃ n : ℤ, x = ap.val.a + n * ap.val.r) := 
sorry

-- Part (b) Problem
theorem part_b : 
  ∀ (aps : List AP), (∀ ap1 ap2 ∈ aps, ap1.val.r ≠ ap2.val.r → coprime ap1.val.r ap2.val.r) 
  → ¬ (∀ x : ℤ, ∃ ap ∈ aps, ∃ n : ℤ, x = ap.val.a + n * ap.val.r) :=
sorry

end part_a_part_b_l181_181420


namespace cost_price_of_article_l181_181482

theorem cost_price_of_article (C MP : ℝ) (h1 : 0.90 * MP = 1.25 * C) (h2 : 1.25 * C = 65.97) : C = 52.776 :=
by
  sorry

end cost_price_of_article_l181_181482


namespace solve_inequality_l181_181101

theorem solve_inequality :
  {y : ℝ | (y^2 + 2*y^3 - 3*y^4)/(y + 2*y^2 - 3*y^3) ≥ -1} = (set.Icc (-1:ℝ) (-1/3) ∪ set.Ioo (-1/3:ℝ) 0 ∪ set.Ioo 0 1 ∪ set.Ioi 1) :=
by
  sorry

end solve_inequality_l181_181101


namespace number_of_differences_l181_181195

def is_difference (x y d : ℕ) : Prop := x ≠ y ∧ d = abs (x - y)

theorem number_of_differences : 
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃! n : ℕ, n = 9 ∧ ∀ d ∈ insert 0 (s.bind (λ x, s.image (λ y, abs (x - y)))), d ≠ 0 → d <= n :=
sorry

end number_of_differences_l181_181195


namespace locus_of_centers_l181_181720

theorem locus_of_centers (a b : ℝ) :
  (∀ (r : ℝ), (a-1)^2 + (b-1)^2 = (r+2)^2 ∧ (a-4)^2 + (b-1)^2 = (3-r)^2) →
  (84 * a^2 + 100 * b^2 - 336 * a - 200 * b + 900 = 0) :=
begin
  sorry,
end

end locus_of_centers_l181_181720


namespace simplify_rational_expr_l181_181783

noncomputable def expr1 := (Real.sqrt 12 + Real.sqrt3 3) / (Real.sqrt 3 + Real.sqrt3 2)
noncomputable def expr2 := (9 - Real.sqrt3 12 - Real.sqrt3 6) / (3 - Real.sqrt3 4)

theorem simplify_rational_expr : expr1 = expr2 :=
by
  sorry

end simplify_rational_expr_l181_181783


namespace Haleigh_needs_leggings_l181_181585

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l181_181585


namespace binomial_sum_sum_of_binomial_solutions_l181_181899

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181899


namespace sale_in_second_month_l181_181428

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale target_sum : ℕ)
  (h1 : sale1 = 3435)
  (h3 : sale3 = 3855)
  (h4 : sale4 = 4230)
  (h5 : sale5 = 3562)
  (h6 : sale6 = 1991)
  (h_avg : average_sale = 3500)
  (h_target : target_sum = average_sale * 6) :
  sale1 + sale3 + sale4 + sale5 + sale6 + ?S = target_sum → ?S = 3927 := by
  sorry

end sale_in_second_month_l181_181428


namespace solve_equation_l181_181709

theorem solve_equation (x y z : ℝ) (m n : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1 / (sin^2 x ))^3 + (cos^2 y + 1 / (cos^2 y))^3 = 16 * cos z) →
  (cos z = 1) ∧ 
  (∃ m : ℤ, x = π / 2 + π * m) ∧ 
  (∃ n : ℤ, y = π * n) ∧ 
  (∃ m : ℤ, z = 2 * π * m) :=
by
  intros hsin_cos hcos_sin heq
  sorry

end solve_equation_l181_181709


namespace identify_curve_as_hyperbola_l181_181879

noncomputable def is_hyperbola (r θ : ℝ) : Prop := 
  r = 1 / (1 - Real.sin θ)

theorem identify_curve_as_hyperbola :
  ∀ r θ : ℝ, is_hyperbola r θ → is_conic_section_hyperbola r θ :=
by
  intro r θ h
  sorry

end identify_curve_as_hyperbola_l181_181879


namespace proof_valid_set_exists_l181_181668

noncomputable def valid_set_exists : Prop :=
∃ (s : Finset ℕ), s.card = 10 ∧ 
(∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → a ≠ b) ∧ 
(∃ (t1 : Finset ℕ), t1 ⊆ s ∧ t1.card = 3 ∧ ∀ n ∈ t1, 5 ∣ n) ∧
(∃ (t2 : Finset ℕ), t2 ⊆ s ∧ t2.card = 4 ∧ ∀ n ∈ t2, 4 ∣ n) ∧
s.sum id < 75

theorem proof_valid_set_exists : valid_set_exists :=
sorry

end proof_valid_set_exists_l181_181668


namespace proof_problem_l181_181275

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < (π / 2))
variable (htan : tan α = (1 + sin β) / cos β)

theorem proof_problem : 2 * α - β = π / 2 :=
by
  sorry

end proof_problem_l181_181275


namespace people_per_column_in_second_scenario_l181_181235

def total_people (num_people_per_column_1 : ℕ) (num_columns_1 : ℕ) : ℕ :=
  num_people_per_column_1 * num_columns_1

def people_per_column_second_scenario (P: ℕ) (num_columns_2 : ℕ) : ℕ :=
  P / num_columns_2

theorem people_per_column_in_second_scenario
  (num_people_per_column_1 : ℕ)
  (num_columns_1 : ℕ)
  (num_columns_2 : ℕ)
  (P : ℕ)
  (h1 : total_people num_people_per_column_1 num_columns_1 = P) :
  people_per_column_second_scenario P num_columns_2 = 48 :=
by
  -- the proof would go here
  sorry

end people_per_column_in_second_scenario_l181_181235


namespace trajectory_polar_eq_l181_181147

-- Define the conditions and the proof problem
theorem trajectory_polar_eq (x y : ℝ) (ρ θ : ℝ)
  (hA : (-6 : ℝ, 0 : ℝ))
  (hB : (6 : ℝ, 0 : ℝ))
  (hM : ∀ (x y : ℝ), sqrt ((x + 6)^2 + y^2) * sqrt ((x - 6)^2 + y^2) = 36)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ)
  (hr : ρ^2 = x^2 + y^2) :
  ρ^2 = 144 * cos(2 * θ) := 
sorry

end trajectory_polar_eq_l181_181147


namespace range_of_g_over_domain_l181_181547

variable (c d : ℝ)
variable (h : c > 0)

noncomputable def g (x : ℝ) : ℝ := c * x + d

theorem range_of_g_over_domain : Set.range (λ x : {x : ℝ // -1 ≤ x ∧ x ≤ 2}, g c d x) = Set.Icc (-c + d) (2 * c + d) := by
  sorry

end range_of_g_over_domain_l181_181547


namespace nonnegative_intervals_l181_181921

def f (x : ℝ) : ℝ := (x - 9 * x^2 + 27 * x^3) / (9 - x^3)

theorem nonnegative_intervals :
  {x : ℝ | f x ≥ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ (1 : ℝ) / 3} ∪ {x : ℝ | x ≥ 3} :=
by
  sorry

end nonnegative_intervals_l181_181921


namespace initial_birds_count_l181_181752

variable (init_birds landed_birds total_birds : ℕ)

theorem initial_birds_count :
  (landed_birds = 8) →
  (total_birds = 20) →
  (init_birds + landed_birds = total_birds) →
  (init_birds = 12) :=
by
  intros h1 h2 h3
  sorry

end initial_birds_count_l181_181752


namespace range_of_x_l181_181181

def f (x : ℝ) : ℝ := Real.log (x^2 + Real.exp (-1)) / Real.log (1/Real.exp (1)) - (Real.abs (x) / Real.exp (1))

theorem range_of_x (x : ℝ) : 0 < x ∧ x < 2 ↔ 
  f (x + 1) < f (2 * x - 1) := 
sorry

end range_of_x_l181_181181


namespace tangent_circles_proof_l181_181757

open Real

variable (R r1 r2 : ℝ) (A B : ℝ) (h1 : r1 < R) (h2 : r2 < R)

theorem tangent_circles_proof :
  AB = 2 * R * sqrt((r1 * r2) / ((R - r1) * (R - r2))) :=
sorry

end tangent_circles_proof_l181_181757


namespace tan_80_l181_181593

theorem tan_80 (m : ℝ) (h : Real.cos (100 * Real.pi / 180) = m) :
    Real.tan (80 * Real.pi / 180) = Real.sqrt (1 - m^2) / -m :=
by
  sorry

end tan_80_l181_181593


namespace max_middle_numbers_correct_l181_181750

noncomputable def max_middle_numbers (N S : ℕ) (weights : Fin N → ℕ) (h_pos : ∀ i, weights i > 0) (h_total : ∑ i, weights i = 2 * S) : ℕ :=
  if hN : N ≥ 5 then N - 3 else 0

theorem max_middle_numbers_correct (N S : ℕ) (weights : Fin N → ℕ) (h_pos : ∀ i, weights i > 0) (h_total : ∑ i, weights i = 2 * S) : 
  (N ≥ 5) → max_middle_numbers N S weights h_pos h_total = N - 3 :=
by
  intro hN
  sorry

end max_middle_numbers_correct_l181_181750


namespace solve_trig_equation_l181_181693

theorem solve_trig_equation (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) → 
  (cos y ≠ 0) → 
  ((sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z) →
  (∃ (n k m : ℤ), x = (π / 2) + π * n ∧ y = π * k ∧ z = 2 * π * m) := 
by 
  sorry

end solve_trig_equation_l181_181693


namespace flowers_total_l181_181829

def red_roses := 1491
def yellow_carnations := 3025
def white_roses := 1768
def purple_tulips := 2150
def pink_daisies := 3500
def blue_irises := 2973
def orange_marigolds := 4234
def lavender_orchids := 350
def sunflowers := 815
def violet_lilies := 26

theorem flowers_total :
  red_roses +
  yellow_carnations +
  white_roses +
  purple_tulips +
  pink_daisies +
  blue_irises +
  orange_marigolds +
  lavender_orchids +
  sunflowers +
  violet_lilies = 21332 := 
by
  -- Simplify and add up all given numbers
  sorry

end flowers_total_l181_181829


namespace range_f_l181_181267

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 

theorem range_f : Set.Icc (0 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_f_l181_181267


namespace sample_not_representative_l181_181445

-- Define the events A and B
def A : Prop := ∃ (x : Type), (x → Prop) -- A person has an email address
def B : Prop := ∃ (x : Type), (x → Prop) -- A person uses the internet

-- Define the dependence and characteristics
def is_dependent (A B : Prop) : Prop := A ∧ B -- A and B are dependent
def linked_to_internet_usage (A : Prop) (B : Prop) : Prop := A → B -- A is closely linked to B
def not_uniform_distribution (B : Prop) : Prop := ¬ (∀ x, B x) -- Internet usage is not uniformly distributed

-- Define the subset condition
def is_subset_of_regular_users (A B : Prop) : Prop := ∀ x, A x → B x -- A represents regular internet users

-- Prove the given statement
theorem sample_not_representative
  (A B : Prop)
  (h_dep : is_dependent A B)
  (h_link : linked_to_internet_usage A B)
  (h_not_unif : not_uniform_distribution B)
  (h_subset : is_subset_of_regular_users A B) :
  ¬ represents_urban_population A :=
sorry

end sample_not_representative_l181_181445


namespace series_sum_value_l181_181168

-- Definitions of the series and their properties
def geometric_series (a : ℕ → ℝ) : Prop := ∃ r (a₀ : ℝ), ∀ n, a n = a₀ * r ^ n

def arithmetic_series (b : ℕ → ℝ) : Prop := ∃ d (b₀ : ℝ), ∀ n, b n = b₀ + d * n

-- The main theorem statement to prove
theorem series_sum_value {a b : ℕ → ℝ} (h_geo : geometric_series a)
  (h_arith : arithmetic_series b) (h1 : a 3 * a 11 = 4 * a 7) (h2 : b 7 = a 7) :
  b 5 + b 9 = 8 :=
by sorry

end series_sum_value_l181_181168


namespace number_of_pups_in_second_round_l181_181833

-- Define the conditions
variable (initialMice : Nat := 8)
variable (firstRoundPupsPerMouse : Nat := 6)
variable (secondRoundEatenPupsPerMouse : Nat := 2)
variable (finalMice : Nat := 280)

-- Define the proof problem
theorem number_of_pups_in_second_round (P : Nat) :
  initialMice + initialMice * firstRoundPupsPerMouse = 56 → 
  56 + 56 * P - 56 * secondRoundEatenPupsPerMouse = finalMice →
  P = 6 := by
  intros h1 h2
  sorry

end number_of_pups_in_second_round_l181_181833


namespace expansion_a0_alternating_sum_l181_181591

open Finset

section
variables {A : Type*} [CommRing A]

theorem expansion_a0 (a : ℕ → A) (x : A) :
  ((1 - (2 : A) * x)^2023 = ∑ i in range (2023 + 1), a i * x^i) →
  a 0 = 1 :=
begin
  intro h,
  have := congr_arg (λ p, p.eval 0) h,
  simp only [eval_pow, eval_sub, eval_one, eval_mul, eval_C, eval_X, zero_pow (by norm_num : 0 < 2023 + 1), zero_mul, mul_zero, sub_zero, one_pow] at this,
  exact this,
end

theorem alternating_sum (a : ℕ → A) (x : A):
  ((1 - (2 : A) * x)^2023 = ∑ i in range (2023 + 1), a i * x^i) →
  a 1 - a 2 + a 3 - a 4 + ∑ i in range (2023).succ.succ \ {0, 1, 2, 3},
    if i.even then -a i else a i = 1 - (3 : A)^2023 :=
begin
  intro h,
  have := congr_arg (λ p, p.eval (-1)) h,
  simp only [eval_pow, eval_sub, eval_one, eval_neg, eval_mul, eval_bit0, eval_bit1, eval_X, neg_one_pow_eq_one_iff_even, add_eq_zero_iff_eq_neg, one_pow, neg_one_pow_eq_zero_iff_odd, eval_C, neg_mul, eval_X, sub_eq_add_neg, zero_pow, mul_one, one_mul, neg_neg, pow_eq_pow] at this,
  rw [finset.sum_sub_distrib, sum_singleton, sum_odd_succ] at this,
  convert this,
end

end

end expansion_a0_alternating_sum_l181_181591


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181894
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181894


namespace solve_trig_eq_l181_181689

   theorem solve_trig_eq (x y z : ℝ) (m n : ℤ): 
     sin x ≠ 0 → cos y ≠ 0 →
     (sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z →
     (∃ m n : ℤ, x = (π / 2) + π * m ∧ y = π * n ∧ z = 2 * π * m) := 
   by
     intros h1 h2 h_eq
     sorry
   
end solve_trig_eq_l181_181689


namespace period_2_students_l181_181330

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l181_181330


namespace arrangement_count_l181_181014

namespace problem

-- Definitions
variable {XiaoWu XiaoYi XiaoJie XiaoKuai XiaoLe : Type}

-- Conditions
-- Representing the set of persons involved
def persons := [XiaoWu, XiaoYi, XiaoJie, XiaoKuai, XiaoLe]

-- Condition 1: Xiao Wu, Xiao Yi, Xiao Jie, Xiao Kuai, and Xiao Le are standing in a row
-- (Implicit in the definition of persons)

-- Condition 2: Xiao Yi does not appear in the first or last position
def yi_not_first_or_last (arrangement : List Type) : Prop :=
  arrangement.head ≠ XiaoYi ∧ arrangement.last ≠ XiaoYi

-- Condition 3: Among Xiao Wu, Xiao Jie, and Xiao Le, exactly two are adjacent
def two_adjacent (arrangement : List Type) : Prop :=
  let triples := List.zip3 arrangement (List.tail arrangement) (List.drop 2 arrangement)
  triples.countp (λ ⟨x, y, z⟩, (x ∈ [XiaoWu, XiaoJie, XiaoLe]) ∧ (y ∈ [XiaoWu, XiaoJie, XiaoLe])) = 1

-- The goal is to prove the total number of arrangements satisfying the conditions
theorem arrangement_count :
  (persons.permutations
    .filter (λ arrangement, yi_not_first_or_last arrangement ∧ two_adjacent arrangement)).length = 48 := 
  sorry
end problem

end arrangement_count_l181_181014


namespace required_hours_per_day_l181_181296

-- Defining the initial conditions
def men_initial : Nat := 100
def days_total : Nat := 50
def hours_per_day_initial : Nat := 8
def work_done_fraction : ℚ := 1 / 3
def days_till_now : Nat := 25
def additional_men : Nat := 60

-- Given these conditions, we need to prove the required hours per day for the new employees
theorem required_hours_per_day
  (men_initial : Nat) (days_total : Nat) (hours_per_day_initial : Nat)
  (work_done_fraction : ℚ) (days_till_now : Nat) (additional_men : Nat) :
  let men_total := men_initial + additional_men
  let man_hours_total := men_initial * days_total * hours_per_day_initial
  let man_hours_done := men_initial * days_till_now * hours_per_day_initial
  let man_hours_remaining := man_hours_total - man_hours_done
  ∃ (hours_per_day_new : Nat),
  hours_per_day_new = 5 :=
by
  have work_done : ℚ := man_hours_total * work_done_fraction
  have man_days_left : ℚ := men_total * 25
  have hours_per_day_new := man_hours_remaining / man_days_left
  exact ⟨5, sorry⟩

end required_hours_per_day_l181_181296


namespace eq_IA_IB_IC_eq_IO_eq_IG_eq_IH_l181_181262

variable {I O G H : Type}
variable {a b c p R r : ℝ}
variable {IA IB IC IO IG IH : Type}

-- Let I, O, G, H be the incenter, circumcenter, centroid, and orthocenter of the triangle ABC
-- Let BC = a, CA = b, AB = c, and p = 1/2 * (a + b + c)
-- Let R and r be the radii of the circumcircle and incircle respectively
axiom incenter : I
axiom circumcenter : O
axiom centroid : G
axiom orthocenter : H
axiom side_a : a = (BC : ℝ)
axiom side_b : b = (CA : ℝ)
axiom side_c : c = (AB : ℝ)
axiom semiperimeter : p = 1/2 * (a + b + c)
axiom circumradius : R = (circumradius_of_triangle (A B C : Point))
axiom inradius : r = (inradius_of_triangle (A B C : Point))

-- (I) a * IA^2 + b * IB^2 + c * IC^2 = abc
theorem eq_IA_IB_IC : a * (IA^2) + b * (IB^2) + c * (IC^2) = a * b * c := sorry

-- (II) IO^2 = R^2 - (abc)/(a + b + c) = R^2 - 2Rr
theorem eq_IO : (IO^2) = (R^2) - (a * b * c) / (a + b + c) := sorry

-- (III) IG^2 = (1/18p)[2a(b^2 + c^2) + 2b(c^2 + a^2) + 2c(a^2 + b^2) - (a^2 + b^2 + c^2) - 9abc] = (2/3)p^2 - (5/18)(a^2 + b^2 + c^2) - 4Rr
theorem eq_IG : (IG^2) = 1/18 * p * (2*a*(b^2 + c^2) + 2*b*(c^2 + a^2) + 2*c*(a^2 + b^2) - (a^2 + b^2 + c^2) - 9*a*b*c) := sorry

-- (IV) IH^2 = 4R^2 - (1/2p)(a^2 + b^2 + c^2 + abc)
theorem eq_IH : (IH^2) = 4 * (R^2) - (1 / (2 * p)) * (a^2 + b^2 + c^2 + a * b * c) := sorry

end eq_IA_IB_IC_eq_IO_eq_IG_eq_IH_l181_181262


namespace Katrina_Bridget_ratio_l181_181290

-- Definitions for the conditions
def Miriam_has_five_times_albums_Katrina (K : ℕ) : ℕ := 5 * K
def Katrina_has_some_times_albums_Bridget (n B : ℕ) : ℕ := n * B
def Bridget_albums (A : ℕ) : ℕ := A - 15  -- Since B = Adele - 15
def Total_albums (M K B A : ℕ) : ℕ := M + K + B + A

-- Given conditions translated into Lean
variables (K B A : ℕ)
variable (n : ℕ)
hypothesis (A_given : A = 30)
hypothesis (Miriam_Katrina_relation : ∀ K, Miriam_has_five_times_albums_Katrina K = 5 * K)
hypothesis (Katrina_Bridget_relation : ∀ n B, Katrina_has_some_times_albums_Bridget n B = n * B)
hypothesis (Bridget_Adele_relation : ∀ A, Bridget_albums A = A - 15)
hypothesis (Total_sum : ∀ M K B A, Total_albums M K B A = 585)

-- The theorem to prove
theorem Katrina_Bridget_ratio (K : ℕ) (B : ℕ) (A : ℕ) (n : ℕ) 
  (A_given : A = 30)
  (B_given : B = 15) : K = 90 → 6 * B = K := 
by 
  sorry -- the proof would go here

end Katrina_Bridget_ratio_l181_181290


namespace sum_fixed_points_equals_factorial_l181_181649

def permutations_with_fixed_points (n k : ℕ) : ℕ := sorry -- p_n(k)

theorem sum_fixed_points_equals_factorial (n : ℕ) :
  (∑ k in Finset.range (n + 1), k * permutations_with_fixed_points n k) = Nat.factorial n :=
sorry

end sum_fixed_points_equals_factorial_l181_181649


namespace arithmetic_sequence_general_formula_new_arithmetic_sequence_general_formula_l181_181572

open Function

theorem arithmetic_sequence_general_formula :
  (∃ (a_n : ℕ → ℤ) (d : ℤ), (∀ n, a_n n = a_n 0 + n * d) ∧ 
  a_n 0 + a_n 6 = 20 ∧ a_n 10 - a_n 7 = 18 ∧ 
  ∀ n, a_n n = 6 * n - 14) := sorry

theorem new_arithmetic_sequence_general_formula :
  (∃ (a_n : ℕ → ℤ) (d : ℤ), (∀ n, a_n n = a_n 0 + n * d) ∧ 
  a_n 0 + a_n 6 = 20 ∧ a_n 10 - a_n 7 = 18 ∧ 
  ∀ n, a_n n = 6 * n - 14 ∧ 
  ∃ (b_n : ℕ → ℤ) (d' : ℤ), (∀ n, b_n n = a_n (n // 3) ∧ 
  b_n 3 - b_n 0 = 6 ∧ 3 * d' = 2 ∧ 
  ∀ n, b_n n = 2 * n - 10)) := sorry

end arithmetic_sequence_general_formula_new_arithmetic_sequence_general_formula_l181_181572


namespace no_root_greater_than_4_l181_181844

lemma equation1_roots (x : ℝ) : 5 * x^2 - 15 = 35 ↔ x = real.sqrt 10 ∨ x = -real.sqrt 10 := 
sorry

lemma equation2_roots (x : ℝ) : (3 * x - 2)^2 = (2 * x - 3)^2 ↔ x = 1 ∨ x = -1 := 
sorry

lemma equation3_roots (x : ℝ) : real.sqrt (x^2 - 16) = real.sqrt (2 * x - 4) ↔ x = 4 ∨ x = -3 := 
sorry

theorem no_root_greater_than_4 :
  ∀ (x : ℝ), (5 * x^2 - 15 = 35 → x ≤ 4) ∧ ((3 * x - 2)^2 = (2 * x - 3)^2 → x ≤ 4) ∧ (real.sqrt (x^2 - 16) = real.sqrt (2 * x - 4) → x ≤ 4) := 
sorry

end no_root_greater_than_4_l181_181844


namespace problem_statement_l181_181582

noncomputable def f (x m : ℝ) : ℝ := real.sqrt (x^2 - 2 * x + m)

theorem problem_statement : 
  (∀ x, f x 0 = 0 → (x ≥ 2 ∨ x ≤ 0)) ∧ f 1 0 = 0 ∧ f 1 2 = 1
  ∧ (∀ x, f x 2 = real.sqrt ((x - 1)^2 + 1)) :=
sorry

end problem_statement_l181_181582


namespace problem_statement_l181_181928

def a : ℝ := 2^(-1 / 3)
def b : ℝ := Real.log 1 / Real.log 2
def c : ℝ := Real.log 4 / Real.log (1 / 3)

theorem problem_statement : c > a ∧ a > b :=
by
  sorry

end problem_statement_l181_181928


namespace reflection_over_y_eq_x_correct_l181_181108

def reflection_matrix_y_eq_x : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.ofVector 2 2 #[#[0, 1], #[1, 0]]

theorem reflection_over_y_eq_x_correct :
  reflection_matrix_y_eq_x = (λ i j, if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0) then 1 else 0) := by
  sorry

end reflection_over_y_eq_x_correct_l181_181108


namespace not_representative_l181_181450

variable (A B : Prop)

axiom has_email : A
axiom uses_internet : B
axiom dependent : A → B
axiom sample_size : Nat := 2000

theorem not_representative (hA : has_email) (hB : uses_internet) (hD : dependent) (hS : sample_size = 2000) : ¬(∀ x, A x) :=
  sorry

end not_representative_l181_181450


namespace find_modulus_l181_181576

open Complex -- Open the Complex namespace for convenience

noncomputable def modulus_of_z (a : ℝ) (h : (1 + 2 * Complex.I) * (a + Complex.I : ℂ) = Complex.re ((1 + 2 * Complex.I) * (a + Complex.I)) + Complex.im ((1 + 2 * Complex.I) * (a + Complex.I)) * Complex.I) : ℝ :=
  Complex.abs ((1 + 2 * Complex.I) * (a + Complex.I))

theorem find_modulus : modulus_of_z (-3) (by {
  -- Provide the condition that real part equals imaginary part
  admit -- This 'admit' serves as a placeholder for the proof of the condition 
}) = 5 * Real.sqrt 2 := sorry

end find_modulus_l181_181576


namespace piecewise_function_sum_of_a_b_l181_181961

theorem piecewise_function_sum_of_a_b
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = if x < 0 then a * x + b else sqrt x + 3)
  (h2 : ∀ x1 : ℝ, ∃! x2 : ℝ, f x1 = f x2)
  (h3 : f (2 * a) = f (3 * b)) :
  a + b = -sqrt 6 / 2 + 3 :=
sorry

end piecewise_function_sum_of_a_b_l181_181961


namespace range_of_a_l181_181217

theorem range_of_a (a b : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → x * abs (x - a) + b < 0) : 
  (b < -1 → 1 + b < a ∧ a < 1 - b) ∧ (-1 ≤ b ∧ b < 2 * sqrt 2 - 3 → 1 + b < a ∧ a < 2 * sqrt (-b)) := 
sorry

end range_of_a_l181_181217


namespace quadrilateral_perimeter_l181_181373

noncomputable def dist (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.fst - q.fst)^2 + (p.snd - q.snd)^2)

theorem quadrilateral_perimeter :
  let A' : ℝ × ℝ := (-5, 2)
  let B' : ℝ × ℝ := (0, 3)
  let C' : ℝ × ℝ := (7, 4)
  let B : ℝ × ℝ := (0, 0)
  in dist A' B' + dist B' C' + dist C' B + dist B A' = 3 + real.sqrt 26 + (5 * real.sqrt 2) :=
by sorry

end quadrilateral_perimeter_l181_181373


namespace family_members_l181_181228

theorem family_members (cost_purify : ℝ) (water_per_person : ℝ) (total_cost : ℝ) 
  (h1 : cost_purify = 1) (h2 : water_per_person = 1 / 2) (h3 : total_cost = 3) : 
  total_cost / (cost_purify * water_per_person) = 6 :=
by
  sorry

end family_members_l181_181228


namespace polynomial_real_root_l181_181866

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a * x^3 + x^2 + a * x - 1 = 0) ↔ a ∈ set.Iic (-1.5) :=
sorry

end polynomial_real_root_l181_181866


namespace find_constants_and_intervals_l181_181176

noncomputable def f (x a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

theorem find_constants_and_intervals {a b c : ℂ} 
  (h₁ : ∀ x, deriv (λ y, y^3 + a*y^2 + b*y + c) x = 3 * x^2 + 2 * a * x + b)
  (h₂ : deriv (λ x, x^3 + a * x^2 + b * x + c) 1 = 0)
  (h₃ : deriv (λ x, x^3 + a * x^2 + b * x + c) (-2/3) = 0)
  (h₄ : f (-1) a b c = 3 / 2) :
  a = -1 / 2 ∧ b = -2 ∧ c = 1 ∧ 
  (∀ x, x < -2/3 → deriv (λ x, x^3 + -1/2 * x^2 + -2 * x + 1) x = 3 * x^2 - x - 2 > 0) ∧ 
  (∀ x, x > 1 → deriv (λ x, x^3 + -1/2 * x^2 + -2 * x + 1) x = 3 * x^2 - x - 2 > 0) ∧ 
  (∀ x, -2/3 < x ∧ x < 1 → deriv (λ x, x^3 + -1/2 * x^2 + -2 * x + 1) x = 3 * x^2 - x - 2 < 0) ∧ 
  f (-2/3) (-1/2) (-2) 1 = 49 / 27 ∧ 
  f 1 (-1/2) (-2) 1 = -1 / 2 :=
by sorry

end find_constants_and_intervals_l181_181176


namespace bathroom_area_is_50_square_feet_l181_181041

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet_l181_181041


namespace sin_cos_15_deg_l181_181686

theorem sin_cos_15_deg : sin (15 * π / 180) * cos (15 * π / 180) = 1 / 4 :=
by
  -- Adding the known value of sin(30 degrees)
  have sin_30_deg : sin (30 * π / 180) = 1 / 2 := by norm_num
  
  -- Using the double angle formula
  calc
    sin (15 * π / 180) * cos (15 * π / 180)
        = (1 / 2) * sin (2 * 15 * π / 180) : by rw [← sin_bit0_mul_angle_iff1 (15 * π / 180)] -- double angle formula
    ... = (1 / 2) * sin (30 * π / 180) : by norm_num
    ... = (1 / 2) * (1 / 2) : by rw [sin_30_deg]
    ... = 1 / 4 : by norm_num

end sin_cos_15_deg_l181_181686


namespace part1_part2_part3_l181_181269

-- Define the function f
def f (x : ℝ) : ℝ := sin (2 * x - π / 6) - 2 * sin x ^ 2 + 1

-- Part (1): Prove that for f(α) = 1/2 and α ∈ [0, π/2], α = 0 or α = π/3
theorem part1 (α : ℝ) (h1 : α ∈ set.Icc 0 (π / 2)) (h2 : f α = 1 / 2) : α = 0 ∨ α = π / 3 :=
sorry

-- Part (2): Prove that for [f(x)]^2 + 2a cos(2x + π/6) - 2a - 2 < 0 for all x ∈ ( -π/12, π/6 )
theorem part2 (a : ℝ) (h : ∀ x : ℝ, x ∈ set.Ioo (-π / 12) (π / 6) → (f x) ^ 2 + 2 * a * cos (2 * x + π / 6) - 2 * a - 2 < 0) : a ∈ set.Ici (-1 / 2) :=
sorry

-- Part (3): Define g and prove the range for m
def g (m : ℝ) (x : ℝ) : ℝ := sin(2 * m * x + π / 3)

theorem part3 (λ m : ℝ) (h : ∀ x : ℝ, g m (x + λ) = λ * g m x) :
  m ∈ { k * π | k : ℤ ∧ k ≠ 0 } ∪ { (2 * n + 1) * π / 2 | n : ℤ } :=
sorry

end part1_part2_part3_l181_181269


namespace Haleigh_needs_leggings_l181_181586

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l181_181586


namespace collectors_edition_combined_l181_181507

def dina_has_60_dolls : Prop := Dina's dolls = 60
def dina_has_twice_as_many_as_ivy (Dina Ivy : ℕ) := Dina = 2 * Ivy
def ivy_has_10_more_than_luna (Ivy Luna : ℕ) := Ivy = Luna + 10
def ivy_collectors_edition_fraction (Ivy_collector Ivy : ℕ) := Ivy_collector = 2/3 * Ivy
def luna_collectors_edition_fraction (Luna_collector Luna : ℕ) := Luna_collector = 1/2 * Luna

theorem collectors_edition_combined (Dina Ivy Ivy_collector Luna Luna_collector : ℕ)
    (h₁ : dina_has_60_dolls)
    (h₂ : dina_has_twice_as_many_as_ivy Dina Ivy)
    (h₃ : ivy_has_10_more_than_luna Ivy Luna)
    (h₄ : ivy_collectors_edition_fraction Ivy_collector Ivy)
    (h₅ : luna_collectors_edition_fraction Luna_collector Luna) :
    Ivy_collector + Luna_collector = 30 :=
sorry

end collectors_edition_combined_l181_181507


namespace remainder_is_x_plus_2_l181_181535

noncomputable def problem_division := 
  ∀ x : ℤ, ∃ q r : ℤ, (x^3 + 2 * x^2) = q * (x^2 + 3 * x + 2) + r ∧ r < x^2 + 3 * x + 2 ∧ r = x + 2

theorem remainder_is_x_plus_2 : problem_division := sorry

end remainder_is_x_plus_2_l181_181535


namespace tickets_needed_l181_181253

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l181_181253


namespace triple_layers_area_l181_181742

-- Defining the conditions
def hall : Type := {x // x = 10 * 10}
def carpet1 : hall := ⟨60, sorry⟩ -- First carpet size: 6 * 8
def carpet2 : hall := ⟨36, sorry⟩ -- Second carpet size: 6 * 6
def carpet3 : hall := ⟨35, sorry⟩ -- Third carpet size: 5 * 7

-- The final theorem statement
theorem triple_layers_area : ∃ area : ℕ, area = 6 :=
by
  have intersection_area : ℕ := 2 * 3
  use intersection_area
  sorry

end triple_layers_area_l181_181742


namespace makarala_meetings_percentage_l181_181286

def work_day_to_minutes (hours: ℕ) : ℕ :=
  60 * hours

def total_meeting_time (first: ℕ) (second: ℕ) : ℕ :=
  let third := first + second
  first + second + third

def percentage_of_day_spent (meeting_time: ℕ) (work_day_time: ℕ) : ℚ :=
  (meeting_time : ℚ) / (work_day_time : ℚ) * 100

theorem makarala_meetings_percentage
  (work_hours: ℕ)
  (first_meeting: ℕ)
  (second_meeting: ℕ)
  : percentage_of_day_spent (total_meeting_time first_meeting second_meeting) (work_day_to_minutes work_hours) = 37.5 :=
by
  sorry

end makarala_meetings_percentage_l181_181286


namespace base6_to_base10_54123_l181_181762

def convert_base6_to_base10 (n6 : Nat) : Nat :=
  let d0 := 3 * 6^0
  let d1 := 2 * 6^1
  let d2 := 1 * 6^2
  let d3 := 4 * 6^3
  let d4 := 5 * 6^4
  d0 + d1 + d2 + d3 + d4

theorem base6_to_base10_54123 : convert_base6_to_base10 54123 = 7395 := by
  unfold convert_base6_to_base10
  simp
  sorry

end base6_to_base10_54123_l181_181762


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181897
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181897


namespace nesbitts_inequality_l181_181251

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end nesbitts_inequality_l181_181251


namespace log_sum_evaluation_l181_181095

theorem log_sum_evaluation :
  (1 / log 2 (12 * sqrt 5) + 1 / log 3 (12 * sqrt 5) + 1 / log 4 (12 * sqrt 5) + 
  1 / log 5 (12 * sqrt 5) + 1 / log 6 (12 * sqrt 5)) = 2 :=
sorry

end log_sum_evaluation_l181_181095


namespace lawn_length_is_correct_l181_181461

noncomputable def lawn_length 
  (width : ℝ) (road_width : ℝ) (cost : ℝ) (cost_per_sqm : ℝ) 
  (total_area : ℝ) : ℝ := 
  let intersection_area := road_width * road_width
  let area_road1 := road_width * total_area
  let area_road2 := road_width * width - intersection_area
  let combined_road_area := area_road1 + area_road2
  let calculated_area := cost / cost_per_sqm
  (calculated_area - combined_road_area) / road_width

theorem lawn_length_is_correct :
  ∀ (width road_width cost cost_per_sqm total_area length : ℝ),
  width = 60 ∧
  road_width = 10 ∧
  cost = 5200 ∧
  cost_per_sqm = 4 ∧
  total_area = 60 ∧
  lawn_length width road_width cost cost_per_sqm total_area = length →
  length = 80 :=
by
  intros
  rw [lawn_length]
  sorry

end lawn_length_is_correct_l181_181461


namespace sum_of_valid_k_equals_26_l181_181888

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181888


namespace time_to_paint_remaining_rooms_l181_181031

-- Definitions for the conditions
def total_rooms : ℕ := 11
def time_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Statement of the problem
theorem time_to_paint_remaining_rooms : 
  total_rooms - painted_rooms = 9 →
  (total_rooms - painted_rooms) * time_per_room = 63 := 
by 
  intros h1
  sorry

end time_to_paint_remaining_rooms_l181_181031


namespace vector_bound_l181_181342

open List

-- Step 1: Define the problem conditions
def isValidVector (v : List Int) (n : ℕ) : Prop :=
  v.length = n ∧ ∀ x, x ∈ v → x = -1 ∨ x = 0 ∨ x = 1

def noThreeSumToZero (V : List (List Int)) : Prop :=
  ∀ (u v w : List Int), u ∈ V → v ∈ V → w ∈ V → u ≠ v → v ≠ w → u ≠ w → u.zip v.zip w = repeat 0 (u.length)

-- Step 2: Define the proof statement
theorem vector_bound (V : List (List Int)) (n : ℕ) (hV : ∀ v ∈ V, isValidVector v n) (hNoThreeSumToZero : noThreeSumToZero V) :
  V.length ≤ 2 * 3 ^ (n - 1) :=
  sorry

end vector_bound_l181_181342


namespace positive_integer_m_divisors_l181_181539

theorem positive_integer_m_divisors : 
  ∃ (count : ℕ), count = 2 ∧ 
  ∀ m : ℕ, (0 < m) → (∃ k : ℕ, 180 = k * (m^2 - 3)) → (Hints.count := count) :=
sorry

end positive_integer_m_divisors_l181_181539


namespace no_real_roots_l181_181852

noncomputable def polynomial : Polynomial ℝ := Polynomial.C 8 + Polynomial.X * Polynomial.C (-4) + Polynomial.X^2

theorem no_real_roots : ¬ ∃ x : ℝ, polynomial.eval x polynomial = 0 := 
by
  sorry

end no_real_roots_l181_181852


namespace average_of_six_numbers_l181_181718

theorem average_of_six_numbers (A : ℝ) (x y z w u v : ℝ)
  (h1 : (x + y + z + w + u + v) / 6 = A)
  (h2 : (x + y) / 2 = 1.1)
  (h3 : (z + w) / 2 = 1.4)
  (h4 : (u + v) / 2 = 5) :
  A = 2.5 :=
by
  sorry

end average_of_six_numbers_l181_181718


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181912

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181912


namespace product_of_digits_in_base8_representation_of_7890_is_336_l181_181393

def base8_representation_and_product (n : ℕ) : ℕ × ℕ :=
  let digits := [1, 7, 2, 4, 6] in
  let product := digits.foldl (· * ·) 1 in 
  (digits.foldr (λ d acc, acc * 8 + d) 0, product)

theorem product_of_digits_in_base8_representation_of_7890_is_336 :
  ∀ (n : ℕ), n = 7890 → (base8_representation_and_product n).2 = 336 :=
by
  intros n h
  rw [← h]
  have := base8_representation_and_product 7890
  simp only [this]
  -- Here proof steps are skipped using sorry
  sorry

end product_of_digits_in_base8_representation_of_7890_is_336_l181_181393


namespace smallest_n_for_multiple_of_11_l181_181715

theorem smallest_n_for_multiple_of_11 
  (x y : ℤ) 
  (hx : x ≡ -2 [ZMOD 11]) 
  (hy : y ≡ 2 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ n = 7 :=
sorry

end smallest_n_for_multiple_of_11_l181_181715


namespace ellipse_major_axis_length_l181_181455

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l181_181455


namespace csc_210_eq_neg2_l181_181862

theorem csc_210_eq_neg2 :
  ∀ (degrees : ℝ),
  degrees = 210 →
  (∀ x, Real.csc x = 1 / Real.sin x) →
  Real.sin 210 = Real.sin (180 + 30) →
  Real.sin (180 + 30) = -Real.sin 30 →
  Real.sin 30 = 1 / 2 →
  Real.csc 210 = -2 := by
  sorry

end csc_210_eq_neg2_l181_181862


namespace parabola_transformation_zeros_sum_l181_181347

theorem parabola_transformation_zeros_sum :
  let y := fun x => (x - 3)^2 + 4
  let y_rotated := fun x => -(x - 3)^2 + 4
  let y_shifted_right := fun x => -(x - 7)^2 + 4
  let y_final := fun x => -(x - 7)^2 + 7
  ∃ a b, y_final a = 0 ∧ y_final b = 0 ∧ (a + b) = 14 :=
by
  sorry

end parabola_transformation_zeros_sum_l181_181347


namespace company_annual_income_l181_181042

variable {p a : ℝ}

theorem company_annual_income (h : 280 * p + (a - 280) * (p + 2) = a * (p + 0.25)) : a = 320 := 
sorry

end company_annual_income_l181_181042


namespace no_H2_from_CH4_C6H6_l181_181110

theorem no_H2_from_CH4_C6H6 (CH4 C6H6 : ℝ) (h1 : CH4 = 3) (h2 : C6H6 = 3) :
  ∀ r : chemical_reaction, r.reactants = ['CH4', 'C6H6'] → r.products = [] → r.H2 = 0 :=
by
  intros r hr1 hr2
  sorry

end no_H2_from_CH4_C6H6_l181_181110


namespace convex_polyhedron_has_even_face_l181_181317

-- Definitions for the conditions
variables {V : Type*} [DecidableEq V] -- V for vertices type with decidable equality
structure Face (V : Type*) :=
(edges : list (V × V))

structure ConvexPolyhedron (V : Type*) :=
(faces : list (Face V))
(is_convex : Prop)

variables (P : ConvexPolyhedron V)
variable (odd_faces : P.faces.length % 2 = 1)

-- The statement we need to prove
theorem convex_polyhedron_has_even_face (P : ConvexPolyhedron V) (odd_faces : P.faces.length % 2 = 1) :
  ∃ f ∈ P.faces, (f.edges.length % 2 = 0) :=
sorry

end convex_polyhedron_has_even_face_l181_181317


namespace mixture_contains_pecans_l181_181202

theorem mixture_contains_pecans 
  (price_per_cashew_per_pound : ℝ)
  (cashews_weight : ℝ)
  (price_per_mixture_per_pound : ℝ)
  (price_of_cashews : ℝ)
  (mixture_weight : ℝ)
  (pecans_weight : ℝ)
  (price_per_pecan_per_pound : ℝ)
  (pecans_price : ℝ)
  (total_cost_of_mixture : ℝ)
  
  (h1 : price_per_cashew_per_pound = 3.50) 
  (h2 : cashews_weight = 2)
  (h3 : price_per_mixture_per_pound = 4.34) 
  (h4 : pecans_weight = 1.33333333333)
  (h5 : price_per_pecan_per_pound = 5.60)
  
  (h6 : price_of_cashews = cashews_weight * price_per_cashew_per_pound)
  (h7 : mixture_weight = cashews_weight + pecans_weight)
  (h8 : pecans_price = pecans_weight * price_per_pecan_per_pound)
  (h9 : total_cost_of_mixture = price_of_cashews + pecans_price)

  (h10 : price_per_mixture_per_pound = total_cost_of_mixture / mixture_weight)
  
  : pecans_weight = 1.33333333333 :=
sorry

end mixture_contains_pecans_l181_181202


namespace esha_time_difference_l181_181071

-- Definitions based on conditions
def cycling_rate_minutes_per_mile := 135 / 18
def strolling_rate_minutes_per_mile := 300 / 12

-- The theorem to prove the result
theorem esha_time_difference :
  strolling_rate_minutes_per_mile - cycling_rate_minutes_per_mile = 17.5 :=
by
  sorry

end esha_time_difference_l181_181071


namespace ratio_inradius_circumradius_l181_181551

noncomputable def hyperbola (a b : ℝ) : set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 / a^2) - (P.2^2 / b^2) = 1}

theorem ratio_inradius_circumradius (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (e : ℝ)
  (h_e : e = Real.sqrt 3)
  (P : ℝ × ℝ) (h_P : P ∈ hyperbola a b) (F1 F2 : ℝ × ℝ)
  (h_dot : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0) :
  let c := a * e in
  ∃ R r : ℝ,
  R = c ∧
  r = (2 * b^2) / (Real.sqrt (4 * a^2 + 8 * b^2) + 2 * c) ∧
  r / R = Real.sqrt 15 / 3 - 1 :=
by
  sorry

end ratio_inradius_circumradius_l181_181551


namespace jar_prob_nickel_l181_181429

theorem jar_prob_nickel 
    (value_dimes : ℚ) 
    (value_nickels : ℚ) 
    (value_pennies : ℚ)
    (dime_value : ℚ) 
    (nickel_value : ℚ) 
    (penny_value : ℚ) 
    (prob_select_nickel : ℚ) :
    value_dimes = 8 ∧ 
    value_nickels = 5 ∧ 
    value_pennies = 3 ∧ 
    dime_value = 0.1 ∧ 
    nickel_value = 0.05 ∧ 
    penny_value = 0.01 ∧ 
    prob_select_nickel = 5 / 24 :=
begin
    sorry
end

end jar_prob_nickel_l181_181429


namespace sum_of_solutions_sum_of_all_solutions_l181_181006

theorem sum_of_solutions (x : ℝ) (h : x = |2 * x - |60 - 2 * x||) : x = 12 ∨ x = 20 ∨ x = 60 :=
  sorry

theorem sum_of_all_solutions : 
  let solutions := {x | x = 12 ∨ x = 20 ∨ x = 60}
  ∃ (S : ℝ), S = ∑ x in solutions, x ∧ S = 92 :=
  sorry

end sum_of_solutions_sum_of_all_solutions_l181_181006


namespace radius_of_roots_on_circle_l181_181820

theorem radius_of_roots_on_circle : 
  ∀ z : ℂ, (z + 2)^6 = 64 * z^6 → abs (z + 2) = 2 * abs z → 
  (∃ r : ℝ, r = 2 / real.sqrt 3) :=
begin
  intros z h1 h2,
  use 2 / real.sqrt 3,
  sorry
end

end radius_of_roots_on_circle_l181_181820


namespace area_of_rectangle_is_32_proof_l181_181470

noncomputable def triangle_sides : ℝ := 7.3 + 5.4 + 11.3
def equality_of_perimeters (rectangle_length rectangle_width : ℝ) : Prop := 
  2 * (rectangle_length + rectangle_width) = triangle_sides

def rectangle_length (rectangle_width : ℝ) : ℝ := 2 * rectangle_width

def area_of_rectangle_is_32 (rectangle_width : ℝ) : Prop :=
  rectangle_length rectangle_width * rectangle_width = 32

theorem area_of_rectangle_is_32_proof : 
  ∃ (rectangle_width : ℝ), 
  equality_of_perimeters (rectangle_length rectangle_width) rectangle_width ∧ area_of_rectangle_is_32 rectangle_width :=
by
  sorry

end area_of_rectangle_is_32_proof_l181_181470


namespace systematic_sampling_correct_l181_181368

noncomputable def total_products : ℕ := 50
noncomputable def sample_count : ℕ := 5
noncomputable def sampling_interval : ℕ := total_products / sample_count

theorem systematic_sampling_correct :
  ∃ a : ℕ, ∃ interval : ℕ, interval = sampling_interval ∧
  (a = 9 ∧
  (a + interval = 19) ∧
  (a + 2 * interval = 29) ∧
  (a + 3 * interval = 39) ∧
  (a + 4 * interval = 49)) :=
begin
  sorry
end

end systematic_sampling_correct_l181_181368


namespace integral_abs_x_minus_1_l181_181091

theorem integral_abs_x_minus_1 :
  ∫ x in -1..1, (|x| - 1) = -1 :=
by
  sorry

end integral_abs_x_minus_1_l181_181091


namespace cos_phi_is_correct_l181_181799

variables (u v : EuclideanSpace ℝ (Fin 3))

def vector1 : EuclideanSpace ℝ (Fin 3) := ![3, 2, 1]
def vector2 : EuclideanSpace ℝ (Fin 3) := ![2, -2, -1]

def diagonal1 := vector1 + vector2
def diagonal2 := vector2 - vector1

def dot_product_diagonals := (diagonal1 ⬝ diagonal2)
def magnitude_diagonal1 := ∥diagonal1∥
def magnitude_diagonal2 := ∥diagonal2∥

noncomputable def cos_angle_between_diagonals := dot_product_diagonals / (magnitude_diagonal1 * magnitude_diagonal2)

theorem cos_phi_is_correct :
  cos_angle_between_diagonals = -1 / Real.sqrt 21 :=
sorry

end cos_phi_is_correct_l181_181799


namespace find_sin_A_l181_181141

theorem find_sin_A 
  (a b : ℝ) 
  (A B C : ℝ) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sum_angles : A + B + C = Real.pi) 
  (h_angle_relation : A + C = 2 * B) : 
  Real.sin A = 1 / 2 :=
by
  -- Proof steps would go here.
  sorry

end find_sin_A_l181_181141


namespace not_representative_l181_181451

variable (A B : Prop)

axiom has_email : A
axiom uses_internet : B
axiom dependent : A → B
axiom sample_size : Nat := 2000

theorem not_representative (hA : has_email) (hB : uses_internet) (hD : dependent) (hS : sample_size = 2000) : ¬(∀ x, A x) :=
  sorry

end not_representative_l181_181451


namespace d_is_integer_for_all_l181_181086

def d : ℕ → ℕ → ℕ
| n, 0 => 1
| n, m if n = m => 1
| n, m if 0 < m ∧ m < n => (d (n - 1) m) + (2 * n - m) * (d (n - 1) (m - 1))


-- Prove that d(n, m) are integers for all m, n ∈ ℕ.

theorem d_is_integer_for_all (n m : ℕ) : ∃ d : ℕ → ℕ → ℕ, d n m = 1 ∨ (md (d n m) = md (d (n - 1) m) + (2 * n - m) * (d (n - 1) (m - 1)) ∧ (n ≥ 0) ∧ (0 < m < n) ) := 
sorry

end d_is_integer_for_all_l181_181086


namespace true_propositions_l181_181577

-- Define the propositions
def proposition_1 (α β : Plane) : Prop :=
  (α ⊥ β) → ∃ m : Line, (m ⊆ α) ∧ (m ∥ β)

def proposition_3 (α β γ : Plane) (l : Line) : Prop :=
  (α ⊥ γ) ∧ (β ⊥ γ) ∧ (α ∩ β = l) → l ⊥ γ

def proposition_4 (α β γ : Plane) (m : Line) : Prop :=
  (α ∥ β) ∧ (m ⊥ α) ∧ (β ∥ γ) → m ⊥ γ

-- Lean statement verifying that propositions 1, 3, and 4 are true
theorem true_propositions (α β γ : Plane) (l m : Line) :
  (proposition_1 α β) ∧ (proposition_3 α β γ l) ∧ (proposition_4 α β γ m) :=
by
  sorry

end true_propositions_l181_181577


namespace rate_of_interest_l181_181797

def principal_B : ℝ := 5000
def time_B : ℝ := 2  -- years
def principal_C : ℝ := 3000
def time_C : ℝ := 4  -- years
def total_interest : ℝ := 1980

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest principal_B R time_B + simple_interest principal_C R time_C = total_interest ∧ R = 9 := by
  sorry

end rate_of_interest_l181_181797


namespace solve_equation_l181_181712

theorem solve_equation (x y z : ℝ) (m n : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1 / (sin^2 x ))^3 + (cos^2 y + 1 / (cos^2 y))^3 = 16 * cos z) →
  (cos z = 1) ∧ 
  (∃ m : ℤ, x = π / 2 + π * m) ∧ 
  (∃ n : ℤ, y = π * n) ∧ 
  (∃ m : ℤ, z = 2 * π * m) :=
by
  intros hsin_cos hcos_sin heq
  sorry

end solve_equation_l181_181712


namespace probability_meeting_of_C_and_D_l181_181661

noncomputable def probability_meeting (C_start D_start : (ℕ × ℕ)) (steps : ℕ) : ℝ :=
  let paths_C := 2^steps
  let paths_D := 2^steps in
  (Real.toNNReal (Finset.sum (Finset.range (steps + 1)) (λ i, (Nat.choose steps i) * (Nat.choose steps (i + 1))))) / (paths_C * paths_D)

theorem probability_meeting_of_C_and_D : 
  probability_meeting (0, 0) (6, 8) 7 ≈ 0.0111 :=
by
  sorry

end probability_meeting_of_C_and_D_l181_181661


namespace circle_tangency_l181_181085

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

def triangle_sides (A B C : A) : ℝ :=
  dist A B + dist B C + dist C A

def circles_radii (a b c : ℝ) : Prop :=
  let s := (a + b + c) / 2 in
  let r1 := (b + c - a) / 2 in
  let r2 := (a + c - b) / 2 in
  let r3 := (a + b - c) / 2 in
  s = r1 + r2 + r3

theorem circle_tangency (a b c : ℝ) :
  ∃ r1 r2 r3 : ℝ,
    circles_radii a b c ∧
    ((r1 = (b + c - a) / 2) ∧ (r2 = (a + c - b) / 2) ∧ (r3 = (a + b - c) / 2)) ∨
    ((r3 = (a + b + c) / 2) ∧ (r1 = (a + c - b) / 2) ∧ (r2 = (b + c - a) / 2)) :=
by
  sorry

end circle_tangency_l181_181085


namespace football_club_balance_l181_181044

theorem football_club_balance
  (initial_balance : ℕ)
  (price_per_player_sold : ℕ)
  (num_players_sold : ℕ)
  (price_per_player_bought : ℕ)
  (final_balance : ℕ)
  (sale_amount := num_players_sold * price_per_player_sold)
  (new_balance := initial_balance + sale_amount)
  (num_players_bought : ℕ)
  (purchase_amount := num_players_bought * price_per_player_bought) :
  initial_balance = 100 ∧
  price_per_player_sold = 10 ∧
  num_players_sold = 2 ∧
  price_per_player_bought = 15 ∧
  final_balance = 60 ∧
  new_balance - purchase_amount = final_balance →
  num_players_bought = 4 :=
begin
  sorry
end

end football_club_balance_l181_181044


namespace correct_track_length_l181_181666

noncomputable def track_length : ℕ :=
  let x := 330 in
  if 120 + (x - 120) = x ∧ (x - 120) + (210 + 120) = 2 * x
  then x
  else 0

theorem correct_track_length (x : ℕ) (h1 : 120 + (x - 120) = x) (h2 : (x - 120) + 210 = 2 * x - 90) : x = 330 :=
  have h_valid1 : 120 + (x - 120) = x := h1
  have h_valid2 : (x - 120) + 210 = 2 * x - 90 := h2
  by
  have h3 : x^2 - 330 * x = 0 := by sorry
  have h4 : x(x - 330) = 0 := by sorry
  have h5 : x ≠ 0 := by sorry
  show x = 330 from by sorry

end correct_track_length_l181_181666


namespace range_of_x_positive_l181_181125

noncomputable def f (x : ℝ) := x^(2/3) - x^(-1/2)

theorem range_of_x_positive : {x : ℝ | f x > 0} = {x : ℝ | 1 < x} :=
by
  sorry

end range_of_x_positive_l181_181125


namespace floor_ceil_eval_l181_181859

theorem floor_ceil_eval :
  (Int.floor (-3.67) + Int.ceil 30.2) = 27 :=
by
  sorry

end floor_ceil_eval_l181_181859


namespace find_natural_number_x_satisfying_equation_l181_181413

theorem find_natural_number_x_satisfying_equation :
  ∃ x : ℕ, x^3 = 2011^2 + 2011 * 2012 + 2012^2 + 2011^3 ∧ x = 2012 :=
by
  use 2012
  split
  { calc 2012^3
        = (2011 + 1)^3 : by rw [pow_succ, pow_succ, pow_one]
    ... = 2011^3 + 3 * (2011^2 * 1) + 3 * (2011 * 1^2) + 1^3 : by rw [add_pow_3]
    ... = 2011^3 + 3 * (2011^2 * 1) + 3 * (2011 * 1) + 1 : by norm_num
    ... = 2011^3 + 2011^2 + 2011 * 2012 + 2012^2 + 2011^2 + 2011*2012 + 2012^2 : by norm_num
    ... = 2011^3 + 2011^2 + 2011 * 2012 + 2012^2 : by ring }
  { refl }

end find_natural_number_x_satisfying_equation_l181_181413


namespace grandchildren_ages_l181_181858

theorem grandchildren_ages :
  ∃ (M T J K I V : ℕ),
    M = T + 8 ∧
    V = I + 7 ∧
    M = J + 1 ∧
    K = T + 11 ∧
    J = I + 4 ∧
    T + J = 13 ∧
    M = 11 ∧
    T = 3 ∧
    J = 10 ∧
    K = 14 ∧
    I = 6 ∧
    V = 13 :=
by
  use 11, 3, 10, 14, 6, 13
  simp
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num;
  split; norm_num
  sorry

end grandchildren_ages_l181_181858


namespace max_triangle_perimeter_l181_181062

theorem max_triangle_perimeter (x : ℕ) (h1 : 8 + 15 > x) (h2 : 8 + x > 15) (h3 : x > 7) : 8 + 15 + x ≤ 45 :=
by
  have h4 : x ≤ 22, from Nat.lt_of_add_lt_add_right h1
  have h5 : x ≥ 8, from Nat.succ_le_iff.mpr h3
  have h6 : x = 22, from (Nat.le_antisymm h4 h5)
  rw [h6]
  exact le_refl (8 + 15 + 22)


end max_triangle_perimeter_l181_181062


namespace find_angle_l181_181338

-- Define the conditions
variables (x : ℝ)

-- Conditions given in the problem
def angle_complement_condition (x : ℝ) := (10 : ℝ) + 3 * x
def complementary_condition (x : ℝ) := x + angle_complement_condition x = 90

-- Prove that the angle x equals to 20 degrees
theorem find_angle : (complementary_condition x) → x = 20 := 
by
  -- Placeholder for the proof
  sorry

end find_angle_l181_181338


namespace cookie_weight_l181_181193

theorem cookie_weight :
  ∀ (pounds_per_box cookies_per_box ounces_per_pound : ℝ),
    pounds_per_box = 40 →
    cookies_per_box = 320 →
    ounces_per_pound = 16 →
    (pounds_per_box * ounces_per_pound) / cookies_per_box = 2 := 
by 
  intros pounds_per_box cookies_per_box ounces_per_pound hpounds hcookies hounces
  rw [hpounds, hcookies, hounces]
  norm_num

end cookie_weight_l181_181193


namespace collinear_on_circumcenter_l181_181730

theorem collinear_on_circumcenter (A B C C_1 A_1 B_1 B_2 B_3 A_3 C_3 : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [AddCommGroup C_1] [AddCommGroup A_1] [AddCommGroup B_1] 
  [AddCommGroup B_2] [AddCommGroup B_3] [AddCommGroup A_3] [AddCommGroup C_3]
  (incircle : triangle ABC → Type) 
  (touches_AB : incircle ABC AB = C_1)
  (touches_BC : incircle ABC BC = A_1)
  (touches_CA : incircle ABC CA = B_1)
  (sym_B1 : symmetric B_1 (line A_1 C_1) = B_2)
  (int_BB2 : intersect_lines B B_2 AC = B_3)
  (int_points_analogous : defined_analogous A_3 C_3 B_3)
  (circumcenter_pass_through : on_line (A_3, B_3, C_3) (circumcenter (triangle ABC))) :
  collinear A_3 B_3 C_3 :=
sorry

end collinear_on_circumcenter_l181_181730


namespace find_m_l181_181270

-- Let m be a real number such that m > 1 and
-- \sum_{n=1}^{\infty} \frac{3n+2}{m^n} = 2.
theorem find_m (m : ℝ) (h1 : m > 1) 
(h2 : ∑' n : ℕ, (3 * (n + 1) + 2) / m^(n + 1) = 2) : 
  m = 3 :=
sorry

end find_m_l181_181270


namespace quadratic_real_roots_l181_181995

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l181_181995


namespace num_terms_divisible_by_b_eq_gcd_l181_181677

theorem num_terms_divisible_by_b_eq_gcd (a b d : ℕ) (h_gcd : Nat.gcd a b = d) :
  (∃ count : ℕ, count = d ∧ ∀ k, (1 ≤ k ∧ k ≤ b) → (a * k) % b = 0 → k = (b / d) * i for some i : ℕ) :=
sorry

end num_terms_divisible_by_b_eq_gcd_l181_181677


namespace compute_expression_l181_181080

theorem compute_expression : 20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1/5 :=
by
  sorry

end compute_expression_l181_181080


namespace pictures_vertically_l181_181282

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end pictures_vertically_l181_181282


namespace max_trig_expr_l181_181532

theorem max_trig_expr (x y z : ℝ) :
  (\sin (2 * x) + \sin y + \sin (3 * z)) * (\cos (2 * x) + \cos y + \cos (3 * z)) ≤ 9 / 2 :=
sorry

end max_trig_expr_l181_181532


namespace range_of_k_l181_181131

theorem range_of_k :
  ∀ (k : ℝ),
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 2*y + 1 = 0 ∧ abs ((3*x - 4*y + k) / sqrt (3^2 + (-4)^2)) = 1)
    ↔ (k ∈ set.Ioo (-17:ℝ) (-7) ∪ set.Ioo (3) (13)) :=
by
  sorry

end range_of_k_l181_181131


namespace radius_of_circle_roots_l181_181821

noncomputable def radius_of_circle : ℝ :=
  ∃ z : ℂ, ((z + 2)^6 = 64 * z^6) ∧ 
           (∀ z', ((z' + 2)^6 = 64 * z'^6) → ‖z' + 2‖ = 2 * ‖z'‖) → 
           ‖z + 2‖ = 2 * ‖z‖ ∧ radius = 2 / 3

theorem radius_of_circle_roots :
  ∀ z : ℂ, ((z + 2)^6 = 64 * z^6) →  ∃ radius : ℝ, radius = 2 / 3 := sorry

end radius_of_circle_roots_l181_181821


namespace kelly_snacks_l181_181255

theorem kelly_snacks (peanuts raisins : ℝ) (h_peanuts : peanuts = 0.1) (h_raisins : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end kelly_snacks_l181_181255


namespace series_sum_l181_181657

noncomputable def S (n : ℕ) : ℝ := 2^(n + 1) + n - 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem series_sum : 
  ∑' i, a i / 4^i = 4 / 3 :=
by 
  sorry

end series_sum_l181_181657


namespace binomial_sum_sum_of_binomial_solutions_l181_181901

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181901


namespace radius_of_circle_roots_l181_181822

noncomputable def radius_of_circle : ℝ :=
  ∃ z : ℂ, ((z + 2)^6 = 64 * z^6) ∧ 
           (∀ z', ((z' + 2)^6 = 64 * z'^6) → ‖z' + 2‖ = 2 * ‖z'‖) → 
           ‖z + 2‖ = 2 * ‖z‖ ∧ radius = 2 / 3

theorem radius_of_circle_roots :
  ∀ z : ℂ, ((z + 2)^6 = 64 * z^6) →  ∃ radius : ℝ, radius = 2 / 3 := sorry

end radius_of_circle_roots_l181_181822


namespace gcd_72_and_120_l181_181880

theorem gcd_72_and_120 : Nat.gcd 72 120 = 24 := 
by
  sorry

end gcd_72_and_120_l181_181880


namespace interest_rate_for_first_part_l181_181059

def sum_amount : ℝ := 2704
def part2 : ℝ := 1664
def part1 : ℝ := sum_amount - part2
def rate2 : ℝ := 0.05
def years2 : ℝ := 3
def interest2 : ℝ := part2 * rate2 * years2
def years1 : ℝ := 8

theorem interest_rate_for_first_part (r1 : ℝ) :
  part1 * r1 * years1 = interest2 → r1 = 0.03 :=
by
  sorry

end interest_rate_for_first_part_l181_181059


namespace calculate_total_marks_l181_181631

def total_marks (sec1_questions sec1_correct_ratio sec1_partial_ratio sec1_partial_marks
                 sec2_questions sec2_correct_ratio sec2_partial_ratio sec2_partial_marks sec2_negative_marks
                 sec3_questions sec3_correct_ratio sec3_wrong_ratio sec3_negative_marks
                 sec4_questions sec4_correct_ratio sec4_partial_ratio sec4_partial_marks : ℕ) : ℝ :=
  let sec1_full_marks := sec1_questions * sec1_correct_ratio
  let sec1_partial_marks_total := (sec1_questions * sec1_partial_ratio) * sec1_partial_marks

  let sec2_full_marks := sec2_questions * sec2_correct_ratio
  let sec2_partial_marks_total := (sec2_questions * sec2_partial_ratio) * sec2_partial_marks
  let sec2_wrong_marks := (sec2_questions - sec2_full_marks - sec2_partial_marks_total) * sec2_negative_marks

  let sec3_full_marks := sec3_questions * sec3_correct_ratio
  let sec3_wrong_marks := (sec3_questions * sec3_wrong_ratio) * sec3_negative_marks

  let sec4_full_marks := sec4_questions * sec4_correct_ratio
  let sec4_partial_marks_total := (sec4_questions * sec4_partial_ratio) * sec4_partial_marks

  sec1_full_marks + sec1_partial_marks_total +
  sec2_full_marks + sec2_partial_marks_total - sec2_wrong_marks +
  sec3_full_marks - sec3_wrong_marks +
  sec4_full_marks + sec4_partial_marks_total

theorem calculate_total_marks :
  total_marks 50 0.82 0.18 0.5
              60 0.75 0.25 (1/3) 0.25
              30 0.9 0.1 0.25
              40 0.95 0.05 0.5 = 157 := by
    sorry

end calculate_total_marks_l181_181631


namespace find_t_l181_181634

-- Definitions from the given conditions
def earning (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

-- The main theorem based on the translated problem
theorem find_t
  (t : ℕ)
  (h1 : earning (t - 4) (3 * t - 7) = earning (3 * t - 12) (t - 3)) :
  t = 4 := 
sorry

end find_t_l181_181634


namespace evaluate_operations_l181_181402

theorem evaluate_operations : 
  (-2)^(2) = -4 ∧ (sqrt 9 ≠ 3) ∧ ((-2)^3 ≠ 8) ∧ (-|(-3)| ≠ 3) :=
by 
  sorry

end evaluate_operations_l181_181402


namespace deal_or_no_deal_min_eliminations_l181_181612

theorem deal_or_no_deal_min_eliminations (n_boxes : ℕ) (n_high_value : ℕ) 
    (initial_count : n_boxes = 26)
    (high_value_count : n_high_value = 9) :
  ∃ (min_eliminations : ℕ), min_eliminations = 8 ∧
    ((n_boxes - min_eliminations - 1) / 2) ≥ n_high_value :=
sorry

end deal_or_no_deal_min_eliminations_l181_181612


namespace total_order_cost_l181_181477

theorem total_order_cost (n : ℕ) (cost_geo cost_eng : ℝ)
  (h1 : n = 35)
  (h2 : cost_geo = 10.50)
  (h3 : cost_eng = 7.50) :
  n * cost_geo + n * cost_eng = 630 := by
  -- proof steps should go here
  sorry

end total_order_cost_l181_181477


namespace point_Q_coords_l181_181669

-- Definition for point P
def point_P := (-3, 4)

-- Translation function
def translate_down (point : (ℤ × ℤ)) (units : ℤ) : (ℤ × ℤ) :=
  (point.1, point.2 - units)

def translate_right (point : (ℤ × ℤ)) (units : ℤ) : (ℤ × ℤ) :=
  (point.1 + units, point.2)

-- Point Q obtained by translating P
def point_Q := translate_right (translate_down point_P 3) 2

-- Theorem to prove coordinates of point Q are (-1, 1)
theorem point_Q_coords : point_Q = (-1, 1) :=
  by
    sorry

end point_Q_coords_l181_181669


namespace plot_length_is_63_total_expense_with_B_in_3_years_l181_181348

noncomputable def plot_breadth := 37 -- The breadth we calculated
noncomputable def plot_length := plot_breadth + 26

noncomputable def perimeter := 4 * plot_breadth + 52
noncomputable def cost_A_per_meter := 26.50
noncomputable def total_cost_A := 5300
noncomputable def annual_increase_A := 0.05
noncomputable def annual_increase_B := 0.03
noncomputable def cost_B_per_meter := 32.75

-- Calculate the perimeter using total cost with Material A
theorem plot_length_is_63 (b : ℝ) (p : ℝ) (cost_A : ℝ):
  5300 = 26.50 * p → 
  p = 4 * b + 52 → 
  b = 37 → 
  p = 200 → 
  plot_breadth = 37 → 
  plot_length = 63 := by
  sorry

-- Calculate the total cost in 3 years with Material B
theorem total_expense_with_B_in_3_years (p : ℝ) :
  p = 200 → 
  cost_B_per_meter = 32.75 →
  annual_increase_B = 0.03 →
  total_expense := 200 * (32.75 * (1 + 0.03)^3) →
  total_expense ≈ 7156 := by
  sorry

end plot_length_is_63_total_expense_with_B_in_3_years_l181_181348


namespace monotonicity_m0_m_range_condition_l181_181580

open Real

def f (x : ℝ) (m : ℝ) : ℝ := x / exp x - m * x

theorem monotonicity_m0 :
  (∀ x : ℝ, (x < 1 → deriv (f x 0) x > 0) ∧
            (x = 1 → deriv (f x 0) x = 0) ∧
            (x > 1 → deriv (f x 0) x < 0)) := by
  sorry

theorem m_range_condition (a b m : ℝ) (ha : a > 0) (hb : b > a)
  (hm : m ≤ -1 - 1 / exp 2) :
  (f b m - f a m) / (b - a) > 1 := by
  sorry

end monotonicity_m0_m_range_condition_l181_181580


namespace kinetic_energy_of_cylinder_l181_181529

variables (h ω ρ₀ k R : ℝ)
variables (hr : 0 ≤ R)

-- Let the kinetic energy be given by the expression
noncomputable def kinetic_energy := π * h * ω^2 * (ρ₀ * R^4 / 4 + k * R^5 / 5)

-- Statement of the theorem
theorem kinetic_energy_of_cylinder 
  (h : ℝ) (ω : ℝ) (ρ₀ : ℝ) (k : ℝ) (R : ℝ) (hr : 0 ≤ R) : 
  let E := π * h * ω^2 * (ρ₀ * R^4 / 4 + k * R^5 / 5) in
  E = π * h * ω^2 * (ρ₀ * R^4 / 4 + k * R^5 / 5) :=
  sorry

end kinetic_energy_of_cylinder_l181_181529


namespace largest_even_integer_sum_l181_181362

theorem largest_even_integer_sum (sum_of_integers : ℕ) (n : ℕ) (seq : ℕ → ℤ) (h_sum : (finset.range n).sum seq = sum_of_integers)
  (h_seq : ∀ i : ℕ, i < n → seq i = 2 * i + seq 0) :
  (seq (n-1) = 429) ↔ sum_of_integers = 12000 ∧ n = 30 :=
begin
  sorry
end

end largest_even_integer_sum_l181_181362


namespace C_cartesian_l_rectangular_intersect_range_l181_181244

-- Definitions for the parametric equations and polar coordinate equation.
@[simp] def curve_C (t : ℝ) : ℝ × ℝ := (1 * (2 + t) / 6, sqrt t) -- parametric equations: x = (2 + t) / 6, y = sqrt t

-- Polar coordinate equation of line l: ρ * cos (5π/6 - θ) - m = 0
@[simp] def line_l (ρ θ : ℝ) (m : ℝ) : Prop := ρ * cos (5 * π / 6 - θ) - m = 0

-- Cartesian equation of curve C: y^2 = 6x - 2, y ≥ 0
theorem C_cartesian (x y : ℝ) : (∃ t : ℝ, x = (2 + t) / 6 ∧ y = sqrt t) ↔ y ^ 2 = 6 * x - 2 ∧ y ≥ 0 := sorry

-- Rectangular coordinate equation of line l: y = sqrt(3) * x + 2m
theorem l_rectangular (x y m : ℝ): 
  (∃ ρ θ : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ * cos (5 * π / 6 - θ) - m = 0) ↔ y = sqrt 3 * x + 2 * m := sorry

-- Range of m for line l to intersect curve C at two distinct points:
-- -√3/6 ≤ m < √3/12
theorem intersect_range (m : ℝ) : 
  (∀ t : ℝ, t ≥ 0 → ∃ x y : ℝ, (x = (2 + t) / 6) ∧ (y = sqrt t) ∧ (y = sqrt 3 * x + 2 * m)) ↔ 
  (-sqrt 3 / 6) ≤ m ∧ m < (sqrt 3 / 12) := sorry

end C_cartesian_l_rectangular_intersect_range_l181_181244


namespace area_ratio_circumcircle_l181_181458

theorem area_ratio_circumcircle (a b c t_a t_b t_c : ℝ) (P : Point)
  (h1 : P ∈ arc AB) 
  (h2 : arc_does_not_contain_C P) 
  (h3 : t_a = area_triangle B C P) 
  (h4 : t_b = area_triangle A C P) 
  (h5 : t_c = area_triangle A B P) :
  \frac{a^2}{t_a} + \frac{b^2}{t_b} = \frac{c^2}{t_c} :=
sorry

end area_ratio_circumcircle_l181_181458


namespace find_x_l181_181916

theorem find_x (x y : ℕ) (h1 : y = 144) (h2 : x^3 * 6^2 / 432 = y) : x = 12 := 
by
  sorry

end find_x_l181_181916


namespace proof_problem_l181_181738

def star (a b : ℕ) : ℕ := a - a / b

theorem proof_problem : star 18 6 + 2 * 6 = 27 := 
by
  admit  -- proof goes here

end proof_problem_l181_181738


namespace irreducible_fractions_count_l181_181998

theorem irreducible_fractions_count :
  (∃ count, count = ∑ n in (1 : ℕ)..2017, if Nat.gcd n (n + 4) = 1 then 1 else 0) ∧
  (∑ n in (1 : ℕ)..2017, if Nat.gcd n (n + 4) = 1 then 1 else 0 = 1009) :=
by
  sorry

end irreducible_fractions_count_l181_181998


namespace ratio_heartsuit_eq_l181_181212

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem ratio_heartsuit_eq :
  (3 \heartsuit 5) / (5 \heartsuit 3) = 5 / 3 :=
by
  sorry

end ratio_heartsuit_eq_l181_181212


namespace expression_divisible_by_9_for_any_int_l181_181320

theorem expression_divisible_by_9_for_any_int (a b : ℤ) : 9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) := 
by 
  sorry

end expression_divisible_by_9_for_any_int_l181_181320


namespace calculate_fx_l181_181595

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem calculate_fx (x : ℝ) : f (x + 1) - f x = 2 * f x := 
by
  rw [f, f]
  unfold f
  sorry

end calculate_fx_l181_181595


namespace max_value_abs_expr_l181_181272

theorem max_value_abs_expr (w : ℂ) (h : abs w = 2) :
  ∃ a, abs ((w - 2) ^ 2 * (w + 2)) = 16 * real.sqrt 2 :=
sorry

end max_value_abs_expr_l181_181272


namespace math_problem_l181_181963

theorem math_problem 
  (m : ℤ) 
  (hm : f(x) = x^(-2 * m^2 + m + 3)) 
  (heven : ∀ x : ℝ, f(x) = f(-x)) 
  (hincreasing : ∀ x x' : ℝ, 0 < x ∧ x < x' → f(x) < f(x')) 
  (a : ℝ) 
  (ha_pos : 0 < a) 
  (ha_ne_one : a ≠ 1) 
  (hg : g(x) = log a (f x - a * x)) : 
  ((m = 1) ∧ (f(x) = x^2) ∧ (∃ (a : ℝ), a = (-3 + 3 * sqrt 5) / 2 ∧ (∀ x ∈ Icc (2:ℝ) (3:ℝ), g(x) ≤ 2))) :=
by
  sorry

end math_problem_l181_181963


namespace solve_trig_equation_l181_181694

theorem solve_trig_equation (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) → 
  (cos y ≠ 0) → 
  ((sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z) →
  (∃ (n k m : ℤ), x = (π / 2) + π * n ∧ y = π * k ∧ z = 2 * π * m) := 
by 
  sorry

end solve_trig_equation_l181_181694


namespace percentage_increase_l181_181218

theorem percentage_increase (d : ℝ) (h1 : 2 * d = 520) (h2 : d * 2 / 2 ∂₃38 ±) :
  (338 - d) / d * 100 = 30 := by
  -- Proof goes here
  sorry

end percentage_increase_l181_181218


namespace final_amount_is_23593_l181_181452

noncomputable def final_amount_after_bets 
  (initial_amount : ℝ) 
  (num_bets : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (bet_results : List Bool) 
  (win_increase : ℝ) 
  (loss_decrease : ℝ) : ℝ :=
bet_results.foldl 
  (fun current_amount bet ->
    if bet then current_amount * win_increase 
    else current_amount * loss_decrease) 
  initial_amount

theorem final_amount_is_23593 : 
  final_amount_after_bets 100 6 4 2 [true, false, true, true, false, true] 1.6 0.6 = 235.93 :=
sorry

end final_amount_is_23593_l181_181452


namespace perfect_squares_multiple_of_36_l181_181199

theorem perfect_squares_multiple_of_36 (N : ℕ) (h1 : N = 99) : 
  {k | k ∈ ℕ ∧ k^2 < 10000 ∧ 36 ∣ k^2}.to_finset.card = 16 := 
sorry

end perfect_squares_multiple_of_36_l181_181199


namespace line_does_not_pass_through_second_quadrant_l181_181809
-- Import the Mathlib library

-- Define the properties of the line
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the condition for a point to be in the second quadrant:
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the proof statement
theorem line_does_not_pass_through_second_quadrant:
  ∀ x y : ℝ, line_eq x y → ¬ in_second_quadrant x y :=
by
  sorry

end line_does_not_pass_through_second_quadrant_l181_181809


namespace triangle_segment_length_le_longest_side_convex_polygon_segment_length_le_longest_side_or_diagonal_l181_181407

-- Define the problem for the triangle
theorem triangle_segment_length_le_longest_side (A B C M N : Point) (ABC_triangle : Triangle A B C) (MN_inside : SegmentInsideTriangle M N A B C):
  length MN ≤ largest_side_length A B C :=
sorry

-- Define the problem for the convex polygon
theorem convex_polygon_segment_length_le_longest_side_or_diagonal (P : Polygon) (convex : ConvexPolygon P) (M N : Point) (MN_inside : SegmentInsidePolygon M N P):
  length MN ≤ max (max_side_length P) (max_diagonal_length P) :=
sorry

end triangle_segment_length_le_longest_side_convex_polygon_segment_length_le_longest_side_or_diagonal_l181_181407


namespace find_a_for_even_function_l181_181157

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l181_181157


namespace negative_root_no_positive_l181_181174

theorem negative_root_no_positive (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = ax + 1) ∧ (¬ ∃ x : ℝ, x > 0 ∧ |x| = ax + 1) → a > -1 :=
by
  sorry

end negative_root_no_positive_l181_181174


namespace product_a_5_to_100_l181_181918

noncomputable def a (n : ℕ) : ℚ :=
  if h : n ≥ 5 then
    (3 * n^2 + 3 * n + 2) / (n^3 - 1)
  else
    0

theorem product_a_5_to_100 :
  (∏ n in Finset.range (100 - 4), a (n + 5)) = 24727272 / Nat.factorial 100 := sorry

end product_a_5_to_100_l181_181918


namespace arccos_zero_eq_pi_div_two_l181_181081

theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l181_181081


namespace square_area_eq_two_l181_181245

noncomputable def square_area (z : ℂ) : Prop :=
  let v1 := z^4 - z in
  let v2 := z^6 - z in
  let a := abs v1 in
  let b := abs v2 in
  a * b = 2

theorem square_area_eq_two (z : ℂ) (h : z ≠ 0) (hz : z ≠ z^4 ∧ z ≠ z^6 ∧ z^4 ≠ z^6) (h_sqr : square_area z) :
  abs (((z^4) - z)) * abs (((z^6) - z)) = 2 :=
sorry

end square_area_eq_two_l181_181245


namespace percentage_of_carnations_l181_181230

variable (C : ℝ) -- number of carnations
variable (V : ℝ) -- number of violets
variable (T : ℝ) -- number of tulips
variable (R : ℝ) -- number of roses

-- Conditions
noncomputable def conditions : Prop :=
  V = (1 / 3) * C ∧
  T = (1 / 3) * V ∧
  R = T

-- The proof problem
theorem percentage_of_carnations (C V T R : ℝ) (h : conditions C V T R) :
    (C / (C + V + T + R)) * 100 ≈ 64.29 := 
by
  sorry

end percentage_of_carnations_l181_181230


namespace luke_played_rounds_l181_181285

theorem luke_played_rounds (total_points : ℕ) (points_per_round : ℕ) (result : ℕ)
  (h1 : total_points = 154)
  (h2 : points_per_round = 11)
  (h3 : result = total_points / points_per_round) :
  result = 14 :=
by
  rw [h1, h2] at h3
  exact h3

end luke_played_rounds_l181_181285


namespace set_M_enumeration_l181_181654

noncomputable def A := { n : ℕ | 1 ≤ n ∧ n ≤ 500 }

noncomputable def f (n : ℕ) : ℝ := log (n + 2) / log (n + 1)

theorem set_M_enumeration : 
  { k : ℕ | ∃ n ∈ A, k = ∏ i in Finset.range (n+1) \ {0}, f i } = {2, 3, 4, 5, 6, 7, 8} :=
by sorry

end set_M_enumeration_l181_181654


namespace sample_not_representative_l181_181433

-- Definitions
def has_email_address (person : Type) : Prop := sorry
def uses_internet (person : Type) : Prop := sorry

-- Problem statement: prove that the sample is not representative of the urban population.
theorem sample_not_representative (person : Type)
  (sample : set person)
  (h_sample_size : set.size sample = 2000)
  (A : person → Prop)
  (A_def : ∀ p, A p ↔ has_email_address p)
  (B : person → Prop)
  (B_def : ∀ p, B p ↔ uses_internet p)
  (dependent : ∀ p, A p → B p)
  : ¬ is_representative_sample sample := 
sorry

end sample_not_representative_l181_181433


namespace batsman_average_after_35th_inning_l181_181619

theorem batsman_average_after_35th_inning
  (A : ℝ)
  (h1 : ∃ A : ℝ, 34 * A + 150 = 35 * (A + 1.75))
  (h2 : ∀ runs average innings, average = runs / innings)
  (pitch_reduction : ℝ := 0.65)
  (weather_reduction : ℝ := 0.45) :
  let new_average := A + 1.75;
      adjusted_average := new_average - (pitch_reduction + weather_reduction) 
  in adjusted_average = 89.4 :=
by
  sorry

end batsman_average_after_35th_inning_l181_181619


namespace derivative_at_2_f_l181_181549

theorem derivative_at_2_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x) = 2 * f(2 - x) - x^2 + 8 * x - 8) : f'(2) = 4 :=
by
  sorry

end derivative_at_2_f_l181_181549


namespace total_grocery_bill_l181_181308

theorem total_grocery_bill
    (hamburger_meat_cost : ℝ := 5.00)
    (crackers_cost : ℝ := 3.50)
    (frozen_vegetables_bags : ℝ := 4)
    (frozen_vegetables_cost_per_bag : ℝ := 2.00)
    (cheese_cost : ℝ := 3.50)
    (discount_rate : ℝ := 0.10) :
    let total_cost_before_discount := hamburger_meat_cost + crackers_cost + (frozen_vegetables_bags * frozen_vegetables_cost_per_bag) + cheese_cost
    let discount := total_cost_before_discount * discount_rate
    let total_cost_after_discount := total_cost_before_discount - discount
in
total_cost_after_discount = 18.00 :=
by
   -- total_cost_before_discount = 5.00 + 3.50 + (4 * 2.00) + 3.50 = 20.00
   -- discount = 20.00 * 0.10 = 2.00
   -- total_cost_after_discount = 20.00 - 2.00 = 18.00
   sorry

end total_grocery_bill_l181_181308


namespace differential_savings_l181_181020

theorem differential_savings (income : ℝ) (tax_rate1 tax_rate2 : ℝ) 
                            (old_tax_rate_eq : tax_rate1 = 0.40) 
                            (new_tax_rate_eq : tax_rate2 = 0.33) 
                            (income_eq : income = 45000) :
    ((tax_rate1 - tax_rate2) * income) = 3150 :=
by
  rw [old_tax_rate_eq, new_tax_rate_eq, income_eq]
  norm_num

end differential_savings_l181_181020


namespace percentage_increase_l181_181222

theorem percentage_increase (x : ℝ) (y : ℝ) (h1 : x = 114.4) (h2 : y = 88) : 
  ((x - y) / y) * 100 = 30 := 
by 
  sorry

end percentage_increase_l181_181222


namespace parallelogram_by_condition_l181_181136

variables (A B C D : Type)
variables (AB CD : A) (AD BC : B)
variables (α β : A → A → Prop)

noncomputable def is_parallelogram (AB CD : A) : Prop :=
  (α AB CD) ∧ (AB = CD)

theorem parallelogram_by_condition (h1 : α AB CD) (h2 : AB = CD) : is_parallelogram AB CD :=
by
  exact ⟨h1, h2⟩

end parallelogram_by_condition_l181_181136


namespace angles_of_inscribed_sphere_l181_181732

theorem angles_of_inscribed_sphere (A B C D E F : Point)
    (inscribed_sphere : Sphere)
    (touches_ABC : inscribed_sphere.touches_triangle (Triangle A B C) = E)
    (touches_ABD : inscribed_sphere.touches_triangle (Triangle A B D) = F) :
    angle A E C = angle B F D :=
by
  sorry

end angles_of_inscribed_sphere_l181_181732


namespace van_capacity_calculation_l181_181629

theorem van_capacity_calculation
  (total_vans : ℕ) (van_capacity : ℕ → ℕ)
  (total_capacity : ℕ) (v1 v2 : ℕ)
  (v3 v4 v5 : ℕ → ℕ)
  (percentage_less : ℕ → ℕ → ℕ)
  (percentage : ℕ) :
  total_vans = 6 →
  v1 = 8000 →
  v2 = 8000 →
  v3 v4 = 12000 →
  v3 v5 = 12000 →
  v3 van_capacity = 12000 →
  total_capacity = 57600 →
  van_capacity v1 + van_capacity v2 + v3 (1) + v3 (2) + v3 (3) + v5 1 = 57600 →
  percentage_less 8000 5600 = 0.3 →
  percentage = 30 :=
by
  intros 
  sorry

end van_capacity_calculation_l181_181629


namespace harvest_days_l181_181372

theorem harvest_days (total_sacks : ℕ) (sacks_per_day : ℕ) : total_sacks = 56 → sacks_per_day = 4 → total_sacks / sacks_per_day = 14 :=
by
  intros h_total h_sacks
  rw [h_total, h_sacks]
  norm_num
  sorry

end harvest_days_l181_181372


namespace func_equation_l181_181537

def f (n : ℤ) (x : ℝ) : ℝ := (n - 1) / (n * x) + 1 / n

theorem func_equation (n : ℤ) (x : ℝ) (hnz : n ≠ 0) (hx : x ≠ 0) (hx_neg3 : x ≠ -3) :
  f n (x + 3) + f n (-9 / x) =
  ((1 - n) * (x^2 + 3*x - 9)) / (9 * n * (x + 3)) + 2 / n :=
by 
  sorry


end func_equation_l181_181537


namespace consecutive_odd_natural_numbers_sum_l181_181361

theorem consecutive_odd_natural_numbers_sum (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : b = a + 6) 
  (h4 : c = a + 12) 
  (h5 : c = 27) 
  (h6 : a % 2 = 1) 
  (h7 : b % 2 = 1) 
  (h8 : c % 2 = 1) 
  (h9 : a % 3 = 0) 
  (h10 : b % 3 = 0) 
  (h11 : c % 3 = 0) 
  : a + b + c = 63 :=
by
  sorry

end consecutive_odd_natural_numbers_sum_l181_181361


namespace total_cost_fencing_l181_181734

-- Define the conditions
def length : ℝ := 75
def breadth : ℝ := 25
def cost_per_meter : ℝ := 26.50

-- Define the perimeter of the rectangular plot
def perimeter : ℝ := 2 * length + 2 * breadth

-- Define the total cost of fencing
def total_cost : ℝ := perimeter * cost_per_meter

-- The theorem statement
theorem total_cost_fencing : total_cost = 5300 := 
by 
  -- This is the statement we want to prove
  sorry

end total_cost_fencing_l181_181734


namespace daughter_weight_l181_181775

variable (Weight : Type)
variable (M D C : Weight)

axiom weight_add : Weight → Weight → Weight → Weight
axiom mul_fraction : Weight → Weight → Weight

axiom condition1 : weight_add M D C = 110
axiom condition2 : weight_add D C = 60
axiom condition3 : C = mul_fraction (1/5) M

theorem daughter_weight : D = 50 :=
by
  sorry

end daughter_weight_l181_181775


namespace perfect_squares_multiple_of_36_l181_181198

theorem perfect_squares_multiple_of_36 (N : ℕ) (h1 : N = 99) : 
  {k | k ∈ ℕ ∧ k^2 < 10000 ∧ 36 ∣ k^2}.to_finset.card = 16 := 
sorry

end perfect_squares_multiple_of_36_l181_181198


namespace geometric_sequence_sum_l181_181167

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₁ : a 3 = 4) (h₂ : a 2 + a 4 = -10) (h₃ : |q| > 1) : 
  (a 0 + a 1 + a 2 + a 3 = -5) := 
by 
  sorry

end geometric_sequence_sum_l181_181167


namespace solve_trig_eq_l181_181688

   theorem solve_trig_eq (x y z : ℝ) (m n : ℤ): 
     sin x ≠ 0 → cos y ≠ 0 →
     (sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z →
     (∃ m n : ℤ, x = (π / 2) + π * m ∧ y = π * n ∧ z = 2 * π * m) := 
   by
     intros h1 h2 h_eq
     sorry
   
end solve_trig_eq_l181_181688


namespace even_function_iff_a_eq_1_l181_181161

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- Lean 4 statement for the given math proof problem
theorem even_function_iff_a_eq_1 :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 1 :=
sorry

end even_function_iff_a_eq_1_l181_181161


namespace minute_hand_angle_l181_181817

theorem minute_hand_angle (minutes_slow : ℕ) (total_minutes : ℕ) (full_rotation : ℝ) (h1 : minutes_slow = 5) (h2 : total_minutes = 60) (h3 : full_rotation = 2 * Real.pi) : 
  (minutes_slow / total_minutes : ℝ) * full_rotation = Real.pi / 6 :=
by
  sorry

end minute_hand_angle_l181_181817


namespace solve_equation_l181_181700

theorem solve_equation (x y z : ℝ) (n k m : ℤ)
  (h1 : sin x ≠ 0)
  (h2 : cos y ≠ 0)
  (h_eq : (sin x ^ 2 + 1 / sin x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * cos z)
  : ∃ n k m : ℤ, x = π / 2 + π * n ∧ y = π * k ∧ z = 2 * π * m :=
by
  sorry

end solve_equation_l181_181700


namespace problem_solution_l181_181116

theorem problem_solution (n : ℕ) (h : n > 0) (hC : finset.card (finset.powerset (finset.range 9)).filter (λ s, s.card = n + 1) = finset.card (finset.powerset (finset.range 9)).filter (λ s, s.card = 2 * n - 1)) :
  n = 2 ∨ n = 3 := 
by 
  sorry

end problem_solution_l181_181116


namespace angle_TCD_is_38_l181_181825

-- Definitions for the conditions of the problem
variables (A B C D T : Type)

-- Isosceles trapezoid ABCD with bases BC and AD and a point T inside
def isosceles_trapezoid (ABCD BC AD : Type) (ADC CAD : ℝ) (T : Type) :=
  ∠ ADC = 82 ∧ 2 * ∠ CAD = 82 ∧ CT = CD ∧ AT = TD

-- The property we need to prove, that angle TCD is 38 degrees
theorem angle_TCD_is_38 
  (isosceles_trapezoid ABCD BC AD ADC CAD T : Type)
  (H : isosceles_trapezoid ABCD BC AD ADC CAD T) :
  ∠ TCD = 38 :=
sorry

end angle_TCD_is_38_l181_181825


namespace find_a_l181_181626

theorem find_a 
  (x y a m n : ℝ)
  (h1 : x - 5 / 2 * y + 1 = 0) 
  (h2 : x = m + a) 
  (h3 : y = n + 1)  -- since k = 1, so we replace k with 1
  (h4 : m + a = m + 1 / 2) : 
  a = 1 / 2 := 
by 
  sorry

end find_a_l181_181626


namespace second_player_wins_l181_181663

theorem second_player_wins :
  ∀ (strip_length : ℕ) (initial_white_pos initial_black_pos : ℕ),
  strip_length = 20 →
  initial_white_pos = 1 →
  initial_black_pos = 20 →
  (∀ W B : ℕ,
    (W ≥ 1 ∧ W ≤ strip_length) →
    (B ≥ 1 ∧ B ≤ strip_length) →
    abs (W - B) ≠ 0 →
    ((∃ d : ℕ, B = W + d ∧ d % 3 = 0) ∨ (∃ d : ℕ, W = B + d ∧ d % 3 = 0)) →
    second_player_wins) :=
by {
    sorry
}

end second_player_wins_l181_181663


namespace triangle_similarity_PNV_PBV_l181_181840

noncomputable section

variable {Point : Type} [MetricSpace Point]
variable (P Q R S N V B : Point)
variable (circle : Set Point)

-- Given conditions
variable (PQ_RS_perpendicular_bisector : ∀ (x : MetricSpace.Segment Point), x ∈ Segment P Q → x ∈ Segment R S → x ∈ Segment N → IsPerpendicularBisector (PQ : Segment P Q) x)
variable (V_between_RN : MetricSpace.Between V R N)
variable (PB_intersects_circle_at_B : MetricSpace.ExtendsAtCircle P V B circle)

-- To prove
theorem triangle_similarity_PNV_PBV :
  MetricSpace.Similar (Triangle P N V) (Triangle P B V) :=
sorry

end triangle_similarity_PNV_PBV_l181_181840


namespace max_value_on_interval_l181_181351

-- Define the function y = x / e^x
def y (x : ℝ) : ℝ := x / Real.exp x

-- State the theorem to be proved
theorem max_value_on_interval : 
  ∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, y y ≤ y x ∧ y x = 1 / Real.exp 1 :=
by
  sorry

end max_value_on_interval_l181_181351


namespace regular_tetrahedron_subdivision_l181_181496

theorem regular_tetrahedron_subdivision :
  ∃ (n : ℕ), n ≤ 7 ∧ (∀ (i : ℕ) (h : i ≥ n), (1 / 2^i) < (1 / 100)) :=
by
  sorry

end regular_tetrahedron_subdivision_l181_181496


namespace meeting_time_and_distance_l181_181683

variable (t : ℝ) -- t is the time in hours since 7:45 AM when they meet

-- Conditions
def samantha_speed : ℝ := 15 -- Speed in miles/hour
def adam_speed : ℝ := 20 -- Speed in miles/hour
def total_distance : ℝ := 75 -- Total distance in miles

-- Time Adam started after Samantha
def adam_delay : ℝ := 0.5 -- Delay in hours (30 minutes)

-- Distance equations
def samantha_distance (t : ℝ) : ℝ := samantha_speed * t
def adam_distance (t : ℝ) : ℝ := adam_speed * (t - adam_delay)

-- The equation solving for t
def time_equation (t : ℝ) : Prop :=
  samantha_distance t + adam_distance t = total_distance

-- The proof that given the conditions, the solution is correct
theorem meeting_time_and_distance :
  time_equation t →
  t = 2.428571 →
  samantha_distance t ≈ 36 →
  t ≈ 10 + 11 ÷ 60 :=
sorry

end meeting_time_and_distance_l181_181683


namespace solve_equation_l181_181710

theorem solve_equation (x y z : ℝ) (m n : ℤ) :
  (sin x ≠ 0) →
  (cos y ≠ 0) →
  ((sin^2 x + 1 / (sin^2 x ))^3 + (cos^2 y + 1 / (cos^2 y))^3 = 16 * cos z) →
  (cos z = 1) ∧ 
  (∃ m : ℤ, x = π / 2 + π * m) ∧ 
  (∃ n : ℤ, y = π * n) ∧ 
  (∃ m : ℤ, z = 2 * π * m) :=
by
  intros hsin_cos hcos_sin heq
  sorry

end solve_equation_l181_181710


namespace scientific_to_decimal_l181_181724

theorem scientific_to_decimal : (2 * 10 ^ (-3) : ℝ) = 0.002 :=
by
  sorry

end scientific_to_decimal_l181_181724


namespace number_of_round_trips_each_bird_made_l181_181376

theorem number_of_round_trips_each_bird_made
  (distance_to_materials : ℕ)
  (total_distance_covered : ℕ)
  (distance_one_round_trip : ℕ)
  (total_number_of_trips : ℕ)
  (individual_bird_trips : ℕ) :
  distance_to_materials = 200 →
  total_distance_covered = 8000 →
  distance_one_round_trip = 2 * distance_to_materials →
  total_number_of_trips = total_distance_covered / distance_one_round_trip →
  individual_bird_trips = total_number_of_trips / 2 →
  individual_bird_trips = 10 :=
by
  intros
  sorry

end number_of_round_trips_each_bird_made_l181_181376


namespace smallest_x_is_1_l181_181395

-- Define the condition for the quadratic expression being prime
def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ (∀ m : ℤ, m > 1 → m < n → m ∣ n → false)

-- Define the absolute value function for integer numbers
def abs (z : ℤ) : ℤ := if z < 0 then -z else z

-- The main statement asserting the smallest integer x such that |4x^2 - 34x + 21| is prime is 1
theorem smallest_x_is_1 : ∃ x : ℤ, (∀ y : ℤ, y < x → abs (4 * y * y - 34 * y + 21) = 1 → false) ∧ abs (4 * x * x - 34 * x + 21) = 5 :=
by
  sorry

end smallest_x_is_1_l181_181395


namespace highest_place_value_734_48_l181_181389

theorem highest_place_value_734_48 : 
  (∃ k, 10^4 = k ∧ k * 10^4 ≤ 734 * 48 ∧ 734 * 48 < (k + 1) * 10^4) := 
sorry

end highest_place_value_734_48_l181_181389


namespace quadratic_always_positive_l181_181877

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 :=
sorry

end quadratic_always_positive_l181_181877


namespace floor_system_unique_solution_l181_181037

noncomputable def floor_system_solution (x y : ℝ) : Prop :=
  (⌊x + y - 3⌋ = 2 - x) ∧ (⌊x + 1⌋ + ⌊y - 7⌋ + x = y)

theorem floor_system_unique_solution : ∃! (x y : ℝ), floor_system_solution x y :=
by
  use [3, -1]
  split
  {
    split
    {
      show (⌊3 + -1 - 3⌋ = 2 - 3), from sorry,
      show (⌊3 + 1⌋ + ⌊-1 - 7⌋ + 3 = -1), from sorry,
    }
    intro xy
    cases xy with x y
    show floor_system_solution x y → (x, y) = (3, -1)
      from sorry
  }
  sorry

end floor_system_unique_solution_l181_181037


namespace sum_k_binomial_l181_181906

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181906


namespace range_of_a_l181_181182

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 / exp 1 ≤ x ∧ x ≤ exp 1 → a - x^2 = - (2 * log x)) →
  1 ≤ a ∧ a ≤ exp 1^2 - 2 :=
by
  sorry

end range_of_a_l181_181182


namespace appropriate_units_for_conditions_l181_181523

theorem appropriate_units_for_conditions :
  (∀ car_speed math_book_thickness truck_capacity person_weight : ℕ, 
    car_speed = 80 → math_book_thickness = 7 → truck_capacity = 4 → person_weight = 35 →
    (car_speed, math_book_thickness, truck_capacity, person_weight) = (80, 7, 4, 35)) →
  (80.units = "kilometers per hour" ∧ 7.units = "millimeters thick" ∧ 
  4.units = "tons of cargo" ∧ 35.units = "kilograms") := 
  sorry

-- Definitions to match conditions (Exemplary, need unit definitions)
def (n : ℕ).units : String := if n = 80 then "kilometers per hour"
                             else if n = 7 then "millimeters thick"
                             else if n = 4 then "tons of cargo"
                             else if n = 35 then "kilograms"
                             else "unknown"


end appropriate_units_for_conditions_l181_181523


namespace solve_trig_equation_l181_181695

theorem solve_trig_equation (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) → 
  (cos y ≠ 0) → 
  ((sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z) →
  (∃ (n k m : ℤ), x = (π / 2) + π * n ∧ y = π * k ∧ z = 2 * π * m) := 
by 
  sorry

end solve_trig_equation_l181_181695


namespace find_a_for_even_l181_181151

def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

theorem find_a_for_even (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = f a x) ↔ a = 1 :=
by
  -- proof steps here
  sorry

end find_a_for_even_l181_181151


namespace slower_speed_l181_181801

theorem slower_speed (x : ℝ) (h_walk_faster : 12 * (100 / x) - 100 = 20) : x = 10 :=
by sorry

end slower_speed_l181_181801


namespace sum_k_binomial_l181_181904

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181904


namespace number_of_males_not_interested_l181_181813

-- Define the conditions 
variable (total_not_interested : ℕ) -- Total number of individuals not interested
variable (females_not_interested : ℕ) -- Number of females not interested

-- Assume concrete values for conditions
axiom h1 : total_not_interested = 200
axiom h2 : females_not_interested = 90

-- Define the conclusion
def males_not_interested : ℕ := total_not_interested - females_not_interested

-- Prove the conclusion
theorem number_of_males_not_interested : males_not_interested total_not_interested females_not_interested = 110 :=
by {
  -- Due to the given axioms
  rw [h1, h2],
  -- Simplify the expression
  exact rfl,
}

end number_of_males_not_interested_l181_181813


namespace intersect_circumcircles_ALM_NCK_no_intersect_circumcircles_LDK_MBN_l181_181671

variables {A B C D M N K L : Type} [EuclideanGeometry A B C D M N K L]
variables (rhombus : Rhombus A B C D)
variables (M_on_AB : On M A B) (N_on_BC : On N B C)
           (K_on_CD : On K C D) (L_on_DA : On L D A)
variables (MN_parallel_LK : Parallel MN LK)
variables (distance_MN_LK_height : Distance MN LK = Height rhombus)

open EuclideanGeometry

theorem intersect_circumcircles_ALM_NCK :
  Circumcircle A L M ∩ Circumcircle N C K ≠ ∅ :=
sorry

theorem no_intersect_circumcircles_LDK_MBN :
  Circumcircle L D K ∩ Circumcircle M B N = ∅ :=
sorry

end intersect_circumcircles_ALM_NCK_no_intersect_circumcircles_LDK_MBN_l181_181671


namespace combined_population_correct_l181_181357

theorem combined_population_correct (W PP LH N : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : LH = 2 * W + 600)
  (hN : N = 3 * (PP - W)) :
  PP + LH + N = 24900 :=
by
  sorry

end combined_population_correct_l181_181357


namespace evaluate_g_ggg_15_l181_181274

def g : ℝ → ℝ :=
  λ x, if x < 5 then x^2 + 2*x - 5 else 2*x - 18

theorem evaluate_g_ggg_15 : g (g (g 15)) = -6 :=
  by sorry

end evaluate_g_ggg_15_l181_181274


namespace arrangement_problem_l181_181735
noncomputable def num_arrangements : ℕ := 144

theorem arrangement_problem (A B C D E F : ℕ) 
  (adjacent_easy : A = B) 
  (not_adjacent_difficult : E ≠ F) : num_arrangements = 144 :=
by sorry

end arrangement_problem_l181_181735


namespace largest_divisor_consecutive_odd_squares_l181_181645

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (hmn : m = n + 2) 
  (hodd_m : m % 2 = 1) 
  (hodd_n : n % 2 = 1) 
  (horder : n < m) : ∃ k : ℤ, m^2 - n^2 = 8 * k :=
by 
  sorry

end largest_divisor_consecutive_odd_squares_l181_181645


namespace trig_eqn_solution_l181_181705

noncomputable def solve_trig_eqn (x y z : ℝ) (m n : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (cos y ≠ 0) ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) ∧
  (x = π / 2 + π * m) ∧
  (y = π * n) ∧
  (z = 2 * π * m)

theorem trig_eqn_solution (x y z : ℝ) (m n : ℤ) :
  sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) →
  x = π / 2 + π * m ∧ y = π * n ∧ z = 2 * π * m :=
by
  sorry

end trig_eqn_solution_l181_181705


namespace mark_bench_press_l181_181503

theorem mark_bench_press :
  ∀ (dave_weight : ℕ) (craig_percentage : ℕ) (mark_difference : ℕ),
  dave_weight = 175 →
  (dave_weight * 3) = 525 →
  craig_percentage = 20 →
  (craig_percentage * 525 / 100) = 105 →
  mark_difference = 50 →
  (105 - mark_difference) = 55 :=
by
  intros dave_weight craig_percentage mark_difference h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  rfl

end mark_bench_press_l181_181503


namespace period2_students_is_8_l181_181332

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l181_181332


namespace regression_line_is_y_eq_x_plus_1_l181_181239

theorem regression_line_is_y_eq_x_plus_1 :
  let points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by
  sorry 

end regression_line_is_y_eq_x_plus_1_l181_181239


namespace min_value_expression_l181_181641

noncomputable def min_expression (a b c : ℝ) : ℝ :=
(a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((4 / c) - 1)^2

theorem min_value_expression :
  ∀ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4 →
  min_expression a b c ≥ 12 - 8 * Real.sqrt 2 :=
by
  intros a b c
  assume h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4
  sorry

end min_value_expression_l181_181641


namespace inequality_solution_l181_181867

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Ioo (-1/4) 0 ∪ Set.Ioo 3/2 2) ↔ 
  (1 ≤ (x - 2) * 4 ∧ x ≠ 2) ∧ (x > 0 ∨ x ≠ 0) := 
sorry

end inequality_solution_l181_181867


namespace leggings_needed_l181_181588

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l181_181588


namespace six_times_expression_l181_181207

theorem six_times_expression {x y Q : ℝ} (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q :=
by
  sorry

end six_times_expression_l181_181207


namespace sample_not_representative_l181_181444

-- Define the events A and B
def A : Prop := ∃ (x : Type), (x → Prop) -- A person has an email address
def B : Prop := ∃ (x : Type), (x → Prop) -- A person uses the internet

-- Define the dependence and characteristics
def is_dependent (A B : Prop) : Prop := A ∧ B -- A and B are dependent
def linked_to_internet_usage (A : Prop) (B : Prop) : Prop := A → B -- A is closely linked to B
def not_uniform_distribution (B : Prop) : Prop := ¬ (∀ x, B x) -- Internet usage is not uniformly distributed

-- Define the subset condition
def is_subset_of_regular_users (A B : Prop) : Prop := ∀ x, A x → B x -- A represents regular internet users

-- Prove the given statement
theorem sample_not_representative
  (A B : Prop)
  (h_dep : is_dependent A B)
  (h_link : linked_to_internet_usage A B)
  (h_not_unif : not_uniform_distribution B)
  (h_subset : is_subset_of_regular_users A B) :
  ¬ represents_urban_population A :=
sorry

end sample_not_representative_l181_181444


namespace angle_between_cube_diagonals_l181_181776

theorem angle_between_cube_diagonals : 
  (∀ (P Q R S : Type), P ≠ Q → Q ≠ R → R ≠ S → S ≠ P → (w : ℝ), 
  let edges_perpendicular := ∀ e1 e2, e1 ∈ {P, Q, R, S} → e2 ∈ {P, Q, R, S} → e1 ≠ e2 → 
  (e1 ∘ e2 = 0),
  let planes_perpendicular := ∀ p1 p2, p1 ∈ {P, Q, R, S} → p2 ∈ {P, Q, R, S} → p1 ≠ p2 → 
  (p1 ∘ p2 = 0),
  edges_perpendicular ∧ planes_perpendicular → w = 90) := 
sorry

end angle_between_cube_diagonals_l181_181776


namespace sum_even_vs_odd_divisors_l181_181596

def even_divisors_count (k : ℕ) : ℕ := 
  -- definition to count even divisors
  sorry

def odd_divisors_count (k : ℕ) : ℕ := 
  -- definition to count odd divisors
  sorry

theorem sum_even_vs_odd_divisors (n : ℕ) :
  |(∑ k in Finset.range (n+1), even_divisors_count k) - (∑ k in Finset.range (n+1), odd_divisors_count k)| ≤ n :=
  sorry

end sum_even_vs_odd_divisors_l181_181596


namespace period_2_students_l181_181331

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l181_181331


namespace max_individual_score_l181_181788

open Nat

theorem max_individual_score (n : ℕ) (total_points : ℕ) (minimum_points : ℕ) (H1 : n = 12) (H2 : total_points = 100) (H3 : ∀ i : Fin n, 7 ≤ minimum_points) :
  ∃ max_points : ℕ, max_points = 23 :=
by 
  sorry

end max_individual_score_l181_181788


namespace range_of_g_l181_181265

noncomputable def f : ℝ → ℝ := λ x, 4 * x + 1
noncomputable def g : ℝ → ℝ := λ x, 256 * x + 85

theorem range_of_g (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) : 
  85 ≤ g x ∧ g x ≤ 853 :=
by
  sorry

end range_of_g_l181_181265


namespace compute_fraction_l181_181597

def x : ℚ := 2 / 3
def y : ℚ := 3 / 2
def z : ℚ := 1 / 3

theorem compute_fraction :
  (1 / 3) * x^7 * y^5 * z^4 = 11 / 600 :=
by
  sorry

end compute_fraction_l181_181597


namespace solve_for_x_l181_181325

theorem solve_for_x (x : ℝ) (h : 4^x * 4^x * 2^(2 * x) = 16 ^ 3) : x = 2 :=
by
  sorry

end solve_for_x_l181_181325


namespace smallest_k_elements_for_triple_sum_l181_181846

theorem smallest_k_elements_for_triple_sum (M : Set ℕ) (hM : M = {n | 1 ≤ n ∧ n ≤ 2020}) :
    ∃ k : ℕ, (k = 1011) ∧ ∀ A : Finset ℕ, A ⊆ M → A.card = k →
    ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ (a + b) ∈ A ∧ (b + c) ∈ A ∧ (c + a) ∈ A := 
sorry

end smallest_k_elements_for_triple_sum_l181_181846


namespace average_speed_of_train_l181_181468

theorem average_speed_of_train (x : ℝ) (h1 : x > 0): 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 :=
by
  sorry

end average_speed_of_train_l181_181468


namespace max_value_permutation_sum_l181_181637

theorem max_value_permutation_sum : 
  let P := (6 * 6 + 6 * 1 + 1 * 2 + 2 * 3 + 3 * 4 + 4 * 6)
  ∧ let Q := 10
  in P + Q = 96 := 
by 
  -- Definitions of P and Q as given
  let P := 6 * 6 + 6 * 1 + 1 * 2 + 2 * 3 + 3 * 4 + 4 * 6
  let Q := 10
  -- Conclusion
  show P + Q = 96, from sorry

end max_value_permutation_sum_l181_181637


namespace cos_75_sub_cos_15_l181_181853

theorem cos_75_sub_cos_15 : 
  cos (75 * Real.pi / 180) - cos (15 * Real.pi / 180) = -sqrt 2 / 2 := 
by sorry

end cos_75_sub_cos_15_l181_181853


namespace trigonometric_identity_solution_l181_181769

open Real

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  (cos x ≠ 0) ∧ (sin x ≠ 0) ∧ (tg x ^ 4 + ctg x ^ 4 = (82 / 9) * (tg x * tg (2 * x) + 1) * cos (2 * x)) ↔
  ∃ (n : ℤ), x = (π / 6) * (3 * k ± n)
  :=
sorry

end trigonometric_identity_solution_l181_181769


namespace trig_eqn_solution_l181_181703

noncomputable def solve_trig_eqn (x y z : ℝ) (m n : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (cos y ≠ 0) ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) ∧
  (x = π / 2 + π * m) ∧
  (y = π * n) ∧
  (z = 2 * π * m)

theorem trig_eqn_solution (x y z : ℝ) (m n : ℤ) :
  sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) →
  x = π / 2 + π * m ∧ y = π * n ∧ z = 2 * π * m :=
by
  sorry

end trig_eqn_solution_l181_181703


namespace no_one_common_tangent_l181_181377

theorem no_one_common_tangent (r1 r2 : ℝ) (h_diff : r1 ≠ r2) (P1 P2 : Point) :

∃ (n : ℕ), n ∈ {0, 2, 3, 4} ∧ n ≠ 1
  := by
  sorry

end no_one_common_tangent_l181_181377


namespace biased_sample_non_representative_l181_181441

/-- 
A proof problem verifying the representativeness of a sample of 2000 email address owners concerning 
the urban population's primary sources of news.
-/
theorem biased_sample_non_representative (
  (U : Type) 
  (email_population : finset U) 
  (sample : finset U) :
  sample.card = 2000 
  ∧ sample ⊆ email_population 
  ∧ ∃ (u : U), u ∈ sample 
  → email_population.card < finset.card email_population
  := sorry

end biased_sample_non_representative_l181_181441


namespace solve_trig_eq_l181_181687

theorem solve_trig_eq (x : ℝ) (k p : ℤ) :
  (cos (4 * x) / (cos (3 * x) - sin (3 * x)) + sin (4 * x) / (cos (3 * x) + sin (3 * x)) = real.sqrt 2) →
  (cos (6 * x) ≠ 0) →
  (x = (real.pi / 52) + (2 * real.pi * k / 13) ∧ ¬(k = 13 * p - 5)) :=
sorry

end solve_trig_eq_l181_181687


namespace unique_nets_of_a_cube_l181_181978

-- Definitions based on the conditions and the properties of the cube
def is_net (net: ℕ) : Prop :=
  -- A placeholder definition of a valid net
  sorry

def is_distinct_by_rotation_or_reflection (net1 net2: ℕ) : Prop :=
  -- Two nets are distinct if they cannot be transformed into each other by rotation or reflection
  sorry

-- The statement to be proved
theorem unique_nets_of_a_cube : ∃ n, n = 11 ∧ (∀ net, is_net net → ∃! net', is_net net' ∧ is_distinct_by_rotation_or_reflection net net') :=
sorry

end unique_nets_of_a_cube_l181_181978


namespace count_prime_dates_2009_l181_181590

open Nat

-- Define the months and their respective number of days in a regular year
def days_in_month (m : ℕ) : ℕ :=
  match m with
  | 2  => 28
  | 3  => 31
  | 5  => 31
  | 7  => 31
  | 11 => 30
  | _  => 0

-- Define a predicate for prime numbers
def is_prime_day (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 17 ∨ d = 19 ∨ d = 23 ∨ d = 29 ∨ d = 31

-- Define the number of prime dates in a given month
def prime_dates_in_month (m : ℕ) : list ℕ :=
  (list.range (days_in_month m)).filter is_prime_day

-- Define the total number of prime dates in the given year
def total_prime_dates_2009 : ℕ :=
  prime_dates_in_month 2 ++ prime_dates_in_month 3 ++ prime_dates_in_month 5 ++ prime_dates_in_month 7 ++ prime_dates_in_month 11

-- Prove that the total number of prime dates in 2009 is 52
theorem count_prime_dates_2009 : total_prime_dates_2009.length = 52 :=
by sorry

end count_prime_dates_2009_l181_181590


namespace find_b1_l181_181967

noncomputable def a_n : ℕ+ → ℕ
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨n+3, _⟩ := a_n ⟨n+1, sorry⟩

noncomputable def b_sequence : (ℕ → ℕ) := sorry

theorem find_b1 :
  (∀ n : ℕ+, a_n ⟨n+2, sorry⟩ = a_n n) →
  (∀ n : ℕ+, b_sequence (n+1) - b_sequence n = a_n n) →
  (∃ m : ℕ, ∃ k : ℕ, m ≠ k ∧ b_sequence m / a_n ⟨m / 2 + 1, sorry⟩ = b_sequence k / a_n ⟨k / 2 + 1, sorry⟩ ∧ m % 2 = 0 ∧ k % 2 = 0 ∧ m > 0 ∧ k > 0) →
  b_sequence 1 = 2 :=
sorry

end find_b1_l181_181967


namespace find_b_from_conditions_l181_181953

theorem find_b_from_conditions (x y z k : ℝ) (h1 : (x + y) / 2 = k) (h2 : (z + x) / 3 = k) (h3 : (y + z) / 4 = k) (h4 : x + y + z = 36) : x + y = 16 := 
by 
  sorry

end find_b_from_conditions_l181_181953


namespace peter_profit_l181_181302

variable (C : ℝ) -- Cost of the scooter
variable (repair_cost : ℝ := 500) -- Cost spent on repairs
variable (repair_percentage : ℝ := 0.10) -- 10% of cost on repairs
variable (profit_percentage : ℝ := 0.20) -- 20% profit made

-- The given condition relating repair cost to cost of the scooter
axiom repair_equation : repair_cost = repair_percentage * C

-- The goal is to prove Peter's profit is $1000
theorem peter_profit : repair_cost = 500 → repair_percentage = 0.10 → profit_percentage = 0.20 → (p : ℝ) := profit_percentage * C = 1000 := by
  intro h₁ h₂ h₃
  sorry

end peter_profit_l181_181302


namespace triangle_BPC_area_l181_181999

universe u

variables {T : Type u} [LinearOrderedField T]

-- Define the points
variables (A B C E F P : T)
variables (area : T → T → T → T) -- A function to compute the area of a triangle

-- Hypotheses
def conditions :=
  E ∈ [A, B] ∧
  F ∈ [A, C] ∧
  (∃ P, P ∈ [B, F] ∧ P ∈ [C, E]) ∧
  area A E P + area E P F + area P F A = 4 ∧ -- AEPF
  area B E P = 4 ∧ -- BEP
  area C F P = 4   -- CFP

-- The theorem to prove
theorem triangle_BPC_area (h : conditions A B C E F P area) : area B P C = 12 :=
sorry

end triangle_BPC_area_l181_181999


namespace infinitely_many_nats_satisfy_equation_l181_181558

noncomputable def alpha : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

def satisfies_equation (n : ℕ) : Prop :=
  ⟦α * n + 1989 * α * ⟦α * n⟧⟧ = 1989 * n + (1989^2 + 1) * ⟦α * n⟧

theorem infinitely_many_nats_satisfy_equation : ∃ᶠ (n : ℕ) in ⊤, satisfies_equation n :=
sorry  -- Proof is omitted.

end infinitely_many_nats_satisfy_equation_l181_181558


namespace change_amount_l181_181515

theorem change_amount 
    (tank_capacity : ℕ) 
    (current_fuel : ℕ) 
    (price_per_liter : ℕ) 
    (total_money : ℕ) 
    (full_tank : tank_capacity = 150) 
    (fuel_in_truck : current_fuel = 38) 
    (cost_per_liter : price_per_liter = 3) 
    (money_with_donny : total_money = 350) : 
    total_money - ((tank_capacity - current_fuel) * price_per_liter) = 14 :=
by
sorr

end change_amount_l181_181515


namespace general_term_sum_of_first_n_terms_l181_181172

-- Define the arithmetic sequence with given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

-- Define the given conditions for the sequence
axiom a5_eq_12 : ∀ {a : ℕ → ℤ}, arithmetic_seq a (-2) → a 5 = 12
axiom a20_eq_neg18 : ∀ {a : ℕ → ℤ}, arithmetic_seq a (-2) → a 20 = -18

-- Proof statement for the general term
theorem general_term (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_seq a d) : 
  a 1 = 20 → d = -2 → ∀ n, a n = 22 - 2 * n :=
by
  -- Proof will go here
  sorry

-- Proof statement for the sum of the first n terms
theorem sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_seq a d) : 
  a 1 = 20 → d = -2 → ∀ n, (finset.range n).sum (λ k, a k) = 21 * n - n^2 :=
by
  -- Proof will go here
  sorry

end general_term_sum_of_first_n_terms_l181_181172


namespace area_of_contained_region_l181_181105

def contained_area (x y : ℝ) : Prop :=
  abs (2 * x + 3 * y) + abs (2 * x - 3 * y) ≤ 12

theorem area_of_contained_region : 
  (realVolume (setOf (λ p : ℝ × ℝ, contained_area p.1 p.2)) = 24) :=
sorry

end area_of_contained_region_l181_181105


namespace trains_cross_time_l181_181039

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (1000 / 3600)
  let speed2_mps := speed2_kmph * (1000 / 3600)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time :
  time_to_cross 280 220.04 120 80 ≈ 9 := 
by
  sorry

end trains_cross_time_l181_181039


namespace find_k_l181_181124

def vector := (ℚ × ℚ × ℚ)

def dot_product (v w : vector) : ℚ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def a : vector := (1, 1, 0)
def b : vector := (-1, 0, 2)

def ka_plus_b (k : ℚ) : vector :=
  (k * 1 + (-1), k * 1 + 0, k * 0 + 2)

def two_a_minus_b : vector :=
  (2 * 1 - (-1), 2 * 1 - 0, 2 * 0 - 2)

theorem find_k (k : ℚ) :
  dot_product (ka_plus_b k) two_a_minus_b = 0 →
  k = 7 / 5 :=
sorry

end find_k_l181_181124


namespace total_grocery_bill_l181_181307

theorem total_grocery_bill
    (hamburger_meat_cost : ℝ := 5.00)
    (crackers_cost : ℝ := 3.50)
    (frozen_vegetables_bags : ℝ := 4)
    (frozen_vegetables_cost_per_bag : ℝ := 2.00)
    (cheese_cost : ℝ := 3.50)
    (discount_rate : ℝ := 0.10) :
    let total_cost_before_discount := hamburger_meat_cost + crackers_cost + (frozen_vegetables_bags * frozen_vegetables_cost_per_bag) + cheese_cost
    let discount := total_cost_before_discount * discount_rate
    let total_cost_after_discount := total_cost_before_discount - discount
in
total_cost_after_discount = 18.00 :=
by
   -- total_cost_before_discount = 5.00 + 3.50 + (4 * 2.00) + 3.50 = 20.00
   -- discount = 20.00 * 0.10 = 2.00
   -- total_cost_after_discount = 20.00 - 2.00 = 18.00
   sorry

end total_grocery_bill_l181_181307


namespace mean_of_children_ages_l181_181329

-- Define the conditions (the ages of the children) and the correct answer

def ages : List ℕ := [6, 6, 6, 6, 12, 14, 14, 16]
def totalChildren := 8
def totalSum := 80
def mean := totalSum / totalChildren  -- The correct answer should be 10

-- Prove that the mean of the given ages is 10
theorem mean_of_children_ages (ages : List ℕ)
  (h_ages : ages = [6, 6, 6, 6, 12, 14, 14, 16])
  (totalChildren = 8) 
  (totalSum = 80):
  mean = 10 := by
  sorry

end mean_of_children_ages_l181_181329


namespace angle_between_vectors_l181_181955

variable {α : Type} [InnerProductSpace ℝ α]

theorem angle_between_vectors 
  {a b : α} 
  (h1 : ∥b∥ = Real.sqrt 2)
  (h2 : ⟪a, b⟫ = 2)
  (h3 : ∥a + b∥ = Real.sqrt 14) : 
  (Real.angle a b) = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l181_181955


namespace solution_set_non_empty_iff_l181_181219

theorem solution_set_non_empty_iff (a : ℝ) : (∃ x : ℝ, |x - 1| + |x + 2| < a) ↔ (a > 3) := 
sorry

end solution_set_non_empty_iff_l181_181219


namespace exists_small_intersecting_subset_l181_181264

variable {X : Type} [Fintype X]

theorem exists_small_intersecting_subset
  (A : Fin 50 → set X)
  (hA : ∀ i, Fintype.card (A i) > Fintype.card X / 2) :
  ∃ B : set X, B.finite ∧ B.card ≤ 5 ∧ ∀ i, (B ∩ A i).nonempty :=
by
  sorry

end exists_small_intersecting_subset_l181_181264


namespace product_of_digits_in_base8_representation_of_7890_is_336_l181_181392

def base8_representation_and_product (n : ℕ) : ℕ × ℕ :=
  let digits := [1, 7, 2, 4, 6] in
  let product := digits.foldl (· * ·) 1 in 
  (digits.foldr (λ d acc, acc * 8 + d) 0, product)

theorem product_of_digits_in_base8_representation_of_7890_is_336 :
  ∀ (n : ℕ), n = 7890 → (base8_representation_and_product n).2 = 336 :=
by
  intros n h
  rw [← h]
  have := base8_representation_and_product 7890
  simp only [this]
  -- Here proof steps are skipped using sorry
  sorry

end product_of_digits_in_base8_representation_of_7890_is_336_l181_181392


namespace new_team_average_weight_is_113_l181_181412

-- Defining the given constants and conditions
def original_players := 7
def original_average_weight := 121 
def weight_new_player1 := 110 
def weight_new_player2 := 60 

-- Definition to calculate the new average weight
def new_average_weight : ℕ :=
  let original_total_weight := original_players * original_average_weight
  let new_total_weight := original_total_weight + weight_new_player1 + weight_new_player2
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

-- Statement to prove
theorem new_team_average_weight_is_113 : new_average_weight = 113 :=
sorry

end new_team_average_weight_is_113_l181_181412


namespace find_a_l181_181553

-- Define sets A and B
def A := {-1, 1, 3}
def B (a : ℝ) := {2, 2^a - 1}

-- Condition given in the problem
def intersection_cond (a : ℝ) : Prop := A ∩ B a = {1}

theorem find_a (a : ℝ) (h : intersection_cond a) : a = 1 :=
by
  sorry  -- Proof is omitted

end find_a_l181_181553


namespace binomial_sum_sum_of_binomial_solutions_l181_181900

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l181_181900


namespace sum_of_valid_k_equals_26_l181_181891

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l181_181891


namespace biased_sample_non_representative_l181_181438

/-- 
A proof problem verifying the representativeness of a sample of 2000 email address owners concerning 
the urban population's primary sources of news.
-/
theorem biased_sample_non_representative (
  (U : Type) 
  (email_population : finset U) 
  (sample : finset U) :
  sample.card = 2000 
  ∧ sample ⊆ email_population 
  ∧ ∃ (u : U), u ∈ sample 
  → email_population.card < finset.card email_population
  := sorry

end biased_sample_non_representative_l181_181438


namespace probability_ratio_l181_181861

theorem probability_ratio (total_slips : ℕ) (num_diff_numbers : ℕ) (slips_per_number : ℕ) (drawn_slips : ℕ) :
  total_slips = 50 ∧ num_diff_numbers = 10 ∧ slips_per_number = 5 ∧ drawn_slips = 4 →
  (let p := (num_diff_numbers * (Nat.comb slips_per_number drawn_slips)) / (Nat.comb total_slips drawn_slips) in
  let q := (num_diff_numbers * (Nat.comb slips_per_number 3) * (num_diff_numbers - 1) * (Nat.comb slips_per_number 1)) 
            / (Nat.comb total_slips drawn_slips) in
  q / p = 90) :=
by
  intros h
  sorry

end probability_ratio_l181_181861


namespace time_drove_in_rain_l181_181685

variables (speed_not_raining speed_raining total_distance total_time break_time : ℕ)
variables (time_in_rain : ℕ)

-- Assumptions based on conditions
def drives_scooter_if_not_raining := speed_not_raining = 40
def drives_scooter_if_raining := speed_raining = 15
def took_break := break_time = 5
def total_journey := total_distance = 20
def journey_took := total_time = 50

-- The equation derived based on distances
def distance_equation := 
  (speed_not_raining * (total_time - time_in_rain - break_time) / 60) +
  (speed_raining * time_in_rain / 60) = total_distance

theorem time_drove_in_rain : 
  drives_scooter_if_not_raining → 
  drives_scooter_if_raining → 
  took_break → 
  total_journey → 
  journey_took → 
  distance_equation → 
  time_in_rain = 24 := 
by
  -- assumptions and definitions would go here
  sorry

end time_drove_in_rain_l181_181685


namespace population_after_panic_l181_181065

noncomputable def original_population : ℕ := 7200
def first_event_loss (population : ℕ) : ℕ := population * 10 / 100
def after_first_event (population : ℕ) : ℕ := population - first_event_loss population
def second_event_loss (population : ℕ) : ℕ := population * 25 / 100
def after_second_event (population : ℕ) : ℕ := population - second_event_loss population

theorem population_after_panic : after_second_event (after_first_event original_population) = 4860 := sorry

end population_after_panic_l181_181065


namespace greatest_saturdays_in_first_45_days_l181_181388

theorem greatest_saturdays_in_first_45_days : ∃ k ≤ 45, max_saturdays k = 7 :=
sorry

end greatest_saturdays_in_first_45_days_l181_181388


namespace find_a_for_even_function_l181_181163

open Function

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- State the problem in Lean.
theorem find_a_for_even_function :
  (∀ x, f a x = f a (-x)) → a = 1 :=
sorry

end find_a_for_even_function_l181_181163


namespace log_identity_l181_181601

theorem log_identity (a b : ℝ) (h1 : a = real.log 343 / real.log 16) (h2 : b = real.log 49 / real.log 2) : a = (3 / 8) * b :=
sorry

end log_identity_l181_181601


namespace range_of_a_l181_181220

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ Set.Icc (-1 : ℝ) 1, |4^x_0 - a * 2^x_0 + 1| ≤ 2^(x_0 + 1)) →
  0 ≤ a ∧ a ≤ (9/2) :=
by
  sorry

end range_of_a_l181_181220


namespace solve_trig_equation_l181_181697

theorem solve_trig_equation (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) → 
  (cos y ≠ 0) → 
  ((sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z) →
  (∃ (n k m : ℤ), x = (π / 2) + π * n ∧ y = π * k ∧ z = 2 * π * m) := 
by 
  sorry

end solve_trig_equation_l181_181697


namespace biased_sample_non_representative_l181_181440

/-- 
A proof problem verifying the representativeness of a sample of 2000 email address owners concerning 
the urban population's primary sources of news.
-/
theorem biased_sample_non_representative (
  (U : Type) 
  (email_population : finset U) 
  (sample : finset U) :
  sample.card = 2000 
  ∧ sample ⊆ email_population 
  ∧ ∃ (u : U), u ∈ sample 
  → email_population.card < finset.card email_population
  := sorry

end biased_sample_non_representative_l181_181440


namespace largest_N_probability_l181_181636

theorem largest_N_probability :
  ∃ N : ℕ, N ≤ 24 ∧ (1 - ((N - 1) / 24)^2) > 0.5 ∧ ∀ M > N, (1 - ((M - 1) / 24)^2) ≤ 0.5 :=
sorry

end largest_N_probability_l181_181636


namespace z6_eq_neg8_solutions_l181_181885

noncomputable def solutions_to_z6_eq_neg8 : set ℂ :=
  { z | z ^ 6 = -8 }

theorem z6_eq_neg8_solutions :
  solutions_to_z6_eq_neg8 = { -real.rpow 2.0 (1.0 / 3.0),
                              complex.I * real.rpow 2.0 (1.0 / 3.0),
                              -complex.I * real.rpow 2.0 (1.0 / 3.0) } ∪ 
  { z | ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧
                   (z = complex.of_real(x) + complex.I * complex.of_real(y)) ∧ 
                   (x^4 - 10 * x^2 * y^2 + y^4 = 0) ∧
                   (x^6 - 15 * x^4 * y^2 + 15 * x^2 * y^4 - y^6 = -8) } :=
by sorry

end z6_eq_neg8_solutions_l181_181885


namespace area_of_given_region_l181_181107

open Real

def area_under_inequality : ℝ :=
  let region := {p : ℝ × ℝ | |2 * p.1 + 3 * p.2| + |2 * p.1 - 3 * p.2| ≤ 12}
  measure (set.univ.restrict region)

theorem area_of_given_region : area_under_inequality = 12 :=
  sorry

end area_of_given_region_l181_181107


namespace transformation_matrix_of_square_rotation_and_scaling_l181_181004

theorem transformation_matrix_of_square_rotation_and_scaling :
  let θ := -30 * Real.pi / 180
  let R := Matrix.of ![![Real.cos θ, -Real.sin θ], [Real.sin θ, Real.cos θ]]
  let S := Matrix.of ![![2, 0], [0, 2]]
  let M := S ⬝ R
  M = Matrix.of ![![Real.sqrt 3, 1], [-1, Real.sqrt 3]] := by
  let θ := -30 * Real.pi / 180
  let R := Matrix.of ![![Real.cos θ, -Real.sin θ], [Real.sin θ, Real.cos θ]]
  let S := Matrix.of ![![2, 0], [0, 2]]
  let M := S ⬝ R
  sorry

end transformation_matrix_of_square_rotation_and_scaling_l181_181004


namespace find_the_number_added_l181_181114

theorem find_the_number_added (x : ℕ) (h : x = 1) : ∃ n, x + n = 2 ∧ n = 1 :=
by {
  have h1 := h.symm,
  use 1,
  rw [h1, Nat.add_one],
  exact ⟨rfl, rfl⟩
}

end find_the_number_added_l181_181114


namespace intersecting_lines_l181_181605

noncomputable def L1 : Set Point := {p | on_line l1 p}
noncomputable def L2 : Set Point := {p | on_line l2 p}
noncomputable def P : Point := intersect_point l1 l2

theorem intersecting_lines :
  (L1 ∩ L2) = {P} := sorry

end intersecting_lines_l181_181605


namespace find_x_satisfies_fraction_eq_l181_181545

-- Lean statement for the proof problem
theorem find_x_satisfies_fraction_eq (a b : ℝ) (hb : b ≠ 0) (ha : a ≠ 1):
  let x := a / (a - 1)
  in (a + x) / (b * x) = a / b :=
by
  let x := a / (a - 1)
  sorry

end find_x_satisfies_fraction_eq_l181_181545


namespace determine_CD_in_triangle_l181_181227

theorem determine_CD_in_triangle
  (A B C D : Type) 
  [angle_ABC_150 : Angle B A C = 150]
  [AB_2 : Distance A B = 2]
  [BC_5 : Distance B C = 5]
  (perpendiculars_constructed : ∃ D, IsPerpendicularAt D A B A ∧ IsPerpendicularAt D C B C) :
  Distance C D = 5 * sqrt 3 / 3 := sorry

end determine_CD_in_triangle_l181_181227


namespace number_of_zeros_l181_181737

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2 * x - 6 + Real.log x

theorem number_of_zeros :
  (∃ x : ℝ, f x = 0) ∧ (∃ y : ℝ, f y = 0) ∧ (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
by
  sorry

end number_of_zeros_l181_181737


namespace difference_in_average_speed_l181_181363

-- Definitions
def distance : ℝ := 150
def v_R : ℝ := 22.83882181415011
def t_R : ℝ := distance / v_R

-- Conditions
def t_P : ℝ := t_R - 2
def v_P : ℝ := distance / t_P

-- Theorem statement
theorem difference_in_average_speed :
  v_P - v_R ≈ 10.008 := sorry

end difference_in_average_speed_l181_181363


namespace find_a_l181_181984

theorem find_a (a b c : ℕ) (h₁ : a + b = c) (h₂ : b + 2 * c = 10) (h₃ : c = 4) : a = 2 := by
  sorry

end find_a_l181_181984


namespace part1_part2_l181_181175

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k - |x - 3|

theorem part1 (k : ℝ) (h : ∀ x, f (x + 3) k ≥ 0 ↔ x ∈ [-1, 1]) : k = 1 :=
sorry

variable (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)

theorem part2 (h : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  (1 / 9) * a + (2 / 9) * b + (3 / 9) * c ≥ 1 :=
sorry

end part1_part2_l181_181175


namespace linear_equation_in_two_variables_l181_181855

def is_linear_equation_two_variables (eq : String → Prop) : Prop :=
  eq "D"

-- Given Conditions
def eqA (x y z : ℝ) : Prop := 2 * x + 3 * y = z
def eqB (x y : ℝ) : Prop := 4 / x + y = 5
def eqC (x y : ℝ) : Prop := 1 / 2 * x^2 + y = 0
def eqD (x y : ℝ) : Prop := y = 1 / 2 * (x + 8)

-- Problem Statement to be Proved
theorem linear_equation_in_two_variables :
  is_linear_equation_two_variables (λ s =>
    ∃ x y z : ℝ, 
      (s = "A" → eqA x y z) ∨ 
      (s = "B" → eqB x y) ∨ 
      (s = "C" → eqC x y) ∨ 
      (s = "D" → eqD x y)
  ) :=
sorry

end linear_equation_in_two_variables_l181_181855


namespace route_comparison_l181_181294

theorem route_comparison :
  let distance_X := 8
  let speed_X := 40
  let time_X := distance_X / speed_X * 60

  let distance_Y1 := 5.5
  let speed_Y1 := 50
  let time_Y1 := distance_Y1 / speed_Y1 * 60

  let distance_Y2 := 1
  let speed_Y2 := 10
  let time_Y2 := distance_Y2 / speed_Y2 * 60

  let distance_Y3 := 0.5
  let speed_Y3 := 20
  let time_Y3 := distance_Y3 / speed_Y3 * 60

  let time_Y := time_Y1 + time_Y2 + time_Y3 in
  time_X - time_Y = -2.1 :=
by
  -- Definitions of distances and speeds.
  let distance_X := 8
  let speed_X := 40
  let time_X := distance_X / speed_X * 60

  let distance_Y1 := 5.5
  let speed_Y1 := 50
  let time_Y1 := distance_Y1 / speed_Y1 * 60

  let distance_Y2 := 1
  let speed_Y2 := 10
  let time_Y2 := distance_Y2 / speed_Y2 * 60

  let distance_Y3 := 0.5
  let speed_Y3 := 20
  let time_Y3 := distance_Y3 / speed_Y3 * 60

  let time_Y := time_Y1 + time_Y2 + time_Y3 in
  show time_X - time_Y = -2.1 from sorry

end route_comparison_l181_181294


namespace smallest_positive_period_value_of_a_decreasing_intervals_l181_181973

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem smallest_positive_period (a : ℝ) : 
    ∃ T > 0, (∀ x : ℝ, f x a = f (x + T) a) ∧ (∀ T' > 0, (∀ x : ℝ, f x a = f (x + T') a) → T ≤ T') :=
sorry

theorem value_of_a (a : ℝ) : 
    (∀ x ∈ Icc (-Real.pi / 6) (Real.pi / 6), 2 * Real.sin (2 * x + Real.pi / 6) ≤ 1) →
    ∃ a, (2 * Real.max_sin_value + a + 1) + (2 * Real.min_sin_value + a + 1) = 3 → a = 0 :=
sorry

theorem decreasing_intervals (a : ℝ) :
    a = 0 →
    ∃ k : ℤ, ∀ x ∈ Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3), f x 0 > f (x + Real.pi) 0 :=
sorry

end smallest_positive_period_value_of_a_decreasing_intervals_l181_181973


namespace distance_from_point_to_line_l181_181625

noncomputable def point_polar : ℝ × ℝ := (2 * real.cos (real.pi / 3), 2 * real.sin (real.pi / 3))

noncomputable def line_polar (x y : ℝ) : Prop := x + sqrt 3 * y = 6

theorem distance_from_point_to_line : 
  let (x1, y1) := point_polar in
  ∃ d : ℝ, d = abs (x1 + sqrt 3 * y1 - 6) / sqrt (1 + (sqrt 3)^2) ∧ d = 1 :=
by
  let p := point_polar
  let (x1, y1) := p
  have distance_formula : abs (x1 + sqrt 3 * y1 - 6) / sqrt ((1:ℝ)^2 + (sqrt 3)^2) = 1 := sorry
  exact ⟨1, distance_formula⟩

end distance_from_point_to_line_l181_181625


namespace distance_A_B_l181_181191

variable (x : ℚ)

def pointA := x
def pointB := 1
def pointC := -1

theorem distance_A_B : |pointA x - pointB| = |x - 1| := by
  sorry

end distance_A_B_l181_181191


namespace domain_comp_l181_181725

theorem domain_comp {f : ℝ → ℝ} (h₁ : ∀ x : ℝ, x > 1 → ∃ y, f(x) = y) : 
  ∀ z : ℝ, z > 0 → ∃ y, f(2*z + 1) = y :=
by
  intros z hz
  have h : 2*z + 1 > 1 := by linarith
  obtain ⟨y, hy⟩ := h₁ (2*z + 1) h
  exact ⟨y, hy⟩

end domain_comp_l181_181725


namespace trapezoid_EFGH_area_l181_181878

structure Point (α : Type) :=
  (x y : α)

def trapezoid_area (E F G H : Point ℝ) : ℝ :=
  let base1 := (F.x - E.x).abs
  let base2 := (G.x - H.x).abs
  let height := (E.y - G.y).abs
  0.5 * (base1 + base2) * height

theorem trapezoid_EFGH_area : 
  let E := Point.mk (-3) 0
  let F := Point.mk 2 0
  let G := Point.mk 5 (-3)
  let H := Point.mk (-1) (-3)
  trapezoid_area E F G H = 16.5 :=
by
  sorry

end trapezoid_EFGH_area_l181_181878


namespace find_days2_l181_181408

/-
  Given:
  Depth1 : ℕ, Length1 : ℕ, Breadth1 : ℕ, Days1 : ℕ,
  Depth2 : ℕ, Length2 : ℕ, Breadth2 : ℕ
  Conditions:
  - Volume1 = Depth1 * Length1 * Breadth1
  - Volume2 = Depth2 * Length2 * Breadth2
  - Volume1 / Days1 = Volume2 / Days2,
  
  Prove:
  Days2 = 12
-/

def Depth1 : ℕ := 100
def Length1 : ℕ := 25
def Breadth1 : ℕ := 30
def Days1 : ℕ := 12

def Depth2 : ℕ := 75
def Length2 : ℕ := 20
def Breadth2 : ℕ := 50

noncomputable def Volume1 : ℕ := Depth1 * Length1 * Breadth1
noncomputable def Volume2 : ℕ := Depth2 * Length2 * Breadth2

theorem find_days2 (V1 : Volume1 = Depth1 * Length1 * Breadth1)
                   (V2 : Volume2 = Depth2 * Length2 * Breadth2)
                   (prop : Volume1 / Days1 = Volume2 / Days2) :
                   ∃ Days2 : ℕ, Days2 = 12 :=
by
  use 12
  sorry

end find_days2_l181_181408


namespace solve_system_of_floor_eqs_l181_181035

noncomputable def floor_function (x : ℝ) : ℤ := int.floor x

theorem solve_system_of_floor_eqs (x y : ℝ) (hx : floor_function (x + y - 3) = 2 - x) (hy : floor_function (x + 1) + floor_function (y - 7) + x = y) :
  x = 3 ∧ y = -1 :=
by
  sorry

end solve_system_of_floor_eqs_l181_181035


namespace probability_E_winning_bid_probability_Henan_province_winning_l181_181754

def companies : List String := ["A", "B", "C", "D", "E", "F"]

def provinces : String → String
| "A" := "Liaoning"
| "B" := "Fujian"
| "C" := "Fujian"
| "D" := "Henan"
| "E" := "Henan"
| "F" := "Henan"
| _ := "Unknown"

-- All combinations of choosing 2 companies out of 6
def combinations : List (String × String) := 
  [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"),
   ("B", "C"), ("B", "D"), ("B", "E"), ("B", "F"), 
   ("C", "D"), ("C", "E"), ("C", "F"),
   ("D", "E"), ("D", "F"),
   ("E", "F")]

-- Question (Ⅰ): The probability of Company E winning the bid is 1/3.
theorem probability_E_winning_bid : 
  (count (fun (x : String × String) => x.fst = "E" ∨ x.snd = "E") combinations) / (length combinations) = 1 / 3 := 
sorry

-- Question (Ⅱ): The probability that at least one of the winning companies is from Henan Province is 4/5.
theorem probability_Henan_province_winning : 
  (((length combinations) - (count (fun (x : String × String) => provinces x.fst ≠ "Henan" ∧ provinces x.snd ≠ "Henan") combinations)) / (length combinations)) = 4 / 5 := 
sorry

end probability_E_winning_bid_probability_Henan_province_winning_l181_181754


namespace cloth_cost_price_l181_181805

theorem cloth_cost_price :
  (∀ (A B C : ℕ) (sA lA sB lB sC lC : ℕ),
    A = 200 → sA = 10000 → lA = 1000 →
    B = 150 → sB = 6000 → lB = 450 →
    C = 100 → sC = 4000 → lC = 200 →
    (sA + lA) / A = 55 ∧ (sB + lB) / B = 43 ∧ (sC + lC) / C = 42) :=
by
  intros A B C sA lA sB lB sC lC hA hsA hlA hB hsB hlB hC hsC hlC
  simp only [hA, hsA, hlA, hB, hsB, hlB, hC, hsC, hlC]
  norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end cloth_cost_price_l181_181805


namespace distance_center_to_point_l181_181387

theorem distance_center_to_point : 
  let center := (2, 3)
  let point  := (5, -2)
  let distance := Real.sqrt ((5 - 2)^2 + (-2 - 3)^2)
  distance = Real.sqrt 34 := by
  sorry

end distance_center_to_point_l181_181387


namespace actual_distance_traveled_l181_181989

theorem actual_distance_traveled (D t : ℝ) 
  (h1 : D = 15 * t)
  (h2 : D + 50 = 35 * t) : 
  D = 37.5 :=
by
  sorry

end actual_distance_traveled_l181_181989


namespace pictures_vertical_l181_181284

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l181_181284


namespace number_of_magpies_l181_181815

/-- Definitions from the problem conditions --/
def blackbirds_per_tree : ℕ := 3
def number_of_trees : ℕ := 7
def total_birds : ℕ := 34

/-- Calculation of total blackbirds --/
def total_blackbirds : ℕ := blackbirds_per_tree * number_of_trees

/-- Statement of the proof problem --/
theorem number_of_magpies : ∃ (magpies : ℕ), magpies = total_birds - total_blackbirds :=
by
  use (total_birds - total_blackbirds)
  sorry

end number_of_magpies_l181_181815


namespace count_nat_divisors_multiple_of_3_l181_181851

theorem count_nat_divisors_multiple_of_3 :
    ∃ n : ℕ, (n = 432) ∧ (∀ m : ℕ, m ∣ (Nat.factorial 11) ∧ (3 ∣ m) → count_of_divisors_with_property m 11) :=
begin
  sorry
end

end count_nat_divisors_multiple_of_3_l181_181851


namespace intersection_of_A_and_B_l181_181945

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l181_181945


namespace evaluate_at_neg_one_l181_181579

def f (x : ℝ) : ℝ := -2 * x ^ 2 + 1

theorem evaluate_at_neg_one : f (-1) = -1 := 
by
  -- Proof goes here
  sorry

end evaluate_at_neg_one_l181_181579


namespace total_percent_samples_l181_181229

noncomputable def percentage_samples (caught_percent : ℝ) (not_caught_percent : ℝ) : ℝ :=
  (caught_percent * 100) / (100 - not_caught_percent)

theorem total_percent_samples (customers : ℕ) :
  let percent_caught := 22.0
  let percent_not_caught := 10.0
  let total := percentage_samples percent_caught percent_not_caught
  total ≈ 24.44 := by
    -- Lean does not support floating-point arithmetic by default
    -- so we use a proximate value assertion for demonstration
    sorry

end total_percent_samples_l181_181229


namespace number_of_true_propositions_l181_181646

variables {m n : Type} {α β : Type}
variables [linear_ordered_field m] [linear_ordered_field n]

def parallel (l1 l2 : Type) : Prop := sorry
def perpendicular (l1 l2 : Type) : Prop := sorry

axiom prop1 (h1 : parallel m n) (h2 : perpendicular m β) : perpendicular n β
axiom prop2 (h1 : parallel m α) (h2 : parallel m β) : parallel α β
axiom prop3 (h1 : parallel m n) (h2 : parallel m β) : parallel n β
axiom prop4 (h1 : perpendicular m α) (h2 : perpendicular m β) : perpendicular α β

theorem number_of_true_propositions : (if prop1 (parallel m n) (perpendicular m β) then 1 else 0) +
                                     (if prop2 (parallel m α) (parallel m β) then 1 else 0) +
                                     (if prop3 (parallel m n) (parallel m β) then 1 else 0) +
                                     (if prop4 (perpendicular m α) (perpendicular m β) then 1 else 0) = 1 := 
sorry

end number_of_true_propositions_l181_181646


namespace supplementary_angles_ratio_l181_181375

theorem supplementary_angles_ratio (A B : ℝ) (h1 : A + B = 180) (h2 : A / B = 5 / 4) : B = 80 :=
by
   sorry

end supplementary_angles_ratio_l181_181375


namespace total_hours_to_afford_TV_l181_181633

def TV_cost : ℝ := 1700
def initial_wage : ℝ := 10
def increased_wage : ℝ := 12
def initial_hours : ℕ := 100
def sales_tax_rate : ℝ := 0.07
def shipping_fee : ℝ := 50

theorem total_hours_to_afford_TV :
  let total_cost := TV_cost * (1 + sales_tax_rate) + shipping_fee
  let earnings_before_raise := initial_wage * initial_hours
  let remaining_amount := total_cost - earnings_before_raise
  (remaining_amount / increased_wage).ceil + initial_hours = 173 := by
  have total_cost := TV_cost * (1 + sales_tax_rate) + shipping_fee
  have earnings_before_raise := initial_wage * initial_hours
  have remaining_amount := total_cost - earnings_before_raise
  have hours_after_raise := (remaining_amount / increased_wage).ceil
  have total_hours := hours_after_raise + initial_hours
  have final_hours : ℕ := 173
  exact Eq.refl total_hours

end total_hours_to_afford_TV_l181_181633


namespace product_A_odot_B_l181_181278

noncomputable def A : Set ℤ := {-2, 1}
noncomputable def B : Set ℤ := {-1, 2}
noncomputable def A_odot_B : Set ℤ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a * b}

theorem product_A_odot_B : ∏ x in (A_odot_B.toFinset), x = 8 := by
  sorry

end product_A_odot_B_l181_181278


namespace keystone_arch_angle_l181_181733

theorem keystone_arch_angle (n : ℕ) (h1 : n = 9)
  (isosceles_trapezoids : ∀ i : ℕ, i < n → is_isosceles (trapezoid i))
  (fitted_non_parallel : ∀ i : ℕ, i < n - 1 → non_parallel (fit (trapezoid i) (trapezoid (i + 1))))
  (horizontal_bases : is_horizontal (base (trapezoid 0)) ∧ is_horizontal (base (trapezoid (n - 1))))
  : angle (interior_angle (large_face (trapezoid 0))) = 100 :=
  sorry

end keystone_arch_angle_l181_181733


namespace calc_value_l181_181076

theorem calc_value (a : ℝ) (h : a = 1024) : (a ^ 0.25) * (a ^ 0.2) = 16 * Real.sqrt 2 := by
  sorry

end calc_value_l181_181076


namespace color_points_l181_181674

theorem color_points (S : Finset (ℤ × ℤ)) :
  ∃ (red white : Finset (ℤ × ℤ)), 
    red ∪ white = S ∧ red ∩ white = ∅ ∧
    (∀ L : ℤ, ∃ (rL wL : Finset (ℤ × ℤ)), 
      (rL ∪ wL = S.filter (λ p, p.1 = L) ∨ 
      rL ∪ wL = S.filter (λ p, p.2 = L)) ∧ 
      rL ∩ wL = ∅ ∧ |rL.card - wL.card| ≤ 1) := 
sorry

end color_points_l181_181674


namespace unique_solution_for_all_y_l181_181115

theorem unique_solution_for_all_y (x : ℝ) (h : ∀ y : ℝ, 8 * x * y - 12 * y + 2 * x - 3 = 0) : x = 3 / 2 :=
sorry

end unique_solution_for_all_y_l181_181115


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181896
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181896


namespace problem_condition_l181_181266

variable {A B : ℝ}

-- Functions definitions
def f (x : ℝ) : ℝ := A * x + B
def g (x : ℝ) : ℝ := B * x + A

-- Conditions
theorem problem_condition (A B : ℝ) (h : A ≠ B) (h1 : f (g x) - g (f x) = B - A) : A + B = 0 := 
begin
  sorry
end

end problem_condition_l181_181266


namespace count_nat_divisors_multiple_of_3_l181_181850

theorem count_nat_divisors_multiple_of_3 :
    ∃ n : ℕ, (n = 432) ∧ (∀ m : ℕ, m ∣ (Nat.factorial 11) ∧ (3 ∣ m) → count_of_divisors_with_property m 11) :=
begin
  sorry
end

end count_nat_divisors_multiple_of_3_l181_181850


namespace sum_k_binomial_l181_181908

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181908


namespace parametric_equations_hyperbola_l181_181343

variable {θ t : ℝ}
variable (n : ℤ)

theorem parametric_equations_hyperbola (hθ : θ ≠ (n / 2) * π)
  (hx : ∀ t, x t = 1 / 2 * (Real.exp t + Real.exp (-t)) * Real.cos θ)
  (hy : ∀ t, y t = 1 / 2 * (Real.exp t - Real.exp (-t)) * Real.sin θ) :
  (∀ t, (x t)^2 / (Real.cos θ)^2 - (y t)^2 / (Real.sin θ)^2 = 1) := sorry

end parametric_equations_hyperbola_l181_181343


namespace adam_change_l181_181474

noncomputable def calculate_change (initial_amount : ℝ) (cost : ℝ) (tax_rate : ℝ) (additional_fee : ℝ) : ℝ :=
let tax := (cost * tax_rate).round;
let total_cost := cost + tax;
let total_cost_with_fee := total_cost + additional_fee;
initial_amount - total_cost_with_fee

theorem adam_change : 
  calculate_change 5 4.28 0.07 0.35 = 0.07 :=
by
  rw calculate_change
  sorry

end adam_change_l181_181474


namespace concurrency_of_lines_l181_181673

noncomputable def centers_of_excircles (A B C I_A I_B I_C : Type) : Prop :=
  ∃ (excircles : I_A = center_of_excircle A B C BC ∧ 
                 I_B = center_of_excircle A B C AC ∧ 
                 I_C = center_of_excircle A B C AB), 
  excircles

noncomputable def perp_intersection (I_A I_B X_C : Type) : Prop :=
  ∃ (perp_I_A_AC : meth1 I_A AC) (perp_I_B_BC : meth2 I_B BC), 
  perp_I_A_AC = perp_I_B_BC → X_C = intersection_method perp_I_A_AC perp_I_B_BC

noncomputable def defined_points_X (I_A I_B I_C X_A X_B X_C : Type) : Prop :=
  perp_intersection I_A I_B X_C ∧ 
  perp_intersection I_C I_A X_A ∧ 
  perp_intersection I_B I_C X_B

theorem concurrency_of_lines (A B C I_A I_B I_C X_A X_B X_C : Type)
  (hc : centers_of_excircles A B C I_A I_B I_C)
  (hp : defined_points_X I_A I_B I_C X_A X_B X_C) :
  intersects_at_one_point (I_A X_A) (I_B X_B) (I_C X_C) := 
sorry

end concurrency_of_lines_l181_181673


namespace find_n_l181_181761

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 13 ∧ 52801 ≡ n [MOD 13] :=
begin
  use 8,
  split,
  { exact zero_le (8 : ℤ), },
  split,
  { norm_num, },
  { norm_num, exact rfl, }
end

end find_n_l181_181761


namespace penguins_difference_l181_181072

def sea_lions := 48
def ratio_sea_lions_to_penguins := 4 / 11
def number_of_penguins (sea_lions : ℕ) (ratio : ℚ) : ℕ := (sea_lions * ratio.denom) / ratio.num

theorem penguins_difference :
  number_of_penguins sea_lions ratio_sea_lions_to_penguins - sea_lions = 84 :=
by
  -- This is the theorem stating the difference in number of penguins and sea lions
  sorry

end penguins_difference_l181_181072


namespace quadratic_real_roots_l181_181994

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l181_181994


namespace product_of_N_l181_181487

theorem product_of_N (M L : ℝ) (N : ℝ) 
  (h1 : M = L + N) 
  (h2 : ∀ M4 L4 : ℝ, M4 = M - 7 → L4 = L + 5 → |M4 - L4| = 4) :
  N = 16 ∨ N = 8 ∧ (16 * 8 = 128) := 
by 
  sorry

end product_of_N_l181_181487


namespace dan_stationery_spent_l181_181502

def total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def notebook_cost : ℕ := 3
def number_of_notebooks : ℕ := 5
def stationery_cost_each : ℕ := 1

theorem dan_stationery_spent : 
  (total_spent - (backpack_cost + notebook_cost * number_of_notebooks)) = 2 :=
by
  sorry

end dan_stationery_spent_l181_181502


namespace sum_and_round_l181_181013

theorem sum_and_round :
  let sum := 5.67 + 2.45 in
  Float.round (sum * 10.0) / 10.0 = 8.1 :=
by
  let sum := 5.67 + 2.45
  have h : Float.round (sum * 10.0) / 10.0 = 8.1
  sorry

end sum_and_round_l181_181013


namespace find_a_for_even_function_l181_181166

open Function

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- State the problem in Lean.
theorem find_a_for_even_function :
  (∀ x, f a x = f a (-x)) → a = 1 :=
sorry

end find_a_for_even_function_l181_181166


namespace not_age_of_child_l181_181292

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

theorem not_age_of_child :
  ∃ n : Nat, 
  let T := { t : Nat | t ∈ {1, 2, 3, 4, 5, 6, 8, 9, 10} } in
  ∀ t ∈ T, divides t n ∧ 
  (n % 100 = n % 10 * 10 + n % 10) ∧ 
  (∀ x ∈ {5, 6, 7, 8, 9}, ¬x ∈ T) ∧ 
  ¬divides 7 n := 
sorry

end not_age_of_child_l181_181292


namespace complex_roots_circle_radius_l181_181066

theorem complex_roots_circle_radius (z : ℂ) (h : (z - 2)^6 = 64 * z^6) : 
  ∃ r : ℝ, r = 2 * real.sqrt 3 / 3 ∧ ∀ z, (z - 2)^6 = 64 * z^6 → complex.abs z = r :=
sorry

end complex_roots_circle_radius_l181_181066


namespace find_angle_C_calculate_area_l181_181135

noncomputable def problem_conditions (A B C: ℝ) (a b c: ℝ) :=
  (C = (π/6)) ∧ (c / a = 2) ∧ (b = 4 * sqrt 3) ∧ ((sqrt 3 * c) / (Real.cos C) = a / (Real.cos (3 * π / 2 + A)))

theorem find_angle_C (A B: ℝ) (a b c: ℝ) (h : C = π / 6) :
  ∃ C, ((sqrt 3 * c) / (Real.cos C) = a / (Real.cos (3 * π / 2 + A))) → C = π / 6 :=
sorry

theorem calculate_area (A B: ℝ) (a b c: ℝ) :
  problem_conditions A B (π / 6) a b c →
  ∃ S, S = 2 * sqrt 15 - 2 * sqrt 3 ∧ S = (1 / 2) * a * b * (Real.sin (π / 6)) :=
sorry

end find_angle_C_calculate_area_l181_181135


namespace hyperbola_dot_product_l181_181610

theorem hyperbola_dot_product (F1 F2 P Q : ℝ × ℝ) :
  (F1 = (-2, 0)) → (F2 = (2, 0)) →
  (∃ l : Set (ℝ × ℝ), l F1 ∧ (l ∩ (SetOf x y, x^2 - y^2 / 3 = 1) = {P, Q})) →
  (4 * (P.1 + 2) = 16) →
  let FP := (P.1 - F2.1, P.2)
  let FQ := (Q.1 - F2.1, Q.2)
  (FP.1 * FQ.1 + FP.2 * FQ.2 = 27 / 13) := sorry

end hyperbola_dot_product_l181_181610


namespace calculate_boundaries_l181_181789

noncomputable def runs_made_by_running (total_score : ℝ) (running_percentage : ℝ) : ℝ :=
  (running_percentage / 100) * total_score

def runs_from_sixes (number_of_sixes : ℝ) : ℝ :=
  number_of_sixes * 6

def runs_from_boundaries (total_score : ℝ) (runs_by_running : ℝ) (runs_from_sixes : ℝ) : ℝ :=
  total_score - runs_by_running - runs_from_sixes

def number_of_boundaries (runs_from_boundaries : ℝ) : ℝ :=
  runs_from_boundaries / 4

theorem calculate_boundaries 
  (total_score : ℝ := 120)
  (number_of_sixes : ℝ := 5)
  (running_percentage : ℝ := 58.333333333333336) :
  number_of_boundaries (runs_from_boundaries total_score (runs_made_by_running total_score running_percentage)
                                               (runs_from_sixes number_of_sixes)) = 5 := 
by 
  sorry

end calculate_boundaries_l181_181789


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181893
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181893


namespace optimal_bathhouse_location_l181_181252

/-- Define the conditions and prove that the optimal location for the bathhouse is in village A, which minimizes the total travel distance. --/
theorem optimal_bathhouse_location (a : ℝ) (h_a : 0 ≤ a) : ∃ x, (0 ≤ x ∧ x ≤ a) ∧ ∀ (x' : ℝ), (0 ≤ x' ∧ x' ≤ a) → 
  (100 * x + 100 * (a - x) ≤ 100 * x' + 100 * (a - x')) :=
by 
  use 0
  intro x'
  intro h_x'
  sorry

end optimal_bathhouse_location_l181_181252


namespace maximum_area_of_garden_l181_181839

theorem maximum_area_of_garden (w l : ℝ) 
  (h_perimeter : 2 * w + l = 400) : 
  ∃ (A : ℝ), A = 20000 ∧ A = w * l ∧ l = 400 - 2 * w ∧ ∀ (w' : ℝ) (l' : ℝ),
    2 * w' + l' = 400 → w' * l' ≤ 20000 :=
by
  sorry

end maximum_area_of_garden_l181_181839


namespace average_speed_of_trip_l181_181980

noncomputable def first_flight_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time
noncomputable def second_flight_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time
noncomputable def return_flight_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem average_speed_of_trip :
  let speed1 := 140
  let time1 := 2
  let speed2 := 88
  let time2 := 1.5
  let speed3 := 73
  let time3 := 3.5
  let total_distance := first_flight_distance speed1 time1 + second_flight_distance speed2 time2 + return_flight_distance speed3 time3
  let total_time := time1 + time2 + time3
  total_distance / total_time ≈ 95.36 :=
begin
  sorry
end

end average_speed_of_trip_l181_181980


namespace polynomial_remainder_zero_l181_181764

-- Definitions for the given conditions
def P := 3 * (x : ℝ) * x - 20 * x + 32    -- Polynomial 3x^2 - 20x + 32
def D := x - 4                             -- Divisor x - 4

-- The main statement we want to prove
theorem polynomial_remainder_zero (x : ℝ) : 
  (P / D).snd = 0 :=
sorry

end polynomial_remainder_zero_l181_181764


namespace nonagon_line_segments_not_adjacent_l181_181975

def nonagon_segments (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

theorem nonagon_line_segments_not_adjacent (h : ∃ n, n = 9) :
  nonagon_segments 9 = 27 :=
by
  -- proof omitted
  sorry

end nonagon_line_segments_not_adjacent_l181_181975


namespace percentage_increase_from_350_to_525_is_50_l181_181419

variable (initial final : ℕ) (percentageIncrease : ℚ)

def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) / initial : ℚ) * 100

theorem percentage_increase_from_350_to_525_is_50 :
  initial = 350 →
  final = 525 →
  percentage_increase initial final = 50 :=
by
  intros h_initial h_final
  rw [←h_initial, ←h_final]
  dsimp [percentage_increase]
  norm_num
  simp
  sorry

end percentage_increase_from_350_to_525_is_50_l181_181419


namespace part_1_part_2_l181_181180

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

-- (Part 1): Prove the value of a
theorem part_1 (a : ℝ) (P : ℝ × ℝ) (hP : P = (a, -4)) :
  (∃ t : ℝ, ∃ t₂ : ℝ, t ≠ t₂ ∧ P.2 = (2 * t^3 - 3 * t^2 + 1) + (6 * t^2 - 6 * t) * (a - t)) →
  a = -1 ∨ a = 7 / 2 :=
sorry

-- (Part 2): Prove the range of k
noncomputable def g (x k : ℝ) : ℝ := k * x + 1 - Real.log x

noncomputable def h (x k : ℝ) : ℝ := min (f x) (g x k)

theorem part_2 (k : ℝ) :
  (∀ x > 0, h x k = 0 → (x = 1 ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 k = 0 ∧ h x2 k = 0)) →
  0 < k ∧ k < 1 / Real.exp 2 :=
sorry

end part_1_part_2_l181_181180


namespace least_sum_of_exponents_1024_l181_181521

theorem least_sum_of_exponents_1024 :
  (∃ exponents : List ℕ, (∑ k in exponents, 2^k) = 1024 ∧ (∀ i j ∈ exponents, i ≠ j) ∧ (∑ k in exponents, k) = 10) :=
sorry

end least_sum_of_exponents_1024_l181_181521


namespace speed_of_stream_l181_181423

theorem speed_of_stream (v : ℝ) (h1 : 22 > 0) (h2 : 8 > 0) (h3 : 216 = (22 + v) * 8) : v = 5 := 
by 
  sorry

end speed_of_stream_l181_181423


namespace product_sine_identity_l181_181676

theorem product_sine_identity (n : ℕ) :
  2^n * (∏ k in Finset.range n, Real.sin (k + 1) * Real.pi / (2 * n + 1)) = Real.sqrt (2 * n + 1) :=
sorry

end product_sine_identity_l181_181676


namespace sample_not_representative_l181_181435

-- Definitions
def has_email_address (person : Type) : Prop := sorry
def uses_internet (person : Type) : Prop := sorry

-- Problem statement: prove that the sample is not representative of the urban population.
theorem sample_not_representative (person : Type)
  (sample : set person)
  (h_sample_size : set.size sample = 2000)
  (A : person → Prop)
  (A_def : ∀ p, A p ↔ has_email_address p)
  (B : person → Prop)
  (B_def : ∀ p, B p ↔ uses_internet p)
  (dependent : ∀ p, A p → B p)
  : ¬ is_representative_sample sample := 
sorry

end sample_not_representative_l181_181435


namespace part1_part2_l181_181140

variable (α : Real)
-- Condition
axiom tan_neg_alpha : Real.tan (-α) = -2

-- Question 1
theorem part1 : ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α)) = 3 := 
by
  sorry

-- Question 2
theorem part2 : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end part1_part2_l181_181140


namespace force_saved_correct_l181_181385

-- Definitions based on the problem's conditions
def R : ℝ := 1000 -- Resistance (in Newtons)
def R_arm : ℝ := 0.6 -- Resistance arm (in meters)
def L_initial : ℝ := 1.5 -- Initial effort arm (in meters)
def L_final : ℝ := 2 -- Final effort arm (in meters)

-- Effort force calculation based on lever principle
def effort_force (L : ℝ) : ℝ := (R * R_arm) / L

-- Force initial and final calculation
def F_initial : ℝ := effort_force L_initial
def F_final : ℝ := effort_force L_final

-- Force saved calculation
def F_saved : ℝ := F_initial - F_final

-- Proof statement
theorem force_saved_correct :
  F_saved = 100 := by
  sorry

end force_saved_correct_l181_181385


namespace typing_speed_reduction_l181_181831

/-- Barbara Blackburn's original typing speed is 212 words per minute.
    She types a 3440-word document in 20 minutes.
    Prove that her typing speed has reduced by 40 words per minute. -/
theorem typing_speed_reduction : 
  let original_speed := 212
  let typed_words := 3440
  let time_minutes := 20
  let current_speed := typed_words / time_minutes in
  original_speed - current_speed = 40 :=
by
  let original_speed := 212
  let typed_words := 3440
  let time_minutes := 20
  let current_speed := typed_words / time_minutes
  calc
    original_speed - current_speed
          = 212 - (3440 / 20)  : by rfl
      ... = 212 - 172         : by rfl
      ... = 40                : by rfl

end typing_speed_reduction_l181_181831


namespace selling_price_calculation_l181_181054

variable {CostPrice : ℝ}
variable {ProfitPercentage : ℝ}
variable {SellingPrice : ℝ}

-- Given conditions
def cost_price := 71.43
def profit_percentage := 40

-- The profit calculation
def profit (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  (profit_percentage / 100) * cost_price

-- The selling price calculation
def selling_price (cost_price : ℝ) (profit : ℝ) : ℝ :=
  cost_price + profit

-- The statement of our proof problem
theorem selling_price_calculation :
  let p := profit cost_price profit_percentage in
  selling_price cost_price p = 100.00 :=
by
  sorry

end selling_price_calculation_l181_181054


namespace degree_of_d_l181_181459

noncomputable def f : Polynomial ℝ := sorry
noncomputable def d : Polynomial ℝ := sorry
noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := 5 * Polynomial.X^2 + 3 * Polynomial.X - 8

axiom deg_f : f.degree = 15
axiom deg_q : q.degree = 7
axiom deg_r : r.degree = 2
axiom poly_div : f = d * q + r

theorem degree_of_d : d.degree = 8 :=
by
  sorry

end degree_of_d_l181_181459


namespace inequality_solution_l181_181876

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 2) :
  ( (x + 1)/(x - 2) + (x + 3)/(3 * x) ≥ 4 ) ↔ (x ∈ Ioo 0 (1/2) ∪ Ioc 2 (11/2) ) := by
  sorry

end inequality_solution_l181_181876


namespace log_sum_seq_l181_181186

noncomputable def a_seq (n : ℕ) : ℤ := sorry -- Define the sequence a(n)

theorem log_sum_seq (h₁ : ∀ n : ℕ, 5^(a_seq (n + 1)) = 25 * 5^(a_seq n))
(h₂ : a_seq 2 + a_seq 4 + a_seq 6 = 9) : 
log (3⁻¹) ((a_seq 5) + (a_seq 7) + (a_seq 9)) = -3 := 
sorry

end log_sum_seq_l181_181186


namespace telescoping_series_l181_181640

noncomputable def infinite_series (c b : ℝ) := 
  ∑' n, (1 / ([(n-1 : ℕ) * c - (n-2 : ℕ) * b] * [(n : ℕ) * c - (n-1 : ℕ) * b]))

theorem telescoping_series (a b c : ℝ) (h1 : 0 < c) (h2 : 0 < b) (h3 : 0 < a) 
  (h4 : a > b) (h5 : b > c) :
  infinite_series c b = (1 / ((c - b) * b)) :=
by sorry

end telescoping_series_l181_181640


namespace f_plus_one_odd_l181_181988

noncomputable def f : ℝ → ℝ := sorry

theorem f_plus_one_odd (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1) :
  ∀ x : ℝ, f x + 1 = -(f (-x) + 1) :=
sorry

end f_plus_one_odd_l181_181988


namespace count_two_digit_decimals_between_0_40_and_0_50_l181_181979

theorem count_two_digit_decimals_between_0_40_and_0_50 : 
  ∃ (n : ℕ), n = 9 ∧ ∀ x : ℝ, 0.40 < x ∧ x < 0.50 → (exists d : ℕ, (1 ≤ d ∧ d ≤ 9 ∧ x = 0.4 + d * 0.01)) :=
by
  sorry

end count_two_digit_decimals_between_0_40_and_0_50_l181_181979


namespace probability_of_less_than_10_minutes_wait_l181_181793

noncomputable def probability_less_than_10_minutes_waiting (arrival : ℝ) (departures : List ℝ) : ℝ :=
  if arrival ∈ (Icc 7.50 (8:00)) ∨ arrival ∈ (Icc 8.20 (8:30)) then 1 else 0 -- Define favorable conditions
  
theorem probability_of_less_than_10_minutes_wait (arrival : ℝ) (departures : List ℝ) (h_arrival_range : Icc 7.50 8.30 arrival) : 
  Probability (λ t, probability_less_than_10_minutes_waiting t departures = 1) = 1 / 2 :=
by
  sorry

end probability_of_less_than_10_minutes_wait_l181_181793


namespace solve_equation_l181_181702

theorem solve_equation (x y z : ℝ) (n k m : ℤ)
  (h1 : sin x ≠ 0)
  (h2 : cos y ≠ 0)
  (h_eq : (sin x ^ 2 + 1 / sin x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * cos z)
  : ∃ n k m : ℤ, x = π / 2 + π * n ∧ y = π * k ∧ z = 2 * π * m :=
by
  sorry

end solve_equation_l181_181702


namespace number_of_positions_forming_cube_with_missing_face_l181_181356

-- Define the polygon formed by 6 congruent squares in a cross shape
inductive Square
| center : Square
| top : Square
| bottom : Square
| left : Square
| right : Square

-- Define the indices for the additional square positions
inductive Position
| pos1 : Position
| pos2 : Position
| pos3 : Position
| pos4 : Position
| pos5 : Position
| pos6 : Position
| pos7 : Position
| pos8 : Position
| pos9 : Position
| pos10 : Position
| pos11 : Position

-- Define a function that takes a position and returns whether the polygon can form the missing-face cube
def can_form_cube_missing_face : Position → Bool
  | Position.pos1   => true
  | Position.pos2   => true
  | Position.pos3   => true
  | Position.pos4   => true
  | Position.pos5   => false
  | Position.pos6   => false
  | Position.pos7   => false
  | Position.pos8   => false
  | Position.pos9   => true
  | Position.pos10  => true
  | Position.pos11  => true

-- Count valid positions for forming the cube with one face missing
def count_valid_positions : Nat :=
  List.length (List.filter can_form_cube_missing_face 
    [Position.pos1, Position.pos2, Position.pos3, Position.pos4, Position.pos5, Position.pos6, Position.pos7, Position.pos8, Position.pos9, Position.pos10, Position.pos11])

-- Prove that the number of valid positions is 7
theorem number_of_positions_forming_cube_with_missing_face : count_valid_positions = 7 :=
  by
    -- Implementation of the proof
    sorry

end number_of_positions_forming_cube_with_missing_face_l181_181356


namespace find_integers_3_9_l181_181865

theorem find_integers_3_9 (a : ℕ) (h_pos : a > 0) :
  (a = 3 ∨ a = 9) ↔ (∀ n, (∀ d : Fin (n + 1), (d.val = 0 ∨ d.val = 2) ∧ (d = 0 → ¬ (d = 0))) →
                       let m := ∑ i in finset.range n, d.val * 10^i in 
                       (m % a ≠ 0)) :=
sorry

end find_integers_3_9_l181_181865


namespace find_xy_l181_181099

theorem find_xy (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : 
  2^x - 5 = 11^y ↔ (x = 4 ∧ y = 1) :=
by sorry

end find_xy_l181_181099


namespace find_parabola_and_new_vertex_l181_181527

-- Definitions
def vertex_of_parabola (a h k : ℝ) := (h, k)
def shifted_vertex (original_vertex : ℝ × ℝ) (shift : ℝ × ℝ) : ℝ × ℝ :=
( original_vertex.1 + shift.1, original_vertex.2 + shift.2 )

lemma parabola_equation_vertex_form (a : ℝ) (x : ℝ) (h : ℝ) (k : ℝ) : ℝ :=
a * (x - h) ^ 2 + k

noncomputable def initial_parabola (a : ℝ) : ℝ → ℝ := parabola_equation_vertex_form a 3 (-2)

-- Lean problem statement
theorem find_parabola_and_new_vertex : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * x^2 - 12 * x + 16) ∧
  (shifted_vertex (3, -2) (2, 3) = (5, 1)) :=
begin
  sorry
end

end find_parabola_and_new_vertex_l181_181527


namespace gabi_final_prices_l181_181544

theorem gabi_final_prices (x y : ℝ) (hx : 0.8 * x = 1.2 * y) (hl : (x - 0.8 * x) + (y - 1.2 * y) = 10) :
  x = 30 ∧ y = 20 := sorry

end gabi_final_prices_l181_181544


namespace sin_cos_product_value_l181_181983

noncomputable def sin_cos_product (θ : ℝ) (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) : ℝ :=
sin θ * cos θ

theorem sin_cos_product_value {θ : ℝ} (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin_cos_product θ h = 3 / 10 :=
sorry

end sin_cos_product_value_l181_181983


namespace moskvich_halfway_from_zhiguli_to_b_l181_181828

-- Define the Moskvich's and Zhiguli's speeds as real numbers
variables (u v : ℝ)

-- Define the given conditions as named hypotheses
axiom speed_condition : u = v
axiom halfway_condition : u = (1 / 2) * (u + v) 

-- The mathematical statement we want to prove
theorem moskvich_halfway_from_zhiguli_to_b (speed_condition : u = v) (halfway_condition : u = (1 / 2) * (u + v)) : 
  ∃ t : ℝ, t = 2 := 
sorry -- Proof omitted

end moskvich_halfway_from_zhiguli_to_b_l181_181828


namespace place_mat_length_l181_181803

theorem place_mat_length
  (R : ℝ) (x : ℝ) (w : ℝ) (n : ℕ)
  (hr : R = 5)
  (hw : w = 1)
  (hn : n = 8)
  (corner_condition : ∀ (i : ℕ), i < n → 
    let θ := 2 * real.pi / n in
    2 * R * real.sin (θ / 2) = x) :
  x = 5.475 :=
by
  sorry

end place_mat_length_l181_181803


namespace years_in_future_l181_181073

theorem years_in_future (Shekhar Shobha : ℕ) (h1 : Shekhar / Shobha = 4 / 3) (h2 : Shobha = 15) (h3 : Shekhar + t = 26)
  : t = 6 :=
by
  sorry

end years_in_future_l181_181073


namespace product_fraction_eq_714_l181_181841

theorem product_fraction_eq_714 :
    ∏ n in Finset.range (30) (λ n, (n + 6) / (n + 3)) = 714 := sorry

end product_fraction_eq_714_l181_181841


namespace probability_of_sum_11_l181_181996

-- Define the properties of a six-faced die roll
def is_valid_die_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

-- Define the event that three dice rolls sum to 11
def event_sum_11 (a b c : ℕ) : Prop := is_valid_die_roll(a) ∧ is_valid_die_roll(b) ∧ is_valid_die_roll(c) ∧ a + b + c = 11

-- Define the total number of possible outcomes when rolling three dice
def total_possibilities : ℕ := 6 * 6 * 6

-- Count the number of valid outcomes that sum to 11
def count_valid_outcomes : ℕ := 
  (if event_sum_11(1, 4, 6) ∨ event_sum_11(1, 5, 5) ∨
      event_sum_11(2, 3, 6) ∨ event_sum_11(2, 4, 5) ∨ event_sum_11(2, 5, 4) ∨ event_sum_11(3, 2, 6) ∨
      event_sum_11(3, 5, 3) ∨ event_sum_11(3, 6, 2) ∨ event_sum_11(4, 1, 6) ∨ event_sum_11(4, 2, 5) ∨
      event_sum_11(4, 3, 4) ∨ event_sum_11(4, 4, 3) ∨ event_sum_11(4, 5, 2) ∨ event_sum_11(4, 6, 1) ∨
      event_sum_11(5, 1, 5) ∨ event_sum_11(5, 2, 4) ∨ event_sum_11(5, 3, 3) ∨ event_sum_11(5, 4, 2) ∨
      event_sum_11(5, 5, 1) ∨ event_sum_11(6, 1, 4) ∨ event_sum_11(6, 2, 3) ∨ event_sum_11(6, 3, 2)) then 24 else 0

-- The probability of the event happening
def probability_sum_11 : ℚ := count_valid_outcomes / total_possibilities

theorem probability_of_sum_11 :
  probability_sum_11 = 1 / 9 :=
by
  sorry

end probability_of_sum_11_l181_181996


namespace integer_solution_count_l181_181197

theorem integer_solution_count : 
  {n : ℤ | (n - 3) * (n + 5) ≤ 12}.card = 13 := 
sorry

end integer_solution_count_l181_181197


namespace range_of_m_l181_181171

theorem range_of_m (m : ℝ) (P : ℝ × ℝ)
  (hP : P = (m, 2))
  (h1 : ∃ l, ∃ A B, l P ∧ l A ∧ l B ∧ (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1))
  (h2 : ∀ Q : ℝ × ℝ, vector_add (vector Q P) (vector Q B) = vector_scale 2 (vector Q A)) :
  -real.sqrt 5 ≤ m ∧ m ≤ real.sqrt 5 :=
sorry

end range_of_m_l181_181171


namespace rectangle_perimeter_l181_181802

theorem rectangle_perimeter (a b : ℤ) (h1 : a ≠ b) (h2 : 2 * (2 * a + 2 * b) - a * b = 12) : 2 * (a + b) = 26 :=
sorry

end rectangle_perimeter_l181_181802


namespace num_divisors_multiple_of_3_l181_181848

theorem num_divisors_multiple_of_3 (n : ℕ) (h : n = 11!) : 
  (∃ d : ℕ, d | n ∧ 3 ∣ d) → finset.card {d : ℕ | d ∣ n ∧ 3 ∣ d} = 432 :=
by {
  sorry
}

end num_divisors_multiple_of_3_l181_181848


namespace increasing_interval_f_on_0_pi_l181_181179

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 4))

theorem increasing_interval_f_on_0_pi :
  ∃ I : set ℝ, I = set.Icc 0 (Real.pi / 8) ∧ (∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y) :=
by
  sorry

end increasing_interval_f_on_0_pi_l181_181179


namespace sum_of_coefficients_l181_181981

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (1 - 3 * x + x^2)^5

theorem sum_of_coefficients :
  let a := polynomial_expansion 0,
      sum_of_coeff := polynomial_expansion 1 - polynomial_expansion 0
  in sum_of_coeff = -2 :=
begin
  -- The exact proof steps will go here
  sorry
end

end sum_of_coefficients_l181_181981


namespace find_a_for_even_l181_181152

def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

theorem find_a_for_even (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = f a x) ↔ a = 1 :=
by
  -- proof steps here
  sorry

end find_a_for_even_l181_181152


namespace number_of_people_study_only_cooking_l181_181234

def total_yoga : Nat := 25
def total_cooking : Nat := 18
def total_weaving : Nat := 10
def cooking_and_yoga : Nat := 5
def all_three : Nat := 4
def cooking_and_weaving : Nat := 5

theorem number_of_people_study_only_cooking :
  (total_cooking - (cooking_and_yoga + cooking_and_weaving - all_three)) = 12 :=
by
  sorry

end number_of_people_study_only_cooking_l181_181234


namespace egg_distribution_l181_181366

-- Definitions of the conditions
def total_eggs := 10.0
def large_eggs := 6.0
def small_eggs := 4.0

def box_A_capacity := 5.0
def box_B_capacity := 4.0
def box_C_capacity := 6.0

def at_least_one_small_egg (box_A_small box_B_small box_C_small : Float) := 
  box_A_small >= 1.0 ∧ box_B_small >= 1.0 ∧ box_C_small >= 1.0

-- Problem statement
theorem egg_distribution : 
  ∃ (box_A_small box_A_large box_B_small box_B_large box_C_small box_C_large : Float),
  box_A_small + box_A_large <= box_A_capacity ∧
  box_B_small + box_B_large <= box_B_capacity ∧
  box_C_small + box_C_large <= box_C_capacity ∧
  box_A_small + box_B_small + box_C_small = small_eggs ∧
  box_A_large + box_B_large + box_C_large = large_eggs ∧
  at_least_one_small_egg box_A_small box_B_small box_C_small :=
sorry

end egg_distribution_l181_181366


namespace distance_house_to_market_l181_181484

-- Define each of the given conditions
def distance_to_school := 50
def distance_to_park_from_school := 25
def return_distance := 60
def total_distance_walked := 220

-- Proven distance to the market
def distance_to_market := 85

-- Statement to prove
theorem distance_house_to_market (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = distance_to_school) 
  (h2 : d2 = distance_to_park_from_school) 
  (h3 : d3 = return_distance) 
  (h4 : d4 = total_distance_walked) :
  d4 - (d1 + d2 + d3) = distance_to_market := 
by
  sorry

end distance_house_to_market_l181_181484


namespace unpainted_cube_count_is_correct_l181_181785

def unit_cube_count : ℕ := 6 * 6 * 6
def opposite_faces_painted_squares : ℕ := 16 * 2
def remaining_faces_painted_squares : ℕ := 9 * 4
def total_painted_squares (overlap_count : ℕ) : ℕ :=
  opposite_faces_painted_squares + remaining_faces_painted_squares - overlap_count
def overlap_count : ℕ := 4 * 2
def painted_cubes : ℕ := total_painted_squares overlap_count
def unpainted_cubes : ℕ := unit_cube_count - painted_cubes

theorem unpainted_cube_count_is_correct : unpainted_cubes = 156 := by
  sorry

end unpainted_cube_count_is_correct_l181_181785


namespace square_dodecagon_intersection_cube_dodecahedron_intersection_l181_181464

open geometric

-- Part I: Circle and Square
theorem square_dodecagon_intersection
  (O : Point)
  (k : Circle O)
  (A B C D : Point)
  (H1 : k.inscribed_square A B C D)
  (K L M : Point)
  (H2 : midpoint A B K)
  (H3 : midpoint B C L)
  (H4 : midpoint C D M)
  (E F : Point)
  (H5 : ray_through M L k E)
  (H6 : ray_through K L k F) :
  regular_dodecagon (sides := {E, F}) (vertices := {A, B, C, D}) :=
sorry

-- Part II: Sphere and Cube
theorem cube_dodecahedron_intersection
  (O : Point)
  (g : Sphere O)
  (A B C D A1 B1 C1 D1 : Point)
  (H1 : g.inscribed_cube A B C D A1 B1 C1 D1)
  (Q R S : Point)
  (H2 : midpoint_face B C C1 B1 Q)
  (H3 : midpoint_face A D D1 A1 R)
  (H4 : midpoint_face A B C D S)
  (E F : Point)
  (H5 : ray_through Q S g E)
  (H6 : ray_through R S g F) :
  regular_dodecahedron (edges := {E, F}) (vertices := {A, B, C, D, A1, B1, C1, D1}) :=
sorry

end square_dodecagon_intersection_cube_dodecahedron_intersection_l181_181464


namespace max_points_two_two_max_points_general_l181_181814

-- Problem statement for part (a)
theorem max_points_two_two (n k : ℕ) (hn : n = 2) (hk : k = 2) : 
  max_points_guaranteed n k = 1 := 
sorry

-- Problem statement for part (b)
theorem max_points_general (n k : ℕ) : 
  max_points_guaranteed n k = n / k :=
sorry

-- Definitions to be used in the statements above
def max_points_guaranteed (n k : ℕ) : ℕ := 
  if h : n = k ∧ k = 2 
  then 1 
  else n / k


end max_points_two_two_max_points_general_l181_181814


namespace infinitely_many_primes_congruent_2_mod_3_l181_181305

theorem infinitely_many_primes_congruent_2_mod_3 : ∀ (p : ℕ), prime p → (p % 3 = 2) → ∃ (q : ℕ), prime q ∧ q % 3 = 2 ∧ q > p :=
begin
  sorry
end

end infinitely_many_primes_congruent_2_mod_3_l181_181305


namespace problem_statement_l181_181941

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := 2
def ellipse_eq (x y : ℝ) := (x^2) / 8 + (y^2) / 4 = 1
def line_eq (x y m : ℝ) := y = x + m
def circle_eq (x y : ℝ) := x^2 + y^2 = 1

theorem problem_statement (x1 y1 x2 y2 x0 y0 m : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
  (hm : line_eq x0 y0 m) (h0 : (x1 + x2) / 2 = -2 * m / 3) (h0' : (y1 + y2) / 2 = m / 3) : 
  (ellipse_eq x y ∧ line_eq x y m ∧ circle_eq x0 y0) → m = (3 * Real.sqrt 5) / 5 ∨ m = -(3 * Real.sqrt 5) / 5 := 
by {
  sorry
}

end problem_statement_l181_181941


namespace ray_total_grocery_bill_l181_181311

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l181_181311


namespace mul_pos_neg_eq_neg_l181_181494

theorem mul_pos_neg_eq_neg (a : Int) : 3 * (-2) = -6 := by
  sorry

end mul_pos_neg_eq_neg_l181_181494


namespace partition_sum_nine_times_l181_181863

theorem partition_sum_nine_times (k : ℕ) :
  ∃ (A B : Set ℕ), 
    (S = {1994 + 3 * i | i ∈ Finset.range (k + 1)}) ∧
    (A ∪ B = S) ∧
    (A ∩ B = ∅) ∧
    (∑ x in A, x = 9 * ∑ x in B, x) →
  (∃ t : ℕ, k = 20 * t - 1 ∨ k = 20 * t + 4) :=
begin
  sorry
end

end partition_sum_nine_times_l181_181863


namespace initial_men_count_l181_181714

noncomputable def provisions_last_initially (M : ℝ) (P : ℝ) : Prop :=
  P = M * 17

noncomputable def provisions_last_with_320_more (M : ℝ) (P : ℝ) : Prop :=
  P = (M + 320) * 14.010989010989011

noncomputable def men_initially (M : ℝ) : Prop :=
  ∃ P : ℝ, provisions_last_initially M P ∧ provisions_last_with_320_more M P

theorem initial_men_count : men_initially 1500 :=
begin
  sorry
end

end initial_men_count_l181_181714


namespace domain_of_rational_function_l181_181003

noncomputable def domain_of_function (y : ℝ → ℝ) : (ℝ → Prop) := 
  λ x, x ≠ 8

theorem domain_of_rational_function :
  ∀ x : ℝ, x ≠ 8 ↔ (x ∈ set.Ioo (-∞ : ℝ) 8 ∪ set.Ioo 8 ∞) :=
by
  intros
  simp [set.Ioo, set.union]
  sorry

end domain_of_rational_function_l181_181003


namespace person_C_balls_l181_181476

theorem person_C_balls (balls : Finset ℕ) (label : ℕ → Finset ℕ) :
  (∀ x ∈ balls, x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) ∧
  (∀ s ∈ (Finset.image label (Finset.range 12)), s.card = 4) ∧
  (∀ s ∈ (Finset.image label (Finset.range 12)), s.sum id = 26) ∧
  label 0 = {6, 11} ∧
  label 1 = {4, 8} ∧
  ∃ t, t ∪ {1} = label 2 ∧ t = {3, 10, 12} :=
begin
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
  let label : ℕ → Finset ℕ := λ x, match x with
    | 0 => {6, 11}
    | 1 => {4, 8}
    | 2 => {1, 3, 10, 12}
    | _ => ∅
  end,
  split,
  { intros x hx,
    by_contradiction h,
    cases x,
    any_goals {linarith},
    repeat {linarith} },
  split,
  { intros s hs,
    rw Finset.mem_image at hs,
    cases hs with x hx,
    cases hx with h1 h2,
    have : x < 3,
    { by_contradiction h,
      simp only [not_lt, Finset.mem_range] at h,
      cases x,
      any_goals {linarith},
      repeat {linarith} },
    have : s = (label x),
    { rw ← h2 },
    simp only [Finset.card_union, Finset.card_singleton, eq_self_iff_true, Finset.card_eq_one, Finset.mem_image, exists_eq_right] },
  split,
  { intros s hs,
    rw Finset.mem_image at hs,
    cases hs with x hx,
    cases hx with h1 h2,
    have : x < 3,
    { by_contradiction h,
      simp only [not_lt, Finset.mem_range] at h,
      cases x,
      any_goals {linarith},
      repeat {linarith} },
    have : s = (label x),
    { rw ← h2 },
    by_cases x = 0,
    { subst h,
      simp only [label, Finset.sum_insert, Finset.sum_singleton, eq_self_iff_true],
      norm_num,
      split,
      { intros,
        use {6, 11} } },
    by_cases x = 1,
    { subst h,
      simp only [label, Finset.sum_insert, Finset.sum_singleton, eq_self_iff_true],
      norm_num,
      split,
      { intros,
        use {4, 8} } },
    simp only [nat.lt_iff, not_le, Finset.mem_range, Finset.sum_insert, Finset.sum_singleton, eq_self_iff_true],
    norm_num },
  split,
  { refl },
  split,
  { refl },
  { use {3, 10, 12},
    simp [label] },
end

end person_C_balls_l181_181476


namespace total_fish_caught_l181_181836

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l181_181836


namespace identical_sets_l181_181301

def divides_with_remainder (a b : ℕ) : ℕ :=
  a % b

def remainders (set1 set2 : Finset ℕ) : Finset ℕ :=
  set1.bind (λ x, set2.image (divides_with_remainder x))

theorem identical_sets (A B : Finset ℕ) (hA : A.card = 100) (hB : B.card = 100)
  (h_diff_A : ∀ (x y : ℕ), x ∈ A → y ∈ A → x ≠ y)
  (h_diff_B : ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y)
  (h_identical_remainders : remainders A B = remainders B A) :
  A = B :=
by 
  sorry

end identical_sets_l181_181301


namespace add_base8_numbers_l181_181475

def fromBase8 (n : Nat) : Nat :=
  Nat.digits 8 n |> Nat.ofDigits 8

theorem add_base8_numbers : 
  fromBase8 356 + fromBase8 672 + fromBase8 145 = fromBase8 1477 :=
by
  sorry

end add_base8_numbers_l181_181475


namespace two_abs_inequality_l181_181534

theorem two_abs_inequality (x y : ℝ) :
  2 * abs (x + y) ≤ abs x + abs y ↔ 
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -x / 3) ∨ 
  (x < 0 ∧ -x / 3 ≤ y ∧ y ≤ -3 * x) :=
by
  sorry

end two_abs_inequality_l181_181534


namespace sum_of_three_numbers_l181_181008

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 36) 
  (h2 : b + c = 55) 
  (h3 : c + a = 60) : 
  a + b + c = 75.5 := 
by 
  sorry

end sum_of_three_numbers_l181_181008


namespace product_of_base8_digits_l181_181391

theorem product_of_base8_digits (n : ℕ) (h : n = 7890) : 
  let base8_repr := [1, 7, 3, 2, 2] in 
  base8_repr.product = 84 :=
by 
  -- Proof omitted
  sorry

end product_of_base8_digits_l181_181391


namespace max_value_is_sqrt_n_by_2_l181_181185

noncomputable def max_value_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  if n = 0 then 0 else
  (Finset.univ.sum (λ i => Real.sin (x i))) /
  Real.sqrt ((Finset.univ.sum (λ i => Real.tan (x i))^2 + n))

theorem max_value_is_sqrt_n_by_2 (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, x i ∈ Ioo 0 (Real.pi / 2)) :
  max_value_expression n x ≤ Real.sqrt n / 2 :=
sorry

end max_value_is_sqrt_n_by_2_l181_181185


namespace rita_canoe_distance_l181_181314

theorem rita_canoe_distance 
  (up_speed : ℕ) (down_speed : ℕ)
  (wind_up_decrease : ℕ) (wind_down_increase : ℕ)
  (total_time : ℕ) 
  (effective_up_speed : ℕ := up_speed - wind_up_decrease)
  (effective_down_speed : ℕ := down_speed + wind_down_increase)
  (T_up : ℚ := D / effective_up_speed)
  (T_down : ℚ := D / effective_down_speed) :
  (T_up + T_down = total_time) ->
  (D = 7) := 
by
  sorry

-- Parameters as defined in the problem
def up_speed : ℕ := 3
def down_speed : ℕ := 9
def wind_up_decrease : ℕ := 2
def wind_down_increase : ℕ := 4
def total_time : ℕ := 8

end rita_canoe_distance_l181_181314


namespace card_distribution_l181_181942

theorem card_distribution (n : ℕ) (h : n ≥ 3) : 
  ∀ (initial_distribution : Fin n → Finset (Fin (n^2))),
  (∀ i, (initial_distribution i).card = n) → 
  ∃ (operations : list ((Fin n) × (Fin n) × (Fin 4))), 
  ∀ (final_distribution : Fin n → Finset (Fin (n^2))),
  (process_operations initial_distribution operations final_distribution) → 
  (∀ i, ∃ k : ℕ, (∀ x ∈ final_distribution i, x.val ∈ (Finset.range n).map (fun a => k + a.val))) :=
begin
  sorry
end

-- Helper function to define the process of operations (not part of the problem, provided for understanding)
def process_operations 
  (initial_distribution : Fin n → Finset (Fin (n^2)))
  (operations : list ((Fin n) × (Fin n) × (Fin 4)))
  (final_distribution : Fin n → Finset (Fin (n^2))) : Prop :=
  -- Process function implementation would go here
  sorry

end card_distribution_l181_181942


namespace Donny_change_l181_181511

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l181_181511


namespace carlson_wins_with_optimal_play_l181_181489

noncomputable def chocolate_bar_game : Prop :=
  ∀ (m n : ℕ), (m = 15) → (n = 100) →
  (∀ (turn : ℕ), turn % 2 = 0 → (start_with_Carlson : Prop) →
  ((Carlson_wins : Prop) ↔ optimal_play))

theorem carlson_wins_with_optimal_play : chocolate_bar_game :=
by sorry

end carlson_wins_with_optimal_play_l181_181489


namespace brady_june_hours_l181_181075

variable (x : ℕ) -- Number of hours worked every day in June

def hoursApril : ℕ := 6 * 30 -- Total hours in April
def hoursSeptember : ℕ := 8 * 30 -- Total hours in September
def hoursJune (x : ℕ) : ℕ := x * 30 -- Total hours in June
def totalHours (x : ℕ) : ℕ := hoursApril + hoursJune x + hoursSeptember -- Total hours over three months
def averageHours (x : ℕ) : ℕ := totalHours x / 3 -- Average hours per month

theorem brady_june_hours (h : averageHours x = 190) : x = 5 :=
by
  sorry

end brady_june_hours_l181_181075


namespace remaining_cooking_time_eq_l181_181832

theorem remaining_cooking_time_eq {
  (recommended_fries: minutes) = 12 
  (recommended_nuggets: minutes) = 18 
  (recommended_sticks: minutes) = 8 
  (cooked_fries: minutes) = 2 
  (cooked_nuggets: minutes) = 5 
  (cooked_sticks: minutes) = 3 
} :
  remaining_seconds_french_fries = 600 ∧ remaining_seconds_chicken_nuggets = 780 ∧ remaining_seconds_mozzarella_sticks = 300 :=
by
  -- Definitions based on conditions
  let remaining_minutes_fries := recommended_fries - cooked_fries
  let remaining_seconds_fries := remaining_minutes_fries * 60
  let remaining_minutes_nuggets := recommended_nuggets - cooked_nuggets
  let remaining_seconds_nuggets := remaining_minutes_nuggets * 60
  let remaining_minutes_sticks := recommended_sticks - cooked_sticks
  let remaining_seconds_sticks := remaining_minutes_sticks * 60
  -- Assertions
  have h_fries : remaining_seconds_fries = 600, by sorry
  have h_nuggets : remaining_seconds_nuggets = 780, by sorry
  have h_sticks : remaining_seconds_sticks = 300, by sorry
  -- Conclusion
  exact ⟨h_fries, h_nuggets, h_sticks⟩

end remaining_cooking_time_eq_l181_181832


namespace broadcasting_methods_correct_l181_181422

-- Let's define the problem constants
def num_commercial_ads : ℕ := 3
def num_olympic_ads : ℕ := 2
def num_public_service_ads : ℕ := 1
def total_ads : ℕ := num_commercial_ads + num_olympic_ads + num_public_service_ads

-- Define the conditions
def last_ad_not_commercial : Prop := true -- For simplicity, assume as true
def no_consecutive_olympic_or_public_service_ad : Prop := true -- For simplicity, assume as true

-- Define the expected result
def expected_broadcasting_methods : ℕ := 108

-- The main theorem statement
theorem broadcasting_methods_correct :
  last_ad_not_commercial ∧ no_consecutive_olympic_or_public_service_ad →
  let arrangements_commercial := 1 -- equivalent to A_3^3 but simplified, as it is a permutation of 3 elements
  let choose_last_ad := 3 -- equivalent to A_3^1: choose 1 of 3 (2 olympic + 1 public service)
  let arrange_remaining_ads := 6 -- equivalent to A_3^2 but simplified, which is arranging 2 elements in 3 places not adjacent
  arrangements_commercial * choose_last_ad * arrange_remaining_ads = expected_broadcasting_methods := 
begin
  intros _,
  exact eq.refl expected_broadcasting_methods,
end

end broadcasting_methods_correct_l181_181422


namespace intersection_is_empty_l181_181209

-- Definitions based on the conditions provided in the problem
def M : Set (ℝ × ℝ) := { p | ∃ m b : ℝ, p = (m, b) } -- Set of all straight lines: y = mx + b
def N : Set (ℝ × ℝ) := { p | ∃ a b c : ℝ, a ≠ 0 ∧ p = (a, b, c) } -- Set of all parabolas: y = ax^2 + bx + c

theorem intersection_is_empty : M ∩ N = ∅ :=
by
  sorry

end intersection_is_empty_l181_181209


namespace sum_of_excluded_numbers_l181_181818

theorem sum_of_excluded_numbers (S : ℕ) (X : ℕ) (n m : ℕ) (averageN : ℕ) (averageM : ℕ)
  (h1 : S = 34 * 8) 
  (h2 : n = 8) 
  (h3 : m = 6) 
  (h4 : averageN = 34) 
  (h5 : averageM = 29) 
  (hS : S = n * averageN) 
  (hX : S - X = m * averageM) : 
  X = 98 := by
  sorry

end sum_of_excluded_numbers_l181_181818


namespace problem_l181_181573

-- Given Definitions
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2  -- Sum of first n terms in smooth progression
def T (n : ℕ) : ℕ := (n + 5) * n        -- Given sum of first n terms for sequence b

-- Given Conditions
def a (n : ℕ) : ℕ := n                  -- Arithmetic sequence a_n

-- Specifying the required proof
theorem problem (n : ℕ) : 
  a n = n ∧ 
  (∑ k in Finset.range n, 1 / (a k * (2 * k + 4))) = 
  (3 / 8) - (1 / (4 * (n + 1))) - (1 / (4 * (n + 2))) := 
by
  sorry

end problem_l181_181573


namespace harmful_bacteria_time_l181_181753

noncomputable def number_of_bacteria (x : ℝ) : ℝ :=
  4000 * 2^x

theorem harmful_bacteria_time :
  ∃ (x : ℝ), number_of_bacteria x > 90000 ∧ x = 4.5 :=
by
  sorry

end harmful_bacteria_time_l181_181753


namespace sum_k_binomial_l181_181907

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181907


namespace prime_triples_l181_181103

theorem prime_triples (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) :
  (p^4 - 1) % (q * r) = 0 ∧ (q^4 - 1) % (p * r) = 0 ∧ (r^4 - 1) % (p * q) = 0 → 
  {p, q, r} = {2, 3, 5} :=
by
  sorry

end prime_triples_l181_181103


namespace intersection_complement_l181_181970

-- Definitions based on the conditions in the problem
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

-- Definition of complement of set M in the universe U
def complement_U (M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

-- The proof statement
theorem intersection_complement :
  N ∩ (complement_U M) = {3, 5} :=
by
  sorry

end intersection_complement_l181_181970


namespace equation_of_tangent_line_l181_181051

-- Define the point P and the circle O
def P : ℝ × ℝ := (-1, Real.sqrt 3)
def circle_O (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 4

-- Definition of a tangent line to a circle at a given point
def is_tangent (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  ∃ m b, P = (-1, Real.sqrt 3) ∧ ∀ (x y : ℝ), l x y ↔ y = m * x + b ∧ (circle_O x y → ¬ (l x y))

-- Theorem stating the equation of the tangent line is x - √3 * y + 4 = 0
theorem equation_of_tangent_line : 
  is_tangent (λ x y, x - Real.sqrt 3 * y + 4 = 0) P :=
sorry

end equation_of_tangent_line_l181_181051


namespace average_of_remaining_numbers_l181_181335

theorem average_of_remaining_numbers 
  (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50) = 20)
  (h_disc : 45 ∈ numbers ∧ 55 ∈ numbers) 
  (h_count_45_55 : numbers.count 45 = 1 ∧ numbers.count 55 = 1) :
  (numbers.sum - 45 - 55) / (50 - 2) = 18.75 :=
by
  sorry

end average_of_remaining_numbers_l181_181335


namespace total_students_l181_181021

theorem total_students (boys girls : ℕ) (ratio : boys / girls = 5 / 7) (girls = 140) : boys + girls = 240 := 
by sorry

end total_students_l181_181021


namespace sum_of_integers_k_sum_of_all_integers_k_l181_181895
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l181_181895


namespace cos_double_angle_l181_181143

noncomputable def cos_2α (α : ℝ) : ℝ := 2 * (cos α) ^ 2 - 1

theorem cos_double_angle (α : ℝ) (h_acute : 0 < α ∧ α < π / 2) (h : Real.cos (α + π / 4) = 3 / 5) :
  cos_2α α = 24 / 25 :=
sorry

end cos_double_angle_l181_181143


namespace biased_sample_non_representative_l181_181439

/-- 
A proof problem verifying the representativeness of a sample of 2000 email address owners concerning 
the urban population's primary sources of news.
-/
theorem biased_sample_non_representative (
  (U : Type) 
  (email_population : finset U) 
  (sample : finset U) :
  sample.card = 2000 
  ∧ sample ⊆ email_population 
  ∧ ∃ (u : U), u ∈ sample 
  → email_population.card < finset.card email_population
  := sorry

end biased_sample_non_representative_l181_181439


namespace part1_range_of_a_part2_area_l181_181781

theorem part1_range_of_a (a : ℝ) : 
  (∃ t : ℝ, a ≠ 0 ∧ -1/6 < a ∧ (a > 0 ∨ (a > -1/6 ∧ a < 0)) ∧ 
    (∃ t : ℝ, (2 * a - (8 / 9) * t ^ 2) ^ 2 = 4 * (a ^ 2 + (16 / 27) * t ^ 3) ∧ 
    t = (3 + Real.sqrt (9 + 54 * a)) / 2)) → -1/6 < a < 0 ∨ a > 0 :=
sorry

theorem part2_area (a : ℝ) (h : -1/6 < a < 0 ∨ a > 0) : 
  ∀ x1 x2 : ℝ, x2 = -a + (4 / 9) * ((3 + Real.sqrt (9 + 54 * a)) / 2) ^ 2 - x1 → 
  x1 = -a + (4 / 9) * ((3 - Real.sqrt (9 + 54 * a)) / 2) ^ 2 → 
  ∫ x in x1..x2, (x - x1)^2 + ∫ x in x1..x2, (x - x2)^2 = (16 / 3) * (2 * a + 1) ^ (3/2) :=
sorry

end part1_range_of_a_part2_area_l181_181781


namespace num_divisors_multiple_of_3_l181_181849

theorem num_divisors_multiple_of_3 (n : ℕ) (h : n = 11!) : 
  (∃ d : ℕ, d | n ∧ 3 ∣ d) → finset.card {d : ℕ | d ∣ n ∧ 3 ∣ d} = 432 :=
by {
  sorry
}

end num_divisors_multiple_of_3_l181_181849


namespace even_function_iff_a_eq_1_l181_181162

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- Lean 4 statement for the given math proof problem
theorem even_function_iff_a_eq_1 :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 1 :=
sorry

end even_function_iff_a_eq_1_l181_181162


namespace radius_of_roots_on_circle_l181_181819

theorem radius_of_roots_on_circle : 
  ∀ z : ℂ, (z + 2)^6 = 64 * z^6 → abs (z + 2) = 2 * abs z → 
  (∃ r : ℝ, r = 2 / real.sqrt 3) :=
begin
  intros z h1 h2,
  use 2 / real.sqrt 3,
  sorry
end

end radius_of_roots_on_circle_l181_181819


namespace solve_trig_equation_l181_181696

theorem solve_trig_equation (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) → 
  (cos y ≠ 0) → 
  ((sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z) →
  (∃ (n k m : ℤ), x = (π / 2) + π * n ∧ y = π * k ∧ z = 2 * π * m) := 
by 
  sorry

end solve_trig_equation_l181_181696


namespace pictures_vertically_l181_181281

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end pictures_vertically_l181_181281


namespace find_value_l181_181540

def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

variable (a b : ℝ)

axiom h1 : 3 * a + 5 * b = 1
axiom h2 : 4 * a + 9 * b = -1

theorem find_value : star a b 1 2 = 2010 := 
by 
  sorry

end find_value_l181_181540


namespace number_of_pairs_l181_181324

theorem number_of_pairs (f m : ℕ) (n : ℕ) :
  n = 6 →
  (f + m ≤ n) →
  ∃! pairs : ℕ, pairs = 2 :=
by
  intro h1 h2
  sorry

end number_of_pairs_l181_181324


namespace inclination_angle_is_pi_div_3_l181_181731

-- Define the parametric equations of the line
def line (s : ℝ) : ℝ × ℝ := (s + 1, sqrt 3 * s)

-- Define the inclination angle θ of the line
def inclination_angle (θ : ℝ) : Prop :=
  ∃ k : ℝ, (∀ s : ℝ, line s = (s + 1, sqrt 3 * s)) ∧ tan θ = sqrt 3 ∧ θ ∈ [0, Real.pi)

-- Prove that the inclination angle θ of the line is π/3.
theorem inclination_angle_is_pi_div_3 : inclination_angle (π / 3) :=
by
  sorry

end inclination_angle_is_pi_div_3_l181_181731


namespace incorrect_synthetic_analytic_method_statement_l181_181010

-- Define the synthetic method and analytic method
def synthetic_method_basic : Prop := "The synthetic method and the analytic method are the two most basic methods of direct proof."
def synthetic_method_forward_reasoning : Prop := "The synthetic method is also called the forward reasoning method or the cause-to-effect method."
def analytic_method_backward_reasoning : Prop := "The analytic method is also called the backward reasoning method or the effect-to-cause method."
def both_methods_cause_effect_reasoning : Prop := "Both the synthetic method and the analytic method involve reasoning from both cause and effect."

theorem incorrect_synthetic_analytic_method_statement :
  synthetic_method_basic ∧ synthetic_method_forward_reasoning ∧ analytic_method_backward_reasoning → ¬ both_methods_cause_effect_reasoning :=
by
  sorry

end incorrect_synthetic_analytic_method_statement_l181_181010


namespace total_games_l181_181045

theorem total_games (n : ℕ) (games_per_pair : ℕ) (teams_play : ℕ) 
  (h1 : n = 12) (h2 : games_per_pair = 4) 
  (h3 : teams_play = (n - 1) / 2): 
  (n * (n - 1) * games_per_pair) / 2 = 264 :=
by
  -- use the hypotheses for the proof
  rw [h1, h2]
  have h4 : n * (n - 1) / 2 = 66 := calc
    n * (n - 1) / 2 = 12 * 11 / 2 := by rw h1
                           ... = 132 / 2 := by norm_num
                           ... = 66 := by norm_num
  calc
    (n * (n - 1) * games_per_pair) / 2 
      = (12 * 11 * 4) / 2 := by rw [h1, h2]
    ... = (4 * 66) := by rw h4
    ... = 264 := by norm_num

end total_games_l181_181045


namespace interest_rate_b_to_c_l181_181430

open Real

noncomputable def calculate_rate_b_to_c (P : ℝ) (r1 : ℝ) (t : ℝ) (G : ℝ) : ℝ :=
  let I_a_b := P * (r1 / 100) * t
  let I_b_c := I_a_b + G
  (100 * I_b_c) / (P * t)

theorem interest_rate_b_to_c :
  calculate_rate_b_to_c 3200 12 5 400 = 14.5 := by
  sorry

end interest_rate_b_to_c_l181_181430


namespace minimum_value_quot_l181_181583

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem minimum_value_quot (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 :=
by
  sorry

end minimum_value_quot_l181_181583


namespace inequality_solution_l181_181871

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l181_181871


namespace compare_abc_l181_181642

noncomputable def a := Real.log 7 / Real.log 3 -- Represents log base 3 of 7
noncomputable def b := 2 ^ 1.1
noncomputable def c := 0.8 ^ 3.1

theorem compare_abc : c < a ∧ a < b :=
by
  -- condition definitions for proofs
  have log3_7_gt_1 : 1 < a := by sorry
  have log3_7_lt_2 : a < 2 := by sorry
  have b_gt_2 : 2 < b := by sorry
  have c_lt_1 : c < 1 := by sorry
  -- proof of c < a
  have c_lt_a : c < a := by sorry
  -- proof of a < b
  have a_lt_b : a < b := by sorry
  exact ⟨c_lt_a, a_lt_b⟩

end compare_abc_l181_181642


namespace ellipse_locus_l181_181130

def f₁ : ℝ × ℝ := (-4, 0)
def f₂ : ℝ × ℝ := (4, 0)

theorem ellipse_locus :
  ∀ (M : ℝ × ℝ), 
    (dist M f₁ + dist M f₂ = 10) ↔ 
    (M.1^2 / 25 + M.2^2 / 9 = 1) :=
sorry

end ellipse_locus_l181_181130


namespace sequence_sum_proof_l181_181959

variable {α : Type*} [OrderedRing α] [Inhabited α]

/-- Sequence sum definition, assumed to exist as a function -/
noncomputable def S : ℕ → α

/-- Sequence terms definition, assumed to exist as a function -/
noncomputable def a : ℕ → α

/-- The proof statement: If a₃ > 0, then S₁₁₃ > 0 -/
theorem sequence_sum_proof (h : a 3 > 0) : S 2013 > 0 :=
sorry

end sequence_sum_proof_l181_181959


namespace slope_angle_of_perpendicular_line_l181_181954

theorem slope_angle_of_perpendicular_line (l : ℝ → ℝ) (h_perp : ∀ x y : ℝ, l x = y ↔ x - y - 1 = 0) : ∃ α : ℝ, α = 135 :=
by
  sorry

end slope_angle_of_perpendicular_line_l181_181954


namespace urea_moles_produced_l181_181111

-- Define the reaction
def chemical_reaction (CO2 NH3 Urea Water : ℕ) :=
  CO2 = 1 ∧ NH3 = 2 ∧ Urea = 1 ∧ Water = 1

-- Given initial moles of reactants
def initial_moles (CO2 NH3 : ℕ) :=
  CO2 = 1 ∧ NH3 = 2

-- The main theorem to prove
theorem urea_moles_produced (CO2 NH3 Urea Water : ℕ) :
  initial_moles CO2 NH3 → chemical_reaction CO2 NH3 Urea Water → Urea = 1 :=
by
  intro H1 H2
  rcases H1 with ⟨HCO2, HNH3⟩
  rcases H2 with ⟨HCO2', HNH3', HUrea, _⟩
  sorry

end urea_moles_produced_l181_181111


namespace lines_concur_at_single_point_on_circumcircle_l181_181938

-- Define the triangle and associated points
variables {A B C M C_1 A_1 B_1 A_2 B_2 C_2 : Type}

-- Assume these points form a certain given geometric configuration
variables [CircularTriangle A B C] [Point M]
variables [IntersectThroughLine M AB BC CA C_1 A_1 B_1]
variables [IntersectCircumcircle AM BM CM A_2 B_2 C_2]

theorem lines_concur_at_single_point_on_circumcircle :
  ∃ P, (P ∈ Circumcircle A B C) ∧ Concur (LinesThrough [A_1 A_2, B_1 B_2, C_1 C_2]) :=
sorry

end lines_concur_at_single_point_on_circumcircle_l181_181938


namespace calc_a_squared_plus_b_squared_and_ab_l181_181982

theorem calc_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) :
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by
  sorry

end calc_a_squared_plus_b_squared_and_ab_l181_181982


namespace range_of_m_l181_181554

variable (A B C : ℝ)
variable (f : ℝ → ℝ)
variable (m : ℝ)

axiom angle_sum_eq_pi : A + B + C = π

def f (B : ℝ) : ℝ := 
  4 * sin B * (cos ((π / 4) - (B / 2)))^2 + cos (2 * B)

lemma B_range : 0 < B ∧ B < π := sorry

theorem range_of_m (h1: B_range B) (h2: ∀ (B : ℝ), 0 < B ∧ B < π → f B - m < 2) : m > 1 :=
sorry

end range_of_m_l181_181554


namespace fifth_dog_weight_l181_181024

theorem fifth_dog_weight (y : ℝ) (h : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y) / 5) : y = 31 :=
by
  sorry

end fifth_dog_weight_l181_181024


namespace find_n_l181_181414

theorem find_n (n : ℕ) (h : (1 + n) / (2 ^ n) = 3 / 16) : n = 5 :=
by sorry

end find_n_l181_181414


namespace evaluate_operations_l181_181401

theorem evaluate_operations : 
  (-2)^(2) = -4 ∧ (sqrt 9 ≠ 3) ∧ ((-2)^3 ≠ 8) ∧ (-|(-3)| ≠ 3) :=
by 
  sorry

end evaluate_operations_l181_181401


namespace bc_length_l181_181607

def triangle_area (a b C : ℝ) : ℝ :=
  (1 / 2) * a * b * (Real.sin C)

def law_of_cosines (a b c C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem bc_length (AB AC : ℝ) (angleA areaABC : ℝ) (h1 : AB = 2) (h2 : angleA = Real.pi / 3) (h3 : areaABC = sqrt 3 / 2) (h4 : triangle_area AB AC angleA = areaABC) :
  law_of_cosines AB AC (law_of_cosines AB AC angleA) angleA = sqrt 3 := by
  sorry

end bc_length_l181_181607


namespace find_f_zero_l181_181150

variable (f : ℝ → ℝ)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = -g (-x + 1)

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 1) = g (-x - 1)

theorem find_f_zero
  (H1 : odd_function f)
  (H2 : even_function f)
  (H3 : f 4 = 6) :
  f 0 = -6 := by
  sorry

end find_f_zero_l181_181150


namespace reflect_point_x_axis_l181_181622

def point_reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

theorem reflect_point_x_axis :
  ∀ P : ℝ × ℝ, P = (-1, 2) → point_reflect_x_axis P = (-1, -2) :=
by
  intro P h
  rw [h, point_reflect_x_axis]
  sorry

end reflect_point_x_axis_l181_181622


namespace term_expansion_l181_181956

theorem term_expansion (n : ℕ) (x : ℂ) :
  (4 * binom n 2 = -2 * binom n 1 + 162) → (n = 9) ∧
  (∃ c : ℂ, c * x^3 = binom 9 1 * (-2 : ℂ) * x^3 ∧ c = -18) :=
by {
  sorry -- Proof is not required, only the statement is provided as per the instructions.
}

end term_expansion_l181_181956


namespace min_value_f_l181_181548

def f (x y : ℝ) : ℝ := (x^2 + y^2 + 2) * (1 / (x + y) + 1 / (x * y + 1))

theorem min_value_f (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, ∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ m ∧ (f 1 1 = m) :=
by
  sorry

end min_value_f_l181_181548


namespace centered_hexagonal_seq_l181_181416

def is_centered_hexagonal (a : ℕ) : Prop :=
  ∃ n : ℕ, a = 3 * n^2 - 3 * n + 1

def are_sequences (a b c d : ℕ) : Prop :=
  (b = 2 * a - 1) ∧ (d = c^2) ∧ (a + b = c + d)

theorem centered_hexagonal_seq (a : ℕ) :
  (∃ b c d, are_sequences a b c d) ↔ is_centered_hexagonal a :=
sorry

end centered_hexagonal_seq_l181_181416


namespace proof_find_a_and_sqrt_difference_l181_181546

noncomputable def find_a_and_sqrt_difference (a b : ℝ) (h : sqrt (a - 5) + sqrt (5 - a) = b + 3) : ℝ × ℝ :=
if ha : a = 5 then
  let b_val := -3 in
  let sqrt_val := real.sqrt (a ^ 2 - b_val ^ 2) in
  (a, sqrt_val)
else
  (0, 0) -- unreachable

theorem proof_find_a_and_sqrt_difference :
  let a := 5
  let b := -3
  let s := find_a_and_sqrt_difference a b (by ring_goal -- sqrt (a - 5) + sqrt (5 - a) = b + 3 reduces to sqrt(0) + sqrt(0) = 0 + 3, which holds)
    in a = 5 ∧ (s = (a, 4) ∨ s = (a, -4)) :=
by
  sorry

end proof_find_a_and_sqrt_difference_l181_181546


namespace sector_area_proof_l181_181562

open Real

noncomputable def sector_angle_deg : ℝ := 60
noncomputable def sector_radius : ℝ := 3

def sector_angle_rad : ℝ := (sector_angle_deg * π) / 180
def sector_arc_length : ℝ := sector_angle_rad * sector_radius
def sector_area : ℝ := (1 / 2) * sector_arc_length * sector_radius

theorem sector_area_proof :
  sector_area = (3 * π) / 2 :=
by
  sorry

end sector_area_proof_l181_181562


namespace vector_parallel_l181_181584

theorem vector_parallel
  (a b : ℝ × ℝ)
  (c : ℝ → ℝ × ℝ)
  (d : ℝ × ℝ)
  (h1 : a = (1, 0))
  (h2 : b = (0, -1))
  (h3 : ∀ k : ℝ, k ≠ 0 → c k = (k^2, -k))
  (h4 : d = (1, -1))
  (h5 : ∀ k : ℝ, k ≠ 0 → ∃ λ : ℝ, c k = (λ * d.1, λ * d.2)) :
  ∃ k : ℝ, k ≠ 0 ∧ k = 1 ∧ c k = d :=
by
  sorry

end vector_parallel_l181_181584


namespace identify_wrong_operator_l181_181767

def original_expr (x y z w u v p q : Int) : Int := x + y - z + w - u + v - p + q
def wrong_expr (x y z w u v p q : Int) : Int := x + y - z - w - u + v - p + q

theorem identify_wrong_operator :
  original_expr 3 5 7 9 11 13 15 17 ≠ -4 →
  wrong_expr 3 5 7 9 11 13 15 17 = -4 :=
by
  sorry

end identify_wrong_operator_l181_181767


namespace number_of_valid_N_l181_181491

def is_valid_N (N : ℕ) : Prop :=
  1000 ≤ N ∧ N < 10000 ∧
  let N_5 := (N / 625) * 625 + ((N % 625) / 125) * 125 + ((N % 125) / 25) * 25 + ((N % 25) / 5) * 5 + (N % 5) in
  let N_6 := (N / 1296) * 1296 + ((N % 1296) / 216) * 216 + ((N % 216) / 36) * 36 + ((N % 36) / 6) * 6 + (N % 6) in
  let N_7 := (N / 2401) * 2401 + ((N % 2401) / 343) * 343 + ((N % 343) / 49) * 49 + ((N % 49) / 7) * 7 + (N % 7) in
  let S := N_5 + N_6 + N_7 in
  (S % 1000) = (2 * N % 1000)

theorem number_of_valid_N : #{N : ℕ | is_valid_N N} = 20 :=
sorry

end number_of_valid_N_l181_181491


namespace sqrt_meaningful_l181_181399

theorem sqrt_meaningful (x : ℝ) : x + 1 >= 0 ↔ (∃ y : ℝ, y * y = x + 1) := by
  sorry

end sqrt_meaningful_l181_181399


namespace relationship_among_a_b_c_l181_181927

noncomputable def a : ℝ := 0.5^3
noncomputable def b : ℝ := 3^0.5
noncomputable def c : ℝ := Real.log 3 / Real.log 0.5

theorem relationship_among_a_b_c (ha : a = 0.5^3) (hb : b = 3^0.5) (hc : c = Real.log 3 / Real.log 0.5) :
  c < a ∧ a < b :=
by
  rw [ha, hb, hc]
  sorry

end relationship_among_a_b_c_l181_181927


namespace problem_statement_l181_181567

noncomputable def f : ℝ → ℝ := sorry

lemma f_periodic_3 (x : ℝ) : f (x + 3) = f x := sorry
lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem problem_statement (α : ℝ) (h : Real.tan α = 3) : f (2015 * Real.sin (2 * α)) = 0 :=
by
  -- Placeholder definitions to establish context
  have sin_double_angle_identity : Real.sin (2 * α) = 2 * Real.sin α * Real.cos α :=
    Real.sin_two_mul α
  have tan_identity : Real.sin (2 * α) = (2 * 3) / (1 + 3^2) :=
    by rw [←Real.sin_two_mul, h, Real.tan_eq_sin_div_cos]; field_simp; ring
  -- Using the given conditions
  sorry

end problem_statement_l181_181567


namespace probability_of_urn_contains_nine_red_and_four_blue_after_operations_l181_181826

-- Definition of the initial urn state
def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1

-- Definition of the number of operations
def num_operations : ℕ := 5

-- Definition of the final state
def final_red_balls : ℕ := 9
def final_blue_balls : ℕ := 4

-- Definition of total number of balls after five operations
def total_balls_after_operations : ℕ := 13

-- The probability we aim to prove
def target_probability : ℚ := 1920 / 10395

noncomputable def george_experiment_probability_theorem 
  (initial_red_balls initial_blue_balls num_operations final_red_balls final_blue_balls : ℕ)
  (total_balls_after_operations : ℕ) : ℚ :=
if initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ num_operations = 5 ∧ final_red_balls = 9 ∧ final_blue_balls = 4 ∧ total_balls_after_operations = 13 then
  target_probability
else
  0

-- The theorem statement, no proof provided (using sorry).
theorem probability_of_urn_contains_nine_red_and_four_blue_after_operations :
  george_experiment_probability_theorem 2 1 5 9 4 13 = target_probability := sorry

end probability_of_urn_contains_nine_red_and_four_blue_after_operations_l181_181826


namespace union_of_subsets_card_geq_165_l181_181647

theorem union_of_subsets_card_geq_165 {A : Finset ℕ}
  (hA_card : A.card = 225)
  (A_subs : Fin n (Fin 11) → Finset ℕ)
  (hA_subs_card : ∀ i, (A_subs i).card = 45)
  (hA_subs_inter_card : ∀ i j, i < j → ((A_subs i) ∩ (A_subs j)).card = 9) :
  (Finset.univ.biUnion A_subs).card ≥ 165 :=
sorry

end union_of_subsets_card_geq_165_l181_181647


namespace period2_students_is_8_l181_181333

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l181_181333


namespace count_5_digit_numbers_with_6_divisible_by_3_l181_181882

theorem count_5_digit_numbers_with_6_divisible_by_3 :
  let count_5_digit := 99999 - 10000 + 1 in
  let multiples_of_3 := count_5_digit / 3 in
  let without_6 := 8 * 9^4 in
  let without_6_div_3 := 17496 in
  multiples_of_3 - without_6_div_3 = 12504 :=
by
  sorry

end count_5_digit_numbers_with_6_divisible_by_3_l181_181882


namespace probability_correct_l181_181221

def set_of_numbers : Finset ℕ := {3, 4, 6, 8, 9}

def is_multiple_of_12 (n : ℕ) : Prop := 12 ∣ n

def valid_pairs : Finset (ℕ × ℕ) :=
(set_of_numbers.product set_of_numbers).filter (λ p, p.1 ≠ p.2 ∧ is_multiple_of_12 (p.1 * p.2))

def total_pairs : ℕ := (set_of_numbers.card.choose 2)

def probability_of_multiple_12 : ℚ := (valid_pairs.card : ℚ) / (total_pairs : ℚ)

theorem probability_correct : probability_of_multiple_12 = 2 / 5 :=
by 
  sorry

end probability_correct_l181_181221


namespace find_solution_set_l181_181189

theorem find_solution_set (a b : ℝ) :
  ( ∀ x : ℝ, -1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0 ) →
  (∀ x : ℝ, ax^2 + bx + 2 = 0 ↔ x = -1 / 2 ∨ x = 1 / 3) →
  (2x^2 + bx + a < 0 ↔ -2 < x ∧ x < 3) :=
by
  sorry

end find_solution_set_l181_181189


namespace perpendicular_points_exists_l181_181550

-- Given data and conditions.
variables {l : Line} {d_c : ℝ}
  (A B C : Point)  -- Points on the line l
  (hAB : dist A B < d_c) (hBC : dist B C < d_c) -- Distances are less than the diameter of the circle template

-- Construct points D and D' such that line (D, D') is perpendicular to line l
theorem perpendicular_points_exists 
  (h_circle_construct : ∀ (X : Point), ∃ (C1 C2 : Circle),
    center C1 ∈ segment X A ∧ radius C1 = d_c / 2 ∧
    center C2 ∈ segment X B ∧ radius C2 = d_c / 2 ∧
    intersects C1 C2 D ∧ D ≠ A ∧ D ≠ B):
  ∃ D D' : Point, line_through D D' ⊥ l :=
sorry

end perpendicular_points_exists_l181_181550


namespace floor_system_unique_solution_l181_181036

noncomputable def floor_system_solution (x y : ℝ) : Prop :=
  (⌊x + y - 3⌋ = 2 - x) ∧ (⌊x + 1⌋ + ⌊y - 7⌋ + x = y)

theorem floor_system_unique_solution : ∃! (x y : ℝ), floor_system_solution x y :=
by
  use [3, -1]
  split
  {
    split
    {
      show (⌊3 + -1 - 3⌋ = 2 - 3), from sorry,
      show (⌊3 + 1⌋ + ⌊-1 - 7⌋ + 3 = -1), from sorry,
    }
    intro xy
    cases xy with x y
    show floor_system_solution x y → (x, y) = (3, -1)
      from sorry
  }
  sorry

end floor_system_unique_solution_l181_181036


namespace find_initial_lion_population_l181_181370

-- Define the conditions as integers
def lion_cubs_per_month : ℕ := 5
def lions_die_per_month : ℕ := 1
def total_lions_after_one_year : ℕ := 148

-- Define a formula for calculating the initial number of lions
def initial_number_of_lions (net_increase : ℕ) (final_count : ℕ) (months : ℕ) : ℕ :=
  final_count - (net_increase * months)

-- Main theorem statement
theorem find_initial_lion_population : initial_number_of_lions (lion_cubs_per_month - lions_die_per_month) total_lions_after_one_year 12 = 100 :=
  sorry

end find_initial_lion_population_l181_181370


namespace total_distance_proof_l181_181823

-- Define the conditions
def amoli_speed : ℕ := 42      -- Amoli's speed in miles per hour
def amoli_time : ℕ := 3        -- Amoli's driving time in hours
def anayet_speed : ℕ := 61     -- Anayet's speed in miles per hour
def anayet_time : ℕ := 2       -- Anayet's driving time in hours
def remaining_distance : ℕ := 121  -- Remaining distance to be traveled in miles

-- Total distance calculation
def total_distance : ℕ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

-- The theorem to prove
theorem total_distance_proof : total_distance = 369 :=
by
  -- Proof goes here
  sorry

end total_distance_proof_l181_181823


namespace trig_eqn_solution_l181_181704

noncomputable def solve_trig_eqn (x y z : ℝ) (m n : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (cos y ≠ 0) ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) ∧
  (x = π / 2 + π * m) ∧
  (y = π * n) ∧
  (z = 2 * π * m)

theorem trig_eqn_solution (x y z : ℝ) (m n : ℤ) :
  sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) →
  x = π / 2 + π * m ∧ y = π * n ∧ z = 2 * π * m :=
by
  sorry

end trig_eqn_solution_l181_181704


namespace max_irrational_nums_on_blackboard_l181_181684

theorem max_irrational_nums_on_blackboard (irrationals : Set ℝ) :
  (∀ a b ∈ irrationals, (∃ k ∈ ℚ, a = k * (b + 1)) ∨ (∃ k ∈ ℚ, b = k * (a + 1))) →
  irrationals.Finite → irrationals.card ≤ 3 :=
by
  intros h h_finite
  have key_lemma : ∀ a b : ℝ,
    a ∈ irrationals → b ∈ irrationals →
    (∃ k ∈ ℚ, a = k * (b + 1)) ∨ (∃ k ∈ ℚ, b = k * (a + 1)) := h
  have fin_irrs : irrationals.Finite := h_finite
  sorry

end max_irrational_nums_on_blackboard_l181_181684


namespace solve_trig_eq_l181_181690

   theorem solve_trig_eq (x y z : ℝ) (m n : ℤ): 
     sin x ≠ 0 → cos y ≠ 0 →
     (sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z →
     (∃ m n : ℤ, x = (π / 2) + π * m ∧ y = π * n ∧ z = 2 * π * m) := 
   by
     intros h1 h2 h_eq
     sorry
   
end solve_trig_eq_l181_181690


namespace find_b_l181_181177

def f (b : ℚ) (x : ℚ) : ℚ :=
  if x < 1 then 3 * x - b else 2 ^ x

theorem find_b (b : ℚ) (h : f b (f b (5 / 6)) = 4) : b = 11 / 8 := by sorry

end find_b_l181_181177


namespace f_even_l181_181346

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  sorry

end f_even_l181_181346


namespace floor_eq_solutions_count_l181_181976

theorem floor_eq_solutions_count :
  ∃ (card : ℕ), card = 110 ∧
    card = (Finset.card (Finset.filter (λ x : ℕ, 
      (floor (x / 10 : ℚ) = floor (x / 11 : ℚ) + 1)) 
      (Finset.range 220))) := by
  sorry

end floor_eq_solutions_count_l181_181976


namespace tetrahedron_count_in_cube_l181_181123

-- Define the 8 vertices of the cube
def cube_vertices : set (fin 3 → bool) := 
  {v | ∀ i, v i = tt ∨ v i = ff}

-- Define a tetrahedron as a set of 4 vertices
def is_tetrahedron (s : set (fin 3 → bool)) : Prop :=
  ∃ v1 v2 v3 v4 ∈ cube_vertices, s = {v1, v2, v3, v4} ∧ ¬ ∃(a b c : ℝ), ∀ v ∈ s, a * (v 0 : ℝ) + b * (v 1 : ℝ) + c * (v 2 : ℝ) = 1

-- The total ways of choosing any 4 points from 8 is C_8^4
def total_combinations : ℕ := nat.choose 8 4

-- The number of ways in which the chosen points are coplanar is 12
def coplanar_sets : ℕ := 12

-- The number of different tetrahedrons
def num_tetrahedrons : ℕ := total_combinations - coplanar_sets

-- The statement to prove
theorem tetrahedron_count_in_cube : num_tetrahedrons = nat.choose 8 4 - 12 := 
by 
  sorry

end tetrahedron_count_in_cube_l181_181123


namespace cheese_partition_l181_181453

-- Define the main objects: cube K and centers of the spherical holes A_i
variables
  (K : set ℝ × set ℝ × set ℝ)  -- cube
  (A : list (ℝ × ℝ × ℝ))       -- centers of the spherical holes

-- State that the holes are non-overlapping
def non_overlapping (A : list (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (i j : ℕ), i ≠ j → dist (A.nth_le i sorry) (A.nth_le j sorry) > 0

-- Define the partitioning property
def partitions_into_convex_polyhedra (K : set ℝ × set ℝ × set ℝ) (M : list (set (ℝ × ℝ × ℝ))) : Prop :=
  (∀ (X : ℝ × ℝ × ℝ), X ∈ K → ∃ (i : ℕ), X ∈ (M.nth_le i sorry)) ∧
  (∀ i, convex (M.nth_le i sorry))

-- Our theorem statement
theorem cheese_partition
  (h1 : ∀ (i : ℕ), i < A.length → (A.nth_le i sorry) ∈ K)
  (h2 : non_overlapping A) :
  ∃ (M : list (set (ℝ × ℝ × ℝ))),
    length M = length A ∧
    partitions_into_convex_polyhedra K M ∧
    (∀ (i : ℕ), i < A.length → (A.nth_le i sorry ∈ M.nth_le i sorry)) := sorry

end cheese_partition_l181_181453


namespace wilson_sledding_l181_181012

variable (T : ℕ)

theorem wilson_sledding :
  (4 * T) + 6 = 14 → T = 2 :=
by
  intros h
  sorry

end wilson_sledding_l181_181012


namespace students_play_both_l181_181665

-- Definitions of problem conditions
def total_students : ℕ := 1200
def play_football : ℕ := 875
def play_cricket : ℕ := 450
def play_neither : ℕ := 100
def play_either := total_students - play_neither

-- Lean statement to prove that the number of students playing both football and cricket
theorem students_play_both : play_football + play_cricket - 225 = play_either :=
by
  -- The proof is omitted
  sorry

end students_play_both_l181_181665


namespace find_a_for_even_function_l181_181158

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l181_181158


namespace part1_part2_l181_181145

-- Definitions for conditions
variable (a b c A B C : ℝ)
variable (h1 : a = Real.sqrt 2)

-- Given condition in the problem
variable (h2 : (1 / 2) * b * c * Real.sin A = (Real.sqrt 2 / 2) * (c * Real.sin C + b * Real.sin B - a * Real.sin A))

-- Part (1): Prove that A = π / 3
theorem part1 : A = π / 3 :=
sorry

-- Part (2): Given A = π / 3 and conditions, prove the maximum area of triangle ABC is sqrt(3)/2
theorem part2 (h3 : A = π / 3) : 
  let area := (1 / 2) * b * c * Real.sin A in 
  area ≤ Real.sqrt 3 / 2 :=
sorry

end part1_part2_l181_181145


namespace balance_weights_l181_181259

theorem balance_weights (n : ℕ) (h : n > 0) : 
  let weights := List.range n |>.map (λ k => 2^k) in
  ∃ (s : ℕ → ℕ), s n = (2 * n - 1)!! :=
by sorry

end balance_weights_l181_181259


namespace monotonicity_of_f_range_of_a_l181_181279

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Define the g(x) function for the inequality condition in part (2)
def g (x : ℝ) (a : ℝ) : ℝ := 2 * Real.exp x - (x - a) ^ 2

-- Prove the monotonicity of f(x) based on the values of a
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, f' x a > 0) ∧
  (a > 0 → ∀ x < Real.ln a, f' x a < 0 ∧ ∀ x > Real.ln a, f' x a > 0) :=
by sorry

-- Prove that for x ≥ 0, the range of values for a such that 2e^x ≥ (x - a)^2 is [ln 2 - 2, sqrt 2]
theorem range_of_a (x : ℝ) (h : x ≥ 0) :
  (2 * Real.exp x ≥ (x - a) ^ 2 ↔ a ∈ Set.Icc (Real.log 2 - 2) (Real.sqrt 2)) :=
by sorry

end monotonicity_of_f_range_of_a_l181_181279


namespace triangle_perimeter_l181_181755

theorem triangle_perimeter (PQ QR PR mPQ mQR mRP : ℝ)
  (hPQ : PQ = 150)
  (hQR : QR = 300)
  (hPR : PR = 200)
  (h_mPQ : mPQ = 75)
  (h_mQR : mQR = 60)
  (h_mRP : mRP = 20) :
  let perimeter := (2 * (mQR + mPQ - mRP)) + QR in
  perimeter = 880 :=
by
  sorry

end triangle_perimeter_l181_181755


namespace initial_bacteria_count_l181_181337

theorem initial_bacteria_count 
  (quadruple_every_20_seconds : ∀ n : ℕ, ∀ t : ℕ, t % 20 = 0 → quadrupling t n = n * 4 ^ (t / 20)) 
  (final_count : ℕ) 
  (H : final_count = 1_048_576) 
  (T : ℕ) 
  (H_T : T = 240) 
  : ∃ n : ℕ, n * 4 ^ (T / 20) = final_count :=
by
  sorry

end initial_bacteria_count_l181_181337


namespace students_between_100_and_110_l181_181236

variables (a : ℝ) (h_a : a > 0)

def X := Normal 100 (a^2)

theorem students_between_100_and_110 (n : ℕ) (h_n : n = 1000) (h_prob : (Normal.cdf X 90) = (1/10)) :
  number_of_students_between_100_and_110 = 400 :=
sorry

end students_between_100_and_110_l181_181236


namespace one_man_completes_work_in_100_days_l181_181417

-- Definitions based on conditions
def total_work_done (W : ℝ) (R_m R_w : ℝ) (days : ℝ) := (10 * R_m + 15 * R_w) * days = W
def woman_rate (R_w : ℝ) (W : ℝ) := R_w * 225 = W
def man_days (R_m : ℝ) (D_m : ℝ) (W : ℝ) := R_m * D_m = W

-- Theorem to be proved
theorem one_man_completes_work_in_100_days (W R_m R_w : ℝ) :
  (total_work_done W R_m R_w 6) →
  (woman_rate R_w W) →
  (man_days R_m 100 W) :=
begin
  -- sorry is a placeholder for the proof
  sorry
end

end one_man_completes_work_in_100_days_l181_181417


namespace solution_set_of_inequality_l181_181919

theorem solution_set_of_inequality (x : ℝ) (n : ℕ) (h1 : n ≤ x ∧ x < n + 1 ∧ 0 < n) :
  4 * (⌊x⌋ : ℝ)^2 - 36 * (⌊x⌋ : ℝ) + 45 < 0 ↔ ∃ k : ℕ, (2 ≤ k ∧ k < 8 ∧ ⌊x⌋ = k) :=
by sorry

end solution_set_of_inequality_l181_181919


namespace ann_subsets_common_element_l181_181747

-- Defining the problem in Lean
theorem ann_subsets_common_element {A : Type} (s : Finset A) (n : ℕ) (h1 : s.card = n) (h2 : 4 < n)
  (subsets : Finset (Finset A)) (h3 : subsets.card = n + 1)
  (h4 : ∀ t ∈ subsets, t.card = 3) :
  ∃ t1 t2 ∈ subsets, t1 ≠ t2 ∧ (t1 ∩ t2).card = 1 := 
by
  sorry

end ann_subsets_common_element_l181_181747


namespace min_cost_per_product_increasing_productive_capacity_l181_181794

-- Define the cost function P(x)
def P (x : ℕ) : ℝ :=
  50 + (7500 + 20 * x) / x + (x ^ 2 - 30 * x + 600) / x

-- Define the selling price Q(x)
def Q (x : ℕ) : ℝ :=
  1240 - (1 / 30) * x ^ 2

-- Define the total profit function f(x)
def f (x : ℕ) : ℝ :=
  x * Q x - x * P x

-- State that P(x) achieves its minimum value at 220 yuan
theorem min_cost_per_product :
  ∃ x : ℕ, P x = 220 :=
sorry

-- State the range of production volume for increasing profit
theorem increasing_productive_capacity :
  ∀ x : ℕ, (0 < x ∧ x < 100) → (f x) > 0 :=
sorry

end min_cost_per_product_increasing_productive_capacity_l181_181794


namespace distance_with_mother_l181_181632

theorem distance_with_mother (d_total d_father d_mother : ℝ) (h_total : d_total = 0.67) (h_father : d_father = 0.5) :
  d_mother = d_total - d_father → d_mother = 0.17 :=
by {
  intros,
  sorry
}

end distance_with_mother_l181_181632


namespace BC_length_l181_181248

-- Defining the setup of the problem
section TriangleProblem

variables {A B C M : Type} [Point A] [Point B] [Point C] [Midpoint M]
variables {AB AC BC AM : ℝ}

-- Conditions from the problem statement
axiom AB_eq_2 : AB = 2
axiom AC_eq_3 : AC = 3
axiom AM_eq_BC : AM = BC

-- The main theorem to prove the length BC
theorem BC_length : BC = sqrt (26 / 5) :=
by
  -- The proof logic is omitted, hence the sorry placeholder.
  sorry

end TriangleProblem

end BC_length_l181_181248


namespace last_two_digits_of_7_pow_10_l181_181760

theorem last_two_digits_of_7_pow_10 :
  (7 ^ 10) % 100 = 49 := by
  sorry

end last_two_digits_of_7_pow_10_l181_181760


namespace no_sol_x_y_pos_int_eq_2015_l181_181100

theorem no_sol_x_y_pos_int_eq_2015 (x y : ℕ) (hx : x > 0) (hy : y > 0) : ¬ (x^2 - y! = 2015) :=
sorry

end no_sol_x_y_pos_int_eq_2015_l181_181100


namespace find_a_l181_181216

open Real

theorem find_a (a : ℝ) (h : ∀ x ∈ set.Icc (1 : ℝ) a, a + log (2) x ≤ 6)
  (hmax : ∃ x ∈ set.Icc (1 : ℝ) a, a + log (2) x = 6) :
  a = 4 :=
sorry

end find_a_l181_181216


namespace carson_pumps_needed_l181_181498

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end carson_pumps_needed_l181_181498


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181914

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181914


namespace triangle_area_is_correct_l181_181231

-- Given conditions in the problem
variables {r : ℝ} (h1 : ∃ O : Point, Circle O r) (h2 : ∃ A B : Point, Chord (O r) A B ∧ length A B = r) 
          (O : Point) (A B M D : Point) (h3 : Perpendicular (O r) (Chord O A B) M) 
          (h4 : Perpendicular M (Line O A) D)

-- The theorem statement that needs to be proven
theorem triangle_area_is_correct : area_triangle M D A = (sqrt 3 * r^2) / 32 :=
by
  sorry

end triangle_area_is_correct_l181_181231


namespace minimum_expression_value_l181_181884

theorem minimum_expression_value (a b c : ℝ) (hbpos : b > 0) (hab : b > a) (hcb : b > c) (hca : c > a) :
  (a + 2 * b) ^ 2 / b ^ 2 + (b - 2 * c) ^ 2 / b ^ 2 + (c - 2 * a) ^ 2 / b ^ 2 ≥ 65 / 16 := 
sorry

end minimum_expression_value_l181_181884


namespace diagonal_length_of_regular_hexagon_l181_181883

-- Define a structure for the hexagon with a given side length
structure RegularHexagon (s : ℝ) :=
(side_length : ℝ := s)

-- Prove that the length of diagonal DB in a regular hexagon with side length 12 is 12√3
theorem diagonal_length_of_regular_hexagon (H : RegularHexagon 12) : 
  ∃ DB : ℝ, DB = 12 * Real.sqrt 3 :=
by
  sorry

end diagonal_length_of_regular_hexagon_l181_181883


namespace percentage_of_students_liking_chess_l181_181616

theorem percentage_of_students_liking_chess (total_students : ℕ) (basketball_percentage : ℝ) (soccer_percentage : ℝ) 
(identified_chess_or_basketball : ℕ) (students_liking_basketball : ℕ) : 
total_students = 250 ∧ basketball_percentage = 0.40 ∧ soccer_percentage = 0.28 ∧ identified_chess_or_basketball = 125 ∧ 
students_liking_basketball = 100 → ∃ C : ℝ, C = 0.10 :=
by
  sorry

end percentage_of_students_liking_chess_l181_181616


namespace derivative_of_y_l181_181723

-- Define the function y(x)
def y (x : ℝ) : ℝ := x * sin x + cos x

-- State the derivative of the function y
theorem derivative_of_y (x : ℝ) : deriv y x = x * cos x :=
by
sorry

end derivative_of_y_l181_181723


namespace solve_equation_l181_181701

theorem solve_equation (x y z : ℝ) (n k m : ℤ)
  (h1 : sin x ≠ 0)
  (h2 : cos y ≠ 0)
  (h_eq : (sin x ^ 2 + 1 / sin x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * cos z)
  : ∃ n k m : ℤ, x = π / 2 + π * n ∧ y = π * k ∧ z = 2 * π * m :=
by
  sorry

end solve_equation_l181_181701


namespace lowest_possible_score_l181_181077

def total_points_first_four_tests : ℕ := 82 + 90 + 78 + 85
def required_total_points_for_seven_tests : ℕ := 80 * 7
def points_needed_for_last_three_tests : ℕ :=
  required_total_points_for_seven_tests - total_points_first_four_tests

theorem lowest_possible_score 
  (max_points_per_test : ℕ)
  (points_first_four_tests : ℕ := total_points_first_four_tests)
  (required_points : ℕ := required_total_points_for_seven_tests)
  (total_points_needed_last_three : ℕ := points_needed_for_last_three_tests) :
  ∃ (lowest_score : ℕ), 
    max_points_per_test = 100 ∧
    points_first_four_tests = 335 ∧
    required_points = 560 ∧
    total_points_needed_last_three = 225 ∧
    lowest_score = 25 :=
by
  sorry

end lowest_possible_score_l181_181077


namespace shorter_leg_equals_segment_l181_181304

-- Define a Right Tangential Trapezoid with appropriate properties
structure RightTangentialTrapezoid (α : Type) :=
(a b c d : ℝ) -- sides
(h : ℝ) -- height
(diag_inter : ℝ × ℝ) -- intersection of diagonals
(line_parallel_base : (ℝ × ℝ) → (ℝ × ℝ)) -- line passing through diag_inter parallel to bases
(is_right : b^2 + c^2 = a^2 + d^2) -- condition for right trapezoid
(is_tangential : a + c = b + d) -- condition for tangential trapezoid

-- The segment length within the trapezoid parallel to bases
def segment_length (t : RightTangentialTrapezoid ℝ) : ℝ :=
  let (x, y) := t.diag_inter in
  let (x', y') := t.line_parallel_base (x, y) in
  ((x' - x)^2 + (y' - y)^2)^(1/2)

-- The Lean theorem statement
theorem shorter_leg_equals_segment (t : RightTangentialTrapezoid ℝ) :
  t.d = segment_length t :=
  by
    sorry

end shorter_leg_equals_segment_l181_181304


namespace cloth_selling_gain_l181_181053

variables (P C : ℝ) (h_cost_pos : C > 0) (h_gain_rate : 0.5) 

theorem cloth_selling_gain :
    ∃ M : ℝ, M * P = 30 * P - 30 * C ∧ (30 * P = 45 * C) ∧ M = 10 :=
by
  use 10
  split
  {
    rw [mul_sub, mul_comm 30 P, mul_comm 30 C],
    ring,
  }
  split
  {
    sorry,  -- Proving 30 * P = 45 * C is consistent with given conditions.
  }
  {
    sorry,  -- Proving M = 10 directly follows from the calculated steps.
  }

end cloth_selling_gain_l181_181053


namespace minimum_period_sine_l181_181352

def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem minimum_period_sine : (∃ T > 0, (∀ x, f (x + T) = f x) ∧ (T = (2 * Real.pi) / 3)) :=
by
  sorry

end minimum_period_sine_l181_181352


namespace turan_inequality_l181_181777

variable {V : Type*} [Fintype V] -- V is a finite type, denoting the vertices
variable {G : SimpleGraph V} -- G is a simple graph with vertices V

/-- The number of triangles in a graph G -/
def triangle_count (G : SimpleGraph V) : ℕ :=
  G.triangle_count

/-- Degree of a vertex u in graph G -/
def degree (G : SimpleGraph V) (u : V) : ℕ :=
  G.degree u

/-- Turan's Theorem (Theorem 3.2.4) -/
theorem turan_inequality (G : SimpleGraph V) (n : ℕ) [Fintype G.edge_set] [DecidableRel G.adj] :
  let T := triangle_count G in
  T ≥ 1/3 * (∑ e in G.edge_set, (degree G e.1 + degree G e.1 - n)) :=
by
  sorry

end turan_inequality_l181_181777


namespace total_fish_caught_l181_181835

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l181_181835


namespace incorrect_statements_count_l181_181824

theorem incorrect_statements_count :
  let s1 := ∀ (q : ℚ), ∃ p : ℝ, p = q
  let s2 := ∀ (x y : ℝ), |x| = |y| → x = y
  let s3 := ∀ (q : ℚ), |q| ≥ 0
  let s4 := ∀ (q : ℚ), ∃ (q' : ℚ), q' = -q
  ¬s2 ∧ s1 ∧ s3 ∧ s4 → 1 :=
by
  intros s1 s2 s3 s4 h
  have h1 : ¬s2 := by assumption
  have h2 : s1 := by assumption
  have h3 : s3 := by assumption
  have h4 : s4 := by assumption
  exact 1

end incorrect_statements_count_l181_181824


namespace cost_per_handle_l181_181297

/-- 
Northwest Molded molds plastic handles. The fixed cost to run the molding machine is $7640 
per week. The company sells the handles for 4.60 dollars each. They must mold and sell 
1910 handles weekly to break even. Prove that the cost per handle to mold is $0.60.
-/
theorem cost_per_handle (fixed_cost : ℝ) (price_per_handle : ℝ) (num_handles : ℝ) 
    (break_even_handles : ℝ) : fixed_cost = 7640 → price_per_handle = 4.60 
    → break_even_handles = 1910 → 
    (price_per_handle * break_even_handles = fixed_cost + num_handles * (fixed_cost / break_even_handles)) → num_handles = 0.60 := 
by {
  intros h_fixed_cost h_price_per_handle h_break_even_handles h_break_even_equation,
  rw [h_fixed_cost, h_price_per_handle, h_break_even_handles] at h_break_even_equation,
  sorry
}

end cost_per_handle_l181_181297


namespace ratio_of_perimeters_l181_181808

theorem ratio_of_perimeters : 
  let side := 8
  let folded_length := side / 2
  let cut_width := side / 2
  let small_rectangle_perimeter := 4 * cut_width
  let large_rectangle_perimeter := 2 * (side + folded_length)
  small_rectangle_perimeter / large_rectangle_perimeter = 2 / 3 := by
  let side := 8
  let folded_length := side / 2
  let cut_width := side / 2
  let small_rectangle_perimeter := 4 * cut_width
  let large_rectangle_perimeter := 2 * (side + folded_length)
  have h1 : small_rectangle_perimeter = 16 := by sorry
  have h2 : large_rectangle_perimeter = 24 := by sorry
  have h3 : (small_rectangle_perimeter : ℚ) / large_rectangle_perimeter = 2 / 3 := by sorry
  exact h3

end ratio_of_perimeters_l181_181808


namespace area_of_second_square_l181_181842

theorem area_of_second_square
  (DE EF : ℝ)
  (h_iso_right : DE = EF)
  (h_square1_area : ∃ s, DE * s / √2 * s / √2 = 784 ∧ s * s = 784) :
  ∃ t, t * t = 784 :=
by
  obtain ⟨s, h1, h2⟩ := h_square1_area
  use s -- using t = s to establish the second square's area
  rw h2
  exact h2

end area_of_second_square_l181_181842


namespace seventh_numbers_sum_l181_181662

def first_row_seq (n : ℕ) : ℕ := n^2 + n - 1

def second_row_seq (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_numbers_sum :
  first_row_seq 7 + second_row_seq 7 = 83 :=
by
  -- Skipping the proof
  sorry

end seventh_numbers_sum_l181_181662


namespace solve_trig_eq_l181_181692

   theorem solve_trig_eq (x y z : ℝ) (m n : ℤ): 
     sin x ≠ 0 → cos y ≠ 0 →
     (sin^2 x + (1 / sin^2 x))^3 + (cos^2 y + (1 / cos^2 y))^3 = 16 * cos z →
     (∃ m n : ℤ, x = (π / 2) + π * m ∧ y = π * n ∧ z = 2 * π * m) := 
   by
     intros h1 h2 h_eq
     sorry
   
end solve_trig_eq_l181_181692


namespace nominal_rate_of_interest_l181_181726

theorem nominal_rate_of_interest
  (EAR : ℝ)
  (n : ℕ)
  (h_EAR : EAR = 0.0609)
  (h_n : n = 2) :
  ∃ i : ℝ, (1 + i / n)^n - 1 = EAR ∧ i = 0.059 := 
by 
  sorry

end nominal_rate_of_interest_l181_181726


namespace main_theorem_l181_181129

noncomputable def proof_problem (N : ℕ) (a x β α : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N → a i > 11) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N → x i > 0) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N → a i > 0) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N → β i > 0) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N, (∑ i in finset.range N, a i * x i ^ β i = c)) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ N → (β 1 * a 1 * x 1 ^ β 1) / a 1 = (β i * a i * x i ^ β i) / α i)

theorem main_theorem (N : ℕ) (a x β α : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ N → a i > 11)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ N → x i > 0)
  (h3 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ N → a i > 0)
  (h4 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ N → β i > 0)
  (h5 : (∑ i in finset.range N, a i * x i ^ β i = c))
  (h6 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ N → (β 1 * a 1 * x 1 ^ β 1) / a 1 = (β i * a i * x i ^ β i) / α i) :
  proof_problem N a x β α c := 
by 
  sorry

end main_theorem_l181_181129


namespace geometric_sequence_general_term_geometric_sequence_sum_l181_181170

theorem geometric_sequence_general_term (a : ℕ → ℝ) 
  (h_geo: ∀ n, a (n+1) = q * a n) (h_pos: ∀ n, 0 < a n) 
  (h_cond1 : a 2 = 2) (h_cond2 : a 3 = 2 + 2 * a 1):
    ∃ q > 0, ∀ n, a n = 2^(n-1) :=
sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo: ∀ n, a (n+1) = q * a n) (h_pos: ∀ n, 0 < a n) 
  (h_cond1 : a 2 = 2) (h_cond2 : a 3 = 2 + 2 * a 1) :
    ∑ i in finset.range n, (2 * i - 1) / (a i) = 6 - (2 * n + 3) / (2 ^ (n - 1)) :=
sorry

end geometric_sequence_general_term_geometric_sequence_sum_l181_181170


namespace find_y_given_conditions_l181_181716

theorem find_y_given_conditions (a x y : ℝ) (h1 : y = a * x + (1 - a)) 
  (x_val : x = 3) (y_val : y = 7) (x_new : x = 8) :
  y = 22 := 
  sorry

end find_y_given_conditions_l181_181716


namespace polynomial_linear_l181_181027

theorem polynomial_linear (a : ℕ → ℝ) (h₀ : a 0 ≠ a 1)
    (h₁ : ∀ i : ℕ, 1 ≤ i → a (i - 1) + a (i + 1) = 2 * a i)
    (n : ℕ) :
    ∃ b c : ℝ, ∀ x : ℝ, (∑ i in finset.range (n + 1), 
                        a i * (nat.choose n i) * (1 - x) ^ (n - i) * x ^ i) = b + c * x := 
by
  sorry

end polynomial_linear_l181_181027


namespace add_to_make_divisible_by_5_l181_181019

theorem add_to_make_divisible_by_5 :
  ∃ n : ℕ, 821562 + n = 821565 ∧ 821565 % 5 = 0 :=
begin
  use 3,
  split,
  { norm_num },
  { norm_num }
end

end add_to_make_divisible_by_5_l181_181019


namespace number_div_0_04_eq_100_9_l181_181049

theorem number_div_0_04_eq_100_9 :
  ∃ number : ℝ, (number / 0.04 = 100.9) ∧ (number = 4.036) :=
sorry

end number_div_0_04_eq_100_9_l181_181049


namespace total_money_l181_181472

-- Define the variables A, B, and C as real numbers.
variables (A B C : ℝ)

-- Define the conditions as hypotheses.
def conditions : Prop :=
  A + C = 300 ∧ B + C = 150 ∧ C = 50

-- State the theorem to prove the total amount of money A, B, and C have.
theorem total_money (h : conditions A B C) : A + B + C = 400 :=
by {
  -- This proof is currently omitted.
  sorry
}

end total_money_l181_181472


namespace homework_problem_l181_181751

theorem homework_problem
  (total_students : ℕ)
  (students_math : ℕ)
  (students_korean : ℕ)
  (students_both : ℕ)
  (no_student_without_homework : total_students = students_math + students_korean - students_both) :
  students_both = 31 :=
by
  have h1 : total_students = 48 := rfl
  have h2 : students_math = 37 := rfl
  have h3 : students_korean = 42 := rfl
  have h4 := no_student_without_homework
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4.symm

end homework_problem_l181_181751


namespace tan_alpha_value_l181_181949

-- Define the conditions and the question to be proven in Lean

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -√10 / 5) (h2 : 0 < α ∧ α < π) : 
  tan α = -1 / 3 :=
sorry

end tan_alpha_value_l181_181949


namespace sin_cos_y_range_l181_181206

theorem sin_cos_y_range (x y : ℝ) (hx : 0 < x) (hπx : x < π / 2) (hy : 0 < y) (hπy : y < π / 2)
    (h : sin x = x * cos y) : x / 2 < y ∧ y < x :=
by
  sorry

end sin_cos_y_range_l181_181206


namespace power_function_not_pass_origin_l181_181603

noncomputable def does_not_pass_through_origin (m : ℝ) : Prop :=
  ∀ x:ℝ, (m^2 - 3 * m + 3) * x^(m^2 - m - 2) ≠ 0

theorem power_function_not_pass_origin (m : ℝ) :
  does_not_pass_through_origin m ↔ (m = 1 ∨ m = 2) :=
sorry

end power_function_not_pass_origin_l181_181603


namespace find_m_l181_181960

theorem find_m (x1 x2 m : ℝ) (h_eq : ∀ x, x^2 + x + m = 0 → (x = x1 ∨ x = x2))
  (h_abs : |x1| + |x2| = 3)
  (h_sum : x1 + x2 = -1)
  (h_prod : x1 * x2 = m) :
  m = -2 :=
sorry

end find_m_l181_181960


namespace find_a_for_even_l181_181154

def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

theorem find_a_for_even (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = f a x) ↔ a = 1 :=
by
  -- proof steps here
  sorry

end find_a_for_even_l181_181154


namespace log₂_2_minus_x_is_decreasing_l181_181478

def is_decreasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y ≤ f x 

theorem log₂_2_minus_x_is_decreasing :
  is_decreasing_on_ℝ (λ x, Real.logb 2 (2 ^ (-x))) :=
by
  sorry

end log₂_2_minus_x_is_decreasing_l181_181478


namespace unique_n_for_trigonometric_identity_l181_181112

theorem unique_n_for_trigonometric_identity :
  ∀ (n : ℕ), n > 0 →
  (sin (Real.pi / (2 * n)) + cos (Real.pi / (2 * n)) = Real.sqrt n / 2) →
  n = 6 :=
by
  intros n hn h
  sorry

end unique_n_for_trigonometric_identity_l181_181112


namespace find_squares_l181_181864

theorem find_squares (s1 s2 : ℕ) (a b : ℕ) (h1 : s1 = a^2) (h2 : s2 = b^2) (h3 : a > b > 0) :
  s1 - s2 = 1989 ↔ 
  (s1, s2) = (995^2, 994^2) ∨ (s1, s2) = (333^2, 330^2) ∨ 
  (s1, s2) = (115^2, 106^2) ∨ (s1, s2) = (83^2, 70^2) ∨ 
  (s1, s2) = (67^2, 50^2) ∨ (s1, s2) = (45^2, 6^2) := by 
  sorry

end find_squares_l181_181864


namespace tan_range_l181_181359

-- Define the function and the interval
def f (x : ℝ) : ℝ := Real.tan x
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 4

-- State the theorem
theorem tan_range : ∀ x, interval x → 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end tan_range_l181_181359


namespace find_g_expression_l181_181183

theorem find_g_expression (g : ℝ → ℝ) (h : ∀ x, g(x + 2) = 2 * x + 3) : ∀ x, g(x) = 2 * x - 1 :=
by
  sorry

end find_g_expression_l181_181183


namespace area_of_contained_region_l181_181104

def contained_area (x y : ℝ) : Prop :=
  abs (2 * x + 3 * y) + abs (2 * x - 3 * y) ≤ 12

theorem area_of_contained_region : 
  (realVolume (setOf (λ p : ℝ × ℝ, contained_area p.1 p.2)) = 24) :=
sorry

end area_of_contained_region_l181_181104


namespace total_distance_covered_l181_181758

theorem total_distance_covered (speed_train_A_kmph : ℕ) (speed_train_B_kmph : ℕ) (time_interval_min : ℕ) :
  (speed_train_A_kmph = 150) →
  (speed_train_B_kmph = 180) →
  (time_interval_min = 25) →
  let speed_train_A_kmpm := speed_train_A_kmph / 60.0 in
  let speed_train_B_kmpm := speed_train_B_kmph / 60.0 in
  let distance_train_A := speed_train_A_kmpm * time_interval_min in
  let distance_train_B := speed_train_B_kmpm * time_interval_min in
  distance_train_A + distance_train_B = 137.5 :=
by
  intros hA hB hT
  let speed_train_A_kmpm := 150 / 60.0
  let speed_train_B_kmpm := 180 / 60.0
  let distance_train_A := speed_train_A_kmpm * 25
  let distance_train_B := speed_train_B_kmpm * 25
  have h1 : distance_train_A = 62.5 := sorry
  have h2 : distance_train_B = 75 := sorry
  rw [h1, h2]
  exact rfl

end total_distance_covered_l181_181758


namespace find_x_l181_181213

theorem find_x (x : ℝ) : (0.75 / x = 10 / 8) → (x = 0.6) := by
  sorry

end find_x_l181_181213


namespace transformed_mean_variance_l181_181604

variable {n : ℕ} {x : Fin n → ℝ} {x_bar S : ℝ}

-- Definitions based on the conditions
def mean_of_data (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n
def variance_of_data (x : Fin n → ℝ) (x_bar : ℝ) : ℝ := (∑ i, (x i - x_bar)^2) / n

-- Mean and variance conditions
axiom mean_condition : mean_of_data x = x_bar
axiom variance_condition : variance_of_data x x_bar = S^2

-- Theorem to prove the mean and variance of transformed data
theorem transformed_mean_variance (h_mean : mean_of_data x = x_bar) (h_variance : variance_of_data x x_bar = S^2) :
  mean_of_data (λ i => 2 * x i - 1) = 2 * x_bar - 1 ∧ variance_of_data (λ i => 2 * x i - 1) (2 * x_bar - 1) = 4 * S^2 := 
by -- proof to be done
  sorry

end transformed_mean_variance_l181_181604


namespace mean_variance_transformation_l181_181169

-- Given conditions
variables {α : Type*} [field α] [decidable_eq α] {n : ℕ}
variables {x : fin n → α}
variable (μ : α)   -- mean of x₁, x₂, ..., xₙ
variable (σ² : α)  -- variance of x₁, x₂, ..., xₙ

-- The conditions
axiom mean_x : μ = 2
axiom variance_x : σ² = 3

-- The theorem to prove
theorem mean_variance_transformation :
  let new_x : fin n → α := λ i, 3 * x i + 5 in
  (mean new_x = 11) ∧ (variance new_x = 27) :=
by {
  sorry
}


end mean_variance_transformation_l181_181169


namespace time_to_fill_tank_l181_181303

-- Definitions based on the given conditions
def rateA : ℝ := 1 / 6  -- Pipe A fills the tank in 6 minutes
def rateB : ℝ := 2 * rateA  -- Pipe B fills the tank twice as fast as Pipe A
def rateC : ℝ := - (1 / 15)  -- Pipe C drains the tank in 15 minutes (negative rate for draining)

-- Combined rate
def combined_rate : ℝ := rateA + rateB + rateC

-- Proof statement
theorem time_to_fill_tank : (1 / combined_rate) = 30 / 13 :=
by
  sorry

end time_to_fill_tank_l181_181303


namespace find_percentage_l181_181418

theorem find_percentage (P : ℕ) : 0.15 * 40 = (P / 100) * 16 + 2 → P = 25 := 
by
  sorry

end find_percentage_l181_181418


namespace proof_triangle_problem_l181_181250

noncomputable def triangle_problem : Prop :=
  ∃ (A B C : ℝ) (a b c : ℝ), 
    A = 60 ∧
    a = Real.sqrt 6 ∧
    b = 2 ∧
    sin A ≠ 0 ∧
    (\forall A B C, sin C = sin (180 - A - B)) ∧
    (\forall a b A B, a = b* sin A / sin B) ∧
    B = 45 ∧
    a * b * sin (180 - 60 - 45) / 2 = (3 + Real.sqrt 3) / 2

theorem proof_triangle_problem : triangle_problem :=
by
  sorry

end proof_triangle_problem_l181_181250


namespace sum_of_valid_n_l181_181397

theorem sum_of_valid_n {n : ℕ} (h : ∃ (n : ℕ), Nat.lcm (2 * n) (n ^ 2) = 14 * n - 24 ∧ n > 0) :
  ({n | ∃ (n : ℕ), Nat.lcm (2 * n) (n ^ 2) = 14 * n - 24 ∧ n > 0}.to_finset.sum id) = 17 :=
by
  sorry

end sum_of_valid_n_l181_181397


namespace problem_G81_G82_l181_181148

theorem problem_G81_G82 (α β m : ℤ) (h_eq : ∀ x, x^2 + (m+1)*x - 2 = 0 → (x = α + 1 ∨ x = β + 1))
(h_lt : α < β) (h_ne : m ≠ 0) : m = -2 ∧ β - α = 3 :=
by
  -- we should have proper definition first for the roots
  have h_roots : α + 1 = -1 ∧ β + 1 = 2 ∨ α + 1 = -2 ∧ β + 1 = 1,
  { sorry }
  
  -- then we have to extract valid pair (α, β)
  have h_valid_pair : α = -2 ∧ β = 1,
  { sorry }
  
  -- finally we can prove m = -2 and β - α = 3
  have h_m : m = -(1 + 2),
  { sorry }
  
  have h_d : β - α = 1 +2,
  { sorry }
  
  exact ⟨h_m, h_d⟩

end problem_G81_G82_l181_181148


namespace sum_of_digits_l181_181025

theorem sum_of_digits (k : ℕ) (h : k = 10 ^ 40 - 46) :
  digit_sum k = 360 :=
  sorry

end sum_of_digits_l181_181025


namespace sum_binomial_coeffs_equal_sum_k_values_l181_181911

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l181_181911


namespace solve_eq_l181_181102

theorem solve_eq : 
  ∀ x : ℂ, x^4 + 64 = 0 ↔ x = 2 + 2 * Complex.i ∨ x = -2 - 2 * Complex.i ∨ x = -2 + 2 * Complex.i ∨ x = 2 - 2 * Complex.i :=
by
  sorry

end solve_eq_l181_181102


namespace sum_of_distinct_complex_nums_l181_181943

noncomputable section

open Complex

theorem sum_of_distinct_complex_nums (m n : ℂ) (h1 : m ≠ n) (h2 : m * n ≠ 0)
    (h3 : {m, n} = {m^2, n^2}) : m + n = -1 :=
by
  sorry

end sum_of_distinct_complex_nums_l181_181943


namespace seating_arrangement_count_l181_181620

theorem seating_arrangement_count
  (people_count : ℕ)
  (specific_between : ℕ)
  (total_ways : ℕ)
  (h1 : people_count = 7)
  (h2 : specific_between = 2)
  (h3 : total_ways = nat.factorial (people_count - 2) * specific_between)
  : total_ways = 240 :=
by
  simp [h1, h2, nat.factorial, nat.factorial_succ, nat.factorial_pos, mul_comm] at h3,
  exact h3
  sorry

end seating_arrangement_count_l181_181620


namespace find_PB_line_l181_181271

-- Define the conditions
def PointOnXAxis (p : ℝ×ℝ) : Prop := p.snd = 0

def xCoord (p : ℝ×ℝ) (x : ℝ) : Prop := p.fst = x

def sameDistance (a b p : ℝ×ℝ) : Prop := dist a p = dist b p

def lineEquation (a b c : ℝ) (p : ℝ×ℝ) : Prop := a * p.fst + b * p.snd + c = 0

-- Define points and line conditions
variables (A B P : ℝ×ℝ)
variable  (x1 : ℝ)

-- Point constraints
hypothesis A_on_xAxis : PointOnXAxis A
hypothesis B_on_xAxis : PointOnXAxis B
hypothesis P_xCoord : xCoord P 1

-- Distance constraint
hypothesis equidistant : sameDistance A B P

-- Equation of PA
hypothesis PA_eq : lineEquation 1 (-1) 1 P

-- Target equation of PB
axiom PB_equation : ∃ (a b c : ℝ), lineEquation a b c P ∧ a * P.fst + b * P.snd + c = 0

-- Final proof statement
theorem find_PB_line (A_on_xAxis : PointOnXAxis A) (B_on_xAxis : PointOnXAxis B)
  (P_xCoord : xCoord P 1) (equidistant : sameDistance A B P) (PA_eq : lineEquation 1 (-1) 1 P) :
  lineEquation 1 1 (-3) P :=
sorry

end find_PB_line_l181_181271


namespace parabola_focus_l181_181505

noncomputable def focus_of_parabola : ℝ × ℝ :=
  let f : ℝ := 1/8 in
  (0, f)

theorem parabola_focus :
  ∀ (x : ℝ), ∃ (f : ℝ), (f = 1/8) ∧
    (let y := 2 * x ^ 2 in -- Parabola y = 2x^2
     let P := (x, y) in
     let PF_sq := x^2 + (y - f)^2 in -- Distance squared from P to F
     let d := f in -- Directrix y = d
     let PQ_sq := (y - d)^2 in -- Distance squared from P to the directrix
     PF_sq = PQ_sq) :=
by
  sorry

end parabola_focus_l181_181505


namespace b_minus_a_condition_l181_181128

theorem b_minus_a_condition (a b : ℝ) (h : {a, 1} = {0, a + b}) : b - a = 1 :=
  sorry

end b_minus_a_condition_l181_181128


namespace maitre_d_solution_l181_181023

def maitre_d_problem (P : Set ℕ → ℝ) (D C : Set ℕ) : Prop :=
  let P_D_and_C : ℝ := 0.60
  let P_D_and_not_C : ℝ := 0.20
  let P_D := P_D_and_C + P_D_and_not_C
  let P_not_D := 1 - P_D
  P_not_D = 0.20

theorem maitre_d_solution : maitre_d_problem P D C := 
by
  let P_D_and_C := 0.60
  let P_D_and_not_C := 0.20
  let P_D := P_D_and_C + P_D_and_not_C
  let P_not_D := 1 - P_D
  have h : P_not_D = 0.20 := rfl
  sorry

end maitre_d_solution_l181_181023


namespace problem_solution_l181_181203

def is_sequence_of_zeros_and_ones (s : List ℕ) : Prop :=
  s.all (λ x, x = 0 ∨ x = 1)

def length_15 (s : List ℕ) : Prop :=
  s.length = 15

def all_zeros_consecutive (s : List ℕ) : Prop :=
  ∃ i j, i ≤ j ∧ (∀ k, i ≤ k ∧ k ≤ j → s[k] = 0) ∧ (∀ k, k < i ∨ k > j → s[k] = 1)

def at_least_three_consecutive_ones (s : List ℕ) : Prop :=
  ∃ i, i + 2 < s.length ∧ (s[i] = 1 ∧ s[i+1] = 1 ∧ s[i+2] = 1)

noncomputable def count_sequences (p : List ℕ → Prop) : ℕ :=
  (List.filter p (List.replicateM 15 [0, 1])).length

theorem problem_solution :
  count_sequences (λ s, is_sequence_of_zeros_and_ones s ∧ length_15 s ∧ (all_zeros_consecutive s ∨ at_least_three_consecutive_ones s)) = 225 := 
sorry

end problem_solution_l181_181203


namespace seqA_increasing_seqB_increasing_seqC_not_increasing_seqD_increasing_l181_181480

-- Define sequence given in option A
def seqA (n : ℕ) : ℝ := n / (n + 1)

-- Prove that sequence in option A is increasing
theorem seqA_increasing : ∀ n : ℕ, seqA (n + 1) > seqA n :=
by
  sorry

-- Define sequence given in option B
def seqB (n : ℕ) : ℝ := -((1/2) ^ n)

-- Prove that sequence in option B is increasing
theorem seqB_increasing : ∀ n : ℕ, seqB (n + 1) > seqB n :=
by
  sorry

-- Define sequence given in option C
def seqC : ℕ → ℝ
| 0       := 1
| (n + 1) := 3 - seqC n

-- Prove that sequence in option C is not increasing
theorem seqC_not_increasing : ∃ n : ℕ, seqC (n + 1) ≤ seqC n :=
by
  sorry

-- Define sequence given in option D
def seqD : ℕ → ℝ
| 0       := 1
| (n + 1) := seqD n ^ 2 - seqD n + 2

-- Prove that sequence in option D is increasing
theorem seqD_increasing : ∀ n : ℕ, seqD (n + 1) > seqD n :=
by
  sorry

end seqA_increasing_seqB_increasing_seqC_not_increasing_seqD_increasing_l181_181480


namespace power_function_odd_f_m_plus_1_l181_181033

noncomputable def f (x : ℝ) (m : ℝ) := x^(2 + m)

theorem power_function_odd_f_m_plus_1 (m : ℝ) (h_odd : ∀ x : ℝ, f (-x) m = -f x m)
  (h_domain : -1 ≤ m) : f (m + 1) m = 1 := by
  sorry

end power_function_odd_f_m_plus_1_l181_181033


namespace problem_statement_l181_181242

noncomputable def find_pq_sum (XZ YZ : ℕ) (XY_perimeter_ratio : ℕ × ℕ) : ℕ :=
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let ZD := Real.sqrt (XZ * YZ)
  let O_radius := 0.5 * ZD
  let tangent_length := Real.sqrt ((XY / 2)^2 - O_radius^2)
  let perimeter := XY + 2 * tangent_length
  let (p, q) := XY_perimeter_ratio
  p + q

theorem problem_statement :
  find_pq_sum 8 15 (30, 17) = 47 :=
by sorry

end problem_statement_l181_181242


namespace average_age_increase_l181_181717

theorem average_age_increase 
    (num_students : ℕ) (avg_age_students : ℕ) (age_staff : ℕ)
    (H1: num_students = 32)
    (H2: avg_age_students = 16)
    (H3: age_staff = 49) : 
    ((num_students * avg_age_students + age_staff) / (num_students + 1) - avg_age_students = 1) :=
by
  sorry

end average_age_increase_l181_181717


namespace sum_of_roots_eq_k_div_4_l181_181651

variables {k d y_1 y_2 : ℝ}

theorem sum_of_roots_eq_k_div_4 (h1 : y_1 ≠ y_2)
                                  (h2 : 4 * y_1^2 - k * y_1 = d)
                                  (h3 : 4 * y_2^2 - k * y_2 = d) :
  y_1 + y_2 = k / 4 :=
sorry

end sum_of_roots_eq_k_div_4_l181_181651


namespace solve_for_x_l181_181854

theorem solve_for_x : 
  (∃ x : ℝ, (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) 
  ∧ x = 4.5) := by
{
  sorry
}

end solve_for_x_l181_181854


namespace major_axis_length_of_intersecting_ellipse_l181_181456

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l181_181456


namespace sequence_value_a_10_l181_181936

theorem sequence_value_a_10 :
  (∀ n, S n = 3 * a n + 1) →
  a 1 = -1/2 →
  (∀ n, a (n + 1) = 3/2 * a n) →
  a 10 = -3^9 / 2^10 :=
by
  sorry

end sequence_value_a_10_l181_181936


namespace fraction_value_l181_181931

theorem fraction_value (x : ℝ) (h₀ : x^2 - 3 * x - 1 = 0) (h₁ : x ≠ 0) : 
  x^2 / (x^4 + x^2 + 1) = 1 / 12 := 
by
  sorry

end fraction_value_l181_181931


namespace gravitational_equal_forces_point_l181_181488

variable (d M m : ℝ) (hM : 0 < M) (hm : 0 < m) (hd : 0 < d)

theorem gravitational_equal_forces_point :
  ∃ x : ℝ, (0 < x ∧ x < d) ∧ x = d / (1 + Real.sqrt (m / M)) :=
by
  sorry

end gravitational_equal_forces_point_l181_181488


namespace amount_received_by_a_l181_181772

namespace ProofProblem

/-- Total amount of money divided -/
def total_amount : ℕ := 600

/-- Ratio part for 'a' -/
def part_a : ℕ := 1

/-- Ratio part for 'b' -/
def part_b : ℕ := 2

/-- Total parts in the ratio -/
def total_parts : ℕ := part_a + part_b

/-- Amount per part when total is divided evenly by the total number of parts -/
def amount_per_part : ℕ := total_amount / total_parts

/-- Amount received by 'a' when total amount is divided according to the given ratio -/
def amount_a : ℕ := part_a * amount_per_part

theorem amount_received_by_a : amount_a = 200 := by
  -- Proof will be filled in here
  sorry

end ProofProblem

end amount_received_by_a_l181_181772


namespace remainder_3_pow_405_mod_13_l181_181394

theorem remainder_3_pow_405_mod_13 : (3^405) % 13 = 1 :=
by
  sorry

end remainder_3_pow_405_mod_13_l181_181394


namespace right_angled_triangle_l181_181322

theorem right_angled_triangle (A B C : ℝ) (h1 : A + B + C = real.pi)
  (h2 : (real.sin A)^2 + (real.sin B)^2 + (real.sin C)^2 = 2 * ((real.cos A)^2 + (real.cos B)^2 + (real.cos C)^2)) :
  C = real.pi / 2 :=
sorry

end right_angled_triangle_l181_181322


namespace direction_vector_projection_matrix_l181_181350

theorem direction_vector_projection_matrix :
  let P := ![
    ![1/7, -2/7, 3/7],
    ![-2/7, 4/7, -6/7],
    ![3/7, -6/7, 9/7]
  ]
  ∃ (v : ℝ × ℝ × ℝ), v = (1, -2, 3) ∧
  let i : ℝ × ℝ × ℝ := (1, 0, 0),
      proj_i := (P 0) :
  (proj_i = 1/7 • (v)).tuple :=
sorry 

end direction_vector_projection_matrix_l181_181350


namespace necessary_but_not_sufficient_l181_181944

noncomputable def real_numbers := Type

def m_in_real (m : real_numbers) := True
def n_in_real (n : real_numbers) := True

def equation_represents_curve (m n : real_numbers) : Prop := 
  mx^2 + ny^2 = 1

theorem necessary_but_not_sufficient (m n : real_numbers) :
  (mn > 0) → (equation_represents_curve m n → is_ellipse m n) ∧
  (is_ellipse m n → mn > 0) :=
by
  sorry

end necessary_but_not_sufficient_l181_181944


namespace B_divides_A_l181_181778

noncomputable def polynomial_division (A B : Polynomial ℝ) : Prop := 
  ∀ (A B : Polynomial (ℝ × ℝ)), 
  (∃ C : Polynomial (ℝ × ℝ), A = B * C) → 
  (∀ y, ∃ L : Polynomial ℝ, A.eval₂ C y = L.eval₂ C y) → 
  (∀ x, ∃ M : Polynomial ℝ, A.eval₂ C x = M.eval₂ C x) → 
  ∃ C : Polynomial (ℝ × ℝ), A = B * C

theorem B_divides_A (A B : Polynomial (ℝ × ℝ)) 
  (h1 : ∀ y : ℝ, ∃ P : Polynomial ℝ, (A.eval₂ (Polynomial.C ∘ Prod.fst) y) = B.eval₂ (Polynomial.C ∘ Prod.fst) y * P) 
  (h2 : ∀ x : ℝ, ∃ Q : Polynomial ℝ, (A.eval₂ (Polynomial.C ∘ Prod.snd) x) = B.eval₂ (Polynomial.C ∘ Prod.snd) x * Q) : 
  ∃ C : Polynomial (ℝ × ℝ), A = B * C :=
sorry

end B_divides_A_l181_181778


namespace james_nickels_l181_181630

theorem james_nickels (p n : ℕ) (h₁ : p + n = 50) (h₂ : p + 5 * n = 150) : n = 25 :=
by
  -- Skipping the proof since only the statement is required
  sorry

end james_nickels_l181_181630


namespace trig_eqn_solution_l181_181707

noncomputable def solve_trig_eqn (x y z : ℝ) (m n : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (cos y ≠ 0) ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) ∧
  (x = π / 2 + π * m) ∧
  (y = π * n) ∧
  (z = 2 * π * m)

theorem trig_eqn_solution (x y z : ℝ) (m n : ℤ) :
  sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  ( (sin^2 x + 1 / sin^2 x)^3 + (cos^2 y + 1 / cos^2 y)^3 = 16 * cos z ) →
  x = π / 2 + π * m ∧ y = π * n ∧ z = 2 * π * m :=
by
  sorry

end trig_eqn_solution_l181_181707


namespace Donny_change_l181_181513

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l181_181513


namespace find_page_added_twice_l181_181739

theorem find_page_added_twice (m p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : (m * (m + 1)) / 2 + p = 2550) : p = 6 :=
sorry

end find_page_added_twice_l181_181739


namespace complex_div_conj_i_l181_181142

-- Given conditions
def i_unit : ℂ := complex.I
def z : ℂ := 3 - 4 * i_unit
def z_conj : ℂ := 3 + 4 * i_unit

-- Statement to prove
theorem complex_div_conj_i : (z_conj / i_unit) = (4 - 3 * i_unit) := 
by sorry

end complex_div_conj_i_l181_181142


namespace jacob_distance_l181_181628

theorem jacob_distance (total_half_marathons : ℕ) (miles_per_half_marathon : ℕ) 
  (yards_per_half_marathon : ℕ) (yards_per_mile : ℕ) 
  (H_total_half_marathons : total_half_marathons = 15) 
  (H_miles_per_half_marathon : miles_per_half_marathon = 13) 
  (H_yards_per_half_marathon : yards_per_half_marathon = 193) 
  (H_yards_per_mile : yards_per_mile = 1760) : 
  let total_yards := total_half_marathons * yards_per_half_marathon in
  total_yards % yards_per_mile = 1135 :=
by
  sorry

end jacob_distance_l181_181628


namespace number_of_a2_values_l181_181052

theorem number_of_a2_values 
  (sequence : ℕ → ℕ)
  (h_def : ∀ n ≥ 1, sequence (n + 2) = abs (sequence (n + 1) - sequence n))
  (h_initial : sequence 1 = 999)
  (h_bound : sequence 2 < 999)
  (h_target : sequence 2023 = 0) :
  (finset.range 999).filter (λ n, n % 2 = 0 ∧ gcd 999 n = 1).card = 131 :=
sorry

end number_of_a2_values_l181_181052


namespace min_ratio_M_over_m_l181_181917

noncomputable def alpha : ℝ := (1 + real.sqrt 5) / 2

theorem min_ratio_M_over_m (S : set (ℝ × ℝ)) (h : S.card = 5) (h_ncl : ∀ t : finset (ℝ × ℝ), t ⊆ S → t.card = 3 → ¬ affine_independent ℝ (coe ∘ t)) :
  (∃ T₁ T₂ : finset (ℝ × ℝ), T₁ ⊆ S ∧ T₂ ⊆ S ∧ T₁.card = 3 ∧ T₂.card = 3 ∧
    T₁.area > 0 ∧ T₂.area > 0 ∧
    (T₁.area / T₂.area = alpha)) :=
sorry

end min_ratio_M_over_m_l181_181917


namespace sequence_terms_divisible_by_b_l181_181680

theorem sequence_terms_divisible_by_b (a b : ℕ) :
  let d := Nat.gcd a b in
  (d = (List.range (b + 1)).filter (λ n, (a * n) % b = 0).length) :=
by
  sorry

end sequence_terms_divisible_by_b_l181_181680


namespace product_of_base8_digits_l181_181390

theorem product_of_base8_digits (n : ℕ) (h : n = 7890) : 
  let base8_repr := [1, 7, 3, 2, 2] in 
  base8_repr.product = 84 :=
by 
  -- Proof omitted
  sorry

end product_of_base8_digits_l181_181390


namespace number_of_integer_roots_eq_3_l181_181353

theorem number_of_integer_roots_eq_3 : 
  (∃ x : ℤ, ⌊(3 * x + 7) / 7⌋ = 4) ∧ (set.finite { x : ℤ | ⌊(3 * x + 7) / 7⌋ = 4 }) ∧ (set.card { x : ℤ | ⌊(3 * x + 7) / 7⌋ = 4 } = 3) :=
sorry

end number_of_integer_roots_eq_3_l181_181353


namespace part_one_part_two_l181_181094

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem part_one (x : ℝ) : f x < 4 - abs (x - 1) ↔ x ∈ Set.Ioo (-5 / 4) (1 / 2) :=
sorry

noncomputable def g (x a : ℝ) : ℝ :=
if x < -2/3 then 2 * x + 2 + a
else if x ≤ a then -4 * x - 2 + a
else -2 * x - 2 - a

theorem part_two (m n a : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) :
  (∀ (x : ℝ), abs (x - a) - f x ≤ 1 / m + 1 / n) ↔ (0 < a ∧ a ≤ 10 / 3) :=
sorry

end part_one_part_two_l181_181094


namespace find_angle_A_find_area_triangle_l181_181226

-- Definitions for the triangle and the angles
def triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- Given conditions
variables (a b c A B C : ℝ)
variables (hTriangle : triangle A B C)
variables (hEq : 2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C)
variables (hAngleB : B = Real.pi / 6)
variables (hMedianAM : Real.sqrt 7 = Real.sqrt (b^2 + (b / 2)^2 - 2 * b * (b / 2) * Real.cos (2 * Real.pi / 3)))

-- Proof statements
theorem find_angle_A : A = Real.pi / 6 :=
sorry

theorem find_area_triangle : (1/2) * b^2 * Real.sin C = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_triangle_l181_181226


namespace sum_k_binomial_l181_181905

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l181_181905


namespace count_valid_positive_integers_l181_181977

theorem count_valid_positive_integers :
  {n : ℕ // n > 0 ∧ (Int.floor (1000000 / n) - Int.floor (1000000 / (n + 1)) = 1)}.card = 1172 :=
by
  sorry

end count_valid_positive_integers_l181_181977


namespace geometric_prod_inequality_l181_181932

theorem geometric_prod_inequality 
  {a : ℕ → ℝ} 
  (n : ℕ) 
  (h : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_prod : ∏ i in finset.range n, a i = 1) :
  ∏ i in finset.range n, (2 + a i) ≥ 3 ^ n := by
  sorry

end geometric_prod_inequality_l181_181932


namespace find_a_for_even_function_l181_181164

open Function

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- State the problem in Lean.
theorem find_a_for_even_function :
  (∀ x, f a x = f a (-x)) → a = 1 :=
sorry

end find_a_for_even_function_l181_181164


namespace work_days_B_l181_181770

theorem work_days_B (A B: ℕ) (work_per_day_B: ℕ) (total_days : ℕ) (total_units : ℕ) :
  (A = 2 * B) → (work_per_day_B = 1) → (total_days = 36) → (B = 1) → (total_units = total_days * (A + B)) → 
  total_units / work_per_day_B = 108 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_days_B_l181_181770


namespace range_of_k_for_real_roots_l181_181992

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l181_181992


namespace prob_one_red_correct_prob_two_red_or_more_correct_l181_181811

-- Defining main conditions
def prob_red_light : ℝ := 2 / 5
def prob_not_red_light : ℝ := 1 - prob_red_light
def num_traffic_lights : ℕ := 3

-- Question 1: Probability of encountering exactly 1 red light
def prob_one_red (n : ℕ) (p_red p_not_red : ℝ) : ℝ :=
  Nat.choose n 1 * (p_red^(1)) * (p_not_red^(n-1))

theorem prob_one_red_correct : 
  prob_one_red num_traffic_lights prob_red_light prob_not_red_light = 54 / 125 := sorry

-- Question 2: Probability of encountering at least 2 red lights
def prob_two_red_or_more (n : ℕ) (p_red p_not_red : ℝ) : ℝ :=
  Nat.choose n 2 * (p_red^(2)) * (p_not_red^(n-2)) + p_red^(n)

theorem prob_two_red_or_more_correct : 
  prob_two_red_or_more num_traffic_lights prob_red_light prob_not_red_light = 44 / 125 := sorry

end prob_one_red_correct_prob_two_red_or_more_correct_l181_181811


namespace arithmetic_seq_sum_ratio_l181_181939

theorem arithmetic_seq_sum_ratio
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : S 25 / a 23 = 5)
  (h3 : S 45 / a 33 = 25) :
  S 65 / a 43 = 45 :=
by sorry

end arithmetic_seq_sum_ratio_l181_181939


namespace min_φ_6n_l181_181538

def φ (n : ℕ) : ℕ :=
n.divisors.card

noncomputable def min_divisors_6n (n : ℕ) : ℕ :=
if hn : φ(2 * n) = 6 then φ(6 * n) else 0

theorem min_φ_6n (n : ℕ) (h : φ(2 * n) = 6) : ∃ k, min_divisors_6n n = k ∧ k = 8 := 
by sorry

end min_φ_6n_l181_181538


namespace tanner_savings_in_november_l181_181326

theorem tanner_savings_in_november(savings_sep : ℕ) (savings_oct : ℕ) 
(spending : ℕ) (leftover : ℕ) (N : ℕ) :
savings_sep = 17 →
savings_oct = 48 →
spending = 49 →
leftover = 41 →
((savings_sep + savings_oct + N - spending) = leftover) →
N = 25 :=
by
  intros h_sep h_oct h_spending h_leftover h_equation
  sorry

end tanner_savings_in_november_l181_181326


namespace molecular_weight_of_4_moles_of_AlOH₃_l181_181005

theorem molecular_weight_of_4_moles_of_AlOH₃ :
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 16.00
  let atomic_weight_H := 1.01
  let molecular_weight_AlOH₃ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H
  4 * molecular_weight_AlOH₃ = 312.04 := by
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 16.00
  let atomic_weight_H := 1.01
  let molecular_weight_AlOH₃ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H
  calc
  4 * molecular_weight_AlOH₃ = 4 * (26.98 + 3 * 16.00 + 3 * 1.01) : by rfl
                        ... = 4 * 78.01 : by rfl
                        ... = 312.04 : by rfl

end molecular_weight_of_4_moles_of_AlOH₃_l181_181005


namespace intersection_difference_l181_181740

theorem intersection_difference :
  let y1 := λ x : ℝ, 3 * x^2 - 6 * x + 6
  let y2 := λ x : ℝ, -2 * x^2 - 4 * x + 6
  ∃ a b c d : ℝ, (a, b) ≠ (c, d) ∧ a ≤ c ∧
  (y1 a = y2 a) ∧ (y1 c = y2 c) ∧ (c - a = 2 / 5) :=
by
  let y1 := λ x : ℝ, 3 * x^2 - 6 * x + 6
  let y2 := λ x : ℝ, -2 * x^2 - 4 * x + 6
  have ha : ∃ a, y1 a = y2 a := sorry
  have hc : ∃ c, y1 c = y2 c := sorry
  obtain ⟨a, ha⟩ := ha
  obtain ⟨c, hc⟩ := hc
  use [a, y1 a, c, y1 c]
  split
  { -- Proof that (a, y1 a) ≠ (c, y1 c)
    sorry
  }
  split
  { -- Proof that a ≤ c
    sorry
  }
  split
  { -- Proof that y1 a = y2 a
    assumption
  }
  split
  { -- Proof that y1 c = y2 c
    assumption
  }
  { -- Proof that c - a = 2 / 5
    sorry
  }

end intersection_difference_l181_181740


namespace cars_in_first_section_l181_181295

noncomputable def first_section_rows : ℕ := 15
noncomputable def first_section_cars_per_row : ℕ := 10
noncomputable def total_cars_first_section : ℕ := first_section_rows * first_section_cars_per_row

theorem cars_in_first_section : total_cars_first_section = 150 :=
by
  sorry

end cars_in_first_section_l181_181295


namespace arithmetic_seq_a11_l181_181940

variable (a : ℕ → ℤ)
variable (d : ℕ → ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n, a (n + 2) - a n = 6
def a1 : Prop := a 1 = 1

-- Statement of the problem
theorem arithmetic_seq_a11 : arithmetic_sequence a ∧ a1 a → a 11 = 31 :=
by sorry

end arithmetic_seq_a11_l181_181940


namespace median_equidistant_from_vertices_l181_181321

theorem median_equidistant_from_vertices (A B C M : Point) (triangle_ABC : Triangle A B C)
    (M_midpoint : midpoint M B C)
    (AM_median : median A M (B, C)) :
    dist A B = dist A C := 
by 
  sorry

end median_equidistant_from_vertices_l181_181321


namespace no_such_integers_exist_l181_181306

theorem no_such_integers_exist (x y z : ℤ) (hx : x ≠ 0) :
  ¬ (2 * x ^ 4 + 2 * x ^ 2 * y ^ 2 + y ^ 4 = z ^ 2) :=
by
  sorry

end no_such_integers_exist_l181_181306
