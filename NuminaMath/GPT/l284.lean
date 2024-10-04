import Mathlib

namespace maximum_value_AP_l284_284483

noncomputable def max_value_vector_AP : Real := sorry

theorem maximum_value_AP (A B C D P : Type) 
  (h1 : ∠ BAC = π / 3) 
  (h2 : Midpoint D A B) 
  (h3 : ∃ t : Real, AP = t • AC + (1 / 3) • AB) 
  (h4 : norm BC = sqrt 6) : 
  max_value_vector_AP = sqrt 2 :=
sorry

end maximum_value_AP_l284_284483


namespace largest_triangle_perimeter_l284_284669

theorem largest_triangle_perimeter (x : ℤ) (hx1 : 7 + 11 > x) (hx2 : 7 + x > 11) (hx3 : 11 + x > 7) (hx4 : 5 ≤ x) (hx5 : x < 18) : 
  7 + 11 + x = 35 :=
sorry

end largest_triangle_perimeter_l284_284669


namespace q_x_value_l284_284961

noncomputable def q (x : ℝ) : ℝ := -2 * x^6 + x^4 + 18 * x^3 + 12 * x^2 + 2

theorem q_x_value (x : ℝ) :
  q x + (2 * x^6 + 4 * x^4 + 8 * x^2) = 5 * x^4 + 18 * x^3 + 20 * x^2 + 2 :=
by
  calc
    q x + (2 * x^6 + 4 * x^4 + 8 * x^2)
        = (-2 * x^6 + x^4 + 18 * x^3 + 12 * x^2 + 2) + (2 * x^6 + 4 * x^4 + 8 * x^2) : rfl
    -- Combine like terms manually without adding proof steps
    ... = (x^4 + 4 * x^4) + 18 * x^3 + (12 * x^2 + 8 * x^2) + 2 : by ring
    ... = 5 * x^4 + 18 * x^3 + 20 * x^2 + 2 : by ring

end q_x_value_l284_284961


namespace students_like_basketball_l284_284471

variable (B C B_inter_C B_union_C : ℕ)

theorem students_like_basketball (hC : C = 8) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) 
    (h_incl_excl : B_union_C = B + C - B_inter_C) : B = 12 := by 
  -- Given: 
  --   C = 8
  --   B_inter_C = 3
  --   B_union_C = 17
  --   B_union_C = B + C - B_inter_C
  -- Prove: 
  --   B = 12
  sorry

end students_like_basketball_l284_284471


namespace pairs_of_positive_integers_l284_284813

theorem pairs_of_positive_integers (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) : 
  (m^2 + 3 * n < 50) → (m, n) ∈ { 1, 2, 3, 4, 5, 6, 7 }.product {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16} ↔ (m, n) :=
sorry

end pairs_of_positive_integers_l284_284813


namespace sets_of_consecutive_integers_summing_to_20_l284_284069

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l284_284069


namespace abs_neg_fraction_is_positive_l284_284197

-- Define the given negative fraction
def neg_fraction := (-1 : ℝ) / 3

-- The absolute value of the given fraction
def abs_of_neg_fraction := abs neg_fraction

-- Define the expected absolute value (correct answer)
def expected_abs_value := (1 : ℝ) / 3

-- The theorem stating that the absolute value of -1/3 is 1/3
theorem abs_neg_fraction_is_positive : abs_of_neg_fraction = expected_abs_value := by
  sorry

end abs_neg_fraction_is_positive_l284_284197


namespace simplify_and_evaluate_expression_l284_284180

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 4) : 
  (x + 2 + 3 / (x - 2)) / (1 + 2 * x + x^2) / (x - 2) = 3 / 5 := by
  rw h
  sorry

end simplify_and_evaluate_expression_l284_284180


namespace find_scalars_p_q_l284_284922

open Matrix

variable (R : Type*) [CommRing R]

def N : Matrix (Fin 2) (Fin 2) R := ![![3, -4], ![2, -5]]
def I : Matrix (Fin 2) (Fin 2) R := 1

theorem find_scalars_p_q (p q : R) (h₁ : p = -1) (h₂ : q = 4) : N R * N R = p • N R + q • I :=
by
  sorry

end find_scalars_p_q_l284_284922


namespace three_digit_numbers_mod_1000_l284_284516

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000_l284_284516


namespace expected_pourings_correct_l284_284535

section
  /-- Four glasses are arranged in a row: the first and third contain orange juice, 
      the second and fourth are empty. Valya can take a full glass and pour its 
      contents into one of the two empty glasses each time. -/
  def initial_state : List Bool := [true, false, true, false]
  def target_state : List Bool := [false, true, false, true]

  /-- Define a function to calculate the expected number of pourings required to 
      reach the target state from the initial state given the probabilities of 
      transitions. -/
  noncomputable def expected_number_of_pourings (init : List Bool) (target : List Bool) : ℕ :=
    if init = initial_state ∧ target = target_state then 6 else 0

  /-- Prove that the expected number of pourings required to transition from 
      the initial state [true, false, true, false] to the target state [false, true, false, true] is 6. -/
  theorem expected_pourings_correct :
    expected_number_of_pourings initial_state target_state = 6 :=
  by
    -- Proof omitted
    sorry
end

end expected_pourings_correct_l284_284535


namespace cos_diff_identity_l284_284745

theorem cos_diff_identity (α β : ℝ) 
  (h1 : cos α + cos β = 1 / 2) 
  (h2 : sin α + sin β = 1 / 3) : 
  cos (α - β) = -59 / 72 :=
sorry

end cos_diff_identity_l284_284745


namespace basic_computer_price_l284_284259

variable (C P : ℕ)

theorem basic_computer_price 
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3) : 
  C = 1500 := 
sorry

end basic_computer_price_l284_284259


namespace cos_alpha_alpha_plus_beta_l284_284264

-- Problem 1: Prove that cos(α) = (sqrt(3) + 2sqrt(2)) / 6 given the conditions

theorem cos_alpha (α : ℝ) (h1 : π / 6 < α) (h2 : α < π / 2) (h3 : cos (α + π / 6) = 1 / 3) : 
  cos α = (sqrt 3 + 2 * sqrt 2) / 6 :=
by
  sorry

-- Problem 2: Prove that α + β = 3π / 4 given the conditions

theorem alpha_plus_beta (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) 
  (h5 : cos α = sqrt 5 / 5) (h6 : cos β = sqrt 10 / 10) : 
  α + β = 3 * π / 4 :=
by
  sorry

end cos_alpha_alpha_plus_beta_l284_284264


namespace millionth_digit_of_fraction_l284_284730

theorem millionth_digit_of_fraction (n : ℕ) (hn : n = 1000000) : 
  ∃ (d : ℕ), d = 7 ∧ (decimal_expansion (3 / 41) (hn - 1) = d) :=
sorry

end millionth_digit_of_fraction_l284_284730


namespace number_of_functions_satisfying_equation_l284_284732

theorem number_of_functions_satisfying_equation :
  (∃ f : ℤ → ℤ, ∀ h k : ℤ, f(h + k) + f(hk) = f(h) * f(k) + 1) →
  (∃ f1 f2 f3 : ℤ → ℤ, 
  (∀ h k : ℤ, f1(h + k) + f1(hk) = f1(h) * f1(k) + 1) ∧
  (∀ h k : ℤ, f2(h + k) + f2(hk) = f2(h) * f2(k) + 1) ∧
  (∀ h k : ℤ, f3(h + k) + f3(hk) = f3(h) * f3(k) + 1) ∧
  (∀ g : ℤ → ℤ, (∀ h k : ℤ, g(h + k) + g(hk) = g(h) * g(k) + 1) → 
  (g = f1 ∨ g = f2 ∨ g = f3))) :=
sorry

end number_of_functions_satisfying_equation_l284_284732


namespace no_s_n_equals_2022_exists_s_n_equals_2023_l284_284900

def is_nice (x y : ℕ) : Prop :=
  1 ≤ y ∧ y < x ∧ (finset.card (finset.filter (λ d, d ∣ x ∧ d ∣ y) (finset.range (x - y + 1))) = x - y)

def s (n : ℕ) : ℕ :=
  finset.card (finset.filter (λ p : ℕ × ℕ, is_nice p.1 p.2) 
    (finset.product (finset.range (n + 1)) (finset.range (n + 1))))

theorem no_s_n_equals_2022 (n : ℕ) (hn : n ≥ 2) : s(n) ≠ 2022 :=
sorry

theorem exists_s_n_equals_2023 : ∃ n, n ≥ 2 ∧ s(n) = 2023 :=
  ⟨1350, by norm_num, sorry⟩

end no_s_n_equals_2022_exists_s_n_equals_2023_l284_284900


namespace find_a_l284_284000

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284000


namespace invalid_perimeters_l284_284238

theorem invalid_perimeters (x : ℕ) (h1 : 18 < x) (h2 : x < 42) :
  (42 + x ≠ 58) ∧ (42 + x ≠ 85) :=
by
  sorry

end invalid_perimeters_l284_284238


namespace correct_relation_l284_284524

def M : set ℝ := {x | x < 2012}
def N : set ℝ := {x | 0 < x < 1}

theorem correct_relation : M ∩ N = {x | 0 < x < 1} :=
by sorry

end correct_relation_l284_284524


namespace even_quadratic_iff_b_zero_l284_284621

-- Define a quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- State the theorem
theorem even_quadratic_iff_b_zero (a b c : ℝ) : 
  (∀ x : ℝ, quadratic a b c x = quadratic a b c (-x)) ↔ b = 0 := 
by
  sorry

end even_quadratic_iff_b_zero_l284_284621


namespace compare_fractions_l284_284312

theorem compare_fractions {x : ℝ} (h : 3 < x ∧ x < 4) : 
  (2 / 3) > ((5 - x) / 3) :=
by sorry

end compare_fractions_l284_284312


namespace petya_payment_l284_284878

theorem petya_payment : 
  ∃ (x y : ℕ), 
  (14 * x + 3 * y = 107) ∧ 
  (|x - y| ≤ 5) ∧
  (x + y = 10) := 
sorry

end petya_payment_l284_284878


namespace jimmy_wins_the_bet_l284_284317

noncomputable def jimmy_wins : Prop :=
  ∃ (shoot_time: ℝ) (bullet_speed: ℝ), 
    -- The bullet's trajectory intersects all four blades
    let blade_rotation_speed := 50 in
    let blade_positions := [0, π/2, π, 3*π/2] in
    ∀ (blade : ℕ) (current_position : ℝ),
      blade < 4 →
      current_position = blade_positions[blade] + blade_rotation_speed * shoot_time →
      -- Bullet intersects when shot at the right time (shoot_time) and speed (bullet_speed)
      bullet_speed * shoot_time > current_position

-- The theorem stating Jimmy wins the bet
theorem jimmy_wins_the_bet : jimmy_wins := sorry

end jimmy_wins_the_bet_l284_284317


namespace general_term_comparison_l284_284010

variable (t : ℝ) (n : ℕ) (hn : n ≠ 0)
variable (a : ℕ → ℝ)
hypothesis (h_t_nonzero : t ≠ 1 ∧ t ≠ -1)

/-- Define the sequence -/
def a_seq : ℝ → ℕ → ℝ
| t 1 := 2 * t - 2
| t (n+1) := (2 * (t^(n+1) - 1) * a_seq t n) / (a_seq t n + 2 * t^n - 2)

/-- General term formula --/
theorem general_term :
  (∀ n > 0, a_seq t n = 2 * (t^n - 1) / n) :=
sorry

/-- Comparison of terms for t > 0 --/
theorem comparison (h_t_positive : t > 0) :
  ∀ n > 0, a_seq t (n+1) > a_seq t n :=
sorry

end general_term_comparison_l284_284010


namespace max_value_omega_is_eleven_l284_284053

noncomputable def max_omega (ω φ : ℝ) (f : ℝ → ℝ) : ℝ :=
  if H : (∀ x, f x = sin (ω * x + φ)) ∧ (ω > 0) ∧ (|φ| ≤ π / 2) ∧ (∀ x, y(x - π / 4) = sin (ω * (x - π / 4) + φ) = sin(ω*x + φ - ω*π/4) ∧ odd(y)) ∧ (∀ x, x = π / 4 -> axis_of_symmetry(y)) ∧ 
          monotonic_on f (Ioo (π / 14) (13 * π / 84)) 
  then 11 
  else 0

theorem max_value_omega_is_eleven : max_omega ω φ f = 11 :=
  sorry

end max_value_omega_is_eleven_l284_284053


namespace total_wheels_l284_284989

theorem total_wheels (bicycles : ℕ) (tricycles : ℕ) (wheels_per_bicycle : ℕ) (wheels_per_tricycle : ℕ) :
  bicycles = 16 → tricycles = 7 → wheels_per_bicycle = 2 → wheels_per_tricycle = 3 →
  2 * bicycles + 3 * tricycles = 53 :=
by
  intros hb ht hwb hwt
  have h1 : 2 * bicycles = 2 * 16 := by rw [hb]
  have h2 : 3 * tricycles = 3 * 7 := by rw [ht]
  have h3 : 2 * 16 + 3 * 7 = 32 + 21 := by norm_num
  have h4 : 32 + 21 = 53 := by norm_num
  rw [h1, h2, h3, h4]
  sorry

end total_wheels_l284_284989


namespace xyz_value_l284_284030

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 :=
by
  sorry

end xyz_value_l284_284030


namespace median_free_throws_l284_284645

open List

-- Define the sequence of free throws
def free_throws : List ℕ := [6, 20, 15, 14, 19, 12, 19, 15, 25, 10]

-- Define the sorted sequence
def sorted_free_throws : List ℕ := [6, 10, 12, 14, 15, 15, 19, 19, 20, 25]

-- Define the median calculation for even length lists
def median_even (l : List ℕ) (h : l.length % 2 = 0) : ℕ :=
  let mid := l.length / 2
  (l.get! (mid - 1) + l.get! mid) / 2

-- The property we are going to prove
theorem median_free_throws : median_even (sorted_list free_throws) (by simp [List.length, free_throws]) = 15 := by
  sorry

end median_free_throws_l284_284645


namespace total_dolls_l284_284164

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l284_284164


namespace Kirill_is_69_l284_284876

/-- Kirill is 14 centimeters shorter than his brother.
    Their sister's height is twice the height of Kirill.
    Their cousin's height is 3 centimeters more than the sister's height.
    Together, their heights equal 432 centimeters.
    We aim to prove that Kirill's height is 69 centimeters.
-/
def Kirill_height (K : ℕ) : Prop :=
  let brother_height := K + 14
  let sister_height := 2 * K
  let cousin_height := 2 * K + 3
  K + brother_height + sister_height + cousin_height = 432

theorem Kirill_is_69 {K : ℕ} (h : Kirill_height K) : K = 69 :=
by
  sorry

end Kirill_is_69_l284_284876


namespace complex_division_l284_284046

theorem complex_division (z : ℂ) (hz : (3 + 4 * I) * z = 25) : z = 3 - 4 * I :=
sorry

end complex_division_l284_284046


namespace average_of_values_of_x_l284_284820

example (x : ℝ) (h : sqrt (3 * x^2 + 2) = sqrt 50) : x = 4 ∨ x = -4 := by
  sorry

theorem average_of_values_of_x (x : ℝ) (hx: sqrt (3 * x^2 + 2) = sqrt 50) :
    (4 + (-4)) / 2 = 0 := by
  have h1 : 3 * x^2 + 2 = 50 := by
    exact (congrArg (fun x => x^2) hx)
  have h2 : 3 * x^2 = 48 := by
    linarith
  have h3 : x^2 = 16 := by
    linarith
  have h4 : x = 4 ∨ x = -4 := by
    apply sqrt_eq_iff_sq_eq'
    linarith
  have h5 : (4 + -4) / 2 = 0 := by
    ring
  exact h5

end average_of_values_of_x_l284_284820


namespace solution_set_inequality_l284_284055

noncomputable def f (x : ℝ) : ℝ := |x| + |x - 4|

theorem solution_set_inequality :
  { x : ℝ | f (x^2 + 2) > f x } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > real.sqrt 2 } :=
by
  sorry

end solution_set_inequality_l284_284055


namespace product_of_divisors_eq_l284_284346

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284346


namespace proof_x_squared_plus_y_squared_l284_284447

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l284_284447


namespace angle_EKB_right_angle_l284_284166

variables {A B C D E K : Type*}
variables {S : Type*} [IsCircumscribed A B C D E S]
variables [denote_point K B C]
variables [equal_sides A B C D]

theorem angle_EKB_right_angle 
  (h1 : IsPentagon A B C D E) 
  (h2 : InscribedCircle S A B C D E)
  (h3 : TangentAt K B C S) 
  (h4 : SideEqual A B C D) : 
  ∠EKB = 90 :=
sorry

end angle_EKB_right_angle_l284_284166


namespace mary_initial_sugar_eq_4_l284_284529

/-- Mary is baking a cake. The recipe calls for 7 cups of sugar and she needs to add 3 more cups of sugar. -/
def total_sugar : ℕ := 7
def additional_sugar : ℕ := 3

theorem mary_initial_sugar_eq_4 :
  ∃ initial_sugar : ℕ, initial_sugar + additional_sugar = total_sugar ∧ initial_sugar = 4 :=
sorry

end mary_initial_sugar_eq_4_l284_284529


namespace tetrahedron_centroid_intersections_l284_284542

-- Define the centroid of a tetrahedron
def centroid (A B C D : Point) : Point := sorry

-- Define the centroid of a triangle
def triangle_centroid (A B C : Point) : Point := sorry

-- Define the midpoint of a segment
def midpoint (A B : Point) : Point := sorry

-- Theorem statements
theorem tetrahedron_centroid_intersections (A B C D : Point) :
  let G := centroid A B C D,
      G_ABC := triangle_centroid A B C,
      G_ABD := triangle_centroid A B D,
      G_ACD := triangle_centroid A C D,
      G_BCD := triangle_centroid B C D,
      M_AB := midpoint A B,
      M_CD := midpoint C D,
      M_AC := midpoint A C,
      M_BD := midpoint B D,
      M_AD := midpoint A D,
      M_BC := midpoint B C in
  MeasurableSpace.IsInstance.is_measurable
    (Line.intersection (line_through A G_BCD) (line_through B G_ACD))
    = G ∧
  Line.segment_ratio (line_segment A G_BCD) G 3 1 ∧
  Line.segment_ratio (line_segment M_AB M_CD) G 1 1 := sorry

end tetrahedron_centroid_intersections_l284_284542


namespace positive_integers_log_b_l284_284067

theorem positive_integers_log_b (n : ℕ) (b : ℕ) (h1 : b ^ n = 216) (h2 : n > 0) : 
    {b : ℕ | b ^ n = 216 ∧ n > 0}.card = 4 :=
sorry

end positive_integers_log_b_l284_284067


namespace tic_tac_toe_tie_fraction_l284_284609

theorem tic_tac_toe_tie_fraction
  (max_wins : ℚ := 4 / 9)
  (zoe_wins : ℚ := 5 / 12) :
  1 - (max_wins + zoe_wins) = 5 / 36 :=
by
  sorry

end tic_tac_toe_tie_fraction_l284_284609


namespace quadratic_condition_l284_284569

theorem quadratic_condition (a : Real) : (∃ x : Real, a * x^2 - x + 2 = 0) → a ≠ 0 :=
begin
  sorry
end

end quadratic_condition_l284_284569


namespace min_distance_l284_284031

variables {P Q : ℝ × ℝ}

def line (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2) ^ 2 + (Q.2 - 2) ^ 2 = 4

theorem min_distance (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : line P) (hQ : circle Q) :
  ∃ d : ℝ, d = dist P Q ∧ d = 9 / 5 := sorry

end min_distance_l284_284031


namespace smallest_possible_sum_of_products_l284_284706

theorem smallest_possible_sum_of_products :
  ∃ (b : Fin 100 → ℤ),
    (∀ i, b i = 1 ∨ b i = -1) ∧
    let T := ∑ i in Finset.range 100, ∑ j in Finset.range 100, if i < j then b i * b j else 0 in
    T = 22 := by
  sorry

end smallest_possible_sum_of_products_l284_284706


namespace specify_waist_size_l284_284466

noncomputable def waist_size_in_centimeters (inches : ℝ) (feet_per_inch : ℝ) (cm_per_foot : ℝ) : ℝ :=
  (inches * feet_per_inch * cm_per_foot).round / 1

theorem specify_waist_size (inches : ℝ) (feet_per_inch : ℝ) (cm_per_foot : ℝ) (cm : ℝ) :
  inches = 28 → feet_per_inch = 1 / 10 → cm_per_foot = 25.4 → cm = 71.1 → 
  waist_size_in_centimeters inches feet_per_inch cm_per_foot = cm :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  norm_num
  assumption

end specify_waist_size_l284_284466


namespace five_distinct_solutions_l284_284571

noncomputable def f (x : ℝ) : ℝ :=
if x = 2 then 1 else log (|x - 2|)

theorem five_distinct_solutions 
  (b c : ℝ)
  (h1 : 1 + b + c = 0)
  (h2 : ∀ (x : ℝ), x ≠ 2 → (log (|x - 2|)^2 + b * log (|x - 2|) + c = 0))
  (h_distinct : ∃ (x1 x2 x3 x4 x5 : ℝ), 
      x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ 
      x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧
      x3 ≠ x4 ∧ x3 ≠ x5 ∧
      x4 ≠ x5 ∧
      f x1 = 1 ∧ f x2 = log 10 ∧ f x3 = log b ∧ f x4 = log 1 ∧ f x5 = log b) :
  f (x1 + x2 + x3 + x4 + x5) = 3 * log 2 :=
sorry

end five_distinct_solutions_l284_284571


namespace can_equalize_tea_in_16_cups_l284_284991

-- Define the conditions
def cups_of_tea (n : ℕ) : Prop :=
  ∃ f : (fin n → ℝ) → (fin n → ℝ), ∀ x : fin n → ℝ,
  (∀ i j : fin n, f x i = f x j)

-- Define the statement that Masha can equalize the tea in 16 cups
theorem can_equalize_tea_in_16_cups : cups_of_tea 16 :=
sorry

end can_equalize_tea_in_16_cups_l284_284991


namespace frac_wx_l284_284814

theorem frac_wx (x y z w : ℚ) (h1 : x / y = 5) (h2 : y / z = 1 / 2) (h3 : z / w = 7) : w / x = 2 / 35 :=
by
  sorry

end frac_wx_l284_284814


namespace baseball_cards_per_friend_l284_284554

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l284_284554


namespace cone_lateral_area_l284_284773

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end cone_lateral_area_l284_284773


namespace sum_of_B_elements_l284_284150

def A : Set Int := {2, 0, 1, 3}
def B : Set Int := {x | -x ∈ A ∧ ¬(2 - x^2 ∈ A)}

theorem sum_of_B_elements : ∑ x in B, x = -5 :=
by
  sorry

end sum_of_B_elements_l284_284150


namespace find_k_find_ab_l284_284209

-- Defining the complex numbers u and v
def u : ℂ := -3 + 4 * complex.I
def v : ℂ := 2 + 2 * complex.I

-- The conditions
def a : ℂ := 5 + 2 * complex.I
def b : ℂ := 5 - 2 * complex.I
def k : ℝ := 23

-- The proof problem statements
theorem find_k : k = 23 := by sorry

theorem find_ab : a * b = 29 := by sorry

end find_k_find_ab_l284_284209


namespace B_subset_A_iff_l284_284804

namespace MathProofs

def A (x : ℝ) : Prop := -2 < x ∧ x < 5

def B (x : ℝ) (m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem B_subset_A_iff (m : ℝ) :
  (∀ x : ℝ, B x m → A x) ↔ m < 3 :=
by
  sorry

end MathProofs

end B_subset_A_iff_l284_284804


namespace coefficient_x2_in_expansion_l284_284320

theorem coefficient_x2_in_expansion : 
  (∑ (n : ℕ) in Finset.range 6, (Nat.choose (n + 3) 2)) = 83 :=
by {
  sorry
}

end coefficient_x2_in_expansion_l284_284320


namespace complex_number_imaginary_axis_l284_284782

theorem complex_number_imaginary_axis (a : ℂ) (h : (1 + complex.i) * (2 * a - complex.i) = 0 + complex.i * (-2)) :
  (1 + complex.i) * (2 * a - complex.i) = -2 * complex.i :=
by {
  sorry
}

end complex_number_imaginary_axis_l284_284782


namespace loaned_books_count_l284_284254

variable (x : ℕ) -- x is the number of books loaned out during the month

theorem loaned_books_count 
  (initial_books : ℕ) (returned_percentage : ℚ) (remaining_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : returned_percentage = 0.80)
  (h3 : remaining_books = 66) :
  x = 45 :=
by
  -- Proof can be inserted here
  sorry

end loaned_books_count_l284_284254


namespace range_of_a_l284_284779

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (if x1 < 1 then (2 - a) * x1 + 1 else a ^ x1) - (if x2 < 1 then (2 - a) * x2 + 1 else a ^ x2) > 0 → (x1 - x2) > 0) →
  (real.sqrt (1.5) <= a ∧ a < 2) :=
by
  sorry

end range_of_a_l284_284779


namespace solve_inequality_l284_284339

noncomputable def valid_x_values : set ℝ :=
  {x | x ∈ set.Icc 3.790 5 \ set.Icc 5 5 ∪ set.Icc 5 7.067}

theorem solve_inequality (x : ℝ) :
  (x ∈ valid_x_values) ↔ ((x * (x + 2) / (x - 5) ^ 2) ≥ 15) :=
sorry

end solve_inequality_l284_284339


namespace solid_is_cone_if_views_are_isosceles_triangle_l284_284086

def is_isosceles_triangle (t : Type) : Prop := sorry -- Define isosceles triangle condition.

def is_front_view_isosceles_triangle (t : Type) : Prop := sorry
def is_side_view_isosceles_triangle (t : Type) : Prop := sorry

def is_cone (s : Type) : Prop := sorry
def is_cylinder (s : Type) : Prop := sorry
def is_sphere (s : Type) : Prop := sorry
def is_frustum (s : Type) : Prop := sorry

theorem solid_is_cone_if_views_are_isosceles_triangle (s : Type) :
  (is_front_view_isosceles_triangle s) ∧ (is_side_view_isosceles_triangle s) →
  (is_cone s) :=
begin
  sorry
end

end solid_is_cone_if_views_are_isosceles_triangle_l284_284086


namespace tan_B_l284_284477

noncomputable def tan_of_B (A B C : Type*) [metric_space A] (ABC : right_triangle A B C) (h : ∠BAC = 90°)
  (AB AC : ℝ) (hAB : AB = 8) (hAC : AC = 17) : ℝ :=
BC / AB
  where BC = sqrt (AC^2 - AB^2)
  by
    sorry

theorem tan_B (A B C : Type*) [metric_space A] (ABC : right_triangle A B C)
  (h : ∠BAC = 90°) (hAB : dist A B = 8) (hAC : dist A C = 17) : tan_of_B A B C ABC h hAB hAC = 15 / 8 :=
by
  sorry

end tan_B_l284_284477


namespace decimal_digits_right_of_decimal_place_l284_284083

theorem decimal_digits_right_of_decimal_place (x : ℝ) (h : x = 3.456789) :
  let y := (10 ^ 4 * x) ^ 9 in
  num_digits_right_of_decimal_place y = 6 :=
by
  sorry

end decimal_digits_right_of_decimal_place_l284_284083


namespace measure_of_stability_is_variance_l284_284606

axiom sample_variance_measures_fluctuation_size : ∀ (x : Set ℝ), variance x = measure_fluctuation_size x

axiom smaller_variance_more_stable : ∀ (x y : Set ℝ), (variance x < variance y) → (stability x > stability y)

axiom larger_variance_less_stable : ∀ (x y : Set ℝ), (variance x > variance y) → (stability x < stability y)

theorem measure_of_stability_is_variance (A B : Set ℝ) (h : stability A ≠ stability B) : measure_of_stability = variance :=
by
  sorry

end measure_of_stability_is_variance_l284_284606


namespace find_other_endpoint_l284_284311

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_endpoint (O A B : ℝ × ℝ) 
    (hO : O = (2, 3)) 
    (hA : A = (-1, -1))
    (hMidpoint : O = midpoint A B) : 
    B = (5, 7) :=
sorry

end find_other_endpoint_l284_284311


namespace range_of_a_monotone_l284_284056

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem range_of_a_monotone (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-real.sqrt 2 ≤ a ∧ a ≤ real.sqrt 2) := 
by
  sorry

end range_of_a_monotone_l284_284056


namespace final_card_value_count_l284_284252

theorem final_card_value_count : ∀ (n : ℕ), 
  let m := 1002 in
  n = 2004 → 
  ∃ c_total,
    c_total = 3^n - 2 * 3^m + 2 := by
  intros
  let m := 1002
  let n := 2004
  have h1 : n = 2004 := by rfl
  use (3^n - 2 * 3^m + 2)
  exact ⟨rfl⟩

end final_card_value_count_l284_284252


namespace enrollment_difference_l284_284240

theorem enrollment_difference :
  let Varsity := 1680
  let Northwest := 1170
  let Central := 1840
  let Greenbriar := 1090
  let Eastside := 1450
  Central - Greenbriar = 750 := 
by
  intros Varsity Northwest Central Greenbriar Eastside
  -- calculate the difference
  have h1 : 750 = 750 := rfl
  sorry

end enrollment_difference_l284_284240


namespace imag_part_of_complex_squared_is_2_l284_284978

-- Define the complex number 1 + i
def complex_num := (1 : ℂ) + (Complex.I : ℂ)

-- Define the squared value of the complex number
def complex_squared := complex_num ^ 2

-- Define the imaginary part of the squared value
def imag_part := complex_squared.im

-- State the theorem
theorem imag_part_of_complex_squared_is_2 : imag_part = 2 := sorry

end imag_part_of_complex_squared_is_2_l284_284978


namespace product_of_divisors_of_72_l284_284357

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284357


namespace num_solutions_abs_eq_l284_284043

theorem num_solutions_abs_eq (B : ℤ) (hB : B = 3) : 
  { x : ℤ | |x - 2| + |x + 1| = B }.finite.to_finset.card = 4 :=
by
  sorry

end num_solutions_abs_eq_l284_284043


namespace total_dolls_l284_284165

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l284_284165


namespace count_ordered_triples_lcm_l284_284131

theorem count_ordered_triples_lcm :
  { (a b c : ℕ) // 
    Nat.lcm a b = 2000 ∧ 
    Nat.lcm b c = 4000 ∧ 
    Nat.lcm c a = 4000 ∧ 
    Nat.lcm (Nat.lcm a b) c = 12000 }.card = 240 :=
sorry

end count_ordered_triples_lcm_l284_284131


namespace S_n_lt_6_l284_284737

-- Conditions
def f : ℕ+ → ℕ+ → ℤ
| ⟨1, _⟩, ⟨1, _⟩ => 1
| ⟨m + 1, hm⟩, n => f ⟨m, nat.succ_pos' m⟩ n + 2 * (m + n)
| m, ⟨n + 1, hn⟩ => f m ⟨n, nat.succ_pos' n⟩ + 2 * (m + n - 1)

def a_n (n : ℕ+) : ℚ := (f n n).sqrt / (2:ℚ)^(n - 1)

def S (n : ℕ) : ℚ := (finset.range n).sum (λ k => a_n ⟨k + 1, nat.succ_pos' k⟩)

-- To Prove
theorem S_n_lt_6 (n: ℕ) : S n < 6 := by
  sorry

end S_n_lt_6_l284_284737


namespace problem_x2_plus_y2_l284_284454

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l284_284454


namespace club_officers_selection_l284_284272

noncomputable def number_of_ways_to_select_officers (total_members : ℕ) (special_members : ℕ) : ℕ :=
let remaining_members := total_members - special_members in
let scenario1 := remaining_members * (remaining_members - 1) * (remaining_members - 2) in
let scenario2 := (3.choose 2) * 2 * remaining_members in
scenario1 + scenario2

theorem club_officers_selection :
  number_of_ways_to_select_officers 25 3 = 9372 :=
by sorry

end club_officers_selection_l284_284272


namespace find_a_l284_284574

-- Define the function transformation and the given conditions for the triangle
def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
axiom a_le_c (a c : ℝ) : a < c
axiom Delta_ABC_area (a b c : ℝ) : 1 / 2 * b * c * Real.sin (Real.pi / 4) = 2
axiom b_value : b = 2
axiom g_A_value (A : ℝ) : g A = -1 / 2

-- The goal is to show that a = 2 under these given conditions
theorem find_a (A a b c : ℝ) (h1 : a_le_c a c) (h2 : Delta_ABC_area a b c)
  (h3 : b_value = b) (h4 : g_A_value A) : a = 2 := by
  sorry

end find_a_l284_284574


namespace arithmetic_sequence_geometric_sum_l284_284893

theorem arithmetic_sequence_geometric_sum (a1 : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), S 1 = a1)
  (h2 : ∀ (n : ℕ), S 2 = 2 * a1 - 1)
  (h3 : ∀ (n : ℕ), S 4 = 4 * a1 - 6)
  (h4 : (2 * a1 - 1)^2 = a1 * (4 * a1 - 6)) 
  : a1 = -1/2 := 
sorry

end arithmetic_sequence_geometric_sum_l284_284893


namespace triangle_area_1420_l284_284691

noncomputable def area_of_triangle (a b c : ℝ) := (sqrt 3 / 4) * a * b

theorem triangle_area_1420 :
  ∀ (ω1 ω2 ω3 : Circle) (P1 P2 P3 : Point),
    (∀ (i j : nat), i ≠ j → Tangent ωi ωj) →
    (∀ (i : nat), radius ωi = 5) →
    (EquilateralTriangle P1 P2 P3) →
    (∀ i : nat, Tangent (ωi) (LineSegment P[i] P[(i + 1) % 3])) →
    area_of_triangle (distance P1 P2) (distance P1 P2) (distance P1 P2) = sqrt 675 + sqrt 745 :=
by
  intros ω1 ω2 ω3 P1 P2 P3 hTangent hRadius hEquilateral hTangentSegments
  sorry

end triangle_area_1420_l284_284691


namespace range_of_k_l284_284992

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + k^2 - 1 ≤ 0) ↔ (-Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2) :=
by 
  sorry

end range_of_k_l284_284992


namespace distance_between_foci_of_ellipse_l284_284342

theorem distance_between_foci_of_ellipse :
  let a := √16
  let b := √4
  let c := √(a^2 - b^2)
  2 * c = 4 * √3 :=
by
  let a := (16 : ℝ)^(1/2) -- a^2 = 16
  let b := (4 : ℝ)^(1/2)  -- b^2 = 4
  let c := (a^2 - b^2)^(1/2) -- c = √(a^2 - b^2)
  have : 2 * c = 4 * (3 : ℝ)^(1/2), from sorry
  exact this

end distance_between_foci_of_ellipse_l284_284342


namespace sum_of_first_16_terms_l284_284856

def sequence_sum_condition (a : ℕ → ℤ) :=
  ∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = 2 * n - 1

theorem sum_of_first_16_terms (a : ℕ → ℤ) (h : sequence_sum_condition a) :
  (∑ i in Finset.range 16, a i) = 136 :=
  by
  sorry

end sum_of_first_16_terms_l284_284856


namespace borrowed_years_l284_284530

noncomputable def principal : ℝ := 5396.103896103896
noncomputable def interest_rate : ℝ := 0.06
noncomputable def total_returned : ℝ := 8310

theorem borrowed_years :
  ∃ t : ℝ, (total_returned - principal) = principal * interest_rate * t ∧ t = 9 :=
by
  sorry

end borrowed_years_l284_284530


namespace max_abs_a_plus_abs_b_l284_284926

open Complex

-- Define the polynomial f(z)
def polynomial (a b c : ℂ) (z : ℂ) : ℂ := a * z^2 + b * z + c

-- Define the condition on the polynomial
def condition (f : ℂ → ℂ) : Prop := ∀ z : ℂ, abs z ≤ 1 → abs (f z) ≤ 1

-- The statement to prove
theorem max_abs_a_plus_abs_b (a b c: ℂ) (h : condition (polynomial a b c)) : 
  abs a + abs b ≤ (2 * sqrt 3) / 3 := 
sorry

end max_abs_a_plus_abs_b_l284_284926


namespace forty_seventh_digit_of_sequence_from_90_to_41_l284_284826

-- defining the sequence
def digitSequence : List Nat := (List.range' 41 50).reverse >>= (λ n, n.digits 10).reverse

-- defining what we're trying to prove
theorem forty_seventh_digit_of_sequence_from_90_to_41 : digitSequence.nth 46 = some 6 := 
sorry

end forty_seventh_digit_of_sequence_from_90_to_41_l284_284826


namespace area_of_regions_II_and_III_l284_284543

theorem area_of_regions_II_and_III
  (ABCD_is_rectangle : is_rectangle ABCD)
  (AB_eq_3 : AB = 3)
  (BC_eq_4 : BC = 4)
  (AD_BD_eq_5 : AD = 5 ∧ BD = 5)
  (circle_D_radius_5_arcs_AEC : circle_with_center_radius_subtends_arc D 5 AEC)
  (circle_B_radius_5_arcs_AFC : circle_with_center_radius_subtends_arc B 5 AFC) :
  total_area_regions_II_III ABCD = (25 * Real.pi) / 4 - 3 :=
sorry

end area_of_regions_II_and_III_l284_284543


namespace unique_a_zero_l284_284716

theorem unique_a_zero (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + f(y)) = f(x) + a * ⌊y⌋₊) → a = 0 :=
sorry

end unique_a_zero_l284_284716


namespace parallel_lines_slope_equal_l284_284829

theorem parallel_lines_slope_equal (m : ℝ) : 
  (∃ m : ℝ, -(m+4)/(m+2) = -(m+2)/(m+1)) → m = 0 := 
by
  sorry

end parallel_lines_slope_equal_l284_284829


namespace product_of_even_and_odd_is_odd_l284_284526

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def odd_product (f g : ℝ → ℝ) : Prop := ∀ x, (f x) * (g x) = - (f x) * (g x)
 
theorem product_of_even_and_odd_is_odd 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : odd_function g) : odd_product f g :=
by
  sorry

end product_of_even_and_odd_is_odd_l284_284526


namespace max_k_for_3_pow_11_as_sum_of_consec_integers_l284_284726

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end max_k_for_3_pow_11_as_sum_of_consec_integers_l284_284726


namespace millionth_digit_of_fraction_l284_284731

theorem millionth_digit_of_fraction (n : ℕ) (hn : n = 1000000) : 
  ∃ (d : ℕ), d = 7 ∧ (decimal_expansion (3 / 41) (hn - 1) = d) :=
sorry

end millionth_digit_of_fraction_l284_284731


namespace product_of_divisors_of_72_l284_284356

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284356


namespace p_p_composition_l284_284493

def p (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 then x + 2*y
  else if x < 0 ∧ y < 0 then x - 3*y
  else if x ≥ 0 ∧ y ≤ 0 then 4*x + 2*y
  else x^2 + y^2

theorem p_p_composition: p (p 2 (-2)) (p (-3) (-1)) = 16 :=
by
  sorry

end p_p_composition_l284_284493


namespace equal_segments_CA_CB_l284_284494

theorem equal_segments_CA_CB
  (A B C M D I1 I2 : Type)
  [NaturallyInhabited A]
  [NaturallyInhabited B]
  [NaturallyInhabited C]
  [NaturallyInhabited M]
  [NaturallyInhabited D]
  [NaturallyInhabited I1]
  [NaturallyInhabited I2]
  (triangle_ABC : Triangle A B C)
  (midpoint_M : Midpoint M A B)
  (point_on_MB : OnSegment D M B)
  (incenter_I1_ADC : Incenter I1 (Triangle A D C))
  (incenter_I2_BDC : Incenter I2 (Triangle B D C))
  (right_angle_I1MI2 : ∠ I1 M I2 = 90) :
  SegmentLength C A = SegmentLength C B :=
by
  sorry

end equal_segments_CA_CB_l284_284494


namespace minimum_value_if_incenter_l284_284741

-- Define the types for points, distances, and triangles
variables {Point : Type} 

structure Triangle :=
(A B C : Point)

structure Perpendiculars :=
(M A1 B1 C1 : Point)

variables (triangle : Triangle)
variables (perpendiculars : Perpendiculars)

variables (a b c : ℝ)
variables (MA1 MB1 MC1 : ℝ)

-- Definitions based on given conditions
def is_incenter (M : Point) (triangle : Triangle) : Prop :=
  ∃ incenter, incenter = M

-- Main theorem with minimum condition proof
theorem minimum_value_if_incenter (M : Point) (hM : ∃A1 B1 C1, perpendiculars = ⟨M, A1, B1, C1⟩)
  (h_MA1 : MA1 = dist M (line (triangle.B, triangle.C)))
  (h_MB1 : MB1 = dist M (line (triangle.C, triangle.A)))
  (h_MC1 : MC1 = dist M (line (triangle.A, triangle.B))) :
  (∃x y z, x = MA1 ∧ y = MB1 ∧ z = MC1 ∧
    (a / x + b / y + c / z) = (a + b + c) / (x + y + z)) ↔ is_incenter M triangle :=
sorry

end minimum_value_if_incenter_l284_284741


namespace sharon_trip_distance_l284_284704

noncomputable section

variable (x : ℝ)

def sharon_original_speed (x : ℝ) := x / 200

def sharon_reduced_speed (x : ℝ) := (x / 200) - 1 / 2

def time_before_traffic (x : ℝ) := (x / 2) / (sharon_original_speed x)

def time_after_traffic (x : ℝ) := (x / 2) / (sharon_reduced_speed x)

theorem sharon_trip_distance : 
  (time_before_traffic x) + (time_after_traffic x) = 300 → x = 200 := 
by
  sorry

end sharon_trip_distance_l284_284704


namespace sequences_properties_l284_284521

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

noncomputable def A₁ (a b : ℝ) := arithmetic_mean a b
noncomputable def G₁ (a b : ℝ) := geometric_mean a b
noncomputable def H₁ (a b : ℝ) := harmonic_mean a b

noncomputable def A (a b : ℝ) : ℕ → ℝ
| 1     := A₁ a b
| (n+1) := arithmetic_mean (G n) (H n)

noncomputable def G (a b : ℝ) : ℕ → ℝ
| 1     := G₁ a b
| (n+1) := geometric_mean (G n) (H n)

noncomputable def H (a b : ℝ) : ℕ → ℝ
| 1     := H₁ a b
| (n+1) := harmonic_mean (G n) (H n)

theorem sequences_properties {a b : ℝ} (h : a ≠ b) (ha : a > 0) (hb : b > 0) :
  (∀ n : ℕ, 1 ≤ n → G a b n = G a b 1) ∧
  (∀ n : ℕ, 1 ≤ n → H a b n < H a b (n + 1)) ∧
  (16 + 256 = 272) :=
begin
  sorry
end

end sequences_properties_l284_284521


namespace statue_model_representation_l284_284215

theorem statue_model_representation
  (S : ℝ) (M : ℝ) (conversion_factor : ℝ)
  (hS : S = 80)
  (hM : M = 10)
  (h_conv : conversion_factor = 2.54) :
  (S / (M / conversion_factor)) / conversion_factor ≈ 6.562 := by
  sorry

end statue_model_representation_l284_284215


namespace scallop_cost_l284_284641

def scallop_price_proof : Prop :=
  ∃ (cost_per_pound : ℝ), 
    let number_of_people := 8 in
    let scallops_per_person := 2 in
    let total_scallops := number_of_people * scallops_per_person in
    let scallops_per_pound := 8 in
    let pound_cost := 48 in
    total_scallops / scallops_per_pound * cost_per_pound = pound_cost ∧
    cost_per_pound = 24

theorem scallop_cost : scallop_price_proof :=
  sorry

end scallop_cost_l284_284641


namespace A_inter_B_A_subset_C_l284_284744

namespace MathProof

def A := {x : ℝ | x^2 - 6*x + 8 ≤ 0 }
def B := {x : ℝ | (x - 1)/(x - 3) ≥ 0 }
def C (a : ℝ) := {x : ℝ | x^2 - (2*a + 4)*x + a^2 + 4*a ≤ 0 }

theorem A_inter_B : (A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 4} := sorry

theorem A_subset_C (a : ℝ) : (A ⊆ C a) ↔ (0 ≤ a ∧ a ≤ 2) := sorry

end MathProof

end A_inter_B_A_subset_C_l284_284744


namespace fencing_required_l284_284256

-- Define the conditions
variables (L W A : ℝ)
axiom length_condition : L = 30
axiom area_condition : A = L * W
axiom area_value : A = 600

-- Define the statement to be proved
theorem fencing_required : 
  let W := 600 / 30 in
  2 * W + L = 70 := 
by 
  -- Proof is skipped
  sorry

end fencing_required_l284_284256


namespace zero_of_function_in_interval_l284_284599

theorem zero_of_function_in_interval (m : ℕ)
  (h1 : m = 2)
  (f : ℝ → ℝ)
  (h2 : ∀ x, f x = 2^x - 5) :
  ∃ x, x ∈ Icc (m:ℝ) (m + 1) ∧ f x = 0 :=
by
  use 2.5  -- example element in interval [2,3]
  simp [h1, h2]
  split
  { norm_cast, linarith }
  { apply exists_eq_norm }
  sorry

end zero_of_function_in_interval_l284_284599


namespace rectangle_area_error_percent_l284_284844

theorem rectangle_area_error_percent (L W : ℝ) :
  let measured_length := 1.05 * L,
      measured_width := 0.96 * W,
      actual_area := L * W,
      measured_area := measured_length * measured_width,
      error := measured_area - actual_area in
  (error / actual_area) * 100 = 0.8 := 
by
  sorry

end rectangle_area_error_percent_l284_284844


namespace min_value_problem_l284_284908

theorem min_value_problem (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2 = 4 * (Real.root 4 5 - 1)^2 := 
sorry

end min_value_problem_l284_284908


namespace proof_x_squared_plus_y_squared_l284_284449

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l284_284449


namespace range_of_t_for_monotonicity_l284_284790

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  (∀ x y ∈ s, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ s, x ≤ y → f x ≥ f y)

theorem range_of_t_for_monotonicity :
  ∀ t : ℝ, t > -2 → (is_monotonic_on f (Set.Icc (-2) t) ↔ t ∈ Set.Ioo (-2:ℝ) (0:ℝ) ∨ t ∈ Set.Icc (-2:ℝ) 0) :=
by sorry

end range_of_t_for_monotonicity_l284_284790


namespace twenty_second_entry_l284_284736

-- Definition of r_9 which is the remainder left when n is divided by 9
def r_9 (n : ℕ) : ℕ := n % 9

-- Statement to prove that the 22nd entry in the ordered list of all nonnegative integers
-- that satisfy r_9(5n) ≤ 4 is 38
theorem twenty_second_entry (n : ℕ) (hn : 5 * n % 9 ≤ 4) :
  ∃ m : ℕ, m = 22 ∧ n = 38 :=
sorry

end twenty_second_entry_l284_284736


namespace max_safe_knights_on_black_squares_l284_284690

-- Define the chessboard and conditions
def chessboard := fin 8 × fin 8

def is_black_square (pos : chessboard) : Prop := 
  (pos.1 + pos.2) % 2 = 1

def knight_attacks (pos : chessboard) (attacking_pos : chessboard) : Prop :=
  (pos.1 + 2 == attacking_pos.1 ˅ pos.1 - 2 == attacking_pos.1) ˄
  (pos.2 + 1 == attacking_pos.2 ˅ pos.2 - 1 == attacking_pos.2) ˅
  (pos.1 + 1 == attacking_pos.1 ˅ pos.1 - 1 == attacking_pos.1) ˄
  (pos.2 + 2 == attacking_pos.2 ˅ pos.2 - 2 == attacking_pos.2)

-- Proving the maximum number of skew knights that can be placed such that they do not attack each other
theorem max_safe_knights_on_black_squares : ∃ (knight_positions : set chessboard), 
  (∀ k1 ∈ knight_positions, ∀ k2 ∈ knight_positions, k1 ≠ k2 → ¬knight_attacks k1 k2) ∧
  (∀ k ∈ knight_positions, is_black_square k) ∧ 
  knight_positions.card = 32 :=
sorry -- skipping the proof

end max_safe_knights_on_black_squares_l284_284690


namespace sequence_sum_l284_284051

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then n * n else -(n * n)

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

def sum_seq (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun k => a (k + 1))

theorem sequence_sum : sum_seq 50 = 50 := 
  sorry

end sequence_sum_l284_284051


namespace tan_beta_given_alpha_max_tan_beta_l284_284026

noncomputable def alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) := α
noncomputable def beta (β : ℝ) (hβ : 0 < β ∧ β < π / 2) := β

theorem tan_beta_given_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : (sin β / sin α) = cos (α + β)) (hα_val : α = π / 6) : tan β = sqrt 3 / 5 := sorry

theorem max_tan_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : (sin β / sin α) = cos (α + β)) : ∃ (t : ℝ), t = sqrt 2 / 4 ∧ ∀ β', tan β' ≤ t := sorry

end tan_beta_given_alpha_max_tan_beta_l284_284026


namespace quadratic_intersect_y_axis_l284_284416

theorem quadratic_intersect_y_axis (m : ℝ) :
  ∃ m : ℝ, (∃ y : ℝ, y = m^2 - 2*m - 3 ∧ y = 0) →
  (m = -1 ∨ m = 3) :=
begin
  sorry
end

end quadratic_intersect_y_axis_l284_284416


namespace min_income_in_third_year_income_exceeds_initial_l284_284848

-- General formula for a_n
def a_n (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a else a * (2/3)^(n-1) + b * (3/2)^(n-2)

-- Minimum income occurs in the third year when b = 8a/27
theorem min_income_in_third_year (a : ℝ) :
  ∃ n : ℕ, ∀ (b : ℝ), b = (8 * a) / 27 → 
  (∀ m : ℕ, m ≠ 3 → a_n a b 3 ≤ a_n a b m) ∧ a_n a b 3 = (8 * a) / 9 :=
sorry

-- Income after one year of transfer exceeds the initial income if b ≥ 3a/8
theorem income_exceeds_initial (a b : ℝ) (n : ℕ) : 
  b ≥ (3 * a) / 8 → n ≥ 2 → a_n a b n > a :=
sorry

end min_income_in_third_year_income_exceeds_initial_l284_284848


namespace minimum_ratio_l284_284899

theorem minimum_ratio (x y : Int) (h1 : 10 ≤ x) (h2 : x ≤ 150) (h3 : 10 ≤ y) (h4 : y ≤ 150)
  (h5 : x + y = 150) : x * 14 ≤ y :=
begin
  sorry
end

end minimum_ratio_l284_284899


namespace karens_speed_l284_284122

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l284_284122


namespace is_isosceles_triangle_l284_284093

theorem is_isosceles_triangle (A B C : ℝ) (h : 2 * cos B * sin A = sin C) : 
  is_isosceles_triangle A B C := 
sorry

end is_isosceles_triangle_l284_284093


namespace segments_perpendicular_l284_284971

-- Definitions: Circle, points A, B, C, D on the circumference, midpoints M1, M2, M3, M4
variables {C : Type*} {O : C} {r : ℝ}
noncomputable def A : C := sorry
noncomputable def B : C := sorry
noncomputable def C1 : C := sorry
noncomputable def D : C := sorry
noncomputable def M1 : C := midpoint O A B
noncomputable def M2 : C := midpoint O B C1
noncomputable def M3 : C := midpoint O C1 D
noncomputable def M4 : C := midpoint O D A

-- Theorem: Proving that the segments M1M3 and M2M4 are perpendicular
theorem segments_perpendicular : 
  ∠ (O, M1, M3) + ∠ (O, M2, M4) = 90 := 
  sorry

end segments_perpendicular_l284_284971


namespace inequality_holds_l284_284747

theorem inequality_holds (a : ℝ) (h : a ≠ 0) : |a + (1/a)| ≥ 2 :=
by
  sorry

end inequality_holds_l284_284747


namespace closest_integer_to_cubed_root_l284_284243

theorem closest_integer_to_cubed_root :
  let a := 5
  let b := 9
  let c := a^3 + b^3
  let k := 9
  (\(\lfloor \sqrt[3]{c} \rfloor = k)) :=
by
  let a := 5
  let b := 9
  let c := a^3 + b^3
  let k := 9
  sorry

end closest_integer_to_cubed_root_l284_284243


namespace buying_pets_l284_284660

theorem buying_pets {puppies kittens hamsters birds : ℕ} :
(∃ pets : ℕ, pets = 12 * 8 * 10 * 5 * 4 * 3 * 2) ∧ 
puppies = 12 ∧ kittens = 8 ∧ hamsters = 10 ∧ birds = 5 → 
12 * 8 * 10 * 5 * 4 * 3 * 2 = 115200 :=
by
  intros h
  sorry

end buying_pets_l284_284660


namespace find_other_two_sides_of_isosceles_right_triangle_l284_284841

noncomputable def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  ((AB.1 ^ 2 + AB.2 ^ 2 = AC.1 ^ 2 + AC.2 ^ 2 ∧ BC.1 ^ 2 + BC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AB.1 ^ 2 + AB.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AC.1 ^ 2 + AC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AC.1 ^ 2 + AC.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AB.1 ^ 2 + AB.2 ^ 2 = 2 * (AC.1 ^ 2 + AC.2 ^ 2)))

theorem find_other_two_sides_of_isosceles_right_triangle (A B C : ℝ × ℝ)
  (h : is_isosceles_right_triangle A B C)
  (line_AB : 2 * A.1 - A.2 = 0)
  (midpoint_hypotenuse : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 2) :
  (A.1 + 2 * A.2 = 2 ∨ A.1 + 2 * A.2 = 14) ∧ 
  ((A.2 = 2 * A.1) ∨ (A.1 = 4)) :=
sorry

end find_other_two_sides_of_isosceles_right_triangle_l284_284841


namespace solve_quadratic_l284_284185

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l284_284185


namespace no_real_roots_l284_284696

def op (m n : ℝ) : ℝ := n^2 - m * n + 1

theorem no_real_roots (x : ℝ) : op 1 x = 0 → ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by {
  sorry
}

end no_real_roots_l284_284696


namespace solve_quadratic_l284_284184

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l284_284184


namespace intersect_sets_example_l284_284402

open Set

theorem intersect_sets_example : 
  let A := {x : ℝ | -1 < x ∧ x ≤ 3}
  let B := {x : ℝ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4}
  A ∩ B = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3} :=
by
  sorry

end intersect_sets_example_l284_284402


namespace inequality_holds_iff_l284_284292

theorem inequality_holds_iff : ∀ x : ℝ, (x + 1) * (1 / x - 1) > 0 ↔ (x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 0 1) := 
by
  intro x
  sorry

end inequality_holds_iff_l284_284292


namespace max_abc_value_l284_284408

noncomputable def max_abc (a b c : ℝ) : ℝ :=
if h : a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + bc + ac = 1 then abc else 0

theorem max_abc_value : ∀ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + bc + ac = 1) → max_abc a b c = (real.sqrt 3) / 9 :=
by sorry

end max_abc_value_l284_284408


namespace domain_of_sqrt_tan_l284_284568

open Real

theorem domain_of_sqrt_tan (k : ℤ) (x : ℝ) :
  (∃ k : ℤ, x ∈ (k * π - π / 4, k * π + π / 2]) ↔ 1 - tan (x - π / 4) ≥ 0 :=
by
  sorry

end domain_of_sqrt_tan_l284_284568


namespace cone_height_is_sqrt3_l284_284596

-- Define the given conditions
def slant_height := 2
def lateral_area := 2 * Real.pi

-- Define the function to calculate the radius from the lateral area and slant height
def radius (lateral_area slant_height : ℝ) : ℝ :=
  lateral_area / (Real.pi * slant_height)

-- Use the radius to find the height using the Pythagorean theorem
def height (slant_height radius : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - radius ^ 2)

-- Define the proof problem statement
theorem cone_height_is_sqrt3 : height slant_height (radius lateral_area slant_height) = Real.sqrt 3 :=
  sorry

end cone_height_is_sqrt3_l284_284596


namespace gary_money_left_l284_284383

variable (initialAmount : Nat)
variable (amountSpent : Nat)

theorem gary_money_left (h1 : initialAmount = 73) (h2 : amountSpent = 55) : initialAmount - amountSpent = 18 :=
by
  sorry

end gary_money_left_l284_284383


namespace fraction_is_one_fourth_l284_284824

theorem fraction_is_one_fourth (f N : ℝ) 
  (h1 : (1/3) * f * N = 15) 
  (h2 : (3/10) * N = 54) : 
  f = 1/4 :=
by
  sorry

end fraction_is_one_fourth_l284_284824


namespace general_term_form_sum_first_n_terms_l284_284038

-- Define the conditions
def is_geometric_sequence (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n+1) = r * a n

def forms_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

-- Main theorem to prove (part 1): General term formula for the sequence {a_n}
theorem general_term_form (a : ℕ → ℕ) (hgeo : is_geometric_sequence a 2) (harith : forms_arithmetic_sequence (a 2) (a 3 + 1) (a 4)) :
  ∀ n, a n = 2^(n-1) := by
  sorry

-- Define the sequence {b_n}
def b (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  a n + n -- a_n + log_2 (a_{n+1}), note: log base 2 of a power of 2 is redundant

-- Sum of the first n terms of the sequence {b_n}
def T (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, b a (i + 1)

-- Main theorem to prove (part 2): Sum of the first n terms of the sequence {b_n}
theorem sum_first_n_terms (a : ℕ → ℕ) (hgeo : is_geometric_sequence a 2) (harith : forms_arithmetic_sequence (a 2) (a 3 + 1) (a 4)) :
  ∀ n, T a n = n * (n + 1) / 2 + 2^n - 1 := by
  sorry

end general_term_form_sum_first_n_terms_l284_284038


namespace solution_set_of_f_x_plus_2_gt_0_l284_284424

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -x^2 - x

theorem solution_set_of_f_x_plus_2_gt_0 : {x : ℝ | f x + 2 > 0} = Set.Ioo (-2) (∞) :=
by
  sorry

end solution_set_of_f_x_plus_2_gt_0_l284_284424


namespace multiple_conditions_false_l284_284192

theorem multiple_conditions_false (a b : ℤ) (m n : ℤ) 
  (ha : a = 3 * m) (hb : b = 5 * n) :
  ¬ (odd (a + b)) ∧ ¬ (3 ∣ (a + b)) ∧ ¬ (5 ∣ (a + b)) ∧ ¬ (¬ (15 ∣ (a + b))) :=
  by
  sorry

end multiple_conditions_false_l284_284192


namespace product_of_divisors_of_72_l284_284364

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284364


namespace range_of_arcsin_shifted_l284_284981

-- Define the arcsin function and conditions
def f (x : ℝ) : ℝ := Real.arcsin (x - 1)

theorem range_of_arcsin_shifted :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → - (Real.pi / 2) ≤ f x ∧ f x ≤ (Real.pi / 2) :=
by
  sorry

end range_of_arcsin_shifted_l284_284981


namespace traveler_distance_l284_284668

theorem traveler_distance (a b c d : ℕ) (h1 : a = 24) (h2 : b = 15) (h3 : c = 10) (h4 : d = 9) :
  let net_ns := a - c
  let net_ew := b - d
  let distance := Real.sqrt ((net_ns ^ 2) + (net_ew ^ 2))
  distance = 2 * Real.sqrt 58 := 
by
  sorry

end traveler_distance_l284_284668


namespace cooling_constant_l284_284247

theorem cooling_constant (θ0 θ1 θ t k : ℝ) (h1 : θ1 = 60) (h0 : θ0 = 15) (ht : t = 3) (hθ : θ = 42)
  (h_temp_formula : θ = θ0 + (θ1 - θ0) * Real.exp (-k * t)) :
  k = 0.17 :=
by sorry

end cooling_constant_l284_284247


namespace ratio_area_triangles_l284_284300

theorem ratio_area_triangles (a b : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let area_A := (1/2) * (2 * a / n) * b in
  let area_B := (1/2) * (2 * b / m) * a in
  area_A / area_B = m / n := 
by
  sorry

end ratio_area_triangles_l284_284300


namespace find_a_l284_284003

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284003


namespace find_k_for_two_identical_solutions_l284_284381

theorem find_k_for_two_identical_solutions (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k) ∧ (∀ x : ℝ, x^2 = 4 * x + k → x = 2) ↔ k = -4 :=
by
  sorry

end find_k_for_two_identical_solutions_l284_284381


namespace min_value_problem_l284_284906

theorem min_value_problem (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2 = 4 * (Real.root 4 5 - 1)^2 := 
sorry

end min_value_problem_l284_284906


namespace horse_problem_l284_284106

-- Definitions based on conditions:
def total_horses : ℕ := 100
def tiles_pulled_by_big_horse (x : ℕ) : ℕ := 3 * x
def tiles_pulled_by_small_horses (x : ℕ) : ℕ := (100 - x) / 3

-- The statement to prove:
theorem horse_problem (x : ℕ) : 
    tiles_pulled_by_big_horse x + tiles_pulled_by_small_horses x = 100 :=
sorry

end horse_problem_l284_284106


namespace smallest_positive_T_l284_284707

theorem smallest_positive_T :
    ∃ T : ℤ, (∀ (b : Fin 100 → ℤ), 
        (∀ i, b i = 1 ∨ b i = -1) → 
        T = ∑ i in Finset.range 100, ∑ j in Finset.Ico i 100, b i * b j) ∧ T > 0 ∧ T = 22 :=
by
    sorry

end smallest_positive_T_l284_284707


namespace find_a_on_60_deg_angle_l284_284044

theorem find_a_on_60_deg_angle (a : ℝ) (ha : real.sin 60 = (√3 / 2)) : 
  (a / 4) = √3 → a = 4 * √3 :=
by
  intro ha_eq
  sorry

end find_a_on_60_deg_angle_l284_284044


namespace tan_C_value_min_tan_C_value_l284_284398

/-
Proof Problem 1: 
- Question: \tan C given a = -8.
- Conditions: 
   - \tan A and \tan B are the roots of x^2 + ax + 4 = 0.
   - a = -8.
- Answer: \tan C = \frac{8}{3}.
-/
theorem tan_C_value (a : ℝ) (tanA tanB : ℝ) (h : tanA * tanB = 4) (h_sum : tanA + tanB = 8) (h_a : a = -8) : 
  let tanC := - (tanA + tanB) / (1 - tanA * tanB) in
  tanC = 8 / 3 := by
    sorry

/-
Proof Problem 2: 
- Question: Minimum value of \tan C.
- Conditions:
   - \tan A and \tan B are the roots of x^2 + ax + 4 = 0.
   - a \leq -4.
- Answer: Minimum value of \tan C is \frac{4}{3} with \tan A = 2 and \tan B = 2.
-/
theorem min_tan_C_value (a : ℝ) (tanA tanB : ℝ) (h_root : tanA * tanB = 4) (h_range : a ≤ -4) :
  let tanC := - (tanA + tanB) / (1 - tanA * tanB) in
  tanC ≥ 4 / 3 ∧ (tanC = 4 / 3 ↔ tanA = 2 ∧ tanB = 2) := by
    sorry

end tan_C_value_min_tan_C_value_l284_284398


namespace triangle_PQR_area_l284_284319

structure Point where
  x : Float
  y : Float

noncomputable def triangle_area (A B C : Point) : Float :=
  0.5 * abs (A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)

def P : Point := { x := -6, y := 4 }
def Q : Point := { x := 1, y := 7 }
def R : Point := { x := 4, y := -3 }

theorem triangle_PQR_area :
  triangle_area P Q R = 59.5 :=
by
  sorry

end triangle_PQR_area_l284_284319


namespace circle_area_difference_l284_284633

theorem circle_area_difference (C1 C2 : ℝ) (hC1 : C1 = 132) (hC2 : C2 = 352) : 
  ∃ (A1 A2 : ℝ), (π * (C1 / (2 * π))^2 = A1) ∧ (π * (C2 / (2 * π))^2 = A2) ∧ (A2 - A1 ≈ 8466.593) :=
by
  sorry

end circle_area_difference_l284_284633


namespace max_min_sum_eq_six_l284_284463

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp (|x - 1|) - Real.sin (x - 1)) / Real.exp (|x - 1|)

theorem max_min_sum_eq_six : 
  let I := Set.Icc (-3 : ℝ) (5 : ℝ),
      f_max := sup {y : ℝ | ∃ x ∈ I, f x = y},
      f_min := inf {y : ℝ | ∃ x ∈ I, f x = y}
  in f_max + f_min = 6 := 
by
  sorry

end max_min_sum_eq_six_l284_284463


namespace profit_from_bracelets_l284_284251

theorem profit_from_bracelets 
  (cost_string : ℕ) (cost_beads : ℕ) (selling_price : ℕ) (num_bracelets : ℕ)
  (total_cost : ℕ) (total_revenue : ℕ) (total_profit : ℕ) :
  cost_string = 1 → cost_beads = 3 → selling_price = 6 → num_bracelets = 25 →
  total_cost = num_bracelets * (cost_string + cost_beads) →
  total_revenue = num_bracelets * selling_price →
  total_profit = total_revenue - total_cost →
  total_profit = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2] at h5
  rw [h3, h4] at h6
  rw [h6, h5] at h7
  exact h7.symm

end profit_from_bracelets_l284_284251


namespace matrix_determinants_exist_l284_284884

variables {m n : ℕ} (a : Fin (m+1) → ℝ)

theorem matrix_determinants_exist
  (h_m : 1 < m)
  (h_n : 1 < n) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) ℝ), 
    (∀ j : Fin m, Matrix.det (A j) = a j) ∧ 
    (Matrix.det (Finset.univ.sum (λ j, A j)) = a ⟨m, Nat.lt_succ_self m⟩) :=
sorry

end matrix_determinants_exist_l284_284884


namespace find_a_l284_284004

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284004


namespace Peggy_dolls_l284_284162

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l284_284162


namespace hunter_mason_meetings_l284_284072

-- Definitions based on conditions
def hunter_speed : ℝ := 200 -- meters per minute
def mason_speed : ℝ := 240 -- meters per minute
def hunter_radius : ℝ := 50 -- meters
def mason_radius : ℝ := 70 -- meters
def duration : ℝ := 45 -- minutes

-- Circumference calculation
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Angular speed calculation in radians per minute
def angular_speed (speed radius : ℝ) : ℝ := (speed / (circumference radius)) * 2 * Real.pi

-- Determine the angular speeds for Hunter and Mason
def omega_hunter : ℝ := angular_speed hunter_speed hunter_radius
def omega_mason : ℝ := angular_speed mason_speed mason_radius

-- Relative angular speed (since they run in opposite directions, add the speeds)
def relative_angular_speed : ℝ := omega_hunter + omega_mason

-- Time to meet (in minutes)
def time_to_meet : ℝ := (2 * Real.pi) / relative_angular_speed

-- Number of meetings in 45 minutes
def number_of_meetings : ℝ := duration / time_to_meet

-- Proof statement
theorem hunter_mason_meetings : ⌊number_of_meetings⌋ = 81 :=
by 
  -- Proof elided; place proof here
  sorry

end hunter_mason_meetings_l284_284072


namespace cycle_of_circles_l284_284294

def circle (P Q : Point) : Type := sorry  -- Define a circle passing through points P and Q (details omitted for brevity)
def is_tangent (C1 C2 : circle) : Prop := sorry  -- Define tangency of circles (details omitted)

theorem cycle_of_circles (A B C : Point) (C1 : circle A B) 
  (C2 : circle B C) (C3 : circle C A) (C4 : circle A B) 
  (C5 : circle B C) (C6 : circle C A) (C7 : circle A B)
  (h1 : is_tangent C1 C2)
  (h2 : is_tangent C2 C3)
  (h3 : is_tangent C3 C4)
  (h4 : is_tangent C4 C5)
  (h5 : is_tangent C5 C6)
  (h6 : is_tangent C6 C7) :
  C1 = C7 :=
sorry

end cycle_of_circles_l284_284294


namespace find_x0_l284_284148

-- Defining the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Defining the integral condition
def integral_condition (a c x0 : ℝ) : Prop :=
  (∫ x in (0 : ℝ)..(1 : ℝ), f a c x) = f a c x0

-- Proving the main statement
theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (h_range : 0 ≤ x0 ∧ x0 ≤ 1) (h_integral : integral_condition a c x0) :
  x0 = Real.sqrt (1 / 3) :=
by
  sorry

end find_x0_l284_284148


namespace geometric_progression_identity_l284_284949

theorem geometric_progression_identity 
  (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 :=
by 
  sorry

end geometric_progression_identity_l284_284949


namespace range_of_m_l284_284400

variables {m x : ℝ}

def p (m : ℝ) : Prop := (16 * (m - 2)^2 - 16 > 0) ∧ (m - 2 < 0)
def q (m : ℝ) : Prop := (9 * m^2 - 4 < 0)
def pq (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(q m)

theorem range_of_m (h : pq m) : m ≤ -2/3 ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end range_of_m_l284_284400


namespace regular_soda_count_l284_284277

theorem regular_soda_count 
  (diet_soda : ℕ) 
  (additional_soda : ℕ) 
  (h1 : diet_soda = 19) 
  (h2 : additional_soda = 41) 
  : diet_soda + additional_soda = 60 :=
by
  sorry

end regular_soda_count_l284_284277


namespace radius_of_circle_l284_284591

theorem radius_of_circle (C : ℝ) (hC : C = 8) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 4 / Real.pi :=
by
  use 4 / Real.pi
  split
  · rw hC
    rw [mul_assoc, ←div_eq_mul_inv, mul_div_cancel_left (4 : ℝ) (Real.pi_ne_zero : Real.pi ≠ 0)]
  · refl

end radius_of_circle_l284_284591


namespace expected_value_X_variance_X_expected_value_2X_plus_3_variance_3X_plus_2_l284_284464

open probability_theory

noncomputable theory

def X_distrib_condition (X : ℝ) : Prop :=
  ∃ (p : ℝ), p = 1 / 4 ∧ P (X = 0) = p ∧ P (X = 1) = 1 - p

theorem expected_value_X (X : ℝ) (hX : X_distrib_condition X) :
  E(X) = 3 / 4 := sorry

theorem variance_X (X : ℝ) (hX : X_distrib_condition X) :
  var(X) = 3 / 16 := sorry

theorem expected_value_2X_plus_3 (X : ℝ) (hX : X_distrib_condition X) :
  E(2 * X + 3) ≠ 3 := sorry

theorem variance_3X_plus_2 (X : ℝ) (hX : X_distrib_condition X) :
  var(3 * X + 2) ≠ 41 / 16 := sorry

end expected_value_X_variance_X_expected_value_2X_plus_3_variance_3X_plus_2_l284_284464


namespace printer_difference_l284_284171

-- Define the conditions
def A := 35 / 60 -- Rate at which Printer A works
def task_pages : ℝ := 35 -- Total pages of the task
def time_together : ℝ := 24 -- Time taken when both printers work together
def time_A : ℝ := 60 -- Time taken by Printer A alone to finish the task

-- Printer A rate derived from the given condition
lemma A_rate : A = 35 / 60 := by
  -- Here we would typically prove that A is indeed 35 / 60
  sorry

-- Define the rate of B given both working together finish the task in 24 minutes
def B := (task_pages / time_together) - (A * time_together) / time_together

-- State the goal to be proved
theorem printer_difference :
  B - A = 7 / 24 := by 
  -- Here we would typically have the steps to show that B - A is indeed 7 / 24
  sorry

end printer_difference_l284_284171


namespace supermarket_profit_maximization_l284_284270

-- Define the conditions
def profit_x1 := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → ∃ a : ℝ, y = x^a
def profit_x2 := ∀ x : ℝ, 2 < x ∧ x ≤ 5 → ∃ (a b c : ℝ), y = a * x^2 + b * x + c

-- Data points as given in the problem
def data_points := [(1, 1), (2, Real.sqrt 2), (3, 5), (4, 4), (5, 1)]

-- Define the functional relationships proven
def functional_relationship_x1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → y = x^(1/2) := sorry
def functional_relationship_x2 : ∀ x : ℝ, 2 < x ∧ x ≤ 5 → y = -x^2 + 6 * x - 4 := sorry

-- Define that the maximum profit is achieved at x = 3
def max_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 5 ∧ x = 3 := sorry

-- Main proof statement
theorem supermarket_profit_maximization :
  profit_x1 ∧ profit_x2 ∧
  functional_relationship_x1 ∧
  functional_relationship_x2 ∧
  max_profit := sorry

end supermarket_profit_maximization_l284_284270


namespace interval_of_monotonic_increase_solution_set_for_inequality_l284_284050

noncomputable def f (x : ℝ) : ℝ := (5 - x + 4 ^ x) / 2 - (abs (5 - x - 4 ^ x)) / 2

theorem interval_of_monotonic_increase :
  {x : ℝ | f x = 5 - x} = (-∞, 1] :=
sorry

theorem solution_set_for_inequality :
  {x : ℝ | f x > sqrt 5} = (1, 5 - sqrt 5) ∪ {x : ℝ | log 4 (sqrt 5) < x ∧ x ≤ 1} :=
sorry

end interval_of_monotonic_increase_solution_set_for_inequality_l284_284050


namespace simplify_complex_expression_l284_284957

open Complex

theorem simplify_complex_expression :
  let a := (4 : ℂ) + 6 * I
  let b := (4 : ℂ) - 6 * I
  ((a / b) - (b / a) = (24 * I) / 13) := by
  sorry

end simplify_complex_expression_l284_284957


namespace propositions_correct_l284_284674

theorem propositions_correct : 
  let prop1 := ¬ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0))
  let prop2 := ∀ (P : Type), (regular_polygon P → similar P) → (similar P → regular_polygon P)
  let prop3 := ¬ (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0
  let prop4 := ∀ (Q : Type), (rectangle Q → equal_diagonals Q) → (equal_diagonals Q → rectangle Q)
  in (prop1 = true ∧ prop3 = true) ∧ (prop2 = false ∧ prop4 = false) :=
by
  have prop1_true : prop1 = true := sorry
  have prop2_false : prop2 = false := sorry
  have prop3_true : prop3 = true := sorry
  have prop4_false : prop4 = false := sorry
  exact ⟨⟨prop1_true, prop3_true⟩, ⟨prop2_false, prop4_false⟩⟩

end propositions_correct_l284_284674


namespace D_double_prime_coordinates_l284_284944

theorem D_double_prime_coordinates :
  let A := (1, 3)
      B := (3, 8)
      C := (7, 6)
      D := (5, 1) in
  let D' := (-D.1, D.2) in
  let D'' := (-D'.2, -D'.1) in
  D'' = (-1, 5) :=
by
  sorry

end D_double_prime_coordinates_l284_284944


namespace find_a_l284_284777

theorem find_a
  (a : ℝ)
  (h1 : ∃ P Q : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 4 * P.2 + 1 = 0) ∧ (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.1 + 4 * Q.2 + 1 = 0) ∧
                         (a * P.1 + 2 * P.2 + 6 = 0) ∧ (a * Q.1 + 2 * Q.2 + 6 = 0) ∧
                         ((P.1 - 1) * (Q.1 - 1) + (P.2 + 2) * (Q.2 + 2) = 0)) :
  a = 2 :=
by
  sorry

end find_a_l284_284777


namespace jerry_needs_shingles_l284_284869

theorem jerry_needs_shingles :
  let roofs := 3 in
  let length := 20 in
  let width := 40 in
  let sides := 2 in
  let shingles_per_sqft := 8 in
  let area_one_side := length * width in
  let area_one_roof := area_one_side * sides in
  let total_area := area_one_roof * roofs in
  let total_shingles := total_area * shingles_per_sqft in
  total_shingles = 38400 :=
by
  sorry

end jerry_needs_shingles_l284_284869


namespace cost_price_bicycle_A_l284_284287

variable {CP_A CP_B SP_C : ℝ}

theorem cost_price_bicycle_A (h1 : CP_B = 1.25 * CP_A) (h2 : SP_C = 1.25 * CP_B) (h3 : SP_C = 225) :
  CP_A = 144 :=
by
  sorry

end cost_price_bicycle_A_l284_284287


namespace second_number_20th_row_l284_284249

theorem second_number_20th_row :
  let n := 20
  in let first_number := (n + 1) ^ 2 - 1
  in let second_number := first_number - 1
  in second_number = 439 := sorry

end second_number_20th_row_l284_284249


namespace part1_part2_l284_284800

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem part1 :
  { x : ℝ | f x ≤ 6 } = set.Icc (-1 : ℝ) 2 :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃ x : ℝ, f x < |a - 1|) ↔ a ∈ set.Iio (-3) ∪ set.Ioi 5 :=
by
  sorry

end part1_part2_l284_284800


namespace slope_of_line_l284_284985

theorem slope_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → (1 = 1) :=
by
  intros x y h
  sorry

end slope_of_line_l284_284985


namespace production_assessment_l284_284275

noncomputable def external_diameter_morning : ℝ := 9.9
noncomputable def external_diameter_afternoon : ℝ := 9.3
noncomputable def mean_external_diameter : ℝ := 10
noncomputable def variance_external_diameter : ℝ := 0.04
noncomputable def std_deviation : ℝ := Real.sqrt variance_external_diameter

theorem production_assessment : 
  ∀ (mean std_dev : ℝ) (morning_diam afternoon_diam : ℝ),
  morning_diam ∈ Icc (mean - 3 * std_dev) (mean + 3 * std_dev) ∧ 
  ¬(afternoon_diam ∈ Icc (mean - 3 * std_dev) (mean + 3 * std_dev)) →
  (morning_diam = 9.9 ∧ afternoon_diam = 9.3 ∧ mean = 10 ∧ std_dev = Real.sqrt 0.04 →
  (morning_diam ∈ Icc 9.4 10.6 ∧ ¬(afternoon_diam ∈ Icc 9.4 10.6))) :=
by
  intro mean std_dev morning_diam afternoon_diam
  intro h1 h2
  sorry

end production_assessment_l284_284275


namespace consecutive_sum_to_20_has_one_set_l284_284070

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l284_284070


namespace integer_solutions_count_l284_284041

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end integer_solutions_count_l284_284041


namespace parabola_focus_a_trajectory_equation_l284_284218

-- Define the parabola and its properties
def parabola (a : ℝ) (x y : ℝ) : Prop :=
  y = a * x^2

-- Define the focus of the parabola
def focus (F : ℝ × ℝ) : Prop :=
  F = (0, 1)

-- Proof that a = 1/4
theorem parabola_focus_a (a : ℝ) (H : focus (0, 1)) : a = 1/4 := sorry

-- Define a moving point on the parabola
def moving_point (a : ℝ) (x y : ℝ) : Prop :=
  parabola a x y

-- Define the midpoint of the segment FP
def midpoint_trajectory (a : ℝ) (x y : ℝ) (F P M : ℝ × ℝ) : Prop :=
  let P := (x, a * x^2)
  let F := (0, 1)
  let M := ((fst P + fst F) / 2, (snd P + snd F) / 2)
  (fst M)^2 - 2 * (snd M) + 1 = 0

-- Proof of the equation of the trajectory of the midpoint
theorem trajectory_equation (a : ℝ) (x y : ℝ) (H1 : focus (0, 1))
    (H2 : moving_point a x y) : midpoint_trajectory a x y (0, 1) (x, a * x^2) ((x + 0) / 2, (a * x^2 + 1) / 2) := sorry

end parabola_focus_a_trajectory_equation_l284_284218


namespace tangent_asymptote_to_circle_l284_284969

noncomputable def hyperbola := { x : ℝ // x^2 / 6 - x^2 / 3 = 1 }
noncomputable def circle (r : ℝ) := { p : ℝ × ℝ // (p.1 - 3)^2 + p.2^2 = r^2 ∧ r > 0 }

theorem tangent_asymptote_to_circle {r : ℝ} (h₁ : ∀ p ∈ hyperbola, (∃ a b : ℝ, b ≠ 0 ∧ p.2 = √2 / √3 * p.1))
  (h₂ : ∀ p ∈ circle r, (p.1 - 3)^2 + p.2^2 = r^2 ∧ r > 0) :
  r = √3 := sorry

end tangent_asymptote_to_circle_l284_284969


namespace inequality_holds_l284_284919

theorem inequality_holds
  {n : ℕ} (x : Fin n → ℝ)
  (h1 : ∀ i, 0 < x i)
  (h2 : (Finset.univ.sum x) < 1) :
  ((n:ℝ) / ((n:ℝ) + 1)) * (Finset.univ.prod x) * (1 - Finset.univ.sum x) ≤ 
  (Finset.univ.sum x) * (Finset.univ.prod (λ i, 1 - x i)) :=
sorry

end inequality_holds_l284_284919


namespace number_of_triangles_l284_284627

open Real

-- Define the point X
structure Point :=
  (x : ℝ)
  (y : ℝ)

def X (p : ℝ) := Point.mk (1994 * p) (7 * 1994 * p)

-- Define the origin O
def O := Point.mk 0 0

-- Main theorem
theorem number_of_triangles (p : ℝ) (hp : Nat.prime p) :
  (p ≠ 2 ∧ p ≠ 997 ∧ number_of_triangles_X p = 36) ∨
  (p = 2 ∧ number_of_triangles_X p = 18) ∨
  (p = 997 ∧ number_of_triangles_X p = 20) :=
sorry

-- Placeholder definition for number_of_triangles_X
-- This would represent the number of valid triangles XYZ fulfilling the problem's conditions.
def number_of_triangles_X (p : ℝ) : ℕ := sorry

end number_of_triangles_l284_284627


namespace john_books_nights_l284_284489

theorem john_books_nights (n : ℕ) (cost_per_night discount amount_paid : ℕ) 
  (h1 : cost_per_night = 250)
  (h2 : discount = 100)
  (h3 : amount_paid = 650)
  (h4 : amount_paid = cost_per_night * n - discount) : 
  n = 3 :=
by
  sorry

end john_books_nights_l284_284489


namespace tan_theta_minus_pi_four_l284_284082

theorem tan_theta_minus_pi_four (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) (h3 : Real.sin θ = -3/5) :
  Real.tan (θ - π / 4) = -1 / 7 :=
sorry

end tan_theta_minus_pi_four_l284_284082


namespace cmp_c_b_a_l284_284748

noncomputable def a : ℝ := 17 / 18
noncomputable def b : ℝ := Real.cos (1 / 3)
noncomputable def c : ℝ := 3 * Real.sin (1 / 3)

theorem cmp_c_b_a:
  c > b ∧ b > a := by
  sorry

end cmp_c_b_a_l284_284748


namespace johnny_closed_days_l284_284119

theorem johnny_closed_days :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let price_per_pound := 8
  let weekly_expenditure := 1920
  let daily_pounds := dishes_per_day * pounds_per_dish
  let daily_cost := daily_pounds * price_per_pound
  let days_open := weekly_expenditure / daily_cost
  let days_in_week := 7
  let days_closed := days_in_week - days_open
  days_closed = 3 :=
by
  sorry

end johnny_closed_days_l284_284119


namespace sunflower_B_percent_correct_l284_284657

noncomputable def percent_sunflower_B (millet_A sunflower_A millet_B sunflower_mix : ℝ) : ℝ := 
  let sunflower_A_proportion := 0.6 * 0.6
  let mix_proportion_A := 0.6
  let mix_proportion_B := 0.4
  let equation := sunflower_A_proportion + (mix_proportion_B * x / 100) = sunflower_mix
  let x := 35 -- derived from solving the equation above
  x

theorem sunflower_B_percent_correct :
  ∀ (millet_A sunflower_A millet_B sunflower_mix : ℝ), 
  millet_A = 0.4 → 
  sunflower_A = 0.6 → 
  millet_B = 0.65 → 
  sunflower_mix = 0.5 → 
  percent_sunflower_B millet_A sunflower_A millet_B sunflower_mix = 35 :=
by
  intros millet_A sunflower_A millet_B sunflower_mix h1 h2 h3 h4
  unfold percent_sunflower_B
  sorry

end sunflower_B_percent_correct_l284_284657


namespace inequality_solution_l284_284379

theorem inequality_solution (x : ℝ) :
    (∀ t : ℝ, abs (t - 3) + abs (2 * t + 1) ≥ abs (2 * x - 1) + abs (x + 2)) ↔ 
    (-1 / 2 ≤ x ∧ x ≤ 5 / 6) :=
by
  sorry

end inequality_solution_l284_284379


namespace range_of_a_l284_284147

def f (x : ℝ) : ℝ :=
  if x < 0 then (1 / 2) ^ x - 7 else sqrt x

theorem range_of_a (a : ℝ) (h : f a < 1) : -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l284_284147


namespace combined_moles_l284_284534

def balanced_reaction (NaHCO3 HC2H3O2 H2O : ℕ) : Prop :=
  NaHCO3 + HC2H3O2 = H2O

theorem combined_moles (NaHCO3 HC2H3O2 : ℕ) 
  (h : balanced_reaction NaHCO3 HC2H3O2 3) : 
  NaHCO3 + HC2H3O2 = 6 :=
sorry

end combined_moles_l284_284534


namespace range_of_m_l284_284048

theorem range_of_m (m : ℝ) (y_P : ℝ) (h1 : -3 ≤ y_P) (h2 : y_P ≤ 0) :
  m = (2 + y_P) / 2 → -1 / 2 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l284_284048


namespace min_value_fraction_sum_equality_when_a_b_equal_one_l284_284144

theorem min_value_fraction_sum (n : ℕ) (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ( 1 / (1 + a^n) + 1 / (1 + b^n) ) ≥ 1 :=
begin
  sorry
end

theorem equality_when_a_b_equal_one (n : ℕ) :
  ( 1 / (1 + 1^n) + 1 / (1 + 1^n) ) = 1 :=
begin
  calc
    1 / (1 + 1^n) + 1 / (1 + 1^n) = 1 / 2 + 1 / 2 : by rw [nat.one_pow]
                               ... = 1           : by norm_num,
end

end min_value_fraction_sum_equality_when_a_b_equal_one_l284_284144


namespace find_angle_between_a_and_b_find_range_of_t_l284_284808

noncomputable theory
open_locale classical

variables {R : Type*} [inner_product_space ℝ R]
          (a b : R) (t : ℝ)

-- Condition definitions
def cond1 := ∥a∥ = 2
def cond2 := ∥b∥ = 1
def cond3 := (a + 2 • b) ⬝ (a - b) = 1
def angle_60 := real.angle.rad (real.pi / 3)

-- Proof statement for the first question
theorem find_angle_between_a_and_b
  (h1 : cond1 a)
  (h2 : cond2 b)
  (h3 : cond3 a b) : real.angle (angle a b) = 2 * real.pi / 3 :=
sorry

-- Condition definition for the second question
def angle_between_ab_is_obtuse (t : ℝ) : Prop :=
  (2 * t • a + 7 • b) ⬝ (a + t • b) < 0

-- Proof statement for the second question
theorem find_range_of_t
  (h1 : cond1 a)
  (h2 : cond2 b)
  (h4 : inner_product_geometry.angle a b = angle_60)
  (ht : angle_between_ab_is_obtuse a b t) : 
  -7 < t ∧ t < -√14 / 2 :=
sorry

end find_angle_between_a_and_b_find_range_of_t_l284_284808


namespace min_value_of_x_under_conditions_l284_284456

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_value_of_x_under_conditions :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 1 →
  (∃ x_min : ℝ, S x y z = S x_min x_min (Real.sqrt 2 - 1) ∧ x_min = Real.sqrt (Real.sqrt 2 - 1)) :=
by
  intros x y z hx hy hz hxyz
  use Real.sqrt (Real.sqrt 2 - 1)
  sorry

end min_value_of_x_under_conditions_l284_284456


namespace axis_of_symmetry_is_correct_strictly_increasing_intervals_l284_284787

noncomputable def f (x : ℝ) : ℝ :=
  4 * cos x * sin (x + π / 6) - 4 * sqrt 3 * sin x * cos x + 1

theorem axis_of_symmetry_is_correct (k : ℤ) :
  ∀ x, f x = 2 * cos (2 * x + π / 3) + 2 → x = k * π / 2 - π / 6 :=
sorry

theorem strictly_increasing_intervals (x : ℝ) (k : ℤ)
  (h : x ∈ Icc (-π / 2) (π / 2)) :
  f x = 2 * cos (2 * x + π / 3) + 2 →
  ((x ∈ Icc (-π / 2) (-π / 6)) ∨ (x ∈ Icc (π / 3) (π / 2))) :=
sorry

end axis_of_symmetry_is_correct_strictly_increasing_intervals_l284_284787


namespace expression_domain_f_range_t_range_m_l284_284797

-- Given function definition and condition
def f (x : ℝ) (a b : ℝ) : ℝ := log10 (2 * x / (a * x + b))

-- Conditions
lemma f_1_zero (a b : ℝ) (h : f 1 a b = 0) : log10 (2 / (a + b)) = 0 := by 
  simp [f] at h; exact h

lemma f_diff_log (x : ℝ) (a b : ℝ) (h : x > 0) (h_diff : f x a b - f (1 / x) a b = log10 x) :
  log10 (2 * x / (a * x + b)) - log10 (2 / (a / x + b)) = log10 x := by 
  simp [f] at h_diff; exact h_diff

-- Proving that the function expression and its domain
theorem expression_domain_f (a b : ℝ) (h1 : f 1 a b = 0) (h2 : ∀ x > 0, f x a b - f (1 / x) a b = log10 x) : 
  ∃ f' : ℝ → ℝ, (f' = λ x, log10 (2 * x / (x + 1))) ∧ (∀ x, x > 0 ∨ x < -1 → f x a b = f' x) := by 
  sorry

-- Proving the range of t
theorem range_t (t : ℝ) : ∃ x, f x 1 1 = log10 t ↔ (0 < t ∧ t < 2) ∨ (t > 2):= by 
  sorry

-- Proving the range of m
theorem range_m (m : ℝ) : ¬∃ x, f x 1 1 = log10 (8 * x + m) ↔ (0 ≤ m ∧ m < 18) := by 
  sorry

end expression_domain_f_range_t_range_m_l284_284797


namespace work_completion_days_l284_284635

theorem work_completion_days :
  let x := 15
  let y := 45
  let z := 30
  let combined_rate := (1 / x.to_rat) + (1 / y.to_rat) + (1 / z.to_rat)
  let days_to_complete := 1 / combined_rate
  abs (days_to_complete - 90 / 11) < 1e-2 := 
by
  sorry

end work_completion_days_l284_284635


namespace son_age_l284_284630

variable (F S : ℕ)
variable (h₁ : F = 3 * S)
variable (h₂ : F - 8 = 4 * (S - 8))

theorem son_age : S = 24 := 
by 
  sorry

end son_age_l284_284630


namespace matrix_neg_identity_l284_284495

open Matrix

variables {R : Type*} [Ring R] [IsDomain R] (A : Matrix (Fin 2) (Fin 2) R) 
  [A ∈ M_2 ℚ]

theorem matrix_neg_identity 
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (h₀ : n ≠ 0) (h₁ : A ^ n = - (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = - (1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = - (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end matrix_neg_identity_l284_284495


namespace calculate_S_l284_284823

noncomputable def c : ℝ := (9 : ℝ) * (9 : ℝ) / (4 : ℝ) -- Based on the provided solution
def R (S T : ℝ) : ℝ := c * (S^2) / (T^2)

theorem calculate_S (S : ℝ) (h : R S 4 = 16) : S = 32 / 9 := by
  sorry

end calculate_S_l284_284823


namespace license_plate_palindrome_probability_l284_284931

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l284_284931


namespace cone_height_l284_284594

theorem cone_height (lateral_area : ℝ) (l : ℝ) (h : ℝ) : lateral_area = 2 * real.pi → l = 2 → h = real.sqrt 3 :=
by
  intros h₁ h₂
  sorry

end cone_height_l284_284594


namespace perimeter_of_triangle_l284_284479

-- Define variables and conditions
variables {P Q R S T U A B C : Point}
variable r : ℝ
variable (circle : Point → ℝ → Set Point)

-- Assume properties of the circles and triangle
def conditions : Prop :=
  let circleCenters : Set Point := {P, Q, R, S, T, U} in
  let radii : ∀ x ∈ circleCenters, r = 2 in
  
  -- All circles have radius 2
  ∀ x ∈ circleCenters, ∀ y ∈ circleCenters, (x ≠ y → (∃ a b, an edge length between them is 2 + 2)) ∧
  -- Circle configurations match the triangle sides
  tangent_to_each_other : (circle P r ∪ circle Q r ∪ circle R r = fun x => x ∈ side AB) ∧
  tangent_to_each_other : (circle R r ∪ circle S r ∪ circle T r = fun x => x ∈ side BC) ∧
  tangent_to_each_other : (circle T r ∪ circle U r ∪ circle P r = fun x => x ∈ side CA)

-- Define the perimeter function
noncomputable def perimeter_ABC (A B C : Point) : ℝ :=
  let AB : ℝ := distance A B
  let BC : ℝ := distance B C
  let CA : ℝ := distance C A
  AB + BC + CA

-- The final statement we want to prove
theorem perimeter_of_triangle : conditions → perimeter_ABC A B C = 12 * real.sqrt 3 + 24 :=
by
  intro h1 h2 -- h1, h2 would be expanded from the assumptions of conditions
  sorry

end perimeter_of_triangle_l284_284479


namespace circle_center_radius_l284_284972

theorem circle_center_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center = (3, 0)) ∧ (radius = 3) ∧ 
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 - 6x = 0) :=
by
  use ((3 : ℝ), 0)
  use (3 : ℝ)
  split
  . refl
  split
  . refl
  sorry

end circle_center_radius_l284_284972


namespace sum_of_exponents_l284_284231

theorem sum_of_exponents (r : ℕ) (n : ℕ → ℕ) (a : ℕ → ℤ)
  (h1 : ∀ i j, i < j → n i > n j)
  (h2 : ∀ k, a k = 1 ∨ a k = -1)
  (h3 : ∑ i in Finset.range r, a i * 3 ^ n i = 2019) :
  ∑ i in Finset.range r, n i = 17 :=
sorry

end sum_of_exponents_l284_284231


namespace range_of_target_function_l284_284223

noncomputable def target_function (x : ℝ) : ℝ :=
  1 - 1 / (x^2 - 1)

theorem range_of_target_function :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ target_function x = y ↔ y ∈ (Set.Iio 1 ∪ Set.Ici 2) :=
by
  sorry

end range_of_target_function_l284_284223


namespace increase_interval_abs_diff_l284_284214

noncomputable def abs_diff (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem increase_interval_abs_diff : ∀ x : ℝ, 1 < x → ∃ ε > 0, ∀ y ∈ set.Ioc x (x + ε), abs_diff y > abs_diff x := by
  sorry

end increase_interval_abs_diff_l284_284214


namespace plates_used_total_l284_284233

theorem plates_used_total 
  (num_parents : ℕ) (num_siblings : ℕ) (num_grandparents : ℕ) (num_cousins : ℕ) 
  (meals_per_day : ℕ) (courses_per_meal : ℕ) (plates_per_course : ℕ) (days : ℕ) :
  let num_guests := num_parents + num_siblings + num_grandparents + num_cousins
  let total_people := num_guests + 1
  let plates_per_meal := courses_per_meal * plates_per_course
  let plates_per_day_per_person := meals_per_day * plates_per_meal
  let total_plates := total_people * plates_per_day_per_person * days
  in total_plates = 1728 :=
by
  -- parameters
  let num_parents := 2
  let num_siblings := 3
  let num_grandparents := 2
  let num_cousins := 4
  let meals_per_day := 4
  let courses_per_meal := 3
  let plates_per_course := 2
  let days := 6

  -- conditions derivation
  let num_guests := num_parents + num_siblings + num_grandparents + num_cousins
  let total_people := num_guests + 1
  let plates_per_meal := courses_per_meal * plates_per_course
  let plates_per_day_per_person := meals_per_day * plates_per_meal
  let total_plates := total_people * plates_per_day_per_person * days

  -- assertion
  show total_plates = 1728, from sorry

end plates_used_total_l284_284233


namespace problem_x2_plus_y2_l284_284452

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l284_284452


namespace function_inequality_l284_284525

variable {f : ℝ → ℝ}
variable [Differentiable ℝ f]

theorem function_inequality (h : ∀ x : ℝ, 2 * f x + x * deriv f x > x^2) : ∀ x : ℝ, f x > 0 := sorry

end function_inequality_l284_284525


namespace stones_on_perimeter_of_square_l284_284597

theorem stones_on_perimeter_of_square (n : ℕ) (h : n = 5) : 
  4 * n - 4 = 16 :=
by
  sorry

end stones_on_perimeter_of_square_l284_284597


namespace product_of_divisors_of_72_l284_284371

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284371


namespace evaluate_expression_l284_284712

theorem evaluate_expression :
  let a := 5 ^ 1001
  let b := 6 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end evaluate_expression_l284_284712


namespace right_triangle_condition_l284_284527

def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 4
  | 2 => 4
  | n + 3 => fib (n + 2) + fib (n + 1)

theorem right_triangle_condition (n : ℕ) : 
  ∃ a b c, a = fib n * fib (n + 4) ∧ 
           b = fib (n + 1) * fib (n + 3) ∧ 
           c = 2 * fib (n + 2) ∧
           a * a + b * b = c * c :=
by sorry

end right_triangle_condition_l284_284527


namespace no_A_neither_l284_284101

-- Define the number of club members, and members receiving an A in art, science, and both activities.
variable (total_members : ℕ) (art_A : ℕ) (science_A : ℕ) (both_A : ℕ)

-- Given specific values for these variables.
axiom h1 : total_members = 50
axiom h2 : art_A = 20
axiom h3 : science_A = 30
axiom h4 : both_A = 15

-- The statement we need to prove: the number of members not receiving an A in either activity.
theorem no_A_neither : (total_members - (art_A + science_A - both_A)) = 15 :=
by
  rw [h1, h2, h3, h4]
  rfl

end no_A_neither_l284_284101


namespace food_cost_l284_284116

variable (F : ℝ)

def service_fee : ℝ := 0.12 * F
def tip : ℝ := 5
def total_spent : ℝ := F + service_fee + tip

theorem food_cost : total_spent = 61 → F = 50 := by
  intro h
  sorry

end food_cost_l284_284116


namespace three_digit_number_l284_284719

/-- 
Prove there exists three-digit number N such that 
1. N is of form 100a + 10b + c
2. 1 ≤ a ≤ 9
3. 0 ≤ b, c ≤ 9
4. N = 11 * (a + b + c)
--/
theorem three_digit_number (N a b c : ℕ) 
  (hN: N = 100 * a + 10 * b + c) 
  (h_a: 1 ≤ a ∧ a ≤ 9)
  (h_b_c: 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_condition: N = 11 * (a + b + c)) :
  N = 198 := 
sorry

end three_digit_number_l284_284719


namespace find_a_l284_284862

noncomputable def a_from_triangle (b c B : ℝ) (h_b : b = 1) (h_c : c = Real.sqrt 3) (h_B : B = Real.pi / 6) : ℝ :=
  Real.sqrt (1 ^ 2 + (Real.sqrt 3) ^ 2)

theorem find_a : a_from_triangle 1 (Real.sqrt 3) (Real.pi / 6) 1 (Real.sqrt 3) (Real.pi / 6) = 2 := by
  sorry

end find_a_l284_284862


namespace find_f_l284_284827

variable (f : ℝ → ℝ)

open Function

theorem find_f (h : ∀ x: ℝ, f (3 * x + 2) = 9 * x + 8) : ∀ x: ℝ, f x = 3 * x + 2 := 
sorry

end find_f_l284_284827


namespace lineD_is_parallel_to_line1_l284_284298

-- Define the lines
def line1 (x y : ℝ) := x - 2 * y + 1 = 0
def lineA (x y : ℝ) := 2 * x - y + 1 = 0
def lineB (x y : ℝ) := 2 * x - 4 * y + 2 = 0
def lineC (x y : ℝ) := 2 * x + 4 * y + 1 = 0
def lineD (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Define a function to check parallelism between lines
def are_parallel (f g : ℝ → ℝ → Prop) :=
  ∀ x y : ℝ, (f x y → g x y) ∨ (g x y → f x y)

-- Prove that lineD is parallel to line1
theorem lineD_is_parallel_to_line1 : are_parallel line1 lineD :=
by
  sorry

end lineD_is_parallel_to_line1_l284_284298


namespace cyclic_trapezoid_radii_relation_l284_284564

variables (A B C D O : Type)
variables (AD BC : Type)
variables (r1 r2 r3 r4 : ℝ)

-- Conditions
def cyclic_trapezoid (A B C D: Type) (AD BC: Type): Prop := sorry
def intersection (A B C D O : Type): Prop := sorry
def radius_incircle (triangle : Type) (radius : ℝ): Prop := sorry

theorem cyclic_trapezoid_radii_relation
  (h1: cyclic_trapezoid A B C D AD BC)
  (h2: intersection A B C D O)
  (hr1: radius_incircle AOD r1)
  (hr2: radius_incircle AOB r2)
  (hr3: radius_incircle BOC r3)
  (hr4: radius_incircle COD r4):
  (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4) :=
sorry

end cyclic_trapezoid_radii_relation_l284_284564


namespace cartesian_coordinates_A_cartesian_line_equation_distance_from_point_to_line_l284_284480

open Real

-- Definitions from the problem conditions
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * cos θ, ρ * sin θ)

def polar_line (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = 1

-- Proven Cartesian equivalents
theorem cartesian_coordinates_A : 
  polar_to_cartesian 4 (π / 4) = (2 * sqrt 2, 2 * sqrt 2) :=
sorry

theorem cartesian_line_equation :
  ∀ x y : ℝ, polar_line (sqrt (x^2 + y^2)) (arctan (y / x)) ↔ x + y - sqrt 2 = 0 :=
sorry

theorem distance_from_point_to_line (px py : ℝ) :
  let d := (abs (px + py - sqrt 2)) / sqrt 2
  in px = 2 * sqrt 2 ∧ py = 2 * sqrt 2 → d = 3 :=
sorry

end cartesian_coordinates_A_cartesian_line_equation_distance_from_point_to_line_l284_284480


namespace population_average_age_l284_284468

theorem population_average_age :
  ∃ (k : ℕ), let women := 3 * k,
                 men := 2 * k,
                 total_population := women + men,
                 total_age := 36 * women + 30 * men,
                 average_age := total_age / total_population in
             average_age = 33 + 3 / 5 :=
by
  sorry

end population_average_age_l284_284468


namespace equal_total_areas_of_checkerboard_pattern_l284_284709

-- Definition representing the convex quadrilateral and its subdivisions
structure ConvexQuadrilateral :=
  (A B C D : ℝ × ℝ) -- vertices of the quadrilateral

-- Predicate indicating the subdivision and coloring pattern
inductive CheckerboardColor
  | Black
  | White

-- Function to determine the area of the resulting smaller quadrilateral
noncomputable def area_of_subquadrilateral 
  (quad : ConvexQuadrilateral) 
  (subdivision : ℕ) -- subdivision factor
  (color : CheckerboardColor) 
  : ℝ := -- returns the area based on the subdivision and color
  -- Simplified implementation of area calculation
  -- (detailed geometric computation should replace this placeholder)
  sorry

-- Function to determine the total area of quadrilaterals of a given color
noncomputable def total_area_of_color 
  (quad : ConvexQuadrilateral) 
  (substution : ℕ) 
  (color : CheckerboardColor) 
  : ℝ := -- Total area of subquadrilaterals of the given color
  sorry

-- Theorem stating the required proof
theorem equal_total_areas_of_checkerboard_pattern
  (quad : ConvexQuadrilateral)
  (subdivision : ℕ)
  : total_area_of_color quad subdivision CheckerboardColor.Black = total_area_of_color quad subdivision CheckerboardColor.White :=
  sorry

end equal_total_areas_of_checkerboard_pattern_l284_284709


namespace perpendicular_planes_l284_284950

variables {a b: Type*} [has_perp a b] [has_parallel a b] [has_subset a b] (M N: a)

-- Hypotheses
axiom perp_a_M : a ⊥ M
axiom subset_a_N : a ⊆ N

-- The proof statement
theorem perpendicular_planes (M N: a) (perp_a_M: a ⊥ M) (subset_a_N: a ⊆ N) : M ⊥ N :=
by
  apply sorry
 
end perpendicular_planes_l284_284950


namespace back_wheel_revolutions_50_front_revolutions_front_wheel_revolutions_to_align_l284_284940

-- Definitions from the conditions
def front_wheel_diameter := 6
def back_wheel_diameter := 2
def pi := Real.pi
def front_wheel_circumference : Real := pi * front_wheel_diameter
def back_wheel_circumference : Real := pi * back_wheel_diameter

-- Proof that the back wheel makes 150 revolutions
theorem back_wheel_revolutions_50_front_revolutions :
  (50 * front_wheel_circumference) / back_wheel_circumference = 150 := by
  sorry

-- Proof that the front wheel needs 1 revolution to align again
theorem front_wheel_revolutions_to_align :
  (front_wheel_circumference / Nat.lcm (6 * pi) (2 * pi)) = 1 := by
  sorry

end back_wheel_revolutions_50_front_revolutions_front_wheel_revolutions_to_align_l284_284940


namespace problem_statement_l284_284794

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * x + 2 * a * Real.log x

-- State the problem
theorem problem_statement (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 1 / 4) 
                                (h3 : 0 < x1) (h4 : x1 < x2) 
                                (h5 : x1 + x2 = 1) (h6 : x1 * x2 = a) :
  f x1 a + f x2 a + Real.log 2 + 3 / 2 > 0 :=
sorry

end problem_statement_l284_284794


namespace larger_crust_flour_amount_l284_284871

theorem larger_crust_flour_amount :
  let p_s := 50 in
  let f_s := 1 / 10 in
  let p_s_new := 30 in
  let p_l := 25 in
  let total_flour := p_s * f_s in
  total_flour = p_s_new * f_s + p_l * (2 / 25) :=
by
  let p_s := 50
  let f_s := 1 / 10
  let p_s_new := 30
  let p_l := 25
  let total_flour := p_s * f_s
  have : total_flour = 5 := by
    simp [p_s, f_s, total_flour]
  have : p_s_new * f_s = 3 := by
    simp [p_s_new, f_s]
  show total_flour = p_s_new * f_s + p_l * (2 / 25)
  calc
    total_flour = 5 : this
    ... = 3 + p_l * (2 / 25) : by simp [p_l]
    ... = 3 + 25 * (2 / 25) : by simp [p_l]
    ... = 3 + 2 : by simp
    ... = p_s_new * f_s + p_l * (2 / 25) : by rw [this, ←two_add_three]

end larger_crust_flour_amount_l284_284871


namespace product_of_divisors_of_72_l284_284361

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284361


namespace mrs_jensens_preschool_l284_284935

theorem mrs_jensens_preschool (total_students students_with_both students_with_neither students_with_green_eyes students_with_red_hair : ℕ) 
(h1 : total_students = 40) 
(h2 : students_with_red_hair = 3 * students_with_green_eyes) 
(h3 : students_with_both = 8) 
(h4 : students_with_neither = 4) :
students_with_green_eyes = 12 := 
sorry

end mrs_jensens_preschool_l284_284935


namespace find_value_of_a_squared_b_plus_ab_squared_l284_284066

theorem find_value_of_a_squared_b_plus_ab_squared 
  (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : ab = 2) : 
  a^2 * b + a * b^2 = -6 :=
by 
  sorry

end find_value_of_a_squared_b_plus_ab_squared_l284_284066


namespace average_of_possible_x_values_l284_284817

theorem average_of_possible_x_values (x : ℝ) (h : sqrt (3 * x ^ 2 + 2) = sqrt 50) : 
  (4 + (-4)) / 2 = 0 := 
by
suffices : x = 4 ∨ x = -4, from sorry,
sorry

end average_of_possible_x_values_l284_284817


namespace stickers_in_either_not_both_l284_284327

def stickers_shared := 12
def emily_total_stickers := 22
def mia_unique_stickers := 10

theorem stickers_in_either_not_both : 
  (emily_total_stickers - stickers_shared) + mia_unique_stickers = 20 :=
by
  sorry

end stickers_in_either_not_both_l284_284327


namespace middle_admitted_is_correct_l284_284196

-- Define the total number of admitted people.
def total_admitted := 100

-- Define the proportions of South, North, and Middle volumes.
def south_ratio := 11
def north_ratio := 7
def middle_ratio := 2

-- Calculating the total ratio.
def total_ratio := south_ratio + north_ratio + middle_ratio

-- Hypothesis that we are dealing with the correct ratio and total.
def middle_admitted (total_admitted : ℕ) (total_ratio : ℕ) (middle_ratio : ℕ) : ℕ :=
  total_admitted * middle_ratio / total_ratio

-- Proof statement
theorem middle_admitted_is_correct :
  middle_admitted total_admitted total_ratio middle_ratio = 10 :=
by
  -- This line would usually contain the detailed proof steps, which are omitted here.
  sorry

end middle_admitted_is_correct_l284_284196


namespace hyperbola_condition_l284_284006

theorem hyperbola_condition (m n : ℝ) : 
  (mn < 0) ↔ (∀ x y : ℝ, ∃ k ∈ {a : ℝ | a ≠ 0}, (x^2 / m + y^2 / n = 1)) := sorry

end hyperbola_condition_l284_284006


namespace find_f2018_l284_284798

variables {a b α β : ℝ}

def f (x : ℝ) : ℝ := a * Real.sin ((π / 2) * x + α) + b * Real.cos ((π / 2) * x + β)

theorem find_f2018 (h : f 8 = 18) : f 2018 = -18 := sorry

end find_f2018_l284_284798


namespace monotonic_range_of_t_l284_284788

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end monotonic_range_of_t_l284_284788


namespace sum_of_reciprocal_squares_l284_284160

-- Define the hypothesis conditions
variables (n : ℕ) (hn : 2 ≤ n)

-- Define the main theorem
theorem sum_of_reciprocal_squares (n : ℕ) (hn : 2 ≤ n) :
  1 + ∑ k in Finset.range (n - 1), (1 : ℝ) / (k + 2)^2 < (2*n - 1) / n :=
by
  sorry

end sum_of_reciprocal_squares_l284_284160


namespace circle_equation_l284_284271

theorem circle_equation (x y : ℝ)
  (h_center : ∀ x y, (x - 3)^2 + (y - 1)^2 = r ^ 2)
  (h_origin : (0 - 3)^2 + (0 - 1)^2 = r ^ 2) :
  (x - 3) ^ 2 + (y - 1) ^ 2 = 10 := by
  sorry

end circle_equation_l284_284271


namespace sin_double_angle_l284_284391

variable (α : ℝ)

def condition : Prop :=
  sin α + 3 * cos α = 0

theorem sin_double_angle (h : condition α) : sin (2 * α) = -3 / 5 := 
by
  sorry

end sin_double_angle_l284_284391


namespace min_value_of_expr_l284_284902

noncomputable def real.min_value_expr (x y z : ℝ) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expr :
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    real.min_value_expr x y z = 4 * (real.sqrt (real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expr_l284_284902


namespace length_of_segment_MN_l284_284857

/-- Define a point in a 3D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Define the distance formula between two points in 3D space -/
def distance (p q : Point) : ℝ :=
  real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Define the symmetry in the xoz plane -/
def symmetric_with_respect_to_xoz (p : Point) : Point :=
  { x := p.x, y := -p.y, z := p.z }

theorem length_of_segment_MN :
  let N := { x := 2, y := -3, z := 5 }
  let M := symmetric_with_respect_to_xoz N
  distance M N = 6 :=
by
  sorry

end length_of_segment_MN_l284_284857


namespace largest_angle_measure_l284_284586

noncomputable def largestAngle (v : ℝ) (h : v > 2 / 3) : ℝ :=
  Real.arccos (1 / (Real.sqrt ((3 * v - 2) * (3 * v + 2))))

theorem largest_angle_measure (v : ℝ) (h : v > 2/3) :
  ∀ a b c : ℝ, (a = Real.sqrt(3 * v - 2)) ∧ (b = Real.sqrt(3 * v + 2)) ∧ (c = 2 * Real.sqrt(v)) →
  ∃ C : ℝ, C = largestAngle v h := sorry

end largest_angle_measure_l284_284586


namespace Peggy_dolls_l284_284163

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l284_284163


namespace probability_of_multiple_of_four_l284_284671

theorem probability_of_multiple_of_four :
  let n := 60
  let multiples_of_4 := 15
  let prob_not_multiple_of_4 := (n - multiples_of_4) / n
  let prob_all_not_multiple_of_4_in_3_choices := prob_not_multiple_of_4 ^ 3
  let prob_at_least_one_multiple_of_4 := 1 - prob_all_not_multiple_of_4_in_3_choices
  prob_at_least_one_multiple_of_4 = 37 / 64 :=
by
  -- Let n be the total number of choices
  let n := 60
  -- Let multiples of 4 within this range be 15
  let multiples_of_4 := 15
  -- Probability that one choice is not a multiple of 4
  let prob_not_multiple_of_4 := (n - multiples_of_4 : ℚ) / n
  -- Probability that all three choices are not multiples of 4
  let prob_all_not_multiple_of_4_in_3_choices := prob_not_multiple_of_4 ^ 3
  -- Probability that at least one choice is a multiple of 4
  let prob_at_least_one_multiple_of_4 := 1 - prob_all_not_multiple_of_4_in_3_choices
  -- Final calculation
  show prob_at_least_one_multiple_of_4 = 37 / 64
  sorry

end probability_of_multiple_of_four_l284_284671


namespace total_students_calculation_l284_284159

variable (x : ℕ)
variable (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ)
variable (total_students : ℕ)
variable (remaining_jelly_beans : ℕ)

-- Defining the number of boys as per the problem's conditions
def boys (x : ℕ) : ℕ := 2 * x + 3

-- Defining the jelly beans given to girls
def jelly_beans_given_to_girls (x girls_jelly_beans : ℕ) : Prop :=
  girls_jelly_beans = 2 * x * x

-- Defining the jelly beans given to boys
def jelly_beans_given_to_boys (x boys_jelly_beans : ℕ) : Prop :=
  boys_jelly_beans = 3 * (2 * x + 3) * (2 * x + 3)

-- Defining the total jelly beans given out
def total_jelly_beans_given_out (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ) : Prop :=
  total_jelly_beans = girls_jelly_beans + boys_jelly_beans

-- Defining the total number of students
def total_students_in_class (x total_students : ℕ) : Prop :=
  total_students = x + boys x

-- Proving that the total number of students is 18 under given conditions
theorem total_students_calculation (h1 : jelly_beans_given_to_girls x girls_jelly_beans)
                                   (h2 : jelly_beans_given_to_boys x boys_jelly_beans)
                                   (h3 : total_jelly_beans_given_out girls_jelly_beans boys_jelly_beans total_jelly_beans)
                                   (h4 : total_jelly_beans - remaining_jelly_beans = 642)
                                   (h5 : remaining_jelly_beans = 3) :
                                   total_students = 18 :=
by
  sorry

end total_students_calculation_l284_284159


namespace circumcircle_area_l284_284861

-- Definitions and conditions
variables {A B C a b c : ℝ}
variables (triangle_ABC : ∀ {A B C a b c : ℝ}, (a^2 + c^2 - b^2) / (2 * a * c) + (a^2 + b^2 - c^2) / (2 * a * b) = (sin A * sin B) / (6 * sin C))

-- Given conditions
def condition1 (B : ℝ) : Prop :=
  sqrt 3 * sin B + 2 * cos (B / 2) ^ 2 = 3

def condition2 (B C : ℝ) (b c : ℝ) : Prop :=
  (cos B / b) + (cos C / c) = (sin A * sin B) / (6 * sin C)

-- Prove the area of the circumcircle of triangle ABC
theorem circumcircle_area
  (h1 : condition1 B)
  (h2 : condition2 B C b c)
  : π * (4 ^ 2) = 16 * π :=
sorry

end circumcircle_area_l284_284861


namespace smallest_possible_sum_of_products_l284_284705

theorem smallest_possible_sum_of_products :
  ∃ (b : Fin 100 → ℤ),
    (∀ i, b i = 1 ∨ b i = -1) ∧
    let T := ∑ i in Finset.range 100, ∑ j in Finset.range 100, if i < j then b i * b j else 0 in
    T = 22 := by
  sorry

end smallest_possible_sum_of_products_l284_284705


namespace possible_row_col_products_l284_284562

theorem possible_row_col_products :
  ∃ (A : Matrix (Fin 4) (Fin 4) ℤ), 
  (Set.ofList (List.map (fun i => (Matrix.rowProd A i)) ([0, 1, 2, 3] : List (Fin 4))) ∪ 
   Set.ofList (List.map (fun j => (Matrix.colProd A j)) ([0, 1, 2, 3] : List (Fin 4)))) = 
  {1, 5, 7, 2019, -1, -5, -7, -2019} := 
  sorry

end possible_row_col_products_l284_284562


namespace range_of_m_solution_set_non_empty_l284_284465

theorem range_of_m_solution_set_non_empty :
  {m : ℝ | ∃ x : ℝ, (m * x^2 - m * x + 1 < 0)} = {m : ℝ | m ∈ Iio 0 ∪ Ioi 4} :=
by
  sorry

end range_of_m_solution_set_non_empty_l284_284465


namespace baseball_cards_per_friend_l284_284557

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l284_284557


namespace closing_price_l284_284303

theorem closing_price
  (opening_price : ℝ)
  (increase_percentage : ℝ)
  (h_opening_price : opening_price = 15)
  (h_increase_percentage : increase_percentage = 6.666666666666665) :
  opening_price * (1 + increase_percentage / 100) = 16 :=
by
  sorry

end closing_price_l284_284303


namespace subset_intersection_exists_l284_284759

theorem subset_intersection_exists 
  (α : ℝ) 
  (hα : α < (3 - Real.sqrt 5) / 2) : 
  ∃ (n p : ℕ) (h₁ : p > 2^n * α) 
    (S T : Finset (Finset (Fin n))),
    (∀ i j, S.contains i → T.contains j → ((S i) \cap (T j) ≠ ∅) ∧ (Finset.card S = p) ∧ (Finset.card T = p)) := 
sorry

end subset_intersection_exists_l284_284759


namespace product_of_x_and_y_l284_284847

def EF : ℝ := 47
def FG (y : ℝ) : ℝ := 6 * y^2
def GH (x : ℝ) : ℝ := 3 * x + 7
def HE : ℝ := 27
def is_parallelogram (EF GH HE FG: ℝ) := EF = GH ∧ FG = HE

theorem product_of_x_and_y
  (x y : ℝ)
  (h1 : is_parallelogram EF (GH x) HE (FG y))
  (hx : 3 * x = 40)
  (hy : y^2 = 9 / 2) :
  x * y = 20 * real.sqrt 2 :=
by
  have h2 : y = real.sqrt (9 / 2) := by sorry
  have h3 : x = 40 / 3 := by sorry
  sorry

end product_of_x_and_y_l284_284847


namespace S_is_infinite_l284_284498

variable (S : Set Point)

-- Define a midpoint property for S
def is_midpoint (S : Set Point) : Prop :=
  ∀ (P : Point), P ∈ S → ∃ (A B : Point), A ∈ S ∧ B ∈ S ∧ P = (A + B) / 2

-- State the theorem to prove S is infinite
theorem S_is_infinite (h : is_midpoint S) : ¬Finite S := by
  sorry

end S_is_infinite_l284_284498


namespace minimum_perimeter_triangle_l284_284419

noncomputable theory
open Real

theorem minimum_perimeter_triangle 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_sum : A + B + C = π)
  (h_sin_cos_relation : sin A + sin C = (cos A + cos C) * sin B)
  (h_area : 1/2 * a * b * sin C = 4) : 
  a + b + c ≥ 4 + 4 * sqrt 2 :=
sorry

end minimum_perimeter_triangle_l284_284419


namespace evaluate_expression_correct_l284_284080

noncomputable def evaluate_expression (x : ℝ) (h : x ≥ 0) : ℝ :=
  real.sqrt (x^2 * real.sqrt (x^3 * real.sqrt (x^4)))

theorem evaluate_expression_correct (x : ℝ) (h : x ≥ 0) : 
  evaluate_expression x h = real.sqrt4 (x^9) :=
sorry

end evaluate_expression_correct_l284_284080


namespace area_inequality_for_alpha_l284_284475

theorem area_inequality_for_alpha
  (A B C P Q M : Point)
  (α : Real)
  (h1 : angle A = α)
  (h2 : circle inscribed_in angle A touching A B at B and A C at C)
  (h3 : tangent_line intersects_segments A B at P and A C at Q)
  (h4 : S_PAQ < S_BMC) :
  sin(α / 2)^2 + sin(α / 2) > 1 := sorry

end area_inequality_for_alpha_l284_284475


namespace find_vertex_angle_l284_284232

-- Defining the problem conditions
variables (R : ℝ) (h : ℝ) (φ : ℝ)
-- Radius R of the spheres, height h of the cone, and the vertex angle φ of the cone

-- Given conditions
def condition1 := h = 2 * R

theorem find_vertex_angle
  (condition1 : h = 2 * R)
  (touching_planes : ∀ (s1 s2 s3 : sphere) (plane : plane), touching_in_pairs s1 s2 s3 plane ∧ each_touches_cone s1 s2 s3 cone)
  (cone : cone) :
  φ = π - 4 * arctan (sqrt 3 / 2) :=
sorry

end find_vertex_angle_l284_284232


namespace ascorbic_acid_weight_l284_284618

def molecular_weight (formula : String) : ℝ :=
  if formula = "C6H8O6" then 176.12 else 0

theorem ascorbic_acid_weight : molecular_weight "C6H8O6" = 176.12 :=
by {
  sorry
}

end ascorbic_acid_weight_l284_284618


namespace loss_percentage_is_10_l284_284670

noncomputable def cost_price : ℝ := 1076.923076923077
noncomputable def gain_percentage : ℝ := 0.03
noncomputable def increased_selling_price : ℝ := 1.03 * cost_price
noncomputable def additional_amount : ℝ := 140
noncomputable def selling_price : ℝ := increased_selling_price - additional_amount

def loss_amount : ℝ := cost_price - selling_price
def loss_percentage : ℝ := (loss_amount / cost_price) * 100

theorem loss_percentage_is_10 : loss_percentage = 10 :=
sorry

end loss_percentage_is_10_l284_284670


namespace least_shaded_symmetric_l284_284108

def is_symmetric (grid : set (ℕ × ℕ)) (width height : ℕ) (vsym hsym : ℕ) : Prop :=
  ∀ (x y : ℕ × ℕ),
    if x.1 < vsym ∧ x.2 < hsym then
      (width + 1 - x.1, height + 1 - x.2) ∈ grid ∧ 
      (width + 1 - x.1, x.2) ∈ grid ∧
      (x.1, height + 1 - x.2) ∈ grid
    else ∀ (x y : ℕ × ℕ), true -- Adding a dummy else condition

def initial_shaded : set (ℕ × ℕ) :=
{(1,6), (3,2), (4,5), (6,1)}

def additional_shaded : set (ℕ × ℕ) :=
{(6,6), (1,1), (4,2), (3,5)}

/--
  Prove that the least number of additional unit squares that need to be shaded such that the resulting figure has two lines of symmetry:
  - vertical line between columns 3 and 4.
  - horizontal line between rows 3 and 4.
  In given 6x6 grid with initial shaded squares at (1,6), (3,2), (4,5), (6,1) is 4.
-/
theorem least_shaded_symmetric :
  ∀ (initial_shade additional_shaded : set (ℕ × ℕ)), 
  (∃ (required_additional : set (ℕ × ℕ)), required_additional.card = 4 ∧ 
  is_symmetric (initial_shaded ∪ required_additional) 6 6 4 4) :=
begin
  sorry
end

end least_shaded_symmetric_l284_284108


namespace number_of_sentences_with_two_words_l284_284437

theorem number_of_sentences_with_two_words :
  let word : List Char := ['Y', 'A', 'R', 'I', 'Ş', 'M', 'A'],
      n : ℕ := word.length,
      repeat_a : ℕ := 2 in
  (n.factorial / repeat_a.factorial) * (n + 1) = 20160 :=
by
  -- Definitions
  let word : List Char := ['Y', 'A', 'R', 'I', 'Ş', 'M', 'A']
  let n : ℕ := word.length
  let repeat_a : ℕ := 2
  -- Calculate number of permutations and positions for space
  have h_factorial_perm : n.factorial / repeat_a.factorial = 2520 := sorry
  have h_positions_space : n + 1 = 8 := sorry
  -- Final proof
  calc
    (n.factorial / repeat_a.factorial) * (n + 1)
        = 2520 * 8 : by rw [h_factorial_perm, h_positions_space]
    ... = 20160 : by norm_num

end number_of_sentences_with_two_words_l284_284437


namespace Merry_sold_470_apples_l284_284934

-- Define the conditions
def boxes_on_Saturday : Nat := 50
def boxes_on_Sunday : Nat := 25
def apples_per_box : Nat := 10
def boxes_left : Nat := 3

-- Define the question as the number of apples sold
theorem Merry_sold_470_apples :
  (boxes_on_Saturday - boxes_on_Sunday) * apples_per_box +
  (boxes_on_Sunday - boxes_left) * apples_per_box = 470 := by
  sorry

end Merry_sold_470_apples_l284_284934


namespace razorback_revenue_difference_l284_284966

theorem razorback_revenue_difference 
  (jersey_revenue_per_unit : ℕ)
  (tshirt_revenue_per_unit : ℕ)
  (num_tshirts_sold : ℕ)
  (num_jerseys_sold : ℕ)
  (jersey_revenue_per_unit_eq : jersey_revenue_per_unit = 210)
  (tshirt_revenue_per_unit_eq : tshirt_revenue_per_unit = 240)
  (num_tshirts_sold_eq : num_tshirts_sold = 177)
  (num_jerseys_sold_eq : num_jerseys_sold = 23) :
  ((num_tshirts_sold * tshirt_revenue_per_unit) - 
   (num_jerseys_sold * jersey_revenue_per_unit) = 37650) :=
by {
  have h1 := num_tshirts_sold_eq,
  have h2 := num_jerseys_sold_eq,
  have h3 := jersey_revenue_per_unit_eq,
  have h4 := tshirt_revenue_per_unit_eq,
  rw [h1, h2, h3, h4],
  calc (177 * 240) - (23 * 210) = 42480 - 4830 : by norm_num
                                 ... = 37650    : by norm_num,
}

end razorback_revenue_difference_l284_284966


namespace abs_neg_fraction_is_positive_l284_284198

-- Define the given negative fraction
def neg_fraction := (-1 : ℝ) / 3

-- The absolute value of the given fraction
def abs_of_neg_fraction := abs neg_fraction

-- Define the expected absolute value (correct answer)
def expected_abs_value := (1 : ℝ) / 3

-- The theorem stating that the absolute value of -1/3 is 1/3
theorem abs_neg_fraction_is_positive : abs_of_neg_fraction = expected_abs_value := by
  sorry

end abs_neg_fraction_is_positive_l284_284198


namespace distance_between_towns_is_20_62_miles_l284_284631

-- Declare the coordinates of town A and Biker Bob's displacements
structure Point :=
  (x : ℝ)
  (y : ℝ)

def initial_position : Point := ⟨0, 0⟩ -- Town A is at (0,0)

def final_position (start : Point) : Point :=
  let step1 := { start with x := start.x - 10 } -- 10 miles west
  let step2 := { step1 with y := step1.y + 5 } -- 5 miles north
  let step3 := { step2 with x := step2.x + 5 } -- 5 miles east
  { step3 with y := step3.y + 15 } -- 15 miles north

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem distance_between_towns_is_20_62_miles :
  distance initial_position (final_position initial_position) = 20.62 := by
  -- The position after all displacements
  let final_pos := final_position initial_position
  -- Using Pythagorean theorem
  have h : (final_pos.x) = -5 := rfl
  have h2 : (final_pos.y) = 20 := rfl
  calc
    distance initial_position final_pos
        = real.sqrt ((final_pos.x - 0)^2 + (final_pos.y - 0)^2) : rfl
    ... = real.sqrt ((-5)^2 + 20^2) : by rw [h, h2]
    ... = real.sqrt (25 + 400) : rfl
    ... = real.sqrt (425) : rfl
    ... = 20.62 : by norm_num

end distance_between_towns_is_20_62_miles_l284_284631


namespace projection_of_skew_lines_cannot_be_two_points_l284_284623

-- Definitions of skew lines, projection, and plane
structure Line := ...
structure Plane := ...
structure Projection := ...

-- Assuming basic definitions of skew lines and projections are provided...
def is_skew (a b : Line) : Prop := -- definition indicating two lines are skew
sorry

def projection (a : Line) (α : Plane) : Projection := -- definition of projection of a line on a plane
sorry

-- Now, state the theorem
theorem projection_of_skew_lines_cannot_be_two_points (a b : Line) (α : Plane) :
  is_skew a b →
  ¬(projection a α = projection b α ∧ projection a α = Point ∧ projection b α = Point) :=
by
  sorry

end projection_of_skew_lines_cannot_be_two_points_l284_284623


namespace equivalent_solution_eq1_eqC_l284_284620

-- Define the given equation
def eq1 (x y : ℝ) : Prop := 4 * x - 8 * y - 5 = 0

-- Define the candidate equations
def eqA (x y : ℝ) : Prop := 8 * x - 8 * y - 10 = 0
def eqB (x y : ℝ) : Prop := 8 * x - 16 * y - 5 = 0
def eqC (x y : ℝ) : Prop := 8 * x - 16 * y - 10 = 0
def eqD (x y : ℝ) : Prop := 12 * x - 24 * y - 10 = 0

-- The theorem that we need to prove
theorem equivalent_solution_eq1_eqC : ∀ x y, eq1 x y ↔ eqC x y :=
by
  sorry

end equivalent_solution_eq1_eqC_l284_284620


namespace problem1_problem2_l284_284411

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (angle_ab : ℝ) (norm_a : ℝ) (norm_b : ℝ)
variables (k : ℝ)

-- Given conditions
axiom angle_ab_def : angle_ab = Real.pi / 6
axiom norm_a_def : ∥a∥ = 2
axiom norm_b_def : ∥b∥ = Real.sqrt 3

-- Questions
-- 1. Prove that |2a - b| = sqrt(7)
theorem problem1 : ∥2 • a - b∥ = Real.sqrt 7 :=
by
  rw [norm_smul, norm_eq_abs, abs_of_nonneg, Real.sqrt_eq_rpow, rpow_nat_cast, sub_eq_add_neg, add_eq_add, sub_of_eq]
  sorry

-- 2. Prove that k = 2/3 or k = 3 satisfies ⟪k • a - b, 2 • a - k • b⟫ = 0
theorem problem2 (k : ℝ): (k = 2/3 ∨ k = 3) → ⟪k • a - b, 2 • a - k • b⟫ = 0 :=
by 
  rintro (rfl | rfl)
  all_goals 
    sorry
  

end problem1_problem2_l284_284411


namespace circle_properties_l284_284854

-- Given the polar coordinate equation of the circle
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

-- Convert to a rectangular coordinate system and prove the statements
theorem circle_properties :
  (∀ ρ θ : ℝ, polar_equation ρ θ → (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2 - 4 * (ρ * Real.cos θ) - 4 * (ρ * Real.sin θ) + 6 = 0) ∧
  ((∃ ρ θ : ℝ, polar_equation ρ θ) →
    ∀ θ : ℝ, ( let x := 2 + Real.sqrt 2 * Real.cos θ in
                let y := 2 + Real.sqrt 2 * Real.sin θ in
                (x - 2)^2 + (y - 2)^2 = 2 )) ∧
  ( (4 + 2 * Real.sin (θ + Real.pi / 4) = 6) ∧
    (P : ℝ × ℝ, P = (3, 3))) :=
by
  sorry

end circle_properties_l284_284854


namespace value_of_a_0_sum_of_coefficients_l284_284742

theorem value_of_a_0 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 * 0 - 1)^5 = a_0 → a_0 = -1 :=
by
  intro h
  rw [zero_mul, sub_self, neg_one_pow, h]
  rfl
  sorry

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 * 1 - 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 → a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1 :=
by
  intro h
  rw [two_mul, one_sub_self, one_pow, h]
  rfl
  sorry

end value_of_a_0_sum_of_coefficients_l284_284742


namespace events_independent_l284_284523

variable {Ω : Type*} 
variable {P : MeasureTheory.Measure Ω}

theorem events_independent (A B : Set Ω) 
  (hA : P A = 0 ∨ P A = 1) 
  : P (A ∩ B) = P A * P B := sorry

end events_independent_l284_284523


namespace error_percent_in_area_l284_284842

theorem error_percent_in_area
  (L W : ℝ)
  (hL : L > 0)
  (hW : W > 0) :
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 0.8 := by
  sorry

end error_percent_in_area_l284_284842


namespace product_of_divisors_of_72_l284_284365

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284365


namespace roger_initial_money_l284_284952

theorem roger_initial_money (spent_on_game : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_spent : ℕ) :
  spent_on_game = 48 →
  cost_per_toy = 3 →
  num_toys = 5 →
  total_money_spent = spent_on_game + num_toys * cost_per_toy →
  total_money_spent = 63 :=
by
  intros h_game h_toy_cost h_num_toys h_total_spent
  rw [h_game, h_toy_cost, h_num_toys] at h_total_spent
  exact h_total_spent

end roger_initial_money_l284_284952


namespace find_natural_numbers_l284_284714

def decimal_digit_one_seven (n : Nat) (k : Nat) : (0 ≤ k ∧ k < n) → Nat :=
  λ _, (10^(n - k - 1) * 7 + ((10^(n) - 10^(n - k - 1)) / 9))

theorem find_natural_numbers (n : Nat) :
  (∀ k, (0 ≤ k ∧ k < n) → Prime (decimal_digit_one_seven n k)) ↔ (n = 1 ∨ n = 2) := by
  sorry

end find_natural_numbers_l284_284714


namespace perpendicular_lines_foot_of_perpendicular_l284_284034

theorem perpendicular_lines_foot_of_perpendicular 
  (m n p : ℝ) 
  (h1 : 2 * 2 + 3 * p - 1 = 0)
  (h2 : 3 * 2 - 2 * p + n = 0)
  (h3 : - (2 / m) * (3 / 2) = -1) 
  : p - m - n = 4 := 
by
  sorry

end perpendicular_lines_foot_of_perpendicular_l284_284034


namespace compute_f_2023_l284_284442

noncomputable def f : ℕ → ℤ :=
  sorry

theorem compute_f_2023
  (h1 : ∀ n : ℕ, f n ≠ 0)
  (h2 : f 1 = 3)
  (h3 : ∀ a b : ℕ, f (a + b) = f a + f b - 2 * f (a * b - 1)) :
  ¬ computable (f 2023) :=
sorry

end compute_f_2023_l284_284442


namespace ratio_of_concentric_circles_l284_284236

theorem ratio_of_concentric_circles (R r a b c : ℝ) (hr : r < R) (hc : c ∈ ℝ):
  (π * R^2 = a / (b + c) * (π * R^2 - π * r^2)) →
  (R / r) = (Real.sqrt a) / (Real.sqrt (b + c - a)) :=
by
  sorry

end ratio_of_concentric_circles_l284_284236


namespace graph_shift_l284_284607

theorem graph_shift :
  ∀ x : ℝ, (cos (2 * x - π / 3) = sin (2 * (x + π / 12))) :=
by
  intro x
  sorry

end graph_shift_l284_284607


namespace product_of_divisors_of_72_l284_284372

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284372


namespace problem_solution_l284_284393

-- Define the circle C
def circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the origin O
def origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Define the equation of the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the fixed point M on the x-axis
def fixed_point_m (x y : ℝ) : Prop := y = 0 ∧ x = 3

-- Main theorem to prove
theorem problem_solution :
  (∃ k : ℝ, k = (sqrt 3) / 3 ∨ k = -(sqrt 3) / 3 ∧ ∀ x y, 
    origin x y → line_l k x y ∧ (∃ a b : ℝ, circle a b ∧ line_l k a b ∧ line_l k a b)) ∧
  (fixed_point_m 3 0 ∧ ∀ k a b : ℝ, 
     circle a b ∧ line_l k a b → 
     let k1 := (a - 3) / (b - 0),
         k2 := (a + 3) / (b + 0) in k1 + k2 = 0) :=
sorry

end problem_solution_l284_284393


namespace center_of_circle_locus_of_midpoint_l284_284756

theorem center_of_circle {x y : ℝ} (h : x^2 + y^2 - 6 * x + 5 = 0) :
    ∃ c : ℝ × ℝ, c = (3, 0) :=
by
    sorry

theorem locus_of_midpoint {x y : ℝ} (h : ∃ k : ℝ, y = k * x ∧ x^2 + (k * x)^2 - 6 * x + 5 = 0) :
    ∀ M : ℝ × ℝ, (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1^2 + A.2^2 - 6 * A.1 + 5 = 0) ∧
                   (B.1^2 + B.2^2 - 6 * B.1 + 5 = 0) ∧ M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
    ∃ l : ℝ, l = ((x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3) :=
by
    sorry

end center_of_circle_locus_of_midpoint_l284_284756


namespace f_is_odd_f_is_decreasing_range_of_m_l284_284796

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = - f x := by
  sorry

-- Prove that f(x) is decreasing on ℝ
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  sorry

-- Prove the range of m if f(m-1) + f(2m-1) > 0
theorem range_of_m (m : ℝ) (h : f (m - 1) + f (2 * m - 1) > 0) : m < 2 / 3 := by
  sorry

end f_is_odd_f_is_decreasing_range_of_m_l284_284796


namespace voronovich_inequality_l284_284384

theorem voronovich_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6 * a * b * c ≥ a * b + b * c + c * a :=
by
  sorry

end voronovich_inequality_l284_284384


namespace total_amount_to_pay_l284_284490

theorem total_amount_to_pay (cost_earbuds cost_smartwatch : ℕ) (tax_rate_earbuds tax_rate_smartwatch : ℚ) 
  (h1 : cost_earbuds = 200) (h2 : cost_smartwatch = 300) 
  (h3 : tax_rate_earbuds = 0.15) (h4 : tax_rate_smartwatch = 0.12) : 
  (cost_earbuds + cost_earbuds * tax_rate_earbuds + cost_smartwatch + cost_smartwatch * tax_rate_smartwatch = 566) := 
by 
  sorry

end total_amount_to_pay_l284_284490


namespace trigonometric_identity_l284_284074

-- Define the main theorem
theorem trigonometric_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l284_284074


namespace vector_sum_property_l284_284805

variables {A B C P E F : Type} [plane_of_triangle : Triangle A B C]
variables (A B C P E F : Point)

-- Conditions from the problem
variables (rot_dil_A : ∀ B P, rotation_dilation_centered_at A B P)
variables (rot_dil_B : ∀ A P, rotation_dilation_centered_at B A P)
variables (move_CE : rot_dil_A B P → move C E)
variables (move_CF : rot_dil_B A P → move C F)

theorem vector_sum_property
  (h₁ : rot_dil_A B P)
  (h₂ : rot_dil_B A P)
  (h₃ : move_CE h₁)
  (h₄ : move_CF h₂) :
  vector_sum (P, E) + vector_sum (P, F) = vector_sum (P, C) := sorry

end vector_sum_property_l284_284805


namespace max_consecutive_sum_k_l284_284724

theorem max_consecutive_sum_k : 
  ∃ k n : ℕ, k = 486 ∧ 3^11 = (0 to k-1).sum + n * k := 
sorry

end max_consecutive_sum_k_l284_284724


namespace negation_equiv_l284_284581

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l284_284581


namespace sum_of_roots_tan_eqn_l284_284378

theorem sum_of_roots_tan_eqn : 
  (∑ x in {x ∈ Icc (0 : ℝ) (2 * Real.pi) | tan x * tan x - 12 * tan x + 4 = 0}, x) = 4 * Real.pi :=
by
  sorry

end sum_of_roots_tan_eqn_l284_284378


namespace product_evaluation_l284_284590

open_locale big_operators

noncomputable def product (n : ℕ) : ℚ :=
∏ k in (finset.range n).filter (λ k, k > 1),
  (1 / (k^3 - 1) + 1 / 2)

def r_condition (r : ℕ) : Prop := 
  r % 2 = 1 

def s_condition (s : ℕ) : Prop := 
  s % 2 = 1 ∧ nat.gcd r s = 1

theorem product_evaluation : 
  ∃ r s t : ℕ, r_condition r ∧ s_condition s ∧ product 100 = (r : ℚ) / (s * 2^t) ∧ (r + s + t = 3769) := 
begin
  sorry
end

end product_evaluation_l284_284590


namespace rectangle_area_error_percent_l284_284845

theorem rectangle_area_error_percent (L W : ℝ) :
  let measured_length := 1.05 * L,
      measured_width := 0.96 * W,
      actual_area := L * W,
      measured_area := measured_length * measured_width,
      error := measured_area - actual_area in
  (error / actual_area) * 100 = 0.8 := 
by
  sorry

end rectangle_area_error_percent_l284_284845


namespace evaluate_expression_l284_284329

theorem evaluate_expression : 
  let z := (\log 3 / \log 2)^2 * (\log 4 / \log 3)^2 * (\log 5 / \log 4)^2 
               * \ldots * (\log 32 / \log 31)^2
  in z = 25 :=
sorry

end evaluate_expression_l284_284329


namespace angles_terminal_side_l284_284405

theorem angles_terminal_side
  {α : ℝ}
  (h1 : α ∈ Ioo (π / 2) π)  -- α is an obtuse angle
  (h2 : Real.sin α = 1 / 2) :  -- sin α = 1 / 2
  {β : ℝ | ∃ k : ℤ, β = 5 * π / 6 + 2 * k * π} :=  -- set of angles β having the same terminal side as α
  sorry

end angles_terminal_side_l284_284405


namespace favorable_probability_l284_284519

noncomputable def probability_favorable_events (L : ℝ) : ℝ :=
  1 - (0.5 * (5 / 12 * L)^2 / (0.5 * L^2))

theorem favorable_probability (L : ℝ) (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ L)
  (h3 : 0 ≤ y) (h4 : y ≤ L)
  (h5 : 0 ≤ x + y) (h6 : x + y ≤ L)
  (h7 : x ≤ 5 / 12 * L) (h8 : y ≤ 5 / 12 * L)
  (h9 : x + y ≥ 7 / 12 * L) :
  probability_favorable_events L = 15 / 16 :=
by sorry

end favorable_probability_l284_284519


namespace tan_addition_l284_284078

variable {x y : Real}

theorem tan_addition (h1 : tan x + tan y = 40) (h2 : cot x + cot y = 50) : tan (x + y) = 200 := 
by
  sorry

end tan_addition_l284_284078


namespace fraction_left_handed_non_throwers_is_one_third_l284_284939

theorem fraction_left_handed_non_throwers_is_one_third :
  let total_players := 70
  let throwers := 31
  let right_handed := 57
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  (left_handed_non_throwers : ℝ) / non_throwers = 1 / 3 := by
  sorry

end fraction_left_handed_non_throwers_is_one_third_l284_284939


namespace karens_speed_l284_284121

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l284_284121


namespace problem_x2_plus_y2_l284_284453

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l284_284453


namespace solve_inequality_l284_284986

theorem solve_inequality (x : ℝ) : x + 1 > 3 → x > 2 := 
sorry

end solve_inequality_l284_284986


namespace find_angle_ABC_l284_284485

variable {A B C M N : Type*}
variable [inner_product_space ℝ A]
variables [has_coe_to_fun B (λ _, ℝ → A)]
variables (abc : triangle A)
variables (AM AB AC AN CN CB NBM BNM ABM NBC : ℝ)

-- Definitions given by the conditions.
def condition_1 (ABC : triangle A) (AC : ℝ) (side : ∀ (a b : ℝ), a ≤ b) : Prop :=
  side ABC.AB AC

def condition_2 (AM AB AC : ℝ) : Prop :=
  AM = AB ∧ CN = CB

def condition_3 (x : ℝ) (angle_NBM angle_ABC : ℝ) : Prop :=
  angle_NBM = x ∧ angle_ABC = 3 * x

-- Mathematically equivalent Lean statement.
theorem find_angle_ABC (ABC : triangle A) (AC AM AB CN CB : ℝ) (angle_NBM : ℝ) :
  condition_1 ABC AC (≤) →
  condition_2 AM AB AC ∧ CN CB AC →
  condition_3 angle_NBM 108 :=
sorry

end find_angle_ABC_l284_284485


namespace product_of_divisors_eq_l284_284349

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284349


namespace find_sum_abc_l284_284126

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end find_sum_abc_l284_284126


namespace convert_knocks_to_knicks_l284_284455

-- Definitions based on conditions
def knicks := ℝ
def knacks := ℝ
def knocks := ℝ

-- Given conditions
def conversion1 (k : knicks) (c : knacks) : Prop := 8 * k = 3 * c
def conversion2 (c : knacks) (n : knocks) : Prop := 4 * c = 5 * n

-- Proof statement for Lean 4
theorem convert_knocks_to_knicks (n_ : knocks) (k_ : knicks) :
  (∀ k c, conversion1 k c) → (∀ c n, conversion2 c n) → n_ = 40 → k_ = 128 / 3 → 
  ∃ k : knicks, k = k_ := by
  sorry

end convert_knocks_to_knicks_l284_284455


namespace prism_volume_l284_284560

noncomputable def volume_of_prism (x y z : ℝ) : ℝ :=
  x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 40) (h2 : x * z = 50) (h3 : y * z = 100) :
  volume_of_prism x y z = 100 * Real.sqrt 2 :=
by
  sorry

end prism_volume_l284_284560


namespace min_value_of_expression_l284_284914

noncomputable def min_of_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expression :
  ∃ x y z : ℝ, (2 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧ (z ≤ 5) ∧ min_of_expression x y z = 4 * (Real.sqrt (Real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expression_l284_284914


namespace tangent_to_CD_of_parallelogram_l284_284566

theorem tangent_to_CD_of_parallelogram
  (A B C D O : Point)
  (parallelogram : parallelogram A B C D)
  (diagonals_intersect : intersection (line A C) (line B D) = O)
  (circle_tangent_to_BC : ∃ (circle : Circle), circle.passes_through A ∧ circle.passes_through O ∧ circle.passes_through B ∧ tangent circle (line B C))
  : ∃ (circle : Circle), circle.passes_through B ∧ circle.passes_through O ∧ circle.passes_through C ∧ tangent circle (line C D) := 
sorry

end tangent_to_CD_of_parallelogram_l284_284566


namespace rational_solution_l284_284262

theorem rational_solution (p x y : ℚ) (hp_prime : prime p) (hp_mod : p % 8 = 3) :
  (p^2 * x^4 - 6 * p * x^2 + 1 = y^2) ↔ ((x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end rational_solution_l284_284262


namespace midpoint_or_midpoint_l284_284538

theorem midpoint_or_midpoint (A B C D E : Type) [Triangle ABC]
  (hD : ∃ t ∈ Icc (0 : ℝ) 1, segment_point? A B t = D)
  (hE : ∃ s ∈ Icc (0 : ℝ) 1, segment_point? C D s = E)
  (h_area_sum : area_of_triangle A C E + area_of_triangle B D E = (1/2) * area_of_triangle A B C) :
  midpoint_segment_point? D == midpoint_segment? E :=
sorry

end midpoint_or_midpoint_l284_284538


namespace find_n_and_terms_l284_284778

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem find_n_and_terms
  (n : ℕ)
  (h1 : 2^n - 128 = 128) :
  n = 8 ∧
  (∃ k x, x = 4 → binomial_coeff 8 4 * x ^ (16 - 3 * 4) = 70) ∧
  (∃ k y, y = 7 → binomial_coeff 8 3 * y ^ (16 - 3 * 3) = -56) :=
by
  sorry

end find_n_and_terms_l284_284778


namespace sum_of_ages_l284_284224

-- Definitions of John's age and father's age according to the given conditions
def John's_age := 15
def Father's_age := 2 * John's_age + 32

-- The proof problem statement
theorem sum_of_ages : John's_age + Father's_age = 77 :=
by
  -- Here we would substitute and simplify according to the given conditions
  sorry

end sum_of_ages_l284_284224


namespace rain_stop_time_on_first_day_l284_284115

-- Define the problem conditions
def raining_time_day1 (x : ℕ) : Prop :=
  let start_time := 7 * 60 -- start time in minutes
  let stop_time := start_time + x * 60 -- stop time in minutes
  stop_time = 17 * 60 -- stop at 17:00 (5:00 PM)

def total_raining_time_46_hours (x : ℕ) : Prop :=
  x + (x + 2) + 2 * (x + 2) = 46

-- Main statement
theorem rain_stop_time_on_first_day (x : ℕ) (h1 : total_raining_time_46_hours x) : raining_time_day1 x :=
  sorry

end rain_stop_time_on_first_day_l284_284115


namespace square_plus_2n_plus_3_mod_100_l284_284816

theorem square_plus_2n_plus_3_mod_100 (k : ℤ) :
  let n := 100 * k - 1 in
  (n^2 + 2*n + 3) % 100 = 2 :=
by
  sorry

end square_plus_2n_plus_3_mod_100_l284_284816


namespace proof_l284_284179

open Real Complex

noncomputable def problem_statement : Prop :=
  let z1 := (2 : ℂ) + i
  let z2 := (2 : ℂ) - i
  (z1 / z2) ^ 12 = Real.cos 11.1276 + Real.sin 11.1276 * Complex.i

theorem proof : problem_statement := by
  sorry

end proof_l284_284179


namespace max_abs_f_le_f0_f1_l284_284513

noncomputable def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

theorem max_abs_f_le_f0_f1 (a b : ℝ) (h : 0 < a) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  |f a b x| ≤ max (|f a b 0|) (|f a b 1|) :=
sorry

end max_abs_f_le_f0_f1_l284_284513


namespace find_x_l284_284895

def star (a b : ℕ) := a * b + b - a - 1

theorem find_x : ∃ x : ℕ, star 3 x = 20 ∧ x = 6 :=
by
  use 6
  split
  · show star 3 6 = 20
    calc
      star 3 6 = 3 * 6 + 6 - 3 - 1 := rfl
              _ = 18 + 6 - 3 - 1 := rfl
              _ = 24 - 3 - 1 := rfl
              _ = 21 - 1 := rfl
              _ = 20 := rfl
  · rfl

end find_x_l284_284895


namespace correct_answer_A_correct_answer_C_correct_answer_D_l284_284765

variable (f g : ℝ → ℝ)

namespace ProofProblem

-- Assume the given conditions
axiom f_eq : ∀ x, f x = 6 - deriv g x
axiom f_compl : ∀ x, f (1 - x) = 6 + deriv g (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- Proving the correct answers
theorem correct_answer_A : g 0 = 2 :=
sorry

theorem correct_answer_C : ∀ x, g (x + 4) = g x :=
sorry

theorem correct_answer_D : f 1 * g 1 + f 3 * g 3 = 24 :=
sorry

end ProofProblem

end correct_answer_A_correct_answer_C_correct_answer_D_l284_284765


namespace intersection_A_B_l284_284767

def A : Set ℝ := { x : ℝ | |x - 1| < 2 }
def B : Set ℝ := { x : ℝ | x^2 - x - 2 > 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l284_284767


namespace quotient_of_sum_of_distinct_remainders_is_zero_l284_284959

theorem quotient_of_sum_of_distinct_remainders_is_zero : 
  let squares_mod_16 := (λ n, (n^2 % 16)) in
  let distinct_remainders := {squares_mod_16 n | n in finset.range 16}.erase_duplicates in
  let m := distinct_remainders.sum in
  m / 16 = 0 := 
by
  sorry

end quotient_of_sum_of_distinct_remainders_is_zero_l284_284959


namespace g_values_range_l284_284503

def g (a b c : ℝ) : ℝ := (a / (a + 2 * b)) + (b / (b + 2 * c)) + (c / (c + 2 * a))

theorem g_values_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3 / 2 ≤ g a b c ∧ g a b c ≤ 3 :=
sorry

end g_values_range_l284_284503


namespace arith_seq_ratio_l284_284129

variable {S T : ℕ → ℚ}

-- Conditions
def is_arith_seq_sum (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * a n) / 2

def ratio_condition (S T : ℕ → ℚ) :=
  ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)

-- Main theorem
theorem arith_seq_ratio
  (a b : ℕ → ℚ)
  (h1 : is_arith_seq_sum S a)
  (h2 : is_arith_seq_sum T b)
  (h3 : ratio_condition S T)
  : a 7 / b 7 = 25 / 41 :=
sorry

end arith_seq_ratio_l284_284129


namespace number_of_ways_to_distribute_students_l284_284326

def student : Type := {name : String}

def A : student := {name := "A"}
def B : student := {name := "B"}
def C : student := {name := "C"}
def D : student := {name := "D"}

def classes : Finset (Finset student) := 
  by sorry -- This is a placeholder for creating classes so that each class has at least one student.

def valid_distributions : Finset (Finset (student → Fin (3))) := 
  by sorry -- This is another placeholder to filter out valid distributions where A and B are not in the same class.

theorem number_of_ways_to_distribute_students : valid_distributions.card = 30 :=
  by sorry

end number_of_ways_to_distribute_students_l284_284326


namespace length_of_one_side_of_regular_octagon_l284_284241

theorem length_of_one_side_of_regular_octagon
  (a b : ℕ)
  (h_pentagon : a = 16)   -- Side length of regular pentagon
  (h_total_yarn_pentagon : b = 80)  -- Total yarn for pentagon
  (hpentagon_yarn_length : 5 * a = b)  -- Total yarn condition
  (hoctagon_total_sides : 8 = 8)   -- Number of sides of octagon
  (hoctagon_side_length : 10 = b / 8)  -- Side length condition for octagon
  : 10 = 10 :=
by
  sorry

end length_of_one_side_of_regular_octagon_l284_284241


namespace simple_interest_calculation_l284_284545

-- Defining the given values
def principal : ℕ := 1500
def rate : ℕ := 7
def time : ℕ := rate -- time is the same as the rate of interest

-- Define the simple interest calculation
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Proof statement
theorem simple_interest_calculation : simple_interest principal rate time = 735 := by
  sorry

end simple_interest_calculation_l284_284545


namespace increase_by_multiplication_l284_284281

theorem increase_by_multiplication (n : ℕ) (h : n = 14) : (15 * n) - n = 196 :=
by
  -- Skip the proof
  sorry

end increase_by_multiplication_l284_284281


namespace hexagon_triangle_ratio_l284_284546

theorem hexagon_triangle_ratio (a : ℝ) (n : ℝ) (h_hex_area : n = (3 * sqrt 3 / 2) * a^2) :
  ∃ m, m = a^2 * sqrt 3 ∧ m / n = 2 / 3 :=
by
  let m := a^2 * sqrt 3
  use m
  split
  · refl
  · calc
    m / n = (a^2 * sqrt 3) / ((3 * sqrt 3 / 2) * a^2) : by rw h_hex_area
       ... = (a^2 * sqrt 3) * (2 / (3 * sqrt 3) * a^2) : by field_simp [ne_of_gt (3 * sqrt 3 / 2), ne_of_gt (sqrt 3)]
       ... = (a^2 * sqrt 3) * (2 / (3 * sqrt 3)) * (a^2 / a^2) : by field_simp
       ... = (a^2 * sqrt 3) * (2 / (3 * sqrt 3)) * 1 : by rw [div_self (pow_ne_zero 2 (ne_of_gt (show 0 < a, by sorry)))]
       ... = 2 / 3 : by field_simp [ne_of_gt (sqrt 3)]
  sorry

end hexagon_triangle_ratio_l284_284546


namespace min_value_inequality_l284_284911

theorem min_value_inequality
    (x y z : ℝ)
    (h1 : 2 ≤ x)
    (h2 : x ≤ y)
    (h3 : y ≤ z)
    (h4 : z ≤ 5) :
    (x - 2) ^ 2 + (y / x - 1) ^ 2 + (z / y - 1) ^ 2 + (5 / z - 1) ^ 2 ≥ 4 * (Real.sqrt (4 : ℝ) 5 - 1) ^ 2 :=
by
    sorry

end min_value_inequality_l284_284911


namespace negation_of_exists_gt0_and_poly_gt0_l284_284583

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l284_284583


namespace original_number_reciprocal_condition_l284_284942

theorem original_number_reciprocal_condition :
  ∃ x : ℚ, 1 + 1 / x = 9 / 4 ∧ x = 4 / 5 :=
by
  use 4 / 5
  split
  · calc
      1 + 1 / (4 / 5) = 1 + 5 / 4 := by rw div_eq_mul_inv; norm_num
                   ... = 9 / 4   := by norm_num
  · refl

end original_number_reciprocal_condition_l284_284942


namespace tan_sum_value_l284_284019

noncomputable def tan_sum 
  (α β : ℝ) 
  (h1 : α + β = π / 3) 
  (h2 : sin α * sin β = (sqrt 3 - 3) / 6) : ℝ := 
  tan α + tan β

theorem tan_sum_value 
  (α β : ℝ) 
  (h1 : α + β = π / 3) 
  (h2 : sin α * sin β = (sqrt 3 - 3) / 6) : 
  tan_sum α β h1 h2 = 3 :=
sorry

end tan_sum_value_l284_284019


namespace log8_1600_approx_4_l284_284614

theorem log8_1600_approx_4 :
  1600 ≥ 8^3 ∧ 1600 ≤ 8^4 →
  (3 : ℝ) < real.log 1600 / real.log 8 ∧ real.log 1600 / real.log 8 < 4 →
  real.log 1600 / real.log 8 ≈ 4 :=
by
  intros h1 h2
  sorry

end log8_1600_approx_4_l284_284614


namespace num_solutions_abs_eq_l284_284042

theorem num_solutions_abs_eq (B : ℤ) (hB : B = 3) : 
  { x : ℤ | |x - 2| + |x + 1| = B }.finite.to_finset.card = 4 :=
by
  sorry

end num_solutions_abs_eq_l284_284042


namespace measure_of_segment_PB_l284_284835

variable (M P C D B : Type) 
variable (DB CD : Type) 
variable (y : ℝ)
variable [MetricSpace M] 
variable [MetricSpace P] 
variable [MetricSpace C] 
variable [MetricSpace D] 
variable [MetricSpace B] 
variable [Semigroup DB] 
variable [AddCommGroup CD]

-- Midpoint of arc CDB
def is_midpoint_arc (M C D B : Type) (mid : M) : Prop := sorry

-- Perpendicular to chord DB at P
def is_perpendicular (MP : Type) (P : Type) (DB : Type) : Prop := sorry

-- Measure of chord CD
def measure_chord_CD (CD : Type) (measure : ℝ) : Prop := sorry

-- Measure of segment DP
def measure_segment_DP (DP : Type) (measure : ℝ) : Prop := sorry

-- Prove PB = y - 2
theorem measure_of_segment_PB 
  (M P C D B : Type) 
  (MP : Type) 
  (DB : Type) 
  (CD : Type) 
  (y : ℝ) 
  [MetricSpace M] 
  [MetricSpace P] 
  [MetricSpace C] 
  [MetricSpace D] 
  [MetricSpace B] 
  [Semigroup DB] 
  [AddCommGroup CD]
  (h1 : is_midpoint_arc M C D B)
  (h2 : is_perpendicular MP P DB)
  (h3 : measure_chord_CD CD y)
  (h4 : measure_segment_DP DP (y - 2)) :
  measure_segment_DP PB (y - 2) := 
sorry

end measure_of_segment_PB_l284_284835


namespace integer_solutions_count_l284_284040

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end integer_solutions_count_l284_284040


namespace trig_identity_theorem_l284_284314

noncomputable def trig_identity_proof : Prop :=
  (1 + Real.cos (Real.pi / 9)) * 
  (1 + Real.cos (2 * Real.pi / 9)) * 
  (1 + Real.cos (4 * Real.pi / 9)) * 
  (1 + Real.cos (5 * Real.pi / 9)) = 
  (1 / 2) * (Real.sin (Real.pi / 9))^4

#check trig_identity_proof

theorem trig_identity_theorem : trig_identity_proof := by
  sorry

end trig_identity_theorem_l284_284314


namespace jerry_needs_shingles_l284_284870

theorem jerry_needs_shingles :
  let roofs := 3 in
  let length := 20 in
  let width := 40 in
  let sides := 2 in
  let shingles_per_sqft := 8 in
  let area_one_side := length * width in
  let area_one_roof := area_one_side * sides in
  let total_area := area_one_roof * roofs in
  let total_shingles := total_area * shingles_per_sqft in
  total_shingles = 38400 :=
by
  sorry

end jerry_needs_shingles_l284_284870


namespace product_of_divisors_of_72_l284_284353

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284353


namespace find_k_value_range_of_m_l284_284429

noncomputable def log4 (x: ℝ) : ℝ := log x / log 4

def f (k : ℝ) (x : ℝ) : ℝ := log4 (4^x + 1) + k*x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def equation_f_has_solution (m : ℝ) (x : ℝ) : Prop :=
  f (-1/2) x = log4 (m / 2^x - 1)

theorem find_k_value (h: is_even_function (f k)) : 
  k = -1/2 :=
sorry

theorem range_of_m (m : ℝ) (h: ∃ x, equation_f_has_solution m x) : 
  1 < m :=
sorry

end find_k_value_range_of_m_l284_284429


namespace part1_union_and_intersection_part2_sufficient_condition_l284_284403

open Set

def A : Set ℝ := { x | x^2 - 3*x - 10 < 0 }
def B (m : ℝ) : Set ℝ := { x | (2 - m) ≤ x ∧ x ≤ (2 + m) ∧ m > 0 }

theorem part1_union_and_intersection (m : ℝ) (h : m = 4) :
  (A ∪ B m) = Icc (-2 : ℝ) 6 ∧ (compl A ∩ B m) = ({-2} ∪ Icc (5 : ℝ) 6) :=
begin
  rw h,
  sorry,
end

theorem part2_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) → (m ≥ 4) :=
begin
  sorry,
end

end part1_union_and_intersection_part2_sufficient_condition_l284_284403


namespace adventurers_min_count_l284_284658

open Set

variables {A : Type} [Fintype A]
variables (R E S D : Set A)

theorem adventurers_min_count 
  (hR : R.card = 13)
  (hE : E.card = 9)
  (hS : S.card = 15)
  (hD : D.card = 6)
  (hS_cond : ∀ a ∈ S, (a ∈ E ∨ a ∈ D) ∧ ¬ (a ∈ E ∧ a ∈ D))
  (hE_cond : ∀ a ∈ E, (a ∈ R ∨ a ∈ S) ∧ ¬ (a ∈ R ∧ a ∈ S)) :
  Fintype.card A = 22 :=
sorry

end adventurers_min_count_l284_284658


namespace Walmart_gift_cards_sent_l284_284865

/--
**Problem:** 
Jack initially had 6 Best Buy gift cards worth $500 each and 9 Walmart gift cards worth $200 each. 
He sent the codes for 1 Best Buy gift card and some Walmart gift cards before hanging up. 
The remaining gift cards have a total value of $3900.

**Prove:** 
The number of Walmart gift cards Jack sent before hanging up is equal to 2.
-/
theorem Walmart_gift_cards_sent
  (initial_BB_cards : ℕ) (BB_value_each : ℕ) 
  (initial_WM_cards : ℕ) (WM_value_each : ℕ)
  (BB_cards_sent : ℕ) (remaining_value : ℕ) :
  (initial_BB_cards = 6) →
  (BB_value_each = 500) →
  (initial_WM_cards = 9) →
  (WM_value_each = 200) →
  (BB_cards_sent = 1) →
  (remaining_value = 3900) →
  let initial_total_value := initial_BB_cards * BB_value_each + initial_WM_cards * WM_value_each in
  let sent_value := initial_total_value - remaining_value in
  let WM_cards_sent := (sent_value - BB_cards_sent * BB_value_each) / WM_value_each in
  WM_cards_sent = 2 := 
sorry

end Walmart_gift_cards_sent_l284_284865


namespace total_people_large_seats_is_84_l284_284195

-- Definition of the number of large seats
def large_seats : Nat := 7

-- Definition of the number of people each large seat can hold
def people_per_large_seat : Nat := 12

-- Definition of the total number of people that can ride on large seats
def total_people_large_seats : Nat := large_seats * people_per_large_seat

-- Statement that we need to prove
theorem total_people_large_seats_is_84 : total_people_large_seats = 84 := by
  sorry

end total_people_large_seats_is_84_l284_284195


namespace time_left_to_use_exerciser_l284_284872

-- Definitions based on the conditions
def total_time : ℕ := 2 * 60  -- Total time in minutes (120 minutes)
def piano_time : ℕ := 30  -- Time spent on piano
def writing_music_time : ℕ := 25  -- Time spent on writing music
def history_time : ℕ := 38  -- Time spent on history

-- The theorem statement that Joan has 27 minutes left
theorem time_left_to_use_exerciser : 
  total_time - (piano_time + writing_music_time + history_time) = 27 :=
by {
  sorry
}

end time_left_to_use_exerciser_l284_284872


namespace intersection_of_A_and_B_l284_284925

open Set

def A : Set Int := {x | x + 2 = 0}

def B : Set Int := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by
  sorry

end intersection_of_A_and_B_l284_284925


namespace cesaro_sum_of_101_term_sequence_l284_284735

noncomputable def cesaro_sum (seq : List ℝ) : ℝ :=
  let T := seq.scanl (+) 0
  T.tail.sum / seq.length

theorem cesaro_sum_of_101_term_sequence : 
  (cesaro_sum (List.cons 3 (List.cons 2 (List.range 1 100 (λ i, sorry))))) = 1191 := 
by
  -- Define the given conditions
  have h1 : (cesaro_sum (List.range 1 100 (λ i, sorry))) = 1200 := sorry
  have h2 : 2 = 2 := by rfl
  -- Assume the proof
  sorry

end cesaro_sum_of_101_term_sequence_l284_284735


namespace ratio_of_areas_l284_284837

theorem ratio_of_areas
  (A B C D E : Point)
  (h : A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (hD : D ∈ LineSegment A B)
  (hE : E ∈ LineSegment A C)
  (trapezium_BCED : ∃ P Q, is_parallel P Q BC DE ∧ is_adjacent P Q B E D C)
  (ratio_DE_BC : ∃ k, k = 3 / 5 ∧ DE = k * BC)
  : area (triangle ADE) / area (trapezium BCED) = 3 / 8 := by
sorry

end ratio_of_areas_l284_284837


namespace remainder_when_divided_by_4_l284_284821

def powers_of_3_sum : ℕ := ∑ i in Finset.range (2016 + 1), 3^i

theorem remainder_when_divided_by_4 : (powers_of_3_sum % 4) = 1 :=
by
  sorry

end remainder_when_divided_by_4_l284_284821


namespace product_of_divisors_of_72_l284_284373

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284373


namespace product_of_divisors_of_72_l284_284359

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284359


namespace find_x_squared_plus_y_squared_l284_284443

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l284_284443


namespace incorrect_statement_is_C_l284_284624

-- Definitions of the conditions
def parallelogram (A B C D : Type*) := sorry -- define parallelogram
def rhombus (A : parallelogram B C D E) := sorry -- define rhombus
def rectangle (A : parallelogram B C D E) := sorry -- define rectangle
def square (A : rhombus B) := sorry -- define square

-- Condition statements
def condition1 (p : parallelogram A B C D) : rhombus p := sorry
def condition2 (p : parallelogram A B C D) : rectangle p := sorry
def condition3 (q : parallelogram A B C D) : rhombus q = false := sorry
def condition4 (r : rhombus p) : square r := sorry

-- Theorem to prove that the incorrect statement is C
theorem incorrect_statement_is_C (p : parallelogram A B C D) (q : parallelogram A B C D) (r : rhombus p):
  condition3 q := 
sorry

end incorrect_statement_is_C_l284_284624


namespace license_plate_palindrome_probability_l284_284928

noncomputable def is_palindrome_prob : ℚ := 775 / 67600

theorem license_plate_palindrome_probability:
  is_palindrome_prob.num + is_palindrome_prob.denom = 68375 := by
  sorry

end license_plate_palindrome_probability_l284_284928


namespace siblings_total_weekly_water_l284_284227

noncomputable def Theo_daily : ℕ := 8
noncomputable def Mason_daily : ℕ := 7
noncomputable def Roxy_daily : ℕ := 9

noncomputable def daily_to_weekly (daily : ℕ) : ℕ := daily * 7

theorem siblings_total_weekly_water :
  daily_to_weekly Theo_daily + daily_to_weekly Mason_daily + daily_to_weekly Roxy_daily = 168 := by
  sorry

end siblings_total_weekly_water_l284_284227


namespace scientific_notation_correct_l284_284938

def number_in_scientific_notation (x : ℝ) : Prop :=
x = 0.0000000099 → x = 9.9 * 10 ^ (-9)

theorem scientific_notation_correct :
  number_in_scientific_notation 0.0000000099 :=
by
  sorry

end scientific_notation_correct_l284_284938


namespace baseball_cards_per_friend_l284_284556

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l284_284556


namespace count_total_balls_l284_284993

def blue_balls : ℕ := 3
def red_balls : ℕ := 2

theorem count_total_balls : blue_balls + red_balls = 5 :=
by {
  sorry
}

end count_total_balls_l284_284993


namespace min_value_inequality_l284_284910

theorem min_value_inequality
    (x y z : ℝ)
    (h1 : 2 ≤ x)
    (h2 : x ≤ y)
    (h3 : y ≤ z)
    (h4 : z ≤ 5) :
    (x - 2) ^ 2 + (y / x - 1) ^ 2 + (z / y - 1) ^ 2 + (5 / z - 1) ^ 2 ≥ 4 * (Real.sqrt (4 : ℝ) 5 - 1) ^ 2 :=
by
    sorry

end min_value_inequality_l284_284910


namespace circle_equation_and_k_range_l284_284752

theorem circle_equation_and_k_range 
  (C: Point)
  (hC: 2 * C.x - C.y - 2 = 0)
  (A: Point := ⟨2, 4⟩)
  (B: Point := ⟨3, 5⟩)
  (h_circle_A: 2^2 + 4^2 - 6 * 2 - 8 * 4 + 24 = 0)
  (h_circle_B: 3^2 + 5^2 - 6 * 3 - 8 * 5 + 24 = 0)
  (h_center: 2 * (- 3) - (- 4) - 2 = 0):
  (circle_eq : x^2 + y^2 - 6 * x - 8 * y + 24 = 0) ∧
  (k_range : ∀ {k : ℝ}, (∀ x y, (y = k * x + 3) → x^2 + y^2 - 6 * x - 8 * y + 24 = 0) ↔ (0 ≤ k ∧ k ≤ 3 / 4)) :=
by
  sorry

end circle_equation_and_k_range_l284_284752


namespace find_B_find_b_l284_284467

-- Definitions
def a (b c : ℝ) (C B : ℝ) : ℝ := b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B
def area (a c B : ℝ) : ℝ := 1/2 * a * c * Real.sin B

-- Given conditions in Lean statements
variables (a c b : ℝ)
variables (A B C : ℝ)
variable (triangle_ABC : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B)
variable (sum_ac : a + c = 6)
variable (area_eq : area a c B = 3 * Real.sqrt 3 / 2)

-- Prove the value of angle B
theorem find_B : B = Real.pi / 3 
by sorry

-- Prove the length of side b
theorem find_b : b = 3 * Real.sqrt 2 
by sorry

end find_B_find_b_l284_284467


namespace cone_height_l284_284593

theorem cone_height (lateral_area : ℝ) (l : ℝ) (h : ℝ) : lateral_area = 2 * real.pi → l = 2 → h = real.sqrt 3 :=
by
  intros h₁ h₂
  sorry

end cone_height_l284_284593


namespace product_of_divisors_of_72_l284_284352

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284352


namespace max_distance_between_car_and_motorcycle_l284_284948

def distance_car (t : ℝ) : ℝ := 40 * t

def distance_motorcycle (t : ℝ) : ℝ := 9 + 16 * t^2

def distance_between (t : ℝ) : ℝ := abs (16 * t^2 - 40 * t + 9)

theorem max_distance_between_car_and_motorcycle :
  ∃ t ∈ Icc (0:ℝ) (2:ℝ), distance_between t = 16 := by
  sorry

end max_distance_between_car_and_motorcycle_l284_284948


namespace subtraction_problem_digits_sum_l284_284110

theorem subtraction_problem_digits_sum :
  ∃ (K L M N : ℕ), K < 10 ∧ L < 10 ∧ M < 10 ∧ N < 10 ∧ 
  ((6000 + K * 100 + 0 + L) - (900 + N * 10 + 4) = 2011) ∧ 
  (K + L + M + N = 17) :=
by
  sorry

end subtraction_problem_digits_sum_l284_284110


namespace number_of_items_in_U_l284_284229

theorem number_of_items_in_U (U A B : Finset ℕ)
  (hB : B.card = 41)
  (not_A_nor_B : U.card - A.card - B.card + (A ∩ B).card = 59)
  (hAB : (A ∩ B).card = 23)
  (hA : A.card = 116) :
  U.card = 193 :=
by sorry

end number_of_items_in_U_l284_284229


namespace real_values_satisfying_inequality_l284_284337

theorem real_values_satisfying_inequality :
  ∀ x : ℝ, x ≠ 5 → (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ x ∈ set.Iic 0.76 ∪ set.Ioo 5 10.1 := by
  sorry

end real_values_satisfying_inequality_l284_284337


namespace solve_inequality_l284_284340

noncomputable def valid_x_values : set ℝ :=
  {x | x ∈ set.Icc 3.790 5 \ set.Icc 5 5 ∪ set.Icc 5 7.067}

theorem solve_inequality (x : ℝ) :
  (x ∈ valid_x_values) ↔ ((x * (x + 2) / (x - 5) ^ 2) ≥ 15) :=
sorry

end solve_inequality_l284_284340


namespace tower_surface_area_l284_284710

noncomputable def total_visible_surface_area (volumes : List ℕ) : ℕ := sorry

theorem tower_surface_area :
  total_visible_surface_area [512, 343, 216, 125, 64, 27, 8, 1] = 882 :=
sorry

end tower_surface_area_l284_284710


namespace find_b2_for_ellipse_l284_284210

theorem find_b2_for_ellipse :
  (∃ b^2 : ℝ, ∀ foci, 
    (b^2 = 14.76) ↔ 
    foci (x^2/25 + y^2/b^2 = 1) = foci (x^2/100 - y^2/64 = 1/16)) :=
by
  sorry

end find_b2_for_ellipse_l284_284210


namespace pipe_A_fill_time_l284_284652

theorem pipe_A_fill_time (B C : ℝ) (hB : B = 8) (hC : C = 14.4) (hB_not_zero : B ≠ 0) (hC_not_zero : C ≠ 0) :
  ∃ (A : ℝ), (1 / A + 1 / B = 1 / C) ∧ A = 24 :=
by
  sorry

end pipe_A_fill_time_l284_284652


namespace arithmetic_square_root_of_quotient_l284_284968

-- Definition of arithmetic square root under given conditions
def arithmetic_sqrt_of_sqrt (x : ℕ) : ℝ :=
  real.sqrt (real.sqrt x)

-- Theorem Statement
theorem arithmetic_square_root_of_quotient :
  arithmetic_sqrt_of_sqrt (16 / 81) = 2 / 3 :=
sorry

end arithmetic_square_root_of_quotient_l284_284968


namespace mandy_total_payment_l284_284153

def promotional_rate := (1 / 3 : ℝ)
def normal_price := 30
def extra_fee := 15

theorem mandy_total_payment : 
  let first_month_cost := promotional_rate * normal_price in
  let fourth_month_cost := normal_price + extra_fee in
  let regular_cost := 4 * normal_price in
  first_month_cost + fourth_month_cost + regular_cost = 175 :=
by 
  -- Define individual costs
  let first_month_cost := promotional_rate * normal_price
  let fourth_month_cost := normal_price + extra_fee
  let regular_cost := 4 * normal_price
  
  -- Simplify and calculate the total cost
  have h_first := show first_month_cost = 10, by norm_num1 [first_month_cost, promotional_rate, normal_price, mul_eq_mul_right_iff]
  have h_fourth := show fourth_month_cost = 45, by norm_num1 [fourth_month_cost, normal_price, extra_fee, add_comm]
  have h_regular := show regular_cost = 120, by norm_num1 [regular_cost, normal_price, mul_comm]

  -- Final total sum
  calc
    first_month_cost + fourth_month_cost + regular_cost
        = 10 + 45 + 120 : by rw [h_first, h_fourth, h_regular]
    ... = 175 : by norma -- Use norm_num to finalize simplification


end mandy_total_payment_l284_284153


namespace petya_payment_l284_284880

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end petya_payment_l284_284880


namespace triangle_obtuseness_l284_284094

theorem triangle_obtuseness (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
(h_sum : A + B + C = π)
(h_inequality : sin (2 * A) + sin (2 * B) < sin (2 * C)) : 
  A > π/2 ∨ B > π/2 ∨ C > π/2 :=
sorry

end triangle_obtuseness_l284_284094


namespace sin_squared_4x_properties_l284_284572

-- Define the function
def f (x : ℝ) : ℝ := sin (4 * x) ^ 2

-- The Lean statement that needs to be proven
theorem sin_squared_4x_properties : 
  (∀ x, f (-x) = f x) ∧ (∀ ε, 0 < ε → ∃ T, 0 < T ∧ ∀ x, abs (f (x + T) - f x) < ε) :=
by 
  sorry

end sin_squared_4x_properties_l284_284572


namespace bailey_total_expense_l284_284682

noncomputable def totalAmountSpentOnTowels := by
  let guestBathroomSets := 2
  let masterBathroomSets := 4
  let handTowelSets := 3
  let kitchenTowelSets := 5

  let guestBathroomPrice := 40.00
  let masterBathroomPrice := 50.00
  let handTowelPrice := 30.00
  let kitchenTowelPrice := 20.00

  let guestBathroomDiscount := 0.15
  let masterBathroomDiscount := 0.20
  let handTowelDiscount := 0.15
  let kitchenTowelDiscount := 0.10

  let salesTax := 0.08

  let discountedGuestBathroomPrice := guestBathroomPrice * (1 - guestBathroomDiscount)
  let discountedMasterBathroomPrice := masterBathroomPrice * (1 - masterBathroomDiscount)
  let discountedHandTowelPrice := handTowelPrice * (1 - handTowelDiscount)
  let discountedKitchenTowelPrice := kitchenTowelPrice * (1 - kitchenTowelDiscount)

  let totalGuestBathroomCost := guestBathroomSets * discountedGuestBathroomPrice
  let totalMasterBathroomCost := masterBathroomSets * discountedMasterBathroomPrice
  let totalHandTowelCost := handTowelSets * discountedHandTowelPrice
  let totalKitchenTowelCost := kitchenTowelSets * discountedKitchenTowelPrice

  let totalCostBeforeTax := totalGuestBathroomCost + totalMasterBathroomCost + totalHandTowelCost + totalKitchenTowelCost

  let totalSalesTax := salesTax * totalCostBeforeTax

  let totalAmount := totalCostBeforeTax + totalSalesTax

  exact totalAmount

theorem bailey_total_expense : totalAmountSpentOnTowels = 426.06 := by
  sorry

end bailey_total_expense_l284_284682


namespace sum_of_48_numbers_l284_284325

theorem sum_of_48_numbers (sheets : Finset ℕ) (numbers : ℕ → ℕ)
  (Hsheet_count : sheets.card = 24)
  (Hnumbering : ∀ n ∈ sheets, numbers (2 * n - 1) + numbers (2 * n) = 4 * n - 1) :
  ∑ n in sheets, (numbers (2 * n - 1) + numbers (2 * n)) ≠ 1990 := by
  sorry

end sum_of_48_numbers_l284_284325


namespace correct_proposition_C_l284_284387

variables {Point : Type} {Line Plane : Type}

-- Definitions for lines and planes and their relationships
variables (l m n : Line) (α β : Plane)

-- Conditions
axiom different_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom different_planes : α ≠ β

-- Parallel and subset relations
axiom line_parallel_plane : (l ∥ α) ↔ (∃ (n : Line), n ≠ l ∧ (l ∥ n) ∧ (n ⊆ α))
axiom plane_parallel_plane : (α ∥ β) ↔ ∀ (x : Point), x ∈ α → x ∈ β ∨ x ∉ β

-- Subset relation
axiom line_in_plane : (l ⊆ α) ↔ ∀ (x : Point), x ∈ l → x ∈ α

-- Main statement to prove
theorem correct_proposition_C : (α ∥ β) ∧ (l ⊆ α) → (l ∥ β) :=
by
  sorry

end correct_proposition_C_l284_284387


namespace sequences_of_lemon_recipients_l284_284157

theorem sequences_of_lemon_recipients :
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  total_sequences = 759375 :=
by
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  have h : total_sequences = 759375 := by sorry
  exact h

end sequences_of_lemon_recipients_l284_284157


namespace trajectory_E_geometric_inequality_minimum_area_l284_284007

-- Given a circle C: (x+1)^2 + y^2 = 8
def is_on_circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 8

-- Circle passing through D(1,0) and tangent to circle C with center at P
def passes_through_D (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = r^2

-- Trajectory E of P is given by the equation of an ellipse
def is_on_trajectory_E (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

theorem trajectory_E :
  ∀ x y, (is_on_circle_C x y ∧ passes_through_D x y) → is_on_trajectory_E x y :=
sorry

-- Given W(x0, y0) lies on the circle with diameter CD, which means x0^2 + y0^2 = 1
def is_on_diameter_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 = 1

-- Prove: x0^2 / 2 + y0^2 < 1
theorem geometric_inequality :
  ∀ (x0 y0 : ℝ), is_on_diameter_circle x0 y0 → (x0^2 / 2 + y0^2 < 1) :=
sorry

-- Prove: The minimum value of the area of quadrilateral QRST is 16/9
theorem minimum_area :
  ∃ (k : ℝ), let area := 4 * ((k^2 + 1)^2 / ((2*k^2 + 1) * (k^2 + 2)))
  in area = 16 / 9 :=
sorry

end trajectory_E_geometric_inequality_minimum_area_l284_284007


namespace probability_three_girls_l284_284491

theorem probability_three_girls :
  let p := 0.5 in
  (((nat.choose 6 3) * (p^3) * (p^3)) = (5 / 16)) :=
by sorry

end probability_three_girls_l284_284491


namespace boys_from_pine_l284_284881

/-- 
Given the following conditions:
1. There are 150 students at the camp.
2. There are 90 boys at the camp.
3. There are 60 girls at the camp.
4. There are 70 students from Maple High School.
5. There are 80 students from Pine High School.
6. There are 20 girls from Oak High School.
7. There are 30 girls from Maple High School.

Prove that the number of boys from Pine High School is 70.
--/
theorem boys_from_pine (total_students boys girls maple_high pine_high oak_girls maple_girls : ℕ)
  (H1 : total_students = 150)
  (H2 : boys = 90)
  (H3 : girls = 60)
  (H4 : maple_high = 70)
  (H5 : pine_high = 80)
  (H6 : oak_girls = 20)
  (H7 : maple_girls = 30) : 
  ∃ pine_boys : ℕ, pine_boys = 70 :=
by
  -- Proof goes here
  sorry

end boys_from_pine_l284_284881


namespace impossibility_of_filling_board_with_kings_l284_284533

def is_multiple_of_100 (n : ℤ) : Prop := ∃ m : ℤ, n = 100 * m

def set_A : set (ℤ × ℤ) := { p | is_multiple_of_100 p.1 ∧ is_multiple_of_100 p.2 }

noncomputable def kings_position : (ℤ × ℤ) → Prop
| (x, y) := ¬ ((x, y) ∈ set_A)

theorem impossibility_of_filling_board_with_kings
(k : ℕ) (mk_moves : (ℤ × ℤ) → (ℤ × ℤ)) :
  ¬ ∀ pos : (ℤ × ℤ), kings_position pos → kings_position (mk_moves pos) :=
sorry

end impossibility_of_filling_board_with_kings_l284_284533


namespace product_of_divisors_of_72_l284_284370

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284370


namespace poly_divisor_problem_l284_284125

open Polynomial

noncomputable def poly_is_divisible (P Q : Polynomial ℤ) : Prop :=
  ∃ R : Polynomial ℤ, Q = P * R

theorem poly_divisor_problem
  (P Q : Polynomial ℤ)
  (p q : ℕ)
  (degP: P.degree = p)
  (degQ: Q.degree = q)
  (divPQ : poly_is_divisible P Q)
  (coeff_cond: ∀ (coeffP coeffQ : ℤ), (coeffP ∈ P.coeffs → coeffP = 1 ∨ coeffP = 2002) ∧ 
                                     (coeffQ ∈ Q.coeffs → coeffQ = 1 ∨ coeffQ = 2002)): 
  (p + 1) ∣ (q + 1) :=
sorry

end poly_divisor_problem_l284_284125


namespace range_transformation_l284_284037

/-- Given the range of the function y = f(x) is [-1, 2],
prove the range of the function y = -f^2(x-1) + 2f(x-1) is [-3, 1]. -/
theorem range_transformation (f : ℝ → ℝ)
  (h : set.range f = {y : ℝ | -1 ≤ y ∧ y ≤ 2}) :
  set.range (λ x, - (f(x-1))^2 + 2 * f(x-1)) = {y : ℝ | -3 ≤ y ∧ y ≤ 1} :=
by sorry

end range_transformation_l284_284037


namespace remainder_is_24x_plus_5_l284_284376

-- Define the polynomials involved in the division
def dividend : Polynomial ℤ := Polynomial.X ^ 3
def divisor : Polynomial ℤ := Polynomial.X ^ 2 + 5 * Polynomial.X + 1

-- Statement: Prove the remainder when dividend is divided by divisor is 24x + 5
theorem remainder_is_24x_plus_5 : (dividend % divisor) = 24 * Polynomial.X + 5 :=
by
  sorry

end remainder_is_24x_plus_5_l284_284376


namespace third_smallest_three_digit_number_l284_284612

theorem third_smallest_three_digit_number (d1 d2 d3 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 8) :
  let numbers := [d1 * 100 + d2 * 10 + d3, d1 * 100 + d3 * 10 + d2, d2 * 100 + d1 * 10 + d3,
                  d2 * 100 + d3 * 10 + d1, d3 * 100 + d1 * 10 + d2, d3 * 100 + d2 * 10 + d1] in
  let sorted_numbers := numbers.sort (λ x y => x < y) in
  sorted_numbers.nth 2 = some 618 :=
by
  sorry

end third_smallest_three_digit_number_l284_284612


namespace increasing_interval_f_range_m_three_zeros_l284_284799

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - 3 * m * x^2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x + 1 - m

theorem increasing_interval_f (m : ℝ) (h : m ≠ 0) :
  (0 ≤ m → (∃ I1 I2 : set ℝ, I1 = Ioo-neg-infty 0 ∧ I2 = Ioo 2 infty ∧ 
  ∀ x, x ∈ I1 ∨ x ∈ I2 ↔ 3 * m * x^2 - 6 * m * x > 0)) ∧
  (0 > m → (∃ I : set ℝ, I = Ioo 0 2 ∧ ∀ x, x ∈ I ↔ 3 * m * x^2 - 6 * m * x > 0)) :=
sorry

theorem range_m_three_zeros (m : ℝ) (h : m > 0) :
  (∀ x, (g m 0 > 0) → (g m 2 < 0) → ((1/5) < m ∧ m < 1)) :=
sorry

#lint -- to bring up any linting issues

end increasing_interval_f_range_m_three_zeros_l284_284799


namespace dinesh_loop_l284_284703

noncomputable def number_of_pentagons (n : ℕ) : ℕ :=
  if (20 * n) % 11 = 0 then 10 else 0

theorem dinesh_loop (n : ℕ) : number_of_pentagons n = 10 :=
by sorry

end dinesh_loop_l284_284703


namespace angle_ABC_is_60_l284_284850

theorem angle_ABC_is_60 (C A E B D : Point) (t x : ℝ)
  (h1 : LiesOn C A E)
  (h2 : AB = BC ∧ BC = CD)
  (h3 : angle C D E = t)
  (h4 : angle D E C = 2 * t)
  (h5 : angle B C A = x ∧ angle B C D = x)
  (h6 : t = 20) : angle A B C = 60 :=
by
  sorry

end angle_ABC_is_60_l284_284850


namespace loss_percentage_calc_l284_284258

def cost_price : ℝ := 4500
def selling_price : ℝ := 3200
def loss_amount : ℝ := cost_price - selling_price

theorem loss_percentage_calc : (loss_amount / cost_price) * 100 ≈ 28.89 :=
by
  norm_num [cost_price, selling_price, loss_amount]
  sorry

end loss_percentage_calc_l284_284258


namespace unique_intersection_of_A_and_B_l284_284924

-- Define the sets A and B with their respective conditions
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 4 }

def B (r : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 3)^2 + (y - 4)^2 = r^2 ∧ r > 0 }

-- Define the main theorem statement
theorem unique_intersection_of_A_and_B (r : ℝ) (h : r > 0) : 
  (∃! p, p ∈ A ∧ p ∈ B r) ↔ r = 3 ∨ r = 7 :=
sorry

end unique_intersection_of_A_and_B_l284_284924


namespace half_angle_quadrant_l284_284023

-- Defining the conditions
def isInThirdQuadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

-- Prove that given α is in the third quadrant, α / 2 is either in the second or fourth quadrant
theorem half_angle_quadrant (α : ℝ) (h : isInThirdQuadrant α) :
  ∃ k : ℤ, (k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135) ∧ 
  ((even k ∧ 180 < ⟨k * 180 + 90, α / 2⟩ < 270) ∨ 
   (odd k ∧ ⟨k * 180 + 270, α / 2⟩ < 405)) := sorry

end half_angle_quadrant_l284_284023


namespace geometric_locus_l284_284700
open_locale classical

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def dist (P1 P2 : Point) : ℝ :=
real.sqrt((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

noncomputable def locus_points (M N : Point) (k a : ℝ) : set Point :=
{P : Point | (dist P M)^2 - (dist P N)^2 = k^2}

theorem geometric_locus (M N : Point) (k : ℝ) (a : ℝ) (h : dist M N = a) :
  exists (z1 z2 : ℝ) (P1 P2 : Point),
    locus_points M N k a = {P : Point | ∃ (P1 P2 : Point), 
      (P.x = P1.x ∧ P.y = P1.y ∧ P1.y = P2.y) ∧ 
      (P1.y = (M.y + N.y) / 2 + k^2 / (2 * a) ∨ P1.y = (M.y + N.y) / 2 - k^2 / (2 * a))} :=
begin
  -- Proof would go here
  sorry
end

end geometric_locus_l284_284700


namespace example_problem_l284_284414

-- Definitions and conditions
variables (f : ℝ → ℝ) (x : ℝ)
hypothesis h_deriv : ∀ x, deriv f x = deriv f (deriv f x)
hypothesis h_tangent_line : ∀ x, tangentLine f 1 = λ x, -x + 3

-- Ensure that f 1 - deriv f 1 = 3
theorem example_problem : f 1 - deriv f 1 = 3 :=
by
  sorry

end example_problem_l284_284414


namespace system1_solution_system2_solution_l284_284553

-- Define the first system of equations and its solution
theorem system1_solution (x y : ℝ) : 
    (3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)) ↔ (x = 5 ∧ y = 7) :=
sorry

-- Define the second system of equations and its solution
theorem system2_solution (x y a : ℝ) :
    (2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a) ↔ 
    (x = (7 / 16) * a ∧ y = (1 / 32) * a) :=
sorry

end system1_solution_system2_solution_l284_284553


namespace find_m_if_lines_parallel_l284_284807

theorem find_m_if_lines_parallel (m : ℝ):
  (∀ x y : ℝ, (m + 1) * x + 2 * y + 2 * m - 2 = 0 ↔ (2 * x + (m - 2) * y + 2 = 0)) →
  (m = -2) :=
begin
  sorry
end

end find_m_if_lines_parallel_l284_284807


namespace translate_sin_function_rightward_l284_284976

theorem translate_sin_function_rightward (x : ℝ) :
  (∀ x, y = √2 * sin (2 * (x - π / 12) + π / 4)) →
  y = √2 * sin (2x + π / 12) :=
by
sorry

end translate_sin_function_rightward_l284_284976


namespace length_of_AB_l284_284090
-- Import the necessary libraries

-- Define the quadratic function
def quad (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define a predicate to state that x is a root of the quadratic
def is_root (x : ℝ) : Prop := quad x = 0

-- Define the length between the intersection points
theorem length_of_AB :
  (is_root (-1)) ∧ (is_root 3) → |3 - (-1)| = 4 :=
by {
  sorry
}

end length_of_AB_l284_284090


namespace product_of_divisors_of_72_l284_284355

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284355


namespace positional_relationship_skew_l284_284980

-- Definitions of lines and intersection conditions
variables {Point : Type} [affine_space Point] (A B C D : Point) 
variables (AC BD AB CD : line Point)

-- Defining the conditions:
def intersects (l1 l2 : line Point) : Prop := ∃ P : Point, P ∈ l1 ∧ P ∈ l2

def skew (l1 l2 : line Point) : Prop := ¬ (intersects l1 l2)

-- Main theorem statement:
theorem positional_relationship_skew 
    (h1 : intersects AC AB) 
    (h2 : intersects BD AB) 
    (h3 : intersects AC CD) 
    (h4 : intersects BD CD) 
    (h5 : skew AB CD)
    : skew AC BD :=
sorry -- Proof is omitted

end positional_relationship_skew_l284_284980


namespace probability_pentagonal_face_l284_284611

open Finset

noncomputable def probability_dodecahedron_face : ℚ := 3 / 19

def is_regular_dodecahedron (G : simple_graph (fin 20)) : Prop :=
G.is_regular_degree 3

def vertices_form_face (G : simple_graph (fin 20)) (v1 v2 v3 : fin 20) : Prop :=
G.adj v1 v2 ∧ G.adj v1 v3 ∧ G.adj v2 v3

theorem probability_pentagonal_face
  (G : simple_graph (fin 20))
  (h : is_regular_dodecahedron G)
  (v1 v2 : fin 20) :
  ∃ v3, G.adj v1 v3 ∧ G.adj v2 v3 ∧ vertices_form_face G v1 v2 v3 → 
  probability_dodecahedron_face = 3 / 19 := by
  sorry

end probability_pentagonal_face_l284_284611


namespace radius_of_circle_with_given_area_l284_284374

noncomputable def π : Real := Real.pi

def circleArea (r : Real) : Real := π * r^2

theorem radius_of_circle_with_given_area :
  ∃ r : Real, circleArea r = 153.93804002589985 ∧ |r - 7| < 1e-9 :=
by
  use 7
  sorry

end radius_of_circle_with_given_area_l284_284374


namespace inequality_ab2_bc2_ca2_leq_27_div_8_l284_284127

theorem inequality_ab2_bc2_ca2_leq_27_div_8 (a b c : ℝ) (h : a ≥ b) (h1 : b ≥ c) (h2 : c ≥ 0) (h3 : a + b + c = 3) :
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end inequality_ab2_bc2_ca2_leq_27_div_8_l284_284127


namespace solve_inequality_l284_284338

noncomputable def valid_x_values : set ℝ :=
  {x | x ∈ set.Icc 3.790 5 \ set.Icc 5 5 ∪ set.Icc 5 7.067}

theorem solve_inequality (x : ℝ) :
  (x ∈ valid_x_values) ↔ ((x * (x + 2) / (x - 5) ^ 2) ≥ 15) :=
sorry

end solve_inequality_l284_284338


namespace first_term_arithmetic_sequence_median_1010_last_2015_l284_284265

theorem first_term_arithmetic_sequence_median_1010_last_2015 (a₁ : ℕ) :
  let median := 1010
  let last_term := 2015
  (a₁ + last_term = 2 * median) → a₁ = 5 :=
by
  intros
  sorry

end first_term_arithmetic_sequence_median_1010_last_2015_l284_284265


namespace missile_time_equation_l284_284559

variable (x : ℝ)

def machToMetersPerSecond := 340
def missileSpeedInMach := 26
def secondsPerMinute := 60
def distanceToTargetInKilometers := 12000
def kilometersToMeters := 1000

theorem missile_time_equation :
  (missileSpeedInMach * machToMetersPerSecond * secondsPerMinute * x) / kilometersToMeters = distanceToTargetInKilometers :=
sorry

end missile_time_equation_l284_284559


namespace sequence_remainder_2500th_term_l284_284316

theorem sequence_remainder_2500th_term :
  let a := (71 : ℕ)
  in a % 7 = 1 := by
sorry

end sequence_remainder_2500th_term_l284_284316


namespace shift_graph_right_by_pi_over_6_l284_284263

theorem shift_graph_right_by_pi_over_6 :
  ∀ (x : ℝ), 2 * sin (2 * (x - π / 6)) = 2 * sin (2 * x - π / 3) :=
by
  intro x
  sorry

end shift_graph_right_by_pi_over_6_l284_284263


namespace ratio_lead_tin_alloy_A_l284_284267

-- Define quantities
def mass_A : ℝ := 100
def mass_B : ℝ := 200

-- Alloy B tin to copper ratio
def ratio_tin_copper_B : ℝ := 2 / 5

-- Total tin in the mixture
def total_tin : ℝ := 117.5

-- Problem statement: What is the ratio of lead to tin in alloy A?
theorem ratio_lead_tin_alloy_A :
  ∃ x y : ℝ, x + (mass_B * ratio_tin_copper_B) = total_tin ∧ 
    (mass_A - x) / x = 5 / 3 := sorry

end ratio_lead_tin_alloy_A_l284_284267


namespace sum_series_equals_l284_284692

theorem sum_series_equals :
  (∑' n : ℕ, if n ≥ 2 then 1 / (n * (n + 3)) else 0) = 13 / 36 :=
by
  sorry

end sum_series_equals_l284_284692


namespace polynomial_unique_f_g_l284_284717

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_unique_f_g :
  (∀ x : ℝ, (x^2 + x + 1) * f (x^2 - x + 1) = (x^2 - x + 1) * g (x^2 + x + 1)) →
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x ∧ g x = k * x) :=
sorry

end polynomial_unique_f_g_l284_284717


namespace range_of_t_for_monotonicity_l284_284791

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  (∀ x y ∈ s, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ s, x ≤ y → f x ≥ f y)

theorem range_of_t_for_monotonicity :
  ∀ t : ℝ, t > -2 → (is_monotonic_on f (Set.Icc (-2) t) ↔ t ∈ Set.Ioo (-2:ℝ) (0:ℝ) ∨ t ∈ Set.Icc (-2:ℝ) 0) :=
by sorry

end range_of_t_for_monotonicity_l284_284791


namespace mean_absolute_temperature_correct_l284_284965

noncomputable def mean_absolute_temperature (temps : List ℝ) : ℝ :=
  (temps.map (λ x => |x|)).sum / temps.length

theorem mean_absolute_temperature_correct :
  mean_absolute_temperature [-6, -3, -3, -6, 0, 4, 3] = 25 / 7 :=
by
  sorry

end mean_absolute_temperature_correct_l284_284965


namespace cube_irregular_triangles_count_l284_284458

-- Define the vertices of the cube
def CubeVertices : Set (Fin 3 → Bool) := {
  (λ i, false), (λ i, if i = 0 then true else false),
  (λ i, if i = 1 then true else false), (λ i, if i = 2 then true else false),
  (λ i, if i ≤ 1 then true else false), (λ i, if i = 0 ∨ i = 2 then true else false),
  (λ i, if i = 1 ∨ i = 2 then true else false), (λ i, true)
}

-- Define a triangle as irregular
def is_irregular_triangle (a b c: Fin 3 → Bool) : Bool :=
  (dist a b ≠ dist a c) ∧ (dist a b ≠ dist b c) ∧ (dist a c ≠ dist b c)

-- Calculate distance between two vertices
def dist (u v : Fin 3 → Bool) : Nat := Finset.card { i : Fin 3 | u i ≠ v i }

-- Statement of the problem
theorem cube_irregular_triangles_count : ; Finset.card { (a, b, c) ∈ (CubeVertices × CubeVertices × CubeVertices) | is_irregular_triangle a b c } = 24 :=
sorry

end cube_irregular_triangles_count_l284_284458


namespace base_five_to_base_ten_modulo_seven_l284_284684

-- Define the base five number 21014_5 as the corresponding base ten conversion
def base_five_number : ℕ := 2 * 5^4 + 1 * 5^3 + 0 * 5^2 + 1 * 5^1 + 4 * 5^0

-- The equivalent base ten result
def base_ten_number : ℕ := 1384

-- Verify the base ten equivalent of 21014_5
theorem base_five_to_base_ten : base_five_number = base_ten_number :=
by
  -- The expected proof should compute the value of base_five_number
  -- and check that it equals 1384
  sorry

-- Find the modulo operation result of 1384 % 7
def modulo_seven_result : ℕ := 6

-- Verify 1384 % 7 gives 6
theorem modulo_seven : base_ten_number % 7 = modulo_seven_result :=
by
  -- The expected proof should compute 1384 % 7
  -- and check that it equals 6
  sorry

end base_five_to_base_ten_modulo_seven_l284_284684


namespace range_of_quadratic_function_l284_284222

-- Define the quadratic function
def quadractic_function (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the range of the function over the interval [-1, 2]
def function_range : set ℝ := {y | ∃ x ∈ (set.Icc (-1 : ℝ) 2), quadractic_function x = y}

-- Theorem statement asserting the range of the function over the interval is [1,5]
theorem range_of_quadratic_function : function_range = set.Icc (1 : ℝ) 5 :=
by sorry

end range_of_quadratic_function_l284_284222


namespace existence_of_points_on_AC_l284_284539

theorem existence_of_points_on_AC (A B C M : ℝ) (hAB : abs (A - B) = 2) (hBC : abs (B - C) = 1) :
  ((abs (A - M) + abs (B - M) = abs (C - M)) ↔ (M = A - 1) ∨ (M = A + 1)) :=
by
  sorry

end existence_of_points_on_AC_l284_284539


namespace find_x_y_l284_284202

theorem find_x_y (x y : ℝ) (h1 : (10 + 25 + x + y) / 4 = 20) (h2 : x * y = 156) :
  (x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12) :=
by
  sorry

end find_x_y_l284_284202


namespace election_winning_candidate_votes_l284_284603

theorem election_winning_candidate_votes (V : ℕ) 
  (h1 : V = (4 / 7) * V + 2000 + 4000) : 
  (4 / 7) * V = 8000 :=
by
  sorry

end election_winning_candidate_votes_l284_284603


namespace num_valid_matrices_l284_284811

open Matrix

-- Define the 3x3 matrix with entries 1 or -1
def is_valid_entry (a : ℤ) : Prop := a = 1 ∨ a = -1

-- Define the row and column sum conditions
def row_sum_zero (m : Matrix (Fin 3) (Fin 3) ℤ) : Prop :=
  ∀ i : Fin 3, (∑ j, m i j) = 0

def col_sum_zero (m : Matrix (Fin 3) (Fin 3) ℤ) : Prop :=
  ∀ j : Fin 3, (∑ i, m i j) = 0

-- Assemble the conditions into a single predicate
def valid_matrix (m : Matrix (Fin 3) (Fin 3) ℤ) : Prop :=
  (∀ i j, is_valid_entry (m i j)) ∧ row_sum_zero m ∧ col_sum_zero m

-- State the theorem with the correct answer
theorem num_valid_matrices : Finset.card {m : Matrix (Fin 3) (Fin 3) ℤ // valid_matrix m} = 3 :=
by sorry

end num_valid_matrices_l284_284811


namespace length_MD_angle_KMD_l284_284758

-- Part (a): Calculate the length of segment MD
theorem length_MD (A B C D M K : Type)
  [parallelogram ABCD]
  (midpoint_M_BC : M = midpoint B C)
  (AD_eq_17 : AD = 17)
  (point_K_on_AD : K ∈ segment A D)
  (BK_eq_BM : BK = BM)
  (cyclic_KBMD : cyclic KBMD) :
  MD = 8.5 :=
sorry

-- Part (b): Calculate the measure of angle KMD
theorem angle_KMD (A B C D M K : Type)
  [parallelogram ABCD]
  (midpoint_M_BC : M = midpoint B C)
  (angle_BAD_eq_46 : angle BAD = 46)
  (point_K_on_AD : K ∈ segment A D)
  (BK_eq_BM : BK = BM)
  (cyclic_KBMD : cyclic KBMD) :
  angle KMD = 48 :=
sorry

end length_MD_angle_KMD_l284_284758


namespace tangent_line_and_minimum_l284_284792

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b * x

noncomputable def g (a b x : ℝ) : ℝ := f a b x - 4 * x

theorem tangent_line_and_minimum 
  (a b : ℝ)
  (h₀ : (∂ (f a b x) / ∂ x).eval 0 = 1)
  (h₂ : (∂ (f a b x) / ∂ x).eval 2 = 1) :
  (∀ x : ℝ, f a b x = (1/3) * x^3 - x^2 + x) ∧ 
  (∀ y : ℝ, 4 * y - (f 1 1 3) - 9 = 0) ∧ 
  (∀ x : ℝ, -9 ≤ g 1 1 x ∧ x ∈ [-3, 2]) :=
by 
  intros
  sorry

end tangent_line_and_minimum_l284_284792


namespace prime_divides_g_g_plus_1_l284_284128

def g : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := g n + g (n + 1) + 1

theorem prime_divides_g_g_plus_1 (n : ℕ) (h1 : Nat.Prime n) (h2 : n > 5) :
  n ∣ (g n * (g n + 1)) := sorry

end prime_divides_g_g_plus_1_l284_284128


namespace unique_positive_integer_solution_l284_284517

theorem unique_positive_integer_solution (p : ℕ) (hp : Nat.Prime p) (hop : p % 2 = 1) :
  ∃! (x y : ℕ), x^2 + p * x = y^2 ∧ x > 0 ∧ y > 0 :=
sorry

end unique_positive_integer_solution_l284_284517


namespace product_of_divisors_of_72_l284_284363

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284363


namespace find_g_at_6_l284_284897

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 20 * x ^ 3 + 37 * x ^ 2 - 18 * x - 80

theorem find_g_at_6 : g 6 = 712 := by
  -- We apply the remainder theorem to determine the value of g(6).
  sorry

end find_g_at_6_l284_284897


namespace distance_between_cities_A_B_l284_284207

-- Define the problem parameters
def train_1_speed : ℝ := 60 -- km/hr
def train_2_speed : ℝ := 75 -- km/hr
def start_time_train_1 : ℝ := 8 -- 8 a.m.
def start_time_train_2 : ℝ := 9 -- 9 a.m.
def meeting_time : ℝ := 12 -- 12 p.m.

-- Define the times each train travels
def hours_train_1_travelled := meeting_time - start_time_train_1
def hours_train_2_travelled := meeting_time - start_time_train_2

-- Calculate the distances covered by each train
def distance_train_1_cover := train_1_speed * hours_train_1_travelled
def distance_train_2_cover := train_2_speed * hours_train_2_travelled

-- Define the total distance between cities A and B
def distance_AB := distance_train_1_cover + distance_train_2_cover

-- The theorem to be proved
theorem distance_between_cities_A_B : distance_AB = 465 := 
  by
    -- placeholder for the proof
    sorry

end distance_between_cities_A_B_l284_284207


namespace product_of_divisors_of_72_l284_284351

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284351


namespace juniors_involved_in_sports_l284_284602

theorem juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (percent_juniors_in_sports : ℝ)
  (num_juniors : ℕ)
  (num_juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : percent_juniors_in_sports = 0.70)
  (h4 : num_juniors = total_students * percent_juniors)
  (h5 : num_juniors_in_sports = num_juniors * percent_juniors_in_sports) :
  num_juniors_in_sports = 140 :=
by {
  rw [h4, h5, h1, h2, h3],
  norm_num,
  sorry
}

end juniors_involved_in_sports_l284_284602


namespace product_of_divisors_eq_l284_284348

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284348


namespace intersection_A_B_l284_284803

/-- Define set A as the set of elements that satisfy the inequality (x + 3)(2 - x) > 0 -/
def setA : set ℝ := {x | (x + 3) * (2 - x) > 0}

/-- Define set B as a specific set of numbers -/
def setB : set ℝ := {-5, -4, 0, 1, 4}

/-- The problem to prove is that the intersection of sets A and B is {0, 1} -/
theorem intersection_A_B :
  setA ∩ setB = {0, 1} :=
sorry

end intersection_A_B_l284_284803


namespace find_polynomial_solution_l284_284333

def polynomial_solution (P : ℝ[X]) : Prop :=
  ∀ x : ℝ, (x-2) * P.eval (x + 2) + (x + 2) * P.eval (x - 2) = 2 * x * P.eval x

theorem find_polynomial_solution (P : ℝ[X]) :
  polynomial_solution P →
  ∃ (a b : ℝ), P = b * (polynomial.X - 2) * polynomial.X * (polynomial.X + 2) + polynomial.C a :=
by
  sorry

end find_polynomial_solution_l284_284333


namespace range_of_f_l284_284701

noncomputable def f (x : ℝ) : ℝ := (4 - 3 * (Real.sin x) ^ 6 - 3 * (Real.cos x) ^ 6) / (Real.sin x * Real.cos x)

theorem range_of_f : 
  ∀ y, y ∈ set.range (λ x, f x) ↔ y ≥ 6 := 
sorry

end range_of_f_l284_284701


namespace other_endpoint_diameter_l284_284309

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end other_endpoint_diameter_l284_284309


namespace trig_expression_value_l284_284693

theorem trig_expression_value :
  (1 - 1 / real.cos (degree_to_radian 30)) *
  (1 + 1 / real.sin (degree_to_radian 60)) *
  (1 - 1 / real.sin (degree_to_radian 30)) *
  (1 + 1 / real.cos (degree_to_radian 60)) = -1 :=
by sorry

end trig_expression_value_l284_284693


namespace solve_inequality_l284_284188

theorem solve_inequality (x : ℝ) : x^2 - 3*x - 10 > 0 ↔ x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 5 (∞) :=
sorry

end solve_inequality_l284_284188


namespace find_a_l284_284005

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284005


namespace n_squared_divides_2n_plus_1_l284_284332

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1_l284_284332


namespace area_of_G1G2G3G4_eq_2_l284_284772

open Set Finset

variables {V : Type*} [InnerProductSpace ℝ V]

-- Define points and centroids on a quadrilateral
variable (A B C D P G1 G2 G3 G4 : V)

-- Conditions
axiom area_quadrilateral_ABCD : 9 = Real.sqrt ((A - C) ⬝ (A - C) * (B - D) ⬝ (B - D) - ((A - C) ⬝ (B - D)) ^ 2)
axiom point_P_inside : ∃ (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ a + b + c + d = 1 ∧ a • A + b • B + c • C + d • D = P
axiom centroid_G1 : G1 = (A + B + P) / 3
axiom centroid_G2 : G2 = (B + C + P) / 3
axiom centroid_G3 : G3 = (C + D + P) / 3
axiom centroid_G4 : G4 = (D + A + P) / 3

-- Theorem stating the area of quadrilateral G1G2G3G4 is 2
theorem area_of_G1G2G3G4_eq_2 : 
  ∀ (e : V), 
  e ≠ 0 → orthogonal 𝕜 e (submodule.span ℝ {A, B, C, D}) → 
  18 * (Real.sqrt ((G3 - G1) ⬝ (G3 - G1) * (G4 - G2) ⬝ (G4 - G2) - ((G3 - G1) ⬝ (G4 - G2)) ^ 2)) = (2 : ℝ) :=
by sorry

end area_of_G1G2G3G4_eq_2_l284_284772


namespace m_squared_minus_n_squared_plus_one_is_perfect_square_l284_284962

theorem m_squared_minus_n_squared_plus_one_is_perfect_square (m n : ℤ)
  (hm : m % 2 = 1) (hn : n % 2 = 1)
  (h : m^2 - n^2 + 1 ∣ n^2 - 1) :
  ∃ k : ℤ, k^2 = m^2 - n^2 + 1 :=
sorry

end m_squared_minus_n_squared_plus_one_is_perfect_square_l284_284962


namespace count_consecutive_binary_sequences_l284_284439

/- Define a sequence of length 10, consisting of zeros and ones -/
def is_binary_sequence (seq : list ℕ) : Prop :=
  seq.length = 10 ∧ ∀ x ∈ seq, x = 0 ∨ x = 1

/- Define the property of having all zeros or all ones consecutive -/
def all_zeros_or_ones_consecutive (seq : list ℕ) : Prop :=
  (∃ l : list ℕ, list.all l (λ x, x = 0) ∧ seq = l ++ list.repeat 0 (10 - l.length)) ∨
  (∃ l : list ℕ, list.all l (λ x, x = 1) ∧ seq = l ++ list.repeat 1 (10 - l.length)) ∨
  (∃ l r : list ℕ, list.all l (λ x, x = 0) ∧ list.all r (λ x, x = 1) ∧ seq = l ++ r ∧ l.length + r.length = 10)

/- State the main theorem -/
theorem count_consecutive_binary_sequences : 
  ∃ n : ℕ, n = 126 ∧ 
  (finset.card (finset.filter all_zeros_or_ones_consecutive (finset.filter is_binary_sequence (finset.range (2^10)))) = n) :=
begin
  sorry
end

end count_consecutive_binary_sequences_l284_284439


namespace ratio_proof_l284_284385

variable {x y : ℝ}

theorem ratio_proof (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 :=
by
  sorry

end ratio_proof_l284_284385


namespace minvalue_expression_l284_284140

theorem minvalue_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
    9 * z / (3 * x + y) + 9 * x / (y + 3 * z) + 4 * y / (x + z) ≥ 3 := 
by
  sorry

end minvalue_expression_l284_284140


namespace x_power_expression_l284_284522

theorem x_power_expression (x : ℝ) (h : x^3 - 3 * x = 5) : x^5 - 27 * x^2 = -22 * x^2 + 9 * x + 15 :=
by
  --proof goes here
  sorry

end x_power_expression_l284_284522


namespace identify_smart_person_l284_284604

-- Define the problem setup.
variables (people : Fin 30 → Prop) -- people(i) = true if person i is smart, false if dumb
variables (answers : Fin 30 → Bool) -- answers(i) = true if person i says their right neighbor is smart, false if dumb

-- Condition: 30 people at a round table, each is smart or dumb.
def is_smart (i : Fin 30) : Prop := people i
def is_dumb (i : Fin 30) : Prop := ¬ is_smart i

variables (F : ℕ)
variables (right_neighbor : Fin 30 → Fin 30) -- right_neighbor(i) is the right neighbor of i

-- Condition: smart person answers correctly while a dumb person may not.
def answers_correctly (i : Fin 30) : Prop :=
  is_smart people i ↔ answers i = is_smart people (right_neighbor i)

-- Condition: number of dumb people does not exceed F.
def dumb_count_le_F : Prop :=
  ∑ i in Finset.univ, cond (is_dumb people i) 1 0 ≤ F

-- Statement to prove
theorem identify_smart_person (h_answers_correctly : ∀ i, answers_correctly people answers right_neighbor i)
  (h_dumb_count : dumb_count_le_F people F) (h_F : F = 8) : ∃ i, is_smart people i :=
sorry

end identify_smart_person_l284_284604


namespace min_value_of_expression_l284_284916

noncomputable def min_of_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expression :
  ∃ x y z : ℝ, (2 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧ (z ≤ 5) ∧ min_of_expression x y z = 4 * (Real.sqrt (Real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expression_l284_284916


namespace millionth_digit_after_decimal_point_l284_284728

noncomputable def fraction := 3 / 41

def decimal_period := "07317" -- this represents the repeating part of the decimal

def millionth_digit := decimal_period.inth 4

theorem millionth_digit_after_decimal_point (n : ℕ) : 
  (1000000 % decimal_period.length = 0) -> 
  millionth_digit = 7 := 
by
  sorry

end millionth_digit_after_decimal_point_l284_284728


namespace find_g_of_3_l284_284975

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end find_g_of_3_l284_284975


namespace find_a_l284_284793

noncomputable def f (a x : ℝ) := a * Real.exp x + 2 * x^2

noncomputable def f' (a x : ℝ) := a * Real.exp x + 4 * x

theorem find_a (a : ℝ) (h : f' a 0 = 2) : a = 2 :=
by
  unfold f' at h
  simp at h
  exact h

end find_a_l284_284793


namespace circumcircle_radius_eq_l284_284651

open Real

variables (u_a u_b a b c p a1 b1 c1 : ℝ)
variables (α β : ℝ)
variables (ABC : Triangle)

def condition_1 := u_a * (tan (α / 2))⁻¹ + 2 * sqrt(u_a * u_b) + u_b * (tan(β / 2))⁻¹ = c
def condition_2 := a1 = sqrt(u_a * (tan (α / 2))⁻¹)
def condition_3 := b1 = sqrt(u_b * (tan (β / 2))⁻¹)
def condition_4 := c1 = sqrt(c)
def condition_5 := p = (a + b + c) / 2

theorem circumcircle_radius_eq (u_a u_b a1 b1 c1 p : ℝ) (α β : ℝ)
  (h1 : condition_1 u_a u_b α β c)
  (h2 : condition_2 u_a α a1)
  (h3 : condition_3 u_b β b1)
  (h4 : condition_4 c c1)
  (h5 : condition_5 a b c p) :
  circumradius (Triangle.mk a1 b1 c1) = sqrt(p) / 2 := sorry

end circumcircle_radius_eq_l284_284651


namespace magnitude_of_w_l284_284508

noncomputable def w_magnitude (s : ℝ) (w : ℂ) : ℂ :=
  if h : |s| < 3 ∧ w + 2 / w = s then
    |w|
  else
    0

theorem magnitude_of_w (s : ℝ) (w : ℂ) (h₁ : |s| < 3) (h₂ : w + 2 / w = s) :
  |w| = real.sqrt 2 :=
by
  have h : |s| < 3 ∧ w + 2 / w = s := ⟨h₁, h₂⟩
  sorry

end magnitude_of_w_l284_284508


namespace choir_average_age_solution_l284_284201

noncomputable def choir_average_age (avg_f avg_m avg_c : ℕ) (n_f n_m n_c : ℕ) : ℕ :=
  (n_f * avg_f + n_m * avg_m + n_c * avg_c) / (n_f + n_m + n_c)

def choir_average_age_problem : Prop :=
  let avg_f := 32
  let avg_m := 38
  let avg_c := 10
  let n_f := 12
  let n_m := 18
  let n_c := 5
  choir_average_age avg_f avg_m avg_c n_f n_m n_c = 32

theorem choir_average_age_solution : choir_average_age_problem := by
  sorry

end choir_average_age_solution_l284_284201


namespace shift_sin_graph_right_l284_284999

theorem shift_sin_graph_right (x : ℝ) :
  ∃ (c : ℝ), ∀ x, sin (3 * x - π / 3) = sin (3 * (x - c)) ∧ c = π / 9 :=
begin
  use π / 9,
  intro x,
  split,
  { 
    calc
      sin (3 * x - π / 3)
          = sin (3 * (x - π / 9)) : by
              nth_rewrite 0 [sub_eq_add_neg x (π / 9)],
    sorry
  },
  { refl }
end

end shift_sin_graph_right_l284_284999


namespace circle1_eq_slope_of_line_l_l284_284751

open Real

noncomputable def circle_center : ℝ × ℝ := (3, 4)

noncomputable def chord_length1 : ℝ := 2 * sqrt 5

noncomputable def line1 (x : ℝ) : Prop := x = 1

noncomputable def radius : ℝ := 
  sqrt (chord_length1^2 / 4 + (3 - 1)^2)

noncomputable def circle_equation (x y : ℝ) : Prop := 
  (x - 3)^2 + (y - 4)^2 = 9

noncomputable def point_D : ℝ × ℝ := (3, 6)

noncomputable def chord_length2 : ℝ := 4 * sqrt 2

noncomputable def line2 (k : ℝ) (x y : ℝ) : Prop := 
  y = k * (x - 3) + 6

theorem circle1_eq : ∀ x y : ℝ, 
  circle_equation x y ↔ (x - 3)^2 + (y - 4)^2 = 9 :=
by { intros, sorry }

theorem slope_of_line_l : ∃ k : ℝ, 
  (∀ x y : ℝ, line2 k x y → circle_equation x y) ∧ (k = sqrt 3 ∨ k = -sqrt 3) :=
by { sorry }

end circle1_eq_slope_of_line_l_l284_284751


namespace lowest_possible_price_l284_284255

theorem lowest_possible_price
  (manufacturer_suggested_price : ℝ := 45)
  (regular_discount_percentage : ℝ := 0.30)
  (sale_discount_percentage : ℝ := 0.20)
  (regular_discounted_price : ℝ := manufacturer_suggested_price * (1 - regular_discount_percentage))
  (final_price : ℝ := regular_discounted_price * (1 - sale_discount_percentage)) :
  final_price = 25.20 :=
by sorry

end lowest_possible_price_l284_284255


namespace roots_sum_to_product_l284_284324

theorem roots_sum_to_product (a b c : ℝ) (h : a ≠ 0) :
  let r := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a),
      s := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) in
  (r + s = r * s) ↔ b = -c :=
by
  sorry

end roots_sum_to_product_l284_284324


namespace john_running_speed_l284_284873

noncomputable def find_running_speed (x : ℝ) : Prop :=
  (12 / (3 * x + 2) + 8 / x = 2.2)

theorem john_running_speed : ∃ x : ℝ, find_running_speed x ∧ abs (x - 0.47) < 0.01 :=
by
  sorry

end john_running_speed_l284_284873


namespace max_b_sub_a_l284_284134

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => (1 / 3) * x^3 - 3 * a * x
noncomputable def g (b : ℝ) : ℝ → ℝ := fun x => x^2 + b * x

theorem max_b_sub_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∀ x ∈ set.Ioo a b, (derivative (f a) x) * (derivative (g b) x) ≤ 0) :
  (b - a) ≤ 3 / 4 :=
sorry

end max_b_sub_a_l284_284134


namespace arithmetic_sequence_property_l284_284763

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property
  (h1 : a 6 + a 8 = 10)
  (h2 : a 3 = 1)
  (property : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q)
  : a 11 = 9 :=
by
  sorry

end arithmetic_sequence_property_l284_284763


namespace incorrect_proposition_c_l284_284297

theorem incorrect_proposition_c
  (A : ∀ (l1 l2 l : Line), (l1 ∥ l2) → (l ⊥ l1) → (l ⊥ l2))
  (B : ∀ (a b c : Line), (a ∥ b) → (¬ ∃ P, P ∈ a ∧ P ∈ c) → (¬ ∃ P, P ∈ b ∧ P ∈ c) → 
    ∀ (θ : ℝ), (angle c a = θ) → (angle c b = θ))
  (D : ∀ (a : Line) (α : Plane) (P : Point), (a ∥ α) → (P ∈ α) → 
    ∃ (b : Line), (b ∥ a) ∧ (P ∈ b) ∧ (b ⊆ α)) :
  ¬ (∀ (P Q R S : Point), ¬ coplanar P Q R S → ∃ (T U V : Point), collinear T U V) :=
by
  sorry

end incorrect_proposition_c_l284_284297


namespace monotone_increasing_interval_for_shifted_function_l284_284415

variable (f : ℝ → ℝ)

-- Given definition: f(x+1) is an even function
def even_function : Prop :=
  ∀ x, f (x+1) = f (-(x+1))

-- Given condition: f(x+1) is monotonically decreasing on [0, +∞)
def monotone_decreasing_on_nonneg : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f (x+1) ≥ f (y+1)

-- Theorem to prove: the interval on which f(x-1) is monotonically increasing is (-∞, 2]
theorem monotone_increasing_interval_for_shifted_function
  (h_even : even_function f)
  (h_mono_dec : monotone_decreasing_on_nonneg f) :
  ∀ x y, x ≤ 2 → y ≤ 2 → x ≤ y → f (x-1) ≤ f (y-1) :=
by
  sorry

end monotone_increasing_interval_for_shifted_function_l284_284415


namespace max_valid_subset_cardinality_l284_284888

-- Define subset of {1..50} where sum of any two elements is not divisible by 7
def valid_subset (S : Finset ℕ) : Prop :=
  S ⊆ Finset.range 51 ∧ (∀ x y ∈ S, x ≠ y → (x + y) % 7 ≠ 0)

-- Our goal is to show there exists a subset S which satisfies the condition and has maximum cardinality of 23
theorem max_valid_subset_cardinality : ∃ S : Finset ℕ, valid_subset S ∧ S.card = 23 := 
sorry

end max_valid_subset_cardinality_l284_284888


namespace fraction_simplification_l284_284687

/-- Given x and y, under the conditions x ≠ 3y and x ≠ -3y, 
we want to prove that (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y). -/
theorem fraction_simplification (x y : ℝ) (h1 : x ≠ 3 * y) (h2 : x ≠ -3 * y) :
  (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y) :=
by
  sorry

end fraction_simplification_l284_284687


namespace min_cubes_box_identical_l284_284994

-- Define the conditions given in the problem.
def used_cubes_first_girl : ℕ := (50^2 - 34^2)
def used_cubes_second_girl : ℕ := (62^2)
def used_cubes_third_girl : ℕ := (72^2 + 4)

-- Define the minimum number of cubes in each box.
def min_cubes_per_box : ℕ := 1344

-- The proof statement.
theorem min_cubes_box_identical (cubes_per_box : ℕ) 
  (h1 : used_cubes_first_girl = cubes_per_box) 
  (h2 : used_cubes_second_girl = cubes_per_box) 
  (h3 : used_cubes_third_girl = cubes_per_box) : 
  min_cubes_per_box = cubes_per_box := 
by 
suffices h1,
suffices h2,
suffices h3,
sorry

end min_cubes_box_identical_l284_284994


namespace proof_x_squared_plus_y_squared_l284_284450

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l284_284450


namespace john_spent_on_rent_l284_284118

theorem john_spent_on_rent
  (R : ℝ)  -- Denote percentage of earnings spent on rent as R
  (hr : ∀ E : ℝ, 0 ≤ R ∧ R ≤ 1)  -- R is a valid percentage (0 ≤ R ≤ 1)
  (dishwasher_spent : R * 0.70)  -- 30% less on dishwasher
  (leftover_percentage : 1 - (R + 0.70 * R) = 1 - 0.68)  -- Leftover is 32%
  : R = 0.40 :=  -- Prove R is 40%
sorry

end john_spent_on_rent_l284_284118


namespace pyramid_volume_correct_l284_284205

noncomputable def volume_of_pyramid (d1 d2 Q : ℝ) (h : d1 > d2) : ℝ :=
  (d1 / 12) * Real.sqrt (16 * Q^2 - d1^2 * d2^2)

theorem pyramid_volume_correct (d1 d2 Q : ℝ) (h : d1 > d2) :
  ∀ (V : ℝ), V = volume_of_pyramid d1 d2 Q h → 
  V = (d1 / 12) * Real.sqrt (16 * Q^2 - d1^2 * d2^2) :=
by
  assumption

end pyramid_volume_correct_l284_284205


namespace powerful_numbers_digit_sum_eq_6_l284_284084

def is_powerful_number (n : ℕ) : Prop :=
  n < 1000 ∧ (n % 10 + (n % 10 + 1) + (n % 10 + 2) < 10) ∧     -- units place
  (n / 10 % 10 + (n / 10 % 10 + 1) + (n / 10 % 10 + 2) < 10) ∧  -- tens place
  (n / 100 % 10 + (n / 100 % 10 + 1) + (n / 100 % 10 + 2) <10)   -- hundreds place

def set_A : Finset ℕ := (Finset.range 1000).filter is_powerful_number

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem powerful_numbers_digit_sum_eq_6 :
  (set_A.image digit_sum).sum = 6 :=
by
  sorry

end powerful_numbers_digit_sum_eq_6_l284_284084


namespace P_trajectory_incenter_l284_284404

variable {α β : Type} [EuclideanSpace α β] (O A B C P : β)
variable (λ : ℝ)

-- Conditions
axiom fixed_point_O : true
axiom moving_point_P  (h : λ ∈ ℝ) : vector 𝕜 A P = λ • (vector 𝕜 A B / ∥ vector 𝕜 A B ∥ + vector 𝕜 A C / ∥ vector 𝕜 A C ∥)

theorem P_trajectory_incenter :
  point_in_trajectory P (incenter A B C) :=
sorry

end P_trajectory_incenter_l284_284404


namespace find_a4_l284_284395

def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem find_a4 (a₁ d : ℤ) (S₅ S₉ : ℤ) 
  (h₁ : arithmetic_sequence_sum 5 a₁ d = 35)
  (h₂ : arithmetic_sequence_sum 9 a₁ d = 117) :
  (a₁ + 3 * d) = 20 := 
sorry

end find_a4_l284_284395


namespace paul_taxes_and_fees_l284_284945

theorem paul_taxes_and_fees 
  (hourly_wage: ℝ) 
  (hours_worked : ℕ)
  (spent_on_gummy_bears_percentage : ℝ)
  (final_amount : ℝ)
  (gross_earnings := hourly_wage * hours_worked)
  (taxes_and_fees := gross_earnings - final_amount / (1 - spent_on_gummy_bears_percentage)):
  hourly_wage = 12.50 →
  hours_worked = 40 →
  spent_on_gummy_bears_percentage = 0.15 →
  final_amount = 340 →
  taxes_and_fees / gross_earnings = 0.20 :=
by
  intros
  sorry

end paul_taxes_and_fees_l284_284945


namespace find_reflection_point_B_l284_284283

noncomputable def point_A : ℝ × ℝ × ℝ := (-2, 8, 12)
noncomputable def point_C : ℝ × ℝ × ℝ := (4, 4, 10)
noncomputable def plane_normal : ℝ × ℝ × ℝ := (2, 1, 1)
noncomputable def plane_eq : ℝ × ℝ × ℝ → ℝ := λ p, 2*p.1 + p.2 + p.3 - 18

def line_reflection (A C : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ) (plane_eq : ℝ × ℝ × ℝ → ℝ) : ℝ × ℝ × ℝ := 
  let t := 1/4 in
  (1, 29/4, 43/4)

theorem find_reflection_point_B :
  let B := line_reflection point_A point_C plane_normal plane_eq in
  B = (1, 29/4, 43/4) := by
  sorry

end find_reflection_point_B_l284_284283


namespace problem_proof_l284_284021

variables {α β : ℝ}
def O := (0 : ℝ, 0 : ℝ)
def P1 := (Real.cos α, Real.sin α)
def P2 := (Real.cos β, Real.sin β)
def P3 := (Real.cos (α - β), Real.sin (α - β))
def A := (1 : ℝ, 0 : ℝ)

theorem problem_proof :
  (dist O P1 = dist O P3) ∧
  (dist A P3 = dist P1 P2) ∧
  (dot A P3 = dot P1 P2) :=
sorry

end problem_proof_l284_284021


namespace find_original_radius_l284_284863

noncomputable def original_radius (x : ℝ) : ℝ :=
  let r := (5 + 5 * sqrt 3) / 2
  r

theorem find_original_radius :
  ∃ x r: ℝ, 
    (2 * real.pi * (r + 5) ^ 2 - 2 * real.pi * r ^ 2 = x) ∧ 
    (real.pi * r ^ 2 * 6 - real.pi * r ^ 2 * 2 = x) ∧ 
    original_radius x = r :=
begin
  use [(20 * real.pi * r) + (50 * real.pi)],
  use ((5 + 5 * sqrt 3) / 2),
  split,
  {
    sorry, -- Steps to show the volume change by increasing the radius by 5 units
  },
  split,
  {
    sorry, -- Steps to show the volume change by increasing the height by 4 units
  },
  {
    sorry, -- Steps to show the original radius is (5 + 5 * sqrt 3) / 2
  }
end

end find_original_radius_l284_284863


namespace common_orthocenter_iff_equilateral_l284_284170

theorem common_orthocenter_iff_equilateral
  (A B C K L M : Point)
  (hK : K ∈ LineSegment A B)
  (hL : L ∈ LineSegment B C)
  (hM : M ∈ LineSegment C A)
  (h_ratio : ((dist A K) / (dist K B)) = ((dist B L) / (dist L C)) ∧ ((dist B L) / (dist L C)) = ((dist C M) / (dist M A))) :
  (orthocenter A B C = orthocenter K L M) ↔ (is_equilateral A B C) := 
sorry

end common_orthocenter_iff_equilateral_l284_284170


namespace marbles_lost_l284_284158

theorem marbles_lost (initial_marbs remaining_marbs marbles_lost : ℕ)
  (h1 : initial_marbs = 38)
  (h2 : remaining_marbs = 23)
  : marbles_lost = initial_marbs - remaining_marbs :=
by
  sorry

end marbles_lost_l284_284158


namespace linear_equation_variables_l284_284076

theorem linear_equation_variables (m n : ℤ) (h1 : 3 * m - 2 * n = 1) (h2 : n - m = 1) : m = 0 ∧ n = 1 :=
by {
  sorry
}

end linear_equation_variables_l284_284076


namespace value_of_f_f_negative_three_l284_284421

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 4 else x - 4

theorem value_of_f_f_negative_three : f (f (-3)) = -3 :=
  sorry

end value_of_f_f_negative_three_l284_284421


namespace sum_of_squares_of_coeffs_l284_284248

theorem sum_of_squares_of_coeffs :
  let expr := 5 * (x^3 - 3 * x^2 + 3) - 9 * (x^4 - 4 * x^2 + 4)
  ∑ c in (expr.expand).coeffs, c^2 = 3148 :=
by
  sorry

end sum_of_squares_of_coeffs_l284_284248


namespace translate_to_avoid_lattice_l284_284548

-- Define the plane and the concept of area
structure Figure :=
  (points : set (ℝ × ℝ))
  (area : ennreal)

-- Define the condition that a figure has an area less than 1
def has_small_area (S : Figure) : Prop :=
  S.area < 1

-- Define the translation operation
def translate (S : Figure) (v : ℝ × ℝ) : Figure :=
  { S with points := {p | ∃ q ∈ S.points, p = (q.1 + v.1, q.2 + v.2)} }

-- The main theorem statement
theorem translate_to_avoid_lattice (S : Figure) (h : has_small_area S) :
  ∃ v : ℝ × ℝ, ∀ (p : ℝ × ℝ), p ∈ (translate S v).points → (∃ i j : ℤ, p = (i, j)) → false :=
sorry

end translate_to_avoid_lattice_l284_284548


namespace transformed_function_is_final_l284_284608

-- Initial function
def initial_function (x : ℝ) : ℝ := sin (x + π / 6)

-- Translated function
def translated_function (x : ℝ) : ℝ := sin (x + 5 * π / 12)

-- Function after scaling the x-coordinates by a factor of 2
def final_function (x : ℝ) : ℝ := sin (x / 2 + 5 * π / 12)

-- Theorem statement
theorem transformed_function_is_final :
  ∀ x : ℝ, translated_function (x / 2) = final_function x := 
by
  sorry

end transformed_function_is_final_l284_284608


namespace number_in_circle_Y_l284_284226

section
variables (a b c d X Y : ℕ)

theorem number_in_circle_Y :
  a + b + X = 30 ∧
  c + d + Y = 30 ∧
  a + b + c + d = 40 ∧
  X + Y + c + b = 40 ∧
  X = 9 → Y = 11 := by
  intros h
  sorry
end

end number_in_circle_Y_l284_284226


namespace complex_expression_evaluation_l284_284783

noncomputable def z : ℂ := 1 - complex.i

theorem complex_expression_evaluation :
  (conj z + 2 * complex.i / z = 2 * complex.i) :=
by
  -- Detailed steps are skipped in the statement; add sorry to skip the proof.
  sorry

end complex_expression_evaluation_l284_284783


namespace remainder_of_M_div_45_zero_l284_284886

-- Define M as a number formed by concatenating integers from 1 to 50
def M : ℕ := -- Define the exact number here, but for now, we abstract it as M
(1 to 50).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_of_M_div_45_zero :
  M % 45 = 0 := by
  sorry

end remainder_of_M_div_45_zero_l284_284886


namespace ratio_shaded_nonshaded_area_l284_284113

theorem ratio_shaded_nonshaded_area (s : ℝ) :
  let ABC_area := (sqrt 3 / 4) * (2 * s) ^ 2
      DFE_area := (sqrt 3 / 4) * s ^ 2
      FGH_area := (sqrt 3 / 4) * (s / 2) ^ 2
      shaded_area := FGH_area + DFE_area / 2
      non_shaded_area := ABC_area - shaded_area in
  shaded_area / non_shaded_area = 3 / 13 :=
by sorry

end ratio_shaded_nonshaded_area_l284_284113


namespace find_value_of_a_l284_284598

-- Definitions for the function and conditions
def f (a : ℝ) (x : ℝ) : ℝ := a^x + Real.logBase a (x + 1)

theorem find_value_of_a (a : ℝ) :  
  (∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), a > 0 ∧ a ≠ 1 ∧ (f a 0 + f a 1 = a) → a = 1/2) :=
begin
  -- Proof logic will go here
  sorry,
end

end find_value_of_a_l284_284598


namespace min_value_problem_l284_284907

theorem min_value_problem (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2 = 4 * (Real.root 4 5 - 1)^2 := 
sorry

end min_value_problem_l284_284907


namespace candle_lighting_time_l284_284235

-- Definitions of conditions
def initial_length (ℓ : ℝ) := ℓ > 0
def burn_rate (burn_time : ℝ) (t : ℝ) (ℓ : ℝ) := ℓ - (ℓ / burn_time * t)
def candle1 (t : ℝ) (ℓ : ℝ) := burn_rate 300 t ℓ
def candle2 (t : ℝ) (ℓ : ℝ) := burn_rate 420 t ℓ

-- The condition stating that at 300 minutes, one stub is three times the length of the other
def condition (t : ℝ) (ℓ : ℝ) :=
  candle2 t ℓ = 3 * candle1 t ℓ

-- The statement to be proven
theorem candle_lighting_time (ℓ : ℝ) (hℓ : initial_length ℓ) : 
  ∃ t : ℝ, t = 240 ∧ condition t ℓ := 
begin
  existsi 240,
  split,
  { refl },
  { sorry }
end

end candle_lighting_time_l284_284235


namespace solve_for_t_l284_284440

variable (u m A j t : ℝ)

theorem solve_for_t (h : A = u^m / (2 + j)^t) : 
  t = Real.log(u^m / A) / Real.log(2 + j) :=
by
  sorry

end solve_for_t_l284_284440


namespace OB_squared_is_25_l284_284832

-- Definition of point A
def pointA : ℝ × ℝ × ℝ := (3, 7, -4)

-- Definition of point B as the projection of A onto xOz plane
def pointB : ℝ × ℝ × ℝ := (pointA.1, 0, pointA.3)

-- Definition of the squared magnitude of vector OB
def vector_OB_squared (O B : ℝ × ℝ × ℝ) : ℝ :=
  (B.1 - O.1)^2 + (B.2 - O.2)^2 + (B.3 - O.3)^2

-- Origin point O
def pointO : ℝ × ℝ × ℝ := (0, 0, 0)

-- The theorem stating that the squared magnitude of vector OB equals 25
theorem OB_squared_is_25 : vector_OB_squared pointO pointB = 25 :=
by
  sorry

end OB_squared_is_25_l284_284832


namespace volume_displaced_squared_l284_284274

-- Define the dimensions of the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Define the edge length of the tetrahedron
def tetrahedron_edge_length : ℝ := 10

-- Define the function to calculate the volume of a regular tetrahedron
noncomputable def tetrahedron_volume (s : ℝ) : ℝ :=
  (s^3 * real.sqrt 2) / 12

-- Define the submerged volume of the tetrahedron, half the total volume
noncomputable def submerged_volume : ℝ :=
  tetrahedron_volume tetrahedron_edge_length / 2

-- Define the square of the submerged volume
noncomputable def submerged_volume_squared : ℝ :=
  submerged_volume^2

-- The aim is to show that submerged_volume_squared equals 3477.78
theorem volume_displaced_squared :
  submerged_volume_squared = 3477.78 :=
by
  sorry

end volume_displaced_squared_l284_284274


namespace monotonic_increase_interval_l284_284423

noncomputable def f (x : ℝ) (φ : ℝ) := sin (2 * x + φ)

theorem monotonic_increase_interval (φ : ℝ)
  (h1 : ∀ x : ℝ, f x φ ≤ abs (f (π / 6) φ))
  (h2 : f (π / 2) φ > f π φ) :
  ∃ k : ℤ, ∀ x : ℝ, (k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) :=
sorry

end monotonic_increase_interval_l284_284423


namespace quadratic_inequality_solution_set_l284_284407

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | ax^2 - (2 + a) * x + 2 > 0} = {x | 2 / a < x ∧ x < 1} :=
sorry

end quadratic_inequality_solution_set_l284_284407


namespace range_of_a_values_l284_284702

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0

theorem range_of_a_values (a : ℝ) : range_of_a a ↔ a ≥ 1/2 :=
by
  sorry

end range_of_a_values_l284_284702


namespace worker_usual_time_l284_284242

theorem worker_usual_time (T : ℝ) (S : ℝ) (h₀ : S > 0) (h₁ : (4 / 5) * S * (T + 10) = S * T) : T = 40 :=
sorry

end worker_usual_time_l284_284242


namespace quadratic_root_a_and_other_root_quadratic_real_roots_l284_284760

-- Part (1) Statement
theorem quadratic_root_a_and_other_root (a : ℝ) (h : ¬a = 3) : 
  ∀ (x : ℝ), x^2 + a*x + a - 1 = 0 → x = -2 → False := 
begin
  intro x,
  intro h_eq,
  intro h_root,
  have subst := calc
    (-2)^2 + a*(-2) + a - 1 = 4 - 2*a + a - 1 : by ring
                         ... = 3 - a         : by ring,
  rw h_root at h_eq,
  rw subst at h_eq,
  linarith,
end

-- Part (2) Statement
theorem quadratic_real_roots (a : ℝ) : 
  ∃ x y, x^2 + a*x + a - 1 = 0 ∧ y^2 + a*y + a - 1 = 0 := 
begin
  use [x, y],
  have discrim := calc
    (a - 2)^2 = a^2 - 4*a + 4 : by ring,
  have h_dis : ∀ u: ℝ, u^2 ≥ 0 := λ u, by {
    apply pow_two_nonneg},
  use x,
  split,
  linarith only [discrim],
  use y,
  split,
  linarith only [discrim],
end

end quadratic_root_a_and_other_root_quadratic_real_roots_l284_284760


namespace contrapositive_property_l284_284563

def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def is_divisible_by_2 (n : ℤ) : Prop := n % 2 = 0

theorem contrapositive_property :
  (∀ n : ℤ, is_divisible_by_6 n → is_divisible_by_2 n) ↔ (∀ n : ℤ, ¬ is_divisible_by_2 n → ¬ is_divisible_by_6 n) :=
by
  sorry

end contrapositive_property_l284_284563


namespace problem_1_problem_2_l284_284062

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  let numerator := |C1 - C2|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  let numerator := |A * x0 + B * y0 + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem problem_1 : distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 :=
  by sorry

theorem problem_2 : distance_point_to_line 2 1 (-1) 0 2 = Real.sqrt 5 / 5 :=
  by sorry

end problem_1_problem_2_l284_284062


namespace equal_sum_of_colored_cells_l284_284515

theorem equal_sum_of_colored_cells {n : ℕ} (h : n % 2 = 0) :
  ∀ (f : Fin n → Fin n → Bool),    -- f represents the coloring function of the cells
  (∀ i : Fin n, finset.univ.filter (λ j, f i j) = n / 2) →   -- equal number of burgundy and yellow cells in each row
  (∀ j : Fin n, finset.univ.filter (λ i, f i j) = n / 2) →   -- equal number of burgundy and yellow cells in each column
  ∑ i in finset.univ, ∑ j in finset.univ, if f i j then (i.val * n + j.val + 1) else 0 =
  ∑ i in finset.univ, ∑ j in finset.univ, if ¬f i j then (i.val * n + j.val + 1) else 0 := 
sorry

end equal_sum_of_colored_cells_l284_284515


namespace sandy_correct_sums_l284_284175

theorem sandy_correct_sums :
  ∃ c i : ℤ,
  c + i = 40 ∧
  4 * c - 3 * i = 72 ∧
  c = 27 :=
by 
  sorry

end sandy_correct_sums_l284_284175


namespace candy_distribution_l284_284228

theorem candy_distribution (n k : ℕ) (h1 : 3 < n) (h2 : n < 15) (h3 : 195 - n * k = 8) : k = 17 :=
  by
    sorry

end candy_distribution_l284_284228


namespace limit_sequence_l284_284592

noncomputable def sequence_x : ℕ → ℝ
| 0       := 1
| (n + 1) := sequence_x n + 3 * real.sqrt (sequence_x n) + (n / real.sqrt (sequence_x n))

theorem limit_sequence (h : sequence_x 0 = 1 ∧ ∀ n : ℕ, sequence_x (n + 1) = sequence_x n + 
                  3 * real.sqrt (sequence_x n) + (n / real.sqrt (sequence_x n))) :
  tendsto (λ n, (n:ℝ)^2 / sequence_x n) at_top (nhds (4 / 9)) :=
sorry

end limit_sequence_l284_284592


namespace bob_distance_when_met_l284_284536

theorem bob_distance_when_met 
  (total_distance : ℕ)
  (yolanda_rate : ℕ)
  (bob_rate : ℕ)
  (head_start : ℕ)
  (meet_distance : ℕ)
  (yolanda_distance : ℕ)
  (bob_distance : ℕ) :
  total_distance = 31 →
  yolanda_rate = 1 →
  bob_rate = 2 →
  head_start = 1 →
  yolanda_distance = (t : ℕ) → yolanda_rate * (t + head_start) →
  bob_distance = (t : ℕ) → bob_rate * t →
  ∀ t, yolanda_distance t + bob_distance t = total_distance →
  bob_distance 10 = 20 :=
by
  intros
  sorry

end bob_distance_when_met_l284_284536


namespace ratio_singers_joined_second_to_remaining_first_l284_284650

-- Conditions
def total_singers : ℕ := 30
def singers_first_verse : ℕ := total_singers / 2
def remaining_after_first : ℕ := total_singers - singers_first_verse
def singers_joined_third_verse : ℕ := 10
def all_singing : ℕ := total_singers

-- Definition for singers who joined in the second verse
def singers_joined_second_verse : ℕ := all_singing - singers_joined_third_verse - singers_first_verse

-- The target proof
theorem ratio_singers_joined_second_to_remaining_first :
  (singers_joined_second_verse : ℚ) / remaining_after_first = 1 / 3 :=
by
  sorry

end ratio_singers_joined_second_to_remaining_first_l284_284650


namespace no_characteristic_values_l284_284178

-- Define the kernel function K
def K (x t : ℝ) : ℝ := sin (π * x) * cos (π * t)

-- Define the integral equation for φ
def integral_eqn (ϕ : ℝ → ℝ) (λ : ℝ) (x : ℝ) : ℝ :=
  λ * sin (π * x) * ∫ t in 0..1, cos (π * t) * ϕ t

-- Theorem stating the only solution to the equation is trivial solution for non-zero λ
theorem no_characteristic_values (ϕ : ℝ → ℝ) (λ : ℝ) (hλ : λ ≠ 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → integral_eqn ϕ λ x = ϕ x) → ϕ = (λ x, 0) :=
begin
  sorry -- Proof to be completed
end

end no_characteristic_values_l284_284178


namespace sum_of_coords_D_eq_eight_l284_284169

def point := (ℝ × ℝ)

def N : point := (4, 6)
def C : point := (10, 2)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem sum_of_coords_D_eq_eight
  (D : point)
  (h_midpoint : is_midpoint N C D) :
  D.1 + D.2 = 8 :=
by 
  sorry

end sum_of_coords_D_eq_eight_l284_284169


namespace AN_greater_than_CM_l284_284997

-- Definitions of the conditions
variables {A B C M N : Point}
variables [InABC : IsoscelesTriangle ABC A B C]
variables [Tangency : CircleThroughAWithTangencyAtM ABC A B C M]
variables [Intersection : CircleIntersectsABAtN ABC A B C N]

-- The goal to prove
theorem AN_greater_than_CM : AN > CM :=
sorry

end AN_greater_than_CM_l284_284997


namespace ellipse_problem_l284_284015

theorem ellipse_problem (a b : ℝ) (h_ab : a > b) (h_b0 : b > 0)
  (e : ℝ) (h_e : e = 1/2)
  (MF : ℝ) (FN : ℝ)
  (h_geom_mean : sqrt (MF * FN) = sqrt 3) :
  (∃ (a : ℝ), ∃ (c : ℝ), (c = e * a) ∧ (b^2 = a^2 - c^2) ∧ (a = 2) ∧ (c = 1) 
          ∧ (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) )
  ∧ (∀ (l : ℝ → ℝ) (h_l : l 0 ≠ 0) (sA sB sO : ℝ) (h_geo : sA * sB * sO > 0),
    ∃ m x1 y1 x2 y2,
    0 < m^2 ∧ m^2 < 6 ∧ m^2 ≠ 3 ∧
    (triangle_OAB_area : ℝ) (h_area : triangle_OAB_area = (1 / 3) * sqrt (3 * m^2 * (6 - m^2)))
    (h_range : 0 < triangle_OAB_area ∧ triangle_OAB_area < sqrt 3)) :=
begin
  sorry
end

end ellipse_problem_l284_284015


namespace infinite_primes_in_polynomial_l284_284143

open Nat

def is_prime (p : ℕ) := ∃ d, d > 1 ∧ d < p ∧ p % d = 0

def divides (a b : ℕ) := ∃ k, b = a * k

theorem infinite_primes_in_polynomial:
  ∀ f : ℤ[X], ¬ is_constant f → ∃ (p : ℕ), is_prime p ∧ ∃ n : ℕ, divides p (eval n f) :=
begin
  sorry
end

end infinite_primes_in_polynomial_l284_284143


namespace calc_eq_neg_ten_thirds_l284_284544

theorem calc_eq_neg_ten_thirds :
  (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8) + (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12) = -10 / 3 := by 
sorry

end calc_eq_neg_ten_thirds_l284_284544


namespace locus_of_midpoints_is_circle_segment_l284_284996

open Classical

noncomputable theory

variables {O P : Point} (circle : Circle)

def midpoint (A B: Point) : Point :=
(Point.mk ((A.1 + B.1) / 2) ((A.2 + B.2) / 2))

def line_through (P : Point) (A B: Point) : Prop :=
∃ m b, A ∈ line.mk m b ∧ B ∈ line.mk m b ∧ P ∈ line.mk m b

def perpendicular (P O M: Point) : Prop :=
∠ P M O = 90 -- assuming ∠ is the angle function simplified for clarity

def locus_midpoints (O P: Point) (circle : Circle) : Set Point :=
{ M : Point | ∃ A B : Point, A∈circle ∧ B∈circle ∧ line_through P A B ∧ M = midpoint A B ∧ perpendicular P O M }

theorem locus_of_midpoints_is_circle_segment (O P: Point) (circle : Circle) :
  locus_midpoints O P circle = { M : Point | M ∈ circle_with_diameter O P ∧ M ∈ circle } :=
sorry

end locus_of_midpoints_is_circle_segment_l284_284996


namespace product_of_primes_l284_284549

theorem product_of_primes (n : ℕ) (h : n ≠ 0) : ∃ (f : ℕ → ℕ), (∀ (i), f i ∈ prime ∧ n = ∏ i in finset.range (n+1), f i) := 
sorry

end product_of_primes_l284_284549


namespace part1_k_real_part2_find_k_l284_284394

-- Part 1: Discriminant condition
theorem part1_k_real (k : ℝ) (h : x^2 + (2*k - 1)*x + k^2 - 1 = 0) : k ≤ 5 / 4 :=
by
  sorry

-- Part 2: Given additional conditions, find k
theorem part2_find_k (x1 x2 k : ℝ) (h_eq : x^2 + (2 * k - 1) * x + k^2 - 1 = 0)
  (h1 : x1 + x2 = 1 - 2 * k) (h2 : x1 * x2 = k^2 - 1) (h3 : x1^2 + x2^2 = 16 + x1 * x2) : k = -2 :=
by
  sorry

end part1_k_real_part2_find_k_l284_284394


namespace triangle_side_length_b_l284_284112

/-
In a triangle ABC with angles such that ∠C = 4∠A, and sides such that a = 35 and c = 64, prove that the length of side b is 140 * cos²(A).
-/
theorem triangle_side_length_b (A C : ℝ) (a c : ℝ) (hC : C = 4 * A) (ha : a = 35) (hc : c = 64) :
  ∃ (b : ℝ), b = 140 * (Real.cos A) ^ 2 :=
by
  sorry

end triangle_side_length_b_l284_284112


namespace lions_in_first_group_l284_284085

-- Auxiliary definitions to capture problem statements
def killing_rate (lions deer minutes : ℕ) : ℕ :=
  deer / minutes / lions

-- The given conditions as Lean definitions
def first_group_rate : ℕ :=
  killing_rate x 10 10

def second_group_rate : ℕ :=
  killing_rate 100 100 10

-- Problem statement: Prove that the number of lions in the first group is 10
theorem lions_in_first_group (x : ℕ) :
  first_group_rate = second_group_rate → x = 10 :=
by
  unfold first_group_rate
  unfold second_group_rate
  unfold killing_rate
  sorry

end lions_in_first_group_l284_284085


namespace millionth_digit_after_decimal_point_l284_284729

noncomputable def fraction := 3 / 41

def decimal_period := "07317" -- this represents the repeating part of the decimal

def millionth_digit := decimal_period.inth 4

theorem millionth_digit_after_decimal_point (n : ℕ) : 
  (1000000 % decimal_period.length = 0) -> 
  millionth_digit = 7 := 
by
  sorry

end millionth_digit_after_decimal_point_l284_284729


namespace bugs_initial_count_l284_284683

theorem bugs_initial_count (B : ℝ) 
  (h_spray : ∀ (b : ℝ), b * 0.8 = b * (4 / 5)) 
  (h_spiders : ∀ (s : ℝ), s * 7 = 12 * 7) 
  (h_initial_spray_spiders : ∀ (b : ℝ), b * 0.8 - (12 * 7) = 236) 
  (h_final_bugs : 320 / 0.8 = 400) : 
  B = 400 :=
sorry

end bugs_initial_count_l284_284683


namespace number_of_rectangles_required_l284_284457

theorem number_of_rectangles_required
  (width : ℝ) (area : ℝ) (total_length : ℝ) (length : ℝ)
  (H1 : width = 42) (H2 : area = 1638) (H3 : total_length = 390) (H4 : length = area / width)
  : (total_length / length) = 10 := 
sorry

end number_of_rectangles_required_l284_284457


namespace S_infinite_l284_284500

noncomputable theory

-- Definition: S is a set of points.
variable (S : Set Point)

-- Condition: every point in S is the midpoint of a segment whose endpoints are in S.
axiom midpoint_in_S (P : Point) (hP : P ∈ S) : ∃ A B : Point, A ∈ S ∧ B ∈ S ∧ P = midpoint A B

-- Goal: Show that S is infinite.
theorem S_infinite : ¬Finite S :=
sorry

end S_infinite_l284_284500


namespace number_of_triangles_l284_284049

/- Define the setup -/
variable {P : Point} (r_A r_B r_C : ℝ)
variable (h_le1 : r_A ≤ r_B) (h_le2 : r_B ≤ r_C)

/- Define the theorem statement -/
theorem number_of_triangles (P : Point) (r_A r_B r_C : ℝ)
  (h_le1 : r_A ≤ r_B) (h_le2 : r_B ≤ r_C) :
  (r_A + r_B < r_C → 0) ∧ 
  (r_A + r_B = r_C → 1) ∧ 
  (r_A + r_B > r_C → 2) := sorry

end number_of_triangles_l284_284049


namespace greatest_x_l284_284151

theorem greatest_x (x : ℤ) (h : (3.71 * 10 ^ x) / (6.52 * 10 ^ (x - 3)) < 10230) : x ≤ 1 :=
by { sorry }

end greatest_x_l284_284151


namespace ratio_of_unit_prices_l284_284304

variable (v p : ℝ) -- volume and price of Brand W soda
constant BrandZ_volume : ℝ := 1.25 * v
constant BrandZ_price : ℝ := 0.85 * p
-- Unit prices
def UnitPrice_BrandW := p / v
def UnitPrice_BrandZ : ℝ := BrandZ_price / BrandZ_volume

theorem ratio_of_unit_prices : 
  UnitPrice_BrandZ / UnitPrice_BrandW = (17 / 25) := 
by
  sorry

end ratio_of_unit_prices_l284_284304


namespace lock_settings_are_5040_l284_284293

def num_unique_settings_for_lock : ℕ := 10 * 9 * 8 * 7

theorem lock_settings_are_5040 : num_unique_settings_for_lock = 5040 :=
by
  sorry

end lock_settings_are_5040_l284_284293


namespace product_of_divisors_of_72_l284_284362

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284362


namespace problem_statement_l284_284389

open Real

theorem problem_statement (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (1/b) = 1) (hn_pos : 0 < n) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry -- proof to be provided

end problem_statement_l284_284389


namespace image_of_neg2_3_preimages_2_neg3_l284_284505

variables {A B : Type}
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Prove that the image of (-2, 3) under f is (1, -6)
theorem image_of_neg2_3 : f (-2) 3 = (1, -6) := sorry

-- Find the preimages of (2, -3) under f
def preimages_of_2_neg3 (p : ℤ × ℤ) : Prop := f p.1 p.2 = (2, -3)

theorem preimages_2_neg3 : preimages_of_2_neg3 (-1, 3) ∧ preimages_of_2_neg3 (3, -1) := sorry

end image_of_neg2_3_preimages_2_neg3_l284_284505


namespace p_sufficient_not_necessary_for_q_l284_284017

-- Define the propositions p and q
def is_ellipse (m : ℝ) : Prop := (1 / 4 < m) ∧ (m < 1)
def is_hyperbola (m : ℝ) : Prop := (0 < m) ∧ (m < 1)

-- Define the theorem to prove the relationship between p and q
theorem p_sufficient_not_necessary_for_q (m : ℝ) :
  (is_ellipse m → is_hyperbola m) ∧ ¬(is_hyperbola m → is_ellipse m) :=
sorry

end p_sufficient_not_necessary_for_q_l284_284017


namespace find_C_coordinates_l284_284014

noncomputable def point (coord : ℝ × ℝ) := coord

def A := point (2, 3)
def B := point (3, 0)
def C := point (4, -3)

theorem find_C_coordinates :
  ∃ C, C = point (4, -3) ∧
    let AC := (C.1 - A.1, C.2 - A.2),
        CB := (B.1 - C.1, B.2 - C.2)
    in AC = (-2 * CB.1, -2 * CB.2) :=
sorry

end find_C_coordinates_l284_284014


namespace mark_owes_after_30_days_l284_284644

/--
Conditions:
1. A 2% late charge is added to Mark’s bill every 10 days.
2. The original bill was $500.
3. The period past due date is 30 days.

Proof:
Given these conditions, prove that after 30 days past the due date, the total amount owed equals $530.604.
-/
theorem mark_owes_after_30_days (original_bill : ℝ) (rate : ℝ) (days_past_due : ℕ)
  (h_orig : original_bill = 500)
  (h_rate : rate = 0.02)
  (h_days : days_past_due = 30) :
  let first_charge := original_bill * (1 + rate),
      second_charge := first_charge * (1 + rate),
      final_amount := second_charge * (1 + rate)
  in final_amount = 530.604 :=
by {
  sorry
}

end mark_owes_after_30_days_l284_284644


namespace find_B_current_age_l284_284096

variable {A B C : ℕ}

theorem find_B_current_age (h1 : A + 10 = 2 * (B - 10))
                          (h2 : A = B + 7)
                          (h3 : C = (A + B) / 2) :
                          B = 37 := by
  sorry

end find_B_current_age_l284_284096


namespace area_BGHI_l284_284104

variable (ABCD : Type) [Rectangle ABCD]
variable (E F G H I : Point)
variable (AD DC : Line)
variable (AF BE CE BF : Line)
variable (AGE DEHF CIF BGHI : Region)

-- Conditions
axiom E_on_AD : On_Line E AD
axiom F_on_DC : On_Line F DC
axiom G_intersection_AF_BE : Intersection G AF BE
axiom H_intersection_AF_CE : Intersection H AF CE
axiom I_intersection_BF_CE : Intersection I BF CE
axiom area_AGE : Area AGE = 2
axiom area_DEHF : Area DEHF = 3
axiom area_CIF : Area CIF = 1

-- Goal
theorem area_BGHI : Area BGHI = 6 := by sorry

end area_BGHI_l284_284104


namespace fifth_number_in_9th_row_l284_284576

theorem fifth_number_in_9th_row : (∀ i, last_number_of_row i = 7 * i) → fifth_number_in_9th_row = 60 :=
by
  -- Defining conditions as per the problem statement
  let rows := 9
  let columns := 7
  let last_number_of_row (i : Nat) := 7 * i
  let fifth_number_of_row (i : Nat) := last_number_of_row i - 3
  
  -- Proof starts here
  intro h
  have last_ninth_row := last_number_of_row 9
  have fifth_ninth_row := fifth_number_of_row 9
  have fifth_number_in_9th_row := last_ninth_row - 3
  exact fifth_ninth_row.sorry

end fifth_number_in_9th_row_l284_284576


namespace greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l284_284617

theorem greatest_possible_sum_of_two_consecutive_integers_product_lt_1000 : 
  ∃ n : ℤ, (n * (n + 1) < 1000) ∧ (n + (n + 1) = 63) :=
sorry

end greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l284_284617


namespace min_integer_solution_l284_284217

theorem min_integer_solution (x : ℤ) (h1 : 3 - x > 0) (h2 : (4 * x / 3 : ℚ) + 3 / 2 > -(x / 6)) : x = 0 := by
  sorry

end min_integer_solution_l284_284217


namespace problem_statement_l284_284889

def omega (k : ℤ) := {p : ℝ × ℝ | (p.1 - k) ^ 2 + (p.2 - k^2) ^ 2 = 4 * |k|}

theorem problem_statement :
  (∃ l : ℝ → ℝ, (∀ p ∈ omega, ¬ (p.2 = l p.1)) ∧
    (∃ q1 ∈ omega, ∃ q2 ∈ omega, q1.2 ≠ l q1.1 ∧ q2.2 ≠ l q2.1 ∧ q1.2 < l q1.1 ∧ q2.2 > l q2.1)) = false ∧
  (∃ l : ℝ → ℝ, ∀ n : ℤ, (∃ p ∈ omega n, p.2 = l p.1)) = true := 
sorry

end problem_statement_l284_284889


namespace knight_moves_everywhere_l284_284941

-- Define the problem conditions in Lean
variables (p q : ℕ)

-- Define the question as a proof goal in Lean
theorem knight_moves_everywhere : 
  (∀ (x y x' y' : ℤ), ∃ (n : ℕ) (moves : Fin n → (ℤ × ℤ)), 
    (moves 0 = (x, y)) ∧ (moves (Fin.last n) = (x', y')) ∧ 
    (∀ i, ∃ (dx dy : ℤ), 
      (dx = p ∧ dy = q ∨ dx = q ∧ dy = p ∨ dx = -p ∧ dy = -q ∨ dx = -q ∧ dy = -p) ∧ 
      (moves i.succ = (moves i).fst + dx, (moves i).snd + dy))) →
  (Nat.gcd p q = 1 ∧ (p + q) % 2 = 1) :=
begin
  -- Proof would go here, but we omit it as per instructions.
  sorry
end

end knight_moves_everywhere_l284_284941


namespace dart_in_center_square_prob_l284_284655

theorem dart_in_center_square_prob :
  let x := 1 in
  let area_center_square := x * x in
  let area_total := 2 * x * x * (1 + Real.sqrt 2) in
  (area_center_square / area_total = (Real.sqrt 2 - 1) / 2) :=
by
  let x := 1
  let area_center_square := x * x
  let area_total := 2 * x * x * (1 + Real.sqrt 2)
  show area_center_square / area_total = (Real.sqrt 2 - 1) / 2
  sorry

end dart_in_center_square_prob_l284_284655


namespace find_x_squared_plus_y_squared_l284_284445

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l284_284445


namespace point_N_coordinates_l284_284013

/--
Given:
- point M with coordinates (5, -6)
- vector a = (1, -2)
- the vector NM equals 3 times vector a
Prove:
- the coordinates of point N are (2, 0)
-/

theorem point_N_coordinates (x y : ℝ) :
  let M := (5, -6)
  let a := (1, -2)
  let NM := (5 - x, -6 - y)
  3 * a = NM → 
  (x = 2 ∧ y = 0) :=
by 
  intros
  sorry

end point_N_coordinates_l284_284013


namespace geometric_sequence_absolute_value_bound_range_of_m_l284_284432

-- (I) Prove the sequence {a_n - (-1)^n} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) + 2 * a n = (-1 : ℝ) ^ n) : 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) - (-1 : ℝ) ^ (n + 1) = r * (a n - (-1 : ℝ) ^ n) :=
sorry

-- (II) Prove |a_n| ≥ (3n + 1) / 2 for any n ∈ ℕ using mathematical induction
theorem absolute_value_bound (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) + 2 * a n = (-1 : ℝ) ^ n) : 
  ∀ n : ℕ, |a n| ≥ (3 * n + 1) / 2 :=
sorry

-- (III) Find the range of m such that T_n < m for any n ∈ ℕ where T_n is the sum of first n terms of b_n
theorem range_of_m (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) + 2 * a n = (-1 : ℝ) ^ n) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h3 : ∀ n : ℕ, b n = -2 ^ n / (a n * a (n + 1))) 
  (h4 : ∀ n : ℕ, T n = ∑ i in finset.range n, b i) :
  ∀ m : ℝ, (∀ n : ℕ, T n < m) ↔ m ≥ 1 / 3 :=
sorry

end geometric_sequence_absolute_value_bound_range_of_m_l284_284432


namespace distinct_complex_powers_of_polygon_vertices_l284_284027

theorem distinct_complex_powers_of_polygon_vertices :
  let Z (n k : ℕ) := exp (2 * π * I * k / n)
  let vertices := finset.range 20
  (finset.image (λ k, (Z 20 k) ^ 1995) vertices).card = 4 :=
by
  sorry

end distinct_complex_powers_of_polygon_vertices_l284_284027


namespace inequality_problem_l284_284016

theorem inequality_problem
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b^3 / (a^2 + 8*b*c) + c^3 / (b^2 + 8*c*a) + a^3 / (c^2 + 8*a*b) ≥ 1/9 * (a + b + c)) :=
begin
  sorry
end

end inequality_problem_l284_284016


namespace sum_of_odd_numbers_l284_284177

theorem sum_of_odd_numbers (n : ℕ) : ∑ k in Finset.range (n + 1), (2 * k + 1 - 1) = n ^ 2 :=
by sorry

end sum_of_odd_numbers_l284_284177


namespace hearing_news_probability_l284_284649

noncomputable def probability_of_hearing_news : ℚ :=
  let broadcast_cycle := 30 -- total time in minutes for each broadcast cycle
  let news_duration := 5  -- duration of each news broadcast in minutes
  news_duration / broadcast_cycle

theorem hearing_news_probability : probability_of_hearing_news = 1 / 6 := by
  sorry

end hearing_news_probability_l284_284649


namespace fred_red_marbles_l284_284739

variable (R G B : ℕ)
variable (total : ℕ := 63)
variable (B_val : ℕ := 6)
variable (G_def : G = (1 / 2) * R)
variable (eq1 : R + G + B = total)
variable (eq2 : B = B_val)

theorem fred_red_marbles : R = 38 := 
by
  sorry

end fred_red_marbles_l284_284739


namespace polynomial_remainder_l284_284733

theorem polynomial_remainder (x : ℤ) : 
  polynomial.mod_by_monic ((polynomial.X ^ 4) : polynomial ℤ) (polynomial.X ^ 3 + 3 * polynomial.X ^ 2 + 2 * polynomial.X + 1) = -polynomial.X ^ 2 - polynomial.X - 1 :=
sorry

end polynomial_remainder_l284_284733


namespace find_AM_length_l284_284520

-- Define the context
variables {Point : Type} [metric_space Point] (A B C D N M P Q : Point)
noncomputable def circle (center : Point) (radius : ℝ) := 
  { x : Point | dist x center = radius }
variables Γ₁ Γ₂ : set Point
variables r1 r2 : ℝ
variables dist_AN dist_NB : ℝ

-- Define given conditions
axiom h1 : AB ∈ Γ₁
axiom h2 : CD ∈ Γ₁
axiom h3 : dist A B = 2 * r1
axiom h4 : dist A N = dist_AN
axiom h5 : dist N B = dist_NB
axiom h6 : CD ⊥ AB
axiom h7 : circle C r2 = Γ₂
axiom h8 : dist C N = r2
axiom h9 : r1 = 61
axiom h10 : r2 = 60
axiom h11 : P ∈ Γ₁ ∩ Γ₂
axiom h12 : Q ∈ Γ₁ ∩ Γ₂

-- Define the length AM
noncomputable def length_AM : ℝ := dist A M

-- Define the theorem to prove
theorem find_AM_length : length_AM A M = 78 :=
by sorry

end find_AM_length_l284_284520


namespace expressible_integers_with_powers_of_three_l284_284713

theorem expressible_integers_with_powers_of_three :
  ∀ (n : ℤ), abs n < 1986 →
    ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ),
      n = a₁ * 1 + a₂ * 3 + a₃ * 9 + a₄ * 27 + a₅ * 81 + a₆ * 243 + a₇ * 729 + a₈ * 2187 ∧
      (∀ i, i ∈ {a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈} → (i = 0 ∨ i = 1 ∨ i = -1)) :=
sorry

end expressible_integers_with_powers_of_three_l284_284713


namespace part1_part2_l284_284784

noncomputable def ellipse_eccentricity (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ (a^2 - b^2 = (sqrt 3 / 2)^2 * a^2)

noncomputable def ellipse_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
P.1 = 2 * sqrt 2 ∧ P.2 = 0 ∧ ((P.1 / a)^2 + (P.2 / b)^2 = 1)

def line_l_slope (m : ℝ) : Prop :=
m = 1 / 2

def point_P : ℝ × ℝ := (2, 1)

noncomputable def max_triangle_area (m : ℝ) (P : ℝ × ℝ) (a b : ℝ) := 
  ∃ C : (ℝ × ℝ → ℝ) × (ℝ × ℝ → ℝ), ellipse_eccentricity a b → ellipse_point a b (2 * sqrt 2, 0) → 
  line_l_slope m → let l := (λ (x : ℝ × ℝ), x.2 = m * x.1 + C.1 x) in ∀ A B : ℝ × ℝ, 
  (l A ∧ l B ∧ (A.1 / a)^2 + (A.2 / b)^2 = 1 ∧ (B.1 / a)^2 + (B.2 / b)^2 = 1) →
  ((2 * abs m / sqrt 5) * sqrt (5 * (4 - m^2)) / 2 = 2)

theorem part1 (a b : ℝ) : ellipse_eccentricity a b → ellipse_point a b (2 * sqrt 2, 0) →
(∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ↔ (x / sqrt 8)^2 + (y / sqrt 2)^2 = 1) := sorry

theorem part2 (m : ℝ) (a b : ℝ) : line_l_slope m → max_triangle_area m point_P a b := sorry

end part1_part2_l284_284784


namespace parallel_vectors_y_value_l284_284809

theorem parallel_vectors_y_value (y : ℝ) :
  let a := (2, 3)
  let b := (4, y)
  ∃ y : ℝ, (2 : ℝ) / 4 = 3 / y → y = 6 :=
sorry

end parallel_vectors_y_value_l284_284809


namespace Michelle_has_35_crayons_l284_284155

theorem Michelle_has_35_crayons :
  (∃ n_boxes n_crayons_per_box total_crayons : ℕ, n_boxes = 7 ∧ n_crayons_per_box = 5 ∧ total_crayons = n_boxes * n_crayons_per_box ∧ total_crayons = 35) :=
by {
-- Definitions and assumptions
  let n_boxes := 7,
  let n_crayons_per_box := 5,
  let total_crayons := n_boxes * n_crayons_per_box,
  have h1 : n_boxes = 7 := rfl,
  have h2 : n_crayons_per_box = 5 := rfl,
  have h3 : total_crayons = n_boxes * n_crayons_per_box := rfl,
  have h4 : total_crayons = 35 := by sorry, -- completion of the actual proof required
  exact ⟨n_boxes, n_crayons_per_box, total_crayons, h1, h2, h3, h4⟩,
}

end Michelle_has_35_crayons_l284_284155


namespace product_of_divisors_of_72_l284_284369

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284369


namespace sqrt3_solves_log_series_l284_284628

noncomputable def log_with_roots (x : ℝ) : ℝ :=
  ∑ i in (Finset.range 8), (2 * (i + 1)) * Real.log x / Real.log (3^(1 / (2 * (i + 1))))

theorem sqrt3_solves_log_series (x : ℝ) (hx : 0 < x) : 
  log_with_roots x = 36 ↔ x = Real.sqrt 3 :=
by
  sorry

end sqrt3_solves_log_series_l284_284628


namespace smallest_positive_T_l284_284708

theorem smallest_positive_T :
    ∃ T : ℤ, (∀ (b : Fin 100 → ℤ), 
        (∀ i, b i = 1 ∨ b i = -1) → 
        T = ∑ i in Finset.range 100, ∑ j in Finset.Ico i 100, b i * b j) ∧ T > 0 ∧ T = 22 :=
by
    sorry

end smallest_positive_T_l284_284708


namespace solve_problem_l284_284137

def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2 ^ x

theorem solve_problem : f(f(2 / 3)) = 2 := by
  sorry

end solve_problem_l284_284137


namespace CE_correct_length_l284_284840

-- Define an equilateral triangle
structure EquilateralTriangle :=
(a : ℝ) -- side length

-- Define points on the triangle
structure PointsOnTriangle (ABC : EquilateralTriangle) :=
(BD AE DE : ℝ)
(BD_eq : BD = (1/3) * ABC.a)
(AE_eq_DE : AE = DE)

-- Define the length CE
def CE_length (ABC : EquilateralTriangle) (P : PointsOnTriangle ABC) : ℝ :=
  let x := (3/5) * ABC.a in
  (ABC.a * real.sqrt 19) / 5

-- The main theorem to prove
theorem CE_correct_length (ABC : EquilateralTriangle) (P : PointsOnTriangle ABC) :
  P.AE = (3/5) * ABC.a →
  P.DE = (3/5) * ABC.a →
  CE_length ABC P = (ABC.a * real.sqrt 19) / 5 :=
by
  intros h1 h2
  sorry

end CE_correct_length_l284_284840


namespace negation_of_exists_gt0_and_poly_gt0_l284_284582

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l284_284582


namespace marvin_number_is_correct_l284_284933

theorem marvin_number_is_correct (y : ℤ) (h : y - 5 = 95) : y + 5 = 105 := by
  sorry

end marvin_number_is_correct_l284_284933


namespace trajectory_moving_point_hyperbola_l284_284282

theorem trajectory_moving_point_hyperbola {n m : ℝ} (h_neg_n : n < 0) :
    (∃ y < 0, (y^2 = 16) ∧ (m^2 = (n^2 / 4 - 4))) ↔ ( ∃ (y : ℝ), (y^2 / 16) - (m^2 / 4) = 1 ∧ y < 0 ) := 
sorry

end trajectory_moving_point_hyperbola_l284_284282


namespace hyperbola_eccentricity_l284_284036

noncomputable def point_on_hyperbola (x y a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focal_length (a b c : ℝ) : Prop :=
  2 * c = 4

noncomputable def eccentricity (e c a : ℝ) : Prop :=
  e = c / a

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_point_on_hyperbola : point_on_hyperbola 2 3 a b h_pos_a h_pos_b)
  (h_focal_length : focal_length a b c)
  : eccentricity e c a :=
sorry -- proof omitted

end hyperbola_eccentricity_l284_284036


namespace part1_part2_l284_284770

namespace VectorProblem

open Real

-- Defining the vectors and their properties
variables (a b : ℝ³)
variables (ha : ‖a‖ = 2)
variables (hb : ‖b‖ = 3)
variables (angle_ab : real.angle a b = π * (2 / 3))

noncomputable def magnitude_of_sum : Real :=
‖a + 2 • b‖

theorem part1 : magnitude_of_sum a b ha hb angle_ab = 2 * sqrt 7 := by
  sorry

noncomputable def projection_on_b : Real :=
(a + 2 • b) ⬝ b / ‖b‖

theorem part2 : projection_on_b a b ha hb angle_ab = 5 := by
  sorry

end VectorProblem

end part1_part2_l284_284770


namespace proof_statement_l284_284762

noncomputable theory
open_locale classical

def seq (n : ℕ) : ℕ → ℝ
| 0 := 1/3
| (n + 1) := seq n + (seq n)^2

def f : ℕ → ℝ := λ n, 1 / (seq n + 1)

def proof_problem : Prop :=
2 < ∑ k in finset.range 2002, f (k + 1) ∧ ∑ k in finset.range 2002, f (k + 1) < 3

-- This is a placeholder proof to show it can be compiled without issues
theorem proof_statement : proof_problem := by sorry

end proof_statement_l284_284762


namespace value_of_a5_l284_284852

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a n * r ^ (m - n) = a m

theorem value_of_a5 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 :=
by
  sorry

end value_of_a5_l284_284852


namespace least_number_of_bills_needed_l284_284260

-- Define the conditions: number of each type of bill
def num_ten_dollar_bills : ℕ := 13
def num_five_dollar_bills : ℕ := 11
def num_one_dollar_bills : ℕ := 17
def required_amount : ℕ := 128

-- Statement of the problem
theorem least_number_of_bills_needed
  (h1 : num_ten_dollar_bills = 13)
  (h2 : num_five_dollar_bills = 11)
  (h3 : num_one_dollar_bills = 17)
  (h4 : required_amount = 128) :
  ∃ n, n = 16 ∧
  (n ≤ num_ten_dollar_bills + num_five_dollar_bills + num_one_dollar_bills) ∧ 
  (12 * 10 + 1 * 5 + 3 * 1 = required_amount) :=
begin
  sorry
end

end least_number_of_bills_needed_l284_284260


namespace length_of_AB_is_1_l284_284035

variables {A B C : ℝ} -- Points defining the triangle vertices
variables {a b c : ℝ} -- Lengths of triangle sides opposite to angles A, B, C respectively
variables {α β γ : ℝ} -- Angles at points A B C
variables {s₁ s₂ s₃ : ℝ} -- Sin values of the angles

noncomputable def length_of_AB (a b c : ℝ) : ℝ :=
  if a + b + c = 4 ∧ a + b = 3 * c then 1 else 0

theorem length_of_AB_is_1 : length_of_AB a b c = 1 :=
by
  have h_perimeter : a + b + c = 4 := sorry
  have h_sin_condition : a + b = 3 * c := sorry
  simp [length_of_AB, h_perimeter, h_sin_condition]
  sorry

end length_of_AB_is_1_l284_284035


namespace turtle_reaches_watering_hole_time_after_lion_incident_l284_284261

theorem turtle_reaches_watering_hole_time_after_lion_incident :
  ∀ x y : ℝ, 
  y > 0 ∧ x > 0 ∧
  ( ∀ minutes_until_second_incident: ℝ, 
  minutes_until_second_incident = 2.4 ∧
  (y / x) = 6 → 
  let distance_watering_hole := 32 in
  distance_watering_hole - minutes_until_second_incident * (x + 1.5 * y) / 
  (x + y) = 28.8) := 
by 
  sorry

end turtle_reaches_watering_hole_time_after_lion_incident_l284_284261


namespace integral_solutions_count_l284_284769

theorem integral_solutions_count (m : ℕ) (h : m > 0) :
  ∃ S : Finset (ℕ × ℕ), S.card = m ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + p.2^2 + 2 * p.1 * p.2 - m * p.1 - m * p.2 - m - 1 = 0) := 
sorry

end integral_solutions_count_l284_284769


namespace graph_not_in_second_quadrant_l284_284087

theorem graph_not_in_second_quadrant (b : ℝ) (h : ∀ x < 0, 2^x + b - 1 < 0) : b ≤ 0 :=
sorry

end graph_not_in_second_quadrant_l284_284087


namespace find_lambda_l284_284801

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Variables declaration
variables (a b : ℝ × ℝ) (λ : ℝ)

-- Given conditions
def a := (2, 1 : ℝ × ℝ)
def b := (-1, 3 : ℝ × ℝ)
def perp_condition := dot_product a (a.1 - λ * b.1, a.2 + λ * b.2) = 0

-- Statement to be proved
theorem find_lambda (h : perp_condition) : λ = -5 := by
  sorry

end find_lambda_l284_284801


namespace union_of_lines_M1_M2_l284_284237

variables {P : Type*} [IncidenceGeometry P] {l1 l2 : Line P} {M1 M2 : P}

def MovingBody (P : Type*) [IncidenceGeometry P] : Type* := 
{ M1 M2 : P // True }

theorem union_of_lines_M1_M2 (h1 : l1 ∥ l2 ∨ ∃ P : P, P ∈ l1 ∧ P ∈ l2)
(h2 : ∀ (M1 M2 : P), M1 ∈ l1 ∧ M2 ∈ l2) :
(∃ pt : P, ∀ (M1 M2 : P), LineSeg M1 M2 ∈ Line pt) ∨
(∃ parabola : ConicSection P, ∀ (M1 M2 : P), LineSeg M1 M2 ∈ Tangent parabola) := 
sorry

end union_of_lines_M1_M2_l284_284237


namespace sum_DE_FG_l284_284839

-- Variables: Let x be AD and y be AG, with DE and FG parallel to BC.
-- Equilateral triangle ABC with side length 2.
variables (x y : ℝ)

-- Conditions derived from the problem statement
axiom AD : x >= 0 ∧ x <= 2
axiom AG : y >= 0 ∧ y <= 2
axiom DE_parallel_BC : ∀ x, DE = x
axiom FG_parallel_BC : ∀ y, FG = y
axiom perimeter_equilateral_triangle: ∀ x y, (x + y) = (3y - x) ∧ (x + y) = (6 - y)

-- Main statement to prove
theorem sum_DE_FG (DE FG : ℝ) : (AD x) ∧ (AG y) ∧ (DE_parallel_BC DE) ∧ (FG_parallel_BC FG) ∧ (perimeter_equilateral_triangle x y) → DE + FG = 2 :=
by
  sorry

end sum_DE_FG_l284_284839


namespace correct_answers_count_l284_284103

theorem correct_answers_count
  (c w : ℕ)
  (h1 : 4 * c - 2 * w = 420)
  (h2 : c + w = 150) : 
  c = 120 :=
sorry

end correct_answers_count_l284_284103


namespace cakes_served_yesterday_l284_284665

theorem cakes_served_yesterday:
  ∃ y : ℕ, (5 + 6 + y = 14) ∧ y = 3 := 
by
  sorry

end cakes_served_yesterday_l284_284665


namespace polynomial_expansion_l284_284073

variable (a_0 a_1 a_2 a_3 a_4 : ℝ)

theorem polynomial_expansion :
  ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 1) :=
by
  sorry

end polynomial_expansion_l284_284073


namespace sum_of_g_49_l284_284764

theorem sum_of_g_49 (f g : ℝ → ℝ) (h_f : ∀ x, f x = 3 * x^2 - 10) (h_g : ∀ x, g (f x) = x^2 + x + 1) :
  g (49) + g (49) = 124 / 3 :=
begin
  sorry
end

end sum_of_g_49_l284_284764


namespace floor_sqrt_sum_l284_284685

theorem floor_sqrt_sum :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 → n > 0) →
  (∑ n in finset.range 26, if n > 0 then (nat.floor (real.sqrt n)) else 0) = 75 :=
by
  intro h
  sorry

end floor_sqrt_sum_l284_284685


namespace mean_reciprocals_first_three_composites_l284_284720

theorem mean_reciprocals_first_three_composites :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = (13 : ℚ) / 72 := 
by
  sorry

end mean_reciprocals_first_three_composites_l284_284720


namespace number_of_positive_integer_solutions_l284_284585

theorem number_of_positive_integer_solutions (n k : ℕ) (h : n ≥ k) :
    ∃ solutions,
    (∀ x : ℕ → ℕ, (∀ i, 1 ≤ x i) → (sum x {0..k-1} = n) ↔ ∃ f : ℕ → ℕ, sum f {0..k-1} = n - k)
    → solutions = (Nat.choose (n - 1) (k - 1)) :=
by
  sorry

end number_of_positive_integer_solutions_l284_284585


namespace no_base_b_for_perf_square_l284_284698

theorem no_base_b_for_perf_square : ∀ (b : ℤ), b > 3 → ¬∃ (k : ℤ), (b^2 + 3 * b + 2) = k^2 :=
begin
  sorry
end

end no_base_b_for_perf_square_l284_284698


namespace sin_add_simplify_l284_284955

variable (x y : ℝ)

theorem sin_add_simplify : sin (x + y) * cos y + cos (x + y) * sin y = sin (x + 2y) :=
by
  sorry

end sin_add_simplify_l284_284955


namespace average_leg_time_l284_284636

theorem average_leg_time (Y_time Z_time : ℕ) (hY : Y_time = 58) (hZ : Z_time = 26) : 
  (Y_time + Z_time) / 2 = 42 :=
by
  rw [hY, hZ]
  norm_num
  sorry

end average_leg_time_l284_284636


namespace g_neg4_l284_284146

def g (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 5 else 6 - x

theorem g_neg4 : g (-4) = -7 := by
  sorry

end g_neg4_l284_284146


namespace milkshake_hours_l284_284681

theorem milkshake_hours (h : ℕ) : 
  (3 * h + 7 * h = 80) → h = 8 := 
by
  intro h_milkshake_eq
  sorry

end milkshake_hours_l284_284681


namespace quadratic_positive_if_and_only_if_l284_284738

-- Define the quadratic function
def quadratic (k x : ℝ) : ℝ := x^2 - 2*k*x + (2*k - 1)

-- Main theorem statement to prove
theorem quadratic_positive_if_and_only_if (k : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → quadratic k x > 0) ↔ (1 ≤ k) :=
begin
  sorry -- This is where the proof would go
end

end quadratic_positive_if_and_only_if_l284_284738


namespace complex_real_part_solution_is_22_l284_284375

noncomputable def complex_real_part_of_solution : ℝ :=
  let z := λ (a b : ℝ), complex.mk a b in
  if h : ∃ a b : ℝ, b > 0 ∧ z(a, b) * z(a, b + 2) * z(a, b - 2) * z(a, b + 5) = 8000 then
    let ⟨a, b, hb_pos, h_eq⟩ := h in a
  else
    0

theorem complex_real_part_solution_is_22 :
  complex_real_part_of_solution = 22 :=
by 
  sorry

end complex_real_part_solution_is_22_l284_284375


namespace complex_number_quadrant_l284_284825

noncomputable def location_of_point (z : ℂ) : Prop :=
  ∀ (z : ℂ), (z / (1 + Complex.i) = 2 * Complex.i) → 
  let w := z in
  Complex.re w < 0 ∧ Complex.im w > 0

theorem complex_number_quadrant :
  ∃ z : ℂ, (z / (1 + Complex.i) = 2 * Complex.i) → location_of_point z :=
begin
  sorry
end

end complex_number_quadrant_l284_284825


namespace S_infinite_l284_284499

noncomputable theory

-- Definition: S is a set of points.
variable (S : Set Point)

-- Condition: every point in S is the midpoint of a segment whose endpoints are in S.
axiom midpoint_in_S (P : Point) (hP : P ∈ S) : ∃ A B : Point, A ∈ S ∧ B ∈ S ∧ P = midpoint A B

-- Goal: Show that S is infinite.
theorem S_infinite : ¬Finite S :=
sorry

end S_infinite_l284_284499


namespace find_x_squared_plus_y_squared_l284_284444

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l284_284444


namespace infinite_solutions_for_abcd_l284_284462

def average_condition (a b c d : ℝ) : Prop :=
  (2 * a + 16 * b) + (3 * c - 8 * d) = 148

def equation_condition (a b c d : ℝ) : Prop :=
  4 * a + 6 * b = 9 * c - 12 * d

theorem infinite_solutions_for_abcd (a b c d : ℝ) :
  average_condition a b c d →
  equation_condition a b c d →
  ∃ (s : ℝ), true := 
  begin
    sorry
  end

end infinite_solutions_for_abcd_l284_284462


namespace sum_a_b_probability_fourth_shiny_sixth_draw_l284_284646

noncomputable def total_combinations : ℕ := (11.choose 5)
noncomputable def event_shiny_pennies : ℕ := (5.choose 3)

theorem sum_a_b_probability_fourth_shiny_sixth_draw :
  let P := event_shiny_pennies / total_combinations
  ∃ a b : ℕ, P = a / b ∧ Int.gcd a b = 1 ∧ a + b = 236 :=
begin
  sorry
end

end sum_a_b_probability_fourth_shiny_sixth_draw_l284_284646


namespace farmer_field_value_of_m_l284_284276

-- Given conditions
variable (m : ℝ)
def field_area (m : ℝ) := (3 * m + 8) * (m - 3)

-- The goal to prove
theorem farmer_field_value_of_m (h : field_area m = 100) : m ≈ 6.597 := sorry

end farmer_field_value_of_m_l284_284276


namespace rectangles_greater_than_one_area_l284_284812

theorem rectangles_greater_than_one_area (n : ℕ) (H : n = 5) : ∃ r, r = 84 :=
by
  sorry

end rectangles_greater_than_one_area_l284_284812


namespace sum_of_binomials_l284_284639

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_of_binomials (n : ℕ) : 
  (∑ k in Finset.range (n + 1), binomial n k) = 2^n := 
  by sorry

end sum_of_binomials_l284_284639


namespace factorization_exists_l284_284330

def polynomial := ℚ[X]

noncomputable def P : polynomial := 
  X^2 - 6*X + 9 - 64*X^4

theorem factorization_exists : 
  ∃ (a b c d e f : ℤ) (h1 : a < d), (∀ (x : polynomial), P = (a•x^2 + b•x + c) * (d•x^2 + e•x + f)) :=
by {
  use [-8, 1, -3, 8, 1, -3],
  split,
  linarith,
  intro x,
  -- Need to prove the factorization here, but we skip the proof steps
  sorry
}

end factorization_exists_l284_284330


namespace part_one_part_two_l284_284795

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 / x

theorem part_one (x a : ℝ) (hx : x > 0) (ineq : x * f' x ≤ x^2 + a * x + 1) : a ∈ Set.Ici (-1) :=
by sorry

theorem part_two (x : ℝ) (hx : x > 0) : (x - 1) * f x ≥ 0 :=
by sorry

end part_one_part_two_l284_284795


namespace real_values_satisfying_inequality_l284_284335

theorem real_values_satisfying_inequality :
  ∀ x : ℝ, x ≠ 5 → (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ x ∈ set.Iic 0.76 ∪ set.Ioo 5 10.1 := by
  sorry

end real_values_satisfying_inequality_l284_284335


namespace sum_of_squares_of_roots_eq_1853_l284_284982

theorem sum_of_squares_of_roots_eq_1853
  (α β : ℕ) (h_prime_α : Prime α) (h_prime_beta : Prime β) (h_sum : α + β = 45)
  (h_quadratic_eq : ∀ x, x^2 - 45*x + α*β = 0 → x = α ∨ x = β) :
  α^2 + β^2 = 1853 := 
by
  sorry

end sum_of_squares_of_roots_eq_1853_l284_284982


namespace total_ages_l284_284875

def Kate_age : ℕ := 19
def Maggie_age : ℕ := 17
def Sue_age : ℕ := 12

theorem total_ages : Kate_age + Maggie_age + Sue_age = 48 := sorry

end total_ages_l284_284875


namespace no_hexagonal_pyramid_with_equal_edges_l284_284662

theorem no_hexagonal_pyramid_with_equal_edges (edges : ℕ → ℝ)
  (regular_polygon : ℕ → ℝ → Prop)
  (equal_length_edges : ∀ (n : ℕ), regular_polygon n (edges n) → ∀ i j, edges i = edges j)
  (apex_above_centroid : ∀ (n : ℕ) (h : regular_polygon n (edges n)), True) :
  ¬ regular_polygon 6 (edges 6) :=
by
  sorry

end no_hexagonal_pyramid_with_equal_edges_l284_284662


namespace decreasing_log_function_range_l284_284211

theorem decreasing_log_function_range : 
  (∀ (a : ℝ), (∀ (x : ℝ), x ∈ Ioo 0 2 → 6 - a * x > 0 ∧ 6 - a * x ∈ Ioi 0 → a > 1) → 1 < a ∧ a ≤ 3) := 
sorry

end decreasing_log_function_range_l284_284211


namespace minimize_expression_l284_284145

theorem minimize_expression (a b c d : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 :=
by
  sorry

end minimize_expression_l284_284145


namespace max_possible_n_l284_284154

noncomputable def max_points_set (S : Finset ℂ) : Prop :=
  ∀ (a b ∈ S), a ≠ b → complex.abs (a - b) ≥ Finset.sup S complex.abs

theorem max_possible_n (S : Finset ℂ) (h : max_points_set S) : S.card ≤ 7 :=
sorry

end max_possible_n_l284_284154


namespace a_2017_value_l284_284063

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ 
  (a 2 = 4) ∧ 
  (a 3 = 9) ∧ 
  (∀ n, n ≥ 4 → a n = a (n-1) + a (n-2) - a (n-3))

theorem a_2017_value (a : ℕ → ℤ) (h : sequence a) : a 2017 = 8065 := 
by sorry

end a_2017_value_l284_284063


namespace parts_per_day_system_l284_284167

variable (x y : ℕ)

def personA_parts_per_day (x : ℕ) : ℕ := x
def personB_parts_per_day (y : ℕ) : ℕ := y

-- First condition
def condition1 (x y : ℕ) : Prop :=
  6 * x = 5 * y

-- Second condition
def condition2 (x y : ℕ) : Prop :=
  30 + 4 * x = 4 * y - 10

theorem parts_per_day_system (x y : ℕ) :
  condition1 x y ∧ condition2 x y :=
sorry

end parts_per_day_system_l284_284167


namespace sufficient_not_necessary_p_q_l284_284766

theorem sufficient_not_necessary_p_q {m : ℝ} 
  (hp : ∀ x, (x^2 - 8*x - 20 ≤ 0) → (-2 ≤ x ∧ x ≤ 10))
  (hq : ∀ x, ((x - 1 - m) * (x - 1 + m) ≤ 0) → (1 - m ≤ x ∧ x ≤ 1 + m))
  (m_pos : 0 < m)  :
  (∀ x, (x - 1 - m) * (x - 1 + m) ≤ 0 → x^2 - 8*x - 20 ≤ 0) ∧ ¬ (∀ x, x^2 - 8*x - 20 ≤ 0 → (x - 1 - m) * (x - 1 + m) ≤ 0) →
  m ≤ 3 :=
sorry

end sufficient_not_necessary_p_q_l284_284766


namespace evaluate_g_sum_l284_284124

def g (a b : ℚ) : ℚ :=
if a + b ≤ 5 then (a^2 * b - a + 3) / (3 * a) 
else (a * b^2 - b - 3) / (-3 * b)

theorem evaluate_g_sum : g 3 2 + g 3 3 = -1 / 3 :=
by
  sorry

end evaluate_g_sum_l284_284124


namespace sum_of_first_five_terms_l284_284012

-- Definitions
def an_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

def roots_of_equation (x y : ℝ) : Prop :=
  x^2 - x - 1 = 0 ∧ y^2 - y - 1 = 0

noncomputable def a_sequence := λ n : ℕ, (sqrt 5 + 1) / 2 * n + 1

-- Theorem
theorem sum_of_first_five_terms:
  (an_arithmetic_sequence a_sequence) ∧
  (roots_of_equation (a_sequence 1) (a_sequence 3)) →
  sum_of_first_n_terms a_sequence 5 = 5 / 2 :=
begin
  sorry
end

end sum_of_first_five_terms_l284_284012


namespace starting_player_wins_l284_284532

-- Define the cards on the table
def cards : List Nat := [0, 1, 2, 3, 4, 5, 6]

-- Define what it means for a player to win
def wins_with_number_divisible_by_17 (cards : List Nat) : Bool :=
  ∃ (n : Nat), (n ∈ (List.permutations cards)) ∧ (17 ∣ n)

-- The statement of the problem
theorem starting_player_wins :
  wins_with_number_divisible_by_17 cards := 
sorry

end starting_player_wins_l284_284532


namespace product_of_divisors_eq_l284_284347

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284347


namespace find_a_l284_284001

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284001


namespace value_of_f_3_div_2_l284_284135

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom f_in_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = x + 1

theorem value_of_f_3_div_2 : f (3 / 2) = 3 / 2 := by
  sorry

end value_of_f_3_div_2_l284_284135


namespace consecutive_sum_to_20_has_one_set_l284_284071

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l284_284071


namespace length_PB_eq_2x_add_2_l284_284469

-- Definitions and conditions
variables (x : ℝ)
variables (circle : Type)
variables (M P C A B C' : circle)
variables (MP_perpendicular_AB : ⟂(MP, AB))
variables (arc_midpoint_M : midpoint(arc C A B) = M)
variables (length_AC : dist(A, C) = 2 * x)
variables (length_AP : dist(A, P) = 2 * x + 2)
variables (length_CC'_eq_AB : dist(A, C) + dist(C, C') = dist(A, B))

-- The mathematical statement to prove that the length of segment PB is 2x + 2
theorem length_PB_eq_2x_add_2 
  (h_symmetry : dist(A, P) = dist(P, B)) :
  dist(P, B) = 2 * x + 2 :=
by
  -- Proof omitted
  sorry

end length_PB_eq_2x_add_2_l284_284469


namespace cone_radius_of_sector_l284_284286

theorem cone_radius_of_sector (R θ : ℝ) (hR : R = 5) (hθ : θ = 120) : 
    ∃ r, r = 5 / 3 :=
by
  have h1 : 2 * real.pi * r = (θ / 360) * 2 * real.pi * R := sorry
  have h2 : r = ( (θ / 360) * 2 * real.pi * R ) / (2 * real.pi) := sorry
  have h3 : r = ( (hθ / 360) * 2 * real.pi * hR ) / (2 * real.pi) := sorry
  let r := 5 / 3 := sorry
  existsi r
  exact eq.refl r

end cone_radius_of_sector_l284_284286


namespace perimeter_triangle_ABF2_l284_284659

-- Definitions based on the conditions
def ellipse_eq (x y : ℝ) := 4 * x^2 + y^2 = 1
def is_focus (x y : ℝ) := 4 * x^2 - y^2 = 0 -- This simplifies to the correct focus location

theorem perimeter_triangle_ABF2 :
  ∀ (A B F₁ F₂ : ℝ × ℝ), is_focus F₁.1 F₁.2 ∧ is_focus F₂.1 F₂.2 ∧ (∃ y, ellipse_eq F₁.1 y) ∧
  (∃ y, ellipse_eq F₂.1 y) ∧ (∀ (P : ℝ × ℝ), ellipse_eq P.1 P.2 →
  P = A ∨ P = B →
  ∥P - F₁∥ + ∥P - F₂∥ = 2) →
  (∥A - B∥ + ∥A - F₂∥ + ∥B - F₂∥ = 4) :=
by {
  sorry
}

end perimeter_triangle_ABF2_l284_284659


namespace no_intersection_of_sets_l284_284028

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end no_intersection_of_sets_l284_284028


namespace chris_pears_eq_lily_apples_l284_284268

-- Definitions based on conditions in part (a)
variables {x y : ℕ}
def apples_total := 2 * x
def pears_total := x
def total_fruit := apples_total + pears_total

def chris_fruit := 2 * x
def lily_fruit := x

-- Chris's distribution
def chris_pears := y
def chris_apples := chris_fruit - chris_pears

-- Lily's distribution
def lily_pears := x - y
def lily_apples := lily_fruit - lily_pears

-- Proof statement based on conclusion in part (b)
theorem chris_pears_eq_lily_apples : chris_pears = lily_apples :=
sorry

end chris_pears_eq_lily_apples_l284_284268


namespace perimeter_of_quadrilateral_l284_284478

theorem perimeter_of_quadrilateral (A B C D : Point) (AB AC BD DC : ℝ) 
  (h1 : right_triangle A B C) (h2 : right_triangle B C D) (h3 : right_triangle A B D) 
  (h4 : ∠CAB = 45) (h5 : ∠CBD = 45) (h6 : ∠ADB = 90) (h7 : AB = 20) : 
  perimeter A B C D = 60 + 20 * real.sqrt 2 :=
by
  sorry

end perimeter_of_quadrilateral_l284_284478


namespace problem_x2_plus_y2_l284_284451

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l284_284451


namespace solutions_of_quadratic_eq_l284_284587

-- Define conditions
def passes_through (a b c : ℝ) (x y : ℝ) : Prop := a * x^2 + b * x + c = y

-- Define quadratic equation
def quadratic_eq (a b c : ℝ) : ℝ → ℝ :=
  λ x, a * (x - 2)^2 - 3 - (2 * b - b * x - c)

-- Theorem statement that can be built successfully:
theorem solutions_of_quadratic_eq (a b c : ℝ)  
  (h1 : passes_through a b c (-1) 3)
  (h2 : passes_through a b c 2 3) :
  quadratic_eq a b c = 0 → (1 = 4 := sorry 

end solutions_of_quadratic_eq_l284_284587


namespace factorial_product_lt_sum_factorial_l284_284392

open BigOperators

theorem factorial_product_lt_sum_factorial (a : ℕ → ℕ) (n : ℕ) (hpos : ∀ i, i < n → 0 < a i) :
  (∏ i in Finset.range n, (a i)!) < ((∑ i in Finset.range n, a i) + 1)! :=
by sorry

end factorial_product_lt_sum_factorial_l284_284392


namespace distinct_painting_methods_l284_284600

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem distinct_painting_methods (n : ℕ) : 
  (n > 0) → 
  let C (n : ℕ) := binomial (2 * n - 2) (n - 1) / n 
  in ∃ C_n, C_n = (1/(n+1)) * (binomial (2*n) n) := 
  sorry

end distinct_painting_methods_l284_284600


namespace card_sending_condition_l284_284964

theorem card_sending_condition (n m : ℕ) (h : m > (n-1)/2) :
  Exists (λ (i j : ℕ), i ≠ j ∧ i < n ∧ j < n ∧ sends_card i j ∧ sends_card j i) := sorry

end card_sending_condition_l284_284964


namespace find_common_ratio_l284_284009

variable (a : ℕ → ℝ) -- represents the geometric sequence
variable (q : ℝ) -- represents the common ratio

-- conditions given in the problem
def a_3_condition : a 3 = 4 := sorry
def a_6_condition : a 6 = 1 / 2 := sorry

-- the general form of the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * q ^ n

-- the theorem we want to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 4) (h2 : a 6 = 1 / 2) 
  (hg : geometric_sequence a q) : q = 1 / 2 :=
sorry

end find_common_ratio_l284_284009


namespace relationship_abc_l284_284397

noncomputable def my_function_condition_odd (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f(-x) = -f(x)

noncomputable def my_function_condition_deriv (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, x ≠ 0 → deriv (deriv f x) + f x / x > 0

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

axiom f_is_odd : my_function_condition_odd f
axiom f_second_deriv_cond : my_function_condition_deriv f

-- Definitions of a, b, c with given conditions.
noncomputable def a_val : ℝ := 1/2 * f (1/2)
noncomputable def b_val : ℝ := -2 * f (-2)
noncomputable def c_val : ℝ := real.log 2 * f (real.log 2)

-- Conjecture to be proved
theorem relationship_abc : b > c ∧ c > a :=
by
  have ha : a = a_val := by rfl
  have hb : b = b_val := by rfl
  have hc : c = c_val := by rfl
  sorry

end relationship_abc_l284_284397


namespace find_b_perpendicular_l284_284734

-- Definitions of the vectors and dot product
def v1 (b : ℝ) : ℝ × ℝ × ℝ := (b, -3, 2)
def v2 : ℝ × ℝ × ℝ := (2, 1, 3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Statement of the proof problem
theorem find_b_perpendicular (b : ℝ) (h : dot_product (v1 b) v2 = 0) : b = -3 / 2 :=
by {
  sorry -- The proof doesn't need to be done, only the statement is required
}

end find_b_perpendicular_l284_284734


namespace question1_question2_l284_284743

-- Given condition: A = { a | ∀ x ∈ ℝ, x^2 + 2 * a * x + 4 > 0 }
def A (a : ℝ) : Set ℝ := { x | ∀ x ∈ ℝ, x^2 + 2 * a * x + 4 > 0 }

-- Given condition: B = { x | 1 < (x + k) / 2 < 2 }
def B (k : ℝ) : Set ℝ := { x | 1 < (x + k) / 2 ∧ (x + k) / 2 < 2 }

-- Complement set Cₐ B = { x | x ∉ B }
def CompSet (B : Set ℝ) : Set ℝ := { x | ¬(x ∈ B) }

-- Statements
theorem question1 (k : ℝ) (hk : k = 1) : A (-2,2) ∩ CompSet (B k) = (-2, 1] := 
sorry

theorem question2 (k_range : Set ℝ) : (CompSet (A (-2,2)) ⊆ CompSet (B k) ∧ B k ⊆ A (-2,2)) → (2 ≤ k ∧ k ≤ 4) :=
sorry

end question1_question2_l284_284743


namespace area_range_of_triangle_l284_284579

-- Defining the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 2

-- Function to compute the area of triangle ABP
noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2))

-- The proof goal statement
theorem area_range_of_triangle (P : ℝ × ℝ) (hp : on_circle P) :
  2 ≤ area_of_triangle P ∧ area_of_triangle P ≤ 6 :=
sorry

end area_range_of_triangle_l284_284579


namespace abs_neg_one_third_l284_284200

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end abs_neg_one_third_l284_284200


namespace largest_possible_4_digit_number_exists_l284_284077

theorem largest_possible_4_digit_number_exists (A B C : ℕ) (hA : A ∈ finset.range 10) 
    (hB : B ∈ finset.range 10) (hC : C ∈ finset.range 10) (h_diff : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
    112 * A * C + 10 * B * C ≤ 8624 :=
begin
  sorry -- Proof not needed
end

end largest_possible_4_digit_number_exists_l284_284077


namespace problem_sqrt_lambda_plus_sqrt_mu_is_constant_l284_284885

noncomputable def lambda_mu_constant (y0 : ℝ) (h : -2 ≤ y0 ∧ y0 ≤ 1) : Prop :=
  let λ := (y0 + 2)^2 / 9
  let μ := (y0 - 1)^2 / 9
  (Real.sqrt λ + Real.sqrt μ = 1)

theorem problem_sqrt_lambda_plus_sqrt_mu_is_constant :
  ∀ (y0 : ℝ) (h : -2 ≤ y0 ∧ y0 ≤ 1), lambda_mu_constant y0 h :=
begin
  sorry -- The proof is omitted as required.
end

end problem_sqrt_lambda_plus_sqrt_mu_is_constant_l284_284885


namespace zero_point_exists_in_interval_l284_284575

noncomputable def f (x : ℝ) : ℝ := x + 2^x

theorem zero_point_exists_in_interval :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f x = 0 :=
by
  existsi -0.5 -- This is not a formal proof; the existi -0.5 is just for example purposes
  sorry

end zero_point_exists_in_interval_l284_284575


namespace transform_probability_in_S_l284_284142

noncomputable def S : Set ℂ := {z : ℂ | let x := z.re, y := z.im in -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2}
noncomputable def f (z : ℂ) : ℂ := (1/2 + complex.I / 2) * z

theorem transform_probability_in_S : 
  (∀ z ∈ S, f(z) ∈ S ∧ (uniform_probability f z) = 1) :=
sorry

end transform_probability_in_S_l284_284142


namespace max_positive_integers_in_circle_l284_284266

-- Definitions for the conditions
def valid_circle (l : list ℤ) : Prop :=
  (l.length = 100) ∧ ∀ i : ℕ, i < 100 → l.nth_le i (by simp [i, l.length]) > l.nth_le ((i + 1) % 100) (by sorry) + l.nth_le ((i + 2) % 100) (by sorry)

-- Statement of the proof problem
theorem max_positive_integers_in_circle (l : list ℤ) (h : valid_circle l) :
  (l.filter (λ x, x > 0)).length ≤ 49 :=
sorry

end max_positive_integers_in_circle_l284_284266


namespace cylinder_properties_l284_284654

def radius := 3
def height := 10
def π := 3.14

def lateral_surface_area (r h : ℕ) : ℕ := 2 * π * r * h
def total_surface_area (lateral_surface_area base_area : ℕ) : ℕ := lateral_surface_area + 2 * base_area
def volume (r h : ℕ) : ℕ := π * r^2 * h

theorem cylinder_properties :
  lateral_surface_area radius height = 188.4 ∧
  total_surface_area (lateral_surface_area radius height) (π * radius^2) = 244.92 ∧
  volume radius height = 282.6 := 
by
  sorry

end cylinder_properties_l284_284654


namespace general_term_formula_sum_of_sequence_l284_284039

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = 2 * a n - a 1)
  (h_arith : a 1 = 2 ∧ (a 2 + 1) - a 1 = (a 3) - (a 2 + 1))
  (n : ℕ) :
  a n = 2 ^ n :=
sorry

theorem sum_of_sequence (T : ℕ → ℝ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = 2 * a n - a 1)
  (h_arith : a 1 = 2 ∧ (a 2 + 1) - a 1 = (a 3) - (a 2 + 1))
  (n : ℕ) :
  T n = 2 - (n + 2) / (2 ^ n) :=
sorry

end general_term_formula_sum_of_sequence_l284_284039


namespace common_points_count_l284_284343

variable (x y : ℝ)

def curve1 : Prop := x^2 + 4 * y^2 = 4
def curve2 : Prop := 4 * x^2 + y^2 = 4
def curve3 : Prop := x^2 + y^2 = 1

theorem common_points_count : ∀ (x y : ℝ), curve1 x y ∧ curve2 x y ∧ curve3 x y → false := by
  intros
  sorry

end common_points_count_l284_284343


namespace find_prime_pairs_l284_284715
open Nat

/--
Problem: Prove that the pairs of prime numbers (p, q) such that for all integers x, 
x^(3*p*q) ≡ x (mod 3*p*q) are (11, 17) and (17, 11).
-/

theorem find_prime_pairs (x : ℤ) (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∀ x : ℤ, x^(3*p*q) ≡ x [MOD (3*p*q)]) ↔ (p = 11 ∧ q = 17) ∨ (p = 17 ∧ q = 11) :=
sorry

end find_prime_pairs_l284_284715


namespace sum_of_first_2015_terms_l284_284697

noncomputable def seq : ℕ → ℤ
| 1     := 1
| 2     := 3
| 3     := 5
| (n+1) := seq n - seq (n - 1) + seq (n - 2) -- for n ≥ 3

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, seq (i + 1))

theorem sum_of_first_2015_terms : S 2015 = 6045 :=
by
  sorry

end sum_of_first_2015_terms_l284_284697


namespace josh_and_carl_work_days_per_week_l284_284120

variable (d : ℕ)
variable (josh_hours_per_day : ℕ := 8)
variable (josh_days_per_week : ℕ := d)
variable (weeks_per_month : ℕ := 4)
variable (carl_hours_per_day : ℕ := josh_hours_per_day - 2)
variable (josh_hourly_wage : ℝ := 9)
variable (carl_hourly_wage : ℝ := josh_hourly_wage / 2)
variable (total_monthly_earnings : ℝ := 1980)

def josh_weekly_hours : ℕ := josh_hours_per_day * josh_days_per_week
def carl_weekly_hours : ℕ := carl_hours_per_day * josh_days_per_week
def josh_weekly_earnings : ℝ := josh_hourly_wage * josh_weekly_hours
def carl_weekly_earnings : ℝ := carl_hourly_wage * carl_weekly_hours
def josh_monthly_earnings : ℝ := josh_weekly_earnings * weeks_per_month
def carl_monthly_earnings : ℝ := carl_weekly_earnings * weeks_per_month
def combined_monthly_earnings : ℝ := josh_monthly_earnings + carl_monthly_earnings

theorem josh_and_carl_work_days_per_week :
  combined_monthly_earnings = total_monthly_earnings → d = 5 :=
by
  sorry

end josh_and_carl_work_days_per_week_l284_284120


namespace find_original_price_l284_284528

theorem find_original_price (reduced_price : ℝ) (percent : ℝ) (original_price : ℝ) 
  (h1 : reduced_price = 6) (h2 : percent = 0.25) (h3 : reduced_price = percent * original_price) : 
  original_price = 24 :=
sorry

end find_original_price_l284_284528


namespace product_of_divisors_of_72_l284_284360

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284360


namespace domain_f_l284_284052

def f (x : ℝ) : ℝ := Math.log (x - 2)

theorem domain_f :
  ∀ x : ℝ, f x ∈ ℝ ↔ x > 2 :=
by
  sorry

end domain_f_l284_284052


namespace gcd_987654_876543_eq_3_l284_284615

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l284_284615


namespace question_1_question_2_l284_284425

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

theorem question_1 :
  f = (λ x : ℝ, 2 * Real.sin (2 * x + (Real.pi / 6))) :=
sorry

theorem question_2 :
  ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2) → f x ∈ Set.Icc (-1) 2 :=
sorry

end question_1_question_2_l284_284425


namespace race_distance_l284_284680

theorem race_distance {a b c : ℝ} (h1 : b = 0.9 * a) (h2 : c = 0.95 * b) :
  let andrei_distance := 1000
  let boris_distance := andrei_distance - 100
  let valentin_distance := boris_distance - 50
  let valentin_actual_distance := (c / a) * andrei_distance
  andrei_distance - valentin_actual_distance = 145 :=
by
  sorry

end race_distance_l284_284680


namespace degree_product_is_six_l284_284246

-- Define the expressions
def expr1 := λ x : ℝ, x ^ 5
def expr2 := λ x : ℝ, x + (1 / x)
def expr3 := λ x : ℝ, 1 + (2 / x) + (3 / x^2)

-- Define the product of the expressions
def product := λ x : ℝ, expr1 x * expr2 x * expr3 x

-- Define the degree of the resulting polynomial (which we aim to prove is 6)
def degree_of_product (f : ℝ → ℝ) : ℕ := sorry -- Placeholder for the actual degree calculation function

-- The proof statement asserting the degree is 6
theorem degree_product_is_six : degree_of_product product = 6 := 
by sorry -- Placeholder for the proof

end degree_product_is_six_l284_284246


namespace polynomial_function_additive_implies_linear_l284_284896

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def k : ℝ := sorry

theorem polynomial_function_additive_implies_linear :
  (∀ (a b : ℝ), f(a + b) = f(a) + f(b)) ↔ ∃ k : ℝ, ∀ x : ℝ, f(x) = k * x :=
sorry

end polynomial_function_additive_implies_linear_l284_284896


namespace voting_proposal_l284_284474

theorem voting_proposal :
  ∀ (T Votes_against Votes_in_favor More_votes_in_favor : ℕ),
    T = 290 →
    Votes_against = (40 * T) / 100 →
    Votes_in_favor = T - Votes_against →
    More_votes_in_favor = Votes_in_favor - Votes_against →
    More_votes_in_favor = 58 :=
by sorry

end voting_proposal_l284_284474


namespace expression_evaluation_l284_284711

noncomputable def evaluate_expression : ℝ :=
  (Real.sin (38 * Real.pi / 180) * Real.sin (38 * Real.pi / 180) 
  + Real.cos (38 * Real.pi / 180) * Real.sin (52 * Real.pi / 180) 
  - Real.tan (15 * Real.pi / 180) ^ 2) / (3 * Real.tan (15 * Real.pi / 180))

theorem expression_evaluation : 
  evaluate_expression = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end expression_evaluation_l284_284711


namespace interior_diagonals_sum_l284_284664

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 64) 
  (h2 : 2 * (a * b + b * c + c * a) = 206) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 2 :=
by
  -- Definitions and assumptions
  have h_sum : a + b + c = 16, from by linarith,
  have h_surface : a * b + b * c + c * a = 103, from by linarith,

  -- Squaring and relating sums of sides
  have h_squared : (a + b + c) ^ 2 = a^2 + b^2 + c^2 + 2 * (a * b + b * c + c * a),
  have h_relate : 256 = a^2 + b^2 + c^2 + 206, from by linarith,

  -- Solving for a^2 + b^2 + c^2
  have h_squares : a^2 + b^2 + c^2 = 50, from by linarith,

  -- Conclusion on interior diagonals sum
  have h_diagonals : 4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 2, 
  sorry

end interior_diagonals_sum_l284_284664


namespace problem_l284_284020

variable (x y z A B C : ℝ)

def F (r : ℕ) : ℝ := x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem problem (h1 : A + B + C = k * Real.pi) (k : ℤ) (hF1 : F x y z A B C 1 = 0) (hF2 : F x y z A B C 2 = 0) : 
  ∀ r : ℕ, 0 < r → F x y z A B C r = 0 := 
by
  sorry

end problem_l284_284020


namespace students_adjacent_permutation_count_l284_284990

-- Defining the problem conditions
def number_of_students := 6
def students_adjacent (n : ℕ) := n = 2

-- Defining the permutation count when two specific students (A and B) must stand next to each other
def adjacent_permutations (n : ℕ) (adj : ℕ) := adj * (number_of_students - 1)!

-- The main statement
theorem students_adjacent_permutation_count :
  adjacent_permutations number_of_students (students_adjacent 2) = 240 :=
by
  sorry

end students_adjacent_permutation_count_l284_284990


namespace enhanced_ohara_triple_y_l284_284995

theorem enhanced_ohara_triple_y (a b : ℕ) (y : ℝ) (h : a = 49 ∧ b = 16 ∧ √a + √b + √(a + b) = y) :
  y = 11 + √65 :=
sorry

end enhanced_ohara_triple_y_l284_284995


namespace cone_height_is_sqrt3_l284_284595

-- Define the given conditions
def slant_height := 2
def lateral_area := 2 * Real.pi

-- Define the function to calculate the radius from the lateral area and slant height
def radius (lateral_area slant_height : ℝ) : ℝ :=
  lateral_area / (Real.pi * slant_height)

-- Use the radius to find the height using the Pythagorean theorem
def height (slant_height radius : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - radius ^ 2)

-- Define the proof problem statement
theorem cone_height_is_sqrt3 : height slant_height (radius lateral_area slant_height) = Real.sqrt 3 :=
  sorry

end cone_height_is_sqrt3_l284_284595


namespace horner_method_multiplications_and_additions_l284_284686

noncomputable def f (x : ℕ) : ℕ :=
  12 * x ^ 6 + 5 * x ^ 5 + 11 * x ^ 2 + 2 * x + 5

theorem horner_method_multiplications_and_additions (x : ℕ) :
  let multiplications := 6
  let additions := 4
  multiplications = 6 ∧ additions = 4 :=
sorry

end horner_method_multiplications_and_additions_l284_284686


namespace problem_equation_has_solution_l284_284107

noncomputable def x (real_number : ℚ) : ℚ := 210 / 23

theorem problem_equation_has_solution (x_value : ℚ) : 
  (3 / 7) + (7 / x_value) = (10 / x_value) + (1 / 10) → 
  x_value = 210 / 23 :=
by
  intro h
  sorry

end problem_equation_has_solution_l284_284107


namespace arithmetic_sequence_m_value_l284_284396

theorem arithmetic_sequence_m_value (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) 
  (h_seq : ∀ n : ℕ, S n = (n + 1) / 2 * (2 * a₁ + n * d)) :
  m = 5 :=
by
  sorry

end arithmetic_sequence_m_value_l284_284396


namespace political_alignment_time_l284_284601

def expected_time_alignment : ℚ :=
  341 / 54

theorem political_alignment_time :
  (expected_time_needed : ℚ) 
  (students : ℕ) 
  (initial_democrats : ℕ) 
  (initial_republicans : ℕ)
  (groups : ℕ)
  (group_size : ℕ)
  (player_switch : ℕ → ℕ)
  (expected_time : ℚ) :
  students = 12 ∧
  initial_democrats = 6 ∧
  initial_republicans = 6 ∧
  groups = 4 ∧
  group_size = 3 ∧
  (∀ n, player_switch n = if n % 2 = 0 then n else n + 1) →
  expected_time = expected_time_alignment :=
sorry

end political_alignment_time_l284_284601


namespace sum_of_every_second_term_l284_284666

variable (x : ℕ → ℕ)

-- Conditions
def sequence_conditions :=
  ∀ n, (n < 999 → x (n + 1) = x n + 1) ∧
       ∑ i in Finset.range 1000, x i = 15000

-- The statement we want to prove: the sum of every second term starting with the first and ending with the second last term is 7250.
theorem sum_of_every_second_term (h : sequence_conditions x) :
  ∑ i in (Finset.range 500).map (Fin.backwardMap Fin.succ) ≈ x = 7250 :=
sorry

end sum_of_every_second_term_l284_284666


namespace find_sum_pqr_l284_284191

theorem find_sum_pqr (p q r : ℤ)
  (h_gcd : Polynomial.gcd (Polynomial.X ^ 2 + Polynomial.C p * Polynomial.X + Polynomial.C q) (Polynomial.X ^ 2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = Polynomial.X - 1)
  (h_lcm : Polynomial.lcm (Polynomial.X ^ 2 + Polynomial.C p * Polynomial.X + Polynomial.C q) (Polynomial.X ^ 2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - 5 * Polynomial.X + 6) :
  p + q + r = -2 :=
sorry

end find_sum_pqr_l284_284191


namespace volume_difference_of_sphere_and_cylinder_l284_284284

theorem volume_difference_of_sphere_and_cylinder (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ) 
  (hs : r_sphere = 6) (rc : r_cylinder = 4) (hc : h_cylinder = 4 * Real.sqrt 5) :
  (4/3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * h_cylinder) = (288 - 64 * Real.sqrt 5) * Real.pi :=
by
  -- Given conditions
  have h1 : r_sphere = 6 := hs,
  have h2 : r_cylinder = 4 := rc,
  have h3 : h_cylinder = 4 * Real.sqrt 5 := hc,
  sorry

end volume_difference_of_sphere_and_cylinder_l284_284284


namespace new_average_after_doubling_l284_284561

theorem new_average_after_doubling
  (avg : ℝ) (num_students : ℕ) (h_avg : avg = 40) (h_num_students : num_students = 10) :
  let total_marks := avg * num_students
  let new_total_marks := total_marks * 2
  let new_avg := new_total_marks / num_students
  new_avg = 80 :=
by
  sorry

end new_average_after_doubling_l284_284561


namespace other_endpoint_diameter_l284_284308

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end other_endpoint_diameter_l284_284308


namespace AM_third_AC1_l284_284757

-- Definition of points and the intersection
variable {Point : Type} [AffineSpace Point]

variables 
  (A B C D A1 B1 C1 D1 M : Point)

-- Assume the points form a parallelepiped
variable (isParallelepiped : ∀ P Q R S U V W X : Point, P ≠ Q ∧ U ≠ V ∧ V ≠ W ∧ W ≠ X ∧ X ≠ U)

-- M is the intersection of the diagonal AC1 with the plane A1 - B - D
variable (isIntersection : ∀ P Q R S T U V : Point, ∃ T, T lies_on diagonal(P, U) ∩ plane(Q, R, S))

-- Proving the relationship
theorem AM_third_AC1 (h1 : isParallelepiped A B C D A1 B1 C1 D1) 
  (h2 : isIntersection A C1 A1 B D M) : distance A M = (distance A C1) / 3 :=
by
  sorry

end AM_third_AC1_l284_284757


namespace necessary_water_quarts_l284_284998

def water_to_lemon_ratio := 5 / 3
def gallons := 2
def quarts_per_gallon := 4
def total_parts := 5 + 3
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

theorem necessary_water_quarts :
  let water_parts := 5 in
  water_parts * quarts_per_part = 5 :=
by
  -- proof omitted
  sorry

end necessary_water_quarts_l284_284998


namespace find_x_squared_plus_y_squared_l284_284446

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l284_284446


namespace range_of_a_l284_284105

variables (e a : ℝ) (t : ℝ)
-- Condition 1: e is a unit vector
def is_unit_vector (e : ℝ) := e = 1

-- Condition 2: a · e = 2
def dot_product_eq_two (a e : ℝ) := a * e = 2

-- Condition 3: |a|^2 ≤ 5 |a + t * e| for all t
def magnitude_inequality_holds (a e t : ℝ) := 
  abs a^2 ≤ 5 * abs (a + t * e)

-- Theorem: |a| lies in the range [sqrt(5), 2 * sqrt(5)]
theorem range_of_a (h1 : is_unit_vector e)
                   (h2 : dot_product_eq_two a e)
                   (h3 : ∀ t, magnitude_inequality_holds a e t) :
  abs a ∈ set.Icc (real.sqrt 5) (2 * real.sqrt 5) :=
sorry

end range_of_a_l284_284105


namespace part_I_part_II_l284_284749

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x - b / x + log x
noncomputable def g (m x : ℝ) : ℝ := x ^ 2 - 2 * m * x + m

theorem part_I {a b : ℝ} (h1 : f a b 1 - f a b (1 / 2) = 0)
  (h2 : f a b (1 / 2) - f a b 1 = 0) : a = -1 / 3 ∧ b = -1 / 3 := sorry

theorem part_II {f g : ℝ → ℝ} (m : ℝ)
  (h : ∀ (x1 : ℝ),
    x1 ∈ set.Icc (1 / 2) 2 →
    ∃ (x2 : ℝ), x2 ∈ set.Icc (1 / 2) 2 ∧ g x1 ≥ f x2 - log x2) : m ≤ (3 + Real.sqrt 51) / 6 := sorry

end part_I_part_II_l284_284749


namespace proof_x_squared_plus_y_squared_l284_284448

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l284_284448


namespace cross_product_and_orthogonality_proof_l284_284305

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   -(a.1 * b.2.2 - a.2.2 * b.1),
   a.1 * b.2.1 - a.2.1 * b.1)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem cross_product_and_orthogonality_proof :
  let a := (4, 3, -5) in
  let b := (2, -1, 4) in
  let cp := cross_product a b in
  cp = (7, -26, -10) ∧ dot_product cp a = 0 :=
by
  let a := (4, 3, -5)
  let b := (2, -1, 4)
  let cp := cross_product a b
  have h1 : cp = (7, -26, -10) := sorry
  have h2 : dot_product cp a = 0 := sorry
  exact ⟨h1, h2⟩

end cross_product_and_orthogonality_proof_l284_284305


namespace solve_quadratic_l284_284186

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l284_284186


namespace band_members_150_l284_284216

theorem band_members_150 :
  ∃ (N : ℕ), N + 2 = 8 * (N / 8) + 2 ∧ N + 3 = 9 * (N / 9) + 3 ∧ 100 ≤ N ∧ N ≤ 200 ∧ N = 150 :=
by
  have h1 : ∃ k, N + 2 = 8 * k := sorry
  have h2 : ∃ m, N + 3 = 9 * m := sorry
  have h3 : 100 ≤ N := sorry
  have h4 : N ≤ 200 := sorry
  existsi 150
  split
  -- Proof for each of the conditions
  { sorry },
  { sorry },
  { sorry },
  { sorry },
  { refl }

end band_members_150_l284_284216


namespace range_of_fx_l284_284024

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (h : k > 0) : 
  set.range (λ x : ℝ, if x ≥ 0.5 then f x k else 0) = set.Ioo ((0.5)^k) ⊤ :=
sorry

end range_of_fx_l284_284024


namespace cos_540_eq_neg_one_l284_284323

theorem cos_540_eq_neg_one : Real.cos (540 : ℝ) = -1 := by
  sorry

end cos_540_eq_neg_one_l284_284323


namespace a_general_term_b_general_term_sum_Tn_condition_l284_284761

/-
Given:
1. A sequence {a_n} such that s_n = 2a_n - 2 where s_n is the sum of the first n terms of {a_n}
2. A sequence {b_n} such that b_1 = 1 and b_{n+1} = b_n + 2

To prove:
1. The general term of {a_n} is a_n = 2^n 
2. The general term of {b_n} is b_n = 2n - 1
3. The sum of the first n terms of c_n = a_n * b_n is T_n, and the largest integer n such that T_n < 167 is 4
-/

def seq_s (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i+1)

def a_n (n : ℕ) : ℕ := 2 ^ n

def b_n (n : ℕ) : ℕ := 2 * n - 1

def c_n (n : ℕ) : ℕ := a_n n * b_n n

def T_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, c_n (i+1)

theorem a_general_term (n : ℕ) : 
  seq_s n a_n = 2 * a_n n - 2 := sorry

theorem b_general_term (n : ℕ) :
  ∀ n > 0, b_n(n+1) = b_n(n) + 2 ∧ b_n(1) = 1 := sorry

theorem sum_Tn_condition (n : ℕ) : 
  T_n 4 < 167 ∧ ∀ m > 4, T_n m ≥ 167 := sorry

end a_general_term_b_general_term_sum_Tn_condition_l284_284761


namespace length_of_UR_l284_284206

open_locale classical

-- Define the points P, Q, R, S as vertices of the square.
-- Define T as midpoint of RS.
-- Define U on QR such that ∠SPT = ∠TPU
structure square_geometry :=
(P Q R S T U : ℝ × ℝ)
(side_length_eq : dist P Q = 2 ∧ dist Q R = 2 ∧ dist R S = 2 ∧ dist S P = 2)
(midpoint_T : T = ((fst R + fst S) / 2, (snd R + snd S) / 2))
(PT_angle_eq_TPU : ∃ θ : ℝ, angle S P T = θ ∧ angle T P U = θ)
(U_on_QR : ∃ u : ℝ, U = (fst Q + u * (fst R - fst Q), snd Q + u * (snd R - snd Q)))

noncomputable 
def length_UR_is_half (g : square_geometry) : Prop :=
dist g.U g.R = 1/2

-- We state the theorem
theorem length_of_UR (g : square_geometry) : length_UR_is_half g :=
sorry

end length_of_UR_l284_284206


namespace average_page_count_l284_284099

theorem average_page_count 
  (n1 n2 n3 n4 : ℕ)
  (p1 p2 p3 p4 total_students : ℕ)
  (h1 : n1 = 8)
  (h2 : p1 = 3)
  (h3 : n2 = 10)
  (h4 : p2 = 5)
  (h5 : n3 = 7)
  (h6 : p3 = 2)
  (h7 : n4 = 5)
  (h8 : p4 = 4)
  (h9 : total_students = 30) :
  (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) / total_students = 36 / 10 := 
sorry

end average_page_count_l284_284099


namespace sum_mod_9_l284_284190

theorem sum_mod_9 (x y z : ℕ) (h1 : x < 9) (h2 : y < 9) (h3 : z < 9) 
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : (x * y * z) % 9 = 1) (h8 : (7 * z) % 9 = 4) (h9 : (8 * y) % 9 = (5 + y) % 9) :
  (x + y + z) % 9 = 7 := 
by {
  sorry
}

end sum_mod_9_l284_284190


namespace monotonic_increase_interval_l284_284427

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem monotonic_increase_interval :
  (∀ x : ℝ, f x = Real.sin (2 * x + Real.pi / 6))
  ∧ (0 < 2 ∧ 2 < 4)
  ∧ (|Real.pi / 6| < Real.pi / 2)
  ∧ (f (Real.pi / 6) - f (2 * Real.pi / 3) = 2) →
  (∃ k : ℤ, 
    ∀ x : ℝ, 
    (k * Real.pi - Real.pi / 3 ≤ x) 
    ∧ (x ≤ k * Real.pi + Real.pi / 6)) :=
begin
  sorry
end

end monotonic_increase_interval_l284_284427


namespace lines_pass_through_incenter_excenter_l284_284091

-- Define the given conditions
variables {A B C D E F I I_A : Point}
variables (triangle_ABC : Triangle A B C) (circumcircle_triangle_ABC : Circumcircle triangle_ABC)
variables (line_ell : Line tangent (circumcircle_triangle_ABC) A)
variables (circle_center_A : Circle A (dist A C))
variables (point_D_on_segment_AB : D ∈ Segment A B)
variables (point_E_on_line_ell : E ∈ line_ell)
variables (point_F_on_line_ell : F ∈ line_ell)

-- Define further given conditions
variables (AD_eq_AC : dist A D = dist A C)

-- State the theorem to be proved
theorem lines_pass_through_incenter_excenter
  (H_incenter : is_incenter I triangle_ABC)
  (H_excenter : is_excenter I_A triangle_ABC)
  (line_DE : Line D E)
  (line_DF : Line D F) :
  passes_through line_DE I ∧ passes_through line_DF I_A :=
sorry

end lines_pass_through_incenter_excenter_l284_284091


namespace find_a_l284_284002

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l284_284002


namespace Ronaldinho_age_2018_l284_284161

variable (X : ℕ)

theorem Ronaldinho_age_2018 (h : X^2 = 2025) : X - (2025 - 2018) = 38 := by
  sorry

end Ronaldinho_age_2018_l284_284161


namespace length_of_BC_is_8_l284_284296

-- Define the coordinates of vertices A, B, and C based on given conditions
variables {a : ℝ}
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1 - a, (1 - a)^2)
def C : ℝ × ℝ := (1 + a, (1 + a)^2)

-- The function for calculating the area of a triangle given vertex points
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Length of BC
noncomputable def length_BC : ℝ :=
  abs ((1 + a) - (1 - a))

-- Main theorem statement
theorem length_of_BC_is_8 (h_area : triangle_area A B C = 128) : length_BC = 8 :=
sorry

end length_of_BC_is_8_l284_284296


namespace geometric_progression_product_of_squares_l284_284755

variable (a r : ℝ) (n : ℕ)
def P (a r : ℝ) (n : ℕ) : ℝ := (a ^ 2) * ((a * r) ^ 2) * ((a * r^2) ^ 2) * ... * ((a * r^(n-1) ) ^ 2)

def S (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

def S' (a r : ℝ) (n : ℕ) : ℝ := (1 / a) * (1 - (1 / r)^n) / (1 - 1 / r)

theorem geometric_progression_product_of_squares (a r : ℝ) (n : ℕ) :
  P a r n = (S a r n * S' a r n) ^ (n / 2) := by
  sorry

end geometric_progression_product_of_squares_l284_284755


namespace angle_ADO_eq_angle_HAN_l284_284887

variables (O H A B C M D N : Point)
variables (circumcircle_BHC : Circle)
variables (circumcenter O : Point)
variables (orthocenter H : Point)
variables [AcuteTriangle A B C]
variables [IsOnCircumcenter O A B C]
variables [IsOnOrthocenter H A B C]
variables [IsMidPoint M B C]
variables [IsAngleBisector D A B C]    -- Angle bisector of ∠BAC by point D
variables [IsOnLine N M O]
variables [IsOnCircumcircle circumcircle_BHC N]

theorem angle_ADO_eq_angle_HAN :
  ∠ A D O = ∠ H A N :=
sorry

end angle_ADO_eq_angle_HAN_l284_284887


namespace min_shift_symmetry_l284_284417

theorem min_shift_symmetry (ω a : ℝ) (hω : ω > 0) (hf_period : ∀ x, (sin(ω * x))^2 - 1/2 = (sin(ω * (x + π)))^2 - 1/2)
    (hf_shift_symmetry : ∀ x, -1/2 * cos(2 * (x - a)) = -1/2 * cos(2 * (- x + a))) :
  a = π / 4 :=
  sorry

end min_shift_symmetry_l284_284417


namespace range_of_a_l284_284776

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_odd (f : ℝ → ℝ) (h_odd: is_odd f) : Prop :=
  ∀ x : ℝ, x < 0 → f x = - x^2 + x → f (-x) = x^2 - x

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_odd: is_odd f)
  (h_fx_neg: ∀ x : ℝ, x < 0 → f x = - x^2 + x) :
  (∀ x : ℝ, 0 < x ∧ x ≤ (sqrt 2 / 2) → f x - x ≤ 2 * log a x) ∧ a > 0 ∧ a ≠ 1 ↔ 
  (1/4 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l284_284776


namespace print_time_is_fifteen_l284_284661

noncomputable def time_to_print (total_pages rate : ℕ) := 
  (total_pages : ℚ) / rate

theorem print_time_is_fifteen :
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  round time = 15 := by
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  have time_val : time = (350 : ℚ) / 24 := by rfl
  let rounded_time := round time
  have rounded_time_val : rounded_time = 15 := by sorry
  exact rounded_time_val

end print_time_is_fifteen_l284_284661


namespace pauly_needs_10_pounds_l284_284537

theorem pauly_needs_10_pounds (ounces_per_cube : ℕ) (weight_per_cube : ℝ)
                                (cubes_per_hour : ℕ) (cost_per_hour : ℝ)
                                (cost_per_ounce_water : ℝ) (total_cost : ℝ) :
  ounces_per_cube = 2 →
  weight_per_cube = 1 / 16 →
  cubes_per_hour = 10 →
  cost_per_hour = 1.50 →
  cost_per_ounce_water = 0.10 →
  total_cost = 56 →
  let cost_per_cube := (ounces_per_cube * cost_per_ounce_water) + (cost_per_hour / cubes_per_hour) in
  let number_of_cubes := total_cost / cost_per_cube in
  let total_weight := number_of_cubes * weight_per_cube in
  total_weight = 10 := 
by
  intros
  sorry

end pauly_needs_10_pounds_l284_284537


namespace value_of_b_l284_284460

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 :=
by
  intro h
  -- Proving that b = 6
  sorry

end value_of_b_l284_284460


namespace monotonicity_of_f_inequality_f_l284_284008

section
variables {f : ℝ → ℝ}
variables (h_dom : ∀ x, x > 0 → f x > 0)
variables (h_f2 : f 2 = 1)
variables (h_fxy : ∀ x y, f (x * y) = f x + f y)
variables (h_pos : ∀ x, 1 < x → f x > 0)

-- Monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Inequality f(x) + f(x-2) ≤ 3 
theorem inequality_f (x : ℝ) :
  2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
sorry

end

end monotonicity_of_f_inequality_f_l284_284008


namespace find_varphi_l284_284212

theorem find_varphi (ϕ : ℝ) (h0 : 0 < ϕ ∧ ϕ < π / 2) :
  (∀ x₁ x₂, |(2 * Real.cos (2 * x₁)) - (2 * Real.cos (2 * x₂ - 2 * ϕ))| = 4 → 
    ∃ (x₁ x₂ : ℝ), |x₁ - x₂| = π / 6 
  ) → ϕ = π / 3 :=
by
  sorry

end find_varphi_l284_284212


namespace abs_neg_one_third_l284_284199

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end abs_neg_one_third_l284_284199


namespace lambda_mu_value_l284_284860

-- Given definitions
variables {α : Type*} [AddCommGroup α] [VectorSpace ℝ α] [FiniteDimensional ℝ α]

def ratio_AM_MC (A M C : α) : Prop := A - M = 2 • (M - C)

def BM_linear_combination (B M A C : α) (λ μ : ℝ) : Prop := B - M = λ • (B - A) + μ • (B - C)

-- The proof problem
theorem lambda_mu_value (A B C M : α) (λ μ : ℝ) (h1 : ratio_AM_MC A M C) (h2 : BM_linear_combination B M A C λ μ) :
  λ - μ = -1/3 :=
sorry

end lambda_mu_value_l284_284860


namespace incorrect_value_in_table_l284_284846

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_value_in_table :
  ∃ a b c : ℝ,
  let values := [3844, 3989, 4144, 4311, 4496, 4689, 4892, 5105] in
  let second_differences := [for i in [1: values.length - 1],
                              (values[i] - values[i - 1]) - (values[i - 1] - values[i - 2])]
  (4496 ∉ values) ∧
  second_differences ≠ [repeat second_differences.head (second_differences.length)] :=
sorry

end incorrect_value_in_table_l284_284846


namespace min_value_of_expr_l284_284904

noncomputable def real.min_value_expr (x y z : ℝ) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expr :
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    real.min_value_expr x y z = 4 * (real.sqrt (real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expr_l284_284904


namespace largest_n_exists_l284_284723

theorem largest_n_exists (n : ℤ) (h1 : ∃ m : ℤ, n^2 = m^3 - 1) (h2 : ∃ a : ℤ, 2 * n + 83 = a^2) : n ≤ 139 :=
by
  sorry

example : ∃ n : ℤ, (n^2 = 27^3 - 1) ∧ (2 * n + 83 = 19^2) ∧ n = 139 :=
by
  use 139
  split
  { 
    show 139^2 = 27^3 - 1,
    sorry 
  }
  split
  {
    show 2 * 139 + 83 = 19^2,
    sorry
  }
  show 139 = 139,
  refl

end largest_n_exists_l284_284723


namespace almonds_addition_l284_284492

theorem almonds_addition (walnuts almonds total_nuts : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_total_nuts : total_nuts = 0.5)
  (h_sum : total_nuts = walnuts + almonds) : 
  almonds = 0.25 := by
  sorry

end almonds_addition_l284_284492


namespace find_n_l284_284075

theorem find_n (n : ℝ) : 4^9 = 16^n → n = 4.5 := by
  intro h
  sorry

end find_n_l284_284075


namespace average_age_group_l284_284632

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = 15 * n) (h2 : T + 37 = 17 * (n + 1)) : n = 10 :=
by
  sorry

end average_age_group_l284_284632


namespace strictly_increasing_function_l284_284718

theorem strictly_increasing_function (f : ℕ → ℕ) (h1 : ∀ x y : ℕ, (f(x) + f(y)) % (1 + f(x + y)) = 0) (h2 : ∀ x y : ℕ, (f(x) + f(y)) / (1 + f(x + y)) > 0) :
  ∃ a : ℕ, (0 < a) ∧ (∀ x : ℕ, f(x) = a * x + 1) :=
sorry

end strictly_increasing_function_l284_284718


namespace petya_payment_l284_284877

theorem petya_payment : 
  ∃ (x y : ℕ), 
  (14 * x + 3 * y = 107) ∧ 
  (|x - y| ≤ 5) ∧
  (x + y = 10) := 
sorry

end petya_payment_l284_284877


namespace product_fraction_l284_284688

theorem product_fraction :
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693 / 256 := by
  sorry

end product_fraction_l284_284688


namespace intersection_complement_l284_284435

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_complement_l284_284435


namespace find_C_value_l284_284033

open Real

theorem find_C_value (C : ℝ) : 
  let x0 := -1
  let y0 := 2
  let A := 4
  let B := -3
  let d := 1
  abs(A * x0 + B * y0 + C) / sqrt(A^2 + B^2) = d ↔ (C = 5 ∨ C = 15) := by
  let x0 := -1
  let y0 := 2
  let A := 4
  let B := -3
  let d := 1
  sorry

end find_C_value_l284_284033


namespace double_factorial_divisible_l284_284541

open Nat

theorem double_factorial_divisible :
  ∃ k : ℕ, 1985 !! + 1986 !! = 1987 * k :=
  sorry

end double_factorial_divisible_l284_284541


namespace license_plate_palindrome_probability_l284_284930

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l284_284930


namespace complex_fraction_evaluation_l284_284504

open Complex

theorem complex_fraction_evaluation (c d : ℂ) (hz : c ≠ 0) (hz' : d ≠ 0) (h : c^2 + c * d + d^2 = 0) :
  (c^12 + d^12) / (c^3 + d^3)^4 = 1 / 8 := 
by sorry

end complex_fraction_evaluation_l284_284504


namespace exists_digit_sum_divisible_by_13_l284_284173

def sum_of_digits (n : ℕ) : ℕ := n.toString.foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

theorem exists_digit_sum_divisible_by_13 (n : ℕ) :
  ∃ m ∈ (list.range' n 79), sum_of_digits m % 13 = 0 :=
by
  sorry

end exists_digit_sum_divisible_by_13_l284_284173


namespace sum_of_roots_of_quadratic_l284_284081

theorem sum_of_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (Polynomial.eval x1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) ∧ 
                 (Polynomial.eval x2 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) -> 
                 x1 + x2 = 3 := 
by
  intro x1 x2
  intro H
  sorry

end sum_of_roots_of_quadratic_l284_284081


namespace simplify_expression_l284_284956

theorem simplify_expression (x y : ℝ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 :=
by
  sorry

end simplify_expression_l284_284956


namespace bike_to_tractor_speed_ratio_l284_284204

theorem bike_to_tractor_speed_ratio
  (car_speed_ratio : ℚ)
  (tractor_distance : ℚ)
  (tractor_time : ℚ)
  (car_distance : ℚ)
  (car_time : ℚ) 
  (B : ℚ)
  (T : ℚ) :
  car_speed_ratio = 9 / 5 →
  tractor_distance = 575 →
  tractor_time = 25 →
  car_distance = 331.2 →
  car_time = 4 →
  T = tractor_distance / tractor_time →
  B = car_distance / car_time / car_speed_ratio →
  (B / T) = 2 :=
by
  intro hcar_speed_ratio htractor_distance htractor_time hcar_distance hcar_time hT hB
  rw [hcar_speed_ratio, htractor_distance, htractor_time, hcar_distance, hcar_time, hT, hB]
  norm_num
  sorry

end bike_to_tractor_speed_ratio_l284_284204


namespace min_value_of_y_l284_284388

theorem min_value_of_y (x : ℝ) (hx : x > 0) : (∃ y, y = x + 4 / x^2 ∧ ∀ z, z = x + 4 / x^2 → z ≥ 3) :=
sorry

end min_value_of_y_l284_284388


namespace part_I_part_II_min_max_l284_284422

noncomputable def f (x : ℝ) : ℝ :=
  1 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

noncomputable def g (x : ℝ) : ℝ :=
  f (x - Real.pi / 6)

def monotonic_intervals : Prop :=
  ∀ k : ℤ, 
    ∃ a b, [a, b] = (k * Real.pi - Real.pi / 3, k * Real.pi + Real.pi / 6) ∨ 
    [a, b] = (k * Real.pi + Real.pi / 6, k * Real.pi + 2 * Real.pi / 3)

def min_max_g : Prop :=
  ∀ x ∈ Icc (-Real.pi / 2) (0 : ℝ), 
    -2 ≤ g x ∧ g x ≤ 1

theorem part_I : monotonic_intervals := sorry

theorem part_II_min_max : min_max_g := sorry

end part_I_part_II_min_max_l284_284422


namespace problem_1_problem_2_l284_284059

theorem problem_1 (a : ℝ) (h_pos : 0 < a) (h_inc : ∀ x ∈ Set.Ici 1, (a * x - 1) / (a * x^2) ≥ 0) : 1 ≤ a :=
sorry

theorem problem_2 (n : ℕ) (h_n : 2 ≤ n) : 
  ∑ i in Finset.range n \ {0, 1}, (1 / i) < Real.log n ∧ Real.log n < 1 + ∑ i in Finset.range (n - 1) \ {0} , (1 / i) :=
sorry

end problem_1_problem_2_l284_284059


namespace chord_length_l284_284970

-- Define the circle centered at (3,0) with radius 3
def circle (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the line equation
def line (x y : ℝ) := 3 * x - 4 * y - 4 = 0

/-- The proof that the chord length intersected 
by the line from the circle is 4 * sqrt 2 -/
theorem chord_length (x y : ℝ) (h_circle : circle x y) (h_line : line x y) : 
  ∃ l, l = 4 * real.sqrt 2 :=
sorry

end chord_length_l284_284970


namespace log_base_30_of_8_l284_284331

theorem log_base_30_of_8 (a b : Real) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
    Real.logb 30 8 = 3 * (1 - a) / (b + 1) := 
  sorry

end log_base_30_of_8_l284_284331


namespace interval_monotonically_increasing_cos_2theta_eq_7_25_l284_284065

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, cos (2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin (2 * x), -√3)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

theorem interval_monotonically_increasing (k : ℤ) :
  ∀ x, f x = 2 * sin (2 * x - π / 3) → k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 := sorry

theorem cos_2theta_eq_7_25 (θ : ℝ) :
  f (θ / 2 + 2 * π / 3) = 6 / 5 → cos (2 * θ) = 7 / 25 := sorry

end interval_monotonically_increasing_cos_2theta_eq_7_25_l284_284065


namespace roots_quadratic_sum_product_l284_284409

theorem roots_quadratic_sum_product :
  (∀ x1 x2 : ℝ, (∀ x, x^2 - 4 * x + 3 = 0 → x = x1 ∨ x = x2) → (x1 + x2 - x1 * x2 = 1)) :=
by
  sorry

end roots_quadratic_sum_product_l284_284409


namespace log_probability_l284_284434

def A : Set ℕ := {1, 2, 3, 4, 5, 6}

def isLogGreaterThanTwo (a b : ℕ) : Prop :=
  a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ log (a : ℝ) (b : ℝ) > 2

theorem log_probability : (count (isLogGreaterThanTwo) (pairs A (a b))) / (count pairs A (a b)) = (1 / 10) :=
sorry

end log_probability_l284_284434


namespace geometric_series_sum_l284_284510

theorem geometric_series_sum : 
  let a := 6
  let r := - (2 / 5)
  let s := a / (1 - r)
  s = 30 / 7 :=
by
  let a := 6
  let r := -(2 / 5)
  let s := a / (1 - r)
  show s = 30 / 7
  sorry

end geometric_series_sum_l284_284510


namespace product_fractions_result_l284_284306

noncomputable def product_fractions (n m : Nat) : ℚ :=
if h : n + 3 * m = 2005 then ∏ i in Finset.range m, ((n + 3 * i) : ℚ) / ((n + 3 * i) + 3) else 0

theorem product_fractions_result : 
  product_fractions 2 667 = (2 : ℚ) / 2005 := 
sorry

end product_fractions_result_l284_284306


namespace find_xy_l284_284025

theorem find_xy (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 :=
by
  sorry

end find_xy_l284_284025


namespace cone_lateral_area_l284_284774

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end cone_lateral_area_l284_284774


namespace find_DE_value_l284_284921

-- Define the unit square ABCD
structure Point :=
(x : ℚ)
(y : ℚ)

def A := Point.mk 0 1
def B := Point.mk 1 1
def C := Point.mk 1 0
def D := Point.mk 0 0

-- Define the circle and its properties
def r : ℚ := 32 / 49
def E : Point := Point.mk 1 1 -- this is a placeholder; E needs to be found in the proof

-- Define the distance function
def distance (p1 p2 : Point) : ℚ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem find_DE_value :
  ∃ (m n : ℕ), m = 8 ∧ n = 7 ∧ Real.gcd m n = 1 ∧ distance D E = (m / n : ℚ) ∧ (100 * m + n = 807) :=
begin
  sorry
end

end find_DE_value_l284_284921


namespace sphere_radius_l284_284634

/-- Given the curved surface area (CSA) of a sphere and its formula, 
    prove that the radius of the sphere is 4 cm.
    Conditions:
    - CSA = 4πr²
    - Curved surface area is 64π cm²
-/
theorem sphere_radius (r : ℝ) (h : 4 * Real.pi * r^2 = 64 * Real.pi) : r = 4 := by
  sorry

end sphere_radius_l284_284634


namespace henry_kombucha_bottles_l284_284810

theorem henry_kombucha_bottles :
  ∀ (monthly_bottles: ℕ) (cost_per_bottle refund_rate: ℝ) (months_in_year total_bottles_in_year: ℕ),
  (monthly_bottles = 15) →
  (cost_per_bottle = 3.0) →
  (refund_rate = 0.10) →
  (months_in_year = 12) →
  (total_bottles_in_year = monthly_bottles * months_in_year) →
  (total_refund = refund_rate * total_bottles_in_year) →
  (bottles_bought_with_refund = total_refund / cost_per_bottle) →
  bottles_bought_with_refund = 6 :=
by
  intros monthly_bottles cost_per_bottle refund_rate months_in_year total_bottles_in_year
  sorry

end henry_kombucha_bottles_l284_284810


namespace sin_C_in_triangle_l284_284859

theorem sin_C_in_triangle (A B C : ℝ) (h1 : B = 60)
  (h2 : cos A = 3 / 5) :
  sin C = (3 * real.sqrt 3 + 4) / 10 :=
sorry

end sin_C_in_triangle_l284_284859


namespace license_plate_palindrome_probability_l284_284929

noncomputable def is_palindrome_prob : ℚ := 775 / 67600

theorem license_plate_palindrome_probability:
  is_palindrome_prob.num + is_palindrome_prob.denom = 68375 := by
  sorry

end license_plate_palindrome_probability_l284_284929


namespace coloring_impossible_l284_284114

theorem coloring_impossible :
  ¬ ∃ (color : ℕ → Prop), (∀ n m : ℕ, (m = n + 5 → color n ≠ color m) ∧ (m = 2 * n → color n ≠ color m)) :=
sorry

end coloring_impossible_l284_284114


namespace max_k_for_3_pow_11_as_sum_of_consec_integers_l284_284727

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end max_k_for_3_pow_11_as_sum_of_consec_integers_l284_284727


namespace problem_statement_l284_284097

-- Definitions of the geometric setup
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Hypotheses from the problem conditions
variables (AB AC BC CD : ℝ) (angle_ABC : ℝ)
variable h_AB : AB = 5
variable h_BC : BC = 7
variable h_angle_ABC : angle_ABC = 150

-- The lengths and properties associated with the problem
-- Perpendicular lines meet at point D

noncomputable def length_CD : ℝ :=
  (170 * real.sqrt 3) / 21

-- The statement to prove
theorem problem_statement : CD = length_CD :=
sorry

end problem_statement_l284_284097


namespace monotonic_range_of_t_l284_284789

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end monotonic_range_of_t_l284_284789


namespace planes_perpendicular_l284_284386

open Real

def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  u1 * v1 + u2 * v2 + u3 * v3 = 0

theorem planes_perpendicular :
  ∀ (u v : ℝ × ℝ × ℝ),
  u = (-2, 2, 5) →
  v = (6, -4, 4) →
  is_perpendicular u v :=
by
  intros u v hu hv
  rw [hu, hv]
  have : (-2) * 6 + 2 * (-4) + 5 * 4 = 0 := by norm_num
  exact this
-- sorry to skip the proof

end planes_perpendicular_l284_284386


namespace probability_of_sum_9_is_one_tenth_l284_284629

-- Define the sets a and b
def set_a : set ℕ := {2, 3, 4, 5}
def set_b : set ℕ := {4, 5, 6, 7, 8}

-- Function to compute the probability of the sum being 9
def probability_sum_9 (a b : set ℕ) : ℚ :=
  let favorable_pairs := (a.product b).filter (λ p, p.1 + p.2 = 9) in
  favorable_pairs.card / (a.card * b.card : ℚ)

-- Theorem statement
theorem probability_of_sum_9_is_one_tenth : probability_sum_9 set_a set_b = 1 / 10 := by
  sorry

end probability_of_sum_9_is_one_tenth_l284_284629


namespace find_other_endpoint_l284_284310

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_endpoint (O A B : ℝ × ℝ) 
    (hO : O = (2, 3)) 
    (hA : A = (-1, -1))
    (hMidpoint : O = midpoint A B) : 
    B = (5, 7) :=
sorry

end find_other_endpoint_l284_284310


namespace probability_both_selected_l284_284234

theorem probability_both_selected (p_ram : ℚ) (p_ravi : ℚ) (h_ram : p_ram = 5/7) (h_ravi : p_ravi = 1/5) : 
  (p_ram * p_ravi = 1/7) := 
by
  sorry

end probability_both_selected_l284_284234


namespace negation_equiv_l284_284580

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l284_284580


namespace evaluate_expression_l284_284328

-- Definitions and conditions from part (a)
def i : ℂ := Complex.I
axiom i_pow_four : i^4 = 1
axiom i_inv : i^(-1) = 1 / i

-- The theorem we want to prove
theorem evaluate_expression : i^10 + i^20 + i^(-34) + 2 = 1 :=
by
  sorry

end evaluate_expression_l284_284328


namespace exists_five_distinct_nat_numbers_l284_284486

theorem exists_five_distinct_nat_numbers 
  (a b c d e : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_no_div_3 : ¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e))
  (h_no_div_4 : ¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e))
  (h_no_div_5 : ¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) :
  (∃ (a b c d e : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e)) ∧
    (¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e)) ∧
    (¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) ∧
    (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y + z = a + b + c + d + e → (x + y + z) % 3 = 0) ∧
    (∀ w x y z : ℕ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z → w + x + y + z = a + b + c + d + e → (w + x + y + z) % 4 = 0) ∧
    (a + b + c + d + e) % 5 = 0) :=
  sorry

end exists_five_distinct_nat_numbers_l284_284486


namespace cuboid_diagonal_angles_l284_284399

theorem cuboid_diagonal_angles (α β γ : ℝ) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 :=
sorry

end cuboid_diagonal_angles_l284_284399


namespace find_point_on_z_axis_l284_284481

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

def point1 := (1 : ℝ, 0 : ℝ, 2 : ℝ)
def point2 := (1 : ℝ, -3 : ℝ, 1 : ℝ)

def on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)

theorem find_point_on_z_axis : ∃ z, 
  distance (on_z_axis z) point1 = distance (on_z_axis z) point2 ∧ 
  (on_z_axis z) = (0, 0, -1) :=
by
  sorry

end find_point_on_z_axis_l284_284481


namespace min_value_inequality_l284_284913

theorem min_value_inequality
    (x y z : ℝ)
    (h1 : 2 ≤ x)
    (h2 : x ≤ y)
    (h3 : y ≤ z)
    (h4 : z ≤ 5) :
    (x - 2) ^ 2 + (y / x - 1) ^ 2 + (z / y - 1) ^ 2 + (5 / z - 1) ^ 2 ≥ 4 * (Real.sqrt (4 : ℝ) 5 - 1) ^ 2 :=
by
    sorry

end min_value_inequality_l284_284913


namespace minimum_routes_islandland_l284_284487

-- Define the conditions
variable (V : Type) [fintype V] [decidable_eq V]
variable [has_size_of V]
variable (G : simple_graph V)
noncomputable def num_vertices := 10

-- Converting the Problem to Lean Definitions
theorem minimum_routes_islandland (hV : fintype.card V = num_vertices) :
  (∀ (V' : finset V), V'.card = 9 → (∃ (C : cycle G), to_finset C.verts = V')) →
  G.edge_finset.card ≥ 15 :=
by {
  sorry
}

end minimum_routes_islandland_l284_284487


namespace can_inscribe_circle_l284_284638

noncomputable def inscribable_quadrilateral (A B C D : Point) 
  (R_A R_B R_C R_D : ℝ) 
  (P Q R S : Point) : Prop :=
  ∃ (tangentPQ tangentRS tangentPS tangentQR : Line),
    tangentPQ.is_tangent_to_Circle (Circle A R_A) ∧ tangentPQ.is_tangent_to_Circle (Circle C R_C) ∧
    tangentRS.is_tangent_to_Circle (Circle C R_C) ∧ tangentRS.is_tangent_to_Circle (Circle D R_D) ∧
    tangentPS.is_tangent_to_Circle (Circle A R_A) ∧ tangentPS.is_tangent_to_Circle (Circle B R_B) ∧
    tangentQR.is_tangent_to_Circle (Circle B R_B) ∧ tangentQR.is_tangent_to_Circle (Circle D R_D) ∧
    tangentPQ.intersect tangentRS P ∧ tangentRS.intersect tangentPS Q ∧
    tangentPS.intersect tangentQR R ∧ tangentQR.intersect tangentPQ S ∧
    PQ_length + RS_length = PS_length + QR_length

theorem can_inscribe_circle (A B C D : Point) 
  (R_A R_B R_C R_D : ℝ)
  (h_nonintersect : ¬intersecting_Circles (Circle A R_A) (Circle C R_C)) 
  (h_nonintersect' : ¬intersecting_Circles (Circle B R_B) (Circle D R_D))
  (h_radii : R_A + R_C = R_B + R_D) : 
  inscribable_quadrilateral A B C D R_A R_B R_C R_D :=
begin
  sorry
end

end can_inscribe_circle_l284_284638


namespace find_intersection_points_l284_284721

def intersection_points (t α : ℝ) : Prop :=
∃ t α : ℝ,
  (2 + t, -1 - t) = (3 * Real.cos α, 3 * Real.sin α) ∧
  ((2 + t = (1 + Real.sqrt 17) / 2 ∧ -1 - t = (1 - Real.sqrt 17) / 2) ∨
   (2 + t = (1 - Real.sqrt 17) / 2 ∧ -1 - t = (1 + Real.sqrt 17) / 2))

theorem find_intersection_points : intersection_points t α :=
sorry

end find_intersection_points_l284_284721


namespace square_of_inverse_sum_is_integer_l284_284172

theorem square_of_inverse_sum_is_integer (a : ℝ) (h : a + 1/a ∈ ℤ) : a^2 + 1/a^2 ∈ ℤ :=
by
  sorry

end square_of_inverse_sum_is_integer_l284_284172


namespace sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l284_284780

variables (A B C a b c S : ℝ)
variables (h_area : S = (a + b) ^ 2 - c ^ 2) (h_sum : a + b = 4)
variables (h_triangle : ∀ (x : ℝ), x = sin C)

open Real

theorem sin_C_value_proof :
  sin C = 8 / 17 :=
sorry

theorem a2_b2_fraction_proof :
  (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C :=
sorry

theorem sides_sum_comparison :
  a ^ 2 + b ^ 2 + c ^ 2 ≥ 4 * sqrt 3 * S :=
sorry

end sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l284_284780


namespace min_value_sum_inverse_sq_l284_284390

theorem min_value_sum_inverse_sq (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 1) : 
  (39 + 1/x + 4/y + 9/z) ≥ 25 :=
by
    sorry

end min_value_sum_inverse_sq_l284_284390


namespace odd_and_periodic_l284_284815

variable {f : ℝ → ℝ}

-- Condition definitions
def cond1 (x : ℝ) : Prop := f (10 + x) = f (10 - x)
def cond2 (x : ℝ) : Prop := f (20 - x) = -f (20 + x)

-- Theorem statement
theorem odd_and_periodic :
  (∀ x : ℝ, cond1 x) →
  (∀ x : ℝ, cond2 x) →
  (∀ x : ℝ, f x = -f (-x)) ∧ (∀ x : ℝ, f x = f (x + 20)) :=
by
  intros h1 h2
  sorry

end odd_and_periodic_l284_284815


namespace ratio_sum_trapezoid_areas_l284_284482

-- Define the trapezoid and its properties
structure Trapezoid (EF GH: ℝ) :=
  (EG EH: ℝ) (GE_eq_HE: EG = EH) 
  (EF_GH_parallel: EF > 0 ∧ GH > 0)

-- Define the specific trapezoid EFGH with the given conditions
def trapezoidEFGH := Trapezoid 14 28 15 15 rfl (by norm_num)

-- Define the midsegment connecting the midpoints of EG and EH
def midsegmentLength (EG EH: ℝ) : ℝ := (EG + EH) / 2

-- Define the height GK
def heightGK (EG EF: ℝ) : ℝ := Real.sqrt (EG^2 - (EF / 2)^2)

-- Define the areas of the trapezoids
def areaTrapezoid (height: ℝ) (base1 base2: ℝ) : ℝ := (height * (base1 + base2)) / 2

-- Define the ratio of areas and sum of the numbers in the ratio
theorem ratio_sum_trapezoid_areas : 
  let height := heightGK 15 14,
      areaEFGH := areaTrapezoid height 14 28,
      areaEFIJ := areaTrapezoid (height / 2) 14 14,
      areaIJGH := areaTrapezoid (height / 2) 28 28 in
  (areaEFIJ / areaIJGH = 1 / 4) ∧ (1 + 4 = 5) := 
by {
  sorry
}

end ratio_sum_trapezoid_areas_l284_284482


namespace find_k_b_correct_l284_284061

noncomputable def find_k_b (k b : ℝ) : Prop :=
  let line := (λ x : ℝ, k * x + b)
  let circle1 := (λ x y : ℝ, x^2 + y^2 = 1)
  let circle2 := (λ x y : ℝ, (x - 4)^2 + y^2 = 1)
  ∃ (k : ℝ) (hk : k > 0) (b : ℝ),
    (∀ x y, circle1 x y → (y = line x) ∨ (y = line x → y ≠ y)) ∧
    (∀ x y, circle2 x y → (y = line x) ∨ (y = line x → y ≠ y)) ∧
    k = (Real.sqrt 3) / 3 ∧ b = -2 * (Real.sqrt 3) / 3

theorem find_k_b_correct :
  find_k_b ((Real.sqrt 3) / 3) (-2 * (Real.sqrt 3) / 3) := 
  sorry

end find_k_b_correct_l284_284061


namespace quadratic_inequality_solution_l284_284322

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5 * x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end quadratic_inequality_solution_l284_284322


namespace problem_statement_l284_284289

noncomputable def erased_number (n : ℕ) (average : ℚ) (x : ℕ) : Prop :=
  let S := (n * (n + 1)) / 2
  in let new_sum := S - x
  in let new_average := (new_sum : ℚ) / (n - 1)
  in new_average = average

noncomputable def correct_erased_number (x : ℕ) : Prop :=
  ∃ n : ℕ, erased_number n (45 + (11 / 19) : ℚ) x

theorem problem_statement : correct_erased_number 326 :=
sorry

end problem_statement_l284_284289


namespace solve_exponential_equation_l284_284584

theorem solve_exponential_equation (x : ℝ) (h : 3^(3^x) = 333) : 1 < x ∧ x < 2 := 
by 
  -- Place solution steps here.
  sorry

end solve_exponential_equation_l284_284584


namespace product_of_divisors_of_72_l284_284354

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284354


namespace triangle_third_side_length_l284_284459

theorem triangle_third_side_length:
  ∃ n : ℕ, n = 15 ∧ (∀ x : ℕ, 3 < x ∧ x < 19 → x ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}) :=
by {
  sorry
}

end triangle_third_side_length_l284_284459


namespace incenter_inequality_l284_284511

variable (A B C I P : Type)
variables [Incenter A B C I]
variables [PointInsideTriangle P A B C]
variables [AngleRelation P B A P C A P B C P C B]

theorem incenter_inequality :
  (AP : ℝ) ≥ (AI : ℝ) ∧ ((AP : ℝ) = (AI : ℝ) ↔ P = I) :=
by
  -- We start from the geometric definitions and required angle relations
  sorry

end incenter_inequality_l284_284511


namespace cos_identity_l284_284406

theorem cos_identity (α : ℝ) (h : cos (70 * Real.pi / 3 - α) = -1 / 3) :
  cos (70 * Real.pi / 3 + 2 * α) = -7 / 9 := sorry

end cos_identity_l284_284406


namespace min_subset_rel_prime5_l284_284512

def is_pairwise_rel_prime (lst : List ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ lst → y ∈ lst → x ≠ y → Nat.gcd x y = 1

theorem min_subset_rel_prime5 :
  let S := Set.range (λ n, n + 1) 280 in
  ∃ (n : ℕ), (∀ (T : Finset ℕ), T.card = n → T ⊆ S → (∃ (lst : List ℕ), lst.length = 5 ∧ is_pairwise_rel_prime lst)) ∧ n = 217 := 
by
  sorry

end min_subset_rel_prime5_l284_284512


namespace monotonicity_of_f_sum_of_zeros_l284_284754

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem monotonicity_of_f (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : 6 * a + b = 0) (h₂ : f a b c 1 = 4 * a) :
  (a > 0 → (∀ x, x < 1 → deriv (f a b c) x > 0) ∧ (∀ x, 1 < x ∧ x < 3 → deriv (f a b c) x < 0) ∧ (∀ x, x > 3 → deriv (f a b c) x > 0)) ∧
  (a < 0 → (∀ x, x < 1 → deriv (f a b c) x < 0) ∧ (∀ x, 1 < x ∧ x < 3 → deriv (f a b c) x > 0) ∧ (∀ x, x > 3 → deriv (f a b c) x < 0)) :=
begin
  sorry
end

noncomputable def F (a b c x : ℝ) : ℝ := f a b c x - x * real.exp (-x)

theorem sum_of_zeros (a b c x1 x2 x3 : ℝ)
  (h₀ : a ≠ 0) (h₁ : 6 * a + b = 0) (h₂ : f a b c 1 = 4 * a)
  (h₃ : 0 ≤ x1) (h₄ : x1 < x2) (h₅ : x2 < x3) (h₆ : x3 ≤ 3)
  (h₇ : F a b c x1 = 0) (h₈ : F a b c x2 = 0) (h₉ : F a b c x3 = 0) :
  x1 + x2 + x3 < 2 :=
begin
  sorry
end

end monotonicity_of_f_sum_of_zeros_l284_284754


namespace friedEdgeProb_l284_284740

-- Define a data structure for positions on the grid
inductive Pos
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq, Repr

-- Define whether a position is an edge square (excluding corners)
def isEdge : Pos → Prop
| Pos.A2 | Pos.A3 | Pos.B1 | Pos.B4 | Pos.C1 | Pos.C4 | Pos.D2 | Pos.D3 => True
| _ => False

-- Define the initial state and max hops
def initialState := Pos.B2
def maxHops := 5

-- Define the recursive probability function (details omitted for brevity)
noncomputable def probabilityEdge (p : Pos) (hops : Nat) : ℚ := sorry

-- The proof problem statement
theorem friedEdgeProb :
  probabilityEdge initialState maxHops = 94 / 256 := sorry

end friedEdgeProb_l284_284740


namespace golden_section_length_l284_284828

theorem golden_section_length (MN : ℝ) (MP NP : ℝ) (hMN : MN = 1) (hP : MP + NP = MN) (hgolden : MN / MP = MP / NP) (hMP_gt_NP : MP > NP) : MP = (Real.sqrt 5 - 1) / 2 :=
by sorry

end golden_section_length_l284_284828


namespace total_stars_l284_284547

-- Define the daily stars earned by Shelby
def shelby_monday : Nat := 4
def shelby_tuesday : Nat := 6
def shelby_wednesday : Nat := 3
def shelby_thursday : Nat := 5
def shelby_friday : Nat := 2
def shelby_saturday : Nat := 3
def shelby_sunday : Nat := 7

-- Define the daily stars earned by Alex
def alex_monday : Nat := 5
def alex_tuesday : Nat := 3
def alex_wednesday : Nat := 6
def alex_thursday : Nat := 4
def alex_friday : Nat := 7
def alex_saturday : Nat := 2
def alex_sunday : Nat := 5

-- Define the total stars earned by Shelby in a week
def total_shelby_stars : Nat := shelby_monday + shelby_tuesday + shelby_wednesday + shelby_thursday + shelby_friday + shelby_saturday + shelby_sunday

-- Define the total stars earned by Alex in a week
def total_alex_stars : Nat := alex_monday + alex_tuesday + alex_wednesday + alex_thursday + alex_friday + alex_saturday + alex_sunday

-- The proof problem statement
theorem total_stars (total_shelby_stars total_alex_stars : Nat) : total_shelby_stars + total_alex_stars = 62 := by
  sorry

end total_stars_l284_284547


namespace average_minutes_run_per_student_l284_284302

variable (e : ℕ)

axiom sixth_graders_avg : ℕ := 20
axiom seventh_graders_avg : ℕ := 12
axiom eighth_graders_avg : ℕ := 18

axiom sixth_graders_num : ℕ := 3 * e
axiom seventh_graders_num : ℕ := 3 * e
axiom eighth_graders_num : ℕ := e

theorem average_minutes_run_per_student :
  (6 * sixth_graders_avg + 3 * seventh_graders_avg + eighth_graders_avg) / (7 : ℝ) = 16.2857 := by
  sorry

end average_minutes_run_per_student_l284_284302


namespace find_principal_amount_correct_l284_284377

noncomputable def find_principal_amount : ℝ :=
  let r1 := 10 / 100 / 2  -- Semi-annual rate for the first year (5%)
  let r2 := 12 / 100 / 2  -- Semi-annual rate for the second year (6%)
  let CI := (P : ℝ) → P * (1 + r1)^2 * (1 + r2)^2 - P
  let SI := (P : ℝ) → P * 0.10 + P * 0.12
  let diff := 15
  (P : ℝ) := P * ((1 + r1)^2 * (1 + r2)^2 - 1) - SI P = diff

theorem find_principal_amount_correct : find_principal_amount ≈ 825.40 := by
  sorry

end find_principal_amount_correct_l284_284377


namespace imag_conj_of_z_l284_284420

open Complex

-- Define the given conditions
noncomputable def b : ℝ := 2
def z : ℂ := Complex.mk 1 b
def z_squared_eq : z^2 = Complex.mk (-3) 4 := by sorry

-- Lean statement to prove the problem
theorem imag_conj_of_z : 
  (z * conj z).im = -2 := by
  -- proof steps will go here
  sorry

end imag_conj_of_z_l284_284420


namespace quadratic_has_equal_roots_l284_284830

-- Proposition: If the quadratic equation 3x^2 + 6x + m = 0 has two equal real roots, then m = 3.

theorem quadratic_has_equal_roots (m : ℝ) : 3 * 6 - 12 * m = 0 → m = 3 :=
by
  intro h
  sorry

end quadratic_has_equal_roots_l284_284830


namespace find_interest_rate_l284_284156

def principal : ℝ := 5525.974025974026
def total_amount_returned : ℝ := 8510
def time_years : ℕ := 9

def interest_rate (P : ℝ) (A : ℝ) (T : ℕ) : ℝ := ((A - P) * 100) / (P * T)

theorem find_interest_rate :
  interest_rate principal total_amount_returned time_years ≈ 6 := sorry

end find_interest_rate_l284_284156


namespace product_of_divisors_of_72_l284_284368

theorem product_of_divisors_of_72 :
  ∏ (d : ℕ) in {d | ∃ a b : ℕ, 72 = a * b ∧ d = a}, d = 139314069504 := sorry

end product_of_divisors_of_72_l284_284368


namespace find_valid_n_l284_284334

def is_valid_n (n : ℕ) : Prop :=
  ∀ A : Finset ℕ, A.card = 35 ∧ A ⊆ Finset.range 51 → 
    ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b = n ∨ a + b = n)

theorem find_valid_n : ∀ n : ℕ, n ∈ Finset.range' 1 70 → is_valid_n n :=
begin
  let M := Finset.range' 1 50,
  intros n hn,
  have H : ∀ A : Finset ℕ, A.card = 35 ∧ A ⊆ M → 
    ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b = n ∨ a + b = n),
  { sorry },
  exact H,
end

end find_valid_n_l284_284334


namespace true_propositions_l284_284786

theorem true_propositions :
  let prop1 := ∀ (P Q : Plane) (l : Line), Perpendicular l P → P ≠ Q → PassesThrough Q l → Perpendicular P Q
  let prop2 := ∀ (P Q R : Plane) (l1 l2 : Line), Parallel l1 l2 → InPlane l1 P → InPlane l2 P → Parallel P Q → Parallel P R
  let prop3 := ∀ (l1 l2 m : Line), Parallel l1 l2 → Perpendicular l1 m → Perpendicular l2 m
  let prop4 := ∀ (P Q : Plane) (l : Line), Perpendicular P Q → Intersects l P → ¬(Perpendicular l (Intersection P Q)) → ¬(Perpendicular l Q)
  prop1 ∧ prop3 ∧ prop4 :=
by {
  -- proof can be filled here, but we use sorry to indicate proof is omitted
  sorry
}

end true_propositions_l284_284786


namespace determine_possible_values_l284_284558

noncomputable def fifth_roots_of_unity : Set ℂ :=
  {1, Complex.exp (2 * Real.pi * Complex.I / 5),
   Complex.exp (4 * Real.pi * Complex.I / 5),
   Complex.exp (6 * Real.pi * Complex.I / 5),
   Complex.exp (8 * Real.pi * Complex.I / 5)}

theorem determine_possible_values
  (p q r s t m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m ∈ fifth_roots_of_unity :=
begin
  sorry
end

end determine_possible_values_l284_284558


namespace jerry_total_shingles_l284_284867

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end jerry_total_shingles_l284_284867


namespace minimum_translation_for_symmetry_l284_284977

theorem minimum_translation_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, sin (2 * (x + m) - π / 6) = sin (2 * (-x - m) - π / 6)) ∧ m = π / 6 :=
begin
  sorry
end

end minimum_translation_for_symmetry_l284_284977


namespace prove_Z_gasoline_percentage_l284_284678

def percentage_Z_gasoline
(tank_capacity : ℕ)
(Z_init : ℝ)
(Y_added1 : ℝ)
(X_added : ℝ)
(Y_added2 : ℝ)
(Z_added : ℝ) :=
let Z_amount1 := Z_init * (3 / 4)
let Z_amount2 := Z_amount1 * (1 / 2)
let Z_amount3 := Z_amount2 * (1 / 2)
let Z_amount4 := Z_amount3 * (1 / 4)
let final_Z := Z_amount4 + Z_added
in (final_Z / tank_capacity) * 100

def total_gasoline_capacity := 100
def init_Z := 100.0
def added_Y1 := 25.0
def added_X := 50.0
def added_Y2 := 50.0
def added_Z := 75.0

theorem prove_Z_gasoline_percentage : 
  percentage_Z_gasoline total_gasoline_capacity init_Z added_Y1 added_X added_Y2 added_Z = 79.6875
  := by
  sorry

end prove_Z_gasoline_percentage_l284_284678


namespace quadratic_inequality_solution_l284_284958

def discriminant (m : ℝ) : ℝ := m^2 + 12 * m

def roots (m : ℝ) : Option (ℝ × ℝ) :=
  let Δ := discriminant m
  if Δ < 0 then none
  else some ((m - real.sqrt Δ) / 6, (m + real.sqrt Δ) / 6)

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, 3 * x^2 - m * x - m > 0 ↔ 
    (m = 0 ∨ m = -12 ∧ x ≠ m / 6) ∨
    (m < -12 ∨ m > 0 ∧ (x < (m - real.sqrt (discriminant m)) / 6 ∨ x > (m + real.sqrt (discriminant m)) / 6)) ∨
    (-12 < m ∧ m < 0 ∧ true)) :=
begin
  sorry,
end

end quadratic_inequality_solution_l284_284958


namespace part1_part2_l284_284057

def f (a x : ℝ) : ℝ := x^2 * log x - a * (x^2 - 1)

theorem part1 (a : ℝ) (h1 : ∀ x > 0, deriv (f a) x = x * (2 * log x + 1 - 2 * a)) (h2 : deriv (f a) 1 = 0) :
  a = 1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) (h1 : ∀ x > 0, deriv (f a) x = x * (2 * log x + 1 - 2 * a)) (h3 : ∀ x ≥ 1, f a x ≥ 0) :
  a ≤ 1 / 2 :=
by
  sorry

end part1_part2_l284_284057


namespace part1_part2_part3_l284_284139

-- Part (1): Proving \( p \implies m > \frac{3}{2} \)
theorem part1 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0) → (m > 3 / 2) :=
by
  sorry

-- Part (2): Proving \( q \implies (m < -1 \text{ or } m > 2) \)
theorem part2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → (m < -1 ∨ m > 2) :=
by
  sorry

-- Part (3): Proving \( (p ∨ q) \implies ((-\infty, -1) ∪ (\frac{3}{2}, +\infty)) \)
theorem part3 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0 ∨ ∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → ((m < -1) ∨ (3 / 2 < m)) :=
by
  sorry

end part1_part2_part3_l284_284139


namespace isosceles_obtuse_triangle_smaller_angle_l284_284301

theorem isosceles_obtuse_triangle_smaller_angle :
  ∀ (A B C : ℝ), (A = 162) → (A + B + C = 180) → (B = C) → B = 9 :=
by
  intros A B C h1 h2 h3
  rw [←h1, ←h3] at h2
  sorry

end isosceles_obtuse_triangle_smaller_angle_l284_284301


namespace quotient_has_no_two_zeros_in_middle_l284_284699

theorem quotient_has_no_two_zeros_in_middle : ¬ (∃ q : ℝ, q = 4.227 / 3 ∧ q_has_two_zeros_in_middle q) :=
by
  -- the proof goes here
  sorry

-- Definitions to support the theorem
-- q_has_two_zeros_in_middle: We need a function to check if a real number has two zeros in the middle
def q_has_two_zeros_in_middle (q : ℝ) : Prop :=
  -- Implementation of this function can include checking string representation etc.
  sorry

end quotient_has_no_two_zeros_in_middle_l284_284699


namespace evaluate_expression_l284_284551

theorem evaluate_expression : 
  let x := (-2 : ℝ)
  let y := (1/2 : ℝ)
  in (x + 2 * y)^2 - (x + y) * (3 * x - y) - 5 * y^2 = -10 := 
by 
  let x := (-2 : ℝ)
  let y := (1/2 : ℝ)
  calc 
    (x + 2 * y)^2 - (x + y) * (3 * x - y) - 5 * y^2 = 
    sorry 

end evaluate_expression_l284_284551


namespace convex_poly_props_l284_284936

-- Define a convex polygon
structure ConvexPolygon (P : Type) [AffineSpace P (EuclideanSpace ℝ)] where
  vertices : Set P
  is_convex: ∀ (x y : P), x ∈ vertices → y ∈ vertices → ∀ t ∈ Icc (0 : ℝ) 1, t • x + (1 - t) • y ∈ vertices

-- State the properties
def AllInteriorAnglesLessThan180 (P : Type) [AffineSpace P (EuclideanSpace ℝ)] (poly : ConvexPolygon P) : Prop :=
  ∀ (x y z : P), x ∈ poly.vertices ∧ y ∈ poly.vertices ∧ z ∈ poly.vertices →
  angle x y z < π

def AnySegmentLiesWithin (P : Type) [AffineSpace P (EuclideanSpace ℝ)] (poly : ConvexPolygon P) : Prop :=
  ∀ (x y : P), x ∈ poly.vertices ∧ y ∈ poly.vertices →
  segment x y ⊆ poly.vertices

-- The theorem to prove
theorem convex_poly_props (P : Type) [AffineSpace P (EuclideanSpace ℝ)] (poly : ConvexPolygon P) :
  (AllInteriorAnglesLessThan180 P poly) ∧ (AnySegmentLiesWithin P poly) ↔ 
  (∀ (x y : P), x ∈ poly.vertices → y ∈ poly.vertices → ∀ t ∈ Icc (0 : ℝ) 1, t • x + (1 - t) • y ∈ poly.vertices) := 
sorry

end convex_poly_props_l284_284936


namespace polar_eq_circle_l284_284589

theorem polar_eq_circle (a : ℝ) (ρ θ : ℝ) : ρ = 2 * a * cos θ :=
  sorry

end polar_eq_circle_l284_284589


namespace symmetric_point_l284_284802

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (2, 7)) (h2 : 1 * (a - 2) + (b - 7) * (-1) = 0) (h3 : (a + 2) / 2 + (b + 7) / 2 + 1 = 0) :
  (a, b) = (-8, -3) :=
sorry

end symmetric_point_l284_284802


namespace min_value_of_expression_l284_284917

noncomputable def min_of_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expression :
  ∃ x y z : ℝ, (2 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧ (z ≤ 5) ∧ min_of_expression x y z = 4 * (Real.sqrt (Real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expression_l284_284917


namespace charity_donation_ratio_l284_284488

theorem charity_donation_ratio :
  let total_winnings := 114
  let hot_dog_cost := 2
  let remaining_amount := 55
  let donation_amount := 114 - (remaining_amount + hot_dog_cost)
  donation_amount = 55 :=
by
  sorry

end charity_donation_ratio_l284_284488


namespace find_a9_l284_284130

variable (a : ℕ → ℤ)  -- Arithmetic sequence
variable (S : ℕ → ℤ)  -- Sum of the first n terms

-- Conditions provided in the problem
axiom Sum_condition : S 8 = 4 * a 3
axiom Term_condition : a 7 = -2
axiom Sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2

-- Hypothesis for common difference
def common_diff (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Proving that a_9 equals -6 given the conditions
theorem find_a9 (d : ℤ) : common_diff a d → a 9 = -6 :=
by
  intros h
  sorry

end find_a9_l284_284130


namespace rounds_before_player_out_of_tokens_l284_284472

-- Game settings and initial conditions
def initial_tokens_X := 18
def initial_tokens_Y := 15
def initial_tokens_Z := 12

def redistribute (X Y Z : ℕ) : ℕ × ℕ × ℕ :=
  if X ≥ Y ∧ X ≥ Z then (X - 3, Y + 1, Z + 1)
  else if Y ≥ X ∧ Y ≥ Z then (X + 1, Y - 3, Z + 1)
  else (X + 1, Y + 1, Z - 3)

-- Function to count rounds before a player runs out of tokens
noncomputable def rounds_until_out : ℕ → ℕ → ℕ → ℕ
| X, Y, Z :=
  if X = 0 ∨ Y = 0 ∨ Z = 0 then 0
  else 1 + rounds_until_out (redistribute X Y Z).1 (redistribute X Y Z).2 (redistribute X Y Z).3

-- Proof statement
theorem rounds_before_player_out_of_tokens :
  rounds_until_out initial_tokens_X initial_tokens_Y initial_tokens_Z = 28 :=
sorry

end rounds_before_player_out_of_tokens_l284_284472


namespace equal_roots_quadratic_k_eq_one_l284_284775

theorem equal_roots_quadratic_k_eq_one
  (k : ℝ)
  (h : ∃ x : ℝ, x^2 - 2 * x + k == 0 ∧ x^2 - 2 * x + k == 0) :
  k = 1 :=
by {
  sorry
}

end equal_roots_quadratic_k_eq_one_l284_284775


namespace remainder_of_expression_l284_284230

theorem remainder_of_expression (x y z : ℕ) (hxy : x < 7 ∧ y < 7 ∧ z < 7) 
  (hx : Nat.gcd x 7 = 1) (hy : Nat.gcd y 7 = 1) (hz : Nat.gcd z 7 = 1) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) : 
  ((x * y + y * z + z * x) * (Nat.modInv (x * y * z) 7)) % 7 = 2 :=
by
  sorry

end remainder_of_expression_l284_284230


namespace problem_statement_l284_284892

noncomputable def m_n_sum : ℕ :=
let AB := 13 in
let AC := 12 in
let BC := 5 in
37

theorem problem_statement
  (CH: ¬(CH = 0))
  {ACH: Type} [ACH: is_incircle (triangle ACH) (line CH)]
  {BCH: Type} [BCH: is_incircle (triangle BCH) (line CH)]
  (R: point_of_tangency ACH CH)
  (S: point_of_tangency BCH CH)
  (AB : ℕ) (hAB : AB = 13)
  (AC : ℕ) (hAC : AC = 12)
  (BC : ℕ) (hBC : BC = 5) :
  m_n_sum = 37 :=
  by sorry

end problem_statement_l284_284892


namespace ratio_of_speeds_l284_284476

noncomputable def time_A : ℝ := 2 -- A takes 2 hours
noncomputable def time_B : ℝ := 3/2 -- B takes 1.5 hours (i.e., 30 minutes less than A)
noncomputable def distance : ℝ := 1 -- assume a unit distance for simplicity

theorem ratio_of_speeds : 
  let v_A := distance / time_A in 
  let v_B := distance / time_B in
  (v_A / v_B) = 3 / 4 :=
by
  sorry

end ratio_of_speeds_l284_284476


namespace FCE_is_equilateral_l284_284920

variable {α : Type} [LinearOrder α]

section Parallelogram

-- Define points A, B, C, D, E, F and show the geometric properties
variables (A B C D E F : α)

-- Parallelogram condition: ABCD is a parallelogram
def is_parallelogram (A B C D : α) : Prop :=
  dist A B = dist C D ∧ dist B C = dist D A ∧ angle A B C = angle C D A ∧ angle B C D = angle D A B

-- Equilateral triangles conditions
def is_equilateral (A B C : α) : Prop := dist A B = dist B C ∧ dist B C = dist C A

-- FCE is equilateral goal
def is_equilateral_triangle_FCE : Prop := dist F C = dist C E ∧ dist C E = dist E F

-- Hypotheses: ABCD is a parallelogram and ABF, ADE are equilateral triangles constructed externally
axiom parallelogram_ABCD : is_parallelogram A B C D
axiom equilateral_ABF : is_equilateral A B F
axiom equilateral_ADE : is_equilateral A D E

-- The final goal to prove
theorem FCE_is_equilateral : is_equilateral_triangle_FCE :=
  sorry

end Parallelogram

end FCE_is_equilateral_l284_284920


namespace fred_sheets_left_l284_284382

def sheets_fred_had_initially : ℕ := 212
def sheets_jane_given : ℕ := 307
def planned_percentage_more : ℕ := 50
def given_percentage : ℕ := 25

-- Prove that after all transactions, Fred has 389 sheets left
theorem fred_sheets_left :
  let planned_sheets := (sheets_jane_given * 100) / (planned_percentage_more + 100)
  let sheets_jane_actual := planned_sheets + (planned_sheets * planned_percentage_more) / 100
  let total_sheets := sheets_fred_had_initially + sheets_jane_actual
  let charles_given := (total_sheets * given_percentage) / 100
  let fred_sheets_final := total_sheets - charles_given
  fred_sheets_final = 389 := 
by
  sorry

end fred_sheets_left_l284_284382


namespace muffin_selection_count_l284_284174

/-- Number of ways Sam can buy six muffins from four types: blueberry, chocolate chip, bran, almond -/
theorem muffin_selection_count : 
  ∀(b ch br a : ℕ), b + ch + br + a = 6 → 
    (finset.card (finset.filter (λ (x : finset ℕ), x.sum = 6) ((finset.range 4).powerset))) = 84 :=
begin
  intros b ch br a h,
  have h1 : nat.choose (6 + 4 - 1) 3 = 84,
  { calc nat.choose 9 3 = 84 : by norm_num },
  exact h1,
end

end muffin_selection_count_l284_284174


namespace arithmetic_mean_problem_l284_284967

noncomputable def find_x (mean : ℝ) (a1 a2 prod : ℝ) : ℝ :=
  let sum := mean * 4
  let y := real.sqrt prod
  sum - a1 - a2 - 2 * y

theorem arithmetic_mean_problem :
  let mean := 20
  let a1 := 12
  let prod := 400
  let x := find_x mean a1 0 prod
  x = 28 :=
by
  sorry

end arithmetic_mean_problem_l284_284967


namespace vector_AE_expression_l284_284484

variables {V : Type*} [AddCommGroup V] [Module ℝ V] {A B C E : V}

def is_interior_division (B C E : V) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ E = B + k • (C - B)

theorem vector_AE_expression (h : ∃ k : ℝ, k = 1/3 ∧ BE = k • EC) :
  AE = (3/4) • AB + (1/4) • AC :=
sorry

end vector_AE_expression_l284_284484


namespace positive_difference_of_solutions_l284_284220

theorem positive_difference_of_solutions :
  let a := 1
  let b := -6
  let c := -28
  let discriminant := b^2 - 4 * a * c
  let solution1 := 3 + (Real.sqrt discriminant) / 2
  let solution2 := 3 - (Real.sqrt discriminant) / 2
  have h_discriminant : discriminant = 148 := by sorry
  Real.sqrt 148 = 2 * Real.sqrt 37 :=
 sorry

end positive_difference_of_solutions_l284_284220


namespace trapezoid_ratio_limit_l284_284834

noncomputable def ratio_approach_one
  (O : Type) [metric_space O] (r d h : ℝ)
  (AB CD : ℝ → ℝ) 
  (R1 R2 : ℝ → ℝ)
  (h0 : CD = λ h, 2 * real.sqrt (r^2 - (d + h)^2) )
  (h1 : AB = 2 * real.sqrt (r^2 - d^2))
  (h2 : R1 = λ h, 1 / 2 * (AB + CD h) * h)
  (h3 : R2 = λ h, AB * h)
  : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, (0 < h ∧ h < δ) → abs ((R1 h / R2 h) - 1) < ε

theorem trapezoid_ratio_limit (O : Type) [metric_space O] (r d : ℝ) :
  ratio_approach_one O r d :=
begin
  sorry,
end

end trapezoid_ratio_limit_l284_284834


namespace frequency_of_group_5_l284_284653

theorem frequency_of_group_5 (total_students freq1 freq2 freq3 freq4 : ℕ)
  (h_total: total_students = 50) 
  (h_freq1: freq1 = 7) 
  (h_freq2: freq2 = 12) 
  (h_freq3: freq3 = 13) 
  (h_freq4: freq4 = 8) :
  (50 - (7 + 12 + 13 + 8)) / 50 = 0.2 :=
by
  sorry

end frequency_of_group_5_l284_284653


namespace circles_externally_tangent_l284_284436

-- Definitions for the two circles
def circle_1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 8 * y + 9 = 0

-- Centers and radii of the circles
def center_1 : ℝ × ℝ := (0, 0)
def radius_1 : ℝ := 1
def center_2 : ℝ × ℝ := (3, -4)
def radius_2 : ℝ := 4

-- Distance between the centers
def center_distance : ℝ := Real.sqrt ((3 - 0)^2 + (-4 - 0)^2)

-- Sum of the radii
def radii_sum : ℝ := radius_1 + radius_2

-- Proof that the two circles are externally tangent
theorem circles_externally_tangent : center_distance = radii_sum :=
by {
  -- Calculate the distance between centers
  show Real.sqrt (3^2 + (-4)^2) = 5,
  sorry
}

end circles_externally_tangent_l284_284436


namespace ac_length_l284_284578

theorem ac_length (AB : ℝ) (H1 : AB = 100)
    (BC AC : ℝ)
    (H2 : AC = (1 + Real.sqrt 5)/2 * BC)
    (H3 : AC + BC = AB) : AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry

end ac_length_l284_284578


namespace cot_sext_sub_cot_sq_l284_284441

theorem cot_sext_sub_cot_sq (x : ℝ) (hx : sin x ≠ 0 ∧ cos x ≠ 0) 
  (h : (cos x, sin x, tan x).GeometricalSequence) :
  (cot x)^6 - (cot x)^2 = 0 := by
  have h1 : sin x * sin x = cos x * tan x := sorry
  have h2 : tan x = sin x / cos x := sorry
  have h3 : sin x ≠ 0 → (sin x * sin x = cos x * (sin x / cos x)) := sorry
  have h4 : (sin x ≠ 0 → (sin x * sin x = sin x)) := sorry
  have h5 : sin x ≠ 0 → (sin x = 1 └ ∨ └ sin x = 0) := sorry
  have h6 : sin x = 1 := sorry
  have h7 : cos x = 0 := sorry
  have h8 : cot x = 0 := sorry
  have h9 : (cot x)^6 = 0 := sorry
  have h10 : (cot x)^2 = 0 := sorry
  rw [h9, h10],
  exact sub_self 0

end cot_sext_sub_cot_sq_l284_284441


namespace geometric_seq_value_l284_284853

variable (a : ℕ → ℝ)
variable (g : ∀ n m : ℕ, a n * a m = a ((n + m) / 2) ^ 2)

theorem geometric_seq_value (h1 : a 2 = 1 / 3) (h2 : a 8 = 27) : a 5 = 3 ∨ a 5 = -3 := by
  sorry

end geometric_seq_value_l284_284853


namespace general_term_a_n_T_k_geq_T_n_minimum_lambda_l284_284433

open Nat

-- Problem 1: General term \(a_n = 2^n\)
theorem general_term_a_n (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 2 * a n - 2)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a n = 2 ^ n :=
sorry

-- Problem 2: \(T_k \geq T_n\) for all \(n \in \mathbb{N}^*\)
theorem T_k_geq_T_n (k : ℕ) (T : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : ∀ n, b n = 1 / (2 ^ n) - (1 / n - 1 / (n + 1)))
  (hT : ∀ n, T n = ∑ i in range n, b i)
  (hn : ∀ n, n > 0) :
  4 = k ∧ (∀ n, T k ≥ T n) :=
sorry

-- Problem 3: Minimum λ such that \(R_n < λ\) for all \(n \in \mathbb{N}^*\)
theorem minimum_lambda (λ : ℝ) (c : ℕ → ℝ) (R : ℕ → ℝ)
  (ha : ∀ n, c n = 2 * (1 / (1 + 2^n) - 1 / (1 + 2^(n + 1))))
  (hR : ∀ n, R n = ∑ i in range n, c i)
  (hn : ∀ n, n > 0) :
  λ = 2 / 3 ∧ (∀ n, R n < λ) :=
sorry

end general_term_a_n_T_k_geq_T_n_minimum_lambda_l284_284433


namespace sum_9_to_12_l284_284927

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variables {S : ℕ → ℝ} -- Define the sum function of the sequence

-- Define the conditions given in the problem
def S_4 : ℝ := 8
def S_8 : ℝ := 20

-- The goal is to show that the sum of the 9th to 12th terms is 16
theorem sum_9_to_12 : (a 9) + (a 10) + (a 11) + (a 12) = 16 :=
by
  sorry

end sum_9_to_12_l284_284927


namespace find_c_l284_284219

noncomputable def parabola_coeffs(a b c : ℝ) (vertex point : ℝ × ℝ) : Prop :=
  vertex = (4, 3) ∧ point = (2, 5) ∧
  a * (5 - 3) ^ 2 + 4 = 2 ∧
  (a : ℝ) = -1 / 2

theorem find_c : ∃ c : ℝ, parabola_coeffs (-1/2) 3 c (4, 3) (2, 5) ∧ c = 0.5 :=
by
  use 0.5
  split
  sorry
  refl

end find_c_l284_284219


namespace new_average_l284_284667

theorem new_average (b : Fin 15 → ℝ) (h : (∑ i, b i) / 15 = 30) :
  ((∑ i, b i + 15) / 15) = 45 :=
by
  sorry

end new_average_l284_284667


namespace estimate_y_value_l284_284431

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value_l284_284431


namespace unique_seven_digit_number_l284_284341

def is_divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def meets_conditions (n : Nat) : Prop :=
  let digits := [0, 1, 2, 3, 4, 5, 6]
  let n_digits := n.digits 10
  ∃ (perm : List Nat), perm.permutations n_digits ∧
    is_divisible_by (perm.take 1).1 2 ∧
    is_divisible_by (perm.drop 6).1 2 ∧
    is_divisible_by (perm.take 2).1 3 ∧
    is_divisible_by (perm.drop 5).1 3 ∧
    is_divisible_by (perm.take 3).1 4 ∧
    is_divisible_by (perm.drop 4).1 4 ∧
    is_divisible_by (perm.take 4).1 5 ∧
    is_divisible_by (perm.drop 3).1 5 ∧
    is_divisible_by (perm.take 5).1 6 ∧
    is_divisible_by (perm.drop 2).1 6

theorem unique_seven_digit_number : ∃ (n : Nat), meets_conditions n ∧ n = 3216540 :=
by {
  use 3216540,
  sorry,
}

end unique_seven_digit_number_l284_284341


namespace max_popcorn_bags_l284_284946

theorem max_popcorn_bags (P : ℕ) (price : ℕ) (discount : ℚ) (budget : ℕ) :
  price = 4 →
  P = 50 →
  discount = 1 / 4 →
  budget = P * price →
  let full_price_bag := 1 * price
  let discounted_bag := 2 * (price - (discount * price).to_nat)
  let cost_per_set := full_price_bag + discounted_bag
  budget / cost_per_set * 3 = 60 :=
by
  sorry

end max_popcorn_bags_l284_284946


namespace square_side_length_of_rearranged_hexagons_l284_284643

theorem square_side_length_of_rearranged_hexagons (a b : ℕ) (h₁ : a = 9) (h₂ : b = 16) (z : ℕ) :
  (a * b = 144) ∧ ((2 * (a * b) / 2) = z^2) → z = 12 :=
by
  intros,
  sorry

end square_side_length_of_rearranged_hexagons_l284_284643


namespace product_of_divisors_of_72_l284_284358

theorem product_of_divisors_of_72 :
  let divisors := [1, 2, 4, 8, 3, 6, 12, 24, 9, 18, 36, 72]
  (list.prod divisors) = 5225476096 := by
  sorry

end product_of_divisors_of_72_l284_284358


namespace solution_set_xf_x_minus_5_l284_284089

variables {f : ℝ → ℝ}

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f (x) > f (y)

axiom odd_fn_defined_on_R : is_odd_function f
axiom f_monotonically_decreasing_on_pos : is_monotonically_decreasing f
axiom f_neg_seven_eq_zero : f (-7) = 0

-- Theorem to be proved
theorem solution_set_xf_x_minus_5 :
  { x : ℝ | x * f (x - 5) ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x < 0 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 12 } :=
begin
  sorry
end

end solution_set_xf_x_minus_5_l284_284089


namespace extreme_value_a_one_range_of_a_l284_284060

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 3

theorem extreme_value_a_one :
  ∀ x > 0, f x 1 ≤ f 1 1 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) → a ≥ Real.exp 2 :=
sorry

end extreme_value_a_one_range_of_a_l284_284060


namespace cost_of_lego_blocks_l284_284123

theorem cost_of_lego_blocks (L : ℝ) : 
  (3 * L) + (7 * 120) + (10 * 35) = 1940 → L = 250 :=
by
  intro h
  have h1 : (7 * 120) = 840 := by norm_num
  have h2 : (10 * 35) = 350 := by norm_num
  have ht : 840 + 350 = 1190 := by norm_num
  rw [h1, h2, ht] at h
  norm_num at h
  linarith

end cost_of_lego_blocks_l284_284123


namespace problem1_problem2_l284_284095

noncomputable theory

-- Define the triangle and its properties
def Triangle (A B C : ℝ) (a b c : ℝ) :=
  A + B + C = Real.pi ∧  -- Sum of angles in a triangle
  2 * B = A + C ∧        -- Angles form an arithmetic sequence
  a = 1 ∧                -- Given side lengths
  b = Real.sqrt 3

-- Problem 1: Prove that sin C = 1
theorem problem1 (A B C : ℝ) (a b c : ℝ) (h : Triangle A B C a b c) :
  Real.sin C = 1 := sorry

-- Define the geometric progression condition
def ArithmeticSequence (a b c : ℝ) :=
  2 * b = a + c

-- Problem 2: Prove that the triangle is equilateral
theorem problem2 (A B C : ℝ) (a b c : ℝ) (h_triangle : Triangle A B C a b c) (h_arith : ArithmeticSequence a b c) :
  ∃ A, A = B ∧ A = C ∧ a = b ∧ a = c := sorry

end problem1_problem2_l284_284095


namespace winnie_keeps_6_lollipops_l284_284626

/-- Winnie has 32 cherry lollipops, 105 wintergreen lollipops, 7 grape lollipops,
198 shrimp cocktail lollipops, and 12 friends. We want to prove that she
keeps 6 lollipops for herself after distributing them equally among her friends. -/
theorem winnie_keeps_6_lollipops :
  let total_lollipops := 32 + 105 + 7 + 198 in
  let friends := 12 in
  let lollipops_kept := total_lollipops % friends in
  lollipops_kept = 6 :=
by
  -- Definitions from the given conditions
  let total_lollipops := 32 + 105 + 7 + 198
  let friends := 12
  let lollipops_kept := total_lollipops % friends
  
  -- sorry placeholder for the proof
  sorry

end winnie_keeps_6_lollipops_l284_284626


namespace no_such_point_exists_l284_284290

theorem no_such_point_exists 
  (side_length : ℝ)
  (original_area : ℝ)
  (total_area_after_first_rotation : ℝ)
  (total_area_after_second_rotation : ℝ)
  (no_overlapping_exists : Prop) :
  side_length = 12 → 
  original_area = 144 → 
  total_area_after_first_rotation = 211 → 
  total_area_after_second_rotation = 287 →
  no_overlapping_exists := sorry

end no_such_point_exists_l284_284290


namespace discount_percentage_chicken_feed_l284_284656

noncomputable def total_cost : ℝ := 35
noncomputable def full_price_cost : ℝ := 49
noncomputable def chicken_feed_fraction : ℝ := 0.40

def spent_on_chicken_feed := chicken_feed_fraction * total_cost
def spent_on_goat_feed := (1 - chicken_feed_fraction) * total_cost
def full_price_chicken_feed := full_price_cost - spent_on_goat_feed

theorem discount_percentage_chicken_feed :
  (full_price_chicken_feed - spent_on_chicken_feed) / full_price_chicken_feed * 100 = 50 :=
by
  sorry

end discount_percentage_chicken_feed_l284_284656


namespace correct_propositions_l284_284132

variables {α β γ : Type} [plane α] [plane β] [plane γ] {l : Type} [line l]

-- Propositions
def prop_1 := ∀ {α β γ : Type} [plane α] [plane β] [plane γ], 
  (α ⊥ β ∧ β ⊥ γ → α ⊥ γ)

def prop_2 := ∀ {α β : Type} [plane α] [line β], 
  (∃ (p1 p2 : point), p1 ∈ β ∧ p2 ∈ β ∧ dist p1 α = dist p2 α → β ∥ α)

def prop_3 := ∀ {α β : Type} [plane α] [line β], 
  (β ⊥ α ∧ β ∥ γ → α ⊥ γ)

def prop_4 := ∀ {α β : Type} [plane α] [plane β] [line γ], 
  (α ∥ β ∧ ¬ (γ ⊆ β) ∧ γ ∥ α → γ ∥ β)

theorem correct_propositions : ¬ prop_1 ∧ ¬ prop_2 ∧ prop_3 ∧ prop_4 :=
by
  sorry

end correct_propositions_l284_284132


namespace arithmetic_sequence_sum_l284_284029

-- Given the sum of the first n odd numbers equals n^2
def sum_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

-- Sum of the arithmetic sequence 4k - 2 from k = 1 to 2020
def sum_arithmetic_sequence (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n - 1).succ, (4 * (k + 1) - 2)

theorem arithmetic_sequence_sum :
  sum_arithmetic_sequence 2020 = 8160800 :=
by
  sorry

end arithmetic_sequence_sum_l284_284029


namespace horner_method_operations_l284_284239

noncomputable def poly (x : ℝ) : ℝ :=
  3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

theorem horner_method_operations : 
  let f := poly 0.4 in 
  ∃ (num_adds num_mults : ℕ), num_adds = 6 ∧ num_mults = 6 ∧ (num_adds + num_mults = 12) :=
by {
  sorry
}

end horner_method_operations_l284_284239


namespace max_a_is_2_l284_284401

noncomputable def max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : ℝ :=
  2

theorem max_a_is_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  max_value_of_a a b c h1 h2 = 2 :=
sorry

end max_a_is_2_l284_284401


namespace power_ordering_l284_284321

theorem power_ordering (a b c : ℝ) : 
  (a = 2^30) → (b = 6^10) → (c = 3^20) → (a < b) ∧ (b < c) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have h1 : 6^10 = (3 * 2)^10 := by sorry
  have h2 : 3^20 = (3^10)^2 := by sorry
  have h3 : 2^30 = (2^10)^3 := by sorry
  sorry

end power_ordering_l284_284321


namespace sum_of_z_values_l284_284133

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_z_values (z : ℝ) : 
  (f (4 * z) = 13) → (∃ z1 z2 : ℝ, z1 = 1/8 ∧ z2 = -1/4 ∧ z1 + z2 = -1/8) :=
sorry

end sum_of_z_values_l284_284133


namespace triangle_is_equilateral_l284_284221

open Real

variables {Triangle : Type}
variables (a b c : Triangle) (r : ℝ) (heights : ℝ)

-- Given conditions
def inscribed_radius_one (T : Triangle) : Prop := r = 1
def integer_heights (T : Triangle) : Prop := ∀ h, S >= 3
def heights_integers (T : Triangle) : Prop := ∀ h, h ∈ ℕ

-- Question to prove: the triangle is equilateral
def is_equilateral_triangle (T : Triangle) : Prop :=
  a = b ∧ b = c ∧ c = a

theorem triangle_is_equilateral
  (T : Triangle)
  (h1 : inscribed_radius_one T)
  (h2 : integer_heights T)
  (h3 : heights_integers T) :
  is_equilateral_triangle T :=
sorry

end triangle_is_equilateral_l284_284221


namespace coordinates_in_new_basis_l284_284045

variables (a b c m : Vector)
variables (x y z : ℝ)
variables (h1 : a ⋅ a = 1 ∧ b ⋅ b = 1 ∧ c ⋅ c = 1 ∧ a ⋅ b = 0 ∧ a ⋅ c = 0 ∧ b ⋅ c = 0)
variables (h2 : m = 1 • a + 2 • b + 3 • c)

theorem coordinates_in_new_basis :
  (∃ (x y z : ℝ), m = x • (a + b) + y • (a - b) + z • c ∧ x = 3/2 ∧ y = -1/2 ∧ z = 3) :=
by {
  sorry
}

end coordinates_in_new_basis_l284_284045


namespace smallest_number_last_four_digits_l284_284507

theorem smallest_number_last_four_digits :
  ∃ (m : ℕ), (m % 4 = 0) ∧ (m % 6 = 0) ∧
  (∀ d ∈ (digits 10 m), d = 4 ∨ d = 6) ∧
  (∃ d ∈ (digits 10 m), d = 4) ∧
  (∃ d ∈ (digits 10 m), d = 6) ∧
  (m % 10000 = 4644) ∧
  (∀ n < m, n % 4 ≠ 0 ∨ n % 6 ≠ 0 ∨
  ∀ d ∈ (digits 10 n), d ≠ 4 ∧ d ≠ 6 ∨
  ¬(∃ d ∈ (digits 10 n), d = 4) ∨
  ¬(∃ d ∈ (digits 10 n), d = 6)) :=
begin
  sorry,
end

end smallest_number_last_four_digits_l284_284507


namespace jerry_total_shingles_l284_284868

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end jerry_total_shingles_l284_284868


namespace evaluate_expr_at_2_l284_284181

def expr (x : ℝ) : ℝ := (2 * x + 3) * (2 * x - 3) + (x - 2) ^ 2 - 3 * x * (x - 1)

theorem evaluate_expr_at_2 : expr 2 = 1 :=
by
  sorry

end evaluate_expr_at_2_l284_284181


namespace digits_difference_base3_base8_2048_l284_284438

theorem digits_difference_base3_base8_2048 : 
  (nat.log 2048 (nat.succ 2) + 1) - (nat.log 2048 (nat.succ 7) + 1) = 4 := 
by
  -- The number of digits in base-3 (log base 3 of 2048, plus 1)
  have h1 : nat.log 2048 (nat.succ 2) + 1 = 8 := 
    by sorry,
  -- The number of digits in base-8 (log base 8 of 2048, plus 1)
  have h2 : nat.log 2048 (nat.succ 7) + 1 = 4 :=
    by sorry,
  -- Proof of the difference being 4
  exact calc 
    (nat.log 2048 (nat.succ 2) + 1) - (nat.log 2048 (nat.succ 7) + 1) = 8 - 4 : by rw [h1, h2]
    ... = 4 : by rfl

end digits_difference_base3_base8_2048_l284_284438


namespace distance_interval_l284_284838

theorem distance_interval (d : ℝ) (h1 : ¬(d ≥ 8)) (h2 : ¬(d ≤ 7)) (h3 : ¬(d ≤ 6 → north)):
  7 < d ∧ d < 8 :=
by
  have h_d8 : d < 8 := by linarith
  have h_d7 : d > 7 := by linarith
  exact ⟨h_d7, h_d8⟩

end distance_interval_l284_284838


namespace additional_days_needed_l284_284285

def totalWork (people days: ℕ) : ℕ := people * days

def workDone (people days: ℕ) : ℕ := people * days

def remainingWork (tw wd: ℕ) : ℕ := tw - wd

def newWorkers (initial additional: ℕ) : ℕ := initial + additional

def additionalDays (rw nw: ℕ) : ℕ := rw / nw

theorem additional_days_needed :
  let tw := totalWork 24 12 in
  let wd := workDone 24 4 in
  let rw := remainingWork tw wd in
  let nw := newWorkers 24 8 in
  additionalDays rw nw = 6 :=
by
  sorry

end additional_days_needed_l284_284285


namespace smallest_n_gt_15_l284_284138

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def g (n : ℕ) : ℕ :=
  sum_of_digits (3 ^ n)

theorem smallest_n_gt_15 : ∃ n, n > 0 ∧ g n > 15 ∧ ∀ m, 0 < m ∧ m < n → g m ≤ 15 :=
by
  use 6
  have h1 : 3 ^ 6 = 729 := rfl
  have h2 : sum_of_digits 729 = 18 := rfl
  have h3 : 18 > 15 := by decide
  split
  · exact zero_lt_six -- Proving 6 > 0
  split
  · rwa [← h2, g, h1]
  · intros m hm
    cases lt_or_eq_of_le (nat.le_of_lt_succ hm.2) with hl he
    case inl =>
      sorry -- In full proof, we need to evaluate sum_of_digits for m = 1 to m = 5
    case inr =>
      exfalso
      exact lt_irrefl _ hm.1

end smallest_n_gt_15_l284_284138


namespace sequence_c1_d1_l284_284288

theorem sequence_c1_d1 
    (c : ℕ → ℝ) (d : ℕ → ℝ) 
    (h_sequence : ∀ n, (c (n + 1), d (n + 1)) = (sqrt 2 * c n - d n, sqrt 2 * d n + c n))
    (h_initial : (c 50, d 50) = (1, 3)) :
    c 1 + d 1 = -4 / (3 ^ 24.5 * sqrt 2) := 
sorry

end sequence_c1_d1_l284_284288


namespace sum_of_first_n_terms_l284_284136

noncomputable def f (x : ℝ) : ℝ := sorry
variable (f_properties : ∀ x y : ℝ, f (x - y) = f x / f y)
variable (f1_eq_two : f 1 = 2)
def a_n : ℕ+ → ℝ := λ n, f n

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n i.succ

theorem sum_of_first_n_terms (n : ℕ) :
  S_n n = 2^(n+1) - 2 := sorry

end sum_of_first_n_terms_l284_284136


namespace sum_after_replacement_l284_284250

theorem sum_after_replacement (n : ℕ) (x : Fin n → ℝ) 
    (h_sum_original : (∑ i, x i) = 40) 
    (h_sum_replacement : (∑ i, (1 - x i)) = 20) : 
    (∑ i, (1 + x i)) = 100 :=
by
  sorry

end sum_after_replacement_l284_284250


namespace Tn_less_Sn_plus_one_fourth_l284_284418

def a_n (n : ℕ) : ℕ := n

def b_n (n : ℕ) : ℚ := 1 / 3^n

def c_n (n : ℕ) : ℚ := a_n n * b_n n

def S_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, c_n (i + 1)

theorem Tn_less_Sn_plus_one_fourth (n : ℕ) : T_n n < S_n n + 1 / 4 := 
  sorry

end Tn_less_Sn_plus_one_fourth_l284_284418


namespace adult_ticket_price_l284_284273

/-- 
The community center sells 85 tickets and collects $275 in total.
35 of those tickets are adult tickets. Each child's ticket costs $2.
We want to find the price of an adult ticket.
-/
theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (adult_tickets_sold : ℕ) 
  (child_ticket_price : ℚ)
  (h1 : total_tickets = 85)
  (h2 : total_revenue = 275) 
  (h3 : adult_tickets_sold = 35) 
  (h4 : child_ticket_price = 2) 
  : ∃ A : ℚ, (35 * A + 50 * 2 = 275) ∧ (A = 5) :=
by
  sorry

end adult_ticket_price_l284_284273


namespace mean_weight_of_soccer_team_l284_284987

-- Define the weights as per the conditions
def weights : List ℕ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

-- Define the total weight
def total_weight : ℕ := 64 + 68 + 71 + 73 + 76 + 76 + 77 + 78 + 80 + 82 + 85 + 87 + 89 + 89

-- Define the number of players
def number_of_players : ℕ := 14

-- Calculate the mean weight
noncomputable def mean_weight : ℚ := total_weight / number_of_players

-- The proof problem statement
theorem mean_weight_of_soccer_team : mean_weight = 75.357 := by
  -- This is where the proof would go.
  sorry

end mean_weight_of_soccer_team_l284_284987


namespace solve_quadratic_l284_284183

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l284_284183


namespace box_height_correct_l284_284642

-- Define the parameters and conditions.
def large_sphere_radius : ℝ := 3
def small_sphere_radius : ℝ := 1.5
def box_width : ℝ := 6
def large_sphere_position : ℝ := box_width / 2

-- Define the height of the box
def box_height := large_sphere_position + large_sphere_radius + small_sphere_radius

-- The final theorem statement
theorem box_height_correct (h : ℝ) : h = box_height → h = 10.5 :=
by
  intros h_eq
  have : box_height = 10.5 := by
    unfold box_height large_sphere_position
    simp [large_sphere_radius, small_sphere_radius, box_width]
  rw [this] at h_eq
  exact h_eq

end box_height_correct_l284_284642


namespace measure_angle_RYS_l284_284851

/- Given conditions -/
variables {W X Y Z F R S : Point}
variables (side_length : ℝ) (y : ℝ)
variables (WF WZ YZ : Line)
variables (triangle_WXF_eq : Equilateral △ W X F) (square_WXYZ : Square W X Y Z)
variables (intersect_WR : ∃ R, R ∈ (WF ∩ WZ)) (RS_perpendicular_YZ : IsPerpendicular (Line.mk R S) YZ)
variables (RS_length : length (Line.mk R S) = y)
variables (side_length_5 : side_length = 5)
variables (angle_RYS : Angle R Y S)

/- Statement to prove -/
theorem measure_angle_RYS :
  Angle.measure (Angle.mk R Y S) = 105 :=
sorry

end measure_angle_RYS_l284_284851


namespace central_angle_of_minor_arc_l284_284979

theorem central_angle_of_minor_arc 
  (x y : ℝ) 
  (h_circle : x^2 + y^2 = 4) 
  (h_line : sqrt 3 * x + y - 2 * sqrt 3 = 0) : 
  central_angle_minor_arc x y h_circle h_line = π / 3 :=
sorry

end central_angle_of_minor_arc_l284_284979


namespace min_f_value_inequality_proof_l284_284058

-- Define the function f
def f (x : ℝ) : ℝ := abs(2 * x - 1) + x + 1/2

-- Define the problem statement
theorem min_f_value :
  ∃ m, (∀ x, f x ≥ m) ∧ f (1/2) = m := sorry

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_sum : a + b + c = 1) :
  2 * (a^3 + b^3 + c^3) ≥ a * b + b * c + c * a - 3 * a * b * c := sorry

end min_f_value_inequality_proof_l284_284058


namespace inequality_of_function_inequality_l284_284426

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + Real.sqrt (x^2 + 1))) + 2 * x + Real.sin x

theorem inequality_of_function_inequality (x1 x2 : ℝ) (h : f x1 + f x2 > 0) : x1 + x2 > 0 :=
sorry

end inequality_of_function_inequality_l284_284426


namespace paint_cost_approximately_400_2_l284_284577

noncomputable def cost_to_paint_floor (length : ℕ) (breadth_ratio: ℚ) (painting_rate : ℕ) : ℚ :=
  let breadth := length / breadth_ratio in
  let area := length * breadth in
  area * painting_rate

theorem paint_cost_approximately_400_2 :
  cost_to_paint_floor 20 (3 : ℚ) 3 ≈ 400.2 :=
sorry

end paint_cost_approximately_400_2_l284_284577


namespace hexagon_divisible_into_n_congruent_triangles_l284_284550

theorem hexagon_divisible_into_n_congruent_triangles (n : ℕ) (hn : n ≥ 6) :
  ∃ hexagon : Set (ℝ × ℝ), convex hexagon ∧ ∃ triangles : (ℝ × ℝ) → Set (ℝ × ℝ), 
    (∀ i : Fin n, ∃ triangle_i : Set (ℝ × ℝ), triangles i = triangle_i ∧ congruent triangle_i)
    ∧ (∀ i : Fin n, triangle_i ⊆ hexagon)
    ∧ (⋃ i : Fin n, triangle_i = hexagon)
    := 
by
  sorry

end hexagon_divisible_into_n_congruent_triangles_l284_284550


namespace part1_part2_l284_284501

noncomputable def S (a : ℕ → ℕ) (n : ℕ) := Nat.sum (List.range (n + 1)) a

def a (n : ℕ) : ℕ := if n = 0 then 1 else n

def b (n : ℕ) : ℝ := 
  if n ≤ 1 then 0 else 1 / (a (n - 1) * a n * a (n + 1))

def T (n : ℕ) : ℝ := Nat.sum (List.range (n + 1)) b

theorem part1 : ∀ n, a n = n := by
  sorry

theorem part2 : ∀ n, T n < 1 → b 1 ≤ 3 / 4 := by
  sorry

end part1_part2_l284_284501


namespace hyperbola_condition_l284_284506

theorem hyperbola_condition (m : ℝ) (hm : m ≠ 0) : 
(sufficient : (m > 0) → (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2/m - y^2/m = 1) ↔ (x^2/a^2 - y^2/b^2 = 1)))) ∧ 
¬ (necessary : (∀ (m > 0), ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, (x^2/m - y^2/m = 1) ↔ (x^2/a^2 - y^2/b^2 = 1))) := 
sorry

end hyperbola_condition_l284_284506


namespace jane_wins_game_l284_284866

noncomputable def jane_win_probability : ℚ :=
  1/3 / (1 - (2/3 * 1/3 * 2/3))

theorem jane_wins_game :
  jane_win_probability = 9/23 :=
by
  -- detailed proof steps would be filled in here
  sorry

end jane_wins_game_l284_284866


namespace centroid_of_quad_l284_284213

variables {A B C D E F G : Type}
variables (a c b d : ℝ)
variables (Af Ce Bg De : Type)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def intersection_point (A B C D : Point) : Point :=
  sorry

noncomputable def measure_AF (A E F : Point) (h : F.x = (A.x + E.x) / 2 ∧ F.y = (A.y + E.y) / 2) : Point :=
  sorry

noncomputable def measure_BG (B E G : Point) (h : G.x = (B.x + E.x) / 2 ∧ G.y = (B.y + E.y) / 2) : Point :=
  sorry

axiom centroid_of_triangle (P Q R : Point) : Point

theorem centroid_of_quad (A B C D : Point) :
  let E := intersection_point A C B D,
  let F := measure_AF A E (centroid_of_triangle A C E),
  let G := measure_BG B E (centroid_of_triangle B D E)
  in centroid_of_triangle A C G = centroid_of_triangle B D F ∧ 
     centroid_of_triangle A C G = centroid_of_triangle E F G ∧ 
     centroid_of_triangle A C G = Point.mk ((a + c) / 3) ((b + d) / 3) :=
begin
  sorry
end

end centroid_of_quad_l284_284213


namespace decreasing_function_passing_point_l284_284610

theorem decreasing_function_passing_point 
  (f : ℝ → ℝ) (k : ℝ)
  (h1 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂)
  (h2 : f 0 = 2)
  (h3 : ∀ x : ℝ, f x = k * x + 2) 
  (hk : k < 0) : 
  ∃ k < 0, ∀ x : ℝ, f x = k * x + 2 :=
begin
  use k,
  exact ⟨hk, h3⟩,
  done
end

end decreasing_function_passing_point_l284_284610


namespace great_grandfather_age_l284_284318

variable (MotherAge GrandmotherAge GreatGrandfatherAge : ℕ)

-- Given conditions
def DarcieAge : ℕ := 4
def condition1 : Prop := DarcieAge = 4
def condition2 : Prop := DarcieAge = 1/6 * MotherAge
def condition3 : Prop := MotherAge = 4/5 * GrandmotherAge
def condition4 : Prop := GrandmotherAge = 3/4 * GreatGrandfatherAge

-- The statement we want to prove
theorem great_grandfather_age : condition1 → condition2 → condition3 → condition4 → GreatGrandfatherAge = 40 :=
by
  sorry

end great_grandfather_age_l284_284318


namespace ball_drawing_probability_l284_284833

theorem ball_drawing_probability :
  ∃ (w : ℕ), (∃ (n : ℕ), n = 10 ∧ (1 - ↑(nat.choose (10 - w) 2) / ↑(nat.choose 10 2) = (7 : ℚ) / 9) ∧ w ∈ ℕ ∧ w = 5)
  ∧ (↑(nat.choose w 0) * ↑(nat.choose (10 - w) 3) + ↑(nat.choose w 1) * ↑(nat.choose (10 - w) 2)) 
      / ↑(nat.choose 10 3) = (1 : ℚ) / 2 :=
by
  sorry

end ball_drawing_probability_l284_284833


namespace solve_arccos_cos_eq_l284_284552

theorem solve_arccos_cos_eq {x : ℝ} (h1 : -real.pi ≤ x ∧ x ≤ real.pi) (h2 : real.arccos (real.cos x) = (3 * x) / 2) : x = 0 :=
sorry

end solve_arccos_cos_eq_l284_284552


namespace factor_expression_l284_284313

theorem factor_expression (x : ℝ) :
  (12 * x ^ 5 + 33 * x ^ 3 + 10) - (3 * x ^ 5 - 4 * x ^ 3 - 1) = x ^ 3 * (9 * x ^ 2 + 37) + 11 :=
by {
  -- Provide the skeleton for the proof using simplification
  sorry
}

end factor_expression_l284_284313


namespace midpoint_trajectory_l284_284722

noncomputable theory

def satisfiescondition (X Y : ℝ) := X^2 + Y^2 = 1

theorem midpoint_trajectory :
  ∀ (x y : ℝ),
    (∃ (X Y : ℝ), X = 2 * x - 3 ∧ Y = 2 * y ∧ satisfiescondition X Y) →
    x^2 + y^2 - 3 * x + 2 = 0 :=
by
  assume x y : ℝ
  intro h
  cases h with X h
  cases h with Y h
  cases h with hX h
  cases h with hY hCond
  -- placeholder for actual proof
  sorry

end midpoint_trajectory_l284_284722


namespace permutations_with_property_P_greater_l284_284898

def has_property_P (n : ℕ) (p : (Fin (2 * n) → Fin (2 * n))) : Prop :=
∃ i : Fin (2 * n - 1), |(p i).val - (p ⟨i.val + 1, Nat.lt_of_succ_lt_succ i.property⟩).val| = n

theorem permutations_with_property_P_greater {n : ℕ} (hn : 0 < n) :
  let perms := equiv.perm (Fin (2 * n))
  ∃ (p₁ p₂ : perms), has_property_P n p₁ ∧ ¬ has_property_P n p₂ ∧
    (∑ p in perms, (if has_property_P n p then 1 else 0)) >
    (∑ p in perms, (if ¬ has_property_P n p then 1 else 0)) := 
sorry

end permutations_with_property_P_greater_l284_284898


namespace S_is_infinite_l284_284497

variable (S : Set Point)

-- Define a midpoint property for S
def is_midpoint (S : Set Point) : Prop :=
  ∀ (P : Point), P ∈ S → ∃ (A B : Point), A ∈ S ∧ B ∈ S ∧ P = (A + B) / 2

-- State the theorem to prove S is infinite
theorem S_is_infinite (h : is_midpoint S) : ¬Finite S := by
  sorry

end S_is_infinite_l284_284497


namespace hose_Z_fill_time_l284_284193

theorem hose_Z_fill_time (P X Y Z : ℝ) (h1 : X + Y = P / 3) (h2 : Y = P / 9) (h3 : X + Z = P / 4) (h4 : X + Y + Z = P / 2.5) : Z = P / 15 :=
sorry

end hose_Z_fill_time_l284_284193


namespace average_score_l284_284203

variable (K M : ℕ) (E : ℕ)

theorem average_score (h1 : (K + M) / 2 = 86) (h2 : E = 98) :
  (K + M + E) / 3 = 90 :=
by
  sorry

end average_score_l284_284203


namespace tangent_line_at_3_l284_284430

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_3
  (h_tangent : ∀ x, f x = 2 ∧ (deriv f x) = -1) :
  f 3 + deriv f 3 = 1 := 
by 
  specialize h_tangent 3
  sorry

end tangent_line_at_3_l284_284430


namespace tommy_gum_given_l284_284932

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end tommy_gum_given_l284_284932


namespace max_white_black_ratio_l284_284637

-- defining the problem
def square_board_checkerboard_pattern (n : ℕ) : Prop :=
  let cells := n * n in
  cells = 64 ∧ -- The board is divided into 64 cells
  ∀ (white_cell black_cell : ℕ),   -- Area Ratio
    white_cell / black_cell ≤ 2

theorem max_white_black_ratio (wb_ratio : ℚ) (h1 : square_board_checkerboard_pattern 8) 
  : wb_ratio = 5 / 4 :=
  sorry  -- Proof is omitted for now.

end max_white_black_ratio_l284_284637


namespace smallest_k_l284_284619

theorem smallest_k (k : ℕ) : 
  (∀ x, x ∈ [13, 7, 3, 5] → k % x = 1) ∧ k > 1 → k = 1366 :=
by
  sorry

end smallest_k_l284_284619


namespace product_of_divisors_of_72_l284_284366

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284366


namespace find_m_value_l284_284849

variable (m : ℝ)
variable (t : ℝ)
variable (θ : ℝ)
variable (ρ : ℝ)
variable (x : ℝ)
variable (y : ℝ)

def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_l (t : ℝ) (m : ℝ) : (ℝ × ℝ) := (m + (√3 / 2) * t, (1 / 2) * t)
def polar_eq_of_curve_C (θ : ℝ) : ℝ := 2 * cos θ

theorem find_m_value (h₁ : curve_C x y) (h₂ : line_l t m = (x, y)) (h₃ : ∀ t₁ t₂, line_l t₁ m = (x₁, y₁) ∧ line_l t₂ m = (x₂, y₂) ∧ |m^2 - 2 * m| = 1) :
  (m = 1 + sqrt 2 ∨ m = 1 ∨ m = 1 - sqrt 2) :=
sorry

end find_m_value_l284_284849


namespace equal_areas_part_A_equal_areas_part_B_l284_284141

-- Define the quadrilaterals and midpoints
variables {A B C D X Y O P Q R S : Point}

-- Assume basic properties and conditions from problem
axiom midpoint_AC (A C : Point) : midpoint A C X
axiom midpoint_BD (B D : Point) : midpoint B D Y
axiom lines_parallel_XY (X Y O : Point) : (line_through X parallel_to line_through B D) ∧ (line_through Y parallel_to line_through A C) ∧ intersection_at (line_through X) (line_through Y) O
axiom midpoint_AB (A B : Point) : midpoint A B P
axiom midpoint_BC (B C : Point) : midpoint B C Q
axiom midpoint_CD (C D : Point) : midpoint C D R
axiom midpoint_DA (D A : Point) : midpoint D A S

-- Prove equal areas for part (A)
theorem equal_areas_part_A : area A P O S = area A P X S :=
by sorry

-- Prove equal areas for part (B)
theorem equal_areas_part_B : area A P O S = area B Q O P ∧ area B Q O P = area C R O Q ∧ area C R O Q = area D S O R :=
by sorry

end equal_areas_part_A_equal_areas_part_B_l284_284141


namespace number_of_ways_to_climb_steps_l284_284988

theorem number_of_ways_to_climb_steps : 
  ∃ (x y z : ℕ), 
    x + 2 * y + 3 * z = 10 ∧ 
    x + y + z = 7 ∧ 
    (nat.choose 7 y * nat.factorial y / (nat.factorial y * nat.factorial (7 - y)) + 
    nat.choose 6 1 * nat.factorial (6 - 1) = 77) :=
sorry

end number_of_ways_to_climb_steps_l284_284988


namespace part_a_part_b_part_c_l284_284890

-- Definitions
variables {α : Type*} (A : ℕ → Set α) (I : Set α → ℝ)

noncomputable def liminf (A : ℕ → Set α) : Set α := {x | ∀ᶠ n in Filter.atTop, x ∈ A n}
noncomputable def limsup (A : ℕ → Set α) : Set α := {x | ∃ᶠ n in Filter.atTop, x ∈ A n}

-- Proof problem statement
theorem part_a (hI : Monotone I) (hI_subadd : ∀ s : Set (Set α), I(⋃ i ∈ s, i) ≤ ∑' i, I i) :
  I (liminf A) = liminf (λ n, I (A n)) ∧ I (limsup A) = limsup (λ n, I (A n)) :=
sorry

theorem part_b (hI : Monotone I) (hI_subadd : ∀ s : Set (Set α), I(⋃ i ∈ s, i) ≤ ∑' i, I i) :
  limsup (λ n, I (A n)) - liminf (λ n, I (A n)) = I (limsup A \ liminf A) :=
sorry

theorem part_c (hI_subadd : ∀ s : Set (Set α), I(⋃ i ∈ s, i) ≤ ∑' i, I i) :
  I (⋃ n, A n) ≤ ∑' n, I (A n) :=
sorry

end part_a_part_b_part_c_l284_284890


namespace sum_of_x_coords_f_eq_3_l284_284315

section
-- Define the piecewise linear function, splits into five segments
def f1 (x : ℝ) : ℝ := 2 * x + 6
def f2 (x : ℝ) : ℝ := -2 * x + 6
def f3 (x : ℝ) : ℝ := 2 * x + 2
def f4 (x : ℝ) : ℝ := -x + 2
def f5 (x : ℝ) : ℝ := 2 * x - 4

-- The sum of x-coordinates where f(x) = 3
noncomputable def x_coords_3_sum : ℝ := -1.5 + 0.5 + 3.5

-- Goal statement
theorem sum_of_x_coords_f_eq_3 : -1.5 + 0.5 + 3.5 = 2.5 := by
  sorry
end

end sum_of_x_coords_f_eq_3_l284_284315


namespace smallest_value_mod_complex_combination_l284_284894

theorem smallest_value_mod_complex_combination {a b c : ℤ} (h_abc_nonzero_even : a ≠ 0 ∧ ¬ odd a ∧ b ≠ 0 ∧ ¬ odd b ∧ c ≠ 0 ∧ ¬ odd c)
    (h_distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a) {ω : ℂ} (h_omega_cube_root_unity : ω^3 = 1 ∧ ω ≠ 1) :
    ∃ (x : ℝ), x = |a + b * ω + c * ω^2| ∧ x = 2 * Real.sqrt 3 :=
  sorry

end smallest_value_mod_complex_combination_l284_284894


namespace breadth_decrease_percentage_l284_284567

theorem breadth_decrease_percentage
  (L B : ℝ)
  (hLpos : L > 0)
  (hBpos : B > 0)
  (harea_change : (1.15 * L) * (B - p/100 * B) = 1.035 * (L * B)) :
  p = 10 := 
sorry

end breadth_decrease_percentage_l284_284567


namespace ellipse_equation_and_min_perpendicular_chords_l284_284785

theorem ellipse_equation_and_min_perpendicular_chords (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (h_minor_axis : 2 * b = 2) 
    (h_vertices : ∀ x y : ℝ, x^2 + (y - real.sqrt 2 / 2)^2 = 1 / 2 →
      (y^2 = 2 * (1 - x^2)) ) :
  (a = real.sqrt 2 ∧ b = 1) → 
  (∀ (F : ℝ × ℝ) (h_focus: F = (0, real.sqrt 2)) (AB CD : set (ℝ × ℝ))
    (h_AB : AB = {p : ℝ × ℝ | p.2 = p.1 + 1})
    (h_CD : CD = {p : ℝ × ℝ | p.2 = - p.1 + 1}) 
    (h_lengths: (|AB| + |CD|)))
  (∃ min_length : ℝ, min_length = 8 * real.sqrt 3 / 3) := sorry

end ellipse_equation_and_min_perpendicular_chords_l284_284785


namespace scalene_triangle_angle_range_l284_284836

theorem scalene_triangle_angle_range
  (A B C : Type)
  [T: Triangle A B C]
  (a b c : ℝ)
  (h1 : scalene_triangle A B C)
  (h2 : longest_side a b c)
  (h3 : a^2 < b^2 + c^2) :
  60 < angle A < 90 :=
by
  sorry

end scalene_triangle_angle_range_l284_284836


namespace better_approximation_pi_l284_284677

-- Define the ellipse perimeter function
def ellipsePerimeter (a b : ℝ) : ℝ :=
  4 * ∫ t in 0..π/2, sqrt (a^2 * sin t ^ 2 + b^2 * cos t ^ 2)

-- Define ε and the condition where b = a * sqrt(1 - ε)
variables (a : ℝ) (ε : ℝ) (hε : 0 < ε ∧ ε < 1)
def b := a * sqrt(1 - ε)

-- Formulate the better approximation question
theorem better_approximation_pi (hε : 0 < ε ∧ ε < 1) : 
  let ab := a * sqrt(a * (sqrt(1 - ε)))
  in abs (ellipsePerimeter a (b a ε) - π * (a + b a ε)) < 
     abs (ellipsePerimeter a (b a ε) - 2 * π * sqrt ab) := 
sorry

end better_approximation_pi_l284_284677


namespace general_formulas_and_sums_l284_284011

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 + a 3 = 4 ∧ a 4 = 3

def geometric_sequence (b : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  b 1 = (1 / 3) * a 1 ∧ b 3 * a 14 = 1

-- Define the problem statement
theorem general_formulas_and_sums
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b a) :
  (∀ n, a n = 0.5 * n + 1) ∧
  (∀ n, b n = (0.5) ^ n) ∧
  (∀ n, (∑ k in finset.range n, a (k + 1)) = (1 / 4) * n ^ 2 + (5 / 4) * n) ∧
  (∀ n, (∑ k in finset.range n, b (k + 1)) = 1 - (0.5) ^ n) :=
sorry

end general_formulas_and_sums_l284_284011


namespace triangle_ratio_l284_284092

theorem triangle_ratio (A B C D : Type) [triangle A B C] (BD : altitude B D A) (A_eq : A = π / 4) (cos_B_eq : cos B = - (sqrt 5 / 5)) : 
  BD / AC = 1 / 4 := 
sorry

end triangle_ratio_l284_284092


namespace number_of_terminating_decimals_l284_284380

theorem number_of_terminating_decimals : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 299 → (∃ k : ℕ, n = 9 * k) → 
  ∃ count : ℕ, count = 33 := 
sorry

end number_of_terminating_decimals_l284_284380


namespace average_and_variance_change_l284_284831

theorem average_and_variance_change :
  ∀ (students : ℕ) (absent_students total_students : ℕ)
    (avg_score initial_variance : ℝ) (makeup_scores : list ℝ),
    total_students = students + absent_students →
    students = 50 →
    absent_students = 3 →
    avg_score = 90 →
    initial_variance = 40 →
    makeup_scores = [88, 90, 92] →
    let new_avg := (students * avg_score + 
                    ∑ (makeup_scores : list ℝ)) / 
                    total_students in
    new_avg = avg_score ∧
    let squared_diff_sum := initial_variance * students + 
                            ∑ ((s - avg_score) ^ 2 | s : makeup_scores) in
    let new_variance := squared_diff_sum / total_students in
    new_variance < initial_variance :=
begin
  intros students absent_students total_students avg_score initial_variance makeup_scores,
  assume h1 h2 h3 h4 h5 h6,
  let new_avg := (50 * 90 + (88 + 90 + 92)) / 53,
  have h_new_avg : new_avg = 90, sorry,
  have hh : ∑ ((s - 90) ^ 2 | s : makeup_scores) = (88 - 90) ^ 2 + (90 - 90) ^ 2 + (92 - 90) ^ 2, sorry,
  have squared_diff_sum := 40 * 50 + (88 - 90) ^ 2 + (90 - 90) ^ 2 + (92 - 90) ^ 2,
  have new_variance := squared_diff_sum / 53,
  have h_new_variance : new_variance < 40, sorry,
  exact ⟨h_new_avg, h_new_variance⟩
end

end average_and_variance_change_l284_284831


namespace ratio_of_areas_l284_284855

-- Defining the triangle and related points
def right_triangle (A B C : Type) (C_right : RightAngle C) := 
  ∃ (CH : LineSegment C A B) (D : Midpoint CH), 
  ∃ (AD : SymmetricLine AB with respect to AD)
     (BD : SymmetricLine AB with respect to BD) (F : IntersectionPoint AD BD), 
  area_ratio ABF ABC = 4/3

theorem ratio_of_areas 
  (A B C : Point) (right_C : is_right_angle ∠ACB)
  (CH : Altitude C AB) (D : Midpoint CH)
  (AD : SymmetricLine AB D) (BD : SymmetricLine AB D)
  (F : intersection_point AD BD) :
  area_ratio (triangle A B F) (triangle A B C) = 4 / 3 :=
sorry

end ratio_of_areas_l284_284855


namespace total_days_2008_to_2011_l284_284307

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

theorem total_days_2008_to_2011 : 
  ∑ y in {2008, 2009, 2010, 2011}, if is_leap_year y then 366 else 365 = 1462 :=
by {
  sorry
}

end total_days_2008_to_2011_l284_284307


namespace math_problem_l284_284502

noncomputable def proof_problem (k : ℝ) (a b k1 k2 : ℝ) : Prop :=
  (a*b) = 7/k ∧ (a + b) = (k-1)/k ∧ (k1^2 - 18*k1 + 1) = 0 ∧ (k2^2 - 18*k2 + 1) = 0 ∧ 
  (a/b + b/a = 3/7) → (k1/k2 + k2/k1 = 322)

theorem math_problem (k a b k1 k2 : ℝ) : proof_problem k a b k1 k2 :=
by
  sorry

end math_problem_l284_284502


namespace rhombus_perimeter_l284_284565

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  let side_length := Real.sqrt (6^2 + 8^2) in
  let perimeter := 4 * side_length in
  perimeter = 40 := 
by
  have half_d1 := d1 / 2
  have half_d2 := d2 / 2
  have s := Real.sqrt (half_d1^2 + half_d2^2)
  calc
    perimeter = 4 * s       : by rwa [s]
    ...      = 4 * 10       : by rw [Real.sqrt, Real.sqrt_eq_rpow]
    ...      = 40           : by norm_num

end rhombus_perimeter_l284_284565


namespace time_to_odd_floor_l284_284269

-- Define the number of even-numbered floors
def evenFloors : Nat := 5

-- Define the number of odd-numbered floors
def oddFloors : Nat := 5

-- Define the time to climb one even-numbered floor
def timeEvenFloor : Nat := 15

-- Define the total time to reach the 10th floor
def totalTime : Nat := 120

-- Define the desired time per odd-numbered floor
def timeOddFloor : Nat := 9

-- Formalize the proof statement
theorem time_to_odd_floor : 
  (oddFloors * timeOddFloor = totalTime - (evenFloors * timeEvenFloor)) :=
by
  sorry

end time_to_odd_floor_l284_284269


namespace shaded_region_area_l284_284984

-- Define the problem conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def area_of_shaded_region : ℝ := 50

-- State the theorem to prove the area of the shaded region
theorem shaded_region_area (n : ℕ) (d : ℝ) (area : ℝ) (h1 : n = num_squares) (h2 : d = diagonal_length) : 
  area = area_of_shaded_region :=
sorry

end shaded_region_area_l284_284984


namespace area_of_sector_l284_284413

-- Given conditions
def central_angle : ℝ := 2
def perimeter : ℝ := 8

-- Define variables and expressions
variable (r l : ℝ)

-- Equations based on the conditions
def eq1 := l + 2 * r = perimeter
def eq2 := l = central_angle * r

-- Assertion of the correct answer
theorem area_of_sector : ∃ r l : ℝ, eq1 r l ∧ eq2 r l ∧ (1 / 2 * l * r = 4) := by
  sorry

end area_of_sector_l284_284413


namespace mul_same_base_exp_ten_pow_1000_sq_l284_284640

theorem mul_same_base_exp (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

-- Given specific constants for this problem
theorem ten_pow_1000_sq : (10:ℝ)^(1000) * (10)^(1000) = (10)^(2000) := by
  exact mul_same_base_exp 10 1000 1000

end mul_same_base_exp_ten_pow_1000_sq_l284_284640


namespace right_triangle_comparison_l284_284496

theorem right_triangle_comparison
  (A B C P : Type)
  (hABC : ∠ACB = 90)
  (hAC : AC = 3)
  (hBC : BC = 4)
  (hAB : AB = 5)
  (hAP_ratio : AP / PB = 1 / 4)
  (hP_on_line : P ∈ segment AB) :
  AP ^ 2 + PB ^ 2 > 2 * (AC * BC / AB) ^ 2 :=
by
  sorry

end right_triangle_comparison_l284_284496


namespace average_of_possible_x_values_l284_284818

theorem average_of_possible_x_values (x : ℝ) (h : sqrt (3 * x ^ 2 + 2) = sqrt 50) : 
  (4 + (-4)) / 2 = 0 := 
by
suffices : x = 4 ∨ x = -4, from sorry,
sorry

end average_of_possible_x_values_l284_284818


namespace conjugate_of_z_find_a_b_l284_284047

noncomputable def z : ℂ := ((1 - I) ^ 2 + 3 * (1 + I)) / (2 - I)

theorem conjugate_of_z :
  conj z = 1 - I := sorry

theorem find_a_b (a b : ℝ) (h : a * z + b = 1 - I) :
  a = -1 ∧ b = 2 := sorry

end conjugate_of_z_find_a_b_l284_284047


namespace january_roses_l284_284570

theorem january_roses (r_october r_november r_december r_february r_january : ℕ)
  (h_october_november : r_november = r_october + 12)
  (h_november_december : r_december = r_november + 12)
  (h_december_january : r_january = r_december + 12)
  (h_january_february : r_february = r_january + 12) :
  r_january = 144 :=
by {
  -- The proof would go here.
  sorry
}

end january_roses_l284_284570


namespace problem1_problem2_l284_284750

variables {a b c : ℝ}

-- (1) Prove that a + b + c = 4 given the conditions
theorem problem1 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_min : ∀ x, abs (x + a) + abs (x - b) + c ≥ 4) : a + b + c = 4 := 
sorry

-- (2) Prove that the minimum value of (1/4)a^2 + (1/9)b^2 + c^2 is 8/7 given the conditions and that a + b + c = 4
theorem problem2 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8 / 7 := 
sorry

end problem1_problem2_l284_284750


namespace min_value_problem_l284_284909

theorem min_value_problem (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2 = 4 * (Real.root 4 5 - 1)^2 := 
sorry

end min_value_problem_l284_284909


namespace area_enclosed_by_graph_l284_284573

theorem area_enclosed_by_graph : 
  (∃ (region : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ region ⟷ y^2 + 2 * x * y + 24 * |x| = 400 ∧ 
      enclosing(region) = 768)) :=
sorry

end area_enclosed_by_graph_l284_284573


namespace product_of_divisors_of_72_l284_284367

-- Definition of 72 with its prime factors
def n : ℕ := 72
def n_factors : Prop := ∃ a b : ℕ, n = 2^3 * 3^2

-- Definition of what we are proving
theorem product_of_divisors_of_72 (h : n_factors) : ∏ d in (finset.divisors n), d = 2^18 * 3^12 :=
by sorry

end product_of_divisors_of_72_l284_284367


namespace sin_double_alpha_plus_pi_third_l284_284410

-- Define the acute angle α.
variable {α : ℝ}

-- Assume the conditions of the problem.
axiom α_acute : 0 < α ∧ α < π / 2
axiom cos_alpha_plus_pi_four : cos (α + π / 4) = sqrt 5 / 5

-- Prove that sin(2α + π/3) = (4 * sqrt 3 + 3) / 10.
theorem sin_double_alpha_plus_pi_third :
  sin (2 * α + π / 3) = (4 * sqrt 3 + 3) / 10 :=
by 
  sorry

end sin_double_alpha_plus_pi_third_l284_284410


namespace find_C_value_l284_284032

open Real

theorem find_C_value (C : ℝ) : 
  let x0 := -1
  let y0 := 2
  let A := 4
  let B := -3
  let d := 1
  abs(A * x0 + B * y0 + C) / sqrt(A^2 + B^2) = d ↔ (C = 5 ∨ C = 15) := by
  let x0 := -1
  let y0 := 2
  let A := 4
  let B := -3
  let d := 1
  sorry

end find_C_value_l284_284032


namespace proposition_1_proposition_2_proposition_3_proposition_4_l284_284428

def f (x b c : ℝ) := x * (abs x) + b * x + c

theorem proposition_1 (b : ℝ) : (∀ x : ℝ, f x b 0 = - (f (-x) b 0)) :=
by
  intros x
  unfold f
  sorry

theorem proposition_2 (b : ℝ) (h : b = 0) (c : ℝ) (hc : c > 0) : ∃! x : ℝ, f x 0 c = 0 :=
by
  intros x
  unfold f
  sorry

theorem proposition_3 (b c : ℝ) : ∀ x y, (x, y) symmetric_about (0, c) → y = f x b c :=
by
  intros x y h
  unfold symmetric_about
  unfold f
  sorry

theorem proposition_4 (b c : ℝ) : ¬ (f x b c).has_at_most_two_real_roots :=
by
  intro h
  unfold has_at_most_two_real_roots f
  existsi [0, 2, -2]
  sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l284_284428


namespace multiple_of_min_weight_l284_284943

def standard_weight : ℝ := 100
def max_weight : ℝ := 210
def min_weight : ℝ := standard_weight + 5
def M := max_weight / min_weight

theorem multiple_of_min_weight :
  M = 2 := by
  sorry

end multiple_of_min_weight_l284_284943


namespace inequality_solution_range_l284_284088

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, 2 * x - 6 + m < 0 ∧ 4 * x - m > 0) → m < 4 :=
by
  intro h
  sorry

end inequality_solution_range_l284_284088


namespace women_in_room_l284_284864

def ratio_initial (M W : ℕ) := 7 * W = 8 * M
def men_after (M : ℕ) := M + 4 = 16
def current_women (W : ℕ) := 3 * (W - 5) = 27

theorem women_in_room (M W : ℕ) (h1 : ratio_initial M W) (h2 : men_after M) : current_women W :=
by {
  sorry,
}

end women_in_room_l284_284864


namespace rays_through_one_point_infinite_line_segment_through_two_points_unique_l284_284605

theorem rays_through_one_point_infinite (P : Point) : 
    ∃ (rays : Set Ray), ∀ (r : Ray), r ∈ rays ↔ ∃ (dir : Direction), r = ⟨P, dir⟩ :=
sorry

theorem line_segment_through_two_points_unique (P Q : Point) (h : P ≠ Q) : 
    ∃! (seg : LineSegment), seg = ⟦P, Q⟧ :=
sorry

end rays_through_one_point_infinite_line_segment_through_two_points_unique_l284_284605


namespace quadrilateral_fourth_side_length_proof_l284_284663

noncomputable def length_of_fourth_side (r l1 l2 l3 : ℝ) (l4 : ℝ) : Prop :=
  let s1 := l1
  let s2 := l2
  let s3 := l3
  let s4 := l4
  s1 = 150 ∧ s2 = 150 ∧ s3 = 150 * Real.sqrt 3 ∧ r = 150 * Real.sqrt 2 ∧ s4 = 150 * Real.sqrt 7

theorem quadrilateral_fourth_side_length_proof : length_of_fourth_side (150 * Real.sqrt 2) 150 150 (150 * Real.sqrt 3) (150 * Real.sqrt 7) :=
begin
  sorry
end

end quadrilateral_fourth_side_length_proof_l284_284663


namespace min_value_of_expr_l284_284903

noncomputable def real.min_value_expr (x y z : ℝ) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expr :
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    real.min_value_expr x y z = 4 * (real.sqrt (real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expr_l284_284903


namespace fifth_girl_siblings_l284_284675

theorem fifth_girl_siblings (mean : ℝ) (siblings : list ℝ) (h_mean : mean = 5.7) (h_siblings : siblings = [1, 6, 10, 4, 3, 11, 3, 10]) : 
  nth_le siblings 4 sorry = 3 :=
by sorry

end fifth_girl_siblings_l284_284675


namespace A_can_finish_work_in_21_days_l284_284647

def work_rate (days : ℕ) : ℝ := 1 / days

def finished_work_rate (rate : ℝ) (days : ℕ) : ℝ := rate * days

theorem A_can_finish_work_in_21_days :
  ∀ (x : ℕ), 
  let B_rate := work_rate 15 in
  let B_work_completed := finished_work_rate B_rate 10 in
  let remaining_work := 1 - B_work_completed in
  let A_rate := work_rate x in
  let A_work_completed := finished_work_rate A_rate 7 in
  (A_work_completed = remaining_work) → x = 21 :=
by
  intros x B_rate B_work_completed remaining_work A_rate A_work_completed h,
  change (finished_work_rate (work_rate x) 7) = 1 / 3,
  sorry

end A_can_finish_work_in_21_days_l284_284647


namespace greatest_integer_pi_minus_five_l284_284613

-- Definitions and conditions
def greatest_integer_function (x : ℝ) : ℤ :=
  ⌊x⌋

-- The theorem to prove
theorem greatest_integer_pi_minus_five : greatest_integer_function (Real.pi - 5) = -2 := by
  sorry

end greatest_integer_pi_minus_five_l284_284613


namespace minimum_sides_divided_into_parallelograms_l284_284245

def NonConvexPolygon (p : ℕ) : Prop := p ≥ 3 ∧ ¬ ConvexPolygon p
def ConvexPolygon (p : ℕ) : Prop := ∀ i j, i ≠ j → ∃ij. parallel sides of p
def CanBeDividedIntoParallelograms (p : ℕ) : Prop := ∃ (n : ℕ), n ≤ p ∧ (∀ k < n, is_parallelogram k)

theorem minimum_sides_divided_into_parallelograms :
  ∀ p, NonConvexPolygon p ∧ CanBeDividedIntoParallelograms p → p = 7 :=
by
  sorry

end minimum_sides_divided_into_parallelograms_l284_284245


namespace min_value_of_expression_l284_284915

noncomputable def min_of_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expression :
  ∃ x y z : ℝ, (2 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧ (z ≤ 5) ∧ min_of_expression x y z = 4 * (Real.sqrt (Real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expression_l284_284915


namespace partiallyFilledBoxes_l284_284117

/-- Define the number of cards Joe collected -/
def numPokemonCards : Nat := 65
def numMagicCards : Nat := 55
def numYuGiOhCards : Nat := 40

/-- Define the number of cards each full box can hold -/
def pokemonBoxCapacity : Nat := 8
def magicBoxCapacity : Nat := 10
def yuGiOhBoxCapacity : Nat := 12

/-- Define the partially filled boxes for each type -/
def pokemonPartialBox : Nat := numPokemonCards % pokemonBoxCapacity
def magicPartialBox : Nat := numMagicCards % magicBoxCapacity
def yuGiOhPartialBox : Nat := numYuGiOhCards % yuGiOhBoxCapacity

/-- Theorem to prove number of cards in each partially filled box -/
theorem partiallyFilledBoxes :
  pokemonPartialBox = 1 ∧
  magicPartialBox = 5 ∧
  yuGiOhPartialBox = 4 :=
by
  -- proof goes here
  sorry

end partiallyFilledBoxes_l284_284117


namespace find_y_when_x_is_8_l284_284189

theorem find_y_when_x_is_8 : 
  ∃ k, (70 * 5 = k ∧ 8 * 25 = k) := 
by
  -- The proof will be filled in here
  sorry

end find_y_when_x_is_8_l284_284189


namespace fraction_of_area_covered_l284_284531

open Real

-- Define the grid size and the position of the vertex of the shaded square
def grid_size : ℕ := 6
def shaded_square_vertex : (ℕ × ℕ) := (3, 3)

-- Define the properties of the shaded square in terms of the problem conditions
def side_length_of_shaded_square : ℝ := sqrt 2 -- Since the square is rotated and its sides are the diagonals of 1x1 grid cells
def area_of_shaded_square : ℝ := (sqrt 2) ^ 2 -- Area calculation
def area_of_larger_square : ℝ := (6:ℝ) ^ 2  -- Area of the 6 by 6 grid

-- Define the proof problem
theorem fraction_of_area_covered : area_of_shaded_square / area_of_larger_square = (1 : ℝ) / 18 :=
by
  have h1 : area_of_shaded_square = 2, by sorry
  have h2 : area_of_larger_square = 36, by sorry
  have h3 : 2 / 36 = 1 / 18, by sorry
  rw [h1, h2, h3]
  sorry

end fraction_of_area_covered_l284_284531


namespace dist_between_centers_l284_284806

noncomputable def dist_centers_tangent_circles : ℝ :=
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  Real.sqrt 2 * (a₁ - a₂)

theorem dist_between_centers :
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  let C₁ := (a₁, a₁)
  let C₂ := (a₂, a₂)
  dist_centers_tangent_circles = 8 :=
by
  sorry

end dist_between_centers_l284_284806


namespace real_values_satisfying_inequality_l284_284336

theorem real_values_satisfying_inequality :
  ∀ x : ℝ, x ≠ 5 → (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ x ∈ set.Iic 0.76 ∪ set.Ioo 5 10.1 := by
  sorry

end real_values_satisfying_inequality_l284_284336


namespace magnitude_of_z_l284_284753

theorem magnitude_of_z (z : ℂ) (h : z ^ 2 = 4 - 3 * complex.I) : complex.abs z = Real.sqrt 5 :=
  sorry

end magnitude_of_z_l284_284753


namespace find_x_plus_y_l284_284518

variables {x y : ℝ}

def f (t : ℝ) : ℝ := t^2003 + 2002 * t

theorem find_x_plus_y (hx : f (x - 1) = -1) (hy : f (y - 2) = 1) : x + y = 3 :=
by
  sorry

end find_x_plus_y_l284_284518


namespace sometimes_MD_eq_ME_but_not_always_l284_284858

-- Defining the conditions of the problem
variables {A B C D E M : Type}
variables [triangle ABC : Type] {D : on_BC ABC ⟨AD⟩ bisects angle_BAC : Type} {E : on_BC ABC ⟨AE⟩ perpendicular_to BC : Type}
variables {M : midpoint_BC BC : Type}

-- Statement of the theorem
theorem sometimes_MD_eq_ME_but_not_always :
  ∃ (ABC : Type) (D : on_BC ABC ⟨AD⟩ bisects angle_BAC : Type) (E : on_BC ABC ⟨AE⟩ perpendicular_to BC : Type) (M : midpoint_BC BC : Type),
  sometimes (MD = ME) ∧ ¬ always (MD = ME) :=
by
  sorry

end sometimes_MD_eq_ME_but_not_always_l284_284858


namespace sum_is_6_l284_284923

-- Define the piecewise function f
def f (a b c x : ℝ) : ℝ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x + c

-- Define the conditions
variable (a b c : ℝ)

lemma problem_conditions : 
  f a b c 2 = 5 ∧ 
  f a b c 0 = 5 ∧ 
  f a b c (-2) = -10 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 :=
  sorry

-- Prove that a + b + c = 6 given the conditions
theorem sum_is_6 : a + b + c = 6 :=
  sorry

end sum_is_6_l284_284923


namespace question_incorrect_statement_l284_284625

theorem question_incorrect_statement (p q : Prop) (h : p ∨ q) : ¬ (p ∧ q) := by
  intro hpq
  cases h
  case inl =>
    exact hpq.left
  case inr =>
    exact hpq.right
  sorry

end question_incorrect_statement_l284_284625


namespace seventh_term_value_l284_284974

theorem seventh_term_value (a d : ℤ) (h1 : a = 12) (h2 : a + 3 * d = 18) : a + 6 * d = 24 := 
by
  sorry

end seventh_term_value_l284_284974


namespace attendance_methods_probability_AB_probability_each_event_l284_284018

open Finset

-- Step 1
/-- Given 6 individuals labeled as A, B, C, etc.,
prove that the number of different attendance methods,
with at least one person required to attend, is 63 -/
theorem attendance_methods (n : ℕ) (hn : n = 6) :
  (2^n - 1) = 63 :=
by {
  rw hn,
  norm_num,
}

-- Step 2
/-- Given 6 individuals participating in 6 different events,
prove that the probability that individual A does not participate in the first event
and individual B does not participate in the third event is 7/10 -/
theorem probability_AB (n : ℕ) (hn : n = 6) :
  (504 / 720 : ℝ) = 7 / 10 :=
by {
  rw hn,
  norm_num,
}

-- Step 3
/-- Given 6 individuals participating in 4 different events,
prove that the probability that each event has at least one person participating is 195/512 -/
theorem probability_each_event (n : ℕ) (hn : n = 6) (m : ℕ) (hm : m = 4) :
  (1560 / m^n : ℝ) = 195 / 512 :=
by {
  rw [hn, hm],
  norm_num,
}

end attendance_methods_probability_AB_probability_each_event_l284_284018


namespace det_dilation_matrix_4_l284_284891

-- Define the dilation matrix E centered at the origin with a scale factor of 4
def dilation_matrix_3x3 (scale_factor : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Diagonal.mk fun _ => scale_factor

-- Define the specific dilation matrix E with a scale factor of 4
noncomputable def E : Matrix (Fin 3) (Fin 3) ℝ := dilation_matrix_3x3 4

-- State the theorem to prove det E = 64
theorem det_dilation_matrix_4 : det E = 64 := by
  sorry

end det_dilation_matrix_4_l284_284891


namespace correct_statements_l284_284149

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * (abs x) + b * x + c

theorem correct_statements (b c : ℝ) :
  (∀ x, c = 0 → f (-x) b 0 = - f x b 0) ∧
  (∀ x, b = 0 → c > 0 → (f x 0 c = 0 → x = 0) ∧ ∀ y, f y 0 c ≤ 0) ∧
  (∀ x, ∃ k : ℝ, f (k + x) b c = f (k - x) b c) ∧
  ¬(∀ x, x > 0 → f x b c = c - b^2 / 2) :=
by
  sorry

end correct_statements_l284_284149


namespace log_equation_solutions_l284_284983

theorem log_equation_solutions :
  ∀ b : ℝ, log 2 (b^2 + 9 * b) = 7 ↔ 
  (b = (-9 + Real.sqrt 593) / 2 ∨ b = (-9 - Real.sqrt 593) / 2) ∧
  (¬∃ n : ℤ, b = n) :=
by
  sorry

end log_equation_solutions_l284_284983


namespace product_of_divisors_of_72_l284_284350

theorem product_of_divisors_of_72 : 
  (∏ d in (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))), d) = 2^18 * 3^12 := 
by
  -- required conditions
  have h72 : 72 = 2^3 * 3^2 := by norm_num
  have num_divisors : finset.card (finset.filter (λ d, 72 % d = 0) (finset.range (72+1))) = 12 := by sorry
  -- expounding solution steps
  -- sorry is used to skip actual proof steps
  sorry

end product_of_divisors_of_72_l284_284350


namespace shekar_biology_marks_l284_284954

theorem shekar_biology_marks (M S SS E A n B : ℕ) 
  (hM : M = 76)
  (hS : S = 65)
  (hSS : SS = 82)
  (hE : E = 67)
  (hA : A = 73)
  (hn : n = 5)
  (hA_eq : A = (M + S + SS + E + B) / n) : 
  B = 75 := 
by
  rw [hM, hS, hSS, hE, hn, hA] at hA_eq
  sorry

end shekar_biology_marks_l284_284954


namespace triangle_is_isosceles_right_l284_284111

-- Lean 4 Statement
theorem triangle_is_isosceles_right (A B C : ℝ) (h : Math.cos (A - B) + Math.sin (A + B) = 2) : 
  is_isosceles_right_triangle A B C :=
sorry

end triangle_is_isosceles_right_l284_284111


namespace alcohol_water_ratio_l284_284822

theorem alcohol_water_ratio (a b : ℚ) (h₁ : a = 3/5) (h₂ : b = 2/5) : a / b = 3 / 2 :=
by
  sorry

end alcohol_water_ratio_l284_284822


namespace cost_of_one_shirt_l284_284257

theorem cost_of_one_shirt
  (J S : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 76) :
  S = 18 :=
by
  sorry

end cost_of_one_shirt_l284_284257


namespace magnitude_of_sum_of_unit_vectors_with_angle_pi_over_3_l284_284768

variable (a b : ℝ^3)

theorem magnitude_of_sum_of_unit_vectors_with_angle_pi_over_3
  (h₁ : ∥a∥ = 1)
  (h₂ : ∥b∥ = 1)
  (h₃ : a ∙ b = Real.cos (Real.pi / 3)) :
  ∥a + b∥ = Real.sqrt 3 := sorry

end magnitude_of_sum_of_unit_vectors_with_angle_pi_over_3_l284_284768


namespace cosine_difference_theorem_l284_284746

noncomputable def cosine_difference : Prop :=
  ∀ (α : ℝ), 
    cos(α) = -3/5 ∧ (π/2 < α ∧ α < π) →
    cos(α - π/3) = (4 * real.sqrt 3 - 3) / 10

-- (Add sorry to skip the proof)
theorem cosine_difference_theorem: cosine_difference := 
  by
    sorry

end cosine_difference_theorem_l284_284746


namespace december_sales_fraction_l284_284882

theorem december_sales_fraction (A : ℝ) : 
  let jan_to_nov_sales := 11 * A,
      dec_sales := 6 * A,
      total_year_sales := jan_to_nov_sales + dec_sales in
  dec_sales / total_year_sales = 6 / 17 :=
by
  sorry

end december_sales_fraction_l284_284882


namespace eventually_not_divisible_by_17_l284_284883

def maximal_proper_divisor (n : ℕ) : ℕ := 
  {d : ℕ // (d ∣ n) ∧ (d < n) ∧ ∀ e : ℕ, (e ∣ n) ∧ (e < n) → (e ≤ d)}.val

def minimal_proper_divisor (n : ℕ) : ℕ := 
  {d : ℕ // (d ∣ n) ∧ (d < n) ∧ ∀ e : ℕ, (e ∣ n) ∧ (e < n) → (e ≥ d)}.val

theorem eventually_not_divisible_by_17 (n : ℕ) (h₁ : n > 1000) :
  ∃ k, let a := (λ (i : ℕ), if i = 0 then n else
              n + (maximal_proper_divisor (i-1)) - (minimal_proper_divisor (i-1))) in
  ¬ (17 ∣ a k) :=
sorry

end eventually_not_divisible_by_17_l284_284883


namespace total_votes_l284_284100

theorem total_votes (V : ℝ) 
  (h1 : 0.5 / 100 * V = 0.005 * V) 
  (h2 : 50.5 / 100 * V = 0.505 * V) 
  (h3 : 0.505 * V - 0.005 * V = 3000) : 
  V = 6000 := 
by
  sorry

end total_votes_l284_284100


namespace condition_sufficient_not_necessary_l284_284918

theorem condition_sufficient_not_necessary
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B ↔ C)
  (h3 : C → D) :
  (A → D) ∧ ¬(D → A) :=
by
  sorry

end condition_sufficient_not_necessary_l284_284918


namespace product_of_divisors_eq_l284_284344

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284344


namespace find_floor_of_apt_l284_284672

-- Define the conditions:
-- Number of stories
def num_stories : Nat := 9
-- Number of entrances
def num_entrances : Nat := 10
-- Total apartments in entrance 10
def apt_num : Nat := 333
-- Number of apartments per floor in each entrance (which is to be found)
def apts_per_floor_per_entrance : Nat := 4 -- from solution b)

-- Assertion: The floor number that apartment number 333 is on in entrance 10
theorem find_floor_of_apt (num_stories num_entrances apt_num apts_per_floor_per_entrance : ℕ) :
  1 ≤ apt_num ∧ apt_num ≤ num_stories * num_entrances * apts_per_floor_per_entrance →
  (apt_num - 1) / apts_per_floor_per_entrance + 1 = 3 :=
by
  sorry

end find_floor_of_apt_l284_284672


namespace quad_to_square_l284_284194

theorem quad_to_square (a b z : ℝ)
  (h_dim : a = 9) 
  (h_dim2 : b = 16) 
  (h_area : a * b = z * z) :
  z = 12 :=
by
  -- Proof outline would go here, but let's skip the actual proof for this definition.
  sorry

end quad_to_square_l284_284194


namespace simplified_fraction_proof_l284_284673

def is_simplified (n d : ℕ) (f : ℚ) : Prop :=
  f = (n : ℚ) / d ∧ (∀ k : ℕ, k ∣ n ∧ k ∣ d → k = 1)

theorem simplified_fraction_proof {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0) (h: a ≠ b):
  (is_simplified 1 (a - b) (1 / (a - b)) ∧ 
   ¬ is_simplified (b - a) (b^2 - a^2) ((b - a) / (b^2 - a^2)) ∧
   ¬ is_simplified 2 (6 * a * b) (2 / (6 * a * b)) ∧
   ¬ is_simplified (ab - a^2) a ((ab - a^2) / a)) :=
begin 
  sorry
end

end simplified_fraction_proof_l284_284673


namespace square_side_length_in_triangle_def_l284_284951

theorem square_side_length_in_triangle_def :
  ∃ t : ℚ, let DE := 6, EF := 8, DF := 10 in
  let altitude_k := (DE * EF) / DF in
  let t_solution := (10 * altitude_k) / (10 + altitude_k) in
  t = t_solution ∧ t = 120 / 37 := by
  sorry

end square_side_length_in_triangle_def_l284_284951


namespace solve_for_x_l284_284187

theorem solve_for_x (x : ℝ) : 10^(x + 4) = 100^x → x = 4 :=
by
  sorry

end solve_for_x_l284_284187


namespace spider_paths_l284_284973

-- Define the grid points and the binomial coefficient calculation.
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- The problem statement
theorem spider_paths : grid_paths 4 3 = 35 := by
  sorry

end spider_paths_l284_284973


namespace apples_first_year_l284_284676

theorem apples_first_year (A : ℕ) 
  (second_year_prod : ℕ := 2 * A + 8)
  (third_year_prod : ℕ := 3 * (2 * A + 8) / 4)
  (total_prod : ℕ := A + second_year_prod + third_year_prod) :
  total_prod = 194 → A = 40 :=
by
  sorry

end apples_first_year_l284_284676


namespace min_value_of_expr_l284_284905

noncomputable def real.min_value_expr (x y z : ℝ) : ℝ :=
  (x - 2)^2 + (y / x - 1)^2 + (z / y - 1)^2 + (5 / z - 1)^2

theorem min_value_of_expr :
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    real.min_value_expr x y z = 4 * (real.sqrt (real.sqrt 5) - 1)^2 :=
sorry

end min_value_of_expr_l284_284905


namespace present_age_of_son_l284_284253

variable (S M : ℕ)

-- Conditions
def condition1 : Prop := M = S + 32
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- Theorem stating the required proof
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 30 := by
  sorry

end present_age_of_son_l284_284253


namespace production_volume_bounds_l284_284937

theorem production_volume_bounds:
  ∀ (x : ℕ),
  (10 * x ≤ 800 * 2400) ∧ 
  (10 * x ≤ 4000000 + 16000000) ∧
  (x ≥ 1800000) →
  (1800000 ≤ x ∧ x ≤ 1920000) :=
by
  sorry

end production_volume_bounds_l284_284937


namespace discount_is_ten_percent_l284_284182

def original_price : ℝ := 500
def sale_price : ℝ := 450
def discount_amount (op sp : ℝ) : ℝ := op - sp
def discount_percentage (da op : ℝ) : ℝ := (da / op) * 100

theorem discount_is_ten_percent : discount_percentage (discount_amount original_price sale_price) original_price = 10 :=
by
  sorry

end discount_is_ten_percent_l284_284182


namespace part_1_part_2_l284_284412

variables (a b : ℝ^3)
variables (angle_a_b : ℝ := 2 * Real.pi / 3) -- 120 degrees in radians
variables (norm_a : ℝ := 2)
variables (norm_b : ℝ := 1)
variables (cos_angle_a_b : ℝ := Real.cos angle_a_b = -1 / 2)

#check angle_a_b
#check norm_a
#check norm_b
#check cos_angle_a_b

theorem part_1:
  (2 * a - b) • a = 9 :=
sorry

theorem part_2:
  |(a + 2 * b)| = Real.sqrt 10 :=
sorry

end part_1_part_2_l284_284412


namespace not_on_line_l284_284079

-- Defining the point (0,20)
def pt : ℝ × ℝ := (0, 20)

-- Defining the line equation
def line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- The proof problem stating that for all real numbers m and b, if m + b < 0, 
-- then the point (0, 20) cannot be on the line y = mx + b
theorem not_on_line (m b : ℝ) (h : m + b < 0) : ¬line m b pt := by
  sorry

end not_on_line_l284_284079


namespace tied_rounds_eq_seven_l284_284947

variable (t : ℕ) -- t represents the number of tied rounds
variable (w : ℕ) -- w represents the number of rounds won by either player

-- Conditions from the problem
def total_rounds : Prop := w + t = 10
def points_change : Prop := 5 * w + 4 * t = 43

-- Prove that the number of tied rounds is 7
theorem tied_rounds_eq_seven (h1 : total_rounds) (h2 : points_change) : t = 7 := 
by 
  sorry

end tied_rounds_eq_seven_l284_284947


namespace find_k_expression_y_minimize_y_l284_284648

open Real

namespace DormitoryCost

noncomputable def k (x : ℝ) (P : ℝ) : ℝ :=
  (P * (3 * x + 2))

theorem find_k :
  k 1 40 = 200 :=
by
  sorry

noncomputable def y (x : ℝ) : ℝ :=
  (200 / (3 * x + 2)) + 6 * x

theorem expression_y :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → y x = (200 / (3 * x + 2)) + 6 * x :=
by
  intros x h
  simp [y, k]
  split
  sorry

theorem minimize_y :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 5) ∧ y x = 36 ∧ x = 8 / 3 :=
by
  sorry

end DormitoryCost

end find_k_expression_y_minimize_y_l284_284648


namespace baseball_cards_per_friend_l284_284555

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l284_284555


namespace main_theorem_l284_284054

noncomputable def problem1 (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∃ x : ℝ, f x < 10 * a + 10) → a > 0

noncomputable def problem2 (a b : ℝ) : Prop :=
  a ∈ set.Ioi (0 : ℝ) ∧ b ∈ set.Ioi (0 : ℝ) ∧ a ≠ b → a^a * b^b > a^b * b^a

-- Main theorem stating the problems
theorem main_theorem (f : ℝ → ℝ) (a b : ℝ) :
  (problem1 f a) ∧ (problem2 a b) :=
by
  sorry

end main_theorem_l284_284054


namespace min_value_inequality_l284_284912

theorem min_value_inequality
    (x y z : ℝ)
    (h1 : 2 ≤ x)
    (h2 : x ≤ y)
    (h3 : y ≤ z)
    (h4 : z ≤ 5) :
    (x - 2) ^ 2 + (y / x - 1) ^ 2 + (z / y - 1) ^ 2 + (5 / z - 1) ^ 2 ≥ 4 * (Real.sqrt (4 : ℝ) 5 - 1) ^ 2 :=
by
    sorry

end min_value_inequality_l284_284912


namespace a4_eq_12_l284_284022

-- Definitions of the sequences and conditions
def S (n : ℕ) : ℕ := 
  -- sum of the first n terms, initially undefined
  sorry  

def a (n : ℕ) : ℕ := 
  -- terms of the sequence, initially undefined
  sorry  

-- Given conditions
axiom a2_eq_3 : a 2 = 3
axiom Sn_recurrence : ∀ n ≥ 2, S (n + 1) = 2 * S n

-- Statement to prove
theorem a4_eq_12 : a 4 = 12 :=
  sorry

end a4_eq_12_l284_284022


namespace sum_le_30_l284_284540

variable (a b x y : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx_pos : 0 < x) (hy_pos : 0 < y)
variable (h1 : a * x ≤ 5) (h2 : a * y ≤ 10) (h3 : b * x ≤ 10) (h4 : b * y ≤ 10)

theorem sum_le_30 : a * x + a * y + b * x + b * y ≤ 30 := sorry

end sum_le_30_l284_284540


namespace error_percent_in_area_l284_284843

theorem error_percent_in_area
  (L W : ℝ)
  (hL : L > 0)
  (hW : W > 0) :
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 0.8 := by
  sorry

end error_percent_in_area_l284_284843


namespace petya_payment_l284_284879

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end petya_payment_l284_284879


namespace average_weight_is_5_l284_284963

-- Define the given conditions
def weights :=
  {brown : ℕ, black : ℕ, white : ℕ, grey : ℕ // brown = 4 ∧ black = brown + 1 ∧ white = 2 * brown ∧ grey = black - 2}

-- Define a term to represent these specific weights
noncomputable def ter_weight : weights := sorry

-- Define the average weight calculation
def average_weight (w : weights) : ℕ :=
  (w.brown + w.black + w.white + w.grey) / 4

-- The theorem to prove the average weight is 5 pounds
theorem average_weight_is_5 : average_weight ter_weight = 5 :=
  sorry

end average_weight_is_5_l284_284963


namespace gcd_987654_876543_eq_3_l284_284616

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l284_284616


namespace sets_of_consecutive_integers_summing_to_20_l284_284068

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l284_284068


namespace symmetrical_reassembly_possible_l284_284695

theorem symmetrical_reassembly_possible (ABC : Triangle) : 
  ∃ pieces : List Triangle, (∀ piece ∈ pieces, is_subtriangle piece ABC) ∧
  (∃ line : Line, symmetrical_reassemble ABC pieces line) := 
sorry

end symmetrical_reassembly_possible_l284_284695


namespace angle_in_fourth_quadrant_l284_284461

noncomputable def in_quadrant_three (α : ℝ) : Prop :=
  (sin α < 0) ∧ (tan α < 0)

theorem angle_in_fourth_quadrant (α : ℝ) (h : in_quadrant_three α) : 
  ∃ k : ℤ, α = (4 * real.pi / 2) + (-α + k * real.pi / 2) := sorry

end angle_in_fourth_quadrant_l284_284461


namespace segments_equal_length_projection_segments_different_length_projection_l284_284168

-- Define lengths of the segments
variables {s₁ s₂ s₃ : Segment}

-- Define the length function
def length (s : Segment) : ℝ := -- assume some definition of length for segments
sorry

-- Problem a) proof statement
theorem segments_equal_length_projection :
  (length s₁ = length s₂ ∧ length s₂ = length s₃) →
  ∃ (plane : Plane), projections_equal_on_plane plane [s₁, s₂, s₃] :=
sorry

-- Problem b) proof statement
theorem segments_different_length_projection :
  (length s₁ = length s₂ ∧ length s₂ ≠ length s₃) →
  ¬ ∃ (plane : Plane), projections_equal_on_plane plane [s₁, s₂, s₃] :=
sorry

end segments_equal_length_projection_segments_different_length_projection_l284_284168


namespace division_problem_l284_284280

theorem division_problem (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end division_problem_l284_284280


namespace tank_capacity_l284_284679

def rate_outlet (C : ℕ) : ℚ := C / 10
def rate_inlet : ℚ := 8 * 60
def effective_rate_outlet (C : ℕ) : ℚ := rate_outlet C - rate_inlet

-- Given that the effective rate, which empties the tank in 16 hours, equals the rate of the tank divided by 16
theorem tank_capacity : ∃ C : ℕ, rate_outlet C - rate_inlet = C / 16 ∧ C = 1280 := 
by {
  -- Let C be the capacity of the tank
  let C := 1280,
  
  -- Calculate the effective rate of outlet when both pipes are open
  have h1 : effective_rate_outlet C = rate_outlet C - rate_inlet, from rfl,
  
  -- Calculate the rate when both pipes work in tandem to empty in 16 hours
  have h2 : C / 16 = rate_outlet C - rate_inlet, from calc
    C / 16 = (1280 : ℕ) / 16 : by sorry
          ... = rate_outlet 1280 - rate_inlet : by sorry,

  -- Prove the final capacity
  exact ⟨C, ⟨h1, h2⟩⟩,
}

end tank_capacity_l284_284679


namespace max_divisible_by_seven_sums_l284_284295

theorem max_divisible_by_seven_sums : 
  let n := 100 in 
  let sequence := {i // i > 0 ∧ i ≤ n} in 
  ∀ (arrangement : sequence -> sequence), 
    let pair_sum_divisible_by_7 := 
      (finset.range n).card (λ i, (arrangement i + arrangement ((i + 1) % n)) % 7 = 0) in
  pair_sum_divisible_by_7 ≤ 96 := 
begin
  sorry,
end

end max_divisible_by_seven_sums_l284_284295


namespace Shekar_weighted_average_l284_284176

def score_weighted_sum (scores_weights : List (ℕ × ℚ)) : ℚ :=
  scores_weights.foldl (fun acc sw => acc + (sw.1 * sw.2 : ℚ)) 0

def Shekar_scores_weights : List (ℕ × ℚ) :=
  [(76, 0.20), (65, 0.15), (82, 0.10), (67, 0.15), (55, 0.10), (89, 0.05), (74, 0.05),
   (63, 0.10), (78, 0.05), (71, 0.05)]

theorem Shekar_weighted_average : score_weighted_sum Shekar_scores_weights = 70.55 := by
  sorry

end Shekar_weighted_average_l284_284176


namespace domain_and_range_of_f_l284_284208

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2 * x + 1) / (3 * x - 4)

-- Prove the domain and range of f
theorem domain_and_range_of_f : 
  (∀ x : ℝ, 3 * x - 4 ≠ 0 → f x ∈ ℝ) ∧ 
  (∀ y : ℝ, y ∈ ℝ → y ≠ 2 / 3 ↔ ∃ x : ℝ, 3 * x - 4 ≠ 0 ∧ f x = y) := 
  by sorry

end domain_and_range_of_f_l284_284208


namespace overall_loss_is_267_percent_l284_284279

-- Define the cost prices and selling prices of the bicycles
def CP1 : ℝ := 1000
def SP1 : ℝ := 1080
def CP2 : ℝ := 1500
def SP2 : ℝ := 1100
def CP3 : ℝ := 2000
def SP3 : ℝ := 2200

-- Define the total cost price and total selling price
def TCP : ℝ := CP1 + CP2 + CP3
def TSP : ℝ := SP1 + SP2 + SP3

-- Define the overall gain or loss
def overall_gain_or_loss : ℝ := TSP - TCP

-- Define the overall loss percentage
def overall_loss_percentage : ℝ := (overall_gain_or_loss / TCP) * 100

-- Prove that the overall loss percentage is -2.67%
theorem overall_loss_is_267_percent : overall_loss_percentage = -2.67 := by
  sorry

end overall_loss_is_267_percent_l284_284279


namespace tan_alpha_is_three_fourths_l284_284771

theorem tan_alpha_is_three_fourths (α : ℝ) (k : ℤ) 
  (h1 : ℝ.cos α = 2 * (1 + ℝ.sin α)) 
  (h2 : α ≠ 2 * k * Real.pi - Real.pi / 2)
  : ℝ.tan α = 3 / 4 := 
sorry

end tan_alpha_is_three_fourths_l284_284771


namespace max_non_functional_segments_l284_284953

theorem max_non_functional_segments (d1 d3 : Fin 3) (d2 d4 : Fin 10) : 
  ∃ (max_segments : Nat), max_segments = 13 := 
by
  let first_digit_max_non_functional := 5
  let second_digit_max_non_functional := 2
  let third_digit_max_non_functional := 4
  let fourth_digit_max_non_functional := 2
  let max_segments := first_digit_max_non_functional + second_digit_max_non_functional + third_digit_max_non_functional + fourth_digit_max_non_functional
  use max_segments
  exact rfl
  sorry

end max_non_functional_segments_l284_284953


namespace josh_found_marbles_l284_284874

theorem josh_found_marbles :
  ∃ (F : ℕ), (F + 14 = 23) → (F = 9) :=
by
  existsi 9
  intro h
  linarith

end josh_found_marbles_l284_284874


namespace magnitude_of_z_l284_284781

-- Given conditions
def z : ℂ := (5 * complex.I) / (2 + complex.I) - 3 * complex.I

-- Statement to prove
theorem magnitude_of_z : complex.abs z = real.sqrt 2 := 
sorry

end magnitude_of_z_l284_284781


namespace mandy_total_payment_l284_284152

def promotional_rate := (1 / 3 : ℝ)
def normal_price := 30
def extra_fee := 15

theorem mandy_total_payment : 
  let first_month_cost := promotional_rate * normal_price in
  let fourth_month_cost := normal_price + extra_fee in
  let regular_cost := 4 * normal_price in
  first_month_cost + fourth_month_cost + regular_cost = 175 :=
by 
  -- Define individual costs
  let first_month_cost := promotional_rate * normal_price
  let fourth_month_cost := normal_price + extra_fee
  let regular_cost := 4 * normal_price
  
  -- Simplify and calculate the total cost
  have h_first := show first_month_cost = 10, by norm_num1 [first_month_cost, promotional_rate, normal_price, mul_eq_mul_right_iff]
  have h_fourth := show fourth_month_cost = 45, by norm_num1 [fourth_month_cost, normal_price, extra_fee, add_comm]
  have h_regular := show regular_cost = 120, by norm_num1 [regular_cost, normal_price, mul_comm]

  -- Final total sum
  calc
    first_month_cost + fourth_month_cost + regular_cost
        = 10 + 45 + 120 : by rw [h_first, h_fourth, h_regular]
    ... = 175 : by norma -- Use norm_num to finalize simplification


end mandy_total_payment_l284_284152


namespace d_won_zero_matches_l284_284473

theorem d_won_zero_matches (A B C D : Type) (plays_match : A → B → Prop)
  (won_against : A → B → Prop)
  (A_D_match : won_against A D)
  (same_wins : ∀ x : A, x ≠ D → (∃ y z : A, y ≠ z ∧ y ≠ D ∧ z ≠ D ∧ won_against x y ∧ won_against x z))
  : (∀ x : A, x = D → ¬ (won_against D x)) :=
by
  sorry

end d_won_zero_matches_l284_284473


namespace probability_five_green_marbles_approx_l284_284098

noncomputable def probability_five_green_marbles :=
  let p_green := 3 / 5
  let p_purple := 2 / 5
  let n_draws := 8
  let k_green := 5
  finset.card (finset.filter (λ s, (finset.card s) = k_green) (finset.powerset_univ (finset.range n_draws))) *
    (p_green ^ k_green) * (p_purple ^ (n_draws - k_green))

theorem probability_five_green_marbles_approx :
  abs (probability_five_green_marbles - 0.279) < 0.001 :=
sorry

end probability_five_green_marbles_approx_l284_284098


namespace seq_general_term_l284_284109

-- Define the sequence
def seq (n : ℕ) : ℕ → ℕ
  | 0     => 9
  | (k+1) => (seq k)^2

-- Define our target value in the sequence
def a (n : ℕ) : ℕ := 3^(2^n)

-- The main theorem to prove
theorem seq_general_term (n : ℕ) : seq n = a n :=
by
  sorry

end seq_general_term_l284_284109


namespace magnitude_of_w_l284_284509

noncomputable def w_magnitude (s : ℝ) (w : ℂ) : ℂ :=
  if h : |s| < 3 ∧ w + 2 / w = s then
    |w|
  else
    0

theorem magnitude_of_w (s : ℝ) (w : ℂ) (h₁ : |s| < 3) (h₂ : w + 2 / w = s) :
  |w| = real.sqrt 2 :=
by
  have h : |s| < 3 ∧ w + 2 / w = s := ⟨h₁, h₂⟩
  sorry

end magnitude_of_w_l284_284509


namespace find_y_l284_284291

-- Given conditions
def x : Int := 129
def student_operation (y : Int) : Int := x * y - 148
def result : Int := 110

-- The theorem statement
theorem find_y :
  ∃ y : Int, student_operation y = result ∧ y = 2 := 
sorry

end find_y_l284_284291


namespace compare_a_b_l284_284514

noncomputable def log : ℝ → ℝ := Real.log

variables (m n : ℝ)

theorem compare_a_b (h1 : m > n) (h2 : n > 1) :
  let a := (log (m * n)) ^ (1 / 2) - (log m) ^ (1 / 2)
  let b := (log m) ^ (1 / 2) - (log (m / n)) ^ (1 / 2)
  in a < b :=
by
  sorry

end compare_a_b_l284_284514


namespace total_sum_of_ages_is_correct_l284_284225

-- Definition of conditions
def ageOfYoungestChild : Nat := 4
def intervals : Nat := 3

-- Total sum calculation
def sumOfAges (ageOfYoungestChild intervals : Nat) :=
  let Y := ageOfYoungestChild
  Y + (Y + intervals) + (Y + 2 * intervals) + (Y + 3 * intervals) + (Y + 4 * intervals)

theorem total_sum_of_ages_is_correct : sumOfAges 4 3 = 50 :=
by
  sorry

end total_sum_of_ages_is_correct_l284_284225


namespace airplane_average_speed_l284_284299

-- Defining the conditions:
def TotalDistance : ℝ := 1140
def TotalTime : ℝ := 38

-- Defining the average speed formula:
def AverageSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- The theorem we need to prove:
theorem airplane_average_speed :
  AverageSpeed TotalDistance TotalTime = 30 := 
by
  sorry

end airplane_average_speed_l284_284299


namespace average_of_values_of_x_l284_284819

example (x : ℝ) (h : sqrt (3 * x^2 + 2) = sqrt 50) : x = 4 ∨ x = -4 := by
  sorry

theorem average_of_values_of_x (x : ℝ) (hx: sqrt (3 * x^2 + 2) = sqrt 50) :
    (4 + (-4)) / 2 = 0 := by
  have h1 : 3 * x^2 + 2 = 50 := by
    exact (congrArg (fun x => x^2) hx)
  have h2 : 3 * x^2 = 48 := by
    linarith
  have h3 : x^2 = 16 := by
    linarith
  have h4 : x = 4 ∨ x = -4 := by
    apply sqrt_eq_iff_sq_eq'
    linarith
  have h5 : (4 + -4) / 2 = 0 := by
    ring
  exact h5

end average_of_values_of_x_l284_284819


namespace trajectory_and_velocity_l284_284694

noncomputable def omega : ℝ := 10
noncomputable def OA : ℝ := 90
noncomputable def AB : ℝ := 90
noncomputable def MB : ℝ := AB / 3
def position_A (t : ℝ) : ℝ × ℝ := (OA * Real.cos (omega * t), OA * Real.sin (omega * t))
def position_B (lambda : ℝ) : ℝ × ℝ := (lambda, 0)

theorem trajectory_and_velocity (t lambda : ℝ) :
  let rA := position_A t;
      rB := position_B lambda;
      rM := (OA * Real.cos (omega * t) + (lambda - OA * Real.cos (omega * t)) / 3, 
            OA * Real.sin (omega * t) - OA * Real.sin (omega * t) / 3);
      vM := ((-1) * omega * 60 * Real.sin (omega * t), omega * 60 * Real.cos (omega * t))
  in rM = (60 * Real.cos (omega * t) + lambda / 3, 60 * Real.sin (omega * t))
  ∧ vM = (-600 * Real.sin (omega * t), 600 * Real.cos (omega * t)) :=
by
  sorry

end trajectory_and_velocity_l284_284694


namespace gcd_consecutive_terms_l284_284244

theorem gcd_consecutive_terms (n : ℕ) : 
  Nat.gcd (2 * Nat.factorial n + n) (2 * Nat.factorial (n + 1) + (n + 1)) = 1 :=
by
  sorry

end gcd_consecutive_terms_l284_284244


namespace locus_of_centers_is_apollonius_l284_284588

noncomputable def locus_of_centers (A B : Point) (φ : Real) (h1 : A ≠ B) (h2 : 0 < φ ∧ φ < π) : Set Point :=
{ O | ∃ k : Circle, k.contains A ∧ angle_at_point k B A = φ ∧ k.center = O }

theorem locus_of_centers_is_apollonius (A B : Point) (φ : Real) (h1 : A ≠ B) (h2 : 0 < φ ∧ φ < π) : 
  ∀ O, O ∈ locus_of_centers A B φ h1 h2 
  ↔ O ∈ apollonius_circle A B (sin (φ / 2)) :=
sorry

end locus_of_centers_is_apollonius_l284_284588


namespace lowest_student_number_l284_284278

theorem lowest_student_number (total_students : ℕ) (sampled_students : ℕ) (highest_number : ℕ) 
  (H1 : total_students = 48) (H2 : sampled_students = 8) (H3 : highest_number = 48) :
  let interval := total_students / sampled_students in
  let lowest_number := highest_number - interval * (sampled_students - 1) in
  lowest_number = 6 :=
by
  sorry

end lowest_student_number_l284_284278


namespace max_consecutive_sum_k_l284_284725

theorem max_consecutive_sum_k : 
  ∃ k n : ℕ, k = 486 ∧ 3^11 = (0 to k-1).sum + n * k := 
sorry

end max_consecutive_sum_k_l284_284725


namespace at_least_one_inequality_holds_l284_284901

theorem at_least_one_inequality_holds
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l284_284901


namespace tournament_total_players_l284_284102

theorem tournament_total_players (n : ℕ) (n_games : ℕ) :
  (∃ n, 
    (∀ i j : ℕ, i ≠ j → i ≠ 0 → j ≠ 0 → 
      ((player_points i = 1 ∧ player_points j = 0) ∨ 
      (player_points i = 0 ∧ player_points j = 1) ∨ 
      (player_points i = 0.5 ∧ player_points j = 0.5))) ∧
    (∀ k : ℕ, k ≠ 0 → (∑ x in range 10, player_points (weakest_players k x)) = 
      0.5 * (∑ y in range 10, player_points y))) →
   (n + 10 = 25)) 
sorry

end tournament_total_players_l284_284102


namespace students_playing_both_correct_l284_284470

def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def neither_players : ℕ := 7
def students_playing_both : ℕ := 17

theorem students_playing_both_correct :
  total_students - neither_players = (football_players + long_tennis_players) - students_playing_both :=
by 
  sorry

end students_playing_both_correct_l284_284470


namespace largest_number_among_pi_neg3_sqrt8_and_3_l284_284622

theorem largest_number_among_pi_neg3_sqrt8_and_3 : 
  ∃ x : ℝ, ((x = real.pi ∨ x = -3 ∨ x = real.sqrt 8 ∨ x = (-(-3))) ∧ 
  (x = real.pi ∨ x < real.pi)) :=
by
  sorry

end largest_number_among_pi_neg3_sqrt8_and_3_l284_284622


namespace product_of_divisors_eq_l284_284345

theorem product_of_divisors_eq :
  ∏ d in (Finset.filter (λ x : ℕ, x ∣ 72) (Finset.range 73)), d = (2^18) * (3^12) := by
  sorry

end product_of_divisors_eq_l284_284345


namespace car_b_speed_l284_284689

theorem car_b_speed (v : ℝ) : 
  let dA := 58 * 6 in
  let dB := v * 6 in
  let relative_distance := 40 + 8 in
  dA = dB + relative_distance → 
  v = 50 :=
by
  intros h
  sorry

end car_b_speed_l284_284689


namespace some_trinks_not_zorbs_l284_284064

variables {Zorb Glarb Trink : Type} 
variable (zorbs : Zorb -> Prop)
variable (glarbs : Glarb -> Prop)
variable (trinks : Trink -> Prop)

-- Hypothesis I: All Zorbs are not Glarbs.
variable (H₁ : ∀ x, zorbs x -> ¬ glarbs x)

-- Hypothesis II: Some Glarbs are Trinks.
variable (H₂ : ∃ y, glarbs y ∧ trinks y)

-- Conclusion: Some Trinks are not Zorbs.
theorem some_trinks_not_zorbs : ∃ z, trinks z ∧ ¬ zorbs z :=
sorry

end some_trinks_not_zorbs_l284_284064


namespace correct_number_of_propositions_l284_284960

-- Definitions for lines and planes, and perpendicularity and parallelism relations
variables {Line Plane : Type}
variables [Different_lines : ∀ (a b : Line), a ≠ b]
variables [Different_planes : ∀ (α β : Plane), α ≠ β]
variables (perp : Line → Line → Prop)
variables (perp_plane_line : Line → Plane → Prop)
variables (parallel_plane_line : Line → Plane → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Definitions from problem
variables a b : Line
variables α β : Plane

-- Propositions to verify
def prop1 := perp a b ∧ perp_plane_line a α → parallel_plane_line b α
def prop2 := parallel_plane_line a α ∧ perp_planes α β → perp_plane_line a β
def prop3 := perp_plane_line a β ∧ perp_planes α β → parallel_plane_line a α
def prop4 := perp a b ∧ perp_plane_line a α ∧ perp_plane_line b β → perp_planes α β

-- The number of correct propositions
def number_of_correct_props := (ite prop1.to_bool 1 0) + (ite prop2.to_bool 1 0) +
                               (ite prop3.to_bool 1 0) + (ite prop4.to_bool 1 0)

theorem correct_number_of_propositions : number_of_correct_props = 1 := 
sorry

end correct_number_of_propositions_l284_284960
