import Mathlib

namespace min_pictures_needed_l560_560963

theorem min_pictures_needed (n m : ℕ) (participants : Fin n → Fin m → Prop)
  (h1 : n = 60) (h2 : m ≤ 30)
  (h3 : ∀ (i j : Fin n), ∃ (k : Fin m), participants i k ∧ participants j k) :
  m = 6 :=
sorry

end min_pictures_needed_l560_560963


namespace twelve_hens_lay_48_eggs_in_twelve_days_l560_560982

theorem twelve_hens_lay_48_eggs_in_twelve_days :
  (∀ (hens eggs days : ℕ), hens = 3 → eggs = 3 → days = 3 → eggs / (hens * days) = 1/3) → 
  ∀ (hens days : ℕ), hens = 12 → days = 12 → hens * days * (1/3) = 48 :=
by
  sorry

end twelve_hens_lay_48_eggs_in_twelve_days_l560_560982


namespace length_of_chord_l560_560391

theorem length_of_chord {M : ℝ × ℝ} (hM : M = (1, 2)) :
  let r := 3,
      O := (0 : ℝ, 0 : ℝ),
      OM := (λ x y: ℝ, (x - 0)^2 + (y - 0)^2),
      dist_OM := sqrt (OM 1 2) in
  dist_OM = sqrt (r^2 - 3) → 2 * sqrt (r^2 - dist_OM^2) = 4 :=
by
  sorry

end length_of_chord_l560_560391


namespace parallel_vectors_addition_l560_560814

theorem parallel_vectors_addition (m : ℝ) 
  (h_parallel : (1 : ℝ) / (-2 : ℝ) = (2 : ℝ) / m) : 
  2 * (1, 2 : ℝ × ℝ) + 3 * (-2, m) = (-4, -8) :=
by
  sorry

end parallel_vectors_addition_l560_560814


namespace evaluate_expression_at_three_l560_560340

theorem evaluate_expression_at_three : 
  (3^2 + 3 * (3^6) = 2196) :=
by
  sorry -- This is where the proof would go

end evaluate_expression_at_three_l560_560340


namespace triangle_division_l560_560863

theorem triangle_division (AB AC : ℝ) (h_AB : 100 < AB ∧ AB < 101) (h_AC : 99 < AC ∧ AC < 100) :
∃ n : ℕ, (n ≤ 21) ∧ ∃ (divs : List {t // is_triangle t ∧ ∃ a : ℝ, a = 1}), length divs = n ∧ (∀ t ∈ divs, 1 ∈ (sides t)) :=
by

-- We have the conditions on AB and AC
have h_AB' := h_AB
have h_AC' := h_AC

-- Placeholder for proof steps, to be elaborated
sorry

end triangle_division_l560_560863


namespace roll_probability_l560_560300

noncomputable def probability_allison_rolls_greater : ℚ :=
  let p_brian := 5 / 6  -- Probability of Brian rolling 5 or lower
  let p_noah := 1       -- Probability of Noah rolling 5 or lower (since all faces roll 5 or lower)
  p_brian * p_noah

theorem roll_probability :
  probability_allison_rolls_greater = 5 / 6 := by
  sorry

end roll_probability_l560_560300


namespace determinant_eval_l560_560647

theorem determinant_eval : 
  let A := ![![3, 0, -2], ![5, 6, -4], ![3, 3, 7]] in
  Matrix.det A = 168 :=
by
  sorry

end determinant_eval_l560_560647


namespace range_of_m_l560_560393

def f (m x : ℝ) : ℝ := 4 * Real.log x - m * x ^ 2 + 1

theorem range_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc (1 : ℝ) Real.exp 1, f m x ≤ 0) : 
  m ≥ 2 * Real.sqrt (Real.exp 1) / Real.exp 1 := 
sorry

end range_of_m_l560_560393


namespace two_d_minus_c_zero_l560_560524

theorem two_d_minus_c_zero :
  ∃ (c d : ℕ), (∀ x : ℕ, x^2 - 18 * x + 72 = (x - c) * (x - d)) ∧ c > d ∧ (2 * d - c = 0) := 
sorry

end two_d_minus_c_zero_l560_560524


namespace geometric_series_sum_squares_l560_560319

theorem geometric_series_sum_squares (a r : ℝ) (hr : -1 < r) (hr2 : r < 1) :
  (∑' n : ℕ, a^2 * r^(3 * n)) = a^2 / (1 - r^3) :=
by
  -- Note: Proof goes here
  sorry

end geometric_series_sum_squares_l560_560319


namespace average_weight_difference_is_6_l560_560572

noncomputable def joe_weight : ℝ := 42
noncomputable def original_avg_weight : ℝ := 30
noncomputable def new_avg_increase : ℝ := 1
noncomputable def final_avg_weight : ℝ := 30

theorem average_weight_difference_is_6 :
  ∃ (n : ℕ), 
    let original_total_weight := original_avg_weight * n,
        new_avg_weight := original_avg_weight + new_avg_increase,
        total_weight_after_joe := original_total_weight + joe_weight,
        num_students_after_joe := n + 1,
        final_total_weight := final_avg_weight * (num_students_after_joe - 2) in
    original_total_weight = 30 * n ∧ 
    total_weight_after_joe / num_students_after_joe = new_avg_weight ∧
    (total_weight_after_joe - 2 * (final_avg_weight + 6)) = final_total_weight ∧
    abs ((final_avg_weight + 6) - joe_weight) = 6 :=
sorry

end average_weight_difference_is_6_l560_560572


namespace sin_cos_positive_implies_first_or_third_quadrant_l560_560121

theorem sin_cos_positive_implies_first_or_third_quadrant (θ : ℝ) (h : sin θ * cos θ > 0) : 
    (0 < θ ∧ θ < π/2) ∨ (π < θ ∧ θ < 3*π/2) :=
sorry

end sin_cos_positive_implies_first_or_third_quadrant_l560_560121


namespace arithmetic_progression_binom_coeff_l560_560304

theorem arithmetic_progression_binom_coeff (n : ℕ) (h1 : n > 3) 
  (h2 : (n - 1)! ≠ 0) (h3 : (n - 2)! ≠ 0) (h4 : (n - 3)! ≠ 0)
  (h5 : ((n * (n - 1) / 2) - n) = ((n * (n - 1) * (n - 2) / 6) - (n * (n - 1) / 2))) :
  n = 7 :=
  sorry

end arithmetic_progression_binom_coeff_l560_560304


namespace coeff_x3_term_l560_560734

theorem coeff_x3_term :
  let f := (3 * X^3 + 2 * X^2 + X + 1) * (2 * X^2 + X + 4) * (X^2 + 2 * X + 3) in
  f.coeff 3 = 73 :=
by
  let f := (3 * X^3 + 2 * X^2 + X + 1) * (2 * X^2 + X + 4) * (X^2 + 2 * X + 3)
  show f.coeff 3 = 73
  sorry

end coeff_x3_term_l560_560734


namespace second_number_l560_560593

theorem second_number (A B : ℝ) (h1 : 0.50 * A = 0.40 * B + 180) (h2 : A = 456) : B = 120 := 
by
  sorry

end second_number_l560_560593


namespace original_rulers_l560_560980

theorem original_rulers : ∃ x : ℕ, x + 25 = 71 ∧ x = 46 :=
begin
  use 46,
  split,
  { 
    -- First condition: x + 25 = 71
    exact rfl, -- This means that 46 + 25 = 71
  },
  { 
    -- Second condition: x = 46
    exact rfl, -- This is the direct assumption of x = 46
  }
end

end original_rulers_l560_560980


namespace problem_statement_l560_560043

noncomputable def problem_data :=
{a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b)
  (h₃ : (1, 2 * real.sqrt 3 / 3) ∈ set_of (λ p : ℝ × ℝ, (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1))

def ellipse_eq (a b : ℝ) := λ (x y : ℝ), (x^2) / a^2 + (y^2) / b^2 = 1

def line_eq (b : ℝ) := λ (x y : ℝ), x + y + b = 0

def circle_eq := λ (x y : ℝ), x^2 + y^2 = 2

theorem problem_statement (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b)
  (h₃ : (1, 2 * real.sqrt 3 / 3) ∈ set_of (λ p : ℝ × ℝ, ellipse_eq a b p.1 p.2))
  (hb : (line_eq b).intersect (circle_eq) with i) (h₄ : length_of (i.chord) = 2) :
  (a = real.sqrt 3 ∧ b = real.sqrt 2 ∧ ellipse_eq a b = λ x y, x^2 / 3 + y^2 / 2 = 1) ∧
  -- Prove if |MN|/|OQ|^2 is constant and its value
  (∀ Q ∉ x_axis, let Q OQ := (Q.1)^2 + (Q.2)^2 = 1 + m^2, ∃ MN. MN / Q^2 = 2.sqrt 3 / 3) ∧
  -- Prove max value of S
  (∃ S : ℝ, S = (area_of_triangle Q F2 M) + (area_of_triangle O F2 N), S.max = 2.sqrt 3 / 3) :=
begin
  sorry,
end

end problem_statement_l560_560043


namespace log_comparison_necessity_log_comparison_sufficiency_log_comparison_false_log_comparison_nec_not_suff_l560_560365

noncomputable def log_condition (a b : ℝ) : Prop :=
  a > b

theorem log_comparison_necessity (a b : ℝ) :
  log_condition a b → (log a > log b) :=
sorry

theorem log_comparison_sufficiency (a b : ℝ) :
  (log a > log b) → log_condition a b :=
sorry

theorem log_comparison_false (a b : ℝ) :
  ¬ log_condition a b → ¬ (log a > log b) :=
sorry

theorem log_comparison_nec_not_suff (a b : ℝ) :
  (∀ a b, log_comparison_sufficiency a b) → (¬(∀ a b, log_comparison_necessity a b)) :=
sorry

end log_comparison_necessity_log_comparison_sufficiency_log_comparison_false_log_comparison_nec_not_suff_l560_560365


namespace g_at_10_is_neg48_l560_560529

variable (g : ℝ → ℝ)

-- Given condition
axiom functional_eqn : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2

-- Mathematical proof statement
theorem g_at_10_is_neg48 : g 10 = -48 :=
  sorry

end g_at_10_is_neg48_l560_560529


namespace stadium_length_in_yards_l560_560959

def length_in_feet := 183
def feet_per_yard := 3

theorem stadium_length_in_yards : length_in_feet / feet_per_yard = 61 := by
  sorry

end stadium_length_in_yards_l560_560959


namespace max_value_of_quadratic_l560_560358

noncomputable def quadratic_function := λ x : ℝ, -2 * x^2 - 5

theorem max_value_of_quadratic :
  ∃ M : ℝ, (∀ x : ℝ, quadratic_function x ≤ M) ∧ (M = -5) := 
sorry

end max_value_of_quadratic_l560_560358


namespace fixed_point_parabola_l560_560894

theorem fixed_point_parabola : ∀ t : ℝ, ∃ x y : ℝ, x = 1.5 ∧ y = 9 ∧ y = 4 * x ^ 2 + 2 * t * x - 3 * t :=
by
  intro t
  let x := 1.5
  let y := 9
  use x, y
  split
  { refl }
  split
  { refl }
  sorry

end fixed_point_parabola_l560_560894


namespace sqrt_case_l560_560842

theorem sqrt_case {x : ℝ} (h : |1 - x| = 1 + |x|) (hx : x ≤ 0) : sqrt ((x - 1)^2) = 1 - x :=
by
  sorry

end sqrt_case_l560_560842


namespace parabola_hyperbola_distance_l560_560953

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

noncomputable def hyperbola_asymptote : ℝ → ℝ := λ x, √3 * x

theorem parabola_hyperbola_distance :
  distance parabola_focus (λ x, hyperbola_asymptote x) = √3 :=
sorry

end parabola_hyperbola_distance_l560_560953


namespace ratio_of_area_l560_560207

noncomputable def area_ratio (l w r : ℝ) : ℝ :=
  if h1 : 2 * l + 2 * w = 2 * Real.pi * r 
  ∧ l = 2 * w then 
    (l * w) / (Real.pi * r ^ 2) 
  else 
    0

theorem ratio_of_area (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) :
  area_ratio l w r = 2 * Real.pi / 9 :=
by
  unfold area_ratio
  simp [h1, h2]
  sorry

end ratio_of_area_l560_560207


namespace response_activity_solutions_l560_560296

theorem response_activity_solutions (x y z : ℕ) :
  5 * x + 4 * y + 3 * z = 15 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1) :=
by
  sorry

end response_activity_solutions_l560_560296


namespace method_A_equals_method_B_method_B_better_l560_560195

noncomputable def P_A : ℕ → ℚ 
| 1 := 1 / 6
| 2 := 1 / 6
| 3 := 1 / 6
| 4 := 1 / 6
| 5 := 1 / 3
| _ := 0 

noncomputable def P_B : ℕ → ℚ 
| 2 := 1 / 3
| 3 := 2 / 3
| _ := 0 

noncomputable def P_equal := P_A 2 * P_B 2 + P_A 3 * P_B 3

noncomputable def E_eta := (1 * (1 / 6) + 2 * (1 / 6) + 3 * (1 / 6) + 4 * (1 / 6) + 5 * (1 / 3))

noncomputable def E_xi := (2 * (1 / 3) + 3 * (2 / 3))

theorem method_A_equals_method_B : P_equal = 1 / 6 :=
by sorry

theorem method_B_better : E_xi < E_eta :=
by sorry

end method_A_equals_method_B_method_B_better_l560_560195


namespace carl_profit_l560_560311

-- Define the conditions
def price_per_watermelon : ℕ := 3
def watermelons_start : ℕ := 53
def watermelons_end : ℕ := 18

-- Define the number of watermelons sold
def watermelons_sold : ℕ := watermelons_start - watermelons_end

-- Define the profit
def profit : ℕ := watermelons_sold * price_per_watermelon

-- State the theorem about Carl's profit
theorem carl_profit : profit = 105 :=
by
  -- Proof can be filled in later
  sorry

end carl_profit_l560_560311


namespace d_minus_r_eq_15_l560_560209

theorem d_minus_r_eq_15 (d r : ℤ) (h_d_gt_1 : d > 1)
  (h1 : 1059 % d = r)
  (h2 : 1417 % d = r)
  (h3 : 2312 % d = r) :
  d - r = 15 :=
sorry

end d_minus_r_eq_15_l560_560209


namespace sum_moments_equal_l560_560586

theorem sum_moments_equal
  (x1 x2 x3 y1 y2 : ℝ)
  (m1 m2 m3 n1 n2 : ℝ) :
  n1 * y1 + n2 * y2 = m1 * x1 + m2 * x2 + m3 * x3 :=
sorry

end sum_moments_equal_l560_560586


namespace triangle_angle_and_side_ratio_l560_560444

theorem triangle_angle_and_side_ratio
  (A B C : Real)
  (a b c : Real)
  (h1 : a / Real.sin A = b / Real.sin B)
  (h2 : b / Real.sin B = c / Real.sin C)
  (h3 : (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C)) :
  C = Real.pi / 3 ∧ (1 < (a + b) / c ∧ (a + b) / c < 2) :=
by
  sorry


end triangle_angle_and_side_ratio_l560_560444


namespace largest_prime_divisor_P2_l560_560561

def P (x : ℕ) : ℤ := ∑ k in finset.range 2014, (2015 - k) * x^k

theorem largest_prime_divisor_P2 : 
  ∃ p : ℕ, nat.prime p ∧ p > 1 ∧ ∀ q : ℕ, nat.prime q ∧ q > p → ¬ nat.dvd q (P 2) :=
sorry

end largest_prime_divisor_P2_l560_560561


namespace alice_ball_two_turns_l560_560624

-- Define the conditions
def aliceStart : Prop := true
def probAliceToBob : ℚ := 1 / 2
def probAliceKeeps : ℚ := 1 / 2
def probBobToAlice : ℚ := 2 / 5
def probBobKeeps : ℚ := 3 / 5

-- Define what we need to prove
theorem alice_ball_two_turns: aliceStart →
  let scenario1 := probAliceToBob * probBobToAlice  -- Alice to Bob, then Bob to Alice
  let scenario2 := probAliceKeeps * probAliceKeeps  -- Alice keeps twice
  (scenario1 + scenario2) = 9 / 20 :=
begin
  intros,
  sorry
end

end alice_ball_two_turns_l560_560624


namespace board_arithmetic_impossibility_l560_560159

theorem board_arithmetic_impossibility :
  ¬ (∃ (a b : ℕ), a ≡ 0 [MOD 7] ∧ b ≡ 1 [MOD 7] ∧ (a * b + a^3 + b^3) = 2013201420152016) := 
    sorry

end board_arithmetic_impossibility_l560_560159


namespace sequence_pos_integer_and_even_iff_l560_560352

namespace MathProof

def sequence (r : ℕ) : ℕ → ℕ
| 1       := 1
| (n + 1) := (n * sequence r n + 2 * (n + 1)^(2 * r)) / (n + 2)

theorem sequence_pos_integer_and_even_iff (r : ℕ) (n : ℕ) (hn : 1 ≤ n) :
  sequence r n > 0 ∧ (even (sequence r n) ↔ n % 4 = 0 ∨ n % 4 = 3) :=
sorry

end MathProof

end sequence_pos_integer_and_even_iff_l560_560352


namespace expand_polynomial_l560_560688

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560688


namespace instantaneous_rate_of_change_at_e_l560_560307

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem instantaneous_rate_of_change_at_e : deriv f e = 0 := by
  sorry

end instantaneous_rate_of_change_at_e_l560_560307


namespace average_first_six_numbers_l560_560950

theorem average_first_six_numbers (A : ℝ) (h1 : (11 : ℝ) * 9.9 = (6 * A + 6 * 11.4 - 22.5)) : A = 10.5 :=
by sorry

end average_first_six_numbers_l560_560950


namespace actual_time_l560_560979

def digit_in_range (a b : ℕ) : Prop := 
  (a = b + 1 ∨ a = b - 1)

def time_malfunctioned (h m : ℕ) : Prop :=
  digit_in_range 0 (h / 10) ∧ -- tens of hour digit (0 -> 1 or 9)
  digit_in_range 0 (h % 10) ∧ -- units of hour digit (0 -> 1 or 9)
  digit_in_range 5 (m / 10) ∧ -- tens of minute digit (5 -> 4 or 6)
  digit_in_range 9 (m % 10)   -- units of minute digit (9 -> 8 or 0)

theorem actual_time : ∃ h m : ℕ, time_malfunctioned h m ∧ h = 11 ∧ m = 48 :=
by
  sorry

end actual_time_l560_560979


namespace ellipse_area_is_50_pi_l560_560104

noncomputable def ellipse_area (a b : ℝ) : ℝ :=
  Real.pi * a * b

theorem ellipse_area_is_50_pi
  (x1 y1 x2 y2 xc yc x3 y3 : ℝ)
  (h1 : x1 = -9)
  (h2 : y1 = 3)
  (h3 : x2 = 11)
  (h4 : y2 = 3)
  (h5 : x3 = 9)
  (h6 : y3 = 6)
  (h7 : xc = (x1 + x2) / 2)
  (h8 : yc = (y1 + y2) / 2)
  (h9 : y1 = yc)
  (h10 : y2 = yc)
  (h11 : (x3 - xc)^2 / ((x2 - xc)^2) + (y3 - yc)^2 / b^2 = 1)
  (h12 : a = (x2 - x1) / 2)
  (h13 : b = sqrt((x3 - xc)^2 / ((x2 - xc)^2) + (y3 - yc)^2))
  : ellipse_area a b = 50 * Real.pi := by
sorry

end ellipse_area_is_50_pi_l560_560104


namespace triangle_area_is_six_l560_560320

-- Conditions
def line_equation (Q : ℝ) : Prop :=
  ∀ (x y : ℝ), 12 * x - 4 * y + (Q - 305) = 0

def area_of_triangle (Q R : ℝ) : Prop :=
  R = (305 - Q) ^ 2 / 96

-- Question: Given a line equation forming a specific triangle, prove the area R equals 6.
theorem triangle_area_is_six (Q : ℝ) (h1 : Q = 281 ∨ Q = 329) :
  ∃ R : ℝ, line_equation Q → area_of_triangle Q R → R = 6 :=
by {
  sorry -- Proof to be provided
}

end triangle_area_is_six_l560_560320


namespace tangent_lines_to_circle_max_min_y_div_x_l560_560795

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

def point_A : ℝ × ℝ := (3, 5)

theorem tangent_lines_to_circle (x y : ℝ) :
  circle_eq x y → 
  let A := point_A in 
  (x = 3 ∨ 4 * y - 3 * x - 11 = 0) :=
sorry

theorem max_min_y_div_x (x y : ℝ) :
  circle_eq x y →
  let k := y / x in 
  k = (6 + 2 * Real.sqrt 3) / 3 ∨ k = (6 - 2 * Real.sqrt 3) / 3 :=
sorry

end tangent_lines_to_circle_max_min_y_div_x_l560_560795


namespace valid_decomposition_2009_l560_560193

/-- A definition to determine whether a number can be decomposed
    into sums of distinct numbers with repeated digits representation. -/
def decomposable_2009 (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  a = 1111 ∧ b = 777 ∧ c = 66 ∧ d = 55 ∧ a + b + c + d = n

theorem valid_decomposition_2009 :
  decomposable_2009 2009 :=
sorry

end valid_decomposition_2009_l560_560193


namespace donut_cubes_eaten_l560_560491

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end donut_cubes_eaten_l560_560491


namespace sequence_periodicity_l560_560933

theorem sequence_periodicity (a : ℕ → ℕ) (n : ℕ) (h : ∀ k, a k = 6^k) :
  a (n + 5) % 100 = a n % 100 :=
by sorry

end sequence_periodicity_l560_560933


namespace g_g_g_8_l560_560653

def g (x : ℝ) : ℝ :=
if x < 5 then x^2 + 1 else 2*x - 8

theorem g_g_g_8 : g (g (g 8)) = 8 :=
by
  sorry

end g_g_g_8_l560_560653


namespace solve_container_capacity_l560_560249

noncomputable def container_capacity (C : ℝ) :=
  (0.75 * C - 0.35 * C = 48)

theorem solve_container_capacity : ∃ C : ℝ, container_capacity C ∧ C = 120 :=
by
  use 120
  constructor
  {
    -- Proof that 0.75 * 120 - 0.35 * 120 = 48
    sorry
  }
  -- Proof that C = 120
  sorry

end solve_container_capacity_l560_560249


namespace vector_ap_eq_l560_560779

variables {A B C P : Type} [affine_space P]
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C]

axiom vector_eq : ∀ (u v : A), (u = v) ↔ (u - v = 0)
axiom bc_eq_2cp (a b c p : A) : 
  (c - b = 2 • (p - c)) ↔ (c - b = 2 • (p - c))

theorem vector_ap_eq (a b c p : A) (h : (c - b = 2 • (p - c))) :
  (p - a) = (-1 / 2 • (b - a)) + (3 /2 • (c - a)) :=
sorry

end vector_ap_eq_l560_560779


namespace log_6_eq_eval_3_5_a_c_log_ratio_log_15_sq_minus_l560_560471

-- Setting up the conditions as definitions
def a : ℝ := log 2
def b : ℝ := log 3
def c : ℝ := log 5

-- Problem 1: Prove \log 6 = a + b
theorem log_6_eq : log 6 = a + b := sorry

-- Problem 2: Prove 3.5a + 3.5c = 3.5
theorem eval_3_5_a_c : 3.5 * a + 3.5 * c = 3.5 := sorry

-- Problem 3: Prove \log 30 / \log 15 = (a + b + c) / (b + c)
theorem log_ratio : log 30 / log 15 = (a + b + c) / (b + c) := sorry

-- Problem 4: Prove (\log 15)^2 - \log 15 = (b + c) * (b + c - 1)
theorem log_15_sq_minus : (log 15) ^ 2 - log 15 = (b + c) * (b + c - 1) := sorry

end log_6_eq_eval_3_5_a_c_log_ratio_log_15_sq_minus_l560_560471


namespace circumcenter_incenter_inequality_l560_560138

variables (A B C H I : Type) [triangle ABC] [orthocenter H ABC] [incenter I ABC]

theorem circumcenter_incenter_inequality 
  (α β γ : ℝ)
  (R : ℝ)
  (A H B : triangle → ℝ)
  (AI BI CI : triangle → ℝ)
  (AI_eq : ∀ (ABC : triangle), AI ABC = 4 * R * sin (β / 2) * sin (γ / 2))
  (BI_eq : ∀ (ABC : triangle), BI ABC = 4 * R * sin (γ / 2) * sin (α / 2))
  (CI_eq : ∀ (ABC : triangle), CI ABC = 4 * R * sin (α / 2) * sin (β / 2))
  (AH_eq : ∀ (ABC : triangle), A H ABC = 2 * R * cos α)
  (BH_eq : ∀ (ABC : triangle), B H ABC = 2 * R * cos β)
  (CH_eq : ∀ (ABC : triangle), C H ABC = 2 * R * cos γ)
  : AH + BH + CH ≥ AI + BI + CI :=
by
  sorry

end circumcenter_incenter_inequality_l560_560138


namespace polynomial_sequence_finite_functions_l560_560045

theorem polynomial_sequence_finite_functions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1) := 
by
  sorry

end polynomial_sequence_finite_functions_l560_560045


namespace sin_cos_valid_n_l560_560354

def sin_cos_identity (x : ℝ) (n : ℤ) : Prop :=
  sin (n * x + 5 * π * n) * cos (6 * (x + 5 * π) / (n + 1)) = sin (n * x) * cos (6 * x / (n + 1))

theorem sin_cos_valid_n (x : ℝ) (n : ℤ) (hn : n ∈ {-31, -16, -11, -7, -6, -4, -3, -2, 0, 1, 2, 4, 5, 9, 14, 29}) : sin_cos_identity x n :=
sorry -- Here should go the proof

end sin_cos_valid_n_l560_560354


namespace range_of_a_l560_560096

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (2 * (x : ℝ) - 7 < 0) ∧ ((x : ℝ) - a > 0) ↔ (x = 3)) →
  (2 ≤ a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l560_560096


namespace cricket_player_average_l560_560601

theorem cricket_player_average (A : ℝ) (h1 : 10 * A + 84 = 11 * (A + 4)) : A = 40 :=
by
  sorry

end cricket_player_average_l560_560601


namespace floor_of_fraction_expression_l560_560645

theorem floor_of_fraction_expression : 
  ( ⌊ (2025^3 / (2023 * 2024)) - (2023^3 / (2024 * 2025)) ⌋ ) = 8 :=
sorry

end floor_of_fraction_expression_l560_560645


namespace average_percent_score_is_77_l560_560154

def numberOfStudents : ℕ := 100

def percentage_counts : List (ℕ × ℕ) :=
[(100, 7), (90, 18), (80, 35), (70, 25), (60, 10), (50, 3), (40, 2)]

noncomputable def average_score (counts : List (ℕ × ℕ)) : ℚ :=
  (counts.foldl (λ acc p => acc + (p.1 * p.2)) 0 : ℚ) / numberOfStudents

theorem average_percent_score_is_77 : average_score percentage_counts = 77 := by
  sorry

end average_percent_score_is_77_l560_560154


namespace expand_expression_l560_560680

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560680


namespace expand_polynomial_eq_l560_560703

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560703


namespace aR_plus_bK_non_neg_definite_RK_non_neg_definite_C_non_neg_definite_fR_non_neg_definite_l560_560905

variables 
  (T : Type)
  (s t : T)
  (u v : T × T)
  (R K : T → T → ℝ)
  (a b : ℝ)
  (f : ℝ → ℝ)

# Check necessary conditions
noncomputable def non_neg_definite (F : T → T → ℝ) : Prop :=
  ∀ (n : ℕ) (t : fin n → T) (α : fin n → ℝ),
    ∑ i j, α i * α j * F (t i) (t j) ≥ 0

# Given conditions
variables (R_non_neg : non_neg_definite R)
          (K_non_neg : non_neg_definite K)
          (a_pos : a > 0)
          (b_pos : b > 0)

-- First goal: aR + bK is non-negative definite
theorem aR_plus_bK_non_neg_definite : non_neg_definite (λ s t, a * R s t + b * K s t) :=
sorry

-- Second goal: RK is non-negative definite
theorem RK_non_neg_definite : non_neg_definite (λ s t, R s t * K s t) :=
sorry

-- Third goal: C(u,v) is non-negative definite
noncomputable def C (u v : T × T) : ℝ :=
  R u.1 v.1 * K u.2 v.2

theorem C_non_neg_definite : non_neg_definite C :=
sorry

-- Fourth goal: f ∘ R is non-negative definite
variables (coeffs : ℕ → ℝ)
          (f_analytic : ∀ x, f x = ∑' n, coeffs n * x ^ n)
          (coeffs_non_neg : ∀ n, coeffs n ≥ 0)

noncomputable def fR (s t : T) : ℝ :=
  f (R s t)

theorem fR_non_neg_definite : non_neg_definite fR :=
sorry

end aR_plus_bK_non_neg_definite_RK_non_neg_definite_C_non_neg_definite_fR_non_neg_definite_l560_560905


namespace intersection_of_sets_M_N_l560_560810

open Set

theorem intersection_of_sets_M_N :
  let M := {x : ℤ | -4 < x ∧ x < 2}
  let N := {x : ℝ | x^2 < 4}
  M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_of_sets_M_N_l560_560810


namespace David_crunches_l560_560248

variable (Zachary_crunches : ℕ)
variable (David_less_crunches : ℕ)

-- provided conditions
def Zachary_crunches := 17
def David_less_crunches := 13

-- statement to be proved
theorem David_crunches : (Zachary_crunches - David_less_crunches) = 4 := by
  sorry

end David_crunches_l560_560248


namespace range_of_a_for_monotonicity_l560_560423

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a_for_monotonicity :
  (∀ x ∈ Ioo (Real.pi / 6) (Real.pi / 3), monotone_on (f a) (Ioo (Real.pi / 6) (Real.pi / 3))) →
  ∀ a ∈ set.Ici (2.0), True := lesorry

end range_of_a_for_monotonicity_l560_560423


namespace lcm_of_40_60_l560_560966

theorem lcm_of_40_60 (a b : ℕ) (h1 : a = 40) (h2 : b = 60) (h3 : a * 3 = b * 2) : Nat.lcm a b = 60 :=
by
  rw [h1, h2]
  apply Nat.lcm_comm
  exact sorry

end lcm_of_40_60_l560_560966


namespace original_profit_percentage_l560_560851

theorem original_profit_percentage (C S : ℝ) 
  (h1 : S - 1.12 * C = 0.5333333333333333 * S) : 
  ((S - C) / C) * 100 = 140 :=
sorry

end original_profit_percentage_l560_560851


namespace probability_other_side_red_l560_560264

theorem probability_other_side_red
  (total_cards : ℕ)
  (black_black_cards : ℕ)
  (black_red_cards : ℕ)
  (red_red_cards : ℕ)
  (red_face : ℕ)
  (red_black_face : ℕ) :
  total_cards = 7 
  ∧ black_black_cards = 2 
  ∧ black_red_cards = 3 
  ∧ red_red_cards = 2 
  ∧ red_face = (red_red_cards * 2 + black_red_cards)
  ∧ black_red_faces = black_red_cards 
  → (4 / 7 : ℚ) = 4 / (red_face) :=
begin
  sorry
end

end probability_other_side_red_l560_560264


namespace gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l560_560587

-- GCD as the greatest common divisor
def GCD (a b : ℕ) : ℕ := Nat.gcd a b

-- LCM as the least common multiple
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- First proof problem in Lean 4
theorem gcd_lcm_relation (a b : ℕ) : GCD a b = (a * b) / (LCM a b) :=
  sorry

-- GCD function extended to three arguments
def GCD3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- LCM function extended to three arguments
def LCM3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- Second proof problem in Lean 4
theorem gcd3_lcm3_relation (a b c : ℕ) : GCD3 a b c = (a * b * c * LCM3 a b c) / (LCM a b * LCM b c * LCM c a) :=
  sorry

-- Third proof problem in Lean 4
theorem lcm3_gcd3_relation (a b c : ℕ) : LCM3 a b c = (a * b * c * GCD3 a b c) / (GCD a b * GCD b c * GCD c a) :=
  sorry

end gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l560_560587


namespace determine_x_l560_560333

def average (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem determine_x (x : ℚ)
  (h : average (2*x + 12) (5*x^2 + 3*x + 1) (3*x + 14) = 6*x^2 + x - 21) :
  x = (5 + real.sqrt 4705) / 26 ∨ x = (5 - real.sqrt 4705) / 26 :=
sorry

end determine_x_l560_560333


namespace sixth_root_of_large_number_l560_560328

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l560_560328


namespace power_function_even_l560_560803

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

theorem power_function_even (m : ℝ) :
  is_even_function (λ x, (m^2 - 3*m - 3) * x^(m / 3)) → m = 4 :=
by
  intro h
  have H1 : (m^2 - 3*m - 3 = 1) := sorry
  have H2 : m = -1 ∨ m = 4 := sorry
  cases H2 with m_neg1 m_4
  · exfalso
    have H3 : ¬is_even_function (λ x, x^(-1/3)) := sorry
    exact H3 (congr_fun h 2)
  · exact m_4

end power_function_even_l560_560803


namespace sixth_root_24414062515625_l560_560324

theorem sixth_root_24414062515625 :
  (∃ (x : ℕ), x^6 = 24414062515625) → (sqrt 6 24414062515625 = 51) :=
by
  -- Applying the condition expressed as sum of binomials
  have h : 24414062515625 = ∑ k in finset.range 7, binom 6 k * (50 ^ (6 - k)),
  sorry
  
  -- Utilize this condition to find the sixth root
  sorry

end sixth_root_24414062515625_l560_560324


namespace minimum_value_N_div_a4_possible_values_a4_l560_560139

noncomputable def lcm_10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) : ℕ := 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a1 a2) a3) a4) a5) a6) a7) a8) a9) a10

theorem minimum_value_N_div_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10) : 
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 := sorry

theorem possible_values_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10)
  (z: 1 ≤ a4 ∧ a4 ≤ 1300) :
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 → a4 = 360 ∨ a4 = 720 ∨ a4 = 1080 := sorry

end minimum_value_N_div_a4_possible_values_a4_l560_560139


namespace largest_k_rooks_l560_560854

noncomputable def rooks_max_k (board_size : ℕ) : ℕ := 
  if board_size = 10 then 16 else 0

theorem largest_k_rooks {k : ℕ} (h : 0 ≤ k ∧ k ≤ 100) :
  k ≤ rooks_max_k 10 := 
sorry

end largest_k_rooks_l560_560854


namespace range_of_a_l560_560839

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 4 ≥ 0) ↔ (2 ≤ a ∧ a ≤ 6) := 
sorry

end range_of_a_l560_560839


namespace equalize_costs_l560_560876

theorem equalize_costs (A B C : ℝ) (h1 : A < B) (h2 : A < C) : 
  ∃ x : ℝ, x = (B + C - 2 * A) / 3 ∧ x > 0 := 
by 
  use (B + C - 2 * A) / 3
  split
  { refl }
  sorry

end equalize_costs_l560_560876


namespace quadratic_inequality_solution_l560_560331

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
by
  sorry

end quadratic_inequality_solution_l560_560331


namespace tap_filling_time_l560_560983

theorem tap_filling_time (T : ℝ) 
  (h_total : (1 / 3) = (1 / T + 1 / 15 + 1 / 6)) : T = 10 := 
sorry

end tap_filling_time_l560_560983


namespace exists_minimal_distance_point_exists_maximal_distance_point_l560_560057

open Real

noncomputable def isPointOnEllipse (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2 / 4) = 1

def isPointOnLine (x y : ℝ) : Prop :=
  x + 2 * y - 10 = 0

def distancePointToLine (x y : ℝ) : ℝ :=
  abs (x + 2 * y - 10) / sqrt (1^2 + 2^2)

theorem exists_minimal_distance_point :
  ∃ (x y : ℝ), isPointOnEllipse x y ∧ distancePointToLine x y = sqrt 5 ∧ x = 9/5 ∧ y = 8/5 := 
sorry

theorem exists_maximal_distance_point :
  ∃ (x y : ℝ), isPointOnEllipse x y ∧ distancePointToLine x y = 3 * sqrt 5 ∧ x = -9/5 ∧ y = -8/5 := 
sorry

end exists_minimal_distance_point_exists_maximal_distance_point_l560_560057


namespace Tim_younger_than_Jenny_l560_560986

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l560_560986


namespace symmetric_line_l560_560201

variable (x : ℝ)

def reflection_over_x_axis (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, -f(x)

def line1 (x : ℝ) : ℝ := 2 * x + 1

theorem symmetric_line :
  reflection_over_x_axis line1 x = -2 * x - 1 :=
by
  sorry

end symmetric_line_l560_560201


namespace score_calculation_l560_560174

theorem score_calculation (N : ℕ) (C : ℕ) (hN: 1 ≤ N ∧ N ≤ 20) (hC: 1 ≤ C) : 
  ∃ (score: ℕ), score = Nat.floor (N / C) :=
by sorry

end score_calculation_l560_560174


namespace expand_product_l560_560693

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560693


namespace sequence_does_not_contain_5_or_11_l560_560458

theorem sequence_does_not_contain_5_or_11 :
  let a : ℕ → ℕ := λ n, if n = 0 then 2 else Nat.greatest_prime_divisor (Finset.range n).prod.succ
  ∀ n, a n ≠ 5 ∧ a n ≠ 11 := 
sorry

end sequence_does_not_contain_5_or_11_l560_560458


namespace house_rent_fraction_l560_560606

-- Define the constants and the problem
def salary : ℝ := 140000
def left_amount : ℝ := 14000
def food_fraction : ℝ := 1 / 5
def clothes_fraction : ℝ := 3 / 5
def rent (S H: ℝ) := H * S

-- Define the main problem
theorem house_rent_fraction (H : ℝ) : 
  salary - (food_fraction * salary + clothes_fraction * salary + rent salary H) = left_amount ->
  H = 1 / 10 :=
by
  sorry

end house_rent_fraction_l560_560606


namespace tetrahedron_range_of_a_l560_560039

theorem tetrahedron_range_of_a 
  (a : ℝ) 
  (h1 : AB = 2 * a) 
  (h2 : CD = 2 * a)
  (h3 : AC = bd := BD := BC := AD := sqrt 10) 
  : 0 < a ∧ a < sqrt 5 :=
sorry

end tetrahedron_range_of_a_l560_560039


namespace h_of_3_l560_560881

def f (x : ℝ) : ℝ := 2 * x + 9
def g (x : ℝ) : ℝ := (f x) ^ (1 / 3 : ℝ) - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_3 : h 3 = 2 * (15 ^ (1 / 3 : ℝ)) + 3 :=
by
  sorry

end h_of_3_l560_560881


namespace farmer_bill_cows_l560_560012

theorem farmer_bill_cows (c r : ℕ) (d : ℕ := 600) (h : d + c + r = 1000)
    (safe_ducks : ∀ d_i : ℕ, d_i < d → (∃ ci ∈ range c ∪ range (c + r), ci = d_i + 1 ∨ ci = d_i - 1)
    ∨ ((d_i > 0) ∧ (d_i < d - 1) ∧ ∃ ri ∈ range r, ri = d_i ∧ ri + 1 = d_i - 1)) : c ≥ 201 := 
sorry

end farmer_bill_cows_l560_560012


namespace maximum_ab_l560_560132

theorem maximum_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3*a + 8*b = 48) : ab ≤ 24 :=
by
  sorry

end maximum_ab_l560_560132


namespace largest_integer_x_l560_560736

theorem largest_integer_x (x : ℤ) :
  (x ^ 2 - 11 * x + 28 < 0) → x ≤ 6 := sorry

end largest_integer_x_l560_560736


namespace flat_gain_loss_percentage_l560_560277

noncomputable def percentage_gain_loss : ℝ :=
  let p : ℝ := 4.000000000000007 / 1024912 in
  p * 100

theorem flat_gain_loss_percentage :
  percentage_gain_loss = 0.00039025 :=
by
  have h_p : (4.000000000000007 / 1024912) * 100 = 0.00039025 := sorry
  exact h_p

end flat_gain_loss_percentage_l560_560277


namespace parallel_lines_distance_l560_560426

open Real

theorem parallel_lines_distance (m : ℝ) :
  (∀ x y : ℝ, 3 * x + y - 3 = 0 → 6 * x + m * y + 4 = 0) → 
  m = 2 ∧ (distance_between_lines := λ (a b c d : ℝ), abs (c - d) / sqrt (a^2 + b^2),
  distance_between_lines (6 : ℝ) (2 : ℝ) (-6 : ℝ) (4 : ℝ) = sqrt 10 / 2) :=
sorry

end parallel_lines_distance_l560_560426


namespace expand_expression_l560_560673

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560673


namespace smallest_n_to_make_183_divisible_by_11_l560_560419

theorem smallest_n_to_make_183_divisible_by_11 : ∃ n : ℕ, 183 + n % 11 = 0 ∧ n = 4 :=
by
  have h1 : 183 % 11 = 7 := 
    sorry
  let n := 11 - (183 % 11)
  have h2 : 183 + n % 11 = 0 :=
    sorry
  exact ⟨n, h2, sorry⟩

end smallest_n_to_make_183_divisible_by_11_l560_560419


namespace value_of_2_pow_neg_x_l560_560088

theorem value_of_2_pow_neg_x (x : ℝ) (h : 64^5 = 32^x) : 2^(-x) = 1 / 64 :=
by
  sorry

end value_of_2_pow_neg_x_l560_560088


namespace min_weights_needed_l560_560576
open Nat

-- Define the conditions
def watermelon_weights : Set ℕ := {w | w ∈ range 21 ∧ w ≠ 0}

-- Predicate to check if a set of weights can cover all watermelon weights
def can_weigh_all (weights : Set ℕ) : Prop :=
  ∀ w ∈ watermelon_weights, ∃ a b ∈ weights, w = a + b ∨ w = a

-- The main theorem stating the minimum number of different weights needed
theorem min_weights_needed : ∃ S : Set ℕ, S.card = 6 ∧ can_weigh_all S :=
sorry

end min_weights_needed_l560_560576


namespace arrangementsCount_l560_560295

def triangularArraySum (eleventhRow : Fin 11 → ℕ) : ℕ :=
  ∑ k in Finset.range 11, Nat.choose 10 k * eleventhRow k

noncomputable def validConfigurations : ℕ :=
  let validCombos := {eleventhRow : Fin 11 → ℕ // ∀ k, eleventhRow k = 0 ∨ eleventhRow k = 1}
  let countMod5 (eleventhRow : Fin 11 → ℕ) : ℕ :=
    eleventhRow 0 + eleventhRow 5 + eleventhRow 10
  let validRows := {eleventhRow ∈ validCombos | countMod5 eleventhRow % 5 = 0}
  validRows.card * 2^8

theorem arrangementsCount : validConfigurations = 1024 := by
  sorry

end arrangementsCount_l560_560295


namespace residue_n_mod_17_l560_560178

noncomputable def satisfies_conditions (m n k : ℕ) : Prop :=
  m^2 + 1 = 2 * n^2 ∧ 2 * m^2 + 1 = 11 * k^2 

theorem residue_n_mod_17 (m n k : ℕ) (h : satisfies_conditions m n k) : n % 17 = 5 :=
  sorry

end residue_n_mod_17_l560_560178


namespace calculate_expression_l560_560638

theorem calculate_expression :
  (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 :=
by
  -- Since all necessary conditions are listed in the problem statement, we honor this structure
  -- The following steps are required logically but are not presently necessary for detailed proof means.
  sorry

end calculate_expression_l560_560638


namespace find_a_l560_560652

variables {f g : ℝ → ℝ} {a : ℝ}

-- Definitions and Conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def condition (f g : ℝ → ℝ) (a : ℝ) := ∀ x, f x + g x = x^2 + ax + a

-- Propositions p and q
def p (f : ℝ → ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≥ 1
def q (g : ℝ → ℝ) := ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ g x ≤ -1

-- Proof statement
theorem find_a (h_odd : is_odd f) (h_even : is_even g)
  (h_cond : condition f g a) (h_pq : p f ∨ q g) : 
  (1 ≤ a) ∨ (a ≤ -1) := by sorry

end find_a_l560_560652


namespace number_of_valid_n_l560_560125

theorem number_of_valid_n (m : ℕ) (h : m = 2007 ^ 2008) : 
  { n : ℕ | n < m ∧ m ∣ n * (2 * n + 1) * (5 * n + 2) }.to_finset.card = 8 :=
by
  sorry

end number_of_valid_n_l560_560125


namespace circumscribed_sphere_does_not_always_exist_l560_560762

/-- Given a polyhedron where all edges are equal in length and each edge touches a certain sphere,
does a circumscribed sphere always exist around such a polyhedron? -/
theorem circumscribed_sphere_does_not_always_exist
  (P : Type) [polyhedron P]
  (sphere : Type) [metric_space sphere]
  (edges_equal_length : ∀ edge1 edge2 : P.edges, length edge1 = length edge2)
  (edges_touch_sphere : ∀ edge : P.edges, ∃ (s : sphere), touch_edge_sphere edge s) :
  ¬ ∀ (circumsphere : Type) [metric_space circumsphere] (centroid : circumsphere),
    (∀ vertex : P.vertices, distance vertex centroid = radius circumsphere) := 
sorry

end circumscribed_sphere_does_not_always_exist_l560_560762


namespace number_of_mappings_l560_560960

theorem number_of_mappings (A : set ℕ) (B : set ℕ) (hA : A = {a, b}) (hB : B = {0, 1}) :
  fintype.card (A → B) = 4 := sorry

end number_of_mappings_l560_560960


namespace compute_floor_expression_l560_560643

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end compute_floor_expression_l560_560643


namespace cyclic_quadrilateral_midpoints_collinear_l560_560932

open_locale real

noncomputable theory

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

structure Quadrilateral :=
(A B C D : Point)
(cyclic : ∃ O : Point, distance O A = distance O B ∧ distance O A = distance O C ∧ distance O A = distance O D)

theorem cyclic_quadrilateral_midpoints_collinear
  (Q : Quadrilateral)
  (O : Point)
  (hO : ∃ O, distance O Q.A = distance O Q.B ∧  distance O Q.A = distance O Q.C ∧ distance O Q.A = distance O Q.D):
  let M := midpoint Q.A Q.C,
      N := midpoint Q.B Q.D
  in collinear [O, M, N] :=
sorry

end cyclic_quadrilateral_midpoints_collinear_l560_560932


namespace trajectory_of_P_is_parabola_circumcircle_of_ABD_l560_560377

noncomputable def point_E := (1 : ℝ, 0 : ℝ)
noncomputable def point_K := (-1 : ℝ, 0 : ℝ)

def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PE := (1 - P.1, -P.2)
  let PK := (-1 - P.1, -P.2)
  let EK := (-2, 0)
  let PE_len := real.sqrt ((1 - P.1)^2 + P.2^2)
  let PK_dot_EK := (-1 - P.1) * (-2)
  PE_len * PE_len = PK_dot_EK

theorem trajectory_of_P_is_parabola {P : ℝ × ℝ} (h : satisfies_condition P) :
  P.2^2 = 4 * P.1 := sorry

theorem circumcircle_of_ABD (A B D : ℝ × ℝ)
  (hA : A.1^2 = 4 * A.1) (hB : B.1^2 = 4 * B.1) (hD : D = (A.1, -A.2))
  (hEA_EB : let EA := (A.1 - point_E.1, A.2); let EB := (B.1 - point_E.1, B.2);
             EA.1 * EB.1 + EA.2 * EB.2 = -8) :
  ∃ (x y : ℝ), (x - 9)^2 + y^2 = 40 := sorry

end trajectory_of_P_is_parabola_circumcircle_of_ABD_l560_560377


namespace maximize_area_of_pasture_l560_560284

theorem maximize_area_of_pasture
  (y : ℝ)
  (h : 0 ≤ y)
  (h_fence : 2 * y ≤ 250)
  (A : ℝ := y * (250 - 2 * y)) : 
  ∃ side_parallel_barn : ℝ, side_parallel_barn = 125 ∧ 
    (∀ z : ℝ, 0 ≤ z ∧ 2 * z ≤ 250 → (z * (250 - 2 * z)) ≤ A) :=
begin
  use 125,
  split,
  {reflexivity},
  {intros z hz1 hz2,
   have : -2 * (z^2) + 250 * z ≤ -2 * ((62.5)^2) + 250 * 62.5,
   {apply sub_nonpos.2,
    rw [mul_square_mul_eq, mul_square_mul_eq],
    linarith [hz2, hz1]},
   exact this,}
end

end maximize_area_of_pasture_l560_560284


namespace vector_relationship_l560_560781

open_locale vector_space

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C P : V}

-- Given conditions
def condition1 : Prop := sorry
def condition2 (B C P : V) : Prop := vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({B} : set V) = 2 • (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({C} : set V))

-- The theorem statement based on the given conditions
theorem vector_relationship (h1 : condition1) (h2 : condition2 B C P) : 
  (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({A} : set V)) = 
  (-1/2 : ℝ) • (vector_span ℝ ({B} : set V) -ᵥ vector_span ℝ ({A} : set V)) + (3/2 : ℝ) • (vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({A} : set V)) :=
sorry

end vector_relationship_l560_560781


namespace expand_product_l560_560694

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560694


namespace probability_at_least_one_B_l560_560220

theorem probability_at_least_one_B (A B : Type) [Fintype A] [Fintype B]
  (total_questions : ℕ) (type_A_questions : ℕ) (type_B_questions : ℕ) :
  total_questions = 5 →
  type_A_questions = 2 →
  type_B_questions = 3 →
  ∀ (select : Finset (ℕ × ℕ)), select.card = 2 →
  ∃ m n, m + n = 9 ∧ n = 1 →
  (9 / 10) = 9 / 10 :=
by {
  intros,
  sorry
}

end probability_at_least_one_B_l560_560220


namespace solve_equation_l560_560511

theorem solve_equation : ∃ x : ℝ, (x = 6) ∧ ((x + 6) / (x - 3) = 4) :=
by
  use 6
  split
  . -- part 1: show that x = 6
    refl
  . -- part 2: show that (x + 6) / (x - 3) = 4 when x = 6
    sorry

end solve_equation_l560_560511


namespace initial_distance_between_projectiles_l560_560228

-- Definitions for each of the conditions
def speed1 : ℝ := 470
def speed2 : ℝ := 500
def time_minutes : ℝ := 90
def time_hours : ℝ := time_minutes / 60
def relative_speed : ℝ := speed1 + speed2
def distance : ℝ := relative_speed * time_hours

-- The theorem to prove the initial distance between the two projectiles
theorem initial_distance_between_projectiles : distance = 1455 := by
  unfold distance relative_speed time_hours time_minutes speed1 speed2
  sorry

end initial_distance_between_projectiles_l560_560228


namespace right_angle_value_l560_560547

theorem right_angle_value (x : ℝ) (left_angle := 2 * x) (top_angle := 70) :
    x + left_angle + top_angle = 180 → left_angle = 2 * x → top_angle = 70 →
    (right_angle : ℝ) = 180 - (left_angle + top_angle) :=
by
  intro h1 h2 h3
  -- h1: x + 2 * x + 70 = 180
  -- h2: left_angle = 2 * x
  -- h3: top_angle = 70
  let right_angle := 180 - (left_angle + top_angle)
  ring
  have h4 : 3 * x + 70 = 180 := by
    rw [←h1]
  have h5 : 3 * x = 110 := by
    linarith
  have h6 : x = 110 / 3 := by
    field_simp at h5
    exact h5
  have left_angle : ℝ := 2 * (110 / 3)
  have right_angle : ℝ := 180 - (70 + left_angle)
  exact sorry

end right_angle_value_l560_560547


namespace fib_add_l560_560011

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem fib_add (n m : ℕ) :
  fib (m + n) = fib (m-1) * fib n + fib m * fib (n+1) :=
sorry

end fib_add_l560_560011


namespace algebraic_expression_no_linear_term_l560_560837

theorem algebraic_expression_no_linear_term (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 1/2) = x^2 - a/2 ↔ a = 1/2) :=
by
  sorry

end algebraic_expression_no_linear_term_l560_560837


namespace evaluate_complex_expression_l560_560009

theorem evaluate_complex_expression :
  sqrt ((5 - 3 * sqrt 2) ^ 2) + sqrt ((5 + 3 * sqrt 2) ^ 2) + 5 = 15 := by
  sorry

end evaluate_complex_expression_l560_560009


namespace min_value_of_expression_l560_560791

theorem min_value_of_expression (a c : ℝ) (ha : a > 0) (hc : c > 0) (hac : a * c = 4) :
  \(\frac{1}{c} + \frac{9}{a} \geq 3\) :=
by
  sorry

end min_value_of_expression_l560_560791


namespace value_of_y_l560_560832

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end value_of_y_l560_560832


namespace probability_of_selection_l560_560362

def numbers : Set ℤ := {-1, 0, 1, 3, 4}

def hyperbola_condition (a : ℤ) : Prop := 7 - 3 * a > 0

def inequalities_no_solution (a : ℤ) : Prop :=
(∀ x : ℤ, ¬ (2 * x + 3 > 9 ∧ x < a))

theorem probability_of_selection :
  let favorable_numbers := {a : ℤ | a ∈ numbers ∧ hyperbola_condition a ∧ inequalities_no_solution a} in
  (favorable_numbers.card : ℚ) / (numbers.card : ℚ) = 3 / 5 := by
sorry

end probability_of_selection_l560_560362


namespace number_of_ordered_pairs_l560_560590

theorem number_of_ordered_pairs (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / 24)) : 
  ∃ n : ℕ, n = 41 :=
by
  sorry

end number_of_ordered_pairs_l560_560590


namespace math_problem_l560_560136

theorem math_problem (m : ℤ) (h₀ : 0 ≤ m) (h₁ : m < 37) (h₂ : 4 * m % 37 = 1) : ((3^m % 37)^2 - 3) % 37 = 19 := by 
  sorry

end math_problem_l560_560136


namespace prime_fraction_sum_eq_natural_l560_560496

theorem prime_fraction_sum_eq_natural (p q n : ℕ) (hp : p.prime) (hq : q.prime) :
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / (p * q) = (1 : ℚ) / n ->
  (p = 2 ∧ q = 3 ∧ n = 1) ∨ (p = 3 ∧ q = 2 ∧ n = 1) :=
by
  sorry

end prime_fraction_sum_eq_natural_l560_560496


namespace joan_payment_l560_560450

theorem joan_payment (cat_toy_cost cage_cost change_received : ℝ) 
  (h1 : cat_toy_cost = 8.77) 
  (h2 : cage_cost = 10.97) 
  (h3 : change_received = 0.26) : 
  cat_toy_cost + cage_cost - change_received = 19.48 := 
by 
  sorry

end joan_payment_l560_560450


namespace log_inverse_l560_560091

theorem log_inverse :
  (∃ x : ℝ, log 16 (x - 6) = 1 / 4) → (1 / log x 2 = 3) :=
by
  intro hx
  cases hx with x hx
  -- Proof that x = 8 omitted
  sorry

end log_inverse_l560_560091


namespace count_three_digit_numbers_l560_560819

theorem count_three_digit_numbers :
  let numbers := [100 * x + 10 * y + 4 | x in finset.range 10, y in finset.range 10]
  let filtered_numbers := numbers.filter (λ n, (n % 10 = 4) ∧ ((n / 10) % 10 + n / 100 + 4 = 15) ∧ (n % 7 ≠ 0))
  filtered_numbers.card = 7 := 
by
  sorry

end count_three_digit_numbers_l560_560819


namespace expand_expression_l560_560725

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560725


namespace sufficient_condition_l560_560439

theorem sufficient_condition (a b : ℝ) (h : b > a ∧ a > 0) : (a + 2) / (b + 2) > a / b :=
by sorry

end sufficient_condition_l560_560439


namespace find_pyramid_volume_l560_560519

-- Define the parameters
variables {a b : ℝ}

-- Define the conditions
def is_equilateral_triangle (ABC : Triangle) (a : ℝ) : Prop :=
  ABC.is_equilateral ∧ ABC.side_length = a

def lateral_faces_congruent (S A B C : Point) : Prop :=
  True -- Simplification, need congruence condition on lateral faces

-- Define the volume formulas under different conditions
def volume_case1 (a b : ℝ) : ℝ := (a^2 / 12) * real.sqrt(3 * b^2 - a^2)
def volume_case2a (a b : ℝ) : ℝ := (a^2 / 12) * real.sqrt(3 * b^2 - a^2)
def volume_case2b (a b : ℝ) : ℝ := (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - a^2)
def volume_case3a (a b : ℝ) : ℝ := (a^2 / 12) * real.sqrt(3 * b^2 - a^2)
def volume_case3b (a b : ℝ) : ℝ := (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - a^2)
def volume_case3c (a b : ℝ) : ℝ := (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - 3 * a^2)

-- State the theorem
theorem find_pyramid_volume (a b : ℝ)
  (h1 : is_equilateral_triangle ABC a)
  (h2 : lateral_faces_congruent S A B C)
  (h3 : (a / real.sqrt 3) < b ∧ b ≤ a)
  (h4 : a < b ∧ b ≤ a * real.sqrt 3)
  (h5 : b > a * real.sqrt 3) :
  -- Case 1
  (h3 → volume_case1 a b = (a^2 / 12) * real.sqrt(3 * b^2 - a^2)) ∧
  -- Case 2
  (h4 → volume_case2a a b = (a^2 / 12) * real.sqrt(3 * b^2 - a^2) ∧
         volume_case2b a b = (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - a^2)) ∧
  -- Case 3
  (h5 → volume_case3a a b = (a^2 / 12) * real.sqrt(3 * b^2 - a^2) ∧
        volume_case3b a b = (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - a^2) ∧
        volume_case3c a b = (a^2 * real.sqrt(3) / 12) * real.sqrt(b^2 - 3 * a^2)) :=
by sorry

end find_pyramid_volume_l560_560519


namespace num_solutions_of_equation_l560_560411

theorem num_solutions_of_equation : 
  {x : ℝ | (x^2 - 10)^2 = 81}.finite.toFinset.card = 4 :=
by sorry

end num_solutions_of_equation_l560_560411


namespace part1_real_part2_complex_part3_pure_imaginary_part4_fourth_quadrant_l560_560024

variable (m : ℝ)
def z (m : ℝ) : ℂ := complex.mk (m^2 - 1) (m^2 - m - 2)

theorem part1_real (h1 : z m = complex.mk (m^2 - 1) 0) : m = -1 ∨ m = 2 :=
by sorry

theorem part2_complex (h2 : z m ≠ complex.mk (m^2 - 1) 0) : m ≠ -1 ∧ m ≠ 2 :=
by sorry

theorem part3_pure_imaginary (h3 : z m = complex.mk 0 (m^2 - m - 2)) : m = 1 :=
by sorry

theorem part4_fourth_quadrant (h4 : ( m > 1 ∧ m < 2 )) : 
  complex.re (z m) > 0 ∧ complex.im (z m) < 0 :=
by sorry

end part1_real_part2_complex_part3_pure_imaginary_part4_fourth_quadrant_l560_560024


namespace race_cars_count_l560_560850

theorem race_cars_count:
  (1 / 7 + 1 / 3 + 1 / 5 = 0.6761904761904762) -> 
  (∀ N : ℕ, (1 / N = 1 / 7 ∨ 1 / N = 1 / 3 ∨ 1 / N = 1 / 5)) -> 
  (1 / 105 = 0.6761904761904762) :=
by
  intro h_sum_probs h_indiv_probs
  sorry

end race_cars_count_l560_560850


namespace sphere_volume_in_cone_l560_560614

def diameter_of_cone_base : ℝ := 24
def radius_of_sphere (d : ℝ) : ℝ := d / 4  -- Derived from d / 2 / 2
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem sphere_volume_in_cone 
  (d : ℝ) (h_d : d = diameter_of_cone_base) :
  volume_of_sphere (radius_of_sphere d) = 288 * Real.pi :=
by
  intros
  rw [h_d]
  have r_def : radius_of_sphere diameter_of_cone_base = 6 := by
    unfold radius_of_sphere
    norm_num
  rw r_def
  unfold volume_of_sphere
  norm_num
  sorry

end sphere_volume_in_cone_l560_560614


namespace function_value_at_minus_two_l560_560064

theorem function_value_at_minus_two {f : ℝ → ℝ} (h : ∀ x : ℝ, x ≠ 0 → f (1/x) + (1/x) * f (-x) = 2 * x) : f (-2) = 7 / 2 :=
sorry

end function_value_at_minus_two_l560_560064


namespace min_value_frac_l560_560759

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (c : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → c ≤ 8 / x + 2 / y) ∧ c = 18 :=
sorry

end min_value_frac_l560_560759


namespace vector_relationship_proof_l560_560775

variables {P A B C : Type} [inner_product_space ℝ P]
variables {V : P → P → ℝ} (AB AC AP BC CP : V)

-- Given conditions as hypotheses
hypothesis h1 : BC = 2 * CP

-- Known vector relationships
hypothesis h2 : BC = AC - AB
hypothesis h3 : CP = AP - AC

-- Prove the given vector equation
theorem vector_relationship_proof :
  AP = - (1/2) * AB + (3/2) * AC :=
by {
  -- Implementation of the proof steps here (skipped)
  sorry
}

end vector_relationship_proof_l560_560775


namespace difference_sum_even_odd_l560_560232

theorem difference_sum_even_odd (n : ℕ) (h : n = 3010) :
  let S_even := (n * (2 + (2 * n))) / 2
      S_odd := (n * (1 + (2 * n - 1))) / 2
  in S_even - S_odd = n := 
  by
    sorry

end difference_sum_even_odd_l560_560232


namespace floor_ceil_sum_l560_560975

theorem floor_ceil_sum (x : ℝ) (h : Int.floor x + Int.ceil x = 7) : x ∈ { x : ℝ | 3 < x ∧ x < 4 } ∪ {3.5} :=
sorry

end floor_ceil_sum_l560_560975


namespace particles_probability_computation_l560_560429

theorem particles_probability_computation : 
  let L0 := 32
  let R0 := 68
  let N := 100
  let a := 1
  let b := 2
  let P_all_on_left := (a:ℚ) / b
  100 * a + b = 102 := by
  sorry

end particles_probability_computation_l560_560429


namespace angle_between_a_b_is_120_degrees_l560_560049

noncomputable def dot_product_vec (v1 v2 : EuclideanSpace ℝ (Fin 2)) :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude_vec (v : EuclideanSpace ℝ (Fin 2)) :=
  Real.sqrt (dot_product_vec v v)

noncomputable def angle_between_vec (v1 v2 : EuclideanSpace ℝ (Fin 2)) :=
  Real.arccos ((dot_product_vec v1 v2) / ((magnitude_vec v1) * (magnitude_vec v2)))

def e1 := EuclideanSpace.from_orthogonal_system_bases (Fin 2) [1, 0]
def e2 := EuclideanSpace.from_orthogonal_system_bases (Fin 2) [Real.sqrt (1 / 2), Real.sqrt (1 / 2)]

def a := 2 • e1 + e2
def b := -3 • e1 + 2 • e2

theorem angle_between_a_b_is_120_degrees :
  angle_between_vec a b = Real.pi * 2 / 3 :=
sorry

end angle_between_a_b_is_120_degrees_l560_560049


namespace six_diggers_five_hours_l560_560557

theorem six_diggers_five_hours (holes_per_hour_per_digger : ℝ) 
  (h1 : 3 * holes_per_hour_per_digger * 3 = 3) :
  6 * (holes_per_hour_per_digger) * 5 = 10 :=
by
  -- The proof will go here, but we only need to state the theorem
  sorry

end six_diggers_five_hours_l560_560557


namespace probability_of_valid_pairs_l560_560656

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ (∀ x: Nat, x ∣ n → x = 1 ∨ x = n)

def is_valid_sum (p1 p2 : Nat) : Prop :=
  is_prime (p1 + p2) ∧ p1 + p2 < 30

def valid_pairs_count (l : List Nat) : Nat :=
  l.product l |>.filter (λ (p : Nat × Nat), p.1 != p.2 ∧ is_valid_sum p.1 p.2) |>.length

def total_pairs_count (n : Nat) : Nat :=
  Nat.choose n 2

theorem probability_of_valid_pairs :
  let total_pairs := total_pairs_count first_ten_primes.length;
  let valid_pairs := valid_pairs_count first_ten_primes;
  valid_pairs / total_pairs = 4 / 45 :=
by
  sorry

end probability_of_valid_pairs_l560_560656


namespace work_required_to_lift_satellite_l560_560334

noncomputable def satellite_lifting_work (m H R3 g : ℝ) : ℝ :=
  m * g * R3^2 * ((1 / R3) - (1 / (R3 + H)))

theorem work_required_to_lift_satellite :
  satellite_lifting_work (7.0 * 10^3) (200 * 10^3) (6380 * 10^3) 10 = 13574468085 :=
by sorry

end work_required_to_lift_satellite_l560_560334


namespace number_of_homologous_functions_l560_560844

def homologous_functions : Set ℕ := {0, 1, 4}

def valid_domains : Set (Set ℤ) :=
  { {0, 1, 2}, {0, 1, -2}, {0, -1, 2}, {0, -1, -2},
    {0, 1, -2, 2}, {0, -1, -2, 2}, {0, 1, -1, -2},
    {0, 1, -1, 2, -2} }

theorem number_of_homologous_functions :
  (∃ f : ℤ → ℕ, ∀ D ∈ valid_domains, f = λ x, x^2 ∧ (∀ y ∈ f '' D, y ∈ homologous_functions) ∧ (f '' D).range = homologous_functions) →
  valid_domains.card = 8 :=
by
  sorry

end number_of_homologous_functions_l560_560844


namespace triangle_ratio_l560_560898

theorem triangle_ratio 
  (A B C D : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (h1 : is_acute (triangle A B C))
  (h2 : ∃ (D : A), angle A D B = angle A C B + 90)
  (h3 : dist A C * dist B D = dist A D * dist B C) :
  dist A B * dist C D / (dist A C * dist B D) = sqrt 2 :=
  sorry

end triangle_ratio_l560_560898


namespace value_of_x_pow_12_l560_560825

theorem value_of_x_pow_12 (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^12 = 439 := sorry

end value_of_x_pow_12_l560_560825


namespace max_distance_right_triangle_l560_560536

theorem max_distance_right_triangle (a b : ℝ) 
  (h1: ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
    (a * A.1 + 2 * b * A.2 = 1) ∧ (a * B.1 + 2 * b * B.2 = 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0,0) ∧ (A.1 * B.1 + A.2 * B.2 = 0)): 
  ∃ (d : ℝ), d = (Real.sqrt (a^2 + b^2)) ∧ d ≤ Real.sqrt 2 :=
sorry

end max_distance_right_triangle_l560_560536


namespace calculate_expression_l560_560310

theorem calculate_expression : abs (-2) - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end calculate_expression_l560_560310


namespace two_dice_probability_l560_560149

def prob_between_10_and_30 : ℚ := 5 / 9

theorem two_dice_probability :
  (∑ i in Finset.range (6 + 1), ∑ j in Finset.range (6 + 1),
    if ((i * 10 + j) ≤ 30 ∧ (i * 10 + j) ≥ 10 ∨ (j * 10 + i) ≤ 30 ∧ (j * 10 + i) ≥ 10) then 1 else 0) / 36 = prob_between_10_and_30 :=
sorry

end two_dice_probability_l560_560149


namespace sqrt_expression_sum_eq_two_l560_560371

theorem sqrt_expression_sum_eq_two (x1 x2 : ℝ) 
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ 0)
  (h3 : x1 + x2 = 2) :
  sqrt (x1 + sqrt (x1^2 - x2^2)) + sqrt (x1 - sqrt (x1^2 - x2^2)) = 2 :=
by
  sorry

end sqrt_expression_sum_eq_two_l560_560371


namespace am_gm_inequality_l560_560142

open Real

theorem am_gm_inequality (
    a b c d e f : ℝ
) (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c)
  (h_nonneg_d : 0 ≤ d)
  (h_nonneg_e : 0 ≤ e)
  (h_nonneg_f : 0 ≤ f)
  (h_cond_ab : a + b ≤ e)
  (h_cond_cd : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) := 
  by sorry

end am_gm_inequality_l560_560142


namespace triangle_perimeter_l560_560443

noncomputable def perimeter_of_triangle
  (AB AC : ℝ)
  (M : ℝ)
  (AB_eq : AB = 8)
  (AC_eq : AC = 17)
  (AM_eq : M = 12) :
  ℝ :=
  AB + AC + sqrt (130)

theorem triangle_perimeter :
  perimeter_of_triangle 8 17 12 8 17 12 =
  25 + sqrt (130) :=
by
  rw [perimeter_of_triangle, sqrt, add, add, add, add, add, add, add, add, add, add, add, add, add, add];
  sorry

end triangle_perimeter_l560_560443


namespace matrix_pow_eq_l560_560877

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![!![0, 1, 0], !![0, 0, 1], !![1, 0, 0]]

theorem matrix_pow_eq :
  B ^ 100 = B :=
by
  sorry

end matrix_pow_eq_l560_560877


namespace largest_sum_of_ABCD_l560_560337

theorem largest_sum_of_ABCD :
  ∃ (A B C D : ℕ), 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100 ∧ 10 ≤ D ∧ D < 100 ∧
  B = 3 * C ∧ D = 2 * B - C ∧ A = B + D ∧ A + B + C + D = 204 :=
by
  sorry

end largest_sum_of_ABCD_l560_560337


namespace no_unfenced_area_l560_560479

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end no_unfenced_area_l560_560479


namespace sufficient_but_not_necessary_condition_l560_560589

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (1 ≤ x ∧ x ≤ 4) ↔ (1 ≤ x^2 ∧ x^2 ≤ 16) :=
by
  sorry

end sufficient_but_not_necessary_condition_l560_560589


namespace calc_f_f_f_neg1_l560_560636

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 else if x = 0 then real.pi else 0

theorem calc_f_f_f_neg1 : f (f (f (-1))) = real.pi^2 :=
by
  -- sorry is used to skip the proof
  sorry

end calc_f_f_f_neg1_l560_560636


namespace yn_arithmetic_sequence_xn_general_term_and_constant_difference_isosceles_right_triangle_l560_560038

noncomputable theory
open_locale big_operators

-- Given sequences B_n and A_n
def B_n (n : ℕ) : ℝ × ℝ := (n, n / 4)
def A_n (x_n : ℕ → ℝ) (n : ℕ) : ℝ × ℝ := (x_n n, 0)

-- Initial condition
axiom ha : ∀ (n : ℕ), 0 < n → 0 < 1 → x_1 = a

-- Condition for isosceles triangles A_nB_nA_{n+1}
axiom isosceles_triangle : ∀ (n : ℕ) (x_n : ℕ → ℝ), 0 < n →
  (let B := B_n n, A := A_n x_n n, A_next := A_n x_n (n + 1) in
   (A.fst - B.fst)^2 + B.snd^2 = (A_next.fst - B.fst)^2 + B.snd^2)

-- Prove that the series {y_n} is arithmetic
theorem yn_arithmetic_sequence : ∀ (n : ℕ), 0 < n → B_n (n + 1).snd - B_n n.snd = 1/4 :=
sorry

-- Prove that the difference x_{n+2} - x_n is constant and find the general term for {x_n}
theorem xn_general_term_and_constant_difference (x_n : ℕ → ℝ) : 
  ∀ (n : ℕ), 0 < n → x_n (n + 2) - x_n n = 2 :=
sorry

-- Determine if there can be a right triangle among the isosceles triangles A_nB_nA_{n+1} and find the values of a
theorem isosceles_right_triangle (x_n : ℕ → ℝ) (a : ℝ) : 
  ∃ n, 0 < n → 0 < a ∧ a < 1 → 
  let B := B_n n, A := A_n x_n n, A_next := A_n x_n (n + 1) in
  ((A.fst - A_next.fst)^2 = 2 * B.snd^2) → 
  a ∈ {3/4, 1/4, 1/2} :=
sorry

end yn_arithmetic_sequence_xn_general_term_and_constant_difference_isosceles_right_triangle_l560_560038


namespace incorrect_statement_C_l560_560824

open Classical

noncomputable theory

variables (Line Plane : Type) (m n : Line) (α β : Plane)

-- Conditions
axiom diff_lines : m ≠ n
axiom diff_planes : α ≠ β
axiom perp_iff_parallel (l1 l2 : Line) (p : Plane) : l1 ⊥ p → l2 ⊥ p → l1 ∥ l2
axiom subset_parallel (l : Line) (p1 p2 : Plane) : l ⊆ p1 → p1 ∥ p2 → l ∥ p2
axiom parallel_extend (l1 l2 : Line) (p : Plane) : l1 ∥ l2 → l1 ∥ p → ¬ (l2 ⊆ p) → l2 ∥ p

-- Statement C to be proven incorrect
theorem incorrect_statement_C : (m ∥ α) → (n ∥ α) → ¬ (m ∥ n) := sorry

end incorrect_statement_C_l560_560824


namespace expand_expression_l560_560715

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560715


namespace triangle_bisector_intersection_ratio_l560_560445

theorem triangle_bisector_intersection_ratio
  {A B C D E P : Type}
  [Real A] [Real B] [Real C]
  (angle_bisectors_intersect : ∀ (A B C D E P : Real), A = D ∧ B = E ∧ D ≠ P → ∃ P, intersection_of_bisectors)
  (AB AC BC : Real) :
  AB = 7 ∧ AC = 5 ∧ BC = 3 → ∃ P, (BP / PE) = 2 := 
by 
  intro h
  sorry

end triangle_bisector_intersection_ratio_l560_560445


namespace inequality_sqrt_ge_sum_mul_l560_560941

theorem inequality_sqrt_ge_sum_mul (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) : 
  √x + √y + √z ≥ x * y + y * z + z * x := 
sorry

end inequality_sqrt_ge_sum_mul_l560_560941


namespace num_of_functions_with_period_pi_l560_560302

def f1 (x : ℝ) : ℝ := Real.tan (x / 2 - Real.pi / 3)
def f2 (x : ℝ) : ℝ := abs (Real.sin x)
def f3 (x : ℝ) : ℝ := Real.sin x * Real.cos x
def f4 (x : ℝ) : ℝ := Real.cos x + Real.sin x

theorem num_of_functions_with_period_pi :
  (∀ x : ℝ, f1 (x + 2 * Real.pi) = f1 x) ∧
  (∀ x : ℝ, f2 (x + Real.pi) = f2 x) ∧
  (∀ x : ℝ, f3 (x + Real.pi) = f3 x) ∧
  (∀ x : ℝ, f4 (x + 2 * Real.pi) = f4 x) →
  (count [f1, f2, f3, f4] (λ f, smallest_positive_period f = Real.pi) = 2) :=
by
sorry

end num_of_functions_with_period_pi_l560_560302


namespace probability_of_two_tails_l560_560571

-- Define a type representing the outcome of a fair coin toss
inductive Coin
| Heads
| Tails

open Coin

-- Define an event for tossing three coins
def toss_three_coins : list (Coin × Coin × Coin) :=
  [ (Heads, Heads, Heads), (Heads, Heads, Tails), (Heads, Tails, Heads), (Heads, Tails, Tails),
    (Tails, Heads, Heads), (Tails, Heads, Tails), (Tails, Tails, Heads), (Tails, Tails, Tails) ]

-- Define a predicate for checking if there are exactly 2 tails
def exactly_two_tails (t: Coin × Coin × Coin) : Prop :=
  t.1 = Tails ∧ t.2 = Tails ∧ t.3 = Heads ∨
  t.1 = Tails ∧ t.2 = Heads ∧ t.3 = Tails ∨
  t.1 = Heads ∧ t.2 = Tails ∧ t.3 = Tails

-- Define the proof problem: prove the probability of exactly 2 tails is 3/8
theorem probability_of_two_tails :
  (∑ t in toss_three_coins, if exactly_two_tails t then 1 else 0) / (2^3 : ℝ) = 3 / 8 := 
by
  sorry

end probability_of_two_tails_l560_560571


namespace find_f_neg3_l560_560789

def f (x : ℝ) : ℝ :=
  if x > 0 then log (x + 1) / log 2 else -log (-x + 1) / log 2

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

theorem find_f_neg3 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : ∀ x : ℝ, x > 0 → f(x) = log (x + 1) / log 2) :
  f (-3) = -2 :=
by
  sorry

end find_f_neg3_l560_560789


namespace largest_product_of_sum_2017_l560_560566

theorem largest_product_of_sum_2017 : 
  (∃ (S : List ℕ), (∀ (x ∈ S), x > 0) ∧ S.sum = 2017 ∧ S.prod = 4 * 3^671) := sorry

end largest_product_of_sum_2017_l560_560566


namespace estimate_arrow_reading_l560_560189

theorem estimate_arrow_reading (x : ℝ) (h1 : 9.80 < x) (h2 : x < 10.0) :
  x ≈ 9.95 := 
sorry

end estimate_arrow_reading_l560_560189


namespace problem1_problem2_l560_560262

-- Problem 1: In any set of b consecutive positive integers,
-- there exist two numbers whose product is divisible by ab.
theorem problem1 (a b : ℕ) (h: 0 < a ∧ a < b) :
  (∀ s : Finset ℕ, s.card = b → ∃ x y ∈ s, x * y % (a * b) = 0) :=
by
  sorry

-- Problem 2: In any set of c consecutive positive integers,
-- there exist three numbers whose product is divisible by abc.
theorem problem2 (a b c : ℕ) (h: 0 < a ∧ a < b ∧ b < c) :
  ¬ ∀ s : Finset ℕ, s.card = c → ∃ x y z ∈ s, x * y * z % (a * b * c) = 0 :=
by
  sorry

end problem1_problem2_l560_560262


namespace expand_polynomial_eq_l560_560704

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560704


namespace quadratic_function_increases_l560_560074

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

-- Prove that for x > 1, the function value y increases as x increases
theorem quadratic_function_increases (x : ℝ) (h : x > 1) : 
  quadratic_function x > quadratic_function 1 :=
sorry

end quadratic_function_increases_l560_560074


namespace num_ways_to_arrange_digits_l560_560852

theorem num_ways_to_arrange_digits : 
  let digits := {4, 0, 5, 2, 1}
  in ∃! (n : ℕ), n = 96 ∧ 
    (n = (+) 
      (4 * (Nat.factorial (digits.size - 1))) 
    ∧ ∀ d ∈ digits, d ≠ 0 → (digits.size == 5) ∧ 
    (∀ (start_pos : Fin 5), start_pos ≠ 0)) :=
by
  sorry

end num_ways_to_arrange_digits_l560_560852


namespace fraction_meaningful_l560_560840

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ¬ (x - 1 = 0) :=
by
  sorry

end fraction_meaningful_l560_560840


namespace hyperbola_equation_theorem_l560_560070

-- Define the conditions as hypotheses in Lean
noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ (SetOf (λ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 / b^2 = 1)) ∧
      ∀ (p : ℝ × ℝ), p = (sqrt 3, 3) → p.2 = (b / a) * p.1 ∧ 
      ∃ c : ℝ, c = 4 ∧ (c^2 = a^2 + b^2) ∧ c^2 = 4 * 4) → 
    (a = 2 ∧ b = 2 * sqrt 3)

-- State the theorem that needs to be proven
theorem hyperbola_equation_theorem : 
  hyperbola_equation →
  (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1) ↔ true) :=
by
  intro h
  sorry

end hyperbola_equation_theorem_l560_560070


namespace product_first_2018_terms_l560_560037

noncomputable def f (x : ℝ) : ℝ := √3 * Real.sin x + Real.cos x

def a1 : ℝ := 2 -- since the maximum value of f(x) is 2

def S (n : ℕ) : ℝ := sorry -- Sum of the first n terms

def a (n : ℕ) : ℝ := sorry -- Sequence definition

axiom sequence_relation (n : ℕ) : a n - a n * S (n + 1) = a1 / 2 - a n * S n

theorem product_first_2018_terms : 
  (List.range 2018).map (λ n => a n).prod = 1 :=
sorry

end product_first_2018_terms_l560_560037


namespace percentage_saved_is_10_l560_560299

-- Given conditions
def rent_expenses : ℕ := 5000
def milk_expenses : ℕ := 1500
def groceries_expenses : ℕ := 4500
def education_expenses : ℕ := 2500
def petrol_expenses : ℕ := 2000
def misc_expenses : ℕ := 3940
def savings : ℕ := 2160

-- Define the total expenses
def total_expenses : ℕ := rent_expenses + milk_expenses + groceries_expenses + education_expenses + petrol_expenses + misc_expenses

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage of savings
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- Prove that the percentage saved is 10%
theorem percentage_saved_is_10 :
  percentage_saved = 10 :=
sorry

end percentage_saved_is_10_l560_560299


namespace tire_price_l560_560212

theorem tire_price (payment : ℕ) (price_ratio : ℕ → ℕ → Prop)
  (h1 : payment = 345)
  (h2 : price_ratio 3 1)
  : ∃ x : ℕ, x = 99 := 
sorry

end tire_price_l560_560212


namespace line_through_points_slope_intercept_sum_l560_560523

theorem line_through_points_slope_intercept_sum :
  (∃ m b : ℝ, (5 = -3 * m + b) ∧ (-4 = b)) → (-3 - 4 = -7) := by
  intros h
  cases h with m hm
  cases hm with b hb
  cases hb

  have hb_val : b = -4 :=
    hb_right
  have hm_val : m = -3 := by
    rw [hb.left, hb_val, add_comm,
    smul_eq_zero, neg_eq_neg_iff, smul_eq_mul] at hb_left
    exact negEqNegOfEq (smul_eq_zero.mp $ eq_of_smul_eq_smul zero_smul.smul_left $ hb.left)
  rw [hb_val, hm_val]
  sorry


end line_through_points_slope_intercept_sum_l560_560523


namespace union_sets_l560_560808

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} := by
  sorry

end union_sets_l560_560808


namespace divisors_of_27_divide_n_squared_plus_2n_plus_27_l560_560344

open Nat

theorem divisors_of_27_divide_n_squared_plus_2n_plus_27 :
  {n : ℕ // n ∣ 27} = {n : ℕ // n ∣ (n^2 + 2*n + 27)} ↔ 
  ∀ (n : ℕ), n ∣ 27 → n ∈ {1, 3, 9, 27} := 
by
  sorry

end divisors_of_27_divide_n_squared_plus_2n_plus_27_l560_560344


namespace expand_expression_l560_560722

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560722


namespace remainder_when_divided_by_6_l560_560251

theorem remainder_when_divided_by_6 (n : ℤ) (h_pos : 0 < n) (h_mod12 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l560_560251


namespace divisible_by_27_l560_560940

theorem divisible_by_27 (n : ℕ) : 27 ∣ (2^(5*n+1) + 5^(n+2)) :=
by
  sorry

end divisible_by_27_l560_560940


namespace people_through_checkpoint_8_l560_560632

theorem people_through_checkpoint_8 (x : ℕ) 
  (h1 : ∀ (k : ℕ), (k = 1 ∨ k = 2 ∨ k = 3) → (k → 2 * x))
  (h2 : ∀ (p : ℕ), (p = 7) → 15 = 5 * x) : 
  24 = ∑ k in {4, 5, 6}, (if k = 6 then x else 2 * x / 2) := by
{
  sorry
}

end people_through_checkpoint_8_l560_560632


namespace count_ordered_pairs_l560_560826

theorem count_ordered_pairs :
  ∃ n : ℕ, n = 12 ∧ ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 3 ∧ x + y < 7 → (x, y) ∈ {(1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3)} :=
by
  sorry

end count_ordered_pairs_l560_560826


namespace smallest_positive_period_of_f_l560_560058

def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin x

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ ε > 0, ε < T → ∃ x, f (x + ε) ≠ f x :=
by
  sorry

end smallest_positive_period_of_f_l560_560058


namespace problem_l560_560414

-- Definition of the condition
def condition (x : ℝ) : Prop :=
  sqrt (10 + x) + sqrt (15 - x) = 6

-- Definition of the proof problem statement
theorem problem (x : ℝ) (h : condition x) : (10 + x) * (15 - x) = 121 / 4 :=
by
  sorry

end problem_l560_560414


namespace impedance_current_l560_560431

noncomputable def V : ℂ := 2 + 2 * Complex.i
noncomputable def Z1 : ℂ := 2 + 1 * Complex.i
noncomputable def Z2 : ℂ := 4 + 2 * Complex.i

noncomputable def Z : ℂ := (Z1 * Z2) / (Z1 + Z2)

theorem impedance_current (hV : V = 2 + 2 * Complex.i)
    (hZ1 : Z1 = 2 + Complex.i)
    (hZ2 : Z2 = 4 + 2 * Complex.i)
    (hZ : Z = (Z1 * Z2) / (Z1 + Z2)) :
    V / Z = 2 + 2 * Complex.i :=
by
  sorry

end impedance_current_l560_560431


namespace solve_for_s_l560_560509

theorem solve_for_s :
  let numerator := Real.sqrt (7^2 + 24^2)
  let denominator := Real.sqrt (64 + 36)
  let s := numerator / denominator
  s = 5 / 2 :=
by
  sorry

end solve_for_s_l560_560509


namespace largest_power_of_3_dividing_product_of_first_150_odd_integers_l560_560885

theorem largest_power_of_3_dividing_product_of_first_150_odd_integers :
  ∃ k' : ℕ, (∀ n : ℕ, k' ≥ n → ¬ (3 ^ n ∣ (∏ i in finset.range 150, 2 * i + 1))) ∧ k' = 76 :=
begin
  sorry
end

end largest_power_of_3_dividing_product_of_first_150_odd_integers_l560_560885


namespace problem_equiv_proof_l560_560440

noncomputable def count_ways : ℕ :=
  let digits : List ℕ := [0, 2, 4, 5, 7, 9]
  if (digits.length) < 6 then 0
  else let ways_div_25 : List (ℕ × ℕ) := [(5, 0), (0, 5), (2, 5), (7, 5), (4, 5), (5, 5), (9, 5)]
    let ways_div_3 := (List.product digits digits).filter (λ (d1 d2 : ℕ), ((d1+d2) % 3 = 2))
    (ways_div_3.length) * (ways_div_3.length)

theorem problem_equiv_proof :
  count_ways = 2592 :=
by sorry

end problem_equiv_proof_l560_560440


namespace angle_omt_eq_half_angle_bac_l560_560456

noncomputable theory

variables {A B C O N M T : Type}
variables [IsoscelesTriangle A B C]
variables [Circumcenter A B C O]
variables [Midpoint B C N]
variables [Reflection N AC M]
variables [Rectangle A N B T]

theorem angle_omt_eq_half_angle_bac (h1 : AB = AC)
  (h2 : is_circumcenter ABC O)
  (h3 : is_midpoint BC N)
  (h4 : is_reflection N AC M)
  (h5 : is_rectangle AN BT) :
  ∠ OMT = (1/2) * ∠ BAC := 
sorry

end angle_omt_eq_half_angle_bac_l560_560456


namespace det_matrix_l560_560315

theorem det_matrix :
  Matrix.det !![!![9, 5], !![-3, 4]] = 51 :=
by sorry

end det_matrix_l560_560315


namespace lydia_flowers_on_porch_l560_560914

theorem lydia_flowers_on_porch:
  ∀ (total_plants : ℕ) (flowering_percentage : ℚ) (fraction_on_porch : ℚ) (flowers_per_plant : ℕ),
  total_plants = 80 →
  flowering_percentage = 0.40 →
  fraction_on_porch = 1 / 4 →
  flowers_per_plant = 5 →
  let flowering_plants := (total_plants : ℚ) * flowering_percentage in
  let porch_plants := flowering_plants * fraction_on_porch in
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ) in
  total_flowers_on_porch = 40 :=
by {
  intros total_plants flowering_percentage fraction_on_porch flowers_per_plant,
  intros h1 h2 h3 h4,
  let flowering_plants := (total_plants : ℚ) * flowering_percentage,
  let porch_plants := flowering_plants * fraction_on_porch,
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ),
  sorry
}

end lydia_flowers_on_porch_l560_560914


namespace find_t_l560_560415

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (2 * t - 1)) = real.sqrt4 (12 - 2 * t)) : t = 21 / 20 := 
  sorry

end find_t_l560_560415


namespace find_number_of_valid_polynomials_l560_560739

noncomputable def number_of_polynomials_meeting_constraints : Nat :=
  sorry

theorem find_number_of_valid_polynomials : number_of_polynomials_meeting_constraints = 11 :=
  sorry

end find_number_of_valid_polynomials_l560_560739


namespace S_10_minus_S_7_l560_560546

-- Define the first term and common difference of the arithmetic sequence
variables (a₁ d : ℕ)

-- Define the arithmetic sequence based on the first term and common difference
def arithmetic_sequence (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions given in the problem
axiom a_5_eq : a₁ + 4 * d = 8
axiom S_3_eq : sum_arithmetic_sequence a₁ 3 = 6

-- The goal: prove that S_10 - S_7 = 48
theorem S_10_minus_S_7 : sum_arithmetic_sequence a₁ 10 - sum_arithmetic_sequence a₁ 7 = 48 :=
sorry

end S_10_minus_S_7_l560_560546


namespace largest_prime_factor_4752_l560_560999

theorem largest_prime_factor_4752 : ∃ p : ℕ, p = 11 ∧ prime p ∧ (∀ q : ℕ, prime q ∧ q ∣ 4752 → q ≤ 11) :=
by
  sorry

end largest_prime_factor_4752_l560_560999


namespace quadratic_roots_are_distinct_real_l560_560501

theorem quadratic_roots_are_distinct_real (a b c : ℝ) (h_eq : a = 1) (h_b : b = 4) (h_c : c = 0) :
  a * c = 0 ∧ b^2 - 4 * a * c > 0 :=
by
  rw [h_eq, h_b, h_c]
  split
  case left => 
    calc 
      1 * 0 = 0 : by norm_num
  case right =>
    calc 
      4^2 - 4 * 1 * 0 = 16 - 0 : by norm_num
      ... = 16      : by norm_num
      ... > 0       : by norm_num

end quadratic_roots_are_distinct_real_l560_560501


namespace monotonically_decreasing_power_function_l560_560965

theorem monotonically_decreasing_power_function (m : ℝ) :
  (m^2 - 1 < 0) ∧ (m + 1 < 0) → m = -√2 :=
by
  -- Definitions and assumptions
  intro h
  have h1 : m^2 - 1 < 0 := h.1
  have h2 : m + 1 < 0 := h.2
  -- Required proof
  sorry

end monotonically_decreasing_power_function_l560_560965


namespace marie_curie_birthdate_day_l560_560948

theorem marie_curie_birthdate_day
  (anniversary_date : Nat) -- 2022
  (anniversary_day : String) -- "Monday"
  (years_between : Nat) -- 150 years
  (leap_year_rule : ∀ (year : Nat), year % 400 = 0 ∨ (year % 4 = 0 ∧ year % 100 ≠ 0) → leap year)
  (regular_year_dayshift : Nat) -- 1 day shift
  (leap_year_dayshift : Nat) -- 2 days shift) :
  (marie_birth_day : String) := -- Result to prove is "Wednesday" 
  sorry

end marie_curie_birthdate_day_l560_560948


namespace number_of_penny_piles_l560_560937

theorem number_of_penny_piles
    (piles_of_quarters : ℕ := 4) 
    (piles_of_dimes : ℕ := 6)
    (piles_of_nickels : ℕ := 9)
    (total_value_in_dollars : ℝ := 21)
    (coins_per_pile : ℕ := 10)
    (quarter_value : ℝ := 0.25)
    (dime_value : ℝ := 0.10)
    (nickel_value : ℝ := 0.05)
    (penny_value : ℝ := 0.01) :
    (total_value_in_dollars - ((piles_of_quarters * coins_per_pile * quarter_value) +
                               (piles_of_dimes * coins_per_pile * dime_value) +
                               (piles_of_nickels * coins_per_pile * nickel_value))) /
                               (coins_per_pile * penny_value) = 5 := 
by
  sorry

end number_of_penny_piles_l560_560937


namespace value_of_a2_l560_560420

variable {R : Type*} [Ring R] (x a_0 a_1 a_2 a_3 : R)

theorem value_of_a2 
  (h : ∀ x : R, x^3 = a_0 + a_1 * (x - 2) + a_2 * (x - 2)^2 + a_3 * (x - 2)^3) :
  a_2 = 6 :=
sorry

end value_of_a2_l560_560420


namespace general_formula_find_k_l560_560792

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Conditions
axiom arithmetic_sequence (a : ℕ → ℤ) : ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
axiom sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : ∀ n, S n = (n * (a 1) + n * (n-1) / 2 * (a 2 - a 1))
axiom a_1 : a 1 = 9
axiom S_3 : S 3 = 21

-- Question Ⅰ: Prove the general formula for the sequence
theorem general_formula : ∀ n, a n = -2 * n + 11 :=
sorry

-- Question Ⅱ: Prove the value of k
theorem find_k (k : ℕ) (hk : (a 5), (a 8), (S k) form_geometric_sequence)
  : k = 5 :=
sorry

end general_formula_find_k_l560_560792


namespace tensor_identity_l560_560651

namespace tensor_problem

def otimes (x y : ℝ) : ℝ := x^2 + y

theorem tensor_identity (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a :=
by sorry

end tensor_problem

end tensor_identity_l560_560651


namespace max_value_frac_l560_560744

theorem max_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    ∃ (c : ℝ), c = 1/4 ∧ (∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (xyz * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ c) := 
by
  sorry

end max_value_frac_l560_560744


namespace flowers_on_porch_l560_560909

-- Definitions based on problem conditions
def total_plants : ℕ := 80
def flowering_percentage : ℝ := 0.40
def fraction_on_porch : ℝ := 0.25
def flowers_per_plant : ℕ := 5

-- Theorem statement
theorem flowers_on_porch (h1 : total_plants = 80)
                         (h2 : flowering_percentage = 0.40)
                         (h3 : fraction_on_porch = 0.25)
                         (h4 : flowers_per_plant = 5) :
    (total_plants * seminal (flowering_percentage * fraction_on_porch) * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560909


namespace expand_expression_l560_560720

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560720


namespace opposite_of_seven_l560_560539

theorem opposite_of_seven : ∃ x : ℤ, 7 + x = 0 ∧ x = -7 :=
by
  sorry

end opposite_of_seven_l560_560539


namespace root_of_quadratic_l560_560052

theorem root_of_quadratic (x m : ℝ) (h : x = -1 ∧ x^2 + m*x - 1 = 0) : m = 0 :=
sorry

end root_of_quadratic_l560_560052


namespace problem_statement_l560_560758

def f (x : ℝ) : ℝ := log10 (sqrt (x^2 + 1) - x) + 1

theorem problem_statement : f 2017 + f (-2017) = 2 := by
  sorry

end problem_statement_l560_560758


namespace quadrilateral_area_is_six_l560_560318

-- Definitions of angles and lengths based on the problem statement
def angle_DAB : ℝ := 60
def angle_ABC : ℝ := 90
def angle_BCD : ℝ := 120

def MB : ℝ := 1
def MD : ℝ := 2

-- Intersection point of diagonals notated as M
axiom diagonals_intersect_at_M 
   (A B C D M : Type*) 
   (AC BD : A → B → ℝ)
   (intersects : ∃ M, AC A C = AC A M + AC M C ∧ BD B D = BD B M + BD M D)

-- The task is to show the area of the quadrilateral ABCD is 6
theorem quadrilateral_area_is_six 
    (A B C D : Type*) 
    (angle_DAB : angle A D B = 60)
    (angle_ABC : angle A B C = 90)
    (angle_BCD : angle B C D = 120)
    (diagonals_intersect_at_M : ∃ M, A → M = 1 ∧ M → D = 2):
    area_quadrilateral A B C D = 6 := 
by
  apply sorry

end quadrilateral_area_is_six_l560_560318


namespace probability_three_dice_sum_to_fourth_l560_560359

-- Define the probability problem conditions
def total_outcomes : ℕ := 8^4
def favorable_outcomes : ℕ := 1120

-- Final probability for the problem
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Lean statement for the proof problem
theorem probability_three_dice_sum_to_fourth :
  probability favorable_outcomes total_outcomes = 35 / 128 :=
by sorry

end probability_three_dice_sum_to_fourth_l560_560359


namespace find_k_for_perpendicular_lines_l560_560387

noncomputable def test_perpendicular_lines (k : ℝ) : Prop :=
  let l1 := (k - 3) * x + (5 - k) * y + 1 = 0
  let l2 := 2 * (k - 3) * x - 2 * y + 3 = 0
  let slope_l1 : ℝ := (3 - k) / (5 - k)
  let slope_l2 : ℝ := (k - 3)
  slope_l1 * slope_l2 = -1

theorem find_k_for_perpendicular_lines (k : ℝ) :
  ((k = 1) ∨ (k = 4)) ↔ test_perpendicular_lines k :=
sorry

end find_k_for_perpendicular_lines_l560_560387


namespace vector_relationship_l560_560782

open_locale vector_space

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C P : V}

-- Given conditions
def condition1 : Prop := sorry
def condition2 (B C P : V) : Prop := vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({B} : set V) = 2 • (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({C} : set V))

-- The theorem statement based on the given conditions
theorem vector_relationship (h1 : condition1) (h2 : condition2 B C P) : 
  (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({A} : set V)) = 
  (-1/2 : ℝ) • (vector_span ℝ ({B} : set V) -ᵥ vector_span ℝ ({A} : set V)) + (3/2 : ℝ) • (vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({A} : set V)) :=
sorry

end vector_relationship_l560_560782


namespace distance_AB_eq_4_sqrt_3_l560_560770

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (1, 1, 1)
def B : ℝ × ℝ × ℝ := (-3, -3, -3)

-- Define a function to calculate the distance between two points in 3D space
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 + (Q.3 - P.3) ^ 2).sqrt

theorem distance_AB_eq_4_sqrt_3 : distance A B = 4 * Real.sqrt 3 :=
  by sorry

end distance_AB_eq_4_sqrt_3_l560_560770


namespace largest_prime_factor_of_4752_l560_560998

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l560_560998


namespace first_interest_rate_is_correct_l560_560506

theorem first_interest_rate_is_correct :
  let A1 := 1500.0000000000007
  let A2 := 2500 - A1
  let yearly_income := 135
  (15.0 * (r / 100) + 6.0 * (A2 / 100) = yearly_income) -> r = 5.000000000000003 :=
sorry

end first_interest_rate_is_correct_l560_560506


namespace parabola_conclusions_correct_l560_560761

theorem parabola_conclusions_correct (a b c t : ℝ) (h1 : b = -2 * a) (h2 : 3 * a + c = 0)
        (h3 : ∀ x : ℝ, y : ℝ, y = a * x^2 + b * x + c → y = t → 
          (ax^2 + bx + c - t = a * (x + 2) * (x - 4))) :
        3 = 3 :=
by
  sorry

end parabola_conclusions_correct_l560_560761


namespace geometric_seq_general_formula_sum_of_inverse_b_n_l560_560217

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_logs_sequence (a : ℕ → ℝ) : (ℕ → ℝ) :=
  λ n, ∑ i in finset.range (n + 1), real.log (a i)

theorem geometric_seq_general_formula
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h1 : 2 * a 1 + 3 * a 2 = 1)
  (h2 : (a 3)^2 = 9 * a 2 * a 6)
  : ∀ n, a n = 1 / (3^n) :=
sorry

theorem sum_of_inverse_b_n
  (a : ℕ → ℝ)
  (h_seq : ∀ n, a n = 1 / (3^n))
  : ∀ n, (∑ k in finset.range (n + 1), 1 / (sum_of_logs_sequence a k)) = - (2 * n) / (n + 1) :=
sorry

end geometric_seq_general_formula_sum_of_inverse_b_n_l560_560217


namespace curve_is_hyperbola_l560_560113

theorem curve_is_hyperbola (m n x y : ℝ) (h_eq : m * x^2 - m * y^2 = n) (h_mn : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2/a^2 - x^2/a^2 = 1 := 
sorry

end curve_is_hyperbola_l560_560113


namespace flowers_on_porch_l560_560911

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560911


namespace remaining_payment_l560_560092

theorem remaining_payment (deposit_percent : ℝ) (deposit_amount : ℝ) (total_percent : ℝ) (total_price : ℝ) :
  deposit_percent = 5 ∧ deposit_amount = 50 ∧ total_percent = 100 → total_price - deposit_amount = 950 :=
by {
  sorry
}

end remaining_payment_l560_560092


namespace sequence_count_l560_560768

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
    a 10 = 3 * a 1 ∧ 
    a 2 + a 8 = 2 * a 5 ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 9 → a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i) ∧ 
    (∃ n, n = 80) :=
sorry

end sequence_count_l560_560768


namespace triangle_area_geometric_mean_l560_560588

-- Define the conditions
variables {A B C H E : Type}
variables {ABC ABH ABE : Set Type}
variables (orthocenter : H ∈ ABC)
variables (point_on_segment : ∃ (E : Type), E ∈ CH ∧ ABE ∈ CH)
variables (right_angle : ∠AEB = 90)

-- Define the areas of triangles
variables (area_ABC area_ABH area_ABE : ℝ)

-- Geometric Mean
def geometric_mean (x y : ℝ) : ℝ := real.sqrt (x * y)

-- Main theorem statement
theorem triangle_area_geometric_mean
  (h_orthocenter : H = orthocenter)
  (h_point_on_segment : point_on_segment H)
  (h_right_angle : right_angle)
  (h_area_ABC : area_ABC > 0)
  (h_area_ABH : area_ABH > 0)
  (h_area_ABE : area_ABE > 0) :
  area_ABE = geometric_mean area_ABC area_ABH :=
sorry

end triangle_area_geometric_mean_l560_560588


namespace main_theorem_l560_560146

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom odd_function_f (x : ℝ) : f(-x) = -f(x)
axiom second_derivative_of_f (x : ℝ) : deriv (deriv f x) = f'' x
axiom f_at_neg2 : f (-2) = 0
axiom condition_for_positive_x (x : ℝ) : 0 < x → f x + x / 3 * f'' x > 0

theorem main_theorem : ∀ x : ℝ, (f x > 0 ↔ (2 < x ∨ (-2 < x ∧ x < 0))) :=
by
  sorry

end main_theorem_l560_560146


namespace magician_strategy_l560_560255

-- Define the problem parameters
def num_boxes : ℕ := 12

-- Define the positions the magician will point to after the assistant opens a box at position k
def open_positions (k : ℕ) : finset ℕ := 
finset.map (function.embedding.subtype _) 
  ({((k + 1) % num_boxes), ((k + 2) % num_boxes), ((k + 5) % num_boxes), ((k + 7) % num_boxes)} : finset ℕ)

-- Define the main strategy theorem that guarantees the magician will always find the coins
theorem magician_strategy (k : ℕ) (h : k < num_boxes) : 
  ∃ C1 C2 : ℕ, 
  C1 ≠ C2 ∧ C1 < num_boxes ∧ C2 < num_boxes ∧ 
  ({C1, C2} ⊆ open_positions k) :=
sorry

end magician_strategy_l560_560255


namespace square_perimeter_l560_560517

-- Define the area of the square
def square_area := 720

-- Define the side length of the square
noncomputable def side_length := Real.sqrt square_area

-- Define the perimeter of the square
noncomputable def perimeter := 4 * side_length

-- Statement: Prove that the perimeter is 48 * sqrt(5)
theorem square_perimeter : perimeter = 48 * Real.sqrt 5 :=
by
  -- The proof is omitted as instructed
  sorry

end square_perimeter_l560_560517


namespace equal_areas_of_parts_l560_560617

theorem equal_areas_of_parts :
  ∀ (S1 S2 S3 S4 : ℝ), 
    S1 = S2 → S2 = S3 → 
    (S1 + S2 = S3 + S4) → 
    (S2 + S3 = S1 + S4) → 
    S1 = S2 ∧ S2 = S3 ∧ S3 = S4 :=
by
  intros S1 S2 S3 S4 h1 h2 h3 h4
  sorry

end equal_areas_of_parts_l560_560617


namespace expand_product_l560_560695

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560695


namespace polygon_assignment_l560_560282

noncomputable def assign_values (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ i, i ∈ {1, 2, ..., 4n+2}) ∧
    (∀ i j, (i, j) ∈ (range (2n + 1) × range (2n + 1)) → f i + f j + f (midpoint i j) = 5n+4)
  
theorem polygon_assignment (n : ℕ) :
    ∃ (f : ℕ → ℕ),
      (∀ i, i ∈ {1, 2, ..., 4n+2}) ∧
      (∀ i j, (i, j) ∈ (range (2n + 1) × range (2n + 1)) → f i + f j + f (midpoint i j) = 5n+4) :=
sorry

end polygon_assignment_l560_560282


namespace find_f_2017_l560_560405

theorem find_f_2017 (f : ℕ → ℕ) (H1 : ∀ x y : ℕ, f (x * y + 1) = f x * f y - f y - x + 2) (H2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end find_f_2017_l560_560405


namespace GNOME_area_sum_l560_560930

/-- Points G and N are chosen on the interiors of sides E D and D O of unit square DOME,
    so that pentagon GNOME has only two distinct side lengths. Given these conditions,
    the sum of all possible areas of quadrilateral NOME can be expressed as a - b √c / d,
    where a, b, c, d are positive integers such that gcd(a, b, d) = 1 and c is square-free,
    we aim to prove that 1000a + 100b + 10c + d = 10324. -/
theorem GNOME_area_sum :
  ∃ (a b c d : ℕ), gcd a b d = 1 ∧ 
                   ¬(∃ e : ℕ, e * e = c) ∧ -- Ensuring c is square-free
                   (a : ℚ) - (b * real.sqrt c) / (d : ℚ) = (10 : ℚ) - 3 * real.sqrt 2 / 4 ∧
                   (1000 * a + 100 * b + 10 * c + d = 10324) :=
by
  sorry

end GNOME_area_sum_l560_560930


namespace A_can_give_C_start_l560_560849

noncomputable def start_A_can_give_C : ℝ :=
  let start_AB := 50
  let start_BC := 157.89473684210532
  start_AB + start_BC

theorem A_can_give_C_start :
  start_A_can_give_C = 207.89473684210532 :=
by
  sorry

end A_can_give_C_start_l560_560849


namespace inequality_solution_set_l560_560970

theorem inequality_solution_set (x : ℝ) : 3 ≤ abs (5 - 2 * x) ∧ abs (5 - 2 * x) < 9 ↔ (x > -2 ∧ x ≤ 1) ∨ (x ≥ 4 ∧ x < 7) := sorry

end inequality_solution_set_l560_560970


namespace michael_bath_times_per_week_l560_560483

theorem michael_bath_times_per_week (B : ℕ) 
  (h1 : ∀ w : ℕ, w = 52 → [w * (B + 1) = 156]) : 
  B = 2 := 
by
  sorry

end michael_bath_times_per_week_l560_560483


namespace maximum_sum_abs_diff_y_l560_560075

-- Given real numbers \( x_1, x_2, \ldots, x_{2001} \)
variables (x : Fin 2001 → ℝ)

-- Hypothesis: \( \sum_{k=1}^{2000} \left| x_k - x_{k+1} \right| = 2001 \)
def sum_abs_diff_x : ℝ :=
  ∑ k in (Finset.range 2000),
    | x k - x (k + 1) |

hypothesis (h1 : sum_abs_diff_x x = 2001)

-- Define y_k = \frac{1}{k} ( x_1 + x_2 + \cdots + x_k )
def y (k : Fin 2001) := (1 / (k + 1)) * (∑ i in (Finset.range (k + 1)), x i)

-- We need to prove ∑_{k=1}^{2000} \left| y_k - y_{k+1} \right| = 2000
def sum_abs_diff_y : ℝ :=
  ∑ k in (Finset.range 2000),
    | y x k - y x (k + 1) |

theorem maximum_sum_abs_diff_y
  (x : Fin 2001 → ℝ)
  (h1 : sum_abs_diff_x x = 2001)
  : sum_abs_diff_y x ≤ 2000 :=
  begin
    sorry
  end

end maximum_sum_abs_diff_y_l560_560075


namespace segment_PR_length_l560_560495

theorem segment_PR_length (radius : ℝ) (PQ : ℝ) (PR : ℝ) :
  (∀ P Q : ℝ × ℝ, dist (0, 0) P = radius ∧ dist (0, 0) Q = radius ∧ dist P Q = PQ ∧ PR = sqrt 32) →
  true :=
begin
  intro h,
  let P := (7, 0 : ℝ × ℝ),
  let Q := (0, 7 : ℝ × ℝ),
  have PQ_length : dist P Q = PQ := by sorry,
  have R_is_mid := by sorry,
  exact trivial,
end

end segment_PR_length_l560_560495


namespace part1_part2_l560_560390

theorem part1 (a b : ℝ) (h : ∀ x, x^2 - (a+1)*x + a < 0 ↔ b < x ∧ x < 2) :
  a = 2 ∧ b = 1 :=
by
  sorry

theorem part2 (x y k : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 / x + 1 / y = 1)
  (h_ineq : x + 2 * y ≥ sqrt k - 9 / sqrt k) :
  0 < k ∧ k ≤ 81 :=
by
  sorry

end part1_part2_l560_560390


namespace neg_prop1_true_neg_prop2_false_l560_560578

-- Proposition 1: The logarithm of a positive number is always positive
def prop1 : Prop := ∀ x : ℝ, x > 0 → Real.log x > 0

-- Negation of Proposition 1: There exists a positive number whose logarithm is not positive
def neg_prop1 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0

-- Proposition 2: For all x in the set of integers Z, the last digit of x^2 is not 3
def prop2 : Prop := ∀ x : ℤ, (x * x % 10 ≠ 3)

-- Negation of Proposition 2: There exists an x in the set of integers Z such that the last digit of x^2 is 3
def neg_prop2 : Prop := ∃ x : ℤ, (x * x % 10 = 3)

-- Proof that the negation of Proposition 1 is true
theorem neg_prop1_true : neg_prop1 := 
  by sorry

-- Proof that the negation of Proposition 2 is false
theorem neg_prop2_false : ¬ neg_prop2 := 
  by sorry

end neg_prop1_true_neg_prop2_false_l560_560578


namespace largest_finite_set_l560_560507

def satisfies_condition (S : Finset ℤ) : Prop := 
  ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → (a + b ∈ S ∨ a + c ∈ S ∨ b + c ∈ S)

theorem largest_finite_set :
  ∀ S : Finset ℤ, satisfies_condition S → S.card ≤ 7 :=
by 
  sorry

end largest_finite_set_l560_560507


namespace complex_number_in_fourth_quadrant_l560_560186

-- Define the conditions:
def z := Real.sin 1 + Complex.I * Real.cos 2

-- Define the result to check if it lies in the fourth quadrant:
def in_fourth_quadrant (x y : ℝ) : Prop := 0 < x ∧ y < 0

-- Statement we need to prove:
theorem complex_number_in_fourth_quadrant : in_fourth_quadrant (Real.sin 1) (Real.cos 2) :=
by
  -- We skip the proof with sorry to satisfy the criteria
  sorry

end complex_number_in_fourth_quadrant_l560_560186


namespace donut_ate_even_neighbors_l560_560490

def cube neighbors (n : ℕ) : ℕ := sorry

theorem donut_ate_even_neighbors : 
  (cube neighbors 5) = 63 := 
by
  sorry

end donut_ate_even_neighbors_l560_560490


namespace problem_correct_statements_l560_560441

open Real

def ast (a b : ℝ) : ℝ := a * b + a + b

def f (x : ℝ) : ℝ := 1 + exp x + (1 / exp x)

theorem problem_correct_statements :
  let min_value_correct := ∀ x, f x ≥ 3,
      even_function     := ∀ x, f (-x) = f x,
      monotonic_increase := ∀ x, 0 ≤ x → derivative f x ≥ 0
  in (min_value_correct ∧ even_function ∧ ¬monotonic_increase) ∨ 
     (min_value_correct ∧ ¬even_function ∧ ¬monotonic_increase) ∨ 
     (¬min_value_correct ∧ even_function ∧ ¬monotonic_increase) :=
sorry

end problem_correct_statements_l560_560441


namespace volume_of_tetrahedron_l560_560855

variables (AB : ℝ) (Area_ABC Area_ABD : ℝ) (angle_AB : ℝ)

-- Defining the conditions
def tetrahedron_conditions : Prop :=
  AB = 5 ∧ Area_ABC = 20 ∧ Area_ABD = 18 ∧ angle_AB = real.pi / 4

-- Statement to prove the volume of the tetrahedron
theorem volume_of_tetrahedron (h_condition : tetrahedron_conditions AB Area_ABC Area_ABD angle_AB) :
  (1 / 3) * 20 * 18 * real.sin (real.pi / 4) = 84.84 :=
sorry

end volume_of_tetrahedron_l560_560855


namespace profit_equation_l560_560289

noncomputable def price_and_profit (x : ℝ) : ℝ :=
  (1 + 0.5) * x * 0.8 - x

theorem profit_equation : ∀ x : ℝ, price_and_profit x = 8 → ((1 + 0.5) * x * 0.8 - x = 8) :=
 by intros x h
    exact h

end profit_equation_l560_560289


namespace archibald_percentage_wins_l560_560628

theorem archibald_percentage_wins (A_wins B_wins : ℕ) (hA : A_wins = 12) (hB : B_wins = 18) :
  let total_games := A_wins + B_wins
  let archibald_percentage := (A_wins / total_games.to_rat) * 100
  archibald_percentage = 40 :=
by
  -- Defining the total number of games
  let total_games := A_wins + B_wins
  -- Defining the percentage of Archibald wins
  let archibald_percentage := (A_wins / total_games.to_rat) * 100
  -- Skipping the final proof with sorry
  sorry

end archibald_percentage_wins_l560_560628


namespace train_length_proof_l560_560293

variable (speed_kmph : ℕ) (bridge_length : ℕ) (time_seconds : ℕ) (speed_mps distance : ℝ)

def convert_speed (speed_kmph : ℕ) : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance (speed_mps : ℝ) (time_seconds : ℕ) : ℝ := speed_mps * time_seconds
noncomputable def train_length (total_distance bridge_length : ℕ) : ℕ := total_distance - bridge_length

theorem train_length_proof (h_speed : speed_kmph = 45) (h_bridge : bridge_length = 140) (h_time : time_seconds = 40) :
  train_length (total_distance (convert_speed speed_kmph) time_seconds) bridge_length = 360 := by
  sorry

end train_length_proof_l560_560293


namespace opening_to_page_50_is_random_l560_560242

def certain_event (e : Type) : Prop := ∀ x : e, true
def random_event (e : Type) : Prop := ∀ x : e, true
def impossible_event (e : Type) : Prop := false
def determined_event (e : Type) (p: e → Prop) := ∃ x : e, p x

variable (Event : Type)

-- Suppose the event of turning to page 50 is an instance of Event
axiom opening_to_page_50 : Event

-- Given definitions
axiom certain_event_def : certain_event Event
axiom random_event_def : random_event Event
axiom impossible_event_def : impossible_event Event
axiom determined_event_def : determined_event Event (λ x, true)

-- Proof statement
theorem opening_to_page_50_is_random : random_event Event :=
by 
  sorry

end opening_to_page_50_is_random_l560_560242


namespace magazines_per_box_l560_560084

theorem magazines_per_box (total_magazines boxes : ℕ) (h1 : total_magazines = 63) (h2 : boxes = 7) : total_magazines / boxes = 9 :=
by
  rw [h1, h2]
  norm_num
  sorry

end magazines_per_box_l560_560084


namespace melany_fence_l560_560482

-- Definitions
def L (total_budget cost_per_foot : ℝ) : ℝ := total_budget / cost_per_foot
noncomputable def length_not_fenced (perimeter length_bought : ℝ) : ℝ := perimeter - length_bought

-- Constants
def total_budget : ℝ := 120000
def cost_per_foot : ℝ := 30
def perimeter : ℝ := 5000

-- Proof problem in Lean 4 statement
theorem melany_fence : length_not_fenced perimeter (L total_budget cost_per_foot) = 1000 := by
  sorry

end melany_fence_l560_560482


namespace math_club_team_selection_l560_560155

open scoped BigOperators

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem math_club_team_selection :
  (comb 7 2 * comb 9 4) + 
  (comb 7 3 * comb 9 3) +
  (comb 7 4 * comb 9 2) +
  (comb 7 5 * comb 9 1) +
  (comb 7 6 * comb 9 0) = 7042 := 
sorry

end math_club_team_selection_l560_560155


namespace probability_four_collinear_dots_l560_560865

noncomputable def probability_collinear_four_dots : ℚ :=
  let total_dots := 25
  let choose_4 := (total_dots.choose 4)
  let successful_outcomes := 60
  successful_outcomes / choose_4

theorem probability_four_collinear_dots :
  probability_collinear_four_dots = 12 / 2530 :=
by
  sorry

end probability_four_collinear_dots_l560_560865


namespace root_interval_existence_l560_560211

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem root_interval_existence : 
  ∃ c ∈ Ioo 1 2, f c = 0 :=
by 
  have h_continuous : Continuous f := sorry
  have h_f1 : f 1 < 0 := by norm_num
  have h_f2 : f 2 > 0 := by norm_num
  exact intermediate_value_Ioo h_continuous h_f1 h_f2

end root_interval_existence_l560_560211


namespace sin_shift_left_by_pi_over_8_l560_560197

theorem sin_shift_left_by_pi_over_8 :
  ∀ (x : ℝ), (sin (2 * x + π / 4)) = sin (2 * (x + π / 8)) := by 
  sorry

end sin_shift_left_by_pi_over_8_l560_560197


namespace intersection_of_circles_l560_560373

noncomputable def required_radius (n : ℕ) : ℝ :=
  1 / (2 * Real.sin (Real.pi / n))

theorem intersection_of_circles (n : ℕ) (h : 3 ≤ n) :
  ∃ r, r = required_radius n ∧
    ∀ (i j : ℕ) (hi : i < n) (hj : j < n),
      (Real.dist (⟨Real.cos (2 * i * Real.pi / n), Real.sin (2 * i * Real.pi / n)⟩)
                 (⟨Real.cos (2 * j * Real.pi / n), Real.sin (2 * j * Real.pi / n)⟩) 
                 ≤ 2 * r) := 
sorry

end intersection_of_circles_l560_560373


namespace evaluate_expression_l560_560660

theorem evaluate_expression :
  (305^2 - 275^2) / 30 = 580 := 
by
  sorry

end evaluate_expression_l560_560660


namespace range_of_m_l560_560746

-- Define the function y and the condition that y increases as x increases
def y (m x : ℝ) : ℝ := (m - 5) / x

theorem range_of_m (m : ℝ) (h : ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂) → y m x₁ < y m x₂) : m < 5 :=
sorry

end range_of_m_l560_560746


namespace N_minus_M_l560_560303

noncomputable def arithmetic_seq (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem N_minus_M (a_n : ℕ → ℝ)
  (h_sum : ∑ i in finset.range 100, a_n (i+1) = 5000)
  (h_lower : ∀ n, 1 ≤ n ∧ n ≤ 100 → 20 ≤ a_n n)
  (h_upper : ∀ n, 1 ≤ n ∧ n ≤ 100 → a_n n ≤ 80) :
  let d_min : ℝ := -30
  let d_max : ℝ := 30 / 99
  let M : ℝ := 50 + 24 * d_min
  let N : ℝ := 50 + 24 * d_max
  in N - M ≈ 727.273 :=
by
  sorry

end N_minus_M_l560_560303


namespace product_multiple_of_4_probability_l560_560452

namespace JuanAmalProbability

def probability_roll_multiple_of (n : ℕ) (sides : ℕ) : ℚ :=
  (Finset.filter (λ x, x % n = 0) (Finset.range (sides + 1))).card / sides

def probability_product_multiple_of_4 : ℚ :=
  let p1 := probability_roll_multiple_of 4 8  -- Juan's roll probability
  let p2 := probability_roll_multiple_of 4 12 -- Amal's roll probability
  1 - (1 - p1) * (1 - p2)

theorem product_multiple_of_4_probability :
  probability_product_multiple_of_4 = 7 / 16 :=
by sorry

end JuanAmalProbability

end product_multiple_of_4_probability_l560_560452


namespace arithmetic_sequence_15th_term_l560_560526

theorem arithmetic_sequence_15th_term :
  let first_term := 3
  let second_term := 8
  let third_term := 13
  let common_difference := second_term - first_term
  (first_term + (15 - 1) * common_difference) = 73 :=
by
  sorry

end arithmetic_sequence_15th_term_l560_560526


namespace first_player_wins_l560_560224

-- Define the board configuration and the initial stone position
structure BoardConfig where
  m : ℕ
  stonePosition : Fin (m + 1) × Fin m
  deriving Repr

-- Define a move as a transition on the board
structure Move where
  from : Fin (m + 1) × Fin m
  to : Fin (m + 1) × Fin m
  deriving Repr

-- Define a rule predicate that verifies a valid move
def validMove (board : BoardConfig) (move : Move) : Prop :=
  let ⟨(i₁, j₁), (i₂, j₂)⟩ := (move.from, move.to)
  (i₁ = i₂ ∧ (j₁ = j₂ + 1 ∨ j₁ = j₂ - 1)) ∨ (j₁ = j₂ ∧ (i₁ = i₂ + 1 ∨ i₁ = i₂ - 1))

-- Define a predicate to check if a move has been used
def usedSegment (usedMoves : List Move) (move : Move) : Prop :=
  move ∈ usedMoves

-- Define winning strategy for the first player
theorem first_player_wins
  (m : ℕ) 
  (initial_board : BoardConfig)
  (initial_moves : List Move)
  (playerOneTurn : Bool)
  : ∀ (usedMoves : List Move), usedMoves = initial_moves → playerOneTurn → ¬ ∃ (secondMove : Move), validMove initial_board secondMove ∧ ¬ usedSegment usedMoves secondMove :=
  sorry

end first_player_wins_l560_560224


namespace angle_measure_l560_560422

theorem angle_measure (x : ℝ) (h : 90 - x = 3 * (180 - x)) : x = 45 := by
  sorry

end angle_measure_l560_560422


namespace odd_function_f_x_pos_l560_560514

variable (f : ℝ → ℝ)

theorem odd_function_f_x_pos {x : ℝ} (h1 : ∀ x < 0, f x = x^2 + x)
  (h2 : ∀ x, f x = -f (-x)) (hx : 0 < x) :
  f x = -x^2 + x := by
  sorry

end odd_function_f_x_pos_l560_560514


namespace sqrt_function_of_x_l560_560575

theorem sqrt_function_of_x (x : ℝ) (h : x > 0) : ∃! y : ℝ, y = Real.sqrt x :=
by
  sorry

end sqrt_function_of_x_l560_560575


namespace cost_of_childrens_ticket_l560_560558

theorem cost_of_childrens_ticket (C : ℝ) : (16 * C + (5 * 5.50) = 83.50) → C = 3.50 :=
by
  intro h
  sorry

end cost_of_childrens_ticket_l560_560558


namespace sum_of_digits_of_9N_is_9_l560_560000

-- Define what it means for a natural number N to have strictly increasing digits.
noncomputable def strictly_increasing_digits (N : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → (N.digits i < N.digits j)

-- Formal statement of the problem
theorem sum_of_digits_of_9N_is_9 (N : ℕ) (h : strictly_increasing_digits N) : ∑ d in (9 * N).digits, d = 9 :=
by sorry

end sum_of_digits_of_9N_is_9_l560_560000


namespace slope_range_l560_560199

theorem slope_range (α : ℝ) (hα : α ∈ Ioo (π / 3) (5 * π / 6)) : 
  (∃ k : ℝ, k = Real.tan α ∧ (k < - (Real.sqrt 3) / 3 ∨ k > Real.sqrt 3)) :=
by
  sorry

end slope_range_l560_560199


namespace melany_fence_l560_560481

-- Definitions
def L (total_budget cost_per_foot : ℝ) : ℝ := total_budget / cost_per_foot
noncomputable def length_not_fenced (perimeter length_bought : ℝ) : ℝ := perimeter - length_bought

-- Constants
def total_budget : ℝ := 120000
def cost_per_foot : ℝ := 30
def perimeter : ℝ := 5000

-- Proof problem in Lean 4 statement
theorem melany_fence : length_not_fenced perimeter (L total_budget cost_per_foot) = 1000 := by
  sorry

end melany_fence_l560_560481


namespace expand_expression_l560_560668

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560668


namespace minimum_x_plus_3y_l560_560418

theorem minimum_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) : x + 3 * y ≥ 16 :=
sorry

end minimum_x_plus_3y_l560_560418


namespace compute_fraction_l560_560515

theorem compute_fraction (a b c : ℝ) (h1 : a + b = 20) (h2 : b + c = 22) (h3 : c + a = 2022) :
  (a - b) / (c - a) = 1000 :=
by
  sorry

end compute_fraction_l560_560515


namespace molecular_weight_is_correct_l560_560567

noncomputable def molecular_weight_of_compound : ℝ :=
  3 * 39.10 + 2 * 51.996 + 7 * 15.999 + 4 * 1.008 + 1 * 14.007

theorem molecular_weight_is_correct : molecular_weight_of_compound = 351.324 := 
by
  sorry

end molecular_weight_is_correct_l560_560567


namespace smallest_positive_integer_for_divisibility_conditions_l560_560237

theorem smallest_positive_integer_for_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (18 ∣ n^2) ∧ (640 ∣ n^3) ∧ ∀ m : ℕ, m > 0 ∧ (18 ∣ m^2) ∧ (640 ∣ m^3) → n ≤ m :=
begin
  use 120,
  split,
  { -- n > 0
    exact lt_add_one (120 - 1),
  },
  split,
  { -- 18 | n^2
    exact dvd.trans (dvd_mul_right 18 (120 / 6)) (dvd_refl _),
  },
  split,
  { -- 640 | n^3
    exact dvd.trans (dvd_mul_right 640 (5 / 1)) (dvd_refl _),
  },
  { -- For all m, if the conditions hold, n ≤ m
    intros m hm,
    cases hm with hm_pos hm_divisibility,
    cases hm_divisibility with hm_div1 hm_div2,
    by_contradiction h,
    sorry,
  }
end

end smallest_positive_integer_for_divisibility_conditions_l560_560237


namespace angle_HAB_eq_angle_OAC_l560_560459

variable (C_star : Type) [Circle C_star]
variable (O A B C H : Point)
variable (distinct_points : A ≠ B ∧ B ≠ C ∧ A ≠ C)
variable (foot_of_perpendicular : is_perpendicular_foot H A)

theorem angle_HAB_eq_angle_OAC :
  ∠HAB = ∠OAC :=
sorry

end angle_HAB_eq_angle_OAC_l560_560459


namespace lcm_and_sum_of_14_21_35_l560_560308

def lcm_of_numbers_and_sum (a b c : ℕ) : ℕ × ℕ :=
  (Nat.lcm (Nat.lcm a b) c, a + b + c)

theorem lcm_and_sum_of_14_21_35 :
  lcm_of_numbers_and_sum 14 21 35 = (210, 70) :=
  sorry

end lcm_and_sum_of_14_21_35_l560_560308


namespace helicopter_rental_cost_l560_560560

noncomputable def rentCost (hours_per_day : ℕ) (days : ℕ) (cost_per_hour : ℕ) : ℕ :=
  hours_per_day * days * cost_per_hour

theorem helicopter_rental_cost :
  rentCost 2 3 75 = 450 := 
by
  sorry

end helicopter_rental_cost_l560_560560


namespace expand_expression_l560_560663

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560663


namespace arithmetic_sequence_a8_l560_560112

variable (a : ℕ → ℝ)
variable (a2_eq : a 2 = 4)
variable (a6_eq : a 6 = 2)

theorem arithmetic_sequence_a8 :
  a 8 = 1 :=
sorry

end arithmetic_sequence_a8_l560_560112


namespace max_3_element_subsets_bound_l560_560969

variables (Z : Type*) [Fintype Z] (n : ℕ) [Fintype.card Z = n]

def max_3_element_subsets (Z : Finset (Finset Z)) : ℕ :=
  Finset.card Z

theorem max_3_element_subsets_bound (Z : Finset (Finset Z)) (h : ∀ s1 s2 ∈ Z, s1 ≠ s2 → Finset.card (s1 ∩ s2) = 1) :
  max_3_element_subsets Z ≤ (n - 1) / 2 :=
sorry

end max_3_element_subsets_bound_l560_560969


namespace expand_expression_l560_560665

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560665


namespace right_angled_triangle_exists_l560_560626

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_right_angled_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem right_angled_triangle_exists :
  is_triangle 3 4 5 ∧ is_right_angled_triangle 3 4 5 :=
by
  sorry

end right_angled_triangle_exists_l560_560626


namespace find_m_n_l560_560396

noncomputable def f (x : ℝ) : ℝ := abs (real.log x / real.log 4)

theorem find_m_n (m n : ℝ) (hm0 : 0 < m) (hn0 : 0 < n) (h_ord : m < n) (h_feq : f m = f n) (h_max : ∀ x ∈ set.Icc (m^5) n, f x ≤ 5) :
  m = 1/4 ∧ n = 4 :=
sorry

end find_m_n_l560_560396


namespace failing_student_cheated_fraction_l560_560108

theorem failing_student_cheated_fraction {n : ℕ} (H1 : n > 0) 
    (H2 : ∀ q, q < n → q lies in the interval {1, 2, 3, 4, 5}) 
    (H3 : ∀ q, q < n → top_student q = correct_answer)
    (H4 : failing_student_correct n = n / 2) 
    (H5 : ∀ g, g = guessed_questions → guessed_correct_answers = g / 5) 
    (H6 : ∀ g, g = n - copied_questions → 4 / 5 * g = n / 2) :
  ∃ s, s = copied_questions ∧ s / n = 3 / 8 := 
begin
  sorry
end

end failing_student_cheated_fraction_l560_560108


namespace lcm_of_ratios_l560_560543

theorem lcm_of_ratios (x : ℕ) (h_r : nat.coprime 5 13) (h_gcd : nat.gcd (5 * x) (13 * x) = 19) :
  nat.lcm (5 * 19) (13 * 19) = 1235 :=
by
  have h_x : x = 19,
  { -- Since gcd(5x, 13x) = 19 and gcd(5, 13) = 1, we deduce that x must be 19.
    sorry
  },
  rw h_x,
  calc
  nat.lcm (5 * 19) (13 * 19)
      = (5 * 19 * 13 * 19) / nat.gcd (5 * 19) (13 * 19) : sorry -- The formula of LCM
  ... = (5 * 19 * 13 * 19) / 19 : by rw h_gcd
  ... = (5 * 19 * 13) : by simp [mul_assoc, mul_comm, mul_left_comm]
  ... = 1235 : by norm_num

end lcm_of_ratios_l560_560543


namespace area_enclosed_by_curve_and_lines_l560_560015

theorem area_enclosed_by_curve_and_lines :
  let f := λ x : ℝ, e^(2 * x)
  let g := λ x : ℝ, 1 - x
  ∫ x in 0..1, (f x - g x) = (1 / 2) * exp 2 - 1 :=
by
  let f := λ x : ℝ, e^(2 * x)
  let g := λ x : ℝ, 1 - x
  have h : ∫ x in 0..1, (f x - g x) = (1 / 2) * exp 2 - 1 := sorry
  exact h

end area_enclosed_by_curve_and_lines_l560_560015


namespace sales_tax_difference_l560_560952

theorem sales_tax_difference :
  let price_before_tax := 40
  let tax_rate_8_percent := 0.08
  let tax_rate_7_percent := 0.07
  let sales_tax_8_percent := price_before_tax * tax_rate_8_percent
  let sales_tax_7_percent := price_before_tax * tax_rate_7_percent
  sales_tax_8_percent - sales_tax_7_percent = 0.4 := 
by
  sorry

end sales_tax_difference_l560_560952


namespace isosceles_triangle_area_l560_560949

theorem isosceles_triangle_area (h p : ℝ) (a b c : ℝ):
  h = 10 ∧ p = 40 ∧ a = b ∧ (a + b + c = p) ∧ (c^2 = h^2 + (c/2)^2) → (10 * (c/2) = 75) :=
by
  -- Given conditions are provided as assumptions in the statement.
  intro h_eq p_eq a_eq_b perimeter_eq pythag_eq
  sorry

end isosceles_triangle_area_l560_560949


namespace negation_of_proposition_l560_560538

open Classical

theorem negation_of_proposition : (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) :=
by
  sorry

end negation_of_proposition_l560_560538


namespace unique_arithmetic_seq_l560_560811

noncomputable def arithmetic_problem (a : ℝ) (q : ℝ) :=
  a > 0 ∧
  let b1 := 1 + a;
      b2 := 2 + a * q;
      b3 := 3 + a * q^2 in
    (b2 - b1 = a * q - 1) ∧
    (b3 - b2 = a * q^2 - a * q - 1) ∧
    (b3 - a = a * q^2 + 2 - a) ∧
    (2 + a * q)^2 = (1 + a) * (3 + a * q^2) ∧
    (a * q^2 - 4 * a * q + 3 * a - 1 = 0) ∧
    (4 * a^2 + 4 * a > 0)

theorem unique_arithmetic_seq :
  ∃ a : ℝ, ∃ q : ℝ, arithmetic_problem a q ∧ a = 1 / 3 :=
by
  sorry

end unique_arithmetic_seq_l560_560811


namespace const_angle_SPM_l560_560598

open Real Geometry

-- Given conditions
variables {O A B S T P M : Point}
variable (C : Circle)
variable (hemicircle : ∀ x, IsHemicircle C A B O)

-- Definitions
def isChord (C : Circle) (S T : Point) : Prop := C.contains S ∧ C.contains T ∧ distance S T = constant_length
def isMidpoint (M S T : Point) : Prop := distance S M = distance M T
def isPerpendicularFoot (P S A B : Point) : Prop := isFootOfPerpendicular P S A B

-- Proposition to prove
theorem const_angle_SPM 
  (h_chord : isChord C S T)
  (h_midpoint : isMidpoint M S T)
  (h_perpendicular : isPerpendicularFoot P S A B) :
  ∃ θ : ℝ, ∀ S T (xl : IsChord xl), angle S P M = θ := sorry

end const_angle_SPM_l560_560598


namespace sum_of_squares_mul_l560_560845

theorem sum_of_squares_mul (a b c d : ℝ) :
(a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 :=
by
  sorry

end sum_of_squares_mul_l560_560845


namespace simplify_fraction_l560_560508

theorem simplify_fraction (k : ℝ) : 
  (∃ a b : ℝ, (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a = 1 ∧ b = 3 ∧ (a / b) = 1/3) := by
  sorry

end simplify_fraction_l560_560508


namespace initial_number_of_men_l560_560183

theorem initial_number_of_men (n : ℕ) (A : ℕ)
  (h1 : 2 * n = 16)
  (h2 : 60 - 44 = 16)
  (h3 : 60 = 2 * 30)
  (h4 : 44 = 21 + 23) :
  n = 8 :=
by
  sorry

end initial_number_of_men_l560_560183


namespace flowers_on_porch_l560_560908

-- Definitions based on problem conditions
def total_plants : ℕ := 80
def flowering_percentage : ℝ := 0.40
def fraction_on_porch : ℝ := 0.25
def flowers_per_plant : ℕ := 5

-- Theorem statement
theorem flowers_on_porch (h1 : total_plants = 80)
                         (h2 : flowering_percentage = 0.40)
                         (h3 : fraction_on_porch = 0.25)
                         (h4 : flowers_per_plant = 5) :
    (total_plants * seminal (flowering_percentage * fraction_on_porch) * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560908


namespace expand_expression_l560_560714

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560714


namespace chord_length_l560_560034

noncomputable theory

variables (x1 x2 y1 y2 : ℝ)

def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def chord_through_focus (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 6

theorem chord_length (x1 x2 : ℝ)
  (hx1 : is_on_parabola x1 y1)
  (hx2 : is_on_parabola x2 y2)
  (hchord : chord_through_focus x1 x2) :
  |x1 - x2 + sqrt (1 / 4 * (4 * x2 + y2^2))| = 8 :=
sorry

end chord_length_l560_560034


namespace find_a2_l560_560767

noncomputable def a : ℕ → ℝ
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a2
  (a : ℕ → ℝ)
  (h1 : is_arithmetic a)
  (h2 : all_positive a)
  (h3 : a 3 * a 7 = 2 * (a 4)^2)
  (h4 : a 3 = 1):
  a 2 = sqrt 2 / 2 :=
sorry

end find_a2_l560_560767


namespace prob_interval_l560_560055

noncomputable def xi : ℝ → ℝ := sorry

variables (σ : ℝ) (h1 : 0 < σ) (h2 : ∀ x : ℝ, ℙ(0) ≤ 0.2)

theorem prob_interval (h1 : ∀ x : ℝ, xi x = Normal(2, σ^2)) (h2 : ℙ(λ ξ, ξ <= 0) = 0.2) :
  ℙ(λ ξ, 2 < ξ ∧ ξ ≤ 4) = 0.3 :=
sorry

end prob_interval_l560_560055


namespace son_age_is_15_l560_560097

theorem son_age_is_15 (S F : ℕ) (h1 : 2 * S + F = 70) (h2 : 2 * F + S = 95) (h3 : F = 40) :
  S = 15 :=
by {
  sorry
}

end son_age_is_15_l560_560097


namespace sixth_root_of_large_number_l560_560327

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l560_560327


namespace negation_of_all_teachers_love_math_l560_560409

-- Definitions of the given statements
def all_students_love_math : Prop := ∀ s, student s → loves_math s
def some_students_love_math : Prop := ∃ s, student s ∧ loves_math s
def no_teachers_dislike_math : Prop := ∀ t, teacher t → ¬ dis_likes_math t
def all_teachers_enjoy_math : Prop := ∀ t, teacher t → enjoys_math t
def at_least_one_teacher_dislikes_math : Prop := ∃ t, teacher t ∧ dis_likes_math t
def all_teachers_love_math : Prop := ∀ t, teacher t → loves_math t

-- Main proof problem statement
theorem negation_of_all_teachers_love_math : at_least_one_teacher_dislikes_math = ¬ all_teachers_love_math :=
sorry

end negation_of_all_teachers_love_math_l560_560409


namespace inverse_cubed_l560_560134

-- Definition of the function f
def f (x : ℝ) : ℝ := 25 / (7 + 4 * x)

-- The main theorem we need to prove
theorem inverse_cubed :
  (f^{-1} 3) ^ -3 = 27 := sorry

end inverse_cubed_l560_560134


namespace intersection_point_of_lines_PQ_RS_l560_560103

def point := ℝ × ℝ × ℝ

def P : point := (4, -3, 6)
def Q : point := (1, 10, 11)
def R : point := (3, -4, 2)
def S : point := (-1, 5, 16)

theorem intersection_point_of_lines_PQ_RS :
  let line_PQ (u : ℝ) := (4 - 3 * u, -3 + 13 * u, 6 + 5 * u)
  let line_RS (v : ℝ) := (3 - 4 * v, -4 + 9 * v, 2 + 14 * v)
  ∃ u v : ℝ,
    line_PQ u = line_RS v →
    line_PQ u = (19 / 5, 44 / 3, 23 / 3) :=
by
  sorry

end intersection_point_of_lines_PQ_RS_l560_560103


namespace option1_cost_option2_cost_compare_costs_at_30_combined_cost_effective_l560_560449

-- Define the constants
def teapot_price : ℝ := 90
def teacup_price : ℝ := 25
def discount_rate : ℝ := 0.9

-- Define the cost expressions
def cost_option1 (x : ℝ) : ℝ := 25 * x + 325
def cost_option2 (x : ℝ) : ℝ := 22.5 * x + 405

-- Theorem 1: Verify the cost expressions for different options
theorem option1_cost (x : ℝ) : cost_option1 x = 25 * x + 325 := by
  sorry

theorem option2_cost (x : ℝ) : cost_option2 x = 22.5 * x + 405 := by
  sorry

-- Theorem 2: Compare the costs at x = 30 
theorem compare_costs_at_30 : 
  let x := 30 in cost_option1 x < cost_option2 x := by
  sorry

-- Theorem 3: Calculate the combined cost-effective purchasing method for x = 30
theorem combined_cost_effective : 
  let x := 30 in
  5 * teapot_price + 25 * teacup_price * discount_rate < cost_option1 x := by
  sorry

-- Load conditions to avoid again defining them during proof
variable (teapot_price teacup_price discount_rate)

end option1_cost_option2_cost_compare_costs_at_30_combined_cost_effective_l560_560449


namespace angle_between_vectors_eq_pi_div_two_l560_560787

variables (e1 e2 : ℝ^3)
hypothesis h1 : ∥e1∥ = 1
hypothesis h2 : ∥e2∥ = 1
hypothesis h3 : real.angle e1 e2 = π / 3

theorem angle_between_vectors_eq_pi_div_two : real.angle (e1 - 2 • e2) e1 = π / 2 :=
sorry

end angle_between_vectors_eq_pi_div_two_l560_560787


namespace sqrt_log_expr_eq_l560_560240

-- Define logarithm and base change properties
noncomputable def log_change_base (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : ℝ := real.log a / real.log b

-- Define the square root of the expression
noncomputable def log_sqrt_expr : ℝ :=
  real.sqrt (log_change_base 8 4 (by norm_num) (by norm_num) - log_change_base 16 8 (by norm_num) (by norm_num))

-- Define the final statement to prove
theorem sqrt_log_expr_eq : log_sqrt_expr = 1 / real.sqrt 6 :=
  sorry

end sqrt_log_expr_eq_l560_560240


namespace range_of_a_l560_560772

variable {x a : ℝ}

noncomputable def p : Prop := ∀ x ∈ (Set.Icc 1 2), x^2 + a * x - 2 > 0
noncomputable def q : Prop := 
  (∀ x ∈ (Set.Ici 1), (x^2 - 2 * a * x + 3 * a)' > 0) ∧ (∀ x ∈ (Set.Ici 1), x^2 - 2 * a * x + 3 * a > 0)

theorem range_of_a (h : p ∨ q) : a > -1 :=
sorry

end range_of_a_l560_560772


namespace initial_ratio_of_stamps_l560_560206

theorem initial_ratio_of_stamps (P Q : ℕ) (h1 : ((P - 8 : ℤ) : ℚ) / (Q + 8) = 6 / 5) (h2 : P - 8 = Q + 8) : P / Q = 6 / 5 :=
sorry

end initial_ratio_of_stamps_l560_560206


namespace g_value_l560_560366

noncomputable def f : ℝ → ℝ := sorry

theorem g_value :
  f 1 = 1 ∧ (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧ (∀ x : ℝ, f (x + 1) ≤ f x + 1) →
  (let g := λ x : ℝ, f x + 1 - x in g 2009 = 1) := sorry

end g_value_l560_560366


namespace find_a_l560_560394

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log2 (x^2 + a)

theorem find_a (a : ℝ) : (f 3 a = 1) → (a = -7) :=
by 
  sorry

end find_a_l560_560394


namespace minimum_value_h_at_a_eq_2_range_of_a_l560_560061

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x

theorem minimum_value_h_at_a_eq_2 : ∃ x, h 2 x = 3 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 1, h a x ≥ 1) ↔ a ≥ 1 :=
sorry

end minimum_value_h_at_a_eq_2_range_of_a_l560_560061


namespace function_equivalence_l560_560731

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (-y) = -g y) ∧ (∀ x : ℝ, f x = g (1 - 2 * x^2) + 1010) :=
sorry

end function_equivalence_l560_560731


namespace segment_PB_measure_l560_560436

variable (y : ℝ)
variable (C D B M P : Type)
variable [AddGroup M]

theorem segment_PB_measure (h_midpoint : M = midpoint (arc C D B)) 
  (h_perpendicular : MP ⟂ DB ∧ P ∈ MP)
  (h_CD_measure : distance C D = y)
  (h_DP_measure : distance D P = y + 3)
  (h_CP_measure : distance C P = 2 * (y + 3)) :
  distance P B = y + 3 := 
sorry

end segment_PB_measure_l560_560436


namespace monotonic_and_extremes_l560_560403

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem monotonic_and_extremes :
  (∀ x, (-∞ < x ∧ x < -2 ∨ 0 < x ∧ x < ∞) → 0 < (x * (x + 2)) * Real.exp x) ∧
  (∀ x, (-2 < x ∧ x < 0) → (x * (x + 2)) * Real.exp x < 0) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Icc (-1 : ℝ) 2, f y ≥ f x ∧ f x = 0) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Icc (-1 : ℝ) 2, f y ≤ f x ∧ f x = 4 * Real.exp 2 ^ 2) :=
by
  sorry

end monotonic_and_extremes_l560_560403


namespace calories_per_pound_of_body_fat_l560_560448

theorem calories_per_pound_of_body_fat (gained_weight : ℕ) (calories_burned_per_day : ℕ) 
  (days_to_lose_weight : ℕ) (calories_consumed_per_day : ℕ) : 
  gained_weight = 5 → 
  calories_burned_per_day = 2500 → 
  days_to_lose_weight = 35 → 
  calories_consumed_per_day = 2000 → 
  (calories_burned_per_day * days_to_lose_weight - calories_consumed_per_day * days_to_lose_weight) / gained_weight = 3500 :=
by 
  intros h1 h2 h3 h4
  sorry

end calories_per_pound_of_body_fat_l560_560448


namespace gas_volume_at_temp_10_l560_560742

theorem gas_volume_at_temp_10 (V₀ : ℝ) (T₀ T₁ : ℝ) (k : ℝ) 
  (T₀_eq : T₀ = 40) (V₀_eq : V₀ = 36) 
  (temp_relation : T₁ = T₀ - 5 * k) (vol_relation : ∀ k, V₀ - 6 * k):
  T₁ = 10 → vol_relation 6 = 0 :=
by
    -- The proof will go here
    sorry

end gas_volume_at_temp_10_l560_560742


namespace inverse_proportion_inequality_l560_560769

theorem inverse_proportion_inequality 
  (k : ℝ) (h : k < 0) (y1 y2 y3 : ℝ) :
  y1 = k / (-3) →
  y2 = k / (-2) →
  y3 = k / 3 →
  y3 < y1 ∧ y1 < y2 :=
by {
  intros h1 h2 h3,
  -- Placeholder to represent steps of the proof
  sorry
}

end inverse_proportion_inequality_l560_560769


namespace exists_perfect_square_ends_in_23456_l560_560342

theorem exists_perfect_square_ends_in_23456 : ∃ n : ℕ, (n * n) % 100000 = 23456 :=
begin
  sorry
end

end exists_perfect_square_ends_in_23456_l560_560342


namespace distance_from_Q_to_EH_l560_560946

-- Definitions based on the conditions
structure Square where
  E F G H : (ℝ × ℝ)
  side_length : ℝ
  square_def : (dist E F = side_length) ∧ 
               (dist F G = side_length) ∧ 
               (dist G H = side_length) ∧ 
               (dist H E = side_length)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def circle (center : ℝ × ℝ) (radius : ℝ) (P : ℝ × ℝ) : Prop :=
  (dist center P = radius)

-- Main problem
theorem distance_from_Q_to_EH :
  ∀ (EFGH : Square) (N Q : ℝ × ℝ),
  EFGH.side_length = 6 →
  N = midpoint EFGH.G EFGH.H →
  circle N 3 Q →
  circle EFGH.E 6 Q →
  Q.2 = 6 / 5 :=
sorry

end distance_from_Q_to_EH_l560_560946


namespace middle_part_of_sum_is_40_l560_560292

theorem middle_part_of_sum_is_40 (x : ℕ) (h1 : 15 * x = 120) :
  5 * x = 40 :=
by
  have h2 : x = 8 := by linarith [h1]
  rw [h2]
  norm_num

end middle_part_of_sum_is_40_l560_560292


namespace infinitely_many_planes_through_collinear_points_l560_560610

noncomputable def point : Type := ℝ × ℝ × ℝ

def collinear (a b c : point) : Prop :=
  ∃ (t1 t2 : ℝ), b = (a.1 + t1 * (c.1 - a.1), a.2 + t1 * (c.2 - a.2), a.3 + t1 * (c.3 - a.3)) ∧
                 c = (a.1 + t2 * (c.1 - a.1), a.2 + t2 * (c.2 - a.2), a.3 + t2 * (c.3 - a.3))

def plane : Type := {n : ℝ × ℝ × ℝ // n ≠ (0,0,0)}

def passes_through (p : point) (π : plane) : Prop :=
  let n := π.val in
  ∀ (x y z : ℝ), p = (x, y, z) → n.1 * x + n.2 * y + n.3 * z = 0

theorem infinitely_many_planes_through_collinear_points
  (a b c : point) (h_collinear : collinear a b c) : 
  ∃ (infinitely_many : set plane), 
    (∀ π ∈ infinitely_many, passes_through a π ∧ passes_through b π ∧ passes_through c π) ∧
    ∀ (π₁ π₂ : plane), π₁ ≠ π₂ → (π₁ ∈ infinitely_many ∧ π₂ ∈ infinitely_many) :=
sorry

end infinitely_many_planes_through_collinear_points_l560_560610


namespace real_part_of_z_l560_560372

open Complex -- Open the Complex namespace to work with complex numbers

-- Define the main problem
theorem real_part_of_z (z : ℂ) (h : z * (1 - I) = |1 - Real.sqrt 3 * I| + I) : z.re = 1 / 2 :=
sorry

end real_part_of_z_l560_560372


namespace golden_section_PB_l560_560383

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem golden_section_PB {A B P : ℝ} (h1 : P = (1 - 1/(golden_ratio)) * A + (1/(golden_ratio)) * B)
  (h2 : AB = 2)
  (h3 : A ≠ B) : PB = 3 - Real.sqrt 5 :=
by
  sorry

end golden_section_PB_l560_560383


namespace portion_of_profit_divided_equally_correct_l560_560917

noncomputable def portion_of_profit_divided_equally
  (mary_investment : ℕ)
  (mike_investment : ℕ)
  (total_profit : ℕ)
  (mary_more_than_mike : ℕ) : ℕ := 
  let e := 1000 in
  e

theorem portion_of_profit_divided_equally_correct :
  ∀ (mary_investment mike_investment total_profit : ℕ) (mary_more_than_mike : ℕ),
  mary_investment = 700 →
  mike_investment = 300 →
  total_profit = 3000 →
  mary_more_than_mike = 800 →
  portion_of_profit_divided_equally mary_investment mike_investment total_profit mary_more_than_mike = 1000 :=
by
  intros
  rw [portion_of_profit_divided_equally]
  sorry

end portion_of_profit_divided_equally_correct_l560_560917


namespace complex_conjugate_power_l560_560838

theorem complex_conjugate_power (z : ℂ) (h : z = (1 + I) / (1 - I)) : (conj z)^2017 = -I :=
by {
  sorry
}

end complex_conjugate_power_l560_560838


namespace total_amount_spent_l560_560873

-- Define the prices related to John's Star Wars toy collection
def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

-- Problem statement in Lean: Prove the total amount spent is $3000
theorem total_amount_spent : (other_toys_cost + lightsaber_cost) = 3000 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end total_amount_spent_l560_560873


namespace expand_polynomial_l560_560682

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560682


namespace floor_of_fraction_expression_l560_560646

theorem floor_of_fraction_expression : 
  ( ⌊ (2025^3 / (2023 * 2024)) - (2023^3 / (2024 * 2025)) ⌋ ) = 8 :=
sorry

end floor_of_fraction_expression_l560_560646


namespace problem_statement_l560_560343

theorem problem_statement :
  ∀ (n : ℕ) (hn : n > 3) (A : fin n → ℝ × ℝ) (r : fin n → ℝ),
  (∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k →
    ¬ collinear ℝ {((A i).1, (A i).2), ((A j).1, (A j).2), ((A k).1, (A k).2)}) →
  (∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k →
    (triangle_area (A i) (A j) (A k)) = r i + r j + r k) →
  n = 4 :=
by
  sorry

end problem_statement_l560_560343


namespace parabola_directrix_l560_560017

theorem parabola_directrix (x : ℝ) :
  (y = (x^2 - 8 * x + 12) / 16) →
  (∃ y, y = -17/4) :=
by
  intro h
  sorry

end parabola_directrix_l560_560017


namespace binomial_sum_of_coefficients_l560_560424

theorem binomial_sum_of_coefficients (n : ℕ) (h₀ : (1 - 2)^n = 8) :
  (1 - 2)^n = -1 :=
sorry

end binomial_sum_of_coefficients_l560_560424


namespace sum_of_digits_878_base8_l560_560239

def sum_base8_digits (n : ℕ) : ℕ :=
  let digits := n.digits 8
  digits.sum

theorem sum_of_digits_878_base8 :
  sum_base8_digits 878 = 17 :=
by {
  sorry
}

end sum_of_digits_878_base8_l560_560239


namespace sum_series_equals_one_l560_560642

theorem sum_series_equals_one :
  ∑' n : ℕ, n ≥ 2 → (4 * ↑n^3 - ↑n^2 - ↑n + 1) / (↑n^6 - ↑n^5 + ↑n^4 - ↑n^3 + ↑n^2 - ↑n) = 1 :=
begin
  sorry
end

end sum_series_equals_one_l560_560642


namespace prob1_prob2_l560_560591

-- Define complex number operations in Lean
noncomputable def complex_one_over (z : ℂ) : ℂ := 1 / z

-- Problem 1 statement
theorem prob1 : (complex_one_over (1 - complex.I) + complex_one_over (2 + 3 * complex.I)) = (17 / 26 : ℂ) + (7 / 26) * complex.I := 
by
  sorry

-- Variables for problem 2
def z1 : ℂ := 3 + 4 * complex.I
variable (z2 : ℂ)

-- Problem 2 conditions
axiom abs_z2 : complex.abs z2 = 5
axiom purely_imaginary : z1 * z2.imaginary

-- Problem 2 statement
theorem prob2 : z2 = 4 + 3 * complex.I ∨ z2 = -4 - 3 * complex.I := 
by
  sorry

end prob1_prob2_l560_560591


namespace sin_2alpha_over_cos_alpha_sin_beta_value_l560_560380

variable (α β : ℝ)

-- Given conditions
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < Real.pi / 2
axiom beta_pos : 0 < β
axiom beta_lt_pi_div_2 : β < Real.pi / 2
axiom cos_alpha_eq : Real.cos α = 3 / 5
axiom cos_beta_plus_alpha_eq : Real.cos (β + α) = 5 / 13

-- The results to prove
theorem sin_2alpha_over_cos_alpha : (Real.sin (2 * α) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 12) :=
sorry

theorem sin_beta_value : (Real.sin β = 16 / 65) :=
sorry


end sin_2alpha_over_cos_alpha_sin_beta_value_l560_560380


namespace solve_log_eq_l560_560943

noncomputable def solution_set := {x : ℝ | ∃ (a: ℝ), a ∈ {5 * (5 + 3 * Real.sqrt 5), 5 * (5 - 3 * Real.sqrt 5)} ∧ x = a}

theorem solve_log_eq:
  ∀ x : ℝ, (log 5 (x^2 - 25 * x) = 3) → x ∈ solution_set :=
by
  sorry

end solve_log_eq_l560_560943


namespace tetrahedron_distance_height_relation_l560_560897

variable (A B C D P : Type)
variable (p_a p_b p_c p_d m_a m_b m_c m_d : ℝ)

theorem tetrahedron_distance_height_relation 
  (tetrahedron: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ interior P) : 
  (p_a / m_a + p_b / m_b + p_c / m_c + p_d / m_d = 1) :=
by
  sorry

end tetrahedron_distance_height_relation_l560_560897


namespace solution_set_of_floor_inequality_l560_560021

theorem solution_set_of_floor_inequality (x : ℝ) :
  (⟨ ∃ ⟨ n : ℤ, x ≤ ↑n ∧ (n : ℝ) ^ 2 - n - 12 ≤ 0 ⟩ ⟩)
  ↔ (-3 : ℝ) ≤ x ∧ x < 5 :=
by sorry

end solution_set_of_floor_inequality_l560_560021


namespace frequency_of_eighth_group_l560_560114

theorem frequency_of_eighth_group (A : ℝ) (hA_pos : 0 < A) (sample_size : ℝ)
  (h_sample_size : sample_size = 200) :
  let area_eighth := (1 / 4) * A in
  let total_area := A + area_eighth in
  let frequency_eighth := (area_eighth / total_area) * sample_size in
  frequency_eighth = 40 :=
by
  sorry

end frequency_of_eighth_group_l560_560114


namespace area_triangle_ACF_l560_560500

variables {A B C D E F : Type*} [geometry.Point A] [geometry.LineSegment B C D E F]

-- Definitions for midpoints and parallelogram conditions
def is_midpoint (p q : B) (x : E) : Prop := dist p x = dist x q
def is_parallelogram (abcd : geometry.Quadrilateral A B C D) : Prop :=
  ∃ (a b c d : A), geometry.Quadrilateral.mk a b c d = abcd ∧
  ∀ (a b c d : A), geometry.is_parallel a b c d ∧ geometry.is_parallel b c d a ∧ geometry.is_parallel c d a b ∧ geometry.is_parallel d a b c

-- Given conditions
theorem area_triangle_ACF 
  (abcd : geometry.Quadrilateral A B C D) 
  (h_parallelogram : is_parallelogram abcd) 
  (area_abcd : geometry.area abcd = 48)
  (h_mid_E : is_midpoint (geometry.LineSegment A B) E)
  (h_mid_F : is_midpoint (geometry.LineSegment C D) F): 
  geometry.area (geometry.Triangle A C F) = 12 := 
sorry

end area_triangle_ACF_l560_560500


namespace solve_for_z_l560_560510

theorem solve_for_z (z i : ℂ) (h1 : 1 - i*z + 3*i = -1 + i*z + 3*i) (h2 : i^2 = -1) : z = -i := 
  sorry

end solve_for_z_l560_560510


namespace max_true_statements_l560_560890

theorem max_true_statements (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) : 
  ∃ (S : set (Prop)), 
  ({ ((1:ℝ) / a > (1:ℝ) / b), (a ^ 2 > b ^ 2), (a > b), (|a| > 1), (b < 1) } ⊆ S ∧ S ⊆ { (1:ℝ) / a > (1:ℝ) / b, a^2 > b^2, a > b, |a| > 1, b < 1 } ∧ S.card = 4) :=
sorry

end max_true_statements_l560_560890


namespace percentage_gain_is_16_67_l560_560202

-- Defining the constants
def manufacturing_cost (shoe : Type) : ℝ := 190
def transportation_cost (shoe : Type) : ℝ := 500 / 100
def selling_price (shoe : Type) : ℝ := 234

-- Calculating the total cost per shoe
def total_cost (shoe : Type) : ℝ := manufacturing_cost shoe + transportation_cost shoe

-- Calculating the gain per shoe
def gain (shoe : Type) : ℝ := selling_price shoe - total_cost shoe

-- Calculating the percentage gain on the selling price
def percentage_gain (shoe : Type) : ℝ := (gain shoe / selling_price shoe) * 100

-- Proof statement, which says the percentage gain is 16.67%
theorem percentage_gain_is_16_67 (shoe : Type) : percentage_gain shoe = 16.67 := 
by sorry

end percentage_gain_is_16_67_l560_560202


namespace range_of_f_sin_double_theta_l560_560402

noncomputable def f (x : ℝ) : ℝ := real.sqrt 2 * real.cos (x + real.pi / 4)

theorem range_of_f :
  ∀ (x : ℝ), x ∈ set.Icc (-real.pi / 2) (real.pi / 2) →
    f x ∈ set.Icc (-1 : ℝ) (real.sqrt 2) := 
begin
  sorry
end

theorem sin_double_theta :
  ∀ (θ : ℝ), θ ∈ set.Ioo 0 (real.pi / 2) →
    f θ = 1 / 2 →
      real.sin (2 * θ) = 3 / 4 := 
begin
  sorry
end

end range_of_f_sin_double_theta_l560_560402


namespace option_a_incorrect_option_b_correct_option_c_correct_option_d_correct_l560_560244

theorem option_a_incorrect (x : ℝ) : -x + 5 ≠ -(x + 5) := sorry

theorem option_b_correct (m n : ℝ) : -7 * m - 2 * n = -(7 * m + 2 * n) := sorry

theorem option_c_correct (a : ℝ) : a^2 - 3 = +(a^2 - 3) := sorry

theorem option_d_correct (x y : ℝ) : 2 * x - y = -(y - 2 * x) := sorry

end option_a_incorrect_option_b_correct_option_c_correct_option_d_correct_l560_560244


namespace donut_cubes_eaten_l560_560492

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end donut_cubes_eaten_l560_560492


namespace fg_minus_gf_l560_560513

-- Definitions provided by the conditions
def f (x : ℝ) : ℝ := 4 * x + 8
def g (x : ℝ) : ℝ := 2 * x - 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -17 := 
  sorry

end fg_minus_gf_l560_560513


namespace lydia_flowers_on_porch_l560_560916

theorem lydia_flowers_on_porch:
  ∀ (total_plants : ℕ) (flowering_percentage : ℚ) (fraction_on_porch : ℚ) (flowers_per_plant : ℕ),
  total_plants = 80 →
  flowering_percentage = 0.40 →
  fraction_on_porch = 1 / 4 →
  flowers_per_plant = 5 →
  let flowering_plants := (total_plants : ℚ) * flowering_percentage in
  let porch_plants := flowering_plants * fraction_on_porch in
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ) in
  total_flowers_on_porch = 40 :=
by {
  intros total_plants flowering_percentage fraction_on_porch flowers_per_plant,
  intros h1 h2 h3 h4,
  let flowering_plants := (total_plants : ℚ) * flowering_percentage,
  let porch_plants := flowering_plants * fraction_on_porch,
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ),
  sorry
}

end lydia_flowers_on_porch_l560_560916


namespace B_pow_100_eq_B_l560_560879

def B : Matrix (Fin 3) (Fin 3) ℝ := !![ 
  [0, 1, 0], 
  [0, 0, 1], 
  [1, 0, 0] 
]

theorem B_pow_100_eq_B : B ^ 100 = B := 
by 
  sorry

end B_pow_100_eq_B_l560_560879


namespace proof_theme_l560_560858

noncomputable def curve_C (α : ℝ) : ℝ × ℝ := 
  (sqrt 2 * Real.cos α, Real.sin α)

def line_l_polar (ρ θ : ℝ) : Prop := 
  ρ * Real.cos (θ + π / 4) = sqrt 2 / 2

def line_l_cartesian (x y : ℝ) : Prop := 
  x - y - 1 = 0

def point_P : ℝ × ℝ := (2, 0)

def is_on_curve_C (p : ℝ × ℝ) : Prop := 
  p.1^2 / 2 + p.2^2 = 1

def is_on_line_l (p : ℝ × ℝ) : Prop := 
  p.1 - p.2 - 1 = 0

def is_symmetric_wrt_x_axis (A B P : ℝ × ℝ) : Prop :=
  let k1 := (A.2 - P.2) / (A.1 - P.1)
  let k2 := (B.2 - P.2) / (B.1 - P.1)
  k1 + k2 = 0

theorem proof_theme :
  ∀ α ρ θ x y A B,
    (curve_C α = (x, y)) →
    line_l_polar ρ θ →
    is_on_curve_C A →
    is_on_curve_C B →
    is_on_line_l A →
    is_on_line_l B →
    is_symmetric_wrt_x_axis A B point_P :=
by
  sorry

end proof_theme_l560_560858


namespace range_of_a_l560_560069

def f (x : ℝ) : ℝ := Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
def h (a : ℝ) (x : ℝ) : ℝ := f x - g a x

theorem range_of_a (a : ℝ) (h_mono_incr :  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → (1 / x) - a * x - 2 ≥ 0) : a ≤ -1 :=
by
  sorry

end range_of_a_l560_560069


namespace part1_part2_l560_560797

-- Definition of f(x)
def f (x : ℝ) := (1 + x) / (1 - x)

-- Part (1): f(x) is monotonically increasing in the interval (1, +∞)
theorem part1 (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : 1 < x2) (h3 : x1 > x2) : 
  f(x1) > f(x2) := 
by
  sorry

-- Definition of g(x)
def g (x : ℝ) := Real.log2 (f(x))

-- Part (2): Find the range of m such that g(x) > (1/2)^x + m for x in [3, 4]
theorem part2 (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 4) (m : ℝ) : 
  g(x) > (1/2)^x + m → m < 7/8 := 
by
  sorry

end part1_part2_l560_560797


namespace find_x_l560_560369

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l560_560369


namespace second_discount_l560_560253

theorem second_discount (list_price : ℝ) (paid_price : ℝ) (first_discount : ℝ) :
  list_price = 68 ∧ paid_price = 56.16 ∧ first_discount = 0.1 → 
  ∃ second_discount : ℝ, second_discount ≈ 0.0824 :=
by
  sorry

end second_discount_l560_560253


namespace modulus_of_z_l560_560071

variables (i : ℂ) (z : ℂ)
noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def defined_z : ℂ := imaginary_unit * (2 - imaginary_unit)

theorem modulus_of_z : Complex.abs defined_z = Real.sqrt 5 := sorry

end modulus_of_z_l560_560071


namespace imaginary_part_of_complex_l560_560532

noncomputable def complex_imaginary_part : Prop :=
  let z := (2 * Complex.I) / (2 + Complex.I^3)
  Complex.im z = 4 / 5

theorem imaginary_part_of_complex : complex_imaginary_part := 
by
  let z := (2 * Complex.I) / (2 + Complex.I^3)
  have : z = (-2 + 4 * Complex.I) / 5 := sorry
  show Complex.im z = 4 / 5 from sorry

end imaginary_part_of_complex_l560_560532


namespace f_increasing_when_a_gt_2_range_of_a_for_two_zeros_l560_560060

-- Definitions
def f (x a : ℝ) : ℝ := 2 * |x + 1| + a * x
def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Question 1
theorem f_increasing_when_a_gt_2 (a : ℝ) (h : a > 2) : 
  increasing (λ x, f x a) := 
sorry

-- Question 2
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) → 
  0 < a ∧ a < 2 := 
sorry

end f_increasing_when_a_gt_2_range_of_a_for_two_zeros_l560_560060


namespace kara_forgot_medication_times_l560_560453

theorem kara_forgot_medication_times :
  let ounces_per_medication := 4
  let medication_times_per_day := 3
  let days_per_week := 7
  let total_weeks := 2
  let total_water_intaken := 160
  let expected_total_water := (ounces_per_medication * medication_times_per_day * days_per_week * total_weeks)
  let water_difference := expected_total_water - total_water_intaken
  let forget_times := water_difference / ounces_per_medication
  forget_times = 2 := by sorry

end kara_forgot_medication_times_l560_560453


namespace smallest_positive_angle_l560_560089

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : ∃ β : ℝ, 0 < β ∧ β < 360 ∧ β = α % 360 := by
  sorry

end smallest_positive_angle_l560_560089


namespace value_of_each_bill_l560_560151

theorem value_of_each_bill 
  (total_money : ℕ) 
  (num_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : num_bills = 9) : 
  total_money / num_bills = 5 := 
by 
  rw [h1, h2] 
  norm_num
  -- or use exact 5 to provide the result directly
  -- exact 5

-- Adding sorry to skip proof

end value_of_each_bill_l560_560151


namespace period_amplitude_range_of_f_on_interval_l560_560398

def f (x : ℝ) : ℝ := sqrt 3 * sin x - cos x

theorem period_amplitude (T : ℝ) (A : ℝ) 
  (hT : T = 2 * π) (hA : A = 2) : ∃ T A, (T = 2 * π) ∧ (A = 2) :=
by {
  use [2 * π, 2],
  exact ⟨hT, hA⟩,
}

theorem range_of_f_on_interval 
  (h : ∀ x ∈ Icc (0 : ℝ) π, f x ∈ Icc (-1 : ℝ) 2) : 
  ∀ x ∈ Icc (0 : ℝ) π, f x ∈ Icc (-1 : ℝ) 2 :=
by {
  intro x,
  intro hx,
  exact h x hx,
}

example : ∃ T A, (T = 2 * π) ∧ (A = 2) := 
period_amplitude 2 * π 2 rfl rfl

example : ∀ x ∈ Icc (0 : ℝ) π, f x ∈ Icc (-1 : ℝ) 2 :=
range_of_f_on_interval (λ x hx, by sorry)

end period_amplitude_range_of_f_on_interval_l560_560398


namespace cathy_1010th_turn_l560_560640

-- Define Cathy's initial position
def initial_position : (ℤ × ℤ) := (15, -15)

-- Define the movement direction
inductive Direction
| west
| north
| east
| south

-- Function to get the next direction after turning 90° right
def next_direction : Direction → Direction
| Direction.west => Direction.north
| Direction.north => Direction.east
| Direction.east => Direction.south
| Direction.south => Direction.west

-- Function to get the new position after a move
def move_position (pos : ℤ × ℤ) (dir : Direction) (dist : ℕ) : (ℤ × ℤ) :=
  match dir with
  | Direction.west => (pos.1 - dist, pos.2)
  | Direction.north => (pos.1, pos.2 + dist)
  | Direction.east => (pos.1 + dist, pos.2)
  | Direction.south => (pos.1, pos.2 - dist)

-- Recursive function to calculate the position after n moves
def cathy_position : ℕ → ℤ × ℤ
| 0 => initial_position
| n + 1 => 
  let (pos, dir, dist) := (cathy_position n, next_direction (direction n), n + 1)
  move_position pos dir dist

-- Function to get the direction at step n
def direction (n : ℕ) : Direction :=
  match n % 4 with
  | 0 => Direction.west
  | 1 => Direction.north
  | 2 => Direction.east
  | 3 => Direction.south
  | _ => Direction.west  -- This case won't be reached due to % 4

-- Statement to prove
theorem cathy_1010th_turn : cathy_position 1010 = (-491, 489) := sorry

end cathy_1010th_turn_l560_560640


namespace smallest_number_of_consecutive_remainders_l560_560485

theorem smallest_number_of_consecutive_remainders (k : ℤ) (h : 15 * k + 21 = 336) :
  min (5 * k + 2) (min (5 * (k + 1) + 2) (5 * (k + 2) + 2)) = 107 :=
by
  have k_val : k = 21 := sorry
  have n1 : 5 * k + 2 = 5 * 21 + 2 := by rw [k_val]
  have n2 : 5 * (k + 1) + 2 = 5 * (21 + 1) + 2 := by rw [k_val]
  have n3 : 5 * (k + 2) + 2 = 5 * (21 + 2) + 2 := by rw [k_val]
  have n1_eval : 5 * 21 + 2 = 107 := by norm_num
  have n2_eval : 5 * 22 + 2 = 112 := by norm_num
  have n3_eval : 5 * 23 + 2 = 117 := by norm_num
  rw [n1, n2, n3, n1_eval, n2_eval, n3_eval]
  norm_num
  sorry

end smallest_number_of_consecutive_remainders_l560_560485


namespace find_areas_l560_560160

variables (A M N B D C : Point)
variables (AN AM MB NC : ℝ)

-- Conditions
def points_on_sides (A M N B D C : Point) :=
  dist A N = 11 ∧ dist N C = 39 ∧
  dist A M = 12 ∧ dist M B = 3 ∧
  A ≠ B ∧ A ≠ D ∧ B ≠ C ∧ D ≠ C

-- Rectangle definition
def is_rectangle (A B C D : Point) :=
  dist A B = dist C D ∧ dist B C = dist D A ∧
  dist A C = dist B D

-- Area of rectangle ABCD
def area_rectangle (A B C D : Point) :=
  dist A B * dist A D

-- Area of triangle MNC
def area_triangle (M N C : Point) :=
  0.5 * (dist M C * dist N C)

-- Problem statement
theorem find_areas (A B C D M N : Point) (AN AM MB NC : ℝ) :
  points_on_sides A M N B D C →
  is_rectangle A B C D →
  (area_rectangle A B C D = 750) ∧
  (area_triangle M N C = 234) :=
by
  sorry

end find_areas_l560_560160


namespace range_fn_11_l560_560971

def sum_of_digits (k : ℕ) : ℕ :=
  k.toString.foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def f1 (k : ℕ) : ℕ :=
  (sum_of_digits k)^2

def fn (n k : ℕ) : ℕ :=
  Nat.iterate n f1 k

theorem range_fn_11 : 
  (n : ℕ) → Set.mem (fn n 11) {4, 16, 49, 169, 256} :=
by
  sorry

end range_fn_11_l560_560971


namespace expand_polynomial_l560_560685

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560685


namespace integral_tan_inv_diff_eval_l560_560010

noncomputable def integral_tan_inv_diff : ℝ :=
  ∫ x in 0..∞, (arctan (π * x) - arctan x) / x

theorem integral_tan_inv_diff_eval : integral_tan_inv_diff = (π / 2) * log π :=
by
  sorry

end integral_tan_inv_diff_eval_l560_560010


namespace crease_length_l560_560281

noncomputable def length_of_crease (a b c : ℕ) (h : a * a + b * b = c * c) : ℝ :=
  c / 2

theorem crease_length (a b c : ℕ) (h : a * a + b * b = c * c) (habc : {a, b, c} = {6, 8, 10}) :
  length_of_crease a b c h = 5 :=
by
  sorry

end crease_length_l560_560281


namespace instantaneous_speed_at_four_seconds_l560_560522

-- Let s be a function representing the object's linear motion over time
variable (s : ℝ → ℝ)

-- Given condition: The derivative of s at t=4 is 10
axiom h1 : deriv s 4 = 10

-- Proof statement: The instantaneous speed of the object at the 4th second is 10m/s
theorem instantaneous_speed_at_four_seconds : deriv s 4 = 10 :=
by
  exact h1

end instantaneous_speed_at_four_seconds_l560_560522


namespace even_product_implies_even_factor_l560_560991

theorem even_product_implies_even_factor (a b : ℕ) (h : Even (a * b)) : Even a ∨ Even b :=
by
  sorry

end even_product_implies_even_factor_l560_560991


namespace increasing_sufficient_not_necessary_l560_560972

-- Conditions
def increasing_function (a : ℝ) (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y
def log2_gt_1 (a : ℝ) := real.log 2 a > 1

-- Statement to be proved
theorem increasing_sufficient_not_necessary {a : ℝ} : 
  (∃ (f : ℝ → ℝ), f = (λ x, a * x) ∧ increasing_function a f) ↔ 
  (log2_gt_1 a) ∧ (∃ b : ℝ, 0 < b ∧ a = b) :=
sorry

end increasing_sufficient_not_necessary_l560_560972


namespace min_value_of_reciprocals_l560_560133

theorem min_value_of_reciprocals (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) : 
  ∃ (x : ℝ), x = 2 * a + b ∧ ∃ (y : ℝ), y = 2 * b + c ∧ ∃ (z : ℝ), z = 2 * c + a ∧ (1 / x + 1 / y + 1 / z = 27 / 8) :=
sorry

end min_value_of_reciprocals_l560_560133


namespace probability_rain_sunday_monday_l560_560541

def P_rain_Sunday : ℝ := 0.30
def P_rain_Monday_given_Sunday : ℝ := 0.50

theorem probability_rain_sunday_monday (h1 : P_rain_Sunday = 0.30) 
                                       (h2 : P_rain_Monday_given_Sunday = 0.50) : 
  P_rain_Sunday * P_rain_Monday_given_Sunday = 0.15 :=
by sorry

end probability_rain_sunday_monday_l560_560541


namespace alex_ate_more_pears_than_sam_l560_560658

namespace PearEatingContest

def number_of_pears_eaten (Alex Sam : ℕ) : ℕ :=
  Alex - Sam

theorem alex_ate_more_pears_than_sam :
  number_of_pears_eaten 8 2 = 6 := by
  -- proof
  sorry

end PearEatingContest

end alex_ate_more_pears_than_sam_l560_560658


namespace find_reciprocal_sum_l560_560026

theorem find_reciprocal_sum (x y : ℝ) (h1 : 2 ^ x = 196) (h2 : 7 ^ y = 196) : 
  1 / x + 1 / y = 1 / 2 :=
by
  sorry -- placeholder for the proof

end find_reciprocal_sum_l560_560026


namespace solution_set_of_inequality_l560_560388

variable {a b x : ℝ}

theorem solution_set_of_inequality (h : ∃ y, y = 3*(-5) + a ∧ y = -2*(-5) + b) :
  (3*x + a < -2*x + b) ↔ (x < -5) :=
by sorry

end solution_set_of_inequality_l560_560388


namespace logarithmic_inequality_l560_560455

theorem logarithmic_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 1) :
  log c (log c b) + log b (log b a) + log a (log a c) ≥ 0 :=
sorry

end logarithmic_inequality_l560_560455


namespace evaluate_binom_mul_factorial_l560_560006

theorem evaluate_binom_mul_factorial (n : ℕ) (h : n > 0) :
  (Nat.choose (n + 2) n) * n! = ((n + 2) * (n + 1) * n!) / 2 := by
  sorry

end evaluate_binom_mul_factorial_l560_560006


namespace fifty_fourth_digit_after_decimal_of_one_over_eleven_l560_560231

theorem fifty_fourth_digit_after_decimal_of_one_over_eleven : 
  let seq := "09" in
  (54 - 1) % 2 = 1 → seq.get ((54 - 1) % 2) = '9' :=
by
  intro seq
  sorry

end fifty_fourth_digit_after_decimal_of_one_over_eleven_l560_560231


namespace polygon_with_12_diagonals_has_6_sides_l560_560836

-- Definition of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem to prove
theorem polygon_with_12_diagonals_has_6_sides : ∃ n : ℕ, number_of_diagonals n = 12 :=
by {
  use 6,
  unfold number_of_diagonals,
  calc
    6 * (6 - 3) / 2 = 6 * 3 / 2 : by norm_num
               ...  = 18 / 2   : by norm_num
               ...  = 9        : by norm_num,
  sorry
}

end polygon_with_12_diagonals_has_6_sides_l560_560836


namespace movie_theater_screens_l560_560279

theorem movie_theater_screens (hours_open : ℕ) (movie_duration : ℕ) (total_movies : ℕ)
    (h_open : hours_open = 8) (h_duration : movie_duration = 2) (h_total_movies : total_movies = 24) :
    (total_movies / (hours_open / movie_duration) = 6) :=
  by
  rw [h_open, h_duration, h_total_movies]
  sorry

end movie_theater_screens_l560_560279


namespace find_xy_l560_560036

-- Define the sample data assuming real numbers
variables {x y : ℝ}

-- Define the conditions given in the problem
def mean_condition : Prop := (x + 1 + y + 5) / 4 = 2
def variance_condition : Prop := ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 = 5

-- State the theorem using the conditions to show the desired result
theorem find_xy (h_mean : mean_condition) (h_variance : variance_condition) : x * y = -4 :=
by
  sorry

end find_xy_l560_560036


namespace general_term_a_find_c_max_value_f_l560_560041

variable (a_n : ℕ → ℝ) (d : ℝ) (S_n : ℕ → ℝ)

-- Condition: Sequence is arithmetic with common difference d > 0
variable (h_arith : ∀ n > 0, a_n n = a_n 1 + (n - 1) * d)
variable (h_d_pos : d > 0)

-- Condition: Sum of the first n terms is S_n
variable (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) * d) / 2)

-- Conditions given in the problem
variable (h_a2a3_45 : a_n 2 * a_n 3 = 45)
variable (h_a1a4_14 : a_n 1 + a_n 4 = 14)

-- (I) Proving the general term formula for the sequence {a_n}
theorem general_term_a (n : ℕ) : a_n n = 4 * n - 3 := 
  sorry

-- Define sequence {b_n}
def b_n (n : ℕ) (c : ℝ) := S_n n / (n + c)

-- Condition for {b_n} to be arithmetic sequence
variable (h_arith_bn : ∀ n, 2 * b_n (n + 1) c = b_n n c + b_n (n + 2) c)

-- (II) Proving the non-zero constant c
theorem find_c (c : ℝ) : c = -1/2 ∧ c ≠ 0 := 
  sorry

-- Function f(n)
def f (n : ℕ) (c : ℝ) := b_n n c / ((n + 25) * b_n (n + 1) c)

-- (III) Proving the maximum value of f(n)
theorem max_value_f (n : ℕ) (c : ℝ) (n_pos : 0 < n) : f n c = 1 / 36 := 
  sorry

end general_term_a_find_c_max_value_f_l560_560041


namespace sum_of_extreme_values_l560_560896

theorem sum_of_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (5 - Real.sqrt 34) / 3
  let M := (5 + Real.sqrt 34) / 3
  m + M = 10 / 3 :=
by
  sorry

end sum_of_extreme_values_l560_560896


namespace circumcircle_tangent_l560_560335

noncomputable def regular_pentagon (A B C D E : Type) (pentagon : Prop) : Prop :=
  ∀ (p : Prop), p = (A = B) = (B = C) = (C = D) = (D = E) = (E = A)

theorem circumcircle_tangent {A B C D E : Type} (pentagon : Prop)
  (h_pentagon : regular_pentagon A B C D E pentagon) 
  (K : Type) (diagonal_AC : Prop) (diagonal_BE : Prop)
  (h_AC : diagonal_AC = (A = C))
  (h_BE : diagonal_BE = (B = E))
  (h_intersection : (A = C) ∧ (B = E) → K) :
  (tangent_to_circle_at C (circumcircle (type_in K (type_out K E)))):
  sorry

end circumcircle_tangent_l560_560335


namespace expand_polynomial_eq_l560_560709

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560709


namespace sum_of_distinct_values_of_intersections_l560_560741

theorem sum_of_distinct_values_of_intersections (N : ℕ) :
  (∃ v : Finset ℕ, (∀ n ∈ v, 2 ≤ n ∧ n ≤ 10) ∧ (v.card = 9) ∧ (v.sum = 54))
by {
  -- The proof is omitted
  sorry
}

end sum_of_distinct_values_of_intersections_l560_560741


namespace expand_expression_l560_560724

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560724


namespace sum_digits_of_power_l560_560425

theorem sum_digits_of_power (A B C : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 949) : 
  A + B + C = 957 := by
  rw [hA, hB, hC]
  norm_num

end sum_digits_of_power_l560_560425


namespace local_minimum_at_x_eq_2_l560_560528

noncomputable def f (x a : ℝ) : ℝ := x * (x - a) ^ 2

theorem local_minimum_at_x_eq_2 (a : ℝ) 
  (h : ∃ f : ℝ → ℝ, ∀ x : ℝ, f = λ x, x * (x - a) ^ 2 ∧
    ∃ δ > 0, ∀ h : ℝ, abs h < δ → f (2 - h) ≥ f 2 ∧ f (2 + h) ≥ f 2) : 
  a = 2 :=
sorry

end local_minimum_at_x_eq_2_l560_560528


namespace y_intercept_of_line_m_l560_560475

/-
  Define the midpoints and the conditions of line m
  Condition 1: line m is in xy-plane (implicitly assumed as it uses 2D coordinates)
  Condition 2: slope of m is 1
  Condition 3: m passes through the midpoint of (2, 8) and (6, -4)
-/

def midpoint (p1 p2 : (ℝ × ℝ)) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def point1 : ℝ × ℝ := (2, 8)
def point2 : ℝ × ℝ := (6, -4)
def mp : ℝ × ℝ := midpoint point1 point2

theorem y_intercept_of_line_m : ∀ m : ℝ × ℝ × ℝ, m = (1, mp.1, mp.2) → mp.2 = 1 * mp.1 + (-2) :=
  by
  sorry

end y_intercept_of_line_m_l560_560475


namespace ways_to_select_defective_products_l560_560301

-- Define the given conditions
def total_products : ℕ := 100
def qualified_products : ℕ := 98
def defective_products : ℕ := 2
def selected_products : ℕ := 3

-- Using combinatorial functions
open_locale big_operators

noncomputable def combination (n k : ℕ) : ℕ :=
  nat.choose n k

-- The statement needs to show that the number of ways to have at least one defective product is correct
theorem ways_to_select_defective_products :
  combination total_products selected_products - combination qualified_products selected_products =
  (combination defective_products 1 * combination qualified_products 2) +
  (combination defective_products 2 * combination qualified_products 1) :=
by
  sorry

end ways_to_select_defective_products_l560_560301


namespace buoy_radius_proof_l560_560616

/-
We will define the conditions:
- width: 30 cm
- radius_ice_hole: 15 cm (half of width)
- depth: 12 cm
Then prove the radius of the buoy (r) equals 15.375 cm.
-/
noncomputable def radius_of_buoy : ℝ :=
  let width : ℝ := 30
  let depth : ℝ := 12
  let radius_ice_hole : ℝ := width / 2
  let r : ℝ := (369 / 24)
  r    -- the radius of the buoy

theorem buoy_radius_proof : radius_of_buoy = 15.375 :=
by 
  -- We assert that the above definition correctly computes the radius.
  sorry   -- Actual proof omitted

end buoy_radius_proof_l560_560616


namespace canteen_distance_l560_560275

theorem canteen_distance
  (AG BG AC GC : ℝ)
  (hAG : AG = 400) 
  (hBG : BG = 700)
  (hC_EQ : AC = GC)
  : AC = 1711 :=
by
  have h1: AB = Real.sqrt (AG^2 + BG^2), by sorry
  have h2: AB = 50 * Real.sqrt 260, by sorry
  have h3: GC = AC, by sorry
  have h4: (50 * Real.sqrt 260)^2 + 400^2 = GC^2, by sorry
  have h5: GC = AC, by sorry
  have h6: GC = 1711, by sorry
  exact h6

end canteen_distance_l560_560275


namespace ratio_of_sides_l560_560935

theorem ratio_of_sides (a b c d : ℝ) 
  (h1 : a / c = 4 / 5) 
  (h2 : b / d = 4 / 5) : b / d = 4 / 5 :=
sorry

end ratio_of_sides_l560_560935


namespace hyperbola_eccentricity_sqrt2_l560_560955

noncomputable def isHyperbolaPerpendicularAsymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let asymptote1 := (1/a : ℝ)
  let asymptote2 := (-1/b : ℝ)
  asymptote1 * asymptote2 = -1

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  isHyperbolaPerpendicularAsymptotes a b ha hb →
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = Real.sqrt 2 :=
by
  intro h
  sorry

end hyperbola_eccentricity_sqrt2_l560_560955


namespace number_of_pairs_l560_560137

theorem number_of_pairs (a d : ℝ) :
  (∃ a d : ℝ, (∀ x y : ℝ, 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0)) 
  ↔ 2 :=
sorry

end number_of_pairs_l560_560137


namespace total_cards_after_giveaway_l560_560633

def ben_basketball_boxes := 8
def cards_per_basketball_box := 20
def ben_baseball_boxes := 10
def cards_per_baseball_box := 15
def ben_football_boxes := 12
def cards_per_football_box := 12

def alex_hockey_boxes := 6
def cards_per_hockey_box := 15
def alex_soccer_boxes := 9
def cards_per_soccer_box := 18

def cards_given_away := 175

def total_cards_for_ben := 
  (ben_basketball_boxes * cards_per_basketball_box) + 
  (ben_baseball_boxes * cards_per_baseball_box) + 
  (ben_football_boxes * cards_per_football_box)

def total_cards_for_alex := 
  (alex_hockey_boxes * cards_per_hockey_box) + 
  (alex_soccer_boxes * cards_per_soccer_box)

def total_cards_before_exchange := total_cards_for_ben + total_cards_for_alex

def ben_gives_to_alex := 
  (ben_basketball_boxes * (cards_per_basketball_box / 2)) + 
  (ben_baseball_boxes * (cards_per_baseball_box / 2))

def total_cards_remaining := total_cards_before_exchange - cards_given_away

theorem total_cards_after_giveaway :
  total_cards_before_exchange - cards_given_away = 531 := by
  sorry

end total_cards_after_giveaway_l560_560633


namespace ab_not_9_l560_560899

theorem ab_not_9 (P : Polynomial ℂ) (a b : ℂ) (hP : P = Polynomial.CRing.HScalar (x^4 + a*x^3 + b*x^2 + x))
  (h_roots_distinct : (P.roots : set ℂ).card = 4) (h_roots_circle : ∃ R: ℝ, ∀ z ∈ P.roots, abs z = R) : a * b ≠ 9 :=
sorry

end ab_not_9_l560_560899


namespace determine_sunday_l560_560488

def Brother := Prop -- A type to represent a brother

variable (A B : Brother)
variable (T D : Brother) -- T representing Tweedledum, D representing Tweedledee

-- Conditions translated into Lean
variable (H1 : (A = T) → (B = D))
variable (H2 : (B = D) → (A = T))

-- Define the day of the week as a proposition
def is_sunday := Prop

-- We want to state that given H1 and H2, it is Sunday
theorem determine_sunday (H1 : (A = T) → (B = D)) (H2 : (B = D) → (A = T)) : is_sunday := sorry

end determine_sunday_l560_560488


namespace log_46328_range_sum_a_b_l560_560549

noncomputable def log_46328 : ℝ :=
Real.log 46328 / Real.log 10

theorem log_46328_range :
  4 < log_46328 ∧ log_46328 < 5 :=
by {
  have h1 : Real.log 10000 / Real.log 10 = 4,
  { rw [Real.log_div_log, Real.log_10000, Real.log_10], norm_num },
  have h2 : Real.log 100000 / Real.log 10 = 5,
  { rw [Real.log_div_log, Real.log_100000, Real.log_10], norm_num },
  split,
  {
    have h : log_46328 = Real.log 46328 / Real.log 10,
    { unfold log_46328 },
    linarith,
  },
  {
    have h : log_46328 = Real.log 46328 / Real.log 10,
    { unfold log_46328 },
    linarith,
  },
  sorry,
}

theorem sum_a_b :
  ∃ a b : ℕ, log_46328_range ∧ a = 4 ∧ b = 5 ∧ a + b = 9 :=
by {
  use [4, 5],
  split,
  {
    exact log_46328_range,
  },
  split,
  {
    refl,
  },
  split,
  {
    refl,
  },
  refl,
  sorry,
}

end log_46328_range_sum_a_b_l560_560549


namespace triangle_in_and_circumcircle_radius_l560_560225

noncomputable def radius_of_incircle (AC : ℝ) (BC : ℝ) (AB : ℝ) (Area : ℝ) (s : ℝ) : ℝ :=
  Area / s

noncomputable def radius_of_circumcircle (AB : ℝ) : ℝ :=
  AB / 2

theorem triangle_in_and_circumcircle_radius :
  ∀ (A B C : ℝ × ℝ) (AC : ℝ) (BC : ℝ) (AB : ℝ)
    (AngleA : ℝ) (AngleC : ℝ),
  AngleC = 90 ∧ AngleA = 60 ∧ AC = 6 ∧
  BC = AC * Real.sqrt 3 ∧ AB = 2 * AC
  → radius_of_incircle AC BC AB (18 * Real.sqrt 3) ((AC + BC + AB) / 2) = 6 * (Real.sqrt 3 - 1) / 13 ∧
    radius_of_circumcircle AB = 6 := by
  intros A B C AC BC AB AngleA AngleC h
  sorry

end triangle_in_and_circumcircle_radius_l560_560225


namespace expand_expression_l560_560721

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560721


namespace find_number_l560_560592

noncomputable def calculate_x (x : ℝ) : ℝ :=
  ((Real.cbrt ((7 * (x + 10))^2 / 5)) - 5) / 3

theorem find_number :
  ∃ x : ℝ, calculate_x x = 44 :=
begin
  sorry
end

end find_number_l560_560592


namespace tangent_line_min_slope_l560_560059

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

theorem tangent_line_min_slope :
  let f' := λ x, deriv f x
  let min_slope := -3
  let point_of_tangency := (0, f 0)
  let eq_of_tangent_line := (y := -3 * x)
  deriv f 0 = min_slope → tangent_line eq_of_tangent_line point_of_tangency :=
by
  sorry

end tangent_line_min_slope_l560_560059


namespace measure_angle_Z_l560_560868

-- Given conditions
def triangle_condition (X Y Z : ℝ) :=
   X = 78 ∧ Y = 4 * Z - 14

-- Triangle angle sum property
def triangle_angle_sum (X Y Z : ℝ) :=
   X + Y + Z = 180

-- Prove the measure of angle Z
theorem measure_angle_Z (X Y Z : ℝ) (h1 : triangle_condition X Y Z) (h2 : triangle_angle_sum X Y Z) : 
  Z = 23.2 :=
by
  -- Lean will expect proof steps here, ‘sorry’ is used to denote unproven parts.
  sorry

end measure_angle_Z_l560_560868


namespace min_rectangles_needed_l560_560569

theorem min_rectangles_needed : ∀ (n : ℕ), n = 12 → (n * n) / (3 * 2) = 24 :=
by sorry

end min_rectangles_needed_l560_560569


namespace tangent_line_at_point_l560_560261

theorem tangent_line_at_point :
  let y := fun x : ℝ => x^3 - x + 3
  let point := (1 : ℝ, 3 : ℝ)
  let tangent_eq := 2 * (point.1) - point.2 + 1
  tangent_eq = 0 :=
sorry

end tangent_line_at_point_l560_560261


namespace exponent_equality_l560_560260

theorem exponent_equality (n : ℕ) : 
    5^n = 5 * (5^2)^2 * (5^3)^3 → n = 14 := by
    sorry

end exponent_equality_l560_560260


namespace expand_expression_l560_560674

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560674


namespace sin_trig_identity_l560_560028

theorem sin_trig_identity (α : ℝ) (h : Real.sin (α - π/4) = 1/2) : Real.sin ((5 * π) / 4 - α) = 1/2 := 
by 
  sorry

end sin_trig_identity_l560_560028


namespace expand_polynomial_l560_560690

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560690


namespace complex_number_in_second_quadrant_l560_560056

theorem complex_number_in_second_quadrant (a : ℝ) :
  let z := (⟨1, -1⟩ : ℂ) * (⟨a, 1⟩ : ℂ) in
  z.re < 0 ∧ z.im > 0 ↔ a < -1 :=
by
  sorry

end complex_number_in_second_quadrant_l560_560056


namespace phoebe_age_l560_560428

theorem phoebe_age (P : ℕ) (h₁ : ∀ P, 60 = 4 * (P + 5)) (h₂: 55 + 5 = 60) : P = 10 := 
by
  have h₃ : 60 = 4 * (P + 5) := h₁ P
  sorry

end phoebe_age_l560_560428


namespace al_bill_cal_probability_l560_560622

-- Let's define the conditions and problem setup
def al_bill_cal_prob : ℚ :=
  let total_ways := 12 * 11 * 10
  let valid_ways := 12 -- This represent the summed valid cases as calculated
  valid_ways / total_ways

theorem al_bill_cal_probability :
  al_bill_cal_prob = 1 / 110 :=
  by
  -- Placeholder for calculation and proof
  sorry

end al_bill_cal_probability_l560_560622


namespace num_divisors_1800_pow_1800_with_180_factors_l560_560818

noncomputable def num_divisors_of_1800_pow_1800 : ℕ := 1800

theorem num_divisors_1800_pow_1800_with_180_factors :
  let p := 1800,
      q := p^p,
      pf := (2^3 * 3^2 * 5^2)^p,
      form (a b c : ℕ) := (q % (2^a * 3^b * 5^c) = 0)
    (condition : (0 ≤ a ∧ a ≤ 5400) ∧ (0 ≤ b ∧ b ≤ 3600) ∧ (0 ≤ c ∧ c ≤ 3600)) :=
  ∃ (n : ℕ), (n = 18 ∧ (a+1) * (b+1) * (c+1) = 180) 

end num_divisors_1800_pow_1800_with_180_factors_l560_560818


namespace locus_of_points_circle_l560_560234

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

theorem locus_of_points_circle (A B : Point) (p q : ℕ) :
  ∃ C : Point, ∀ P : Point, (distance P A) / (distance P B) = (p:ℝ) / (q:ℝ) ↔ 
  distance P C = distance A C ∧ distance B C = distance B C :=
sorry

end locus_of_points_circle_l560_560234


namespace clown_balloon_count_l560_560520

theorem clown_balloon_count (initial_balloons : ℕ) (additional_balloons : ℕ) :
  initial_balloons = 47 → additional_balloons = 13 → initial_balloons + additional_balloons = 60 :=
by
  intro h1 h2
  rw [h1, h2]
  rfl

end clown_balloon_count_l560_560520


namespace finite_integer_solutions_l560_560884

theorem finite_integer_solutions (n : ℕ) (hn1 : 1 < n) (hn2 : odd n) :
  ∃ (f : ℤ → ℤ), (degree f = n ∧
  (∀ k : fin (n+1), f k.val = 2^k.val)) →
  {x : ℤ | ∃ m : ℤ, f x = 2^m}.finite :=
by
  sorry

end finite_integer_solutions_l560_560884


namespace expand_polynomial_l560_560684

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560684


namespace converse_of_right_triangle_l560_560187

-- First, we need to define a triangle and the right triangle condition
structure Triangle :=
  (a b c : ℝ) -- representing the lengths of the sides of a triangle

def is_right_triangle (T : Triangle) : Prop :=
  ∃ (α β γ : ℝ), T.a^2 + T.b^2 = T.c^2 ∨ T.a^2 + T.c^2 = T.b^2 ∨ T.b^2 + T.c^2 = T.a^2

-- The theorem statement proving the converse
theorem converse_of_right_triangle (T : Triangle) : (T.a^2 + T.b^2 = T.c^2 ∨ T.a^2 + T.c^2 = T.b^2 ∨ T.b^2 + T.c^2 = T.a^2) → is_right_triangle T :=
by
  intro h,
  apply is_right_triangle,
  exact h,
  sorry

end converse_of_right_triangle_l560_560187


namespace baskets_containing_neither_l560_560484

-- Definitions representing the conditions
def total_baskets : ℕ := 15
def baskets_with_apples : ℕ := 10
def baskets_with_oranges : ℕ := 8
def baskets_with_both : ℕ := 5

-- Theorem statement to prove the number of baskets containing neither apples nor oranges
theorem baskets_containing_neither : total_baskets - (baskets_with_apples + baskets_with_oranges - baskets_with_both) = 2 :=
by
  sorry

end baskets_containing_neither_l560_560484


namespace expand_polynomial_l560_560683

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560683


namespace performance_bonus_l560_560874

def daily_earnings_normal := 80
def work_hours_normal := 8
def additional_hours := 2
def hourly_rate_bonus := 10

theorem performance_bonus : 
  let normal_earnings := daily_earnings_normal in
  let bonus_earnings := (work_hours_normal + additional_hours) * hourly_rate_bonus in
  bonus_earnings - normal_earnings = 20 :=
sorry

end performance_bonus_l560_560874


namespace min_val_f_range_of_m_l560_560063

noncomputable def f (x m : ℝ) : ℝ := abs (x + 1) + abs (x + m)

theorem min_val_f (m : ℝ) : (∀ x, f x m ≥ 2) → (m = 3 ∨ m = -1) :=
sorry
  
theorem range_of_m (m : ℝ) : (∀ x ∈ set.Icc (-1:ℝ) 1, f x m ≤ 2 * x + 3) → (0 ≤ m ∧ m ≤ 2) :=
sorry

end min_val_f_range_of_m_l560_560063


namespace f_difference_l560_560790

noncomputable def f (x a b : ℝ) : ℝ := 
  (1/3) * x^3 - (5/2) * a * x^2 + 6 * a * x + b

theorem f_difference {a b x1 x2 : ℝ} 
  (h1 : (∂ f x / ∂ x) = x^2 - 5 * a * x + 6 * a) 
  (h2 : x1 + x2 = 5 * a)
  (h3 : x1 * x2 = 6 * a)
  (h4 : x2 = (3 / 2) * x1) : 
  f x1 a b - f x2 a b = 1 / 6 :=
sorry

end f_difference_l560_560790


namespace cost_per_meat_with_rush_shipping_l560_560748

theorem cost_per_meat_with_rush_shipping (cost_per_pack : ℝ) (num_types : ℕ) (rush_percentage : ℝ) : num_types = 4 ∧ cost_per_pack = 40 ∧ rush_percentage = 0.30 → cost_per_pack * (1 + rush_percentage) / num_types = 13 := 
by
  intro h
  obtain ⟨hn, hc, hr⟩ := h
  rw [hn, hc, hr]
  norm_num
  sorry

end cost_per_meat_with_rush_shipping_l560_560748


namespace quadratic_roots_are_distinct_real_l560_560502

theorem quadratic_roots_are_distinct_real (a b c : ℝ) (h_eq : a = 1) (h_b : b = 4) (h_c : c = 0) :
  a * c = 0 ∧ b^2 - 4 * a * c > 0 :=
by
  rw [h_eq, h_b, h_c]
  split
  case left => 
    calc 
      1 * 0 = 0 : by norm_num
  case right =>
    calc 
      4^2 - 4 * 1 * 0 = 16 - 0 : by norm_num
      ... = 16      : by norm_num
      ... > 0       : by norm_num

end quadratic_roots_are_distinct_real_l560_560502


namespace YZ_length_l560_560099

axiom triangle_XYZ {X Y Z M N : Type} (d_XM d_MY d_NZ : ℝ) (h_MN_parallel_XY : Prop) : Prop

noncomputable def XM : ℝ := 5
noncomputable def MY : ℝ := 8
noncomputable def NZ : ℝ := 9

theorem YZ_length {X Y Z M N : Type} (h : triangle_XYZ XM MY NZ h_MN_parallel_XY) : YZ = 23 := sorry

end YZ_length_l560_560099


namespace interval_length_le_sqrt2_l560_560467

open Real

theorem interval_length_le_sqrt2 (k m a b : ℝ) 
  (h : ∀ x ∈ Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * sqrt 2 :=
sorry

end interval_length_le_sqrt2_l560_560467


namespace total_length_of_free_sides_l560_560579

theorem total_length_of_free_sides (L W : ℝ) 
  (h1 : L = 2 * W) 
  (h2 : L * W = 128) : 
  L + 2 * W = 32 := by 
sorry

end total_length_of_free_sides_l560_560579


namespace total_fruit_count_l560_560552

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l560_560552


namespace greatest_monthly_drop_is_march_l560_560981

-- Define the price changes for each month
def price_change_january : ℝ := -0.75
def price_change_february : ℝ := 1.50
def price_change_march : ℝ := -3.00
def price_change_april : ℝ := 2.50
def price_change_may : ℝ := -1.00
def price_change_june : ℝ := 0.50
def price_change_july : ℝ := -2.50

-- Prove that the month with the greatest drop in price is March
theorem greatest_monthly_drop_is_march :
  (price_change_march = -3.00) →
  (∀ m, m ≠ price_change_march → m ≥ price_change_march) :=
by
  intros h1 h2
  sorry

end greatest_monthly_drop_is_march_l560_560981


namespace lydia_flowers_on_porch_l560_560915

theorem lydia_flowers_on_porch:
  ∀ (total_plants : ℕ) (flowering_percentage : ℚ) (fraction_on_porch : ℚ) (flowers_per_plant : ℕ),
  total_plants = 80 →
  flowering_percentage = 0.40 →
  fraction_on_porch = 1 / 4 →
  flowers_per_plant = 5 →
  let flowering_plants := (total_plants : ℚ) * flowering_percentage in
  let porch_plants := flowering_plants * fraction_on_porch in
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ) in
  total_flowers_on_porch = 40 :=
by {
  intros total_plants flowering_percentage fraction_on_porch flowers_per_plant,
  intros h1 h2 h3 h4,
  let flowering_plants := (total_plants : ℚ) * flowering_percentage,
  let porch_plants := flowering_plants * fraction_on_porch,
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ),
  sorry
}

end lydia_flowers_on_porch_l560_560915


namespace system_solution_find_a_l560_560256

theorem system_solution (x y : ℝ) (a : ℝ) :
  (|16 + 6 * x - x ^ 2 - y ^ 2| + |6 * x| = 16 + 12 * x - x ^ 2 - y ^ 2)
  ∧ ((a + 15) * y + 15 * x - a = 0) →
  ( (x - 3) ^ 2 + y ^ 2 ≤ 25 ∧ x ≥ 0 ) :=
sorry

theorem find_a (a : ℝ) :
  ∃ (x y : ℝ), 
  ((a + 15) * y + 15 * x - a = 0 ∧ x ≥ 0 ∧ (x - 3) ^ 2 + y ^ 2 ≤ 25) ↔ 
  (a = -20 ∨ a = -12) :=
sorry

end system_solution_find_a_l560_560256


namespace g_ratio_l560_560530

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_property (c d : ℝ) : c^2 * g(d) = d^2 * g(c)
axiom g_not_zero : g 3 ≠ 0

theorem g_ratio :
  (g 6 - g 2) / g 3 = 32 / 9 := 
sorry

end g_ratio_l560_560530


namespace greatest_distance_C_D_l560_560860

noncomputable def C : set ℂ := {1, -1/2 + complex.I * (real.sqrt 3) / 2, -1/2 - complex.I * (real.sqrt 3) / 2}

noncomputable def D : set ℂ := {1, 13 + 4 * real.sqrt 3, 13 - 4 * real.sqrt 3}

theorem greatest_distance_C_D :
  ∃ A ∈ C, ∃ B ∈ D, dist A B = real.sqrt (184.5 + 60 * real.sqrt 3) := 
sorry

end greatest_distance_C_D_l560_560860


namespace fx_in_Q_l560_560809

def f (x : ℝ) : ℝ := (x - 1) ^ 3

def M : set (ℝ → ℝ) := {f | ∀ x : ℝ, f (-x) = f x}
def N : set (ℝ → ℝ) := {f | ∀ x : ℝ, f (-x) = -f x}
def P : set (ℝ → ℝ) := {f | ∀ x : ℝ, f (1 - x) = f (1 + x)}
def Q : set (ℝ → ℝ) := {f | ∀ x : ℝ, f (1 - x) = -f (1 + x)}

theorem fx_in_Q : f ∈ Q :=
sorry

end fx_in_Q_l560_560809


namespace amount_of_silver_l560_560862

-- Definitions
def total_silver (x : ℕ) : Prop :=
  (x - 4) % 7 = 0 ∧ (x + 8) % 9 = 1

-- Theorem to be proven
theorem amount_of_silver (x : ℕ) (h : total_silver x) : (x - 4)/7 = (x + 8)/9 :=
by sorry

end amount_of_silver_l560_560862


namespace circle_A_diameter_l560_560226

theorem circle_A_diameter (r_B : ℝ) (r_A : ℝ) (d_B : ℝ) (d_A : ℝ) (shaded_to_A_ratio : ℝ) 
  (diameter_B_is_20 : d_B = 20) 
  (r_B_is_half_d_B : r_B = d_B / 2) 
  (shaded_A_ratio_is_5_1 : shaded_to_A_ratio = 5) 
  (two_non_overlapping_circles : true)
  (circle_A_equals_circle_C : true) : 
  d_A ≈ 7.56 :=
by 
  have diameter := 20 / 2 -> 10 (half_diameter),
  have Pi := 100π - 2π r² = 5π r² (area and ratio_shaded),
  have sqrt100 := sqrt(100 )  = 10 (calculated),
  have sqrt_ratio := sqrt(100 / 7) = sqrt(14.285) (resolved),
  exact 2 * sqrt(14.285) = 2 * 3.78 (diameter) = 7.56

end circle_A_diameter_l560_560226


namespace exists_infinite_sequence_no_exact_powers_l560_560120

open Nat

/-- Is there an infinite sequence of natural numbers such that it contains no exact powers 
of natural numbers and no sum of any finite non-empty subset of its elements is an exact 
power of a natural number? -/
theorem exists_infinite_sequence_no_exact_powers :
  ∃ (a : ℕ → ℕ), (∀ n, ∃ k, a n = (∏ i in finset.range k, (p i)^2) * p k) ∧
    (∀ (s : finset ℕ), s.nonempty → ¬ is_power (∑ i in s, a i)) :=
by
  sorry

end exists_infinite_sequence_no_exact_powers_l560_560120


namespace solve_equation_l560_560346

theorem solve_equation : ∃ x : ℝ, (x = 19) →
  (∀ y : ℝ, y = real.sqrt (x - 10) →
    5 / (y - 8) + 2 / (y - 5) + 8 / (y + 5) + 10 / (y + 8) = 0)
:= sorry

end solve_equation_l560_560346


namespace total_profit_percentage_is_correct_l560_560288

noncomputable def total_profit_percentage : ℚ :=
  let selling_price_A := 50
  let cost_price_A := 0.95 * selling_price_A
  let profit_A_per_unit := selling_price_A - cost_price_A
  let total_profit_A := 100 * profit_A_per_unit
  
  let selling_price_B := 60
  let cost_price_B := 0.9 * selling_price_B
  let profit_B_per_unit := selling_price_B - cost_price_B
  let total_profit_B := 150 * profit_B_per_unit

  let total_profit := total_profit_A + total_profit_B
  
  let total_cost_price_A := cost_price_A * 100
  let total_cost_price_B := cost_price_B * 150
  let total_cost_price := total_cost_price_A + total_cost_price_B

  (total_profit / total_cost_price) * 100

theorem total_profit_percentage_is_correct (h : total_profit_percentage ≈ 8.95) : true :=
sorry

end total_profit_percentage_is_correct_l560_560288


namespace solve_problem_l560_560888

noncomputable def problem : Prop :=
  ∃ (ABC : Triangle) (ω : Circle) (T X Y : Point) (BT CT BC TX TY XY : ℝ),
    ABC.acute ∧ ABC.scalene ∧ ABC.circumcircle = ω ∧
    (tangent ω ABC.B) ∧ (tangent ω ABC.C) ∧
    are_projections T X ABC.A ABC.B ∧ 
    are_projections T Y ABC.A ABC.C ∧
    BT = 18 ∧ CT = 18 ∧ BC = 24 ∧
    TX^2 + TY^2 + XY^2 = 1529 ∧
    XY^2 = 383.75

theorem solve_problem : problem :=
  sorry

end solve_problem_l560_560888


namespace sixth_root_24414062515625_l560_560325

theorem sixth_root_24414062515625 :
  (∃ (x : ℕ), x^6 = 24414062515625) → (sqrt 6 24414062515625 = 51) :=
by
  -- Applying the condition expressed as sum of binomials
  have h : 24414062515625 = ∑ k in finset.range 7, binom 6 k * (50 ^ (6 - k)),
  sorry
  
  -- Utilize this condition to find the sixth root
  sorry

end sixth_root_24414062515625_l560_560325


namespace min_shift_value_monotonic_dec_l560_560167

theorem min_shift_value_monotonic_dec :
  (∀ x m : ℝ, (y : ℝ) = sin (2 * x + 2 * m + (π / 6))) ∧
  (∀ m > 0, ∀ k : ℤ,
    2 * (- π / 12) + 2 * m + π / 6 ≥ 2 * k * π + π / 2 ∧
    2 * (5 * π / 12) + 2 * m + π / 6 ≤ 2 * k * π + 3 * π / 2)
  → m = k * π + π / 4 := sorry

end min_shift_value_monotonic_dec_l560_560167


namespace range_of_a_l560_560397

def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a^x

theorem range_of_a (a : ℝ) :
  (∀ x < 1, (2 * a - 1) < 0) ∧
  (∀ x ≥ 1, 0 < a ∧ a < 1) ∧
  ((2 * a - 1) + 7 * a - 2 ≥ a) ↔
  (3 / 8 < a ∧ a < 1 / 2) := 
sorry

end range_of_a_l560_560397


namespace satisfy_conditions_l560_560269

noncomputable def linear_function (k b x : ℕ) : ℕ := k * x + b

def find_price_and_profit 
  (purchase_price : ℕ)
  (data_points : List (ℕ × ℕ))
  (target_profit : ℕ)
  (max_profit_percentage : ℕ) : 
  (ℕ × ℕ) :=
sorry

theorem satisfy_conditions:
  ∀ (purchase_price : ℕ) (target_profit max_profit : ℕ),
    purchase_price = 50 →
    target_profit = 24000 →
    max_profit = 19500 →
    (data_points : List (ℕ × ℕ)) = [(60, 1400), (65, 1300), (70, 1200)] →
    let (k, b) := (-20, 2600) in
    linear_function k b 65 = 1300 ∧
    linear_function k b 70 = 1200 ∧ 
    find_price_and_profit purchase_price data_points target_profit 30 = (70, 24000) ∧
    find_price_and_profit purchase_price data_points max_profit 30 = (65, 19500) :=
sorry

end satisfy_conditions_l560_560269


namespace expand_product_l560_560700

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560700


namespace expand_expression_l560_560726

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560726


namespace smallest_b_l560_560066

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

theorem smallest_b (a : ℝ) (b : ℝ) (x : ℝ) : (1 < a ∧ a < 4) → (0 < x) → (f a b x > 0) → b ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end smallest_b_l560_560066


namespace positive_value_of_X_l560_560465

def hash_relation (X Y : ℕ) : ℕ := X^2 + Y^2

theorem positive_value_of_X (X : ℕ) (h : hash_relation X 7 = 290) : X = 17 :=
by sorry

end positive_value_of_X_l560_560465


namespace find_the_number_l560_560190

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number_l560_560190


namespace Tim_younger_than_Jenny_l560_560987

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l560_560987


namespace f_g_3_value_l560_560408

def f (x : ℝ) := x^3 + 1
def g (x : ℝ) := 3 * x + 2

theorem f_g_3_value : f (g 3) = 1332 := by
  sorry

end f_g_3_value_l560_560408


namespace meeting_greetings_l560_560631

theorem meeting_greetings (k : ℕ) (h₁: 12 * k > 0) (h₂ : ∀ (a b : ℕ), a ≠ b → ∃ ℓ : ℕ, ∀ c : ℕ, (c ≠ a ∧ c ≠ b) → ((a → 3*k+6) ∧ (b → 3*k+6) → c -> ℓ)) :
  12 * k = 36 :=
by
  -- Assume k = 3 satisfies the condition
  let k := 3
  exact by sorry

end meeting_greetings_l560_560631


namespace bench_arrangement_count_round_table_arrangement_count_l560_560585

-- Definitions based on conditions.
def number_of_french := 4
def number_of_ivory_coast := 2
def number_of_english := 3
def number_of_swedish := 4

-- Statement for the bench arrangement
theorem bench_arrangement_count : 
  (fact number_of_french) * 
  (fact number_of_ivory_coast) * 
  (fact number_of_english) * 
  (fact number_of_swedish) * 
  (fact 4) = 
  165888 :=
by sorry

-- Statement for the round table arrangement
theorem round_table_arrangement_count : 
  (fact number_of_french) * 
  (fact number_of_ivory_coast) * 
  (fact number_of_english) * 
  (fact number_of_swedish) * 
  (fact 3) = 
  41472 :=
by sorry

end bench_arrangement_count_round_table_arrangement_count_l560_560585


namespace max_r_value_l560_560763

theorem max_r_value (r : ℝ) (hr_pos : 0 < r)
  (hT_in_S : ∀ x y : ℝ, x^2 + (y - 7)^2 ≤ r^2 → ∀ θ : ℝ, cos (2 * θ) + x * cos θ + y ≥ 0) : r = 4 * real.sqrt 2 :=
sorry

end max_r_value_l560_560763


namespace count_multiples_of_3_with_units_digit_3_or_6_l560_560413

theorem count_multiples_of_3_with_units_digit_3_or_6 (n : ℕ) (h : n = 150) :
  (∑ k in finset.range n, if (3 * k < n) ∧ ((3 * k % 10 = 3) ∨ (3 * k % 10 = 6)) then 1 else 0) = 10 :=
begin
  sorry
end

end count_multiples_of_3_with_units_digit_3_or_6_l560_560413


namespace total_animal_crackers_eaten_l560_560918

-- Define the context and conditions
def number_of_students : ℕ := 20
def uneaten_students : ℕ := 2
def crackers_per_pack : ℕ := 10

-- Define the statement and prove the question equals the answer given the conditions
theorem total_animal_crackers_eaten : 
  (number_of_students - uneaten_students) * crackers_per_pack = 180 := by
  sorry

end total_animal_crackers_eaten_l560_560918


namespace find_x_l560_560368

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l560_560368


namespace count_elements_in_A_l560_560939

variables (a b : ℕ)

def condition1 : Prop := a = 3 * b / 2
def condition2 : Prop := a + b - 1200 = 4500

theorem count_elements_in_A (h1 : condition1 a b) (h2 : condition2 a b) : a = 3420 :=
by sorry

end count_elements_in_A_l560_560939


namespace monotonic_increasing_interval_l560_560203

open Real

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π ↔ 
    ∀ t, -π / 2 + 2 * k * π ≤ 2 * t - π / 3 ∧ 2 * t - π / 3 ≤ π / 2 + 2 * k * π :=
sorry

end monotonic_increasing_interval_l560_560203


namespace turtles_received_l560_560476

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l560_560476


namespace short_show_episodes_l560_560559

theorem short_show_episodes (E : ℕ) (H1 : ∀ e, e = 0.5) (H2 : 12 * 1 + E * 0.5 = 24) : E = 24 := 
by sorry

end short_show_episodes_l560_560559


namespace candy_box_price_increase_l560_560740

theorem candy_box_price_increase
  (C : ℝ) -- Original price of the candy box
  (S : ℝ := 12) -- Original price of a can of soda
  (combined_price : C + S = 16) -- Combined price before increase
  (candy_box_increase : C + 0.25 * C = 1.25 * C) -- Price increase definition
  (soda_increase : S + 0.50 * S = 18) -- New price of soda after increase
  : 1.25 * C = 5 := sorry

end candy_box_price_increase_l560_560740


namespace percentage_increased_is_correct_l560_560263

-- Define the initial and final numbers
def initial_number : Nat := 150
def final_number : Nat := 210

-- Define the function to compute the percentage increase
def percentage_increase (initial final : Nat) : Float :=
  ((final - initial).toFloat / initial.toFloat) * 100.0

-- The theorem we need to prove
theorem percentage_increased_is_correct :
  percentage_increase initial_number final_number = 40 := 
by
  simp [percentage_increase, initial_number, final_number]
  sorry

end percentage_increased_is_correct_l560_560263


namespace find_f_neg_l560_560046

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 < x then x^2 + x - 1
  else if x < 0 then -x^2 + x + 1
  else 0

-- Define the property of an odd function
def is_odd (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

-- Conditions from the problem
def f_pos (x : ℝ) : Prop := 0 < x → f x = x^2 + x - 1
def f_neg : Prop := ∀ x : ℝ, x < 0 → f x = -x^2 + x + 1

-- Prove that f is odd and satisfies the given conditions
theorem find_f_neg (f: ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ (x > 0), f x = x^2 + x - 1) :
  ∀ (x : ℝ), x < 0 → f x = -x^2 + x + 1 := 
by {
  intros x hx,
  specialize h_odd x,
  have : f (-x) = x^2 + x - 1 := h_pos (-x) (by linarith),
  rw h_odd at this,
  linarith,
}

end find_f_neg_l560_560046


namespace distance_AB_l560_560442

/-- Define points A and B in 3D space --/
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := ⟨0, 2, 5⟩
def B : Point3D := ⟨-1, 3, 3⟩

/-- Define the distance function between two points in 3D space --/
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

/-- Theorem: The distance between points A and B is sqrt(6) --/
theorem distance_AB : distance A B = real.sqrt 6 := 
  sorry

end distance_AB_l560_560442


namespace debate_organizing_committees_count_l560_560107

theorem debate_organizing_committees_count :
    ∃ (n : ℕ), n = 5 * (Nat.choose 8 4) * (Nat.choose 8 3)^4 ∧ n = 3442073600 :=
by
  sorry

end debate_organizing_committees_count_l560_560107


namespace median_number_of_children_is_3_l560_560198

def number_of_children := [0, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 7]
def total_families := number_of_children.length
def median_position := (total_families + 1) / 2

theorem median_number_of_children_is_3
  (h1 : total_families = 15)
  (h2 : number_of_children.sorted) :
  number_of_children.nth (median_position - 1) = some 3 :=
by
  sorry

end median_number_of_children_is_3_l560_560198


namespace ratio_of_segments_invariant_l560_560812

variable {P E F Oe Of : Point}
variable {circle_e circle_f : Circle}
variable {r_e r_f : ℝ}
variable {angle_PEF angle_PFE : ℝ}

theorem ratio_of_segments_invariant
  (tangent_from_P_to_e : is_tangent P E circle_e)
  (tangent_from_P_to_f : is_tangent P F circle_f)
  (radius_circle_e : circle_e.radius = r_e)
  (radius_circle_f : circle_f.radius = r_f)
  (sin_angle_E : sin (angle P E F) = sin angle_PEF)
  (sin_angle_F : sin (angle P F E) = sin angle_PFE) :
  (2 * r_f * sin_angle_F) / (2 * r_e * sin_angle_E) = (PE / PF) :=
sorry

end ratio_of_segments_invariant_l560_560812


namespace distance_between_points_l560_560348

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

theorem distance_between_points :
  distance ⟨3, 3, 1⟩ ⟨-2, -2, 3⟩ = 3 * Real.sqrt 6 :=
by
  sorry

end distance_between_points_l560_560348


namespace no_integer_m_for_exponent_l560_560829

theorem no_integer_m_for_exponent (k : ℝ) (hk : 4 ^ k = 5) : ¬ ∃ (m : ℤ), 4 ^ (m * k + 2) = 400 :=
by
  sorry

end no_integer_m_for_exponent_l560_560829


namespace i_pow_2006_l560_560783

-- Definitions based on given conditions
def i : ℂ := Complex.I

-- Cyclic properties of i (imaginary unit)
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- The proof statement
theorem i_pow_2006 : (i^2006 = -1) :=
by
  sorry

end i_pow_2006_l560_560783


namespace range_of_e_l560_560804

theorem range_of_e (a b c d e : ℝ) (h₁ : a + b + c + d + e = 8) (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l560_560804


namespace donut_ate_even_neighbors_l560_560489

def cube neighbors (n : ℕ) : ℕ := sorry

theorem donut_ate_even_neighbors : 
  (cube neighbors 5) = 63 := 
by
  sorry

end donut_ate_even_neighbors_l560_560489


namespace odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l560_560817

theorem odd_positive_multiples_of_7_with_units_digit_1_lt_200_count : 
  ∃ (count : ℕ), count = 3 ∧
  ∀ n : ℕ, (n % 2 = 1) → (n % 7 = 0) → (n < 200) → (n % 10 = 1) → count = 3 :=
sorry

end odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l560_560817


namespace estimate_pi_simulation_l560_560434

theorem estimate_pi_simulation :
  let side := 2
  let radius := 1
  let total_seeds := 1000
  let seeds_in_circle := 778
  (π : ℝ) * radius^2 / side^2 = (seeds_in_circle : ℝ) / total_seeds → π = 3.112 :=
by
  intros
  sorry

end estimate_pi_simulation_l560_560434


namespace turtles_received_l560_560477

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l560_560477


namespace remainder_of_2_pow_33_mod_9_l560_560967

theorem remainder_of_2_pow_33_mod_9 : (2 ^ 33) % 9 = 8 :=
by
  sorry

end remainder_of_2_pow_33_mod_9_l560_560967


namespace crop_planting_methods_l560_560494

/-- There are 3 types of crops and 5 fields. Each field must be planted with one type of crop,
    and no two adjacent fields can be planted with the same type of crop.
    Prove that the number of different planting methods is equal to 42. -/
theorem crop_planting_methods : 
  let num_crops := 3 in
  let num_fields := 5 in
  let planting_methods := 3 * (2^(num_fields - 1)) in
  let invalid_configs := 3 * (num_crops - 1) in
  planting_methods - invalid_configs = 42 := 
by 
  let num_crops := 3
  let num_fields := 5
  let planting_methods := 3 * (2^(num_fields - 1))
  let invalid_configs := 3 * (num_crops - 1)
  sorry

end crop_planting_methods_l560_560494


namespace scramble_time_is_correct_l560_560875

-- Define the conditions
def sausages : ℕ := 3
def fry_time_per_sausage : ℕ := 5
def eggs : ℕ := 6
def total_time : ℕ := 39

-- Define the time to scramble each egg
def scramble_time_per_egg : ℕ :=
  let frying_time := sausages * fry_time_per_sausage
  let scrambling_time := total_time - frying_time
  scrambling_time / eggs

-- The theorem stating the main question and desired answer
theorem scramble_time_is_correct : scramble_time_per_egg = 4 := by
  sorry

end scramble_time_is_correct_l560_560875


namespace lambda_mu_squared_l560_560384

variable {ℝ : Type}

theorem lambda_mu_squared {x₁ y₁ x₂ y₂ λ μ : ℝ}
    (h₁ : (x₁^2) / 4 + (y₁^2) / 3 = 1)
    (h₂ : (x₂^2) / 4 + (y₂^2) / 3 = 1)
    (h₃ : y₁ * y₂ = (-3 / 4) * (x₁ * x₂)) 
    (hP : ∀ x y : ℝ, (x / 4) + (y / 3) = 1 → (x = λ * x₁ + μ * x₂) ∧ (y = λ * y₁ + μ * y₂)) :
    λ^2 + μ^2 = 1 := 
sorry

end lambda_mu_squared_l560_560384


namespace dave_average_speed_l560_560323

def total_distance (d1 d2 : ℝ) : ℝ := d1 + d2

def total_time (d1 d2 s1 s2 : ℝ) : ℝ := (d1 / s1) + (d2 / s2)

def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

theorem dave_average_speed :
  let d1 := 30
  let d2 := 10
  let s1 := 10
  let s2 := 30
  let total_dist := total_distance d1 d2
  let total_t := total_time d1 d2 s1 s2
  (average_speed total_dist total_t) = 12 :=
by
  -- Definitions for total distance and total time
  have h_total_dist : total_dist = 40 := rfl
  have h_total_t : total_t = 10 / 3 := rfl
  -- Calculation of average speed
  rw [h_total_dist, h_total_t]
  norm_num
  sorry

end dave_average_speed_l560_560323


namespace sum_of_abs_coeffs_l560_560025

theorem sum_of_abs_coeffs (a : ℕ → ℤ) :
  (∀ x, (1 - 3 * x)^9 = ∑ i in Finset.range 10, a i * x^i) →
  ∑ i in Finset.range 10, |a i| = 4^9 :=
by
  intro h
  sorry

end sum_of_abs_coeffs_l560_560025


namespace total_votes_cast_l560_560433

variable (total_votes : ℕ)
variable (emily_votes : ℕ)
variable (emily_share : ℚ := 4 / 15)
variable (dexter_share : ℚ := 1 / 3)

theorem total_votes_cast :
  emily_votes = 48 → 
  emily_share * total_votes = emily_votes → 
  total_votes = 180 := by
  intro h_emily_votes
  intro h_emily_share
  sorry

end total_votes_cast_l560_560433


namespace present_age_of_B_l560_560582

theorem present_age_of_B 
    (a b : ℕ) 
    (h1 : a + 10 = 2 * (b - 10)) 
    (h2 : a = b + 12) : 
    b = 42 := by 
  sorry

end present_age_of_B_l560_560582


namespace system_has_three_real_k_with_unique_solution_l560_560077

theorem system_has_three_real_k_with_unique_solution :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) → (x, y) = (0, 0)) → 
  ∃ (k : ℝ), ∃ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) :=
by
  sorry

end system_has_three_real_k_with_unique_solution_l560_560077


namespace savings_percentage_correct_individual_amounts_correct_combined_total_correct_l560_560122

def budget : ℝ := 1000

def food_percentage : ℝ := 0.22
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.18
def transportation_percentage : ℝ := 0.12
def clothes_percentage : ℝ := 0.08
def miscellaneous_percentage : ℝ := 0.05

def food_expense : ℝ := food_percentage * budget
def accommodation_expense : ℝ := accommodation_percentage * budget
def entertainment_expense : ℝ := entertainment_percentage * budget
def transportation_expense : ℝ := transportation_percentage * budget
def clothes_expense : ℝ := clothes_percentage * budget
def miscellaneous_expense : ℝ := miscellaneous_percentage * budget

def total_spent : ℝ := food_expense + accommodation_expense + entertainment_expense + transportation_expense + clothes_expense + miscellaneous_expense
def savings : ℝ := budget - total_spent

def percentage_savings : ℝ := (savings / budget) * 100
noncomputable def combined_total : ℝ := entertainment_expense + transportation_expense + miscellaneous_expense

theorem savings_percentage_correct : percentage_savings = 20 := by
  sorry

theorem individual_amounts_correct :
  food_expense = 220 ∧
  accommodation_expense = 150 ∧
  entertainment_expense = 180 ∧
  transportation_expense = 120 ∧
  clothes_expense = 80 ∧
  miscellaneous_expense = 50 := by
  sorry

theorem combined_total_correct : 
  combined_total = 350 := by
  sorry

end savings_percentage_correct_individual_amounts_correct_combined_total_correct_l560_560122


namespace apples_last_28_days_l560_560119

theorem apples_last_28_days 
  (half_apple_per_day : ℕ → ℕ)
  (small_apple_weight : ℚ)
  (apple_price_per_pound : ℚ)
  (total_spent : ℚ)
  : (1/4 : ℚ) = small_apple_weight →
    2 = apple_price_per_pound →
    7 = total_spent →
    half_apple_per_day 2 = 1 →
    let days := (total_spent / apple_price_per_pound) / small_apple_weight * 2 in
    days = 28 :=
sorry

end apples_last_28_days_l560_560119


namespace tangent_line_through_M_l560_560349

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * x

-- Define the point M
def M : (ℝ × ℝ) := (0, 0)

-- Define the slope of the tangent line at a point (x, y) on the parabola
def slope (x : ℝ) : ℝ := x + 2

-- Now, let's create the proof statement.
theorem tangent_line_through_M (x y : ℝ) (h : parabola x = y):
∃ a b c : ℝ, a ≠ 0 ∧ (∀ (t : ℝ), y = (1 / 2) * t^2 + 2 * t → y - M.snd = (slope t) * (x - M.fst) ) ∧ (a * x + b * y + c = 0) :=
by
  use [2, -1, 0]
  sorry

end tangent_line_through_M_l560_560349


namespace correct_answer_l560_560853

noncomputable def ratio_of_areas (BC : ℝ) : ℝ :=
  let DC := 3 * BC
  let AB := DC
  let AD := BC
  -- Area of rectangle ABCD
  let area_rect := AB * BC
  -- Assuming properties of trisection and 30-60-90 triangles' relationships:
  let AE := BC / 3
  let DE := BC * sqrt 3 / 3
  let DF := BC * sqrt 3 / 3
  -- Area of triangle DEF
  let area_tri := (1 / 2) * DF * (BC * sqrt 3 / 6)
  area_tri / area_rect

theorem correct_answer (BC : ℝ) : ratio_of_areas BC = sqrt 3 / 36 :=
by
  unfold ratio_of_areas
  -- Assume BC is positive for simplicity in calculation, but it does not affect the result
  have hBC_pos : 0 < BC := sorry
  -- Computation steps...
  sorry

end correct_answer_l560_560853


namespace instantaneous_rate_of_change_eq_derivative_l560_560533

variable {α : Type*} {β : Type*}
variable [TopologicalSpace α] [TopologicalSpace β] [MetricSpace α] [MetricSpace β]
variable {f : α → β} {x₀ : α}

theorem instantaneous_rate_of_change_eq_derivative 
  (h_diff : DifferentiableAt α β f x₀) : 
  fderiv α β f x₀ = 
  (λ (Δx : α), f (x₀ + Δx) - f x₀) :=
by
  sorry

end instantaneous_rate_of_change_eq_derivative_l560_560533


namespace solution_exists_l560_560338

def equation (x : ℝ) : Prop := sqrt (4 * x - 3) + 10 / sqrt (4 * x - 3) = 7

theorem solution_exists (x : ℝ) :
  equation x ↔ (x = 7/4 ∨ x = 7) :=
by 
  sorry

end solution_exists_l560_560338


namespace expand_product_l560_560698

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560698


namespace find_linear_function_l560_560887

theorem find_linear_function (α : ℝ) (hα : α > 0)
  (f : ℕ+ → ℝ)
  (h : ∀ (k m : ℕ+), α * (m : ℝ) ≤ (k : ℝ) ∧ (k : ℝ) < (α + 1) * (m : ℝ) → f (k + m) = f k + f m)
: ∃ (b : ℝ), ∀ (n : ℕ+), f n = b * (n : ℝ) :=
sorry

end find_linear_function_l560_560887


namespace min_advantageous_discount_l560_560745

theorem min_advantageous_discount : ∃ n : ℕ, n = 39 ∧ (n > 36) ∧ (n > 38.5875) ∧ (n > 37) :=
by
  use 39
  split
  · rfl
  split
  · exact dec_trivial
  split
  · exact dec_trivial
  · exact dec_trivial

end min_advantageous_discount_l560_560745


namespace frank_fence_l560_560250

theorem frank_fence (L W F : ℝ) (hL : L = 40) (hA : 320 = L * W) : F = 2 * W + L → F = 56 := by
  sorry

end frank_fence_l560_560250


namespace smallest_n_property_l560_560350

theorem smallest_n_property :
  ∃ n : ℕ, (∀ x y z : ℕ, (x > 0) ∧ (y > 0) ∧ (z > 0) → (x ∣ y ^ 3) ∧ (y ∣ z ^ 3) ∧ (z ∣ x ^ 3) →
    (x * y * z ∣ (x + y + z) ^ n)) ∧ (∀ m : ℕ, (∀ x y z : ℕ, (x > 0) ∧ (y > 0) ∧ (z > 0) →
      (x ∣ y ^ 3) ∧ (y ∣ z ^ 3) ∧ (z ∣ x ^ 3) →
      (x * y * z ∣ (x + y + z) ^ m)) → n ≤ m) :=
by
  let n := 13
  existsi n
  split
  sorry
  sorry

end smallest_n_property_l560_560350


namespace repayment_amount_formula_l560_560563

def loan_principal := 480000
def repayment_years := 20
def repayment_months := repayment_years * 12
def monthly_interest_rate := 0.004
def monthly_principal_repayment := loan_principal / repayment_months

def interest_for_nth_month (n : ℕ) : ℚ :=
  (loan_principal - (n - 1) * monthly_principal_repayment) * monthly_interest_rate

def repayment_amount_nth_month (n : ℕ) : ℚ :=
  monthly_principal_repayment + interest_for_nth_month n

theorem repayment_amount_formula (n : ℕ) (hn : 1 ≤ n ∧ n ≤ repayment_months) :
  repayment_amount_nth_month n = 3928 - 8 * n := by
sorry

end repayment_amount_formula_l560_560563


namespace possible_denominators_count_l560_560176

theorem possible_denominators_count :
  ∀ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) →
  ∃ (D : Finset ℕ), D.card = 7 ∧ 
  ∀ num denom, (num = 100*a + 10*b + c) → (denom = 999) → (gcd num denom > 1) → 
  denom ∈ D := 
sorry

end possible_denominators_count_l560_560176


namespace Sam_has_correct_amount_of_dimes_l560_560166

-- Definitions for initial values and transactions
def initial_dimes := 9
def dimes_from_dad := 7
def dimes_taken_by_mom := 3
def sets_from_sister := 4
def dimes_per_set := 2

-- Definition of the total dimes Sam has now
def total_dimes_now : Nat :=
  initial_dimes + dimes_from_dad - dimes_taken_by_mom + (sets_from_sister * dimes_per_set)

-- Proof statement
theorem Sam_has_correct_amount_of_dimes : total_dimes_now = 21 := by
  sorry

end Sam_has_correct_amount_of_dimes_l560_560166


namespace no_real_roots_fff_eq_4x_l560_560062

-- Definitions and conditions
variable {a b c : ℝ} (h_a : a ≠ 0)
def f (x : ℝ) := a * x^2 + b * x + c
def no_real_roots_f (x : ℝ) := f x ≠ 2 * x

-- The main theorem statement
theorem no_real_roots_fff_eq_4x (h : no_real_roots_f 0) : ¬ ∃ x, f (f x) = 4 * x :=
sorry

end no_real_roots_fff_eq_4x_l560_560062


namespace domain_of_f_x_squared_l560_560054

-- Given conditions: domain of f(x) is [1, 4]
def domain_of_f (f : ℝ → ℝ) : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

-- Prove the domain of f(x^2) is [-2, -1] ∪ [1, 2]
theorem domain_of_f_x_squared (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4} = domain_of_f f →
  {x : ℝ | (-2) ≤ x ∧ x ≤ (-1) ∨ 1 ≤ x ∧ x ≤ 2} = {x : ℝ | 1 ≤ x^2 ∧ x^2 ≤ 4} :=
by
  assume h : {x : ℝ | 1 ≤ x ∧ x ≤ 4} = domain_of_f f
  sorry

end domain_of_f_x_squared_l560_560054


namespace common_ratio_of_geometric_seq_l560_560974

variable {α : Type*} [Field α]

-- Definition of the geometric sequence
def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ n

-- Sum of the first three terms of the geometric sequence
def sum_first_three_terms (a q: α) : α :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

theorem common_ratio_of_geometric_seq (a q : α) (h : sum_first_three_terms a q = 3 * a) : q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_seq_l560_560974


namespace remove_wallpaper_time_l560_560659

theorem remove_wallpaper_time 
    (total_walls : ℕ := 8)
    (remaining_walls : ℕ := 7)
    (time_for_remaining_walls : ℕ := 14) :
    time_for_remaining_walls / remaining_walls = 2 :=
by
sorry

end remove_wallpaper_time_l560_560659


namespace percy_swimming_hours_l560_560925

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l560_560925


namespace sally_sews_number_of_buttons_l560_560938

theorem sally_sews_number_of_buttons :
  let monday_shirts := 4 * 5,
      monday_pants := 2 * 3,
      monday_jacket := 1 * 10,
      tuesday_shirts := 3 * 5,
      tuesday_pants := 1 * 3,
      tuesday_jackets := 2 * 10,
      wednesday_shirts := 2 * 5,
      wednesday_pants := 3 * 3,
      wednesday_jacket := 1 * 10 in
  (monday_shirts + monday_pants + monday_jacket) + 
  (tuesday_shirts + tuesday_pants + tuesday_jackets) + 
  (wednesday_shirts + wednesday_pants + wednesday_jacket) = 103 :=
by
  sorry

end sally_sews_number_of_buttons_l560_560938


namespace number_of_possible_values_for_b_l560_560177

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 2 ∧ ∀ b : ℕ, (b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4) → b = 4 ∨ b = 5 :=
sorry

end number_of_possible_values_for_b_l560_560177


namespace expand_expression_l560_560716

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560716


namespace geometric_sequence_a2_value_l560_560115

theorem geometric_sequence_a2_value
    (a : ℕ → ℝ)
    (h1 : a 1 = 1/5)
    (h3 : a 3 = 5)
    (geometric : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) :
    a 2 = 1 ∨ a 2 = -1 := by
  sorry

end geometric_sequence_a2_value_l560_560115


namespace expand_polynomial_eq_l560_560708

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560708


namespace palindrome_count_valid_l560_560427

def is_palindrome (t : String) : Prop :=
  t = t.reverse

def is_valid_time (h m : Nat) : Prop :=
  h < 24 ∧ m < 60

def generate_palindromes : List String :=
  let hours := List.range 24
  let minutes := List.range 60
  hours.bind (λ h => minutes.filterMap (λ m => 
    let t := s! "{h:02d}{m:02d}"
    if is_valid_time h m ∧ is_palindrome t then some t else none))

noncomputable def palindrome_count := generate_palindromes.length

theorem palindrome_count_valid : palindrome_count = 99 := by 
  sorry

end palindrome_count_valid_l560_560427


namespace value_of_f_neg2017_l560_560386

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom periodicity (x : ℝ) : f x = -f (x + 2)
axiom functional_form (h : 0 ≤ x ∧ x ≤ 2) : f x = x * (x - 2)

-- Question with expected answer
theorem value_of_f_neg2017 : f (-2017) = 1 := sorry

end value_of_f_neg2017_l560_560386


namespace max_annual_profit_l560_560179

-- Conditions
def fixed_RD_cost (ten_thousand_yuan : ℝ) := 50
def variable_cost (x : ℝ) := 80 * x
def revenue_x_leq_20 (x : ℝ) := 180 * x - 2 * x^2
def revenue_x_gt_20 (x : ℝ) := 70 * x + 2000 - (9000 / (x + 1))
def cost (x : ℝ) := fixed_RD_cost 1 + variable_cost x

-- Define Profit
def profit (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 20 then
    revenue_x_leq_20 x - cost x
  else if h : x > 20 then
    revenue_x_gt_20 x - cost x
  else
    0

-- Prove that the maximum profit is 1360 at x = 29
theorem max_annual_profit :
  ∃ x : ℝ, x = 29 ∧ profit 29 = 1360 :=
  by
    exists 29
    split
    · rfl
    · sorry

end max_annual_profit_l560_560179


namespace x_intercept_of_perpendicular_line_l560_560993

theorem x_intercept_of_perpendicular_line (x y : ℝ) (b : ℕ) :
  let line1 := 2 * x + 3 * y
  let slope1 := -2/3
  let slope2 := 3/2
  let y_intercept := -1
  let perp_line := slope2 * x + y_intercept
  let x_intercept := 2/3
  line1 = 12 → perp_line = 0 → x = x_intercept :=
by
  sorry

end x_intercept_of_perpendicular_line_l560_560993


namespace derivative_y_l560_560404

def y (x : ℝ) : ℝ := x^2 * sin x

theorem derivative_y (x : ℝ) : deriv y x = 2 * x * sin x + x^2 * cos x :=
by
  sorry

end derivative_y_l560_560404


namespace rectangle_width_to_length_ratio_l560_560861

theorem rectangle_width_to_length_ratio {w : ℕ} 
  (h1 : ∀ (l : ℕ), l = 10)
  (h2 : ∀ (p : ℕ), p = 32)
  (h3 : ∀ (P : ℕ), P = 2 * 10 + 2 * w) :
  (w : ℚ) / 10 = 3 / 5 :=
by
  sorry

end rectangle_width_to_length_ratio_l560_560861


namespace sum_of_midpoints_l560_560973

theorem sum_of_midpoints 
  (a b c d e f : ℝ)
  (h1 : a + b + c = 15)
  (h2 : d + e + f = 15) :
  ((a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15) ∧ 
  ((d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15) :=
by
  sorry

end sum_of_midpoints_l560_560973


namespace sqrt_inequality_l560_560162

theorem sqrt_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x + y + z = 1) :
  sqrt (7 * x + 3) + sqrt (7 * y + 3) + sqrt (7 * z + 3) ≤ 7 :=
sorry

end sqrt_inequality_l560_560162


namespace gcd_adjacent_numbers_l560_560629

/-- Arrange the numbers 2, 3, 4, 6, 8, 9, 12, 15 in a row such that the greatest common divisor (GCD) of any two adjacent numbers is greater than 1. The number of possible arrangements is 1296. -/
theorem gcd_adjacent_numbers (numbers : List ℕ) (h : numbers = [2, 3, 4, 6, 8, 9, 12, 15]) : 
  ∃ n : ℕ, n = 1296 ∧ (∀ (l : List ℕ), l ∈ numbers.permutations → 
    ∀ i, i < l.length - 1 → Nat.gcd (l.nth_le i sorry) (l.nth_le (i + 1) sorry) > 1) :=
  sorry

end gcd_adjacent_numbers_l560_560629


namespace tan_pi_minus_theta_l560_560752

noncomputable def theta := sorry -- This represents the angle theta in radians

def sin_theta : ℝ := -3 / 4
def cos_theta : ℝ := sqrt (1 - sin_theta ^ 2)

theorem tan_pi_minus_theta : 
  θ > (3 * Real.pi / 2) ∧ θ < 2 * Real.pi ∧ Real.sin θ = sin_theta ∧ Real.cos θ = cos_theta -> 
  Real.tan (Real.pi - θ) = 3 * sqrt 7 / 7 := 
by 
  intros h;
  sorry

end tan_pi_minus_theta_l560_560752


namespace construct_triangle_l560_560229

noncomputable theory

variables {A B B₁ C : Type} 
variables (l : Line) (α φ : ℝ)
variables (angle_A angle_B : ℝ) 
variables (on_line_l : ∀ C, C ∈ l)

def symmetric_point (B : Type) (l : Line) : Type := -- definition of the symmetric point
  sorry

def circumscribed_angle (A B : Type) (angle_ABC : ℝ) : ℝ :=
  180 - (angle_A - angle_B) + 2 * α

def locate_vertex_C (A B B₁ : Type) (l : Line) (angle_ABC : ℝ) : Prop :=
  ∃ C, C ∈ l ∧ ∃ arc, arc ∋ C ∧ arc.subtends angle_ABC C B₁

theorem construct_triangle 
  (A B : Type) {l : Line} (angle_A angle_B : ℝ) (φ : ℝ) (hφ : angle_A - angle_B = φ) :
  locate_vertex_C A B (symmetric_point B l) l (circumscribed_angle A B (angle_A - angle_B)) :=
sorry

end construct_triangle_l560_560229


namespace area_comparison_l560_560641

open EuclideanGeometry

variable {Point : Type} [EuclideanSpace Point]

def midpoint (p1 p2 : Point) : Point := sorry -- this needs a proper definition

def intersect (l1 l2 : Set Point) : Point := sorry -- this needs a proper definition

variable (A B C D : Point)
variable (squareABCD : IsSquare A B C D)

def E : Point := midpoint B C
def F : Point := midpoint C D
def AE := lineThrough A E
def BF := lineThrough B F
def K : Point := intersect AE BF

theorem area_comparison
  (hABCD : IsSquare A B C D)
  (hE : E = midpoint B C)
  (hF : F = midpoint C D)
  (hK : K = intersect AE BF) :
  triangleArea A K F > quadrilateralArea K E C F := sorry

end area_comparison_l560_560641


namespace range_of_a3_range_of_q_sequence_properties_l560_560764

-- Proof Problem 1
theorem range_of_a3 (a : ℕ → ℝ) (h1 : a 1 = 1) (h_cond : ∀ n, a (n + 1) ^ 2 + a n ^ 2 < (5 / 2) * a (n + 1) * a n) 
(h2 : a 2 = 3 / 2) (h4 : a 4 = 4) :
  2 < a 3 ∧ a 3 < 3 :=
sorry

-- Proof Problem 2
theorem range_of_q (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (h_geo : ∀ n, a (n + 1) = q * a n)
(h_sum_cond : ∀ n, (1 / 2) * (finset.range (n + 1)).sum a < (finset.range (n + 2)).sum a ∧ (finset.range (n + 2)).sum a < 2 * (finset.range (n + 1)).sum a) :
  1 / 2 < q ∧ q < 1 :=
sorry

-- Proof Problem 3
theorem sequence_properties (a : ℕ → ℝ) (k : ℕ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1) 
(h_sum : (finset.range (k + 1)).sum a = 120) (h1 : a 1 = 1) 
(h_cond : ∀ n, (1/2) < a (n+1) / a n ∧ a (n+1) / a n < 2) :
  k = 16 ∧ (∀ n, a n = (13 * n + 2) / 15) :=
sorry

end range_of_a3_range_of_q_sequence_properties_l560_560764


namespace tangent_lines_through_point_l560_560336

theorem tangent_lines_through_point {x y : ℝ} (h_circle : (x-1)^2 + (y-1)^2 = 1)
  (h_point : ∀ (x y: ℝ), (x, y) = (2, 4)) :
  (x = 2 ∨ 4 * x - 3 * y + 4 = 0) :=
sorry

end tangent_lines_through_point_l560_560336


namespace min_m_quad_eq_integral_solutions_l560_560568

theorem min_m_quad_eq_integral_solutions :
  (∃ m : ℕ, (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42) ∧ m > 0) →
  (∃ m : ℕ, m = 130 ∧ (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42)) :=
by
  sorry

end min_m_quad_eq_integral_solutions_l560_560568


namespace trigonometric_identity_l560_560498

theorem trigonometric_identity (α β γ : ℝ) :
  (sin α * sin (β - γ))^3 + (sin β * sin (γ - α))^3 + (sin γ * sin (α - β))^3 =
  3 * sin α * sin β * sin γ * sin (α - β) * sin (β - γ) * sin (γ - α) :=
by
  sorry

end trigonometric_identity_l560_560498


namespace expand_product_l560_560697

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560697


namespace expand_expression_l560_560718

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560718


namespace root_of_equation_l560_560968

theorem root_of_equation :
  ∀ x : ℝ, (x - 3)^2 = x - 3 ↔ x = 3 ∨ x = 4 :=
by
  sorry

end root_of_equation_l560_560968


namespace expand_polynomial_eq_l560_560710

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560710


namespace part1_part2_l560_560400

-- Definition of the function and its derivative
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + b * x + c
def f' (x : ℝ) (b : ℝ) : ℝ := x^2 - x + b

-- Part 1: Proving that if f is increasing on (-∞, +∞), then b ≥ 1/4
theorem part1 (b : ℝ) (c : ℝ) (h : ∀ x : ℝ, f' x b ≥ 0) : b ≥ 1/4 := 
sorry

-- Part 2: Proving the range of c under given conditions
theorem part2 (c : ℝ) (h1 : f' 1 0 = 0) (h2 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f x 0 c < c^2) :
  c > (3 + Real.sqrt 33) / 6 ∨ c < (3 - Real.sqrt 33) / 6 :=
sorry

end part1_part2_l560_560400


namespace sum_even_numbers_l560_560254

theorem sum_even_numbers (n : ℕ) (h1 : n % 2 = 1) (h2 : (∑ k in range (n // 2), 2 * (k + 1)) = 89 * 90) : n = 179 :=
by
  sorry

end sum_even_numbers_l560_560254


namespace correct_operation_B_l560_560574

theorem correct_operation_B (a b : ℝ) : - (a - b) = -a + b := 
by sorry

end correct_operation_B_l560_560574


namespace expand_expression_l560_560677

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560677


namespace triangle_ratio_l560_560270

theorem triangle_ratio (a b c : ℕ) (r s : ℕ) (h1 : a = 9) (h2 : b = 15) (h3 : c = 18) (h4 : r + s = a) (h5 : r < s) : r * 2 = s :=
by
  sorry

end triangle_ratio_l560_560270


namespace solve_system_of_equations_l560_560172

theorem solve_system_of_equations (x y_1 y_2 y_3: ℝ) (n : ℤ) (h1 : -3 ≤ n) (h2 : n ≤ 3)
  (h_eq1 : (1 - x^2) * y_1 = 2 * x)
  (h_eq2 : (1 - y_1^2) * y_2 = 2 * y_1)
  (h_eq3 : (1 - y_2^2) * y_3 = 2 * y_2)
  (h_eq4 : y_3 = x) :
  y_1 = Real.tan (2 * n * Real.pi / 7) ∧
  y_2 = Real.tan (4 * n * Real.pi / 7) ∧
  y_3 = Real.tan (n * Real.pi / 7) ∧
  x = Real.tan (n * Real.pi / 7) :=
sorry

end solve_system_of_equations_l560_560172


namespace part_one_part_two_part_three_l560_560989

-- Given definitions and probabilities
def prob_A_shoots_once_hits := 1/2
def prob_B_shoots_once_hits := 1/3

-- Probability computation when each shoots once, and at least one hits the target.
theorem part_one :
  let prob_achieve_goal := 2/3 in
  prob_A_shoots_once_hits * (1 - prob_B_shoots_once_hits) +
  (1 - prob_A_shoots_once_hits) * prob_B_shoots_once_hits +
  prob_A_shoots_once_hits * prob_B_shoots_once_hits = prob_achieve_goal :=
sorry

-- Probability computation when each shoots twice, and at least three hits the target.
theorem part_two :
  let prob_achieve_goal := 7/36 in
  (binomial 2 2 * (1/2)^2 * (2/3)^0 * binomial 2 1 * (1/3)^1 * (2/3)^1 +
   binomial 2 1 * (1/2)^1 * (1/2)^1 * binomial 2 2 * (1/3)^2 * (2/3)^0 +
   binomial 2 2 * (1/2)^2 * (2/3)^0 * binomial 2 2 * (1/3)^2 * (2/3)^0) = prob_achieve_goal :=
sorry

-- Confidence assertion with 99% certainty when each shoots five times.
theorem part_three : 
  let prob_achieve_goal := 242/243 in
  1 - (binomial 5 0 * (1/2)^5 * binomial 5 0 * (2/3)^5) > 0.99 :=
sorry

end part_one_part_two_part_three_l560_560989


namespace sum_of_intersection_coordinates_l560_560438

def Point (ℝ) := (ℝ, ℝ)

noncomputable def midpoint (P Q : Point ℝ) : Point ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def line (P Q : Point ℝ) : (ℝ × ℝ → Prop) :=
  λ R, R.2 = ((Q.2 - P.2) / (Q.1 - P.1)) * (R.1 - P.1) + P.2

noncomputable def intersection (l1 l2 : ℝ × ℝ → Prop) : Point ℝ :=
  let x := (l2 (1, 0) - l1 (1, 0)) / ((l1 (1, 1) - l1 (1, 0)) - (l2 (1, 1) - l2 (1, 0)))
  let y := l1 (x, 0).2
  (x, y)

theorem sum_of_intersection_coordinates :
  let A : Point ℝ := (0, 8)
  let B : Point ℝ := (0, 0)
  let C : Point ℝ := (10, 0)
  let D : Point ℝ := midpoint A B
  let E : Point ℝ := midpoint B C
  let AE : ℝ × ℝ → Prop := line A E
  let CD : ℝ × ℝ → Prop := line C D
  let F := intersection AE CD
  F.1 + F.2 = 6 :=
by
  sorry

-- Definitions for clarity but not used in the statement.
#eval midpoint (0, 8) (0, 0) -- (0, 4)
#eval midpoint (0, 0) (10, 0) -- (5, 0)
#eval line (0, 8) (5, 0) (1, 1) -- y = -8/5 x + 8
#eval line (10, 0) (0, 4) (1, 1) -- y = -2/5 x + 4
#eval intersection (λ r => r.2 = -8/5 * r.1 + 8) (λ r => r.2 = -2/5 * r.1 + 4) -- (10/3, 8/3)


end sum_of_intersection_coordinates_l560_560438


namespace expression_value_l560_560542

theorem expression_value (a b c : ℚ) (h₁ : b = 8) (h₂ : c = 5) (h₃ : a * b * c = 2 * (a + b + c) + 14) : 
  (c - a) ^ 2 + b = 8513 / 361 := by 
  sorry

end expression_value_l560_560542


namespace correct_values_of_x_l560_560648

noncomputable def matrix_expression (x : ℝ) : ℝ := (3 * x) * (2 * x - 1) - (x + 1) * (2 * x)

theorem correct_values_of_x (x : ℝ) :
  matrix_expression x = 2 ↔
  (x = (5 + Real.sqrt 57) / 8) ∨ (x = (5 - Real.sqrt 57) / 8) :=
begin
  sorry
end

end correct_values_of_x_l560_560648


namespace range_of_inequality_l560_560406

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x else x

theorem range_of_inequality : {x : ℝ | f (x + 1) < f (2 * x)} = {x : ℝ | x > 1} :=
by
  sorry

end range_of_inequality_l560_560406


namespace imaginary_part_of_z_l560_560050

noncomputable def complex_number : ℂ := (complex.I / (1 + complex.I))

theorem imaginary_part_of_z : complex.im complex_number = 1 / 2 :=
by
  -- problem statement and necessary conditions have been set up
  sorry

end imaginary_part_of_z_l560_560050


namespace sum_of_nine_consecutive_quotients_multiple_of_9_l560_560363

def a (i : ℕ) : ℕ := (10^(2 * i) - 1) / 9
def q (i : ℕ) : ℕ := a i / 11
def s (i : ℕ) : ℕ := q i + q (i + 1) + q (i + 2) + q (i + 3) + q (i + 4) + q (i + 5) + q (i + 6) + q (i + 7) + q (i + 8)

theorem sum_of_nine_consecutive_quotients_multiple_of_9 (i n : ℕ) (h : n > 8) 
  (h2 : i ≤ n - 8) : s i % 9 = 0 :=
sorry

end sum_of_nine_consecutive_quotients_multiple_of_9_l560_560363


namespace cheaper_fuji_shimla_l560_560548

variable (S R F : ℝ)
variable (h : 1.05 * (S + R) = R + 0.90 * F + 250)

theorem cheaper_fuji_shimla : S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 :=
by
  sorry

end cheaper_fuji_shimla_l560_560548


namespace min_value_of_roots_quad_poly_l560_560140

-- Define the context and conditions
def monic_deg2_poly (P : ℝ[X]) : Prop := 
  P.degree = 2 ∧ P.leadingCoeff = 1

theorem min_value_of_roots_quad_poly (P : ℝ[X]) (x1 x2 : ℝ) 
  (h_monic : monic_deg2_poly P) 
  (h_cond : P.eval 1 ≥ P.eval 0 + 3) 
  (h_roots : P.root x1 ∧ P.root x2) : 
  (x1^2 + 1) * (x2^2 + 1) ≥ 4 :=
sorry

end min_value_of_roots_quad_poly_l560_560140


namespace floor_width_l560_560612

theorem floor_width (W : ℕ) : 
    (∃ (width_strip : ℕ), (area_rug : ℕ) 
    (length_floor : ℕ) (width_floor : ℕ) (length_rug : ℕ) (width_rug : ℕ),
    width_strip = 4 ∧ area_rug = 204 ∧ length_floor = 25 ∧ 
    length_rug = length_floor - 2 * width_strip ∧
    width_rug = W - 2 * width_strip ∧
    area_rug = length_rug * width_rug) → W = 20 :=
by
  sorry

end floor_width_l560_560612


namespace sixth_root_of_large_number_l560_560329

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l560_560329


namespace expand_expression_l560_560666

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560666


namespace selection_of_1325_has_three_rel_prime_selection_of_1324_no_three_rel_prime_l560_560164

open Set

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1987}

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem selection_of_1325_has_three_rel_prime :
  ∀ S : Finset ℕ, S ⊆ M ∧ S.card = 1325 →
  ∃ a b c ∈ S, is_pairwise_rel_prime a b c :=
sorry

theorem selection_of_1324_no_three_rel_prime :
  ∀ S : Finset ℕ, S ⊆ M ∧ S.card = 1324 →
  ∃! abc : Finset ℕ, abc ⊆ S ∧ abc.card = 3 ∧ ¬ is_pairwise_rel_prime abc :=
sorry

end selection_of_1325_has_three_rel_prime_selection_of_1324_no_three_rel_prime_l560_560164


namespace cut_and_assemble_l560_560157

def figure : Type := -- Assuming the type figure describes the initial figure drawn on graph paper.
sorry

def can_cut_into_5_triangles_and_form_square (f : figure) : Prop :=
 ∃ (triangles : list triangle), triangles.length = 5 ∧ arrange_into_square triangles

/-- Given a figure drawn on graph paper, prove it is possible to cut it into 
    exactly 5 triangles and assemble them into a square -/
theorem cut_and_assemble (f : figure) : can_cut_into_5_triangles_and_form_square f :=
sorry

end cut_and_assemble_l560_560157


namespace tangents_not_necessarily_coincide_at_B_l560_560035

-- Definitions for points A and B, and the parabola y = x^2.
def parabola (x : ℝ) : ℝ := x^2

-- Conditions
variable (A B : ℝ × ℝ)
variable hA1 : A = (1, 1)
variable hB1 : B = (-3, 9)

-- Tangents at points A and B.
variable (circle : ℝ × ℝ → ℝ)
variable tangent_parabola_A tangent_circle_A : linear_map ℝ (ℝ × ℝ) (ℝ × ℝ)
variable tangent_parabola_B tangent_circle_B : linear_map ℝ (ℝ × ℝ) (ℝ × ℝ)

-- Given intersections and tangency condition at point A.
axiom tangency_at_A : tangent_parabola_A = tangent_circle_A

-- Question: Are the tangents at point B necessarily coinciding?
theorem tangents_not_necessarily_coincide_at_B : ¬ (tangent_parabola_B = tangent_circle_B) :=
  sorry

end tangents_not_necessarily_coincide_at_B_l560_560035


namespace total_points_scored_l560_560150

-- Define the variables
def games : ℕ := 10
def points_per_game : ℕ := 12

-- Formulate the proposition to prove
theorem total_points_scored : games * points_per_game = 120 :=
by
  sorry

end total_points_scored_l560_560150


namespace expand_polynomial_eq_l560_560707

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560707


namespace hyperbola_equation_l560_560951

theorem hyperbola_equation (a b : ℝ) (O : ℝ × ℝ) (c : ℝ) (P Q : ℝ × ℝ)
  (h1 : O = (0, 0))
  (h2 : ∃ e : ℝ, c^2 = a^2 + b^2 ∧ e = 2)
  (h3 : ∃ m : ℝ, m = sqrt (3 / 5) ∧ line_eq := fun x => m * (x - c))
  (h4 : ∃ k x1 x2 y1 y2 : ℝ, y1 = sqrt (3 / 5) * (x1 - c) ∧ y2 = sqrt (3 / 5) * (x2 - c) ∧
    (x1, y1) = P ∧ (x2, y2) = Q ∧ (⊥ := line_slope_perp (P, O) (O, Q)) ∧ |PQ| = 4) :
  (a, b) = (1, sqrt 3) :=
sorry

end hyperbola_equation_l560_560951


namespace rate_per_sq_meter_l560_560534

theorem rate_per_sq_meter (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : total_cost = 16500) : 
  total_cost / (length * width) = 800 :=
by
  sorry

end rate_per_sq_meter_l560_560534


namespace sector_area_l560_560788

theorem sector_area (θ r arc_length : ℝ) (h_arc_length : arc_length = r * θ) (h_values : θ = 2 ∧ arc_length = 2) :
  1 / 2 * r^2 * θ = 1 := by
  sorry

end sector_area_l560_560788


namespace percy_swimming_hours_l560_560924

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l560_560924


namespace proof_of_inequality_l560_560649

noncomputable def function_with_conditions 
  (f : ℝ → ℝ) 
  (f'' : ℝ → ℝ) 
  (h_neg : ∀ x : ℝ, f x < 0) 
  (h_ineq : ∀ x : ℝ, 2 * f'' x + f x > 0) 
  (x1 x2 : ℝ) 
  (h_lt : x1 < x2) : 
  Prop :=
  f(x1)^2 > real.exp((x1 - x2) / 2) * f(x2)^2

theorem proof_of_inequality 
  (f : ℝ → ℝ)
  (f'' : ℝ → ℝ)
  (h_neg : ∀ x : ℝ, f x < 0)
  (h_ineq : ∀ x : ℝ, 2 * f'' x + f x > 0)
  (x1 x2 : ℝ)
  (h_lt : x1 < x2) : 
  function_with_conditions f f'' h_neg h_ineq x1 x2 h_lt :=
sorry

end proof_of_inequality_l560_560649


namespace inequality_proof_l560_560473

variable (a b c : ℝ)
variable (h_pos : a > 0) (h_pos2 : b > 0) (h_pos3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) > 1 / 2 := by
  sorry

end inequality_proof_l560_560473


namespace coefficient_ab3c_in_expansion_l560_560185

theorem coefficient_ab3c_in_expansion :
  let exp := (λ (a b c : ℕ), (a + b)^2 * (b + c)^3)
  let ab3c := (1 * b^3 * c)
  let coeff_ab3c := 6
  exp = (λ (a b c : ℕ), coeff_ab3c) := by
  sorry

end coefficient_ab3c_in_expansion_l560_560185


namespace sin_2alpha_eq_f_increasing_interval_l560_560027

noncomputable def sin_alpha := 4 / 5
noncomputable def alpha := real.sin_pi_sub (4/5)

theorem sin_2alpha_eq : α \in (0, real.pi / 2) -> sin 2 * α - (cos (α / 2))^2 = 4 / 25 :=
by {
  sorry
}

theorem f_increasing_interval : α \in (0, real.pi / 2) -> sin (real.pi - α) = 4 / 5 -> 
(intervals k : \mathbb{Z}, f x = 5 / 6 * cos α * sin (2 * x) - 1 / 2 * cos (2 * x)) -> 
(x > 0, f(x) > f(y)) = (\forall k \in Int, [k real.pi - real.pi / 8, k real.pi + 3 * real.pi / 8]) :=
by {
  sorry
}

end sin_2alpha_eq_f_increasing_interval_l560_560027


namespace matrix_pow_C_50_l560_560124

def C : Matrix (Fin 2) (Fin 2) ℤ := 
  !![3, 1; -4, -1]

theorem matrix_pow_C_50 : C^50 = !![101, 50; -200, -99] := 
  sorry

end matrix_pow_C_50_l560_560124


namespace c_linear_combination_of_a_b_l560_560843

-- Definitions of vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

-- Theorem stating the relationship between vectors a, b, and c
theorem c_linear_combination_of_a_b :
  c = (1 / 2 : ℝ) • a + (-3 / 2 : ℝ) • b :=
  sorry

end c_linear_combination_of_a_b_l560_560843


namespace general_term_sum_of_first_n_terms_l560_560042

open Real

-- Definitions and conditions
def a (n : ℕ) : ℝ := 5 * (sqrt 2)^(n-1)

-- Problem 1: Proof of the general term of the sequence
theorem general_term (a1 a3 a4 a6 : ℝ) (h1 : a1 + a3 = 10) (h2 : a4 + a6 = 80) :
  ∃ q, q = sqrt 2 ∧ a1 = 5 ∧ ∀ n, a n = 5 * (sqrt 2)^(n-1) :=
sorry

-- Definitions for summation problem
def b (n : ℕ) : ℝ := (2 * n - 1) * a n
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

-- Problem 2: Proof of the sum of the first n terms
theorem sum_of_first_n_terms (n : ℕ) : 
  S n = 5 * ( (1 - (sqrt 2)^n) / (1 - sqrt 2) + 
              (2 * sqrt 2 - (2 * n - 3) * (sqrt 2)^n) / (sqrt 2 - 1)^2 ) :=
sorry

end general_term_sum_of_first_n_terms_l560_560042


namespace find_angle_and_sum_of_sides_l560_560100

noncomputable def triangle_conditions 
    (a b c : ℝ) (C : ℝ)
    (area : ℝ) : Prop :=
  a^2 + b^2 - c^2 = a * b ∧
  c = Real.sqrt 7 ∧
  area = (3 * Real.sqrt 3) / 2 

theorem find_angle_and_sum_of_sides
    (a b c C : ℝ)
    (area : ℝ)
    (h : triangle_conditions a b c C area) :
    C = Real.pi / 3 ∧ a + b = 5 := by
  sorry

end find_angle_and_sum_of_sides_l560_560100


namespace sequence_divisibility_24_l560_560650

theorem sequence_divisibility_24 :
  ∀ (x : ℕ → ℕ), (x 0 = 2) → (x 1 = 3) →
    (∀ n : ℕ, x (n+2) = 7 * x (n+1) - x n + 280) →
    (∀ n : ℕ, (x n * x (n+1) + x (n+1) * x (n+2) + x (n+2) * x (n+3) + 2018) % 24 = 0) :=
by
  intro x h1 h2 h3
  sorry

end sequence_divisibility_24_l560_560650


namespace largest_of_three_numbers_l560_560584

noncomputable def hcf := 23
noncomputable def factors := [11, 12, 13]

/-- The largest of the three numbers, given the H.C.F is 23 and the other factors of their L.C.M are 11, 12, and 13, is 39468. -/
theorem largest_of_three_numbers : hcf * factors.prod = 39468 := by
  sorry

end largest_of_three_numbers_l560_560584


namespace teresa_class_size_l560_560961

theorem teresa_class_size :
  ∃ (a : ℤ), 50 < a ∧ a < 100 ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 2) ∧ 
  (a % 5 = 2) ∧ 
  a = 62 := 
by {
  sorry
}

end teresa_class_size_l560_560961


namespace expand_expression_l560_560676

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560676


namespace select_three_consecutive_circles_l560_560760

noncomputable def number_of_ways_to_select_three_consecutive_circles:
  (grid_length : ℤ) →
  (grid_directions : ℕ) →
  ℤ
| l, d := (l - 2) * d

theorem select_three_consecutive_circles 
(grid_length: ℤ) 
(grid_height: ℤ) 
(horizontal_ways: ℤ) 
(diagonal1_ways_diagonal2_ways : ℤ) : 
grid_length = 7 → 
grid_height = 7 → 
horizontal_ways = 21 → 
diagonal1_ways_diagonal2_ways = 18 → 
number_of_ways_to_select_three_consecutive_circles grid_length 3 + 
number_of_ways_to_select_three_consecutive_circles grid_height 2 + 
number_of_ways_to_select_three_consecutive_circles grid_height 2 
= 57 := 
by 
  intros; 
  rw [number_of_ways_to_select_three_consecutive_circles, 
      number_of_ways_to_select_three_consecutive_circles, 
      number_of_ways_to_select_three_consecutive_circles, 
      h_2, h_3]; 
  exact rfl

end select_three_consecutive_circles_l560_560760


namespace fruit_total_l560_560550

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l560_560550


namespace sum_of_digits_9N_l560_560002

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9N (N : ℕ) (h : ∀ i j : ℕ, i < j → (N.digit i) < (N.digit j)) :
  sum_of_digits (9 * N) = 9 :=
by
  sorry

end sum_of_digits_9N_l560_560002


namespace eccentricity_of_hyperbola_l560_560801

-- Definitions and conditions
def is_hyperbola (a b : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote_condition (a b : ℝ) := (b / a) = 2
def eccentricity (a : ℝ) := (sqrt (a^2 + (2*a)^2)) / a

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_asymptote : asymptote_condition a b) : 
  eccentricity a = sqrt 5 :=
by
  -- [Proof steps will be added here]
  sorry

end eccentricity_of_hyperbola_l560_560801


namespace students_per_section_l560_560621

theorem students_per_section (rows sections students : ℕ) (h1 : rows = 13) (h2 : sections = 2) (h3 : students = 52) :
  students / (rows * sections) = 2 :=
by
  rw [h1, h2, h3]
  calc 
    52 / (13 * 2) = 52 / 26 : by rw [mul_comm]
    ... = 2 : by norm_num

end students_per_section_l560_560621


namespace rectangular_field_perimeter_l560_560200

-- Definitions for conditions
def width : ℕ := 75
def length : ℕ := (7 * width) / 5
def perimeter (L W : ℕ) : ℕ := 2 * (L + W)

-- Statement to prove
theorem rectangular_field_perimeter : perimeter length width = 360 := by
  sorry

end rectangular_field_perimeter_l560_560200


namespace expand_expression_l560_560719

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560719


namespace license_plate_count_l560_560820

theorem license_plate_count :
  (∃ A B C D : Char,
    (is_letter A) ∧
    (is_letter_or_digit B) ∧
    (is_digit C) ∧
    (true) ∧  -- Any character is allowed for D
    (exactly_two_equal A B C D)) →
  (number_of_valid_plates = 29640) :=
sorry

end license_plate_count_l560_560820


namespace expand_polynomial_eq_l560_560702

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560702


namespace four_divides_n_l560_560901

variable {n : ℕ}
variable {a : Fin n → ℤ}

def is_valid_sequence (a : Fin n → ℤ) : Prop :=
  ∀ i, a i = 1 ∨ a i = -1

def sequence_condition (a : Fin n → ℤ) : Prop :=
  (Finset.finRange n).sum (λ i, a i * a ⟨(i + 1) % n, sorry⟩) = 0

theorem four_divides_n (a : Fin n → ℤ) (h1 : is_valid_sequence a) (h2 : sequence_condition a) : 4 ∣ n :=
sorry

end four_divides_n_l560_560901


namespace fried_corner_probability_l560_560360

def hop_probability (n : ℕ) (start : ℕ × ℕ) (end : ℕ × ℕ) : ℚ :=
-- Random hop probability calculation function
sorry

theorem fried_corner_probability :
  hop_probability 4 (0, 2) ∈ {(0, 0), (0, 3), (3, 0), (3, 3)} = 35 / 64 :=
sorry

end fried_corner_probability_l560_560360


namespace expand_expression_l560_560672

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560672


namespace B_pow_100_eq_B_l560_560880

def B : Matrix (Fin 3) (Fin 3) ℝ := !![ 
  [0, 1, 0], 
  [0, 0, 1], 
  [1, 0, 0] 
]

theorem B_pow_100_eq_B : B ^ 100 = B := 
by 
  sorry

end B_pow_100_eq_B_l560_560880


namespace expand_expression_l560_560667

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560667


namespace boy_sits_every_seat_l560_560265

theorem boy_sits_every_seat (n : ℕ) (moves : list ℕ)
  (h_diff : moves.nodup) (h_len : moves.length = n - 1) : 
  ∃ k : ℕ, n = 2 * k :=
by
  sorry

end boy_sits_every_seat_l560_560265


namespace largest_number_systematic_sampling_l560_560555

theorem largest_number_systematic_sampling (n k a1 a2: ℕ) (h1: n = 60) (h2: a1 = 3) (h3: a2 = 9) (h4: k = a2 - a1):
  ∃ largest, largest = a1 + k * (n / k - 1) := by
  sorry

end largest_number_systematic_sampling_l560_560555


namespace sqrt_domain_condition_l560_560823

theorem sqrt_domain_condition (x : ℝ) (h : ∃ y : ℝ, y = real.sqrt (2 - x)) : x ≤ 2 :=
by
  sorry

end sqrt_domain_condition_l560_560823


namespace part_i_l560_560581

theorem part_i (n : ℤ) : (∃ k : ℤ, n = 225 * k + 99) ↔ (n % 9 = 0 ∧ (n + 1) % 25 = 0) :=
by 
  sorry

end part_i_l560_560581


namespace eq_a_sub_b_l560_560087

theorem eq_a_sub_b (a b : ℝ) (i : ℂ) (hi : i * i = -1) (h1 : (a + 4 * i) * i = b + i) : a - b = 5 :=
by
  have := hi
  have := h1
  sorry

end eq_a_sub_b_l560_560087


namespace g_g_neg_1_l560_560799

def g (x : ℝ) : ℝ := x^(-2) + x^(-2) / (1 + x^(-2))

theorem g_g_neg_1 : g (g (-1)) = 88 / 117 := by
  sorry

end g_g_neg_1_l560_560799


namespace series_sum_l560_560306

theorem series_sum :
  (∑ n in Finset.range 100, 1 / ((n + 1) * real.sqrt n + n * real.sqrt (n + 1))) = 0.9 := 
sorry

end series_sum_l560_560306


namespace connected_graph_hamiltonian_l560_560272

-- Definition of a graph
structure Graph :=
  (V : Type) -- Vertices are of some type V
  (E : V → V → Prop) -- Edges are a binary relation on vertices

-- Hamiltonian cycle definition (simplified)
def isHamiltonian (G : Graph) : Prop :=
  ∃ (cycle : List G.V), -- There exists a list of vertices representing the Hamiltonian cycle
    ∀ v ∈ cycle, (∃ w ∈ cycle, G.E v w) -- Every vertex in the cycle has an edge to another vertex in the cycle

-- Degree of a vertex
def degree (G : Graph) (u : G.V) : ℕ :=
  {v : G.V | G.E u v}.card

-- Neighborhood of a vertex
def neighborhood (G : Graph) (u : G.V) : Set G.V :=
  {v | G.E u v}

theorem connected_graph_hamiltonian
  (G : Graph) (h_connected : ∀ u v : G.V, u ≠ v → ∃ p : List G.V, p.head = u ∧ p.ilast = v ∧ ∀ (i < p.length - 1), G.E (p.nth_le i (by linarith [Nat.lt_trans zero_lt_one i.lt_succ_self])) (p.nth_le (i + 1) (by sorry)))
  (h_order : 3 ≤ (G.V).card)
  (h_condition : ∀ (u v w : G.V) (h_path : G.E u v ∧ G.E v w), degree G u + degree G w ≥ (neighborhood G u ∪ neighborhood G v ∪ neighborhood G w).card) :
  isHamiltonian G :=
sorry

end connected_graph_hamiltonian_l560_560272


namespace problem_given_conditions_l560_560053

theorem problem_given_conditions (x y z : ℝ) 
  (h : x / 3 = y / (-4) ∧ y / (-4) = z / 7) : (3 * x + y + z) / y = -3 := 
by 
  sorry

end problem_given_conditions_l560_560053


namespace yard_length_l560_560109

theorem yard_length (trees : ℕ) (distance_per_gap : ℕ) (gaps : ℕ) :
  trees = 26 → distance_per_gap = 16 → gaps = trees - 1 → length_of_yard = gaps * distance_per_gap → length_of_yard = 400 :=
by 
  intros h_trees h_distance_per_gap h_gaps h_length_of_yard
  sorry

end yard_length_l560_560109


namespace symmetric_line_correct_l560_560537

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0

-- Define the point of symmetry
def point_of_symmetry : ℝ × ℝ := (1, -1)

-- Define the symmetric line we want to prove
def symmetric_line (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0

-- The main theorem statement
theorem symmetric_line_correct :
  ∀ (x y : ℝ), original_line x y →
  let (x', y') := (2 * point_of_symmetry.1 - x, 2 * point_of_symmetry.2 - y) in
  symmetric_line x' y' :=
sorry

end symmetric_line_correct_l560_560537


namespace circumcenter_on_circumcircle_of_right_triangle_l560_560381

variables {A B C P Q R : Point}

-- Definition of a right triangle
def is_right_triangle (A B C : Point) : Prop :=
  ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ (C = Poisition of right triangle ABC at angle 90)

-- Definition of excircle tangent points
def is_tangent_points (A B C P Q R : Point) : Prop :=
  ∃ excircle_ABC_P excircle_ABC_Q excircle_ABC_R : Circle,
    excircle_ABC_P.is_tangent (line_segment B C) P ∧
    excircle_ABC_Q.is_tangent (line_segment C A) Q ∧
    excircle_ABC_R.is_tangent (line_segment A B) R

-- Definition that a point O is the circumcenter of a triangle
def is_circumcenter (O : Point) (P Q R : Point) : Prop :=
  dist O P = dist O Q ∧ dist O Q = dist O R

-- Definition that quadrilateral points are cyclic
def is_cyclic (A B C O : Point) : Prop :=
  ∃ circumcircle : Circle, circumcircle.contains A ∧ circumcircle.contains B ∧ circumcircle.contains C ∧ circumcircle.contains O

-- The theorem to prove
theorem circumcenter_on_circumcircle_of_right_triangle
  (h_right_triangle : is_right_triangle A B C)
  (h_tangent_points : is_tangent_points A B C P Q R) :
  ∃ O : Point, is_circumcenter O P Q R ∧ is_cyclic A B C O := 
sorry

end circumcenter_on_circumcircle_of_right_triangle_l560_560381


namespace min_value_eq_18sqrt3_l560_560785

noncomputable def min_value (x y : ℝ) (h : x + y = 5) : ℝ := 3^x + 3^y

theorem min_value_eq_18sqrt3 {x y : ℝ} (h : x + y = 5) : min_value x y h ≥ 18 * Real.sqrt 3 := 
sorry

end min_value_eq_18sqrt3_l560_560785


namespace find_real_solutions_l560_560594

theorem find_real_solutions :
  {x : ℝ | (real.cbrt (4 * x - 1) + real.cbrt (4 * x + 1) = real.cbrt (8 * x))} =
  {0, 1 / 4, -1 / 4} :=
sorry

end find_real_solutions_l560_560594


namespace expand_polynomial_eq_l560_560706

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560706


namespace equal_angles_l560_560227

theorem equal_angles
  (A B C X Y : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space X]
  [metric_space Y]
  (h1 : metric_space.dist A B ≠ 0)
  (h2 : metric_space.dist A C ≠ 0)
  (h3 : metric_space.dist B C ≠ 0)
  (h4 : ∀ x : X, x ∈ metric_space.dist A B)
  (h5 : ∀ y : Y, y ∈ metric_space.dist A B)
  (h6 : (metric_space.dist A X * metric_space.dist B X) / (metric_space.dist C X ^ 2) = 
        (metric_space.dist A Y * metric_space.dist B Y) / (metric_space.dist C Y ^ 2))
  (AngleACX : Type)
  (AngleBCY : Type)
  [measurable_space AngleACX]
  [measurable_space AngleBCY]
  (angle_ACX : AngleACX)
  (angle_BCY : AngleBCY) :
  angle_ACX = angle_BCY :=
sorry

end equal_angles_l560_560227


namespace minimum_shirts_for_savings_l560_560297

theorem minimum_shirts_for_savings (x : ℕ) : 75 + 8 * x < 16 * x ↔ 10 ≤ x :=
by
  sorry

end minimum_shirts_for_savings_l560_560297


namespace sum_x_leq_n_div_3_l560_560032

theorem sum_x_leq_n_div_3 (n : ℕ) (x : ℕ → ℝ) (h1 : n ≥ 3)
    (h2 : ∀ i, i < n → x i ∈ Set.Ici (-1))
    (h3 : ∑ i in Finset.range n, (x i)^3 = 0) :
    (∑ i in Finset.range n, x i) ≤ (n : ℝ) / 3 ∧ 
    (∑ i in Finset.range n, x i = (n : ℝ) / 3 ↔ ∃ k : ℕ, n = 9 * k ∧ 
       ∀ i < 9 * k, (i < k → x i = -1) ∧ (k ≤ i ∧ i < 9 * k → x i = 1/2)) := sorry

end sum_x_leq_n_div_3_l560_560032


namespace expand_polynomial_l560_560687

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560687


namespace a5_a6_values_b_n_general_formula_minimum_value_T_n_l560_560765

section sequence_problems

def sequence_n (n : ℕ) : ℤ :=
if n = 0 then 1
else if n = 1 then 1
else sequence_n (n - 2) + 2 * (-1)^(n - 2)

def b_sequence (n : ℕ) : ℤ :=
sequence_n (2 * n)

def S_n (n : ℕ) : ℤ :=
(n + 1) * (sequence_n n)

def T_n (n : ℕ) : ℤ :=
(S_n (2 * n) - 18)

theorem a5_a6_values :
  sequence_n 4 = -3 ∧ sequence_n 5 = 5 := by
  sorry

theorem b_n_general_formula (n : ℕ) :
  b_sequence n = 2 * n - 1 := by
  sorry

theorem minimum_value_T_n :
  ∃ n, T_n n = -72 := by
  sorry

end sequence_problems

end a5_a6_values_b_n_general_formula_minimum_value_T_n_l560_560765


namespace impossible_partition_l560_560447

theorem impossible_partition : ¬ ∃ (A B C : Finset ℕ), 
  (∀ x ∈ A, x ∈ Finset.range 101) ∧ (∀ x ∈ B, x ∈ Finset.range 101) ∧ 
  (∀ x ∈ C, x ∈ Finset.range 101) ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range 101 ∧ 
  (∑ x in A, x) % 102 = 0 ∧ (∑ x in B, x) % 203 = 0 ∧ 
  (∑ x in C, x) % 304 = 0 :=
by
  sorry

end impossible_partition_l560_560447


namespace second_smallest_five_digit_palindromic_prime_l560_560204

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem second_smallest_five_digit_palindromic_prime :
  ∃ n : ℕ, is_five_digit n ∧ is_palindrome n ∧ is_prime n ∧
  ∀ m : ℕ, is_five_digit m ∧ is_palindrome m ∧ 
  is_prime m → n ≠ 10301 → m ≠ 10301 → n ≤ m → n = 11011 :=
by
  sorry

end second_smallest_five_digit_palindromic_prime_l560_560204


namespace find_x_l560_560847

theorem find_x (P0 P1 P2 P3 P4 P5 : ℝ) (y : ℝ) (h1 : P1 = P0 * 1.10)
                                      (h2 : P2 = P1 * 0.85)
                                      (h3 : P3 = P2 * 1.20)
                                      (h4 : P4 = P3 * (1 - x/100))
                                      (h5 : y = 0.15)
                                      (h6 : P5 = P4 * 1.15)
                                      (h7 : P5 = P0) : x = 23 :=
sorry

end find_x_l560_560847


namespace no_rational_roots_l560_560330

theorem no_rational_roots (x : ℚ) : ¬(3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1 = 0) :=
by sorry

end no_rational_roots_l560_560330


namespace total_fruit_count_l560_560553

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l560_560553


namespace possible_to_form_square_l560_560871

def shape_covers_units : ℕ := 4

theorem possible_to_form_square (shape : ℕ) : ∃ n : ℕ, ∃ k : ℕ, n * n = shape * k :=
by
  use 4
  use 4
  sorry

end possible_to_form_square_l560_560871


namespace no_odd_multiples_between_1500_and_3000_l560_560816

theorem no_odd_multiples_between_1500_and_3000 :
  ∀ n : ℤ, 1500 ≤ n → n ≤ 3000 → (18 ∣ n) → (24 ∣ n) → (36 ∣ n) → ¬(n % 2 = 1) :=
by
  -- The proof steps would go here, but we skip them according to the instructions.
  sorry

end no_odd_multiples_between_1500_and_3000_l560_560816


namespace work_completion_days_l560_560583

theorem work_completion_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) :
  (q_efficiency = 0.4 * p_efficiency) → (p_days = 26) → (p_efficiency = 1 / p_days) →
  let q_days := 1 / q_efficiency in
  let combined_efficiency := p_efficiency + q_efficiency in
  let combined_days := 1 / combined_efficiency in
  combined_days ≈ 18.57 :=
by
  intro h1 h2 h3
  let p_eff := 1 / p_days
  let q_eff := 0.4 * p_eff
  have h4 : q_eff = 1 / 65 := by sorry
  have h5 : combined_efficiency = p_eff + q_eff := by sorry
  let combined_days := 1 / (p_eff + q_eff)
  have h6 : combined_days = 18.57 := by sorry
  exact h6

end work_completion_days_l560_560583


namespace equation_of_perpendicular_line_l560_560793

theorem equation_of_perpendicular_line 
  (b : ℝ) (hb : b = 1) 
  (m : ℝ) (hm : m = -2)
  (line_perpendicular : ∀ x : ℝ, y = (1 / 2) * x → has_slope x y (1 / 2)) :
  ∃ x y, equation_of_line l (-2) 1 = y :=
by 
  sorry

end equation_of_perpendicular_line_l560_560793


namespace three_points_inequality_l560_560516

theorem three_points_inequality (n : ℕ) (h : n ≥ 3) (points : Fin n → ℝ × ℝ) :
  ∃ A B C : Fin n, 
    let d := dist points in
    1 ≤ d A B / d A C ∧ d A B / d A C < (n.succ + 1) / (n.succ - 1) :=
sorry

end three_points_inequality_l560_560516


namespace definite_integral_value_l560_560309

noncomputable def integral_function (x : ℝ) : ℝ := sqrt (1 - x^2) + x

theorem definite_integral_value :
  ∫ x in 0..1, integral_function x = (Real.pi / 4) + (1 / 2) :=
by
  sorry

end definite_integral_value_l560_560309


namespace example_l560_560835

open Real

noncomputable def unique_zero_point (f : ℝ → ℝ) (a b : ℝ) :=
  (∃ x ∈ Ioo a b, f x = 0) ∧ (∀ y, f y = 0 → y ∈ Ioo a b)

theorem example (f : ℝ → ℝ) (hf_continuous : Continuous f)
  (h1 : unique_zero_point f 0 16)
  (h2 : unique_zero_point f 0 8)
  (h3 : unique_zero_point f 0 4)
  (h4 : unique_zero_point f 0 2) :
  f 2 * f 16 > 0 := 
sorry

end example_l560_560835


namespace certain_event_of_triangle_interior_angles_sum_l560_560243

theorem certain_event_of_triangle_interior_angles_sum :
  ∀ (T : Type) [Triangle T], sum_of_interior_angles T = 180 := by
sorry

end certain_event_of_triangle_interior_angles_sum_l560_560243


namespace expand_polynomial_eq_l560_560701

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560701


namespace expand_expression_l560_560675

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560675


namespace bert_spent_fraction_at_hardware_store_l560_560634

variable (f : ℝ)

def initial_money : ℝ := 41.99
def after_hardware (f : ℝ) := (1 - f) * initial_money
def after_dry_cleaners (f : ℝ) := after_hardware f - 7
def after_grocery (f : ℝ) := 0.5 * after_dry_cleaners f

theorem bert_spent_fraction_at_hardware_store 
(h1 : after_grocery f = 10.50) : 
  f = 0.3332 :=
by
  sorry

end bert_spent_fraction_at_hardware_store_l560_560634


namespace integral_evaluation_l560_560341

theorem integral_evaluation : 
  ∫ (x : ℝ) in (1 : ℝ)..(3 : ℝ), (2 * x - x ^ (-2)) = 22 / 3 :=
by
  sorry

end integral_evaluation_l560_560341


namespace smallest_K_for_triangle_l560_560655

theorem smallest_K_for_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) 
  : ∃ K : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → a + c > b → (a^2 + c^2) / b^2 > K) ∧ K = 1 / 2 :=
by
  sorry

end smallest_K_for_triangle_l560_560655


namespace cos_alpha_plus_2_cos_beta_eq_2_m_range_l560_560030

-- For part (1)
def f (x : ℝ) (α β : ℝ) := x^3 - 9 * x^2 * real.cos α + 48 * x * real.cos β + 18 * real.sin(α)^2
def g (x : ℝ) (α β : ℝ) := 3 * x^2 - 18 * x * real.cos α + 48 * real.cos β
def condition1 (α β : ℝ) := (∀ x, 1 < x ∧ x ≤ 2 → g x α β ≥ 0) ∧ (∀ x, 2 ≤ x ∧ x ≤ 4 → g x α β ≤ 0) ∧ g 2 α β = 0 ∧ g 4 α β ≤ 0

theorem cos_alpha_plus_2_cos_beta_eq_2 (α β : ℝ) (h : condition1 α β) : real.cos α + 2 * real.cos β = 2 := 
sorry

-- For part (2)
def varphi (x : ℝ) (α β : ℝ) := (1/3) * x^3 - 2 * x^2 * real.cos β + x * real.cos α
def hfn (x : ℝ) (α β : ℝ) := real.log (varphi x α β)
def condition2 (α β : ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → hfn (x + 1 - m) α β < hfn (2 * x + 2) α β

theorem m_range (α β : ℝ) (h : condition2 α β) : -1 < m ∧ m < 0 := 
sorry

end cos_alpha_plus_2_cos_beta_eq_2_m_range_l560_560030


namespace meeting_time_l560_560923

noncomputable def time_meeting_paramon_solomon  (S x y z : ℝ) (h1 : 12 = S / (2 * x)) (h2 : 2 * z = S / 2 + 2 * x) (h3 : 80 / 60 * (y + z) = S) : ℝ :=
  let t := (S / (2 * x)) in
  have ht : S / (2 * x) = 1 := by sorry,
  (12:ℝ) + 1

theorem meeting_time : time_meeting_paramon_solomon S x y z h1 h2 h3 = 13 :=
by
  sorry
  
end meeting_time_l560_560923


namespace total_flowers_received_l560_560156

-- Definitions
def pieces_of_each_flower : ℕ := 40
def number_of_flower_types : ℕ := 4

-- Statement to prove
theorem total_flowers_received (n : ℕ) (m : ℕ) (h1 : n = 40) (h2 : m = 4) : n * m = 160 :=
by 
  rw [h1, h2]
  sorry

end total_flowers_received_l560_560156


namespace points_in_groups_l560_560554

theorem points_in_groups (x y a b : ℕ) (hx : x = 10) (hy : y = 7)
    (h1 : a = 66) (h2 : b = 136) :
    ∃ x y, (x + y) = 17 ∧ 
           (x * (x - 1) / 2 + y * (y - 1) / 2 = 66) ∧ 
           ((x + y) * (x + y - 1) / 2 = 136) := 
  by
  use 10, 7
  split
  . exact rfl
  split
  . exact calc
    10 * 9 / 2 + 7 * 6 / 2 = 45 + 21 : by norm_num
    ... = 66 : by norm_num
  exact calc
    17 * 16 / 2 = 136 : by norm_num

end points_in_groups_l560_560554


namespace find_floors_l560_560623

theorem find_floors (a b : ℕ) 
  (h1 : 3 * a + 4 * b = 25)
  (h2 : 2 * a + 3 * b = 18) : 
  a = 3 ∧ b = 4 := 
sorry

end find_floors_l560_560623


namespace sequence_diff_l560_560127

theorem sequence_diff (x : ℕ → ℕ)
  (h1 : ∀ n, x n < x (n + 1))
  (h2 : ∀ n, 2 * n + 1 ≤ x (2 * n + 1)) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_l560_560127


namespace total_handshakes_l560_560828

theorem total_handshakes (n : ℕ) (h : n = 10) : ∃ k, k = (n * (n - 1)) / 2 ∧ k = 45 :=
by {
  sorry
}

end total_handshakes_l560_560828


namespace expand_product_l560_560696

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560696


namespace part1_part2_l560_560111

-- Define the curve E: x^2 + y^2 = 4
def isCircleE (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l: y = kx - 4
def isLineL (k x y : ℝ) : Prop := y = k * x - 4

-- Define points C and D with the condition they are intersection points
def isIntersectionC (k x y : ℝ) : Prop := isCircleE x y ∧ isLineL k x y
def isIntersectionD (k x y : ℝ) : Prop := isCircleE x y ∧ isLineL k x y

-- Define angle condition
def angleCondition := 30 * Real.pi / 180

-- Problem Part 1 statement
theorem part1 (k : ℝ) (existsC existsD : ∃ x y : ℝ, isIntersectionC k x y ∧ isIntersectionD k x y ∧ ∠ (0, 0) (1, 0) (1/2, sqrt 3 / 2) = angleCondition): k = sqrt 15 ∨ k = -sqrt 15 :=
sorry

-- Problem Part 2 definitions
-- Define the line x - y - 4 = 0
def isLineQ (x y : ℝ) : Prop := x - y = 4

-- Define the tangents QM and QN
def isTangentQM (x y qx qy : ℝ) : Prop := isCircleE x y ∧ isLineL 1 qx qy
def isTangentQN (x y qx qy : ℝ) : Prop := isCircleE x y ∧ isLineL 1 qx qy

-- Define moving point Q
def isPointQ (t : ℝ) : ℝ × ℝ := ⟨t, t - 4⟩

-- Problem Part 2 statement
theorem part2 (t : ℝ) :
  (∃ x y : ℝ, isTangentQM x y (isPointQ t).1 (isPointQ t).2 ∧ 
  ∃ x' y' : ℝ, isTangentQN x' y' (isPointQ t).1 (isPointQ t).2) →
  ∃ (x_fixed y_fixed : ℝ), x_fixed = 1 ∧ y_fixed = -1 := 
sorry

end part1_part2_l560_560111


namespace expand_polynomial_l560_560681

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560681


namespace pyramid_surface_area_l560_560257

variables {A B C D : Type} [EuclideanSpace V]

-- Define the pyramid ABCD with the given properties
noncomputable def pyramid (A B C D : V) : Prop :=
  ∑ θ in [angle A B C, angle A B D, angle A C D], θ = 180 ∧
  (distance A B = distance C D) ∧
  (distance A D = distance B C)

-- Define the conditions for face BCD having area s
variables (s : ℝ) (area_face_BCD : Set (EuclideanSpace V))

-- Define the surface area of the pyramid
def surface_area_pyramid (A B C D : EuclideanSpace V) (s : ℝ) : ℝ :=
  4 * s

-- The theorem stating the problem
theorem pyramid_surface_area (A B C D : V) (s : ℝ) (h : pyramid A B C D) : 
  surface_area_pyramid A B C D s = 4 * s :=
by sorry

end pyramid_surface_area_l560_560257


namespace initial_number_of_girls_l560_560945

theorem initial_number_of_girls (p : ℝ) (h : (0.4 * p - 2) / p = 0.3) : 0.4 * p = 8 := 
by
  sorry

end initial_number_of_girls_l560_560945


namespace minimum_area_APQB_l560_560802

-- Definition of the parabola and its properties
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of the focus of the parabola
def focus : (ℝ × ℝ) := (2, 0)

-- Definition of the line l passing through the focus F and intersecting the parabola at points A and B
def line_through_focus (m x y : ℝ) : Prop := x = m * y + 2

-- Definition of points A and B on the parabola
def point_on_parabola (y : ℝ) : (ℝ × ℝ) := (y^2 / 8, y)

-- Definitions of points P and Q on the tangent lines at A and B and intersecting the y-axis
def tangent_y_intercept (x y : ℝ) : ℝ := y / 2

-- Function representing the area of quadrilateral APQB
def area_quadrilateral (y1 y2 : ℝ) : ℝ :=
  let x1 := y1^2 / 8
  let x2 := y2^2 / 8
  let yP := y1 / 2
  let yQ := y2 / 2
  1/2 * (x1 + x2) * (y1 - y2) - 1/4 * (y1 * x1 - y2 * x2)

-- The theorem we need to prove
theorem minimum_area_APQB : (∀ (m : ℝ), ∃ (y1 y2 : ℝ), y1 + y2 = 8 * m ∧ y1 * y2 = -16) →
  ∃ (m : ℝ), area_quadrilateral (let ⟨y1, y2, hy⟩ := some_such m in y1) (let ⟨y1, y2, hy⟩ := some_such m in y2) = 12 :=
sorry


end minimum_area_APQB_l560_560802


namespace vector_relationship_l560_560780

open_locale vector_space

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C P : V}

-- Given conditions
def condition1 : Prop := sorry
def condition2 (B C P : V) : Prop := vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({B} : set V) = 2 • (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({C} : set V))

-- The theorem statement based on the given conditions
theorem vector_relationship (h1 : condition1) (h2 : condition2 B C P) : 
  (vector_span ℝ ({P} : set V) -ᵥ vector_span ℝ ({A} : set V)) = 
  (-1/2 : ℝ) • (vector_span ℝ ({B} : set V) -ᵥ vector_span ℝ ({A} : set V)) + (3/2 : ℝ) • (vector_span ℝ ({C} : set V) -ᵥ vector_span ℝ ({A} : set V)) :=
sorry

end vector_relationship_l560_560780


namespace num_points_circle_line_l560_560654

-- Definitions for conditions
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 4
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 16 = 0
def distance_from_line (x y : ℝ) : ℝ := (abs (3 * x + 4 * y - 16)) / (sqrt (3^2 + 4^2))

-- Proof statement
theorem num_points_circle_line : 
  (∀ (x y : ℝ), circle x y → distance_from_line x y = 1) → 
  ∃! (p : ℝ × ℝ), circle p.1 p.2 ∧ distance_from_line p.1 p.2 = 1 :=
sorry

end num_points_circle_line_l560_560654


namespace no_ordered_pairs_no_real_solutions_l560_560356

noncomputable theory
open polynomial

-- Define the conditions for the discriminant being positive meaning no real solutions
def no_real_solutions (b c : ℕ) : Prop :=
  -27 * (c : ℤ)^2 - 4 * (b : ℤ)^3 > 0 ∧ -27 * (b : ℤ)^2 - 4 * (c : ℤ)^3 > 0

-- The main theorem stating there are no such positive integer pairs (b, c)
theorem no_ordered_pairs_no_real_solutions :
  ¬ ∃ b c : ℕ, b > 0 ∧ c > 0 ∧ no_real_solutions b c :=
sorry

end no_ordered_pairs_no_real_solutions_l560_560356


namespace calculate_fraction_product_l560_560305

noncomputable def b8 := 2 * (8^2) + 6 * (8^1) + 2 * (8^0) -- 262_8 in base 10
noncomputable def b4 := 1 * (4^1) + 3 * (4^0) -- 13_4 in base 10
noncomputable def b7 := 1 * (7^2) + 4 * (7^1) + 4 * (7^0) -- 144_7 in base 10
noncomputable def b5 := 2 * (5^1) + 4 * (5^0) -- 24_5 in base 10

theorem calculate_fraction_product : 
  ((b8 : ℕ) / (b4 : ℕ)) * ((b7 : ℕ) / (b5 : ℕ)) = 147 :=
by
  sorry

end calculate_fraction_product_l560_560305


namespace smallest_angle_of_triangle_proof_l560_560180

noncomputable def smallest_angle_of_triangle (h₁ h₂ h₃ : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) : ℝ :=
  let a := 6
  let b := 5
  let c := 4
  have h₄ : a = 6 := by sorry
  have h₅ : b = 5 := by sorry
  have h₆ : c = 4 := by sorry
  have cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  real.arccos cos_C

theorem smallest_angle_of_triangle_proof :
  smallest_angle_of_triangle 10 12 15 (by linarith) (by linarith) (by linarith) = real.arccos (9 / 40) := 
  by sorry

end smallest_angle_of_triangle_proof_l560_560180


namespace focus_of_parabola_y_squared_eq_neg_16x_l560_560859

theorem focus_of_parabola_y_squared_eq_neg_16x :
  ∀ (x y : ℝ), y^2 = -16 * x → ∃ p : ℝ, p = -4 ∧ (p, 0) = (-4, 0) :=
begin
  sorry
end

end focus_of_parabola_y_squared_eq_neg_16x_l560_560859


namespace part1_inequality_solution_l560_560891

theorem part1_inequality_solution (x m : ℝ) (h : m > 0) :
  let f := (x: ℝ) -> x^2 - x + 1 in
  ((m = 1) → (x ≠ 1 → m * f x > x + m - 1)) ∧
  ((0 < m ∧ m < 1) → ((x < 1 ∨ x > 1 / m) → m * f x > x + m - 1)) ∧
  ((m > 1) → ((x < 1 / m ∨ x > 1) → m * f x > x + m - 1)) := 
by
  sorry

end part1_inequality_solution_l560_560891


namespace odd_pos_4_digit_ints_div_5_no_digit_5_l560_560412

open Nat

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 5

def valid_odd_4_digit_ints_count : Nat :=
  let a := 8  -- First digit possibilities: {1, 2, 3, 4, 6, 7, 8, 9}
  let bc := 9  -- Second and third digit possibilities: {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let d := 4  -- Fourth digit possibilities: {1, 3, 7, 9}
  a * bc * bc * d

theorem odd_pos_4_digit_ints_div_5_no_digit_5 : valid_odd_4_digit_ints_count = 2592 := by
  sorry

end odd_pos_4_digit_ints_div_5_no_digit_5_l560_560412


namespace f_neg_3_eq_2_l560_560065

noncomputable def f : ℤ → ℤ
| x => if h : x ≥ 0 then x + 1 else f (x + 2)

theorem f_neg_3_eq_2 : f (-3) = 2 := by
  sorry

end f_neg_3_eq_2_l560_560065


namespace cos_four_plus_cos_square_eq_one_l560_560090

theorem cos_four_plus_cos_square_eq_one (α : ℝ) (h : sin α * sin α + sin α = 1) : cos α * cos α * cos α * cos α + cos α * cos α = 1 := 
by
  sorry

end cos_four_plus_cos_square_eq_one_l560_560090


namespace rectangle_dimensions_l560_560283

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 3 * w) 
  (h2 : 2 * (l + w) = 2 * l * w) : 
  w = 4 / 3 ∧ l = 4 := 
by
  sorry

end rectangle_dimensions_l560_560283


namespace calculate_a_minus_b_l560_560976

theorem calculate_a_minus_b (a b c : ℝ) (h1 : a - b - c = 3) (h2 : a - b + c = 11) : a - b = 7 :=
by 
  -- The proof would be fleshed out here.
  sorry

end calculate_a_minus_b_l560_560976


namespace merry_go_round_diameter_l560_560278

theorem merry_go_round_diameter (A : ℝ) (hA : A = 3.14) : ∃ d : ℝ, d = 2 :=
by
  let π := Real.pi
  let r := Real.sqrt (A / π)
  have hr : r = 1
  { sorry }
  let d := 2 * r
  use d
  have hd : d = 2 := by norm_num[hr]
  exact hd

end merry_go_round_diameter_l560_560278


namespace students_in_both_band_and_chorus_l560_560219

-- Define the assumptions/conditions
variables (total_students band_students chorus_students band_or_chorus_students : ℕ)
variables (h1 : total_students = 300)
variables (h2 : band_students = 120)
variables (h3 : chorus_students = 180)
variables (h4 : band_or_chorus_students = 250)

-- Define the theorem
theorem students_in_both_band_and_chorus :
  ∃ (both_band_and_chorus : ℕ), both_band_and_chorus = (band_students + chorus_students - band_or_chorus_students) :=
begin
  use band_students + chorus_students - band_or_chorus_students,
  rw [h2, h3, h4],
  norm_num,
end

end students_in_both_band_and_chorus_l560_560219


namespace blue_jelly_bean_probability_l560_560596

/-- A bag contains 5 red, 6 green, 7 yellow, and 8 blue jelly beans. A jelly bean is selected at random. 
    Prove that the probability of selecting a blue jelly bean is 4/13. -/
theorem blue_jelly_bean_probability :
  let total_jelly_beans := 5 + 6 + 7 + 8 in
  let blue_jelly_beans := 8 in
  (blue_jelly_beans / total_jelly_beans : ℚ) = 4 / 13 :=
by
  sorry

end blue_jelly_bean_probability_l560_560596


namespace triangle_inequality_proof_l560_560883

-- Define sides of triangle and the condition abc = 1
variable (a b c : ℝ)
variable (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b)
variable (abc_eq_one : a * b * c = 1)

theorem triangle_inequality_proof 
  (h_triangle_ineq : triangle_ineq)
  (h_abc_eq_one : abc_eq_one) :
  (√(b + c - a) / a + √(c + a - b) / b + √(a + b - c) / c) ≥ a + b + c := 
by
  sorry

end triangle_inequality_proof_l560_560883


namespace number_of_girls_l560_560619

-- Define the problem conditions as constants
def total_saplings : ℕ := 44
def teacher_saplings : ℕ := 6
def boy_saplings : ℕ := 4
def girl_saplings : ℕ := 2
def total_students : ℕ := 12
def students_saplings : ℕ := total_saplings - teacher_saplings

-- The proof problem statement
theorem number_of_girls (x y : ℕ) (h1 : x + y = total_students)
  (h2 : boy_saplings * x + girl_saplings * y = students_saplings) :
  y = 5 :=
by
  sorry

end number_of_girls_l560_560619


namespace size_ratio_l560_560222

variable {A B C : ℝ} -- Declaring that A, B, and C are real numbers (their sizes)
variable (h1 : A = 3 * B) -- A is three times the size of B
variable (h2 : B = (1 / 2) * C) -- B is half the size of C

theorem size_ratio (h1 : A = 3 * B) (h2 : B = (1 / 2) * C) : A / C = 1.5 :=
by
  sorry -- Proof goes here, to be completed

end size_ratio_l560_560222


namespace arithmetic_sequence_terms_l560_560432

variable (n : ℕ)
variable (sumOdd sumEven : ℕ)
variable (terms : ℕ)

theorem arithmetic_sequence_terms
  (h1 : sumOdd = 120)
  (h2 : sumEven = 110)
  (h3 : terms = 2 * n + 1)
  (h4 : sumOdd + sumEven = 230) :
  terms = 23 := 
sorry

end arithmetic_sequence_terms_l560_560432


namespace trigonometric_identity_l560_560416

noncomputable def tan_alpha : ℝ := 4

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = tan_alpha) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / Real.cos (-α) = 3 :=
by
  sorry

end trigonometric_identity_l560_560416


namespace sum_f_values_l560_560316

theorem sum_f_values (a b c d e f g : ℕ) 
  (h1: 100 * a * b = 100 * d)
  (h2: c * d * e = 100 * d)
  (h3: b * d * f = 100 * d)
  (h4: b * f = 100)
  (h5: 100 * d = 100) : 
  100 + 50 + 25 + 20 + 10 + 5 + 4 + 2 + 1 = 217 :=
by
  sorry

end sum_f_values_l560_560316


namespace compound_interest_correct_l560_560016

variables (P r n t : ℝ)

-- Definitions based on conditions
def principal := 1200
def annual_rate := 0.20
def compounding_yearly := 1
def time_year := 1

-- Theorem to prove the correct answer
theorem compound_interest_correct :
  (principal * (1 + annual_rate / compounding_yearly) ^ (compounding_yearly * time_year) - principal) = 240 := by
  sorry

end compound_interest_correct_l560_560016


namespace min_value_of_f_l560_560895

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 / (x + 1)

theorem min_value_of_f : ∃ x > 0, f x = 4 * real.sqrt 3 - 3 := 
by
  sorry

end min_value_of_f_l560_560895


namespace a_div_b_div_10_l560_560920

theorem a_div_b_div_10 (a b : ℕ)
  (h1 : a + b = 10^1000)
  (h2 : ∀ d, Nat.digits 10 a ≠ 0 d ↔ Nat.digits 10 b ≠ 0 d) :
  10 ∣ a ∧ 10 ∣ b :=
by
  sorry

end a_div_b_div_10_l560_560920


namespace prove_moles_of_C2H6_l560_560737

def moles_of_CCl4 := 4
def moles_of_Cl2 := 14
def moles_of_C2H6 := 2

theorem prove_moles_of_C2H6
  (h1 : moles_of_Cl2 = 14)
  (h2 : moles_of_CCl4 = 4)
  : moles_of_C2H6 = 2 := 
sorry

end prove_moles_of_C2H6_l560_560737


namespace total_number_of_birds_l560_560922

theorem total_number_of_birds : 
  ∀ (chickens ducks turkeys : ℕ),
  chickens = 200 →
  ducks = 2 * chickens →
  turkeys = 3 * ducks →
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys h1 h2 h3
  rw [h1, h2, h3]
  simp only
  exact sorry

end total_number_of_birds_l560_560922


namespace inverse_value_l560_560068

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value (x : ℝ) (h : g (-3) = x) : (g ∘ g⁻¹) x = x := by
  sorry

end inverse_value_l560_560068


namespace three_zeros_iff_a_greater_than_e_l560_560399

noncomputable def f (x a : ℝ) : ℝ :=
  - (1 / (x + 1)) - (a + 1) * Math.log (x + 1) + a * x + Real.exp 1 - 2

theorem three_zeros_iff_a_greater_than_e (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) ↔ a > Real.exp 1 :=
sorry

end three_zeros_iff_a_greater_than_e_l560_560399


namespace add_base_3_l560_560298

def base3_addition : Prop :=
  2 + (1 * 3^2 + 2 * 3^1 + 0 * 3^0) + 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) + 
  (1 * 3^3 + 2 * 3^1 + 0 * 3^0) = 
  (1 * 3^3) + (1 * 3^2) + (0 * 3^1) + (2 * 3^0)

theorem add_base_3 : base3_addition :=
by 
  -- We will skip the proof as per instructions
  sorry

end add_base_3_l560_560298


namespace five_digit_pairs_unique_no_six_digit_pairs_l560_560962

theorem five_digit_pairs_unique
  (a b : ℕ)
  (a_is_perfect_square : ∃ m : ℕ, a = m * m)
  (b_is_perfect_square : ∃ n : ℕ, b = n * n)
  (cond : a - b = 11111 ∧ ∀ d_a d_b : list ℕ, (d_a.length = 5 ∧ d_b.length = 5 ∧ (∀ i : ℕ, i < 5 → d_b.nth i = d_a.nth i.map (λ (d : ℕ), d + 1))) ∧ ∃ d_a d_b : list ℕ, d_a = a.digits 10 ∧ d_b = b.digits 10) :
  (a = 24336 ∧ b = 13225) :=
sorry

theorem no_six_digit_pairs (a b : ℕ)
  (a_is_perfect_square : ∃ m : ℕ, a = m * m)
  (b_is_perfect_square : ∃ n : ℕ, b = n * n)
  (cond : a - b = 111111 ∧ ∀ d_a d_b : list ℕ, (d_a.length = 6 ∧ d_b.length = 6 ∧ (∀ i : ℕ, i < 6 → d_b.nth i = d_a.nth i.map (λ (d : ℕ), d + 1))) ∧ ∃ d_a d_b : list ℕ, d_a = a.digits 10 ∧ d_b = b.digits 10) :
  false :=
sorry

end five_digit_pairs_unique_no_six_digit_pairs_l560_560962


namespace trapezoid_BC_squared_l560_560866

open Real

theorem trapezoid_BC_squared (A B C D : Point)
    (AB BC CD AC BD : Line)
    (hAB : AB ⊥ BC)
    (hBC : BC ⊥ CD)
    (hACBD : AC ⊥ BD)
    (hAB_len : line_length AB = 2)
    (hAD_len : line_length (line_through A D) = 10 * sqrt 2) :
    let BC_len := line_length BC,
    BC_len^2 = 18 := 
    sorry

end trapezoid_BC_squared_l560_560866


namespace expand_expression_l560_560727

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560727


namespace sum_of_digits_three_n_l560_560215

theorem sum_of_digits_three_n (n : ℕ) (h1 : nat.digits 10 n.sum = 100) (h2 : nat.digits 10 (44 * n).sum = 800) :
  nat.digits 10 (3 * n).sum = 300 :=
sorry

end sum_of_digits_three_n_l560_560215


namespace inequality_sqrt_sum_of_products_leq_sum_of_sqrts_l560_560031

theorem inequality_sqrt_sum_of_products_leq_sum_of_sqrts {a b c : ℝ}
  (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h_sum : a + b + c = 9) :
  sqrt (a * b + b * c + c * a) ≤ sqrt a + sqrt b + sqrt c :=
by
  sorry

end inequality_sqrt_sum_of_products_leq_sum_of_sqrts_l560_560031


namespace prove_ratio_l560_560833

variable (a b c d : ℚ)

-- Conditions
def cond1 : a / b = 5 := sorry
def cond2 : b / c = 1 / 4 := sorry
def cond3 : c / d = 7 := sorry

-- Theorem to prove the final result
theorem prove_ratio (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end prove_ratio_l560_560833


namespace probability_of_cycles_l560_560470

noncomputable def Probability_of_Cycles (n : ℕ) (f : ℕ → ℕ) (a : ℕ) :=
  let cycle := {b // ∃ c, b ≥ 1 ∧ c ≥ 1 ∧ (f^[b] 1 = a) ∧ (f^[c] a = 1)}
  classical.some (set.exists_of_finite_of_ne_empty (set.finite_of_finite_cycles (finset.univ (fin n))))

theorem probability_of_cycles (n : ℕ) (a : ℕ) (h_a : a ∈ finset.univ (fin n)) :
  ∃ (f : fin n → fin n), Probability_of_Cycles n f a = 1 / n :=
sorry

end probability_of_cycles_l560_560470


namespace externally_tangent_circles_l560_560098

theorem externally_tangent_circles (m : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x^2 + y^2 - 2 * m * x + m^2 - 1 = 0) → 
  (∃ m, |m| = 3) :=
by
  sorry

end externally_tangent_circles_l560_560098


namespace tim_youth_comparison_l560_560984

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l560_560984


namespace expand_expression_l560_560711

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560711


namespace log_3_0_216_l560_560007

theorem log_3_0_216 : 
  let log (b x : ℝ) := Real.log x / Real.log b in
  (0.216 = 3^3 / 5^3) →
  (∀ x y, log 3 (x / y) = log 3 x - log 3 y) →
  (∀ b n, log b (b^n) = n) →
  log 3 5 ≈ 1.4649 →
  log 3 0.216 ≈ -1.3947 :=
by
  intros h1 h2 h3 h4
  sorry

end log_3_0_216_l560_560007


namespace dihedral_angle_tetrahedron_eq_l560_560047

variables {A B C D E F G : Type*}
variables [RegularTetrahedron A B C D] [Midpoint E A B] [Midpoint F B C] [Midpoint G C D]

noncomputable def dihedral_angle_CFG_E : Real :=
  π - Real.arccot (Real.sqrt 2 / 2)

theorem dihedral_angle_tetrahedron_eq:
  ∀ (A B C D E F G : Type*) [RegularTetrahedron A B C D] [Midpoint E A B] [Midpoint F B C] [Midpoint G C D],
  dihedral_angle_CFG_E = π - Real.arccot (Real.sqrt 2 / 2) := sorry

end dihedral_angle_tetrahedron_eq_l560_560047


namespace cosine_alpha_plus_pi_over_6_cosine_2alpha_plus_pi_over_12_l560_560786

variable (α : ℝ)

axiom α_prop : α ∈ set.Ioo 0 (π / 3)
axiom trig_eqn : sqrt 6 * sin α + sqrt 2 * cos α = sqrt 3

theorem cosine_alpha_plus_pi_over_6 : 
  cos (α + π / 6) = sqrt 10 / 4 :=
by
  have h : α ∈ set.Ioo 0 (π / 3) := α_prop
  have heqn : sqrt 6 * sin α + sqrt 2 * cos α = sqrt 3 := trig_eqn
  sorry

theorem cosine_2alpha_plus_pi_over_12 :
  cos (2 * α + π / 12) = (sqrt 30 + sqrt 2) / 8 :=
by
  have h : α ∈ set.Ioo 0 (π / 3) := α_prop
  have heqn : sqrt 6 * sin α + sqrt 2 * cos α = sqrt 3 := trig_eqn
  sorry

end cosine_alpha_plus_pi_over_6_cosine_2alpha_plus_pi_over_12_l560_560786


namespace min_value_proven_l560_560751

open Real

noncomputable def min_value (x y : ℝ) (h1 : log x + log y = 1) : Prop :=
  2 * x + 5 * y ≥ 20 ∧ (2 * x + 5 * y = 20 ↔ 2 * x = 5 * y ∧ x * y = 10)

theorem min_value_proven (x y : ℝ) (h1 : log x + log y = 1) :
  min_value x y h1 :=
sorry

end min_value_proven_l560_560751


namespace bottle_caps_difference_l560_560322

variable (found_bottle_caps : ℕ) (thrown_bottle_caps : ℕ)

theorem bottle_caps_difference (h1 : found_bottle_caps = 36) (h2 : thrown_bottle_caps = 35) : 
  found_bottle_caps - thrown_bottle_caps = 1 := 
by 
  rw [h1, h2] 
  exact rfl

end bottle_caps_difference_l560_560322


namespace mars_colony_cost_l560_560947

theorem mars_colony_cost :
  let total_cost := 45000000000
  let number_of_people := 300000000
  total_cost / number_of_people = 150 := 
by sorry

end mars_colony_cost_l560_560947


namespace sandwich_cost_l560_560570

theorem sandwich_cost (total_cost soda_cost sandwich_count soda_count : ℝ) :
  total_cost = 8.38 → soda_cost = 0.87 → sandwich_count = 2 → soda_count = 4 → 
  (∀ S, sandwich_count * S + soda_count * soda_cost = total_cost → S = 2.45) :=
by
  intros h_total h_soda h_sandwich_count h_soda_count S h_eqn
  sorry

end sandwich_cost_l560_560570


namespace increasing_interval_l560_560956

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * (Real.logb (1/2) x) ^ 2 - 2 * Real.logb (1/2) x + 1

theorem increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ {y}, y ≥ x → function_y y ≥ function_y x) ↔ x ∈ Set.Ici (Real.sqrt 2 / 2) :=
by
  sorry

end increasing_interval_l560_560956


namespace inequality_solution_set_l560_560213

theorem inequality_solution_set (x : ℝ) : (|x - 1| + 2 * x > 4) ↔ (x > 3) := 
sorry

end inequality_solution_set_l560_560213


namespace integrate_differential_eq_l560_560446

theorem integrate_differential_eq {x y C : ℝ} {y' : ℝ → ℝ → ℝ} (h : ∀ x y, (4 * y - 3 * x - 5) * y' x y + 7 * x - 3 * y + 2 = 0) : 
    ∃ C : ℝ, ∀ x y : ℝ, 2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C :=
by
  sorry

end integrate_differential_eq_l560_560446


namespace sum_of_digits_of_9N_is_9_l560_560001

-- Define what it means for a natural number N to have strictly increasing digits.
noncomputable def strictly_increasing_digits (N : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → (N.digits i < N.digits j)

-- Formal statement of the problem
theorem sum_of_digits_of_9N_is_9 (N : ℕ) (h : strictly_increasing_digits N) : ∑ d in (9 * N).digits, d = 9 :=
by sorry

end sum_of_digits_of_9N_is_9_l560_560001


namespace manhattan_distance_part1_manhattan_distance_part2_manhattan_distance_part3_l560_560856

-- Part (1) Statement
theorem manhattan_distance_part1 (x : ℝ) (y : ℝ) (h : y = 1 - x) (hp : abs x + abs y ≤ 1) : 0 ≤ x ∧ x ≤ 1 := sorry

-- Part (2) Statement
theorem manhattan_distance_part2 (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hP : y1 = 2 * x1 - 2) (hQ : y2 = x2^2) : 
  (|x1 - x2| + |y1 - y2|) ≥ (1 / 2) := sorry

-- Part (3) Statement
theorem manhattan_distance_part3 (a b : ℝ) : 
  let f (x : ℝ) := a - x + b - x^2 in
  (∀ x ∈ set.Icc (-2 : ℝ) 2, |f x| ≤ 25 / 8) → 
  (a = 0 ∧ b = 23 / 8) := sorry

end manhattan_distance_part1_manhattan_distance_part2_manhattan_distance_part3_l560_560856


namespace coefficient_x2_in_product_l560_560994

theorem coefficient_x2_in_product :
  let p1 := 3 * X ^ 3 - 4 * X ^ 2 - 9 * X + 2 
  let p2 := 2 * X ^ 2 - 8 * X + 3 
  coefficient (p1 * p2) 2 = 68 :=
by
  sorry

end coefficient_x2_in_product_l560_560994


namespace largest_prime_factor_of_4752_l560_560996

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l560_560996


namespace contradiction_unique_solution_l560_560163

theorem contradiction_unique_solution :
  (¬ (∃! x, equation_solution x)) ↔ (¬ ∃! x, ∃ y, x ≠ y ∨ ∀ z, ¬ equation_solution z) :=
by
  sorry

end contradiction_unique_solution_l560_560163


namespace fruit_total_l560_560551

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l560_560551


namespace angle_between_a_b_l560_560902

theorem angle_between_a_b (a b c d : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1)
  (h : a + 2 • b + sqrt 2 • c + d = 0) : real.angle_between a b = 120 :=
  sorry

end angle_between_a_b_l560_560902


namespace inequality_always_true_l560_560753

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l560_560753


namespace kilo_apples_second_scenario_l560_560830

theorem kilo_apples_second_scenario (O : ℝ) (x : ℝ) :
  6 * O + 5 * 29 = 419 →
  5 * O + x * 29 = 488 →
  x ≈ 9 := 
by
  sorry

end kilo_apples_second_scenario_l560_560830


namespace youngest_sibling_age_l560_560221

-- Definitions of the conditions
def num_siblings : ℕ := 6
def differences : List ℝ := [4, 5, 7, 9, 11]
def average_age : ℝ := 23.5

-- Main theorem statement
theorem youngest_sibling_age :
  ∃ (Y : ℝ), 
  let ages := Y :: (differences.map (λ d, Y + d)),
  (ages.sum / num_siblings) = average_age ∧ Y = 17.5 :=
by
  sorry

end youngest_sibling_age_l560_560221


namespace new_average_weight_l560_560518

theorem new_average_weight 
  (num_students_before : ℕ) 
  (avg_weight_before : ℕ) 
  (new_student_weight : ℕ) 
  (num_students_after : ℕ) 
  (total_weight_before : ℕ) 
  (total_weight_after : ℕ) 
  (new_avg_weight : ℚ) 
  (h1 : num_students_before = 29)
  (h2 : avg_weight_before = 28)
  (h3 : new_student_weight = 13)
  (h4 : num_students_after = num_students_before + 1)
  (h5 : total_weight_before = num_students_before * avg_weight_before)
  (h6 : total_weight_after = total_weight_before + new_student_weight)
  (h7 : new_avg_weight = total_weight_after / num_students_after) :
  new_avg_weight = 27.5 := 
sorry

end new_average_weight_l560_560518


namespace degree_g_of_degree_f_and_h_l560_560175

noncomputable def degree (p : ℕ) := p -- definition to represent degree of polynomials

theorem degree_g_of_degree_f_and_h (f g : ℕ → ℕ) (h : ℕ → ℕ) 
  (deg_h : ℕ) (deg_f : ℕ) (deg_10 : deg_h = 10) (deg_3 : deg_f = 3) 
  (h_eq : ∀ x, degree (h x) = degree (f (g x)) + degree x ^ 5) :
  degree (g 0) = 4 :=
by
  sorry

end degree_g_of_degree_f_and_h_l560_560175


namespace expand_expression_l560_560664

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560664


namespace complement_inter_section_l560_560903

-- Define the sets M and N
def M : Set ℝ := { x | x^2 - 2*x - 3 >= 0 }
def N : Set ℝ := { x | abs (x - 2) <= 1 }

-- Define the complement of M in ℝ
def compl_M : Set ℝ := { x | -1 < x ∧ x < 3 }

-- Define the expected result set
def expected_set : Set ℝ := { x | 1 <= x ∧ x < 3 }

-- State the theorem to prove
theorem complement_inter_section : compl_M ∩ N = expected_set := by
  sorry

end complement_inter_section_l560_560903


namespace other_solution_eq_three_l560_560382

theorem other_solution_eq_three (a : ℝ) (h : (-1)^2 - 2*(-1) + a = 0) : 
  is_solution (x : ℝ) : (x^2 - 2*x + a = 0) → (x = 3) :=
begin
  -- The proof would go here
  sorry,
end

end other_solution_eq_three_l560_560382


namespace simplify_and_evaluate_expression_l560_560171

theorem simplify_and_evaluate_expression : 
  ∀ a : ℚ, a = -1/2 → (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := 
by
  intro a ha
  simp only [ha]
  sorry

end simplify_and_evaluate_expression_l560_560171


namespace area_of_triangle_BPG_relative_to_ABC_l560_560867

-- Given definitions and conditions
variables {A B C F D G P : Type}
variables [IsTriangle A B C]
variables [IsMedian A F B] [IsMedian A D C]
variables [Centroid G A B C]
variables [MedianIntersect G F D]
variables [Midpoint P B D]

-- Theorem statement
theorem area_of_triangle_BPG_relative_to_ABC :
  let area_ABC := area (triangle A B C) in
  let area_BPG := area (triangle B P G) in
  area_BPG = (1 / 6) * area_ABC :=
sorry

end area_of_triangle_BPG_relative_to_ABC_l560_560867


namespace power_function_through_point_l560_560072

theorem power_function_through_point {a : ℝ} (h : (2 : ℝ) ^ a = real.sqrt 2) : a = 1 / 2 :=
sorry

end power_function_through_point_l560_560072


namespace min_area_sum_3_l560_560126

theorem min_area_sum_3
  (O : Type) [inner_product_space ℝ O]
  (A B C : O)
  (hAB : dist A B = 2)
  (r1 r2 : ℝ)
  (hTangentA : dist A (A + r1 • (B - A)) = r1)
  (hTangentB : dist B (B - r2 • (B - A)) = r2)
  (hExternalTangent : dist (A + r1 • (B - A)) (B - r2 • (B - A)) = 2 - r1 - r2) :
  let S := π * (r1^2 + r2^2)
  in ∀ r1 r2, r1 + r2 = 1 → (S = π/2) → (nat.gcd 1 2 = 1) → 3 = 3 :=
sorry

end min_area_sum_3_l560_560126


namespace compute_floor_expression_l560_560644

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end compute_floor_expression_l560_560644


namespace sum_of_coordinates_l560_560841

noncomputable def f : ℝ → ℝ := sorry
noncomputable def k (x : ℝ) := (f x) ^ 3

theorem sum_of_coordinates (h1 : f 4 = 8) : 4 + k 4 = 516 :=
by
  have h2 : k 4 = (f 4) ^ 3 := by rfl
  rw [h2, h1]
  norm_num
  sorry

end sum_of_coordinates_l560_560841


namespace max_prob_of_one_six_l560_560743

theorem max_prob_of_one_six (n : ℕ) : 
  let V (x : ℕ) := (x * 5^(x - 1) : ℚ) / 6^x in
  V n ≤ V 5 ∨ V n ≤ V 6 :=
by
  sorry

end max_prob_of_one_six_l560_560743


namespace initial_deposit_l560_560451

theorem initial_deposit :
  ∀ (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ),
    r = 0.05 → n = 1 → t = 2 → P * (1 + r / n) ^ (n * t) = 6615 → P = 6000 :=
by
  intros P r n t h_r h_n h_t h_eq
  rw [h_r, h_n, h_t] at h_eq
  norm_num at h_eq
  sorry

end initial_deposit_l560_560451


namespace roots_conjugate_implies_pq_zero_l560_560893

theorem roots_conjugate_implies_pq_zero (p q : ℝ) :
  ∀ z : ℂ, (z^2 + (6 + p * complex.I) * z + (10 + q * complex.I)) = 0 ∧ ∃ x y : ℝ, z = x + y * complex.I ∧ z = x - y * complex.I → (p = 0 ∧ q = 0) :=
by {
  sorry,
}

end roots_conjugate_implies_pq_zero_l560_560893


namespace speed_of_bicyclist_during_remainder_l560_560487

-- Define the given conditions
def distance_total : ℝ := 400
def distance_first : ℝ := 100
def speed_first : ℝ := 20
def avg_speed_total : ℝ := 16

-- Define the unknown speed for the remainder of the trip
def distance_remaining : ℝ := distance_total - distance_first
def speed_remaining : ℝ := 15

-- State the theorem
theorem speed_of_bicyclist_during_remainder :
  let time_total := distance_total / avg_speed_total,
      time_first := distance_first / speed_first,
      time_remaining := time_total - time_first
  in distance_remaining / time_remaining = speed_remaining := by
  sorry

end speed_of_bicyclist_during_remainder_l560_560487


namespace find_a_l560_560395

noncomputable def f (x a : ℝ) : ℝ := x + a / (2 * x)

noncomputable def isTangentLineAt (x a : ℝ) : ℝ × ℝ := 
  let slope := 1 - a / (2 * x ^ 2)
  (x, f x a, slope)

noncomputable def satisfiesConditions (x1 x2 a : ℝ) : Prop :=
  0 < x1 ∧ x1 < x2 ∧
  f x1 a = 0 ∧ f x2 a = 0 ∧
  (∃! (i : ℤ), ↑x1 < i ∧ i < ↑x2)

theorem find_a (a x1 x2 : ℝ) (h : satisfiesConditions x1 x2 a) : 
  a ∈ set.Ico (-8 / 3) (-2) :=
sorry

end find_a_l560_560395


namespace find_C_coordinates_l560_560615

def coordinates_A : ℝ × ℝ := (3, 3)
def coordinates_B : ℝ × ℝ := (15, 9)
def AB_distance : ℝ := real.sqrt ((coordinates_B.1 - coordinates_A.1)^2 + (coordinates_B.2 - coordinates_A.2)^2)
def BC_distance : ℝ := 1 / 2 * AB_distance
def coordinates_C : ℝ × ℝ := (coordinates_B.1 + 1 / 2 * (coordinates_B.1 - coordinates_A.1), coordinates_B.2 + 1 / 2 * (coordinates_B.2 - coordinates_A.2))

theorem find_C_coordinates 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (hA : A = coordinates_A) 
  (hB : B = coordinates_B)
  (hBC : real.sqrt ((coordinates_C.1 - coordinates_B.1)^2 + (coordinates_C.2 - coordinates_B.2)^2) = BC_distance) : 
  coordinates_C = (21, 12) := by 
  sorry

end find_C_coordinates_l560_560615


namespace parity_decreasing_on_interval_l560_560147

noncomputable def f (x : Real) : Real := Real.log (Real.exp 1 + x) + Real.log (Real.exp 1 - x)

theorem parity_decreasing_on_interval :
  (∀ x : Real, f (-x) = f x) ∧ (∀ x : Real, 0 < x ∧ x < Real.exp 1 → f' x < 0) :=
by
  sorry

end parity_decreasing_on_interval_l560_560147


namespace no_unfenced_area_l560_560480

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end no_unfenced_area_l560_560480


namespace distinct_values_count_l560_560008

theorem distinct_values_count :
  let e1 := 3 ^ (3 ^ (3 ^ 3))
  let e2 := 3 ^ ((3 ^ 3) ^ 3)
  let e3 := ((3 ^ 3) ^ 3) ^ 3
  let e4 := (3 ^ (3 ^ 3)) ^ 3
  let e5 := (3 ^ 3) ^ (3 ^ 3)
  (Set.card ({e1, e2, e3, e4, e5} : Set ℕ)) = 3 :=
begin
  -- Here you would include the detailed proof steps.
  sorry
end

end distinct_values_count_l560_560008


namespace solution_constant_term_of_edgars_polynomial_l560_560085

noncomputable def constant_term_of_edgars_polynomial : ℕ := 
let p := (k : ℝ) → (z : ℝ) → z^3 + a_2*z^2 + a_1*z + k in
let q := (k : ℝ) → (z : ℝ) → z^3 + b_2*z^2 + b_1*z + k in
let product_polynomial := (z : ℝ) → z^6 + 2*z^5 + 7*z^4 + 8*z^3 + 9*z^2 + 6*z + 9 in
let k := 3 in
k

theorem solution_constant_term_of_edgars_polynomial :
  ∀ (k : ℝ) (a_2 a_1 b_2 b_1 : ℝ),
  (∀ (z : ℝ), (z^3 + a_2*z^2 + a_1*z + k) * (z^3 + b_2*z^2 + b_1*z + k) = z^6 + 2*z^5 + 7*z^4 + 8*z^3 + 9*z^2 + 6*z + 9) →
  (0 < k) →
  k = 3 :=
begin
  sorry,
end

end solution_constant_term_of_edgars_polynomial_l560_560085


namespace situation1_correct_situation2_correct_situation3_correct_l560_560218

noncomputable def situation1 : Nat :=
  let choices_for_A := 4
  let remaining_perm := Nat.factorial 6
  choices_for_A * remaining_perm

theorem situation1_correct : situation1 = 2880 := by
  sorry

noncomputable def situation2 : Nat :=
  let permutations_A_B := Nat.factorial 2
  let remaining_perm := Nat.factorial 5
  permutations_A_B * remaining_perm

theorem situation2_correct : situation2 = 240 := by
  sorry

noncomputable def situation3 : Nat :=
  let perm_boys := Nat.factorial 3
  let perm_girls := Nat.factorial 4
  perm_boys * perm_girls

theorem situation3_correct : situation3 = 144 := by
  sorry

end situation1_correct_situation2_correct_situation3_correct_l560_560218


namespace expand_expression_l560_560671

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560671


namespace find_x0_l560_560407

-- Define the given conditions
variable (p x_0 : ℝ) (P : ℝ × ℝ) (O : ℝ × ℝ)
variable (h_parabola : x_0^2 = 2 * p * 1)
variable (h_p_gt_zero : p > 0)
variable (h_point_P : P = (x_0, 1))
variable (h_origin : O = (0, 0))
variable (h_distance_condition : dist (x_0, 1) (0, 0) = dist (x_0, 1) (0, -p / 2))

-- The theorem we aim to prove
theorem find_x0 : x_0 = 2 * Real.sqrt 2 :=
  sorry

end find_x0_l560_560407


namespace total_people_on_bus_l560_560158

def students_left := 42
def students_right := 38
def students_back := 5
def students_aisle := 15
def teachers := 2
def bus_driver := 1

theorem total_people_on_bus : students_left + students_right + students_back + students_aisle + teachers + bus_driver = 103 :=
by
  sorry

end total_people_on_bus_l560_560158


namespace polygon_interior_angles_sum_l560_560188

theorem polygon_interior_angles_sum (n : ℕ) (hn : 180 * (n - 2) = 1980) : 180 * (n + 4 - 2) = 2700 :=
by
  sorry

end polygon_interior_angles_sum_l560_560188


namespace min_value_expression_l560_560766

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := 5 + (n - 1) * d
noncomputable def S_n (n : ℕ) (d : ℝ) : ℝ := (n * (a_n n d) + n * (a_n n d)) / 2

theorem min_value_expression (d : ℝ) (h : d > 0) :
  let a2 := a_n 2 d
  let a5 := a_n 5 d - 1
  let a10 := a_n 10 d
  let S := S_n
  (a5 * a5 = a2 * a10) →
  ∃ n : ℕ, (n > 0) → 
  2 * S n + n + 32 / (a_n n d + 1) = 20 / 3 :=
sorry

end min_value_expression_l560_560766


namespace hyperbola_eccentricity_proof_l560_560474

open Real

noncomputable def hyperbola_eccentricity (a b c : ℝ) (λ μ : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : λ * μ = 1 / 16) 
  (h4 : (c /(c + b / 2)) * 2 = 1) 
  (h5 : (c / (c - b / 2)) * 2 = b / c) : 
  Real :=
2

theorem hyperbola_eccentricity_proof (a b c : ℝ) (λ μ : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : λ * μ = 1 / 16) 
  (h4 : (c /(c + b / 2)) * 2 = 1) 
  (h5 : (c / (c - b / 2)) * 2 = b / c) :
  hyperbola_eccentricity a b c λ μ h1 h2 h3 h4 h5 = 2 := 
sorry

end hyperbola_eccentricity_proof_l560_560474


namespace distinct_real_roots_of_quadratic_l560_560504

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end distinct_real_roots_of_quadratic_l560_560504


namespace find_distance_between_sides_l560_560347

-- Define the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 247

-- Define the distance h between parallel sides
def distance_between_sides (h : ℝ) : Prop :=
  area_trapezium = (1 / 2) * (length_side1 + length_side2) * h

-- Define the theorem we want to prove
theorem find_distance_between_sides : ∃ h : ℝ, distance_between_sides h ∧ h = 13 := by
  sorry

end find_distance_between_sides_l560_560347


namespace round_6703_4999_l560_560505

-- Conditions
def decimal_part (x : ℝ) : ℝ := x - real.floor x

-- Problem statement
theorem round_6703_4999 :
  decimal_part 6703.4999 < 0.5 → Real.round 6703.4999 = 6703 :=
by
  intro h
  sorry

end round_6703_4999_l560_560505


namespace expand_expression_l560_560669

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560669


namespace reducible_fraction_probability_with_12_is_four_sevenths_l560_560806

open Set

-- Define the set A
def A : Set ℕ := {2, 4, 5, 6, 8, 11, 12, 17}

-- Define the condition that one of the chosen numbers is 12
def chosen_number := 12

-- Define the definition of a fraction being reducible
def is_reducible_fraction (n d : ℕ) : Prop :=
  n.gcd d ≠ 1

-- Calculate the probability of forming a reducible fraction when one number is fixed as 12
def reducible_fraction_probability : ℚ :=
  let total_fractions := 7
  let reducible_fractions := 4
  reducible_fractions / total_fractions

theorem reducible_fraction_probability_with_12_is_four_sevenths :
  reducible_fraction_probability = 4 / 7 :=
by
  sorry

end reducible_fraction_probability_with_12_is_four_sevenths_l560_560806


namespace expand_polynomial_eq_l560_560705

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l560_560705


namespace ms_lee_class_difference_l560_560846

noncomputable def boys_and_girls_difference (ratio_b : ℕ) (ratio_g : ℕ) (total_students : ℕ) : ℕ :=
  let x := total_students / (ratio_b + ratio_g)
  let boys := ratio_b * x
  let girls := ratio_g * x
  girls - boys

theorem ms_lee_class_difference :
  boys_and_girls_difference 3 4 42 = 6 :=
by
  sorry

end ms_lee_class_difference_l560_560846


namespace smallest_four_digit_multiple_of_3_4_5_l560_560236

-- Conditions and definitions
def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- The main theorem
theorem smallest_four_digit_multiple_of_3_4_5 :
  ∃ n : ℕ, is_four_digit n ∧ is_multiple n 3 ∧ is_multiple n 4 ∧ is_multiple n 5 ∧ 
           ∀ m : ℕ, is_four_digit m ∧ is_multiple m 3 ∧ is_multiple m 4 ∧ is_multiple m 5 → n ≤ m :=
begin
  use 1020,
  split,
  { -- proving 1020 is a four-digit number
    unfold is_four_digit,
    exact ⟨by norm_num1, by norm_num1⟩,
  },
  split,
  { -- proving 1020 is a multiple of 3
    unfold is_multiple,
    use 340,
    norm_num1, 
  },
  split,
  { -- proving 1020 is a multiple of 4
    unfold is_multiple,
    use 255,
    norm_num1,
  },
  split,
  { -- proving 1020 is a multiple of 5
    unfold is_multiple,
    use 204,
    norm_num1,
  },
  { -- proving 1020 is the smallest four-digit number with 3, 4, and 5 as factors
    intros m Hm,
    cases Hm with Hm_four_digit Hm_p_factors,
    cases Hm_p_factors with Hm_3 Hm_p_factors,
    cases Hm_p_factors with Hm_4 Hm_5,
    have h_lcm: (60 ∣ m), { 
      apply exists.intro ((m / 60)), 
      exact (nat.eq_of_dvd_of_div_eq_one (dvd_trans (dvd_mul_right 12 5) (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd 
        (dvd_trans (dvd_mul_right 3 4) (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd Hm_3))))))), 
    },
    have h: 1020 ≤ m := by sorry, -- Here, we assume the proof for brevity
    exact h,
  }
end

end smallest_four_digit_multiple_of_3_4_5_l560_560236


namespace bailing_rate_bailing_problem_l560_560246

theorem bailing_rate (distance : ℝ) (rate_in : ℝ) (sink_limit : ℝ) (speed : ℝ) : ℝ :=
  let time_to_shore := distance / speed * 60 -- convert hours to minutes
  let total_intake := rate_in * time_to_shore
  let excess_water := total_intake - sink_limit
  excess_water / time_to_shore

theorem bailing_problem : bailing_rate 2 12 40 3 = 11 := by
  sorry

end bailing_rate_bailing_problem_l560_560246


namespace vector_ap_eq_l560_560777

variables {A B C P : Type} [affine_space P]
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C]

axiom vector_eq : ∀ (u v : A), (u = v) ↔ (u - v = 0)
axiom bc_eq_2cp (a b c p : A) : 
  (c - b = 2 • (p - c)) ↔ (c - b = 2 • (p - c))

theorem vector_ap_eq (a b c p : A) (h : (c - b = 2 • (p - c))) :
  (p - a) = (-1 / 2 • (b - a)) + (3 /2 • (c - a)) :=
sorry

end vector_ap_eq_l560_560777


namespace sphere_radius_l560_560286

-- Define a structure for the segment and properties
structure Segment (A B : Type) :=
(length : real)
(split_ratio : real)
(angle_with_plane : real)

-- Define the specific segment AB and its properties
def AB_segment : Segment Point Point := 
{ length := 8,
  split_ratio := 1 / 3,
  angle_with_plane := 30 }

-- Define the sphere radius proof statement
theorem sphere_radius 
  (A B : Point) 
  (s : Segment A B)
  (h1 : s.length = 8) 
  (h2 : s.split_ratio = 1 / 3)
  (h3 : s.angle_with_plane = 30) :
  ∃ R : real, R = 2 * sqrt 7 := 
sorry

end sphere_radius_l560_560286


namespace ellipse_equation_ellipse_standard_equation_l560_560376

theorem ellipse_equation
    (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (c : ℝ) (h3 : c = 2)
    (e : ℝ) (h4 : e = c / a) (h5 : e = 1 / 2)
    : (a = 4) ∧ (b = 2 * Real.sqrt 3) :=
by
  sorry

theorem ellipse_standard_equation
    (a b : ℝ) (h : a = 4) (h' : b = 2 * Real.sqrt 3)
    : \(\frac{x^2}{16} + \frac{y^2}{12} = 1\) :=
by
  sorry

end ellipse_equation_ellipse_standard_equation_l560_560376


namespace polynomial_has_at_most_one_real_root_l560_560165

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), x^i / Nat.factorial i

theorem polynomial_has_at_most_one_real_root (n : ℕ) :
  n > 0 → ∀ x1 x2 : ℝ, f n x1 = 0 → f n x2 = 0 → x1 = x2 := 
by
  intros
  sorry

end polynomial_has_at_most_one_real_root_l560_560165


namespace sum_of_solutions_l560_560351

theorem sum_of_solutions :
  (∑ x in {x : ℝ | |x - 3| = 3 * |x + 3|}.toFinset, x) = -15 / 2 :=
by
  sorry

end sum_of_solutions_l560_560351


namespace right_triangle_acute_angles_45_degrees_l560_560181

theorem right_triangle_acute_angles_45_degrees
  (A B C K M H : Type)
  [Triangle A B C]
  [RightAngle C (Angle A B)]
  [AngleBisector A M B]
  [Altitude C H]
  (AM_CH_intersect_K : AM ∩ CH = K)
  (ratio_AK_KM : ∀ {x y : ℝ}, AK x / KM x = 1 + Real.sqrt 2 ) :
  (Angle BAC = 45) ∧ (Angle ABC = 45) :=
sorry

end right_triangle_acute_angles_45_degrees_l560_560181


namespace max_area_ABC_l560_560081

noncomputable def max_area_of_triangle (x₁ x₂ y₁ y₂ y₀ : ℝ) : ℝ := 
  1 / 3 * real.sqrt((9 + y₀^2) * (12 - y₀^2)) * real.sqrt(9 + y₀^2)

theorem max_area_ABC 
  (x₁ x₂ y₁ y₂ y₀ : ℝ)
  (h1 : y₁^2 = 6 * x₁)
  (h2: y₂^2 = 6 * x₂)
  (h3 : x₁ ≠ x₂)
  (h4 : x₁ + x₂ = 4)
  (y0_def : y₀ = (y₁ + y₂) / 2) :
  ∃ y₀, max_area_of_triangle x₁ x₂ y₁ y₂ y₀ = 14 * real.sqrt(7) / 3 :=
by
  sorry

end max_area_ABC_l560_560081


namespace sum_of_selected_numbers_l560_560238

def sum_of_greater_than_or_equal_04 (lst : List ℚ) : ℚ :=
  (lst.filter (λ x, x ≥ 0.4)).sum

theorem sum_of_selected_numbers :
  sum_of_greater_than_or_equal_04 [0.8, 1/2, 0.9, 1/3] = 2.2 := 
  by
    sorry

end sum_of_selected_numbers_l560_560238


namespace problem_1_problem_2_l560_560906

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
def vector_b : ℝ × ℝ := (-1, 1)
def vector_c : ℝ × ℝ := (1, 1)

def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem problem_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) :
  parallel (vector_a x + vector_b) vector_c → x = 5 * π / 6 :=
begin
  -- proof omitted
  sorry
end

theorem problem_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) :
  dot_product (vector_a x) vector_b = 1/2 →
  Real.sin (x + π / 6) = Real.sqrt 15 / 4 :=
begin
  -- proof omitted
  sorry
end

end problem_1_problem_2_l560_560906


namespace problem1_problem2_l560_560364

noncomputable def polynomial := (1 + x)^6 * (1 - 2 * x)^5

theorem problem1 (a_ : ℝ → ℝ) (a_0 a_1 : ℝ) :
  (∃ (a : ℕ → ℝ), polynomial = ∑ i in (range 12), a i * x^i) →
  a 0 = 1 →
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = -2^6 →
  a_1 + a_2 + ... + a_11 = -65 :=
begin
  intro H1,
  intro H2,
  intro H3,
  sorry
end

theorem problem2 (a_ : ℝ → ℝ) (a_0 a_1 : ℝ) :
  (∃ (a : ℕ → ℝ), polynomial = ∑ i in (range 12), a i * x^i) →
  a 0 = 1 →
  (a 0 + a 2 + a 4 + a 6 + a 8 + a 10) = -32 :=
begin
  intro H4,
  intro H5,
  sorry
end

end problem1_problem2_l560_560364


namespace no_reappearance_of_141_l560_560486

theorem no_reappearance_of_141 :
  ∀ (n : ℕ), n = 141 →
  ∀ (f : ℕ → ℕ → ℕ), (∀ m, f m (digits_product m) = m + digits_product m ∨ f m (digits_product m) = m - digits_product m) →
  ∃ x, x ≠ 141 :=
by sorry

-- Helper Function to calculate the product of the digits of a number
def digits_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

 -- Proof to be filled in

end no_reappearance_of_141_l560_560486


namespace difference_is_cube_sum_1996_impossible_l560_560630

theorem difference_is_cube (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  M - m = (n - 1)^3 := 
by {
  sorry
}

theorem sum_1996_impossible (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  ¬(1996 ∈ {x | m ≤ x ∧ x ≤ M}) := 
by {
  sorry
}

end difference_is_cube_sum_1996_impossible_l560_560630


namespace intersection_is_single_point_l560_560076

def A : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x - y = 0 }
def B : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 3 * x + y = 0 }
def intersection : set (ℝ × ℝ) := { p | p ∈ A ∧ p ∈ B }

theorem intersection_is_single_point : intersection = {(0, 0)} :=
by
  sorry

end intersection_is_single_point_l560_560076


namespace diagonal_sum_of_symmetric_table_l560_560618

theorem diagonal_sum_of_symmetric_table :
  ∀ (n : ℕ), n = 1861 →
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (∃ f : ℕ → ℕ → ℕ, 
    ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → 1 ≤ f i j ∧ f i j ≤ n) ∧ 
    (∀ i j, f i j = f j i) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → (∃! j, 1 ≤ j ∧ j ≤ n ∧ f k j = i)))) →
  (∑ k in finset.range n.succ, k) = 1732591 :=
by
  intros n hn H
  have : n = 1861, by rw hn
  sorry

end diagonal_sum_of_symmetric_table_l560_560618


namespace math_problem_correct_answer_is_9_l560_560129

open Finset

noncomputable def permutations_not_starting_with_one : Finset (Permutation (Fin 6)) :=
  univ.filter (λ σ => σ 0 ≠ 0)

def favorable_permutations : Finset (Permutation (Fin 6)) :=
  permutations_not_starting_with_one.filter (λ σ => σ 2 = 2)

def probability_favorable : ℚ :=
  (favorable_permutations.card : ℚ) / (permutations_not_starting_with_one.card : ℚ)

def correct_answer : ℕ :=
  let a := (probability_favorable.numerator : ℕ)
  let b := (probability_favorable.denominator : ℕ)
  a + b

theorem math_problem_correct_answer_is_9 : correct_answer = 9 :=
by
  -- This placeholder skips the proof.
  sorry

end math_problem_correct_answer_is_9_l560_560129


namespace expand_expression_l560_560729

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560729


namespace vector_relationship_proof_l560_560774

variables {P A B C : Type} [inner_product_space ℝ P]
variables {V : P → P → ℝ} (AB AC AP BC CP : V)

-- Given conditions as hypotheses
hypothesis h1 : BC = 2 * CP

-- Known vector relationships
hypothesis h2 : BC = AC - AB
hypothesis h3 : CP = AP - AC

-- Prove the given vector equation
theorem vector_relationship_proof :
  AP = - (1/2) * AB + (3/2) * AC :=
by {
  -- Implementation of the proof steps here (skipped)
  sorry
}

end vector_relationship_proof_l560_560774


namespace amy_local_calls_l560_560210

-- Define the conditions as hypotheses
variable (L I : ℕ)
variable (h1 : L = (5 / 2 : ℚ) * I)
variable (h2 : L = (5 / 3 : ℚ) * (I + 3))

-- Statement of the theorem
theorem amy_local_calls : L = 15 := by
  sorry

end amy_local_calls_l560_560210


namespace ice_cream_flavors_l560_560821

-- Definition of the problem setup
def number_of_flavors : ℕ :=
  let scoops := 5
  let dividers := 2
  let total_objects := scoops + dividers
  Nat.choose total_objects dividers

-- Statement of the theorem
theorem ice_cream_flavors : number_of_flavors = 21 := by
  -- The proof of the theorem will use combinatorics to show the result.
  sorry

end ice_cream_flavors_l560_560821


namespace range_of_m_l560_560051

theorem range_of_m (m : ℝ) (x : ℝ) 
  (h1 : ∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3)
  (h2 : ¬ (∀ x : ℝ, x > 2 * m^2 - 3 → -1 < x ∧ x < 4))
  :
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l560_560051


namespace total_swimming_hours_over_4_weeks_l560_560927

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l560_560927


namespace eccentricity_of_ellipse_l560_560044

/-- Given an ellipse C: x^2/a^2 + y^2/b^2 = 1 (a > b > 0) with left and right foci F₁ and F₂,
respectively, and given that there exists a point M in the first quadrant of ellipse C such that
|MF₁| = |F₁F₂|, and the line F₁M intersects the y-axis at point A, and F₂A bisects the angle
∠MF₂F₁, proves that the eccentricity e of the ellipse C is (√5 - 1)/2. -/
theorem eccentricity_of_ellipse 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (C : set (ℝ × ℝ)) (hC : C = { p | ∃ x y, p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1 })
  (F₁ F₂ M A : ℝ × ℝ)
  (hfoci1 : F₁ = (-c, 0)) (hfoci2 : F₂ = (c, 0))
  (hM : M ∈ C) (hF1M : |M.1 + c| = 2 * c)
  (hAint : ∃ t, A = (0, t) ∧ ∃ u, M = (u, t)) 
  (hangleBisect : ∃ θ, θ = |∠MF₂F₁| / 2 ∧ F₂A bisects θ) 
  : eccentricity C = (Real.sqrt 5 - 1) / 2 := sorry

end eccentricity_of_ellipse_l560_560044


namespace number_of_triples_is_odd_l560_560022

theorem number_of_triples_is_odd
  (p : ℕ) 
  (h_prime : Nat.prime p)
  (h_p_gt : p > 3) 
  (h_modulo: p % 8 = 1 ∨ p % 8 = 3) :
  Odd (card { (a, b, c) : ℕ × ℕ × ℕ | p = a^2 + b * c ∧ 0 < b ∧ b < c ∧ c < Nat.sqrt p }) :=
sorry

end number_of_triples_is_odd_l560_560022


namespace jasmine_percentage_after_adding_l560_560580

def initial_solution_volume : ℕ := 80
def initial_jasmine_percentage : ℝ := 0.10
def additional_jasmine_volume : ℕ := 5
def additional_water_volume : ℕ := 15

theorem jasmine_percentage_after_adding :
  let initial_jasmine_volume := initial_jasmine_percentage * initial_solution_volume
  let total_jasmine_volume := initial_jasmine_volume + additional_jasmine_volume
  let total_solution_volume := initial_solution_volume + additional_jasmine_volume + additional_water_volume
  let final_jasmine_percentage := (total_jasmine_volume / total_solution_volume) * 100
  final_jasmine_percentage = 13 := by
  sorry

end jasmine_percentage_after_adding_l560_560580


namespace expand_expression_l560_560712

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560712


namespace coefficient_xy2_l560_560385

theorem coefficient_xy2 (a : ℚ) (h : (1 + a)^5 = 243) : 
  (∃ c : ℚ, c = 40 ∧ c = (binomial 5 2 * 2^2)) :=
by
  have h_a : a = 2 :=
    by {
      sorry -- solving for a using the given equation
    }
  use 40
  split
  { exact rfl }
  { simp [binomial, *] }

end coefficient_xy2_l560_560385


namespace sum_of_diffs_l560_560886

def S : Finset ℕ := (Finset.range 12).image (λ x, 2^x)

def pair_diff_sum (S : Finset ℕ) : ℕ :=
  (S.product S).sum (λ p, if p.1 > p.2 then p.1 - p.2 else 0)

theorem sum_of_diffs (N : ℕ) (h : N = pair_diff_sum S) : N = 51204 := by
  sorry

end sum_of_diffs_l560_560886


namespace sum_of_divisors_divisible_by_24_l560_560540

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : 24 ∣ n) :
  24 ∣ (Finset.sum (Finset.filter (λ d, d ∣ (n - 1)) (Finset.range n))) :=
by
  sorry

end sum_of_divisors_divisible_by_24_l560_560540


namespace plane_through_A_perpendicular_to_BC_l560_560577

-- Define points A, B, and C
def A : ℝ × ℝ × ℝ := (-8, 0, 7)
def B : ℝ × ℝ × ℝ := (-3, 2, 4)
def C : ℝ × ℝ × ℝ := (-1, 4, 5)

-- Define vector BC
def vectorBC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- Define the normal vector to the plane
def normalVector : ℝ × ℝ × ℝ := vectorBC

-- Define the plane equation using point A and the normal vector
def planeEquation (x y z : ℝ) : Prop := 2 * (x + 8) + 2 * y + 1 * (z - 7) = 0

-- Simplify the plane equation to the desired form
def simplifiedPlaneEquation (x y z : ℝ) : Prop := 2 * x + 2 * y + z + 9 = 0

-- Prove that the simplified plane equation represents the plane passing through A and perpendicular to vector BC
theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, planeEquation x y z ↔ simplifiedPlaneEquation x y z :=
  by sorry

end plane_through_A_perpendicular_to_BC_l560_560577


namespace range_of_a_l560_560889

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / (4 * x)
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) → 
  2 * Real.sqrt (Real.exp 1 - 2) ≤ a := sorry

end range_of_a_l560_560889


namespace max_value_PXQ_l560_560573

theorem max_value_PXQ :
  ∃ (X P Q : ℕ), (XX = 10 * X + X) ∧ (10 * X + X) * X = 100 * P + 10 * X + Q ∧ 
  (X = 1 ∨ X = 5 ∨ X = 6) ∧ 
  (100 * P + 10 * X + Q) = 396 :=
sorry

end max_value_PXQ_l560_560573


namespace num_repeating_two_digit_nums_l560_560602

theorem num_repeating_two_digit_nums : 
  let f := λ n : ℕ, 1001 * n
  ∃ (count : ℕ), count = (79 - 20 + 1) ∧ 
  (∀ x : ℕ, (20 ≤ x ∧ x ≤ 99) → (2000 ≤ f x ∧ f x ≤ 9999)) → count = 80 :=
by
  let f := λ n : ℕ, 1001 * n
  exists 80
  split
  {
    calc 99 - 20 + 1 = 80 : by norm_num
  }
  {
    intro h
    intros x hx
    split
    {
      norm_num
    }
    {
      norm_num
    }
  }

end num_repeating_two_digit_nums_l560_560602


namespace sixth_root_24414062515625_l560_560326

theorem sixth_root_24414062515625 :
  (∃ (x : ℕ), x^6 = 24414062515625) → (sqrt 6 24414062515625 = 51) :=
by
  -- Applying the condition expressed as sum of binomials
  have h : 24414062515625 = ∑ k in finset.range 7, binom 6 k * (50 ^ (6 - k)),
  sorry
  
  -- Utilize this condition to find the sixth root
  sorry

end sixth_root_24414062515625_l560_560326


namespace simplify_and_evaluate_expression_l560_560170

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l560_560170


namespace total_shaded_area_eq_l560_560105

/-- Define the overall grid as a rectangle with given width and height. -/
def grid := {width : ℕ // width = 15} × {height : ℕ // height = 5}

/-- Define the first shaded region as a horizontal stretch from the left edge, extending 3 units upward from the bottom -/
def shaded_region_1 (width : ℕ) (height : ℕ) : grid := 
  (width, height)
  where width = 6, height = 3

/-- Define the second shaded region as a stretch that begins 6 units from the left and extends 9 units horizontally, spanning from 3 units height to the top of the grid. -/
def shaded_region_2 (width : ℕ) (bottom_height : ℕ) (total_height : ℕ) : grid := 
  (width, total_height - bottom_height)
  where width = 9, bottom_height = 3, total_height = 5

/-- Prove the total area of the shaded region -/
theorem total_shaded_area_eq :
  let total_area := shaded_region_1.width * shaded_region_1.height + shaded_region_2.width * shaded_region_2.height
  total_area = 36 := 
by
  sorry

end total_shaded_area_eq_l560_560105


namespace total_games_in_season_l560_560291

theorem total_games_in_season (teams: ℕ) (division_teams: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) (total_games: ℕ) : 
  teams = 18 → division_teams = 9 → intra_division_games = 3 → inter_division_games = 2 → total_games = 378 :=
by
  sorry

end total_games_in_season_l560_560291


namespace gifts_needed_l560_560116

def num_teams : ℕ := 7
def num_gifts_per_team : ℕ := 2

theorem gifts_needed (h1 : num_teams = 7) (h2 : num_gifts_per_team = 2) : num_teams * num_gifts_per_team = 14 := 
by
  -- proof skipped
  sorry

end gifts_needed_l560_560116


namespace unique_total_scores_count_l560_560600

-- Define the possible scores a contestant can achieve on a single problem
inductive ProblemScore
| zero : ProblemScore
| one : ProblemScore
| seven : ProblemScore

-- Define the total scores possible for six problems
def possible_scores : list ℕ :=
  (List.range 7).bind (λ ones,
    (List.range 7).map (λ sevens,
      7 * sevens + ones
    )
  ) ++ (List.range 7).map (λ ones, 6 + ones)

-- Define the main theorem stating that there are 28 unique possible total scores
theorem unique_total_scores_count : (possible_scores.erase_dup.length = 28) := by
  sorry

end unique_total_scores_count_l560_560600


namespace cos_angle_OAB_is_negative_sqrt_2_over_10_l560_560110

-- Define two points A and B in a 2D Cartesian coordinate system
structure Point2D where
  x : ℝ
  y : ℝ

def O : Point2D := {x := 0, y := 0}

def A : Point2D := {x := -3, y := -4}
def B : Point2D := {x := 5, y := -12}

-- Define a function to calculate the vector from point P to point Q
def vector (P Q : Point2D) : Point2D := 
  {x := Q.x - P.x, y := Q.y - P.y}

-- The dot product of two vectors
def dotProduct (v1 v2 : Point2D) : ℝ := 
  v1.x * v2.x + v1.y * v2.y

-- The magnitude of a vector
def magnitude (v : Point2D) : ℝ := 
  Real.sqrt (v.x * v.x + v.y * v.y)

-- Prove the cosine of angle OAB
theorem cos_angle_OAB_is_negative_sqrt_2_over_10 :
  let AO := vector A O
  let AB := vector A B
  let dot := dotProduct AO AB
  let magAO := magnitude AO
  let magAB := magnitude AB
  (dot / (magAO * magAB)) = -Real.sqrt 2 / 10 := 
by
  let AO := vector A O
  let AB := vector A B
  let dot := dotProduct AO AB
  let magAO := magnitude AO
  let magAB := magnitude AB
  sorry

end cos_angle_OAB_is_negative_sqrt_2_over_10_l560_560110


namespace collinear_points_l560_560464

/-- Let ABC be a triangle with incenter I. The points where the incircle touches the sides BC and AC are D and E respectively. 
The intersection of the lines AI and DE is P. The midpoints of the sides BC and AB are M and N respectively. 
Prove that the points M, N, and P are collinear. --/
theorem collinear_points (ABC : Triangle) (I : Point) (D: Point) (E: Point) (P: Point)
  (M: Point) (N: Point)
  (h1 : Incenter ABC I)
  (h2 : TouchesIncircle ABC D BC)
  (h3 : TouchesIncircle ABC E AC)
  (h4 : Intersect AI DE P)
  (h5 : Midpoint M BC)
  (h6 : Midpoint N AB) :
  Collinear M N P :=
sorry

end collinear_points_l560_560464


namespace count_ways_to_fill_positions_l560_560599

theorem count_ways_to_fill_positions (n : ℕ) (h : n = 12) : 
    (choose n 1 * choose (n - 1) 1 * choose (n - 2) 1) = 1320 :=
by
    rw [h]
    -- Calculate the value according to the conditions and given number of members
    sorry

end count_ways_to_fill_positions_l560_560599


namespace sum_of_exponents_l560_560978

theorem sum_of_exponents (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ)
  (hm : ∀ i j, i < j → m i > m j)
  (hb : ∀ k, b k = 1 ∨ b k = -1)
  (h : (∑ i, b i * 3 ^ m i) = 2500) :
  (∑ i, m i) = 15 :=
sorry

end sum_of_exponents_l560_560978


namespace vector_relationship_proof_l560_560776

variables {P A B C : Type} [inner_product_space ℝ P]
variables {V : P → P → ℝ} (AB AC AP BC CP : V)

-- Given conditions as hypotheses
hypothesis h1 : BC = 2 * CP

-- Known vector relationships
hypothesis h2 : BC = AC - AB
hypothesis h3 : CP = AP - AC

-- Prove the given vector equation
theorem vector_relationship_proof :
  AP = - (1/2) * AB + (3/2) * AC :=
by {
  -- Implementation of the proof steps here (skipped)
  sorry
}

end vector_relationship_proof_l560_560776


namespace part1_simplify_expression_l560_560259

theorem part1_simplify_expression (x y : ℝ) : 
  ([(x^2 + y^2) - (x - y)^2 + 2 * y * (x - y)] / (4 * y) = x - (1/2) * y) :=
  sorry

end part1_simplify_expression_l560_560259


namespace sum_of_digits_9N_l560_560004

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9N (N : ℕ) (h : ∀ i j : ℕ, i < j → (N.digit i) < (N.digit j)) :
  sum_of_digits (9 * N) = 9 :=
by
  sorry

end sum_of_digits_9N_l560_560004


namespace sin_sum_l560_560794

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : Real.sin α = 5 / 13
axiom h4 : Real.cos β = 4 / 5

theorem sin_sum : Real.sin (α + β) = 56 / 65 := by
  sorry

end sin_sum_l560_560794


namespace largest_prime_factor_of_4752_l560_560997

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l560_560997


namespace arithmetic_sequences_count_l560_560900

/-
Problem Statement:
Let \( S = \{1, 2, \cdots, n\} \). Define \( A \) as an arithmetic sequence with a positive common difference consisting of at least two terms, where all terms are in \( S \). Additionally, adding any other element from \( S \) to \( A \) should not form an extended arithmetic sequence with the same common difference. Determine the number of such sequences \( A \).
-/
noncomputable def number_of_arithmetic_sequences (n : ℕ) : ℕ :=
  ⌊n ^ 2 / 4⌋

theorem arithmetic_sequences_count (n : ℕ) :
  number_of_arithmetic_sequences n = ⌊n ^ 2 / 4⌋ :=
by sorry

end arithmetic_sequences_count_l560_560900


namespace venician_angle_vlecks_l560_560921

theorem venician_angle_vlecks (full_circle_vlecks : ℕ) (earth_full_circle_degrees : ℕ) (angle_degrees : ℕ) :
  full_circle_vlecks = 600 → earth_full_circle_degrees = 360 → angle_degrees = 45 → (angle_degrees * full_circle_vlecks) / earth_full_circle_degrees = 75 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end venician_angle_vlecks_l560_560921


namespace volume_of_solid_0_l560_560131

-- Define the conditions of the region S
def regionS (x y : ℝ) : Prop := 
  (|6 - x| + y ≤ 8) ∧ (2 * y - x ≥ 10)

-- Define the solid of revolution volume problem
theorem volume_of_solid_0 : 
  let S := {p : ℝ × ℝ | regionS p.1 p.2} in
  let line := (λ (p : ℝ × ℝ), 2 * p.2 - p.1 - 10 = 0) in
  volume_of_solid S line = 0 :=
by
  sorry

end volume_of_solid_0_l560_560131


namespace O2_eq_O4_l560_560870

variable (A B C D O1 O2 O3 O4 : Type)
variable [InnerProductSpace ℝ O1] [InnerProductSpace ℝ O2]
variable [InnerProductSpace ℝ O3] [InnerProductSpace ℝ O4]
variable [Convex ℝ (A : Set ℝ)] [Convex ℝ (B : Set ℝ)]
variable [Convex ℝ (C : Set ℝ)] [Convex ℝ (D : Set ℝ)]

def isosceles_right_triangle (a b c : Type) : Prop := sorry

axiom h1 : isosceles_right_triangle A B O1
axiom h2 : isosceles_right_triangle B C O2
axiom h3 : isosceles_right_triangle C D O3
axiom h4 : isosceles_right_triangle D A O4
axiom h5 : O1 = O3

theorem O2_eq_O4 (ABCD : Convex ℝ (A ∪ B ∪ C ∪ D : Set ℝ)) 
  (h1 : isosceles_right_triangle A B O1)
  (h2 : isosceles_right_triangle B C O2)
  (h3 : isosceles_right_triangle C D O3)
  (h4 : isosceles_right_triangle D A O4)
  (h5 : O1 = O3) : O2 = O4 :=
by
  sorry

end O2_eq_O4_l560_560870


namespace correct_sequence_statements_l560_560936

-- Define the function f
def f (x : ℝ) : ℝ := log ((x^2 + 1) / |x|)

theorem correct_sequence_statements :
  (f_graph_symmetric about the y_axis : ∀ x ≠ 0, f(-x) = f(x)) ∧
  (f_increasing_decreasing : ∀ x, if x > 0 then (0 < x ∧ x < 1 → f(x) decreasing ∧ x ≥ 1 → f(x) increasing) else (x < 0 → (-1 < x ∧ x < 0 → f(x) increasing ∧ x ≤ -1 → f(x) decreasing))) ∧
  (f_extremum : (∀ x, x ≠ 0 → f(x) ≥ log 2) ∧ (∃ x, x ∉ [0, 1] → f(x) has no maximum value))
  :=
  -- Provide missing proof details by continuing steps logically discussed
  sorry

end correct_sequence_statements_l560_560936


namespace sum_of_five_digits_l560_560216

def smallest_prime_digit (n : ℕ) : Prop := nat.prime n ∧ 10 ≤ n ∧ n < 100
def perfect_square_digit (n : ℕ) : Prop := (∃ k, n = k * k) ∧ 10 ≤ n ∧ n < 100
def has_six_divisors (n : ℕ) : Prop := (finset.card (n.divisors) = 6) ∧ 10 ≤ n ∧ n < 100

theorem sum_of_five_digits : 
  ∃ (a b c d e : ℕ), 
  smallest_prime_digit a ∧ 
  perfect_square_digit b ∧ 
  has_six_divisors c ∧ 
  d ≠ max a (max b (max c (max d e))) ∧ 
  (∃ x ∈ [a, b, c], e = 3 * x) ∧ 
  finset.card (finset.from_list [a, b, c, d, e].bind nat.digits) = 10 ∧ 
  a + b + c + d + e = 180 := 
sorry

end sum_of_five_digits_l560_560216


namespace segment_length_condition_l560_560287

theorem segment_length_condition (a : ℝ) :
  (∀ (segments : Fin 11 → ℝ), (∀ i, segments i ≤ a) ∧ (∑ i, segments i = 1) → ∀ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i → (segments i + segments j > segments k) ∧ (segments i + segments k > segments j) ∧ (segments j + segments k > segments i)) ↔ (1 / 11 ≤ a ∧ a < 1 / 10) :=
sorry

end segment_length_condition_l560_560287


namespace javier_time_outlining_l560_560872

variable (O : ℕ)
variable (W : ℕ := O + 28)
variable (P : ℕ := (O + 28) / 2)
variable (total_time : ℕ := O + W + P)

theorem javier_time_outlining
  (h1 : total_time = 117)
  (h2 : W = O + 28)
  (h3 : P = (O + 28) / 2)
  : O = 30 := by 
  sorry

end javier_time_outlining_l560_560872


namespace yellow_balls_count_l560_560268

theorem yellow_balls_count {totalBalls white green red purple yellow : ℕ} 
  (h_total : totalBalls = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_prob : (white + green + yellow : ℕ) / totalBalls.toRat = 0.8) :
  yellow = 10 :=
by
  sorry

end yellow_balls_count_l560_560268


namespace negate_forall_cos_le_one_l560_560073

theorem negate_forall_cos_le_one :
  (¬ ∀ x : ℝ, cos x ≤ 1) ↔ ∃ x : ℝ, cos x > 1 :=
by
  sorry

end negate_forall_cos_le_one_l560_560073


namespace vector_ap_eq_l560_560778

variables {A B C P : Type} [affine_space P]
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C]

axiom vector_eq : ∀ (u v : A), (u = v) ↔ (u - v = 0)
axiom bc_eq_2cp (a b c p : A) : 
  (c - b = 2 • (p - c)) ↔ (c - b = 2 • (p - c))

theorem vector_ap_eq (a b c p : A) (h : (c - b = 2 • (p - c))) :
  (p - a) = (-1 / 2 • (b - a)) + (3 /2 • (c - a)) :=
sorry

end vector_ap_eq_l560_560778


namespace mira_additional_stickers_l560_560152

-- Define the conditions
def mira_stickers : ℕ := 31
def row_size : ℕ := 7

-- Define the proof statement
theorem mira_additional_stickers (a : ℕ) (h : (31 + a) % 7 = 0) : 
  a = 4 := 
sorry

end mira_additional_stickers_l560_560152


namespace coin_toss_binomial_distribution_l560_560271

noncomputable theory
open_locale classical

def X (n : ℕ) (p : ℝ) : Type := ℕ

theorem coin_toss_binomial_distribution :
  (∀ k : ℕ, k ≤ 4 → 
    P(X 4 (1/2) = k) = @finset.sum ℕ _ (finset.range (k+1))
    (λ m, (nat.choose 4 k * (1/2 : ℝ)^k * (1/2 : ℝ)^(4 - k)))) ∧
  (E(X 4 (1/2)) = 2) :=
by sorry

end coin_toss_binomial_distribution_l560_560271


namespace congruent_triangle_of_symmetric_lines_l560_560040

variable {ABC l : Type*} [Triangle ABC] [TangentToIncircle l ABC]
variable (l_a l_b l_c : Type*) [SymmetricTo l ABC l_a l_b l_c]

theorem congruent_triangle_of_symmetric_lines
  (h₁ : TangentToIncircle l ABC)
  (h₂ : SymmetricTo l ABC l_a l_b l_c) :
  Congruent
    (TriangleFormedByLines l_a l_b l_c)
    (Triangle ABC) :=
sorry

end congruent_triangle_of_symmetric_lines_l560_560040


namespace find_matrix_N_l560_560733

theorem find_matrix_N : ∃ (N : Matrix (Fin 4) (Fin 4) ℝ), (∀ w : Fin 4 → ℝ, (N.mulVec w) = (λ i, -6 * w i)) ∧ 
  N = ![
    ![-6, 0, 0, 0],
    ![0, -6, 0, 0],
    ![0, 0, -6, 0],
    ![0, 0, 0, -6]
  ] :=
by {
  sorry
}

end find_matrix_N_l560_560733


namespace bugs_meet_at_QS_five_l560_560988

structure Triangle :=
  (P Q R : Type)
  (PQ QR PR : ℝ)
  (PQ_eq : PQ = 7)
  (QR_eq : QR = 8)
  (PR_eq : PR = 9)

def bugs_meet (t : Triangle) : ℝ := 
  let perimeter := t.PQ + t.QR + t.PR in
  let half_perimeter := perimeter / 2 in
  half_perimeter - t.PQ

theorem bugs_meet_at_QS_five (t : Triangle) : 
  t.PQ = 7 ∧ t.QR = 8 ∧ t.PR = 9 → 
  bugs_meet t = 5 :=
  by
  sorry

end bugs_meet_at_QS_five_l560_560988


namespace sum_of_two_dice_is_4_l560_560033

open Classical

variable (die_faces : List ℕ := [1, 2, 3, 4, 5, 6])

def number_of_ways_to_sum_to (sum_target : ℕ) : ℕ :=
  (die_faces.product die_faces).count (λ (pair : ℕ × ℕ), pair.fst + pair.snd = sum_target)

theorem sum_of_two_dice_is_4 : number_of_ways_to_sum_to 4 = 3 := by
  sorry

end sum_of_two_dice_is_4_l560_560033


namespace equation_of_circle_l560_560191

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem equation_of_circle:
  let A := (2, 0)
  let B := (2, -2)
  let C := midpoint A B in
  let r := distance C A in
  C = (2, -1) ∧ r = 1 →
  ∃ k l R, (k, l) = (2, -1) ∧ R = 1 ∧ ∀ x y: ℝ, (x - k)^2 + (y - l)^2 = R :=
by
  sorry

end equation_of_circle_l560_560191


namespace sum_of_digits_9N_l560_560003

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9N (N : ℕ) (h : ∀ i j : ℕ, i < j → (N.digit i) < (N.digit j)) :
  sum_of_digits (9 * N) = 9 :=
by
  sorry

end sum_of_digits_9N_l560_560003


namespace circumcircle_tangent_l560_560184

variables {A B C K L M P Q : Type}
variables [IsoscelesTriangle A B C] [Circle σ] [Tangent σ A B] [Tangent σ A C]
variables [OnCircle K σ] [OnCircle L σ] [Intersects AK σ M] 
variables [Symmetric K B P] [Symmetric K C Q]
variables [OnCircle PM σ] [OnCircle PQ σ]

theorem circumcircle_tangent :
  tangent (circumcircle ⟨P, M, Q⟩) σ :=
sorry

end circumcircle_tangent_l560_560184


namespace expand_polynomial_l560_560689

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560689


namespace minimize_squares_in_rectangle_l560_560247

theorem minimize_squares_in_rectangle (w h : ℕ) (hw : w = 63) (hh : h = 42) : 
  ∃ s : ℕ, s = Nat.gcd w h ∧ s = 21 :=
by
  sorry

end minimize_squares_in_rectangle_l560_560247


namespace expand_product_l560_560692

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560692


namespace flowers_on_porch_l560_560913

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560913


namespace tim_youth_comparison_l560_560985

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l560_560985


namespace grading_combinations_l560_560280

/-- There are 12 students in the class. -/
def num_students : ℕ := 12

/-- There are 4 possible grades (A, B, C, and D). -/
def num_grades : ℕ := 4

/-- The total number of ways to assign grades. -/
theorem grading_combinations : (num_grades ^ num_students) = 16777216 := 
by
  sorry

end grading_combinations_l560_560280


namespace sequence_contains_infinite_perfect_squares_l560_560934

theorem sequence_contains_infinite_perfect_squares (α : ℝ) (hα : α = Real.sqrt 2) : 
  ∃ (f : ℕ → ℕ), 
    (∀ n : ℕ, f n = Int.to_nat ⌊ (n : ℝ) * α ⌋) ∧ 
    (∀ m : ℕ, ∃ n : ℕ, f n = m * m) :=
by 
  sorry

end sequence_contains_infinite_perfect_squares_l560_560934


namespace pow_mod_eq_l560_560235

theorem pow_mod_eq {a : ℕ} (h : a % 13 = 11) : (11 ^ 2023) % 13 = 11 :=
by
have h1 : 11 % 13 = -2 % 13 := by exact rfl
have h2 : 11 ^ 2023 % 13 = (-2 ^ 2023) % 13 := by { rw [h1], ring_nf }
have h3 : -2 ^ 2023 = -1 * 2 ^ 2023 := by { rw neg_pow, exact pow_one 2 }
have h4 : -2 ^ 2023 % 13 = (-1 * 2 ^ 2023) % 13 := by rw h3
have h5 : 2 ^ 2023 = (2 ^ 3) ^ 674 * 2 := by exact_mod_cast pow_succ' 2 2022
have h6 : (2 ^ 3) % 13 = 8 % 13 := by exact_mod_cast nat.mod_eq_of_lt (by norm_num)
have h7 : 8 % 13 = (-5) % 13 := by exact_mod_cast (by norm_num)
have h8 : ((-5) ^ 674 % 13) = (5 ^ 674 % 13) := by { 
  rw [neg_pow, pow_mul, pow_two, h7], 
  exact pow_674 5,
}
have h9 : (5 ^ 2 % 13) = 12 := by norm_num
have h10 : (5 ^ 674 % 13) = (-1) ^ 337 % 13 := by rw [← h9, pow_mul, pow_two, pow_mod_13]
rw [h2, h4, h5, h6, h8, h10, neg_eq_neg_one, pow_674_odd, by mod_add_equiv]
exact_mod_cast 11

end pow_mod_eq_l560_560235


namespace problem_proof_l560_560466

-- Given a, b are positive integers such that gcd(a, b) = 1 and a, b have different parity
variables {a b : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b)
variables (h3 : Nat.gcd a b = 1)
variables (h4 : ¬ Nat.even a ∧ Nat.even b ∨ Nat.even a ∧ ¬ Nat.even b)
variables (S : set ℕ)

-- Given properties of set S
variables (h5 : a ∈ S) (h6 : b ∈ S)
variables (h7 : ∀ {x y z : ℕ}, x ∈ S → y ∈ S → z ∈ S → x + y + z ∈ S)

-- Prove every positive integer greater than 2ab belongs to S
theorem problem_proof : ∀ n : ℕ, 2 * a * b < n → n ∈ S :=
begin
  sorry
end

end problem_proof_l560_560466


namespace probability_of_three_common_books_l560_560919

-- Define conditions as hypotheses
variables (A : Type) (B : Type)
variables (books : Finset A) (H_choices B : Finset A) (B_choices B : Finset A)

-- Define the conditions
def ms_carr_books := books.card = 12
def harold_picks := H_choices.card = 5
def betty_picks := B_choices.card = 5

-- Define the events
def exactly_three_common_books := (H_choices ∩ B_choices).card = 3

-- Define the probability function
noncomputable def probability_common_books : ℚ :=
  ((finset.card (finset.pow books 3)).nat_choose 3 * 
  (finset.card (finset.erase books 3)).nat_choose 2 *
  (finset.card (finset.erase books 7)).nat_choose 2).to_rat /
  (finset.card (finset.powerset_len 5 books) * 
  (finset.card (finset.powerset_len 5 books))).to_rat

-- The statement we need to prove
theorem probability_of_three_common_books
  (h_books : ms_carr_books books)
  (h_harold : harold_picks H_choices)
  (h_betty : betty_picks B_choices)
  (h_three_common : exactly_three_common_books H_choices B_choices) :
  probability_common_books books = 55 / 209 := sorry

end probability_of_three_common_books_l560_560919


namespace complex_abs_value_l560_560773

theorem complex_abs_value (x y : ℝ) (h : (x + y * complex.I) * complex.I = 1 + complex.I) : 
  complex.abs (x + 2 * y * complex.I) = real.sqrt 5 :=
sorry

end complex_abs_value_l560_560773


namespace repayment_amount_formula_l560_560562

def loan_principal := 480000
def repayment_years := 20
def repayment_months := repayment_years * 12
def monthly_interest_rate := 0.004
def monthly_principal_repayment := loan_principal / repayment_months

def interest_for_nth_month (n : ℕ) : ℚ :=
  (loan_principal - (n - 1) * monthly_principal_repayment) * monthly_interest_rate

def repayment_amount_nth_month (n : ℕ) : ℚ :=
  monthly_principal_repayment + interest_for_nth_month n

theorem repayment_amount_formula (n : ℕ) (hn : 1 ≤ n ∧ n ≤ repayment_months) :
  repayment_amount_nth_month n = 3928 - 8 * n := by
sorry

end repayment_amount_formula_l560_560562


namespace xyz_identity_l560_560834

theorem xyz_identity (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : xy + xz + yz = 32) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 1400 := 
by 
  -- Proof steps will be placed here, use sorry for now
  sorry

end xyz_identity_l560_560834


namespace proof_problem_l560_560798

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 1 - x else (0.2 : ℝ)^(x.log) -- note that (x.log) gives natural logarithm

theorem proof_problem (a : ℝ) (h : f (a + 5) = -1) : f a = 1 :=
sorry

end proof_problem_l560_560798


namespace perpendicular_planes_and_lines_l560_560095

-- Defining the planes and lines
variables (α β γ : Plane) (l m : Line)

-- Defining the conditions
def intersection_of_planes (β γ : Plane) (l : Line) : Prop :=
  ∃ p, p ∈ l ∧ p ∈ β ∧ p ∈ γ

def line_parallel_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ p ∈ l, p ∈ α

def line_contained_in_plane (m : Line) (α : Plane) : Prop :=
  ∀ p ∈ m, p ∈ α

def line_perpendicular_to_plane (m : Line) (γ : Plane) : Prop :=
  ∃ q ∈ γ, ∀ r ∈ m, q ∉ r

-- The statement we need to prove
theorem perpendicular_planes_and_lines (h1 : intersection_of_planes β γ l)
    (h2 : line_parallel_to_plane l α) (h3 : line_contained_in_plane m α)
    (h4 : line_perpendicular_to_plane m γ) : 
    (α ⟂ γ ∧ l ⟂ m) :=
sorry

end perpendicular_planes_and_lines_l560_560095


namespace divides_euler_totient_l560_560168

open Nat

-- Definition of Euler's totient function (φ)
def euler_totient (n : ℕ) : ℕ := n.to_nat.totient

-- Main statement: Prove that n divides φ(a^n - 1) for any integers a and n
theorem divides_euler_totient {a n : ℤ} (ha : a ≠ 0) (hn : n ≠ 0) :
  n.to_nat ∣ euler_totient ((a ^ n - 1).to_nat) :=
by 
  -- In this proof, we should prove that n | φ(a^n - 1)
  sorry

end divides_euler_totient_l560_560168


namespace matrix_pow_eq_l560_560878

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![!![0, 1, 0], !![0, 0, 1], !![1, 0, 0]]

theorem matrix_pow_eq :
  B ^ 100 = B :=
by
  sorry

end matrix_pow_eq_l560_560878


namespace inequality_abc_l560_560141

open Real

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b * c = 1) : 
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_abc_l560_560141


namespace tetrahedron_edge_length_l560_560020

theorem tetrahedron_edge_length :
  ∀ (r : ℝ) (s : ℝ),
    r = 2 ∧
    (∃ p1 p2 p3 p4 p5 : ℝ × ℝ × ℝ,
      -- p1, p2, p3, p4 are the centers of the four balls on the floor
      p1.2.2 = 0 ∧ p2.2.2 = 0 ∧ p3.2.2 = 0 ∧ p4.2.2 = 0 ∧
      -- distances between these centers on the floor is 4 (each shifted by diameter)
      dist p1 p2 = 4 ∧ dist p1 p3 = 4 ∧ dist p1 p4 = 4 ∧ 
      dist p2 p3 = 4 ∧ dist p2 p4 = 4 ∧ dist p3 p4 = 4 ∧
      -- p5 is the center of the top ball, which is directly above the center of the square
      p5 = (0, 0, 4) ∧
      -- s is the edge length between the center of the top ball to any of the bottom balls
      (∀ x ∈ {p1, p2, p3, p4}, dist p5 x = s)) →
  s = 2 * Real.sqrt 6 :=
by
  intros r s hr hs
  cases hr
  cases hs with p1 hs
  cases hs with p2 hs
  cases hs with p3 hs
  cases hs with p4 hs
  cases hs with p5 hs
  cases hs
  -- We would proceed with the proof, but we are adding sorry as per the guidelines
  sorry

end tetrahedron_edge_length_l560_560020


namespace lines_parallel_k_eq_3_l560_560080

/-- Given lines l₁ : (k - 3) * x + (4 - k) * y + 1 = 0
    and l₂ : (k - 3) * x - y + 1 = 0, 
    if these lines are parallel, 
    then k must be 3. -/
theorem lines_parallel_k_eq_3 
    (k : ℝ) (x y : ℝ) 
    (l₁ : (k - 3) * x + (4 - k) * y + 1 = 0)
    (l₂ : (k - 3) * x - y + 1 = 0)
    (h_parallel : ∀ x y, l₁ = l₂) : 
    k = 3 := by 
  sorry

end lines_parallel_k_eq_3_l560_560080


namespace angle_ABC_is_60_l560_560964

/-- 
  Given the parallelogram ABCD with ∠B < 90° and AB < BC. Points E and F are chosen on the
  circumcircle of triangle ABC such that the tangents to the circumcircle at these points pass 
  through D. Given that ∠EDA = ∠FDC, prove that ∠ABC = 60°.
-/
theorem angle_ABC_is_60
  (A B C D E F : Point)
  (h1 : Parallelogram A B C D)
  (h2 : Angle B < 90)
  (h3 : AB < BC)
  (h4 : OnCircumcircle E (Triangle A B C))
  (h5 : OnCircumcircle F (Triangle A B C))
  (h6 : TangentToCircumcircleAt D E)
  (h7 : TangentToCircumcircleAt D F)
  (h8 : Angle EDA = Angle FDC)
  : Angle ABC = 60 :=
sorry

end angle_ABC_is_60_l560_560964


namespace flowers_on_porch_l560_560910

-- Definitions based on problem conditions
def total_plants : ℕ := 80
def flowering_percentage : ℝ := 0.40
def fraction_on_porch : ℝ := 0.25
def flowers_per_plant : ℕ := 5

-- Theorem statement
theorem flowers_on_porch (h1 : total_plants = 80)
                         (h2 : flowering_percentage = 0.40)
                         (h3 : fraction_on_porch = 0.25)
                         (h4 : flowers_per_plant = 5) :
    (total_plants * seminal (flowering_percentage * fraction_on_porch) * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560910


namespace carol_rectangle_length_l560_560312

theorem carol_rectangle_length :
  let j_length := 6
  let j_width := 30
  let c_width := 15
  let c_length := j_length * j_width / c_width
  c_length = 12 := by
  sorry

end carol_rectangle_length_l560_560312


namespace unruly_quadratic_q1_l560_560317

-- Definitions for condition
def quadratic_poly (r s : ℝ) : ℝ → ℝ := λ x, (x - r)^2 - s
noncomputable def q_q_eq_zero_four_real_roots (q : ℝ → ℝ) : Prop :=
  -- Placeholder for the condition ensuring q(q(x)) = 0 has four real roots with one double root
  sorry -- You would usually specify the detailed condition here

-- The statement to prove: Given the conditions, the value of q(1) gets computed as 7/4.
theorem unruly_quadratic_q1 (r : ℝ) (s : ℝ) (h_s_eq_neg_r : s = -r)
  (h_q_unruly : q_q_eq_zero_four_real_roots (quadratic_poly r s)) :
  quadratic_poly r s 1 = 7 / 4 := by
  sorry

end unruly_quadratic_q1_l560_560317


namespace smallest_x_is_1_l560_560463

noncomputable def smallest_x : ℝ :=
  let x := classical.some (exists_real_of_coss_eq_coss (λ x, x > 0)) in x

theorem smallest_x_is_1 : smallest_x = 1 := by
  -- Proof implementation would go here
  sorry

/-- Auxiliary lemma to help set up the main problem -/
lemma exists_real_of_coss_eq_coss (H : ℝ → Prop) : ∃ x, H x ∧ ∀ y, H y → x ≤ y := by
  sorry

end smallest_x_is_1_l560_560463


namespace sum_alternating_series_l560_560639

theorem sum_alternating_series :
  (∑ i in Finset.range 101, if i % 2 = 0 then i + 1 else -(i + 1)) = 51 :=
by
  sorry

end sum_alternating_series_l560_560639


namespace seq_eq_a1_b1_l560_560882

theorem seq_eq_a1_b1 {a b : ℕ → ℝ} 
  (h1 : ∀ n, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n, a n > 0) :
  a 1 = b 1 := 
sorry

end seq_eq_a1_b1_l560_560882


namespace terminating_decimal_count_l560_560355

theorem terminating_decimal_count :
  (∃ k : ℕ, ∀ n : ℕ, 1 ≤ n ∧ n ≤ 598 → (n % 3 = 0) →
    ∃ m : ℕ, n = 3 * m) ∧ 
    ∀ n : ℕ, (1 ≤ n ∧ n ≤ 598 ∧ n % 3 = 0) → 
    (nat.ceil ((598 - 1) / 3) + 1 = 199) := sorry

end terminating_decimal_count_l560_560355


namespace area_of_triangle_H1_H2_H3_l560_560469

variable (Q : Point) (D E F : Point)
variable (H1 H2 H3 : Point)
variable [triQEF: Triangle (Q, E, F)] [triQFD: Triangle (Q, F, D)] [triQDE: Triangle (Q, D, E)]

noncomputable def centroid (triangle : Triangle) : Point := sorry

axiom area_triangle_DEF : area (triangle_DEF) = 24

theorem area_of_triangle_H1_H2_H3 (h1_centroid : H1 = centroid (triangle Q E F))
    (h2_centroid : H2 = centroid (triangle Q F D)) (h3_centroid : H3 = centroid (triangle Q D E)) : 
  area (triangle H1 H2 H3) = 2.666... :=
sorry

end area_of_triangle_H1_H2_H3_l560_560469


namespace length_of_field_l560_560535

variable (w : ℕ) (l : ℕ)

def length_field_is_double_width (w l : ℕ) : Prop :=
  l = 2 * w

def pond_area_equals_one_eighth_field_area (w l : ℕ) : Prop :=
  36 = 1 / 8 * (l * w)

theorem length_of_field (w l : ℕ) (h1 : length_field_is_double_width w l) (h2 : pond_area_equals_one_eighth_field_area w l) : l = 24 := 
by
  sorry

end length_of_field_l560_560535


namespace find_x_l560_560831

theorem find_x
  (a b x : ℝ)
  (h1 : a * (x + 2) + b * (x + 2) = 60)
  (h2 : a + b = 12) :
  x = 3 :=
by
  sorry

end find_x_l560_560831


namespace parallel_vectors_lambda_eq_one_l560_560082

variable (λ : ℝ)

def a : ℝ × ℝ := (1, 3 * λ)
def b : ℝ × ℝ := (2, 7 - λ)

theorem parallel_vectors_lambda_eq_one
  (h : ∃ k : ℝ, a = (k * b.1, k * b.2)) : λ = 1 := by
  sorry

end parallel_vectors_lambda_eq_one_l560_560082


namespace greatest_sum_on_circle_l560_560977

theorem greatest_sum_on_circle : 
  ∃ x y : ℤ, x^2 + y^2 = 169 ∧ x ≥ y ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 169 → x' ≥ y' → x + y ≥ x' + y') := 
sorry

end greatest_sum_on_circle_l560_560977


namespace percy_swimming_hours_l560_560926

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l560_560926


namespace intersection_of_M_and_N_l560_560128

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := 
by sorry

end intersection_of_M_and_N_l560_560128


namespace smallest_k_l560_560144

theorem smallest_k (q : ℕ) (hq_prm : prime q) (hq_dig : 10^1004 ≤ q ∧ q < 10^1005) : ∃ k : ℕ, k = 1 ∧ 24 ∣ (q^2 - k) :=
by
  use 1
  sorry

end smallest_k_l560_560144


namespace unsatisfactory_tests_l560_560223

theorem unsatisfactory_tests {n k : ℕ} (h1 : n < 50) 
  (h2 : n % 7 = 0) 
  (h3 : n % 3 = 0) 
  (h4 : n % 2 = 0)
  (h5 : n = 7 * (n / 7) + 3 * (n / 3) + 2 * (n / 2) + k) : 
  k = 1 := 
by 
  sorry

end unsatisfactory_tests_l560_560223


namespace unique_solution_of_system_l560_560023

theorem unique_solution_of_system :
  ∀ (a : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) →
  ((a = 1 ∧ ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃ x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0)) :=
by
  sorry

end unique_solution_of_system_l560_560023


namespace deputy_more_enemies_than_friends_l560_560435

theorem deputy_more_enemies_than_friends (deputies : Type) 
  (friendship hostility indifference : deputies → deputies → Prop)
  (h_symm_friend : ∀ (a b : deputies), friendship a b → friendship b a)
  (h_symm_hostile : ∀ (a b : deputies), hostility a b → hostility b a)
  (h_symm_indiff : ∀ (a b : deputies), indifference a b → indifference b a)
  (h_enemy_exists : ∀ (d : deputies), ∃ (e : deputies), hostility d e)
  (h_principle : ∀ (a b c : deputies), hostility a b → friendship b c → hostility a c) :
  ∃ (d : deputies), ∃ (f e : ℕ), f < e :=
sorry

end deputy_more_enemies_than_friends_l560_560435


namespace problem_solution_l560_560757

theorem problem_solution (a b c d : ℕ) (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  (a * 10^3 + b * 10^2 + c * 10 + d) = 1949 :=
by
  sorry

end problem_solution_l560_560757


namespace total_swimming_hours_over_4_weeks_l560_560929

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l560_560929


namespace square_in_semicircle_l560_560101

theorem square_in_semicircle (Q : ℝ) (h1 : ∃ Q : ℝ, (Q^2 / 4) + Q^2 = 4) : Q = 4 * Real.sqrt 5 / 5 := sorry

end square_in_semicircle_l560_560101


namespace ratio_proof_l560_560457

-- Define the given triangle and relevant points
variables {A B C J D T E P F A' M : Point}
variables (h_triangle: Triangle A B C)
variables (h_AB_less_AC: AB < AC)
variables (h_J_excenter: IsExcenter J A B C)
variables (h_intersect_D: Line AJ ∩ Line BC = {D})
variables (h_perp_bisector_T: IsPerpendicularBisector A J T BC)
variables (h_circle_E: OnCircle E (Circumcircle A B C))
variables (h_TE_equal_TA: distance T E = distance T A)
variables (h_TP_equal_TB: distance T P = distance T B)
variables (h_BF_equal_AP: distance B F = distance A P)
variables (h_symmetric_A_prime: PointReflectLineSymmetric A A' BC)
variables (h_circle_intersect_M: IntersectCircles (Circumcircle A' E F) (Circumcircle A B C) = {M})

theorem ratio_proof : 
  ratio (distance M D) (distance M F) (distance M J) =
  ratio (distance A D) (distance A F) (distance A J) :=
sorry

end ratio_proof_l560_560457


namespace expand_expression_l560_560713

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560713


namespace four_member_table_seating_l560_560290

theorem four_member_table_seating (n : ℕ) (members : Finset ℕ) (knows : ℕ → ℕ → Prop)
  (Hmembers : members.card = n)
  (Hknows : ∀ x ∈ members, 1 ≤ (members.filter (knows x)).card ∧ (members.filter (knows x)).card ≤ n - 2)
  (Hmutual : ∀ x y ∈ members, knows x y ↔ knows y x) :
  ∃ A B C D ∈ members, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ knows A D ∧ ¬knows A B ∧ knows B C ∧ ¬knows B A ∧ knows C B ∧ ¬knows C D ∧ knows D A ∧ ¬knows D C :=
begin
  sorry
end

end four_member_table_seating_l560_560290


namespace num_three_digit_palindromes_eq_90_l560_560611

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n

theorem num_three_digit_palindromes_eq_90 : 
  (Finset.filter is_three_digit_palindrome (Finset.range 1000)).card = 90 :=
  sorry

end num_three_digit_palindromes_eq_90_l560_560611


namespace compare_fractions_l560_560367

theorem compare_fractions (x y : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : 0 < n) :
  (x^n / (1 - x^2) + y^n / (1 - y^2)) ≥ ((x^n + y^n) / (1 - x * y)) :=
by sorry

end compare_fractions_l560_560367


namespace count_matching_placements_l560_560493

theorem count_matching_placements : 
  ∃ (placements : Finset (Fin 6 → Fin 6)), 
  (∀ f ∈ placements, (Finset.card (Finset.filter (λ x => f x = x) (Finset.univ : Finset (Fin 6))) = 3)) ∧ 
  Finset.card placements = 40 :=
by
  sorry

end count_matching_placements_l560_560493


namespace least_possible_value_l560_560460

def is_valid_set (T : Finset ℕ) : Prop :=
  T ⊆ Finset.range (15+1) ∧ T.card = 8 ∧ (∀ x y ∈ T, x < y → ¬ (y % x = 0))

theorem least_possible_value :
  ∀ T : Finset ℕ, is_valid_set T → ∃ a ∈ T, a = 5 := 
by
  sorry

end least_possible_value_l560_560460


namespace blue_part_length_l560_560609

variable (total_length : ℝ) (black_part white_part blue_part : ℝ)

-- Conditions
axiom h1 : black_part = 1 / 8 * total_length
axiom h2 : white_part = 1 / 2 * (total_length - black_part)
axiom h3 : total_length = 8

theorem blue_part_length : blue_part = total_length - black_part - white_part :=
by
  sorry

end blue_part_length_l560_560609


namespace find_x_l560_560370

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l560_560370


namespace find_a_l560_560048

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {1, 3, a}) (hB : B = {1, a^2 - a + 1}) (h_subset : B ⊆ A) :
  a = -1 ∨ a = 2 := 
by
  sorry

end find_a_l560_560048


namespace inscribed_triangles_count_l560_560078

theorem inscribed_triangles_count (n : ℕ) (h : n = 10) : 
  let vertices := n.choose 3 in
  vertices = 360 :=
by
  sorry

end inscribed_triangles_count_l560_560078


namespace prove_probability_prove_initial_coin_requirement_l560_560194

noncomputable def minibus_change_probability : ℝ :=
  -- Define the conditions and state the target probability outcome
  let num_passengers := 15
  let face_value_100 := (1 / 2 : ℝ)
  let face_value_50 := (1 / 2 : ℝ)
  let probability_100 := face_value_100 
  let probability_50 := face_value_50
  let trajectories_count := 2 ^ num_passengers
  let favourable_trajectories := ((nat.choose (num_passengers) (num_passengers / 2 + 1)) / 2 : ℝ) in
  favourable_trajectories / trajectories_count

noncomputable def initial_coins_requirement : ℚ :=
  -- Define the conditions and state the minimum initial coin requirement to ensure 0.95 probability
  let num_passengers := 15
  let threshold_probability := (0.95 : ℝ)
  let required_probability := threshold_probability * 2 ^ num_passengers 
  let min_initial_coins := 275 -- Derived from calculations in solution
  min_initial_coins 

theorem prove_probability :
  minibus_change_probability ≈ 0.196 := by
  -- Detailed mathematical proof required here
  sorry

theorem prove_initial_coin_requirement :
  initial_coins_requirement = 275 := by
  -- Detailed mathematical proof required here
  sorry

end prove_probability_prove_initial_coin_requirement_l560_560194


namespace fans_received_all_items_l560_560353

open Nat

-- Given conditions
def isNthFan (n: ℕ) (m: ℕ) : Prop := m % n = 0

def t (m: ℕ) : Prop := isNthFan 60 m
def h (m: ℕ) : Prop := isNthFan 45 m
def k (m: ℕ) : Prop := isNthFan 75 m
def total_fans : ℕ := 4500

-- Proof statement
theorem fans_received_all_items : 
  ∃ n, n = 5 ∧ 
  ∀ m, m <= total_fans → (t m ∧ h m ∧ k m → m % (LCM 60 (LCM 45 75)) = 0) → n := 
  sorry

end fans_received_all_items_l560_560353


namespace lily_pad_covering_entire_lake_l560_560106

theorem lily_pad_covering_entire_lake (doubles_in_size_every_day : ∀ (n : ℕ), lily_pad_size n) 
    (half_lake_covered_in_24_days : lily_pad_size 24 = lake_size / 2)
    (lake_size : ℝ) (lily_pad_size : ℕ → ℝ) :
  lily_pad_size 25 = lake_size :=
by
  sorry

end lily_pad_covering_entire_lake_l560_560106


namespace reflection_points_reflection_line_l560_560531

-- Definitions of given points and line equation
def original_point : ℝ × ℝ := (2, 3)
def reflected_point : ℝ × ℝ := (8, 7)

-- Definitions of line parameters for y = mx + b
variable {m b : ℝ}

-- Statement of the reflection condition
theorem reflection_points_reflection_line : m + b = 9.5 := by
  -- sorry to skip the actual proof
  sorry

end reflection_points_reflection_line_l560_560531


namespace expand_expression_l560_560670

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560670


namespace find_number_of_friends_l560_560556

def dante_balloons : Prop :=
  ∃ F : ℕ, (F > 0 ∧ (250 / F) - 11 = 39) ∧ F = 5

theorem find_number_of_friends : dante_balloons :=
by
  sorry

end find_number_of_friends_l560_560556


namespace expand_expression_l560_560728

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560728


namespace range_y_minus_x_l560_560869

theorem range_y_minus_x (A B C D P : Point) (x y : ℝ)
  (hD : D = 2 * (A + D))
  (hP : ∃ (P : Point), segment BD P ∧ AP = x * AB + y * AC)
  (hx : x > 0)
  (hy : y > 0) :
  -1 < y - x ∧ y - x < 1 / 3 :=
by 
  sorry

end range_y_minus_x_l560_560869


namespace cost_per_meat_with_rush_shipping_l560_560747

theorem cost_per_meat_with_rush_shipping (cost_per_pack : ℝ) (num_types : ℕ) (rush_percentage : ℝ) : num_types = 4 ∧ cost_per_pack = 40 ∧ rush_percentage = 0.30 → cost_per_pack * (1 + rush_percentage) / num_types = 13 := 
by
  intro h
  obtain ⟨hn, hc, hr⟩ := h
  rw [hn, hc, hr]
  norm_num
  sorry

end cost_per_meat_with_rush_shipping_l560_560747


namespace expand_product_l560_560699

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560699


namespace pentagon_equality_product_l560_560468

theorem pentagon_equality_product :
  ∀ (x y : ℕ → ℝ),
  (∀ k : ℕ,
    (∀ (i : ℕ), (x (i + 1)) ^ k + (x (i + 2)) ^ k + (x (i + 3)) ^ k + (x (i + 4)) ^ k + (x (i + 5)) ^ k
    = (y (i + 1)) ^ k + (y (i + 2)) ^ k + (y (i + 3)) ^ k + (y (i + 4)) ^ k + (y (i + 5)) ^ k))
  → (∏ (k : ℕ) in {1, 2, 3, 4, 6, 8}, k = 1152) :=
sorry

end pentagon_equality_product_l560_560468


namespace expand_product_l560_560691

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l560_560691


namespace cats_needed_cats_example_l560_560258

theorem cats_needed (cats_have : ℕ) (total_cats : ℕ) (cats_needed : ℕ) 
  (h_1 : cats_have = 11) (h_2 : total_cats = 43) : 
  cats_needed = total_cats - cats_have :=
by sorry

theorem cats_example : 
  cats_needed 11 43 32 sorry sorry = 32 :=
by sorry

end cats_needed_cats_example_l560_560258


namespace multiply_m_t_l560_560461

-- Define the conditions
variable {g : ℝ → ℝ}
hypotheses (h : ∀ x y : ℝ, g(g(x) + y) = g(x^2 - y) + 2 * g(x) * y)

-- Define m and t according to the problem's conditions
def m : ℕ := 2
def t : ℕ := g 4 + 16

-- Define the main statement to prove
theorem multiply_m_t : m * t = 32 :=
by
  sorry

end multiply_m_t_l560_560461


namespace inequality_always_true_l560_560754

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l560_560754


namespace quadratic_expression_factors_l560_560827

theorem quadratic_expression_factors (m : ℝ) :
  (m - 8) * (m + 3) = m^2 - 5m - 24 := 
by
  sorry

end quadratic_expression_factors_l560_560827


namespace expand_expression_l560_560662

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560662


namespace volume_percentage_error_l560_560627

theorem volume_percentage_error (L W H : ℝ) (hL : L > 0) (hW : W > 0) (hH : H > 0) :
  let V_true := L * W * H
  let L_meas := 1.08 * L
  let W_meas := 1.12 * W
  let H_meas := 1.05 * H
  let V_calc := L_meas * W_meas * H_meas
  let percentage_error := ((V_calc - V_true) / V_true) * 100
  percentage_error = 25.424 :=
by
  sorry

end volume_percentage_error_l560_560627


namespace minimum_checks_for_code_l560_560521

-- Define the conditions and question
def five_digit_code_includes_21_16 (n : ℕ) : Prop :=
  (∃ a b c d e : ℕ, n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧ 
  (21 = 10 * b + c ∨ 21 = 10 * c + d ∨ 21 = 10 * d + e) ∧ 
  (16 = 10 * c + d ∨ 16 = 10 * d + e))

-- Problem: Prove the minimum number of five-digit numbers that need to be checked
theorem minimum_checks_for_code : 
  ∃ m, (∀ n, five_digit_code_includes_21_16 n → n ≤ m) ∧ m = 4025 :=
begin
  sorry
end

end minimum_checks_for_code_l560_560521


namespace prove_BS_eq_BC_l560_560620

noncomputable def problem_statement : Prop :=
  ∃ (A B C D E S : Point)
    (AngleA : Real)
    (AngleC : Real)
    (AngleCAD : Real)
    (AngleACE : Real)
    (B_eq_C : Bool), 
    AngleA = 30 ∧
    AngleC = 54 ∧
    AngleCAD = 12 ∧
    AngleACE = 6 ∧
    B_eq_C = (BS = BC)

theorem prove_BS_eq_BC : problem_statement :=
sorry

end prove_BS_eq_BC_l560_560620


namespace max_and_next_max_values_l560_560735

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log a) / b

theorem max_and_next_max_values :
  let values := [4.0^(1/4), 5.0^(1/5), 16.0^(1/16), 25.0^(1/25)]
  ∃ max2 max1, 
    max1 = 4.0^(1/4) ∧ max2 = 5.0^(1/5) ∧ 
    (∀ x ∈ values, x <= max1) ∧ 
    (∀ x ∈ values, x < max1 → x <= max2) :=
by
  sorry

end max_and_next_max_values_l560_560735


namespace extreme_value_at_1_over_a_l560_560527

variable {a b : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem extreme_value_at_1_over_a (h : ∀ x : ℝ, deriv f x = 2 * a * x + b)
  (hx : deriv f (1/a) = 0) : b = -2 :=
sorry

end extreme_value_at_1_over_a_l560_560527


namespace find_positive_integers_n_satisfying_equation_l560_560732

theorem find_positive_integers_n_satisfying_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  (x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) →
  (n = 1 ∨ n = 3) :=
by
  sorry

end find_positive_integers_n_satisfying_equation_l560_560732


namespace ball_radius_l560_560597

theorem ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) (r : ℝ) :
    hole_diameter = 30 ∧ hole_depth = 10 →
    r = (sqrt ((30 / 2) ^ 2 + hole_depth^2) - hole_depth) + hole_depth :=
begin
  intros h,
  have h_diam := h.1,
  have h_depth := h.2,
  sorry
end

end ball_radius_l560_560597


namespace number_of_valid_permutations_l560_560738

open Function

def is_permutation (l : List ℕ) : Prop :=
  l.perm (List.range (List.length l)) ∧ l.nodup

def valid_permutation (p : List ℕ) : Prop :=
  p.perm [1, 2, 3, 4, 5, 6] ∧ ∀ k, 1 ≤ k ∧ k ≤ 5 → ¬is_permutation (p.take k)

theorem number_of_valid_permutations : (List.permutations [1, 2, 3, 4, 5, 6]).countp valid_permutation = 461 :=
  sorry

end number_of_valid_permutations_l560_560738


namespace hares_per_rabbit_l560_560273

theorem hares_per_rabbit (H : ℕ) :
  (let num_dogs := 1,
       num_cats := 4,
       num_rabbits := 2 * num_cats,
       num_hares := H * num_rabbits,
       total_animals := num_dogs + num_cats + num_rabbits + num_hares
   in total_animals = 37) → H = 3 := 
by
  sorry

end hares_per_rabbit_l560_560273


namespace length_of_long_axis_l560_560389

variable (b a : ℝ)
variable (e : ℝ := 4 / 5)

def is_short_axis (s_axis : ℝ) : Prop := s_axis = 2 * b
def is_eccentricity_of_ellipse (e : ℝ) : Prop := e = Real.sqrt (1 - b^2 / a^2)
def is_long_axis (l_axis : ℝ) : Prop := l_axis = 2 * a

theorem length_of_long_axis (b_val : 3) (a_val : 5) (h_short_axis : is_short_axis 6) (h_ecc : is_eccentricity_of_ellipse e) : 
  is_long_axis 10 := by
  sorry

end length_of_long_axis_l560_560389


namespace juice_drank_is_correct_l560_560361

theorem juice_drank_is_correct :
  ∃ x : ℝ,
  x > 1 ∧
  (let final_juice := (1 - 1 / x)^3 * x / x^2 in
  let final_water := x - final_juice in
  final_water = final_juice + 1.5 ∧
  x - final_juice = 1.75) :=
sorry

end juice_drank_is_correct_l560_560361


namespace integral_a_integral_b_integral_c_l560_560637

-- Problem (a)
theorem integral_a (f : ℝ → ℝ → ℝ) (x y: ℝ) :
  (∫ x in 0..2, ∫ y in 0..3, (x^2 + 2*x*y) dy) = 26 :=
sorry

-- Problem (b)
theorem integral_b (f : ℝ → ℝ → ℝ) (x y: ℝ) :
  (∫ y in -2..0, ∫ x in 0..(y^2), (x + 2*y) dx) = -24 / 5 :=
sorry

-- Problem (c)
theorem integral_c (f : ℝ → ℝ → ℝ) (x y: ℝ) :
  (∫ x in 0..5, ∫ y in 0..(5 - x), sqrt(4 + x + y) dy) = 33.73 :=
sorry

end integral_a_integral_b_integral_c_l560_560637


namespace roots_quadratic_expression_l560_560784

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + m - 2023 = 0) (h2 : n^2 + n - 2023 = 0) :
  m^2 + 2 * m + n = 2022 :=
by
  -- proof steps would go here
  sorry

end roots_quadratic_expression_l560_560784


namespace verify_sum_of_fourth_powers_l560_560230

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_fourth_powers (n : ℕ) : ℕ :=
  ((n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30)

noncomputable def square_of_sum (n : ℕ) : ℕ :=
  (n * (n + 1) / 2)^2

theorem verify_sum_of_fourth_powers (n : ℕ) :
  5 * sum_of_fourth_powers n = (4 * n + 2) * square_of_sum n - sum_of_squares n := 
  sorry

end verify_sum_of_fourth_powers_l560_560230


namespace expand_expression_l560_560717

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560717


namespace alcohol_mixture_l560_560942

theorem alcohol_mixture (y : ℕ) :
  let x_vol := 200 -- milliliters
  let y_conc := 30 / 100 -- 30% alcohol
  let x_conc := 10 / 100 -- 10% alcohol
  let final_conc := 20 / 100 -- 20% target alcohol concentration
  let x_alcohol := x_vol * x_conc -- alcohol in x
  (x_alcohol + y * y_conc) / (x_vol + y) = final_conc ↔ y = 200 :=
by 
  sorry

end alcohol_mixture_l560_560942


namespace lara_total_space_larger_by_1500_square_feet_l560_560454

theorem lara_total_space_larger_by_1500_square_feet :
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  total_area - area_square = 1500 :=
by
  -- Definitions
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  
  -- Calculation
  have h_area_rect : area_rect = 1500 := by
    norm_num [area_rect, length_rect, width_rect]

  have h_area_square : area_square = 2500 := by
    norm_num [area_square, side_square]

  have h_total_area : total_area = 4000 := by
    norm_num [total_area, h_area_rect, h_area_square]

  -- Final comparison
  have h_difference : total_area - area_square = 1500 := by
    norm_num [total_area, area_square, h_area_square]

  exact h_difference

end lara_total_space_larger_by_1500_square_feet_l560_560454


namespace area_excluding_hole_correct_l560_560276

def large_rectangle_area (x: ℝ) : ℝ :=
  4 * (x + 7) * (x + 5)

def hole_area (x: ℝ) : ℝ :=
  9 * (2 * x - 3) * (x - 2)

def area_excluding_hole (x: ℝ) : ℝ :=
  large_rectangle_area x - hole_area x

theorem area_excluding_hole_correct (x: ℝ) :
  area_excluding_hole x = -14 * x^2 + 111 * x + 86 :=
by
  -- The proof is omitted
  sorry

end area_excluding_hole_correct_l560_560276


namespace find_h_l560_560822

theorem find_h (h : ℝ) : (sqrt 3) / (2 * sqrt 7 - sqrt 3) = (2 * sqrt 21 + h) / 25 → h = 3 :=
sorry

end find_h_l560_560822


namespace angle_MKB_l560_560079

variables (α : Real) -- Angle BAC

-- Definitions of points and properties in triangle ABC
variables {A B C M K : EuclideanGeometry.Point}

-- Conditions:
-- 1. ∠BAC = α
axiom angle_BAC_eq_alpha : EuclideanGeometry.angle A B C = α

-- 2. M is the projection of B onto the angle bisector of ∠C
axiom projection_M : EuclideanGeometry.is_projection B (EuclideanGeometry.angle_bisector C) M

-- 3. K is the point where the incircle touches side BC
axiom incircle_touching_point_K : EuclideanGeometry.is_incircle_touching_point K B C

-- Theorem: ∠MKB = (π - α) / 2
theorem angle_MKB (α : Real) 
[EuclideanGeometry.angle A B C = α]
[EuclideanGeometry.is_projection B (EuclideanGeometry.angle_bisector C) M]
[EuclideanGeometry.is_incircle_touching_point K B C] :
EuclideanGeometry.angle M K B = (Real.pi - α) / 2 := 
sorry

end angle_MKB_l560_560079


namespace expand_expression_l560_560730

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560730


namespace speed_of_bus_l560_560266

def distance : ℝ := 500.04
def time : ℝ := 20.0
def conversion_factor : ℝ := 3.6

theorem speed_of_bus :
  (distance / time) * conversion_factor = 90.0072 := 
sorry

end speed_of_bus_l560_560266


namespace nth_partial_sum_b_l560_560375

-- Definitions according to the conditions
def seq_a (n : ℕ) : ℕ := if n = 1 then 1 else 2^(n-1)

def seq_s (n : ℕ) : ℕ := 2 * (seq_a n) - 1

noncomputable def seq_b : ℕ → ℕ
| 1       := 3
| (k + 1) := seq_a k + seq_b k

-- Theorem statement according to the question and correct answer
theorem nth_partial_sum_b (n : ℕ) : (finset.range(n + 1)).sum seq_b = 2^n + 2 * n - 1 :=
sorry

end nth_partial_sum_b_l560_560375


namespace circles_intersect_circles_intersect_proof_l560_560332

def circle_standard_form (a b c d : ℝ) : ℝ × ℝ × ℝ :=
  let h := -a / 2
  let k := -b / 2
  let r := real.sqrt((h * h) + (k * k) - c)
  (h, k, r)

def distance_center (center₁ center₂ : ℝ × ℝ) : ℝ :=
  real.sqrt (((center₁.1 - center₂.1) ^ 2) + ((center₁.2 - center₂.2) ^ 2))

theorem circles_intersect (k1 k2 : ℝ) : Prop :=
  sorry

theorem circles_intersect_proof :
  let centerO1 := circle_standard_form 6 0 -7
  let centerO2 := circle_standard_form 0 6 -27
  let r1 := centerO1.2
  let r2 := centerO2.2
  let dist_centers := distance_center (centerO1.1, centerO1.2) (centerO2.1, centerO2.2)
  (r1 - r2 < dist_centers ∧ dist_centers < r1 + r2) →
  circles_intersect r1 r2 :=
by {
  intro h,
  have h1: (circle_standard_form 6 0 -7).2 = 4,
  { sorry },
  have h2: (circle_standard_form 0 6 -27).2 = 6,
  { sorry },
  have h3: distance_center ((circle_standard_form 6 0 -7).1, (circle_standard_form 6 0 -7).2)
    ((circle_standard_form 0 6 -27).1, (circle_standard_form 0 6 -27).2) = 3 * real.sqrt 2,
  { sorry },
  sorry
}

end circles_intersect_circles_intersect_proof_l560_560332


namespace parabola_equation_focus_l560_560214

theorem parabola_equation_focus (p : ℝ) (h₀ : p > 0)
  (h₁ : (p / 2 = 2)) : (y^2 = 2 * p * x) :=
  sorry

end parabola_equation_focus_l560_560214


namespace setB_can_form_triangle_l560_560245

-- Define the lengths of the sets
def setA := (1 : ℝ, 2 : ℝ, 4 : ℝ)
def setB := (4 : ℝ, 6 : ℝ, 8 : ℝ)
def setC := (5 : ℝ, 6 : ℝ, 12 : ℝ)
def setD := (2 : ℝ, 3 : ℝ, 5 : ℝ)

-- Define the triangle inequality theorem
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Lean 4 statement to prove set B can form a triangle
theorem setB_can_form_triangle : satisfies_triangle_inequality 4 6 8 :=
by 
  admit -- or sorry

end setB_can_form_triangle_l560_560245


namespace integer_count_in_interval_l560_560815

theorem integer_count_in_interval : 
  let π : ℝ := Real.pi in
  let n_set : Set ℤ := {n : ℤ | -8 * π ≤ (n : ℝ) ∧ (n : ℝ) ≤ 10 * π} in
  n_set.card = 57 := 
by
  sorry

end integer_count_in_interval_l560_560815


namespace train_crossing_signal_pole_time_l560_560595

theorem train_crossing_signal_pole_time :
  ∀ (length_train length_platform total_time speed time_to_cross_signal_pole : ℝ),
    length_train = 300 →
    length_platform = 550.0000000000001 →
    total_time = 51 →
    speed = (length_train + length_platform) / total_time →
    time_to_cross_signal_pole = length_train / speed →
    time_to_cross_signal_pole ≈ 18 :=
by
  intros length_train length_platform total_time speed time_to_cross_signal_pole
  intros h1 h2 h3 h4 h5
  sorry

end train_crossing_signal_pole_time_l560_560595


namespace distinct_real_roots_of_quadratic_l560_560503

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end distinct_real_roots_of_quadratic_l560_560503


namespace berries_initial_state_l560_560173

noncomputable def initialBerries (initialStacy: ℕ) : ℤ :=
  let halfStacy := initialStacy / 2
  let initialSteve := halfStacy - 7.5
  initialSteve

noncomputable def finalSteveBerries (initialStacy: ℕ) (berriesTakenBySteve: ℕ) : ℤ :=
  let initialSteve := initialBerries initialStacy
  initialSteve + berriesTakenBySteve

noncomputable def initialAmandaBerries (initialStacy: ℕ) (berriesTakenBySteve berriesTakenByAmanda: ℕ) : ℤ :=
  let berriesLeftWithSteve := finalSteveBerries initialStacy berriesTakenBySteve - berriesTakenByAmanda
  berriesLeftWithSteve - 5.75

theorem berries_initial_state :
  let initialStacy: ℕ := 32
  let berriesTakenBySteve: ℕ := 4
  let berriesTakenByAmanda: ℕ := 3.25
  initialBerries initialStacy = 8.5 ∧
  initialAmandaBerries initialStacy berriesTakenBySteve berriesTakenByAmanda = 3.5 :=
by {
  let initialStacy := 32
  let berriesTakenBySteve := 4
  let berriesTakenByAmanda := 3.25
  have h1: initialBerries initialStacy = 8.5 := sorry,
  have h2: initialAmandaBerries initialStacy berriesTakenBySteve berriesTakenByAmanda = 3.5 := sorry,
  exact ⟨h1, h2⟩
}

end berries_initial_state_l560_560173


namespace expand_expression_l560_560679

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560679


namespace find_solution_pairs_l560_560013

theorem find_solution_pairs (m n : ℕ) (t : ℕ) (ht : t > 0) (hcond : 2 ≤ m ∧ 2 ≤ n ∧ n ∣ (1 + m^(3^n) + m^(2 * 3^n))) : 
  ∃ t : ℕ, t > 0 ∧ m = 3 * t - 2 ∧ n = 3 :=
by sorry

end find_solution_pairs_l560_560013


namespace car_speed_l560_560267

theorem car_speed (v : ℝ) : 
  (∀ v > 0, let time200 := 3600 / 200 in let timeV := 3600 / v in 
  time200 = timeV + 2) → 
  v = 225 :=
by sorry

end car_speed_l560_560267


namespace horse_distribution_l560_560607

variable (b₁ b₂ b₃ : ℕ) 
variable (a : Matrix (Fin 3) (Fin 3) ℝ)
variable (h1 : a 0 0 > a 0 1 ∧ a 0 0 > a 0 2)
variable (h2 : a 1 1 > a 1 0 ∧ a 1 1 > a 1 2)
variable (h3 : a 2 2 > a 2 0 ∧ a 2 2 > a 2 1)

theorem horse_distribution :
  ∃ n : ℕ, ∀ (b₁ b₂ b₃ : ℕ), min b₁ (min b₂ b₃) > n → 
  ∃ (x1 y1 x2 y2 x3 y3 : ℕ), 3*x1 + y1 = b₁ ∧ 3*x2 + y2 = b₂ ∧ 3*x3 + y3 = b₃ ∧
  y1*a 0 0 > y2*a 0 1 ∧ y1*a 0 0 > y3*a 0 2 ∧
  y2*a 1 1 > y1*a 1 0 ∧ y2*a 1 1 > y3*a 1 2 ∧
  y3*a 2 2 > y1*a 2 0 ∧ y3*a 2 2 > y2*a 2 1 :=
sorry

end horse_distribution_l560_560607


namespace cost_per_meat_with_rush_delivery_l560_560749

noncomputable def initial_cost : ℝ := 40.00
noncomputable def rush_delivery_percentage : ℝ := 0.30
noncomputable def total_cost : ℝ := initial_cost + initial_cost * rush_delivery_percentage
noncomputable def number_of_meats : ℝ := 4
noncomputable def cost_per_meat : ℝ := total_cost / number_of_meats

theorem cost_per_meat_with_rush_delivery :
  cost_per_meat = 13.00 :=
by
  unfold initial_cost rush_delivery_percentage total_cost number_of_meats
  -- The proof steps would go here
  sorry

end cost_per_meat_with_rush_delivery_l560_560749


namespace train_length_l560_560093

theorem train_length 
  (speed_kmph : ℕ) 
  (time_sec : ℕ) 
  (speed_converted : speed_kmph * 1000 / 3600 = 5) 
  (time_value : time_sec = 5) : 
  speed_kmph = 18 → 
  ∃ length_m : ℕ, length_m = 25 :=
by
  intros hspeed
  use 25
  simp [time_value, hspeed, speed_converted]
  sorry

end train_length_l560_560093


namespace exist_epsilon_for_polynomial_approx_l560_560143

-- Definitions from conditions
def is_nonconstant {X : Type} [linear_ordered_field X] {Y : Type} (f : X → Y): Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂

variables {f : ℝ → ℂ} (hf : is_nonconstant f) 

-- Main theorem
theorem exist_epsilon_for_polynomial_approx : 
  ∃ ε > 0, ∀ P : polynomial ℂ, ∃ z : ℂ, |z| ≤ 1 ∧ |f(|z|) - P.eval z| ≥ ε :=
sorry

end exist_epsilon_for_polynomial_approx_l560_560143


namespace billy_piles_of_dimes_l560_560635

theorem billy_piles_of_dimes (num_quarter_piles num_coins_per_pile total_coins : ℕ) :
    num_quarter_piles = 2 →
    num_coins_per_pile = 4 →
    total_coins = 20 →
    (total_coins - num_quarter_piles * num_coins_per_pile) / num_coins_per_pile = 3 :=
by
  intros h_quarter_piles h_coins_per_pile h_total_coins
  rw [h_quarter_piles, h_coins_per_pile, h_total_coins]
  norm_num
  exact rfl

end billy_piles_of_dimes_l560_560635


namespace flowers_on_porch_l560_560912

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l560_560912


namespace value_of_a_l560_560029

noncomputable def f : ℝ → ℝ := λ x, if x ≤ 0 then 2^x else log x / log 2

theorem value_of_a (a : ℝ) (h : f a + f 1 = 1 / 2) : a = sqrt 2 ∨ a = -1 :=
sorry

end value_of_a_l560_560029


namespace area_of_square_l560_560931

noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

theorem area_of_square
  {E F G H : Type}
  (ABCD : Type)
  (on_segments : E → F → G → H → Prop)
  (EG FH : ℝ)
  (angle_intersection : ℝ)
  (hEG : EG = 7)
  (hFH : FH = 8)
  (hangle : angle_intersection = 30) :
  ∃ s : ℝ, square_area s = 147 / 4 :=
sorry

end area_of_square_l560_560931


namespace rectangle_k_value_l560_560957

theorem rectangle_k_value (a k : ℝ) (h1 : k > 0) (h2 : 2 * (3 * a + a) = k) (h3 : 3 * a^2 = k) : k = 64 / 3 :=
by
  sorry

end rectangle_k_value_l560_560957


namespace minimum_value_function_tangent_line_values_l560_560904

noncomputable def f (a b x : ℝ) : ℝ := a * x + (1 / (a * x)) + b

theorem minimum_value_function (a b : ℝ) (ha : a > 0) : 
  ∃ x₀, (∀ x > 0, f a b x ≥ f a b x₀) ∧ f a b x₀ = b + 2 :=
sorry

theorem tangent_line_values (a b : ℝ)
  (ha : a > 0)
  (H_tangent : tangent_line (f a b) 1 = (λ x, (3 : ℝ) / 2 * x)) :
  a = 2 ∧ b = -1 :=
sorry

end minimum_value_function_tangent_line_values_l560_560904


namespace cover_points_with_circles_l560_560430

theorem cover_points_with_circles : 
  ∀ (points : Fin 100 → ℝ × ℝ), 
  ∃ (circles : List (ℝ × (ℝ × ℝ))), 
    (∀ (c1 c2 ∈ circles), c1 ≠ c2 → (let ⟨r1, (x1, y1)⟩ := c1 in 
    let ⟨r2, (x2, y2)⟩ := c2 in 
    (x1 - x2)^2 + (y1 - y2)^2 > 1)) ∧ 
    (∀ p ∈ points, ∃ c ∈ circles, let ⟨r, (x, y)⟩ := c in 
    (p.1 - x)^2 + (p.2 - y)^2 ≤ r^2) ∧ 
    (List.sum (List.map (fun ⟨r, _⟩ => 2 * r) circles) < 100) := sorry

end cover_points_with_circles_l560_560430


namespace countValidSeqs_length_21_l560_560086

def isValidSeq (seq : List ℕ) : Prop :=
  (seq.head = 0) ∧ 
  (seq.last = 0) ∧ 
  (∀ i, i < seq.length - 1 → ¬ (seq[i] = 0 ∧ seq[i+1] = 0)) ∧ 
  (∀ i, i < seq.length - 2 → ¬ (seq[i] = 1 ∧ seq[i+1] = 1 ∧ seq[i+2] = 1))

def countValidSeqs (n : ℕ) : ℕ :=
  -- Here we would implement the function to count valid sequences
  -- which is abstracted away from the proof statement.
  sorry

theorem countValidSeqs_length_21 : countValidSeqs 21 = 114 :=
  sorry

end countValidSeqs_length_21_l560_560086


namespace power_function_convex_upwards_l560_560357

theorem power_function_convex_upwards (f : ℝ → ℝ) (x1 x2 : ℝ) 
  (hx1 : 0 < x1) (hx2 : x1 < x2) :
  f(x) = x^(1/5) → 
  f((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end power_function_convex_upwards_l560_560357


namespace find_angle_x_l560_560437

theorem find_angle_x (A B C : Type) (angle_ABC angle_CAB x : ℝ) 
  (h1 : angle_ABC = 40) 
  (h2 : angle_CAB = 120)
  (triangle_sum : x + angle_ABC + (180 - angle_CAB) = 180) : 
  x = 80 :=
by 
  -- actual proof goes here
  sorry

end find_angle_x_l560_560437


namespace count_of_n_not_dividing_g_l560_560135

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d, d > 0 ∧ n % d = 0)

def g (n : ℕ) : ℕ :=
  (proper_divisors n).prod id

def n_not_divide_g (n : ℕ) : Prop :=
  ¬ n ∣ g n

def count_not_divide_g_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp (λ n, n_not_divide_g n)

theorem count_of_n_not_dividing_g {a b : ℕ} (h1 : a = 3) (h2 : b = 100) :
  count_not_divide_g_in_range a b = 29 := by
  rw [h1, h2]
  -- The proof will proceed from here.
  sorry

end count_of_n_not_dividing_g_l560_560135


namespace reduce_fraction_l560_560252

-- Defining a structure for a fraction
structure Fraction where
  num : ℕ
  denom : ℕ
  deriving Repr

-- The original fraction
def originalFraction : Fraction :=
  { num := 368, denom := 598 }

-- The reduced fraction
def reducedFraction : Fraction :=
  { num := 184, denom := 299 }

-- The statement of our theorem
theorem reduce_fraction :
  ∃ (d : ℕ), d > 0 ∧ (originalFraction.num / d = reducedFraction.num) ∧ (originalFraction.denom / d = reducedFraction.denom) := by
  sorry

end reduce_fraction_l560_560252


namespace divide_numbers_into_quadruples_l560_560958

theorem divide_numbers_into_quadruples :
  ∀ (R Y G : Fin 51 → ℕ) 
    (red yellow green : Finset ℕ),
    (red.card = 50) → 
    (yellow.card = 25) → 
    (green.card = 25) →
    (∀ k : Fin 25, ∃ i j : Fin 50, i < j ∧ k < 25 ∧ Y k = R i ∧ R j) →
    (∀ k : Fin 25, ∃ i j : Fin 50, i < j ∧ k < 25 ∧ G k = R i ∧ R j) →
    (∃ Q : Fin 25 → Fin 4 → ℕ,
      ∀ k : Fin 25, ∃ (i j : Fin 50) (y g : Fin 25), 
        Q k 0 = R i ∧ Q k 1 = R j ∧ Q k 2 = Y y ∧ Q k 3 = G g ∧
        R i < Y y ∧ Y y < R j ∧ 
        R i < G g ∧ G g < R j) :=
begin
  sorry
end

end divide_numbers_into_quadruples_l560_560958


namespace prime_pairs_difference_of_squares_l560_560014

-- Definition of primality
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

-- Main statement: 
theorem prime_pairs_difference_of_squares (p q : ℕ) 
  (hp : is_prime p) (hq : is_prime q) 
  (h : is_prime (p^2 - q^2)) : (p, q) = (3, 2) :=
by sorry

end prime_pairs_difference_of_squares_l560_560014


namespace region_R_area_l560_560907

-- Problem statement
def A := (-36 : ℝ, 0 : ℝ)
def B := (36 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 0 : ℝ)
def D := (0 : ℝ, 30 : ℝ)
def neg_D := (0 : ℝ, -30 : ℝ)

def X (x : ℝ) := (x, 0)
def Y (y : ℝ) := (0, y)

def midpoint (X Y : ℝ × ℝ) : ℝ × ℝ :=
(X.1 / 2 + Y.1 / 2, X.2 / 2 + Y.2 / 2)

def R (x y : ℝ) (hx : -36 ≤ x ∧ x ≤ 36) (hy : -30 ≤ y ∧ y ≤ 30) : (ℝ × ℝ) :=
midpoint (X x) (Y y)

noncomputable def area_R : ℝ :=
36 * 30

theorem region_R_area : area_R = 1080 := by
  -- The proof goes here
  sorry

end region_R_area_l560_560907


namespace greatest_number_of_balls_l560_560995

theorem greatest_number_of_balls (r : ℝ) (l w h : ℝ) (d : ℝ) :
  r = 1 / 2 ∧ l = 10 ∧ w = 10 ∧ h = 1 ∧ d = 1 →
  ∃ n : ℕ, n = 100 :=
by
  assume hconds : r = 1 / 2 ∧ l = 10 ∧ w = 10 ∧ h = 1 ∧ d = 1
  have n : ℕ := 100
  exact ⟨n, by sorry⟩

end greatest_number_of_balls_l560_560995


namespace expand_polynomial_l560_560686

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l560_560686


namespace ellipse_formula_line_through_N_l560_560392

-- Define the necessary variables and constants
variables {a b : ℝ}
variables {k₁ k₂ : ℝ}
variables {x y : ℝ}

-- Conditions from the problem
def ellipse_eq (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1
def k_sum := k₁ + k₂ = 4
def line_eq := ∀ x y: ℝ, x - y + sqrt 2 = 0

-- Correct Answers to be proved
theorem ellipse_formula : ∃ (a b : ℝ), a = sqrt 2 ∧ b = 1 :=
by 
  use [sqrt 2, 1]
  sorry

theorem line_through_N : ∃ x y : ℝ, (x + 1 / 2) / ((1 + 2 * k₁^2) / 2 + 4 * k₁ / 2) = (y + 1) / (2 / (1 + 2 * k₁^2) ∧ (x + 1 / 2) / ((1 + 2 * k₂^2) / 2 + 4 * k₂ / 2) = (y + 1) / (2 / (1 + 2 * k₂^2)) :=
by
  have h₁ : k₁ + k₂ = 4 := k_sum
  sorry

end ellipse_formula_line_through_N_l560_560392


namespace no_such_P_exists_l560_560472

def τ (n : ℕ) : ℕ := (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card

noncomputable def a (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  if P n > 0 then Nat.gcd (Int.toNat (P n)) (τ (Int.toNat (P n))) else 0

def hasLimitInfinity (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ N : ℕ, ∀ n ≥ N, a n ≠ k

theorem no_such_P_exists : ¬ ∃ P : Polynomial ℤ, hasLimitInfinity (a P) := 
sorry

end no_such_P_exists_l560_560472


namespace count_valid_numbers_l560_560992

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_valid_number (digits : List ℕ) : Prop :=
  digits.length = 5 ∧ 
  (digits.toFinset = {1, 2, 3, 4, 5}) ∧ 
  (∃ i, (0 < i ∧ i < 4) ∧ is_even (digits[i-1]) ∧ is_odd (digits[i]) ∧ is_even (digits[i+1]) ∧ 
  ∀ j, (j ≠ i - 1 ∧ j ≠ i ∧ j ≠ i + 1) → is_odd (digits[j]))

theorem count_valid_numbers : {n : List ℕ // is_valid_number n}.card = 36 := 
  sorry

end count_valid_numbers_l560_560992


namespace num_presses_to_exceed_2000_l560_560608

noncomputable def f (x : Nat) : Nat := x * x - 3

theorem num_presses_to_exceed_2000 : 
  ∃ n : Nat, (nat.iterate f n 4) > 2000 ∧ n = 3 := 
by
  sorry

end num_presses_to_exceed_2000_l560_560608


namespace sum_of_digits_l560_560094

theorem sum_of_digits (a b c d : ℕ) (h1 : a + c = 11) (h2 : b + c = 9) (h3 : a + d = 10) (h_d : d - c = 1) : 
  a + b + c + d = 21 :=
sorry

end sum_of_digits_l560_560094


namespace ticket_costs_l560_560990

-- Define the conditions
def cost_per_ticket : ℕ := 44
def number_of_tickets : ℕ := 7

-- Define the total cost calculation
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Prove that given the conditions, the total cost is 308
theorem ticket_costs :
  total_cost = 308 :=
by
  -- Proof steps here
  sorry

end ticket_costs_l560_560990


namespace sequence_is_integer_three_divides_sequence_l560_560497

noncomputable def a_n (n : ℕ) : ℝ := ((2 + real.sqrt 3)^n - (2 - real.sqrt 3)^n) / (2 * real.sqrt 3)

theorem sequence_is_integer (n : ℕ) : ∃ k : ℤ, a_n n = k := 
sorry

theorem three_divides_sequence (n : ℕ) : 3 ∣ a_n n ↔ 3 ∣ n := 
sorry

end sequence_is_integer_three_divides_sequence_l560_560497


namespace smallest_n_l560_560892

noncomputable def f (x : ℝ) : ℝ := 
  let frac_x := x - real.floor x in
  abs (frac_x^2 - 2 * frac_x + 1.5)

def sufficient_solutions (n : ℝ) : Prop :=
  ∃ xlist : list ℝ, xlist.length ≥ 2023 ∧ ∀ x ∈ xlist, n * f (x * f x) = x

theorem smallest_n (n : ℕ) : n = 45 ↔ ∀ n' : ℕ, sufficient_solutions n' → n' ≥ 45 := 
sorry

end smallest_n_l560_560892


namespace proj_onto_v_is_correct_l560_560018

open Matrix

def proj_matrix (v : Vector ℝ 2) : Matrix (Fin 2) (Fin 2) ℝ :=
  let a := v 0
  let b := v 1
  let denom := (a^2 + b^2 : ℝ)
  fun i j => (v i * v j) / denom

theorem proj_onto_v_is_correct :
  proj_matrix ![1, 2] = !![1/5, 2/5; 2/5, 4/5] :=
by
  sorry

end proj_onto_v_is_correct_l560_560018


namespace circumscribed_sphere_surface_area_l560_560374

theorem circumscribed_sphere_surface_area 
  (side_len_6sqrt3 : ℝ) 
  (base_len_6 : ℝ) 
  (h_side_len : side_len_6sqrt3 = 6 * Real.sqrt 3) 
  (h_base_len : base_len_6 = 6) 
  : 
  ∃ (area : ℝ), area = 243 * π / 3 :=
by 
  use 243 * π / 3 
  sorry

end circumscribed_sphere_surface_area_l560_560374


namespace simplify_and_evaluate_expression_l560_560169

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (\frac{x^2 - 1}{x^2 - 2x + 1} + \frac{1}{1 - x}) ÷ \frac{x^2}{x - 1} = -1 :=
by
  sorry

end simplify_and_evaluate_expression_l560_560169


namespace fractional_part_lawn_remains_l560_560478

-- Define the conditions as constants
constant Mary_hours : ℕ := 3
constant Tom_hours : ℕ := 6
constant together_hours : ℕ := 1
constant Mary_alone_hours : ℕ := 1

-- Define the proof problem as follows
theorem fractional_part_lawn_remains :
  (1 - (together_hours * (1 / Mary_hours + 1 / Tom_hours) + Mary_alone_hours * 1 / Mary_hours)) = 1 / 6 :=
by
  sorry

end fractional_part_lawn_remains_l560_560478


namespace convert_to_rectangular_form_l560_560321

noncomputable def θ : ℝ := 15 * Real.pi / 2

noncomputable def EulerFormula (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

theorem convert_to_rectangular_form : EulerFormula θ = Complex.I := by
  sorry

end convert_to_rectangular_form_l560_560321


namespace cos_sum_l560_560314

theorem cos_sum :
  cos (24 * Real.pi / 180) + cos (144 * Real.pi / 180) + cos (264 * Real.pi / 180) =
  (3 - Real.sqrt 5) / 4 - Real.sin (3 * Real.pi / 180) * Real.sqrt (10 + 2 * Real.sqrt 5) := 
sorry

end cos_sum_l560_560314


namespace triangle_hypotenuse_l560_560864

-- Given conditions
variables (ML LN : ℝ)
variables (M N L : Type) -- Vertices of the triangle
variable (sin_N : ℝ)
hypothesis1 : ML = 10
hypothesis2 : sin_N = 5 / 13

-- We aim to prove LN = 26 under these conditions.
theorem triangle_hypotenuse : LN = 26 :=
by
  sorry

end triangle_hypotenuse_l560_560864


namespace fourth_pedal_triangle_congruence_l560_560182

noncomputable def original_triangle_angles (α β γ : ℝ) :=
  α / 12 = 1 ∧ β / 12 = 4 ∧ γ / 12 = 10 ∧ α + β + γ = 180

noncomputable def second_pedal_triangle_angles (α₀ β₀ γ₀ α₂ β₂ γ₂ : ℝ) :=
  (α₂ = 2 * (2 * α₀) ∧ β₂ = 2 * (2 * β₀) - 180 ∧ γ₂ = 2 * (2 * γ₀))

noncomputable def fourth_pedal_triangle_same_position (α₀ β₀ γ₀ α₄ β₄ γ₄ : ℝ) :=
  α₄ = α₀ ∧ β₄ = β₀ ∧ γ₄ = γ₀

theorem fourth_pedal_triangle_congruence :
  ∀ (α₀ β₀ γ₀ α₂ β₂ γ₂ α₄ β₄ γ₄ : ℝ),
    original_triangle_angles α₀ β₀ γ₀ →
    second_pedal_triangle_angles α₀ β₀ γ₀ α₂ β₂ γ₂ →
    fourth_pedal_triangle_same_position α₀ β₀ γ₀ α₄ β₄ γ₄ :=
by
  intros α₀ β₀ γ₀ α₂ β₂ γ₂ α₄ β₄ γ₄ h₀ h₁ h₂
  sorry

end fourth_pedal_triangle_congruence_l560_560182


namespace length_ratio_is_correct_width_ratio_is_correct_l560_560285

-- Definitions based on the conditions
def room_length : ℕ := 25
def room_width : ℕ := 15

-- Calculated perimeter
def room_perimeter : ℕ := 2 * (room_length + room_width)

-- Ratios to be proven
def length_to_perimeter_ratio : ℚ := room_length / room_perimeter
def width_to_perimeter_ratio : ℚ := room_width / room_perimeter

-- Stating the theorems to be proved
theorem length_ratio_is_correct : length_to_perimeter_ratio = 5 / 16 :=
by sorry

theorem width_ratio_is_correct : width_to_perimeter_ratio = 3 / 16 :=
by sorry

end length_ratio_is_correct_width_ratio_is_correct_l560_560285


namespace Brian_age_in_eight_years_l560_560313

-- Definitions based on conditions
variable {Christian Brian : ℕ}
variable (h1 : Christian = 2 * Brian)
variable (h2 : Christian + 8 = 72)

-- Target statement to prove Brian's age in eight years
theorem Brian_age_in_eight_years : (Brian + 8) = 40 :=
by 
  sorry

end Brian_age_in_eight_years_l560_560313


namespace equation_has_one_real_root_l560_560192

noncomputable def f (x : ℝ) : ℝ :=
  (3 / 11)^x + (5 / 11)^x + (7 / 11)^x - 1

theorem equation_has_one_real_root :
  ∃! x : ℝ, f x = 0 := sorry

end equation_has_one_real_root_l560_560192


namespace eval_complex_expression_l560_560005

theorem eval_complex_expression (i : ℂ) (h1 : i^2 = -1) :
  2 * i^45 + 3 * i^123 = -i :=
by
  -- Definitions and conditions to use in the proof
  have h_period : ∀ n, i^(4 * n) = 1,
  {
    intro n,
    induction n with n hn,
    { simp },
    { rw [Nat.succ_eq_add_one, pow_add, pow_mul],
      simp [hn] }
  },
  have h_mod : ∀ n, n % 4 = 1 → i^n = i,
  {
    intros n hn_mod,
    exact calc
      i^n = i^(4 * (n / 4) + 1) : by rw [Nat.div_add_mod]
      ... = i^(4 * (n / 4)) * i : by rw [pow_add]
      ... = 1 * i : by rw [h_period]
      ... = i : by simp
  },
  have h_mod_45 : 45 % 4 = 1 := by norm_num,
  have h_mod_123 : 123 % 4 = 3 := by norm_num,
  have h_exp_45 : i^45 = i, from h_mod 45 h_mod_45,
  have h_exp_123 : i^123 = -i,
  {
    calc
      i^123 = i^(4 * 30 + 3) : by exact (Nat.div_add_mod 123 4).symm
      ... = (i^4)^30 * i^3 : by rw [pow_add]
      ... = 1^30 * -i : by rw [h_period, pow_succ, pow_two, h1]
      ... = -i : by simp
  },
  calc
    2 * i^45 + 3 * i^123 = 2 * i + 3 * -i : by rw [h_exp_45, h_exp_123]
    ... = 2 * i - 3 * i : by ring
    ... = -i : by ring

end eval_complex_expression_l560_560005


namespace geometric_sequence_general_term_formula_l560_560805

def sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (5 * (sequence n) - 8) / ((sequence n) - 1)

theorem geometric_sequence (a : ℝ) (n : ℕ) (h : a = 3) :
  ∃ r : ℝ, ∀ n : ℕ, (sequence a (n + 1) - 2) / (sequence a (n + 1) - 4) = r * (sequence a n - 2) / (sequence a n - 4) :=
sorry

theorem general_term_formula (a : ℝ) (h : a = 3) :
  ∀ n : ℕ, sequence a n = (4 * 3 ^ n + 2) / (3 ^ n + 1) :=
sorry

end geometric_sequence_general_term_formula_l560_560805


namespace inequality_always_true_l560_560755

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l560_560755


namespace inequality_always_true_l560_560756

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l560_560756


namespace num_ways_to_fill_matrix_2x3_l560_560657

open Matrix

-- Define the given set of numbers
def nums : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a 2x3 matrix
def matrix2x3 : Type := Matrix (Fin 2) (Fin 3) ℕ

-- Define the condition that the sum of each row and column is divisible by 3
def sum_divisible_by_3 (m : matrix2x3) : Prop :=
  (∀ i, (∑ j, m i j) % 3 = 0) ∧ (∀ j, (∑ i, m i j) % 3 = 0)

-- The statement of the problem as a theorem in Lean
theorem num_ways_to_fill_matrix_2x3 : 
  ∃ M : Finset matrix2x3, (∀ m ∈ M, (∀ i, ∑ j, m i j % 3 = 0) ∧ (∀ j, ∑ i, m i j % 3 = 0)) ∧ M.card = 48 :=
sorry

end num_ways_to_fill_matrix_2x3_l560_560657


namespace ratio_eleven_to_fifteen_rounded_l560_560161

theorem ratio_eleven_to_fifteen_rounded :
  Float.to_digits 10 ((11:Float) / 15) = (0.7 : Float) :=
by
  -- Sorry is used here to skip the proof, which would include necessary steps
  sorry

end ratio_eleven_to_fifteen_rounded_l560_560161


namespace f_ge_1_solution_set_f_le_g_l560_560800

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (1 - x)
noncomputable def g (x : ℝ) (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) : ℝ := 
  abs (x + a^2) + abs (x - b^2)

theorem f_ge_1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | 1 / 2 ≤ x} :=
begin
  sorry
end

theorem f_le_g (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2):
  ∀ x : ℝ, f x ≤ g x a b ha hb h :=
begin
  sorry
end

end f_ge_1_solution_set_f_le_g_l560_560800


namespace sum_m_n_d_l560_560848

noncomputable def m : ℕ := 250
noncomputable def n : ℕ := 101
noncomputable def d : ℕ := 19

theorem sum_m_n_d :
    let r := 50
    let l := 90
    let P_dist_center := 24
    ∃ m n d : ℕ,
    (m * (Real.pi : ℝ) - n * Real.sqrt d = 
        area_of_region_bordered_by_chords r l P_dist_center) ∧
    Nat.sqrt' d <∧ (Prime.is_square_free d) ∧
    m + n + d = 370 := 
sorry

end sum_m_n_d_l560_560848


namespace hexagon_shaded_area_l560_560613

noncomputable def hexagon_side_length : ℝ := 8
noncomputable def sector_radius : ℝ := 4
noncomputable def sector_angle_degrees : ℝ := 120

theorem hexagon_shaded_area :
  let hexagon_area := 6 * (sqrt 3 / 4 * hexagon_side_length ^ 2)
  let sector_area := (sector_angle_degrees / 360) * π * sector_radius ^ 2
  let total_hexagon_area := hexagon_area
  let total_sector_area := 6 * sector_area
  total_hexagon_area - total_sector_area = 96 * sqrt 3 - 32 * π :=
by
  sorry

end hexagon_shaded_area_l560_560613


namespace part1_solution_part2_solution_l560_560067

def f (x : ℝ) : ℝ := 2 * x^2 - x - 1

theorem part1_solution : {x : ℝ | f x ≤ 1 - x^2} = set.Icc (-2/3) 1 :=
by
  sorry

theorem part2_solution (m : ℝ) : (∀ x ∈ set.Icc (Real.exp 1) (Real.exp 2), f (Real.log x) + 5 > m * Real.log x) ↔ m < 4 * Real.sqrt 2 - 1 :=
by
  sorry

end part1_solution_part2_solution_l560_560067


namespace correct_choice_l560_560771

-- Define propositions p and q
def p : Prop := ((λ (x y : ℝ), x^2 + y^2 - 2 * x + 4 * y - 1 = 0) → (area = 6 * π))
def q : Prop := (∀ (a β B : Type), (a  ⊥ β) ∧ (a ⊆ a) → ¬ a ⊥ B)

-- The statement to be proved
theorem correct_choice : p ∧ ¬ q :=
by sorry

end correct_choice_l560_560771


namespace solve_equation_l560_560345

theorem solve_equation (x : ℝ) (h : x^2 + 6 * x + 6 * x * sqrt (x + 4) = 24) :
    x = (17 - sqrt 241) / 2 := 
  sorry

end solve_equation_l560_560345


namespace non_mobile_payment_probability_40_60_l560_560153

variable (total_customers : ℕ)
variable (num_non_mobile_40_50 : ℕ)
variable (num_non_mobile_50_60 : ℕ)

theorem non_mobile_payment_probability_40_60 
  (h_total_customers: total_customers = 100)
  (h_num_non_mobile_40_50: num_non_mobile_40_50 = 9)
  (h_num_non_mobile_50_60: num_non_mobile_50_60 = 5) : 
  (num_non_mobile_40_50 + num_non_mobile_50_60 : ℚ) / total_customers = 7 / 50 :=
by
  -- Placeholder for the actual proof
  sorry

end non_mobile_payment_probability_40_60_l560_560153


namespace expand_expression_l560_560678

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560678


namespace speed_in_still_water_l560_560605

theorem speed_in_still_water (upstream downstream : ℕ) (h1 : upstream = 26) (h2 : downstream = 40) :
  (upstream + downstream) / 2 = 33 :=
by 
  rw [h1, h2]
  exact rfl

end speed_in_still_water_l560_560605


namespace max_distance_on_spheres_l560_560233

noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((-5 - 7)^2 + (-15 - 12)^2 + (10 - (-20))^2)

theorem max_distance_on_spheres :
  let distance := distance_between_centers
  let max_distance := 23 + distance + 95
  max_distance = 118 + Real.sqrt(1773) :=
by
  -- let O be the center of the first sphere
  -- let P be the center of the second sphere
  -- distance between centers O and P is √(((-5 - 7)^2) + ((-15 - 12)^2) + ((10 - (-20))^2))
  let distance := distance_between_centers
  have h1 : distance = Real.sqrt(1773),
  -- calculation details skipped
  -- hence the maximum distance is 23 + distance + 95
  let max_distance := 23 + distance + 95
  show max_distance = (118 + Real.sqrt(1773))
  exact sorry

end max_distance_on_spheres_l560_560233


namespace modulus_of_complex_l560_560796

open Complex

theorem modulus_of_complex : ∀ (z : ℂ), z = 3 - 2 * I → Complex.abs z = Real.sqrt 13 :=
by
  intro z
  intro h
  rw [h]
  simp [Complex.abs]
  sorry

end modulus_of_complex_l560_560796


namespace zhang_hua_repayment_l560_560564

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end zhang_hua_repayment_l560_560564


namespace a_minus_c_eq_neg_120_l560_560421

variables (a b c d : ℕ)

-- Conditions
def avg_abc_eq_110 (h1 : (a + b + d) / 3 = 110) : Prop := h1
def avg_bcd_eq_150 (h2 : (b + c + d) / 3 = 150) : Prop := h2

-- Theorem to prove
theorem a_minus_c_eq_neg_120 (h1 : (a + b + d) / 3 = 110) (h2 : (b + c + d) / 3 = 150) : 
  a - c = -120 :=
by
  sorry

end a_minus_c_eq_neg_120_l560_560421


namespace perpendicular_condition_l560_560410

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def linear_combination (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def opposite_combination : ℝ × ℝ := (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (linear_combination k) opposite_combination = 0 → k = 19 :=
begin
  sorry
end

end perpendicular_condition_l560_560410


namespace p_6_is_126_l560_560462

noncomputable def p (x : ℝ) : ℝ := sorry

axiom h1 : p 1 = 1
axiom h2 : p 2 = 2
axiom h3 : p 3 = 3
axiom h4 : p 4 = 4
axiom h5 : p 5 = 5

theorem p_6_is_126 : p 6 = 126 := sorry

end p_6_is_126_l560_560462


namespace final_game_deficit_l560_560148

-- Define the points for each scoring action
def free_throw_points := 1
def three_pointer_points := 3
def jump_shot_points := 2
def layup_points := 2
def and_one_points := layup_points + free_throw_points

-- Define the points scored by Liz
def liz_free_throws := 5 * free_throw_points
def liz_three_pointers := 4 * three_pointer_points
def liz_jump_shots := 5 * jump_shot_points
def liz_and_one := and_one_points

def liz_points := liz_free_throws + liz_three_pointers + liz_jump_shots + liz_and_one

-- Define the points scored by Taylor
def taylor_three_pointers := 2 * three_pointer_points
def taylor_jump_shots := 3 * jump_shot_points

def taylor_points := taylor_three_pointers + taylor_jump_shots

-- Define the points for Liz's team
def team_points := liz_points + taylor_points

-- Define the points scored by the opposing team players
def opponent_player1_points := 4 * three_pointer_points

def opponent_player2_jump_shots := 4 * jump_shot_points
def opponent_player2_free_throws := 2 * free_throw_points
def opponent_player2_points := opponent_player2_jump_shots + opponent_player2_free_throws

def opponent_player3_jump_shots := 2 * jump_shot_points
def opponent_player3_three_pointer := 1 * three_pointer_points
def opponent_player3_points := opponent_player3_jump_shots + opponent_player3_three_pointer

-- Define the points for the opposing team
def opponent_team_points := opponent_player1_points + opponent_player2_points + opponent_player3_points

-- Initial deficit
def initial_deficit := 25

-- Final net scoring in the final quarter
def net_quarter_scoring := team_points - opponent_team_points

-- Final deficit
def final_deficit := initial_deficit - net_quarter_scoring

theorem final_game_deficit : final_deficit = 12 := by
  sorry

end final_game_deficit_l560_560148


namespace expression_value_l560_560019

theorem expression_value : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end expression_value_l560_560019


namespace John_needs_13_more_to_buy_pogo_stick_l560_560123

def amount_needed_to_buy_pogo_stick (earnings_sat earnings_sun earnings_prev pogo_cost : ℕ) : ℕ :=
  let total_earnings := earnings_sat + earnings_sun + earnings_prev in
  pogo_cost - total_earnings

theorem John_needs_13_more_to_buy_pogo_stick :
  amount_needed_to_buy_pogo_stick 18 (18 / 2) 20 60 = 13 := by
  sorry

end John_needs_13_more_to_buy_pogo_stick_l560_560123


namespace measure_angle_B_l560_560118

theorem measure_angle_B (A B C : ℝ) (h1 : B = 2*A) (h2 : C = 4*A) (h3 : A + B + C = 180) : B = 360/7 :=
by
  have hA : A = 180 / 7 := by sorry
  exact calc
    B = 2 * A : h1
    ... = 2 * (180 / 7) : by rw hA
    ... = 360 / 7 : by norm_num

end measure_angle_B_l560_560118


namespace proof_2d_minus_r_l560_560417

theorem proof_2d_minus_r (d r: ℕ) (h1 : 1059 % d = r)
  (h2 : 1482 % d = r) (h3 : 2340 % d = r) (hd : d > 1) : 2 * d - r = 6 := 
by 
  sorry

end proof_2d_minus_r_l560_560417


namespace value_of_BH_l560_560117

-- Definitions of the conditions
def triangle_ABC : Type :=
{ A B C : Point // B.dist_to C = 4 ∧ C.dist_to A = 5 ∧ A.dist_to B = 3 ∧ angle C = 90 }

def point_GH (ABC : triangle_ABC) : Type :=
{ G H : Point // ABC.A.dist_to G > 3 ∧ ABC.A.dist_to H > ABC.A.dist_to G }

def intersection_point_I (ABC : triangle_ABC) (GH : point_GH ABC) : Type :=
{ I : Point // I ≠ ABC.C ∧ circumcircle_contains (ABC.A, ABC.C, GH.G) I ∧ circumcircle_contains (ABC.B, GH.H, ABC.C) I ∧ GH.G.dist_to I = 3 ∧ GH.H.dist_to I = 8 }

-- The statement of the theorem
theorem value_of_BH (ABC : triangle_ABC) (GH : point_GH ABC) (I : intersection_point_I ABC GH) : 
  ∃ p q r s : ℕ, p + q + r + s = 107 :=
sorry -- Proof would go here

end value_of_BH_l560_560117


namespace length_of_bridge_l560_560294

def train_length : ℝ := 155 -- meters
def train_speed : ℝ := 45 * (1000 / 3600) -- converting km/hr to m/s
def crossing_time : ℝ := 30 -- seconds
def total_distance : ℝ := train_speed * crossing_time -- total distance covered

theorem length_of_bridge : (total_distance - train_length) = 220 :=
by
  -- skipping proof, add specific proof steps here
  sorry

end length_of_bridge_l560_560294


namespace omega_range_for_monotonicity_l560_560083

theorem omega_range_for_monotonicity {ω : ℝ} (hω : ω > 0) :
  (∀ x ∈ (Set.Ioo (π / 3 : ℝ) (π / 2 : ℝ)), 
      Max (Real.sin (ω * x)) (Real.cos (ω * x)) = Real.sin (ω * x)
      ∧ Max (Real.sin (ω * x)) (Real.cos (ω * x)) = Real.cos (ω * x) → False) ↔ ω > 3 / 2 := 
sorry

end omega_range_for_monotonicity_l560_560083


namespace problem_a_plus_b_l560_560499

structure Pyramid :=
  (P Q R S T : Point ℝ)
  (Q_eq : EuclideanGeometry.orthogonal 𝕜 P Q T)
  (congruent_edges : dist P R = dist P Q ∧ dist P R = dist P S ∧ dist P R = dist P T)
  (angle_RQT : ∠ Q R T = π / 3)

noncomputable def CosPhi (P Q R S T : Point ℝ) :=
  let φ := euclideanAngle (P, Q, R) (P, Q, S)
  Float.cos φ

theorem problem_a_plus_b :
  ∀ (P Q R S T : Point ℝ) 
    (h: Pyramid P Q R S T),
  ∃ a b : ℤ, CosPhi P Q R S T = a + Real.sqrt b ∧ a + b = 0 :=
by
  sorry

end problem_a_plus_b_l560_560499


namespace solution_set_of_inequality_l560_560545

theorem solution_set_of_inequality (x: ℝ) : 
  (1 / x ≤ 1) ↔ (x < 0 ∨ x ≥ 1) :=
sorry

end solution_set_of_inequality_l560_560545


namespace cost_per_meat_with_rush_delivery_l560_560750

noncomputable def initial_cost : ℝ := 40.00
noncomputable def rush_delivery_percentage : ℝ := 0.30
noncomputable def total_cost : ℝ := initial_cost + initial_cost * rush_delivery_percentage
noncomputable def number_of_meats : ℝ := 4
noncomputable def cost_per_meat : ℝ := total_cost / number_of_meats

theorem cost_per_meat_with_rush_delivery :
  cost_per_meat = 13.00 :=
by
  unfold initial_cost rush_delivery_percentage total_cost number_of_meats
  -- The proof steps would go here
  sorry

end cost_per_meat_with_rush_delivery_l560_560750


namespace expand_expression_l560_560723

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l560_560723


namespace sum_has_4_digits_l560_560512

theorem sum_has_4_digits (C D : ℕ) (hC : 1 ≤ C ∧ C ≤ 9) (hD : 1 ≤ D ∧ D ≤ 9) :
  let sum := 7654 + (C * 10 + 7) + (D * 10 + 9) + 81 in
  1000 ≤ sum ∧ sum < 10000 :=
by
  sorry

end sum_has_4_digits_l560_560512


namespace number_of_students_who_chose_fish_l560_560196

theorem number_of_students_who_chose_fish (h : ∀ n, n = 40) : ∃ n, n = 40 :=
by
  use 40
  exact h 40

end number_of_students_who_chose_fish_l560_560196


namespace intersection_A_B_l560_560379

-- Definition of sets A and B based on given conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3 }
def B : Set ℝ := {y | ∃ x : ℝ, x < 0 ∧ y = x + 1 / x }

-- Proving the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {y | -4 ≤ y ∧ y ≤ -2} := 
by
  sorry

end intersection_A_B_l560_560379


namespace expand_expression_l560_560661

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l560_560661


namespace functions_with_two_zeros_l560_560625

theorem functions_with_two_zeros :
  let f1 := λ x : ℝ, real.log x
  let f2 := λ x : ℝ, 2^x
  let f3 := λ x : ℝ, x^2
  let f4 := λ x : ℝ, | x | - 1
  (∃ a b : ℝ, a ≠ b ∧ f4 a = 0 ∧ f4 b = 0) ∧
  ¬(∃ a b : ℝ, a ≠ b ∧ f1 a = 0 ∧ f1 b = 0) ∧
  ¬(∃ a b : ℝ, a ≠ b ∧ f2 a = 0 ∧ f2 b = 0) ∧
  ¬(∃ a b : ℝ, a ≠ b ∧ f3 a = 0 ∧ f3 b = 0) :=
by
  sorry

end functions_with_two_zeros_l560_560625


namespace num_boys_l560_560102

theorem num_boys (total_students : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) (r : girls_ratio = 4) (b : boys_ratio = 3) (o : others_ratio = 2) (total_eq : girls_ratio * k + boys_ratio * k + others_ratio * k = total_students) (total_given : total_students = 63) : 
  boys_ratio * k = 21 :=
by
  sorry

end num_boys_l560_560102


namespace closest_approximation_l560_560339

theorem closest_approximation :
  ∃ (closest : ℝ), closest = 2000 ∧
    ∀ (options : list ℝ), options = [0.2, 2, 20, 200, 2000] →
    (∃ (n d : ℝ), n = 401 ∧ d = 0.205 ∧
      ∀ (approx_n approx_d : ℝ), approx_n = 400 ∧ approx_d = 0.2 →
        closest = 400 / 0.2) :=
begin
  sorry
end

end closest_approximation_l560_560339


namespace intersection_A_B_l560_560378

open Set

variable (A B : Set ℝ)
variable (H_A : A = Icc (-2 : ℝ) 3)
variable (H_B : B = Ioo (-1 : ℝ) 6)

theorem intersection_A_B :
  A ∩ B = Ioo (-1) 3 :=
by
  rw [H_A, H_B]
  simp
  sorry

end intersection_A_B_l560_560378


namespace zhang_hua_repayment_l560_560565

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end zhang_hua_repayment_l560_560565


namespace solve_diff_eq_l560_560944

theorem solve_diff_eq (C₁ C₂ : ℝ) (y : ℝ → ℝ) (y' y'' : ℝ → ℝ) : 
  (∀ x, y x = (C₁ + C₂ * x) * Real.exp (3 * x) + Real.exp x - 8 * x^2 * Real.exp (3 * x)) →
  (∀ x, y' x = (C₂ * Real.exp (3 * x) + (C₁ + C₂ * x) * 3 * Real.exp (3 * x)) + Real.exp x - 8 * (2 * x * Real.exp (3 * x) + x^2 * 3 * Real.exp (3 * x))) →
  (∀ x, y'' x = (C₂ * 3 * Real.exp (3 * x) + (C₂ * Real.exp (3 * x) + (C₁ + C₂ * x) * 3 * Real.exp (3 * x)) * 3 * Real.exp (3 * x)) + Real.exp x - 8 * (2 * Real.exp (3 * x) + 2 * x * 3 * Real.exp (3 * x) + 3 * x * Real.exp (3 * x) + x^2 * 9 * Real.exp (3 * x))) →
  (∀ x, y'' x - 6 * y' x + 9 * y x = 4 * Real.exp x - 16 * Real.exp (3 * x)) := by
  assume h1 h2 h3
  sorry

end solve_diff_eq_l560_560944


namespace equal_sets_P_and_Q_l560_560954

-- Define the function f with the properties given
variable {f : ℝ → ℝ}
axiom f_increasing : ∀ a b : ℝ, a < b → f(a) < f(b)
axiom f_domain_range : ∀ x : ℝ, x ∈ ℝ

-- Define the sets P and Q
def P : set ℝ := { x | f(x) = x }
def Q : set ℝ := { x | f(f(x)) = x }

-- The theorem to prove
theorem equal_sets_P_and_Q : P = Q :=
by sorry

end equal_sets_P_and_Q_l560_560954


namespace sample_size_proportion_l560_560274

theorem sample_size_proportion (n : ℕ) (ratio_A B C : ℕ) (A_sample : ℕ) (ratio_A_val : ratio_A = 5) (ratio_B_val : ratio_B = 2) (ratio_C_val : ratio_C = 3) (A_sample_val : A_sample = 15) (total_ratio : ratio_A + ratio_B + ratio_C = 10) : 
  15 / n = 5 / 10 → n = 30 :=
sorry

end sample_size_proportion_l560_560274


namespace value_of_a_plus_b_l560_560807

open Set Real

def A : Set ℝ := {x | x^2 - 2x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem value_of_a_plus_b (a b : ℝ) (h1 : A ∪ B a b = univ) (h2 : A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) :
  a + b = 5 :=
sorry

end value_of_a_plus_b_l560_560807


namespace sufficient_condition_perpendicular_planes_l560_560130

-- Definitions of the conditions
variable (α β : Type) [plane α] [plane β] -- α and β are planes
variable (l : Type) [line l] -- l is a line
variable (subset_l_α : l ⊆ α) -- l is a subset of α
variable (perpendicular_l_β : perpendicular l β) -- l is perpendicular to β

-- Theorem Statement
theorem sufficient_condition_perpendicular_planes :
  subset_l_α → perpendicular_l_β → perpendicular α β ∧ ¬ (perpendicular α β → perpendicular_l_β) :=
begin
  sorry -- Proof is omitted
end

end sufficient_condition_perpendicular_planes_l560_560130


namespace axis_symmetry_f_range_g_l560_560401

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := 
  sin (2 * x + π / 3) + cos (2 * x + π / 6) + 2 * sin x * cos x

-- Statement (I): The equation for the axis of symmetry of f(x)
theorem axis_symmetry_f (k : ℤ) : ∃ k : ℤ, ∀ x : ℝ, f(x) = f(π * k.toReal / 2 + π / 12) :=
sorry

-- Definition of the function g(x)
def g (x : ℝ) : ℝ := 
  2 * sin (x / 2 + π / 6)

-- Statement (II): The range of g(x) on [π/3, 2π]
theorem range_g : ∀ x : ℝ, (π / 3 ≤ x ∧ x ≤ 2 * π) → (g(x) ∈ set.Icc (-1 : ℝ) 2) :=
sorry

end axis_symmetry_f_range_g_l560_560401


namespace area_of_remaining_shape_l560_560544

theorem area_of_remaining_shape :
  let side_len := 1 in
  let area_of_shape :=
    let area_of_octagon := (3 + 2 * Real.sqrt 2) - 1 in
    let area_of_removed_triangles := 4 * (Real.sqrt 3 / 4) in
    area_of_octagon - area_of_removed_triangles in
  area_of_shape = 2 + 2 * Real.sqrt 2 - Real.sqrt 3 :=
by
  -- Let side length of the octagon and equilateral triangles be 1
  let side_len := 1
  -- Calculate the area of the initial octagon
  let area_of_octagon :=
    let side_square := 1 + Real.sqrt 2
    let area_square := Real.sqrt (side_square ^ 2)
    area_square - side_len
  -- Calculate the area of the four removed equilateral triangles
  let area_of_removed_triangles := 4 * (Real.sqrt 3 / 4)
  -- Calculate the remaining area
  let area_of_shape := area_of_octagon - area_of_removed_triangles
  -- Assert the area matches the expected value
  have : area_of_shape = 2 + 2 * Real.sqrt 2 - Real.sqrt 3 := by sorry
  exact this

end area_of_remaining_shape_l560_560544


namespace distance_between_x_intercepts_l560_560604

theorem distance_between_x_intercepts
  (slope1 slope2 : ℝ) (intersect : ℝ × ℝ)
  (h_slope1 : slope1 = 2) (h_slope2 : slope2 = 6) (h_intersect : intersect = (40, 30)) :
  let x_intercept1 := (40 - 30 / 2) in
  let x_intercept2 := (40 - 30 / 6) in
  |x_intercept2 - x_intercept1| = 10 :=
by
  let x1 := intersect.1
  let y1 := intersect.2
  let x_intercept1 := x1 - y1 / slope1
  let x_intercept2 := x1 - y1 / slope2
  have h1 : x_intercept1 = 25 := sorry
  have h2 : x_intercept2 = 35 := sorry
  show |x_intercept2 - x_intercept1| = 10 from
    calc
      |x_intercept2 - x_intercept1| = |35 - 25| : by rw [h1, h2]
      ... = 10 : by norm_num

end distance_between_x_intercepts_l560_560604


namespace average_lecture_minutes_l560_560603

theorem average_lecture_minutes
  (lecture_duration : ℕ)
  (total_audience : ℕ)
  (percent_entire : ℝ)
  (percent_missed : ℝ)
  (percent_half : ℝ)
  (average_minutes : ℝ) :
  lecture_duration = 90 →
  total_audience = 200 →
  percent_entire = 0.30 →
  percent_missed = 0.20 →
  percent_half = 0.40 →
  average_minutes = 56.25 :=
by
  sorry

end average_lecture_minutes_l560_560603


namespace last_two_nonzero_digits_80_factorial_l560_560205

theorem last_two_nonzero_digits_80_factorial : 
  (let N := 80.factorial in
   N % (10^19)) % 100 = 12 :=
by
  let N := 80.factorial
  let T := 10^19
  have h1 : N % T = (N / T) * T + (N % T),
  have h2 : N / T = sorry,
  have h3 : N % T = sorry,
  exact sorry

end last_two_nonzero_digits_80_factorial_l560_560205


namespace inscribed_quadrilateral_rhombus_l560_560525

-- Definitions of the conditions in our problem.
variable (A B C D P Q M N R S : Type)
variable (angle : A → B → ℝ)
variable [IsInscribedQuadrilateral A B C D]

-- Lean statement to prove that points of intersection form a rhombus.
theorem inscribed_quadrilateral_rhombus
  (h₁ : extends_1 : intersection_of (A ↔ B, C ↔ D) = P)
  (h₂ : extends_2 : intersection_of (B ↔ C, AD ↔ D) = Q)
  (h₃ : bisectors : intersection_points_of (A ↔ Q ↔ B, B ↔ P ↔ C) = (M, N, R, S)) :
  is_rhombus vertices_of (M, N, R, S) := by
  sorry

end inscribed_quadrilateral_rhombus_l560_560525


namespace total_swimming_hours_over_4_weeks_l560_560928

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l560_560928


namespace distribute_footballs_l560_560208

theorem distribute_footballs : 
  ∃ (f : (ℕ × ℕ × ℕ)), 
    (f.1 + f.2 + f.3 = 9) ∧ 
    (f.1 ≥ 1) ∧ (f.2 ≥ 2) ∧ (f.3 ≥ 3) ∧
    ((∃! (g : (ℕ → ℕ)), g 0 + g 1 + g 2 = 3 ∧ (g 0 ≥ 0) ∧ (g 1 ≥ 1) ∧ (g 2 ≥ 2) 
    ∧ (finset.card {t ∈ finset.powerset (finset.range 3) | 
        (t.card = 3)} +
       finset.card {t ∈ finset.powerset (finset.range 3) | 
        (t.card = 1)} +
       finset.card {t ∈ finset.powerset (finset.range 3) | 
        (t.card = 2)} = 10)) := sorry

end distribute_footballs_l560_560208


namespace collinear_P_O_Q_l560_560145

universe u

-- Define the necessary entities and conditions
variables {Point : Type u} [affine_space Point ℝ] {A B C D O M P Q : Point}

-- Definitions of parallel lines
def parallel (l1 l2 : set Point) : Prop := ∃ a b : Point, l1 = line a b ∧ l2 = line a b

-- Definition indicating collinear points
def collinear (a b c : Point) : Prop := ∃ l : set Point, a ∈ l ∧ b ∈ l ∧ c ∈ l

-- Assume trapezoid with AD || BC
def is_trapezoid (A B C D : Point) : Prop :=
  parallel (line A D) (line B C)

-- Assume conditions from the problem
variables (trapezoid_ABCD : is_trapezoid A B C D)
          (intersect_O : ∃ (l1 l2 : set Point), l1 = line A C ∧ l2 = line B D ∧ O ∈ l1 ∧ O ∈ l2)
          (M_on_CD : M ∈ segment C D)
          (P_on_BC : ∃ l3 : set Point, parallel l3 (line B D) ∧ P ∈ l3 ∧ P ∈ segment B C)
          (Q_on_AD : ∃ l4 : set Point, parallel l4 (line A C) ∧ Q ∈ l4 ∧ Q ∈ segment A D)

-- The theorem stating the collinearity of P, O, and Q
theorem collinear_P_O_Q : collinear P O Q :=
sorry

end collinear_P_O_Q_l560_560145


namespace no_e_line_possible_l560_560813

theorem no_e_line_possible
  (a b : Line)
  (M O : Point) 
  (h_perpendicular : a ⊥ b)
  (h_intersection : a ∩ b = {M})
  (circle_center_O : Circle O (dist O M))
  (A B C : Point)
  (h_A : A ∈ a)
  (h_B : B ∈ b)
  (h_OA : O ∈ line_through O A)
  (h_OB : O ∈ line_through O B)
  (h_circle_intersection : C ∈ (line_through O A ∩ Circle O (dist O M)) ∨ C ∈ (line_through O B ∩ Circle O (dist O M)))
  (h_between : O < A < B ∨ O < B < A) :
  ¬ ∃ (e : Line), (e ∋ O) ∧ (dist A C = dist C B) := 
sorry

end no_e_line_possible_l560_560813


namespace distance_from_P_to_origin_l560_560857

-- Definitions of the point and the origin
def P : ℝ × ℝ := (2, -3)
def origin : ℝ × ℝ := (0, 0)

-- Distance function
def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Theorem statement
theorem distance_from_P_to_origin : distance P origin = Real.sqrt 13 :=
by sorry

end distance_from_P_to_origin_l560_560857


namespace fraction_value_is_one_fourth_l560_560241

theorem fraction_value_is_one_fourth (k : Nat) (hk : k ≥ 1) :
  (10^k + 6 * (10^k - 1) / 9) / (60 * (10^k - 1) / 9 + 4) = 1 / 4 :=
by
  sorry

end fraction_value_is_one_fourth_l560_560241
