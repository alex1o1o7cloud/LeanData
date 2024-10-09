import Mathlib

namespace net_change_in_price_net_change_percentage_l816_81655

theorem net_change_in_price (P : ℝ) :
  0.80 * P * 1.55 - P = 0.24 * P :=
by sorry

theorem net_change_percentage (P : ℝ) :
  ((0.80 * P * 1.55 - P) / P) * 100 = 24 :=
by sorry


end net_change_in_price_net_change_percentage_l816_81655


namespace quadratic_roots_unique_l816_81686

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l816_81686


namespace parts_drawn_l816_81652

-- Given that a sample of 30 parts is drawn and each part has a 25% chance of being drawn,
-- prove that the total number of parts N is 120.

theorem parts_drawn (N : ℕ) (h : (30 : ℚ) / N = 0.25) : N = 120 :=
sorry

end parts_drawn_l816_81652


namespace shaded_region_area_l816_81650

-- Define the problem conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def area_of_shaded_region : ℝ := 50

-- State the theorem to prove the area of the shaded region
theorem shaded_region_area (n : ℕ) (d : ℝ) (area : ℝ) (h1 : n = num_squares) (h2 : d = diagonal_length) : 
  area = area_of_shaded_region :=
sorry

end shaded_region_area_l816_81650


namespace cubic_sum_divisible_by_9_l816_81631

theorem cubic_sum_divisible_by_9 (n : ℕ) (hn : n > 0) : 
  ∃ k, n^3 + (n+1)^3 + (n+2)^3 = 9*k := by
  sorry

end cubic_sum_divisible_by_9_l816_81631


namespace unique_digits_addition_l816_81626

theorem unique_digits_addition :
  ∃ (X Y B M C : ℕ), 
    -- Conditions
    X ≠ 0 ∧ Y ≠ 0 ∧ B ≠ 0 ∧ M ≠ 0 ∧ C ≠ 0 ∧
    X ≠ Y ∧ X ≠ B ∧ X ≠ M ∧ X ≠ C ∧ Y ≠ B ∧ Y ≠ M ∧ Y ≠ C ∧ B ≠ M ∧ B ≠ C ∧ M ≠ C ∧
    -- Addition equation with distinct digits
    (X * 1000 + Y * 100 + 70) + (B * 100 + M * 10 + C) = (B * 1000 + M * 100 + C * 10 + 0) ∧
    -- Correct Answer
    X = 9 ∧ Y = 8 ∧ B = 3 ∧ M = 8 ∧ C = 7 :=
sorry

end unique_digits_addition_l816_81626


namespace solutions_to_quadratic_l816_81666

noncomputable def a : ℝ := (6 + Real.sqrt 92) / 2
noncomputable def b : ℝ := (6 - Real.sqrt 92) / 2

theorem solutions_to_quadratic :
  a ≥ b ∧ ((∀ x : ℝ, x^2 - 6 * x + 11 = 25 → x = a ∨ x = b) → 3 * a + 2 * b = 15 + Real.sqrt 92 / 2) := by
  sorry

end solutions_to_quadratic_l816_81666


namespace marbles_remainder_l816_81692

theorem marbles_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 :=
by sorry

end marbles_remainder_l816_81692


namespace student_weight_l816_81691

variable (S W : ℕ)

theorem student_weight (h1 : S - 5 = 2 * W) (h2 : S + W = 110) : S = 75 :=
by
  sorry

end student_weight_l816_81691


namespace min_value_of_a_plus_b_minus_c_l816_81696

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ)
  (h : ∀ x y : ℝ, 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) :
  a = 3 ∧ b = 4 ∧ -5 ≤ c ∧ c ≤ 5 ∧ a + b - c = 2 :=
by {
  sorry
}

end min_value_of_a_plus_b_minus_c_l816_81696


namespace chalkboard_area_l816_81675

theorem chalkboard_area (width : ℝ) (h₁ : width = 3.5) (length : ℝ) (h₂ : length = 2.3 * width) : 
  width * length = 28.175 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end chalkboard_area_l816_81675


namespace abc_product_l816_81681

theorem abc_product (A B C D : ℕ) 
  (h1 : A + B + C + D = 64)
  (h2 : A + 3 = B - 3)
  (h3 : A + 3 = C * 3)
  (h4 : A + 3 = D / 3) :
  A * B * C * D = 19440 := 
by
  sorry

end abc_product_l816_81681


namespace n_squared_divisible_by_12_l816_81656

theorem n_squared_divisible_by_12 (n : ℕ) : 12 ∣ n^2 * (n^2 - 1) :=
  sorry

end n_squared_divisible_by_12_l816_81656


namespace two_digit_numbers_div_by_7_with_remainder_1_l816_81694

theorem two_digit_numbers_div_by_7_with_remainder_1 :
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (10 * a + b) % 7 = 1 ∧ (10 * b + a) % 7 = 1} 
  = {22, 29, 92, 99} := 
by
  sorry

end two_digit_numbers_div_by_7_with_remainder_1_l816_81694


namespace c_is_perfect_square_l816_81606

theorem c_is_perfect_square (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : c = a + b / a - 1 / b) : ∃ m : ℕ, c = m * m :=
by
  sorry

end c_is_perfect_square_l816_81606


namespace common_altitude_l816_81685

theorem common_altitude (A1 A2 b1 b2 h : ℝ)
    (hA1 : A1 = 800)
    (hA2 : A2 = 1200)
    (hb1 : b1 = 40)
    (hb2 : b2 = 60)
    (h1 : A1 = 1 / 2 * b1 * h)
    (h2 : A2 = 1 / 2 * b2 * h) :
    h = 40 := 
sorry

end common_altitude_l816_81685


namespace find_other_integer_l816_81619

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 150) (h2 : x = 15 ∨ y = 15) : y = 30 :=
by
  sorry

end find_other_integer_l816_81619


namespace distance_behind_C_l816_81688

-- Conditions based on the problem
def distance_race : ℕ := 1000
def distance_B_when_A_finishes : ℕ := 50
def distance_C_when_B_finishes : ℕ := 100

-- Derived condition based on given problem details
def distance_run_by_B_when_A_finishes : ℕ := distance_race - distance_B_when_A_finishes
def distance_run_by_C_when_B_finishes : ℕ := distance_race - distance_C_when_B_finishes

-- Ratios
def ratio_B_to_A : ℚ := distance_run_by_B_when_A_finishes / distance_race
def ratio_C_to_B : ℚ := distance_run_by_C_when_B_finishes / distance_race

-- Combined ratio
def ratio_C_to_A : ℚ := ratio_C_to_B * ratio_B_to_A

-- Distance run by C when A finishes
def distance_run_by_C_when_A_finishes : ℚ := distance_race * ratio_C_to_A

-- Distance C is behind the finish line when A finishes
def distance_C_behind_when_A_finishes : ℚ := distance_race - distance_run_by_C_when_A_finishes

theorem distance_behind_C (d_race : ℕ) (d_BA : ℕ) (d_CB : ℕ)
  (hA : d_race = 1000) (hB : d_BA = 50) (hC : d_CB = 100) :
  distance_C_behind_when_A_finishes = 145 :=
  by sorry

end distance_behind_C_l816_81688


namespace chocolates_total_l816_81601

theorem chocolates_total (x : ℕ)
  (h1 : x - 12 + x - 18 + x - 20 = 2 * x) :
  x = 50 :=
  sorry

end chocolates_total_l816_81601


namespace remainder_is_three_l816_81693

def eleven_div_four_has_remainder_three (A : ℕ) : Prop :=
  11 = 4 * 2 + A

theorem remainder_is_three : eleven_div_four_has_remainder_three 3 :=
by
  sorry

end remainder_is_three_l816_81693


namespace functional_eq_l816_81641

theorem functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_eq_l816_81641


namespace incorrect_correlation_coefficient_range_l816_81663

noncomputable def regression_analysis_conditions 
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) : Prop :=
  non_deterministic_relationship ∧
  correlation_coefficient_range ∧
  perfect_correlation ∧
  correlation_coefficient_sign

theorem incorrect_correlation_coefficient_range
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) :
  regression_analysis_conditions 
    non_deterministic_relationship 
    correlation_coefficient_range 
    perfect_correlation 
    correlation_coefficient_sign →
  ¬ correlation_coefficient_range :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end incorrect_correlation_coefficient_range_l816_81663


namespace euclid_middle_school_math_students_l816_81623

theorem euclid_middle_school_math_students
  (students_Germain : ℕ)
  (students_Newton : ℕ)
  (students_Young : ℕ)
  (students_Euler : ℕ)
  (h_Germain : students_Germain = 12)
  (h_Newton : students_Newton = 10)
  (h_Young : students_Young = 7)
  (h_Euler : students_Euler = 6) :
  students_Germain + students_Newton + students_Young + students_Euler = 35 :=
by {
  sorry
}

end euclid_middle_school_math_students_l816_81623


namespace keith_attended_games_l816_81682

def total_games : ℕ := 8
def missed_games : ℕ := 4
def attended_games (total : ℕ) (missed : ℕ) : ℕ := total - missed

theorem keith_attended_games : attended_games total_games missed_games = 4 := by
  sorry

end keith_attended_games_l816_81682


namespace peter_total_spent_l816_81667

/-
Peter bought a scooter for a certain sum of money. He spent 5% of the cost on the first round of repairs, another 10% on the second round of repairs, and 7% on the third round of repairs. After this, he had to pay a 12% tax on the original cost. Also, he offered a 15% holiday discount on the scooter's selling price. Despite the discount, he still managed to make a profit of $2000. How much did he spend in total, including repairs, tax, and discount if his profit percentage was 30%?
-/

noncomputable def total_spent (C S P : ℝ) : Prop :=
    (0.3 * C = P) ∧
    (0.85 * S = 1.34 * C + P) ∧
    (C = 2000 / 0.3) ∧
    (1.34 * C = 8933.33)

theorem peter_total_spent
  (C S P : ℝ)
  (h1 : 0.3 * C = P)
  (h2 : 0.85 * S = 1.34 * C + P)
  (h3 : C = 2000 / 0.3)
  : 1.34 * C = 8933.33 := by 
  sorry

end peter_total_spent_l816_81667


namespace range_of_a_l816_81625

variable (a b c : ℝ)

def condition1 := a^2 - b * c - 8 * a + 7 = 0

def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
  sorry

end range_of_a_l816_81625


namespace total_handshakes_l816_81684

theorem total_handshakes (gremlins imps unfriendly_gremlins : ℕ) 
    (handshakes_among_friendly : ℕ) (handshakes_friendly_with_unfriendly : ℕ) 
    (handshakes_between_imps_and_gremlins : ℕ) 
    (h_friendly : gremlins = 30) (h_imps : imps = 20) 
    (h_unfriendly : unfriendly_gremlins = 10) 
    (h_handshakes_among_friendly : handshakes_among_friendly = 190) 
    (h_handshakes_friendly_with_unfriendly : handshakes_friendly_with_unfriendly = 200)
    (h_handshakes_between_imps_and_gremlins : handshakes_between_imps_and_gremlins = 600) : 
    handshakes_among_friendly + handshakes_friendly_with_unfriendly + handshakes_between_imps_and_gremlins = 990 := 
by 
    sorry

end total_handshakes_l816_81684


namespace single_intersection_not_necessarily_tangent_l816_81644

structure Hyperbola where
  -- Placeholder for hyperbola properties
  axis1 : Real
  axis2 : Real

def is_tangent (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for tangency
  ∃ p : Real × Real, l = { p }

def is_parallel_to_asymptote (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for parallelism to asymptote 
  ∃ A : Real, l = { (x, A * x) | x : Real }

theorem single_intersection_not_necessarily_tangent
  (l : Set (Real × Real)) (H : Hyperbola) (h : ∃ p : Real × Real, l = { p }) :
  ¬ is_tangent l H ∨ is_parallel_to_asymptote l H :=
sorry

end single_intersection_not_necessarily_tangent_l816_81644


namespace L_like_reflexive_l816_81695

-- Definitions of the shapes and condition of being an "L-like shape"
inductive Shape
| A | B | C | D | E | LLike : Shape → Shape

-- reflection_equiv function representing reflection equivalence across a vertical dashed line
def reflection_equiv (s1 s2 : Shape) : Prop :=
sorry -- This would be defined according to the exact conditions of the shapes and reflection logic.

-- Given the shapes
axiom L_like : Shape
axiom A : Shape
axiom B : Shape
axiom C : Shape
axiom D : Shape
axiom E : Shape

-- The proof problem: Shape D is the mirrored reflection of the given "L-like shape" across a vertical dashed line
theorem L_like_reflexive :
  reflection_equiv L_like D :=
sorry

end L_like_reflexive_l816_81695


namespace ratio_of_fifth_terms_l816_81677

theorem ratio_of_fifth_terms (a_n b_n : ℕ → ℕ) (S T : ℕ → ℕ)
  (hs : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (ht : ∀ n, T n = n * (b_n 1 + b_n n) / 2)
  (h : ∀ n, S n / T n = (7 * n + 2) / (n + 3)) :
  a_n 5 / b_n 5 = 65 / 12 :=
by
  sorry

end ratio_of_fifth_terms_l816_81677


namespace pairs_of_positive_integers_l816_81689

theorem pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
    (∃ (m : ℕ), m ≥ 2 ∧ (x = m^3 + 2*m^2 - m - 1 ∧ y = m^3 + m^2 - 2*m - 1 ∨ 
                        x = m^3 + m^2 - 2*m - 1 ∧ y = m^3 + 2*m^2 - m - 1)) ∨
    (x = 1 ∧ y = 1) ↔ 
    (∃ n : ℝ, n^3 = 7*x^2 - 13*x*y + 7*y^2) ∧ (Int.natAbs (x - y) - 1 = n) :=
by
  sorry

end pairs_of_positive_integers_l816_81689


namespace remaining_rectangle_area_l816_81608

theorem remaining_rectangle_area (s a b : ℕ) (hs : s = a + b) (total_area_cut : a^2 + b^2 = 40) : s^2 - 40 = 24 :=
by
  sorry

end remaining_rectangle_area_l816_81608


namespace fruit_display_l816_81674

theorem fruit_display (bananas : ℕ) (Oranges : ℕ) (Apples : ℕ) (hBananas : bananas = 5)
  (hOranges : Oranges = 2 * bananas) (hApples : Apples = 2 * Oranges) :
  bananas + Oranges + Apples = 35 :=
by sorry

end fruit_display_l816_81674


namespace inequality_proof_l816_81647

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3)^(1 / 8) - 1) :=
by
  sorry

end inequality_proof_l816_81647


namespace thirty_percent_greater_l816_81618

theorem thirty_percent_greater (x : ℝ) (h : x = 1.3 * 88) : x = 114.4 :=
sorry

end thirty_percent_greater_l816_81618


namespace diagonal_BD_size_cos_A_value_l816_81637

noncomputable def AB := 250
noncomputable def CD := 250
noncomputable def angle_A := 120
noncomputable def angle_C := 120
noncomputable def AD := 150
noncomputable def BC := 150
noncomputable def perimeter := 800

/-- The size of the diagonal BD in isosceles trapezoid ABCD is 350, given the conditions -/
theorem diagonal_BD_size (AB CD AD BC : ℕ) (angle_A angle_C : ℝ) :
  AB = 250 → CD = 250 → AD = 150 → BC = 150 →
  angle_A = 120 → angle_C = 120 →
  ∃ BD : ℝ, BD = 350 :=
by
  sorry

/-- The cosine of angle A is -0.5, given the angle is 120 degrees -/
theorem cos_A_value (angle_A : ℝ) :
  angle_A = 120 → ∃ cos_A : ℝ, cos_A = -0.5 :=
by
  sorry

end diagonal_BD_size_cos_A_value_l816_81637


namespace sum_of_angles_l816_81621

theorem sum_of_angles (a b : ℝ) (ha : a = 45) (hb : b = 225) : a + b = 270 :=
by
  rw [ha, hb]
  norm_num -- Lean's built-in tactic to normalize numerical expressions

end sum_of_angles_l816_81621


namespace width_of_first_sheet_l816_81632

theorem width_of_first_sheet (w : ℝ) (h : 2 * (w * 17) = 2 * (8.5 * 11) + 100) : w = 287 / 34 :=
by
  sorry

end width_of_first_sheet_l816_81632


namespace area_white_portion_l816_81697

/-- The dimensions of the sign --/
def sign_width : ℝ := 7
def sign_height : ℝ := 20

/-- The areas of letters "S", "A", "V", and "E" --/
def area_S : ℝ := 14
def area_A : ℝ := 16
def area_V : ℝ := 12
def area_E : ℝ := 12

/-- Calculate the total area of the sign --/
def total_area_sign : ℝ := sign_width * sign_height

/-- Calculate the total area covered by the letters --/
def total_area_letters : ℝ := area_S + area_A + area_V + area_E

/-- Calculate the area of the white portion of the sign --/
theorem area_white_portion : total_area_sign - total_area_letters = 86 := by
  sorry

end area_white_portion_l816_81697


namespace rectangle_ratio_l816_81662

theorem rectangle_ratio {l w : ℕ} (h_w : w = 5) (h_A : 50 = l * w) : l / w = 2 := by 
  sorry

end rectangle_ratio_l816_81662


namespace slopes_hyperbola_l816_81605

theorem slopes_hyperbola 
  (x y : ℝ)
  (M : ℝ × ℝ) 
  (t m : ℝ) 
  (h_point_M_on_line: M = (9 / 5, t))
  (h_hyperbola : ∀ t: ℝ, (16 * m^2 - 9) * t^2 + 160 * m * t + 256 = 0)
  (k1 k2 k3 : ℝ)
  (h_k2 : k2 = -5 * t / 16) :
  k1 + k3 = 2 * k2 :=
sorry

end slopes_hyperbola_l816_81605


namespace digit_possibilities_757_l816_81683

theorem digit_possibilities_757
  (N : ℕ)
  (h : N < 10) :
  (∃ d₀ d₁ d₂ : ℕ, (d₀ = 2 ∨ d₀ = 5 ∨ d₀ = 8) ∧
  (d₁ = 2 ∨ d₁ = 5 ∨ d₁ = 8) ∧
  (d₂ = 2 ∨ d₂ = 5 ∨ d₂ = 8) ∧
  (d₀ ≠ d₁) ∧
  (d₀ ≠ d₂) ∧
  (d₁ ≠ d₂)) :=
by
  sorry

end digit_possibilities_757_l816_81683


namespace base_nine_to_mod_five_l816_81649

-- Define the base-nine number N
def N : ℕ := 2 * 9^10 + 7 * 9^9 + 0 * 9^8 + 0 * 9^7 + 6 * 9^6 + 0 * 9^5 + 0 * 9^4 + 0 * 9^3 + 0 * 9^2 + 5 * 9^1 + 2 * 9^0

-- Theorem statement
theorem base_nine_to_mod_five : N % 5 = 3 :=
by
  sorry

end base_nine_to_mod_five_l816_81649


namespace maximum_value_expression_l816_81633

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end maximum_value_expression_l816_81633


namespace zero_in_interval_l816_81614

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_in_interval : 
  ∃ x₀, f x₀ = 0 ∧ (2 : ℝ) < x₀ ∧ x₀ < (3 : ℝ) :=
by
  sorry

end zero_in_interval_l816_81614


namespace parabola_vertex_above_x_axis_l816_81622

theorem parabola_vertex_above_x_axis (k : ℝ) (h : k > 9 / 4) : 
  ∃ y : ℝ, ∀ x : ℝ, y = (x - 3 / 2) ^ 2 + k - 9 / 4 ∧ y > 0 := 
by
  sorry

end parabola_vertex_above_x_axis_l816_81622


namespace fourth_person_height_is_82_l816_81665

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end fourth_person_height_is_82_l816_81665


namespace divisibility_problem_l816_81616

theorem divisibility_problem (q : ℕ) (hq : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬((q + 2)^(q - 3) + 1) % (q - 4) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % q = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 6) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 3) = 0 := sorry

end divisibility_problem_l816_81616


namespace ice_cream_volume_l816_81610

-- Definitions based on Conditions
def radius_cone : Real := 3 -- radius at the opening of the cone
def height_cone : Real := 12 -- height of the cone

-- The proof statement
theorem ice_cream_volume :
  (1 / 3 * Real.pi * radius_cone^2 * height_cone) + (4 / 3 * Real.pi * radius_cone^3) = 72 * Real.pi := by
  sorry

end ice_cream_volume_l816_81610


namespace bottles_purchased_l816_81659

/-- Given P bottles can be bought for R dollars, determine how many bottles can be bought for M euros
    if 1 euro is worth 1.2 dollars and there is a 10% discount when buying with euros. -/
theorem bottles_purchased (P R M : ℝ) (hR : R > 0) (hP : P > 0) :
  let euro_to_dollars := 1.2
  let discount := 0.9
  let dollars := euro_to_dollars * M * discount
  (P / R) * dollars = (1.32 * P * M) / R :=
by
  sorry

end bottles_purchased_l816_81659


namespace repeat_decimals_subtraction_l816_81603

-- Define repeating decimal 0.4 repeating as a fraction
def repr_decimal_4 : ℚ := 4 / 9

-- Define repeating decimal 0.6 repeating as a fraction
def repr_decimal_6 : ℚ := 2 / 3

-- Theorem stating the equivalence of subtraction of these repeating decimals
theorem repeat_decimals_subtraction :
  repr_decimal_4 - repr_decimal_6 = -2 / 9 :=
sorry

end repeat_decimals_subtraction_l816_81603


namespace remainder_b100_mod_81_l816_81607

def b (n : ℕ) := 7^n + 9^n

theorem remainder_b100_mod_81 : (b 100) % 81 = 38 := by
  sorry

end remainder_b100_mod_81_l816_81607


namespace quarters_to_dollars_l816_81640

theorem quarters_to_dollars (total_quarters : ℕ) (quarters_per_dollar : ℕ) (h1 : total_quarters = 8) (h2 : quarters_per_dollar = 4) : total_quarters / quarters_per_dollar = 2 :=
by {
  sorry
}

end quarters_to_dollars_l816_81640


namespace num_integers_satisfying_inequality_l816_81636

theorem num_integers_satisfying_inequality : 
  ∃ (xs : Finset ℤ), (∀ x ∈ xs, -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9) ∧ xs.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l816_81636


namespace find_other_number_l816_81613

theorem find_other_number (a b : ℕ) (h1 : (a + b) / 2 = 7) (h2 : a = 5) : b = 9 :=
by
  sorry

end find_other_number_l816_81613


namespace slope_intercept_form_of_line_l816_81620

theorem slope_intercept_form_of_line :
  ∀ (x y : ℝ), (∀ (a b : ℝ), (a, b) = (0, 4) ∨ (a, b) = (3, 0) → y = - (4 / 3) * x + 4) := 
by
  sorry

end slope_intercept_form_of_line_l816_81620


namespace equal_naturals_of_infinite_divisibility_l816_81670

theorem equal_naturals_of_infinite_divisibility
  (a b : ℕ)
  (h : ∀ᶠ n in Filter.atTop, (a^(n + 1) + b^(n + 1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end equal_naturals_of_infinite_divisibility_l816_81670


namespace parabola_equation_l816_81648

-- Defining the point F and the line
def F : ℝ × ℝ := (0, 4)

def line_eq (y : ℝ) : Prop := y = -5

-- Defining the condition that point M is closer to F(0, 4) than to the line y = -5 by less than 1
def condition (M : ℝ × ℝ) : Prop :=
  let dist_to_F := (M.1 - F.1)^2 + (M.2 - F.2)^2
  let dist_to_line := abs (M.2 - (-5))
  abs (dist_to_F - dist_to_line) < 1

-- The equation we need to prove under the given condition
theorem parabola_equation (M : ℝ × ℝ) (h : condition M) : M.1^2 = 16 * M.2 := 
sorry

end parabola_equation_l816_81648


namespace number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l816_81671

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l816_81671


namespace pieces_from_rod_l816_81643

theorem pieces_from_rod (length_of_rod : ℝ) (length_of_piece : ℝ) 
  (h_rod : length_of_rod = 42.5) 
  (h_piece : length_of_piece = 0.85) :
  length_of_rod / length_of_piece = 50 :=
by
  rw [h_rod, h_piece]
  calc
    42.5 / 0.85 = 50 := by norm_num

end pieces_from_rod_l816_81643


namespace last_two_digits_of_7_pow_5_pow_6_l816_81624

theorem last_two_digits_of_7_pow_5_pow_6 : (7 ^ (5 ^ 6)) % 100 = 7 := 
  sorry

end last_two_digits_of_7_pow_5_pow_6_l816_81624


namespace not_possible_1006_2012_gons_l816_81680

theorem not_possible_1006_2012_gons :
  ∀ (n : ℕ), (∀ (k : ℕ), k ≤ 2011 → 2 * n ≤ k) → n ≠ 1006 :=
by
  intro n h
  -- Here goes the skipped proof part
  sorry

end not_possible_1006_2012_gons_l816_81680


namespace standard_equation_hyperbola_l816_81678

-- Define necessary conditions
def condition_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def condition_asymptote (a b : ℝ) :=
  b / a = Real.sqrt 3

def condition_focus_hyperbola_parabola (a b : ℝ) :=
  (a^2 + b^2).sqrt = 4

-- Define the proof problem
theorem standard_equation_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : condition_asymptote a b)
  (h_focus : condition_focus_hyperbola_parabola a b) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
sorry

end standard_equation_hyperbola_l816_81678


namespace find_n_l816_81690

theorem find_n (n : ℕ) (m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_div : (2^n - 1) ∣ (m^2 + 81)) : 
  ∃ k : ℕ, n = 2^k := 
sorry

end find_n_l816_81690


namespace yangmei_1_yangmei_2i_yangmei_2ii_l816_81653

-- Problem 1: Prove that a = 20
theorem yangmei_1 (a : ℕ) (h : 160 * a + 270 * a = 8600) : a = 20 := by
  sorry

-- Problem 2 (i): Prove x = 44 and y = 36
theorem yangmei_2i (x y : ℕ) (h1 : 160 * x + 270 * y = 16760) (h2 : 8 * x + 18 * y = 1000) : x = 44 ∧ y = 36 := by
  sorry

-- Problem 2 (ii): Prove b = 9 or 18
theorem yangmei_2ii (m n b : ℕ) (h1 : 8 * (m + b) + 18 * n = 1000) (h2 : 160 * m + 270 * n = 16760) (h3 : 0 < b)
: b = 9 ∨ b = 18 := by
  sorry

end yangmei_1_yangmei_2i_yangmei_2ii_l816_81653


namespace distribute_weights_l816_81687

theorem distribute_weights (max_weight : ℕ) (w_gbeans w_milk w_carrots w_apples w_bread w_rice w_oranges w_pasta : ℕ)
  (h_max_weight : max_weight = 20)
  (h_w_gbeans : w_gbeans = 4)
  (h_w_milk : w_milk = 6)
  (h_w_carrots : w_carrots = 2 * w_gbeans)
  (h_w_apples : w_apples = 3)
  (h_w_bread : w_bread = 1)
  (h_w_rice : w_rice = 5)
  (h_w_oranges : w_oranges = 2)
  (h_w_pasta : w_pasta = 3)
  : (w_gbeans + w_milk + w_carrots + w_apples + w_bread - 2 = max_weight) ∧ 
    (w_rice + w_oranges + w_pasta + 2 ≤ max_weight) :=
by
  sorry

end distribute_weights_l816_81687


namespace percentage_decrease_l816_81604

variable (current_price original_price : ℝ)

theorem percentage_decrease (h1 : current_price = 760) (h2 : original_price = 1000) :
  (original_price - current_price) / original_price * 100 = 24 :=
by
  sorry

end percentage_decrease_l816_81604


namespace tangent_line_ratio_l816_81654

variables {x1 x2 : ℝ}

theorem tangent_line_ratio (h1 : 2 * x1 = 3 * x2^2) (h2 : x1^2 = 2 * x2^3) : (x1 / x2) = 4 / 3 :=
by sorry

end tangent_line_ratio_l816_81654


namespace candy_bar_sales_ratio_l816_81609

theorem candy_bar_sales_ratio
    (candy_bar_cost : ℕ := 2)
    (marvin_candy_sold : ℕ := 35)
    (tina_extra_earnings : ℕ := 140)
    (marvin_earnings := marvin_candy_sold * candy_bar_cost)
    (tina_earnings := marvin_earnings + tina_extra_earnings)
    (tina_candy_sold := tina_earnings / candy_bar_cost):
  tina_candy_sold / marvin_candy_sold = 3 :=
by
  sorry

end candy_bar_sales_ratio_l816_81609


namespace f_is_odd_l816_81664

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2 * x

-- State the problem
theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

end f_is_odd_l816_81664


namespace probability_same_color_two_dice_l816_81629

theorem probability_same_color_two_dice :
  let total_sides : ℕ := 30
  let maroon_sides : ℕ := 5
  let teal_sides : ℕ := 10
  let cyan_sides : ℕ := 12
  let sparkly_sides : ℕ := 3
  (maroon_sides / total_sides)^2 + (teal_sides / total_sides)^2 + (cyan_sides / total_sides)^2 + (sparkly_sides / total_sides)^2 = 139 / 450 :=
by
  sorry

end probability_same_color_two_dice_l816_81629


namespace vincent_total_cost_l816_81658

theorem vincent_total_cost :
  let day1_packs := 15
  let day1_pack_cost := 2.50
  let discount_percent := 0.10
  let day2_packs := 25
  let day2_pack_cost := 3.00
  let tax_percent := 0.05
  let day1_total_cost_before_discount := day1_packs * day1_pack_cost
  let day1_discount_amount := discount_percent * day1_total_cost_before_discount
  let day1_total_cost_after_discount := day1_total_cost_before_discount - day1_discount_amount
  let day2_total_cost_before_tax := day2_packs * day2_pack_cost
  let day2_tax_amount := tax_percent * day2_total_cost_before_tax
  let day2_total_cost_after_tax := day2_total_cost_before_tax + day2_tax_amount
  let total_cost := day1_total_cost_after_discount + day2_total_cost_after_tax
  total_cost = 112.50 :=
by 
  -- Mathlib can be used for floating point calculations, if needed
  -- For the purposes of this example, we assume calculations are correct.
  sorry

end vincent_total_cost_l816_81658


namespace poly_has_one_positive_and_one_negative_root_l816_81646

theorem poly_has_one_positive_and_one_negative_root :
  ∃! r1, r1 > 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) ∧ 
  ∃! r2, r2 < 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) := by
sorry

end poly_has_one_positive_and_one_negative_root_l816_81646


namespace range_of_m_l816_81600

-- Define the two vectors a and b
def vector_a := (1, 2)
def vector_b (m : ℝ) := (m, 3 * m - 2)

-- Define the condition for non-collinearity
def non_collinear (m : ℝ) := ¬ (m / 1 = (3 * m - 2) / 2)

theorem range_of_m (m : ℝ) : non_collinear m ↔ m ≠ 2 :=
  sorry

end range_of_m_l816_81600


namespace remainder_when_divided_by_x_minus_2_l816_81615

def polynomial (x : ℝ) := x^5 + 2 * x^3 - x + 4

theorem remainder_when_divided_by_x_minus_2 :
  polynomial 2 = 50 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l816_81615


namespace total_number_of_cats_l816_81699

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats_l816_81699


namespace meeting_time_eqn_l816_81627

-- Mathematical definitions derived from conditions:
def distance := 270 -- Cities A and B are 270 kilometers apart.
def speed_fast_train := 120 -- Speed of the fast train is 120 km/h.
def speed_slow_train := 75 -- Speed of the slow train is 75 km/h.
def time_head_start := 1 -- Slow train departs 1 hour before the fast train.

-- Let x be the number of hours it takes for the two trains to meet after the fast train departs
def x : Real := sorry

-- Proving the equation representing the situation:
theorem meeting_time_eqn : 75 * 1 + (120 + 75) * x = 270 :=
by
  sorry

end meeting_time_eqn_l816_81627


namespace smaller_number_of_two_digits_product_3774_l816_81673

theorem smaller_number_of_two_digits_product_3774 (a b : ℕ) (ha : 9 < a ∧ a < 100) (hb : 9 < b ∧ b < 100) (h : a * b = 3774) : a = 51 ∨ b = 51 :=
by
  sorry

end smaller_number_of_two_digits_product_3774_l816_81673


namespace pascal_50_5th_element_is_22050_l816_81639

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l816_81639


namespace cos_squared_formula_15deg_l816_81698

theorem cos_squared_formula_15deg :
  (Real.cos (15 * Real.pi / 180))^2 - (1 / 2) = (Real.sqrt 3) / 4 :=
by
  sorry

end cos_squared_formula_15deg_l816_81698


namespace central_angle_of_spherical_sector_l816_81617

theorem central_angle_of_spherical_sector (R α r m : ℝ) (h1 : R * Real.pi * r = 2 * R * Real.pi * m) (h2 : R^2 = r^2 + (R - m)^2) :
  α = 2 * Real.arccos (3 / 5) :=
by
  sorry

end central_angle_of_spherical_sector_l816_81617


namespace set_D_not_right_triangle_l816_81672

theorem set_D_not_right_triangle :
  let a := 11
  let b := 12
  let c := 15
  a ^ 2 + b ^ 2 ≠ c ^ 2
:=
by
  let a := 11
  let b := 12
  let c := 15
  sorry

end set_D_not_right_triangle_l816_81672


namespace inverse_of_true_implies_negation_true_l816_81645

variable (P : Prop)
theorem inverse_of_true_implies_negation_true (h : ¬ P) : ¬ P :=
by 
  exact h

end inverse_of_true_implies_negation_true_l816_81645


namespace dasha_meeting_sasha_l816_81611

def stripes_on_zebra : ℕ := 360

variables {v : ℝ} -- speed of Masha
def dasha_speed (v : ℝ) : ℝ := 2 * v -- speed of Dasha (twice Masha's speed)

def masha_distance_before_meeting_sasha : ℕ := 180
def total_stripes_met : ℕ := stripes_on_zebra
def relative_speed_masha_sasha (v : ℝ) : ℝ := v + v -- combined speed of Masha and Sasha
def relative_speed_dasha_sasha (v : ℝ) : ℝ := 3 * v -- combined speed of Dasha and Sasha

theorem dasha_meeting_sasha (v : ℝ) (hv : 0 < v) :
  ∃ t' t'', 
  (t'' = 120 / v) ∧ (dasha_speed v * t' = 240) :=
by {
  sorry
}

end dasha_meeting_sasha_l816_81611


namespace lines_intersect_not_perpendicular_l816_81651

noncomputable def slopes_are_roots (m k1 k2 : ℝ) : Prop :=
  k1^2 + m*k1 - 2 = 0 ∧ k2^2 + m*k2 - 2 = 0

theorem lines_intersect_not_perpendicular (m k1 k2 : ℝ) (h : slopes_are_roots m k1 k2) : (k1 * k2 = -2 ∧ k1 ≠ k2) → ∃ l1 l2 : ℝ, l1 ≠ l2 ∧ l1 = k1 ∧ l2 = k2 :=
by
  sorry

end lines_intersect_not_perpendicular_l816_81651


namespace divisibility_by_six_l816_81661

theorem divisibility_by_six (n : ℤ) : 6 ∣ (n^3 - n) := 
sorry

end divisibility_by_six_l816_81661


namespace william_farm_tax_l816_81612

theorem william_farm_tax :
  let total_tax_collected := 3840
  let william_land_percentage := 0.25
  william_land_percentage * total_tax_collected = 960 :=
by sorry

end william_farm_tax_l816_81612


namespace probability_male_is_2_5_l816_81668

variable (num_male_students num_female_students : ℕ)

def total_students (num_male_students num_female_students : ℕ) : ℕ :=
  num_male_students + num_female_students

def probability_of_male (num_male_students num_female_students : ℕ) : ℚ :=
  num_male_students / (total_students num_male_students num_female_students : ℚ)

theorem probability_male_is_2_5 :
  probability_of_male 2 3 = 2 / 5 := by
    sorry

end probability_male_is_2_5_l816_81668


namespace a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l816_81660

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_div_3_sum_two_cubes (n : ℕ) : ∃ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a_n n / 3 = x^3 + y^3) := sorry

theorem a_n_div_3_not_sum_two_squares (n : ℕ) : ¬ (∃ x y : ℤ, a_n n / 3 = x^2 + y^2) := sorry

end a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l816_81660


namespace yarn_length_proof_l816_81630

def green_length := 156
def total_length := 632

noncomputable def red_length (x : ℕ) := green_length * x + 8

theorem yarn_length_proof (x : ℕ) (green_length_eq : green_length = 156)
  (total_length_eq : green_length + red_length x = 632) : x = 3 :=
by {
  sorry
}

end yarn_length_proof_l816_81630


namespace marble_count_calculation_l816_81669

theorem marble_count_calculation (y b g : ℕ) (x : ℕ)
  (h1 : y = 2 * x)
  (h2 : b = 3 * x)
  (h3 : g = 4 * x)
  (h4 : g = 32) : y + b + g = 72 :=
by
  sorry

end marble_count_calculation_l816_81669


namespace exists_triangle_with_side_lengths_l816_81676

theorem exists_triangle_with_side_lengths (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end exists_triangle_with_side_lengths_l816_81676


namespace min_square_distance_l816_81602

theorem min_square_distance (x y z w : ℝ) (h1 : x * y = 4) (h2 : z^2 + 4 * w^2 = 4) : (x - z)^2 + (y - w)^2 ≥ 1.6 :=
sorry

end min_square_distance_l816_81602


namespace percentage_increase_l816_81634

theorem percentage_increase (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 480) :
  ((new_price - original_price) / original_price) * 100 = 60 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l816_81634


namespace total_dividends_received_l816_81628

theorem total_dividends_received
  (investment : ℝ)
  (share_price : ℝ)
  (nominal_value : ℝ)
  (dividend_rate_year1 : ℝ)
  (dividend_rate_year2 : ℝ)
  (dividend_rate_year3 : ℝ)
  (num_shares : ℝ)
  (total_dividends : ℝ) :
  investment = 14400 →
  share_price = 120 →
  nominal_value = 100 →
  dividend_rate_year1 = 0.07 →
  dividend_rate_year2 = 0.09 →
  dividend_rate_year3 = 0.06 →
  num_shares = investment / share_price → 
  total_dividends = (dividend_rate_year1 * nominal_value * num_shares) +
                    (dividend_rate_year2 * nominal_value * num_shares) +
                    (dividend_rate_year3 * nominal_value * num_shares) →
  total_dividends = 2640 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_dividends_received_l816_81628


namespace smallest_f1_value_l816_81642

noncomputable def polynomial := 
  fun (f : ℝ → ℝ) (r s : ℝ) => 
    f = λ x => (x - r) * (x - s) * (x - ((r + s)/2))

def distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ polynomial f r s ∧ 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (f ∘ f) a = 0 ∧ (f ∘ f) b = 0 ∧ (f ∘ f) c = 0)

theorem smallest_f1_value
  (f : ℝ → ℝ)
  (hf : distinct_real_roots f) :
  ∃ r s : ℝ, r ≠ s ∧ f 1 = 3/8 :=
sorry

end smallest_f1_value_l816_81642


namespace zero_points_of_gx_l816_81635

noncomputable def fx (a x : ℝ) : ℝ := (1 / 2) * x^2 - abs (x - 2 * a)
noncomputable def gx (a x : ℝ) : ℝ := 4 * a * x^2 + 2 * x + 1

theorem zero_points_of_gx (a : ℝ) (h : -1 / 4 ≤ a ∧ a ≤ 1 / 4) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (∃ x1 x2, gx a x1 = 0 ∧ gx a x2 = 0) := 
sorry

end zero_points_of_gx_l816_81635


namespace cost_price_per_meter_l816_81657

-- Definitions for conditions
def total_length : ℝ := 9.25
def total_cost : ℝ := 416.25

-- The theorem to be proved
theorem cost_price_per_meter : total_cost / total_length = 45 := by
  sorry

end cost_price_per_meter_l816_81657


namespace radius_of_larger_circle_l816_81638

theorem radius_of_larger_circle (r : ℝ) (r_pos : r > 0)
    (ratio_condition : ∀ (rs : ℝ), rs = 3 * r)
    (diameter_condition : ∀ (ac : ℝ), ac = 6 * r)
    (chord_tangent_condition : ∀ (ab : ℝ), ab = 12) :
     (radius : ℝ) = 3 * r :=
by
  sorry

end radius_of_larger_circle_l816_81638


namespace min_weighings_to_find_heaviest_l816_81679

-- Given conditions
variable (n : ℕ) (hn : n > 2)
variables (coins : Fin n) -- Representing coins with distinct masses
variables (scales : Fin n) -- Representing n scales where one is faulty

-- Theorem statement: Minimum number of weighings to find the heaviest coin
theorem min_weighings_to_find_heaviest : ∃ m, m = 2 * n - 1 := 
by
  existsi (2 * n - 1)
  rfl

end min_weighings_to_find_heaviest_l816_81679
