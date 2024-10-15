import Mathlib

namespace NUMINAMATH_GPT_determine_m_n_l1287_128729

theorem determine_m_n 
  {a b c d m n : ℕ} 
  (h₁ : a + b + c + d = m^2)
  (h₂ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₃ : max (max a b) (max c d) = n^2) 
  : m = 9 ∧ n = 6 := by 
  sorry

end NUMINAMATH_GPT_determine_m_n_l1287_128729


namespace NUMINAMATH_GPT_total_balloons_l1287_128701

theorem total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) 
  (h1 : fred_balloons = 10) 
  (h2 : sam_balloons = 46) 
  (h3 : dan_balloons = 16) 
  (total : fred_balloons + sam_balloons + dan_balloons = 72) :
  fred_balloons + sam_balloons + dan_balloons = 72 := 
sorry

end NUMINAMATH_GPT_total_balloons_l1287_128701


namespace NUMINAMATH_GPT_circumcircle_of_right_triangle_l1287_128753

theorem circumcircle_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  ∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_of_right_triangle_l1287_128753


namespace NUMINAMATH_GPT_train_speed_l1287_128731

/-- Proof that calculates the speed of a train given the times to pass a man and a platform,
and the length of the platform, and shows it equals 54.00432 km/hr. -/
theorem train_speed (L V : ℝ) 
  (platform_length : ℝ := 360.0288)
  (time_to_pass_man : ℝ := 20)
  (time_to_pass_platform : ℝ := 44)
  (equation1 : L = V * time_to_pass_man)
  (equation2 : L + platform_length = V * time_to_pass_platform) :
  V = 15.0012 → V * 3.6 = 54.00432 :=
by sorry

end NUMINAMATH_GPT_train_speed_l1287_128731


namespace NUMINAMATH_GPT_arithmetic_geometric_ratio_l1287_128776

theorem arithmetic_geometric_ratio
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n+1) = a n + d)
  (h_geometric : (a 3)^2 = a 1 * a 9)
  (h_nonzero_d : d ≠ 0) :
  a 11 / a 5 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_ratio_l1287_128776


namespace NUMINAMATH_GPT_range_of_b_l1287_128709

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b) (h2 : a + b < 1) (h3 : 2 ≤ a - b) (h4 : a - b < 3) :
  -3 / 2 < b ∧ b < -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1287_128709


namespace NUMINAMATH_GPT_remainder_M_divided_by_1000_l1287_128725

/-- Define flag problem parameters -/
def flagpoles: ℕ := 2
def blue_flags: ℕ := 15
def green_flags: ℕ := 10

/-- Condition: Two flagpoles, 15 blue flags and 10 green flags -/
def arrangable_flags (flagpoles blue_flags green_flags: ℕ) : Prop :=
  blue_flags + green_flags = 25 ∧ flagpoles = 2

/-- Condition: Each pole contains at least one flag -/
def each_pole_has_flag (arranged_flags: ℕ) : Prop :=
  arranged_flags > 0

/-- Condition: No two green flags are adjacent in any arrangement -/
def no_adjacent_green_flags (arranged_greens: ℕ) : Prop :=
  arranged_greens > 0

/-- Main theorem statement with correct answer -/
theorem remainder_M_divided_by_1000 (M: ℕ) : 
  arrangable_flags flagpoles blue_flags green_flags ∧ 
  each_pole_has_flag M ∧ 
  no_adjacent_green_flags green_flags ∧ 
  M % 1000 = 122
:= sorry

end NUMINAMATH_GPT_remainder_M_divided_by_1000_l1287_128725


namespace NUMINAMATH_GPT_correct_scientific_notation_l1287_128795

def scientific_notation (n : ℝ) : ℝ × ℝ := 
  (4, 5)

theorem correct_scientific_notation : scientific_notation 400000 = (4, 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_scientific_notation_l1287_128795


namespace NUMINAMATH_GPT_mohamed_donated_more_l1287_128754

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end NUMINAMATH_GPT_mohamed_donated_more_l1287_128754


namespace NUMINAMATH_GPT_bryce_raisins_l1287_128749

theorem bryce_raisins (x : ℕ) (h1 : x = 2 * (x - 8)) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_bryce_raisins_l1287_128749


namespace NUMINAMATH_GPT_num_square_tiles_is_zero_l1287_128704

def triangular_tiles : ℕ := sorry
def square_tiles : ℕ := sorry
def hexagonal_tiles : ℕ := sorry

axiom tile_count_eq : triangular_tiles + square_tiles + hexagonal_tiles = 30
axiom edge_count_eq : 3 * triangular_tiles + 4 * square_tiles + 6 * hexagonal_tiles = 120

theorem num_square_tiles_is_zero : square_tiles = 0 :=
by
  sorry

end NUMINAMATH_GPT_num_square_tiles_is_zero_l1287_128704


namespace NUMINAMATH_GPT_S9_value_l1287_128755

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a_n (n + 1) - a_n n) = (a_n 1 - a_n 0)

-- Sum of the first n terms of arithmetic sequence
def sum_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = n * (a_n 0 + a_n (n - 1)) / 2

-- Given conditions: 
axiom a4_plus_a6 : a_n 4 + a_n 6 = 12
axiom S_definition : sum_first_n_terms S_n a_n

theorem S9_value : S_n 9 = 54 :=
by
  -- assuming the given conditions and definitions, we aim to prove the desired theorem.
  sorry

end NUMINAMATH_GPT_S9_value_l1287_128755


namespace NUMINAMATH_GPT_fare_per_1_5_mile_l1287_128737

-- Definitions and conditions
def fare_first : ℝ := 1.0
def total_fare : ℝ := 7.3
def increments_per_mile : ℝ := 5.0
def total_miles : ℝ := 3.0
def remaining_increments : ℝ := (total_miles * increments_per_mile) - 1
def remaining_fare : ℝ := total_fare - fare_first

-- Theorem to prove
theorem fare_per_1_5_mile : remaining_fare / remaining_increments = 0.45 :=
by
  sorry

end NUMINAMATH_GPT_fare_per_1_5_mile_l1287_128737


namespace NUMINAMATH_GPT_tan_sum_l1287_128766

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 96 / 65)
  (h2 : Real.cos x + Real.cos y = 72 / 65) :
  Real.tan x + Real.tan y = 507 / 112 := 
sorry

end NUMINAMATH_GPT_tan_sum_l1287_128766


namespace NUMINAMATH_GPT_average_age_of_women_l1287_128726

theorem average_age_of_women (A : ℕ) :
  (6 * (A + 2) = 6 * A - 22 + W) → (W / 2 = 17) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_average_age_of_women_l1287_128726


namespace NUMINAMATH_GPT_trajectory_of_point_l1287_128760

theorem trajectory_of_point (x y : ℝ) (P A : ℝ × ℝ × ℝ) (hP : P = (x, y, 0)) (hA : A = (0, 0, 4)) (hPA : dist P A = 5) : 
  x^2 + y^2 = 9 :=
by sorry

end NUMINAMATH_GPT_trajectory_of_point_l1287_128760


namespace NUMINAMATH_GPT_probability_at_least_one_unqualified_l1287_128743

theorem probability_at_least_one_unqualified :
  let total_products := 6
  let qualified_products := 4
  let unqualified_products := 2
  let products_selected := 2
  (1 - (Nat.choose qualified_products 2 / Nat.choose total_products 2)) = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_unqualified_l1287_128743


namespace NUMINAMATH_GPT_Jim_runs_total_distance_l1287_128702

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end NUMINAMATH_GPT_Jim_runs_total_distance_l1287_128702


namespace NUMINAMATH_GPT_correct_subtraction_result_l1287_128762

theorem correct_subtraction_result (n : ℕ) (h : 40 / n = 5) : 20 - n = 12 := by
sorry

end NUMINAMATH_GPT_correct_subtraction_result_l1287_128762


namespace NUMINAMATH_GPT_hexagon_perimeter_arithmetic_sequence_l1287_128774

theorem hexagon_perimeter_arithmetic_sequence :
  let a₁ := 10
  let a₂ := 12
  let a₃ := 14
  let a₄ := 16
  let a₅ := 18
  let a₆ := 20
  let lengths := [a₁, a₂, a₃, a₄, a₅, a₆]
  let perimeter := lengths.sum
  perimeter = 90 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_arithmetic_sequence_l1287_128774


namespace NUMINAMATH_GPT_arccos_one_half_l1287_128748

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_half_l1287_128748


namespace NUMINAMATH_GPT_min_value_inequality_l1287_128790

theorem min_value_inequality (a b c d e f : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f)
    (h_sum : a + b + c + d + e + f = 9) : 
    1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f ≥ 676 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_inequality_l1287_128790


namespace NUMINAMATH_GPT_simplify_expression_l1287_128783

variable {x y : ℝ}
variable (h : x * y ≠ 0)

theorem simplify_expression (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^2 + 1) / y) - ((x^2 - 1) / y) * ((y^3 - 1) / x) =
  (x^3*y^2 - x^2*y^3 + x^3 + x^2 + y^2 + y^3) / (x*y) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1287_128783


namespace NUMINAMATH_GPT_distinct_real_roots_sum_l1287_128750

theorem distinct_real_roots_sum (p r_1 r_2 : ℝ) (h_eq : ∀ x, x^2 + p * x + 18 = 0)
  (h_distinct : r_1 ≠ r_2) (h_root1 : x^2 + p * x + 18 = 0)
  (h_root2 : x^2 + p * x + 18 = 0) : |r_1 + r_2| > 6 :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_sum_l1287_128750


namespace NUMINAMATH_GPT_Sarah_correct_responses_l1287_128714

theorem Sarah_correct_responses : ∃ x : ℕ, x ≥ 22 ∧ (7 * x - (26 - x) + 4 ≥ 150) :=
by
  sorry

end NUMINAMATH_GPT_Sarah_correct_responses_l1287_128714


namespace NUMINAMATH_GPT_DF_is_5_point_5_l1287_128792

variables {A B C D E F : Type}
variables (congruent : triangle A B C ≃ triangle D E F)
variables (ac_length : AC = 5.5)

theorem DF_is_5_point_5 : DF = 5.5 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_DF_is_5_point_5_l1287_128792


namespace NUMINAMATH_GPT_count_not_divisible_by_5_or_7_l1287_128739

theorem count_not_divisible_by_5_or_7 :
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  total_numbers - count_divisible_by_5_or_7 = 343 :=
by
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  have h : total_numbers - count_divisible_by_5_or_7 = 343 := by sorry
  exact h

end NUMINAMATH_GPT_count_not_divisible_by_5_or_7_l1287_128739


namespace NUMINAMATH_GPT_sec_150_eq_neg_two_div_sqrt_three_l1287_128717

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_sec_150_eq_neg_two_div_sqrt_three_l1287_128717


namespace NUMINAMATH_GPT_ticket_distribution_l1287_128782

theorem ticket_distribution 
    (A Ad C Cd S : ℕ) 
    (h1 : 25 * A + 20 * 50 + 15 * C + 10 * 30 + 20 * S = 7200) 
    (h2 : A + 50 + C + 30 + S = 400)
    (h3 : A + 50 = 2 * S)
    (h4 : Ad = 50)
    (h5 : Cd = 30) : 
    A = 102 ∧ Ad = 50 ∧ C = 142 ∧ Cd = 30 ∧ S = 76 := 
by 
    sorry

end NUMINAMATH_GPT_ticket_distribution_l1287_128782


namespace NUMINAMATH_GPT_total_votes_is_5000_l1287_128788

theorem total_votes_is_5000 :
  ∃ (V : ℝ), 0.45 * V - 0.35 * V = 500 ∧ 0.35 * V - 0.20 * V = 350 ∧ V = 5000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_is_5000_l1287_128788


namespace NUMINAMATH_GPT_angle_between_NE_and_SW_l1287_128771

theorem angle_between_NE_and_SW
  (n : ℕ) (hn : n = 12)
  (total_degrees : ℚ) (htotal : total_degrees = 360)
  (spaced_rays : ℚ) (hspaced : spaced_rays = total_degrees / n)
  (angles_between_NE_SW : ℕ) (hangles : angles_between_NE_SW = 4) :
  (angles_between_NE_SW * spaced_rays = 120) :=
by
  rw [htotal, hn] at hspaced
  rw [hangles]
  rw [hspaced]
  sorry

end NUMINAMATH_GPT_angle_between_NE_and_SW_l1287_128771


namespace NUMINAMATH_GPT_diameter_correct_l1287_128756

noncomputable def diameter_of_circle (C : ℝ) (hC : C = 36) : ℝ :=
  let r := C / (2 * Real.pi)
  2 * r

theorem diameter_correct (C : ℝ) (hC : C = 36) : diameter_of_circle C hC = 36 / Real.pi := by
  sorry

end NUMINAMATH_GPT_diameter_correct_l1287_128756


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l1287_128738

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l1287_128738


namespace NUMINAMATH_GPT_chord_length_l1287_128761

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1287_128761


namespace NUMINAMATH_GPT_vasya_days_l1287_128791

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end NUMINAMATH_GPT_vasya_days_l1287_128791


namespace NUMINAMATH_GPT_product_identity_l1287_128780

theorem product_identity (x y : ℝ) : (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end NUMINAMATH_GPT_product_identity_l1287_128780


namespace NUMINAMATH_GPT_max_distance_difference_l1287_128705

-- Given definitions and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 15 = 1
def circle1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Main theorem to prove the maximum value of |PM| - |PN|
theorem max_distance_difference (P M N : ℝ × ℝ) :
  hyperbola P.1 P.2 →
  circle1 M.1 M.2 →
  circle2 N.1 N.2 →
  ∃ max_val : ℝ, max_val = 5 :=
by
  -- Proof skipped, only statement is required
  sorry

end NUMINAMATH_GPT_max_distance_difference_l1287_128705


namespace NUMINAMATH_GPT_inequality_holds_l1287_128796

theorem inequality_holds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a + (1 / b))^2 + (b + (1 / c))^2 + (c + (1 / a))^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1287_128796


namespace NUMINAMATH_GPT_quadruple_pieces_sold_l1287_128797

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_quadruple_pieces_sold_l1287_128797


namespace NUMINAMATH_GPT_intersect_complementB_l1287_128794

def setA (x : ℝ) : Prop := ∃ y : ℝ, y = Real.log (9 - x^2)

def setB (x : ℝ) : Prop := ∃ y : ℝ, y = Real.sqrt (4 * x - x^2)

def complementB (x : ℝ) : Prop := x < 0 ∨ 4 < x

theorem intersect_complementB :
  { x : ℝ | setA x } ∩ { x : ℝ | complementB x } = { x : ℝ | -3 < x ∧ x < 0 } :=
sorry

end NUMINAMATH_GPT_intersect_complementB_l1287_128794


namespace NUMINAMATH_GPT_polynomial_coeff_sum_abs_l1287_128779

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℤ) 
  (h : (2*x - 1)^5 + (x + 2)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  |a| + |a_2| + |a_4| = 30 :=
sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_abs_l1287_128779


namespace NUMINAMATH_GPT_F_minimum_value_neg_inf_to_0_l1287_128781

variable (f g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) := ∀ x, h (-x) = - (h x)

theorem F_minimum_value_neg_inf_to_0 
  (hf_odd : is_odd f) 
  (hg_odd : is_odd g)
  (hF_max : ∀ x > 0, f x + g x + 2 ≤ 8) 
  (hF_reaches_max : ∃ x > 0, f x + g x + 2 = 8) :
  ∀ x < 0, f x + g x + 2 ≥ -4 :=
by
  sorry

end NUMINAMATH_GPT_F_minimum_value_neg_inf_to_0_l1287_128781


namespace NUMINAMATH_GPT_abs_diff_squares_1055_985_eq_1428_l1287_128799

theorem abs_diff_squares_1055_985_eq_1428 :
  abs ((105.5: ℝ)^2 - (98.5: ℝ)^2) = 1428 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_1055_985_eq_1428_l1287_128799


namespace NUMINAMATH_GPT_T_n_correct_l1287_128712

def a_n (n : ℕ) : ℤ := 2 * n - 5

def b_n (n : ℕ) : ℤ := 2^n

def C_n (n : ℕ) : ℤ := |a_n n| * b_n n

def T_n : ℕ → ℤ
| 1     => 6
| 2     => 10
| n     => if n >= 3 then 34 + (2 * n - 7) * 2^(n + 1) else 0  -- safeguard for invalid n

theorem T_n_correct (n : ℕ) (hyp : n ≥ 1) : 
  T_n n = 
  if n = 1 then 6 
  else if n = 2 then 10 
  else if n ≥ 3 then 34 + (2 * n - 7) * 2^(n + 1) 
  else 0 := 
by 
sorry

end NUMINAMATH_GPT_T_n_correct_l1287_128712


namespace NUMINAMATH_GPT_adelaide_ducks_l1287_128768

variable (A E K : ℕ)

theorem adelaide_ducks (h1 : A = 2 * E) (h2 : E = K - 45) (h3 : (A + E + K) / 3 = 35) :
  A = 30 := by
  sorry

end NUMINAMATH_GPT_adelaide_ducks_l1287_128768


namespace NUMINAMATH_GPT_good_horse_catchup_l1287_128720

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end NUMINAMATH_GPT_good_horse_catchup_l1287_128720


namespace NUMINAMATH_GPT_find_two_digit_number_l1287_128716

theorem find_two_digit_number : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 10^6 ≤ n^3 ∧ n^3 < 10^7 ∧ 101010 * x + 1 = n^3 ∧ x = 93) := 
 by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1287_128716


namespace NUMINAMATH_GPT_picture_edge_distance_l1287_128736

theorem picture_edge_distance 
    (wall_width : ℕ) 
    (picture_width : ℕ) 
    (centered : Bool) 
    (h_w : wall_width = 22) 
    (h_p : picture_width = 4) 
    (h_c : centered = true) : 
    ∃ (distance : ℕ), distance = 9 := 
by
  sorry

end NUMINAMATH_GPT_picture_edge_distance_l1287_128736


namespace NUMINAMATH_GPT_same_number_of_acquaintances_l1287_128772

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) (acquaintances : Fin n → Fin n) :
  ∃ i j : Fin n, i ≠ j ∧ acquaintances i = acquaintances j :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_same_number_of_acquaintances_l1287_128772


namespace NUMINAMATH_GPT_victoria_should_return_22_l1287_128758

theorem victoria_should_return_22 :
  let initial_money := 50
  let pizza_cost_per_box := 12
  let pizzas_bought := 2
  let juice_cost_per_pack := 2
  let juices_bought := 2
  let total_spent := (pizza_cost_per_box * pizzas_bought) + (juice_cost_per_pack * juices_bought)
  let money_returned := initial_money - total_spent
  money_returned = 22 :=
by
  sorry

end NUMINAMATH_GPT_victoria_should_return_22_l1287_128758


namespace NUMINAMATH_GPT_minimize_S_n_l1287_128775

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

axiom arithmetic_sequence : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
axiom sum_first_n_terms : ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)
axiom condition1 : a 0 + a 4 = -14
axiom condition2 : S 9 = -27

theorem minimize_S_n : ∃ n, ∀ m, S n ≤ S m := sorry

end NUMINAMATH_GPT_minimize_S_n_l1287_128775


namespace NUMINAMATH_GPT_aquarium_water_ratio_l1287_128703

theorem aquarium_water_ratio :
  let length := 4
  let width := 6
  let height := 3
  let volume := length * width * height
  let halfway_volume := volume / 2
  let water_after_cat := halfway_volume / 2
  let final_water := 54
  (final_water / water_after_cat) = 3 := by
  sorry

end NUMINAMATH_GPT_aquarium_water_ratio_l1287_128703


namespace NUMINAMATH_GPT_small_seat_capacity_indeterminate_l1287_128721

-- Conditions
def small_seats : ℕ := 3
def large_seats : ℕ := 7
def capacity_per_large_seat : ℕ := 12
def total_large_capacity : ℕ := 84

theorem small_seat_capacity_indeterminate
  (h1 : large_seats * capacity_per_large_seat = total_large_capacity)
  (h2 : ∀ s : ℕ, ∃ p : ℕ, p ≠ s * capacity_per_large_seat) :
  ¬ ∃ n : ℕ, ∀ m : ℕ, small_seats * m = n * small_seats :=
by {
  sorry
}

end NUMINAMATH_GPT_small_seat_capacity_indeterminate_l1287_128721


namespace NUMINAMATH_GPT_actual_plot_area_in_acres_l1287_128789

-- Define the conditions
def base1_cm := 18
def base2_cm := 12
def height_cm := 8
def scale_cm_to_miles := 5
def sq_mile_to_acres := 640

-- Prove the question which is to find the actual plot area in acres
theorem actual_plot_area_in_acres : 
  (1/2 * (base1_cm + base2_cm) * height_cm * (scale_cm_to_miles ^ 2) * sq_mile_to_acres) = 1920000 :=
by
  sorry

end NUMINAMATH_GPT_actual_plot_area_in_acres_l1287_128789


namespace NUMINAMATH_GPT_solve_for_y_l1287_128798

noncomputable def g (y : ℝ) : ℝ := (30 * y + (30 * y + 27)^(1/3))^(1/3)

theorem solve_for_y :
  (∃ y : ℝ, g y = 15) ↔ (∃ y : ℝ, y = 1674 / 15) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1287_128798


namespace NUMINAMATH_GPT_soccer_team_percentage_l1287_128732

theorem soccer_team_percentage (total_games won_games : ℕ) (h1 : total_games = 140) (h2 : won_games = 70) :
  (won_games / total_games : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_soccer_team_percentage_l1287_128732


namespace NUMINAMATH_GPT_value_range_of_func_l1287_128711

-- Define the function y = x^2 - 4x + 6 for x in the interval [1, 4]
def func (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem value_range_of_func : 
  ∀ y, ∃ x, (1 ≤ x ∧ x ≤ 4) ∧ y = func x ↔ 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_func_l1287_128711


namespace NUMINAMATH_GPT_sequence_26th_term_l1287_128742

theorem sequence_26th_term (a d : ℕ) (n : ℕ) (h_a : a = 4) (h_d : d = 3) (h_n : n = 26) :
  a + (n - 1) * d = 79 :=
by
  sorry

end NUMINAMATH_GPT_sequence_26th_term_l1287_128742


namespace NUMINAMATH_GPT_work_completion_days_l1287_128785

theorem work_completion_days
  (E_q : ℝ) -- Efficiency of q
  (E_p : ℝ) -- Efficiency of p
  (E_r : ℝ) -- Efficiency of r
  (W : ℝ)  -- Total work
  (H1 : E_p = 1.5 * E_q) -- Condition 1
  (H2 : W = E_p * 25) -- Condition 2
  (H3 : E_r = 0.8 * E_q) -- Condition 3
  : (W / (E_p + E_q + E_r)) = 11.36 := -- Prove the days_needed is 11.36
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l1287_128785


namespace NUMINAMATH_GPT_distance_M_to_AB_l1287_128787

noncomputable def distance_to_ab : ℝ := 5.8

theorem distance_M_to_AB
  (M : Point)
  (A B C : Point)
  (d_AC d_BC : ℝ)
  (AB BC AC : ℝ)
  (H1 : d_AC = 2)
  (H2 : d_BC = 4)
  (H3 : AB = 10)
  (H4 : BC = 17)
  (H5 : AC = 21) :
  distance_to_ab = 5.8 :=
by
  sorry

end NUMINAMATH_GPT_distance_M_to_AB_l1287_128787


namespace NUMINAMATH_GPT_evaluate_six_applications_problem_solution_l1287_128765

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

theorem evaluate_six_applications (θ : ℚ) : 
  r (r (r (r (r (r θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

theorem problem_solution : r (r (r (r (r (r 30))))) = 158 / 253 :=
by
  have h : r (r (r (r (r (r 30))))) = (8 + 5 * 30) / (13 + 8 * 30) := by
    exact evaluate_six_applications 30
  rw [h]
  norm_num

end NUMINAMATH_GPT_evaluate_six_applications_problem_solution_l1287_128765


namespace NUMINAMATH_GPT_total_books_l1287_128719

def school_books : ℕ := 19
def sports_books : ℕ := 39

theorem total_books : school_books + sports_books = 58 := by
  sorry

end NUMINAMATH_GPT_total_books_l1287_128719


namespace NUMINAMATH_GPT_maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l1287_128733

open Real

theorem maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism 
  (a b : ℝ)
  (ha : a^2 + b^2 = 25) 
  (AC_eq_5 : AC = 5) :
  ∃ (r : ℝ), 4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end NUMINAMATH_GPT_maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l1287_128733


namespace NUMINAMATH_GPT_candy_cost_l1287_128777

theorem candy_cost (C : ℝ) 
  (h1 : 20 + 40 = 60) 
  (h2 : 5 * 40 + 20 * C = 60 * 6) : 
  C = 8 :=
by
  sorry

end NUMINAMATH_GPT_candy_cost_l1287_128777


namespace NUMINAMATH_GPT_daughter_age_l1287_128722

-- Define the conditions and the question as a theorem
theorem daughter_age (D F : ℕ) (h1 : F = 3 * D) (h2 : F + 12 = 2 * (D + 12)) : D = 12 :=
by
  -- We need to provide a proof or placeholder for now
  sorry

end NUMINAMATH_GPT_daughter_age_l1287_128722


namespace NUMINAMATH_GPT_correct_expression_l1287_128793

theorem correct_expression (a : ℝ) :
  (a^3 * a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ ¬((a - 1)^2 = a^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_l1287_128793


namespace NUMINAMATH_GPT_divisibility_of_product_l1287_128723

def three_consecutive_integers (a1 a2 a3 : ℤ) : Prop :=
  a1 = a2 - 1 ∧ a3 = a2 + 1

theorem divisibility_of_product (a1 a2 a3 : ℤ) (h : three_consecutive_integers a1 a2 a3) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by
  cases h with
  | intro ha1 ha3 =>
    sorry

end NUMINAMATH_GPT_divisibility_of_product_l1287_128723


namespace NUMINAMATH_GPT_find_x_values_l1287_128728

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem find_x_values (x : ℝ) :
  (f (f x) = f x) ↔ (x = 0 ∨ x = 2 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1287_128728


namespace NUMINAMATH_GPT_passed_both_tests_l1287_128707

theorem passed_both_tests :
  ∀ (total_students passed_long_jump passed_shot_put failed_both passed_both: ℕ),
  total_students = 50 →
  passed_long_jump = 40 →
  passed_shot_put = 31 →
  failed_both = 4 →
  passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both = total_students →
  passed_both = 25 :=
by
  intros total_students passed_long_jump passed_shot_put failed_both passed_both h1 h2 h3 h4 h5
  -- proof can be skipped using sorry
  sorry

end NUMINAMATH_GPT_passed_both_tests_l1287_128707


namespace NUMINAMATH_GPT_positive_integer_solutions_count_l1287_128746

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end NUMINAMATH_GPT_positive_integer_solutions_count_l1287_128746


namespace NUMINAMATH_GPT_passengers_landed_in_newberg_last_year_l1287_128724

theorem passengers_landed_in_newberg_last_year :
  let airport_a_on_time : ℕ := 16507
  let airport_a_late : ℕ := 256
  let airport_b_on_time : ℕ := 11792
  let airport_b_late : ℕ := 135
  airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690 :=
by
  let airport_a_on_time := 16507
  let airport_a_late := 256
  let airport_b_on_time := 11792
  let airport_b_late := 135
  show airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690
  sorry

end NUMINAMATH_GPT_passengers_landed_in_newberg_last_year_l1287_128724


namespace NUMINAMATH_GPT_sqrt10_solution_l1287_128773

theorem sqrt10_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : (1/a) + (1/b) = 2) :
  m = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_sqrt10_solution_l1287_128773


namespace NUMINAMATH_GPT_highway_total_vehicles_l1287_128764

theorem highway_total_vehicles (num_trucks : ℕ) (num_cars : ℕ) (total_vehicles : ℕ)
  (h1 : num_trucks = 100)
  (h2 : num_cars = 2 * num_trucks)
  (h3 : total_vehicles = num_cars + num_trucks) :
  total_vehicles = 300 :=
by
  sorry

end NUMINAMATH_GPT_highway_total_vehicles_l1287_128764


namespace NUMINAMATH_GPT_triangle_side_lengths_count_l1287_128713

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_count_l1287_128713


namespace NUMINAMATH_GPT_second_student_marks_l1287_128745

theorem second_student_marks (x y : ℝ) 
  (h1 : x = y + 9) 
  (h2 : x = 0.56 * (x + y)) : 
  y = 33 := 
sorry

end NUMINAMATH_GPT_second_student_marks_l1287_128745


namespace NUMINAMATH_GPT_geometric_sequence_a_equals_minus_four_l1287_128727

theorem geometric_sequence_a_equals_minus_four (a : ℝ) 
(h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : a = -4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a_equals_minus_four_l1287_128727


namespace NUMINAMATH_GPT_max_planes_15_points_l1287_128741

-- Define the total number of points
def total_points : ℕ := 15

-- Define the number of collinear points
def collinear_points : ℕ := 5

-- Compute the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of planes formed by any 3 out of 15 points
def total_planes : ℕ := binom total_points 3

-- Number of degenerate planes formed by the collinear points
def degenerate_planes : ℕ := binom collinear_points 3

-- Maximum number of unique planes
def max_unique_planes : ℕ := total_planes - degenerate_planes

-- Lean theorem statement
theorem max_planes_15_points : max_unique_planes = 445 :=
by
  sorry

end NUMINAMATH_GPT_max_planes_15_points_l1287_128741


namespace NUMINAMATH_GPT_encryption_of_hope_is_correct_l1287_128767

def shift_letter (c : Char) : Char :=
  if 'a' ≤ c ∧ c ≤ 'z' then
    Char.ofNat ((c.toNat - 'a'.toNat + 4) % 26 + 'a'.toNat)
  else 
    c

def encrypt (s : String) : String :=
  s.map shift_letter

theorem encryption_of_hope_is_correct : encrypt "hope" = "lsti" :=
by
  sorry

end NUMINAMATH_GPT_encryption_of_hope_is_correct_l1287_128767


namespace NUMINAMATH_GPT_broccoli_sales_l1287_128710

theorem broccoli_sales (B C S Ca : ℝ) (h1 : C = 2 * B) (h2 : S = B / 2 + 16) (h3 : Ca = 136) (total_sales : B + C + S + Ca = 380) :
  B = 57 :=
by
  sorry

end NUMINAMATH_GPT_broccoli_sales_l1287_128710


namespace NUMINAMATH_GPT_range_of_f_l1287_128734

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc (1 : ℝ) (Real.sqrt 2) := 
by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc 1 (Real.sqrt 2) := sorry
  exact this x hx

end NUMINAMATH_GPT_range_of_f_l1287_128734


namespace NUMINAMATH_GPT_binom_divisibility_l1287_128706

theorem binom_divisibility (k n : ℕ) (p : ℕ) (h1 : k > 1) (h2 : n > 1) 
  (h3 : p = 2 * k - 1) (h4 : Nat.Prime p) (h5 : p ∣ (Nat.choose n 2 - Nat.choose k 2)) : 
  p^2 ∣ (Nat.choose n 2 - Nat.choose k 2) := 
sorry

end NUMINAMATH_GPT_binom_divisibility_l1287_128706


namespace NUMINAMATH_GPT_scooter_safety_gear_price_increase_l1287_128786

theorem scooter_safety_gear_price_increase :
  let last_year_scooter_price := 200
  let last_year_gear_price := 50
  let scooter_increase_rate := 0.08
  let gear_increase_rate := 0.15
  let total_last_year_price := last_year_scooter_price + last_year_gear_price
  let this_year_scooter_price := last_year_scooter_price * (1 + scooter_increase_rate)
  let this_year_gear_price := last_year_gear_price * (1 + gear_increase_rate)
  let total_this_year_price := this_year_scooter_price + this_year_gear_price
  let total_increase := total_this_year_price - total_last_year_price
  let percent_increase := (total_increase / total_last_year_price) * 100
  percent_increase = 9 :=
by
  -- sorry is added here to skip the proof steps
  sorry

end NUMINAMATH_GPT_scooter_safety_gear_price_increase_l1287_128786


namespace NUMINAMATH_GPT_find_a2023_l1287_128715

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end NUMINAMATH_GPT_find_a2023_l1287_128715


namespace NUMINAMATH_GPT_geometric_series_first_term_l1287_128778

theorem geometric_series_first_term (a : ℕ) (r : ℚ) (S : ℕ) (h_r : r = 1 / 4) (h_S : S = 40) (h_sum : S = a / (1 - r)) : a = 30 := sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1287_128778


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1287_128784

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1287_128784


namespace NUMINAMATH_GPT_KimFridayToMondayRatio_l1287_128763

variable (MondaySweaters : ℕ) (TuesdaySweaters : ℕ) (WednesdaySweaters : ℕ) (ThursdaySweaters : ℕ) (FridaySweaters : ℕ)

def KimSweaterKnittingConditions (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ) : Prop :=
  MondaySweaters = 8 ∧
  TuesdaySweaters = MondaySweaters + 2 ∧
  WednesdaySweaters = TuesdaySweaters - 4 ∧
  ThursdaySweaters = TuesdaySweaters - 4 ∧
  MondaySweaters + TuesdaySweaters + WednesdaySweaters + ThursdaySweaters + FridaySweaters = 34

theorem KimFridayToMondayRatio 
  (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ)
  (h : KimSweaterKnittingConditions MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters) :
  FridaySweaters / MondaySweaters = 1/2 :=
  sorry

end NUMINAMATH_GPT_KimFridayToMondayRatio_l1287_128763


namespace NUMINAMATH_GPT_find_x_plus_y_l1287_128718

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005) (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2009 + Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1287_128718


namespace NUMINAMATH_GPT_find_solution_l1287_128770

def satisfies_conditions (x y z : ℝ) :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solution (x y z : ℝ) :
  satisfies_conditions x y z →
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_solution_l1287_128770


namespace NUMINAMATH_GPT_geom_seq_sum_4n_l1287_128769

-- Assume we have a geometric sequence with positive terms and common ratio q
variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)

-- The sum of the first n terms of the geometric sequence is S_n
noncomputable def S_n : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Given conditions
axiom h1 : S_n a q n = 2
axiom h2 : S_n a q (3 * n) = 14

-- We need to prove that S_{4n} = 30
theorem geom_seq_sum_4n : S_n a q (4 * n) = 30 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_4n_l1287_128769


namespace NUMINAMATH_GPT_quadratic_complete_square_l1287_128730

theorem quadratic_complete_square (c r s k : ℝ) (h1 : 8 * k^2 - 6 * k + 16 = c * (k + r)^2 + s) 
  (h2 : c = 8) 
  (h3 : r = -3 / 8) 
  (h4 : s = 119 / 8) : 
  s / r = -119 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l1287_128730


namespace NUMINAMATH_GPT_ratio_of_money_with_Ram_and_Gopal_l1287_128751

noncomputable section

variable (R K G : ℕ)

theorem ratio_of_money_with_Ram_and_Gopal 
  (hR : R = 735) 
  (hK : K = 4335) 
  (hRatio : G * 17 = 7 * K) 
  (hGCD : Nat.gcd 735 1785 = 105) :
  R * 17 = 7 * G := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_money_with_Ram_and_Gopal_l1287_128751


namespace NUMINAMATH_GPT_evaluate_expression_at_values_l1287_128744

theorem evaluate_expression_at_values (x y : ℤ) (h₁ : x = 1) (h₂ : y = -2) :
  (-2 * x ^ 2 + 2 * x - y) = 2 :=
by
  subst h₁
  subst h₂
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_values_l1287_128744


namespace NUMINAMATH_GPT_units_digit_of_17_pow_549_l1287_128752

theorem units_digit_of_17_pow_549 : (17 ^ 549) % 10 = 7 :=
by {
  -- Provide the necessary steps or strategies to prove the theorem
  sorry
}

end NUMINAMATH_GPT_units_digit_of_17_pow_549_l1287_128752


namespace NUMINAMATH_GPT_number_of_toys_sold_l1287_128700

theorem number_of_toys_sold (total_selling_price gain_per_toy cost_price_per_toy : ℕ)
  (h1 : total_selling_price = 25200)
  (h2 : gain_per_toy = 3 * cost_price_per_toy)
  (h3 : cost_price_per_toy = 1200) : 
  (total_selling_price - gain_per_toy) / cost_price_per_toy = 18 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_toys_sold_l1287_128700


namespace NUMINAMATH_GPT_complex_number_real_l1287_128759

theorem complex_number_real (m : ℝ) (z : ℂ) 
  (h1 : z = ⟨1 / (m + 5), 0⟩ + ⟨0, m^2 + 2 * m - 15⟩)
  (h2 : m^2 + 2 * m - 15 = 0)
  (h3 : m ≠ -5) :
  m = 3 :=
sorry

end NUMINAMATH_GPT_complex_number_real_l1287_128759


namespace NUMINAMATH_GPT_minimal_fencing_l1287_128757

theorem minimal_fencing (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l ≥ 400) : 
  2 * (w + l) = 60 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimal_fencing_l1287_128757


namespace NUMINAMATH_GPT_expenditure_of_neg_50_l1287_128735

/-- In the book "Nine Chapters on the Mathematical Art," it is noted that
"when two calculations have opposite meanings, they should be named positive
and negative." This means: if an income of $80 is denoted as $+80, then $-50
represents an expenditure of $50. -/
theorem expenditure_of_neg_50 :
  (∀ (income : ℤ), income = 80 → -income = -50 → ∃ (expenditure : ℤ), expenditure = 50) := sorry

end NUMINAMATH_GPT_expenditure_of_neg_50_l1287_128735


namespace NUMINAMATH_GPT_log_conversion_l1287_128708

theorem log_conversion (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : a = (2 * b) / 3 := 
sorry

end NUMINAMATH_GPT_log_conversion_l1287_128708


namespace NUMINAMATH_GPT_largest_number_formed_l1287_128740

-- Define the digits
def digit1 : ℕ := 2
def digit2 : ℕ := 6
def digit3 : ℕ := 9

-- Define the function to form the largest number using the given digits
def largest_three_digit_number (a b c : ℕ) : ℕ :=
  if a > b ∧ a > c then
    if b > c then 100 * a + 10 * b + c
    else 100 * a + 10 * c + b
  else if b > a ∧ b > c then
    if a > c then 100 * b + 10 * a + c
    else 100 * b + 10 * c + a
  else
    if a > b then 100 * c + 10 * a + b
    else 100 * c + 10 * b + a

-- Statement that this function correctly computes the largest number
theorem largest_number_formed :
  largest_three_digit_number digit1 digit2 digit3 = 962 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_formed_l1287_128740


namespace NUMINAMATH_GPT_Mildred_heavier_than_Carol_l1287_128747

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end NUMINAMATH_GPT_Mildred_heavier_than_Carol_l1287_128747
